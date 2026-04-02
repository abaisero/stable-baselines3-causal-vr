# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Confidence Estimates

Every single message MUST end with a confidence estimate in this format:

**Confidence:** [0–100%] — [one sentence explaining why]

No exceptions. This applies to all messages, including code, explanations, and answers to questions.

## Overview

This is a fork of [Stable Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3), a PyTorch-based RL library. The fork adds infrastructure for **causal RL**: augmenting policy gradient critics with causal knowledge for variance reduction. The upstream SB3 codebase is mature and stable; new development focuses on the causal extensions.

## Commands

```bash
make pytest          # Run tests (excludes @pytest.mark.expensive tests)
make mypy            # Type checking
make lint            # Ruff linting
make format          # Auto-format with ruff + black
make check-codestyle # Verify style without modifying
make commit-checks   # format + mypy + lint all-in-one
```

Run a single test file:
```bash
python -m pytest tests/test_foo.py -v
```

Code style: Black + Ruff, 127-char line length. Type hints required everywhere (mypy enforced).

## Architecture

### Upstream SB3 Structure

**Algorithm inheritance:**
- `PPO`, `A2C` → `OnPolicyAlgorithm` → `BaseAlgorithm`
- `DQN`, `DDPG`, `SAC`, `TD3` → `OffPolicyAlgorithm` → `BaseAlgorithm`

**Policy inheritance:**
- `MlpPolicy`, `CnnPolicy` → `ActorCriticPolicy` → `BasePolicy` → `nn.Module`

**On-policy training flow (PPO/A2C):**
1. `OnPolicyAlgorithm.collect_rollouts()` fills `RolloutBuffer` with `(obs, action, reward, value_pred, log_prob)`
2. `RolloutBuffer.compute_returns_and_advantage()` adds advantages and returns
3. `PPO.train()` / `A2C.train()` runs mini-batch SGD

**Critic network path:**
```
obs → FeaturesExtractor → MlpExtractor.value_net → latent_vf → Linear(latent_dim_vf, 1)
```
`ActorCriticPolicy.evaluate_actions(obs, actions)` (policies.py ~line 719) returns `(values, log_probs, entropy)`.

### Causal Extensions (this fork)

- `stable_baselines3/common/causal_wrappers.py` — Abstract `CausalEnvWrapper(gym.Wrapper)` base class. Defines `causal_graph` as a set of `(CausalNode, CausalNode)` edges over a single `(s, a, s')` transition. Provides `descendant_mask` / `non_descendant_mask` interface for identifying which observation features are causal descendants of actions.

- `stable_baselines3/common/envs/causal_half_cheetah.py` — Stub `CausalHalfCheetah` wrapper for `HalfCheetah-v4`. Contains detailed comments mapping obs indices (0–16) and action indices (0–5) to physical joints; `causal_graph` is a TODO.

**Planned additions** (see `claude-notes/causal_critic_roadmap.md`):
- `causal_layers.py` — `CausalCritic` / `CausalMlpExtractor` for structured value estimation
- `CausalActorCriticPolicy` — subclass of `ActorCriticPolicy` with causal-aware `evaluate_actions()`
- Possible `CausalPPO` algorithm only if training loop needs modification
- `CausalRolloutBuffer` if causal side-information needs to be stored per-step

## Reasoning and Uncertainty

Before answering a technical question, analyze the relevant trade-offs and uncertainties upfront. State your confidence level and the reasoning behind it clearly at the start. If you are unsure, say so explicitly rather than giving a confident answer and then reversing it in the next message.

Every message MUST end with a confidence estimate in this format:

**Confidence:** [0–100%] — [one sentence explaining why]

## Code Style: Tensor Shape Comments

Add a shape comment on the line after every tensor or array variable is introduced or modified. This includes:
- When a tensor/array is first created
- When a tensor/array is received as a function argument (comment at the top of the function body)
- After every operation that produces a new tensor/array

Use the variable's actual name and semantic dimension names, not numeric sizes. Example:

```python
def foo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x.shape == (n_actions, n_obs)
    # y.shape == (n_obs, n_obs)
    z = x @ y
    # z.shape == (n_actions, n_obs)
    return z
```

This applies to both numpy arrays and PyTorch tensors.

## Key Files for Causal Development

| File | Role |
|------|------|
| `stable_baselines3/common/causal_wrappers.py` | Causal graph interface |
| `stable_baselines3/common/envs/causal_half_cheetah.py` | Reference environment |
| `stable_baselines3/common/policies.py` | `ActorCriticPolicy`, `evaluate_actions` |
| `stable_baselines3/common/buffers.py` | `RolloutBuffer` |
| `stable_baselines3/common/on_policy_algorithm.py` | `collect_rollouts` |
| `stable_baselines3/common/torch_layers.py` | `MlpExtractor` |
| `stable_baselines3/ppo/ppo.py` | PPO training loop |
