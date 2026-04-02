# Causal Critic Implementation Roadmap

**Date:** 2026-03-20
**Goal:** Augment the critic in a policy gradient method with causal knowledge for variance reduction. Also add environment wrappers exposing causal domain knowledge.

---

## Repository Overview

This is a fork of Stable Baselines3 (SB3), PyTorch-based RL library. No causal-specific code exists yet — clean slate.

### Relevant Inheritance Chains

```
PPO/A2C → OnPolicyAlgorithm → BaseAlgorithm
ActorCriticPolicy → BasePolicy → BaseModel (nn.Module)
```

### Key Files

| File | Purpose |
|------|---------|
| `stable_baselines3/common/on_policy_algorithm.py` | `collect_rollouts()` — collects experience, populates buffer |
| `stable_baselines3/common/buffers.py` | `RolloutBuffer` — stores (obs, action, reward, value, log_prob, advantage, return) |
| `stable_baselines3/common/policies.py` | `ActorCriticPolicy` — base for causal policy |
| `stable_baselines3/common/torch_layers.py` | `MlpExtractor` — base for causal network layers |
| `stable_baselines3/ppo/ppo.py` | PPO training loop template |
| `stable_baselines3/a2c/a2c.py` | A2C training loop template |

---

## Architecture of the Existing Critic

The critic is a single linear layer:

```
Observation
  → FeaturesExtractor (CNN or Flatten)
  → MlpExtractor.value_net  → latent_vf
  → value_net: Linear(latent_dim_vf, 1)  ← target for causal replacement
```

The key method is `ActorCriticPolicy.evaluate_actions()` (`common/policies.py:~719`):

```python
def evaluate_actions(self, obs, actions):
    features = self.extract_features(obs)
    latent_pi, latent_vf = self.mlp_extractor(features)
    values = self.value_net(latent_vf)   # ← replace with causal critic
    distribution = self._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    return values, log_prob, entropy
```

This is called in:
- `ppo/ppo.py:train()` — per minibatch, `mse_loss(returns, values_pred)`
- `a2c/a2c.py:train()` — once per update, same loss

### GAE / Advantage Computation

Lives in `RolloutBuffer.compute_returns_and_advantage()`:

```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ...
returns_t = A_t + V(s_t)
```

---

## Three Implementation Tracks

### Track 1 — Environment Wrappers with Causal Knowledge

Create `stable_baselines3/common/causal_wrappers.py` (or a top-level `envs/` dir).

Standard Gymnasium wrappers via `gym.Wrapper`. Design decision: how to expose causal info?

**Options:**
1. Augment observation space — append causal variable annotations to obs vector
2. Add `info` dict fields — causal graph structure per step (no obs space change)
3. Dict observation space — `{"obs": ..., "causal": ...}` (cleanest for separate processing)

`make_vec_env()` in `env_util.py` accepts `wrapper_class` — wrappers integrate cleanly.

**Start here** — forces you to concretize the causal structure before touching the algorithm.

### Track 2 — Causal Critic Network

Create `stable_baselines3/common/causal_layers.py`.

Build a `CausalCritic` or `CausalMlpExtractor` that takes `(latent_vf, causal_info)` and outputs a lower-variance value estimate.

Causal variance reduction approaches:
- **Causal baseline:** $V(s) - \sum_i \hat{Q}(s, a_i) \cdot p(a_i \mid s_{\text{non-causal-parents of } a_i})$
- **Structured critic:** decompose value along the causal graph
- **Counterfactual baselines:** use do-calculus interventions

### Track 3 — Causal Policy

Subclass `ActorCriticPolicy` in `common/policies.py`:

```python
class CausalActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # replace self.value_net with causal critic

    def evaluate_actions(self, obs, actions):
        # inject causal graph information into value computation
```

### Track 4 — Algorithm (minimal, if needed)

Create `stable_baselines3/causal_ppo/` inheriting from `OnPolicyAlgorithm` (or PPO directly).

**Only needed if:**
- Extra causal quantities must be stored during rollout → subclass `RolloutBuffer`
- Training loss needs modification (e.g., causal regularization term)

Otherwise, just pass `policy=CausalActorCriticPolicy` to standard PPO/A2C.

---

## Recommended Implementation Order

1. **Define the causal representation** — what is the causal graph for target environments? This determines everything downstream.
2. **Write environment wrappers** — self-contained, forces concretization of causal structure.
3. **Build `CausalCritic`** in `causal_layers.py`.
4. **Subclass `ActorCriticPolicy`** → `CausalActorCriticPolicy`.
5. **Wire into algorithm** — try passing the policy to standard PPO first; only create `CausalPPO` if the training loop needs changes.

---

## Notes on RolloutBuffer Extension

If causal quantities need to be stored per timestep during rollout, subclass `RolloutBuffer`:

```python
class CausalRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, causal_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_info = np.zeros((self.buffer_size, self.n_envs, causal_dim))

    def add(self, *args, causal_info, **kwargs):
        self.causal_info[self.pos] = causal_info
        super().add(*args, **kwargs)
```

Pass it via `rollout_buffer_class` in `OnPolicyAlgorithm.__init__()`.
