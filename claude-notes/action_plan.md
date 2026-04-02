# Causal RL — Action Plan

**Date:** 2026-03-27

This document tracks what has been implemented and what still needs to be done.

---

## What Exists

| File | Status | Description |
|------|--------|-------------|
| `common/causal_wrappers.py` | Done | `CausalEnvWrapper` base class with `adjacency_as`, `adjacency_ss`, `descendant_mask`, `non_descendant_mask` |
| `common/envs/causal_half_cheetah.py` | Done | `CausalHalfCheetah` with full adjacency matrices; has a temporary `_debug_reset_prob` hook (remove before real training) |
| `common/causal_buffers.py` | Done | `FutureRolloutBuffer` + `FutureRolloutBufferSamples` — provides future observation sequences and mask alongside standard PPO fields |
| `common/causal_policies.py` | Skeleton | `CausalActorCriticPolicy` with `_build_mlp_extractor` and `_build_causal_critic` / `predict_causal_values` stubs |
| `a2c/causal_a2c.py` | Skeleton | `CausalA2C` with `collect_rollouts`, `causal_baseline_values`, `causal_baseline_loss`, `causal_baseline_predictions`, `causal_baseline_targets` stubs |
| `ppo/causal_ppo.py` | Skeleton | `CausalPPO` with same stubs as `CausalA2C` |

---

## Action Plan

### ~~1. Remove debug hook from `CausalHalfCheetah`~~ — Done

`_debug_reset_prob` set to `0.0` in `common/envs/causal_half_cheetah.py`.

---

### 2. Implement `FutureRolloutBuffer` causal returns/advantages

**File:** `common/causal_buffers.py`

Add fields to `FutureRolloutBuffer`:
- `causal_values: np.ndarray` — shape `(T, E)`, causal critic predictions collected at rollout time
- `causal_advantages: np.ndarray` — shape `(T, E)`, GAE computed using causal values
- `causal_returns: np.ndarray` — shape `(T, E)`, lambda-returns computed using causal values

Add method:
```python
def compute_causal_returns_and_advantage(self, last_causal_values: np.ndarray, dones: np.ndarray) -> None:
    # same GAE recurrence as compute_returns_and_advantage, using causal_values
```

Add `causal_advantages` and `causal_returns` to `FutureRolloutBufferSamples`.

Update `_get_samples` and `get` to include these new fields (swap_and_flatten them alongside the others).

---

### 3. Implement `CausalActorCriticPolicy`

**File:** `common/causal_policies.py`

#### 3a. `_build_causal_critic()`
Instantiate the causal critic network as an `nn.Module` attribute (e.g. `self.causal_value_net`).
It must be assigned here so its parameters are included in `self.parameters()` when the optimizer
is built at the end of `_build()`.

Architecture decision: does the causal critic share the MLP extractor with the standard critic,
or process raw features independently? Sharing is cheaper; separate is more flexible.

#### 3b. `predict_causal_values(obs)`
Implement the causal critic forward pass. Called during `collect_rollouts` to fill
`causal_values` in the buffer.

---

### 4. Wire `CausalActorCriticPolicy` into `CausalA2C`

**File:** `a2c/causal_a2c.py`

#### 4a. `__init__`
- Remove the `causal_baseline_values` stub on `CausalA2C` — this logic moves to the policy.
- Use `CausalActorCriticPolicy` as the default policy (or enforce it via an assertion).

#### 4b. `collect_rollouts`
After `super().collect_rollouts(...)`:
1. Call `self.policy.predict_causal_values(all_obs)` on all buffer observations (single batched forward pass).
2. Store results in `rollout_buffer.causal_values`.
3. Call `rollout_buffer.compute_causal_returns_and_advantage(last_causal_values, dones)`.

The `last_causal_values` (bootstrap value for the state after the final step) needs to be
computed separately — mirror how `collect_rollouts` handles `last_values` for the standard critic.

#### 4c. `train`
- Replace `rollout_data.advantages` with `rollout_data.causal_advantages` for the policy gradient.
- Implement `causal_baseline_loss` using `rollout_data.causal_returns` as the learning target.
- Remove TODO comments once implemented.

---

### 5. Implement `causal_baseline_loss` in a concrete subclass

**New file:** (e.g. `a2c/causal_a2c_halfcheetah.py` or directly in `causal_a2c.py`)

The loss trains the causal critic to predict `causal_returns`. Standard MSE:
```python
def causal_baseline_loss(self, rollout_data):
    preds = self.policy.predict_causal_values(rollout_data.observations)
    return F.mse_loss(rollout_data.causal_returns, preds)
```

Implement `causal_baseline_predictions` and `causal_baseline_targets` to return predictions
and `causal_returns` respectively, enabling `train/baseline_explained_variance` logging.

---

### 6. Mirror all of the above in `CausalPPO`

**File:** `ppo/causal_ppo.py`

Same changes as `CausalA2C` (steps 4–5). PPO-specific considerations:
- Multiple epochs: causal advantages are fixed at rollout time, same as standard advantages.
- Value clipping (`clip_range_vf`): decide whether to apply it to the causal critic too.
- The extra `rollout_buffer.get(batch_size=None)` call after the training loop (for baseline
  explained variance) should be removed once `causal_baseline_predictions` /
  `causal_baseline_targets` are implemented — at that point they are called per mini-batch.

---

### 7. Decide: shared or separate features extractor for the causal critic

Currently unresolved. Options:

- **Shared** (`share_features_extractor=True` style): causal critic takes `latent_vf` from
  `MlpExtractor` as input. Fewer parameters, faster. Risk: the shared extractor may not learn
  features useful for the causal structure.
- **Separate**: causal critic has its own feature extraction path. More parameters, more
  flexibility. Could also use `future_observations` as input (not just the anchor obs).

This decision affects `_build_causal_critic` and `predict_causal_values` architecture.

---

### 8. (Optional) Use `future_observations` in the causal critic

The causal critic currently receives only the anchor observation. `FutureRolloutBufferSamples`
also provides `future_observations (batch, max_future_len, obs_dim)` and `future_mask`.
A sequential model (e.g. transformer, LSTM) over the future trajectory could produce a
lower-variance value estimate. This is the core research contribution.

`predict_causal_values` would need to accept the full `FutureRolloutBufferSamples` rather than
just `observations` if this path is taken.
