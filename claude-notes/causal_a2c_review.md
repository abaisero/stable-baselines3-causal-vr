# Causal A2C review notes (2026-07-02)

Findings from a code review of `CausalA2C` vs standard `A2C`, covering
`stable_baselines3/a2c/causal_a2c.py`, `common/causal_buffers.py`,
`common/causal_policies.py`, `common/causal_encoders.py`, and
`common/causal_utils.py`.

**Overall verdict:** the idea is sound and the implementation is faithful to it.
No correctness bugs were found. The items below are (1) validity conditions the
framework silently assumes, (2) a design decision about boundary bootstrapping,
(3) minor implementation observations, and (4) a proposed extension —
conditioning on exogenous variables instead of masked futures — that resolves
most of the earlier concerns (section 5).

---

## 1. What was verified as correct

- **Unbiasedness of the baseline as applied.** The policy gradient uses
  `advantages = returns - causal_old_values` (causal_a2c.py:137), where
  `causal_old_values` is frozen at collection time. The baseline applied to a
  rollout was therefore never fit on that same rollout, so there is no
  same-batch fitting bias. `causal_advantages` / `causal_returns` are used only
  to train the causal critic, never in the policy gradient.

- **Optimizer / gradient-clipping split.** The two losses touch disjoint
  parameter sets: `predict_causal_values` bypasses the shared feature extractor
  entirely, and `CausalActorCriticPolicy._build` asserts that the base and
  causal parameter sets are disjoint and jointly cover all parameters. Because
  of this disjointness, a single `loss.backward()` followed by per-group
  clipping and two optimizer steps is exactly equivalent to two separate
  backward passes. Correct as written.

- **`FutureRolloutBuffer._build_future_arrays` boundary logic.** Checked by
  hand: reverse iteration over episode starts, multiple boundaries per env, and
  a boundary at t=0 are all handled correctly. The anchor immediately before a
  reset gets an empty future rather than leaking the post-reset observation.

---

## 2. Validity conditions the framework assumes (not checked by code)

### 2a. Descendant propagation ignores policy-mediated paths

`CausalMaskManager` propagates descendance only through `adjacency_ss`
(causal_utils.py:38). But the real on-policy causal graph also contains the
path:

```
a_t  →  s_{t+1} (descendant features)  →  a_{t+1}  →  image(adjacency_as)
```

Because the policy reads the *whole* state, once any feature is a descendant of
`a_t`, every feature directly caused by actions becomes a descendant one step
later. The mask computed by `CausalMaskManager` is therefore correct **only
when `image(adjacency_as)` is already contained in the ss-propagated descendant
set** — in practice, when action-affected features have self-loops in
`adjacency_ss`.

- **Tracking env:** satisfied. Action → x only, and x → x', so the descendant
  set is exactly {x} at every horizon even accounting for policy-mediated
  paths. The mask is exactly right.
- **HalfCheetah:** fine as long as the hand-written adjacency keeps
  action-affected joint states self-persistent. Verify this when the adjacency
  is written.
- This is an assumption of the framework, not something any code checks. A
  future assert (image of `adjacency_as` ⊆ its own ss-closure) would make it
  explicit.

### 2b. The future mask leaks episode length

The padding pattern in `future_mask` encodes time-until-episode-end. If episode
termination were action-dependent (e.g. Hopper-style falls), the baseline would
condition on a descendant of `a_t` and become **biased** — silently.

- Both current envs terminate only via TimeLimit (exogenous), and truncation at
  the rollout-buffer end is also exogenous, so this is a non-issue today.
- It becomes a real issue the moment the method is applied to an env with
  early, action-dependent termination. Keep this in mind when choosing new
  benchmark environments.

---

## 3. Boundary bootstrapping: use the standard critic when no future exists

**Decision reached in discussion (2026-07-02): whenever no future window is
observable, fall back to the standard critic.**

Reasoning:

- With an empty future, the causal critic's target degenerates to the marginal
  E[G | s] — which is *exactly* the standard critic's native objective. The
  standard critic is trained toward that quantity on every anchor of every
  rollout; the causal critic's empty-future mode is supervised only at
  rollout-final anchors. At any boundary, the standard critic is simply the
  better-estimated version of the same function.
- The coupling is one-directional (the standard critic's targets never depend
  on the causal critic), so there is no feedback loop. The V_std term acts as
  an anchor tying the causal critic's degenerate mode to a well-trained
  estimate rather than letting it self-bootstrap.
- "The two critics coincide" holds at convergence, not during training — but
  early in training the alternative injects the causal critic's own
  empty-future bias, which is trained on strictly less data. The standard
  critic is the better of the two imperfect options at every stage.

Consequences for the code:

1. **Timeout bootstrap (already correct, by inheritance).** The parent
   `collect_rollouts` patches `rewards[step] += γ · V_std(terminal_obs)` for
   truncated episodes, and `compute_causal_returns_and_advantage` reuses those
   patched rewards. Originally flagged as "cross-critic contamination", but per
   the reasoning above this is actually the *desired* behavior. No change
   needed.

2. **Buffer-end bootstrap (should change; resolves the TODO at
   causal_a2c.py:98).** The rollout-end bootstrap currently calls
   `predict_causal_values` with an empty future — the causal critic's
   least-trained mode. It should instead use
   `self.policy.predict_values(last_obs)`. As written the code is
   inconsistent: timeout boundaries bootstrap with V_std while the buffer end
   bootstraps with the causal critic's empty-future output. Switching unifies
   both under the same rule.

3. **The empty-future mode still gets trained either way.** Rollout-final
   anchors keep their empty-future input with target
   r + γ · V_std · (1 − done) + …, so `predict_causal_values` with no future is
   still pulled toward the standard critic's estimate — the right fixed point —
   rather than being left unsupervised.

---

## 4. Minor implementation observations

- **`_build_future_arrays` runs twice per iteration** — once in
  `CausalA2C.collect_rollouts` (causal_a2c.py:71) and again inside
  `FutureRolloutBuffer.get()` (causal_buffers.py:216). Pure duplicated work.

- **Quadratic memory in the future window.** The buffer materializes the full
  (T, E, T−1, O) future array and tensorizes it, even though `MLPFutureModel`
  immediately truncates to horizon 20. Fine at current sizes (`n_steps` is
  small); worth restructuring to build only horizon-length windows if
  `n_steps` grows.

- **`MLPFutureModel` cannot distinguish padding from genuine zeros.** The
  temporal mask multiplies features to zero but is not itself an input, and in
  the tracking env a value near 0 is entirely plausible. The encoder-decoder
  variant handles this properly (learned sentinel for causal masking +
  attention padding mask); for the MLP, concatenating `future_mask` to the
  input would remove the ambiguity.

- **Hardcoded hyperparameters** in `_build_causal_critic`
  (causal_policies.py:31): horizon H = 20, future-encoding dim F = 16, and the
  choice of `MLPFutureModel` itself. Fine for now; these are exactly the knobs
  sweeps will eventually want to reach.

---

## 5. Proposed extension: condition on exogenous variables instead of masked futures

**Idea (proposed 2026-07-02):** instead of conditioning the causal critic on
the causally-masked future observations, condition it on the *exogenous
variables* (structural noise terms) that drive the future. In simulation these
are directly available by tracking the RNG samples.

### 5a. Why this is valid, and strictly stronger than masking

- **Unbiasedness becomes structural rather than graph-dependent.** Exogenous
  noises U are independent of a_t given s_t *by construction* — no propagation
  through adjacency matrices, no policy-mediated-path condition, no self-loop
  assumption. E[b(s_t, U) · ∇log π] = 0 holds regardless of what the causal
  graph looks like. This entirely eliminates the section 2a class of problems
  (and the `CausalMaskManager` correctness burden with it). This is
  essentially the input-driven baseline of Mao et al. (ICLR 2019), which comes
  with exactly this unbiasedness proof.

- **Strictly more variance-reduction potential, not just equivalent.** The
  exogenous variables determine all non-descendants of the action (given s_t),
  so no information is lost relative to masking. But they contain *more*:
  exogenous noises that drive *descendant* features (e.g. noise_x in the
  tracking env) are still exogenous, so conditioning on them is valid — and
  the optimal b(s, U) then cancels their contribution to return variance too,
  which the masked approach structurally cannot (x is masked out entirely).
  With `reward_type="transition"`, noise_x leaks into the reward through x′,
  so this is a real (if small, σ_x = 0.1) term. Richer conditioning never
  raises the variance floor of the optimal baseline.

### 5b. The episode-length leak, resolved via fixed windows

The section 2b leak never came from the window's *contents* — it came from the
window's *length* being an action-dependent stopping time. Two principles,
both established in discussion:

- **Imaginary continuations are valid.** The exogenous process is by
  definition autonomous (its law does not depend on the agent), so it is
  well-defined arbitrarily far beyond episode termination, and in simulation
  it can be materialized by rolling the RNG stream forward (or pre-sampling
  it). Conditioning on a *fixed-length* window u_{t:t+H} of that stream is
  unbiased no matter when the episode ends — there is no length variable left
  to leak anything.

- **Conditioning on fewer things is always valid — IF the selection rule is
  exogenous.** Any deterministic function of a valid conditioning set is
  itself valid: if Φ ⊥ a_t | s_t then f(Φ) ⊥ a_t | s_t. So truncating to a
  shorter fixed horizon, dropping variables, or compressing lossily are all
  safe. The sharpening: the selection rule must itself be exogenous.
  Truncating *at episode end* is NOT "conditioning on fewer things" in this
  sense — the truncation index is an endogenous stopping time, so the observed
  object is f(Φ, T(a)), a function of the actions too. That is the entire
  content of the leak. Fixed horizon H, buffer-end truncation, and TimeLimit
  are exogenous selection rules and fine; "until the agent died" is not.

- **Counterfactual well-definedness caveat:** the k-th future exogenous draw
  must be well-defined regardless of agent behavior, i.e. the exogenous RNG
  stream must not be interleaved with action-dependent RNG consumption.
  Pre-sampling the exogenous trajectory per episode, or giving the exogenous
  process a dedicated RNG stream, guarantees this.

### 5c. Follow-on simplifications under the fixed-window design

If every anchor gets a full H-window of exogenous variables (stream extends
past both episode ends and the rollout-buffer end):

1. **The empty-future degenerate mode disappears.** The critic's input
   distribution becomes stationary (no shorter futures near the buffer end),
   and the section 3 boundary-bootstrap question mostly dissolves — the causal
   critic can be queried with a full window everywhere, including at the
   buffer-end bootstrap, instead of falling back to the standard critic.

2. **The temporal validity mask goes away entirely.** No padding, no
   `future_mask` input, no padding-vs-genuine-zero ambiguity in the MLP
   encoder — the window is always dense.

### 5d. Costs and remaining requirements

- **Correctness burden shifts to the exogeneity labels.** "Exogenous" must
  mean uninfluenced by actions through *any* path, including policy-mediated
  ones; a mislabeled variable reintroduces silent bias. This is a much more
  robust object than a full adjacency graph plus closure computation (one bit
  per variable, usually obvious by design), but it is still an assumption.

- **Harder critic-modeling task.** b(s, U) over a noise sequence is a richer
  regression target; with function approximation, richer inputs can hurt even
  though the optimal baseline can only improve. This is the deliberate
  trade-off of the extension.

- **Simulator instrumentation.** The env must expose its exogenous draws (or
  accept an injected pre-sampled exogenous trajectory) on a dedicated stream.
  Small change for the tracking env. For MuJoCo-style envs the exogenous
  content is mostly just initial-state noise, so there is little to condition
  on regardless.

- **Expected experimental impact on tracking: minimal.** The exogenous set
  there (future y, z, plus σ_x = 0.1 worth of noise_x) is nearly what the mask
  already passes through. The payoff is generality and the removal of the
  graph-correctness assumption, not tracking-env performance.
