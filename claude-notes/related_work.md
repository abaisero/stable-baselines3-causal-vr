# Related work: future-conditional / exogenous baselines (2026-07-02)

Literature survey for the causal-baseline project. Verified via web search
2026-07-02. Companion to `causal_a2c_review.md` (section 5 describes our
proposed exogenous-conditioning extension).

**The uncomfortable fact up front:** the core theorem — *a baseline conditioned
on future information that is (conditionally) independent of the current action
keeps the policy gradient unbiased and weakly reduces variance* — is
independently established in at least four published papers (Mao 2019, Mesnard
2021, Nota 2021, Guo 2021). It cannot be claimed as a contribution. See the
positioning section at the end for what remains.

---

## Core papers (must cite and differentiate)

### 1. Mao, Venkatakrishnan, Schwarzkopf, Alizadeh — "Variance Reduction for RL in Input-Driven Environments" (ICLR 2019)

- arXiv: https://arxiv.org/abs/1807.02264 · code: https://github.com/hongzimao/input_driven_rl_example
- **Setting:** "input-driven MDPs" — dynamics/rewards depend on an exogenous,
  stochastic input process z (network traces, job arrivals, disturbances).
- **Contribution:** proves the *input-dependent baseline* b(s, z_{t:∞}) is
  bias-free; analytically shows its advantage over state-only baselines;
  handles the learning problem with (a) a meta-learned baseline (MAML across
  input sequences) and (b) "input repetition" (reusing one input sequence
  across rollouts).
- **Relation to us:** closest to our *exogenous-conditioning extension* — their
  theorem covers it. Differences: their exogenous process is a **given,
  monolithic, observable entity**; they never derive what is safe to condition
  on (it is handed to them), and they do not confront action-dependent
  termination or feature-level partial exogeneity.

### 2. Mesnard et al. — "Counterfactual Credit Assignment in Model-Free RL" (ICML 2021)

- arXiv: https://arxiv.org/abs/2011.09464 · PMLR: https://proceedings.mlr.press/v139/mesnard21a.html
- **Setting:** general MDPs, no structural knowledge assumed.
- **Contribution:** family of policy gradient algorithms using
  *future-conditional value functions* V(s, Φ) as baselines/critics; proves
  they are unbiased and low-variance **provided Φ ⊥ a_t | s_t**; since Φ is a
  *learned* hindsight embedding of the trajectory, independence is enforced
  approximately via a learned constraint (hindsight classifier / conditional
  independence penalty).
- **Relation to us:** the closest in spirit — future-conditional baseline with
  an explicit action-independence requirement. Difference: they **learn**
  approximate independence (residual bias, no guarantee); we get **exact**
  validity from known structure (causal graph now, exogeneity labels in the
  extension). We are the white-box end of their spectrum.

### 3. Nota, Thomas, Castro da Silva — "Posterior Value Functions: Hindsight Baselines for Policy Gradient Methods" (ICML 2021)

- PMLR: https://proceedings.mlr.press/v139/nota21a.html
- **Setting:** POMDPs; extended to arbitrary stochastic MDPs by modeling
  environment stochasticity as a special case of partial observability.
- **Contribution:** *posterior value functions* — baselines that use future
  observations to infer the posterior over hidden state components at earlier
  timesteps; proves these baselines **reduce and never increase** variance
  relative to state value functions; convergence guarantees for learning them.
- **Relation to us:** another independent proof of the core theorem, from the
  partial-observability angle. Their "hidden component inferred from the
  future" plays the role of our exogenous variables. No causal graph, no
  feature-level criterion.

### 4. Guo et al. — "Hindsight Value Function for Variance Reduction in Stochastic Dynamic Environment" (IJCAI 2021)

- arXiv: https://arxiv.org/abs/2107.12216 · IJCAI: https://www.ijcai.org/proceedings/2021/341
- **Contribution:** replaces the state value function with a *hindsight value
  function* conditioned on learned future embeddings; enforces independence of
  the embedding from previous actions via an **information-theoretic penalty**
  (mutual-information style), i.e. the same learned-independence strategy as
  CCA with a different regularizer. Shows consistent variance reduction and
  improved policies.
- **Relation to us:** same contrast as Mesnard — learned approximate
  independence vs. our exact structural validity.

---

## Adjacent papers (cite for context)

### 5. Dietterich, Trimponias, Chen — "Discovering and Removing Exogenous State Variables and Rewards for RL" (ICML 2018)

- arXiv: https://arxiv.org/abs/1806.01584 · PMLR: http://proceedings.mlr.press/v80/dietterich18a.html
- Formalizes exo/endo state decomposition: MDP splits into an exogenous Markov
  reward process + endogenous MDP; gives a variance-covariance condition under
  which solving the endogenous MDP alone accelerates learning; algorithms for
  *discovering* the decomposition from data.
- **Relation:** they **remove** exogenous reward; we **condition on** the
  exogenous future. Complementary mechanism, same enemy (exogenous reward
  variance). Their discovery algorithms are relevant if we ever want to learn
  exogeneity labels instead of assuming them.
- Follow-up: "RL with Exogenous States and Rewards" (arXiv 2023,
  https://arxiv.org/abs/2303.12957) — expanded journal treatment.

### 6. Venuto, Lau, Precup, Nachum — "Policy Gradients Incorporating the Future" (NeurIPS 2021)

- arXiv: https://arxiv.org/abs/2108.02096
- Lets the agent condition on hindsight trajectory information during training
  via latent variables from a backwards RNN / transformer, with a KL
  information bottleneck to limit reliance on privileged future information.
  Applied on top of PPO, SAC, BRAC.
- **Relation:** conditions the *policy/critic learning*, not specifically an
  unbiased baseline; the bottleneck is a soft version of the independence
  requirement. Context citation, not a direct competitor.

### 7. Harutyunyan et al. — "Hindsight Credit Assignment" (NeurIPS 2019)

- arXiv: https://arxiv.org/abs/1912.02503
- Reverses the credit-assignment question: learns the distribution of past
  actions given future outcomes and uses it to reweight credit. Future-aware
  credit assignment, but a different estimator family (not a baseline
  subtraction). Context citation.

### 8. Wu et al. — "Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines" (ICLR 2018)

- arXiv: https://arxiv.org/abs/1803.07246
- Action-dependent (not future-dependent) baselines exploiting factorized
  action spaces; unbiased per-factor baselines conditioned on *other* action
  dimensions. Orthogonal axis of baseline enrichment; useful to cite when
  discussing per-action-dim masks (`descendant_mask_per_dim` is the
  future-side analogue of their factorization).

### 9. To check: "Causal Graph-Factored Advantage PPO" (CGFA-PPO), arXiv 2605.06066 (May 2026)

- Surfaced in search results (Magic: The Gathering benchmark paper); reportedly
  uses per-factor critic heads indexed by SCM parents.
- **Action item:** read this — sounds like the nearest recent work using an
  explicit causal graph inside a PPO critic. Unclear from the abstract whether
  it targets baseline validity/variance or just critic architecture.

---

## Taxonomy: what prior work actually conditions on

Prior work landed on two versions of the future-conditional baseline — neither
of which is our masked-features version:

| Cell | Conditioning information | Independence guarantee | Who |
|------|--------------------------|------------------------|-----|
| A | **Exogenous process, given.** A monolithic, observable input stream z handed to the algorithm. No criterion for finding it; no extraction from state features. | Exact, by assumption | Mao 2019 |
| B | **Endogenous future observations, learned selection.** Arbitrary learned embeddings of the full future trajectory (endogenous state included), pushed toward action-independence by a soft penalty (hindsight classifier / MI penalty / KL bottleneck). | Approximate; residual bias, no guarantee | Mesnard 2021, Guo 2021, (Venuto 2021); Nota 2021 is analytical rather than learned but likewise uses future observations without feature-level structure |
| C | **Endogenous future observation *features*, selected exactly by a known causal graph.** Deterministic, checkable criterion (descendant propagation, per-horizon masks). | Exact, by structure | **Us** (no prior occupant found; CGFA-PPO 2026 unread) |

Two consequences:

- **It is NOT correct that prior work is exclusively exogenous-based.** Only
  Mao is. Mesnard/Guo/Venuto condition on endogenous future data — but with
  learned, approximate independence rather than a graph criterion. Cell C is
  a genuinely unoccupied third option, not a rediscovery of either existing
  cell.
- **The exogenous-conditioning extension (causal_a2c_review.md §5) moves
  TOWARD Mao's occupied cell A, not away from it.** Its novel content over
  Mao is real but incremental: per-feature structural noises inside coupled
  dynamics (vs. a monolithic input stream), the fixed-window /
  imaginary-continuation treatment of termination, and recoverability of
  noises from transitions. So the extension buys robustness and simplicity at
  the cost of distinctiveness, while the masked-endogenous method (cell C) is
  the more differentiated contribution but carries the graph-correctness
  assumptions. **A paper containing both, with this trade-off made explicit,
  is stronger than either alone.**

---

## Positioning: what is NOT novel, and what still is

**Not novel (do not claim):**
- The theorem that future-conditional baselines with action-independent
  conditioning are unbiased (Mao, Mesnard, Nota, Guo — four independent
  versions).
- The observation that exogenous processes cause baseline-resistant variance
  (Mao, Dietterich).
- Conditioning on an exogenous input sequence per se (Mao).

**Still open / plausibly ours:**
1. **The feature-level, known-graph criterion.** Deriving *what* is safe to
   condition on from a causal graph over observation features
   (adjacency_as/adjacency_ss + descendant propagation + per-horizon masks),
   covering environments where exogenous structure is *embedded in the state*
   and features are partially coupled — rather than a monolithic given input
   process (Mao) or a learned embedding with approximate independence
   (Mesnard, Guo).
2. **The validity analysis.** When ss-propagation suffices vs. policy-mediated
   descendant paths (self-loop closure condition); the endogenous-stopping-time
   leak; the "subsetting is valid iff the selection rule is exogenous" rule;
   fixed windows via imaginary continuations of the autonomous exogenous
   process. None of this appears in the papers above.
3. **Exact vs. learned independence as a study.** The known-structure setting
   gives an upper bound on what CCA-style learned methods can achieve; a
   controlled comparison (exact masking vs. learned constraint) would be a
   contribution.
4. **The diagnostic/empirical angle.** Explicitly measuring
   Var[A·∇log π] with an environment (tracking) purpose-built to decouple
   exogenous reward noise from action effect — the papers above evaluate
   end-task return, not the variance mechanism itself.
5. **Per-action-dimension future masks** (`descendant_mask_per_dim`) — a
   future-side analogue of Wu et al.'s action factorization; unexplored.

**Framing recommendation: criterion + analysis + mechanism study** — not "we
prove future-conditional baselines are unbiased." Concretely:

- **Criterion** = the rule deciding *what future information may enter the
  baseline*. Mao's answer: "the input process someone hands you."
  Mesnard/Guo's answer: "whatever a network learns, penalized toward
  independence." Ours: a deterministic rule computed from the feature-level
  causal graph (descendant propagation, per-horizon masks,
  per-action-dimension variants). The criterion itself — not the unbiasedness
  theorem it plugs into — is the contribution.
- **Analysis** = the validity conditions under which the practical estimator
  stays unbiased, none of which appear in prior work because their settings
  never surface them: when ss-propagation suffices vs. policy-mediated
  descendant paths (self-loop closure condition); the endogenous-stopping-time
  leak; subsetting-is-valid-iff-the-selection-rule-is-exogenous; fixed windows
  via imaginary continuations.
- **Mechanism study** = experiments that measure the variance mechanism itself
  — Var[A·∇log π], with the tracking env purpose-built to decouple exogenous
  reward noise from action effect — instead of only end-task return, which is
  all the prior papers report. Plus the comparison the field hasn't run:
  exact structural conditioning as an upper bound vs. learned-independence
  methods (CCA-style) on the same environments.

One-sentence framing for the paper: *the unbiasedness of future-conditional
baselines is known (cite Mao, Mesnard); what is not known is how to identify
valid conditioning information from feature-level structure, under what
conditions the resulting practical estimator stays valid, and how much of the
learned methods' gap to the exact baseline remains — and that is what we
supply.*
