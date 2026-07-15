"""Regression test for CausalA2C's decoupled causal-critic optimizer.

The one invariant worth pinning: the base optimizer (policy + standard critic)
and the causal optimizer must partition the policy's parameters exactly, with the
causal critic's params in the causal optimizer only. Breaking this is silent --
causal params ending up in both optimizers (double-stepped) or neither (never
trained) raises no error, just invalidates experiments.
"""

import stable_baselines3.common.envs  # noqa: F401  # registers CausalTracking-* envs
from stable_baselines3.a2c.causal_a2c import CausalA2C
from stable_baselines3.common.causal_policies import CausalActorCriticPolicy
from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs.causal_tracking import CausalTrackingEnv


def _make_policy() -> CausalActorCriticPolicy:
    env = CausalTrackingEnv()
    mask_manager = CausalMaskManager(env.adjacency_as(), env.adjacency_ss())
    env.close()
    venv = make_vec_env("CausalTracking-TransitionRewards-v0", n_envs=2, env_kwargs={"max_episode_steps": 20})
    return CausalA2C(
        CausalActorCriticPolicy, venv, n_steps=16, policy_kwargs={"mask_manager": mask_manager}, verbose=0
    ).policy


def test_optimizers_partition_params_with_causal_isolated():
    policy = _make_policy()

    causal_modules = {id(p) for p in policy.future_encoder.parameters()}
    causal_modules |= {id(p) for p in policy.causal_value_net.parameters()}
    base_in_opt = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    causal_in_opt = {id(p) for group in policy.causal_optimizer.param_groups for p in group["params"]}
    all_params = {id(p) for p in policy.parameters()}

    # Causal optimizer holds exactly the causal critic's params.
    assert causal_in_opt == causal_modules
    # Base optimizer holds none of them.
    assert base_in_opt.isdisjoint(causal_modules)
    # Together they cover every parameter exactly once.
    assert base_in_opt.isdisjoint(causal_in_opt)
    assert base_in_opt | causal_in_opt == all_params


def test_param_counts_resolve_and_are_positive():
    # Guards against a wrong module-attribute path silently crashing experiment launch.
    counts = _make_policy().param_counts()
    assert set(counts) == {"n_params_actor", "n_params_critic", "n_params_causal"}
    assert all(v > 0 for v in counts.values())
