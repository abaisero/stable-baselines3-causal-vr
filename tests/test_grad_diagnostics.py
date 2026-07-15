"""Tests for the policy gradient variance diagnostic.

The invariant worth pinning: the vectorized (``vmap``) per-sample gradient variance
must match an explicit per-sample autograd loop through ``evaluate_actions``. This
also guards against ``torch.distributions`` construction becoming vmap-incompatible
in a future torch version — it would fail loudly here, not silently drift.
"""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

import stable_baselines3.common.envs  # noqa: F401  # registers CausalTracking-* envs
from stable_baselines3 import A2C
from stable_baselines3.a2c.causal_a2c import CausalA2C
from stable_baselines3.common.causal_policies import CausalActorCriticPolicy
from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs.causal_tracking import CausalTrackingEnv
from stable_baselines3.common.grad_diagnostics import policy_gradient_variance
from stable_baselines3.common.policies import ActorCriticPolicy

N_OBS = 5
BATCH = 16


def _make_policy(action_space: spaces.Space) -> ActorCriticPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(N_OBS,), dtype=np.float32)
    return ActorCriticPolicy(obs_space, action_space, lr_schedule=lambda _: 3e-4)


def _make_batch(policy: ActorCriticPolicy) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    observations = th.randn(BATCH, N_OBS)
    # observations.shape == (BATCH, N_OBS)
    if isinstance(policy.action_space, spaces.Discrete):
        actions = th.randint(int(policy.action_space.n), (BATCH,))
        # actions.shape == (BATCH,)
    else:
        actions = th.randn(BATCH, *policy.action_space.shape)
        # actions.shape == (BATCH, action_dim)
    advantages = th.randn(BATCH)
    # advantages.shape == (BATCH,)
    return observations, actions, advantages


def _actor_parameters(policy: ActorCriticPolicy) -> list[th.Tensor]:
    params = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    if hasattr(policy, "log_std"):
        params.append(policy.log_std)
    return params


@pytest.fixture(params=["discrete", "box"])
def policy(request: pytest.FixtureRequest) -> ActorCriticPolicy:
    th.manual_seed(0)
    action_space: spaces.Space
    if request.param == "discrete":
        action_space = spaces.Discrete(4)
    else:
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return _make_policy(action_space)


def test_grad_variance_matches_autograd_loop(policy: ActorCriticPolicy) -> None:
    observations, actions, advantages = _make_batch(policy)

    result = policy_gradient_variance(policy, observations, actions, advantages)

    _, log_prob, _ = policy.evaluate_actions(observations, actions)
    # log_prob.shape == (BATCH,)
    actor_params = _actor_parameters(policy)
    per_sample_grads = []
    for i in range(BATCH):
        grads = th.autograd.grad(advantages[i] * log_prob[i], actor_params, retain_graph=True)
        per_sample_grads.append(th.cat([g.flatten() for g in grads]))
    grad_matrix = th.stack(per_sample_grads)
    # grad_matrix.shape == (BATCH, n_actor_params)
    expected = grad_matrix.var(dim=0).sum().item()

    assert result == pytest.approx(expected, rel=1e-3, abs=1e-8)


def test_a2c_records_grad_variance() -> None:
    venv = make_vec_env("CausalTracking-TransitionRewards-v0", n_envs=2, env_kwargs={"max_episode_steps": 20})
    model = A2C("MlpPolicy", venv, n_steps=8, seed=0)
    model.learn(total_timesteps=16)
    grad_variance = model.logger.name_to_value["diagnostics/grad_variance"]
    grad_variance_per_sample = model.logger.name_to_value["diagnostics/grad_variance_per_sample"]
    assert np.isfinite(grad_variance)
    assert grad_variance_per_sample == pytest.approx(grad_variance / 16)


def test_a2c_records_grad_variance_box_actions() -> None:
    model = A2C("MlpPolicy", "Pendulum-v1", n_steps=8, seed=0)
    model.learn(total_timesteps=8)
    assert np.isfinite(model.logger.name_to_value["diagnostics/grad_variance"])


def test_causal_a2c_records_grad_variance() -> None:
    env = CausalTrackingEnv()
    mask_manager = CausalMaskManager(env.adjacency_as(), env.adjacency_ss())
    env.close()
    venv = make_vec_env("CausalTracking-TransitionRewards-v0", n_envs=2, env_kwargs={"max_episode_steps": 20})
    model = CausalA2C(CausalActorCriticPolicy, venv, n_steps=8, policy_kwargs={"mask_manager": mask_manager}, seed=0)
    model.learn(total_timesteps=16)
    assert np.isfinite(model.logger.name_to_value["diagnostics/grad_variance"])
