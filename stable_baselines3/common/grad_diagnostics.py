"""Diagnostics for the variance of the policy gradient estimator.

Computes exact per-sample gradients g_i = grad_theta(A_i * log pi_theta(a_i | s_i))
in a single vectorized pass via ``torch.func`` (``functional_call`` + ``grad`` + ``vmap``)
through ``ActorCriticPolicy.evaluate_actions``.

Requires ``torch.distributions`` construction to be vmap-compatible (verified on
torch 2.13); ``tests/test_grad_diagnostics.py`` pins the result against an explicit
per-sample autograd loop.
"""

import torch as th
from torch.func import functional_call, grad, vmap

from stable_baselines3.common.policies import ActorCriticPolicy

ParamsDict = dict[str, th.Tensor]


class _EvaluateActionsWrapper(th.nn.Module):
    """``functional_call`` only invokes ``forward``, so wrap ``evaluate_actions``."""

    def __init__(self, policy: ActorCriticPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        # obs.shape == (1, obs_dim)
        # actions.shape == (1, action_dim) [Box] or (1,) [Discrete]
        _, log_prob, _ = self.policy.evaluate_actions(obs, actions)
        # log_prob.shape == (1,)
        return log_prob.squeeze(0)


def policy_gradient_variance(
    policy: ActorCriticPolicy,
    observations: th.Tensor,
    actions: th.Tensor,
    advantages: th.Tensor,
) -> float:
    """Trace of the sample covariance of the per-sample policy gradients.

    Returns sum over parameters p of Var_i[g_{i,p}], where
    g_i = grad_theta(advantages[i] * log pi_theta(actions[i] | observations[i])).
    Parameters not on the actor path (e.g. the value net) receive zero gradient and
    contribute zero variance. The variance of the batch-mean gradient estimator is
    this value divided by the batch size.

    :param policy: The actor-critic policy.
    :param observations: Batch of observations.
    :param actions: Batch of actions (long for Discrete, as in ``train()``).
    :param advantages: The exact advantage tensor used in the policy loss.
    :return: The gradient-variance trace as a python float.
    """
    # observations.shape == (batch, obs_dim)
    # actions.shape == (batch, action_dim) [Box] or (batch,) [Discrete]
    # advantages.shape == (batch,)
    assert not policy.use_sde, "policy_gradient_variance does not support gSDE policies"

    wrapper = _EvaluateActionsWrapper(policy)
    params: ParamsDict = {name: p.detach() for name, p in wrapper.named_parameters()}

    def objective(params: ParamsDict, obs: th.Tensor, action: th.Tensor, advantage: th.Tensor) -> th.Tensor:
        # obs.shape == (obs_dim,)
        # action.shape == (action_dim,) [Box] or () [Discrete]
        # advantage.shape == ()
        log_prob = functional_call(wrapper, params, (obs.unsqueeze(0), action.unsqueeze(0)))
        # log_prob.shape == ()
        return advantage * log_prob

    per_sample_grads = vmap(grad(objective), in_dims=(None, 0, 0, 0))(
        params, observations.detach(), actions.detach(), advantages.detach()
    )
    # per_sample_grads: dict with same keys as params; each value.shape == (batch, *param_shape)
    grad_variance = sum(
        (g.var(dim=0).sum() for g in per_sample_grads.values()),
        start=th.zeros((), device=advantages.device),
    )
    # grad_variance.shape == ()
    return float(grad_variance.item())
