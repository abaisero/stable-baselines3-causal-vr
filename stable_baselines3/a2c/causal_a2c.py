import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.causal_buffers import FutureRolloutBuffer, FutureRolloutBufferSamples
from stable_baselines3.common.causal_policies import CausalActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv


class CausalA2C(A2C):
    """
    A2C subclass that uses :class:`FutureRolloutBuffer` and overrides ``train()``
    to additionally train a causal baseline using future observation sequences.

    The policy gradient update is identical to standard A2C.  Subclasses should
    override ``causal_value_loss()`` to implement the baseline objective.

    Accepts all the same arguments as :class:`A2C`.  If ``rollout_buffer_class``
    is provided it must be :class:`FutureRolloutBuffer` or a subclass thereof.
    """

    def __init__(self, *args, causal_vf_coef: float = 1.0, rollout_buffer_class: type[RolloutBuffer] | None = None, **kwargs):
        if rollout_buffer_class is None:
            rollout_buffer_class = FutureRolloutBuffer
        assert issubclass(rollout_buffer_class, FutureRolloutBuffer), (
            f"CausalA2C requires a FutureRolloutBuffer, got {rollout_buffer_class}"
        )
        super().__init__(*args, rollout_buffer_class=rollout_buffer_class, **kwargs)
        self.causal_vf_coef = causal_vf_coef

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        """
        Collect rollouts and populate the causal fields of the buffer.

        Calls the parent implementation to fill standard fields (observations, actions,
        rewards, values, log_probs, advantages, returns), then:

        1. Builds padded future observation windows via ``_build_future_arrays()``.
        2. Runs ``predict_causal_values()`` on all buffer observations in a single
           batched forward pass to fill ``rollout_buffer.causal_values``.
        3. Calls ``compute_causal_returns_and_advantage()`` to derive
           ``causal_returns`` and ``causal_advantages`` from ``causal_values``.

        :param env: The training environment.
        :param callback: Callback to call at each step.
        :param rollout_buffer: Buffer to fill; must be a :class:`FutureRolloutBuffer`.
        :param n_rollout_steps: Number of steps to collect per environment.
        :returns: ``True`` if rollout collection succeeded.
        """
        assert isinstance(rollout_buffer, FutureRolloutBuffer)
        assert isinstance(self.policy, CausalActorCriticPolicy)
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not result:
            return False

        # Build future observation windows from the filled buffer.
        future_obs, future_mask = rollout_buffer._build_future_arrays()
        # future_obs.shape == (T, E, K, obs_dim)
        # future_mask.shape == (T, E, K)
        K = future_mask.shape[-1]

        T, E = rollout_buffer.buffer_size, rollout_buffer.n_envs
        obs_flat = th.as_tensor(rollout_buffer.observations.reshape(T * E, -1), dtype=th.float32, device=self.device)
        # obs_flat.shape == (T*E, obs_dim)
        future_obs_flat = th.as_tensor(future_obs.reshape(T * E, K, -1), dtype=th.float32, device=self.device)
        # future_obs_flat.shape == (T*E, K, obs_dim)
        future_mask_flat = th.as_tensor(future_mask.reshape(T * E, K), device=self.device)
        # future_mask_flat.shape == (T*E, K)

        with th.no_grad():
            causal_values_flat = self.policy.predict_causal_values(obs_flat, future_obs_flat, future_mask_flat)
        # causal_values_flat.shape == (T*E, 1)
        rollout_buffer.causal_values = causal_values_flat.cpu().numpy().reshape(T, E)
        # rollout_buffer.causal_values.shape == (T, E)

        # Bootstrap causal value for the state after the final step.
        last_obs = th.as_tensor(self._last_obs.reshape(E, -1), dtype=th.float32, device=self.device)
        # last_obs.shape == (E, obs_dim)
        last_future_obs = th.zeros(E, 0, obs_flat.shape[-1], dtype=th.float32, device=self.device)
        # last_future_obs.shape == (E, 0, obs_dim)
        last_future_mask = th.zeros(E, 0, dtype=th.bool, device=self.device)
        # last_future_mask.shape == (E, 0)
        with th.no_grad():
            # TODO: is last_causal_values as obtained with an empty last_future_obs a good value to use?
            # shouldn't we maybe just use normal values to begin with?
            # or maybe, we should use a single model for both...? but then it would be a bit hard to train appropriately
            last_causal_values = self.policy.predict_causal_values(last_obs, last_future_obs, last_future_mask)
        # last_causal_values.shape == (E, 1)

        rollout_buffer.compute_causal_returns_and_advantage(
            last_causal_values.squeeze(-1).cpu().numpy(),
            dones=self._last_episode_starts,
        )

        return True

    def train(self) -> None:
        assert isinstance(self.rollout_buffer, FutureRolloutBuffer)

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            assert isinstance(rollout_data, FutureRolloutBufferSamples)

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            # values.shape == (batch, 1)
            values = values.flatten()
            # values.shape == (batch,)

            causal_values = self.policy.predict_causal_values(rollout_data.observations, rollout_data.future_observations, rollout_data.future_mask)
            # causal_values.shape == (batch, 1)
            causal_values = causal_values.flatten()
            # causal_values.shape == (batch,)

            # NOTE: advantages based on causal baseline
            advantages = rollout_data.returns - rollout_data.causal_old_values
            # advantages.shape == (batch,)
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # advantages.shape == (batch,)

            policy_loss = -(advantages * log_prob).mean()

            value_loss = F.mse_loss(rollout_data.returns, values)

            causal_value_loss = F.mse_loss(rollout_data.causal_returns, causal_values)

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.causal_vf_coef * causal_value_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        causal_explained_var = explained_variance(self.rollout_buffer.causal_values.flatten(), self.rollout_buffer.causal_returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/causal_value_loss", causal_value_loss.item())
        self.logger.record("train/causal_explained_variance", causal_explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
