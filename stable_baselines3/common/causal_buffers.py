from collections.abc import Generator
from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples


class FutureRolloutBufferSamples(NamedTuple):
    """
    A batch of (anchor, future-sequence) pairs for sequential baseline training.

    Each item corresponds to one (timestep, env) anchor point.  The future
    observations are padded to ``max_future_len = buffer_size - 1`` and a boolean
    mask indicates which positions are valid (True = valid, False = padding or
    past an episode boundary).

    Contains all standard PPO fields so it can be used directly wherever
    ``RolloutBufferSamples`` is expected.
    """

    observations: th.Tensor
    """Anchor observations. shape == (batch, obs_dim)"""
    actions: th.Tensor
    """Anchor actions. shape == (batch, action_dim)"""
    future_observations: th.Tensor
    """Padded future observations. shape == (batch, max_future_len, obs_dim)"""
    future_mask: th.Tensor
    """True where future_observations contains real data. shape == (batch, max_future_len)"""
    old_values: th.Tensor
    """Critic value estimates at anchor. shape == (batch,)"""
    old_log_prob: th.Tensor
    """Log-prob of anchor action. shape == (batch,)"""
    advantages: th.Tensor
    """GAE advantages at anchor. shape == (batch,)"""
    returns: th.Tensor
    """Lambda-returns at anchor. shape == (batch,)"""
    causal_old_values: th.Tensor
    """Causal critic value estimates at anchor (frozen at collection time). shape == (batch,)"""
    causal_advantages: th.Tensor
    """GAE advantages computed using causal critic values. shape == (batch,)"""
    causal_returns: th.Tensor
    """Lambda-returns computed using causal critic values. shape == (batch,)"""

    def to_rollout_buffer_samples(self) -> RolloutBufferSamples:
        """Return the standard PPO fields as a :class:`RolloutBufferSamples`, dropping the future data."""
        return RolloutBufferSamples(
            observations=self.observations,
            actions=self.actions,
            old_values=self.old_values,
            old_log_prob=self.old_log_prob,
            advantages=self.advantages,
            returns=self.returns,
        )


class FutureRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer subclass whose ``get()`` yields :class:`FutureRolloutBufferSamples` instead of
    ``RolloutBufferSamples``.  Each mini-batch contains the standard PPO fields plus
    padded future observations and a validity mask, so a single training loop has
    access to both the anchor transition and the future trajectory.

    :param buffer_size: Max number of elements in the buffer (n_steps per env).
    :param observation_space: Observation space.
    :param action_space: Action space.
    :param device: PyTorch device.
    :param gae_lambda: GAE lambda.
    :param gamma: Discount factor.
    :param n_envs: Number of parallel environments.
    """

    # Precomputed future arrays, set at the start of get() before swap_and_flatten.
    _future_obs: np.ndarray
    _future_mask: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    causal_values: np.ndarray
    causal_advantages: np.ndarray
    causal_returns: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.causal_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # self.causal_values.shape == (T, E)
        self.causal_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # self.causal_advantages.shape == (T, E)
        self.causal_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # self.causal_returns.shape == (T, E)

    def compute_causal_returns_and_advantage(self, last_causal_values: np.ndarray, dones: np.ndarray) -> None:
        """
        Compute causal lambda-returns and GAE advantages using causal critic value predictions.

        Mirrors :meth:`RolloutBuffer.compute_returns_and_advantage` but uses
        ``self.causal_values`` for bootstrapping instead of ``self.values``.

        Must be called after ``self.causal_values`` has been filled (shape == (T, E)).

        :param last_causal_values: Causal critic value estimate for the state after the final
            rollout step. shape == (n_envs,)
        :param dones: Whether the final step was terminal. shape == (n_envs,)
        """
        # last_causal_values.shape == (N,)
        self.causal_advantages = np.zeros_like(self.advantages)
        # self.causal_advantages.shape == (T, E)

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                # next_non_terminal.shape == (N,)
                next_causal_values = last_causal_values
                # next_causal_values.shape == (N,)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                # next_non_terminal.shape == (N,)
                next_causal_values = self.causal_values[step + 1]
                # next_causal_values.shape == (N,)
            delta = self.rewards[step] + self.gamma * next_causal_values * next_non_terminal - self.causal_values[step]
            # delta.shape == (N,)
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            # last_gae_lam.shape == (N,)
            self.causal_advantages[step] = last_gae_lam

        self.causal_returns = self.causal_advantages + self.causal_values
        # self.causal_returns.shape == (T, E)

    def _build_future_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute padded future-observation arrays and masks for all transitions.

        Works in ``(T, E, ...)`` space so that each environment's trajectory is
        processed independently.  Future sequences never cross episode boundaries.

        A future sequence for anchor ``(t, e)`` contains observations at timesteps
        t+1, t+2, ..., up to (but not including) the next episode reset, or the
        buffer end.  Sequences are right-padded with zeros.

        Returns:
            future_obs:  shape == (T, E, max_future_len, O)
                where max_future_len is the longest actual future in this rollout.
            future_mask: shape == (T, E, max_future_len), dtype bool
        """
        T = self.buffer_size
        E = self.n_envs

        obs = self.observations
        # obs.shape == (T, E, O)
        episode_starts = self.episode_starts.astype(bool)
        # episode_starts.shape == (T, E)

        # Initialize: distance to end of buffer per env.
        future_lengths = np.tile(np.arange(T - 1, -1, -1).reshape(T, 1), (1, E))
        # future_lengths.shape == (T, E)

        # Process boundaries from last to first — each one overwrites earlier anchors.
        for t, e in np.argwhere(episode_starts)[::-1]:
            future_lengths[:t, e] = np.arange(t - 1, -1, -1)
        # future_lengths.shape == (T, E)

        max_future_len = int(future_lengths.max()) if future_lengths.max() > 0 else 0

        # offsets[k] = k, i.e. the relative step ahead from an anchor.
        # future_obs[t, e, k] = obs[t + k + 1, e], so offset k=0 is the next timestep t+1.
        offsets = np.arange(max_future_len).reshape(1, 1, max_future_len)
        # offsets.shape == (1, 1, max_future_len)

        # future_mask[t, e, k] = True iff offset k is within the valid future for anchor (t, e).
        future_mask = offsets < future_lengths.reshape(T, E, 1)
        # future_mask.shape == (T, E, max_future_len)

        # timestep_indices[t, k] = t + k + 1, the absolute timestep for anchor t at offset k.
        # Clipped to [0, T-1] to avoid out-of-bounds; only valid positions are written via future_mask.
        timestep_indices = np.clip(np.arange(T).reshape(T, 1, 1) + offsets + 1, 0, T - 1)
        # timestep_indices.shape == (T, 1, max_future_len)
        env_indices = np.arange(E).reshape(1, E, 1)
        # env_indices.shape == (1, E, 1)
        future_obs = np.zeros((T, E, max_future_len, *self.obs_shape), dtype=obs.dtype)
        # future_obs.shape == (T, E, max_future_len, O)
        future_obs[future_mask] = obs[timestep_indices, env_indices][future_mask]

        return future_obs, future_mask

    def get(self, batch_size: int | None = None) -> Generator[FutureRolloutBufferSamples, None, None]:  # type: ignore[override]
        """
        Yield mini-batches of :class:`FutureRolloutBufferSamples` covering all transitions.

        On the first call, builds the future observation arrays and flattens all buffer
        arrays via ``swap_and_flatten``.  Subsequent calls (additional
        epochs) reuse the already-flattened arrays.  Each epoch shuffles the transitions
        independently.

        :param batch_size: Mini-batch size.  Defaults to the full buffer (``n_steps * n_envs``).
        """
        assert self.full, ""

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        if not self.generator_ready:
            self._future_obs, self._future_mask = self._build_future_arrays()
            self._future_obs = self.swap_and_flatten(self._future_obs)
            self._future_mask = self.swap_and_flatten(self._future_mask)

            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns", "causal_values", "causal_advantages", "causal_returns"]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

            self.generator_ready = True

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: None = None,
    ) -> FutureRolloutBufferSamples:
        """
        Collect a mini-batch by selecting transitions at ``batch_inds`` from the flattened arrays.

        :param batch_inds: Flat indices into the ``(n_steps * n_envs,)`` dimension. shape == (batch,)
        """
        # batch_inds.shape == (batch,)
        return FutureRolloutBufferSamples(
            observations=self.to_torch(self.observations[batch_inds].astype(np.float32, copy=False)),
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            future_observations=self.to_torch(self._future_obs[batch_inds].astype(np.float32)),
            future_mask=self.to_torch(self._future_mask[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            causal_old_values=self.to_torch(self.causal_values[batch_inds].flatten()),
            causal_advantages=self.to_torch(self.causal_advantages[batch_inds].flatten()),
            causal_returns=self.to_torch(self.causal_returns[batch_inds].flatten()),
        )
