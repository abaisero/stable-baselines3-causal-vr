from typing import Any, cast

import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.causal_buffers import FutureRolloutBuffer, FutureRolloutBufferSamples
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo.ppo import PPO


class CausalPPO(PPO):
    """
    PPO subclass that uses :class:`FutureRolloutBuffer` and overrides ``train()``
    to additionally train a causal baseline using future observation sequences.

    The policy gradient update is identical to standard PPO.  Subclasses should
    override ``causal_baseline_loss()`` to implement the baseline objective.

    Accepts all the same arguments as :class:`PPO`.  If ``rollout_buffer_class``
    is provided it must be :class:`FutureRolloutBuffer` or a subclass thereof.
    """

    def __init__(self, *args, baseline_coef: float = 1.0, rollout_buffer_class: type[RolloutBuffer] | None = None, **kwargs):
        if rollout_buffer_class is None:
            rollout_buffer_class = cast(type[RolloutBuffer], FutureRolloutBuffer)

        assert issubclass(rollout_buffer_class, FutureRolloutBuffer), (
            f"CausalPPO requires a FutureRolloutBuffer, got {rollout_buffer_class}"
        )
        super().__init__(*args, rollout_buffer_class=rollout_buffer_class, **kwargs)
        self.baseline_coef = baseline_coef

    def causal_baseline_loss(self, rollout_data: FutureRolloutBufferSamples) -> th.Tensor:
        """
        Compute the causal baseline loss for a mini-batch.

        Override this in subclasses to implement the actual baseline objective.
        The default returns zero (no causal baseline).

        :param rollout_data: Mini-batch from :class:`FutureRolloutBuffer`.
        :returns: Scalar loss tensor.
        """
        return th.tensor(0.0, device=self.device)

    def causal_baseline_predictions(self, rollout_data: FutureRolloutBufferSamples) -> th.Tensor | None:
        """
        Return the causal baseline's value predictions for the given batch.

        Override in subclasses.  Return ``None`` if not applicable.

        :param rollout_data: Mini-batch from :class:`FutureRolloutBuffer`.
        :returns: Predicted values, shape == (batch,), or None.
        """
        return None

    def causal_baseline_targets(self, rollout_data: FutureRolloutBufferSamples) -> th.Tensor | None:
        """
        Return the learning targets for the causal baseline for the given batch.

        Override in subclasses.  Return ``None`` if not applicable.

        :param rollout_data: Mini-batch from :class:`FutureRolloutBuffer`.
        :returns: Target values, shape == (batch,), or None.
        """
        return None

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, baseline_losses = [], [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                assert isinstance(rollout_data, FutureRolloutBufferSamples)

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # values.shape == (batch,)

                advantages = rollout_data.advantages
                # advantages.shape == (batch,)
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # advantages.shape == (batch,)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # ratio.shape == (batch,)

                policy_loss_1 = advantages * ratio
                # policy_loss_1.shape == (batch,)
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss_2.shape == (batch,)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # values_pred.shape == (batch,)

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                baseline_loss = self.causal_baseline_loss(rollout_data)
                baseline_losses.append(baseline_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.baseline_coef * baseline_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Compute causal baseline explained variance over the full rollout.
        (full_batch,) = self.rollout_buffer.get(batch_size=None)
        assert isinstance(full_batch, FutureRolloutBufferSamples)
        baseline_preds = self.causal_baseline_predictions(full_batch)
        # baseline_preds.shape == (batch,), or None
        baseline_targets = self.causal_baseline_targets(full_batch)
        # baseline_targets.shape == (batch,), or None

        self.logger.record("train/entropy_loss", float(sum(entropy_losses) / len(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(sum(pg_losses) / len(pg_losses)))
        self.logger.record("train/value_loss", float(sum(value_losses) / len(value_losses)))
        self.logger.record("train/baseline_loss", float(sum(baseline_losses) / len(baseline_losses)))
        self.logger.record("train/approx_kl", float(sum(approx_kl_divs) / len(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(sum(clip_fractions) / len(clip_fractions)))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if baseline_preds is not None and baseline_targets is not None:
            baseline_explained_var = explained_variance(baseline_preds.cpu().numpy().flatten(), baseline_targets.cpu().numpy().flatten())
            self.logger.record("train/baseline_explained_variance", baseline_explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
