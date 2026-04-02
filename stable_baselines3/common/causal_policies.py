from abc import ABC, abstractmethod

import torch as th
from torch import nn

from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp


class FutureEncoder(nn.Module, ABC):
    """
    Base class for modules that encode a future observation sequence into a
    fixed-size representation.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the encoded future representation.
    """

    def __init__(self, mask_manager: CausalMaskManager, obs_dim: int, output_dim: int):
        super().__init__()
        self.mask_manager = mask_manager
        self.obs_dim = obs_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        """
        :param future_observations: shape == (B, K, O)
        :param future_mask: Temporal validity mask — True where future_observations contains a real
            timestep, False where it is padding. This is unrelated to causal masking; causal feature
            masking is applied internally. shape == (B, K)
        :returns: Encoded future representation. shape == (B, output_dim)
        """


class MaskedMeanFutureEncoder(FutureEncoder):
    """
    Encodes future observations by mean-pooling valid timesteps after zeroing
    out action-descendant features via the causal non-descendant mask, then
    projecting to ``output_dim`` with a linear layer and ReLU.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the encoded future representation.
    """

    def __init__(self, mask_manager: CausalMaskManager, obs_dim: int, output_dim: int):
        super().__init__(mask_manager, obs_dim, output_dim)
        self.projection = nn.Sequential(nn.Linear(obs_dim, output_dim), nn.ReLU())

    def forward(self, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # future_observations.shape == (B, K, O)
        # future_mask: boolean mask indicating which timesteps are real (not padding).
        #   True = valid timestep, False = padded. Has nothing to do with causal masking.
        # future_mask.shape == (B, K)
        B, K, _ = future_observations.shape
        F = self.output_dim

        if K == 0:
            return th.zeros(B, F, device=future_observations.device)
            # return shape == (B, F)

        non_descendant_mask = self.mask_manager.non_descendant_mask(K)
        # non_descendant_mask.shape == (K, O)
        non_descendant_mask = th.as_tensor(non_descendant_mask, dtype=th.float32, device=future_observations.device)
        # non_descendant_mask_t.shape == (K, O)

        mask = future_mask.float().unsqueeze(dim=-1) * non_descendant_mask.unsqueeze(dim=0)
        # mask.shape == (B, K, O)
        future_obs = (mask * future_observations).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        # future_obs.shape == (B, O)

        future_enc = self.projection(future_obs)
        # future_enc.shape == (B, F)
        return future_enc


class CausalActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy subclass that adds a causal critic alongside the standard
    value network.

    The causal critic receives the anchor observation and an encoded representation
    of future observations (with action-descendant features zeroed out), and outputs a
    scalar value estimate.  It shares no weights with the standard critic.

    :param mask_manager: Encodes the causal structure of the environment.
        Used to zero out action-descendant features in future observations.
    """

    def __init__(self, *args, mask_manager: CausalMaskManager, **kwargs):
        self.mask_manager = mask_manager
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        self._build_causal_critic()

    def _build_causal_critic(self) -> None:
        O = self.observation_space.shape[0]  # type: ignore[index]
        F = O
        # TODO: this is a placeholder architecture. Replace with the intended design
        #       (e.g. sequential model over future observations) once the pipeline is validated.
        self.future_encoder = MaskedMeanFutureEncoder(self.mask_manager, obs_dim=O, output_dim=F)
        modules = create_mlp(input_dim=O + F, output_dim=1, net_arch=[64, 64])
        self.causal_value_net = nn.Sequential(*modules)

    def predict_causal_values(
        self,
        obs: th.Tensor,
        future_observations: th.Tensor,
        future_mask: th.Tensor,
    ) -> th.Tensor:
        # obs.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)

        assert isinstance(obs, th.Tensor), f"predict_causal_values only supports plain tensor observations, got {type(obs)}"

        future_enc = self.future_encoder(future_observations, future_mask)
        # future_enc.shape == (B, F)
        x = th.cat([obs, future_enc], dim=-1)
        # x.shape == (B, O + F)
        values = self.causal_value_net(x)
        # values.shape == (B, 1)
        return values
