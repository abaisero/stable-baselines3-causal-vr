import torch as th
from torch import nn

from stable_baselines3.common.causal_encoders import MeanFutureEncoder
from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp


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
        self.future_encoder = MeanFutureEncoder(self.mask_manager, obs_dim=O, output_dim=F)
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

        future_enc = self.future_encoder(obs, future_observations, future_mask)
        # future_enc.shape == (B, F)
        x = th.cat([obs, future_enc], dim=-1)
        # x.shape == (B, O + F)
        values = self.causal_value_net(x)
        # values.shape == (B, 1)
        return values
