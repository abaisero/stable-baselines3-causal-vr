import torch as th
from torch import nn

from stable_baselines3.common.causal_encoders import EncoderDecoderFutureModel, MLPFutureModel
from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


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
        F = 16
        H = 20  # hardcoded for now
        # self.future_encoder = EncoderDecoderFutureModel(self.mask_manager, obs_dim=O, output_dim=F)
        self.future_encoder = MLPFutureModel(self.mask_manager, obs_dim=O, output_dim=F, horizon=H, net_arch=[64, 64])
        self.causal_value_net = nn.Linear(O + F, 1)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # super() built self.optimizer over ALL params (including the causal critic). Split the causal
        # critic onto its own optimizer so its large/unstable gradients do not share the policy's
        # grad-norm clip budget. Same learning rate.
        #
        # Grab each side directly from its own modules, then assert the two are disjoint and together
        # cover every parameter -- this catches accidental weight sharing and any base module we forgot.
        # (pi_features_extractor is always the same object as features_extractor; log_std exists only
        # for continuous action spaces. Dedup by id since the feature extractor may be shared pi/vf.)
        self.causal_parameters = list(self.future_encoder.parameters()) + list(self.causal_value_net.parameters())

        base_modules = [
            self.features_extractor,
            self.vf_features_extractor,
            self.mlp_extractor,
            self.action_net,
            self.value_net,
        ]
        base_params = {id(p): p for m in base_modules for p in m.parameters()}
        if hasattr(self, "log_std"):
            base_params[id(self.log_std)] = self.log_std
        self.base_parameters = list(base_params.values())

        causal_ids = set(map(id, self.causal_parameters))
        base_ids = set(base_params)
        assert causal_ids.isdisjoint(base_ids), "causal critic shares parameters with the policy/standard critic"
        assert base_ids | causal_ids == set(map(id, self.parameters())), "param split does not cover all parameters"
        self.optimizer = self.optimizer_class(self.base_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
        self.causal_optimizer = self.optimizer_class(self.causal_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

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

    def param_counts(self) -> dict[str, int]:
        """Parameter counts per component: actor, standard critic, causal critic."""

        def count(*mods: nn.Module) -> int:
            return sum(p.numel() for m in mods for p in m.parameters())

        return {
            "n_params_actor": count(self.mlp_extractor.policy_net, self.action_net),
            "n_params_critic": count(self.mlp_extractor.value_net, self.value_net),
            "n_params_causal": sum(p.numel() for p in self.causal_parameters),
        }
