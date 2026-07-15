from abc import ABC, abstractmethod

import torch as th
from torch import nn

from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.torch_layers import create_mlp


class FutureModel(nn.Module, ABC):
    """
    Base class for modules that encode a future observation sequence into a
    fixed-size representation.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the encoded future representation.
    :param horizon: If set, only the first ``horizon`` future timesteps (nearest-first)
        are used; the remainder is discarded before encoding. ``None`` uses the entire future.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        horizon: int | None = None,
    ):
        super().__init__()
        self.mask_manager = mask_manager
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.horizon = horizon

    def _truncate(self, future_observations: th.Tensor, future_mask: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Restrict the future window to the first ``horizon`` timesteps (nearest-first).

        Future timesteps are ordered immediate-next-first, so slicing the leading
        ``horizon`` columns keeps the nearest future and drops the rest. A ``horizon``
        of ``None`` returns the inputs unchanged.

        :param future_observations: shape == (B, K, O)
        :param future_mask: shape == (B, K)
        :returns: Truncated ``(future_observations, future_mask)`` with K' = min(K, horizon).
        """
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)
        if self.horizon is None:
            return future_observations, future_mask
        future_observations = future_observations[:, : self.horizon]
        # future_observations.shape == (B, K', O)  where K' = min(K, horizon)
        future_mask = future_mask[:, : self.horizon]
        # future_mask.shape == (B, K')
        return future_observations, future_mask

    @abstractmethod
    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        """
        :param observation: Anchor observation. shape == (B, O)
        :param future_observations: shape == (B, K, O)
        :param future_mask: Temporal validity mask — True where future_observations contains a real
            timestep, False where it is padding. This is unrelated to causal masking; causal feature
            masking is applied internally. shape == (B, K)
        :returns: Encoded future representation. shape == (B, output_dim)
        """


class MeanFutureModel(FutureModel):
    """
    Encodes future observations by mean-pooling valid timesteps after zeroing
    out action-descendant features via the causal non-descendant mask, then
    projecting to ``output_dim`` with a linear layer and ReLU.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the encoded future representation.
    :param horizon: If set, only the first ``horizon`` future timesteps are used.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        horizon: int | None = None,
    ):
        super().__init__(mask_manager, obs_dim, output_dim, horizon=horizon)
        self.projection = nn.Sequential(nn.Linear(obs_dim, output_dim), nn.ReLU())

    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # observation.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask: boolean mask indicating which timesteps are real (not padding).
        #   True = valid timestep, False = padded. Has nothing to do with causal masking.
        # future_mask.shape == (B, K)
        future_observations, future_mask = self._truncate(future_observations, future_mask)
        # future_observations.shape == (B, K, O)  where K is now min(K, horizon)
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


class AttentionFutureModel(FutureModel):
    """
    Encodes future observations using multi-head self-attention over valid timesteps.

    Applies the per-timestep causal non-descendant mask to zero out action-descendant
    features, projects to ``d_model``, runs the sequence through a stack of Transformer
    encoder layers, then mean-pools over valid positions to produce a fixed-size vector.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the encoded future representation (= ``d_model``).
    :param n_heads: Number of attention heads.
    :param n_layers: Number of Transformer encoder layers.
    :param dim_feedforward: Inner dimension of the Transformer feed-forward sublayer.
    :param dropout: Dropout probability inside Transformer layers.
    :param horizon: If set, only the first ``horizon`` future timesteps are used.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        horizon: int | None = None,
    ):
        super().__init__(mask_manager, obs_dim, output_dim, horizon=horizon)
        self.projection = nn.Linear(obs_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # observation.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)
        future_observations, future_mask = self._truncate(future_observations, future_mask)
        # future_observations.shape == (B, K, O)  where K is now min(K, horizon)
        # future_mask.shape == (B, K)
        B, K, _ = future_observations.shape
        F = self.output_dim

        if K == 0:
            return th.zeros(B, F, device=future_observations.device)
            # return shape == (B, F)

        non_descendant_mask = self.mask_manager.non_descendant_mask(K)
        # non_descendant_mask.shape == (K, O)
        non_descendant_mask = th.as_tensor(non_descendant_mask, dtype=th.float32, device=future_observations.device)
        # non_descendant_mask.shape == (K, O)

        masked_obs = future_observations * non_descendant_mask.unsqueeze(0)
        # masked_obs.shape == (B, K, O)

        x = self.projection(masked_obs)
        # x.shape == (B, K, F)

        # PyTorch's src_key_padding_mask: True = position to be ignored (padding).
        src_key_padding_mask = ~future_mask
        # src_key_padding_mask.shape == (B, K)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # x.shape == (B, K, F)

        valid = future_mask.float().unsqueeze(-1)
        # valid.shape == (B, K, 1)
        future_enc = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        # future_enc.shape == (B, F)

        return future_enc


class EncoderDecoderFutureModel(FutureModel):
    """
    Encodes future observations using a full encoder-decoder transformer.

    Future observations (with action-descendant features zeroed out) are first
    processed by a self-attention encoder so timesteps can exchange information.
    The anchor ``observation`` is then projected to a single query token and
    cross-attends to the encoded future tokens via a transformer decoder.
    The single output token is returned as the fixed-size representation.

    Compared to ``CrossAttentionFutureModel``, this variant lets future timesteps
    interact with each other before the anchor queries them.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the output representation (= ``d_model``).
    :param n_heads: Number of attention heads.
    :param n_encoder_layers: Number of self-attention encoder layers over futures.
    :param n_decoder_layers: Number of cross-attention decoder layers.
    :param dim_feedforward: Inner dimension of the feed-forward sublayer.
    :param dropout: Dropout probability inside transformer layers.
    :param horizon: If set, only the first ``horizon`` future timesteps are used.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        horizon: int | None = None,
    ):
        super().__init__(mask_manager, obs_dim, output_dim, horizon=horizon)
        self.observation_projection = nn.Linear(obs_dim, output_dim)
        self.future_projection = nn.Linear(obs_dim, output_dim)
        # Learned per-feature sentinel substituted in place of action-descendant features,
        # so the network can distinguish "causally masked" from a genuine value of 0.
        self.mask_value = nn.Parameter(th.empty(obs_dim))
        nn.init.normal_(self.mask_value, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # observation.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)
        future_observations, future_mask = self._truncate(future_observations, future_mask)
        # future_observations.shape == (B, K, O)  where K is now min(K, horizon)
        # future_mask.shape == (B, K)
        B, K, _ = future_observations.shape
        F = self.output_dim

        if K == 0:
            return th.zeros(B, F, device=observation.device)
            # return shape == (B, F)

        non_descendant_mask = self.mask_manager.non_descendant_mask(K)
        # non_descendant_mask.shape == (K, O)
        non_descendant_mask = th.as_tensor(non_descendant_mask, dtype=th.float32, device=future_observations.device)
        # non_descendant_mask.shape == (K, O)

        nd_mask = non_descendant_mask.unsqueeze(0)
        # nd_mask.shape == (1, K, O)
        masked_future = future_observations * nd_mask + self.mask_value * (1.0 - nd_mask)
        # masked_future.shape == (B, K, O)

        kv_tokens = self.future_projection(masked_future)
        # kv_tokens.shape == (B, K, F)

        # PyTorch's key_padding_mask convention: True = position to be ignored.
        future_padding_mask = ~future_mask
        # future_padding_mask.shape == (B, K)

        future_tokens = self.encoder(kv_tokens, src_key_padding_mask=future_padding_mask)
        # future_tokens.shape == (B, K, F)

        query_tokens = self.observation_projection(observation).unsqueeze(1)
        # query_tokens.shape == (B, 1, F)

        output = self.decoder(query_tokens, future_tokens, memory_key_padding_mask=future_padding_mask)
        # output.shape == (B, 1, F)
        output = output.squeeze(1)
        # output.shape == (B, F)

        return output


class CrossAttentionFutureModel(FutureModel):
    """
    Encodes future observations via cross-attention, using ``observation`` as the query
    and causally-masked ``future_observations`` as keys and values.

    Built on a stack of ``nn.TransformerDecoderLayer`` modules: each layer applies
    self-attention on the (single-token) query, cross-attention to the future
    memory, and a feed-forward sublayer, with residual + layer-norm around each.
    The self-attention on a single-token query is functionally inert but comes
    for free with the standard primitive.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the output representation (= ``d_model``).
    :param n_heads: Number of attention heads.
    :param n_layers: Number of decoder layers.
    :param dim_feedforward: Inner dimension of the feed-forward sublayer.
    :param dropout: Dropout probability inside decoder layers.
    :param horizon: If set, only the first ``horizon`` future timesteps are used.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        n_heads: int = 4,
        n_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        horizon: int | None = None,
    ):
        super().__init__(mask_manager, obs_dim, output_dim, horizon=horizon)
        self.observation_projection = nn.Linear(obs_dim, output_dim)
        self.future_projection = nn.Linear(obs_dim, output_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # observation.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)
        future_observations, future_mask = self._truncate(future_observations, future_mask)
        # future_observations.shape == (B, K, O)  where K is now min(K, horizon)
        # future_mask.shape == (B, K)
        B, K, _ = future_observations.shape
        F = self.output_dim

        if K == 0:
            return th.zeros(B, F, device=observation.device)
            # return shape == (B, F)

        non_descendant_mask = self.mask_manager.non_descendant_mask(K)
        # non_descendant_mask.shape == (K, O)
        non_descendant_mask = th.as_tensor(non_descendant_mask, dtype=th.float32, device=future_observations.device)
        # non_descendant_mask.shape == (K, O)

        masked_future = future_observations * non_descendant_mask.unsqueeze(0)
        # masked_future.shape == (B, K, O)

        kv_tokens = self.future_projection(masked_future)
        # kv_tokens.shape == (B, K, F)

        query_tokens = self.observation_projection(observation).unsqueeze(1)
        # query_tokens.shape == (B, 1, F)

        # ``memory_key_padding_mask`` is PyTorch's name for the cross-attn padding mask;
        # True = position to be ignored. Here ``kv_tokens`` plays the role of the
        # decoder's "memory".
        future_padding_mask = ~future_mask
        # future_padding_mask.shape == (B, K)

        out = self.decoder(query_tokens, kv_tokens, memory_key_padding_mask=future_padding_mask)
        # out.shape == (B, 1, F)

        future_enc = out.squeeze(1)
        # future_enc.shape == (B, F)
        return future_enc


class MLPFutureModel(FutureModel):
    """
    Encodes a fixed-horizon future window with a plain MLP.

    Unlike the attention-based variants, this model requires a fixed ``horizon`` so the
    future window has a constant size and can be flattened into a single vector. The
    first ``horizon`` future observations (nearest-first) have their action-descendant
    features zeroed via the causal non-descendant mask; padded timesteps (when the real
    future is shorter than ``horizon``, near episode ends) are zeroed via ``future_mask``.
    The anchor ``observation`` is concatenated to the flattened window so the MLP can
    condition on the current state, mirroring how the attention variants use it as a query.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the encoded future representation.
    :param horizon: Number of future timesteps to encode. Required (unlike the base class).
    :param net_arch: Hidden layer sizes of the MLP.
    :param activation_fn: Activation function used between layers.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        horizon: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(mask_manager, obs_dim, output_dim, horizon=horizon)
        if net_arch is None:
            net_arch = [256, 256]
        # Input is the anchor observation concatenated with the flattened horizon window.
        input_dim = obs_dim + horizon * obs_dim
        self.mlp = nn.Sequential(*create_mlp(input_dim, output_dim, net_arch, activation_fn))

    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # observation.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)
        assert self.horizon is not None
        H = self.horizon
        B, _, O = future_observations.shape

        future_observations, future_mask = self._truncate(future_observations, future_mask)
        # future_observations.shape == (B, K', O)  where K' = min(K, horizon)
        # future_mask.shape == (B, K')
        K = future_observations.shape[1]

        if K < H:
            # Pad up to exactly ``horizon`` timesteps so the flattened input has fixed width.
            pad_obs = th.zeros(B, H - K, O, dtype=future_observations.dtype, device=future_observations.device)
            # pad_obs.shape == (B, H - K, O)
            future_observations = th.cat([future_observations, pad_obs], dim=1)
            # future_observations.shape == (B, H, O)
            pad_mask = th.zeros(B, H - K, dtype=th.bool, device=future_mask.device)
            # pad_mask.shape == (B, H - K)
            future_mask = th.cat([future_mask, pad_mask], dim=1)
            # future_mask.shape == (B, H)

        non_descendant_mask = self.mask_manager.non_descendant_mask(H)
        # non_descendant_mask.shape == (H, O)
        non_descendant_mask = th.as_tensor(non_descendant_mask, dtype=th.float32, device=future_observations.device)
        # non_descendant_mask.shape == (H, O)

        mask = non_descendant_mask.unsqueeze(0) * future_mask.float().unsqueeze(-1)
        # mask.shape == (B, H, O)  -- combines causal feature masking and temporal padding
        masked_future = future_observations * mask
        # masked_future.shape == (B, H, O)

        flat_future = masked_future.reshape(B, H * O)
        # flat_future.shape == (B, H * O)

        mlp_input = th.cat([observation, flat_future], dim=1)
        # mlp_input.shape == (B, O + H * O)

        future_enc = self.mlp(mlp_input)
        # future_enc.shape == (B, F)
        return future_enc
