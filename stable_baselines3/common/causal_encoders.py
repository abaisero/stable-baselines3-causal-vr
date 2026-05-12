from abc import ABC, abstractmethod

import torch as th
from torch import nn

from stable_baselines3.common.causal_utils import CausalMaskManager


class FutureModel(nn.Module, ABC):
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
    """

    def __init__(self, mask_manager: CausalMaskManager, obs_dim: int, output_dim: int):
        super().__init__(mask_manager, obs_dim, output_dim)
        self.projection = nn.Sequential(nn.Linear(obs_dim, output_dim), nn.ReLU())

    def forward(self, observation: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # observation.shape == (B, O)
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
    ):
        super().__init__(mask_manager, obs_dim, output_dim)
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
    ):
        super().__init__(mask_manager, obs_dim, output_dim)
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
    ):
        super().__init__(mask_manager, obs_dim, output_dim)
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
