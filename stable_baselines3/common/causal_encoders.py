from abc import ABC, abstractmethod

import torch as th
from torch import nn

from stable_baselines3.common.causal_utils import CausalMaskManager


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
    def forward(self, obs: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        """
        :param obs: Anchor observation. shape == (B, O)
        :param future_observations: shape == (B, K, O)
        :param future_mask: Temporal validity mask — True where future_observations contains a real
            timestep, False where it is padding. This is unrelated to causal masking; causal feature
            masking is applied internally. shape == (B, K)
        :returns: Encoded future representation. shape == (B, output_dim)
        """


class MeanFutureEncoder(FutureEncoder):
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

    def forward(self, obs: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # obs.shape == (B, O)
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


class AttentionFutureEncoder(FutureEncoder):
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
        self.input_proj = nn.Linear(obs_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, obs: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # obs.shape == (B, O)
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

        x = self.input_proj(masked_obs)
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


class CrossAttentionFutureEncoder(FutureEncoder):
    """
    Encodes future observations via cross-attention, using ``obs`` as the query
    and causally-masked ``future_observations`` as keys and values.

    The anchor observation attends to the future sequence directly, producing a
    single output vector with no pooling required.  Multiple layers refine the
    query against the same fixed future memory.

    :param mask_manager: Provides the non-descendant mask for causal feature masking.
    :param obs_dim: Dimensionality of each observation.
    :param output_dim: Dimensionality of the output representation (= attention embed_dim).
    :param n_heads: Number of attention heads.
    :param n_layers: Number of cross-attention layers.
    """

    def __init__(
        self,
        mask_manager: CausalMaskManager,
        obs_dim: int,
        output_dim: int,
        *,
        n_heads: int = 4,
        n_layers: int = 1,
    ):
        super().__init__(mask_manager, obs_dim, output_dim)
        self.obs_proj = nn.Linear(obs_dim, output_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=output_dim, num_heads=n_heads, kdim=obs_dim, vdim=obs_dim, batch_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, obs: th.Tensor, future_observations: th.Tensor, future_mask: th.Tensor) -> th.Tensor:
        # obs.shape == (B, O)
        # future_observations.shape == (B, K, O)
        # future_mask.shape == (B, K)
        B, K, _ = future_observations.shape
        F = self.output_dim

        if K == 0:
            return th.zeros(B, F, device=obs.device)
            # return shape == (B, F)

        non_descendant_mask = self.mask_manager.non_descendant_mask(K)
        # non_descendant_mask.shape == (K, O)
        non_descendant_mask = th.as_tensor(non_descendant_mask, dtype=th.float32, device=future_observations.device)
        # non_descendant_mask.shape == (K, O)

        kv = future_observations * non_descendant_mask.unsqueeze(0)
        # kv.shape == (B, K, O)

        # PyTorch's key_padding_mask: True = position to be ignored (padding).
        key_padding_mask = ~future_mask
        # key_padding_mask.shape == (B, K)

        query = self.obs_proj(obs).unsqueeze(1)
        # query.shape == (B, 1, F)

        for attn in self.attn_layers:
            query, _ = attn(query, kv, kv, key_padding_mask=key_padding_mask)
            # query.shape == (B, 1, F)

        future_enc = query.squeeze(1)
        # future_enc.shape == (B, F)
        return future_enc
