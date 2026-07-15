"""Smoke tests for the future-encoder modules in causal_encoders.py.

These tests do not check learning behaviour — they verify that each encoder
runs end-to-end on representative inputs and returns finite tensors of the
expected shape, including under stress cases like fully-padded rows.
"""

import numpy as np
import pytest
import torch as th

from stable_baselines3.common.causal_encoders import (
    AttentionFutureModel,
    CrossAttentionFutureModel,
    EncoderDecoderFutureModel,
    FutureModel,
    MeanFutureModel,
    MLPFutureModel,
)
from stable_baselines3.common.causal_utils import CausalMaskManager

B, K, O, F = 4, 5, 17, 32
N_ACTIONS = 6
# Default horizon for fixed-horizon encoders in the shared tests: cover the full K-step
# window so the MLP sees the same span as the full-future encoders.
DEFAULT_HORIZON = K


def _build_encoder(
    encoder_cls: type[FutureModel], mask_manager: CausalMaskManager, *, horizon: int | None = None
) -> FutureModel:
    """Construct an encoder, supplying ``horizon`` only when the class requires/accepts it.

    ``MLPFutureModel`` has a *required* ``horizon`` (it flattens a fixed-size window),
    so it cannot be built without one; the others accept ``horizon=None`` (full future).
    """
    if encoder_cls is MLPFutureModel:
        return MLPFutureModel(
            mask_manager, obs_dim=O, output_dim=F, horizon=horizon if horizon is not None else DEFAULT_HORIZON
        )
    return encoder_cls(mask_manager, obs_dim=O, output_dim=F, horizon=horizon)


# Encoders that accept the full future (``horizon=None``); excludes MLP, which needs a fixed horizon.
FULL_FUTURE_ENCODERS = [MeanFutureModel, AttentionFutureModel, CrossAttentionFutureModel, EncoderDecoderFutureModel]
# All encoders, including the fixed-horizon MLP.
ALL_ENCODERS = [*FULL_FUTURE_ENCODERS, MLPFutureModel]


def _make_mask_manager(seed: int = 0) -> CausalMaskManager:
    rng = np.random.default_rng(seed)
    adjacency_as = rng.random((N_ACTIONS, O)) > 0.7
    # adjacency_as.shape == (N_ACTIONS, O)
    adjacency_ss = rng.random((O, O)) > 0.8
    # adjacency_ss.shape == (O, O)
    return CausalMaskManager(adjacency_as, adjacency_ss)


def _make_independent_mask_manager(seed: int = 0) -> CausalMaskManager:
    """Mask manager with no state->state propagation (``adjacency_ss = 0``).

    Without propagation the descendant set does not grow over time, so non-descendant
    features remain at every future timestep. The horizon tests use this so that the
    thing being varied is the *horizon* (temporal truncation), not causal feature masking
    — under the dense ``_make_mask_manager`` adjacency, far timesteps become fully
    descendant and would be zeroed regardless of horizon.
    """
    rng = np.random.default_rng(seed)
    adjacency_as = rng.random((N_ACTIONS, O)) > 0.7
    # adjacency_as.shape == (N_ACTIONS, O)
    adjacency_ss = np.zeros((O, O), dtype=bool)
    # adjacency_ss.shape == (O, O)
    return CausalMaskManager(adjacency_as, adjacency_ss)


def _make_inputs() -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    th.manual_seed(0)
    observation = th.randn(B, O)
    # observation.shape == (B, O)
    future_observations = th.randn(B, K, O)
    # future_observations.shape == (B, K, O)
    # Stress the padding logic across rows:
    #   row 0: all valid
    #   row 1: partial (first 2 valid)
    #   row 2: a single valid timestep
    #   row 3: fully padded (no valid timesteps)
    future_mask = th.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    # future_mask.shape == (B, K)
    return observation, future_observations, future_mask


@pytest.fixture
def mask_manager() -> CausalMaskManager:
    return _make_mask_manager()


@pytest.fixture
def independent_mask_manager() -> CausalMaskManager:
    return _make_independent_mask_manager()


@pytest.mark.parametrize("encoder_cls", ALL_ENCODERS)
def test_encoder_forward_shape_and_finite(encoder_cls: type[FutureModel], mask_manager: CausalMaskManager) -> None:
    encoder = _build_encoder(encoder_cls, mask_manager)
    observation, future_observations, future_mask = _make_inputs()

    out = encoder(observation, future_observations, future_mask)
    # out.shape == (B, F)

    assert out.shape == (B, F)
    assert th.isfinite(out).all(), f"{encoder_cls.__name__} produced non-finite values"


@pytest.mark.parametrize("encoder_cls", ALL_ENCODERS)
def test_encoder_handles_zero_horizon(encoder_cls: type[FutureModel], mask_manager: CausalMaskManager) -> None:
    encoder = _build_encoder(encoder_cls, mask_manager)
    observation = th.randn(B, O)
    # observation.shape == (B, O)
    future_observations = th.zeros(B, 0, O)
    # future_observations.shape == (B, 0, O)
    future_mask = th.zeros(B, 0, dtype=th.bool)
    # future_mask.shape == (B, 0)

    out = encoder(observation, future_observations, future_mask)
    # out.shape == (B, F)

    assert out.shape == (B, F)
    assert th.isfinite(out).all()


@pytest.mark.parametrize("encoder_cls", ALL_ENCODERS)
def test_causal_mask_zeroes_descendant_contribution(encoder_cls: type[FutureModel], mask_manager: CausalMaskManager) -> None:
    """Perturbing descendant features should not change the encoder output.

    This is the causal-safety invariant: the baseline must be a function only
    of non-descendant features at each future timestep.
    """
    encoder = _build_encoder(encoder_cls, mask_manager)
    encoder.eval()
    observation, future_observations, future_mask = _make_inputs()

    descendant_mask = th.as_tensor(mask_manager.descendant_mask(K), dtype=th.float32)
    # descendant_mask.shape == (K, O)

    with th.no_grad():
        out_a = encoder(observation, future_observations, future_mask)
        # out_a.shape == (B, F)

        perturbation = th.randn_like(future_observations) * descendant_mask.unsqueeze(0)
        # perturbation.shape == (B, K, O); nonzero only on descendant features
        future_observations_perturbed = future_observations + perturbation
        # future_observations_perturbed.shape == (B, K, O)

        out_b = encoder(observation, future_observations_perturbed, future_mask)
        # out_b.shape == (B, F)

    assert th.allclose(out_a, out_b, atol=1e-6), "encoder output depends on action-descendant features"


@pytest.mark.parametrize("encoder_cls", ALL_ENCODERS)
def test_horizon_ignores_future_beyond_window(
    encoder_cls: type[FutureModel], independent_mask_manager: CausalMaskManager
) -> None:
    """With ``horizon=h``, perturbing future steps at/after index ``h`` must not change the output.

    Uses ``independent_mask_manager`` so far timesteps still carry non-descendant features;
    under the dense graph they would be causally zeroed and the test would pass trivially.
    """
    h = 2
    encoder = _build_encoder(encoder_cls, independent_mask_manager, horizon=h)
    encoder.eval()
    observation, future_observations, future_mask = _make_inputs()

    with th.no_grad():
        out_a = encoder(observation, future_observations, future_mask)
        # out_a.shape == (B, F)

        perturbed = future_observations.clone()
        # perturbed.shape == (B, K, O)
        perturbed[:, h:] += th.randn_like(perturbed[:, h:])
        # perturbed[:, h:] is now off-window noise that the horizon should discard

        out_b = encoder(observation, perturbed, future_mask)
        # out_b.shape == (B, F)

    assert th.allclose(out_a, out_b, atol=1e-6), f"{encoder_cls.__name__} attends beyond its horizon"


def test_full_future_depends_on_far_steps(independent_mask_manager: CausalMaskManager) -> None:
    """Sanity guard: without a horizon the encoder *does* depend on far-future steps.

    Ensures ``test_horizon_ignores_future_beyond_window`` is meaningful rather than
    passing trivially (e.g. an encoder that ignores the future entirely). Uses the same
    ``independent_mask_manager`` so far steps are not causally zeroed.
    """
    encoder = _build_encoder(MeanFutureModel, independent_mask_manager, horizon=None)
    encoder.eval()
    observation, future_observations, future_mask = _make_inputs()

    with th.no_grad():
        out_a = encoder(observation, future_observations, future_mask)
        # out_a.shape == (B, F)

        perturbed = future_observations.clone()
        # perturbed.shape == (B, K, O)
        perturbed[:, 2:] += 5.0 * th.randn_like(perturbed[:, 2:])

        out_b = encoder(observation, perturbed, future_mask)
        # out_b.shape == (B, F)

    assert not th.allclose(out_a, out_b, atol=1e-6), "full-future encoder ignores far-future steps"


def test_mlp_horizon_exceeds_available_future(mask_manager: CausalMaskManager) -> None:
    """The MLP must pad up to its fixed width when the real future is shorter than ``horizon``."""
    h = K + 3
    encoder = _build_encoder(MLPFutureModel, mask_manager, horizon=h)
    observation, future_observations, future_mask = _make_inputs()
    # future_observations.shape == (B, K, O) with K < h, forcing padding

    out = encoder(observation, future_observations, future_mask)
    # out.shape == (B, F)

    assert out.shape == (B, F)
    assert th.isfinite(out).all()


def test_mlp_fixed_width_across_varying_future_lengths(mask_manager: CausalMaskManager) -> None:
    """A single MLP encoder handles inputs both longer and shorter than its horizon."""
    h = 3
    encoder = _build_encoder(MLPFutureModel, mask_manager, horizon=h)
    observation = th.randn(B, O)
    # observation.shape == (B, O)

    for k in (1, h, h + 4):
        future_observations = th.randn(B, k, O)
        # future_observations.shape == (B, k, O)
        future_mask = th.ones(B, k, dtype=th.bool)
        # future_mask.shape == (B, k)

        out = encoder(observation, future_observations, future_mask)
        # out.shape == (B, F)

        assert out.shape == (B, F), f"unexpected shape for future length k={k}"
        assert th.isfinite(out).all()
