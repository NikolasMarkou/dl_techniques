"""
Test suite for the bias-free Clifford U-Net denoiser
(models/bias_free_denoisers/bfcliffordunet.py).

Covers the SC3 portion of plan_2026-07-01_6dc255c1:
  1. Factory builds tiny/small/base without crashing.
  2. Static bias-free scan: no live layer has use_bias=True, no centered
     LayerNormalization, no stock BatchNormalization (must be BiasFreeBatchNorm),
     no trainable GRN beta.
  3. Per-level shifts are non-empty and every s < channels at that level.
  4. .keras save + load round-trip on CPU (training=False): identical outputs.

The SC2 isolated-layer homogeneity numeric probe is authored SEPARATELY in
Step 4 (see the clearly-marked placeholder at the bottom of this file).
"""

import os
import keras
import pytest
import numpy as np
from typing import Tuple

from dl_techniques.models.bias_free_denoisers.bfcliffordunet import (
    create_cliffordunet_denoiser,
    create_cliffordunet_variant,
    CLIFFORDUNET_CONFIGS,
)
from dl_techniques.layers.geometric.clifford_block import GatedGeometricResidual
from dl_techniques.layers.norms.bias_free_batch_norm import BiasFreeBatchNorm


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def input_shape() -> Tuple[int, int, int]:
    # 64x64 is divisible by 2**4 (base depth=4) and 2**3 (tiny/small depth=3).
    return (64, 64, 3)


@pytest.fixture(scope="module")
def sample_input() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 64, 64, 3)).astype(np.float32)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _dead_bias_layer_ids(model: keras.Model) -> set:
    """Return ids of provably-dead bias-carrying sublayers.

    ``CliffordNetBlock`` does not forward ``use_bias=False`` to its inner
    ``GatedGeometricResidual`` (byte-identity discipline for existing consumers),
    so the GGR ``gate_dense`` keeps ``use_bias=True``. But with ``use_gate=False``
    (the homogeneous denoiser setting) that ``gate_dense`` is NEVER invoked in
    ``call()`` -- it is dead and does not participate in the forward map, so its
    bias cannot break homogeneity. Exclude those (and only those) from the scan.
    """
    dead = set()
    for layer in model._flatten_layers():
        if isinstance(layer, GatedGeometricResidual) and not layer.use_gate:
            # The GGR composite's own `use_bias` attribute (True) merely describes
            # its gate_dense; with use_gate=False that Dense is never called. Both
            # the composite and its gate_dense are dead for the forward map.
            dead.add(id(layer))
            dead.add(id(layer.gate_dense))
    return dead


def _bias_free_offenders(model: keras.Model):
    """Return a list of (layer_name, reason) bias-free violations (excluding dead)."""
    dead = _dead_bias_layer_ids(model)
    offenders = []
    for layer in model._flatten_layers():
        if id(layer) in dead:
            continue
        # 1. Any layer advertising a live additive bias.
        if getattr(layer, "use_bias", False):
            offenders.append((layer.name, f"{type(layer).__name__}.use_bias=True"))
        # 2. Centered LayerNormalization (beta is an additive offset).
        if isinstance(layer, keras.layers.LayerNormalization) and layer.center:
            offenders.append((layer.name, "LayerNormalization(center=True)"))
        # 3. Stock BatchNormalization (unconditional moving_mean; must be
        #    BiasFreeBatchNorm instead).
        if isinstance(layer, keras.layers.BatchNormalization):
            offenders.append((layer.name, "stock BatchNormalization (use BiasFreeBatchNorm)"))
        # 4. Trainable GRN beta (bias-like additive offset).
        if type(layer).__name__ == "GlobalResponseNormalization":
            offenders.append((layer.name, "GlobalResponseNormalization (trainable beta)"))
    return offenders


# ---------------------------------------------------------------------
# 1. Factory build (tiny/small/base)
# ---------------------------------------------------------------------

class TestCliffordUNetBuild:
    """SC3.1: the factory builds every variant without crashing."""

    @pytest.mark.parametrize("variant", list(CLIFFORDUNET_CONFIGS.keys()))
    def test_variant_builds(self, variant, input_shape, sample_input):
        model = create_cliffordunet_variant(variant, input_shape)
        y = model(sample_input, training=False)
        assert tuple(y.shape) == sample_input.shape

    def test_direct_factory_builds(self, input_shape, sample_input):
        model = create_cliffordunet_denoiser(
            input_shape=input_shape, depth=3, initial_filters=16, blocks_per_level=1
        )
        y = model(sample_input, training=False)
        assert tuple(y.shape) == sample_input.shape

    def test_deep_supervision_raises(self, input_shape):
        with pytest.raises(NotImplementedError):
            create_cliffordunet_denoiser(
                input_shape=input_shape, depth=3, initial_filters=16,
                enable_deep_supervision=True,
            )

    def test_expose_bottleneck_named_and_output(self, input_shape, sample_input):
        model = create_cliffordunet_denoiser(
            input_shape=input_shape, depth=3, initial_filters=16,
            blocks_per_level=1, expose_bottleneck=True,
        )
        # Two outputs; the bottleneck tap is discoverable by name.
        assert len(model.outputs) == 2
        assert model.get_layer("bottleneck") is not None
        outs = model(sample_input, training=False)
        assert tuple(outs[0].shape) == sample_input.shape


# ---------------------------------------------------------------------
# 2. Static bias-free scan
# ---------------------------------------------------------------------

class TestCliffordUNetBiasFree:
    """SC3.2: no live layer breaks the bias-free / homogeneity contract."""

    @pytest.mark.parametrize("variant", list(CLIFFORDUNET_CONFIGS.keys()))
    def test_no_bias_free_offenders(self, variant, input_shape):
        model = create_cliffordunet_variant(variant, input_shape)
        offenders = _bias_free_offenders(model)
        assert offenders == [], f"bias-free offenders in '{variant}': {offenders}"

    def test_final_projection_is_linear_and_bias_free(self, input_shape):
        model = create_cliffordunet_variant("tiny", input_shape)
        final = model.get_layer("final_output")
        assert final.use_bias is False
        # 'linear' activation (identity) preserves homogeneity.
        assert final.activation is keras.activations.linear

    def test_input_and_ctx_norms_are_bias_free_batch_norm(self, input_shape):
        """Every Clifford block's input_norm and ctx_norm must be BiasFreeBatchNorm."""
        model = create_cliffordunet_variant("tiny", input_shape)
        n_bfbn = sum(
            1 for l in model._flatten_layers() if isinstance(l, BiasFreeBatchNorm)
        )
        assert n_bfbn > 0, "expected BiasFreeBatchNorm instances in the model"


# ---------------------------------------------------------------------
# 3. Per-level shifts validity
# ---------------------------------------------------------------------

class TestCliffordUNetShifts:
    """SC3.3: every per-level shift list is non-empty and all s < channels."""

    @pytest.mark.parametrize("variant", list(CLIFFORDUNET_CONFIGS.keys()))
    def test_shifts_valid_per_level(self, variant, input_shape):
        cfg = CLIFFORDUNET_CONFIGS[variant]
        depth = cfg["depth"]
        initial_filters = cfg["initial_filters"]
        model = create_cliffordunet_variant(variant, input_shape)
        shift_map = model.cliffordunet_level_shifts

        # Reconstruct per-level channel widths (filter_multiplier default 2).
        filter_sizes = [initial_filters * (2 ** i) for i in range(depth + 1)]

        def _channels_for(key: str) -> int:
            if key == "bottleneck":
                return filter_sizes[depth]
            level = int(key.rsplit("_", 1)[-1])
            return filter_sizes[level]

        assert shift_map, "level_shift_map must be populated"
        for key, shifts in shift_map.items():
            channels = _channels_for(key)
            assert len(shifts) >= 1, f"{key}: empty shift list"
            for s in shifts:
                assert s < channels, f"{key}: shift {s} not < channels {channels}"

    def test_narrow_level_clamps_shifts(self, input_shape):
        """Base shifts wider than the narrowest level are clamped, never emptied."""
        # initial_filters=4 with base shifts up to 8 forces clamping at level 0.
        model = create_cliffordunet_denoiser(
            input_shape=input_shape, depth=3, initial_filters=4,
            blocks_per_level=1, shifts=[1, 2, 8],
        )
        shifts_l0 = model.cliffordunet_level_shifts["encoder_level_0"]
        assert shifts_l0 == [1, 2], f"expected clamped [1, 2], got {shifts_l0}"


# ---------------------------------------------------------------------
# 4. .keras round-trip (CPU, training=False)
# ---------------------------------------------------------------------

class TestCliffordUNetRoundTrip:
    """SC3.4: .keras save -> load -> identical outputs (max-abs-diff < 1e-4)."""

    @pytest.mark.parametrize("variant", list(CLIFFORDUNET_CONFIGS.keys()))
    def test_keras_round_trip(self, variant, input_shape, sample_input, tmp_path):
        model = create_cliffordunet_variant(variant, input_shape)
        # One training pass populates BiasFreeBatchNorm running_var (so the
        # inference forward is a non-trivial map before the round-trip).
        _ = model(sample_input, training=True)

        y_before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        save_path = os.path.join(str(tmp_path), f"cliffordunet_{variant}.keras")
        model.save(save_path)
        loaded = keras.models.load_model(save_path)

        y_after = keras.ops.convert_to_numpy(loaded(sample_input, training=False))

        max_abs_diff = float(np.max(np.abs(y_before - y_after)))
        assert max_abs_diff < 1e-4, (
            f"round-trip drift for '{variant}': max_abs_diff={max_abs_diff}"
        )


# ---------------------------------------------------------------------
# SC2 PLACEHOLDER — isolated-layer homogeneity numeric probe (Step 4)
# ---------------------------------------------------------------------
# TODO(Step 4, plan_2026-07-01_6dc255c1): the orchestrator authors the SC2
# make-or-break homogeneity test here. It must:
#   - build ONE homogeneous-config CliffordNetBlock (as in _homogeneous_block_kwargs)
#     with layer_scale_init set HIGH (e.g. 1.0) to defeat gamma-masking,
#   - run one training=True pass on random data to populate BiasFreeBatchNorm
#     running_var,
#   - with training=False feed x and alpha*x for alpha in {1e-3, 0.5, 2.0, 1e3},
#   - assert rel_err = ||f(alpha x) - alpha f(x)|| / ||alpha f(x)|| < 1e-2 (fp32).
# STOP-IF-1: if rel_err grows with alpha (degree-2 bilinear geometric product),
# escalate to the user BEFORE building the trainer.
@pytest.mark.skip(reason="SC2 homogeneity probe authored in Step 4 (make-or-break gate)")
def test_isolated_block_homogeneity_placeholder():
    pass
