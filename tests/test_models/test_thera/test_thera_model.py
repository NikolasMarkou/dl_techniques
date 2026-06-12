"""Scoped tests for the THERA model (step 8): Thera + build_thera + from_variant.

Covers SC-8 (the central gate): all six ``from_variant`` configs instantiate,
forward, and ``.keras`` round-trip with a PER-WEIGHT comparison.

Speed note: the full-size ``plus``/``pro`` variants use ``hidden_dim=512`` and
heavy tails (512-dim SwinIR). The full-size 6 variants are exercised at
INSTANTIATION only (test 1). Forward + ``.keras`` round-trip (tests 2/3) use six
SMALL hand-built configs (``hidden_dim=16``, tiny backbones) so the suite stays
fast while still exercising every (backbone x tail) pair end-to-end.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.grid_sample import make_grid
from dl_techniques.models.thera import (
    Thera,
    build_thera,
    EDSRBackbone,
    RDNBackbone,
    build_thera_tail,
)

# ---------------------------------------------------------------------
# constants / helpers
# ---------------------------------------------------------------------

_VARIANTS = [
    "edsr-air",
    "edsr-plus",
    "edsr-pro",
    "rdn-air",
    "rdn-plus",
    "rdn-pro",
]

# (backbone-key, size) pairs for the small forward / round-trip configs.
_SMALL_CONFIGS = [
    ("edsr", "air"),
    ("edsr", "plus"),
    ("edsr", "pro"),
    ("rdn", "air"),
    ("rdn", "plus"),
    ("rdn", "pro"),
]


def _small_backbone(kind: str) -> keras.layers.Layer:
    """A tiny backbone (few feats/blocks) to keep the test fast."""
    if kind == "edsr":
        return EDSRBackbone(num_feats=32, num_blocks=2)
    return RDNBackbone(growth_rate_0=16, config="B")


def _small_thera(kind: str, size: str, out_dim: int = 3) -> Thera:
    """A small Thera (hidden_dim=16, tiny backbone) for fast forward/round-trip."""
    return Thera(
        hidden_dim=16,
        out_dim=out_dim,
        backbone=_small_backbone(kind),
        tail=build_thera_tail(size),
    )


def _coords(batch: int, hq: int, wq: int) -> tf.Tensor:
    """Broadcast a pixel-center query grid to ``(batch, hq, wq, 2)``."""
    grid = keras.ops.convert_to_tensor(make_grid((hq, wq)))  # (hq, wq, 2)
    return keras.ops.broadcast_to(grid[None, ...], (batch, hq, wq, 2))


def _inputs(batch: int = 2, hs: int = 16, ws: int = 16, hq: int = 24, wq: int = 24):
    source = keras.random.normal((batch, hs, ws, 3))
    coords = _coords(batch, hq, wq)
    t = keras.ops.ones((batch, 1))
    return source, coords, t


# ---------------------------------------------------------------------
# Test 1: all six full-size variants instantiate via from_variant
# ---------------------------------------------------------------------


class TestVariantInstantiation:
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_from_variant_instantiates(self, variant: str) -> None:
        model = Thera.from_variant(variant)
        assert isinstance(model, Thera)

        backbone_key, size = Thera.MODEL_VARIANTS[variant]
        # Correct backbone type.
        if backbone_key == "edsr-baseline":
            assert isinstance(model.backbone, EDSRBackbone)
        else:
            assert isinstance(model.backbone, RDNBackbone)

        # Correct hidden_dim per size (32 for air, 512 otherwise).
        expected_hidden = 32 if size == "air" else 512
        assert model.hidden_dim == expected_hidden

        # Tail type matches the size key.
        assert size in type(model.tail).__name__.lower()

    def test_all_six_variants_distinct(self) -> None:
        # INV-8: six genuinely distinct (backbone, size) configs.
        assert len(Thera.MODEL_VARIANTS) == 6
        assert len(set(Thera.MODEL_VARIANTS.values())) == 6


# ---------------------------------------------------------------------
# Test 2: small forward for all six (backbone x size) configs
# ---------------------------------------------------------------------


class TestForward:
    @pytest.mark.parametrize("kind,size", _SMALL_CONFIGS)
    def test_forward_shape_and_finite(self, kind: str, size: str) -> None:
        model = _small_thera(kind, size)
        source, coords, t = _inputs()  # 16x16 -> 24x24 (arbitrary upscale)
        out = model((source, coords, t))
        assert tuple(out.shape) == (2, 24, 24, 3)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


# ---------------------------------------------------------------------
# Test 3: .keras round-trip (SC-8) for all six small configs, per-weight
# ---------------------------------------------------------------------


class TestKerasRoundTrip:
    @pytest.mark.parametrize("kind,size", _SMALL_CONFIGS)
    def test_keras_roundtrip_per_weight(self, kind: str, size: str) -> None:
        model = _small_thera(kind, size)
        source, coords, t = _inputs()

        # Dummy forward to fully build before saving (LESSONS.md).
        out_before = model((source, coords, t))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "thera.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        # --- PER-WEIGHT parity (LESSONS iter-1: nested-list weights can silently
        # fail to restore; assert EVERY variable matches by VALUE, not just shape).
        #
        # We compare in WEIGHT ORDER, not by path. Keras builds the original and
        # the reloaded model by walking the SAME layer tree in the SAME sequence,
        # so model.weights and reloaded.weights are aligned index-for-index. We do
        # NOT match by `w.path`: the Swin attention block contains an inner
        # auto-named `single_window_attention_N` sublayer whose global name counter
        # advances on every fresh instance (orig -> ..._12, reload -> ..._25), so
        # absolute paths legitimately differ while the WEIGHTS are identical. The
        # order-aligned value check is the true reload oracle (corroborated by the
        # forward-output 1e-4 parity assertion below). ---
        orig_weights = model.weights
        new_weights = reloaded.weights
        assert len(orig_weights) == len(new_weights), (
            f"weight count mismatch: {len(orig_weights)} vs {len(new_weights)}"
        )
        for i, (wa, wb) in enumerate(zip(orig_weights, new_weights)):
            a = keras.ops.convert_to_numpy(wa)
            b = keras.ops.convert_to_numpy(wb)
            assert a.shape == b.shape, (
                f"shape mismatch at weight {i}: {wa.path} {a.shape} vs "
                f"{wb.path} {b.shape}"
            )
            np.testing.assert_allclose(
                a, b, atol=1e-6,
                err_msg=f"weight value mismatch at {i}: {wa.path} vs {wb.path}",
            )

        # --- Identical forward within tol. ---
        out_after = reloaded((source, coords, t))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out_before),
            keras.ops.convert_to_numpy(out_after),
            atol=1e-4,
        )


# ---------------------------------------------------------------------
# Test 4: get_config / from_config round-trip reproduces the variant
# ---------------------------------------------------------------------


class TestConfigRoundTrip:
    @pytest.mark.parametrize("kind,size", _SMALL_CONFIGS)
    def test_get_from_config(self, kind: str, size: str) -> None:
        model = _small_thera(kind, size, out_dim=3)
        config = model.get_config()
        rebuilt = Thera.from_config(config)

        assert rebuilt.hidden_dim == model.hidden_dim
        assert rebuilt.out_dim == model.out_dim
        assert rebuilt.k_init == pytest.approx(model.k_init)
        assert rebuilt.components_init_scale == pytest.approx(
            model.components_init_scale
        )
        assert type(rebuilt.backbone) is type(model.backbone)
        assert type(rebuilt.tail) is type(model.tail)

    def test_full_variant_config_roundtrip(self) -> None:
        # Full-size variant get_config/from_config (no forward -> cheap).
        model = Thera.from_variant("rdn-plus")
        rebuilt = Thera.from_config(model.get_config())
        assert rebuilt.hidden_dim == 512
        assert isinstance(rebuilt.backbone, RDNBackbone)


# ---------------------------------------------------------------------
# Test 5: ValueError on bad backbone / size / variant
# ---------------------------------------------------------------------


class TestValidation:
    def test_build_thera_bad_backbone(self) -> None:
        with pytest.raises(ValueError):
            build_thera(backbone="nope", size="air")

    def test_build_thera_bad_size(self) -> None:
        with pytest.raises(ValueError):
            build_thera(backbone="edsr-baseline", size="ultra")

    def test_from_variant_bad_name(self) -> None:
        with pytest.raises(ValueError):
            Thera.from_variant("edsr-mega")

    def test_ctor_bad_dims(self) -> None:
        with pytest.raises(ValueError):
            Thera(
                hidden_dim=0,
                out_dim=3,
                backbone=_small_backbone("edsr"),
                tail=build_thera_tail("air"),
            )


# ---------------------------------------------------------------------
# Test 6: arbitrary-scale -- same model, two different coord grids
# ---------------------------------------------------------------------


class TestArbitraryScale:
    def test_two_scales_same_model(self) -> None:
        model = _small_thera("edsr", "air")
        source = keras.random.normal((2, 16, 16, 3))
        t = keras.ops.ones((2, 1))

        # 24x24 grid.
        out_a = model((source, _coords(2, 24, 24), t))
        assert tuple(out_a.shape) == (2, 24, 24, 3)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out_a)))

        # 31x37 non-square grid (proves continuous-scale decode).
        out_b = model((source, _coords(2, 31, 37), t))
        assert tuple(out_b.shape) == (2, 31, 37, 3)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out_b)))
