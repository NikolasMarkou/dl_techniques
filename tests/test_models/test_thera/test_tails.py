"""Scoped tests for the THERA feature-refiner tails (step 6).

Covers air/plus/pro forward shapes, the E1 non-window-divisible non-square pro
input, ``.keras`` round-trip for each tail, and config + builder dispatch.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.thera.tails import (
    TheraTailAir,
    TheraTailPlus,
    TheraTailPro,
    build_thera_tail,
)


def _finite(t) -> bool:
    return bool(np.all(np.isfinite(keras.ops.convert_to_numpy(t))))


# ---------------------------------------------------------------------
# 1. air: identity passthrough
# ---------------------------------------------------------------------


def test_air_forward_is_passthrough():
    tail = TheraTailAir()
    x = keras.random.normal((2, 16, 16, 64))
    y = tail(x, training=False)  # 2nd positional arg accepted (THERA contract)
    assert tuple(y.shape) == (2, 16, 16, 64)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(y), keras.ops.convert_to_numpy(x)
    )


# ---------------------------------------------------------------------
# 2. plus: ConvNeXt refiner grows channels to last block dim (128)
# ---------------------------------------------------------------------


def test_plus_forward_shape_and_finite():
    tail = TheraTailPlus()  # THERA default: ...-> (128, 3) * 3
    x = keras.random.normal((2, 16, 16, 64))
    y = tail(x, training=False)
    assert tuple(y.shape) == (2, 16, 16, 128)
    assert _finite(y)


def test_plus_first_block_no_projection_when_channels_match():
    # Default first block dim is 64 == input channels -> no leading _Projection.
    tail = TheraTailPlus()
    x = keras.random.normal((1, 8, 8, 64))
    tail(x)  # build
    from dl_techniques.models.thera.tails import _Projection

    # 16 ConvNeXt blocks; projections only at the 64->96 and 96->128 boundaries.
    n_proj = sum(isinstance(layer, _Projection) for layer in tail._sublayers)
    assert n_proj == 2


# ---------------------------------------------------------------------
# 3. pro: SwinIR refiner -> num_feat channels (window-divisible input)
# ---------------------------------------------------------------------


def test_pro_forward_shape_window_divisible():
    # Smaller config for speed; embed_dim divisible by num_heads (60/3=20).
    tail = TheraTailPro(embed_dim=60, depths=(2, 2), num_heads=(3, 3), num_feat=64)
    x = keras.random.normal((2, 16, 16, 64))  # 16 = 2 * window_size(8)
    y = tail(x, training=False)
    assert tuple(y.shape) == (2, 16, 16, 64)
    assert _finite(y)


def test_pro_default_config_builds():
    # At least one test at the THERA-default config (embed_dim=180, depths=(7,6)).
    tail = TheraTailPro()
    x = keras.random.normal((1, 16, 16, 64))
    y = tail(x, training=False)
    assert tuple(y.shape) == (1, 16, 16, 64)
    assert _finite(y)
    n_params = int(sum(np.prod(w.shape) for w in tail.trainable_weights))
    assert n_params > 0


# ---------------------------------------------------------------------
# 4. E1: pro on NON-window-divisible, NON-square input -> exact H,W restored
# ---------------------------------------------------------------------


def test_pro_non_window_divisible_non_square():
    tail = TheraTailPro(embed_dim=60, depths=(2, 2), num_heads=(3, 3), num_feat=64)
    x = keras.random.normal((1, 20, 28, 64))  # neither 20 nor 28 divisible by 8
    y = tail(x, training=False)
    assert tuple(y.shape) == (1, 20, 28, 64)
    assert _finite(y)


# ---------------------------------------------------------------------
# 5. .keras round-trip for each tail
# ---------------------------------------------------------------------


def _roundtrip(tail, x, *, expect_params: bool, atol: float = 1e-4):
    inp = keras.Input(shape=x.shape[1:])
    out = tail(inp)
    model = keras.Model(inp, out)

    y_before = keras.ops.convert_to_numpy(model(x))

    n_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    if expect_params:
        assert n_params > 0

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tail.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)

    y_after = keras.ops.convert_to_numpy(reloaded(x))
    np.testing.assert_allclose(y_before, y_after, atol=atol, rtol=atol)


def test_air_roundtrip():
    _roundtrip(TheraTailAir(), keras.random.normal((2, 16, 16, 64)), expect_params=False)


def test_plus_roundtrip():
    _roundtrip(TheraTailPlus(), keras.random.normal((2, 16, 16, 64)), expect_params=True)


def test_pro_roundtrip():
    tail = TheraTailPro(embed_dim=60, depths=(2, 2), num_heads=(3, 3), num_feat=64)
    _roundtrip(tail, keras.random.normal((2, 16, 16, 64)), expect_params=True)


# ---------------------------------------------------------------------
# 6. get_config / from_config + builder dispatch
# ---------------------------------------------------------------------


def test_air_config_roundtrip():
    rebuilt = TheraTailAir.from_config(TheraTailAir().get_config())
    assert isinstance(rebuilt, TheraTailAir)


def test_plus_config_roundtrip():
    tail = TheraTailPlus()
    rebuilt = TheraTailPlus.from_config(tail.get_config())
    assert rebuilt.block_defs == tail.block_defs
    assert rebuilt.block_defs[-1][0] == 128


def test_pro_config_roundtrip():
    tail = TheraTailPro(embed_dim=60, depths=(2, 2), num_heads=(3, 3), num_feat=48)
    rebuilt = TheraTailPro.from_config(tail.get_config())
    assert rebuilt.embed_dim == 60
    assert rebuilt.depths == (2, 2)
    assert rebuilt.num_heads == (3, 3)
    assert rebuilt.num_feat == 48


def test_builder_dispatch_types():
    assert isinstance(build_thera_tail("air"), TheraTailAir)
    assert isinstance(build_thera_tail("plus"), TheraTailPlus)
    assert isinstance(build_thera_tail("pro"), TheraTailPro)


def test_builder_invalid_size_raises():
    with pytest.raises(ValueError):
        build_thera_tail("xyz")


def test_import_smoke():
    from dl_techniques.models.thera.tails import (  # noqa: F401
        build_thera_tail as _b,
        TheraTailAir as _A,
        TheraTailPlus as _P,
        TheraTailPro as _R,
    )


# ---------------------------------------------------------------------
# 7. iter-2 compliance: Pitfall-1 (sub-layers in __init__), compute_output_shape,
#    and pro dim validation.
# ---------------------------------------------------------------------


def test_tailplus_sublayers_created_in_init():
    # Pitfall 1 fixed: sub-layers must exist immediately after __init__,
    # BEFORE any build / forward pass.
    tail = TheraTailPlus()
    assert len(tail._sublayers) > 0


def test_tail_compute_output_shape():
    assert build_thera_tail("air").compute_output_shape((2, 16, 16, 64)) == (
        2,
        16,
        16,
        64,
    )
    assert build_thera_tail("plus").compute_output_shape((2, 16, 16, 64))[-1] == 128
    assert build_thera_tail("pro").compute_output_shape((2, 16, 16, 64))[-1] == 64


def test_tailpro_dim_validation():
    with pytest.raises(ValueError):
        TheraTailPro(embed_dim=0)
