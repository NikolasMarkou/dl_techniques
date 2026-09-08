"""Contract and numerical-parity suite for :class:`MultiDconvHeadTransposedAttention`.

The centrepiece is `_mdta_reference`, an independent NumPy transcription of
Restormer's PyTorch MDTA (`models/restormer_arch.py:99-132`). It was written
from the upstream spec -- the `[q, k, v]` chunk order, the einops pattern
`'b (head c) h w -> b head c (h w)'`, `F.normalize(..., dim=-1)`, the
`(heads, 1, 1)` temperature and the `(b, head, c, c)` affinity matrix -- and
NOT from the Keras layer, because a reference derived from the subject is a
second copy, not an oracle.

The reference carries three deliberate MUTATION KNOBS (`chunk_order`,
`head_factor`, `norm_axis`). Each is one of the orderings that shape,
serialization and gradient tests are all structurally blind to, and each has a
guard below asserting that the layer is FAR from the mutant while being equal
to the correct reference. Those two halves together are what makes the parity
test non-vacuous: agreeing with a reference is only evidence if disagreeing
with its neighbours is also demonstrated.

TF32 is scoped OFF for this module (`tests/test_layers/conftest.py`'s
`tf32_disabled`). The parity bound is expressed in `eps_f32 = 1.19e-07`; one
TF32 ulp is ~4100x larger, so on a TF32-capable GPU the bound would be
unattainable and every parity assertion here would be measuring the toggle
instead of the layer.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.multi_dconv_head_transposed_attention import (
    MultiDconvHeadTransposedAttention,
)
from tests.numerics import reassociation_atol
from tests.test_models.test_sam.dead_component_oracle import (
    fit_one_step_moved_variables,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

# ---------------------------------------------------------------------
# Fixed parity subject. `dim=8, num_heads=2` is the smallest configuration in
# which the head-factorisation mutation is OBSERVABLE: at `num_heads == 1` or
# `dim // num_heads == 1` the outer and inner factorisations coincide (the same
# degeneracy D-007 records for pixel-shuffle at `C' == 1`) and the guard would
# prove nothing.
# ---------------------------------------------------------------------

_DIM = 8
_NUM_HEADS = 2
_HEAD_DIM = _DIM // _NUM_HEADS
_HEIGHT = 4
_WIDTH = 5

# ---------------------------------------------------------------------
# NumPy reference
# ---------------------------------------------------------------------


def _conv1x1(x: np.ndarray, kernel: np.ndarray, bias) -> np.ndarray:
    """Apply a 1x1 convolution given a Keras `(1, 1, C_in, C_out)` kernel."""
    y = np.einsum("bhwi,io->bhwo", x, kernel[0, 0])
    return y if bias is None else y + bias


def _depthwise3x3(x: np.ndarray, kernel: np.ndarray, bias) -> np.ndarray:
    """Apply a fully depthwise 3x3 'same' convolution (Keras cross-correlation).

    `kernel` has the Keras `DepthwiseConv2D` shape `(3, 3, C, 1)` with
    `depth_multiplier=1`, which is the upstream `Conv2d(C, C, 3, groups=C)`:
    output channel `c` sees only input channel `c`.
    """
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros_like(x)
    for u in range(3):
        for v in range(3):
            out += (
                padded[:, u:u + x.shape[1], v:v + x.shape[2], :]
                * kernel[u, v, :, 0]
            )
    return out if bias is None else out + bias


def _softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _mdta_reference(
    x: np.ndarray,
    *,
    w_qkv: np.ndarray,
    b_qkv,
    w_dw: np.ndarray,
    b_dw,
    temperature: np.ndarray,
    w_proj: np.ndarray,
    b_proj,
    num_heads: int,
    chunk_order: str = "qkv",
    head_factor: str = "outer",
    norm_axis: str = "spatial",
) -> np.ndarray:
    """Independent NumPy MDTA, in float64, from the PyTorch spec.

    :param x: Input, shape `(B, H, W, C)`.
    :param chunk_order: `"qkv"` (correct, upstream `torch.chunk` order) or
        `"vkq"` (mutant).
    :param head_factor: `"outer"` (correct, einops `'b (head c) h w'`) or
        `"inner"` (mutant, `'b (c head) h w'`).
    :param norm_axis: `"spatial"` (correct, `F.normalize(..., dim=-1)` after
        the rearrange) or `"channel"` (mutant).
    :return: Output, shape `(B, H, W, C)`.
    """
    x = np.asarray(x, dtype=np.float64)
    batch, height, width, channels = x.shape
    head_dim = channels // num_heads

    qkv = _conv1x1(x, w_qkv, b_qkv)
    qkv = _depthwise3x3(qkv, w_dw, b_dw)

    parts = np.split(qkv, 3, axis=-1)
    if chunk_order == "qkv":
        q, k, v = parts
    elif chunk_order == "vkq":
        v, k, q = parts
    else:
        raise ValueError(f"unknown chunk_order {chunk_order!r}")

    def to_heads(t: np.ndarray) -> np.ndarray:
        flat = t.reshape(batch, height * width, channels)
        if head_factor == "outer":
            # channel index == head * head_dim + c
            return flat.reshape(
                batch, height * width, num_heads, head_dim
            ).transpose(0, 2, 3, 1)
        if head_factor == "inner":
            # channel index == c * num_heads + head
            return flat.reshape(
                batch, height * width, head_dim, num_heads
            ).transpose(0, 3, 2, 1)
        raise ValueError(f"unknown head_factor {head_factor!r}")

    def from_heads(t: np.ndarray) -> np.ndarray:
        if head_factor == "outer":
            flat = t.transpose(0, 3, 1, 2)
        else:
            flat = t.transpose(0, 3, 2, 1)
        return flat.reshape(batch, height, width, channels)

    def l2(t: np.ndarray) -> np.ndarray:
        # `t` is (B, heads, head_dim, H*W): axis -1 is the flattened SPATIAL
        # axis, axis 2 is the per-head CHANNEL axis.
        axis = -1 if norm_axis == "spatial" else 2
        norm = np.sqrt(np.sum(t * t, axis=axis, keepdims=True))
        return t / np.maximum(norm, 1e-12)

    q, k, v = to_heads(q), to_heads(k), to_heads(v)
    q, k = l2(q), l2(k)

    attn = np.matmul(q, np.swapaxes(k, -1, -2)) * temperature
    attn = _softmax_last(attn)
    out = from_heads(np.matmul(attn, v))
    return _conv1x1(out, w_proj, b_proj)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _randomized_layer(use_bias: bool, seed: int = 0):
    """Build an MDTA layer and give every weight a distinct random value.

    Weights are randomized (temperature included, away from its `ones`
    initializer) so that a defect cannot hide behind a symmetric or unit
    parameter. Values are read back through the same attribute accessors the
    reference is fed with, so no `set_weights` ordering is assumed.
    """
    rng = np.random.default_rng(seed)
    layer = MultiDconvHeadTransposedAttention(
        dim=_DIM, num_heads=_NUM_HEADS, use_bias=use_bias
    )
    layer.build((None, _HEIGHT, _WIDTH, _DIM))

    def fill(var, low=-0.5, high=0.5):
        var.assign(rng.uniform(low, high, size=var.shape).astype("float32"))

    fill(layer.qkv.kernel)
    fill(layer.qkv_dwconv.kernel)
    fill(layer.project_out.kernel)
    # A temperature away from 1.0, and positive, so the multiply is visible.
    fill(layer.temperature, 0.5, 1.5)
    if use_bias:
        fill(layer.qkv.bias)
        fill(layer.qkv_dwconv.bias)
        fill(layer.project_out.bias)
    return layer


def _reference_kwargs(layer) -> dict:
    """Read the layer's weights into the reference's keyword arguments."""
    np_ = keras.ops.convert_to_numpy
    return dict(
        w_qkv=np_(layer.qkv.kernel).astype(np.float64),
        b_qkv=np_(layer.qkv.bias).astype(np.float64) if layer.use_bias else None,
        w_dw=np_(layer.qkv_dwconv.kernel).astype(np.float64),
        b_dw=(
            np_(layer.qkv_dwconv.bias).astype(np.float64)
            if layer.use_bias
            else None
        ),
        temperature=np_(layer.temperature).astype(np.float64),
        w_proj=np_(layer.project_out.kernel).astype(np.float64),
        b_proj=(
            np_(layer.project_out.bias).astype(np.float64)
            if layer.use_bias
            else None
        ),
        num_heads=_NUM_HEADS,
    )


def _parity_atol(scale: float) -> float:
    """Derived float32 bound for the layer-vs-float64-reference comparison.

    Not a pasted `1e-6`. `tests.numerics.reassociation_atol` charges an
    8-sigma random walk of `u = eps_f32 / 2` over the rounded multiply-adds on
    the compared path; the contraction lengths below are exactly MDTA's, in
    call order:

    ==========================  ======================
    stage                       contraction length
    ==========================  ======================
    qkv 1x1 conv                `_DIM` (8)
    qkv depthwise 3x3           9
    L2 norm sum over H*W        `_HEIGHT * _WIDTH` (20)
    q @ k^T over H*W            `_HEIGHT * _WIDTH` (20)
    attn @ v over head_dim      `_HEAD_DIM` (4)
    project_out 1x1 conv        `_DIM` (8)
    ==========================  ======================

    The helper already charges BOTH sides of the comparison (`2 * num_steps`),
    which is conservative here: only one side (the layer) runs in float32, the
    reference being float64.
    """
    hw = _HEIGHT * _WIDTH
    return reassociation_atol(
        reduction_lengths=[_DIM, 9, hw, hw, _HEAD_DIM, _DIM],
        num_steps=1,
        scale=scale,
    )


def _fixed_input(seed: int = 7, batch: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(batch, _HEIGHT, _WIDTH, _DIM)).astype(
        "float32"
    )


# ---------------------------------------------------------------------
# 1. Numerical parity
# ---------------------------------------------------------------------


@pytest.mark.parametrize("use_bias", [False, True])
def test_matches_the_numpy_reference(use_bias):
    """The layer reproduces the independent NumPy MDTA at `rtol=0`."""
    layer = _randomized_layer(use_bias)
    x = _fixed_input()

    actual = keras.ops.convert_to_numpy(layer(x)).astype(np.float64)
    expected = _mdta_reference(x, **_reference_kwargs(layer))

    assert actual.shape == expected.shape == (2, _HEIGHT, _WIDTH, _DIM)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=_parity_atol(float(np.max(np.abs(expected)))),
    )


# ---------------------------------------------------------------------
# 2. Mutation guards -- the three orderings a shape test cannot see
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"chunk_order": "vkq"}, id="chunk_order_vkq"),
        pytest.param({"head_factor": "inner"}, id="head_as_inner_factor"),
        pytest.param({"norm_axis": "channel"}, id="l2_over_channels"),
    ],
)
def test_the_layer_is_far_from_the_mutant(mutation):
    """The layer must NOT match a reference built with the wrong ordering.

    Each mutation is a plausible port slip that leaves every shape, config and
    serialization assertion in this file green. The separation is required to
    exceed the parity bound by 100x, so "far" is measured against the same
    derived tolerance the parity test passes at, not against zero.
    """
    layer = _randomized_layer(use_bias=False)
    x = _fixed_input()

    actual = keras.ops.convert_to_numpy(layer(x)).astype(np.float64)
    kwargs = _reference_kwargs(layer)
    correct = _mdta_reference(x, **kwargs)
    mutant = _mdta_reference(x, **kwargs, **mutation)

    atol = _parity_atol(float(np.max(np.abs(correct))))

    # Anti-vacuity: the mutant must actually differ from the correct
    # reference, or this guard could pass on two identical arrays.
    assert np.max(np.abs(mutant - correct)) > 100.0 * atol, (
        f"mutation {mutation} did not change the reference; the guard would "
        "be vacuous"
    )
    assert np.max(np.abs(actual - mutant)) > 100.0 * atol, (
        f"the layer matches the {mutation} mutant, i.e. it implements the "
        "wrong ordering"
    )


# ---------------------------------------------------------------------
# 3. The depthwise stage really is `groups == 3 * dim`
# ---------------------------------------------------------------------


def test_the_qkv_dwconv_is_fully_depthwise():
    """`DepthwiseConv2D(3, depth_multiplier=1)` == upstream `groups=3*dim`.

    Two independent readings: the weight count (`3 * 3 * 3 * dim * 1`, i.e. no
    cross-channel term) and per-channel independence measured by perturbing one
    input channel and observing which outputs move.
    """
    layer = _randomized_layer(use_bias=False)
    kernel = keras.ops.convert_to_numpy(layer.qkv_dwconv.kernel)

    assert kernel.shape == (3, 3, 3 * _DIM, 1)
    assert layer.qkv_dwconv.count_params() == 3 * 3 * 3 * _DIM

    dw = layer.qkv_dwconv
    base = np.zeros((1, _HEIGHT, _WIDTH, 3 * _DIM), dtype="float32")
    y0 = keras.ops.convert_to_numpy(dw(base))
    for probe_channel in (0, 5, 3 * _DIM - 1):
        perturbed = base.copy()
        perturbed[0, 1, 1, probe_channel] = 1.0
        y1 = keras.ops.convert_to_numpy(dw(perturbed))
        moved = np.nonzero(np.abs(y1 - y0).max(axis=(0, 1, 2)) > 0.0)[0]
        assert moved.tolist() == [probe_channel], (
            f"perturbing input channel {probe_channel} moved output channels "
            f"{moved.tolist()}; the convolution is not fully depthwise"
        )


# ---------------------------------------------------------------------
# 4. Temperature contract
# ---------------------------------------------------------------------


def test_temperature_is_trainable_ones_of_the_right_shape():
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    layer.build((None, _HEIGHT, _WIDTH, _DIM))

    assert layer.temperature.shape == (_NUM_HEADS, 1, 1)
    assert layer.temperature.trainable
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(layer.temperature),
        np.ones((_NUM_HEADS, 1, 1), dtype="float32"),
    )
    assert any(v is layer.temperature for v in layer.trainable_variables)


def test_temperature_survives_a_save_load_round_trip(tmp_path):
    """Weight VALUES compare at `atol=0.0` BEFORE the loaded model's first call.

    Comparing after a call would let a re-initialized variable be masked by a
    lazy rebuild; comparing at `atol=0.0` is what makes this a byte-level
    check rather than a numerical one.
    """
    layer = _randomized_layer(use_bias=True, seed=3)
    inputs = keras.Input(shape=(_HEIGHT, _WIDTH, _DIM))
    model = keras.Model(inputs, layer(inputs))

    before = keras.ops.convert_to_numpy(layer.temperature)
    assert not np.allclose(before, 1.0), "randomization did not take"

    path = tmp_path / "mdta.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    restored_layer = next(
        sub
        for sub in loaded.layers
        if isinstance(sub, MultiDconvHeadTransposedAttention)
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(restored_layer.temperature),
        before,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------
# 5. Serialization round trip on OUTPUT VALUES
# ---------------------------------------------------------------------


def test_serialization_round_trip_preserves_outputs(tmp_path):
    layer = _randomized_layer(use_bias=True, seed=11)
    inputs = keras.Input(shape=(_HEIGHT, _WIDTH, _DIM))
    model = keras.Model(inputs, layer(inputs))

    x = _fixed_input(seed=13)
    original = keras.ops.convert_to_numpy(model(x, training=False))

    path = tmp_path / "mdta_model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    restored = keras.ops.convert_to_numpy(loaded(x, training=False))

    np.testing.assert_allclose(restored, original, rtol=0.0, atol=0.0)


def test_get_config_round_trips_every_constructor_argument():
    layer = MultiDconvHeadTransposedAttention(
        dim=16, num_heads=4, use_bias=True, name="mdta_cfg"
    )
    config = layer.get_config()
    assert config["dim"] == 16
    assert config["num_heads"] == 4
    assert config["use_bias"] is True

    clone = MultiDconvHeadTransposedAttention.from_config(config)
    assert (clone.dim, clone.num_heads, clone.use_bias) == (16, 4, True)


def test_compute_output_shape_matches_a_real_call():
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    declared = layer.compute_output_shape((None, _HEIGHT, _WIDTH, _DIM))
    actual = keras.ops.convert_to_numpy(layer(_fixed_input())).shape
    assert declared == (None, _HEIGHT, _WIDTH, _DIM)
    assert actual[1:] == declared[1:]


# ---------------------------------------------------------------------
# 6. Gradient flow after ONE real optimizer step
# ---------------------------------------------------------------------


def test_every_trainable_variable_moves_after_one_step():
    """Reuses the shared instrument; it reports moved variables BY NAME."""
    layer = MultiDconvHeadTransposedAttention(
        dim=_DIM, num_heads=_NUM_HEADS, use_bias=True
    )
    inputs = keras.Input(shape=(_HEIGHT, _WIDTH, _DIM))
    model = keras.Model(inputs, layer(inputs))
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")

    x = _fixed_input(seed=17, batch=4)
    y = _fixed_input(seed=19, batch=4)

    report = fit_one_step_moved_variables(model, x, y, batch_size=4)
    assert report.unmoved == (), (
        f"variables did not move under a real optimizer step: {report}"
    )
    assert any("temperature" in name for name in report.moved)


# ---------------------------------------------------------------------
# 7. Dynamic spatial extents
# ---------------------------------------------------------------------


def test_builds_with_unknown_spatial_dims_and_runs_at_two_sizes():
    """DocRes runs at arbitrary page sizes; `H`/`W` are unknown at build time."""
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    inputs = keras.Input(shape=(None, None, _DIM))
    model = keras.Model(inputs, layer(inputs))
    assert model.output_shape == (None, None, None, _DIM)

    rng = np.random.default_rng(23)
    for height, width in ((4, 5), (9, 3)):
        x = rng.uniform(-1.0, 1.0, size=(2, height, width, _DIM)).astype(
            "float32"
        )
        y = keras.ops.convert_to_numpy(model(x))
        assert y.shape == (2, height, width, _DIM)
        assert np.all(np.isfinite(y))


def test_the_dynamic_path_agrees_with_the_reference():
    """A dynamically-built layer computes the same values as the static one."""
    layer = _randomized_layer(use_bias=False, seed=29)
    # Rebuild-free: the same instance is called through a dynamic-shape graph.
    inputs = keras.Input(shape=(None, None, _DIM))
    model = keras.Model(inputs, layer(inputs))

    x = _fixed_input(seed=31)
    actual = keras.ops.convert_to_numpy(model(x)).astype(np.float64)
    expected = _mdta_reference(x, **_reference_kwargs(layer))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=_parity_atol(float(np.max(np.abs(expected)))),
    )


# ---------------------------------------------------------------------
# 8. Precision arms
# ---------------------------------------------------------------------


def test_output_is_finite_under_every_dtype_policy(dtype_policy):
    """`mixed_float16` and `float64` must both produce finite output.

    In float16 the `1e-12` normalisation floor of the PyTorch original rounds
    to exactly `0.0`; `stability_floor` lifts it to the smallest float16 normal
    so the division cannot produce `inf`/`NaN`.
    """
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    x = keras.ops.cast(_fixed_input(seed=37), layer.compute_dtype)
    y = layer(x)
    assert bool(keras.ops.all(keras.ops.isfinite(y))), (
        f"non-finite output under policy {dtype_policy}"
    )
    assert keras.ops.convert_to_numpy(y).shape == (2, _HEIGHT, _WIDTH, _DIM)


def test_an_all_zero_input_stays_finite():
    """The degenerate case the normalisation floor exists for: `||q|| == 0`."""
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    y = layer(np.zeros((1, _HEIGHT, _WIDTH, _DIM), dtype="float32"))
    assert bool(keras.ops.all(keras.ops.isfinite(y)))


# ---------------------------------------------------------------------
# 9. Configuration validation
# ---------------------------------------------------------------------


def test_indivisible_dim_raises_naming_both_values():
    with pytest.raises(ValueError, match=r"dim \(10\).*num_heads \(4\)"):
        MultiDconvHeadTransposedAttention(dim=10, num_heads=4)


@pytest.mark.parametrize(
    "dim,num_heads,match",
    [
        (0, 2, "dim must be positive"),
        (-8, 2, "dim must be positive"),
        (8, 0, "num_heads must be positive"),
        (8, -2, "num_heads must be positive"),
    ],
)
def test_non_positive_configuration_raises(dim, num_heads, match):
    with pytest.raises(ValueError, match=match):
        MultiDconvHeadTransposedAttention(dim=dim, num_heads=num_heads)


def test_a_channel_mismatch_is_caught_at_build():
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    with pytest.raises(ValueError, match="match"):
        layer.build((None, _HEIGHT, _WIDTH, _DIM + 1))


def test_a_non_rank_4_input_is_caught_at_build():
    layer = MultiDconvHeadTransposedAttention(dim=_DIM, num_heads=_NUM_HEADS)
    with pytest.raises(ValueError, match="4D input shape"):
        layer.build((None, _HEIGHT * _WIDTH, _DIM))
