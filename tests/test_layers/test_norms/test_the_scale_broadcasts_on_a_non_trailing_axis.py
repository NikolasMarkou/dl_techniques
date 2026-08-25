"""
Guard for B2: the learnable ``scale`` must broadcast against a NON-trailing ``axis``.

``RMSNorm.build`` / ``ZeroCenteredRMSNorm.build`` create ``scale`` with
``shape = tuple(input_shape[i] for i in param_axes)`` — only the sizes at the
normalized axes, with no ``1`` inserted for the unnormalized ones. That shape is
only broadcast-compatible with the input when the normalized axes ARE the
trailing axes. For ``axis=1`` on ``(2, 5, 4)`` the weight is ``(5,)`` and the
multiply in ``call()`` cannot broadcast against ``(2, 5, 4)``.

``tests/test_layers/test_norms/test_rms_norm.py:303-314`` parametrizes over
non-trailing axes but constructs the layer with ``use_scale=False`` and swallows
any exception, which is exactly why this defect survived. This module always
uses ``use_scale=True`` and swallows nothing.

Expectations, stated per test so a "passes both ways" guard is impossible:

* ``test_the_forward_pass_preserves_shape_on_a_non_trailing_axis`` — RED at HEAD.
* ``test_the_output_matches_a_numpy_reference_on_a_non_trailing_axis`` — RED at HEAD.
* ``test_the_saved_model_round_trips_on_a_non_trailing_axis`` — RED at HEAD.
* ``test_the_built_scale_shape_on_the_trailing_axis_is_unchanged`` — GREEN at
  HEAD and must STAY green: it pins invariant I2 (no built weight shape may
  change, or every existing ``.keras`` checkpoint stops loading).
* ``test_the_trailing_axis_output_matches_a_numpy_reference`` — GREEN at HEAD and
  must stay green: the trailing-axis path is the one ~50 consumers use.

The reference is deliberately computed with a NON-uniform scale, so an
implementation that ignored the scale, or that broadcast it against the wrong
axis, produces a different answer than one that broadcasts it correctly.
"""

import os
from typing import Tuple, Type, Union

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm

EPSILON = 1e-6

# (layer class, axis, input shape) — every case has a NON-trailing normalized axis.
NON_TRAILING_CASES = [
    (RMSNorm, 1, (2, 5, 4)),
    (RMSNorm, (1, 2), (2, 5, 4, 3)),
    (ZeroCenteredRMSNorm, 1, (2, 5, 4)),
    (ZeroCenteredRMSNorm, (1, 2), (2, 5, 4, 3)),
]

NON_TRAILING_IDS = [
    "RMSNorm-axis1-2x5x4",
    "RMSNorm-axis1,2-2x5x4x3",
    "ZeroCenteredRMSNorm-axis1-2x5x4",
    "ZeroCenteredRMSNorm-axis1,2-2x5x4x3",
]


def _normalized_axes(axis: Union[int, Tuple[int, ...]], rank: int) -> Tuple[int, ...]:
    """Resolve ``axis`` to a tuple of non-negative axis indices."""
    axes = [axis] if isinstance(axis, int) else list(axis)
    return tuple(ax % rank for ax in axes)


def _scale_values(shape: Tuple[int, ...], seed: int = 7) -> np.ndarray:
    """A deliberately non-uniform, strictly positive scale."""
    rng = np.random.default_rng(seed)
    return (1.0 + rng.uniform(0.25, 1.75, size=shape)).astype("float32")


def _numpy_reference(
    layer_cls: Type[keras.layers.Layer],
    x: np.ndarray,
    axis: Union[int, Tuple[int, ...]],
    scale: np.ndarray,
) -> np.ndarray:
    """
    Hand-written reference: reduce over the SAME axes, then multiply by the scale
    broadcast to those axes.
    """
    x = x.astype("float32")
    axes = _normalized_axes(axis, x.ndim)
    reduce_axes = tuple(axes)

    if layer_cls is ZeroCenteredRMSNorm:
        x = x - np.mean(x, axis=reduce_axes, keepdims=True)

    mean_square = np.mean(np.square(x), axis=reduce_axes, keepdims=True)
    normalized = x / np.sqrt(mean_square + EPSILON)

    broadcast_shape = [1] * x.ndim
    for ax in axes:
        broadcast_shape[ax] = x.shape[ax]
    return (normalized * scale.reshape(broadcast_shape)).astype("float32")


def _build_layer_with_scale(
    layer_cls: Type[keras.layers.Layer],
    axis: Union[int, Tuple[int, ...]],
    shape: Tuple[int, ...],
) -> Tuple[keras.layers.Layer, np.ndarray]:
    """Build the layer on ``shape`` and install a non-uniform scale."""
    layer = layer_cls(axis=axis, epsilon=EPSILON, use_scale=True)
    layer.build(shape)
    values = _scale_values(tuple(layer.scale.shape))
    layer.scale.assign(values)
    return layer, values


def _sample(shape: Tuple[int, ...], seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype("float32")


@pytest.mark.parametrize("layer_cls, axis, shape", NON_TRAILING_CASES, ids=NON_TRAILING_IDS)
def test_the_forward_pass_preserves_shape_on_a_non_trailing_axis(
    layer_cls: Type[keras.layers.Layer],
    axis: Union[int, Tuple[int, ...]],
    shape: Tuple[int, ...],
) -> None:
    """(a) RED at HEAD: the scale multiply cannot broadcast, so ``call()`` raises."""
    layer, _ = _build_layer_with_scale(layer_cls, axis, shape)
    output = layer(keras.ops.convert_to_tensor(_sample(shape)))
    assert tuple(output.shape) == shape


@pytest.mark.parametrize("layer_cls, axis, shape", NON_TRAILING_CASES, ids=NON_TRAILING_IDS)
def test_the_output_matches_a_numpy_reference_on_a_non_trailing_axis(
    layer_cls: Type[keras.layers.Layer],
    axis: Union[int, Tuple[int, ...]],
    shape: Tuple[int, ...],
) -> None:
    """(b) RED at HEAD. The scale is non-uniform, so a dropped or mis-broadcast
    scale cannot accidentally reproduce the reference."""
    layer, scale = _build_layer_with_scale(layer_cls, axis, shape)
    x = _sample(shape)

    actual = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))
    expected = _numpy_reference(layer_cls, x, axis, scale)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("layer_cls, axis, shape", NON_TRAILING_CASES, ids=NON_TRAILING_IDS)
def test_the_saved_model_round_trips_on_a_non_trailing_axis(
    layer_cls: Type[keras.layers.Layer],
    axis: Union[int, Tuple[int, ...]],
    shape: Tuple[int, ...],
    tmp_path,
) -> None:
    """(c) RED at HEAD: the model cannot even be traced. Post-fix, save/load must
    reproduce the output exactly (``max|delta| == 0.0`` on CPU)."""
    inputs = keras.Input(shape=shape[1:])
    layer = layer_cls(axis=axis, epsilon=EPSILON, use_scale=True)
    model = keras.Model(inputs=inputs, outputs=layer(inputs))
    layer.scale.assign(_scale_values(tuple(layer.scale.shape)))

    x = _sample(shape, seed=11)
    before = keras.ops.convert_to_numpy(model(x))

    path = os.path.join(str(tmp_path), "non_trailing_axis.keras")
    model.save(path)
    restored = keras.models.load_model(path)
    after = keras.ops.convert_to_numpy(restored(x))

    assert float(np.max(np.abs(before - after))) == 0.0


@pytest.mark.parametrize("layer_cls", [RMSNorm, ZeroCenteredRMSNorm])
def test_the_built_scale_shape_on_the_trailing_axis_is_unchanged(
    layer_cls: Type[keras.layers.Layer],
) -> None:
    """
    (d) GREEN at HEAD and must STAY green — invariant I2.

    At ``axis=-1`` the built ``scale`` weight shape is recorded LITERALLY here.
    The B2 fix must reshape at CALL time only; if it ever changes what
    ``add_weight`` allocates, every ``.keras`` checkpoint holding one of these
    layers stops loading, silently and unrecoverably.
    """
    layer = layer_cls(axis=-1, epsilon=EPSILON, use_scale=True)
    layer.build((2, 5, 4))
    assert tuple(layer.scale.shape) == (4,)

    layer_4d = layer_cls(axis=-1, epsilon=EPSILON, use_scale=True)
    layer_4d.build((2, 5, 4, 3))
    assert tuple(layer_4d.scale.shape) == (3,)

    layer_multi = layer_cls(axis=(-2, -1), epsilon=EPSILON, use_scale=True)
    layer_multi.build((2, 5, 4, 3))
    assert tuple(layer_multi.scale.shape) == (4, 3)


@pytest.mark.parametrize("layer_cls", [RMSNorm, ZeroCenteredRMSNorm])
def test_the_trailing_axis_output_matches_a_numpy_reference(
    layer_cls: Type[keras.layers.Layer],
) -> None:
    """GREEN at HEAD and must stay green: the trailing-axis path is what every
    live consumer uses, and the B2 fix must leave it untouched (invariant I6)."""
    shape = (2, 5, 4)
    layer, scale = _build_layer_with_scale(layer_cls, -1, shape)
    x = _sample(shape, seed=23)

    actual = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x)))
    expected = _numpy_reference(layer_cls, x, -1, scale)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
