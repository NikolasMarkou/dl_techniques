"""R-038 root cause **RD-9**: ``KANvolution`` was not a KAN.

Plan ``plan-2026-08-22T035419-a11304c8``, ruling **D-052**.

The inventory recorded one dead weight -- ``control_points`` ``add_weight``-ed at
``layers/convolutional_kan.py:257`` and read nowhere in ``src/``. Measurement
found the defect is larger than that: the whole learnable-univariate-function
apparatus was decorative. The forward path was

    effective_kernel = self.w_spline + self.w_silu
    outputs = ops.conv(inputs, transpose(effective_kernel), ...)

which never called ``_compute_bspline_basis``, never read ``control_points`` and
never read ``grid``. **Measured at that revision, CPU, seeded:**

===========================================================  ==================
reading                                                      value
===========================================================  ==================
``max|KANvolution(x) - ops.conv(x, w_spline + w_silu)|``     ``0.0`` EXACTLY
``max|f(2x) - 2 f(x)|`` (degree-1 homogeneity)               ``0.0`` EXACTLY
``control_points`` movement after one ``SGD(lr=1.0)`` step   ``0.000000e+00``
``w_spline`` vs ``w_silu`` movement, same step               both ``1.221391e+00``
===========================================================  ==================

Zero homogeneity error means the layer carried **no non-linearity at all** -- not
the B-spline and not even the advertised SiLU -- and the identical movement of
``w_spline`` and ``w_silu`` shows they were a redundant reparameterization of a
single conv kernel. All 83 tests in ``test_convolutional_kan.py`` passed against
that implementation, which is why this file exists: those tests pin shapes,
finiteness and weight *existence*, and none of them pins what the layer computes.

Every test below fails against the pre-D-052 forward path.
"""

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.convolutional_kan import KANvolution


def _reference_kanvolution(x, layer):
    """A naive nested-loop transcription of ``K(x) = w_spline*B(x) + w_silu*SiLU(x)``.

    Deliberately written from the module docstring's equation, with explicit
    Python loops over every index, so it shares no code -- and therefore no
    transpose bug -- with the vectorized implementation under test. Only
    ``padding='valid'``, unit strides and unit dilation are covered; that is
    enough to pin the tap ordering, which is the part a reshape can silently
    get wrong.
    """
    control = ops.convert_to_numpy(layer.control_points)
    w_spline = ops.convert_to_numpy(layer.w_spline)
    w_silu = ops.convert_to_numpy(layer.w_silu)
    grid = ops.convert_to_numpy(layer.grid)
    spacing = 2.0 / layer.grid_size

    def basis(value):
        weights = np.maximum(0.0, 1.0 - np.abs(value - grid) / spacing)
        return weights / (weights.sum() + 1e-8)

    def silu(value):
        return value / (1.0 + np.exp(-value))

    kh, kw = layer.kernel_size
    batch, in_h, in_w, channels = x.shape
    out_h, out_w = in_h - kh + 1, in_w - kw + 1
    out = np.zeros((batch, out_h, out_w, layer.filters), dtype="float32")
    for b in range(batch):
        for oh in range(out_h):
            for ow in range(out_w):
                for f in range(layer.filters):
                    total = 0.0
                    for i in range(kh):
                        for j in range(kw):
                            for c in range(channels):
                                tap = np.tanh(x[b, oh + i, ow + j, c])
                                total += (
                                    w_spline[f, c, i, j] * float(basis(tap) @ control[f, c, i, j])
                                    + w_silu[f, c, i, j] * silu(tap)
                                )
                    out[b, oh, ow, f] = total
    return out


def test_the_layer_computes_the_equation_its_docstring_advertises():
    """Identity pin, not a shape pin, against an independent transcription.

    Uses a NON-SQUARE kernel (2x3) on a non-square input so that a swapped
    height/width transpose in the weight reshapes cannot pass.
    """
    keras.utils.set_random_seed(3)
    layer = KANvolution(
        filters=2, kernel_size=(2, 3), grid_size=4, padding="valid", use_bias=False
    )
    x = np.random.RandomState(0).randn(1, 4, 5, 3).astype("float32")
    actual = ops.convert_to_numpy(layer(x))
    expected = _reference_kanvolution(x, layer)

    assert actual.shape == expected.shape
    delta = float(np.max(np.abs(actual - expected)))
    scale = float(np.max(np.abs(expected)))
    assert scale > 1e-3, f"degenerate reference (scale {scale}); the pin is vacuous"
    assert delta < 1e-5, (
        f"KANvolution disagrees with a hand-written transcription of its own "
        f"equation by {delta:.6e} (reference scale {scale:.6e})"
    )


def test_the_layer_is_not_degree_one_homogeneous():
    """A plain convolution satisfies f(2x) == 2f(x) EXACTLY. A KAN must not.

    This is the single cheapest detector for the pre-D-052 forward path, which
    read 0.0 here.
    """
    keras.utils.set_random_seed(3)
    layer = KANvolution(
        filters=3, kernel_size=3, grid_size=8, padding="same", use_bias=False
    )
    x = keras.random.normal((2, 8, 8, 4), seed=1)
    once = ops.convert_to_numpy(layer(x))
    twice = ops.convert_to_numpy(layer(2.0 * x))
    gap = float(np.max(np.abs(twice - 2.0 * once)))
    assert gap > 1e-3, (
        f"|f(2x) - 2f(x)| = {gap:.6e}: the layer is (numerically) degree-1 "
        "homogeneous, i.e. it is an affine convolution and carries no learnable "
        "univariate function at all. This is exactly the pre-D-052 reading of 0.0."
    )


def test_control_points_move_after_one_real_optimizer_step():
    """Per LESSONS: assert gradient flow AFTER an optimizer step, never at init."""
    keras.utils.set_random_seed(5)
    model = keras.Sequential([
        keras.Input(shape=(8, 8, 4)),
        KANvolution(filters=3, kernel_size=3, grid_size=6, padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(2),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss="mse")
    before = {v.path: ops.convert_to_numpy(v).copy() for v in model.trainable_variables}
    model.fit(
        np.random.RandomState(0).rand(8, 8, 8, 4).astype("float32"),
        np.random.RandomState(1).rand(8, 2).astype("float32"),
        epochs=1, batch_size=8, verbose=0,
    )
    moved = {
        v.path: float(np.max(np.abs(ops.convert_to_numpy(v) - before[v.path])))
        for v in model.trainable_variables
    }
    control = [p for p in moved if p.endswith("control_points")]
    assert len(control) == 1, f"expected exactly one control_points weight, got {control}"
    assert moved[control[0]] > 0.0, (
        "control_points did not move under a real optimizer step -- it is not on "
        f"the forward path. Full movement report: {moved}"
    )
    assert all(d > 0.0 for d in moved.values()), (
        f"a trainable weight is dead: {[p for p, d in moved.items() if d == 0.0]}"
    )


def test_w_spline_and_w_silu_are_not_the_same_weight_twice():
    """They moved by the IDENTICAL 1.221391e+00 before D-052.

    Identical movement is the signature of `w_spline + w_silu` collapsing into
    one kernel, where both factors necessarily receive the same gradient.
    """
    keras.utils.set_random_seed(5)
    model = keras.Sequential([
        keras.Input(shape=(8, 8, 4)),
        KANvolution(filters=3, kernel_size=3, grid_size=6, padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(2),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss="mse")
    before = {v.path: ops.convert_to_numpy(v).copy() for v in model.trainable_variables}
    model.fit(
        np.random.RandomState(0).rand(8, 8, 8, 4).astype("float32"),
        np.random.RandomState(1).rand(8, 2).astype("float32"),
        epochs=1, batch_size=8, verbose=0,
    )
    deltas = {}
    for v in model.trainable_variables:
        if v.path.endswith(("w_spline", "w_silu")):
            deltas[v.path.rsplit("/", 1)[-1]] = (
                ops.convert_to_numpy(v) - before[v.path]
            )
    assert set(deltas) == {"w_spline", "w_silu"}, deltas.keys()
    identical = float(np.max(np.abs(deltas["w_spline"] - deltas["w_silu"])))
    assert identical > 1e-6, (
        "w_spline and w_silu received identical updates (max elementwise "
        f"difference {identical:.6e}), which means they are being summed into a "
        "single effective kernel rather than weighting two different functions."
    )


@pytest.mark.parametrize("grid_size", [2, 4, 16])
def test_the_grid_size_changes_the_function(grid_size):
    """``grid_size`` must not be a decorative constructor argument."""
    outputs = []
    for gs in (grid_size, grid_size * 2):
        keras.utils.set_random_seed(7)
        layer = KANvolution(
            filters=2, kernel_size=3, grid_size=gs, padding="same", use_bias=False
        )
        x = np.random.RandomState(0).randn(1, 6, 6, 3).astype("float32")
        outputs.append(ops.convert_to_numpy(layer(x)))
    gap = float(np.max(np.abs(outputs[0] - outputs[1])))
    assert gap > 1e-6, (
        f"grid_size {grid_size} and {grid_size * 2} produce the same output "
        f"(max delta {gap:.6e}); the spline grid is not consulted"
    )
