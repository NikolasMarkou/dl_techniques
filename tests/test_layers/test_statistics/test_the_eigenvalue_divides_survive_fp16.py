"""The kernel-PCA eigenvalue divides must survive a float16 compute dtype.

Guard for `plan-2026-08-31T134711-6271592d` step 8, defect class I-3. Two sites
whiten by dividing by ``sqrt(|eigenvalues| + eps)``:

- ``DeepKernelPCA.extract_components``      (called from ``call``)
- ``InvertibleKernelPCA.call``              (the ``whiten=True`` branch)

Both floored with a bare literal ``1e-10``. ``float16(1e-10)`` is exactly
``0.0``, and a zero eigenvalue is not hypothetical: both layers' ``adapt()``
deliberately CLAMP tiny or negative eigenvalues to exactly ``0.0`` (a
rank-deficient kernel matrix is the normal case for a low-rank input), so the
divide is ``x / sqrt(0)`` under ``mixed_float16``.

**A second, larger defect was found while writing this guard and is fixed in the
same commit**: neither layer could REACH its own floor under ``mixed_float16``.
``DeepKernelPCA`` raised ``cannot compute AddV2 ... expected to be a half tensor
but is a float tensor`` from ``ops.eye(batch_size)`` (which defaults to
``floatx()``), and ``InvertibleKernelPCA`` raised ``cannot compute Mul ...
expected to be a float tensor but is a half tensor`` from
``ops.sqrt(2.0 / n_random_features)``. Both layers were DEAD ON FORWARD PASS at
every float16 policy while 306 tests in this directory stayed green. The first
test class below pins that separately: an fp16 forward pass must not raise.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.statistics.deep_kernel_pca import DeepKernelPCA
from dl_techniques.layers.statistics.invertible_kernel_pca import (
    InvertibleKernelPCA,
)
from dl_techniques.utils.dtype_policy import stability_floor

BATCH = 4
FEATURES = 6
COMPONENTS = 3


class TestTheEigenvalueFloorHazardIsReal:
    """Anti-vacuity: the literal these sites used really is zero in float16."""

    def test_the_original_literal_is_exactly_zero_in_float16(self):
        assert np.float16(1e-10) == np.float16(0.0)
        assert np.float32(1e-10) > np.float32(0.0)

    def test_the_policy_floor_is_strictly_positive_in_float16(self):
        assert np.float16(stability_floor("float16", 1e-10)) > np.float16(0.0)
        # ...and moves nothing in the wide dtypes.
        assert stability_floor("float32", 1e-10) == 1e-10
        assert stability_floor("float64", 1e-10) == 1e-10


def _deep_kernel_pca(dtype_policy):
    layer = DeepKernelPCA(num_levels=1, components_per_level=[COMPONENTS])
    x = keras.ops.ones((BATCH, FEATURES), dtype=layer.compute_dtype)
    layer(x)
    return layer, x


def _invertible_kernel_pca(dtype_policy):
    layer = InvertibleKernelPCA(
        n_components=COMPONENTS, n_random_features=8, whiten=True
    )
    x = keras.ops.ones((BATCH, 5), dtype=layer.compute_dtype)
    layer(x)
    return layer, x


class TestTheLayersRunAtAllUnderTheComputeDtype:
    """The enabling defect: a float16 forward pass must not RAISE.

    This is deliberately separate from the floor tests. If it regresses, the
    floor tests below go from meaningful to unreachable, and an unreachable
    test is the failure mode this plan exists to remove.
    """

    def test_deep_kernel_pca_forward_does_not_raise(self, dtype_policy):
        layer, x = _deep_kernel_pca(dtype_policy)
        out = layer(x)
        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        if dtype_policy == "mixed_float16":
            assert layer.compute_dtype == "float16"

    def test_invertible_kernel_pca_forward_does_not_raise(self, dtype_policy):
        layer, x = _invertible_kernel_pca(dtype_policy)
        out = layer(x)
        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        if dtype_policy == "mixed_float16":
            assert layer.compute_dtype == "float16"


class TestTheWhiteningDivideSurvivesAZeroEigenvalue:
    """`adapt()` clamps a rank-deficient eigenvalue to exactly 0.0."""

    def test_deep_kernel_pca_tolerates_an_all_zero_eigenvalue_vector(
        self, dtype_policy
    ):
        """A forward-only pin. It does NOT discriminate on its own -- see below.

        MEASURED with the bare `1e-10` literal restored: `extract_components`
        really does return `inf` under `mixed_float16`, but `call()` then applies
        `ops.tanh`, and `tanh(inf)` is exactly `1.0`. The layer's OUTPUT is
        therefore finite while its internals are not, which is precisely why 306
        green tests in this directory were blind to the defect. The gradient test
        below is the arm that fails; this one is kept as a boundary pin.
        """
        layer, x = _deep_kernel_pca(dtype_policy)
        layer.eigenvalues[0].assign(np.zeros((COMPONENTS,), np.float32))

        out = layer(x)

        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        values = keras.ops.convert_to_numpy(out)
        assert np.all(np.isfinite(values)), (
            f"components / sqrt(|eigenvalues| + eps) went non-finite under "
            f"{dtype_policy}: {values}"
        )

    def test_deep_kernel_pca_is_differentiable_at_a_zero_eigenvalue(
        self, dtype_policy
    ):
        """`ops.tanh` hides the `inf` forward, but not backward.

        `d(tanh)/dx` at `tanh(inf) == 1.0` is `1 - 1**2 == 0.0`, and `0 * inf`
        is `NaN`, so the infinity the forward pass swallowed reappears in every
        input gradient.
        """
        layer, x = _deep_kernel_pca(dtype_policy)
        layer.eigenvalues[0].assign(np.zeros((COMPONENTS,), np.float32))
        source = tf.Variable(np.ones((BATCH, FEATURES), np.float32))

        with tf.GradientTape() as tape:
            tape.watch(source)
            out = layer(keras.ops.cast(source, layer.compute_dtype))
            loss = keras.ops.sum(keras.ops.cast(out, "float32"))
        grad = tape.gradient(loss, source)

        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        assert grad is not None
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad))), (
            f"d(components)/d(inputs) went non-finite under {dtype_policy}"
        )

    def test_invertible_kernel_pca_tolerates_a_rank_deficient_spectrum(
        self, dtype_policy
    ):
        layer, x = _invertible_kernel_pca(dtype_policy)
        # One surviving direction, the rest clamped to 0.0 exactly as adapt does.
        spectrum = np.zeros((COMPONENTS,), np.float32)
        spectrum[0] = 4.0
        layer.eigenvalues.assign(spectrum)

        out = layer(x)

        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype
        values = keras.ops.convert_to_numpy(out)
        assert np.all(np.isfinite(values)), (
            f"the whitening divide went non-finite under {dtype_policy}: "
            f"{values}"
        )

    def test_the_whitening_divide_is_differentiable_at_a_zero_eigenvalue(
        self, dtype_policy
    ):
        layer, x = _invertible_kernel_pca(dtype_policy)
        spectrum = np.zeros((COMPONENTS,), np.float32)
        spectrum[0] = 4.0
        layer.eigenvalues.assign(spectrum)

        with tf.GradientTape() as tape:
            out = layer(x)
            loss = keras.ops.sum(keras.ops.cast(out, "float32"))
        grads = tape.gradient(loss, layer.trainable_weights)

        assert grads, "the layer reported no trainable weights"
        for weight, grad in zip(layer.trainable_weights, grads):
            if grad is None:
                continue
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad))), (
                f"d(loss)/d({weight.path}) went non-finite under {dtype_policy}"
            )


class TestTheExplainedVarianceRatioIsNotAFloat16Site:
    """A REFUTED premise, pinned so it is not "fixed" later on a false ground.

    `deep_kernel_pca.py`'s `get_explained_variance_ratio` also divides by
    `total_variance + 1e-10`, and the plan listed it as an fp16 floor site. It is
    NOT one: the method is not `call()`, so Keras' autocast scope never applies,
    and the `eigenvalues` weight keeps its float32 VARIABLE dtype under
    `mixed_float16`. MEASURED: the ratio comes back `float32` even with the
    global policy set to `mixed_float16`. Leave the literal alone.
    """

    def test_the_ratio_is_computed_in_the_variable_dtype_not_the_compute_dtype(
        self, dtype_policy
    ):
        layer, _ = _deep_kernel_pca(dtype_policy)
        layer.eigenvalues[0].assign(np.zeros((COMPONENTS,), np.float32))

        ratios = layer.get_explained_variance_ratio()

        assert ratios[0].dtype == np.dtype(layer.variable_dtype)
        if dtype_policy == "mixed_float16":
            assert layer.compute_dtype == "float16"
            assert layer.variable_dtype == "float32"
            assert ratios[0].dtype == np.float32
        assert np.all(np.isfinite(ratios[0]))
