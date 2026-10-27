"""Directory-local fixtures for `tests/test_layers/test_activations/`.

Hosts the one fixture this directory needs and the parent `tests/test_layers/conftest.py`
does not provide: `assert_xla_matches_eager`.

The genuine float64 arm (`float64_policy`: global policy AND `floatx`) is NOT defined here.
Its one owner is `tests/test_layers/conftest.py`, which every module in this directory
reaches by pytest's conftest lookup; the activation precision guards
(`test_the_dtype_floor_never_narrows.py`, `test_the_gelu_constant_follows_the_input_dtype.py`)
use it from there. Do not define a second `float64_policy` here: a directory-local fixture of
that name would silently shadow the shared one for this tree.
"""

import keras
import pytest

# ---------------------------------------------------------------------


@pytest.fixture
def assert_xla_matches_eager():
    """Return a checker: a layer's `call()` under XLA must agree with eager.

    Guide rule L-39 ("eager-only is not a fix"): the regime `model.fit()` actually
    runs is a traced `tf.function`, not eager, so a layer whose `call()` carries a
    Python loop, a dynamic shape read or a policy-dependent branch can be correct
    in the notebook and wrong (or un-lowerable) in training.

    **Interface contract.**

    :returns: a callable ``check(layer, x, atol, label) -> float``.

        * ``layer`` -- an ALREADY-CONSTRUCTED layer instance. The checker calls it
          eagerly first, so the weights are built once and the SAME weights are
          seen by both arms; passing a factory instead would compare two different
          random initializations and the reading would be meaningless.
        * ``x`` -- a NumPy array. Converted once and fed to both arms.
        * ``atol`` -- absolute tolerance, per call site, derived from a measured
          number rather than tuned (see each call site's comment).
        * ``label`` -- a string that appears in the failure message.

        The checker asserts (1) that XLA can LOWER the graph at all --
        ``jit_compile=True`` raises rather than silently falling back, so the
        `tf.function` call itself is the assertion -- (2) that both outputs are
        finite, and (3) that ``max|eager - xla| < atol``. It RETURNS the measured
        deviation so a caller can record it.

    :rtype: collections.abc.Callable

    **TF32 is reported, not silently absorbed.** On this repo's GPUs, eager and XLA
    do not use TensorFloat-32 identically, so a float32 matmul can differ between
    the two arms by far more than float32 epsilon while both are correct. MEASURED
    on `RoutingProbabilitiesLayer(output_dim=5, mode='deterministic')`, one fixed
    input: ``3.28e-04`` with TF32 on (GPU), ``5.96e-08`` with TF32 off (GPU),
    ``8.94e-08`` on CPU. The failure message therefore prints the live TF32 flag,
    because "XLA disagrees with eager" and "this matmul ran at TF32 precision in
    one of the two arms" look identical from the number alone.
    """
    import numpy as np
    import tensorflow as tf

    def _check(layer, x, atol: float, label: str) -> float:
        x_t = keras.ops.convert_to_tensor(np.asarray(x))

        # Eager FIRST: this builds the layer, so the traced arm below reuses the
        # very same weights instead of triggering a build inside the trace.
        eager = np.asarray(keras.ops.convert_to_numpy(layer(x_t)), dtype=np.float64)

        @tf.function(jit_compile=True)
        def _traced(t):
            return layer(t)

        # `jit_compile=True` RAISES if XLA cannot lower the graph -- there is no
        # silent fallback -- so this line is itself the "it compiles" assertion.
        xla = np.asarray(keras.ops.convert_to_numpy(_traced(x_t)), dtype=np.float64)

        assert np.all(np.isfinite(eager)), f"{label}: eager output is non-finite"
        assert np.all(np.isfinite(xla)), f"{label}: XLA output is non-finite"
        assert eager.shape == xla.shape, (
            f"{label}: XLA returned shape {xla.shape}, eager returned {eager.shape}"
        )

        dev = float(np.max(np.abs(eager - xla)))
        tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        assert dev < atol, (
            f"{label}: max|eager - xla| = {dev:.6e}, above atol {atol:.1e} "
            f"(TF32 enabled: {tf32}). A traced `jit_compile=True` graph is the "
            f"regime `fit()` runs in; a disagreement here is a real divergence "
            f"unless it is attributable to TF32 -- re-measure with "
            f"`tf.config.experimental.enable_tensor_float_32_execution(False)` "
            f"and on CPU before relaxing this bound."
        )
        return dev

    return _check
