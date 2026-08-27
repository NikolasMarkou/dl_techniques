"""Directory-local fixtures for `tests/test_layers/test_activations/`.

Hosts the ONE fixture this directory needs and its parent `tests/test_layers/conftest.py`
deliberately does not provide: a **genuine** float64 arm.

The parent conftest's `dtype_policy` / `mixed_float16_policy` fixtures set
`keras.mixed_precision.set_global_policy(...)` and nothing else. That is correct for the
mixed-precision policies, but it is NOT sufficient for float64: with the policy alone,
`keras.backend.floatx()` stays `'float32'`, so `keras.ops.convert_to_tensor` and
`keras.Input` keep producing float32 and the "float64" arm silently agrees with float32 to
eight digits. A precision guard written on top of that cannot fail. Guide rule L-38.

`float64_policy` therefore sets `floatx` IN ADDITION to the policy, and restores BOTH in a
`finally`. It lives here rather than being copy-pasted into the two modules that need it
(`test_the_dtype_floor_never_narrows.py`,
`test_the_gelu_constant_follows_the_input_dtype.py`), and it is NOT pushed up into
`tests/test_layers/conftest.py`, because `set_floatx` is process-global and several hundred
default-policy parametrizations in sibling directories have no reason to be exposed to an
extra teardown they never ask for.

**Every test using this fixture must additionally assert the REALISED input dtype.** The
fixture makes float64 reachable; only the assert proves it arrived. A related measured trap:
`set_floatx` does not re-point an already-materialised policy, so a layer built before the
fixture runs stays float32 -- build inside the test, never at module import.
"""

import keras
import pytest

# ---------------------------------------------------------------------


@pytest.fixture
def float64_policy():
    """Force a genuine float64 policy for one test, then ALWAYS restore both globals.

    Sets ``keras.backend.set_floatx('float64')`` as well as the global mixed-precision
    policy; see this module's docstring for why the policy alone is not enough.

    :yield: the literal string ``'float64'``.
    :rtype: str
    """
    previous_policy = keras.mixed_precision.global_policy().name
    previous_floatx = keras.backend.floatx()
    keras.backend.set_floatx("float64")
    keras.mixed_precision.set_global_policy("float64")
    try:
        yield "float64"
    finally:
        # Runs even if the test body raises. A leaked policy or a leaked `floatx`
        # poisons every subsequent test in the session.
        keras.mixed_precision.set_global_policy(previous_policy)
        keras.backend.set_floatx(previous_floatx)


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
