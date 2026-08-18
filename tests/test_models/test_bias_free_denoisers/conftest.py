"""Shared homogeneity instrumentation for the bias-free denoiser suites.

A bias-free denoiser exists to be degree-1 positively homogeneous:
``f(c * x) == c * f(x)`` for ``c > 0``. Every test in this package that claims to
check that property must go through :func:`homogeneity_error` here, so the regime
and the tolerance are decided in ONE place.

**The regime is the whole point.** Before 2026-08-15 both `test_scaling_invariance_property`
tests built an UNTRAINED model, where stock ``BatchNormalization``'s ``moving_mean`` is still
exactly 0 and ``moving_var`` exactly 1 — so stock BN is *exactly* homogeneous at
``training=False`` and the tests passed for a reason unrelated to what they claimed to check
(at ``rtol=atol=1e-1`` they would not have caught it either way). The property can only break
once training has moved ``moving_mean`` off zero, which is why :func:`fit_one_step` exists and
why every homogeneity assertion in this package must run after it.

**The probe forces TRUE float32 (TF32 off).** Measured 2026-08-18 on GPU 1 (RTX 4070):
with TF32 left at its TensorFlow default (ON), this same probe reads ~2.9e-04 on the depth-2
bfunet -- 29x over the 1e-5 bar -- and `test_scaling_invariance_property` was RED on GPU while
green on CPU. That number is the HARDWARE floor, not a model defect: TF32 truncates the conv
mantissa to 10 bits, so one TF32 ulp is ``2**-11 = 4.88e-04``. Flipping the flag on the SAME
already-trained model object mid-process moves the reading 2.861e-04 -> 1.186e-06 -> 2.861e-04,
reversibly, which is what identifies the cause. A per-layer walk puts ~3.7e-04 at the FIRST
Conv2D (on the raw input) with no downstream jump, and `BiasFreeBatchNorm` adds nothing
(3.672e-04 -> 3.799e-04). Homogeneity is an exact-arithmetic ARCHITECTURAL property, so the
measurement must not be taken in a reduced-precision matmul regime. See
:func:`homogeneity_error`.

Derivation of ``HOMOGENEITY_RTOL`` (all figures measured on **CPU**,
``CUDA_VISIBLE_DEVICES=""``, which is equivalent to the TF32-off regime the probe now forces
everywhere):

* ``BiasFreeBatchNorm`` at ``training=False`` divides by a FROZEN ``running_var`` and adds
  nothing, so ``f(c*x) == c*f(x)`` holds exactly in exact arithmetic. The only residual is
  float32 round-off.
* Measured on a depth-2 / 8-filter bfunet and a 2-block / 8-filter bfcnn, each after one
  ``fit()`` step: relative error **0.0** at ``c=0.5`` (a power of two — binary scaling is
  exact) and **1.12e-06** (bfunet) / **2.67e-07** (bfcnn) at ``c=3.0``. float32 eps is
  1.19e-07, so the worst case is ~10 eps, consistent with accumulation down the graph.
* The same models with stock ``BatchNormalization`` measure **6.84e-03** (bfunet) and
  **1.25e-02** (bfcnn) at ``c=0.5``.

``1e-5`` therefore sits ~9x above the float32 floor and ~680x below the defect. Keep the
models these tests build SMALL and fixed: a deeper graph accumulates more round-off, and the
headroom is what makes the tolerance meaningful rather than decorative.
"""

import contextlib
from typing import Callable, Iterator

import keras
import numpy as np
import pytest
import tensorflow as tf

# See the module docstring for the derivation. Do not loosen this without re-measuring
# both the float32 floor and the stock-BN defect on CPU.
HOMOGENEITY_RTOL = 1e-5

# Two scale factors, neither 1.0 (trivially true) nor 2.0. 0.5 is a power of two, so binary
# scaling of the input mantissa is exact and the measured error isolates the graph; 3.0 is
# not, so it also exercises input-side rounding.
HOMOGENEITY_SCALES = (0.5, 3.0)


def fit_one_step(model: keras.Model, seed: int = 0) -> keras.Model:
    """Run exactly one ``fit()`` step so any ``moving_mean`` leaves zero.

    Interface contract:

    :param model: An uncompiled denoiser. Compiled in place with plain SGD (no momentum,
        no Adam state) — the optimizer is irrelevant here; what matters is that the
        BatchNormalization running statistics update, which happens in the forward pass.
    :type model: keras.Model
    :param seed: Seed for the synthetic batch.
    :type seed: int
    :return: The same model object, trained for one step.
    :rtype: keras.Model
    :raises ValueError: Never; propagates whatever ``fit`` raises.
    """
    shape = tuple(d if d is not None else 32 for d in model.input_shape[1:])
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, *shape)).astype("float32")
    y = (rng.normal(size=(8, *shape)) * 0.1).astype("float32")
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    return model


@contextlib.contextmanager
def tf32_disabled() -> Iterator[None]:
    """Force TRUE float32 matmul/conv for the duration of the block.

    # DECISION plan-2026-08-18T123346-c3c4a681/D-001
    MEASURED CAUSE of the pre-2026-08-18 RED
    `test_bfunet_denoiser.py::TestBiasFreeUNet::test_scaling_invariance_property` (5.063e-04
    vs a 1e-5 bar, GPU-only, green on CPU): NVIDIA TensorFloat-32. It is not a bias leak and
    there is nothing wrong with `bfunet` or `BiasFreeBatchNorm`. Evidence, all on GPU 1:
    the error is FLAT in ``c`` (2.86e-04 at c=3 through 2.75e-04 at c=1e4 -- an additive-bias
    leak would decay as 1/c and a clipping nonlinearity would grow), it is EXACTLY 0.0 at
    every power of two, it appears in full at the first Conv2D on the raw input, and toggling
    ``enable_tensor_float_32_execution`` on one already-trained model object moves it
    2.861e-04 -> 1.186e-06 -> 2.861e-04. 2.9e-04 is simply the TF32 ulp (2**-11 = 4.88e-04).

    Do NOT "fix" that failure by widening ``HOMOGENEITY_RTOL`` instead: the stock-BN defect
    this suite exists to catch measures 6.8e-03 (bfunet) / 1.3e-02 (bfcnn), so a TF32-proof
    tolerance of ~1e-3 would leave under 7x headroom and the guard would be decorative.
    Do NOT move this disable to import time or to a module-level bare call either -- that is
    PROCESS-GLOBAL and silently changes the regime of every module collected afterwards
    (the exact defect `tests/test_layers/conftest.py` was written to undo). Capture / restore
    in ``finally`` / assert the restoration: same convention as that file's ``tf32_disabled``
    fixture, which is not importable from this tree.

    The previous value is CAPTURED, never assumed ``True``: another module may already have
    disabled it. On CPU the flag is inert and this is a no-op.

    :yield: Nothing; TF32 is off inside the block.
    :rtype: Iterator[None]
    :raises AssertionError: If the prior setting fails to restore.
    """
    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        yield
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(previous)
        assert (
            tf.config.experimental.tensor_float_32_execution_enabled() == previous
        ), "TF32 setting leaked out of the homogeneity probe"


def homogeneity_error(model: keras.Model, x: np.ndarray, c: float) -> float:
    """Relative violation of ``f(c*x) == c*f(x)`` at inference.

    Interface contract:

    :param model: A built denoiser. Called with ``training=False`` -- homogeneity is an
        INFERENCE-time property; during training a batch norm uses per-batch statistics and
        is degree-0 whichever variant it is, so a ``training=True`` reading proves nothing.
    :type model: keras.Model
    :param x: Input batch.
    :type x: np.ndarray
    :param c: Positive scale factor.
    :type c: float

    Runs inside :func:`tf32_disabled`, so the reading is a TRUE-float32 number on every
    device. Without that the GPU reports the TF32 ulp (~2.9e-04) rather than the model's
    homogeneity -- see the module docstring and D-001 on :func:`tf32_disabled`.
    :return: ``max|f(c*x) - c*f(x)| / max|c*f(x)|``. Relative, so it is comparable across
        models of different output magnitude. Returns ``inf`` if the model output is
        identically zero (a dead model must not read as perfectly homogeneous).
    :rtype: float
    """
    with tf32_disabled():
        out = model(x, training=False)
        out_scaled = model(c * x, training=False)
    if isinstance(out, list):  # deep supervision: index 0 is the full-resolution output
        out, out_scaled = out[0], out_scaled[0]
    f = np.asarray(out, dtype=np.float64)
    fc = np.asarray(out_scaled, dtype=np.float64)
    denom = np.abs(c * f).max()
    if denom == 0.0:
        return float("inf")
    return float(np.abs(fc - c * f).max() / denom)


@pytest.fixture
def homogeneity_probe() -> Callable[[keras.Model, np.ndarray, float], float]:
    """Expose :func:`homogeneity_error` as a fixture for readability at the call site."""
    return homogeneity_error
