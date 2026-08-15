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

Derivation of ``HOMOGENEITY_RTOL`` (all figures measured on **CPU**,
``CUDA_VISIBLE_DEVICES=""``; GPU 1 disagrees with itself at ~5e-6 on this probe, so a GPU
number cannot bound a float32-round-off tolerance):

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

from typing import Callable

import keras
import numpy as np
import pytest

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
    :return: ``max|f(c*x) - c*f(x)| / max|c*f(x)|``. Relative, so it is comparable across
        models of different output magnitude. Returns ``inf`` if the model output is
        identically zero (a dead model must not read as perfectly homogeneous).
    :rtype: float
    """
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
