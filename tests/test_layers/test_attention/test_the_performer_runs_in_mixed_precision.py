"""Guard: ``PerformerAttention`` must run under ``mixed_float16``.

Why this test exists
--------------------
``_create_projection_matrix`` called ``keras.random.normal(...)`` with no
``dtype=``, so the random projection followed the GLOBAL float policy and came
back float32 while ``q``/``k`` arrived in the layer's compute dtype. Under
``keras.mixed_precision.set_global_policy("mixed_float16")`` the projection einsum
and the ``features * self._feature_scale`` multiply then mixed a half tensor with
a float tensor, and the layer raised::

    InvalidArgumentError: cannot compute Mul as input #1(zero-based) was expected
    to be a float tensor but is a half tensor

i.e. the layer could not run at all under the project's standard mixed-precision
training policy -- a hard crash on the first forward pass, not a silent numeric
drift.

Why the existing suite could not see it: ``test_performer_attention.py`` never
sets a mixed-precision policy. ``grep mixed_precision`` on that file returns
nothing.

Why this can fail if the implementation is wrong: dropping ``dtype=`` from the
``keras.random.normal`` call reinstates the crash, which is how this guard was
proven RED.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.performer_attention import PerformerAttention

DIM = 32
HEADS = 4
FEATURES = 64
# Deliberately a realistic length: a reduction defect can hide at a toy size.
SEQ = 512


@pytest.fixture(name="mixed_float16")
def _mixed_float16():
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


@pytest.fixture(name="tokens")
def _tokens():
    return np.random.default_rng(0).normal(size=(2, SEQ, DIM)).astype("float32")


@pytest.mark.usefixtures("mixed_float16")
@pytest.mark.parametrize("causal", [False, True])
def test_a_forward_pass_does_not_raise(tokens, causal):
    """The regression itself."""
    layer = PerformerAttention(
        dim=DIM, num_heads=HEADS, nb_features=FEATURES, causal=causal
    )
    out = layer(tokens)
    assert tuple(out.shape) == (2, SEQ, DIM)


@pytest.mark.usefixtures("mixed_float16")
def test_the_output_carries_the_layers_own_compute_dtype(tokens):
    """A value can be finite and still be the WRONG dtype -- check both."""
    layer = PerformerAttention(dim=DIM, num_heads=HEADS, nb_features=FEATURES)
    out = layer(tokens)
    assert layer.compute_dtype == "float16"
    assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype


@pytest.mark.usefixtures("mixed_float16")
def test_the_output_is_finite_at_a_realistic_length(tokens):
    """The `1e-6` normalizer floor must survive fp16 at N=512, not just a toy N."""
    layer = PerformerAttention(dim=DIM, num_heads=HEADS, nb_features=FEATURES)
    values = np.asarray(layer(tokens))
    assert np.all(np.isfinite(values)), "non-finite output under mixed_float16"


@pytest.mark.usefixtures("mixed_float16")
def test_the_projection_matrix_matches_the_compute_dtype(tokens):
    """Pin the root cause directly, not only its symptom."""
    layer = PerformerAttention(dim=DIM, num_heads=HEADS, nb_features=FEATURES)
    layer(tokens)
    projection = layer._create_projection_matrix(2)
    assert (
        keras.backend.standardize_dtype(projection.dtype) == layer.compute_dtype
    ), (
        "the random projection is not in the layer's compute dtype; this is the "
        "exact mismatch that made the layer unusable under mixed_float16"
    )


def test_float32_is_unaffected(tokens):
    """The fix must not perturb the default policy's numerics."""
    keras.utils.set_random_seed(11)
    layer = PerformerAttention(dim=DIM, num_heads=HEADS, nb_features=FEATURES)
    out = np.asarray(layer(tokens))
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
