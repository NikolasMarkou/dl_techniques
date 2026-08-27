"""Guard: the equal-length fast path must be CROSS attention, not a hybrid.

Why this test exists
--------------------
``_two_modality_attention`` takes a batch-stacked "fast path" when the two
modalities have equal length. Un-swapping K there -- one character, ``k_splits[1],
k_splits[0]`` becoming ``k_splits[0], k_splits[1]`` -- turns stream A's attention
from ``softmax(Q_A . K_B) V_B`` into ``softmax(Q_A . K_A) V_B``: neither pure
cross- nor pure self-attention, and no longer the layer's documented contract
(``Out_A = Attention(Q_A, K_B, V_B)``). That mutation passes all 38 existing
tests while changing the output by 0.787, because every equal-length test checks
only shape and NaN-freedom.

The fast path is taken whenever the two modalities happen to be the same length,
which is a common case (two image-patch streams, or text pairs padded to a shared
length), so this is the layer's headline feature running untested.

The oracle
----------
The expected values are recomputed BY HAND from the layer's own already-built
``qkv_dense``/``proj_dense`` -- identical weights -- but never by calling
``_two_modality_attention``. So it is external to BOTH code paths rather than
"call the general branch and compare", which would only prove the two branches
agree with each other.

TF32: the comparison is a GPU matmul against a hand-built one at ``atol=1e-5``,
which Ampere TF32 precision (~1e-3 relative) breaks. This module opts into the
repo's ``tf32_disabled`` fixture, as ``test_linear_attention.py`` does.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.shared_weights_cross_attention import (
    SharedWeightsCrossAttention,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 256
HEADS = 8
N = 128


def test_the_equal_length_fast_path_matches_an_external_oracle():
    """The regression itself."""
    keras.utils.set_random_seed(7)
    layer = SharedWeightsCrossAttention(
        dim=DIM, num_heads=HEADS, dropout_rate=0.0
    )

    x_a = keras.random.normal((2, N, DIM), seed=11)
    x_b = keras.random.normal((2, N, DIM), seed=22)
    combined = keras.ops.concatenate([x_a, x_b], axis=1)

    actual = np.asarray(
        layer(combined, split_sizes=[N, N], training=False)
    )

    # --- oracle: rebuild the documented formula from the layer's own weights ---
    qkv = layer.qkv_dense(combined)
    qkv = keras.ops.reshape(qkv, (-1, 2 * N, 3, HEADS, DIM // HEADS))
    qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]

    q_a, q_b = q[:, :, :N, :], q[:, :, N:, :]
    k_a, k_b = k[:, :, :N, :], k[:, :, N:, :]
    v_a, v_b = v[:, :, :N, :], v[:, :, N:, :]

    # Out_A = Attention(Q_A, K_B, V_B) -- the CROSS pairing.
    scores_a = keras.ops.matmul(
        q_a, keras.ops.transpose(k_b, (0, 1, 3, 2))
    ) * layer.scale
    out_a = keras.ops.matmul(keras.ops.softmax(scores_a, axis=-1), v_b)

    # Out_B = Attention(Q_B, K_A, V_A)
    scores_b = keras.ops.matmul(
        q_b, keras.ops.transpose(k_a, (0, 1, 3, 2))
    ) * layer.scale
    out_b = keras.ops.matmul(keras.ops.softmax(scores_b, axis=-1), v_a)

    merged = keras.ops.concatenate([out_a, out_b], axis=2)
    merged = keras.ops.transpose(merged, (0, 2, 1, 3))
    merged = keras.ops.reshape(merged, (-1, 2 * N, DIM))
    expected = np.asarray(layer.proj_dense(merged))

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-5,
        rtol=0,
        err_msg=(
            "the equal-length fast path does not compute "
            "Out_A = Attention(Q_A, K_B, V_B); the most likely cause is the K "
            "(or V) streams no longer being swapped, which turns cross-attention "
            "into a self/cross hybrid"
        ),
    )


def test_stream_a_depends_on_stream_b():
    """A coarser, independent statement of the same contract.

    Honest about its own reach: this does NOT catch the K-un-swap mutation. That
    mutant still routes ``V_B`` into stream A, so A genuinely still depends on B
    and this assertion passes -- measured. It is kept because it fails for a
    different family of breakages (dropping the cross term entirely, or splitting
    the streams) that the exact oracle above would also catch but less legibly,
    and because it states in plain terms what "cross attention" has to mean. The
    exact oracle is what pins the specific Q/K/V pairing.
    """
    keras.utils.set_random_seed(7)
    layer = SharedWeightsCrossAttention(
        dim=DIM, num_heads=HEADS, dropout_rate=0.0
    )
    x_a = np.asarray(keras.random.normal((2, N, DIM), seed=11))
    x_b = np.asarray(keras.random.normal((2, N, DIM), seed=22))

    before = np.asarray(
        layer(
            keras.ops.concatenate([x_a, x_b], axis=1),
            split_sizes=[N, N],
            training=False,
        )
    )

    x_b_perturbed = x_b.copy()
    x_b_perturbed[:, 0, :] += 10.0
    after = np.asarray(
        layer(
            keras.ops.concatenate([x_a, x_b_perturbed], axis=1),
            split_sizes=[N, N],
            training=False,
        )
    )

    delta_a = np.abs(before[:, :N, :] - after[:, :N, :]).max()
    assert delta_a > 1e-4, (
        f"stream A moved by only {delta_a} when stream B was perturbed: A is not "
        "attending to B, so this is not cross-attention"
    )
