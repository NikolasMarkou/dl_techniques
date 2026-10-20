"""One PFT layer's forward equals the official composition of its sub-ops.

The oracle is ``pft_reference_numpy.pft_layer``, a line-for-line numpy
transcription of the official ``PFTransformerLayer.forward`` and
``WindowAttention.forward`` (CVL-UESTC/PFT-SR ``basicsr/archs/pft_arch.py``,
:240-348 and :438-481): LayerNorm, ``wqkv`` on the whole map, the LePE
depthwise conv plus GELU on V over the whole map (added to ``attn @ v``, not to
V), cyclic shift, window partition, ``q * scale @ k^T`` plus the relative
position bias and (dense, shifted) the -100 mask, softmax, the Hadamard product
with the inherited map and its ``(a + eps) / (sum + eps)`` renormalisation,
top-k, and ``ConvFFN`` (fc1, GELU, ``x + GELU(dwconv(x))``, fc2). The oracle
runs the official SPARSE form (gathers over an index tensor); the port runs a
dense map under a boolean mask. They share no code.

Every arm runs at float64 on a non-square ``8 x 12`` map with ``window_size=4``
(6 windows), so a transposed window order, a swapped H/W or a square-only
shortcut cannot pass, and the bound is float64 reassociation, not float32
noise. The oracle rounds two constants as TensorFlow does (GELU's ``sqrt(2)``
and the LeakyReLU slope are float32 inside TF even at float64; see
``pft_reference_numpy.TF_SQRT2``); with exact constants the gap is 4e-11,
still four orders below any structural error.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.super_resolution.pft_sr.blocks import PFTLayer

from . import pft_reference_numpy as ref
from ..precision_arm_oracle import float64_scope

DIM, HEADS, WS, H, W = 16, 2, 4, 8, 12
N = WS * WS
ATOL = 1e-12          # measured max 4e-15 (CPU and GPU), float64 reassociation


def _layer(shift_size, keep_topk, seed):
    keras.utils.set_random_seed(seed)
    layer = PFTLayer(dim=DIM, num_heads=HEADS, window_size=WS, shift_size=shift_size,
                     convffn_kernel_size=5, mlp_ratio=2.0, keep_topk=keep_topk)
    layer.build((None, H, W, DIM))
    # Non-zero biases and LayerNorm affines, so a dropped bias or affine shows.
    rng = np.random.default_rng(seed)
    for v in layer.weights:
        if v.path.endswith(("bias", "beta", "gamma", "relative_position_bias_table")):
            v.assign(rng.normal(0.0, 0.3, v.shape) + (1.0 if v.path.endswith("gamma") else 0.0))
    return layer


def _dense(values, indices):
    """The official sparse ``(values, indices)`` as a dense ``(.., n, n)`` map."""
    if indices is None:
        return values
    out = np.zeros(values.shape[:-1] + (N,))
    np.put_along_axis(out, indices, values, axis=-1)
    return out


def _x(seed):
    return np.random.default_rng(seed).standard_normal((2, H, W, DIM))


@pytest.mark.parametrize("shift_size", [0, WS // 2])
def test_a_first_layer_equals_the_official_dense_attention(shift_size):
    with float64_scope():
        layer = _layer(shift_size, None, seed=1)
        x = _x(0)
        y, values, mask = layer(x)
        pv, pi = [None, None], [None, None]
        expect = ref.pft_layer(x, layer, HEADS, WS, shift_size, N, pv, pi)
    assert mask is None
    np.testing.assert_allclose(np.asarray(y), expect, rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(values), pv[1 if shift_size else 0],
                               rtol=0, atol=ATOL)


@pytest.mark.parametrize("shift_size", [0, WS // 2])
def test_a_chain_of_layers_equals_the_official_sparse_chain(shift_size):
    """Four layers of one chain: dense, top-k 10 of 16, top-k 6 of 10 on the
    sparse map, then keep the width (6 is not below 6, so no top-k: the
    official code reuses the indices). Outputs, inherited values and the kept
    key sets all equal the official chain's."""
    schedule = [16, 10, 6, 6]
    keeps = [None, 10, 6, None]
    chain = 1 if shift_size else 0
    with float64_scope():
        x = _x(2)
        pv, pi = [None, None], [None, None]
        values = mask = None
        for i, (k_official, keep) in enumerate(zip(schedule, keeps)):
            layer = _layer(shift_size, keep, seed=10 + i)
            y, values, mask = layer(x, prev_values=values, prev_mask=mask)
            expect = ref.pft_layer(x, layer, HEADS, WS, shift_size, k_official, pv, pi)
            np.testing.assert_allclose(np.asarray(y), expect, rtol=0, atol=ATOL,
                                       err_msg=f"layer {i}")
            np.testing.assert_allclose(np.asarray(values), _dense(pv[chain], pi[chain]),
                                       rtol=0, atol=ATOL, err_msg=f"layer {i} values")
            if pi[chain] is None:
                assert mask is None
            else:
                official = np.zeros_like(np.asarray(mask), dtype=bool)
                np.put_along_axis(official, pi[chain], True, axis=-1)
                assert np.array_equal(np.asarray(mask), official), f"layer {i} kept keys"
            x = np.asarray(y)
    assert pi[chain].shape[-1] == 6


def test_the_sparse_map_skips_the_shift_mask_as_the_official_code_does():
    """After a top-k the official sparse branch adds no -100 mask (:271-284).

    Where that shows: a shifted chain with 4-key regions keeps 10 of 16 keys,
    so some kept keys cross a region. Their inherited weight is about
    ``eps / sum``, so the OUTPUT barely moves either way (below 1e-12 at
    float64), but the next top-k ranks them: unmasked, by their own scores, as
    the official code does; masked, they all tie at ``eps / sum`` and the tie
    order decides. So this arm compares the kept keys of the sparse top-k
    layer, not only its output (measured: re-adding the mask in the sparse
    branch keeps every output arm green and fails this one)."""
    shift = WS // 2
    with float64_scope():
        x = _x(3)
        pv, pi = [None, None], [None, None]
        values = mask = None
        for i, (k_official, keep) in enumerate([(16, None), (10, 10), (6, 6)]):
            layer = _layer(shift, keep, seed=20 + i)
            y, values, mask = layer(x, prev_values=values, prev_mask=mask)
            expect = ref.pft_layer(x, layer, HEADS, WS, shift, k_official, pv, pi)
            np.testing.assert_allclose(np.asarray(y), expect, rtol=0, atol=ATOL)
            x = np.asarray(y)
    official = np.zeros_like(np.asarray(mask), dtype=bool)
    np.put_along_axis(official, pi[1], True, axis=-1)
    region = ref.calculate_mask(H, W, WS)                       # (nw, n, n)
    crossing = np.broadcast_to(region[None, :, None] != 0, (2, 6, HEADS, N, N))
    kept = np.asarray(mask).reshape(2, 6, HEADS, N, N)
    assert np.any(kept & crossing), "no kept key crosses a region: the arm tests nothing"
    assert np.array_equal(np.asarray(mask), official), "the sparse top-k kept other keys"
