"""Tests for SAM 3's decoder layer (`models/SAM/SAM3/decoder.py`).

Four design choices in this file are load-bearing, and every one of them exists
because of a measurement, not a preference.

**The oracles are written from the SPEC, in float64, from the layer's WEIGHTS.**
``_reference_layer_forward`` re-derives the whole four-sub-block forward pass --
three attentions, four normalizations, the feed-forward, the presence-token
prepend and split -- in double precision without ever calling the layer. An
oracle that calls the implementation is a tautology, and this plan has already
shipped one green oracle certifying the wrong quantity.

**The tolerance is 1e-3 because of TF32.** GPU 1 is an RTX 4070 and runs float32
matmuls through TF32; the same float64-vs-float32 comparison measures ~1e-7 on
CPU and ~1e-4 there. ``test_the_oracle_probes_are_unsaturated`` pins the wrong
candidates' margins so a loose tolerance cannot quietly swallow a defect.

**The boxRPB probe set deliberately contains ``d = 0``.** That point is a
COINCIDENCE point for three of the four wrong candidates -- and it is the only
point that separates the fourth (dropping the ``+ 1`` makes ``log2(0) = -inf``).
The negative probe is what separates an unsigned form, and ``|d * 8| = 1`` is
what separates a missing ``/ log2(8)``.

**Every "is invariant" assertion is flanked by a positive liveness arm.** An
absence assertion (``delta == 0.0``) is satisfied by construction by a dead
component, which is exactly what a dead-component probe exposes; the presence
token's bias-invariance and the ``memory_mask`` refusal are both that shape.
"""

import math

import keras
import numpy as np
import pytest
from keras import layers, ops

from dl_techniques.models.SAM.SAM3.decoder import (
    Sam3DecoderLayer,
    Sam3TransformerDecoder,
    _Sam3DecoderAttention,
    _box_rpb_bias,
    _box_rpb_log_compress,
)

# ---------------------------------------------------------------------
# tiny variant
# ---------------------------------------------------------------------

TINY = dict(d_model=8, num_heads=2, dim_feedforward=16, dropout_rate=0.0,
            activation="relu", use_text_cross_attention=True,
            norm_epsilon=1e-5)

# The settled SAM 3 decoder layer, re-read from the pinned upstream clone's
# `_create_transformer_decoder()`: activation "relu", d_model 256,
# dim_feedforward 2048, dropout 0.1, 8 heads, text cross-attention ON.
SHIPPED = dict(d_model=256, num_heads=8, dim_feedforward=2048,
               dropout_rate=0.1, activation="relu",
               use_text_cross_attention=True, norm_epsilon=1e-5)

BATCH, QUERIES, GRID_H, GRID_W, TOKENS = 2, 5, 3, 4, 6
KEYS = GRID_H * GRID_W
TOL = 1e-3


def _params(cfg: dict) -> int:
    """Closed-form parameter count, written from the STRUCTURE."""
    d, f = cfg["d_model"], cfg["dim_feedforward"]
    dense = d * d + d
    attention = 4 * dense                      # q, k, v, out
    cross_text = dense + (d * 2 * d + 2 * d) + dense
    ffn = (d * f + f) + (f * d + d)
    norms = 4 * 2 * d if cfg["use_text_cross_attention"] else 3 * 2 * d
    total = attention + cross_text + attention + ffn + norms
    if not cfg["use_text_cross_attention"]:
        total -= cross_text
    return total


def _randomize(layer: keras.layers.Layer, seed: int = 0) -> None:
    """Give every weight -- kernels AND biases -- a non-trivial value."""
    rng = np.random.default_rng(seed)
    for weight in layer.weights:
        weight.assign(rng.normal(0.0, 0.4, size=weight.shape).astype("float32"))


def _inputs(seed: int = 7) -> dict:
    """A full, non-degenerate call payload."""
    rng = np.random.default_rng(seed)
    padding = np.zeros((BATCH, TOKENS), dtype=bool)
    padding[0, TOKENS - 2:] = True
    return dict(
        tgt=rng.normal(size=(BATCH, QUERIES, TINY["d_model"])).astype("f4"),
        memory=rng.normal(size=(BATCH, KEYS, TINY["d_model"])).astype("f4"),
        query_pos=rng.normal(size=(BATCH, QUERIES, TINY["d_model"])).astype("f4"),
        memory_pos=rng.normal(size=(BATCH, KEYS, TINY["d_model"])).astype("f4"),
        memory_text=rng.normal(size=(BATCH, TOKENS, TINY["d_model"])).astype("f4"),
        text_padding_mask=padding,
        presence_token=rng.normal(size=(BATCH, 1, TINY["d_model"])).astype("f4"),
    )


def _boxes(seed: int = 11, queries: int = QUERIES) -> np.ndarray:
    """Reference boxes in normalized ``cxcywh``, all strictly inside the image."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.25, 0.75, size=(BATCH, queries, 2))
    sizes = rng.uniform(0.1, 0.4, size=(BATCH, queries, 2))
    return np.concatenate([centres, sizes], axis=-1).astype("float32")


def _rpb_mlps(num_heads: int = 2, hidden: int = 8, seed: int = 5):
    """The two per-axis boxRPB embedding MLPs, as the layer stack will own them.

    ``Linear(2, hidden) -> ReLU -> Linear(hidden, num_heads)``, matching the
    reference's ``MLP(n_input=2, d_model, nheads, num_layers=2)`` whose final
    layer carries NO activation.
    """
    rng = np.random.default_rng(seed)
    made = []
    for _ in range(2):
        mlp = keras.Sequential([layers.Dense(hidden, activation="relu"),
                                layers.Dense(num_heads)])
        mlp.build((None, None, None, 2))
        for weight in mlp.weights:
            weight.assign(
                rng.normal(0.0, 0.6, size=weight.shape).astype("float32"))
        made.append(mlp)
    return made[0], made[1]


@pytest.fixture()
def layer() -> Sam3DecoderLayer:
    built = Sam3DecoderLayer(**TINY)
    built.build((BATCH, QUERIES, TINY["d_model"]),
                (BATCH, KEYS, TINY["d_model"]),
                (BATCH, TOKENS, TINY["d_model"]))
    _randomize(built)
    return built


# ---------------------------------------------------------------------
# float64 oracles, written from the SPEC (never from the implementation)
# ---------------------------------------------------------------------


def _np(x) -> np.ndarray:
    return np.asarray(ops.convert_to_numpy(x), dtype=np.float64)


def _oracle_log_compress(deltas: np.ndarray) -> np.ndarray:
    scaled = deltas * 8.0
    return np.sign(scaled) * np.log2(np.abs(scaled) + 1.0) / np.log2(8.0)


def _oracle_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _oracle_layer_norm(x: np.ndarray, gamma, beta, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _oracle_heads(x: np.ndarray, heads: int) -> np.ndarray:
    batch, seq, dim = x.shape
    return x.reshape(batch, seq, heads, dim // heads).transpose(0, 2, 1, 3)


def _oracle_attention(query, key, value, kernels, heads, bias=None,
                      keep=None) -> np.ndarray:
    """Independent multi-head attention over three separate tensors."""
    (wq, bq), (wk, bk), (wv, bv), (wo, bo) = kernels
    q = _oracle_heads(query @ wq + bq, heads)
    k = _oracle_heads(key @ wk + bk, heads)
    v = _oracle_heads(value @ wv + bv, heads)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
    if bias is not None:
        scores = scores + bias
    if keep is not None:
        scores = np.where(keep[:, None, None, :], scores, -1e9)
    out = _oracle_softmax(scores) @ v
    batch, _, seq, head_dim = out.shape
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq, heads * head_dim)
    return out @ wo + bo


def _dense(layer_obj):
    return _np(layer_obj.kernel), _np(layer_obj.bias)


def _reference_layer_forward(layer_obj, payload, bias=None,
                             with_presence=True):
    """The whole decoder layer, in float64, from the layer's WEIGHTS only."""
    heads = layer_obj.num_heads
    eps = layer_obj.norm_epsilon
    tgt = _np(payload["tgt"])
    query_pos = _np(payload["query_pos"])

    if with_presence:
        presence = _np(payload["presence_token"])
        tgt = np.concatenate([presence, tgt], axis=1)
        query_pos = np.concatenate([np.zeros_like(presence), query_pos], axis=1)
        if bias is not None:
            bias = np.concatenate([np.zeros_like(bias[:, :, :1, :]), bias],
                                  axis=2)

    # 1. self-attention: q = k = tgt + pos, v = tgt (the pos is NOT on v).
    self_kernels = [_dense(layer_obj.self_attn.q_proj),
                    _dense(layer_obj.self_attn.k_proj),
                    _dense(layer_obj.self_attn.v_proj),
                    _dense(layer_obj.self_attn.out_proj)]
    attended = _oracle_attention(tgt + query_pos, tgt + query_pos, tgt,
                                 self_kernels, heads)
    gamma, beta = _np(layer_obj.norm2.gamma), _np(layer_obj.norm2.beta)
    tgt = _oracle_layer_norm(tgt + attended, gamma, beta, eps)

    # 2. text cross-attention, through the STOCK layer's q / kv / proj shape.
    if layer_obj.use_text_cross_attention:
        text = _np(payload["memory_text"])
        wq, bq = _dense(layer_obj.ca_text.q_dense)
        wkv, bkv = _dense(layer_obj.ca_text.kv_dense)
        dim = layer_obj.d_model
        text_kernels = [(wq, bq), (wkv[:, :dim], bkv[:dim]),
                        (wkv[:, dim:], bkv[dim:]),
                        _dense(layer_obj.ca_text.proj_dense)]
        keep = None
        if payload.get("text_padding_mask") is not None:
            keep = ~np.asarray(payload["text_padding_mask"], dtype=bool)
        attended = _oracle_attention(tgt + query_pos, text, text,
                                     text_kernels, heads, keep=keep)
        gamma, beta = (_np(layer_obj.catext_norm.gamma),
                       _np(layer_obj.catext_norm.beta))
        tgt = _oracle_layer_norm(tgt + attended, gamma, beta, eps)

    # 3. image cross-attention: keys carry memory_pos, values do not.
    memory = _np(payload["memory"])
    memory_pos = _np(payload["memory_pos"])
    cross_kernels = [_dense(layer_obj.cross_attn.q_proj),
                     _dense(layer_obj.cross_attn.k_proj),
                     _dense(layer_obj.cross_attn.v_proj),
                     _dense(layer_obj.cross_attn.out_proj)]
    attended = _oracle_attention(tgt + query_pos, memory + memory_pos, memory,
                                 cross_kernels, heads, bias=bias)
    gamma, beta = _np(layer_obj.norm1.gamma), _np(layer_obj.norm1.beta)
    tgt = _oracle_layer_norm(tgt + attended, gamma, beta, eps)

    # 4. feed-forward: fc1 -> relu -> fc2, residual, norm.
    w1, b1 = _dense(layer_obj.ffn.fc1)
    w2, b2 = _dense(layer_obj.ffn.fc2)
    projected = np.maximum(tgt @ w1 + b1, 0.0) @ w2 + b2
    gamma, beta = _np(layer_obj.norm3.gamma), _np(layer_obj.norm3.beta)
    tgt = _oracle_layer_norm(tgt + projected, gamma, beta, eps)

    if not with_presence:
        return tgt, None
    return tgt[:, 1:], tgt[:, :1]


def _oracle_rpb(boxes: np.ndarray, grid, embed_x, embed_y, heads,
                mode="log") -> np.ndarray:
    """boxRPB, in float64, from the MLPs' weights only."""
    height, width = grid
    boxes = np.asarray(boxes, dtype=np.float64)
    cx, cy, bw, bh = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    edges_x = np.stack([cx - 0.5 * bw, cx + 0.5 * bw], axis=-1)
    edges_y = np.stack([cy - 0.5 * bh, cy + 0.5 * bh], axis=-1)
    coords_y = np.arange(height, dtype=np.float64) / height
    coords_x = np.arange(width, dtype=np.float64) / width
    deltas_y = coords_y[None, None, :, None] - edges_y[:, :, None, :]
    deltas_x = coords_x[None, None, :, None] - edges_x[:, :, None, :]
    if mode == "log":
        deltas_y = _oracle_log_compress(deltas_y)
        deltas_x = _oracle_log_compress(deltas_x)

    def run(mlp, x):
        w1, b1 = _np(mlp.layers[0].kernel), _np(mlp.layers[0].bias)
        w2, b2 = _np(mlp.layers[1].kernel), _np(mlp.layers[1].bias)
        return np.maximum(x @ w1 + b1, 0.0) @ w2 + b2

    bias_y, bias_x = run(embed_y, deltas_y), run(embed_x, deltas_x)
    bias = bias_y[:, :, :, None, :] + bias_x[:, :, None, :, :]
    bias = bias.transpose(0, 4, 1, 2, 3)
    return bias.reshape(bias.shape[0], heads, bias.shape[2], height * width)


# ---------------------------------------------------------------------
# G6.1 -- boxRPB's log formula (value oracle (a))
# ---------------------------------------------------------------------

# `d = 0` separates a missing `+ 1`; `|d * 8| = 1` separates a missing
# `/ log2(8)`; the two negatives separate an unsigned form; every non-zero
# point separates a linear form.
LOG_PROBES = np.array([0.0, 0.125, -0.125, 0.5, -0.375], dtype=np.float64)


class TestBoxRpbLogFormula:

    def test_matches_the_float64_oracle_at_every_probe(self):
        measured = _np(_box_rpb_log_compress(
            ops.convert_to_tensor(LOG_PROBES.astype("float32"))))
        np.testing.assert_allclose(measured, _oracle_log_compress(LOG_PROBES),
                                   atol=TOL)

    def test_zero_delta_is_exactly_zero_and_finite(self):
        value = _np(_box_rpb_log_compress(
            ops.convert_to_tensor(np.zeros(3, dtype="float32"))))
        assert np.all(np.isfinite(value)), "the `+ 1` is missing: log2(0)"
        assert np.max(np.abs(value)) == 0.0

    def test_eight_scaled_unit_delta_is_exactly_one_third(self):
        value = _np(_box_rpb_log_compress(
            ops.convert_to_tensor(np.array([0.125], dtype="float32"))))
        np.testing.assert_allclose(value, [1.0 / 3.0], atol=1e-6)

    def test_the_form_is_sign_preserving(self):
        pair = _np(_box_rpb_log_compress(
            ops.convert_to_tensor(np.array([0.3, -0.3], dtype="float32"))))
        assert pair[0] > 0.0 and pair[1] < 0.0
        np.testing.assert_allclose(pair[0], -pair[1], atol=1e-6)

    def test_the_probe_set_separates_every_wrong_candidate(self):
        """The oracle's discriminating power, asserted rather than assumed."""
        reference = _oracle_log_compress(LOG_PROBES)
        scaled = LOG_PROBES * 8.0
        with np.errstate(divide="ignore", invalid="ignore"):
            candidates = {
                "linear": LOG_PROBES,
                "unsigned": np.log2(np.abs(scaled) + 1.0) / np.log2(8.0),
                "no_normalization": np.sign(scaled) * np.log2(
                    np.abs(scaled) + 1.0),
                "no_plus_one": np.sign(scaled) * np.log2(
                    np.abs(scaled)) / np.log2(8.0),
            }
        for name, candidate in candidates.items():
            gap = np.abs(candidate - reference)
            separated = np.nanmax(gap) > 0.2 or bool(
                np.any(~np.isfinite(candidate)))
            assert separated, f"probe set cannot separate {name}"


# ---------------------------------------------------------------------
# G6.2 -- the boxRPB bias tensor
# ---------------------------------------------------------------------


class TestBoxRpbBias:

    def test_shape_is_batch_heads_queries_keys(self):
        embed_x, embed_y = _rpb_mlps()
        bias = _box_rpb_bias(_boxes(), (GRID_H, GRID_W), embed_x, embed_y, 2)
        assert tuple(bias.shape) == (BATCH, 2, QUERIES, KEYS)

    def test_matches_the_float64_outer_sum_oracle(self):
        embed_x, embed_y = _rpb_mlps()
        boxes = _boxes()
        measured = _np(_box_rpb_bias(boxes, (GRID_H, GRID_W), embed_x,
                                     embed_y, 2))
        expected = _oracle_rpb(boxes, (GRID_H, GRID_W), embed_x, embed_y, 2)
        np.testing.assert_allclose(measured, expected, atol=TOL)

    def test_a_product_combination_is_separated_by_the_oracle(self):
        """The outer SUM is not interchangeable with an outer product."""
        embed_x, embed_y = _rpb_mlps()
        boxes = _boxes()
        reference = _oracle_rpb(boxes, (GRID_H, GRID_W), embed_x, embed_y, 2)
        deltas = _oracle_rpb(boxes, (GRID_H, GRID_W), embed_x, embed_y, 2)
        # Rebuild the product candidate from the same per-axis embeddings.
        height, width = GRID_H, GRID_W
        cx, cy = boxes[..., 0], boxes[..., 1]
        bw, bh = boxes[..., 2], boxes[..., 3]
        edges_x = np.stack([cx - 0.5 * bw, cx + 0.5 * bw], axis=-1)
        edges_y = np.stack([cy - 0.5 * bh, cy + 0.5 * bh], axis=-1)
        cw = np.arange(width, dtype=np.float64) / width
        ch = np.arange(height, dtype=np.float64) / height
        dx = _oracle_log_compress(cw[None, None, :, None] - edges_x[:, :, None, :])
        dy = _oracle_log_compress(ch[None, None, :, None] - edges_y[:, :, None, :])

        def run(mlp, x):
            w1, b1 = _np(mlp.layers[0].kernel), _np(mlp.layers[0].bias)
            w2, b2 = _np(mlp.layers[1].kernel), _np(mlp.layers[1].bias)
            return np.maximum(x @ w1 + b1, 0.0) @ w2 + b2

        product = run(embed_y, dy)[:, :, :, None, :] * run(embed_x, dx)[:, :, None, :, :]
        product = product.transpose(0, 4, 1, 2, 3).reshape(
            BATCH, 2, QUERIES, KEYS)
        assert np.max(np.abs(product - reference)) > 0.2
        np.testing.assert_allclose(deltas, reference, atol=1e-12)

    def test_linear_mode_is_measurably_different_from_log_mode(self):
        embed_x, embed_y = _rpb_mlps()
        boxes = _boxes()
        log_bias = _np(_box_rpb_bias(boxes, (GRID_H, GRID_W), embed_x,
                                     embed_y, 2, "log"))
        linear_bias = _np(_box_rpb_bias(boxes, (GRID_H, GRID_W), embed_x,
                                        embed_y, 2, "linear"))
        assert np.max(np.abs(log_bias - linear_bias)) > 0.2
        np.testing.assert_allclose(
            linear_bias,
            _oracle_rpb(boxes, (GRID_H, GRID_W), embed_x, embed_y, 2,
                        mode="linear"), atol=TOL)

    def test_an_unknown_mode_is_refused(self):
        embed_x, embed_y = _rpb_mlps()
        with pytest.raises(ValueError, match="mode must be"):
            _box_rpb_bias(_boxes(), (GRID_H, GRID_W), embed_x, embed_y, 2,
                          mode="both")

    def test_the_two_axis_embeddings_are_independent(self):
        """Perturbing the x MLP must not move the y-only part of the bias."""
        embed_x, embed_y = _rpb_mlps()
        boxes = _boxes()
        before = _np(_box_rpb_bias(boxes, (GRID_H, GRID_W), embed_x, embed_y, 2))
        weight = embed_x.layers[1].bias
        weight.assign(weight + 3.0)
        after = _np(_box_rpb_bias(boxes, (GRID_H, GRID_W), embed_x, embed_y, 2))
        # An x-only bias shift is constant along every ROW of the grid.
        delta = (after - before).reshape(BATCH, 2, QUERIES, GRID_H, GRID_W)
        assert np.max(np.abs(delta)) > 1e-3
        assert np.max(np.std(delta, axis=3)) < 1e-5


# ---------------------------------------------------------------------
# G6.3 -- the bias reaches the softmax (value oracle (b))
# ---------------------------------------------------------------------


class TestBiasInjection:

    @staticmethod
    def _attention(heads: int = 2, dim: int = 8) -> _Sam3DecoderAttention:
        attn = _Sam3DecoderAttention(dim, heads)
        attn.build((BATCH, QUERIES, dim), (BATCH, KEYS, dim),
                   (BATCH, KEYS, dim))
        _randomize(attn, seed=3)
        return attn

    def test_matches_the_float64_biased_attention_oracle(self):
        attn = self._attention()
        rng = np.random.default_rng(1)
        q = rng.normal(size=(BATCH, QUERIES, 8)).astype("float32")
        k = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        v = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        bias = rng.normal(size=(BATCH, 2, QUERIES, KEYS)).astype("float32")
        measured = _np(attn(q, k, v, additive_bias=bias, training=False))
        kernels = [_dense(attn.q_proj), _dense(attn.k_proj),
                   _dense(attn.v_proj), _dense(attn.out_proj)]
        expected = _oracle_attention(_np(q), _np(k), _np(v), kernels, 2,
                                     bias=_np(bias))
        np.testing.assert_allclose(measured, expected, atol=TOL)

    def test_perturbing_one_heads_bias_moves_only_that_head(self):
        """The liveness oracle: the bias reaches the softmax, per head."""
        dim, heads = 8, 2
        attn = self._attention(heads, dim)
        # An identity output projection keeps each head in its own channels.
        attn.out_proj.kernel.assign(np.eye(dim, dtype="float32"))
        attn.out_proj.bias.assign(np.zeros(dim, dtype="float32"))
        rng = np.random.default_rng(2)
        q = rng.normal(size=(BATCH, QUERIES, dim)).astype("float32")
        k = rng.normal(size=(BATCH, KEYS, dim)).astype("float32")
        v = rng.normal(size=(BATCH, KEYS, dim)).astype("float32")
        bias = np.zeros((BATCH, heads, QUERIES, KEYS), dtype="float32")
        before = _np(attn(q, k, v, additive_bias=bias, training=False))
        # The perturbation must VARY along the key axis. A uniform shift is a
        # provable no-op -- softmax is shift-invariant along its reduction
        # axis -- and a first draft of this probe was blind for exactly that
        # reason (measured movement 2.09e-07, i.e. float32 noise).
        bias[:, 0, :, 0] += 2.5
        after = _np(attn(q, k, v, additive_bias=bias, training=False))
        head_dim = dim // heads
        moved = np.max(np.abs(after - before)[..., :head_dim])
        still = np.max(np.abs(after - before)[..., head_dim:])
        assert moved > 1e-3, "head 0's bias never reached the softmax"
        assert still < 1e-5, "head 1 moved under head 0's bias"

    def test_a_graded_positive_bias_is_not_a_no_op(self):
        """A binarizing mask helper leaves every positive entry untouched."""
        attn = self._attention()
        rng = np.random.default_rng(4)
        q = rng.normal(size=(BATCH, QUERIES, 8)).astype("float32")
        k = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        v = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        zero = np.zeros((BATCH, 2, QUERIES, KEYS), dtype="float32")
        graded = np.full((BATCH, 2, QUERIES, KEYS), 0.0, dtype="float32")
        graded[:, :, :, 0] = 1.75
        base = _np(attn(q, k, v, additive_bias=zero, training=False))
        biased = _np(attn(q, k, v, additive_bias=graded, training=False))
        assert np.max(np.abs(biased - base)) > 1e-3

    def test_a_small_negative_bias_is_not_a_hard_mask(self):
        """A binarizing mask helper turns -0.5 into a full -1e9 mask."""
        dim, heads = 8, 2
        attn = self._attention(heads, dim)
        attn.out_proj.kernel.assign(np.eye(dim, dtype="float32"))
        attn.out_proj.bias.assign(np.zeros(dim, dtype="float32"))
        rng = np.random.default_rng(6)
        q = rng.normal(size=(BATCH, QUERIES, dim)).astype("float32")
        k = rng.normal(size=(BATCH, KEYS, dim)).astype("float32")
        v = rng.normal(size=(BATCH, KEYS, dim)).astype("float32")
        soft = np.full((BATCH, heads, QUERIES, KEYS), -0.5, dtype="float32")
        hard = np.full((BATCH, heads, QUERIES, KEYS), -1e9, dtype="float32")
        hard[:, :, :, 0] = 0.0
        soft_out = _np(attn(q, k, v, additive_bias=soft, training=False))
        hard_out = _np(attn(q, k, v, additive_bias=hard, training=False))
        # A uniform additive bias is a no-op for softmax; a hard mask is not.
        zero_out = _np(attn(q, k, v, additive_bias=np.zeros_like(soft),
                            training=False))
        np.testing.assert_allclose(soft_out, zero_out, atol=TOL)
        assert np.max(np.abs(hard_out - zero_out)) > 1e-2

    def test_the_bias_is_optional(self):
        attn = self._attention()
        rng = np.random.default_rng(9)
        q = rng.normal(size=(BATCH, QUERIES, 8)).astype("float32")
        k = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        out = attn(q, k, k, training=False)
        assert tuple(out.shape) == (BATCH, QUERIES, 8)
        assert tuple(attn.compute_output_shape((BATCH, QUERIES, 8))) == \
               (BATCH, QUERIES, 8)

    def test_keys_and_values_are_separately_projected(self):
        """The single asymmetry that disqualifies a stock cross-attention."""
        attn = self._attention()
        rng = np.random.default_rng(12)
        q = rng.normal(size=(BATCH, QUERIES, 8)).astype("float32")
        k = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        v = rng.normal(size=(BATCH, KEYS, 8)).astype("float32")
        assert np.max(np.abs(_np(attn(q, k, v, training=False))
                             - _np(attn(q, v, v, training=False)))) > 1e-3
        assert attn.k_proj is not attn.v_proj


# ---------------------------------------------------------------------
# G6.4 -- the whole layer against the float64 reference forward
# ---------------------------------------------------------------------


class TestLayerForward:

    def test_output_shapes(self, layer):
        payload = _inputs()
        out, presence = layer(**payload, training=False)
        assert tuple(out.shape) == (BATCH, QUERIES, TINY["d_model"])
        assert tuple(presence.shape) == (BATCH, 1, TINY["d_model"])

    def test_matches_the_float64_reference_forward(self, layer):
        payload = _inputs()
        embed_x, embed_y = _rpb_mlps()
        bias = _np(_box_rpb_bias(_boxes(), (GRID_H, GRID_W), embed_x,
                                 embed_y, 2))
        out, presence = layer(**payload, image_cross_bias=bias.astype("f4"),
                              training=False)
        expected_out, expected_presence = _reference_layer_forward(
            layer, payload, bias=bias)
        np.testing.assert_allclose(_np(out), expected_out, atol=TOL)
        np.testing.assert_allclose(_np(presence), expected_presence, atol=TOL)

    def test_the_reference_forward_is_fed_a_non_constant_bias(self):
        """The oracle above is only independent of the LAYER, not of boxRPB.

        MEASURED: forcing the boxRPB embeddings to emit zeros leaves
        ``test_matches_the_float64_reference_forward`` GREEN, because both
        sides then receive the same zero bias. This guard makes that
        degeneracy detectable instead of silent.
        """
        embed_x, embed_y = _rpb_mlps()
        bias = _np(_box_rpb_bias(_boxes(), (GRID_H, GRID_W), embed_x,
                                 embed_y, 2))
        assert float(np.std(bias)) > 0.1
        assert len(np.unique(np.round(bias, 6))) > 10

    def test_matches_the_reference_without_a_presence_token(self, layer):
        payload = _inputs()
        payload.pop("presence_token")
        out, presence = layer(**payload, training=False)
        assert presence is None
        expected_out, _ = _reference_layer_forward(layer, payload,
                                                   with_presence=False)
        np.testing.assert_allclose(_np(out), expected_out, atol=TOL)

    def test_the_oracle_probes_are_unsaturated(self, layer):
        """The oracle's discriminating power, MEASURED not assumed.

        Two wrong candidates -- a bias that never reaches the scores, and a
        shifted query position -- must each miss the reference by far more than
        the tolerance. The margins are MEASURED (0.087 and 1.44 at this
        fixture) and pinned here, so a future author cannot loosen ``TOL``
        past the point where it would swallow a real defect. Note the terminal
        LayerNorm compresses every downstream difference, which is why 0.087
        rather than something of order 1 is the honest floor for the bias arm.
        """
        payload = _inputs()
        embed_x, embed_y = _rpb_mlps()
        bias = _np(_box_rpb_bias(_boxes(), (GRID_H, GRID_W), embed_x,
                                 embed_y, 2))
        reference, _ = _reference_layer_forward(layer, payload, bias=bias)

        unbiased, _ = _reference_layer_forward(layer, payload, bias=None)
        assert np.max(np.abs(unbiased - reference)) > 20 * TOL

        shifted = dict(payload)
        shifted["query_pos"] = payload["query_pos"] + 1.0
        moved, _ = _reference_layer_forward(layer, shifted, bias=bias)
        assert np.max(np.abs(moved - reference)) > 20 * TOL

    def test_the_text_padding_mask_changes_the_output(self, layer):
        payload = _inputs()
        masked, _ = layer(**payload, training=False)
        payload_open = dict(payload)
        payload_open["text_padding_mask"] = np.zeros((BATCH, TOKENS), bool)
        unmasked, _ = layer(**payload_open, training=False)
        assert np.max(np.abs(_np(masked) - _np(unmasked))) > 1e-3

    def test_memory_pos_is_added_to_keys_but_not_to_values(self, layer):
        """A stock cross-attention cannot express this; the private one can."""
        payload = _inputs()
        base, _ = layer(**payload, training=False)
        shifted = dict(payload)
        shifted["memory_pos"] = payload["memory_pos"] * 0.0
        assert np.max(np.abs(_np(base) - _np(layer(**shifted,
                                                  training=False)[0]))) > 1e-3

    def test_gradients_reach_every_sub_block(self, layer):
        import tensorflow as tf
        payload = _inputs()
        with tf.GradientTape() as tape:
            out, presence = layer(**payload, training=False)
            loss = ops.sum(out ** 2) + ops.sum(presence ** 2)
        grads = tape.gradient(loss, layer.trainable_weights)
        dead = [w.path for w, g in zip(layer.trainable_weights, grads)
                if g is None]
        assert dead == [], f"no gradient reaches {dead}"


# ---------------------------------------------------------------------
# G6.5 -- the presence token
# ---------------------------------------------------------------------


class TestPresenceToken:

    def test_presence_output_is_invariant_to_the_per_query_bias(self, layer):
        """The zero mask row: the presence token attends everywhere."""
        payload = _inputs()
        embed_x, embed_y = _rpb_mlps()
        bias_a = _np(_box_rpb_bias(_boxes(seed=11), (GRID_H, GRID_W),
                                   embed_x, embed_y, 2)).astype("f4")
        bias_b = _np(_box_rpb_bias(_boxes(seed=23), (GRID_H, GRID_W),
                                   embed_x, embed_y, 2)).astype("f4")
        _, presence_a = layer(**payload, image_cross_bias=bias_a,
                              training=False)
        _, presence_b = layer(**payload, image_cross_bias=bias_b,
                              training=False)
        assert np.max(np.abs(_np(presence_a) - _np(presence_b))) == 0.0

    def test_the_real_queries_do_move_with_that_bias(self, layer):
        """Positive liveness arm for the invariance above."""
        payload = _inputs()
        embed_x, embed_y = _rpb_mlps()
        bias_a = _np(_box_rpb_bias(_boxes(seed=11), (GRID_H, GRID_W),
                                   embed_x, embed_y, 2)).astype("f4")
        bias_b = _np(_box_rpb_bias(_boxes(seed=23), (GRID_H, GRID_W),
                                   embed_x, embed_y, 2)).astype("f4")
        out_a, _ = layer(**payload, image_cross_bias=bias_a, training=False)
        out_b, _ = layer(**payload, image_cross_bias=bias_b, training=False)
        assert np.max(np.abs(_np(out_a) - _np(out_b))) > 1e-3

    def test_presence_path_equals_a_manual_prepend_with_a_zero_pos_row(
            self, layer):
        """The presence token's query position is ZEROED, not learned."""
        payload = _inputs()
        out, presence = layer(**payload, training=False)

        manual = dict(payload)
        manual["tgt"] = np.concatenate(
            [payload["presence_token"], payload["tgt"]], axis=1)
        manual["query_pos"] = np.concatenate(
            [np.zeros_like(payload["presence_token"]), payload["query_pos"]],
            axis=1)
        manual.pop("presence_token")
        combined, none_token = layer(**manual, training=False)
        assert none_token is None
        np.testing.assert_allclose(_np(combined[:, 1:]), _np(out), atol=TOL)
        np.testing.assert_allclose(_np(combined[:, :1]), _np(presence),
                                   atol=TOL)

    def test_a_non_zero_presence_query_pos_would_be_measurably_different(
            self, layer):
        """The zeroing is not vacuous: a learned row changes the output."""
        payload = _inputs()
        out, presence = layer(**payload, training=False)
        manual = dict(payload)
        manual["tgt"] = np.concatenate(
            [payload["presence_token"], payload["tgt"]], axis=1)
        manual["query_pos"] = np.concatenate(
            [np.ones_like(payload["presence_token"]), payload["query_pos"]],
            axis=1)
        manual.pop("presence_token")
        combined, _ = layer(**manual, training=False)
        assert np.max(np.abs(_np(combined[:, :1]) - _np(presence))) > 1e-3
        assert np.max(np.abs(_np(combined[:, 1:]) - _np(out))) > 1e-3

    def test_the_presence_token_participates_in_self_attention(self, layer):
        """It is not a bystander: perturbing it moves the real queries."""
        payload = _inputs()
        base, _ = layer(**payload, training=False)
        moved = dict(payload)
        moved["presence_token"] = payload["presence_token"] + 2.0
        assert np.max(np.abs(_np(base)
                             - _np(layer(**moved, training=False)[0]))) > 1e-3

    def test_the_presence_token_is_split_off_after_the_feed_forward(self,
                                                                    layer):
        payload = _inputs()
        out, presence = layer(**payload, training=False)
        assert tuple(out.shape)[1] == QUERIES
        assert tuple(presence.shape)[1] == 1


# ---------------------------------------------------------------------
# G6.6 -- I-5, construction, serialization, structure
# ---------------------------------------------------------------------


class TestMemoryMaskRefusal:

    def test_an_external_memory_mask_is_refused(self, layer):
        payload = _inputs()
        with pytest.raises(ValueError, match="memory_mask"):
            layer(**payload, memory_mask=np.ones((BATCH, KEYS), "float32"),
                  image_cross_bias=None, training=False)

    def test_the_same_call_without_the_mask_succeeds(self, layer):
        """The refusal is about the mask, not about the rest of the payload."""
        out, presence = layer(**_inputs(), training=False)
        assert out is not None and presence is not None


class TestConstruction:

    @pytest.mark.parametrize("bad", [
        dict(d_model=0), dict(num_heads=0), dict(dim_feedforward=0),
        dict(d_model=9, num_heads=2), dict(dropout_rate=1.0),
        dict(dropout_rate=-0.1),
    ])
    def test_invalid_configuration_is_refused(self, bad):
        config = dict(TINY)
        config.update(bad)
        with pytest.raises(ValueError):
            Sam3DecoderLayer(**config)

    def test_missing_text_memory_is_refused(self, layer):
        payload = _inputs()
        payload.pop("memory_text")
        with pytest.raises(ValueError, match="memory_text"):
            layer(**payload, training=False)

    def test_wrong_width_is_refused_at_build(self):
        built = Sam3DecoderLayer(**TINY)
        with pytest.raises(ValueError, match="width"):
            built.build((BATCH, QUERIES, 9), (BATCH, KEYS, 8),
                        (BATCH, TOKENS, 8))

    def test_wrong_rank_is_refused_at_build(self):
        built = Sam3DecoderLayer(**TINY)
        with pytest.raises(ValueError, match="batch, seq"):
            built.build((BATCH, 8), (BATCH, KEYS, 8), (BATCH, TOKENS, 8))

    def test_text_cross_attention_requires_a_text_shape_at_build(self):
        built = Sam3DecoderLayer(**TINY)
        with pytest.raises(ValueError, match="memory_text_shape"):
            built.build((BATCH, QUERIES, 8), (BATCH, KEYS, 8))

    def test_tiny_parameter_count_matches_the_closed_form(self, layer):
        assert layer.count_params() == _params(TINY)

    def test_shipped_parameter_count_matches_the_closed_form(self):
        built = Sam3DecoderLayer(**SHIPPED)
        dim = SHIPPED["d_model"]
        built.build((1, 200, dim), (1, 72 * 72, dim), (1, 32, dim))
        assert built.count_params() == _params(SHIPPED)
        # The reference's own arithmetic: three 256-wide attentions at
        # 4 * (256*256 + 256) each for the two private ones, the stock text
        # block at the same total, a 256 -> 2048 -> 256 FFN and four norms.
        assert _params(SHIPPED) == 3 * 263168 + (256 * 2048 + 2048) + (
            2048 * 256 + 256) + 4 * 512

    def test_the_audit_is_not_vacuous(self):
        """Deleting the text sub-block must move the total, by its own size."""
        without = dict(TINY)
        without["use_text_cross_attention"] = False
        built = Sam3DecoderLayer(**without)
        built.build((BATCH, QUERIES, 8), (BATCH, KEYS, 8))
        assert built.count_params() == _params(without)
        assert _params(TINY) - _params(without) > 0

    def test_a_disabled_text_block_costs_exactly_zero_parameters(self):
        without = dict(TINY)
        without["use_text_cross_attention"] = False
        built = Sam3DecoderLayer(**without)
        built.build((BATCH, QUERIES, 8), (BATCH, KEYS, 8))
        assert built.ca_text.weights == []
        assert built.catext_norm.weights == []

    def test_the_normalization_epsilon_is_the_reference_value(self, layer):
        """Keras defaults LayerNormalization to 1e-3; the reference is 1e-5."""
        for norm in (layer.norm1, layer.norm2, layer.norm3, layer.catext_norm):
            assert norm.epsilon == 1e-5

    def test_the_feed_forward_activation_is_relu_not_gelu(self, layer):
        assert layer.ffn.activation_name == "relu"
        assert layer.ffn.activation_fn is keras.activations.relu

    def test_compute_output_shape_is_derived_from_config(self, layer):
        shapes = layer.compute_output_shape((BATCH, QUERIES, 8))
        assert shapes == ((BATCH, QUERIES, 8), (BATCH, 1, 8))


class TestSerialization:

    def test_config_keys_equal_init_signature(self, layer):
        import inspect
        expected = {name for name in
                    inspect.signature(Sam3DecoderLayer.__init__).parameters
                    if name not in ("self", "kwargs")}
        missing = expected - set(layer.get_config())
        assert missing == set(), f"get_config() is missing {sorted(missing)}"

    def test_round_trip_by_value(self, layer):
        payload = _inputs()
        out, presence = layer(**payload, training=False)
        clone = Sam3DecoderLayer.from_config(layer.get_config())
        clone.build((BATCH, QUERIES, 8), (BATCH, KEYS, 8), (BATCH, TOKENS, 8))
        clone.set_weights(layer.get_weights())
        clone_out, clone_presence = clone(**payload, training=False)
        np.testing.assert_allclose(_np(out), _np(clone_out), atol=1e-6)
        np.testing.assert_allclose(_np(presence), _np(clone_presence),
                                   atol=1e-6)

    def test_keras_model_round_trip_by_value(self, layer, tmp_path):
        """D-098: a round trip must be checked by VALUE, never by weight count.

        A nested sub-layer store restores FRESHLY INITIALIZED kernels while the
        weight count, the weight paths and the parameter total all match. This
        layer stores its sub-layers FLAT for exactly that reason.
        """
        payload = _inputs()
        dim = TINY["d_model"]
        tgt_in = keras.Input(shape=(QUERIES, dim))
        memory_in = keras.Input(shape=(KEYS, dim))
        text_in = keras.Input(shape=(TOKENS, dim))
        presence_in = keras.Input(shape=(1, dim))
        out, presence = layer(tgt_in, memory_in, memory_text=text_in,
                              presence_token=presence_in)
        model = keras.Model([tgt_in, memory_in, text_in, presence_in],
                            [out, presence])
        probe = [payload["tgt"], payload["memory"], payload["memory_text"],
                 payload["presence_token"]]
        before = [np.asarray(t) for t in model.predict(probe, verbose=0)]
        path = str(tmp_path / "decoder.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        after = [np.asarray(t) for t in restored.predict(probe, verbose=0)]
        for lhs, rhs in zip(before, after):
            assert np.max(np.abs(lhs - rhs)) == 0.0

    def test_the_round_trip_guard_can_see_a_difference(self, layer):
        """The comparator is RED-proven before the exact-zero PASS is trusted."""
        payload = _inputs()
        out, _ = layer(**payload, training=False)
        layer.norm3.gamma.assign(layer.norm3.gamma + 0.5)
        moved, _ = layer(**payload, training=False)
        assert np.max(np.abs(_np(out) - _np(moved))) > 1e-3


# =====================================================================
# Step 7 -- `Sam3TransformerDecoder`, the layer stack
#
# The oracle below re-derives the WHOLE stack in float64 -- the per-layer
# conditional query position, boxRPB's bias, all four sub-blocks of every
# layer, the terminal normalization, the logit-space box refinement and the
# presence readout -- from the stack's WEIGHTS only. It never calls the stack.
#
# Two absence-shaped assertions live here and both are flanked by positive
# arms, because a dead box head satisfies them by construction:
# `test_layer_zero_leaves_its_reference_box_exactly_where_it_was` (the delta is
# zero at init) and
# `test_a_later_layers_hidden_state_has_no_gradient_to_the_box_head` (the
# detach). The liveness arms are named at each assertion site.
# =====================================================================

STACK_LAYERS = 3
STACK = dict(d_model=8, num_heads=2, num_layers=STACK_LAYERS,
             num_queries=QUERIES, feat_size=(GRID_H, GRID_W),
             dim_feedforward=16, dropout_rate=0.0, activation="relu",
             use_text_cross_attention=True, box_rpb="log",
             use_presence_token=True, clamp_presence_logits=True,
             clamp_presence_logit_max_val=10.0,
             use_normed_output_consistently=True, norm_epsilon=1e-5)

# Re-read from the pinned clone's `_create_transformer_decoder()`: 6 layers,
# 200 queries, d_model 256, boxRPB "log", presence token on, and
# `resolution=1008, stride=14` -> a 72 x 72 image-memory grid.
SHIPPED_STACK = dict(d_model=256, num_heads=8, num_layers=6, num_queries=200,
                     feat_size=(72, 72), dim_feedforward=2048,
                     dropout_rate=0.1, box_rpb="log")


def _stack_inputs(seed: int = 21) -> dict:
    """A full call payload for the stack."""
    rng = np.random.default_rng(seed)
    dim = STACK["d_model"]
    padding = np.zeros((BATCH, TOKENS), dtype=bool)
    padding[0, TOKENS - 2:] = True
    return dict(
        memory=rng.normal(size=(BATCH, KEYS, dim)).astype("f4"),
        memory_text=rng.normal(size=(BATCH, TOKENS, dim)).astype("f4"),
        memory_pos=rng.normal(size=(BATCH, KEYS, dim)).astype("f4"),
        text_padding_mask=padding,
    )


def _build_stack(randomize: bool = True, seed: int = 3, **overrides):
    """Construct, build and (optionally) give every weight a real value."""
    config = dict(STACK)
    config.update(overrides)
    built = Sam3TransformerDecoder(**config)
    built.build((BATCH, KEYS, config["d_model"]),
                (BATCH, TOKENS, config["d_model"]))
    if randomize:
        rng = np.random.default_rng(seed)
        for weight in built.weights:
            weight.assign(
                rng.normal(0.0, 0.4, size=weight.shape).astype("float32"))
    return built


@pytest.fixture()
def stack():
    """A stack whose box head is still ZERO-initialized (the shipped init)."""
    return _build_stack(randomize=False)


@pytest.fixture()
def trained_stack():
    """A stack whose every weight -- box head included -- is non-trivial."""
    return _build_stack(randomize=True)


# ---------------------------------------------------------------------
# float64 stack oracle
# ---------------------------------------------------------------------


class _MlpView:
    """Adapt a flat ``List[Dense]`` to the ``.layers`` shape `_oracle_rpb` wants."""

    def __init__(self, stack_list):
        self.layers = stack_list


def _oracle_mlp(stack_list, x: np.ndarray) -> np.ndarray:
    """A flat MLP stack in float64; ReLU on every projection but the last."""
    for index, dense in enumerate(stack_list):
        kernel, bias = _dense(dense)
        x = x @ kernel + bias
        if index < len(stack_list) - 1:
            x = np.maximum(x, 0.0)
    return x


def _oracle_sine_embed(boxes: np.ndarray, d_model: int) -> np.ndarray:
    """The reference's 4-scalar box sine embedding, in ``y, x, w, h`` order."""
    num_feats = d_model // 2
    ladder = np.arange(num_feats // 2, dtype=np.float64)
    dim_t = 10000.0 ** (2.0 * ladder / num_feats)
    parts = []
    for axis in (1, 0, 2, 3):
        scaled = boxes[..., axis:axis + 1] * (2.0 * np.pi) / dim_t
        pair = np.stack([np.sin(scaled), np.cos(scaled)], axis=-1)
        parts.append(pair.reshape(boxes.shape[0], boxes.shape[1], num_feats))
    return np.concatenate(parts, axis=-1)


def _oracle_inverse_sigmoid(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.log(np.maximum(x, eps) / np.maximum(1.0 - x, eps))


def _oracle_stack_forward(obj, payload, boxes, use_normed=True, detach=True):
    """The whole decoder stack in float64, from the stack's WEIGHTS only.

    ``detach`` has no effect on the VALUES -- it is the gradient path that the
    detach governs -- so it is not a parameter here; the gradient guards probe
    it directly with a tape instead.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    presence = np.broadcast_to(_np(obj.presence_token)[None],
                               (BATCH, 1, obj.d_model)).copy()
    output = np.broadcast_to(_np(obj.query_embed)[None],
                             (BATCH, obj.num_queries, obj.d_model)).copy()
    hidden, refs, logits = [], [boxes], []
    gamma, beta = _np(obj.norm.gamma), _np(obj.norm.beta)
    pgamma = _np(obj.presence_token_out_norm.gamma)
    pbeta = _np(obj.presence_token_out_norm.beta)

    for index in range(obj.num_layers):
        query_pos = _oracle_mlp(obj.ref_point_head,
                                _oracle_sine_embed(boxes, obj.d_model))
        bias = _oracle_rpb(boxes, obj.feat_size,
                           _MlpView(obj.box_rpb_embed_x),
                           _MlpView(obj.box_rpb_embed_y), obj.num_heads,
                           mode=obj.box_rpb)
        layer_payload = dict(tgt=output, query_pos=query_pos,
                             memory=payload["memory"],
                             memory_pos=payload["memory_pos"],
                             memory_text=payload["memory_text"],
                             text_padding_mask=payload["text_padding_mask"],
                             presence_token=presence)
        output, presence = _reference_layer_forward(
            obj.decoder_layers[index], layer_payload, bias=bias,
            with_presence=True)

        normed = _oracle_layer_norm(output, gamma, beta, obj.norm_epsilon)
        delta = _oracle_mlp(obj.bbox_embed, normed if use_normed else output)
        refined = 1.0 / (1.0 + np.exp(-(delta + _oracle_inverse_sigmoid(boxes))))
        boxes = refined
        if index != obj.num_layers - 1:
            refs.append(refined)
        hidden.append(normed)

        raw = _oracle_mlp(obj.presence_token_head,
                          _oracle_layer_norm(presence, pgamma, pbeta,
                                             obj.norm_epsilon))[..., 0]
        if obj.clamp_presence_logits:
            raw = np.clip(raw, -obj.clamp_presence_logit_max_val,
                          obj.clamp_presence_logit_max_val)
        logits.append(raw)

    return np.stack(hidden), np.stack(refs), np.stack(logits)


class TestStackConstruction:

    @pytest.mark.parametrize("bad", [
        dict(d_model=0), dict(num_heads=0), dict(num_layers=0),
        dict(num_queries=0), dict(box_rpb="quadratic"), dict(d_model=7),
        dict(feat_size=(3,)), dict(feat_size=(0, 4)),
    ])
    def test_invalid_configuration_is_refused(self, bad):
        config = dict(STACK)
        config.update(bad)
        with pytest.raises(ValueError):
            Sam3TransformerDecoder(**config)

    def test_a_memory_whose_key_count_contradicts_feat_size_is_refused(self):
        """boxRPB's bias is built on `feat_size`; a mismatch is silent."""
        built = Sam3TransformerDecoder(**STACK)
        with pytest.raises(ValueError, match="wrong-geometry|implies"):
            built.build((BATCH, KEYS + 1, STACK["d_model"]),
                        (BATCH, TOKENS, STACK["d_model"]))

    def test_wrong_memory_width_is_refused(self):
        built = Sam3TransformerDecoder(**STACK)
        with pytest.raises(ValueError, match="d_model"):
            built.build((BATCH, KEYS, STACK["d_model"] + 2),
                        (BATCH, TOKENS, STACK["d_model"]))

    def test_wrong_memory_rank_is_refused(self):
        built = Sam3TransformerDecoder(**STACK)
        with pytest.raises(ValueError, match="batch, keys, d_model"):
            built.build((BATCH, KEYS), (BATCH, TOKENS, STACK["d_model"]))

    def test_a_disabled_presence_token_costs_zero_parameters(self):
        with_presence = _build_stack(randomize=False)
        without = _build_stack(randomize=False, use_presence_token=False)
        head = 2 * (8 * 8 + 8) + (8 * 1 + 1)      # 3-layer MLP, d_model 8
        expected = head + 2 * 8 + 8               # + out-norm + the token
        assert (with_presence.count_params()
                - without.count_params()) == expected

    def test_a_disabled_presence_token_returns_none_for_both_outputs(self):
        built = _build_stack(use_presence_token=False)
        _, _, logits, feats = built(**_stack_inputs(), training=False)
        assert logits is None and feats is None

    def test_compute_output_shape_is_derived_from_config(self, trained_stack):
        payload = _stack_inputs()
        declared = trained_stack.compute_output_shape(
            (BATCH, KEYS, STACK["d_model"]))
        produced = trained_stack(**payload, training=False)
        for lhs, rhs in zip(declared, produced):
            assert tuple(lhs) == tuple(rhs.shape)


class TestStackForward:

    def test_every_per_layer_output_is_stacked_on_a_leading_layer_axis(
            self, trained_stack):
        hidden, boxes, logits, feats = trained_stack(**_stack_inputs(),
                                                     training=False)
        dim, layers_n = STACK["d_model"], STACK_LAYERS
        assert tuple(hidden.shape) == (layers_n, BATCH, QUERIES, dim)
        assert tuple(boxes.shape) == (layers_n, BATCH, QUERIES, 4)
        assert tuple(logits.shape) == (layers_n, BATCH, 1)
        assert tuple(feats.shape) == (BATCH, 1, dim)

    def test_matches_the_float64_stack_oracle(self, trained_stack):
        """The oracle re-derives the whole stack; it never calls the stack.

        This is also the guard that dies when the box head is killed: the
        oracle computes the TRUE delta from the head's weights, so a dead head
        disagrees with it at every layer after the first.
        """
        payload = _stack_inputs()
        boxes_in = _boxes()
        hidden, refs, logits, _ = trained_stack(
            **payload, reference_boxes=boxes_in, training=False)
        want_hidden, want_refs, want_logits = _oracle_stack_forward(
            trained_stack, payload, boxes_in)
        np.testing.assert_allclose(_np(hidden), want_hidden, atol=TOL)
        np.testing.assert_allclose(_np(refs), want_refs, atol=TOL)
        np.testing.assert_allclose(_np(logits), want_logits, atol=TOL)

    def test_the_oracle_probes_are_unsaturated_so_the_chain_is_visible(
            self, trained_stack):
        """A stack whose references never move would pass a vacuous oracle."""
        payload = _stack_inputs()
        _, refs, _ = _oracle_stack_forward(trained_stack, payload, _boxes())
        assert np.max(np.abs(refs[-1] - refs[0])) > 100 * TOL

    def test_the_reference_stack_holds_one_box_per_layer(self, trained_stack):
        """`num_layers` boxes: the initial one plus every refinement but the
        last. The final layer's refinement is deliberately dropped, so entry
        `k` is the box layer `k` actually consumed."""
        _, refs, _, _ = trained_stack(**_stack_inputs(),
                                      reference_boxes=_boxes(), training=False)
        assert refs.shape[0] == STACK_LAYERS
        np.testing.assert_allclose(_np(refs[0]), _boxes().astype(np.float64),
                                   atol=1e-6)

    def test_the_default_reference_is_the_sigmoid_of_the_learned_points(
            self, trained_stack):
        _, refs, _, _ = trained_stack(**_stack_inputs(), training=False)
        want = 1.0 / (1.0 + np.exp(-_np(trained_stack.reference_points)))
        np.testing.assert_allclose(_np(refs[0])[0], want, atol=1e-6)

    def test_the_default_query_is_the_learned_query_embedding(
            self, trained_stack):
        """Feeding `query_embed` explicitly must reproduce the default path."""
        payload = _stack_inputs()
        default, _, _, _ = trained_stack(**payload, training=False)
        explicit_tgt = np.broadcast_to(
            _np(trained_stack.query_embed)[None],
            (BATCH, QUERIES, STACK["d_model"])).astype("float32")
        explicit, _, _, _ = trained_stack(**payload, tgt=explicit_tgt,
                                          training=False)
        assert np.max(np.abs(_np(default) - _np(explicit))) == 0.0

    def test_the_reference_boxes_actually_move_across_layers(
            self, trained_stack):
        """POSITIVE liveness arm for the dead-box-head probe."""
        _, refs, _, _ = trained_stack(**_stack_inputs(),
                                      reference_boxes=_boxes(), training=False)
        refs = _np(refs)
        assert np.max(np.abs(refs[-1] - refs[0])) > 1e-3

    def test_every_layer_sees_a_different_reference_box(self, trained_stack):
        """POSITIVE liveness arm: a dead head makes every entry identical."""
        _, refs, _, _ = trained_stack(**_stack_inputs(),
                                      reference_boxes=_boxes(), training=False)
        refs = _np(refs)
        for index in range(1, STACK_LAYERS):
            assert np.max(np.abs(refs[index] - refs[index - 1])) > 1e-4

    def test_the_conditional_query_position_is_rebuilt_from_the_new_box(
            self, trained_stack):
        """POSITIVE liveness arm: two different initial boxes must give two
        different hidden states at EVERY layer, including the last."""
        payload = _stack_inputs()
        first, _, _, _ = trained_stack(**payload, reference_boxes=_boxes(11),
                                       training=False)
        second, _, _, _ = trained_stack(**payload, reference_boxes=_boxes(19),
                                        training=False)
        moved = np.abs(_np(first) - _np(second)).max(axis=(1, 2, 3))
        assert np.all(moved > 1e-3), moved


class TestBoxRefinementIdentity:
    """M7.2's family: the zero-initialized box head."""

    def test_the_box_heads_last_projection_is_zero_initialized(self, stack):
        assert np.max(np.abs(_np(stack.bbox_embed[-1].kernel))) == 0.0
        assert np.max(np.abs(_np(stack.bbox_embed[-1].bias))) == 0.0

    def test_layer_zero_leaves_its_reference_box_exactly_where_it_was(
            self, stack):
        """The layer-0 identity.

        VACUITY NOTE: this assertion has an ABSENCE shape -- a box head that
        emits zeros satisfies it by construction, and the dead-component probe
        confirmed it survives that kill. Its liveness arm is the next test.
        """
        _, refs, _, _ = stack(**_stack_inputs(), reference_boxes=_boxes(),
                              training=False)
        refs = _np(refs)
        np.testing.assert_allclose(refs[1], refs[0], atol=1e-6)

    def test_a_non_zero_box_head_would_move_the_layer_zero_box(self):
        """The comparator for the identity assertion, RED-proven."""
        moved = _build_stack(randomize=False)
        moved.bbox_embed[-1].kernel.assign(
            np.full(moved.bbox_embed[-1].kernel.shape, 0.3, dtype="float32"))
        _, refs, _, _ = moved(**_stack_inputs(), reference_boxes=_boxes(),
                              training=False)
        refs = _np(refs)
        assert np.max(np.abs(refs[1] - refs[0])) > 1e-2


class TestReferenceChainIsDetached:
    """M7.1's family: the reference chain carries no gradient across layers."""

    def _tape(self, built, pick):
        import tensorflow as tf
        payload = _stack_inputs()
        with tf.GradientTape() as tape:
            hidden, refs, _, _ = built(**payload,
                                       reference_boxes=_boxes(),
                                       training=False)
            loss = tf.reduce_sum(pick(hidden, refs))
        return tape.gradient(loss, built.bbox_embed[-1].trainable_weights)

    def test_a_later_layers_hidden_state_has_no_gradient_to_the_box_head(
            self, trained_stack):
        """The box head reaches layer 2 ONLY through the reference chain, and
        that chain is detached -- so the gradient is `None`, not merely small.

        VACUITY NOTE: a dead box head also yields `None` here. The two tests
        below are the positive arms that separate the two.
        """
        grads = self._tape(trained_stack, lambda hidden, refs: hidden[2])
        assert all(g is None for g in grads), \
            "the reference chain is NOT detached: a layer-2 loss reached the " \
            "layer-0/1 box head"

    def test_the_box_head_does_receive_gradient_from_its_own_refinement(
            self, trained_stack):
        """POSITIVE arm: the head is alive, it is only the CHAIN that is cut."""
        grads = self._tape(trained_stack, lambda hidden, refs: refs[1])
        assert all(g is not None for g in grads)
        assert max(float(np.max(np.abs(_np(g)))) for g in grads) > 0.0

    def test_the_gradient_probe_can_see_a_connected_path(self, trained_stack):
        """POSITIVE arm: the same layer-2 loss DOES reach a decoder layer."""
        import tensorflow as tf
        payload = _stack_inputs()
        target = trained_stack.decoder_layers[0].norm3.gamma
        with tf.GradientTape() as tape:
            hidden, _, _, _ = trained_stack(**payload,
                                            reference_boxes=_boxes(),
                                            training=False)
            loss = tf.reduce_sum(hidden[2])
        grad = tape.gradient(loss, [target])[0]
        assert grad is not None and float(np.max(np.abs(_np(grad)))) > 0.0


class TestDeltaReadsTheNormedState:
    """M7.3's family."""

    def test_the_delta_is_computed_on_the_normed_hidden_state(
            self, trained_stack):
        payload = _stack_inputs()
        _, refs, _, _ = trained_stack(**payload, reference_boxes=_boxes(),
                                      training=False)
        _, want, _ = _oracle_stack_forward(trained_stack, payload, _boxes(),
                                           use_normed=True)
        np.testing.assert_allclose(_np(refs), want, atol=TOL)

    def test_the_raw_state_candidate_is_measurably_different(
            self, trained_stack):
        """The wrong candidate's margin is PINNED, so `atol` cannot hide it."""
        payload = _stack_inputs()
        _, want_normed, _ = _oracle_stack_forward(trained_stack, payload,
                                                  _boxes(), use_normed=True)
        _, want_raw, _ = _oracle_stack_forward(trained_stack, payload,
                                               _boxes(), use_normed=False)
        assert np.max(np.abs(want_normed - want_raw)) > 100 * TOL

    def test_the_raw_state_branch_reproduces_the_raw_state_oracle(self):
        """Both branches are live; the flag is not decorative."""
        built = _build_stack(use_normed_output_consistently=False)
        payload = _stack_inputs()
        _, refs, _, _ = built(**payload, reference_boxes=_boxes(),
                              training=False)
        _, want, _ = _oracle_stack_forward(built, payload, _boxes(),
                                           use_normed=False)
        np.testing.assert_allclose(_np(refs), want, atol=TOL)


class TestPresenceClamp:
    """M7.4's family. The probe points are what make the bound visible."""

    @staticmethod
    def _with_presence_bias(value: float, **overrides):
        built = _build_stack(**overrides)
        built.presence_token_head[-1].bias.assign(
            np.array([value], dtype="float32"))
        built.presence_token_head[-1].kernel.assign(
            np.zeros(built.presence_token_head[-1].kernel.shape, "float32"))
        return built

    def test_a_logit_far_beyond_the_bound_is_clamped(self):
        built = self._with_presence_bias(25.0)
        _, _, logits, _ = built(**_stack_inputs(), training=False)
        assert np.max(np.abs(_np(logits) - 10.0)) < 1e-6

    def test_the_negative_side_is_clamped_too(self):
        built = self._with_presence_bias(-25.0)
        _, _, logits, _ = built(**_stack_inputs(), training=False)
        assert np.max(np.abs(_np(logits) + 10.0)) < 1e-6

    def test_this_bound_is_ten_and_not_the_scorers_twelve(self):
        """The MANDATED (10, 12] probe.

        `Sam3DotProductScoring` clamps at 12.0. Any probe whose magnitude stays
        under 10 cannot tell the two bounds apart -- the coincidence point that
        has eaten five guards in this build.
        """
        built = self._with_presence_bias(11.5)
        _, _, logits, _ = built(**_stack_inputs(), training=False)
        np.testing.assert_allclose(_np(logits), 10.0, atol=1e-6)

    def test_a_logit_inside_the_bound_is_untouched(self):
        built = self._with_presence_bias(4.25)
        _, _, logits, _ = built(**_stack_inputs(), training=False)
        np.testing.assert_allclose(_np(logits), 4.25, atol=1e-5)

    def test_the_unclamped_configuration_returns_the_raw_magnitude(self):
        """POSITIVE arm: it pins the wrong candidate's margin at 25 vs 10."""
        built = self._with_presence_bias(25.0, clamp_presence_logits=False)
        _, _, logits, _ = built(**_stack_inputs(), training=False)
        assert float(np.max(_np(logits))) > 24.0


class TestStackParameterAudit:

    def test_tiny_parameter_count_matches_the_closed_form(self, stack):
        dim, heads = STACK["d_model"], STACK["num_heads"]
        per_layer = _params(dict(d_model=dim, dim_feedforward=16,
                                 use_text_cross_attention=True))
        dense = dim * dim + dim
        expected = (
            STACK_LAYERS * per_layer                 # the decoder layers
            + 2 * dim                                # terminal norm
            + 2 * dense + (dim * 4 + 4)              # bbox_embed
            + (2 * dim * dim + dim) + dense          # ref_point_head
            + 2 * ((2 * dim + dim) + (dim * heads + heads))   # boxRPB x and y
            + 2 * dim                                # presence out-norm
            + 2 * dense + (dim + 1)                  # presence head
            + STACK["num_queries"] * dim             # query_embed
            + STACK["num_queries"] * 4               # reference_points
            + dim                                    # presence token
        )
        assert stack.count_params() == expected

    def test_shipped_parameter_count_matches_the_closed_form(self):
        """The settled 6-layer / 256-wide / 200-query stack, INSTANTIATED."""
        built = Sam3TransformerDecoder(**SHIPPED_STACK)
        built.build((1, 72 * 72, 256), (1, 32, 256))
        dim, heads, queries = 256, 8, 200
        per_layer = _params(dict(d_model=256, dim_feedforward=2048,
                                 use_text_cross_attention=True))
        dense = dim * dim + dim
        expected = (
            6 * per_layer + 2 * dim
            + 2 * dense + (dim * 4 + 4)
            + (2 * dim * dim + dim) + dense
            + 2 * ((2 * dim + dim) + (dim * heads + heads))
            + 2 * dim + 2 * dense + (dim + 1)
            + queries * dim + queries * 4 + dim
        )
        assert built.count_params() == expected == 11_575_093

    def test_the_audit_is_not_vacuous(self):
        """Deleting a whole component must break the closed form."""
        built = Sam3TransformerDecoder(**dict(STACK, num_layers=STACK_LAYERS - 1))
        built.build((BATCH, KEYS, STACK["d_model"]),
                    (BATCH, TOKENS, STACK["d_model"]))
        reference = _build_stack(randomize=False)
        assert built.count_params() != reference.count_params()


class TestStackSerialization:

    def test_config_keys_equal_init_signature(self, stack):
        import inspect
        expected = {name for name in
                    inspect.signature(
                        Sam3TransformerDecoder.__init__).parameters
                    if name not in ("self", "kwargs")}
        missing = expected - set(stack.get_config())
        assert missing == set(), f"get_config() is missing {sorted(missing)}"

    def test_round_trip_by_value(self, trained_stack):
        payload = _stack_inputs()
        produced = trained_stack(**payload, training=False)
        clone = Sam3TransformerDecoder.from_config(trained_stack.get_config())
        clone.build((BATCH, KEYS, STACK["d_model"]),
                    (BATCH, TOKENS, STACK["d_model"]))
        clone.set_weights(trained_stack.get_weights())
        restored = clone(**payload, training=False)
        for lhs, rhs in zip(produced, restored):
            np.testing.assert_allclose(_np(lhs), _np(rhs), atol=1e-6)

    def test_keras_model_round_trip_by_value(self, trained_stack, tmp_path):
        """D-098: a stack of per-layer sub-layers is EXACTLY the shape that
        loses its weights silently when stored nested. Every store here is
        flat, and this checks VALUES, never counts."""
        payload = _stack_inputs()
        dim = STACK["d_model"]
        memory_in = keras.Input(shape=(KEYS, dim))
        text_in = keras.Input(shape=(TOKENS, dim))
        outputs = trained_stack(memory_in, memory_text=text_in)
        model = keras.Model([memory_in, text_in], list(outputs))
        probe = [payload["memory"], payload["memory_text"]]
        before = [np.asarray(t) for t in model.predict(probe, verbose=0)]
        path = str(tmp_path / "stack.keras")
        model.save(path)
        after = [np.asarray(t) for t in
                 keras.models.load_model(path).predict(probe, verbose=0)]
        for lhs, rhs in zip(before, after):
            assert np.max(np.abs(lhs - rhs)) == 0.0

    def test_the_round_trip_guard_can_see_a_difference(self, trained_stack):
        """The comparator is RED-proven before the exact-zero PASS is trusted."""
        payload = _stack_inputs()
        hidden, _, _, _ = trained_stack(**payload, training=False)
        trained_stack.norm.gamma.assign(trained_stack.norm.gamma + 0.5)
        moved, _, _, _ = trained_stack(**payload, training=False)
        assert np.max(np.abs(_np(hidden) - _np(moved))) > 1e-3


class TestPresenceTokenInitScale:
    """D-137: the reference's global xavier pass reaches `presence_token`.

    `TransformerWrapper._reset_parameters` (`sam3/model/model_misc.py:846-854`
    at the pinned SHA) xavier-uniform-initializes every `dim > 1` parameter
    except names holding `box_embed` / `query_embed` / `reference_points`, and
    the image path constructs that wrapper (`model_builder.py:528-536`).
    `presence_token` is NOT on the exclusion list; its two neighbours ARE. The
    port shipped all three at `RandomNormal(stddev=1.0)`, an 11.3x scale
    divergence on the model's only presence signal.
    """

    @staticmethod
    def _xavier_std(fan_in: int, fan_out: int) -> float:
        return math.sqrt(6.0 / (fan_in + fan_out)) / math.sqrt(3.0)

    @staticmethod
    def _presence_draws(d_model: int, stacks: int, defect: bool = False):
        """Collect `presence_token` from `stacks` freshly BUILT stacks.

        `defect=True` overwrites each built weight with the pre-fix
        `RandomNormal(stddev=1.0)` draw, so the RED arm below runs through
        `Sam3TransformerDecoder` rather than through a bare initializer.
        """
        draws = []
        for seed in range(stacks):
            keras.utils.set_random_seed(seed)
            built = _build_stack(randomize=False, d_model=d_model,
                                 dim_feedforward=32)
            if defect:
                built.presence_token.assign(
                    keras.initializers.RandomNormal(stddev=1.0)(
                        built.presence_token.shape)
                )
            draws.append(_np(built.presence_token).ravel())
        return np.concatenate(draws)

    def _assert_at_xavier_scale(self, sample, d_model: int) -> None:
        """The load-bearing assertion, shared by the GREEN and RED arms."""
        expected = self._xavier_std(d_model, 1)
        assert float(sample.std()) == pytest.approx(expected, rel=0.06)
        # The pre-fix value is 1.0; this must be nowhere near it.
        assert float(sample.std()) < 0.3

    def test_the_presence_token_is_at_xavier_scale_not_unit_normal(self):
        """A 4,096-draw estimate, so the assertion is on the DISTRIBUTION."""
        d_model = 64
        sample = self._presence_draws(d_model, 64)
        assert sample.size == 64 * d_model
        self._assert_at_xavier_scale(sample, d_model)

    def test_the_unit_normal_candidate_is_measurably_different(self):
        """RED proof: the DEFECT is constructed IN THE LAYER and fails.

        Earlier this arm drew from `RandomNormal(stddev=1.0)` directly, which
        proved a fact about the initializer rather than about
        `Sam3TransformerDecoder` -- it never re-entered `build`, so it could not
        notice the weight's shape moving (which moves the fans and the expected
        std together). Now 16 real stacks are built and each built
        `presence_token` is overwritten with the pre-fix draw, and the SAME
        assertion the green arm uses is required to raise.
        """
        d_model = 64
        defect = self._presence_draws(d_model, 16, defect=True)
        assert defect.size == 16 * d_model
        expected = self._xavier_std(d_model, 1)
        assert float(defect.std()) == pytest.approx(1.0, rel=0.06)
        # 5.7x at this probe width; 11.3x at the shipped d_model=256, which
        # the closed-form test below pins.
        assert float(defect.std()) / expected > 5.0
        with pytest.raises(AssertionError):
            self._assert_at_xavier_scale(defect, d_model)

    def test_the_two_excluded_neighbours_stay_unit_normal(self):
        """`query_embed` and `reference_points` ARE on the exclusion list.

        The reference keeps both at `nn.Embedding`'s unit normal, so the port's
        `RandomNormal(stddev=1.0)` is correct for them and the asymmetry with
        `presence_token` is the reference's, not a mistake.
        """
        d_model = 64
        q, r = [], []
        for seed in range(32):
            keras.utils.set_random_seed(seed)
            built = _build_stack(randomize=False, d_model=d_model,
                                 dim_feedforward=32)
            q.append(_np(built.query_embed).ravel())
            r.append(_np(built.reference_points).ravel())
        assert float(np.concatenate(q).std()) == pytest.approx(1.0, rel=0.06)
        assert float(np.concatenate(r).std()) == pytest.approx(1.0, rel=0.06)

    def test_the_shipped_width_scale_is_the_pinned_number(self):
        """d_model=256: xavier std 0.088216, versus the pre-fix 1.0."""
        assert self._xavier_std(256, 1) == pytest.approx(0.08821622, abs=1e-7)
        assert 1.0 / self._xavier_std(256, 1) == pytest.approx(11.336, rel=1e-3)
