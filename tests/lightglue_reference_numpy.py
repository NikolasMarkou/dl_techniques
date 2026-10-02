"""Explicit-loop numpy transcription of the official LightGlue reference code.

Source transcribed: ``cvg/LightGlue`` ``lightglue/lightglue.py`` (arXiv:2306.13643),
fetched 2026-10-02. Torch is not installed in this repository's ``.venv``, so this module is
the parity ground truth for every LightGlue layer test (plan
``plan-2026-10-02T084508-dd2c07ac``, decision D-002).

**Why it is written as loops.** The Keras code under test is vectorised. An oracle written in
the same vectorised style would share its misreadings (axis order, tile versus repeat,
softmax axis). Here every pair, head and token is visited by an explicit Python loop and the
convention-bearing steps (rotate_half, the (H, Dh, 3) qkv split, the double softmax, the
mutual check) are spelled one element at a time. It is slow and only meant for tiny shapes.

**Conventions (all taken verbatim from the reference).**

- ``Linear`` weights are torch layout ``(out, in)``: ``y = x @ W.T + b``.
- Weight dictionaries use the torch ``state_dict`` names, e.g.
  ``transformers.0.self_attn.Wqkv.weight``.
- ``Wqkv`` output columns are ordered ``(num_heads, head_dim, 3)``, i.e. column
  ``h * (Dh * 3) + d * 3 + s`` is head ``h``, channel ``d``, slot ``s`` in ``{q, k, v}``.
- Rotary is INTERLEAVED adjacent pairs: the table is ``repeat_interleave(2)``
  ([c0, c0, c1, c1, ...]), and ``rotate_half`` maps every pair ``(a, b)`` to ``(-b, a)``.
- LayerNorm epsilon is 1e-5, GELU is the exact erf form.

All math is float64. Functions are single-sample (no batch axis) unless the name says
``batched``. Masks are optional boolean keep-matrices; the oracle needs none for padding
tests, because a padded input is compared against the oracle run on the UNPADDED real
keypoints (that is the definition of padding invariance).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

LN_EPS = 1e-5


# ---------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------

def linear(x: np.ndarray, weights: Dict[str, np.ndarray], name: str) -> np.ndarray:
    """Torch ``nn.Linear``: ``x @ W.T + b`` (bias optional), x is ``(N, in)``."""
    w = weights[name + ".weight"]
    out = np.zeros((x.shape[0], w.shape[0]))
    for i in range(x.shape[0]):
        for o in range(w.shape[0]):
            acc = 0.0
            for k in range(w.shape[1]):
                acc += x[i, k] * w[o, k]
            out[i, o] = acc
    bias = weights.get(name + ".bias")
    if bias is not None:
        out = out + bias[None, :]
    return out


def layer_norm(x: np.ndarray, weights: Dict[str, np.ndarray], name: str) -> np.ndarray:
    """Torch ``nn.LayerNorm`` over the last axis, eps 1e-5, with affine parameters."""
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        mean = sum(x[i]) / x.shape[1]
        var = sum((v - mean) ** 2 for v in x[i]) / x.shape[1]
        for j in range(x.shape[1]):
            out[i, j] = (x[i, j] - mean) / math.sqrt(var + LN_EPS)
    return out * weights[name + ".weight"][None, :] + weights[name + ".bias"][None, :]


def gelu_erf(x: np.ndarray) -> np.ndarray:
    """Exact GELU, ``0.5 x (1 + erf(x / sqrt 2))`` (torch ``nn.GELU()`` default)."""
    out = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        v = x[idx]
        out[idx] = 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable ``log(sigmoid(x)) = -softplus(-x)``."""
    return -(np.maximum(-x, 0.0) + np.log1p(np.exp(-np.abs(x))))


def log_softmax_1d(v: np.ndarray) -> np.ndarray:
    m = np.max(v)
    return v - m - math.log(float(np.sum(np.exp(v - m))))


def softmax_1d(v: np.ndarray) -> np.ndarray:
    m = np.max(v)
    e = np.exp(v - m)
    return e / np.sum(e)


# ---------------------------------------------------------------------
# keypoint normalisation and positional encoding
# ---------------------------------------------------------------------

def normalize_keypoints(kpts: np.ndarray, size: np.ndarray) -> np.ndarray:
    """``(kpts - size / 2) / (max(size) / 2)``. ``kpts`` is ``(N, 2)`` xy, ``size`` is (w, h)."""
    out = np.zeros_like(kpts, dtype=np.float64)
    scale = max(size[0], size[1]) / 2.0
    for i in range(kpts.shape[0]):
        for c in range(2):
            out[i, c] = (kpts[i, c] - size[c] / 2.0) / scale
    return out


def positional_encoding(wr_weight: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """``LearnableFourierPositionalEncoding.forward`` for one sample.

    :param wr_weight: ``Wr.weight`` of shape ``(Dh // 2, M)``.
    :param kpts: ``(N, M)`` normalised keypoints.
    :return: ``(2, N, Dh)``; slot 0 is cos and slot 1 is sin, each ``repeat_interleave(2)``
        along the channel axis, so ``out[s, n, 2 j] == out[s, n, 2 j + 1]``.
    """
    half = wr_weight.shape[0]
    n_pts = kpts.shape[0]
    out = np.zeros((2, n_pts, 2 * half))
    for n in range(n_pts):
        for j in range(half):
            proj = 0.0
            for m in range(kpts.shape[1]):
                proj += kpts[n, m] * wr_weight[j, m]
            out[0, n, 2 * j] = out[0, n, 2 * j + 1] = math.cos(proj)
            out[1, n, 2 * j] = out[1, n, 2 * j + 1] = math.sin(proj)
    return out


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Adjacent pairs ``(a, b) -> (-b, a)`` along the last axis of an ``(N, Dh)`` array."""
    out = np.zeros_like(x)
    for n in range(x.shape[0]):
        for j in range(x.shape[1] // 2):
            a = x[n, 2 * j]
            b = x[n, 2 * j + 1]
            out[n, 2 * j] = -b
            out[n, 2 * j + 1] = a
    return out


def apply_rotary(encoding: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``t * cos + rotate_half(t) * sin`` with ``encoding`` ``(2, N, Dh)``, ``t`` ``(N, Dh)``."""
    return t * encoding[0] + rotate_half(t) * encoding[1]


# ---------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------

def ffn(x: np.ndarray, message: np.ndarray, weights: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """``x + ffn(cat([x, message]))`` with ffn = Linear(2d,2d) -> LayerNorm -> GELU -> Linear(2d,d)."""
    h = np.concatenate([x, message], axis=-1)
    h = linear(h, weights, prefix + ".ffn.0")
    h = layer_norm(h, weights, prefix + ".ffn.1")
    h = gelu_erf(h)
    h = linear(h, weights, prefix + ".ffn.3")
    return x + h


def self_block(
    x: np.ndarray,
    encoding: np.ndarray,
    weights: Dict[str, np.ndarray],
    prefix: str,
    num_heads: int,
    keep: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``SelfBlock.forward`` for one sample. ``x`` is ``(N, d)``, ``keep`` optional ``(N, N)`` bool.

    A query row with no allowed key produces 0 (the reference's ``nan_to_num``).
    """
    n_pts, d = x.shape
    dh = d // num_heads
    qkv = linear(x, weights, prefix + ".Wqkv")
    context = np.zeros((n_pts, d))
    for h in range(num_heads):
        # (H, Dh, 3) layout: column = h * Dh * 3 + c * 3 + slot
        q = np.zeros((n_pts, dh))
        k = np.zeros((n_pts, dh))
        v = np.zeros((n_pts, dh))
        for n in range(n_pts):
            for c in range(dh):
                base = h * dh * 3 + c * 3
                q[n, c] = qkv[n, base + 0]
                k[n, c] = qkv[n, base + 1]
                v[n, c] = qkv[n, base + 2]
        q = apply_rotary(encoding, q)
        k = apply_rotary(encoding, k)
        scale = dh ** -0.5
        for i in range(n_pts):
            sim = np.array([sum(q[i, c] * k[j, c] for c in range(dh)) * scale for j in range(n_pts)])
            if keep is not None:
                allowed = [j for j in range(n_pts) if keep[i, j]]
                if not allowed:
                    continue  # row stays 0 (nan_to_num)
                sim = np.where(keep[i], sim, -np.inf)
            attn = softmax_1d(sim)
            for c in range(dh):
                context[i, h * dh + c] = sum(attn[j] * v[j, c] for j in range(n_pts))
    message = linear(context, weights, prefix + ".out_proj")
    return ffn(x, message, weights, prefix)


def cross_block(
    x0: np.ndarray,
    x1: np.ndarray,
    weights: Dict[str, np.ndarray],
    prefix: str,
    num_heads: int,
    keep: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``CrossBlock.forward`` for one sample. ``keep`` optional ``(M, N)`` bool.

    The similarity is computed once from ``to_qk`` of both images (each scaled by
    ``dh ** -0.25``), softmaxed along the other image for the 0->1 and the 1->0 direction.
    """
    m_pts, d = x0.shape
    n_pts = x1.shape[0]
    dh = d // num_heads
    qk0 = linear(x0, weights, prefix + ".to_qk")
    qk1 = linear(x1, weights, prefix + ".to_qk")
    v0 = linear(x0, weights, prefix + ".to_v")
    v1 = linear(x1, weights, prefix + ".to_v")
    s = (dh ** -0.5) ** 0.5
    ctx0 = np.zeros((m_pts, d))
    ctx1 = np.zeros((n_pts, d))
    for h in range(num_heads):
        sl = slice(h * dh, (h + 1) * dh)
        sim = np.zeros((m_pts, n_pts))
        for i in range(m_pts):
            for j in range(n_pts):
                sim[i, j] = sum((qk0[i, sl][c] * s) * (qk1[j, sl][c] * s) for c in range(dh))
        if keep is not None:
            sim = np.where(keep, sim, -np.inf)
        for i in range(m_pts):
            if np.all(np.isneginf(sim[i])):
                continue
            attn = softmax_1d(sim[i])
            for c in range(dh):
                ctx0[i, h * dh + c] = sum(attn[j] * v1[j, sl][c] for j in range(n_pts))
        for j in range(n_pts):
            if np.all(np.isneginf(sim[:, j])):
                continue
            attn = softmax_1d(sim[:, j])
            for c in range(dh):
                ctx1[j, h * dh + c] = sum(attn[i] * v0[i, sl][c] for i in range(m_pts))
    m0 = linear(ctx0, weights, prefix + ".to_out")
    m1 = linear(ctx1, weights, prefix + ".to_out")
    return ffn(x0, m0, weights, prefix), ffn(x1, m1, weights, prefix)


# ---------------------------------------------------------------------
# assignment, confidence, matches
# ---------------------------------------------------------------------

def match_assignment(
    desc0: np.ndarray, desc1: np.ndarray, weights: Dict[str, np.ndarray], prefix: str
) -> Tuple[np.ndarray, np.ndarray]:
    """``MatchAssignment.forward`` for one sample, returning ``(scores (M+1, N+1), sim (M, N))``.

    ``scores[:M, :N] = log_softmax_rows(sim) + log_softmax_cols(sim) + logsig(z0) + logsig(z1)^T``,
    the dustbin column is ``logsig(-z0)`` and the dustbin row is ``logsig(-z1)``, the corner is 0.
    """
    m_pts, d = desc0.shape
    n_pts = desc1.shape[0]
    md0 = linear(desc0, weights, prefix + ".final_proj") / d ** 0.25
    md1 = linear(desc1, weights, prefix + ".final_proj") / d ** 0.25
    sim = np.zeros((m_pts, n_pts))
    for i in range(m_pts):
        for j in range(n_pts):
            sim[i, j] = sum(md0[i, c] * md1[j, c] for c in range(d))
    z0 = linear(desc0, weights, prefix + ".matchability")[:, 0]
    z1 = linear(desc1, weights, prefix + ".matchability")[:, 0]
    scores = np.zeros((m_pts + 1, n_pts + 1))
    row_ls = np.stack([log_softmax_1d(sim[i, :]) for i in range(m_pts)]) if m_pts else sim
    col_ls = np.stack([log_softmax_1d(sim[:, j]) for j in range(n_pts)], axis=1) if n_pts else sim
    for i in range(m_pts):
        for j in range(n_pts):
            scores[i, j] = row_ls[i, j] + col_ls[i, j] + log_sigmoid(z0[i]) + log_sigmoid(z1[j])
    for i in range(m_pts):
        scores[i, n_pts] = log_sigmoid(-z0[i])
    for j in range(n_pts):
        scores[m_pts, j] = log_sigmoid(-z1[j])
    return scores, sim


def matchability(desc: np.ndarray, weights: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """``MatchAssignment.get_matchability``: ``sigmoid(Linear(d, 1)(desc))`` as ``(N,)``."""
    return sigmoid(linear(desc, weights, prefix + ".matchability")[:, 0])


def token_confidence(desc: np.ndarray, weights: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """``TokenConfidence`` for one image: ``sigmoid(Linear(d, 1)(desc))`` as ``(N,)``."""
    return sigmoid(linear(desc, weights, prefix + ".token.0")[:, 0])


def filter_matches(scores: np.ndarray, th: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``filter_matches`` for one ``(M+1, N+1)`` log-assignment.

    :return: ``(m0, m1, mscores0, mscores1)``; ``m0[i]`` is the matched index in image 1 or -1.
    """
    m_pts = scores.shape[0] - 1
    n_pts = scores.shape[1] - 1
    core = scores[:m_pts, :n_pts]
    m0 = np.array([int(np.argmax(core[i, :])) for i in range(m_pts)], dtype=np.int64)
    max0 = np.array([core[i, m0[i]] for i in range(m_pts)])
    m1 = np.array([int(np.argmax(core[:, j])) for j in range(n_pts)], dtype=np.int64)
    mutual0 = np.array([i == m1[m0[i]] for i in range(m_pts)])
    mutual1 = np.array([j == m0[m1[j]] for j in range(n_pts)])
    max0_exp = np.exp(max0)
    mscores0 = np.where(mutual0, max0_exp, 0.0)
    mscores1 = np.array([mscores0[m1[j]] if mutual1[j] else 0.0 for j in range(n_pts)])
    valid0 = mutual0 & (mscores0 > th)
    valid1 = np.array([mutual1[j] and valid0[m1[j]] for j in range(n_pts)])
    out0 = np.array([m0[i] if valid0[i] else -1 for i in range(m_pts)], dtype=np.int64)
    out1 = np.array([m1[j] if valid1[j] else -1 for j in range(n_pts)], dtype=np.int64)
    return out0, out1, mscores0, mscores1


def confidence_threshold(layer_index: int, n_layers: int) -> float:
    """``clip(0.8 + 0.1 * exp(-4 i / L), 0, 1)``."""
    return float(min(max(0.8 + 0.1 * math.exp(-4.0 * layer_index / n_layers), 0.0), 1.0))


# ---------------------------------------------------------------------
# full forward (all layers, no early stop, no pruning)
# ---------------------------------------------------------------------

def forward_single(
    weights: Dict[str, np.ndarray],
    kpts0: np.ndarray,
    desc0: np.ndarray,
    size0: np.ndarray,
    kpts1: np.ndarray,
    desc1: np.ndarray,
    size1: np.ndarray,
    n_layers: int,
    num_heads: int,
    extra0: Optional[np.ndarray] = None,
    extra1: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """The reference ``_forward`` with early stopping and pruning disabled, one sample.

    ``extra0/1`` are the optional ``(N, 2)`` ``(scales, oris)`` columns appended to the
    normalised keypoints when ``add_scale_ori`` is on (then ``Wr`` has 4 input columns).

    :return: dict with ``log_assignments`` (list of ``n_layers`` arrays ``(M+1, N+1)``),
        ``confidences0`` / ``confidences1`` (lists of ``n_layers - 1`` vectors), the final
        ``desc0`` / ``desc1``, and the final-layer ``matches0`` / ``matches1`` via
        ``filter_matches`` at threshold 0.1.
    """
    k0 = normalize_keypoints(kpts0, size0)
    k1 = normalize_keypoints(kpts1, size1)
    if extra0 is not None:
        k0 = np.concatenate([k0, extra0], axis=-1)
        k1 = np.concatenate([k1, extra1], axis=-1)
    d0, d1 = desc0.astype(np.float64), desc1.astype(np.float64)
    if "input_proj.weight" in weights:
        d0 = linear(d0, weights, "input_proj")
        d1 = linear(d1, weights, "input_proj")
    enc0 = positional_encoding(weights["posenc.Wr.weight"], k0)
    enc1 = positional_encoding(weights["posenc.Wr.weight"], k1)

    log_assignments: List[np.ndarray] = []
    conf0: List[np.ndarray] = []
    conf1: List[np.ndarray] = []
    for i in range(n_layers):
        p = f"transformers.{i}"
        d0 = self_block(d0, enc0, weights, p + ".self_attn", num_heads)
        d1 = self_block(d1, enc1, weights, p + ".self_attn", num_heads)
        d0, d1 = cross_block(d0, d1, weights, p + ".cross_attn", num_heads)
        scores, _ = match_assignment(d0, d1, weights, f"log_assignment.{i}")
        log_assignments.append(scores)
        if i < n_layers - 1:
            conf0.append(token_confidence(d0, weights, f"token_confidence.{i}"))
            conf1.append(token_confidence(d1, weights, f"token_confidence.{i}"))
    m0, m1, ms0, ms1 = filter_matches(log_assignments[-1], 0.1)
    return {
        "log_assignments": log_assignments,
        "confidences0": conf0,
        "confidences1": conf1,
        "desc0": d0,
        "desc1": d1,
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": ms0,
        "matching_scores1": ms1,
    }


def forward_batched(weights, kpts0, desc0, size0, kpts1, desc1, size1, n_layers, num_heads):
    """Loop :func:`forward_single` over the batch axis and stack ``log_assignments`` to
    ``(B, L, M+1, N+1)`` and the confidences to ``(B, L-1, M)`` / ``(B, L-1, N)``."""
    outs = [
        forward_single(
            weights, kpts0[b], desc0[b], size0[b], kpts1[b], desc1[b], size1[b], n_layers, num_heads
        )
        for b in range(kpts0.shape[0])
    ]
    return {
        "log_assignments": np.stack([np.stack(o["log_assignments"]) for o in outs]),
        "confidences0": np.stack([np.stack(o["confidences0"]) for o in outs]),
        "confidences1": np.stack([np.stack(o["confidences1"]) for o in outs]),
    }


# ---------------------------------------------------------------------
# adaptive path: early stop (depth confidence) and point pruning (width confidence)
# ---------------------------------------------------------------------

def get_pruning_mask_ref(
    confidences: Optional[np.ndarray],
    scores: np.ndarray,
    layer_index: int,
    n_layers: int,
    width_confidence: float,
) -> np.ndarray:
    """Reference ``get_pruning_mask``: keep a point when ``matchability > 1 - width_confidence``
    OR (when confidences exist) its token confidence is ``<= confidence_threshold(i)``.

    Low-confidence points are never pruned. ``confidences`` is ``None`` when the depth
    confidence is disabled (the confidence head is then not evaluated at all).
    """
    th = confidence_threshold(layer_index, n_layers)
    keep = np.zeros(scores.shape[0], dtype=bool)
    for k in range(scores.shape[0]):
        keep[k] = bool(scores[k] > 1.0 - width_confidence)
        if confidences is not None and confidences[k] <= th:
            keep[k] = True
    return keep


def check_if_stop_ref(
    conf0: np.ndarray,
    conf1: np.ndarray,
    layer_index: int,
    n_layers: int,
    num_points: int,
    depth_confidence: float,
) -> bool:
    """Reference ``check_if_stop``: ``1 - #(conf < threshold) / num_points > depth_confidence``.

    ``num_points`` is the ORIGINAL ``m + n``, so a pruned point counts as confident.
    """
    th = confidence_threshold(layer_index, n_layers)
    below = sum(1 for c in list(conf0) + list(conf1) if c < th)
    return (1.0 - below / num_points) > depth_confidence


def forward_adaptive(
    weights: Dict[str, np.ndarray],
    kpts0: np.ndarray,
    desc0: np.ndarray,
    size0: np.ndarray,
    kpts1: np.ndarray,
    desc1: np.ndarray,
    size1: np.ndarray,
    n_layers: int,
    num_heads: int,
    depth_confidence: float,
    width_confidence: float,
    filter_threshold: float = 0.1,
    pruning_min_kpts: int = -1,
) -> Dict[str, object]:
    """The reference ``LightGlue._forward`` with early stop and pruning, one sample, no padding.

    Transcribed from ``cvg/LightGlue`` ``lightglue/lightglue.py`` ``_forward`` (fetched
    2026-10-02). Quirks kept on purpose: the empty-side check sits at the top of the loop,
    so ``stop`` is ``i + 1`` of the iteration that BROKE (one more than the layers run);
    ``prune`` counters start at 1 and add 1 for each pruning step a point survives; without
    point pruning they are all ``n_layers``.

    :return: dict with ``matches0`` (M,), ``matches1`` (N,), ``matching_scores0/1``,
        ``stop`` (int), ``matches`` (S, 2) in ORIGINAL indices, ``scores`` (S,),
        ``prune0`` / ``prune1``.
    """
    m, n = kpts0.shape[0], kpts1.shape[0]
    k0 = normalize_keypoints(kpts0, size0)
    k1 = normalize_keypoints(kpts1, size1)
    d0, d1 = desc0.astype(np.float64), desc1.astype(np.float64)
    if "input_proj.weight" in weights:
        d0 = linear(d0, weights, "input_proj")
        d1 = linear(d1, weights, "input_proj")
    enc0 = positional_encoding(weights["posenc.Wr.weight"], k0)
    enc1 = positional_encoding(weights["posenc.Wr.weight"], k1)

    do_early_stop = depth_confidence > 0
    do_point_pruning = width_confidence > 0
    ind0, ind1 = np.arange(m), np.arange(n)
    prune0, prune1 = np.ones(m, dtype=np.int64), np.ones(n, dtype=np.int64)
    i = 0
    for i in range(n_layers):
        if d0.shape[0] == 0 or d1.shape[0] == 0:
            break
        p = f"transformers.{i}"
        d0 = self_block(d0, enc0, weights, p + ".self_attn", num_heads)
        d1 = self_block(d1, enc1, weights, p + ".self_attn", num_heads)
        d0, d1 = cross_block(d0, d1, weights, p + ".cross_attn", num_heads)
        if i == n_layers - 1:
            continue
        t0 = t1 = None
        if do_early_stop:
            t0 = token_confidence(d0, weights, f"token_confidence.{i}")
            t1 = token_confidence(d1, weights, f"token_confidence.{i}")
            if check_if_stop_ref(t0, t1, i, n_layers, m + n, depth_confidence):
                break
        if do_point_pruning and d0.shape[0] > pruning_min_kpts:
            s0 = matchability(d0, weights, f"log_assignment.{i}")
            keep0 = np.nonzero(get_pruning_mask_ref(t0, s0, i, n_layers, width_confidence))[0]
            ind0, d0, enc0 = ind0[keep0], d0[keep0], enc0[:, keep0]
            prune0[ind0] += 1
        if do_point_pruning and d1.shape[0] > pruning_min_kpts:
            s1 = matchability(d1, weights, f"log_assignment.{i}")
            keep1 = np.nonzero(get_pruning_mask_ref(t1, s1, i, n_layers, width_confidence))[0]
            ind1, d1, enc1 = ind1[keep1], d1[keep1], enc1[:, keep1]
            prune1[ind1] += 1

    if d0.shape[0] == 0 or d1.shape[0] == 0:
        if not do_point_pruning:
            prune0 = np.ones(m, dtype=np.int64) * n_layers
            prune1 = np.ones(n, dtype=np.int64) * n_layers
        return {
            "matches0": -np.ones(m, dtype=np.int64), "matches1": -np.ones(n, dtype=np.int64),
            "matching_scores0": np.zeros(m), "matching_scores1": np.zeros(n),
            "stop": i + 1, "matches": np.zeros((0, 2), dtype=np.int64), "scores": np.zeros(0),
            "prune0": prune0, "prune1": prune1,
        }

    scores, _ = match_assignment(d0, d1, weights, f"log_assignment.{i}")
    m0, m1, ms0, ms1 = filter_matches(scores, filter_threshold)
    valid = np.nonzero(m0 > -1)[0]
    matches = (np.stack([ind0[valid], ind1[m0[valid]]], axis=-1)
               if len(valid) else np.zeros((0, 2), dtype=np.int64))
    out0 = -np.ones(m, dtype=np.int64)
    out1 = -np.ones(n, dtype=np.int64)
    for k in range(len(ind0)):
        out0[ind0[k]] = -1 if m0[k] == -1 else ind1[m0[k]]
    for k in range(len(ind1)):
        out1[ind1[k]] = -1 if m1[k] == -1 else ind0[m1[k]]
    sc0, sc1 = np.zeros(m), np.zeros(n)
    sc0[ind0] = ms0
    sc1[ind1] = ms1
    if not do_point_pruning:
        prune0 = np.ones(m, dtype=np.int64) * n_layers
        prune1 = np.ones(n, dtype=np.int64) * n_layers
    return {
        "matches0": out0, "matches1": out1, "matching_scores0": sc0, "matching_scores1": sc1,
        "stop": i + 1, "matches": matches, "scores": ms0[valid],
        "prune0": prune0, "prune1": prune1,
    }


# ---------------------------------------------------------------------
# random torch-layout weights
# ---------------------------------------------------------------------

def random_weights(
    rng: np.random.RandomState,
    d: int,
    num_heads: int,
    n_layers: int,
    m: int = 2,
    input_dim: Optional[int] = None,
    gamma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Random torch-named, torch-layout weights (Linear ``(out, in)``, LayerNorm affine non-trivial)."""
    dh = d // num_heads

    def lin(prefix, out_f, in_f, bias=True, scale=0.2):
        w = {prefix + ".weight": rng.normal(0.0, scale, size=(out_f, in_f))}
        if bias:
            w[prefix + ".bias"] = rng.normal(0.0, 0.1, size=(out_f,))
        return w

    def ffn_w(prefix):
        w = {}
        w.update(lin(prefix + ".ffn.0", 2 * d, 2 * d))
        w[prefix + ".ffn.1.weight"] = rng.normal(1.0, 0.1, size=(2 * d,))
        w[prefix + ".ffn.1.bias"] = rng.normal(0.0, 0.1, size=(2 * d,))
        w.update(lin(prefix + ".ffn.3", d, 2 * d))
        return w

    weights: Dict[str, np.ndarray] = {}
    if input_dim is not None and input_dim != d:
        weights.update(lin("input_proj", d, input_dim))
    weights["posenc.Wr.weight"] = rng.normal(0.0, gamma ** -2, size=(dh // 2, m))
    for i in range(n_layers):
        p = f"transformers.{i}"
        weights.update(lin(p + ".self_attn.Wqkv", 3 * d, d))
        weights.update(lin(p + ".self_attn.out_proj", d, d))
        weights.update(ffn_w(p + ".self_attn"))
        weights.update(lin(p + ".cross_attn.to_qk", d, d))
        weights.update(lin(p + ".cross_attn.to_v", d, d))
        weights.update(lin(p + ".cross_attn.to_out", d, d))
        weights.update(ffn_w(p + ".cross_attn"))
        weights.update(lin(f"log_assignment.{i}.matchability", 1, d))
        weights.update(lin(f"log_assignment.{i}.final_proj", d, d))
        if i < n_layers - 1:
            weights.update(lin(f"token_confidence.{i}.token.0", 1, d))
    return weights
