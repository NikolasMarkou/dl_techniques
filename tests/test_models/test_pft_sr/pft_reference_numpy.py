"""A numpy transcription of the official PFT-SR forward, for equation-level guards.

Shared instrument (no ``test_`` prefix, so pytest does not collect it); its own
RED proof is ``test_pft_reference_numpy.py`` beside it.

Source: CVL-UESTC/PFT-SR ``basicsr/archs/pft_arch.py`` (master, file commit
``0f4f38039d3e``, retrieved 2026-10-20). torch is not installed here and the
official top-k path runs custom CUDA kernels (``smm_cuda``), so this module
re-states the official code line for line in numpy, in the official SPARSE form:
after a top-k the attention is a ``(b_, heads, n, k)`` value tensor plus a
``(b_, heads, n, k)`` index tensor, and ``SMM_QmK`` / ``SMM_AmV`` are gathers.
The Keras port computes the same thing densely under a boolean mask, so the two
share no code: agreement between them is evidence, not a tautology.

Line references below are to that file.

Weights are read from a Keras model or layer through these attribute names
(the port's public layout): a layer has ``norm1``, ``wqkv``, ``v_lepe``,
``relative_position_bias_table``, ``proj``, ``norm2``, ``fc1``, ``ffn_dwconv``,
``fc2``; the model has ``conv_first``, ``patch_norm``, ``pft_layers``,
``stage_convs``, ``norm``, ``conv_after_body`` and ``tail_layers``. Keras Dense
kernels are ``(in, out)`` and conv kernels ``(kh, kw, in, out)``; the maths
below is written for that layout (torch's ``x @ W.T`` is ``x @ kernel`` here).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erf

#: torch ``nn.LayerNorm`` default (``norm_layer=nn.LayerNorm`` everywhere).
LN_EPS = 1e-5
#: ``WindowAttention.eps`` (:225).
PFA_EPS = 1e-20
#: ``PFT.__init__`` (:915), used when ``in_chans == 3``.
RGB_MEAN = np.array([0.4488, 0.4371, 0.4040])

#: TensorFlow builds two constants of the port's ops in float32 even when the
#: tensor is float64 (measured 2026-10-20, TF 2.18 / Keras 3.8):
#: ``tf.nn.gelu(approximate=False)`` divides by ``cast(1.4142135623730951)``,
#: a float32 constant cast up, and ``LeakyRelu``'s ``alpha`` is a float32
#: attribute. At float64 that moves a GELU by up to 5e-9 and the 0.01 slope by
#: 2.2e-10 (2.2e-8 relative; 0.2 moves by 3.0e-9, 1.5e-8 relative), above
#: float64 reassociation noise and far below anything a structural error
#: moves. The oracle uses the same two roundings so the guards can hold the
#: port to float64 noise; ``test_pft_reference_numpy.py`` pins that the
#: framework still rounds them. Both are TensorFlow-internal and stay (step
#: 7.1, D-218): the port calls ``keras.ops.gelu`` and ``keras.layers.LeakyReLU``.
TF_SQRT2 = float(np.float32(np.sqrt(2.0)))


def tf_slope(slope: float) -> float:
    """A LeakyReLU slope as TensorFlow applies it (a float32 attribute)."""
    return float(np.float32(slope))


def to_np(t: Any) -> np.ndarray:
    """A Keras variable or tensor as a float64 numpy array."""
    return np.asarray(t, dtype=np.float64) if isinstance(t, np.ndarray) \
        else np.asarray(t.numpy() if hasattr(t, "numpy") else t, dtype=np.float64)


def layer_norm(x: np.ndarray, layer: Any) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * to_np(layer.gamma) + to_np(layer.beta)


def gelu(x: np.ndarray, sqrt2: float = TF_SQRT2) -> np.ndarray:
    """torch ``nn.GELU()``: the exact erf form (with TensorFlow's float32
    ``sqrt(2)`` by default, see :data:`TF_SQRT2`)."""
    return 0.5 * x * (1.0 + erf(x / sqrt2))


def dense(x: np.ndarray, layer: Any) -> np.ndarray:
    y = x @ to_np(layer.kernel)
    return y + to_np(layer.bias) if layer.bias is not None else y


def conv2d(x: np.ndarray, layer: Any) -> np.ndarray:
    """``nn.Conv2d(i, o, k, 1, (k-1)//2)``, NHWC, zero padding."""
    kernel = to_np(layer.kernel)
    k = kernel.shape[0]
    p = (k - 1) // 2
    b, h, w, _ = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
    out = np.zeros((b, h, w, kernel.shape[-1]))
    for i in range(k):
        for j in range(k):
            out += xp[:, i:i + h, j:j + w, :] @ kernel[i, j]
    return out + to_np(layer.bias)


def depthwise(x: np.ndarray, layer: Any) -> np.ndarray:
    """``nn.Conv2d(c, c, k, 1, (k-1)//2, groups=c)``, NHWC, zero padding."""
    kernel = to_np(layer.kernel)[:, :, :, 0]
    k = kernel.shape[0]
    p = (k - 1) // 2
    b, h, w, c = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
    out = np.zeros_like(x)
    for i in range(k):
        for j in range(k):
            out += xp[:, i:i + h, j:j + w, :] * kernel[i, j]
    return out + to_np(layer.bias)


def window_partition(x: np.ndarray, ws: int) -> np.ndarray:
    """:172."""
    b, h, w, c = x.shape
    x = x.reshape(b, h // ws, ws, w // ws, ws, c)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c)


def window_reverse(windows: np.ndarray, ws: int, h: int, w: int) -> np.ndarray:
    """:186."""
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.reshape(b, h // ws, w // ws, ws, ws, -1)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


def relative_position_index(ws: int) -> np.ndarray:
    """``PFT.calculate_rpi_sa`` (:1067)."""
    coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij"))
    flat = coords.reshape(2, -1)
    rel = (flat[:, :, None] - flat[:, None, :]).transpose(1, 2, 0).copy()
    rel[:, :, 0] += ws - 1
    rel[:, :, 1] += ws - 1
    rel[:, :, 0] *= 2 * ws - 1
    return rel.sum(-1)


def calculate_mask(h: int, w: int, ws: int) -> np.ndarray:
    """``PFT.calculate_mask`` (:1081): ``(nw, n, n)``, 0 or -100."""
    img_mask = np.zeros((1, h, w, 1))
    slices = (slice(0, -ws), slice(-ws, -(ws // 2)), slice(-(ws // 2), None))
    cnt = 0
    for hs in slices:
        for wsl in slices:
            img_mask[:, hs, wsl, :] = cnt
            cnt += 1
    mw = window_partition(img_mask, ws).reshape(-1, ws * ws)
    m = mw[:, None, :] - mw[:, :, None]
    return np.where(m != 0, -100.0, 0.0)


def smm_qmk(q: np.ndarray, k: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``SMM_QmK``: ``out[b, i, j] = q[b, i] . k[b, index[b, i, j]]``."""
    kg = np.take_along_axis(k[:, None, :, :], index[..., None], axis=2)
    return np.einsum("bid,bijd->bij", q, kg)


def smm_amv(a: np.ndarray, v: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``SMM_AmV``: ``out[b, i] = sum_j a[b, i, j] * v[b, index[b, i, j]]``."""
    vg = np.take_along_axis(v[:, None, :, :], index[..., None], axis=2)
    return np.einsum("bij,bijd->bid", a, vg)


def topk(a: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """``torch.topk(a, k, dim=-1)``; ties to the lower index (random inputs
    have none, so the official ``sorted=False`` order does not matter: every
    later use is a gather that permutes values and indices together)."""
    idx = np.argsort(-a, axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(a, idx, axis=-1), idx


def window_attention(qkvp: np.ndarray, layer: Any, num_heads: int, ws: int, topk_l: int,
                     pfa_values: List[Optional[np.ndarray]],
                     pfa_indices: List[Optional[np.ndarray]],
                     mask: Optional[np.ndarray], shift: int) -> np.ndarray:
    """``WindowAttention.forward`` (:240), eval mode. Mutates the two lists."""
    b_, n, c4 = qkvp.shape
    c = c4 // 4
    d = c // num_heads
    t = qkvp.reshape(b_, n, 4, num_heads, d).transpose(2, 0, 3, 1, 4)
    q, k, v, v_lepe = t[0], t[1], t[2], t[3]
    q = q * d ** -0.5
    table = to_np(layer.relative_position_bias_table)
    rpb = table[relative_position_index(ws).reshape(-1)].reshape(n, n, -1)
    rpb = rpb.transpose(2, 0, 1)[None]                       # (1, nH or 1, n, n)
    if pfa_indices[shift] is None:
        attn = q @ k.transpose(0, 1, 3, 2) + rpb
        if shift:
            nw = mask.shape[0]
            attn = attn.reshape(b_ // nw, nw, num_heads, n, n) + mask[None, :, None]
            attn = attn.reshape(-1, num_heads, n, n)
    else:
        idx = pfa_indices[shift]
        kk = idx.shape[-1]
        attn = smm_qmk(q.reshape(b_ * num_heads, n, d), k.reshape(b_ * num_heads, n, d),
                       idx.reshape(b_ * num_heads, n, kk)).reshape(b_, num_heads, n, kk)
        rpb_full = np.broadcast_to(rpb, (b_, num_heads, n, n))
        attn = attn + np.take_along_axis(rpb_full, idx, axis=-1)
    attn = np.exp(attn - attn.max(-1, keepdims=True))
    attn = attn / attn.sum(-1, keepdims=True)
    if pfa_values[shift] is not None:
        attn = attn * pfa_values[shift]
        attn = (attn + PFA_EPS) / (attn.sum(-1, keepdims=True) + PFA_EPS)
    if topk_l < attn.shape[-1]:
        attn, idx = topk(attn, topk_l)
        if pfa_indices[shift] is not None:
            pfa_indices[shift] = np.take_along_axis(pfa_indices[shift], idx, axis=-1)
        else:
            pfa_indices[shift] = idx
    pfa_values[shift] = attn
    if pfa_indices[shift] is None:
        x = attn @ v + v_lepe
    else:
        idx = pfa_indices[shift]
        kk = idx.shape[-1]
        x = smm_amv(attn.reshape(b_ * num_heads, n, kk), v.reshape(b_ * num_heads, n, d),
                    idx.reshape(b_ * num_heads, n, kk)).reshape(b_, num_heads, n, d) + v_lepe
    x = x.transpose(0, 2, 1, 3).reshape(b_, n, c)
    return dense(x, layer.proj)


def pft_layer(x: np.ndarray, layer: Any, num_heads: int, ws: int, shift_size: int,
              topk_l: int, pfa_values: List[Optional[np.ndarray]],
              pfa_indices: List[Optional[np.ndarray]]) -> np.ndarray:
    """``PFTransformerLayer.forward`` (:438) on an NHWC map ``(b, h, w, c)``."""
    b, h, w, c = x.shape
    shortcut = x
    xn = layer_norm(x, layer.norm1)
    x_qkv = dense(xn, layer.wqkv)
    v_lepe = gelu(depthwise(x_qkv[..., 2 * c:], layer.v_lepe))      # dwconv: conv + GELU
    x_qkvp = np.concatenate([x_qkv, v_lepe], axis=-1)
    if shift_size > 0:
        shift = 1
        shifted = np.roll(x_qkvp, (-shift_size, -shift_size), axis=(1, 2))
    else:
        shift = 0
        shifted = x_qkvp
    win = window_partition(shifted, ws).reshape(-1, ws * ws, 4 * c)
    mask = calculate_mask(h, w, ws) if shift else None
    out = window_attention(win, layer, num_heads, ws, topk_l, pfa_values, pfa_indices,
                           mask, shift)
    out = window_reverse(out.reshape(-1, ws, ws, c), ws, h, w)
    if shift_size > 0:
        out = np.roll(out, (shift_size, shift_size), axis=(1, 2))
    x = shortcut + out
    # ConvFFN (:155): fc1, GELU, x + dwconv(x) (dwconv = depthwise conv + GELU), fc2.
    y = gelu(dense(layer_norm(x, layer.norm2), layer.fc1))
    y = y + gelu(depthwise(y, layer.ffn_dwconv))
    return x + dense(y, layer.fc2)


def mirror_pad(x: np.ndarray, ws: int) -> np.ndarray:
    """``PFT.forward`` (:1103-1109), NHWC."""
    h0, w0 = x.shape[1], x.shape[2]
    h = ((h0 + ws - 1) // ws) * ws
    w = ((w0 + ws - 1) // ws) * ws
    x = np.concatenate([x, np.flip(x, 1)], 1)[:, :h]
    return np.concatenate([x, np.flip(x, 2)], 2)[:, :, :w]


def pixel_shuffle(x: np.ndarray, r: int) -> np.ndarray:
    """The port's ``PixelShuffle2D`` order ``(i, j, c)`` (README section 4.3)."""
    b, h, w, c = x.shape
    x = x.reshape(b, h, w, r, r, c // (r * r)).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h * r, w * r, c // (r * r))


def model_forward(model: Any, x: np.ndarray, num_topk: Optional[Sequence[int]]) -> np.ndarray:
    """``PFT.forward`` (:1102) for the ``pixelshuffle`` and ``pixelshuffledirect``
    branches, or the features the tail reads when ``model.include_top`` is off."""
    ws, heads, depths = model.window_size, model.num_heads, list(model.num_blocks)
    h0, w0 = x.shape[1], x.shape[2]
    x = mirror_pad(np.asarray(x, np.float64), ws)
    mean = RGB_MEAN if x.shape[-1] == 3 else np.zeros(1)
    x = (x - mean) * model.img_range
    x = conv2d(x, model.conv_first)
    shallow = x
    # forward_features (:1047): patch_embed (flatten + LayerNorm), the PFTBs,
    # the final LayerNorm.
    pfa_values: List[Optional[np.ndarray]] = [None, None]
    pfa_indices: List[Optional[np.ndarray]] = [None, None]
    t = layer_norm(x, model.patch_norm)
    layer_id = 0
    for s, depth in enumerate(depths):
        stage_in = t
        for i in range(depth):
            topk_l = num_topk[layer_id] if num_topk is not None else ws * ws
            t = pft_layer(t, model.pft_layers[layer_id], heads, ws,
                          0 if i % 2 == 0 else ws // 2, topk_l, pfa_values, pfa_indices)
            layer_id += 1
        # PFTB.forward (:713): patch_embed(conv(patch_unembed(x))) + x
        t = conv2d(t, model.stage_convs[s]) + stage_in
    t = layer_norm(t, model.norm)
    x = conv2d(t, model.conv_after_body) + shallow
    if not model.include_top:
        return x[:, :h0, :w0]
    layers: Dict[str, Any] = {layer.name: layer for layer in model.tail_layers}
    scale = model.scale
    if model.upsampler == "pixelshuffledirect":
        x = pixel_shuffle(conv2d(x, layers["upsample_conv"]), scale)
    elif model.upsampler == "pixelshuffle":
        x = conv2d(x, layers["conv_before_upsample"])
        x = np.where(x >= 0, x, tf_slope(0.01) * x)
        factors = [3] if scale == 3 else [2] * (scale.bit_length() - 1)
        for i, r in enumerate(factors):
            x = pixel_shuffle(conv2d(x, layers[f"upsample_conv_{i}"]), r)
        x = conv2d(x, layers["conv_last"])
    else:
        raise NotImplementedError(model.upsampler)
    x = x / model.img_range + mean
    return x[:, :h0 * scale, :w0 * scale]
