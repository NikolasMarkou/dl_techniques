"""A pure-numpy DINOv2 forward over the official safetensors tensors (float64).

Shared instrument (no ``test_`` prefix; its RED proof is
``test_dinov2_numpy_reference.py``) for the official-weights guards of
plan-2026-10-14T062616-81938e81 iteration 2 step 1.7. Written from the two
references, not from the Keras port: transformers'
``models/dinov2/modeling_dinov2.py`` (tensor names, separate q/k/v, LayerNorm
eps 1e-6, exact GELU, ``layer_scale{1,2}.lambda1``) and facebookresearch
``dinov2/models/vision_transformer.py`` (``interpolate_pos_encoding``: the table
is used as is when the input grid equals the stored one and is square, else
bicubic at ``scale_factor = (n + offset) / M``, or at the output size when the
offset is 0). torch's ``upsample_bicubic2d`` is written out here as a
per-output-pixel 16-tap loop (``aten/src/ATen/native/UpSampleBicubic2d.cpp``:
source ``(o + 0.5) / scale - 0.5``, ``A = -0.75``, clamped taps), deliberately
not as the separable matrix the port uses, so the two are independent.

Interface contract: every function takes the official tensor dict
(``safetensors.numpy.load_file`` output, torch layout) and numpy arrays, and
returns float64 numpy arrays. Nothing here imports keras or tensorflow.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import erf

_A = -0.75


def _near(x: float) -> float:
    return ((_A + 2.0) * x - (_A + 3.0)) * x * x + 1.0


def _far(x: float) -> float:
    return ((_A * x - 5.0 * _A) * x + 8.0 * _A) * x - 4.0 * _A


def torch_bicubic_2d(image: np.ndarray, out_h: int, out_w: int,
                     scale_h: Optional[float] = None,
                     scale_w: Optional[float] = None) -> np.ndarray:
    """``(C, H, W) -> (C, out_h, out_w)`` as torch's bicubic, ``align_corners=False``.

    ``scale_h``/``scale_w`` are torch's ``scale_factor`` per axis; ``None``
    means "resample to the output size".
    """
    c, h, w = image.shape
    step_h = 1.0 / scale_h if scale_h else h / out_h
    step_w = 1.0 / scale_w if scale_w else w / out_w
    out = np.zeros((c, out_h, out_w))
    for oy in range(out_h):
        sy = step_h * (oy + 0.5) - 0.5
        iy = math.floor(sy)
        ty = sy - iy
        wy = (_far(ty + 1.0), _near(ty), _near(1.0 - ty), _far(2.0 - ty))
        for ox in range(out_w):
            sx = step_w * (ox + 0.5) - 0.5
            ix = math.floor(sx)
            tx = sx - ix
            wx = (_far(tx + 1.0), _near(tx), _near(1.0 - tx), _far(2.0 - tx))
            acc = np.zeros(c)
            for j in range(4):
                yy = min(max(iy - 1 + j, 0), h - 1)
                for i in range(4):
                    xx = min(max(ix - 1 + i, 0), w - 1)
                    acc += wy[j] * wx[i] * image[:, yy, xx]
            out[:, oy, ox] = acc
    return out


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _linear(x: np.ndarray, t: Dict[str, np.ndarray], name: str) -> np.ndarray:
    return x @ t[name + ".weight"].astype(np.float64).T + t[name + ".bias"].astype(np.float64)


def embed(t: Dict[str, np.ndarray], image: np.ndarray, offset: float = 0.1,
          patch: int = 14) -> np.ndarray:
    """Image ``(H, W, 3)`` -> tokens ``(1 + N, D)``: patches, [CLS], resampled positions."""
    h, w, _ = image.shape
    gh, gw = h // patch, w // patch
    kernel = t["embeddings.patch_embeddings.projection.weight"].astype(np.float64)  # (D, 3, p, p)
    dim = kernel.shape[0]
    x = image.astype(np.float64)[:gh * patch, :gw * patch]
    x = x.reshape(gh, patch, gw, patch, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, -1)
    x = x @ kernel.reshape(dim, -1).T + t["embeddings.patch_embeddings.projection.bias"]
    x = np.concatenate([t["embeddings.cls_token"].astype(np.float64).reshape(1, dim), x], 0)
    pos = t["embeddings.position_embeddings"].astype(np.float64)[0]
    m = int(round(math.sqrt(pos.shape[0] - 1)))
    if not (gh == gw == m):
        grid = pos[1:].reshape(m, m, dim).transpose(2, 0, 1)
        if offset:
            grid = torch_bicubic_2d(grid, gh, gw, (gh + offset) / m, (gw + offset) / m)
        else:
            grid = torch_bicubic_2d(grid, gh, gw)
        pos = np.concatenate([pos[:1], grid.transpose(1, 2, 0).reshape(gh * gw, dim)], 0)
    return x + pos


def block(t: Dict[str, np.ndarray], i: int, x: np.ndarray, num_heads: int) -> np.ndarray:
    """Block ``i`` of the official model on tokens ``(N, D)``."""
    p = f"encoder.layer.{i}."
    n, dim = x.shape
    hd = dim // num_heads
    y = layer_norm(x, t[p + "norm1.weight"], t[p + "norm1.bias"])
    q, k, v = (_linear(y, t, p + f"attention.attention.{name}").reshape(n, num_heads, hd).transpose(1, 0, 2)
               for name in ("query", "key", "value"))
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores = scores / scores.sum(-1, keepdims=True)
    attended = (scores @ v).transpose(1, 0, 2).reshape(n, dim)
    x = x + _linear(attended, t, p + "attention.output.dense") * t[p + "layer_scale1.lambda1"]
    y = layer_norm(x, t[p + "norm2.weight"], t[p + "norm2.bias"])
    y = _linear(y, t, p + "mlp.fc1")
    y = 0.5 * y * (1.0 + erf(y / math.sqrt(2.0)))
    return x + _linear(y, t, p + "mlp.fc2") * t[p + "layer_scale2.lambda1"]


def forward(t: Dict[str, np.ndarray], image: np.ndarray, num_heads: int,
            offset: float = 0.1, last: int = 4,
            patch: int = 14) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``(final normed tokens (1 + N, D), the last `last` block outputs, each normed)``."""
    depth = 1 + max(int(k.split(".")[2]) for k in t if k.startswith("encoder.layer."))
    x = embed(t, image, offset, patch)
    kept = []
    for i in range(depth):
        x = block(t, i, x, num_heads)
        if i >= depth - last:
            kept.append(x)

    def norm(z):
        return layer_norm(z, t["layernorm.weight"], t["layernorm.bias"])

    return norm(x), [norm(z) for z in kept]


def official_manifest(embed_dim: int, depth: int, mlp_ratio: int = 4,
                      grid: int = 37, patch: int = 14) -> Dict[str, Tuple[int, ...]]:
    """Name -> shape of every tensor in an official ``facebook/dinov2-*`` file."""
    d, hidden = embed_dim, embed_dim * mlp_ratio
    out = {
        "embeddings.cls_token": (1, 1, d),
        "embeddings.mask_token": (1, d),
        "embeddings.position_embeddings": (1, 1 + grid * grid, d),
        "embeddings.patch_embeddings.projection.weight": (d, 3, patch, patch),
        "embeddings.patch_embeddings.projection.bias": (d,),
        "layernorm.weight": (d,),
        "layernorm.bias": (d,),
    }
    for i in range(depth):
        p = f"encoder.layer.{i}."
        for name in ("query", "key", "value"):
            out[p + f"attention.attention.{name}.weight"] = (d, d)
            out[p + f"attention.attention.{name}.bias"] = (d,)
        out.update({
            p + "attention.output.dense.weight": (d, d), p + "attention.output.dense.bias": (d,),
            p + "layer_scale1.lambda1": (d,), p + "layer_scale2.lambda1": (d,),
            p + "norm1.weight": (d,), p + "norm1.bias": (d,),
            p + "norm2.weight": (d,), p + "norm2.bias": (d,),
            p + "mlp.fc1.weight": (hidden, d), p + "mlp.fc1.bias": (hidden,),
            p + "mlp.fc2.weight": (d, hidden), p + "mlp.fc2.bias": (d,),
        })
    return out
