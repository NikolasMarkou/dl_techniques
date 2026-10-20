"""PFTSR's parameter count is a hand sum over the official ``PFT`` module list.

The module list is transcribed from CVL-UESTC/PFT-SR ``basicsr/archs/pft_arch.py``
(master, file commit ``0f4f38039d3e``; line numbers below are that file's):

- ``conv_first`` ``nn.Conv2d(in, C, 3, 1, 1)`` (:923)
- ``patch_embed.norm`` ``nn.LayerNorm(C)`` (:936, ``patch_norm=True``)
- per layer (``PFTransformerLayer``, :418-436):
  ``norm1``, ``norm2`` LayerNorm(C); ``wqkv`` Linear(C, 3C, bias=qkv_bias);
  ``v_LePE`` = ``dwconv`` depthwise ``Conv2d(C, C, k, groups=C)`` with bias;
  ``attn_win``: ``relative_position_bias_table`` ``((2ws-1)^2, nH)`` when
  ``dim > 100`` and ``((2ws-1)^2, 1)`` otherwise (:228-233), ``proj``
  Linear(C, C); ``convffn``: ``fc1`` Linear(C, H), ``dwconv`` depthwise
  ``Conv2d(H, H, k, groups=H)``, ``fc2`` Linear(H, C), ``H = int(C * mlp_ratio)``,
  ``k = convffn_kernel_size``
- per stage (``PFTB``, ``resi_connection='1conv'``): ``conv`` ``Conv2d(C, C, 3, 1, 1)`` (:705)
- ``norm`` LayerNorm(C) (:988), ``conv_after_body`` ``Conv2d(C, C, 3, 1, 1)`` (:992)
- the tail (:1002-1022), already pinned by step 6.1 (D-022)

``self.mean`` is a plain tensor and ``relative_position_index_SA`` a buffer:
neither is a parameter. Before step 7 the body had no patch-embed or final
LayerNorm, no per-stage conv, no bias table, a 3x3 per-window LePE and an MLP:
the light body was 440,180 against the official 769,904 by the sum below, and
the base body 14,500,560 against 19,335,780.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.super_resolution.pft_sr import PFTSR, create_pft_sr


def _conv(i, o, k=3):
    return k * k * i * o + o


def _linear(i, o, bias=True):
    return i * o + (o if bias else 0)


def _depthwise(c, k):
    return k * k * c + c


def _ln(c):
    return 2 * c


def official_body_params(embed_dim, depths, num_heads, window_size, mlp_ratio,
                         convffn_kernel_size, in_chans=3, qkv_bias=True):
    """The ``PFT`` body: everything before the reconstruction tail."""
    c, k = embed_dim, convffn_kernel_size
    hidden = int(c * mlp_ratio)
    table = (2 * window_size - 1) ** 2 * (num_heads if c > 100 else 1)
    layer = (_ln(c) + _linear(c, 3 * c, qkv_bias) + _depthwise(c, k) + table
             + _linear(c, c) + _ln(c) + _linear(c, hidden) + _depthwise(hidden, k)
             + _linear(hidden, c))
    return (_conv(in_chans, c) + _ln(c) + sum(depths) * layer
            + len(depths) * _conv(c, c) + _ln(c) + _conv(c, c))


def official_tail_params(upsampler, embed_dim, scale, out=3, feat=64):
    if upsampler == "pixelshuffledirect":
        return _conv(embed_dim, scale * scale * out)
    stages = (_conv(feat, 9 * feat) if scale == 3
              else (scale.bit_length() - 1) * _conv(feat, 4 * feat))
    return _conv(embed_dim, feat) + stages + _conv(feat, out)


def _built_count(model, channels=3):
    model.build((None, None, None, channels))
    return sum(int(np.prod(w.shape)) for w in model.weights)


#: The two released configs (options/train/*.yml network_g), by hand.
_RELEASED = {
    "light": dict(embed_dim=52, depths=[2, 4, 6, 6, 6], num_heads=4, window_size=32,
                  mlp_ratio=1.0, convffn_kernel_size=7, upsampler="pixelshuffledirect"),
    "base": dict(embed_dim=240, depths=[4, 4, 4, 6, 6, 6], num_heads=6, window_size=32,
                 mlp_ratio=2.0, convffn_kernel_size=7, upsampler="pixelshuffle"),
}


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("variant", sorted(_RELEASED))
def test_a_released_variant_is_the_official_hand_sum(variant, scale):
    cfg = dict(_RELEASED[variant])
    upsampler = cfg.pop("upsampler")
    expected = official_body_params(**cfg) + official_tail_params(
        upsampler, cfg["embed_dim"], scale)
    keras.utils.set_random_seed(0)
    assert _built_count(create_pft_sr(scale, variant)) == expected


@pytest.mark.parametrize("variant", sorted(_RELEASED))
def test_a_headless_released_variant_is_the_official_body(variant):
    cfg = dict(_RELEASED[variant])
    cfg.pop("upsampler")
    keras.utils.set_random_seed(0)
    model = create_pft_sr(4, variant, include_top=False)
    assert _built_count(model) == official_body_params(**cfg)


@pytest.mark.parametrize("embed_dim, num_heads", [
    (16, 2),     # dim <= 100: one bias column shared by the heads
    (104, 8),    # dim > 100: one column per head
])
@pytest.mark.parametrize("qkv_bias", [True, False])
def test_the_bias_table_and_qkv_bias_follow_the_official_rule(embed_dim, num_heads, qkv_bias):
    kw = dict(embed_dim=embed_dim, num_heads=num_heads, window_size=4, mlp_ratio=1.5,
              convffn_kernel_size=5)
    model = PFTSR(scale=2, num_blocks=[1, 2], include_top=False, qkv_bias=qkv_bias, **kw)
    assert _built_count(model) == official_body_params(depths=[1, 2], qkv_bias=qkv_bias, **kw)


def test_an_explicit_build_and_a_call_build_the_same_weights():
    """The hand sum above reads an explicit ``build`` with unknown spatial
    dims; a call must not add or drop a weight."""
    kw = dict(scale=2, embed_dim=16, num_blocks=[2], num_heads=2, window_size=4,
              num_topk=[16, 8])
    built = PFTSR(**kw)
    count = _built_count(built)
    called = PFTSR(**kw)
    called(np.zeros((1, 8, 12, 3), "float32"))

    def paths(model):
        return sorted((w.path.split("/", 1)[1], tuple(w.shape)) for w in model.weights)

    assert paths(called) == paths(built)
    assert count == sum(int(np.prod(w.shape)) for w in called.weights)
