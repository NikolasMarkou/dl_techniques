# FastViT / MobileCLIP2 (MCi) Backbone Primitives

Channels-last Keras 3 transcriptions of the blocks that make up timm's FastViT
**MCi** image tower — the image encoder of Apple's MobileCLIP / MobileCLIP2.

```python
from dl_techniques.layers.fastvit import (
    FastVitConvMlp,
    RepConditionalPosEnc,
    FastVitRepMixer,
    FastVitRepMixerBlock,
    ReparamLargeKernelConv,
    FastVitPatchEmbed,
    FastVitAttentionBlock,
    FastVitStage,
)
```

## ⚠ Name collision: `FastVitRepMixerBlock` is NOT `RepMixerBlock`

This repository contains **two different architectures whose names both say
"RepMixer"**. They are unrelated implementations that happen to share a name in
their respective source lineages. Getting them confused silently ships a
different model.

| | `dl_techniques.layers.fastvit.FastVitRepMixerBlock` (here) | `dl_techniques.layers.repmixer_block.RepMixerBlock` (pre-existing) |
|---|---|---|
| Origin | timm `fastvit.py` `RepMixerBlock`, the FastViT / MobileCLIP MCi tower | a separate standalone block already in the tree |
| Token mixer | `FastVitRepMixer`: `x + gamma * (mixer(x) - norm(x))`, both arms being depthwise `MobileOneBlock`s, the `norm` arm degenerating to a single BatchNorm | a different mixer |
| Channel MLP | `FastVitConvMlp`: depthwise 7x7 + BN (no activation) -> 1x1 -> GELU -> drop -> 1x1 -> drop | a different FFN |
| LayerScale | yes, per-channel, on both the token-mixer and the MLP branch | none |
| Stochastic depth | yes, per residual branch | none |
| Consumed by | `models/mobile_clip_v2/` (this port) | **`models/fastvlm/`** |

**Which one do you want?**

* Building a **faithful FastViT / MobileCLIP2 image tower** — or anything that
  must correspond block-for-block to timm's `fastvit.py` — use
  `FastVitRepMixerBlock` from **this** package.
* Touching **`models/fastvlm/`**, or loading one of its checkpoints — that model
  consumes `layers/repmixer_block.py::RepMixerBlock`. It is deliberately left
  untouched by this package; substituting the FastViT block would change a
  shipped model's semantics and its checkpoint layout.

The same disambiguation applies to `layers/repmixer_block.py::ConvolutionalStem`,
which is likewise not the FastViT stem.

Everything in this package whose reference name is generic (`ConvMlp`,
`PatchEmbed`, `Stage`, `AttentionBlock`, `RepMixer`, `RepMixerBlock`) carries a
`FastVit` prefix, because the repo's serialization registry is keyed by bare
class name: a generic name silently shadows — or is shadowed by — an unrelated
class depending on import order, and the loser's `save`/`load` breaks. Names that
are already distinctive in the reference (`ReparamLargeKernelConv`,
`RepConditionalPosEnc`) are kept unprefixed.

## No structural reparameterization — deliberately

FastViT's "Rep" prefix refers to **structural reparameterization**: at inference
time the multi-branch train-time blocks (`MobileOneBlock`'s parallel k×k / 1×1 /
identity branches, `ReparamLargeKernelConv`'s large + small kernels) can be fused
into a single convolution.

**This package implements the train-time multi-branch form only. There is no
`reparameterize()` / fusion path, and none is planned here.**

This is a deliberate scope decision, not an omission:

* The fused form is **mathematically identical** to the multi-branch form, so
  nothing about model quality or correspondence to the reference depends on it.
  It is a pure inference-latency optimization.
* The reference implementation this port follows passes `inference_mode=False`
  everywhere, i.e. it too runs the multi-branch form.
* A fusion path would be a second, weight-surgery code path per block with its
  own correctness burden, and it cannot be validated against anything this port
  ships (no pretrained weights are ported).

If you need the fused form, implement it as an explicit, separately tested
conversion pass over a trained model — do not silently change these blocks'
`call()` paths.

## Components

| Class | Reference name | What it is |
|---|---|---|
| `FastVitConvMlp` | `ConvMlp` | depthwise 7x7 + BN (no act) -> 1x1 (biased) -> GELU -> drop -> 1x1 (biased) -> drop. Kernels `TruncatedNormal(0.02)`, biases zero. |
| `RepConditionalPosEnc` | `RepConditionalPosEnc` | depthwise 7x7 biased conv, `padding='same'`, output `conv(x) + x`. |
| `FastVitRepMixer` | `RepMixer` | `x + gamma * (mixer(x) - norm(x))`; `norm` is a degenerate single-BatchNorm `MobileOneBlock`, `mixer` a depthwise `MobileOneBlock(use_act=False)`. |
| `FastVitRepMixerBlock` | `RepMixerBlock` | `x = token_mixer(x)`, then `x = x + drop_path(gamma2 * ConvMlp(x))`. |
| `ReparamLargeKernelConv` | `ReparamLargeKernelConv` | large-kernel conv + optional small-kernel conv (both Conv+BN, no act), optional SE, optional activation. |
| `FastVitPatchEmbed` | `PatchEmbed` | `ReparamLargeKernelConv(k=7, stride=2, depthwise, small_kernel=3)` -> `MobileOneBlock(k=1)`. The downsample. |
| `FastVitAttentionBlock` | `AttentionBlock` | pre-norm global self-attention + ConvMlp, LayerScale + stochastic depth on both residual branches. Owns the rank-4 <-> rank-3 flatten. |
| `FastVitStage` | `FastVitStage` | optional `FastVitPatchEmbed` -> optional `RepConditionalPosEnc` -> `depth` x (RepMixer block \| attention block). |

The `MobileOneBlock` these build on is the shared
`dl_techniques.layers.mobile_one_block.MobileOneBlock`, extended additively (every
new kwarg defaults to its previous behaviour) rather than copied.

## `FastVitStage` — usage and the drop-path contract

```python
from dl_techniques.layers.fastvit import FastVitStage

stage = FastVitStage(
    dim=128,
    depth=4,
    token_mixer='repmixer',        # or 'attention'
    downsample=True,               # False for the first stage (the stem did the /4)
    use_pos_emb=False,             # True for the attention stages of mci3 / mci4
    mlp_ratio=4.0,
    drop_path_rates=[0.00, 0.02, 0.04, 0.06],   # EXACTLY `depth` entries
)
y = stage(x, training=False)       # (B, H, W, C_in) -> (B, H/2, W/2, 128)
```

**The stage does not compute its own stochastic-depth schedule.** The reference
computes the schedule once, globally, across every block of every stage
(`calculate_drop_path_rates(rate, layers, stagewise=True)`) and hands each stage
its slice — stage 2 of a `[2, 12, 24, 4]` model starts where stage 1's rates
ended, not at zero. A per-stage computation cannot reproduce that. So
`drop_path_rates` is an explicit list of exactly `depth` floats supplied by the
caller (the encoder), and the stage's only job is to give element `i` to block
`i`. Use `dl_techniques.utils.drop_path.linear_drop_path_rates(total_blocks,
max_rate)` to build the global schedule, then slice it stagewise.

`None` means all zeros. Any other length raises `ValueError` naming both `depth`
and the length received.

Two further contracts worth knowing:

* The blocks live in a **flat** Python list, `stage.blocks`, named `block_0` …
  `block_{depth-1}`. The names do not encode the token mixer, so a
  repmixer/attention change does not move the variable paths.
* `StochasticDepth` short-circuits to the identity only when `training is False`
  (or the rate is exactly `0.0`); `training=None` runs the stochastic path. Pass
  `training=False` **explicitly** for deterministic behaviour — every block also
  contains BatchNormalization, which updates its moving statistics otherwise.

## Spatial geometry (256 px input, stem /4)

| Variant family | Per-stage spatial size |
|---|---|
| 4-stage (mci0 / mci1 / mci2) | 64 -> 64 -> 32 -> 16 -> 8 (attention at 8x8 = 64 tokens) |
| 5-stage (mci3 / mci4) | 64 -> 64 -> 32 -> 16 -> 8 -> 4 (attention at 64 and 16 tokens) |

`RepConditionalPosEnc`'s 7x7 depthwise kernel is legal on a 4x4 feature map under
`padding='same'`.

## Other recorded deviations from the reference

* **Attention projection bias.** timm's `Attention` is `qkv_bias=False` with a
  **biased** output projection. The shared `MultiHeadAttention` exposes a single
  `use_bias` covering both, so `use_bias=False` is used: it costs exactly one
  missing bias vector of length `dim` per attention block, whereas `use_bias=True`
  would add a spurious `3 * dim` qkv bias the reference does not have. Pinned by a
  weight-inventory test so it cannot drift silently.
* **Normalization epsilon.** Three different defaults are wrong here:
  `create_normalization_layer` `setdefault`s `epsilon=1e-6`, Keras'
  `BatchNormalization` defaults to `1e-3`, and the reference uses `1e-5`. The
  value is defined ONCE, as `reference.py::REFERENCE_NORM_EPSILON`, and passed
  explicitly at every construction site — including through
  `MobileOneBlock(norm_epsilon=...)`, whose own default stays at Keras' `1e-3`
  because `models/fastvlm/` shares that block. Pinned tower-wide by
  `test_all_batchnorms_use_reference_epsilon`, which asserts the epsilon
  HISTOGRAM of a built mci0 and mci3 tower is a single `1e-05` bucket.
  *(This bullet previously claimed every call site already passed it. It did not:
  MEASURED on a built mci0 tower, the histogram was `{0.001: 86, 1e-05: 28}`.
  Fixed, and the claim is now the assertion of a test.)*
* **Padding grid at stride > 1.** PyTorch pads symmetrically by `kernel_size // 2`;
  Keras' `padding='same'` pads asymmetrically (extra row/column at the
  bottom/right), so at stride 2 the sampled grid depends on the kernel size.
  MEASURED with Dirac kernels on `arange(16).reshape(1, 4, 4, 1)` at stride 2:
  `'same'` gives `[[0, 2], [8, 10]]` for `k=1` but `[[5, 7], [13, 15]]` for `k=3`,
  while symmetric `k//2` padding gives `[[0, 2], [8, 10]]` for both. Every
  convolution authored here therefore uses `padding_mode='reference'`
  (`ZeroPadding2D(k // 2)` + a `'valid'` convolution). Applied UNIFORMLY, not only
  at stride > 1: for an odd kernel at stride 1 the two conventions are
  value-identical, which is measured by
  `test_reference_mode_stride_one_odd_kernel_is_value_identical`, and the 256px
  geometry table above is unchanged (measured).
* **LayerScale constraint.** `LearnableMultiplier` defaults to
  `constraint='non_neg'`, which would clamp a legitimately negative gamma at zero
  during optimization. `constraint=None` is load-bearing here, not cosmetic.
* **No pretrained weights** are ported. Architecture only — no accuracy claim.

## Testing

`tests/test_layers/test_fastvit/`, one module per class. Every class carries
initialization, invalid-config, forward-shape, `compute_output_shape` (pre- and
post-build) and `.keras` save->load VALUE round-trip tests, plus per-class
behavioural pins that were each proven RED against a deliberately broken variant.

## References

* Vasu et al., 2023. *FastViT: A Fast Hybrid Vision Transformer using Structural
  Reparameterization.* https://arxiv.org/abs/2303.14189
* Vasu et al., 2024. *MobileCLIP: Fast Image-Text Models through Multi-Modal
  Reinforced Training.* https://arxiv.org/abs/2311.17049
* Touvron et al., 2021. *Going Deeper with Image Transformers* (LayerScale).
  https://arxiv.org/abs/2103.17239
* Huang et al., 2016. *Deep Networks with Stochastic Depth.*
  https://arxiv.org/abs/1603.09382
