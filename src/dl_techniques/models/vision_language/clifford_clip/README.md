# CliffordCLIP: CLIP with Clifford Geometric-Algebra Towers

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A CLIP-style contrastive dual encoder whose towers are `CliffordNet` geometric-algebra
blocks instead of attention, with a selectable Clifford-aware projection head. It shares
the sibling `dl_techniques.models.vision_language.clip` package's contrastive objective
and I/O contract, but not the tower internals — this is a separate architecture, not a
config of plain `CLIP`.

Based on: **"CliffordNet: All You Need is Geometric Algebra"** (Ji, 2026, arXiv:2601.06793v2)

---

## Table of Contents

1. [What CliffordCLIP Is](#1-what-cliffordclip-is)
2. [Architecture](#2-architecture)
3. [Why a Clifford Projection Head](#3-why-a-clifford-projection-head)
4. [Constructor Parameters](#4-constructor-parameters)
5. [Variants](#5-variants)
6. [Usage](#6-usage)
7. [Differences from Plain CLIP](#7-differences-from-plain-clip)
8. [Training](#8-training)
9. [Citation](#9-citation)

---

## 1. What CliffordCLIP Is

`CliffordCLIP` maps an image and a caption to the same unit sphere, supervised only by
which pairing in the batch is the true one, exactly as in standard CLIP. What differs is
how each tower mixes information: a `CliffordNetBlock` reads features as multivectors
over the channel axis and combines channel pairs at a fixed shift through the geometric
product `a b = <a, b> + a ^ b`, whose inner part behaves like a dot-product score and
whose wedge part carries the orientation a symmetric similarity discards. The shift set
is small and fixed rather than all-pairs, so cost is linear in sequence or spatial size.
The product mixes channels, not positions, so spatial and sequential context comes from a
depthwise convolution inside each block — bidirectional in the vision tower, causal in
the text tower.

The contrastive loss is **not** defined in this module;
`dl_techniques.losses.CLIPContrastiveLoss` matches this model's output schema. See
`src/train/clip/train_clip.py` for a training usage example of the CLIP family (the CliffordCLIP trainer `train/cliffordnet/train_clip.py` was removed from the tree; see its git history).

References (from the module docstring, `model.py`):

- Ji, 2026. CliffordNet: All You Need is Geometric Algebra. arXiv:2601.06793v2.
- Radford et al., 2021. Learning Transferable Visual Models From Natural Language
  Supervision. (https://arxiv.org/abs/2103.00020)
- Zhang et al., 2026. Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based
  Vision Encoders. arXiv:2603.06569v2.
- Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using Shifted
  Windows. (https://arxiv.org/abs/2103.14030)
- Touvron et al., 2021. Going deeper with Image Transformers.
  (https://arxiv.org/abs/2103.17239)
- Huang et al., 2016. Deep Networks with Stochastic Depth.
  (https://arxiv.org/abs/1603.09382)

---

## 2. Architecture

```
image [B, H, W, C]              text [B, L] int32
         |                              |
         v                              v
+----------------------+   +----------------------+
| vision tower         |   | text tower           |
+----------------------+   +----------------------+
         |                              |
         v                              v
   projection head                projection head
         |  L2 normalize                |  L2 normalize
         v                              v
  image_features [B, E]          text_features [B, E]
         '-----------> logits <------------------'
                         |  * exp(logit_scale), capped
                         v
        logits_per_image [B_img, B_txt], logits_per_text [B_txt, B_img]
```

Either tower runs alone: an image-only or a text-only input returns that tower's
features and no logit keys (§ 6).

`logit_scale` stays float32 even under a mixed-precision policy.

### Vision tower

```
image
         |
         v
+----------------------+
| stem  conv, stride p |
+----------------------+
         |  [B, H/p, W/p, stage_channels[0]]
         v  + vision_pos_embed (optional)
+----------------------+
| stage i blocks       |  depths[i] CliffordNetBlock layers, external residual
+----------------------+
         |
         v
+----------------------+
| patch_merge_i        |  halves H and W, emits 2 * src channels
+----------------------+
         |
         v
+----------------------+
| merge_proj_i         |  only when target != 2 * src
+----------------------+
         |  repeats at every stage boundary
         v
last stage [B, H', W', stage_channels[-1]]
```

The vision tower is **hierarchical**: a patch stem, then stages linked by
`PatchMerging` (mirroring Swin's downsampling, not its attention). Each
`CliffordNetBlock` is transform-only and wrapped in an external residual — the block
itself has no skip connection.

The patch stem is one `ConvStem` named `vision_stem`, the recipe of `CliffordNet`'s stem.
`vision_patch_size` 1 and 4 give `Conv(c/2) -> BN -> SiLU -> Conv(c) -> BN` (both convs
bias-free, 3x3, stride 1 or 2); 2 and any other size give `Conv(c) -> BN` (3x3 stride 2, or
kernel and stride equal to the patch) and follow `use_bias`. Every BatchNormalization takes
momentum 0.9 and epsilon 1e-3 (the Keras value the hand-rolled stem had; the norm factory's own
default is 1e-6). Its weights sit under `vision_stem/vision_stem_{conv1,norm1,conv2,norm2}/`.
The removed trainer (`train/cliffordnet/train_clip.py`) classified the sub-layer by the `vision_stem`
prefix.

### Text tower

The text tower is **isotropic**: token + position embeddings, then `text_depth`
`CausalCliffordNetBlock` layers, each depthwise-convolution-causal so position `i` only
reads positions `<= i`. The pooled anchor is the last non-pad token
(`input_ids != pad_token_id`), found via `last_non_pad_token`.

### Projection heads

| `head_kind` | `z_det` | `z_ctx` | output |
|:---|:---|:---|:---|
| `plain` | - | - | anchor |
| `mean_max` | mean pool | max (vision) / last token (text) | `geo(z_det, z_ctx)` |
| `learned_query` | mean pool | attention pool | `geo(z_det, z_ctx)` |
| `learned_query_residual` (default) | mean pool | attention pool | `anchor + scale * geo(z_det, z_ctx)` |

The anchor is the mean pool for vision and the last non-pad token for text. `geo` is
`SparseRollingGeometricProduct` from `dl_techniques.layers.geometric.clifford_block`, the
same primitive the backbone blocks use.

---

## 3. Why a Clifford Projection Head

Plain CLIP pools the backbone output to a single vector and compares two vectors by
cosine similarity — the *scalar* (coherence) term of the geometric product only. The
*bivector* (structural, orientation-carrying) term, half of what the Clifford backbone
computes, is thrown away at the last step if the head is a bare `Dense` + L2-normalize.

The default head (`head_kind="learned_query_residual"`) keeps the canonical CLIP anchor
(mean pool for vision, last-non-pad-token for text) as the base signal and adds a
`LayerScale`-gated residual carrying a geometric product of two pooled views
(`z_det`, an attention-pooled `z_ctx`) on top. `LayerScale` gamma starts near 1e-5, so
training begins indistinguishable from plain CLIP and only introduces wedge/inner
content as it measurably helps — the same `GatedGeometricResidual` pattern the backbone
blocks use internally. Three other `head_kind` values (`plain`, `mean_max`,
`learned_query`) exist for A/B comparison; see `src/train/cliffordnet/README.md` for the
CC3M-smoke sweep that motivated the default.

---

## 4. Constructor Parameters

Full parameter documentation is in the `CliffordCLIP` class docstring in `model.py`
(Sphinx/reST `:param:` entries) — this section summarizes the groups.

**Vision**: `image_size`, `image_channels`, `vision_patch_size` (stem stride; `1`/`2`/`4`
match `CliffordNet` stem variants, any other positive int is a generic single-conv
stem), `vision_stage_channels` / `vision_stage_depths` / `vision_stage_shifts` (the
preferred staged config — every shipped variant uses these), `vision_channels` /
`vision_depth` / `vision_shifts` (legacy single-stage path, used only when no stage
lists are given), `vision_cli_mode` (`"inner"` / `"wedge"` / `"full"`), `vision_ctx_mode`
(`"diff"` / `"abs"`), `vision_use_global_context`, `vision_stochastic_depth_rate`,
`vision_positional_encoding` (off by default — the geometric product mixes channels, not
space, so no positional weight exists unless this is set).

**Text**: `vocab_size` (must match the tokenizer producing `input_ids`; the shipped
trainer defaults to tiktoken `gpt2`, 50,257 tokens), `context_length`, `text_channels`,
`text_depth`, `text_shifts`, `text_cli_mode`, `text_ctx_mode`,
`text_use_global_context` (default `False`), `text_stochastic_depth_rate`.

**Shared**: `embed_dim`, `layer_scale_init`, `dropout_rate` (gates an optional
pre-projection `head_dropout` on both towers; `0.0` skips the sublayer entirely, matching
`CliffordNet` / `CliffordNetLM`), `pad_token_id`, `logit_scale_init` (default `2.6592`,
i.e. `exp(2.6592) ≈ 14.3`, matching the CLIP paper), `logit_scale_max` (default `100.0`,
matching OpenCLIP).

**Projection head**: `head_shifts` (shifts `>= channels` are filtered out; if all are
filtered, falls back to `[1]`), `head_cli_mode`, `head_kind` (see § 2 table above),
`use_bias`, `kernel_initializer`, `bias_initializer`, `kernel_regularizer`,
`bias_regularizer`. `use_bias` reaches every Dense except the two final projections
(`vision_projection`, `text_projection`), which are bias-free whatever it says, as OpenAI
CLIP's `x @ self.proj` is (since 2026-10-20; archives saved before carry two bias
variables this model no longer builds and do not load). Parameter counts per variant:
`src/dl_techniques/models/vision/cliffordnet/README.md`, Variants.

`get_config()` round-trips every constructor argument (staged vision fields, not the
legacy scalar ones); `from_config()` deserializes the two regularizers.

---

## 5. Variants

`CliffordCLIP.MODEL_VARIANTS`:

| Variant | Vision channels | Vision depths | Text channels | Text depth | `embed_dim` |
|:--------|:-----------------|:--------------|:--------------:|:----------:|:-----------:|
| `nano`   | 128,128,256,256 | 3,3,3,3 | 128 | 12 | 256 |
| `nano_g` | 128,128,256,256 | 3,3,3,3 | 128 | 12 | 256 |
| `mini`   | 192,192,384,384 | 3,3,3,3 | 192 | 12 | 384 |
| `small`  | 192,192,384,384 | 4,4,4,3 | 192 | 15 | 384 |
| `base`   | 256,256,512,512 | 4,4,4,4 | 256 | 12 | 512 |
| `large`  | 384,384,768,768 | 5,5,5,5 | 384 | 16 | 768 |

`nano_g` is `nano` with the vision global-context branch switched on (mirrors
the `lite_g` variant of `CliffordNet`). Both towers share depth and channels at each scale so the
contrastive temperature update sees balanced gradient magnitudes; `nano` / `nano_g` match
`CliffordNet` / `CliffordNetLM` `nano` (channels=128, depth=12, shifts=[1,2]) so the CLIP
backbone can be ablated against the vanilla classifier / LM at matched capacity.

`create_clifford_clip(variant, vocab_size, image_size, context_length, pretrained=False,
**kwargs)` is the module-level factory; it delegates to `CliffordCLIP.from_variant` with the
same arguments. `vocab_size`, `image_size` and `context_length` are required by both: they
describe the data (tokenizer, image resolution, caption length), not the variant.
`pretrained=True` or a path raises `NotImplementedError`: no weights are distributed, and a
local `.keras` archive loads with `keras.models.load_model(path)`. Use the factory or
`from_variant` for a preset, or the constructor directly for a custom config.

---

## 6. Usage

```python
import keras
from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP

model = CliffordCLIP.from_variant(
    "nano", vocab_size=100352, image_size=96, context_length=64,
)
images = keras.random.normal((4, 96, 96, 3))
tokens = keras.random.randint((4, 64), 0, 100277, dtype="int32")
out = model({"image": images, "text": tokens})
# out has keys: image_features, text_features,
# logits_per_image, logits_per_text, logit_scale

# One modality: that tower's features only, no logit keys (the vanilla CLIP contract).
model({"image": images})   # {"image_features"}
model({"text": tokens})    # {"text_features"}; the tuple form (None, tokens) too
```

A caption bank is encoded without a dummy image batch. `compute_output_shape` reports the
same keys for each input form; an input with neither modality raises `ValueError`.

Custom config (bypassing `from_variant`):

```python
from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP

model = CliffordCLIP(
    vocab_size=50257,
    image_size=112,
    context_length=64,
    vision_stage_channels=[192, 192, 384, 384],
    vision_stage_depths=[3, 3, 3, 3],
    text_channels=192,
    text_depth=12,
    embed_dim=384,
    head_kind="learned_query_residual",
)
```

Serialization round-trips through the standard Keras path (`@register_dl_technique`
under `dl_techniques.models.clifford_clip.model`):

```python
model.save("cliffordclip_nano.keras")
restored = keras.models.load_model("cliffordclip_nano.keras")
```

---

## 7. Differences from Plain CLIP

| | `clip.CLIP` | `clifford_clip.CliffordCLIP` |
|:---|:---|:---|
| Tower mechanism | `TransformerLayer` (GQA attention, RMSNorm, SwiGLU, RoPE) | `CliffordNetBlock` / `CausalCliffordNetBlock` (geometric product, no attention, no FFN) |
| Vision tower shape | Isotropic (CLS token + patch sequence) | Hierarchical (4 stages, `PatchMerging` between them) |
| Position encoding | RoPE inside attention | None inherent; optional learned 2D weight over the post-stem map (`vision_positional_encoding`) |
| Sequence cost | Attention is quadratic in sequence length | Depthwise convolution is linear |
| Projection head | Plain pooled anchor -> `Dense` | Selectable; default injects a Clifford geometric-product residual (§ 3) |
| Registration key | `dl_techniques.models.clip.model` | `dl_techniques.models.clifford_clip.model` |

Both share the CLIP call contract: with both modalities `call` returns `image_features`,
`text_features`, `logits_per_image`, `logits_per_text`, `logit_scale`; with one, only that
tower's features (until 2026-10-15 CliffordCLIP required both and raised `KeyError`). The
contrastive loss lives outside the model in `dl_techniques.losses.CLIPContrastiveLoss`.

---

## 8. Training

No public pretrained weights are distributed. The training entry point was
`src/train/cliffordnet/train_clip.py` (removed from the tree; see its git history), which wrote a single timestamped run directory
per launch:

```
results/cliffordclip_<variant>_<timestamp>/
```

(or `--output-dir`). It trains with stock `compile`/`fit` through the stacking bridge of
`src/train/clip/train_clip.py` (`compile_for_contrastive_training`), and every `.keras` it
writes, step checkpoints included, is the bare `CliffordCLIP`, never the training wrapper.

See `src/train/cliffordnet/README.md` for the full protocol, flags and CC3M-smoke
head-variant sweep results.

---

## 9. Citation

```bibtex
@article{ji2026cliffordnet,
  title={CliffordNet: All You Need is Geometric Algebra},
  author={Ji},
  journal={arXiv preprint arXiv:2601.06793},
  year={2026}
}
```

CliffordCLIP additionally builds on:

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya
          and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda
          and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2021}
}
```

---

## Archives

**Archive break.** Measured by saving the four roster subjects (`nano`, `vision_patch_size` 1 to 4) in a
separate process with the code of `be92a74eb` (`git archive`, `python -I`) and restoring them with the current
code. The vision stem was four tracked layers (`vision_stem_conv1`, `vision_stem_bn1`, `vision_stem_conv2`,
`vision_stem_norm`; two for patch 2 and 3) and is now one `ConvStem`:

- `keras.models.load_model(path)` and `model.load_weights(path)` into a freshly built model both refuse all four
  archives: `ValueError: A total of 4 objects could not be loaded ... Layer 'vision_stem_conv1' expected 1
  variables, but received 0 variables during loading` (2 objects, `expected 2 variables`, for patch 2 and 3).
  The four (or two) objects are exactly the stem's sub-layers; no other object is named.
- Transplanting by `w.path` restores every tensor of all four subjects (373, 369, 369 and 373 tensors) with
  inference outputs equal at `atol=0`, once the old stem paths are renamed: `vision_stem_conv1/*` and
  `vision_stem_conv/*` to `vision_stem/vision_stem_conv1/*`, `vision_stem_bn1/*` to
  `vision_stem/vision_stem_norm1/*`, `vision_stem_conv2/*` to `vision_stem/vision_stem_conv2/*`, and
  `vision_stem_norm/*` to `vision_stem/vision_stem_norm2/*` (patch 1 and 4) or `.../vision_stem_norm1/*` (patch 2
  and 3). The weight order does not change.
- A model built with a given seed does NOT have the same starting weights as before. The stem's second
  convolution, every kernel of the vision and text blocks, the merge and projection kernels, the two token tables and
  the attention pools' `context_vector` are new draws; only the first stem convolution (`vision_stem_conv1`)
  and the constants (biases, norm scales and offsets and moving statistics, LayerScale gains, `logit_scale`) are
  equal to the earlier build (136 of 373 tensors differ at patch 1 and 4, 135 of 369 at patch 2 and 3). The cause is
  that the earlier code handed one initializer instance to several kernels (the two stem convolutions, every
  sub-layer of every block of a stage, the two projections: 776 of 9,316 kernel pairs of `nano` carried one draw), and a
  kernel now draws its own.
- Transcripts: `plans/plan-2026-10-26T125959-7af851e5/scratch_iter2/step20/archive_summary.txt`,
  `archive_load_model_be92a74eb.txt`, `archive_load_weights_be92a74eb.txt`, `transplant_by_path_clip.txt`.
