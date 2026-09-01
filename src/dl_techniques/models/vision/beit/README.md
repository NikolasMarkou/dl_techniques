# BEiT: BERT Pre-Training of Image Transformers

A Keras 3 implementation of **BEiT** (Bao, Dong, Piao & Wei, *BEiT: BERT Pre-Training of Image
Transformers*, ICLR 2022, [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)) — a pre-norm
Vision Transformer trained with a **masked image modeling** (MIM) objective that predicts
*discrete visual token ids* rather than raw pixels.

One shared trunk, two consumers that compose it under the same layer name with **disjoint head
prefixes**, so a classifier warm-starts from an MIM checkpoint layer-for-layer:

| Class | Head prefix | Output |
|:---|:---:|:---|
| `BeitModel` | — | `(B, N+1, D)` token sequence, cls first |
| `BeitForMaskedImageModeling` | `decoder_` | `(B, N, vocab_size)` logits, cls excluded |
| `BeitForImageClassification` | `head_` | `(B, num_classes)` logits |

> **No pretrained weights are distributed, and there is no `pretrained=` argument anywhere in this
> package.** Every constructor and factory returns a randomly-initialized model. To get a
> pre-trained trunk, run the pipeline in `src/train/beit/` and transfer the resulting checkpoint
> with `load_weights_from_checkpoint` (§ 9.1).

> **Six recorded deviations from the reference** — most importantly, the discrete visual tokenizer
> is a **VQ-VAE**, not BEiT v1's Gumbel-softmax DALL·E dVAE. Comparison against published BEiT
> numbers is **invalid by construction**. Read § 15 before citing anything. Architecture facts in
> this file come from directly-fetched primary sources: `microsoft/unilm/beit/*.py`,
> `timm/models/beit.py` and the raw HF `config.json` files.

## 1. Overview: What is BEiT and Why It Matters

BEiT transplants BERT's masked-token pre-training into vision. An image patch is a continuous vector,
not a symbol, so there is nothing to "predict the id of"; BEiT gives every image **two views**. The
**image view** is the usual patch grid fed to the ViT encoder, with ~40% of positions replaced by a
learnable `[MASK]` token; the **token view** is the same image through a *frozen* discrete visual
tokenizer, one integer code id per patch. The encoder sees the corrupted image view and predicts the
token view's code id at each masked position — plain cross-entropy over a codebook.

Four properties of this package: everything except the attention is an existing layer from
`dl_techniques.layers` and exactly one new layer was authored (§ 4.1); the trunk is shared and
separately checkpointable, making the warm start a 1:1 transfer rather than a hopeful name match;
there is **no custom `train_step`**, because the mask reaches the loss as `sample_weight`; and both
heads emit **logits**, so compile with `from_logits=True`.

## 2. The Problem BEiT Solves

Pre-BEiT self-supervised vision had two camps. **Pixel regression** targets raw pixels, so the loss
is dominated by high-frequency texture the model does not need. **Contrastive / instance
discrimination** targets "is this the same image?", needing large batches, negative sampling and
careful augmentation. BEiT inserts a **discretization step** — `image -> frozen tokenizer -> code
ids -> cross-entropy` — discarding exactly the detail that made pixel regression a poor objective
and restoring the batch-size-insensitive loss that made BERT practical.

## 3. How BEiT Works: Core Concepts

**Three stages**, with stages 0 and 1 wired end to end in `src/train/beit/`: train a discrete visual
tokenizer and freeze it; block-mask ~40% of patch positions and predict the code id there, giving a
`BeitForMaskedImageModeling` checkpoint; then warm-start a `BeitForImageClassification` from that
trunk and fine-tune on labels.

**Block-wise masking, not i.i.d. masking.** BEiT stamps rectangular blocks with a log-uniform aspect
ratio into the patch grid until a budget is reached, so a masked patch is usually surrounded by
masked ones and long-range reasoning is required, where i.i.d. masking leaves every masked patch with
unmasked neighbours. `BeitMaskingGenerator` (`dl_techniques/datasets/vision/beit_masking.py`)
transcribes the official `masking_generator.py`, quirks included: the 10-attempt retry, the strict
`h < H` / `w < W` rejection, and the **early-termination under-fill** — after 10 consecutive failed
placements it returns a mask with *fewer* cells set, without raising. Preserved on purpose.

**The mask token replaces, it does not drop.** Unlike MAE, BEiT never removes tokens: at a masked
position the patch embedding is *substituted* by a shared learnable `[MASK]` vector, and the
transformer processes the full `N+1` sequence.

**The loss is restricted by `sample_weight`, not by code.** The `tf.data` element is
`((image, bool_mask), target_ids, sample_weight)` with `sample_weight = cast(bool_mask, float32)`
exactly, so Keras zeroes the unmasked positions. **No `train_step`, `test_step` or `compute_loss`
override exists in this package, and none may be added.**

## 4. Architecture Deep Dive

```
   image (B, H, W, 3)                    bool_mask (B, N)  [MIM only]
     PatchEmbedding2D  Conv2d(k=p, s=p)                        │
     MaskTokenApply  x[mask] = mask_token  ◄────────────────────┘
       always CREATED and BUILT; called only when a mask is passed
     ClassTokenPrepend                            -> (B, N+1, D)
     [ optional absolute position embedding -- OFF by default ]
     num_layers x TransformerLayer(attention_type='beit'):
       LayerNorm(1e-12) -> BeitAttention  <- THE ONLY NEW LAYER
         -> StochasticDepth -> LayerScale g1 -> + residual
       LayerNorm(1e-12) -> MLP(GELU)
         -> StochasticDepth -> LayerScale g2 -> + residual
     [ final LayerNorm -- ONLY when use_mean_pooling=False, § 15.4 ]
         ╱                              ╲
 decoder_norm -> decoder_head     head_pool (cls excluded) -> head_norm
   (B, N, vocab)                    -> head_dropout -> head_classifier
```

Everything above except `BeitAttention` is an existing, tested layer: `PatchEmbedding2D`,
`MaskTokenApply`, `ClassTokenPrepend`, `PositionalEmbedding`, `TransformerLayer` (which brings
`LayerScale`, `StochasticDepth`, the norms and the MLP), `linear_drop_path_rates` and
`SequencePooling`. `TransformerLayer`'s signature was **not** changed: it gained one `'beit'` case
in its attention-parameter table, and the attention factory one `'beit'` registry entry.

> **Order note (measured).** `TransformerLayer` applies `StochasticDepth` *before* `LayerScale`,
> where the reference writes `x + drop_path(gamma * attn(x))`. They are numerically identical —
> `StochasticDepth` multiplies the whole sample by a scalar and LayerScale elementwise by `gamma`
> broadcast over the batch, so they commute exactly. Do not "fix" this.

### 4.1 `BeitAttention` — why one new layer was unavoidable

**Asymmetric QKV bias (q and v only).** BEiT's key projection has *no bias parameter at all* —
structurally absent, not zero-initialized and not frozen. Every existing attention class here
exposes one `use_bias` / `qkv_bias` flag governing Q, K and V together, and
`MultiHeadCrossAttention` fuses the three into a single `Dense(dim * 3)`, putting the asymmetry
structurally out of reach.

**A cls-augmented relative position bias**, added to the logits before the softmax as
`A_h = softmax(Q_h K_hᵀ / √d_h + T[R, h])`, where `T` is a `(M, num_heads)` learnable table and `R`
a static integer buffer over the patch grid:

```
   R[i, j] = ((y_i − y_j) + Wh − 1)·(2·Ww − 1) + ((x_i − x_j) + Ww − 1)   patch↔patch
   R[0, j] = M − 3   R[i, 0] = M − 2   R[0, 0] = M − 1
   M = (2·Wh − 1)(2·Ww − 1) + 3
```

The `+3` rows cover cls→token, token→cls and cls→cls, which have no 2D displacement. The
repository's Swin-family tables are inlined inside `WindowAttention`, window-scoped, square
`(2W − 1)²` and have no cls slots, so their index arithmetic is a *different function of the window
size*. Only `T` is learned; non-square grids are supported and tested. The three extra rows exist
because the cls token has no grid position, so those relations have no displacement to index by.

### 4.2 Two constants that bite

**`window_size` is the PATCH GRID** for `'beit'` — the `(Wh, Ww)` grid of the whole image, not the
scalar edge length the Swin-family types take; `BeitModel` computes it for you.
**`layer_norm_eps = 1e-12`**, six orders of magnitude tighter than a generic ViT's `1e-6`, is passed
explicitly at every normalization site; copy-pasting a generic ViT block here would silently change
the architecture.

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.vision.beit import create_beit_classifier, create_beit_mim

# A classifier. build() is optional before fit(); it lets summary()/count_params() run.
clf = create_beit_classifier("tiny", (224, 224, 3), 16, num_classes=10)
clf.build((None, 224, 224, 3))
print(clf(np.random.rand(2, 224, 224, 3).astype("float32"), training=False).shape)
# (2, 10)   -- LOGITS, not probabilities

# The MIM model. Its second input is a boolean per-patch mask.
mim = create_beit_mim("tiny", (64, 64, 3), 16, vocab_size=512)
mim.build((None, 64, 64, 3))
mask = np.zeros((2, 16), dtype=bool)     # 4x4 patch grid -> N = 16
mask[:, :6] = True                       # mask 6 of the 16 positions
print(mim((np.random.rand(2, 64, 64, 3).astype("float32"), mask), training=False).shape)
# (2, 16, 512)  -- the cls position is already excluded

mim.compile(optimizer="adamw",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# The mask reaches the loss as `sample_weight` in the tf.data element -- § 8.2.
```

## 6. Component Reference

**`BeitModel(input_shape=(224,224,3), patch_size=16, scale='base', ...)`** — the trunk. Emits the
**full** `(B, N+1, D)` sequence, cls first. Accepts `image`, `(image, bool_mask)` or
`{'images': ..., 'mask': ...}`. Named `beit_backbone` (`BACKBONE_NAME`) by default — **do not rename
it** on a model that must warm-start; `load_weights_from_checkpoint` matches *by name*. Every
argument below is serialized by `get_config()`.

| Argument | Default | Notes |
|:---|:---:|:---|
| `patch_size` | `16` | `int` or `(h, w)` |
| `scale` | `'base'` | key of `SCALE_CONFIGS`, or a `'beit_*'` variant spelling |
| `layer_norm_eps` | `1e-12` | § 4.2 |
| `drop_path_rate` | `0.1` | maximum of the linear ramp over `num_layers` |
| `hidden_dropout_rate` / `attention_probs_dropout_rate` | `0.0` | off during pre-training |
| `use_absolute_position_embeddings` / `use_relative_position_bias` / `use_shared_relative_position_bias` | `False` / `True` / `False` | BEiT uses per-layer relative bias; `shared=True` **raises** (§ 14) |
| `use_mean_pooling` | `True` | also controls the trunk's final norm, § 15.4 |
| `hidden_size` / `num_layers` / `num_heads` / `intermediate_size` / `layer_scale_init_value` | `None` | `None` means "take it from `scale`" |

**`BeitForMaskedImageModeling(backbone, vocab_size=8192)`** — trunk → `decoder_norm` →
`decoder_head` → `(B, N, vocab_size)` logits. The cls position is sliced off **before** the head, so
output index `i` is patch `i`; emitting `N+1` logits would put every target off by one with no error
anywhere. **`BeitForImageClassification(backbone, num_classes, dropout_rate=0.0)`** — trunk →
pooling → `head_norm` → `head_dropout` → `head_classifier` → logits, with pooling following the
backbone's `use_mean_pooling` (§ 15.4).

**Factories** — `create_beit_backbone(variant, input_shape, patch_size, **overrides)`,
`create_beit_mim(..., vocab_size=8192, **overrides)` and
`create_beit_classifier(..., num_classes=1000, dropout_rate=0.0, **overrides)` forward `**overrides`
verbatim to the constructor. Prefer them over the classes: they name the backbone `BACKBONE_NAME`,
which is a warm-start precondition.

Elsewhere: `dl_techniques.layers.attention.BeitAttention` (factory key `'beit'`);
`dl_techniques.datasets.vision.beit_masking` for masking and the `tf.data` map fn;
`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint` for the warm start;
`src/train/beit/` for the trainers. `BACKBONE_NAME` and `DEFAULT_VOCAB_SIZE` (`8192`) are constants.

## 7. Configuration & Model Variants

Parameter counts are **measured** at `create_beit_backbone(scale, (224, 224, 3), 16)` then
`.build((None, 224, 224, 3))` — backbone only, no head, a 196-patch grid.

| Scale | `hidden_size` | Layers | Heads | FFN | `layer_scale_init_value` | Backbone params | Upstream? |
|:---:|:---:|:---:|:---:|:---:|:---:|---:|:---|
| `tiny` | 192 | 12 | 3 | 768 | 0.1 | 5,515,056 | **repo invention** — no BEiT this size in the paper, HF or timm |
| `small` | 384 | 12 | 6 | 1536 | 0.1 | 21,646,944 | **repo invention** — same |
| `base` | 768 | 12 | 12 | 3072 | 0.1 | 85,761,216 | `microsoft/beit-base-patch16-224` |
| `large` | 1024 | 24 | 16 | 4096 | **1e-5** | 303,404,544 | `microsoft/beit-large-patch16-224` (but § 15.2) |

> **No accuracy, throughput or convergence number appears in this README, and that is deliberate.**
> Nothing here has ever trained a BEiT to a published benchmark, and a number in a README is
> indistinguishable from a measurement to every reader. For reference numbers read the paper; for
> numbers about *this* code, train it and measure.

`MODEL_VARIANTS` maps `'beit_tiny'` … `'beit_large'` onto those scales, and **both spellings are
accepted everywhere a variant is taken**. Any scale field can be overridden individually; `None`
means "inherit from the scale".

```python
from dl_techniques.models.vision.beit import BeitModel, create_beit_backbone

a = BeitModel.from_variant("beit_tiny", input_shape=(64, 64, 3), patch_size=16)
b = create_beit_backbone("tiny", (64, 64, 3), 16)
print(a.scale, b.scale, a.grid_size, b.num_patches)   # tiny tiny (4, 4) 16

m = create_beit_backbone("base", (224, 224, 3), 16,
                         drop_path_rate=0.2, hidden_dropout_rate=0.1, num_layers=6)
print(m.num_layers, m.hidden_size, len(m.drop_path_rates))   # 6 768 6
```

## 8. Comprehensive Usage Examples

### 8.1 Non-square images

Rectangular images work: the relative-position index is built for a general `(Wh, Ww)` grid and the
only requirement is divisibility, since there is no padding path.
`create_beit_backbone("tiny", input_shape=(64, 96, 3), patch_size=16)` gives `grid_size == (4, 6)`
and an output of `(B, 25, 192)`.

### 8.2 A complete MIM `tf.data` pipeline

`make_beit_mim_map_fn` builds the `((image, bool_mask), target_ids, sample_weight)` element, with
`sample_weight` exactly the mask and no rescaling. You supply `tokenizer_fn`, mapping one
**unbatched** image to per-patch code ids in TensorFlow ops, because it runs inside the `tf.data`
graph. `src/train/beit/` supplies the real one from a trained VQ-VAE.

```python
import keras, numpy as np, tensorflow as tf
from dl_techniques.datasets.vision.beit_masking import make_beit_mim_map_fn
from dl_techniques.models.vision.beit import create_beit_mim

VOCAB = 512

def fake_tokenizer_fn(image):          # (H, W, C) -> (gh, gw) int code ids
    pooled = tf.nn.avg_pool2d(image[None], ksize=16, strides=16, padding="VALID")[0]
    return tf.cast(tf.reduce_mean(pooled, axis=-1) * (VOCAB - 1), tf.int32)

map_fn = make_beit_mim_map_fn(tokenizer_fn=fake_tokenizer_fn, grid_size=(4, 4),
                              num_masking_patches=6, min_num_patches=2)
ds = tf.data.Dataset.from_tensor_slices(
    np.random.rand(8, 64, 64, 3).astype("float32")).map(map_fn).batch(4)
(img_b, mask_b), targets_b, weights_b = next(iter(ds))
print(img_b.shape, mask_b.shape, targets_b.shape, weights_b.shape)
# (4, 64, 64, 3) (4, 16) (4, 16) (4, 16)

mim = create_beit_mim("tiny", (64, 64, 3), 16, vocab_size=VOCAB)
mim.compile(optimizer="adamw",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
mim.fit(ds, epochs=1, verbose=0)
```

`BeitMaskingGenerator` also produces masks standalone; its `rng=` argument is optional because the
reference uses the global `random` module, and the returned mask can hold *fewer* cells than
requested (§ 3).

### 8.3 The attention layer on its own

`create_attention_layer("beit", dim=192, window_size=(4, 4), num_heads=3)` returns a
`BeitAttention` expecting `Wh*Ww + 1` tokens; on it, `k_dense.use_bias` is `False` while
`q_dense.use_bias` is `True`. A whole BEiT block through `TransformerLayer` needs
`attention_type="beit"`, `window_size=(Wh, Ww)`, `attention_norm_args={"epsilon": 1e-12}`,
`ffn_norm_args={"epsilon": 1e-12}`, `use_layer_scale=True` and `use_stochastic_depth=True`.

## 9. Advanced Usage Patterns

### 9.1 Warm-starting a classifier from an MIM checkpoint

The pattern the whole prefix discipline exists to serve. **Three preconditions, each of which fails
silently if violated**: the target must be **built before** the transfer; the two backbones must share
the identical **name** and config; and the transfer must be **asserted**, because
`load_weights_from_checkpoint` does *not* raise on a zero-layer trunk transfer.

```python
import os, tempfile
from dl_techniques.models.vision.beit import (
    BACKBONE_NAME, create_beit_mim, create_beit_classifier)
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

CFG = dict(variant="tiny", input_shape=(64, 64, 3), patch_size=16)
mim = create_beit_mim(vocab_size=512, **CFG)
mim.build((None, 64, 64, 3))
clf = create_beit_classifier(num_classes=10, **CFG)
clf.build((None, 64, 64, 3))          # (1) BEFORE the transfer

with tempfile.TemporaryDirectory() as tmp:
    ckpt = os.path.join(tmp, "beit_mim.keras")
    mim.save(ckpt)
    report = load_weights_from_checkpoint(
        target=clf, ckpt_path=ckpt, skip_prefixes=("decoder_", "head_"))

# (3) ASSERT the transfer happened -- never just log the report. A value check on
# mim.backbone.get_weights() vs clf.backbone.get_weights() confirms the 1:1 move.
assert BACKBONE_NAME in report.loaded, report.summary_string()
assert BACKBONE_NAME not in [n for n, _, _ in report.shape_mismatch]
print(sorted(report.skipped_by_prefix))       # ['decoder_head', 'decoder_norm']
```

`src/train/beit/train_classification.py` wraps this in a `warm_start_encoder` helper that raises on
both failure modes. Note `MaskTokenApply` is created and built by *every* backbone, the classifier's
included, even though it never calls it: without it the trunks would not be weight-identical and the
name-matched transfer would move a different set of layers.

### 9.2 Linear probing, features, and the pooling fork

```python
from dl_techniques.models.vision.beit import create_beit_classifier

clf = create_beit_classifier("tiny", (64, 64, 3), 16, num_classes=10)
clf.build((None, 64, 64, 3))
clf.backbone.trainable = False
print(len(clf.trainable_weights), len(clf.weights))
# 4 224  <- head kernel+bias and head_norm gamma+beta, out of 224 tensors
```

For raw features call the trunk directly: it emits the full token sequence with cls at index 0, so
`tokens[:, 0, :]` is the cls feature and `tokens[:, 1:, :].mean(1)` is BEiT's own pooling. Passing
`use_mean_pooling=False` switches to cls pooling and moves the norm INTO the trunk
(`clf.backbone.final_norm` becomes non-`None` while `head_pool` and `head_norm` become `None`; see
§ 15.4). That changes the **layer set**, so an MIM checkpoint trained at `True` will not transfer
cleanly into a `False` classifier — keep the flag identical across stages.

## 10. Performance Optimization

**Sequence length is quadratic**: `N = (H/p)·(W/p)`, so 196 tokens at `224/16` and 576 at `384/16`,
i.e. `(576/196)² = 8.64x` the attention FLOPs — analytic, **not measured here**. The bias also
materializes a `(num_heads, N+1, N+1)` tensor per block. **`training=None` is not inference**:
dropout and stochastic depth stay live, so pass `training=False` for a deterministic pass.
**Mixed precision works as shipped** — `layer_norm_eps=1e-12` is below float16's smallest normal
(~6.1e-5), but it is added to a variance inside a `LayerNormalization` that Keras runs in its
variable dtype, so only forcing a norm into float16 compute needs a bigger epsilon. The cheapest
levers are a smaller `variant`, a larger `patch_size`, and freezing the trunk (§ 9.2).

## 11. Training and Best Practices

Rules this package enforces: dropout is off during pre-training (regularization comes from stochastic
depth); weight decay comes from the optimizer only, never also from a `kernel_regularizer`;
`from_logits=True` always; no custom `train_step`; and the backbone config must stay byte-identical
across stages 1 and 2 or the warm start transfers only part of the trunk. `model.build(...)` before
`compile()`/`fit()` is optional — a lazy build inside the traced training step works — but the
stage-2 warm start *does* require a built target (§ 9.1). BEiT v1's masking budget is 75 masked
patches with a 16-patch minimum block on a 14x14 grid (38.3%), exported as `BEIT_NUM_MASK_PATCHES`
and `BEIT_MIN_MASK_PATCHES_PER_BLOCK`; scale it down yourself on a smaller grid, because
`BeitMaskingGenerator` raises if the budget exceeds the grid.

The paper's appendix gives Adam at peak LR 1.5e-3, weight decay 0.05, 10 warmup epochs, 800 epochs at
batch 2048 and stochastic depth 0.1 for MIM pre-training, and AdamW with layer-wise LR decay 0.65
(base) / 0.75 (large) for fine-tuning — **quoted, not measured here.**

## 12. Serialization & Deployment

All three classes register through `@register_dl_technique("dl_techniques.models.beit.model")` and
round-trip through `.keras` with **value** equality (asserted at `atol=1e-6` in the suite);
`keras.models.load_model` needs no `custom_objects` and the restored trunk keeps the name
`beit_backbone`. `BeitModel.from_config` is a real override: `get_config()` emits `patch_size` as a
tuple, JSON turns it into a list, and `cls(**config)` would then store a `TrackedList`, so
`get_config()` would stop being a fixed point.

## 13. Testing & Validation

Suites: `tests/test_models/test_beit/`, `tests/test_layers/test_attention/test_beit_attention.py`,
`tests/test_datasets/test_beit_masking.py`,
`tests/test_layers/test_transformers/test_transformer_beit_integration.py`,
`tests/test_train/test_beit/`. Run **one directory per process** with
`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg`: two concurrent BEiT pytest processes on one GPU have been
measured to manufacture failures neither produces alone. The guards worth knowing encode facts, not
shapes: `TestBeitWarmStart` fails on a **zero-layer** transfer instead of passing it;
`TestBeitAttentionBiasOrientation` pins `bias[h, query, key]` on a non-square grid (without it a
transposed bias left 97/97 attention and 122/122 model tests green); and
`test_the_head_reads_the_patch_tokens_not_a_shifted_window` pins the cls slice by identity
(`[:, 1:, :] -> [:, :-1, :]` left 91/91 model tests green before it existed).

## 14. Troubleshooting & FAQs

- **`ValueError: use_shared_relative_position_bias=True is not implemented`.** Only per-layer tables
  are implemented, which is what every shipped BEiT/BEiTv2 *fine-tuned* variant uses. Leave it
  `False` — but not because it is harmless: the shared table is what BEiT v1 uses during
  pre-training, and `src/train/beit/train_mim.py` *is* pre-training. See § 15.6.
- **The MIM logits and target ids are off by one position.** A head that does not slice the cls
  token off. `BeitForMaskedImageModeling` outputs `(B, N, vocab)` precisely so output index `i` is
  patch `i`. If you write your own head, slice `tokens[:, 1:, :]` before projecting. This failure
  produces a finite, plausible loss curve and no error.
- **A `TypeError` mentioning `window_size`.** It is a `(Wh, Ww)` patch grid for `'beit'`, not a
  scalar edge length (§ 4.2).
- **The warm start "succeeded" but the classifier trains as if from scratch.** A renamed backbone,
  or a target that was not built first, produces a report that looks fine. Assert
  `BACKBONE_NAME in report.loaded` (§ 9.1); never just log the report.
- **`ValueError: Image height must be divisible by patch height`.** The grid must be exact; there is
  no padding path. Non-square grids are fine (§ 8.1).
- **Can I load `microsoft/beit-base-patch16-224` weights?** Not without writing a converter, and the
  result would not be faithful anyway (§ 15). There is no `pretrained=` argument.
- **Why does the key projection have no bias?** Because BEiT's does not: the reference concatenates
  `[q_bias, zeros_like(v_bias, requires_grad=False), v_bias]` into the fused QKV bias.

## 15. Deviations from the reference implementation

**Read this before comparing anything from this package to published BEiT results.** Six deviations
are recorded; three are pinned by a named test (§ 15.2, § 15.4, § 15.6), the rest are configuration
choices with no reference behaviour to assert against.

**15.1 The visual tokenizer is a VQ-VAE, not a Gumbel-softmax DALL·E dVAE (X-1).** BEiT v1's MIM
target comes from a frozen DALL·E dVAE trained by OpenAI on data this repository does not have.
There is no Gumbel-softmax codebook mechanism here, so `src/train/beit/` trains a
`VQVAERotationTrick` (`dl_techniques.models.vision.vq_vae_rotation.model`; that package's
`__init__.py` is empty, so import from the submodule) as stage 0 and uses its `encode_to_indices`
output as the MIM target. No temperature annealing, no soft relaxation, a different codebook
geometry — **any comparison to published BEiT accuracy is invalid by construction**. BEiT v2 itself
replaced the dVAE with a VQ-style target, so this is a recognised variant rather than a shortcut,
but it remains a deviation from v1. The target construction *is* faithful: both use a hard argmax.
*(D-002.)*

**15.2 `layer_scale_init_value` follows timm's split, not HF's uniform value (X-2).** The primary
sources disagree about the same official checkpoints: HF's `config.json` says `0.1` for base and
large, timm's `beit.py` says `0.1` for base and `1e-5` for large. This package adopts timm's split,
so `SCALE_CONFIGS['large']` does not match HF's `config.json` field-for-field. Layer-scale init is
training-time-only, so neither is "wrong"; recording it stops the pick being re-litigated.
*(D-003. Pinned by `test_layer_scale_init_value_split_is_timms`.)*

**15.3 `tiny` and `small` are repo inventions (X-3).** No BEiT of either size exists in the paper,
HF or timm; they exist so the test suite and CPU smoke runs are cheap, reproduce nothing, and should
never be cited as "BEiT-tiny". *(D-003.)*

**15.4 The trunk's final LayerNorm follows the `use_mean_pooling` fork (D-007).**
`BeitModel.final_norm` is created and applied **only when `use_mean_pooling is False`**, mirroring
the reference, where `BeitModel.layernorm` is `nn.Identity()` at `use_mean_pooling=True` and
`BeitPooler` owns the only LayerNorm on that path. At the default there is no trunk norm, the
classifier means over the patch tokens with cls excluded, and `head_norm` is present; at `False` the
trunk norms, the classifier reads the cls token, and `head_norm` is absent. Exactly one normalization
on every path. **Do not "clean this up" by always applying a final norm in the trunk**: at the
default that inserts a normalization the reference does not have, in front of *both* heads — no
error, no shape change, a plausible loss curve. `final_norm` is therefore absent from every
default-config checkpoint. *(D-007. Pinned by `test_final_norm_follows_the_mean_pooling_fork`.)*

**15.5 A single-resolution pipeline (X-4).** BEiT v1 feeds 224 to the encoder and 112 to the dVAE
tokenizer with two different interpolation filters, purely so a fixed /8 dVAE lands on 14x14.
`src/train/beit/` trains its own tokenizer, so one image tensor feeds both the encoder
(`patch_size=16`) and the tokenizer (`downsample_factor=16`) onto the same 14x14 grid, at the cost of
the tokenizer seeing 4x the pixels per sample. *(D-004.)*

**15.6 Per-layer relative-position tables during *pre-training* (X-5).** BEiT v1 shares **one** bias
table across every self-attention layer during pre-training and forks to per-layer tables only at
fine-tuning. **This package uses per-layer tables everywhere, including stage 1 — which *is*
pre-training.** `use_shared_relative_position_bias=True` is therefore refused rather than silently
ignored: `_validate_config` raises a `ValueError` naming the flag and saying why (a shared table
would have to be threaded through `TransformerLayer.call()` as a per-forward tensor, a shared-block
signature change that is out of scope). The cost is parameters only: at `base` / 14x14 each table is
`(2·14−1)·(2·14−1) + 3 = 732` rows x 12 heads = 8,784 floats, so v1's single shared table costs 8,784
and this package's 12 per-layer tables cost 105,408 — +96,624 parameters, about +0.11% of the `base`
backbone. Re-derive it from `[w for w in model.weights if "relative_position_bias_table" in w.path]`,
which returns 12 tensors of shape `(732, 12)`. Negligible in memory, but a real capacity difference
in the pre-training objective. This is the *sharing topology*, not the table: its shape and its three
cls slots are faithful. *(D-013. Pinned by
`test_shared_relative_position_bias_is_refused_not_ignored`.)*

**15.7 What is *not* a deviation.** Faithful: the shape of the `(2Wh−1)(2Ww−1)+3` table with its
three cls slots; the structurally-absent K bias; `layer_norm_eps=1e-12`; mask-token substitution with
full-sequence processing; pre-norm blocks with LayerScale on both branches and a linear
stochastic-depth ramp; GELU MLP at 4x width; hard-argmax MIM targets; the block-wise masking
algorithm including its under-fill; `use_absolute_position_embeddings=False`; and mean pooling with
cls excluded.

Authoring conventions: [`models/CLAUDE.md`](../../CLAUDE.md). Mandatory guide:
`research/2026_keras_custom_models_instructions_v2.md`. BEiT v2 (arXiv:2208.06366) keeps this backbone
with a VQ-KD target and BEiT-3 (arXiv:2208.10442) generalizes it into a Multiway Transformer; neither
is implemented here.

## 16. Citation

```bibtex
@inproceedings{bao2022beit,
  title={{BEiT}: {BERT} Pre-Training of Image Transformers},
  author={Bao, Hangbo and Dong, Li and Piao, Songhao and Wei, Furu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}, eprint={2106.08254}, url={https://arxiv.org/abs/2106.08254}
}
```
