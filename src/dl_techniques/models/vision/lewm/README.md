# LeVJEPA - Joint-Embedding Video Pretraining

A Keras 3 port of the **LeVJEPA** Vision Transformer encoder: one ViT-style backbone that
takes either a video clip (tubelet patches, `PatchEmbed3D`) or a still image (2D patches,
`PatchEmbedding2D`), carries position information as *either* a frozen 3D sin-cos table *or*
a 3-axis rotary embedding rotating `q`/`k` inside every block, and optionally gates attention
with a **block-causal** mask that is bidirectional within a frame and causal across frames.

`LeVJEPAEncoder` and `LeVJEPATrainingModel` are **subclassed** `keras.Model`s, not Functional
graphs. That one fact explains most of the surprises below: no `.input`, no `.output`, no
`output_names`, and `count_params()` raises until the model is built.

> **`create_levjepa()` returns a bare encoder, not a trained or trainable-as-is model.** It
> emits `(batch, 1 + num_patches_kept, embed_dim)` token sequences with the CLS token at
> index 0 and nothing else. Self-supervised pretraining lives in `LeVJEPATrainingModel`
> (§ 10), which is **not** exported from the package `__init__` (§ 16).

## 1. Overview: What LeVJEPA Is and Why It Matters

Pixel-reconstruction pretraining (MAE-style) spends capacity predicting texture the
downstream task never uses. A **joint-embedding predictive architecture** moves the
prediction target into representation space instead: encode several views of the same clip,
project them, and ask the views to agree. Nothing predicts pixels.

Agreement alone has a trivial solution, which is for the encoder to emit a constant. LeVJEPA
blocks it with **SIGReg**, a distributional regularizer over the projected embeddings
(`SIGRegLayer`, `knots=17`, `num_proj=1024`, `normalize_by_n=True`), weighted by
`sigreg_weight` (paper's lambda, default `0.02`). The total training objective is

```
loss = mean((global_emb - all_view_embs)**2)  +  sigreg_weight * sigreg(embeddings)
        \______________ pred_loss ___________/    \____ collapse prevention ____/
```

Both terms are registered with `self.add_loss(...)` inside `call()`, so the model trains
under a stock `model.compile(loss=None)` and `model.fit()` with **no custom `train_step`**.

The second idea is **time as a causal axis**. With `attn_mode="block_causal"` a token may
attend to every token in its own frame and to every token in earlier frames, never to a later
one. That makes the encoder autoregressive over time while staying fully bidirectional within
a frame, which is what a video model actually wants.

## 2. The Two Problems This Architecture Solves

**Collapse.** Any objective that rewards view agreement is minimized by a constant encoder.
Contrastive methods pay for large negative batches to avoid it; LeVJEPA instead adds a
uniformity statistic computed on the projector output. The SIGReg term is what makes
`pred_loss` safe to minimize.

**Temporal leakage.** Full attention over a flattened `(T', H', W')` token grid lets frame 0
read frame 7. For any pretext task that involves predicting forward in time, that is the
answer leaking into the question. `build_block_causal_mask` removes exactly those edges and
nothing else: the within-frame block stays dense.

## 3. How It Works: Data Flow

```
  video (B, T, H, W, C)                      image (B, H, W, C)
        │ num_frames > 1                            │ num_frames == 1
        ▼                                           ▼
   PatchEmbed3D  (tubelet_size x p x p)     PatchEmbedding2D  (p x p)
        └──────────────────┬────────────────────────┘
                           ▼
                 x (B, N, D),  N = T' * H' * W',  T' = num_frames // tubelet_size
                           ▼
            + frozen 3D sincos table        (use_rope=False only)
                           ▼
            random_token_drop               (training and token_dropout_rate > 0)
                           ▼
                 x (B, N_kept, D) and token_ids (B, N_kept)
                           ▼
            prepend cls_token (+ its own pos row)
                           ▼
                 (B, 1 + N_kept, D)
                           ▼
            build_block_causal_mask         (attn_mode="block_causal" only)
                           ▼
            LeVJEPABlock x depth            (t/h/w grid, token_ids, attn_mask)
                           ▼
            LayerNormalization(eps=1e-6)
                           ▼
                 (B, 1 + N_kept, D),  CLS at index 0
```

`T'` is the number of *tubelets*, not frames: `num_frames=16, tubelet_size=2` gives `T' = 8`,
and `num_frames` must divide by `tubelet_size` or `__init__` raises.

## 4. Architecture Deep Dive

### 4.1 The block

`LeVJEPABlock` is pre-norm with **plain residual addition on both branches**:

```
  x ─► LayerNorm(1e-6) ─► self-attention ─► (+) ─► LayerNorm(1e-6) ─► MLP ─► (+) ─► x'
  └────────────────────────────────────────┘  └──────────────────────────────┘
```

There is **no `LayerScale`**. The reference block has none, and an earlier draft of this port
that added one was corrected (`decisions.md` D-011). If you are diffing against a timm-style
ViT block, that is the difference you will find.

Attention is written out inside the block rather than delegated to
`layers/attention/multi_head_attention.py`, for one concrete reason: `VideoRoPE3D` has to
rotate `q` and `k` *after* the head split and *before* the softmax, and the shared attention
layer exposes no hook there.

```
  y (B, N, D) ─► Dense(3D, use_bias=qkv_bias) ─► reshape (B,N,3,H,d) ─► transpose (3,B,H,N,d)
                                                         │
                          q, k, v each (B, H, N, d) ◄────┘
                                                         ▼
                      VideoRoPE3D rotates q, k, skipping the first num_prefix_tokens
                                                         ▼
                      logits = q k^T * scale ─► apply_attention_mask ─► softmax ─► @ v
                                                         ▼
                      merge heads ─► Dense(D) ─► attention output (B, N, D)
```

Two details that matter when porting weights or debugging numerics:

- **The softmax runs in float32 and casts back** to `v.dtype`, so a `mixed_float16` block
  keeps a stable normalization.
- **The mask is a keep predicate.** `attn_mask[..., i, j] = True` means query `i` *may*
  attend to key `j`. The layer infers no polarity of its own, and passing an additive
  `-inf` mask will not work.

### 4.2 Depth rescaling is an initializer, not a post-build division

The reference divides the `proj` and `fc2` kernels by `sqrt(2 * layer_id)` after
construction. This port folds that into the initializer instead:

```
  kernel      TruncatedNormal stddev
  ──────      ──────────────────────────────
  qkv, fc1    init_std
  proj, fc2   init_std / sqrt(2 * layer_id)
              init_std, when layer_id is None
```

`LeVJEPAEncoder` passes `layer_id=i + 1` (1-indexed), so `block_0` gets
`0.02 / sqrt(2) = 0.01414...`. A post-build `.assign()` would be **discarded by Keras 3's
`StatelessScope`** (`decisions.md` D-012). The same rule is why the sincos table reaches its
weight through a `Constant` initializer rather than `add_weight(zeros)` plus `.assign()`.
Constructing a block directly with `layer_id=None` (the default) disables rescaling entirely.

### 4.3 Position: `use_rope` picks both halves

| `use_rope` | `pos_embed` weight | Inside blocks |
|:---|:---|:---|
| `False` (default) | frozen `(1, 1 + num_patches, D)`, non-trainable | no rotation |
| `True` | `None`, not created at all | `VideoRoPE3D` on `q` and `k` |

**There is deliberately no separate `pos_embed=` argument** to conflict with `use_rope`
(`decisions.md` D-013). The table is sliced `[:, :1, :]` onto the CLS token and `[:, 1:, :]`
onto the patch tokens, which is why it is stored with a leading batch axis.

There is **no positional-embedding interpolation**. The table is sized once, for the
constructed `input_shape` and `num_frames`, and an input whose rank does not match
`is_video` raises in `build()`.

> **The sincos table is built from `h_patches` alone** (`grid_size=self.h_patches`, plus
> `grid_depth=self.t_patches` on the video path), so it assumes a **square** patch grid. With
> `use_rope=False`, keep `input_shape[0] == input_shape[1]`. Non-square inputs belong on
> `use_rope=True`.

### 4.4 The block-causal mask

`build_block_causal_mask` returns a boolean `(B, 1, N', N')` tensor,
`N' = num_patches + num_prefix_tokens`. Patch-to-patch attention is
`frame_ids[query] >= frame_ids[key]`, where `frame_ids = token_ids // tokens_per_frame`.

```
  query \ key      CLS      frame 0    frame 1    frame 2
  CLS              True     True       True       True      <- all-seeing
  frame 0          False    True       False      False
  frame 1          False    True       True       False
  frame 2          False    True       True       True
```

The CLS row is all-`True` and the CLS **column** is `False` for every patch row: a patch
query never attends to CLS as a key. `num_prefix_tokens=0` skips the carve-out and returns
the bare patch grid.

**This is the single most common way to get a LeVJEPA that is quietly doing nothing
temporal.** With `num_frames=1` (image mode) `t_patches == 1`, every token shares frame 0,
and `attn_mode="block_causal"` degenerates to full attention plus the CLS-column carve-out.
It does not error. Block-causal attention is a **video-mode** feature.

### 4.5 Token dropping, and why the mask needs `token_ids`

`random_token_drop(x, rate, training=...)` draws uniform noise over the `N` sequence
positions, keeps the `keep_len = max(1, round(N * (1 - rate)))` lowest-noise positions via an
argsort permutation-sample, and gathers `x` down. It returns `(dropped_x, token_ids)`.

`token_ids` holds the **true pre-drop grid index** of each surviving position. That is the
whole point: after dropping, sequence position 3 is no longer grid position 3, so frame
membership cannot be recovered from the sequence's own ordering. Feed `token_ids` straight
into `build_block_causal_mask(token_ids=...)` (and to `VideoRoPE3D`, which the encoder also
does) or mask and rotation silently address the wrong frames.

When `training` is falsy or `rate <= 0` the function short-circuits: `dropped_x is x`, no
argsort or gather runs at all, and `token_ids` is `None`. Neither this function nor
`build_block_causal_mask` carries `@register_dl_technique`, because plain functions have no
config to round-trip.

### 4.6 The projector

`LeVJEPAProjector` is `Dense(hidden_dim) -> BatchNormalization(axis=-1) -> GELU ->
Dense(output_dim or input_dim)`. The input width is inferred in `build()` rather than passed
as a constructor argument, and `fc2` is therefore created in `build()` too.

The PyTorch reference flattens to `(-1, input_dim)` before `BatchNorm1d` and reshapes back,
because `BatchNorm1d` only accepts rank-2 input. Keras' `BatchNormalization(axis=-1)` has no
such restriction. This port applies it directly with **no reshape-around**, measured
equivalent to the reference path at max abs diff `0.0` (`decisions.md` D-014).

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.vision.levjepa import create_levjepa

# Image mode, frozen sincos positions, full attention.
encoder = create_levjepa("vit_tiny", input_shape=(64, 64, 3))

# `count_params()` needs the model BUILT -- subclassed model, no weights exist
# until a shape is known.
encoder.build((None, 64, 64, 3))

tokens = encoder(np.random.rand(2, 64, 64, 3).astype("float32"), training=False)
print(tokens.shape)          # (2, 17, 192)   -> 1 CLS + (64/16)**2 = 16 patches

cls = tokens[:, 0, :]        # (2, 192) -- the only output consumed downstream
patches = tokens[:, 1:, :]   # (2, 16, 192)
```

Video mode, rotary positions, block-causal attention:

```python
encoder = create_levjepa(
    "vit_tiny",
    input_shape=(32, 32, 3),
    num_frames=4,
    tubelet_size=2,
    use_rope=True,
    attn_mode="block_causal",
)
x = np.random.rand(2, 4, 32, 32, 3).astype("float32")
print(encoder(x, training=False).shape)   # (2, 9, 192)
# T' = 4 // 2 = 2 tubelets, H' = W' = 2  ->  N = 8 patches, + 1 CLS
```

Note the **rank**: video mode wants `(B, T, H, W, C)` and image mode wants `(B, H, W, C)`.
Passing the wrong rank raises in `build()` with an explicit message rather than broadcasting
into something plausible.

## 6. Model Variants

`SCALE_CONFIGS` holds exactly seven keys, ported from the reference's `vit_tiny` through
`vit_gigantic` factory functions. `MODEL_VARIANTS` is a thin `{name: {"scale": name}}`
wrapper over it, kept explicit so `MODEL_VARIANTS.keys()` can be introspected the same way
`ViT.MODEL_VARIANTS` supports.

| Variant | `embed_dim` | `depth` | `num_heads` | `head_dim` | `mlp_ratio` | `patch_size` |
|:---|---:|---:|---:|---:|---:|---:|
| `vit_tiny` | 192 | 12 | 3 | 64 | 4.0 | 16 |
| `vit_small` | 384 | 12 | 6 | 64 | 4.0 | 16 |
| `vit_base` | 768 | 12 | 12 | 64 | 4.0 | 16 |
| `vit_large` | 1024 | 24 | 16 | 64 | 4.0 | 16 |
| `vit_huge` | 1280 | 32 | 16 | 80 | 4.0 | 16 |
| `vit_giant` | 1408 | 40 | 16 | 88 | 48/11 | 16 |
| `vit_gigantic` | 1664 | 48 | 16 | 104 | 64/13 | **14** |

`vit_gigantic` is the only variant that changes `patch_size`, which changes the token count
for the same input: at 224x224 it produces 256 tokens per frame against 196 for the
patch-16 variants. `qkv_bias=True` and `LayerNorm(eps=1e-6)` are constant across every scale
and deliberately absent from the table, since they are the block's and encoder's own
defaults.

> **No accuracy numbers appear anywhere in this README, and that is deliberate.** No
> pretrained weights exist for this architecture in this repository (§ 9), so nothing here
> has ever been evaluated on anything. § 17 gives parameter counts, which are architectural
> arithmetic, not measurements on a checkpoint.

### Constructor arguments

`LeVJEPAEncoder(...)`, `from_variant(variant, ...)` and `create_levjepa(variant, ...)` all
accept these. `from_variant` and `create_levjepa` take the five shape/scale fields from
`SCALE_CONFIGS` and forward everything else verbatim.

| Argument | Default | Meaning |
|:---|:---|:---|
| `input_shape` | `(224, 224, 3)` | `(height, width, channels)`; both spatial dims must divide by `patch_size` |
| `num_frames` | `1` | `1` is the image path; `> 1` is the video path and must divide by `tubelet_size` |
| `patch_size` | `16` | spatial patch size (set by the variant) |
| `tubelet_size` | `2` | temporal patch size, read only when `num_frames > 1` |
| `embed_dim` | `192` | token width; must divide by `num_heads` |
| `depth` | `12` | number of `LeVJEPABlock`s |
| `num_heads` | `3` | attention heads per block |
| `mlp_ratio` | `4.0` | MLP hidden width is `int(embed_dim * mlp_ratio)` |
| `qkv_bias` | `True` | bias on the fused QKV projection |
| `use_rope` | `False` | `True` swaps the sincos table for `VideoRoPE3D`; § 4.3 |
| `rope_theta` | `10000.0` | rotary base frequency, read only when `use_rope=True` |
| `attn_mode` | `'full'` | `'full'` or `'block_causal'`; § 4.4 |
| `token_dropout_rate` | `0.0` | train-time patch drop fraction, in `[0, 1)`; § 4.5 |
| `dropout_rate` | `0.0` | post-projection and in-MLP dropout, forwarded to every block |
| `attention_dropout_rate` | `0.0` | post-softmax attention-weight dropout |
| `init_std` | `0.02` | base truncated-normal std for every kernel and the CLS token |
| `uniform_power` | `False` | forwarded to the 3D sincos builder; the image path ignores it |

Every one of these is validated in `__init__` and raises `ValueError` on a bad value, so a
typo surfaces at construction rather than three epochs in. `num_prefix_tokens` is **not** a
constructor argument on the encoder: it is fixed at `1` (the CLS token) and only exposed on
`LeVJEPABlock` and `build_block_causal_mask` directly.

## 7. Token Dropping as an Augmentation

`token_dropout_rate` is the cheapest lever here, and it is train-time only:

```python
encoder = create_levjepa(
    "vit_base", input_shape=(224, 224, 3), num_frames=16, tubelet_size=2,
    attn_mode="block_causal", token_dropout_rate=0.5,
)
# T' = 8, H' = W' = 14  ->  N = 1568 patches
# training=True : keep_len = round(1568 * 0.5) = 784  ->  (B, 785, D)
# training=False: no drop at all                      ->  (B, 1569, D)
```

`compute_output_shape` reports the **upper bound** (`1 + num_patches`), because the kept-token
count is a runtime quantity. Anything downstream that hard-codes a sequence length will
disagree with the training-time tensor; consume the CLS token or pool, do not index a fixed
position.

## 8. Usage Examples

### Example 1: Linear probe on the CLS token

Wrap the backbone in a Functional model starting from a `keras.Input` you own; a subclassed
model has no `.input` to re-wire from.

```python
import keras
import numpy as np
from dl_techniques.models.vision.levjepa import create_levjepa

base = create_levjepa("vit_tiny", input_shape=(64, 64, 3))
base.trainable = False                       # frozen backbone = linear probe

inputs = keras.Input(shape=(64, 64, 3))
tokens = base(inputs)                        # (B, 17, 192)
cls = keras.layers.Lambda(lambda t: t[:, 0, :])(tokens)
outputs = keras.layers.Dense(10, name="probe")(cls)   # LOGITS, no activation
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
x = np.random.rand(8, 64, 64, 3).astype("float32")
y = np.random.randint(0, 10, (8,))
model.fit(x, y, epochs=1, verbose=0)
```

The wrapper holds the whole backbone as **one** layer, so per-layer freezing goes through
`base.blocks` and `base.layers`, never `model.layers`. A `trainable` change has no effect
until you `compile()` again.

### Example 2: Mean-pooled patch tokens instead of CLS

```python
cls_and_patches = base(inputs)
pooled = keras.layers.Lambda(lambda t: keras.ops.mean(t[:, 1:, :], axis=1))(cls_and_patches)
```

Valid at inference. Under `token_dropout_rate > 0` the pool is over a random subset, which is
a different (and noisier) quantity than the full-grid mean.

### Example 3: A block on its own

```python
import keras
from dl_techniques.models.vision.levjepa import LeVJEPABlock

block = LeVJEPABlock(dim=192, num_heads=3, use_rope=True, layer_id=1)
x = keras.random.normal((2, 1 + 18, 192))
print(block(x, num_frames=2, height_patches=3, width_patches=3).shape)   # (2, 19, 192)
```

`use_rope=True` **requires** `height_patches` and `width_patches` at call time and raises
without them. `num_frames` here is the tubelet count `T'`, matching what the encoder passes.

### Example 4: A hand-built mask

```python
from dl_techniques.models.vision.levjepa import build_block_causal_mask

mask = build_block_causal_mask(num_frames=3, tokens_per_frame=4, num_prefix_tokens=1)
print(mask.shape)        # (1, 1, 13, 13)   -- broadcastable over any batch
```

The batch axis is `1` unless you pass `batch_size=` or `token_ids=`; either one sets it.
`token_ids` wins: when it is given, its own leading dimension is the batch size and
`batch_size` is ignored.

## 9. Pretrained Weights

> **There are no LeVJEPA weights to download, from this repository or anywhere else in a
> format this package loads.** There is no `pretrained=` argument on `from_variant` or
> `create_levjepa` at all, which is at least honest about it.

The working path is a checkpoint you produced yourself:

```python
import keras

model = keras.models.load_model("levjepa_pretrain.keras")   # LeVJEPATrainingModel
encoder = model.encoder                                     # the reusable half
```

Every class here registers under a package-qualified key
(`dl_techniques.models.levjepa.encoder`, `.blocks`, `.projector`, `.training`), so a
`.keras` file loads with no `custom_objects`.

## 10. Self-Supervised Pretraining

`LeVJEPATrainingModel` is the multiview wrapper. It owns **one shared encoder**, run on the
global view and on every local view, one `LeVJEPAProjector`, and one `SIGRegLayer`.

```
  {"global_frame": (B,T,H,W,C), "local_frames": (B,V,T,H,W,C)}
        │                                    │
   encoder(global)                    reshape to (B*V,T,H,W,C), encoder(...)
   -> (B, 1+N, D), take CLS           -> (B*V, 1+N, D), take CLS, reshape (B,V,D)
        └──────────── concat(axis=1) ────────┘
                          (B, 1+V, D)
                              ▼
                          projector
                              ▼
             embeddings (B, 1+V, D)
              /                        \
   global_emb = embeddings[:, :1]     sigreg(transpose -> (1+V, B, D))
              \                        /
   pred_loss = mean((global_emb - embeddings)**2)     sigreg_loss
              \                        /
        add_loss(pred_loss);  add_loss(sigreg_weight * sigreg_loss)
```

```python
import keras
import numpy as np
from dl_techniques.models.vision.levjepa.model import create_levjepa
from dl_techniques.models.vision.levjepa.training import LeVJEPATrainingModel

encoder = create_levjepa("vit_tiny", input_shape=(32, 32, 3), num_frames=4,
                         attn_mode="block_causal")
model = LeVJEPATrainingModel(encoder=encoder, sigreg_num_proj=64, sigreg_weight=0.02)

# `loss=None`: both terms come from add_loss inside call(). Passing a loss here
# adds a second objective on top, it does not replace them.
model.compile(optimizer=keras.optimizers.AdamW(1e-4))

batch = {
    "global_frame": np.random.rand(2, 4, 32, 32, 3).astype("float32"),
    "local_frames": np.random.rand(2, 2, 4, 32, 32, 3).astype("float32"),
}
model.fit(batch, epochs=1, verbose=0)
print(model(batch).shape)        # (2, 3, 192) = (B, 1 + V, D)
```

Four things worth knowing before a long run:

- **The encoder must be in video mode.** `num_frames > 1`, or the constructor raises. The
  multiview forward has no image-mode meaning.
- **The global-vs-itself term is kept.** `global_emb` broadcasts against all `1 + V` columns,
  so column 0 contributes exactly `0` and stays in the mean. This is faithful to the
  reference, which does not slice it out, and it scales `pred_loss` by `V / (1 + V)`
  relative to a version that does.
- **Input is a dict, not a tuple.** Both `"global_frame"` and `"local_frames"` keys are
  required; `build()` raises with an explicit message otherwise.
- **`pred_loss` and `sigreg_loss` are exposed as metrics** via the `metrics` property, so a
  `CSVLogger` carries them next to `loss`. They are updated inside `call()`, which means
  `evaluate()` and `predict()` move them too.

Inputs are cast to `float32` and nothing else. There is **no ImageNet mean/std
normalization**: this port's data sources already emit bounded `float32`
(`decisions.md` D-017). Add it at the dataset boundary if a future domain needs it.

### EMA shadow weights

`update_ema_shadow(decay)` maintains one non-trainable shadow variable per encoder weight, a
duck-typed contract for `dl_techniques.callbacks.ema_shadow_callback.EMAShadowCallback`
(which lives in `callbacks/`, not this package, per `decisions.md` D-006, and reuses
`teacher_ema.py`'s decay schedules).

```python
from dl_techniques.callbacks.ema_shadow_callback import EMAShadowCallback

model.fit(ds, epochs=100, callbacks=[EMAShadowCallback(...)])
```

The shadow variables are created in `build()` (Keras 3's `StatelessScope` forbids adding
state after `built=True`) but **zero-initialized**, because `ops.convert_to_numpy` raises
inside `build()`'s tracing context. They are copy-seeded from the live weights on the
**first** `update_ema_shadow` call, which is a genuinely eager context (`decisions.md`
D-020). A checkpoint saved before that first call therefore carries zeros in the shadow
slots, which is expected and harmless.

### Multi-crop data

`dl_techniques/datasets/vision/multi_crop_video.py` produces the `{"global_frame",
"local_frames"}` dict as a `tf.data` transform: 1 global plus N local views, with the **same
crop box applied across every frame of one clip** via a stateless-RNG replay
(`decisions.md` D-019). It reuses `multi_crop.py`'s crop and augmentation primitives rather
than re-deriving them (D-005). A per-frame independent crop would destroy the temporal
correspondence the objective depends on.

### The training CLI

`src/train/levjepa/train_levjepa.py` wires all of the above:
`--dataset {synthetic_drone,bdd100k}`, a `--smoke` preset, `optimizer_builder()` with
`WarmupSchedule(primary_schedule=FlatSchedule(...))`, and `EMAShadowCallback` as a callback.

## 11. Fine-Tuning Strategies

**Route 1 (recommended): staged unfreezing.** Blocks are named `block_0` .. `block_{depth-1}`
and are reachable as the `encoder.blocks` list, so grouping by depth is a slice. Freeze the
patch embedding and the early blocks, train, then unfreeze and re-`compile()` at a lower
learning rate.

```python
encoder.trainable = True
encoder.patch_embed.trainable = False
for block in encoder.blocks[:6]:
    block.trainable = False
model.compile(optimizer=keras.optimizers.AdamW(1e-5), loss=...)   # required
```

**Route 2: two optimizers over disjoint variable sets** in a custom training loop. This
repository's convention is to avoid a custom `train_step`; prefer route 1.

`pos_embed` is non-trainable by construction, and `encoder.trainable = True` does not change
that. The frozen sincos table stays frozen.

## 12. Advanced Notes

- **`out_layers` is not ported.** The reference can return intermediate features; this port
  returns only the final normalized sequence, because only the CLS token is ever consumed
  downstream (`decisions.md` D-011..D-013). Reading intermediates means looping
  `encoder.blocks` yourself and reproducing the call-time arguments.
- **`num_prefix_tokens`** is exposed on `LeVJEPABlock` and `build_block_causal_mask` but
  fixed at `1` by the encoder. Setting `0` on the mask builder returns the bare causal patch
  grid, which is what an `attn_mode="full"` bypass would want if it did not simply skip the
  call.
- **Register custom layers** with `@register_dl_technique("dl_techniques.<module.path>")`, a
  package-qualified key. Never a bare `@keras.saving.register_keras_serializable()`: its
  `Custom>ClassName` key carries no module path, so two same-named classes claim one slot.
- **House convention**: `import keras` and qualify at the call site
  (`keras.ops.<fn>(...)`), never `from keras import ops`. All four sites that violated this
  were fixed in the step-6.2 completion round; the check is
  `grep -rn "from keras import ops" src/dl_techniques/models/vision/levjepa/`.

## 13. Performance Optimization

| Lever | Effect |
|:---|:---|
| `token_dropout_rate=0.5` | halves the sequence at train time; attention cost is quadratic in it |
| `keras.mixed_precision.set_global_policy("mixed_float16")` | roughly halves activation memory; the softmax already upcasts |
| `model.compile(..., jit_compile=True)` | XLA fusion; measure, not always faster |
| larger `tubelet_size` | divides `T'`, so divides the token count linearly |
| `use_rope=True` | drops the `(1 + N) x D` frozen table; matters most at long sequences |

Two cost traps specific to this architecture. First, **`attn_mode="block_causal"` materializes
a boolean `(B, 1, N, N)` mask per forward pass**: at `N = 1569` that is about 2.5M bool
entries per batch element, and it is built inside `call()`, not cached. Second, **the
training wrapper runs the encoder `1 + V` times per step** (once on the global view, once on
a `B*V`-batched flatten of the locals), so `V` multiplies the step cost directly.

The input pipeline is usually the bottleneck before the model: `tf.data` with `.cache()`,
`.prefetch(AUTOTUNE)` and a parallel `map` first.

## 14. Serialization

```python
import keras
import numpy as np
from dl_techniques.models.vision.levjepa import create_levjepa

model = create_levjepa("vit_tiny", input_shape=(64, 64, 3))
model.build((None, 64, 64, 3))
model.save("levjepa_tiny.keras")

restored = keras.models.load_model("levjepa_tiny.keras")
x = np.random.rand(2, 64, 64, 3).astype("float32")
assert np.allclose(np.asarray(model(x, training=False)),
                   np.asarray(restored(x, training=False)), atol=1e-6)
```

`LeVJEPAEncoder.get_config()` carries every constructor argument, `input_shape` included (as
`input_shape_config`, to avoid colliding with the `Model` property).

`LeVJEPATrainingModel` round-trips through **nested encoder serialization**: its config holds
`keras.saving.serialize_keras_object(self.encoder)` rather than duplicated constructor
arguments, and `from_config` deserializes the encoder first and passes the instance in
(`decisions.md` D-018). `projector` and `sigreg` are plain `Layer`s that Keras already
reconstructs through build-then-load-weights.

Compare `training=False` outputs. With `token_dropout_rate > 0` a `training=True` comparison
compares two different random subsets and will fail for reasons that have nothing to do with
serialization.

## 15. Testing & Validation

`pytest tests/test_models/test_levjepa/ -q`. The training CLI was verified with a **real
end-to-end smoke run**, not an import check. When adding tests, the useful invariants are:
mask polarity (`True` means attend), `token_ids` consistency between `random_token_drop` and
`build_block_causal_mask`, serialization round-trip at `training=False`, and the
`use_rope` / `pos_embed` exclusivity (exactly one of the two exists on any instance).

## 16. Troubleshooting & FAQs

- **`cannot import name 'LeVJEPATrainingModel'` from the package.** The curated `__init__`
  exports the encoder, block, projector, masking functions and factories, **not** the
  training wrapper. Import it from
  `dl_techniques.models.vision.levjepa.training` directly.
- **`LeVJEPATrainingModel's multiview forward requires a video-mode encoder`.** You passed an
  encoder built with the default `num_frames=1`. Pass `num_frames=4` (or more, divisible by
  `tubelet_size`) to `create_levjepa`.
- **`Expected 5D input ... got 4D input`.** Rank follows `num_frames`: `> 1` wants
  `(B, T, H, W, C)`, `== 1` wants `(B, H, W, C)`. There is no auto-squeeze.
- **`You tried to call count_params on layer 'levjepa_encoder', but the layer isn't built`.**
  Subclassed model: call `build((None, ...))` or run one forward pass first. Same cause for
  `summary()`.
- **Block-causal mode changes nothing.** You are in image mode, or `num_frames == tubelet_size`
  so `T' == 1`. See § 4.4.
- **`LeVJEPABlock(use_rope=True) requires height_patches and width_patches at call time`.**
  Calling a block directly, not through the encoder. Pass the patch-grid dimensions.
- **Output sequence length changes between `fit` and `predict`.** `token_dropout_rate > 0` is
  train-time only, by design. `compute_output_shape` reports the no-drop upper bound.
- **`num_frames (5) must be divisible by tubelet_size (2)`.** Exactly what it says; the video
  path tubelets the time axis.
- **`input_shape spatial dims (...) must be divisible by patch_size`.** Note that
  `vit_gigantic` uses `patch_size=14`, not 16.
- **Loss is `0.0` and stays there.** With `sigreg_weight=0.0` the objective is pure agreement
  and the encoder collapses to a constant, which is its global minimum. That term is not
  optional.
- **Freezing had no effect.** You must `compile()` again after changing `trainable`.
- **The EMA shadow weights are all zero in a checkpoint.** Expected if `update_ema_shadow`
  never ran; they are seeded on the first call, not at `build()`. See § 10.

## 17. Technical Details

### Parameter counts

**Derived from the code, not measured on a checkpoint and not quoted from a paper.** Every
`Dense` shape in `LeVJEPABlock` is fully determined by `dim` and `mlp_ratio`, so the block
stack is exact arithmetic:

```
  H = int(embed_dim * mlp_ratio)

  per_block = 4*D**2 + 2*D*H + 9*D + H
              \_____/   \___/   \_____/
              qkv+proj   MLP     norms and biases

  block_stack = depth * per_block
```

| Variant | `H` | Params per block | Block stack (`depth x`) |
|:---|---:|---:|---:|
| `vit_tiny` | 768 | 444,864 | 5,338,368 |
| `vit_small` | 1,536 | 1,774,464 | 21,293,568 |
| `vit_base` | 3,072 | 7,087,872 | 85,054,464 |
| `vit_large` | 4,096 | 12,596,224 | 302,309,376 |
| `vit_huge` | 5,120 | 19,677,440 | 629,678,080 |
| `vit_giant` | 6,144 | 25,250,176 | 1,010,007,040 |
| `vit_gigantic` | 8,192 | 38,361,728 | 1,841,362,944 |

On top of the stack: the patch embedding projection, the `cls_token` (`D` trainable), the
final `LayerNormalization` (`2D`), and, when `use_rope=False`, the frozen
`(1 + num_patches) * D` sincos table, which is **non-trainable** and therefore shows up in
`count_params()` but not in `trainable_weights`. Those four depend on `input_shape`,
`num_frames`, `tubelet_size` and `use_rope`, so they are not tabulated here.

Re-derive the real numbers for your configuration:

```python
import numpy as np
from dl_techniques.models.vision.levjepa import SCALE_CONFIGS, create_levjepa

for variant in SCALE_CONFIGS:
    m = create_levjepa(variant, input_shape=(224, 224, 3))
    m.build((None, 224, 224, 3))
    tr = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    nt = sum(int(np.prod(w.shape)) for w in m.non_trainable_weights)
    print(f"{variant}: {tr:,} + {nt:,} = {m.count_params():,}")
```

### Sequence lengths

`N = (num_frames // tubelet_size) * (H // patch_size) * (W // patch_size)`, plus 1 for CLS.
Attention cost is quadratic in that total.

| Configuration | `T'` | tokens/frame | `N` | with CLS |
|:---|---:|---:|---:|---:|
| 224x224 image, patch 16 | 1 | 196 | 196 | 197 |
| 224x224 image, patch 14 (`vit_gigantic`) | 1 | 256 | 256 | 257 |
| 16 frames 224x224, tubelet 2, patch 16 | 8 | 196 | 1,568 | 1,569 |
| 16 frames 224x224, tubelet 4, patch 16 | 4 | 196 | 784 | 785 |
| 4 frames 32x32, tubelet 2, patch 16 | 2 | 4 | 8 | 9 |

### Deliberate scope simplifications (not gaps)

- No multi-output-feature (`out_layers`) branch: only the last layer's normalized sequence is
  returned.
- No dynamic positional-embedding interpolation: the sincos table is sized once, for the
  constructed `input_shape` and `num_frames`.
- `use_rope` is the only position-mode toggle, matching the reference's constructor exactly.
  There is no separate `pos_embed=` argument to conflict with it.
- No `LayerScale`, matching the reference's plain-residual `Block`.
- No channels-first/channels-last `rearrange` pair in the training wrapper: this repo's video
  tensors are already channels-last at every call site.

`decisions.md` D-011 through D-014 and D-017 through D-020 in this plan's directory carry the
reasoning for each.

### Authoring rules

Conventions: [`models/CLAUDE.md`](../../CLAUDE.md). Mandatory guide:
`research/2026_keras_custom_models_instructions_v2.md`.

## 18. References

The LeVJEPA reference itself was ported from a pasted PyTorch transcript
(`module.py::VisionTransformer` / `Block` / `Attention` / `RoPEAttention` / `Projector`,
`main.py::multiview_forward`) with **no public arXiv id available in this port's context**,
so no BibTeX entry is invented for it here. The architectural ancestors the code cites:

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and
          Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for
         Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```