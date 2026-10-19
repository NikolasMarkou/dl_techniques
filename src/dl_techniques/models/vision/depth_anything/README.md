# Depth Anything

Keras 3 reference implementation of the *Depth Anything* monocular depth
estimation architecture (`encoder + DPT head`). Source files:

```
src/dl_techniques/models/vision/depth_anything/
├── __init__.py        # public API: DepthAnything, create_depth_anything, DPTHead, DPTDecoder
├── components.py      # DPTHead (DepthAnything's decoder), DPTDecoder (OmniPoint's)
├── model.py           # DepthAnything keras.Model + create_depth_anything factory
└── teacher_ema.py     # TeacherEMACallback, cosine_ema_schedule, linear_ema_schedule
```

The encoder is the reference's DINOv2 (`encoder_kind='real'`, the default):
`dl_techniques.models.vision.dino.DINOv2VisionTransformer` at ViT-S/B/L/14, pre-norm, LayerScale,
no register tokens, with the official 37 x 37 position table resampled to the input's patch grid
(torch's bicubic kernel at scale `(n + 0.1) / 37`, as the reference's `interpolate_pos_encoding`).
Image sides must be multiples of 14 (the reference trains at 518, the default).
`DepthAnything.from_variant('vit_s', pretrained=True)` (and `create_depth_anything(...,
pretrained=True)`) load the official DINOv2 weights (`facebook/dinov2-{small,base,large}`, a local
download, see "Pretrained encoder" below) into the encoder; the head starts from its
initialization, as in the paper. No trained weights ship with this package: `pretrained=True`
reads the encoder file you download yourself, and without it the call is a hard error
(`FileNotFoundError`, naming the path, before anything is built). What that start buys was
measured, not assumed: from the official DINOv2-small encoder, 3,000 steps on MegaDepth reach a
held-out masked affine-invariant loss of 0.57 to 0.59 on three seeds, against 1.0 for a constant
prediction and 1.045 for the average training layout, with a per-crop Pearson correlation of 0.80
to 0.81 (the layout: 0.42). From a random encoder the same trainer only ties the average layout.
The table and settings are in `src/train/depth_anything/README.md`, "Learning rate". The decoder is `DPTHead`, a port of the reference V1 head that fuses the encoder's
last four layers, and it ends in a ReLU, as the reference heads do (see "DPT head" and "Output
range" below). `train_step` dispatches to a
clean labeled-only path or a semi-supervised path that adds FAL on pooled features **and**
a pseudo-label consistency term on student-vs-teacher depth over `x_unlab` (the paper's
affine-invariant loss per CutMix region, eq. 6-8). On-step EMA
decay is provided by `TeacherEMACallback` with cosine/linear schedules, and
`DepthAnything.from_pretrained_encoder(path)` loads encoder weights from a saved `.keras`
checkpoint and re-syncs the teacher. `StrongAugmentation` supports any number of channels,
applies per-sample brightness/contrast factors, and mixes the depth target by the same
CutMix box as the image (the mixing happens in `train_step`, not in `call`). The legacy
Conv-BN-ReLU encoder is preserved behind `encoder_kind='placeholder'` for back-compat.

---

## Overview

`DepthAnything` is a `keras.Model` subclass that takes an RGB image batch
and produces a single-channel depth map at the same spatial resolution:

```
input  : (B, H, W, 3)   float32 RGB
output : (B, H, W, 1)   float32 relative depth, non-negative (final ReLU)
```

### Output range

The output is the ReLU of the decoder's last conv, so it is non-negative with no upper bound and no
range parameters. This is what the reference does, re-derived from the paper and the official code
(fetched 2026-10-15):

* The Depth Anything paper (arXiv 2401.10891) trains relative depth: depth is turned into disparity
  `d = 1/t`, normalized per map, and supervised with an affine-invariant loss. There is no metric
  range in the model.
* V1 `depth_anything/dpt.py` (LiheYoung/Depth-Anything): the head's `output_conv2` ends
  `Conv2d(32, 1, 1) -> nn.ReLU(True) -> nn.Identity()`, and `DPT_DINOv2.forward` applies `F.relu`
  again after the bilinear resize.
* V2 `depth_anything_v2/dpt.py` (DepthAnything/Depth-Anything-V2): the same ReLU tail and `F.relu`
  after the resize.
* The only range maps are in the metric variants, which this package does not implement: V2's
  `metric_depth/depth_anything_v2/dpt.py` ends in `nn.Sigmoid()` and multiplies by `max_depth`
  (20 indoor, 80 outdoor), and V1's metric models are ZoeDepth with metric bins.

The decoder used to be linear and received `min_depth=0.1`, `max_depth=100.0` and
`use_log_depth=True`, which the DPT decoder stored and never applied; the decoder now refuses them.
A `.keras` archive saved before 2026-10-15 still loads, with every weight restored, and its output
is then the ReLU of what it returned before (measured: max difference 0.0 against the old output
clamped at zero). Nothing marks such an archive, so the change is silent.

The trainer (`src/train/depth_anything/`) supervises the paper's target and loss: per-map normalized
MegaDepth disparity in `[0, 1]` and the masked affine-invariant MAE (`AffineInvariantLoss(masked=True)`).

### DPT head

`DPTHead` (`components.py`) ports `DPTHead` of LiheYoung/Depth-Anything `depth_anything/dpt.py`
(with `FeatureFusionBlock` and `ResidualConvUnit` of `blocks.py`, fetched 2026-10-17) for one
output map with `use_bn=False` and `use_clstoken=False`, the configuration of all three published
checkpoints:

```
ViT layers L-4..L-1, patch tokens, each through the final norm, as (B, g, g, D)
  projects[i]       Conv1x1 D -> C_i
  resize_layers     ConvTranspose 4x4/4 (4g) | ConvTranspose 2x2/2 (2g) | identity (g) | Conv3x3/2, pad 1 (g/2)
  layer{i}_rn       Conv3x3 C_i -> F, no bias
  refinenet4..1     ResidualConvUnit(skip) + path -> ResidualConvUnit -> bilinear (align_corners) -> Conv1x1
  output_conv1      Conv3x3 F -> F/2, bilinear (align_corners) to g * 16
  output_conv2      Conv3x3 -> 32, ReLU, Conv1x1 -> 1, ReLU
```

| Encoder | `dpt_features` (F) | `dpt_out_channels` (C_1..C_4) | Head parameters | Reference module list |
|---------|--------------------|-------------------------------|-----------------|-----------------------|
| `vit_s` | 64  | 48, 96, 192, 384      | 2,654,657  | 2,728,513 (summed from the published `vits14` checkpoint's tensors) |
| `vit_b` | 128 | 96, 192, 384, 768     | 10,595,137 | 10,890,305 (hand sum of `DPTHead`'s module list) |
| `vit_l` | 256 | 256, 512, 1024, 1024  | 29,766,849 | 30,947,009 (hand sum of `DPTHead`'s module list) |

The defaults are the published `config.json` values per encoder size. The difference from the
official count is exactly `refinenet4.resConfUnit1`, which the reference builds and its forward
never calls (refinenet4 has no skip input): this port does not build it. The ResidualConvUnits have
no BatchNorm (`use_bn=False`), so the head computes the same function in training and inference
mode. Torch's `Conv2d(3, stride 2, padding 1)` pads both sides, which Keras' `'same'` does not for
an even input, so `resize_layers_3` is an explicit `ZeroPadding2D(1)` and a `'valid'` conv; Keras'
bilinear resize uses half-pixel centres, so the head resizes with
`resize_bilinear_align_corners`, the reference's `align_corners=True`. The DINOv2 encoder has
patch 14 and the head interpolates to `14 * g`, the input size, as the reference does. The
placeholder Conv encoder (stride 16) has no intermediate layers; its one feature map (projected
to the DINOv2 width of `encoder_type`) feeds all four paths. Guard:
`tests/test_models/test_depth_anything/test_the_dpt_head_is_the_references.py`.

Until 2026-10-17 the decoder was the shared `DepthEstimationHead(decoder_style='dpt')`: one
feature map through `Conv3x3 -> BN -> ReLU` stages and `decoder_dims` (removed, no alias). A
`.keras` archive saved before that does not load into this model.

It is composed of three sub-networks:

| Sub-network        | Type                          | Where built       | Trainable |
|--------------------|-------------------------------|-------------------|-----------|
| `encoder`          | `DINOv2VisionTransformer` or the Conv placeholder | `build()` | yes (the unused iBOT mask token frozen) |
| `decoder`          | `DPTHead`, named `depth_head` | `build()`, only when `include_top=True` | yes |
| `frozen_encoder`   | a second encoder of the same config (DINOv2 rebuilt from its config, the placeholder by `clone_model`), weights copied at build | `build()` if `use_feature_alignment=True` **or** `enable_semi_supervised=True` | no |
| `augmentation`     | `StrongAugmentation` layer    | `build()`         | n/a |

Forward path (`call`):

```
x → encoder (last 4 layers, or the placeholder's map 4 times) → decoder → [resize to H, W] → ReLU → depth
```

`call` does **not** augment. CutMix mixes across batch rows, so a target has to
be mixed by the same rectangle, and only the training path holds one: the
semi-supervised `train_step` calls `_augment_with_targets(x_unlab, [pseudo])`,
which returns the augmented unlabeled image together with the identically-mixed
teacher pseudo-depth. The labeled batch is **not** strongly augmented in either
path: the paper's only labeled-image augmentation is a horizontal flip (arXiv
2401.10891, section 4.1), which the MegaDepth loader applies; CutMix and colour
jitter are its perturbations for unlabeled images (section 3.2). Until
2026-10-17 both paths CutMixed and jittered the labeled batch and its depth
target as well (D-131). `model(x, training=True)` is a plain un-augmented
forward pass.

`train_step` accepts two input shapes:

* `(x, y)` — labeled-only path (default).
* `((x_lab, x_unlab), y_lab)` — semi-supervised path. Active when
  `enable_semi_supervised=True`. Adds a stop-gradient pseudo-label
  consistency term over `x_unlab` against the weight-shared frozen
  teacher (affine-invariant, per CutMix region, eq. 6-8 of the paper), plus a Feature-Alignment Loss term on unlabeled features when
  `use_feature_alignment=True` as well.

---

## Components

### `DepthAnything` (in `model.py`)

A `keras.Model` registered as `dl_techniques.models.depth_anything.model>DepthAnything`
via `@register_dl_technique(...)` (from `dl_techniques.utils.keras_registration`), with
full `get_config()` / `from_config()` round-trip.

**Constructor (relevant args)**:

```python
DepthAnything(
    encoder_type:           str = 'vit_l',                # {'vit_s','vit_b','vit_l'}
    image_shape:            Tuple[int,int,int] = (518, 518, 3),   # sides multiples of 14
    dpt_features:           Optional[int] = None,         # None: the published value per encoder
    dpt_out_channels:       Optional[List[int]] = None,   # None: the published values per encoder
    output_channels:        int  = 1,
    kernel_initializer:     Union[str, Initializer] = 'he_normal',
    kernel_regularizer:     Optional[Regularizer]   = None,
    loss_weights:           Optional[Dict[str,float]] = {'labeled':1.0,
                                                         'unlabeled':0.5,
                                                         'feature':0.1},
    cutmix_prob:            float = 0.5,
    color_jitter_strength:  float = 0.2,
    input_value_range:      Optional[Tuple[float,float]] = (0.0, 1.0),
    use_feature_alignment:  bool  = True,
    encoder_kind:           str   = 'real',           # 'real' | 'placeholder'
    enable_semi_supervised: bool  = False,
    encoder:                Optional[keras.Model] = None,   # for from_config
    include_top:            bool  = True,
    **kwargs,
)
```

`input_shape`, a deprecated alias for `image_shape`, was removed on 2026-10-15 with no replacement:
pass `image_shape`. A saved config that still carries `input_shape` no longer loads.

**Save/load.** `DepthAnything` overrides Keras 3's `load_own_variables` to
force-build the nested sub-Models with a dummy forward before the framework
restores into them; without it the path-based restore lands on a tree whose
sub-layers do not exist yet and (with the wrapped `ViT` sub-Model of the time) 55 of 172 kernel
arrays come back re-initialized. The save side is the stock Keras recursive save;
the model must not add a second, flat dump of `self.weights` on top of it, because
that runs alongside the framework's path-walking rather than replacing it and
doubles every archive (`vit_l`/384: 610 weights, 1220 HDF5 datasets, 4.88 GB
instead of 2.44 GB). The guard is
`tests/test_models/test_depth_anything/test_the_archive_holds_each_weight_once.py`.

**EMA teacher.** `update_teacher_ema(decay=0.999)` advances the frozen
teacher's weights toward the student via in-place EMA. Call this from a
custom on-step callback when training semi-supervised. Default decay is
0.999.

### `create_depth_anything(...)` and `DepthAnything.from_variant(...)` (in `model.py`)

`from_variant(variant, pretrained=False, **kwargs)` takes the encoder size (`'vit_s'`, `'vit_b'`,
`'vit_l'`) and returns a built model; `create_depth_anything` delegates to it and ends with a dummy
forward pass:

```python
from dl_techniques.models.vision.depth_anything import create_depth_anything

model = create_depth_anything(
    encoder_kind='real',
    encoder_type='vit_l',
    image_shape=(518, 518, 3),
    use_feature_alignment=False,
    pretrained=True,          # the official DINOv2 ViT-L/14 encoder weights
)
```

### Pretrained encoder

`pretrained=True` follows `models/GUIDE.md` section 1: the official file is located first
(`FileNotFoundError`, naming the path, before anything is built), the model is built, then
`load_official_encoder_weights()` loads the encoder strictly through
`dl_techniques.models.vision.dino.official_weights` (every official tensor used once with a
matching shape, every encoder weight filled, or `ValueError` with nothing assigned) and re-copies
student -> teacher. The files are not distributed; download each `model.safetensors` once to
`<dir>/dinov2-{small,base,large}/model.safetensors`, where `<dir>` is
`/media/arxwn/data0_4tb/datasets/pretrained/dinov2` or `$DL_TECHNIQUES_DINOV2_WEIGHTS_DIR`.
All three sizes are wired (sha256 of each published file in `OFFICIAL_DINOV2_CHECKPOINTS`).
The encoder expects ImageNet-standardized RGB, as the reference's transform produces. Guards:
`tests/test_models/test_depth_anything/test_the_encoder_is_the_references_dinov2.py` (the four
maps the head reads equal a float64 numpy port of the official model over the downloaded file)
and `tests/test_models/test_dino/test_the_official_dinov2_weights_load_strictly.py`.

Until 2026-10-17 the `'real'` encoder was the in-tree `ViT` at patch 16, with its own position
table; an archive saved before that does not load into this model, and `image_shape` sides that
are not multiples of 14 are now refused with the DINOv2 encoder.

### `DPTDecoder` (in `components.py`)

`keras.layers.Layer` registered as
`dl_techniques.models.depth_anything.components>DPTDecoder` via
`@register_dl_technique(...)`. Convolutional decoder head with optional bilinear
upsampling. `DepthAnything` does not use it (its decoder is `DPTHead`); OmniPoint's dense heads
do, and keep it unchanged (BatchNorm included).

**Architecture**:

```
features (B, h, w, C_in)
   │
   ├── for i, dim in enumerate(dims):
   │       Conv3x3(dim) → BN → activation → [UpSample2D(2x bilinear) if i < log2(upsample_factor)]
   │
   ├── Conv3x3(output_channels) with output_activation        # default 'linear'
   │
   └── output (B, h*upsample_factor, w*upsample_factor, output_channels)
```

`output_activation` defaults to `'linear'`. `upsample_factor` defaults to `1` and must be a power of
two at most `2 ** len(dims)`.

### `StrongAugmentation` (in `dl_techniques.layers.signal_processing.strong_augmentation`)

CutMix + color jitter. Used by `DepthAnything` from inside `train_step`, not
from `call`.

* `call(x, training=True)` returns the mixed image alone and is correct only
  for a target-free consumer. A supervised consumer uses
  `augment_with_mix(x, training=True) -> (x_aug, (mix_mask, perm_indices))` and
  feeds that descriptor to `apply_mix_to_target(target, mix)` for every target
  sharing the batch and spatial axes — the channel count is free, so a
  `(B,H,W,2)` depth+validity target works unchanged.
* `input_value_range` is the caller's **declared** input range; colour jitter
  clips its result back into it. `None` disables clipping, which is what
  standardized or `[-1, +1]` images need — `src/train/depth_anything/` sets it,
  since `src/train/common/megadepth.py` emits RGB in `[-1, +1]` and clipping
  those to `[0, 1]` would zero every negative pixel on the training path only.
* Colour jitter is dtype-preserving: the per-sample factors are cast to the
  batch dtype, so a float16 batch stays float16 on the training path.
* `cutmix_ratio_range` is the fraction of each image *side* that is cut, so the
  pasted area is the ratio squared.
* `keras.random.uniform/shuffle` is used in place of the nonexistent
  `keras.ops.random.*`.
* Cutmix gating uses a symbolic mask multiplier (no Python `if`), so the
  layer is fully graph-traceable inside `model.fit`.

---

## Usage

### Forward pass / inference

```python
import keras
from dl_techniques.models.vision.depth_anything import create_depth_anything

model = create_depth_anything(encoder_kind='real', encoder_type='vit_l',
                              image_shape=(518, 518, 3),
                              use_feature_alignment=False)
x = keras.random.normal([2, 518, 518, 3])
depth = model(x, training=False)
print(depth.shape)  # (2, 518, 518, 1)
```

### Compile / train (labeled-only)

```python
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5),
    loss=keras.losses.MeanSquaredError(),
)
# model.fit(x_train, y_train, epochs=...)
```

### Semi-supervised usage

```python
import keras
from dl_techniques.models.vision.depth_anything import (
    create_depth_anything, TeacherEMACallback, cosine_ema_schedule,
)
from train.common.megadepth import (
    MegaDepthDataset, UnlabeledImageDataset, pair_labeled_unlabeled,
)

model = create_depth_anything(
    encoder_kind='real', encoder_type='vit_l', image_shape=(518, 518, 3),
    use_feature_alignment=True, enable_semi_supervised=True,
)

# Optional: load encoder weights from a saved .keras checkpoint
# (re-syncs the EMA teacher automatically).
# model.from_pretrained_encoder('/path/to/encoder.keras')

model.compile(optimizer=keras.optimizers.AdamW(1e-4))

# Build a paired ((x_lab, x_unlab), y_lab) tf.data.Dataset.
labeled_ds   = MegaDepthDataset(rgb_paths, depth_paths, batch_size=8, patch_size=518)
unlabeled_ds = UnlabeledImageDataset(unlab_paths, batch_size=8, patch_size=518)
paired_ds    = pair_labeled_unlabeled(labeled_ds, unlabeled_ds, patch_size=518, batch_size=8)

# On-step EMA decay (cosine 0.5 → 0.999 over the run).
total_steps = len(labeled_ds) * 100
ema_cb = TeacherEMACallback(
    schedule=cosine_ema_schedule(0.5, 0.999, total_steps),
    warmup_steps=0,
)

model.fit(paired_ds, epochs=100, steps_per_epoch=len(labeled_ds),
          callbacks=[ema_cb])
```

The semi-supervised `train_step` adds two losses on top of the labeled loss:
**FAL** between pooled student/teacher features, and **pseudo-label
consistency** between the student's depth and the teacher's stop-gradient depth
pseudo-labels. The consistency term is the paper's (arXiv 2401.10891, eq. 6-8): the
affine-invariant loss of eq. 1-3 (median shift, mean-absolute-deviation scale, MAE)
computed separately over the pixels outside the CutMix box and inside it, averaged by
their pixel shares (`DepthAnything._unlabeled_consistency`); without a mix it is that
loss over the whole map. It was a plain L1 until plan-2026-10-14T062616-81938e81 D-113. In both terms the student sees the *augmented* unlabeled batch
while the teacher reads the *clean* one — that asymmetry is the recipe — and the
teacher's pseudo-depth is then mixed by the same CutMix box the student's input
received, so the consistency term never asks the student to reproduce another
scene's depth over the cut rectangle.

### Serialization

```python
model.save('depth_anything.keras')
loaded = keras.models.load_model('depth_anything.keras')
```

The round-trip is exact on CPU (max-abs-diff = 0.0).

---

## Configuration

| Argument                 | Type                       | Default                                | Notes |
|--------------------------|----------------------------|----------------------------------------|-------|
| `encoder_type`           | `str`                      | `'vit_l'`                              | One of `{vit_s, vit_b, vit_l}`: DINOv2 ViT-S/14, ViT-B/14, ViT-L/14. |
| `encoder_kind`           | `str`                      | `'real'`                               | `'real'` builds the DINOv2 backbone; `'placeholder'` builds the legacy Conv-BN-ReLU. |
| `image_shape`            | `Tuple[int,int,int]`       | `(518, 518, 3)`                        | Input image shape; sides multiples of 14 with the DINOv2 encoder. |
| `dpt_features`           | `int` \| `None`            | `None` (64 / 128 / 256)                | `DPTHead` fusion width; `None` takes the published value for `encoder_type`. |
| `dpt_out_channels`       | `List[int]` \| `None`      | `None` (see "DPT head")                | The head's four projection widths. |
| `output_channels`        | `int`                      | `1`                                    | Final depth-map channel count. |
| `kernel_initializer`     | `str` \| `Initializer`     | `'he_normal'`                          | Reaches every decoder conv (one clone per conv) and the placeholder encoder; the DINOv2 encoder uses its own initializers (or the official weights). |
| `kernel_regularizer`     | `Regularizer` \| `None`    | `None`                                 | Reaches every decoder conv and the placeholder encoder, not the DINOv2 encoder. Do **not** combine with `AdamW(weight_decay=...)`. |
| `loss_weights`           | `Dict[str,float]`          | `{labeled:1.0, unlabeled:0.5, feature:0.1}` | Consumed by the semi-sup `train_step` path. |
| `cutmix_prob`            | `float`                    | `0.5`                                  | Forwarded to `StrongAugmentation`; unlabeled batches only. |
| `color_jitter_strength`  | `float`                    | `0.2`                                  | Forwarded to `StrongAugmentation`; unlabeled batches only. |
| `input_value_range`      | `Tuple[float,float]` \| `None` | `(0.0, 1.0)`                       | Declared input range for the colour-jitter clip. `None` for standardized / `[-1,+1]` inputs. |
| `use_feature_alignment`  | `bool`                     | `True`                                 | Adds the FAL term in the semi-supervised step. Also builds `frozen_encoder` — but so does `enable_semi_supervised`, which needs a teacher for its consistency term. Setting this True with `enable_semi_supervised=False` builds a teacher nothing reads, and warns. |
| `enable_semi_supervised` | `bool`                     | `False`                                | Switches `train_step` to `((x_lab, x_unlab), y_lab)` mode. |

The decoder, `DPTHead`, built in `build()` when `include_top=True`, takes `features=dpt_features`,
`out_channels=dpt_out_channels`, `patch_size=14` (16 with the placeholder), `output_channels`, and `kernel_initializer`
(cloned per conv) and `kernel_regularizer`. It has no range knob (`min_depth`, `max_depth`,
`use_log_depth`); see "Output range".

---

## Known Issues

* Custom `train_step` uses `tf.GradientTape` rather than the default Keras
  `train_step`. This is a *sanctioned* exception, not an open defect: the
  semi-supervised path is dual-batch with asymmetric augmentation and reads a
  teacher outside the loss graph, none of which
  `compute_loss(x, y, y_pred, sample_weight, training)` can carry. The rationale
  is written into `model.py`'s module docstring so it ships with the code. Do not
  cite this file as precedent for an ordinary single-batch model. *(LOW.)*

Both `train_step` paths route their metrics through `_finalize_train_step`, which
feeds `self._loss_tracker`; Keras 3 updates that tracker inside the default
`train_step` rather than inside `compute_loss`, so a custom step that skips
`"loss"` in its metrics loop makes `history.history["loss"]` all zeros and kills
every `ModelCheckpoint` / `EarlyStopping` / `ReduceLROnPlateau` that monitors it.
The returned logs dict is flat, not nested under `"compile_metrics"`.
Guards: `tests/test_models/test_depth_anything/test_train_step.py`.

---

## References

- Yang, Lihe et al. **"Depth Anything: Unleashing the Power of Large-Scale
  Unlabeled Data."** CVPR 2024.
- Ranftl, René et al. **"Vision Transformers for Dense Prediction"** (DPT
  decoder). ICCV 2021.
- LiheYoung/Depth-Anything, `depth_anything/dpt.py` and `depth_anything/blocks.py` (the head
  `DPTHead` ports), and the `config.json` of `LiheYoung/depth_anything_vit{s,b,l}14`.
- Oquab, Maxime et al. **"DINOv2: Learning Robust Visual Features without
  Supervision."** 2023. The encoder; facebookresearch/dinov2
  `dinov2/models/vision_transformer.py` and the Hugging Face `facebook/dinov2-*` checkpoints.
- In-tree canonical Keras-3 `train_step` pattern:
  `src/dl_techniques/models/common/masked_language_model/mlm.py`.

---

## See also

- `src/train/depth_anything/` — Pattern-5 training scaffold for this model
  (MegaDepth + masked depth loss + visualization callbacks).
- `src/train/AGENTS.md` § "Pattern 5: Depth Estimation (MegaDepth)" — the pattern
  this trainer implements. `train_depth_anything.py` is the only Pattern-5 trainer.
- `src/dl_techniques/models/vision/depth_anything/components.py` — `DPTHead` and `DPTDecoder` source.
- `src/dl_techniques/models/vision/depth_anything/model.py` — `DepthAnything` source.
- `tests/test_models/test_depth_anything/` — pytest coverage.
