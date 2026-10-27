# RADConvNet

A ConvNeXt-shaped classification backbone that swaps the depthwise `KxK` convolution inside
each block for
[`RADConv2D`](../../../layers/conv_blocks/rad_conv.py) — Region-Aware Deformable Convolution
(Maleki & Imani, 2025, [arXiv:2509.15436](https://arxiv.org/abs/2509.15436)).

RAD-Conv predicts, per kernel element and channel group, four non-negative boundary offsets
that define an axis-aligned rectangle whose extent and aspect ratio adapt to image content, and
exactly spatially-integrates the input over that rectangle via a summed-area table. This
decouples receptive-field geometry from kernel size — even a `rad_kernel_size=1` block can
aggregate over a region as large as the whole feature map — while keeping a convolution's
translation-equivariant, locally-parameterized inductive bias.

## Usage

```python
from dl_techniques.models.vision.rad_convnet.model import RADConvNet, create_rad_convnet

model = RADConvNet.from_variant("tiny", num_classes=47, input_shape=(96, 96, 3))
# or
model = create_rad_convnet(variant="tiny", num_classes=47, input_shape=(96, 96, 3))

# A feature extractor: the final stage's maps, classification_head is None
backbone = RADConvNet.from_variant("small", include_top=False)

# An architecture field of the variant can be replaced
shallow = RADConvNet.from_variant("tiny", depths=[1, 1, 1, 1])
```

`from_variant(variant, num_classes=1000, input_shape=None, pretrained=False, **kwargs)` and
`create_rad_convnet` take the same arguments. `input_shape=None` resolves to `(None, None, 3)`.

## Variants

| Variant | Depths        | Dims                  |
|---------|---------------|------------------------|
| `tiny`  | `[2, 2, 4, 2]`  | `[32, 64, 128, 256]`   |
| `small` | `[3, 3, 9, 3]`  | `[48, 96, 192, 384]`   |
| `base`  | `[3, 3, 9, 3]`  | `[64, 128, 256, 512]`  |

Deliberately smaller than ConvNeXt's own `[96, 192, 384, 768]`-style defaults: RAD-Conv's exact
region integration (a summed-area table plus per-kernel-element box sampling) is materially more
expensive per channel than a depthwise convolution.

## Head

The head is the shared vision
[`ClassificationHead`](../../../layers/heads/vision/classification.py), stored as
`model.classification_head`: global average pooling, a `pooled_norm` LayerNorm at epsilon `1e-6`,
and one linear `classifier` (kernel drawn from `truncated_normal`), no hidden block. It is
ConvNeXt's `self.head(self.norm(x.mean([-2, -1])))`. `call` returns the head's logits.

- `use_bias=False` builds the whole model bias-free, the head included: the `pooled_norm` has no
  `beta` and the `classifier` has no `bias` (the head's `classifier_use_bias` knob).
- `include_top=False` leaves `classification_head` defined as `None` and returns the final stage's
  feature maps `(batch, H', W', dims[-1])`.
- `num_classes` must be positive while `include_top=True`: `num_classes=0` raises `ValueError`
  naming the value. The earlier pooled-feature mode (`num_classes=0` returning the normalized pooled
  vector) is gone; build the model with `include_top=False` and pool the maps.
- `head_config_overrides` is merged into the head configuration (for example
  `{"classifier_dropout_rate": 0.2}`); `name`, `num_classes` and `task_type` are owned by the model
  and raise `ValueError`, and every value must be JSON data.
- `dropout_rate` is the dropout inside each block's MLP and must lie in `[0, 1)`; at `0.0` no
  Dropout layer is created.

## Layer names

| Layer | Name |
|-------|------|
| patchify stem | `stem` (a `ConvStem`: `stem_conv1`, `stem_norm1`) |
| block layers | `stage{S}_block{B}_<role>` with S and B counted from 1; roles `rad_conv`, `norm`, `expand`, `reduce`, `dropout` |
| downsampler | `downsample_{i}_norm`, `downsample_{i}_conv` (i counts from 0: the downsampler before stage i + 2) |
| shared activation | `mlp_activation` (one layer serves every block, no weight) |
| head | `classification_head` (`pooled_norm`, `classifier`) |

The Python attributes follow: `stages` (a list per stage of role-keyed dicts), `downsamplers`
(a `{"norm", "conv"}` dict each), `activation` (the constructor argument, as stored),
`classification_head`, `input_shape_config`.

## Loading weights

No pretrained weights are distributed with this package. Setting the `pretrained` argument to
`True` raises `NotImplementedError` naming the variant; pass a path to a local `.keras` checkpoint
instead. The
loader (`load_pretrained_checkpoint`) takes a different classifier width (`num_classes`), a headless
target and a different input size: only the `classification_head` layers may differ in shape or be
absent, a head left at its initialization is named in a warning, and every other layer must be
restored. A checkpoint written by this class whose configuration differs in an argument that is not
listed below raises `ValueError` naming the argument and both values (measured for `depths` and
`dims`, `strides`, `activation`, `rad_kernel_size`, `rad_groups`, `mlp_ratio` and `use_bias`), even
when every weight shape would fit (`activation`). `dropout_rate`, `input_shape`,
`num_classes` and `head_config_overrides` may differ (measured). A load that restores no layer at all
also raises. `input_shape` may stay at its default `(None, None, 3)`.

## Archives written before the shared head

A `.keras` file saved before 2026-10-26, when the stem moved onto one `ConvStem` named `stem`,
does not load: `keras.models.load_model` raises `ValueError`, and no legacy shim exists.

An archive saved at `be92a74eb` (the revision before the head and the layer names moved,
2026-10-27) loads when it is headless and no longer loads when it is headed (measured, tiny, with
`load_model`, `load_weights` and `from_variant(pretrained=<path>)`):

- the headless archive restores every tensor and the output is equal at `atol=0` in both
  `training=False` and `training=True`; only the `w.path` strings change, because archives load by
  tracked-layer order;
- the headed archive raises `ValueError: ... Layer 'classifier' expected 2 variables, but received 0
  variables during loading`: `head_norm` and `classifier` are now two layers inside one
  `classification_head`, so the tracked layers no longer line up.

The weights themselves are not lost. Carried over by `w.path` through the rename map below, the
140 weights of a headed `tiny` model with `use_bias=True` (90 with `use_bias=False`) give an output
equal at `atol=0` in `training=False` and `training=True`. The map: `stage{s}_block{b}_<role>` to
`stage{s+1}_block{b+1}_<role>`, `downsample_norm_{k}` to `downsample_{k}_norm`, `downsample_conv_{k}`
to `downsample_{k}_conv`, `head_norm/*` to `classification_head/pooled_norm/*` and `classifier/*`
to `classification_head/classifier/*`. 136 of the 140 weights of a headed `tiny` model move (the
stem and nothing else keeps its path), for example `stage0_block0_rad_conv/kernel` is now
`stage1_block1_rad_conv/kernel`. Code that selects weights by path must follow.

A seeded build is unchanged: with `keras.utils.set_random_seed` set, every weight and the output
equal the build before this change (`tiny`, headed and headless, and `use_bias=False`, dropout and
stride-1 variants).

Removed with this change: the attributes `gap`, `head_norm` and `classifier` (use
`classification_head.pooled_norm` and `classification_head.classifier`), `activation_name` (now
`activation`), `stages_list` and `downsample_layers_list` (now `stages` and `downsamplers`), and the
pooled-feature mode of `num_classes=0`.

## Training

See `src/train/rad_convnet/` for a training pipeline against the on-disk
[DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) (Describable Textures Dataset) texture
classification dataset.

## Known v1 limitations

- `RADConv2D` supports `strides=1` only; downsampling is handled by the surrounding
  ConvNeXt-style strided-conv layers between stages, exactly as in `ConvNeXtV1`/`ConvNeXtV2`.
- No stochastic-depth / drop-path (unlike `ConvNeXtV1`) — kept out to bound scope; can be added
  the same way `ConvNeXtV1` wires it, without changing `RADConv2D` itself.
