# `dl_techniques.layers.heads.vision`

Task heads that sit on top of a vision backbone: they take the backbone's feature map and produce
detections, masks, depth, class logits or an enhanced image. Two modules, sixteen exported names —
eight head classes, three factory functions, `HeadConfiguration`, and the task-type vocabulary
(`VisionTaskType`, `TaskConfiguration`, `CommonTaskConfigurations`, `parse_task_list`).

```python
from dl_techniques.layers.heads.vision import create_vision_head
head = create_vision_head('detection', num_classes=80, num_anchors=9)
```

The wider facade `dl_techniques.layers.heads.create_head('vision', ...)` calls `create_vision_head`
for you. `get_task_suggestions` and `validate_task_combination` are **not** exported from the
package; import them from `.task_types`. Nor is `merge_head_config_overrides`, the helper a model
uses to merge its `head_config_overrides` into its head config (it refuses `name`, `num_classes` and
`task_type`, which the model owns); import it from `.factory`.

## Head dispatch: 10 of 37 task types reach a head

`VisionTaskType` names 37 tasks. `create_vision_head` is the arbiter of which head class runs which;
**the other 27 raise `ValueError("Unsupported task type: ...")`.**

| Task type | Head built | Output |
|---|---|---|
| `detection` | `DetectionHead` | `{'classifications': (B, H, W, A*C), 'regressions': (B, H, W, A*4)}` |
| `keypoint_detection` | `DetectionHead` | same two keys |
| `segmentation` | `SegmentationHead` | one tensor `(B, H*4, W*4, num_classes)` of class logits (no dict, no softmax) |
| `instance_segmentation` | `InstanceSegmentationHead` | detection's two keys plus `'instance_masks'`, mask logits (no sigmoid) |
| `classification` | `ClassificationHead` | `{'logits': (B, C), 'probabilities': (B, C)}` |
| `depth_estimation` | `DepthEstimationHead` | `{'depth': (B, 8H, 8W, 1), 'confidence': same}` |
| `surface_normals` | `DepthEstimationHead(output_channels=3)` | same keys, 3 channels |
| `optical_flow` | `DepthEstimationHead(output_channels=2)` | same keys, 2 channels |
| `denoising` | `EnhancementHead` | `{'enhanced': (B, H, W, 3)}` |
| `super_resolution` | `EnhancementHead(scale_factor=2)` | `{'enhanced': (B, 2H, 2W, 3)}` |

No head (raises): `panoptic_segmentation`, `stereo_matching`, `motion_segmentation`,
`pose_estimation`, `edge_detection`, `line_detection`, `saliency_detection`,
`attention_prediction`, `inpainting`, `dehaze`, `shadow_removal`, `reflection_removal`,
`colorization`, `style_transfer`, `white_balance`, `matting`, `hair_segmentation`,
`sky_segmentation`, `medical_segmentation`, `cell_counting`, `text_detection`, `document_layout`,
`depth_completion`, `surface_reconstruction`, `camera_pose`, `image_quality`, `aesthetic_scoring`.

`VisionTaskType.get_task_categories()` files every member into one of twelve categories. That filing
is descriptive only — the factory never reads it.

## Head-specific constructor arguments

| Head | Arguments |
|---|---|
| `DetectionHead` | `num_classes` (required), `num_anchors=9`, `bbox_dims=4`, `input_format='spatial'` (`'sequence'` takes `(B, N, D)` patch tokens), `patch_grid_size=None` (the `(h, w)` token grid, required for `'sequence'`), `use_regression=True` (`False` builds no regression branch and returns only `'classifications'`) |
| `SegmentationHead` | `num_classes` (required), `upsampling_factor=4` (output over input resolution, a power of two, else `ValueError`; until 2026-10-15 it built `int(sqrt(f))` doublings, so 8 gave 4x), `use_skip_connections=True`, `skip_channel_mode='auto'`, `explicit_skip_channels=None` (required for `skip_channel_mode='explicit'`), `input_format='spatial'`, `patch_grid_size=None` (as for `DetectionHead`) |
| `InstanceSegmentationHead` | `num_classes` (required), `num_instances=100`, `mask_size=(28, 28)` |
| `ClassificationHead` | `num_classes` (required), `use_global_pooling=True`, `pooling_type='avg'`, `block_dims=None` (hidden `DenseBlock` widths; `None` is `(hidden_dim, hidden_dim // 2)`, `()` is none), `pooled_normalization_type=None` (a norm on the pooled vector), `normalization_kwargs=None` (to every head norm), `classifier_dropout_rate=0.0` (a dropout on the classifier input), `block_use_bias=True` (the hidden blocks' Dense bias; `False` before a `batch_norm`), `kernel_initializer='glorot_uniform'`, `bias_initializer='zeros'`, `kernel_regularizer=None`, `bias_regularizer=None` (the last four reach every `Dense`, the classifier included) |
| `DepthEstimationHead` | `output_channels=1`, `decoder_style='progressive'` (`'dpt'` is DepthAnything's decoder, see the output contract). Progressive only: `min_depth` (0.1), `max_depth` (100.0), `use_log_depth` (`True`). DPT only: `dpt_dims` (`[256, 128, 64, 32]`), `dpt_use_bias` (`False`), `dpt_activation` (`'relu'`), `dpt_output_activation` (`'linear'`), `dpt_upsample_factor` (16; a power of 2, at most `2 ** len(dpt_dims)`), `dpt_kernel_initializer` (`'he_normal'`, cloned per conv), `dpt_kernel_regularizer` (none). Each mode-specific argument defaults to `None`, meaning the value in parentheses in its own mode; the other mode refuses it with `ValueError` and stores `None`. The DPT mode also refuses `normalization_type` and `activation_type` (only the progressive `ConvBlock`s read them) |
| `EnhancementHead` | `output_channels=3`, `scale_factor=1`, `block_dims=None` (refinement `ConvBlock` widths; `None` is three at `hidden_dim`, `()` is none, so the head is ONE 3x3 conv: the tail conv SCUNet and DarkIR use, with the model applying any skip or residual), `output_use_bias=True` (the output conv's bias; SCUNet's tail has none) |
| `MultiTaskHead` | `task_configs` (required dict), `shared_backbone_dim=256`, `use_task_specific_attention=True` |

Every head also takes the shared `BaseVisionHead` options:

| Argument | Default | Notes |
|---|---|---|
| `hidden_dim` | `256` | Internal width. |
| `normalization_type` | `'layer_norm'` | Any `norms/` factory key. |
| `activation_type` | `'gelu'` | Any `activations/` factory key. |
| `dropout_rate` | `0.1` | `0.0` and refused above 0 in `EnhancementHead`, which applies no dropout. |
| `use_attention` | `False` | See the gotchas — the default `attention_type` needs 3D input. |
| `attention_type` | `'multi_head'` | Any `ATTENTION_REGISTRY` key. |
| `use_ffn` | `True` | `False` and refused in `ClassificationHead` and `EnhancementHead`. |
| `ffn_type` | `'mlp'` | Any `FFN_REGISTRY` key. |
| `ffn_expansion_factor` | `4` | |

`HeadConfiguration` supplies presets for those: `get_default_config(task)` and
`get_efficient_config(task)`, each returning a plain dict you can splat into a head. Both apply
`PRESET_FIXED_KEYS` last, so a classification preset carries `use_ffn=False` and a denoising or
super-resolution preset `use_attention=False, use_ffn=False, dropout_rate=0.0`. A third preset,
`get_high_performance_config`, was deleted on 2026-10-14: its `attention_type='differential'`
needs `num_heads` and `head_dim`, which the heads never pass, so it raised `ValueError` for every
task type.

## Output contract: logits, one range map

No head applies an activation to a class score. `ClassificationHead`'s `'logits'`, the
`SegmentationHead` map, `DetectionHead`'s `'classifications'` and `InstanceSegmentationHead`'s
`'instance_masks'` are pre-activation: compile with `from_logits=True`, or apply
`keras.ops.softmax` (classes) / `keras.ops.sigmoid` (masks) yourself. `ClassificationHead` also
returns `'probabilities' = keras.ops.softmax('logits')`, computed in the compute dtype (a float16
softmax under `mixed_float16`, as in the NLP `TextClassificationHead`).

The one squashed output is the progressive `DepthEstimationHead`'s sigmoid, a range map the depth is
scaled from; its `'confidence'` is that sigmoid output. The DPT decoder (`decoder_style='dpt'`) is
different: `'depth'` is `dpt_output_conv` under `dpt_output_activation` (`'linear'` by default;
DepthAnything passes `'relu'`, the reference Depth Anything head's final ReLU), there is no range
scaling, and `'confidence'` is the last decoder stage's features, `dpt_dims[-1]` channels wide.
`min_depth`, `max_depth` and `use_log_depth` used to be accepted, stored and ignored in DPT mode, as
every `dpt_*` argument was in progressive mode; since 2026-10-15 each mode refuses the other's
arguments, so a DPT-mode archive whose config carries the three range values no longer loads.

Archives saved before 2026-10-15 are not supported. Measured on archives written at `def68694a`:
`ClassificationHead` and `EnhancementHead` archives fail to load (their stored `use_ffn=True` is now
refused); `InstanceSegmentationHead` archives (sub-layer paths moved) and progressive
`DepthEstimationHead` archives (they never held the refinement kernels, see below) fail to load; a
`SegmentationHead` archive loads and returns class logits where it returned probabilities; a
`DetectionHead` archive loads.

Three save/load defects were fixed on 2026-10-15. Before then a model holding a `SegmentationHead`
could be saved but never loaded (`load_model` rebuilt the head from a JSON shape list it misread as
several shapes); a progressive `DepthEstimationHead` kept its three `ConvBlock`s and three
`Conv2DTranspose` layers in a nested list, which Keras does not save, so a reloaded head silently
ran with fresh kernels there; and a model holding a `MultiTaskHead` could not be saved at all (its
config held `VisionTaskType` members). Every head now round-trips by value
(`tests/test_layers/test_heads/test_every_vision_head_survives_a_keras_round_trip.py`).

## Input contract

Heads consume a **4D feature map** `(B, H, W, C)`. The exceptions take a ViT-style token sequence
`(B, N, C)`: `ClassificationHead` with `pooling_type='cls_token'` (classifies token 0) or with
`use_global_pooling=False` (flattens every token), and `DetectionHead` / `SegmentationHead` with
`input_format='sequence'` and `patch_grid_size=(h, w)`, which lay the tokens out on that grid.
`SegmentationHead` with `use_skip_connections=True` also accepts a list `[*skips, features]` — the
highest-level map goes last, and the skips are concatenated deepest-first, one after each refinement
block. The refinement blocks keep the resolution and the upsampling runs only after them, so every
skip must have the feature map's own height and width.

```python
import keras
from dl_techniques.layers.heads.vision import (
    create_vision_head, create_multi_task_head, HeadConfiguration, VisionTaskType,
)

features = keras.random.normal((2, 16, 16, 64))     # any backbone's feature map

det = create_vision_head('detection', num_classes=80, num_anchors=9)
out = det(features)
# out['classifications'] -> (2, 16, 16, 720);  out['regressions'] -> (2, 16, 16, 36)

seg = create_vision_head('segmentation', num_classes=21, use_skip_connections=False)
mask = seg(features)                                 # (2, 64, 64, 21)

depth = create_vision_head('depth_estimation')
d = depth(features)                                  # d['depth'], d['confidence'] -> (2, 128, 128, 1)

cls_head = create_vision_head('classification', num_classes=1000)
logits = cls_head(features)['logits']                # (2, 1000)

multi = create_multi_task_head(
    {
        'detection': {'task_type': VisionTaskType.DETECTION, 'num_classes': 80},
        'segmentation': {'task_type': VisionTaskType.SEGMENTATION, 'num_classes': 21,
                         'use_skip_connections': False},
    },
    use_task_specific_attention=False,               # required for 4D input, see gotchas
)
both = multi(features)                               # {'detection': {...}, 'segmentation': tensor}

cfg = HeadConfiguration.get_efficient_config(VisionTaskType.DETECTION)
cheap_det = create_vision_head('detection', **cfg)
```

## Gotchas

- **`BaseVisionHead` is exported but is not a usable head.** It defines no `call`. It builds
  `attention` and `ffn` when their flags ask, and nothing else; each head normalizes and drops
  inside its own `ConvBlock` / `DenseBlock`. Use one of the seven task heads.
- **`ClassificationHead(use_ffn=True)` and `EnhancementHead(use_attention=True, use_ffn=True or
  dropout_rate > 0)` raise `ValueError`.** Their `call` never runs those layers, so the arguments
  default to off there and any other value is refused rather than stored and ignored.
- **`use_attention=True` fails on a 4D feature map with the default `attention_type='multi_head'`**,
  which requires `(batch, seq_len, dim)`. For a feature map choose a 4D attention type
  (`cbam`, `channel`, `spatial`, `tripse*`, `non_local`).
- **`cbam` / `channel` attention require the input channel count to equal `hidden_dim`** (default
  256). Pass `hidden_dim=<your backbone's channels>` or you get
  `ValueError: Expected input channels (64) to match layer channels (256)`.
- **`MultiTaskHead` defaults to `use_task_specific_attention=True`, which builds `multi_head`
  attention** and therefore fails on a 4D input. Pass `use_task_specific_attention=False` for
  feature maps.
- **`create_multi_task_head`'s list and `TaskConfiguration` forms take each task's options as a
  keyword dict named after the task's value**, e.g.
  `create_multi_task_head([VisionTaskType.DETECTION, 'classification'], detection={'num_classes': 80},
  classification={'num_classes': 10}, use_task_specific_attention=False)`. Each such dict is
  consumed; the remaining keywords go to `MultiTaskHead`. A detection, segmentation or
  classification task given no `num_classes` raises `TypeError`. (Until 2026-10-15 the per-task
  dicts were also forwarded to `MultiTaskHead`, which refused them, so these forms built only
  depth tasks.)
- **`MultiTaskHead` accepts `task_type` as a `VisionTaskType` member or its string value**, and
  `get_config` stores the string, so a model holding it survives a `.keras` round trip.
- **Detection outputs are per-cell, not per-anchor-row.** `classifications` is
  `(B, H, W, num_anchors * num_classes)` and `regressions` is `(B, H, W, num_anchors * bbox_dims)`;
  reshaping to `(B, num_anchors, num_classes)` is the caller's job.
- **`SegmentationHead` returns a bare tensor**, not a dict, unlike every other head here.
- **Depth-family heads upsample by 8x** with the default progressive decoder — a `(2, 16, 16, 64)`
  input gives `(2, 128, 128, C)` — and by `dpt_upsample_factor` (16 by default) with the DPT
  decoder. Size your backbone stride accordingly.
