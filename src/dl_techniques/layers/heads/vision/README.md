# `dl_techniques.layers.heads.vision`

Task heads that sit on top of a vision backbone: they take the backbone's feature map and produce
detections, masks, depth, class logits or an enhanced image. Two modules, seventeen exported names —
eight head classes, three factory functions, `HeadConfiguration`, and the task-type vocabulary
(`VisionTaskType`, its `TaskType` alias, `TaskConfiguration`, `CommonTaskConfigurations`,
`parse_task_list`).

```python
from dl_techniques.layers.heads.vision import create_vision_head
head = create_vision_head('detection', num_classes=80, num_anchors=9)
```

The wider facade `dl_techniques.layers.heads.create_head('vision', ...)` calls `create_vision_head`
for you. `get_task_suggestions` and `validate_task_combination` are **not** exported from the
package; import them from `.task_types`.

## Head dispatch: 10 of 37 task types reach a head

`VisionTaskType` names 37 tasks. `create_vision_head` is the arbiter of which head class runs which;
**the other 27 raise `ValueError("Unsupported task type: ...")`.**

| Task type | Head built | Output |
|---|---|---|
| `detection` | `DetectionHead` | `{'classifications': (B, H, W, A*C), 'regressions': (B, H, W, A*4)}` |
| `keypoint_detection` | `DetectionHead` | same two keys |
| `segmentation` | `SegmentationHead` | one tensor `(B, H*4, W*4, num_classes)` (no dict) |
| `instance_segmentation` | `InstanceSegmentationHead` | detection's two keys plus `'instance_masks'` |
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
| `DetectionHead` | `num_classes` (required), `num_anchors=9`, `bbox_dims=4` |
| `SegmentationHead` | `num_classes` (required), `upsampling_factor=4`, `use_skip_connections=True` |
| `InstanceSegmentationHead` | `num_classes` (required), `num_instances=100`, `mask_size=(28, 28)` |
| `ClassificationHead` | `num_classes` (required), `use_global_pooling=True`, `pooling_type='avg'` |
| `DepthEstimationHead` | `output_channels=1`, `min_depth=0.1`, `max_depth=100.0`, `use_log_depth=True` |
| `EnhancementHead` | `output_channels=3`, `scale_factor=1` |
| `MultiTaskHead` | `task_configs` (required dict), `shared_backbone_dim=256`, `use_task_specific_attention=True` |

Every head also takes the shared `BaseVisionHead` options:

| Argument | Default | Notes |
|---|---|---|
| `hidden_dim` | `256` | Internal width. |
| `normalization_type` | `'layer_norm'` | Any `norms/` factory key. |
| `activation_type` | `'gelu'` | Any `activations/` factory key. |
| `dropout_rate` | `0.1` | |
| `use_attention` | `False` | See the gotchas — the default `attention_type` needs 3D input. |
| `attention_type` | `'multi_head'` | Any `ATTENTION_REGISTRY` key. |
| `use_ffn` | `True` | |
| `ffn_type` | `'mlp'` | Any `FFN_REGISTRY` key. |
| `ffn_expansion_factor` | `4` | |

`HeadConfiguration` supplies presets for those: `get_default_config(task)`,
`get_efficient_config(task)` and `get_high_performance_config(task)`, each returning a plain dict you
can splat into a head.

## Input contract

Heads consume a **4D feature map** `(B, H, W, C)`. The exception is `ClassificationHead` with
`use_global_pooling=False`, which accepts a ViT-style token sequence `(B, N, C)`.
`SegmentationHead` with `use_skip_connections=True` also accepts a list `[*skips, features]` — the
highest-level map goes last, and the skips are consumed deepest-first, so each skip's spatial size
must match the head's intermediate resolution at that step.

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

- **`BaseVisionHead` is exported but is not a usable head.** It defines no `call`. It builds four
  sub-layers — `norm`, `dropout`, `attention`, `ffn` — but only `attention` and `ffn` are ever
  applied by a subclass `call`. `norm` carries weights that no forward pass reaches; each head
  normalizes and drops inside its own `ConvBlock` / `DenseBlock` instead. Use one of the seven task
  heads.
- **`use_attention=True` fails on a 4D feature map with the default `attention_type='multi_head'`**,
  which requires `(batch, seq_len, dim)`. For a feature map choose a 4D attention type
  (`cbam`, `channel`, `spatial`, `tripse*`, `non_local`).
- **`cbam` / `channel` attention require the input channel count to equal `hidden_dim`** (default
  256). Pass `hidden_dim=<your backbone's channels>` or you get
  `ValueError: Expected input channels (64) to match layer channels (256)`.
- **`MultiTaskHead` defaults to `use_task_specific_attention=True`, which builds `multi_head`
  attention** and therefore fails on a 4D input. Pass `use_task_specific_attention=False` for
  feature maps.
- **`create_multi_task_head`'s per-task option dicts do not work with a list or `TaskConfiguration`
  argument.** `kwargs` is read for per-task options *and* forwarded whole to `MultiTaskHead`, so
  `create_multi_task_head([...], detection={'num_classes': 80})` raises
  `ValueError: Unrecognized keyword arguments passed to MultiTaskHead`. Pass the dict form shown
  above, where each entry carries its own `task_type` key.
- **Detection outputs are per-cell, not per-anchor-row.** `classifications` is
  `(B, H, W, num_anchors * num_classes)` and `regressions` is `(B, H, W, num_anchors * bbox_dims)`;
  reshaping to `(B, num_anchors, num_classes)` is the caller's job.
- **`SegmentationHead` returns a bare tensor**, not a dict, unlike every other head here.
- **Depth-family heads upsample by 8x** — a `(2, 16, 16, 64)` input gives `(2, 128, 128, C)`. Size
  your backbone stride accordingly.
