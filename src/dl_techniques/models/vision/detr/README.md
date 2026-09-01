# DETR: End-to-End Object Detection with Transformers

A Keras 3 implementation of **DETR** (Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko,
*End-to-End Object Detection with Transformers*, ECCV 2020,
[arXiv:2005.12872](https://arxiv.org/abs/2005.12872)) — object detection as **direct set
prediction**, with no anchors, no region proposals and no non-maximum suppression.

> **This package ships the architecture only.** There is no Hungarian matcher and no set loss
> here, so `model.compile(loss=...)` with a stock Keras loss will not train a detector. The
> ImageNet weights on the ResNet-50 backbone are real and do download; nothing above the
> backbone is pretrained and there is no DETR checkpoint. Read § 7 before comparing anything to
> published numbers.

## 1. Overview: What is DETR and Why It Matters

Classical detectors emit thousands of candidate boxes and then post-process them into a final
set: anchor generation, box regression against hand-designed priors, NMS to remove duplicates.
Every one of those steps carries hyperparameters that must be tuned per dataset.

DETR replaces the whole pipeline with one forward pass. A CNN backbone produces a feature map, a
transformer encoder-decoder attends over it, and `num_queries` learned **object queries** each
emit exactly one prediction: a class (including a "no object" class) and a box. Training uses
**bipartite matching** — a Hungarian assignment between predictions and ground truth — so each
object is claimed by exactly one query and duplicate suppression is learned rather than coded.

The cost is convergence speed — the original schedule is hundreds of epochs on COCO — and small
objects are its weakest case.

## 2. The Problem DETR Solves

Three structural problems with anchor-based detection:

- **Duplicates are a post-processing problem.** Many anchors fire on one object, so NMS is
  required, and NMS is a non-differentiable heuristic with its own IoU threshold.
- **Priors are hand-designed.** Anchor scales, aspect ratios and assignment rules encode dataset
  assumptions that do not transfer.
- **The receptive field is local.** A convolutional detector reasons about an object from a
  neighbourhood, not from the whole scene.

DETR's set prediction removes the first two and its encoder self-attention removes the third:
every feature cell attends to every other, so the model can reason about occlusion and about
relations between distant objects directly.

## 3. How DETR Works: Core Concepts

**Object queries.** `num_queries` learned embeddings (100 in the paper) are the decoder's input.
They are not tied to positions or scales; each one learns a specialization over training. The
model can therefore never detect more than `num_queries` objects in an image.

**Bipartite matching loss.** Predictions and ground-truth boxes are matched one-to-one by a
Hungarian assignment that minimizes a combined classification + L1 + GIoU cost. Unmatched
predictions are trained toward the "no object" class. This is what makes the output a *set*.

**Auxiliary losses.** With `aux_loss=True` the prediction heads are applied to the output of
every decoder layer except the last, and the same set loss is computed on each. Without this,
DETR converges much more slowly.

**Padding masks.** DETR is trained on variable-size images padded into a batch. A boolean mask
(`True` = padding) marks the padded pixels; the transformer excludes them from encoder
self-attention and decoder cross-attention.

## 4. Architecture Deep Dive

```
  images (B, H, W, 3)          padding_mask (B, H, W)  True = padding
      │                                │
   backbone (CNN)                      │  nearest-downsampled to the
      │  (B, H/16, W/16, 1024)         │  feature grid, then inverted
   input_proj  Conv2D 1x1              │  into a keep mask
      │  (B, H/16, W/16, hidden_dim)   │
   + PositionEmbeddingSine2D  (2-D sinusoidal, hidden_dim // 2 per axis)
      │  flattened to (B, H*W/256, hidden_dim)
   DetrTransformer
      encoder x num_encoder_layers   self-attention + FFN
      decoder x num_decoder_layers   self-attn + cross-attn to memory + FFN
      │  (num_decoder_layers, B, num_queries, hidden_dim)
      ├─► class_embed  Dense(num_classes + 1)          -> pred_logits (LOGITS)
      └─► bbox_embed   Dense(d)->ReLU->Dense(d)->ReLU->Dense(4), then sigmoid
                                                       -> pred_boxes in [0, 1]
```

`bbox_embed` is the paper's 3-layer perceptron, built explicitly rather than through
`create_ffn_layer('mlp', ...)`: that factory key is `MLPBlock`, which is `fc1 -> act -> fc2` —
**two** Dense layers, not three. Do not "restore reuse" by swapping the factory back in.

`hidden_dim` **must be a multiple of 4**. `PositionEmbeddingSine2D` receives
`num_pos_feats = hidden_dim // 2`, and that value must itself be even because the layer splits it
between its sine and cosine halves. The constructor raises with the next valid value.

Output dictionary:

| Key | Shape | Notes |
|:---|:---|:---|
| `pred_logits` | `(B, num_queries, num_classes + 1)` | **logits**; the extra class is "no object" |
| `pred_boxes` | `(B, num_queries, 4)` | sigmoid, so `[0, 1]`. Nothing here interprets the four numbers; the paper's convention is `cxcywh` and your loss decides |
| `aux_outputs` | list of `num_decoder_layers - 1` dicts | present only when `aux_loss=True` |

## 5. Quick Start Guide

`create_detr` builds the paper's configuration: a `keras.applications.ResNet50` backbone with
real ImageNet weights, frozen by default, tapped at `conv4_block6_out`.

```python
import numpy as np
from dl_techniques.models.vision.detr import create_detr

model = create_detr(
    num_classes=80,          # COCO classes, excluding "no object"
    num_queries=100,         # max detections per image
    backbone_name="resnet50",
    hidden_dim=256,
)

images = np.random.rand(1, 256, 256, 3).astype("float32")
mask = np.zeros((1, 256, 256), dtype=bool)     # False = valid pixel, True = padding

out = model([images, mask], training=False)
print(out["pred_logits"].shape, out["pred_boxes"].shape, len(out["aux_outputs"]))
# (1, 100, 81) (1, 100, 4) 5
```

The input is always a **two-element list** `[images, padding_mask]`; passing the images alone
raises. `padding_mask` may be `None` if you have no padding, but the position must be there.

Reading predictions — `pred_logits` holds logits, so apply a softmax before thresholding:

```python
import keras
import numpy as np

probs = keras.ops.softmax(out["pred_logits"], axis=-1)
probs = np.asarray(probs)[..., :-1]              # drop the "no object" column
scores = probs.max(axis=-1)
labels = probs.argmax(axis=-1)
boxes = np.asarray(out["pred_boxes"])            # [0, 1]; cxcywh by convention

keep = scores[0] > 0.7
print(labels[0][keep], scores[0][keep], boxes[0][keep])
```

Variable-size images: resize keeping aspect ratio, pad to a common size, and mark the padding.

```python
import numpy as np

def letterbox(image, target=(512, 512)):
    """Return (padded_image, padding_mask). True in the mask means padding."""
    h, w = image.shape[:2]
    padded = np.zeros((*target, 3), dtype="float32")
    padded[:h, :w] = image[:target[0], :target[1]]
    mask = np.ones(target, dtype=bool)
    mask[:min(h, target[0]), :min(w, target[1])] = False
    return padded, mask
```

The mask is honoured **above the backbone**: it is nearest-downsampled onto the feature grid and
applied as a key mask in encoder self-attention and decoder cross-attention. The convolutional
backbone is not masked, so padding still leaks into feature cells within one receptive field of
the boundary — the same behaviour as the reference implementation.

## 6. Component Reference & Configuration

### `create_detr(...)`

| Argument | Default | Meaning |
|:---|:---:|:---|
| `num_classes` | — | object classes, **excluding** "no object" |
| `num_queries` | — | max detections per image; hard ceiling |
| `backbone_name` | `"resnet50"` | the only supported value; anything else raises `NotImplementedError` |
| `backbone_trainable` | `False` | `True` fine-tunes the ImageNet backbone |
| `hidden_dim` | `256` | transformer width; **must be a multiple of 4** |
| `num_heads` | `8` | attention heads |
| `num_encoder_layers` / `num_decoder_layers` | `6` / `6` | encoder is the more expensive half |
| `ffn_dim` | `2048` | FFN hidden width |
| `dropout_rate` | `0.1` | |
| `aux_loss` | `True` | emit per-decoder-layer predictions |
| `activation` | `"relu"` | FFN activation |
| `normalization_type` | `"layer_norm"` | any normalization-factory key (`"rms_norm"`, ...) |
| `ffn_type` | `"mlp"` | any FFN-factory key (`"swiglu"`, `"geglu"`, ...) |

`normalization_type` and `ffn_type` are passed straight to `TransformerLayer`'s factories, so
swapping in RMSNorm or a gated FFN is a one-argument change.

### `DETR(num_classes, num_queries, backbone, transformer, hidden_dim=256, aux_loss=True)`

Use this directly to supply your own backbone. Any `keras.Model` mapping `(B, H, W, 3)` to a 4-D
feature map works; `input_proj` projects whatever channel count it emits to `hidden_dim`.

```python
import keras
import numpy as np
from dl_techniques.models.vision.detr import DETR, DetrTransformer

backbone = keras.Sequential([
    keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
    keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
], name="tiny_backbone")

transformer = DetrTransformer(
    hidden_dim=64, num_heads=4, num_encoder_layers=2, num_decoder_layers=2,
    ffn_dim=128, dropout_rate=0.1,
)

model = DETR(num_classes=10, num_queries=20, backbone=backbone,
             transformer=transformer, hidden_dim=64, aux_loss=True)

images = np.random.rand(2, 128, 128, 3).astype("float32")
mask = np.zeros((2, 128, 128), dtype=bool)
out = model([images, mask], training=False)
print(out["pred_logits"].shape, out["pred_boxes"].shape, len(out["aux_outputs"]))
# (2, 20, 11) (2, 20, 4) 1
```

### `DetrTransformer(hidden_dim, num_heads, num_encoder_layers, num_decoder_layers, ffn_dim, ...)`

The encoder-decoder stack, usable on its own. `call(src, key_keep_mask, query_embed, pos_embed)`
returns one tensor per decoder layer. `key_keep_mask` is a **keep** mask (1 = attend), which is
the inverse of the padding mask `DETR` accepts.

## 7. Deviations from the paper

Read this before comparing numbers with the reference implementation.

| Item | This implementation | Paper |
|:---|:---|:---|
| Set loss | **Not implemented.** No Hungarian matcher, no GIoU loss. The model is architecture-only. | Hungarian matching + classification/L1/GIoU set loss. |
| Padding mask | Honoured above the backbone; the CNN itself is not masked, so padding leaks into feature cells within one receptive field of the boundary. | Same — the reference also masks only the transformer. |
| Positional encoding | Added to the running `memory` at the input of **every** encoder layer, so it accumulates down the stack; the decoder adds `query_embed` to the whole decoder input the same way. | Re-injected into `Q` and `K` only, identically at every layer, never into `V`. |
| Backbone tap | `conv4_block6_out` (C4, stride 16, 1024 channels), frozen by default. | C5 (stride 32, 2048 channels), fine-tuned at a reduced learning rate. |
| Decoder output norm | None; auxiliary outputs are read raw from each decoder layer. | Final `LayerNorm` on the decoder stack. |
| Pretrained weights | ImageNet weights for the ResNet backbone only. Nothing above it is pretrained; there is no DETR checkpoint. | Full COCO-trained detector. |

The positional-encoding accumulation is a deliberate simplification, not a bug report — but it
does mean encoder layer *k* sees the encoding scaled roughly *k* times, which is not what the
paper computes.

No accuracy, AP or throughput number appears in this README. Nothing in this repository has ever
trained a DETR, so any such number would describe the paper, not this code.

## 8. Training

Training DETR needs a **custom loop or a custom `train_step`** that performs Hungarian matching,
because the loss is a set loss over an assignment. Neither ships here. The pieces you must supply:

1. A matcher (e.g. `scipy.optimize.linear_sum_assignment`) over the cost matrix
   `-p(class) + λ_L1 · ||b_pred − b_gt||₁ + λ_giou · (1 − GIoU)`.
2. A classification loss over all `num_queries` predictions, with the unmatched ones pushed to
   the "no object" class (the paper down-weights that class by 0.1).
3. Box L1 + GIoU losses over the matched pairs only.
4. The same loss applied to every entry of `aux_outputs`, summed.

The paper's recipe: AdamW, `1e-4` for the transformer and `1e-5` for the backbone, weight decay
`1e-4`, gradient clipping at `0.1`, 300 epochs with a 10x LR drop at epoch 200, batch 2 per GPU.
Quoted from the paper — not measured here.

Practical notes:

- Freeze the backbone (`backbone_trainable=False`, the default) for the first phase; unfreezing
  it at a 10x lower learning rate later is the standard schedule.
- Keep `aux_loss=True`. Convergence is much slower without it.
- The encoder is the expensive half — attention is quadratic in `H·W / 256` feature cells.
  Reducing `num_encoder_layers` or the input resolution buys more than anything else.
- Normalize images the way the backbone expects (ImageNet mean/std for `keras.applications`
  ResNet-50).

## 9. Serialization & Deployment

`DETR` and `DetrTransformer` register through
`@register_dl_technique("dl_techniques.models.detr.model")`, so a `.keras` file round-trips with
no `custom_objects`. `DETR.build` explicitly builds every sublayer, which is what makes the
weight restore work: Keras calls `load_own_variables()` on each sublayer, and an unbuilt sublayer
with no variables raises when the saved store has more entries than its (empty) variable list.

```python
import keras
import numpy as np

model.build([(None, 128, 128, 3), (None, 128, 128)])
model.save("detr.keras")
restored = keras.models.load_model("detr.keras")

a = np.asarray(model([images, mask], training=False)["pred_boxes"])
b = np.asarray(restored([images, mask], training=False)["pred_boxes"])
assert np.allclose(a, b, atol=1e-6)
```

For mixed precision, set `keras.mixed_precision.set_global_policy("mixed_float16")` before
constructing the model.

## 10. Troubleshooting

- **`ValueError: hidden_dim (10) must be a multiple of 4`.** The sine position encoder gets
  `hidden_dim // 2` features per axis and that must itself be even. The message names the next
  valid value.
- **`NotImplementedError: Backbone 'resnet101' not supported.`** `create_detr` only builds
  `"resnet50"`. For anything else, construct `DETR` directly with your own backbone (§ 6).
- **The forward pass raises on a bare image tensor.** `call` unpacks `(images, padding_mask)`.
  Pass `[images, mask]`, or `[images, None]` if there is no padding.
- **Confidences look wrong / exceed 1.** `pred_logits` are logits. Softmax first, and drop the
  last column, which is "no object".
- **Boxes are all near 0.5 and the model does not converge.** Expected without a set loss — see
  § 8. Also check that `aux_loss=True` and that you are training on all of `aux_outputs`.
- **Out of memory.** Encoder attention is quadratic in the number of feature cells. Halve the
  input resolution, cut `num_encoder_layers`, or drop the batch size.
- **More than `num_queries` objects in an image.** They cannot all be detected. Raise
  `num_queries`; the paper uses 100 for COCO.

Authoring conventions: [`models/CLAUDE.md`](../../CLAUDE.md). Mandatory guide for new models and
layers: `research/2026_keras_custom_models_instructions_v2.md`.

## 11. Citation

```bibtex
@inproceedings{carion2020detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and
          Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}, eprint={2005.12872},
  url={https://arxiv.org/abs/2005.12872}
}
```
