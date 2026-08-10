# SuperPoint: Self-Supervised Interest-Point Detection and Description

A Keras 3 implementation of **SuperPoint** (DeTone, Malisiewicz & Rabinovich, 2018) with the
original VGG-style encoder replaced by a 3-stage **ConvNeXt V2** backbone. One forward pass
produces both a keypoint-detection heatmap and a full-resolution descriptor field from a
shared encoder and neck.

---

## 1. Overview

SuperPoint solves the interest-point problem end to end: instead of a hand-crafted detector
(FAST, Harris) followed by a separate descriptor (SIFT, ORB), a single convolutional network
emits both, so the two are trained jointly and are consistent by construction.

The detector predicts, for every non-overlapping `8x8` cell of the input, a distribution over
**65 classes** — the 64 pixel positions inside the cell plus one "dustbin" class meaning *no
keypoint here*. The descriptor head emits a semi-dense `H/8 x W/8` descriptor map that is
bicubically upsampled to full resolution and L2-normalized along the channel axis, so any
pixel has a unit-norm descriptor and cosine similarity is a plain dot product.

The detector head emits **raw logits** — the softmax lives in the loss, per this repository's
convention. Compile with a from-logits loss, or use `SuperPointDetectorLoss` (below), which
expects logits.

### Data flow

```
Input (B, H, W, C)
      │
      ▼
ConvNeXtV2(strides=2, include_top=False, 3 stages)
      │   stem /2 → stage-1 /4 → stage-2 /8
      ▼
feat (B, H/8, W/8, dims[-1])
      │
      ▼
proj  Conv2D 1x1 → (B, H/8, W/8, descriptor_dim)      [shared neck]
      ├────────────────────────────────┐
      ▼                                 ▼
detector_head Conv2D 1x1        descriptor_head Conv2D 1x1
→ (B, H/8, W/8, 65) LOGITS      → (B, H/8, W/8, descriptor_dim)
                                        │ resize bicubic → (H, W)
                                        │ L2-normalize (axis=-1)
                                        ▼
                                 (B, H, W, descriptor_dim)
```

`H` and `W` should be divisible by 8 so the semi-dense maps are exactly `H/8 x W/8`.

---

## 2. What this package contains

`src/dl_techniques/models/superpoint/` is a single module:

| File | Contents |
| :--- | :--- |
| `model.py` | `SuperPoint` (the `keras.Model`) and `create_superpoint` (factory). |
| `__init__.py` | Re-exports both under `__all__`, so `from dl_techniques.models.superpoint import SuperPoint` works. |

The rest of the SuperPoint surface lives in the packages that own each concern:

| Concern | Location |
| :--- | :--- |
| Losses | `dl_techniques.losses.superpoint_loss` — `SuperPointDetectorLoss`, `SuperPointDescriptorLoss` |
| Synthetic pre-training data | `dl_techniques.datasets.synthetic_shapes` |
| Homography sampling / warping | `dl_techniques.utils.homography` |
| Training pipelines | `src/train/superpoint/` — `train_magicpoint.py`, `train_superpoint.py`, `homographic_adaptation.py` |
| Tests | `tests/test_models/test_superpoint/test_model.py`, `tests/test_losses/test_superpoint_loss.py`, `tests/test_train/test_superpoint/` |

---

## 3. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`SuperPoint`** | `models.superpoint.model.SuperPoint` | The subclassed `keras.Model`: ConvNeXt V2 encoder, 1x1 neck, detector and descriptor heads. |
| **`SuperPoint.from_variant`** | `models.superpoint.model.SuperPoint.from_variant` | Classmethod building a named variant; forwards `**kwargs` to the constructor. |
| **`create_superpoint`** | `models.superpoint.model.create_superpoint` | Convenience factory: `create_superpoint(variant="base", input_shape=(256, 256, 1), **kwargs)`. |
| **`SuperPointDetectorLoss`** | `losses.superpoint_loss.SuperPointDetectorLoss` | Cross-entropy over the 65-class cell grid, on logits. |
| **`SuperPointDescriptorLoss`** | `losses.superpoint_loss.SuperPointDescriptorLoss` | Double-hinge correspondence loss over the coarse descriptor maps; the real entry point is `.compute(desc1, desc2, correspondence)`. |

### Constructor parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `depths` | `[3, 3, 9]` | ConvNeXt V2 blocks per stage (3 stages). Must be the same length as `dims`. |
| `dims` | `[96, 192, 384]` | Channel width per stage. |
| `input_shape` | `(256, 256, 1)` | `(height, width, channels)`; grayscale by default. |
| `descriptor_dim` | `256` | Descriptor channels, and the width of the shared 1x1 neck. |
| `drop_path_rate` | `0.0` | Stochastic depth, forwarded to the encoder. |
| `kernel_size` | `7` | ConvNeXt V2 block kernel size. |
| `activation` | `"gelu"` | ConvNeXt V2 block activation. |
| `use_bias` | `True` | Bias on encoder and head convolutions. |
| `kernel_regularizer` | `None` | Applied to encoder and head kernels. |

### Variants (`SuperPoint.MODEL_VARIANTS`)

| Variant | `depths` | `dims` |
| :--- | :--- | :--- |
| `"tiny"` | `[3, 3, 9]` | `[96, 192, 384]` |
| `"base"` | `[3, 3, 27]` | `[128, 256, 512]` |
| `"large"` | `[3, 3, 27]` | `[192, 384, 768]` |

---

## 4. Quick Start

```python
import keras
from dl_techniques.models.superpoint import create_superpoint

model = create_superpoint(variant="tiny", input_shape=(256, 256, 1))

out = model(keras.ops.zeros((1, 256, 256, 1)))
print(out["keypoints"].shape)    # (1, 32, 32, 65)  -- raw logits
print(out["descriptors"].shape)  # (1, 256, 256, 256) -- unit-L2 along axis -1
```

`call` returns a dict, so `compute_output_shape` returns a dict of shapes too.

The detector head composes with `model.compile` in the ordinary way — `SuperPointDetectorLoss`
is a plain `keras.losses.Loss` over `(labels, logits)`. The **descriptor** loss does not: its
real objective needs two descriptor maps plus a homography-derived correspondence tensor,
which does not fit the `(y_true, y_pred)` signature, so it is exposed as
`SuperPointDescriptorLoss.compute(desc1, desc2, correspondence)`. Its `call` is only a
convenience adapter that assumes an identity correspondence, and it is defined on the
**coarse** `H/8` descriptor map, not on the full-resolution field this model returns. See
`src/train/superpoint/train_superpoint.py` for the real two-view training loop.

---

## 5. Serialization

`SuperPoint` is registered with `@keras.saving.register_keras_serializable()` and implements
`get_config` / `from_config`, so it round-trips through the `.keras` format:

```python
model.save("superpoint.keras")
restored = keras.models.load_model("superpoint.keras")
```

`get_config` emits every constructor parameter, with `kernel_regularizer` passed through
`keras.regularizers.serialize`. `from_config` is overridden solely to run the matching
`keras.regularizers.deserialize` before calling the constructor.

---

## 6. References

- DeTone, Malisiewicz, Rabinovich. *SuperPoint: Self-Supervised Interest Point Detection and
  Description.* CVPRW 2018. https://arxiv.org/abs/1712.07629
- Woo et al. *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.*
  CVPR 2023. https://arxiv.org/abs/2301.00808
