# ConvNeXt: A Modern ConvNet Architecture

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **ConvNeXt V1 and V2** architectures. ConvNeXt models are pure convolutional networks that were modernized to compete with Vision Transformers by progressively folding ViT design decisions into a standard ResNet.

The implementation covers both ConvNeXt V1 and ConvNeXt V2, which adds Global Response Normalization (GRN).

---

## 1. Overview: What is ConvNeXt and Why It Matters

### What is ConvNeXt?

**ConvNeXt** is a family of pure convolutional networks. The authors of the original paper walked a standard ResNet through the architectural differences that separate it from a Vision Transformer, one change at a time, and reported a ConvNet that matches or exceeds contemporary Transformers on ImageNet classification.

### Key Innovations

1.  **Modernized architecture (V1)**: a "patchify" stem (`4x4` conv, stride 4), large `7x7` depthwise kernels, an inverted bottleneck, LayerNorm instead of BatchNorm, and fewer activation and normalization layers overall.
2.  **Global Response Normalization (V2)**: one extra layer per block that encourages channel-wise feature competition. It is the only architectural difference between V1 and V2, and it pairs well with masked-autoencoder pre-training.
3.  **Simplicity**: no self-attention, so cost is linear in the number of pixels and the operators are the ones every accelerator already optimizes.

---

## 2. The Problem ConvNeXt Solves

Vision architectures have traded accuracy against compute:

| Family | Strength | Cost |
| :--- | :--- | :--- |
| CNNs | local operations and weight sharing make them cheap | historically weaker at long-range dependencies |
| ViTs | self-attention models global context directly | quadratic in the number of patches, so high-resolution input is expensive |

ConvNeXt argues that most of the Transformer's gain came from macro/micro design choices and training recipes rather than from self-attention itself. Applying those choices to a ConvNet gives a competitive model that keeps convolutional cost:

-   **Macro design**: ViT-like stage compute ratios and a patchify stem.
-   **Micro design**: `7x7` depthwise convolutions for a wide receptive field, an inverted bottleneck for capacity, LayerNorm and GELU for consistency with Transformer blocks.

---

## 3. How ConvNeXt Works: Core Concepts

ConvNeXt keeps the classic multi-stage hierarchy of a CNN, reducing resolution between stages.

```
Input (H, W, 3)
   │
   ├─► Stem: Conv2D SxS stride S, padding "valid" -> LayerNorm   (H/S,  W/S,  D0)
   │
   ├─► Stage 0: N0 x ConvNeXt block                              (H/S,  W/S,  D0)
   ├─► Downsample: LayerNorm -> Conv2D SxS stride S, "same"
   ├─► Stage 1: N1 x ConvNeXt block                              (H/S², W/S², D1)
   ├─► Downsample
   ├─► Stage 2: N2 x ConvNeXt block                              (H/S³, W/S³, D2)
   ├─► Downsample
   ├─► Stage 3: N3 x ConvNeXt block                              (H/S⁴, W/S⁴, D3)
   │
   └─► Head (include_top): GlobalAveragePooling -> LayerNorm -> Dense
                                                                 (B, num_classes)
```

`S` is the `strides` argument, default 4. Note the **divergence from the paper**: the inter-stage
downsamples here reuse the stem stride, so with the default `S = 4` a four-stage model reduces
resolution by 4⁴ = 256x in total, not the paper's 32x. A 512x512 input leaves a 2x2 final feature
map. Set `strides=2` for a stem-plus-2x-downsample layout closer to the reference.

The blocks are **transform-only**: a block returns `F(x)`, and the model owns the residual add and
the drop-path (`residual = x; x = block(x); x = drop_path(x); x = add([residual, x])`). With
`include_top=False` the head is dropped and the final stage feature map is returned. `depths` gives
the per-stage block counts and `dims` the widths; the number of stages follows the length of those
lists, so a two-stage variant such as `cifar10` is a legal configuration.

---

## 4. Architecture Deep Dive

### 4.1 Patchify stem

A single `Conv2D` with kernel size and stride both equal to `strides` (default 4) and padding
`"valid"` (`"same"` when `strides == 1`), followed by `LayerNormalization`. It splits the image into
non-overlapping patches and embeds them, reducing resolution 4x (224 -> 56).

### 4.2 `ConvNextV1Block`

The inverted bottleneck that carries the V1 network. Its `call` is, in order:

1.  **Depthwise `7x7` convolution** (`kernel_size`) — mixes spatial information and supplies the large receptive field.
2.  **LayerNorm** (or a bias-free BatchNorm, via the block's `normalization_type`).
3.  **Pointwise `1x1` convolution** expanding the channels 4x, then the activation (GELU by default).
4.  **Dropout and spatial dropout** (both no-ops at the default rate 0).
5.  **Pointwise `1x1` convolution** projecting back down.
6.  **Learnable per-channel `gamma` scaling** (`use_gamma`).

The residual add is not inside the block; the model applies it around the block's output.

### 4.3 `ConvNextV2Block` and Global Response Normalization

Identical to the V1 block plus a **GRN** layer inserted between the activation and the projecting
`1x1` convolution — that is, inside the widened 4x channel space, not at the end of the block. GRN
works in three steps:

1.  **Aggregate**: the L2 norm of each channel's feature map over the spatial dimensions, one scalar per channel.
2.  **Normalize**: divide each channel's aggregate by the mean of all aggregates, giving a relative-importance score.
3.  **Recalibrate**: rescale the features by those scores, apply the layer's own `gamma`/`beta`, and add the result to its input. Channels carrying distinctive responses are amplified and redundant ones damped, which is the feature competition the paper describes.

---

## 5. Quick Start Guide

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

Build a small ConvNeXt V2 for CIFAR-10-shaped input:

```python
import keras
import numpy as np

from dl_techniques.models.vision.convnext.convnext_v2 import create_convnext_v2

# 1. Create a tiny ConvNeXtV2 model for CIFAR-10 (32x32 images, 10 classes)
model = create_convnext_v2(
    variant="atto",  # a very small and fast V2 variant
    num_classes=10,
    input_shape=(32, 32, 3),
)

# 2. Compile
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

# 3. Dummy data
dummy_images = np.random.rand(16, 32, 32, 3).astype("float32")
dummy_labels = np.random.randint(0, 10, 16)

# 4. One training step
loss, acc = model.train_on_batch(dummy_images, dummy_labels)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 5. Inference
predictions = model.predict(dummy_images)
print(f"Predictions shape: {predictions.shape}")  # (16, 10)
```

---

## 6. Component Reference

### 6.1 Model classes and creation functions

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`ConvNeXtV1`** | `...convnext.convnext_v1.ConvNeXtV1` | Keras `Model` for the V1 architecture. |
| **`create_convnext_v1`** | `...convnext.convnext_v1.create_convnext_v1` | Convenience factory for `ConvNeXtV1`. |
| **`ConvNeXtV2`** | `...convnext.convnext_v2.ConvNeXtV2` | Keras `Model` for the V2 architecture. |
| **`create_convnext_v2`** | `...convnext.convnext_v2.create_convnext_v2` | Convenience factory for `ConvNeXtV2`. |

All four names are exported from `dl_techniques.models.vision.convnext`.

### 6.2 Core building blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`ConvNextV1Block`** | `dl_techniques.layers.convnext_v1_block` | The core block of the V1 architecture. |
| **`ConvNextV2Block`** | `dl_techniques.layers.convnext_v2_block` | The V1 block plus GRN. |

### 6.3 Factory signature

```python
create_convnext_v1(  # identical signature for create_convnext_v2
    variant="tiny",
    num_classes=1000,
    input_shape=(None, None, 3),
    pretrained=False,
    weights_dataset="imagenet",
    weights_input_shape=None,
    cache_dir=None,
    **kwargs,          # forwarded to the model, e.g. include_top, drop_path_rate
)
```

Constructor arguments reachable through `**kwargs` include `depths`, `dims`, `drop_path_rate`,
`stochastic_mode`, `kernel_size`, `activation`, `use_bias`, `kernel_regularizer`, `dropout_rate`,
`spatial_dropout_rate`, `strides`, `use_gamma`, `use_softorthonormal_regularizer` and
`include_top`.

---

## 7. Configuration & Model Variants

### ConvNeXt V1 variants (`ConvNeXtV1.MODEL_VARIANTS`)

| Variant | Depths | Dimensions |
|:---|:---|:---|
| **`cifar10`** | `[5, 5]` | `[96, 192]` |
| **`tiny`** | `[3, 3, 9, 3]` | `[96, 192, 384, 768]` |
| **`small`**| `[3, 3, 27, 3]` | `[96, 192, 384, 768]` |
| **`base`** | `[3, 3, 27, 3]` | `[128, 256, 512, 1024]` |
| **`large`**| `[3, 3, 27, 3]`| `[192, 384, 768, 1536]` |
| **`xlarge`**|`[3, 3, 27, 3]` | `[256, 512, 1024, 2048]` |

### ConvNeXt V2 variants (`ConvNeXtV2.MODEL_VARIANTS`)

| Variant | Depths | Dimensions |
|:---|:---|:---|
| **`cifar10`** | `[5, 5]` | `[96, 192]` |
| **`atto`** | `[2, 2, 6, 2]` | `[40, 80, 160, 320]` |
| **`femto`**| `[2, 2, 6, 2]` | `[48, 96, 192, 384]` |
| **`pico`** | `[2, 2, 6, 2]` | `[64, 128, 256, 512]` |
| **`nano`** | `[2, 2, 8, 2]` | `[80, 160, 320, 640]` |
| **`tiny`** | `[3, 3, 9, 3]` | `[96, 192, 384, 768]` |
| **`base`** | `[3, 3, 27, 3]` | `[128, 256, 512, 1024]` |
| **`large`**| `[3, 3, 27, 3]`| `[192, 384, 768, 1536]` |
| **`huge`** | `[3, 3, 27, 3]` | `[352, 704, 1408, 2816]`|

`cifar10` is a two-stage configuration for small images; it is not part of the paper's variant
family. `MODEL_VARIANTS` is the authoritative list — passing any other name raises `ValueError`.

---

## 8. Usage Examples

### Example 1: ConvNeXt as a feature-extraction backbone

```python
import numpy as np

from dl_techniques.models.vision.convnext.convnext_v1 import create_convnext_v1

# include_top=False drops the classification head
backbone = create_convnext_v1(
    variant="tiny",
    include_top=False,
    input_shape=(512, 512, 3),
)

features = backbone.predict(np.random.rand(1, 512, 512, 3).astype("float32"))

# Final-stage feature map: 512 / 4**4 = 2 spatial, width = dims[-1] = 768
print(f"Output shape: {features.shape}")  # (1, 2, 2, 768)
```

### Example 2: A micro variant with a custom class count

```python
import numpy as np

from dl_techniques.models.vision.convnext.convnext_v2 import create_convnext_v2

pico_model = create_convnext_v2(
    variant="pico",
    num_classes=100,
    input_shape=(96, 96, 3),
)

# The model is lazily built: run one forward pass before counting parameters,
# otherwise count_params() raises "the layer isn't built".
pico_model(np.zeros((1, 96, 96, 3), dtype="float32"))
print(f"Pico model params: {pico_model.count_params():,}")
```

---

## 9. Advanced Usage Patterns

### Fine-tuning from a local checkpoint

No pretrained weights are distributed with this package, and none are downloadable. Setting the
`pretrained` argument to `True` raises `NotImplementedError` rather than silently returning random
weights; pass a path to a local `.keras` checkpoint instead. The loader tolerates a different classifier width and a different
input resolution: mismatching weights are skipped and reported.

```python
from dl_techniques.models.vision.convnext.convnext_v2 import create_convnext_v2

fine_tune_model = create_convnext_v2(
    variant="base",
    num_classes=20,                 # may differ from the checkpoint
    input_shape=(128, 128, 3),      # may differ from the checkpoint
    pretrained="path/to/convnext_v2_base.keras",
)
```

`input_shape` may be left at its default `(None, None, 3)`; the pre-load build then materializes
weights at `PRETRAINED_BUILD_SPATIAL` (224) and the model stays fully convolutional.

---

## 10. Performance Optimization

ConvNeXt trains well under mixed precision, which is worth a substantial speedup on GPUs with
Tensor Cores.

```python
import keras

from dl_techniques.models.vision.convnext.convnext_v2 import create_convnext_v2

keras.mixed_precision.set_global_policy("mixed_float16")

model = create_convnext_v2("atto", num_classes=10, input_shape=(32, 32, 3))
# model.fit() applies loss scaling automatically.
```

Set the policy before constructing the model, and reset it with
`keras.mixed_precision.set_global_policy("float32")` when you are done.

---

## 11. Training and Best Practices

-   **Optimizer**: AdamW. Its decoupled weight decay is an important regularizer here.
-   **Schedule**: cosine decay with a short linear warmup.
-   **Stochastic depth**: `drop_path_rate` randomly drops residual branches during training. Start
    around `0.1`-`0.2` for the small variants and raise it for the large ones.
-   **Augmentation**: ConvNeXt has weaker inductive biases than older CNNs and benefits from strong
    augmentation (RandAugment, Mixup, CutMix). The paper's reported numbers depend on it.

---

## 12. Serialization & Deployment

`ConvNeXtV1`, `ConvNeXtV2` and their custom layers round-trip through the `.keras` format.

```python
import keras

from dl_techniques.models.vision.convnext.convnext_v1 import create_convnext_v1

model = create_convnext_v1("cifar10", num_classes=10, input_shape=(32, 32, 3))
model.save("my_convnext_model.keras")

loaded_model = keras.models.load_model("my_convnext_model.keras")
```

---

## 13. Testing & Validation

The package's own tests live in `tests/test_models/test_convnext/`. A minimal check that the small
variants build and produce the right output shape (`from_variant` is the classmethod the factories
call):

```python
import numpy as np

from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2

for variant in ("cifar10", "tiny"):
    assert ConvNeXtV1.from_variant(variant, num_classes=10, input_shape=(64, 64, 3))

model = ConvNeXtV2.from_variant("atto", num_classes=10, input_shape=(64, 64, 3))
output = model.predict(np.random.rand(4, 64, 64, 3).astype("float32"))
assert output.shape == (4, 10)
```

---

## 14. Troubleshooting

-   **`ValueError: Unknown variant`** — the name is not a key of `MODEL_VARIANTS`; see section 7.
-   **`NotImplementedError` from the `pretrained` flag** — no trained weights ship with this
    package. Pass a local checkpoint path instead.
-   **Loss goes to `NaN`** — lower the peak learning rate (`5e-4` or below) and add a warmup.
-   **Overfitting on a small dataset** — raise `weight_decay` (0.05 is a common value) and
    `drop_path_rate`.
-   **Feature map collapses to 1x1** — with the default `strides=4` every stage divides resolution
    by 4, so a four-stage model needs a 256-pixel side to keep any spatial extent. Use the
    two-stage `cifar10` variant, `strides=2`, or a larger input.

---

## 15. Technical Details

### Stochastic depth (drop path)

Per-block drop rates are `linear_drop_path_rates(sum(depths), drop_path_rate)`: linearly spaced
from 0.0 at the first block to `drop_path_rate` at the last, counted over the whole network rather
than per stage. The noise shape is `(B, 1, 1, 1)`, so an entire residual branch is dropped per
sample rather than individual pixels. `stochastic_mode="depth"` drops the branch outright;
`stochastic_mode="gradient"` keeps the forward value and stochastically drops the gradient.

### V1 block vs V2 block

-   **V1**: `DepthwiseConv -> Norm -> Expand 1x1 -> GELU -> Project 1x1 -> gamma`
-   **V2**: `DepthwiseConv -> Norm -> Expand 1x1 -> GELU -> GRN -> Project 1x1 -> gamma`

GRN acts on the expanded 4x-wide activations, so the channel competition happens where the block
has the most channels to compete over. The residual add wraps the block in both cases.

---

## 16. Citation

-   **ConvNeXt V1** — *A ConvNet for the 2020s*, arXiv:2201.03545:
    ```bibtex
    @article{liu2022convnet,
      title={A ConvNet for the 2020s},
      author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
      journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }
    ```
-   **ConvNeXt V2** — *Co-designing and Scaling ConvNets with Masked Autoencoders*, arXiv:2301.00808:
    ```bibtex
    @article{woo2023convnextv2,
      title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
      author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Dollar, Piotr and Xie, Saining},
      journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
    }
    ```
