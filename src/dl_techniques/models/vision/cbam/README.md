# CBAMNet: A Convolutional Network with Attention

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **CBAMNet**, a convolutional network built from plain conv stages, each followed by a **Convolutional Block Attention Module (CBAM)**. CBAM is a lightweight module that refines a feature map by learning *what* to emphasize (channel attention) and then *where* to emphasize it (spatial attention).

---

## 1. Overview: What is CBAMNet and Why It Matters

### What is CBAMNet?

**CBAMNet** is a convolutional network whose every stage ends in a CBAM block. Rather than treating all channels and all pixels as equally informative, it infers two attention maps per stage — one along the channel axis, one along the spatial axes — and multiplies them into the features.

### Key Innovations

1.  **Factorized attention**: attention is split into a channel module and a spatial module, each cheap on its own, instead of one dense attention over the whole tensor.
2.  **Channel attention ("what")**: average- and max-pool over space give two channel descriptors; a shared bottleneck MLP maps both, the results are summed and passed through a sigmoid.
3.  **Spatial attention ("where")**: average- and max-pool along the channel axis give two 2-D maps; they are concatenated and passed through a single convolution and a sigmoid.
4.  **Sequential refinement**: channel first, then spatial. The paper reports this ordering beating both the reverse order and a parallel arrangement.

### Why it matters

A plain convolution applies the same filters everywhere and weighs every channel alike, so background pixels get the same treatment as the object and an edge detector counts as much as a texture detector. CBAM adds an input-dependent recalibration that costs very few parameters — the channel MLP is a bottleneck and the spatial branch is one convolution — and it is a drop-in that needs no change to the surrounding architecture.

---

## 2. The Problem CBAMNet Solves

A standard CNN learns filters but has no explicit mechanism to rank them at inference time. Two consequences follow:

| Uniformity | Consequence |
| :--- | :--- |
| Every spatial location is processed identically | Background gets as much capacity as the object of interest. |
| Every channel counts equally | Weakly informative channels dilute the strong ones downstream. |

CBAM makes the refinement explicit. For a feature map `F`:

```
F'  = F  * Mc(F)      # channel attention: what
F'' = F' * Ms(F')     # spatial attention: where
```

Both maps are functions of the input, so what the network emphasizes changes from image to image.

---

## 3. How CBAMNet Works: Core Concepts

Each stage is `Conv2D(dim, 3x3, ReLU) -> BatchNormalization -> CBAM(dim) -> MaxPooling2D(2x2)`, so
each stage halves the spatial resolution. The order is deliberate: attention operates on normalized
activations, and pooling then acts on features whose salience has already been accounted for.

```
Input (B, H, W, 3)
   │
   ├─► Stage 0: Conv2D(D0) -> BN -> CBAM -> MaxPool2x2     (B, H/2,  W/2,  D0)
   ├─► Stage 1: Conv2D(D1) -> BN -> CBAM -> MaxPool2x2     (B, H/4,  W/4,  D1)
   ├─► ...                                                  one stage per entry of `dims`
   │
   └─► Head (include_top): GlobalAveragePooling2D -> Dense(num_classes, softmax)
                                                            (B, num_classes)
```

Inside one CBAM block:

```
F ──► ChannelAttention ──► Mc ──► multiply ──► F' ──► SpatialAttention ──► Ms ──► multiply ──► F''
```

The head ends in **softmax**, so the model outputs probabilities, not logits — compile with
`from_logits=False`. With `include_top=False` the final stage's 4-D feature map is returned.

---

## 4. Architecture Deep Dive

### 4.1 `ChannelAttention`

Decides "what". Global average pooling and global max pooling produce two channel descriptors; a
**shared** MLP with a reduction/expansion bottleneck (`ratio`) maps each; the two outputs are added
and squashed by a sigmoid into per-channel weights.

### 4.2 `SpatialAttention`

Decides "where". Average and max pooling along the channel axis give two `(H, W, 1)` maps; they are
concatenated and passed through one `kernel_size x kernel_size` convolution (7x7 by default) and a
sigmoid, producing a single spatial weight map.

### 4.3 `CBAM`

Sequences the two: multiply `F` by the channel map, then multiply the result by the spatial map
computed from that result. Constructor arguments are `channels`, `ratio` (default 8),
`kernel_size` (default 7), plus separate initializers, regularizers and bias flags for the channel
and spatial branches (`channel_kernel_regularizer`, `spatial_kernel_regularizer`, ...).

---

## 5. Quick Start Guide

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

```python
import keras
import numpy as np

from dl_techniques.models.vision.cbam import CBAMNet

# 1. A tiny CBAMNet for CIFAR-10 (32x32 images, 10 classes)
model = CBAMNet.from_variant(
    "tiny",
    num_classes=10,
    input_shape=(32, 32, 3),
)

# 2. Compile. The head ends in softmax, so from_logits=False.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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

### 6.1 `CBAMNet` and its factory

Both names are exported from `dl_techniques.models.vision.cbam`:

| Name | Purpose |
| :--- | :--- |
| **`CBAMNet`** | The Keras `Model` subclass. `CBAMNet.from_variant(...)` builds a named variant. |
| **`create_cbam_net`** | Thin factory over `from_variant` with a default variant of `"tiny"`. |

```python
from dl_techniques.models.vision.cbam import CBAMNet, create_cbam_net

# Same model, two entry points.
model = CBAMNet.from_variant("base", num_classes=1000, input_shape=(224, 224, 3))
model = create_cbam_net("base", num_classes=1000, input_shape=(224, 224, 3))

# Or a fully custom stage ladder.
custom_model = CBAMNet(
    num_classes=100,
    dims=[32, 64, 128, 256],
    attention_ratio=16,
    input_shape=(64, 64, 3),
)
```

Constructor arguments: `num_classes`, `dims`, `attention_ratio` (default 8),
`attention_kernel_size` (default 7), `kernel_initializer`, `kernel_regularizer`, `include_top`,
`input_shape`. `from_variant` and `create_cbam_net` additionally take `pretrained` and
`weights_dataset`; `input_shape` defaults to `(224, 224, 3)` on both.

### 6.2 Core building blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`CBAM`** | `dl_techniques.layers.attention.convolutional_block_attention` | Channel attention then spatial attention. |
| **`ChannelAttention`** | `dl_techniques.layers.attention.channel_attention` | Channel weights ("what"). |
| **`SpatialAttention`** | `dl_techniques.layers.attention.spatial_attention` | Spatial weights ("where"). |

---

## 7. Configuration & Model Variants

`CBAMNet.MODEL_VARIANTS` sets only the stage widths; the number of stages is the length of `dims`,
and each stage halves the resolution.

| Variant | Dims | Stages | Total downsampling |
|:---|:---|:---:|:---:|
| **`tiny`** | `[64, 128]` | 2 | 4x |
| **`small`**| `[64, 128, 256]` | 3 | 8x |
| **`base`** | `[128, 256, 512]`| 3 | 8x |

Any other name raises `ValueError`. Widths are free-form through the constructor's `dims`.

---

## 8. Usage Examples

### Example 1: CBAMNet as a feature-extraction backbone

```python
import numpy as np

from dl_techniques.models.vision.cbam import CBAMNet

backbone = CBAMNet.from_variant(
    "base",
    include_top=False,
    input_shape=(224, 224, 3),
)

features = backbone(np.zeros((2, 224, 224, 3), dtype="float32"))

# Three stages of 2x pooling: 224 / 8 = 28, width = dims[-1] = 512
print(f"Output shape: {features.shape}")  # (2, 28, 28, 512)
```

### Example 2: Multi-scale features for an FPN

`CBAMNet` is a subclassed model, so it has no functional graph: `model.input` raises
`AttributeError` and you cannot slice it with `keras.Model(inputs, outputs)`. Walk `model.stages`
instead — it is a list of per-stage layer lists, in call order.

```python
import numpy as np

from dl_techniques.models.vision.cbam import CBAMNet

backbone = CBAMNet.from_variant(
    "small",
    include_top=False,
    input_shape=(256, 256, 3),
)
backbone(np.zeros((1, 256, 256, 3), dtype="float32"))  # build

x = np.zeros((1, 256, 256, 3), dtype="float32")
for stage_index, stage_layers in enumerate(backbone.stages):
    for layer in stage_layers:
        x = layer(x, training=False)
    print(f"stage {stage_index}: {tuple(x.shape)}")
# stage 0: (1, 128, 128, 64)
# stage 1: (1, 64, 64, 128)
# stage 2: (1, 32, 32, 256)
```

To capture the pre-pooling feature map of a stage instead, stop after the stage's `CBAM` layer —
each stage list is `[conv, bn, cbam, pool]`, and every layer carries a name such as
`stage_0_cbam`, reachable with `backbone.get_layer("stage_0_cbam")`.

---

## 9. Advanced Usage Patterns

### Regularizing the two attention branches separately

`CBAM` takes independent regularizers for the channel MLP and the spatial convolution, so you can
push the spatial map towards smoother solutions without over-constraining the channel weights.

```python
import keras

from dl_techniques.layers.attention.convolutional_block_attention import CBAM

custom_cbam = CBAM(
    channels=128,
    ratio=16,
    channel_kernel_regularizer=keras.regularizers.L2(1e-5),
    spatial_kernel_regularizer=keras.regularizers.L2(1e-4),  # stronger
)
```

`CBAMNet` itself exposes a single `kernel_regularizer`, which reaches its `Conv2D` and `Dense`
layers; per-branch control means building the blocks yourself.

---

## 10. Performance Optimization

```python
import keras

from dl_techniques.models.vision.cbam import CBAMNet

keras.mixed_precision.set_global_policy("mixed_float16")

model = CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
# model.fit() applies loss scaling automatically.
```

Set the policy before constructing the model, and restore `"float32"` afterwards. Note that the
softmax head runs in the compute dtype; if you need a float32 output layer, build the head yourself
with an explicit `dtype="float32"`.

---

## 11. Training and Best Practices

-   **Optimizer**: AdamW, with weight decay doing the regularization work.
-   **Schedule**: cosine decay works well for training these stacks from scratch.
-   **Augmentation**: flips, crops and colour jitter as a baseline; RandAugment or Mixup for harder datasets.
-   **`attention_ratio`**: the channel MLP bottleneck. Lower it (4, or 2) on small datasets to give the channel branch more capacity; raise it to save parameters.

---

## 12. Serialization & Deployment

`CBAMNet` and its custom layers round-trip through the `.keras` format; the classes are registered,
so no `custom_objects` argument is needed.

```python
import keras
import numpy as np

from dl_techniques.models.vision.cbam import CBAMNet

model = CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
model(np.zeros((1, 32, 32, 3), dtype="float32"))  # build before saving
model.save("my_cbam_model.keras")

loaded_model = keras.models.load_model("my_cbam_model.keras")
```

### Pretrained weights

None are distributed and none are downloadable: setting the `pretrained` argument to `True` raises
`NotImplementedError`. Pass a local checkpoint path instead, e.g.
`CBAMNet.from_variant("tiny", pretrained="/path/to/weights.keras")`.

---

## 13. Troubleshooting

-   **`model.input` raises `AttributeError`** — this is a subclassed model with no functional graph. See Example 2 for the multi-scale pattern.
-   **`count_params()` says the layer isn't built** — call the model on one batch first.
-   **Loss looks wrong / accuracy stuck at chance** — the head already applies softmax; compile with `from_logits=False`.
-   **`ValueError` on a variant name** — only `tiny`, `small`, `base` exist; use `dims` for anything else.
-   **`NotImplementedError` from the `pretrained` flag** — no trained weights ship with this package.
-   **Input too small** — each stage halves the resolution, so a 3-stage variant needs at least 8 pixels per side.

---

## 14. Technical Details

### Channel attention (`Mc`)

`Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )`

-   `AvgPool` and `MaxPool` reduce over the spatial axes, giving two channel descriptors.
-   `MLP` is shared between the two paths and has one hidden layer of `channels / ratio` units.
-   `σ` is the sigmoid.

### Spatial attention (`Ms`)

`Ms(F) = σ( f⁷ˣ⁷([AvgPool(F); MaxPool(F)]) )`

-   Here the pools reduce over the channel axis, giving two `(H, W, 1)` maps.
-   `[;]` is concatenation; `f⁷ˣ⁷` is one convolution with `kernel_size` 7 by default.
-   `σ` is the sigmoid.

Both maps broadcast against `F`, so the module adds no reshaping cost to the trunk.

---

## 15. Citation

-   **CBAM: Convolutional Block Attention Module**, arXiv:1807.06521:
    ```bibtex
    @inproceedings{woo2018cbam,
      title={CBAM: Convolutional block attention module},
      author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
      booktitle={Proceedings of the European conference on computer vision (ECCV)},
      pages={3--19},
      year={2018}
    }
    ```
