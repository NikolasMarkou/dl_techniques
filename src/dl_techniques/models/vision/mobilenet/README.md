# MobileNet Family (V1-V4): Efficient Vision Models

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **MobileNet** family, V1 through V4 — a lineage of efficient convolutional networks designed for on-device vision, where compute, power and memory are all scarce.

V2, V3 and V4 are built on a shared `UniversalInvertedBottleneck` layer, which shows how one configurable block covers a wide range of modern architectures.

---

## 1. Overview: What is MobileNet and Why It Matters

### What is the MobileNet Family?

**MobileNet** is a class of convolutional networks from Google aimed at mobile and edge devices: the best accuracy attainable under strict latency, power and memory budgets. Each version adds an architectural idea that moves that trade-off.

### The Evolution of Efficiency

| Model | Key Innovation | Description |
| :--- | :--- | :--- |
| **MobileNetV1** | Depthwise separable convolutions | Factorizes a standard convolution into a depthwise and a pointwise convolution, cutting computation by roughly 8-9x. |
| **MobileNetV2** | Inverted residuals and linear bottlenecks | Expands, processes, then projects back down, with residual connections between the narrow bottlenecks. |
| **MobileNetV3** | Hardware-aware NAS, squeeze-and-excite, hard-swish | Architecture found by search, with lightweight channel attention and a cheaper non-linearity. |
| **MobileNetV4** | Universal Inverted Bottleneck (UIB) and Mobile MQA | One flexible block that can act as several block styles, plus optional multi-query attention in the late stages. |

### Why it matters

Running the model on the device instead of a server removes the round trip: it works offline, adds no network latency, keeps the image on the phone, and costs nothing to serve. That is only possible if inference fits inside a frame budget on a phone CPU or NPU, which is the constraint every MobileNet version is designed against.

---

## 2. The Problem MobileNet Solves

Large models reach high accuracy at a cost that edge devices cannot pay. Three constraints bind at once:

| Constraint | What it means in practice |
| :--- | :--- |
| **Latency** | Real-time video work must finish within a frame, roughly 33 ms at 30 fps. |
| **Power** | Heavy compute drains the battery and thermally throttles the device. |
| **Size** | The model must fit in storage and in the RAM available at inference time. |

MobileNet answers this by redesigning the building block itself rather than by pruning a large network after the fact.

---

## 3. How MobileNet Works: Core Concepts & Evolution

**1. Depthwise separable convolutions (V1).** A standard convolution is split into a *depthwise* convolution (one filter per input channel, mixing space but not channels) and a *pointwise* `1x1` convolution (mixing channels). The factorization is the foundation of the whole family.

**2. Inverted residuals and linear bottlenecks (V2).** The block runs `narrow -> wide -> narrow` rather than ResNet's `wide -> narrow -> wide`. The final projection is **linear** — no activation — because a non-linearity in a low-dimensional bottleneck destroys information. Residual connections join the narrow ends when the stride is 1.

**3. NAS, squeeze-and-excite and h-swish (V3).** The layout was searched rather than hand-designed, targeting measured mobile-CPU latency. It adds squeeze-and-excite channel attention and the cheaper hard-swish activation, and trims the final stage.

**4. Universal Inverted Bottleneck and Mobile MQA (V4).** One configurable block subsumes the classic inverted residual, a ConvNeXt-like ordering, a plain FFN and a two-depthwise "ExtraDW" form, which widens the search space. The hybrid variants add mobile multi-query attention to the late stages.

---

## 4. Architecture Deep Dive

### 4.1 `MobileNetV1`
- **Stem**: one standard `3x3` convolution.
- **Body**: 13 `DepthwiseSeparableBlock` layers (`MobileNetV1.ARCHITECTURE`), some with stride 2.
- **Head**: global average pooling, dropout, then a dense classifier.

### 4.2 `MobileNetV2`
- **Stem**: one standard `3x3` convolution.
- **Body**: 17 inverted residual blocks in 7 stages, from the paper's `(t, c, n, s)` table. The first stage has expansion factor `t=1`. Residual connections apply to stride-1 blocks whose input and output widths match.
- **Head**: a `1x1` convolution expanding to 1280 channels, then pooling, dropout and a dense classifier.

### 4.3 `MobileNetV3`
- **Stem**: an efficiency-tuned `3x3` convolution.
- **Body**: NAS-derived inverted residual blocks — `5x5` kernels in some layers, squeeze-and-excite in some, and ReLU or hard-swish depending on depth.
- **Head**: the paper's "efficient last stage", which moves work before the final pooling.

### 4.4 `MobileNetV4`
- **Stem**: a `3x3` convolution.
- **Body**: seven stages of UIB blocks; the `block_types` entry for each stage selects the block shape. Hybrid variants insert `MobileMQA` attention at `attention_stages` (5 and 6 by default).
- **Head**: global average pooling, then `Dense(1280) -> ReLU -> Dropout -> Dense(num_classes, softmax)`.

---

## 5. Quick Start Guide

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

```python
import keras
import numpy as np

from dl_techniques.models.vision.mobilenet.mobilenet_v4 import create_mobilenetv4

# 1. A small MobileNetV4 for CIFAR-10 (32x32 images, 10 classes)
model = create_mobilenetv4(
    variant="small",
    num_classes=10,
    input_shape=(32, 32, 3),
)

# 2. Compile. The V4 head ends in softmax, so from_logits=False.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
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

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`MobileNetV1`** | `...mobilenet.mobilenet_v1.MobileNetV1` | Keras `Model` for V1. |
| **`create_mobilenetv1`** | `...mobilenet.mobilenet_v1.create_mobilenetv1` | Factory for `MobileNetV1`. |
| **`MobileNetV2`** | `...mobilenet.mobilenet_v2.MobileNetV2` | Keras `Model` for V2. |
| **`create_mobilenetv2`** | `...mobilenet.mobilenet_v2.create_mobilenetv2` | Factory for `MobileNetV2`. |
| **`MobileNetV3`** | `...mobilenet.mobilenet_v3.MobileNetV3` | Keras `Model` for V3. |
| **`create_mobilenetv3`** | `...mobilenet.mobilenet_v3.create_mobilenetv3` | Factory for `MobileNetV3`. |
| **`MobileNetV4`** | `...mobilenet.mobilenet_v4.MobileNetV4` | Keras `Model` for V4. |
| **`create_mobilenetv4`** | `...mobilenet.mobilenet_v4.create_mobilenetv4` | Factory for `MobileNetV4`. |

All four factories share the same signature shape:

```python
create_mobilenetv4(          # same for v1 / v2 / v3
    variant="medium",        # default variant differs per version: see section 7
    num_classes=1000,
    input_shape=None,        # V4 defaults to (224, 224, 3)
    width_multiplier=1.0,
    pretrained=False,        # True raises NotImplementedError
    **kwargs,                # forwarded to the model, e.g. include_top, dropout_rate
)
```

`kwargs` reaches the constructor arguments `dropout_rate`, `weight_decay`,
`kernel_initializer` and `include_top` on every version, plus `depths`, `dims`,
`block_types`, `strides`, `use_attention` and `attention_stages` on V4.

---

## 7. Configuration & Model Variants

The variant key is a **name**, not a width string. Each entry of `MODEL_VARIANTS` is the config the
factory applies; `width_multiplier` passed alongside a variant overrides the variant's own value.

| Version | Variants (`MODEL_VARIANTS` keys) | What the key sets | Factory default |
| :--- | :--- | :--- | :--- |
| **V1** | `large` (α=1.0), `medium` (0.75), `small` (0.5), `pico` (0.25) | `width_multiplier` | `large` |
| **V2** | `large` (α=1.4), `medium` (1.0), `small` (0.75), `nano` (0.5), `pico` (0.35) | `width_multiplier` | `medium` |
| **V3** | `large`, `small` | the NAS block table | `large` |
| **V4** | `small`, `medium`, `large`, `hybrid_medium`, `hybrid_large` | depths, dims, block types, attention | `medium` |

Anything else raises `ValueError`. In particular there is no `"conv_small"`, `"conv_medium"` or
`"conv_large"` on V4, and no `"1.0"` / `"0.5"` style key on V1 or V2 — use `width_multiplier` for
that. (The V4 names are deliberately not the paper's MNv4-Conv-S/M/L: these ladders are
hand-written depth/width tables, not the NAS-found specifications — see `mobilenet_v4.py`'s module
docstring.)

---

## 8. Usage Examples

### Example 1: A model for a custom dataset (CIFAR-100)

```python
from dl_techniques.models.vision.mobilenet.mobilenet_v3 import create_mobilenetv3

cifar100_model = create_mobilenetv3(
    variant="small",
    num_classes=100,
    input_shape=(32, 32, 3),
)
cifar100_model.summary()
```

### Example 2: Scaling with the width multiplier

`width_multiplier` (α) scales the channel count of every layer, trading accuracy for size and
latency. Halving it cuts parameters by roughly 4x.

```python
import numpy as np

from dl_techniques.models.vision.mobilenet.mobilenet_v2 import create_mobilenetv2

default_v2 = create_mobilenetv2(variant="medium", input_shape=(224, 224, 3))   # alpha = 1.0
nano_v2 = create_mobilenetv2(variant="nano", input_shape=(224, 224, 3))        # alpha = 0.5
custom_v2 = create_mobilenetv2(                                                # same, explicitly
    variant="medium", width_multiplier=0.5, input_shape=(224, 224, 3)
)

# Models are lazily built; one forward pass before count_params().
for model in (default_v2, nano_v2, custom_v2):
    model(np.zeros((1, 224, 224, 3), dtype="float32"))

print(f"medium (alpha=1.0):  {default_v2.count_params():,}")  # 3,540,136
print(f"nano   (alpha=0.5):  {nano_v2.count_params():,}")     # 1,987,544
print(f"medium x 0.5:        {custom_v2.count_params():,}")   # 1,987,544
```

### Example 3: MobileNet as a feature extractor

```python
import numpy as np

from dl_techniques.models.vision.mobilenet.mobilenet_v4 import create_mobilenetv4

backbone = create_mobilenetv4(
    variant="hybrid_medium",
    include_top=False,
    input_shape=(256, 256, 3),
)

features = backbone(np.zeros((1, 256, 256, 3), dtype="float32"))
print(f"Output shape: {features.shape}")  # (1, 8, 8, 320)
```

All four versions share this contract: `include_top=False` returns the **4-D feature map**, never a
pooled vector. If you want the pooled vector, add a `keras.layers.GlobalAveragePooling2D()`
yourself.

`summary()` on a never-called model runs a real dummy forward pass to materialize the weights;
`keras.Model.build(...)` alone does not, since on a subclassed model it only marks the model built.

---

## 9. Advanced Usage Patterns

### Fine-tuning from your own checkpoint

No pretrained weights ship with this package and nothing is downloadable: `pretrained=True` raises
`NotImplementedError` on all four factories. To fine-tune, train a model yourself, save it, then
reuse the backbone:

```python
import keras
import numpy as np

from dl_techniques.models.vision.mobilenet.mobilenet_v4 import create_mobilenetv4

source = create_mobilenetv4(variant="small", num_classes=10, input_shape=(64, 64, 3))
source(np.zeros((1, 64, 64, 3), dtype="float32"))  # build before saving
source.save("mobilenet_v4_small.keras")

# Later: reload and put a fresh head on the headless backbone.
backbone = create_mobilenetv4(
    variant="small", include_top=False, input_shape=(64, 64, 3)
)
backbone(np.zeros((1, 64, 64, 3), dtype="float32"))
restored = keras.models.load_model("mobilenet_v4_small.keras")
backbone.set_weights(restored.get_weights()[: len(backbone.get_weights())])
backbone.trainable = False

inputs = keras.Input(shape=(64, 64, 3))
x = backbone(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(20, activation="softmax")(x)
fine_tune_model = keras.Model(inputs, outputs)
```

The weight-slice copy above works only because the headless model is a strict prefix of the full
one; check `len(backbone.get_weights())` against your own checkpoint before relying on it.

---

## 10. Performance Optimization

MobileNets train well under mixed precision on GPUs with Tensor Cores.

```python
import keras

from dl_techniques.models.vision.mobilenet.mobilenet_v4 import create_mobilenetv4

keras.mixed_precision.set_global_policy("mixed_float16")

model = create_mobilenetv4("small", num_classes=10, input_shape=(32, 32, 3))
# model.fit() applies loss scaling automatically.
```

Set the policy before the model is constructed, and restore it with
`keras.mixed_precision.set_global_policy("float32")` afterwards.

Depthwise convolutions are memory-bandwidth bound rather than FLOP bound, so a MobileNet often
does not speed up on a big GPU the way its FLOP count suggests. The gain shows on the mobile
hardware it targets.

---

## 11. Training and Best Practices

-   **Optimizer**: AdamW is a solid default; the original papers used RMSprop with tuned decay and momentum.
-   **Weight decay**: MobileNets are sensitive to it. The constructor defaults (`4e-5` for V1/V2, `1e-5` for V3/V4) come from the papers and target ImageNet — reduce them for small datasets.
-   **Schedule**: cosine or exponential decay beats a fixed learning rate.
-   **Augmentation**: RandAugment, Mixup and CutMix help when training from scratch, though a small model needs less augmentation than a large one.

---

## 12. Serialization & Deployment

All four model classes and their custom layers round-trip through the `.keras` format.

```python
import keras
import numpy as np

from dl_techniques.models.vision.mobilenet.mobilenet_v3 import create_mobilenetv3

model = create_mobilenetv3("small", num_classes=10, input_shape=(32, 32, 3))
model(np.zeros((1, 32, 32, 3), dtype="float32"))  # build before saving
model.save("my_mobilenetv3_model.keras")

loaded_model = keras.models.load_model("my_mobilenetv3_model.keras")
```

---

## 13. Testing & Validation

The package's tests live in `tests/test_models/test_mobilenet/`, and
`tests/test_models/test_readme_examples.py` pins the V4 variant names this README documents. A
minimal self-check:

```python
import numpy as np

from dl_techniques.models.vision.mobilenet.mobilenet_v1 import MobileNetV1
from dl_techniques.models.vision.mobilenet.mobilenet_v2 import MobileNetV2
from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3
from dl_techniques.models.vision.mobilenet.mobilenet_v4 import MobileNetV4

for cls in (MobileNetV1, MobileNetV2, MobileNetV3, MobileNetV4):
    for variant in cls.MODEL_VARIANTS:
        model = cls.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))
        assert model is not None

model = MobileNetV4.from_variant("small", num_classes=10, input_shape=(96, 96, 3))
output = model.predict(np.random.rand(4, 96, 96, 3).astype("float32"))
assert output.shape == (4, 10)
```

---

## 14. Troubleshooting

-   **`ValueError` on a variant name** — see section 7; V1/V2 keys are names, not width strings, and V4 has no `conv_*` keys.
-   **`count_params()` raises "the layer isn't built"** — these are subclassed models; call the model on one batch first.
-   **`summary()` shows 0 parameters** — same cause, on an older build. Run a forward pass first.
-   **A downstream head gets the wrong rank** — `include_top=False` returns a 4-D feature map on all four versions; pool it yourself.
-   **Accuracy stalls** — lower the learning rate (start near `1e-4`), check input normalization, and reduce the ImageNet-tuned weight decay for a small dataset.
-   **`pretrained=True` raises `NotImplementedError`** — no trained weights ship with this package.

---

## 15. Technical Details: The Evolution of the Block

-   **V1 (depthwise separable)**:
    `Input -> 3x3 DWConv -> BN -> ReLU -> 1x1 PWConv -> BN -> ReLU`

-   **V2 (inverted residual)**:
    `Input -> 1x1 Expand -> BN -> ReLU6 -> 3x3 DWConv -> BN -> ReLU6 -> 1x1 Project (linear) -> BN -> Add`

-   **V3 (V2 + SE + h-swish)**:
    `... -> 1x1 Expand -> ... -> DWConv (3x3 or 5x5) -> ... -> Squeeze-Excite -> 1x1 Project -> ...`

-   **V4 (Universal Inverted Bottleneck)**: the UIB has an optional depthwise convolution on
    *either side* of the expansion, and which positions are occupied names the block:

    | `block_type` | start DW (pre-expansion) | middle DW (post-expansion) |
    |---|---|---|
    | `"FFN"`      | –       | –   |
    | `"IB"`       | –       | 3x3 |
    | `"ConvNext"` | 7x7     | –   |
    | `"ExtraDW"`  | 3x3     | 3x3 |

    So `"ExtraDW"` is
    `Input -> 3x3 DWConv -> BN -> Act -> 1x1 Expand -> BN -> Act -> 3x3 DWConv -> BN -> Act -> 1x1 Project -> ...`

    **The position matters, not the count.** `"IB"` and `"ConvNext"` each own exactly one depthwise
    convolution and are different architectures: the start DW mixes space at the *unexpanded*
    channel count, before the block goes wide, which is the ConvNeXt ordering and the reason that
    position conventionally uses a larger kernel. Setting `use_dw1=True, use_dw2=True` on the layer
    stacks two middle depthwise convolutions instead — a shape the layer supports, but not the
    paper's ExtraDW.

---

## 16. Citation

-   **MobileNetV1** — *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, arXiv:1704.04861:
    ```bibtex
    @article{howard2017mobilenets,
      title={MobileNets: Efficient convolutional neural networks for mobile vision applications},
      author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
      journal={arXiv preprint arXiv:1704.04861},
      year={2017}
    }
    ```
-   **MobileNetV2** — *Inverted Residuals and Linear Bottlenecks*, arXiv:1801.04381:
    ```bibtex
    @inproceedings{sandler2018mobilenetv2,
      title={MobileNetV2: Inverted residuals and linear bottlenecks},
      author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={4510--4520},
      year={2018}
    }
    ```
-   **MobileNetV3** — *Searching for MobileNetV3*, arXiv:1905.02244:
    ```bibtex
    @inproceedings{howard2019searching,
      title={Searching for MobileNetV3},
      author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V and Adam, Hartwig},
      booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
      pages={1314--1324},
      year={2019}
    }
    ```
-   **MobileNetV4** — *MobileNetV4: Universal Models for the Mobile Ecosystem*, arXiv:2404.10518:
    ```bibtex
    @article{qin2024mobilenetv4,
      title={MobileNetV4: Universal Models for the Mobile Ecosystem},
      author={Qin, Danfeng and Leichner, Chas and Delakis, Manolis and Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and Banbury, Colby and Ye, Chengxi and Akin, Berkin and Aggarwal, Vaibhav and Zhu, Tenghui and Moro, Daniele and Howard, Andrew},
      journal={arXiv preprint arXiv:2404.10518},
      year={2024}
    }
    ```
