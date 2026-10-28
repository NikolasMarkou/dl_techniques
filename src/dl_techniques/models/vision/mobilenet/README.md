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
- **Stem**: one `ConvStem` named `stem`: a standard `3x3` stride-2 convolution without bias, BatchNorm at the family's reference epsilon `1e-3`, ReLU. Its width is `int(32 * width_multiplier)`; V1 truncates, V2 to V4 round to a multiple of 8.
- **Body**: 13 `DepthwiseSeparableBlock` layers (`MobileNetV1.ARCHITECTURE`), some with stride 2, named `block_{i}` with `i` counted from 0 (the block's own layers are `conv_dw_{i}`, `conv_pw_{i}` and their `_norm` layers).
- **Head**: the paper's classifier (Table 1, "Avg Pool, FC, Softmax") as Keras Applications builds it: global average pooling, `Dropout(dropout_rate)` (1e-3), one linear `Dense(num_classes)` (`block_dims=()`, `classifier_dropout_rate`).

### 4.2 `MobileNetV2`
- **Stem**: one `ConvStem` named `stem`: a standard `3x3` stride-2 convolution without bias, BatchNorm at the reference epsilon, ReLU6 (`ReLU(max_value=6)`), `make_divisible(32 * width_multiplier)` channels.
- **Body**: 17 inverted residual blocks named `block_{i}` (`i` from 0) in 7 stages, from the paper's `(t, c, n, s)` table. The first stage has expansion factor `t=1` and, as in the paper, Keras Applications and torchvision, no expansion conv, so `count_params()` equals `keras.applications.MobileNetV2(weights=None)` at each variant's `alpha` (until 2026-10-21 the universal block always expanded: `C*C + 4*C` extra, 1,152 at `alpha=1.0`; archives saved before then do not load). Residual connections apply to stride-1 blocks whose input and output widths match.
- **Head**: a `1x1` convolution expanding to 1280 channels (backbone; the raw layers `conv_last`, `conv_last_bn` and `conv_last_relu6`, because only the input stem is a `ConvStem`), then the torchvision classifier: global average pooling, `Dropout(dropout_rate)` (0.2), one linear `Dense(num_classes)`.

### 4.3 `MobileNetV3`
- **One reference at every width**: Keras Applications `MobileNetV3`. `count_params()` equals a live `keras.applications.MobileNetV3Small` / `MobileNetV3Large` (`weights=None`, `minimalistic=False`) at every `width_multiplier`, with and without the top, and so does every weight shape (`test_the_v3_widths_follow_keras_applications_at_every_width.py`, widths 0.25 to 1.25).
- **Stem**: a `3x3` stride-2 convolution of 16 channels at every width (Keras Applications does not scale the stem).
- **Body**: NAS-derived inverted residual blocks — `5x5` kernels in some layers, squeeze-and-excite in some, and ReLU or hard-swish depending on depth. A block's expanded width is `make_divisible(C_in * exp / in)`: its actual input width times the table's expansion ratio (Keras Applications' `_depth(infilters * expansion)`), not torchvision's `make_divisible(exp * width_multiplier)`; the two agree only at 1.0. The first row (expansion 1) has no expansion conv. The last conv is `make_divisible(6 * C_last)`. Until 2026-10-21 the port used torchvision's widths and scaled the stem, and built a first-row expansion conv: Keras Applications plus 320 parameters at 1.0, 65,584 (Large) and 21,264 (Small) short at 0.75. Archives saved before then do not load.
- **Squeeze-and-excite**: the shared `SqueezeExcitation` with `divisor=8`, biased 1x1 convs and a hard-sigmoid gate (`relu6(z + 3) / 6`), as Keras Applications' `_se_block`, torchvision's `SqueezeExcitation` and the paper (Section 5.2) build it; until 2026-10-20 the convs were bias-free and the gate a plain sigmoid. The bottleneck is `make_divisible(expanded / 4, 8)`, Keras Applications' `_depth(filters * 0.25)` and torchvision's `_make_divisible(expanded // 4, 8)`, which agree because every expanded width is a multiple of 8. Until 2026-10-20 it was `int(expanded * 0.25)`: 2 channels on Small's first SE block at `width_multiplier=0.25`, whose ReLU died on about one draw in eight. Archives saved before then do not load (the SE weights moved to `block_i/se/conv_reduce` and `conv_restore` and changed width).
- **Head**: the paper's "efficient last stage", which pools before the wide projection: global average pooling, `Dense(1280)` (Large) or `Dense(1024)` (Small) with no normalization (the paper's "NBN"), hard-swish, `Dropout(dropout_rate)`, one linear `Dense(num_classes)`, as Keras Applications builds it. The projection widens only for `width_multiplier > 1`.

### 4.4 `MobileNetV4`
- **Stem**: a `3x3` convolution.
- **Body**: seven stages of UIB blocks; the `block_types` entry for each stage selects the block shape. Hybrid variants insert `MobileMQA` attention at `attention_stages` (5 and 6 by default). The layers are named `stem`, `stage{S}_block{B}` and `stage{S}_mqa` (S and B count from 1, as in ResNet) and `classification_head`.
- **Head**: the paper's classifier as timm builds it (`head_norm=True`): global average pooling, `Dense(1280) -> BatchNorm -> ReLU -> Dropout -> Dense(num_classes)`, the BatchNorm at the family's reference epsilon 1e-3.

Every version returns **logits** (the classifier is linear): compile with `from_logits=True` and apply `keras.ops.softmax` to read probabilities. The model's `kernel_initializer` and its `kernel_regularizer` reach the head's `Dense` layers. `kernel_regularizer` is a regularizer (an object, its serialized config or a name) and is `None` by default: a default model has no regularization loss, and the optimizer's decoupled `weight_decay` is the only decay. Every conv and `Dense` draws from its own clone of `kernel_initializer`, so a name, a dict or a seedless instance gives every kernel its own draw (the stem, every block's convs and the classifier were tied before 2026-10-16). A seeded `Initializer` instance, or a dict with an explicit `seed`, reuses its seed at every site by contract and ties same-shaped kernels (`HeNormal(seed=7)` ties 903 of 1431 kernel pairs of V3 `small` and 351 of 861 of V4 `small`). V4 stores the serialized spec and hands it to each site; V1, V2 and V3 keep the instance and hand each site a clone, which leaves their seeded kernels where they were (a stored spec would move every one of them: 0 of 28 kernels of V1 `small` and 0 of 53 of V2 `small` stay equal to the previous revision). `HeNormal(seed=7)` ties 378 of 378 kernel pairs of V1 `small` and 630 of 1378 of V2 `small`. The `return_dict` and `head_config_preset` arguments are gone; `head_config_overrides` still reshapes the head, and `name`, `num_classes` and `task_type` in it raise `ValueError`. Archives saved before 2026-10-14 (a dict output, a softmax head, or the two-`DenseBlock` head) do not load and are not supported.

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

# 2. Compile. The model returns logits, so from_logits=True.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
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

All four versions take the house signature; `pretrained` is a path to a local `.keras` checkpoint, `True`
(raises `NotImplementedError`) or `False`. A path loads every layer outside `classification_head` or
raises `ValueError` and copies no weight (a checkpoint of another variant is refused; only a
head of another `num_classes` is skipped, with a warning). Loading with another `num_classes` discards the
whole `classification_head`, so the head's pretrained hidden layer is skipped with the classifier and keeps
its initialization (V1 and V2 have no hidden layer, so only the classifier is skipped; measured parameter counts of the hidden layer: V3 `small` 590,848, V3 `large`
1,230,080, V4 `small` 209,920, V4 `medium` and `hybrid_medium` 414,720; V4's includes its BatchNorm).
The checkpoint's own configuration is compared too: when the archive records the same class, every
constructor argument that both the archive and the model hold, except the ones that leave the weight
layout and the computed function alone (`num_classes`, `include_top`,
initializers, regularizers, dropout and stochastic-depth rates), must be equal. A V4 checkpoint into
the same variant under other `strides`, or a V1, V2 or V3 checkpoint under another `width_multiplier`, is
refused, naming the argument and both values. An argument that only one side holds is not compared
(one added since the archive was written still loads).
The head is compared too, but only where it is copied: `head_config_overrides` (leaving out dropout,
initializer and regularizer keys) must be equal whenever every weight shape of `classification_head`
fits, so a checkpoint saved with `{"pooling_type": "max"}` is refused by an average-pool model
(before this check it loaded). A head of another `num_classes`, `include_top=False` and a headless
checkpoint are not compared; name `head_config_overrides` in `ignore_config_keys` of
`load_pretrained_checkpoint` to accept another head on purpose. An archive that is a functional
wrapper around one model of this class (what the trainers save) is compared by the wrapped model's
configuration; when no single model of the class is found in the archive, the load logs a warning
and checks the layers only. An argument the class has
removed fails earlier, when Keras rebuilds the archived model from its configuration: see the
`weight_decay` note in section 9.

```python
create_mobilenetv4(          # same for v1 to v3
    variant="medium",        # default variant differs per version: see section 7
    num_classes=1000,
    input_shape=None,        # defaults to (224, 224, 3)
    pretrained=False,        # True raises NotImplementedError; a path loads that checkpoint
    **kwargs,                # forwarded to MobileNetV4.from_variant, e.g. width_multiplier, include_top
)
```

`kwargs` reaches the constructor arguments `width_multiplier`, `dropout_rate`, `kernel_regularizer`,
`kernel_initializer`, `include_top` and `head_config_overrides` on every version, plus `depths`, `dims`,
`block_types`, `strides`, `use_attention` and `attention_stages` on V4. A keyword given for a field of the
variant (`MobileNetV4.from_variant("small", depths=[1] * 7, use_attention=True)`,
`MobileNetV2.from_variant("small", width_multiplier=0.5)`) replaces the variant's own value; it is not
multiplied into it.

Constructor arguments, in the order every version takes them (`num_classes` first, `**kwargs` last):

| Argument | V1 | V2 | V3 | V4 | Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `num_classes` | 1000 | 1000 | 1000 | 1000 | classifier width; must be positive when `include_top=True` |
| `variant` | | | `"large"` | | the NAS block table (V1, V2 and V4 take their variant through `from_variant`) |
| `depths`, `dims`, `block_types`, `strides`, `use_attention`, `attention_stages` | | | | yes | the stage table |
| `width_multiplier` | 1.0 | 1.0 | 1.0 | 1.0 | channel multiplier, positive |
| `dropout_rate` | 0.001 | 0.2 | 0.2 | 0.2 | dropout before the classifier, in `[0, 1)` |
| `kernel_regularizer` | `None` | `None` | `None` | `None` | regularizer of the convolution and head kernels (depthwise kernels: V1 only); replaces `weight_decay` |
| `kernel_initializer` | `he_normal` | `he_normal` | `glorot_uniform` | `he_normal` | name, serialized dict or instance |
| `include_top` | `True` | `True` | `True` | `True` | `False` returns the 4-D feature map; a `bool` or `numpy.bool_`, anything else raises `TypeError` |
| `input_shape` | `None` | `None` | `(224, 224, 3)` | `(224, 224, 3)` | `None` means `(224, 224, 3)` |
| `head_config_overrides` | `None` | `None` | `None` | `None` | merged into the head config; `name`, `num_classes`, `task_type` raise |

`weight_decay` is gone from all four constructors: it was an L2 factor that defaulted to a positive value
(`4e-5` for V1 and V2, `1e-5` for V3 and V4), so a default model carried an L2 loss on every kernel (28, 36, 43
and 27 entries in `model.losses` for the four) while the trainer's AdamW already applied its own decoupled
decay. Pass `kernel_regularizer=keras.regularizers.L2(...)` to get an L2 loss; `weight_decay=` now raises
`ValueError`.

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

print(f"medium (alpha=1.0):  {default_v2.count_params():,}")  # 3,538,984
print(f"nano   (alpha=0.5):  {nano_v2.count_params():,}")     # 1,987,224
print(f"medium x 0.5:        {custom_v2.count_params():,}")   # 1,987,224
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

`summary()` on a never-called model runs a real dummy forward pass to materialize the weights.
On every version `build((None, H, W, C))` also creates every weight, by tracing `call`.

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

A `MobileNetV3` or `MobileNetV4` `.keras` file saved before 2026-10-26, when the stem moved onto
one `ConvStem` named `stem` (its weights were `stem_conv/...` and `stem_bn/...`), does not load:
`keras.models.load_model` raises `ValueError`, and no legacy shim exists.

A `MobileNetV4` archive saved at the revision before the layer names moved (2026-10-27, 0192d3e45)
still loads: `keras.models.load_model` and `load_weights` restore every tensor for `small`,
`hybrid_medium` and `small` at `width_multiplier=0.75`, headed and headless, and the inference
output (`training=False`) is equal at `atol=0` (measured; `load_weights` is equal in `training=True`
too, and `load_model` of a headed model differs there exactly as it does between two copies of one
unchanged revision, so that difference is not new). Only the `w.path` strings change, because
archives load by tracked-layer order: `stage_{s}_block_{b}` is now `stage{S}_block{B}` and
`stage_{s}_mqa` is `stage{S}_mqa` (S and B count from 1; 195 of the 207 weights of `small` move,
for example `stage_0_block_0/expand_conv/kernel` is `stage1_block1/expand_conv/kernel`). Code that
selects weights by path must follow.

A `MobileNetV3` archive saved at that revision loads unchanged: `load_model` and `load_weights` restore
every tensor for `large`, `small` and `large` at `width_multiplier=0.75`, headed and headless, and the
inference output is equal at `atol=0` (measured). No `w.path` moves for V3 (`stem`, `block_{i}`,
`last_conv`, `last_bn`, `classification_head`). As for V4, `load_weights` is also equal in
`training=True` and `load_model` of a headed model is not: the head's Dropout draws a different mask
after a reload (with `dropout_rate=0.0` the two are equal), which is independent of this change.

A `MobileNetV1` or `MobileNetV2` archive saved at the revision before their stem moved onto one
`ConvStem` named `stem` (2026-10-27, `be92a74eb`) does not load. `keras.models.load_model`,
`load_weights` and the `pretrained=<path>` loader all raise `ValueError` for V1 `small` (headed and
headless) and `large` at `width_multiplier=0.75`, and V2 `small` (headed and headless) and `medium` at
`width_multiplier=0.75` (6 of 6 subjects, each of the three): V1 says "A total of 54 objects could not be loaded ... Layer 'conv_dw_0'
expected 1 variables, but received 0 variables during loading", V2 "A total of 2 objects could not be
loaded ... Layer 'stem_conv1' expected 1 variables, but received 0 variables during loading" (measured).
Archives load by tracked-layer order and the stem is one tracked layer where three were (`conv1`,
`conv1_bn`, `conv1_relu`), so no shim can pair them. The weights themselves are intact: written by path
into the new model they give outputs equal at `atol=0` in `training=False` and `training=True` for all six
subjects. The paths: `conv1/kernel` is `stem/stem_conv1/kernel` and `conv1_bn/gamma`, `beta`,
`moving_mean` and `moving_variance` are `stem/stem_norm1/...` in both versions. V1's blocks now count from
0 like every flat block list, so `block_{n}/conv_dw_{n}/kernel` is `block_{n-1}/conv_dw_{n-1}/kernel`
(likewise `conv_pw_{n}` and the `_norm` layers; `block_1/conv_dw_1/kernel` is
`block_0/conv_dw_0/kernel`). Code that selects weights by path must follow.

The constructor argument `weight_decay` of all four versions is `kernel_regularizer`, in `__init__` and in
`get_config`. A V1 to V4 archive written with `weight_decay` fails in `keras.models.load_model` and in the
`pretrained=<path>` loader with `ValueError: Unrecognized keyword arguments passed to MobileNetV3:
{'weight_decay': 1e-05}` (the class name changes; measured for V3 and V4 from `0a908d16e`, `small`, `large`
and `hybrid_medium`, headed and headless, 10 of 10 subjects each way), because Keras rebuilds the model from
the archived configuration before it reads a tensor. `load_weights` into a model built by the new code
is unaffected: it restores every tensor of those V3 and V4 archives, outputs equal at `atol=0` in both
training modes (10 of 10).

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
-   **Weight decay**: MobileNets are sensitive to it. The models carry no L2 of their own (`kernel_regularizer=None`), so the optimizer's decoupled `weight_decay` is the one decay; `src/train/mobilenet` passes no regularizer for that reason. An L2 regularizer on top applies the decay twice. The papers' values (`4e-5` for V1/V2, `1e-5` for V3/V4) target ImageNet — reduce them for small datasets.
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
