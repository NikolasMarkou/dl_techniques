# SqueezeNet and SqueezeNodule-Net

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

Keras 3 implementations of two related efficient ConvNets:

- **SqueezeNet V1** (`squeezenet_v1.py`) -- the original architecture built from **Fire modules**. The authors report AlexNet-level ImageNet accuracy with 50x fewer parameters.
- **SqueezeNodule-Net** (`squeezenet_v2.py`) -- a medical-imaging variant built from **simplified Fire modules** (3x3 expand only), with 2D and 3D configurations.

They are separate architectures, not a version upgrade; neither deprecates the other.

---

## 1. Overview: What is SqueezeNet and Why It Matters

### What is SqueezeNet?

SqueezeNet is a small convolutional classifier whose whole design is organised around cutting parameters rather than adding capacity. Its building block, the Fire module, first "squeezes" the channel count with `1x1` convolutions and then "expands" it again with a mix of `1x1` and `3x3` convolutions.

### Key Innovations

1. **The Fire module.** A `1x1` squeeze convolution creates a narrow bottleneck, then two parallel expand convolutions (`1x1` and `3x3`) widen it back and are concatenated on the channel axis.
2. **Three design rules.** Prefer `1x1` filters over `3x3`; reduce the input channel count seen by the surviving `3x3` filters; downsample late so most layers work on large feature maps.
3. **No fully-connected classifier.** The head is a `1x1` convolution with `num_classes` filters followed by global average pooling, which removes the parameter-heavy dense layers of AlexNet-era models.
4. **Simplified Fire module (SqueezeNodule-Net).** The `1x1` expand path is removed entirely, so every expanded channel carries `3x3` spatial context, and the squeeze ratio is raised to widen the bottleneck.

---

## 2. The Problem SqueezeNet Solves

Parameter count in a convolution is `in_channels * k * k * out_channels`. A stack of `3x3` layers over wide feature maps therefore spends most of its budget on the `3x3` kernels, which is why AlexNet-era classifiers ran to hundreds of megabytes and could not be shipped to a phone or an embedded board.

The Fire module attacks both factors at once: `1x1` kernels cost `9x` less than `3x3` kernels for the same channel counts, and the squeeze layer shrinks `in_channels` before the remaining `3x3` convolution ever runs. The paper's result is that the accuracy cost of this is much smaller than the size saving.

---

## 3. How SqueezeNet Works: Core Concepts

### The Fire module

```
Input (C_in channels)
    |
    +-> squeeze:  Conv 1x1, s1x1 filters, ReLU        -> (H, W, s1x1)
    |
    +-> expand 1x1: Conv 1x1, e1x1 filters, ReLU  --+
    +-> expand 3x3: Conv 3x3, e3x3 filters, ReLU  --+-> concat -> (H, W, e1x1 + e3x3)
```

The expensive `3x3` convolution sees only `s1x1` input channels. In `fire3` of the V1 configuration that is 16 channels instead of 128.

### Squeeze ratio

`SR = s1x1 / (e1x1 + e3x3)` measures how hard the bottleneck squeezes.

| Configuration | Typical SR | Effect |
| :--- | :--- | :--- |
| SqueezeNet V1 | 0.125 (e.g. `16 / (64 + 64)`) | Aggressive compression, fewest parameters. |
| SqueezeNodule-Net `v1` | 0.25 (`16 / 64`) | Lighter of the two nodule configurations. |
| SqueezeNodule-Net `v2` | 0.50 early (`32 / 64`), 0.25 late | Wider bottleneck through the early stages. |

### Network shape

```
Input (B, H, W, 3)
    |
    +-> conv1 (kernel/stride from the variant, ReLU, valid padding)
    +-> maxpool 3x3 stride 2, valid
    |
    +-> fire2 ... fire9, with maxpool 3x3/2 inserted at the variant's pool positions
    +-> dropout (after the last Fire module)
    |
    +-- include_top=False -> the feature map after dropout
    +-- include_top=True  -> conv10 (1x1, num_classes filters, ReLU)
                             -> GlobalAveragePooling -> softmax -> (B, num_classes)
```

The head ends in **softmax**, so the model outputs probabilities. Compile with `from_logits=False` (the default).

---

## 4. Architecture Deep Dive

### 4.1 `FireModule` (V1)

`FireModule(s1x1, e1x1, e3x3, kernel_regularizer=None, kernel_initializer=...)`. Squeeze `1x1` -> parallel `1x1` and `3x3` expands -> channel concatenation. Output width is `e1x1 + e3x3`; spatial size is unchanged.

### 4.2 `SimplifiedFireModule` (SqueezeNodule-Net)

`SimplifiedFireModule(s1x1, e3x3, ...)`. Same squeeze, but only the `3x3` expand path. Output width is `e3x3`. Dropping the `1x1` path means no output channel is a pure channel mixture -- every one carries local spatial context, which is the stated motivation for texture-heavy medical images.

### 4.3 Valid padding and the minimum input size

Every downsampling stage uses `padding='valid'`: a strided stem convolution plus three `pool_size=3, strides=2` max-pools. Under valid padding an axis of length `n` becomes `(n - k) // s + 1`, which reaches **zero** for small inputs, and Keras 3.8 does not raise on a zero-length spatial axis -- the model would emit a correctly shaped, all-NaN tensor.

`spatial_guard.py` walks that arithmetic before construction and raises a `ValueError` naming the collapsing stage, the axis and the minimum legal extent. The minima are computed per variant, not tabulated:

| Variant family | Minimum spatial extent per axis |
| :--- | ---: |
| V1 `"1.0"`, `"1.0_bypass"` (7x7 stem, pools after conv1/fire4/fire8) | 35 |
| V1 `"1.1"` (3x3 stem, pools after conv1/fire3/fire5) | 31 |
| All SqueezeNodule-Net variants | 35 |

This is why CIFAR-sized `32x32` input works with `"1.1"` and raises with `"1.0"`.

### 4.4 Bypass connections

The `"1.0_bypass"` variant adds identity residuals around `fire3`, `fire5`, `fire7` and `fire9` -- the four positions where a Fire module's input and output widths match (128, 256, 384, 512). This is the paper's "simple bypass"; it adds no parameters.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np

from dl_techniques.models.vision.squeezenet import create_squeezenet_v1

# 1. Model. "1.1" clears the 31-pixel floor, so 32x32 input is legal.
model = create_squeezenet_v1(
    variant="1.1",
    num_classes=10,
    input_shape=(32, 32, 3),
)

# 2. Compile. The head ends in softmax, so the loss reads probabilities.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.summary()

# 3. One training step on dummy data.
images = np.random.rand(16, 32, 32, 3).astype("float32")
labels = np.random.randint(0, 10, 16)
loss, acc = model.train_on_batch(images, labels)
print(f"loss={loss:.4f} acc={acc:.4f}")

# 4. Inference.
predictions = model.predict(images, verbose=0)
print(predictions.shape)  # (16, 10)
```

---

## 6. Component Reference

### 6.1 Public API

Everything below is importable straight from `dl_techniques.models.vision.squeezenet`.

| Name | Module | Purpose |
| :--- | :--- | :--- |
| `SqueezeNetV1` | `squeezenet_v1` | Functional `keras.Model` for the V1 architecture. |
| `create_squeezenet_v1` | `squeezenet_v1` | Factory for `SqueezeNetV1`. |
| `FireModule` | `squeezenet_v1` | Squeeze/expand block with both expand paths. |
| `SqueezeNoduleNetV2` | `squeezenet_v2` | Functional `keras.Model` for SqueezeNodule-Net (2D and 3D). |
| `create_squeezenodule_net_v2` | `squeezenet_v2` | Factory for `SqueezeNoduleNetV2`. |
| `SimplifiedFireModule` | `squeezenet_v2` | Squeeze/expand block with only the `3x3` path. |

### 6.2 Factory arguments

Both factories take the same four named arguments and forward everything else to the constructor:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `variant` | `"1.0"` / `"v2"` | Key into the class's `MODEL_VARIANTS`. |
| `num_classes` | `1000` | Output classes. |
| `input_shape` | `(224, 224, 3)` | Input shape; 4D `(D, H, W, C)` for the 3D variants. |
| `weights` | `None` | Unsupported. Any non-`None` value raises `NotImplementedError`. |

Constructor arguments reachable through `**kwargs`:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `include_top` | `True` | `False` returns the post-dropout feature map. |
| `dropout_rate` | `0.5` | Dropout after the last Fire module. |
| `kernel_regularizer` | `None` | Applied to every convolution. |
| `kernel_initializer` | Caffe xavier | Fire and bypass convolutions. |
| `use_bypass` (V1 only) | from variant | `False`, `"simple"` or `"complex"`; an explicit value overrides the variant. |
| `use_3d` (V2 only) | from variant | Switches Conv/MaxPool/GlobalAveragePooling to their 3D forms. |

### 6.3 Pretrained weights

None are distributed. Passing `weights=<anything>` to either factory (or to `from_variant`) raises `NotImplementedError` rather than quietly returning a random model. To reuse a local checkpoint, build the architecture and `keras.models.load_model()` the file.

---

## 7. Configuration & Model Variants

### SqueezeNetV1

Parameter counts measured at `input_shape=(224, 224, 3)`, `num_classes=1000`.

| Variant | Stem | Pools after | Bypass | Params |
| :--- | :--- | :--- | :--- | ---: |
| `1.0` | `7x7`, 96 filters, stride 2 | conv1, fire4, fire8 | none | 1,248,424 |
| `1.1` | `3x3`, 64 filters, stride 2 | conv1, fire3, fire5 | none | 1,235,496 |
| `1.0_bypass` | `7x7`, 96 filters, stride 2 | conv1, fire4, fire8 | simple, at fire3/5/7/9 | 1,248,424 |

All three share one Fire schedule:

| Module | `s1x1` | `e1x1` | `e3x3` | Output width |
| :--- | ---: | ---: | ---: | ---: |
| fire2, fire3 | 16 | 64 | 64 | 128 |
| fire4, fire5 | 32 | 128 | 128 | 256 |
| fire6, fire7 | 48 | 192 | 192 | 384 |
| fire8, fire9 | 64 | 256 | 256 | 512 |

### SqueezeNoduleNetV2

| Variant | Squeeze filters (fire2..fire9) | Dimensionality | Params at 224x224x3, 1000 classes |
| :--- | :--- | :--- | ---: |
| `v1` | 16, 16, 32, 32, 48, 48, 64, 64 | 2D | 878,504 |
| `v2` | 32, 32, 64, 64, 96, 96, 64, 64 | 2D | 1,160,808 |
| `v1_3d` | same as `v1` | 3D | -- |
| `v2_3d` | same as `v2` | 3D | -- |

Expand widths are the same for all four: 64, 64, 128, 128, 192, 192, 256, 256. All four use the `7x7` stem and pool after conv1, fire4 and fire8. The `_3d` variants take a 4D `(D, H, W, C)` input, so they have no entry in that column; `v2_3d` with `num_classes=2` measures 2,545,154 parameters.

---

## 8. Usage Examples

### Example 1: backbone for a downstream task

```python
import numpy as np
from dl_techniques.models.vision.squeezenet import create_squeezenet_v1

backbone = create_squeezenet_v1(
    variant="1.0",
    include_top=False,
    input_shape=(224, 224, 3),
)

images = np.random.rand(2, 224, 224, 3).astype("float32")
features = backbone.predict(images, verbose=0)
print(features.shape)  # (2, 12, 12, 512)
```

The trailing width is always 512 (the `fire9` output). The spatial size follows the variant's valid-padding arithmetic, not a clean power of two: `"1.0"` gives `12x12` at 224 input and `"1.1"` gives `13x13`.

### Example 2: a 3D model for volumetric data

```python
from dl_techniques.models.vision.squeezenet import create_squeezenodule_net_v2

model_3d = create_squeezenodule_net_v2(
    variant="v2_3d",
    num_classes=2,
    input_shape=(64, 64, 64, 1),   # (depth, height, width, channels)
)
print(f"{model_3d.count_params():,}")  # 2,545,154
```

### Example 3: bypass connections

```python
from dl_techniques.models.vision.squeezenet import create_squeezenet_v1

bypass_model = create_squeezenet_v1(
    variant="1.0_bypass",
    num_classes=100,
    input_shape=(224, 224, 3),
)
print([l.name for l in bypass_model.layers if l.name.startswith("add")])
# ['add_fire3', 'add_fire5', 'add_fire7', 'add_fire9']
```

---

## 9. Advanced Usage Patterns

### Overriding a variant's bypass setting

`use_bypass` is resolved from an `is None` sentinel, not from truthiness, so an explicit `False` genuinely turns the bypass off even on `"1.0_bypass"`:

```python
plain = create_squeezenet_v1("1.0_bypass", num_classes=10,
                             input_shape=(64, 64, 3), use_bypass=False)
print([l.name for l in plain.layers if l.name.startswith("add")])  # []
```

`"complex"` is also accepted: it bypasses every Fire module, inserting a `1x1` projection where the widths do not match.

### Mixed precision

```python
keras.mixed_precision.set_global_policy("mixed_float16")
```

Set the policy before constructing the model. `model.fit()` handles loss scaling.

---

## 10. Training Notes

- **Optimizer**: Adam or AdamW work well; the models are small enough that the schedule rarely needs tuning.
- **Regularization**: `dropout_rate` (default `0.5`) sits between the last Fire module and the head. `kernel_regularizer` reaches every convolution.
- **Initialization**: the stem and Fire convolutions use Caffe's xavier filler (`fan_in`-normalized, i.e. `lecun_uniform` here) and `conv10` alone uses `Normal(0, 0.01)`, transcribed from the reference prototxt. `caffe_reference_init.py` carries the derivation.
- **Input size**: check the minimum extent in 4.3 before picking a variant for small images.

---

## 11. Serialization

```python
model.save("my_squeezenet.keras")
loaded = keras.models.load_model("my_squeezenet.keras")
```

The models and both Fire-module layers are registered, so no `custom_objects` argument is needed.

---

## 12. Troubleshooting

- `ValueError: ... collapses to length 0 at stage ...` -- the input is smaller than the variant's minimum extent (4.3). Use `"1.1"` for 32-pixel inputs, or enlarge the input.
- `NotImplementedError` from `weights=` -- expected; no checkpoints ship with this package.
- `ValueError: num_classes must be a positive integer` / `dropout_rate must be in range [0, 1)` -- constructor validation.
- Loss much higher than expected -- the head already applies softmax; do not pass `from_logits=True`.
- A backbone output with an unexpected spatial size -- valid padding, not `same`. See Example 1.

---

## 13. Technical Details

**Head.** `conv10` is a `1x1` convolution with `num_classes` filters and a ReLU activation, followed by global average pooling and softmax. The softmax is applied at every `num_classes`, including 2; there is no binary special case, so the output shape is always `(B, num_classes)`.

**Initializers.** `STEM_INITIALIZER` and `HEAD_INITIALIZER` are different fillers on purpose. `HEAD_INITIALIZER` is stored as a serialized config dict and copied at each call site, because one shared seedless `Initializer` instance replays its draw.

**Fire numbering.** Modules are named `fire2` through `fire9`, matching the paper; the stem convolution is `conv1` and the head convolution is `conv10`.

---

## 14. Citation

- **SqueezeNet**:
    ```bibtex
    @article{iandola2016squeezenet,
      title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size},
      author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
      journal={arXiv preprint arXiv:1602.07360},
      year={2016}
    }
    ```
- **SqueezeNodule-Net**:
    ```bibtex
    @article{tsivgoulis2022improved,
      title={An improved SqueezeNet model for the diagnosis of lung cancer in CT scans},
      author={Tsivgoulis, Georgios and Skiadopoulos, Spiros and Vassilacopoulos, George},
      journal={Machine Learning with Applications},
      volume={9},
      pages={100399},
      year={2022},
      publisher={Elsevier}
    }
    ```
