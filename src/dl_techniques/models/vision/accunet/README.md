# ACC-UNet: A Completely Convolutional UNet for the 2020s

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **ACC-UNet**, based on the paper ["ACC-UNet: A Completely
Convolutional UNet model for the 2020s"](https://arxiv.org/abs/2308.13680) by Ibtehaz & Kihara
(MICCAI 2023).

The architecture rebuilds a U-Net out of three pieces: **HANC blocks** that approximate global
context with pooling instead of attention, **ResPath** and **MLFC** stages that close the
semantic gap in the skip connections, and squeeze-excitation throughout. Nothing in it is a
self-attention layer.

---

## 1. Overview: What is ACC-UNet and Why It Matters

### What is ACC-UNet?

ACC-UNet is a purely convolutional U-Net for segmentation. Its thesis is that the two
properties usually credited to transformer U-Nets — every position sees the whole image, and
features are exchanged *across* scales rather than only within a matched encoder-decoder pair —
can be obtained with pooling and 1x1 convolutions alone, without paying attention's `O(N^2)`
cost in the pixel count `N`. The authors report reaching transformer-level segmentation quality
at a fraction of the parameter count.

### Key Innovations

1. **HANC (Hierarchical Aggregation of Neighborhood Context).** Replaces the standard
   convolution block. Each pixel is compared not against every other pixel but against the mean
   and the peak of its neighborhood at several radii, at `O(N * k)` cost.
2. **MLFC (Multi-Level Feature Compilation).** Every skip connection is enriched with
   information from *all* encoder levels, so a shallow feature acquires deep semantics and a
   deep feature reacquires spatial detail.
3. **ResPath.** Before MLFC, each encoder level passes through a stack of residual conv-SE
   blocks, with the most refinement where the semantic gap is widest.
4. **Efficient primitives.** Inverted bottlenecks, depthwise convolutions and
   squeeze-excitation throughout.

---

## 2. The Problem ACC-UNet Solves

| Architecture | What limits it |
| :--- | :--- |
| **Standard U-Net** | A 3x3 kernel grows the receptive field slowly, so global context is hard to reach. And a plain concatenation joins a detailed-but-context-poor encoder feature to a context-rich-but-coarse decoder feature: the semantic gap. |
| **Transformer U-Net** | Self-attention gives the global view, but at `O(N^2)` in the pixel count, which dominates at segmentation resolutions. It also lacks the convolutional inductive bias, so it needs more data and more parameters. |

ACC-UNet's answer to each: for the **receptive field**, the HANC block pools the feature map at
strides `2, 4, ..., 2^(k-1)`, resizes every summary back to full resolution and lets a 1x1
convolution weigh them — the mean carries texture, the max carries the salient activation, and
their difference at a given radius tells a pixel whether it sits inside a homogeneous region or
on a boundary. For the **semantic gap**, ResPath refines each encoder level and MLFC then mixes
all four levels into each other before the decoder ever sees them.

---

## 3. How ACC-UNet Works: Core Concepts

### The High-Level Architecture

```
Input (H, W, C)
   |
   v  ENCODER: 5 levels of HANC blocks, stride-2 max-pool between them
   |  level 0 (H)   level 1 (H/2)   level 2 (H/4)   level 3 (H/8)  ->  bottleneck (H/16)
   |     |             |               |               |                     |
   |     +-------------+---------------+---------------+                     |
   |                   v                                                     |
   |   SKIP PROCESSING: ResPath (per level) -> MLFC (across levels)          |
   |                   |                                                     |
   v                   v                                                     v
DECODER: 4 levels. Conv2DTranspose upsample, concatenate the processed skip, HANC blocks.
   |
   v
1x1 convolution to num_classes, then sigmoid (num_classes == 1) or softmax
```

The bottleneck bypasses ResPath and MLFC entirely and goes straight into the decoder.

### The `k` Schedule

`k` is the number of hierarchical pooling levels in a HANC block, and it shrinks with depth:
`[3, 3, 3, 2, 1]` down the encoder and `[2, 2, 3, 3]` up the decoder. This is deliberate — a
stride-4 pool at the bottleneck already spans a large fraction of the image, so extra levels
there summarize nearly the same thing.

The endpoint is worth stating plainly, because it is easy to misread as a milder version of the
same block: **at `k = 1` the HANC layer pools nothing at all** and degenerates to a 1x1
projection of the input. The bottleneck level therefore carries no hierarchical context
whatsoever; its receptive field is whatever the two stacked depthwise convolutions give it.

### The Output Is Probabilities, Not Logits

The head applies its own activation — sigmoid for `num_classes == 1`, softmax otherwise. Compile
with `from_logits=False` (the Keras default). Passing `from_logits=True` silently trains against
a double-squashed target.

---

## 4. Architecture Deep Dive

### 4.1 HANC Block

An inverted bottleneck with the context aggregation in the middle. The order in `call` is:

```
Input (H, W, C_in)
  |-> 1x1 Conv expand by inv_factor  + BN + LeakyReLU
  |-> 3x3 Depthwise Conv             + BN + LeakyReLU
  |-> HANCLayer                       (multi-scale pool, resize, concat, 1x1 compile)
  |-> Add input + BN                  (ONLY when C_in == C_out)
  |-> 1x1 Conv project to C_out      + BN + LeakyReLU
  +-> SqueezeExcitation
Output (H, W, C_out)
```

The residual shortcut exists only when the input and output widths agree, which in practice
means the second block of each level; the first block of a level changes width and runs without
one. `inv_factor` is 3 everywhere except decoder level 3, which uses 4.

For a block with `k` levels the HANCLayer concatenates the untouched input with `k-1`
average-pooled and `k-1` max-pooled summaries, giving `C * (2k - 1)` channels for the 1x1
convolution to weigh.

### 4.2 MLFC Layer

For each target level `i` in turn: resize all four levels to level `i`'s resolution,
concatenate, compile back down with a 1x1 convolution, concatenate that with the original
`Feat_i`, merge with a second 1x1 convolution, and add residually. Shapes in equal shapes out.
Squeeze-excitation is applied once per level at the end.

### 4.3 ResPath

A stack of residual blocks (3x3 conv + BN, squeeze-excitation, add input) applied to one encoder
level before MLFC. The stack depths are `[4, 3, 2, 1]` from the shallowest level down: most
refinement where the gap to the decoder is widest.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np

from dl_techniques.models.vision.accunet import create_acc_unet_binary

# 1. Dummy segmentation data: find the bright circle. H and W must be multiples of 16.
def make_data(n, size=128):
    images = np.zeros((n, size, size, 1), dtype="float32")
    masks = np.zeros((n, size, size, 1), dtype="float32")
    rng = np.random.default_rng(0)
    rr, cc = np.ogrid[:size, :size]
    for i in range(n):
        x, y = rng.integers(20, size - 20, size=2)
        circle = (rr - y) ** 2 + (cc - x) ** 2 < rng.integers(5, 20) ** 2
        images[i, circle] = 1.0
        masks[i, circle] = 1.0
        images[i] += rng.normal(0, 0.1, (size, size, 1))
    return np.clip(images, 0, 1), masks

x_train, y_train = make_data(32)
x_val, y_val = make_data(8)

# 2. Build. The head already applies sigmoid, so from_logits stays False.
model = create_acc_unet_binary(input_channels=1, input_shape=(128, 128), base_filters=16)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 3. Train and predict
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=1, batch_size=4, verbose=0)
predicted_mask = model.predict(x_val[:1], verbose=0)[0]
print(predicted_mask.shape)  # (128, 128, 1)
```

---

## 6. Component Reference

### 6.1 `AccUNet` (Model Class)

**Location**: `dl_techniques.models.vision.accunet.AccUNet`

The subclassed model. Usually you want a factory instead, which wraps it in the Functional API
(§6.2).

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `input_channels` | (required) | Channels in the input image. A constructor argument, not inferred at build time, because `HANCBlock` fixes its expansion width from the channel count at construction. |
| `num_classes` | (required) | Output classes. `1` selects the sigmoid head. |
| `base_filters` | `32` | Width of encoder level 0. Levels then run `[b, 2b, 4b, 8b, 16b]`. |
| `mlfc_iterations` | `3` | Number of stacked MLFC layers. |
| `kernel_initializer` | `'glorot_uniform'` | Initializer for the convolutions. |
| `kernel_regularizer` | `None` | Optional Keras regularizer. |

`mlfc_iterations` does **not** set `MLFCLayer.num_iterations`. It creates that many *separate*
single-iteration MLFC layers applied in sequence. The two forms are not equivalent, because
`MLFCLayer` applies its per-level squeeze-excitation once after its internal loop — stacking
three layers recalibrates channels three times, not once.

### 6.2 Factory Functions

All three live in `dl_techniques.models.vision.accunet` and return an `AccUNetFunctional` — a
`keras.Model` subclass built with the Functional API, which adds nothing but the automatic XLA
opt-out of §10.

| Factory | Signature |
| :--- | :--- |
| `create_acc_unet` | `(input_channels, num_classes, base_filters=32, mlfc_iterations=3, input_shape=None, **kwargs)` |
| `create_acc_unet_binary` | `(input_channels, input_shape=None, base_filters=32, mlfc_iterations=3, **kwargs)` |
| `create_acc_unet_multiclass` | `(input_channels, num_classes, input_shape=None, base_filters=32, mlfc_iterations=3, **kwargs)` |

`input_shape` is the spatial pair `(height, width)`, not the full input shape; `None` builds a
dynamic-shape model. The wrappers put it in a different position from `create_acc_unet`, so
always pass it by keyword.

```python
model = create_acc_unet(
    input_channels=3, num_classes=5, input_shape=(256, 256),
    base_filters=32, mlfc_iterations=3,
)
binary = create_acc_unet_binary(input_channels=1, input_shape=(512, 512))
multi = create_acc_unet_multiclass(input_channels=3, num_classes=8, input_shape=(224, 224))
```

### 6.3 Core Layers

Every building block lives in `layers/`, not in this package, and is reusable on its own.

| Layer | Location |
| :--- | :--- |
| `HANCBlock` | `dl_techniques.layers.hanc_block.HANCBlock` |
| `HANCLayer` | `dl_techniques.layers.hanc_layer.HANCLayer` |
| `MLFCLayer` | `dl_techniques.layers.multi_level_feature_compilation.MLFCLayer` |
| `ResPath` | `dl_techniques.layers.res_path.ResPath` |
| `SqueezeExcitation` | `dl_techniques.layers.squeeze_excitation.SqueezeExcitation` |

---

## 7. Configuration & Model Variants

### Input Size Contract

The encoder performs **4 stride-2 downsampling stages** and the decoder **4 stride-2 transposed
convolutions**. Because `Conv2DTranspose(strides=2, padding='same')` always emits exactly
`2 * H_in`, the model can only round-trip spatial dimensions when **`H` and `W` are divisible by
16**. A static shape that is not is rejected with a `ValueError` naming `divisible by 16` at the
first call, rather than failing deeper with a concatenate mismatch.

Dynamic (`None`) dimensions are accepted and unchecked. The contract still holds at run time; it
simply cannot be verified at trace time. **Resize your inputs to a multiple of 16 before feeding
the model** — the trainer in `src/train/accunet/` does this for you.

### Choosing `base_filters`

This sets the capacity of the whole network: the five encoder levels are
`[base_filters, base_filters*2, ..., base_filters*16]`.

| `base_filters` | Use Case |
| :---: | :--- |
| **16** | Lightweight tasks, fast inference. |
| **32 (Default)** | Good balance for most medical and natural image tasks. |
| **64** | Complex tasks, large datasets, high-resolution images. |

**Parameter counts are not quoted here.** This file used to carry two of them — a "Verified
parameter counts ... ~16.8 M trainable" line and a `~4.5 M / ~16.8 M / ~66.5 M` column in the
table above — and all of them were wrong, contradicting `model.py`'s own docstring by about 25%.
The single home for the measured counts, together with the one-line command that re-derives
them, is the module docstring of
`model.py`; they are pinned by
`tests/test_models/test_accunet/test_model.py::TestDocumentedParameterCounts`.

The count is resolution-independent — all spatial dependence lives in weightless `Resizing`
layers — which is what makes a single documented number legitimate.

### Choosing `mlfc_iterations`

Each stacked layer mixes the levels again and applies its own squeeze-excitation.

| `mlfc_iterations` | Effect |
| :---: | :--- |
| **1** | Basic cross-level mixing. Fastest, but the semantic gap is only partly bridged. |
| **3 (Default)** | Iterative refinement; the value the paper uses. |
| **4+** | Deeper compilation, at a real cost in computation. |

---

## 8. Usage Examples

### Example 1: Binary segmentation with a Dice + BCE loss

```python
import keras
from dl_techniques.models.vision.accunet import create_acc_unet_binary

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = keras.ops.reshape(y_true, (-1,))
    y_pred_f = keras.ops.reshape(y_pred, (-1,))
    intersection = keras.ops.sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) + smooth
    )

def bce_dice_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return keras.ops.mean(bce) + dice_loss(y_true, y_pred)

model = create_acc_unet_binary(input_channels=1, input_shape=(256, 256), base_filters=32)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=bce_dice_loss,
              metrics=["binary_accuracy"])
```

### Example 2: Multi-class segmentation with a dynamic input size

```python
import numpy as np
from dl_techniques.models.vision.accunet import create_acc_unet_multiclass

model = create_acc_unet_multiclass(input_channels=3, num_classes=5, input_shape=None)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])

# Any spatial size works as long as both dimensions are multiples of 16.
print(model.predict(np.random.rand(1, 224, 224, 3), verbose=0).shape)  # (1, 224, 224, 5)
print(model.predict(np.random.rand(1, 256, 320, 3), verbose=0).shape)  # (1, 256, 320, 5)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: HANC blocks in your own network

`HANCBlock` is self-contained and can replace a `Conv2D` block anywhere. `input_channels` must
be passed explicitly and must match the incoming channel count.

```python
import keras
from dl_techniques.layers.hanc_block import HANCBlock

inputs = keras.Input(shape=(64, 64, 3))
x = HANCBlock(filters=32, input_channels=3, k=3)(inputs)
x = keras.layers.MaxPooling2D(2)(x)
x = HANCBlock(filters=64, input_channels=32, k=3)(x)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
custom_model = keras.Model(inputs, outputs)
```

### Pattern 2: MLFC on another U-Net's skips

```python
from dl_techniques.layers.multi_level_feature_compilation import MLFCLayer

# encoder_features is a list of four tensors, one per level, with these widths:
channels = [64, 128, 256, 512]
mlfc = MLFCLayer(channels_list=channels, num_iterations=3)
# processed = mlfc(encoder_features)   # same shapes in, same shapes out
```

Note the difference from `AccUNet`: here `num_iterations` loops *inside* one layer and applies
squeeze-excitation once at the end, whereas `mlfc_iterations` stacks separate layers.

---

## 10. Performance Optimization

### XLA is disabled automatically

`HANCLayer` resizes its pooled summaries back to full resolution with nearest-neighbour
interpolation, and the backward pass of that op, `ResizeNearestNeighborGrad`, has no registered
XLA-GPU kernel in TF 2.18, so an XLA training step cannot compile.

`AccUNet` and the `AccUNetFunctional` the factories return therefore force `jit_compile=False`
on every compile path, including the recompile behind `load_model()`. The forced value wins over
an explicit `jit_compile=True`, and a warning is logged when it overrides one. Nothing is
required of you; inference (`predict`) and CPU training were never affected.

### Mixed precision

```python
import keras
from dl_techniques.models.vision.accunet import create_acc_unet_binary

keras.mixed_precision.set_global_policy("mixed_float16")

model = create_acc_unet_binary(input_channels=1, input_shape=(256, 256))
model.compile(optimizer="adam", loss="binary_crossentropy")
```

Mixed precision cuts activation memory roughly in half and speeds up the convolutions on Tensor
Core hardware. Set the policy before building the model.

---

## 11. Training and Best Practices

- **Binary.** `BinaryCrossentropy` combined with a Dice or Tversky loss is the usual choice —
  BCE handles pixel-wise accuracy, Dice handles region overlap. Track `BinaryAccuracy` and IoU.
- **Multi-class.** `SparseCategoricalCrossentropy` for integer masks, `CategoricalCrossentropy`
  for one-hot. Track `MeanIoU`.
- **Never set `from_logits=True`.** The head already applied sigmoid or softmax (§3).
- **Augment.** Geometric (flips, rotations, scaling, cropping), photometric (brightness and
  contrast), and elastic deformation for medical images.
- **Resize to a multiple of 16** in the data pipeline, not in the model.

---

## 12. Serialization & Deployment

The model and every custom layer are registered, so a `.keras` file round-trips without
`custom_objects`.

```python
import keras
import numpy as np
from dl_techniques.models.vision.accunet import create_acc_unet_binary

model = create_acc_unet_binary(input_channels=1, input_shape=(128, 128), base_filters=16)
model.save("acc_unet_model.keras")

loaded = keras.models.load_model("acc_unet_model.keras")
print(loaded.predict(np.random.rand(1, 128, 128, 1), verbose=0).shape)  # (1, 128, 128, 1)
```

The saved file is a standard Keras model: serve it with TensorFlow Serving, or convert it to
ONNX or TFLite.

---

## 13. Testing & Validation

The tests live in `tests/test_models/test_accunet/`: construction, forward-pass shapes, the
divisible-by-16 rejection, serialization, gradient flow to every trainable weight, and the
parameter counts documented in `model.py`.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_accunet -q
```

---

## 14. Troubleshooting

- **`ValueError: ... divisible by 16`.** A static `input_shape` whose height or width is not a
  multiple of 16. Resize the data; the check fires at the first call, not at construction.
- **A shape mismatch with `input_shape=None`.** The divisibility contract is unchecked for
  dynamic shapes but still real — feed multiples of 16.
- **Outputs saturate at 0 or 1, or the loss will not move.** Check the learning rate (start at
  `1e-4`), that the inputs are normalized, and that you did not pass `from_logits=True`.
- **`ValueError` about input channels in a hand-built `HANCBlock`.** `input_channels` must match
  the preceding layer's channel count. `AccUNet` wires this for you; custom architectures do not.
- **Out of memory at `base_filters=64`.** The five levels reach `16 * base_filters` channels at
  full bottleneck depth. Drop to 32, or reduce the tile size.
- **`InvalidArgumentError: ... ResizeNearestNeighborGrad ... on XLA_GPU_JIT` during `fit()`.**
  Cannot come from this package's models — they disable XLA themselves (§10). The remaining way
  to see it is a network of your own that embeds `HANCBlock` or `HANCLayer` (§9) and inherits no
  such override; that model has to train with XLA off.

---

## 15. Technical Details

**2D only.** Every layer is a 2D convolution or pooling op. The concepts carry over to 3D by
substituting the 3D counterparts, but that is a source change, not a configuration one.

**Why "completely convolutional".** Global context comes from pooling and 1x1 convolutions.
There is no self-attention, no patch embedding and no multi-head layer anywhere in the model.

**Against other U-Net variants.** U-Net++ densifies the skips with nested decoders; Attention
U-Net gates them spatially. ACC-UNet works on a different axis: MLFC mixes *across* levels
rather than nesting or gating within them, and its attention is channel-wise
(squeeze-excitation) plus the HANC pooling proxy for a global view.

---

## 16. Citation

```bibtex
@inproceedings{ibtehaz2023acc,
  title={ACC-UNet: A Completely Convolutional UNet model for the 2020s},
  author={Ibtehaz, Nabil and Kihara, Daisuke},
  booktitle={International Conference on Medical Image Computing and
             Computer-Assisted Intervention},
  pages={692--702},
  year={2023},
  organization={Springer}
}
```
