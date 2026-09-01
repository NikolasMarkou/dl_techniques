# Vision Transformer (ViT)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **Vision Transformer (ViT)**, from ["An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020).

Six scales are provided (`pico`, `tiny`, `small`, `base`, `large`, `huge`). Block internals -- normalization type and position, feed-forward network type, activation -- are resolved through the `dl_techniques` factories, so a variant can be swapped in without forking the file.

---

## 1. Overview: What is ViT and Why It Matters

### What is a Vision Transformer?

A ViT applies the standard Transformer encoder directly to images. The image is cut into non-overlapping square patches, each patch is flattened and linearly projected to `embed_dim`, and from that point the network has no notion of two dimensions at all: it sees a set of tokens, and any geometry it uses it must learn.

### Key Innovations

1. **Sequence-based image processing.** The convolutional priors -- locality, translation equivariance, a hierarchy of scales -- are dropped entirely rather than built into the weights.
2. **Global receptive field from layer one.** Self-attention lets any patch influence any other patch in a single layer; a ConvNet needs many downsampling stages to reach the same interaction.
3. **Explicit position.** Because the patch grid is discarded, spatial position is re-injected by a learned positional embedding. Remove it and the model is permutation-invariant over patches: an image and its shuffled version look identical.
4. **Scaling.** The authors report that with enough pre-training data (ImageNet-21k, JFT-300M) the architecture matches or beats the best CNNs of the time.

---

## 2. The Problem ViT Solves

A CNN kernel sees only its immediate neighbourhood, so relating a dog's head to its tail requires information to propagate through many layers. Dilated convolutions, larger kernels and bolt-on attention modules all mitigate this, but each is a patch on a locality assumption that is baked into the operator.

ViT removes the assumption instead. Self-attention weights are content-based and computed over the whole sequence, so long-range structure is available immediately and is learned from data rather than hard-wired.

The trade-off is sample efficiency. Priors that constrain a CNN are also what make it learn from little data. On small datasets a ViT trained from scratch overfits or plateaus unless it gets heavy augmentation, strong regularization, or pre-trained weights.

---

## 3. How ViT Works: Core Concepts

### From image to sequence

An `H x W` image with patch size `P` becomes `(H/P) * (W/P)` tokens. Attention cost is quadratic in that count, so patch size is the central efficiency knob: halving `P` quadruples the sequence length and multiplies attention cost by roughly sixteen.

### The CLS token and positional embeddings

- **CLS token**: a learnable vector prepended to the sequence. It carries no image content; its job is to accumulate a whole-image summary through attention, giving the head a single vector to read.
- **Positional embeddings**: learned, absolute, 1D. One vector per position (including position 0, the CLS token), added to the patch embeddings.

### Complete data flow

```
Input image (B, H, W, C)
    |
    +-> PatchEmbedding (Conv2D, kernel=stride=patch_size)  -> (B, N, D)
    +-> prepend CLS token                                  -> (B, N+1, D)
    +-> add positional embedding, optional dropout
    |
    +-> TransformerLayer x num_layers
    |     [Norm] -> Multi-Head Self-Attention -> Add
    |     [Norm] -> Feed-Forward Network      -> Add
    |
    +-> final Layer Normalization                          -> (B, N+1, D)
    |
    +-- include_top=True  -> take CLS state -> optional dropout -> Dense -> logits (B, num_classes)
    +-- include_top=False -> pooling: 'cls' | 'mean' | 'max' -> (B, D), or None -> (B, N+1, D)
```

The head emits **logits**, so compile with `from_logits=True`.

---

## 4. Architecture Deep Dive

### 4.1 Patch embedding

Implemented as a single `Conv2D` with `kernel_size = strides = patch_size` and `filters = embed_dim`, which extracts and embeds every patch in one operation. The output is reshaped from `(B, H', W', D)` to `(B, H'*W', D)`.

Image height and width must be exact multiples of the patch size; the constructor raises `ValueError` otherwise.

### 4.2 Transformer layer

The repeated block is a configurable `TransformerLayer`. Both normalization placements are supported through `normalization_position`:

| Value | Order | Notes |
| :--- | :--- | :--- |
| `"post"` | `SubLayer -> Add -> Norm` | **Default in this implementation.** Can need learning-rate warmup at depth. |
| `"pre"` | `Norm -> SubLayer -> Add` | Matches the published ViT. Generally more stable to train. |

The default is `"post"` and therefore does **not** reproduce the paper's block; pass `normalization_position="pre"` if you want the published configuration.

### 4.3 Pooling and the CLS token

When `include_top=False`, `pooling="mean"` and `pooling="max"` deliberately **exclude position 0**: averaging the CLS token into the patch statistics mixes a summary vector into the thing it summarizes. Only `pooling="cls"` reads position 0, and it reads it alone. `pooling=None` returns the full token sequence, which is what a detection or segmentation head wants.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from keras.datasets import cifar10

from dl_techniques.models.vision.vit.model import ViT

# 1. Data (a subset keeps this example fast).
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:2000].astype("float32") / 255.0
y_train = y_train[:2000]
x_test = x_test[:1000].astype("float32") / 255.0
y_test = y_test[:1000]

# 2. Model. 'pico' and patch_size=4 suit 32x32 inputs.
model = ViT(
    input_shape=(32, 32, 3),
    num_classes=10,
    scale="pico",
    patch_size=4,
    include_top=True,
    dropout_rate=0.1,
)

# 3. Compile. The head emits logits.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary_detailed()

# 4. Train and evaluate.
model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc * 100:.2f}%")
```

---

## 6. Component Reference

### 6.1 `ViT` (model class)

`dl_techniques.models.vision.vit.model.ViT` -- the `keras.Model` subclass that assembles the architecture.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `input_shape` | `(224, 224, 3)` | Input image shape `(H, W, C)`; `H` and `W` must divide by the patch size. |
| `num_classes` | `1000` | Output classes; used only when `include_top=True`. |
| `scale` | `"base"` | Key into `ViT.SCALE_CONFIGS`. |
| `patch_size` | `16` | Int for square patches, or `(h, w)`. |
| `include_top` | `True` | `False` turns the model into a feature extractor. |
| `pooling` | `None` | `'cls'`, `'mean'`, `'max'` or `None`. |
| `dropout_rate` | `0.0` | General dropout. |
| `attention_dropout_rate` | `0.0` | Dropout on attention weights. |
| `pos_dropout_rate` | `0.0` | Dropout after the positional embedding. |
| `kernel_initializer` | `None` | `None` resolves to `TruncatedNormal(stddev=0.02)`, the ViT reference. |
| `normalization_type` | `"layer_norm"` | Also `rms_norm`, `batch_norm`, `band_rms`, `adaptive_band_rms`, `dynamic_tanh`. |
| `normalization_position` | `"post"` | `'pre'` or `'post'` (see 4.2). |
| `ffn_type` | `"mlp"` | Also `swiglu`, `differential`, `glu`, `geglu`, `residual`, `swin_mlp`. |
| `activation` | `"gelu"` | FFN activation. |
| `use_layer_scale` | `False` | Enables per-block LayerScale. |
| `layer_scale_init_value` | `1e-5` | Initial LayerScale value. |

Also available: `normalization_kwargs`, `kernel_regularizer`, `bias_initializer`, `bias_regularizer`, `name`.

Useful methods:

| Method | Purpose |
| :--- | :--- |
| `ViT.from_variant(variant, ...)` | Build from a `MODEL_VARIANTS` key (see 6.2). |
| `model.get_feature_extractor()` | Return a **new, randomly initialized** twin with `include_top=False, pooling="cls"`. It does not copy this instance's weights. |
| `model.summary_detailed()` | Log scale, patch grid, sequence length, dimensions and parameter count. |
| `model.load_pretrained_weights(path)` | Load a local `.keras` checkpoint layer by layer. |

### 6.2 `create_vit` and `from_variant`

`create_vit` is the factory; its first argument is the variant key and it routes through `ViT.from_variant`.

```python
from dl_techniques.models.vision.vit import ViT, create_vit

print(sorted(ViT.MODEL_VARIANTS.keys()))
# ['vit_base', 'vit_huge', 'vit_large', 'vit_pico', 'vit_small', 'vit_tiny']

# Factory, CIFAR-sized.
model = create_vit("vit_pico", num_classes=10, input_shape=(32, 32, 3), patch_size=4)

# Classmethod, equivalent entry point.
model = ViT.from_variant("vit_pico", num_classes=10, input_shape=(32, 32, 3), patch_size=4)
```

### 6.3 Pretrained weights

There are **no hosted checkpoints** for this implementation. `pretrained=True` raises `NotImplementedError` rather than silently returning a randomly initialized model:

```python
try:
    ViT.from_variant("vit_base", pretrained=True)
except NotImplementedError as exc:
    print(exc)
    # No public ViT checkpoints are distributed for this implementation. ...
```

A local checkpoint path works:

```python
model = ViT.from_variant("vit_base", num_classes=1000,
                         pretrained="/path/to/vit_base.keras")
```

Weight loading goes through `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint` (full load plus layer-by-layer `set_weights` after a probe forward pass) rather than `model.load_weights(by_name=True)`, which Keras 3.8+ rejects for `.keras` files.

---

## 7. Configuration & Model Variants

Parameter counts below were measured by building each model at `input_shape=(224, 224, 3)`, `patch_size=16`, `num_classes=1000`.

| Scale | Variant key | Embed dim | Heads | Layers | MLP ratio | Params |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `pico` | `vit_pico` | 192 | 3 | 6 | 4.0 | 3,048,232 |
| `tiny` | `vit_tiny` | 192 | 3 | 12 | 4.0 | 5,717,416 |
| `small` | `vit_small` | 384 | 6 | 12 | 4.0 | 22,050,664 |
| `base` | `vit_base` | 768 | 12 | 12 | 4.0 | 86,567,656 |
| `large` | `vit_large` | 1024 | 16 | 24 | 4.0 | 304,326,632 |
| `huge` | `vit_huge` | 1280 | 16 | 32 | 4.0 | 632,199,400 |

Guidance:

- `pico` / `tiny` / `small`: small-image experiments and fine-tuning on medium datasets.
- `base`: the usual choice for ImageNet-scale pre-training and transfer.
- `large` / `huge`: only worth training with very large datasets; otherwise use them via transfer learning.

---

## 8. Usage Examples

### Example 1: feature extraction

```python
import numpy as np
from dl_techniques.models.vision.vit.model import ViT

extractor = ViT(
    input_shape=(224, 224, 3),
    scale="small",
    patch_size=16,
    include_top=False,
    pooling="cls",     # one vector per image
)

images = np.random.rand(8, 224, 224, 3).astype("float32")
features = extractor.predict(images, verbose=0)
print(features.shape)  # (8, 384)
```

Use `pooling=None` instead to get the full `(B, num_patches + 1, embed_dim)` token sequence for a dense-prediction head.

### Example 2: frozen backbone plus a new head

```python
import keras
from dl_techniques.models.vision.vit.model import ViT

backbone = ViT(
    input_shape=(224, 224, 3),
    scale="small",
    patch_size=16,
    include_top=False,
    pooling="cls",
)
# backbone.load_pretrained_weights("/path/to/vit_small.keras")
backbone.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = backbone(inputs, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, name="new_head")(x)
finetune = keras.Model(inputs, outputs)

finetune.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

---

## 9. Advanced Usage Patterns

### Swapping block internals

```python
from dl_techniques.models.vision.vit.model import ViT

modern_vit = ViT(
    input_shape=(256, 256, 3),
    num_classes=100,
    scale="small",
    patch_size=16,
    normalization_type="rms_norm",
    ffn_type="swiglu",
    normalization_position="pre",
    use_layer_scale=True,
)
```

### Reading every block's output

`model.transformer_layers` is a plain Python list, so intermediate activations can be collected by running the stack manually.

```python
import keras
from dl_techniques.models.vision.vit.model import ViT

vit = ViT(input_shape=(32, 32, 3), scale="pico", patch_size=4, include_top=False)
vit(keras.ops.zeros((1, 32, 32, 3)))  # build

x = keras.ops.zeros((2, 32, 32, 3))
x = vit.patch_embed(x)
cls = keras.ops.broadcast_to(vit.cls_token, (2, 1, vit.embed_dim))
x = keras.ops.concatenate([cls, x], axis=1)
x = vit.pos_embed(x)

per_layer = []
for layer in vit.transformer_layers:
    x = layer(x)
    per_layer.append(x)
print(len(per_layer), per_layer[-1].shape)  # 6 (2, 65, 192)
```

---

## 10. Performance and Training Notes

### Mixed precision

```python
keras.mixed_precision.set_global_policy("mixed_float16")
```

Set the policy before constructing the model; every layer then picks it up.

### Training recipe

- **Optimizer**: AdamW. Weight decay does real work here.
- **Schedule**: cosine decay with a linear warmup. Warmup matters more with the default `normalization_position="post"`.
- **Augmentation**: ViTs lean on it. RandAugment, Mixup and CutMix are the usual set.
- **Resolution**: pre-train at 224 and fine-tune higher. Positional embeddings are learned per position, so changing resolution changes the sequence length and the embeddings must be interpolated or re-learned.

---

## 11. Serialization

The model and its layers are registered for the `.keras` format.

```python
model.save("my_vit.keras")
loaded = keras.models.load_model("my_vit.keras")
```

---

## 12. Troubleshooting

- `ValueError: Image height (...) must be divisible by patch height (...)` -- resize the input or change `patch_size`.
- `ValueError: Unsupported scale: ...` -- `scale` takes `pico|tiny|small|base|large|huge`; the `vit_*` keys belong to `from_variant`/`create_vit`, not to the constructor.
- `pretrained=True` raises `NotImplementedError` -- expected, since no public pretrained weights are hosted; pass a local `.keras` path instead.
- Loss stuck near `ln(num_classes)` on a small dataset -- ViTs from scratch need augmentation, weight decay and warmup, or pre-trained weights.
- Accuracy far below expectation with a correct-looking setup -- check that the loss uses `from_logits=True`; the head is unactivated.
- Out of memory at high resolution -- attention is quadratic in patch count. Increase `patch_size` before reducing batch size.

---

## 13. Technical Details

**Positional embeddings.** Learned, absolute, 1D, one vector per sequence position, added to the patch embeddings. Not implemented here: 2D factorized embeddings, relative position biases, and fixed sinusoidal embeddings.

**Initialization.** `kernel_initializer=None` resolves to `TruncatedNormal(stddev=0.02)`, matching the reference ViT convention (HuggingFace's `ViTConfig.initializer_range`). It is passed as a config dict, not an `Initializer` instance, so that each layer draws its own weights.

**Deep supervision.** Not implemented. `ViT` has no `enable_deep_supervision` parameter and no deep-supervision code path.

---

## 14. Citation

```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

Related work referenced by the implementation:

- Vaswani et al., 2017. Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Touvron et al., 2021. Training data-efficient image transformers & distillation through attention. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Xiong et al., 2020. On Layer Normalization in the Transformer Architecture. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
