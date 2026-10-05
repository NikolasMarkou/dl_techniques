# AlexNet

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

The original AlexNet (Krizhevsky, Sutskever & Hinton, NeurIPS 2012), the network
that started the deep-learning era in computer vision — including its **local
response normalization**, which is the part most modern ports quietly drop.

```python
from dl_techniques.models.vision.alexnet import AlexNet, create_alexnet

model = create_alexnet(num_classes=1000)
```

---

## 1. What AlexNet was, and why it still matters

AlexNet won ImageNet 2012 by a margin that ended the era of hand-built features: top-5
error **15.3%** against **26.2%** for the runner-up. Four ideas carried it, and all
four are still load-bearing in networks written today:

1. **ReLU instead of saturating tanh/sigmoid.** Faster to optimize, and it is why
   deep nets train at all.
2. **Dropout** (0.5) in the fully connected layers, to suppress co-adaptation.
3. **Data augmentation** — 224×224 crops from 256×256 images, plus horizontal flips
   and PCA-based colour jitter.
4. **GPU parallelism** — the model was split across two GPUs.

Local response normalization (LRN) is in this list as a *historical* entry: it was
part of the original design and is reproduced here, but it did not survive contact
with Batch Normalization.

---

## 2. The architecture

```
227 × 227 × 3
  │
  ├─ conv1  11×11 / 4   pad 2    96            →  56 × 56 × 96
  ├─ ReLU
  ├─ LRN    (n=5, α=1e-4, β=0.75, k=1)           →  56 × 56 × 96
  ├─ pool1   3×3 / 2    pad 1                    →  28 × 28 × 96
  │
  ├─ conv2   5×5 / 1   pad 1   256 (groups=2)   →  26 × 26 × 256
  ├─ ReLU
  ├─ LRN                                     →  26 × 26 × 256
  ├─ pool2   3×3 / 2    pad 1                    →  13 × 13 × 256
  │
  ├─ conv3   3×3 / 1   pad 1   384 (groups=2)   →  13 × 13 × 384
  ├─ ReLU
  ├─ conv4   3×3 / 1   pad 1   384 (groups=2)   →  13 × 13 × 384
  ├─ ReLU
  ├─ conv5   3×3 / 1   pad 1   256 (groups=2)   →  13 × 13 × 256
  ├─ ReLU
  ├─ pool3   3×3 / 2    pad 0                    →   6 × 6 × 256
  │
  ├─ flatten 9216
  ├─ fc6     4096 → ReLU → dropout 0.5
  ├─ fc7     4096 → ReLU → dropout 0.5
  └─ fc8     1000 → softmax
```

Every spatial extent above is **measured**, not copied from the paper's diagram, and
each one matches it.

### Parameter breakdown

| Layer | Output shape | Parameters |
|---|---|---:|
| `conv1` | (None, 56, 56, 96) | 34,944 |
| `lrn1` | (None, 56, 56, 96) | 9,216 |
| `pool1` | (None, 28, 28, 96) | 0 |
| `conv2` | (None, 26, 26, 256) | 307,456 |
| `lrn2` | (None, 26, 26, 256) | 65,536 |
| `pool2` | (None, 13, 13, 256) | 0 |
| `conv3` | (None, 13, 13, 384) | 442,752 |
| `conv4` | (None, 13, 13, 384) | 663,936 |
| `conv5` | (None, 13, 13, 256) | 442,624 |
| `pool3` | (None, 6, 6, 256) | 0 |
| `flatten` | (None, 9216) | 0 |
| `fc6` | (None, 4096) | 37,752,832 |
| `fc7` | (None, 4096) | 16,781,312 |
| `fc8` | (None, 1000) | 4,097,000 |
| **Total** | | **60,597,608** |

Three things about those numbers are worth stating plainly, because each one is
usually reported incorrectly:

- **The folklore figure is high.** "About 62 million" is in wide circulation; the
  exact sum for this architecture at this configuration is **60,597,608**. This
  package prints its own measured figure rather than the folklore one.
- **`lrn1`/`lrn2` are not free.** Each LRN holds its `C × C` neighbourhood band as a
  **non-trainable** weight — 96² and 256², i.e. **74,752** parameters of the total.
  A layer with no learnable values still costs memory here. They do not appear in
  `trainable_parameters`.
- **`conv3` reads 442,752, not 885,120.** It is grouped like the rest: 384 filters ×
  (256/2) input channels × 9 + bias. See below.

---

## 3. Padding: the part the paper does not tell you

**The paper never states its padding.** It lists kernel sizes and strides and omits
the parameter entirely. This package therefore transcribes the **released Caffe
model**, which is the only place the specification survives — and then verifies that
the transcription reproduces the paper's own Figure 3.

That verification is not a formality. Measured at input 227:

| Padding scheme | conv5 out | pool3 out | fc6 `in_features` | Matches paper? |
|---|---|---|---|---|
| **Caffe (shipped here)** | 13×13 | **6×6** | **9216** | ✅ |
| Strict `valid` everywhere | 5×5 | 2×2 | 1024 | ❌ |
| All-`same` convolutions | 14×14 | 7×7 | 12544 | ❌ |

Only the first reproduces the `6 × 6 × 256` that Figure 3 labels.

### The Keras `'same'` trap

Keras's `padding='same'` is **not** the Caffe padding, and using it silently
produces a different network:

- **TF/Keras `'same'`** pads to `ceil(in/stride)`, splitting an odd total unevenly.
  Measured: `conv1` with kernel 11 / stride 4 at input 227 gives **57**, not 56.
- **Caffe** pads by a fixed symmetric amount, giving **56**.

This port therefore uses **explicit `ZeroPadding2D` layers** in front of every
padded convolution and pool, rather than the `padding` argument. The asymmetry of the
released model — padded pools, **unpadded** final pool — is exactly what produces
the paper's 6×6.

### About the 224 vs 227 question

227 is the paper's number (a crop of a 256-pixel image) and is kept as the default.
It is **not** a magic value, though: measured with this padding, inputs **224, 225,
226 and 227 all reach 6 × 6 × 256** and the same 9216-wide flatten, because the
strides round down to the same extent. 256 gives 7 × 7, and fc6's input becomes
12544. So the frequently-repeated claim that "AlexNet needs 227, not 224" is
**wrong** for this configuration — 224 works and gives an identical shape.

---

## 4. Local Response Normalization

LRN makes neighbouring channels compete: each channel is divided by the squared
energy of a window of channels centred on it.

```
Y_c = X_c / (k + α · Σ_{j=c-r}^{c+r} X_j²)^β
```

with the paper's `r = 2` (so `n = 5`), `α = 1e-4`, `β = 0.75`, `k = 1`. It follows
**conv1 and conv2 only, after the ReLU and before the pool**.

It is provided by
[`LocalResponseNormalization`](../../layers/norms/local_response_norm.py) and reached
through the norms factory:

```python
from dl_techniques.layers.norms import create_normalization_layer

lrn = create_normalization_layer('local_response_norm',
                                depth_radius=2, alpha=1e-4, beta=0.75, k=1.0)
```

Verified bit-exact against `tf.nn.local_response_normalization` — the numerical
oracle is in `tests/test_layers/test_norms/test_local_response_norm.py`.

### LRN vs GRN — not the same thing

|  | `LocalResponseNormalization` | `GlobalResponseNormalization` |
|---|---|---|
| Paper | AlexNet 2012 | ConvNeXt V2 |
| Neighbourhood | **local**, across channels | **global**, over all positions |
| Operation | **divides** | multiplies, then adds input back |
| Weights | none learnable | trainable `gamma`, `beta` |

Neither substitutes for the other despite the similar names.

### Is it worth keeping?

Honestly, **no** — not for a model you intend to train. Jain & Wallace (2013) showed
LRN performs comparably to Batch Normalization while being far slower and harder to
tune; BatchNorm won. LRN is here because the request was the *original* AlexNet, and
a port that drops it is not the original AlexNet.

---

## 5. Grouped convolutions are not decoration

`conv2`–`conv5` use `groups=2`. The paper splits each across two GPUs, and each
output channel there connects to only **half** the preceding feature map. On one
device `groups=2` is the standard expression of that, and it changes the
architecture: `conv3` takes 384 × (256/2) × 9 + 384 parameters rather than
384 × 256 × 9 + 384.

---

## 6. Quick start

```python
from dl_techniques.models.vision.alexnet import create_alexnet

model = create_alexnet(num_classes=1000, input_shape=(227, 227, 3))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(images, labels, epochs=90, batch_size=256)
```

### Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `num_classes` | 1000 | Width of the final softmax |
| `input_shape` | (227, 227, 3) | Spatial shape plus channels |
| `dropout_rate` | 0.5 | After fc6 and fc7 |
| `kernel_initializer` | `'glorot_uniform'` | All five convolutions and the dense layers |
| `bias_initializer` | `'zeros'` | Every bias |
| `include_top` | `True` | `False` stops at `pool3` and returns a feature map |
| `lrn_hyperparameters` | paper's | Forwarded to both LRN layers |

### No variant table

This package has **no `MODEL_VARIANTS` table**, deliberately. AlexNet is one
architecture with one configuration, and `models/AGENTS.md` ("When the shape does not
apply") says not to invent a table to satisfy a template.

---

## 7. `include_top=False` as a feature extractor

```python
backbone = create_alexnet(include_top=False)
features = backbone(images)   # (batch, 6, 6, 256)
```

This drops the 58.6M fully connected parameters and leaves **1,966,464**. It is the
form to reach for when you want AlexNet's convolution stack as a generic encoder.

Note that the resulting map is small — 6×6 at input 227. For dense prediction you
want a larger input, not a smaller model.

---

## 8. Pretrained weights

**None ship with this port.** `create_alexnet(pretrained=True)` raises
`NotImplementedError` rather than quietly returning random weights under that name,
and `AlexNet.load_pretrained_weights(path)` raises too.

If you need weights, convert an external Caffe checkpoint yourself and load it by
hand. The parameter names above map onto the Caffe blobs directly
(`conv1`/`fc6`/…), and note the two frictions: the fc6 kernel is
`(9216, 4096)` here against Caffe's transposed `(4096, 9216)`, and the LRN bands are
constants this port adds that a Caffe file will not contain.

---

## 9. Training

Trainer: `src/train/alexnet/train_alexnet.py` (Pattern 1, vision classification).

```bash
MPLBACKEND=Agg .venv/bin/python -m train.alexnet.train_alexnet \
    --dataset cifar10 --epochs 90 --batch-size 128
```

The paper's recipe is **90 epochs, SGD with momentum 0.9, learning rate 0.1 halved
every 30 epochs, weight decay 5e-4, batch size 256**. On modern hardware that
schedule is aggressive; expect to tune it.

> **Never double weight decay.** `AdamW` applies decoupled decay internally, so do not
> *also* pass `kernel_regularizer=L2(...)`.

---

## 10. Performance and cost

- **60.6M parameters, 97% of them fully connected.** fc6 alone is 37.8M. This is the
  architecture's defining cost and the reason it is rarely trained from scratch today.
- **Memory during training** is dominated by activations at 56×56×96 and
  26×26×256, not by parameters.
- **No pretrained weights and no BNN-heavy ops**: expect roughly 5–10 s/epoch on a
  modern GPU at 227×227 with batch 256. CPU training is not practical.
- **Input cost.** At 227×227 with no resizing this is one of the most expensive
  classic CNNs per sample; a smaller input (160, 128) cuts it sharply and changes
  the feature map to 4×4 or 3×3.

---

## 11. What this port does *not* claim

- **Not a state-of-the-art model.** Its ImageNet top-5 error was 15.3%; a modern
  ResNet is around 3%. Use it to study the architecture, not to win a benchmark.
- **Not trained here.** No run was launched against this port, so no accuracy claim is
  made. The parameter counts and shapes above are all measured from the code.
- **Not the tracking variant.** `models/vision/siamfc` and `models/vision/dasiamrpn`
  each inline an AlexNet-*shaped* trunk with `valid` padding throughout, BatchNorm
  instead of LRN, and no LRN at all — the tracking papers ablated it out. Those are
  separate implementations and were deliberately left untouched rather than refactored
  onto this package.

---

## 12. Citation

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title     = {ImageNet Classification with Deep Convolutional Neural Networks},
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
  booktitle = {Advances in Neural Information Processing Systems 25},
  year      = {2012}
}
```

Padding transcribed from the released Caffe model; LRN hyper-parameters from
Section 4. Also cited for the LRN ablation:
Jain & Wallace, *Supervised Learning of Image Restoration with Convolutional
Networks*, arXiv:1307.3065.