# ResNet - Deep Residual Networks

A Keras 3 implementation of **ResNet** (He, Zhang, Ren & Sun, *Deep Residual Learning for
Image Recognition*, CVPR 2016, [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) — the
architecture that made very deep convolutional networks trainable by adding an identity
shortcut around every pair or triple of convolutions.

`ResNet` is a **subclassed** `keras.Model`, not a Functional graph. That one fact explains
most of the surprises below: no `.input`, no `.output`, no `output_names`, and
`count_params()` raises until the model is built.

> **`pretrained=True` raises `NotImplementedError` — no weights are downloadable.**
> `_download_weights` raises by design. The working form is a local checkpoint path:
> `ResNet.from_variant('resnet50', pretrained='/path/to/file.keras')`. See § 9.

## 1. Overview: What is ResNet and Why It Matters

A plain deep convolutional stack computes `x -> H(x)`. ResNet reparameterizes each block to
compute a **residual** instead: `y = F(x, W) + x`, where `F` is 2 or 3 convolutions and `x`
arrives on a parameter-free identity shortcut. Nothing else changes — same convolutions,
normalization and activations. What it buys is that "do nothing" becomes a *reachable*
solution: driving `F` to zero is easy, driving a stack of convolutions to the identity is not.

152 layers trains where a 34-layer plain network does not, and the residual path became the
backbone of every later vision architecture. Feature maps from a trained ResNet transfer
well, which is why `include_top=False` is the most common way this class is used.

## 2. The Degradation Problem

Adding layers to a plain network makes **training** error go up, not just test error. That is
not overfitting — it is an optimization failure.

Two mechanisms combine. **Vanishing gradients**: backpropagation multiplies Jacobians layer by
layer, so with a typical per-layer factor `k < 1` the gradient reaching layer 1 of an
`L`-layer stack scales like `k^L` (at `k = 0.9`, `L = 50`: `~0.005`). **Unreachable
identity**: if a 20-layer network is already good, a 56-layer one only needs its extra layers
to compute the identity, and `conv -> BN -> ReLU` cannot easily represent it. Differentiating
`y = F(x) + x` gives `dy/dx = dF/dx + I` — an ungated path along which the gradient reaches
every earlier layer undiminished, and `F = 0` gives the identity exactly.

## 3. How ResNet Works: The Residual Learning Framework

A block learns the *change* to apply to its input rather than a whole new representation (§ 4.1
draws one). Three properties follow: every block contributes an additive `I` to the chain rule, so
the gradient reaches the stem unattenuated; zero weights in `F` give `y = x`, so depth can only
help; and unrolling the sum over `n` blocks yields `2^n` input-to-output paths of varying depth,
most of them short.

Full data flow at the ImageNet default (`stem_type='imagenet'`, 224x224 input):

```
  input (224, 224, 3)
    stem     conv 7x7 s2 -> norm -> act -> maxpool 3x3 s2  -> (56, 56, 64)
    stage 1  blocks_per_stage[0] blocks, stride 1          -> (56, 56, 64|256)
    stage 2  blocks_per_stage[1] blocks, first stride 2    -> (28, 28, 128|512)
    stage 3  blocks_per_stage[2] blocks, first stride 2    -> (14, 14, 256|1024)
    stage 4  blocks_per_stage[3] blocks, first stride 2    -> ( 7,  7, 512|2048)
    head     global average pool -> dense(num_classes)     -> (num_classes,) LOGITS
```

Channel counts read `basic|bottleneck`: a bottleneck block emits `4x` its nominal filters.

## 4. Architecture Deep Dive

### 4.1 The two block types

```
  BasicBlock  (block_type='basic', ResNet-18/34)     -> F output channels
   x ─► conv 3x3 (F) ─► norm ─► act ─► conv 3x3 (F) ─► norm ─► (+) ─► act
   └──────────────────── shortcut ─────────────────────────────┘

  BottleneckBlock  (block_type='bottleneck', ResNet-50/101/152) -> 4F channels
   x ─► conv 1x1 (F) reduce ─► conv 3x3 (F) ─► conv 1x1 (4F) expand ─► (+) ─► act
   └──────────────────── shortcut ──────────────────────────────────┘
```

The bottleneck runs the expensive 3x3 at `F` channels instead of `4F`: at `F=64` its three
convolutions cost `4,096 + 36,864 + 16,384 = 57,344` weights against `589,824` for two 3x3
convolutions at 256 channels. That is why a 50-layer bottleneck network costs about the same
as a 34-layer basic one while being much deeper.

### 4.2 Projection shortcuts

The addition needs matching shapes, so a `1x1` convolution carrying the block's stride is
inserted on the shortcut whenever input and output shapes differ: at the first block of every
stage, and at the first block of stage 1 for bottlenecks (channels `64 -> 256`). Every other
block uses a parameter-free identity.

### 4.3 Downsampling and the stem

Resolution halves at the stride-2 convolution in the first block of stages 2, 3 and 4. The
stem is selected by `stem_type`:

| `stem_type` | Stem | Total downsampling | For |
|:---|:---|:---:|:---|
| `'imagenet'` (default) | conv 7x7 s2 + maxpool 3x3 s2 | 32x | 224x224-class inputs |
| `'cifar'` | conv 3x3 s1, no pool | 8x | 32x32 inputs (He et al. § 4.2) |

**This is the single most common way to get a silently broken ResNet.** Measured on this repo,
`resnet18` at `(1, 32, 32, 3)` with `include_top=False` reaches the global pool at
`(1, 1, 1, 512)` under `'imagenet'` and `(1, 4, 4, 512)` under `'cifar'` — under the default
the last two stages stride a feature map that has already collapsed to one pixel. Use
`stem_type='cifar'` for small images.

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.vision.resnet import create_resnet

model = create_resnet("resnet50", num_classes=1000, input_shape=(224, 224, 3))
# `count_params()` needs the model BUILT -- ResNet is subclassed, so no weights
# exist until a shape is known ("ValueError: You tried to call `count_params` on
# layer 'res_net', but the layer isn't built").
model.build((None, 224, 224, 3))
print(f"{model.count_params():,}")          # 25,610,152

logits = model(np.random.rand(2, 224, 224, 3).astype("float32"), training=False)
print(logits.shape)                         # (2, 1000)

# The classifier is a bare Dense with NO activation, so this holds LOGITS.
# Apply softmax before reading anything as a probability.
probs = keras.ops.softmax(logits)
```

CIFAR-scale, with the stem that matches the input size (§ 4.3):

```python
model = create_resnet(
    "resnet18", num_classes=10, input_shape=(32, 32, 3), stem_type="cifar"
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
x = np.random.rand(32, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, (32,))
model.fit(x, y, epochs=1, batch_size=8, verbose=0)
```

## 6. Model Variants

`ResNet.MODEL_VARIANTS` holds exactly five keys.

| Variant | `blocks_per_stage` | `block_type` | `count_params()` @ 1000 classes | Notes |
|---|---|---|---:|---|
| `resnet18` | `[2, 2, 2, 2]` | basic | 11,699,112 | cheapest |
| `resnet34` | `[3, 4, 6, 3]` | basic | 21,814,696 | |
| `resnet50` | `[3, 4, 6, 3]` | bottleneck | 25,610,152 | most common |
| `resnet101` | `[3, 4, 23, 3]` | bottleneck | 44,654,504 | |
| `resnet152` | `[3, 8, 36, 3]` | bottleneck | 60,344,232 | deepest |

All five use `filters_per_stage = [64, 128, 256, 512]`. § 17 splits the parameter column into
trainable and non-trainable; it is `count_params()` after `build((None, 224, 224, 3))`, an
architectural fact rather than a measurement on a checkpoint.

> **No accuracy numbers are quoted anywhere in this README, and that is deliberate.** This
> table used to carry `Top-1 Acc` / `Top-5 Acc` columns and a per-variant
> `Performance (ImageNet)` block, all of them He et al.'s published figures presented as
> though they described this code. **No pretrained weights exist for any architecture in this
> repository** — `_download_weights` raises `NotImplementedError` by design — so nothing here
> has ever been evaluated on ImageNet. For reference numbers read the paper; for numbers about
> *this* code, train it and measure.

### Constructor arguments

`ResNet(...)`, `ResNet.from_variant(variant, ...)` and `create_resnet(variant, ...)` all
accept these.

| Argument | Default | Meaning |
|:---|:---|:---|
| `num_classes` | `1000` | head width; ignored when `include_top=False` |
| `blocks_per_stage` | `[3, 4, 6, 3]` | blocks per stage (set by the variant) |
| `filters_per_stage` | `[64, 128, 256, 512]` | nominal filters per stage; also sets the stem width |
| `block_type` | `'bottleneck'` | `'basic'` or `'bottleneck'` |
| `stem_type` | `'imagenet'` | `'imagenet'` or `'cifar'` — see § 4.3 |
| `input_shape` | `(224, 224, 3)` | must be a 3-tuple |
| `include_top` | `True` | `False` returns the `(B, H, W, C)` feature map |
| `enable_deep_supervision` | `False` | adds per-stage auxiliary heads — § 7 |
| `normalization_type` | `'batch_norm'` | any key of the normalization factory |
| `normalization_kwargs` | `None` | forwarded to every norm; `batch_norm` gets `momentum=0.9` unless you override it |
| `activation_type` | `'relu'` | activation factory key |
| `kernel_regularizer` | `None` | applied to every convolution |

`from_variant` / `create_resnet` additionally take `pretrained` (§ 9), `weights_dataset`,
`weights_input_shape` and `cache_dir`. The `momentum=0.9` default is deliberate: Keras and
PyTorch define BatchNorm momentum oppositely (`keras_momentum = 1 - torch_momentum`), so
torchvision's `0.1` *is* Keras' `0.9`. Do not "correct" it.

## 7. Deep Supervision Feature

With `enable_deep_supervision=True` (and `include_top=True`) each stage gets its own
classification head, injecting gradient into the middle of the network. The model returns a
**list** of logit tensors: primary head first, then the auxiliary heads deepest-first.

Because `ResNet` is subclassed it has **no output names**, so losses, weights and metrics are
given **positionally, in lists**. A dict keyed by `'output_0'` resolves against `None` and
raises `TypeError`.

```python
import keras
import numpy as np
from dl_techniques.models.vision.resnet import create_resnet
from dl_techniques.utils.deep_supervision import (
    create_inference_model_from_training_model,
    get_model_output_info,
)

model = create_resnet("resnet18", num_classes=10, input_shape=(32, 32, 3),
                      stem_type="cifar", enable_deep_supervision=True)

# `model.output` RAISES even after the model has been called. Ask the helper
# instead; it needs the per-sample input_shape to trace `call()`.
num_outputs = get_model_output_info(model, input_shape=(32, 32, 3))["num_outputs"]
print(num_outputs)                              # 4

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    # One loss per output, positionally. The heads emit LOGITS.
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)] * num_outputs,
    # Index 0 is the primary (deepest) head.
    loss_weights=[1.0] + [0.5 / (num_outputs - 1)] * (num_outputs - 1),
    # NAMING the primary metric is what makes it monitorable: there is no
    # 'val_output_0_accuracy' on a model with no output names.
    metrics=(
        [[keras.metrics.SparseCategoricalAccuracy(name="primary_accuracy")]]
        + [[]] * (num_outputs - 1)
    ),
)

x = np.random.rand(8, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, (8,))
# Labels replicated once per output -- a list, never a dict. With tf.data:
#     ds.map(lambda x, y: (x, tuple([y] * num_outputs)))
history = model.fit(x, [y] * num_outputs, epochs=1, batch_size=4, verbose=0)
print(sorted(history.history))
# ['loss', 'primary_accuracy', 'sparse_categorical_crossentropy_loss']

# Single-output model for inference. `input_shape` is required for the same reason.
inference = create_inference_model_from_training_model(model, input_shape=(32, 32, 3))
print(inference(x).shape)                       # (8, 10)
```

For loss weighting, decay from deep to shallow (`[1.0, 0.3, 0.2, 0.1]`) or ramp the auxiliary
weights toward zero over training. Changing them needs a **re-`compile()`**; mutating the list
in place does nothing.

## 8. Usage Examples

### Example 1: Feature extraction and a new head

Wrap the backbone in a Functional model, starting from a `keras.Input` you own: a subclassed
model has no `.input` to re-wire from.

```python
import keras
import numpy as np
from dl_techniques.models.vision.resnet import create_resnet

base = create_resnet("resnet18", num_classes=10, input_shape=(32, 32, 3),
                     stem_type="cifar", include_top=False)
inputs = keras.Input(shape=(32, 32, 3))
features = keras.layers.GlobalAveragePooling2D()(base(inputs))
outputs = keras.layers.Dense(10, name="new_predictions")(features)
model = keras.Model(inputs, outputs)

print(len(model.layers), [l.name for l in model.layers])
# 4 ['input_layer', 'res_net', 'global_average_pooling2d', 'new_predictions']
# (Keras appends a numeric suffix when other models were built earlier in the same
# process; the LENGTH is the point.) The wrapper holds the whole backbone as ONE
# layer, so freeze parts of it through `base.layers` -- not `model.layers`.
for layer in base.layers[:6]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
x = np.random.rand(8, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, (8,))
model.fit(x, y, epochs=1, verbose=0)
print(len(model.trainable_weights))             # 38
```

Stage two: `base.trainable = True`, re-set the per-layer flags you cleared, drop the learning
rate 10x, **re-compile** — a `trainable` change has no effect until you do.

### Example 2: Custom architecture

Pass `blocks_per_stage` / `filters_per_stage` instead of a variant name; same length required.

```python
from dl_techniques.models.vision.resnet import ResNet

model = ResNet(
    num_classes=5, blocks_per_stage=[2, 3, 4, 2],
    filters_per_stage=[32, 64, 128, 256], block_type="bottleneck",
    stem_type="cifar", activation_type="gelu", input_shape=(64, 64, 3),
)
model.build((None, 64, 64, 3))
```

## 9. Pretrained Weights & Transfer Learning

> **`pretrained=True` raises `NotImplementedError`. There are no ResNet weights to download,
> from this repository or anywhere else in a format this package loads.** Every model package
> in this repository that exposes a `pretrained=` argument behaves the same way.

The only working form is a local checkpoint path:

```python
from dl_techniques.models.vision.resnet import ResNet

CKPT = "/path/to/resnet50.keras"    # something you trained and saved

model = ResNet.from_variant("resnet50", pretrained=CKPT, num_classes=1000)

# A different head width is fine: the incompatible classifier layer is SKIPPED,
# not refused. `include_top=False` gives the bare feature extractor.
small_head = ResNet.from_variant("resnet50", pretrained=CKPT, num_classes=100)
backbone = ResNet.from_variant("resnet50", pretrained=CKPT, include_top=False)

# DOES NOT WORK -- raises NotImplementedError, no download exists:
#   ResNet.from_variant("resnet50", pretrained=True, weights_dataset="imagenet")
```

Once you have a checkpoint the three usual strategies apply: freeze everything and train a new
head; freeze the stem and early stages and fine-tune the rest at `1e-4`; or unfreeze
everything at `1e-5`. Each needs a re-`compile()`. Section 8 Example 1 runs the first two end
to end. No accuracy benefit is quoted here, for the same reason as in § 6.

## 10. Training from Scratch

```python
import keras
from dl_techniques.models.vision.resnet import create_resnet

model = create_resnet("resnet18", num_classes=10, input_shape=(32, 32, 3),
                      stem_type="cifar",
                      kernel_regularizer=keras.regularizers.L2(1e-4))
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=50_000,
    warmup_target=1e-2, warmup_steps=2_000,
)
model.compile(
    optimizer=keras.optimizers.SGD(schedule, momentum=0.9, nesterov=True),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

Augmentation matters more than architecture here: random crop with padding, horizontal flip,
and for long runs mixup or cutout. Weight decay belongs in the optimizer **or** in
`kernel_regularizer`, never both. Add `ModelCheckpoint` / `EarlyStopping` as usual.

## 11. Fine-Tuning Strategies

**Route 1 (recommended): staged unfreezing.** Layer names are `stem_conv`/`stem_bn`/`stem_act`,
`stageN_blockM`, `global_avg_pool` and `classifier`, so grouping `model.layers` by prefix is a
one-liner. Freeze the early groups, train, then unfreeze and re-compile at a lower learning
rate. Section 8 Example 1 runs this.

**Route 2: two optimizers over disjoint variable sets** in a custom training loop. This
repository's convention is to avoid a custom `train_step`; prefer route 1.

## 12. Advanced Techniques

- **Stochastic depth and squeeze-and-excitation** are not built into
  `BasicBlock`/`BottleneckBlock`; both ship as layers you can compose into a custom block.
  `normalization_type` takes any normalization-factory key and `normalization_kwargs` reaches
  every site.
- **Register custom layers** with `@register_dl_technique("dl_techniques.<module.path>")` — a
  package-qualified key. Never a bare `@keras.saving.register_keras_serializable()`: its
  `Custom>ClassName` key carries no module path, so two same-named classes claim one slot.

## 13. Performance Optimization

| Lever | Effect |
|:---|:---|
| `keras.mixed_precision.set_global_policy("mixed_float16")` | roughly halves activation memory |
| `model.compile(..., jit_compile=True)` | XLA fusion; measure, not always faster |
| smaller `input_shape` | quadratic in activation memory |
| `include_top=False` + cached features | frozen-backbone training becomes a linear probe |

The input pipeline is usually the bottleneck before the model: `tf.data` with `.cache()`,
`.prefetch(AUTOTUNE)` and a parallel `map` first.

## 14. Serialization & Deployment

```python
import keras
import numpy as np
from dl_techniques.models.vision.resnet import create_resnet

model = create_resnet("resnet18", num_classes=10, input_shape=(32, 32, 3),
                      stem_type="cifar")
model.build((None, 32, 32, 3))
model.save("resnet18.keras")

restored = keras.models.load_model("resnet18.keras")

x = np.random.rand(2, 32, 32, 3).astype("float32")
assert np.allclose(np.asarray(model(x)), np.asarray(restored(x)), atol=1e-6)
```

`get_config()` carries every constructor argument, `stem_type` included. `save_weights` requires
the target model to be built first. `ResNet` and its blocks register under
`dl_techniques.models.resnet.model>ClassName`, so a `.keras` file loads with no `custom_objects`.

## 15. Testing & Validation

`pytest tests/test_models/test_resnet/ -q`. Four files pin this README specifically:
`test_the_readme_flops_table_reproduces.py` (§ 17's numbers),
`test_the_readme_deep_supervision_pattern_runs.py` (§ 7 and § 8 execute as written),
`test_stem_type.py` (both stems, against a golden reference) and `test_model.py` (fails if an
accuracy claim reappears here).

## 16. Troubleshooting & FAQs

- **`You tried to call count_params on layer 'res_net', but the layer isn't built`.**
  Subclassed model: call `model.build((None, H, W, C))` or run one forward pass first. Same
  cause for `summary()`.
- **`The layer res_net has never been called and thus has no defined output`.** There is no
  `.input`, `.output` or `.output_names`. Use
  `dl_techniques.utils.deep_supervision.get_model_output_info(model, input_shape=...)`, and
  wrap the model from a `keras.Input` you create.
- **Predictions look like unbounded scores, sometimes negative.** They are logits: the
  classifier is a bare `Dense` with no activation. Compile with `from_logits=True` and apply
  `keras.ops.softmax` before reading a probability.
- **Accuracy stuck near chance on 32x32 images.** You are on the default `stem_type='imagenet'`
  and the feature map collapsed to 1x1 before stage 3. Pass `stem_type='cifar'`.
- **`NotImplementedError` from `pretrained=True`.** By design. Use a local path (§ 9).
- **`model.layers.pop()` appears to work and changes nothing.** `layers` is a recomputed
  property, so `pop()` mutates a throwaway list. Use `include_top=False` instead.
- **Freezing had no effect.** You must `compile()` again after changing `trainable`.
- **`Length of blocks_per_stage must equal length of filters_per_stage`.** The two lists
  define the same stages; give them the same length.

## 17. Technical Details

### Parameter counts and FLOPs

**MEASURED on this implementation**, `num_classes=1000`, `input_shape=(224, 224, 3)`,
`include_top=True`, default `stem_type='imagenet'`. Not quoted from a paper.

| Variant | Trainable params | Non-trainable (BN stats) | `count_params()` | MACs | FLOPs (2xMACs) |
|:---|---:|---:|---:|---:|---:|
| ResNet-18  | 11,689,512 |   9,600 | 11,699,112 |  1.818 G |  3.636 G |
| ResNet-34  | 21,797,672 |  17,024 | 21,814,696 |  3.669 G |  7.338 G |
| ResNet-50  | 25,557,032 |  53,120 | 25,610,152 |  4.104 G |  8.208 G |
| ResNet-101 | 44,549,160 | 105,344 | 44,654,504 |  7.823 G | 15.646 G |
| ResNet-152 | 60,192,808 | 151,424 | 60,344,232 | 11.544 G | 23.088 G |

**Read the unit before comparing.** "ResNet-50 is 4.1 GFLOPs" in the literature counts
multiply-**accumulates**; the profiler here counts the multiply and the add separately, hence
8.208 G for the same network. Compare the **MACs** column, which matches the usual published
values (1.8 / 3.6 / 4.1 / 7.8 / 11.6) to within rounding. The Trainable column matches
torchvision for all five variants to the digit; Keras counts BatchNorm's
`moving_mean`/`moving_variance` and PyTorch does not, which is the entire Non-trainable
column.

Re-derive the parameter columns:

```python
import numpy as np
from dl_techniques.models.vision.resnet import ResNet

for variant in ResNet.MODEL_VARIANTS:
    m = ResNet.from_variant(variant, num_classes=1000, input_shape=(224, 224, 3))
    m.build((1, 224, 224, 3))
    tr = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    nt = sum(int(np.prod(w.shape)) for w in m.non_trainable_weights)
    print(f"{variant}: {tr:,} + {nt:,} = {m.count_params():,}")
```

The FLOPs column is counted on the **frozen graph**. That matters: a layer-tree walk stops at
a custom subclassed layer and silently undercounts (this repository has a recorded ~50x
undercount from exactly that). The profiler and its calibration and descent arms live in
`tests/test_models/test_resnet/test_the_readme_flops_table_reproduces.py`.

### Authoring rules

Conventions: [`models/CLAUDE.md`](../../CLAUDE.md). Mandatory guide:
`research/2026_keras_custom_models_instructions_v2.md`.

## 18. Citation

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
