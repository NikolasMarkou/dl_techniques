# FastVLM: A Fast Hybrid Vision Model

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A hybrid convolution/transformer **image model**: a `ConvolutionalStem` of MobileOne blocks, two `RepMixer` stages for cheap token mixing, and one attention stage for global context, ending in a classification head.

> **Two things the name invites you to assume, and shouldn't.**
>
> 1. **This is not a vision-language model.** It takes images and returns class logits or feature maps. There is no text tower and no cross-modal head.
> 2. **This is not a weight-compatible port of any published FastViT or FastVLM checkpoint.** It is assembled from this repo's own blocks. In particular `dl_techniques.layers.repmixer_block.RepMixerBlock` shares a *name* with FastViT's RepMixer and is a different construction; the faithful FastViT port lives in `dl_techniques.layers.fastvit` and `models/vision/fastvit/`.
>
> No pretrained weights are distributed, and the model has not been trained or benchmarked in this repository. No accuracy or latency claim is made for it.

---

## 1. Overview: What is FastVLM and Why It Matters

The model combines layer types hierarchically instead of using one everywhere. Cheap convolutional and mixer-style blocks handle the early stages, where feature maps are large; expensive attention is reserved for the last stage, where the grid has already shrunk.

### Key ideas

1. **Hybrid architecture.** Convolutional stem, then `RepMixer` blocks, then Transformer attention — each component where it is worth its cost.
2. **RepMixer blocks.** A convolutional alternative to self-attention in the early stages: depthwise convolutions mix spatially, `1x1` convolutions mix channels, and the whole thing is linear in the number of spatial locations.
3. **Efficient stem.** Three `MobileOneBlock` layers do the initial `/4` downsample.
4. **Hierarchical structure.** Three stages, progressively downsampling while widening — features at multiple scales, and `extract_features` exposes all of them.

---

## 2. The Problem FastVLM Solves

| | Pure CNN | Pure ViT | This hybrid |
|---|---|---|---|
| Early-stage cost | Cheap, linear | Quadratic in a large token count | Cheap, linear (RepMixer) |
| Global context | Weak; grows with depth | Strong from layer 1 | Strong, from stage 3 |
| Cost driver | Convolution FLOPs | `O(N^2)` attention at every layer | Convolutions dominate |

Self-attention is `O(N^2)` in the number of spatial locations, so it is at its most expensive exactly where the feature map is largest and its content is most local anyway. Deferring it to the `H/16 x W/16` grid makes the quadratic term affordable while still giving every location a view of every other one before the head.

---

## 3. How FastVLM Works: Core Concepts

```
Input (B, H, W, 3)
    │
    ├─► ConvolutionalStem (3 x MobileOneBlock)   -> (B, H/4,  W/4,  D0)
    │
    ├─► Stage 1: depths[0] x RepMixerBlock       -> (B, H/4,  W/4,  D0)
    ├─► Downsample Conv2D                        -> (B, H/8,  W/8,  D1)
    │
    ├─► Stage 2: depths[1] x RepMixerBlock       -> (B, H/8,  W/8,  D1)
    ├─► Downsample Conv2D                        -> (B, H/16, W/16, D2)
    │
    ├─► Stage 3: depths[2] x AttentionBlockVLM   -> (B, H/16, W/16, D2)
    │
    └─► [include_top] GAP -> [Dropout] -> Dense  -> (B, num_classes)
```

`embed_dims` is `[D0, D1, D2]` and `depths` is per stage; both must have exactly 3 entries. With `include_top=False` the model stops at the stage-3 feature map, downsampled `16x` from the input.

---

## 4. Architecture Deep Dive

### 4.1 `ConvolutionalStem`

Three `MobileOneBlock` layers performing the initial `/4` downsample and low-level feature extraction, e.g. `224x224 -> 56x56`.

`MobileOneBlock` is a *reparameterizable* block family in principle. **In this stack the fusion pass is not implemented** — see §9.

### 4.2 `RepMixerBlock`

A residual block that decouples spatial from channel mixing:

```
Y = X + TokenMixer(Norm1(X))       DWConv3x3 -> BN -> Act -> DWConv1x1 -> BN
Z = Y + ChannelMixer(Norm2(Y))     Conv1x1(expand) -> Act -> Conv1x1(project)
```

Both halves are convolutional, so the cost is linear in `H * W` rather than quadratic. The channel mixer is an inverted-bottleneck MLP whose expansion is `mlp_ratio`.

Again: this is **this repo's** `RepMixerBlock`, not FastViT's. Use `dl_techniques.layers.fastvit.FastVitRepMixerBlock` for anything that must match timm block-for-block.

### 4.3 `AttentionBlockVLM`

```
Input (B, H, W, C) ─► flatten to (B, H*W, C) ─► TransformerLayer ─► reshape ─► [LayerScale] ─► (B, H, W, C)
```

**The default `attention_type` is `'group_query'`, not plain multi-head.** It is configured with `num_kv_heads == num_heads`, so the arithmetic is that of ordinary MHA — but it is the only wired attention type that carries **positional information** into stage 3. `'multi_head'` and `'window'` are accepted and carry none.

Because attention is positional here, `attention_max_seq_len` (default `2048`) is the RoPE table length and is a real limit: the stage-3 grid is `(H/16) * (W/16)` tokens, so the default covers inputs up to roughly 720 px. A larger encoder input needs a larger value or RoPE raises. It is consumed only when `attention_type='group_query'`.

`num_heads` defaults to `[max(1, dim // 32) for dim in embed_dims]` and each entry must divide its `embed_dims` entry.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.vision_language.fastvlm import FastVLM

model = FastVLM.from_variant("nano", num_classes=10, input_shape=(32, 32, 3))
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

images = np.random.rand(16, 32, 32, 3).astype("float32")
labels = np.random.randint(0, 10, 16)

loss, acc = model.train_on_batch(images, labels)
print(model.predict(images).shape)   # (16, 10)
print(f"{model.count_params():,}")   # 505,978
model.summary()
```

The head emits **logits**: keep `from_logits=True`, or pass your own activation through the classification `Dense`.

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`FastVLM`** | `...fastvlm.model.FastVLM` | The `keras.Model`. |
| **`FastVLM.from_variant`** | — | Build a preset; kwargs override the row. |
| **`create_fastvlm`** | `...fastvlm.model.create_fastvlm` | Module-level factory; forwards to `from_variant`. |
| **`FastVLM.extract_features`** | — | All four intermediate feature maps (§8). |
| **`AttentionBlockVLM`** | `...fastvlm.components.AttentionBlockVLM` | This package's own attention block. |
| **`RepMixerBlock`** | `...layers.repmixer_block.RepMixerBlock` | Convolutional token + channel mixing. |
| **`ConvolutionalStem`** | `...layers.repmixer_block.ConvolutionalStem` | Stem; `/4` downsample. |
| **`MobileOneBlock`** | `...layers.mobile_one_block.MobileOneBlock` | The stem's conv block. |

`FastVLM`, `create_fastvlm` and `AttentionBlockVLM` are exported from the package:

```python
from dl_techniques.models.vision_language.fastvlm import FastVLM, create_fastvlm, AttentionBlockVLM
```

### Constructor

`num_classes=1000`, `embed_dims=[64, 128, 256]`, `depths=[3, 4, 6]`, `num_heads=None`, `mlp_ratio=4.0`, `dropout_rate=0.0`, `drop_path_rate=0.1`, `use_se=False`, `attention_type='group_query'`, `use_layer_scale=True`, `attention_max_seq_len=2048`, `activation='gelu'`, `kernel_initializer='he_normal'`, `include_top=True`, `input_shape=None` (defaults to `(224, 224, 3)`).

`num_classes=0` builds a feature extractor, as does `include_top=False`.

---

## 7. Configuration & Model Variants

`FastVLM.MODEL_VARIANTS`:

| Variant | `embed_dims` | `depths` | `num_heads` | `mlp_ratio` | `dropout_rate` | `drop_path_rate` | `use_se` |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| **`nano`** | `[24, 48, 96]` | `[1, 2, 3]` | `[1, 2, 3]` | 2.0 | 0.0 | 0.0 | No |
| **`tiny`** | `[32, 64, 128]` | `[2, 3, 4]` | `[1, 2, 4]` | 3.0 | 0.0 | 0.05 | No |
| **`small`** | `[48, 96, 192]` | `[3, 4, 6]` | `[2, 3, 6]` | 4.0 | 0.1 | 0.1 | No |
| **`base`** | `[64, 128, 256]` | `[3, 4, 6]` | `[2, 4, 8]` | 4.0 | 0.1 | 0.1 | No |
| **`large`** | `[96, 192, 384]` | `[4, 6, 8]` | `[3, 6, 12]` | 4.0 | 0.1 | 0.2 | Yes |
| **`huge`** | `[128, 256, 512]` | `[6, 8, 12]` | `[4, 8, 16]` | 4.0 | 0.1 | 0.3 | Yes |

`use_se` toggles squeeze-and-excitation inside the stem's MobileOne blocks. `from_variant(variant, num_classes=1000, input_shape=None, **kwargs)` applies the row and then any kwargs on top:

```python
model = FastVLM.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3),
                             drop_path_rate=0.2, attention_type="multi_head")
```

**No pretrained weights ship with this package.** There is no `pretrained` argument; build the architecture and warm-start from a local checkpoint with `model.load_weights(path)` — or better, `dl_techniques.utils.weight_transfer.load_weights_or_raise(model, path)`, which raises when a load changes zero variables.

---

## 8. Comprehensive Usage Examples

### Example 1: As a feature-extraction backbone

```python
import numpy as np
from dl_techniques.models.vision_language.fastvlm import create_fastvlm

backbone = create_fastvlm("base", include_top=False, input_shape=(256, 256, 3))
features = backbone.predict(np.random.rand(2, 256, 256, 3).astype("float32"))
print(features.shape)   # (2, 16, 16, 256)  -- 16x downsampled, embed_dims[2] channels
```

### Example 2: Multi-scale features for an FPN

```python
feats = backbone.extract_features(np.random.rand(1, 256, 256, 3).astype("float32"))
for name, f in zip(["stem", "stage 1", "stage 2", "stage 3"], feats):
    print(f"{name}: {tuple(f.shape)}")
# stem:    (1, 64, 64, 64)     /4
# stage 1: (1, 64, 64, 64)     /4   (stage 1 does not downsample)
# stage 2: (1, 32, 32, 128)    /8
# stage 3: (1, 16, 16, 256)    /16
```

`extract_features` takes a tensor and runs eagerly; it is not part of `call`, so it does not appear in a functional graph. Wrap it yourself if you need a traced multi-output model.

---

## 9. Advanced Usage Patterns

### Structural reparameterization is NOT available

`ConvolutionalStem` exposes `reparameterize()` and `reset_reparameterization()`, but `MobileOneBlock` implements neither:

- `stem.reparameterize()` is a **silent no-op**. It catches the `AttributeError` per block and logs `Failed to reparameterize stem block N` at WARNING level, then returns normally. Nothing is fused and inference speed is unchanged.
- `stem.reset_reparameterization()` does **not** catch, and raises `AttributeError: 'MobileOneBlock' object has no attribute 'reset_reparameterization'`.

Do not build a deployment path on either. If you need the fusion, implement it as an explicit, separately tested conversion pass over a trained model.

### Mixed precision

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = FastVLM.from_variant("base", num_classes=1000)
```

### Larger inputs

Raise `attention_max_seq_len` past `(H/16) * (W/16)` before going beyond ~720 px (§4.3).

---

## 10. Training and Best Practices

- **Optimizer**: AdamW. Weight decay matters for the attention stage.
- **Schedule**: cosine decay with a few epochs of linear warmup.
- **Stochastic depth**: `drop_path_rate` is the endpoint of a linearly increasing schedule — the drop probability is lowest at the first block and highest at the last. Raise it with model size, as the variant table does.
- **Regularization**: hybrid models with attention inherit the weak inductive biases of Transformers, so RandAugment, Mixup and CutMix are effective and often necessary.
- **Learning rate**: attention stages are LR-sensitive; if training diverges, drop the peak to `1e-4` or `5e-5` and lengthen the warmup before touching anything else.

---

## 11. Serialization & Deployment

`FastVLM` and its layers register through `@register_dl_technique` (key `dl_techniques.models.fastvlm.model>FastVLM` — the `vision_language/` family directory is stripped) and carry a complete `get_config`, so the standard round trip works with no `custom_objects`:

```python
model.save('my_fastvlm.keras')
loaded = keras.models.load_model('my_fastvlm.keras')
```

---

## 12. Testing & Validation

```python
import numpy as np
from dl_techniques.models.vision_language.fastvlm import FastVLM

def test_creation_all_variants():
    for variant in FastVLM.MODEL_VARIANTS:
        assert FastVLM.from_variant(variant, num_classes=10,
                                    input_shape=(64, 64, 3)) is not None

def test_forward_pass_shape():
    model = FastVLM.from_variant("tiny", num_classes=10, input_shape=(128, 128, 3))
    assert model.predict(np.random.rand(4, 128, 128, 3).astype("float32")).shape == (4, 10)

def test_feature_pyramid_strides():
    backbone = FastVLM.from_variant("nano", include_top=False, input_shape=(64, 64, 3))
    feats = backbone.extract_features(np.random.rand(1, 64, 64, 3).astype("float32"))
    assert [tuple(f.shape)[1] for f in feats] == [16, 16, 8, 4]
```

The package suite is at `tests/test_models/test_fastvlm/`.

---

## 13. Troubleshooting & FAQs

**Training diverges.** Lower the peak learning rate and lengthen warmup first; then raise `dropout_rate` / `drop_path_rate` and check that AdamW's `weight_decay` is set.

**RoPE raises on a large input.** The stage-3 token count exceeded `attention_max_seq_len`. Raise it (§4.3).

**`stem.reparameterize()` logged three warnings and changed nothing.** Working as documented — the fusion pass is not implemented (§9).

**`ValueError` about `num_heads` and `embed_dims`.** Each `num_heads` entry must divide its corresponding `embed_dims` entry, and both lists must have exactly 3 entries.

**How does `RepMixer` differ from `ConvMixer` or `MLP-Mixer`?** All three separate spatial from channel mixing. `MLP-Mixer` uses `Dense` layers for both and needs flattened patches; `ConvMixer` uses standard and depthwise convolutions throughout; `RepMixer` is a convolutional mixer designed around structural reparameterization at inference — a design this port does not exploit (§9).

**Which stage sees global context?** Only stage 3. Stages 1 and 2 have the local receptive field of their depthwise kernels.

---

## 14. Technical Details

| | RepMixer (stages 1–2) | Self-attention (stage 3) |
|---|---|---|
| Receptive field | Local, fixed `3x3` depthwise kernel | Global |
| Complexity | `O(N)` in spatial locations | `O(N^2)` |
| Mixing weights | Content-agnostic, learned once | Content-aware, recomputed per input |
| Positional information | Implicit, from the convolution | Explicit, RoPE — **only** under `attention_type='group_query'` |

The hybrid design uses the former where `N` is large and the latter where `N` is small enough to afford it.

---

## 15. Citation

This implementation follows ideas from the papers below; it is not a port of any of them (see the header note).

```bibtex
@inproceedings{vasu2023fastvit,
  title={FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff
          and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}

@inproceedings{vasu2023mobileone,
  title={MobileOne: An Improved One Millisecond Mobile Backbone},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff
          and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

The transformer stage follows ViT (Dosovitskiy et al., 2021, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)).
