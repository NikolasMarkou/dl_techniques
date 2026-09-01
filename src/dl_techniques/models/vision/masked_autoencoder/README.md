# Masked Autoencoder (MAE)

A convolutional **masked autoencoder** for self-supervised pre-training: mask most of an image,
reconstruct it, and keep the encoder. The recipe follows He, Chen, Xie, Li, Dollár & Girshick,
*Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022
([arXiv:2111.06377](https://arxiv.org/abs/2111.06377)), adapted to **convolutional** encoders in
the FCMAE style of ConvNeXt V2 (Woo et al., CVPR 2023,
[arXiv:2301.00808](https://arxiv.org/abs/2301.00808)).

`MaskedAutoencoder` wraps **an encoder you supply**. It is not a ViT and it does not drop tokens:
masked patches are replaced in place and the full image goes through the encoder.

> `pretrained=True` raises `NotImplementedError` on every backbone in this repository, so there is
> no ConvNeXt V2 checkpoint to hand this class. Pre-train one here, or pass an encoder you already
> trained and saved.

## 1. Overview: What is Masked Autoencoding and Why It Matters

Hide most of an image, ask a model to reconstruct the hidden parts, and the only way to succeed is
to learn what things look like. No labels are needed, so the training corpus is limited by disk
rather than by an annotation budget, and the resulting encoder transfers to classification,
detection and segmentation with far less labelled data than training from scratch.

The pipeline is four steps:

```
  image (224, 224, 3)
    -> PatchMasking: hide 75% of the 16x16 patches
    -> encoder: a conv feature extractor you supply   -> (14, 14, C)
    -> ConvDecoder: lightweight upsampling stack      -> (224, 224, 3)
    -> loss: MSE on the MASKED patches only
```

Two design consequences worth internalizing: the loss is computed **only on masked patches**, so
the model gets no credit for copying visible pixels; and the decoder is deliberately cheap, so
representation quality is forced into the encoder, which is the part you keep.

## 2. The Problem MAE Solves

Supervised pre-training needs a labelled image for every gradient step, which caps the usable
corpus at the labelling budget. Contrastive self-supervision removes the labels but replaces them
with a training-systems problem: large batches, negative sampling, and augmentation recipes that
have to be tuned.

Masked autoencoding needs neither. The pretext task is generated from the image itself, the loss
is a plain MSE, the batch size is a memory decision rather than a hyperparameter, and a high mask
ratio (75%) makes the task hard enough that trivial interpolation does not solve it.

## 3. The Encoder Contract

`MaskedAutoencoder` wraps an encoder you supply. Three rules, **all enforced by the constructor**:

1. **`encoder` is a `keras.Model`** — the first positional argument. There is no `encoder_dims`
   and no `encoder_output_shape` parameter; a non-model raises `TypeError`.
2. **The encoder must return a 4-D `(B, H', W', C')` feature map.** A token-sequence ViT does not
   fit and raises `ValueError`.
3. **The encoder's total downsampling must equal `2 ** len(decoder_dims)`** — that is
   `2 ** decoder_depth`, so **16x** at the defaults. `ConvDecoder` upsamples exactly 2x per
   `decoder_dims` entry, so a /4 encoder under the default decoder would reconstruct at 4x the
   input resolution. The constructor raises, naming both sizes and suggesting a `decoder_depth`.

The check compares resolved **spatial sizes**, not a downsampling ratio: a ratio comparison
silently accepts a 33x33 -> 8x8 encoder (floor division gives 4 either way) whose decoder then
emits 128x128 against a 33x33 target.

## 4. Architecture Deep Dive

| Component | What it does |
|:---|:---|
| `PatchMasking(patch_size, mask_ratio, mask_value)` | Splits the image into `patch_size` squares, samples a per-sample mask, and substitutes `mask_value` at masked positions. Returns `(masked_images, mask, patches)`; `mask` is `(B, num_patches)`. |
| your encoder | Any `keras.Model` meeting § 3. Sees the *masked* image, full resolution. |
| `ConvDecoder(decoder_dims, output_channels)` | One 2x upsample + conv block per `decoder_dims` entry, then a projection to `output_channels`. |
| `MaskedAutoencoder` | Composes the three, owns `compute_loss`, `train_step`, `test_step` and a `reconstruction_loss` metric. |

`call()` returns a dictionary, not a tensor:

| Key | Shape |
|:---|:---|
| `reconstruction` | `(B, H, W, C)` |
| `mask` | `(B, num_patches)`, 1 = masked |
| `masked_input` | `(B, H, W, C)` — the encoder's actual input |
| `encoded` | the encoder's feature map |

When `decoder_dims` is left `None` it is derived from the encoder's channel count: halve it
`decoder_depth` times, with a floor of 64. A /16 encoder emitting 768 channels therefore gets
`[384, 192, 96, 64]`, and one emitting 32 channels gets `[64, 64, 64, 64]`.

`mask_value` selects what replaces a masked patch: `"learnable"` (a trained mask token, the
default), `"zero"`, `"noise"`, or any float.

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.vision.masked_autoencoder import MaskedAutoencoder
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2

# `strides=2` is REQUIRED. ConvNeXtV2 applies `strides` at the stem AND at each of
# the 3 inter-stage downsamples, so strides=2 gives 2**4 = 16x total -- exactly the
# default decoder's upsampling factor. The shipped default strides=4 gives 256x and
# a 1x1 feature map at 224, which the constructor rejects (§ 3).
encoder = ConvNeXtV2.from_variant(
    "tiny", include_top=False, input_shape=(224, 224, 3), strides=2
)   # -> (None, 14, 14, 768)

mae = MaskedAutoencoder(
    encoder=encoder,
    patch_size=16,               # 16x16 patches -> 14x14 = 196 of them
    mask_ratio=0.75,             # hide 75%
    input_shape=(224, 224, 3),
)
mae.compile(optimizer=keras.optimizers.Adam(1e-4))

images = np.random.rand(8, 224, 224, 3).astype("float32")   # unlabelled, in [0, 1]
mae.fit(images, epochs=1, batch_size=4, verbose=0)

out = mae(images[:1], training=True)
print(tuple(int(d) for d in out["reconstruction"].shape))   # (1, 224, 224, 3)

pretrained_encoder = mae.encoder      # this is the artifact you keep
```

Note `mae.fit(images, ...)` takes **only** images: `train_step` builds its own target from the
input, so there is no `y` argument.

A tiny encoder for tests and smoke runs, and the shape the constructor demands:

```python
import keras
import numpy as np
from dl_techniques.models.vision.masked_autoencoder import MaskedAutoencoder

def conv_encoder(input_shape=(64, 64, 3), width=(16, 24, 32, 32)):
    """A /16 encoder: four stride-2 stages, matching the default decoder_depth=4."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in width:
        x = keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = keras.layers.Activation("gelu")(x)
    return keras.Model(inputs, x, name="mae_encoder")

mae = MaskedAutoencoder(encoder=conv_encoder(), patch_size=16, mask_ratio=0.75,
                        input_shape=(64, 64, 3))
x = np.random.rand(4, 64, 64, 3).astype("float32")
out = mae(x, training=True)
print(sorted(out))
# ['encoded', 'mask', 'masked_input', 'reconstruction']
print(tuple(int(d) for d in out["reconstruction"].shape),
      tuple(int(d) for d in out["mask"].shape), mae.decoder_dims)
# (4, 64, 64, 3) (4, 16) [64, 64, 64, 64]
```

## 6. Component Reference

### `MaskedAutoencoder(...)`

| Argument | Default | Meaning |
|:---|:---:|:---|
| `encoder` | — | a `keras.Model`; see § 3 |
| `patch_size` | `16` | masking granularity, in pixels |
| `mask_ratio` | `0.75` | fraction of patches hidden |
| `decoder_dims` | `None` | channel widths, one per 2x upsample. `None` derives them from the encoder |
| `decoder_depth` | `4` | number of upsamples when `decoder_dims is None` |
| `norm_pix_loss` | `False` | normalize each target patch before the MSE |
| `mask_value` | `"learnable"` | `"learnable"`, `"zero"`, `"noise"`, or a float |
| `input_shape` | `(224, 224, 3)` | image shape `(H, W, C)` |
| `non_mask_value` | `0.0` | weight given to unmasked patches in the loss |

`create_mae_model(encoder, patch_size=16, mask_ratio=0.75, decoder_dims=None,
input_shape=(224, 224, 3), **kwargs)` is a thin factory over the same constructor.

### Choosing the knobs

| Knob | Guidance |
|:---|:---|
| `mask_ratio` | 0.75 is the paper's value; its ablation shows a broad plateau around it. Much lower and the pretext task reduces to interpolation. |
| `patch_size` | Must divide the input. 16 at 224 gives 196 patches; use 8 for small images. |
| `decoder_depth` | Must match the encoder's downsampling: `2 ** decoder_depth == input / feature_size`. |
| `norm_pix_loss` | `True` normalizes each target patch, which sharpens local texture at the cost of a decoder output that no longer looks like the image. |
| `non_mask_value` | `0.0` is the paper's rule — loss on masked patches only. It is applied as a floor (`maximum(mask, non_mask_value)`), so a small positive value adds a weak reconstruction term everywhere. |

### Visualization

`mae.visualize(image)` takes one `(H, W, C)` array and returns
`(original, masked, reconstructed)`. `visualize_reconstruction(mae, images, num_samples=4)`
returns a single `(num_samples * H, 3 * W, C)` composite laid out as
`[original | masked | reconstructed]` per row, clipped to `[0, 1]` — it is an array, not a plot,
so pass it to `plt.imshow` yourself.

```python
from dl_techniques.models.vision.masked_autoencoder import visualize_reconstruction

original, masked, reconstructed = mae.visualize(x[0])
grid = visualize_reconstruction(mae, x, num_samples=2)
print(original.shape, grid.shape)      # (64, 64, 3) (128, 192, 3)
```

## 7. Fine-tuning for Downstream Tasks

The encoder is the deliverable. Pull it out, wrap it, attach a head.

```python
import keras

encoder = mae.encoder                     # or keras.models.load_model("encoder.keras")

inputs = keras.Input(shape=(64, 64, 3))
features = keras.layers.GlobalAveragePooling2D()(encoder(inputs))
outputs = keras.layers.Dense(10, name="classifier")(features)
model = keras.Model(inputs, outputs)

# Stage 1: frozen encoder, train the head at a normal learning rate.
encoder.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# Stage 2: unfreeze and RE-COMPILE at a 100x lower learning rate. A `trainable`
# change has no effect until compile() runs again.
encoder.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```

Save the encoder on its own (`mae.encoder.save("encoder.keras")`) rather than the whole MAE when
the decoder has served its purpose — it reloads with `keras.models.load_model` and no
`custom_objects`.

## 8. Training

`MaskedAutoencoder` defines its own `train_step` and `test_step`, so `compile()` takes an
optimizer and nothing else — no loss, no metrics. The tracked metric is `reconstruction_loss`.

```python
import keras
import tensorflow as tf

mae.compile(optimizer=keras.optimizers.AdamW(learning_rate=1.5e-4, weight_decay=0.05))

ds = (tf.data.Dataset.from_tensor_slices(images)
      .shuffle(1000)
      .map(lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=tf.data.AUTOTUNE)
      .batch(32)
      .prefetch(tf.data.AUTOTUNE))

history = mae.fit(ds, epochs=1, verbose=0)
print(sorted(history.history))      # ['loss', 'reconstruction_loss']
```

Practical notes:

- **Augmentation should be mild.** Random crop and horizontal flip. Heavy colour jitter fights the
  reconstruction objective, which is a pixel-space loss.
- **Normalize to `[0, 1]` or standardize** — but be consistent, because the loss is an MSE in
  whatever space you feed it.
- **Long schedules pay.** The paper pre-trains for 800–1600 epochs; the loss curve keeps
  improving well past the point where reconstructions look plausible.
- **Judge by transfer, not by reconstruction.** Blurry reconstructions are normal and expected
  with a lightweight decoder. A linear probe on frozen features is the metric that matters.
- **Mixed precision** works: set `keras.mixed_precision.set_global_policy("mixed_float16")` before
  building. `call()` already casts the masked image to the compute dtype, because `PatchMasking`'s
  scatter ops return float32.

No accuracy or transfer number appears in this README. Nothing in this repository has trained an
MAE to a published benchmark, so any such number would describe the papers, not this code.

## 9. Serialization

```python
import keras
import numpy as np

mae(x, training=False)            # build it first
mae.save("mae.keras")
restored = keras.models.load_model("mae.keras")
print(type(restored).__name__)    # MaskedAutoencoder
```

`get_config()` serializes the nested encoder via `serialize_keras_object`, so the whole model
round-trips with no `custom_objects`. `input_shape` is coerced back to a tuple on load, because
JSON returns it as a list and `(None,) + list` raises.

## 10. Troubleshooting

- **`ValueError: Encoder/decoder scale mismatch.`** The encoder's downsampling is not
  `2 ** len(decoder_dims)`. The message names both spatial sizes and suggests a `decoder_depth`.
  With a `ConvNeXtV2` encoder this almost always means you left `strides` at its default of 4.
- **`TypeError: encoder must be a keras.Model instance.`** There is no `encoder_dims` and no
  `encoder_output_shape` parameter; pass the model itself.
- **`ValueError: Encoder main output must be 4D tensor (B, H, W, C).`** Token-sequence encoders
  (a plain ViT) are not supported by `ConvDecoder`.
- **Reconstructions are blurry.** Expected. The decoder is deliberately lightweight and the loss
  is an MSE, which is minimized by the conditional mean. Evaluate the encoder, not the pictures.
- **Loss goes to zero almost immediately.** `mask_ratio` is too low, or `non_mask_value` is large
  enough that the model is being rewarded for copying visible pixels.
- **`fit(x, y)` fails or is ignored.** `train_step` derives its own target; pass images only.
- **Fine-tuning does not improve after unfreezing.** You changed `trainable` without calling
  `compile()` again.

Authoring conventions: [`models/CLAUDE.md`](../../CLAUDE.md). Mandatory guide for new models and
layers: `research/2026_keras_custom_models_instructions_v2.md`.

## 11. Citation

```bibtex
@inproceedings{he2022mae,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and
          Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}, eprint={2111.06377}, url={https://arxiv.org/abs/2111.06377}
}

@inproceedings{woo2023convnextv2,
  title={{ConvNeXt V2}: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and
          Liu, Zhuang and Kweon, In So and Xie, Saining},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}, eprint={2301.00808}, url={https://arxiv.org/abs/2301.00808}
}
```
