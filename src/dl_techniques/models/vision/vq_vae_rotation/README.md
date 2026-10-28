# Rotation-Trick VQ-VAE

A VQ-VAE whose quantizer passes gradients through the codebook lookup with the rotation trick of
Fifty et al. (2025, [arXiv:2410.06424](https://arxiv.org/abs/2410.06424)) instead of the
straight-through estimator of van den Oord et al. (2017).

| | |
|---|---|
| Model | `VQVAERotationTrick` ([`model.py`](model.py)) |
| Factory | `create_vq_vae_rotation` |
| Quantizer | `VectorQuantizerRotationTrick` (`src/dl_techniques/layers/generative/vector_quantizer_rotation_trick.py`); `gradient_mode` is `rotation`, `reflection`, `no_grad_scale` or `ste`, `distance_mode` is `euclidean` or `cosine` |
| Variants | none: the paper publishes a quantizer, not a backbone family, so there is no `MODEL_VARIANTS` table |
| Tests | `tests/test_models/test_vq_vae_rotation/` |

```python
import keras
from dl_techniques.models.vision.vq_vae_rotation import create_vq_vae_rotation

model = create_vq_vae_rotation(input_shape=(32, 32, 3), num_embeddings=64, embedding_dim=16)
recon = model(keras.random.normal((2, 32, 32, 3)))
```

Pass both `encoder` and `decoder` to bring your own, or leave both `None` and pass `input_shape`
for the auto-built pair, scaled by `hidden_channels`, `downsample_factor` (a power of 2) and
`num_res_blocks`:

- encoder: one `ConvStem` per factor of 2, named `enc_down_{i}` from 0 (a `Conv2D(hidden, 4, strides=2)`,
  the normalization chosen by `norm_type`, a ReLU), then the residual blocks, then a 1x1 `Conv2D` to
  `embedding_dim`;
- decoder: a 1x1 `Conv2D`, the residual blocks, `Conv2DTranspose(hidden, 4, strides=2)` +
  normalization + ReLU once per factor of 2, then `Conv2D(C, 3)` with a sigmoid.

`num_embeddings`, `embedding_dim` and (on the auto-built path) `hidden_channels` and `downsample_factor`
are positive integers and `reconstruction_loss_weight` is a positive finite number; anything else raises
`ValueError` naming the argument. A numpy integer is stored as a Python `int`.

**Archives written before the encoder's `ConvStem` down-steps.** The encoder's first layers were the loose
`enc_down_{i}`, `enc_norm_{i}` and `enc_act_{i}`; each step is now one tracked layer, `enc_down_{i}`. Measured
with `.keras` files of the roster model (`layer_norm` and `batch_norm`) saved by `be92a74eb` and by
`20b48aba6^`, each in its own process and loaded by the current code, with identical results:
`keras.models.load_model` restores them with equal outputs for `training=False` and `True` at `atol=0`,
because the archive carries the encoder's own serialized graph, so the model comes back with the old loose
layers; `model.load_weights` into a freshly built current model fails ("A total of 7 objects could not be
loaded ... Layer 'enc_down_0_conv1' expected 2 variables, but received 0"). Copying the weights of an earlier
model by path into a current one gives outputs equal at `atol=0` (all 35 weights with `layer_norm`, all 47
with `batch_norm`):

| Earlier path | Current path |
| :--- | :--- |
| `enc_down_{i}/{kernel,bias}` | `enc_down_{i}/enc_down_{i}_conv1/{kernel,bias}` |
| `enc_norm_{i}/*` | `enc_down_{i}/enc_down_{i}_norm1/*` |

The weight count (35 with `layer_norm`, 47 with `batch_norm`) and the parameter count (90,643 and 91,027)
are unchanged.

**GPU determinism of the auto-built decoder.** Its `Conv2DTranspose` layers run on a GPU as cuDNN
backward-data convolutions whose engine is chosen by timing. Once another transposed convolution
has warmed the autotuner in the same process, the choice can be an engine cuDNN flags as
non-deterministic (`ConvBwdData_eng25`), and two calls of the same model on the same input then
differ in the last bits (measured 1.86e-09 on a test codec with two transposed convolutions,
1.19e-07 on PFT-SR's upsampler). For bit-identical outputs, bring a decoder that upsamples with
`UpSampling2D` followed by `Conv2D`, or set `TF_CUDNN_DETERMINISTIC=1` in the environment before
the process starts; `tf.config.experimental.enable_op_determinism()` called mid-process keeps an
engine the autotuner already cached. Measured in plan-2026-10-14T062616-81938e81, D-024 and D-083.
