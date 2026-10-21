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

- encoder: `Conv2D(hidden, 4, strides=2)` + normalization + ReLU, once per factor of 2, then the
  residual blocks, then a 1x1 `Conv2D` to `embedding_dim`;
- decoder: a 1x1 `Conv2D`, the residual blocks, `Conv2DTranspose(hidden, 4, strides=2)` +
  normalization + ReLU once per factor of 2, then `Conv2D(C, 3)` with a sigmoid.

**GPU determinism of the auto-built decoder.** Its `Conv2DTranspose` layers run on a GPU as cuDNN
backward-data convolutions whose engine is chosen by timing. Once another transposed convolution
has warmed the autotuner in the same process, the choice can be an engine cuDNN flags as
non-deterministic (`ConvBwdData_eng25`), and two calls of the same model on the same input then
differ in the last bits (measured 1.86e-09 on a test codec with two transposed convolutions,
1.19e-07 on PFT-SR's upsampler). For bit-identical outputs, bring a decoder that upsamples with
`UpSampling2D` followed by `Conv2D`, or set `TF_CUDNN_DETERMINISTIC=1` in the environment before
the process starts; `tf.config.experimental.enable_op_determinism()` called mid-process keeps an
engine the autotuner already cached. Measured in plan-2026-10-14T062616-81938e81, D-024 and D-083.
