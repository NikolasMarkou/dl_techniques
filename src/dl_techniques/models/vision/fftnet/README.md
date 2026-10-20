# FFTNet — an adaptive spectral-filtering vision model

`model.py`'s module docstring is the long-form reference; this README is the
orientation layer.

## What replaces attention

Self-attention mixes tokens through an `N x N` score matrix: `O(N^2)` time and
memory. The convolution theorem gives a different route — a pointwise multiplication
in the frequency domain is a circular convolution in the token domain, so a
length-`N` filter couples every token to every other at `O(N log N)` with `O(N)`
parameters.

A *fixed* filter, though, makes the mixing input-independent, which is exactly what
attention buys. FFTNet restores content dependence by conditioning the filter on a
global summary of the input:

```
W = W_base + MLP(mean(x, axis=tokens))
y = IFFT(modReLU(FFT(x) * W))
```

The trade, stated plainly: a global receptive field at log-linear cost, in exchange
for content dependence that is **global rather than pairwise**. One summary vector
modulates the filter; there is no per-token-pair score.

`modReLU` is what keeps a stack from collapsing — a real ReLU on a complex tensor is
not meaningful, and *no* nonlinearity would let consecutive spectral filters compose
into a single linear filter. It shifts the magnitude by a learned per-feature bias,
rectifies, and rescales, leaving the phase (where spatial arrangement lives)
untouched.

## Three properties to know before using it

1.  **Fixed token count.** `W_base` has shape `(seq_len, embed_dim)` — a gain per
    frequency bin per feature — so a model is tied to the token count it was built
    for, i.e. a fixed image resolution. Attention is not.
2.  **The FFT axis is load-bearing.** `tf.signal.fft` transforms the INNERMOST
    axis; the token state is `(B, N, D)`, so a direct call would transform the
    FEATURE axis and the layer would perform no token mixing at all. The sequence
    axis is therefore transposed to the end for the transform and back afterwards.
3.  **Accepted raw-TF exception.** `FFTMixer.call` uses `tf.signal.fft` /
    `tf.signal.ifft` on a complex64 tensor. This cannot migrate to `keras.ops`,
    which exposes only a real/imag-tuple `fft` and **no** `ifft`, so a
    backend-agnostic complex forward+inverse transform is not expressible. Documented
    exception to the keras.ops-only forward-path rule.

## One class, with or without its task head

`FFTNet` follows the models GUIDE section 1 shape: one class, a task head built in
`__init__` through the vision head factory (`build_model_vision_head`) and applied in
`call`, `FFTNet.from_variant`, and `create_fftnet(variant, pretrained=False, **kwargs)`,
which only delegates to it. `pretrained` must be `False`: no weights are distributed.

`include_top=True` (the default) applies one head, chosen by `task_type`. Every
class-score output is a logit, with no activation inside the model:

| `task_type` | head | output |
|:---|:---|:---|
| `classification` (default) | `ClassificationHead` on the final-norm CLS token, `block_dims=()`: an optional dropout (`classifier_dropout_rate`, 0.0) and one linear `Dense`, kernel `TruncatedNormal(0.02)` | one `(B, num_classes)` tensor of class logits |
| `detection` | sequence-format `DetectionHead` on the patch features, 9 anchors, `multi_head` attention | `{"classifications": (B, H/p, W/p, 9*C), "regressions": (B, H/p, W/p, 36)}`, class logits and raw box offsets |
| `segmentation` | sequence-format `SegmentationHead` on the patch features, upsampled by `patch_size` (a power of two), `cbam` attention | one `(B, H, W, num_classes)` tensor of per-pixel class logits |

`head_config_overrides` is applied last, by the factory: every key reaches the head
or the head refuses it, and `name`, `num_classes`, `task_type` raise `ValueError`.

`include_top=False` builds no head and returns the encoder's three outputs together:

| key | shape |
|:---|:---|
| `last_hidden_state` | `(B, 1 + num_patches, embed_dim)` |
| `cls_token` | `(B, embed_dim)` |
| `patch_features` | `(B, num_patches, embed_dim)` |

For `create_fftnet("tiny", image_size=32, patch_size=16, include_top=False)`
(4 patches + CLS, `embed_dim=384`) these are `(2, 5, 384)` / `(2, 384)` / `(2, 4, 384)`,
at 5,336,832 parameters.

Compile a classifier or segmenter with `from_logits=True`
(`SparseCategoricalCrossentropy(from_logits=True)` works on both) and apply
`keras.ops.softmax` to read probabilities. Before 2026-10-14 the classifier returned a
`{"logits": ...}` dict and the segmenter a softmax map. Until 2026-10-20 the heads were
attached by `create_fftnet_with_head` / `create_fftnet_classifier`, functional
`keras.Model` wrappers around the encoder; both are deleted, and `create_fftnet` now
returns the classifier by default (pass `include_top=False` for the encoder). Archives
saved by either wrapper do not load and are not supported.

## Variants

`FFTNet.MODEL_VARIANTS` keys: `base`, `large`, `huge`, `small`, `tiny`. They set
`embed_dim`, `num_layers`, `mlp_hidden_dim` and `ffn_ratio`. The parameter figures
in each variant's `description` string assume the default `image_size=224,
patch_size=16`; re-derive with `create_fftnet(variant, ...).count_params()` for any
other resolution, since `W_base` scales with the token count.

## Usage

```python
import numpy as np
from dl_techniques.models.vision.fftnet import create_fftnet

x = np.random.rand(2, 32, 32, 3).astype("float32")

classifier = create_fftnet("tiny", num_classes=10, image_size=32, patch_size=16)
logits = classifier(x, training=False)            # (2, 10)

segmenter = create_fftnet("tiny", task_type="segmentation", num_classes=4,
                          image_size=32, patch_size=16)
masks = segmenter(x, training=False)              # (2, 32, 32, 4)

encoder = create_fftnet("tiny", image_size=32, patch_size=16, include_top=False)
out = encoder(x, training=False)
cls = out["cls_token"]            # (2, 384)
patches = out["patch_features"]   # (2, 4, 384)
```

## References

- Fein-Ashley, 2025. *The FFT Strikes Back: An Efficient Alternative to
  Self-Attention.* https://arxiv.org/abs/2502.18394 — reference implementation:
  https://github.com/jacobfa/fft/blob/main/spectre.py
- Lee-Thorp et al., 2021. *FNet: Mixing Tokens with Fourier Transforms.*
  https://arxiv.org/abs/2105.03824
- Arjovsky et al., 2015. *Unitary Evolution Recurrent Neural Networks* (modReLU).
  https://arxiv.org/abs/1511.06464
- Rao et al., 2021. *Global Filter Networks for Image Classification.*
  https://arxiv.org/abs/2107.00645
- Dosovitskiy et al., 2020. *An Image is Worth 16x16 Words* (ViT).
  https://arxiv.org/abs/2010.11929
