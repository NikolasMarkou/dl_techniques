# FFTNet — an adaptive spectral-filtering vision encoder

Until 2026-08-18 this file contained a single bare URL
(`https://github.com/jacobfa/fft/blob/main/spectre.py`, the reference
implementation this port was read against — kept below under *References*).
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
2.  **The FFT axis is load-bearing and was wrong once.** `tf.signal.fft` transforms
    the INNERMOST axis; the token state is `(B, N, D)`, so a direct call transformed
    the FEATURE axis and the layer performed no token mixing at all. The sequence
    axis is transposed to the end for the transform and back afterwards.
3.  **Accepted raw-TF exception.** `FFTMixer.call` uses `tf.signal.fft` /
    `tf.signal.ifft` on a complex64 tensor. This cannot migrate to `keras.ops`,
    which exposes only a real/imag-tuple `fft` and **no** `ifft`, so a
    backend-agnostic complex forward+inverse transform is not expressible. Documented
    exception to the keras.ops-only forward-path rule.

## The encoder contract

`FFTNet` is a **pure encoder**: no pooling layer, no classification head. It embeds
patches, prepends a CLS token, adds a learned positional embedding, runs the block
stack, and returns all three of these unconditionally rather than switching return
type on a flag:

| key | shape |
|:---|:---|
| `last_hidden_state` | `(B, 1 + num_patches, embed_dim)` |
| `cls_token` | `(B, embed_dim)` |
| `patch_features` | `(B, num_patches, embed_dim)` |

Measured 2026-08-18 for `create_fftnet("tiny", image_size=32, patch_size=16)`
(4 patches + CLS, `embed_dim=384`): `(2, 5, 384)` / `(2, 384)` / `(2, 4, 384)`,
5,336,832 parameters.

Heads are attached externally, via `create_fftnet_with_head` or
`create_fftnet_classifier`.

## Variants

`FFTNet.MODEL_VARIANTS` keys: `base`, `large`, `huge`, `small`, `tiny`. They set
`embed_dim`, `num_layers`, `mlp_hidden_dim` and `ffn_ratio`. The parameter figures
in each variant's `description` string assume the default `image_size=224,
patch_size=16`; re-derive with `create_fftnet(variant, ...).count_params()` for any
other resolution, since `W_base` scales with the token count.

## Usage

```python
import numpy as np
from dl_techniques.models.fftnet import create_fftnet

model = create_fftnet("tiny", image_size=32, patch_size=16)
out = model(np.random.rand(2, 32, 32, 3).astype("float32"), training=False)
cls = out["cls_token"]            # (2, 384) -- for a classification head
patches = out["patch_features"]   # (2, 4, 384) -- for a dense-prediction head
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
