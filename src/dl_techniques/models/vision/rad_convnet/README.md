# RADConvNet

A ConvNeXt-shaped classification backbone that swaps the depthwise `KxK` convolution inside
each block for
[`RADConv2D`](../../../layers/conv_blocks/rad_conv.py) — Region-Aware Deformable Convolution
(Maleki & Imani, 2025, [arXiv:2509.15436](https://arxiv.org/abs/2509.15436)).

RAD-Conv predicts, per kernel element and channel group, four non-negative boundary offsets
that define an axis-aligned rectangle whose extent and aspect ratio adapt to image content, and
exactly spatially-integrates the input over that rectangle via a summed-area table. This
decouples receptive-field geometry from kernel size — even a `rad_kernel_size=1` block can
aggregate over a region as large as the whole feature map — while keeping a convolution's
translation-equivariant, locally-parameterized inductive bias.

## Usage

```python
from dl_techniques.models.vision.rad_convnet.model import RADConvNet, create_rad_convnet

model = RADConvNet.from_variant("tiny", num_classes=47, input_shape=(96, 96, 3))
# or
model = create_rad_convnet(variant="tiny", num_classes=47, input_shape=(96, 96, 3))
```

## Variants

| Variant | Depths        | Dims                  |
|---------|---------------|------------------------|
| `tiny`  | `[2, 2, 4, 2]`  | `[32, 64, 128, 256]`   |
| `small` | `[3, 3, 9, 3]`  | `[48, 96, 192, 384]`   |
| `base`  | `[3, 3, 9, 3]`  | `[64, 128, 256, 512]`  |

Deliberately smaller than ConvNeXt's own `[96, 192, 384, 768]`-style defaults: RAD-Conv's exact
region integration (a summed-area table plus per-kernel-element box sampling) is materially more
expensive per channel than a depthwise convolution.

## Training

See `src/train/rad_convnet/` for a training pipeline against the on-disk
[DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) (Describable Textures Dataset) texture
classification dataset.

## Known v1 limitations

- `RADConv2D` supports `strides=1` only; downsampling is handled by the surrounding
  ConvNeXt-style strided-conv layers between stages, exactly as in `ConvNeXtV1`/`ConvNeXtV2`.
- No stochastic-depth / drop-path (unlike `ConvNeXtV1`) — kept out to bound scope; can be added
  the same way `ConvNeXtV1` wires it, without changing `RADConv2D` itself.
