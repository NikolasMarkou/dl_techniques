"""Bias-free denoisers: public API re-exports for three denoiser architectures.

All three builders are functional (`keras.Model(inputs, outputs)`) rather
than `keras.Model` subclasses, since converting them would break existing
checkpoints used by `src/train/bfunet/` and `src/applications/`. Bias-free
means no additive constants anywhere in the forward path, which makes the
residual degree-1 homogeneous in the input and therefore equal to a scaled
score, in the Miyasawa sense.

One exception: homogeneity also depends on the normalization used.
`create_convunext_denoiser`'s `block_normalization` defaults to LayerNorm,
which divides by a per-sample std that scales with the input, making the
residual scale-invariant (degree 0) rather than degree-1. That builder
defaults the argument to a `None` sentinel and warns when the choice was
not made explicitly. The other five entry points here default to
`'batchnorm'`, `BiasFreeBatchNorm`, and stay homogeneous. Pass
`block_normalization='batchnorm'` to `create_convunext_denoiser` for the
Miyasawa reading.
"""
from .bfcnn import create_bfcnn_denoiser, create_bfcnn_variant
from .bfconvunext import create_convunext_denoiser, create_convunext_variant
from .bfunet import create_bfunet_denoiser, create_bfunet_variant

__all__ = [
    "create_bfcnn_denoiser",
    "create_bfcnn_variant",
    "create_bfunet_denoiser",
    "create_bfunet_variant",
    "create_convunext_denoiser",
    "create_convunext_variant",
]
