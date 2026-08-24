"""Bias-free denoisers — public API re-exports.

Three denoiser architectures as sibling modules, all functional builders rather
than `keras.Model` subclasses: converting them would break every existing
checkpoint, and these checkpoints are actively used by `src/train/bfunet/` and
`src/applications/`. Bias-free means no additive constants anywhere in the
forward path, which is what makes the residual degree-1 homogeneous in the
input and therefore equal to a scaled score (Miyasawa).

One documented exception to that last sentence: homogeneity is also
NORM-dependent, and `create_convunext_denoiser`'s `block_normalization` default
is LayerNorm, which divides by a per-sample std that itself scales with the
input and is therefore scale-INVARIANT (degree 0), not degree-1. Removing the
biases is necessary but not sufficient. That builder now defaults the argument
to a `None` sentinel and WARNS when the choice was not made; the five other
entry points here (`create_bfcnn_*`, `create_bfunet_*`,
`create_convunext_variant`) default to `'batchnorm'` -> `BiasFreeBatchNorm` and
are homogeneous. Pass `block_normalization='batchnorm'` to
`create_convunext_denoiser` when the Miyasawa reading is what you need.
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
