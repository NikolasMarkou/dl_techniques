"""PW-FNet (Pyramid Wavelet-Fourier Network) — public API re-exports.

There is no ``MODEL_VARIANTS`` table and none was invented: PW-FNet is a single
3-level U-Net parameterized continuously by ``width`` and the per-level block
counts, with no published named scale family, so ``create_pw_fnet`` constructs
the class with the reference defaults rather than delegating to a
``from_variant``.

``PW_FNet_Block`` is exported alongside the model because the FFN-factory test
suite builds it standalone.
"""
from dl_techniques.models.pw_fnet.model import (
    PW_FNet,
    PW_FNet_Block,
    Downsample,
    Upsample,
    create_pw_fnet,
)

__all__ = [
    "PW_FNet",
    "PW_FNet_Block",
    "Downsample",
    "Upsample",
    "create_pw_fnet",
]
