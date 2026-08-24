"""SCUNet (Swin-Conv-UNet) — public API re-exports.

There is no ``MODEL_VARIANTS`` table and none was invented: Zhang et al. ship a
single SCUNet configuration (``config=[4]*7``, ``dim=64``) with no named scale
family, so ``create_scunet`` constructs the class with those defaults rather
than delegating to a ``from_variant``.
"""
from dl_techniques.models.scunet.model import SCUNet, create_scunet

__all__ = [
    "SCUNet",
    "create_scunet",
]
