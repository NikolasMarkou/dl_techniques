"""CBAM classifier — public API re-exports.

The convolutional backbone with CBAM (channel + spatial) attention blocks.
"""
from .model import CBAMNet, create_cbam_net

__all__ = [
    "CBAMNet",
    "create_cbam_net",
]
