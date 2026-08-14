"""FractalNet-style classifier — public API re-exports.

Note the architecture deviates from the paper: `FractalBlock` applies both
branches to the SAME input rather than composing one through the other, so every
input-to-output path traverses exactly one convolution and `depth` buys width,
not depth. See the module docstring of `model.py` for the full derivation.
"""
from .model import FractalNet, create_fractal_net

__all__ = [
    "FractalNet",
    "create_fractal_net",
]
