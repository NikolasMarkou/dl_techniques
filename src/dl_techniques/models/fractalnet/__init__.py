"""FractalNet — public API re-exports.

A very deep classifier built by recursive expansion with no residual connection
anywhere: `f_{C+1}(z) = [f_C(f_C(z))] join [conv(z)]`, so the longest path is
`2^C` blocks while the shortest stays a single convolution. `depths` is the
per-stage expansion level, not a block count.
"""
from .model import FractalNet, create_fractal_net

__all__ = [
    "FractalNet",
    "create_fractal_net",
]
