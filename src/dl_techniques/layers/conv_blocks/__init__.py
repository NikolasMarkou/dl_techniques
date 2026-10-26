"""Convolutional building blocks.

``conv_blocks`` re-exports ``ConvStem``; the remaining blocks are imported
directly from their modules (see ``layers/AGENTS.md`` for the
"exports nothing" policy this package follows).
"""

from .stem import ConvStem

__all__ = ["ConvStem"]
