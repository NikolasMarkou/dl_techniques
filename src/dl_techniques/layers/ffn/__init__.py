"""
Feed-forward network layers, and one factory that builds any of them.

This package holds 19 FFN / MLP layer classes plus a factory that constructs
them from a string key. The factory maps 21 keys onto the 19 classes —
``'glu'``, ``'reglu'`` and ``'bilinear'`` are three configurations of one
class, ``GLUFFN``. It is strict: a keyword the chosen type does not declare
raises ``ValueError`` rather than being silently dropped.

.. code-block:: python

    from dl_techniques.layers.ffn import create_ffn_layer

    ffn = create_ffn_layer('swiglu', output_dim=512)

Importing a class directly also works, and is the right move when the choice
is fixed rather than config-driven.

Architecture:

.. code-block:: text

    caller
      │
      ├─────────────────────────┐
      │ config-driven           │ direct import
      ▼                         ▼
    create_ffn_layer(key, **kw) SwiGLUFFN(output_dim=512)
      │                         │
      ▼                         │
    validate_ffn_config(...)    │
      ▼                         │
    FFN_REGISTRY[key]['class']  │
      ▼                         │
    strict dropped-key check    │
      ▼                         │
    ffn_class(**final_params)   │
      └───────────┬─────────────┘
                  ▼
          keras.layers.Layer

``factory.py``'s module docstring carries the full registry table and
dispatch flow. ``README.md`` documents each layer's own parameters.
"""

from .mlp import MLPBlock
from .swiglu_ffn import SwiGLUFFN
from .diff_ffn import DifferentialFFN
from .glu_ffn import GLUFFN
from .geglu_ffn import GeGLUFFN
from .gelu_mlp_ffn import GELUMLPFFN
from .gated_mlp import GatedMLP
from .orthoglu_ffn import OrthoGLUFFN
from .power_mlp_layer import PowerMLPLayer
from .residual_block import ResidualBlock
from .swin_mlp import SwinMLP
from .logic_ffn import LogicFFN
from .counting_ffn import CountingFFN
from .kan_linear import KANLinear
from .tversky_projection import TverskyProjectionLayer
from .monarch_ffn import MonarchFFN
from .mlp_mixer_block import MixerBlock
from .squared_relu_ffn import SquaredReLUFFN
from .lowrank_ffn import LowRankFFN

from .factory import (
    FFNType,
    assemble_ffn_config,
    STRICT_DROPPED_KEY_MARKER,
    create_ffn_layer,
    create_ffn_from_config,
    get_ffn_info,
    validate_ffn_config
)

__all__ = [
    # Layer classes
    "MLPBlock",
    "SwiGLUFFN",
    "DifferentialFFN",
    "GLUFFN",
    "GeGLUFFN",
    "GELUMLPFFN",
    "GatedMLP",
    "OrthoGLUFFN",
    "PowerMLPLayer",
    "ResidualBlock",
    "SwinMLP",
    "LogicFFN",
    "CountingFFN",
    "KANLinear",
    "TverskyProjectionLayer",
    "MonarchFFN",
    "MixerBlock",
    "SquaredReLUFFN",
    "LowRankFFN",
    # Factory interface
    "FFNType",
    "assemble_ffn_config",
    "STRICT_DROPPED_KEY_MARKER",
    "create_ffn_layer",
    "create_ffn_from_config",
    "get_ffn_info",
    "validate_ffn_config",
]
