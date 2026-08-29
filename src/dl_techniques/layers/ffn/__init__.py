"""
Feed-forward network layers, and one factory that builds any of them.

This package holds 19 FFN / MLP layer classes plus the factory that constructs
them from a string. ``__all__`` exports **26** names: those 19 classes and 7
factory-interface names (``FFNType``, ``create_ffn_layer``,
``create_ffn_from_config``, ``assemble_ffn_config``, ``validate_ffn_config``,
``get_ffn_info``, ``STRICT_DROPPED_KEY_MARKER``).

The normal way in is the factory:

.. code-block:: python

    from dl_techniques.layers.ffn import create_ffn_layer

    ffn = create_ffn_layer('swiglu', output_dim=512)

``FFN_REGISTRY`` in ``factory.py`` maps **21** type keys onto those 19
classes. There are two more keys than classes because ``'glu'``, ``'reglu'``
and ``'bilinear'`` are three configurations of one class, ``GLUFFN``; every
other class has exactly one key. The factory is strict: a keyword the chosen
type does not declare raises ``ValueError``, it is never dropped.

Importing a class directly also works, and is the right move when you already
know which layer you want and are not driving the choice from a config.

**Architecture Overview:**

.. code-block:: text

    caller
      │
      ├──────────────────────────┐
      │ config-driven            │ direct import
      ▼                          ▼
    create_ffn_layer(key, **kw)  SwiGLUFFN(output_dim=512)
      │                          │
      ▼                          │
    validate_ffn_config(...)     │
      │   unknown key      ─► ValueError
      │   missing required ─► ValueError
      │   bad value/name   ─► ValueError
      ▼                          │
    FFN_REGISTRY[key]['class']   │
      │                          │
      ▼                          │
    strict dropped-key check     │
      │   undeclared kwarg ─► ValueError
      ▼                          │
    ffn_class(**final_params)    │
      │                          │
      └────────────┬─────────────┘
                   ▼
           keras.layers.Layer

    create_ffn_from_config(cfg) pops cfg['type'] and then calls
    create_ffn_layer. assemble_ffn_config() builds the kwargs
    dict for a wrapper layer that has generic defaults to push
    down into whichever FFN the user picked.

``factory.py``'s module docstring carries the full 21-row registry table and
the dispatch flow in detail. ``README.md`` documents each layer's own
parameters.
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

# ---------------------------------------------------------------------
# Export public interface
# ---------------------------------------------------------------------

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
