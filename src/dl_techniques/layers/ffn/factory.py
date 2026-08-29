"""
Single entry point for building any of this package's 21 FFN layers.

``create_ffn_layer('swiglu', output_dim=512)`` returns a configured Keras
layer. The mapping from the string ``'swiglu'`` to the class ``SwiGLUFFN``, to
the parameters that class requires, and to the defaults it gets when you do not
supply them, lives in one dict: ``FFN_REGISTRY``. Nothing else in the package
needs to know which FFN classes exist.

Two things follow. A model can pick its FFN from a config file, because the
choice is a string. And a new FFN type becomes available everywhere by adding
one registry entry, with no change to any model-building code.

The factory is strict about parameters. A key the chosen ``ffn_type`` does not
accept raises ``ValueError``; it is never silently dropped. If you are a
wrapper layer pushing your own generic defaults (an ``activation``, a
``dropout_rate``) down into whatever FFN the user picked, run them through
``assemble_ffn_config`` first. That drops the keys the chosen type cannot take,
then merges the user's own arguments on top without filtering them, so a user
typo still reaches the factory and still raises.

**Dispatch Flow:**

.. code-block:: text

    wrapper_config ─┐
                    ├─► assemble_ffn_config()   (optional)
    caller_args ────┘   drops keys this ffn_type cannot take,
                        then merges caller_args on top UNFILTERED
                                 │
    config dict ──► create_ffn_from_config()   pops config['type']
                                 │
                                 ▼
      create_ffn_layer(ffn_type, name=None, **kwargs)
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ try:                                                         │
    │   validate_ffn_config(ffn_type, **kwargs)                    │
    │     ├─ ffn_type not in FFN_REGISTRY ──────────► raise        │
    │     ├─ a required_params name is missing ─────► raise        │
    │     └─ a value fails a range / name check ────► raise        │
    │     │                                                        │
    │     ▼                                                        │
    │   ffn_class = FFN_REGISTRY[ffn_type]['class']                │
    │   valid  = required_params + optional_params names           │
    │   params = optional_params defaults, then kwargs             │
    │   final  = params filtered down to valid                     │
    │     │                                                        │
    │     ▼                                                        │
    │   dropped = set(kwargs) - valid              (STRICT)        │
    │     └─ dropped is not empty ──────────────────► raise        │
    │     │                                                        │
    │     ▼                                                        │
    │   final['name'] = name    (only when name is given)          │
    │   logger.info, one line per parameter                        │
    │     │                                                        │
    │     ▼                                                        │
    │   ffn_layer = ffn_class(**final)                             │
    ├──────────────────────────────────────────────────────────────┤
    │ except (TypeError, ValueError) as e:                         │
    │   every raise above lands here and is re-raised as ONE       │
    │   ValueError naming ffn_type, its required params and the    │
    │   params you provided, chained with `from e`.                │
    └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        keras.layers.Layer

**FFN Registry:**

All 21 entries, rendered from ``FFN_REGISTRY`` itself. The class each type
builds and the parameters you must supply:

.. code-block:: text

    type key      class                  required params
    ------------  ---------------------- -------------------------------
    bilinear      GLUFFN                 hidden_dim,output_dim
    counting      CountingFFN            output_dim,count_dim
    differential  DifferentialFFN        hidden_dim,output_dim
    gated_mlp     GatedMLP               filters
    geglu         GeGLUFFN               hidden_dim,output_dim
    gelu_tanh     GELUMLPFFN             hidden_dim
    glu           GLUFFN                 hidden_dim,output_dim
    kan           KANLinear              features
    logic         LogicFFN               output_dim,logic_dim
    lowrank       LowRankFFN             hidden_dim,output_dim
    mixer         MixerBlock             tokens_mlp_dim,channels_mlp_dim
    mlp           MLPBlock               hidden_dim,output_dim
    monarch       MonarchFFN             hidden_dim,output_dim
    orthoglu      OrthoGLUFFN            hidden_dim,output_dim
    power_mlp     PowerMLPLayer          units
    reglu         GLUFFN                 hidden_dim,output_dim
    residual      ResidualBlock          hidden_dim,output_dim
    squared_relu  SquaredReLUFFN         hidden_dim,output_dim
    swiglu        SwiGLUFFN              output_dim
    swin_mlp      SwinMLP                hidden_dim
    tversky       TverskyProjectionLayer units,num_features

The output width parameter is the constructor argument that sets
``compute_output_shape(...)[-1]``. Its name is not uniform across types, so a
caller that pattern-matches the literal string ``"output_dim"`` silently no-ops
for four of them. ``mixer`` has no output width parameter at all: its output
shape equals its input shape. The third column names one optional parameter per
type that is not shared boilerplate; ``get_ffn_info()`` returns the full set.

.. code-block:: text

    type key      output width  notable optional default
    ------------  ------------  --------------------------------
    bilinear      output_dim    activation='linear'
    counting      output_dim    counting_scope='local'
    differential  output_dim    gate_activation='sigmoid'
    gated_mlp     filters       attention_activation='relu'
    geglu         output_dim    activation='gelu'
    gelu_tanh     output_dim    output_dim=None
    glu           output_dim    activation='swish'
    kan           features      grid_size=5, spline_order=3
    logic         output_dim    temperature=1.0
    lowrank       output_dim    rank=None
    mixer         None          activation='gelu'
    mlp           output_dim    output_kernel_initializer=None
    monarch       output_dim    nblocks=4
    orthoglu      output_dim    ortho_reg_factor=1.0
    power_mlp     units         k=3
    reglu         output_dim    activation='relu'
    residual      output_dim    activation='relu'
    squared_relu  output_dim    (boilerplate only)
    swiglu        output_dim    ffn_expansion_factor=4
    swin_mlp      output_dim    output_dim=None
    tversky       units         intersection_reduction='product'

What each type is for:

.. code-block:: text

    type key      use case
    ------------  --------------------------------------------
    bilinear      identity-gated GLU variant (Shazeer 2020)
    counting      sequences where feature frequency matters
    differential  dual-pathway feature processing
    gated_mlp     channel gating in conv nets; no token mixing
    geglu         GELU-gated transformer FFN
    gelu_tanh     SD3 / MMDiT FeedForward
    glu           gated FFN for better gradient flow
    kan           B-spline per-connection activations
    logic         symbolic-like reasoning, feature interaction
    lowrank       parameter-efficient factorized FFN
    mixer         attention-free token + channel mixing
    mlp           general-purpose transformer FFN
    monarch       structured parameter-efficient FFN
    orthoglu      stable training, decorrelated features
    power_mlp     sharp + smooth function approximation
    reglu         ReLU-gated GLU variant (Shazeer 2020)
    residual      deep nets needing skip connections
    squared_relu  Primer squared-ReLU transformer FFN
    swiglu        modern LLMs (LLaMA, Qwen)
    swin_mlp      Swin Transformer and vision heads
    tversky       asymmetric similarity projection, rank-2

``gated_mlp``'s registry ``description`` and ``use_case`` no longer call
it spatially-gated or an attention alternative. Every kernel in it is
1x1, so it cannot mix across positions: measured off-pixel delta 0.0 on
a ``(1, 5, 5, 4)`` input with ``filters=8``, against an on-pixel delta
of 1.27. Both strings are CONSTANTS in ``FFN_REGISTRY``, not docstrings,
so they are part of what ``get_ffn_info()`` returns. The full account is
in ``gated_mlp.py``'s module docstring.

Public functions:

- ``get_ffn_info()`` -- a deep copy of the registry, for callers that
  enumerate types or parameters at runtime.
- ``validate_ffn_config(ffn_type, **kwargs)`` -- runs the checks and returns
  ``None``; raises on anything it rejects.
- ``assemble_ffn_config(ffn_type, wrapper_config, caller_args=None)`` -- builds
  the kwargs dict for a wrapper layer, as described above.
- ``create_ffn_layer(ffn_type, name=None, **kwargs)`` -- builds the layer.
- ``create_ffn_from_config(config)`` -- builds the layer from a dict whose
  ``'type'`` key names the ``ffn_type``.

Module constants:

- ``STRICT_DROPPED_KEY_MARKER`` -- the substring every strict dropped-key
  ``ValueError`` from ``create_ffn_layer`` carries. Match on this constant
  rather than retyping the phrase, so rewording the message cannot blind a
  guard.
- ``_FFN_CONFIG_PASSTHROUGH_KEYS`` -- ``('type', 'name')``, the keys every
  construction path carries that are not FFN constructor parameters.
  ``assemble_ffn_config`` keeps them whatever the registry says. Private: pass
  its ``passthrough`` argument instead of importing this.

References:
-   Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design
    Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.
-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202.

"""

import copy
import keras
from typing import Dict, Any, Literal, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .mlp import MLPBlock
from .swiglu_ffn import SwiGLUFFN
from .diff_ffn import DifferentialFFN
from .glu_ffn import GLUFFN
from .geglu_ffn import GeGLUFFN
from .gelu_mlp_ffn import GELUMLPFFN
from .residual_block import ResidualBlock
from .swin_mlp import SwinMLP
from .counting_ffn import CountingFFN
from .logic_ffn import LogicFFN
from .gated_mlp import GatedMLP
from .orthoglu_ffn import OrthoGLUFFN
from .power_mlp_layer import PowerMLPLayer
from .kan_linear import KANLinear
from .tversky_projection import (
    TverskyProjectionLayer,
    VALID_INTERSECTION_REDUCTIONS,
    VALID_DIFFERENCE_REDUCTIONS,
)
from .monarch_ffn import MonarchFFN
from .mlp_mixer_block import MixerBlock
from .squared_relu_ffn import SquaredReLUFFN
from .lowrank_ffn import LowRankFFN

# ---------------------------------------------------------------------
# Type definition for FFN types
# ---------------------------------------------------------------------

FFNType = Literal[
    'bilinear',
    'counting',
    'differential',
    'gated_mlp',
    'geglu',
    'gelu_tanh',
    'glu',
    'kan',
    'logic',
    'lowrank',
    'mixer',
    'mlp',
    'monarch',
    'orthoglu',
    'power_mlp',
    'reglu',
    'residual',
    'squared_relu',
    'swiglu',
    'swin_mlp',
    'tversky'
]

# ---------------------------------------------------------------------
# FFN layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------
#
# Entry schema (every key is MANDATORY for every entry):
#   'class'            -- the Keras Layer class to instantiate.
#   'description'      -- human-readable one-liner.
#   'required_params'  -- names a caller MUST supply; enforced by
#                         ``validate_ffn_config``.
#   'output_dim_param' -- CONTRACT: the name of the constructor parameter that
#                         sets this FFN's OUTPUT WIDTH (i.e. the value that
#                         becomes ``compute_output_shape(...)[-1]``), or ``None``
#                         for a type whose output width is structurally equal to
#                         its input width and therefore not settable.
#                         The value, when not ``None``, is always a member of
#                         ``required_params`` U ``optional_params`` for the same
#                         entry. The names are NOT uniform across types
#                         ('output_dim' | 'filters' | 'features' | 'units'), which
#                         is precisely why this field exists: a caller that
#                         pattern-matches the literal string "output_dim" silently
#                         no-ops for 'gated_mlp'/'kan'/'power_mlp'/'tversky'.
#                         Consumers: the two VLM head FFN construction sites in
#                         ``layers/heads/vlm/factory.py``.
#                         Invariants pinned by
#                         ``tests/test_layers/test_ffn/test_factory.py``
#                         (``TestOutputDimParamRegistryField``).
#   'optional_params'  -- name -> default, merged under caller kwargs.
#   'use_case'         -- human-readable guidance.

FFN_REGISTRY: Dict[str, Dict[str, Any]] = {
    'counting': {
        'class': CountingFFN,
        'description': 'Feed-Forward Network that learns to count features in a sequence',
        'required_params': ['output_dim', 'count_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'counting_scope': 'local',
            'activation': 'gelu',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Sequence processing where feature frequency or position is important'
    },
    'differential': {
        'class': DifferentialFFN,
        'description': 'Differential Feed-Forward Network with dual-pathway processing',
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'branch_activation': 'gelu',
            'gate_activation': 'sigmoid',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Enhanced feature processing with differential pathways'
    },
    'gated_mlp': {
        'class': GatedMLP,
        'description': 'Channel-wise gated linear unit built from three 1x1 convolutions, applied position-wise',
        'required_params': ['filters'],
        'output_dim_param': 'filters',
        'optional_params': {
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'attention_activation': 'relu',
            'output_activation': 'linear',
            'data_format': None
        },
        'use_case': 'Convolutional feature maps needing channel gating; it mixes no spatial positions'
    },
    'geglu': {
        'class': GeGLUFFN,
        'description': 'GELU Gated Linear Unit Feed-Forward Network',
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'GELU-based gated processing for transformers'
    },
    'gelu_tanh': {
        'class': GELUMLPFFN,
        'description': 'SD3-style GELU (tanh-approximation) MLP FeedForward (Dense -> gelu(approximate=True) -> Dense)',
        'required_params': ['hidden_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'output_dim': None,
            'dropout_rate': 0.0,
            'use_bias': True
        },
        'use_case': 'SD3 / MMDiT FeedForward; tanh-approximate GELU MLP, output_dim defaults to input dim'
    },
    'glu': {
        'class': GLUFFN,
        'description': 'Gated Linear Unit Feed Forward Network',
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'activation': 'swish',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Gated processing for improved gradient flow'
    },
    'reglu': {
        'class': GLUFFN,
        'description': "ReGLU: ReLU-gated GLU FFN (Shazeer 2020) — alias of GLUFFN with activation='relu'",
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'activation': 'relu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'ReLU-gated FFN variant (Shazeer 2020 GLU variants)'
    },
    'bilinear': {
        'class': GLUFFN,
        'description': "Bilinear GLU FFN (Shazeer 2020) — alias of GLUFFN with activation='linear' (no gate nonlinearity)",
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'activation': 'linear',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Bilinear (identity-gated) FFN variant (Shazeer 2020 GLU variants)'
    },
    'kan': {
        'class': KANLinear,
        'description': (
            'Kolmogorov-Arnold Network linear layer with learnable per-connection '
            'univariate activations parameterized by B-splines. Supports N-D inputs via einsum.'
        ),
        'required_params': ['features'],
        'output_dim_param': 'features',
        'optional_params': {
            'grid_size': 5,
            'spline_order': 3,
            'grid_range': (-2.0, 2.0),
            'activation': 'swish',
            'base_trainable': True,
            'spline_trainable': True,
            'kernel_initializer': 'glorot_uniform',
            'base_scaler_initializer': 'ones',
            'epsilon': 1e-7
        },
        'use_case': 'Learnable per-connection univariate activations via B-splines (Kolmogorov-Arnold)'
    },
    'logic': {
        'class': LogicFFN,
        'description': 'Feed-Forward Network that performs soft logical reasoning',
        'required_params': ['output_dim', 'logic_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'temperature': 1.0
        },
        'use_case': 'Tasks requiring symbolic-like reasoning or feature interaction modeling'
    },
    'mlp': {
        'class': MLPBlock,
        'description': 'Standard MLP with intermediate expansion',
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            # DECISION plan-2026-08-22T035419-a11304c8/D-160. See decisions.md D-160.
            # `output_kernel_initializer` is declared here and on none of the
            # other 20 entries: `mlp` is the only type whose output projection is
            # the transformer residual projection. Do NOT add it elsewhere -- that
            # turns a wrong-`ffn_type` raise into a silent no-op.
            'output_kernel_initializer': None,
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'General purpose feed-forward processing in transformers'
    },
    'lowrank': {
        'class': LowRankFFN,
        'description': (
            'Low-rank factorized FFN: each expand/contract projection is a '
            'product Dense(rank, no bias) -> Dense(out), giving a sub-quadratic '
            'parameter count when rank << dims. Same shape contract as MLPBlock.'
        ),
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'rank': None,
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Parameter-efficient FFN via low-rank factorized projections'
    },
    'mixer': {
        'class': MixerBlock,
        'description': (
            'Canonical MLP-Mixer block on rank-3 (B,S,C): pre-LN residual '
            'token-mixing MLP (over the token axis via transpose) followed by a '
            'pre-LN residual channel-mixing MLP (over the channel axis) '
            '(Tolstikhin et al. 2021). Output shape == input shape.'
        ),
        'required_params': ['tokens_mlp_dim', 'channels_mlp_dim'],
        # DECISION plan-2026-07-30T140922-8af1028f/D-004: `None` is MEANINGFUL,
        # not a missing entry. MixerBlock.compute_output_shape returns the input
        # shape unchanged, so this type has no output width parameter.
        # Do NOT 'fix' this to 'output_dim' (no such key) nor to 'channels_mlp_dim'
        # (the channel-mixing MLP's inner width). See decisions.md D-004.
        'output_dim_param': None,
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Attention-free token+channel mixing for patch/token sequences'
    },
    'monarch': {
        'class': MonarchFFN,
        'description': (
            'Order-2 Monarch-structured FFN: each projection is a product of two '
            'block-diagonal matrices interleaved with a reshape/permute (Dao et al. 2022). '
            'Sub-quadratic parameter count; nblocks must divide input_dim, hidden_dim and output_dim.'
        ),
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'nblocks': 4,
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Parameter-efficient structured replacement for dense FFN projections'
    },
    'orthoglu': {
        'class': OrthoGLUFFN,
        'description': 'Orthogonally-regularized Gated Linear Unit for disciplined routing',
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'ortho_reg_factor': 1.0
        },
        'use_case': 'Deep networks requiring stable training and decorrelated features'
    },
    'power_mlp': {
        'class': PowerMLPLayer,
        'description': 'Dual-branch MLP with ReLUK and basis functions for enhanced expressiveness',
        'required_params': ['units'],
        'output_dim_param': 'units',
        'optional_params': {
            'k': 3,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'use_bias': True
        },
        'use_case': 'Tasks requiring approximation of complex functions with both sharp and smooth components'
    },
    'residual': {
        'class': ResidualBlock,
        'description': 'Residual block with skip connections',
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'dropout_rate': 0.0,
            'activation': 'relu',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Deep networks requiring skip connections for gradient flow'
    },
    'squared_relu': {
        'class': SquaredReLUFFN,
        'description': (
            'Primer squared-ReLU FFN: Dense(hidden_dim) -> relu(x)**2 -> Dropout '
            '-> Dense(output_dim) (So et al. 2021). Same shape contract as MLPBlock '
            'but with a fixed (non-configurable) squared-ReLU non-linearity.'
        ),
        'required_params': ['hidden_dim', 'output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Efficient transformer FFN with sharpened squared-ReLU activation'
    },
    'swiglu': {
        'class': SwiGLUFFN,
        'description': 'SwiGLU Feed-Forward Network with gating mechanism',
        'required_params': ['output_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            # Explicit size; None means the 2/3 rule from ffn_expansion_factor.
            'hidden_dim': None,
            'ffn_expansion_factor': 4,
            'ffn_multiple_of': 256,
            'dropout_rate': 0.0,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Modern transformer architectures (LLaMa, Qwen, etc.)'
    },
    'tversky': {
        'class': TverskyProjectionLayer,
        'description': (
            'Asymmetric Tversky-similarity projection layer. NOTE: operates on rank-2 inputs '
            '(batch, input_dim) only; output shape is (batch, units). Not suitable for rank-3 '
            '(batch, time, dim) consumers.'
        ),
        'required_params': ['units', 'num_features'],
        'output_dim_param': 'units',
        'optional_params': {
            'intersection_reduction': 'product',
            'difference_reduction': 'subtractmatch',
            'prototype_initializer': 'glorot_uniform',
            'feature_initializer': 'glorot_uniform',
            'contrast_initializer': 'ones'
        },
        'use_case': 'Asymmetric, psychologically-grounded similarity-based projection alternative to Dense (rank-2 only)'
    },
    'swin_mlp': {
        'class': SwinMLP,
        'description': 'Swin Transformer MLP with configurable activation and regularization',
        'required_params': ['hidden_dim'],
        'output_dim_param': 'output_dim',
        'optional_params': {
            'use_bias': True,
            'output_dim': None,
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': 'Swin Transformer architectures and vision_heads models'
    }
}


# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_ffn_info() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the FFN registry, one entry per supported type.

    Use this to enumerate the available types or to read a type's parameter
    schema at runtime instead of hard-coding it.

    The result is a DEEP COPY, so the caller owns it: editing the returned
    ``required_params`` list or ``optional_params`` mapping cannot reach
    ``FFN_REGISTRY``. The one object still shared with the registry is each
    entry's ``class`` value, which is the layer type itself, not mutable
    payload -- ``copy.deepcopy`` returns a class unchanged.

    :return: A mapping from ``ffn_type`` to a private copy of that type's
        registry entry, with the keys ``class``, ``description``,
        ``required_params``, ``output_dim_param``, ``optional_params`` and
        ``use_case``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return copy.deepcopy(FFN_REGISTRY)


def validate_ffn_config(ffn_type: str, **kwargs: Any) -> None:
    """Check that ``ffn_type`` exists and that ``kwargs`` is a usable config.

    Returns ``None`` on success. Every problem it finds is a ``ValueError``.
    It checks, in order:

    * ``ffn_type`` is a key of ``FFN_REGISTRY``.
    * every name in that entry's ``required_params`` is present in ``kwargs``.
    * ``dropout_rate``, if given, is in ``[0.0, 1.0]``.
    * every dimension argument that is given and not ``None`` is positive
      (``hidden_dim``, ``output_dim``, ``count_dim``, ``logic_dim``, ``filters``,
      ``units``, ``features``, ``num_features``, ``tokens_mlp_dim``,
      ``channels_mlp_dim``).
    * the per-type constraints for ``swiglu``, ``counting``, ``logic``,
      ``gated_mlp``, ``monarch``, ``lowrank``, ``power_mlp``, ``kan`` and
      ``tversky`` -- positive factors, allowed enum values, integer types, and
      ``grid_range`` being a ``(low, high)`` pair with ``low < high``.
    * every activation string other than ``'linear'`` resolves through
      ``keras.activations.get``, and every initializer string resolves through
      ``keras.initializers.get``.

    It does NOT check for keys the type does not accept. That is
    ``create_ffn_layer``'s strict dropped-key check.

    :param ffn_type: An ``FFN_REGISTRY`` key.
    :type ffn_type: str
    :param kwargs: The parameters you intend to pass to the layer constructor.
    :type kwargs: Any
    :return: ``None``. The function is called for its exceptions.
    :rtype: None
    :raises ValueError: If ``ffn_type`` is not registered, a required parameter
        is missing, a value is out of range or has the wrong type, or an
        activation or initializer string does not resolve.
    """
    if ffn_type not in FFN_REGISTRY:
        available_types = sorted(list(FFN_REGISTRY.keys()))
        raise ValueError(
            f"Unknown FFN type '{ffn_type}'. "
            f"Available types: {available_types}"
        )

    ffn_info = FFN_REGISTRY[ffn_type]
    required_params = ffn_info['required_params']

    # Check for required parameters
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        raise ValueError(
            f"Required parameters missing for {ffn_type}: {missing_params}. "
            f"Required: {required_params}"
        )

    # Validate common parameter constraints
    if 'dropout_rate' in kwargs:
        dropout_rate = kwargs['dropout_rate']
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

    positive_dims = ['hidden_dim', 'output_dim', 'count_dim', 'logic_dim', 'filters', 'units', 'features', 'num_features', 'tokens_mlp_dim', 'channels_mlp_dim']
    for dim_param in positive_dims:
        if dim_param in kwargs and kwargs[dim_param] is not None:
            if kwargs[dim_param] <= 0:
                raise ValueError(f"{dim_param} must be positive, got {kwargs[dim_param]}")

    # Validate type-specific parameters
    if ffn_type == 'swiglu':
        if 'ffn_expansion_factor' in kwargs and kwargs['ffn_expansion_factor'] <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {kwargs['ffn_expansion_factor']}")
        if 'ffn_multiple_of' in kwargs and kwargs['ffn_multiple_of'] <= 0:
            raise ValueError(f"ffn_multiple_of must be positive, got {kwargs['ffn_multiple_of']}")
    elif ffn_type == 'counting':
        if 'counting_scope' in kwargs and kwargs['counting_scope'] not in ["global", "local", "causal"]:
            raise ValueError("counting_scope must be one of 'global', 'local', 'causal'")
    elif ffn_type == 'logic':
        if 'temperature' in kwargs and kwargs['temperature'] <= 0:
            raise ValueError(f"temperature must be positive, got {kwargs['temperature']}")
    elif ffn_type == 'gated_mlp':
        valid_activations = {"relu", "gelu", "swish", "silu", "linear"}
        if 'attention_activation' in kwargs and kwargs['attention_activation'] not in valid_activations:
            raise ValueError(f"attention_activation must be one of {valid_activations}")
        if 'output_activation' in kwargs and kwargs['output_activation'] not in valid_activations:
            raise ValueError(f"output_activation must be one of {valid_activations}")
    elif ffn_type == 'monarch':
        if 'nblocks' in kwargs:
            nblocks = kwargs['nblocks']
            if not isinstance(nblocks, int) or nblocks <= 0:
                raise ValueError(f"nblocks must be a positive integer, got {nblocks}")
    elif ffn_type == 'lowrank':
        if 'rank' in kwargs and kwargs['rank'] is not None and kwargs['rank'] <= 0:
            raise ValueError(f"rank must be positive, got {kwargs['rank']}")
    elif ffn_type == 'power_mlp':
        if 'k' in kwargs:
            k = kwargs['k']
            if not isinstance(k, int) or k <= 0:
                raise ValueError(f"k must be a positive integer, got {k}")
    elif ffn_type == 'kan':
        if 'grid_size' in kwargs and (not isinstance(kwargs['grid_size'], int) or kwargs['grid_size'] <= 0):
            raise ValueError(f"grid_size must be a positive integer, got {kwargs['grid_size']}")
        if 'spline_order' in kwargs and (not isinstance(kwargs['spline_order'], int) or kwargs['spline_order'] < 0):
            raise ValueError(f"spline_order must be a non-negative integer, got {kwargs['spline_order']}")
        if 'grid_range' in kwargs:
            gr = kwargs['grid_range']
            if not (isinstance(gr, (tuple, list)) and len(gr) == 2 and gr[0] < gr[1]):
                raise ValueError(f"grid_range must be a (low, high) tuple with low < high, got {gr}")
        if 'epsilon' in kwargs and kwargs['epsilon'] <= 0:
            raise ValueError(f"epsilon must be positive, got {kwargs['epsilon']}")
    elif ffn_type == 'tversky':
        # The valid sets are OWNED by tversky_projection.py and imported at
        # the top of this module; there is no copy here. The anchor at that
        # definition carries the reasoning and the pointer.
        valid_ir = VALID_INTERSECTION_REDUCTIONS
        if 'intersection_reduction' in kwargs and kwargs['intersection_reduction'] not in valid_ir:
            raise ValueError(
                f"intersection_reduction must be one of {sorted(valid_ir)}, "
                f"got '{kwargs['intersection_reduction']}'"
            )
        valid_dr = VALID_DIFFERENCE_REDUCTIONS
        if 'difference_reduction' in kwargs and kwargs['difference_reduction'] not in valid_dr:
            raise ValueError(
                f"difference_reduction must be one of {sorted(valid_dr)}, "
                f"got '{kwargs['difference_reduction']}'"
            )

    # Validate activation functions are valid strings
    activation_params = ['activation', 'branch_activation', 'gate_activation', 'attention_activation', 'output_activation']
    for param in activation_params:
        if param in kwargs:
            activation = kwargs[param]
            if isinstance(activation, str) and activation != 'linear':
                try:
                    keras.activations.get(activation)
                except (ValueError, KeyError):
                    raise ValueError(f"Unknown {param} function: '{activation}'")

    # Validate initializer strings
    initializer_params = ['kernel_initializer', 'bias_initializer']
    for param in initializer_params:
        if param in kwargs:
            initializer = kwargs[param]
            if isinstance(initializer, str):
                try:
                    keras.initializers.get(initializer)
                except (ValueError, KeyError):
                    raise ValueError(f"Unknown {param}: '{initializer}'")


#: The stable substring every strict dropped-key ``ValueError`` from
#: :func:`create_ffn_layer` carries. Guards match on THIS constant rather than
#: re-typing the phrase, so rewording the message cannot silently blind them
#: (and so the phrase has exactly one home).
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"


# Keys every construction path carries that are NOT FFN constructor parameters:
# `type` is consumed by `create_ffn_from_config` to pick the class, and `name` is
# accepted by every Keras layer. Neither appears in any registry entry, so both
# must survive the intersection below.
_FFN_CONFIG_PASSTHROUGH_KEYS: Tuple[str, ...] = ('type', 'name')


def assemble_ffn_config(
        ffn_type: str,
        wrapper_config: Mapping[str, Any],
        caller_args: Optional[Mapping[str, Any]] = None,
        *,
        passthrough: Sequence[str] = _FFN_CONFIG_PASSTHROUGH_KEYS,
) -> Dict[str, Any]:
    """Pre-filter a WRAPPER's own generic FFN conveniences, then merge the CALLER's.

    Interface contract (4 call sites: ``TransformerLayer._get_ffn_config``,
    ``TransformerDecoderLayer._get_ffn_config``,
    ``BaseVLMHead._build_common_layers`` and
    ``ImageCaptioningHead.__init__``'s per-layer FFN loop):

    * ``wrapper_config`` -- the wrapper layer's OWN generic conveniences
      (``activation``, ``dropout_rate``, ``kernel_initializer``, the dims it
      derives from its own hyperparameters, ...). It is INTERSECTED with
      ``FFN_REGISTRY[ffn_type]``'s ``required_params | optional_params``, plus
      ``passthrough``. Keys the target type does not accept are dropped HERE,
      silently and correctly -- they are this wrapper's defaults, not anybody's
      expressed intent.
    * ``caller_args`` -- the end user's own ``ffn_args``/``encoder_ffn_args``
      dict. Merged on top of the filtered result **verbatim, never filtered**,
      and therefore still reaches ``create_ffn_layer``. A caller key the type
      does not accept must stay visible to the factory so the factory can
      complain about it.
    * Returns a NEW dict; neither input is mutated.
    * Raises ``ValueError`` naming the available types if ``ffn_type`` is not in
      ``FFN_REGISTRY`` -- matching what the wrapper sites raised before.

    # DECISION plan-2026-07-30T140922-8af1028f/D-017. See decisions.md D-017.
    This function owns the MERGE, not only the filter, which is why it takes two
    dicts. Do NOT reduce it to a single-dict filter that call sites apply after
    merging their ``ffn_args`` in: that order makes the pre-filter eat the
    caller's keys, so a caller typo goes back to being silently dropped.

    :param ffn_type: An ``FFN_REGISTRY`` key.
    :type ffn_type: str
    :param wrapper_config: The wrapper's own generic config; filtered.
    :type wrapper_config: Mapping[str, Any]
    :param caller_args: The caller's explicit args; NEVER filtered.
    :type caller_args: Optional[Mapping[str, Any]]
    :param passthrough: Keys kept regardless of the registry intersection.
    :type passthrough: Sequence[str]
    :return: The assembled config dict for ``create_ffn_layer`` /
        ``create_ffn_from_config``.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``ffn_type`` is not a registered FFN type.
    """
    ffn_info = FFN_REGISTRY.get(ffn_type)
    if ffn_info is None:
        raise ValueError(
            f"Unknown ffn_type '{ffn_type}'. Available: {sorted(FFN_REGISTRY)}."
        )

    accepted = (
        set(ffn_info['required_params'])
        | set(ffn_info['optional_params'])
        | set(passthrough)
    )
    config = {k: v for k, v in wrapper_config.items() if k in accepted}
    if caller_args:
        config.update(caller_args)
    return config


def create_ffn_layer(
        ffn_type: FFNType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """Build one FFN layer of the requested type.

    Validates ``kwargs``, fills in the registry defaults for anything you left
    out, then calls the registered class. See the module docstring for the
    dispatch flow diagram and the full registry table.

    Parameter handling is STRICT. A key ``ffn_type`` does not accept raises
    instead of being dropped. A wrapper layer that wants to offer generic
    conveniences must pre-filter them through :func:`assemble_ffn_config` first;
    the error message says so too.

    Every exception raised inside this function is caught and re-raised as a
    single ``ValueError`` that names the type, its required parameters and the
    parameters you provided, chained to the original with ``from``. So a
    ``TypeError`` from the layer constructor reaches you as a ``ValueError``
    whose ``__cause__`` is that ``TypeError``.

    Example:

    .. code-block:: python

        ffn = create_ffn_layer('swiglu', output_dim=512)
        ffn = create_ffn_layer('mlp', hidden_dim=2048, output_dim=512,
                               name='block0_ffn')

    :param ffn_type: An ``FFN_REGISTRY`` key. See ``FFNType`` for the 21
        supported values.
    :type ffn_type: FFNType
    :param name: Keras layer name. Passed to the constructor only when it is not
        ``None``.
    :type name: Optional[str]
    :param kwargs: Parameters for the chosen type. The type's
        ``required_params`` must all be present; its ``optional_params`` default
        in if absent. Any other key raises.
    :type kwargs: Any
    :return: The constructed layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``ffn_type`` is not registered, a required parameter
        is missing, a value fails validation, a supplied key is not accepted by
        ``ffn_type``, or the layer constructor itself fails.
    """
    try:
        # Validate configuration
        validate_ffn_config(ffn_type, **kwargs)

        # Get FFN info and class
        ffn_info = FFN_REGISTRY[ffn_type]
        ffn_class = ffn_info['class']

        # Prepare parameters with defaults
        params = {}

        # Get all valid parameter names for this ffn_type
        valid_param_names = set(ffn_info['required_params']) | set(ffn_info['optional_params'].keys())

        # Start with defaults for all optional parameters
        params.update(ffn_info['optional_params'])
        # Update with any user-provided kwargs
        params.update(kwargs)

        # Filter out any unknown parameters to avoid "Unrecognized keyword arguments" error
        final_params = {key: val for key, val in params.items() if key in valid_param_names}

        # DECISION plan-2026-07-30T140922-8af1028f/D-023. See decisions.md D-023.
        # RAISE on an unsupported key; a warning re-opens the silent-typo trap.
        # Subtract `valid_param_names` (required | optional), NOT `required_params`
        # -- the narrow form fires for 21/21 types. Read `kwargs`, not `params`:
        # equal today (205/205 green), still right if an entry gains a bad key.
        dropped = sorted(set(kwargs) - valid_param_names)
        if dropped:
            raise ValueError(
                f"create_ffn_layer('{ffn_type}'): "
                f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
                f"'{ffn_type}' ({ffn_class.__name__}) accepts only "
                f"{sorted(valid_param_names)}. "
                f"Either you mistyped one of those names, or you chose the "
                f"wrong ffn_type for the parameters you are passing. If these "
                f"keys are a WRAPPER's own generic defaults rather than an "
                f"explicit request, pre-filter them with assemble_ffn_config() "
                f"instead of passing them here."
            )

        # Add name if provided
        if name is not None:
            final_params['name'] = name

        # Log final parameters before creation
        logger.info(f"Creating {ffn_type} FFN layer with parameters:")
        log_params = final_params.copy()
        if name:
            log_params['name'] = name
        for param_name, param_value in sorted(log_params.items()):
            if param_name == 'name':
                logger.info(f"  {param_name}: '{param_value}'")
            elif isinstance(param_value, str):
                logger.info(f"  {param_name}: '{param_value}'")
            elif param_value is None:
                logger.info(f"  {param_name}: None")
            else:
                logger.info(f"  {param_name}: {param_value}")

        # Create FFN layer using registry class directly (no if/elif chain)
        ffn_layer = ffn_class(**final_params)

        logger.debug(f"Successfully created {ffn_type} FFN layer: {ffn_layer.name}")
        return ffn_layer

    except (TypeError, ValueError) as e:
        # Enhanced error reporting with context
        ffn_info = FFN_REGISTRY.get(ffn_type)
        if ffn_info:
            required_params = ffn_info.get('required_params', [])
            provided_params = list(kwargs.keys())
            class_name = ffn_info.get('class', type(None)).__name__
            error_msg = (
                f"Failed to create {ffn_type} FFN layer ({class_name}). "
                f"Required parameters: {required_params}. "
                f"Provided parameters: {provided_params}. "
                f"Check parameter compatibility and types. "
                f"Use get_ffn_info() for detailed parameter information. "
                f"Original error: {e}"
            )
        else:
            error_msg = f"Failed to create FFN layer. Unknown FFN type '{ffn_type}'. Original error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_ffn_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Build one FFN layer from a config dict.

    The dict's ``'type'`` key names the ``ffn_type``; every other key is passed
    to :func:`create_ffn_layer` as a keyword argument, ``'name'`` included. The
    input dict is copied, not mutated.

    Example:

    .. code-block:: python

        ffn = create_ffn_from_config(
            {'type': 'mlp', 'hidden_dim': 2048, 'output_dim': 512}
        )

    :param config: A dict with a ``'type'`` key plus the parameters for that
        type.
    :type config: Dict[str, Any]
    :return: The constructed layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict, if it has no ``'type'``
        key, or if :func:`create_ffn_layer` rejects the type or the parameters.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration must include 'type' key")

    config_copy = config.copy()
    ffn_type = config_copy.pop('type')

    logger.debug(f"Creating FFN from config - type: {ffn_type}, params: {list(config_copy.keys())}")

    return create_ffn_layer(ffn_type, **config_copy)

# ---------------------------------------------------------------------
