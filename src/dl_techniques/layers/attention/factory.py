"""
Attention layer factory: one registry, one construction path, strict kwargs.

``ATTENTION_REGISTRY`` maps 33 string keys to the callable that builds each
attention layer, plus that layer's metadata. ``create_attention_layer`` is
the single construction path: it looks the key up, REJECTS any keyword the
target type does not declare, fills in the registry defaults, and
constructs. Nothing on that path filters and drops. A keyword the type does
not accept is a ``ValueError``, never a discarded argument.

Public functions:

    ``get_attention_info()``           all entries' metadata, copied
    ``list_attention_types()``         the 33 keys, sorted
    ``get_attention_requirements()``   one entry's metadata, copied
    ``validate_attention_config()``    pre-flight check; raises, builds nothing
    ``assemble_attention_config()``    filter a wrapper's own defaults, then
                                       merge the caller's args on top
    ``create_attention_layer()``       build a layer
    ``create_attention_from_config()`` build from a ``{'type': ...}`` dict

**Architecture Overview:**

.. code-block:: text

    create_attention_layer(attention_type, name=None, **kwargs)
                      │
                      ▼
    ┌───────────────────────────────────────────────┐
    │ 1. ATTENTION_REGISTRY.get(attention_type)     │
    │    a miss skips step 2 and falls into step 3  │
    └───────────────────────────────────────────────┘
                      ▼
    ┌───────────────────────────────────────────────┐
    │ 2. STRICT KWARG CHECK, outside the try        │
    │    declared = required_params                 │
    │             | optional_params keys            │
    │    leftover = set(kwargs) - declared          │
    └───────────────────────────────────────────────┘
                │                    │
        empty   │                    │  non-empty
                ▼                    ▼
             continue         ► ValueError carrying
                                STRICT_DROPPED_KEY_MARKER
                                (RAISE, never drop a key)
                │
                ▼
    ┌───────────────────────────────────────────────┐
    │ 3. try:                                       │
    │      validate_attention_config(type, **kw)    │
    │        unknown type      ►   ValueError       │
    │        missing required  ►   ValueError       │
    │        bad value / range ►   ValueError       │
    │      params = optional_params.copy()          │
    │      params.update(kwargs)                    │
    │      keep declared names, add `name` if given │
    │      return info['class'](**params)           │
    │    except (TypeError, ValueError) as e:       │
    │      ► ValueError, "verify parameter          │
    │        compatibility", chained from e         │
    └───────────────────────────────────────────────┘
                      ▼
               keras.layers.Layer

**Registered types and what each key constructs:**

Generated from ``ATTENTION_REGISTRY`` below. Every entry carries the same
seven keys: ``class``, ``complexity``, ``description``, ``optional_params``,
``paper``, ``required_params``, ``use_case``.

.. code-block:: text

    anchor                AnchorAttention
    beit                  BeitAttention
    capsule_routing       CapsuleRoutingSelfAttention
    cbam                  CBAM
    channel               ChannelAttention
    differential          DifferentialMultiHeadAttention
    energy                EnergyAttention
    fnet                  FNetFourierTransform
    gated                 GatedAttention
    group_query           GroupedQueryAttention
    hopfield              HopfieldAttention
    lighthouse            LighthouseAttention
    linear                LinearAttention
    mobile_mqa            MobileMQA
    multi_head            MultiHeadAttention
    multi_head_cross      MultiHeadCrossAttention
    multi_head_latent     MultiHeadLatentAttention
    non_local             NonLocalAttention
    perceiver             PerceiverAttention
    performer             PerformerAttention
    ring                  RingAttention
    rpc                   RPCAttention
    shared_weights_cross  SharedWeightsCrossAttention
    single_window         SingleWindowAttention
    spatial               SpatialAttention
    tripse1               TripSE1
    tripse2               TripSE2
    tripse3               TripSE3
    tripse4               TripSE4
    wave_field            WaveFieldAttention
    window                create_grid_window_attention
    window_band           create_band_window_attention
    window_zigzag         create_zigzag_window_attention

**Required parameters, grouped:**

The same 33 keys, grouped by the ``required_params`` list they share. Every
other parameter is optional and has a default in the entry.

.. code-block:: text

    (none)
        fnet, spatial, tripse1, tripse2, tripse3, tripse4
    attention_channels
        non_local
    channels
        cbam, channel
    dim
        energy, linear, mobile_mqa, multi_head,
        multi_head_cross, perceiver, performer, ring, rpc,
        shared_weights_cross, wave_field
    num_heads
        capsule_routing
    dim, num_heads
        anchor, gated, lighthouse
    num_heads, key_dim
        hopfield
    dim, num_heads, head_dim
        differential
    dim, num_heads, kv_latent_dim
        multi_head_latent
    dim, num_heads, num_kv_heads
        group_query
    dim, window_size, num_heads
        beit, single_window, window, window_band, window_zigzag

Both tables are rendered from the registry, not transcribed. ``README.md``
carries its own 33-row table and agrees on all 33 keys; its Class column
names the INSTANCE type, so the three window keys read ``WindowAttention``
there and name the builder function here. The registry is the source of
truth.

FROZEN PUBLIC SURFACE
---------------------
``ATTENTION_REGISTRY``'s key set, the ``AttentionType`` literals, and every
entry's ``required_params`` / ``optional_params`` are **public API**. They are
consumed by config-driven callers (``layers/transformers/adaln_zero.py``,
``models/vision/bias_free_denoisers/bfconvunext.py``,
``models/vision_language/fastvlm/``, ``models/vision/dino/``,
``models/language/gemma/``) and asserted by
``tests/test_layers/test_factory_registry_drift.py``. Adding, renaming or
removing any of them is a breaking change, not a cleanup. Docstrings and
comments in this module may be improved freely; the data may not.

Registry entries whose 'class' is NOT a class
---------------------------------------------
Three of the 33 entries map to module-level FUNCTIONS rather than layer
classes::

    'window'         -> create_grid_window_attention    (window_attention.py)
    'window_zigzag'  -> create_zigzag_window_attention  (window_attention.py)
    'window_band'    -> create_band_window_attention    (window_attention.py)

Each wrapper builds a ``WindowAttention`` locked to one ``partition_mode``
('grid' / 'zigzag' / 'band') and carrying that mode's
``use_relative_position_bias`` default. The class itself does not encode that
pairing, which is why the keys point at the wrappers.

Consequence: the general ``WindowAttention`` class, the one whose
``partition_mode`` the caller chooses, has **no factory key of its own** and
must be imported directly
(``from dl_techniques.layers.attention import WindowAttention``).

``window_attention.py`` also defines ``create_kan_key_window_attention`` and
``create_adaptive_softmax_window_attention``. Both are non-public on purpose:
registered here by neither key nor import, absent from
``attention/__init__.py``, and called only by
``tests/test_layers/test_attention/test_window_attention.py``. Both
configurations are reachable by passing ``attention_mode='kan_key'`` /
``probability_type='adaptive'`` to ``WindowAttention``. Do not register them
"for consistency" — that grows the frozen surface above.

Known shape of `validate_attention_config` (documented, not a defect to fix here)
--------------------------------------------------------------------------------
The numeric checks in ``validate_attention_config`` are a single flat allowlist
of parameter NAMES ('dim', 'num_heads', 'dropout_rate', ...) applied uniformly
to all 33 types. There are no per-type schemas. A parameter gets range-checked
because of what it is CALLED, so a type-specific constraint — one type's
``window_size`` upper bound, or a parameter two types read differently —
cannot be expressed.

The one concession is that the positive-value check compares a sequence
COMPONENTWISE, because ``window_size`` is a scalar edge length for three types
and a ``(Wh, Ww)`` grid for 'beit' (D-006). That is still keyed on the NAME,
not on the type. Per-type schemas would be the right shape; converting is out
of scope for a behavior-preserving pass, because it changes raised message text
that tests match on.
"""

import keras
from typing import Dict, Any, Literal, Mapping, Optional, List, Sequence

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .anchor_attention import AnchorAttention
from .beit_attention import BeitAttention
from .capsule_routing_attention import CapsuleRoutingSelfAttention
from .channel_attention import ChannelAttention
from .convolutional_block_attention import CBAM
from .differential_attention import DifferentialMultiHeadAttention
from .energy_attention import EnergyAttention
from .fnet_fourier_transform import FNetFourierTransform
from .gated_attention import GatedAttention
from .group_query_attention import GroupedQueryAttention
from .hopfield_attention import HopfieldAttention
from .lighthouse_attention import LighthouseAttention
from .linear_attention import LinearAttention
from .mobile_mqa import MobileMQA
from .multi_head_attention import MultiHeadAttention
from .multi_head_cross_attention import MultiHeadCrossAttention
from .multi_head_latent_attention import MultiHeadLatentAttention
from .non_local_attention import NonLocalAttention
from .perceiver_attention import PerceiverAttention
from .performer_attention import PerformerAttention
from .ring_attention import RingAttention
from .rpc_attention import RPCAttention
from .shared_weights_cross_attention import SharedWeightsCrossAttention
from .single_window_attention import SingleWindowAttention
from .spatial_attention import SpatialAttention
from .tripse_attention import TripSE1, TripSE2, TripSE3, TripSE4
from .wave_field_attention import WaveFieldAttention
from .window_attention import (
    create_zigzag_window_attention,
    create_grid_window_attention,
    create_band_window_attention
)

# ---------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------

AttentionType = Literal[
    'anchor',
    'beit',
    'capsule_routing',
    'cbam',
    'channel',
    'differential',
    'energy',
    'fnet',
    'gated',
    'group_query',
    'hopfield',
    'lighthouse',
    'linear',
    'mobile_mqa',
    'multi_head',
    'multi_head_cross',
    'multi_head_latent',
    'non_local',
    'perceiver',
    'performer',
    'ring',
    'rpc',
    'shared_weights_cross',
    'single_window',
    'spatial',
    'tripse1',
    'tripse2',
    'tripse3',
    'tripse4',
    'wave_field',
    'window',
    'window_zigzag',
    'window_band'
]
"""
Type alias for supported attention mechanisms.

This literal type provides IDE autocompletion and type checking for valid
attention layer types supported by the factory.
"""

# ---------------------------------------------------------------------
# Attention Layer Registry
# ---------------------------------------------------------------------

ATTENTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    'anchor': {
        'class': AnchorAttention,
        'description': (
            'Memory-efficient hierarchical attention mechanism that reduces computational '
            'complexity by designating anchor tokens. Anchors perform full self-attention '
            'among themselves, while query tokens cross-attend only to anchors, creating '
            'sparse attention patterns ideal for long sequences.'
        ),
        'required_params': ['dim', 'num_heads'],
        'optional_params': {
            'head_dim': None,
            'dropout_rate': 0.0,
            'use_bias': True,
            'probability_type': 'softmax',
            'probability_config': None,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'Long sequence modeling (high-resolution images, long documents, extended '
            'audio) where O(n²) attention complexity is prohibitive. Provides efficiency-'
            'expressiveness trade-off for transformer architectures.'
        ),
        'complexity': 'O(n√n) vs O(n²) for standard attention',
        'paper': 'Anchored Attention: Efficient Self-Attention for Long Sequences'
    },

    'beit': {
        'class': BeitAttention,
        'description': (
            'BEiT self-attention: multi-head self-attention over a (Wh, Ww) patch grid '
            'preceded by a single cls token, with a learnable T5-style relative position '
            'bias added to the attention logits BEFORE the softmax, and an asymmetric QKV '
            'bias in which the query and value projections carry a bias and the key '
            'projection has none at all. The bias table has (2*Wh-1)*(2*Ww-1)+3 rows: one '
            'per distinct patch-to-patch displacement, plus three dedicated rows for the '
            'cls-to-token, token-to-cls and cls-to-cls relations. Expects a sequence '
            'length of exactly Wh*Ww + 1.'
        ),
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'use_relative_position_bias': True,
            'qv_bias': True,
            'use_proj_bias': True,
            'attn_dropout_rate': 0.0,
            'proj_dropout_rate': 0.0,
            'scale': None,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'BEiT-style vision transformers — masked image modeling pre-training and '
            'image-classification fine-tuning — and any ViT variant that wants a learned '
            'relative position bias over a fixed patch grid with a cls token. Note the '
            'grid is static config: window_size fixes the expected sequence length, so a '
            'single instance does not generalize across input resolutions.'
        ),
        'complexity': 'O(N^2 * D) over N = Wh*Ww + 1 tokens, plus an (N^2) bias gather',
        'paper': 'BEiT: BERT Pre-Training of Image Transformers (arXiv:2106.08254)'
    },

    'capsule_routing': {
        'class': CapsuleRoutingSelfAttention,
        'description': (
            'Advanced attention mechanism incorporating capsule network routing algorithms. '
            'Organizes attention weights into vertical (head-wise) and horizontal (token-wise) '
            'capsules with iterative refinement for enhanced contextual awareness and robust '
            'feature grouping.'
        ),
        'required_params': ['num_heads'],
        'optional_params': {
            'key_dim': None,
            'value_dim': None,
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'routing_iterations': 3,
            'use_vertical_routing': True,
            'use_horizontal_routing': True,
            'use_positional_routing': True,
            'epsilon': 1e-8,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Experimental architectures requiring robust part-whole relationship modeling. '
            'Suitable for complex scene understanding, hierarchical feature grouping, and '
            'disambiguation tasks in vision_heads and language domains.'
        ),
        'complexity': 'O(n²) with additional routing overhead',
        'paper': 'Dynamic Routing Between Capsules + Self-Attention'
    },

    'cbam': {
        'class': CBAM,
        'description': (
            'Convolutional Block Attention Module combining sequential channel and spatial '
            'attention mechanisms. Channel attention identifies "what" features are important '
            'while spatial attention determines "where" to focus, creating comprehensive '
            'feature refinement for CNN architectures.'
        ),
        'required_params': ['channels'],
        'optional_params': {
            'ratio': 8,
            'kernel_size': 7,
            'channel_kernel_initializer': 'glorot_uniform',
            'spatial_kernel_initializer': 'glorot_uniform',
            'channel_kernel_regularizer': None,
            'spatial_kernel_regularizer': None,
            'channel_use_bias': False,
            'spatial_use_bias': True
        },
        'use_case': (
            'Drop-in enhancement for any CNN architecture (ResNet, MobileNet, EfficientNet). '
            'Provides consistent performance improvements in image classification, object '
            'detection, and semantic segmentation with minimal computational overhead.'
        ),
        'complexity': 'O(HWC) - lightweight addition to CNN forward pass',
        'paper': 'CBAM: Convolutional Block Attention Module'
    },

    'channel': {
        'class': ChannelAttention,
        'description': (
            'Channel attention submodule from CBAM that recalibrates feature channels by '
            'modeling inter-channel dependencies. Uses global average and max pooling '
            'followed by shared MLP to generate channel-wise attention weights for '
            'feature recalibration.'
        ),
        'required_params': ['channels'],
        'optional_params': {
            'ratio': 8,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'use_bias': False,
            'intermediate_activation_type': 'relu',
            'intermediate_activation_args': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None
        },
        'use_case': (
            'Selective channel emphasis in CNN feature maps. Ideal when spatial information '
            'should be preserved while enhancing the most informative feature channels. '
            'Often used as a building block in larger attention mechanisms.'
        ),
        'complexity': 'O(C²/r + C) where r is reduction ratio',
        'paper': 'CBAM: Convolutional Block Attention Module'
    },

    'differential': {
        'class': DifferentialMultiHeadAttention,
        'description': (
            'Novel attention mechanism using parallel attention paths to distinguish signal '
            'from noise. Computes weighted difference between two multi-head attention layers '
            'to amplify relevant context while actively suppressing irrelevant information, '
            'reducing hallucination and improving factual accuracy.'
        ),
        'required_params': ['dim', 'num_heads', 'head_dim'],
        'optional_params': {
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'dropout_rate': 0.0,
            'attention_dropout_rate': 0.0,
            'lambda_init': 0.8,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': (
            'Large language models and generative transformers where factual accuracy and '
            'hallucination reduction are critical. Particularly effective for knowledge-'
            'intensive tasks and reasoning applications requiring clean context separation.'
        ),
        'complexity': '2x standard MHA computational cost',
        'paper': 'Differential Transformer'
    },

    'energy': {
        'class': EnergyAttention,
        'description': (
            'Energy-based attention from the Energy Transformer. NOT a weighted sum of '
            'values — there is NO value matrix. The layer defines a scalar token-mixing '
            'energy E_ATT(g) = -(1/beta) * sum_h sum_m logsumexp_n(beta * A_hnm) over '
            'bias-free (head_dim, num_heads, dim) key/query projections, and its call() '
            'returns the exact closed-form NEGATIVE GRADIENT of that energy (a descent '
            'direction). The update carries a second, ET-specific term (the token in its '
            'KEY role) that is absent from vanilla attention and is what makes the '
            'dynamics provably energy-descending. Also exposes energy(g) and update(g).'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'head_dim': None,
            'beta': None,
            'attn_self': False,
            'kernel_initializer': None
        },
        'use_case': (
            'Energy Transformer blocks performing recurrent gradient descent on a single '
            'global energy (associative-memory retrieval, masked-patch inpainting, graph '
            'anomaly detection). Use when the residual stream must be interpretable as a '
            'Lyapunov descent rather than an opaque attn -> FFN stack. NOT a drop-in '
            'replacement for standard self-attention: the output is an update, not a '
            'contextualized value.'
        ),
        'complexity': (
            'O(N²) parametric scaling, same as standard attention, but ~2x its flops: '
            'the energy gradient has two terms (query-role and key-role)'
        ),
        'paper': 'Energy Transformer (arXiv:2302.07253)'
    },

    'fnet': {
        'class': FNetFourierTransform,
        'description': (
            'Parameter-free token mixing mechanism replacing self-attention with 2D Fourier '
            'Transform operations. Applies DFT along sequence and hidden dimensions for '
            'efficient global information mixing with O(N log N) complexity and zero '
            'learnable parameters.'
        ),
        'required_params': [],
        'optional_params': {
            'implementation': 'matrix',
            'normalize_dft': True,
            'epsilon': 1e-12
        },
        'use_case': (
            'Ultra-efficient transformer architectures for large-scale pre-training or '
            'resource-constrained deployment. Excellent for tasks where attention patterns '
            'are less critical than global context mixing, such as certain NLP tasks.'
        ),
        'complexity': 'O(N log N) vs O(N²) for attention',
        'paper': 'FNet: Mixing Tokens with Fourier Transforms'
    },

    'group_query': {
        'class': GroupedQueryAttention,
        'description': (
            'Efficient attention variant balancing multi-head and multi-query approaches. '
            'Reduces KV cache size by sharing Key and Value projections across query head '
            'groups, enabling longer context windows with reduced memory footprint during '
            'autoregressive generation.'
        ),
        'required_params': ['dim', 'num_heads', 'num_kv_heads'],
        'optional_params': {
            'max_seq_len': 2048,
            'dropout_rate': 0.0,
            'rope_percentage': 1.0,
            'rope_theta': 10000.0,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Large language models requiring extended context windows with memory efficiency. '
            'Critical for autoregressive generation tasks where KV cache growth becomes a '
            'bottleneck. Enables practical deployment of large context models.'
        ),
        'complexity': 'Reduces KV cache by factor of num_heads/num_kv_heads',
        'paper': 'GQA: Training Generalized Multi-Query Transformer Models'
    },

    'hopfield': {
        'class': HopfieldAttention,
        'description': (
            'Modern Hopfield Network implementing content-addressable memory through attention '
            'mechanisms. Functions as associative memory that can iteratively refine queries '
            'to retrieve stored patterns, enabling pattern completion and noise correction '
            'capabilities beyond standard attention.'
        ),
        'required_params': ['num_heads', 'key_dim'],
        'optional_params': {
            'value_dim': None,
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'update_steps_max': 0,
            'update_steps_eps': 1e-4,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': 'layer_norm',
            'qk_norm_kwargs': None,
        },
        'use_case': (
            'Associative memory tasks, pattern completion, and robust retrieval with noise '
            'correction. With update_steps_max=0 functions as attention; with >0 enables '
            'powerful iterative pattern retrieval and memory consolidation.'
        ),
        'complexity': 'O(n²) per update step, configurable iteration count',
        'paper': 'Hopfield Networks is All You Need'
    },

    'lighthouse': {
        'class': LighthouseAttention,
        'description': (
            'Coarse-to-fine pyramid attention with top-K causal SDPA. Builds a '
            'multi-level mean-pooled Q/K/V pyramid (branch factor p, L levels), '
            'scores entries with a per-head L2-norm scorer (joint QK/KQ max), '
            'selects top-K pyramid entries per batch (always retaining the '
            'coarsest level), runs a single causal SDPA over the gathered '
            'sub-sequence, and scatters results back via deterministic '
            'segment_sum. A full_attention toggle bypasses the pyramid for '
            'Stage-2 SDPA-resume training.'
        ),
        'required_params': ['dim', 'num_heads'],
        'optional_params': {
            'head_dim': None,
            'num_levels': 3,
            'pooling_factor': 4,
            'top_k': 1536,
            'scorer': 'norm',
            'score_head_reduction': 'mean',
            'full_attention': False,
            # WAS 'rms_norm', which disagreed with the constructor's None and
            # was therefore SILENTLY APPLIED to every factory-built lighthouse
            # layer. Per the class's own D-004(a) that is actively harmful: the
            # scorer ranks ||Q|| and ||K||, and RMSNorm makes both near-constant
            # across positions, erasing the very signal the selection reads.
            # Factory-built layers were getting a degraded scorer that
            # direct-constructed ones were not.
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'dropout_rate': 0.0,
        },
        'use_case': (
            'Long-context causal language modeling where dense O(N^2) attention '
            'is too expensive but exact attention is preferred over linearized '
            'approximations. Requires statically known sequence length N '
            'divisible by pooling_factor ** (num_levels - 1).'
        ),
        'complexity': 'O(N + K log K) per batch element with K << N',
        'paper': 'Lighthouse Attention (arXiv:2605.06554v1)'
    },

    'mobile_mqa': {
        'class': MobileMQA,
        'description': (
            'Mobile-optimized Multi-Query Attention designed for vision_heads transformers on edge '
            'devices. Uses shared Key-Value projections with optional spatial downsampling to '
            'minimize memory bandwidth requirements while maintaining competitive performance '
            'on mobile hardware.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'use_downsampling': False,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': None
        },
        'use_case': (
            'Mobile vision_heads transformers and edge AI applications where memory bandwidth is '
            'the primary bottleneck. Optimized for mobile GPUs and specialized accelerators '
            'with limited memory I/O capabilities.'
        ),
        'complexity': 'Significantly reduced memory bandwidth vs standard MHA',
        'paper': 'MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer'
    },

    'multi_head': {
        'class': MultiHeadAttention,
        'description': (
            'Standard multi-head self-attention mechanism forming the foundation of modern '
            'Transformer architectures. Provides parallel attention computation across '
            'multiple representation subspaces, enabling rich contextual modeling through '
            'diverse attention patterns.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'dropout_rate': 0.0,
            'kernel_initializer': 'he_normal',
            # DECISION plan-2026-08-22T035419-a11304c8/D-160 — declared on
            # 'multi_head' and 'multi_head_cross' ONLY, of the 33 registered types,
            # the two whose output projection IS the block's residual-path
            # projection. Do NOT declare it anywhere else: leaving it undeclared is
            # what turns the request into a LOUD raise. See decisions.md D-160.
            'output_kernel_initializer': None,
            'kernel_regularizer': None,
            'use_bias': False,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Core building block for Transformer architectures across vision_heads, NLP, and '
            'multi-modal tasks. The default choice for most attention-based models requiring '
            'rich contextual understanding and sequence modeling capabilities.'
        ),
        'complexity': 'O(n²d) for sequence length n and dimension d',
        'paper': 'Attention Is All You Need'
    },

    'multi_head_cross': {
        'class': MultiHeadCrossAttention,
        'description': (
            'Unified, highly configurable multi-head attention layer supporting both self-attention '
            'and cross-attention. Features optional adaptive temperature softmax for dynamic '
            'attention sharpening and flexible projection strategies for diverse architectures.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'dropout_rate': 0.0,
            'shared_qk_projections': False,
            'use_bias': True,
            'kernel_initializer': "glorot_uniform",
            'output_kernel_initializer': None,
            'bias_initializer': "zeros",
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
        },
        'use_case': (
            'Core component for encoder-decoder models, Perceiver-style architectures, and any '
            'scenario requiring interaction between two distinct sequences. Can also be used '
            'for self-attention with fine-grained control over projections.'
        ),
        'complexity': 'O(nm*d) where n is query length, m is key/value length',
        'paper': 'Attention Is All You Need'
    },

    'multi_head_latent': {
        'class': MultiHeadLatentAttention,
        'description': (
            'Multi-Head Latent Attention (MLA). Significantly reduces KV cache '
            'memory usage through low-rank compression while maintaining performance comparable '
            'to MHA. Features decoupled RoPE and optional query compression.'
        ),
        'required_params': ['dim', 'num_heads', 'kv_latent_dim'],
        'optional_params': {
            'qk_nope_head_dim': 128,
            'qk_rope_head_dim': 64,
            'v_head_dim': 128,
            'q_latent_dim': None,
            'dropout_rate': 0.0,
            'use_bias': False,
            'max_seq_len': 4096,
            'rope_theta': 10000.0,
            'rope_percentage': 1.0,
            'qk_norm_type': "rms_norm",
            'qk_norm_kwargs': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None
        },
        'use_case': (
            'Large Language Models (LLMs) where inference memory bandwidth and KV cache capacity '
            'are bottlenecks. Enables significantly larger batch sizes or longer context windows.'
        ),
        'complexity': 'KV cache reduced by ~90% compared to standard MHA',
        'paper': 'DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model'
    },

    'non_local': {
        'class': NonLocalAttention,
        'description': (
            'Computer vision_heads attention mechanism capturing long-range spatial dependencies in '
            '4D tensors. Computes weighted responses across all spatial positions to overcome '
            'limited receptive fields of convolutional operations, enabling global context '
            'reasoning in CNN architectures.'
        ),
        'required_params': ['attention_channels'],
        'optional_params': {
            'kernel_size': (7, 7),
            'use_bias': False,
            'output_norm_type': 'batch_norm',
            'output_norm_kwargs': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'intermediate_activation': 'relu',
            'intermediate_activation_args': None,
            'output_activation': 'linear',
            'output_activation_args': None,
            'output_channels': -1,
            'dropout_rate': 0.0,
            'attention_mode': 'gaussian',
            'kernel_initializer': 'glorot_normal',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': (
            'CNN enhancement for tasks requiring global spatial reasoning: video analysis, '
            'instance segmentation, pose estimation, and action recognition. Particularly '
            'effective when distant spatial relationships are crucial for understanding.'
        ),
        'complexity': 'O(HWC²) for spatial dimensions H,W and channels C',
        'paper': 'Non-local Neural Networks'
    },

    'perceiver': {
        'class': PerceiverAttention,
        'description': (
            'Cross-attention mechanism from Perceiver architecture enabling processing of '
            'arbitrarily large and multi-modal inputs. Attends from fixed-size latent queries '
            'to variable-size byte arrays, decoupling computational complexity from input size '
            'while maintaining expressive power.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Large-scale and multi-modal input processing: raw pixel classification, '
            'multi-modal fusion, and handling of heterogeneous data streams. Essential '
            'for Perceiver-style architectures processing diverse input modalities.'
        ),
        'complexity': 'O(MN) where M=latent size, N=input size (vs O(N²))',
        'paper': 'Perceiver: General Perception with Iterative Attention'
    },

    'shared_weights_cross': {
        'class': SharedWeightsCrossAttention,
        'description': (
            'Parameter-efficient cross-attention enabling bidirectional information flow '
            'between two modalities using shared projection weights. Reduces parameter count '
            'while maintaining expressive cross-modal interaction capabilities for fusion '
            'tasks.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Multi-modal fusion with parameter constraints: vision_heads-language models, '
            'audio-visual processing, and sensor fusion in robotics. Ideal when model '
            'size is limited but cross-modal interaction is essential.'
        ),
        'complexity': 'Reduces cross-attention parameters by ~50%',
        'paper': 'Shared-weight Cross-attention for Multi-modal Fusion'
    },

    'single_window': {
        'class': SingleWindowAttention,
        'description': (
            'Unified multi-head self-attention restricted to a single square window '
            'of side window_size (window_size**2 tokens). Internally pads inputs up to '
            'window_size**2 tokens before attention and strips the padding from the '
            'output. Supports a standard linear QKV projection or a non-linear '
            'KAN-based Key projection (attention_mode), a configurable probability '
            'output strategy, optional QK-normalization, and an optional learnable '
            'relative position bias (Swin convention).'
        ),
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'attention_mode': 'linear',
            'use_relative_position_bias': True,
            'qkv_bias': True,
            'qk_scale': None,
            'dropout_rate': 0.0,
            'proj_bias': True,
            'kan_grid_size': 5,
            'kan_spline_order': 3,
            'kan_activation': 'swish',
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'Window-local attention for vision transformers and patch-based models '
            'where attention should be confined to a single fixed-size spatial window. '
            'The kan_key mode injects a non-linear key projection for richer local '
            'feature interactions; the relative position bias preserves intra-window '
            'spatial structure.'
        ),
        'complexity': 'O(W^4) for a window of side W (W**2 tokens)',
        'paper': 'Swin Transformer (windowed self-attention) + KAN key projection'
    },

    'spatial': {
        'class': SpatialAttention,
        'description': (
            'Spatial attention submodule from CBAM generating 2D attention maps highlighting '
            'spatially significant regions. Uses channel-wise pooling operations followed by '
            'convolution to identify important spatial locations while preserving channel '
            'information.'
        ),
        'required_params': [],
        'optional_params': {
            'kernel_size': 7,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'use_bias': True,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None
        },
        'use_case': (
            'Spatial focus enhancement in CNN feature maps. Effective for object detection '
            'and segmentation by suppressing background noise and emphasizing salient object '
            'regions. Often combined with channel attention for comprehensive feature refinement.'
        ),
        'complexity': 'O(HW) for spatial dimensions H,W',
        'paper': 'CBAM: Convolutional Block Attention Module'
    },

    'tripse1': {
        'class': TripSE1,
        'description': (
            'Triplet Attention with Post-Fusion Squeeze-and-Excitation. Combines multi-axis '
            'triplet attention (capturing cross-dimensional interactions) with a global channel '
            'recalibration block after branch fusion, achieving comprehensive 3D attention.'
        ),
        'required_params': [],
        'optional_params': {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None
        },
        'use_case': (
            'Computer vision tasks requiring both spatial and channel-wise refinement. '
            'A powerful drop-in replacement for standard CBAM or SE blocks in CNNs '
            'where capturing inter-dimensional relationships is beneficial.'
        ),
        'complexity': 'O(HWC) + SE overhead',
        'paper': 'Achieving 3D Attention via Triplet Squeeze and Excitation Block'
    },

    'tripse2': {
        'class': TripSE2,
        'description': (
            'Triplet Attention with Pre-Process Squeeze-and-Excitation. Applies channel '
            'recalibration independently to each permuted branch before spatial processing, '
            'allowing the network to weight features prior to rotation and filtering.'
        ),
        'required_params': [],
        'optional_params': {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None
        },
        'use_case': (
            'Variants of 3D attention where channel importance is dominant and should '
            'condition the spatial attention generation process. Effective for complex '
            'feature extraction in deeper network stages.'
        ),
        'complexity': '3x SE overhead compared to TripSE1',
        'paper': 'Achieving 3D Attention via Triplet Squeeze and Excitation Block'
    },

    'tripse3': {
        'class': TripSE3,
        'description': (
            'Triplet Attention with Parallel Squeeze-and-Excitation. Processes spatial '
            'and channel attention paths concurrently and combines them via element-wise '
            'multiplication, treating them as independent descriptors of feature importance.'
        ),
        'required_params': [],
        'optional_params': {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None
        },
        'use_case': (
            'Architectures favoring parallel processing paths to preserve gradient flow '
            'for both spatial and channel characteristics. Useful when spatial and channel '
            'information are relatively decoupled.'
        ),
        'complexity': 'O(HWC) with parallel execution paths',
        'paper': 'Achieving 3D Attention via Triplet Squeeze and Excitation Block'
    },

    'tripse4': {
        'class': TripSE4,
        'description': (
            'Hybrid 3D Attention with Affine Fusion. Merges spatial and channel logits '
            'before activation, creating a unified 3D attention map that jointly optimizes '
            'spatial locations and channel features rather than applying them sequentially.'
        ),
        'required_params': [],
        'optional_params': {
            'reduction_ratio': 0.0625,
            'kernel_size': 7,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None,
            'se_reduction_activation_type': 'relu',
            'se_reduction_activation_args': None
        },
        'use_case': (
            'Advanced vision tasks where the correlation between "where" (spatial) and '
            '"what" (channel) is highly entangled. Provides the most strictly defined '
            '"3D attention" mechanism among the variants.'
        ),
        'complexity': 'Similar to TripSE2 but with broadcasted logit fusion',
        'paper': 'Achieving 3D Attention via Triplet Squeeze and Excitation Block'
    },

    'wave_field': {
        'class': WaveFieldAttention,
        'description': (
            'Physics-inspired multi-head attention that replaces dot-product '
            'attention with an FFT-based damped-wave field convolution. Tokens '
            'deposit information onto a 1-D field grid weighted by key magnitude, '
            'a per-head damped-wave kernel is convolved via FFT, a learnable '
            'coupling matrix mixes across heads at each field position, and each '
            'token gathers from the convolved field. A query-dependent sigmoid '
            'modulation and an input-based content gate refine the output before '
            'the final projection. The left-aligned kernel makes information flow '
            'inherently causal.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'field_size': 512,
            'max_seq_len': 128,
            'dropout_rate': 0.0,
            'use_bias': True,
            'gate_bias_init': 2.0,
            'coupling_noise_stddev': 0.01,
            'coupling_seed': None,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'query_modulation_activation_type': 'sigmoid',
            'query_modulation_activation_args': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None
        },
        'use_case': (
            'Long-range causal sequence modeling where a linear-in-sequence-length, '
            'FFT-based field propagation is an attractive alternative to quadratic '
            'dot-product attention. The inherent causality (left-aligned wave kernel) '
            'suits autoregressive tasks without explicit masking.'
        ),
        'complexity': 'O(N*D + G*log(G)*H*D_h) where G is the field grid size',
        'paper': 'Wave Field Attention (damped-wave FFT field convolution)'
    },

    # NOTE: 'class' here is a FUNCTION, not a layer class — see "Registry entries whose
    # 'class' is NOT a class" in the module docstring. The wrapper pins
    # partition_mode='grid'. `WindowAttention` itself has no key of its own.
    'window': {
        'class': create_grid_window_attention,
        'description': (
            'Windowed multi-head self-attention from Swin Transformer, partitioning inputs '
            'into non-overlapping grids for local attention computation, with spatial '
            'awareness from an optional learnable relative position bias. '
            'READ THE COMPLEXITY FIELD BEFORE PICKING THIS FOR EFFICIENCY. A 1-D input of '
            'length N is folded into a ceil(sqrt(N))-square grid and attention runs inside '
            'window_size-square blocks of it, so for a SEQUENCE this invents a strided, '
            'non-contiguous adjacency — if your data has no 2-D layout you want the '
            "'window_band' key, not this one. "
            'THIS ENTRY WAS REWRITTEN 2026-08-25 and the claim it replaced was the exact '
            'opposite. It used to say that for N <= M (M = window_size**2) the grid pads to '
            'a SINGLE window and the layer costs (M/N)**2 times MORE than plain global '
            'attention over the N real tokens. That was TRUE and is now FALSE: the layer '
            'short-circuits that regime and attends over the N REAL tokens, gathering the '
            'relative-position bias at their grid coordinates. MEASURED on the same '
            '(1, 128, 64) input at window_size=128, CPU, peak RSS: 21.695 GB before the '
            "fix, 0.681 GB after, against 'multi_head''s 0.674 GB on the same input — "
            'parity, where it used to be an inversion. The padded slots were also LEAKING '
            'into the softmax, so this was a CORRECTNESS fix as well as a cost fix: an '
            'all-ones attention mask, a mathematical no-op, used to move the output by up '
            "to 0.980964. The sibling 'window_zigzag' key got the SAME short-circuit later "
            'the same day and is now at parity too (0.678 GB on this input, from '
            '17.503 GB); no partition mode of this layer costs more than full attention '
            'any more.'
        ),
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'qkv_bias': True,
            'qk_scale': None,
            'dropout_rate': 0.0,
            'proj_bias': True,
            'attention_mode': 'linear',
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'use_relative_position_bias': True,
            'kan_grid_size': 5,
            'kan_spline_order': 3,
            'kan_activation': 'swish',
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'High-resolution vision transformers requiring scalable attention mechanisms. '
            'Core component of Swin-style architectures for image classification, object '
            'detection, and semantic segmentation where input resolution scalability is crucial.'
        ),
        'complexity': (
            'O(N * M) with M = W**2 slots per window (W = window_size), for N > M. '
            'For N <= M there is exactly one window and the cost is O(N**2) — dense '
            'attention over the N REAL tokens, never over M padded slots. The '
            'O(M**2) = O(W**4) floor this field advertised until 2026-08-25 is GONE; '
            'so is the inversion it implied, where a large window cost MORE than global '
            'attention. Measure both, do not trust this field: '
            '.venv/bin/python -c "import resource, numpy as np; from '
            'dl_techniques.layers.attention import create_attention_layer as c; '
            "x = np.zeros((1, 128, 64), 'float32'); "
            "c('window', dim=64, window_size=128, num_heads=4)(x); "
            'print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)" '
            '# MEASURED 2026-08-25 on CPU: 0.681 GB, against 0.674 GB for multi_head on '
            'the same input. Before the short-circuit the same probe reported 21.695 GB. '
            'CUDA_VISIBLE_DEVICES matters: with a GPU visible the same probe reads '
            '1.361 GB for both, because the CUDA runtime dominates the number.'
        ),
        'paper': 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows'
    },
    # NOTE: also a FUNCTION, pinning partition_mode='zigzag' (and defaulting
    # use_relative_position_bias to False, unlike 'window' above).
    'window_zigzag': {
        'class': create_zigzag_window_attention,
        'description': (
            'Windowed multi-head self-attention that first reorders the input sequence along '
            'a 2D zigzag path to group frequency-proximate tokens. This induces a frequency-based '
            'locality bias, useful for image data. Supports advanced normalization like adaptive softmax.'
        ),
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'qkv_bias': True,
            'qk_scale': None,
            'dropout_rate': 0.0,
            'proj_bias': True,
            'attention_mode': 'linear',
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'use_relative_position_bias': False,
            'kan_grid_size': 5,
            'kan_spline_order': 3,
            'kan_activation': 'swish',
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'Vision transformers processing image data where frequency-domain relationships '
            'are important. Advanced normalization options are suitable for models requiring '
            'better calibration or exploring alternatives to softmax.'
        ),
        'complexity': (
            'O(max(N, M) * M) with M = W**2, and NO O(M**2) = O(W**4) floor: for N <= M '
            'the zigzag layout degenerates to a single window (it folds the sequence into '
            'a ceil(sqrt(N))-square grid, so N_grid <= M), and that case is '
            'SHORT-CIRCUITED to dense attention over the N REAL tokens with the '
            'relative-position bias gathered at each token\'s position in the scan — '
            'O(N**2), never worse than plain global attention. THIS ENTRY WAS REWRITTEN '
            '2026-08-25 (twice). It first said the cost was an unavoidable W**4 floor; it '
            'then said the "window" key\'s short-circuit did NOT apply here and this path '
            'still measured 17.503 GB. Both are now FALSE. MEASURED 2026-08-25 after the '
            'fix, same command, same (1, 128, 64) input at window_size=128, CPU peak RSS: '
            "'window' 0.680 GB, 'multi_head' 0.674 GB, 'window_band' 0.679 GB, "
            "'window_zigzag' 0.678 GB — four-way parity, where this key used to be a "
            '26x inversion. Measure it yourself: '
            '.venv/bin/python -c "import resource, numpy as np; from '
            'dl_techniques.layers.attention import create_attention_layer as c; '
            "x = np.zeros((1, 128, 64), 'float32'); "
            "c('window_zigzag', dim=64, window_size=128, num_heads=4)(x); "
            'print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)"'
        ),
        'paper': "Extends 'Swin Transformer' with zigzag partitioning and advanced normalization"
    },
    # NOTE: also a FUNCTION, pinning partition_mode='band' (and defaulting
    # use_relative_position_bias to False, which 'band' in fact REFUSES if set True).
    'window_band': {
        'class': create_band_window_attention,
        'description': (
            'One-dimensional SYMMETRIC sliding-band self-attention over the token sequence: '
            'query i attends key j iff abs(i - j) <= window_size. Here window_size is a '
            'HALF-WIDTH IN TOKENS, not a 2-D edge length — there is no grid folding and no '
            'square padding, unlike the "window" and "window_zigzag" keys above. This is the '
            'layout text encoders specify (Longformer / Mistral / ModernBERT); upstream '
            "ModernBERT's local_attention is a FULL span, so pass local_attention // 2. "
            'Non-causal, so it is an ENCODER band, not a decoder one. '
            'READ THE COMPLEXITY FIELD BEFORE PICKING THIS FOR EFFICIENCY: the band is a '
            'dense N x N mask over standard attention, which is O(N^2) — the SAME '
            'asymptotics as plain multi_head attention, NOT the O(N*W) the name suggests. '
            'The relative position bias is unavailable (it indexes a 2-D tile this layout '
            'does not have) and defaults to False here.'
        ),
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'qkv_bias': True,
            'qk_scale': None,
            'dropout_rate': 0.0,
            'proj_bias': True,
            'attention_mode': 'linear',
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None,
            'use_relative_position_bias': False,
            'kan_grid_size': 5,
            'kan_spline_order': 3,
            'kan_activation': 'swish',
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'Text encoders whose local layers must see a 1-D token neighbourhood — the '
            'ModernBERT / Longformer local-attention pattern, interleaved with global '
            'layers. Use it wherever a sequence has no meaningful 2-D layout, i.e. wherever '
            'folding it into a ceil(sqrt(N)) square would invent an adjacency the data does '
            'not have.'
        ),
        'complexity': (
            'O(N^2) — a dense N x N banded mask over standard attention, the SAME order as '
            'full multi_head attention, and deliberately NOT advertised as O(N*W): a true '
            'banded kernel is not reachable from keras.ops. What it does remove is the '
            "'window' key's O(W**4) floor — N real tokens are never inflated to "
            'window_size**2 slots. THE PREMIUM OVER multi_head GROWS WITH N, so one '
            'measuring point is not a cost model. MEASURED CPU peak RSS, dim=64, '
            'window_size=64, interpreter+import floor 0.655 GB: at N=512 window_band '
            '0.705 GB vs multi_head 0.684 GB (+3.1%); at N=8192 window_band 5.751 GB vs '
            'multi_head 2.796 GB (+105.7%) — the dense N x N int32 band predicate is '
            'itself O(N^2) and at N=8192 it costs about as much as the scores. Pick this '
            'key for the ADJACENCY it gives a 1-D sequence, never to save memory. Measure '
            'at YOUR N, do not trust this field: '
            '.venv/bin/python -c "import sys, resource, numpy as np; from '
            'dl_techniques.layers.attention import create_attention_layer as c; '
            "N = int(sys.argv[1]); x = np.zeros((1, N, 64), 'float32'); "
            "c('window_band', dim=64, window_size=64, num_heads=4)(x); "
            'print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)" 8192'
        ),
        'paper': (
            'Longformer / Mistral / ModernBERT 1-D sliding-window attention '
            '(ModernBERT: local_attention, half-width local_attention // 2)'
        )
    },

    # DECISION plan_2026-06-14_0c5d4a21/D-007 — 'gated', 'performer', 'ring' and
    # 'rpc' are registered for CONSTRUCTION only, and their call-signature quirks
    # are documented rather than renamed. Do NOT rename `PerformerAttention.call`
    # (it has no `attention_mask` parameter) or `RPCAttention.call` (its parameter
    # is `mask`, not `attention_mask`) to make the four look uniform: this factory
    # only constructs, so a call-signature difference is a documented caveat, not
    # a registration blocker, and renaming it changes behaviour for every direct
    # caller of those layers.
    # The originating plan directory is gone, so this comment is the record.
    'gated': {
        'class': GatedAttention,
        'description': (
            'Gated multi-head self-attention with partial rotary position embeddings '
            '(RoPE) and a learned per-head output gate. Combines QK-normalization, '
            'configurable attention-probability function, and a sigmoid-style gate that '
            'modulates the attention output, improving training stability and selectivity.'
        ),
        'required_params': ['dim', 'num_heads'],
        'optional_params': {
            'head_dim': None,
            'max_seq_len': 4096,
            'rope_percentage': 0.5,
            'dropout_rate': 0.0,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': 'zero_centered_rms_norm',
            'qk_norm_kwargs': None,
            'gate_activation_type': 'sigmoid',
            'gate_activation_args': None,
            # DECISION plan-2026-08-17T183311-79c63e38/D-011 — `num_kv_heads` is
            # DECLARED here, the one exception to this module's FROZEN PUBLIC
            # SURFACE note. Do NOT clear the drift test by deleting it or exempting
            # the entry: with the strict raise below an undeclared `num_kv_heads` is
            # a hard ValueError, not a supported feature. See decisions.md D-011.
            'num_kv_heads': None
        },
        'use_case': (
            'Transformer language and sequence models that benefit from gated, '
            'rotary-position self-attention with QK-norm for stable training. A '
            'drop-in self-attention block where output gating improves selectivity.'
        ),
        'complexity': 'O(n²d) standard self-attention plus a per-head output gate',
        'paper': 'Gated Attention (QK-norm + partial RoPE + output gating)'
    },

    'performer': {
        'class': PerformerAttention,
        'description': (
            'Linear-complexity self-attention via the FAVOR+ random-feature approximation '
            'of softmax attention. Projects queries/keys into a random feature space '
            '(nb_features) to compute attention in O(n) time and memory. NOTE: '
            'ortho_scaling is a scalar multiply of the random features, NOT FAVOR+ '
            'orthogonalization (see layer docstring). CALL-SIG CAVEAT: performer.call '
            'has NO attention_mask parameter (construction-only registration; the '
            'mask-less call signature is a documented, intentional quirk, not renamed).'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'nb_features': 256,
            'ortho_scaling': 0.0,
            'causal': False,
            'dropout_rate': 0.0,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'Long-sequence modeling where quadratic attention is prohibitive and a '
            'linear-attention approximation is acceptable. Suited to large context '
            'windows in NLP and other long-range sequence tasks.'
        ),
        'complexity': 'O(n) linear attention via FAVOR+ random features (vs O(n²))',
        'paper': 'Rethinking Attention with Performers (FAVOR+)'
    },

    'linear': {
        'class': LinearAttention,
        'description': (
            'Bias-free, degree-1 positively-homogeneous linear (O(N)) self-attention '
            'for Miyasawa-compliant denoising. Replaces the softmax kernel with a '
            'positively-homogeneous, non-negative feature map phi (relu / relu_squared '
            '/ abs) and a mandatory normalizer, computed via matmul associativity so '
            'the N x N attention matrix is never materialized. Both Miyasawa properties '
            'hold by construction: every Q/K/V/output projection is bias-free '
            '(use_bias=False by default) and f(alpha*x) = alpha*f(x) for alpha > 0. '
            'Unlike Performer, phi is NOT Gaussian and the denominator floor is an '
            'input-scaled epsilon (epsilon * mean_over_tokens(z)), preserving exact '
            'degree-1 homogeneity. Non-causal (v1). CALL-SIG CAVEAT: linear.call has '
            'NO attention_mask parameter (accepts an ignored `mask=` kwarg for API '
            'uniformity); v1 is unmasked and non-causal.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'head_dim': None,
            'dropout_rate': 0.0,
            'use_bias': False,
            'feature_map': 'relu',
            'epsilon': 1e-6,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': (
            'Bias-free / Miyasawa-compliant denoiser stacks (bfunet family) needing a '
            'linear-complexity, degree-1-homogeneous attention block that preserves the '
            'additive-Gaussian residual = sigma^2 * score identity. Also general '
            'long-sequence self-attention where O(N) cost and homogeneity are desired '
            'over a softmax approximation.'
        ),
        'complexity': 'O(N * d^2) = O(N) in sequence length (associativity path)',
        'paper': 'Transformers are RNNs (Katharopoulos et al., 2020) + Miyasawa/bias-free denoising'
    },

    'ring': {
        'class': RingAttention,
        'description': (
            'Block-wise self-attention designed for memory-efficient distribution of '
            'long sequences across devices in a ring topology. Processes the sequence in '
            'blocks of block_size with optional QK-normalization, enabling near-linear '
            'memory scaling for very long contexts.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'block_size': 512,
            'dropout_rate': 0.0,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Extremely long-context training/inference where the full sequence does not '
            'fit in a single device, leveraging block-wise computation and ring '
            'communication for memory-efficient exact attention.'
        ),
        'complexity': 'O(n²) compute, O(n·block_size) peak memory via block-wise tiling',
        'paper': 'Ring Attention with Blockwise Transformers for Near-Infinite Context'
    },

    'rpc': {
        'class': RPCAttention,
        'description': (
            'Robust PCA attention that decomposes the attention signal into a low-rank '
            'plus sparse structure (via an iterative PCP-style refinement with an SVD '
            'threshold) to suppress noise/outliers in the attention map. CALL-SIG '
            'CAVEAT: rpc.call uses a `mask` parameter (NOT `attention_mask`); this is a '
            'construction-only registration and the parameter name is a documented, '
            'intentional quirk, not renamed (Invariant 5). NOTE: lambda_sparse is a '
            'sparsity-regularization weight (>0), NOT a 0-1 dropout-style rate.'
        ),
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'lambda_sparse': 0.1,
            'max_pcp_iter': 10,
            'svd_threshold': 1.0,
            'qkv_bias': False,
            'dropout_rate': 0.0,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'probability_type': 'softmax',
            'probability_config': None,
            'qk_norm_type': None,
            'qk_norm_kwargs': None
        },
        'use_case': (
            'Attention scenarios with noisy or outlier-heavy correspondences where a '
            'robust low-rank + sparse decomposition of the attention map improves '
            'stability over standard softmax attention.'
        ),
        'complexity': 'O(n²d) plus iterative PCP/SVD refinement overhead per call',
        'paper': 'Robust PCA Attention (low-rank + sparse decomposition)'
    }
}
"""
Comprehensive registry of attention layer implementations with detailed metadata.

Each entry contains:
    - class: The actual layer class implementation
    - description: Detailed technical description of the mechanism
    - required_params: List of mandatory parameters for instantiation
    - optional_params: Dict of optional parameters with default values
    - use_case: Specific scenarios and applications where this attention excels
    - complexity: Computational complexity analysis compared to alternatives
    - paper: Reference to the original research paper
"""


# ---------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------

def get_attention_info() -> Dict[str, Dict[str, Any]]:
    """Return the metadata of every registered attention type.

    One entry per registry key, each a SHALLOW copy: the outer dict is new, so
    adding or deleting a key cannot reach the registry, but the nested
    ``optional_params`` dict and ``required_params`` list are the registry's own
    objects. Do not mutate them in place.

    :return: Mapping from attention type to its metadata. Every value carries
        ``class``, ``complexity``, ``description``, ``optional_params``,
        ``paper``, ``required_params`` and ``use_case``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {
        attn_type: info.copy() for attn_type, info in ATTENTION_REGISTRY.items()
    }


def validate_attention_config(attention_type: str, **kwargs: Any) -> None:
    """Check a configuration without building anything.

    A pre-flight check for a caller who wants to fail early, or who validates
    here and then constructs the layer class directly. It runs the same checks
    ``create_attention_layer`` runs, in this order: undeclared keyword, unknown
    type, missing required parameter, then the value checks (positive integers,
    rates in ``[0, 1]``, positive ratios, and the two type-specific rules for
    'group_query' and 'differential').

    Returns ``None`` on success. The only signal is the exception.

    :param attention_type: The attention layer type to validate against.
    :type attention_type: str
    :param kwargs: The parameters to validate for that type.
    :raises ValueError: If a keyword is not declared by the type, if
        ``attention_type`` is unknown, if a required parameter is missing, or if
        a value violates a range or a type-specific constraint.
    """
    # DECISION plan-2026-08-27T040114-580f8b63/D-025 — the undeclared-key check
    # lives HERE as well as in `create_attention_layer`. Do NOT delete either
    # copy: this function is documented as a PRE-FLIGHT check and used to pass an
    # undeclared key silently, and the other copy runs BEFORE it, which is what
    # preserves the unknown-type failure mode. See decisions.md D-025.
    _info = ATTENTION_REGISTRY.get(attention_type)
    if _info is not None:
        _declared = set(_info['required_params']) | set(
            _info['optional_params'].keys()
        )
        _undeclared = sorted(set(kwargs) - _declared)
        if _undeclared:
            raise ValueError(
                f"validate_attention_config('{attention_type}'): "
                f"{len(_undeclared)} {STRICT_DROPPED_KEY_MARKER} {_undeclared}. "
                f"'{attention_type}' ({_info['class'].__name__}) accepts only "
                f"{sorted(_declared)}."
            )
    if attention_type not in ATTENTION_REGISTRY:
        available_types = list(ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Available types: {available_types}"
        )

    info = ATTENTION_REGISTRY[attention_type]
    required = info['required_params']
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{attention_type}' are missing: {missing}. "
            f"Required: {required}, Provided: {list(kwargs.keys())}"
        )

    # NOTE (documented shape, not fixed here): everything below is a FLAT
    # ALLOWLIST OF PARAMETER NAMES applied uniformly to all 33 attention types.
    # There are no per-type schemas. A parameter is range-checked because of its
    # NAME, wherever it appears, and a type-specific bound cannot be expressed.
    # Per-type schemas would be the right shape; converting is out of scope for a
    # behavior-preserving pass, because it changes raised message text that tests
    # match on.
    # Validate positive integer parameters
    positive_int_params = [
        'dim', 'channels', 'attention_channels', 'num_heads', 'num_kv_heads',
        'window_size', 'head_dim', 'kv_latent_dim'
    ]
    # DECISION plan-2026-08-11T012340-f63796dc/D-006 — the COMPONENTS are compared,
    # not the value, because `window_size` is a scalar edge length for
    # 'window'/'window_zigzag'/'single_window' but a `(Wh, Ww)` grid for 'beit'; a
    # bare `<= 0` raises a TypeError the caller re-wraps as a vague "parameter
    # compatibility" ValueError. Scalars are wrapped in a 1-tuple, unchanged.
    #   * Do NOT special-case 'beit' here — this is a FLAT allowlist of NAMES.
    #   * Do NOT drop `window_size` from `positive_int_params`; that also removes
    #     the `> 0` guard from the three scalar-window types.
    #   * Do NOT rename `BeitAttention.window_size`; it moves the hole one param on.
    # See decisions.md D-006 (plan-2026-08-11T012340-f63796dc).
    for param in positive_int_params:
        if param not in kwargs:
            continue
        value = kwargs[param]
        components = value if isinstance(value, (tuple, list)) else (value,)
        if any(component <= 0 for component in components):
            raise ValueError(
                f"Parameter '{param}' must be positive, got {value}"
            )

    # Validate positive float parameters
    positive_float_params = []
    for param in positive_float_params:
        if param in kwargs and kwargs[param] <= 0:
            raise ValueError(
                f"Parameter '{param}' must be positive, got {kwargs[param]}"
            )

    # Validate probability/rate parameters (0.0 to 1.0)
    rate_params = [
        'dropout_rate', 'attention_dropout_rate', 'lambda_init', 'rope_percentage'
    ]
    for param in rate_params:
        if param in kwargs and not (0.0 <= kwargs[param] <= 1.0):
            raise ValueError(
                f"Parameter '{param}' must be between 0.0 and 1.0, "
                f"got {kwargs[param]}"
            )

    # Validate ratio parameters (must be positive)
    if 'ratio' in kwargs and kwargs['ratio'] <= 0:
        raise ValueError(f"Parameter 'ratio' must be positive, got {kwargs['ratio']}")
    if 'reduction_ratio' in kwargs and kwargs['reduction_ratio'] <= 0:
        raise ValueError(
            f"Parameter 'reduction_ratio' must be positive, "
            f"got {kwargs['reduction_ratio']}"
        )

    # Validate max_seq_len parameter
    if 'max_seq_len' in kwargs and kwargs['max_seq_len'] <= 0:
        raise ValueError(
            f"Parameter 'max_seq_len' must be positive, got {kwargs['max_seq_len']}"
        )

    # Type-specific validations
    if attention_type == 'group_query':
        if ('num_heads' in kwargs and 'num_kv_heads' in kwargs and
                kwargs['num_heads'] % kwargs['num_kv_heads'] != 0):
            raise ValueError(
                f"For group_query attention, num_heads ({kwargs['num_heads']}) "
                f"must be divisible by num_kv_heads ({kwargs['num_kv_heads']})"
            )

    if attention_type == 'differential':
        if all(p in kwargs for p in ('dim', 'num_heads', 'head_dim')):
            expected = kwargs['num_heads'] * kwargs['head_dim']
            if kwargs['dim'] != expected:
                raise ValueError(
                    f"For differential attention, dim ({kwargs['dim']}) must "
                    f"equal num_heads * head_dim "
                    f"({kwargs['num_heads']} * {kwargs['head_dim']} = {expected})"
                )

    logger.debug(
        f"Validation successful for '{attention_type}' with parameters: {kwargs}"
    )


#: The stable substring every strict dropped-key ``ValueError`` from
#: :func:`create_attention_layer` carries. Guards match on THIS constant instead
#: of re-typing the phrase, so rewording the message cannot silently blind them,
#: and so the phrase has exactly one home. ``layers/ffn/factory.py`` and
#: ``layers/embedding/factory.py`` carry a constant of the same name with the
#: same wording, on purpose — same contract.
#: NOTE for test authors: the ``(s)`` makes this string a REGEX with a group, so
#: ``pytest.raises(match=...)`` needs ``re.escape()`` around it (or use a plain
#: ``in str(excinfo.value)`` substring check, as the FFN tests do).
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"

#: Keys accepted by :func:`create_attention_layer` that are NOT registry
#: parameters. ``name`` is bound by the function's own signature (it never
#: reaches ``**kwargs``), so it is listed here only for wrapper pre-filtering
#: through :func:`assemble_attention_config`, which builds the dict a wrapper
#: later splats.
_ATTENTION_CONFIG_PASSTHROUGH_KEYS: Sequence[str] = ('name',)


def assemble_attention_config(
        attention_type: str,
        wrapper_config: Mapping[str, Any],
        caller_args: Optional[Mapping[str, Any]] = None,
        *,
        passthrough: Sequence[str] = _ATTENTION_CONFIG_PASSTHROUGH_KEYS,
) -> Dict[str, Any]:
    """Filter a WRAPPER's own generic defaults, then merge the CALLER's args.

    Two dicts go in, one goes out. The wrapper's dict is filtered against the
    target type; the caller's dict is not filtered at all. The ORDER is the
    point, and it is why this function owns the merge.

    **Architecture Overview:**

    .. code-block:: text

        wrapper_config              caller_args
        (the wrapper's own          (the user's own
         generic defaults)           attention_args)
              │                            │
              ▼                            │
        ┌─────────────────────────────┐    │
        │ keep only keys in           │    │
        │   required_params           │    │
        │ | optional_params           │    │
        │ | passthrough  ('name')     │    │
        └─────────────────────────────┘    │
              │                            │
              └─────────────┬──────────────┘
                            ▼
                    dict.update: the caller wins
                            ▼
                   config for create_attention_layer
                            ▼
              a caller key the type does not accept
              still reaches the factory, and the
              factory RAISES on it

    Call sites: ``MixedSequentialBlock.__init__`` and
    ``FreeTransformerLayer.__init__``.

    ``wrapper_config`` holds the wrapper layer's own generic conveniences —
    ``dim`` / ``num_heads`` derived from its hyperparameters, ``dropout_rate``,
    ``use_bias``, the initializers and regularizers it hands down. Keys the
    target type does not accept are dropped here, silently and correctly: they
    are the wrapper's defaults, not anybody's expressed intent.

    ``caller_args`` is the end user's own ``attention_args`` dict. It is merged
    on top **verbatim, never filtered**, so it still reaches
    :func:`create_attention_layer`, which raises on a key the type does not
    accept. Filtering it here would turn the caller's typo back into a silent
    drop.

    This helper is a near-copy of the equivalents in ``layers/ffn/factory.py``
    (``assemble_ffn_config``, D-017/D-023) and ``layers/embedding/factory.py``
    rather than a shared generic helper: each of the three binds its own registry
    and its own frozen public surface, so unifying them is a refactor of three
    public APIs, not a de-duplication.

    # DECISION plan-2026-08-17T183311-79c63e38/D-011
    It owns the MERGE, not just the filter — that is why it takes two dicts. Do
    NOT reduce it to a single-dict filter applied AFTER call sites merge their
    ``attention_args`` in: the pre-filter would EAT the caller's keys, so a typo
    could never reach the raise. See decisions.md D-011.

    :param attention_type: An ``ATTENTION_REGISTRY`` key.
    :type attention_type: str
    :param wrapper_config: The wrapper's own generic config; filtered.
    :type wrapper_config: Mapping[str, Any]
    :param caller_args: The caller's explicit args; NEVER filtered.
    :type caller_args: Optional[Mapping[str, Any]]
    :param passthrough: Keys kept regardless of the registry intersection.
    :type passthrough: Sequence[str]
    :return: The assembled config dict for :func:`create_attention_layer`.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``attention_type`` is not a registered attention type.
    """
    info = ATTENTION_REGISTRY.get(attention_type)
    if info is None:
        raise ValueError(
            f"Unknown attention_type '{attention_type}'. "
            f"Available: {sorted(ATTENTION_REGISTRY)}."
        )

    accepted = (
        set(info['required_params'])
        | set(info['optional_params'])
        | set(passthrough)
    )
    config = {k: v for k, v in wrapper_config.items() if k in accepted}
    if caller_args:
        config.update(caller_args)
    return config


def create_attention_layer(
        attention_type: AttentionType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """Build one attention layer. STRICT: an undeclared keyword raises.

    The single construction path for all 33 registered types. It looks
    ``attention_type`` up, rejects any keyword that type does not declare, fills
    the registry defaults in under the caller's values, and constructs. The
    module's Architecture Overview draws the full ordering, including which
    branch each ``ValueError`` comes from.

    Nothing here filters and drops. If a wrapper wants to offer generic
    conveniences that only some types accept, it pre-filters them through
    :func:`assemble_attention_config` first; whatever it then passes here is
    treated as an explicit request and is checked.

    :param attention_type: The type of attention layer to create.
    :type attention_type: AttentionType
    :param name: Optional name for the layer instance.
    :type name: Optional[str]
    :param kwargs: Type-specific parameters for the attention layer. See
        ``get_attention_info()`` for detailed parameter specifications. This is
        STRICT: any key ``attention_type`` does not accept raises rather than
        being silently dropped. A wrapper that wants to offer generic
        conveniences should pre-filter them through
        :func:`assemble_attention_config` before calling here.
    :return: A fully configured and instantiated attention layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If attention_type is invalid, required parameters are missing,
        parameter values are out of valid ranges, a supplied keyword is not a
        parameter of ``attention_type`` (the message then carries
        :data:`STRICT_DROPPED_KEY_MARKER`), or layer construction fails.
    :raises TypeError: ONLY if ``attention_type`` is unhashable (a list, say), so
        the registry lookup itself fails before the ``try`` below. A ``TypeError``
        raised by the target layer's constructor never escapes: it is caught and
        re-raised as ``ValueError``. Measured 2026-08-28, ``dim='not-an-int'``
        gives ``ValueError`` from both entry points.
    """
    # DECISION plan-2026-08-17T183311-79c63e38/D-011 — RAISE, do not drop. Since
    # 2026-08-17 this factory raises on a keyword the type does not declare. It
    # used to filter `kwargs` against the declared names and discard the rest
    # without a word, and that silent drop masked two real defects here: TRM
    # advertised `rope_theta` on every constructor and handed it to
    # `'multi_head'`, which declares no RoPE parameter at all, leaving the whole
    # reasoning stack exactly permutation-equivariant; and
    # `MixedSequentialBlock`'s `'window'` branch passed a `normalization` choice
    # `WindowAttention` has no parameter for, since the day it was written. Same
    # fix as the sibling factories `layers/ffn/factory.py` (D-023) and
    # `layers/embedding/factory.py`.
    #
    # Three things below must not change:
    #
    #   * Subtract `valid_param_names` (required | optional), NOT just
    #     `required_params`. The narrower right-hand side is the tempting
    #     "simplification" and it is CATASTROPHIC: every optional parameter
    #     anyone passes becomes an error.
    #   * Read `kwargs`, not the merged `params` dict. The two agree today, since
    #     `params` is `optional_params` updated with `kwargs` and every
    #     `optional_params` key is already declared. The `kwargs` form survives
    #     that relation breaking — a registry entry gaining an `optional_params`
    #     key its class does not accept — where the `params` form would blame the
    #     caller for the registry's own bug.
    #   * Keep this check BEFORE the `try`. Copying
    #     `layers/ffn/factory.py`'s version verbatim goes wrong here because that
    #     factory has no outer error wrapper and this one does: the
    #     `except (TypeError, ValueError)` below re-wraps every ValueError into a
    #     generic "Please verify parameter compatibility" message, which would
    #     swallow the wording and bury STRICT_DROPPED_KEY_MARKER in a nested
    #     message. Do not move it inside, and do not fold it into the filter at
    #     the merge step.
    #
    # Unknown-`attention_type` calls fall through untouched: `_info` is None
    # here, so they keep their existing failure mode from
    # `validate_attention_config` inside the `try`.
    # See decisions.md D-011 (plan-2026-08-17T183311-79c63e38).
    _info = ATTENTION_REGISTRY.get(attention_type)
    if _info is not None:
        _valid_param_names = set(_info['required_params']) | set(
            _info['optional_params'].keys()
        )
        dropped = sorted(set(kwargs) - _valid_param_names)
        if dropped:
            raise ValueError(
                f"create_attention_layer('{attention_type}'): "
                f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
                f"'{attention_type}' ({_info['class'].__name__}) accepts only "
                f"{sorted(_valid_param_names)}. "
                f"Either you mistyped one of those names, or you chose the "
                f"wrong attention_type for the parameters you are passing. If "
                f"these keys are a WRAPPER's own generic defaults rather than "
                f"an explicit request, pre-filter them with "
                f"assemble_attention_config() instead of passing them here."
            )

    try:
        # Validate configuration before proceeding
        validate_attention_config(attention_type, **kwargs)

        # Get layer information and class
        info = ATTENTION_REGISTRY[attention_type]
        attn_class = info['class']

        # Merge user parameters with defaults
        params = info['optional_params'].copy()
        params.update(kwargs)

        # Filter parameters to match constructor signature
        valid_param_names = set(info['required_params']) | set(
            info['optional_params'].keys()
        )
        final_params = {
            k: v for k, v in params.items() if k in valid_param_names
        }

        # Add name if provided
        if name:
            final_params['name'] = name

        logger.info(
            f"Creating '{attention_type}' attention layer "
            f"({attn_class.__name__}) with parameters: {final_params}"
        )

        # Instantiate the attention layer
        return attn_class(**final_params)

    except (TypeError, ValueError) as e:
        # Provide detailed error context
        info = ATTENTION_REGISTRY.get(attention_type)
        if info:
            class_name = info['class'].__name__
            error_msg = (
                f"Failed to create '{attention_type}' attention layer "
                f"({class_name}). "
                f"Required parameters: {info['required_params']}. "
                f"Provided parameters: {list(kwargs.keys())}. "
                f"Please verify parameter compatibility. Original error: {e}"
            )
        else:
            error_msg = (
                f"Failed to create attention layer. "
                f"Unknown type '{attention_type}'. Error: {e}"
            )

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_attention_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Build an attention layer from a single ``{'type': ..., ...}`` dict.

    Pops ``'type'`` off a copy of ``config`` and splats the rest into
    :func:`create_attention_layer`, so every rule that function enforces applies
    here too — including the strict keyword check. The input dict is not
    mutated. Useful for JSON/YAML architectures and hyperparameter sweeps.

    :param config: Configuration dict with a ``'type'`` key naming the attention
        type, plus that type's parameters.
    :type config: Dict[str, Any]
    :return: The instantiated attention layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict, if it has no ``'type'`` key,
        or for any reason :func:`create_attention_layer` raises.
    :raises TypeError: ONLY if ``config['type']`` is unhashable, which fails the
        registry lookup inside :func:`create_attention_layer` before its ``try``.
        Every other type error surfaces as ``ValueError``; see that function's
        own ``:raises TypeError:`` note.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a dictionary, got {type(config).__name__}. "
            f"Expected format: {{'type': 'attention_type', ...}}"
        )

    if 'type' not in config:
        available_keys = list(config.keys()) if config else []
        raise ValueError(
            f"Configuration dictionary must include a 'type' key specifying the "
            f"attention layer type. Available keys in config: {available_keys}. "
            f"Valid attention types: {list(ATTENTION_REGISTRY.keys())}"
        )

    # Extract type and pass remaining parameters
    config_copy = config.copy()
    attention_type = config_copy.pop('type')

    logger.debug(f"Creating attention layer from config: {config}")
    return create_attention_layer(attention_type, **config_copy)


def list_attention_types() -> List[str]:
    """List every registered attention type key.

    Sorted alphabetically, so the order is stable across runs and does not
    follow the registry's insertion order.

    :return: The registry's keys, sorted.
    :rtype: List[str]
    """
    return sorted(list(ATTENTION_REGISTRY.keys()))


def get_attention_requirements(attention_type: str) -> Dict[str, Any]:
    """Return one attention type's registry entry.

    The same seven keys :func:`get_attention_info` returns per entry, for a
    single type. A SHALLOW copy: the nested ``optional_params`` dict and
    ``required_params`` list are the registry's own objects, so do not mutate
    them in place.

    :param attention_type: The attention layer type to query.
    :type attention_type: str
    :return: That entry's ``class``, ``complexity``, ``description``,
        ``optional_params``, ``paper``, ``required_params`` and ``use_case``.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``attention_type`` is not registered. The message
        lists every available type.
    """
    if attention_type not in ATTENTION_REGISTRY:
        available_types = list(ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Available types: {available_types}"
        )

    return ATTENTION_REGISTRY[attention_type].copy()
