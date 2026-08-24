"""
Attention Layer Factory

A comprehensive factory system for creating and managing various attention mechanisms
with unified interfaces, type safety, parameter validation, and detailed documentation.
This factory enables seamless integration and experimentation with different attention
types across vision_heads, NLP, and multi-modal architectures.

The factory supports thirty-two different attention mechanisms, from standard multi-head attention
to specialized variants like differential attention, mobile-optimized MQA, and hierarchical
anchor attention. Each layer is fully documented with use cases, parameter requirements,
and architectural considerations.

Key Features:
    - Type-safe attention layer creation with comprehensive validation
    - Unified interface across all attention mechanisms
    - Detailed parameter documentation and error handling
    - Support for both dictionary-based and direct configuration
    - Integration with the dl_techniques logging system
    - Complete compatibility with Keras 3 serialization

FROZEN PUBLIC SURFACE
---------------------
`ATTENTION_REGISTRY`'s key set, the `AttentionType` literals, and every entry's
`required_params` / `optional_params` are **public API** consumed by config-driven
callers (`layers/transformers/adaln_zero.py`, `models/bias_free_denoisers/bfconvunext.py`,
`models/fastvlm/`, `models/dino/`, `models/gemma/`) and asserted by
`tests/test_layers/test_factory_registry_drift.py`. Adding, renaming, or removing any of
them is a breaking change, not a cleanup. Docstrings and comments in this module may be
improved freely; the data above may not.

Registry entries whose 'class' is NOT a class
---------------------------------------------
Two of the 32 entries map to module-level FUNCTIONS rather than layer classes:

    'window'         -> create_grid_window_attention    (window_attention.py)
    'window_zigzag'  -> create_zigzag_window_attention  (window_attention.py)

Both wrappers construct a `WindowAttention` locked to one `partition_mode` ('grid' /
'zigzag') carrying that mode's `use_relative_position_bias` default — a distinction the
class itself does not encode, which is exactly why the keys point at the wrappers.
Consequence: the general `WindowAttention` class, the one whose `partition_mode` the
caller chooses, has **no factory key of its own** and must be imported directly
(`from dl_techniques.layers.attention import WindowAttention`).

`window_attention.py` also defines `create_kan_key_window_attention` and
`create_adaptive_softmax_window_attention`. They are **intentionally non-public**:
registered here by neither key nor import, absent from `attention/__init__.py`, and
called only by `tests/test_layers/test_attention/test_window_attention.py`. Both
configurations are reachable by passing `attention_mode='kan_key'` /
`probability_type='adaptive'` to `WindowAttention`. Do not register them "for
consistency" — that grows the frozen surface above.

Known shape of `validate_attention_config` (documented, not a defect to fix here)
--------------------------------------------------------------------------------
The numeric checks in `validate_attention_config` below are a single flat allowlist of
parameter NAMES ('dim', 'num_heads', 'dropout_rate', ...) applied uniformly to all 32
types, not per-type schemas. A parameter therefore gets range-checked purely because of
what it is called, and a type-specific constraint (e.g. one type's `window_size` upper
bound, or a parameter two types interpret differently) cannot be expressed. The one
concession to that shape is that the positive-value check compares a sequence value
COMPONENTWISE, because `window_size` is a scalar edge length for three types and a
`(Wh, Ww)` grid for 'beit' (D-006); it is still keyed on the NAME, not on the type.
Per-type
schemas would be the correct shape; converting is out of scope for a behavior-preserving
pass and would change raised message text that tests match on.
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
    create_grid_window_attention
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
    'window_zigzag'
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
            # DECISION plan-2026-08-22T035419-a11304c8/D-160
            # Declared on 'multi_head' and 'multi_head_cross' ONLY, of the 32
            # registered types: these two are the ones whose output projection
            # is the transformer block's residual-path projection (GPT-2's
            # `attn.c_proj`). Leaving it undeclared everywhere else is
            # load-bearing -- it turns
            # `TransformerLayer(residual_output_kernel_initializer=...,
            # attention_type=<anything else>)` into a LOUD
            # `create_attention_layer` raise rather than a silently-ignored
            # request. See decisions.md D-160.
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
            'READ THE COMPLEXITY FIELD BEFORE PICKING THIS FOR EFFICIENCY: a 1-D input of '
            'length N is folded into a ceil(sqrt(N))-square grid and every window is PADDED '
            'up to window_size**2 slots, so cost is O(max(N, M) * M) with M = window_size**2. '
            'It beats global attention only for N > M. For N <= M the grid pads to a SINGLE '
            'window and this layer computes dense attention over M padded positions — '
            '(M/N)**2 times MORE work than plain global attention over the N real tokens.'
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
            'O(max(N, M) * M) with M = W**2 slots per window (W = window_size); '
            'linear in N only for N > M, and a constant O(M**2) = O(W**4) floor for '
            'N <= M, where it degenerates to dense attention over one padded window'
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
            'O(max(N, M) * M) with M = W**2, same as standard window attention above — '
            'including the O(M**2) floor: the zigzag path pads the reordered sequence up '
            'to a multiple of M as well, so N <= M is one dense padded window here too'
        ),
        'paper': "Extends 'Swin Transformer' with zigzag partitioning and advanced normalization"
    },

    # DECISION plan_2026-06-14_0c5d4a21/D-007: gated/performer/ring/rpc registered
    # (construction-only); performer/rpc call-mask quirks documented not renamed
    # (F1, user D1). Do NOT rename performer.call (no attention_mask param) or
    # rpc.call (uses `mask` not `attention_mask`) — that is behavior-touching and
    # out of scope (Invariant 5). The factory is construction-only; these are
    # documented caveats, not registration blockers. See decisions.md D-007.
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
            # DECISION plan-2026-08-17T183311-79c63e38/D-011
            # `num_kv_heads` is DECLARED here, and this addition is the one
            # exception to the FROZEN PUBLIC SURFACE note at the top of this
            # module (see decisions.md D-008/D-011). `GatedAttention.__init__`
            # has always ACCEPTED it (gated_attention.py:250, default `None`
            # meaning "one K/V head per query head"); the entry omitted it, so
            # `tests/test_layers/test_factory_registry_drift.py::
            # test_registry_declares_every_constructor_param[attention:gated]`
            # was RED. Do NOT "fix" that red node by deleting the parameter or
            # by exempting the entry: with the strict raise below, an undeclared
            # `num_kv_heads` turns a silently-ignored-but-supported parameter
            # into a hard ValueError for a genuinely supported feature. The
            # declaration must land in the SAME commit as the raise.
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
    """
    Retrieve comprehensive information about all available attention layer types.

    This function provides complete metadata for each supported attention mechanism,
    including technical descriptions, parameter specifications, use cases, and
    computational complexity analysis.

    :return: Comprehensive attention layer information containing description,
        required_params, optional_params, use_case, complexity, and paper
        for each attention type.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {
        attn_type: info.copy() for attn_type, info in ATTENTION_REGISTRY.items()
    }


def validate_attention_config(attention_type: str, **kwargs: Any) -> None:
    """
    Validate attention layer configuration parameters against type requirements.

    Performs comprehensive validation including type existence, required parameter
    completeness, value range validation, and type-specific constraint verification.

    :param attention_type: The attention layer type to validate against.
    :type attention_type: str
    :param kwargs: Parameter dictionary to validate for the specified attention type.
    :raises ValueError: If attention_type is not supported, required parameters are missing,
        or parameter values violate constraints.
    """
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

    # NOTE (documented shape, not fixed here): everything below is a FLAT ALLOWLIST OF
    # PARAMETER NAMES applied uniformly to all 32 attention types — there are no per-type
    # schemas. A parameter is range-checked because of its NAME, wherever it appears, and
    # a type-specific bound cannot be expressed. Per-type schemas would be the right
    # shape; the conversion is out of scope for a behavior-preserving pass because it
    # changes raised message text that tests match on.
    # Validate positive integer parameters
    positive_int_params = [
        'dim', 'channels', 'attention_channels', 'num_heads', 'num_kv_heads',
        'window_size', 'head_dim', 'kv_latent_dim'
    ]
    # DECISION plan-2026-08-11T012340-f63796dc/D-006
    # The components are compared, not the value itself, because a value in this list
    # may legitimately be a SEQUENCE: `window_size` is a scalar edge length for
    # 'window'/'window_zigzag'/'single_window' but a `(Wh, Ww)` patch grid for 'beit'.
    # A bare `kwargs[param] <= 0` raises `TypeError: '<=' not supported between
    # instances of 'tuple' and 'int'`, which `create_attention_layer` then catches and
    # re-raises as a ValueError about "parameter compatibility" — an error that names
    # neither the parameter nor the real cause, for a configuration that is valid.
    #
    # WHAT NOT TO DO, and why:
    #   * Do NOT special-case `'beit'` (or any type) here. This validator is
    #     deliberately a FLAT allowlist of parameter NAMES applied to every type (see
    #     the module docstring); adding a per-type branch starts the per-type-schema
    #     conversion that docstring rules out of scope, one exception at a time.
    #   * Do NOT drop `window_size` from `positive_int_params` to dodge the TypeError.
    #     That silently removes the `> 0` guard from the three scalar-window types too.
    #   * Do NOT "fix" this in `BeitAttention` by renaming its `window_size`. The name
    #     is what makes it match the 'window'/'single_window' registry precedent, and
    #     renaming would only move the same hole to the next sequence-valued parameter.
    # Scalar behaviour is unchanged (a scalar is wrapped in a 1-tuple), and the raised
    # message still reports the whole value. Pinned by
    # `TestBeitAttentionFactory::test_tuple_window_size_survives_validation` and
    # `::test_tuple_window_size_still_rejects_a_non_positive_component`.
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
#: :func:`create_attention_layer` carries. Guards match on THIS constant rather
#: than re-typing the phrase, so rewording the message cannot silently blind them
#: (and so the phrase has exactly one home). Mirrors ``layers/ffn/factory.py``
#: and ``layers/embedding/factory.py``'s constants of the same name -- same
#: contract, same wording, deliberately.
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
    """Pre-filter a WRAPPER's own generic attention conveniences, then merge the CALLER's.

    Interface contract (call sites: ``MixedSequentialBlock.__init__`` and
    ``FreeTransformerLayer.__init__``):

    * ``wrapper_config`` -- the wrapper layer's OWN generic conveniences
      (``dim``/``num_heads`` derived from its own hyperparameters,
      ``dropout_rate``, ``use_bias``, the initializers/regularizers it hands
      down, ...). It is INTERSECTED with ``ATTENTION_REGISTRY[attention_type]``'s
      ``required_params | optional_params``, plus ``passthrough``. Keys the
      target type does not accept are dropped HERE, silently and correctly --
      they are this wrapper's defaults, not anybody's expressed intent.
    * ``caller_args`` -- the end user's own ``attention_args`` dict. Merged on
      top of the filtered result **verbatim, never filtered**, and therefore
      still reaches :func:`create_attention_layer`, which RAISES on a key the
      type does not accept. A caller key the type does not accept must stay
      visible to the factory so the factory can complain about it; filtering it
      here would turn the caller's typo back into a silent drop.
    * Returns a NEW dict; neither input is mutated.
    * Raises ``ValueError`` naming the available types if ``attention_type`` is
      not in ``ATTENTION_REGISTRY``.

    # DECISION plan-2026-08-17T183311-79c63e38/D-011
    This is the attention twin of ``layers/ffn/factory.py::assemble_ffn_config``
    (D-017/D-023) and it owns the MERGE, not just the filter -- that is why it
    takes two dicts instead of one. Do NOT "simplify" it to a single-dict filter
    that call sites apply AFTER merging their ``attention_args`` in: that
    ordering makes the pre-filter EAT the caller's keys, so a caller typo can
    never reach the raise below. With the merge inside, a call site cannot
    express the wrong order.

    It is a deliberate near-copy rather than a shared generic helper: the FFN,
    embedding and attention factories each bind their own module-level registry
    and their own frozen public surface, and unifying the three is a refactor of
    three public APIs, not a de-duplication. Recorded rather than silently
    repeated.

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
    """
    Factory function for creating attention layers with unified interface and validation.

    This is the primary factory function providing a centralized, type-safe way to
    instantiate any attention layer supported by the framework. It includes comprehensive
    parameter validation, default value handling, and detailed error reporting.

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
    :raises TypeError: If parameter types are incompatible with the target layer class.
    """
    # DECISION plan-2026-08-17T183311-79c63e38/D-011
    # RAISE, do not drop. This factory used to filter `kwargs` against the
    # registry's declared names and discard the rest without a word, and that
    # silent drop has now masked two real defects in this tree: TRM advertised
    # `rope_theta` on every constructor and handed it to `'multi_head'`, which
    # declares no RoPE parameter at all (the whole reasoning stack was exactly
    # permutation-equivariant, D-007), and `MixedSequentialBlock`'s `'window'`
    # branch has been passing a `normalization` choice that `WindowAttention`
    # does not have a parameter for since it was written. Same class of defect
    # as the recorded `ffn_type` regression, same fix as the sibling factories:
    # `layers/ffn/factory.py` (D-023) and `layers/embedding/factory.py`.
    #
    # Both halves of the predicate are load-bearing, as at the FFN site:
    #
    # * subtracting `valid_param_names` (required | optional), NOT just
    #   `required_params`. The narrower right-hand side is the tempting
    #   "simplification" and it is CATASTROPHIC: every optional parameter
    #   anyone passes becomes an error.
    # * reading `kwargs` rather than the merged `params` dict. These are
    #   EXTENSIONALLY EQUAL today, because `params` is `optional_params`
    #   updated with `kwargs` and `optional_params`'s keys are a subset of
    #   `valid_param_names` by construction. The `kwargs` form is kept because
    #   it stays correct if that subset relation ever breaks -- i.e. if a
    #   registry entry gains an `optional_params` key its class does not accept
    #   -- where the `params` form would then blame the caller for the
    #   registry's own bug.
    #
    # PLACEMENT IS LOAD-BEARING, and this is where copying
    # `layers/ffn/factory.py:808-820` verbatim goes wrong: that factory has no
    # outer error wrapper, this one does. The `except (TypeError, ValueError)`
    # below RE-WRAPS every ValueError into a generic "Please verify parameter
    # compatibility" message, which would swallow the wording AND bury
    # STRICT_DROPPED_KEY_MARKER inside a nested message. So the check runs
    # BEFORE the `try`. Do NOT move it inside, and do not "tidy" it by folding
    # it into the existing filter at the merge step. Measured by asserting the
    # message TEXT, not the exception type:
    # `tests/test_layers/test_attention/test_attention_factory.py::
    # TestStrictDroppedKeys`.
    #
    # Unknown-`attention_type` calls fall through untouched: `info` is None
    # here, so they keep their existing failure mode from
    # `validate_attention_config` inside the `try`.
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
    """
    Create an attention layer from a configuration dictionary.

    Convenience function for instantiating attention layers from dictionary-based
    configurations, useful for loading architectures from JSON/YAML files,
    hyperparameter optimization, and configuration-driven model building.

    :param config: Configuration dictionary containing a 'type' key specifying the
        attention layer type and additional keys for layer-specific parameters.
    :type config: Dict[str, Any]
    :return: Instantiated and configured attention layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If config is not a dictionary or missing required 'type' key.
    :raises TypeError: If config parameter types are invalid.
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
    """
    Get a list of all supported attention layer types.

    :return: Alphabetically sorted list of supported attention layer types.
    :rtype: List[str]
    """
    return sorted(list(ATTENTION_REGISTRY.keys()))


def get_attention_requirements(attention_type: str) -> Dict[str, Any]:
    """
    Get parameter requirements for a specific attention layer type.

    Returns detailed parameter information for a single attention type,
    useful for dynamic UI generation, parameter validation, and documentation.

    :param attention_type: The attention layer type to query.
    :type attention_type: str
    :return: Parameter requirements containing required_params, optional_params,
        description, and use_case.
    :rtype: Dict[str, Any]
    :raises ValueError: If attention_type is not supported.
    """
    if attention_type not in ATTENTION_REGISTRY:
        available_types = list(ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Available types: {available_types}"
        )

    return ATTENTION_REGISTRY[attention_type].copy()
