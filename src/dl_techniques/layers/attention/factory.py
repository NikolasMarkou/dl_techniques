"""
Attention Layer Factory

A comprehensive factory system for creating and managing various attention mechanisms
with unified interfaces, type safety, parameter validation, and detailed documentation.
This factory enables seamless integration and experimentation with different attention
types across vision_heads, NLP, and multi-modal architectures.

The factory supports twenty-one different attention mechanisms, from standard multi-head attention
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

Example:
    >>> # Create a standard multi-head attention layer
    >>> mha = create_attention_layer('multi_head', dim=512, num_heads=8)
    >>>
    >>> # Create an efficient mobile MQA layer
    >>> mobile_attn = create_attention_layer('mobile_mqa', dim=256, use_downsampling=True)
    >>>
    >>> # Create TripSE1 for 3D attention
    >>> tripse = create_attention_layer('tripse1', reduction_ratio=0.0625, kernel_size=7)
"""

import keras
from typing import Dict, Any, Literal, Optional, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .anchor_attention import AnchorAttention
from .capsule_routing_attention import CapsuleRoutingSelfAttention
from .channel_attention import ChannelAttention
from .convolutional_block_attention import CBAM
from .differential_attention import DifferentialMultiHeadAttention
from .fnet_fourier_transform import FNetFourierTransform
from .group_query_attention import GroupedQueryAttention
from .hopfield_attention import HopfieldAttention
from .mobile_mqa import MobileMQA
from .multi_head_attention import MultiHeadAttention
from .multi_head_cross_attention import MultiHeadCrossAttention
from .non_local_attention import NonLocalAttention
from .perceiver_attention import PerceiverAttention
from .shared_weights_cross_attention import SharedWeightsCrossAttention
from .spatial_attention import SpatialAttention
from .tripse_attention import TripSE1, TripSE2, TripSE3, TripSE4
from .window_attention import (
    create_zigzag_window_attention,
    create_grid_window_attention
)

# ---------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------

AttentionType = Literal[
    'anchor',
    'capsule_routing',
    'cbam',
    'channel',
    'differential',
    'fnet',
    'group_query',
    'hopfield',
    'mobile_mqa',
    'multi_head',
    'multi_head_cross',
    'non_local',
    'perceiver',
    'shared_weights_cross',
    'spatial',
    'tripse1',
    'tripse2',
    'tripse3',
    'tripse4',
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
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8,
            'dropout_rate': 0.0,
            'use_bias': True,
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
            'epsilon': 1e-8
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
            'use_bias': False
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
            'bias_regularizer': None
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
            'normalize_patterns': True,
            'update_steps_max': 0,
            'update_steps_eps': 1e-4
        },
        'use_case': (
            'Associative memory tasks, pattern completion, and robust retrieval with noise '
            'correction. With update_steps_max=0 functions as attention; with >0 enables '
            'powerful iterative pattern retrieval and memory consolidation.'
        ),
        'complexity': 'O(n²) per update step, configurable iteration count',
        'paper': 'Hopfield Networks is All You Need'
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
            'kernel_regularizer': None,
            'use_bias': False
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
            'bias_initializer': "zeros",
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'use_hierarchical_routing': False,
            'use_adaptive_softmax': False,
            'adaptive_softmax_config': None,
        },
        'use_case': (
            'Core component for encoder-decoder models, Perceiver-style architectures, and any '
            'scenario requiring interaction between two distinct sequences. Can also be used '
            'for self-attention with fine-grained control over projections.'
        ),
        'complexity': 'O(nm*d) where n is query length, m is key/value length',
        'paper': 'Attention Is All You Need'
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
            'normalization': 'batch',
            'intermediate_activation': 'relu',
            'output_activation': 'linear',
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
            'bias_regularizer': None
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
            'bias_regularizer': None
        },
        'use_case': (
            'Multi-modal fusion with parameter constraints: vision_heads-language models, '
            'audio-visual processing, and sensor fusion in robotics. Ideal when model '
            'size is limited but cross-modal interaction is essential.'
        ),
        'complexity': 'Reduces cross-attention parameters by ~50%',
        'paper': 'Shared-weight Cross-attention for Multi-modal Fusion'
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
            'use_bias': True
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
            'kernel_regularizer': None
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
            'kernel_regularizer': None
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
            'kernel_regularizer': None
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
            'kernel_regularizer': None
        },
        'use_case': (
            'Advanced vision tasks where the correlation between "where" (spatial) and '
            '"what" (channel) is highly entangled. Provides the most strictly defined '
            '"3D attention" mechanism among the variants.'
        ),
        'complexity': 'Similar to TripSE2 but with broadcasted logit fusion',
        'paper': 'Achieving 3D Attention via Triplet Squeeze and Excitation Block'
    },

    'window': {
        'class': create_grid_window_attention,
        'description': (
            'Windowed multi-head self-attention from Swin Transformer, partitioning inputs '
            'into non-overlapping grids for local attention computation. Achieves linear '
            'complexity while maintaining spatial awareness through an optional learnable '
            'relative position bias.'
        ),
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'qkv_bias': True,
            'qk_scale': None,
            'dropout_rate': 0.0,
            'proj_bias': True,
            'attention_mode': 'linear',
            'normalization': 'softmax',
            'use_relative_position_bias': True,
            'adaptive_softmax_config': None,
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
        'complexity': 'O(W²) per window vs O(n²) global attention',
        'paper': 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows'
    },
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
            'normalization': 'softmax',
            'use_relative_position_bias': False,
            'adaptive_softmax_config': None,
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
        'complexity': 'O(W²) per window, same as standard window attention',
        'paper': "Extends 'Swin Transformer' with zigzag partitioning and advanced normalization"
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
    computational complexity analysis. Essential for understanding which attention
    mechanism is appropriate for specific architectural requirements.

    Returns:
        Dict[str, Dict[str, Any]]: Comprehensive attention layer information containing:
            - description: Technical mechanism description
            - required_params: List of mandatory instantiation parameters
            - optional_params: Dict of optional parameters with defaults
            - use_case: Recommended applications and scenarios
            - complexity: Computational complexity analysis
            - paper: Reference to original research

    Example:
        >>> info = get_attention_info()
        >>> # Explore available attention mechanisms
        >>> for attn_type, details in info.items():
        ...     print(f"{attn_type}: {details['description'][:100]}...")
        ...     print(f"  Required: {details['required_params']}")
        ...     print(f"  Complexity: {details['complexity']}")
        ...     print()
        >>>
        >>> # Check specific attention requirements
        >>> gqa_info = info['group_query']
        >>> required = gqa_info['required_params']  # ['dim', 'num_heads', 'num_kv_heads']
    """
    return {
        attn_type: info.copy() for attn_type, info in ATTENTION_REGISTRY.items()
    }


def validate_attention_config(attention_type: str, **kwargs: Any) -> None:
    """
    Validate attention layer configuration parameters against type requirements.

    Performs comprehensive validation of attention layer parameters including:
    - Attention type existence validation
    - Required parameter completeness checking
    - Common parameter value range validation
    - Type-specific constraint verification

    This function should be called before layer instantiation to catch configuration
    errors early and provide clear diagnostic messages.

    Args:
        attention_type (str): The attention layer type to validate against.
        **kwargs: Parameter dictionary to validate for the specified attention type.

    Raises:
        ValueError: If attention_type is not supported, required parameters are missing,
                   or parameter values violate constraints.

    Example:
        >>> # Validate configuration before creation
        >>> try:
        ...     validate_attention_config(
        ...         'group_query',
        ...         dim=1024,
        ...         num_heads=16,
        ...         num_kv_heads=4
        ...     )
        ...     print("Configuration is valid")
        >>> except ValueError as e:
        ...     print(f"Validation failed: {e}")
        >>>
        >>> # This will raise an error due to missing required parameters
        >>> validate_attention_config('differential', dim=256)  # Missing num_heads, head_dim
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

    # Validate positive integer parameters
    positive_int_params = [
        'dim', 'channels', 'attention_channels', 'num_heads', 'num_kv_heads',
        'window_size', 'head_dim'
    ]
    for param in positive_int_params:
        if param in kwargs and kwargs[param] <= 0:
            raise ValueError(
                f"Parameter '{param}' must be positive, got {kwargs[param]}"
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

    logger.debug(
        f"Validation successful for '{attention_type}' with parameters: {kwargs}"
    )


def create_attention_layer(
        attention_type: AttentionType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating attention layers with unified interface and validation.

    This is the primary factory function providing a centralized, type-safe way to
    instantiate any attention layer supported by the framework. It includes comprehensive
    parameter validation, default value handling, and detailed error reporting to
    facilitate debugging and development.

    The factory approach enables:
    - Easy experimentation with different attention mechanisms
    - Consistent parameter handling across all attention types
    - Type safety with IDE autocompletion support
    - Comprehensive validation and error reporting
    - Simplified model architecture definition

    Args:
        attention_type (AttentionType): The type of attention layer to create.
                                       Must be one of the supported literal types.
        name (Optional[str], default=None): Optional name for the layer instance.
                                          If provided, will be passed to the layer constructor.
        **kwargs: Type-specific parameters for the attention layer. See `get_attention_info()`
                 for detailed parameter specifications for each attention type.

    Returns:
        keras.layers.Layer: A fully configured and instantiated attention layer ready
                           for integration into model architectures.

    Raises:
        ValueError: If attention_type is invalid, required parameters are missing,
                   parameter values are out of valid ranges, or layer construction fails.
        TypeError: If parameter types are incompatible with the target layer class.

    Example:
        >>> # Standard multi-head attention for transformers
        >>> mha = create_attention_layer('multi_head', dim=512, num_heads=8, dropout_rate=0.1)
        >>>
        >>> # Mobile-optimized attention for edge deployment
        >>> mobile_attn = create_attention_layer(
        ...     'mobile_mqa',
        ...     dim=256,
        ...     num_heads=8,
        ...     use_downsampling=True,
        ...     name='mobile_attention_1'
        ... )
        >>>
        >>> # CBAM for CNN enhancement
        >>> cbam_block = create_attention_layer(
        ...     'cbam',
        ...     channels=128,
        ...     ratio=16,
        ...     kernel_size=7
        ... )
        >>>
        >>> # Grouped Query Attention for large language models
        >>> gqa = create_attention_layer(
        ...     'group_query',
        ...     dim=2048,
        ...     num_heads=32,
        ...     num_kv_heads=8,
        ...     max_seq_len=4096
        ... )

    Note:
        The factory automatically merges user-provided parameters with defaults
        from the registry and filters parameters to match the target layer's
        constructor signature, ensuring compatibility across different attention
        implementations.
    """
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
    configurations. This is particularly useful for:
    - Loading model architectures from JSON/YAML files
    - Hyperparameter optimization pipelines
    - Dynamic model architecture generation
    - Configuration-driven model building

    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
                                - 'type': The attention layer type (required)
                                - Additional keys for layer-specific parameters

    Returns:
        keras.layers.Layer: Instantiated and configured attention layer.

    Raises:
        ValueError: If config is not a dictionary or missing required 'type' key.
        TypeError: If config parameter types are invalid.

    Example:
        >>> # Configuration-based layer creation
        >>> window_config = {
        ...     'type': 'window',
        ...     'dim': 96,
        ...     'window_size': 7,
        ...     'num_heads': 3,
        ...     'name': 'swin_attention_stage2'
        ... }
        >>> window_attn = create_attention_from_config(window_config)
        >>>
        >>> # Batch creation from multiple configurations
        >>> attention_configs = [
        ...     {'type': 'multi_head', 'dim': 512, 'num_heads': 8},
        ...     {'type': 'group_query', 'dim': 1024, 'num_heads': 16, 'num_kv_heads': 4},
        ...     {'type': 'cbam', 'channels': 256, 'ratio': 16}
        ... ]
        >>> attention_layers = [create_attention_from_config(cfg) for cfg in attention_configs]
        >>>
        >>> # Integration with hyperparameter optimization
        >>> def build_model_with_attention(attn_config):
        ...     attention = create_attention_from_config(attn_config)
        ...     # ... build rest of model
        ...     return model
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

    Convenience function returning a simple list of supported attention mechanisms
    for programmatic access, iteration, and validation purposes.

    Returns:
        List[str]: Alphabetically sorted list of supported attention layer types.

    Example:
        >>> # Get all available types
        >>> attention_types = list_attention_types()
        >>> print(f"Supported attention mechanisms: {len(attention_types)}")
        >>> for attn_type in attention_types:
        ...     print(f"  - {attn_type}")
        >>>
        >>> # Validate user input
        >>> user_choice = "multi_head"
        >>> if user_choice in list_attention_types():
        ...     layer = create_attention_layer(user_choice, dim=256)
        >>> else:
        ...     print(f"Invalid choice. Available: {list_attention_types()}")
    """
    return sorted(list(ATTENTION_REGISTRY.keys()))


def get_attention_requirements(attention_type: str) -> Dict[str, Any]:
    """
    Get parameter requirements for a specific attention layer type.

    Returns detailed parameter information for a single attention type,
    useful for dynamic UI generation, parameter validation, and documentation.

    Args:
        attention_type (str): The attention layer type to query.

    Returns:
        Dict[str, Any]: Parameter requirements containing:
                       - required_params: List of mandatory parameters
                       - optional_params: Dict of optional parameters with defaults
                       - description: Technical description
                       - use_case: Recommended applications

    Raises:
        ValueError: If attention_type is not supported.

    Example:
        >>> # Get requirements for specific attention type
        >>> reqs = get_attention_requirements('group_query')
        >>> print(f"Required: {reqs['required_params']}")
        >>> print(f"Optional defaults: {reqs['optional_params']}")
        >>>
        >>> # Dynamic parameter validation
        >>> def validate_user_params(attn_type, user_params):
        ...     reqs = get_attention_requirements(attn_type)
        ...     missing = [p for p in reqs['required_params'] if p not in user_params]
        ...     return len(missing) == 0, missing
    """
    if attention_type not in ATTENTION_REGISTRY:
        available_types = list(ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Available types: {available_types}"
        )

    return ATTENTION_REGISTRY[attention_type].copy()
