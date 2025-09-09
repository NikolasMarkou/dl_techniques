"""
Attention Layer Factory for the dl_techniques Framework

Provides a centralized factory function for creating various attention layers
with a unified interface, type safety, and comprehensive parameter validation.
This factory allows for easy swapping of attention mechanisms within models
by simply changing the `attention_type` string.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
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
from .non_local_attention import NonLocalAttention
from .perceiver_attention import PerceiverAttention
from .shared_weights_cross_attention import SharedWeightsCrossAttention
from .spatial_attention import SpatialAttention
from .window_attention import WindowAttention

# ---------------------------------------------------------------------
# Type definition for Attention types
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
    'non_local',
    'perceiver',
    'shared_weights_cross',
    'spatial',
    'window'
]

# ---------------------------------------------------------------------
# Attention layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

ATTENTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    'anchor': {
        'class': AnchorAttention,
        'description': 'Implements a memory-efficient hierarchical attention mechanism that reduces computational complexity by designating a subset of tokens as "anchors". These anchors perform full self-attention among themselves, while the remaining "query" tokens only cross-attend to the anchors, creating a sparse and scalable attention pattern.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Ideal for models processing very long sequences (e.g., high-resolution images, long documents, or audio) where the quadratic complexity of standard self-attention is computationally prohibitive. It offers a trade-off between expressive power and efficiency.'
    },
    'capsule_routing': {
        'class': CapsuleRoutingSelfAttention,
        'description': 'Extends standard multi-head attention by incorporating the dynamic routing mechanism from Capsule Networks. It organizes attention weights into vertical (head-wise) and horizontal (token-wise) capsules, iteratively refining them to achieve a more robust and contextually aware attention distribution.',
        'required_params': ['num_heads'],
        'optional_params': {
            'key_dim': None, 'value_dim': None, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None,
            'routing_iterations': 3, 'use_vertical_routing': True, 'use_horizontal_routing': True,
            'use_positional_routing': True, 'epsilon': 1e-8
        },
        'use_case': 'Primarily for experimental architectures and research aimed at improving contextual representation by modeling part-whole relationships. Suitable for tasks requiring robust feature grouping and disambiguation, such as complex scene understanding or nuanced language interpretation.'
    },
    'cbam': {
        'class': CBAM,
        'description': 'Implements the Convolutional Block Attention Module, a lightweight and effective attention mechanism for CNNs. It sequentially applies channel attention (to focus on "what" is important) and spatial attention (to focus on "where" it is important), refining feature maps for improved representation.',
        'required_params': ['channels'],
        'optional_params': {
            'ratio': 8, 'kernel_size': 7, 'channel_kernel_initializer': 'glorot_uniform',
            'spatial_kernel_initializer': 'glorot_uniform', 'channel_kernel_regularizer': None,
            'spatial_kernel_regularizer': None, 'channel_use_bias': False, 'spatial_use_bias': True
        },
        'use_case': 'A versatile, plug-and-play module for any Convolutional Neural Network (CNN). It can be inserted into existing architectures (like ResNet, MobileNet) to enhance feature representation with minimal computational overhead, often leading to performance gains in image classification, object detection, and segmentation.'
    },
    'channel': {
        'class': ChannelAttention,
        'description': 'Implements the channel attention submodule from CBAM. It generates channel-wise attention weights by aggregating spatial information using both average and max pooling, then processing it through a shared Multi-Layer Perceptron (MLP). This allows the network to selectively emphasize more informative feature channels.',
        'required_params': ['channels'],
        'optional_params': {
            'ratio': 8, 'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None, 'use_bias': False
        },
        'use_case': 'Used within CNN architectures to explicitly model inter-channel relationships and recalibrate feature responses. It is a lightweight method to boost performance by allowing the model to focus on the most relevant channels for a given task, without altering spatial features.'
    },
    'differential': {
        'class': DifferentialMultiHeadAttention,
        'description': 'Implements Differential Attention, a mechanism that uses two parallel multi-head attention layers to distinguish between primary context (signal) and irrelevant information (noise). The final output is a weighted difference between the two, effectively amplifying relevant signals while actively canceling out noise.',
        'required_params': ['dim', 'num_heads', 'head_dim'],
        'optional_params': {
            'dropout_rate': 0.0, 'attention_dropout_rate': 0.0, 'lambda_init': 0.8,
            'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None,
            'bias_initializer': 'zeros', 'bias_regularizer': None, 'activity_regularizer': None
        },
        'use_case': 'Designed for transformer models where enhanced focus and noise reduction are critical. Particularly effective for generative tasks to reduce hallucination and improve factual accuracy, and for any sequence modeling task that benefits from a cleaner, more focused context representation.'
    },
    'fnet': {
        'class': FNetFourierTransform,
        'description': 'A parameter-free token mixing layer that replaces the self-attention mechanism with a 2D Fourier Transform. It applies DFTs along both the sequence and hidden dimensions to mix information efficiently, offering a non-learnable alternative to attention that has O(N log N) complexity.',
        'required_params': [],
        'optional_params': {
            'implementation': 'matrix', 'normalize_dft': True, 'epsilon': 1e-12
        },
        'use_case': 'An extremely efficient drop-in replacement for self-attention in transformer-like architectures. Ideal for models where speed and parameter efficiency are paramount, such as on-device deployment or pre-training on massive datasets where O(N²) complexity is a bottleneck.'
    },
    'group_query': {
        'class': GroupedQueryAttention,
        'description': 'Implements Grouped Query Attention (GQA), an efficient compromise between multi-head and multi-query attention. It reduces the number of Key and Value heads, which are then shared across groups of Query heads, significantly reducing the size of the KV cache during inference.',
        'required_params': ['dim', 'num_heads', 'num_kv_heads'],
        'optional_params': {
            'max_seq_len': 2048, 'dropout_rate': 0.0, 'rope_percentage': 1.0,
            'rope_theta': 10000.0, 'use_bias': False, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Essential for large language models (LLMs) where the Key-Value cache is a major memory bottleneck during autoregressive decoding. GQA allows for much longer context windows and faster inference with minimal degradation in model quality compared to standard MHA.'
    },
    'hopfield': {
        'class': HopfieldAttention,
        'description': 'A Modern Hopfield Network that functions as a content-addressable memory, using a transformer-style attention mechanism as its update rule. It can iteratively refine a query state to retrieve the most similar stored pattern (Key-Value pair), converging to a stable fixed point.',
        'required_params': ['num_heads', 'key_dim'],
        'optional_params': {
            'value_dim': None, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None,
            'normalize_patterns': True, 'update_steps_max': 0, 'update_steps_eps': 1e-4
        },
        'use_case': 'Ideal for associative memory, pattern completion, and noise correction tasks. Setting `update_steps_max=0` makes it behave like a standard attention layer, while `update_steps_max > 0` enables its powerful pattern retrieval and cleaning capabilities.'
    },
    'mobile_mqa': {
        'class': MobileMQA,
        'description': 'An implementation of Multi-Query Attention (MQA) highly optimized for 4D vision tensors on mobile and edge devices. It uses a single, shared Key and Value projection for all query heads to dramatically reduce memory bandwidth, with an optional spatial downsampling step for further efficiency.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'use_downsampling': False, 'kernel_initializer': 'he_normal',
            'kernel_regularizer': None
        },
        'use_case': 'Designed for efficient attention in mobile-first vision models (e.g., MobileViT). Its reduced memory I/O makes it well-suited for deployment on hardware with limited memory bandwidth, such as mobile GPUs and accelerators.'
    },
    'multi_head': {
        'class': MultiHeadAttention,
        'description': 'A streamlined implementation of the standard Multi-Head Self-Attention mechanism. As a wrapper around a more general cross-attention layer, it provides a clean, focused interface for the common use case where a sequence attends to itself.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout_rate': 0.0, 'kernel_initializer': 'he_normal',
            'kernel_regularizer': None, 'use_bias': False
        },
        'use_case': 'The foundational building block for most Transformer architectures (e.g., ViT, BERT). It is the default choice for capturing rich, contextual relationships within a single sequence in both vision and NLP tasks.'
    },
    'non_local': {
        'class': NonLocalAttention,
        'description': 'Implements the Non-local Neural Network block, a self-attention mechanism tailored for 4D computer vision tensors. It captures long-range spatial dependencies by computing the response at each position as a weighted sum of features at all other positions, overcoming the limited receptive field of convolutions.',
        'required_params': ['attention_channels'],
        'optional_params': {
            'kernel_size': (7, 7), 'use_bias': False, 'normalization': 'batch',
            'intermediate_activation': 'relu', 'output_activation': 'linear', 'output_channels': -1,
            'dropout_rate': 0.0, 'attention_mode': 'gaussian', 'kernel_initializer': 'glorot_normal',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': 'Used to augment CNNs with global context reasoning. It can be inserted into deep vision models to help with tasks that require understanding relationships between distant parts of an image, such as video analysis, instance segmentation, and pose estimation.'
    },
    'perceiver': {
        'class': PerceiverAttention,
        'description': 'Implements the cross-attention mechanism from the Perceiver architecture. It is designed to process inputs from different modalities by attending from a smaller, fixed-size latent query array to a larger, potentially multi-modal, byte array. This decouples network depth from input size.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Core component for Perceiver-style models that handle large and multi-modal inputs. Perfect for tasks like image classification from raw pixels, language modeling from text, or fusing audio-visual data by attending from a latent space to the raw inputs.'
    },
    'shared_weights_cross': {
        'class': SharedWeightsCrossAttention,
        'description': 'An efficient cross-attention layer where two distinct modalities attend to each other using a single, shared set of projection weights. This design significantly reduces parameters while enabling bidirectional information flow between the two input sequences.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Excellent for parameter-efficient multi-modal fusion tasks where two data streams need to interact, such as fusing image and text features, or combining different sensor readings in a robotics application. The shared weights make it suitable for models with tight parameter budgets.'
    },
    'spatial': {
        'class': SpatialAttention,
        'description': 'Implements the spatial attention submodule from CBAM. It generates a 2D spatial attention map by applying average and max pooling along the channel axis and passing the result through a convolution layer. This highlights the most spatially significant regions in a feature map.',
        'required_params': [],
        'optional_params': {
            'kernel_size': 7, 'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None, 'use_bias': True
        },
        'use_case': 'Used within CNNs to help the network focus on the most informative spatial locations. It is effective at improving object detection and segmentation performance by suppressing irrelevant background features and emphasizing salient object regions.'
    },
    'window': {
        'class': WindowAttention,
        'description': 'Implements the windowed multi-head self-attention from the Swin Transformer. It mitigates the quadratic complexity of global self-attention by partitioning the input into non-overlapping windows and computing attention locally within each window, using a learnable relative position bias for spatial awareness.',
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'qkv_bias': True, 'qk_scale': None, 'attn_dropout_rate': 0.0,
            'proj_dropout_rate': 0.0, 'proj_bias': True, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'The core computational block for Swin-style vision transformers. Its linear complexity with respect to image size makes it highly scalable for high-resolution vision tasks like image classification, object detection, and semantic segmentation.'
    }
}


# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_attention_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available attention layer types.

    Returns:
        Dict containing details for each attention type, including description,
        required/optional parameters, and primary use case.

    Example:
        ```python
        info = get_attention_info()
        for attn_type, details in info.items():
            print(f"{attn_type}: {details['description']}")
            print(f"  Required: {details['required_params']}")
        ```
    """
    return {attn_type: info.copy() for attn_type, info in ATTENTION_REGISTRY.items()}


def validate_attention_config(attention_type: str, **kwargs: Any) -> None:
    """
    Validate attention layer configuration parameters.

    Args:
        attention_type: The type of attention layer to validate.
        **kwargs: Parameters to validate against the layer's requirements.

    Raises:
        ValueError: If `attention_type` is invalid, required parameters are
                    missing, or common parameters have invalid values.
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
            f"Required: {required}"
        )

    # Common parameter validation
    dim_params = ['dim', 'channels', 'attention_channels']
    for p in dim_params:
        if p in kwargs and kwargs[p] <= 0:
            raise ValueError(f"Parameter '{p}' must be positive, got {kwargs[p]}")

    head_params = ['num_heads', 'num_kv_heads']
    for p in head_params:
        if p in kwargs and kwargs[p] <= 0:
            raise ValueError(f"Parameter '{p}' must be positive, got {kwargs[p]}")

    dropout_params = [
        'dropout_rate', 'attention_dropout_rate',
        'attn_dropout_rate', 'proj_dropout_rate'
    ]
    for p in dropout_params:
        if p in kwargs and not (0.0 <= kwargs[p] <= 1.0):
            raise ValueError(f"Parameter '{p}' must be between 0.0 and 1.0, got {kwargs[p]}")

# ---------------------------------------------------------------------

def create_attention_layer(
        attention_type: AttentionType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating attention layers with a unified interface.

    This function provides a centralized way to instantiate any attention layer
    supported by the framework, with comprehensive parameter validation.

    Args:
        attention_type: The type of attention layer to create.
        name: Optional name for the created layer.
        **kwargs: Parameters specific to the chosen attention type. Use
                  `get_attention_info()` to see requirements for each type.

    Returns:
        An instantiated and configured Keras attention layer.

    Raises:
        ValueError: If `attention_type` is invalid, required parameters are
                    missing, or parameter values are incorrect.

    Example:
        ```python
        # Create a standard multi-head attention layer
        mha = create_attention_layer('multi_head', dim=256, num_heads=8)

        # Create a CBAM block for a CNN
        cbam = create_attention_layer('cbam', channels=128, ratio=16)

        # Create a Grouped Query Attention layer for an LLM
        gqa = create_attention_layer(
            'group_query',
            dim=1024,
            num_heads=16,
            num_kv_heads=4,
            name='gqa_block_1'
        )
        ```
    """
    try:
        validate_attention_config(attention_type, **kwargs)
        info = ATTENTION_REGISTRY[attention_type]
        attn_class = info['class']

        # Prepare parameters by merging defaults with user-provided kwargs
        params = info['optional_params'].copy()
        params.update(kwargs)

        # Filter out any kwargs not expected by the class constructor
        valid_param_names = set(info['required_params']) | set(info['optional_params'].keys())
        final_params = {k: v for k, v in params.items() if k in valid_param_names}

        if name:
            final_params['name'] = name

        logger.info(f"Creating '{attention_type}' layer with parameters: {final_params}")
        return attn_class(**final_params)

    except (TypeError, ValueError) as e:
        info = ATTENTION_REGISTRY.get(attention_type)
        if info:
            class_name = info['class'].__name__
            error_msg = (
                f"Failed to create '{attention_type}' layer ({class_name}). "
                f"Required: {info['required_params']}. "
                f"Provided: {list(kwargs.keys())}. "
                f"Please check parameter compatibility. Original error: {e}"
            )
        else:
            error_msg = f"Failed to create layer. Unknown type '{attention_type}'. Error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e

# ---------------------------------------------------------------------

def create_attention_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create an attention layer from a configuration dictionary.

    Args:
        config: A dictionary containing a 'type' key specifying the attention
                layer and other keys for its parameters.

    Returns:
        An instantiated and configured Keras attention layer.

    Raises:
        ValueError: If the config is not a dict or is missing the 'type' key.

    Example:
        ```python
        attn_config = {
            'type': 'window',
            'dim': 96,
            'window_size': 7,
            'num_heads': 4,
            'name': 'swin_attn_block_2'
        }
        window_attn = create_attention_from_config(attn_config)
        ```
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    if 'type' not in config:
        raise ValueError("Configuration dictionary must include a 'type' key.")

    config_copy = config.copy()
    attention_type = config_copy.pop('type')
    return create_attention_layer(attention_type, **config_copy)

# ---------------------------------------------------------------------
