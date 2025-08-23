"""Advanced Spatial Layers for Geometric Deep Learning and Multi-Modal AI.

This module contains a collection of state-of-the-art neural network layers specifically
designed for processing spatial data, point clouds, and multi-modal inputs. These layers
were extracted and adapted from cutting-edge research in computational fluid dynamics (CFD),
geometric deep learning, and spatial transformers, then converted to Keras 3.x with full
backend compatibility.

Key Innovations:
    • Continuous coordinate embeddings for arbitrary dimensional spaces (2D, 3D, N-D)
    • Rotary Position Embedding (RoPE) extended to continuous spatial coordinates
    • Hierarchical attention mechanisms with anchor-query patterns for efficiency
    • Cross-modal attention for multi-sensor and multi-modal data fusion
    • Graph-based pooling operations implemented in dense tensor format
    • Physics-aware neural network components for scientific computing

Core Capabilities:
    • Point Cloud Processing: Classification, segmentation, object detection
    • Multi-Modal Fusion: Vision-language, sensor fusion, cross-modal retrieval
    • Scientific Computing: Physics simulations, climate modeling, molecular analysis
    • Robotics: SLAM, manipulation planning, navigation, sensor processing
    • Geospatial AI: Environmental monitoring, urban planning, precision agriculture
    • Creative AI: Procedural generation, digital twins, interactive simulations

Architecture Overview:
    The layers follow a modular design philosophy where each component serves a specific
    spatial reasoning purpose:

    1. Embedding Layers: Convert coordinates to learned representations
    2. Attention Mechanisms: Enable spatial reasoning and cross-modal interaction
    3. Pooling Operations: Aggregate information across spatial neighborhoods
    4. Utility Functions: Support common spatial AI operations and transformations

Key Layers:

    ContinuousSincosEmbed:
        Embeds continuous coordinates (2D, 3D, N-D) using sinusoidal functions.
        Unlike standard positional encodings for discrete sequences, this handles
        arbitrary coordinate systems and scales.

    ContinuousRoPE:
        Rotary Position Embedding for continuous coordinates. Generates complex
        frequencies that enable position-aware attention for spatial data.

    AnchorAttention:
        Hierarchical attention where anchor tokens have full self-attention while
        query tokens only attend to anchors. Reduces O(n²) complexity to O(n·k).

    PerceiverAttention & PerceiverBlock:
        Cross-attention mechanisms for different input modalities. Enables fusion
        of heterogeneous data types (vision, text, sensors).

    SupernodePooling:
        Graph-based message passing pooling for point clouds. Aggregates local
        neighborhoods using spatial relationships and learned embeddings.

Applications and Use Cases:

    Computer Vision & 3D Processing:
        >>> # Point cloud classification
        >>> points = keras.random.uniform((1000, 3)) * 10  # 1000 3D points
        >>> embed = ContinuousSinCosEmbed(dim=256, ndim=3)
        >>> features = embed(points)
        >>> attn = AnchorAttention(dim=256, num_heads=8)
        >>> output = attn(features, num_anchor_tokens=100)

    Multi-Modal AI:
        >>> # Vision-language cross-attention
        >>> visual_features = keras.random.normal((50, 256))    # Image patches
        >>> text_features = keras.random.normal((20, 256))     # Text tokens
        >>> combined = ops.concatenate([visual_features, text_features], axis=0)
        >>> cross_attn = SharedWeightsCrossAttention(dim=256, num_heads=8)
        >>> output = cross_attn(combined, split_sizes=[50, 20])

    Scientific Computing:
        >>> # Physics simulation with spatial coordinates
        >>> coords = keras.random.uniform((1000, 3)) * 100  # Simulation grid
        >>> initial_state = keras.random.normal((1000, 7))  # Physical quantities
        >>> pos_embed = ContinuousSinCosEmbed(dim=512, ndim=3)
        >>> spatial_features = pos_embed(coords)
        >>> physics_attn = AnchorAttention(dim=512, num_heads=16)
        >>> evolved_state = physics_attn(spatial_features, num_anchor_tokens=200)

    Robotics & Sensor Fusion:
        >>> # Multi-sensor data processing
        >>> lidar_points = keras.random.uniform((500, 3)) * 20
        >>> pooling = SupernodePooling(hidden_dim=256, ndim=3, radius=2.0)
        >>> supernode_idx = keras.random.uniform((50,), 0, 500, dtype="int32")
        >>> features = pooling({
        ...     "positions": lidar_points,
        ...     "supernode_indices": supernode_idx
        ... })

Technical Implementation:

    Backend Compatibility:
        All layers are implemented using keras.ops for compatibility across
        TensorFlow, JAX, and PyTorch backends. No backend-specific operations
        are used, ensuring portability and future-proofing.

    Serialization Support:
        Full support for model saving/loading with proper get_config(),
        get_build_config(), and build_from_config() implementations.

    Memory Efficiency:
        Hierarchical attention patterns and anchor-query mechanisms reduce
        memory usage for large spatial datasets while maintaining accuracy.

    Numerical Stability:
        Careful handling of coordinate scales, attention normalization, and
        gradient flow to ensure stable training on diverse spatial data.

Performance Considerations:

    Coordinate Normalization:
        Normalize coordinates to appropriate ranges (typically [0, 1000] or [-1, 1])
        for optimal embedding performance and numerical stability.

    Attention Scaling:
        For large point clouds (>10k points), use hierarchical processing:
        1. SupernodePooling for initial dimensionality reduction
        2. AnchorAttention with appropriate anchor counts
        3. Multiple attention layers with decreasing anchor tokens

    Memory Usage:
        • SupernodePooling: O(n·k) where k is max neighbors per supernode
        • AnchorAttention: O(n·a) where a is number of anchor tokens
        • Standard Attention: O(n²) - use sparingly for large sequences

Research Background:

    This module builds upon several key research areas:

    1. Geometric Deep Learning:
        - Graph Neural Networks for irregular spatial data
        - Point cloud processing with permutation invariance
        - Continuous convolutions and spatial embeddings

    2. Spatial Transformers:
        - Attention mechanisms for spatial reasoning
        - Position-aware neural networks
        - Multi-scale feature processing

    3. Multi-Modal Learning:
        - Cross-attention for heterogeneous data fusion
        - Shared representations across modalities
        - Efficient parameter sharing strategies

    4. Scientific Machine Learning:
        - Physics-informed neural networks
        - Spatiotemporal modeling and prediction
        - Domain-specific inductive biases

Related Work and References:

    • "Attention Is All You Need" - Transformer architecture foundations
    • "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    • "Perceiver: General Perception with Iterative Attention"
    • "Point Transformer" - Attention mechanisms for point clouds
    • "Neural Message Passing for Quantum Chemistry" - Graph neural networks
    • "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"
"""

from .continuous_sin_cos_embed import ContinuousSinCosEmbed
from .continuous_rope import ContinuousRoPE, apply_rope
from .anchor_attention import AnchorAttention
from .perceiver_attention import PerceiverAttention
from .perceiver_block import PerceiverBlock
from .supernode_pooling import SupernodePooling
from .rope_utilities import (
    apply_rope_to_attention,
    apply_rope_cross_attention,
    create_rope_frequencies,
    scaled_dot_product_attention_with_rope,
    rope_attention_block
)
from .transformer_block import TransformerBlock

__all__ = [
    # Core spatial embedding layers
    "ContinuousSinCosEmbed",
    "ContinuousRoPE",

    # Attention mechanisms
    "AnchorAttention",
    "PerceiverAttention",
    "PerceiverBlock",

    # Pooling and aggregation
    "SupernodePooling",

    # Utility functions
    "apply_rope",
    "apply_rope_to_attention",
    "apply_rope_cross_attention",
    "create_rope_frequencies",
    "scaled_dot_product_attention_with_rope",
    "rope_attention_block",
]

