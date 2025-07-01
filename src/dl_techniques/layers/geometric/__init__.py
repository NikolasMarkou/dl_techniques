"""Spatial and attention layers for geometric deep learning.

This module contains layers implementations for spatial data processing, including:

- Continuous positional embeddings for arbitrary coordinate systems
- Rotary Position Embedding (RoPE) for spatial transformers
- Hierarchical attention mechanisms for efficient processing
- Cross-attention for multi-modal learning
- Graph-based pooling for point cloud processing

These layers are particularly useful for:
- Point cloud processing
- Geometric deep learning
- Multi-modal learning
- Spatial transformers
- Computer vision with spatial reasoning
"""

from .continuous_sin_cos_embed import ContinuousSincosEmbed
from .continuous_rope import ContinuousRoPE, apply_rope
from .anchor_attention import AnchorAttention
from .perceiver_attention import PerceiverAttention
from .perceiver_block import PerceiverBlock
from .shared_weights_cross_attention import SharedWeightsCrossAttention
from .supernode_pooling import SupernodePooling
from .rope_utilities import (
    apply_rope_to_attention,
    apply_rope_cross_attention,
    create_rope_frequencies,
    scaled_dot_product_attention_with_rope,
    rope_attention_block
)

__all__ = [
    # Core spatial embedding layers
    "ContinuousSincosEmbed",
    "ContinuousRoPE",

    # Attention mechanisms
    "AnchorAttention",
    "PerceiverAttention",
    "PerceiverBlock",
    "SharedWeightsCrossAttention",

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

# Layer categories for easy discovery
EMBEDDING_LAYERS = [
    "ContinuousSincosEmbed",
    "ContinuousRoPE",
]

ATTENTION_LAYERS = [
    "AnchorAttention",
    "PerceiverAttention",
    "PerceiverBlock",
    "SharedWeightsCrossAttention",
]

POOLING_LAYERS = [
    "SupernodePooling",
]

UTILITY_FUNCTIONS = [
    "apply_rope",
    "apply_rope_to_attention",
    "apply_rope_cross_attention",
    "create_rope_frequencies",
    "scaled_dot_product_attention_with_rope",
    "rope_attention_block",
]


def get_layer_info():
    """Get information about available spatial layers.

    Returns:
        Dictionary with layer categories and descriptions.
    """
    return {
        "embedding_layers": {
            "ContinuousSincosEmbed": "Continuous coordinate embedding using sinusoidal functions",
            "ContinuousRoPE": "Rotary Position Embedding for continuous coordinates"
        },
        "attention_layers": {
            "AnchorAttention": "Hierarchical attention with anchor-query structure",
            "PerceiverAttention": "Cross-attention mechanism from Perceiver architecture",
            "PerceiverBlock": "Complete Perceiver transformer block",
            "SharedWeightsCrossAttention": "Efficient cross-attention between modalities"
        },
        "pooling_layers": {
            "SupernodePooling": "Graph-based pooling with message passing for point clouds"
        },
        "utilities": {
            "apply_rope": "Apply RoPE rotation to tensors",
            "apply_rope_to_attention": "Apply RoPE to query and key tensors",
            "create_rope_frequencies": "Create RoPE frequencies from positions",
            "scaled_dot_product_attention_with_rope": "Complete attention with RoPE support"
        }
    }


def print_layer_info():
    """Print information about available spatial layers."""
    info = get_layer_info()

    print("Spatial Layers for Geometric Deep Learning")
    print("=" * 45)

    for category, layers in info.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print("-" * (len(category) + 1))

        for layer_name, description in layers.items():
            print(f"  {layer_name}: {description}")

    print(f"\nTotal layers: {len(__all__)}")
    print("\nExample usage:")
    print("  from dl_techniques.layers.spatial import ContinuousSincosEmbed, AnchorAttention")
    print("  embed = ContinuousSincosEmbed(dim=256, ndim=3)")
    print("  attn = AnchorAttention(dim=256, num_heads=8)")


if __name__ == "__main__":
    print_layer_info()