"""
Attention Layers Module.

This module exposes a comprehensive collection of attention mechanisms, ranging
from standard multi-head attention to specialized variants for vision, efficiency,
and advanced modeling.

It includes a factory interface (`create_attention_layer`) for unified instantiation
and direct access to all layer classes.
"""

# Factory and Utility Functions
from .factory import (
    create_attention_from_config,
    create_attention_layer,
    validate_attention_config,
    AttentionType,
    get_attention_info
)

# Standard and Efficient Attention
from .multi_head_attention import MultiHeadAttention
from .multi_head_cross_attention import MultiHeadCrossAttention
from .group_query_attention import GroupedQueryAttention
from .differential_attention import DifferentialMultiHeadAttention
from .shared_weights_cross_attention import SharedWeightsCrossAttention

# Vision and Spatial Attention
from .convolutional_block_attention import CBAM
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention
from .non_local_attention import NonLocalAttention
from .mobile_mqa import MobileMQA
from .window_attention import WindowAttention
from .tripse_attention import TripSE1, TripSE2, TripSE3, TripSE4

# Advanced / Specialized Attention
from .anchor_attention import AnchorAttention
from .capsule_routing_attention import CapsuleRoutingSelfAttention
from .fnet_fourier_transform import FNetFourierTransform
from .hopfield_attention import HopfieldAttention
from .perceiver_attention import PerceiverAttention

# Layers available for direct instantiation (not yet in factory registry)
from .gated_attention import GatedAttention
from .performer_attention import PerformerAttention
from .ring_attention import RingAttention
from .rpc_attention import RPCAttention

__all__ = [
    # Factory Interface
    "create_attention_from_config",
    "create_attention_layer",
    "validate_attention_config",
    "AttentionType",
    "get_attention_info",

    # Standard & Efficient
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "GroupedQueryAttention",
    "DifferentialMultiHeadAttention",
    "SharedWeightsCrossAttention",

    # Vision & Spatial
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "NonLocalAttention",
    "MobileMQA",
    "WindowAttention",
    "TripSE1",
    "TripSE2",
    "TripSE3",
    "TripSE4",

    # Advanced / Specialized
    "AnchorAttention",
    "CapsuleRoutingSelfAttention",
    "FNetFourierTransform",
    "HopfieldAttention",
    "PerceiverAttention",

    # Direct Instantiation Only
    "GatedAttention",
    "PerformerAttention",
    "RingAttention",
    "RPCAttention",
]