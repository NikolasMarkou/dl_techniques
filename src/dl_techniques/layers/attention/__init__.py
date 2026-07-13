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
    get_attention_info,
    list_attention_types,
    get_attention_requirements
)

# Standard and Efficient Attention
from .multi_head_attention import MultiHeadAttention
from .multi_head_cross_attention import MultiHeadCrossAttention
from .group_query_attention import GroupedQueryAttention
from .differential_attention import DifferentialMultiHeadAttention
from .multi_head_latent_attention import MultiHeadLatentAttention
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
from .lighthouse_attention import LighthouseAttention
from .perceiver_attention import PerceiverAttention

# Factory-registered layers (also available for direct instantiation)
from .energy_attention import EnergyAttention
from .gated_attention import GatedAttention
from .linear_attention import LinearAttention
from .performer_attention import PerformerAttention
from .ring_attention import RingAttention
from .rpc_attention import RPCAttention

# Additional specialized attention (direct instantiation only)
from .progressive_focused_attention import ProgressiveFocusedAttention
from .single_window_attention import SingleWindowAttention
from .wave_field_attention import WaveFieldAttention
from .ideogram4_attention import Ideogram4Attention
from .mmdit_joint_attention import MMDiTJointAttention
from .attention_routing_capsule import AttentionRoutingCapsule, CapsuleBlockV2

__all__ = [
    # Factory Interface
    "create_attention_from_config",
    "create_attention_layer",
    "validate_attention_config",
    "AttentionType",
    "get_attention_info",
    "list_attention_types",
    "get_attention_requirements",

    # Standard & Efficient
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "GroupedQueryAttention",
    "DifferentialMultiHeadAttention",
    "MultiHeadLatentAttention",
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
    "LighthouseAttention",
    "PerceiverAttention",

    # Direct Instantiation Only
    "EnergyAttention",
    "GatedAttention",
    "LinearAttention",
    "PerformerAttention",
    "RingAttention",
    "RPCAttention",

    # Additional Specialized (Direct Instantiation Only)
    "ProgressiveFocusedAttention",
    "SingleWindowAttention",
    "WaveFieldAttention",
    "Ideogram4Attention",
    "MMDiTJointAttention",
    "AttentionRoutingCapsule",
    "CapsuleBlockV2",
]