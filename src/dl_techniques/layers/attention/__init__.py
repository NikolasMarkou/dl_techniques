"""
Attention Layers Module.

This module exposes a comprehensive collection of attention mechanisms, ranging
from standard multi-head attention to specialized variants for vision, efficiency,
and advanced modeling.

It includes a factory interface (`create_attention_layer`) for unified instantiation
and direct access to all layer classes.

Factory coverage of the names exported here
-------------------------------------------
`factory.py` registers **31** keys. Of the 35 layer classes re-exported below, 29 are
reachable through `create_attention_layer(type=...)` and 6 are **direct-import only**:

    ProgressiveFocusedAttention, Ideogram4Attention, MMDiTJointAttention,
    AttentionRoutingCapsule, CapsuleBlockV2, WindowAttention

`WindowAttention` is the subtle one. It has **no factory key of its own**. The keys
`'window'` and `'window_zigzag'` are registered against the module-level wrapper
FUNCTIONS `create_grid_window_attention` / `create_zigzag_window_attention`
(`window_attention.py`), each of which constructs a `WindowAttention` locked to one
`partition_mode` ('grid' / 'zigzag') with that mode's `use_relative_position_bias`
default. The general class — the one whose `partition_mode` you choose yourself — is
NOT factory-constructible; import it directly from this package, as done below.

The remaining two window helpers, `create_kan_key_window_attention` and
`create_adaptive_softmax_window_attention` (`window_attention.py`), are
**intentionally non-public**: referenced by neither this module nor `factory.py`, their
only callers are in `tests/test_layers/test_attention/test_window_attention.py`. They
are convenience constructors for the `attention_mode='kan_key'` and
`probability_type='adaptive'` configurations, both of which are reachable today by
passing those kwargs to `WindowAttention` directly. Do NOT "fix the inconsistency" by
registering them — the registry surface and `__all__` are frozen public API.

The `__all__` grouping comments below are documentation, not semantics: import order is
load-bearing (it fixes Keras registration order) and is deliberately left untouched.
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
# All factory-registered except WindowAttention: keys 'window'/'window_zigzag' point at
# the create_grid_window_attention / create_zigzag_window_attention wrappers, not at the
# class. The general partition_mode-selectable class is direct-import only.
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

# Efficient / linear-complexity attention (all factory-registered:
# 'energy', 'gated', 'linear', 'performer', 'ring', 'rpc')
from .energy_attention import EnergyAttention
from .gated_attention import GatedAttention
from .linear_attention import LinearAttention
from .performer_attention import PerformerAttention
from .ring_attention import RingAttention
from .rpc_attention import RPCAttention

# Additional specialized attention — MIXED factory coverage.
# The blanket "(direct instantiation only)" heading this comment replaced was wrong for
# two of the names below: SingleWindowAttention IS registered (key 'single_window') and
# WaveFieldAttention IS registered (key 'wave_field'). Per-name status:
from .progressive_focused_attention import ProgressiveFocusedAttention  # direct only
from .single_window_attention import SingleWindowAttention              # key 'single_window'
from .wave_field_attention import WaveFieldAttention                    # key 'wave_field'
from .ideogram4_attention import Ideogram4Attention                     # direct only
from .mmdit_joint_attention import MMDiTJointAttention                  # direct only
# AttentionRoutingCapsule + CapsuleBlockV2: direct only (CapsuleBlockV2 is a composite
# block, not a bare attention layer; consumed by models/capsnet/model_v2.py).
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
    "WindowAttention",  # no factory key; 'window'/'window_zigzag' use the wrappers
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

    # Efficient / linear-complexity (all factory-registered)
    "EnergyAttention",
    "GatedAttention",
    "LinearAttention",
    "PerformerAttention",
    "RingAttention",
    "RPCAttention",

    # Additional Specialized (MIXED factory coverage — see the per-name comments
    # beside the imports above; NOT all direct-instantiation-only)
    "ProgressiveFocusedAttention",
    "SingleWindowAttention",  # factory key 'single_window'
    "WaveFieldAttention",     # factory key 'wave_field'
    "Ideogram4Attention",
    "MMDiTJointAttention",
    "AttentionRoutingCapsule",
    "CapsuleBlockV2",
]