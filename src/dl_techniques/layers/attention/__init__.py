"""Attention layers.

This package exports every attention layer in the library plus the factory
interface that builds them from a config dict. Import a class directly, or
call `create_attention_layer(type=...)` when the type comes from
configuration.

`factory.py` registers 34 keys; this package re-exports 37 layer classes,
31 of them reachable through `create_attention_layer(type=...)`. The other
6 are direct-import only: `ProgressiveFocusedAttention`,
`Ideogram4Attention`, `MMDiTJointAttention`, `AttentionRoutingCapsule`,
`CapsuleBlockV2`, `WindowAttention`.

`WindowAttention` has no factory key of its own. The keys `'window'`,
`'window_zigzag'` and `'window_band'` are registered against three
module-level wrapper functions in `window_attention.py`
(`create_grid_window_attention`, `create_zigzag_window_attention`,
`create_band_window_attention`), each building a `WindowAttention` locked
to one `partition_mode` ('grid', 'zigzag' or 'band') with that mode's
`use_relative_position_bias` default. For `'band'`, the 1-D mode,
`window_size` is a half-width in tokens: query `i` attends key `j` when
`abs(i - j) <= window_size`, with no grid folding or square padding. To
choose `partition_mode` yourself, import `WindowAttention` directly; the
general class is not factory-constructible.

Two more window helpers in `window_attention.py`,
`create_kan_key_window_attention` and
`create_adaptive_softmax_window_attention`, are not exported here or
registered in `factory.py`; their only callers are in
`tests/test_layers/test_attention/test_window_attention.py`. They are
convenience constructors for `attention_mode='kan_key'` and
`probability_type='adaptive'`, both reachable today by passing those
kwargs to `WindowAttention` directly.

The `__all__` grouping comments below are documentation, not semantics.
Import order fixes Keras registration order, so it is left as written.
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
# Every class below is factory-registered except WindowAttention. The keys
# 'window', 'window_zigzag' and 'window_band' point at the
# create_grid_window_attention / create_zigzag_window_attention /
# create_band_window_attention wrappers, not at the class. The general
# partition_mode-selectable class is direct-import only.
from .convolutional_block_attention import CBAM
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention
from .non_local_attention import NonLocalAttention
from .mobile_mqa import MobileMQA
# key 'area'
from .area_attention import AreaAttention
# key 'beit'
from .beit_attention import BeitAttention
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

# Additional specialized attention, with mixed factory coverage.
# SingleWindowAttention is registered under key 'single_window' and
# WaveFieldAttention under 'wave_field'; the rest are direct-import only.
# direct only
from .progressive_focused_attention import ProgressiveFocusedAttention
# key 'single_window'
from .single_window_attention import SingleWindowAttention
# key 'wave_field'
from .wave_field_attention import WaveFieldAttention
# direct only
from .ideogram4_attention import Ideogram4Attention
# direct only
from .mmdit_joint_attention import MMDiTJointAttention
# AttentionRoutingCapsule + CapsuleBlockV2: direct only. CapsuleBlockV2 is a
# composite block, not a bare attention layer; it is consumed by
# models/vision/capsnet/model_v2.py.
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
    # factory key 'area'
    "AreaAttention",
    # factory key 'beit'
    "BeitAttention",
    # no factory key; the 'window'/'window_zigzag'/'window_band' wrappers
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

    # Efficient / linear-complexity (all factory-registered)
    "EnergyAttention",
    "GatedAttention",
    "LinearAttention",
    "PerformerAttention",
    "RingAttention",
    "RPCAttention",

    # Additional specialized, mixed factory coverage — see the per-name
    # comments above each import. Comments here sit on their own line.
    "ProgressiveFocusedAttention",
    # factory key 'single_window'
    "SingleWindowAttention",
    # factory key 'wave_field'
    "WaveFieldAttention",
    "Ideogram4Attention",
    "MMDiTJointAttention",
    "AttentionRoutingCapsule",
    "CapsuleBlockV2",
]
