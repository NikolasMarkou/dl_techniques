"""BiT/BiB bidirectional text<->image diffusion bridge -- public API re-exports.

A Keras 3 port of the DiTXA cross-attention diffusion transformer and the bridge
machinery around it: a lossless channels-last token<->bridge packing, the SDE
base processes and their direction-specific score-matching targets, the sampler,
and a decoder that reads text back out of a sampled bridge tensor.

Currently exported: the bridge geometry, the packing bijection, and the four SDE
base processes. The model and the decoder land in later steps of the port.

    from dl_techniques.models.vision_language.bit_diffusion import (
        BRIDGE_PRESETS, PeriodicVolatilitySDE, token_flat_to_bridge,
    )
"""

from .config import (
    BRIDGE_PRESETS,
    PROMPT_KIND_TO_LABEL,
    PROMPT_NUM_CLASSES,
    TIME_EPS,
    TOKEN_LAYOUTS,
    BridgeConfig,
    get_bridge_config,
)
from .sde import (
    SDE_VARIANTS,
    BridgeSDE,
    CosineDecayingVolatilitySDE,
    FlowMatchingODE,
    PeriodicVolatilitySDE,
    UniformVolatilitySDE,
    bridge_math_dtype,
    create_bridge_sde,
)
from .token_bridge import (
    bridge_to_token_flat,
    compute_token_norms,
    norm_based_token_stops,
    pad_id_token_stops,
    prepare_bridge_batch,
    token_flat_to_bridge,
)

__all__ = [
    "BRIDGE_PRESETS",
    "BridgeConfig",
    "BridgeSDE",
    "CosineDecayingVolatilitySDE",
    "FlowMatchingODE",
    "PeriodicVolatilitySDE",
    "PROMPT_KIND_TO_LABEL",
    "PROMPT_NUM_CLASSES",
    "TIME_EPS",
    "SDE_VARIANTS",
    "TOKEN_LAYOUTS",
    "UniformVolatilitySDE",
    "bridge_math_dtype",
    "bridge_to_token_flat",
    "compute_token_norms",
    "create_bridge_sde",
    "get_bridge_config",
    "norm_based_token_stops",
    "pad_id_token_stops",
    "prepare_bridge_batch",
    "token_flat_to_bridge",
]
