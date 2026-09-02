"""BiT/BiB bidirectional text<->image diffusion bridge -- public API re-exports.

A Keras 3 port of the DiTXA cross-attention diffusion transformer and the bridge
machinery around it: a lossless channels-last token<->bridge packing, the SDE
base processes and their direction-specific score-matching targets, the sampler,
and a decoder that reads text back out of a sampled bridge tensor.

Currently exported: the bridge geometry and the packing bijection. The model,
the SDE family and the decoder land in later steps of the port.

    from dl_techniques.models.vision_language.bit_diffusion import (
        BRIDGE_PRESETS, token_flat_to_bridge,
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
    "PROMPT_KIND_TO_LABEL",
    "PROMPT_NUM_CLASSES",
    "TIME_EPS",
    "TOKEN_LAYOUTS",
    "bridge_to_token_flat",
    "compute_token_norms",
    "get_bridge_config",
    "norm_based_token_stops",
    "pad_id_token_stops",
    "prepare_bridge_batch",
    "token_flat_to_bridge",
]
