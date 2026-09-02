"""BiT/BiB bidirectional text<->image diffusion bridge -- public API re-exports.

A Keras 3 port of the DiTXA cross-attention diffusion transformer and the bridge
machinery around it: a lossless channels-last token<->bridge packing, the SDE
base processes and their direction-specific score-matching targets, the sampler,
and a decoder that reads text back out of a sampled bridge tensor.

Currently exported: the bridge geometry, the packing bijection, the four SDE
base processes, the `DiTXA` model and the `SharedTokenDecoder` that reads text
back out of a sampled bridge tensor. `blocks.py` is deliberately NOT exported --
its layers are implementation detail of `DiTXA`, not a public surface.

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
from .model import (
    DiTXA,
    DiTXAFinalLayer,
    create_ditxa,
)
from .sde import (
    SDE_TYPES,
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
from .token_decoder import (
    SharedTokenDecoder,
    create_shared_token_decoder,
)

__all__ = [
    "BRIDGE_PRESETS",
    "BridgeConfig",
    "BridgeSDE",
    "CosineDecayingVolatilitySDE",
    "DiTXA",
    "DiTXAFinalLayer",
    "FlowMatchingODE",
    "PeriodicVolatilitySDE",
    "PROMPT_KIND_TO_LABEL",
    "PROMPT_NUM_CLASSES",
    "SharedTokenDecoder",
    "TIME_EPS",
    "SDE_TYPES",
    "TOKEN_LAYOUTS",
    "UniformVolatilitySDE",
    "bridge_math_dtype",
    "bridge_to_token_flat",
    "compute_token_norms",
    "create_bridge_sde",
    "create_ditxa",
    "create_shared_token_decoder",
    "get_bridge_config",
    "norm_based_token_stops",
    "pad_id_token_stops",
    "prepare_bridge_batch",
    "token_flat_to_bridge",
]
