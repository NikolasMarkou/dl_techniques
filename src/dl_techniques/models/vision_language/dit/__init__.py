"""Class-conditional latent Diffusion Transformer (DiT) -- public API re-exports.

A Keras 3, channels-LAST port of Peebles & Xie's Diffusion Transformer: patchify
a VAE latent, add a frozen 2-D sin-cos table, condition every block on
``c = t_emb + y_emb`` through adaLN-Zero, and read out through a zero-initialised
final layer. The twelve published sizes are reachable by name.

Currently exported: the variant registry and the diffusion-side config, the two
block layers, the model and its factory. The sampler (``diffusion.py``) lands at
step 7 of the port and is added here with it.

    from dl_techniques.models.vision_language.dit import DiT, create_dit

References:
    - Peebles, W. and Xie, S. "Scalable Diffusion Models with Transformers."
      arXiv:2212.09748, 2022. https://arxiv.org/abs/2212.09748
"""

from .blocks import (
    DIT_ADALN_CHUNK_NAMES,
    DIT_FINAL_CHUNK_NAMES,
    DiTBlock,
    DiTFinalLayer,
)
from .config import (
    DIT_VARIANTS,
    VARIANT_FIELDS,
    DiffusionConfig,
    get_variant_config,
    normalize_variant_name,
)
from .model import (
    CFG_GUIDED_CHANNELS,
    MODEL_INPUT_NAMES,
    DiT,
    create_dit,
    flattened_linear_xavier,
    unpatchify_tokens,
)

__all__ = [
    "CFG_GUIDED_CHANNELS",
    "DIT_ADALN_CHUNK_NAMES",
    "DIT_FINAL_CHUNK_NAMES",
    "DIT_VARIANTS",
    "DiT",
    "DiTBlock",
    "DiTFinalLayer",
    "DiffusionConfig",
    "MODEL_INPUT_NAMES",
    "VARIANT_FIELDS",
    "create_dit",
    "flattened_linear_xavier",
    "get_variant_config",
    "normalize_variant_name",
    "unpatchify_tokens",
]
