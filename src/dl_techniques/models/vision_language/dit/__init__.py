"""Class-conditional latent Diffusion Transformer (DiT) -- public API re-exports.

A Keras 3, channels-last port of Peebles and Xie's Diffusion Transformer:
patchify a VAE latent, add a frozen 2-D sin-cos table, condition every block
on ``c = t_emb + y_emb`` through adaLN-Zero, and read out through a
zero-initialized final layer. The twelve published sizes are reachable by
name.

Exported (see ``__all__``): the variant registry (``DIT_VARIANTS``,
``VARIANT_FIELDS``, ``get_variant_config``, ``normalize_variant_name``), the
diffusion-side ``DiffusionConfig``, the block layers ``DiTBlock`` and
``DiTFinalLayer`` with their chunk-name tuples, the model ``DiT``, its
factory ``create_dit``, the pure helpers ``flattened_linear_xavier`` and
``unpatchify_tokens``, and the sampler ``GaussianDiffusion`` (``q_sample``,
``p_mean_variance``, the ancestral and DDIM reverse steps, and timestep
respacing). Block internals and leading-underscore helpers stay private to
their submodule.

    from dl_techniques.models.vision_language.dit import DiT, create_dit
    from dl_techniques.models.vision_language.dit.blocks import NUM_DIT_ADALN_CHUNKS

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
from .diffusion import (
    DEFAULT_CLIP_DENOISED,
    GaussianDiffusion,
    MODEL_MEAN_TYPES,
    MODEL_VAR_TYPES,
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
    "DEFAULT_CLIP_DENOISED",
    "DIT_ADALN_CHUNK_NAMES",
    "DIT_FINAL_CHUNK_NAMES",
    "DIT_VARIANTS",
    "DiT",
    "DiTBlock",
    "DiTFinalLayer",
    "DiffusionConfig",
    "GaussianDiffusion",
    "MODEL_INPUT_NAMES",
    "MODEL_MEAN_TYPES",
    "MODEL_VAR_TYPES",
    "VARIANT_FIELDS",
    "create_dit",
    "flattened_linear_xavier",
    "get_variant_config",
    "normalize_variant_name",
    "unpatchify_tokens",
]
