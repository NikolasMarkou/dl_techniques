"""Class-conditional latent Diffusion Transformer (DiT) -- public API re-exports.

A Keras 3, channels-LAST port of Peebles & Xie's Diffusion Transformer: patchify
a VAE latent, add a frozen 2-D sin-cos table, condition every block on
``c = t_emb + y_emb`` through adaLN-Zero, and read out through a zero-initialised
final layer. The twelve published sizes are reachable by name.

**Exported** (:data:`__all__`, alphabetized): the variant registry
(``DIT_VARIANTS``, ``VARIANT_FIELDS``, ``get_variant_config``,
``normalize_variant_name``) and the diffusion-side ``DiffusionConfig``; the two
block layers ``DiTBlock`` and ``DiTFinalLayer`` with the two chunk-name tuples
that pin their modulation order; the model ``DiT``, its factory ``create_dit``
and the two pure helpers a caller may legitimately need to reason about the
port (``flattened_linear_xavier``, ``unpatchify_tokens``); the named constants
``CFG_GUIDED_CHANNELS``, ``MODEL_INPUT_NAMES``, ``DEFAULT_CLIP_DENOISED``,
``MODEL_MEAN_TYPES``, ``MODEL_VAR_TYPES``; and the sampler
``GaussianDiffusion``, which owns ``q_sample``, ``p_mean_variance``, the
ancestral and DDIM reverse steps, their loops, and the timestep respacing that
remaps a shortened chain's index back to the original one before the model sees
it.

**Deliberately NOT exported.** Block internals and module-private arithmetic stay
behind their submodule: the chunk counts ``NUM_DIT_ADALN_CHUNKS`` /
``NUM_DIT_FINAL_CHUNKS`` (derived from the exported name tuples, so exporting
them would be a second thing to keep in lockstep), ``LABEL_TABLE_INIT_STDDEV``,
and every leading-underscore helper. Nothing here binds a name equal to a
submodule (``blocks``, ``config``, ``diffusion``, ``model``), so
``from ... import model`` still reaches the module and not a symbol.

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
