"""Embedding layers for ``dl_techniques``.

This package holds the library's embedding layers: positional embeddings
(learned, sinusoidal, rotary and its variants), patch embeddings, and the
transformer word/token embedding blocks. It also holds the factory that builds
them from a string key.

What this module exports
------------------------

``__all__`` names 11 things, and only THREE of them are layer classes:

* :class:`AxialRoPE2D` - 2D axial rotary position embedding.
* :class:`ClassLabelEmbedding` - class-label lookup table with a
  classifier-free-guidance dropout row.
* :class:`TimestepEmbedding` - cos-first sinusoidal timestep basis plus a
  ``Dense -> SiLU -> Dense`` head, for diffusion transformers.
* :func:`get_1d_sincos_pos_embed_from_grid`,
  :func:`get_2d_sincos_pos_embed_from_grid`, :func:`get_2d_sincos_pos_embed`,
  :func:`get_3d_sincos_pos_embed` - the four pure-NumPy MAE-style sin-cos
  table builders (three 2D, one 3D/video). Not layers: they return NumPy
  arrays meant to be installed with
  ``add_weight(trainable=False, initializer=keras.initializers.Constant(...))``.
* :func:`create_embedding_layer` - build a layer from a type key plus kwargs.
* :func:`create_embedding_from_config` - build a layer from a config dict.
* :func:`validate_embedding_config` - check a config without building.
* ``STRICT_DROPPED_KEY_MARKER`` - the substring the factory puts in its error
  message when strict mode rejects unsupported parameters. Callers match on it.

The package defines 22 layer classes. The other 19 are NOT re-exported here.
Import them from their own module, for example::

    from dl_techniques.layers.embedding.patch_embedding import (
        PatchEmbedding2D,
    )
    from dl_techniques.layers.embedding.patch_embed_3d import PatchEmbed3D
    from dl_techniques.layers.embedding.video_rope import VideoRoPE3D

Getting a layer
---------------

The factory in ``factory.py`` registers 15 keys, one per class it can build::

    from dl_techniques.layers.embedding import create_embedding_layer

    layer = create_embedding_layer(
        "positional_learned", max_seq_len=512, dim=768,
    )

Seven classes have no factory key and must be imported directly:
:class:`AxialRoPE2D`, :class:`ClassTokenPrepend`, :class:`MaskTokenApply`,
:class:`RegisterTokens`, :class:`HierarchicalCodebookEmbedding`,
:class:`PatchEmbed3D` and :class:`VideoRoPE3D`.

See ``README.md`` in this directory for per-class notes.
"""

from .axial_rope_2d import AxialRoPE2D
from .class_label_embedding import ClassLabelEmbedding
from .factory import (
    STRICT_DROPPED_KEY_MARKER,
    create_embedding_from_config,
    create_embedding_layer,
    validate_embedding_config,
)
from .sincos_pos_embed_2d import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
)
from .sincos_pos_embed_3d import get_3d_sincos_pos_embed
from .timestep_embedding import TimestepEmbedding

__all__ = [
    "AxialRoPE2D",
    "ClassLabelEmbedding",
    "STRICT_DROPPED_KEY_MARKER",
    "TimestepEmbedding",
    "create_embedding_from_config",
    "create_embedding_layer",
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "get_3d_sincos_pos_embed",
    "validate_embedding_config",
]
