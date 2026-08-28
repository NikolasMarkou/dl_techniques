"""Embedding layers for ``dl_techniques``.

This package holds the library's embedding layers: positional embeddings
(learned, sinusoidal, rotary and its variants), patch embeddings, and the
transformer word/token embedding blocks. It also holds the factory that builds
them from a string key.

What this module exports
------------------------

``__all__`` names 5 things, and only ONE of them is a layer class:

* :class:`AxialRoPE2D` - 2D axial rotary position embedding.
* :func:`create_embedding_layer` - build a layer from a type key plus kwargs.
* :func:`create_embedding_from_config` - build a layer from a config dict.
* :func:`validate_embedding_config` - check a config without building.
* ``STRICT_DROPPED_KEY_MARKER`` - the substring the factory puts in its error
  message when strict mode rejects unsupported parameters. Callers match on it.

The package defines 18 layer classes. The other 17 are NOT re-exported here.
Import them from their own module, for example::

    from dl_techniques.layers.embedding.patch_embedding import (
        PatchEmbedding2D,
    )

Getting a layer
---------------

The factory in ``factory.py`` registers 13 keys, one per class it can build::

    from dl_techniques.layers.embedding import create_embedding_layer

    layer = create_embedding_layer(
        "positional_learned", max_seq_len=512, dim=768,
    )

Five classes have no factory key and must be imported directly:
:class:`AxialRoPE2D`, :class:`ClassTokenPrepend`, :class:`MaskTokenApply`,
:class:`RegisterTokens` and :class:`HierarchicalCodebookEmbedding`.

See ``README.md`` in this directory for per-class notes.
"""

from .axial_rope_2d import AxialRoPE2D
from .factory import (
    STRICT_DROPPED_KEY_MARKER,
    create_embedding_from_config,
    create_embedding_layer,
    validate_embedding_config,
)

__all__ = [
    "AxialRoPE2D",
    "STRICT_DROPPED_KEY_MARKER",
    "create_embedding_from_config",
    "create_embedding_layer",
    "validate_embedding_config",
]
