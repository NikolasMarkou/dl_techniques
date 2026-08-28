"""Experimental text-embedding encoders over a character-level ASCII vocabulary.

This is a FAMILY directory, not a namespace: it carries this docstring and
nothing else. Import from the leaf package, e.g.::

    from dl_techniques.models.embeddings_experimental.ascii_bert import (
        create_ascii_bert,
    )

The family exists to run a controlled study. Every arm shares one skeleton --
the same ASCII embeddings, the same depth and width ladder, the same pooling
options, the same heads -- and differs ONLY in the sequence-mixing block, so a
difference in a reported metric is attributable to the block rather than to the
surrounding plumbing.

Leaf packages
-------------
``shared``
    The shared skeleton: :class:`~...shared.encoder.EmbeddingEncoder`, the
    ``BLOCK_REGISTRY`` that resolves a block type, and the heads. Adding an arm
    means one registry entry plus a thin leaf package, never a second copy of
    the skeleton.
``ascii_bert``
    Baseline arm. Multi-head self-attention via ``TransformerLayer``.
``ascii_clifford_bert``
    Clifford arm. Bidirectional sequence-mode ``CliffordNetBlock``, i.e. shifted
    geometric-product mixing with a depthwise-convolution context branch and no
    attention anywhere.

See ``README.md`` for the catalogue and the study's measured caveats.
"""
