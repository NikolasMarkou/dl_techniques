"""H-Net dynamic-chunking layers.

Three layers implement H-Net's learned, data-dependent sequence chunking, and
they only make sense as a triple::

    hidden, mask                                    (B, L, D) / (B, L)
        │
        ▼
    RoutingModule    -> boundary_prob, boundary_mask, selected_probs
        │
        ▼
    ChunkLayer       -> inner hidden (B, max_chunks, D) + inner validity mask
        │
        ▼  (inner network runs on the shorter sequence)
        │
    DeChunkLayer     -> hidden back at full resolution (B, L, D)

:class:`~dl_techniques.layers.dynamic_chunking.routing_module.RoutingModule`
detects boundaries from adjacent-token cosine similarity,
:class:`~dl_techniques.layers.dynamic_chunking.chunk_layer.ChunkLayer`
downsamples to the boundary positions with a position-order stable partition,
and
:class:`~dl_techniques.layers.dynamic_chunking.dechunk_layer.DeChunkLayer`
upsamples back with the EMA recurrence over the inner sequence.

This package has **no factory and no registry**.

.. code-block:: text

    # DECISION plan-2026-09-09T042752-6d66ac56/D-004
    # Do NOT add a `factory.py` / `DYNAMIC_CHUNKING_REGISTRY` here, and do NOT
    # register these three classes into a sibling domain factory
    # (`attention/`, `ffn/`, ...) to "follow the DocRes precedent". There is no
    # sequence-chunking domain factory to join, and there is exactly ONE
    # consumer (`models/language/hnet/`). A registry with one key is dispatch
    # machinery with no payoff, and DocRes measured that each new registry
    # entry reds hard-coded population counts in unrelated suites (six of them
    # in its step 4). The shipped house shape for a model-family-specific
    # sequence-restructuring stack with a single consumer and no factory is
    # `layers/blt/`. See decisions.md D-004 for the rejected alternative and
    # the price paid (a 33rd `layers/` subpackage, three single-use classes).
    #
    # DECISION plan-2026-09-09T042752-6d66ac56/D-015: the SAME ruling, restated
    # at the step that actually created this package surface, so the anchor
    # audit can link both entries to the site. D-015 also records that every
    # count `layers/CLAUDE.md` states about this package was RE-DERIVED rather
    # than incremented; if you add or remove a class here, re-run the commands
    # printed beside those numbers instead of editing them.

Each class is registered under its OWN module path
(``dl_techniques.layers.dynamic_chunking.<module>``), so importing this package
is what makes all three deserializable.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
"""

from .chunk_layer import ChunkLayer
from .dechunk_layer import DeChunkLayer
from .routing_module import RoutingModule

__all__ = [
    "ChunkLayer",
    "DeChunkLayer",
    "RoutingModule",
]
