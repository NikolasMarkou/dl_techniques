"""H-Net dynamic-chunking layers.

Three layers implement H-Net's learned, data-dependent sequence chunking:
:class:`~dl_techniques.layers.dynamic_chunking.routing_module.RoutingModule`
(boundary detection), ``ChunkLayer`` (downsampling to the boundary positions)
and ``DeChunkLayer`` (EMA upsampling back to full resolution).

Import from the leaf module, e.g.::

    from dl_techniques.layers.dynamic_chunking.routing_module import RoutingModule

This ``__init__`` is deliberately a docstring and nothing else for now; the
curated package surface (``__all__`` plus the re-exports) is created in one
place once all three layers exist, so that the ``layers/`` subpackage and
``__all__`` census numbers stated in ``layers/CLAUDE.md`` are re-derived exactly
once rather than three times.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
"""
