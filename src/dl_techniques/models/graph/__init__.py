"""Graph models — architectures over graph-structured input.

- `graph_energy_transformer/` — Graph Energy Transformer (node anomaly + graph
  classification)
- `relgt/` — Relational Graph Transformer
- `shgcn/` — Simplified Hyperbolic GCN

Import from the leaf package, not from here — this family package carries no re-exports
by design (the reasoning is written out in `models/vision/__init__.py`):

    from dl_techniques.models.graph.relgt import create_relgt_model
"""
