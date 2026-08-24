"""General-purpose models — architectures not tied to a single input modality.

- `kan/` — Kolmogorov-Arnold Networks
- `mothnet/` — MothNet (bio-inspired)
- `power_mlp/` — Power MLP

Import from the leaf package, not from here — this family package carries no re-exports
by design (the reasoning is written out in `models/vision/__init__.py`):

    from dl_techniques.models.general_purpose.kan import KAN
"""
