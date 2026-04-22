"""Video-JEPA-Clifford model package.

A patch-based video JEPA backbone using CliffordNet primitives for video
streaming. Core components:

- :class:`VideoJEPAConfig` — dataclass config (:mod:`.config`).
- :class:`VideoJEPACliffordEncoder` — hybrid PatchEmbedding + Clifford blocks
  (:mod:`.encoder`).
- :class:`VideoJEPAPredictor` — factorized spatial/causal-temporal predictor
  (pixels-only, iter-3 / D-013) (:mod:`.predictor`).
- :class:`VideoJEPA` — top-level model with streaming inference API
  (:mod:`.model`).

Import from submodules directly per package convention.
"""
