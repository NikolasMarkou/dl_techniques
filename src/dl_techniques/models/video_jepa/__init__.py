"""Video-JEPA-Clifford model package.

A patch-based video JEPA backbone using CliffordNet primitives for drone-
footage streaming. Core components:

- :class:`VideoJEPAConfig` — dataclass config (:mod:`.config`).
- :class:`VideoJEPACliffordEncoder` — hybrid PatchEmbedding + Clifford blocks
  (:mod:`.encoder`).
- :class:`TelemetryEmbedder` — continuous sin/cos telemetry encoder
  (:mod:`.telemetry_embedder`).
- :class:`VideoJEPAPredictor` — factorized spatial/causal-temporal predictor
  with AdaLN-zero telemetry conditioning (:mod:`.predictor`).
- :class:`VideoJEPA` — top-level model with streaming inference API
  (:mod:`.model`).

Import from submodules directly per package convention.
"""
