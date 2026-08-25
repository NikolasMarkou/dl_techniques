"""Model-agnostic machinery shared across the families. Nothing here is an architecture.

- `power_sampling/` — inference-time power sampling for any causal LM or VLM

Import from the leaf package, not from here — family packages carry no re-exports by
design (the reasoning is written out in `models/vision/__init__.py`).
"""
