"""Model-agnostic machinery shared across the families. Nothing here is an architecture.

- `power_sampling/` — inference-time power sampling for any causal LM or VLM
- `masked_language_model/` — MLM/CLM training heads wrapping any backbone

Import from the leaf package, not from here — family packages carry no re-exports by
design (the reasoning is written out in `models/vision/__init__.py`).
"""
