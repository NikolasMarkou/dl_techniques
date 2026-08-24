"""FastVLM — public API re-exports.

``FastVLM.MODEL_VARIANTS`` already carried the canonical spelling; only the
module-level ``create_fastvlm`` factory and these exports were missing.
``AttentionBlockVLM`` is re-exported because it is this package's own block,
not a shared layer.
"""
from dl_techniques.models.fastvlm.components import AttentionBlockVLM
from dl_techniques.models.fastvlm.model import FastVLM, create_fastvlm

__all__ = [
    "FastVLM",
    "create_fastvlm",
    "AttentionBlockVLM",
]
