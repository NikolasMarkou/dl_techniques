"""FastVLM — public API re-exports.

Exports the ``FastVLM`` model, its ``create_fastvlm`` factory, and
``AttentionBlockVLM``, the package's own attention block.
"""
from dl_techniques.models.vision_language.fastvlm.components import AttentionBlockVLM
from dl_techniques.models.vision_language.fastvlm.model import FastVLM, create_fastvlm

__all__ = [
    "FastVLM",
    "create_fastvlm",
    "AttentionBlockVLM",
]
