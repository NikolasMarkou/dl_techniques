"""Pattern-3 (subword CLM) trainer for the Qwen3 model package.

See ``train_qwen.py`` for the entry point and ``common.py`` for the
config/pipeline it drives. **Scope: `Qwen3`'s plain causal-LM surface only.**
The ``qwen/`` package also hosts ``Qwen3Next`` (MoE-hybrid, own routing/loss
shape) and ``Qwen3EmbeddingModel``/``Qwen3RerankerModel`` (contrastive
retrieval) in sibling modules (``qwen3_next.py``, ``qwen3_embeddings.py``) --
neither is imported here. See
``plans/plan-2026-09-12T173329-e20362c4/decisions.md`` D-001 for why.
"""
