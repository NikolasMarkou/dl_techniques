"""Trainers for `Qwen3EmbeddingModel` (bi-encoder) and `Qwen3RerankerModel`.

See `common.py` for the shared config/dataset/model scaffold and
`train_qwen3_embedding.py` / `train_qwen3_reranker.py` for the two entry
points. Both models are architecturally NOT causal-LM shaped (a pooled
vector and a scalar probability, respectively), so neither is wrapped in
`CausalLanguageModel` -- see `common.py`'s module docstring and
`plans/plan-2026-09-13T073704-245ab5d5/decisions.md` D-004/D-013.

New sibling package, not colocated in `src/train/qwen/`: that package's own
D-001 anchor (`plans/plan-2026-09-12T173329-e20362c4/decisions.md`) forbids
importing `qwen3_embeddings.py` from it.
"""
