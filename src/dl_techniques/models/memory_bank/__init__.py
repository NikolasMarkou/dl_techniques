"""Memory bank package for dual-tap neural memory-augmented transformers.

Provides:
    - ``LongTermMemoryBank`` / ``WorkingMemoryBank`` — key/value memory stores.
    - ``MemoryWriteController`` — projects pre-block hidden state into M_WM.
    - ``MemoryReadController`` — top-K STE retrieval + gated injection +
      4 anti-collapse aux losses.
    - ``PhaseScheduler`` — 4-phase curriculum callback.
    - ``WaveFieldMemoryLLM`` — sibling-stack memory-augmented model.

Per ``models/CLAUDE.md`` convention this ``__init__`` is intentionally
near-empty; submodules are imported directly by callers
(e.g. ``from dl_techniques.models.memory_bank.memory_banks import ...``).
"""
