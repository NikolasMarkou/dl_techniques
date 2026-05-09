"""Memory bank package for dual-tap neural memory-augmented transformers.

Provides:
    - ``LongTermMemoryBank`` / ``WorkingMemoryBank`` — key/value memory stores.
    - ``MemoryWriteController`` — projects pre-block hidden state into M_WM.
    - ``MemoryReadController`` — top-K STE retrieval + gated injection +
      4 anti-collapse aux losses.
    - ``PhaseScheduler`` — 4-phase curriculum callback.
    - ``WaveFieldMemoryLLM`` — sibling-stack memory-augmented model.
    - ``memory_llm_custom_objects`` — ``custom_objects`` dict for
      ``keras.models.load_model`` (re-exported here per O9 so callers
      don't need to import from ``wave_field_memory_llm``).

Per ``models/CLAUDE.md`` convention this ``__init__`` is intentionally
near-empty; submodules are imported directly by callers
(e.g. ``from dl_techniques.models.memory_bank.memory_banks import ...``).
The ``memory_llm_custom_objects`` re-export is the one exception — it is
the canonical ``custom_objects`` source for save/load round-trip.
"""

from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    memory_llm_custom_objects,
)

__all__ = ["memory_llm_custom_objects"]
