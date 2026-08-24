"""Memory bank package for dual-tap neural memory-augmented transformers.

Provides:
    - ``LongTermMemoryBank`` / ``WorkingMemoryBank`` — key/value memory stores.
    - ``MemoryWriteController`` — projects pre-block hidden state into M_WM.
    - ``MemoryReadController`` — top-K STE retrieval + gated injection +
      anti-collapse aux losses, scaled at runtime by the ``aux_scale``
      gate the model derives from ``current_phase``.
    - ``PhaseScheduler`` — 4-phase curriculum callback. It assigns the
      model's ``current_phase`` ``Variable`` (and zeroes the backbone
      optimizer's learning rate for the frozen phase); it flips no Python
      attribute, because ``fit()`` traces ``train_function`` before the
      first callback hook runs.
    - ``WaveFieldMemoryLLM`` — sibling-stack memory-augmented model, with
      ``MODEL_VARIANTS`` ("tiny", "small", "medium", "large", "xl") and the
      ``create_wave_field_memory_llm`` factory over them.
    - ``memory_llm_custom_objects`` — ``custom_objects`` dict for
      ``keras.models.load_model`` (re-exported here per O9 so callers
      don't need to import from ``wave_field_memory_llm``).

Submodules may still be imported directly (e.g.
``from dl_techniques.models.memory_bank.memory_banks import ...``); the names
below are the curated public surface.
"""

from dl_techniques.models.memory_bank.memory_banks import (
    LongTermMemoryBank,
    WorkingMemoryBank,
)
from dl_techniques.models.memory_bank.write_controller import (
    MemoryWriteController,
)
from dl_techniques.models.memory_bank.read_controller import (
    MemoryReadController,
)
from dl_techniques.models.memory_bank.phase_scheduler import PhaseScheduler
from dl_techniques.models.memory_bank.memory_stats import MemoryStats
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
    create_wave_field_memory_llm,
    memory_llm_custom_objects,
)

__all__ = [
    "WaveFieldMemoryLLM",
    "create_wave_field_memory_llm",
    "memory_llm_custom_objects",
    "LongTermMemoryBank",
    "WorkingMemoryBank",
    "MemoryWriteController",
    "MemoryReadController",
    "PhaseScheduler",
    "MemoryStats",
]
