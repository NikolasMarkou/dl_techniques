"""4-phase curriculum scheduler for WaveFieldMemoryLLM.

Phase boundaries (steps, defaults)::

    Phase 1: [0, phase1_steps)                         50_000
    Phase 2: [phase1_steps, phase1+phase2)             25_000
    Phase 3: [phase1+phase2, phase1+phase2+phase3)    100_000
    Phase 4: >= sum                                  (no-op extension)

Behavior at each boundary:

- **Phase 1 -> 2**: freeze backbone (token_embeddings + decoder blocks +
  final_norm), unfreeze memory + gate, enable all aux losses, call
  ``model.warmup_memory_keys(warmup_dataset, num_batches=64)``.
- **Phase 2 -> 3**: unfreeze backbone (everything trainable), keep aux
  losses on.
- **Phase 3 -> 4**: identical trainable surface to phase 3 (no-op).

The scheduler reads the global step from ``model._global_step`` (a
non-trainable ``add_weight`` int64 counter) and writes the new phase to
``model.current_phase`` (a non-trainable ``add_weight`` int32 counter).
Both are managed by :class:`WaveFieldMemoryLLM` so that the values
survive ``model.save`` / ``load_model`` round-trips.

The ``--init-from`` flag in ``train_memory.py`` sets ``phase1_steps=0``
to skip Phase 1 entirely.
"""

from typing import Any, Dict, Optional

import keras
import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


class PhaseScheduler(keras.callbacks.Callback):
    """Curriculum callback flipping phase + trainable flags + aux flags.

    :param phase1_steps: Length of Phase 1 in train batches.
    :param phase2_steps: Length of Phase 2.
    :param phase3_steps: Length of Phase 3 (Phase 4 is open-ended).
    :param warmup_dataset: ``tf.data.Dataset`` slice used by
        :meth:`WaveFieldMemoryLLM.warmup_memory_keys` to seed ``K_lt``
        from offline KMeans on hidden states. Required at Phase 1 -> 2
        boundary.
    :param warmup_num_batches: Number of batches consumed by the warmup.

    The scheduler is intentionally minimal — all heavy lifting (KMeans,
    trainable-flag flipping for the entire backbone) lives on the model.
    """

    def __init__(
        self,
        phase1_steps: int = 50_000,
        phase2_steps: int = 25_000,
        phase3_steps: int = 100_000,
        warmup_dataset: Optional[Any] = None,
        warmup_num_batches: int = 64,
    ) -> None:
        super().__init__()
        if phase1_steps < 0 or phase2_steps < 0 or phase3_steps < 0:
            raise ValueError("phase steps must be non-negative")

        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.phase3_steps = phase3_steps
        self.warmup_num_batches = warmup_num_batches
        self._warmup_dataset = warmup_dataset

        self._b1 = phase1_steps
        self._b2 = phase1_steps + phase2_steps
        self._b3 = phase1_steps + phase2_steps + phase3_steps

        self._last_phase: Optional[int] = None
        self._warmup_done = False

        logger.info(
            f"PhaseScheduler: boundaries (steps) "
            f"P1={phase1_steps}, P2={phase2_steps}, P3={phase3_steps}; "
            f"warmup_num_batches={warmup_num_batches}"
        )

    # ------------------------------------------------------------------
    # Phase computation
    # ------------------------------------------------------------------

    def _step_to_phase(self, step: int) -> int:
        if step < self._b1:
            return 1
        if step < self._b2:
            return 2
        if step < self._b3:
            return 3
        return 4

    def _read_global_step(self) -> int:
        if not hasattr(self.model, "_global_step"):
            return 0
        return int(self.model._global_step.numpy())

    # ------------------------------------------------------------------
    # Phase application
    # ------------------------------------------------------------------

    def _apply_phase(self, phase: int) -> None:
        """Apply trainable flags + aux-loss flags + warmup for `phase`."""
        if phase == 1:
            self._set_backbone_trainable(True)
            self._set_memory_trainable(True)
            self._set_aux_losses(False)
        elif phase == 2:
            # Freeze backbone, unfreeze memory, enable aux losses.
            self._set_backbone_trainable(False)
            self._set_memory_trainable(True)
            self._set_aux_losses(True)
            # Warmup K_lt via offline KMeans (once).
            if not self._warmup_done and self._warmup_dataset is not None:
                if hasattr(self.model, "warmup_memory_keys"):
                    self.model.warmup_memory_keys(
                        self._warmup_dataset,
                        num_batches=self.warmup_num_batches,
                    )
                    self._warmup_done = True
                else:
                    logger.warning(
                        "PhaseScheduler: model has no warmup_memory_keys"
                    )
        elif phase == 3:
            # Unfreeze backbone; aux losses stay on.
            self._set_backbone_trainable(True)
            self._set_memory_trainable(True)
            self._set_aux_losses(True)
        else:  # phase 4: same trainable surface as phase 3.
            self._set_backbone_trainable(True)
            self._set_memory_trainable(True)
            self._set_aux_losses(True)

        # Persist the phase value on the model.
        if hasattr(self.model, "current_phase"):
            self.model.current_phase.assign(phase)

        logger.info(f"PhaseScheduler: entered phase {phase}")

    def _set_backbone_trainable(self, flag: bool) -> None:
        m = self.model
        for attr in (
            "token_embeddings",
            "position_embeddings",
            "embed_norm",
            "final_norm",
            "lm_head",
        ):
            obj = getattr(m, attr, None)
            if obj is not None:
                obj.trainable = flag
        blocks = getattr(m, "blocks", None) or []
        for blk in blocks:
            blk.trainable = flag

    def _set_memory_trainable(self, flag: bool) -> None:
        m = self.model
        for attr in (
            "lt_memory",
            "wm_memory",
            "read_controller",
            "write_controller",
        ):
            obj = getattr(m, attr, None)
            if obj is not None:
                obj.trainable = flag

    def _set_aux_losses(self, flag: bool) -> None:
        rc = getattr(self.model, "read_controller", None)
        if rc is None:
            return
        rc.enable_gate_entropy = flag
        rc.enable_load_balance = flag
        rc.enable_z_loss = flag
        rc.enable_diversity = flag
        rc.enable_infonce = flag

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        # Apply the phase implied by the current global step (handles
        # resume + --init-from with phase1_steps=0).
        step = self._read_global_step()
        phase = self._step_to_phase(step)
        self._apply_phase(phase)
        self._last_phase = phase

    def on_train_batch_begin(self, batch, logs=None):
        step = self._read_global_step()
        phase = self._step_to_phase(step)
        if phase != self._last_phase:
            self._apply_phase(phase)
            self._last_phase = phase

    # ------------------------------------------------------------------
    # Config (for callback save/restore via training logs)
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "phase1_steps": self.phase1_steps,
            "phase2_steps": self.phase2_steps,
            "phase3_steps": self.phase3_steps,
            "warmup_num_batches": self.warmup_num_batches,
        }
