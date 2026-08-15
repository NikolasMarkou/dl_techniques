"""4-phase curriculum scheduler for WaveFieldMemoryLLM.

Phase boundaries (steps, defaults)::

    Phase 1: [0, phase1_steps)                         50_000
    Phase 2: [phase1_steps, phase1+phase2)             25_000
    Phase 3: [phase1+phase2, phase1+phase2+phase3)    100_000
    Phase 4: >= sum                                  (no-op extension)

Behavior at each boundary:

- **Phase 1 -> 2**: freeze the backbone, enable memory injection and all
  aux losses, call
  ``model.warmup_memory_keys(warmup_dataset, num_batches=64)``.
- **Phase 2 -> 3**: unfreeze the backbone; memory and aux losses stay on.
- **Phase 3 -> 4**: identical trainable surface to phase 3 (no-op).

The scheduler reads the global step from ``model._global_step`` and
writes the new phase to ``model.current_phase``. Both are non-trainable
``add_weight`` float32 counters managed by :class:`WaveFieldMemoryLLM`,
so their values survive ``model.save`` / ``load_model`` round-trips.

**This callback flips no Python attribute, by design.** Assigning
``current_phase`` *is* the whole state change: the model derives memory
injection, aux-loss scaling and the backbone gradient mask from that one
``Variable`` inside its traced ``call``/``train_step``. The scheduler
previously set ``layer.trainable = False`` and
``read_controller.enable_gate_entropy = True`` (and friends) instead, and
neither could work: ``fit()`` traces ``train_function`` **before**
``on_train_begin`` (``keras/src/backend/tensorflow/trainer.py:360-361``)
and rebuilds it only from ``compile()``
(``keras/src/trainers/trainer.py:187``), so every later flip landed on
Python objects the already-traced graph no longer consults. Under the
documented 50k-step curriculum the log printed "entered phase 2" while
the backbone kept training and not one anti-collapse loss ever
contributed a gradient. The single eager action left here is zeroing the
backbone optimizer's learning-rate ``Variable``
(``model.set_backbone_optimizer_active``) — also a ``Variable``, also
read inside the traced step.

Skipping Phase 1 entirely is done by setting ``phase1_steps=0``. The trainer
that did this behind an ``--init-from`` flag, ``src/train/wave_field/train_memory.py``,
was DELETED on user instruction on 2026-08-13 (last present at commit ``9f3208319``);
no trainer ships for this package now, so ``phase1_steps=0`` is something a caller
sets directly.
"""

from typing import Any, Dict, Optional

import keras
import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# D2: Module-level phase constants. These are the canonical names —
# any code path that checks `current_phase == N` should import these.
# ---------------------------------------------------------------------

PHASE_WARMUP = 1            # P1 — backbone trainable, memory bypassed
PHASE_FREEZE_BACKBONE = 2   # P2 — backbone frozen, memory + aux losses on
PHASE_FULL = 3              # P3 — everything trainable, aux losses on
PHASE_EXTEND = 4            # P4 — same trainable surface as P3


# ---------------------------------------------------------------------


class PhaseScheduler(keras.callbacks.Callback):
    """Curriculum callback publishing the current phase to the model.

    :param phase1_steps: Length of Phase 1 in train batches.
    :param phase2_steps: Length of Phase 2.
    :param phase3_steps: Length of Phase 3 (Phase 4 is open-ended).
    :param warmup_dataset: ``tf.data.Dataset`` slice used by
        :meth:`WaveFieldMemoryLLM.warmup_memory_keys` to seed ``K_lt``
        from offline KMeans on hidden states. Required at Phase 1 -> 2
        boundary.
    :param warmup_num_batches: Number of batches consumed by the warmup.

    The scheduler is intentionally minimal — all heavy lifting (KMeans
    warmup, and every gate the phase implies) lives on the model, behind
    the ``current_phase`` ``Variable`` this callback assigns.
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
            return PHASE_WARMUP
        if step < self._b2:
            return PHASE_FREEZE_BACKBONE
        if step < self._b3:
            return PHASE_FULL
        return PHASE_EXTEND

    def _read_global_step(self) -> int:
        if not hasattr(self.model, "_global_step"):
            return 0
        return int(self.model._global_step.numpy())

    # ------------------------------------------------------------------
    # Phase application
    # ------------------------------------------------------------------

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-016 — the ONLY state
    # `_apply_phase` changes is the `current_phase` Variable plus the
    # backbone optimizer's learning-rate Variable. Do not add a
    # `layer.trainable = ...` or `read_controller.enable_* = ...` line
    # back: both are Python state that an already-traced `train_function`
    # never re-reads, which is how the entire curriculum came to be inert.
    # See decisions.md D-016.
    def _apply_phase(self, phase: int) -> None:
        """Publish `phase` to the model and run the P1->P2 warmup once."""
        # Assigned first: the model's in-graph gates (memory injection,
        # aux-loss scale, backbone gradient mask) all read this Variable,
        # so it must be current before the next batch executes.
        if hasattr(self.model, "current_phase"):
            self.model.current_phase.assign(phase)

        self._set_backbone_frozen(phase == PHASE_FREEZE_BACKBONE)

        if phase == PHASE_FREEZE_BACKBONE:
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

        logger.info(f"PhaseScheduler: entered phase {phase}")

    def _set_backbone_frozen(self, frozen: bool) -> None:
        """Zero (or restore) the backbone optimizer's learning rate.

        The in-graph gradient mask in ``WaveFieldMemoryLLM.train_step``
        already stops new gradient from reaching backbone weights in
        phase 2. This closes the other half: with Adam/AdamW a gradient of
        exactly zero still moves a weight, from the phase-1 moment
        estimates and from decoupled weight decay. Both terms are scaled
        by the learning rate, so zeroing it makes the freeze exact.

        A model without a ``set_backbone_optimizer_active`` method (a
        stand-in in tests, or a caller reusing this callback on another
        architecture) is left alone.
        """
        setter = getattr(self.model, "set_backbone_optimizer_active", None)
        if setter is None:
            return
        setter(not frozen)

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
            # O7: apply optional top_k schedule on phase transitions.
            self._apply_top_k_schedule(step)

    def _apply_top_k_schedule(self, step: int) -> None:
        """O7: if `model.top_k_schedule` is set, evaluate it at the
        current step and assign the result to
        `model.read_controller.top_k`. Done only on phase transitions
        (cheap; phase-1 -> phase-2 is the only frequent case)."""
        m = self.model
        sched = getattr(m, "top_k_schedule", None)
        if sched is None:
            return
        rc = getattr(m, "read_controller", None)
        if rc is None:
            return
        try:
            new_top_k = int(sched(step))
        except Exception as exc:
            logger.warning(
                f"PhaseScheduler: top_k_schedule({step}) raised {exc}; "
                f"keeping top_k={rc.top_k}"
            )
            return
        if new_top_k <= 0:
            logger.warning(
                f"PhaseScheduler: top_k_schedule returned {new_top_k}; "
                f"keeping top_k={rc.top_k}"
            )
            return
        if new_top_k != rc.top_k:
            logger.info(
                f"PhaseScheduler: top_k {rc.top_k} -> {new_top_k} (step={step})"
            )
            rc.top_k = new_top_k

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
