"""Token Superposition Training (TST) — standalone, model-agnostic training utility.

This module implements TST as an *opt-in* training recipe usable with any
``keras.Model`` causal language model in ``dl_techniques`` (CliffordNetLM, NAM,
CLIP-text-tower, BLT-style models, etc.). It does **not** import any concrete
model class; the user wires it in at model-construction time.

Phase mechanics
---------------
Training is split into two phases controlled by a single boolean flag
``state.phase_active``:

::

    Phase 1  (state.phase_active == True;  step <  ratio * total_steps)
    ──────────────────────────────────────────────────────────────────
        inputs   : (B, N/s, s)  int   — bagged token IDs
            │
            ▼
        TSTEmbedding (rank-3 path):
            inner_emb(inputs)            → (B, N/s, s, d)
            mean(..., axis=-2)           → (B, N/s, d)
            │
            ▼
        Model trunk (sees (B, N/s, d), unchanged)
            │
            ▼
        Head logits: (B, N/s, V)
            │
            ▼
        TSTCausalLMLoss (rank-3 path):
            for j in range(s):
                loss += w_j * CE(logits, labels[..., j])
            return masked-mean(loss)

    Phase 2  (state.phase_active == False; step >= ratio * total_steps)
    ──────────────────────────────────────────────────────────────────
        inputs   : (B, N)       int   — plain NTP token IDs
            │
            ▼
        TSTEmbedding (rank-2 path / canary):
            inner_emb(inputs)            → (B, N, d)
            │
            ▼
        Model trunk (sees (B, N, d), unchanged)
            │
            ▼
        Head logits: (B, N, V)
            │
            ▼
        TSTCausalLMLoss (rank-2 path / canary):
            ≡ MaskedCausalLMLoss (bit-equivalent up to floating-point order)

The phase flip is implemented at the **dataset level** by using **two distinct
dataset transforms** — one per phase — and TWO ``model.fit(...)`` calls
(DECISION plan_2026-05-17_413eae7d/D-007; see D-005 fallback and D-006). Each
transform has a single, statically-determined output rank, so there is no
``tf.cond`` in the data graph and no shape-inference hazard. ``TSTEmbedding``
still dispatches on **input rank**, not on a phase flag (decision D-002 below).
This makes the dataset the source of truth for "what phase are we in" and
makes the boundary explicit in the user's training script.

``TSTState.phase_active`` exists for **observability only** — the callback
flips it at the boundary so users can hook into the transition for logging or
metric splits, but neither dataset transform reads it. ``state.global_step``
is bumped per-batch for the same reason. Invariant 6 is therefore preserved
trivially: no trainable weight or optimizer slot is touched by the flip.

Canonical user pattern::

    state, callbacks, phase1_fn, phase2_fn = apply_tst(config, total_steps=T)
    # Phase 1 — bagged training
    model.fit(
        phase1_fn(raw_ds),
        epochs=phase1_epochs,
        callbacks=user_callbacks + callbacks,
    )
    # Phase 2 — vanilla NTP fine-tune; use initial_epoch= for clean resume
    model.fit(
        phase2_fn(raw_ds),
        epochs=total_epochs,
        initial_epoch=phase1_epochs,
        callbacks=user_callbacks + callbacks,
    )

bag_size=1 canary
-----------------
The load-bearing correctness invariant: with ``bag_size=1``,
``TSTEmbedding`` and ``TSTCausalLMLoss`` are bit-equivalent (atol 1e-6) to a
plain ``keras.layers.Embedding`` + ``MaskedCausalLMLoss`` pipeline. This is
verified by unit tests (``inv1_canary_embedding`` and ``inv2_canary_loss``).
If those fail, **do not proceed** — every other claim depends on them.

What the tests in this module verify
------------------------------------
Validated by ``tests/test_train/test_common_token_superposition.py``:

1. TSTEmbedding ``bag_size=1`` ≡ plain Embedding (canary).
2. TSTCausalLMLoss ``bag_size=1`` ≡ MaskedCausalLMLoss (canary).
3. Sum-of-CE loop equals mean of one-hot CE terms (paper Appendix B).
4. Embedding aggregation is ``mean``, not ``sum``.
5. TSTEmbedding has exactly ``vocab*dim`` trainable params; TSTCausalLMLoss
   has zero trainable params.
6. Phase flip mutates no trainable weight and no optimizer slot.
7. Full ``get_config`` round-trip for layer + loss.
8. Tied-LM-head passthrough: ``TSTEmbedding.embeddings is _inner.embeddings``.

What the tests do **not** verify
--------------------------------
The paper's 2–3× wall-clock-to-target-PPL claim is **not** tested here — it
requires real training runs with held-out perplexity tracking. Without a
held-out PPL callback you cannot detect whether TST helped. See
``analyses/analysis_2026-05-17_76fa50b7/summary.md`` Prereq A for the
recommended evaluation harness before claiming a paper-replication result.

Decisions anchored in this module
---------------------------------
The five plan-level decisions are anchored at their point of impact with
``# DECISION plan_2026-05-17_413eae7d/D-NNN`` comments. Summary:

1. **D-001 Construction-time wiring**. The user instantiates ``TSTEmbedding``
   in their model factory; ``apply_tst`` does not introspect or mutate the
   model. Anchored at ``TSTEmbedding.__init__`` docstring.
2. **D-002 Input-rank dispatch in ``TSTEmbedding.call``**. The layer switches
   on ``len(inputs.shape)`` rather than reading ``state.phase_active`` via
   ``keras.ops.cond`` (LESSONS L23: cond traces both branches under
   ``tf.function``). Anchored at ``TSTEmbedding.call``.
3. **D-003 Sum-of-CE via Python ``for`` loop**, not multi-hot CE. Paper
   Appendix B proves the identity; ``s ∈ [4, 16]`` keeps perf irrelevant.
   Anchored at ``TSTCausalLMLoss.call`` (bagged path).
4. **D-004 Off-by-``s`` handled via label reshape**, not by changing the
   upstream chunker. We reuse the existing ``preprocess_clm_packed_dataset``
   ``(input_ids, labels)`` pair (already shift-by-1) and reshape labels
   ``(B, N) → (B, N/s, s)`` for Phase 1. Constraint ``N % s == 0`` is
   enforced loudly at transform construction time (``ValueError``).
   Anchored at ``tst_dataset_transform``.
5. **D-005 (HEDGED) Single-dataset state-read vs two-dataset swap**. The
   first attempt was a single ``tf.data.Dataset`` with ``tf.cond`` on
   ``state.phase_active``. Falsification C fired (soft form): no trace
   error, but a one-batch shape lag because ``tf.data`` advances one batch
   ahead of the captured ``tf.Variable`` read. See D-006 in
   ``plans/plan_2026-05-17_413eae7d/decisions.md``.
6. **D-006 / D-007 Two-transform fallback chosen**. ``apply_tst`` returns
   ``(state, callbacks, phase1_transform, phase2_transform)``. Each
   transform has a single static output rank. No ``tf.cond`` in the data
   graph. ``state.phase_active`` becomes advisory.

References
----------
- TST paper: §20 (the load-bearing section for invariants 1–6).
- ``dl_techniques.losses.MaskedCausalLMLoss`` — canary baseline.
- ``plans/plan_2026-05-17_413eae7d/decisions.md`` — full decision log.
- ``plans/plan_2026-05-17_413eae7d/findings.md`` — exploration findings F-1..F-3.
"""

import dataclasses
import functools
from typing import Callable, List, Literal, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# TSTConfig — frozen hyperparameter container
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TSTConfig:
    """Hyperparameters for Token Superposition Training.

    :param bag_size: Number of tokens superposed per latent position. ``s`` in
        the paper. ``bag_size=1`` is the canary configuration (equivalent to
        vanilla next-token prediction). Must be ``>= 1``.
    :param phase1_step_ratio: Fraction of total training steps spent in Phase
        1 (bagged). The phase callback flips ``state.phase_active`` to False
        at step ``floor(total_steps * phase1_step_ratio)``. Must lie in
        ``[0.0, 1.0]``.
    :param within_bag_weighting: How to weight within-bag CE terms in the
        sum-of-CE form. ``"uniform"`` gives each ``j ∈ [0, bag_size)`` weight
        ``1/bag_size``. ``"power_law"`` uses
        ``w_j ∝ (j+1)^(-within_bag_alpha)`` then normalises to sum 1.
    :param within_bag_alpha: Exponent for the power-law weighting scheme. Must
        be ``> 0``. Ignored when ``within_bag_weighting == "uniform"``.
    """

    bag_size: int = 6
    phase1_step_ratio: float = 0.25
    within_bag_weighting: Literal["uniform", "power_law"] = "uniform"
    within_bag_alpha: float = 0.6

    def __post_init__(self) -> None:
        if not isinstance(self.bag_size, int) or self.bag_size < 1:
            raise ValueError(
                f"TSTConfig.bag_size must be a positive int, got {self.bag_size!r}."
            )
        if not (0.0 <= self.phase1_step_ratio <= 1.0):
            raise ValueError(
                "TSTConfig.phase1_step_ratio must be in [0.0, 1.0], got "
                f"{self.phase1_step_ratio!r}."
            )
        if self.within_bag_weighting not in ("uniform", "power_law"):
            raise ValueError(
                "TSTConfig.within_bag_weighting must be 'uniform' or 'power_law', "
                f"got {self.within_bag_weighting!r}."
            )
        if self.within_bag_alpha <= 0.0:
            raise ValueError(
                "TSTConfig.within_bag_alpha must be > 0, got "
                f"{self.within_bag_alpha!r}."
            )


# ---------------------------------------------------------------------
# TSTState — single source of truth for the phase flag + global step
# ---------------------------------------------------------------------


class TSTState:
    """Training-loop observability state (advisory after D-007).

    Holds two ``tf.Variable``s (``phase_active``, ``global_step``) and a
    constant ``bag_size``. The variables are deliberately non-trainable and
    untracked by Keras — they are training-loop scaffolding, not model
    parameters (invariant 6: the phase flip must not touch any trainable
    weight or optimizer slot).

    **Role after D-007 (Falsification C resolution).** ``phase_active`` is
    now **advisory only**: neither ``tst_phase1_transform`` nor
    ``tst_phase2_transform`` reads it. The two-transform fallback (D-006 /
    D-007) makes the phase boundary explicit in the user's training script.
    ``TSTPhaseCallback`` still flips ``phase_active`` at the boundary so
    user code can hook into the transition for logging or metric splits.
    ``global_step`` is bumped per-batch for the same reason.

    :param bag_size: The TST ``bag_size`` (``s``). Stored for downstream
        consumers (the dataset transform, callbacks) so they don't need to
        carry it separately.
    :param phase_active_init: Initial value of ``phase_active``. Default
        ``True`` so a fresh state starts in Phase 1.
    """

    def __init__(self, bag_size: int, phase_active_init: bool = True) -> None:
        if not isinstance(bag_size, int) or bag_size < 1:
            raise ValueError(
                f"TSTState.bag_size must be a positive int, got {bag_size!r}."
            )
        self.bag_size = bag_size
        self.phase_active = tf.Variable(
            bool(phase_active_init),
            dtype=tf.bool,
            trainable=False,
            name="tst_phase_active",
        )
        self.global_step = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name="tst_global_step",
        )

    def reset(self, phase_active: bool = True) -> None:
        """Reset state to a freshly-initialised configuration. Test helper."""
        self.phase_active.assign(bool(phase_active))
        self.global_step.assign(0)


# ---------------------------------------------------------------------
# TSTEmbedding — rank-dispatched drop-in replacement for keras.layers.Embedding
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TSTEmbedding(keras.layers.Layer):
    """Drop-in replacement for ``keras.layers.Embedding`` with rank-3 bagged path.

    The layer wraps an inner ``keras.layers.Embedding`` and dispatches on
    **input rank** (DECISION plan_2026-05-17_413eae7d/D-002):

    - ``rank == 2``: plain lookup. Returns ``inner(x)`` of shape ``(B, N, d)``.
      Also the path taken when ``bag_size == 1`` regardless of rank (canary).
    - ``rank == 3``: bagged lookup. Input ``(B, N/s, s)`` → ``inner(x)`` of
      shape ``(B, N/s, s, d)`` → mean over the bag axis → ``(B, N/s, d)``.

    The user wires this in at **construction time** (DECISION
    plan_2026-05-17_413eae7d/D-001) — ``apply_tst`` does not introspect or
    mutate the model. To preserve tied-LM heads, the inner embedding's
    ``embeddings`` variable is exposed via ``self.embeddings``.

    :param vocab_size: Vocabulary size ``V``.
    :param output_dim: Embedding dimension ``d``.
    :param bag_size: Bag size ``s``. When ``bag_size == 1`` the rank-3 path
        is skipped even if a rank-3 input is given (canary semantics).
    :param embeddings_initializer: Forwarded to the inner Embedding.
    :param name: Layer name.
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        bag_size: int = 6,
        embeddings_initializer: str = "uniform",
        name: str = None,
        **kwargs,
    ):
        # DECISION plan_2026-05-17_413eae7d/D-001: construction-time wiring.
        # User instantiates TSTEmbedding in the model factory; apply_tst()
        # does NOT mutate a built model.
        super().__init__(name=name, **kwargs)
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size!r}.")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim!r}.")
        if bag_size < 1:
            raise ValueError(f"bag_size must be >= 1, got {bag_size!r}.")
        self.vocab_size = int(vocab_size)
        self.output_dim = int(output_dim)
        self.bag_size = int(bag_size)
        self.embeddings_initializer = embeddings_initializer
        self._inner: keras.layers.Embedding = None  # built lazily

    def build(self, input_shape):
        # Build the inner Embedding manually (LESSONS L-pattern: explicit
        # child.build avoids fragile lazy-build behavior under tf.function).
        self._inner = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.output_dim,
            embeddings_initializer=self.embeddings_initializer,
            name="inner_embedding",
        )
        # Embedding builds on any int shape — pass (None, None) for generic
        # 2D and rely on call() to handle rank-3 inputs by lookup.
        self._inner.build((None, None))
        super().build(input_shape)

    @property
    def embeddings(self) -> keras.Variable:
        """Tied-LM-head passthrough: returns the inner Embedding's variable.

        Models like ``CliffordNetLM`` and ``NAM`` re-use this variable via
        ``matmul(x, transpose(self.token_embedding.embeddings))`` to tie the
        LM head to the input embedding. Preserving this attribute is the
        load-bearing reason ``TSTEmbedding`` is a wrapper rather than a
        subclass of ``Embedding``.
        """
        if self._inner is None:
            # Force-build with a generic shape so ``layer.embeddings`` works
            # before the first call (used by tied heads at model build time).
            self.build((None, None))
        return self._inner.embeddings

    def call(self, inputs):
        # DECISION plan_2026-05-17_413eae7d/D-002: dispatch on static input
        # rank, not on a phase flag. Avoids LESSONS L23 (keras.ops.cond
        # traces both branches under tf.function on TF backend).
        #
        # Robust rank detection — keras.ops.ndim returns -1 for tensors with
        # unknown static rank (which happens when Keras strips dataset
        # element_specs inside model.fit). When we get -1, we LOOK UP first
        # (works on any rank) and dispatch on the result's rank.
        if self.bag_size == 1:
            return self._inner(inputs)
        rank = ops.ndim(inputs)
        if rank == 2:
            return self._inner(inputs)
        if rank == 3:
            static_s = inputs.shape[-1]
            if static_s is not None and static_s != self.bag_size:
                raise ValueError(
                    f"TSTEmbedding rank-3 input last-axis ({static_s}) must "
                    f"equal bag_size ({self.bag_size})."
                )
            emb = self._inner(inputs)
            return ops.mean(emb, axis=-2)
        if rank == -1:
            # Unknown rank — defer to runtime. Look up then dispatch on
            # the result's static rank (inner Embedding may preserve rank).
            emb = self._inner(inputs)
            emb_rank = ops.ndim(emb)
            if emb_rank == 4:
                # rank-3 input (B, N_lat, s) → lookup → (B, N_lat, s, d).
                return ops.mean(emb, axis=-2)
            if emb_rank == 3:
                # rank-2 input (B, N) → lookup → (B, N, d).
                return emb
            raise ValueError(
                f"TSTEmbedding: unknown-rank input produced lookup of "
                f"unexpected rank {emb_rank} (shape={emb.shape})."
            )
        raise ValueError(
            f"TSTEmbedding expects rank-2 or rank-3 input, got rank={rank} "
            f"(shape={inputs.shape})."
        )

    def compute_output_shape(self, input_shape):
        rank = len(input_shape)
        if rank == 2 or self.bag_size == 1:
            return tuple(input_shape) + (self.output_dim,)
        if rank == 3:
            # Drop the bag axis.
            return tuple(input_shape[:-1]) + (self.output_dim,)
        raise ValueError(
            f"TSTEmbedding expects rank-2 or rank-3 input shape, got {input_shape!r}."
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "output_dim": self.output_dim,
            "bag_size": self.bag_size,
            "embeddings_initializer": self.embeddings_initializer,
        })
        return config


# ---------------------------------------------------------------------
# TSTCausalLMLoss — rank-dispatched CE: vanilla NTP (rank-2) or sum-of-CE (rank-3)
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TSTCausalLMLoss(keras.losses.Loss):
    """Drop-in replacement for ``MaskedCausalLMLoss`` with rank-3 bagged path.

    Dispatches on ``y_true`` rank:

    - **rank-2** (``y_true`` shape ``(B, N)``): canary path. Bit-equivalent
      (atol 1e-6) to ``MaskedCausalLMLoss`` — masked sparse CE with the same
      ``ignore_index`` and ``label_smoothing`` semantics.
    - **rank-3** (``y_true`` shape ``(B, N_lat, s)``): bagged path. Sum-of-CE
      implementation per DECISION plan_2026-05-17_413eae7d/D-003: for each
      ``j ∈ [0, bag_size)`` compute ``CE(logits, y_true[..., j])`` and
      accumulate with weight ``w_j``. Mask: a latent position is "real" iff
      at least one bag-target at that position is non-ignored.

    ``y_pred`` may be a plain logits tensor of shape ``(B, N, V)`` /
    ``(B, N_lat, V)`` OR a dict ``{"logits": ...}`` (the latter is unwrapped
    defensively to match ``prepare_dict_keyed_compile``).

    :param bag_size: TST bag size ``s``. Determines the loop length on the
        rank-3 path. Has no effect on the rank-2 path.
    :param within_bag_weighting: ``"uniform"`` (``w_j = 1/s``) or
        ``"power_law"`` (``w_j ∝ (j+1)^(-alpha)``, normalised).
    :param within_bag_alpha: Exponent for the power-law scheme.
    :param label_smoothing: Optional smoothing ``α ∈ [0, 1)`` applied per
        bag-term on the rank-3 path and per token on the rank-2 path.
    :param ignore_index: Label value treated as masked. Default ``-1``.
    :param from_logits: Whether ``y_pred`` is logits (True) or probabilities.
    :param name: Loss instance name.
    """

    def __init__(
        self,
        bag_size: int = 6,
        within_bag_weighting: Literal["uniform", "power_law"] = "uniform",
        within_bag_alpha: float = 0.6,
        label_smoothing: float = 0.0,
        ignore_index: int = -1,
        from_logits: bool = True,
        name: str = "tst_causal_lm_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if bag_size < 1:
            raise ValueError(f"bag_size must be >= 1, got {bag_size!r}.")
        if within_bag_weighting not in ("uniform", "power_law"):
            raise ValueError(
                f"within_bag_weighting must be 'uniform' or 'power_law', "
                f"got {within_bag_weighting!r}."
            )
        if within_bag_alpha <= 0.0:
            raise ValueError(
                f"within_bag_alpha must be > 0, got {within_bag_alpha!r}."
            )
        self.bag_size = int(bag_size)
        self.within_bag_weighting = within_bag_weighting
        self.within_bag_alpha = float(within_bag_alpha)
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)
        self.from_logits = bool(from_logits)

        # Precompute within-bag weights (sum to 1).
        if within_bag_weighting == "uniform":
            w = np.ones((self.bag_size,), dtype=np.float64) / float(self.bag_size)
        else:
            w = np.arange(1, self.bag_size + 1, dtype=np.float64) ** (
                -self.within_bag_alpha
            )
            w = w / w.sum()
        self._w_j = w.astype(np.float32)

        self._base_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits,
            reduction="none",
        )

    # -- helpers ---------------------------------------------------------

    def _per_token_smoothing(self, y_pred):
        """Compute the smoothing-only per-token loss term (rank-2 path semantics)."""
        if self.from_logits:
            log_probs = y_pred - ops.logsumexp(y_pred, axis=-1, keepdims=True)
            return -ops.mean(log_probs, axis=-1)
        return -ops.mean(ops.log(y_pred + 1e-8), axis=-1)

    def _rank2_call(self, y_true, y_pred):
        """Vanilla masked CE — bit-equivalent to MaskedCausalLMLoss."""
        mask = ops.cast(y_true != self.ignore_index, "float32")
        safe_labels = ops.maximum(y_true, 0)
        per_token = self._base_ce(safe_labels, y_pred)
        if self.label_smoothing > 0.0:
            smooth = self._per_token_smoothing(y_pred)
            per_token = (
                (1.0 - self.label_smoothing) * per_token
                + self.label_smoothing * smooth
            )
        numerator = ops.sum(per_token * mask)
        denominator = ops.sum(mask) + 1e-8
        return numerator / denominator

    def _rank3_call(self, y_true, y_pred):
        # DECISION plan_2026-05-17_413eae7d/D-003: sum-of-CE via Python `for
        # j in range(s)` loop. Paper Appendix B identity; s ∈ [4,16] so the
        # loop unrolls at trace time with negligible perf cost.
        # Mask per latent position: "real" if at least one bag-target is real.
        real_per_j = ops.cast(y_true != self.ignore_index, "float32")
        mask = ops.cast(ops.sum(real_per_j, axis=-1) > 0, "float32")  # (B, N_lat)

        # Accumulate weighted per-position CE.
        loss = ops.zeros_like(mask)
        for j in range(self.bag_size):
            targets_j = y_true[..., j]
            mask_j = ops.cast(targets_j != self.ignore_index, "float32")
            safe_j = ops.maximum(targets_j, 0)
            ce_j = self._base_ce(safe_j, y_pred)  # (B, N_lat)
            if self.label_smoothing > 0.0:
                smooth = self._per_token_smoothing(y_pred)
                ce_j = (
                    (1.0 - self.label_smoothing) * ce_j
                    + self.label_smoothing * smooth
                )
            loss = loss + float(self._w_j[j]) * ce_j * mask_j

        numerator = ops.sum(loss)
        denominator = ops.sum(mask) + 1e-8
        return numerator / denominator

    # -- main ------------------------------------------------------------

    def call(self, y_true, y_pred):
        # Defensive unwrap for dict-output models (SYSTEM: prepare_dict_keyed_compile).
        if isinstance(y_pred, dict):
            y_pred = y_pred["logits"]

        rank = ops.ndim(y_true)
        if rank == 2:
            return self._rank2_call(y_true, y_pred)
        if rank == 3:
            static_s = y_true.shape[-1]
            if static_s is not None and static_s != self.bag_size:
                raise ValueError(
                    f"TSTCausalLMLoss rank-3 y_true last-axis ({static_s}) "
                    f"must equal bag_size ({self.bag_size})."
                )
            return self._rank3_call(y_true, y_pred)
        if rank == -1:
            # Unknown static rank — dispatch on y_pred's rank instead. y_pred
            # is always (B, N or N_lat, V); y_true rank matches model output
            # mode. If y_pred has the same time-axis length as y_true would
            # have in rank-2, we use rank-2; otherwise rank-3 with bag_size.
            # Practical heuristic: rank-2 ⇔ y_pred rank-3 with y_true having
            # same first-2 dims; rank-3 y_true has an extra bag axis.
            # Use the last static dim of y_true if available.
            last = y_true.shape[-1]
            if last is not None and int(last) == self.bag_size and self.bag_size > 1:
                return self._rank3_call(y_true, y_pred)
            return self._rank2_call(y_true, y_pred)
        raise ValueError(
            f"TSTCausalLMLoss expects rank-2 or rank-3 y_true, got rank={rank} "
            f"(shape={y_true.shape})."
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "bag_size": self.bag_size,
            "within_bag_weighting": self.within_bag_weighting,
            "within_bag_alpha": self.within_bag_alpha,
            "label_smoothing": self.label_smoothing,
            "ignore_index": self.ignore_index,
            "from_logits": self.from_logits,
        })
        return config


# ---------------------------------------------------------------------
# TSTPhaseCallback — advisory phase-boundary logger + global-step counter
# ---------------------------------------------------------------------


class TSTPhaseCallback(keras.callbacks.Callback):
    """Logs the Phase 1 → Phase 2 boundary and bumps ``state.global_step``.

    **Role after D-007.** The phase boundary is now controlled by the user via
    two separate ``model.fit(...)`` calls (one per phase transform). This
    callback no longer governs *which dataset shape is fed* — it only:

    1. Bumps ``state.global_step`` once per train batch (observability).
    2. Flips ``state.phase_active`` to ``False`` exactly once when
       ``state.global_step >= flip_step``, where ``flip_step =
       floor(total_steps * phase1_step_ratio)``. The flip is advisory:
       neither dataset transform reads ``phase_active``. The flag exists
       so user code (custom metrics, loggers, learning-rate schedules) can
       hook into the transition. Documented in the module banner.

    The callback is safe to attach to BOTH phase fits — the flip-once guard
    (``_already_flipped``) is idempotent across multiple ``.fit(...)`` calls.

    :param state: The ``TSTState`` to mutate.
    :param total_steps: Total number of training steps across both phases.
    :param phase1_step_ratio: Fraction of ``total_steps`` spent in Phase 1.
    """

    def __init__(
        self,
        state: "TSTState",
        total_steps: int,
        phase1_step_ratio: float,
    ) -> None:
        super().__init__()
        if state is None:
            raise ValueError("TSTPhaseCallback requires a TSTState (got None).")
        if total_steps < 1:
            raise ValueError(
                f"total_steps must be >= 1, got {total_steps!r}."
            )
        if not (0.0 <= phase1_step_ratio <= 1.0):
            raise ValueError(
                "phase1_step_ratio must be in [0.0, 1.0], got "
                f"{phase1_step_ratio!r}."
            )
        self._state = state
        self._total_steps = int(total_steps)
        self._phase1_step_ratio = float(phase1_step_ratio)
        self._flip_step = int(self._total_steps * self._phase1_step_ratio)
        self._already_flipped = False
        logger.info(
            "TSTPhaseCallback: total_steps=%d, phase1_step_ratio=%.3f → "
            "flip_step=%d (advisory boundary; phase_active is observational).",
            self._total_steps,
            self._phase1_step_ratio,
            self._flip_step,
        )

    @property
    def flip_step(self) -> int:
        """Public read of the computed flip step (for tests / introspection)."""
        return self._flip_step

    @property
    def already_flipped(self) -> bool:
        """Whether the one-shot flip has fired."""
        return self._already_flipped

    def on_train_batch_end(self, batch, logs=None):  # noqa: D401, ARG002
        # Increment first so global_step counts batches actually completed.
        self._state.global_step.assign_add(1)
        if self._already_flipped:
            return
        if int(self._state.global_step.numpy()) >= self._flip_step:
            if bool(self._state.phase_active.numpy()):
                self._state.phase_active.assign(False)
            self._already_flipped = True
            logger.info(
                "TSTPhaseCallback: phase boundary reached at global_step=%d "
                "(flip_step=%d). phase_active flipped to False (advisory).",
                int(self._state.global_step.numpy()),
                self._flip_step,
            )


# ---------------------------------------------------------------------
# Dataset transforms — TWO named callables, one per phase (D-007).
#
# Two named callables, one per phase: each transform has a single
# statically-determined output rank — no tf.cond in the data graph,
# no shape-inference hazard, no tf.Variable-read race between tf.data
# and the rest of the training loop. The phase boundary is explicit in
# the user's training script (two .fit() calls with `initial_epoch=` for
# resume).
# ---------------------------------------------------------------------


def _validate_phase1_divisibility(n: int, bag_size: int) -> None:
    # DECISION plan_2026-05-17_413eae7d/D-004 + R1: reuse upstream
    # (input_ids, labels) pair; reshape labels to (B, N/s, s) for Phase 1.
    # Constraint N % s == 0 raised loudly here.
    if bag_size < 1:
        raise ValueError(f"bag_size must be >= 1, got {bag_size!r}.")
    if n % bag_size != 0:
        raise ValueError(
            f"tst_phase1_transform: label length N={n} is not divisible by "
            f"bag_size={bag_size}. TST requires N % bag_size == 0 so that "
            f"each latent position covers exactly s tokens (see D-004). "
            f"Adjust upstream chunk_length so that chunk_length - 1 is a "
            f"multiple of bag_size."
        )


def tst_phase1_transform(
    ds: tf.data.Dataset,
    bag_size: int,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """Phase 1 transform: bagged inputs + bagged labels (rank-3 / rank-3).

    Input contract: ``ds`` yields ``(input_ids, labels)`` with shape
    ``(B, N)`` int (the standard output of ``preprocess_clm_packed_dataset``).

    Output: ``(input_ids_lat, labels_lat)`` where both are
    ``(B, N/bag_size, bag_size)`` int. Validates ``N % bag_size == 0`` at
    build time (user rule R1; see D-004).

    No ``tf.Variable`` reads inside the graph — the output rank is statically
    determined by this function's arguments (D-007).

    :param ds: Input dataset yielding ``(input_ids, labels)``.
    :param bag_size: TST bag size ``s``.
    :param drop_remainder: Reserved for future batch-padding behaviour; the
        current implementation requires that ``ds`` be pre-batched and that
        ``N % bag_size == 0``. Kept on the signature for API parity with
        :func:`tst_phase2_transform`.
    :raises ValueError: if ``bag_size < 1``, the dataset element-spec does
        not look like ``(input_ids, labels)`` of rank 2, or the static
        last-axis is known and not divisible by ``bag_size``.
    """
    del drop_remainder  # not used in the reshape path

    if bag_size < 1:
        raise ValueError(f"bag_size must be >= 1, got {bag_size!r}.")

    # Build-time static shape check (user rule R1 + D-004).
    spec = ds.element_spec
    if not (isinstance(spec, tuple) and len(spec) == 2):
        raise ValueError(
            "tst_phase1_transform expects a dataset yielding "
            "(input_ids, labels); got element_spec="
            f"{spec!r}."
        )
    inp_spec, lab_spec = spec
    static_n = None
    for s in (inp_spec, lab_spec):
        last = s.shape[-1] if s.shape.rank is not None else None
        if last is not None:
            if static_n is None:
                static_n = int(last)
            elif int(last) != static_n:
                raise ValueError(
                    "tst_phase1_transform: input_ids and labels must share "
                    f"the same last-axis length; got {static_n} vs {int(last)}."
                )
    if static_n is not None:
        _validate_phase1_divisibility(static_n, bag_size)

    # Compute static N_lat when known so the output element_spec carries a
    # known rank (Keras model.fit downstream needs static rank on inputs).
    n_lat_static = (static_n // bag_size) if static_n is not None else None

    def _reshape(input_ids, labels):
        # Dynamic-shape reshape; we then set_shape so the static rank is
        # known to downstream consumers (Keras model.fit / TSTEmbedding.call).
        dyn_n = tf.shape(input_ids)[-1]
        new_n_lat = dyn_n // bag_size
        leading_inp = tf.shape(input_ids)[:-1]
        leading_lab = tf.shape(labels)[:-1]
        inputs_lat = tf.reshape(
            input_ids, tf.concat([leading_inp, [new_n_lat, bag_size]], axis=0)
        )
        labels_lat = tf.reshape(
            labels, tf.concat([leading_lab, [new_n_lat, bag_size]], axis=0)
        )
        # Set static shapes so the output rank is known (D-007 canary).
        inp_rank = input_ids.shape.rank
        lab_rank = labels.shape.rank
        if inp_rank is not None:
            leading_static = input_ids.shape[:-1].as_list()
            inputs_lat.set_shape(leading_static + [n_lat_static, bag_size])
        if lab_rank is not None:
            leading_static = labels.shape[:-1].as_list()
            labels_lat.set_shape(leading_static + [n_lat_static, bag_size])
        return inputs_lat, labels_lat

    return ds.map(_reshape, num_parallel_calls=tf.data.AUTOTUNE)


def tst_phase2_transform(
    ds: tf.data.Dataset,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """Phase 2 transform: pass-through NTP shapes (rank-2 / rank-2).

    Input contract: identical to :func:`tst_phase1_transform`. Output is the
    dataset unchanged — ``TSTEmbedding`` takes the plain-lookup path on
    rank-2 inputs, and ``TSTCausalLMLoss`` takes the NTP path on rank-2
    labels. The transform is provided as a separate callable (rather than
    "use the raw dataset") so users can compose Phase 1 / Phase 2 the same
    way in their training scripts (D-007 API symmetry).

    :param ds: Input dataset yielding ``(input_ids, labels)``.
    :param drop_remainder: Currently a no-op; kept for API parity with
        :func:`tst_phase1_transform`.
    """
    del drop_remainder
    spec = ds.element_spec
    if not (isinstance(spec, tuple) and len(spec) == 2):
        raise ValueError(
            "tst_phase2_transform expects a dataset yielding "
            "(input_ids, labels); got element_spec="
            f"{spec!r}."
        )
    return ds


# ---------------------------------------------------------------------
# apply_tst — bundle state + callbacks + the two phase transforms
# ---------------------------------------------------------------------


def apply_tst(
    config: "TSTConfig",
    total_steps: int,
) -> Tuple[
    "TSTState",
    List[keras.callbacks.Callback],
    Callable[[tf.data.Dataset], tf.data.Dataset],
    Callable[[tf.data.Dataset], tf.data.Dataset],
]:
    """Bundle TST plumbing for the canonical two-``model.fit(...)`` pattern.

    DECISION plan_2026-05-17_413eae7d/D-007: returns FOUR values, not three.
    The phase boundary is the user's responsibility — they run two
    ``model.fit(...)`` calls, one with ``phase1_transform(raw_ds)`` and one
    with ``phase2_transform(raw_ds)``. This explicit pattern is the resolution
    of Falsification C (see D-006 in decisions.md for the trace).

    :param config: A :class:`TSTConfig` with ``bag_size`` and
        ``phase1_step_ratio``.
    :param total_steps: Total training steps across both phases. Used to
        size ``TSTPhaseCallback._flip_step`` for observability logging.

    :returns: 4-tuple ``(state, callbacks, phase1_transform, phase2_transform)``:

        * **state** (:class:`TSTState`): Holds ``phase_active`` and
          ``global_step`` ``tf.Variable``s. Advisory only (D-007).
        * **callbacks** (``list[keras.callbacks.Callback]``): Contains a
          single :class:`TSTPhaseCallback`. Attach to BOTH ``.fit(...)``
          calls; the flip-once guard makes it idempotent.
        * **phase1_transform** (``Callable[[tf.data.Dataset], tf.data.Dataset]``):
          Bagged transform — yields rank-3 inputs and rank-3 labels.
        * **phase2_transform** (``Callable[[tf.data.Dataset], tf.data.Dataset]``):
          Pass-through transform — yields rank-2 inputs and rank-2 labels.

    Canonical usage::

        state, callbacks, phase1_fn, phase2_fn = apply_tst(
            TSTConfig(bag_size=6, phase1_step_ratio=0.25),
            total_steps=100_000,
        )
        # Phase 1
        model.fit(
            phase1_fn(raw_ds),
            epochs=phase1_epochs,
            callbacks=user_cbs + callbacks,
        )
        # Phase 2 — initial_epoch= for clean resume / TensorBoard alignment
        model.fit(
            phase2_fn(raw_ds),
            epochs=total_epochs,
            initial_epoch=phase1_epochs,
            callbacks=user_cbs + callbacks,
        )
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps!r}.")

    state = TSTState(bag_size=config.bag_size, phase_active_init=True)
    callback = TSTPhaseCallback(
        state=state,
        total_steps=total_steps,
        phase1_step_ratio=config.phase1_step_ratio,
    )
    phase1_fn = functools.partial(tst_phase1_transform, bag_size=config.bag_size)
    return state, [callback], phase1_fn, tst_phase2_transform
