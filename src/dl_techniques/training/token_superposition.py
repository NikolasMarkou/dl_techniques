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

The phase flip is a one-shot mutation of ``state.phase_active`` (and only that
flag — invariant 6). Crucially: ``TSTEmbedding.call`` dispatches on **input
rank**, not on the phase flag (decision D-002 below). The dataset transform
controls which rank reaches the layer, making the dataset the single source of
truth for "what phase are we in".

bag_size=1 canary
-----------------
The load-bearing correctness invariant: with ``bag_size=1``,
``TSTEmbedding`` and ``TSTCausalLMLoss`` are bit-equivalent (atol 1e-6) to a
plain ``keras.layers.Embedding`` + ``MaskedCausalLMLoss`` pipeline. This is
verified by unit tests (``inv1_canary_embedding`` and ``inv2_canary_loss``).
If those fail, **do not proceed** — every other claim depends on them.

What the tests in this module verify
------------------------------------
Validated by ``tests/test_training/test_token_superposition.py``:

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
   first attempt uses one ``tf.data.Dataset`` with ``tf.cond`` on
   ``state.phase_active``. If trace fails to unify the rank-3 (Phase 1)
   and rank-2 (Phase 2) branches (Falsification C), the user is notified
   to switch to the two-dataset fallback explicitly — no silent fallback.
   Anchored at ``tst_dataset_transform``.

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
    """Mutable training-loop state shared between callback and dataset.

    Holds two ``tf.Variable``s (``phase_active``, ``global_step``) and a
    constant ``bag_size``. The variables are deliberately non-trainable and
    untracked by Keras — they are training-loop scaffolding, not model
    parameters (invariant 6: the phase flip must not touch any trainable
    weight or optimizer slot).

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
        rank = len(inputs.shape)
        if rank == 2 or self.bag_size == 1:
            # Plain lookup (Phase 2 / canary path).
            return self._inner(inputs)
        if rank == 3:
            # Bagged lookup (Phase 1 path): inputs is (B, N_lat, s).
            static_s = inputs.shape[-1]
            if static_s is not None and static_s != self.bag_size:
                raise ValueError(
                    f"TSTEmbedding rank-3 input last-axis ({static_s}) must "
                    f"equal bag_size ({self.bag_size})."
                )
            emb = self._inner(inputs)  # (B, N_lat, s, d)
            return ops.mean(emb, axis=-2)  # (B, N_lat, d)
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
# Step 4..6 will populate this module further.
#   TSTCausalLMLoss, TSTPhaseCallback, tst_dataset_transform, apply_tst
# ---------------------------------------------------------------------
