"""An exponential moving average of the DiT's trainable weights, as a callback.

Upstream keeps a second copy of the whole model and samples from it, never from
the live weights::

    ema = deepcopy(model); requires_grad(ema, False)   # decay 0.9999
    ...
    opt.step()
    update_ema(ema, model)      # ema = decay*ema + (1-decay)*param

(``reference/train_and_sample_excerpts.py:16,26,31-37``.) That is the whole
mechanism, and this module is its Keras spelling: :class:`WeightEMACallback`
holds NumPy shadows of the model's trainable variables, advances them once per
training batch, and can swap them into the model for sampling or evaluation and
back out again.

What it looks like
------------------

.. code-block:: text

    per training batch                      for sampling / evaluation
    ------------------                      -------------------------

      live weights  w_t                       live weights  w
      [B, ...] batch      │                        │
              │           │                        │  apply_to(model)
              ▼           ▼                        ▼
    ┌───────────────────────────┐          ┌──────────────────┐
    │  on_train_batch_end       │          │  backup <- w     │
    │                           │          │  model  <- s     │
    │  s ⊕= decay*s             │          └────────┬─────────┘
    │       + (1-decay)*w_t     │                   │  sample / evaluate
    └─────────────┬─────────────┘                   ▼
                  │                        ┌──────────────────┐
                  ▼                        │  restore(model)  │
           shadows  s (numpy)              │  model <- backup │
                                           └──────────────────┘

The shadows never enter the graph: they are plain ``numpy`` arrays, read out of
the variables with ``keras.ops.convert_to_numpy`` after the optimizer has already
applied its update, and written back with ``Variable.assign``. Nothing here
touches ``train_step``.

Design notes
------------

* **Which variables are shadowed: the TRAINABLE ones only.** See
  :data:`SHADOWED_VARIABLE_SET` and the anchor on
  :meth:`WeightEMACallback.on_train_begin`. This diverges from upstream, which
  averages everything ``named_parameters()`` returns -- including DiT's frozen
  ``pos_embed`` table, which upstream's own TODO flags. The divergence is
  deliberate and measured, not an oversight.
* **The decay is a constant**, exactly as upstream. No warmup and no ramp: this
  package deliberately does not reuse
  ``dl_techniques.models.vision.depth_anything.teacher_ema``'s
  ``cosine_ema_schedule`` / ``linear_ema_schedule``, because a ramp is a
  Mean-Teacher/DINO device for a student->teacher pair whose teacher must track
  early updates quickly, and adding one here would silently change the published
  DiT recipe while looking like an improvement.
* **No logging in the per-batch hook.** It runs every step; a log line there is
  a training-speed regression and a log-file flood.

References:
    - Peebles, W., & Xie, S. (2022). Scalable Diffusion Models with
      Transformers. https://arxiv.org/abs/2212.09748
    - Upstream ``update_ema``: ``chuanyangjin/fast-DiT``, ``train.py``,
      transcribed at ``reference/train_and_sample_excerpts.py:31-37``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

import keras

from dl_techniques.utils.logger import logger

__all__ = [
    "DEFAULT_EMA_DECAY",
    "SHADOWED_VARIABLE_SET",
    "WeightEMACallback",
]

#: Upstream's decay (``reference/train_and_sample_excerpts.py:16,32``).
DEFAULT_EMA_DECAY: float = 0.9999

#: Which of the model's variables this callback shadows. Named so a test can
#: assert the choice instead of re-deriving it, and so the divergence from
#: upstream's ``named_parameters()`` has one written home.
SHADOWED_VARIABLE_SET: str = "trainable_weights"


class WeightEMACallback(keras.callbacks.Callback):
    r"""Maintain an exponential moving average of the model's trainable weights.

    Snapshots ``model.trainable_weights`` into NumPy at ``on_train_begin`` and
    advances each shadow at ``on_train_batch_end`` with

    .. math::

        s \leftarrow \mathrm{decay} \cdot s + (1 - \mathrm{decay}) \cdot w

    which is upstream's ``update_ema`` verbatim. :meth:`apply_to` installs the
    shadows into a model for sampling and :meth:`restore` puts the live weights
    back; :meth:`applied_to` is the two of them under ``try/finally``.

    .. code-block:: text

        on_train_begin           on_train_batch_end          apply_to / restore
        ┌───────────────┐        ┌────────────────────┐      ┌────────────────┐
        │ w  [B, ...]-  │        │  w := optimizer(w) │      │ backup := w    │
        │ trained model │───────▶│  s ⊕= decay*s      │─────▶│ w      := s    │
        │ s := w  (copy)│        │       +(1-d)*w     │      │  ...sample...  │
        └───────────────┘        └────────────────────┘      │ w      := bkp  │
                                                             └────────────────┘

    The initial snapshot reproduces upstream's ``update_ema(ema, model,
    decay=0)`` initialisation call: the EMA starts EQUAL to the model, so an
    ``apply_to`` before any batch is an exact no-op on the weights.

    :param decay: EMA decay in ``[0, 1]``. ``0.0`` makes the shadow track the
        live weights exactly; ``1.0`` freezes it at the initial snapshot.
    :type decay: float
    :raises ValueError: If ``decay`` is outside ``[0, 1]`` or is not finite.

    Example:
        >>> ema = WeightEMACallback(decay=0.9999)
        >>> model.fit(dataset, epochs=1, callbacks=[ema])   # doctest: +SKIP
        >>> with ema.applied_to(model):                     # doctest: +SKIP
        ...     samples = diffusion.p_sample_loop(model, shape, seed=0)
    """

    def __init__(self, decay: float = DEFAULT_EMA_DECAY) -> None:
        super().__init__()
        decay = float(decay)
        if not np.isfinite(decay) or not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must lie in [0, 1], got {decay}")

        self.decay: float = decay
        self._variables: List[Any] = []
        self._shadows: List[np.ndarray] = []
        self._paths: List[str] = []
        self._backup: Optional[Dict[str, np.ndarray]] = None
        self._updates: int = 0

        logger.info(
            "WeightEMACallback: decay=%.6f over %s", self.decay,
            SHADOWED_VARIABLE_SET,
        )

    # -----------------------------------------------------------------
    # state
    # -----------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        """``True`` once the shadows have been snapshotted."""
        return bool(self._paths)

    @property
    def updates(self) -> int:
        """Number of ``on_train_batch_end`` updates applied so far."""
        return self._updates

    @property
    def applied(self) -> bool:
        """``True`` while the shadows are installed in a model."""
        return self._backup is not None

    def shadow_values(self) -> Dict[str, np.ndarray]:
        """Return a COPY of the shadows, keyed by variable path.

        Interface contract: pure; the returned arrays are copies, so a caller
        cannot mutate the callback's state through them.

        :return: ``{variable.path: shadow}``. Empty before ``on_train_begin``.
        :rtype: Dict[str, np.ndarray]
        """
        return {
            path: shadow.copy()
            for path, shadow in zip(self._paths, self._shadows)
        }

    # -----------------------------------------------------------------
    # the hooks
    # -----------------------------------------------------------------

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Snapshot the model's trainable variables into NumPy shadows.

        Idempotent across repeated ``fit()`` calls: a second call finds the
        shadows already present and leaves them alone, so an interrupted-and-
        resumed run continues its average instead of restarting it.

        If the model is not built yet the snapshot is DEFERRED to the first
        ``on_train_batch_end`` -- see :meth:`_snapshot`.

        :param logs: Keras hook argument, unused.
        :type logs: Optional[Dict[str, Any]]
        :raises RuntimeError: If no model is attached.
        """
        if self.initialized:
            return
        if self.model is None:
            raise RuntimeError(
                "WeightEMACallback has no model; it must be passed to "
                "model.fit(callbacks=[...])"
            )
        if not self.model.trainable_weights:
            # MEASURED, and the whole reason this method can defer: Keras 3
            # calls `on_train_begin` BEFORE the first batch, so a lazily-built
            # model is still UNBUILT here and `trainable_weights` is `[]`. An
            # implementation that snapshots that empty list is a permanent
            # silent no-op -- it averages nothing, forever, with no exception.
            # This is the one place a log line is allowed (once per fit, not
            # per batch). Pinned by `TestTheSnapshotSurvivesALazilyBuiltModel`.
            logger.info(
                "WeightEMACallback: model not built at on_train_begin; "
                "deferring the shadow snapshot to the first batch"
            )
            return
        self._snapshot()

    def _snapshot(self) -> None:
        """Copy the model's trainable variables into fresh NumPy shadows.

        :raises RuntimeError: If the model has no trainable weights (an EMA
            over nothing is a silent no-op, so it is an error, not a warning).
        """
        # DECISION plan-2026-09-02T170923-1285ed83/D-023
        # Shadow `trainable_weights`, NOT `weights`. Upstream averages every
        # entry of `named_parameters()`, which in PyTorch INCLUDES DiT's frozen
        # `pos_embed` (registered as a Parameter with requires_grad=False), and
        # upstream leaves a TODO about it. Do NOT "fix" this by widening to
        # `self.model.weights` on the theory that the EMA of a constant is that
        # constant: that identity holds in exact arithmetic only. MEASURED on
        # this port's own float32 sin-cos table at decay=0.9999, the shadow of a
        # never-changing variable drifts 1.19e-05 away from it after 200
        # updates, because `d*c + (1-d)*c` does not round back to `c`. Widening
        # would therefore inject rounding error into an exactly-computed table
        # for no benefit. See decisions.md D-023; pinned by
        # tests/test_train/test_dit/test_the_ema_callback_actually_moves.py::
        # TestTheShadowedSetIsTrainableOnly.
        variables = list(self.model.trainable_weights)
        if not variables:
            raise RuntimeError(
                "WeightEMACallback: the model has no trainable weights, so the "
                "average would be over nothing"
            )

        self._variables = variables
        self._paths = [v.path for v in variables]
        # Upstream initialises with `update_ema(ema, model, decay=0)`, i.e. the
        # EMA starts EQUAL to the model. A copy is that call's exact effect.
        self._shadows = [
            np.array(keras.ops.convert_to_numpy(v), copy=True)
            for v in variables
        ]
        self._updates = 0
        logger.info(
            "WeightEMACallback: shadowing %d trainable variables",
            len(self._variables),
        )

    def on_train_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Advance every shadow by one EMA step.

        Emits NO log line and allocates no Keras op: this runs on every
        training batch. (The deferred first-batch snapshot logs once; every
        subsequent call is silent.)

        :param batch: Keras hook argument, unused.
        :type batch: int
        :param logs: Keras hook argument, unused.
        :type logs: Optional[Dict[str, Any]]
        """
        if not self.initialized:
            # The deferred snapshot (a lazily-built model): take it now, and do
            # NOT also update on the same batch -- the shadow would be a
            # convex combination of one array with itself.
            self._snapshot()
            return
        decay = self.decay
        one_minus = 1.0 - decay
        for index, variable in enumerate(self._variables):
            live = keras.ops.convert_to_numpy(variable)
            shadow = self._shadows[index]
            # DECISION plan-2026-09-02T170923-1285ed83/D-024
            # The literal two-term form, in the shadow's own dtype. Do NOT
            # "simplify" to the algebraically identical increment
            # `s += (1 - decay) * (live - s)`: it is better conditioned in the
            # middle of the range but LOSES both endpoints in floating point.
            # At decay=0 it computes `s + (live - s)`, which is not `live` when
            # the two differ in magnitude, so the "decay=0 tracks the weights
            # exactly" contract dies at atol=0. The literal form keeps both
            # endpoints EXACT: `0*s + 1*live == live` and `1*s + 0*live == s`.
            # See decisions.md D-024.
            self._shadows[index] = (
                decay * shadow + one_minus * live
            ).astype(shadow.dtype, copy=False)
        self._updates += 1

    # -----------------------------------------------------------------
    # swapping the shadows in and out
    # -----------------------------------------------------------------

    def apply_to(self, model: keras.Model) -> None:
        """Install the shadows into ``model``, stashing the live weights.

        Interface contract: pairs with :meth:`restore`, which is the ONLY way
        back. Every swapped-out value is kept, so ``restore`` is bit-exact.
        Prefer :meth:`applied_to`, which cannot leak the swap on an exception.

        :param model: The model to write into. Its trainable-variable paths
            must match the snapshotted set exactly, so a differently-configured
            model is rejected instead of silently half-updated.
        :type model: keras.Model
        :raises RuntimeError: If the shadows are not initialized, or if a
            previous :meth:`apply_to` has not been restored.
        :raises ValueError: If ``model``'s trainable-variable paths or shapes
            do not match the snapshot.
        """
        if not self.initialized:
            raise RuntimeError(
                "WeightEMACallback: no shadows yet; run on_train_begin (i.e. "
                "fit()) before apply_to()"
            )
        if self.applied:
            raise RuntimeError(
                "WeightEMACallback: shadows are already applied; call "
                "restore() before applying again"
            )

        targets = self._resolve(model)
        backup: Dict[str, np.ndarray] = {}
        for path, variable, shadow in zip(self._paths, targets, self._shadows):
            backup[path] = np.array(
                keras.ops.convert_to_numpy(variable), copy=True
            )
            variable.assign(shadow)
        self._backup = backup

    def restore(self, model: keras.Model) -> None:
        """Put the weights :meth:`apply_to` swapped out back, bit-exactly.

        :param model: The same model :meth:`apply_to` was given.
        :type model: keras.Model
        :raises RuntimeError: If nothing is currently applied.
        :raises ValueError: If ``model``'s trainable-variable paths or shapes
            do not match the snapshot.
        """
        if not self.applied:
            raise RuntimeError(
                "WeightEMACallback: nothing to restore; apply_to() was not "
                "called (or restore() already ran)"
            )
        backup = self._backup
        assert backup is not None  # narrowed by `self.applied`
        for path, variable in zip(self._paths, self._resolve(model)):
            variable.assign(backup[path])
        self._backup = None

    @contextlib.contextmanager
    def applied_to(self, model: keras.Model) -> Iterator[keras.Model]:
        """Context manager wrapping :meth:`apply_to` / :meth:`restore`.

        This is the recommended entry point. The bare pair is public because
        the plan's step text names ``apply_to``, but a sampling call that raises
        while the shadows are installed would otherwise leave the model holding
        EMA weights permanently -- a corruption with no shape, dtype or
        finiteness symptom, and one that a later ``save()`` would make
        permanent.

        :param model: The model to swap into for the duration of the block.
        :type model: keras.Model
        :yield: ``model``, with the EMA weights installed.
        :rtype: Iterator[keras.Model]
        """
        self.apply_to(model)
        try:
            yield model
        finally:
            self.restore(model)

    # -----------------------------------------------------------------

    def _resolve(self, model: keras.Model) -> List[Any]:
        """Match ``model``'s trainable variables onto the snapshotted paths.

        :param model: The model to resolve against.
        :type model: keras.Model
        :return: The variables, in snapshot order.
        :rtype: List[Any]
        :raises ValueError: On a path-set or shape mismatch.
        """
        by_path = {v.path: v for v in model.trainable_weights}
        missing = [path for path in self._paths if path not in by_path]
        if missing:
            raise ValueError(
                f"WeightEMACallback: {len(missing)} shadowed variable(s) are "
                f"absent from this model, first: {missing[0]!r}"
            )
        extra = sorted(set(by_path) - set(self._paths))
        if extra:
            raise ValueError(
                f"WeightEMACallback: this model has {len(extra)} trainable "
                f"variable(s) the snapshot does not cover, first: {extra[0]!r}"
            )
        resolved = [by_path[path] for path in self._paths]
        for path, variable, shadow in zip(self._paths, resolved, self._shadows):
            if tuple(variable.shape) != shadow.shape:
                raise ValueError(
                    f"WeightEMACallback: shape mismatch for {path!r}: model "
                    f"{tuple(variable.shape)} vs shadow {shadow.shape}"
                )
        return resolved
