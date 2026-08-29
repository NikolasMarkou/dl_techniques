"""
ColBERT training objectives: v1 pairwise softmax cross-entropy and v2 KL distillation.

This module ships the two loss functions that separate ColBERT v1 from ColBERT v2.
The *network* is identical between the two versions -- the official
``stanford-futuredata/ColBERT`` repository has a single ``colbert/modeling/colbert.py``
serving both -- so the recipe, and specifically the objective implemented here, is what
"v1" and "v2" actually name.

:mod:`ColBERTPairwiseSoftmaxLoss` (v1)
    Softmax cross-entropy over the ``nway`` MaxSim scores of one training tuple, with the
    positive candidate always at **index 0**. This mirrors the reference trainer, which
    reshapes the flat score vector to ``(batch, nway)`` and calls
    ``nn.CrossEntropyLoss()(scores, labels)`` with ``labels = torch.zeros(bsize)``.

:mod:`ColBERTDistillationLoss` (v2)
    Denoised supervision. A cross-encoder teacher scores the same ``nway`` candidates;
    the student is trained to reproduce the teacher's *distribution* over those
    candidates rather than the teacher's raw score values.

**Why v2 matches distributions rather than regressing raw scores.** The ColBERTv2 paper
gives the reason directly: ColBERT "produces scores (i.e., the sum of cosine
similarities) with a restricted scale, which may not align directly with the output
scores of the cross-encoder". A MaxSim score is a sum of ``query_maxlen`` cosine
similarities and therefore lives in ``[-query_maxlen, +query_maxlen]``; a cross-encoder
logit has no such bound and no shared zero point. Regressing one onto the other (an MSE
on raw scores) would ask the student to reproduce an arbitrary affine convention rather
than a ranking. Softmax-normalizing both sides first discards exactly that nuisance
scale -- softmax is invariant to an additive shift -- and leaves a comparison of the two
*rankings*, expressed as a KL divergence between the two candidate distributions.

Denoised supervision, concretely: instead of the binary "``d+`` is relevant, ``d-`` is
not" signal that v1's cross-entropy encodes -- which is noisy, because a sampled negative
is frequently a false negative -- v2 hands the student a graded, teacher-produced
relevance profile over all ``nway`` candidates. A hard negative the teacher rates highly
is no longer punished as though it were irrelevant.

Both objectives consume the *scores* produced by
:class:`dl_techniques.models.language.colbert.components.MaxSimScorer`; neither touches
the encoder, the projection, or the index-time residual codec.

References:
    - Khattab, O. and Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage
      Search via Contextualized Late Interaction over BERT." SIGIR 2020.
      https://arxiv.org/abs/2004.12832
    - Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C. and Zaharia, M. (2022).
      "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction."
      NAACL 2022. https://arxiv.org/abs/2112.01488
    - Reference trainer (both objectives, fetched 2026-08-25):
      https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/training/training.py
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _validate_nway(nway: int) -> int:
    """Validate the candidates-per-query group size.

    Interface contract (called by both loss constructors in this module):

    :param nway: Number of candidate documents scored per query in one training tuple.
    :returns: ``nway`` as a plain ``int``.
    :raises ValueError: If ``nway`` is not an integer, or is smaller than 2. A group of
        one candidate has a degenerate softmax (identically 1.0) and a zero gradient, so
        it is rejected rather than silently trained on.
    """
    if isinstance(nway, bool) or not isinstance(nway, (int,)):
        raise ValueError(f"nway must be an int, got {type(nway).__name__}: {nway!r}")
    if nway < 2:
        raise ValueError(
            f"nway must be >= 2 (a one-candidate group has an identically-1.0 softmax "
            f"and therefore no gradient), got {nway}"
        )
    return int(nway)


def _reshape_to_nway(
    scores: Any,
    nway: int,
    argument_name: str,
) -> keras.KerasTensor:
    """Reshape a score tensor to ``(batch, nway)``.

    Interface contract (called by both loss ``call`` implementations, for both the
    student and the teacher operand):

    :param scores: Either a flat score vector of shape ``(batch * nway,)`` (or the
        equivalent column ``(batch * nway, 1)``), or an already-grouped
        ``(batch, nway)`` tensor.
    :param nway: The group size to reshape to.
    :param argument_name: Name used in error messages, e.g. ``"y_pred"``.
    :returns: A tensor of shape ``(batch, nway)``.
    :raises ValueError: If the rank is neither 1 nor 2, if a rank-2 tensor's trailing
        axis is neither ``nway`` nor 1, or if a statically-known flat length is not
        divisible by ``nway``.
    """
    scores = keras.ops.convert_to_tensor(scores)
    shape = tuple(scores.shape)

    if len(shape) == 2 and shape[-1] == nway:
        return scores

    if len(shape) == 1 or (len(shape) == 2 and shape[-1] == 1):
        total = shape[0]
        if total is not None and int(total) % nway != 0:
            raise ValueError(
                f"{argument_name} has {int(total)} scores, which is not divisible by "
                f"nway={nway}. A ColBERT training batch is a flat concatenation of "
                f"equal-size candidate groups, so the length must be batch * nway."
            )
        return keras.ops.reshape(scores, (-1, nway))

    raise ValueError(
        f"{argument_name} must be a flat (batch * nway,) score vector or an already "
        f"grouped (batch, nway) tensor with nway={nway}; got shape {shape}."
    )


@register_dl_technique("dl_techniques.losses.colbert_loss")
class ColBERTPairwiseSoftmaxLoss(keras.losses.Loss):
    """ColBERT v1 objective: softmax cross-entropy over ``nway`` candidate scores.

    The reference trainer reshapes the flat MaxSim score vector to ``(batch, nway)`` and
    applies ``nn.CrossEntropyLoss()(scores, labels)`` with ``labels = zeros``. So the
    per-sample loss is::

        loss_b = logsumexp_k(s[b, k]) - s[b, 0]

    and the reported loss is its mean over the batch.

    .. warning::

        **The positive is positional, not labelled.** The positive candidate is always
        at index 0 of each ``nway`` group -- that is what ``labels = zeros`` means in the
        reference. ``y_true`` is therefore **not consulted by this loss at all**. It is
        accepted only because :meth:`keras.losses.Loss.__call__` requires the two-argument
        signature. This is stated here rather than left implicit precisely because a
        silently-ignored argument is otherwise indistinguishable from a wiring bug: a
        caller who passes real labels and sees the loss move would reasonably conclude the
        labels were used. They are not. Order your candidate tuples with the positive
        first, or this loss trains on the wrong target while looking healthy.

    .. note::

        ``y_true`` must still be a real tensor -- ``loss(None, scores)`` raises inside
        Keras' own ``__call__`` wrapper, which converts ``y_true`` before dispatching.
        Pass ``keras.ops.zeros_like(scores)`` (or any same-dtype placeholder) as the
        label argument.

    :param nway: Number of candidates per query in one training tuple, positive first.
        The reference default is 2 (a ``<q, d+, d->`` triple). Must be ``>= 2``.
    :param kwargs: Standard :class:`keras.losses.Loss` keyword arguments (``name``,
        ``reduction``, ``dtype``).
    :raises ValueError: If ``nway < 2``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.losses import ColBERTPairwiseSoftmaxLoss

        loss_fn = ColBERTPairwiseSoftmaxLoss(nway=2)
        scores = keras.ops.convert_to_tensor([3.0, 1.0, 0.5, 2.5])  # 2 triples
        loss_fn(keras.ops.zeros_like(scores), scores)
    """

    def __init__(self, nway: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.nway = _validate_nway(nway)

    def call(
        self,
        y_true: Optional[keras.KerasTensor],
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the per-sample cross-entropy of each ``nway`` group.

        :param y_true: **Unused.** See the class-level warning: the positive is index 0
            by position. Accepted for the Keras loss API contract only.
        :param y_pred: Student MaxSim scores, ``(batch * nway,)`` or ``(batch, nway)``.
        :returns: Per-sample losses of shape ``(batch,)``. The parent
            :class:`keras.losses.Loss` applies the reduction; under the default
            ``sum_over_batch_size`` this reproduces PyTorch's ``CrossEntropyLoss``
            default reduction (mean over the batch) exactly.
        """
        del y_true  # Positional positive (index 0); see the class docstring.

        scores = _reshape_to_nway(y_pred, self.nway, "y_pred")

        # logsumexp_k(s_k) - s_0, computed via log_softmax for numerical stability.
        log_probs = keras.ops.log_softmax(scores, axis=-1)
        return -log_probs[:, 0]

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration.

        :returns: Base :class:`keras.losses.Loss` config plus ``nway``.
        """
        config = super().get_config()
        config.update({"nway": self.nway})
        return config


@register_dl_technique("dl_techniques.losses.colbert_loss")
class ColBERTDistillationLoss(keras.losses.Loss):
    """ColBERT v2 objective: batch-mean KL between two log-probability distributions.

    The reference trainer computes, per training tuple::

        target = log_softmax(teacher_scores * distillation_alpha, axis=-1)
        student = log_softmax(student_scores, axis=-1)
        loss = KLDivLoss(reduction='batchmean', log_target=True)(student, target)

    Two details of that snippet are load-bearing and are easy to get silently wrong:

    ``log_target=True``
        Both operands are **log**-probabilities, and PyTorch's kernel is then
        ``sum_k exp(t_k) * (t_k - s_k)``. It is *not* ``sum p * log(p/q)`` over
        probabilities. :class:`keras.losses.KLDivergence` implements the latter and
        clips its inputs into ``[epsilon, 1]``; handing it log-probabilities returns a
        small, plausible, wrong number without raising. This class therefore implements
        the ``log_target=True`` kernel directly with ``keras.ops``.

    ``reduction='batchmean'``
        Divides by the **batch size**, not by the number of elements. :meth:`call` here
        returns the per-row sums, shape ``(batch,)``, so the parent's *default*
        ``sum_over_batch_size`` reduction divides by exactly the batch size and
        reproduces batchmean. Averaging over the ``nway`` axis instead would divide by
        ``batch * nway`` -- an ``nway``-fold (64x under the v2 recipe) silent
        under-scaling of the loss and every gradient. Passing a non-default
        ``reduction=`` to the constructor moves this class off batchmean semantics;
        that is permitted and unguarded, so do not do it unless you mean to.

    :param nway: Candidates per query. The v2 recipe uses 64-way tuples; the reference
        config default (a v1-era value) is 2. Must be ``>= 2``.
    :param distillation_alpha: Positive scale applied to the teacher scores **before**
        the log-softmax. It is a temperature in disguise: ``alpha > 1`` sharpens the
        teacher distribution, ``alpha < 1`` flattens it. Reference default 1.0.
    :param kwargs: Standard :class:`keras.losses.Loss` keyword arguments.
    :raises ValueError: If ``nway < 2`` or ``distillation_alpha <= 0``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.losses import ColBERTDistillationLoss

        loss_fn = ColBERTDistillationLoss(nway=3, distillation_alpha=1.0)
        teacher = keras.ops.convert_to_tensor([[2.0, 0.0, 1.0]])
        student = keras.ops.convert_to_tensor([[3.0, 1.0, 0.0]])
        loss_fn(teacher, student)
    """

    def __init__(
        self,
        nway: int = 64,
        distillation_alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nway = _validate_nway(nway)

        if not isinstance(distillation_alpha, (int, float)) or isinstance(
            distillation_alpha, bool
        ):
            raise ValueError(
                f"distillation_alpha must be a number, got "
                f"{type(distillation_alpha).__name__}: {distillation_alpha!r}"
            )
        if not distillation_alpha > 0.0:
            raise ValueError(
                f"distillation_alpha must be > 0 (it scales the teacher scores before "
                f"the log-softmax; 0 collapses the teacher to a uniform distribution "
                f"and a negative value inverts the teacher's ranking), got "
                f"{distillation_alpha}"
            )
        self.distillation_alpha = float(distillation_alpha)

    def _log_distributions(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Group and log-softmax both operands.

        Interface contract (2 callers: :meth:`call` and the module's own tests, which
        use it to separate the grouping/validation stage from the divergence stage):

        :param y_true: Teacher scores, ``(batch * nway,)`` or ``(batch, nway)``.
        :param y_pred: Student scores, same accepted shapes.
        :returns: ``(target_log_probs, student_log_probs)``, both ``(batch, nway)``.
        :raises ValueError: Propagated from :func:`_reshape_to_nway`.
        """
        teacher = _reshape_to_nway(y_true, self.nway, "y_true (teacher scores)")
        student = _reshape_to_nway(y_pred, self.nway, "y_pred (student scores)")

        alpha = keras.ops.cast(self.distillation_alpha, teacher.dtype)
        target_log_probs = keras.ops.log_softmax(teacher * alpha, axis=-1)
        student_log_probs = keras.ops.log_softmax(student, axis=-1)
        return target_log_probs, student_log_probs

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the per-sample KL divergence of the two log-probability rows.

        :param y_true: Cross-encoder **teacher** scores (raw, unscaled),
            ``(batch * nway,)`` or ``(batch, nway)``.
        :param y_pred: **Student** MaxSim scores, same accepted shapes.
        :returns: Per-sample losses of shape ``(batch,)``, each the row sum
            ``sum_k exp(t_k) * (t_k - s_k)``.
        """
        # DECISION plan-2026-08-25T121346-c71fc3ad/D-015
        # BOTH OPERANDS ARE LOG-PROBABILITIES. This is PyTorch's
        # ``KLDivLoss(..., log_target=True)``, whose kernel is
        # ``sum(exp(target) * (target - input))``. Do NOT replace this expression with
        # ``keras.losses.KLDivergence`` / ``keras.ops.kl_divergence``: those implement
        # ``sum(p * log(p / q))`` for operands that are PROBABILITIES, and they clip
        # their inputs to ``[epsilon, 1]``. Feeding log-probabilities (which are
        # negative) into them does not raise -- it clips every entry to ~1e-7 and
        # returns a small, plausible, WRONG number. The reference is
        # ``colbert/training/training.py`` lines ~104-114; see decisions.md D-015.
        target_log_probs, student_log_probs = self._log_distributions(y_true, y_pred)
        target_probs = keras.ops.exp(target_log_probs)
        per_sample = keras.ops.sum(
            target_probs * (target_log_probs - student_log_probs), axis=-1
        )

        # DECISION plan-2026-08-25T121346-c71fc3ad/D-016
        # ``reduction='batchmean'`` DIVIDES BY THE BATCH SIZE, not by the element count.
        # It is realized here by returning the per-ROW sums (shape ``(batch,)``) and
        # letting the parent's DEFAULT ``sum_over_batch_size`` divide that vector by its
        # own length, which is the batch size -- so ``loss_fn(t, s)`` equals
        # ``sum_all(exp(t)*(t-s)) / batch`` exactly. Do NOT "simplify" this to
        # ``keras.ops.mean(..., axis=-1)`` or to a scalar ``keras.ops.mean`` over the
        # full matrix: either divides by ``batch * nway`` instead, an ``nway``-fold
        # silent under-scaling of the loss and of every gradient (64x under the v2
        # recipe). Constructing this loss with a non-default ``reduction`` also leaves
        # batchmean semantics; that is documented, not guarded. See decisions.md D-016.
        return per_sample

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration.

        :returns: Base :class:`keras.losses.Loss` config plus ``nway`` and
            ``distillation_alpha``.
        """
        config = super().get_config()
        config.update(
            {
                "nway": self.nway,
                "distillation_alpha": self.distillation_alpha,
            }
        )
        return config


logger.debug("ColBERT loss objectives registered (v1 pairwise softmax, v2 KL distillation)")
