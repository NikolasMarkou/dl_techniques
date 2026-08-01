"""Frozen-feature k-NN evaluation + the DINO collapse diagnostic, as one callback.

SSL pretraining has no validation loss to watch. `src/train/dino/train_dino.py` cannot
even pass ``validation_data`` -- `DINOLoss` advances its centering EMA inside ``call()``,
so a validation set silently multiplies the per-epoch centering updates. The real
validation signal is this module: a k-NN probe on FROZEN student features, which never
invokes the loss, plus the two numbers that say whether the representation collapsed.

-------------------------------------------------------------------------------
Why the collapse diagnostic exists (Pre-Mortem 3 -- read this before reading a run)
-------------------------------------------------------------------------------
**A DECREASING LOSS DOES NOT RULE OUT COLLAPSE.** The collapsed solution -- every view
of every image mapping to the same output distribution -- is a genuine minimum of DINO's
cross-view cross-entropy whenever centering/sharpening is not doing its job. A run can
therefore produce a textbook loss curve, a clean exit and a completely dead
representation. "The loss went down" is a near-vacuous assertion on its own; these three
numbers are what make it mean something:

======================================  ==============================  ===============
number                                  collapse drives it              logged as
======================================  ==============================  ===============
mean pairwise cosine of CLS features    toward 1.0                      ``dino_feat_mean_cos``
entropy of the mean teacher softmax     toward 0                        ``dino_teacher_entropy``
  (as a fraction of ``log(out_dim)``)   toward 0                        ``dino_teacher_entropy_norm``
k-NN top-1                              toward chance                   ``dino_knn_top1_k{10,20}``
======================================  ==============================  ===============

**STOP thresholds.** The run is COLLAPSED -- and that is a REPORTABLE OUTCOME, not a
pass -- if the loss decreases while ANY of these holds:

* ``dino_feat_mean_cos`` > :data:`COLLAPSE_COSINE_THRESHOLD` (0.95); or
* ``dino_teacher_entropy_norm`` < :data:`COLLAPSE_ENTROPY_FRACTION` (0.10), i.e. the
  teacher's mean distribution is within ~10% of zero entropy; or
* ``dino_knn_top1_k20`` sits at chance -- **0.10 for imagenette's 10 classes**.

Do NOT widen these to make a run pass, and do NOT declare the smoke run green on the
loss curve alone. When the flag fires, diagnose it: centering dead (the loss's `center`
Variable never moved), teacher EMA dead (`TeacherEMACallback` self-disabled -- grep the
run log for its warning), or the teacher temperature schedule wrong.

The first two are computed on every evaluation and summarized into
``dino_collapse_flag`` (1.0 = at least one threshold tripped, 0.0 = none), which is also
logged at WARNING level so it appears in the run log, not only in the CSV.

-------------------------------------------------------------------------------
Two things about this callback that are load-bearing, not cosmetic
-------------------------------------------------------------------------------
1. **It must sit BEFORE `keras.callbacks.CSVLogger` in the callback list, and it must
   write its keys on EPOCH 0.** MEASURED on keras 3.8.0: `CSVLogger` freezes its
   fieldnames from ``sorted(logs.keys())`` on the first epoch it sees. A key first added
   on epoch 1 NEVER appears in the CSV (measured header ``['epoch', 'loss', 'val_loss']``
   for all three epochs), and a callback placed AFTER `CSVLogger` writes into a `logs`
   dict that has already been serialized -- also measured, also silently absent. Hence
   :meth:`KNNEvalCallback.on_epoch_end` writes EVERY key on EVERY epoch, filling skipped
   epochs with ``nan`` rather than omitting them.
2. **Features come from the STUDENT BACKBONE, not the projection head.** DINO's protocol
   evaluates the backbone representation; the head is a training-time device thrown away
   at transfer. The tensor is `student.head.input` -- the post-final-norm CLS token, width
   ``embed_dim`` -- NOT `student.output`, width ``dino_out_dim``.

Run the guards::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_dino/test_knn_eval.py -q
"""

import keras
import numpy as np
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

#: DINO's k-NN protocol reports k = 10 and k = 20 (Caron et al. 2021, §5 / Table 2).
DEFAULT_KS: Tuple[int, ...] = (10, 20)

#: Temperature of the exp(sim / T) neighbour weighting. The DINO paper uses 0.07.
DEFAULT_KNN_TEMPERATURE: float = 0.07

#: Mean pairwise cosine above which the features are called collapsed (Pre-Mortem 3).
COLLAPSE_COSINE_THRESHOLD: float = 0.95

#: Teacher entropy, as a fraction of log(out_dim), below which it is called collapsed.
COLLAPSE_ENTROPY_FRACTION: float = 0.10

#: Cosine similarity above which two feature rows are treated as the SAME sample.
_NEAR_DUPLICATE_COSINE: float = 1.0 - 1e-6

# Guards the normalization of an all-zero feature row rather than dividing by zero.
_NORM_FLOOR: float = 1e-12


# ---------------------------------------------------------------------
# feature math
# ---------------------------------------------------------------------


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    """Row-normalize in float64. Zero rows come back as zero rows, not NaN."""
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, _NORM_FLOOR)


def _mean_pairwise_cosine(unit_features: np.ndarray) -> float:
    """Mean cosine over the strictly-upper triangle (self-similarity excluded)."""
    n = unit_features.shape[0]
    if n < 2:
        return float("nan")
    gram = unit_features @ unit_features.T
    upper = np.triu_indices(n, k=1)
    return float(np.mean(gram[upper]))


def knn_top1_accuracy(
        bank_features: np.ndarray,
        bank_labels: np.ndarray,
        query_features: np.ndarray,
        query_labels: np.ndarray,
        *,
        k: int = 20,
        temperature: float = DEFAULT_KNN_TEMPERATURE,
        num_classes: Optional[int] = None,
        check_overlap: bool = True,
) -> float:
    """Weighted k-NN top-1 accuracy over cosine similarity -- DINO's eval protocol.

    Interface contract:
        Parameters:
            bank_features: ``(n_bank, d)`` memory-bank features (the TRAIN split).
            bank_labels: ``(n_bank,)`` integer class labels for the bank.
            query_features: ``(n_query, d)`` features to classify (the VALIDATION
                split). Must come from samples DISJOINT from the bank -- see
                ``check_overlap``.
            query_labels: ``(n_query,)`` integer ground-truth labels.
            k: Neighbours per query. Clipped to ``n_bank`` when the bank is smaller.
            temperature: Weighting temperature ``T`` in ``exp(sim / T)``. The paper
                uses 0.07; a larger ``T`` approaches an unweighted majority vote.
            num_classes: Class count. ``None`` infers ``max(label) + 1`` over both
                label arrays.
            check_overlap: Run the memory-bank/query overlap guard (below). Only a
                deliberate self-retrieval experiment should switch it off.
        Returns:
            Top-1 accuracy in ``[0, 1]`` as a Python float.
        Failure mode:
            ``ValueError`` for a shape/width mismatch, an empty set, a non-positive
            ``k`` or ``temperature``, or -- from the overlap guard -- a query set
            that appears to contain its own bank entries.

    **The weighting is temperature-weighted, not a majority vote**: each of the ``k``
    neighbours contributes ``exp(sim / T)`` to its own class, and the arg-max class wins.
    This is the paper's rule; the unweighted variant is NOT what ships here.

    **The overlap guard, and why it is not simply "any near-duplicate is a bug".** A k-NN
    whose memory bank contains its own queries reports a high, meaningless accuracy --
    every query retrieves itself at cosine 1.0. But a COLLAPSED representation also puts
    every pair at cosine ~1.0, and that must be reported as collapse rather than refused
    as an overlap. The guard therefore fires only on the combination that is uniquely
    overlap: MOST queries have a near-duplicate in the bank, WHILE the queries are not
    mutually degenerate (their own mean pairwise cosine is below
    :data:`COLLAPSE_COSINE_THRESHOLD`).
    """
    bank_features = np.asarray(bank_features)
    query_features = np.asarray(query_features)
    bank_labels = np.asarray(bank_labels).reshape(-1).astype(np.int64)
    query_labels = np.asarray(query_labels).reshape(-1).astype(np.int64)

    if bank_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError(
            f"bank_features and query_features must be rank-2 (n, d), got "
            f"{bank_features.shape} and {query_features.shape}."
        )
    if bank_features.shape[0] == 0 or query_features.shape[0] == 0:
        raise ValueError(
            f"k-NN needs a non-empty bank and query set, got "
            f"{bank_features.shape[0]} bank and {query_features.shape[0]} query rows."
        )
    if bank_features.shape[1] != query_features.shape[1]:
        raise ValueError(
            f"bank and query features must have the same width, got "
            f"{bank_features.shape[1]} and {query_features.shape[1]}."
        )
    if bank_features.shape[0] != bank_labels.shape[0]:
        raise ValueError(
            f"bank_features has {bank_features.shape[0]} rows but bank_labels has "
            f"{bank_labels.shape[0]}."
        )
    if query_features.shape[0] != query_labels.shape[0]:
        raise ValueError(
            f"query_features has {query_features.shape[0]} rows but query_labels has "
            f"{query_labels.shape[0]}."
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    unit_bank = _l2_normalize(bank_features)
    unit_query = _l2_normalize(query_features)

    # (n_query, n_bank) cosine similarities.
    similarity = unit_query @ unit_bank.T

    if check_overlap:
        # DECISION plan-2026-08-01T105809-dc0c402e/D-030
        # The guard is a CONJUNCTION on purpose. Do NOT simplify it to "any query with
        # a bank entry at cosine ~1.0 is an overlap": a COLLAPSED representation puts
        # EVERY pair at cosine ~1.0, so the simple form would raise on exactly the run
        # this module exists to detect, converting Pre-Mortem 3's reportable outcome
        # into a crash and hiding the diagnostic. Overlap is distinguished from
        # collapse by the queries' OWN mutual similarity being low.
        has_near_duplicate = np.any(similarity > _NEAR_DUPLICATE_COSINE, axis=1)
        duplicate_fraction = float(np.mean(has_near_duplicate))
        query_self_cosine = _mean_pairwise_cosine(unit_query)
        queries_are_degenerate = (
            not np.isnan(query_self_cosine)
            and query_self_cosine > COLLAPSE_COSINE_THRESHOLD
        )
        if duplicate_fraction > 0.5 and not queries_are_degenerate:
            raise ValueError(
                f"the memory bank and the query set OVERLAP: "
                f"{duplicate_fraction:.1%} of queries have a bank entry at cosine "
                f"> {_NEAR_DUPLICATE_COSINE} while the queries are not mutually "
                f"degenerate (their mean pairwise cosine is "
                f"{query_self_cosine:.4f}, below the {COLLAPSE_COSINE_THRESHOLD} "
                f"collapse threshold). Every such query retrieves ITSELF, so the "
                f"reported accuracy would be meaningless. Draw the bank from the "
                f"TRAIN split and the queries from the VALIDATION split."
            )

    if num_classes is None:
        num_classes = int(max(bank_labels.max(), query_labels.max())) + 1
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    effective_k = int(min(k, bank_features.shape[0]))

    # Top-k by similarity, per query.
    top_index = np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]
    top_similarity = np.take_along_axis(similarity, top_index, axis=1)

    # exp(sim / T) weighting (the paper's rule). Shift by the row max first: the
    # weights are used only relatively, and exp(1/0.07) already overflows float32.
    shifted = top_similarity - np.max(top_similarity, axis=1, keepdims=True)
    weights = np.exp(shifted / float(temperature))

    neighbour_labels = bank_labels[top_index]
    scores = np.zeros((query_features.shape[0], num_classes), dtype=np.float64)
    np.add.at(
        scores,
        (np.arange(query_features.shape[0])[:, None], neighbour_labels),
        weights,
    )

    predictions = np.argmax(scores, axis=1)
    return float(np.mean(predictions == query_labels))


def collapse_metrics(
        features: np.ndarray,
        teacher_probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Pre-Mortem 3's collapse detector: feature similarity + teacher entropy.

    Interface contract:
        Parameters:
            features: ``(n, d)`` frozen student CLS features.
            teacher_probabilities: ``(n, out_dim)`` centered+sharpened teacher
                softmax rows, or ``None`` to skip the entropy half (both entropy
                keys then come back as ``nan``).
        Returns:
            ``{"mean_pairwise_cosine", "teacher_entropy",
            "teacher_entropy_normalized", "collapse_flag"}``. ``collapse_flag`` is
            ``1.0`` when either threshold trips and ``0.0`` when neither does; a
            metric that could not be computed (``nan``) never trips it.
        Failure mode:
            ``ValueError`` if ``features`` is not rank-2.

    ``teacher_entropy`` is in NATS and ``teacher_entropy_normalized`` divides it by
    ``log(out_dim)``, the entropy of the uniform distribution -- so it lands in
    ``[0, 1]`` and the "within ~10% of zero" threshold means the same thing at every
    ``out_dim``. The entropy is taken of the MEAN teacher distribution over the batch,
    which is what detects DINO's collapse mode: all mass concentrating on a few output
    dimensions regardless of the input.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(
            f"features must be rank-2 (n, d), got shape {features.shape}.")

    mean_cosine = _mean_pairwise_cosine(_l2_normalize(features))

    entropy = float("nan")
    entropy_normalized = float("nan")
    if teacher_probabilities is not None:
        probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError(
                f"teacher_probabilities must be rank-2 (n, out_dim), got shape "
                f"{probabilities.shape}."
            )
        mean_distribution = np.mean(probabilities, axis=0)
        total = float(np.sum(mean_distribution))
        if total > 0:
            mean_distribution = mean_distribution / total
        entropy = float(
            -np.sum(mean_distribution * np.log(np.maximum(mean_distribution, 1e-12)))
        )
        uniform_entropy = float(np.log(max(probabilities.shape[1], 2)))
        entropy_normalized = entropy / uniform_entropy

    collapsed = (
        (not np.isnan(mean_cosine) and mean_cosine > COLLAPSE_COSINE_THRESHOLD)
        or (not np.isnan(entropy_normalized)
            and entropy_normalized < COLLAPSE_ENTROPY_FRACTION)
    )

    return {
        "mean_pairwise_cosine": float(mean_cosine),
        "teacher_entropy": entropy,
        "teacher_entropy_normalized": entropy_normalized,
        "collapse_flag": 1.0 if collapsed else 0.0,
    }


# ---------------------------------------------------------------------
# the callback
# ---------------------------------------------------------------------


class KNNEvalCallback(keras.callbacks.Callback):
    """Evaluate frozen student features by k-NN, and report the collapse diagnostic.

    Every ``every_n_epochs`` epochs (and ALWAYS on epoch 0) this extracts frozen
    student BACKBONE CLS features for a labelled memory bank and a labelled query set,
    runs :func:`knn_top1_accuracy` at each ``k``, computes :func:`collapse_metrics`, and
    writes every number into the epoch ``logs`` dict so `CSVLogger` carries it into
    ``training_log.csv``.

    Args:
        bank_dataset: Batched ``(image, label)`` dataset for the memory bank. Draw it
            from the TRAIN split.
        query_dataset: Batched ``(image, label)`` dataset for the queries. Draw it from
            the VALIDATION split -- disjoint from the bank by construction, and checked
            numerically by :func:`knn_top1_accuracy`'s overlap guard.
        bank_batches: Batches consumed from ``bank_dataset``.
        query_batches: Batches consumed from ``query_dataset``.
        ks: Neighbour counts to report. DINO reports 10 and 20.
        temperature: Neighbour-weighting temperature.
        every_n_epochs: Evaluation period. Epoch 0 is always evaluated (the CSV
            fieldnames are frozen there).
        dino_loss: The compiled `DINOLoss`, optional. When given, the teacher softmax
            is CENTERED and SHARPENED exactly as the loss does it (``softmax((logits -
            center) / teacher_temp)``), which is the distribution whose entropy the
            diagnostic is about. Without it, a plain ``softmax(logits)`` is used and
            the entropy reads high for a reason that has nothing to do with training.
        num_classes: Class count for the k-NN vote; ``None`` infers it from the labels.

    Raises:
        ValueError: For a non-positive ``every_n_epochs`` / batch count / temperature,
            an empty ``ks``, or -- at evaluation time -- a model that is not a
            `DINOTrainingModel` (no ``student`` attribute) or a student with no
            projection head to read the backbone feature from.

    **This callback never self-disables.** If the model does not expose what it needs it
    RAISES. (`TeacherEMACallback`'s one-warning self-disable is a trap this run already
    guards against; a k-NN probe that quietly stops reporting would hide exactly the
    collapse this module exists to detect.)

    Example:
        ```python
        callbacks.insert(
            index_of_csv_logger,
            KNNEvalCallback(bank_ds, query_ds, bank_batches=16, query_batches=8,
                            dino_loss=loss),
        )
        ```
    """

    def __init__(
            self,
            bank_dataset: Iterable[Any],
            query_dataset: Iterable[Any],
            *,
            bank_batches: int = 16,
            query_batches: int = 8,
            ks: Sequence[int] = DEFAULT_KS,
            temperature: float = DEFAULT_KNN_TEMPERATURE,
            every_n_epochs: int = 1,
            dino_loss: Optional[Any] = None,
            num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        if bank_batches <= 0 or query_batches <= 0:
            raise ValueError(
                f"bank_batches and query_batches must be positive, got "
                f"{bank_batches} and {query_batches}."
            )
        ks = tuple(int(value) for value in ks)
        if not ks or any(value <= 0 for value in ks):
            raise ValueError(f"ks must be a non-empty tuple of positive ints, got {ks}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if every_n_epochs <= 0:
            raise ValueError(
                f"every_n_epochs must be positive, got {every_n_epochs}")

        self.bank_dataset = bank_dataset
        self.query_dataset = query_dataset
        self.bank_batches = int(bank_batches)
        self.query_batches = int(query_batches)
        self.ks = ks
        self.temperature = float(temperature)
        self.every_n_epochs = int(every_n_epochs)
        self.dino_loss = dino_loss
        self.num_classes = num_classes

        self._feature_model: Optional[keras.Model] = None

    # -- keys -------------------------------------------------------------

    @property
    def log_keys(self) -> Tuple[str, ...]:
        """Every key this callback writes, in a fixed order.

        The set is FIXED and written on every epoch -- see the module docstring's
        measured note on `CSVLogger` freezing its fieldnames at epoch 0.
        """
        return tuple(
            [f"dino_knn_top1_k{k}" for k in self.ks]
            + [
                "dino_feat_mean_cos",
                "dino_teacher_entropy",
                "dino_teacher_entropy_norm",
                "dino_collapse_flag",
            ]
        )

    # -- feature extraction ------------------------------------------------

    def _build_feature_model(self) -> keras.Model:
        """Expose the student's BACKBONE CLS feature, not the projection head output.

        Interface contract:
            Parameters: none (reads ``self.model``).
            Returns: a `keras.Model` mapping one image batch to ``(batch, embed_dim)``.
            Failure mode: ``ValueError`` when the attached model is not a
                `DINOTrainingModel`, or its student has no projection head.

        The tensor is ``student.head.input`` -- the post-final-norm CLS token that
        `DINOv1._build_model` feeds into `DINOHead`. Do NOT "simplify" this to
        ``student.output``: that is the ``dino_out_dim``-wide projection-head output,
        which is a training-time device the DINO protocol explicitly does not evaluate.
        Its width is pinned by a test against ``student.embed_dim``.
        """
        # DECISION plan-2026-08-01T105809-dc0c402e/D-030
        # The output is `head.input`, NOT `student.output`. Do NOT "simplify" this to
        # `keras.Model(student.inputs[0], student.output)` or to calling `student(x)`
        # directly -- both give the PROJECTION HEAD's `dino_out_dim`-wide logits, which
        # DINO's k-NN protocol explicitly does not evaluate (the head is a training-time
        # device discarded at transfer). The substitution is SILENT: it changes no shape
        # the callback checks, raises nothing, and still produces a plausible accuracy.
        # MEASURED at the smoke scale: backbone width 192, head width 32.
        # RED-proven -- swapping to `head.output` fails two tests in
        # tests/test_train/test_dino/test_knn_eval.py::TestBackboneFeatures.
        model = self.model
        student = getattr(model, "student", None)
        if student is None:
            raise ValueError(
                f"{type(self).__name__} needs a DINOTrainingModel (an object with a "
                f"`student` sub-model) but was attached to {type(model).__name__}, "
                f"which has no `student` attribute."
            )
        head = getattr(student, "head", None)
        if head is None:
            raise ValueError(
                "the student has no projection head, so there is no backbone/head "
                "boundary to read the CLS feature from. Build it with "
                "include_projection_head=True."
            )
        return keras.Model(inputs=student.inputs[0], outputs=head.input)

    def _extract(
            self,
            dataset: Iterable[Any],
            n_batches: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run the frozen networks over ``n_batches`` and return numpy arrays.

        Interface contract:
            Parameters:
                dataset: Batched ``(image, label)`` iterable.
                n_batches: How many batches to consume.
            Returns:
                ``(features, labels, teacher_logits)`` -- ``teacher_logits`` is
                ``None`` when the model exposes no ``teacher``.
            Failure mode:
                ``ValueError`` if the dataset yielded nothing.

        Everything runs with ``training=False``: this is a frozen-feature probe, so
        dropout / stochastic depth must be off or the features are not the ones the
        checkpoint would produce at transfer time.
        """
        if self._feature_model is None:
            self._feature_model = self._build_feature_model()

        teacher = getattr(self.model, "teacher", None)

        features, labels, teacher_logits = [], [], []
        for index, batch in enumerate(dataset):
            if index >= n_batches:
                break
            images, batch_labels = batch[0], batch[1]
            features.append(
                np.asarray(self._feature_model(images, training=False)))
            labels.append(np.asarray(batch_labels).reshape(-1))
            if teacher is not None:
                teacher_logits.append(np.asarray(teacher(images, training=False)))

        if not features:
            raise ValueError(
                f"{type(self).__name__} consumed 0 batches from its dataset; it "
                f"cannot evaluate anything."
            )

        return (
            np.concatenate(features, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(teacher_logits, axis=0) if teacher_logits else None,
        )

    def _teacher_probabilities(
            self,
            teacher_logits: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Center and sharpen the teacher logits the way `DINOLoss` does.

        Interface contract:
            Parameters:
                teacher_logits: ``(n, out_dim)`` raw teacher outputs, or ``None``.
            Returns:
                ``(n, out_dim)`` softmax rows, or ``None`` when there were no logits.
            Failure mode: none -- a missing/mis-shaped center is ignored rather than
                raising, because a diagnostic must never take a run down.

        Without ``dino_loss`` this is a plain ``softmax(logits)``, whose entropy says
        nothing about the distribution the student is actually trained against.
        """
        if teacher_logits is None:
            return None

        logits = np.asarray(teacher_logits, dtype=np.float64)
        temperature = 1.0
        if self.dino_loss is not None:
            center = np.asarray(
                keras.ops.convert_to_numpy(self.dino_loss.center),
                dtype=np.float64,
            ).reshape(-1)
            if center.shape[0] == logits.shape[1]:
                logits = logits - center[None, :]
            temperature = float(self.dino_loss.teacher_temp)

        logits = logits / max(temperature, 1e-12)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

    # -- the Keras hook ----------------------------------------------------

    def on_epoch_end(
            self,
            epoch: int,
            logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write every key into ``logs``; evaluate on epoch 0 and every N-th epoch."""
        if logs is None:
            # Keras always passes a dict; a None would mean the numbers go nowhere,
            # which is worse than a loud failure in a diagnostic.
            raise ValueError(
                f"{type(self).__name__}.on_epoch_end was called with logs=None; its "
                f"metrics would be discarded."
            )

        # DECISION plan-2026-08-01T105809-dc0c402e/D-029
        # Every key, EVERY epoch -- including epochs this callback does not evaluate.
        # Do NOT move this block below the `every_n_epochs` early-return "so skipped
        # epochs write nothing": `CSVLogger` freezes its fieldnames from
        # `sorted(logs.keys())` on the FIRST epoch it sees, so a key first written on
        # epoch 1 never appears in the CSV at all. MEASURED on keras 3.8.0: header
        # ['epoch', 'loss', 'val_loss'] for all three epochs. `nan` marks a skipped
        # evaluation honestly; carrying the previous epoch's value forward would be
        # worse, since a stale number is indistinguishable from a fresh one.
        for key in self.log_keys:
            logs.setdefault(key, float("nan"))

        if epoch != 0 and epoch % self.every_n_epochs != 0:
            return

        bank_features, bank_labels, _ = self._extract(
            self.bank_dataset, self.bank_batches)
        query_features, query_labels, teacher_logits = self._extract(
            self.query_dataset, self.query_batches)

        for k in self.ks:
            logs[f"dino_knn_top1_k{k}"] = knn_top1_accuracy(
                bank_features,
                bank_labels,
                query_features,
                query_labels,
                k=k,
                temperature=self.temperature,
                num_classes=self.num_classes,
            )

        diagnostics = collapse_metrics(
            query_features, self._teacher_probabilities(teacher_logits))
        logs["dino_feat_mean_cos"] = diagnostics["mean_pairwise_cosine"]
        logs["dino_teacher_entropy"] = diagnostics["teacher_entropy"]
        logs["dino_teacher_entropy_norm"] = diagnostics["teacher_entropy_normalized"]
        logs["dino_collapse_flag"] = diagnostics["collapse_flag"]

        summary = " ".join(f"{key}={logs[key]:.4f}" for key in self.log_keys)
        if diagnostics["collapse_flag"] > 0.0:
            logger.warning(
                f"COLLAPSE DIAGNOSTIC FIRED at epoch {epoch}: {summary}. A decreasing "
                f"loss does NOT rule this out -- the collapsed solution is a real "
                f"minimum of the cross-view objective. Report it as the run's outcome; "
                f"do not widen the thresholds "
                f"(cosine > {COLLAPSE_COSINE_THRESHOLD}, normalized entropy < "
                f"{COLLAPSE_ENTROPY_FRACTION})."
            )
        else:
            logger.info(f"k-NN eval at epoch {epoch}: {summary}")


# ---------------------------------------------------------------------
