"""Guards for the DINO k-NN probe and the collapse diagnostic (``train.dino.knn_eval``).

This is the only validation signal a DINO run has, and every way it can be wrong is
SILENT. The five things pinned here are exactly the five silent failures:

1. **The k-NN actually classifies.** A function returning a constant, or one whose
   memory bank contains its own queries, reports a fine-looking number that means
   nothing. So: perfectly separable synthetic features must score ~1.0, identical
   features must fall to chance, and an overlapping bank/query pair must RAISE.

2. **The collapse detector can FIRE.** A detector that has never fired proves nothing
   about a run where it did not fire. It is therefore fed a deliberately collapsed
   feature matrix and required to trip.

3. **The numbers reach ``training_log.csv``.** MEASURED on keras 3.8.0: `CSVLogger`
   freezes its fieldnames from ``sorted(logs.keys())`` on the first epoch it sees, so a
   key first written on epoch 1 never appears at all, and a callback placed AFTER
   `CSVLogger` writes into a dict that has already been serialized. Both produce a
   well-formed CSV with the columns missing.

4. **Features come from the BACKBONE, not the projection head.** DINO evaluates the
   backbone representation; the head is discarded at transfer. The widths differ, and
   that difference is asserted.

5. **Bank and query sets are DISJOINT.** Guarded numerically, and RED-proven.

Run:
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_dino/test_knn_eval.py -q
"""

import csv
import json
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.models.dino.training import create_dino_training_model
from train.dino import train_dino as trainer
from train.dino.knn_eval import (
    COLLAPSE_COSINE_THRESHOLD,
    COLLAPSE_ENTROPY_FRACTION,
    DEFAULT_KNN_TEMPERATURE,
    DEFAULT_KS,
    KNNEvalCallback,
    collapse_metrics,
    knn_top1_accuracy,
)

# ---------------------------------------------------------------------------
# synthetic feature fixtures -- seeded and NON-ZERO throughout
# ---------------------------------------------------------------------------

N_CLASSES = 4
FEATURE_DIM = 8


def _separable(
        n_per_class: int,
        seed: int,
        jitter: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """One tight, well-separated Gaussian blob per class, at a NON-ZERO mean.

    A zeros-centred fixture would make several probes below structurally blind: a
    dead feature extractor returning zeros is indistinguishable from a live one when
    the signal itself is zero.
    """
    rng = np.random.default_rng(seed)
    centres = np.eye(N_CLASSES, FEATURE_DIM) * 3.0 + 1.0  # non-zero everywhere
    features, labels = [], []
    for label in range(N_CLASSES):
        blob = centres[label][None, :] + rng.normal(
            scale=jitter, size=(n_per_class, FEATURE_DIM))
        features.append(blob)
        labels.append(np.full(n_per_class, label))
    return (
        np.concatenate(features, axis=0).astype("float32"),
        np.concatenate(labels, axis=0).astype("int64"),
    )


def _collapsed(n: int) -> np.ndarray:
    """The same NON-ZERO feature row repeated -- DINO's collapse, exactly."""
    return np.tile(np.arange(1.0, FEATURE_DIM + 1.0)[None, :], (n, 1)).astype("float32")


# ---------------------------------------------------------------------------
# 1. the k-NN classifier itself
# ---------------------------------------------------------------------------


class TestKNNClassifier:
    """PROBE (i)/(ii): separable features -> ~1.0; identical features -> chance."""

    @pytest.mark.parametrize("k", DEFAULT_KS)
    def test_perfectly_separable_features_score_one(self, k: int) -> None:
        bank_features, bank_labels = _separable(n_per_class=25, seed=0)
        query_features, query_labels = _separable(n_per_class=10, seed=1)

        accuracy = knn_top1_accuracy(
            bank_features, bank_labels, query_features, query_labels, k=k)

        assert accuracy == pytest.approx(1.0), (
            f"perfectly separable blobs must be classified perfectly at k={k}, got "
            f"{accuracy}"
        )

    def test_identical_features_fall_to_chance(self) -> None:
        """Every sample identical -> the vote is decided by the bank's label mix.

        With a balanced bank the accuracy must sit at chance (1 / N_CLASSES). This is
        the vacuity boundary for the whole probe: a k-NN that cannot tell samples apart
        MUST report chance, not a high number.
        """
        n_bank, n_query = 40, 20
        bank_features = _collapsed(n_bank)
        # A balanced, deterministic label mix; class 0 wins every tie by arg-max.
        bank_labels = np.arange(n_bank) % N_CLASSES
        query_features = _collapsed(n_query)
        query_labels = np.arange(n_query) % N_CLASSES

        accuracy = knn_top1_accuracy(
            bank_features,
            bank_labels,
            query_features,
            query_labels,
            k=20,
            # the query set IS degenerate here, so the overlap guard correctly stays
            # silent; this asserts the k-NN's own behaviour under collapse.
        )

        assert accuracy == pytest.approx(1.0 / N_CLASSES, abs=1e-9), (
            f"identical features must classify at chance ({1.0 / N_CLASSES}), got "
            f"{accuracy}"
        )

    def test_a_constant_predictor_would_not_pass_both(self) -> None:
        """Non-vacuity control: the two expectations above are far apart.

        A k-NN hard-wired to return a constant cannot satisfy both 1.0 and 0.25, which
        is what makes the pair of assertions above load-bearing rather than decorative.
        """
        assert abs(1.0 - 1.0 / N_CLASSES) > 0.5

    def test_temperature_weighting_beats_a_far_majority(self) -> None:
        """The shipped weighting is exp(sim / T), NOT an unweighted majority vote.

        The bank holds ONE very close neighbour of class 0 and THREE distant ones of
        class 1. A majority vote answers 1; temperature weighting at T=0.07 answers 0.
        """
        query = np.array([[1.0, 0.0]], dtype="float32")
        bank = np.array(
            [
                [1.0, 0.02],    # class 0, cosine ~0.9998
                [0.6, 0.8],     # class 1, cosine 0.6
                [0.6, 0.81],    # class 1
                [0.61, 0.8],    # class 1
            ],
            dtype="float32",
        )
        bank_labels = np.array([0, 1, 1, 1])

        weighted = knn_top1_accuracy(
            bank, bank_labels, query, np.array([0]), k=4,
            temperature=DEFAULT_KNN_TEMPERATURE, num_classes=2)
        near_majority = knn_top1_accuracy(
            bank, bank_labels, query, np.array([0]), k=4,
            temperature=1e3, num_classes=2)

        assert weighted == pytest.approx(1.0), (
            "exp(sim / 0.07) must let the single close neighbour win")
        assert near_majority == pytest.approx(0.0), (
            "at a huge temperature the weighting degenerates to a majority vote, "
            "which the 3-vs-1 bank must win for class 1 -- if this is not 0.0 the "
            "temperature is not actually being used")

    def test_k_is_clipped_to_the_bank_size(self) -> None:
        bank_features, bank_labels = _separable(n_per_class=2, seed=3)
        query_features, query_labels = _separable(n_per_class=2, seed=4)
        accuracy = knn_top1_accuracy(
            bank_features, bank_labels, query_features, query_labels, k=1000)
        assert 0.0 <= accuracy <= 1.0

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"k": 0}, "k must be positive"),
            ({"temperature": 0.0}, "temperature must be positive"),
        ],
    )
    def test_bad_arguments_raise_on_the_message(
            self, kwargs: Dict[str, Any], message: str) -> None:
        bank_features, bank_labels = _separable(n_per_class=3, seed=5)
        query_features, query_labels = _separable(n_per_class=3, seed=6)
        with pytest.raises(ValueError, match=message):
            knn_top1_accuracy(
                bank_features, bank_labels, query_features, query_labels, **kwargs)

    def test_mismatched_widths_raise(self) -> None:
        bank_features, bank_labels = _separable(n_per_class=3, seed=7)
        query_features, query_labels = _separable(n_per_class=3, seed=8)
        with pytest.raises(ValueError, match="same width"):
            knn_top1_accuracy(
                bank_features, bank_labels, query_features[:, :3], query_labels)


# ---------------------------------------------------------------------------
# 2. the disjointness guard
# ---------------------------------------------------------------------------


class TestDisjointnessGuard:
    """A bank containing its own queries reports a free 1.0. That must RAISE."""

    def test_overlapping_bank_and_query_raises(self) -> None:
        features, labels = _separable(n_per_class=20, seed=9)

        with pytest.raises(ValueError, match="OVERLAP"):
            knn_top1_accuracy(features, labels, features, labels, k=10)

    def test_the_guard_names_the_fix(self) -> None:
        features, labels = _separable(n_per_class=20, seed=10)
        with pytest.raises(ValueError, match="TRAIN split"):
            knn_top1_accuracy(features, labels, features, labels, k=10)

    def test_disjoint_sets_do_not_trip_the_guard(self) -> None:
        """Non-vacuity control: the guard must not fire on a legitimate pair."""
        bank_features, bank_labels = _separable(n_per_class=20, seed=11)
        query_features, query_labels = _separable(n_per_class=8, seed=12)
        accuracy = knn_top1_accuracy(
            bank_features, bank_labels, query_features, query_labels, k=10)
        assert accuracy == pytest.approx(1.0)

    def test_a_collapsed_query_set_is_reported_not_refused(self) -> None:
        """Collapse also puts every pair at cosine 1.0 -- and must NOT raise here.

        This is the distinction the guard is built around: overlap and collapse look
        identical to a naive near-duplicate check. Collapse belongs to the collapse
        diagnostic, not to an exception.
        """
        bank = _collapsed(30)
        query = _collapsed(12)
        accuracy = knn_top1_accuracy(
            bank, np.arange(30) % N_CLASSES, query, np.arange(12) % N_CLASSES, k=10)
        assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# 3. the collapse detector
# ---------------------------------------------------------------------------


class TestCollapseDetector:
    """PROBE (iii): RED-prove the detector by feeding it a collapsed matrix."""

    def test_collapsed_features_fire_the_flag(self) -> None:
        metrics = collapse_metrics(_collapsed(32))

        assert metrics["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-9), (
            f"identical features must have mean pairwise cosine 1.0, got "
            f"{metrics['mean_pairwise_cosine']}"
        )
        assert metrics["collapse_flag"] == 1.0, (
            "the collapse detector did NOT fire on a deliberately collapsed feature "
            "matrix; it can therefore never be trusted when it stays silent"
        )

    def test_spread_features_do_not_fire_the_flag(self) -> None:
        """Non-vacuity control: a detector that always fires is also useless."""
        features, _ = _separable(n_per_class=16, seed=13, jitter=0.4)
        metrics = collapse_metrics(features)

        assert metrics["mean_pairwise_cosine"] < COLLAPSE_COSINE_THRESHOLD
        assert metrics["collapse_flag"] == 0.0

    def test_a_one_hot_teacher_distribution_fires_on_entropy_alone(self) -> None:
        """All mass on one output dimension -> entropy 0 -> the flag trips.

        The features here are deliberately SPREAD, so the cosine half cannot be what
        fires: this isolates the entropy half of the detector.
        """
        features, _ = _separable(n_per_class=16, seed=14, jitter=0.4)
        one_hot = np.zeros((64, 128), dtype="float64")
        one_hot[:, 7] = 1.0

        metrics = collapse_metrics(features, one_hot)

        assert metrics["mean_pairwise_cosine"] < COLLAPSE_COSINE_THRESHOLD, (
            "the feature half must be quiet, or this test does not isolate entropy")
        assert metrics["teacher_entropy"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["teacher_entropy_normalized"] < COLLAPSE_ENTROPY_FRACTION
        assert metrics["collapse_flag"] == 1.0

    def test_a_uniform_teacher_distribution_reads_full_entropy(self) -> None:
        features, _ = _separable(n_per_class=16, seed=15, jitter=0.4)
        out_dim = 128
        uniform = np.full((64, out_dim), 1.0 / out_dim)

        metrics = collapse_metrics(features, uniform)

        assert metrics["teacher_entropy"] == pytest.approx(np.log(out_dim), rel=1e-9)
        assert metrics["teacher_entropy_normalized"] == pytest.approx(1.0, rel=1e-9)
        assert metrics["collapse_flag"] == 0.0

    def test_entropy_is_of_the_MEAN_distribution_not_the_mean_entropy(self) -> None:
        """Per-sample one-hots at DIFFERENT dimensions are NOT collapse.

        Mean per-sample entropy would read 0 here and call a perfectly diverse teacher
        collapsed. The entropy of the mean distribution reads log(n) and does not.
        """
        features, _ = _separable(n_per_class=16, seed=16, jitter=0.4)
        out_dim = 64
        diverse = np.eye(out_dim)

        metrics = collapse_metrics(features, diverse)

        assert metrics["teacher_entropy"] == pytest.approx(np.log(out_dim), rel=1e-9)
        assert metrics["collapse_flag"] == 0.0

    def test_missing_teacher_probabilities_give_nan_not_a_false_flag(self) -> None:
        features, _ = _separable(n_per_class=16, seed=17, jitter=0.4)
        metrics = collapse_metrics(features, None)
        assert np.isnan(metrics["teacher_entropy"])
        assert np.isnan(metrics["teacher_entropy_normalized"])
        assert metrics["collapse_flag"] == 0.0

    def test_rank_one_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rank-2"):
            collapse_metrics(np.arange(8.0))


# ---------------------------------------------------------------------------
# 2b. the teacher distribution the entropy is DEFINED on (D-039)
# ---------------------------------------------------------------------------


def _centered_softmax_oracle(
        logits: np.ndarray,
        center: np.ndarray,
        temperature: float,
) -> np.ndarray:
    """Independent numpy oracle for ``softmax((logits - center) / temp)``."""
    scaled = (logits.astype(np.float64) - center.reshape(1, -1)) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class TestTeacherProbabilitiesAreTheDocumentedQuantity:
    """`dino_teacher_entropy` is DEFINED on the loss's centered+sharpened softmax.

    Before D-039 a center whose width disagreed with the teacher head was skipped
    with no else-branch, so the column carried an UNCENTERED softmax under the same
    name. MEASURED at HEAD on 64 rows of width 8 against ``DINOLoss(out_dim=16)``:
    the fallback's rows were BIT-IDENTICAL to a plain softmax, per-row probabilities
    differed from the centered ones by up to 1.0, and the normalized entropy read
    0.966176 instead of 0.844065 -- 0.122 FURTHER from the STOP threshold, i.e. a
    false NEGATIVE.
    """

    WIDTH = 8

    def _logits(self, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.normal(size=(64, self.WIDTH)) * 3.0).astype("float32")

    def _callback(self, loss: Any) -> KNNEvalCallback:
        return KNNEvalCallback([], [], bank_batches=1, query_batches=1,
                               dino_loss=loss)

    def test_a_matched_center_gives_the_centered_sharpened_softmax(self) -> None:
        logits = self._logits()
        loss = DINOLoss(out_dim=self.WIDTH, teacher_temp=0.04)
        rng = np.random.default_rng(99)
        center = (rng.normal(size=(1, self.WIDTH)) * 2.0).astype("float32")
        loss.center.assign(center)

        probabilities, is_centered = self._callback(loss)._teacher_probabilities(
            logits)

        assert is_centered == 1.0
        expected = _centered_softmax_oracle(logits, center, 0.04)
        np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-12)

        # NON-VACUITY: the UNCENTERED softmax must be a visibly different answer,
        # or this test would pass whether or not the center were subtracted.
        uncentered = _centered_softmax_oracle(
            logits, np.zeros(self.WIDTH), 0.04)
        assert np.max(np.abs(expected - uncentered)) > 0.1, (
            "the fixture's center is too weak to separate centered from "
            "uncentered; this test cannot see the substitution it exists for")

    def test_a_mismatched_center_yields_nan_not_an_uncentered_softmax(
            self, caplog: Any) -> None:
        logits = self._logits()
        loss = DINOLoss(out_dim=self.WIDTH * 2, teacher_temp=0.04)

        with caplog.at_level("WARNING", logger="dl"):
            probabilities, is_centered = self._callback(
                loss)._teacher_probabilities(logits)

        assert probabilities is None, (
            "a mis-shaped center must NOT fall through to an uncentered softmax")
        assert is_centered == 0.0
        assert any("out_dim disagrees" in record.message
                   for record in caplog.records), (
            f"the degradation must be LOUD; captured {[r.message for r in caplog.records]}")

        # And it must reach the CSV as nan, not as a number.
        metrics = collapse_metrics(
            _separable(n_per_class=16, seed=21, jitter=0.4)[0], probabilities)
        assert np.isnan(metrics["teacher_entropy"])
        assert np.isnan(metrics["teacher_entropy_normalized"])

    def test_no_dino_loss_yields_nan_not_a_plain_softmax(
            self, caplog: Any) -> None:
        callback = KNNEvalCallback([], [], bank_batches=1, query_batches=1)
        with caplog.at_level("WARNING", logger="dl"):
            probabilities, is_centered = callback._teacher_probabilities(
                self._logits())
        assert probabilities is None
        assert is_centered == 0.0
        assert any("without dino_loss" in record.message
                   for record in caplog.records)

    def test_no_teacher_logits_is_nan_not_zero(self) -> None:
        """No logits at all is a THIRD state, and must not read as 'degraded'."""
        loss = DINOLoss(out_dim=self.WIDTH)
        probabilities, is_centered = self._callback(loss)._teacher_probabilities(
            None)
        assert probabilities is None
        assert np.isnan(is_centered)

    def test_the_warning_is_emitted_once_not_once_per_epoch(
            self, caplog: Any) -> None:
        logits = self._logits()
        callback = self._callback(DINOLoss(out_dim=self.WIDTH * 2))
        with caplog.at_level("WARNING", logger="dl"):
            for _ in range(5):
                callback._teacher_probabilities(logits)
        hits = [r for r in caplog.records if "out_dim disagrees" in r.message]
        assert len(hits) == 1, (
            f"expected exactly one warning across 5 evaluations, got {len(hits)}; "
            f"a per-epoch repeat would bury the COLLAPSE DIAGNOSTIC FIRED line")


# ---------------------------------------------------------------------------
# a real, tiny DINO model + datasets for the callback tests
# ---------------------------------------------------------------------------

IMAGE_SIZE = 32
PATCH_SIZE = 16
OUT_DIM = 32
BATCH = 4


@pytest.fixture(scope="module")
def dino_model() -> keras.Model:
    keras.utils.set_random_seed(20260801)
    model = create_dino_training_model(
        "tiny",
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        n_local_views=1,
        dino_out_dim=OUT_DIM,
    )
    model.build((None, model.n_views, IMAGE_SIZE, IMAGE_SIZE, 3))
    return model


def _labelled_dataset(seed: int, n: int = BATCH * 2) -> tf.data.Dataset:
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(n, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")
    labels = (np.arange(n) % N_CLASSES).astype("int32")
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH)


# ---------------------------------------------------------------------------
# 4. BACKBONE, not projection head
# ---------------------------------------------------------------------------


class TestBackboneFeatures:
    """PROBE (v): swapping the backbone feature for the head output must be caught."""

    def test_the_feature_model_emits_embed_dim_not_out_dim(
            self, dino_model: keras.Model) -> None:
        callback = KNNEvalCallback(
            _labelled_dataset(1), _labelled_dataset(2),
            bank_batches=2, query_batches=2)
        callback.set_model(dino_model)

        feature_model = callback._build_feature_model()
        embed_dim = int(dino_model.student.embed_dim)

        assert embed_dim != OUT_DIM, (
            "this test is only meaningful when the backbone and head widths DIFFER; "
            f"both are {embed_dim}")
        assert feature_model.output_shape[-1] == embed_dim, (
            f"the k-NN must read the STUDENT BACKBONE CLS feature (width "
            f"{embed_dim}), got width {feature_model.output_shape[-1]}. Width "
            f"{OUT_DIM} would mean it is reading the PROJECTION HEAD output, which "
            f"DINO's protocol explicitly does not evaluate."
        )
        assert feature_model.output_shape[-1] != OUT_DIM

    def test_extracted_features_are_backbone_width_and_non_degenerate(
            self, dino_model: keras.Model) -> None:
        callback = KNNEvalCallback(
            _labelled_dataset(3), _labelled_dataset(4),
            bank_batches=2, query_batches=2)
        callback.set_model(dino_model)

        features, labels, teacher_logits = callback._extract(
            _labelled_dataset(3), 2)

        assert features.shape == (BATCH * 2, int(dino_model.student.embed_dim))
        assert labels.shape == (BATCH * 2,)
        assert teacher_logits is not None and teacher_logits.shape[-1] == OUT_DIM
        assert np.all(np.isfinite(features))
        assert float(np.std(features)) > 0.0, (
            "a dead feature extractor returning a constant would pass every shape "
            "assertion above")

    def test_a_non_dino_model_raises_rather_than_self_disabling(self) -> None:
        callback = KNNEvalCallback(
            _labelled_dataset(5), _labelled_dataset(6),
            bank_batches=1, query_batches=1)
        callback.set_model(keras.Sequential([keras.layers.Dense(2)]))

        with pytest.raises(ValueError, match="no `student` attribute"):
            callback._build_feature_model()


# ---------------------------------------------------------------------------
# 5. the numbers reach training_log.csv
# ---------------------------------------------------------------------------


def _run_two_epochs_with_csv(
        model: keras.Model,
        tmp_path: Any,
        callbacks: List[keras.callbacks.Callback],
) -> List[List[str]]:
    """Run a real 2-epoch fit() and return the parsed CSV rows (header first)."""
    rng = np.random.default_rng(99)
    views = rng.normal(
        size=(BATCH * 2, model.n_views, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")
    labels = np.zeros((BATCH * 2,), dtype="int32")

    model.compile(optimizer=keras.optimizers.SGD(1e-4), loss=DINOLoss(out_dim=OUT_DIM))
    model.fit(views, labels, epochs=2, batch_size=BATCH, verbose=0,
              callbacks=callbacks)

    with open(tmp_path / "log.csv") as handle:
        return list(csv.reader(handle))


class TestLogsReachTheCSV:
    """PROBE (iv): if the callback does not write into `logs`, the columns vanish."""

    def test_knn_and_collapse_columns_land_in_the_csv(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        knn = KNNEvalCallback(
            _labelled_dataset(7), _labelled_dataset(8),
            bank_batches=2, query_batches=2)
        rows = _run_two_epochs_with_csv(
            dino_model,
            tmp_path,
            [knn, keras.callbacks.CSVLogger(str(tmp_path / "log.csv"))],
        )

        header = rows[0]
        for key in knn.log_keys:
            assert key in header, (
                f"{key!r} is missing from the CSV header {header}. CSVLogger freezes "
                f"its fieldnames on the first epoch it sees, so this column can never "
                f"appear later either."
            )

        values = dict(zip(header, rows[1]))
        for k in DEFAULT_KS:
            accuracy = float(values[f"dino_knn_top1_k{k}"])
            assert 0.0 <= accuracy <= 1.0
        assert np.isfinite(float(values["dino_feat_mean_cos"]))
        assert float(values["dino_collapse_flag"]) in (0.0, 1.0)

    def test_a_callback_after_csvlogger_loses_every_column(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        """RED PROOF of the ORDERING rule, executed rather than asserted in prose."""
        knn = KNNEvalCallback(
            _labelled_dataset(9), _labelled_dataset(10),
            bank_batches=2, query_batches=2)
        rows = _run_two_epochs_with_csv(
            dino_model,
            tmp_path,
            [keras.callbacks.CSVLogger(str(tmp_path / "log.csv")), knn],
        )

        header = rows[0]
        assert not any(key in header for key in knn.log_keys), (
            f"expected every k-NN column to be ABSENT when the callback runs after "
            f"CSVLogger, but the header was {header}; if this fails the ordering rule "
            f"in create_callbacks is no longer load-bearing and its comment is wrong"
        )

    def test_a_skipped_epoch_still_writes_the_keys(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        """every_n_epochs=5 over 2 epochs: epoch 0 evaluates, epoch 1 writes `nan`.

        Omitting the keys on a skipped epoch would be indistinguishable in the CSV --
        but it also freezes them out of the fieldnames if the FIRST epoch is the
        skipped one, which is why the keys are unconditional.
        """
        knn = KNNEvalCallback(
            _labelled_dataset(11), _labelled_dataset(12),
            bank_batches=2, query_batches=2, every_n_epochs=5)
        rows = _run_two_epochs_with_csv(
            dino_model,
            tmp_path,
            [knn, keras.callbacks.CSVLogger(str(tmp_path / "log.csv"))],
        )

        header = rows[0]
        first = dict(zip(header, rows[1]))
        second = dict(zip(header, rows[2]))

        assert np.isfinite(float(first["dino_feat_mean_cos"])), (
            "epoch 0 must always be evaluated")
        assert second["dino_feat_mean_cos"] in ("nan", "NA"), (
            f"a skipped epoch must be marked, got {second['dino_feat_mean_cos']!r}")

    def test_a_degraded_entropy_is_distinguishable_from_a_skipped_one_in_the_csv(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        """D-039: the CSV alone must separate SKIPPED (nan) from DEGRADED (0.0).

        Both write ``nan`` into the entropy columns, so ``nan`` alone cannot tell
        them apart -- which is why ``dino_teacher_entropy_is_centered`` exists.
        """
        mismatched = DINOLoss(out_dim=OUT_DIM * 2)
        knn = KNNEvalCallback(
            _labelled_dataset(31), _labelled_dataset(32),
            bank_batches=2, query_batches=2, every_n_epochs=5,
            dino_loss=mismatched)
        rows = _run_two_epochs_with_csv(
            dino_model,
            tmp_path,
            [knn, keras.callbacks.CSVLogger(str(tmp_path / "log.csv"))],
        )

        header = rows[0]
        assert "dino_teacher_entropy_is_centered" in header, header
        evaluated = dict(zip(header, rows[1]))
        skipped = dict(zip(header, rows[2]))

        # Epoch 0 WAS evaluated, but the center is unusable -> 0.0, entropy nan.
        assert float(evaluated["dino_teacher_entropy_is_centered"]) == 0.0, (
            f"an evaluated epoch with an unusable center must be marked 0.0, got "
            f"{evaluated['dino_teacher_entropy_is_centered']!r}")
        assert evaluated["dino_teacher_entropy"] in ("nan", "NA"), (
            f"the entropy must be nan, not an uncentered softmax reported under "
            f"the centered name, got {evaluated['dino_teacher_entropy']!r}")
        assert np.isfinite(float(evaluated["dino_feat_mean_cos"])), (
            "the OTHER half of the diagnostic must still be computed")

        # Epoch 1 was SKIPPED -> nan, which is a different value from 0.0.
        assert skipped["dino_teacher_entropy_is_centered"] in ("nan", "NA"), (
            f"a skipped epoch must read nan, not 0.0, or the CSV cannot separate "
            f"'not evaluated' from 'evaluated but degraded'; got "
            f"{skipped['dino_teacher_entropy_is_centered']!r}")

    def test_logs_none_raises_rather_than_discarding_the_metrics(self) -> None:
        knn = KNNEvalCallback(
            _labelled_dataset(13), _labelled_dataset(14),
            bank_batches=1, query_batches=1)
        with pytest.raises(ValueError, match="logs=None"):
            knn.on_epoch_end(0, None)


# ---------------------------------------------------------------------------
# 6. the trainer wiring
# ---------------------------------------------------------------------------


class TestCallbackOrdering:
    """The trainer must INSERT the callback before CSVLogger, never append it."""

    def _callbacks(self, tmp_path: Any, **overrides: Any) -> List[Any]:
        config = trainer.TrainingConfig(
            dataset="cifar10",
            global_crop_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            variant="tiny",
            dino_out_dim=OUT_DIM,
            batch_size=BATCH,
            epochs=1,
            max_steps=1,
            knn_bank_batches=1,
            knn_query_batches=1,
            output_dir=str(tmp_path),
            **overrides,
        )
        loss = DINOLoss(out_dim=OUT_DIM)
        callbacks, _ = trainer.create_callbacks(
            config, loss, str(tmp_path / "run"), steps_per_epoch=1)
        return callbacks

    def test_the_knn_callback_precedes_the_csv_logger(self, tmp_path: Any) -> None:
        callbacks = self._callbacks(tmp_path)

        knn_index = next(
            i for i, c in enumerate(callbacks) if isinstance(c, KNNEvalCallback))
        csv_index = next(
            i for i, c in enumerate(callbacks)
            if isinstance(c, keras.callbacks.CSVLogger))

        assert knn_index < csv_index, (
            f"KNNEvalCallback is at index {knn_index} and CSVLogger at {csv_index}. "
            f"A callback that runs after CSVLogger writes into an already-serialized "
            f"logs dict and every k-NN column silently disappears from "
            f"training_log.csv."
        )

    def test_knn_eval_every_zero_removes_the_callback(self, tmp_path: Any) -> None:
        callbacks = self._callbacks(tmp_path, knn_eval_every=0)
        assert not any(isinstance(c, KNNEvalCallback) for c in callbacks)

    def test_the_callback_is_given_the_compiled_loss(self, tmp_path: Any) -> None:
        """Without the loss the entropy is of an UNCENTERED, UNSHARPENED softmax.

        That number is high for reasons unrelated to training and would make the
        entropy half of the collapse detector permanently silent.
        """
        callbacks = self._callbacks(tmp_path)
        knn = next(c for c in callbacks if isinstance(c, KNNEvalCallback))
        assert isinstance(knn.dino_loss, DINOLoss)


class TestZeroStepControl:
    """PROBE (vi): the run must produce its OWN random-init baseline.

    Before this hook existed the earliest k-NN number any shipped code produced was
    ``on_epoch_end(epoch=0)`` -- POST one full epoch -- and every "control" ever
    quoted (0.2754 / 0.2900 / 0.2910 / 0.2949) came from an uncommitted scratch
    script. A delta quoted against no baseline is the measured failure mode of this
    module, so the baseline now ships.
    """

    def _callback(self, tmp_path: Any, **overrides: Any) -> KNNEvalCallback:
        kwargs: Dict[str, Any] = dict(
            bank_batches=2,
            query_batches=2,
            control_json_path=tmp_path / "random_init_control.json",
            random_init_repeats=2,
        )
        kwargs.update(overrides)
        return KNNEvalCallback(
            _labelled_dataset(41), _labelled_dataset(42), **kwargs)

    def test_the_control_json_carries_every_key_with_its_spread(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        callback = self._callback(tmp_path, random_init_repeats=3)
        callback.set_model(dino_model)

        callback.on_train_begin()

        payload = json.loads((tmp_path / "random_init_control.json").read_text())
        assert payload["random_init_repeats"] == 3
        assert list(payload["ks"]) == list(DEFAULT_KS)
        assert "ZERO-OPTIMIZER-STEP" in payload["note"], (
            "the JSON must state on its face that it predates every update; a "
            "reader with only the run directory cannot otherwise tell it from an "
            "epoch row")
        for key in callback.log_keys:
            entry = payload["metrics"][key]
            assert len(entry["values"]) == 3, (
                f"{key} has {len(entry['values'])} per-repeat values, expected 3")
            for statistic in ("min", "max", "mean", "range"):
                assert statistic in entry, f"{key} is missing {statistic!r}"
            finite = [v for v in entry["values"] if np.isfinite(v)]
            if finite:
                assert entry["min"] == pytest.approx(min(finite))
                assert entry["max"] == pytest.approx(max(finite))
                assert entry["range"] == pytest.approx(max(finite) - min(finite))

    def test_a_dead_feature_extractor_collapses_the_control_to_chance(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        """DEAD-COMPONENT PROBE: constant features must NOT score above chance.

        RED-PROVEN: with the monkeypatch removed (i.e. the live extractor), the
        same assertion fails -- the real network scores well above chance on the
        separable fixture. That is what makes this test able to see a dead
        extractor at all; a shape-only assertion could not.
        """
        live = self._callback(tmp_path, random_init_repeats=1)
        live.set_model(dino_model)
        live.on_train_begin()
        live_score = json.loads(
            (tmp_path / "random_init_control.json").read_text()
        )["metrics"][f"dino_knn_top1_k{DEFAULT_KS[0]}"]["mean"]

        dead_path = tmp_path / "dead.json"
        dead = self._callback(
            dead_path.parent, random_init_repeats=1, control_json_path=dead_path)
        dead.set_model(dino_model)

        def _constant_extract(dataset: Any, n_batches: int) -> Tuple[Any, Any, Any]:
            n = BATCH * n_batches
            return (
                _collapsed(n),
                (np.arange(n) % N_CLASSES).astype("int64"),
                None,
            )

        dead._extract = _constant_extract  # type: ignore[assignment]
        dead.on_train_begin()

        payload = json.loads(dead_path.read_text())
        dead_score = payload["metrics"][f"dino_knn_top1_k{DEFAULT_KS[0]}"]["mean"]

        assert dead_score == pytest.approx(1.0 / N_CLASSES, abs=1e-9), (
            f"a feature extractor returning a CONSTANT must drive the k-NN to "
            f"chance ({1.0 / N_CLASSES}), got {dead_score}. If this passes with "
            f"the live extractor too, the control cannot see a dead component.")
        assert payload["metrics"]["dino_collapse_flag"]["mean"] == 1.0, (
            "constant features are collapse; the flag must fire in the control too")
        assert live_score > dead_score + 0.1, (
            f"NON-VACUITY: the live extractor scored {live_score} and the dead one "
            f"{dead_score}; if these are close the assertion above is decorative")

    def test_zero_repeats_writes_no_json_and_extracts_nothing(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        callback = self._callback(tmp_path, random_init_repeats=0)
        callback.set_model(dino_model)

        def _explode(dataset: Any, n_batches: int) -> Tuple[Any, Any, Any]:
            raise AssertionError(
                "random_init_repeats=0 must skip the probe ENTIRELY, not run it "
                "and discard the result -- the flag exists to buy back the cost")

        callback._extract = _explode  # type: ignore[assignment]
        callback.on_train_begin()

        assert not (tmp_path / "random_init_control.json").exists()

    def test_no_control_path_also_skips_the_probe(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        """Nowhere to write it => do not pay for it. Guards every other test's cost."""
        callback = self._callback(tmp_path, control_json_path=None)
        callback.set_model(dino_model)
        callback._extract = None  # type: ignore[assignment]
        callback.on_train_begin()
        assert not any(tmp_path.iterdir())

    def test_repeat_independence_is_measured_not_asserted(
            self, dino_model: keras.Model, tmp_path: Any) -> None:
        """D-005: the JSON must SAY whether repeats redrew, and be right about it.

        A cached probe dataset replays bit-identically, so a 0.0 range is a fact
        about the instrument's determinism, not evidence that the probe is
        noise-free. Reporting it without the flag would be a fake spread.
        """
        cached = _labelled_dataset(43).cache()
        callback = KNNEvalCallback(
            cached, _labelled_dataset(44), bank_batches=2, query_batches=2,
            control_json_path=tmp_path / "control.json", random_init_repeats=2)
        callback.set_model(dino_model)
        callback.on_train_begin()

        payload = json.loads((tmp_path / "control.json").read_text())
        assert len(payload["bank_fingerprints"]) == 2
        assert payload["repeats_are_independent"] is (
            len(set(payload["bank_fingerprints"])) == 2), (
            "the reported independence must agree with the fingerprints it is "
            "derived from")
        assert payload["repeats_are_independent"] is False, (
            f"a cached bank replays the SAME samples, so the repeats cannot be "
            f"independent; fingerprints were {payload['bank_fingerprints']}")
        assert payload["metrics"][f"dino_knn_top1_k{DEFAULT_KS[1]}"]["range"] == 0.0

    def test_a_non_dino_model_raises_at_train_begin(self, tmp_path: Any) -> None:
        """Edge case: the control must RAISE, never self-disable."""
        callback = self._callback(tmp_path)
        callback.set_model(keras.Sequential([keras.layers.Dense(2)]))

        with pytest.raises(ValueError, match="no `student` attribute"):
            callback.on_train_begin()
        assert not (tmp_path / "random_init_control.json").exists()


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"knn_eval_every": -1}, "knn_eval_every must be >= 0"),
            ({"knn_bank_batches": 0}, "must be positive"),
            ({"knn_query_batches": 0}, "must be positive"),
            ({"knn_temperature": 0.0}, "knn_temperature must be positive"),
            ({"random_init_repeats": -1}, "random_init_repeats must be >= 0"),
        ],
    )
    def test_bad_knn_config_raises_on_the_message(
            self, overrides: Dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            trainer.TrainingConfig(**overrides)


# ---------------------------------------------------------------------
