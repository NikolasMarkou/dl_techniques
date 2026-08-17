"""`evaluate()` must not reconstruct from the labels, and the loss tracker must
track the loss.

Guard for plan-2026-08-17T183311-79c63e38 step 12 (prior review finding G-11,
which survived a 139-commit repair round unfixed).

1. ``test_step`` called ``self(x, training=False, mask=y)``. ``_reconstruct``
   takes ``reconstruction_mask = mask`` whenever a mask is given, so the
   inference branch — ``one_hot(argmax(lengths))`` — was unreachable from
   ``test_step`` and the reconstruction loss reported by ``evaluate()`` was
   teacher-forced, i.e. optimistic by construction and unattainable at inference.
   ``train_step`` passing ``mask=y`` is the paper's recipe and is unchanged.
2. ``_update_metrics`` iterated ``self.metrics``, whose first entry is Keras'
   ``_loss_tracker`` (a ``keras.metrics.Mean`` named "loss"). The
   ``isinstance(metric, CapsuleAccuracy)`` guard does not match it, so it got
   ``Mean.update_state(values=y_onehot, sample_weight=lengths)`` — both
   ``(B, num_classes)``, so NO exception fires and the tracker silently
   accumulated ``mean(y * lengths)``. Measured before the fix:
   ``model.metrics[0].result()`` was ``0.3082`` while the reported loss was
   ``0.7796``.
"""

import os

import keras
import numpy as np
import pytest

from dl_techniques.models.capsnet.model import CapsNet


NUM_CLASSES = 3
INPUT_SHAPE = (28, 28, 1)


@pytest.fixture
def model() -> CapsNet:
    keras.utils.set_random_seed(0)
    m = CapsNet(
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        conv_filters=[16],
        primary_capsules=4,
        primary_capsule_dim=4,
        digit_capsule_dim=8,
        reconstruction=True,
    )
    m.compile(optimizer="adam")
    return m


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.random((8, *INPUT_SHAPE)).astype("float32")
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    permuted = np.array([2, 0, 1, 2, 0, 1, 2, 0])
    to_onehot = lambda a: np.array(keras.ops.one_hot(a, NUM_CLASSES), dtype="float32")
    return x, to_onehot(labels), to_onehot(permuted)


class TestEvaluateIsNotTeacherForced:

    def test_reconstruction_loss_does_not_depend_on_the_labels(self, model, data):
        """The same inputs with DIFFERENT labels must give the identical
        reconstruction loss: at evaluation the decoder is driven by the model's
        own prediction. Measured before the fix: the two differed by 9.8e-06."""
        x, y_true, y_permuted = data
        first = model.evaluate(x, y_true, verbose=0, return_dict=True)
        second = model.evaluate(x, y_permuted, verbose=0, return_dict=True)
        assert float(first["reconstruction_loss"]) == float(
            second["reconstruction_loss"]
        )

    def test_the_mask_is_not_inert(self, model, data):
        """Anti-vacuity for the test above: masking DOES change the
        reconstruction at these weights, so the exact equality is a property of
        ``test_step``, not of a mask that does nothing."""
        x, y_true, _ = data
        masked = model(x, training=False, mask=y_true)["reconstructed"]
        unmasked = model(x, training=False)["reconstructed"]
        delta = float(
            np.abs(
                keras.ops.convert_to_numpy(masked)
                - keras.ops.convert_to_numpy(unmasked)
            ).max()
        )
        assert delta > 0.0

    def test_margin_loss_still_uses_the_labels(self, model, data):
        """The leak fix must not make ``evaluate()`` blind to the labels
        entirely — the margin loss is a supervised term and must still move."""
        x, y_true, y_permuted = data
        first = model.evaluate(x, y_true, verbose=0, return_dict=True)
        second = model.evaluate(x, y_permuted, verbose=0, return_dict=True)
        assert float(first["margin_loss"]) != float(second["margin_loss"])


class TestLossTrackerTracksTheLoss:

    def test_first_metric_is_the_loss_and_holds_the_loss(self, model, data):
        x, y_true, _ = data
        results = model.evaluate(x, y_true, verbose=0, return_dict=True)
        tracker = model.metrics[0]
        assert tracker.name == "loss"
        assert float(tracker.result()) == pytest.approx(
            float(results["loss"]), rel=1e-6
        )

    def test_fit_reports_a_loss_that_matches_the_tracker(self, model, data):
        x, y_true, _ = data
        history = model.fit(x, y_true, epochs=1, verbose=0)
        assert float(history.history["loss"][0]) == pytest.approx(
            float(model.metrics[0].result()), rel=1e-6
        )


class TestSaveModelHasNoDeprecatedSaveFormat:

    def test_save_format_is_gone_from_the_signature(self, model, data, tmp_path):
        x, _, _ = data
        model(x[:1])
        with pytest.raises(TypeError):
            model.save_model(str(tmp_path / "m.keras"), save_format="keras")

    def test_saving_to_a_keras_path_works(self, model, data, tmp_path):
        x, _, _ = data
        model(x[:1])
        path = tmp_path / "nested" / "m.keras"
        model.save_model(str(path))
        assert os.path.exists(path)

    def test_an_unclassifiable_path_raises_about_the_extension(
        self, model, data, tmp_path
    ):
        """With ``save_format`` forwarded, Keras 3 raised about the DEPRECATED
        ARGUMENT here, naming the wrong cause. Now it names the path."""
        x, _, _ = data
        model(x[:1])
        # Match the EXTENSION message specifically: the pre-fix deprecation
        # error also mentioned ".keras", so a looser regex passed both ways.
        with pytest.raises(ValueError, match=r"Invalid filepath extension"):
            model.save_model(str(tmp_path / "no_extension"))
