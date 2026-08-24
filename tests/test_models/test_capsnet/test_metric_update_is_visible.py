"""C-45(b)/(e): a dropped metric must be VISIBLE, and the docstring must be true.

MEASURED while writing this: ``self.metrics`` yields Keras' ``CompileMetrics``
WRAPPER rather than the individual metrics, so a single misshaped metric took
**every** compiled metric down with it -- worse than the finding described. The
warning therefore names the wrapper's CONTENTS.

(b) ``train_step`` and ``test_step`` wrapped ``metric.update_state(y,
outputs["length"])`` in a bare ``except: continue``. Any metric whose signature
or shape did not match was silently dropped from training AND from the returned
logs, with no warning -- and a bare ``except`` also swallows
``KeyboardInterrupt`` and ``SystemExit``.

(e) ``model_v2``'s ``stem_pretrained`` docstring promised a "graceful fallback
to random init on download failure" while
``dl_techniques.models.resnet.create_resnet`` raises unconditionally.
"""

import inspect

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.capsnet.model import CapsNet
from dl_techniques.models.capsnet import model_v2


class _MisshapedMetric(keras.metrics.Metric):
    """A metric that cannot consume `(y, outputs["length"])`."""

    def __init__(self, name: str = "misshaped", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # `y_pred` here is (batch, num_classes); demand a rank it never has.
        raise ValueError("expected rank 4, got rank 2")

    def result(self):
        return self.total


@pytest.fixture
def tiny_capsnet() -> CapsNet:
    return CapsNet(
        num_classes=3,
        input_shape=(28, 28, 1),
        conv_filters=[8],
        primary_capsules=2,
        primary_capsule_dim=4,
        digit_capsule_dim=4,
        routing_iterations=1,
        reconstruction=False,
    )


class TestAMisshapedMetricIsReported:

    def test_it_warns_naming_the_metric(self, tiny_capsnet, caplog):
        metric = _MisshapedMetric()
        tiny_capsnet.compile(optimizer="adam", metrics=[metric])

        x = np.random.rand(4, 28, 28, 1).astype("float32")
        y = keras.utils.to_categorical(
            np.random.randint(0, 3, size=(4,)), num_classes=3)

        with caplog.at_level("WARNING"):
            tiny_capsnet.fit(x, y, epochs=1, batch_size=4, verbose=0)

        assert "SKIPPED" in caplog.text, (
            "ASSERT-DROPPED-METRIC-IS-VISIBLE: the metric was skipped from "
            "training and from the logs with no warning at all."
        )
        # MEASURED: `self.metrics` yields Keras' `CompileMetrics` WRAPPER, so
        # one misshaped metric takes the whole container down -- the warning
        # must therefore name the CONTENTS, not just 'compile_metrics'.
        assert "misshaped" in caplog.text, (
            "ASSERT-WARNING-NAMES-THE-CULPRIT: a warning about "
            "'compile_metrics' does not tell the user which metric broke."
        )
        assert "expected rank 4" in caplog.text

    def test_the_run_still_completes(self, tiny_capsnet):
        """Fail-soft, not fail-hard: a multi-hour run must not abort."""
        tiny_capsnet.compile(optimizer="adam", metrics=[_MisshapedMetric()])
        x = np.random.rand(4, 28, 28, 1).astype("float32")
        y = keras.utils.to_categorical(
            np.random.randint(0, 3, size=(4,)), num_classes=3)

        history = tiny_capsnet.fit(x, y, epochs=1, batch_size=4, verbose=0)
        assert np.all(np.isfinite(history.history["loss"]))

    def test_a_well_shaped_metric_is_not_dropped(self, tiny_capsnet, caplog):
        """Anti-vacuity: the normal path warns about nothing."""
        tiny_capsnet.compile(
            optimizer="adam", metrics=[keras.metrics.CategoricalAccuracy()])
        x = np.random.rand(4, 28, 28, 1).astype("float32")
        y = keras.utils.to_categorical(
            np.random.randint(0, 3, size=(4,)), num_classes=3)

        with caplog.at_level("WARNING"):
            history = tiny_capsnet.fit(x, y, epochs=1, batch_size=4, verbose=0)

        assert "SKIPPED" not in caplog.text
        assert any("categorical_accuracy" in key
                   for key in history.history), history.history.keys()

    def test_the_handler_is_narrow(self):
        """A bare `except` also swallows KeyboardInterrupt / SystemExit."""
        source = inspect.getsource(CapsNet._update_metrics)
        assert "except:" not in source, (
            "ASSERT-NARROW-EXCEPT: a bare `except:` is back."
        )
        assert "except (" in source


class TestTheStemPretrainedDocstringIsTrue:

    def test_it_no_longer_promises_a_graceful_fallback(self):
        docstring = model_v2.CapsNetV2.__doc__ or ""
        assert "graceful fallback to random" not in docstring, (
            "ASSERT-NO-FALSE-FALLBACK-PROMISE: resnet/model.py raises "
            "unconditionally; there is no fallback to be graceful about."
        )

    def test_it_says_what_actually_happens(self):
        docstring = model_v2.CapsNetV2.__doc__ or ""
        assert "NotImplementedError" in docstring
