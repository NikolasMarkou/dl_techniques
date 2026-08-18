"""Guards for `DepthAnything`'s custom `train_step` (plan step 18).

`depth_anything` is a *sanctioned* exception to the repo's "no custom
`train_step`" rule (the semi-supervised path is dual-batch with asymmetric
augmentation plus a teacher outside the loss graph, which
`compute_loss(x, y, y_pred, sample_weight, training)` cannot carry). The price
of the exception is that everything Keras' default `train_step` does for free
must be done by hand here, and it was not. These tests pin the three things
that were wrong:

1. `self._loss_tracker` was never fed, so `history.history["loss"]` was the
   `Mean` metric's reset default `0.0` on every step of every run -- and any
   `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau` monitoring `"loss"`
   was dead. The pre-existing smoke tests asserted only `np.isfinite(loss)`,
   and `0.0` is finite.
2. `{m.name: m.result() for m in self.metrics}` nests a **dict** under the key
   `"compile_metrics"`, because `self.metrics` yields Keras' `CompileMetrics`
   *container*, not its contents.
3. The pseudo-label consistency term (and the teacher EMA machinery it needs)
   was gated on `use_feature_alignment`, a knob independent of
   `enable_semi_supervised`, so the documented combination
   `enable_semi_supervised=True, use_feature_alignment=False` unpacked the
   unlabeled batch and then used it for nothing.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.depth_anything import DepthAnything


IMAGE_SHAPE = (32, 32, 3)


def _make_model(**kwargs) -> DepthAnything:
    """A tiny placeholder-encoder DepthAnything, built, with augmentation off.

    The placeholder Conv-BN-ReLU encoder keeps these tests CPU-cheap; the
    defects under test live in `train_step`, which is encoder-agnostic.
    Augmentation is disabled so a two-run weight comparison is deterministic.
    """
    m = DepthAnything(
        encoder_kind="placeholder",
        encoder_type="vit_s",
        image_shape=IMAGE_SHAPE,
        **kwargs,
    )
    _ = m(keras.ops.zeros((1,) + IMAGE_SHAPE))
    m.augmentation = None
    return m


def _labeled_data(batch: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, *IMAGE_SHAPE)).astype("float32")
    y = rng.standard_normal((batch, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1)).astype(
        "float32"
    )
    return x, y


class TestReportedLoss:
    """`history.history["loss"]` must be the loss that was optimized."""

    def test_reported_loss_is_not_the_reset_default(self):
        keras.utils.set_random_seed(1234)
        m = _make_model(use_feature_alignment=False)
        x, y = _labeled_data()
        m.compile(optimizer=keras.optimizers.SGD(1e-3))
        history = m.fit(x, y, epochs=2, batch_size=2, verbose=0)

        losses = [float(v) for v in history.history["loss"]]
        assert all(np.isfinite(v) for v in losses), losses
        assert losses[0] != 0.0, (
            "history.history['loss'] is the loss tracker's reset default -- "
            f"the tracker is never fed. Got {losses}"
        )
        assert losses[0] != losses[1], (
            f"reported loss is constant across epochs: {losses}"
        )

    def test_semi_supervised_reported_loss_is_not_the_reset_default(self):
        keras.utils.set_random_seed(1234)
        m = _make_model(use_feature_alignment=True, enable_semi_supervised=True)
        x_lab, y_lab = _labeled_data(seed=0)
        x_unlab, _ = _labeled_data(seed=1)

        def gen():
            for _ in range(2):
                yield (x_lab, x_unlab), y_lab

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (
                    tf.TensorSpec((2,) + IMAGE_SHAPE, tf.float32),
                    tf.TensorSpec((2,) + IMAGE_SHAPE, tf.float32),
                ),
                tf.TensorSpec((2, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1), tf.float32),
            ),
        )
        m.compile(optimizer=keras.optimizers.SGD(1e-3))
        history = m.fit(ds, epochs=2, steps_per_epoch=1, verbose=0)

        losses = [float(v) for v in history.history["loss"]]
        assert losses[0] != 0.0, (
            f"semi-supervised path never feeds the loss tracker: {losses}"
        )
        assert losses[0] != losses[1], (
            f"semi-supervised reported loss is constant: {losses}"
        )


class TestLogsAreFlat:
    """The dict `train_step` RETURNS must be flat scalars, not nested dicts.

    Note carefully what this can and cannot observe. Keras' callback path runs
    every logs dict through `keras/src/utils/python_utils.py::pythonify_logs`,
    which recursively splices nested dicts into the parent and DISCARDS the
    outer key -- so `history.history` never showed a `"compile_metrics"` key
    even while `train_step` was returning one. The nesting is only visible to a
    caller invoking `model.train_step(batch)` directly, which is where this
    test looks. Asserting on `history.history` here would pass identically
    before and after the fix.
    """

    def test_raw_train_step_return_is_flat(self):
        keras.utils.set_random_seed(1234)
        m = _make_model(use_feature_alignment=False)
        x, y = _labeled_data()
        m.compile(
            optimizer=keras.optimizers.SGD(1e-3),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        logs = m.train_step((x, y))

        nested = {k: v for k, v in logs.items() if isinstance(v, dict)}
        assert not nested, (
            f"train_step returned nested dict(s): {sorted(nested)}"
        )
        assert "compile_metrics" not in logs, sorted(logs)
        assert "mae" in logs, sorted(logs)
        assert "loss" in logs, sorted(logs)

    def test_history_keys_are_scalars(self):
        keras.utils.set_random_seed(1234)
        m = _make_model(use_feature_alignment=False)
        x, y = _labeled_data()
        m.compile(
            optimizer=keras.optimizers.SGD(1e-3),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        history = m.fit(x, y, epochs=1, batch_size=2, verbose=0)
        assert "mae" in history.history, sorted(history.history)
        for key, values in history.history.items():
            assert np.ndim(values[0]) == 0, (
                f"log entry {key!r} is not a scalar: {values[0]!r}"
            )


class TestUnlabeledBatchIsUsed:
    """`enable_semi_supervised=True` must consume `x_unlab` on its own."""

    @staticmethod
    def _one_step_weights(model, x_lab, x_unlab, y_lab, initial):
        model.set_weights(initial)

        def gen():
            yield (x_lab, x_unlab), y_lab

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (
                    tf.TensorSpec((2,) + IMAGE_SHAPE, tf.float32),
                    tf.TensorSpec((2,) + IMAGE_SHAPE, tf.float32),
                ),
                tf.TensorSpec((2, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1), tf.float32),
            ),
        )
        model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
        return [np.asarray(w) for w in model.get_weights()]

    @pytest.mark.parametrize("use_feature_alignment", [True, False])
    def test_unlabeled_batch_influences_the_gradient(self, use_feature_alignment):
        """Swapping ONLY `x_unlab` must move the student's weights.

        The comparison is made AGAINST A CONTROL, not against zero. A bare
        `max|w_a - w_b| > 0` assertion is a false GREEN on GPU: re-running the
        identical step twice already differs by ~2e-7 from nondeterministic
        reduction order, which is enough to pass a `> 0` check while the
        semi-supervised branch is completely inert (measured at `b0914e836`).
        So the floor is measured in-test by re-running with the SAME unlabeled
        batch, and the swap must beat it by orders of magnitude.

        SGD is used deliberately: MEMORY records that Adam's normalization
        makes a total-|dW| probe nearly blind (~0.9x) where SGD reads the same
        signal at ~26x.
        """
        keras.utils.set_random_seed(1234)
        m = _make_model(
            use_feature_alignment=use_feature_alignment,
            enable_semi_supervised=True,
        )
        x_lab, y_lab = _labeled_data(seed=0)
        x_unlab_a, _ = _labeled_data(seed=1)
        x_unlab_b, _ = _labeled_data(seed=2)

        m.compile(optimizer=keras.optimizers.SGD(1e-2))
        initial = [np.asarray(w) for w in m.get_weights()]

        w_a = self._one_step_weights(m, x_lab, x_unlab_a, y_lab, initial)
        w_a2 = self._one_step_weights(m, x_lab, x_unlab_a, y_lab, initial)
        w_b = self._one_step_weights(m, x_lab, x_unlab_b, y_lab, initial)

        def _max_abs(u, v):
            return max(float(np.max(np.abs(p - q))) for p, q in zip(u, v))

        floor = _max_abs(w_a, w_a2)  # nondeterminism only
        signal = _max_abs(w_a, w_b)  # nondeterminism + the unlabeled batch

        assert signal > max(100.0 * floor, 1e-6), (
            "the unlabeled batch had no measurable influence on the update at "
            f"use_feature_alignment={use_feature_alignment}: swap delta "
            f"{signal:.3e} vs same-input nondeterminism floor {floor:.3e}"
        )


class TestDegenerateConfigIsAnnounced:
    """A configuration that used to be silently inert must not be silent."""

    def test_semi_supervised_without_feature_alignment_still_builds_teacher(self):
        m = _make_model(use_feature_alignment=False, enable_semi_supervised=True)
        assert m.frozen_encoder is not None, (
            "enable_semi_supervised=True needs a teacher for the pseudo-label "
            "consistency term, but none was built"
        )
        assert m.use_feature_alignment is False, (
            "building the teacher must not silently re-enable the FAL term"
        )

    def test_teacher_ema_moves_without_feature_alignment(self):
        m = _make_model(use_feature_alignment=False, enable_semi_supervised=True)
        before = [np.asarray(w) for w in m.frozen_encoder.get_weights()]
        # Perturb the student so an EMA step has something to move toward.
        m.encoder.set_weights(
            [np.asarray(w) + 1.0 for w in m.encoder.get_weights()]
        )
        m.update_teacher_ema(decay=0.5)
        after = [np.asarray(w) for w in m.frozen_encoder.get_weights()]
        moved = max(float(np.max(np.abs(a - b))) for a, b in zip(before, after))
        assert moved > 0.0, (
            "update_teacher_ema is a no-op when use_feature_alignment is False, "
            "so the teacher stays pinned for the whole run"
        )
