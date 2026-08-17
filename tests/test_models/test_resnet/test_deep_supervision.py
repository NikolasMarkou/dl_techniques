"""Deep supervision must actually supervise something the main head does not.

Guard for the plan-2026-08-17T183311-79c63e38 step-7 defect. `ResNet` built its
supervision heads over ``range(1, len(blocks_per_stage))``, while ``call``
appends one entry to ``stage_features`` per stage — so the head at
``stage_idx == len - 1`` read the **exact tensor** ``self.gap`` /
``self.classifier`` consume for ``final_output``. That head was a second,
redundant classifier on the main objective, and stage 0 — the only stage for
which deep supervision shortens the backpropagation path at all — was never
supervised.

The property under test is a GRADIENT property, not a shape or count one: an
auxiliary head that reads the post-final-stage tensor has a nonzero gradient with
respect to the last stage's weights. A head reading an earlier stage has exactly
zero. `tests/test_models/test_resnet/` contained no occurrence of
``deep_supervision`` before this file.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.resnet.model import ResNet
from dl_techniques.optimization.deep_supervision import schedule_builder


BLOCKS = [1, 1, 1, 1]
# 64 at stage 0 matches the stem's fixed 64-channel output. A narrower stage 0
# under ``block_type="basic"`` needs a shortcut projection that BasicBlock
# creates lazily inside ``call``, which Keras 3 refuses once the model is built
# ("You cannot add new elements of state ... that is already built"). That is a
# pre-existing ResNet defect, independent of deep supervision (it reproduces at
# ``enable_deep_supervision=False``); it is dodged here, not fixed.
FILTERS = [64, 128, 256, 512]
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 5


def _model(**kwargs) -> ResNet:
    """A ResNet that has already been called once.

    The blocks are subclassed layers built lazily inside ``call``, so
    ``block.trainable_weights`` is EMPTY until a forward pass has happened --
    collecting weights from a fresh instance makes every gradient probe below
    vacuously pass (measured: the anti-vacuity assertion fired first).
    """
    model = ResNet(
        num_classes=NUM_CLASSES,
        blocks_per_stage=list(BLOCKS),
        filters_per_stage=list(FILTERS),
        block_type="basic",
        enable_deep_supervision=True,
        input_shape=INPUT_SHAPE,
        **kwargs,
    )
    model(tf.zeros((1,) + INPUT_SHAPE), training=False)
    return model


def _last_stage_weights(model: ResNet):
    """Every trainable weight belonging to the final stage's blocks."""
    weights = []
    for block in model.stages[-1]:
        weights.extend(block.trainable_weights)
    assert weights, "final stage has no trainable weights; the probe is vacuous"
    return weights


def _first_stage_weights(model: ResNet):
    weights = []
    for block in model.stages[0]:
        weights.extend(block.trainable_weights)
    assert weights, "first stage has no trainable weights; the probe is vacuous"
    return weights


class TestSupervisionHeadsDoNotDuplicateTheMainHead:

    def test_no_supervision_head_reads_the_final_stage_tensor(self):
        """The gradient of every auxiliary output w.r.t. the LAST stage's
        weights must be exactly zero. Before the fix, the stage-4 head made one
        of them nonzero — it consumed the same tensor as the main head."""
        model = _model()
        x = tf.constant(np.random.rand(2, *INPUT_SHAPE).astype("float32"))
        last_weights = _last_stage_weights(model)

        with tf.GradientTape(persistent=True) as tape:
            outputs = model(x, training=False)
            aux_losses = [tf.reduce_sum(o) for o in outputs[1:]]

        assert len(aux_losses) >= 1
        for idx, loss in enumerate(aux_losses):
            grads = tape.gradient(loss, last_weights)
            total = sum(
                0.0 if g is None else float(np.abs(keras.ops.convert_to_numpy(g)).sum())
                for g in grads
            )
            assert total == 0.0, (
                f"supervision output {idx + 1} has gradient {total} into the "
                f"final stage: it is reading the same tensor as the main head"
            )
        del tape

    def test_the_main_head_does_depend_on_the_final_stage(self):
        """Anti-vacuity for the assertion above: the zero is a property of the
        auxiliary heads, not of the probe."""
        model = _model()
        x = tf.constant(np.random.rand(2, *INPUT_SHAPE).astype("float32"))
        last_weights = _last_stage_weights(model)

        with tf.GradientTape() as tape:
            final_loss = tf.reduce_sum(model(x, training=False)[0])
        grads = tape.gradient(final_loss, last_weights)
        total = sum(
            0.0 if g is None else float(np.abs(keras.ops.convert_to_numpy(g)).sum())
            for g in grads
        )
        assert total > 0.0

    def test_the_shallowest_stage_is_supervised(self):
        """Stage 0 must carry a head: it is the only stage for which deep
        supervision shortens the backprop path. The last auxiliary output (the
        list is reversed, deepest first) must reach stage 0's weights."""
        model = _model()
        assert 0 in [h["stage_idx"] for h in model.supervision_heads]

        x = tf.constant(np.random.rand(2, *INPUT_SHAPE).astype("float32"))
        first_weights = _first_stage_weights(model)
        with tf.GradientTape() as tape:
            shallowest_aux = tf.reduce_sum(model(x, training=False)[-1])
        grads = tape.gradient(shallowest_aux, first_weights)
        total = sum(
            0.0 if g is None else float(np.abs(keras.ops.convert_to_numpy(g)).sum())
            for g in grads
        )
        assert total > 0.0

    def test_the_shallowest_head_ignores_every_deeper_stage(self):
        """The stage-0 head must not depend on stages 1..3 — otherwise it is not
        a shallow supervision signal at all."""
        model = _model()
        x = tf.constant(np.random.rand(2, *INPUT_SHAPE).astype("float32"))
        deeper = []
        for stage in model.stages[1:]:
            for block in stage:
                deeper.extend(block.trainable_weights)

        with tf.GradientTape() as tape:
            shallowest_aux = tf.reduce_sum(model(x, training=False)[-1])
        grads = tape.gradient(shallowest_aux, deeper)
        total = sum(
            0.0 if g is None else float(np.abs(keras.ops.convert_to_numpy(g)).sum())
            for g in grads
        )
        assert total == 0.0

    @pytest.mark.parametrize("blocks", [[1, 1], [1, 1, 1], [1, 1, 1, 1]])
    def test_head_count_is_stages_minus_one(self, blocks):
        model = ResNet(
            num_classes=NUM_CLASSES,
            blocks_per_stage=list(blocks),
            filters_per_stage=[64 * 2 ** i for i in range(len(blocks))],
            block_type="basic",
            enable_deep_supervision=True,
            input_shape=INPUT_SHAPE,
        )
        assert len(model.supervision_heads) == len(blocks) - 1
        assert [h["stage_idx"] for h in model.supervision_heads] == list(
            range(len(blocks) - 1)
        )

    def test_documented_output_order_is_deepest_supervision_first(self):
        """The docstrings promise ``[final, stage3, stage2, stage1]``; the layer
        names must actually come out in that order."""
        model = _model()
        names = [
            h["classifier"].name for h in reversed(model.supervision_heads)
        ]
        assert names == [
            "supervision_classifier_stage3",
            "supervision_classifier_stage2",
            "supervision_classifier_stage1",
        ]

    def test_output_count_matches_the_schedule_width_the_trainer_builds(self):
        """`src/train/resnet/train_resnet.py` derives ``num_outputs`` from
        ``len(model.output)`` and hands it to ``schedule_builder(...,
        invert_order=True)``, which SIZES its per-epoch weight array from it.
        `DeepSupervisionWeightScheduler` has no coverage of its own; this pins
        the coupling."""
        model = _model()
        x = tf.constant(np.random.rand(2, *INPUT_SHAPE).astype("float32"))
        num_outputs = len(model(x, training=False))
        assert num_outputs == len(BLOCKS)  # 1 main + (len - 1) supervision

        scheduler = schedule_builder(
            {"type": "step_wise", "config": {}}, num_outputs, invert_order=True
        )
        for progress in (0.0, 0.5, 1.0):
            weights = scheduler(progress)
            assert weights.shape == (num_outputs,)
            np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-5, atol=1e-5)
