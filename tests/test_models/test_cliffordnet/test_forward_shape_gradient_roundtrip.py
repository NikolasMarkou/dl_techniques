"""F-54: the four tests ``test_cliffordnet/`` never had.

Before this file the whole package was covered by ``test_model.py`` — 61 lines,
3 tests, asserting only that ``create_cliffordnet`` returns an instance, that
``model.built`` is True, and that ``__all__`` has 2 names. **No forward pass on
real data, no output-shape assertion, no gradient-flow test, no
``save``/``load_model``.**

The round-trip is the reason this file exists in this shape. ``CliffordNet``
stores its blocks as ``List[Dict[str, Layer]]``
(``self.blocks_list.append({"block": ..., "drop_path": ...})``), and a ``dict``
inside a ``list`` is not directly Keras-trackable — if the saver's container
traversal skips it, the restored model has weights that match in **count, path
AND shape** while being freshly initialized. Every cheap round-trip assertion
(``len(model.weights)``, the weight-name list, ``count_params()``) is satisfied
identically under that failure. Only a VALUE comparison sees it, so that is what
``test_round_trip_restores_weight_VALUES_not_just_shapes`` does, and it does it
on weights that have been PERTURBED away from their initializer first — a
round-trip on freshly-initialized weights would be satisfied by a model that
dropped them and re-ran the same deterministic initializer.

**The suspected trap does NOT bite.** MEASURED at ``38c7493c6``: all 40 weights
of a ``channels=16, depth=2`` model come back bit-identical after
``save``/``load_model``, blocks included. The tracker's ``_layers`` mirror does
reach the dicts. F-54's "CONFIRMED test gap" half was real; its "SUSPECTED trap"
half is refuted, and this file is the standing detector if the container is ever
changed.

**The detector is not vacuous**, verified by a dead-component injection that
deletes weight RESTORATION for the blocks. Note that the obvious form of that
injection is inert: ``CliffordNetBlock`` holds **zero own variables** (all 15 of
its weights live in sublayers), so stubbing ``load_own_variables`` on the block
class changes nothing — a vacuous injection that "passes" for the wrong reason.
Stubbing it on every layer whose path contains ``clifford_block`` is the live
form, and it reproduces the trap's exact signature: **paths equal, param totals
equal, 18 of 40 weight VALUES wrong, outputs differ.** Every value assertion
below goes RED under it; the count/path/param assertions stay GREEN, which is
precisely why they are not the test.
"""

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.cliffordnet.model import CliffordNet

IMG = (16, 16, 3)
NUM_CLASSES = 4
BATCH = 2


def _tiny() -> CliffordNet:
    """A CliffordNet small enough to save/load repeatedly in a unit test."""
    return CliffordNet(
        num_classes=NUM_CLASSES,
        channels=16,
        depth=2,
        patch_size=2,
        shifts=[1, 2],
        stochastic_depth_rate=0.0,
        dropout_rate=0.0,
    )


def _x(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(
        size=(BATCH,) + IMG
    ).astype("float32")


def _np(t) -> np.ndarray:
    return keras.ops.convert_to_numpy(t)


class TestCliffordNetForward:

    def test_forward_pass_on_real_data(self):
        model = _tiny()
        y = _np(model(_x(1), training=False))

        assert np.isfinite(y).all(), "forward pass produced non-finite logits"
        # Non-vacuity: a model that annihilated its signal would emit a
        # constant. This repo has measured exactly that failure in a
        # transform-only Clifford block wired without its external residual.
        assert float(np.std(y)) > 0.0, "logits are constant across the batch"

    def test_output_shape(self):
        model = _tiny()
        y = _np(model(_x(2), training=False))
        assert y.shape == (BATCH, NUM_CLASSES)
        assert model.compute_output_shape((BATCH,) + IMG) == (BATCH, NUM_CLASSES)

    def test_training_and_inference_paths_both_run(self):
        model = _tiny()
        x = _x(3)
        assert np.isfinite(_np(model(x, training=True))).all()
        assert np.isfinite(_np(model(x, training=False))).all()


class TestCliffordNetGradientFlow:

    def test_every_trainable_weight_receives_a_gradient(self):
        model = _tiny()
        x = tf.convert_to_tensor(_x(4))
        labels = tf.one_hot([0, 1], NUM_CLASSES)

        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(
                keras.losses.categorical_crossentropy(
                    labels, logits, from_logits=True
                )
            )
        grads = tape.gradient(loss, model.trainable_weights)

        assert len(model.trainable_weights) > 0
        dead = [
            w.path
            for w, g in zip(model.trainable_weights, grads)
            if g is None or float(np.max(np.abs(_np(g)))) == 0.0
        ]
        assert not dead, f"{len(dead)} trainable weights got no gradient: {dead}"

    def test_the_blocks_are_on_the_gradient_path(self):
        """Named separately: the blocks are the container under suspicion."""
        model = _tiny()
        x = tf.convert_to_tensor(_x(5))

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model(x, training=True) ** 2)

        block_weights = [
            w
            for info in model.blocks_list
            for w in info["block"].trainable_weights
        ]
        assert block_weights, "the blocks hold no trainable weights at all"
        grads = tape.gradient(loss, block_weights)
        assert all(g is not None for g in grads)
        assert any(float(np.max(np.abs(_np(g)))) > 0.0 for g in grads)


class TestCliffordNetRoundTrip:
    """The ``List[Dict[str, Layer]]`` container, checked by VALUE."""

    @staticmethod
    def _perturb(model):
        """Move every weight off its initializer, deterministically.

        A round-trip test on freshly-initialized weights is satisfied by a
        model that dropped them and re-ran the same initializer — the exact
        failure this class exists to detect.
        """
        rng = np.random.default_rng(7)
        for w in model.weights:
            w.assign(
                keras.ops.convert_to_tensor(
                    (_np(w) + rng.normal(size=w.shape) * 0.05).astype(
                        w.dtype
                    )
                )
            )

    def test_round_trip_restores_weight_VALUES_not_just_shapes(self, tmp_path):
        model = _tiny()
        x = _x(6)
        model(x, training=False)  # ensure fully built
        self._perturb(model)

        before = _np(model(x, training=False))
        path = os.path.join(str(tmp_path), "cliffordnet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = _np(loaded(x, training=False))

        by_path = {w.path: _np(w) for w in model.weights}
        loaded_by_path = {w.path: _np(w) for w in loaded.weights}

        # The cheap assertions FIRST, so a failure report says which level
        # broke. These three are exactly what the suspected trap preserves.
        assert set(by_path) == set(loaded_by_path), "weight paths differ"
        assert model.count_params() == loaded.count_params()

        mismatched = [
            p
            for p in by_path
            if not np.allclose(by_path[p], loaded_by_path[p], atol=0, rtol=0)
        ]
        assert not mismatched, (
            f"{len(mismatched)} of {len(by_path)} weights came back with "
            f"DIFFERENT VALUES while count, path and shape all matched — the "
            f"List[Dict[str, Layer]] container did not round-trip: "
            f"{mismatched[:5]}"
        )
        np.testing.assert_allclose(
            before, after, atol=1e-5,
            err_msg="outputs differ after a .keras round trip",
        )

    def test_the_block_weights_specifically_survive(self, tmp_path):
        """Named separately so a failure points straight at the container."""
        model = _tiny()
        x = _x(8)
        model(x, training=False)
        self._perturb(model)

        path = os.path.join(str(tmp_path), "cliffordnet_blocks.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        for i, (a, b) in enumerate(zip(model.blocks_list, loaded.blocks_list)):
            aw = {w.path: _np(w) for w in a["block"].weights}
            bw = {w.path: _np(w) for w in b["block"].weights}
            assert aw, f"block {i} holds no weights"
            assert set(aw) == set(bw)
            for p in aw:
                np.testing.assert_allclose(
                    aw[p], bw[p], atol=0, rtol=0,
                    err_msg=f"block {i} weight {p} was NOT restored",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
