"""Training integration for the crystal-structured layers.

Every other suite in this package checks forward passes, shapes, gradients and
serialization. None of them ever calls ``fit()`` on a layer other than
``RigidSimplexLayer`` -- measured before this module existed: ``fit()`` call
count of 0 in the ``CrystalLayer``, ``HierarchicalCrystalLayer`` and
``CrystalMultiHeadProjection`` suites.

That gap matters for two reasons the forward-pass tests cannot cover:

* **A gradient existing is not the same as a layer being trainable.** A weight
  can receive a non-zero gradient, serialise cleanly, and still fail to reduce
  a loss because something upstream consumes the signal.
* **The geometric claims are claims about TRAINING, not about construction.**
  ``trainable_crystal=True`` is documented as letting a frame drift off the unit
  sphere while ``call()`` re-normalizes it so the logits stay cosines. Nothing
  tested that. Nor was there anything showing the hierarchical head learns its
  macro/micro split rather than collapsing onto one group -- a composition
  property, and shape and gradient assertions are blind to it by construction.
"""

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.structured_linear.crystal_layer import CrystalLayer
from dl_techniques.layers.structured_linear.hierarchical_crystal_layer import (
    HierarchicalCrystalLayer
)
from dl_techniques.layers.structured_linear.crystal_multihead_projection import (
    CrystalMultiHeadProjection
)

# ---------------------------------------------------------------------

_SEED = 2024
_BATCH = 64
_STEPS = 3


def _dataset(num_macro: int = 3, num_micro: int = 4, input_dim: int = 16):
    """A learnable classification problem: Gaussian blobs, one per class.

    Deterministic (``np.random.default_rng``), and separable enough that a
    working head must reduce the loss -- so a head that produces constant
    logits cannot pass by sitting still.
    """
    rng = np.random.default_rng(_SEED)
    total = num_macro * num_micro

    centres = rng.standard_normal((total, input_dim)) * 3.0
    labels = np.repeat(np.arange(total), _BATCH)
    features = centres[labels] + rng.standard_normal((total * _BATCH, input_dim))

    perm = rng.permutation(len(labels))
    features, labels = features[perm], labels[perm]

    return (
        features.astype(np.float32),
        labels.astype(np.int32),
    )


def _fit(model, x, y, steps: int = _STEPS) -> list:
    """Fit and return the per-epoch training loss."""
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model.fit(x, y, epochs=steps, verbose=0).history['loss']


class TestCrystalLayerTrains:
    """The flat head must reduce a real loss."""

    def test_training_reduces_the_loss(self):
        x, y = _dataset(num_macro=1, num_micro=8)

        inputs = keras.Input(shape=(x.shape[1],))
        # trainable_crystal: the frame is the thing being learned here, so a
        # frozen one could only move the head's lack of a trainable kernel.
        outputs = CrystalLayer(
            num_classes=8, scale=5.0, trainable_crystal=True, seed=_SEED
        )(inputs)
        model = keras.Model(inputs, outputs)

        losses = _fit(model, x, y)

        assert losses[-1] < losses[0], (
            f"training did not reduce the loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_a_frozen_frame_still_trains_through_its_input_path(self):
        """A frozen frame is a valid head when something upstream can learn.

        Probes the other branch of the same knob: with the frame frozen there
        are no trainable weights in the layer at all, so the reduction has to
        come from whatever is placed before it. Without this, the previous test
        would pass while proving only that a trainable frame can move.
        """
        x, y = _dataset(num_macro=1, num_micro=8)

        inputs = keras.Input(shape=(x.shape[1],))
        features = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = CrystalLayer(
            num_classes=8, scale=5.0, trainable_crystal=False, seed=_SEED
        )(features)
        model = keras.Model(inputs, outputs)

        losses = _fit(model, x, y)

        assert losses[-1] < losses[0], (
            f"a frozen frame trained through a preceding Dense still did not "
            f"reduce the loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_a_trained_frame_still_yields_cosines(self):
        """The documented claim, tested: a drifted frame still gives cosines.

        ``trainable_crystal=True`` lets the frame leave the unit sphere, and
        ``call()`` re-normalizes it precisely so the logits stay cosines. That
        sentence was previously untested; here the frame is perturbed far off
        the sphere and the logits are checked against the closed form.
        """
        scale = 4.0
        layer = CrystalLayer(
            num_classes=9, scale=scale, trainable_crystal=True, seed=_SEED
        )

        rng = np.random.default_rng(_SEED)
        x = rng.standard_normal((8, 16)).astype(np.float32)
        layer(x)

        for factor in (1e-8, 1.0, 1e4, 1e12):
            layer.crystal_weights.assign(
                ops.convert_to_tensor(
                    ops.convert_to_numpy(layer.crystal_weights) * np.float32(factor)
                )
            )
            output = ops.convert_to_numpy(layer(x)).astype(np.float64)

            unit_w = ops.convert_to_numpy(layer.crystal_weights).astype(np.float64)
            unit_w = unit_w / np.linalg.norm(unit_w, axis=0, keepdims=True)
            unit_x = x.astype(np.float64) / np.linalg.norm(
                x.astype(np.float64), axis=-1, keepdims=True
            )

            np.testing.assert_allclose(
                output, scale * (unit_x @ unit_w), rtol=0, atol=1e-3,
                err_msg=(
                    f"with the frame scaled by {factor:g} the logits are no "
                    f"longer the scaled cosine"
                )
            )
            assert np.abs(output).max() <= scale * (1 + 1e-3), (
                f"a frame scaled by {factor:g} broke the cosine bound"
            )


class TestHierarchicalCrystalLayerTrains:
    """The two-tier head must learn the tree, not collapse onto one tier."""

    def test_training_reduces_the_loss(self):
        num_macro, num_micro = 3, 4
        x, y = _dataset(num_macro=num_macro, num_micro=num_micro)

        inputs = keras.Input(shape=(x.shape[1],))
        outputs = keras.layers.Dense(64, activation='relu')(inputs)
        outputs = HierarchicalCrystalLayer(
            num_macro_classes=num_macro,
            num_micro_per_macro=num_micro,
            scale=5.0,
            trainable_crystal=True,
            seed=_SEED,
        )(outputs)
        model = keras.Model(inputs, outputs)

        losses = _fit(model, x, y)

        assert losses[-1] < losses[0], (
            f"training did not reduce the loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_it_learns_the_macro_split_rather_than_collapsing(self):
        """The composition claim: both tiers must be used.

        A head that sent every input to one macro group would still reduce the
        loss on 3 groups by memorizing, and would pass any shape or gradient
        assertion. Requiring the predicted macro distribution to spread across
        all ``num_macro`` groups separates that from a head that learned the
        tree.
        """
        num_macro, num_micro = 3, 4
        x, y = _dataset(num_macro=num_macro, num_micro=num_micro)

        inputs = keras.Input(shape=(x.shape[1],))
        features = keras.layers.Dense(64, activation='relu')(inputs)
        outputs = HierarchicalCrystalLayer(
            num_macro_classes=num_macro,
            num_micro_per_macro=num_micro,
            cluster_spread=0.25,
            scale=5.0,
            return_hierarchy=True,
            trainable_crystal=True,
            seed=_SEED,
            name='crystal_head',
        )(features)
        model = keras.Model(inputs, outputs)

        # A dict output needs a dict `loss` keyed to the same output names.
        # Nesting under the producing layer's name is NOT the shape Keras wants:
        # the dict is the model's output structure directly, and `fit` receives
        # a flat dict of targets.
        model.compile(
            optimizer='adam',
            loss={'micro_logits': 'sparse_categorical_crossentropy',
                  'macro_logits': 'sparse_categorical_crossentropy'},
        )
        targets = {
            'micro_logits': y,
            'macro_logits': y // num_micro,
        }
        model.fit(x, targets, epochs=_STEPS, verbose=0)

        macro_probs = ops.convert_to_numpy(
            model(x, training=False)['macro_logits']
        )
        predicted = np.argmax(macro_probs, axis=-1)

        assert len(set(predicted.tolist())) == num_macro, (
            f"only {len(set(predicted.tolist()))} of {num_macro} macro groups "
            f"were ever predicted; the head collapsed onto a subset"
        )

    def test_the_frames_stay_unit_norm_through_training(self):
        """A trained frame is re-normalized on every forward, so it must not
        drift to a value the normalization can no longer handle."""
        x, y = _dataset(num_macro=2, num_micro=3)

        inputs = keras.Input(shape=(x.shape[1],))
        outputs = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = HierarchicalCrystalLayer(
            num_macro_classes=2,
            num_micro_per_macro=3,
            trainable_crystal=True,
            seed=_SEED,
        )(outputs)
        model = keras.Model(inputs, outputs)
        _fit(model, x, y, steps=2)

        layer = model.layers[-1]
        for name in ('macro_weights', 'micro_weights'):
            frame = ops.convert_to_numpy(getattr(layer, name)).astype(np.float64)
            norms = np.linalg.norm(frame, axis=0)

            assert np.all(np.isfinite(frame)), f"{name} went non-finite"
            assert norms.min() > 1e-6, (
                f"{name} has a column of norm {norms.min():.3e} after training; "
                f"the per-forward renormalization cannot recover from a "
                f"collapse to zero"
            )


class TestCrystalMultiHeadProjectionTrains:
    """The filter bank must reduce a loss through its Dense."""

    def test_training_reduces_the_loss(self):
        x, y = _dataset(num_macro=1, num_micro=4)

        inputs = keras.Input(shape=(x.shape[1],))
        features = keras.layers.Dense(32, activation='relu')(inputs)
        projected = CrystalMultiHeadProjection(num_heads=4, seed=_SEED)(features)
        outputs = keras.layers.Dense(4)(projected)
        model = keras.Model(inputs, outputs)

        losses = _fit(model, x, y)

        assert losses[-1] < losses[0], (
            f"training did not reduce the loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_the_prisms_stay_finite_and_scaled_through_training(self):
        """A released prism is re-normalized per forward; it must not collapse."""
        x, y = _dataset(num_macro=1, num_micro=4)

        inputs = keras.Input(shape=(x.shape[1],))
        projected = CrystalMultiHeadProjection(
            num_heads=4, trainable_crystal=True, seed=_SEED
        )(inputs)
        outputs = keras.layers.Dense(4)(projected)
        model = keras.Model(inputs, outputs)
        _fit(model, x, y, steps=2)

        prisms = ops.convert_to_numpy(model.layers[-2].crystal_prisms)

        assert np.all(np.isfinite(prisms)), "the prisms went non-finite"
        norms = np.linalg.norm(prisms.astype(np.float64), axis=1)
        assert norms.min() > 1e-6, (
            f"a prism column collapsed to {norms.min():.3e} during training"
        )


class TestTrainingDoesNotBreakTheBayesIdentity:
    """The joint must stay normalized on a TRAINED layer, not only a fresh one.

    Every existing Bayes assertion runs on an untrained layer. A trained one
    has drifted frames, and the identity has to survive that.
    """

    def test_the_identity_holds_after_training(self):
        num_macro, num_micro = 3, 4
        x, y = _dataset(num_macro=num_macro, num_micro=num_micro)

        inputs = keras.Input(shape=(x.shape[1],))
        features = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = HierarchicalCrystalLayer(
            num_macro_classes=num_macro,
            num_micro_per_macro=num_micro,
            scale=5.0,
            return_hierarchy=True,
            trainable_crystal=True,
            seed=_SEED,
            name='crystal_head',
        )(features)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss={'micro_logits': 'sparse_categorical_crossentropy',
                  'macro_logits': 'sparse_categorical_crossentropy'},
        )
        model.fit(x, {'micro_logits': y, 'macro_logits': y // num_micro},
                  epochs=_STEPS, verbose=0)

        joint = ops.convert_to_numpy(
            model(x, training=False)['joint_probabilities']
        )

        assert np.all(np.isfinite(joint)), "a trained head produced non-finite joint"
        np.testing.assert_allclose(
            joint.sum(axis=-1), 1.0, rtol=0, atol=1e-5,
            err_msg="the Bayes identity did not survive training"
        )
