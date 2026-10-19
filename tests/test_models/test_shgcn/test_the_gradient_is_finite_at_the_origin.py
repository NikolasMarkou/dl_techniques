"""SHGCN's gradient is finite at the Poincare origin (plan-2026-10-14T062616-81938e81 D-155).

Two backward paths through ``PoincareMath`` were NaN at a zero vector, and an
SHGCN layer meets the origin on its first step by construction:

- the zero-initialised bias: ``exp_map_0(bias)`` takes the norm of a zero
  vector. sqrt's backward with ``dy = 0`` is 0 in TF's kernel and NaN under
  XLA, so every compiled SHGCN fit had a NaN loss from its second step;
- a node whose features are all zero (a padding or isolated node): with the
  zero bias its point is the origin, and ``log_map_0``'s discarded branch read
  ``arctanh(sqrt(c) * 1.0)``, which is NaN, eagerly too, once the learnable
  curvature reaches 1. It starts at ``softplus(0.54) = 0.9992``, so the arms
  run at that start and at ``c_theta = 1.0`` (``c = 1.31``).
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.graph.shgcn.model import SHGCNNodeClassifier

NUM_NODES, INPUT_DIM, NUM_CLASSES = 10, 5, 3


def _graph(zero_row: bool):
    rng = np.random.default_rng(1)
    features = rng.normal(size=(1, NUM_NODES, INPUT_DIM)).astype("float32")
    if zero_row:
        features[0, -1] = 0.0
    adjacency = np.eye(NUM_NODES, dtype="float32")[None]
    labels = (np.arange(NUM_NODES) % NUM_CLASSES)[None]
    return features, adjacency, labels


def _set_curvature_theta(model, c_theta):
    layers = [layer for layer in model._flatten_layers() if hasattr(layer, "c_theta")]
    assert layers, "no SHGCN layer found"
    for layer in layers:
        layer.c_theta.assign(c_theta)


@pytest.mark.parametrize("c_theta", [None, 1.0])
@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("zero_row", [False, True])
def test_every_gradient_is_finite_on_the_first_step(jit_compile, zero_row, c_theta):
    keras.utils.set_random_seed(0)
    features, adjacency, labels = _graph(zero_row)
    model = SHGCNNodeClassifier(
        num_classes=NUM_CLASSES, hidden_dims=[8], embedding_dim=4, dropout_rate=0.0)
    inputs = [tf.constant(features), tf.constant(adjacency)]
    model(inputs)
    if c_theta is not None:
        _set_curvature_theta(model, c_theta)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def step():
        with tf.GradientTape() as tape:
            loss = loss_fn(labels, model(inputs, training=True))
        return tape.gradient(loss, model.trainable_variables)

    grads = tf.function(step, jit_compile=jit_compile)()
    bad = [v.path for v, g in zip(model.trainable_variables, grads)
           if not np.all(np.isfinite(np.asarray(g)))]
    assert not bad, f"non-finite gradients (jit_compile={jit_compile}, c_theta={c_theta}): {bad}"


def test_a_compiled_fit_keeps_a_finite_loss():
    keras.utils.set_random_seed(0)
    features, adjacency, labels = _graph(zero_row=True)
    model = SHGCNNodeClassifier(
        num_classes=NUM_CLASSES, hidden_dims=[8], embedding_dim=4, dropout_rate=0.0)
    model.compile(optimizer=keras.optimizers.Adam(1e-2),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  jit_compile=True)
    history = model.fit([features, adjacency], labels, batch_size=1, epochs=5, verbose=0)
    assert np.all(np.isfinite(history.history["loss"])), history.history["loss"]
