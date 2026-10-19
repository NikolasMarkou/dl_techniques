"""`SHGCNNodeClassifier` returns LOGITS (plan-2026-10-14T062616-81938e81 D-147).

Until 2026-10-17 the classifier was ``Dense(num_classes, activation='softmax')``
and this file pinned that the rows summed to 1 (C-44(b): a variable named
``logits`` held probabilities). The repository's logits rule moved the head to a
linear ``Dense``. Pinned on the axis a softmax would bend:

- a shift of the classifier bias moves every output by exactly that shift (a
  softmax is invariant to it, a sigmoid squashes it);
- stock ``fit`` with ``SparseCategoricalCrossentropy(from_logits=True)``
  memorises one small graph (the D-122 training bar), fed as the README's
  whole-graph sample: a leading batch axis and ``batch_size=1``.
"""

import keras
import numpy as np

from dl_techniques.models.graph.shgcn.model import SHGCNNodeClassifier

NUM_NODES, INPUT_DIM, NUM_CLASSES = 12, 6, 4


def _tiny_graph():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(NUM_NODES, INPUT_DIM)).astype("float32")
    adjacency = np.eye(NUM_NODES, dtype="float32")
    for i in range(NUM_NODES - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    return features, adjacency


def test_a_bias_shift_moves_every_output_by_exactly_the_shift():
    features, adjacency = _tiny_graph()
    model = SHGCNNodeClassifier(num_classes=NUM_CLASSES, hidden_dims=[8], dropout_rate=0.0)
    out = np.asarray(model([features, adjacency], training=False))
    assert out.shape == (NUM_NODES, NUM_CLASSES)
    shift = -50.0
    model.classifier.bias.assign(model.classifier.bias + shift)
    moved = np.asarray(model([features, adjacency], training=False))
    bound = 8 * np.finfo(np.float32).eps * (np.abs(out).max() + abs(shift))
    np.testing.assert_allclose(moved - out, shift, rtol=0, atol=bound)


def test_the_head_is_linear():
    model = SHGCNNodeClassifier(num_classes=NUM_CLASSES, hidden_dims=[8])
    assert model.classifier.activation is keras.activations.linear


def test_fit_with_from_logits_memorises_one_graph():
    keras.utils.set_random_seed(0)
    features, adjacency = _tiny_graph()
    labels = np.arange(NUM_NODES) % NUM_CLASSES
    model = SHGCNNodeClassifier(
        num_classes=NUM_CLASSES, hidden_dims=[32], embedding_dim=16, dropout_rate=0.0)
    # jit_compile=False: compiled with XLA (jit_compile=True on CPU, or "auto",
    # which resolves to XLA on a GPU) this model's loss is NaN from the second
    # step on, with the old softmax head too (measured 2026-10-19 at 44ea8be64
    # and after; without XLA it is finite). That defect is reported
    # separately; this arm measures the head.
    model.compile(optimizer=keras.optimizers.Adam(1e-2),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"], jit_compile=False)
    history = model.fit([features[None], adjacency[None]], labels[None],
                        batch_size=1, epochs=300, verbose=0)
    chance = float(np.log(NUM_CLASSES))
    assert history.history["loss"][-1] < 0.1 * chance, history.history["loss"][-1]
    assert history.history["accuracy"][-1] == 1.0


def test_the_call_docstring_says_logits_not_probabilities():
    docstring = (SHGCNNodeClassifier.call.__doc__ or "").lower()
    assert "logits" in docstring
    assert "probabilit" not in docstring
