"""`SHGCNLinkPredictor` returns edge LOGITS (plan-2026-10-14T062616-81938e81 D-159).

Until 2026-10-19 its `FermiDiracDecoder` returned ``sigmoid(-(d^2 - r) / t)``,
an edge probability, and the README compiled ``loss="binary_crossentropy"`` on
it. Pinned on the axis a sigmoid would bend:

- a shift of the decoder threshold ``r`` by ``delta`` moves every output by
  exactly ``delta / t`` (a sigmoid squashes it, and saturates at 1);
- ``train_on_batch`` with ``BinaryCrossentropy(from_logits=True)`` (the
  docstring's whole-graph form) memorises a small set of positive and negative
  edges, the D-122 training bar, with every logit's sign matching its label.
"""

import keras
import numpy as np

from dl_techniques.models.graph.shgcn.model import SHGCNLinkPredictor

NUM_NODES, INPUT_DIM = 10, 6


def _graph():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(NUM_NODES, INPUT_DIM)).astype("float32")
    adjacency = np.eye(NUM_NODES, dtype="float32")
    for i in range(NUM_NODES - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, 9], [1, 7], [2, 8], [3, 6]], dtype="int32")
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype="float32")
    return features, adjacency, edges, labels


def test_a_threshold_shift_moves_every_output_by_exactly_the_shift():
    features, adjacency, edges, _ = _graph()
    model = SHGCNLinkPredictor(hidden_dims=[8], embedding_dim=4, dropout_rate=0.0)
    out = np.asarray(model([features, adjacency, edges], training=False))
    assert out.shape == (len(edges),)
    shift = 50.0
    t = float(model.decoder.t.numpy())
    model.decoder.r.assign(model.decoder.r + shift)
    moved = np.asarray(model([features, adjacency, edges], training=False))
    bound = 8 * np.finfo(np.float32).eps * (np.abs(out).max() + shift / t)
    np.testing.assert_allclose(moved - out, shift / t, rtol=0, atol=bound)


def test_train_on_batch_with_from_logits_memorises_the_edges():
    keras.utils.set_random_seed(0)
    features, adjacency, edges, labels = _graph()
    model = SHGCNLinkPredictor(hidden_dims=[16], embedding_dim=8, dropout_rate=0.0)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=loss_fn)
    for _ in range(300):
        model.train_on_batch([features, adjacency, edges], labels)
    logits = np.asarray(model([features, adjacency, edges], training=False))
    loss = float(loss_fn(labels, logits))
    assert loss < 0.1 * np.log(2.0), loss
    np.testing.assert_array_equal(logits > 0, labels > 0.5)


def test_the_call_docstring_says_logits_not_probabilities():
    docstring = (SHGCNLinkPredictor.call.__doc__ or "").lower()
    assert "logit" in docstring
    assert "probabilit" not in docstring
