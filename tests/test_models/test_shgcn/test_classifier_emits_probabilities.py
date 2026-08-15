"""C-44(b): `SHGCNNodeClassifier` returns PROBABILITIES, not logits.

``model.py:373`` named the classifier output ``logits`` while
``self.classifier`` is ``Dense(num_classes, activation='softmax')``. The class
docstring and the ``call`` docstring both correctly said probabilities, so the
variable name was the only thing lying -- and a caller who compiled
``from_logits=True`` on the strength of it got a silently wrong loss.

A name-level assertion is weak, so the PROPERTY is pinned: the rows sum to 1.
The prose is checked too, because the defect was a naming/prose disagreement
and a rename alone leaves nothing testable.
"""

import numpy as np
import keras
import scipy.sparse
import tensorflow as tf

from dl_techniques.models.shgcn.model import SHGCNNodeClassifier


def _tiny_graph(num_nodes: int = 12, input_dim: int = 6):
    features = np.random.normal(size=(num_nodes, input_dim)).astype("float32")
    dense_adj = np.eye(num_nodes, dtype="float32")
    for i in range(num_nodes - 1):
        dense_adj[i, i + 1] = 1.0
        dense_adj[i + 1, i] = 1.0
    dense_adj /= dense_adj.sum(axis=1, keepdims=True)
    sparse = scipy.sparse.coo_matrix(dense_adj)
    adjacency = tf.SparseTensor(
        indices=np.stack([sparse.row, sparse.col], axis=1),
        values=sparse.data.astype("float32"),
        dense_shape=(num_nodes, num_nodes),
    )
    return features, tf.sparse.reorder(adjacency)


class TestTheClassifierHeadEmitsProbabilities:

    def test_rows_sum_to_one(self):
        features, adjacency = _tiny_graph()
        model = SHGCNNodeClassifier(
            num_classes=4, hidden_dims=[8], dropout_rate=0.0)

        outputs = np.asarray(model([features, adjacency], training=False))

        assert outputs.shape == (12, 4)
        row_sums = outputs.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(12), atol=1e-5)

    def test_outputs_are_non_negative(self):
        """A logit tensor would be mixed-sign; a softmax one cannot be."""
        features, adjacency = _tiny_graph()
        model = SHGCNNodeClassifier(
            num_classes=4, hidden_dims=[8], dropout_rate=0.0)

        outputs = np.asarray(model([features, adjacency], training=False))
        assert outputs.min() >= 0.0, (
            "ASSERT-PROBABILITIES-NOT-LOGITS: a negative value here would mean "
            "the head stopped applying softmax."
        )

    def test_the_head_is_declared_softmax(self):
        model = SHGCNNodeClassifier(num_classes=4, hidden_dims=[8])
        assert model.classifier.activation is keras.activations.softmax

    def test_the_call_docstring_says_probabilities_not_logits(self):
        """The defect was a naming disagreement; pin both halves."""
        docstring = SHGCNNodeClassifier.call.__doc__ or ""
        assert "probabilities" in docstring.lower()
        assert "logits" not in docstring.lower(), (
            "ASSERT-NO-LOGITS-CLAIM: the head applies softmax, so nothing in "
            "its contract may promise logits."
        )
