"""Quirk guard: the classifier-free-guidance null label is row ``num_classes``.

**The line this file pins.**
``src/dl_techniques/layers/embedding/class_label_embedding.py``,
``ClassLabelEmbedding._token_drop``::

    cfg_row = keras.ops.full_like(labels, self.num_classes)
    return keras.ops.where(drop_ids, cfg_row, labels)

with the table sized ``num_classes + use_cfg_embedding`` in ``__init__``. That is
upstream's ``LabelEmbedder`` (``reference/models.py``): ``self.embedding_table =
nn.Embedding(num_classes + use_cfg_embedding, hidden_size)`` and
``labels = torch.where(drop_ids, self.num_classes, labels)``. ``DiT`` reaches this
through ``self.y_embedder`` on BOTH paths -- training-time dropout and
sampling-time ``y_null`` (``reference/train_and_sample_excerpts.py``, and
``forward_with_cfg``'s second half).

**The plausible WRONG alternatives this file is RED against.**

1. ``num_classes - 1`` as the null index. The table is still the right size, the
   gather is still in bounds, the loss still falls, and every dropped sample is
   silently conditioned on the LAST REAL CLASS. Nothing about shape, dtype or
   finiteness can see it.
2. A SEPARATE null embedding -- a second weight, or a hard-coded zero vector,
   instead of row ``num_classes`` of the one table. That row is a trained
   parameter upstream; a zero vector or an untracked variable is a different
   model with the same output shape.

**Why the existing arm does not cover this.**
``test_dit_diffusion.py::test_the_null_row_is_index_num_classes`` asserts only
that a label of ``num_classes`` produces a FINITE forward pass, which
``num_classes - 1`` also does. The arms here compare the embedded vector against
the table ROW at ``atol=0``, drive the drop deterministically through
``force_drop_ids`` rather than through a probability, and count the weights.

**RED proof (step 10).** Two injections into ``class_label_embedding.py``:

* ``cfg_row = full_like(labels, self.num_classes - 1)`` -- **3 failed / 10 passed**:
  ``test_every_forced_drop_returns_the_num_classes_row``,
  ``test_it_is_not_the_num_classes_minus_one_row``,
  ``test_training_time_dropout_sends_labels_to_that_row_too``.
* a SEPARATE ``null_embedding`` weight added in ``build()`` -- **1 failed /
  12 passed**: ``test_the_label_path_holds_exactly_one_weight``. Only that arm
  can see it; every value arm stays green because the table row still exists.
"""

from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.class_label_embedding import ClassLabelEmbedding

from ._dit_helpers import TINY, built_model, np_


def label_table(model: Any) -> np.ndarray:
    """The ``(rows, hidden)`` embedding matrix, found by weight path, not index."""
    matches = [
        w for w in model.weights if w.path.endswith("y_embedder/embedding_table/embeddings")
    ]
    assert len(matches) == 1, [w.path for w in model.weights]
    return np_(matches[0])


# ---------------------------------------------------------------------
# The table has exactly one extra row, and it is the LAST one
# ---------------------------------------------------------------------


class TestTheTableShape:
    """``num_classes + 1`` rows when dropout > 0, exactly ``num_classes`` when not."""

    def test_dropout_adds_exactly_one_row(self) -> None:
        model = built_model(seed=0)
        table = label_table(model)
        assert TINY["class_dropout_rate"] > 0.0
        assert table.shape == (TINY["num_classes"] + 1, TINY["hidden_size"])
        assert model.y_embedder.table_size == TINY["num_classes"] + 1

    def test_without_dropout_there_is_no_extra_row(self) -> None:
        model = built_model(seed=0, class_dropout_rate=0.0)
        table = label_table(model)
        assert table.shape == (TINY["num_classes"], TINY["hidden_size"])
        assert model.y_embedder.use_cfg_embedding is False

    def test_the_null_index_is_the_last_valid_row(self) -> None:
        """``num_classes`` indexes IN BOUNDS only because of the extra row."""
        model = built_model(seed=0)
        rows = label_table(model).shape[0]
        assert TINY["num_classes"] == rows - 1

    def test_the_label_path_holds_exactly_one_weight(self) -> None:
        """RED against a SEPARATE null embedding living beside the table."""
        model = built_model(seed=0)
        label_weights = [w for w in model.weights if "/y_embedder/" in w.path]
        assert len(label_weights) == 1, [w.path for w in label_weights]
        assert label_weights[0].trainable, "the null row must be TRAINED, not frozen"


# ---------------------------------------------------------------------
# A dropped label lands on row `num_classes`, exactly
# ---------------------------------------------------------------------


class TestADroppedLabelLandsOnRowNumClasses:
    """Driven by ``force_drop_ids``, so nothing depends on a dropout draw."""

    def _layer(self, seed: int = 3) -> ClassLabelEmbedding:
        keras.utils.set_random_seed(seed)
        layer = ClassLabelEmbedding(
            num_classes=TINY["num_classes"],
            hidden_size=TINY["hidden_size"],
            dropout_rate=0.1,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        layer.build((None,))
        return layer

    def test_every_forced_drop_returns_the_num_classes_row(self) -> None:
        layer = self._layer()
        table = np_(layer.embedding_table.embeddings)
        labels = np.arange(TINY["num_classes"], dtype="int32")
        drop = np.ones_like(labels)

        out = np_(layer(labels, force_drop_ids=drop))
        expected = np.broadcast_to(table[TINY["num_classes"]], out.shape)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0.0)

    def test_it_is_not_the_num_classes_minus_one_row(self) -> None:
        """The sharp discriminator. Both rows exist; only one is the null row."""
        layer = self._layer()
        table = np_(layer.embedding_table.embeddings)
        null_row = table[TINY["num_classes"]]
        last_real_row = table[TINY["num_classes"] - 1]
        # Anti-vacuity: the two candidate rows must be distinguishable at all.
        assert float(np.max(np.abs(null_row - last_real_row))) > 0.0

        out = np_(layer(np.array([0, 5], "int32"), force_drop_ids=np.array([1, 1])))
        for row in out:
            np.testing.assert_allclose(row, null_row, rtol=0, atol=0.0)
            assert not np.allclose(row, last_real_row, rtol=0, atol=1e-8)

    def test_an_undropped_label_returns_its_own_row(self) -> None:
        """The other half of the claim: dropping is the ONLY thing that redirects."""
        layer = self._layer()
        table = np_(layer.embedding_table.embeddings)
        labels = np.arange(TINY["num_classes"], dtype="int32")
        out = np_(layer(labels, force_drop_ids=np.zeros_like(labels)))
        np.testing.assert_allclose(out, table[: TINY["num_classes"]], rtol=0, atol=0.0)

    def test_an_explicit_num_classes_label_reaches_the_same_row(self) -> None:
        """Sampling-time ``y_null``: no dropout involved, same destination."""
        layer = self._layer()
        table = np_(layer.embedding_table.embeddings)
        y_null = np.full((3,), TINY["num_classes"], dtype="int32")
        out = np_(layer(y_null, training=False))
        np.testing.assert_allclose(
            out, np.broadcast_to(table[TINY["num_classes"]], out.shape), rtol=0, atol=0.0
        )


# ---------------------------------------------------------------------
# The same row, reached through the whole model
# ---------------------------------------------------------------------


class TestTheModelReachesTheSameRow:
    """``DiT.call`` builds ``c = t_emb + y_emb``; ``y_emb`` is that table row."""

    def test_the_conditioning_addend_for_y_null_is_the_num_classes_row(self) -> None:
        model = built_model(seed=0)
        table = label_table(model)
        y_null = np.full((2,), TINY["num_classes"], dtype="int32")
        emb = np_(model.y_embedder(keras.ops.convert_to_tensor(y_null), training=False))
        np.testing.assert_allclose(
            emb, np.broadcast_to(table[TINY["num_classes"]], emb.shape), rtol=0, atol=0.0
        )

    def test_training_time_dropout_sends_labels_to_that_row_too(self) -> None:
        """Driven through ``force_drop_ids``, not through a probability.

        ``DiT`` rejects ``class_dropout_rate >= 1.0`` (``model.py`` validation),
        so "every label is dropped" cannot be arranged by turning the rate up.
        ``force_drop_ids`` is honoured regardless of ``training``
        (``class_label_embedding.py``), which makes the drop deterministic and
        the assertion exact instead of a seeded coin flip.
        """
        model = built_model(seed=0)
        table = label_table(model)
        labels = np.array([0, 1, 2, 3], dtype="int32")
        emb = np_(
            model.y_embedder(
                keras.ops.convert_to_tensor(labels),
                force_drop_ids=keras.ops.ones((4,), dtype="int32"),
            )
        )
        np.testing.assert_allclose(
            emb, np.broadcast_to(table[TINY["num_classes"]], emb.shape), rtol=0, atol=0.0
        )
        assert not np.allclose(
            emb,
            np.broadcast_to(table[TINY["num_classes"] - 1], emb.shape),
            rtol=0,
            atol=1e-8,
        )

    def test_the_label_actually_reaches_the_output(self) -> None:
        """Anti-vacuity for the whole file: a woken model is label-sensitive.

        If the label path were dead, every arm above would be pinning a row that
        changes nothing. This one measures that swapping the class -- and
        swapping a real class for the null row -- both move the output.
        """
        from ._dit_helpers import activate, tiny_inputs

        model = activate(built_model(seed=0), seed=5)
        x, t, _ = tiny_inputs(seed=2)
        batch = x.shape[0]

        def forward(label: int) -> np.ndarray:
            return np_(
                model([x, t, np.full((batch,), label, "int32")], training=False)
            )

        real, other, null = forward(0), forward(7), forward(TINY["num_classes"])
        assert float(np.max(np.abs(real - other))) > 0.0
        assert float(np.max(np.abs(real - null))) > 0.0


# ---------------------------------------------------------------------
# Without the extra row, guidance is refused rather than silently wrong
# ---------------------------------------------------------------------


class TestWithoutTheExtraRowGuidanceIsRefused:
    """``class_dropout_rate = 0`` removes the null row, and says so."""

    def test_force_drop_ids_raises(self) -> None:
        keras.utils.set_random_seed(4)
        layer = ClassLabelEmbedding(
            num_classes=TINY["num_classes"], hidden_size=TINY["hidden_size"],
            dropout_rate=0.0,
        )
        layer.build((None,))
        with pytest.raises(ValueError, match="force_drop_ids"):
            layer(np.array([0, 1], "int32"), force_drop_ids=np.array([1, 1]))

    def test_an_out_of_range_label_is_not_silently_absorbed(self) -> None:
        """A ``num_classes`` label against a ``num_classes``-row table is a defect.

        This arm does not assert an exception: an out-of-bounds gather is
        backend-defined and TensorFlow's CPU kernel raises while other backends
        clamp. What it pins is that the table is one row SHORT, which is the
        checkable fact, and which is exactly what makes the ``+1`` load-bearing.
        """
        model = built_model(seed=0, class_dropout_rate=0.0)
        assert label_table(model).shape[0] == TINY["num_classes"]
        assert model.y_embedder.use_cfg_embedding is False
