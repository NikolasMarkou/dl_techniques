"""`ClassLabelEmbedding`: the CFG row, the drop, the `training` threading.

Four things here are invisible to shape, config and finiteness assertions, and
each gets its own value-level arm:

1. **The extra row exists only when ``dropout_rate > 0``.** Sizing the table at
   ``num_classes`` while still dropping to index ``num_classes`` is an
   out-of-bounds gather -- which raises on CPU and returns zeros on some GPU
   backends, so it can ship green.
2. **The drop is gated on ``training``.** Dropping at inference makes the model
   output non-deterministic in a way that a mean or a shape never reports.
3. **``force_drop_ids`` is NOT gated on ``training``.** The classifier-free
   guidance sampler needs the unconditional row at inference time; a
   ``training``-gated force silently returns the conditional embedding and CFG
   degenerates to the conditional pass, which still runs and still produces
   images.
4. **The lookup is a lookup.** Label ``k`` must return row ``k`` of the table,
   not row ``k`` of something permuted.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding import ClassLabelEmbedding
from dl_techniques.layers.embedding.factory import (
    EMBEDDING_REGISTRY,
    create_embedding_layer,
)


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _table(layer):
    return _np(layer.embedding_table.embeddings)


class TestTheCfgRowExistsOnlyWithDropout:

    def test_dropout_adds_exactly_one_row(self):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                    dropout_rate=0.1)
        layer.build((None,))
        assert layer.table_size == 4
        assert _table(layer).shape == (4, 8)
        assert layer.use_cfg_embedding is True

    def test_no_dropout_means_no_extra_row(self):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                    dropout_rate=0.0)
        layer.build((None,))
        assert layer.table_size == 3
        assert _table(layer).shape == (3, 8)
        assert layer.use_cfg_embedding is False

    def test_the_row_count_tracks_dropout_across_several_values(self):
        # An anti-vacuity arm for the two above: the +1 must be a FUNCTION of
        # dropout_rate, not a constant that happens to match at one value.
        for prob, expected in ((0.0, 5), (1e-6, 6), (0.5, 6), (1.0, 6)):
            layer = ClassLabelEmbedding(num_classes=5, hidden_size=4,
                                        dropout_rate=prob)
            layer.build((None,))
            assert layer.table_size == expected, prob

    def test_force_drop_without_a_cfg_row_raises_instead_of_gathering_oob(self):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                    dropout_rate=0.0)
        labels = keras.ops.convert_to_tensor(np.array([0, 1, 2], dtype="int32"))
        with pytest.raises(ValueError, match="no CFG row"):
            layer(labels, force_drop_ids=keras.ops.ones((3,), dtype="int32"))


class TestTheDropIsGatedOnTraining:

    def _layer(self, prob):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                    dropout_rate=prob, seed=1234)
        layer.build((None,))
        return layer

    def test_at_dropout_one_every_label_becomes_the_cfg_row_when_training(self):
        layer = self._layer(1.0)
        labels = keras.ops.convert_to_tensor(np.array([0, 1, 2], dtype="int32"))
        out = _np(layer(labels, training=True))
        cfg_row = _table(layer)[3]
        for i in range(3):
            np.testing.assert_allclose(out[i], cfg_row, atol=0.0)

    def test_at_dropout_one_no_label_moves_when_not_training(self):
        layer = self._layer(1.0)
        labels = keras.ops.convert_to_tensor(np.array([0, 1, 2], dtype="int32"))
        table = _table(layer)
        for training in (False, None):
            out = _np(layer(labels, training=training))
            for i in range(3):
                np.testing.assert_allclose(out[i], table[i], atol=0.0)
                # And it is NOT the CFG row -- stated separately so a table
                # whose rows happened to coincide could not pass silently.
                assert float(np.max(np.abs(out[i] - table[3]))) > 0.0

    def test_at_dropout_zero_training_true_changes_nothing(self):
        layer = self._layer(0.0)
        labels = keras.ops.convert_to_tensor(np.array([0, 1, 2], dtype="int32"))
        table = _table(layer)
        out = _np(layer(labels, training=True))
        np.testing.assert_allclose(out, table[:3], atol=0.0)

    def test_an_intermediate_rate_drops_some_but_not_all(self):
        # Guards against a fixed-True or fixed-False mask: with p = 0.5 over 512
        # samples, both outcomes must occur.
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=4,
                                    dropout_rate=0.5, seed=7)
        layer.build((None,))
        labels = keras.ops.zeros((512,), dtype="int32")
        out = _np(layer(labels, training=True))
        cfg_row = _table(layer)[3]
        dropped = np.all(np.isclose(out, cfg_row[None, :]), axis=1)
        assert 0 < int(dropped.sum()) < 512
        # Loose containment only -- this is a liveness arm, not a distribution
        # test.
        assert 0.3 < dropped.mean() < 0.7


class TestForceDropIdsIgnoresTraining:

    def _layer(self):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                    dropout_rate=0.1, seed=99)
        layer.build((None,))
        return layer

    @pytest.mark.parametrize("training", [True, False, None])
    def test_force_drop_ids_applies_at_every_training_value(self, training):
        layer = self._layer()
        labels = keras.ops.convert_to_tensor(np.array([0, 1, 2], dtype="int32"))
        force = keras.ops.convert_to_tensor(np.array([1, 0, 1], dtype="int32"))
        out = _np(layer(labels, training=training, force_drop_ids=force))
        table = _table(layer)
        np.testing.assert_allclose(out[0], table[3], atol=0.0)  # forced
        np.testing.assert_allclose(out[1], table[1], atol=0.0)  # not forced
        np.testing.assert_allclose(out[2], table[3], atol=0.0)  # forced

    def test_only_the_value_one_forces_a_drop(self):
        layer = self._layer()
        labels = keras.ops.zeros((4,), dtype="int32")
        force = keras.ops.convert_to_tensor(np.array([0, 1, 2, 1], dtype="int32"))
        out = _np(layer(labels, training=False, force_drop_ids=force))
        table = _table(layer)
        np.testing.assert_allclose(out[0], table[0], atol=0.0)
        np.testing.assert_allclose(out[1], table[3], atol=0.0)
        np.testing.assert_allclose(out[2], table[0], atol=0.0)  # 2 is not 1
        np.testing.assert_allclose(out[3], table[3], atol=0.0)

    def test_a_forced_call_consumes_no_rng(self):
        # The forced branch must not draw: two forced calls at training=True
        # have to agree exactly, even at dropout_rate = 0.5.
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                    dropout_rate=0.5, seed=3)
        layer.build((None,))
        labels = keras.ops.zeros((16,), dtype="int32")
        force = keras.ops.zeros((16,), dtype="int32")
        first = _np(layer(labels, training=True, force_drop_ids=force))
        second = _np(layer(labels, training=True, force_drop_ids=force))
        np.testing.assert_allclose(first, second, atol=0.0)


class TestTheLookupAndTheMechanics:

    def test_label_k_returns_row_k(self):
        layer = ClassLabelEmbedding(num_classes=5, hidden_size=6)
        layer.build((None,))
        labels = keras.ops.convert_to_tensor(np.arange(5, dtype="int32"))
        np.testing.assert_allclose(
            _np(layer(labels, training=False)), _table(layer), atol=0.0
        )

    def test_rank_1_and_rank_2_inputs_agree(self):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8)
        labels = np.array([0, 2, 1], dtype="int32")
        flat = _np(layer(keras.ops.convert_to_tensor(labels)))
        column = _np(
            layer(keras.ops.convert_to_tensor(labels.reshape(3, 1)))
        )
        assert flat.shape == (3, 8)
        np.testing.assert_allclose(flat, column, atol=0.0)

    def test_compute_output_shape(self):
        layer = ClassLabelEmbedding(num_classes=3, hidden_size=8)
        assert layer.compute_output_shape((None,)) == (None, 8)
        assert layer.compute_output_shape((4, 1)) == (4, 8)

    def test_two_layers_do_not_share_an_initializer_instance(self):
        shared = keras.initializers.RandomNormal(stddev=0.02, seed=None)
        a = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                embeddings_initializer=shared)
        b = ClassLabelEmbedding(num_classes=3, hidden_size=8,
                                embeddings_initializer=shared)
        assert (
            a.embedding_table.embeddings_initializer
            is not b.embedding_table.embeddings_initializer
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(num_classes=0, hidden_size=8),
            dict(num_classes=3, hidden_size=0),
            dict(num_classes=3, hidden_size=8, dropout_rate=-0.1),
            dict(num_classes=3, hidden_size=8, dropout_rate=1.5),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            ClassLabelEmbedding(**kwargs)

    def test_config_round_trip_preserves_every_knob(self):
        layer = ClassLabelEmbedding(
            num_classes=7,
            hidden_size=12,
            dropout_rate=0.25,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            seed=41,
        )
        restored = ClassLabelEmbedding.from_config(layer.get_config())
        assert restored.num_classes == 7
        assert restored.hidden_size == 12
        assert restored.dropout_rate == 0.25
        assert restored.seed == 41
        assert restored.table_size == 8
        assert isinstance(
            restored.embeddings_initializer, keras.initializers.RandomNormal
        )
        assert restored.embeddings_initializer.stddev == 0.02

    def test_keras_round_trip_preserves_values(self, tmp_path):
        inputs = keras.Input(shape=(), dtype="int32")
        outputs = ClassLabelEmbedding(
            num_classes=3, hidden_size=8, dropout_rate=0.1, seed=5, name="y_emb"
        )(inputs)
        model = keras.Model(inputs, outputs)
        labels = keras.ops.convert_to_tensor(np.array([0, 1, 2], dtype="int32"))
        before = _np(model(labels, training=False))

        path = tmp_path / "cls.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = _np(reloaded(labels, training=False))
        np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-6)
        assert reloaded.get_layer("y_emb").table_size == 4


class TestTheFactoryRegistration:

    def test_the_parameter_follows_the_house_dropout_naming_convention(self):
        # The repo-wide AST guard
        # (`tests/test_the_dropout_rate_naming_convention_holds.py`) checks
        # `def` parameters only, so it is blind to the registry DICT KEY and to
        # keyword call sites. This arm covers the registry key, which is the
        # half that reaches config-driven callers.
        entry = EMBEDDING_REGISTRY["class_label"]
        assert "dropout_rate" in entry["optional_params"]
        assert "dropout_prob" not in entry["optional_params"]
        assert "dropout_rate" in ClassLabelEmbedding(
            num_classes=3, hidden_size=8
        ).get_config()

    def test_the_key_is_registered(self):
        assert "class_label" in EMBEDDING_REGISTRY
        entry = EMBEDDING_REGISTRY["class_label"]
        assert entry["class"] is ClassLabelEmbedding
        assert entry["required_params"] == ["num_classes", "hidden_size"]
        assert set(entry["optional_params"]) == {
            "dropout_rate", "embeddings_initializer", "seed"
        }

    def test_the_factory_builds_it(self):
        layer = create_embedding_layer(
            "class_label", num_classes=3, hidden_size=8, dropout_rate=0.1
        )
        assert isinstance(layer, ClassLabelEmbedding)
        assert layer.table_size == 4

    def test_the_factory_defaults_match_the_constructor_defaults(self):
        from_factory = create_embedding_layer(
            "class_label", num_classes=3, hidden_size=8
        )
        direct = ClassLabelEmbedding(num_classes=3, hidden_size=8)
        assert from_factory.dropout_rate == direct.dropout_rate
        assert from_factory.table_size == direct.table_size
        assert from_factory.seed == direct.seed

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(num_classes=0, hidden_size=8),
            dict(num_classes=3, hidden_size=-1),
            dict(num_classes=3, hidden_size=8, dropout_rate=1.5),
            dict(num_classes=3, hidden_size=8, dropout_rate=-0.5),
        ],
    )
    def test_the_factory_validates(self, kwargs):
        with pytest.raises(ValueError):
            create_embedding_layer("class_label", **kwargs)

    def test_the_factory_rejects_an_undeclared_keyword(self):
        # `dropout_prob` is upstream's spelling of `dropout_rate`. This port
        # uses the repo convention, so the upstream name must be REJECTED
        # rather than silently dropped -- a filter-and-drop here would leave
        # every caller who copied the upstream signature at the 0.0 default.
        with pytest.raises(ValueError, match="unsupported parameter"):
            create_embedding_layer(
                "class_label", num_classes=3, hidden_size=8, dropout_prob=0.1
            )

    def test_it_is_re_exported_from_the_package(self):
        import dl_techniques.layers.embedding as pkg

        assert "ClassLabelEmbedding" in pkg.__all__
        assert pkg.ClassLabelEmbedding is ClassLabelEmbedding
