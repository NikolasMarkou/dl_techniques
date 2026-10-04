"""Tests for PeriodicReassignCallback."""

import numpy as np
import pytest
import keras

from dl_techniques.callbacks.periodic_reassign import PeriodicReassignCallback
from dl_techniques.layers.structured_linear.hierarchical_harmonic import (
    HierarchicalHarmonicHead,
)


def _tiny_model(ncls=8, branching=(2, 4), dim=6):
    trunk = keras.layers.Dense(dim, activation="gelu", name="trunk")
    head = HierarchicalHarmonicHead(ncls, branching=branching, name="hhead")
    inp = keras.Input((dim,))
    return keras.Model(inp, head(trunk(inp))), head


def _nested_model(ncls=8, dim=6):
    inner_inp = keras.Input((dim,))
    inner = keras.Model(
        inner_inp, keras.layers.Dense(dim, activation="gelu")(inner_inp),
        name="nested_trunk",
    )
    head = HierarchicalHarmonicHead(ncls, branching=(2, 4), name="hhead")
    inp = keras.Input((dim,))
    feats = inner(inp)  # outer-graph tensor: the extractor must use THIS
    return keras.Model(inp, head(feats)), head, (inp, feats)


class TestPeriodicReassignCallback:
    def test_validation(self):
        x = np.zeros((8, 6), dtype="float32")
        y = np.arange(8)
        with pytest.raises(ValueError, match="every_n_epochs"):
            PeriodicReassignCallback("h", x, y, 8, every_n_epochs=0)
        with pytest.raises(ValueError, match="start_epoch"):
            PeriodicReassignCallback("h", x, y, 8, start_epoch=0)
        with pytest.raises(ValueError, match="disagree on sample count"):
            PeriodicReassignCallback("h", x, y[:4], 8)
        with pytest.raises(ValueError, match="no samples"):
            PeriodicReassignCallback("h", x, np.zeros(8, dtype=int), 8)
        with pytest.raises(ValueError, match="positive integer"):
            PeriodicReassignCallback("h", x, y, 0)

    def test_due_schedule(self):
        cb = PeriodicReassignCallback(
            "h", np.zeros((2, 6), np.float32), np.array([0, 1]), 2,
            every_n_epochs=2, start_epoch=2,
        )
        assert cb._due(0) is False
        assert cb._due(1) is True
        assert cb._due(2) is False
        assert cb._due(3) is True

    def test_unknown_head_raises(self):
        model, _ = _tiny_model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        cb = PeriodicReassignCallback(
            "nope", np.zeros((8, 6), np.float32), np.arange(8), 8,
            feature_layer_name="trunk", every_n_epochs=1,
        )
        cb.set_model(model)
        with pytest.raises(ValueError, match="not found"):
            cb.on_epoch_end(0)

    def test_head_without_reassign_raises(self):
        inp = keras.Input((6,))
        out = keras.layers.Dense(8, name="dense_head")(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        cb = PeriodicReassignCallback(
            "dense_head", np.zeros((8, 6), np.float32), np.arange(8), 8,
            feature_layer_name="dense_head", every_n_epochs=1,
        )
        cb.set_model(model)
        with pytest.raises(ValueError, match="no reassign"):
            cb.on_epoch_end(0)

    def test_neither_feature_source_raises(self):
        model, _ = _tiny_model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        cb = PeriodicReassignCallback(
            "hhead", np.zeros((8, 6), np.float32), np.arange(8), 8,
            every_n_epochs=1,
        )
        cb.set_model(model)
        with pytest.raises(ValueError, match="Neither feature"):
            cb.on_epoch_end(0)

    def test_fit_refreshes_table(self):
        model, head = _tiny_model()
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        )
        rng = np.random.default_rng(0)
        x = rng.standard_normal((64, 6)).astype("float32")
        y = rng.integers(0, 8, size=64)
        # Ensure full coverage for the class-means guard.
        y[:8] = np.arange(8)
        cb = PeriodicReassignCallback(
            "hhead", x, y, 8, feature_layer_name="trunk", every_n_epochs=1
        )
        model.fit(x, y, epochs=2, batch_size=32, verbose=0, callbacks=[cb])
        slots = np.asarray(keras.ops.convert_to_numpy(head.slot_of_class))
        assert sorted(slots.astype(int).tolist()) == list(range(8))

    def test_nested_trunk_via_feature_model_object(self):
        model, head, (inp, feats) = _nested_model()
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        )
        rng = np.random.default_rng(1)
        x = rng.standard_normal((64, 6)).astype("float32")
        y = rng.integers(0, 8, size=64)
        y[:8] = np.arange(8)
        extractor = keras.Model(inp, feats)
        cb = PeriodicReassignCallback(
            head, x, y, 8, feature_model=extractor, every_n_epochs=1
        )
        model.fit(x, y, epochs=1, batch_size=32, verbose=0, callbacks=[cb])
        slots = np.asarray(keras.ops.convert_to_numpy(head.slot_of_class))
        assert sorted(slots.astype(int).tolist()) == list(range(8))

    def test_get_config_round_trip(self):
        cb = PeriodicReassignCallback(
            "hhead", np.zeros((8, 6), np.float32), np.arange(8), 8,
            feature_layer_name="trunk",
            every_n_epochs=3, start_epoch=2, batch_size=16,
        )
        cfg = cb.get_config()
        assert cfg["head_layer_name"] == "hhead"
        assert cfg["every_n_epochs"] == 3
        assert "x" not in cfg and "y" not in cfg
