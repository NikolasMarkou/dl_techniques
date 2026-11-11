"""Tests for PeriodicReassignCallback."""

import inspect

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

    @staticmethod
    def _fired_epochs(every, start, epochs=10):
        cb = PeriodicReassignCallback(
            "h", np.zeros((2, 6), np.float32), np.array([0, 1]), 2,
            every_n_epochs=every, start_epoch=start,
        )
        return [e + 1 for e in range(epochs) if cb._due(e)]

    def test_first_firing_is_start_epoch_then_every_n(self):
        """``start_epoch`` is documented as the 1-based epoch of the first
        reassignment; the old rule fired at the first multiple of
        ``every_n_epochs`` at or after it."""
        assert self._fired_epochs(every=3, start=2, epochs=9) == [2, 5, 8]
        assert self._fired_epochs(every=2, start=3, epochs=8) == [3, 5, 7]
        assert self._fired_epochs(every=4, start=10, epochs=20) == [10, 14, 18]
        assert self._fired_epochs(every=2, start=1, epochs=6) == [1, 3, 5]
        assert self._fired_epochs(every=2, start=2, epochs=6) == [2, 4, 6]

    def test_reassign_is_called_at_the_scheduled_epochs(self):
        model, head = _tiny_model()
        calls = []
        head.reassign = lambda means: calls.append(means.shape) or {"moved": 0}
        cb = PeriodicReassignCallback(
            "hhead", np.zeros((16, 6), np.float32), np.arange(16) % 8, 8,
            feature_layer_name="trunk", every_n_epochs=3, start_epoch=2,
        )
        cb.set_model(model)
        fired = []
        for epoch in range(8):
            before = len(calls)
            cb.on_epoch_end(epoch)
            if len(calls) > before:
                fired.append(epoch + 1)
        assert fired == [2, 5, 8]
        assert all(shape == (8, 6) for shape in calls)

    def test_the_cached_extractor_is_rebuilt_for_a_new_model(self):
        """The extractor was cached once, so after ``set_model(m2)`` the class
        means still came from model 1's trunk."""
        m1, _ = _tiny_model()
        m2, _ = _tiny_model()
        x = np.random.default_rng(0).standard_normal((8, 6)).astype("float32")
        cb = PeriodicReassignCallback(
            "hhead", x, np.arange(8), 8, feature_layer_name="trunk", every_n_epochs=1,
        )
        cb.set_model(m1)
        cb.on_train_begin()
        cb._resolve()
        e1 = cb._extractor
        assert e1 is not None

        cb.set_model(m2)
        cb.on_train_begin()
        cb._resolve()
        e2 = cb._extractor
        assert e2 is not e1
        assert not np.allclose(e1.predict(x, verbose=0), e2.predict(x, verbose=0))
        np.testing.assert_allclose(
            e2.predict(x, verbose=0), m2.get_layer("trunk")(x).numpy(), rtol=1e-5, atol=1e-6,
        )

    def test_a_directly_passed_feature_model_survives_on_train_begin(self):
        model, head, (inp, feats) = _nested_model()
        extractor = keras.Model(inp, feats)
        cb = PeriodicReassignCallback(
            head, np.zeros((8, 6), np.float32), np.arange(8), 8, feature_model=extractor,
        )
        cb.set_model(model)
        cb.on_train_begin()
        cb._resolve()
        assert cb._extractor is extractor

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
        # Keys are constructor argument names (the head rides as ``head``).
        assert cfg["head"] == "hhead"
        assert cfg["every_n_epochs"] == 3
        assert "x" not in cfg and "y" not in cfg
        assert set(cfg) <= set(inspect.signature(PeriodicReassignCallback).parameters)

    def test_get_config_keys_are_constructor_arguments_not_a_loader(self):
        """``x``, ``y`` and ``num_classes`` are required constructor arguments
        and ``x`` / ``y`` are deliberately not in the config, so the callback
        cannot be rebuilt from its config alone. That is a documented limit
        (plan-2026-11-11T131349-13b28b37/D-030); the keys must still be real
        argument names so a caller can re-supply the arrays."""
        model, head = _tiny_model()
        x, y = np.zeros((8, 6), np.float32), np.arange(8)
        cb = PeriodicReassignCallback(
            head, x, y, 8, feature_layer_name="trunk",
            every_n_epochs=3, start_epoch=2, batch_size=16,
        )
        cfg = cb.get_config()
        rebuilt = PeriodicReassignCallback(x=x, y=y, **cfg)
        assert rebuilt.every_n_epochs == 3 and rebuilt.start_epoch == 2
        with pytest.raises(TypeError):
            PeriodicReassignCallback(**cfg)
