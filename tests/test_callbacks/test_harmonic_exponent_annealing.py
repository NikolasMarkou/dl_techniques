"""Tests for HarmonicExponentAnnealingCallback."""

import math

import numpy as np
import pytest
import keras

from dl_techniques.callbacks.anneal_harmonic_exponent import (
    HarmonicExponentAnnealingCallback,
)
from dl_techniques.layers.structured_linear.hierarchical_harmonic import (
    HierarchicalHarmonicHead,
)


class TestHarmonicExponentAnnealingCallback:
    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            HarmonicExponentAnnealingCallback(schedule="bogus")

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            HarmonicExponentAnnealingCallback(n_init=0.0)
        with pytest.raises(ValueError, match="positive"):
            HarmonicExponentAnnealingCallback(n_final=[1.0, -2.0])

    def test_invalid_epochs_raise(self):
        with pytest.raises(ValueError, match="total_epochs"):
            HarmonicExponentAnnealingCallback(total_epochs=0)

    def test_linear_endpoints(self):
        cb = HarmonicExponentAnnealingCallback(
            schedule="linear", n_init=1.0, n_final=9.0, total_epochs=5
        )
        assert math.isclose(cb._n_at(0, 1.0, 9.0), 1.0, abs_tol=1e-9)
        assert math.isclose(cb._n_at(4, 1.0, 9.0), 9.0, abs_tol=1e-9)
        assert math.isclose(cb._n_at(2, 1.0, 9.0), 5.0, abs_tol=1e-9)

    def test_cosine_endpoints(self):
        cb = HarmonicExponentAnnealingCallback(
            schedule="cosine", n_init=1.0, n_final=9.0, total_epochs=5
        )
        assert math.isclose(cb._n_at(0, 1.0, 9.0), 1.0, abs_tol=1e-9)
        assert math.isclose(cb._n_at(4, 1.0, 9.0), 9.0, abs_tol=1e-9)
        assert math.isclose(cb._n_at(2, 1.0, 9.0), 5.0, abs_tol=1e-9)

    def test_exp_endpoints(self):
        cb = HarmonicExponentAnnealingCallback(
            schedule="exp", n_init=1.0, n_final=16.0, total_epochs=3
        )
        assert math.isclose(cb._n_at(0, 1.0, 16.0), 1.0, abs_tol=1e-9)
        assert math.isclose(cb._n_at(2, 1.0, 16.0), 16.0, abs_tol=1e-9)
        assert math.isclose(cb._n_at(1, 1.0, 16.0), 4.0, abs_tol=1e-9)

    def test_single_epoch_returns_final(self):
        cb = HarmonicExponentAnnealingCallback(
            schedule="linear", n_init=1.0, n_final=9.0, total_epochs=1
        )
        assert math.isclose(cb._n_at(0, 1.0, 9.0), 9.0, abs_tol=1e-9)

    def test_per_level_length_mismatch_raises(self):
        cb = HarmonicExponentAnnealingCallback(n_init=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="n_init"):
            cb._as_list(cb.n_init, 2, "n_init")

    def test_callback_drives_layer_exponents_through_fit(self):
        inputs = keras.Input((6,))
        head = HierarchicalHarmonicHead(8, branching=(2, 4), n=1.0, name="hier_head")
        outputs = head(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        )
        cb = HarmonicExponentAnnealingCallback(
            schedule="linear", n_init=1.0, n_final=5.0, total_epochs=3
        )
        x = np.random.default_rng(0).standard_normal((16, 6)).astype("float32")
        y = np.random.default_rng(1).integers(0, 8, size=(16,))
        model.fit(x, y, epochs=3, batch_size=8, verbose=0, callbacks=[cb])
        # on_epoch_begin for the final epoch (index 2) set n_final on both levels.
        assert head.effective_n == [pytest.approx(5.0), pytest.approx(5.0)]

    def test_layer_names_filter(self):
        inputs = keras.Input((6,))
        outputs = HierarchicalHarmonicHead(8, branching=(2, 4), n=1.0)(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        cb = HarmonicExponentAnnealingCallback(
            n_init=1.0, n_final=5.0, total_epochs=3, layer_names=["no_such_layer"]
        )
        x = np.zeros((4, 6), dtype="float32")
        y = np.zeros((4,), dtype="int32")
        model.fit(x, y, epochs=1, verbose=0, callbacks=[cb])
        assert model.layers[-1].effective_n == [pytest.approx(1.0)] * 2

    def test_get_config_round_trip(self):
        cb = HarmonicExponentAnnealingCallback(
            schedule="exp", n_init=[1.0, 2.0], n_final=8.0, total_epochs=7
        )
        cfg = cb.get_config()
        rebuilt = HarmonicExponentAnnealingCallback(**cfg)
        assert rebuilt.schedule == "exp"
        assert rebuilt.n_init == [1.0, 2.0]
        assert rebuilt.total_epochs == 7
