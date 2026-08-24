"""C-41: a second `train()` must adapt, and inference must not materialize 75 GB.

(a) ``train()`` assigned ``max_iterations = epochs * len(x_train)`` on every
call but never reset ``iterations`` to 0. After a first ``train(x, epochs=E)``
the counter sits exactly AT ``max_iterations``, so a SECOND ``train()`` rewrote
the same budget and left the decayed rate clamped at 0 for the whole run: zero
adaptation, a flat quantization error, silently. Every existing test calls
``train()`` exactly once.

The history matters and is deliberately not undone: an earlier fix corrected the
UNITS (samples, not batches) and is what made this reachable, by changing the
budget from a value the counter overran to one it lands exactly on.

(b) ``SOMLayer._find_bmu`` broadcasts to ``(batch, num_neurons, input_dim)``,
and five ``model.py`` sites passed the ENTIRE unbatched dataset. The class
docstring's own MNIST example (``map_size=(20,20)``, ``input_dim=784``, 60000
samples) is a ~75 GB float32 intermediate.
"""

import numpy as np
import pytest

from dl_techniques.models.som.model import SOMModel


@pytest.fixture
def som() -> SOMModel:
    return SOMModel(
        map_size=(4, 4), input_dim=6, initial_learning_rate=0.5, sigma=2.0)


@pytest.fixture
def data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(64, 6)).astype("float32")


def _weights(model: SOMModel) -> np.ndarray:
    from keras import ops
    return np.asarray(ops.convert_to_numpy(model.som_layer.weights_map))


class TestASecondTrainCallStillAdapts:

    def test_the_counter_restarts_from_zero(self, som, data):
        from keras import ops

        som.train(data, epochs=2, batch_size=16, verbose=0)
        after_first = float(ops.convert_to_numpy(som.som_layer.iterations))
        budget = float(ops.convert_to_numpy(som.som_layer.max_iterations))
        assert after_first == pytest.approx(budget), (
            "precondition: the counter lands exactly on its budget, which is "
            "what makes a second call a no-op"
        )

        som.train(data, epochs=2, batch_size=16, verbose=0)
        # Same budget again, so a reset run must land on the budget once more
        # rather than at twice it.
        after_second = float(ops.convert_to_numpy(som.som_layer.iterations))
        assert after_second == pytest.approx(budget), (
            f"ASSERT-ITERATIONS-RESET: after a second train() the counter is "
            f"{after_second}, expected {budget} -- it was not reset to 0."
        )

    def test_the_second_call_moves_the_weights(self, som, data):
        som.train(data, epochs=2, batch_size=16, verbose=0)
        before_second = _weights(som).copy()

        som.train(data, epochs=2, batch_size=16, verbose=0)
        after_second = _weights(som)

        delta = float(np.max(np.abs(after_second - before_second)))
        assert delta > 0.0, (
            "ASSERT-SECOND-TRAIN-ADAPTS: the second train() moved the map by "
            "exactly 0.0 -- the decayed learning rate is clamped at 0."
        )

    def test_the_first_call_moves_them_too(self, som, data):
        """Liveness arm: proves the delta detector can see adaptation at all."""
        som.build((None, 6))
        before_first = _weights(som).copy()
        som.train(data, epochs=2, batch_size=16, verbose=0)
        assert float(np.max(np.abs(_weights(som) - before_first))) > 0.0

    def test_the_reported_error_is_not_frozen(self, som, data):
        """The user-visible symptom: a flat quantization error.

        ``shuffle=False`` is load-bearing. MEASURED: with the default shuffle,
        this assertion passes even against the defect -- a frozen map still
        reports a different per-epoch mean because the batches differ. Only
        with a fixed batch order does "no adaptation" mean "identical value".
        """
        som.train(data, epochs=3, batch_size=16, shuffle=False, verbose=0)
        history = som.train(
            data, epochs=3, batch_size=16, shuffle=False, verbose=0)

        errors = history["mean_quantization_error"]
        assert len(errors) == 3
        assert len(set(np.round(errors, 12))) > 1, (
            f"ASSERT-ERROR-NOT-FLAT: the second run reported {errors}, an "
            f"identical value every epoch -- nothing adapted."
        )


class TestInferenceIsBatched:

    def test_batched_equals_unbatched(self, som, data):
        """Chunking is mathematically inert -- assert EQUALITY, not closeness."""
        from keras import ops

        som.train(data, epochs=1, batch_size=16, verbose=0)

        unbatched, _ = som.som_layer(
            ops.convert_to_tensor(data.reshape(data.shape[0], -1)),
            training=False)
        unbatched = ops.convert_to_numpy(unbatched)

        batched = som._bmu_indices_in_batches(data, batch_size=7)

        assert batched.shape == unbatched.shape
        np.testing.assert_array_equal(batched, unbatched)

    def test_more_than_one_chunk_is_actually_used(self, som, data, monkeypatch):
        """Otherwise the equality above is satisfied by never chunking."""
        calls = []
        original = som.som_layer.call

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(som.som_layer, "call", counting)
        som._bmu_indices_in_batches(data, batch_size=7)

        expected = int(np.ceil(data.shape[0] / 7))
        assert len(calls) == expected, (
            f"ASSERT-CHUNKED: expected {expected} forward passes for "
            f"{data.shape[0]} samples at batch_size=7, saw {len(calls)}."
        )
        assert expected > 1

    def test_no_site_passes_a_full_dataset_to_the_layer(self, som, data):
        """The five inference sites must go through the chunked helper."""
        import inspect

        source = inspect.getsource(SOMModel)
        direct = [
            line.strip() for line in source.splitlines()
            if "= self.som_layer(" in line
            and "training=False" in line
            and "chunk" not in line
        ]
        # The only survivor is the single-sample retrieval probe.
        assert len(direct) == 1, (
            f"ASSERT-ALL-SITES-BATCHED: {len(direct)} direct inference calls "
            f"remain: {direct}"
        )
        assert "test_sample_tensor" in direct[0]

    def test_an_empty_input_is_handled(self, som):
        result = som._bmu_indices_in_batches(np.zeros((0, 6), dtype="float32"))
        assert result.shape == (0, 2)

    def test_a_non_positive_batch_size_raises(self, som, data):
        with pytest.raises(ValueError, match="batch_size"):
            som._bmu_indices_in_batches(data, batch_size=0)
