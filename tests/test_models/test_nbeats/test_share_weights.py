"""`share_weights_in_stack` must actually tie the stack's weights.

Before 2026-08-14 the flag was threaded down to the blocks as a `share_weights`
constructor argument that the block stored and never read. Probe (1 generic
stack, 3 blocks, 16 hidden units, seed 7): both `False` and `True` built 4416
parameters and 3 distinct block objects — a parameter-sharing ablation that
could not have shared anything.

The same commit removed `reconstruction_weight`, which was stored and serialized
but never applied (`model.losses == []` at every setting). The reconstruction
penalty is real, but it lives in the trainer, which compiles a second loss
against `call`'s second output with its own `loss_weights` entry; the model's
copy of the number was decoration.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.time_series.nbeats.nbeats import NBeatsNet


def _build(share: bool, nb_blocks: int = 3) -> NBeatsNet:
    keras.utils.set_random_seed(7)
    model = NBeatsNet(
        backcast_length=24,
        forecast_length=8,
        stack_types=['generic'],
        nb_blocks_per_stack=nb_blocks,
        thetas_dim=[4],
        hidden_layer_units=16,
        share_weights_in_stack=share,
    )
    model(np.zeros((2, 24, 1), "float32"))
    return model


class TestShareWeightsInStack:

    def test_sharing_collapses_the_stack_to_one_object(self) -> None:
        shared = _build(True)
        assert len({id(b) for b in shared.blocks[0]}) == 1

    def test_not_sharing_keeps_blocks_independent(self) -> None:
        independent = _build(False)
        assert len({id(b) for b in independent.blocks[0]}) == 3

    def test_sharing_divides_the_parameter_count(self) -> None:
        """The guard: identical config, k-fold fewer parameters."""
        independent = _build(False)
        shared = _build(True)
        assert shared.count_params() * 3 == independent.count_params(), (
            f"shared={shared.count_params()} independent="
            f"{independent.count_params()} — sharing changed nothing"
        )

    def test_shared_model_still_forecasts(self) -> None:
        shared = _build(True)
        forecast, residual = shared(np.random.RandomState(0).randn(4, 24, 1).astype("float32"))
        assert tuple(forecast.shape) == (4, 8, 1)
        assert np.all(np.isfinite(np.asarray(forecast)))
        assert np.all(np.isfinite(np.asarray(residual)))

    def test_shared_model_round_trips(self, tmp_path) -> None:
        shared = _build(True)
        x = np.random.RandomState(1).randn(2, 24, 1).astype("float32")
        before = np.asarray(shared(x)[0])

        path = tmp_path / "nbeats_shared.keras"
        shared.save(path)
        restored = keras.models.load_model(path)

        assert restored.count_params() == shared.count_params()
        np.testing.assert_allclose(
            np.asarray(restored(x)[0]), before, rtol=1e-6, atol=1e-6
        )


class TestReconstructionWeightIsGone:

    def test_constructor_rejects_it(self) -> None:
        with pytest.raises(ValueError, match="reconstruction_weight"):
            NBeatsNet(
                backcast_length=24,
                forecast_length=8,
                stack_types=['generic'],
                nb_blocks_per_stack=1,
                thetas_dim=[4],
                hidden_layer_units=16,
                reconstruction_weight=0.5,
            )

    def test_config_does_not_carry_it(self) -> None:
        assert 'reconstruction_weight' not in _build(False, nb_blocks=1).get_config()
