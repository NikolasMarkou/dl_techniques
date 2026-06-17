"""
Tests for the Gefen-lite (shared-v) optimizer implementation.

Six class-based suites mirroring the SGLD/VSGD test layout:
Instantiation / Build / Update / Serialization / Integration /
ConvergenceAndComparison.
"""

import os
import math
import tempfile
import numpy as np
import pytest
import tensorflow as tf
import keras
from keras import ops

# Top-level import so @register_keras_serializable fires before any load_model call.
from dl_techniques.optimization.gefen_optimizer import Gefen
from dl_techniques.optimization.optimizer import optimizer_builder


class TestGefenInstantiation:
    """Instantiation, default hyperparams, and range validation."""

    def test_import_and_register(self):
        """Import succeeds and the class is registered under Custom>Gefen."""
        # Keras registers under the default 'Custom>' prefix (verified Step 1/3).
        assert keras.saving.get_registered_object("Custom>Gefen") is not None
        assert keras.saving.get_registered_name(Gefen) == "Custom>Gefen"

    def test_defaults(self):
        """A default Gefen carries the documented hyperparameters."""
        opt = Gefen()
        config = opt.get_config()

        assert config["learning_rate"] == pytest.approx(1e-3)
        assert config["beta_1"] == pytest.approx(0.9)
        assert config["beta_2"] == pytest.approx(0.999)
        assert config["epsilon"] == pytest.approx(1e-8)
        assert config["weight_decay"] == pytest.approx(0.0)
        assert config["max_block_size"] == 1024
        assert config["min_block_size"] == 8

    def test_custom_args(self):
        """Custom args are stored and emitted through get_config."""
        opt = Gefen(
            learning_rate=5e-4,
            beta_1=0.8,
            beta_2=0.99,
            epsilon=1e-7,
            weight_decay=0.01,
            max_block_size=256,
            min_block_size=4,
        )
        config = opt.get_config()

        assert config["learning_rate"] == pytest.approx(5e-4)
        assert config["beta_1"] == pytest.approx(0.8)
        assert config["beta_2"] == pytest.approx(0.99)
        assert config["epsilon"] == pytest.approx(1e-7)
        assert config["weight_decay"] == pytest.approx(0.01)
        assert config["max_block_size"] == 256
        assert config["min_block_size"] == 4

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"learning_rate": -0.1},
            {"beta_1": 1.0},
            {"beta_1": 1.5},
            {"beta_2": -0.1},
            {"beta_2": 1.0},
            {"weight_decay": -0.5},
            {"epsilon": -1e-8},
            {"max_block_size": 0},
            {"min_block_size": 0},
        ],
    )
    def test_invalid_args(self, kwargs):
        """Out-of-range hyperparameters raise ValueError."""
        with pytest.raises(ValueError):
            Gefen(**kwargs)


class TestGefenBuild:
    """State allocation and the shape-based period rule."""

    def test_state_shapes(self):
        """momentum matches var shape; vmean is (K,) float32."""
        opt = Gefen(max_block_size=1024, min_block_size=8)

        var1 = keras.Variable(np.ones((8, 16), dtype=np.float32))  # numel 128
        var2 = keras.Variable(np.ones((120,), dtype=np.float32))   # numel 120
        opt.build([var1, var2])

        for i, var in enumerate([var1, var2]):
            idx = opt._get_variable_index(var)
            momentum = opt._momentum[idx]
            vmean = opt._vmean[idx]
            k = opt._blocks[idx]

            assert tuple(momentum.shape) == tuple(var.shape)
            assert tuple(vmean.shape) == (k,)
            assert vmean.dtype == "float32"

    @pytest.mark.parametrize(
        "shape",
        [
            (1,),        # scalar-ish -> period 1
            (32, 32),    # numel 1024 -> period 1024, K 1
            (2050,),     # numel 2050 = 2*5*5*41 -> largest divisor <=1024 is 410
            (1031,),     # prime > max_block_size -> period 1
        ],
    )
    def test_period_rule(self, shape):
        """period divides numel, period<=max_block_size, K*period==numel."""
        opt = Gefen(max_block_size=1024, min_block_size=8)
        var = keras.Variable(np.ones(shape, dtype=np.float32))
        grad = ops.convert_to_tensor(np.ones(shape, dtype=np.float32))

        # Apply one grad to trigger build, then read the recorded ints.
        opt.apply_gradients([(grad, var)])

        idx = opt._get_variable_index(var)
        period = opt._period[idx]
        k = opt._blocks[idx]
        n = int(math.prod(shape))

        assert n % period == 0, f"period {period} does not divide numel {n}"
        assert period <= opt._max_block_size
        assert k * period == n

        # Spot-check the specific expected values.
        if shape == (1,):
            assert period == 1
        elif shape == (32, 32):
            assert period == 1024 and k == 1
        elif shape == (2050,):
            assert period == 410 and k == 5
        elif shape == (1031,):
            assert period == 1 and k == 1031


class TestGefenUpdate:
    """Update math: eager, jit, multi-step stability, mixed precision."""

    def test_step_eager(self):
        """One apply_gradients in eager mode changes the variable."""
        np.random.seed(0)
        opt = Gefen(learning_rate=1e-2)
        var = keras.Variable(np.ones((4, 8), dtype=np.float32))
        initial = ops.convert_to_numpy(var).copy()
        grad = ops.convert_to_tensor(np.random.randn(4, 8).astype(np.float32))

        opt.apply_gradients([(grad, var)])

        assert not np.allclose(initial, ops.convert_to_numpy(var))

    def test_step_jit(self):
        """apply_gradients under @tf.function(jit_compile=True) runs and changes var."""
        np.random.seed(1)
        opt = Gefen(learning_rate=1e-2)
        var = keras.Variable(np.ones((4, 8), dtype=np.float32))
        initial = ops.convert_to_numpy(var).copy()
        grad = ops.convert_to_tensor(np.random.randn(4, 8).astype(np.float32))

        # Build outside the traced fn so slot creation is not inside XLA.
        opt.build([var])

        @tf.function(jit_compile=True)
        def step():
            opt.apply_gradients([(grad, var)])

        step()

        assert not np.allclose(initial, ops.convert_to_numpy(var))

    def test_multiple_steps(self):
        """5 steps keep changing the variable with no NaN/Inf."""
        np.random.seed(2)
        opt = Gefen(learning_rate=1e-2)
        var = keras.Variable(np.ones((6, 10), dtype=np.float32))

        prev = ops.convert_to_numpy(var).copy()
        for _ in range(5):
            grad = ops.convert_to_tensor(np.random.randn(6, 10).astype(np.float32))
            opt.apply_gradients([(grad, var)])
            cur = ops.convert_to_numpy(var)
            assert not np.allclose(prev, cur)
            prev = cur.copy()

        assert bool(ops.all(ops.isfinite(var)))

    def test_mixed_precision(self):
        """float16 var + float32 vmean: one step runs, vmean float32, var finite."""
        np.random.seed(3)
        opt = Gefen(learning_rate=1e-2)
        var = keras.Variable(np.ones((8, 16), dtype=np.float16))
        initial = ops.convert_to_numpy(var).copy()
        grad = ops.convert_to_tensor(np.random.randn(8, 16).astype(np.float16))

        opt.apply_gradients([(grad, var)])

        idx = opt._get_variable_index(var)
        assert opt._vmean[idx].dtype == "float32"
        final = ops.convert_to_numpy(var)
        assert not np.allclose(initial, final)
        assert bool(ops.all(ops.isfinite(var)))


class TestGefenSerialization:
    """Config and keras-object round-trips."""

    def test_config_round_trip(self):
        """get_config has no 'period' key; from_config reproduces hyperparams."""
        original = Gefen(
            learning_rate=5e-4,
            beta_1=0.85,
            beta_2=0.995,
            epsilon=1e-7,
            weight_decay=0.02,
            max_block_size=512,
            min_block_size=16,
        )
        config = original.get_config()

        assert "period" not in config
        assert "K" not in config

        restored = Gefen.from_config(config)
        rc = restored.get_config()

        assert rc["beta_1"] == pytest.approx(0.85)
        assert rc["beta_2"] == pytest.approx(0.995)
        assert rc["epsilon"] == pytest.approx(1e-7)
        assert rc["weight_decay"] == pytest.approx(0.02)
        assert rc["max_block_size"] == 512
        assert rc["min_block_size"] == 16

    def test_keras_object_round_trip(self):
        """deserialize(serialize(opt)) is a Gefen with equal hyperparams."""
        opt = Gefen(learning_rate=2e-3, beta_1=0.88, max_block_size=128)
        serialized = keras.saving.serialize_keras_object(opt)
        restored = keras.saving.deserialize_keras_object(serialized)

        assert isinstance(restored, Gefen)
        rc = restored.get_config()
        assert rc["beta_1"] == pytest.approx(0.88)
        assert rc["max_block_size"] == 128


class TestGefenIntegration:
    """Keras model save/load and the factory builder."""

    def test_keras_model_round_trip(self, tmp_path):
        """Save/load a compiled Sequential and resume training."""
        np.random.seed(4)
        keras.utils.set_random_seed(4)

        model = keras.Sequential([
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=Gefen(learning_rate=1e-2), loss="mse")

        x = np.random.randn(32, 4).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)
        model.fit(x, y, epochs=1, batch_size=16, verbose=0)

        path = os.path.join(tmp_path, "gefen_model.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert isinstance(loaded.optimizer, Gefen)

        pred_original = model.predict(x, verbose=0)
        pred_loaded = loaded.predict(x, verbose=0)
        np.testing.assert_allclose(pred_original, pred_loaded, atol=1e-5)

        # A further fit step runs without error.
        loaded.fit(x, y, epochs=1, batch_size=16, verbose=0)

    def test_factory(self):
        """optimizer_builder dispatches type 'gefen' and propagates hyperparams."""
        opt = optimizer_builder({"type": "gefen"}, 1e-3)
        assert isinstance(opt, Gefen)

        opt2 = optimizer_builder(
            {
                "type": "gefen",
                "beta_1": 0.85,
                "beta_2": 0.995,
                "max_block_size": 256,
                "min_block_size": 4,
            },
            5e-4,
        )
        assert isinstance(opt2, Gefen)
        c = opt2.get_config()
        assert c["beta_1"] == pytest.approx(0.85)
        assert c["beta_2"] == pytest.approx(0.995)
        assert c["max_block_size"] == 256
        assert c["min_block_size"] == 4


class TestGefenConvergenceAndComparison:
    """Convergence on a toy problem and a vs-AdamW math-correctness guard."""

    @staticmethod
    def _make_problem(seed=0):
        """Deterministic linear regression: y = X @ w_true + b_true."""
        rng = np.random.RandomState(seed)
        x = rng.randn(256, 8).astype(np.float32)
        w_true = rng.randn(8, 1).astype(np.float32)
        b_true = rng.randn(1).astype(np.float32)
        y = (x @ w_true + b_true).astype(np.float32)
        return x, y

    @staticmethod
    def _train(optimizer, x, y, init_seed, steps=200):
        """Train a fresh single Dense layer; return (initial_loss, final_loss)."""
        keras.utils.set_random_seed(init_seed)
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile(optimizer=optimizer, loss="mse")

        xt = ops.convert_to_tensor(x)
        yt = ops.convert_to_tensor(y)

        initial_loss = float(model.evaluate(xt, yt, verbose=0))
        model.fit(xt, yt, epochs=steps, batch_size=256, verbose=0)
        final_loss = float(model.evaluate(xt, yt, verbose=0))
        return initial_loss, final_loss

    def test_convergence(self):
        """Gefen drives the toy loss well below its initial value."""
        x, y = self._make_problem(seed=0)
        initial_loss, final_loss = self._train(
            Gefen(learning_rate=1e-2), x, y, init_seed=42, steps=200
        )
        assert final_loss < 0.5 * initial_loss, (
            f"insufficient decrease: initial={initial_loss}, final={final_loss}"
        )

    def test_vs_adamw(self):
        """Gefen final loss is within 2x AdamW on the same problem/init."""
        x, y = self._make_problem(seed=0)

        _, gefen_final = self._train(
            Gefen(learning_rate=1e-2), x, y, init_seed=123, steps=200
        )
        _, adamw_final = self._train(
            keras.optimizers.AdamW(learning_rate=1e-2), x, y, init_seed=123, steps=200
        )

        assert gefen_final <= 2.0 * adamw_final, (
            f"gefen_final={gefen_final} > 2x adamw_final={adamw_final}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
