"""Fast unit smoke test for the THERA trainer (``train.thera.train_thera``).

Builds a tiny corpus + a deliberately small ``Thera`` (so the nested-tape
``train_step`` runs in a few seconds), wraps it in :class:`TheraTrainingModel`,
and asserts:

- ``model.fit`` runs without error; ``loss`` / ``tv`` / ``psnr`` are finite,
  with ``loss > 0`` and ``tv > 0`` (the nested-tape Jacobian-TV is active).
- The nested tape produced REAL weight updates (a trainable var changed and no
  weights went NaN) -- proving second-order grads flowed (STOP-IF #1 family).
- ``test_step`` path: ``model.evaluate`` returns finite ``val_loss``.

The full ``train_thera.train`` script run is step 12; here we exercise only the
trainer + a few-step unit smoke (NO full script run).
"""

import os
import numpy as np
import pytest
import keras

# Build a genuinely tiny inner Thera directly (the 'air' factory size already
# uses hidden_dim=32; we construct it by hand with a 32-feat / 2-block EDSR
# backbone for speed -- 'pro'/'plus' Swin/ConvNeXt tails are heavy).
from dl_techniques.models.thera import EDSRBackbone, build_thera_tail
from dl_techniques.models.thera.model import Thera, DEFAULT_K_INIT

from train.thera.train_thera import TheraTrainingModel, TheraConfig, main  # noqa: F401
from train.thera.data import build_arbitrary_scale_dataset


# ---------------------------------------------------------------------


def _write_tiny_corpus(tmpdir, n=6, hw=128):
    """Write ``n`` random RGB PNGs of side ``hw`` into ``tmpdir``."""
    import tensorflow as tf

    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, size=(hw, hw, 3), dtype=np.uint8)
        png = tf.io.encode_png(tf.convert_to_tensor(arr))
        tf.io.write_file(os.path.join(tmpdir, f"img_{i:02d}.png"), png)
    return str(tmpdir)


def _build_small_thera():
    """A small Thera: 32-feat/2-block EDSR backbone + 'air' tail, hidden_dim=16."""
    return Thera(
        hidden_dim=16,
        out_dim=3,
        backbone=EDSRBackbone(num_feats=32, num_blocks=2, name="edsr_small"),
        tail=build_thera_tail("air"),
        k_init=DEFAULT_K_INIT,
        components_init_scale=16.0,
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("thera_corpus")
    return _write_tiny_corpus(d, n=6, hw=128)


@pytest.fixture
def dataset(corpus):
    # source_size=24 + target_samples=24 keeps the forward small/fast while the
    # crop>=target_size>=target_samples invariant still holds for scale_min 1.2.
    return build_arbitrary_scale_dataset(
        corpus,
        source_size=24,
        target_samples=24,
        scale_range=(1.2, 2.0),
        augment_scale_range=(1.0, 1.5),
        augment_scale_prob=0.5,
        batch_size=2,
        shuffle=True,
        seed=0,
        repeat=True,
    )


# ---------------------------------------------------------------------


class TestTheraTrainer:
    def test_config_dataclass(self):
        cfg = TheraConfig(backbone="edsr-baseline", size="air")
        assert cfg.tv_weight > 0.0
        assert cfg.max_grad_norm == 10.0
        with pytest.raises(ValueError):
            TheraConfig(backbone="bogus")
        with pytest.raises(ValueError):
            TheraConfig(size="bogus")

    def test_fit_runs_and_metrics_finite(self, dataset):
        model = TheraTrainingModel(_build_small_thera(), tv_weight=1e-4)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
            jit_compile=False,
        )
        history = model.fit(dataset, steps_per_epoch=2, epochs=2, verbose=0)

        loss = history.history["loss"][-1]
        tv = history.history["tv"][-1]
        psnr = history.history["psnr"][-1]
        mae = history.history["mae"][-1]

        assert np.isfinite(loss) and loss > 0.0, f"loss not finite/positive: {loss}"
        assert np.isfinite(tv) and tv > 0.0, f"tv not active: {tv}"
        assert np.isfinite(psnr), f"psnr not finite: {psnr}"
        assert np.isfinite(mae), f"mae not finite: {mae}"

    def test_nested_tape_updates_weights(self, dataset):
        """The outer/inner-tape composition must produce real weight updates."""
        thera = _build_small_thera()
        model = TheraTrainingModel(thera, tv_weight=1e-4)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-2),
            jit_compile=False,
        )

        # Build via one forward so trainable_variables exist; snapshot a var.
        batch = next(iter(dataset))
        _ = model(
            (batch["source"], batch["target_coords"],
             keras.ops.reshape(batch["scale"], (-1, 1)) ** -2.0),
            training=False,
        )
        # heat_field.components is a global shared weight that should get TV +
        # recon grads (D-010 asserts non-None grads to it).
        var = thera.hypernetwork.heat_field.components
        before = np.array(var)

        model.fit(dataset, steps_per_epoch=3, epochs=1, verbose=0)

        after = np.array(var)
        assert not np.allclose(before, after), (
            "heat_field.components did not change -> no grad flow through the "
            "nested tape (STOP-IF #1)."
        )
        # No NaNs anywhere in the inner model after training.
        for w in thera.weights:
            assert np.all(np.isfinite(np.array(w))), f"non-finite weight: {w.path}"

    def test_evaluate_returns_finite_val_metrics(self, dataset):
        model = TheraTrainingModel(_build_small_thera(), tv_weight=1e-4)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
            jit_compile=False,
        )
        model.fit(dataset, steps_per_epoch=2, epochs=1, verbose=0)
        results = model.evaluate(dataset.take(1), verbose=0, return_dict=True)
        assert "loss" in results
        assert np.isfinite(results["loss"]), f"val loss not finite: {results}"
        for k, v in results.items():
            assert np.isfinite(v), f"non-finite eval metric {k}={v}"
