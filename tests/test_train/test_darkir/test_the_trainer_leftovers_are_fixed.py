"""DarkIR trainer: the LR decays over the real run, the analysis runs, the synthetic set is lazy, every flag is read.

Guard for plan-2026-10-14T062616-81938e81 step 13.5 (D-105, D-106). Measured at
ffa4c2870:

- The cosine decay length was ``epochs * 1000`` (synthetic) or ``epochs * 500``
  (paired), neither of which is a step count: the synthetic set at batch 8 has
  6,250 steps per epoch, so the LR reached its floor after 16 % of the run.
- ``run_model_analysis`` raised ``ValueError: Data is required for:
  {'information_flow'}`` on every run (the analyzer's information-flow
  analysis defaults on and needs data, and the call passes none), and its
  ``except`` logged it.
- ``_create_synthetic_dataset`` resized all 50,000 CIFAR images to ``img_size``
  and darkened them in NumPy up front (about 10 GB per copy at 128 px).
- The parser came from ``create_base_argument_parser``, whose ``--dataset``,
  ``--image-size``, ``--lr-schedule`` and ``--show-plots`` the trainer never read.
"""

import argparse
import json
import subprocess
import sys
import textwrap

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.darkir import train_darkir as module


class _Stop(Exception):
    pass


def _pairs(n=24, size=16):
    x = np.zeros((n, size, size, 3), dtype=np.float32)
    return tf.data.Dataset.from_tensor_slices((x, x)).batch(4)


def test_the_cosine_decay_spans_the_run_after_the_warmup(monkeypatch):
    seen = {}

    def fake_schedule(config):
        seen.update(config)
        raise _Stop

    train_ds = _pairs()  # 6 batches per epoch
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "load_restoration_dataset",
                        lambda *a, **k: (train_ds, train_ds, 24))
    monkeypatch.setattr(module, "learning_rate_schedule_builder", fake_schedule)
    # The fields `train_model` reads, so this runs against any parser.
    args = argparse.Namespace(
        gpu=None, dataset_path=None, img_size=16, batch_size=4, variant="medium",
        side_loss=False, side_loss_weight=0.2, warmup_steps=4, learning_rate=2e-4,
        epochs=5, weight_decay=1e-4, ssim_weight=0.2, perceptual_weight=0.0,
        output_dir="unused", patience=1, seed=42,
    )
    with pytest.raises(_Stop):
        module.train_model(args)
    # WarmupSchedule starts the decay after the warmup: 5 * 6 - 4 steps.
    assert seen["decay_steps"] == 26
    assert seen["warmup_steps"] == 4


def test_the_decay_refuses_a_dataset_of_unknown_length():
    infinite = _pairs().repeat()
    with pytest.raises(ValueError, match="cardinality"):
        module.cosine_decay_steps(infinite, epochs=2, warmup_steps=0)


def _tiny_model():
    inputs = keras.Input((8, 8, 3))
    return keras.Model(inputs, keras.layers.Conv2D(3, 3, padding="same")(inputs))


def test_the_model_analysis_runs_and_writes_its_output(tmp_path):
    history = keras.callbacks.History()
    history.history = {"loss": [1.0, 0.8], "val_loss": [1.1, 0.9],
                       "psnr": [10.0, 11.0], "val_psnr": [9.0, 10.0]}
    module.run_model_analysis(_tiny_model(), history, "tiny", str(tmp_path))
    written = list((tmp_path / "model_analysis").rglob("*"))
    assert any(path.suffix == ".json" for path in written), written


def test_a_failing_model_analysis_fails_the_run(monkeypatch, tmp_path):
    def broken(self, *args, **kwargs):
        raise RuntimeError("analysis broke")

    monkeypatch.setattr(module.ModelAnalyzer, "analyze", broken)
    history = keras.callbacks.History()
    history.history = {"loss": [1.0], "val_loss": [1.0]}
    with pytest.raises(RuntimeError, match="analysis broke"):
        module.run_model_analysis(_tiny_model(), history, "tiny", str(tmp_path))


def test_the_synthetic_pairs_are_the_documented_degradation():
    rng = np.random.default_rng(0)
    images = rng.random((3, 8, 8, 3)).astype(np.float32)
    ds = module.synthetic_low_light_pairs(images, (16, 16), noise_seed=7)
    first = [(np.asarray(low), np.asarray(high)) for low, high in ds]
    second = [np.asarray(low) for low, _ in ds]
    for (low, high), image, again in zip(first, images, second):
        assert low.shape == high.shape == (16, 16, 3)
        np.testing.assert_allclose(
            high, tf.image.resize(image, (16, 16)).numpy(), rtol=0, atol=1e-6)
        noise = low - high ** module.SYNTHETIC_GAMMA
        unclipped = (low > 0) & (low < 1)
        # The noise is the documented std (768 draws per image: within 25 %).
        assert abs(noise[unclipped].std() - module.SYNTHETIC_NOISE_STD) < 0.005
        # One fixed noisy copy per image: the same on every pass.
        np.testing.assert_array_equal(low, again)


def test_the_synthetic_set_is_not_materialised(tmp_path):
    """Build the 256 px synthetic set over 1,000 images; the old code held ~4 GB."""
    script = textwrap.dedent("""
        import json, resource, sys
        import numpy as np
        from train.darkir import train_darkir as module
        images = np.random.default_rng(0).random((1000, 32, 32, 3)).astype(np.float32)
        module.load_dataset = lambda name: ((images, None), (images[:100], None), None, None)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        train_ds, val_ds, n = module._create_synthetic_dataset((256, 256), 8)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps({"grew_mib": (after - before) / 1024}))
    """)
    env_python = sys.executable
    out = subprocess.run(
        [env_python, "-c", script], capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    grew = json.loads(out.stdout.strip().splitlines()[-1])["grew_mib"]
    # 1,000 x 256 x 256 x 3 float32 is 750 MiB per materialised copy.
    assert grew < 300, f"building the synthetic set grew the peak RSS by {grew:.0f} MiB"


def test_every_offered_flag_is_one_the_trainer_reads():
    options = {
        option for action in module.build_parser()._actions
        for option in action.option_strings
    }
    for unread in ("--dataset", "--image-size", "--lr-schedule", "--show-plots"):
        assert unread not in options, unread
    assert {"--epochs", "--batch-size", "--learning-rate", "--weight-decay",
            "--patience", "--gpu", "--img-size", "--output-dir"} <= options
