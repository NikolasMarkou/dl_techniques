"""`train.nano_vlm.train_nano_vlm.main` must train, from argv to a saved model.

Rationale
---------
The trainer's old suite (12 tests on a hand-written `NanoVLMTrainer` loop)
was green while `main` could not take a single step
(plan-2026-10-14T062616-81938e81 F-4): its only data source,
`load_cauldron_sample()`, returned three rows whose image paths did not
exist, so the first batch was empty and `np.stack` raised. The suite never
saw it because it patched the data, the model and the dataset factory. Behind
that crash two more defects waited: `NanoVLMLoss` refused NanoVLM's own
`[vision; text]` logits (a length mismatch the loop caught per step as a
"bad sample"), and the old loop was a custom training step against the house
rule of stock `fit`.

This file drives `main(argv)` itself on a tiny configuration with every
output under `tmp_path`, on the in-memory `--synthetic` source and on real
image files, and patches neither data nor model. The only patch is a spy on
`keras.Model.fit`, which proves training went through stock `fit`. The
autouse guard in `tests/conftest.py` fails the test if anything lands in the
repo-root `results/`.
"""

import csv
import json
import math

import keras
import numpy as np
import pytest
from PIL import Image

from dl_techniques.models.vision_language.nano_vlm import NanoVLM
from train.nano_vlm.train_nano_vlm import main, make_synthetic_samples

#: The 'mini' row shrunk to seconds: 32px images in 4 patches, one block per
#: tower, one fusion block, 12 text positions.
_TINY = (
    "--variant", "mini", "--vocab-size", "64", "--image-size", "32",
    "--embed-dim", "32", "--num-heads", "2", "--vision-depth", "1",
    "--text-depth", "1", "--num-fusion-layers", "1", "--max-text-length", "12",
    "--warmup-steps", "0", "--learning-rate", "1e-3", "--seed", "3",
    "--no-mixed-precision",
)


@pytest.fixture(autouse=True)
def restore_global_dtype_policy():
    """`main` sets the process-global dtype policy."""
    previous = keras.mixed_precision.global_policy()
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


class _BatchCounter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.per_epoch = []

    def on_epoch_begin(self, epoch, logs=None):
        self.per_epoch.append(0)

    def on_train_batch_end(self, batch, logs=None):
        self.per_epoch[-1] += 1


def _spy_on_fit(monkeypatch):
    """Record every `keras.Model.fit` call and count the batches it trains."""
    calls = []
    stock_fit = keras.Model.fit

    def spy(self, *args, **kwargs):
        counter = _BatchCounter()
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), counter]
        history = stock_fit(self, *args, **kwargs)
        calls.append((self, counter, history))
        return history

    monkeypatch.setattr(keras.Model, "fit", spy)
    return calls


def _losses(run_dir):
    with open(run_dir / "training_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["loss"]) for row in rows], rows


def test_synthetic_run_trains_two_steps_under_stock_fit(tmp_path, monkeypatch):
    fit_calls = _spy_on_fit(monkeypatch)
    run_dir = main([
        *_TINY, "--synthetic", "--synthetic-samples", "4", "--batch-size", "4",
        "--epochs", "2", "--steps-per-epoch", "1",
        "--output-dir", str(tmp_path / "run"),
    ])

    # Training went through stock `keras.Model.fit`, once, on the NanoVLM
    # itself, and both steps were fit batches.
    assert len(fit_calls) == 1, fit_calls
    trained, counter, history = fit_calls[0]
    assert isinstance(trained, NanoVLM)
    assert counter.per_epoch == [1, 1]
    with open(run_dir / "training_history.json") as handle:
        assert json.load(handle) == pytest.approx(history.history)

    assert run_dir == tmp_path / "run"
    for name in ("config.json", "training_log.csv", "training_history.json",
                 "final_model.keras"):
        assert (run_dir / name).is_file(), name

    losses, rows = _losses(run_dir)
    assert len(losses) == 2, rows
    assert all(math.isfinite(v) for v in losses), losses
    # Exact inequality: the same 4 samples both epochs, so an optimizer that
    # applied nothing would log the same train-mode loss twice.
    assert losses[0] != losses[1], losses
    assert losses == pytest.approx(history.history["loss"])
    assert all(math.isfinite(float(row["lr"])) for row in rows), rows

    restored = keras.models.load_model(run_dir / "final_model.keras")
    assert isinstance(restored, NanoVLM)
    assert restored.vision_config["img_size"] == 32
    assert restored.text_config["max_seq_len"] == 12
    assert restored.vocab_size == 64
    assert restored.output_projection is None  # 'mini' is tied
    for saved, live in zip(restored.weights, trained.weights):
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(saved), keras.ops.convert_to_numpy(live))


def test_every_epoch_trains_its_steps_per_epoch_batches(tmp_path, monkeypatch):
    """3 batches per pass at 2 steps per epoch: each epoch still trains 2.

    Stock `fit` keeps one iterator across epochs and resets it only when it
    runs dry; without the trainer's `.repeat()` the epochs train `[2, 1]`
    and Keras warns "ran out of data" (an error under this suite's filter).
    """
    fit_calls = _spy_on_fit(monkeypatch)
    main([
        *_TINY, "--synthetic", "--synthetic-samples", "6", "--batch-size", "2",
        "--epochs", "2", "--steps-per-epoch", "2",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert fit_calls[0][1].per_epoch == [2, 2]


def test_fewer_samples_than_one_batch_is_refused_before_anything_is_built(tmp_path):
    with pytest.raises(ValueError, match=r"no full batch: 3 samples // batch_size=4 == 0"):
        main([*_TINY, "--synthetic", "--synthetic-samples", "3", "--batch-size", "4",
              "--epochs", "1", "--output-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run" / "final_model.keras").exists()


def _write_json_dataset(tmp_path, n, missing=False):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        path = tmp_path / f"img_{i}.png"
        if not missing:
            Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(path)
        rows.append({"image_path": str(path), "question": f"what is {i}?",
                     "answer": f"it is {i}."})
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(rows))
    return json_path


def test_a_json_dataset_of_image_files_trains(tmp_path, monkeypatch):
    fit_calls = _spy_on_fit(monkeypatch)
    json_path = _write_json_dataset(tmp_path, n=4)
    run_dir = main([*_TINY, "--data-json", str(json_path), "--batch-size", "2",
                    "--epochs", "1", "--output-dir", str(tmp_path / "run")])
    assert fit_calls[0][1].per_epoch == [2]
    losses, rows = _losses(run_dir)
    assert len(losses) == 1 and math.isfinite(losses[0]), rows


def test_a_json_dataset_whose_images_are_all_missing_is_refused_by_name(tmp_path):
    """The F-4 crash, now a named refusal instead of `np.stack([])`."""
    json_path = _write_json_dataset(tmp_path, n=2, missing=True)
    with pytest.raises(Exception, match="none of its 2 samples could be loaded"):
        main([*_TINY, "--data-json", str(json_path), "--batch-size", "2",
              "--epochs", "1", "--output-dir", str(tmp_path / "run")])


def test_exactly_one_data_source_is_required(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([*_TINY, "--output-dir", str(tmp_path / "run")])
    assert "one of the arguments --synthetic --data-json is required" in capsys.readouterr().err


def test_synthetic_samples_have_the_processor_layout():
    inputs, labels = make_synthetic_samples(8, image_size=16, seq_len=10,
                                            vocab_size=50, seed=0)
    tokens = inputs["text_tokens"]
    assert labels is tokens
    assert inputs["images"].shape == (8, 16, 16, 3)
    assert inputs["images"].min() >= -1.0 and inputs["images"].max() < 1.0
    assert np.all(tokens[:, 0] == 1)
    for row, mask in zip(tokens, inputs["attention_mask"]):
        eos = int(np.flatnonzero(row == 2)[0])
        assert np.all(row[1:eos] >= 3) and np.all(row[eos + 1:] == 0)
        np.testing.assert_array_equal(mask, (row != 0).astype(np.int32))
    # Not one sequence repeated.
    assert len({row.tobytes() for row in tokens}) == 8
