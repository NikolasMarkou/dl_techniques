"""`train.clip.train_clip.main` must train, from argv to a saved model.

Rationale
---------
The step-level tests of the old hand-written `CLIPTrainer` drove its
`train_step` on a hand-built model and passed for months while `main()` could
not start:

    ValueError: You tried to call `count_params` on layer 'clip', but the
    layer isn't built.

at `train_clip.py:631` (plan-2026-10-14T062616-81938e81 F-4). Past that line
nothing had ever executed. Running `main()` end to end then showed two more
defects the step-level tests could not see:

* `metric.update_state(outputs)` bound the output dict to `y_true`, so every
  step with a metric raised `ValueError: y_pred cannot be None` (the step
  tests built the trainer without metrics);
* the only data path without files was a demo vocabulary of
  ``"word_0".."word_999"``, so every real caption tokenized to the same
  ``<unk>`` run and the text tower saw ONE input for the whole dataset.

This file drives `main(argv)` itself, on a tiny configuration, with every
output under `tmp_path`. The autouse guard in `tests/conftest.py` fails the
test if anything lands in the repo-root `results/`.

Since plan-2026-10-14T062616-81938e81 D-036 the trainer has no loop of its
own: `main` compiles CLIP through `compile_for_contrastive_training` and
trains with stock `keras.Model.fit`. The synthetic run asserts that, by
spying on `keras.Model.fit` and counting the batches its callbacks see.
"""

import csv
import json
import math

import keras
import numpy as np
import pytest
from PIL import Image

from train.clip.train_clip import (
    CLIPDataLoader,
    build_caption_vocabulary,
    main,
)
from train.common.contrastive import StackedCLIPLogits

#: A ViT-B/32 row shrunk to seconds: 4 patches of 32px, one block per tower,
#: widths divisible by the row's own head and kv-head counts (12/4 and 8/8).
_TINY = (
    "--variant", "ViT-B/32", "--image-size", "64", "--context-length", "8",
    "--vision-layers", "1", "--vision-width", "48",
    "--text-layers", "1", "--text-width", "32",
    "--embed-dim", "16", "--vocab-size", "64", "--ffn-multiple-of", "8",
    "--warmup-steps", "0", "--learning-rate", "1e-3", "--seed", "3",
)
_BATCH = 8


def _losses(run_dir):
    with open(run_dir / "training_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["loss"]) for row in rows], rows


class _BatchCounter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batches = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batches += 1


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


def test_synthetic_run_trains_two_steps_and_saves_a_loadable_model(tmp_path, monkeypatch):
    """Two single-step epochs on 8 fixed pairs, read back from the CSV."""
    fit_calls = _spy_on_fit(monkeypatch)
    run_dir = main([
        *_TINY, "--synthetic", "--synthetic-samples", str(_BATCH),
        "--batch-size", str(_BATCH), "--epochs", "2", "--steps-per-epoch", "1",
        "--output-dir", str(tmp_path / "run"),
    ])

    # Training went through stock `keras.Model.fit`, once, on the bridge
    # wrapper, and every one of the 2 steps was a fit batch.
    assert len(fit_calls) == 1, fit_calls
    trained, counter, history = fit_calls[0]
    assert isinstance(trained, StackedCLIPLogits)
    assert counter.batches == 2
    with open(run_dir / "training_history.json") as handle:
        assert json.load(handle) == pytest.approx(history.history)

    assert run_dir == tmp_path / "run"
    for name in ("config.json", "training_log.csv", "training_history.json",
                 "final_model.keras"):
        assert (run_dir / name).is_file(), name

    losses, rows = _losses(run_dir)
    assert len(losses) == 2, rows
    assert all(math.isfinite(v) for v in losses), losses
    # Exact inequality: the same 8 pairs both epochs, so an optimizer that
    # applied nothing would log the same train-mode loss twice.
    assert losses[0] != losses[1], losses
    assert losses == pytest.approx(history.history["loss"])
    # The metrics ran (they raised before the update_state fix).
    assert all(0.0 <= float(row["clip_i2t_accuracy"]) <= 1.0 for row in rows)
    # The schedule's rate, read at each epoch start (warmup 0: the peak).
    assert all(math.isfinite(float(row["lr"])) for row in rows), rows

    restored = keras.models.load_model(run_dir / "final_model.keras")
    assert type(restored).__name__ == "CLIP"
    assert restored.image_size == 64 and restored.embed_dim == 16
    assert restored.vision_layers == 1 and restored.vocab_size == 64
    # The learnable temperature trained: the saved log-scale left its init.
    # Compared in float32: the Python float 2.6592 is not a float32 value, so a
    # float64 comparison would pass for an untrained model too.
    assert restored.logit_scale.numpy() != np.float32(restored.logit_scale_init)


def test_every_epoch_trains_its_steps_per_epoch_batches(tmp_path, monkeypatch):
    """3 batches per pass at 2 steps per epoch: each epoch still trains 2.

    Stock `fit` keeps one iterator across epochs and resets it only when it
    runs dry, so without a repeated dataset the second epoch would train the
    one batch left in the first pass (measured: [2, 1, 2] over 3 epochs).
    """
    fit_calls = _spy_on_fit(monkeypatch)
    run_dir = main([
        *_TINY, "--synthetic", "--synthetic-samples", str(3 * _BATCH),
        "--batch-size", str(_BATCH), "--epochs", "2", "--steps-per-epoch", "2",
        "--output-dir", str(tmp_path / "run"),
    ])
    (_, counter, _), = fit_calls
    assert counter.batches == 4
    losses, rows = _losses(run_dir)
    assert len(losses) == 2 and all(math.isfinite(v) for v in losses), rows


def test_file_run_tokenizes_each_caption_from_the_captions_vocabulary(tmp_path):
    """The image-dir path, one step, on PNGs and a captions file in tmp_path."""
    rng = np.random.default_rng(0)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    captions = [f"a photo of shape {i} in color {i % 3}" for i in range(_BATCH)]
    for i in range(_BATCH):
        pixels = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / f"img_{i:02d}.png")
    captions_file = tmp_path / "captions.txt"
    captions_file.write_text("\n".join(captions) + "\n")

    run_dir = main([
        *_TINY, "--image-dir", str(image_dir),
        "--captions-file", str(captions_file),
        "--batch-size", str(_BATCH), "--epochs", "1",
        "--output-dir", str(tmp_path / "run"),
    ])

    losses, rows = _losses(run_dir)
    assert len(losses) == 1 and math.isfinite(losses[0]), rows


def test_distinct_captions_reach_the_model_as_distinct_tokens():
    """The vocabulary comes from the captions, pad is 0, nothing is <unk>."""
    captions = ["a red car", "a blue sky", "the red sky at night"]
    vocab = build_caption_vocabulary(captions, vocab_size=64)
    assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1
    assert max(vocab.values()) < 64

    loader = CLIPDataLoader(image_size=32, context_length=8, tokenizer=vocab)
    rows = [tuple(np.array(loader.preprocess_text(c))) for c in captions]
    assert len(set(rows)) == len(captions), rows
    for caption, row in zip(captions, rows):
        n_words = len(caption.split())
        assert all(t >= 2 for t in row[:n_words]), (caption, row)
        assert all(t == 0 for t in row[n_words:]), (caption, row)


def test_the_vocabulary_is_capped_at_the_model_vocabulary():
    """Most frequent words win; every id stays below ``vocab_size``."""
    vocab = build_caption_vocabulary(["b a a", "c a b"], vocab_size=4)
    assert vocab == {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}
    with pytest.raises(ValueError, match="at least 3"):
        build_caption_vocabulary(["a"], vocab_size=2)


def test_the_data_loader_has_no_default_tokenizer():
    """A demo vocabulary that matches no caption is how every text collapsed."""
    with pytest.raises(ValueError, match="needs a tokenizer"):
        CLIPDataLoader(image_size=32, context_length=8)


def test_a_dataset_smaller_than_one_batch_is_refused(tmp_path):
    """drop_remainder empties it; the epoch used to log a loss of exactly 0.0."""
    with pytest.raises(ValueError, match="no batch"):
        main([
            *_TINY, "--synthetic", "--synthetic-samples", "4",
            "--batch-size", str(_BATCH), "--epochs", "1",
            "--output-dir", str(tmp_path / "run"),
        ])


@pytest.mark.parametrize("argv", [
    ("--epochs", "1"),
    ("--synthetic", "--image-dir", "x", "--captions-file", "y"),
    ("--image-dir", "x"),
])
def test_the_command_line_needs_exactly_one_data_source(argv, tmp_path):
    # The output dir is routed through tmp_path even though the parse refuses
    # first: a parse regression must not train into the repo-root results/
    # (D-048, D-084).
    with pytest.raises(SystemExit):
        main([*argv, "--output-dir", str(tmp_path / "run")])
