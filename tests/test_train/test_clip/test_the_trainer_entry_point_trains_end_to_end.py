"""`train.clip.train_clip.main` must train, from argv to a saved model.

Rationale
---------
`test_the_trainer_steps_do_not_raise.py` drives `CLIPTrainer.train_step` on a
hand-built model and passed for months while `main()` could not start:

    ValueError: You tried to call `count_params` on layer 'clip', but the
    layer isn't built.

at `train_clip.py:631` (plan-2026-10-14T062616-81938e81 F-4). Past that line
nothing had ever executed. Running `main()` end to end then showed two more
defects the step-level tests could not see:

* `metric.update_state(outputs)` bound the output dict to `y_true`, so every
  step with a metric raised `ValueError: y_pred cannot be None` (the step
  tests build `CLIPTrainer` without metrics);
* the only data path without files was a demo vocabulary of
  ``"word_0".."word_999"``, so every real caption tokenized to the same
  ``<unk>`` run and the text tower saw ONE input for the whole dataset.

This file drives `main(argv)` itself, on a tiny configuration, with every
output under `tmp_path`. The autouse guard in `tests/conftest.py` fails the
test if anything lands in the repo-root `results/`.
"""

import csv
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


def test_synthetic_run_trains_two_steps_and_saves_a_loadable_model(tmp_path):
    """Two single-step epochs on 8 fixed pairs, read back from the CSV."""
    run_dir = main([
        *_TINY, "--synthetic", "--synthetic-samples", str(_BATCH),
        "--batch-size", str(_BATCH), "--epochs", "2", "--steps-per-epoch", "1",
        "--output-dir", str(tmp_path / "run"),
    ])

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
    # The metrics ran (they raised before the update_state fix).
    assert all(0.0 <= float(row["clip_i2t_accuracy"]) <= 1.0 for row in rows)

    restored = keras.models.load_model(run_dir / "final_model.keras")
    assert restored.image_size == 64 and restored.embed_dim == 16
    assert restored.vision_layers == 1 and restored.vocab_size == 64


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
def test_the_command_line_needs_exactly_one_data_source(argv):
    with pytest.raises(SystemExit):
        main(list(argv))
