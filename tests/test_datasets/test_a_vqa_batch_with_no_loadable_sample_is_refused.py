"""`VQADataSequence` refuses a batch in which no sample could be loaded.

Each failing sample is logged and replaced by the last good one, so a batch
whose FIRST samples fail and whose others load is short but trainable. A
batch in which EVERY sample fails used to reach `np.stack([])` and die with
"need at least one array to stack", which names neither the batch nor the
cause. That is what NanoVLM's trainer hit on the placeholder rows of the
deleted `load_cauldron_sample()` (plan-2026-10-14T062616-81938e81 F-4).
"""

import numpy as np
import pytest
from PIL import Image

from dl_techniques.datasets.vqa_dataset import VQADataProcessor, create_vqa_dataset

SIZE = 16


def _processor():
    return VQADataProcessor(image_size=SIZE, max_text_length=12, vocab_size=64)


def _row(path):
    return {"image_path": str(path), "question": "what?", "answer": "this."}


def test_a_batch_of_missing_images_is_refused_by_name(tmp_path):
    rows = [_row(tmp_path / f"missing_{i}.png") for i in range(2)]
    sequence = create_vqa_dataset(rows, _processor(), batch_size=2, shuffle=False)
    with pytest.raises(ValueError) as excinfo:
        sequence[0]
    message = str(excinfo.value)
    assert "none of its 2 samples could be loaded" in message, message
    assert "missing_0.png" in message, message


def test_a_batch_with_one_loadable_image_still_loads(tmp_path):
    """The control: the refusal is for an EMPTY batch only."""
    good = tmp_path / "good.png"
    Image.fromarray(np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)).save(good)
    rows = [_row(good), _row(tmp_path / "missing.png")]
    sequence = create_vqa_dataset(rows, _processor(), batch_size=2, shuffle=False)
    inputs, labels = sequence[0]
    assert inputs["images"].shape == (2, SIZE, SIZE, 3)
    assert labels.shape == (2, 12)
