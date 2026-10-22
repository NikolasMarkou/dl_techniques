"""SAM 3's ``build_sam3_dataset`` states its length, so ``fit`` logs true means.

The source is a ``from_generator`` dataset; with unknown cardinality Keras 3.8
counts one phantom zero-valued batch per epoch in its loss trackers, so
``train_sam3``'s logged ``loss``, ``val_loss`` and per-term losses were
``n / (n + 1)`` of the mean over ``n`` batches (plan-2026-10-14T062616-81938e81
D-258, D-262). ``val_box_iou`` and the other selection metrics are written by
``_Sam3EvalCallback`` from ``evaluate_sam3``, which iterates the dataset itself,
and were never affected. The mechanism is pinned by value, on a
stub, in ``tests/test_train/test_sam2/test_the_pipeline_states_its_length_and_redraws_its_prompts.py``;
here the property that removes it: the stated batch count.
"""

import pytest

from dl_techniques.models.vision_language.sam.sam3 import Sam3Image
from dl_techniques.models.vision_language.sam.sam3.training_model import Sam3TrainingModel
from train.sam3.data import build_sam3_dataset


@pytest.fixture(scope="module")
def tiny_model() -> Sam3TrainingModel:
    return Sam3TrainingModel(Sam3Image.from_variant("tiny"))


@pytest.mark.parametrize("num_samples, batch_size, batches", [(8, 4, 2), (10, 4, 2), (4, 4, 1)])
def test_the_dataset_states_its_batch_count(tiny_model, num_samples, batch_size, batches) -> None:
    dataset = build_sam3_dataset(tiny_model, num_samples=num_samples, batch_size=batch_size, seed=1)
    assert int(dataset.cardinality()) == batches
    assert sum(1 for _ in dataset) == batches
