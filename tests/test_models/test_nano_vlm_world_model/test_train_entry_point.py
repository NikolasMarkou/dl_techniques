"""``train_score_vlm`` is the package's ONLY training entry point, and it could
not be called at all.

``tests/test_train.py`` repaired three blockers *inside* ``ScoreVLMTrainer``, but
every one of its probes constructs the trainer itself. Nothing ever called the
function that constructs the trainer -- it has zero callers in ``src/`` and had
zero tests -- so two further blockers survived that whole round, in this masking
order:

(a) ``optimizer_builder(optimizer_config)``. The builder is
    ``(config, lr_schedule)`` and ``lr_schedule`` has no default, so this raised
    ``TypeError: optimizer_builder() missing 1 required positional argument``
    two statements before ``ScoreVLMTrainer`` was even constructed. The learning
    rate was also sitting unused in the config dict, which the builder never
    reads.
(b) ``for step, (images, text_tokens) in enumerate(train_dataset)``.
    ``keras.utils.Sequence`` is ``PyDataset`` in Keras 3 and defines no
    ``__iter__``, so this used Python's legacy ``__getitem__`` protocol, which
    walks 0, 1, 2, ... until ``IndexError`` and ignores ``__len__``. Against a
    dataset that generates a batch for any index -- including this module's own
    ``example_training`` ``DummyDataset`` -- epoch 0 never ended.

This file runs the entry point end to end on a tiny synthetic batch. It is the
"run the CLI, don't just import the module" test: importing ``train.py``, or
driving ``ScoreVLMTrainer`` directly, passes with both defects fully present.

``generation_mode='text_to_image'`` deliberately: the denoiser depths are
hard-coded (12 / 12 / 16 layers), and at ``'joint'`` the ``tf.function`` trace of
a single step costs ~180 s on this hardware against ~30 s here. That is a
tracing cost, not a defect, but it is not worth paying twice per test.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.nano_vlm_world_model.model import ScoreBasedNanoVLM
from dl_techniques.models.nano_vlm_world_model.train import train_score_vlm

BATCH = 2
IMG = 32
SEQ = 16
VOCAB = 32


def _tiny_model():
    return ScoreBasedNanoVLM(
        vision_config={"img_size": IMG, "patch_size": 16, "embed_dim": 32,
                       "depth": 1, "num_heads": 2, "output_mode": "none"},
        text_config={"vocab_size": VOCAB, "embed_dim": 32, "depth": 1,
                     "num_heads": 2, "max_seq_len": SEQ},
        diffusion_config={"num_timesteps": 20, "beta_schedule": "cosine"},
        vocab_size=VOCAB,
        generation_mode="text_to_image",
    )


class _CountingDataset(keras.utils.Sequence):
    """One batch, and it records every index it is asked for.

    ``__getitem__`` deliberately serves ANY index rather than raising
    ``IndexError`` past the end -- that is what a normal `PyDataset` does, and
    it is precisely the shape that made the old `enumerate` loop infinite. The
    hard stop after `_LIMIT` calls turns that infinite loop into a bounded,
    deterministic failure instead of a hung test.
    """

    _LIMIT = 4

    def __init__(self, length=1):
        super().__init__()
        rng = np.random.default_rng(0)
        self._images = rng.random((BATCH, IMG, IMG, 3), dtype="float32")
        self._text = rng.integers(0, VOCAB, (BATCH, SEQ)).astype("int32")
        self._length = length
        self.requested = []

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self.requested.append(idx)
        if len(self.requested) > self._LIMIT:
            raise RuntimeError(
                "the training loop asked for more batches than __len__ declares "
                f"({self.requested}) -- it is iterating past the end of the "
                "dataset and would not terminate"
            )
        return self._images, self._text


class TestTrainScoreVLMIsCallable:

    def test_entry_point_runs_one_epoch_on_a_synthetic_batch(self, tmp_path):
        dataset = _CountingDataset()
        result = train_score_vlm(
            model=_tiny_model(),
            train_dataset=dataset,
            epochs=1,
            checkpoint_dir=str(tmp_path),
            log_frequency=1,
            # `epoch % sample_every_n_epochs == 0` is TRUE at epoch 0 for every
            # value of `sample_every_n_epochs`, so the monitoring hook cannot be
            # switched off from the outside. Only `num_sample_steps` bounds its
            # cost; two steps is enough to reach the code.
            num_sample_steps=2,
        )

        assert result is None
        assert dataset.requested == [0], (
            "the loop must consume exactly len(dataset) batches per epoch"
        )
        checkpoints = sorted(tmp_path.glob("*.keras"))
        assert len(checkpoints) == 1, (
            f"expected exactly one epoch checkpoint, got {checkpoints}"
        )
        assert checkpoints[0].stat().st_size > 0

    def test_a_caller_supplied_optimizer_config_still_reaches_the_optimizer(
            self, tmp_path
    ):
        """The learning rate lives in the config dict the docstring advertises,
        but ``optimizer_builder`` takes it as a separate positional argument.
        A non-default optimizer type plus a distinctive rate must simply run."""
        dataset = _CountingDataset()
        train_score_vlm(
            model=_tiny_model(),
            train_dataset=dataset,
            epochs=1,
            optimizer_config={"type": "sgd", "learning_rate": 0.0123},
            checkpoint_dir=str(tmp_path),
            log_frequency=1,
            num_sample_steps=2,
        )
        assert dataset.requested == [0]
        assert sorted(tmp_path.glob("*.keras"))


class TestOptimizerBuilderSignatureIsWhatTrainAssumes:
    """A cheap guard against the call regressing again: the defect was a
    signature mismatch, and nothing pinned the signature from this side."""

    def test_optimizer_builder_requires_an_explicit_learning_rate(self):
        import inspect

        from dl_techniques.optimization import optimizer_builder

        params = inspect.signature(optimizer_builder).parameters
        assert list(params) == ["config", "lr_schedule"]
        assert params["lr_schedule"].default is inspect.Parameter.empty, (
            "if lr_schedule ever gains a default, the bridge in "
            "train_score_vlm can be simplified -- but do not assume it has one"
        )


class TestKerasSequenceHasNoIter:
    """The mechanism behind blocker (b), pinned so a future reader does not
    'simplify' the indexed loop back to a `for ... in dataset`."""

    def test_pydataset_does_not_define_iter(self):
        assert not hasattr(keras.utils.Sequence, "__iter__"), (
            "keras.utils.Sequence gained __iter__; the indexed loop in "
            "train_score_vlm is still correct, but the reason recorded in its "
            "D-016 anchor has changed"
        )

    def test_legacy_getitem_iteration_ignores_len(self):
        dataset = _CountingDataset(length=1)
        seen = 0
        for _ in dataset:
            seen += 1
            if seen > 2:
                break
        assert seen == 3, (
            "a plain `for ... in` over a PyDataset walks past __len__ -- this "
            "is why train_score_vlm indexes by range(len(...))"
        )
