"""``--resume`` restores the checkpoint's weights (review pass 6, concern 8).

``src/train/topolm/pretrain.py`` advertised "Path to a .keras checkpoint to resume
from" since the trainer's first commit, but ``config.resume_from`` was read in one
place only: ``extract_step_from_checkpoint`` for the step offset and the data seed.
``create_topolm_model`` built a FRESH model and nothing loaded the file, so a "resumed"
run trained from random weights while its step counters and figure labels said it
continued. The tests here run a real (tiny, CPU) training, resume from its checkpoint
and compare the weights the resumed run STARTS from with the weights the checkpoint
holds, read back through ``keras.models.load_model`` (an independent route).

What a resume restores, and what it does not (the help text and the README say the
same): the weights, trainable and non-trainable (the tap tables and the last penalty).
Not the optimizer state, the position of the learning-rate schedule, the data position
or the random streams: the optimizer and the schedule start afresh (a warm start), the
data seed moves by the checkpoint's step so the article order is new, and the step
counters and figure labels continue from the number in the file name
(``step_0000004.keras`` is step 4; ``final.keras`` has no number and is step 0).

Run on CPU (``CUDA_VISIBLE_DEVICES=``).
"""

import os
import shutil
from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.topolm import common as C
from train.topolm import pretrain as cli
from train.topolm import train_topolm

from .test_topolm_trainer import (  # noqa: F401  (the autouse fixture must be in scope)
    LENGTH,
    VOCAB,
    StubTokenizer,
    _config,
    _never_write_to_repo_results,
)

#: The imported autouse fixture is what keeps run directories out of ``results/``.
_AUTOUSE_FIXTURES_IN_SCOPE = (_never_write_to_repo_results,)

DATA_SEEDS: List[int] = []


def _patch_data(monkeypatch, rows: int = 16, batch_size: int = 4) -> None:
    """A synthetic packed-CLM stream; records the data seed each run asked for."""
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, VOCAB, size=(rows, LENGTH)).astype("int32")
    labels = generator.integers(0, VOCAB, size=(rows, LENGTH)).astype("int32")
    DATA_SEEDS.clear()

    def fake_load(config, preprocessor, data_seed, wrap_for_dict_output=True):
        DATA_SEEDS.append(data_seed)
        train = tf.data.Dataset.from_tensor_slices((tokens, labels)).batch(batch_size).repeat()
        validation = tf.data.Dataset.from_tensor_slices(
            (tokens[:8], labels[:8])).batch(batch_size)
        return train, validation, rows

    monkeypatch.setattr(C, "load_train_val_datasets", fake_load)


def _snapshot_at_train_begin(monkeypatch) -> List[List[np.ndarray]]:
    """Record the head's weights at the start of ``fit``, before any update."""
    taken: List[List[np.ndarray]] = []
    real = C.build_callbacks

    class Snapshot(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            taken.append([keras.ops.convert_to_numpy(w) for w in self.model.weights])

    def build(config, model, results_dir, initial_step, **kwargs):
        return [Snapshot()] + real(config, model, results_dir, initial_step, **kwargs)

    monkeypatch.setattr(C, "build_callbacks", build)
    return taken


def _train(monkeypatch, **overrides) -> Dict[str, Any]:
    _patch_data(monkeypatch)
    config = _config(eval_every_steps=2, save_dir="run", **overrides)
    return train_topolm(config, preprocessor=StubTokenizer(), run_topography=False)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """``final.keras`` of a real 4-step run, copied to ``step_0000004.keras``.

    A step file is only written at a checkpoint interval, where the generation probe
    also fires (and decodes ids of a 300-row vocabulary with the real tokenizer), so
    the run's ``final.keras`` is renamed instead: the same archive, with the step in
    its name the way ``extract_step_from_checkpoint`` reads it. One training for the
    whole module; the run directory is redirected under ``tmp_path_factory`` like the
    autouse fixture does for the other tests.
    """
    root = tmp_path_factory.mktemp("resume_source")
    with pytest.MonkeyPatch.context() as patch:
        real_run_dir = C._run_dir
        patch.setattr(C, "_run_dir", lambda config: str(
            root / "runs" / os.path.basename(real_run_dir(config))))
        result = _train(patch, train_steps=4)
    final = os.path.join(result["results_dir"], "checkpoints", "final.keras")
    assert os.path.isfile(final)
    path = str(root / "step_0000004.keras")
    shutil.copyfile(final, path)
    return path


def _saved_weights(path: str) -> List[np.ndarray]:
    model = keras.models.load_model(path)
    return [keras.ops.convert_to_numpy(w) for w in model.weights]


def _max_difference(left: List[np.ndarray], right: List[np.ndarray]) -> float:
    assert len(left) == len(right)
    return max(float(np.abs(a.astype("float64") - b.astype("float64")).max())
               for a, b in zip(left, right) if a.size)


class TestAResumedRunStartsFromTheSavedWeights:
    def test_the_weights_at_the_first_step_are_the_checkpoints(self, monkeypatch, checkpoint):
        saved = _saved_weights(checkpoint)
        taken = _snapshot_at_train_begin(monkeypatch)
        _train(monkeypatch, train_steps=2, resume_from=checkpoint)
        (start,) = taken
        assert _max_difference(start, saved) == 0.0

    def test_a_fresh_run_does_not_start_from_them(self, monkeypatch, checkpoint):
        """The instrument can fail: the same comparison on a run with no --resume."""
        saved = _saved_weights(checkpoint)
        taken = _snapshot_at_train_begin(monkeypatch)
        _train(monkeypatch, train_steps=2)
        (start,) = taken
        assert _max_difference(start, saved) > 1e-3

    def test_the_checkpoint_differs_from_the_initial_weights_of_the_same_seed(
            self, monkeypatch, checkpoint):
        """So equality above cannot be an accident of a checkpoint that never trained."""
        taken = _snapshot_at_train_begin(monkeypatch)
        _train(monkeypatch, train_steps=2)
        (initial,) = taken
        assert _max_difference(initial, _saved_weights(checkpoint)) > 1e-3

    def test_the_step_offset_and_the_data_seed_still_follow_the_file_name(
            self, monkeypatch, checkpoint):
        result = _train(monkeypatch, train_steps=2, resume_from=checkpoint)
        assert DATA_SEEDS == [_config().seed + 4]
        assert os.path.isdir(result["results_dir"])


class TestAResumeThatCannotBeHonouredIsRefused:
    def test_a_missing_file_is_refused_by_its_path_before_any_run_work(
            self, monkeypatch, tmp_path):
        _patch_data(monkeypatch)
        missing = str(tmp_path / "no" / "step_0000004.keras")
        with pytest.raises(FileNotFoundError, match="no/step_0000004.keras"):
            train_topolm(_config(resume_from=missing, save_dir="run"),
                         preprocessor=StubTokenizer(), run_topography=False)
        assert DATA_SEEDS == [], "the dataset was built before the checkpoint was checked"

    def test_a_file_that_is_not_a_keras_archive_is_refused(self, monkeypatch, tmp_path):
        _patch_data(monkeypatch)
        wrong = tmp_path / "step_0000004.h5"
        wrong.write_bytes(b"x")
        with pytest.raises(ValueError, match=r"\.keras"):
            train_topolm(_config(resume_from=str(wrong), save_dir="run"),
                         preprocessor=StubTokenizer(), run_topography=False)

    def test_a_checkpoint_of_another_vocabulary_is_refused_by_name(
            self, monkeypatch, checkpoint):
        _patch_data(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            train_topolm(_config(vocab_size=VOCAB + 64, resume_from=checkpoint, save_dir="run"),
                         preprocessor=StubTokenizer(), run_topography=False)
        message = str(excinfo.value)
        assert "resume_from" in message and checkpoint in message
        assert "vocab_size" in message

    def test_a_checkpoint_of_another_depth_is_refused_by_name(self, monkeypatch, checkpoint):
        _patch_data(monkeypatch)
        with pytest.raises(ValueError) as excinfo:
            train_topolm(_config(num_layers=3, resume_from=checkpoint, save_dir="run"),
                         preprocessor=StubTokenizer(), run_topography=False)
        assert "resume_from" in str(excinfo.value)


class TestTheFlagSaysWhatItDoes:
    def test_the_help_names_what_is_and_is_not_restored(self):
        action = next(a for a in cli._build_parser()._actions if "--resume" in a.option_strings)
        text = " ".join(action.help.split())
        assert "weights" in text
        for not_restored in ("optimizer", "learning-rate", "data position", "random"):
            assert not_restored in text, not_restored
        assert "step_" in text and "final.keras" in text

    def test_the_readme_says_the_same(self):
        readme = open(os.path.join(os.path.dirname(cli.__file__), "README.md"),
                      encoding="utf-8").read()
        section = readme.split("`--resume`", 1)[1][:1800]
        text = " ".join(section.split())
        assert "weights" in text and "optimizer" in text and "final.keras" in text
