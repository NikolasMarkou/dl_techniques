"""The finetune post-training analysis must read what training WROTE (F-23).

Regression guard for the defect this plan's step 15b fixed: for its whole life
``post_training_analysis`` loaded
``os.path.join(config.save_dir, "best_sentiment_model.keras")`` -- a filename
with several READ sites and **zero WRITE sites anywhere in src/**. The real
best-validation checkpoint is ``<results_dir>/best_model.keras``, written by the
``ModelCheckpoint`` that ``create_callbacks`` configures, where ``results_dir``
is the TIMESTAMPED run directory ``create_nlp_callbacks`` returns. A different
directory *and* a different filename, so ``run_post_training_analysis``
(default **True**) made every default ``train.bert.finetune`` /
``train.fnet.finetune`` run die with ``ValueError: File not found`` at the very
end, after training had completed and the final model had been saved.

Why these guards are not satisfied-by-construction:

* ``test_the_analysis_loads_the_file_the_checkpoint_callback_writes`` compares
  the two PRODUCERS -- the ``filepath`` the real ``ModelCheckpoint`` from
  ``create_nlp_callbacks`` is configured with, against the path the real
  ``run_finetune_post_training_analysis`` hands to ``keras.models.load_model``.
  It pins no path literal of its own, so it cannot be satisfied by both sides
  agreeing on a string that is wrong. It additionally lets the REAL
  ``keras.models.load_model`` run against a REAL saved ``.keras`` file, so a
  path that does not exist raises the same exception a production run raises.
* ``test_main_hands_the_run_directory_to_the_analysis`` covers the wiring the
  first guard cannot see: ``main()`` must actually carry the run directory from
  the training call to the analysis call.

Nothing here touches a dataset, a GPU, or the repo's ``results/`` directory
(the run dir is created under ``tmp_path`` via ``monkeypatch.chdir``).
"""

import os
import pickle
import sys
from typing import Any, Dict, List

import keras
import pytest

import train.bert.finetune as bert_finetune
import train.fnet.finetune as fnet_finetune
import train.common.nlp as nlp

from train.common.callbacks import best_checkpoint_path

BERT_ID = pytest.param("bert", id="bert")
FNET_ID = pytest.param("fnet", id="fnet")

MODULES = {"bert": bert_finetune, "fnet": fnet_finetune}


class _StubConfig:
    """Only the fields ``run_finetune_post_training_analysis`` reads."""

    def __init__(self, root: str) -> None:
        self.save_dir = os.path.join(root, "save_dir")
        self.full_analysis_dir = os.path.join(root, "save_dir", "full_analysis")
        self.dataset_name = "imdb_reviews"
        self.max_samples = 4
        self.max_seq_length = 8
        self.batch_size = 2
        self.encoding_name = "cl100k_base"
        self.cls_token_id = 100264
        self.sep_token_id = 100265
        self.pad_token_id = 100266
        self.mask_token_id = 100267
        self.analysis_n_samples = 2
        os.makedirs(self.save_dir, exist_ok=True)


def _tiny_saved_model(path: str) -> None:
    """Save a real (trivial) ``.keras`` file so the REAL loader can open it."""
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
    model.save(path)


def _checkpoint_filepath(callbacks: List[keras.callbacks.Callback]) -> str:
    """The single ``ModelCheckpoint``'s configured filepath."""
    paths = [
        str(cb.filepath)
        for cb in callbacks
        if isinstance(cb, keras.callbacks.ModelCheckpoint)
    ]
    assert len(paths) == 1, (
        f"expected exactly one ModelCheckpoint in the training callbacks, "
        f"got {len(paths)}: {paths!r}"
    )
    return paths[0]


@pytest.mark.parametrize("name", [BERT_ID, FNET_ID])
def test_the_analysis_loads_the_file_the_checkpoint_callback_writes(
    name, tmp_path, monkeypatch
):
    """Producer vs producer: ModelCheckpoint's filepath == the analysis' read.

    Both sides are computed by the shipped code. The test asserts they agree
    AND that the file the analysis opens actually exists -- the real
    ``keras.models.load_model`` runs, so a stale path fails exactly the way a
    production run fails.
    """
    monkeypatch.chdir(tmp_path)

    # --- producer 1: what the TRAINING path is configured to write ----------
    callbacks, results_dir = nlp.create_nlp_callbacks(
        model_name=f"{name}-guard",
        results_dir_prefix=f"{name}_finetune",
        monitor="val_accuracy",
        patience=3,
        include_analyzer=False,
    )
    written_path = _checkpoint_filepath(callbacks)
    _tiny_saved_model(written_path)  # stand in for a ModelCheckpoint save

    # --- the rest of a finished run's artifacts ------------------------------
    config = _StubConfig(str(tmp_path))
    _tiny_saved_model(
        os.path.join(config.save_dir, nlp.sentiment_final_model_filename(name))
    )
    with open(os.path.join(config.save_dir, "training_history.pkl"), "wb") as f:
        pickle.dump({"loss": [1.0, 0.5]}, f)

    # --- neutralise every boundary EXCEPT the model loading ------------------
    loaded: List[str] = []
    real_load_model = keras.models.load_model

    def _recording_load_model(path, *args, **kwargs):
        loaded.append(str(path))
        return real_load_model(path, *args, **kwargs)

    analyzed: Dict[str, Any] = {}

    class _RecordingAnalyzer:
        def __init__(self, models, training_history, config, output_dir):
            analyzed["model_keys"] = list(models)
            analyzed["output_dir"] = output_dir

        def analyze(self, data):
            analyzed["analyzed"] = True

    monkeypatch.setattr(keras.models, "load_model", _recording_load_model)
    monkeypatch.setattr(nlp, "create_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(nlp, "load_text_dataset", lambda *a, **k: object())
    monkeypatch.setattr(
        nlp, "preprocess_classification_dataset", lambda *a, **k: object()
    )
    monkeypatch.setattr(nlp, "prepare_data_for_analyzer", lambda *a, **k: object())
    monkeypatch.setattr(nlp, "AnalysisConfig", lambda **k: object())
    monkeypatch.setattr(nlp, "ModelAnalyzer", _RecordingAnalyzer)

    # --- producer 2: what the ANALYSIS path resolves to ----------------------
    nlp.run_finetune_post_training_analysis(
        config,
        model_name=name,
        create_initial_model=lambda: keras.Sequential(
            [keras.layers.Dense(1, input_shape=(1,))]
        ),
        results_dir=results_dir,
    )

    assert analyzed.get("analyzed") is True, (
        "the analysis never reached ModelAnalyzer.analyze; this probe measured "
        "nothing"
    )
    assert analyzed["model_keys"] == [
        "Initial_Model", "Best_Model(ValAcc)", "Final_Model",
    ]
    assert len(loaded) == 2, f"expected 2 snapshot loads, got {loaded!r}"

    best_read_path = loaded[0]
    assert best_read_path == written_path, (
        "the post-training analysis reads a DIFFERENT path from the one the "
        f"training callbacks write (F-23):\n  written by ModelCheckpoint: "
        f"{written_path}\n  read by the analysis:      {best_read_path}"
    )
    assert os.path.exists(best_read_path), (
        f"the analysis resolved {best_read_path}, which no training path wrote"
    )
    # ...and the path really is the run dir's, not the static save_dir's.
    assert best_read_path == best_checkpoint_path(results_dir)
    assert not best_read_path.startswith(config.save_dir)


@pytest.mark.parametrize("name", [BERT_ID, FNET_ID])
def test_main_hands_the_run_directory_to_the_analysis(name, monkeypatch):
    """``main()`` must carry the training call's run dir into the analysis.

    The first guard proves the shared analysis function reads the right path
    GIVEN a run directory. This one proves ``main()`` supplies it -- before the
    fix, ``finetune_sentiment_model``'s ``results_dir`` was a discarded local
    and the analysis got only ``config``.
    """
    module = MODULES[name]
    sentinel_dir = f"/tmp/probe_{name}_run_dir"
    captured: Dict[str, Any] = {}

    def _fake_train(config, *args, **kwargs):
        return "MODEL", "HISTORY", sentinel_dir

    def _fake_analysis(config, *args, **kwargs):
        captured["config"] = config
        captured["positional"] = args
        captured["keyword"] = kwargs

    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "finetune_sentiment_model", _fake_train)
    monkeypatch.setattr(module, "post_training_analysis", _fake_analysis)
    monkeypatch.setattr(module, "create_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(module, "evaluate_model", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["prog"])

    module.main()

    assert "config" in captured, (
        f"{module.__name__}.main() never called post_training_analysis; "
        f"run_post_training_analysis defaults to True, so it must"
    )
    supplied = list(captured["positional"]) + list(captured["keyword"].values())
    assert sentinel_dir in supplied, (
        "main() called post_training_analysis WITHOUT the run directory "
        f"returned by finetune_sentiment_model (got {supplied!r}); the "
        "analysis would fall back to a path nothing writes (F-23)"
    )
