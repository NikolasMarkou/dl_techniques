"""`train.cliffordnet.train_clip.main` trains CliffordCLIP through stock `fit`.

Rationale
---------
Until plan-2026-10-14T062616-81938e81 D-045 the CliffordCLIP trainer trained a
`ContrastiveCliffordCLIP` wrapper with its own `train_step`/`test_step` (frozen
in the step-override census), saved that WRAPPER as every step checkpoint, had
no `--output-dir` (every run wrote into the repo-root `results/`) and no step
limit, and no test had ever driven `main()`. It now trains through the D-036
bridge of `train.common.contrastive` (`compile_for_contrastive_training`) and
writes the bare `CliffordCLIP` everywhere.

This file drives `main(argv)` itself on the in-memory synthetic pairs, with the
pretraining stages skipped and every output under `tmp_path`. Nothing about
the data is monkeypatched; `keras.Model.fit` is spied on (not replaced) to
prove it is the training path and to count the batches it trains.

The model is a one-block-per-stage row added to `CliffordCLIP.MODEL_VARIANTS`
for the duration of the module (the parser's `--variant` choices are read
from that table): MEASURED, `nano` spends 245 s in the XLA compile of the
first `fit` step on GPU 1, which is the cost of the configuration, not of the
path under test. The `nano` CLI run is the Step 11 smoke in the plan record.
"""

import csv
import json
import math

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
from train.common.contrastive import StackedCLIPLogits
from train.cliffordnet.train_clip import RetrievalProbeCallback, main, parse_arguments

_BATCH = 8
#: Train 16 pairs, selection 12, test 10: three different sizes, so a validation_data, a
#: probe or a summary block built from the wrong split shows in its count.
_TRAIN_PAIRS = 2 * _BATCH
_SELECT_PAIRS = _BATCH + 4
_TEST_PAIRS = _BATCH + 2
#: Between the test split (10) and the selection slice (12): a probe cut from the test split
#: could hold only 10 pairs, one cut from the selection slice holds all 11.
_PROBE_PAIRS = _BATCH + 3
_VARIANT = "entry_point_test_tiny"

#: nano's ladder (4 stages, PatchMerging between them) at one block per stage
#: and a few channels, so the run is the shipped code path at a fraction of the
#: compile time.
_TINY_ROW = dict(
    vision_stage_channels=[8, 8, 16, 16],
    vision_stage_depths=[1, 1, 1, 1],
    vision_stage_shifts=[[1, 2], [1, 2], [1, 2], [1, 2]],
    text_channels=16,
    text_depth=2,
    text_shifts=[1, 2],
    embed_dim=16,
    vision_stochastic_depth_rate=0.0,
    text_stochastic_depth_rate=0.0,
)

#: The smallest resolution a 4-stage ladder accepts (64 / patch 4 = 16, halved
#: three times to 2x2), 2 epochs of 1 step, and every per-step side effect
#: (checkpoint, CSV row, retrieval probe, gamma probe) switched on so each one
#: runs at least once.
_ARGV = (
    "--synthetic", "--skip-pretrain", "--variant", _VARIANT,
    "--image-size", "64", "--context-length", "16", "--vision-patch-size", "4",
    "--batch-size", str(_BATCH), "--max-train-samples", str(_TRAIN_PAIRS),
    "--max-test-samples", str(_TEST_PAIRS), "--selection-images", str(_SELECT_PAIRS),
    "--epochs", "2", "--steps-per-epoch", "1",
    "--save-every-steps", "1", "--log-every-steps", "1", "--max-checkpoints", "2",
    "--probe-every-steps", "1", "--probe-num-pairs", str(_PROBE_PAIRS),
    "--gamma-probe-every-steps", "1", "--seed", "3",
)


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
        counter.fit_kwargs = dict(kwargs)
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), counter]
        history = stock_fit(self, *args, **kwargs)
        calls.append((self, counter, history))
        return history

    monkeypatch.setattr(keras.Model, "fit", spy)
    return calls


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One `main()` run shared by the assertions below (it takes ~a minute)."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(CliffordCLIP.MODEL_VARIANTS, _VARIANT, _TINY_ROW)
        fit_calls = _spy_on_fit(monkeypatch)
        out = tmp_path_factory.mktemp("cliffordclip") / "run"
        run_dir = main([*_ARGV, "--output-dir", str(out)])
    finally:
        monkeypatch.undo()
    return out, run_dir, fit_calls


def test_fit_is_the_training_path(run):
    """Exactly one `fit`, on the bridge, training every one of the 2 steps."""
    out, run_dir, fit_calls = run
    assert run_dir == out
    assert len(fit_calls) == 1, fit_calls
    trained, counter, history = fit_calls[0]
    assert isinstance(trained, StackedCLIPLogits)
    assert isinstance(trained.clip_model, CliffordCLIP)
    assert counter.batches == 2
    losses = history.history["loss"]
    assert len(losses) == 2 and all(math.isfinite(v) for v in losses), losses
    # The temperature column the old train_step logged is still there.
    assert len(history.history["logit_scale"]) == 2
    assert "val_loss" in history.history


def test_the_temperature_trained(run):
    """`logit_scale` is a CliffordCLIP weight; a step must move it.

    Compared in float32: the Python float init is not a float32 value, so a
    float64 comparison would pass for an untrained model too.
    """
    _, _, fit_calls = run
    clip_model = fit_calls[0][0].clip_model
    assert clip_model.logit_scale.numpy() != np.float32(clip_model.logit_scale_init)


def test_every_saved_model_is_the_bare_cliffordclip(run):
    """Step, final and run-root archives load as `CliffordCLIP`, weights equal."""
    out, _, fit_calls = run
    clip_model = fit_calls[0][0].clip_model
    paths = sorted((out / "checkpoints").glob("step_*.keras"))
    assert [p.name for p in paths] == ["step_0000001.keras", "step_0000002.keras"]
    for path in (out / "checkpoints" / "final.keras",
                 out / f"cliffordclip_{_VARIANT}.keras", paths[-1]):
        restored = keras.models.load_model(path)
        assert type(restored) is CliffordCLIP, (path, type(restored))
        assert len(restored.weights) == len(clip_model.weights)
        for source, target in zip(clip_model.weights, restored.weights):
            np.testing.assert_array_equal(
                np.array(source), np.array(target), err_msg=f"{path.name}: {source.path}")


def test_the_run_directory_holds_its_records(run):
    """Step CSV with loss and temperature, both probes, the summary."""
    out, _, _ = run
    with open(out / "training_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [1, 2], rows
    for row in rows:
        assert math.isfinite(float(row["loss"])), row
        assert float(row["logit_scale"]) > 1.0, row
        assert 0.0 <= float(row["clip_i2t_accuracy"]) <= 1.0, row
    with open(out / "retrieval_probes" / "probes.jsonl") as handle:
        probes = [json.loads(line) for line in handle]
    assert [p["step"] for p in probes] == [1, 2], probes
    assert all(0.0 <= p["i2t_r@1"] <= 1.0 for p in probes), probes
    assert (out / "training_summary.txt").is_file()
    assert (out / "tensorboard" / "clip").is_dir()


def test_a_zero_probe_cadence_disables_the_probe(tmp_path):
    """`--probe-every-steps 0` is documented as "disabled"; it raised
    ZeroDivisionError on the first batch."""
    probe = RetrievalProbeCallback(
        clip_model=None, save_dir=str(tmp_path), probe_every_steps=0)
    probe.set_probe_dataset(probe_ds=object(), stage_label="clip")
    probe.on_train_batch_end(0, {"loss": 1.0})
    assert probe.global_step == 1
    assert not (tmp_path / "retrieval_probes" / "probes.jsonl").exists()


@pytest.mark.parametrize("argv", [
    ("--steps-per-epoch", "0"),
    ("--loss", "siglip", "--label-smoothing", "0.1"),
])
def test_the_command_line_refuses_a_dead_setting(argv, tmp_path):
    # `parse_arguments`, not `main`: a parser that stopped refusing must fail
    # this test, never start a run (a `main` call without `--output-dir` trained
    # into the repo-root results/ under exactly that mutation).
    base = ["--synthetic", "--skip-pretrain", "--output-dir", str(tmp_path / "run")]
    assert parse_arguments(base).output_dir == str(tmp_path / "run")
    with pytest.raises(SystemExit):
        parse_arguments([*base, *argv])


def _rows(dataset):
    """Number of (image, text) rows a batched contrastive-fit dataset holds."""
    return sum(int(batch[0]["image"].shape[0]) for batch in dataset)


def test_fit_validates_on_the_selection_slice_not_the_test_split(run):
    """`validation_data` is the held-out slice of train (12 pairs), not the official
    validation split (10) and not the fit set (16)."""
    _, _, fit_calls = run
    validation = fit_calls[0][1].fit_kwargs["validation_data"]
    assert _rows(validation) == _SELECT_PAIRS, (
        f"validation_data holds {_rows(validation)} rows; the selection slice has "
        f"{_SELECT_PAIRS} and the test split {_TEST_PAIRS}")


def test_the_probes_read_the_selection_slice(run):
    """The first `--probe-num-pairs` (11) of the 12 selection pairs; the test split has only 10,
    so a probe built from it would hold 10."""
    out, _, _ = run
    with open(out / "retrieval_probes" / "probes.jsonl") as handle:
        probes = [json.loads(line) for line in handle]
    assert {p["num_pairs"] for p in probes} == {float(_PROBE_PAIRS)}


def test_the_summary_scores_the_selection_slice_and_the_test_split_separately(run):
    out, _, _ = run
    text = (out / "training_summary.txt").read_text()
    selection_block = text.split("Retrieval metrics (selection slice")[1].split("Retrieval metrics (test")[0]
    test_block = text.split("Retrieval metrics (test:")[1]
    assert f"num_pairs: {float(_SELECT_PAIRS):.4f}" in selection_block
    assert f"num_pairs: {float(_TEST_PAIRS):.4f}" in test_block
    assert "scored once, after training" in test_block
