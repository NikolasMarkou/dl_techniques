"""``init`` for the tokenizer: a ``beit_tokenizer`` run starts from an earlier ``beit_tokenizer`` run, and no other run seeds it.

plan-2026-11-25T124419-fa94f59c Step 8 (D-012, D-050, D-051). The family row declares the channels, the image size, the patch and the
codebook size ``fixed`` (another value is refused BEFORE the claim, naming both), no head and no source of another task; the model keys the
record does not carry (a width, a depth, the embedding size) are the engine's, refused after the claim, complete-or-raise. Pinned here:

* ``preclaim``: a fake source directory (a record written by the real builder and a placeholder archive) is read before the run directory is
  claimed. Another patch, size or codebook size is refused naming both values (every difference at once); a ``beit_mim`` or a classifier run is
  refused with the declared-sources sentence, and a tokenizer run is no source of ``beit_mim`` (it takes a tokenizer by path); the identity is
  the RESOLVED request; the command line exits 2 and makes no device; a source of another width is taken (the engine refuses it later).
* ``split``: the source's seeded train cut against this run's selection rows, counted and compared with a numpy oracle written here, and the
  oracle against the rows the loader really reads back.
* ``wire`` (real tiny runs on CIFAR-10 rows capped by ``data.max_samples``, CPU, the floor patched to what a toy reaches): the weights ``fit``
  starts from are the source's, the ``init`` block and the summary say so, epoch zero is the loaded weights' ``val_reconstruction_loss`` (an
  independent evaluate of the file), the bar is strictly below ``min(baseline, loaded)`` (a fine-tune that moves no weight misses, one that
  learns meets it), another width is refused after the claim with nothing trained, and a resume of an ``init`` run opens no file of its source.

The runs read the REAL CIFAR-10 rows (a loud skip without the Keras cache); tests write only under ``tmp_path``.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.utils.logger import logger
from train.api import init_from, predictor, registry
from train.api.config import validate
from train.api.errors import ConfigError
from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer, train_beit_tokenizer
from train.api.tasks.pretrain import train_pretrain
from train.common import classification_run, pretrain_run
from train.common import tokenizer_data as td
from train.common.image_streams import stream_statistics

from ._tokenizer_world import (  # noqa: F401  (``world`` is a fixture)
    BATCH, CIFAR_CACHE, LOW_FLOOR, MAX_SAMPLES, MODEL, NO_CIFAR, TINY_MODEL_ARGS, prepared, request, world,
)
from .test_init_from_runs import (
    DRIVER, PLACEHOLDER, STILL, a_source, archive_weights, check_copied, classification_record, cut, pretrain_record, record_of, started,
    summary_of, watch_opens, write_source,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY, TASK = "vq_vae_rotation", "beit_tokenizer"
#: The train pool of CIFAR-10; ``test_split_the_oracles_cut_is_what_the_loader_reads_back`` holds it against the real array.
CIFAR_TRAIN_ROWS = 50000


def tokenizer_record(**changes: Any) -> Dict[str, Any]:
    """The record of a tiny tokenizer run at CIFAR-10's geometry, written by the real builder; ``changes`` replace its arguments."""
    mean, std = stream_statistics("cifar10")
    arguments = dict(model=MODEL, family=FAMILY, variant="small", input_size=(32, 32), channels=3, mean=mean, std=std, patch_size=4,
                     num_embeddings=TINY_MODEL_ARGS["num_embeddings"])
    return predictor.build_tokenizer_record(**{**arguments, **changes})


def beit_mim_request(out: Path) -> Dict[str, Any]:
    return {"out": str(out), "model": "beit_mim/beit_tiny", "data": {"dataset": "cifar10"}, "model_args": {"tokenizer_checkpoint": "t.keras"}}


# ---------------------------------------------------------------------
# preclaim: a fake source directory, read before the run directory is claimed
# ---------------------------------------------------------------------

DECLARED = r"vq_vae_rotation takes weights from vq_vae_rotation \(beit_tokenizer\)$"
#: ``name: (the source record, the task function, its request, the message pattern)``.
REFUSALS = {
    "another_patch": (
        lambda: tokenizer_record(patch_size=8), train_beit_tokenizer, request,
        r"vq_vae_rotation sizes its weights by data\.patch_size; the source trained at 8, this run asks for 4$"),
    "another_size": (
        lambda: tokenizer_record(input_size=(64, 64)), train_beit_tokenizer, request,
        r"vq_vae_rotation sizes its weights by transform_args\.image_size; the source trained at \[64, 64\], this run asks for \[32, 32\]$"),
    "another_codebook": (
        lambda: tokenizer_record(num_embeddings=128), train_beit_tokenizer, request,
        r"vq_vae_rotation sizes its weights by the codebook size \(model_args\.num_embeddings\); the source trained at 128, this run asks "
        r"for 64$"),
    "every_difference_at_once": (
        lambda: tokenizer_record(input_size=(64, 64), patch_size=8, num_embeddings=128), train_beit_tokenizer, request,
        r"vq_vae_rotation sizes its weights by transform_args\.image_size and data\.patch_size and the codebook size "
        r"\(model_args\.num_embeddings\); the source trained at \[64, 64\] and 8 and 128, this run asks for \[32, 32\] and 4 and 64$"),
    "a_beit_mim_run": (
        lambda: pretrain_record("beit_mim/beit_tiny"), train_beit_tokenizer, request,
        r"beit_mim \(pretrain\) cannot seed vq_vae_rotation \(beit_tokenizer\); " + DECLARED),
    "a_classifier": (
        classification_record, train_beit_tokenizer, request,
        r"vit \(image_classification\) cannot seed vq_vae_rotation \(beit_tokenizer\); " + DECLARED),
    "a_tokenizer_into_beit_mim": (
        tokenizer_record, train_pretrain, beit_mim_request,
        r"vq_vae_rotation \(beit_tokenizer\) cannot seed beit_mim \(pretrain\); beit_mim takes weights from beit_mim \(pretrain\)$"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_preclaim_a_source_the_row_cannot_take_is_refused_naming_both_sides_and_nothing_is_claimed(world, tmp_path, case):
    make, function, ask, pattern = REFUSALS[case]
    source = write_source(tmp_path / "source", make(), {})
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + pattern):
        function(**{**ask(out), "init": source})
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


def test_preclaim_the_command_line_exits_2_claims_nothing_and_makes_no_device(tmp_path):
    source = write_source(tmp_path / "source", tokenizer_record(patch_size=8, num_embeddings=512), {})
    out = tmp_path / "run"
    argv = [TASK, f"out={out}", f"model={MODEL}", "data.dataset=cifar10", f"init={source}"]
    done = subprocess.run([sys.executable, "-c", DRIVER, *argv], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})
    seen = done.stdout.strip().splitlines()[-1]
    assert seen == '{"code": 2, "setup_gpu_reached": false, "devices_created": false}', done.stderr[-2000:]
    assert "vq_vae_rotation sizes its weights by data.patch_size; the source trained at 8, this run asks for 4" in done.stderr
    assert not out.exists()


def test_preclaim_a_source_of_the_same_geometry_and_codebook_is_taken_whatever_its_width(world, tmp_path):
    # The record carries no width, so the engine (after the claim) is what refuses another one: `test_wire_another_width...` and the census.
    record = predictor.build_tokenizer_record(
        model=MODEL, family=FAMILY, variant="small", input_size=(32, 32), channels=3, mean=[0.5] * 3, std=[0.25] * 3, patch_size=4,
        num_embeddings=TINY_MODEL_ARGS["num_embeddings"], variant_overrides={"hidden_channels": 32})
    source = write_source(tmp_path / "source", record, {})
    trainer, _, context = prepared(tmp_path / "run", init=str(source))
    taken = trainer._init_source
    assert (taken.path, taken.relation, dict(taken.pins), dict(context.pinned)) == (source, None, {}, {})
    assert taken.sha256 == hashlib.sha256(PLACEHOLDER).hexdigest() and taken.record == record
    assert not (tmp_path / "run").exists()


def test_the_identity_is_the_resolved_size_patch_and_codebook_and_every_fixed_key_is_compared(world, tmp_path):
    fixed = registry.MODELS[FAMILY].transfer.fixed
    cases = {
        "the tiny request": ({}, {"channels": 3, "normalize": "standardize", "input_size": [32, 32], "patch_size": 4,
                                  "num_embeddings": TINY_MODEL_ARGS["num_embeddings"]}),
        # the resolved values, never the dataset's table: a typed size, patch and codebook are what the record will hold
        "a typed geometry": ({"transform_args": {"augment": False, "image_size": 64}, "data": {"dataset": "cifar10", "patch_size": 8},
                              "model_args": {**TINY_MODEL_ARGS, "num_embeddings": 128}},
                             {"channels": 3, "normalize": "standardize", "input_size": [64, 64], "patch_size": 8, "num_embeddings": 128}),
        "the recipe's codebook": ({"model_args": {}}, {"channels": 3, "normalize": "standardize", "input_size": [32, 32], "patch_size": 4,
                                                       "num_embeddings": 512}),
    }
    for name, (changes, expected) in cases.items():
        trainer, resolved, context = prepared(tmp_path / "run", **changes)
        identity = trainer.init_identity(resolved, context)
        assert identity == expected, name
        # A fixed key the record or the identity lacks is skipped by the comparison without a word.
        assert set(fixed) <= set(identity) and set(fixed) <= set(tokenizer_record()), name
    assert not (tmp_path / "run").exists()


def test_the_engines_tables_know_the_task(world):
    spec = registry.MODELS[FAMILY].transfer
    assert (spec.fixed, spec.head, dict(spec.sources), spec.pinned) == (
        ("channels", "input_size", "patch_size", "num_embeddings"), (), {}, ())
    assert set(spec.fixed) <= set(init_from.GEOMETRY_NAMES)
    # No head: `head_differs` indexes the table for every same-task transfer, so the row exists and is empty, and the codebook is the model's.
    record = tokenizer_record()
    assert init_from.HEAD_KEYS[TASK] == () and init_from.head_differs(TASK, record, {**record, "num_embeddings": 999}) is False
    assert "num_embeddings" in init_from.GEOMETRY_KEYS


# ---------------------------------------------------------------------
# split: the source's training rows against this run's selection rows, as an oracle written here
# ---------------------------------------------------------------------

#: ``name: (the source's seed, its data keys that differ, this run's seed)``, on CIFAR-10's 50,000 train rows.
SPLITS = {
    "same_keys": (42, {}, 42),
    "other_seed": (42, {}, 43),
    "capped_source": (42, {"max_samples": 20000}, 42),
    "other_dataset": (42, {"dataset": "cifar100"}, 42),
}


@pytest.mark.parametrize("case", sorted(SPLITS))
def test_split_a_target_counts_the_sources_rows_as_the_oracle(world, tmp_path, case):
    seed, extra, target_seed = SPLITS[case]
    data = {"format": "builtin", "dataset": "cifar10", "validation_split": None, "max_samples": None, **extra}
    trainer = BeitTokenizerTrainer(validate(BeitTokenizerConfig, request(tmp_path / "run", seed=target_seed, data={"dataset": "cifar10"})))
    fake = SimpleNamespace(n_train=45000, n_select=5000, n_test=10000, fingerprint=None)
    split = trainer.init_overlap(a_source({"seed": seed, "data": data}, tokenizer_record()), trainer.config, fake)
    assert predictor._split_problems(split) == []
    flags = init_from.bias_of({"split": split})
    if case == "other_dataset":
        assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("different_data", 0, 0)
        assert not any(flags.values())
        return
    expected = len(cut(CIFAR_TRAIN_ROWS, seed, cap=extra.get("max_samples"))[0] & cut(CIFAR_TRAIN_ROWS, target_seed)[1])
    assert split["n_select_overlap"] == expected and split["n_test_overlap"] == 0
    assert (expected > 0) == (case in ("other_seed", "capped_source")), "the control: the oracle counts what it should"
    assert (split["relation"], flags["selection_biased"]) == (("same_split", False) if case == "same_keys" else ("overlap", True))


@pytest.fixture(scope="module")
def cifar():
    if not CIFAR_CACHE.exists():
        pytest.skip(NO_CIFAR)


def test_split_the_oracles_cut_is_what_the_loader_reads_back(cifar):
    (raw, _), _ = keras.datasets.cifar10.load_data()
    assert raw.shape[0] == CIFAR_TRAIN_ROWS, "the oracle's pool is the real train array"
    mean, std = stream_statistics("cifar10")
    selected = {}
    for seed in (42, 43):
        held = np.random.default_rng(seed).permutation(CIFAR_TRAIN_ROWS)[:MAX_SAMPLES][:MAX_SAMPLES // 10]
        expected = ((raw[held].astype(np.float64) / 255.0 - mean) / std).astype(np.float32)
        rows = td.load_tokenizer_batches(dataset="cifar10", image_size=32, batch_size=BATCH, seed=seed, stream_seed=None,
                                         validation_split=None, max_samples=MAX_SAMPLES, augment=False)
        read = np.concatenate([np.asarray(batch[0]) for batch in rows.select_ds])
        np.testing.assert_allclose(read, expected, atol=1e-5, err_msg=f"seed {seed}")
        selected[seed] = expected
    assert not np.allclose(selected[42], selected[43]), "the control: two seeds select other rows"


# ---------------------------------------------------------------------
# wire: real tiny runs, a source and what it seeds
# ---------------------------------------------------------------------


def tiny_run(out: Path, **changes: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """``(the request, the weights fit started from)`` of one tiny run in a fresh Keras session, the floor patched to what a toy reaches."""
    asked = request(out, **changes)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        return asked, started(BeitTokenizerTrainer, train_beit_tokenizer, **asked)


def read_sidecar(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "checkpoints" / "fit_state.json").read_text())


def loaded_loss(run_dir: Path, seed: int) -> float:
    """``reconstruction_loss`` of ``run_dir``'s best_model.keras on the selection rows of ``seed``: the file's own evaluate, not a run's."""
    rows = td.load_tokenizer_batches(dataset="cifar10", image_size=32, batch_size=BATCH, seed=seed, stream_seed=None, validation_split=None,
                                     max_samples=MAX_SAMPLES, augment=False)
    model = keras.models.load_model(str(Path(run_dir) / "best_model.keras"))
    return float(classification_run.evaluate_dict(model, rows.select_ds)["reconstruction_loss"])


@pytest.fixture(scope="module")
def source(world, cifar, tmp_path_factory):
    """A tiny tokenizer on 1,600 CIFAR-10 rows, seed 42, two epochs."""
    out = tmp_path_factory.mktemp("tokenizer_source") / "run"
    tiny_run(out)
    return out


@pytest.fixture(scope="module")
def pair(world, source, tmp_path_factory):
    """A tokenizer from a copy of ``source`` (the resume test renames the copy away) at a rate that moves no float32 weight: the fine-tune
    scores what the loaded weights scored at epoch zero."""
    root = tmp_path_factory.mktemp("tokenizer_pair")
    copy = root / "source"
    shutil.copytree(source, copy)
    asked, start = tiny_run(root / "run", init=str(copy), optim_args=STILL, epochs=1)
    return SimpleNamespace(source=copy, out=root / "run", request=asked, start=start)


@pytest.fixture(scope="module")
def finetune(world, source, tmp_path_factory):
    """A tokenizer from ``source`` at the tiny recipe's rate and ANOTHER seed: a selection split of other rows."""
    out = tmp_path_factory.mktemp("tokenizer_finetune") / "run"
    asked, start = tiny_run(out, init=str(source), seed=43)
    return SimpleNamespace(out=out, start=start, request=asked)


def test_wire_a_tokenizer_run_seeds_a_tokenizer_run_with_every_weight_and_its_bar_is_strictly_below_the_loaded_loss(source, pair):
    out, weights = pair.out, archive_weights(pair.source)
    record, summary = record_of(out), summary_of(out)
    block, theirs = record["init"], record_of(pair.source)
    assert (block["task"], block["family"], block["relation"], block["pinned"]) == (TASK, FAMILY, "same_task", {})
    assert block["geometry"] == {key: theirs[key] for key in ("input_size", "channels", "patch_size", "num_embeddings")}
    # No head, the same codebook: every weight of the archive is copied, and the model fit starts from has no other.
    assert block["transfer"]["copied"] == len(weights) == len(pair.start) > 0
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, pair.start, weights)
    trained, held = cut(CIFAR_TRAIN_ROWS, 42, cap=MAX_SAMPLES)
    assert not trained & held, "the oracle: the source trained on none of the rows this run selects"
    split = block["split"]
    assert (split["relation"], split["unit"], split["n_select_overlap"], split["n_test_overlap"]) == ("same_split", "rows", 0, 0)
    assert predictor.record_problems(record) == []
    assert (summary["init"], summary["epoch_zero_of"]) == (block, "init") and not any(init_from.bias_of(block).values())
    # Epoch zero is the LOADED weights' val_reconstruction_loss: the file's own evaluate on the same fixed rows, and the source's best.
    loaded = read_sidecar(out)["initial"]["metrics"]["reconstruction_loss"]
    assert loaded == pytest.approx(loaded_loss(pair.source, 42), rel=1e-6)
    assert loaded == pytest.approx(summary_of(source)["success_value"], rel=1e-5)
    baseline = summary["baseline"]["reconstruction_loss"]
    assert summary["epoch_zero"] == {"reconstruction_loss": loaded, "binds": True}, "the loaded weights beat the constant image: they set the bar"
    assert loaded < baseline and summary["success_threshold"] == min(baseline, loaded) == loaded
    logger.info(f"TOKENIZER-INIT-STILL copied {block['transfer']['copied']} of {len(pair.start)} weights, reset 0; epoch zero (the loaded "
                f"weights) {loaded:.9f}, constant image {baseline:.9f}, bar {summary['success_threshold']:.9f}, best of the fine-tune "
                f"{summary['success_value']:.9f}, beats_bar {summary['reconstruction']['beats_bar']}, success {summary['success']}")
    # Strictly below, beyond the tie band: a fine-tune that only holds its loaded weights misses on the reconstruction and on that part alone.
    assert summary["success_value"] == pytest.approx(loaded, rel=1e-6)
    assert summary["reconstruction"]["beats_bar"] is False and summary["codebook_usage"]["meets_floor"] is True
    assert summary["success"] is False and "loaded_success_value" not in summary
    assert (summary["initial_loss_ratio"], summary["init_scale_warning"]) == (None, None)
    assert init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    assert not any(note.startswith("initial reconstruction loss") for note in summary["notes"])
    assert any("the loaded weights' (under `init`)" in note for note in summary["notes"])
    log = (out / "run.log").read_text()
    assert "Evaluate of the loaded weights before fit: " in log and "Untrained evaluate BEFORE fit" not in log


def test_wire_a_fine_tune_that_learns_meets_the_bar_and_the_split_of_other_rows_is_counted_as_the_oracle(source, finetune):
    record, summary = record_of(finetune.out), summary_of(finetune.out)
    loaded = summary["epoch_zero"]["reconstruction_loss"]
    assert loaded == pytest.approx(loaded_loss(source, 43), rel=1e-6), "the loaded weights on the rows of THIS run's seed"
    assert summary["success_threshold"] == min(summary["baseline"]["reconstruction_loss"], loaded)
    assert summary["reconstruction"]["beats_bar"] is True and np.float32(summary["success_value"]) < np.float32(loaded)
    assert summary["codebook_usage"]["meets_floor"] is True and summary["success"] is True
    logger.info(f"TOKENIZER-INIT-LEARNS epoch zero (the loaded weights) {loaded:.6f}, bar {summary['success_threshold']:.6f}, best of the "
                f"fine-tune {summary['success_value']:.6f}")
    trained, selected = cut(CIFAR_TRAIN_ROWS, 42, cap=MAX_SAMPLES)[0], cut(CIFAR_TRAIN_ROWS, 43, cap=MAX_SAMPLES)[1]
    expected = len(trained & selected)
    split = record["init"]["split"]
    assert expected > 0 and (split["relation"], split["unit"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", "rows", expected, 0)
    assert (summary["selection_biased"], summary["test_biased"]) == (True, False) and predictor.record_problems(record) == []
    assert record["init"]["transfer"]["copied"] == len(finetune.start)


def test_wire_another_width_is_taken_before_the_claim_and_refused_after_it_with_nothing_trained(world, cifar, source, tmp_path):
    out = tmp_path / "run"
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_beit_tokenizer(**request(out, init=str(source), model_args={**TINY_MODEL_ARGS, "hidden_channels": 32}))
    text = str(caught.value)
    assert "weights of vq_vae_rotation/small have no source weight of their shape and no declared rule" in text and "nothing was copied" in text
    assert f"The run directory {out} was claimed" in text
    assert (out / "config.json").exists() and not (out / "training_log.csv").exists()
    assert not (out / "best_model.keras").exists() and not (out / "task.json").exists()


def test_resume_of_an_init_run_with_its_source_renamed_away_opens_no_source_file(pair, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    shutil.copytree(pair.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report
    moved = pair.source.with_name("moved_away")
    pair.source.rename(moved)
    try:
        monkeypatch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        opened = watch_opens(monkeypatch, str(pair.source))
        keras.backend.clear_session()
        train_beit_tokenizer(**{**pair.request, "out": str(run_dir), "resume_interrupted": True})
    finally:
        monkeypatch.undo()
        moved.rename(pair.source)
    assert summary_of(run_dir)["reporting_resumed"] is True and opened == []
    assert summary_of(run_dir)["init"] == record_of(run_dir)["init"] == record_of(pair.out)["init"]
