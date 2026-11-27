"""``init`` for the ``dino`` task: a dino run seeds a ``vit`` classifier, and no run seeds a dino run.

plan-2026-11-26T224507-e3381378 Step 8 (D-013, D-029, D-056, D-068 to D-070). The pair is ``("dino", "dino_v1")`` in ``recipes.VIT_TRANSFER.sources``: the
loaded training model holds a student and a teacher at every weight path, so the row names a ``view`` (``student``), drops the projection head and
renames the library's ``transformer_block_N`` to the classifier's ``transformer_layer_N`` (any number of digits). Pinned here:

* ``wire`` (one real 12-block ``dino_v1/tiny`` run on the identity fake of CIFAR-10, 32 px, patch 8, and one real ``vit/vit_tiny`` run seeded by it, CPU):
  every weight ``fit`` starts from is the STUDENT's at the renamed path (blocks 10 and 11 included, and not the teacher's), the counts are the models' own
  (copied is every backbone weight, reset is the classification head, dropped is the projection head: derived here and printed), the ``init`` block is in
  ``task.json`` and the summary with ``relation`` ``cross_task``, the source archive is byte for byte what it was, and a resume of the run opens no
  file of its source;
* ``preclaim`` (a fake source directory, nothing is claimed): a typed patch other than the source's is refused by name, a dino source into another
  classifier family is refused with the declared-sources sentence, and any source into ``dino`` (a dino, a ``beit_mim``, a ``vit``) is refused with the
  one sentence of D-013, on the command line with exit code 2 and no device made.

The runs write only under ``tmp_path``; tests that need no model use no run.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.utils.logger import logger
from train.api import predictor, resume_decisions
from train.api.errors import ConfigError
from train.api.tasks.dino import train_dino
from train.api.tasks.image_classification import ImageClassificationTrainer, train_image_classification

from ._dino_world import CLAIM_LEAVES, IMAGE, PATCH, identity_world, task_request, train_real_run  # noqa: F401  (``identity_world`` is a fixture)
from .test_init_from_runs import (
    DRIVER, QUIET, REPO_ROOT, archive_weights, check_copied, classification_record, dino_record, pretrain_record, record_of, relative, sha, started,
    summary_of, watch_opens, write_source,
)

#: What the D-013 refusal says, in words of its own (not the module's constant): the three reasons.
REFUSAL = (r"Invalid value for 'init': a dino run is no `init` target: its model holds a student and a teacher at the same weight paths, so a copy "
          r"would need a teacher re-sync that this task does not do, and epoch zero under `init` would call `model.evaluate`, which advances the loss's "
          r"centre; a dino run can seed a `vit` classification run of its own width and depth instead \(model=vit/vit_tiny after dino_v1/tiny\)$")
#: The cap both runs read (so their seeded selection rows are the same rows and nothing is biased).
CAP = 1000


def vit_request(out: Any, **changes: Any) -> Dict[str, Any]:
    """A tiny ``vit_tiny`` classification of CIFAR-10 (the identity fake) on the first ``CAP`` rows, the 12 blocks of a DINO ``tiny`` backbone."""
    return {"out": str(out), "model": "vit/vit_tiny", "data": {"dataset": "cifar10", "max_samples": CAP}, **QUIET, **changes}


# ---------------------------------------------------------------------
# preclaim: a fake source directory, read before the run directory is claimed
# ---------------------------------------------------------------------


def test_preclaim_a_dino_source_for_a_vit_is_read_with_its_declared_relation_and_its_patch_pinned(identity_world, tmp_path):
    from .test_init_from_runs import classifier_config, prepared

    source = write_source(tmp_path / "source", dino_record(), {})
    config = classifier_config(tmp_path / "run", init=str(source), model="vit/vit_tiny")
    trainer, _, context = prepared(config, ImageClassificationTrainer)
    taken = trainer._init_source
    assert (taken.relation.view, taken.relation.drop, taken.relation.prefix) == ("student", ("dino_projection_head",), "")
    assert (dict(taken.pins), dict(context.pinned)) == ({"patch_size": PATCH}, {"patch_size": PATCH})  # read from the RECORD, not the config
    assert taken.record == dino_record() and not (tmp_path / "run").exists()


def test_preclaim_a_patch_other_than_the_sources_is_refused_by_name_and_nothing_is_claimed(identity_world, tmp_path):
    source = write_source(tmp_path / "source", dino_record(), {})
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=rf"Invalid value for 'model_args\.patch_size': the source was built with patch_size={PATCH} and init keeps"):
        train_image_classification(**{**vit_request(out), "init": str(source), "model_args": {"patch_size": PATCH // 2}})
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


@pytest.mark.parametrize("model", ["beit/beit_tiny", "energy_transformer/et_tiny", "resnet/resnet18"])
def test_preclaim_a_dino_source_for_a_classifier_that_declares_no_such_pair_is_refused_naming_both_sides(identity_world, tmp_path, model):
    source = write_source(tmp_path / "source", dino_record(), {})
    out = tmp_path / "run"
    family = model.split("/")[0]
    with pytest.raises(ConfigError, match=rf"Invalid value for 'init': dino_v1 \(dino\) cannot seed {family} \(image_classification\); {family} takes weights from "):
        train_image_classification(**{**vit_request(out), "model": model, "init": str(source)})
    assert not out.exists()


SOURCES_OF_OTHER_TASKS = {
    "a_dino_run": dino_record,
    "a_beit_mim_run": lambda: pretrain_record("beit_mim/beit_tiny"),
    "a_vit_classifier": classification_record,
}


@pytest.mark.parametrize("case", sorted(SOURCES_OF_OTHER_TASKS))
def test_preclaim_any_source_into_a_dino_run_is_refused_with_the_one_sentence_and_nothing_is_claimed(identity_world, tmp_path, case):
    source = write_source(tmp_path / "source", SOURCES_OF_OTHER_TASKS[case](), {})
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=REFUSAL):
        train_dino(**{**task_request(out), "init": str(source)})
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


def test_preclaim_a_dino_target_refuses_before_it_reads_the_source_at_all(identity_world, tmp_path):
    # The sentence is the identity hook's, so a path that does not exist gets the same sentence and not "does not exist".
    with pytest.raises(ConfigError, match=REFUSAL):
        train_dino(**{**task_request(tmp_path / "run"), "init": str(tmp_path / "no_such_run")})


def test_preclaim_the_command_line_exits_2_for_a_dino_target_claims_nothing_and_makes_no_device(tmp_path):
    source = write_source(tmp_path / "source", dino_record(), {})
    out = tmp_path / "run"
    argv = ["dino", f"out={out}", "model=dino_v1/tiny", "data.dataset=cifar10", f"init={source}"]
    done = subprocess.run([sys.executable, "-c", DRIVER, *argv], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"code": 2, "setup_gpu_reached": False, "devices_created": False}, done.stderr[-2000:]
    assert "a dino run is no `init` target" in done.stderr and not out.exists()


# ---------------------------------------------------------------------
# wire: one real dino run seeds one real vit run
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def dino_source(identity_world, tmp_path_factory):
    """One finished 1-epoch run of the full ``dino_v1/tiny`` (``max_samples`` 1000: the fake holds 60 rows, so every row)."""
    out = tmp_path_factory.mktemp("dino_source") / "run"
    assert train_real_run(out) == 0
    return out


@pytest.fixture(scope="module")
def student(dino_source):
    """``({relative path: value}, {...})`` of the student and of the teacher in the archive the run saved."""
    model = keras.models.load_model(str(dino_source / "best_model.keras"), compile=False)
    return SimpleNamespace(student=relative(model.student), teacher=relative(model.teacher))


@pytest.fixture(scope="module")
def pair(identity_world, dino_source, student):
    """A ``vit_tiny`` run seeded by the dino run; ``before`` is the source archive's sha256 before it started, ``start`` the weights ``fit`` began from."""
    out = dino_source.parent / "vit_run"
    before = sha(dino_source / "best_model.keras")
    request = vit_request(out, init=str(dino_source))
    keras.backend.clear_session()
    start = started(ImageClassificationTrainer, train_image_classification, **request)
    return SimpleNamespace(source=dino_source, out=out, request=request, start=start, before=before)


def renamed(student: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The student's weights as the classifier names them: the projection head gone, ``transformer_block_N`` read ``transformer_layer_N``.

    Written here with a plain ``str.replace`` (never the row's regex) so the test is an oracle of it."""
    return {path.replace("transformer_block_", "transformer_layer_"): value for path, value in student.items()
            if not path.startswith("dino_projection_head/")}


def test_wire_every_weight_fit_starts_from_is_the_students_at_its_renamed_path_blocks_ten_and_eleven_included(pair, student):
    wanted = renamed(student.student)
    block = record_of(pair.out)["init"]
    check_copied(block, pair.start, wanted)
    layers = {path.split("/")[0] for path in pair.start if path.startswith("transformer_layer_")}
    assert layers == {f"transformer_layer_{index}" for index in range(12)}
    for index in (10, 11):  # two digits: a pattern for one leaves them random and the transfer refused
        path = f"transformer_layer_{index}/attention/cross_attention/qkv/kernel"
        assert path in pair.start
        np.testing.assert_array_equal(pair.start[path], student.student[path.replace("layer", "block")], err_msg=path)


def test_wire_the_student_is_the_source_and_the_teacher_is_not(pair, student):
    # The two were one set of weights at the start of the run and are two after it: the patch kernel is where they must differ.
    path = "patch_embed/projection/kernel"
    assert not np.array_equal(student.student[path], student.teacher[path]), "the control is void: student and teacher are equal"
    np.testing.assert_array_equal(pair.start[path], student.student[path])
    assert not np.array_equal(pair.start[path], student.teacher[path])


def test_wire_the_block_counts_are_the_models_own_and_the_two_heads_are_apart(pair, student):
    block = record_of(pair.out)["init"]["transfer"]
    projection = sorted(path for path in student.student if path.startswith("dino_projection_head/"))
    heads = sorted(path for path in pair.start if path.startswith("classification_head/"))
    logger.info(f"INIT COUNTS dino student into vit_tiny: source {len(student.student)} target {len(pair.start)} copied {block['copied']} "
                f"reset {len(block['reset'])} dropped {len(block['dropped'])}")
    assert projection and heads
    assert sorted(block["dropped"]) == projection and sorted(block["reset"]) == heads
    assert block["copied"] == len(student.student) - len(projection) == len(pair.start) - len(heads)
    assert (block["resampled"], block["grown"]) == ([], [])


def test_wire_the_init_block_is_in_the_record_and_the_summary_and_says_cross_task(pair, dino_source):
    record, summary = record_of(pair.out), summary_of(pair.out)
    block = record["init"]
    assert (block["source"], block["file"], block["sha256"]) == (str(dino_source), "best_model.keras", pair.before)
    assert (block["task"], block["family"], block["variant"], block["relation"]) == ("dino", "dino_v1", "tiny", "cross_task")
    assert block["pinned"] == {"patch_size": PATCH}
    assert block["geometry"] == {"input_size": [IMAGE, IMAGE], "channels": 3, "patch_size": PATCH}
    used = json.loads((pair.out / "config.json").read_text())["used"]["model_args"]
    assert used["patch_size"] == PATCH  # the recipe gives 4 at 32 px; the source's 8 is kept
    assert summary["init"] == block and predictor.record_problems(record) == []
    assert block["split"]["relation"] == "same_split" and (summary["selection_biased"], summary["test_biased"]) == (False, False)


def test_wire_the_loaded_sources_archive_is_byte_for_byte_what_it_was_after_the_transfer(pair):
    assert sha(pair.source / "best_model.keras") == pair.before == record_of(pair.out)["init"]["sha256"]
    assert hashlib.sha256((pair.source / "best_model.keras").read_bytes()).hexdigest() == pair.before


def test_wire_a_resume_of_the_run_opens_no_file_of_its_source(pair, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    shutil.copytree(pair.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report
    moved = pair.source.with_name("dino_moved_away")
    pair.source.rename(moved)
    rebuilt: Dict[str, Any] = {}
    real_restore = resume_decisions.restore_finished_fit

    def restore(model, *arguments):
        rebuilt.update({path: value.shape for path, value in relative(model).items()})
        return real_restore(model, *arguments)

    try:
        opened = watch_opens(monkeypatch, str(pair.source))
        monkeypatch.setattr(resume_decisions, "restore_finished_fit", restore)
        keras.backend.clear_session()
        train_image_classification(**{**pair.request, "out": str(run_dir), "resume_interrupted": True})
    finally:
        monkeypatch.undo()
        moved.rename(pair.source)
    assert summary_of(run_dir)["reporting_resumed"] is True and opened == []
    assert rebuilt == {path: value.shape for path, value in archive_weights(pair.out).items()}  # the patch of the record, no source read
    assert summary_of(run_dir)["init"] == record_of(run_dir)["init"] == record_of(pair.out)["init"]


# ---------------------------------------------------------------------
# a vit of another width than the dino source: refused AFTER the claim (D-105)
# ---------------------------------------------------------------------


def test_a_vit_of_another_width_than_the_dino_source_is_refused_after_the_claim_by_the_weight_check_and_leaves_no_run_to_resume(
        identity_world, dino_source, tmp_path):
    """MEASURED (D-105): the width of the target lives in the library's ``ViT.SCALE_CONFIGS``, not in the registry's variants table (``{"scale": "small"}``), so
    no record key of the source can be compared with it before the claim without a new declared field on every target family; the weight check names the
    first weight that does not fit, the claimed directory holds only what the claim writes, and nothing was copied or trained."""
    out = tmp_path / "vit_small_run"
    before = sha(dino_source / "best_model.keras")
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(**vit_request(out, model="vit/vit_small", init=str(dino_source)))
    message = str(caught.value)
    assert "vit pos_embed (1, 17, 192) cannot become (1, 17, 384)" in message, message
    assert "was claimed before this failed, so 'out' is now used" in message, "the message says the claim happened and what to do"
    assert sorted(path.name for path in out.rglob("*")) == sorted(CLAIM_LEAVES), "only what the claim writes"
    for name in ("task.json", "best_model.keras", "final_model.keras", "training_log.csv", "results_summary.json"):
        assert not (out / name).exists(), f"{name} of a run that never trained"
    assert sha(dino_source / "best_model.keras") == before, "the source archive is byte for byte what it was"
