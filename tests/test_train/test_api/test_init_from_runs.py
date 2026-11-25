"""``init``: a run that starts from an earlier run's weights, through the task functions and the command line.

plan-2026-11-18T072904-0fd54b91 (D-002, D-003, D-004, D-008, D-011, D-021, D-033). The ``preclaim`` tests: the
source run is read before the run directory is claimed (``init_from.read_source``, called by
``TaskTrainer._resolve``), and every refusal there exits 2, names what is wrong and claims nothing; ``checkpoint``,
the key ``init`` replaced, is refused by name on the command line and in YAML and is no Python keyword any more;
the ``model_args.pretrained`` doors (a classifier takes ``false`` only; a depth run under ``init`` reads no official
weights and keeps the warm-start rate and the head's 10x).

A ``preclaim`` source is a FAKE run directory: a ``task.json`` written by the record builders of
:mod:`train.api.predictor` (so it passes the real record check), a ``config.json`` holding the resolved
``used.model_args``, and a placeholder ``best_model.keras``. Nothing before the claim opens the archive, so no
training is needed.

The ``wire`` and ``resume`` tests (D-010, D-014, D-016, D-018) train REAL tiny CPU sources (module-scope fixtures)
and start targets from them: the weights ``fit`` starts from are read by a callback and compared, path by path
relative to the model, with the source archive's; the ``task.json`` ``init`` block is compared with what was
copied, resampled, reset, grown and dropped; a resume reads its pins from its own record and opens no file of the
source. ``keras.backend.clear_session()`` runs before every run, as a fresh process would start.

The ``split`` tests (D-012, D-037): the source's training data against this run's selection and test data, counted and
never refused. Each count is compared with an oracle written here (numpy set intersections of the seeded cuts, the
forecasting segment arithmetic, the scene sets); the task hooks are called on fake data where a run would add nothing,
and two real tiny runs check the record, the summary flags, the success line and that the run finishes.

The ``bar`` tests (D-013): under ``init`` a classification, segmentation or denoising run must also beat the loaded
weights' epoch-zero success value, read from the fit sidecar (also on a resume); a stub sidecar drives the real bars,
and runs at a 1e-30 learning rate (no float32 weight moves) miss the bar their absolute threshold alone would pass.

The ``record`` tests (D-014): the summary of a fine-tuned run holds its ``task.json`` block, the flags and the loaded
weights' value its bar compared; it predicts and exports with its source gone; each record kind takes the block.

The ``pretrain`` tests (plan-2026-11-19T081214-5c6cc105 Step 6, U4, D-008, D-012, D-035): an ``et_mim`` or ``beit_mim``
run seeds its own family and its classifier (``energy_transformer``, ``beit``, whose patch the source's record pins), on
the identity fake of CIFAR-10 (row ``i`` has the mean colour ``4i``, so the rows a stream holds read back); any other
pair, another dataset geometry and a grid change no position rule makes are refused before the claim (D-046, D-055),
and a pin that moves the patch off the recipe's is a notice in run.log (D-053); a pretrain run recorded before the rename
is refused on resume by name (D-052); the decoder is reset when the tokenizer or the input statistics
differ (D-048); the pretrain bar under ``init`` stays strictly below the loaded weights' ``val_loss`` and the
higher-is-better rule never judges it.

The ``language`` tests (plan-2026-11-24T092348-7806a290 Step 6, D-005, D-040, D-041): a ``gpt2_lm`` run seeds a ``gpt2_lm`` run of
its own tokenizer (the vocabulary and the tokenizer's digest are ``fixed``: another one is refused before the claim, naming both
values; the sequence length is not, no weight depends on it) and nothing else; its split counts DOCUMENTS against an oracle
written here; its bar under ``init`` is strictly below ``min(unigram, loaded val_loss)``, missed by a fine-tune that moves no
weight and met by one that learns.
"""

import builtins
import contextlib
import dataclasses
import functools
import hashlib
import inspect
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional

import keras
import numpy as np
import pytest
from PIL import Image

from dl_techniques.utils.logger import logger
from train.api import cli, data_args, init_from, predictor, recipes, registry, resume_decisions
from train.api.args import AutoContext
from train.api.config import validate
from train.api.depth_args import PRETRAINED_LR
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks.depth_estimation import DepthEstimationConfig, DepthEstimationTrainer, train_depth_estimation
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)
from train.api.tasks.image_denoising import ImageDenoisingConfig, ImageDenoisingTrainer, train_image_denoising
from train.api.tasks.image_segmentation import ImageSegmentationTrainer, train_image_segmentation
from train.api.tasks.language_model_pretraining import (
    LanguageModelPretrainingConfig,
    LanguageModelPretrainingTrainer,
    train_language_model_pretraining,
)
from train.api.tasks.pretrain import TOKENIZER_KEY, PretrainConfig, PretrainTrainer, train_pretrain
from train.api.tasks.time_series_forecasting import (
    TimeSeriesForecastingConfig,
    TimeSeriesForecastingTrainer,
    train_time_series_forecasting,
)
from train.api.text_tokenizer import tokenizer_spec
from train.api.trainer import TaskTrainer

from ._forecast_fixtures import world  # noqa: F401  (a module-scope fixture: the one registry world)
from ._text_world import request as language_request

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ENV = "DL_TECHNIQUES_DINOV2_WEIGHTS_DIR"
#: What a placeholder archive holds: nothing before the claim reads it, only its hash.
PLACEHOLDER = b"not a keras archive: nothing before the claim opens it"

#: The classification target every function case asks for (built-in cifar10: no data is loaded before the claim).
CLASSIFIER = {"model": "vit/vit_pico", "data": {"dataset": "cifar10"}, "epochs": 1, "batch_size": 8}
#: What a vit_pico source resolved at 32 px on cifar10 records under ``used.model_args``.
VIT_USED = {"patch_size": 4, "kernel_regularizer": None}


def classification_record(model: str = "vit/vit_pico", channels: int = 3, normalize: str = "standardize",
                          side: int = 32) -> Dict[str, Any]:
    family, variant = model.split("/")
    unit = normalize == "unit"
    return predictor.build_task_record(
        task="image_classification", model=model, family=family, variant=variant,
        class_names=[f"c{index}" for index in range(10)], input_shape=[side, side, channels], normalize=normalize,
        mean=[0.0 if unit else 0.5] * channels, std=[1.0 if unit else 0.25] * channels, data_kind="cifar10",
        source_channels=channels)


def segmentation_record(model: str = "convunext/tiny") -> Dict[str, Any]:
    family, variant = model.split("/")
    return predictor.build_segmentation_record(
        model=model, family=family, variant=variant, class_names=["a", "b", "c"], input_size=[64, 64], channels=3,
        normalize="unit", mean=[0.0] * 3, std=[1.0] * 3)


def forecast_record(model: str = "nbeats/tiny", lookback: int = 168, horizon: int = 24) -> Dict[str, Any]:
    family, variant = model.split("/")
    return predictor.build_forecast_record(
        model=model, family=family, variant=variant, lookback=lookback, horizon=horizon, channels=1, value_cols=None,
        time_col=None, scale_floor=1e-6, decode=family, quantile_levels=None, point_index=None,
        data_format="builtin", data_sha256=None)


def depth_record(side: int = 28) -> Dict[str, Any]:
    return predictor.build_depth_record(
        model="depth_anything/vit_s", family="depth_anything", variant="vit_s", input_size=[side, side], channels=3)


def pretrain_record(model: str = "et_mim/et_tiny", side: int = 32, patch: int = 4) -> Dict[str, Any]:
    """A pretrain record at CIFAR-10's pretrain geometry; a ``beit_mim`` one names a tokenizer by a sha256."""
    family, variant = model.split("/")
    return predictor.build_pretrain_record(
        model=model, family=family, variant=variant, input_size=[side, side], channels=3, mean=[0.5] * 3,
        std=[0.25] * 3, patch_size=patch, tokenizer=None if family == "et_mim" else "a" * 64)


def language_record(tokenizer: str = "ascii95", seq_len: int = 32) -> Dict[str, Any]:
    """A language-model record of the named tokenizer (a ``gpt2`` one holds its BPE file hashes)."""
    return predictor.build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=seq_len,
        tokenizer=tokenizer_spec(tokenizer).record())


def text_classification_record(tokenizer: str = "ascii95", seq_len: int = 32) -> Dict[str, Any]:
    """A text-classification record of the named tokenizer."""
    return predictor.build_text_classification_record(
        model="gpt2/tiny", family="gpt2", variant="tiny", seq_len=seq_len, tokenizer=tokenizer_spec(tokenizer).record(),
        class_names=["neg", "pos"])


def write_source(root: Path, record: Optional[Dict[str, Any]], used: Optional[Dict[str, Any]] = None, *,
                 config: bool = True, best: bool = True) -> Path:
    """A fake run directory: ``task.json`` (``record``; ``None`` writes none), ``config.json`` and a placeholder archive."""
    root.mkdir(parents=True)
    if record is not None:
        (root / "task.json").write_text(json.dumps(record))
    if config:
        model = record["model"] if record is not None else "vit/vit_pico"
        (root / "config.json").write_text(json.dumps({"model": model, "used": {"model_args": dict(used or {})}}))
    if best:
        (root / "best_model.keras").write_bytes(PLACEHOLDER)
    return root


def old_record() -> Dict[str, Any]:
    return {**classification_record(), "schema_version": 1}


def as_file(root: Path) -> Path:
    write_source(root, classification_record(), VIT_USED)
    return root / "best_model.keras"


#: ``name: (make the source under a tmp directory, the extra request, the message pattern)``: one row per pre-claim
#: refusal of a classification target (vit_pico on cifar10 at 32 px, standardized).
CLASSIFIER_CASES = {
    "missing": (lambda root: root, {}, r"Invalid value for 'init': .*nowhere does not exist; give the directory of an "
                                       r"earlier python -m train run"),
    "a_file": (as_file, {}, r"Invalid value for 'init': .*best_model\.keras is a file; give the run directory that "
                            r"holds it \(init reads task\.json, config\.json and best_model\.keras\)"),
    "no_record": (lambda root: write_source(root, None, VIT_USED), {},
                  r"task\.json does not exist; a source is a python -m train run, which writes it before its first "
                  r"epoch \(legacy trainer runs and runs made before their task wrote one have none\)"),
    "no_config": (lambda root: write_source(root, classification_record(), config=False), {},
                  r"config\.json does not exist; a source is a python -m train run directory"),
    "no_best_model": (lambda root: write_source(root, classification_record(), VIT_USED, best=False), {},
                      r"holds no best_model\.keras \(the run ended before its first epoch\)"),
    "old_record": (lambda root: write_source(root, old_record(), VIT_USED), {}, r"'init': the source's record cannot be used: .*'schema_version' is 1, an older layout"),
    "undeclared_pair": (lambda root: write_source(root, segmentation_record()), {},
                        r"convunext \(image_segmentation\) cannot seed vit \(image_classification\); vit takes weights "
                        r"from vit \(image_classification\)"),
    "channels": (lambda root: write_source(root, classification_record(channels=1), VIT_USED), {},
                 r"vit sizes its weights by the input channels; the source trained at 1, this run asks for 3"),
    "input_scaling": (lambda root: write_source(root, classification_record(normalize="unit"), VIT_USED), {},
                      r"the source takes its input in \[0, 1\] and this run standardizes its input "
                      r"\(transform_args\.normalize\)"),
    "no_pin": (lambda root: write_source(root, classification_record(), {}), {},
               r"config\.json records no used\.model_args\.patch_size, which init keeps from the source"),
    "pin_changed": (lambda root: write_source(root, classification_record(), VIT_USED),
                    {"model_args": {"patch_size": 8}},
                    r"Invalid value for 'model_args\.patch_size': the source was built with patch_size=4 and init "
                    r"keeps the source's architecture"),
}


@pytest.mark.parametrize("case", sorted(CLASSIFIER_CASES))
def test_preclaim_a_classifier_source_is_refused_by_name_and_nothing_is_claimed(world, tmp_path, case):
    make, extra, pattern = CLASSIFIER_CASES[case]
    source = make(tmp_path / "nowhere")
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=pattern):
        train_image_classification(out=out, init=source, **{**CLASSIFIER, **extra})
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


def prepared(config: Any, trainer_class: Any) -> Any:
    """Resolve ``config`` as ``TaskTrainer.run`` does, up to the claim; return ``(trainer, resolved, context)``."""
    trainer = trainer_class(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


def classifier_config(out: Path, **changes: Any) -> ImageClassificationConfig:
    return validate(ImageClassificationConfig, {"out": str(out), **CLASSIFIER, **changes})


def test_preclaim_a_well_formed_source_passes_and_its_pins_and_hash_are_taken(world, tmp_path):
    source = write_source(tmp_path / "source", classification_record(), VIT_USED)
    for asked in ({}, {"model_args": {"patch_size": 4}}):  # the source's own value, typed, is accepted
        trainer, resolved, _ = prepared(classifier_config(tmp_path / "run", init=str(source), **asked),
                                        ImageClassificationTrainer)
        taken = trainer._init_source
        assert taken.path == source and taken.pins == {"patch_size": 4} and taken.relation is None
        assert taken.sha256 == hashlib.sha256(PLACEHOLDER).hexdigest()
        assert taken.record == classification_record()
    assert not (tmp_path / "run").exists()


def test_preclaim_a_relative_init_resolves_at_the_repo_root_as_out_does(world, tmp_path):
    relative = "results/p4g_test_no_such_source_run"
    assert not (REPO_ROOT / relative).exists()
    with pytest.raises(ConfigError, match=f"'init': {REPO_ROOT / relative} does not exist"):
        prepared(classifier_config(tmp_path / "run", init=relative), ImageClassificationTrainer)


def test_preclaim_a_cross_task_source_is_read_with_its_declared_relation(world, tmp_path):
    from train.api.tasks.image_segmentation import ImageSegmentationConfig, ImageSegmentationTrainer
    from train.api.segmentation_models import CONVNEXT_SEG_TRANSFER

    # A segmenter's input is never standardized, so its classifier source left its input in [0, 1] too.
    source = write_source(tmp_path / "source", classification_record("convnext_v1/cifar10", normalize="unit"),
                          {"strides": 2})
    images, masks = segmentation_folders(tmp_path / "data")
    config = validate(ImageSegmentationConfig, {
        "out": str(tmp_path / "run"), "model": "convnext_seg/cifar10", "init": str(source),
        "data": {"format": "folder", "train_images": str(images), "train_masks": str(masks), "num_classes": 3},
        "transform_args": {"image_size": [64, 64]}})
    trainer, _, _ = prepared(config, ImageSegmentationTrainer)
    assert trainer._init_source.relation == CONVNEXT_SEG_TRANSFER.sources[("image_classification", "convnext_v1")]
    assert trainer._init_source.pins == {}


def segmentation_folders(root: Path):
    from ._segmentation_fixtures import make_folder

    return make_folder(root, count=4, size=(64, 64), classes=3)


def test_preclaim_a_lookback_the_family_sizes_its_weights_by_is_refused(world, tmp_path):
    source = write_source(tmp_path / "source", forecast_record("nbeats/tiny", lookback=168))
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=(
            r"nbeats sizes its weights by data\.lookback; the source trained at 168, this run asks for 336 "
            r"\(families of time_series_forecasting that keep every weight across a data\.lookback change: ets, "
            r"timestral, tirex, xlstm\)")):
        train_time_series_forecasting(
            out=out, model="nbeats/tiny", init=source, batch_size=8,
            data={"lookback": 336, "horizon": 24, "realizations": 1, "series_length": 5000,
                  "patterns": ["ar_process"], "n_select": 16, "n_test": 16})
    assert not out.exists()


#: The pretrain target of the pre-claim cases (nothing is loaded before the claim; the tokenizer is checked after the
#: source, so a beit_mim refusal needs no file).
PRETRAIN_TARGET = {"model": "et_mim/et_tiny", "data": {"dataset": "cifar10"}, "batch_size": 4,
                   "model_args": {"num_steps": 2}}
BEIT_MIM_TARGET = {**PRETRAIN_TARGET, "model": "beit_mim/beit_tiny", "model_args": {TOKENIZER_KEY: "t.keras"}}
BEIT_TARGET = {**CLASSIFIER, "model": "beit/beit_tiny"}

#: ``name: (the source record, the task function, the request, the message pattern)``: a pair no row declares, another
#: scaling, another dataset geometry (U4, D-008) and a BEiT grid past its rule's reach (D-046), each refused before the claim.
PRETRAIN_REFUSALS = {
    "classifier_into_pretrain": (lambda: classification_record("beit/beit_tiny"), train_pretrain, BEIT_MIM_TARGET,
                                 r"beit \(image_classification\) cannot seed beit_mim \(pretrain\); beit_mim takes "
                                 r"weights from beit_mim \(pretrain\)$"),
    "other_family_into_pretrain": (pretrain_record, train_pretrain, BEIT_MIM_TARGET,
                                   r"et_mim \(pretrain\) cannot seed beit_mim \(pretrain\); beit_mim takes weights "
                                   r"from beit_mim \(pretrain\)$"),
    "other_family_into_classifier": (pretrain_record, train_image_classification, BEIT_TARGET,
                                     r"et_mim \(pretrain\) cannot seed beit \(image_classification\); beit takes "
                                     r"weights from beit \(image_classification\) or beit_mim \(pretrain\)$"),
    "unit_classifier": (lambda: pretrain_record("beit_mim/beit_tiny"), train_image_classification,
                        {**BEIT_TARGET, "transform_args": {"normalize": "unit"}},
                        r"the source standardizes its input and this run takes its input in \[0, 1\] "
                        r"\(transform_args\.normalize\)$"),
    "other_dataset": (pretrain_record, train_pretrain, {**PRETRAIN_TARGET, "data": {"dataset": "imagenette"}},
                      r"et_mim sizes its weights by transform_args\.image_size and data\.patch_size; the source "
                      r"trained at \[32, 32\] and 4, this run asks for \[224, 224\] and 16$"),
    "other_patch": (pretrain_record, train_pretrain,
                    {**PRETRAIN_TARGET, "data": {"dataset": "cifar10", "patch_size": 8}},
                    r"et_mim sizes its weights by data\.patch_size; the source trained at 4, this run asks for 8$"),
    # D-046 (review R5): a CIFAR trunk pins patch 4 into a 224 px classifier, a grid past the reach of the BEiT rule.
    "beit_beyond_reach": (lambda: pretrain_record("beit_mim/beit_tiny"), train_image_classification,
                          {**BEIT_TARGET, "data": {"dataset": "imagenette"}}, re.escape(
                              "the source's grid of 8x8 patches (32 px at patch_size 4, which init keeps) would become "
                              "56x56 at 224 px, 3136 tokens; unilm's rule grows the relative position bias table of a grid "
                              "of 8x8 patches to at most 33x33 (past it the ratio of the rule's geometric grid stops at its "
                              "upper bound 1.5 and the source grid no longer spans the target, so every outer target "
                              "displacement takes the value at the source's edge; unilm's published fine-tunes grow 14x14 "
                              "patches to 24x24, far inside the reach of 14 (388), and no reference measures a fine-tune "
                              "past it); ask for transform_args.image_size of at most 132 px") + "$"),
}


@pytest.mark.parametrize("case", sorted(PRETRAIN_REFUSALS))
def test_preclaim_pretrain_a_pair_no_row_declares_or_another_geometry_is_refused(world, tmp_path, monkeypatch, case):
    make, function, request, pattern = PRETRAIN_REFUSALS[case]
    # The Imagenette records are looked for when the data is described, before the source is read: no file is needed.
    monkeypatch.setattr(data_args, "_check_imagenette_records", lambda: None)
    source = write_source(tmp_path / "source", make(), {})
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + pattern):
        function(out=out, init=source, **request)
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


def grid_words(theirs: int, size: int, patch: int, ours: int, side: int) -> str:
    return (f"Invalid value for 'init': the source's grid of {theirs}x{theirs} patches ({size} px at patch_size {patch}, "
            f"which init keeps) would become {ours}x{ours} at {side} px, {ours * ours} tokens; ")


#: ``name: (the source record, its used.model_args, the request, the message)``: a grid change no position rule makes is
#: refused before the claim, and a typed patch meets the pin's refusal first (D-055, review S5).
BEIT_224 = (lambda: classification_record("beit/beit_tiny", side=224), {"patch_size": 16})  # a 224/16 beit source
NETTE = {"data": {"dataset": "imagenette"}}
AT_64 = {**NETTE, "transform_args": {"image_size": 64}}
TABLE_WORDS = "BEiT's relative position bias table: "
GRID_REFUSALS = {
    "et_grid": (pretrain_record, {}, {**BEIT_TARGET, **NETTE, "model": "energy_transformer/et_tiny"}, grid_words(
        8, 32, 4, 56, 224) + "the Energy Transformer position table cannot change its grid: no reference rule resamples it "
        "(the Energy Transformer paper describes no fine-tuning at another resolution); train this run at the source's grid "
        "(its image_size and patch_size)"),
    "beit_below_3": (*BEIT_224, BEIT_TARGET, grid_words(14, 224, 16, 2, 32) + TABLE_WORDS + "the rule resamples no grid "
                     "below 3x3 patches"),
    "beit_source_below_3": (lambda: classification_record("beit/beit_tiny"), {"patch_size": 16}, {**BEIT_TARGET, **AT_64},
                            grid_words(2, 32, 16, 4, 64) + TABLE_WORDS + "the rule resamples no grid below 3x3 patches"),
    "beit_smaller": (*BEIT_224, {**BEIT_TARGET, **AT_64}, grid_words(14, 224, 16, 4, 64) + TABLE_WORDS + "only a larger "
                     "grid is resampled (unilm's rule grows a grid; for a smaller one its search for the ratio stops at its "
                     "lower bound)"),
    "typed_patch": (lambda: pretrain_record("beit_mim/beit_tiny"), {}, {**BEIT_TARGET, **NETTE, "model_args": {
        "patch_size": 16}}, "Invalid value for 'model_args.patch_size': the source was built with patch_size=4 and init "
        "keeps the source's architecture"),
}


@pytest.mark.parametrize("case", sorted(GRID_REFUSALS))
def test_preclaim_pretrain_a_grid_no_position_rule_makes_is_refused_and_a_typed_patch_meets_the_pin_first(
        world, tmp_path, monkeypatch, case):
    make, used, request, message = GRID_REFUSALS[case]
    monkeypatch.setattr(data_args, "_check_imagenette_records", lambda: None)
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as refused:
        train_image_classification(out=out, init=write_source(tmp_path / "source", make(), used), **request)
    assert (str(refused.value), out.exists()) == (message, False)


def test_preclaim_pretrain_a_source_of_the_same_family_is_taken_with_no_pin(world, tmp_path):
    source = write_source(tmp_path / "source", pretrain_record(), {"num_steps": 2})
    config = validate(PretrainConfig, {"out": str(tmp_path / "run"), "init": str(source), **PRETRAIN_TARGET})
    trainer, _, context = prepared(config, PretrainTrainer)
    taken = trainer._init_source
    assert (taken.relation, taken.pins, context.pinned, taken.record) == (None, {}, {}, pretrain_record())
    assert not (tmp_path / "run").exists()


#: The one notice when a cross-task pin moves the patch off the recipe's (D-046), replayed into run.log (D-053).
PIN_LINE = ("init: the source's record pins patch_size 4 (the recipe gives 16 at 64 px), so this run cuts its 64 px input "
            "into 16x16 = 256 tokens")


def test_preclaim_pretrain_a_classifier_takes_its_patch_from_the_source_record_and_a_typed_one_must_repeat_it(
        world, trees, tmp_path, caplog):
    """D-035: a 64 px beit from a 32/4 beit_mim keeps patch 4 (grid 16; its patch rule gives 16, a 4x4 grid). The pin is
    read from the source's record: its ``config.json`` records no ``used.model_args`` here. The pin line is one trainer
    notice, which run.log replays after the claim, and nothing logs it before the claim (D-053)."""
    caplog.set_level(logging.INFO, logger="dl")
    source = write_source(tmp_path / "source", pretrain_record("beit_mim/beit_tiny"), {})

    def pin_lines(trainer):
        return ([record.getMessage() for record in caplog.records if "pins patch_size" in record.getMessage()],
                [text for _level, text in trainer._notices if "pins patch_size" in text])

    def resolved(**changes: Any) -> Any:
        config = validate(ImageClassificationConfig, classify(str(tmp_path / "run"), trees["ten"], 64,
                                                              model="beit/beit_tiny", **changes))
        trainer, final, context = prepared(config, ImageClassificationTrainer)
        return trainer, context, trainer._model_kwargs(final, context)["patch_size"]

    for typed in ({}, {"model_args": {"patch_size": 4}}):
        caplog.clear()
        trainer, context, patch = resolved(init=str(source), **typed)
        assert trainer._init_source.pins == context.pinned == {"patch_size": 4} and patch == 4
        assert trainer._init_source.relation == registry.MODELS["beit"].transfer.sources[("pretrain", "beit_mim")]
        assert pin_lines(trainer) == ([], [PIN_LINE])
    caplog.clear()
    trainer, context, patch = resolved()
    assert (context.pinned, patch, pin_lines(trainer)) == ({}, 16, ([], []))
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.patch_size': the source was built with "
                                          r"patch_size=4 and init keeps the source's architecture"):
        resolved(init=str(source), model_args={"patch_size": 16})
    assert not (tmp_path / "run").exists()


def test_preclaim_pretrain_the_decoder_head_is_keyed_by_the_tokenizers_sha256():
    """D-008: the same tokenizer bytes at any path keep the decoder; another file resets it; et_mim has none."""
    record = pretrain_record("beit_mim/beit_tiny")
    assert init_from.head_differs("pretrain", record, dict(record)) is False
    assert init_from.head_differs("pretrain", record, {**record, "tokenizer": "b" * 64}) is True
    assert init_from.head_differs("pretrain", pretrain_record(), pretrain_record()) is False


@pytest.mark.parametrize("model", ["et_mim/et_tiny", "beit_mim/beit_tiny"])
def test_preclaim_pretrain_the_decoder_head_is_reset_when_the_input_statistics_differ(model):
    """D-048 (review R7): an imagenette run overridden to CIFAR's 32 px and patch 4 passes the identity check against a
    cifar10 source, but its input is standardized with other statistics, which ET's pixel decoder emits."""
    record = pretrain_record(model)
    assert init_from.HEAD_KEYS["pretrain"] == ("tokenizer", "mean", "std")
    for key in ("mean", "std"):
        assert init_from.head_differs("pretrain", record, {**record, key: [0.45, 0.44, 0.41]}) is True, key
    assert init_from.head_differs("pretrain", record, {**record, "mean": list(record["mean"])}) is False


def test_preclaim_a_row_with_no_transfer_is_refused_by_name(world, tmp_path, monkeypatch):
    spec = registry.MODELS["vit"]
    monkeypatch.setitem(registry.MODELS, "vit", dataclasses.replace(spec, transfer=None))
    with pytest.raises(ConfigError, match=r"'init': the family 'vit' declares no transfer, so it takes no earlier "
                                          r"run's weights$"):
        prepared(classifier_config(tmp_path / "run", init=str(tmp_path / "src")), ImageClassificationTrainer)


def test_preclaim_the_default_init_identity_refuses_naming_the_task():
    with pytest.raises(ConfigError, match=r"Invalid value for 'init': the task 'some_task' does not start from "
                                          r"earlier weights"):
        TaskTrainer.init_identity(SimpleNamespace(task="some_task"), None, None)


def test_preclaim_the_classifier_channels_are_the_loaders(world, tmp_path):
    """The identity's channel count of a built-in dataset is the dataset table's, unless the recipe's loader repeats."""
    from dl_techniques.datasets.vision.classification import get_spec

    from train.api.data_args import DATASETS

    for dataset in DATASETS:
        for model, loader_channels in (("vit/vit_pico", None), ("convnext_v1/cifar10", 3)):
            trainer = ImageClassificationTrainer(classifier_config(tmp_path / "run", model=model))
            spec = registry.MODELS[model.split("/")[0]]
            context = SimpleNamespace(dataset=dataset, defaults=spec.recipe(dataset, model.split("/")[1]))
            expected = context.defaults.get("loader", {}).get("channels") or get_spec(dataset).input_shape[-1]
            identity = trainer.init_identity(trainer.config, context)
            assert identity["channels"] == expected, (dataset, model)
            assert identity["normalize"] == "standardize"


# ---------------------------------------------------------------------
# The command line, in a child: exit 2, nothing claimed, no device made; the reader itself loads no TensorFlow
# ---------------------------------------------------------------------

#: ``argv``: the command; prints what the child saw as one JSON line (the recorder of the builtin-sources test).
DRIVER = textwrap.dedent(
    """
    import json
    import sys

    from tensorflow.python.eager import context as eager

    import train.common.gpu as gpu
    from train.api import cli

    reached = []


    class Stop(Exception):
        pass


    def recorder(*args, **kwargs):
        reached.append("setup_gpu")
        raise Stop


    gpu.setup_gpu = recorder
    try:
        code = cli.main(sys.argv[1:])
    except Stop:
        code = "stopped at setup_gpu"
    print(json.dumps({"code": code, "setup_gpu_reached": bool(reached),
                      "devices_created": eager.context()._context_handle is not None}))
    """
)


@pytest.mark.parametrize("case", ["missing", "no_record", "undeclared_pair", "pin_changed"])
def test_preclaim_the_command_line_exits_2_and_claims_nothing(tmp_path, case):
    make, extra, pattern = CLASSIFIER_CASES[case]
    source = make(tmp_path / "nowhere")
    out = tmp_path / "run"
    argv = ["image_classification", f"out={out}", "model=vit/vit_pico", "data.dataset=cifar10", f"init={source}"]
    argv += [f"model_args.{key}={value}" for key, value in extra.get("model_args", {}).items()]
    done = subprocess.run([sys.executable, "-c", DRIVER, *argv], capture_output=True, text=True, timeout=300)
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen == {"code": 2, "setup_gpu_reached": False, "devices_created": False}, done.stderr[-2000:]
    assert re.search(pattern, done.stderr), done.stderr[-2000:]
    assert not out.exists()


#: Reads a passing source and a refused one through ``read_source`` alone, then reports the heavy modules loaded.
UNIT_DRIVER = textwrap.dedent(
    """
    import json
    import sys
    from types import SimpleNamespace

    from train.api import init_from, recipes, registry
    from train.api.errors import ConfigError

    recipes.register_builtin_models()
    trainer = SimpleNamespace(task="image_classification",
                              init_identity=lambda config, context: {"channels": 3, "normalize": "standardize"})
    config = SimpleNamespace(model_args={})
    context = SimpleNamespace(family="vit")
    taken = init_from.read_source(sys.argv[1], trainer, config, context)
    try:
        init_from.read_source(sys.argv[2], trainer, config, context)
        refused = False
    except ConfigError:
        refused = True
    print(json.dumps({"pins": taken.pins, "refused": refused,
                      "loaded": sorted(name for name in ("tensorflow", "keras", "numpy") if name in sys.modules)}))
    """
)


def test_preclaim_the_source_reader_loads_no_tensorflow_keras_or_numpy(tmp_path):
    source = write_source(tmp_path / "source", classification_record(), VIT_USED)
    done = subprocess.run([sys.executable, "-c", UNIT_DRIVER, str(source), str(tmp_path / "nowhere")],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"pins": {"patch_size": 4}, "refused": True,
                                                                 "loaded": []}


# ---------------------------------------------------------------------
# checkpoint: removed, refused by name; no Python keyword
# ---------------------------------------------------------------------

REMOVED = r"Unknown key 'checkpoint': it was replaced by 'init', which takes the directory of an earlier run"


def test_preclaim_checkpoint_on_the_command_line_is_refused_naming_init(world, tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["image_classification", f"out={out}", "model=vit/vit_pico", "data.dataset=cifar10",
                     "checkpoint=weights.keras"]) == 2
    err = capsys.readouterr().err
    assert re.search(REMOVED, err) and "checkpoint_args" not in err
    assert not out.exists()


def test_preclaim_checkpoint_in_a_config_file_is_refused_naming_init(world, tmp_path, capsys):
    out = tmp_path / "run"
    request = tmp_path / "request.yaml"
    request.write_text(f"out: {out}\nmodel: vit/vit_pico\ndata:\n  dataset: cifar10\ncheckpoint: weights.keras\n")
    assert cli.main(["image_classification", f"config={request}"]) == 2
    err = capsys.readouterr().err
    assert re.search(REMOVED, err) and "checkpoint_args" not in err
    assert not out.exists()


def test_preclaim_checkpoint_in_a_mapping_is_refused_naming_init(tmp_path):
    with pytest.raises(ConfigValidationError, match=REMOVED) as caught:
        validate(ImageClassificationConfig, {"out": str(tmp_path / "run"), **CLASSIFIER, "checkpoint": "w.keras"})
    assert "checkpoint_args" not in str(caught.value)


TASK_FUNCTIONS = (train_image_classification, train_image_segmentation, train_image_denoising,
                  train_time_series_forecasting, train_depth_estimation, train_pretrain)


@pytest.mark.parametrize("function", TASK_FUNCTIONS, ids=[f.__name__ for f in TASK_FUNCTIONS])
def test_preclaim_checkpoint_is_no_keyword_of_a_task_function(function, tmp_path):
    parameters = inspect.signature(function).parameters
    assert "init" in parameters and "checkpoint" not in parameters
    with pytest.raises(TypeError, match="checkpoint"):
        function(out=tmp_path / "run", model="vit/vit_pico", data={}, checkpoint="w.keras")
    assert not (tmp_path / "run").exists()


# ---------------------------------------------------------------------
# model_args.pretrained
# ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, "weights/vit.keras"])
def test_preclaim_a_classifier_takes_pretrained_false_only(world, tmp_path, value):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.pretrained': .*init takes a run directory"):
        train_image_classification(out=out, model_args={"pretrained": value}, **CLASSIFIER)
    assert not out.exists()


def depth_config(out: Path, root: Path, **changes: Any) -> DepthEstimationConfig:
    return validate(DepthEstimationConfig, {
        "out": str(out), "model": "depth_anything/vit_s", "batch_size": 4, "epochs": 1,
        "data": {"root": str(root), "train_split": 0.6, "n_select": 4, "n_test": 4, "layout_samples": 4,
                 "steps_per_epoch": 1, "workers": 0},
        "transform_args": {"image_size": 28}, **changes})


@pytest.fixture(scope="module")
def megadepth_tree(tmp_path_factory):
    from tests.test_train.test_common_depth_data import write_megadepth_tree

    return write_megadepth_tree(tmp_path_factory.mktemp("megadepth"))


def test_preclaim_depth_under_init_reads_no_official_weights_and_keeps_the_warm_start(
        world, megadepth_tree, tmp_path, monkeypatch):
    """The source recorded ``pretrained: true``; the target under ``init`` asks the factory for no official weights
    (the weights directory is empty, so a read would be refused), and keeps 5e-6 with the head at 10x (D-004)."""
    from train.api import depth_models

    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "no_weights"))
    source = write_source(tmp_path / "source", depth_record(), {"pretrained": True})
    trainer, resolved, context = prepared(depth_config(tmp_path / "run", megadepth_tree, init=str(source)),
                                          DepthEstimationTrainer)
    kwargs = trainer._model_kwargs(resolved, context)
    assert kwargs["pretrained"] is False
    assert resolved.optim_args.lr == PRETRAINED_LR
    model = depth_models.create_depth_anything(variant="vit_s", input_shape=(28, 28, 3), **kwargs)
    trainer._built_kwargs = kwargs
    table = trainer.optimizer_options(resolved, model)["lr_multipliers"]
    assert table and set(table.values()) == {10.0}
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("value", [True, False])
def test_preclaim_depth_under_init_refuses_an_explicit_pretrained(world, megadepth_tree, tmp_path, value):
    source = write_source(tmp_path / "source", depth_record(), {"pretrained": True})
    config = depth_config(tmp_path / "run", megadepth_tree, init=str(source), model_args={"pretrained": value})
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.pretrained': init replaces every weight"):
        prepared(config, DepthEstimationTrainer)
    assert not (tmp_path / "run").exists()


def test_preclaim_depth_under_init_needs_adamw_for_the_heads_10x(world, megadepth_tree, tmp_path):
    source = write_source(tmp_path / "source", depth_record(), {"pretrained": True})
    config = depth_config(tmp_path / "run", megadepth_tree, init=str(source),
                          optim_args={"optimizer": "adam", "lr": 1e-5, "weight_decay": None})
    with pytest.raises(ConfigError, match=r"'optim_args\.optimizer'.*adamw"):
        prepared(config, DepthEstimationTrainer)


# ---------------------------------------------------------------------
# wire: real tiny CPU sources, and targets started from them
# ---------------------------------------------------------------------

#: What every classification run of this section asks besides its model, data and side.
QUIET = {"epochs": 1, "batch_size": 8, "analysis_args": {"model_analysis": False, "figures": False},
         "metric_args": {"success_threshold": 0.0}}
TEN = [f"k{index}" for index in range(10)]
RENAMED = [f"r{index}" for index in range(10)]
HUNDRED = [f"h{index:02d}" for index in range(100)]


def write_tree(root: Path, names, per_class: int = 3, side: int = 16, seed: int = 0) -> Path:
    """``root/<class>/<i>.png``: noise images, a tint per class."""
    rng = np.random.default_rng(seed)
    for k, name in enumerate(names):
        for index in range(per_class):
            path = root / name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((rng.integers(0, 120, (side, side, 3)) + (k % 8) * 16).astype(np.uint8)).save(path)
    return root


def relative(model: Any) -> Dict[str, np.ndarray]:
    """``{path relative to the model: value}``: the model's own leading name dropped, as a fresh process would not have it."""
    root = f"{model.name}/"
    return {(v.path[len(root):] if v.path.startswith(root) else v.path): np.array(v) for v in model.weights}


def archive_weights(run_dir: Path) -> Dict[str, np.ndarray]:
    return relative(keras.models.load_model(str(run_dir / "best_model.keras"), compile=False))


def record_of(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "task.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def started_weights(trainer_class: Any) -> Iterator[Dict[str, np.ndarray]]:
    """Patch ``trainer_class.extra_callbacks`` so the weights ``fit`` starts from are kept (relative paths)."""
    seen: Dict[str, np.ndarray] = {}
    original = trainer_class.extra_callbacks

    class Started(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            seen.update(relative(self.model))

    def extra(self, config, data):
        return [*original(self, config, data), Started()]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer_class, "extra_callbacks", extra)
        yield seen


def started(trainer_class: Any, function: Any, **request: Any) -> Dict[str, np.ndarray]:
    """Run ``function(**request)`` in a fresh Keras session and return the weights its ``fit`` started from."""
    keras.backend.clear_session()
    with started_weights(trainer_class) as seen:
        function(**request)
    return dict(seen)


def classify(out: Any, tree: Path, side: int, model: str = "vit/vit_pico", **changes: Any) -> Dict[str, Any]:
    return {"out": out, "model": model, "data": {"format": "folder", "train_dir": str(tree)},
            "transform_args": {"image_size": side}, **QUIET, **changes}


def check_copied(block: Dict[str, Any], start: Dict[str, np.ndarray], source: Dict[str, np.ndarray],
                 prefix: str = "") -> None:
    """Every target weight the block does not name as reset, grown or resampled equals the source's at its path."""
    transfer = block["transfer"]
    others = set(transfer["reset"]) | set(transfer["grown"]) | {row["path"] for row in transfer["resampled"]}
    copied = [path for path in start if path not in others]
    assert transfer["copied"] == len(copied)
    for path in copied:
        assert path.startswith(prefix), path
        np.testing.assert_array_equal(start[path], source[path[len(prefix):]], err_msg=path)


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    root = tmp_path_factory.mktemp("trees")
    return {"ten": write_tree(root / "ten", TEN), "renamed": write_tree(root / "renamed", RENAMED, seed=1),
            "hundred": write_tree(root / "hundred", HUNDRED, per_class=2, seed=2)}


@pytest.fixture(scope="module")
def vit_source(world, trees, tmp_path_factory):
    """vit_pico on 10 classes at 32 px (patch 4, a 65-row position table), 1 epoch."""
    out = tmp_path_factory.mktemp("vit_source") / "run"
    keras.backend.clear_session()
    train_image_classification(**classify(str(out), trees["ten"], 32))
    return out


@pytest.fixture(scope="module")
def vit_64(world, trees, vit_source, tmp_path_factory):
    """vit_pico on 100 classes at 64 px, from a copy of ``vit_source`` (a resume test renames the copy away)."""
    root = tmp_path_factory.mktemp("vit_64")
    source = root / "source"
    shutil.copytree(vit_source, source)
    out = root / "run"
    request = classify(str(out), trees["hundred"], 64, init=str(source), batch_size=16)
    start = started(ImageClassificationTrainer, train_image_classification, **request)
    return SimpleNamespace(source=source, out=out, request=request, start=start)


def test_wire_a_vit_at_32_px_seeds_one_at_64_px_keeping_its_patch_and_resampling_its_table(vit_source, vit_64):
    from tests.test_models.test_dino.dinov2_numpy_reference import torch_bicubic_2d

    used = json.loads((vit_64.out / "config.json").read_text())["used"]["model_args"]
    assert used["patch_size"] == 4  # the recipe derives 16 at 64 px; the source's 4 is kept
    source, start = archive_weights(vit_source), vit_64.start
    block = record_of(vit_64.out)["init"]
    assert (block["source"], block["file"], block["sha256"]) == (
        str(vit_64.source), "best_model.keras", sha(vit_64.source / "best_model.keras"))
    assert (block["task"], block["family"], block["variant"], block["relation"]) == (
        "image_classification", "vit", "vit_pico", "same_task")
    assert block["pinned"] == {"patch_size": 4}
    assert block["geometry"] == {key: record_of(vit_source)[key] for key in ("input_shape", "channels")}
    table, grid = "pos_embed/pos_embedding", (32 // 4, 64 // 4)
    assert block["transfer"]["resampled"] == [{"path": table, "rule": "vit_pos_bicubic",
                                               "from": [1, 1 + grid[0] ** 2, 192], "to": [1, 1 + grid[1] ** 2, 192]}]
    heads = sorted(path for path in start if path.startswith("classification_head/"))
    assert heads and sorted(block["transfer"]["reset"]) == heads  # 10 classes to 100: the head is fresh
    assert sorted(block["transfer"]["dropped"]) == sorted(p for p in source if p.startswith("classification_head/"))
    assert block["transfer"]["grown"] == []
    check_copied(block, start, source)
    assert start[table].shape == (1, 1 + grid[1] ** 2, 192)
    np.testing.assert_array_equal(start[table][:, :1], source[table][:, :1])  # the class row is kept
    image = source[table][0, 1:].astype(np.float64).reshape(grid[0], grid[0], 192).transpose(2, 0, 1)
    expected = torch_bicubic_2d(image, grid[1], grid[1]).transpose(1, 2, 0).reshape(grid[1] ** 2, 192)
    np.testing.assert_allclose(start[table][0, 1:], expected, atol=1e-5)


def test_wire_the_block_passes_the_record_check_and_a_block_without_a_key_is_refused(vit_64):
    record = record_of(vit_64.out)
    assert predictor.record_problems(record) == []
    assert list(record)[-1] == "init" and list(record)[:-1] == list(predictor.RECORD_KEYS["image_classification"])
    for key in record["init"]:
        broken = {**record, "init": {name: value for name, value in record["init"].items() if name != key}}
        assert any(repr(key) in line for line in predictor.record_problems(broken)), key
    unknown = {**record, "init": {**record["init"], "extra": 1}}
    assert any("'extra'" in line for line in predictor.record_problems(unknown))


@pytest.fixture(scope="module")
def vit_tiny(world, trees, vit_source, tmp_path_factory):
    """vit_tiny (deeper, same width) from ``vit_source`` on the same classes; ``out`` and ``init`` given as ``Path``."""
    out = tmp_path_factory.mktemp("vit_tiny") / "run"
    start = started(ImageClassificationTrainer, train_image_classification,
                    **classify(out, trees["ten"], 32, model="vit/vit_tiny", init=vit_source))
    return out, start


def test_wire_a_deeper_vit_grows_its_new_blocks_and_copies_every_other_weight_with_the_head(vit_source, vit_tiny):
    out, start = vit_tiny
    source = archive_weights(vit_source)
    block = record_of(out)["init"]
    depth = 1 + max(int(re.match(r"transformer_layer_(\d+)/", p).group(1)) for p in source if p.startswith("transformer_layer_"))
    grown = sorted(p for p in start if re.match(r"transformer_layer_(\d+)/", p)
                   and int(re.match(r"transformer_layer_(\d+)/", p).group(1)) >= depth)
    assert grown and sorted(block["transfer"]["grown"]) == grown
    assert block["transfer"]["reset"] == [] and block["transfer"]["dropped"] == []  # the same 10 class names
    assert block["transfer"]["resampled"] == []
    check_copied(block, start, source)
    assert block["transfer"]["copied"] == len(source)


def test_wire_a_path_init_and_out_are_recorded_as_text_once(vit_source, vit_tiny):
    out, _ = vit_tiny
    text = (out / "config.json").read_text()
    config = json.loads(text)
    assert (config["out"], config["init"]) == (str(out), str(vit_source))
    assert text.count(f'"init": "{vit_source}"') == 1 and text.count(str(out)) == 1


def test_wire_the_head_is_reset_when_the_class_names_differ_at_an_equal_count(world, trees, vit_source, tmp_path):
    out = tmp_path / "run"
    start = started(ImageClassificationTrainer, train_image_classification,
                    **classify(str(out), trees["renamed"], 32, init=str(vit_source)))
    source = archive_weights(vit_source)
    block = record_of(out)["init"]
    heads = sorted(p for p in start if p.startswith("classification_head/"))
    assert sorted(block["transfer"]["reset"]) == heads and sorted(block["transfer"]["dropped"]) == heads
    assert all(start[p].shape == source[p].shape for p in heads)  # same shapes: only the names say it is not the same head
    assert not all(np.array_equal(start[p], source[p]) for p in heads)
    check_copied(block, start, source)


def test_wire_another_architecture_is_refused_after_the_claim_and_nothing_is_trained(world, trees, vit_source, tmp_path):
    out = tmp_path / "run"
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(**classify(str(out), trees["ten"], 32, init=str(vit_source),
                                              model_args={"use_layer_scale": True}))
    text = str(caught.value)
    assert re.search(r"'init': \d+ of \d+ weights of vit/vit_pico have no source weight of their shape and no "
                     r"declared rule: .*nothing was copied", text), text
    assert f"The run directory {out} was claimed" in text
    assert (out / "config.json").exists() and not (out / "training_log.csv").exists()
    assert not (out / "best_model.keras").exists() and not (out / "task.json").exists()


def test_wire_an_archive_that_changes_after_the_claim_stops_the_run_naming_both_hashes(
        world, trees, vit_source, tmp_path, monkeypatch):
    source = tmp_path / "source"
    shutil.copytree(vit_source, source)
    before = sha(source / "best_model.keras")
    real = predictor.load_model

    def load_then_rewrite(path):
        loaded = real(path)
        with open(Path(path) / "best_model.keras", "ab") as handle:
            handle.write(b"rewritten while the run read it")
        return loaded

    monkeypatch.setattr(predictor, "load_model", load_then_rewrite)
    out = tmp_path / "run"
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(**classify(str(out), trees["ten"], 32, init=str(source)))
    text, after = str(caught.value), sha(source / "best_model.keras")
    assert before != after and before in text and after in text and f"The run directory {out} was claimed" in text
    assert not (out / "training_log.csv").exists() and not (out / "best_model.keras").exists()


def test_wire_a_pin_is_laid_over_the_computed_value_and_another_user_value_is_refused():
    context = AutoContext(family="vit", variant="vit_pico", dataset="folder", num_classes=3, image_size=64,
                                  defaults={}, pinned={"patch_size": 4})
    optim = SimpleNamespace(optimizer="adamw")
    assert recipes.merge_model_args(context, optim, {}, {"patch_size": 16})["patch_size"] == 4
    assert recipes.merge_model_args(context, optim, {"patch_size": 4}, {"patch_size": 16})["patch_size"] == 4
    with pytest.raises(ConfigError, match=r"Invalid value for 'model_args\.patch_size': the source was built with "
                                          r"patch_size=4 and init keeps the source's architecture"):
        recipes.merge_model_args(context, optim, {"patch_size": 8}, {"patch_size": 16})
    unpinned = dataclasses.replace(context, pinned={})
    assert AutoContext(family="vit", variant="vit_pico", dataset="folder", num_classes=3, image_size=64,
                               defaults={}).pinned == {}
    assert recipes.merge_model_args(unpinned, optim, {}, {"patch_size": 16})["patch_size"] == 16


@pytest.fixture(scope="module")
def segmentation_data(tmp_path_factory):
    from ._segmentation_fixtures import make_folder

    images, masks = make_folder(tmp_path_factory.mktemp("segmentation"), count=12, size=(64, 64), classes=3)
    return {"format": "folder", "train_images": str(images), "train_masks": str(masks), "num_classes": 3}


def test_wire_a_convnext_classifier_seeds_convnext_seg_under_backbone(world, segmentation_data, tmp_path, monkeypatch):
    """The gate's pair (D-035): a cifar10 classifier with its recipe's strides 2, trained on [0, 1] inputs."""
    from tests.classification_fakes import use_tiny_raw

    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    source_dir, out = tmp_path / "source", tmp_path / "run"
    keras.backend.clear_session()
    train_image_classification(out=str(source_dir), model="convnext_v1/cifar10", data={"dataset": "cifar10"},
                               transform_args={"normalize": "unit"}, **QUIET)
    start = started(ImageSegmentationTrainer, train_image_segmentation,
                    out=str(out), model="convnext_seg/cifar10", data=segmentation_data, init=str(source_dir),
                    epochs=1, batch_size=4, transform_args={"image_size": [32, 32]},
                    analysis_args={"model_analysis": False, "figures": False})
    source = archive_weights(source_dir)
    block = record_of(out)["init"]
    assert block["relation"] == "cross_task" and block["pinned"] == {}
    kept = [p for p in source if not p.startswith("classification_head/")]
    assert sorted(block["transfer"]["dropped"]) == sorted(set(source) - set(kept))
    assert block["transfer"]["copied"] == len(kept)
    assert set(block["transfer"]["reset"]) == {p for p in start if not p.startswith("backbone/")}
    check_copied(block, start, source, prefix="backbone/")


def test_wire_a_convunext_segmenter_seeds_the_biased_denoiser_with_a_fresh_output(world, segmentation_data, tmp_path):
    from .test_image_denoising_task import mapping, write_images

    source_dir, out = tmp_path / "source", tmp_path / "run"
    keras.backend.clear_session()
    train_image_segmentation(out=str(source_dir), model="convunext/tiny", data=segmentation_data, epochs=1,
                             batch_size=4, transform_args={"image_size": [32, 32]},
                             analysis_args={"model_analysis": False, "figures": False})
    folders = write_images(tmp_path / "train", 6, 0), write_images(tmp_path / "val", 3, 1)
    start = started(ImageDenoisingTrainer, train_image_denoising,
                    **mapping(out, folders, "convunext_biased", epochs=1, init=str(source_dir),
                              transform_args={"patch_size": 32, "patches_per_image": 2}))
    source = archive_weights(source_dir)
    block = record_of(out)["init"]
    assert block["relation"] == "cross_task"
    assert sorted(block["transfer"]["reset"]) == sorted(p for p in start if p.startswith("final_output/"))
    assert sorted(block["transfer"]["dropped"]) == sorted(p for p in source if p.startswith("final_output/"))
    assert block["transfer"]["copied"] == len(start) - len(block["transfer"]["reset"])
    check_copied(block, start, source)


def test_wire_a_denoiser_warm_start_scores_the_sources_last_validation_loss_at_epoch_zero(world, tmp_path):
    from .test_image_denoising_task import mapping, write_images

    folders = write_images(tmp_path / "train", 6, 0), write_images(tmp_path / "val", 3, 1)
    keras.backend.clear_session()
    first = train_image_denoising(**mapping(tmp_path / "first", folders, "bfcnn", epochs=1))
    start = started(ImageDenoisingTrainer, train_image_denoising,
                    **mapping(tmp_path / "warm", folders, "bfcnn", epochs=1, init=str(first.run_dir)))
    keras.backend.clear_session()
    cold = train_image_denoising(**mapping(tmp_path / "cold", folders, "bfcnn", epochs=1))
    summaries = [json.loads((tmp_path / name / "results_summary.json").read_text()) for name in ("first", "warm", "cold")]
    # The source's one epoch scored these weights on this very validation set: the warm start's epoch 0 is that score.
    last = json.loads((Path(first.run_dir) / "training_history.json").read_text())["val_loss"][-1]
    assert summaries[1]["initial_loss_sanity_eval"]["loss"] == pytest.approx(last, rel=1e-5)
    assert summaries[2]["initial_loss_sanity_eval"]["loss"] != pytest.approx(last, rel=1e-3)
    assert summaries[0]["input_psnr"] == summaries[1]["input_psnr"] == summaries[2]["input_psnr"]
    block = record_of(tmp_path / "warm")["init"]
    assert block["transfer"]["reset"] == [] and block["transfer"]["copied"] == len(start)
    check_copied(block, start, archive_weights(Path(first.run_dir)))


def test_wire_tirex_at_a_longer_lookback_copies_every_weight(world, tmp_path):
    data = {"lookback": 32, "horizon": 8, "realizations": 1, "series_length": 600, "patterns": ["ar_process"],
            "n_select": 16, "n_test": 16, "steps_per_epoch": 2}
    request = {"model": "tirex/tiny", "epochs": 1, "batch_size": 8,
               "analysis_args": {"model_analysis": False, "figures": False}}
    keras.backend.clear_session()
    train_time_series_forecasting(out=str(tmp_path / "source"), data=data, **request)
    start = started(TimeSeriesForecastingTrainer, train_time_series_forecasting, out=str(tmp_path / "run"),
                    data={**data, "lookback": 64}, init=str(tmp_path / "source"), **request)
    source = archive_weights(tmp_path / "source")
    block = record_of(tmp_path / "run")["init"]
    assert block["geometry"]["lookback"] == 32 and record_of(tmp_path / "run")["lookback"] == 64
    summary = summary_of(tmp_path / "run")  # epoch_zero_of once, from the head, not again from the task's extras
    assert (summary["epoch_zero_of"], summary["init"]) == ("init", block) and "loaded_success_value" not in summary
    assert block["transfer"]["copied"] == len(source) == len(start)
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, start, source)


@pytest.fixture(scope="module")
def wide_megadepth_tree(tmp_path_factory):
    """The depth tests' MegaDepth-shaped tree with 48 px images, so a 42 px crop fits."""
    import tests.test_train.test_common_depth_data as depth_data

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(depth_data, "SIDE", 48)
        return depth_data.write_megadepth_tree(tmp_path_factory.mktemp("wide_megadepth"))


def test_wire_depth_at_a_larger_side_copies_every_weight_and_reads_no_official_file(
        world, wide_megadepth_tree, tmp_path, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "no_weights"))
    source_dir, out = tmp_path / "source", tmp_path / "run"
    keras.backend.clear_session()
    train_depth_estimation(**depth_request(source_dir, wide_megadepth_tree, 28, model_args={"pretrained": False}))
    start = started(DepthEstimationTrainer, train_depth_estimation, **depth_request(out, wide_megadepth_tree, 42, init=str(source_dir)))
    source = archive_weights(source_dir)
    block = record_of(out)["init"]
    assert block["geometry"]["input_size"] == [28, 28] and record_of(out)["input_size"] == [42, 42]
    assert (summary_of(out)["epoch_zero_of"], summary_of(out)["init"]) == ("init", block)
    assert block["transfer"]["copied"] == len(source) == len(start)
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, start, source)


def depth_request(out: Path, root: Path, side: int, **changes: Any) -> Dict[str, Any]:
    return {"out": str(out), "model": "depth_anything/vit_s", "batch_size": 4, "epochs": 1,
            "data": {"root": str(root), "train_split": 0.6, "n_select": 4, "n_test": 4, "layout_samples": 4,
                     "steps_per_epoch": 1, "workers": 0},
            "transform_args": {"image_size": side}, "analysis_args": {"model_analysis": False, "figures": False},
            **changes}


# ---------------------------------------------------------------------
# resume: the pins come from the run's own record; the source is never opened
# ---------------------------------------------------------------------


def watch_opens(monkeypatch: Any, prefix: str) -> list:
    """Spy on ``open`` (builtins, io, os); return the list of the paths opened under ``prefix``, filled as it goes."""
    opened: list = []

    def spied(real):
        def spy(file, *arguments, **keywords):
            if isinstance(file, (str, os.PathLike)) and os.fspath(file).startswith(prefix):
                opened.append(os.fspath(file))
            return real(file, *arguments, **keywords)
        return spy

    for owner in (builtins, io, os):
        monkeypatch.setattr(owner, "open", spied(owner.open))
    return opened


def test_resume_an_init_run_with_its_source_renamed_away_rebuilds_its_architecture_and_opens_no_source_file(
        vit_64, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report
    moved = vit_64.source.with_name("moved_away")
    vit_64.source.rename(moved)
    rebuilt = {}
    real_restore = resume_decisions.restore_finished_fit

    def restore(model, *arguments):
        rebuilt.update({path: value.shape for path, value in relative(model).items()})
        return real_restore(model, *arguments)

    try:
        opened = watch_opens(monkeypatch, str(vit_64.source))
        monkeypatch.setattr(resume_decisions, "restore_finished_fit", restore)
        keras.backend.clear_session()
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True})
    finally:
        monkeypatch.undo()
        moved.rename(vit_64.source)
    summary = json.loads((run_dir / "results_summary.json").read_text())
    assert summary["reporting_resumed"] is True
    assert opened == []
    first = {path: value.shape for path, value in archive_weights(vit_64.out).items()}
    kernel = next(p for p in rebuilt if p.startswith("patch_embed/") and p.endswith("kernel"))
    assert rebuilt == first and rebuilt[kernel][:2] == (4, 4)  # patch 4 at 64 px: the record's pin


def test_resume_a_record_without_an_init_block_is_refused_before_anything_is_written(vit_64, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    record = record_of(run_dir)
    del record["init"]
    (run_dir / "task.json").write_text(json.dumps(record))
    before = sorted(p.name for p in run_dir.iterdir())
    with pytest.raises(ConfigError, match=r"'init': .*task\.json holds no init block"):
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True})
    assert sorted(p.name for p in run_dir.iterdir()) == before


#: The refusal of a run recorded before init replaced checkpoint (review P2): no request can resume it, so it says so.
PRE_INIT = (r"Cannot resume (?P<run>\S+): the run was recorded before 'init' replaced 'checkpoint' \(its "
            r"config_requested\.json holds 'checkpoint'\), so it cannot be resumed\. Start a new run with another "
            r"'out', and pass init=(?P=run) if its weights are wanted \(init reads its best_model\.keras\)$")


def test_resume_a_run_started_before_init_is_refused_as_not_resumable_naming_init_before_any_write(vit_64, tmp_path):
    # A run interrupted before init existed: its config_requested.json holds "checkpoint": null and its record no init
    # block. No request can resume it (checkpoint is an unknown key now), so the refusal names the way forward instead.
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    (run_dir / "results_summary.json").unlink()
    record = record_of(run_dir)
    del record["init"]
    (run_dir / "task.json").write_text(json.dumps(record))
    requested = json.loads((run_dir / "config_requested.json").read_text())
    requested = {("checkpoint" if key == "init" else key): (None if key == "init" else value)
                 for key, value in requested.items()}
    (run_dir / "config_requested.json").write_text(json.dumps(requested))
    before = {path.name: path.stat().st_mtime_ns for path in run_dir.iterdir()}
    with pytest.raises(ConfigError) as refused:
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True, "init": None})
    logger.info(f"pre-init resume refused: {refused.value}")
    assert re.fullmatch(PRE_INIT, str(refused.value)).group("run") == str(run_dir)
    assert {path.name: path.stat().st_mtime_ns for path in run_dir.iterdir()} == before


def test_resume_a_pre_init_forecasting_request_is_refused_by_checkpoint_before_any_request_difference(tmp_path):
    # A pre-4g forecasting run also recorded data.noise_level, which is gone too; the refusal names neither difference.
    from train.api.run_dir import load_resume_point

    (tmp_path / "config_requested.json").write_text(json.dumps({"checkpoint": None, "data": {"noise_level": 0.1}}))
    with pytest.raises(ConfigError) as refused:
        load_resume_point(tmp_path, {"init": None, "data": {}})
    assert re.fullmatch(PRE_INIT, str(refused.value)) and "noise_level" not in str(refused.value)


#: The refusal of a pretrain run recorded before its family was renamed (D-052, review S2): no request can resume it.
PRE_RENAME = (r"Cannot resume (?P<run>\S+): it is a pretrain run recorded before its family was renamed \(energy_transformer "
              r"to et_mim, beit to beit_mim\), so its config_requested\.json names '(?P<model>[^']+)', which no "
              r"pretrain request can name, and it cannot be resumed\. Start a new run with another 'out'; it cannot be "
              r"an init source either \(it has no task\.json\)$")


def test_resume_a_pretrain_run_recorded_before_the_rename_is_refused_as_not_resumable_before_any_write(world, tmp_path):
    # The pass-2 probe: a killed pre-4h et_mim run (backup, fit sidecar, the old name in its request, no task.json),
    # resumed under the new name; then the p4d shape (beit, `checkpoint` held too), which this refusal answers first.
    from train.api.run_dir import _plain_request, load_resume_point

    run = tmp_path / "run"
    (run / "checkpoints" / "backup").mkdir(parents=True)
    (run / "checkpoints" / "fit_state.json").write_text("{}")
    request = _plain_request(validate(PretrainConfig, {"out": str(run), "model": "et_mim/et_tiny",
                                                       "data": {"dataset": "cifar10"}, "resume_interrupted": True}))
    (run / "config_requested.json").write_text(json.dumps({**request, "model": "energy_transformer/et_tiny"}))
    before = {path: path.stat().st_mtime_ns for path in run.rglob("*")}
    with pytest.raises(ConfigError) as refused:
        train_pretrain(out=str(run), model="et_mim/et_tiny", data={"dataset": "cifar10"}, resume_interrupted=True)
    assert re.fullmatch(PRE_RENAME, str(refused.value)).group("run", "model") == (str(run), "energy_transformer/et_tiny")
    assert {path: path.stat().st_mtime_ns for path in run.rglob("*")} == before
    (run / "config_requested.json").write_text(json.dumps({**request, "model": "beit/beit_tiny", "checkpoint": None}))
    with pytest.raises(ConfigError) as refused:
        load_resume_point(run, request)
    assert re.fullmatch(PRE_RENAME, str(refused.value)).group("model") == "beit/beit_tiny"
    (run / "task.json").write_text("{}")  # a classifier run under the same name holds one: not this refusal
    with pytest.raises(ConfigError, match=PRE_INIT):
        load_resume_point(run, request)


def test_resume_a_user_value_against_the_records_pin_is_refused_before_the_claim(vit_64, tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(vit_64.out, run_dir)
    (run_dir / "results_summary.json").unlink()
    with pytest.raises(ConfigError, match=r"'model_args\.patch_size': the source was built with patch_size=4"):
        train_image_classification(**{**vit_64.request, "out": str(run_dir), "resume_interrupted": True,
                                      "model_args": {"patch_size": 16}})


# ---------------------------------------------------------------------
# split: the source's training data against this run's held-out data; counted and marked, never refused
# ---------------------------------------------------------------------


def a_source(config: Dict[str, Any], record: Optional[Dict[str, Any]] = None, path: Path = Path("/nowhere")) -> Any:
    """A source as ``read_source`` returns it, from its ``config.json`` (only what the hooks read)."""
    return init_from.Source(path, record or classification_record(), config, "0" * 64, None, {})


def cut(n: int, seed: int, fraction: float = 0.1, cap: Optional[int] = None):
    """The ORACLE's seeded cut, written here: ``n`` rows permuted by ``seed``, capped, the first ``fraction`` held out.

    :return: ``(trained rows, selection rows)`` as sets.
    """
    order = np.random.default_rng(seed).permutation(n)[:cap]
    held = int(len(order) * fraction)
    return set(order[held:].tolist()), set(order[:held].tolist())


#: ``name: (the source's seed, data, the target's seed)``, CIFAR-10-shaped (50,000 train rows, 10,000 test rows).
CIFAR_CASES = {
    "seed_43": (42, {}, 43),
    "same_keys": (42, {}, 42),
    "capped_source": (42, {"max_samples": 20000}, 42),
    "other_dataset": (42, {"dataset": "cifar100"}, 42),
}


@pytest.mark.parametrize("case", sorted(CIFAR_CASES))
def test_split_a_cifar10_target_counts_the_sources_rows_as_the_oracle(world, tmp_path, case):
    seed, extra, target_seed = CIFAR_CASES[case]
    data = {"format": "builtin", "dataset": "cifar10", "validation_split": None, "max_samples": None, **extra}
    trainer = ImageClassificationTrainer(classifier_config(tmp_path / "run", seed=target_seed))
    fake = SimpleNamespace(n_train=45000, n_select=5000, n_test=10000, fingerprint=None)
    split = trainer.init_overlap(a_source({"seed": seed, "data": data}), trainer.config, fake)
    assert predictor._split_problems(split) == []
    flags = init_from.bias_of({"split": split})
    if case == "other_dataset":
        assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("different_data", 0, 0)
        assert not any(flags.values())
        return
    trained = cut(50000, seed, cap=extra.get("max_samples"))[0]
    expected = len(trained & cut(50000, target_seed)[1])
    assert split["n_select_overlap"] == expected and split["n_test_overlap"] == 0
    if case == "same_keys":
        assert expected == 0 and split["relation"] == "same_split" and not any(flags.values())
    else:
        assert expected > 0 and split["relation"] == "overlap" and split["unit"] == "rows"
        assert flags == {"selection_biased": True, "test_biased": False, "overlap_unknown": False}
        assert "seed=42" in split["reason"] and f"of this run's 5000 selection rows" in split["reason"]


def cifar_request(out: Path, **changes: Any) -> Dict[str, Any]:
    return {"out": str(out), "model": "vit/vit_pico", "data": {"dataset": "cifar10"}, **QUIET, **changes}


@pytest.fixture(scope="module")
def cifar_source(world, tmp_path_factory):
    """vit_pico on tiny CIFAR-10 stand-in arrays (40 train rows), seed 42, 1 epoch."""
    from tests.classification_fakes import use_tiny_raw

    out = tmp_path_factory.mktemp("cifar_source") / "run"
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        keras.backend.clear_session()
        train_image_classification(**cifar_request(out, seed=42))
    return out


def summary_of(run_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "results_summary.json").read_text())


def test_split_end_to_end_an_overlap_and_an_unknown_split_are_marked_and_the_runs_finish(
        world, cifar_source, trees, tmp_path, monkeypatch):
    from tests.classification_fakes import use_tiny_raw

    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    out = tmp_path / "seed43"
    keras.backend.clear_session()
    # The verdict is the bar's (D-013: a fine-tune must beat its loaded weights), never the overlap's: the run finishes.
    assert train_image_classification(**cifar_request(out, seed=43, init=str(cifar_source))).success is summary_of(
        out)["success"]
    record = record_of(out)
    split = record["init"]["split"]
    expected = len(cut(40, 42)[0] & cut(40, 43)[1])
    assert expected > 0 and (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", expected, 0)
    flags = {key: summary_of(out)[key] for key in ("selection_biased", "test_biased", "overlap_unknown")}
    assert flags == {"selection_biased": True, "test_biased": False, "overlap_unknown": False}
    log = (out / "run.log").read_text()
    assert re.search(r"(Reached|Missed) success_threshold \S+ on \S+: best \S+ \(selection-biased\)", log), log[-2000:]
    assert "seed=42, data.validation_split=0.1, data.max_samples=null as the source's give 0" in log
    assert predictor.record_problems(record) == []
    for broken in ({**split, "n_select_overlap": None}, {**split, "relation": "leaked"}, {**split, "extra": 1}):
        assert predictor.record_problems({**record, "init": {**record["init"], "split": broken}}) != []
    # A folder target of a built-in source: no rule matches its files to the source's rows.
    out = tmp_path / "folder"
    keras.backend.clear_session()
    assert train_image_classification(**classify(str(out), trees["ten"], 32, init=str(cifar_source))).success is (
        summary_of(out)["success"])
    split = record_of(out)["init"]["split"]
    assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("unknown", None, None)
    assert summary_of(out)["overlap_unknown"] is True and summary_of(out)["selection_biased"] is False
    assert "(split overlap unknown)" in (out / "run.log").read_text()


def test_split_the_default_hook_is_unknown_with_its_reason():
    split = TaskTrainer.init_overlap(SimpleNamespace(), None, None, None)
    assert split == init_from.split_block("unknown", "this task does not rebuild a source's rows")
    assert predictor._split_problems(split) == []
    assert init_from.bias_words({"split": split}) == " (split overlap unknown)" and init_from.bias_words(None) == ""


FORECAST_DATA = {"format": "builtin", "dataset": "synthetic", "lookback": 32, "horizon": 8, "realizations": 1,
                 "series_length": 600, "patterns": ["ar_process"], "select_fraction": 0.15,
                 "test_fraction": 0.15}


def forecaster(tmp_path: Path, seed: int = 42, **data: Any) -> Any:
    return TimeSeriesForecastingTrainer(validate(TimeSeriesForecastingConfig, {
        "out": str(tmp_path / "run"), "model": "tirex/tiny", "seed": seed, "data": {**FORECAST_DATA, **data}}))


def windows_at(starts: np.ndarray, test: Any = ((0, 520), (0, 550)), **data: Any) -> Any:
    """The ``data`` a forecasting hook reads: a 600-row bank, these select window starts and two test windows."""
    return SimpleNamespace(bank=np.zeros((1, 600, 1)), select_windows=np.stack([0 * starts, starts], 1),
                           test_windows=np.array(test), **{"csv_sha256": None, "value_cols": None, **data})


def test_split_forecasting_counts_windows_in_the_sources_train_segment(world, tmp_path):
    trainer = functools.partial(forecaster, tmp_path)
    source = a_source({"seed": 42, "data": dict(FORECAST_DATA)}, forecast_record("tirex/tiny", 32, 8))
    # This run's select segment begins at 600 - 180 - 90 = 330; the source trained on rows [0, 600 - 90 - 90 = 420).
    starts = np.arange(330, 500, 7)
    fake = windows_at(starts)
    longer = trainer(select_fraction=0.3)
    split = longer.init_overlap(source, longer.config, fake)
    assert (split["relation"], split["unit"]) == ("overlap", "windows")
    assert 0 < (starts < 420).sum() < len(starts)
    assert (split["n_select_overlap"], split["n_test_overlap"]) == (int((starts < 420).sum()), 0)
    same = trainer(lookback=64)
    assert same.init_overlap(source, same.config, fake)["relation"] == "same_split"
    shared = trainer(realizations=2)
    assert shared.init_overlap(source, shared.config, fake)["relation"] == "unknown"


def test_split_forecasting_a_bank_of_another_length_or_seed_is_unknown_with_its_reason(world, tmp_path):
    # Review R2 and R4: at one seed a 600-row series is the first 600 rows of the 1,200-row one, so a source that
    # trained on rows [0, 840) holds every window of this run verbatim; and the generator's deterministic patterns
    # (trends, chaotic maps, fixed seasonalities) barely move with the seed. Neither is other data, so neither is a 0.
    source = a_source({"seed": 42, "data": {**FORECAST_DATA, "series_length": 1200}}, forecast_record("tirex/tiny", 32, 8))
    fake = windows_at(np.arange(420, 560, 10), test=((0, 560), (0, 570)))
    for seed, length, why in ((42, 600, "begins with the rows"), (43, 1200, "barely move with the seed")):
        target = forecaster(tmp_path, seed, series_length=length)
        split = target.init_overlap(source, target.config, fake)
        assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("unknown", None, None)
        assert why in split["reason"], split["reason"]
        assert init_from.bias_of({"split": split})["overlap_unknown"] is True


def test_split_forecasting_a_recorded_noise_level_is_not_other_data(world, tmp_path):
    # Review R2 scenario 2 and R5 (D-044): every generator pattern drew at its own noise level, so a source that
    # recorded data.noise_level 0.2 trained on this run's very bank; its train segment is counted, not called other data.
    source = a_source({"seed": 42, "data": {**FORECAST_DATA, "noise_level": 0.2}}, forecast_record("tirex/tiny", 32, 8))
    starts = np.arange(330, 500, 7)
    target = forecaster(tmp_path, select_fraction=0.3)
    split = target.init_overlap(source, target.config, windows_at(starts))
    assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", int((starts < 420).sum()), 0)


def test_split_forecasting_a_csv_is_counted_within_one_file_and_its_columns(world, tmp_path):
    # The csv branch: the same file (sha256) and value columns count windows as a builtin bank does; another file or
    # other columns are unknown, since the rows may still be the same.
    csv = tmp_path / "x.csv"
    csv.write_text("v\n" + "\n".join(str(i) for i in range(600)))

    def target(**data: Any) -> Any:
        return TimeSeriesForecastingTrainer(validate(TimeSeriesForecastingConfig, {
            "out": str(tmp_path / "run"), "model": "tirex/tiny",
            "data": {"format": "csv", "path": str(csv), "value_cols": ["v"], "lookback": 32, "horizon": 8, **data}}))

    source = a_source({"seed": 42, "data": {"format": "csv", "select_fraction": 0.15, "test_fraction": 0.15}},
                      {**forecast_record("tirex/tiny", 32, 8), "data_sha256": "a" * 64, "value_cols": ["v"]})
    starts = np.arange(330, 500, 7)
    fake = windows_at(starts, csv_sha256="a" * 64, value_cols=("v",))
    longer = target(select_fraction=0.3)
    split = longer.init_overlap(source, longer.config, fake)
    assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", int((starts < 420).sum()), 0)
    same = target()
    assert same.init_overlap(source, same.config, fake)["relation"] == "same_split"
    for other in (windows_at(starts, csv_sha256="b" * 64, value_cols=("v",)),
                  windows_at(starts, csv_sha256="a" * 64, value_cols=("w",))):
        assert same.init_overlap(source, same.config, other)["relation"] == "unknown"


def test_split_depth_counts_the_sources_train_scenes_among_this_runs(world, megadepth_tree, tmp_path):
    from train.common.depth_data import deal_megadepth_scenes
    from train.common.megadepth import split_megadepth_scenes

    def trainer(split: float) -> Any:
        return DepthEstimationTrainer(validate(DepthEstimationConfig, {
            "out": str(tmp_path / "run"), "model": "depth_anything/vit_s",
            "data": {"root": str(megadepth_tree), "train_split": split}}))

    deal = deal_megadepth_scenes(str(megadepth_tree), 0.4)
    fake = SimpleNamespace(select_scenes=[name for name, _, _ in deal.select],
                           test_scenes=[name for name, _, _ in deal.test])
    source = a_source({"data": {"root": str(megadepth_tree), "train_split": 0.6}}, depth_record())
    trained = {name for name, _, _ in split_megadepth_scenes(str(megadepth_tree), 0.6)[0]}
    expected = len(trained & set(fake.select_scenes)), len(trained & set(fake.test_scenes))
    target = trainer(0.4)
    split = target.init_overlap(source, target.config, fake)
    assert sum(expected) > 0 and split["relation"] == "overlap"
    assert (split["n_select_overlap"], split["n_test_overlap"]) == expected
    same = trainer(0.6)
    assert same.init_overlap(source, same.config, fake)["relation"] == "same_split"
    moved = a_source({"data": {"root": str(tmp_path / "moved"), "train_split": 0.6}}, depth_record())
    assert target.init_overlap(moved, target.config, fake)["relation"] == "unknown"


def test_split_denoising_counts_the_sources_train_files_among_this_runs_held_out_files(world, tmp_path):
    from .test_image_denoising_task import mapping, write_images

    train, val, held = (write_images(tmp_path / name, 3, seed) for seed, name in enumerate(("train", "val", "held")))
    target = ImageDenoisingTrainer(validate(ImageDenoisingConfig, mapping(tmp_path / "run", (train, val))))
    target._held_out_sets = {"held": sorted(str(path) for path in held.iterdir())}
    fake = SimpleNamespace(val_files=tuple(sorted(str(path) for path in val.iterdir())))
    # The source trained on this run's validation and held-out directories.
    source = a_source({"data": {"train_dirs": [str(val), str(held)], "val_dirs": [str(train)]}})
    split = target.init_overlap(source, target.config, fake)
    assert (split["relation"], split["unit"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", "files", 3, 3)
    assert init_from.bias_of({"split": split}) == {"selection_biased": True, "test_biased": True,
                                                  "overlap_unknown": False}
    apart = a_source({"data": {"train_dirs": [str(train)], "val_dirs": [str(val)]}})
    assert target.init_overlap(apart, target.config, fake)["n_select_overlap"] == 0
    gone = a_source({"data": {"train_dirs": [str(tmp_path / "gone")], "val_dirs": [str(val)]}})
    assert target.init_overlap(gone, target.config, fake)["relation"] == "unknown"


# ---------------------------------------------------------------------
# bar: under init a fine-tune must beat the loaded weights' epoch zero (D-013)
# ---------------------------------------------------------------------

#: ``task: (trainer, initial.metrics key, the loaded weights' score, the identity's PSNR the success value is net of,
#: the absolute bar)``; every one of the three is higher-is-better, and the loaded weights already clear the bar.
BAR_TASKS = {
    "classification": (ImageClassificationTrainer, "accuracy", 0.6, 0.0, 0.2),
    "segmentation": (ImageSegmentationTrainer, "miou", 0.6, 0.0, 0.2),
    "denoising": (ImageDenoisingTrainer, "psnr", 26.0, 20.0, 1.0),
}


#: A rate that moves no float32 weight: lr 1e-30 and no warmup (whose start rate, 1e-8, would train).
STILL = {"lr": 1e-30, "warmup_epochs": 0}


def write_sidecar(run_dir: Path, metrics: Dict[str, float]) -> Path:
    from train.common.resume_state import update_fit_state

    path = Path(run_dir) / "checkpoints" / "fit_state.json"
    update_fit_state(path, initial={"metrics": metrics, "ratio": None, "warned": None})
    return path


@pytest.mark.parametrize("task", sorted(BAR_TASKS))
def test_bar_under_init_a_fine_tune_must_beat_the_loaded_weights_epoch_zero_read_from_the_sidecar(task, tmp_path):
    trainer_class, key, score, net_of, threshold = BAR_TASKS[task]
    write_sidecar(tmp_path, {"loss": 1.0, key: score})
    config = SimpleNamespace(init="/the/source", metric_args=SimpleNamespace(success_threshold=None))
    trainer = object.__new__(trainer_class)
    trainer._identity = {"psnr": net_of}  # the input's PSNR, which denoising's success value is net of
    held = score - net_of  # what the fine-tune reaches when it only holds the loaded weights' score
    assert trainer.bar_is_met(config, held, threshold) is True  # the absolute bar alone: today's verdict
    loaded = init_from.loaded_value(config, tmp_path, key, net_of)
    assert loaded == held
    assert trainer.bar_is_met(config, held, threshold, loaded) is False
    raw = np.float32(score)  # the band is in float32 steps of the metric's own value (review R3), not of the gain
    step = np.spacing(raw)
    assert trainer.bar_is_met(config, float(raw + np.float32(16) * step) - net_of, threshold, loaded) is False  # band
    assert trainer.bar_is_met(config, float(raw + np.float32(17) * step) - net_of, threshold, loaded) is True
    assert trainer.bar_is_met(config, held - 0.1, threshold, loaded) is False
    # Without init nothing is read (not even a missing file) and the verdict is today's.
    assert init_from.loaded_value(SimpleNamespace(init=None), tmp_path / "missing", key) is None
    assert trainer.bar_is_met(config, held, threshold, None) is True
    # A sidecar without the value under init is named, not read as zero.
    write_sidecar(tmp_path / "bare", {"loss": 1.0})
    with pytest.raises(ValueError, match=rf"initial\.metrics\.{key}"):
        init_from.loaded_value(config, tmp_path / "bare", key)


def test_bar_the_denoising_band_is_on_the_raw_val_psnr_so_a_low_gain_keeps_its_band(tmp_path):
    # Review R3: input 20.0 dB, loaded val_psnr 20.5 dB, a gain of 0.5. Sixteen float32 steps of the gain are half a step
    # of the 20.5 the metric rounds, so on the gain's scale a re-evaluation one step higher would beat the loaded weights.
    write_sidecar(tmp_path, {"loss": 1.0, "psnr": 20.5})
    config = SimpleNamespace(init="/the/source")
    trainer = object.__new__(ImageDenoisingTrainer)
    trainer._identity = {"psnr": 20.0}
    loaded = init_from.loaded_value(config, tmp_path, "psnr", 20.0)
    raw = np.float32(20.5)
    for steps, met in ((1, False), (16, False), (17, True)):
        reached = float(raw + np.float32(steps) * np.spacing(raw)) - 20.0
        assert trainer.bar_is_met(config, reached, 0.1, loaded) is met, steps


def test_bar_a_classifier_that_does_not_improve_on_its_loaded_weights_misses_and_its_resume_reads_the_sidecar(
        world, trees, vit_source, tmp_path):
    from .test_image_denoising_task import flatten

    out = tmp_path / "run"
    # lr 1e-30 with no warmup (whose start rate is 1e-8) moves no float32 weight, so the fine-tune scores what the
    # loaded weights scored at epoch zero; QUIET's success_threshold 0.0 is met by any score.
    request = classify(str(out), trees["ten"], 32, init=str(vit_source), optim_args=STILL)
    keras.backend.clear_session()
    assert train_image_classification(**request).success is False
    sidecar_path = out / "checkpoints" / "fit_state.json"
    sidecar, summary = json.loads(sidecar_path.read_text()), summary_of(out)
    loaded = sidecar["initial"]["metrics"]["accuracy"]
    assert summary["success_value"] == pytest.approx(loaded, abs=1e-6) and loaded >= summary["success_threshold"]
    log = (out / "run.log").read_text()
    assert "Evaluate of the loaded weights before fit: " in log and "Untrained evaluate BEFORE fit" not in log
    assert re.search(r"Under init the best val_accuracy \S+ does not beat the loaded weights' epoch-zero \S+", log)
    assert (summary["initial_loss_ratio"], summary["init_scale_warning"], sidecar["initial"]["ratio"]) == (None,) * 3
    assert init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    # A reporting-only resume reads epoch zero from the sidecar, not from the restored weights: the same number is the
    # same miss (exit 1 on the command line), a lower number there is a met bar.
    for score, code in ((loaded, 1), (-1.0, 0)):
        sidecar["initial"]["metrics"]["accuracy"] = score
        sidecar_path.write_text(json.dumps(sidecar))
        (out / "results_summary.json").unlink()
        keras.backend.clear_session()
        assert cli.main(["image_classification", *flatten("", {**request, "resume_interrupted": True}, [])]) == code
        assert summary_of(out)["reporting_resumed"] is True and summary_of(out)["success"] is (code == 0)


@pytest.fixture(scope="module")
def segmenter(world, segmentation_data, tmp_path_factory):
    """convunext/tiny segmentation at 32 px, 1 epoch (layernorm only: a 1e-30 rate changes no score)."""
    out = tmp_path_factory.mktemp("segmenter") / "run"
    keras.backend.clear_session()
    train_image_segmentation(out=str(out), model="convunext/tiny", data=segmentation_data, epochs=1, batch_size=4,
                             transform_args={"image_size": [32, 32]},
                             analysis_args={"model_analysis": False, "figures": False})
    return out


def test_bar_a_segmenter_that_does_not_improve_on_its_loaded_weights_misses(world, segmentation_data, segmenter,
                                                                             tmp_path):
    out = tmp_path / "run"
    keras.backend.clear_session()
    result = train_image_segmentation(out=str(out), model="convunext/tiny", data=segmentation_data, epochs=1,
                                      batch_size=4, transform_args={"image_size": [32, 32]}, init=str(segmenter),
                                      optim_args=STILL, metric_args={"success_threshold": 0.0},
                                      analysis_args={"model_analysis": False, "figures": False})
    loaded = json.loads((out / "checkpoints" / "fit_state.json").read_text())["initial"]["metrics"]["miou"]
    summary = summary_of(out)
    assert result.success is False and summary["success"] is False
    assert summary["success_value"] == pytest.approx(loaded, abs=1e-6) and loaded >= summary["success_threshold"]


# ---------------------------------------------------------------------
# record: the summary of a fine-tuned run, and its predict and export with the source gone
# ---------------------------------------------------------------------

#: The summary keys an ``init`` run adds; a run without ``init`` has none of them.
INIT_SUMMARY_KEYS = {"init", "epoch_zero_of", "selection_biased", "test_biased", "overlap_unknown",
                     "loaded_success_value"}


def test_record_the_summary_holds_the_init_block_and_what_the_bar_compared(vit_source, vit_64):
    summary, block = summary_of(vit_64.out), record_of(vit_64.out)["init"]
    assert summary["init"] == block and block["transfer"]["copied"] > 0  # the counts are the task.json block's
    assert summary["epoch_zero_of"] == "init"
    assert {key: summary[key] for key in init_from.bias_of(block)} == init_from.bias_of(block)
    sidecar = json.loads((vit_64.out / "checkpoints" / "fit_state.json").read_text())
    assert summary["loaded_success_value"] == sidecar["initial"]["metrics"]["accuracy"]
    assert summary["initial_loss_ratio"] is None and init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    assert not INIT_SUMMARY_KEYS & set(summary_of(vit_source))


def test_record_a_fine_tuned_run_loads_predicts_and_exports_at_its_size_with_its_source_renamed_away(vit_64, tmp_path):
    from train.api.exporting import export

    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (64, 64, 3)).astype(np.uint8)).save(image)
    moved = vit_64.source.with_name("moved_for_predict")
    vit_64.source.rename(moved)
    try:
        loaded = predictor.load_model(vit_64.out)
        answer = loaded.predict(str(image))
        target = export(model=loaded.model, out=tmp_path / "exported.keras", format="keras",
                        input_shape=loaded.record["input_shape"], record=loaded.record)
        beside = json.loads(Path(f"{target}.task.json").read_text())
        again = predictor.load_model(target).predict(str(image))
    finally:
        moved.rename(vit_64.source)
    assert loaded.record["input_shape"] == [64, 64, 3] and beside["init"] == record_of(vit_64.out)["init"]
    assert sorted(answer["labels"]) == HUNDRED and again["labels"] == answer["labels"]
    np.testing.assert_allclose(again["scores"], answer["scores"], atol=1e-5)


#: One record of each of the eight kinds, built by the predictor's own builders.
RECORD_KINDS = {
    "image_classification": classification_record,
    "image_segmentation": segmentation_record,
    "image_denoising": lambda: predictor.build_denoising_record(
        model="bfunet/tiny", family="bfunet", variant="tiny", channels=3, size_multiple=8),
    "time_series_forecasting": forecast_record,
    "depth_estimation": depth_record,
    "pretrain": pretrain_record,
    "language_model_pretraining": language_record,
    "text_classification": text_classification_record,
}


@pytest.mark.parametrize("kind", sorted(RECORD_KINDS))
def test_record_every_kind_takes_a_well_formed_init_block_and_refuses_an_unknown_key_in_it(kind):
    block = {"source": "/runs/a", "file": "best_model.keras", "sha256": "0" * 64, "task": kind, "family": "f",
             "variant": "v", "geometry": {"input_shape": [32, 32, 3]}, "relation": "same_task", "pinned": {},
             "transfer": {"copied": 3, "resampled": [{"path": "t", "rule": "vit_pos_bicubic", "from": [1, 5],
                                                     "to": [1, 17]}], "reset": ["h/kernel"], "grown": [],
                          "dropped": []},
             "split": init_from.split_block("unknown", init_from.NOT_REBUILT)}
    record = {**RECORD_KINDS[kind](), "init": block}
    assert record["task"] == kind and predictor.record_problems(record) == []
    for where in ("init", "init.transfer", "init.split"):
        broken = json.loads(json.dumps(record))
        part = broken
        for name in where.split("."):
            part = part[name]
        part["extra"] = 1
        assert any(f"{where!r} has unknown key(s) 'extra'" in line for line in predictor.record_problems(broken)), where


def test_record_the_kinds_are_the_tasks_of_the_record_tables():
    assert sorted(RECORD_KINDS) == sorted(predictor.RECORD_KEYS)


# ---------------------------------------------------------------------
# pretrain: an et_mim or beit_mim run as a source and as a target (plan-2026-11-19T081214-5c6cc105 Step 6, U4)
# ---------------------------------------------------------------------

#: A beit_mim trunk the CPU trains in seconds: 2 layers (2 relative-bias tables), 32 wide. A beit classifier asks for
#: the same keys, since the weights must fit.
MIM_SMALL = {"num_layers": 2, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64}
MIM_MODELS = {"et_mim": "et_mim/et_tiny", "beit_mim": "beit_mim/beit_tiny"}
#: The train rows of the identity fake of CIFAR-10 (row ``i`` has the mean colour ``4i``).
IDENTITY_ROWS = 60


def mim_world(patch: Any) -> None:
    """The identity fake of CIFAR-10 and the toy tokenizer's usage floor, on ``patch`` (a ``MonkeyPatch``)."""
    from tests.classification_fakes import use_identity_cifar10
    from train.common import pretrain_run

    from .test_pretrain_task import LOW_FLOOR

    use_identity_cifar10(patch, n_train=IDENTITY_ROWS)
    patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)


def mim_request(out: Path, family: str, tokenizer: Optional[str] = None, **changes: Any) -> Dict[str, Any]:
    """One epoch of ``family`` at batch 4: et_mim at ``num_steps`` 2, beit_mim ``MIM_SMALL`` on ``tokenizer``."""
    args = {"num_steps": 2} if family == "et_mim" else {TOKENIZER_KEY: tokenizer, **MIM_SMALL}
    return {"out": str(out), "model": MIM_MODELS[family], "data": {"dataset": "cifar10"}, "epochs": 1, "batch_size": 4,
            "model_args": args, "analysis_args": {"model_analysis": False, "figures": False}, **changes}


def mim_source(family: str, root: Path, tokenizer: Optional[str] = None) -> Path:
    """Train one source of ``family`` (seed 42) on the fake, in a fresh Keras session; return its run directory."""
    with pytest.MonkeyPatch.context() as patch:
        mim_world(patch)
        keras.backend.clear_session()
        train_pretrain(**mim_request(root / "run", family, tokenizer))
    return root / "run"


@pytest.fixture(scope="module")
def mim_tokenizer(tmp_path_factory) -> str:
    from .test_pretrain_task import make_tokenizer

    return make_tokenizer(tmp_path_factory.mktemp("tokenizer") / "tokenizer.keras")


@pytest.fixture(scope="module")
def et_source(world, tmp_path_factory) -> Path:
    return mim_source("et_mim", tmp_path_factory.mktemp("et_source"))


@pytest.fixture(scope="module")
def beit_source(world, mim_tokenizer, tmp_path_factory) -> Path:
    return mim_source("beit_mim", tmp_path_factory.mktemp("beit_source"), mim_tokenizer)


@pytest.fixture(scope="module")
def et_pair(world, et_source, tmp_path_factory):
    """et_mim from a copy of ``et_source`` (the resume test renames the copy away) at a rate that moves no float32
    weight (``STILL``): the fine-tune scores what the loaded weights scored at epoch zero."""
    root = tmp_path_factory.mktemp("et_pair")
    source = root / "source"
    shutil.copytree(et_source, source)
    request = mim_request(root / "run", "et_mim", init=str(source), optim_args=STILL)
    with pytest.MonkeyPatch.context() as patch:
        mim_world(patch)
        start = started(PretrainTrainer, train_pretrain, **request)
    return SimpleNamespace(source=source, out=root / "run", request=request, start=start)


def test_pretrain_et_mim_seeds_et_mim_with_every_weight_and_its_bar_is_strictly_below_the_loaded_val_loss(et_pair):
    out, source = et_pair.out, archive_weights(et_pair.source)
    record, summary = record_of(out), summary_of(out)
    block = record["init"]
    assert (block["task"], block["family"], block["relation"], block["pinned"]) == (
        "pretrain", "et_mim", "same_task", {})
    geometry = ("input_size", "channels", "patch_size")
    assert block["geometry"] == {key: record_of(et_pair.source)[key] for key in geometry}
    # No tokenizer on either side, so the decoder means the same: every weight is copied.
    assert block["transfer"]["copied"] == len(source) == len(et_pair.start)
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, et_pair.start, source)
    assert block["split"]["relation"] == "same_split" and predictor.record_problems(record) == []
    assert (summary["init"], summary["epoch_zero_of"]) == (block, "init") and not any(init_from.bias_of(block).values())
    # Epoch zero is the loaded weights' val_loss on the same fixed selection masks: the source's best (D-012).
    loaded = json.loads((out / "checkpoints" / "fit_state.json").read_text())["initial"]["metrics"]["loss"]
    assert loaded == pytest.approx(summary_of(et_pair.source)["success_value"], rel=1e-5)
    assert summary["epoch_zero"]["val_loss"] == summary["initial_loss_sanity_eval"]["loss"] == loaded
    assert summary["success_threshold"] == min(summary["baseline"]["val_loss"], loaded)
    # Strictly below, in float32: a fine-tune that only holds its loaded weights misses. Evaluation.loaded stays None,
    # so the higher-is-better rule of the other tasks never judges this loss and the summary has no loaded_success_value.
    assert summary["success_value"] == pytest.approx(loaded, rel=1e-6) and summary["success"] is False
    assert "loaded_success_value" not in summary
    assert (summary["initial_loss_ratio"], summary["init_scale_warning"]) == (None, None)
    assert init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    assert not any(note.startswith("initial loss") for note in summary["notes"])
    log = (out / "run.log").read_text()
    assert "Evaluate of the loaded weights before fit: " in log and "Untrained evaluate BEFORE fit" not in log


def test_pretrain_resume_of_an_init_run_with_its_source_renamed_away_opens_no_source_file(
        et_pair, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    shutil.copytree(et_pair.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report
    moved = et_pair.source.with_name("moved_away")
    et_pair.source.rename(moved)
    try:
        mim_world(monkeypatch)
        opened = watch_opens(monkeypatch, str(et_pair.source))
        keras.backend.clear_session()
        train_pretrain(**{**et_pair.request, "out": str(run_dir), "resume_interrupted": True})
    finally:
        monkeypatch.undo()
        moved.rename(et_pair.source)
    assert summary_of(run_dir)["reporting_resumed"] is True and opened == []
    assert summary_of(run_dir)["init"] == record_of(run_dir)["init"] == record_of(et_pair.out)["init"]


def test_pretrain_an_et_mim_run_seeds_the_energy_transformer_classifier_and_its_split_counts_the_rows_read_back(
        world, et_source, tmp_path, monkeypatch):
    from tests.classification_fakes import identities_of
    from train.common.classification_data import load_split_arrays
    from train.common.image_streams import load_image_streams

    mim_world(monkeypatch)
    out = tmp_path / "run"
    start = started(ImageClassificationTrainer, train_image_classification, out=str(out), seed=43,
                    model="energy_transformer/et_tiny", data={"dataset": "cifar10"}, init=str(et_source),
                    model_args={"num_steps": 2}, **QUIET)
    source, record, summary = archive_weights(et_source), record_of(out), summary_of(out)
    block = record["init"]
    assert (block["task"], block["family"], block["relation"]) == ("pretrain", "et_mim", "cross_task")
    assert block["pinned"] == {"patch_size": record_of(et_source)["patch_size"]}  # D-035: read from the source's record
    assert block["geometry"] == {key: record_of(et_source)[key] for key in ("input_size", "channels", "patch_size")}
    decoder = sorted(path for path in source if path.startswith(("decoder_norm/", "decoder_proj/")))
    head = tuple(f"{name}/" for name in registry.MODELS["energy_transformer"].transfer.head)
    assert decoder and sorted(block["transfer"]["dropped"]) == decoder
    assert sorted(block["transfer"]["reset"]) == sorted(path for path in start if path.startswith(head))
    assert block["transfer"]["copied"] == len(source) - len(decoder) and block["transfer"]["resampled"] == []
    check_copied(block, start, source)
    # The split, from the rows each loader reads back: the source trained on every row its selection stream (seed 42)
    # does not hold, and this run selects at seed 43. Both cuts are also the oracle's own permutation.
    streams = load_image_streams("cifar10", image_size=32, batch_size=4, seed=42, augment=False)
    theirs = set(identities_of("cifar10", streams.select_ds).tolist())
    selected = load_split_arrays("cifar10", seed=43).x_select  # bytes / 255, unaugmented
    ours = set(np.rint(selected.mean(axis=(1, 2, 3)) * 255 / 4).astype(int).tolist())
    assert (theirs, ours) == (cut(IDENTITY_ROWS, 42)[1], cut(IDENTITY_ROWS, 43)[1])
    expected = len((set(range(IDENTITY_ROWS)) - theirs) & ours)
    split = block["split"]
    assert expected > 0 and (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", expected, 0)
    assert (summary["init"], summary["selection_biased"], summary["test_biased"]) == (block, True, False)
    assert predictor.record_problems(record) == []


def test_pretrain_a_beit_mim_run_seeds_the_beit_classifier_dropping_its_decoder_on_the_same_rows(
        world, beit_source, tmp_path, monkeypatch):
    mim_world(monkeypatch)
    out = tmp_path / "run"
    start = started(ImageClassificationTrainer, train_image_classification, out=str(out), model="beit/beit_tiny",
                    data={"dataset": "cifar10"}, init=str(beit_source), model_args=MIM_SMALL, **QUIET)
    source, block = archive_weights(beit_source), record_of(out)["init"]
    assert (block["family"], block["relation"], block["pinned"]) == ("beit_mim", "cross_task", {"patch_size": 4})
    decoder = sorted(path for path in source if path.startswith(("decoder_norm/", "decoder_head/")))
    head = tuple(f"{name}/" for name in registry.MODELS["beit"].transfer.head)
    assert decoder and sorted(block["transfer"]["dropped"]) == decoder
    assert sorted(block["transfer"]["reset"]) == sorted(path for path in start if path.startswith(head))
    assert block["transfer"]["copied"] == len(source) - len(decoder) and block["transfer"]["resampled"] == []
    check_copied(block, start, source)
    # Seed 42 with the source's split and cap: the same rows, so nothing is biased.
    assert block["split"]["relation"] == "same_split" and summary_of(out)["selection_biased"] is False


def test_pretrain_a_beit_mim_source_at_32_px_seeds_a_64_px_beit_with_its_patch_pinned_and_every_table_resampled(
        world, beit_source, trees, tmp_path):
    out = tmp_path / "run"
    start = started(ImageClassificationTrainer, train_image_classification,
                    **classify(str(out), trees["ten"], 64, model="beit/beit_tiny", init=str(beit_source),
                               model_args=MIM_SMALL))
    source, block = archive_weights(beit_source), record_of(out)["init"]
    used = json.loads((out / "config.json").read_text())["used"]["model_args"]
    assert block["pinned"] == {"patch_size": 4} and used["patch_size"] == 4  # the patch rule gives 16 at 64 px (D-035)
    tables = sorted(path for path in start if path.endswith("/relative_position_bias_table"))
    rows = [(2 * (side // 4) - 1) ** 2 + 3 for side in (32, 64)]  # a k x k grid: (2k - 1)^2 offsets and 3 class rows
    heads = MIM_SMALL["num_heads"]
    assert len(tables) == MIM_SMALL["num_layers"]
    assert sorted(block["transfer"]["resampled"], key=lambda row: row["path"]) == [
        {"path": path, "rule": "beit_relpos_cubic", "from": [rows[0], heads], "to": [rows[1], heads]}
        for path in tables]
    for path in tables:
        np.testing.assert_allclose(start[path], init_from.beit_relpos_cubic(source[path], start[path].shape), atol=1e-6)
    check_copied(block, start, source)
    assert (out / "run.log").read_text().count(PIN_LINE) == 1  # D-053: the pin line is a notice run.log replays


def test_pretrain_an_et_mim_source_at_32_px_is_refused_by_a_64_px_energy_transformer_before_the_claim(
        world, et_source, trees, tmp_path):
    out = tmp_path / "run"
    keras.backend.clear_session()
    with pytest.raises(ConfigError) as caught:
        train_image_classification(**classify(str(out), trees["ten"], 64, model="energy_transformer/et_tiny",
                                              init=str(et_source), model_args={"num_steps": 2}))
    # The pin keeps patch 4, so the grid is 16x16 at 64 px: refused before the claim, naming both grids (D-055).
    assert "grid of 8x8 patches (32 px at patch_size 4, which init keeps) would become 16x16 at 64 px" in str(caught.value)
    assert "the Energy Transformer position table cannot change its grid" in str(caught.value) and not out.exists()


def test_pretrain_beit_mim_on_another_tokenizer_resets_its_decoder_and_copies_the_trunk(
        world, beit_source, mim_tokenizer, tmp_path, monkeypatch):
    mim_world(monkeypatch)
    # The same codebook moved by 1e-6: the same codes, other bytes, so another sha256, by which D-008 keys the head.
    tokenizer = keras.models.load_model(mim_tokenizer, compile=False)
    tokenizer.quantizer.embeddings.assign(tokenizer.quantizer.embeddings + 1e-6)
    other = tmp_path / "other.keras"
    tokenizer.save(str(other))
    out = tmp_path / "run"
    start = started(PretrainTrainer, train_pretrain, **mim_request(out, "beit_mim", str(other), init=str(beit_source)))
    source, record = archive_weights(beit_source), record_of(out)
    block = record["init"]
    assert record["tokenizer"] == sha(other) != record_of(beit_source)["tokenizer"]
    decoder = sorted(path for path in start if path.startswith(("decoder_norm/", "decoder_head/")))
    assert decoder and sorted(block["transfer"]["reset"]) == sorted(block["transfer"]["dropped"]) == decoder
    assert block["relation"] == "same_task" and block["transfer"]["copied"] == len(start) - len(decoder)
    check_copied(block, start, source)


# ---------------------------------------------------------------------
# language: a gpt2_lm run as a source and as a target (plan-2026-11-24T092348-7806a290 Step 6, D-005, D-040, D-041)
# ---------------------------------------------------------------------

ASCII, GPT2_TOKENIZER = tokenizer_spec("ascii95"), tokenizer_spec("gpt2")
#: The ``data`` of a tiny language run (``_text_world.request``): what a source's ``config.json`` holds under ``data``.
LANGUAGE_DATA = {"dataset": "synthetic", "validation_split": 0.1, "max_samples": 200}
LANGUAGE_TARGET = {key: value for key, value in language_request("unused").items() if key != "out"}
#: ``name: (the source record, the task function, the request, the message pattern)``: sources the language row cannot take.
LANGUAGE_REFUSALS = {
    "another_tokenizer": (
        lambda: {**language_record(), "tokenizer": "f" * 64}, train_language_model_pretraining, LANGUAGE_TARGET,
        rf"gpt2_lm sizes its weights by the tokenizer \(data\.tokenizer, as its digest\); the source trained at {'f' * 64}, "
        rf"this run asks for {ASCII.digest}$"),
    "another_vocabulary": (
        lambda: {**language_record(), "vocab_size": 120}, train_language_model_pretraining, LANGUAGE_TARGET,
        r"gpt2_lm sizes its weights by the model's vocabulary size; the source trained at 120, this run asks for 101$"),
    "another_encoding": (
        lambda: language_record("gpt2"), train_language_model_pretraining, LANGUAGE_TARGET,
        rf"gpt2_lm sizes its weights by the model's vocabulary size and the tokenizer \(data\.tokenizer, as its digest\); the "
        rf"source trained at {GPT2_TOKENIZER.vocab_size} and {GPT2_TOKENIZER.digest}, this run asks for 101 and {ASCII.digest}$"),
    "classifier_into_language": (
        classification_record, train_language_model_pretraining, LANGUAGE_TARGET,
        r"vit \(image_classification\) cannot seed gpt2_lm \(language_model_pretraining\); gpt2_lm takes weights from gpt2_lm "
        r"\(language_model_pretraining\)$"),
    "language_into_classifier": (
        language_record, train_image_classification, CLASSIFIER,
        r"gpt2_lm \(language_model_pretraining\) cannot seed vit \(image_classification\); vit takes weights from vit "
        r"\(image_classification\)$"),
}


@pytest.mark.parametrize("case", sorted(LANGUAGE_REFUSALS))
def test_language_preclaim_a_source_the_row_cannot_take_is_refused_naming_both_sides_and_nothing_is_claimed(
        world, tmp_path, case):
    make, function, request, pattern = LANGUAGE_REFUSALS[case]
    source = write_source(tmp_path / "source", make(), {})
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + pattern):
        function(out=out, init=source, **request)
    assert not out.exists(), "a pre-claim refusal claimed the run directory"


def test_language_preclaim_the_command_line_exits_2_claims_nothing_and_makes_no_device(tmp_path):
    source = write_source(tmp_path / "source", {**language_record(), "tokenizer": "f" * 64}, {})
    out = tmp_path / "run"
    argv = ["language_model_pretraining", f"out={out}", "model=gpt2_lm/tiny", "data.dataset=synthetic",
            "data.tokenizer=ascii95", f"init={source}"]
    done = subprocess.run([sys.executable, "-c", DRIVER, *argv], capture_output=True, text=True, timeout=300)
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen == {"code": 2, "setup_gpu_reached": False, "devices_created": False}, done.stderr[-2000:]
    assert "gpt2_lm sizes its weights by the tokenizer (data.tokenizer, as its digest)" in done.stderr and not out.exists()


def test_language_preclaim_a_source_of_the_same_family_and_tokenizer_is_taken_whatever_its_length(world, tmp_path):
    source = write_source(tmp_path / "source", language_record(seq_len=64), {})  # this run's window is 32
    config = validate(LanguageModelPretrainingConfig, language_request(tmp_path / "run", init=str(source)))
    trainer, _, context = prepared(config, LanguageModelPretrainingTrainer)
    taken = trainer._init_source
    assert (taken.path, taken.relation, dict(taken.pins), context.pinned) == (source, None, {}, {})
    assert taken.sha256 == hashlib.sha256(PLACEHOLDER).hexdigest() and taken.record == language_record(seq_len=64)
    assert not (tmp_path / "run").exists()


def test_language_the_identity_is_the_vocabulary_and_the_tokenizers_digest_and_every_fixed_key_is_compared(world, tmp_path):
    fixed = registry.MODELS["gpt2_lm"].transfer.fixed
    for name in ("ascii95", "gpt2"):  # no file is read and no tensor made: the identity is the tokenizer record's
        spec = tokenizer_spec(name)
        config = validate(LanguageModelPretrainingConfig, language_request(tmp_path / "run", **{"data.tokenizer": name}))
        identity = LanguageModelPretrainingTrainer(config).init_identity(config, None)
        assert identity == {"vocab_size": spec.vocab_size, "tokenizer": spec.digest}
        # a fixed key that the record or the identity lacks is skipped by the comparison without a word
        assert fixed == ("vocab_size", "tokenizer") and set(fixed) <= set(identity) and set(fixed) <= set(language_record(name))
    assert set(fixed) <= set(init_from.GEOMETRY_NAMES)


def test_language_the_head_key_is_the_tokenizer_and_the_block_keeps_the_sources_length_and_vocabulary_only():
    record = language_record()
    assert init_from.HEAD_KEYS["language_model_pretraining"] == ("tokenizer",)
    assert init_from.head_differs("language_model_pretraining", record, dict(record)) is False
    assert init_from.head_differs("language_model_pretraining", record, {**record, "tokenizer": "b" * 64}) is True
    # The block copies the record's length and vocabulary. `tokenizer` is not one of its keys: the pretrain record has a key of
    # that name too (a file's sha256) and its block must not change (D-041).
    assert {"seq_len", "vocab_size"} <= set(init_from.GEOMETRY_KEYS) and "tokenizer" not in init_from.GEOMETRY_KEYS


def held_out(n: int, seed: int, fraction: float) -> set:
    """The ORACLE's split, written here from the rule of D-003: document ``i`` is held out when the first 8 bytes of
    ``sha256(f"{seed}:select:{i}")``, read as a unit float, are below ``fraction``."""
    return {i for i in range(n)
            if int.from_bytes(hashlib.sha256(f"{seed}:select:{i}".encode()).digest()[:8], "big") / 2 ** 64 < fraction}


def fake_documents(n: int, seed: int, fraction: float) -> SimpleNamespace:
    """The ``data`` of a run that read ``n`` documents: the indices of its train and selection documents (the oracle's)."""
    selected = held_out(n, seed, fraction)
    return SimpleNamespace(split=SimpleNamespace(
        train_index=np.array(sorted(set(range(n)) - selected)), select_index=np.array(sorted(selected)), n_test_docs=n // 4))


#: ``name: (the source's seed, its data keys that differ, this run's seed, its data keys that differ)`` on the synthetic bank
#: (200 documents; a null cap is the bank's 600).
LANGUAGE_SPLITS = {
    "same_keys": (42, {}, 42, {}),
    "other_seed": (42, {}, 43, {}),
    "capped_source": (42, {"max_samples": 100}, 43, {}),
    "uncapped_source": (42, {"max_samples": None}, 43, {}),
    "narrower_source_selection": (42, {"validation_split": 0.05}, 42, {}),
    "wider_source_selection": (42, {"validation_split": 0.3}, 42, {}),
    "other_dataset": (42, {"dataset": "imdb_reviews"}, 42, {}),
}


@pytest.mark.parametrize("case", sorted(LANGUAGE_SPLITS))
def test_language_split_counts_the_sources_train_documents_among_the_selection_documents_as_the_oracle(world, tmp_path, case):
    seed, theirs, target_seed, ours = LANGUAGE_SPLITS[case]
    source_data, target_data = {**LANGUAGE_DATA, **theirs}, {**LANGUAGE_DATA, **ours}
    n = target_data["max_samples"] or 600
    fake = fake_documents(n, target_seed, target_data["validation_split"])
    trainer = LanguageModelPretrainingTrainer(validate(LanguageModelPretrainingConfig, language_request(
        tmp_path / "run", seed=target_seed, **{f"data.{key}": value for key, value in ours.items()})))
    split = trainer.init_overlap(a_source({"seed": seed, "data": source_data}, language_record()), trainer.config, fake)
    assert predictor._split_problems(split) == []
    flags = init_from.bias_of({"split": split})
    if case == "other_dataset":
        assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("different_data", 0, 0)
        assert not any(flags.values())
        return
    read = source_data["max_samples"] or 600
    expected = len((set(range(read)) - held_out(read, seed, source_data["validation_split"])) & set(fake.split.select_index.tolist()))
    assert (split["n_select_overlap"], split["n_test_overlap"], split["unit"]) == (expected, 0, "documents")
    assert (expected > 0) == (case not in ("same_keys", "wider_source_selection")), "the control: the oracle counts what it should"
    assert (split["relation"] == "same_split") == (case == "same_keys")
    assert flags == {"selection_biased": expected > 0, "test_biased": False, "overlap_unknown": False}
    if expected:
        assert f"of this run's {len(fake.split.select_index)} selection documents" in split["reason"] and "seed=42" in split["reason"]


#: ``name: (the source's cap, this run's cap, the documents this run read, the documents the source read or None)`` on
#: ``imdb_reviews``: the hook does not look the corpus up, it derives what the source read from the two caps and from what this run read.
IMDB_CAPS = {
    "source_cap_below_ours": (60, 100, 100, 60),
    "source_cap_on_a_whole_corpus": (5000, None, 25000, 5000),
    "both_read_everything": (None, None, 25000, 25000),
    "a_cap_above_the_corpus": (None, 30000, 25000, 25000),
    "the_source_may_have_read_more": (None, 100, 100, None),
    "source_cap_above_ours": (500, 100, 100, None),
}


@pytest.mark.parametrize("case", sorted(IMDB_CAPS))
def test_language_split_on_imdb_derives_how_many_documents_the_source_read_or_says_it_is_unknown(world, tmp_path, case):
    cap, ours, read, n_source = IMDB_CAPS[case]
    fake = fake_documents(read, 43, 0.1)
    trainer = LanguageModelPretrainingTrainer(validate(LanguageModelPretrainingConfig, language_request(
        tmp_path / "run", seed=43, **{"data.dataset": "imdb_reviews", "data.max_samples": ours})))
    source = a_source({"seed": 42, "data": {**LANGUAGE_DATA, "dataset": "imdb_reviews", "max_samples": cap}}, language_record())
    split = trainer.init_overlap(source, trainer.config, fake)
    assert predictor._split_problems(split) == []
    if n_source is None:
        assert (split["relation"], split["n_select_overlap"]) == ("unknown", None) and "data.max_samples" in split["reason"]
        assert init_from.bias_of({"split": split})["overlap_unknown"] is True
        return
    expected = len((set(range(n_source)) - held_out(n_source, 42, 0.1)) & set(fake.split.select_index.tolist()))
    assert expected > 0 and (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("overlap", expected, 0)


def test_a_capped_labelled_source_overlaps_this_run_by_the_documents_it_really_shares():
    # review pass 1, W1: the count is by document index, so document i has to be the same text and label at every cap (it was not:
    # a source capped at 200 and a run capped at 300 on seed 7 counted 24 of 39 selection documents, 15 of them really shared)
    from train.api.text_args import documents_overlap
    from train.common import text_data

    source, target = (text_data.synthetic_documents(text_data.BANK_SEED, n, text_data.CLASSIFICATION) for n in (200, 300))
    fake = fake_documents(300, 7, 0.1)
    ours = SimpleNamespace(seed=7, data=SimpleNamespace(dataset="synthetic", validation_split=0.1, max_samples=300))
    theirs = a_source({"seed": 42, "data": {**LANGUAGE_DATA, "max_samples": 200}}, text_classification_record())
    split = documents_overlap(theirs, ours, fake, "text_classification")
    trained = set(range(200)) - held_out(200, 42, 0.1)
    shared = [i for i in fake.split.select_index.tolist() if i in trained and source[0][i] == target[0][i] and source[1][i] == target[1][i]]
    assert split["relation"] == "overlap" and split["n_select_overlap"] == len(shared) > 0, (split["n_select_overlap"], len(shared))


def test_a_source_that_read_other_documents_under_the_same_keys_is_unknown_and_not_counted(tmp_path):
    # review pass 1, N6: the count is by document index, and the same keys over another corpus are other documents
    from train.api.text_args import documents_overlap

    (tmp_path / "checkpoints").mkdir()
    fake = fake_documents(200, 42, 0.1)
    fake.fingerprint = "b" * 64
    ours = SimpleNamespace(seed=42, data=SimpleNamespace(dataset="synthetic", validation_split=0.1, max_samples=200))
    for earlier, relation in (("a" * 64, "unknown"), ("b" * 64, "same_split")):
        (tmp_path / "checkpoints" / "fit_state.json").write_text(json.dumps({"data_fingerprint": earlier}))
        theirs = a_source({"seed": 42, "data": LANGUAGE_DATA}, text_classification_record(), tmp_path)
        split = documents_overlap(theirs, ours, fake, "text_classification")
        assert split["relation"] == relation and ("data_fingerprint" in split["reason"]) == (relation == "unknown"), split


def language_run(out: Path, **changes: Any) -> Dict[str, np.ndarray]:
    """A tiny ``gpt2_lm`` run in a fresh Keras session; return the weights its ``fit`` started from."""
    return started(LanguageModelPretrainingTrainer, train_language_model_pretraining, **language_request(out, **changes))


@pytest.fixture(scope="module")
def language_source(world, tmp_path_factory):
    """gpt2_lm on the synthetic bank (64 wide, 2 blocks, windows of 32, seed 42), 3 epochs."""
    out = tmp_path_factory.mktemp("language_source") / "run"
    language_run(out)
    return out


@pytest.fixture(scope="module")
def language_pair(world, language_source, tmp_path_factory):
    """gpt2_lm from a copy of ``language_source`` (the resume test renames the copy away) at a rate that moves no float32
    weight (``STILL``): the fine-tune scores what the loaded weights scored at epoch zero."""
    root = tmp_path_factory.mktemp("language_pair")
    source = root / "source"
    shutil.copytree(language_source, source)
    request = language_request(root / "run", init=str(source), optim_args=STILL, epochs=1)
    return SimpleNamespace(source=source, out=root / "run", request=request, start=started(
        LanguageModelPretrainingTrainer, train_language_model_pretraining, **request))


@pytest.fixture(scope="module")
def language_finetune(world, language_source, tmp_path_factory):
    """gpt2_lm from ``language_source`` at the tiny recipe's rate and ANOTHER seed: a split of other documents."""
    out = tmp_path_factory.mktemp("language_finetune") / "run"
    return SimpleNamespace(out=out, start=language_run(out, init=str(language_source), seed=43))


def test_language_wire_a_gpt2_lm_run_seeds_a_gpt2_lm_run_with_every_weight_and_its_bar_is_strictly_below_the_loaded_val_loss(
        language_pair):
    out, source = language_pair.out, archive_weights(language_pair.source)
    record, summary = record_of(out), summary_of(out)
    block, theirs = record["init"], record_of(language_pair.source)
    assert (block["task"], block["family"], block["relation"], block["pinned"]) == (
        "language_model_pretraining", "gpt2_lm", "same_task", {})
    assert block["geometry"] == {key: theirs[key] for key in ("seq_len", "vocab_size")}
    # The same tokenizer on both sides, so every weight means the same: the family has no head to reset.
    assert block["transfer"]["copied"] == len(source) == len(language_pair.start) > 0
    assert all(not block["transfer"][key] for key in ("reset", "grown", "dropped", "resampled"))
    check_copied(block, language_pair.start, source)
    assert (block["split"]["relation"], block["split"]["unit"]) == ("same_split", "documents")
    assert predictor.record_problems(record) == []
    assert (summary["init"], summary["epoch_zero_of"]) == (block, "init") and not any(init_from.bias_of(block).values())
    # Epoch zero is the loaded weights' val_loss on the same fixed selection windows: the source's best (D-012 of plan 5c6cc105).
    loaded = json.loads((out / "checkpoints" / "fit_state.json").read_text())["initial"]["metrics"]["loss"]
    assert loaded == pytest.approx(summary_of(language_pair.source)["success_value"], rel=1e-5)
    assert summary["epoch_zero"] == {"val_loss": loaded, "binds": True}  # the loaded weights beat the unigram: they set the bar
    assert summary["initial_loss_sanity_eval"]["loss"] == loaded and summary["success_threshold"] == loaded
    assert loaded < summary["baseline"]["val_loss"]
    logger.info(f"LM-INIT-STILL copied {block['transfer']['copied']} of {len(language_pair.start)} weights; epoch zero (the loaded "
                f"weights) {loaded:.9f}, unigram {summary['baseline']['val_loss']:.9f}, bar {summary['success_threshold']:.9f}, "
                f"best of the fine-tune {summary['success_value']:.9f}, success {summary['success']}")
    # Strictly below, in float32: a fine-tune that only holds its loaded weights misses. `Evaluation.loaded` stays None, so the
    # higher-is-better rule of the other tasks never judges this loss and the summary has no loaded_success_value.
    assert summary["success_value"] == pytest.approx(loaded, rel=1e-6) and summary["success"] is False
    assert "loaded_success_value" not in summary
    assert (summary["initial_loss_ratio"], summary["init_scale_warning"]) == (None, None)
    assert init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    assert not any(note.startswith("initial loss") for note in summary["notes"])
    assert any("the loaded weights' (under `init`)" in note for note in summary["notes"])
    log = (out / "run.log").read_text()
    assert "Evaluate of the loaded weights before fit: " in log and "Untrained evaluate BEFORE fit" not in log


def test_language_a_fine_tune_that_learns_meets_the_bar_and_the_split_of_other_documents_is_counted_as_the_oracle(
        language_finetune):
    out = language_finetune.out
    record, summary = record_of(out), summary_of(out)
    loaded = summary["epoch_zero"]["val_loss"]
    assert summary["success_threshold"] == min(summary["baseline"]["val_loss"], loaded)
    # Met, though the higher-is-better rule would have judged a LOWER loss a miss: the bar is strictly below, beyond the tie band.
    assert summary["success"] is True and np.float32(summary["success_value"]) < np.float32(loaded)
    logger.info(f"LM-INIT-LEARNS epoch zero (the loaded weights) {loaded:.6f}, unigram {summary['baseline']['val_loss']:.6f}, bar "
                f"{summary['success_threshold']:.6f}, best of the fine-tune {summary['success_value']:.6f}")
    n = LANGUAGE_DATA["max_samples"]
    expected = len((set(range(n)) - held_out(n, 42, 0.1)) & held_out(n, 43, 0.1))  # the source trained at seed 42, this run selects at 43
    split = record["init"]["split"]
    assert expected > 0 and (split["relation"], split["unit"], split["n_select_overlap"], split["n_test_overlap"]) == (
        "overlap", "documents", expected, 0)
    assert (summary["selection_biased"], summary["test_biased"]) == (True, False) and predictor.record_problems(record) == []
    assert record["init"]["transfer"]["copied"] == len(language_finetune.start)


def test_language_resume_of_an_init_run_with_its_source_renamed_away_opens_no_source_file(
        language_pair, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    shutil.copytree(language_pair.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report
    moved = language_pair.source.with_name("moved_away")
    language_pair.source.rename(moved)
    try:
        opened = watch_opens(monkeypatch, str(language_pair.source))
        keras.backend.clear_session()
        train_language_model_pretraining(**{**language_pair.request, "out": str(run_dir), "resume_interrupted": True})
    finally:
        monkeypatch.undo()
        moved.rename(language_pair.source)
    assert summary_of(run_dir)["reporting_resumed"] is True and opened == []
    assert summary_of(run_dir)["init"] == record_of(run_dir)["init"] == record_of(language_pair.out)["init"]
