"""The help and the README state what a full ImageNet run costs, from the measured numbers.

Step 18 of plan-2026-10-31T180033-2faf890a. ImageNet's train images sit on a spinning disk; step 17
measured 96.3, 97.5 and 106.9 images per second for the first read of files in training runs
(``memory_fit_evidence.json``), so one epoch is 3.3 to 3.7 hours and the 50-epoch recipe takes about a
week. This file pins:

* ``python -m train image_classification help`` says that ImageNet is read from a spinning disk, the
  measured images per second (the lowest and the highest cold row of the evidence, rounded), the
  length of the 50-epoch recipe, that a killed epoch is lost, and where the exact commands are. It
  names no key that does not exist; since step 19 it also says that ``checkpoint_args.backup_every_steps``
  bounds the loss (in the ``data.dataset`` help and in ``resume_interrupted``'s);
* the README's table of days per recipe length is the arithmetic of the epoch size over those two
  rates, and the sentence that follows it says the right thing about the 50- and 100-epoch recipes;
* the README's exact commands are valid requests: each line is parsed with the CLI's own parser and
  validated against the real ``ImageClassificationConfig`` (every key exists, every value has its
  type, the model is a registered family and variant), the 50- and 100-epoch lines ask for those
  epochs, and each is marked NOT RUN;
* the new sections stand OUTSIDE the knob region of the README, so the knob-document guard
  (``test_the_knob_docs_name_real_keys.py``, whose duration rule covers that region only) neither
  checks nor forbids the durations stated here; the tables above are the check for them.

NOT checked: that the commands train (they were not run), the wording of the prose, and the rates
against the machine (the evidence file is a copy of one measurement).
"""

import json
import re
import shlex
from pathlib import Path

import pytest

from train.api import recipes, registry
from train.api.cli import coerce_mapping, parse_dotlist
from train.api.config import validate
from train.api.tasks.image_classification import ImageClassificationConfig

from .test_cli_help_allocates_nothing import run_train

HERE = Path(__file__).resolve().parent
README = (HERE.parents[2] / "src" / "train" / "README.md").read_text()
EVIDENCE = json.loads((HERE / "memory_fit_evidence.json").read_text())
IMAGENET_TRAIN_IMAGES = 1_271_167
#: The first read of files in a training run: ImageNet rows that are limited by the disk, not by the GPU.
COLD_MODELS = ("vit/vit_pico", "resnet/resnet18", "convnext_v1/tiny")
KNOB_BEGIN, KNOB_END = "<!-- knobs:begin -->", "<!-- knobs:end -->"


def cold_rates():
    rows = [
        r for r in EVIDENCE["rows"]
        if r["dataset"] == "imagenet" and r["model"] in COLD_MODELS and r["image"] == 224
        and r["fits"] and r["batch"] == {"vit/vit_pico": 128, "resnet/resnet18": 256, "convnext_v1/tiny": 64}[r["model"]]
    ]
    assert len(rows) == len(COLD_MODELS), rows
    return sorted(r["images_per_s"] for r in rows)


@pytest.fixture(scope="module")
def help_text(tmp_path_factory):
    result = run_train(["image_classification", "help"], tmp_path_factory.mktemp("cwd"), importtime=False)
    assert result.returncode == 0, result.stderr[-2000:]
    return " ".join(result.stdout.split())


def test_the_help_states_the_disk_the_rate_the_week_and_the_lost_epoch(help_text):
    low, high = cold_rates()[0], cold_rates()[-1]
    for phrase in (
        "spinning disk",
        f"{low:.0f} to {high:.0f} images per second",
        "a disk-read bound of 3 to 4 hours for one epoch of its 1,271,167 images",
        "the 50-epoch recipe takes about a week",
        "a 150-step probe projected 4.6 hours of training per epoch (9.6 days for 50 epochs, 19.1 for 100)",
        "the one measured epoch took about 6.7 hours of training on the RTX 4090 with CPU test jobs sharing the host",
        "A killed epoch is lost",
        "src/train/README.md",
        "which were not run",
    ):
        assert phrase in help_text, phrase


def test_the_measured_epoch_hours_in_the_help_are_the_readmes():
    """Step 12.2 of plan 98b866fb: the help said only the disk-read bound (3 to 4 hours), while the README measured a slower
    epoch. The help's probe and measured-epoch figures are the README's own sentences."""
    readme = " ".join(README.split())
    assert "puts one epoch at 4.6 hours of training" in readme
    assert "at 77 images per second the table above is 9.6 days for 50 epochs and 19.1 for 100" in readme
    assert "about 6.7 hours of training for the epoch" in readme
    assert "with CPU test jobs" in readme


def test_the_readme_states_the_measured_schedule_of_the_one_epoch():
    """review-iter-2-pass2 WARNING 3: the README called the epoch's rate a constant 0.0099; ``run.log`` of the run shows a
    ``CosineDecay`` from 0.1 with ``decay_steps`` 4965 and ``alpha`` 0.01, and 0.0099 is its value at step 4,000."""
    import math

    readme = " ".join(README.split())
    assert "constant learning rate of 0.0099" not in readme and "a 1-epoch cosine" not in readme
    assert "a cosine decay from 0.1 to 0.001 over the 4,965 steps with no warmup" in readme
    assert "The 0.0099 in `training_log.csv`'s `lr` column" in readme and "value at step 4,000" in readme
    rate = lambda step: 0.1 * ((1 - 0.01) * 0.5 * (1 + math.cos(math.pi * step / 4965)) + 0.01)
    assert f"{rate(4000):.4f}" == "0.0099" and abs(rate(4000) - 0.009944482706487179) < 1e-8  # results_summary.json lr_first_epoch
    assert (round(rate(0), 6), round(rate(4965), 6)) == (0.1, 0.001)


def test_the_resume_help_says_a_killed_epoch_is_lost(help_text):
    assert "a killed epoch is lost and a half-done epoch is trained again from its first batch" in help_text


def test_both_helps_name_the_key_that_bounds_the_lost_epoch(help_text):
    assert ("A killed epoch is lost; checkpoint_args.backup_every_steps bounds the loss to that many "
            "batches of training") in help_text
    assert ("checkpoint_args.backup_every_steps bounds the loss to that many batches of training, and the "
            "epoch is then continued from the last backup: only the batches that backup had not taken run, "
            "from a fresh pass of the data, so some examples of that epoch repeat and some are missed") in help_text
    assert "epoch is then trained again from its first batch on the weights of the last backup" not in help_text, (
        "D-038's sentence, false since the review of iteration 2 (D-063)")
    assert help_text.count("checkpoint_args.backup_every_steps") >= 3, "the key's own help is the third"


def test_the_help_names_no_key_that_does_not_exist(help_text):
    import dataclasses

    groups = {f.name for f in dataclasses.fields(ImageClassificationConfig)}
    for group, field in re.findall(r"\b([a-z]+_args)\.([a-z_0-9]+)", help_text):
        assert group in groups, group
        names = {f.name for f in dataclasses.fields(
            {f.name: f for f in dataclasses.fields(ImageClassificationConfig)}[group].default_factory)}
        assert field in names, f"{group}.{field}"


def test_the_epoch_size_and_the_rates_in_the_help_are_the_readmes():
    assert IMAGENET_TRAIN_IMAGES == 1_271_167
    low, high = cold_rates()[0], cold_rates()[-1]
    assert (round(low), round(high)) == (96, 107)
    hours = [IMAGENET_TRAIN_IMAGES / rate / 3600 for rate in (low, high)]
    assert 3 <= min(hours) and max(hours) < 4, hours  # "3 to 4 hours"
    assert f"{max(hours):.1f} to {min(hours):.1f} hours" in README


def time_rows():
    low, high = cold_rates()[0], cold_rates()[-1]
    return [
        f"| {epochs} | {epochs * IMAGENET_TRAIN_IMAGES / low / 86400:.1f} | "
        f"{epochs * IMAGENET_TRAIN_IMAGES / high / 86400:.1f} |"
        for epochs in (50, 100, 200)
    ]


def region(begin, end):
    assert README.count(begin) == 1 and README.count(end) == 1, (begin, end)
    return README[README.index(begin) + len(begin):README.index(end)]


def test_the_readme_days_table_is_the_arithmetic_of_the_measured_rates():
    low, high = cold_rates()[0], cold_rates()[-1]
    lines = [l for l in region("<!-- imagenet-time:begin -->", "<!-- imagenet-time:end -->").splitlines() if l.startswith("|")]
    assert lines[0] == f"| Epochs | Days at {low} images per second | Days at {high} images per second |"
    assert lines[2:] == time_rows()
    # The sentence that reads it: 50 epochs about a week, 100 about two.
    days = {epochs: (epochs * IMAGENET_TRAIN_IMAGES / high / 86400, epochs * IMAGENET_TRAIN_IMAGES / low / 86400)
            for epochs in (50, 100)}
    assert 6 <= days[50][0] and days[50][1] <= 8
    assert 13 <= days[100][0] and days[100][1] <= 16
    assert "So the 50-epoch ImageNet recipe (`vit`) takes about a week, and the 100-epoch ones (`resnet`, `convnext`) about two" in " ".join(README.split())


def commands():
    block = region("<!-- imagenet-commands:begin -->", "<!-- imagenet-commands:end -->")
    found, pending, mark = [], "", None
    for line in block.splitlines():
        if line.startswith("#"):
            mark = line
            continue
        if line.startswith("```") or not line.strip():
            continue
        pending += line.rstrip("\\ ").strip() + " "
        if not line.rstrip().endswith("\\"):
            found.append((mark, pending.strip()))
            pending = ""
    assert not pending
    return found


def arguments(command):
    words = shlex.split(command)
    start = words.index("image_classification")
    assert words[start - 3:start] == [".venv/bin/python", "-m", "train"], words
    return words[start + 1:]


def test_the_readme_has_the_three_commands_each_marked_not_run():
    found = commands()
    assert len(found) == 3
    for mark, _ in found:
        assert mark.startswith("# NOT RUN."), mark


@pytest.fixture(scope="module")
def builtin():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield registry.MODELS


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_command_is_a_valid_request(builtin, index):
    _, command = commands()[index]
    config_path, raw = parse_dotlist(arguments(command))
    assert config_path is None
    mapping = coerce_mapping(ImageClassificationConfig, raw)
    config = validate(ImageClassificationConfig, mapping)
    family, variant = config.model.split("/")
    assert variant in builtin[family].load_variants()
    assert config.data.dataset == "imagenet" and config.out.startswith("results/")


def test_the_50_and_100_epoch_commands_ask_for_those_epochs(builtin):
    requested = []
    for _, command in commands():
        _, raw = parse_dotlist(arguments(command))
        requested.append(coerce_mapping(ImageClassificationConfig, raw))
    assert [m["epochs"] for m in requested] == [50, 100, 100]
    # Each asks for a backup every 500 batches (a resume must repeat the request it started with).
    assert [m["checkpoint_args"] for m in requested] == [{"backup_every_steps": 500}] * 3
    assert requested[2].get("resume_interrupted") is True and requested[1].get("resume_interrupted") is None
    # The resume line is the 100-epoch line plus the one key.
    assert {k: v for k, v in requested[2].items() if k != "resume_interrupted"} == requested[1]
    # The 50 and 100 epochs are the recipes' own (vit 50, resnet 100), so the request leaves the rest to the recipe.
    assert builtin["vit"].recipe("imagenet", "vit_pico")["epochs"] == 50
    assert builtin["resnet"].recipe("imagenet", "resnet18")["epochs"] == 100


def test_a_broken_command_is_found():
    """The control: a misspelt key in a command makes the same check fail."""
    _, command = commands()[0]
    broken = command.replace("data.dataset=imagenet", "data.datasett=imagenet")
    _, raw = parse_dotlist(arguments(broken))
    with pytest.raises(Exception):
        validate(ImageClassificationConfig, coerce_mapping(ImageClassificationConfig, raw))


def test_the_new_sections_stand_outside_the_knob_region():
    """The knob guard's duration rule covers that region only; the tables above check these."""
    begin, end = README.index(KNOB_BEGIN), README.index(KNOB_END)
    for marker in ("<!-- memory:begin -->", "<!-- imagenet-time:begin -->", "<!-- imagenet-commands:begin -->",
                   "<!-- backup-cost:begin -->"):
        assert not begin < README.index(marker) < end, marker
