"""The batch of each ImageNet and Imagenette recipe is one that was measured to fit a 24 GB card.

Step 17 of plan-2026-10-31T180033-2faf890a ran 20 real train steps through ``python -m train
image_classification`` on GPU 0 (an RTX 4090, 24,564 MiB) with XLA as Keras sets it, for the
smallest variant of each of the six families on ImageNet and on Imagenette, for the default and the
largest cliffordnet variants, and for ``resnet50`` (the legacy ImageNet trainer's default variant).
Every row is in ``memory_fit_evidence.json`` (a copy of the untracked
``plans/.../scratch_step17/results.tsv``; this file does not re-measure). It pins:

* the evidence itself is consistent with the rule it was judged by: a row ``fits`` exactly when it
  completed 20 steps, did not run out of memory, had no allocator garbage collection, kept the
  allocator peak at or under 20,000 MiB and the device at or under 23,000 MiB;
* every recipe the step sized (the six families' smallest variants, and ``cliffordnet/lite``) has a
  batch that is a row of the evidence that fits, on both datasets, and the next batch up is a row that
  does not fit (or the recipe's own batch is the first one tried);
* the two variants known NOT to fit the recipe's batch (``resnet50`` on ImageNet at 256,
  ``cliffordnet/cn_64_5`` at 8) are named here with the batch that fits, so a recipe change that
  covers them has to change this list too;
* the README's memory table is the evidence, row by row, written by :func:`table_row`.

NOT checked: that a batch which fit 20 steps keeps fitting for 200 epochs (a longer run was not made),
that a larger variant than the ones measured fits, and the evidence file against the machine.
"""

import json
import re
from pathlib import Path

import pytest

from train.api import recipes, registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = json.loads((HERE / "memory_fit_evidence.json").read_text())
ROWS = EVIDENCE["rows"]
README = (ROOT / "src" / "train" / "README.md").read_text()
BEGIN, END = "<!-- memory:begin -->", "<!-- memory:end -->"
RULE = EVIDENCE["rule"]

#: (family, variant) the step sized: the recipe batch must fit on both datasets at 224 px.
SIZED = [
    ("vit", "vit_pico"), ("resnet", "resnet18"), ("convnext_v1", "tiny"), ("convnext_v2", "atto"),
    ("cliffordnet", "nano"), ("cliffordnet", "lite"), ("cliffordnet_hier", "hier_345"),
]
#: (family, variant, dataset) -> the batch that fits, where it is BELOW the recipe's. The recipe is per
#: family, and neither variant can take its own batch without a mechanism the recipes do not have
#: (``per_variant`` applies to every dataset and cn_64_5's CIFAR recipe is the paper's).
KNOWN_TO_NEED_LESS = {
    ("resnet", "resnet50", "imagenet"): 128,
    ("cliffordnet", "cn_64_5", "imagenette"): 2,
}
DATASETS = ("imagenet", "imagenette")


def rule_fits(row) -> bool:
    return bool(
        row["completed"] and not row["oom"] and row["allocator_gc_lines"] == 0
        and row["steps"] >= RULE["min_steps"]
        and row["peak_mib"] is not None and row["peak_mib"] <= RULE["peak_limit_mib"]
        and row["device_mib"] is not None and row["device_mib"] <= RULE["device_limit_mib"]
    )


def result_of(row) -> str:
    if row["fits"]:
        return "fits"
    if row["oom"] and not row["completed"]:
        asked = row["oom_request_gib"]
        return f"OOM, asks {asked:.1f} GiB" if asked is not None else "OOM"
    return "completed, allocator garbage collection, no margin"


def table_row(row) -> str:
    """The README line of one evidence row; the README table is these lines, in this order."""
    done = row["completed"]
    cells = [
        f"`{row['model']}`", row["dataset"], str(row["image"]), str(row["batch"]),
        f"{row['peak_mib']:,.0f}" if done else "-",
        f"{row['device_mib']:,}",
        str(row["steps"]) if done else "-",
        f"{row['steady_s_per_step']:.3f}" if done else "-",
        f"{row['images_per_s']:,.1f}" if done else "-",
        result_of(row),
    ]
    return "| " + " | ".join(cells) + " |"


def readme_rows():
    text = README
    assert text.count(BEGIN) == 1 and text.count(END) == 1, "the README needs one memory table"
    block = text[text.index(BEGIN) + len(BEGIN):text.index(END)]
    lines = [line for line in block.splitlines() if line.startswith("|")]
    return lines[2:]  # header and separator


def fitting(model: str, dataset: str, image: int = 224):
    return sorted(
        (r for r in ROWS if (r["model"], r["dataset"], r["image"]) == (model, dataset, image) and r["fits"]),
        key=lambda r: -r["batch"],
    )


def rows_of(model: str, dataset: str, image: int = 224):
    return sorted(
        (r for r in ROWS if (r["model"], r["dataset"], r["image"]) == (model, dataset, image)),
        key=lambda r: -r["batch"],
    )


@pytest.fixture(scope="module")
def builtin():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield registry.MODELS


def recipe_batch(builtin, family, variant, dataset) -> int:
    return builtin[family].recipe(dataset, variant)["batch_size"]


def test_the_evidence_is_judged_by_the_stated_rule():
    assert len(ROWS) == 48
    for row in ROWS:
        assert row["fits"] == rule_fits(row), row
    assert RULE == {"peak_limit_mib": 20000, "device_limit_mib": 23000, "no_allocator_gc_or_oom_line": True,
                    "min_steps": 20}


def test_an_oom_row_has_no_steps_and_a_completed_row_has_its_peak():
    for row in ROWS:
        if row["oom"]:
            assert not row["completed"] and row["peak_mib"] is None, row
        if row["completed"]:
            assert row["steps"] >= 20 and row["peak_mib"] is not None and row["images_per_s"], row


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("family, variant", SIZED, ids=lambda v: str(v))
def test_the_recipe_batch_is_a_row_that_fits_and_the_next_one_up_is_not(builtin, family, variant, dataset):
    model = f"{family}/{variant}"
    batch = recipe_batch(builtin, family, variant, dataset)
    fits = fitting(model, dataset)
    assert fits, f"no fitting row for {model} on {dataset}"
    assert fits[0]["batch"] == batch, (
        f"{model} on {dataset}: the recipe's batch is {batch}, the largest batch that fit is {fits[0]['batch']}")
    larger = [r for r in rows_of(model, dataset) if r["batch"] > batch]
    assert all(not r["fits"] for r in larger)
    # The ladder started at the recipe batch it replaced: a recipe that was cut shows its failed rows.
    assert rows_of(model, dataset)[0]["batch"] >= batch


@pytest.mark.parametrize("key, fit_batch", sorted(KNOWN_TO_NEED_LESS.items()))
def test_the_two_variants_that_need_less_than_the_recipe_say_so(builtin, key, fit_batch):
    family, variant, dataset = key
    model = f"{family}/{variant}"
    fits = fitting(model, dataset)
    assert fits and fits[0]["batch"] == fit_batch, fits
    recipe = recipe_batch(builtin, family, variant, dataset)
    assert recipe > fit_batch, (
        f"{model} on {dataset}: the recipe's batch {recipe} now fits; drop it from KNOWN_TO_NEED_LESS")
    # The recipe's own batch was tried (or a larger one) and did not fit, as the README says.
    tried = [r for r in rows_of(model, dataset) if r["batch"] >= recipe]
    assert tried and all(not r["fits"] for r in tried), tried


def test_resnet50_on_imagenette_fits_at_the_recipe_batch(builtin):
    """The Imagenette recipe is already 128 for resnet, so ``resnet50`` fits there."""
    assert recipe_batch(builtin, "resnet", "resnet50", "imagenette") == 128
    assert fitting("resnet/resnet50", "imagenette")[0]["batch"] == 128


def test_the_legacy_readmes_128_px_form_fits_for_nano_and_lite_needs_16():
    assert fitting("cliffordnet/nano", "imagenette", 128)[0]["batch"] == 32
    assert fitting("cliffordnet/lite", "imagenette", 128)[0]["batch"] == 16


def test_the_batch_of_each_cliffordnet_family_is_the_smaller_of_its_variants(builtin):
    assert recipe_batch(builtin, "cliffordnet", "nano", "imagenette") == recipe_batch(
        builtin, "cliffordnet", "lite", "imagenette") == 8
    assert recipe_batch(builtin, "cliffordnet_hier", "hier_345", "imagenette") == 16


@pytest.mark.parametrize("name", ["cifar10", "cifar100"])
def test_the_cifar_recipes_of_cliffordnet_keep_the_paper_batch(builtin, name):
    for family, variant in (("cliffordnet", "nano"), ("cliffordnet", "cn_64_5"), ("cliffordnet_hier", "hier_345")):
        assert recipe_batch(builtin, family, variant, name) == 128


def test_the_readme_memory_table_is_the_evidence_row_by_row():
    assert readme_rows() == [table_row(row) for row in ROWS]


def test_the_readme_memory_table_reads_the_rule_and_the_card_from_the_evidence():
    text = README
    for needle in (
        f"{RULE['peak_limit_mib']:,} MiB", f"{RULE['device_limit_mib']:,} MiB", f"{EVIDENCE['card_mib']:,} MiB",
    ):
        assert needle in text, needle


def test_a_row_that_drifted_would_be_found():
    """The control: the comparison above is sensitive to one changed cell."""
    row = dict(ROWS[0])
    row["peak_mib"] = row["peak_mib"] + 1.0
    assert table_row(row) != table_row(ROWS[0])
    assert re.fullmatch(r"\| .* \|", table_row(ROWS[0]))
