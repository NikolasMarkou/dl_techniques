"""The recipe batch of ``resnet_seg`` and ``convnext_seg`` is the one D-036's procedure measured (plan 98b866fb step 2, D-012).

The two families have no legacy trainer, so their batch is not copied from one: it is the batch at which 20 real train steps of the
family's default variant on Oxford-IIIT Pet at 128 px completed on the 24 GB card, judged by the rule of D-036 (read from
``memory_fit_evidence.json``, so there is one copy of it). The rows are in ``backbone_seg_memory_evidence.json`` (a copy of the untracked
``scratch_step2/logs``; this file does not re-measure). It pins:

* every row is judged by that rule: a row ``fits`` exactly when it completed the steps, ran out of nothing, had no allocator collection and
  kept the peak and the device under the limits;
* the recipe's batch (16, for both families) fits for the default variant of each family and for the other variants the ladder tried,
  with the exceptions the ladder found written out (``convnext_seg/base`` completes at 16 only with an allocator collection and fits at 8;
  ``convnext_seg/xlarge`` ran out of memory at every batch down to 1 on the card as it was shared);
* the card was shared: the device figure includes another session's two ``llama-server`` processes (15.9 GB), recorded per run, so a
  batch the rule refuses here might fit on an idle card, and the evaluation batch is the training batch;
* the README says it.

NOT checked: that a batch which fit 20 steps fits 30 epochs (the real Pet runs of the step completed), and the evidence against the machine.
"""

import json
from pathlib import Path

import pytest

from train.api import recipes, registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = json.loads((HERE / "backbone_seg_memory_evidence.json").read_text())
RULE = json.loads((HERE / "memory_fit_evidence.json").read_text())["rule"]
ROWS = EVIDENCE["rows"]


def rule_fits(row) -> bool:
    return bool(
        row["completed"] and not row["oom"] and row["allocator_gc_lines"] == 0
        and row["steps"] >= RULE["min_steps"]
        and row["peak_mib"] is not None and row["peak_mib"] <= RULE["peak_limit_mib"]
        and row["device_mib"] <= RULE["device_limit_mib"]
    )


def by(model, batch):
    (row,) = [r for r in ROWS if r["model"] == model and r["batch"] == batch]
    return row


@pytest.fixture(scope="module")
def recipe_of():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        yield lambda family, variant: registry.MODELS[family].recipe("oxford_pet", variant)


def test_every_row_of_the_evidence_is_judged_by_the_rule_the_step_17_rows_were():
    assert [row["fits"] for row in ROWS] == [rule_fits(row) for row in ROWS]


def test_the_card_was_shared_and_the_evidence_says_how_much_was_taken_before_each_run():
    assert EVIDENCE["idle_device_mib"] == 15948 and EVIDENCE["card_mib"] == 24564
    assert all(14000 < row["device_before_mib"] < 17000 for row in ROWS), "another session held 15 to 16 GB throughout"


@pytest.mark.parametrize("model", ["resnet_seg/resnet18", "resnet_seg/resnet50", "resnet_seg/resnet152",
                                   "convnext_seg/cifar10", "convnext_seg/tiny"])
def test_these_variants_fit_at_the_recipe_batch(model, recipe_of):
    family, variant = model.split("/")
    assert recipe_of(family, variant)["batch_size"] == 16
    assert by(model, 16)["fits"] and by(model, 16)["steps"] >= 20


def test_the_exceptions_are_measured_and_written_out(recipe_of):
    base16, base8 = by("convnext_seg/base", 16), by("convnext_seg/base", 8)
    assert base16["completed"] and base16["allocator_gc_lines"] >= 1 and not base16["fits"], "completes, with a collection: the rule refuses it"
    assert base8["fits"]
    ladder = {row["batch"]: row for row in ROWS if row["model"] == "convnext_seg/xlarge"}
    assert sorted(ladder) == [1, 2, 4, 8, 16]
    assert all(row["oom"] and not row["fits"] and not row["completed"] for row in ladder.values())
    assert recipe_of("convnext_seg", "xlarge")["batch_size"] == 16, "the batch is a family default; the exception is documented, not a per-variant overlay"


def test_the_recipe_batch_is_the_largest_tried_for_the_default_variants_and_the_evaluation_batch_is_the_training_batch(recipe_of):
    for family, variant in (("resnet_seg", "resnet18"), ("convnext_seg", "tiny")):
        recipe = recipe_of(family, variant)
        tried = [row["batch"] for row in ROWS if row["model"] == f"{family}/{variant}"]
        assert recipe["batch_size"] == max(tried) == 16, "the ladder starts at the recipe batch (D-036) and the first rung fit"
        assert recipe["analysis_args"]["eval_batch_size"] == recipe["batch_size"]


def test_the_readme_says_batch_16_and_where_it_stops_fitting():
    readme = (ROOT / "src" / "train" / "README.md").read_text()
    assert "backbone_seg_memory_evidence.json" in readme
    assert "`convnext_seg/base` needs batch 8" in readme and "`convnext_seg/xlarge` did not fit" in readme
