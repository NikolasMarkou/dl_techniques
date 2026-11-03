"""AccUNet's recipe batch is the largest one measured to fit the 24 GB card (D-088, review-iter-3 concern 6).

The recipe took the legacy trainer's batch 16, and the first GPU run of the family through the API ran out
of memory at the first training step (``[16,128,128,128]`` in ``decoder_l3_block*``). The legacy trainer
does the same on that card (17.4 GB free beside another process's 6.6 GB), so the API adds nothing to find;
D-036's procedure applies: halve until the rule holds. The rows are in ``accunet_memory_evidence.json``
(a copy of the untracked ``scratch_step33/logs``; this file does not re-measure). It pins:

* the evidence is consistent with the rule it was judged by (D-036's, read from ``memory_fit_evidence.json``
  so there is one copy of it): a row ``fits`` exactly when it completed the steps, did not run out of memory,
  had no allocator collection and kept the peak and the device under the limits;
* batch 16 does NOT fit, through the API and through the legacy trainer, and the recipe's batch is the largest
  batch with a row that fits (8) with the next one up a row that does not;
* the recipe is half the legacy trainer's batch, the learning rate is the legacy trainer's (D-036 scales no
  rate without a source rule) and the evaluation batch is the training batch;
* the README says it.

NOT checked: that a batch which fit 21 steps fits 30 epochs (the review's 3-epoch run at batch 8 completed),
and the evidence file against the machine.
"""

import json
from pathlib import Path

from train.api import recipes, registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = json.loads((HERE / "accunet_memory_evidence.json").read_text())
RULE = json.loads((HERE / "memory_fit_evidence.json").read_text())["rule"]
ROWS = EVIDENCE["rows"]


def rule_fits(row) -> bool:
    return bool(
        row["completed"] and not row["oom"] and row["allocator_gc_lines"] == 0
        and row["steps"] >= RULE["min_steps"]
        and row["peak_mib"] is not None and row["peak_mib"] <= RULE["peak_limit_mib"]
        and row["device_mib"] <= RULE["device_limit_mib"]
    )


def test_every_row_of_the_evidence_is_judged_by_the_rule_the_step_17_rows_were():
    assert [row["fits"] for row in ROWS] == [rule_fits(row) for row in ROWS]


def test_batch_16_does_not_fit_in_either_arm_and_batch_8_does():
    by = {(row["arm"], row["batch"]): row for row in ROWS}
    assert by[("api", 16)]["oom"] and by[("legacy", 16)]["oom"], "the legacy trainer fails at 16 too: the API adds nothing"
    assert by[("api", 8)]["fits"] and by[("api", 8)]["peak_mib"] < 0.5 * RULE["peak_limit_mib"]


def test_the_recipe_batch_is_the_largest_that_fits(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        recipe = registry.MODELS["accunet"].recipe("oxford_pet", "base")
    fitting = sorted(row["batch"] for row in ROWS if row["arm"] == "api" and row["fits"])
    failing = sorted(row["batch"] for row in ROWS if row["arm"] == "api" and not row["fits"])
    assert recipe["batch_size"] == max(fitting) == 8
    assert min(failing) == 2 * recipe["batch_size"], "the next batch up was measured and does not fit"
    assert recipe["analysis_args"]["eval_batch_size"] == recipe["batch_size"]


def test_the_learning_rate_is_the_legacy_trainers_and_the_batch_is_its_batch_halved(monkeypatch):
    from train.accunet.train_accunet import AccUNetTrainingConfig

    legacy = AccUNetTrainingConfig()
    with monkeypatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        recipe = registry.MODELS["accunet"].recipe("oxford_pet", "base")
    assert legacy.batch_size == 16 and recipe["batch_size"] == legacy.batch_size // 2
    assert recipe["optim_args"]["lr"] == legacy.lr == 1e-4


def test_the_readme_says_batch_8_and_why():
    readme = (ROOT / "src" / "train" / "README.md").read_text()
    assert "AccUNet's batch is 8" in readme and "accunet_memory_evidence.json" in readme
