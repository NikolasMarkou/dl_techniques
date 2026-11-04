"""``convnext_seg``'s learning rate is per variant, and each value is held to the 30-epoch runs that chose it (plan 98b866fb fix 9.3, D-029).

The family's rate is 1e-4: at ConvUNext's 1e-3 the ``tiny`` variant stays at the pixel prior (step 2). The iteration 1 gate then ran the
shallow ``cifar10`` variant at both rates and found it 0.012 to 0.016 better at 1e-3 on two seeds, so its recipe takes 1e-3 and the
family keeps 1e-4 (``segmentation_models.CONVNEXT_SEG_CIFAR10_LR``). Fix 9.3 ran a third seed (44) at the shipped rate AND its pair at 1e-4:
1e-3 is ahead again, by 0.0013 only. The effect is therefore modest and uneven (+0.016, +0.012, +0.001, mean +0.010 in test mIoU); what
the evidence supports is "ahead on every seed", not a large gain, and the test says no more than that. The rows are in
``convnext_seg_lr_evidence.json`` (30 epochs, Pet at 128 px, batch 16, copied from the runs' summaries with their sha256; this file does
not re-measure). What is claimed:

* every variant MEASURED at two rates has, as its recipe rate, the rate with the higher test mIoU on EVERY seed, and the rows are paired
  (same variant, same seeds at both rates, same epochs requested, batch and image size);
* for ``cifar10`` (three seeds) the recipe's rate is ahead on all three and by 0.010 on average; ``tiny`` at 1e-3 is at the pixel prior (a test
  mIoU under its majority-class baseline plus 0.2, against 0.59 or more at 1e-4);
* a variant with NO row (``small``, ``base``, ``large``, ``xlarge``) takes the family's rate, and the recipe's note and the README say they were
  not measured at either rate; adding a variant to ``ConvNeXtV1.MODEL_VARIANTS`` fails here until it is placed in one list or the other;
* the per-variant rate changes the rate and nothing else of the recipe, and a run at the shipped rate is among the rows with its pair.
"""

import json
from pathlib import Path

import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api import segmentation_models as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = json.loads((HERE / "convnext_seg_lr_evidence.json").read_text())["rows"]
MEASURED = ("cifar10", "tiny")
UNMEASURED = ("small", "base", "large", "xlarge")


@pytest.fixture(scope="module")
def recipe_of():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        yield lambda variant, dataset="oxford_pet": registry.MODELS["convnext_seg"].recipe(dataset, variant)


def rows(variant):
    return [row for row in EVIDENCE if row["variant"] == variant]


def by_rate_and_seed(variant):
    table = {}
    for row in rows(variant):
        assert (row["lr"], row["seed"]) not in table, f"{variant}: two rows for {(row['lr'], row['seed'])}"
        table[(row["lr"], row["seed"])] = row["test_miou"]
    return table


def test_every_row_is_a_30_epoch_pet_run_of_the_same_batch_and_size():
    assert EVIDENCE
    for row in EVIDENCE:
        assert row["model"] == f"convnext_seg/{row['variant']}" and row["dataset"] == "oxford_pet", row["run"]
        assert row["epochs_requested"] == 30 and row["batch_size"] == 16 and row["input_size"] == [128, 128], row["run"]
        assert row["epochs_run"] >= 13 and len(row["summary_sha256"]) == 64


def test_the_measured_variants_have_two_rates_on_the_same_seeds():
    for variant in MEASURED:
        table = by_rate_and_seed(variant)
        rates = sorted({rate for rate, _ in table})
        assert rates[:2] == [1e-4, 1e-3], variant
        seeds = {rate: sorted(seed for r, seed in table if r == rate) for rate in (1e-4, 1e-3)}
        assert seeds[1e-4] == seeds[1e-3] and len(seeds[1e-4]) >= 2, f"{variant}: {seeds}"


@pytest.mark.parametrize("variant", MEASURED)
def test_the_recipe_rate_of_a_measured_variant_is_the_rate_that_won_on_every_seed(variant, recipe_of):
    table = by_rate_and_seed(variant)
    seeds = sorted({seed for _, seed in table})
    winner = {seed: max((1e-4, 1e-3), key=lambda rate: table[(rate, seed)]) for seed in seeds}
    assert len(set(winner.values())) == 1, f"{variant}: the rate that wins changes with the seed {winner}"
    assert recipe_of(variant)["optim_args"]["lr"] == winner[seeds[0]]


def test_cifar10_is_ahead_at_the_recipe_rate_on_every_one_of_three_seeds_by_a_modest_mean(recipe_of):
    table = by_rate_and_seed("cifar10")
    seeds = sorted({seed for _, seed in table})
    assert seeds == [42, 43, 44]
    gaps = [table[(1e-3, seed)] - table[(1e-4, seed)] for seed in seeds]
    assert all(gap > 0 for gap in gaps), gaps
    assert sum(gaps) / len(gaps) > 0.008, gaps
    # What is NOT claimed, pinned so a reader does not take more from the rows: the third seed's gain is a tenth of the others'.
    assert min(gaps) < 0.005 < max(gaps), gaps
    assert recipe_of("cifar10")["optim_args"]["lr"] == 1e-3


def test_tiny_at_the_unet_rate_is_at_the_pixel_prior_and_at_the_family_rate_is_not(recipe_of):
    table = by_rate_and_seed("tiny")
    baseline = max(row["majority_baseline_miou"] for row in rows("tiny"))
    for seed in sorted({seed for _, seed in table}):
        assert table[(1e-3, seed)] < baseline + 0.2 and table[(1e-4, seed)] >= 0.59, seed
    assert recipe_of("tiny")["optim_args"]["lr"] == sm.CONVNEXT_SEG_LR == 1e-4


def test_every_variant_of_the_backbone_is_measured_or_listed_as_not(recipe_of):
    from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1

    assert set(ConvNeXtV1.MODEL_VARIANTS) == set(MEASURED) | set(UNMEASURED)
    for variant in UNMEASURED:
        assert not rows(variant), f"{variant} has rows: move it to MEASURED and test its winner"
        assert recipe_of(variant)["optim_args"]["lr"] == sm.CONVNEXT_SEG_LR, variant


def test_the_recipe_note_and_the_readme_say_which_variants_were_not_measured():
    source = (ROOT / "src/train/api/segmentation_models.py").read_text()
    readme = (ROOT / "src/train/README.md").read_text()
    assert "`small`, `base`, `large` and `xlarge` were\n# NOT measured at either rate" in source
    assert "`small`, `base`, `large` and `xlarge` have been\n  measured at neither rate" in readme


def test_the_per_variant_rate_changes_the_rate_and_nothing_else(recipe_of):
    family = recipe_of("tiny")
    cifar = recipe_of("cifar10")
    assert {**cifar, "optim_args": {**cifar["optim_args"], "lr": family["optim_args"]["lr"]}} == family
    assert cifar["optim_args"]["lr"] == sm.CONVNEXT_SEG_CIFAR10_LR
    assert recipe_of("cifar10", "cifar10")["optim_args"]["lr"] == sm.CONVNEXT_SEG_CIFAR10_LR, "a per-variant rate holds on every dataset"


def test_the_other_families_are_untouched():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        unet = registry.MODELS["convunext"].recipe("oxford_pet", "tiny")["optim_args"]
        assert registry.MODELS["resnet_seg"].recipe("oxford_pet", "resnet18")["optim_args"] == unet
        for variant in ("tiny", "small", "base", "large", "xlarge"):
            assert registry.MODELS["convunext"].recipe("oxford_pet", variant)["optim_args"]["lr"] == 1e-3


def test_a_run_at_the_shipped_rate_is_among_the_rows_with_its_pair(recipe_of):
    shipped = [row for row in EVIDENCE if row["why"].startswith("fix 9.3")]
    assert [row["variant"] for row in shipped] == ["cifar10"] and shipped[0]["seed"] == 44
    for row in shipped:
        assert row["lr"] == recipe_of(row["variant"])["optim_args"]["lr"], row["run"]
    (pair,) = [row for row in EVIDENCE if row["why"].startswith("seed-44 pair")]
    assert pair["variant"] == "cifar10" and pair["seed"] == 44 and pair["lr"] == sm.CONVNEXT_SEG_LR
