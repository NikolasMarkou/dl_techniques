"""``convnext_seg``'s learning rate is ConvUNext's 1e-3 for every variant, and the 30-epoch runs on the paper's layer scale are what hold it (plan 98b866fb step 10b, D-034).

History, kept in the evidence file: with the V1 block's old layer scale (init 1.0, capped to [1e-6, 1.0], L2 1e-5) ``tiny`` stayed at the
pixel prior at 1e-3, so fix 9.3 (D-029) gave the family 1e-4 and ``cifar10`` alone 1e-3. Step 10b gave the block the paper's layer scale
(init 1e-6, no cap, no penalty) and measured both variants at both rates again: ``tiny`` trains at 1e-3 and is ahead of 1e-4 on both
seeds, ``cifar10`` is ahead at 1e-3 (seed 44). The per-variant rate is gone and the family takes ConvUNext's settings. The rows are in
``convnext_seg_lr_evidence.json`` (30 epochs, Pet at 128 px, batch 16, copied from the runs' summaries with their sha256; this file does
not re-measure). What is claimed:

* a row is a 30-epoch Pet run of the same batch and size, and carries the layer scale it was measured on (``block``);
* every variant measured at two rates on the paper's layer scale has two rates on the same seeds, and the recipe's rate is the rate with
  the higher test mIoU on EVERY seed (``tiny``: seeds 42 and 43, ahead by 0.043 and 0.009; ``cifar10``: seed 44, ahead by 0.045), a
  modest and uneven margin, which is all the test says;
* the old stall of ``tiny`` at 1e-3 is a row of the old layer scale and is not reproduced on the paper's: that is the reason the rate
  changed, pinned so the reason is not forgotten;
* a variant with NO row (``small``, ``base``, ``large``, ``xlarge``) takes the same rate and the recipe's note and the README say they
  were not measured at either rate; adding a variant to ``ConvNeXtV1.MODEL_VARIANTS`` fails here until it is placed in one list or the other;
* the recipe has no per-variant rate (the optimizer settings are ConvUNext's, read from one source), and no other family is touched.
"""

import json
from pathlib import Path

import pytest

import train.api.recipes as recipes
from train.api import registry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = json.loads((HERE / "convnext_seg_lr_evidence.json").read_text())["rows"]
OLD, PAPER = "before step 10b", "paper layer scale"
MEASURED = ("cifar10", "tiny")
UNMEASURED = ("small", "base", "large", "xlarge")
RATES = (1e-4, 1e-3)


@pytest.fixture(scope="module")
def recipe_of():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        yield lambda variant, dataset="oxford_pet": registry.MODELS["convnext_seg"].recipe(dataset, variant)


def rows(variant, block):
    return [row for row in EVIDENCE if row["variant"] == variant and row["block"].startswith(block)]


def by_rate_and_seed(variant, block):
    table = {}
    for row in rows(variant, block):
        assert (row["lr"], row["seed"]) not in table, f"{variant}: two rows for {(row['lr'], row['seed'])}"
        table[(row["lr"], row["seed"])] = row["test_miou"]
    return table


def test_every_row_is_a_30_epoch_pet_run_of_the_same_batch_and_size_on_a_named_layer_scale():
    assert EVIDENCE
    for row in EVIDENCE:
        assert row["model"] == f"convnext_seg/{row['variant']}" and row["dataset"] == "oxford_pet", row["run"]
        assert row["epochs_requested"] == 30 and row["batch_size"] == 16 and row["input_size"] == [128, 128], row["run"]
        assert row["epochs_run"] >= 13 and len(row["summary_sha256"]) == 64
        assert row["block"].startswith((OLD, PAPER)), row["run"]


def test_the_measured_variants_have_two_rates_on_the_same_seeds_on_the_papers_layer_scale():
    for variant, seeds in (("cifar10", [44]), ("tiny", [42, 43])):
        table = by_rate_and_seed(variant, PAPER)
        assert sorted({rate for rate, _ in table}) == list(RATES), variant
        assert sorted(seed for rate, seed in table if rate == 1e-4) == sorted(seed for rate, seed in table if rate == 1e-3) == seeds, variant


@pytest.mark.parametrize("variant", MEASURED)
def test_the_recipe_rate_of_a_measured_variant_is_the_rate_that_won_on_every_seed(variant, recipe_of):
    table = by_rate_and_seed(variant, PAPER)
    seeds = sorted({seed for _, seed in table})
    winner = {seed: max(RATES, key=lambda rate: table[(rate, seed)]) for seed in seeds}
    assert len(set(winner.values())) == 1, f"{variant}: the rate that wins changes with the seed {winner}"
    assert recipe_of(variant)["optim_args"]["lr"] == winner[seeds[0]] == 1e-3


def test_the_margins_are_modest_and_uneven_and_the_test_claims_no_more():
    gaps = {
        variant: [table[(1e-3, seed)] - table[(1e-4, seed)] for seed in sorted({s for _, s in table})]
        for variant in MEASURED
        for table in [by_rate_and_seed(variant, PAPER)]
    }
    assert all(gap > 0 for gaps_ in gaps.values() for gap in gaps_), gaps
    assert min(gaps["tiny"]) < 0.02 < max(gaps["tiny"]), "tiny: the second seed's margin is a fifth of the first's"
    assert gaps["cifar10"][0] > 0.03, gaps


def test_the_old_stall_of_tiny_at_1e_minus_3_is_a_property_of_the_old_layer_scale():
    old, paper = by_rate_and_seed("tiny", OLD), by_rate_and_seed("tiny", PAPER)
    baseline = max(row["majority_baseline_miou"] for row in rows("tiny", OLD) + rows("tiny", PAPER))
    for seed in (42, 43):
        assert old[(1e-3, seed)] < baseline + 0.2, f"old layer scale, seed {seed}: at the pixel prior"
        assert old[(1e-4, seed)] >= 0.59, f"old layer scale, seed {seed}: 1e-4 was the working rate"
        assert paper[(1e-3, seed)] >= baseline + 0.3, f"paper layer scale, seed {seed}: 1e-3 trains"


def test_every_variant_of_the_backbone_is_measured_or_listed_as_not(recipe_of):
    from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1

    assert set(ConvNeXtV1.MODEL_VARIANTS) == set(MEASURED) | set(UNMEASURED)
    for variant in UNMEASURED:
        assert not rows(variant, OLD) and not rows(variant, PAPER), f"{variant} has rows: move it to MEASURED and test its winner"
        assert recipe_of(variant)["optim_args"]["lr"] == 1e-3, variant


def test_the_recipe_note_and_the_readme_say_which_variants_were_not_measured():
    source = (ROOT / "src/train/api/segmentation_models.py").read_text()
    readme = (ROOT / "src/train/README.md").read_text()
    assert "`small`, `base`, `large` and `xlarge` were NOT measured at either rate" in source
    assert "`small`, `base`, `large` and `xlarge` have been\n  measured at neither rate" in readme


def test_the_recipe_has_no_per_variant_rate_and_takes_conv_unexts_settings_from_one_source(recipe_of):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        spec = registry.MODELS["convnext_seg"]
        unet = registry.MODELS["convunext"].recipe("oxford_pet", "tiny")["optim_args"]
        assert "per_variant" not in spec.defaults or not spec.defaults["per_variant"]
    for variant in MEASURED + UNMEASURED:
        for dataset in ("oxford_pet", "folder"):
            assert recipe_of(variant, dataset)["optim_args"] == unet, (variant, dataset)


def test_the_other_families_are_untouched():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        recipes.register_builtin_models()
        unet = registry.MODELS["convunext"].recipe("oxford_pet", "tiny")["optim_args"]
        assert registry.MODELS["resnet_seg"].recipe("oxford_pet", "resnet18")["optim_args"] == unet
        for variant in ("tiny", "small", "base", "large", "xlarge"):
            assert registry.MODELS["convunext"].recipe("oxford_pet", variant)["optim_args"]["lr"] == 1e-3


def test_the_runs_at_the_recipe_rate_are_among_the_rows_with_their_pairs():
    for variant, seeds in (("cifar10", [44]), ("tiny", [42, 43])):
        for seed in seeds:
            paper = {row["lr"] for row in rows(variant, PAPER) if row["seed"] == seed}
            assert paper == set(RATES), (variant, seed)
