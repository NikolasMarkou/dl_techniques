"""A segmentation ``task.json`` that names another family or variant than the model is refused (D-088).

Review-iter-3 concern 3. The two segmentation families are functional models: no class to be an
instance of and no table keys in ``get_config()``, so D-080 checked only that the variant was in the
table and that the model took ``input_size``. A ``convunext/tiny`` file loaded and predicted under
records naming ``convunext/base``, ``convunext/xlarge`` and ``accunet/base``, and ``export`` of the
``accunet/base`` copy exited 0 and wrote ``"family": "accunet"`` beside it. The families share no layer
class (ConvUNext: ``ConvStem``, ``DownsampleAndSkip``, ``ConvNextV2Block``; AccUNet: ``AccUNet``), so
``ModelSpec.observe`` reads the family and the variant-table keys the layers show, and
``check_model_against_record`` refuses a mismatch.

What is claimed, each with the control that the same model loads under the right record:

* ConvUNext ``tiny`` is refused under ``small`` (width), ``base``, ``large`` and ``xlarge`` (depth, and
  blocks per level), and under AccUNet's record; AccUNet is refused under ConvUNext's;
* ``base`` (built once, 44M parameters) loads under its own record and is refused under ``tiny``;
* an override the run recorded (``depth``, ``initial_filters``, ``base_filters``, ``mlfc_iterations``) is
  honoured, and the same model under a record that does not list it is refused naming the key;
* ``convnext_version`` v1 is told from v2;
* through ``load_model`` on a saved file, and through ``export``, which must not write a record for a
  model that is not the one it names;
* the README no longer says the family cannot be compared.

Everything is built small and written under ``tmp_path``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from train.api import cli, recipes, registry
from train.api.errors import ConfigError
from train.api.predictor import (
    build_segmentation_record,
    check_model_against_record,
    load_model,
    overrides_of_variant,
)

SIZE = (16, 32)
CLASSES = ["pet", "background", "border"]


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def build(family, variant, **model_args):
    model = recipes.build_model(registry.MODELS[family], variant, len(CLASSES), (*SIZE, 3), model_args)
    model(np.zeros((1, *SIZE, 3), np.float32), training=False)
    return model


@pytest.fixture(scope="module")
def tiny(world):
    return build("convunext", "tiny")


@pytest.fixture(scope="module")
def small_accunet(world):
    return build("accunet", "base", base_filters=8, mlfc_iterations=1)


def record_of(family, variant, model_args=None, **changes):
    table = registry.MODELS[family].load_variants()[variant]
    value = build_segmentation_record(
        model=f"{family}/{variant}", family=family, variant=variant, class_names=CLASSES,
        input_size=SIZE, channels=3, normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
        variant_overrides=overrides_of_variant(table, model_args or {}),
    )
    value.update(changes)
    return value


def check(model, record):
    check_model_against_record(
        model, record, registry.MODELS[record["family"]], Path("m.keras"), Path("task.json"))


def refusal(model, record):
    with pytest.raises(ConfigError) as caught:
        check(model, record)
    return str(caught.value)


# ---------------------------------------------------------------------
# Variants of ConvUNext, read from its layers
# ---------------------------------------------------------------------

def test_the_right_record_is_accepted(tiny, small_accunet):
    check(tiny, record_of("convunext", "tiny"))
    check(small_accunet, record_of("accunet", "base", {"base_filters": 8, "mlfc_iterations": 1}))


@pytest.mark.parametrize("variant, key", [
    ("small", "initial_filters is 32 in the model, 48"),
    ("base", "depth is 3 in the model, 4"),
    ("large", "depth is 3 in the model, 4"),
    ("xlarge", "depth is 3 in the model, 5"),
])
def test_a_tiny_model_is_refused_under_another_variant_naming_the_key(tiny, variant, key):
    text = refusal(tiny, record_of("convunext", variant))
    assert key in text and f"names the variant '{variant}'" in text and "not from one run" in text


def test_the_blocks_per_level_and_the_version_are_read_too(world):
    wide = build("convunext", "tiny", blocks_per_level=3)
    text = refusal(wide, record_of("convunext", "tiny"))
    assert "blocks_per_level is 3 in the model, 2" in text
    # 70 blocks at depth 3 is 10 per level of 7; a divisor of 6 instead of 7 would read 11 (a floor makes
    # 2 and 3 per level agree with it, so those alone cannot tell the two apart).
    many = build("convunext", "tiny", blocks_per_level=10)
    assert "blocks_per_level is 10 in the model, 2" in refusal(many, record_of("convunext", "tiny"))
    v1 = build("convunext", "tiny", convnext_version="v1")
    assert "convnext_version is \"v1\" in the model, \"v2\"" in refusal(v1, record_of("convunext", "tiny"))
    check(v1, record_of("convunext", "tiny", {"convnext_version": "v1"}))


def test_a_base_model_is_accepted_under_base_and_refused_under_tiny(world):
    base = build("convunext", "base")
    check(base, record_of("convunext", "base"))
    assert "depth is 4 in the model, 3" in refusal(base, record_of("convunext", "tiny"))


def test_the_recorded_overrides_are_honoured_and_an_unrecorded_one_is_refused(world):
    args = {"depth": 4, "initial_filters": 16}
    model = build("convunext", "tiny", **args)
    check(model, record_of("convunext", "tiny", args))
    text = refusal(model, record_of("convunext", "tiny"))
    assert "depth is 4 in the model, 3" in text and "initial_filters is 16 in the model, 32" in text


# ---------------------------------------------------------------------
# The family
# ---------------------------------------------------------------------

def test_convunext_weights_under_accunets_record_are_refused_naming_what_the_file_holds(tiny):
    text = refusal(tiny, record_of("accunet", "base"))
    assert "names the family 'accunet'" in text and "holds no layer of that family" in text
    assert "ConvStem" in text and "DownsampleAndSkip" in text, "the message says what the file does hold"


def test_accunet_weights_under_convunexts_record_are_refused(small_accunet):
    text = refusal(small_accunet, record_of("convunext", "tiny"))
    assert "names the family 'convunext'" in text and "holds no layer of that family" in text
    assert "AccUNet" in text


def test_accunets_own_keys_are_checked_and_its_recorded_overrides_honoured(small_accunet):
    text = refusal(small_accunet, record_of("accunet", "base"))
    assert "base_filters is 8 in the model, 32" in text and "mlfc_iterations is 1 in the model, 3" in text


# ---------------------------------------------------------------------
# Through load_model and export
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def saved(tiny, tmp_path_factory):
    directory = tmp_path_factory.mktemp("seg_family") / "run"
    directory.mkdir()
    tiny.save(directory / "best_model.keras")
    return directory


def place(saved, tmp_path, name, record):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "best_model.keras").write_bytes((saved / "best_model.keras").read_bytes())
    (directory / "task.json").write_text(json.dumps(record))
    return directory


def test_load_model_refuses_the_three_swaps_of_the_review_and_loads_the_control(saved, tmp_path):
    assert load_model(place(saved, tmp_path, "right", record_of("convunext", "tiny"))).record["variant"] == "tiny"
    for name, family, variant in (("base", "convunext", "base"), ("xlarge", "convunext", "xlarge"),
                                  ("accunet", "accunet", "base")):
        with pytest.raises(ConfigError, match="not from one run"):
            load_model(place(saved, tmp_path, name, record_of(family, variant)))


def test_export_of_a_swapped_record_exits_2_and_writes_nothing(saved, tmp_path, capsys):
    swapped = place(saved, tmp_path, "swapped", record_of("accunet", "base"))
    out = tmp_path / "exported.keras"
    assert cli.main(["export", f"model={swapped}", f"out={out}", "format=keras"]) == 2
    assert "holds no layer of that family" in capsys.readouterr().err
    assert not out.exists() and not Path(f"{out}.task.json").exists()
    good = place(saved, tmp_path, "good", record_of("convunext", "tiny"))
    assert cli.main(["export", f"model={good}", f"out={tmp_path / 'ok.keras'}", "format=keras"]) == 0
    assert json.loads(Path(f"{tmp_path / 'ok.keras'}.task.json").read_text())["family"] == "convunext"


def test_the_readme_states_what_is_checked_not_that_it_cannot_be(tmp_path):
    readme = (Path(__file__).resolve().parents[3] / "src" / "train" / "README.md").read_text()
    assert "cannot be checked" not in readme and "cannot compare the model with the family" not in readme
    assert "holds no layer of the family" in readme or "the layers it shows" in readme
