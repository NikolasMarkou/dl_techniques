"""``load_model`` refuses, at load and naming the problem, a record and a file that are not one run.

Review-iter-1 concern 17. Before: a ``task`` of another task, a real family or variant other
than the model's, a wrong ``input_shape`` and a wrong ``channels`` were accepted or died with a
Keras traceback at predict; a zip with no ``config.json`` raised ``KeyError`` and one with no
weights ``ValueError``. Each is a ``ConfigError`` now, and a control in each test shows the same
file loads once the record is right. The family and variant rule is checked on all six families:
the smallest variant of each loads with its own record (no false refusal) and is refused under
another variant of its family and under another family.

Everything writes under ``tmp_path``.
"""

import json
import shutil
import zipfile
from pathlib import Path

import keras
import numpy as np
import pytest

from train.api import recipes, registry
from train.api.errors import ConfigError
from train.api.predictor import Predictor, build_task_record, load_model

#: The smallest variant of each family, and a second variant of the same family whose table
#: differs from it in a key the model's config also holds (None where the family has one).
SMALLEST = {
    "vit": ("vit_pico", "vit_tiny"),
    "resnet": ("resnet18", "resnet34"),
    "convnext_v1": ("cifar10", "tiny"),
    "convnext_v2": ("atto", "femto"),
    "cliffordnet": ("nano", "lite"),
    "cliffordnet_hier": ("hier_345", None),
}
CLASSES = ["a", "b", "c"]
SIDE = 32


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def record_for(family, variant, **changes):
    base = build_task_record(
        task="image_classification",
        model=f"{family}/{variant}",
        family=family,
        variant=variant,
        class_names=CLASSES,
        input_shape=(SIDE, SIDE, 3),
        normalize="unit",
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        data_kind="folder",
        source_channels=3,
    )
    base.update(changes)
    return base


@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    """One saved model per family: ``{family: path to its .keras file}``."""
    root = tmp_path_factory.mktemp("models")
    files = {}
    for family, (variant, _) in SMALLEST.items():
        spec = registry.MODELS[family]
        model = recipes.build_model(spec, variant, len(CLASSES), (SIDE, SIDE, 3), {})
        model(np.zeros((1, SIDE, SIDE, 3), np.float32), training=False)
        files[family] = root / f"{family}.keras"
        model.save(files[family])
    return files


def place(built, held, tmp_path, name="run", **changes):
    """A directory holding the model of family ``held`` and a record edited by ``changes``.

    ``changes`` may carry ``family`` and ``variant`` (the record's claim) as well as record keys.
    """
    directory = tmp_path / name
    directory.mkdir(parents=True)
    shutil.copy(built[held], directory / "best_model.keras")
    variant = changes.pop("variant", SMALLEST[held][0])
    record = record_for(changes.pop("family", held), variant, **changes)
    (directory / "task.json").write_text(json.dumps(record))
    return directory


@pytest.mark.parametrize("family", sorted(SMALLEST))
def test_each_family_loads_under_its_own_record(built, family, tmp_path):
    """The control, and the proof that the family and variant rule refuses no honest file."""
    predictor = load_model(place(built, family, tmp_path))
    assert predictor.class_names == CLASSES


@pytest.mark.parametrize("family", sorted(SMALLEST))
def test_another_variant_of_the_same_family_is_refused_naming_the_variant(built, family, tmp_path):
    other = SMALLEST[family][1]
    if other is None:
        pytest.skip("this family has one variant; the unknown-variant test covers it")
    with pytest.raises(ConfigError, match=rf"names the variant '{other}' but .* is not that variant"):
        load_model(place(built, family, tmp_path, variant=other))


@pytest.mark.parametrize("family", sorted(SMALLEST))
def test_a_variant_the_family_does_not_have_is_refused_naming_it(built, family, tmp_path):
    with pytest.raises(ConfigError, match=r"'nosuchvariant' is not a variant of .* \(known: "):
        load_model(place(built, family, tmp_path, variant="nosuchvariant"))


def test_another_registered_family_is_refused_naming_both(built, tmp_path):
    """A resnet file under a vit record: the record's own halves agree (``vit/vit_pico``)."""
    with pytest.raises(ConfigError, match=r"names the family 'vit' \(ViT\) but .* holds a ResNet"):
        load_model(place(built, "resnet", tmp_path, family="vit", variant="vit_pico"))
    with pytest.raises(ConfigError, match=r"names the family 'cliffordnet' \(CliffordNet\) but .* holds a HierarchicalCliffordNet"):
        load_model(place(built, "cliffordnet_hier", tmp_path, name="hier", family="cliffordnet", variant="nano"))


def test_a_record_of_another_task_is_refused_naming_the_task(built, tmp_path):
    with pytest.raises(ConfigError, match=r"the task is 'image_segmentation' but this predictor serves 'image_classification'"):
        load_model(place(built, "vit", tmp_path, task="image_segmentation"))


def test_a_wrong_input_side_is_refused_at_load_naming_input_shape(built, tmp_path):
    with pytest.raises(ConfigError, match=r"does not take the input_shape \[64, 64, 3\] it names") as caught:
        load_model(place(built, "vit", tmp_path, input_shape=[64, 64, 3]))
    assert caught.value.__cause__ is not None  # the Keras or TensorFlow error stays attached


def test_a_wrong_channel_count_is_refused_at_load_naming_input_shape(built, tmp_path):
    with pytest.raises(ConfigError, match=r"does not take the input_shape \[32, 32, 1\] it names"):
        load_model(place(built, "vit", tmp_path, input_shape=[SIDE, SIDE, 1], channels=1, mean=[0.0], std=[1.0]))


def rewrite_without(model_file: Path, drop: str) -> None:
    """Rewrite the .keras zip at ``model_file`` without the members whose name contains ``drop``."""
    with zipfile.ZipFile(model_file) as archive:
        kept = {name: archive.read(name) for name in archive.namelist() if drop not in name}
    model_file.unlink()
    with zipfile.ZipFile(model_file, "w") as archive:
        for name, data in kept.items():
            archive.writestr(name, data)


@pytest.mark.parametrize("drop, missing", [("config.json", "config.json"), ("weights", "model.weights.h5")])
def test_a_zip_missing_its_config_or_its_weights_is_refused_naming_the_member(built, tmp_path, drop, missing):
    directory = place(built, "vit", tmp_path)
    rewrite_without(directory / "best_model.keras", drop)
    with pytest.raises(ConfigError, match=rf"is a zip but not a \.keras model: it has no .*{missing}"):
        load_model(directory)


def test_a_zip_with_neither_names_both(built, tmp_path):
    directory = place(built, "vit", tmp_path)
    model_file = directory / "best_model.keras"
    model_file.unlink()
    with zipfile.ZipFile(model_file, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(ConfigError, match=r"no 'config\.json' and no 'model\.weights\.h5' or 'model\.weights\.npz' \(members: hello\.txt\)"):
        load_model(directory)


def test_the_untouched_directory_still_loads_after_those_edits(built, tmp_path):
    """Control for the zip tests: the same writer, nothing dropped, loads."""
    directory = place(built, "vit", tmp_path)
    rewrite_without(directory / "best_model.keras", "nothing-has-this-name")
    assert load_model(directory).class_names == CLASSES


def test_a_model_whose_output_is_not_one_row_of_scores_per_image_is_refused(tmp_path):
    """Width alone is not enough: ``(N, 64, 3)`` has the right last axis and no softmax over classes."""
    inputs = keras.Input((8, 8, 3))
    outputs = keras.layers.Dense(3)(keras.layers.Reshape((64, 3))(inputs))
    model = keras.Model(inputs, outputs)
    record = record_for("vit", "vit_pico", input_shape=[8, 8, 3])
    with pytest.raises(ConfigError, match=r"outputs \(64, 3\) per image but task\.json names 3 classes"):
        Predictor(model, record).predict(np.zeros((8, 8, 3), np.uint8))
    flat = keras.Model(inputs, keras.layers.Dense(3)(keras.layers.Flatten()(inputs)))
    assert Predictor(flat, record).predict(np.zeros((8, 8, 3), np.uint8))["labels"]  # control


class Owner(keras.Model):
    """A stand-in for a family's model class; its config and its variant table are set per test."""

    VARIANTS = {"small": {"stages": [1, 2], "width": 8}, "large": {"stages": [3, 4], "width": 16}}

    def __init__(self, stages, width):
        super().__init__()
        self.stages, self.width = stages, width

    def get_config(self):
        return {"stages": self.stages, "width": self.width, "name": "owner"}


def test_a_tuple_in_the_models_config_equals_the_same_list_in_the_variant_table(tmp_path):
    """A config may hold a tuple where the table holds a list (Keras round trips turn one into the
    other); that is the same architecture and is not a refusal, while a different list is."""
    from train.api.predictor import check_model_against_record

    spec = registry.ModelSpec(
        family="owner", factory=f"{__name__}:Owner", variants_attr=f"{__name__}:Owner.VARIANTS",
        tasks=("image_classification",),
    )
    record = record_for("owner", "small")
    files = (tmp_path / "m.keras", tmp_path / "task.json")
    check_model_against_record(Owner((1, 2), 8), record, spec, *files)  # a tuple against the list: equal
    with pytest.raises(ConfigError, match=r"stages is \[3, 4\] in the model, \[1, 2\] in 'small'"):
        check_model_against_record(Owner((3, 4), 8), record, spec, *files)
