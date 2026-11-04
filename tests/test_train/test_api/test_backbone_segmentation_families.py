"""``resnet_seg`` and ``convnext_seg``: a classification backbone plus a single-map head (plan 98b866fb step 2, D-012).

Each family is the classifier's own factory with ``include_top=False`` and one ``SegmentationHead`` on its last
feature map, an FCN-32s-shaped network (Long et al. 2015, section 4.1) with no skip connection. What is claimed,
each with the control that fails when it is not:

* both families train ``image_segmentation`` only and read the classifiers' own variants tables (one table, not a
  copy: the rows are the very objects of ``ResNet.MODEL_VARIANTS`` and ``ConvNeXtV1.MODEL_VARIANTS``);
* the stride the callables return is the one measured by building every variant at 128x256 (the table below is
  written out from ``scratch_step2/strides_table.json``; the test does not read the callables it checks), and every
  variant really builds with an output the size of its input at a non-square size;
* a ``.keras`` round trip keeps every weight path and value, one train step moves the backbone, the head and the
  BatchNorm statistics, and a second run in training mode is deterministic (the head holds no live dropout);
* a side that is not a multiple of the stride, a ``strides`` that gives no power of two, an unknown ``stem_type`` and
  a key the adapter sets are refused by name BEFORE the run directory exists, through the function and the command line;
* ``check_model_against_record`` refuses a file under a record that names another variant or another family, by what
  the layers show (the functional wrapper is not an instance of ``ResNet``), and accepts the right record;
* the adapter source names no ``create_head``, ``HeadDomain`` or ``heads.task_types``, imports the head from its leaf
  module only, and passes EVERY setting the head takes (a new knob of the head fails the coverage test, so the
  decision is made, not inherited: the head's defaults changed once under this family's feet).

Everything is built small and written under ``tmp_path``.
"""

import ast
import inspect
import json
from pathlib import Path

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api import segmentation_models as sm
from train.api.errors import ConfigError
from train.api.predictor import (
    build_segmentation_record,
    check_model_against_record,
    load_model,
    overrides_of_variant,
)
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_segmentation import train_image_segmentation
from train.common.classification_run import unset_batch_statistics

from . import _segmentation_fixtures as fx
from .test_image_segmentation_task import SIZE as FIT_SIZE, data_of, folders  # noqa: F401  (fixture: 3-class synthetic pairs)

SEGMENTATION = "image_segmentation"
CLASSES = ["pet", "background", "border"]

#: Measured by building every variant with ``include_top=False`` at 128x256 (scratch_step2/strides_table.json).
RESNET_STRIDES = {"resnet18": 32, "resnet34": 32, "resnet50": 32, "resnet101": 32, "resnet152": 32}
RESNET_CIFAR_STRIDES = {name: 8 for name in RESNET_STRIDES}
#: ConvNeXt V1 at ``strides=2`` (the family's) and the classifier's ``strides=4``: ``strides ** stages``.
CONVNEXT_STRIDES = {"cifar10": 4, "tiny": 16, "small": 16, "base": 16, "large": 16, "xlarge": 16}
CONVNEXT_STRIDES_4 = {"cifar10": 16, "tiny": 256, "small": 256, "base": 256, "large": 256, "xlarge": 256}


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield registry.MODELS


def build(family, variant, size, classes=len(CLASSES), **model_args):
    return recipes.build_model(registry.MODELS[family], variant, classes, (*size, 3), model_args)


# ---------------------------------------------------------------------
# Registration and the tables
# ---------------------------------------------------------------------


def test_both_families_train_segmentation_only_and_read_the_classifiers_tables(world):
    from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
    from dl_techniques.models.vision.resnet.model import ResNet

    for family in ("resnet_seg", "convnext_seg"):
        assert world[family].tasks == (SEGMENTATION,)
        assert any(name.startswith(f"{family}/") for name in registry.list_models(SEGMENTATION))
        assert not any(name.startswith(f"{family}/") for name in registry.list_models("image_classification"))
    assert world["resnet_seg"].load_variants() is ResNet.MODEL_VARIANTS
    assert world["convnext_seg"].load_variants() is ConvNeXtV1.MODEL_VARIANTS
    assert sorted(RESNET_STRIDES) == sorted(ResNet.MODEL_VARIANTS)
    assert sorted(CONVNEXT_STRIDES) == sorted(ConvNeXtV1.MODEL_VARIANTS)


@pytest.mark.parametrize("task, trains", [("image_classification", SEGMENTATION)])
def test_a_classification_task_refuses_a_backbone_segmenter_naming_the_siblings(world, task, trains):
    for family in ("resnet_seg", "convnext_seg"):
        with pytest.raises(ConfigError, match=rf"the family '{family}' trains {trains}, not '{task}'"):
            registry.parse_model_name(f"{family}/{'resnet18' if family == 'resnet_seg' else 'tiny'}", task=task)


# ---------------------------------------------------------------------
# The stride: written out, then checked against a real build
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant, stride", sorted(RESNET_STRIDES.items()))
def test_the_resnet_stride_is_32_with_the_imagenet_stem_and_8_with_the_cifar_stem(world, variant, stride):
    spec = world["resnet_seg"]
    assert spec.stride_of(variant, {}) == stride
    assert spec.stride_of(variant, {"stem_type": "imagenet"}) == stride
    assert spec.stride_of(variant, {"stem_type": "cifar"}) == RESNET_CIFAR_STRIDES[variant] == 8


@pytest.mark.parametrize("variant", sorted(CONVNEXT_STRIDES))
def test_the_convnext_stride_is_strides_to_the_number_of_stages(world, variant):
    spec = world["convnext_seg"]
    assert spec.stride_of(variant, {}) == CONVNEXT_STRIDES[variant]
    assert spec.stride_of(variant, {"strides": 2}) == CONVNEXT_STRIDES[variant]
    assert spec.stride_of(variant, {"strides": 4}) == CONVNEXT_STRIDES_4[variant]
    assert spec.stride_of(variant, {"strides": 1}) == 1


def test_a_model_arg_that_changes_the_stage_count_moves_the_stride(world):
    assert world["resnet_seg"].stride_of("resnet18", {"blocks_per_stage": [2, 2, 2]}) == 16
    assert world["convnext_seg"].stride_of("tiny", {"depths": [3, 3, 9], "dims": [96, 192, 384]}) == 8


@pytest.mark.parametrize(
    "family, variant, extra",
    [(f, v, e) for f, table, e in (
        ("resnet_seg", RESNET_STRIDES, {}), ("resnet_seg", RESNET_CIFAR_STRIDES, {"stem_type": "cifar"}),
        ("convnext_seg", CONVNEXT_STRIDES, {}), ("convnext_seg", CONVNEXT_STRIDES_4, {"strides": 4}),
    ) for v in table],
    ids=lambda value: str(value),
)
def test_every_variant_builds_with_an_output_the_size_of_a_non_square_input(world, family, variant, extra):
    """The factory itself checks the built output; the shape here is read from the graph (no forward pass)."""
    stride = world[family].stride_of(variant, extra)
    size = (2 * stride, 3 * stride) if stride <= 32 else (stride, 2 * stride)
    model = build(family, variant, size, **extra)
    assert tuple(model.output_shape) == (None, *size, len(CLASSES))
    assert [type(layer).__name__ for layer in model.layers][-1] == "SegmentationHead"


def test_the_factory_refuses_an_input_that_is_no_multiple_and_says_the_stride(world):
    with pytest.raises(ValueError, match="resnet_seg needs each side of input_shape to be a multiple of 32, got 50x64"):
        sm.create_resnet_seg(variant="resnet18", num_classes=3, input_shape=(50, 64, 3))
    with pytest.raises(ValueError, match="convnext_seg needs each side of input_shape to be a multiple of 16, got 32x40"):
        sm.create_convnext_seg(variant="tiny", num_classes=3, input_shape=(32, 40, 3))
    with pytest.raises(ValueError, match="Unknown variant 'resnet19'"):
        sm.create_resnet_seg(variant="resnet19", num_classes=3, input_shape=(64, 64, 3))


def test_the_factory_refuses_an_output_that_is_not_the_inputs_size_even_when_the_stride_rule_is_wrong(world, monkeypatch):
    """The second line of defence: a stride callable that disagrees with the backbone is caught at the build, not at the loss."""
    monkeypatch.setattr(sm, "resnet_seg_stride", lambda variant, model_args: 16)  # the real stride is 32
    with pytest.raises(ValueError, match=r"the resnet_seg output is 32x32 for a 64x64 input \(output stride 16\)"):
        sm.create_resnet_seg(variant="resnet18", num_classes=3, input_shape=(64, 64, 3))


# ---------------------------------------------------------------------
# What the models do
# ---------------------------------------------------------------------

SMALL = [
    pytest.param(("resnet_seg", "resnet18", (64, 96)), id="resnet18"),
    pytest.param(("resnet_seg", "resnet50", (32, 64)), id="resnet50"),
    pytest.param(("convnext_seg", "cifar10", (16, 32)), id="cifar10"),
    pytest.param(("convnext_seg", "tiny", (32, 48)), id="tiny"),
]


def weights_of(model):
    return {variable.path: np.array(variable.numpy()) for variable in model.weights}


@pytest.fixture(scope="module", params=SMALL)
def small(world, request):
    family, variant, size = request.param
    model = build(family, variant, size)
    images = np.random.default_rng(0).random((4, *size, 3), np.float32)
    return family, variant, size, model, images


def test_the_output_is_one_logit_per_class_at_the_input_size_and_is_not_a_softmax(small):
    _, _, size, model, images = small
    logits = np.asarray(model(images, training=False))
    assert logits.shape == (4, *size, len(CLASSES))
    assert np.isfinite(logits).all()
    assert float(np.abs(logits.sum(-1) - 1.0).max()) > 1e-3, "linear logits: the channels do not sum to one"


def test_a_keras_round_trip_keeps_every_weight_path_and_value(small, tmp_path):
    _, _, _, model, images = small
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.saving.load_model(path)
    before, after = weights_of(model), weights_of(reloaded)
    assert sorted(before) == sorted(after) and len(before) > 50
    assert all(np.array_equal(before[key], after[key]) for key in before)
    assert np.array_equal(np.asarray(model(images, training=False)), np.asarray(reloaded(images, training=False)))


def test_the_head_is_the_configured_one_and_holds_no_live_dropout(small):
    family, variant, _, model, images = small
    head = model.get_layer("segmentation_head")
    config = head.get_config()
    assert (config["use_ffn"], config["use_attention"], config["dropout_rate"]) == (False, False, 0.0)
    assert (config["normalization_type"], config["activation_type"]) == ("layer_norm", "gelu")
    assert config["use_skip_connections"] is False and config["skip_channel_mode"] == "auto"
    assert config["hidden_dim"] == min(256, int(head.input.shape[-1]))
    assert config["upsampling_factor"] == registry.MODELS[family].stride_of(variant, {})
    assert not [layer for layer in head._flatten_layers() if isinstance(layer, keras.layers.Dropout)]
    # No random op: two training-mode passes give one output (a BatchNorm backbone normalizes by the batch, and
    # its moving statistics enter only the update).
    assert np.array_equal(np.asarray(model(images, training=True)), np.asarray(model(images, training=True)))


def test_the_head_width_is_256_capped_by_the_backbone(world):
    wide = build("resnet_seg", "resnet18", (32, 32))
    assert wide.get_layer("segmentation_head").hidden_dim == 256
    narrow = build("convnext_seg", "cifar10", (16, 16))
    assert narrow.get_layer("segmentation_head").hidden_dim == 192 == narrow.get_layer("backbone").dims[-1]


def test_one_train_step_moves_the_backbone_the_head_and_the_batchnorm_statistics(world):
    for family, variant, size in (("resnet_seg", "resnet18", (32, 64)), ("convnext_seg", "cifar10", (16, 32))):
        model = build(family, variant, size)
        keras.utils.set_random_seed(1)
        images = np.random.default_rng(1).random((4, *size, 3), np.float32)
        masks = np.random.default_rng(2).integers(0, len(CLASSES), (4, *size)).astype(np.int32)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        before = weights_of(model)
        loss = model.train_on_batch(images, masks)
        after = weights_of(model)
        assert np.isfinite(loss)
        changed = {key for key in before if not np.array_equal(before[key], after[key])}
        assert any(key.startswith("backbone/") for key in changed), family
        assert any(key.startswith("segmentation_head/") for key in changed), family
        if family == "resnet_seg":
            assert any("moving_mean" in key for key in changed), "the BatchNorm statistics are updated in training"


# ---------------------------------------------------------------------
# Refusals before the run directory exists
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    root = tmp_path_factory.mktemp("backbone_seg_folder")
    images, masks = fx.make_folder(root / "train", count=12, size=(32, 64), classes=3)
    return {"format": "folder", "train_images": str(images), "train_masks": str(masks), "num_classes": 3}


def refused(tmp_path, folder, match, model, **changes):
    out = tmp_path / "never"
    request = dict(out=str(out), model=model, data=folder, epochs=1, batch_size=4,
                   analysis_args={"model_analysis": False, "figures": False})
    request.update(changes)
    request.setdefault("transform_args", {"image_size": [32, 64]})
    with pytest.raises(ConfigError, match=match):
        train_image_segmentation(**request)
    assert not out.exists(), "the refusal comes before the run directory is claimed"


@pytest.mark.parametrize(
    "model, size, match",
    [
        ("resnet_seg/resnet18", [50, 64], r"the height 50 is not a multiple of 32, which resnet_seg/resnet18 needs.*use 32 or 64"),
        ("resnet_seg/resnet50", [64, 100], r"the width 100 is not a multiple of 32.*use 96 or 128"),
        ("convnext_seg/tiny", [40, 64], r"the height 40 is not a multiple of 16, which convnext_seg/tiny needs.*use 32 or 48"),
        ("convnext_seg/cifar10", [30, 64], r"the height 30 is not a multiple of 4.*use 28 or 32"),
    ],
)
def test_a_side_that_is_not_a_multiple_of_the_stride_is_refused_before_the_claim(
    world, folder, tmp_path, model, size, match
):
    refused(tmp_path, folder, match, model, transform_args={"image_size": size})


def test_the_stem_and_the_depth_move_the_rule_the_user_is_told(world, folder, tmp_path):
    refused(tmp_path, folder, r"the height 12 is not a multiple of 8",
            "resnet_seg/resnet18", transform_args={"image_size": [12, 64]}, model_args={"stem_type": "cifar"})


@pytest.mark.parametrize(
    "model, model_args, match",
    [
        ("convnext_seg/tiny", {"strides": 3}, r"model_args\.strides.*3 over the 4 stages of convnext_seg/tiny is an output stride of 81, which is not a power of two"),
        ("convnext_seg/tiny", {"strides": 0}, r"model_args\.strides.*an int of 1 or more"),
        ("convnext_seg/tiny", {"strides": True}, r"model_args\.strides.*an int of 1 or more"),
        ("convnext_seg/tiny", {"strides": 2.0}, r"model_args\.strides.*an int of 1 or more"),
        ("resnet_seg/resnet18", {"stem_type": "wide"}, r"model_args\.stem_type.*'cifar', 'imagenet'.*'wide'"),
        ("resnet_seg/resnet18", {"blocks_per_stage": []}, r"model_args\.blocks_per_stage.*non-empty list"),
        ("resnet_seg/resnet18", {"include_top": True}, r"'include_top'.*set by the 'resnet_seg' adapter"),
        ("resnet_seg/resnet18", {"enable_deep_supervision": True}, r"'enable_deep_supervision'.*set by the 'resnet_seg' adapter"),
        # `pretrained` is NOT set by the adapter, which never passes it: the message must not say so.
        ("resnet_seg/resnet18", {"pretrained": "w.keras"}, r"'pretrained' is not offered by the 'resnet_seg' family yet.*does not load a classifier's weights"),
        ("convnext_seg/tiny", {"pretrained": False}, r"'pretrained' is not offered by the 'convnext_seg' family yet"),
        ("convnext_seg/tiny", {"head_config_overrides": {}}, r"'head_config_overrides'.*set by the 'convnext_seg' adapter"),
        ("convnext_seg/tiny", {"name": "x"}, r"'name'.*set by the 'convnext_seg' adapter"),
    ],
)
def test_a_model_arg_the_family_cannot_take_is_refused_by_name_before_the_claim(
    world, folder, tmp_path, model, model_args, match
):
    refused(tmp_path, folder, match, model, model_args=model_args,
            transform_args={"image_size": [64, 64]})


@pytest.mark.parametrize("family", ["resnet_seg", "convnext_seg"])
def test_pretrained_is_a_key_the_family_does_not_offer_and_never_one_the_adapter_sets(family):
    # The adapter never passes `pretrained`; the old message said it did (review-iter-1 concern 5).
    alone = sm._fixed_clash(family, {"pretrained": False})
    assert f"'pretrained' is not offered by the {family!r} family yet" in alone
    assert "is set by" not in alone and alone.count("'pretrained'") == 1
    both = sm._fixed_clash(family, {"pretrained": False, "name": "x"})
    assert f"'name' is set by the {family!r} adapter" in both
    assert f"'pretrained' is not offered by the {family!r} family yet" in both
    assert "'pretrained' is set by" not in both and both.count("'pretrained'") == 1
    assert sm._fixed_clash(family, {}) == ""


def test_the_command_line_exits_2_with_one_line_and_no_directory(world, folder, tmp_path, capsys):
    out = tmp_path / "run"
    code = cli.main([
        SEGMENTATION, f"out={out}", "model=resnet_seg/resnet18", "data.format=folder",
        f"data.train_images={folder['train_images']}", f"data.train_masks={folder['train_masks']}",
        "data.num_classes=3", "transform_args.image_size=[50,64]", "epochs=1",
    ])
    captured = capsys.readouterr()
    assert code == 2 and not out.exists()
    lines = [line for line in captured.err.splitlines() if line.startswith("error:")]
    assert len(lines) == 1 and "the height 50 is not a multiple of 32" in lines[0]
    assert "Traceback" not in captured.err


def test_the_function_and_the_command_line_write_one_config_for_a_backbone_family(world, folder, tmp_path):
    """The run stops at ``build_model`` after ``config.json``; both entry points write the same bytes."""
    from train.api.tasks.image_segmentation import ImageSegmentationTrainer

    class Stop(Exception):
        pass

    def stop(self, *args, **kwargs):
        raise Stop

    outs = {name: tmp_path / name for name in ("function", "cli")}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ImageSegmentationTrainer, "build_model", stop)
        with pytest.raises(Stop):
            train_image_segmentation(
                out=str(outs["function"]), model="convnext_seg/tiny", data=folder, epochs=1, batch_size=4,
                transform_args={"image_size": [32, 64]}, analysis_args={"model_analysis": False, "figures": False})
        with pytest.raises(Stop):
            cli.main([
                SEGMENTATION, f"out={outs['cli']}", "model=convnext_seg/tiny", "data.format=folder",
                f"data.train_images={folder['train_images']}", f"data.train_masks={folder['train_masks']}",
                "data.num_classes=3", "epochs=1", "batch_size=4", "transform_args.image_size=[32,64]",
                "analysis_args.model_analysis=false", "analysis_args.figures=false",
            ])
    texts = {name: (path / "config.json").read_text().replace(str(path), "<out>") for name, path in outs.items()}
    assert texts["function"] == texts["cli"]
    assert json.loads(texts["function"])["model"] == "convnext_seg/tiny"


# ---------------------------------------------------------------------
# Family and variant of a record, read from the layers
# ---------------------------------------------------------------------

SIZE = (32, 64)


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


@pytest.fixture(scope="module")
def resnet18(world):
    return build("resnet_seg", "resnet18", SIZE)


@pytest.fixture(scope="module")
def convnext_cifar(world):
    return build("convnext_seg", "cifar10", SIZE)


def test_the_right_record_is_accepted(resnet18, convnext_cifar):
    check(resnet18, record_of("resnet_seg", "resnet18"))
    check(convnext_cifar, record_of("convnext_seg", "cifar10"))


@pytest.mark.parametrize("variant, key", [
    ("resnet34", "blocks_per_stage is [2, 2, 2, 2] in the model, [3, 4, 6, 3]"),
    ("resnet50", "block_type is \"basic\" in the model, \"bottleneck\""),
    ("resnet101", "blocks_per_stage is [2, 2, 2, 2] in the model, [3, 4, 23, 3]"),
])
def test_a_resnet18_model_is_refused_under_another_variant_naming_the_key(resnet18, variant, key):
    text = refusal(resnet18, record_of("resnet_seg", variant))
    assert key in text and f"names the variant '{variant}' of 'resnet_seg'" in text and "not from one run" in text


@pytest.mark.parametrize("variant, key", [
    ("tiny", "depths is [5, 5] in the model, [3, 3, 9, 3]"),
    ("base", "dims is [96, 192] in the model, [128, 256, 512, 1024]"),
])
def test_a_convnext_cifar10_model_is_refused_under_another_variant(convnext_cifar, variant, key):
    assert key in refusal(convnext_cifar, record_of("convnext_seg", variant))


def test_the_two_backbone_families_are_told_apart_and_so_are_the_u_nets(resnet18, convnext_cifar, world):
    text = refusal(resnet18, record_of("convnext_seg", "cifar10"))
    assert "names the family 'convnext_seg' but m.keras holds no layer of that family" in text
    assert "ResNet" in text and "SegmentationHead" in text, "the message says what the file does hold"
    assert "holds no layer of that family" in refusal(convnext_cifar, record_of("resnet_seg", "resnet18"))
    assert "holds no layer of that family" in refusal(resnet18, record_of("convunext", "tiny"))
    unet = recipes.build_model(world["convunext"], "tiny", 3, (32, 64, 3), {})
    assert "holds no layer of that family" in refusal(unet, record_of("resnet_seg", "resnet18"))


def test_a_convnext_v2_backbone_and_a_bare_resnet_are_not_the_families(world):
    from dl_techniques.models.vision.convnext.convnext_v2 import create_convnext_v2
    from dl_techniques.models.vision.resnet.model import create_resnet

    v2 = sm._segmenter(
        "convnext_seg", create_convnext_v2("cifar10", include_top=False, input_shape=(*SIZE, 3), strides=2),
        (*SIZE, 3), 3, 4)
    assert sm.observe_convnext_seg(v2) is None
    assert "holds no layer of that family" in refusal(v2, record_of("convnext_seg", "cifar10"))
    bare = create_resnet("resnet18", include_top=False, input_shape=(*SIZE, 3))
    inputs = keras.Input((*SIZE, 3))
    headless = keras.Model(inputs, bare(inputs))
    assert sm.observe_resnet_seg(headless) is None, "no SegmentationHead: not a segmenter of this family"


def test_a_recorded_override_is_honoured_and_an_unrecorded_one_is_refused(world):
    args = {"blocks_per_stage": [1, 1, 1, 1]}
    model = build("resnet_seg", "resnet18", SIZE, **args)
    check(model, record_of("resnet_seg", "resnet18", args))
    assert "blocks_per_stage is [1, 1, 1, 1] in the model, [2, 2, 2, 2]" in refusal(
        model, record_of("resnet_seg", "resnet18"))


@pytest.fixture(scope="module")
def saved(resnet18, tmp_path_factory):
    directory = tmp_path_factory.mktemp("backbone_seg") / "run"
    directory.mkdir()
    resnet18.save(directory / "best_model.keras")
    return directory


def place(saved, tmp_path, name, record):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "best_model.keras").write_bytes((saved / "best_model.keras").read_bytes())
    (directory / "task.json").write_text(json.dumps(record))
    return directory


def test_load_model_refuses_the_swaps_and_loads_the_control(saved, tmp_path):
    assert load_model(place(saved, tmp_path, "right", record_of("resnet_seg", "resnet18"))).record["variant"] == "resnet18"
    for name, family, variant in (("r34", "resnet_seg", "resnet34"), ("cx", "convnext_seg", "cifar10")):
        with pytest.raises(ConfigError, match="not from one run"):
            load_model(place(saved, tmp_path, name, record_of(family, variant)))


def test_export_of_a_swapped_record_exits_2_and_writes_nothing(saved, tmp_path, capsys):
    swapped = place(saved, tmp_path, "swapped", record_of("resnet_seg", "resnet34"))
    out = tmp_path / "exported.keras"
    assert cli.main(["export", f"model={swapped}", f"out={out}", "format=keras"]) == 2
    assert "is not that variant" in capsys.readouterr().err
    assert not out.exists() and not Path(f"{out}.task.json").exists()
    good = place(saved, tmp_path, "good", record_of("resnet_seg", "resnet18"))
    assert cli.main(["export", f"model={good}", f"out={tmp_path / 'ok.keras'}", "format=keras"]) == 0
    assert json.loads(Path(f"{tmp_path / 'ok.keras'}.task.json").read_text())["family"] == "resnet_seg"


def test_a_classifier_family_with_a_class_table_is_still_checked_by_its_class(world):
    """The control for the change in ``check_model_against_record``: a classifier has no observer, so its class counts."""
    assert all(world[family].observe is None for family in ("vit", "resnet", "convnext_v1", "convnext_v2"))
    assert all(world[family].observe is not None for family in ("resnet_seg", "convnext_seg", "convunext", "accunet"))


# ---------------------------------------------------------------------
# The head: leaf import only, every setting passed
# ---------------------------------------------------------------------

SOURCE = Path(sm.__file__)


def test_the_adapter_source_names_no_head_factory_domain_or_task_types():
    tree = ast.parse(SOURCE.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append((node.module or "", [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            imported.append(("", [alias.name for alias in node.names]))
    heads = [(module, names) for module, names in imported if "heads" in module or any("heads" in n for n in names)]
    assert heads == [("dl_techniques.layers.heads.vision.segmentation", ["SegmentationHead"])], heads
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not names & {"create_head", "HeadDomain", "task_types", "create_vision_head", "build_model_vision_head"}


def test_every_setting_the_head_takes_is_passed_so_a_new_default_or_knob_moves_nothing():
    from dl_techniques.layers.heads.vision.base import BaseVisionHead
    from dl_techniques.layers.heads.vision.segmentation import SegmentationHead

    taken = set()
    for cls in (SegmentationHead, BaseVisionHead):
        taken |= {
            name for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if name != "self" and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        }
    passed = set(sm._HEAD_SETTINGS) | {"num_classes", "upsampling_factor", "hidden_dim"}
    assert taken - passed == set(), f"the head takes {sorted(taken - passed)} that the adapter leaves to its defaults"
    assert passed - taken == set(), f"the adapter passes {sorted(passed - taken)} the head no longer takes"


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------


def test_the_recipes_are_conv_unexts_optimizer_settings_but_the_convnext_one_has_a_tenth_of_the_rate(world):
    """ConvUNext's lr 1e-3 stalls ``convnext_seg/tiny`` at the pixel prior (val_miou 0.37 at 1e-3, at 3e-4 and at 1e-3
    with warmup in 4 epochs; 0.397 at 2e-4); 1e-4 reached 0.421. The readings are in ``scratch_step2/logs/diag_runs.out``."""
    unet = world["convunext"].recipe("oxford_pet", "tiny")["optim_args"]
    assert unet["lr"] == 1e-3
    assert world["resnet_seg"].recipe("oxford_pet", "resnet18")["optim_args"] == unet
    convnext = world["convnext_seg"].recipe("oxford_pet", "tiny")["optim_args"]
    assert convnext == {**unet, "lr": sm.CONVNEXT_SEG_LR} and sm.CONVNEXT_SEG_LR == 1e-4
    # D-029: the shallow variant is the exception, per variant (the 30-epoch evidence is held by
    # test_the_convnext_seg_rate_is_what_the_30_epoch_runs_measured.py).
    assert world["convnext_seg"].recipe("oxford_pet", "cifar10")["optim_args"] == {**unet, "lr": 1e-3}
    for family, variant in (("resnet_seg", "resnet18"), ("convnext_seg", "tiny")):
        recipe = world[family].recipe("oxford_pet", variant)
        assert recipe["transform_args"] == {"augment": "flip", "image_size": 128}
        assert recipe["metric_args"] == {"watch_metric": "val_miou"}
        assert recipe["analysis_args"]["eval_batch_size"] == recipe["batch_size"]
        assert recipe["epochs"] == 30 and recipe["checkpoint_args"] == {"early_stopping_patience": 10}
    assert world["resnet_seg"].recipe("oxford_pet", "resnet18")["model_args"] == {}
    assert world["convnext_seg"].recipe("oxford_pet", "tiny")["model_args"] == {"strides": 2}


def test_file_data_gets_no_size_from_the_recipe(world):
    for family, variant in (("resnet_seg", "resnet18"), ("convnext_seg", "tiny")):
        assert "image_size" not in world[family].recipe("folder", variant)["transform_args"]


# ---------------------------------------------------------------------
# A real CPU fit per family: the summary, the reload check and the prediction
# ---------------------------------------------------------------------

NOT_JUDGED = "`init_scale_warning` is null (not judged)"


@pytest.fixture(scope="module", params=["resnet_seg/resnet18", "convnext_seg/tiny"])
def fitted(request, world, folders, tmp_path_factory):  # noqa: F811
    """One real CPU fit (1 epoch, 32 px) per family: ``(model name, run directory, summary)``."""
    name = request.param
    out = tmp_path_factory.mktemp("backbone_fit_" + name.replace("/", "_")) / "run"
    result = train_image_segmentation(
        out=str(out), model=name, data=data_of(folders, "clean"), epochs=1, batch_size=8,
        transform_args={"image_size": FIT_SIZE}, analysis_args={"model_analysis": False, "figures": False},
    )
    return name, Path(result.run_dir), json.loads(Path(result.summary_path).read_text())


def test_a_batchnorm_backbone_gives_a_null_init_scale_warning_with_its_note_and_a_layernorm_one_a_bool(fitted):
    """D-045: a ResNet's BatchNorm layers hold the statistics they were built with at epoch 0 (the nested backbone's too)."""
    name, _, summary = fitted
    text = " ".join(summary["notes"])
    if name.startswith("resnet_seg"):
        assert summary["init_scale_warning"] is None
        assert NOT_JUDGED in text and "BatchNorm" in text and "NOT caught here" in text
    else:
        assert summary["init_scale_warning"] is False and NOT_JUDGED not in text
    assert summary["model_family"] == name.split("/")[0] and summary["variant"] == name.split("/")[1]


def test_the_batchnorm_layers_of_the_nested_backbone_are_listed_at_epoch_zero(world):
    resnet = unset_batch_statistics(build("resnet_seg", "resnet18", (32, 32)))
    assert len(resnet) == 20, resnet  # resnet18: the stem's, 16 block norms and 3 projection norms; the head has none
    assert unset_batch_statistics(build("convnext_seg", "tiny", (32, 32))) == []


def test_the_reload_check_of_the_run_is_the_exact_one_and_true_for_a_nested_backbone(fitted):
    """The plan's edge case: ``exact_reload=True`` on a functional model that holds a subclassed model."""
    _, _, summary = fitted
    check = summary["model_loading_check"]
    assert summary["model_loading_validated"] is True and summary["status"] == "ok"
    assert check["ok"] is True and check["weights_equal"] is True and check["max_abs_diff"] <= check["tolerance"]


def test_the_run_directory_predicts_and_records_the_family(fitted):
    name, run_dir, _ = fitted
    record = json.loads((run_dir / "task.json").read_text())
    assert (record["family"], record["variant"], record["model"]) == (*name.split("/"), name)
    assert record["input_size"] == [FIT_SIZE, FIT_SIZE] and record["schema_version"] == 2
    image = np.random.default_rng(3).integers(0, 256, (45, 37, 3), dtype=np.uint8)
    masks = load_model(run_dir).predict(image)["masks"]
    assert masks[0].shape == (45, 37) and masks[0].max() < len(record["class_names"])
