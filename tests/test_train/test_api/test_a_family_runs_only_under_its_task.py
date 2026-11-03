"""Step 23: a family runs only under its task, and a segmentation family takes only sizes it can.

Three rules, each observed where a user meets it:

* ``parse_model_name(name, task=)`` refuses a registered family under a task its ``tasks`` do not
  list, in both directions, before the family's variants table (and so its model code) is imported;
  the trainer asks it first, so the refusal is a ``ConfigError`` (exit 2 at the command line) and
  ``out`` does not exist afterwards.
* ``ModelSpec.stride`` and ``segmentation_models.check_image_size``: every side of a segmentation
  ``image_size`` is a multiple of the family's stride (``2 ** depth`` for ConvUNext, which follows a
  ``depth`` in ``model_args``; 16 for AccUNet), an int is a square and a pair is ``[height, width]``.
* The two adapters build the model the legacy trainers build, and each family really builds at a
  non-square size with the output the size of the input.

The segmentation families are NOT in ``recipes.register_builtin_models`` (the task that trains them
registers them), so the fixture below adds them to an isolated registry.
"""

import json

import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import cli, registry, segmentation_models
from train.api.errors import ConfigError, UnknownModelError
from train.api.registry import ModelSpec
from train.api.segmentation_models import (
    ACCUNET_VARIANTS,
    check_image_size,
    convunext_stride,
    create_accunet,
    create_convunext,
)
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import (
    ImageClassificationTrainer,
    train_image_classification,
)

from tests.classification_fakes import use_tiny_raw

CLASSIFICATION = "image_classification"
SEGMENTATION = "image_segmentation"


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        for spec in segmentation_models.builtin_segmentation_models():
            registry.register_model(spec)
        register_builtin_tasks()
        # The segmentation task itself is step 28's; a stand-in row lets `list_models(task=)` name it.
        registry.register_task(
            registry.TaskSpec(name=SEGMENTATION, config_class="m:C", function="m:f", summary="stand-in")
        )
        yield registry.MODELS


# ---------------------------------------------------------------------
# The family-to-task refusal
# ---------------------------------------------------------------------


def test_the_two_segmentation_families_are_segmentation_only_and_the_classifiers_are_not(world):
    for family in ("convunext", "accunet"):
        assert world[family].tasks == (SEGMENTATION,)
    for family in ("vit", "resnet", "convnext_v1", "convnext_v2", "cliffordnet", "cliffordnet_hier"):
        assert world[family].tasks == (CLASSIFICATION,)


def test_a_family_under_its_own_task_is_accepted(world):
    assert registry.parse_model_name("vit/vit_pico", task=CLASSIFICATION) == ("vit", "vit_pico")
    assert registry.parse_model_name("convunext/tiny", task=SEGMENTATION) == ("convunext", "tiny")
    assert registry.parse_model_name("accunet/base", task=SEGMENTATION) == ("accunet", "base")


def test_no_task_asks_no_question(world):
    assert registry.parse_model_name("convunext/tiny") == ("convunext", "tiny")
    assert registry.parse_model_name("vit/vit_pico", task=None) == ("vit", "vit_pico")


@pytest.mark.parametrize(
    "name, task, trains, siblings",
    [
        ("convunext/tiny", CLASSIFICATION, SEGMENTATION, "cliffordnet, cliffordnet_hier, convnext_v1"),
        ("accunet/base", CLASSIFICATION, SEGMENTATION, "cliffordnet, cliffordnet_hier, convnext_v1"),
        ("vit/vit_pico", SEGMENTATION, CLASSIFICATION, "accunet, convunext"),
        ("resnet/resnet18", SEGMENTATION, CLASSIFICATION, "accunet, convunext"),
    ],
)
def test_a_family_under_the_wrong_task_is_refused_in_both_directions_naming_the_key(
    world, name, task, trains, siblings
):
    with pytest.raises(ConfigError) as caught:
        registry.parse_model_name(name, task=task)
    text = str(caught.value)
    family = name.partition("/")[0]
    assert text.startswith("Invalid value for 'model'")
    assert f"the family {family!r} trains {trains}, not {task!r}" in text
    assert siblings in text


def test_the_task_is_checked_before_the_variants_table_is_imported(world, monkeypatch):
    """A family whose variants cannot even be imported is still refused for its task first."""
    monkeypatch.setitem(
        registry.MODELS,
        "ghost",
        ModelSpec(
            family="ghost",
            factory="no_such_module_for_step_23:make",
            variants_attr="no_such_module_for_step_23:VARIANTS",
            tasks=(SEGMENTATION,),
        ),
    )
    with pytest.raises(ConfigError, match="the family 'ghost' trains image_segmentation, not 'image_classification'"):
        registry.parse_model_name("ghost/any", task=CLASSIFICATION)
    with pytest.raises(ModuleNotFoundError):  # the control: under its own task the import is reached
        registry.parse_model_name("ghost/any", task=SEGMENTATION)


def test_a_task_nobody_trains_says_so(world, monkeypatch):
    with pytest.raises(ConfigError, match="the families of 'no_such_task' are none registered"):
        registry.parse_model_name("vit/vit_pico", task="no_such_task")


def test_an_unknown_family_is_still_unknown_and_not_a_task_error(world):
    with pytest.raises(UnknownModelError, match="Unknown model family 'vitt'"):
        registry.parse_model_name("vitt/vit_pico", task=SEGMENTATION)


def test_the_classifier_refuses_a_segmentation_family_before_the_directory_exists(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="the family 'convunext' trains image_segmentation"):
        train_image_classification(
            out=str(out), model="convunext/tiny", data={"dataset": "cifar10"}, batch_size=4,
            analysis_args={"model_analysis": False},
        )
    assert not out.exists()


def test_a_task_that_is_not_classification_refuses_a_classifier_before_the_directory_exists(
    world, tmp_path
):
    """The other direction, through the trainer: the task word comes from the trainer class."""
    from train.api.config import validate
    from train.api.tasks.image_classification import ImageClassificationConfig

    class AsSegmenter(ImageClassificationTrainer):
        task = SEGMENTATION

    out = tmp_path / "run"
    config = validate(
        ImageClassificationConfig,
        {"out": str(out), "model": "vit/vit_pico", "data": {"dataset": "cifar10"}, "batch_size": 4,
         "analysis_args": {"model_analysis": False}},
    )
    with pytest.raises(ConfigError, match="the family 'vit' trains image_classification, not 'image_segmentation'"):
        AsSegmenter(config).run()
    assert not out.exists()


def test_the_command_line_exits_2_with_one_line_and_leaves_no_directory(world, tmp_path, capsys):
    out = tmp_path / "run"
    code = cli.main([CLASSIFICATION, f"out={out}", "model=accunet/base", "data.dataset=cifar10"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    lines = [line for line in captured.err.splitlines() if line.startswith("error:")]
    assert len(lines) == 1 and "the family 'accunet' trains image_segmentation" in lines[0]
    assert not out.exists()


def test_list_models_with_a_task_lists_that_tasks_families_only(world):
    seg = registry.list_models(SEGMENTATION)
    assert seg[0].startswith("accunet/") and all(n.partition("/")[0] in ("accunet", "convunext") for n in seg)
    assert "convunext/xlarge" in seg and "accunet/base" in seg
    assert not any(n.startswith(("vit/", "resnet/")) for n in seg)
    assert not any(n.startswith(("convunext/", "accunet/")) for n in registry.list_models(CLASSIFICATION))


def test_a_record_naming_a_family_that_does_not_train_its_task_is_refused_at_load(world, tmp_path):
    """A classification record under a segmentation family: refused as that, not as a type crash."""
    from train.api.predictor import build_task_record, load_model

    run = tmp_path / "run"
    run.mkdir()
    (run / "best_model.keras").write_bytes(b"not read: the task is checked first")
    record = build_task_record(
        task=CLASSIFICATION, model="convunext/tiny", family="convunext", variant="tiny",
        class_names=["a", "b"], input_shape=(32, 32, 3), normalize="unit", mean=[0.0] * 3,
        std=[1.0] * 3, data_kind="folder", source_channels=3,
    )
    (run / "task.json").write_text(json.dumps(record))
    with pytest.raises(ConfigError, match="the family 'convunext' trains image_segmentation, not 'image_classification'"):
        load_model(run)


# ---------------------------------------------------------------------
# ModelSpec.stride
# ---------------------------------------------------------------------


def base_spec(**changes):
    values = dict(family="f", factory="m:f", variants_attr="m:V", tasks=(SEGMENTATION,))
    values.update(changes)
    return ModelSpec(**values)


@pytest.mark.parametrize("stride", [0, -8, True, 2.5, "nocolon", "m:", ""])
def test_a_malformed_stride_is_refused_when_the_spec_is_built(stride):
    with pytest.raises(ValueError, match="stride"):
        base_spec(stride=stride)


def test_stride_of_is_an_int_a_function_or_none():
    assert base_spec().stride_of("v") is None
    assert base_spec(stride=16).stride_of("v", {"depth": 9}) == 16
    spec = base_spec(stride="train.api.segmentation_models:convunext_stride")
    assert spec.stride_of("tiny") == 8
    assert spec.stride_of("tiny", {"depth": 5}) == 32


def test_no_classifier_has_a_stride(world):
    for family in ("vit", "resnet", "convnext_v1", "convnext_v2", "cliffordnet", "cliffordnet_hier"):
        assert world[family].stride is None and world[family].stride_of("any") is None


# ---------------------------------------------------------------------
# The size rule, per family
# ---------------------------------------------------------------------

#: ConvUNext's depth per variant, written out (the test does not read the table it checks).
CONVUNEXT_DEPTHS = {"tiny": 3, "small": 3, "base": 4, "large": 4, "xlarge": 5}


@pytest.mark.parametrize("variant, depth", sorted(CONVUNEXT_DEPTHS.items()))
def test_convunext_needs_a_multiple_of_two_to_its_depth(world, variant, depth):
    stride = 2**depth
    spec = world["convunext"]
    assert spec.stride_of(variant, {}) == stride
    assert check_image_size(spec, variant, {}, stride * 4) == (stride * 4, stride * 4)
    assert check_image_size(spec, variant, {}, [stride * 4, stride * 8]) == (stride * 4, stride * 8)
    assert check_image_size(spec, variant, {}, (stride, stride * 2)) == (stride, stride * 2)
    with pytest.raises(ConfigError) as caught:
        check_image_size(spec, variant, {}, stride * 4 + 1)
    text = str(caught.value)
    assert text.startswith("Invalid value for 'transform_args.image_size': the height ")
    assert f"is not a multiple of {stride}, which convunext/{variant} needs" in text
    assert f"use {stride * 4} or {stride * 5}" in text


def test_a_convunext_depth_in_model_args_moves_the_rule(world):
    spec = world["convunext"]
    assert check_image_size(spec, "tiny", {}, 136) == (136, 136)  # 17 * 8
    with pytest.raises(ConfigError, match="not a multiple of 16"):
        check_image_size(spec, "tiny", {"depth": 4}, 136)
    assert check_image_size(spec, "tiny", {"depth": 2}, 132) == (132, 132)  # 33 * 4
    for bad in (0, -1, True, 2.0, "3"):
        with pytest.raises(ConfigError, match="model_args.depth"):
            convunext_stride("tiny", {"depth": bad})


def test_accunet_needs_a_multiple_of_sixteen(world):
    spec = world["accunet"]
    assert spec.stride == 16 == spec.stride_of("base")
    assert check_image_size(spec, "base", {}, 128) == (128, 128)
    assert check_image_size(spec, "base", {}, [128, 256]) == (128, 256)
    with pytest.raises(ConfigError, match=r"the width 250 is not a multiple of 16.*use 240 or 256"):
        check_image_size(spec, "base", {}, [128, 250])
    with pytest.raises(ConfigError, match=r"the height 100 is not a multiple of 16.*use 96 or 112"):
        check_image_size(spec, "base", {}, [100, 128])


def test_the_first_bad_side_is_named_and_a_side_smaller_than_the_stride_is_refused(world):
    spec = world["accunet"]
    with pytest.raises(ConfigError, match=r"the height 8 is not a multiple of 16.*use 16$"):
        check_image_size(spec, "base", {}, [8, 32])
    with pytest.raises(ConfigError) as caught:
        check_image_size(spec, "base", {}, [17, 19])
    assert "the height 17 " in str(caught.value) and "the width" not in str(caught.value)


@pytest.mark.parametrize("bad", [True, 0, -16, 1.5, "128", [128], [128, 128, 128], [128, True], [0, 128], [128, 12.8], None, {"h": 1}])
def test_a_value_that_is_not_one_int_or_two_is_refused_as_that(world, bad):
    with pytest.raises(ConfigError, match=r"a segmentation image_size is one positive int \(a square\) or \[height, width\]"):
        check_image_size(world["accunet"], "base", {}, bad)


def test_a_classifier_has_no_size_rule_and_asking_is_a_bug_not_a_request(world):
    with pytest.raises(ValueError) as caught:
        check_image_size(world["vit"], "vit_pico", {}, 32)
    assert not isinstance(caught.value, ConfigError)


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------


def test_the_variants_the_user_sees_are_the_models_own_tables(world):
    from dl_techniques.models.vision.convunext.model import CONVUNEXT_CONFIGS

    convunext = [n for n in registry.list_models(SEGMENTATION) if n.startswith("convunext/")]
    assert convunext == sorted(f"convunext/{v}" for v in CONVUNEXT_CONFIGS)
    assert [n for n in registry.list_models(SEGMENTATION) if n.startswith("accunet/")] == ["accunet/base"]
    assert ACCUNET_VARIANTS["base"] == {"base_filters": 32, "mlfc_iterations": 3}


def test_the_convunext_adapter_builds_what_the_legacy_trainer_builds():
    from dl_techniques.models.vision.convunext.model import create_convunext_variant

    shape = (32, 64, 3)
    ours = create_convunext(variant="tiny", num_classes=5, input_shape=shape)
    legacy = create_convunext_variant(
        "tiny", shape, use_bias=True, output_channels=5, final_activation="linear"
    )
    assert ours.count_params() == legacy.count_params()
    assert json.dumps(ours.get_config(), sort_keys=True, default=repr) == json.dumps(
        legacy.get_config(), sort_keys=True, default=repr
    )
    assert tuple(ours.output_shape) == (None, 32, 64, 5)
    assert any(getattr(layer, "use_bias", False) for layer in ours.layers)  # the biased arm
    logits = ours(np.random.default_rng(0).random((2, 32, 64, 3), np.float32), training=False)
    assert float(np.abs(np.asarray(logits).sum(-1) - 1.0).max()) > 0.1  # linear: not a softmax


def test_the_accunet_adapter_builds_what_its_trainer_builds():
    from dl_techniques.models.vision.accunet.model import create_acc_unet_multiclass

    ours = create_accunet(variant="base", num_classes=5, input_shape=(32, 48, 3))
    legacy = create_acc_unet_multiclass(
        input_channels=3, num_classes=5, input_shape=(32, 48), base_filters=32, mlfc_iterations=3
    )
    assert ours.count_params() == legacy.count_params()
    assert tuple(ours.output_shape) == (None, 32, 48, 5)


def test_a_model_arg_overrides_the_variants_table_for_accunet():
    small = create_accunet(variant="base", num_classes=3, input_shape=(32, 32, 3), base_filters=8)
    default = create_accunet(variant="base", num_classes=3, input_shape=(32, 32, 3))
    assert small.count_params() < default.count_params() / 4


def test_the_variants_table_is_what_the_accunet_adapter_reads(monkeypatch):
    """The one shipped row equals the factory's defaults, so a second row makes the lookup observable."""
    monkeypatch.setitem(ACCUNET_VARIANTS, "narrow", {"base_filters": 8, "mlfc_iterations": 1})
    narrow = create_accunet(variant="narrow", num_classes=3, input_shape=(32, 32, 3))
    default = create_accunet(variant="base", num_classes=3, input_shape=(32, 32, 3))
    assert narrow.count_params() < default.count_params() / 8


@pytest.mark.parametrize(
    "build, key",
    [
        (create_convunext, "use_bias"),
        (create_convunext, "final_activation"),
        (create_convunext, "output_channels"),
        (create_accunet, "input_channels"),
    ],
)
def test_a_key_the_adapter_sets_itself_is_refused_naming_it(build, key):
    with pytest.raises(ValueError, match=rf"'{key}' is set by the .* adapter"):
        build(variant="tiny" if build is create_convunext else "base", num_classes=3,
              input_shape=(32, 32, 3), **{key: 1})


def test_a_refusal_of_the_adapter_reaches_the_user_as_a_config_error(world):
    with pytest.raises(ConfigError, match="'final_activation' is set by the 'convunext' adapter"):
        recipes.build_model(world["convunext"], "tiny", 3, (32, 32, 3), {"final_activation": "softmax"})


def test_an_unknown_accunet_variant_and_a_one_class_model_are_refused():
    with pytest.raises(ValueError, match="Unknown variant 'huge'"):
        create_accunet(variant="huge", num_classes=3, input_shape=(32, 32, 3))
    with pytest.raises(ValueError, match="num_classes must be > 1"):
        create_accunet(variant="base", num_classes=1, input_shape=(32, 32, 3))
    with pytest.raises(ValueError, match="input_shape must be"):
        create_convunext(variant="tiny", num_classes=3, input_shape=(32, 32))


# ---------------------------------------------------------------------
# Every family builds at a non-square size and returns the input's size
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, variant, size",
    [
        ("convunext", "tiny", [128, 256]),
        ("convunext", "tiny", [256, 128]),
        ("accunet", "base", [128, 256]),
        ("accunet", "base", [64, 128]),
    ],
)
def test_each_family_builds_at_a_non_square_size_the_rule_allows_with_output_equal_to_input(
    world, family, variant, size
):
    spec = world[family]
    height, width = check_image_size(spec, variant, {}, size)
    model = recipes.build_model(spec, variant, 4, (height, width, 3), {})
    out = model(np.zeros((1, height, width, 3), np.float32), training=False)
    assert tuple(out.shape) == (1, height, width, 4)


def test_accunet_itself_refuses_what_its_rule_refuses_and_convunext_does_not(world):
    """Pins the measured facts the module docstring states (scratch_step23/logs/p1_build.txt)."""
    # Keras reports a build that raised as a UserWarning (an error under this tree's pytest filter)
    # and then lets the call raise the ValueError; either way the model's own text is what is said.
    with pytest.raises((ValueError, UserWarning), match="divisible by 16"):
        model = recipes.build_model(world["accunet"], "base", 3, (100, 128, 3), {})
        model(np.zeros((1, 100, 128, 3), np.float32), training=False)
    # ConvUNext resizes its upsampled map to the skip's size, so the stride rule is the task's choice.
    model = recipes.build_model(world["convunext"], "tiny", 3, (36, 64, 3), {})
    assert tuple(model(np.zeros((1, 36, 64, 3), np.float32), training=False).shape) == (1, 36, 64, 3)
    with pytest.raises(ConfigError, match="not a multiple of 8"):
        check_image_size(world["convunext"], "tiny", {}, [36, 64])
