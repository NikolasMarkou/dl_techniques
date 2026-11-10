"""The five image-denoising families: what they build, what they refuse, and how a record reads them back.

``train.api.denoising_models`` (plan-2026-11-10T095307-d483ffab step 7, D-010 and D-015) adapts three bias-free
denoisers (``bfcnn``, ``bfunet``, ``bfconvunext``: ``properties=("bias_free",)``) and two biased ones
(``convunext_biased``, ``scunet``) to the factory call the API makes. This file pins, per family:

* the listing: every variant by literal name, the task, the properties, the stride rule (a literal table, read
  from the library's variant tables, not from the module under test);
* the refusals: one row per family and refused key, run through the real command line machinery in a child
  process: exit 2, the run directory ``out`` absent, a message naming the key and the rule, and not one
  TensorFlow device initialised (the same recorder as ``test_no_check_before_setup_gpu_initialises_tensorflow``,
  with its control arm); the signature census, which fails when a factory gains a parameter no table classifies;
  the same keys on ``convunext_biased``, which build;
* the build: flexible input at two sizes, an output the size of the input, ONE output, a LeakyReLU of slope 0.1
  (the bare string is Keras' 0.2), a float32 output under ``mixed_float16``, an exact ``.keras`` round trip;
* the record: ``observe`` reads the variant back, so a wrong variant or a wrong family is refused LOUDLY (a
  module-level table with no ``observe`` is checked by nothing and passes silently);
* the recipes: the legacy optimizer, and every variant's recipe passes the refusals it makes.

No task is registered for ``image_denoising`` yet (the task row is step 8), so ``list_models task=`` is exercised
through a TaskSpec this test registers in a patched registry. Everything writes under ``tmp_path``.
"""

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.api import denoising_models as dm
from train.api import predictor, recipes, registry
from train.api.args import AutoContext, OptimArgs
from train.api.errors import ConfigError, ConfigValidationError
from train.api.segmentation_models import check_image_size
from train.common.denoising_homogeneity import probe_numeric, scan_static

REPO_ROOT = Path(__file__).resolve().parents[3]

TASK = "image_denoising"
FAMILIES = ("bfcnn", "bfunet", "bfconvunext", "convunext_biased", "scunet")
BIAS_FREE = ("bfcnn", "bfunet", "bfconvunext")
BIASED = ("convunext_biased", "scunet")

#: Every variant of every denoising family, by literal name. 5 + 5 + 5 + 5 + 3 = 23. The four library tables
#: (``BFCNN_CONFIGS``, ``BFUNET_CONFIGS``, ``CONVUNEXT_CONFIGS``) hold these five names; SCUNet's is this plan's own.
VARIANTS = {
    "bfcnn": ["tiny", "small", "base", "large", "xlarge"],
    "bfunet": ["tiny", "small", "base", "large", "xlarge"],
    "bfconvunext": ["tiny", "small", "base", "large", "xlarge"],
    "convunext_biased": ["tiny", "small", "base", "large", "xlarge"],
    "scunet": ["tiny", "small", "base"],
}
ALL_NAMES = sorted(f"{family}/{variant}" for family, names in VARIANTS.items() for variant in names)

#: The multiple each side of the input must be, by (family, variant); ``None`` is no rule (``bfcnn`` takes any size,
#: ``scunet`` pads to 64 inside). ``2 ** depth`` with the depth of the library's table: 3, 3, 4, 4, 5.
STRIDES = {
    "bfcnn": dict.fromkeys(VARIANTS["bfcnn"]),
    "bfunet": {"tiny": 8, "small": 8, "base": 16, "large": 16, "xlarge": 32},
    "bfconvunext": {"tiny": 8, "small": 8, "base": 16, "large": 16, "xlarge": 32},
    "convunext_biased": {"tiny": 8, "small": 8, "base": 16, "large": 16, "xlarge": 32},
    "scunet": dict.fromkeys(VARIANTS["scunet"]),
}

#: What each family's own layers show for ``tiny``: the keys ``observe`` returns and their values.
OBSERVED = {
    "bfcnn": {"tiny": {"num_blocks": 2, "filters": 32}},
    "bfunet": {"tiny": {"depth": 3, "initial_filters": 16, "blocks_per_level": 2}},
    "bfconvunext": {"tiny": {"depth": 3, "initial_filters": 32, "convnext_version": "v1", "blocks_per_level": 2}},
    "convunext_biased": {"tiny": {"depth": 3, "initial_filters": 32, "convnext_version": "v2", "blocks_per_level": 2}},
    "scunet": {"tiny": {"config": [1] * 7, "dim": 32, "head_dim": 16}},
}

#: The legacy denoiser's optimizer, one literal row (``train.bfunet.common.TrainingConfig``).
LEGACY_OPTIM = dict(
    optimizer="adamw", lr=1e-3, weight_decay=4e-3, schedule="cosine_decay", warmup_epochs=10, min_lr_ratio=0.01,
    decay_over="total", clipnorm=None, global_clipnorm=1.0,
)

SIZES = ((32, 32), (24, 40))


@pytest.fixture
def registered(monkeypatch):
    """A registry holding only the built-in families."""
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    return registry.MODELS


def context(family: str, variant: str = "tiny") -> AutoContext:
    spec = registry.MODELS[family]
    return AutoContext(
        family=family, variant=variant, dataset="folder", num_classes=None, image_size=None,
        defaults=spec.recipe("folder", variant),
    )


def resolved(family: str, variant: str = "tiny", **model_args: Any) -> Dict[str, Any]:
    """The factory keywords the task's ``check_model_args`` would get from the resolver."""
    ctx = context(family, variant)
    spec = registry.MODELS[family]
    return spec.load_resolver()(ctx, OptimArgs().resolve_auto(ctx), model_args)


def build(family: str, variant: str = "tiny", input_shape=(None, None, 3), **model_args: Any):
    """Build through the real path: recipe, resolver, ``recipes.build_model``."""
    spec = registry.MODELS[family]
    return recipes.build_model(spec, variant, None, input_shape, resolved(family, variant, **model_args))


@pytest.fixture(scope="module")
def models():
    """One flexible tiny model per family, built once, from the recipe (a module cache: builds are the cost).

    The registry is patched for the whole module and restored after it; a test that needs the registry itself
    (``registered``) patches it again for its own length.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield {family: build(family) for family in FAMILIES}


def batch(height: int, width: int, channels: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(1, height, width, channels)).astype("float32")


def dtype_of(tensor) -> str:
    return keras.backend.standardize_dtype(tensor.dtype)


def leaky_slopes(model) -> List[float]:
    return sorted(
        {round(float(layer.negative_slope), 6)
         for layer in model._flatten_layers(include_self=False, recursive=True)
         if isinstance(layer, keras.layers.LeakyReLU)}
    )


# ---------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------


def test_the_listing_is_the_literal_variant_tables(registered):
    assert [family for family in FAMILIES if family in registered] == list(FAMILIES)
    for family, names in VARIANTS.items():
        assert sorted(registered[family].load_variants()) == sorted(names), family
    listed = [name for name in registry.list_models() if name.partition("/")[0] in FAMILIES]
    assert listed == ALL_NAMES
    assert len(ALL_NAMES) == 23


def test_five_families_list_under_the_task_filter(registered):
    # The TaskSpec row of the task is step 8's: `list_models(task=)` refuses a task no row names, so the row is
    # registered here, in the patched registry, with the strings step 8 will use.
    registry.register_task(registry.TaskSpec(
        TASK, "train.api.tasks.image_denoising:ImageDenoisingConfig",
        "train.api.tasks.image_denoising:train_image_denoising", "image denoising",
    ))
    registry.register_task(registry.TaskSpec("image_classification", "x.y:Config", "x.y:run", "classification"))
    assert registry.list_models(TASK) == ALL_NAMES
    families = sorted({name.partition("/")[0] for name in registry.list_models(TASK)})
    assert families == sorted(FAMILIES)
    # Control: no denoising name leaks into another task's listing, and no other family into this one.
    assert not [name for name in registry.list_models("image_classification") if name.startswith(tuple(f"{f}/" for f in FAMILIES))]
    assert all(registered[family].tasks == (TASK,) for family in FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
def test_only_the_bias_free_families_declare_the_property(family, registered):
    assert registered[family].properties == (("bias_free",) if family in BIAS_FREE else ())


def test_a_denoising_family_is_refused_under_another_task(registered):
    with pytest.raises(ConfigError, match="trains image_denoising, not 'image_classification'"):
        registry.parse_model_name("bfunet/tiny", task="image_classification")
    with pytest.raises(ConfigError, match="trains image_classification, not 'image_denoising'"):
        registry.parse_model_name("vit/vit_pico", task=TASK)


def test_importing_the_module_and_listing_the_specs_imports_no_model_code():
    code = (
        "import sys\n"
        "from train.api import denoising_models as dm, recipes, registry\n"
        "specs = dm.builtin_denoising_models()\n"
        "recipes.register_builtin_models()\n"
        "assert set(registry.MODELS) >= {s.family for s in specs}\n"
        "loaded = sorted(m for m in ('keras', 'tensorflow', 'dl_techniques') if m in sys.modules)\n"
        "print('LOADED', loaded)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"


# ---------------------------------------------------------------------
# The refusals, through the real command line, in a child process
# ---------------------------------------------------------------------

#: One row per family and refused key: ``(family, arguments, fragments of the message)``. Each fragment names the
#: key AND the rule that refused it (``is not offered by`` for a key refused whatever its value, ``is refused for`` for
#: a value outside its rule, ``sets it to`` for a key the adapter forces, ``Unknown key`` for a key the table
#: does not name), so a dropped row, which turns into an unknown-key refusal, fails the row by its message.
REFUSALS: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    ("bfcnn", ("model_args.normalization_type=layernorm",), ("model_args.normalization_type", "is refused for 'bfcnn'")),
    ("bfcnn", ("model_args.activation=gelu",), ("model_args.activation", "is refused for 'bfcnn'")),
    ("bfcnn", ("model_args.final_activation=tanh",), ("model_args.final_activation", "is refused for 'bfcnn'")),
    ("bfcnn", ("model_args.no_such_key=1",), ("Unknown key 'model_args.no_such_key'",)),
    ("bfcnn", ("model_args.trainable_gabor_stem=false",), ("Unknown key 'model_args.trainable_gabor_stem'",)),
    ("bfunet", ("model_args.block_normalization=layernorm",), ("model_args.block_normalization", "is refused for 'bfunet'")),
    ("bfunet", ("model_args.activation=gelu",), ("model_args.activation", "is refused for 'bfunet'")),
    ("bfunet", ("model_args.final_activation=tanh",), ("model_args.final_activation", "is refused for 'bfunet'")),
    ("bfunet", ("model_args.gabor_activation=gelu",), ("model_args.gabor_activation", "is refused for 'bfunet'")),
    ("bfunet", ("model_args.enable_deep_supervision=true",), ("model_args.enable_deep_supervision", "sets it to False")),
    ("bfunet", ("model_args.expose_bottleneck=true",), ("model_args.expose_bottleneck", "is not offered by 'bfunet'")),
    ("bfunet", ("model_args.pretrained=true",), ("model_args.pretrained", "is not offered by 'bfunet'")),
    ("bfunet", ("model_args.weights_dataset=general_denoising",), ("model_args.weights_dataset", "is not offered by 'bfunet'")),
    ("bfunet", ("model_args.high_freq_blocks=2",), ("model_args.high_freq_blocks", "do nothing without")),
    (
        "bfunet", ("model_args.use_gabor_stem=false", "model_args.trainable_gabor_stem=false"),
        ("model_args.trainable_gabor_stem", "builds none unless"),
    ),
    ("bfconvunext", ("model_args.use_bias=false",), ("model_args.use_bias", "is not offered by 'bfconvunext'", "convunext_biased")),
    ("bfconvunext", ("model_args.use_bias=true",), ("model_args.use_bias", "is not offered by 'bfconvunext'")),
    ("bfconvunext", ("model_args.stem_normalization=layer_norm",), ("model_args.stem_normalization", "is not offered by 'bfconvunext'")),
    ("bfconvunext", ("model_args.block_normalization=layernorm",), ("model_args.block_normalization", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.block_activation=gelu",), ("model_args.block_activation", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.stem_activation=gelu",), ("model_args.stem_activation", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.supervision_activation=gelu",), ("model_args.supervision_activation", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.final_activation=tanh",), ("model_args.final_activation", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.gabor_activation=gelu",), ("model_args.gabor_activation", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.supervision_norm_center=true",), ("model_args.supervision_norm_center", "is refused for 'bfconvunext'")),
    ("bfconvunext", ("model_args.enable_deep_supervision=true",), ("model_args.enable_deep_supervision", "sets it to False")),
    ("bfconvunext", ("model_args.include_top=false",), ("model_args.include_top", "sets it to True")),
    ("bfconvunext", ("model_args.output_channels=1",), ("model_args.output_channels", "is not offered by 'bfconvunext'")),
    ("bfconvunext", ("model_args.expose_bottleneck=true",), ("model_args.expose_bottleneck", "is not offered by 'bfconvunext'")),
    ("bfconvunext", ("model_args.high_freq_blocks=2",), ("model_args.high_freq_blocks", "do nothing without")),
    ("convunext_biased", ("model_args.use_bias=false",), ("model_args.use_bias", "sets it to True", "bfconvunext")),
    ("convunext_biased", ("model_args.enable_deep_supervision=true",), ("model_args.enable_deep_supervision", "sets it to False")),
    ("convunext_biased", ("model_args.include_top=false",), ("model_args.include_top", "sets it to True")),
    ("convunext_biased", ("model_args.output_channels=1",), ("model_args.output_channels", "is not offered by 'convunext_biased'")),
    ("convunext_biased", ("model_args.expose_bottleneck=true",), ("model_args.expose_bottleneck", "is not offered by 'convunext_biased'")),
    ("convunext_biased", ("model_args.high_freq_blocks=2",), ("model_args.high_freq_blocks", "do nothing without")),
    ("scunet", ("model_args.window_size=4",), ("model_args.window_size", "is not offered by 'scunet'")),
    ("scunet", ("model_args.include_top=false",), ("model_args.include_top", "sets it to True")),
    ("scunet", ("model_args.in_nc=1",), ("model_args.in_nc", "is not offered by 'scunet'")),
    ("scunet", ("model_args.no_such_key=1",), ("Unknown key 'model_args.no_such_key'",)),
    # The names the library supplies are refused by the shared rule, before any table here is read.
    *[(family, ("model_args.input_shape=[8,8,3]",), ("'input_shape'", "is set by the library")) for family in FAMILIES],
]

#: Requests that must NOT be refused (the control of the refusals: a driver that refused everything would pass the
#: rows above): the recipe as it is, and the keys a family does offer.
ACCEPTED: List[Tuple[str, Tuple[str, ...]]] = [
    ("bfcnn", ()),
    ("bfunet", ("model_args.depth=2",)),
    ("bfunet", ("model_args.use_laplacian_pyramid=true", "model_args.high_freq_blocks=1")),
    ("bfunet", ("model_args.enable_deep_supervision=false", "model_args.activation=relu")),
    ("bfconvunext", ("model_args.use_gabor_stem=false", "model_args.convnext_version=v2")),
    ("bfconvunext", ("model_args.supervision_norm_center=false", "model_args.final_activation=linear")),
    ("convunext_biased", ("model_args.use_bias=true", "model_args.block_activation=gelu")),
    ("scunet", ("model_args.include_top=true", "model_args.dim=32")),
]

CHILD_MODULE = "fake_denoising_refusal_task"

CHILD_SOURCE = '''
import dataclasses
import os
import typing
from types import SimpleNamespace

from train.api import registry
from train.api.args import AutoContext, OptimArgs
from train.api.config import StrictConfig

TASK = "image_denoising"


@dataclasses.dataclass(frozen=True)
class StubConfig(StrictConfig):
    out: str = dataclasses.field(metadata={"help": "where to write"})
    model: str = dataclasses.field(metadata={"help": "family/variant"})
    model_args: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict, metadata={"help": "free keys"})


def run_stub(*, out, model, model_args=None):
    """What ``TaskTrainer`` does before the claim: parse the model, resolve its arguments, THEN create ``out``."""
    family, variant = registry.parse_model_name(model, task=TASK)
    spec = registry.MODELS[family]
    context = AutoContext(
        family=family, variant=variant, dataset="folder", num_classes=None, image_size=None,
        defaults=spec.recipe("folder", variant),
    )
    spec.load_resolver()(context, OptimArgs().resolve_auto(context), model_args or {})
    os.makedirs(out)
    return SimpleNamespace(success=True)
'''

CHILD_DRIVER = textwrap.dedent(
    """
    import contextlib
    import io
    import json
    import os
    import sys

    from tensorflow.python.eager import context as eager

    sys.path.insert(0, sys.argv[2])
    from train.api import cli
    from train.api.registry import TaskSpec, register_task

    def initialised():
        return eager.context()._context_handle is not None

    register_task(TaskSpec("fake_denoising_refusals", "MODULE:StubConfig", "MODULE:run_stub", "a stub"))
    rows = json.load(open(sys.argv[1]))
    seen = {"at_start": initialised(), "results": []}
    for index, row in enumerate(rows):
        out = os.path.join(sys.argv[3], f"run_{index}")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.main([
                "fake_denoising_refusals", f"out={out}", f"model={row['family']}/{row.get('variant', 'tiny')}",
                *row["args"],
            ])
        seen["results"].append({"code": code, "stderr": err.getvalue(), "out_exists": os.path.exists(out)})
    seen["initialised"] = initialised()
    import keras

    keras.ops.zeros(1)
    seen["after_op"] = initialised()
    print("SEEN " + json.dumps(seen))
    """
).replace("MODULE", CHILD_MODULE)


@pytest.fixture(scope="module")
def refusal_run(tmp_path_factory):
    """Every refusal row and every accepted control through ``cli.main`` in ONE child process."""
    work = tmp_path_factory.mktemp("refusals")
    source = work / "source"
    source.mkdir()
    (source / f"{CHILD_MODULE}.py").write_text(CHILD_SOURCE)
    rows = [{"family": f, "args": list(a)} for f, a, _ in REFUSALS] + [{"family": f, "args": list(a)} for f, a in ACCEPTED]
    (work / "rows.json").write_text(json.dumps(rows))
    runs = work / "runs"
    runs.mkdir()
    env = {**os.environ, "MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1",
           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-c", CHILD_DRIVER, str(work / "rows.json"), str(source), str(runs)],
        capture_output=True, text=True, timeout=900, cwd=work, env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    line = [row for row in done.stdout.splitlines() if row.startswith("SEEN ")][-1]
    return json.loads(line[len("SEEN "):])


@pytest.mark.parametrize(
    "family, arguments, fragments", REFUSALS, ids=[f"{f}:{' '.join(a)}" for f, a, _ in REFUSALS]
)
def test_every_refused_key_exits_two_with_out_absent_naming_the_key_and_the_rule(
    family, arguments, fragments, refusal_run
):
    index = [(f, a) for f, a, _ in REFUSALS].index((family, arguments))
    result = refusal_run["results"][index]
    assert result["code"] == 2, result["stderr"][-800:]
    assert not result["out_exists"], "the run directory was claimed before the refusal"
    assert result["stderr"].startswith("error: "), result["stderr"]
    assert "Traceback" not in result["stderr"]
    for fragment in fragments:
        assert fragment in result["stderr"], (fragment, result["stderr"])


@pytest.mark.parametrize("index", range(len(ACCEPTED)), ids=[f"{f}:{' '.join(a) or 'recipe'}" for f, a in ACCEPTED])
def test_the_keys_a_family_offers_are_not_refused(index, refusal_run):
    result = refusal_run["results"][len(REFUSALS) + index]
    assert result["code"] == 0, result["stderr"][-800:]
    assert result["out_exists"], "the stub claims out only after the resolver accepted the request"


def test_no_refusal_and_no_resolution_initialised_a_tensorflow_device(refusal_run):
    # The control arm: the same child runs one op afterwards, and the recorder must see it. A TensorFlow release
    # that moves the attribute turns this red instead of turning the assertion above blind.
    assert refusal_run["at_start"] is False
    assert refusal_run["initialised"] is False
    assert refusal_run["after_op"] is True


# ---------------------------------------------------------------------
# The refusal tables and the signature census
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_classifies_every_parameter_of_every_factory(family):
    census = dm.signature_census(family)
    assert census == {"unclassified": [], "stale": []}, census


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_fails_when_a_factory_gains_a_parameter_no_table_names(family):
    real = dm.factory_signatures(family)
    assert dm.signature_census(family, real) == {"unclassified": [], "stale": []}
    # A factory that gained `new_option`: the census names it.
    grown = {path: [*names, "new_option"] for path, names in real.items()}
    assert dm.signature_census(family, grown)["unclassified"] == ["new_option"]
    # A table key no factory has any more: the census names that too.
    gone = sorted(dm.ALLOWED[family])[0]
    shrunk = {path: [name for name in names if name != gone] for path, names in real.items()}
    assert dm.signature_census(family, shrunk)["stale"] == [gone]


def test_the_three_classes_do_not_overlap():
    for family in FAMILIES:
        classes = [set(dm.ALLOWED[family]), set(dm.REFUSED[family]), set(dm.FORCED[family])]
        assert not (classes[0] & classes[1]) and not (classes[0] & classes[2]) and not (classes[1] & classes[2]), family


def test_only_the_bias_free_families_have_value_rules():
    for family in BIASED:
        assert not [key for key, (kind, _) in dm.REFUSED[family].items() if kind != dm.NEVER], family
    for family in BIAS_FREE:
        assert any(kind != dm.NEVER for kind, _ in dm.REFUSED[family].values()), family


@pytest.mark.parametrize("family", BIAS_FREE)
def test_the_activation_rule_is_the_librarys_allowlist_not_a_copy(family):
    from dl_techniques.models.vision.convunext.model import POSITIVELY_HOMOGENEOUS_ACTIVATIONS

    assert set(dm._accepted(dm.ACTIVATION)) == set(POSITIVELY_HOMOGENEOUS_ACTIVATIONS)
    for value in POSITIVELY_HOMOGENEOUS_ACTIVATIONS:
        key = next(key for key, (kind, _) in dm.REFUSED[family].items() if kind == dm.ACTIVATION)
        assert dm.problems_of(family, {key: value}) == [], (key, value)
    key = next(key for key, (kind, _) in dm.REFUSED[family].items() if kind == dm.ACTIVATION)
    for value in ("gelu", "swish", "tanh", 1, ["relu"]):
        assert dm.problems_of(family, {key: value}), (key, value)


def test_the_same_keys_bfconvunext_refuses_build_on_the_biased_convunext(registered):
    # The refused values of bfconvunext, handed to the biased family: its table offers each, and the model builds.
    refused_on_bias_free = {
        "block_normalization": "layernorm", "block_activation": "gelu", "stem_activation": "gelu",
        "supervision_activation": "gelu", "final_activation": "tanh", "gabor_activation": "gelu",
        "supervision_norm_center": True, "stem_normalization": "global_response_norm", "use_gabor_stem": True,
    }
    assert len(dm.problems_of("bfconvunext", refused_on_bias_free)) == len(refused_on_bias_free) - 1  # control
    assert dm.problems_of("convunext_biased", refused_on_bias_free) == []
    model = build("convunext_biased", input_shape=(32, 32, 3), **refused_on_bias_free)
    assert model(batch(32, 32)).shape == (1, 32, 32, 3)
    # The biased family refuses only what its factory cannot take.
    assert dm.problems_of("convunext_biased", {"use_bias": False})


# ---------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_flexible_model_runs_at_two_sizes_with_one_output_of_the_inputs_size(family, models):
    model = models[family]
    assert len(model.outputs) == 1
    assert tuple(model.input_shape) == (None, None, None, 3)
    assert tuple(model.output_shape) == (None, None, None, 3)
    for height, width in SIZES:
        output = model(batch(height, width), training=False)
        assert tuple(output.shape) == (1, height, width, 3), (family, height, width)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_head_is_linear_and_the_output_float32(family, models):
    model = models[family]
    last = model.layers[-1]
    assert type(last).__name__ == "Activation" and last.name == "output_float32"
    assert last.get_config()["activation"] == "linear"
    assert last.dtype_policy.name == "float32"
    assert dtype_of(model(batch(32, 32))) == "float32"


@pytest.mark.parametrize("family", FAMILIES)
def test_the_output_is_float32_under_mixed_float16_where_the_library_emits_float16(family, registered):
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        model = build(family)
        output = model(batch(32, 32), training=False)
        assert dtype_of(output) == "float32"
        # Control: the tensor the cast reads is float16, so without the layer the loss and the metrics would be
        # computed on half precision (D-019 measured every family doing so).
        assert dtype_of(model.layers[-2].output) == "float16", family
    finally:
        keras.config.set_dtype_policy(previous.name if hasattr(previous, "name") else previous)


@pytest.mark.parametrize("family", ("bfcnn", "bfunet", "bfconvunext"))
def test_the_leaky_relu_has_slope_one_tenth_not_the_strings_two_tenths(family, models):
    # DECISION plan_2026-06-21_eb7fd829/D-003: the string "leaky_relu" builds Keras' slope 0.2 and no LeakyReLU layer
    # at all; the instance carries the legacy 0.1. An empty list (the string) fails here as a 0.2 would.
    assert leaky_slopes(models[family]) == [dm.LEAKY_RELU_SLOPE] == [0.1]


def test_a_leaky_relu_named_for_the_biased_convunext_is_also_slope_one_tenth(registered):
    model = build("convunext_biased", block_activation="leaky_relu", stem_activation="leaky_relu")
    assert leaky_slopes(model) == [0.1]


@pytest.mark.parametrize("family", FAMILIES)
def test_the_keras_round_trip_is_exact_and_keeps_the_slope_and_the_variant(family, models, tmp_path):
    model = models[family]
    path = tmp_path / f"{family}.keras"
    model.save(path)
    loaded = keras.saving.load_model(path, compile=False)
    for height, width in SIZES:
        x = batch(height, width, seed=height)
        np.testing.assert_array_equal(
            np.asarray(model(x, training=False)), np.asarray(loaded(x, training=False)), err_msg=family
        )
    assert leaky_slopes(loaded) == leaky_slopes(model)
    assert len(loaded.outputs) == 1
    assert registry.MODELS[family].load_observer()(loaded) == OBSERVED[family]["tiny"]


@pytest.mark.parametrize("family", FAMILIES)
def test_observe_reads_the_variant_back(family, models, registered):
    observer = registered[family].load_observer()
    assert observer is not None, f"{family} has no observe: its record would be checked by nothing"
    assert observer(models[family]) == OBSERVED[family]["tiny"]


def record_of(family: str, **changes: Any) -> Dict[str, Any]:
    """The record fields the model check reads, with the variant overrides a tiny run records."""
    overrides = predictor.overrides_of_variant(registry.MODELS[family].load_variants()["tiny"], resolved(family, "tiny"))
    return {"family": family, "variant": "tiny", **({"variant_overrides": overrides} if overrides else {}), **changes}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_record_check_passes_the_honest_record_and_refuses_a_wrong_variant_loudly(family, models, registered):
    spec = registered[family]
    here = Path("model.keras"), Path("task.json")
    predictor.check_model_against_record(models[family], record_of(family), spec, *here)
    # A file of `tiny` under a record naming `small`: refused. Without `observe` (a module-level table has no class to
    # be an instance of) the check reads nothing and this call returns, which is the failure the test exists for.
    with pytest.raises(ConfigError, match="is not that variant"):
        predictor.check_model_against_record(
            models[family], record_of(family, variant="small", variant_overrides={}), spec, *here
        )


@pytest.mark.parametrize("family, other", [("bfcnn", "bfunet"), ("bfunet", "bfcnn"), ("bfunet", "bfconvunext"),
                                           ("bfconvunext", "bfunet"), ("bfcnn", "scunet"), ("scunet", "bfcnn")])
def test_a_model_of_another_family_is_refused_by_the_record_check_not_passed_silently(family, other, models, registered):
    spec = registered[other]
    record = {"family": other, "variant": "tiny"}
    with pytest.raises(ConfigError, match="holds no layer of that family"):
        predictor.check_model_against_record(models[family], record, spec, Path("model.keras"), Path("task.json"))


def test_the_recipe_v1_against_the_tables_v2_is_a_recorded_override_only_for_bfconvunext(registered):
    for family in FAMILIES:
        spec = registered[family]
        for variant in VARIANTS[family]:
            expected = {"convnext_version": "v1"} if family == "bfconvunext" else {}
            overrides = predictor.overrides_of_variant(spec.load_variants()[variant], resolved(family, variant))
            assert overrides == expected, (family, variant)


@pytest.mark.parametrize("family", ("bfunet", "bfconvunext", "convunext_biased"))
def test_a_stem_frozen_by_the_key_is_not_trainable_before_compile(family, registered):
    trainable = build(family, input_shape=(32, 32, 3), use_gabor_stem=True)
    frozen = build(family, input_shape=(32, 32, 3), use_gabor_stem=True, trainable_gabor_stem=False)
    stem = lambda model: next(layer for layer in model.layers if layer.name == "gabor_stem")
    ids = lambda model: {id(w) for w in model.trainable_weights}
    assert all(id(w) in ids(trainable) for w in stem(trainable).weights)
    assert stem(frozen).weights and not any(id(w) in ids(frozen) for w in stem(frozen).weights)
    assert len(frozen.trainable_weights) == len(trainable.trainable_weights) - len(stem(trainable).weights)


@pytest.mark.parametrize("family", ("bfunet", "bfconvunext"))
def test_one_channel_images_build_with_the_gabor_stem(family, registered):
    model = build(family, input_shape=(None, None, 1))
    assert any(layer.name == "gabor_stem" for layer in model.layers)
    assert tuple(model(batch(32, 32, channels=1)).shape) == (1, 32, 32, 1)


@pytest.mark.parametrize("family", ("bfcnn", "convunext_biased", "scunet"))
def test_one_channel_images_build_for_the_other_families(family, registered):
    model = build(family, input_shape=(None, None, 1))
    assert tuple(model(batch(32, 32, channels=1)).shape) == (1, 32, 32, 1)


# ---------------------------------------------------------------------
# Bias-free families stay homogeneous through the adapters (no refusal for v2 or the missing Gabor stem, D-015)
# ---------------------------------------------------------------------

HOMOGENEOUS_CASES = [
    ("bfcnn", {}), ("bfunet", {}), ("bfunet", {"use_gabor_stem": False}), ("bfconvunext", {}),
    ("bfconvunext", {"use_gabor_stem": False}), ("bfconvunext", {"convnext_version": "v2"}),
    ("bfconvunext", {"use_gabor_stem": False, "convnext_version": "v2"}),
]


@pytest.mark.parametrize("family, model_args", HOMOGENEOUS_CASES, ids=[f"{f}:{sorted(a.items())}" for f, a in HOMOGENEOUS_CASES])
def test_a_bias_free_family_scans_clean_and_reads_homogeneous_whatever_the_stem_and_block_version(family, model_args, registered):
    assert dm.problems_of(family, {**resolved(family), **model_args}) == []
    model = build(family, input_shape=(None, None, 3), **model_args)
    assert scan_static(model)["passed"], scan_static(model)["offenders"]
    report = probe_numeric(model, batch(32, 32, seed=3))
    assert report["passed"], report


# ---------------------------------------------------------------------
# Refusals of the adapter itself, and the resolver
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_class_count_is_refused_by_the_adapter_and_by_build_model(family, registered):
    spec = registered[family]
    with pytest.raises(ValueError, match="has no classes"):
        spec.load_factory()(variant="tiny", num_classes=10, input_shape=(32, 32, 3))
    with pytest.raises(ConfigError, match="has no classes"):
        recipes.build_model(spec, "tiny", 10, (32, 32, 3), resolved(family))
    # Control: None builds.
    assert recipes.build_model(spec, "tiny", None, (32, 32, 3), resolved(family)) is not None


def test_the_adapter_refuses_a_refused_key_as_a_value_error_not_a_type_error(registered):
    # `bfconvunext` pins use_bias in its wrapper: handed the key, the library raises TypeError, which no layer converts.
    factory = registered["bfconvunext"].load_factory()
    with pytest.raises(ValueError, match="use_bias"):
        factory(variant="tiny", num_classes=None, input_shape=(32, 32, 3), use_bias=False)
    with pytest.raises(ValueError, match="Unknown key 'model_args.no_such_key'"):
        factory(variant="tiny", num_classes=None, input_shape=(32, 32, 3), no_such_key=1)


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 3, 1), (0, 32, 3), (32, 32, 0), (32.0, 32, 3), (32, 32, None)])
def test_a_malformed_input_shape_is_refused(shape, registered):
    factory = registered["bfcnn"].load_factory()
    with pytest.raises(ValueError, match="input_shape"):
        factory(variant="tiny", num_classes=None, input_shape=shape)


def test_an_unknown_scunet_variant_is_refused(registered):
    with pytest.raises(ValueError, match="Unknown variant 'huge'"):
        registered["scunet"].load_factory()(variant="huge", num_classes=None, input_shape=(32, 32, 3))


def test_an_sgd_weight_decay_is_refused_because_no_regularizer_is_built(registered):
    ctx = context("bfunet")
    optim = OptimArgs().resolve_auto(ctx)
    resolver = registered["bfunet"].load_resolver()
    sgd = dataclasses.replace(optim, optimizer="sgd", weight_decay=1e-4)
    with pytest.raises(ConfigError, match="image_denoising task builds none for 'bfunet'"):
        resolver(ctx, sgd, {})
    # Controls: adamw with its decay, and sgd with none, resolve.
    assert resolver(ctx, optim, {})
    assert resolver(ctx, dataclasses.replace(optim, optimizer="sgd", weight_decay=None), {})


def test_the_resolver_names_every_refusal_at_once(registered):
    ctx = context("bfconvunext")
    resolver = registered["bfconvunext"].load_resolver()
    with pytest.raises(ConfigValidationError) as raised:
        resolver(ctx, OptimArgs().resolve_auto(ctx), {"block_activation": "gelu", "use_bias": False, "typo_key": 1})
    text = str(raised.value)
    assert "model_args.block_activation" in text and "model_args.use_bias" in text and "model_args.typo_key" in text


def test_a_key_the_library_supplies_is_refused_before_the_tables_are_read(registered):
    ctx = context("bfunet")
    with pytest.raises(ConfigError, match="is set by the library"):
        registered["bfunet"].load_resolver()(ctx, OptimArgs().resolve_auto(ctx), {"input_shape": [8, 8, 3]})


def test_an_adamw_run_refuses_an_explicit_kernel_regularizer_like_the_other_tasks(registered):
    ctx = context("bfunet")
    with pytest.raises(ConfigError, match="kernel_regularizer"):
        registered["bfunet"].load_resolver()(ctx, OptimArgs().resolve_auto(ctx), {"kernel_regularizer": "l2"})


# ---------------------------------------------------------------------
# The stride rule
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family, variant", [(f, v) for f in FAMILIES for v in VARIANTS[f]])
def test_the_stride_is_two_to_the_depth_for_the_unets_and_none_for_the_others(family, variant, registered):
    assert registered[family].stride_of(variant, {}) == STRIDES[family][variant]


@pytest.mark.parametrize("family", ("bfunet", "bfconvunext", "convunext_biased"))
def test_a_depth_override_changes_the_rule_and_a_bad_patch_is_refused_naming_patch_size(family, registered):
    spec = registered[family]
    assert check_image_size(spec, "tiny", {}, 64, key="transform_args.patch_size") == (64, 64)
    with pytest.raises(ConfigError, match=r"'transform_args.patch_size': the height 60 is not a multiple of 8.*56 or 64"):
        check_image_size(spec, "tiny", {}, 60, key="transform_args.patch_size")
    # depth=4 makes 16 the multiple: 56 passed at depth 3 and is refused at depth 4.
    assert check_image_size(spec, "tiny", {}, 56, key="transform_args.patch_size") == (56, 56)
    with pytest.raises(ConfigError, match="not a multiple of 16"):
        check_image_size(spec, "tiny", {"depth": 4}, 56, key="transform_args.patch_size")
    assert spec.stride_of("base", {"depth": 3}) == 8


def test_a_bad_depth_is_named(registered):
    for depth in (True, 0, "3"):
        with pytest.raises(ConfigError, match="model_args.depth"):
            registered["bfunet"].stride_of("tiny", {"depth": depth})
    with pytest.raises(ConfigError, match="model_args.depth"):
        registered["bfunet"].stride_of("tiny", {"depth": 1})


@pytest.mark.parametrize("family", ("bfcnn", "scunet"))
def test_the_families_without_a_stride_take_any_size(family, models):
    assert registry.MODELS[family].stride is None
    for height, width in ((33, 47), (17, 17)):
        assert tuple(models[family](batch(height, width)).shape) == (1, height, width, 3)


@pytest.mark.parametrize("family", ("bfunet", "bfconvunext", "convunext_biased"))
def test_the_unets_take_their_stride_and_the_rule_matches_the_model(family, models):
    # The rule is not a guess: a side that is a multiple of the stride runs, one that is not fails inside the model.
    assert tuple(models[family](batch(24, 40)).shape) == (1, 24, 40, 3)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        models[family](batch(35, 35))


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_every_recipe_holds_the_legacy_optimizer(family, registered):
    ctx = context(family)
    optim = OptimArgs().resolve_auto(ctx)
    for name, value in LEGACY_OPTIM.items():
        assert getattr(optim, name) == value, (family, name)
    assert ctx.defaults["epochs"] == 100
    assert ctx.defaults["checkpoint_args"] == {"early_stopping_patience": 100}


@pytest.mark.parametrize("family, variant", [(f, v) for f in FAMILIES for v in VARIANTS[f]])
def test_every_variants_recipe_passes_the_refusals_it_makes(family, variant, registered):
    assert set(registered[family].recipe("folder", variant)) <= set(recipes.RECIPE_KEYS)
    assert dm.problems_of(family, resolved(family, variant)) == []


def test_the_base_u_nets_take_batch_four_and_the_rest_sixteen(registered):
    for family in FAMILIES:
        for variant in VARIANTS[family]:
            batch_size = registered[family].recipe("folder", variant)["batch_size"]
            expected = 8 if family == "scunet" else 4 if family in ("bfunet", "bfconvunext", "convunext_biased") and variant == "base" else 16
            assert batch_size == expected, (family, variant)


def test_the_bias_free_recipes_are_the_legacy_builds(registered):
    assert resolved("bfcnn") == {"activation": "leaky_relu", "normalization_type": "batchnorm"}
    assert resolved("bfunet") == {
        "activation": "leaky_relu", "block_normalization": "batchnorm", "use_gabor_stem": True,
        "final_projection_groups": 1,
    }
    assert resolved("bfconvunext") == {
        "convnext_version": "v1", "block_normalization": "batchnorm", "block_activation": "leaky_relu",
        "stem_activation": "leaky_relu", "supervision_activation": "leaky_relu", "use_gabor_stem": True,
        "final_projection_groups": 1,
    }
    assert resolved("convunext_biased") == {} and resolved("scunet") == {}


def test_the_serialised_model_arguments_are_plain_json(registered):
    for family in FAMILIES:
        json.dumps(recipes.serialize_model_args(resolved(family)))
