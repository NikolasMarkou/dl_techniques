"""The census of ``init``: every family row's ``TransferSpec`` held against its real model.

plan-2026-11-18T072904-0fd54b91 Step 3 (D-007, D-030, D-031, D-033). Every registered row of a task
that takes ``init`` declares a :class:`~train.api.registry.TransferSpec`; a pretrain row declares
none and gives a reason. Each row's variant below is built through the real resolver path (the
recipe, ``OptimArgs.resolve_auto``, the family resolver, ``recipes.build_model``) at its task's
source geometry, then at each change the census names, one key at a time: the head (classes; a
forecasting horizon or channel count), the image size, the channels, the lookback, a deeper variant
where the spec declares ``grow``, and each declared cross-task pair. The engine's verdict
(``plan_transfer`` then ``check``) must be the one the spec predicts: a change of a ``fixed`` key is
refused at the weight level, every other change passes. The target carries the spec's ``pinned``
arguments from the source's resolved ones, as a run under ``init`` does.

Every build follows ``keras.backend.clear_session()``, so each model gets the name a first build in
a fresh process gets. xlstm is also checked in the production order (target built, then the
source's saved file loaded), where its auto-named inner layers drift. The counts are derived here
and logged as ``CENSUS`` lines.
"""

import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Tuple

import keras
import pytest

from dl_techniques.utils.logger import logger
from train.api import recipes, registry
from train.api.args import AutoContext, OptimArgs
from train.api.errors import ConfigError
from train.api.forecast_models import FAMILIES as FORECAST_FAMILIES
from train.api.init_from import check, plan_transfer

recipes.register_builtin_models()

CLASSIFICATION, SEGMENTATION, DENOISING = "image_classification", "image_segmentation", "image_denoising"
FORECASTING, DEPTH, PRETRAIN = "time_series_forecasting", "depth_estimation", "pretrain"
#: The tasks that take ``init``.
INIT_TASKS = (CLASSIFICATION, SEGMENTATION, DENOISING, FORECASTING, DEPTH)

#: The variant the census builds of each row: its smallest (convnext_v1 and convnext_seg: the 32 px recipe's).
VARIANT = {
    "vit": "vit_pico", "resnet": "resnet18", "convnext_v1": "cifar10", "convnext_v2": "atto", "cliffordnet": "nano",
    "cliffordnet_hier": "hier_345", "convunext": "tiny", "accunet": "base", "resnet_seg": "resnet18",
    "convnext_seg": "cifar10", "bfcnn": "tiny", "bfunet": "tiny", "bfconvunext": "tiny", "convunext_biased": "tiny",
    "scunet": "tiny", "nbeats": "tiny", "prism": "tiny", "tirex": "tiny", "xlstm": "tiny", "ets": "ANN",
    "timestral": "5M", "mdn": "tiny", "depth_anything": "vit_s",
}
#: The geometry a source of each task is built at.
SOURCE = {
    CLASSIFICATION: {"image_size": 32, "channels": 3, "classes": 10},
    SEGMENTATION: {"image_size": 64, "channels": 3, "classes": 3},
    DENOISING: {"channels": 3},
    FORECASTING: {"lookback": 168, "horizon": 24, "channels": 1},
    DEPTH: {"image_size": 112, "channels": 3},
}
#: The changes the census makes, one key at a time: the key and its new value.
CHANGES = {
    CLASSIFICATION: {"classes": 100, "image_size": 64, "channels": 1},
    SEGMENTATION: {"classes": 5, "image_size": 128, "channels": 1},
    DENOISING: {"channels": 1},
    FORECASTING: {"lookback": 336, "horizon": 12, "channels": 2},
    DEPTH: {"image_size": 168},
}
#: The keys of the head's identity (D-008): a change of one resets the target head and drops the source's.
HEAD_KEYS = {CLASSIFICATION: ("classes",), SEGMENTATION: ("classes",), DENOISING: ("channels",),
             FORECASTING: ("horizon", "channels"), DEPTH: ()}
#: The deeper variant each ``grow`` row is grown to.
GROWN = {"vit": "vit_tiny", "resnet": "resnet34", "resnet_seg": "resnet34"}
#: The dataset name a task's builds resolve their recipe for.
DATASET = {CLASSIFICATION: "cifar10", SEGMENTATION: "folder", DENOISING: "folder", FORECASTING: "builtin",
           DEPTH: "megadepth"}
#: ``model_args`` a row is built with: depth from scratch (no official weights read); a resnet classifier at
#: 32 px resolves the cifar stem, which resnet_seg never builds, so its cross-task source asks for the other.
MODEL_ARGS = {"depth_anything": {"pretrained": False}}
CROSS_SOURCE_ARGS = {"resnet": {"stem_type": "imagenet"}}

_BUILT: Dict[Tuple[Any, ...], Tuple[keras.Model, Dict[str, Any]]] = {}


def task_of(family: str) -> str:
    return registry.MODELS[family].tasks[0]


def init_families() -> List[str]:
    return sorted(family for family, spec in registry.MODELS.items() if spec.tasks[0] in INIT_TASKS)


def build(family: str, variant: str, geometry: Mapping[str, Any],
          model_args: Optional[Mapping[str, Any]] = None) -> Tuple[keras.Model, Dict[str, Any]]:
    """Build through the real resolver path after ``clear_session``; return the model and its resolved kwargs."""
    spec = registry.MODELS[family]
    task = spec.tasks[0]
    dataset, size, channels = DATASET[task], geometry.get("image_size"), geometry["channels"]
    context = AutoContext(family=family, variant=variant, dataset=dataset, num_classes=geometry.get("classes"),
                          image_size=size if task == CLASSIFICATION else None, defaults=spec.recipe(dataset, variant))
    args = {**MODEL_ARGS.get(family, {}), **(model_args or {})}
    kwargs = spec.load_resolver()(context, OptimArgs().resolve_auto(context), args)
    if task == FORECASTING:
        kwargs = {**kwargs, "horizon": geometry["horizon"]}
        if FORECAST_FAMILIES[family].static_batch:
            kwargs["batch_size"] = 4
        shape: Tuple[Optional[int], ...] = (geometry["lookback"], channels)
    elif task == DENOISING:
        shape = (None, None, channels)
    else:
        shape = (size, size, channels)
    keras.backend.clear_session()
    model = recipes.build_model(spec, variant, geometry.get("classes"), shape, kwargs)
    if not model.built:
        model.build((None, *shape))
    return model, kwargs


def source_of(family: str, geometry: Mapping[str, Any], model_args: Optional[Mapping[str, Any]] = None):
    """A source build, kept for the module: one source serves every change of its row."""
    key = (family, tuple(sorted(geometry.items())), tuple(sorted((model_args or {}).items())))
    if key not in _BUILT:
        _BUILT[key] = build(family, VARIANT[family], geometry, model_args)
    return _BUILT[key]


def verdict(row: str, change: str, source: keras.Model, target: keras.Model, spec: registry.TransferSpec,
            relation: Optional[registry.CrossTask] = None, head_reset: bool = False):
    """``("PASS" | "REFUSE", plan or None, message)``; the counts are logged as one ``CENSUS`` line."""
    plan, message = None, ""
    try:
        plan = plan_transfer(source, target, spec, relation=relation, head_reset=head_reset, name=row)
        check(plan)
        got = "PASS"
    except ConfigError as error:
        got, message = "REFUSE", str(error)
    counts = "refused while planning" if plan is None else (
        f"copied {len(plan.copied)} resampled {len(plan.resampled)} reset {len(plan.reset)} grown {len(plan.grown)} "
        f"dropped {len(plan.dropped)} missing {len(plan.missing)} unused {len(plan.unused)} "
        f"(source {plan.n_source}, target {plan.n_target})")
    logger.info(f"CENSUS {row} | {change} | {got} | {counts}")
    return got, plan, message


def changes() -> List[Tuple[str, str]]:
    """Every ``(family, key)`` the census builds; a forecasting family of one channel has no channel change."""
    cases = []
    for family in init_families():
        task = task_of(family)
        for key in CHANGES[task]:
            one_channel = task == FORECASTING and (FORECAST_FAMILIES[family].max_channels or 2) < 2
            if not (key == "channels" and one_channel):
                cases.append((family, key))
    return cases


def undeclared(specs: Mapping[str, registry.ModelSpec]) -> List[str]:
    """The rows that declare neither a spec nor a reason, or both."""
    return sorted(name for name, spec in specs.items() if (spec.transfer is None) == (not spec.transfer_reason))


# --- declarations --------------------------------------------------------------------------------


def test_every_row_declares_a_spec_or_a_reason():
    assert {task_of(family) for family in registry.MODELS} <= {*INIT_TASKS, PRETRAIN}
    assert all(len(spec.tasks) == 1 for spec in registry.MODELS.values())
    assert undeclared(registry.MODELS) == []
    for family, spec in registry.MODELS.items():
        assert (spec.transfer is None) == (task_of(family) == PRETRAIN), family
    pretrain = [spec for spec in registry.MODELS.values() if spec.tasks[0] == PRETRAIN]
    assert len(pretrain) == 2 and all("pretrain does not start from earlier weights" in spec.transfer_reason
                                      for spec in pretrain)


def test_a_row_that_declares_neither_or_both_is_found():
    plain = registry.ModelSpec(family="plain", factory="m:f", variants_attr="m:V", tasks=(CLASSIFICATION,))
    both = dataclasses.replace(plain, family="both", transfer=registry.TransferSpec(), transfer_reason="why")
    assert undeclared({"plain": plain, "both": both, "vit": registry.MODELS["vit"]}) == ["both", "plain"]


def test_the_census_covers_every_row_and_measures_every_declaration():
    families = init_families()
    assert sorted(VARIANT) == families and len(families) == 23
    assert sorted(GROWN) == sorted(f for f in families if registry.MODELS[f].transfer.grow)
    assert sorted(f for f in families if registry.MODELS[f].transfer.positional) == ["xlstm"]
    built = set(changes())
    for family in families:
        spec = registry.MODELS[family].transfer
        # A fixed key the census never changes would be a declaration nothing measures.
        assert all((family, key) in built for key in spec.fixed), (family, spec.fixed)
        for task, source in spec.sources:
            assert task != task_of(family) and task_of(source) == task, (family, task, source)


def test_a_pinned_argument_is_a_resolved_factory_argument():
    pinned = {f: registry.MODELS[f].transfer.pinned for f in init_families() if registry.MODELS[f].transfer.pinned}
    assert pinned == {"vit": ("patch_size",), "resnet": ("stem_type",), "convnext_v1": ("strides",),
                      "convnext_v2": ("strides",)}
    for family, keys in pinned.items():
        _, kwargs = source_of(family, SOURCE[task_of(family)])
        assert all(key in kwargs for key in keys), (family, sorted(kwargs))


# --- verdicts on the real models -----------------------------------------------------------------


@pytest.mark.parametrize("family", init_families())
def test_the_same_geometry_copies_every_weight(family):
    task = task_of(family)
    source, kwargs = source_of(family, SOURCE[task])
    spec = registry.MODELS[family].transfer
    target, _ = build(family, VARIANT[family], SOURCE[task], {key: kwargs[key] for key in spec.pinned})
    got, plan, message = verdict(family, "same", source, target, spec)
    assert got == "PASS", message
    assert len(plan.copied) == plan.n_source == plan.n_target > 0


@pytest.mark.parametrize("family,key", changes(), ids=[f"{f}-{k}" for f, k in changes()])
def test_a_change_gives_the_verdict_the_spec_predicts(family, key):
    task = task_of(family)
    source, kwargs = source_of(family, SOURCE[task])
    spec = registry.MODELS[family].transfer
    geometry = {**SOURCE[task], key: CHANGES[task][key]}
    target, _ = build(family, VARIANT[family], geometry, {name: kwargs[name] for name in spec.pinned})
    head_reset = key in HEAD_KEYS[task]
    got, plan, message = verdict(family, f"{key} {SOURCE[task][key]} to {CHANGES[task][key]}", source, target, spec,
                                 head_reset=head_reset)
    want = "REFUSE" if key in spec.fixed else "PASS"
    assert got == want, f"{family} {key}: the spec predicts {want}, the engine says {got}: {message}"
    if want == "REFUSE":
        assert message.startswith("Invalid value for 'init': ") and "nothing was copied" in message
    else:
        assert len(plan.copied) + len(plan.resampled) + len(plan.reset) == plan.n_target
        assert len(plan.reset) == len(plan.dropped) and (len(plan.reset) > 0) == (head_reset and bool(spec.head))


@pytest.mark.parametrize("family", sorted(GROWN))
def test_a_deeper_variant_grows_the_new_blocks(family):
    task = task_of(family)
    source, kwargs = source_of(family, SOURCE[task])
    spec = registry.MODELS[family].transfer
    target, _ = build(family, GROWN[family], SOURCE[task], {key: kwargs[key] for key in spec.pinned})
    got, plan, message = verdict(family, f"grow {VARIANT[family]} to {GROWN[family]}", source, target, spec)
    assert got == "PASS", message
    assert len(plan.copied) == plan.n_source and len(plan.grown) == plan.n_target - plan.n_source > 0


def cross_pairs() -> List[Tuple[str, str, str]]:
    return [(family, task, source) for family in init_families()
            for task, source in registry.MODELS[family].transfer.sources]


@pytest.mark.parametrize("family,task,source_family", cross_pairs(), ids=[f"{s}-to-{f}" for f, _, s in cross_pairs()])
def test_a_declared_cross_task_pair_fills_the_target_and_resets_its_head(family, task, source_family):
    spec = registry.MODELS[family].transfer
    relation = spec.sources[(task, source_family)]
    source, _ = source_of(source_family, SOURCE[task], CROSS_SOURCE_ARGS.get(source_family))
    target, _ = build(family, VARIANT[family], SOURCE[task_of(family)])
    got, plan, message = verdict(family, f"from {source_family} ({task})", source, target, spec, relation=relation)
    assert got == "PASS", message
    assert len(plan.copied) == plan.n_source - len(plan.dropped) and len(plan.dropped) > 0
    assert len(plan.copied) + len(plan.reset) == plan.n_target and len(plan.reset) > 0


def test_xlstm_pairs_its_drifting_blocks_by_position_in_the_production_order(tmp_path):
    # D-031: the target is built first, then the source's file is loaded; its auto-named rnn and cell layers take the
    # next free names (rnn_2, m_lstm_cell_1), so the paths inside those blocks differ while every weight pairs by order.
    spec = registry.MODELS["xlstm"].transfer
    variants = sorted(registry.load_attr(registry.MODELS["xlstm"].variants_attr))
    drifting = set()
    for variant in variants:
        source, _ = build("xlstm", variant, SOURCE[FORECASTING])
        path = tmp_path / f"xlstm_{variant}.keras"
        source.save(path)
        target, _ = build("xlstm", variant, SOURCE[FORECASTING])
        loaded = keras.saving.load_model(path, compile=False)
        root = f"{target.name}/"
        pairs = list(zip(target.weights, loaded.weights))
        found = {t.path[len(root):].split("/")[0] for t, s in pairs if t.path != s.path}
        assert found and found <= set(spec.positional), (variant, sorted(found))
        drifting |= found
        got, plan, message = verdict("xlstm", f"{variant} production order", loaded, target, spec)
        assert got == "PASS" and len(plan.copied) == plan.n_target, message
        plain = dataclasses.replace(spec, positional=())
        assert verdict("xlstm", f"{variant} production order, no positional", loaded, target, plain)[0] == "REFUSE"
    assert drifting == set(spec.positional)
