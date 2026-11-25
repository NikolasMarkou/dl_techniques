"""The census of ``init``: every family row's ``TransferSpec`` held against its real model.

plan-2026-11-18T072904-0fd54b91 Step 3 (D-007, D-030, D-031, D-033). Every registered row declares a
:class:`~train.api.registry.TransferSpec` (the two pretrain rows since plan 5c6cc105 Step 6, at the
end of this file). The variant of each row of the other tasks is built through the real resolver path (the
recipe, ``OptimArgs.resolve_auto``, the family resolver, ``recipes.build_model``) at its task's
source geometry, then at each change the census names, one key at a time: the head (classes; a
forecasting horizon or channel count), the image size, the channels, the lookback, a deeper variant
where the spec declares ``grow``, and each declared cross-task pair. The engine's verdict
(``plan_transfer`` then ``check``) must be the one the spec predicts: a change of a ``fixed`` key is
refused at the weight level, every other change passes. The target carries the spec's ``pinned``
arguments from the source's resolved ones, as a run under ``init`` does.

Every build follows ``keras.backend.clear_session()``, so each model gets the name a first build in
a fresh process gets. Every row is also checked in the production order (the target built first, then
the source's saved archive loaded beside it, as a run under ``init`` does) at the same geometry, with
one changed-geometry case per kind of change, every cross-task pair and every xlstm variant: a loaded
model's paths can lose the root a fresh build gives them (accunet's ``acc_u_net/``, the depth
backbone's ``depth_anything/``), and xlstm's auto-named inner layers drift. The counts are derived
here and logged as ``CENSUS`` lines.

plan-2026-11-19T081214-5c6cc105 Step 5 adds the ``beit`` and ``energy_transformer`` classifiers and their
pretrain sources (``beit_mim``, ``et_mim``, built at the pretrain geometry of CIFAR-10): a larger grid
resamples every BEiT relative-bias table (and the absolute table where the option is on) and is refused
by the Energy Transformer position rule, a class-token head resets on a change of classes, a class-token pretrain
source seeds a class-token classifier with the README's counts, and a pooling mismatch is refused in both directions
(D-049). A grid change must resample exactly the weights whose shape changed. Step 6 holds
the two pretrain rows (D-008): their own family only, the decoder their head, the size and the patch
fixed; in the production order every weight is copied, another tokenizer resets exactly the decoder, and
another image size is refused at the weight level, behind the refusal before the claim.

plan-2026-11-24T092348-7806a290 Step 6 (D-005, D-040) holds the language-model row (``gpt2_lm``) the same way, at the end of this
file: its own family only; in the production order every weight is copied, a window of another length changes no shape (so the
length is not a ``fixed`` key), and another vocabulary is refused at the weight level, behind the refusal before the claim.

plan-2026-11-25T124419-fa94f59c Step 8 (D-012, D-050) holds the tokenizer row (``vq_vae_rotation``) the same way, at the end of this
file: its own family only and no head; in the production order every weight is copied (35 of a tiny model, 47 of the gate's), another
channel count, patch or codebook size is refused at the weight level behind the refusal before the claim, and so is another width, depth
or embedding size, which the record does not carry (the engine refuses those after the claim). The image size is the one ``fixed`` key
the engine would take: a convolutional tokenizer has no weight that depends on it, so the row refuses it before the claim by POLICY
(the plan's contract is the same size), and a test says so and would fail the day a size-dependent weight made it a fact.
"""

import dataclasses
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import keras
import pytest

from dl_techniques.utils.logger import logger
from train.api import recipes, registry, tokenizer_models
from train.api.args import AutoContext, OptimArgs
from train.api.errors import ConfigError
from train.api.forecast_models import FAMILIES as FORECAST_FAMILIES
from train.api.init_from import GEOMETRY_NAMES, _by_path, check, plan_transfer
from train.api.pretrain_args import BEIT as BEIT_MIM, ET as ET_MIM, GEOMETRY

from ._text_world import ASCII_VOCAB, SEQ_LEN, TINY_MODEL_ARGS

recipes.register_builtin_models()

CLASSIFICATION, SEGMENTATION, DENOISING = "image_classification", "image_segmentation", "image_denoising"
FORECASTING, DEPTH, PRETRAIN = "time_series_forecasting", "depth_estimation", "pretrain"
#: The language-model task has one family, ``gpt2_lm``: censused at the end of the file, as the pretrain rows are.
LANGUAGE_MODEL, GPT2_LM = "language_model_pretraining", "gpt2_lm"
#: The text-classification task has two families, ``bert`` and ``gpt2``: their pair with the language model is censused in
#: ``test_text_init_pairs.py`` (plan-2026-11-24T092348-7806a290 steps 9 and 10), not here.
TEXT_CLASSIFICATION = "text_classification"
#: The tokenizer task has one family, ``vq_vae_rotation`` (plan-2026-11-25T124419-fa94f59c): censused at the end of the file, as the pretrain
#: and language rows are.
TOKENIZER, TOKENIZER_FAMILY = "beit_tokenizer", "vq_vae_rotation"
#: The tasks that take ``init``.
INIT_TASKS = (CLASSIFICATION, SEGMENTATION, DENOISING, FORECASTING, DEPTH)

#: The variant the census builds of each row: its smallest (convnext_v1 and convnext_seg: the 32 px recipe's).
VARIANT = {
    "vit": "vit_pico", "resnet": "resnet18", "convnext_v1": "cifar10", "convnext_v2": "atto", "cliffordnet": "nano",
    "cliffordnet_hier": "hier_345", "convunext": "tiny", "accunet": "base", "resnet_seg": "resnet18",
    "convnext_seg": "cifar10", "bfcnn": "tiny", "bfunet": "tiny", "bfconvunext": "tiny", "convunext_biased": "tiny",
    "scunet": "tiny", "nbeats": "tiny", "prism": "tiny", "tirex": "tiny", "xlstm": "tiny", "ets": "ANN",
    "timestral": "5M", "mdn": "tiny", "depth_anything": "vit_s", "beit": "beit_tiny", "energy_transformer": "et_tiny",
    BEIT_MIM: "beit_tiny", ET_MIM: "et_tiny", GPT2_LM: "tiny", TOKENIZER_FAMILY: "small",
}
#: The geometry a source of each task is built at.
SOURCE = {
    CLASSIFICATION: {"image_size": 32, "channels": 3, "classes": 10},
    SEGMENTATION: {"image_size": 64, "channels": 3, "classes": 3},
    DENOISING: {"channels": 3},
    FORECASTING: {"lookback": 168, "horizon": 24, "channels": 1},
    DEPTH: {"image_size": 112, "channels": 3},
    PRETRAIN: {"image_size": GEOMETRY["cifar10"][0], "channels": 3},
    LANGUAGE_MODEL: {"seq_len": SEQ_LEN, "vocab": ASCII_VOCAB},
    TOKENIZER: {"image_size": 32, "channels": 3, "patch": 4},
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
           DEPTH: "megadepth", PRETRAIN: "cifar10", LANGUAGE_MODEL: "synthetic", TOKENIZER: "cifar10"}
#: The tokenizer codebook a ``beit_mim`` source is built with; its decoder head is dropped by the classifier pair.
VOCAB = 64
#: A tokenizer the CPU builds in a third of a second: 64 codes of 8 numbers, 16 wide, one residual block (35 weights).
TOKENIZER_ARGS = {"num_embeddings": 64, "embedding_dim": 8, "hidden_channels": 16, "num_res_blocks": 1}
#: ``model_args`` a row is built with: depth from scratch (no official weights read); a resnet classifier at
#: 32 px resolves the cifar stem, which resnet_seg never builds, so its cross-task source asks for the other.
MODEL_ARGS = {"depth_anything": {"pretrained": False}, GPT2_LM: TINY_MODEL_ARGS, TOKENIZER_FAMILY: TOKENIZER_ARGS}
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
    dataset, size, channels = DATASET[task], geometry.get("image_size"), geometry.get("channels")
    context = AutoContext(family=family, variant=variant, dataset=dataset, num_classes=geometry.get("classes"),
                          image_size=size if task == CLASSIFICATION else None, defaults=spec.recipe(dataset, variant))
    args = {**MODEL_ARGS.get(family, {}), **(model_args or {})}
    kwargs = spec.load_resolver()(context, OptimArgs().resolve_auto(context), args)
    if task == PRETRAIN:  # what the pretrain task adds (tasks/pretrain.py build_model): the data's patch, the codebook
        kwargs = {**kwargs, "patch_size": GEOMETRY[dataset][1], **({"vocab_size": VOCAB} if family == BEIT_MIM else {})}
    if task == LANGUAGE_MODEL:  # what the language task adds (tasks/language_model_pretraining.py build_model): the tokenizer's vocabulary
        kwargs = {**kwargs, "vocab_size": geometry["vocab"]}
    if task == TOKENIZER:  # what the tokenizer task adds (tasks/beit_tokenizer.py build_model): the patch the encoder's stride is
        kwargs = {**kwargs, "patch_size": geometry["patch"]}
    if task == FORECASTING:
        kwargs = {**kwargs, "horizon": geometry["horizon"]}
        if FORECAST_FAMILIES[family].static_batch:
            kwargs["batch_size"] = 4
        shape: Tuple[Optional[int], ...] = (geometry["lookback"], channels)
    elif task == DENOISING:
        shape = (None, None, channels)
    elif task == LANGUAGE_MODEL:
        shape = (geometry["seq_len"],)
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
    """The rows that declare no spec (``init`` refuses such a row with one generic sentence, D-011)."""
    return sorted(name for name, spec in specs.items() if spec.transfer is None)


# --- declarations --------------------------------------------------------------------------------


def test_every_row_declares_a_spec():
    assert {task_of(family) for family in registry.MODELS} <= {*INIT_TASKS, PRETRAIN, LANGUAGE_MODEL, TEXT_CLASSIFICATION, TOKENIZER}
    assert all(len(spec.tasks) == 1 for spec in registry.MODELS.values())
    assert undeclared(registry.MODELS) == []
    assert sum(task_of(family) == PRETRAIN for family in registry.MODELS) == 2
    assert sum(task_of(family) == LANGUAGE_MODEL for family in registry.MODELS) == 1
    assert sum(task_of(family) == TEXT_CLASSIFICATION for family in registry.MODELS) == 2
    assert sum(task_of(family) == TOKENIZER for family in registry.MODELS) == 1


def test_a_row_without_a_spec_is_found_and_no_row_carries_a_reason_instead():
    plain = registry.ModelSpec(family="plain", factory="m:f", variants_attr="m:V", tasks=(CLASSIFICATION,))
    assert undeclared({"plain": plain, "vit": registry.MODELS["vit"]}) == ["plain"]
    assert "transfer_reason" not in {field.name for field in dataclasses.fields(registry.ModelSpec)}  # D-011


def test_the_census_covers_every_row_and_measures_every_declaration():
    families = init_families()
    assert sorted(set(VARIANT) - {BEIT_MIM, ET_MIM, GPT2_LM, TOKENIZER_FAMILY}) == families
    # 23 rows before plan 5c6cc105 Step 5, which adds the beit and energy_transformer classifiers.
    assert len(families) == 25
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
                      "convnext_v2": ("strides",), "beit": ("patch_size",), "energy_transformer": ("patch_size",)}
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
    judge_change(family, key, source, target)


#: The rules that refuse instead of resampling: a change of the grid they key is refused while planning, by name.
REFUSING_RULES = ("et_pos_refused",)


def reshaped(source: Any, target: keras.Model) -> List[str]:
    """The target paths whose source weight has another shape: what a grid change must resample (derived, not declared)."""
    theirs, ours = _by_path(source, "source"), _by_path(target, "target")
    return sorted(path for path, var in ours.items() if path in theirs and tuple(var.shape) != tuple(theirs[path].shape))


def judge_change(family: str, key: str, source: Any, target: keras.Model) -> None:
    """The verdict of one census change is the one the spec predicts, with its counts."""
    task, spec = task_of(family), registry.MODELS[family].transfer
    head_reset = key in HEAD_KEYS[task]
    got, plan, message = verdict(family, f"{key} {SOURCE[task][key]} to {CHANGES[task][key]}", source, target, spec,
                                 head_reset=head_reset)
    refused_by_rule = key == "image_size" and any(rule in REFUSING_RULES for rule in spec.resample.values())
    want = "REFUSE" if key in spec.fixed or refused_by_rule else "PASS"
    assert got == want, f"{family} {key}: the spec predicts {want}, the engine says {got}: {message}"
    if want == "REFUSE" and key in spec.fixed:
        assert message.startswith("Invalid value for 'init': ") and "nothing was copied" in message
    elif want == "REFUSE":  # refused while planning, so nothing was assigned; the sentence names both grids
        assert plan is None and message.startswith("Invalid value for 'init': the Energy Transformer position table")
        assert "(8x8 patches)" in message and "(16x16 patches)" in message and "train this run at the source's grid" in message
    else:
        assert len(plan.copied) + len(plan.resampled) + len(plan.reset) == plan.n_target
        assert len(plan.reset) == len(plan.dropped) and (len(plan.reset) > 0) == (head_reset and bool(spec.head))
        # Every weight whose shape the change altered (outside a reset head) is resampled, by the rule its key names.
        heads = spec.head if head_reset else ()
        altered = [path for path in reshaped(source, target) if not any(path.startswith(h + "/") for h in heads)]
        assert sorted(path for path, *_ in plan.resampled) == altered, (family, key)
        logger.info(f"CENSUS {family} | {key} | reshaped {len(altered)} | resampled {len(plan.resampled)} by "
                    f"{sorted({rule for _, rule, *_ in plan.resampled})}")


@pytest.mark.parametrize("family", sorted(GROWN))
def test_a_deeper_variant_grows_the_new_blocks(family):
    task = task_of(family)
    source, kwargs = source_of(family, SOURCE[task])
    spec = registry.MODELS[family].transfer
    target, _ = build(family, GROWN[family], SOURCE[task], {key: kwargs[key] for key in spec.pinned})
    judge_grow(family, source, target)


def judge_grow(family: str, source: Any, target: keras.Model) -> None:
    spec = registry.MODELS[family].transfer
    got, plan, message = verdict(family, f"grow {VARIANT[family]} to {GROWN[family]}", source, target, spec)
    assert got == "PASS", message
    assert len(plan.copied) == plan.n_source and len(plan.grown) == plan.n_target - plan.n_source > 0


def cross_pairs() -> List[Tuple[str, str, str]]:
    return [(family, task, source) for family in init_families()
            for task, source in registry.MODELS[family].transfer.sources]


@pytest.mark.parametrize("family,task,source_family", cross_pairs(), ids=[f"{s}-to-{f}" for f, _, s in cross_pairs()])
def test_a_declared_cross_task_pair_fills_the_target_and_resets_its_head(family, task, source_family):
    source, _ = source_of(source_family, SOURCE[task], CROSS_SOURCE_ARGS.get(source_family))
    target, _ = build(family, VARIANT[family], SOURCE[task_of(family)])
    judge_cross(family, task, source_family, source, target)


def judge_cross(family: str, task: str, source_family: str, source: Any, target: keras.Model) -> None:
    spec = registry.MODELS[family].transfer
    relation = spec.sources[(task, source_family)]
    got, plan, message = verdict(family, f"from {source_family} ({task})", source, target, spec, relation=relation)
    assert got == "PASS", message
    assert len(plan.copied) == plan.n_source - len(plan.dropped) and len(plan.dropped) > 0
    assert len(plan.copied) + len(plan.reset) == plan.n_target and len(plan.reset) > 0


#: xlstm in the production order: each variant at its defaults, and the offered ``model_args`` that change the blocks
#: (``num_layers`` 6 adds mlstm_block_2 to slstm_block_5; ``mlstm_ratio`` 0 builds sLSTM blocks only, from
#: slstm_block_0).
XLSTM_CASES = [("tiny", {}), ("small", {}), ("tiny", {"num_layers": 6}), ("tiny", {"mlstm_ratio": 0.0})]


@pytest.mark.parametrize("variant,model_args", XLSTM_CASES, ids=[f"{v}-{sorted(a.items())}" for v, a in XLSTM_CASES])
def test_xlstm_pairs_its_drifting_blocks_by_position_in_the_production_order(variant, model_args, tmp_path):
    # D-031: the target is built first, then the source's file is loaded; its auto-named rnn and cell layers take the
    # next free names (rnn_2, m_lstm_cell_1), so the paths inside those blocks differ while every weight pairs by order.
    spec = registry.MODELS["xlstm"].transfer
    source, _ = build("xlstm", variant, SOURCE[FORECASTING], model_args)
    source.save(tmp_path / "xlstm.keras")
    target, _ = build("xlstm", variant, SOURCE[FORECASTING], model_args)
    loaded = keras.saving.load_model(tmp_path / "xlstm.keras", compile=False)
    root = f"{target.name}/"
    found = {t.path[len(root):].split("/")[0] for t, s in zip(target.weights, loaded.weights) if t.path != s.path}
    # Every drifting block matches a declared pattern, and no pattern is dead.
    assert found and all(any(re.fullmatch(p, c) for p in spec.positional) for c in found), sorted(found)
    assert all(any(re.fullmatch(p, c) for c in found) for p in spec.positional), spec.positional
    got, plan, message = verdict("xlstm", f"{variant} {model_args} production order", loaded, target, spec)
    assert got == "PASS" and len(plan.copied) == plan.n_target, message
    plain = dataclasses.replace(spec, positional=())
    assert verdict("xlstm", f"{variant} {model_args} no positional", loaded, target, plain)[0] == "REFUSE"


# --- the production order: the target built first, then the source's archive loaded beside it -----------------------


@pytest.fixture(scope="module")
def archives(tmp_path_factory) -> Dict[Any, Any]:
    """Each source build saved once for the module: ``{(family, geometry, model_args): path}``."""
    return {"dir": tmp_path_factory.mktemp("archives")}


def loaded_beside(family: str, variant: str, geometry: Mapping[str, Any], model_args: Mapping[str, Any],
                  archives: Dict[Any, Any], source_family: Optional[str] = None,
                  source_geometry: Optional[Mapping[str, Any]] = None,
                  source_args: Optional[Mapping[str, Any]] = None) -> Tuple[keras.Model, keras.Model]:
    """``(loaded source, target)``: the target is built after ``clear_session``, as a run builds its model, then the
    source's saved archive is loaded beside it, as ``init_from.transfer`` loads it (no ``clear_session`` between).
    ``source_args`` are the source's ``model_args`` (default: none for its own family, ``CROSS_SOURCE_ARGS`` else)."""
    source_family = source_family or family
    source_geometry = source_geometry or SOURCE[task_of(source_family)]
    if source_args is None:
        source_args = CROSS_SOURCE_ARGS.get(source_family) if source_family != family else None
    key = (source_family, tuple(sorted(source_geometry.items())), tuple(sorted((source_args or {}).items())))
    if key not in archives:
        archives[key] = archives["dir"] / f"{len(archives)}_{source_family}.keras"
        source_of(source_family, source_geometry, source_args)[0].save(archives[key])
    target, _ = build(family, variant, geometry, model_args)
    return keras.saving.load_model(archives[key], compile=False), target


def pins_of(family: str) -> Dict[str, Any]:
    _, kwargs = source_of(family, SOURCE[task_of(family)])
    return {key: kwargs[key] for key in registry.MODELS[family].transfer.pinned}


@pytest.mark.parametrize("family", init_families())
def test_production_order_the_same_geometry_copies_every_weight_and_the_head_names_a_weight(family, archives):
    spec, geometry = registry.MODELS[family].transfer, SOURCE[task_of(family)]
    loaded, target = loaded_beside(family, VARIANT[family], geometry, pins_of(family), archives)
    got, plan, message = verdict(family, "same, production order", loaded, target, spec)
    assert got == "PASS", message
    assert len(plan.copied) == plan.n_source == plan.n_target > 0
    # A declared head that names no weight would reset nothing when the head changes.
    assert not spec.head or any(path == p or path.startswith(p + "/") for p in spec.head for path in plan.copied)


#: One changed-geometry case per kind of change the census makes, in the production order: a head (accunet and mdn,
#: whose paths share a root other than the model's name), the image size, the channels and the lookback.
PRODUCTION_CHANGES = [("accunet", "classes"), ("mdn", "horizon"), ("vit", "image_size"), ("timestral", "channels"),
                      ("tirex", "lookback"), ("beit", "classes"), ("beit", "image_size"), ("energy_transformer", "classes"),
                      ("energy_transformer", "image_size")]


@pytest.mark.parametrize("family,key", PRODUCTION_CHANGES, ids=[f"{f}-{k}" for f, k in PRODUCTION_CHANGES])
def test_production_order_a_change_gives_the_verdict_the_spec_predicts(family, key, archives):
    task = task_of(family)
    geometry = {**SOURCE[task], key: CHANGES[task][key]}
    loaded, target = loaded_beside(family, VARIANT[family], geometry, pins_of(family), archives)
    judge_change(family, key, loaded, target)


def test_production_order_a_deeper_vit_grows_its_new_blocks(archives):
    loaded, target = loaded_beside("vit", GROWN["vit"], SOURCE[CLASSIFICATION], pins_of("vit"), archives)
    judge_grow("vit", loaded, target)


@pytest.mark.parametrize("family,task,source_family", cross_pairs(), ids=[f"{s}-to-{f}" for f, _, s in cross_pairs()])
def test_production_order_a_declared_cross_task_pair_fills_the_target(family, task, source_family, archives):
    loaded, target = loaded_beside(family, VARIANT[family], SOURCE[task_of(family)], {}, archives, source_family,
                                   SOURCE[task])
    judge_cross(family, task, source_family, loaded, target)


# --- the BEiT and Energy Transformer classifiers and their pretrain sources (plan 5c6cc105 Step 5) -------------------

#: Each classifier row and the pretrain row that trains its trunk.
MIM_SOURCE = {"beit": BEIT_MIM, "energy_transformer": ET_MIM}
LARGER = {**SOURCE[CLASSIFICATION], "image_size": CHANGES[CLASSIFICATION]["image_size"]}


def relative_tables(model: keras.Model) -> List[str]:
    return [path for path in _by_path(model, "model") if path.endswith("/relative_position_bias_table")]


def test_the_two_classifier_rows_take_exactly_their_pretrain_rows_trunk():
    for family, source in MIM_SOURCE.items():
        spec = registry.MODELS[family].transfer
        assert sorted(spec.sources) == [(PRETRAIN, source)], family
        assert spec.pinned == ("patch_size",) and spec.fixed == ("channels",), family


def test_the_pretrain_geometry_is_the_classifier_patch_rule_on_both_datasets():
    """A pretrain source and its classifier agree on the patch at the same dataset (the pair needs no pin)."""
    for family in MIM_SOURCE:
        spec = registry.MODELS[family]
        for dataset, (side, patch) in GEOMETRY.items():
            context = AutoContext(family=family, variant=VARIANT[family], dataset=dataset, num_classes=10,
                                  image_size=side, defaults=spec.recipe(dataset, VARIANT[family]))
            assert spec.load_resolver()(context, OptimArgs().resolve_auto(context), {})["patch_size"] == patch


@pytest.mark.parametrize("family", sorted(MIM_SOURCE))
def test_production_order_a_pretrain_source_at_a_larger_grid(family, archives):
    """beit_mim 32/4 into beit 64/4 resamples every relative-bias table; et_mim into energy_transformer is refused."""
    spec = registry.MODELS[family].transfer
    # A run takes the source record's patch as a pin (D-035); a build here passes it itself. Without it the patch rule
    # gives 16 at 64 px (a 4x4 grid, a SMALLER one, which the BEiT rule refuses while planning).
    patch = {"patch_size": GEOMETRY["cifar10"][1]}
    loaded, target = loaded_beside(family, VARIANT[family], LARGER, patch, archives, MIM_SOURCE[family], SOURCE[PRETRAIN])
    relation = spec.sources[(PRETRAIN, MIM_SOURCE[family])]
    got, plan, message = verdict(family, f"from {MIM_SOURCE[family]} at 64/4, production order", loaded, target, spec,
                                 relation=relation)
    if family == "energy_transformer":
        assert got == "REFUSE" and plan is None and "(8x8 patches) cannot become (1, 256, 192) (16x16 patches)" in message
        return
    assert got == "PASS", message
    tables = relative_tables(target)
    assert len(tables) == target.backbone.num_layers and sorted(p for p, *_ in plan.resampled) == sorted(tables)
    assert {rule for _, rule, *_ in plan.resampled} == {"beit_relpos_cubic"}
    assert len(plan.copied) == plan.n_source - len(plan.dropped) - len(tables) and len(plan.reset) == len(plan.dropped)
    logger.info(f"CENSUS {family} | from {MIM_SOURCE[family]} at 64/4 | tables {len(tables)} of "
                f"{target.backbone.num_layers} layers")


def test_production_order_a_class_token_beit_resets_its_classification_head(archives):
    """``use_mean_pooling=False``: the head is ``classification_head``, the second name of the spec's head."""
    pooling = {"use_mean_pooling": False}
    spec = registry.MODELS["beit"].transfer
    geometry = {**SOURCE[CLASSIFICATION], "classes": CHANGES[CLASSIFICATION]["classes"]}
    loaded, target = loaded_beside("beit", VARIANT["beit"], geometry, {**pins_of("beit"), **pooling}, archives,
                                   source_args=pooling)
    got, plan, message = verdict("beit", "classes 10 to 100, class token, production order", loaded, target, spec,
                                 head_reset=True)
    assert got == "PASS", message
    assert sorted(plan.reset) == ["classification_head/classifier/bias", "classification_head/classifier/kernel"]
    assert sorted(plan.dropped) == sorted(plan.reset) and len(plan.copied) == plan.n_target - len(plan.reset)
    assert "final_norm/gamma" in {path.rpartition("beit_backbone/")[2] for path in plan.copied}


@pytest.mark.parametrize("source_family", ["beit", BEIT_MIM])
def test_production_order_a_pooling_mismatch_is_refused(source_family, archives):
    """A mean-pooled source has no ``final_norm`` for a class-token target: nothing is copied."""
    spec = registry.MODELS["beit"].transfer
    relation = spec.sources.get((task_of(source_family), source_family))
    loaded, target = loaded_beside("beit", VARIANT["beit"], SOURCE[CLASSIFICATION], {"use_mean_pooling": False},
                                   archives, source_family, SOURCE[task_of(source_family)])
    got, _, message = verdict("beit", f"pooling mismatch from {source_family}", loaded, target, spec, relation=relation)
    assert got == "REFUSE" and "nothing was copied" in message and "beit_backbone/final_norm/gamma" in message
    if source_family == BEIT_MIM:  # the README's sentence for the pair
        assert "2 of 224 weights of beit have no source weight of their shape and no declared rule" in message, message


# plan 5c6cc105 D-049 (review R8): the cross-task class-token pair the README gives numbers for, and its other direction.
def test_production_order_a_class_token_beit_mim_seeds_a_class_token_beit_with_the_readmes_counts(archives):
    """``use_mean_pooling=false`` on both runs: the trunk and its ``final_norm`` copied, ``classification_head`` fresh,
    the decoder dropped. The README states these counts: 222 copied, 2 reset, 4 dropped."""
    pooling = {"use_mean_pooling": False}
    spec = registry.MODELS["beit"].transfer
    loaded, target = loaded_beside("beit", VARIANT["beit"], SOURCE[CLASSIFICATION], pooling, archives, BEIT_MIM,
                                   SOURCE[PRETRAIN], source_args=pooling)
    got, plan, message = verdict("beit", "from beit_mim, class token on both, production order", loaded, target, spec,
                                 relation=spec.sources[(PRETRAIN, BEIT_MIM)])
    assert got == "PASS", message
    assert (len(plan.copied), len(plan.reset), len(plan.dropped), len(plan.resampled)) == (222, 2, 4, 0)
    assert sorted(plan.reset) == ["classification_head/classifier/bias", "classification_head/classifier/kernel"]
    assert {path.split("/")[0] for path in plan.dropped} == set(DECODER[BEIT_MIM])
    assert "final_norm/gamma" in {path.rpartition("beit_backbone/")[2] for path in plan.copied}


def test_production_order_a_class_token_beit_mim_into_a_mean_pooled_beit_is_refused(archives):
    """The source's ``final_norm`` has no target weight: nothing is copied (D-031)."""
    spec = registry.MODELS["beit"].transfer
    loaded, target = loaded_beside("beit", VARIANT["beit"], SOURCE[CLASSIFICATION], {}, archives, BEIT_MIM,
                                   SOURCE[PRETRAIN], source_args={"use_mean_pooling": False})
    got, _, message = verdict("beit", "pooling mismatch, class-token beit_mim into mean pooling", loaded, target, spec,
                              relation=spec.sources[(PRETRAIN, BEIT_MIM)])
    assert got == "REFUSE" and "nothing was copied" in message and "beit_backbone/final_norm/gamma" in message, message
    assert "2 of 226 weights of the source run are neither used nor dropped by a declared rule" in message, message


def test_production_order_the_absolute_beit_table_is_resampled_by_the_vit_rule(archives):
    """D-007: ``use_absolute_position_embeddings`` reaches the factory, so its table is declared and measured here."""
    absolute = {"use_absolute_position_embeddings": True}
    spec = registry.MODELS["beit"].transfer
    loaded, target = loaded_beside("beit", VARIANT["beit"], LARGER, {**pins_of("beit"), **absolute}, archives,
                                   source_args=absolute)
    got, plan, message = verdict("beit", "image_size 32 to 64, absolute table, production order", loaded, target, spec)
    assert got == "PASS", message
    rules = {path: rule for path, rule, *_ in plan.resampled}
    assert rules.pop("beit_backbone/pos_embed/pos_embedding") == "vit_pos_bicubic"
    assert sorted(rules) == sorted(relative_tables(target)) and set(rules.values()) == {"beit_relpos_cubic"}
    assert sorted(path for path, *_ in plan.resampled) == reshaped(loaded, target)


# --- the pretrain rows (plan 5c6cc105 Step 6, D-008): their own family only, the decoder their head ------------------

#: Each pretrain row's decoder: the head a pretrain source on another tokenizer resets, and what a classifier drops.
DECODER = {ET_MIM: ("decoder_norm", "decoder_proj"), BEIT_MIM: ("decoder_norm", "decoder_head")}


def test_the_pretrain_rows_take_their_own_family_only_with_the_size_and_the_patch_fixed():
    for family, decoder in DECODER.items():
        spec = registry.MODELS[family].transfer
        assert (spec.head, spec.fixed, dict(spec.sources), spec.pinned) == (
            decoder, ("channels", "input_size", "patch_size"), {}, ()), family
        assert not spec.resample and spec.grow is None, family
        classifier = next(name for name, source in MIM_SOURCE.items() if source == family)
        assert registry.MODELS[classifier].transfer.sources[(PRETRAIN, family)].drop == decoder, family


@pytest.mark.parametrize("family", sorted(DECODER))
def test_production_order_a_pretrain_row_copies_every_weight_and_resets_exactly_its_decoder(family, archives):
    spec = registry.MODELS[family].transfer
    loaded, target = loaded_beside(family, VARIANT[family], SOURCE[PRETRAIN], {}, archives)
    got, plan, message = verdict(family, "same, production order", loaded, target, spec)
    assert got == "PASS" and len(plan.copied) == plan.n_source == plan.n_target > 0, message
    decoder = sorted(path for path in plan.copied if path.startswith(tuple(f"{name}/" for name in spec.head)))
    # Each declared head names a weight: a name that matched none would reset nothing on another tokenizer.
    assert all(any(path.startswith(f"{name}/") for path in decoder) for name in spec.head), decoder
    got, plan, message = verdict(family, "another tokenizer, production order", loaded, target, spec, head_reset=True)
    assert got == "PASS" and sorted(plan.reset) == sorted(plan.dropped) == decoder, message
    assert len(plan.copied) == plan.n_target - len(decoder)


@pytest.mark.parametrize("family", sorted(DECODER))
def test_production_order_a_pretrain_row_at_another_size_is_refused_at_the_weight_level(family, archives):
    """Behind the refusal before the claim (``fixed``): no rule resizes a pretrain row's tables; nothing is copied."""
    spec = registry.MODELS[family].transfer
    loaded, target = loaded_beside(family, VARIANT[family], {**SOURCE[PRETRAIN], "image_size": 64}, {}, archives)
    got, plan, message = verdict(family, "image_size 32 to 64, production order", loaded, target, spec)
    assert got == "REFUSE" and "nothing was copied" in message and plan.missing, message


# --- the language-model row (plan 7806a290 Step 6, D-005, D-040): its own family only, the vocabulary and the tokenizer fixed ---


def test_the_language_row_takes_its_own_family_only_and_every_fixed_key_of_every_row_has_a_name_in_the_refusal():
    spec = registry.MODELS[GPT2_LM].transfer
    # No head (a source of this family has the same tokenizer, so an untied lm_head is copied like any weight), no rule, no pin: a
    # different tokenizer is refused before the claim by the two fixed keys. The length is not one of them.
    assert (spec.fixed, spec.head, dict(spec.sources), spec.pinned) == (("vocab_size", "tokenizer"), (), {}, ())
    assert not spec.resample and spec.grow is None and not spec.positional
    for family, row in registry.MODELS.items():  # a fixed key with no name is a KeyError where the refusal is worded
        assert set(row.transfer.fixed) <= set(GEOMETRY_NAMES), (family, row.transfer.fixed)


def language_pair(archives, **target: Any):
    """``(loaded source, target)`` of ``gpt2_lm`` in the production order; ``target`` changes the target's geometry."""
    return loaded_beside(GPT2_LM, VARIANT[GPT2_LM], {**SOURCE[LANGUAGE_MODEL], **target}, {}, archives)


def test_production_order_a_language_row_copies_every_weight(archives):
    loaded, target = language_pair(archives)
    got, plan, message = verdict(GPT2_LM, "same, production order", loaded, target, registry.MODELS[GPT2_LM].transfer)
    assert got == "PASS" and len(plan.copied) == plan.n_source == plan.n_target > 0, message
    assert not (plan.reset or plan.dropped or plan.grown or plan.resampled)


def test_production_order_a_language_row_at_another_length_copies_every_weight(archives):
    """No weight depends on ``seq_len`` (the position table has ``max_seq_len`` rows), so it is no ``fixed`` key: a source trained
    on shorter windows seeds a run on longer ones (D-040)."""
    loaded, target = language_pair(archives, seq_len=2 * SEQ_LEN)
    got, plan, message = verdict(GPT2_LM, f"seq_len {SEQ_LEN} to {2 * SEQ_LEN}, production order", loaded, target,
                                 registry.MODELS[GPT2_LM].transfer)
    assert got == "PASS" and len(plan.copied) == plan.n_source == plan.n_target > 0, message


def test_production_order_a_language_row_at_another_vocabulary_is_refused_at_the_weight_level(archives):
    """Behind the refusal before the claim (``fixed``): the embedding table has another number of rows; nothing is copied."""
    loaded, target = language_pair(archives, vocab=ASCII_VOCAB + 19)
    got, plan, message = verdict(GPT2_LM, "vocab 101 to 120, production order", loaded, target, registry.MODELS[GPT2_LM].transfer)
    assert got == "REFUSE" and "nothing was copied" in message and plan.copied, message
    assert len(plan.missing) == 1 and "word_embeddings/embeddings" in plan.missing[0]
    assert f"target ({ASCII_VOCAB + 19}, 64)" in plan.missing[0] and f"source ({ASCII_VOCAB}, 64)" in plan.missing[0]


# --- the tokenizer row (plan-2026-11-25T124419-fa94f59c Step 8, D-012, D-050): its own family only, no head ---------------------------

#: The gate's model: the variant's widths and the recipe's codebook, derived from the tables and never typed.
TOKENIZER_DEFAULTS = {**tokenizer_models.VARIANTS["small"], "num_embeddings": tokenizer_models.CODEBOOK["cifar10"]}
#: Each ``fixed`` key the engine refuses at the weight level: ``(the change of the geometry, the change of model_args)``. The census counts
#: them, so a ``fixed`` key it never changes would be a declaration nothing measures.
TOKENIZER_CHANGES = {"channels": ({"channels": 1}, {}), "patch_size": ({"patch": 8}, {}), "num_embeddings": ({}, {"num_embeddings": 128})}
#: The ``fixed`` key the engine would TAKE: a convolutional tokenizer has no weight that depends on the image size, so the row refuses a size
#: before the claim by policy (the plan's contract is the same size) and not because the weights could not be copied.
TOKENIZER_BY_POLICY = {"input_size": {"image_size": 64}}
#: Keys of the model the record does not carry (so no refusal before the claim), which the engine refuses after it: a width, a depth, an
#: embedding size, and an EMA codebook (its moving averages are weights).
TOKENIZER_UNRECORDED = {"hidden_channels": 32, "num_res_blocks": 2, "embedding_dim": 16, "use_ema": True}


def tokenizer_pair(archives, geometry=None, model_args=None, source_args=None):
    """``(loaded source, target)`` of the tokenizer row in the production order; the target takes ``geometry`` and ``model_args`` over the tiny
    model's, the source is the tiny one (or ``source_args``)."""
    return loaded_beside(TOKENIZER_FAMILY, "small", {**SOURCE[TOKENIZER], **(geometry or {})}, model_args or {}, archives,
                         source_args=source_args)


def test_the_tokenizer_row_takes_its_own_family_only_with_the_geometry_and_the_codebook_fixed():
    spec = registry.MODELS[TOKENIZER_FAMILY].transfer
    assert (spec.fixed, spec.head, dict(spec.sources), spec.pinned) == (
        ("channels", "input_size", "patch_size", "num_embeddings"), (), {}, ())
    assert not spec.resample and spec.grow is None and not spec.positional
    # A fixed key the census never changes is a declaration nothing measures; a key it changes and the row does not declare is a hole.
    assert sorted(spec.fixed) == sorted({*TOKENIZER_CHANGES, *TOKENIZER_BY_POLICY})
    assert set(spec.fixed) <= set(GEOMETRY_NAMES)


@pytest.mark.parametrize("size", ["tiny", "default"])
def test_production_order_a_tokenizer_row_copies_every_weight(size, archives):
    args = {} if size == "tiny" else TOKENIZER_DEFAULTS
    loaded, target = tokenizer_pair(archives, model_args=args, source_args=args)
    got, plan, message = verdict(TOKENIZER_FAMILY, f"same, {size}, production order", loaded, target, registry.MODELS[TOKENIZER_FAMILY].transfer)
    assert got == "PASS" and len(plan.copied) == plan.n_source == plan.n_target == len(target.weights) > 0, message
    assert not (plan.reset or plan.dropped or plan.grown or plan.resampled)


@pytest.mark.parametrize("key", sorted(TOKENIZER_CHANGES))
def test_production_order_a_change_of_a_fixed_key_is_refused_at_the_weight_level(key, archives):
    """Behind the refusal before the claim (``fixed``): the engine refuses it too, and nothing is copied."""
    geometry, model_args = TOKENIZER_CHANGES[key]
    loaded, target = tokenizer_pair(archives, geometry, model_args)
    got, _, message = verdict(TOKENIZER_FAMILY, f"{key} changed, production order", loaded, target, registry.MODELS[TOKENIZER_FAMILY].transfer)
    assert got == "REFUSE" and message.startswith("Invalid value for 'init': ") and "nothing was copied" in message, message


def test_production_order_the_image_size_is_refused_before_the_claim_by_policy_though_every_weight_would_copy(archives):
    """D-050: a size is a ``fixed`` key because the plan's contract is the same size, not because a weight depends on it. If this fails, a
    size-dependent weight exists and the key is fixed by fact: say so where the row declares it."""
    ((key, change),) = TOKENIZER_BY_POLICY.items()
    loaded, target = tokenizer_pair(archives, change)
    got, plan, message = verdict(TOKENIZER_FAMILY, "image_size 32 to 64, production order", loaded, target, registry.MODELS[TOKENIZER_FAMILY].transfer)
    assert got == "PASS" and len(plan.copied) == plan.n_source == plan.n_target > 0, message
    assert key in registry.MODELS[TOKENIZER_FAMILY].transfer.fixed


@pytest.mark.parametrize("key", sorted(TOKENIZER_UNRECORDED))
def test_production_order_a_width_a_depth_or_an_embedding_the_record_does_not_carry_is_refused_by_the_engine(key, archives):
    spec = registry.MODELS[TOKENIZER_FAMILY].transfer
    loaded, target = tokenizer_pair(archives, model_args={key: TOKENIZER_UNRECORDED[key]})
    got, _, message = verdict(TOKENIZER_FAMILY, f"{key} changed, production order", loaded, target, spec)
    assert got == "REFUSE" and "nothing was copied" in message, message
    assert key not in spec.fixed, "the record has no such key: the refusal before the claim cannot name it"


def test_production_order_a_commitment_cost_has_no_weight_so_every_weight_is_copied(archives):
    loaded, target = tokenizer_pair(archives, model_args={"commitment_cost": 0.5})
    got, plan, message = verdict(TOKENIZER_FAMILY, "commitment_cost changed, production order", loaded, target,
                                 registry.MODELS[TOKENIZER_FAMILY].transfer)
    assert got == "PASS" and len(plan.copied) == plan.n_source == plan.n_target > 0, message
