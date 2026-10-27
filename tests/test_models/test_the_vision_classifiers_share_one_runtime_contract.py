"""The vision classifiers answer the same questions the same way, built and run.

The static half of the house shape (module layout, member order, constructor
surface) is read from source in ``test_the_vision_models_share_one_house_shape.py``.
This file builds tiny instances of the twelve classifiers (ResNet, CliffordNet,
HierarchicalCliffordNet, MobileNet V1 to V4, ConvNeXt V1 and V2, RADConvNet,
SqueezeNet V1 and V2) and asks each the same questions:

``unknown_variant_lists_the_keys``
    ``from_variant("nope")`` raises ``ValueError`` saying ``Unknown variant
    'nope'. Available variants: [...]`` and lists every key of ``MODEL_VARIANTS``.
``documented_override_is_accepted``
    A keyword the docstring advertises (an architecture field, or
    ``width_multiplier``, which overrides the variant's own value instead of
    multiplying it) reaches ``get_config``.
``pretrained_true_names_the_variant``
    ``pretrained=True`` raises ``NotImplementedError`` naming ``variant '<v>'``.
``missing_path_raises_file_not_found``
    A path that does not exist raises ``FileNotFoundError``.
``local_path_round_trip_headed`` and ``..._headless``
    A model saved to disk and loaded by ``from_variant(pretrained=path)`` carries
    the donor's tensors exactly, into a headed model and into one built with
    ``include_top=False``. A control inside the row proves the receiver's own
    initialization differs from the donor's, so equal tensors mean the load ran.
``local_path_refuses_a_trunk_mismatch``
    For the classifiers on the shared pretrained loader (``MIGRATED_LOADER_KEYS``), a
    checkpoint whose trunk does not fit (another variant, block type or stem patch,
    chosen so that SOME layers still match by name) makes ``from_variant(pretrained=path)``
    raise ``ValueError`` naming the unrestored layers, instead of loading the layers that
    fit and returning (D-040, ``utils/weight_transfer.py``).
``headless_attribute_is_none`` and ``headless_model_has_no_head_weights``
    ``include_top=False`` leaves ``classification_head`` defined as ``None`` and
    builds no head weight; a toy model that leaks its head into the headless build
    proves the second check can fail.
``explicit_build_equals_lazy`` and ``..._headless``
    ``build((None, H, W, C))`` materializes exactly what ``call`` runs: the same
    relative weight paths in the same order, and a non-zero ``count_params()``.
``compute_output_shape_*``
    ``compute_output_shape`` works on an unbuilt model, leaves it unbuilt, and
    equals the forward shape at an odd size, a non-square size and an unknown
    (``None``) size, headed and headless.
``config_round_trip``
    ``get_config()`` is JSON-safe with an ``Initializer`` instance given, and
    ``from_config`` rebuilds a model with the same config.
``config_fidelity``
    Idempotence is not fidelity: a ``get_config`` that silently drops an argument
    rebuilds a model with the default, and the rebuilt config still equals the
    original's. So every classifier is built with a NON-DEFAULT value for each
    constructor argument it exposes (:data:`PROBES`; a meta test keeps the table
    complete and non-default), and ``get_config`` must report exactly what was
    given, ``from_config`` must leave its dict alone, and the rebuilt model must
    report the same config and own the same weights. A toy class that drops a key,
    rewrites a value or mutates the dict proves the check can fail.
``invalid_values_name_the_value``
    A bad argument raises ``ValueError`` naming the argument as a word and the
    value, not the sub-layer's own spelling of it.
``no_replayed_draw``
    No live initializer instance reaches two sites: ``replayed_pairs`` finds no
    two kernels that carry one random draw, built with the default and with a
    seedless ``Initializer`` instance; an arm that compares no pair fails too.
``layer_names_follow_the_table``
    Every governed layer name follows the naming table (``stem``; a flat block
    ``block_{i}`` counted from 0 with its ``block_{i}_drop_path``; a staged block
    ``stage{S}_block{B}`` counted from 1, with ``stage{S}_mqa``;
    ``downsample_{i}_{norm,conv,depthwise,pointwise}``; ``classification_head``);
    the rule is a function of a list of names, so planted bad names prove it fails.
    For the finished families (:data:`FAMILY_CHILD_NAMES`) the rule is also
    positive: every layer the model owns directly and that holds a weight must
    match the family's own table, so a spelling the deny-list never heard of (a
    block called ``uib1_1``) is caught too.
``mobilenet_default_has_no_l2_losses``
    ``kernel_regularizer`` defaults to ``None``: ``model.losses == []``.

The no-replayed-draw and name rows also cover the models that are not classifiers (VAE, the
RBF pair, UcaEncoder, CapsNet, and the stem sites that are not migrated yet).
SqueezeNet is a functional model: it is exempt from the build-parity and the
unbuilt ``compute_output_shape`` rows because it is built in ``__init__``.

The ledger
----------
:data:`LEDGER` lists every row that fails today as a strict ``xfail`` carrying the
exception it raises, the number measured by running this file, the migration that
closes it and a ``signature``: a piece of the failure text that names which
violation it is, holding no count. A strict ``xfail`` that starts passing fails the
run, and a ledgered row that fails with text lacking its signature is a plain
failure (the exception type alone cannot tell one violation from another of the
same type), so the migration that fixes or changes a row edits its ledger line in
the same commit. An empty ledger means the contract is complete.
"""

import copy
import functools
import importlib
import inspect
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.utils.weight_transfer import load_pretrained_checkpoint

from .initializer_replay_oracle import replayed_pairs
from .roundtrip_instrument_oracle import relative_path, weight_map
from tests.layer_naming import PublicAutoName

NUM_CLASSES = 5
SRC = Path(__file__).resolve().parents[2] / "src" / "dl_techniques"


def _images(*shape: int, batch: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(batch, *shape)).astype("float32")


def _ctor(module: str, name: str, **base: Any) -> Callable[..., keras.Model]:
    """A builder calling ``Class(num_classes=5, **base, **knobs)``; knobs win."""

    def make(**knobs: Any) -> keras.Model:
        cls = getattr(importlib.import_module(module), name)
        return cls(**{"num_classes": NUM_CLASSES, **base, **knobs})

    return make


def _variant(module: str, name: str, variant: str, **base: Any) -> Callable[..., keras.Model]:
    """A builder calling ``Class.from_variant(variant, num_classes=5, **base, **knobs)``."""

    def make(**knobs: Any) -> keras.Model:
        cls = getattr(importlib.import_module(module), name)
        return cls.from_variant(variant, **{"num_classes": NUM_CLASSES, **base, **knobs})

    return make


_VISION = "dl_techniques.models.vision"


@dataclass(frozen=True)
class Classifier:
    """One classifier and the tiny configuration every row builds it with."""

    key: str
    module: str  # dotted module that defines the class
    name: str
    variant: str  # a variant name exercised through ``from_variant``
    input_shape: Tuple[int, int, int]
    make: Callable[..., keras.Model]  # tiny constructor-built (or variant-built) model
    tiny: Dict[str, Any]  # ``from_variant`` keywords that keep the variant small
    override: Dict[str, Any]  # a documented override, checked in ``get_config``
    head_markers: Tuple[str, ...]  # weight-path pieces that mean "this is the head"
    head_attribute: bool = True  # has a ``classification_head`` attribute
    functional: bool = False  # SqueezeNet: built in ``__init__``, no build/call override
    knob: bool = True  # takes ``kernel_initializer`` (so has an instance arm)
    invalid: Tuple[Tuple[Dict[str, Any], str, Any], ...] = ()  # (kwargs, argument, value)
    sizes: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)

    @property
    def cls(self) -> type:
        return getattr(importlib.import_module(self.module), self.name)

    def from_variant(self, **kwargs: Any) -> keras.Model:
        return self.cls.from_variant(
            self.variant, **{"num_classes": NUM_CLASSES, "input_shape": self.input_shape,
                             **self.tiny, **kwargs})


_ODD, _NON_SQUARE, _UNKNOWN = "odd", "non_square", "unknown"
_DEFAULT_SIZES = {_ODD: (33, 33), _NON_SQUARE: (32, 48), _UNKNOWN: None}

CLASSIFIERS: Tuple[Classifier, ...] = (
    Classifier(
        "resnet", f"{_VISION}.resnet.model", "ResNet", "resnet18", (32, 48, 3),
        _ctor(f"{_VISION}.resnet.model", "ResNet", blocks_per_stage=[1, 1],
              filters_per_stage=[8, 16], block_type="basic", input_shape=(32, 48, 3)),
        tiny={"blocks_per_stage": [1, 1], "filters_per_stage": [8, 16], "block_type": "basic"},
        override={"blocks_per_stage": [1, 1], "filters_per_stage": [8, 16], "block_type": "basic"},
        head_markers=("classification_head",),
        invalid=(({"num_classes": 0}, "num_classes", 0),), sizes=_DEFAULT_SIZES),
    Classifier(
        "cliffordnet", f"{_VISION}.cliffordnet.model", "CliffordNet", "nano", (32, 48, 3),
        _ctor(f"{_VISION}.cliffordnet.model", "CliffordNet", channels=16, depth=2,
              patch_size=2, shifts=[1, 2]),
        tiny={"channels": 16, "depth": 2, "patch_size": 2, "shifts": [1, 2]},
        override={"channels": 16, "depth": 2, "patch_size": 2, "shifts": [1, 2]},
        head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0),
                 ({"stochastic_depth_rate": -1.0}, "stochastic_depth_rate", -1.0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "hierarchical", f"{_VISION}.cliffordnet.hierarchical", "HierarchicalCliffordNet",
        "hier_345", (32, 48, 3),
        _ctor(f"{_VISION}.cliffordnet.hierarchical", "HierarchicalCliffordNet",
              stage_depths=[1, 1], stage_dims=[8, 16], patch_size=2, shifts=[1, 2]),
        tiny={"stage_depths": [1, 1], "stage_dims": [8, 16], "patch_size": 2, "shifts": [1, 2]},
        override={"stage_depths": [1, 1], "stage_dims": [8, 16], "patch_size": 2, "shifts": [1, 2]},
        head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0),
                 ({"stochastic_depth_rate": -1.0}, "stochastic_depth_rate", -1.0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "mobilenet_v1", f"{_VISION}.mobilenet.mobilenet_v1", "MobileNetV1", "small", (32, 48, 3),
        _ctor(f"{_VISION}.mobilenet.mobilenet_v1", "MobileNetV1", width_multiplier=0.25,
              input_shape=(32, 48, 3)),
        tiny={}, override={"width_multiplier": 0.25}, head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0), ({"num_classes": 0}, "num_classes", 0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "mobilenet_v2", f"{_VISION}.mobilenet.mobilenet_v2", "MobileNetV2", "small", (32, 48, 3),
        _ctor(f"{_VISION}.mobilenet.mobilenet_v2", "MobileNetV2", width_multiplier=0.35,
              input_shape=(32, 48, 3)),
        tiny={}, override={"width_multiplier": 0.25}, head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0), ({"num_classes": 0}, "num_classes", 0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "mobilenet_v3", f"{_VISION}.mobilenet.mobilenet_v3", "MobileNetV3", "small", (32, 48, 3),
        _ctor(f"{_VISION}.mobilenet.mobilenet_v3", "MobileNetV3", variant="small",
              width_multiplier=0.5, input_shape=(32, 48, 3)),
        tiny={"width_multiplier": 0.5}, override={"width_multiplier": 0.5},
        head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0), ({"num_classes": 0}, "num_classes", 0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "mobilenet_v4", f"{_VISION}.mobilenet.mobilenet_v4", "MobileNetV4", "small", (32, 48, 3),
        _variant(f"{_VISION}.mobilenet.mobilenet_v4", "MobileNetV4", "small", input_shape=(32, 48, 3)),
        tiny={}, override={"width_multiplier": 0.5, "use_attention": True},
        head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0), ({"num_classes": 0}, "num_classes", 0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "convnext_v1", f"{_VISION}.convnext.convnext_v1", "ConvNeXtV1", "tiny", (32, 48, 3),
        _ctor(f"{_VISION}.convnext.convnext_v1", "ConvNeXtV1", depths=[1, 1], dims=[8, 16],
              input_shape=(32, 48, 3)),
        tiny={"depths": [1, 1], "dims": [8, 16]}, override={"depths": [1, 1], "dims": [8, 16]},
        head_markers=("classification_head",), knob=False, sizes=_DEFAULT_SIZES),
    Classifier(
        "convnext_v2", f"{_VISION}.convnext.convnext_v2", "ConvNeXtV2", "tiny", (32, 48, 3),
        _ctor(f"{_VISION}.convnext.convnext_v2", "ConvNeXtV2", depths=[1, 1], dims=[8, 16],
              input_shape=(32, 48, 3)),
        tiny={"depths": [1, 1], "dims": [8, 16]}, override={"depths": [1, 1], "dims": [8, 16]},
        head_markers=("classification_head",), knob=False, sizes=_DEFAULT_SIZES),
    Classifier(
        "rad_convnet", f"{_VISION}.rad_convnet.model", "RADConvNet", "tiny", (32, 48, 3),
        _ctor(f"{_VISION}.rad_convnet.model", "RADConvNet", depths=[1, 1], dims=[8, 16],
              strides=2, input_shape=(32, 48, 3)),
        tiny={"depths": [1, 1], "dims": [8, 16], "strides": 2},
        override={"depths": [1, 1], "dims": [8, 16], "strides": 2},
        head_markers=("classification_head", "head_norm", "classifier"), knob=False,
        invalid=(({"num_classes": 0}, "num_classes", 0),), sizes=_DEFAULT_SIZES),
    Classifier(
        "squeezenet_v1", f"{_VISION}.squeezenet.squeezenet_v1", "SqueezeNetV1", "1.1",
        (48, 64, 3),
        _variant(f"{_VISION}.squeezenet.squeezenet_v1", "SqueezeNetV1", "1.1", input_shape=(48, 64, 3)),
        tiny={}, override={"dropout_rate": 0.25}, head_markers=("conv10",),
        head_attribute=False, functional=True,
        invalid=(({"use_bypass": "bogus"}, "use_bypass", "bogus"),),
        sizes={_NON_SQUARE: (48, 64)}),
    Classifier(
        "squeezenet_v2", f"{_VISION}.squeezenet.squeezenet_v2", "SqueezeNoduleNetV2", "v2",
        (48, 64, 3),
        _variant(f"{_VISION}.squeezenet.squeezenet_v2", "SqueezeNoduleNetV2", "v2",
                 input_shape=(48, 64, 3)),
        tiny={}, override={"dropout_rate": 0.25}, head_markers=("conv10",),
        head_attribute=False, functional=True, sizes={_NON_SQUARE: (48, 64)}),
)
CLASSIFIER_KEYS = tuple(c.key for c in CLASSIFIERS)
#: SqueezeNet is a functional model, built in ``__init__`` (the census's E-2 exemption): it has no unbuilt
#: state to compare and no head attribute, so those rows are not generated for it.
BUILT_LATER_KEYS = tuple(c.key for c in CLASSIFIERS if not c.functional)
HEAD_ATTRIBUTE_KEYS = tuple(c.key for c in CLASSIFIERS if c.head_attribute)


def classifier(key: str) -> Classifier:
    return next(c for c in CLASSIFIERS if c.key == key)


# ---------------------------------------------------------------------------
# The probes: a NON-DEFAULT value for every constructor argument of every classifier
# ---------------------------------------------------------------------------


def _he(seed: int) -> keras.initializers.Initializer:
    """A seeded instance, so its serialized form is the same every time it is built."""
    return keras.initializers.HeNormal(seed=seed)


def _l2() -> keras.regularizers.Regularizer:
    return keras.regularizers.L2(1e-3)


def _bias_init() -> keras.initializers.Initializer:
    return keras.initializers.RandomNormal(stddev=0.01, seed=4)


def _squeezenet_table(module: str, name: str, variant: str) -> Dict[str, Any]:
    cls = getattr(importlib.import_module(module), name)
    return copy.deepcopy(cls.MODEL_VARIANTS[variant])


_NO_BIAS_HEAD = {"classifier_use_bias": False}

#: ``key -> builder of the constructor keywords`` the fidelity row builds the model with.
#: Every keyword differs from the constructor's default; together they cover every
#: argument the constructor takes (``test_every_constructor_argument_has_a_probe_value``).
#: A builder, not a dict, so each row gets fresh regularizer and initializer objects.
PROBES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "resnet": lambda: dict(
        num_classes=7, blocks_per_stage=[1, 1], filters_per_stage=[8, 16],
        block_type="basic", kernel_regularizer=_l2(), kernel_initializer=_he(3),
        normalization_type="layer_norm", normalization_kwargs={"epsilon": 1e-3},
        activation_type="gelu", enable_deep_supervision=True, input_shape=(32, 48, 3),
        stem_type="cifar", head_config_overrides={"classifier_dropout_rate": 0.1}),
    "cliffordnet": lambda: dict(
        num_classes=7, channels=16, depth=2, patch_size=4, shifts=[1, 3], cli_mode="wedge",
        ctx_mode="abs", use_global_context=True, layer_scale_init=1e-4,
        stochastic_depth_rate=0.3, dropout_rate=0.4, use_bias=False,
        kernel_initializer=_he(3), bias_initializer=_bias_init(),
        kernel_regularizer=_l2(), bias_regularizer=keras.regularizers.L1(1e-4),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "hierarchical": lambda: dict(
        num_classes=7, stage_depths=[1, 1], stage_dims=[8, 16], patch_size=2, shifts=[1, 3],
        cli_mode="inner", ctx_mode="abs", layer_scale_init=1e-4, stochastic_depth_rate=0.3,
        dropout_rate=0.4, use_bias=False, kernel_initializer=_he(3),
        bias_initializer=_bias_init(), kernel_regularizer=_l2(),
        bias_regularizer=keras.regularizers.L1(1e-4), head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v1": lambda: dict(
        num_classes=7, width_multiplier=0.25, dropout_rate=0.3, weight_decay=1e-4,
        kernel_initializer=_he(3), input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v2": lambda: dict(
        num_classes=7, width_multiplier=0.35, dropout_rate=0.3, weight_decay=1e-4,
        kernel_initializer=_he(3), input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v3": lambda: dict(
        num_classes=7, variant="small", width_multiplier=0.5, dropout_rate=0.35,
        weight_decay=1e-4, kernel_initializer=_he(3), input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v4": lambda: dict(
        num_classes=7, depths=[1, 1], dims=[16, 32], block_types=["IB", "ExtraDW"],
        strides=[1, 2], width_multiplier=0.5, use_attention=True, attention_stages=[1],
        dropout_rate=0.35, weight_decay=1e-4, kernel_initializer=_he(3),
        input_shape=(64, 96, 3), head_config_overrides=dict(_NO_BIAS_HEAD)),
    "convnext_v1": lambda: dict(
        num_classes=7, depths=[1, 1], dims=[8, 16], drop_path_rate=0.2,
        stochastic_mode="gradient", kernel_size=5, activation="relu", use_bias=False,
        kernel_regularizer=_l2(), dropout_rate=0.1, spatial_dropout_rate=0.05, strides=2,
        use_gamma=False, use_softorthonormal_regularizer=True, input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "convnext_v2": lambda: dict(
        num_classes=7, depths=[1, 1], dims=[8, 16], drop_path_rate=0.2,
        stochastic_mode="gradient", kernel_size=5, activation="relu", use_bias=False,
        kernel_regularizer=_l2(), dropout_rate=0.1, spatial_dropout_rate=0.05, strides=2,
        use_gamma=False, use_softorthonormal_regularizer=True, input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "rad_convnet": lambda: dict(
        num_classes=7, depths=[1, 1], dims=[8, 16], rad_kernel_size=5, rad_groups=2,
        mlp_ratio=2, activation="relu", use_bias=False, dropout_rate=0.1, strides=2,
        input_shape=(32, 48, 3)),
    "squeezenet_v1": lambda: dict(
        num_classes=7,
        variant_config=_squeezenet_table(f"{_VISION}.squeezenet.squeezenet_v1", "SqueezeNetV1", "1.0"),
        use_bypass="simple", dropout_rate=0.25, kernel_regularizer=_l2(),
        kernel_initializer=_he(3), input_shape=(48, 64, 3)),
    "squeezenet_v2": lambda: dict(
        num_classes=7,
        variant_config=_squeezenet_table(f"{_VISION}.squeezenet.squeezenet_v2", "SqueezeNoduleNetV2", "v2"),
        dropout_rate=0.25, kernel_regularizer=_l2(), kernel_initializer=_he(3),
        input_shape=(48, 64, 3)),
}
#: A second probe for an argument the first cannot carry (SqueezeNet V2's 3-D model needs
#: a volume of at least 35 voxels per axis).
EXTRA_PROBES: Dict[str, Tuple[Callable[[], Dict[str, Any]], ...]] = {
    "squeezenet_v2": (lambda: dict(use_3d=True, input_shape=(36, 36, 36, 1)),),
}
#: The smallest architecture of each classifier; the headless probe is this plus
#: ``include_top=False``, which the headed probe cannot carry.
ARCH: Dict[str, Dict[str, Any]] = {
    "resnet": dict(blocks_per_stage=[1, 1], filters_per_stage=[8, 16], block_type="basic"),
    "cliffordnet": dict(channels=16, depth=2),
    "hierarchical": dict(stage_depths=[1, 1], stage_dims=[8, 16]),
    "mobilenet_v1": dict(width_multiplier=0.25),
    "mobilenet_v2": dict(width_multiplier=0.35),
    "mobilenet_v3": dict(variant="small", width_multiplier=0.5),
    "mobilenet_v4": dict(depths=[1, 1], dims=[16, 32], block_types=["IB", "ExtraDW"],
                         strides=[1, 2]),
    "convnext_v1": dict(depths=[1, 1], dims=[8, 16]),
    "convnext_v2": dict(depths=[1, 1], dims=[8, 16]),
    "rad_convnet": dict(depths=[1, 1], dims=[8, 16], strides=2),
    "squeezenet_v1": dict(input_shape=(48, 64, 3)),
    "squeezenet_v2": dict(input_shape=(48, 64, 3)),
}


def fidelity_probes(key: str) -> List[Dict[str, Any]]:
    """Every constructor-keyword dict the fidelity row builds ``key`` with."""
    return [PROBES[key](), *(extra() for extra in EXTRA_PROBES.get(key, ())),
            {**ARCH[key], "include_top": False}]


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Open:
    """A row that fails today: how it fails, what was measured, what closes it.

    ``signature`` is a substring of ``"<ExceptionType>: <message>"`` that names the
    failure's identity and holds no measured count. A ledgered row that fails with a
    message lacking it is a plain failure, not an expected one: a partial closure
    that leaves another violation of the same type shows up as red.
    """

    raises: type
    closed_by: str
    measured: str
    signature: str


#: Every row that fails today, keyed ``(rule, subject)``.
LEDGER: Dict[Tuple[str, str], Open] = {
    ("unknown_variant_lists_the_keys", "mobilenet_v2"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        "the message reads \"Unknown variant 'nope'. Available: [...]\", not \"Available "
        'variants: [...]"',
        'Available: ['),
    ("documented_override_is_accepted", "mobilenet_v1"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        "width_multiplier=0.25 gives 0.125: the variant's own value is multiplied, not "
        'overridden',
        "{'width_multiplier': 0.125}"),
    ("documented_override_is_accepted", "mobilenet_v2"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        "width_multiplier=0.25 gives 0.1875: the variant's own value is multiplied, not "
        'overridden',
        "{'width_multiplier': 0.1875}"),
    ("documented_override_is_accepted", "convnext_v1"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths'",
        "got multiple values for keyword argument 'depths'"),
    ("documented_override_is_accepted", "convnext_v2"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths'",
        "got multiple values for keyword argument 'depths'"),
    ("pretrained_true_names_the_variant", "mobilenet_v1"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': True}, no "
        'NotImplementedError',
        "Unrecognized keyword arguments passed to MobileNetV1: {'pretrained'"),
    ("pretrained_true_names_the_variant", "mobilenet_v2"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': True}, no "
        'NotImplementedError',
        "Unrecognized keyword arguments passed to MobileNetV2: {'pretrained'"),
    ("pretrained_true_names_the_variant", "rad_convnet"): Open(
        ValueError,
        "the RADConvNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': True}, no "
        'NotImplementedError',
        "Unrecognized keyword arguments passed to RADConvNet: {'pretrained'"),
    ("pretrained_true_names_the_variant", "squeezenet_v1"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': True}, no "
        'NotImplementedError',
        "Unrecognized keyword arguments passed to SqueezeNetV1: {'pretrained'"),
    ("pretrained_true_names_the_variant", "squeezenet_v2"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': True}, no "
        'NotImplementedError',
        "Unrecognized keyword arguments passed to SqueezeNoduleNetV2: {'pretrained'"),
    ("missing_path_raises_file_not_found", "mobilenet_v1"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to MobileNetV1: {'pretrained'"),
    ("missing_path_raises_file_not_found", "mobilenet_v2"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to MobileNetV2: {'pretrained'"),
    ("missing_path_raises_file_not_found", "convnext_v1"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths' (a tiny from_variant "
        'cannot be built)',
        "got multiple values for keyword argument 'depths'"),
    ("missing_path_raises_file_not_found", "convnext_v2"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths' (a tiny from_variant "
        'cannot be built)',
        "got multiple values for keyword argument 'depths'"),
    ("missing_path_raises_file_not_found", "rad_convnet"): Open(
        ValueError,
        "the RADConvNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to RADConvNet: {'pretrained'"),
    ("missing_path_raises_file_not_found", "squeezenet_v1"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to SqueezeNetV1: {'pretrained'"),
    ("missing_path_raises_file_not_found", "squeezenet_v2"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to SqueezeNoduleNetV2: {'pretrained'"),
    ("local_path_round_trip_headed", "mobilenet_v1"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to MobileNetV1: {'pretrained'"),
    ("local_path_round_trip_headed", "mobilenet_v2"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to MobileNetV2: {'pretrained'"),
    ("local_path_round_trip_headed", "convnext_v1"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths' (a tiny from_variant "
        'cannot be built)',
        "got multiple values for keyword argument 'depths'"),
    ("local_path_round_trip_headed", "convnext_v2"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths' (a tiny from_variant "
        'cannot be built)',
        "got multiple values for keyword argument 'depths'"),
    ("local_path_round_trip_headed", "rad_convnet"): Open(
        ValueError,
        "the RADConvNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to RADConvNet: {'pretrained'"),
    ("local_path_round_trip_headed", "squeezenet_v1"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to SqueezeNetV1: {'pretrained'"),
    ("local_path_round_trip_headed", "squeezenet_v2"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to SqueezeNoduleNetV2: {'pretrained'"),
    ("local_path_round_trip_headless", "mobilenet_v1"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to MobileNetV1: {'pretrained'"),
    ("local_path_round_trip_headless", "mobilenet_v2"): Open(
        ValueError,
        "the MobileNet V1 and V2 migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to MobileNetV2: {'pretrained'"),
    ("local_path_round_trip_headless", "convnext_v1"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths' (a tiny from_variant "
        'cannot be built)',
        "got multiple values for keyword argument 'depths'"),
    ("local_path_round_trip_headless", "convnext_v2"): Open(
        TypeError,
        "the ConvNeXt migration",
        "TypeError: got multiple values for keyword argument 'depths' (a tiny from_variant "
        'cannot be built)',
        "got multiple values for keyword argument 'depths'"),
    ("local_path_round_trip_headless", "rad_convnet"): Open(
        ValueError,
        "the RADConvNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to RADConvNet: {'pretrained'"),
    ("local_path_round_trip_headless", "squeezenet_v1"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to SqueezeNetV1: {'pretrained'"),
    ("local_path_round_trip_headless", "squeezenet_v2"): Open(
        ValueError,
        "the SqueezeNet migration",
        "ValueError: Unrecognized keyword arguments {'pretrained': <path>}, from_variant "
        'takes no pretrained path',
        "Unrecognized keyword arguments passed to SqueezeNoduleNetV2: {'pretrained'"),
    ("headless_attribute_is_none", "convnext_v1"): Open(
        AssertionError,
        "the ConvNeXt migration",
        'include_top=False leaves no classification_head attribute (hasattr is False)',
        'include_top=False leaves no attribute'),
    ("headless_attribute_is_none", "convnext_v2"): Open(
        AssertionError,
        "the ConvNeXt migration",
        'include_top=False leaves no classification_head attribute (hasattr is False)',
        'include_top=False leaves no attribute'),
    ("headless_attribute_is_none", "rad_convnet"): Open(
        AssertionError,
        "the RADConvNet migration",
        'include_top=False leaves no classification_head attribute (hasattr is False)',
        'include_top=False leaves no attribute'),
    ("explicit_build_equals_lazy", "mobilenet_v1"): Open(
        UserWarning,
        "the MobileNet V1 and V2 migration",
        'UserWarning: build() on a layer with no build() method leaves count_params() at 0',
        'does not have a `build()` method implemented'),
    ("explicit_build_equals_lazy_headless", "mobilenet_v1"): Open(
        UserWarning,
        "the MobileNet V1 and V2 migration",
        'UserWarning: build() on a layer with no build() method leaves count_params() at 0',
        'does not have a `build()` method implemented'),
    ("invalid_values_name_the_value::dropout_rate=2.0", "mobilenet_v1"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        'the message reads "classifier_dropout_rate must be in [0, 1), got 2.0" (a sub-'
        "layer's argument, not the model's)",
        'classifier_dropout_rate must be in [0, 1)'),
    ("invalid_values_name_the_value::dropout_rate=2.0", "mobilenet_v2"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        'the message reads "classifier_dropout_rate must be in [0, 1), got 2.0" (a sub-'
        "layer's argument, not the model's)",
        'classifier_dropout_rate must be in [0, 1)'),
    ("invalid_values_name_the_value::num_classes=0", "rad_convnet"): Open(
        AssertionError,
        "the RADConvNet migration",
        'num_classes=0 was accepted',
        'num_classes=0 was accepted'),
    ("invalid_values_name_the_value::use_bypass=bogus", "squeezenet_v1"): Open(
        AssertionError,
        "the SqueezeNet migration",
        "use_bypass='bogus' was accepted",
        "use_bypass='bogus' was accepted"),
    ("mobilenet_default_has_no_l2_losses", "mobilenet_v1"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        '28 regularization losses by default',
        'regularization losses by default'),
    ("mobilenet_default_has_no_l2_losses", "mobilenet_v2"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        '36 regularization losses by default',
        'regularization losses by default'),
    ("mobilenet_default_has_no_l2_losses", "mobilenet_v3"): Open(
        AssertionError,
        "the MobileNet kernel_regularizer rename",
        '43 regularization losses by default',
        'regularization losses by default'),
    ("mobilenet_default_has_no_l2_losses", "mobilenet_v4"): Open(
        AssertionError,
        "the MobileNet kernel_regularizer rename",
        '27 regularization losses by default',
        'regularization losses by default'),
    ("no_replayed_draw::instance", "squeezenet_v1"): Open(
        AssertionError,
        "the SqueezeNet migration",
        '276 of 325 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "squeezenet_v2"): Open(
        AssertionError,
        "the SqueezeNet migration",
        '120 of 153 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "vae"): Open(
        AssertionError,
        "the ConvUNext and VAE migration",
        '91 of 136 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "uca_encoder"): Open(
        AssertionError,
        "the RBFProtoNet and UcaEncoder migration",
        '3 of 120 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "capsnet"): Open(
        AssertionError,
        "the CapsNet migration",
        '21 of 21 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "capsnet"): Open(
        AssertionError,
        "the CapsNet migration",
        '21 of 21 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "capsnet_v2_legacy"): Open(
        AssertionError,
        "the CapsNet migration",
        '15 of 15 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "capsnet_v2_legacy"): Open(
        AssertionError,
        "the CapsNet migration",
        '15 of 15 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "capsnet_v2_resnet18"): Open(
        AssertionError,
        "the CapsNet migration",
        '6 of 276 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "capsnet_v2_resnet18"): Open(
        AssertionError,
        "the CapsNet migration",
        '6 of 276 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "siamfc"): Open(
        AssertionError,
        "the SiamFC and DaSiamRPN stem migration",
        '10 of 10 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::instance", "dasiamrpn"): Open(
        AssertionError,
        "the SiamFC and DaSiamRPN stem migration",
        '45 of 45 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "depth_anything_real"): Open(
        AssertionError,
        "the depth_anything stem migration",
        '12 of 3403 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "depth_anything_placeholder"): Open(
        AssertionError,
        "the depth_anything stem migration",
        '45 of 861 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("no_replayed_draw::default", "clifford_clip"): Open(
        AssertionError,
        "the CliffordCLIP stem migration",
        '776 of 9316 kernel pairs replay one draw',
        'kernel pairs replay one draw'),
    ("layer_names_follow_the_table", "mobilenet_v1"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        '1 violation(s): no layer is named `stem`',
        'no layer is named `stem`'),
    ("layer_names_follow_the_table", "mobilenet_v2"): Open(
        AssertionError,
        "the MobileNet V1 and V2 migration",
        '1 violation(s): no layer is named `stem`',
        'no layer is named `stem`'),
    ("layer_names_follow_the_table", "convnext_v1"): Open(
        AssertionError,
        "the ConvNeXt migration",
        '4 violation(s): stage_0_block_0, downsample_norm_0, downsample_conv_0, '
        'stage_1_block_0',
        'is not a table spelling'),
    ("layer_names_follow_the_table", "convnext_v2"): Open(
        AssertionError,
        "the ConvNeXt migration",
        '4 violation(s): stage_0_block_0, downsample_norm_0, downsample_conv_0, '
        'stage_1_block_0',
        'is not a table spelling'),
    ("layer_names_follow_the_table", "rad_convnet"): Open(
        AssertionError,
        "the RADConvNet migration",
        '13 violation(s): stage0_block0_rad_conv, stage0_block0_norm, stage0_block0_expand, '
        'stage0_block0_reduce, ...',
        'is not a table spelling'),
    ("layer_names_follow_the_table", "clifford_rbf_protonet"): Open(
        AssertionError,
        "the RBFProtoNet and UcaEncoder migration",
        '4 violation(s): clifford_block_0, drop_path_0, clifford_block_1, drop_path_1',
        'is not a table spelling'),
    ("layer_names_follow_the_table", "uca_encoder"): Open(
        AssertionError,
        "the RBFProtoNet and UcaEncoder migration",
        '7 violation(s): cpe_0, stage_0_block_0, merge_1, merge_norm_1, ...',
        'is not a table spelling'),
    ("layer_names_follow_the_table", "capsnet"): Open(
        AssertionError,
        "the CapsNet migration",
        '7 violation(s): conv_1, bn_1, relu_1, conv_2, ...',
        'is not a table spelling'),
    ("layer_names_follow_the_table", "capsnet_v2_legacy"): Open(
        AssertionError,
        "the CapsNet migration",
        '7 violation(s): legacy_conv_1, legacy_bn_1, legacy_relu_1, legacy_conv_2, ...',
        'is not a table spelling'),
    ("config_fidelity", "convnext_v1"): Open(
        AssertionError,
        "the ConvNeXt migration",
        "from_config rewrote the dict it was given",
        'from_config rewrote the dict it was given'),
    ("config_fidelity", "convnext_v2"): Open(
        AssertionError,
        "the ConvNeXt migration",
        "from_config rewrote the dict it was given",
        'from_config rewrote the dict it was given'),
    ("config_fidelity", "squeezenet_v1"): Open(
        AssertionError,
        "the SqueezeNet migration",
        "from_config rewrote the dict it was given, and the rebuilt model reports "
        "another kernel_initializer",
        'from_config rewrote the dict it was given'),
    ("config_fidelity", "squeezenet_v2"): Open(
        AssertionError,
        "the SqueezeNet migration",
        "from_config rewrote the dict it was given, and the rebuilt model reports "
        "another kernel_initializer",
        'from_config rewrote the dict it was given'),
}


def _param(rule: str, key: str, *rest: Any):
    """One ``pytest.param`` for ``rule`` on ``key`` (and any sub-case), xfail if the ledger says so."""
    row = LEDGER.get((_row_name(rule, rest), key))
    marks = []
    if row is not None:
        marks = [pytest.mark.xfail(
            strict=True, raises=row.raises,
            reason=f"closed by {row.closed_by}; measured: {row.measured}")]
    return pytest.param(key, *rest, id="-".join([key, *map(str, rest)]), marks=marks)


def _rows(rule: str, keys: Sequence[str], *axes: Sequence[Any]):
    """``pytest.param`` rows for ``rule`` over ``keys`` and every combination of ``axes``."""
    return [_param(rule, key, *rest) for key in keys for rest in itertools.product(*axes)]


def _row_name(rule: str, rest: Sequence[Any]) -> str:
    return "::".join([rule, *map(str, rest)])


class LedgerSignatureMismatch(RuntimeError):
    """A ledgered row failed, but not the way its ledger line says it fails.

    Not an ``AssertionError`` nor any type a ledger row names in ``raises``, so the
    strict ``xfail`` does not absorb it: the row is red.
    """


def failure_text(error: BaseException) -> str:
    """The text a ledger signature is matched against: ``"<Type>: <message>"``."""
    return f"{type(error).__name__}: {error}"


def with_signature_check(test: Callable, rule: str, axes: Sequence[str],
                         ledger: Dict[Tuple[str, str], Open]) -> Callable:
    """``test`` wrapped so that a ledgered row's failure must carry the row's signature.

    Interface contract: ``rule`` is the ledger rule name, ``axes`` the names of the
    test's sub-case parameters in the order :func:`_row_name` joins them; the test
    takes ``key`` and those parameters by keyword. A row with no ledger line is
    untouched. A ledgered row whose failure text lacks its ``signature`` raises
    :class:`LedgerSignatureMismatch`, chained to the original error; any other
    failure propagates unchanged.
    """

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return test(*args, **kwargs)
        except Exception as error:
            row = ledger.get((_row_name(rule, [kwargs[a] for a in axes]), kwargs["key"]))
            if row is not None and row.signature not in failure_text(error):
                raise LedgerSignatureMismatch(
                    f"{rule} on {kwargs['key']} failed without its signature "
                    f"{row.signature!r}: {failure_text(error)}") from error
            raise

    run.ledger_rule = rule
    return run


def ledgered(rule: str, *axes: str) -> Callable[[Callable], Callable]:
    """Decorator form of :func:`with_signature_check` over the module's :data:`LEDGER`."""
    return lambda test: with_signature_check(test, rule, axes, LEDGER)


# ---------------------------------------------------------------------------
# Rows that need a variant, a path or a built model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", _rows("unknown_variant_lists_the_keys", CLASSIFIER_KEYS))
@ledgered("unknown_variant_lists_the_keys")
def test_an_unknown_variant_lists_the_keys(key: str) -> None:
    c = classifier(key)
    with pytest.raises(ValueError) as caught:
        c.cls.from_variant("nope", num_classes=NUM_CLASSES)
    message = str(caught.value)
    assert "Unknown variant 'nope'. Available variants: [" in message, message
    missing = [v for v in c.cls.MODEL_VARIANTS if f"'{v}'" not in message]
    assert not missing, f"the message omits {missing}: {message}"


@pytest.mark.parametrize("key", _rows("documented_override_is_accepted", CLASSIFIER_KEYS))
@ledgered("documented_override_is_accepted")
def test_a_documented_override_reaches_the_config(key: str) -> None:
    c = classifier(key)
    model = c.cls.from_variant(c.variant, num_classes=NUM_CLASSES, input_shape=c.input_shape,
                               **c.override)
    config = json.loads(json.dumps(model.get_config()))
    got = {name: config[name] for name in c.override}
    assert got == json.loads(json.dumps(c.override)), got


@pytest.mark.parametrize("key", _rows("pretrained_true_names_the_variant", CLASSIFIER_KEYS))
@ledgered("pretrained_true_names_the_variant")
def test_pretrained_true_names_the_variant(key: str) -> None:
    c = classifier(key)
    with pytest.raises(NotImplementedError) as caught:
        c.from_variant(pretrained=True)
    assert f"variant '{c.variant}'" in str(caught.value), str(caught.value)


@pytest.mark.parametrize("key", _rows("missing_path_raises_file_not_found", CLASSIFIER_KEYS))
@ledgered("missing_path_raises_file_not_found")
def test_a_missing_checkpoint_raises_file_not_found(key: str, tmp_path: Path) -> None:
    c = classifier(key)
    with pytest.raises(FileNotFoundError):
        c.from_variant(pretrained=str(tmp_path / "missing.keras"))


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> Dict[str, Tuple[Path, keras.Model]]:
    """``{key: (saved path, donor)}``, one donor per classifier, filled on demand."""
    return {}


def _donor(key: str, checkpoints: Dict[str, Tuple[Path, keras.Model]],
           tmp_path_factory) -> Tuple[Path, keras.Model]:
    if key not in checkpoints:
        c = classifier(key)
        keras.utils.set_random_seed(1)
        donor = c.make()
        donor(_images(*c.input_shape))
        path = tmp_path_factory.mktemp(f"donor_{key}") / "donor.keras"
        donor.save(path)
        checkpoints[key] = (path, donor)
    return checkpoints[key]


def _assert_donor_tensors(receiver: keras.Model, donor: keras.Model, headless: bool) -> None:
    donor_weights = dict(weight_map(donor))
    received = weight_map(receiver)
    assert received, "the receiver has no weights"
    for path, value in received:
        assert path in donor_weights, f"{path} is not in the donor"
        np.testing.assert_array_equal(value, donor_weights[path], err_msg=path)
    if headless:
        assert len(received) < len(donor_weights), "the headless receiver kept the head"


def _round_trip(key: str, headless: bool, checkpoints, tmp_path_factory) -> None:
    c = classifier(key)
    path, donor = _donor(key, checkpoints, tmp_path_factory)
    keras.utils.set_random_seed(2)
    own = c.make(include_top=not headless)
    own(_images(*c.input_shape))
    donor_weights = dict(weight_map(donor))
    assert any(not np.array_equal(v, donor_weights[p]) for p, v in weight_map(own)
               if p in donor_weights), "control: the receiver's own init already equals the donor's"
    keras.utils.set_random_seed(2)
    receiver = c.from_variant(pretrained=str(path), include_top=not headless)
    _assert_donor_tensors(receiver, donor, headless)


@pytest.mark.parametrize("key", _rows("local_path_round_trip_headed", CLASSIFIER_KEYS))
@ledgered("local_path_round_trip_headed")
def test_a_local_checkpoint_loads_into_a_headed_model(key, checkpoints, tmp_path_factory) -> None:
    _round_trip(key, False, checkpoints, tmp_path_factory)


@pytest.mark.parametrize("key", _rows("local_path_round_trip_headless", CLASSIFIER_KEYS))
@ledgered("local_path_round_trip_headless")
def test_a_local_checkpoint_loads_into_a_headless_model(key, checkpoints, tmp_path_factory) -> None:
    _round_trip(key, True, checkpoints, tmp_path_factory)


#: The classifiers whose ``pretrained=<path>`` goes through ``load_pretrained_checkpoint``,
#: with a donor that shares layer names with the receiver and differs in its trunk:
#: ``(variant, from_variant keywords)``. Some layers fit by name, so the load used to
#: restore them, report the rest and return.
MIGRATED_LOADER_DONORS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "resnet": ("resnet18", {"block_type": "bottleneck"}),
    "cliffordnet": ("nano", {"patch_size": 1}),
    "hierarchical": ("hier_345", {"patch_size": 1}),
    "mobilenet_v3": ("large", {}),
    "mobilenet_v4": ("hybrid_medium", {}),
}
MIGRATED_LOADER_KEYS = tuple(MIGRATED_LOADER_DONORS)


@pytest.mark.parametrize(
    "key", _rows("local_path_refuses_a_trunk_mismatch", MIGRATED_LOADER_KEYS))
@ledgered("local_path_refuses_a_trunk_mismatch")
def test_a_checkpoint_whose_trunk_does_not_fit_is_refused(key, tmp_path) -> None:
    c = classifier(key)
    variant, donor_kwargs = MIGRATED_LOADER_DONORS[key]
    donor = c.cls.from_variant(variant, **{"num_classes": NUM_CLASSES,
                                           "input_shape": c.input_shape, **c.tiny,
                                           **donor_kwargs})
    donor.build((None, *c.input_shape))
    path = tmp_path / "donor.keras"
    donor.save(path)
    donor_layers = {layer.name for layer in donor.layers if layer.weights}
    own = c.make()
    own.build((None, *c.input_shape))
    assert donor_layers & {layer.name for layer in own.layers if layer.weights} - {
        "classification_head"}, "control: no trunk layer is shared by name, the old load restored nothing"
    with pytest.raises(ValueError):
        c.from_variant(pretrained=str(path))
    # The configuration check refuses these donors first (they differ in a structural argument);
    # with it switched off the layer check is what refuses them, which this row is about.
    with pytest.raises(ValueError) as caught:
        load_pretrained_checkpoint(own, str(path), c.input_shape,
                                   ignore_config_keys=tuple(own.get_config()))
    message = str(caught.value)
    assert "shape_mismatch" in message and "missing_in_source" in message, message


#: The same classifiers, with a donor that differs by ONE constructor argument that moves
#: no weight shape: ``(from_variant keywords, the config key that differs)``. The layer
#: check restores every layer of such a donor, so only the archive's own config can tell
#: the architectures apart.
SAME_SHAPE_DONORS: Dict[str, Tuple[Dict[str, Any], str]] = {
    "resnet": ({"activation_type": "gelu"}, "activation_type"),
    "cliffordnet": ({"ctx_mode": "abs"}, "ctx_mode"),
    "hierarchical": ({"ctx_mode": "abs"}, "ctx_mode"),
    "mobilenet_v3": ({"width_multiplier": 0.51}, "width_multiplier"),
    "mobilenet_v4": ({"strides": [1, 1, 2, 2, 1, 2, 1]}, "strides"),
}
SAME_SHAPE_KEYS = tuple(SAME_SHAPE_DONORS)


def test_the_same_shape_donors_cover_every_loader_family() -> None:
    assert set(SAME_SHAPE_KEYS) == set(MIGRATED_LOADER_KEYS)


@pytest.mark.parametrize(
    "key", _rows("local_path_refuses_another_architecture_of_equal_shape", SAME_SHAPE_KEYS))
@ledgered("local_path_refuses_another_architecture_of_equal_shape")
def test_a_checkpoint_of_another_architecture_with_every_shape_equal_is_refused(
        key, tmp_path) -> None:
    c = classifier(key)
    change, config_key = SAME_SHAPE_DONORS[key]
    donor = c.from_variant(**change)
    donor.build((None, *c.input_shape))
    path = tmp_path / "donor.keras"
    donor.save(path)
    own = c.from_variant()
    own.build((None, *c.input_shape))
    assert _weight_layout(own) == _weight_layout(donor), (
        "control: the donor must differ in no weight shape, or the layer check would refuse it")
    before = [keras.ops.convert_to_numpy(w) for w in own.weights]
    with pytest.raises(ValueError) as caught:
        own.load_pretrained_weights(str(path))
    assert f"{config_key}: checkpoint" in str(caught.value), str(caught.value)
    assert all(np.array_equal(b, keras.ops.convert_to_numpy(w))
               for b, w in zip(before, own.weights)), "a refused load wrote a weight"
    c.from_variant(**change, pretrained=str(path))  # the donor's own architecture loads


@pytest.mark.parametrize(
    "key", _rows("head_config_overrides_are_json", MIGRATED_LOADER_KEYS))
@ledgered("head_config_overrides_are_json")
def test_a_live_object_in_head_config_overrides_is_refused_by_name(key) -> None:
    c = classifier(key)
    with pytest.raises(ValueError, match=r"head_config_overrides\['kernel_regularizer'\]"):
        c.make(head_config_overrides={"kernel_regularizer": keras.regularizers.L2(1e-4)})
    serialized = {"kernel_regularizer": keras.regularizers.serialize(keras.regularizers.L2(1e-4))}
    model = c.make(head_config_overrides=serialized)
    assert json.loads(json.dumps(model.get_config()))["head_config_overrides"] == serialized


def _built(c: Classifier, **knobs: Any) -> keras.Model:
    model = c.make(**knobs)
    model(_images(*c.input_shape))
    return model


@pytest.mark.parametrize("key", _rows("headless_attribute_is_none", HEAD_ATTRIBUTE_KEYS))
@ledgered("headless_attribute_is_none")
def test_a_headless_model_still_defines_its_head_attribute(key: str) -> None:
    c = classifier(key)
    model = c.make(include_top=False)
    assert hasattr(model, "classification_head"), "include_top=False leaves no attribute"
    assert model.classification_head is None


def _head_paths(markers: Sequence[str], model: keras.Model) -> List[str]:
    return [w.path for w in model.weights if any(m in w.path for m in markers)]


def head_weight_problems(make: Callable[..., keras.Model], markers: Sequence[str],
                         inputs: np.ndarray) -> List[str]:
    """What is wrong with how ``make(include_top=...)`` builds its head weights.

    Interface contract: ``make`` returns an unbuilt model and takes ``include_top``;
    ``markers`` are the weight-path pieces that mean "this is the head"; ``inputs`` is
    one batch the model can run. Returns a list of problems, empty when the headed
    model has a head weight (the control) and the headless one has none. Raises what
    ``make`` or the forward pass raises.
    """
    headed, headless = make(include_top=True), make(include_top=False)
    headed(inputs)
    headless(inputs)
    out = [] if _head_paths(markers, headed) else ["control: the headed model has no head weight"]
    leftovers = _head_paths(markers, headless)
    return out + ([f"the headless model still builds {leftovers}"] if leftovers else [])


@pytest.mark.parametrize("key", _rows("headless_model_has_no_head_weights", CLASSIFIER_KEYS))
@ledgered("headless_model_has_no_head_weights")
def test_a_headless_model_builds_no_head_weight(key: str) -> None:
    c = classifier(key)
    assert head_weight_problems(c.make, c.head_markers, _images(*c.input_shape)) == []


class _ToyHeaded(PublicAutoName, keras.Model):
    """A one-layer body and a head named ``classification_head``.

    ``leaks`` makes the headless model build the head anyway; ``head_name`` renames it.
    """

    def __init__(self, include_top: bool = True, leaks: bool = False,
                 head_name: str = "classification_head", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.body = keras.layers.Dense(4, name="body")
        self.include_top = include_top
        self.head = (keras.layers.Dense(3, name=head_name)
                     if include_top or leaks else None)

    def call(self, inputs: Any) -> Any:
        x = self.body(inputs)
        return self.head(x) if self.head is not None else x


def test_the_head_weight_rule_sees_a_headless_model_that_still_builds_its_head() -> None:
    inputs = np.ones((2, 5), dtype="float32")
    markers = ("classification_head",)
    assert head_weight_problems(_ToyHeaded, markers, inputs) == []
    leaking = head_weight_problems(functools.partial(_ToyHeaded, leaks=True), markers, inputs)
    assert len(leaking) == 1 and "the headless model still builds" in leaking[0], leaking
    unmarked = head_weight_problems(functools.partial(_ToyHeaded, head_name="logits"), markers, inputs)
    assert unmarked and unmarked[0].startswith("control:"), unmarked


def _paths(model: keras.Model) -> List[str]:
    return [relative_path(model, w) for w in model.weights]


def _explicit_equals_lazy(c: Classifier, **knobs: Any) -> None:
    explicit = c.make(**knobs)
    explicit.build((None, *c.input_shape))
    assert explicit.weights and explicit.count_params() > 0, "build() created no weight"
    lazy = _built(c, **knobs)
    assert _paths(explicit) == _paths(lazy)


@pytest.mark.parametrize("key", _rows("explicit_build_equals_lazy", BUILT_LATER_KEYS))
@ledgered("explicit_build_equals_lazy")
def test_an_explicit_build_materializes_what_call_runs(key: str) -> None:
    _explicit_equals_lazy(classifier(key))


@pytest.mark.parametrize("key", _rows("explicit_build_equals_lazy_headless", BUILT_LATER_KEYS))
@ledgered("explicit_build_equals_lazy_headless")
def test_an_explicit_build_of_a_headless_model_materializes_what_call_runs(key: str) -> None:
    _explicit_equals_lazy(classifier(key), include_top=False)


def _forward_shape(c: Classifier, headless: bool, size: Tuple[int, int]) -> Tuple[Optional[int], ...]:
    model = c.make(include_top=not headless)
    out = model(_images(*size, c.input_shape[2]))
    return (None, *tuple(out.shape)[1:])


@pytest.mark.parametrize(
    ("key", "size", "head"),
    [_param("compute_output_shape", c.key, size, head) for c in CLASSIFIERS
     for size in (_ODD, _NON_SQUARE, _UNKNOWN) if size in c.sizes for head in ("headed", "headless")])
@ledgered("compute_output_shape", "size", "head")
def test_compute_output_shape_works_unbuilt_and_equals_the_forward_shape(key, size, head) -> None:
    c = classifier(key)
    headless = head == "headless"
    model = c.make(include_top=not headless)
    channels = c.input_shape[2]
    if c.sizes[size] is None:
        got = tuple(model.compute_output_shape((None, None, None, channels)))
        native = _forward_shape(c, headless, _DEFAULT_SIZES[_NON_SQUARE])
        want = native[:1] + (None, None) + native[3:] if headless else native
    else:
        height, width = c.sizes[size]
        if not c.functional:
            assert not model.built
        got = tuple(model.compute_output_shape((None, height, width, channels)))
        want = _forward_shape(c, headless, (height, width))
    if not c.functional:
        assert not model.built, "compute_output_shape built the model"
    assert got == want, f"compute_output_shape {got} against the forward shape {want}"


def _config_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


@pytest.mark.parametrize("key", _rows("config_round_trip", CLASSIFIER_KEYS))
@ledgered("config_round_trip")
def test_get_config_is_json_safe_and_from_config_rebuilds_the_model(key: str) -> None:
    c = classifier(key)
    knobs = {"kernel_initializer": keras.initializers.HeNormal()} if c.knob else {}
    model = c.make(**knobs)
    config = model.get_config()
    text = json.dumps(config)
    rebuilt = c.cls.from_config(model.get_config())
    assert _config_equal(json.loads(text), rebuilt.get_config())


def _spec(value: Any) -> Any:
    """``value`` as ``get_config`` reports it: initializers and regularizers serialized."""
    if isinstance(value, keras.initializers.Initializer):
        return keras.initializers.serialize(value)
    if isinstance(value, keras.regularizers.Regularizer):
        return keras.regularizers.serialize(value)
    return value


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _weight_layout(model: keras.Model) -> List[Tuple[str, Tuple[int, ...]]]:
    return [(relative_path(model, w), tuple(w.shape)) for w in model.weights]


def config_fidelity_problems(cls: type, kwargs: Dict[str, Any],
                             input_shape: Tuple[int, ...]) -> List[str]:
    """What is wrong with how ``cls`` reports and rebuilds the model ``cls(**kwargs)``.

    Interface contract: ``kwargs`` is a complete constructor call (the arguments the
    caller GAVE, objects included); ``input_shape`` is one unbatched input the model can
    run, used to materialize the weights of the original and the rebuilt model. Returns
    a list of problems, empty when ``get_config`` is JSON-safe and reports every given
    argument as it was given (an initializer or regularizer as its serialized form),
    ``from_config`` leaves the dict it receives alone, and the rebuilt model reports the
    same config and owns the same weights. Raises what the constructor, ``json.dumps``
    or the forward pass raises.
    """
    model = cls(**kwargs)
    given = _json({name: _spec(value) for name, value in kwargs.items()})
    config = _json(model.get_config())
    problems = []
    for name, want in given.items():
        if name not in config:
            problems.append(f"get_config writes no key for `{name}`")
        elif config[name] != want:
            problems.append(f"get_config reports {name}={config[name]!r} for the argument "
                            f"given as {want!r}")
    handed = copy.deepcopy(config)
    rebuilt = cls.from_config(handed)
    if handed != config:
        problems.append("from_config rewrote the dict it was given")
    again = _json(rebuilt.get_config())
    if again != config:
        changed = sorted(name for name in {*config, *again} if config.get(name) != again.get(name))
        problems.append(f"the rebuilt model reports other values for {changed}")
    inputs = np.zeros((1, *input_shape), dtype="float32")
    model(inputs, training=False)
    rebuilt(inputs, training=False)
    if _weight_layout(rebuilt) != _weight_layout(model):
        problems.append("the rebuilt model owns other weights than the original")
    return problems


@pytest.mark.parametrize("key", _rows("config_fidelity", CLASSIFIER_KEYS))
@ledgered("config_fidelity")
def test_get_config_reproduces_every_constructor_argument(key: str) -> None:
    c = classifier(key)
    problems = []
    for kwargs in fidelity_probes(key):
        problems += config_fidelity_problems(
            c.cls, {"num_classes": NUM_CLASSES, **kwargs}, kwargs.get("input_shape", c.input_shape))
    assert not problems, sorted(set(problems))


@pytest.mark.parametrize("key", CLASSIFIER_KEYS)
def test_every_constructor_argument_has_a_probe_value_that_is_not_its_default(key: str) -> None:
    """The probe table is complete and each value differs from the default it replaces."""
    c = classifier(key)
    parameters = {name: p for name, p in inspect.signature(c.cls.__init__).parameters.items()
                  if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    probes = fidelity_probes(key)
    given = set().union(*probes)
    assert set(parameters) <= given, f"no probe value for {sorted(set(parameters) - given)}"
    assert given <= set(parameters), f"probe keys that are no argument: {sorted(given - set(parameters))}"
    same = [name for probe in probes for name, value in probe.items()
            if _json(_spec(value)) == _json(_spec(parameters[name].default))]
    assert not same, f"a probe value equals the default of {sorted(set(same))}"


class _ConfigToy(PublicAutoName, keras.Model):
    """A one-layer model with a faithful ``get_config``; the subclasses break it one way each."""

    def __init__(self, num_classes: int = 3, rate: float = 0.0,
                 kernel_initializer: Any = "he_normal", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.rate = rate
        self.kernel_initializer = keras.initializers.serialize(
            keras.initializers.get(kernel_initializer))
        self.dense = keras.layers.Dense(
            num_classes, kernel_initializer=self.kernel_initializer, name="dense")

    def call(self, inputs: Any, training: Any = None) -> Any:
        return self.dense(inputs)

    def get_config(self) -> Dict[str, Any]:
        config = {"num_classes": self.num_classes, "rate": self.rate,
                  "kernel_initializer": self.kernel_initializer}
        return {**super().get_config(), **config}


class LossyConfigToy(_ConfigToy):
    """Drops ``rate``: the rebuilt model silently has the default."""

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        del config["rate"]
        return config


class LyingConfigToy(_ConfigToy):
    """Reports another ``rate`` than it was given."""

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "rate": self.rate + 0.5}


class ForgetfulConfigToy(_ConfigToy):
    """Reports ``rate`` truthfully but rebuilds without it."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ForgetfulConfigToy":
        config = dict(config)
        config.pop("rate")
        return cls(**config)


def test_the_fidelity_check_passes_a_faithful_model_and_fails_each_flaw() -> None:
    kwargs = dict(num_classes=4, rate=0.3, kernel_initializer=_he(3))
    assert config_fidelity_problems(_ConfigToy, kwargs, (5,)) == []
    expected = {
        LossyConfigToy: "get_config writes no key for `rate`",
        LyingConfigToy: "get_config reports rate=0.8 for the argument given as 0.3",
        ForgetfulConfigToy: "the rebuilt model reports other values for ['rate']",
    }
    for toy, problem in expected.items():
        assert problem in config_fidelity_problems(toy, kwargs, (5,)), toy.__name__


def test_the_fidelity_check_sees_a_from_config_that_rewrites_its_dict() -> None:
    class PoppingConfigToy(_ConfigToy):
        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "PoppingConfigToy":
            config.pop("trainable")
            return cls(**config)

    kwargs = dict(num_classes=4, rate=0.3, kernel_initializer=_he(3))
    assert "from_config rewrote the dict it was given" in config_fidelity_problems(
        PoppingConfigToy, kwargs, (5,))


def names_the_value(message: str, argument: str, value: Any) -> bool:
    """Whether ``message`` spells ``argument`` as a word (``classifier_dropout_rate``
    does not spell ``dropout_rate``) and prints ``value``."""
    word = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(argument)}(?![A-Za-z0-9_])")
    return bool(word.search(message)) and str(value) in message


def test_the_value_rule_can_fail() -> None:
    assert names_the_value("dropout_rate must be in [0, 1), got 2.0", "dropout_rate", 2.0)
    assert not names_the_value("classifier_dropout_rate must be in [0, 1), got 2.0", "dropout_rate", 2.0)
    assert not names_the_value("drop_path_rate must be in [0, 1), got -0.090909",
                               "stochastic_depth_rate", -1.0)
    assert not names_the_value("dropout_rate must be in [0, 1)", "dropout_rate", 2.0)


@pytest.mark.parametrize(
    ("key", "case"),
    [_param("invalid_values_name_the_value", c.key, f"{argument}={value}")
     for c in CLASSIFIERS for _, argument, value in c.invalid])
@ledgered("invalid_values_name_the_value", "case")
def test_an_invalid_value_is_refused_by_name(key: str, case: str) -> None:
    c = classifier(key)
    kwargs, argument, value = next(
        (kw, arg, val) for kw, arg, val in c.invalid if f"{arg}={val}" == case)
    try:
        c.make(**kwargs)
    except ValueError as error:
        assert names_the_value(str(error), argument, value), str(error)
    else:
        raise AssertionError(f"{argument}={value!r} was accepted")


_MOBILENETS = tuple(k for k in CLASSIFIER_KEYS if k.startswith("mobilenet"))


@pytest.mark.parametrize("key", _rows("mobilenet_default_has_no_l2_losses", _MOBILENETS))
@ledgered("mobilenet_default_has_no_l2_losses")
def test_a_default_mobilenet_adds_no_regularization_loss(key: str) -> None:
    c = classifier(key)
    model = _built(c)
    assert len(model.losses) == 0, f"{len(model.losses)} regularization losses by default"


# ---------------------------------------------------------------------------
# No live initializer instance reaches two sites
# ---------------------------------------------------------------------------


def _kw(init: Optional[keras.initializers.Initializer]) -> Dict[str, Any]:
    return {} if init is None else {"kernel_initializer": init}


def _forward(model: keras.Model, *inputs: Any) -> keras.Model:
    model(*inputs, training=False)
    return model


def _selector(cls: type, value: str) -> Dict[str, str]:
    """CapsNetV2's backbone selector under whichever spelling the constructor has."""
    params = inspect.signature(cls.__init__).parameters
    return {next(name for name in ("stem_type", "stem") if name in params): value}


def _build_resnet(init):
    from dl_techniques.models.vision.resnet.model import ResNet
    return _forward(ResNet.from_variant("resnet18", num_classes=10, input_shape=(64, 64, 3),
                                        **_kw(init)), _images(64, 64, 3))


def _build_cliffordnet(init):
    from dl_techniques.models.vision.cliffordnet.model import CliffordNet
    return _forward(CliffordNet.from_variant("nano", num_classes=5, **_kw(init)), _images(16, 16, 3))


def _build_hierarchical(init):
    from dl_techniques.models.vision.cliffordnet.hierarchical import HierarchicalCliffordNet
    return _forward(HierarchicalCliffordNet.from_variant("hier_345", num_classes=5, **_kw(init)),
                    _images(16, 16, 3))


def _mobilenet(module: str, name: str, variant: str, size: int = 32):
    def build(init):
        cls = getattr(importlib.import_module(f"{_VISION}.mobilenet.{module}"), name)
        return _forward(cls.from_variant(variant, num_classes=10, input_shape=(size, size, 3),
                                         **_kw(init)), _images(size, size, 3))
    return build


def _convnext(module: str, name: str):
    def build(init):
        cls = getattr(importlib.import_module(f"{_VISION}.convnext.{module}"), name)
        return _forward(cls.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3)),
                        _images(32, 32, 3))
    return build


def _build_rad(init):
    from dl_techniques.models.vision.rad_convnet.model import RADConvNet
    return _forward(RADConvNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3)),
                    _images(32, 32, 3))


def _build_squeezenet_v1(init):
    from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1
    return _forward(SqueezeNetV1.from_variant("1.0", num_classes=4, input_shape=(64, 64, 3),
                                              **_kw(init)), _images(64, 64, 3))


def _build_squeezenet_v2(init):
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
    return _forward(SqueezeNoduleNetV2.from_variant("v2", num_classes=4, input_shape=(64, 64, 1),
                                                    **_kw(init)), _images(64, 64, 1))


def _build_vae(init):
    from dl_techniques.models.vision.vae.model import VAE
    return _forward(VAE.from_variant("small", input_shape=(32, 32, 3), **_kw(init)),
                    np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32"))


def _build_convunext(init):
    from dl_techniques.models.vision.convunext.model import create_convunext_variant
    return _forward(create_convunext_variant("tiny", (32, 32, 3)),
                    np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32"))


def _build_rbf(init):
    from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet
    return _forward(RBFProtoNet(input_shape=(32, 32, 3), num_classes=7, stem_filters=8,
                                filters_per_stage=[8, 16], feature_dim=16), _images(32, 32, 3))


def _build_clifford_rbf(init):
    from dl_techniques.models.vision.rbf_protonet.model import CliffordRBFProtoNet
    return _forward(CliffordRBFProtoNet(input_shape=(32, 32, 3), num_classes=7, channels=16,
                                        depth=2, feature_dim=16), _images(32, 32, 3))


def _build_uca(init):
    from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder
    return _forward(UcaEncoder(stage_dims=[16, 32], stage_depths=[1, 2], num_heads=[2, 4]),
                    _images(16, 16, 3))


def _build_capsnet(init):
    from dl_techniques.models.vision.capsnet.model import CapsNet
    return _forward(CapsNet(num_classes=10, input_shape=(28, 28, 1), **_kw(init)),
                    np.random.default_rng(0).uniform(size=(2, 28, 28, 1)).astype("float32"))


def _capsnet_v2(backbone: str, shape: Tuple[int, int, int]):
    def build(init):
        from dl_techniques.models.vision.capsnet.model_v2 import CapsNetV2
        return _forward(CapsNetV2(num_classes=10, input_shape=shape, **_selector(CapsNetV2, backbone),
                                  **_kw(init)),
                        np.random.default_rng(0).uniform(size=(2, *shape)).astype("float32"))
    return build


def _build_siamfc(init):
    from dl_techniques.models.vision.siamfc.model import SiamFC
    rng = np.random.default_rng(0)
    return _forward(SiamFC(use_batch_norm=True, **_kw(init)),
                    (rng.normal(size=(1, 127, 127, 3)).astype("float32"),
                     rng.normal(size=(1, 255, 255, 3)).astype("float32")))


def _build_dasiamrpn(init):
    from dl_techniques.models.vision.dasiamrpn.model import DaSiamRPN
    rng = np.random.default_rng(0)
    return _forward(DaSiamRPN.from_variant("otb", **_kw(init)),
                    (rng.normal(size=(1, 127, 127, 3)).astype("float32"),
                     rng.normal(size=(1, 271, 271, 3)).astype("float32")))


def _build_deepsort(init):
    from dl_techniques.models.vision.deepsort.appearance import create_deepsort_embedding
    return _forward(create_deepsort_embedding(),
                    np.random.default_rng(0).uniform(size=(2, 128, 64, 3)).astype("float32"))


def _depth_anything(kind: str, size: int):
    def build(init):
        from dl_techniques.models.vision.depth_anything.model import DepthAnything
        model = DepthAnything(encoder_kind=kind, encoder_type="vit_s", image_shape=(size, size, 3))
        return _forward(model, np.random.default_rng(0).uniform(
            size=(1, size, size, 3)).astype("float32"))
    return build


def _build_clifford_clip(init):
    from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
    rng = np.random.default_rng(0)
    model = CliffordCLIP.from_variant("nano", vocab_size=1000, image_size=16, context_length=16,
                                      vision_patch_size=1)
    return _forward(model, {"image": rng.uniform(size=(1, 16, 16, 3)).astype("float32"),
                            "text": rng.integers(1, 100, size=(1, 16)).astype("int32")})


def _build_vq_vae_rotation(init):
    from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick
    return _forward(VQVAERotationTrick(
        num_embeddings=32, embedding_dim=16, input_shape=(32, 32, 3), hidden_channels=32,
        downsample_factor=4, num_res_blocks=1, norm_type="layer_norm"),
        np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32"))


def _build_sam1_prompt_encoder(init):
    from dl_techniques.models.vision_language.sam.sam1.prompt_encoder import PromptEncoder
    rng = np.random.default_rng(0)
    model = PromptEncoder(embed_dim=32, image_embedding_size=(8, 8), input_image_size=(128, 128),
                          mask_in_chans=8)
    model(points=(rng.uniform(0, 128, size=(2, 3, 2)).astype("float32"),
                  np.array([[1, 0, 1], [0, 1, 1]], dtype="int32")),
          boxes=rng.uniform(0, 128, size=(2, 1, 4)).astype("float32"),
          masks=rng.uniform(size=(2, 1, 32, 32)).astype("float32"), training=False)
    return model


#: ``key -> (builder(init), takes_kernel_initializer)``. The builder returns a built
#: model; ``init`` is ``None`` (the default arm) or a seedless ``HeNormal()`` instance
#: (the instance arm, only for a model that takes a ``kernel_initializer``). Presets are
#: the shipped ones where one exists (``nano``, ``hier_345``), because that is what a
#: trainer builds.
REPLAY_SUBJECTS: Dict[str, Tuple[Callable, bool]] = {
    "resnet": (_build_resnet, True),
    "cliffordnet": (_build_cliffordnet, True),
    "hierarchical": (_build_hierarchical, True),
    "mobilenet_v1": (_mobilenet("mobilenet_v1", "MobileNetV1", "small"), True),
    "mobilenet_v2": (_mobilenet("mobilenet_v2", "MobileNetV2", "small"), True),
    "mobilenet_v3": (_mobilenet("mobilenet_v3", "MobileNetV3", "small"), True),
    # 128 px: at 32 px the attention stages see a one-key softmax and Keras warns.
    "mobilenet_v4": (_mobilenet("mobilenet_v4", "MobileNetV4", "hybrid_medium", 128), True),
    "convnext_v1": (_convnext("convnext_v1", "ConvNeXtV1"), False),
    "convnext_v2": (_convnext("convnext_v2", "ConvNeXtV2"), False),
    "rad_convnet": (_build_rad, False),
    "squeezenet_v1": (_build_squeezenet_v1, True),
    "squeezenet_v2": (_build_squeezenet_v2, True),
    "vae": (_build_vae, True),
    "convunext": (_build_convunext, False),
    "rbf_protonet": (_build_rbf, False),
    "clifford_rbf_protonet": (_build_clifford_rbf, False),
    "uca_encoder": (_build_uca, False),
    "capsnet": (_build_capsnet, True),
    "capsnet_v2_legacy": (_capsnet_v2("legacy", (28, 28, 1)), True),
    "capsnet_v2_resnet18": (_capsnet_v2("resnet18", (32, 32, 3)), True),
    "siamfc": (_build_siamfc, True),
    "dasiamrpn": (_build_dasiamrpn, True),
    "deepsort": (_build_deepsort, False),
    "depth_anything_real": (_depth_anything("real", 56), False),
    "depth_anything_placeholder": (_depth_anything("placeholder", 64), False),
    "clifford_clip": (_build_clifford_clip, False),
    "vq_vae_rotation": (_build_vq_vae_rotation, False),
    "sam1_prompt_encoder": (_build_sam1_prompt_encoder, False),
}
_REPLAY_ARMS = [(key, arm) for key, (_, knob) in REPLAY_SUBJECTS.items()
                for arm in (("default", "instance") if knob else ("default",))]


def replay_report(key: str, arm: str) -> Tuple[List[str], int]:
    build, _ = REPLAY_SUBJECTS[key]
    keras.utils.set_random_seed(1234)
    model = build(keras.initializers.HeNormal() if arm == "instance" else None)
    return replayed_pairs(model)


def test_the_replay_oracle_sees_a_shared_instance_and_not_independent_draws() -> None:
    init = keras.initializers.HeNormal()
    keras.utils.set_random_seed(1234)
    shared = keras.Sequential([keras.layers.Dense(32, kernel_initializer=init, name="a"),
                               keras.layers.Dense(32, kernel_initializer=init, name="b")])
    shared.build((None, 32))
    keras.utils.set_random_seed(1234)
    fresh = keras.Sequential([keras.layers.Dense(32, kernel_initializer="he_normal", name="a"),
                              keras.layers.Dense(32, kernel_initializer="he_normal", name="b")])
    fresh.build((None, 32))
    assert replayed_pairs(shared)[0] and not replayed_pairs(fresh)[0]
    assert replayed_pairs(fresh)[1] > 0, "the clean model's kernels were never compared"


@pytest.mark.parametrize(
    ("key", "arm"),
    [_param("no_replayed_draw", key, arm) for key, arm in _REPLAY_ARMS])
@ledgered("no_replayed_draw", "arm")
def test_no_two_kernels_replay_one_draw(key: str, arm: str) -> None:
    pairs, compared = replay_report(key, arm)
    assert compared > 0, "the model exposes fewer than two kernels: nothing was compared"
    assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:2]}"


# ---------------------------------------------------------------------------
# Layer names follow the naming table
# ---------------------------------------------------------------------------

#: A layer name the table governs starts with one of these concept words; the
#: others (``conv3x3``, ``layer_norm``, ``fire2``) are the layers' own business.
_GOVERNED = re.compile(
    r"^(?:clifford_)?(?:block|drop_path|merge|merge_norm|cpe|conv_block)_\d"
    r"|^stage_?\d|^downsample_|^(?:gap|head_norm)$|^(?:legacy_)?(?:conv|bn|relu)_\d+$")
#: ... and must then read exactly like one of these (a flat or staged block with its
#: drop path, the CapsNet trunk, a downsampler, and the stem's inner layers).
_CANONICAL = re.compile(
    r"block_\d+(?:_drop_path)?"
    r"|conv_block_\d+(?:_(?:conv|norm|act)[12]|_pool)?"
    r"|stage[1-9]\d*(?:_(?:conv|norm|activation))?"
    r"|stage[1-9]\d*_block[1-9]\d*(?:_\w+)?"
    r"|stage[1-9]\d*_(?:mqa|cpe)(?:_\w+)?"
    r"|downsample_\d+_(?:norm|conv|depthwise|pointwise)")


def name_violations(names: Sequence[str], required: Sequence[str] = ()) -> List[str]:
    """The governed names in ``names`` that break the table, and the required names missing.

    Interface contract: ``names`` is the flat list of layer names of a built model
    (``[l.name for l in model._flatten_layers()]``); ``required`` is the exact names
    the model must have (``stem``, ``classification_head``). Returns a list of
    human-readable violations, empty when the model follows the table. Raises nothing.
    """
    out = [f"`{n}` is not a table spelling" for n in names
           if _GOVERNED.search(n) and not _CANONICAL.fullmatch(n)]
    out += [f"no layer is named `{r}`" for r in required if r not in names]
    return out


#: Models this rule does not read, and why. Their spelling predates the table and
#: they were not part of the work that adopted it.
NAME_RULE_EXEMPT_FILES: Dict[str, str] = {
    "models/language/cliffordlm": "CliffordNetLM keeps `clifford_block_{i}`",
    "models/world_models/video_jepa": "the video JEPA encoder keeps `clifford_block_{i}`",
    "models/vision_language/clifford_clip": "the vision tower's blocks are `vision_clifford_block_{i}`",
}

#: ``(names, required)`` pairs that are clean, and the plants that must fail.
_CLEAN_NAMES = ["stem", "stem_conv1", "stem_norm1", "block_0", "block_0_drop_path",
                "stage1_block1", "stage1_block1_conv1", "stage2_mqa", "stage1_cpe",
                "downsample_0_norm", "downsample_0_conv", "classification_head", "classifier"]
_PLANTED_NAMES = {
    "clifford_block_0": "the Clifford spelling of a flat block",
    "clifford_drop_path_0": "the Clifford drop path",
    "drop_path_0": "a drop path with no block",
    "stage0_block_0": "a 0-based, underscore hierarchical block",
    "stage_1_block_1": "an underscore staged block",
    "stage_1_mqa": "an underscore attention",
    "stage1_drop_path_1": "a drop path with no block",
    "downsample_norm_0": "the ConvNeXt downsampler",
    "downsample_0_bn": "a downsampler norm spelled bn",
    "merge_0": "UcaEncoder's merge",
    "cpe_0": "UcaEncoder's position conv",
    "gap": "RAD's pooling layer",
    "head_norm": "RAD's head norm",
    "conv_1": "CapsNet's first trunk convolution",
    "legacy_conv_1": "CapsNetV2's legacy trunk convolution",
}


def test_the_clean_names_pass_and_every_planted_name_fails() -> None:
    assert name_violations(_CLEAN_NAMES, required=("stem", "classification_head")) == []
    for name, why in _PLANTED_NAMES.items():
        assert name_violations([*_CLEAN_NAMES, name]), f"{name} ({why}) passed the name rule"
    assert name_violations(["classifier"], required=("stem",))


def test_the_exempt_files_exist_and_no_subject_lives_in_one() -> None:
    for path in NAME_RULE_EXEMPT_FILES:
        assert (SRC / path).is_dir(), path
    for c in CLASSIFIERS:
        assert not any(p.replace("/", ".") in c.module for p in NAME_RULE_EXEMPT_FILES), c.module


@dataclass(frozen=True)
class NamedSubject:
    key: str
    build: Callable[[], keras.Model]
    required: Tuple[str, ...]


def _named_classifier(c: Classifier) -> NamedSubject:
    required = ("stem",) + (("classification_head",) if c.head_attribute else ())
    return NamedSubject(c.key, lambda: _built(c), required)


def _named_other(key: str, replay_key: str, required: Tuple[str, ...]) -> NamedSubject:
    return NamedSubject(key, lambda: REPLAY_SUBJECTS[replay_key][0](None), required)


NAMED_SUBJECTS: Tuple[NamedSubject, ...] = (
    *(_named_classifier(c) for c in CLASSIFIERS),
    _named_other("rbf_protonet", "rbf_protonet", ("stem",)),
    _named_other("clifford_rbf_protonet", "clifford_rbf_protonet", ("stem",)),
    _named_other("uca_encoder", "uca_encoder", ("stem",)),
    _named_other("capsnet", "capsnet", ("conv_block_0",)),
    _named_other("capsnet_v2_legacy", "capsnet_v2_legacy", ("conv_block_0",)),
)


@pytest.mark.parametrize(
    "key", [_param("layer_names_follow_the_table", s.key) for s in NAMED_SUBJECTS])
@ledgered("layer_names_follow_the_table")
def test_the_layer_names_follow_the_naming_table(key: str) -> None:
    subject = next(s for s in NAMED_SUBJECTS if s.key == key)
    model = subject.build()
    names = [layer.name for layer in model._flatten_layers()]
    violations = name_violations(names, subject.required)
    assert not violations, sorted(set(violations))


#: The finished families, each with the one pattern every layer the model OWNS (a direct
#: sublayer, weighted or not: a drop path, an activation, a pooling, a flatten) must match.
#: The deny-list above only reads names it has heard of; this table is positive, so a
#: spelling nobody anticipated (``uib1_1`` for a MobileNetV4 block, ``dp0`` for a drop path)
#: fails here. Deeper names belong to the layers themselves (``stem_conv1``, ``dw_conv``,
#: ``classifier``) and are not read.
FAMILY_CHILD_NAMES: Dict[str, str] = {
    "resnet": r"stem|stage[1-9]\d*_block[1-9]\d*|classification_head|supervision_head_\d+",
    "cliffordnet": r"stem|block_\d+(?:_drop_path)?|classification_head",
    "hierarchical": (r"stem|stage[1-9]\d*_block[1-9]\d*(?:_drop_path)?"
                     r"|downsample_\d+_(?:depthwise|norm|pointwise)|classification_head"),
    "mobilenet_v3": r"stem|block_\d+|last_conv|last_bn|last_hard_swish|classification_head",
    "mobilenet_v4": r"stem|stage[1-9]\d*_(?:block[1-9]\d*|mqa)|classification_head",
}


def direct_children(model: keras.Model) -> List[str]:
    """The names of the layers ``model`` owns directly, whether or not they hold a weight.

    Interface contract: ``model.layers`` is the model's direct sublayers (Keras tracks
    the ones kept in a list or a dict attribute too). Returns the distinct names, sorted.
    A weightless child (a drop path, an activation, a pooling) is therefore read, which
    the weight paths of a built model cannot show.
    """
    return sorted({layer.name for layer in model.layers})


def child_name_violations(key: str, children: Sequence[str]) -> List[str]:
    """The ``children`` that the family ``key`` does not name by its table."""
    pattern = re.compile(FAMILY_CHILD_NAMES[key])
    return [f"`{name}` is not a {key} layer name" for name in children
            if not pattern.fullmatch(name)]


def _hybrid_v4() -> keras.Model:
    c = classifier("mobilenet_v4")
    model = c.cls.from_variant("hybrid_medium", num_classes=NUM_CLASSES, input_shape=(128, 128, 3))
    model(_images(128, 128, 3))
    return model


#: The builds that between them create every kind of direct child a family has: the head and
#: the headless build, the deep-supervision heads, the attention stages, a second stem.
FAMILY_BUILDS: Dict[str, Tuple[Callable[[], keras.Model], ...]] = {
    "resnet": (lambda: _built(classifier("resnet")),
               lambda: _built(classifier("resnet"), enable_deep_supervision=True),
               lambda: _built(classifier("resnet"), include_top=False)),
    "cliffordnet": (lambda: _built(classifier("cliffordnet")),
                    lambda: _built(classifier("cliffordnet"), patch_size=1)),
    "hierarchical": (lambda: _built(classifier("hierarchical")),
                     lambda: _built(classifier("hierarchical"), include_top=False)),
    "mobilenet_v3": (lambda: _built(classifier("mobilenet_v3")),),
    "mobilenet_v4": (lambda: _built(classifier("mobilenet_v4")), _hybrid_v4),
}


def test_every_finished_family_has_a_name_table_and_builds() -> None:
    assert set(FAMILY_CHILD_NAMES) == set(FAMILY_BUILDS) <= set(CLASSIFIER_KEYS)


@pytest.mark.parametrize("key", sorted(FAMILY_CHILD_NAMES))
def test_a_finished_family_owns_only_the_layer_names_of_its_table(key: str) -> None:
    for build in FAMILY_BUILDS[key]:
        children = direct_children(build())
        assert children, "control: the model owns no layer"
        assert child_name_violations(key, children) == []


def test_the_positive_name_rule_sees_a_spelling_the_deny_list_does_not_know() -> None:
    clean = ["stem", "stage1_block1", "stage1_mqa", "classification_head"]
    assert child_name_violations("mobilenet_v4", clean) == []
    for planted in ("uib1_1", "stage0_block1", "stage1_block0", "block_0", "head"):
        assert child_name_violations("mobilenet_v4", [*clean, planted]), planted
    # the deny-list reads none of them as governed, which is why the table exists
    assert name_violations(["uib1_1", "head"]) == []
    assert child_name_violations("cliffordnet", ["stem", "block_0", "stage1_block1"])
    assert child_name_violations("resnet", ["stem", "supervision_head_0", "block_0"])


def test_the_positive_name_rule_reads_a_weightless_layer() -> None:
    clean = ["stem", "block_0", "block_0_drop_path", "classification_head"]
    assert child_name_violations("cliffordnet", clean) == []
    for planted in ("dp0", "drop_path_0", "block_0_dp", "gap", "flatten", "activation"):
        assert child_name_violations("cliffordnet", [*clean, planted]), planted
    assert child_name_violations("mobilenet_v3", ["stem", "last_hard_swish"]) == []
    assert child_name_violations("mobilenet_v3", ["stem", "last_swish"])


def test_direct_children_lists_every_direct_sublayer_weighted_or_not() -> None:
    model = _ToyHeaded(head_name="classification_head")
    model.drop = keras.layers.Dropout(0.1, name="dropout_0")
    model(np.ones((2, 5), dtype="float32"))
    assert direct_children(model) == ["body", "classification_head", "dropout_0"]


# ---------------------------------------------------------------------------
# The ledger names only rows that exist
# ---------------------------------------------------------------------------


def _all_row_names() -> set:
    rows = set()
    classifier_rules = [
        "unknown_variant_lists_the_keys", "documented_override_is_accepted",
        "pretrained_true_names_the_variant", "missing_path_raises_file_not_found",
        "local_path_round_trip_headed", "local_path_round_trip_headless",
        "headless_attribute_is_none", "headless_model_has_no_head_weights",
        "explicit_build_equals_lazy", "explicit_build_equals_lazy_headless", "config_round_trip",
        "config_fidelity"]
    for c in CLASSIFIERS:
        rows |= {(rule, c.key) for rule in classifier_rules}
        rows |= {(_row_name("compute_output_shape", [size, head]), c.key)
                 for size in (_ODD, _NON_SQUARE, _UNKNOWN) for head in ("headed", "headless")}
        rows |= {(_row_name("invalid_values_name_the_value", [f"{a}={v}"]), c.key)
                 for _, a, v in c.invalid}
        if c.key.startswith("mobilenet"):
            rows.add(("mobilenet_default_has_no_l2_losses", c.key))
    rows |= {(_row_name("no_replayed_draw", [arm]), key) for key, arm in _REPLAY_ARMS}
    rows |= {("layer_names_follow_the_table", s.key) for s in NAMED_SUBJECTS}
    return rows


def test_the_ledger_names_only_rows_that_exist() -> None:
    assert set(LEDGER) <= _all_row_names(), sorted(set(LEDGER) - _all_row_names())
    for row in LEDGER.values():
        assert row.closed_by and row.measured and issubclass(row.raises, BaseException)
        assert row.signature, "a ledger row without a signature pins only its exception type"


def test_a_ledgered_failure_without_its_signature_is_a_plain_failure() -> None:
    ledger = {("toy_rule", "toy"): Open(AssertionError, "x", "m", "Available: ["),
              ("toy_rule::odd", "toy"): Open(AssertionError, "x", "m", "Available: [")}

    def failing(message: str, *axes: str) -> Callable:
        def test(key: str, **sub: str) -> None:
            raise AssertionError(message)
        return with_signature_check(test, "toy_rule", axes, ledger)

    with pytest.raises(AssertionError) as expected:
        failing("Unknown variant 'x'. Available: ['a']")(key="toy")
    assert not isinstance(expected.value, LedgerSignatureMismatch)
    with pytest.raises(LedgerSignatureMismatch, match="without its signature") as changed:
        failing("Unknown variant 'x'. Available variants: ['a']")(key="toy")
    assert isinstance(changed.value.__cause__, AssertionError)
    with pytest.raises(LedgerSignatureMismatch):
        failing("Unknown variant 'x'. Available variants: ['a']", "size")(key="toy", size="odd")
    with pytest.raises(AssertionError) as unledgered:
        failing("Available variants: ['a']")(key="other")
    assert not isinstance(unledgered.value, LedgerSignatureMismatch)
    assert with_signature_check(lambda key: 7, "toy_rule", (), ledger)(key="toy") == 7


def test_a_signature_mismatch_is_not_absorbed_by_any_strict_xfail() -> None:
    assert not any(issubclass(LedgerSignatureMismatch, row.raises) for row in LEDGER.values())


def test_every_ledger_rule_is_checked_for_its_signature() -> None:
    checked = {getattr(fn, "ledger_rule", None) for fn in globals().values() if inspect.isfunction(fn)}
    ruled = {rule.split("::")[0] for rule, _ in LEDGER}
    assert ruled <= checked, sorted(ruled - checked)
