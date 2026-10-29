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
    block called ``uib1_1``) is caught too. The two RBF models and UcaEncoder
    (:data:`OTHER_NUMBERED`) have no table of that kind but an exact set: the
    layers each owns directly are compared to the set its own arguments give.
``mobilenet_default_has_no_l2_losses``
    ``kernel_regularizer`` defaults to ``None``: ``model.losses == []``.

The no-replayed-draw and name rows also cover the models that are not classifiers (VAE, the
RBF pair, UcaEncoder, CapsNet, DeepSORT, SiamFC, DaSiamRPN, the DINOv2 backbone and wrapper with
each attention type its block accepts, CliffordCLIP, the VQ-VAE with the rotation trick and the SAM1
prompt encoder).
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

import ast
import collections
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
        tiny={"width_multiplier": 0.25}, override={"width_multiplier": 0.25},
        head_markers=("classification_head",),
        invalid=(({"dropout_rate": 2.0}, "dropout_rate", 2.0), ({"num_classes": 0}, "num_classes", 0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "mobilenet_v2", f"{_VISION}.mobilenet.mobilenet_v2", "MobileNetV2", "small", (32, 48, 3),
        _ctor(f"{_VISION}.mobilenet.mobilenet_v2", "MobileNetV2", width_multiplier=0.35,
              input_shape=(32, 48, 3)),
        tiny={"width_multiplier": 0.35}, override={"width_multiplier": 0.25},
        head_markers=("classification_head",),
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
        head_markers=("classification_head",), knob=False,
        invalid=(({"drop_path_rate": -1.0}, "drop_path_rate", -1.0),
                 ({"drop_path_rate": 2.0}, "drop_path_rate", 2.0),
                 ({"dropout_rate": 1.0}, "dropout_rate", 1.0),
                 ({"spatial_dropout_rate": 1.0}, "spatial_dropout_rate", 1.0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "convnext_v2", f"{_VISION}.convnext.convnext_v2", "ConvNeXtV2", "tiny", (32, 48, 3),
        _ctor(f"{_VISION}.convnext.convnext_v2", "ConvNeXtV2", depths=[1, 1], dims=[8, 16],
              input_shape=(32, 48, 3)),
        tiny={"depths": [1, 1], "dims": [8, 16]}, override={"depths": [1, 1], "dims": [8, 16]},
        head_markers=("classification_head",), knob=False,
        invalid=(({"drop_path_rate": -1.0}, "drop_path_rate", -1.0),
                 ({"drop_path_rate": 2.0}, "drop_path_rate", 2.0),
                 ({"dropout_rate": 1.0}, "dropout_rate", 1.0),
                 ({"spatial_dropout_rate": 1.0}, "spatial_dropout_rate", 1.0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "rad_convnet", f"{_VISION}.rad_convnet.model", "RADConvNet", "tiny", (32, 48, 3),
        _ctor(f"{_VISION}.rad_convnet.model", "RADConvNet", depths=[1, 1], dims=[8, 16],
              strides=2, input_shape=(32, 48, 3)),
        tiny={"depths": [1, 1], "dims": [8, 16], "strides": 2},
        override={"depths": [1, 1], "dims": [8, 16], "strides": 2},
        head_markers=("classification_head",), knob=False,
        invalid=(({"num_classes": 0}, "num_classes", 0),
                 ({"dropout_rate": 2.0}, "dropout_rate", 2.0)),
        sizes=_DEFAULT_SIZES),
    Classifier(
        "squeezenet_v1", f"{_VISION}.squeezenet.squeezenet_v1", "SqueezeNetV1", "1.1",
        (48, 64, 3),
        _variant(f"{_VISION}.squeezenet.squeezenet_v1", "SqueezeNetV1", "1.1", input_shape=(48, 64, 3)),
        tiny={}, override={"dropout_rate": 0.25}, head_markers=("conv10",),
        head_attribute=False, functional=True,
        invalid=(({"use_bypass": "bogus"}, "use_bypass", "bogus"),
                 ({"use_bypass": True}, "use_bypass", True),
                 ({"num_classes": 0}, "num_classes", 0),
                 ({"num_classes": True}, "num_classes", True),
                 ({"num_classes": 2.5}, "num_classes", 2.5),
                 ({"dropout_rate": 2.0}, "dropout_rate", 2.0),
                 ({"dropout_rate": float("nan")}, "dropout_rate", float("nan"))),
        sizes={_NON_SQUARE: (48, 64)}),
    Classifier(
        "squeezenet_v2", f"{_VISION}.squeezenet.squeezenet_v2", "SqueezeNoduleNetV2", "v2",
        (48, 64, 3),
        _variant(f"{_VISION}.squeezenet.squeezenet_v2", "SqueezeNoduleNetV2", "v2",
                 input_shape=(48, 64, 3)),
        tiny={}, override={"dropout_rate": 0.25}, head_markers=("conv10",),
        head_attribute=False, functional=True,
        invalid=(({"input_shape": (36, 36, 36, 1)}, "input_shape", (36, 36, 36, 1)),
                 ({"num_classes": 0}, "num_classes", 0),
                 ({"num_classes": True}, "num_classes", True),
                 ({"num_classes": 2.5}, "num_classes", 2.5),
                 ({"dropout_rate": 2.0}, "dropout_rate", 2.0),
                 ({"dropout_rate": float("nan")}, "dropout_rate", float("nan"))),
        sizes={_NON_SQUARE: (48, 64)}),
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
        num_classes=7, width_multiplier=0.25, dropout_rate=0.3, kernel_regularizer=_l2(),
        kernel_initializer=_he(3), input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v2": lambda: dict(
        num_classes=7, width_multiplier=0.35, dropout_rate=0.3, kernel_regularizer=_l2(),
        kernel_initializer=_he(3), input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v3": lambda: dict(
        num_classes=7, variant="small", width_multiplier=0.5, dropout_rate=0.35,
        kernel_regularizer=_l2(), kernel_initializer=_he(3), input_shape=(32, 48, 3),
        head_config_overrides=dict(_NO_BIAS_HEAD)),
    "mobilenet_v4": lambda: dict(
        num_classes=7, depths=[1, 1], dims=[16, 32], block_types=["IB", "ExtraDW"],
        strides=[1, 2], width_multiplier=0.5, use_attention=True, attention_stages=[1],
        dropout_rate=0.35, kernel_regularizer=_l2(), kernel_initializer=_he(3),
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
        mlp_ratio=2, activation="relu", use_bias=False, kernel_regularizer=_l2(),
        dropout_rate=0.1, strides=2, input_shape=(32, 48, 3),
        head_config_overrides={"classifier_dropout_rate": 0.1}),
    "squeezenet_v1": lambda: dict(
        num_classes=7, variant="1.1", use_bypass="simple", dropout_rate=0.25,
        kernel_regularizer=_l2(), kernel_initializer=_he(3), input_shape=(48, 64, 3)),
    "squeezenet_v2": lambda: dict(
        num_classes=7, variant="v1", dropout_rate=0.25, kernel_regularizer=_l2(),
        kernel_initializer=_he(3), input_shape=(48, 64, 3)),
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


#: Every row that fails today, keyed ``(rule, subject)``. Empty: the contract is complete.
LEDGER: Dict[Tuple[str, str], Open] = {}


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
    "mobilenet_v1": ("small", {"width_multiplier": 0.5}),
    "mobilenet_v2": ("small", {"width_multiplier": 0.5}),
    "mobilenet_v3": ("large", {}),
    "mobilenet_v4": ("hybrid_medium", {}),
    "convnext_v1": ("tiny", {"depths": [1], "dims": [16]}),
    "convnext_v2": ("tiny", {"depths": [1], "dims": [16]}),
    "rad_convnet": ("tiny", {"depths": [1], "dims": [16]}),
    # Another variant: its stem and first Fire module have other shapes, while every later
    # layer has the same name and shape, so SOME layers fit by name.
    "squeezenet_v1": ("1.0", {}),
    "squeezenet_v2": ("v1", {}),
}
MIGRATED_LOADER_KEYS = tuple(MIGRATED_LOADER_DONORS)
#: The loader families that take ``head_config_overrides``. SqueezeNet's head is its own ``conv10``
#: (no shared ``ClassificationHead`` to configure), so it has no such argument.
HEAD_OVERRIDE_KEYS = tuple(
    key for key in MIGRATED_LOADER_KEYS
    if "head_config_overrides" in inspect.signature(classifier(key).cls.__init__).parameters)


def test_only_the_conv10_models_take_no_head_override() -> None:
    assert set(MIGRATED_LOADER_KEYS) - set(HEAD_OVERRIDE_KEYS) == {"squeezenet_v1", "squeezenet_v2"}

#: What the layer check must name when a donor's trunk does not fit. A donor of another
#: depth or block type misses layers the receiver has; MobileNet V1 and V2 have one fixed
#: layer list, so a donor of another width has every layer and no shape that fits.
TRUNK_MISMATCH_REPORT: Dict[str, Tuple[str, ...]] = {
    "mobilenet_v1": ("shape_mismatch",),
    "mobilenet_v2": ("shape_mismatch",),
    # A donor of another variant has the same layers, in other widths.
    "squeezenet_v1": ("shape_mismatch",),
    "squeezenet_v2": ("shape_mismatch",),
}
_DEFAULT_TRUNK_MISMATCH_REPORT = ("shape_mismatch", "missing_in_source")


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
    named = TRUNK_MISMATCH_REPORT.get(key, _DEFAULT_TRUNK_MISMATCH_REPORT)
    assert all(word in message for word in named), message


#: The same classifiers, with a donor that differs by ONE constructor argument that moves
#: no weight shape: ``(from_variant keywords, the config key that differs)``. The layer
#: check restores every layer of such a donor, so only the archive's own config can tell
#: the architectures apart.
SAME_SHAPE_DONORS: Dict[str, Tuple[Dict[str, Any], str]] = {
    "resnet": ({"activation_type": "gelu"}, "activation_type"),
    "cliffordnet": ({"ctx_mode": "abs"}, "ctx_mode"),
    "hierarchical": ({"ctx_mode": "abs"}, "ctx_mode"),
    "mobilenet_v1": ({"width_multiplier": 0.2501}, "width_multiplier"),
    "mobilenet_v2": ({"width_multiplier": 0.36}, "width_multiplier"),
    "mobilenet_v3": ({"width_multiplier": 0.51}, "width_multiplier"),
    "mobilenet_v4": ({"strides": [1, 1, 2, 2, 1, 2, 1]}, "strides"),
    "convnext_v1": ({"activation": "relu"}, "activation"),
    "convnext_v2": ({"activation": "relu"}, "activation"),
    "rad_convnet": ({"activation": "relu"}, "activation"),
    # A simple bypass adds an Add layer and no weight: the same tensors compute another function.
    "squeezenet_v1": ({"use_bypass": "simple"}, "use_bypass"),
}
SAME_SHAPE_KEYS = tuple(SAME_SHAPE_DONORS)

#: Loader families with no architecture argument that moves no weight shape, and why.
NO_SAME_SHAPE_ARGUMENT: Dict[str, str] = {
    "squeezenet_v2": "its only architecture arguments are `variant` and `use_3d`, and each moves a "
                     "kernel shape (another squeeze width, a rank-5 kernel)",
}


def test_the_same_shape_donors_cover_every_loader_family() -> None:
    assert set(SAME_SHAPE_KEYS) | set(NO_SAME_SHAPE_ARGUMENT) == set(MIGRATED_LOADER_KEYS)
    assert not set(SAME_SHAPE_KEYS) & set(NO_SAME_SHAPE_ARGUMENT)


def test_squeezenodule_net_has_no_architecture_argument_that_keeps_every_weight_shape() -> None:
    """The exemption above is a measurement: both of its architecture arguments move a shape."""
    c = classifier("squeezenet_v2")
    base = _weight_layout(c.from_variant())
    another_variant = c.cls.from_variant("v1", num_classes=NUM_CLASSES, input_shape=c.input_shape)
    three_d = c.cls.from_variant("v2", num_classes=NUM_CLASSES, use_3d=True,
                                 input_shape=(36, 36, 36, 1))
    assert _weight_layout(another_variant) != base, "`variant` moves no weight shape"
    assert _weight_layout(three_d) != base, "`use_3d` moves no weight shape"


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


#: A head override that computes another function and moves no weight shape.
_HEAD_MAX = {"pooling_type": "max"}


def _saved(model: keras.Model, c: Classifier, path: Path) -> str:
    model.build((None, *c.input_shape))
    model.save(path)
    return str(path)


@pytest.mark.parametrize(
    "key", _rows("local_path_refuses_a_head_that_computes_another_function", HEAD_OVERRIDE_KEYS))
@ledgered("local_path_refuses_a_head_that_computes_another_function")
def test_a_head_that_computes_another_function_is_refused_unless_it_is_a_new_head(
        key, tmp_path) -> None:
    """The head layer is copied when its shapes fit, so then its configuration must be the same."""
    c = classifier(key)
    donor = c.from_variant(head_config_overrides=_HEAD_MAX)
    path = _saved(donor, c, tmp_path / "donor.keras")
    own = c.from_variant()
    own.build((None, *c.input_shape))
    assert _weight_layout(own) == _weight_layout(donor), (
        "control: the donor must differ in no weight shape, or the layer check would decide")
    before = [keras.ops.convert_to_numpy(w) for w in own.weights]
    with pytest.raises(ValueError) as caught:
        own.load_pretrained_weights(path)
    assert "head_config_overrides['pooling_type']: checkpoint 'max'" in str(caught.value), (
        str(caught.value))
    assert all(np.array_equal(b, keras.ops.convert_to_numpy(w))
               for b, w in zip(before, own.weights)), "a refused load wrote a weight"
    c.from_variant(head_config_overrides=_HEAD_MAX, pretrained=path)  # the same head loads
    c.from_variant(num_classes=NUM_CLASSES + 2, pretrained=path)  # a new head is not compared
    c.from_variant(include_top=False, pretrained=path)  # and neither is no head


@pytest.mark.parametrize(
    "key", _rows("local_path_reads_the_config_of_a_wrapped_checkpoint", SAME_SHAPE_KEYS))
@ledgered("local_path_reads_the_config_of_a_wrapped_checkpoint")
def test_a_functional_wrapper_is_compared_by_the_config_of_the_model_it_wraps(
        key, tmp_path) -> None:
    """The trainers save ``keras.Model(inp, model(inp))``; its archive holds the model's config."""
    c = classifier(key)
    change, config_key = SAME_SHAPE_DONORS[key]
    inner = c.from_variant(**change)
    inner.build((None, *c.input_shape))
    inp = keras.Input(c.input_shape)
    path = str(tmp_path / "wrapped.keras")
    keras.Model(inp, inner(inp)).save(path)
    own = c.from_variant()
    own.build((None, *c.input_shape))
    before = [keras.ops.convert_to_numpy(w) for w in own.weights]
    with pytest.raises(ValueError) as caught:
        own.load_pretrained_weights(path)
    assert f"{config_key}: checkpoint" in str(caught.value), str(caught.value)
    assert all(np.array_equal(b, keras.ops.convert_to_numpy(w))
               for b, w in zip(before, own.weights)), "a refused load wrote a weight"
    loaded = c.from_variant(**change, pretrained=path)  # the wrapped model's own architecture loads
    donor_weights = dict(weight_map(inner))
    assert any(np.array_equal(v, donor_weights[p]) for p, v in weight_map(loaded)
               if p in donor_weights), "the load restored no tensor"


#: The classifiers with an int-or-pair kernel argument: ``(argument, integer, same pair)``.
PAIR_SPELLED_KERNELS: Dict[str, Tuple[str, int, Tuple[int, int]]] = {
    "convnext_v1": ("kernel_size", 7, (7, 7)),
    "convnext_v2": ("kernel_size", 7, (7, 7)),
    "rad_convnet": ("rad_kernel_size", 3, (3, 3)),
}


@pytest.mark.parametrize(
    "key", _rows("local_path_reads_an_integer_and_its_pair_as_one_kernel",
                 tuple(PAIR_SPELLED_KERNELS)))
@ledgered("local_path_reads_an_integer_and_its_pair_as_one_kernel")
def test_an_integer_kernel_and_its_pair_are_one_architecture(key, tmp_path) -> None:
    c = classifier(key)
    argument, size, pair = PAIR_SPELLED_KERNELS[key]
    path = _saved(c.from_variant(**{argument: size}), c, tmp_path / "donor.keras")
    c.from_variant(**{argument: pair}, pretrained=path)  # the same kernel, spelled as a pair
    with pytest.raises(ValueError, match=argument):  # a pair of two sizes is another kernel
        c.from_variant(**{argument: (size, size + 2)}, pretrained=path)


@pytest.mark.parametrize(
    "key", _rows("head_config_overrides_are_json", HEAD_OVERRIDE_KEYS))
@ledgered("head_config_overrides_are_json")
def test_a_live_object_in_head_config_overrides_is_refused_by_name(key) -> None:
    c = classifier(key)
    with pytest.raises(ValueError, match=r"head_config_overrides\['kernel_regularizer'\]"):
        c.make(head_config_overrides={"kernel_regularizer": keras.regularizers.L2(1e-4)})
    serialized = {"kernel_regularizer": keras.regularizers.serialize(keras.regularizers.L2(1e-4))}
    model = c.make(head_config_overrides=serialized)
    assert json.loads(json.dumps(model.get_config()))["head_config_overrides"] == serialized


#: Shapes with no channel count: the first convolution would meet ``None % groups``.
_CHANNELLESS_SHAPES = [(None, None, None), (32, 48, None)]


@pytest.mark.parametrize("shape", _CHANNELLESS_SHAPES, ids=str)
@pytest.mark.parametrize("key", MIGRATED_LOADER_KEYS)
def test_a_pretrained_load_without_a_channel_count_is_refused_naming_input_shape(
        key, shape, tmp_path) -> None:
    """``from_variant(pretrained=path, input_shape=(None, None, None))`` cannot build the model.

    It failed as "unsupported operand type(s) for %" (wrapped in a load error for most
    families, bare for the Clifford pair), which names neither the argument nor its value.
    """
    c = classifier(key)
    path = _saved(c.make(), c, tmp_path / "donor.keras")
    with pytest.raises(ValueError, match="input_shape") as caught:
        c.from_variant(pretrained=path, input_shape=shape)
    assert str(shape) in str(caught.value), str(caught.value)


#: The classifiers that build in ``from_variant`` or the constructor whenever they have a shape
#: (the others store ``input_shape`` and build at the first call or at a pretrained load).
_BUILD_AT_CONSTRUCTION = ("cliffordnet", "hierarchical", "squeezenet_v1", "squeezenet_v2")


@pytest.mark.parametrize("key", _BUILD_AT_CONSTRUCTION)
def test_a_model_that_builds_at_construction_refuses_a_channelless_input_shape_by_name(key) -> None:
    """Without a checkpoint too: the build is the constructor's (SqueezeNet) or ``from_variant``'s."""
    with pytest.raises(ValueError, match="input_shape") as caught:
        classifier(key).from_variant(input_shape=(None, None, None))
    assert "(None, None, None)" in str(caught.value)


@pytest.mark.parametrize("key", [k for k in CLASSIFIER_KEYS if k not in _BUILD_AT_CONSTRUCTION])
def test_a_model_that_builds_later_stores_a_channelless_input_shape_unbuilt(key) -> None:
    """The control of the row above: these classifiers have nothing to refuse until they build."""
    model = classifier(key).from_variant(input_shape=(None, None, None))
    assert not model.built


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


@pytest.mark.parametrize("key", CLASSIFIER_KEYS)
def test_include_top_is_a_real_bool(key: str) -> None:
    """A string, an int or ``None`` is refused by value and type; ``numpy.bool_`` is a bool.

    ``bool("no")`` is ``True``: an accepted string built a headed model for a caller who
    wrote ``include_top="no"``. A ``numpy.bool_`` (what a comparison on an array yields)
    is coerced, so ``get_config`` stays JSON-safe and the stored attribute is a ``bool``.
    """
    c = classifier(key)
    for value in ("no", 0, 1, None):
        with pytest.raises(TypeError, match=r"include_top must be a bool or numpy\.bool_") as caught:
            c.make(include_top=value)
        assert repr(value) in str(caught.value)
    model = c.make(include_top=np.False_)
    assert model.include_top is False
    assert json.loads(json.dumps(model.get_config()))["include_top"] is False
    if c.head_attribute:
        assert model.classification_head is None


#: ``key -> the module-level factory`` of each classifier (house shape H9).
FACTORY_NAMES: Dict[str, str] = {
    "resnet": "create_resnet", "cliffordnet": "create_cliffordnet",
    "hierarchical": "create_hierarchical_cliffordnet",
    "mobilenet_v1": "create_mobilenetv1", "mobilenet_v2": "create_mobilenetv2",
    "mobilenet_v3": "create_mobilenetv3", "mobilenet_v4": "create_mobilenetv4",
    "convnext_v1": "create_convnext_v1", "convnext_v2": "create_convnext_v2",
    "rad_convnet": "create_rad_convnet",
    "squeezenet_v1": "create_squeezenet_v1", "squeezenet_v2": "create_squeezenodule_net_v2",
}


def test_every_classifier_module_defines_exactly_the_factories_this_file_calls() -> None:
    assert set(FACTORY_NAMES) == set(CLASSIFIER_KEYS)
    for module in {c.module for c in CLASSIFIERS}:
        defined = {name for name, obj in vars(importlib.import_module(module)).items()
                   if name.startswith("create_") and inspect.isfunction(obj)
                   and obj.__module__ == module}
        called = {FACTORY_NAMES[c.key] for c in CLASSIFIERS if c.module == module}
        assert defined == called, (module, defined, called)


@pytest.mark.parametrize("key", CLASSIFIER_KEYS)
def test_a_factory_gives_its_caller_the_include_top_the_caller_asked_for(key: str) -> None:
    """``create_*(include_top=False)`` is headless, and ``include_top=True`` is headed.

    The factory forwards the caller's extras to ``from_variant``; a factory that forces the
    argument (``**{**kwargs, "include_top": True}``) hands a headed model to every caller that
    asked for a backbone, and only a trainer that happened to call it caught that before.
    """
    c = classifier(key)
    factory = getattr(importlib.import_module(c.module), FACTORY_NAMES[key])
    for requested in (False, True):
        model = factory(c.variant, num_classes=NUM_CLASSES, input_shape=c.input_shape,
                        include_top=requested, **c.tiny)
        if c.head_attribute:
            assert (model.classification_head is not None) is requested, (
                f"{FACTORY_NAMES[key]}(include_top={requested}) gave include_top="
                f"{model.include_top}")
        model(_images(*c.input_shape))
        assert bool(_head_paths(c.head_markers, model)) is requested, (
            f"{FACTORY_NAMES[key]}(include_top={requested}) built "
            f"{'a head' if requested is False else 'no head'}")


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


def _build_rad_hidden_head(init):
    """The advertised ``block_dims`` override: three head kernels, none of them the default head's.

    One explicit seed handed to the default head's classifier is fine; handed to every Dense of
    a head with hidden blocks it replays one draw three times (measured r = 1.000000).
    """
    from dl_techniques.models.vision.rad_convnet.model import RADConvNet
    return _forward(RADConvNet.from_variant(
        "tiny", num_classes=10, input_shape=(32, 32, 3),
        head_config_overrides={"block_dims": [64, 64]}), _images(32, 32, 3))


def _build_squeezenet_v1(init):
    from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1
    return _forward(SqueezeNetV1.from_variant("1.0", num_classes=4, input_shape=(64, 64, 3),
                                              **_kw(init)), _images(64, 64, 3))


def _build_squeezenet_v2(init):
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
    return _forward(SqueezeNoduleNetV2.from_variant("v2", num_classes=4, input_shape=(64, 64, 1),
                                                    **_kw(init)), _images(64, 64, 1))


def _build_squeezenet_v2_3d(init):
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
    return _forward(SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=4, input_shape=(36, 36, 36, 1),
                                                    **_kw(init)), _images(36, 36, 36, 1))


def _build_vae(init):
    from dl_techniques.models.vision.vae.model import VAE
    return _forward(VAE.from_variant("small", input_shape=(32, 32, 3), **_kw(init)),
                    np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32"))


def _convunext(bias_free: bool = False, **config: Any):
    """A tiny ConvUNext; the instance arm hands ONE instance to both initializer arguments.

    ConvUNext has two initializer knobs, ``kernel_initializer`` (stem, channel adjusts, strided
    downsamplers, projections, supervision heads) and ``depthwise_initializer`` (every ConvNeXt
    block). Both get the instance, so a model that stores either one raw replays it.
    """
    def build(init):
        from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import (
            create_convunext_variant as create_bias_free)
        from dl_techniques.models.vision.convunext.model import create_convunext_variant
        arms = {} if init is None else {"kernel_initializer": init, "depthwise_initializer": init}
        create = create_bias_free if bias_free else create_convunext_variant
        return _forward(create("tiny", (32, 32, 3), **config, **arms),
                        np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32"))
    return build


def _build_rbf(init):
    from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet
    return _forward(RBFProtoNet(num_classes=7, stem_filters=8, filters_per_stage=[8, 16],
                                feature_dim=16), _images(32, 32, 3))


def _build_clifford_rbf(init):
    from dl_techniques.models.vision.rbf_protonet.model import CliffordRBFProtoNet
    return _forward(CliffordRBFProtoNet(num_classes=7, channels=16, depth=2, feature_dim=16),
                    _images(32, 32, 3))


def _uca(attention_type: str):
    def build(init):
        from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder
        return _forward(UcaEncoder(stage_dims=[16, 32], stage_depths=[1, 2], num_heads=[2, 4],
                                   attention_type=attention_type), _images(16, 16, 3))
    return build


def _build_mambalct(init):
    """The tracker the package's own tests build: encoder, context SSM, fusion attention, head."""
    from dl_techniques.models.vision.mambalct import MambaLCT
    rng = np.random.default_rng(0)
    model = MambaLCT(template_size=32, search_size=64, context_len=1, stage_dims=[64, 128, 512],
                     stage_depths=[1, 1, 1], num_heads=[2, 4, 8], fusion_heads=8,
                     head_hidden_dim=32, name="mambalct_tiny")
    model([rng.uniform(0, 1, size=(1, 32, 32, 3)).astype("float32"),
           rng.uniform(0, 1, size=(1, 2, 64, 64, 3)).astype("float32")], training=False)
    return model


def _build_capsnet(init):
    from dl_techniques.models.vision.capsnet.model import CapsNet
    return _forward(CapsNet(num_classes=10, input_shape=(28, 28, 1), **_kw(init)),
                    np.random.default_rng(0).uniform(size=(2, 28, 28, 1)).astype("float32"))


def _capsnet_v2(backbone: str, shape: Tuple[int, int, int], reconstruction: bool = False):
    def build(init):
        from dl_techniques.models.vision.capsnet.model_v2 import CapsNetV2
        model = CapsNetV2(num_classes=10, input_shape=shape, reconstruction=reconstruction,
                          stem_type=backbone, **_kw(init))
        model.build((None, *shape))  # the decoder is built here, not by a forward pass
        return _forward(model, np.random.default_rng(0).uniform(size=(2, *shape)).astype("float32"))
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
        model = DepthAnything(encoder_kind=kind, encoder_type="vit_s", image_shape=(size, size, 3),
                              **_kw(init))
        return _forward(model, np.random.default_rng(0).uniform(
            size=(1, size, size, 3)).astype("float32"))
    return build


def _dinov2_vit(attention_type: str = "multi_head"):
    """A tiny DINOv2 backbone (three blocks, so a draw replayed across blocks is seen too).

    The backbone has no initializer knob: the blocks' attention layers draw their own, so the subject
    has a default arm only.
    """
    def build(init):
        from dl_techniques.models.vision.dino.dino_v2 import DINOv2VisionTransformer
        model = DINOv2VisionTransformer(image_size=28, patch_size=14, embed_dim=64, depth=3,
                                        num_heads=4, init_values=1e-5, attention_type=attention_type)
        return _forward(model, [_images(28, 28, 3), np.zeros((2, 4), dtype="bool")])
    return build


def _dino_v1_vit(init):
    """A two-block DINOv1 trunk (the attention output projection is built by the model, D-056)."""
    from dl_techniques.models.vision.dino.dino_v1 import DINOv1
    model = DINOv1(image_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4)
    return _forward(model, _images(32, 32, 3))


def _dino_v3_vit(init):
    """A two-block DINOv3: its ``kernel_initializer`` reaches ``qkv``, the FFN and the output projection."""
    from dl_techniques.models.vision.dino.dino_v3 import DINOv3
    model = DINOv3(image_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4, **_kw(init))
    return _forward(model, _images(32, 32, 3))


def _build_dinov2_classifier(init):
    """DINOv2 itself: the backbone plus its classifier, whose initializer the wrapper builds."""
    from dl_techniques.models.vision.dino.dino_v2 import DINOv2
    model = DINOv2(image_size=28, patch_size=14, num_classes=5, embed_dim=64, depth=3, num_heads=4,
                   init_values=1e-5)
    return _forward(model, [_images(28, 28, 3), np.zeros((2, 4), dtype="bool")])


def _build_clifford_clip(init):
    """Both towers, patch 1 (the two-conv stem); ``init`` is the model's own ``kernel_initializer``."""
    from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
    rng = np.random.default_rng(0)
    knob = {} if init is None else {"kernel_initializer": init}
    model = CliffordCLIP.from_variant("nano", vocab_size=1000, image_size=16, context_length=16,
                                      vision_patch_size=1, **knob)
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
    "rad_convnet_hidden_head": (_build_rad_hidden_head, False),
    "squeezenet_v1": (_build_squeezenet_v1, True),
    "squeezenet_v2": (_build_squeezenet_v2, True),
    "squeezenet_v2_3d": (_build_squeezenet_v2_3d, True),
    "vae": (_build_vae, True),
    "convunext": (_convunext(), True),
    "convunext_bias_free": (_convunext(bias_free=True), True),
    "convunext_gabor_stem": (_convunext(use_gabor_stem=True, gabor_filters=32, use_bias=False,
                                        final_activation="linear", block_normalization="batchnorm"), True),
    "convunext_deep_supervision": (_convunext(enable_deep_supervision=True), True),
    "convunext_strided_downsample": (_convunext(downsample_pool_type="strided_conv"), True),
    "rbf_protonet": (_build_rbf, False),
    "clifford_rbf_protonet": (_build_clifford_rbf, False),
    "uca_encoder": (_uca("multi_head"), False),
    "uca_encoder_group_query": (_uca("group_query"), False),
    "uca_encoder_multi_head_latent": (_uca("multi_head_latent"), False),
    # The four types whose layers began to clone an initializer per projection (D-056); the encoder has
    # no initializer knob, the layer-level replay file measures them with an instance.
    "uca_encoder_differential": (_uca("differential"), False),
    "uca_encoder_anchor": (_uca("anchor"), False),
    "uca_encoder_window": (_uca("window"), False),
    "uca_encoder_window_band": (_uca("window_band"), False),
    "mambalct": (_build_mambalct, False),
    "capsnet": (_build_capsnet, True),
    "capsnet_v2_legacy": (_capsnet_v2("legacy", (28, 28, 1)), True),
    "capsnet_v2_legacy_reconstruction": (_capsnet_v2("legacy", (28, 28, 1), reconstruction=True), True),
    "capsnet_v2_resnet18": (_capsnet_v2("resnet18", (32, 32, 3)), True),
    "siamfc": (_build_siamfc, True),
    "dasiamrpn": (_build_dasiamrpn, True),
    "deepsort": (_build_deepsort, False),
    "depth_anything_real": (_depth_anything("real", 56), False),
    # The real encoder takes no initializer (DINOv2VisionTransformer has no such knob), so the knob
    # reaches only the head, which the placeholder subject measures with an instance as well.
    "depth_anything_placeholder": (_depth_anything("placeholder", 64), True),
    # DINOv2 itself: the blocks build their own attention initializers, so there is no knob to arm.
    "dinov2_vit": (_dinov2_vit(), False),
    "dinov2_classifier": (_build_dinov2_classifier, False),
    "dinov2_vit_multi_head_cross": (_dinov2_vit("multi_head_cross"), False),
    "dinov2_vit_gated": (_dinov2_vit("gated"), False),
    "dinov2_vit_linear": (_dinov2_vit("linear"), False),
    "dinov2_vit_ring": (_dinov2_vit("ring"), False),
    # The five types that replayed a draw inside a block until their layers cloned per projection (D-056).
    "dinov2_vit_anchor": (_dinov2_vit("anchor"), False),
    "dinov2_vit_perceiver": (_dinov2_vit("perceiver"), False),
    "dinov2_vit_performer": (_dinov2_vit("performer"), False),
    "dinov2_vit_rpc": (_dinov2_vit("rpc"), False),
    "dinov2_vit_wave_field": (_dinov2_vit("wave_field"), False),
    # Registered after that table was measured; its layer clones one initializer per projection itself.
    "dinov2_vit_mean_shift": (_dinov2_vit("mean_shift"), False),
    # DINOv1 has no initializer knob; DINOv3 takes `kernel_initializer`, so it has both arms.
    "dino_v1": (_dino_v1_vit, False),
    "dino_v3": (_dino_v3_vit, True),
    "clifford_clip": (_build_clifford_clip, True),
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
    _named_other("rbf_protonet", "rbf_protonet", ("stem", "stage1")),
    _named_other("clifford_rbf_protonet", "clifford_rbf_protonet", ("stem", "block_0")),
    _named_other("uca_encoder", "uca_encoder", ("stem", "stage1_cpe", "stage1_block1")),
    _named_other("capsnet", "capsnet", ("conv_block_0",)),
    _named_other("capsnet_v2_legacy", "capsnet_v2_legacy", ("conv_block_0",)),
    _named_other("deepsort", "deepsort", ("stem",)),
    _named_other("siamfc", "siamfc", ("stem",)),
    _named_other("dasiamrpn", "dasiamrpn", ("stem",)),
    # The VAE's conv-norm-activation sites are ConvStems that keep the name the convolution had, so each
    # site's name and its inner convolution `<site>_conv1` must both be there.
    _named_other("vae", "vae", ("encoder_stem", "encoder_stem_conv1",
                                "encoder_downsample_0", "encoder_downsample_0_conv1",
                                "decoder_conv_0", "decoder_conv_0_conv1",
                                "encoder_0_0_conv_1", "encoder_0_0_conv_1_conv1")),
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
_CONVNEXT_CHILDREN = (r"stem|stage[1-9]\d*_block[1-9]\d*(?:_drop_path)?"
                      r"|downsample_\d+_(?:norm|conv)|classification_head")
FAMILY_CHILD_NAMES: Dict[str, str] = {
    "resnet": r"stem|stage[1-9]\d*_block[1-9]\d*|classification_head|supervision_head_\d+",
    "cliffordnet": r"stem|block_\d+(?:_drop_path)?|classification_head",
    "hierarchical": (r"stem|stage[1-9]\d*_block[1-9]\d*(?:_drop_path)?"
                     r"|downsample_\d+_(?:depthwise|norm|pointwise)|classification_head"),
    "mobilenet_v1": r"stem|block_\d+|classification_head",
    "mobilenet_v2": r"stem|block_\d+|conv_last|conv_last_bn|conv_last_relu6|classification_head",
    "mobilenet_v3": r"stem|block_\d+|last_conv|last_bn|last_hard_swish|classification_head",
    "mobilenet_v4": r"stem|stage[1-9]\d*_(?:block[1-9]\d*|mqa)|classification_head",
    "convnext_v1": _CONVNEXT_CHILDREN,
    "convnext_v2": _CONVNEXT_CHILDREN,
    "rad_convnet": (r"stem|mlp_activation|stage[1-9]\d*_block[1-9]\d*_"
                    r"(?:rad_conv|norm|expand|reduce|dropout)"
                    r"|downsample_\d+_(?:norm|conv)|classification_head"),
    # SqueezeNet is a functional model: its head is the paper's ``conv10`` and a pool, and its
    # Fire modules, pools and bypass layers are named by the paper's fire numbers (2 to 9).
    "squeezenet_v1": (r"stem|fire[2-9]|maxpool[2-9]|add_fire[2-9]|bypass_conv_fire[2-9]"
                      r"|dropout|conv10|avgpool"),
    "squeezenet_v2": r"stem|simpfire[2-9]|maxpool[2-9]|dropout|conv10|avgpool",
}


def direct_children(model: keras.layers.Layer) -> List[str]:
    """The names of the layers ``model`` owns directly, whether or not they hold a weight.

    Interface contract: the direct sublayers of a model or of a plain layer (for a model
    this is ``model.layers``; Keras tracks the ones kept in a list or a dict attribute
    too). Returns the names sorted, ONE PER LAYER: a name two children share appears twice,
    so a row that compares the list with the names an architecture gives (:func:`numbering_problems`)
    fails on a duplicate (it used to return the distinct names, and a student and a frozen teacher
    that both read ``encoder_vit_s`` looked like one encoder). A weightless child (a drop path, an
    activation, a pooling) is read, which the weight paths of a built model cannot show. The
    ``InputLayer`` a functional model holds is not read: it is not a layer the model chose, and
    its name is a process-wide counter (``input_layer_7``).
    """
    return sorted(layer.name
                  for layer in model._flatten_layers(include_self=False, recursive=False)
                  if not isinstance(layer, keras.layers.InputLayer))


def sibling_name_duplicates(model: keras.layers.Layer) -> List[str]:
    """Every pair of layers that share a name under one parent, anywhere in ``model``.

    Interface contract: walks the layer tree below ``model`` (each layer once, however many
    attributes hold it) and, for every layer, counts its direct children by name
    (:func:`direct_children` semantics, ``InputLayer`` excluded). Returns one message per
    repeated name, ``"<path>: `<name>` x<count>"``, sorted; empty when every layer's children
    carry distinct names. Names that repeat under DIFFERENT parents (``attention`` in each
    block) are the layers' own business and are not read. Raises nothing.
    """
    problems: List[str] = []
    seen = set()
    stack = [(model.name, model)]
    while stack:
        path, layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        children = [child for child in layer._flatten_layers(include_self=False, recursive=False)
                    if not isinstance(child, keras.layers.InputLayer)]
        counts = collections.Counter(child.name for child in children)
        problems.extend(f"{path}: `{name}` x{count}" for name, count in counts.items() if count > 1)
        stack.extend((f"{path}/{child.name}", child) for child in children)
    return sorted(problems)


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


_VOLUME = (36, 36, 36, 1)  # the smallest cube SqueezeNodule-Net accepts: 35 voxels per axis


def _volume_v2(**knobs) -> keras.Model:
    """The 3D SqueezeNodule-Net, built and run once."""
    model = classifier("squeezenet_v2").make(use_3d=True, input_shape=_VOLUME, **knobs)
    model(_images(*_VOLUME))
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
    "mobilenet_v1": (lambda: _built(classifier("mobilenet_v1")),
                     lambda: _built(classifier("mobilenet_v1"), include_top=False)),
    "mobilenet_v2": (lambda: _built(classifier("mobilenet_v2")),
                     lambda: _built(classifier("mobilenet_v2"), include_top=False)),
    "mobilenet_v3": (lambda: _built(classifier("mobilenet_v3")),),
    "mobilenet_v4": (lambda: _built(classifier("mobilenet_v4")), _hybrid_v4),
    # A drop path exists only for a block with a non-zero rate, so one build has them.
    "convnext_v1": (lambda: _built(classifier("convnext_v1"), drop_path_rate=0.2),
                    lambda: _built(classifier("convnext_v1"), drop_path_rate=0.2,
                                   stochastic_mode="gradient", include_top=False)),
    "convnext_v2": (lambda: _built(classifier("convnext_v2"), drop_path_rate=0.2),
                    lambda: _built(classifier("convnext_v2"), drop_path_rate=0.2,
                                   stochastic_mode="gradient", include_top=False)),
    # A dropout layer per block exists only with a non-zero rate.
    "rad_convnet": (lambda: _built(classifier("rad_convnet"), dropout_rate=0.1),
                    lambda: _built(classifier("rad_convnet"), include_top=False)),
    # The plain, the simple-bypass and the complex-bypass graph, and the 3D model.
    "squeezenet_v1": (lambda: _built(classifier("squeezenet_v1")),
                      lambda: _built(classifier("squeezenet_v1"), use_bypass="simple"),
                      lambda: _built(classifier("squeezenet_v1"), use_bypass="complex",
                                     include_top=False)),
    "squeezenet_v2": (lambda: _built(classifier("squeezenet_v2")),
                      _volume_v2),
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


class _Twinned(PublicAutoName, keras.Model):
    """Planted: two children of the model named ``twin``, and one nested layer with two named ``inner``.

    ``twinned=False`` is the clean twin: distinct child names, and ``inner`` once under each of two parents.
    """

    def __init__(self, twinned: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.first = keras.layers.Dense(3, name="twin" if twinned else "first")
        self.second = keras.layers.Dense(3, name="twin" if twinned else "second")
        self.nested = _Nest(name="nest", twins=twinned)
        self.fine = _Nest(name="fine", twins=False)

    def call(self, x):
        return self.fine(self.nested(self.second(self.first(x))))


class _Nest(keras.layers.Layer):
    def __init__(self, twins: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.twins = twins
        self.a = keras.layers.Dense(3, name="inner")
        if twins:
            self.b = keras.layers.Dense(3, name="inner")

    def call(self, x):
        return self.b(self.a(x)) if self.twins else self.a(x)


def test_direct_children_lists_two_layers_of_one_name_twice() -> None:
    model = _Twinned()
    model(np.ones((2, 3), dtype="float32"))
    assert direct_children(model) == ["fine", "nest", "twin", "twin"]


def test_the_exact_name_set_rule_fails_on_a_name_two_layers_share() -> None:
    assert numbering_problems(["a", "b"], {"a", "b"}) == []
    assert numbering_problems(["a", "a", "b"], {"a", "b"}) == ["`a` is the name of 2 layers"]
    assert numbering_problems(["a", "a"], {"a", "b"}) == ["missing `b`", "`a` is the name of 2 layers"]


def test_the_sibling_name_rule_sees_a_repeat_at_every_depth_and_only_under_one_parent() -> None:
    model = _Twinned(name="planted")
    model(np.ones((2, 3), dtype="float32"))
    assert sibling_name_duplicates(model) == ["planted/nest: `inner` x2", "planted: `twin` x2"]
    clean = _Twinned(twinned=False, name="clean")
    clean(np.ones((2, 3), dtype="float32"))
    assert sibling_name_duplicates(clean) == [], "`inner` under two parents is not a repeat"


@functools.lru_cache(maxsize=None)
def roster_facts(key: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], int]:
    """What one build of a replay-roster subject shows, read once per process.

    Interface contract: builds ``REPLAY_SUBJECTS[key]`` at seed 1234 with its default arm and returns
    ``(sibling_name_duplicates, the directories of the packages under dl_techniques/models that define a class
    in the model's layer tree, the number of layers below the model)``. The tree is read, not only the model's
    own class, because a functional model (ConvUNext) is a plain Keras ``Functional`` whose family shows in its
    layers. Cached, so the rows that read the roster share one build per subject; the model itself is not kept.
    Raises whatever the builder raises.
    """
    build, _ = REPLAY_SUBJECTS[key]
    keras.utils.set_random_seed(1234)
    model = build(None)
    layers = list(model._flatten_layers(include_self=True))
    models_root = SRC / "models"
    packages = set()
    for layer in layers:
        module = inspect.getmodule(type(layer))
        origin = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
        if origin is not None and models_root in origin.parents:
            packages.add(str(origin.parent))
    return tuple(sibling_name_duplicates(model)), tuple(sorted(packages)), len(layers) - 1


@pytest.mark.parametrize("key", list(REPLAY_SUBJECTS))
def test_no_two_sibling_layers_of_a_finished_model_share_a_name(key: str) -> None:
    """Every finished family's built model, its frozen teacher and nested stems included.

    A transfer that matches layers by name keeps one layer per name, and a by-path reader keeps one
    weight per path: the student and the frozen teacher of ``DepthAnything`` shared a name and filled
    the wrong one (R4 finding 2). The subjects are the replay roster, so a model added to it is read
    here without anyone remembering to list it.
    """
    duplicates, _, layers = roster_facts(key)
    assert layers > 0, "control: the model owns no layer"
    assert duplicates == ()


@pytest.mark.parametrize(
    ("key", "index"),
    [(key, i) for key, builds in FAMILY_BUILDS.items() for i in range(len(builds))],
    ids=lambda v: str(v))
def test_no_two_sibling_layers_of_a_family_variant_share_a_name(key: str, index: int) -> None:
    """The variants of the classifier families: headless, deep supervision, bypass, drop path, the 3D model."""
    keras.utils.set_random_seed(1234)
    model = FAMILY_BUILDS[key][index]()
    assert len(list(model._flatten_layers(include_self=False))) > 0, "control: the model owns no layer"
    assert sibling_name_duplicates(model) == []


# ---------------------------------------------------------------------------
# The numbered names are exactly the names the model's own depths give
# ---------------------------------------------------------------------------

#: Three stages of three different depths, so a stage index swapped with a block index
#: (``stage{B}_block{S}``) names a block that does not exist. The patterns above accept
#: ANY index (``stage[1-9]\d*_block[1-9]\d*``), so a block counted from 2 or a
#: downsampler counted from 1 passed them; the sets below do not.
_DEPTHS = [2, 1, 3]
_DIMS = [8, 16, 16]


def _staged_blocks(depths: Sequence[int]) -> set:
    """``stage{S}_block{B}`` with S and B counted from 1 (N-3), one per block of ``depths``."""
    return {f"stage{s}_block{b}" for s, depth in enumerate(depths, 1) for b in range(1, depth + 1)}


def _flat_blocks(count: int) -> set:
    """``block_{i}`` counted from 0 (N-2)."""
    return {f"block_{i}" for i in range(count)}


def _downsamplers(stages: int, roles: Sequence[str]) -> set:
    """``downsample_{i}_<role>`` for i from 0: ``stages - 1`` downsamplers sit between the stages (N-4)."""
    return {f"downsample_{i}_{role}" for i in range(stages - 1) for role in roles}


def _with_drop_paths(blocks: Sequence[str], rates: Sequence[float]) -> set:
    """Each block's ``<block>_drop_path``, for the blocks whose rate is above 0.

    ConvNeXt creates the layer only for a non-zero rate, and the ramp starts at 0.
    """
    return {f"{name}_drop_path" for name, rate in zip(blocks, rates) if rate > 0}


def _ordered_staged_blocks(depths: Sequence[int]) -> List[str]:
    return [f"stage{s}_block{b}" for s, depth in enumerate(depths, 1) for b in range(1, depth + 1)]


def _head(headed: bool) -> set:
    return {"classification_head"} if headed else set()


_CONVNEXT_RATE = 0.2
_V4_DEPTHS = [2, 1, 3, 1, 2, 1, 2]
_V4_ATTENTION_STAGES = [5, 6]  # 0-based indices into the stages, named stage6_mqa and stage7_mqa
_V4_INPUT = (128, 128, 3)  # attention needs a feature map large enough to pool at stage 6


def _convnext_owned(headed: bool) -> set:
    blocks = _ordered_staged_blocks(_DEPTHS)
    ramp = np.linspace(0.0, _CONVNEXT_RATE, sum(_DEPTHS))
    return ({"stem"} | set(blocks) | _with_drop_paths(blocks, ramp)
            | _downsamplers(len(_DEPTHS), ("norm", "conv")) | _head(headed))


def _hierarchical_owned(headed: bool) -> set:
    # A pointwise convolution exists only where the width changes between two stages.
    pointwise = {f"downsample_{i}_pointwise" for i in range(len(_DIMS) - 1)
                 if _DIMS[i] != _DIMS[i + 1]}
    blocks = _staged_blocks(_DEPTHS)
    return ({"stem"} | blocks | {f"{b}_drop_path" for b in blocks}
            | _downsamplers(len(_DEPTHS), ("depthwise", "norm")) | pointwise | _head(headed))


def _rad_owned(headed: bool) -> set:
    roles = ("rad_conv", "norm", "expand", "reduce", "dropout")
    return ({"stem", "mlp_activation"}
            | {f"{b}_{role}" for b in _staged_blocks(_DEPTHS) for role in roles}
            | _downsamplers(len(_DEPTHS), ("norm", "conv")) | _head(headed))


#: ``key -> (constructor knobs, headed -> the exact set of layers the model owns directly)``.
#: MobileNet V1 to V3 have a fixed block list, so their counts are the architectures' own:
#: 13 depthwise-separable blocks (Howard 2017, Table 1), 17 inverted residuals (Sandler 2018,
#: Table 2) and 11 bottlenecks for V3 small (Howard 2019, Table 2).
NUMBERED_FAMILIES: Dict[str, Tuple[Dict[str, Any], Callable[[bool], set]]] = {
    "resnet": (dict(blocks_per_stage=_DEPTHS, filters_per_stage=_DIMS),
               lambda headed: {"stem"} | _staged_blocks(_DEPTHS) | _head(headed)),
    "cliffordnet": (dict(depth=3),
                    lambda headed: {"stem"} | _flat_blocks(3)
                    | {f"block_{i}_drop_path" for i in range(3)} | _head(headed)),
    "hierarchical": (dict(stage_depths=_DEPTHS, stage_dims=_DIMS), _hierarchical_owned),
    "mobilenet_v1": ({}, lambda headed: {"stem"} | _flat_blocks(13) | _head(headed)),
    "mobilenet_v2": ({}, lambda headed: {"stem", "conv_last", "conv_last_bn", "conv_last_relu6"}
                     | _flat_blocks(17) | _head(headed)),
    "mobilenet_v3": ({}, lambda headed: {"stem", "last_conv", "last_bn", "last_hard_swish"}
                     | _flat_blocks(11) | _head(headed)),
    "mobilenet_v4": (dict(depths=_V4_DEPTHS, use_attention=True,
                          attention_stages=_V4_ATTENTION_STAGES, input_shape=_V4_INPUT),
                     lambda headed: {"stem"} | _staged_blocks(_V4_DEPTHS)
                     | {f"stage{s + 1}_mqa" for s in _V4_ATTENTION_STAGES} | _head(headed)),
    "convnext_v1": (dict(depths=_DEPTHS, dims=_DIMS, drop_path_rate=_CONVNEXT_RATE, strides=2),
                    _convnext_owned),
    "convnext_v2": (dict(depths=_DEPTHS, dims=_DIMS, drop_path_rate=_CONVNEXT_RATE, strides=2),
                    _convnext_owned),
    "rad_convnet": (dict(depths=_DEPTHS, dims=_DIMS, strides=2, dropout_rate=0.1), _rad_owned),
    # The paper's numbering: fire2 to fire9, the stem's pool inside the stem, the other pools
    # after the fire module they follow. Variant 1.1 pools after fire3 and fire5; the complex
    # bypass adds every fire module and a 1x1 projection where the width changes (64 to 128 at
    # fire2, 128 to 256 at fire4, 256 to 384 at fire6, 384 to 512 at fire8).
    "squeezenet_v1": (dict(use_bypass="complex", input_shape=(64, 64, 3)),
                      lambda headed: {"stem", "maxpool3", "maxpool5", "dropout"}
                      | {f"fire{n}" for n in range(2, 10)}
                      | {f"add_fire{n}" for n in range(2, 10)}
                      | {f"bypass_conv_fire{n}" for n in (2, 4, 6, 8)}
                      | ({"conv10", "avgpool"} if headed else set())),
    # Variant v2 pools after fire4 and fire8; there is no bypass.
    "squeezenet_v2": (dict(input_shape=(64, 64, 1)),
                      lambda headed: {"stem", "maxpool4", "maxpool8", "dropout"}
                      | {f"simpfire{n}" for n in range(2, 10)}
                      | ({"conv10", "avgpool"} if headed else set())),
}


def numbering_problems(owned: Sequence[str], expected: set) -> List[str]:
    """How the layers a model owns differ from the exact set its depths give.

    Interface contract: ``owned`` is :func:`direct_children` of a built model; ``expected``
    the set derived from the arguments the test built it with. Returns one message per name
    that is missing, unexpected or owned by more than one layer, empty when the layers are
    exactly the set, each once. Raises nothing.
    """
    counts = collections.Counter(owned)
    return ([f"missing `{name}`" for name in sorted(expected - set(counts))]
            + [f"unexpected `{name}`" for name in sorted(set(counts) - expected)]
            + [f"`{name}` is the name of {count} layers" for name, count in sorted(counts.items())
               if count > 1])


def test_every_finished_family_has_an_exact_numbered_name_set() -> None:
    assert set(NUMBERED_FAMILIES) == set(FAMILY_CHILD_NAMES)


@pytest.mark.parametrize("headed", [True, False], ids=["headed", "headless"])
@pytest.mark.parametrize("key", sorted(NUMBERED_FAMILIES))
def test_a_family_numbers_its_layers_exactly_as_its_depths_say(key: str, headed: bool) -> None:
    """Stages and blocks from 1, flat blocks and downsamplers from 0, nothing else owned."""
    knobs, owned = NUMBERED_FAMILIES[key]
    c = classifier(key)
    model = c.make(include_top=headed, **knobs)
    model(_images(*knobs.get("input_shape", c.input_shape)))
    assert numbering_problems(direct_children(model), owned(headed)) == []


@pytest.mark.parametrize("headed", [True, False], ids=["headed", "headless"])
def test_the_3d_squeezenodule_net_owns_the_layers_of_the_2d_one(headed: bool) -> None:
    """The volume model is the same graph of names: a Conv3D stem, 3D pools and a 3D head."""
    _, owned = NUMBERED_FAMILIES["squeezenet_v2"]
    model = _volume_v2(include_top=headed)
    assert numbering_problems(direct_children(model), owned(headed)) == []


def test_the_numbering_rule_sees_a_block_counted_from_two_and_a_downsampler_counted_from_one() -> None:
    expected = _staged_blocks([2, 1]) | _downsamplers(2, ("norm", "conv"))
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "a block counted from 2": ["stage2_block2" if n == "stage2_block1" else n for n in clean],
        "a downsampler counted from 1": [n.replace("downsample_0", "downsample_1") for n in clean],
        "a stage and a block swapped": ["stage1_block1", "stage2_block1", "stage2_block2",
                                        *[n for n in clean if n.startswith("downsample")]],
        "a missing block": [n for n in clean if n != "stage1_block2"],
        "an extra block": [*clean, "stage3_block1"],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"
    # The pattern table above is why this rule exists: it reads every one of these as a
    # well-formed ConvNeXt name.
    assert child_name_violations("convnext_v1", plants["a block counted from 2"]) == []
    assert child_name_violations("convnext_v1", plants["a downsampler counted from 1"]) == []


# ---------------------------------------------------------------------------
# The two RBF models and UcaEncoder number their layers exactly as their arguments say
# ---------------------------------------------------------------------------

_RBF_STAGES = [8, 16, 16]  # three stages, so a stage counted from 0 or swapped is seen
_RBF_FEATURE_DIM = _RBF_STAGES[-1]
_RBF_PROJECTED_FEATURE_DIM = 24  # differs from the last stage: a projection Dense exists
_CRP_DEPTH = 3
_CRP_CHANNELS = 16
_UCA_DEPTHS = [2, 1, 3]
_UCA_DIMS = [16, 32, 32]
_UCA_HEADS = [2, 4, 4]


def _rbf_owned(feature_dim: int) -> set:
    """RBFProtoNet: the stem, one ``stage{k}`` per entry of ``_RBF_STAGES`` counted from 1, the
    pool, a projection only where the last stage is not ``feature_dim`` wide, and the head."""
    projection = {"feature_proj"} if _RBF_STAGES[-1] != feature_dim else set()
    return ({"stem", "global_avg_pool", "rbf_head"}
            | {f"stage{k}" for k in range(1, len(_RBF_STAGES) + 1)} | projection)


def _crp_owned(feature_dim: int) -> set:
    """CliffordRBFProtoNet: the stem, ``block_{i}`` from 0 with each block's drop path, the pool,
    a projection only where ``channels`` is not ``feature_dim``, and the head."""
    projection = {"feature_proj"} if _CRP_CHANNELS != feature_dim else set()
    return ({"stem", "global_avg_pool", "rbf_head"} | _flat_blocks(_CRP_DEPTH)
            | {f"block_{i}_drop_path" for i in range(_CRP_DEPTH)} | projection)


def _uca_owned() -> set:
    """UcaEncoder: the stem, ``stage{S}_block{B}`` and ``stage{S}_cpe`` from 1, the
    ``downsample_{i}_{conv,norm}`` between stages from 0, and ``final_norm``."""
    return ({"stem", "final_norm"} | _staged_blocks(_UCA_DEPTHS)
            | {f"stage{s}_cpe" for s in range(1, len(_UCA_DEPTHS) + 1)}
            | _downsamplers(len(_UCA_DEPTHS), ("conv", "norm")))


def _rbf_build(feature_dim: int) -> Callable[[], keras.layers.Layer]:
    def build() -> keras.layers.Layer:
        from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet
        return RBFProtoNet(num_classes=7, stem_filters=8, filters_per_stage=_RBF_STAGES,
                           feature_dim=feature_dim)
    return build


def _crp_build(feature_dim: int) -> Callable[[], keras.layers.Layer]:
    def build() -> keras.layers.Layer:
        from dl_techniques.models.vision.rbf_protonet.model import CliffordRBFProtoNet
        return CliffordRBFProtoNet(num_classes=7, channels=_CRP_CHANNELS, depth=_CRP_DEPTH,
                                   feature_dim=feature_dim)
    return build


def _uca_build() -> keras.layers.Layer:
    from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder
    return UcaEncoder(stage_dims=_UCA_DIMS, stage_depths=_UCA_DEPTHS, num_heads=_UCA_HEADS)


#: ``case id -> (build, the exact set of layers the build owns directly)``; the expected sets
#: read the constants above, never the model.
OTHER_NUMBERED: Dict[str, Tuple[Callable[[], keras.layers.Layer], Callable[[], set]]] = {
    "rbf_protonet": (_rbf_build(_RBF_FEATURE_DIM), lambda: _rbf_owned(_RBF_FEATURE_DIM)),
    "rbf_protonet_projected": (_rbf_build(_RBF_PROJECTED_FEATURE_DIM),
                               lambda: _rbf_owned(_RBF_PROJECTED_FEATURE_DIM)),
    "clifford_rbf_protonet": (_crp_build(_CRP_CHANNELS), lambda: _crp_owned(_CRP_CHANNELS)),
    "clifford_rbf_protonet_projected": (_crp_build(24), lambda: _crp_owned(24)),
    "uca_encoder": (_uca_build, _uca_owned),
}


@pytest.mark.parametrize("case", sorted(OTHER_NUMBERED))
def test_the_rbf_pair_and_uca_number_their_layers_exactly_as_their_arguments_say(case: str) -> None:
    build, owned = OTHER_NUMBERED[case]
    layer = build()
    layer(_images(32, 32, 3))
    assert numbering_problems(direct_children(layer), owned()) == []


def test_the_numbering_rule_sees_the_spellings_the_rbf_pair_and_uca_had() -> None:
    cases = {
        "clifford_rbf_protonet": [
            ("a Clifford-spelled block", {"block_0": "clifford_block_0"}),
            ("a drop path with no block", {"block_0_drop_path": "drop_path_0"}),
            ("a block counted from 1", {"block_2": "block_3"}),
        ],
        "uca_encoder": [
            ("a block counted from 0 with underscores", {"stage1_block1": "stage_0_block_0"}),
            ("a block counted from 0", {"stage1_block1": "stage0_block0"}),
            ("a merge instead of a downsample", {"downsample_0_conv": "merge_1"}),
            ("a merge norm", {"downsample_0_norm": "merge_norm_1"}),
            ("a downsampler counted from 1", {"downsample_0_conv": "downsample_1_conv"}),
            ("a cpe counted from 0", {"stage1_cpe": "cpe_0"}),
        ],
    }
    for key, plants in cases.items():
        expected = OTHER_NUMBERED[key][1]()
        clean = sorted(expected)
        assert numbering_problems(clean, expected) == []
        for why, rename in plants:
            owned = [rename.get(name, name) for name in clean]
            assert numbering_problems(owned, expected), f"{key}: {why} passed the numbering rule"
    # an extra and a missing block, in the Uca set
    expected = _uca_owned()
    assert numbering_problems([*sorted(expected), "stage4_block1"], expected)
    assert numbering_problems(sorted(expected - {"stage3_block3"}), expected)


# ---------------------------------------------------------------------------
# CapsNet and CapsNetV2 number their layers exactly as their arguments say
# ---------------------------------------------------------------------------

_CAPS_TRUNK = [8, 8, 8]  # three blocks, so a block counted from 1 or swapped is seen
_CAPS_IMAGE = (28, 28, 1)  # 28 -> 20 -> 16 -> 12 through the 9x9, 5x5, 5x5 valid blocks


def _caps_trunk() -> set:
    """``conv_block_{i}`` counted from 0, one per entry of the trunk's filter list (N-2)."""
    return {f"conv_block_{i}" for i in range(len(_CAPS_TRUNK))}


def _caps_v1_owned(reconstruction: bool) -> set:
    """CapsNet: the trunk, the two capsule layers and, with reconstruction, the decoder."""
    return (_caps_trunk() | {"primary_caps", "digit_caps"}
            | ({"reconstruction_decoder"} if reconstruction else set()))


def _caps_v2_owned(reconstruction: bool) -> set:
    """CapsNetV2 with the legacy trunk: the same set. ``stem`` is None there, so it owns no stem."""
    return _caps_v1_owned(reconstruction)


def _caps_v1_build(reconstruction: bool) -> Callable[[], keras.Model]:
    def build() -> keras.Model:
        from dl_techniques.models.vision.capsnet.model import CapsNet
        return CapsNet(num_classes=7, conv_filters=_CAPS_TRUNK, primary_capsules=4,
                       input_shape=_CAPS_IMAGE, reconstruction=reconstruction)
    return build


def _caps_v2_build(reconstruction: bool) -> Callable[[], keras.Model]:
    def build() -> keras.Model:
        from dl_techniques.models.vision.capsnet.model_v2 import CapsNetV2
        return CapsNetV2(num_classes=7, input_shape=_CAPS_IMAGE, legacy_conv_filters=_CAPS_TRUNK,
                         primary_capsules=4, reconstruction=reconstruction,
                         stem_type="legacy")
    return build


#: ``case id -> (build, the exact set of layers the build owns directly)``; the expected sets
#: read the constants above, never the model.
CAPSNET_NUMBERED: Dict[str, Tuple[Callable[[], keras.Model], Callable[[], set]]] = {
    "capsnet": (_caps_v1_build(False), lambda: _caps_v1_owned(False)),
    "capsnet_reconstruction": (_caps_v1_build(True), lambda: _caps_v1_owned(True)),
    "capsnet_v2_legacy": (_caps_v2_build(False), lambda: _caps_v2_owned(False)),
    "capsnet_v2_legacy_reconstruction": (_caps_v2_build(True), lambda: _caps_v2_owned(True)),
}


@pytest.mark.parametrize("case", sorted(CAPSNET_NUMBERED))
def test_the_capsnet_pair_number_their_trunk_blocks_exactly_as_their_filter_lists_say(case: str) -> None:
    build, owned = CAPSNET_NUMBERED[case]
    model = build()
    model(_images(*_CAPS_IMAGE))
    assert numbering_problems(direct_children(model), owned()) == []


def test_capsnet_v2_with_a_resnet_stem_owns_the_backbone_and_no_trunk() -> None:
    from dl_techniques.models.vision.capsnet.model_v2 import CapsNetV2
    model = CapsNetV2(num_classes=7, input_shape=(32, 32, 3), primary_capsules=4,
                      stem_type="resnet18")
    model(_images(32, 32, 3))
    assert model.conv_blocks == []
    assert numbering_problems(direct_children(model), {"stem", "primary_caps", "digit_caps"}) == []


def test_the_numbering_rule_sees_the_spellings_the_capsnet_pair_had() -> None:
    expected = _caps_v1_owned(True)
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "the old conv_{i+1} spelling": ["conv_1" if n == "conv_block_0" else n for n in clean],
        "the old legacy_conv spelling": ["legacy_conv_1" if n == "conv_block_0" else n for n in clean],
        "a block counted from 1": [n.replace("conv_block_0", "conv_block_3") for n in clean],
        "a missing block": [n for n in clean if n != "conv_block_2"],
        "an extra block": [*clean, "conv_block_3"],
        "the decoder gone": [n for n in clean if n != "reconstruction_decoder"],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"


# ---------------------------------------------------------------------------
# DeepSORT, SiamFC and DaSiamRPN own exactly the layers their arguments give
# ---------------------------------------------------------------------------

#: The residual stages of the mars-small128 trunk, in the reference's own numbering.
_DEEPSORT_BLOCKS = ("2_1", "2_3", "3_1", "3_3", "4_1", "4_3")


def _deepsort_owned() -> set:
    """DeepSortAppearanceNet: the stem, six residual blocks, the head dropout, the 128-unit
    bottleneck with its norm, and the cosine head (created always, built only with a top)."""
    return ({"stem", "head_drop", "fc1", "bn_fc1", "cosine_head"}
            | {f"block{n}" for n in _DEEPSORT_BLOCKS})


def _deepsort_build() -> keras.Model:
    from dl_techniques.models.vision.deepsort.appearance import DeepSortAppearanceNet
    return DeepSortAppearanceNet()


def _siam_backbone_owned() -> set:
    """The AlexNet-style trunk of SiamFC and DaSiamRPN: stage 1 is the ``stem``; stages two to
    five keep the reference's own names, each convolution with its BatchNorm (``bn2`` to
    ``bn4`` are created even when the model uses none) and the second pool."""
    return {"stem", "conv2", "bn2", "pool2", "conv3", "bn3", "conv4", "bn4", "conv5"}


def _dasiamrpn_owned() -> set:
    """DaSiamRPN: the shared trunk and the five adjust convolutions."""
    return {"backbone", "conv_r1", "conv_r2", "conv_cls1", "conv_cls2", "regress_adjust"}


def _siamfc_pair() -> Tuple[np.ndarray, np.ndarray]:
    return _images(127, 127, 3, batch=1), _images(255, 255, 3, batch=1)


def _dasiamrpn_pair() -> Tuple[np.ndarray, np.ndarray]:
    return _images(127, 127, 3, batch=1), _images(271, 271, 3, batch=1)


def _siamfc_backbone_build() -> keras.layers.Layer:
    from dl_techniques.models.vision.siamfc.model import SiamFCBackbone
    return SiamFCBackbone()


def _siamrpn_backbone_build() -> keras.layers.Layer:
    from dl_techniques.models.vision.dasiamrpn.model import SiamRPNBackbone
    return SiamRPNBackbone()


def _depth_anything_encoder_owned() -> set:
    """The placeholder encoder's graph: the stem, four two-convolution stages (the body keeps its
    ``conv_block_{i}_{j}`` spelling; only stage one of the trunk is a stem), three pools between
    them, and the projection to the ViT width."""
    body = {f"{kind}_block_{i}_{j}" for kind in ("conv", "bn", "relu") for i in range(4) for j in (1, 2)}
    return {"initial_stem", "feature_projection"} | body | {f"pool_block_{i}" for i in range(3)}


def _depth_anything_build(teacher: bool = False) -> keras.Model:
    from dl_techniques.models.vision.depth_anything.model import DepthAnything
    model = DepthAnything(encoder_kind="placeholder", encoder_type="vit_s", image_shape=(64, 64, 3),
                          use_feature_alignment=teacher, enable_semi_supervised=teacher)
    model.build((None, 64, 64, 3))
    return model


def _depth_anything_encoder_build() -> keras.Model:
    return _depth_anything_build().encoder


def _vq_encoder_owned() -> set:
    """The auto-built encoder of ``VQVAERotationTrick`` at a downsample factor of 4 with one
    residual block: two ``ConvStem`` down-steps counted from 0, the residual block's own layers
    (its norm and activation are named inside the block), and the 1x1 projection to the code size."""
    return ({"enc_down_0", "enc_down_1", "enc_to_embedding"}
            | {f"enc_res_0_{part}" for part in ("a", "norm_a", "act", "b", "add")})


def _vq_encoder_build() -> keras.Model:
    from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick
    return VQVAERotationTrick(
        num_embeddings=32, embedding_dim=16, input_shape=(32, 32, 3), hidden_channels=32,
        downsample_factor=4, num_res_blocks=1, norm_type="layer_norm").encoder


def _sam1_prompt_encoder_owned() -> set:
    """SAM1's prompt encoder: the random-Fourier position layer, the four point-type embeddings,
    the padding-point and no-mask embeddings, and the mask stack (a Sequential, named inside)."""
    return ({"pe_layer", "not_a_point_embed", "no_mask_embed", "mask_downscaling"}
            | {f"point_embedding_{i}" for i in range(4)})


def _sam1_mask_stack_owned() -> set:
    """The mask stack: the two-convolution ``stem`` and the 1x1 ``conv3`` to the embedding size."""
    return {"stem", "conv3"}


def _sam1_prompt_encoder_build() -> keras.layers.Layer:
    return REPLAY_SUBJECTS["sam1_prompt_encoder"][0](None)


def _sam1_mask_stack_build() -> keras.Model:
    return _sam1_prompt_encoder_build().mask_downscaling


def _siamfc_build() -> keras.Model:
    from dl_techniques.models.vision.siamfc.model import SiamFC
    return SiamFC()


def _dasiamrpn_build() -> keras.Model:
    from dl_techniques.models.vision.dasiamrpn.model import DaSiamRPN
    return DaSiamRPN.from_variant("otb")


#: ``case id -> (build, the inputs of one forward, the exact set of layers the build owns
#: directly)``; the expected sets are written from the architecture, never read off a model.
TRACKER_NUMBERED: Dict[str, Tuple[Callable[[], keras.Model], Callable[[], Any],
                                  Callable[[], set]]] = {
    "deepsort": (_deepsort_build, lambda: _images(128, 64, 3), _deepsort_owned),
    "siamfc": (_siamfc_build, _siamfc_pair, lambda: {"backbone"}),
    "siamfc_backbone": (_siamfc_backbone_build, lambda: _images(127, 127, 3, batch=1),
                        _siam_backbone_owned),
    "dasiamrpn": (_dasiamrpn_build, _dasiamrpn_pair, _dasiamrpn_owned),
    "siamrpn_backbone": (_siamrpn_backbone_build, lambda: _images(127, 127, 3, batch=1),
                         _siam_backbone_owned),
    # The frozen teacher is ``frozen_<student name>`` (D-055). It used to share the student's name, and
    # the old distinct-name set read one encoder whether or not a teacher existed.
    "depth_anything": (_depth_anything_build, lambda: _images(64, 64, 3),
                       lambda: {"encoder_vit_s", "depth_head", "strong_augmentation"}),
    "depth_anything_with_teacher": (lambda: _depth_anything_build(teacher=True), lambda: _images(64, 64, 3),
                                    lambda: {"encoder_vit_s", "frozen_encoder_vit_s", "depth_head",
                                             "strong_augmentation"}),
    "depth_anything_encoder": (_depth_anything_encoder_build, lambda: _images(64, 64, 3),
                               _depth_anything_encoder_owned),
    "vq_vae_rotation_encoder": (_vq_encoder_build, lambda: _images(32, 32, 3), _vq_encoder_owned),
    "sam1_mask_stack": (_sam1_mask_stack_build, lambda: _images(32, 32, 1), _sam1_mask_stack_owned),
}


@pytest.mark.parametrize("case", sorted(TRACKER_NUMBERED))
def test_a_tracker_owns_exactly_the_layers_its_architecture_names(case: str) -> None:
    build, inputs, owned = TRACKER_NUMBERED[case]
    model = build()
    model(inputs())
    assert numbering_problems(direct_children(model), owned()) == []


def test_the_tracker_name_rule_sees_the_spellings_deepsort_had() -> None:
    expected = _deepsort_owned()
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "the loose first conv": [*(n for n in clean if n != "stem"), "conv1_1"],
        "the loose first norm": [*(n for n in clean if n != "stem"), "bn1_1"],
        "the loose pool": [*clean, "pool1"],
        "a stem left beside the loose layers": [*clean, "conv1_2", "bn1_2"],
        "a missing stage": [n for n in clean if n != "block3_1"],
        "a renamed head": ["classifier" if n == "cosine_head" else n for n in clean],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"


def test_the_tracker_name_rule_sees_the_spellings_the_siamese_trunk_had() -> None:
    expected = _siam_backbone_owned()
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "the loose first conv": [*(n for n in clean if n != "stem"), "conv1"],
        "the loose first norm": [*clean, "bn1"],
        "the loose first pool": [*clean, "pool1"],
        "the stem beside its three loose layers": [*clean, "conv1", "bn1", "pool1"],
        "a missing body norm": [n for n in clean if n != "bn3"],
        "a body stage renamed": ["conv_2" if n == "conv2" else n for n in clean],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"


def test_the_tracker_name_rule_sees_the_spellings_the_depth_anything_encoder_had() -> None:
    expected = _depth_anything_encoder_owned()
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "the loose first conv": [*(n for n in clean if n != "initial_stem"), "initial_conv"],
        "the loose first norm": [*clean, "initial_bn"],
        "the loose first activation": [*clean, "initial_relu"],
        "the stem beside its three loose layers": [*clean, "initial_conv", "initial_bn", "initial_relu"],
        "a missing pool": [n for n in clean if n != "pool_block_1"],
        "a stage of three convolutions": [*clean, "conv_block_0_3"],
        "a stage renamed": ["conv_block_4_1" if n == "conv_block_3_1" else n for n in clean],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"


def test_the_sam1_prompt_encoder_owns_exactly_the_layers_its_architecture_names() -> None:
    """``PromptEncoder.call`` takes points, boxes and masks, not one tensor, so it has its own row."""
    assert numbering_problems(direct_children(_sam1_prompt_encoder_build()),
                              _sam1_prompt_encoder_owned()) == []


def test_the_tracker_name_rule_sees_the_spellings_the_sam1_mask_stack_had() -> None:
    expected = _sam1_mask_stack_owned()
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "the loose first conv": [*(n for n in clean if n != "stem"), "conv1"],
        "the loose norms and activations": [*clean, "norm1", "act1", "conv2", "norm2", "act2"],
        "the projection renamed": ["conv_3" if n == "conv3" else n for n in clean],
        "a missing projection": [n for n in clean if n != "conv3"],
        "a second stem": [*clean, "stem_1"],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"


def test_the_tracker_name_rule_sees_the_spellings_the_vq_encoder_had() -> None:
    expected = _vq_encoder_owned()
    clean = sorted(expected)
    assert numbering_problems(clean, expected) == []
    plants = {
        "the loose down-step norm": [*clean, "enc_norm_0"],
        "the loose down-step activation": [*clean, "enc_act_1"],
        "a down-step beside its loose layers": [*clean, "enc_norm_0", "enc_act_0"],
        "a missing down-step": [n for n in clean if n != "enc_down_1"],
        "a down-step counted from 1": ["enc_down_2" if n == "enc_down_1" else n for n in clean],
        "a third down-step": [*clean, "enc_down_2"],
    }
    for why, owned in plants.items():
        assert numbering_problems(owned, expected), f"{why} passed the numbering rule"


# ---------------------------------------------------------------------------
# A constructor with no input_shape parameter refuses the keyword
# ---------------------------------------------------------------------------

#: ``key -> constructor taking the keywords under test``, for every finished model or layer
#: whose ``__init__`` has no ``input_shape`` parameter. ``keras.layers.Layer.__init__`` pops
#: ``input_shape`` and ``input_dim`` and only warns, so a caller written against a signature
#: that once took one keeps constructing a model that ignores it.
NO_SHAPE_CONSTRUCTORS: Dict[str, Callable[..., Any]] = {}


def _no_shape(key: str):
    def register(build: Callable[..., Any]) -> Callable[..., Any]:
        NO_SHAPE_CONSTRUCTORS[key] = build
        return build
    return register


@_no_shape("rbf_protonet")
def _rbf_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet
    return RBFProtoNet(num_classes=5, **kw)


@_no_shape("clifford_rbf_protonet")
def _clifford_rbf_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.rbf_protonet.model import CliffordRBFProtoNet
    return CliffordRBFProtoNet(num_classes=5, channels=16, depth=1, **kw)


@_no_shape("rbf_protonet_factory")
def _rbf_factory_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.rbf_protonet.model import create_rbf_protonet
    return create_rbf_protonet(num_classes=5, **kw)


@_no_shape("clifford_rbf_protonet_factory")
def _clifford_rbf_factory_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.rbf_protonet.model import create_clifford_rbf_protonet
    return create_clifford_rbf_protonet(num_classes=5, channels=16, depth=1, **kw)


@_no_shape("uca_encoder")
def _uca_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder
    return UcaEncoder(stage_dims=[16, 32], stage_depths=[1, 1], num_heads=[2, 4], **kw)


@_no_shape("mambalct")
def _mambalct_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.mambalct import MambaLCT
    return MambaLCT(template_size=32, search_size=64, context_len=1, stage_dims=[64, 128, 512],
                    stage_depths=[1, 1, 1], num_heads=[2, 4, 8], fusion_heads=8,
                    head_hidden_dim=32, **kw)


@_no_shape("convunext_stem")
def _convunext_stem_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.convunext.model import ConvUNextStem
    return ConvUNextStem(filters=4, **kw)


@_no_shape("cliffordnet")
def _cliffordnet_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.cliffordnet.model import CliffordNet
    return CliffordNet(num_classes=5, channels=16, depth=1, **kw)


@_no_shape("hierarchical_cliffordnet")
def _hierarchical_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.cliffordnet.hierarchical import HierarchicalCliffordNet
    return HierarchicalCliffordNet(num_classes=5, **kw)


@_no_shape("deepsort")
def _deepsort_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.deepsort.appearance import DeepSortAppearanceNet
    return DeepSortAppearanceNet(**kw)


@_no_shape("deepsort_residual_block")
def _deepsort_block_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.deepsort.appearance import DeepSortResidualBlock
    return DeepSortResidualBlock(filters=8, **kw)


@_no_shape("deepsort_cosine_classifier")
def _deepsort_head_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.deepsort.appearance import CosineClassifier
    return CosineClassifier(num_classes=3, **kw)


@_no_shape("siamfc")
def _siamfc_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.siamfc.model import SiamFC
    return SiamFC(**kw)


@_no_shape("siamfc_backbone")
def _siamfc_backbone_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.siamfc.model import SiamFCBackbone
    return SiamFCBackbone(**kw)


@_no_shape("dasiamrpn")
def _dasiamrpn_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dasiamrpn.model import DaSiamRPN
    return DaSiamRPN(**kw)


@_no_shape("siamrpn_backbone")
def _siamrpn_backbone_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dasiamrpn.model import SiamRPNBackbone
    return SiamRPNBackbone(**kw)


@_no_shape("sam1_position_embedding_random")
def _sam1_position_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.prompt_encoder import PositionEmbeddingRandom
    return PositionEmbeddingRandom(**kw)


@_no_shape("sam1_prompt_encoder")
def _sam1_prompt_encoder_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.prompt_encoder import PromptEncoder
    return PromptEncoder(embed_dim=32, image_embedding_size=(8, 8), input_image_size=(128, 128), **kw)


@_no_shape("depth_anything")
def _depth_anything_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.depth_anything.model import DepthAnything
    return DepthAnything(encoder_kind="placeholder", image_shape=(64, 64, 3), **kw)


@_no_shape("siamfc_factory")
def _siamfc_factory_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.siamfc.model import create_siamfc
    return create_siamfc(**kw)


@_no_shape("dasiamrpn_factory")
def _dasiamrpn_factory_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dasiamrpn.model import create_dasiamrpn
    return create_dasiamrpn(**kw)


@_no_shape("deepsort_factory")
def _deepsort_factory_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.deepsort.appearance import create_deepsort_embedding
    return create_deepsort_embedding(**kw)


@_no_shape("depth_anything_factory")
def _depth_anything_factory_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.depth_anything.model import create_depth_anything
    return create_depth_anything(encoder_kind="placeholder", image_shape=(64, 64, 3), **kw)


#: Rows added by the census (step 19.2): the owned blocks and the models of the finished families that had no row,
#: each built with the smallest arguments its constructor accepts.
@_no_shape("conv_stem")
def _conv_stem_without_shape(**kw: Any) -> Any:
    from dl_techniques.layers.conv_blocks.stem import ConvStem
    return ConvStem(8, **kw)


@_no_shape("convunext_spatial_linear_attention")
def _spatial_linear_attention_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.convunext.model import SpatialLinearAttention
    return SpatialLinearAttention(dim=8, num_heads=2, **kw)


@_no_shape("dpt_head")
def _dpt_head_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.depth_anything.components import DPTHead
    return DPTHead(**kw)


@_no_shape("dpt_decoder")
def _dpt_decoder_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.depth_anything.components import DPTDecoder
    return DPTDecoder(**kw)


@_no_shape("dpt_feature_fusion_block")
def _dpt_fusion_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.depth_anything.components import DPTFeatureFusionBlock
    return DPTFeatureFusionBlock(features=8, **kw)


@_no_shape("dpt_residual_conv_unit")
def _dpt_residual_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.depth_anything.components import DPTResidualConvUnit
    return DPTResidualConvUnit(features=8, **kw)


@_no_shape("dino_head")
def _dino_head_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dino.dino_v1 import DINOHead
    return DINOHead(in_dim=8, out_dim=16, **kw)


@_no_shape("dinov2_block")
def _dinov2_block_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dino.dino_v2 import DINOv2Block
    return DINOv2Block(dim=16, num_heads=2, **kw)


@_no_shape("dinov2_position_embedding")
def _dinov2_position_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dino.dino_v2 import DINOv2PositionEmbedding
    return DINOv2PositionEmbedding(grid_size=2, output_grid=(2, 2), **kw)


@_no_shape("dino_training_model")
def _dino_training_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.dino.training import DINOTrainingModel
    return DINOTrainingModel(student=None, teacher=None, **kw)


@_no_shape("squeezenet_fire_module")
def _fire_module_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.squeezenet.squeezenet_v1 import FireModule
    return FireModule(s1x1=8, e1x1=16, e3x3=16, **kw)


@_no_shape("squeezenet_simplified_fire_module")
def _simplified_fire_module_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SimplifiedFireModule
    return SimplifiedFireModule(s1x1=8, e3x3=16, **kw)


@_no_shape("clifford_clip")
def _clifford_clip_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.clifford_clip.model import CliffordCLIP
    return CliffordCLIP(**kw)


@_no_shape("sam1_image_encoder")
def _sam1_image_encoder_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.image_encoder import ImageEncoderViT
    return ImageEncoderViT(**kw)


@_no_shape("sam1_vit_block")
def _sam1_vit_block_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.image_encoder import ViTBlock
    return ViTBlock(dim=16, num_heads=2, **kw)


@_no_shape("sam1_windowed_attention")
def _sam1_attention_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.image_encoder import WindowedAttentionWithRelPos
    return WindowedAttentionWithRelPos(dim=16, num_heads=2, **kw)


@_no_shape("sam1_mask_decoder")
def _sam1_mask_decoder_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.mask_decoder import MaskDecoder
    return MaskDecoder(transformer_dim=16, transformer=None, **kw)


@_no_shape("sam1_model")
def _sam1_model_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.model import SAM
    return SAM(image_encoder=None, prompt_encoder=None, mask_decoder=None, **kw)


@_no_shape("sam1_training_model")
def _sam1_training_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.training_model import SAMTrainingModel
    return SAMTrainingModel(sam=None, **kw)


@_no_shape("sam1_two_way_attention_block")
def _sam1_two_way_block_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.transformer import TwoWayAttentionBlock
    return TwoWayAttentionBlock(embedding_dim=16, num_heads=2, mlp_dim=32, **kw)


@_no_shape("sam1_two_way_transformer")
def _sam1_two_way_transformer_without_shape(**kw: Any) -> Any:
    from dl_techniques.models.vision_language.sam.sam1.transformer import TwoWayTransformer
    return TwoWayTransformer(depth=1, embedding_dim=16, num_heads=2, mlp_dim=32, **kw)


#: Rows whose arguments are placeholders (``None`` for a wrapped model) or whose valid construction is a full model: the
#: constructor refuses a shape keyword FIRST, which these rows observe; an unknown keyword reaches Keras only after the
#: constructor has checked or built its arguments, so the unknown-keyword row does not apply to them.
SHAPE_REFUSAL_ONLY = frozenset({
    "dino_training_model", "clifford_clip", "sam1_image_encoder", "sam1_mask_decoder", "sam1_model",
    "sam1_training_model"})


def swallowed_shape_keywords(build: Callable[..., Any]) -> List[str]:
    """The shape keywords ``build`` accepts without raising ``ValueError`` (empty when it refuses both).

    Interface contract: ``build(**kw)`` constructs the subject with the given extra keywords. A
    keyword counts as swallowed when construction succeeds (or Keras only warns); a ``ValueError``
    or ``TypeError`` that does not name the keyword also counts, because the refusal must say what was
    wrong. A ``TypeError`` is Python's own refusal of a function that has no ``**kwargs`` (a factory with
    a fixed signature) and names the keyword. Any other exception propagates.
    """
    import warnings
    swallowed = []
    for key in ("input_shape", "input_dim"):
        value = (8, 8, 3) if key == "input_shape" else 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                build(**{key: value})
            except (ValueError, TypeError) as error:
                if key not in str(error):
                    swallowed.append(key)
            else:
                swallowed.append(key)
    return swallowed


@pytest.mark.parametrize("key", [_param("no_shape_keyword_swallowed", k) for k in NO_SHAPE_CONSTRUCTORS])
def test_a_constructor_without_an_input_shape_parameter_refuses_the_keyword(key: str) -> None:
    assert swallowed_shape_keywords(NO_SHAPE_CONSTRUCTORS[key]) == []


@pytest.mark.parametrize("key", [_param("unknown_keyword_refused", k)
                                 for k in NO_SHAPE_CONSTRUCTORS if k not in SHAPE_REFUSAL_ONLY])
def test_an_unknown_keyword_is_refused_naming_it(key: str) -> None:
    # ValueError from Keras; TypeError from Python for a factory whose signature has no **kwargs
    with pytest.raises((ValueError, TypeError), match="not_an_argument"):
        NO_SHAPE_CONSTRUCTORS[key](not_an_argument=1)


def test_the_no_shape_rule_sees_a_constructor_that_swallows_the_keyword() -> None:
    class Swallows(keras.layers.Layer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    class RefusesElsewhere(keras.layers.Layer):
        def __init__(self, **kwargs: Any) -> None:
            if "input_shape" in kwargs:
                raise ValueError("something unrelated is wrong")
            super().__init__(**kwargs)

    assert swallowed_shape_keywords(lambda **kw: Swallows(**kw)) == ["input_shape", "input_dim"]
    assert swallowed_shape_keywords(lambda **kw: RefusesElsewhere(**kw)) == ["input_shape", "input_dim"]
    from dl_techniques.utils.model_build import refuse_input_shape

    class Refuses(keras.layers.Layer):
        def __init__(self, **kwargs: Any) -> None:
            refuse_input_shape(kwargs, "Refuses", "it has none")
            super().__init__(**kwargs)

    assert swallowed_shape_keywords(lambda **kw: Refuses(**kw)) == []


# ---------------------------------------------------------------------------
# The refusal is derived from the source, not remembered
# ---------------------------------------------------------------------------

#: Calls that refuse a shape keyword by name: ``refuse_input_shape`` for both keywords, and
#: ``refuse_unknown_model_kwargs`` for a functional ``keras.Model`` subclass (every key outside ``name``, ``trainable``
#: and ``dtype`` is refused, and the message prints the dict, so it names ``input_shape``).
_REFUSERS = ("refuse_input_shape", "refuse_unknown_model_kwargs")
_KERAS_BASES = frozenset({"Layer", "Model", "Sequential"})
_STEM_FILE = SRC / "layers" / "conv_blocks" / "stem.py"


def scan_constructors(sources: Dict[str, str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Which Keras layers and models in ``sources`` would swallow an ``input_shape`` keyword.

    Interface contract: ``sources`` maps a label (a path) to Python source text. A class counts when it derives
    from ``Layer``, ``Model`` or ``Sequential``, directly or through a class in ``sources`` (resolved by name).
    ``keras.layers.Layer.__init__`` pops ``input_shape`` and ``input_dim`` and only warns, so a class that has no
    such parameter and forwards ``**kwargs`` must refuse them itself. Returns ``(violations, exempt, checked)``:
    ``violations`` are ``"<label>:<line> <Class>"`` for a class that takes no shape parameter, accepts ``**kwargs``
    and neither calls a refuser (:data:`_REFUSERS`) nor inherits a constructor that does; ``exempt`` maps a class
    name to the reason it needs no refusal of its own (it inherits a constructor, or forwards ``**kwargs`` to a parent
    that takes or refuses the keyword); ``checked`` lists every class the scan read, so a caller can assert the scan was
    not empty. A class with no ``**kwargs`` is in neither (Python raises ``TypeError`` for the unknown keyword), a
    class that is not a Keras layer or model is not read. Raises ``SyntaxError`` for source that does not parse.
    """
    classes: Dict[Tuple[str, str], ast.ClassDef] = {}
    for label, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef):
                classes[(label, node.name)] = node
    by_name: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for key in classes:
        by_name[key[1]].append(key)

    def base_names(node: ast.ClassDef) -> List[str]:
        return [ast.unparse(b).split(".")[-1] for b in node.bases]

    derived = set()
    changed = True
    while changed:
        changed = False
        for key, node in classes.items():
            if key not in derived and any(
                    b in _KERAS_BASES or any(k in derived for k in by_name.get(b, ()))
                    for b in base_names(node)):
                derived.add(key)
                changed = True

    verdicts: Dict[Tuple[str, str], str] = {}

    def verdict(key: Tuple[str, str]) -> str:
        """``ok`` / ``no-kwargs`` / ``inherits <Parent>`` / ``forwards <Parent>`` / ``violation``."""
        if key in verdicts:
            return verdicts[key]
        verdicts[key] = "violation"  # a cycle cannot happen in valid source; this keeps the walk finite
        node = classes[key]
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        parents = [p for b in base_names(node) for p in by_name.get(b, ()) if p in derived]
        parent_ok = next((p for p in parents if verdict(p) != "violation"), None)
        if init is None:
            result = f"inherits {parent_ok[1]}" if parent_ok else "violation"
        else:
            params = {a.arg for a in init.args.posonlyargs + init.args.args + init.args.kwonlyargs}
            text = ast.unparse(init)
            if "input_shape" in params or "input_dim" in params or any(r in text for r in _REFUSERS):
                result = "ok"
            elif init.args.kwarg is None:
                result = "no-kwargs"
            elif parent_ok and "super().__init__(" in text:
                result = f"forwards {parent_ok[1]}"
            else:
                result = "violation"
        verdicts[key] = result
        return result

    violations, exempt = [], {}
    for key in sorted(derived):
        result = verdict(key)
        if result == "violation":
            violations.append(f"{key[0]}:{classes[key].lineno} {key[1]}")
        elif result.startswith(("inherits", "forwards")):
            exempt[key[1]] = result
    return violations, exempt, sorted(k[1] for k in derived)


def _finished_package_sources() -> Dict[str, str]:
    """The source of every module in the packages of the replay roster's models, and of ``ConvStem``."""
    packages = sorted({Path(p) for key in REPLAY_SUBJECTS for p in roster_facts(key)[1]})
    files = [f for package in packages for f in sorted(package.rglob("*.py"))] + [_STEM_FILE]
    return {str(f.relative_to(SRC)): f.read_text() for f in files}


#: The classes the census exempts, with the reason (the scan derives the reason; this pins WHICH classes, so a new one
#: is a visible edit). ``ConvNeXtV1`` and ``ConvNeXtV2`` have no ``__init__``: ``_ConvNeXtBase`` takes ``input_shape``
#: and rank-checks it, and they run exactly that constructor.
CENSUS_EXEMPT = {
    "ConvNeXtV1": "inherits _ConvNeXtBase",
    "ConvNeXtV2": "inherits _ConvNeXtBase",
}


def test_every_keras_constructor_of_a_finished_package_that_takes_no_shape_refuses_the_keyword() -> None:
    """The census behind ``NO_SHAPE_CONSTRUCTORS``: derived from the roster and the source, not from memory.

    R4 finding 3: ``refuse_input_shape`` sat in the constructors someone remembered, and ``DINOv2Block``,
    ``FireModule``, ``SpatialLinearAttention`` and ``ConvStem`` still built with ``input_shape=...`` and only a Keras
    warning. The packages are the ones the replay roster's models are defined in (so a finished family joins when its
    model joins the roster); every module in them is parsed, and so is ``ConvStem``'s.
    """
    sources = _finished_package_sources()
    violations, exempt, checked = scan_constructors(sources)
    assert violations == []
    assert exempt == CENSUS_EXEMPT
    # control: the scan read the classes it is about, in a package of every kind
    assert {"ConvStem", "DINOv2Block", "FireModule", "SimplifiedFireModule", "SpatialLinearAttention",
            "DeepSortResidualBlock", "PromptEncoder", "DepthAnything", "DPTHead"} <= set(checked)
    assert len(checked) > 50


def test_every_class_the_census_names_has_a_row_that_builds_it() -> None:
    """A census class that is a block or a model of a finished family has a runtime row, except the listed ones."""
    _, _, checked = scan_constructors(_finished_package_sources())
    rows = {"ConvStem", "SpatialLinearAttention", "DPTHead", "DPTDecoder", "DPTFeatureFusionBlock",
            "DPTResidualConvUnit", "DINOHead", "DINOv2Block", "DINOv2PositionEmbedding", "DINOTrainingModel",
            "FireModule", "SimplifiedFireModule", "CliffordCLIP", "ImageEncoderViT", "ViTBlock",
            "WindowedAttentionWithRelPos", "MaskDecoder", "SAM", "SAMTrainingModel", "TwoWayAttentionBlock",
            "TwoWayTransformer"}
    census_classes = {name for name in checked}
    # every class of the table above is read by the scan, and the runtime table has a row for each
    assert rows <= census_classes
    assert len(NO_SHAPE_CONSTRUCTORS) >= 17 + len(rows)


def test_the_constructor_census_sees_each_kind_of_swallow_and_each_exemption() -> None:
    planted = {"planted.py": """
import keras
class Swallows(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
class SwallowsModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
class NoConstructor(keras.layers.Layer):
    pass
class Refuses(keras.layers.Layer):
    def __init__(self, **kwargs):
        refuse_input_shape(kwargs, "Refuses", "it has none")
        super().__init__(**kwargs)
class RefusesUnknown(keras.Model):
    def __init__(self, **kwargs):
        refuse_unknown_model_kwargs(kwargs, "RefusesUnknown")
        super().__init__(**kwargs)
class Takes(keras.layers.Layer):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
class TakesDim(keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
class NoKwargs(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
class InheritsRefuser(Refuses):
    pass
class InheritsTaker(Takes):
    pass
class ForwardsToRefuser(Refuses):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
class SubclassOfSwallower(Swallows):
    def __init__(self, **kwargs):
        super().__init__(units=1, **kwargs)
class NotKeras:
    def __init__(self, **kwargs):
        pass
class Outer:
    class Nested(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
"""}
    violations, exempt, checked = scan_constructors(planted)
    assert sorted(v.split()[-1] for v in violations) == sorted(
        ["Nested", "NoConstructor", "Swallows", "SwallowsModel", "SubclassOfSwallower"])
    assert exempt == {"InheritsRefuser": "inherits Refuses", "InheritsTaker": "inherits Takes",
                      "ForwardsToRefuser": "forwards Refuses"}
    assert "NotKeras" not in checked and "NoKwargs" in checked
    assert scan_constructors({"empty.py": "x = 1\n"}) == ([], {}, [])


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
    rows |= {("no_shape_keyword_swallowed", key) for key in NO_SHAPE_CONSTRUCTORS}
    rows |= {("unknown_keyword_refused", key) for key in NO_SHAPE_CONSTRUCTORS if key not in SHAPE_REFUSAL_ONLY}
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
