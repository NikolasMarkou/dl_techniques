"""``ConvStem`` replicates every stem it replaced, layer for layer.

The commit ``20b48aba6`` moved ten hand-rolled stems onto ``ConvStem`` and the
move was not exact: ResNet's ``MaxPooling2D(3, strides=2, padding="same")`` became
a stride-3 ``valid`` pool (224 px -> 37x37 instead of 56x56), and no test saw
it because every comparison in the tree was between the NEW code and itself.
This file compares the stem a model builds against the ORIGINAL layers, rebuilt
inline here from the pre-migration revision ``20b48aba6^`` (``git show
20b48aba6^:<path>``) with every constructor argument written out as a literal, never read back from the
model under test.

How a case is checked (``assert_stem_matches_original``):

1. The original layers are built and every weight is replaced by a distinct
   non-trivial value. A freshly built BatchNormalization is the identity at
   ``training=False`` (moving mean 0, variance 1, gamma 1, beta 0), so a
   comparison made on it cannot see a lost norm or a changed epsilon.
2. Each original sub-layer's weights are transplanted into the sub-layer of the
   same role in the model's ``ConvStem``. The ROLE sets must be identical (a
   stem with an extra or a missing norm is a defect by itself) and so must the
   weight counts.
3. Inference output (``training=False``), the regularization loss, the
   ``training=True`` output and every weight after the ``training=True`` pass
   (BN moving statistics move) are compared at ``rtol=0, atol=0``. The op
   sequence of the original and of ``ConvStem`` is identical, so the result is
   exactly reproducible on CPU; a bound would only hide a defect. A case whose
   original is NOT op-for-op identical (for example a fused ``activation=``
   against a separate ``Activation`` layer) must state and derive its own bound
   with ``tests.numerics`` instead of widening this one.

How to add a case (one per original variant, later steps extend the table):

* Write ``_<model>_original(...)`` returning an ``Original``: ``sublayers``
  keyed by the ``ConvStem`` attribute the layer corresponds to (``conv1``,
  ``norm1``, ``act1``, ``conv2``, ``norm2``, ``act2``, ``pool``;
  omit a role the original had no layer for) and ``forward(x, training)``
  chaining them exactly as the original ``call()`` did. An activation the
  original FUSED into its conv (``Conv2D(activation="relu")``, no layer of its
  own, no weights) is declared in ``Original.fused`` (``("act1",)``)
  instead of ``sublayers``: the role-set check then expects ``ConvStem`` to
  carry a weightless layer in that role and nothing else, and the comparison
  stays at ``rtol=0, atol=0`` because the conv, then the same ReLU, is the same
  op sequence. Every other case must still list every role the stem carries.
* Write a ``_<model>_model()`` factory returning a built-able model whose
  ``.stem`` is the ``ConvStem`` under test. A model that keeps its stem
  elsewhere (the VAE builds a functional graph, so its stem is a layer of it)
  gives the case a ``stem_of(model)`` accessor instead; ``StemCase.stem_of``
  and the optional fourth element of a ``BUILT_STEM_ROWS`` row carry it.
* Append ONE ``StemCase`` to ``CASES`` and, when the model has a ``stem``
  attribute, ONE row to ``BUILT_STEM_ROWS`` (``test_every_migrated_model_
  builds_its_stem``). Nothing else in this file changes.

Planted controls prove the harness can see each defect class it exists for:
a wrong pool, a wrong epsilon and a norm applied twice are each handed to the
SAME comparison and the comparison must FAIL.

Four guards cover what the weight transplant cannot see (D-014, D-020):

* ``assert_stem_settings_match_original``: the transplant overwrites every
  weight, so a dropped or wrong kernel/bias initializer is invisible to it. The
  serialized initializers (``seed`` ignored, so D-005's independent ``conv2``
  draw is not flagged) and regularizers of each original conv are compared with
  the stem's. That comparison is only as sharp as the value the model is built
  with: a model whose knob default equals ConvStem's (MobileNetV3's
  ``glorot_uniform``) would pass with a dropped passthrough.
* ``INITIALIZER_AUDIT`` and ``assert_initializer_audit``: one row per model
  class saying how its initializer reaches the stem (a knob built with a value
  that is neither the model's default nor ConvStem's, a hardcoded literal, a
  knob the stem does not read, or no knob at all) with the reason where nothing
  can be made non-default. The audit is checked against each model's real
  constructor signature, so a model that gains or loses a knob forces the table
  to change. What it does NOT cover: normalization gamma/beta/moving-statistic
  initializers and norm regularizers (no migrated model sets a non-default one).
* ``test_every_migrated_model_applies_its_stem_exactly_once``: the stem is
  compared in isolation, so a model that applies it (or a layer of it) again in
  its own ``call`` is invisible; the calls of each stem layer in one forward are
  counted.
* ``check_training_none``: ``training=None`` is compared as well as False/True.
"""

import contextlib
import inspect
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, Optional, Tuple
from unittest import mock

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations import create_activation_layer
from dl_techniques.layers.activations.hard_swish import HardSwish
from dl_techniques.layers.activations.exact_activations import (
    exact_gelu,
    exact_leaky_relu,
)
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.vision.capsnet.model import CapsNet
from dl_techniques.models.vision.capsnet.model_v2 import CapsNetV2
from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2
from dl_techniques.models.vision.convunext.model import ConvUNextStem
from dl_techniques.models.vision.dasiamrpn.model import DaSiamRPN, SiamRPNBackbone
from dl_techniques.models.vision.deepsort.appearance import DeepSortAppearanceNet
from dl_techniques.models.vision.cliffordnet.hierarchical import (
    HierarchicalCliffordNet,
)
from dl_techniques.models.vision.cliffordnet.model import CliffordNet
from dl_techniques.models.vision.mambalct.uca_encoder import UcaEncoder
from dl_techniques.models.vision.mobilenet.mobilenet_v1 import MobileNetV1
from dl_techniques.models.vision.mobilenet.mobilenet_v2 import MobileNetV2
from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3
from dl_techniques.models.vision.mobilenet.mobilenet_v4 import MobileNetV4
from dl_techniques.models.vision.rad_convnet.model import RADConvNet
from dl_techniques.models.vision.rbf_protonet.model import CliffordRBFProtoNet
from dl_techniques.models.vision.resnet.model import ResNet
from dl_techniques.models.vision.siamfc.model import SiamFC, SiamFCBackbone
from dl_techniques.models.vision.squeezenet.squeezenet_v1 import SqueezeNetV1
from dl_techniques.models.vision.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2
from dl_techniques.models.vision.vae.model import VAE
from tests.layer_naming import PublicAutoName
from tests.numerics import true_float32_matmul



@pytest.fixture(autouse=True)
def _true_float32():
    """Scope TF32 off so the exact comparisons are made in true float32 on a
    TF32-capable GPU as well as on CPU (restored, and asserted, on exit)."""
    with true_float32_matmul():
        yield

BATCH = 2

#: Sub-layer attributes of ``ConvStem`` a case may supply, in forward order.
ROLES = ("conv1", "norm1", "act1", "conv2", "norm2", "act2", "pool")


# ---------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------


@dataclass
class Original:
    """The pre-migration stem: its sub-layers by ConvStem role and its forward.

    :param sublayers: ``{role: layer}`` over ``ROLES``; a role with no entry
        means the original had no such layer.
    :param forward: ``forward(x, training)`` chaining the layers exactly as the
        original ``call()`` did.
    :param fused: Roles the original had NO layer for because the activation
        was fused into its conv; ``ConvStem`` must carry a weightless layer in
        each of them.
    """

    sublayers: Dict[str, keras.layers.Layer]
    forward: Callable[[object, Optional[bool]], object]
    fused: Tuple[str, ...] = ()


@dataclass
class StemCase:
    """One original stem variant.

    :param case_id: pytest id.
    :param make_model: returns a fresh model that holds the ``ConvStem`` under test.
    :param make_original: returns the ``Original`` (fresh layers every call).
    :param input_shape: ``(H, W, C)`` fed to the stem.
    :param expected_output_shape: ``(H', W', C')`` the ORIGINAL stem yields, so
        a geometry defect is named by its shape, not only by its values.
    :param input_scale: Factor on the standard-normal input. A case whose
        activation is clipped needs inputs large enough to reach the clip: with
        the randomized weights below the activation input is about 0.3 standard
        deviations wide, so a ReLU6 never clips and a plain ReLU passes for it.
    """

    case_id: str
    make_model: Callable[[], keras.Model]
    make_original: Callable[[], Original]
    input_shape: Tuple[int, int, int]
    expected_output_shape: Tuple[int, int, int]
    stem_of: Callable[[keras.Model], ConvStem] = lambda model: model.stem
    input_scale: float = 1.0


def _np(tensor) -> np.ndarray:
    return keras.ops.convert_to_numpy(tensor)


def _randomize(layer: keras.layers.Layer, rng: np.random.Generator) -> None:
    """Give every weight of ``layer`` a distinct non-trivial value."""
    for weight in layer.weights:
        shape = tuple(weight.shape)
        path = weight.path
        if "moving_variance" in path or "gamma" in path:
            values = rng.uniform(0.5, 1.5, size=shape)
        elif "kernel" in path:
            values = rng.standard_normal(shape) * 0.05
        else:  # beta, moving_mean, bias
            values = rng.standard_normal(shape) * 0.1
        weight.assign(values.astype("float32"))


def _stem_sublayers(stem: ConvStem) -> Dict[str, keras.layers.Layer]:
    return {
        role: getattr(stem, role)
        for role in ROLES
        if getattr(stem, role) is not None
    }


def _regularization_loss(layers) -> float:
    return float(sum(float(_np(loss)) for layer in layers for loss in layer.losses))


def _default_forward(stem: ConvStem, x, training):
    return stem(x, training=training)


def assert_stem_matches_original(
    case: StemCase,
    stem: ConvStem,
    forward: Callable = _default_forward,
    seed: int = 0,
    check_training_none: bool = False,
) -> None:
    """Assert ``forward(stem, ...)`` reproduces the original stem exactly.

    Interface contract: raises ``AssertionError`` (never another type) on any
    difference, including a role-set, weight-count or shape mismatch, so a
    planted control can require ``AssertionError`` specifically. Mutates
    ``stem`` (its weights are overwritten) and builds it when unbuilt.

    :param case: The original variant to compare against.
    :param stem: The ``ConvStem`` under test.
    :param forward: ``forward(stem, x, training)``; defaults to ``stem(x)``.
        A control that applies a layer twice passes its own.
    :param seed: Seed of the weights and the input.
    :param check_training_none: Also compare ``training=None``, after the
        inference comparison and before any ``training=True`` pass moves the
        BN statistics. Keras 3's ``BatchNormalization.call`` takes its batch
        statistics branch only on ``if training and self.trainable``, so ``None``
        is the INFERENCE path (moving statistics), not "unspecified, pick one":
        a top-level call has no enclosing call context to inherit a flag from,
        and ``Layer.__call__`` leaves ``None`` as ``None``. A stem that turns
        ``None`` into ``True`` (or a norm given a different default) would
        change a model's output for every caller that omits ``training``, and
        neither ``False`` nor ``True`` can see it.
    """
    rng = np.random.default_rng(seed)
    x = (
        rng.standard_normal((BATCH,) + tuple(case.input_shape)) * case.input_scale
    ).astype("float32")

    original = case.make_original()
    original.forward(x, False)  # builds the original sub-layers
    for layer in original.sublayers.values():
        _randomize(layer, rng)

    # Build the stem BEFORE touching its sub-layers: building one first makes a
    # later ``stem.build`` raise "already built".
    if not stem.built:
        stem.build(x.shape)

    stem_layers = _stem_sublayers(stem)
    expected_roles = set(original.sublayers) | set(original.fused)
    assert set(stem_layers) == expected_roles, (
        f"{case.case_id}: the stem's sub-layer roles {sorted(stem_layers)} differ "
        f"from the original's {sorted(expected_roles)}"
    )
    for role in original.fused:
        assert role not in original.sublayers, (
            f"{case.case_id}: {role} is declared both fused and a sub-layer"
        )
        assert stem_layers[role].get_weights() == [], (
            f"{case.case_id}: the fused role {role} carries weights"
        )
    for role, source in original.sublayers.items():
        source_weights = source.get_weights()
        target = stem_layers[role]
        assert len(target.get_weights()) == len(source_weights), (
            f"{case.case_id}: {role} carries {len(target.get_weights())} weights, "
            f"the original carried {len(source_weights)}"
        )
        target.set_weights(source_weights)

    def _compare(label: str, got, want) -> None:
        got, want = _np(got), _np(want)
        assert got.shape == want.shape, (
            f"{case.case_id}: {label} shape {got.shape} != original {want.shape}"
        )
        diff = float(np.max(np.abs(got - want)))
        assert diff == 0.0, (
            f"{case.case_id}: {label} differs from the original, "
            f"max|delta| = {diff:.6e} (required exactly 0.0)"
        )

    # Inference.
    want = original.forward(x, False)
    got = forward(stem, x, False)
    expected = (BATCH,) + tuple(case.expected_output_shape)
    assert tuple(want.shape) == expected, (
        f"{case.case_id}: the ORIGINAL rebuilt here yields {tuple(want.shape)}, "
        f"expected {expected}; the reference itself is wrong"
    )
    _compare("training=False output", got, want)
    assert _regularization_loss(stem_layers.values()) == _regularization_loss(
        original.sublayers.values()
    ), f"{case.case_id}: the regularization loss differs from the original"

    if check_training_none:
        _compare(
            "training=None output",
            forward(stem, x, None),
            original.forward(x, None),
        )

    # Training: batch statistics, and the moving statistics move.
    want = original.forward(x, True)
    got = forward(stem, x, True)
    _compare("training=True output", got, want)
    for role, source in original.sublayers.items():
        for index, (w_got, w_want) in enumerate(
            zip(stem_layers[role].get_weights(), source.get_weights())
        ):
            diff = float(np.max(np.abs(w_got - w_want)))
            assert diff == 0.0, (
                f"{case.case_id}: {role} weight {index} differs from the original "
                f"after a training=True pass, max|delta| = {diff:.6e}"
            )


def _initializer_signature(initializer) -> dict:
    """The serialized initializer WITHOUT its ``seed``: class and parameters.

    The seed is a draw detail, not a configuration: a seedless initializer gets
    a fresh default seed per instance, and D-005 deliberately gives the second
    conv of a two-conv stem its own clone (same class and parameters, other
    seed). Comparing the seed would flag that on purpose and nothing else.
    """
    serialized = keras.initializers.serialize(initializer)
    config = dict(serialized["config"])
    config.pop("seed", None)
    return {**serialized, "config": config}


_CONVOLUTIONS = (keras.layers.Conv2D, keras.layers.Conv3D)


def _conv_settings(conv: keras.layers.Layer) -> dict:
    """Kernel/bias initializer (seed-free) and regularizer of one conv."""
    return {
        "kernel_initializer": _initializer_signature(conv.kernel_initializer),
        "bias_initializer": _initializer_signature(conv.bias_initializer),
        "kernel_regularizer": keras.regularizers.serialize(conv.kernel_regularizer),
        "bias_regularizer": keras.regularizers.serialize(conv.bias_regularizer),
    }


def assert_stem_settings_match_original(case: StemCase, stem: ConvStem) -> None:
    """Assert each stem conv draws and regularizes like the original conv.

    ``assert_stem_matches_original`` transplants every weight, so a dropped or
    wrong ``kernel_initializer`` / ``bias_initializer`` is invisible to it (and
    so is a regularizer on a bias-free conv). This compares the serialized
    kernel and bias initializer (class and parameters, ``seed`` ignored) and the
    kernel and bias regularizer of every ORIGINAL conv with the stem conv of the
    same role. Where the original relied on Keras' default (``glorot_uniform`` /
    ``zeros``) that default is the expectation, because the original is a real
    ``Conv2D``.

    Interface contract: raises ``AssertionError`` naming the case, the role and
    the setting on any difference. Builds nothing and mutates nothing: the
    settings are constructor attributes.
    """
    original = case.make_original()
    stem_layers = _stem_sublayers(stem)
    compared = 0
    for role, source in original.sublayers.items():
        if not isinstance(source, _CONVOLUTIONS):
            continue
        target = stem_layers[role]
        assert isinstance(target, type(source)), (
            f"{case.case_id}: {role} is a {type(target).__name__}, "
            f"the original a {type(source).__name__}"
        )
        want, got = _conv_settings(source), _conv_settings(target)
        for setting in want:
            assert got[setting] == want[setting], (
                f"{case.case_id}: {role} {setting} is {got[setting]}, "
                f"the original used {want[setting]}"
            )
        compared += 1
    assert compared >= 1, f"{case.case_id}: the original has no Conv2D or Conv3D to compare"


# ---------------------------------------------------------------------
# ResNet (imagenet and cifar stems)
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/resnet/model.py``
# ``_build_stem`` (Conv2D he_normal, no bias, padding same -> BN via the norm
# factory with momentum 0.9 -> factory ReLU -> MaxPooling2D(3, strides=2,
# padding="same") for the imagenet stem only).

#: ``ResNet``'s ``kernel_initializer`` knob, built with a value that is neither
#: its default (``he_normal``) nor ConvStem's (``glorot_uniform``), so a stem that
#: dropped the argument is visible. The original's literal ``he_normal`` becomes
#: this value, the one place the knob is expected to reach.
_RESNET_KERNEL_INIT = "lecun_normal"


def _resnet_original(stem_type: str) -> Original:
    imagenet = stem_type == "imagenet"
    conv = keras.layers.Conv2D(
        filters=64,
        kernel_size=7 if imagenet else 3,
        strides=2 if imagenet else 1,
        padding="same",
        use_bias=False,
        kernel_initializer=_RESNET_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(1e-4),
        name="stem_conv",
    )
    norm = create_normalization_layer("batch_norm", name="stem_bn", momentum=0.9)
    act = create_activation_layer("relu", name="stem_act")
    pool = (
        keras.layers.MaxPooling2D(
            pool_size=3, strides=2, padding="same", name="stem_pool"
        )
        if imagenet
        else None
    )

    def forward(x, training):
        x = conv(x)
        x = norm(x, training=training)
        x = act(x)
        if pool is not None:
            x = pool(x)
        return x

    sublayers = {"conv1": conv, "norm1": norm, "act1": act}
    if pool is not None:
        sublayers["pool"] = pool
    return Original(sublayers=sublayers, forward=forward)


def _resnet_model(stem_type: str, input_shape=(64, 64, 3)) -> ResNet:
    return ResNet.from_variant(
        "resnet18",
        num_classes=10,
        input_shape=input_shape,
        include_top=False,
        stem_type=stem_type,
        kernel_regularizer=keras.regularizers.L2(1e-4),
        kernel_initializer=_RESNET_KERNEL_INIT,
    )


RESNET_IMAGENET = StemCase(
    case_id="resnet-imagenet",
    make_model=lambda: _resnet_model("imagenet", (224, 224, 3)),
    make_original=lambda: _resnet_original("imagenet"),
    input_shape=(224, 224, 3),
    expected_output_shape=(56, 56, 64),
)
RESNET_CIFAR = StemCase(
    case_id="resnet-cifar",
    make_model=lambda: _resnet_model("cifar", (32, 32, 3)),
    make_original=lambda: _resnet_original("cifar"),
    input_shape=(32, 32, 3),
    expected_output_shape=(32, 32, 64),
)

# ---------------------------------------------------------------------
# CliffordNet and HierarchicalCliffordNet
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/cliffordnet/model.py``
# (``_build_stem`` / ``_apply_stem``) and ``.../hierarchical.py``. Every conv took
# the model's kernel/bias initializer and regularizer; the BatchNormalization
# layers were bare ``keras.layers.BatchNormalization(momentum=0.9)``, so their
# epsilon was the Keras default 1e-3 (D-007 of plan-2026-10-26T125959-7af851e5:
# the pre-migration value, no paper reference was found). The two-conv stems
# (CliffordNet patch 1 and 4, hierarchical patch 1) are bias-free, run
# conv -> BN -> functional silu -> conv and then the shared ``stem_norm`` BN;
# the single-conv stems run conv (bias per ``use_bias``) -> ``stem_norm`` BN.
# The functional ``keras.activations.silu`` is represented here by a
# parameterless ``Activation`` layer over the same op, which fills the
# ``act1`` role the ConvStem has.

#: Non-default conv knobs the models are built with, so a dropped one is seen.
_CLIFFORD_KERNEL_REG = 1e-4
_CLIFFORD_BIAS_REG = 3e-5
_CLIFFORD_BIAS_INIT = 0.25
#: The model's own ``kernel_initializer``. Every pre-migration stem conv took
#: the model's (default ``glorot_uniform``), so a non-default one is passed to
#: the models below: with the default, a stem that dropped it and fell back to
#: ConvStem's own ``glorot_uniform`` default would be indistinguishable.
_CLIFFORD_KERNEL_INIT_STDDEV = 0.02


def _clifford_family_original(
    width: int, kernel: int, strides: int, two_conv: bool, use_bias: bool
) -> Original:
    """The pre-migration CliffordNet / Hierarchical stem, as literals."""

    def conv(filters: int, conv_strides: int, bias: bool, name: str):
        return keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=conv_strides,
            padding="same",
            use_bias=bias,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=_CLIFFORD_KERNEL_INIT_STDDEV
            ),
            bias_initializer=keras.initializers.Constant(_CLIFFORD_BIAS_INIT),
            kernel_regularizer=keras.regularizers.L2(_CLIFFORD_KERNEL_REG),
            bias_regularizer=keras.regularizers.L2(_CLIFFORD_BIAS_REG),
            name=name,
        )

    def bn(name: str):
        return keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-3, name=name
        )

    if two_conv:
        conv1 = conv(width // 2, strides, False, "stem_conv1")
        bn1 = bn("stem_bn1")
        silu = keras.layers.Activation(keras.activations.silu, name="stem_silu")
        conv2 = conv(width, strides, False, "stem_conv2")
        norm = bn("stem_norm")

        def forward(x, training):
            x = silu(bn1(conv1(x), training=training))
            x = conv2(x)
            return norm(x, training=training)

        return Original(
            sublayers={
                "conv1": conv1,
                "norm1": bn1,
                "act1": silu,
                "conv2": conv2,
                "norm2": norm,
            },
            forward=forward,
        )

    stem_conv = conv(width, strides, use_bias, "stem_conv")
    norm = bn("stem_norm")

    def forward_single(x, training):
        return norm(stem_conv(x), training=training)

    return Original(
        sublayers={"conv1": stem_conv, "norm1": norm}, forward=forward_single
    )


def _clifford_model(patch_size: int, use_bias: bool = True) -> CliffordNet:
    return CliffordNet(
        num_classes=10,
        channels=16,
        depth=1,
        patch_size=patch_size,
        use_bias=use_bias,
        kernel_regularizer=keras.regularizers.L2(_CLIFFORD_KERNEL_REG),
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=_CLIFFORD_KERNEL_INIT_STDDEV
        ),
        bias_initializer=keras.initializers.Constant(_CLIFFORD_BIAS_INIT),
        bias_regularizer=keras.regularizers.L2(_CLIFFORD_BIAS_REG),
    )


def _hierarchical_model(patch_size: int, use_bias: bool = True):
    return HierarchicalCliffordNet(
        num_classes=10,
        stage_depths=[1, 1],
        stage_dims=[16, 32],
        patch_size=patch_size,
        use_bias=use_bias,
        kernel_regularizer=keras.regularizers.L2(_CLIFFORD_KERNEL_REG),
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=_CLIFFORD_KERNEL_INIT_STDDEV
        ),
        bias_initializer=keras.initializers.Constant(_CLIFFORD_BIAS_INIT),
        bias_regularizer=keras.regularizers.L2(_CLIFFORD_BIAS_REG),
    )


CLIFFORD_P1 = StemCase(
    case_id="cliffordnet-patch1",
    make_model=lambda: _clifford_model(1),
    make_original=lambda: _clifford_family_original(16, 3, 1, True, False),
    input_shape=(16, 16, 3),
    expected_output_shape=(16, 16, 16),
)
CLIFFORD_P2 = StemCase(
    case_id="cliffordnet-patch2",
    make_model=lambda: _clifford_model(2),
    make_original=lambda: _clifford_family_original(16, 3, 2, False, True),
    input_shape=(16, 16, 3),
    expected_output_shape=(8, 8, 16),
)
CLIFFORD_P4 = StemCase(
    case_id="cliffordnet-patch4",
    make_model=lambda: _clifford_model(4),
    make_original=lambda: _clifford_family_original(16, 3, 2, True, False),
    input_shape=(16, 16, 3),
    expected_output_shape=(4, 4, 16),
)
# The "other" branch: one conv whose kernel and stride both equal patch_size,
# 16 -> ceil(16 / 3) = 6 under ``same`` padding.
CLIFFORD_OTHER = StemCase(
    case_id="cliffordnet-patch3-kernel-eq-stride",
    make_model=lambda: _clifford_model(3),
    make_original=lambda: _clifford_family_original(16, 3, 3, False, True),
    input_shape=(16, 16, 3),
    expected_output_shape=(6, 6, 16),
)
CLIFFORD_P2_NOBIAS = StemCase(
    case_id="cliffordnet-patch2-no-bias",
    make_model=lambda: _clifford_model(2, use_bias=False),
    make_original=lambda: _clifford_family_original(16, 3, 2, False, False),
    input_shape=(16, 16, 3),
    expected_output_shape=(8, 8, 16),
)
HIER_P1 = StemCase(
    case_id="hierarchical-patch1",
    make_model=lambda: _hierarchical_model(1),
    make_original=lambda: _clifford_family_original(16, 3, 1, True, False),
    input_shape=(16, 16, 3),
    expected_output_shape=(16, 16, 16),
)
HIER_P2 = StemCase(
    case_id="hierarchical-patch2",
    make_model=lambda: _hierarchical_model(2),
    make_original=lambda: _clifford_family_original(16, 3, 2, False, True),
    input_shape=(16, 16, 3),
    expected_output_shape=(8, 8, 16),
)

# ---------------------------------------------------------------------
# UcaEncoder (MambaLCT)
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/mambalct/uca_encoder.py``
# lines 128-134: ``Conv2D(dims[0], kernel 4, strides 4, padding "valid",
# use_bias, name="stem")`` then ``create_normalization_layer(normalization_type,
# epsilon=norm_epsilon, name="stem_norm")`` and ``call()`` ran conv then norm
# ONCE, with no activation. Default initializers and no regularizer. Built here
# with a non-default epsilon so a dropped ``norm_epsilon`` is seen.
_UCA_EPSILON = 1e-3


def _uca_original(use_bias: bool = True) -> Original:
    conv = keras.layers.Conv2D(
        filters=16,
        kernel_size=4,
        strides=4,
        padding="valid",
        use_bias=use_bias,
        name="stem",
    )
    norm = create_normalization_layer(
        "layer_norm", epsilon=_UCA_EPSILON, name="stem_norm"
    )

    def forward(x, training):
        return norm(conv(x), training=training)

    return Original(sublayers={"conv1": conv, "norm1": norm}, forward=forward)


def _uca_model(use_bias: bool = True) -> UcaEncoder:
    return UcaEncoder(
        stage_dims=[16, 32],
        stage_depths=[1, 1],
        num_heads=[2, 4],
        normalization_type="layer_norm",
        norm_epsilon=_UCA_EPSILON,
        use_bias=use_bias,
    )


UCA_ENCODER = StemCase(
    case_id="uca-encoder",
    make_model=_uca_model,
    make_original=_uca_original,
    input_shape=(16, 16, 3),
    expected_output_shape=(4, 4, 16),
)
UCA_ENCODER_NOBIAS = StemCase(
    case_id="uca-encoder-no-bias",
    make_model=lambda: _uca_model(use_bias=False),
    make_original=lambda: _uca_original(use_bias=False),
    input_shape=(16, 16, 3),
    expected_output_shape=(4, 4, 16),
)

# ---------------------------------------------------------------------
# ConvUNextStem
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/convunext/model.py``
# ``ConvUNextStem.build``: ``Conv2D(filters, kernel, padding "same", use_bias,
# he_normal, kernel regularizer, name="stem_conv")`` -> ``create_normalization_layer(
# stem_normalization, name="stem_norm")`` (no kwargs, epsilon 1e-6) ->
# ``keras.layers.Activation(substitute_exact(activation))`` which is ALWAYS built,
# even for ``"linear"``. ``call()`` ran ``self.norm(x)`` without ``training``; Keras 3
# hands a child called without ``training`` the training flag of the enclosing
# call (measured: the moving mean of a BatchNormalization moved under
# ``outer(x, training=True)``), so ``training=training`` below is what the
# original ran. The exact forms (D-277) are the library's ``exact_gelu`` /
# ``exact_leaky_relu`` callables that ``substitute_exact`` returns for the
# strings ``"gelu"`` / ``"leaky_relu"``. A ``"linear"`` original carries an
# identity ``Activation`` with no weights: it stays in ``forward`` but, as
# ConvStem builds no layer for ``"linear"``, it is left out of ``sublayers``
# (the role-set comparison then pins that ConvStem really has none).
_CONVUNEXT_FILTERS = 16
_CONVUNEXT_KERNEL = 5
_CONVUNEXT_KERNEL_REG = 1e-4
#: Not ConvUNextStem's default (``he_normal``) and not ConvStem's
#: (``glorot_uniform``): the original took the layer's ``kernel_initializer``.
_CONVUNEXT_KERNEL_INIT = "he_uniform"


def _convunext_original(
    norm_type: str, activation, use_bias: bool, identity_activation: bool = False
) -> Original:
    conv = keras.layers.Conv2D(
        filters=_CONVUNEXT_FILTERS,
        kernel_size=_CONVUNEXT_KERNEL,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=_CONVUNEXT_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_CONVUNEXT_KERNEL_REG),
        name="stem_conv",
    )
    norm = create_normalization_layer(norm_type, name="stem_norm")
    act = keras.layers.Activation(activation, name="stem_activation")

    def forward(x, training):
        x = conv(x)
        x = norm(x, training=training)
        return act(x)

    sublayers = {"conv1": conv, "norm1": norm}
    if not identity_activation:
        sublayers["act1"] = act
    return Original(sublayers=sublayers, forward=forward)


def _convunext_model(
    norm_type: str, activation, use_bias: bool
) -> ConvUNextStem:
    """The layer under test: its ``.stem`` is the ``ConvStem``."""
    return ConvUNextStem(
        filters=_CONVUNEXT_FILTERS,
        kernel_size=_CONVUNEXT_KERNEL,
        activation=activation,
        use_bias=use_bias,
        stem_normalization=norm_type,
        kernel_initializer=_CONVUNEXT_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_CONVUNEXT_KERNEL_REG),
    )


def _convunext_case(
    case_id: str,
    norm_type: str,
    make_activation: Callable[[], object],
    make_original_activation: Callable[[], object],
    use_bias: bool,
    identity_activation: bool = False,
) -> StemCase:
    """One case; both activations are zero-argument factories so a Layer-instance
    activation is fresh for the model and for the original."""
    return StemCase(
        case_id=case_id,
        make_model=lambda: _convunext_model(norm_type, make_activation(), use_bias),
        make_original=lambda: _convunext_original(
            norm_type, make_original_activation(), use_bias, identity_activation
        ),
        input_shape=(16, 16, 3),
        expected_output_shape=(16, 16, _CONVUNEXT_FILTERS),
    )


CONVUNEXT_GELU_GRN_NOBIAS = _convunext_case(
    "convunext-gelu-grn-no-bias",
    "global_response_norm",
    lambda: "gelu",
    lambda: exact_gelu,
    False,
)
CONVUNEXT_GELU_LAYERNORM = _convunext_case(
    "convunext-gelu-layer-norm-bias",
    "layer_norm",
    lambda: "gelu",
    lambda: exact_gelu,
    True,
)
CONVUNEXT_LEAKY_RELU_STRING = _convunext_case(
    "convunext-leaky-relu-string",
    "layer_norm",
    lambda: "leaky_relu",
    lambda: exact_leaky_relu,
    True,
)
CONVUNEXT_LEAKY_RELU_INSTANCE = _convunext_case(
    "convunext-leaky-relu-instance",
    "global_response_norm",
    lambda: keras.layers.LeakyReLU(negative_slope=0.1),
    lambda: keras.layers.LeakyReLU(negative_slope=0.1),
    False,
)
CONVUNEXT_LINEAR = _convunext_case(
    "convunext-linear",
    "layer_norm",
    lambda: "linear",
    lambda: "linear",
    True,
    identity_activation=True,
)
CONVUNEXT_BATCH_NORM = _convunext_case(
    "convunext-batch-norm",
    "batch_norm",
    lambda: "gelu",
    lambda: exact_gelu,
    True,
)

# ---------------------------------------------------------------------
# ConvNeXt V1, ConvNeXt V2 and RADConvNet
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/convnext/convnext_v1.py``
# (``_build_stem``), ``convnext_v2.py`` and ``rad_convnet/model.py``. A Conv2D
# patchify (kernel = stride = ``strides``; padding ``"same"`` at stride 1 and
# ``"valid"`` above it, the D-125 rule) followed by a ``LayerNormalization``
# with epsilon 1e-6, ``center=use_bias``, ``scale=True``. No activation, no
# pool. The ConvNeXt convs also took the model's ``kernel_regularizer``; RAD's
# took none because RADConvNet had no such argument. It has one now, so the
# hand-rolled original of all three carries the regularizer.
# LayerNormalization ignores ``training``, so the original calling it without
# one is not a difference.

_CONVNEXT_DIMS = [8, 16]
_CONVNEXT_KERNEL_REG = 1e-4


#: ConvNeXt V1/V2: ``STEM_INITIALIZER`` was this dict (the reference's
#: ``trunc_normal_(std=.02)``). RADConvNet's was the bare string
#: ``"truncated_normal"``, Keras' default stddev 0.05.
_CONVNEXT_STEM_INITIALIZER = {
    "class_name": "TruncatedNormal",
    "config": {"mean": 0.0, "stddev": 0.02},
}
_RAD_STEM_INITIALIZER = "truncated_normal"


def _convnext_family_original(
    strides: int, use_bias: bool, kernel_initializer
) -> Original:
    """The pre-migration ConvNeXt / RAD patchify stem, as literals."""
    conv = keras.layers.Conv2D(
        filters=_CONVNEXT_DIMS[0],
        kernel_size=strides,
        strides=strides,
        padding="same" if strides == 1 else "valid",
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(_CONVNEXT_KERNEL_REG),
        name="stem_conv",
    )
    norm = keras.layers.LayerNormalization(
        epsilon=1e-6, center=use_bias, scale=True, name="stem_norm"
    )

    def forward(x, training):
        return norm(conv(x))

    return Original(sublayers={"conv1": conv, "norm1": norm}, forward=forward)


def _convnext_family_model(
    model_class, strides: int, use_bias: bool
) -> keras.Model:
    kwargs = dict(
        depths=[1, 1],
        dims=_CONVNEXT_DIMS,
        num_classes=4,
        include_top=False,
        strides=strides,
        use_bias=use_bias,
    )
    kwargs["kernel_regularizer"] = keras.regularizers.L2(_CONVNEXT_KERNEL_REG)
    return model_class(**kwargs)


def _convnext_family_case(
    case_id: str,
    model_class,
    strides: int,
    use_bias: bool,
    spatial: int,
    output_spatial: int,
) -> StemCase:
    return StemCase(
        case_id=case_id,
        make_model=lambda: _convnext_family_model(model_class, strides, use_bias),
        make_original=lambda: _convnext_family_original(
            strides,
            use_bias,
            _RAD_STEM_INITIALIZER if model_class is RADConvNet else _CONVNEXT_STEM_INITIALIZER,
        ),
        input_shape=(spatial, spatial, 3),
        expected_output_shape=(output_spatial, output_spatial, _CONVNEXT_DIMS[0]),
    )


# Stride 4, valid; divisible input 32 -> 8, and 30 -> 7 where "same" would give 8.
CONVNEXT_V1_VALID = _convnext_family_case(
    "convnext-v1-stride4-valid", ConvNeXtV1, 4, True, 32, 8
)
CONVNEXT_V1_VALID_NONDIVISIBLE = _convnext_family_case(
    "convnext-v1-stride4-valid-nondivisible", ConvNeXtV1, 4, True, 30, 7
)
CONVNEXT_V1_SAME_NOBIAS = _convnext_family_case(
    "convnext-v1-stride1-same-no-bias", ConvNeXtV1, 1, False, 8, 8
)
CONVNEXT_V2_VALID = _convnext_family_case(
    "convnext-v2-stride4-valid", ConvNeXtV2, 4, True, 32, 8
)
CONVNEXT_V2_VALID_NONDIVISIBLE_NOBIAS = _convnext_family_case(
    "convnext-v2-stride4-valid-nondivisible-no-bias", ConvNeXtV2, 4, False, 30, 7
)
CONVNEXT_V2_SAME = _convnext_family_case(
    "convnext-v2-stride1-same", ConvNeXtV2, 1, True, 8, 8
)
RAD_VALID = _convnext_family_case("rad-stride4-valid", RADConvNet, 4, True, 32, 8)
RAD_VALID_NONDIVISIBLE_NOBIAS = _convnext_family_case(
    "rad-stride4-valid-nondivisible-no-bias", RADConvNet, 4, False, 30, 7
)
RAD_SAME = _convnext_family_case("rad-stride1-same", RADConvNet, 1, True, 8, 8)

# ---------------------------------------------------------------------
# MobileNet V1 to V4
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/mobilenet/mobilenet_v3.py``
# (``__init__``) and ``mobilenet_v4.py`` (``_build_stem``); V1 and V2 are
# ``git show 0a908d16e:src/dl_techniques/models/vision/mobilenet/mobilenet_v1.py``
# (``_build_layers``) and ``mobilenet_v2.py`` (``_build_model_layers``). A 3x3 stride-2
# ``same`` bias-free Conv2D with the model's L2 kernel regularizer, then
# ``BatchNormalization(momentum=0.99, epsilon=1e-3)`` (TF Model Garden's
# MobileNet backbone, D-111 of plan-2026-08-22T035419-a11304c8), then
# ``HardSwish`` (V3), ``ReLU`` (V1, V4) or ``ReLU(max_value=6)`` (V2); no pool.
# The momentum and epsilon are written as literals here, not read from
# ``mobilenet.common``. The V3 stem is 16 channels at every width, the V4 stem
# is ``dims[0]``, the V1 stem is ``int(32 * alpha)`` and the V2 stem
# ``make_divisible(32 * alpha)``. Each activation was a separate layer in the
# original, and is one in ``ConvStem``, so the op sequence is identical and
# the comparison stays at ``rtol=0, atol=0``.

_MOBILENET_KERNEL_REG = 1e-4
_MOBILENET_V4_STEM_FILTERS = 24
#: The models' ``kernel_initializer`` knobs, each built with a value that is
#: neither the model's default (V3 ``glorot_uniform``, V4 ``he_normal``) nor
#: ConvStem's (``glorot_uniform``): V3's default IS ConvStem's, so with it a
#: stem that dropped the argument was indistinguishable from a correct one
#: (pass-2 WARNING 1, reproduced with a real mutation).
_MOBILENET_V3_KERNEL_INIT = "he_uniform"
_MOBILENET_V4_KERNEL_INIT = "lecun_normal"
#: V1 and V2 default to ``he_normal``; these differ from it and from ConvStem's.
_MOBILENET_V1_KERNEL_INIT = "he_uniform"
_MOBILENET_V2_KERNEL_INIT = "lecun_normal"
#: A width at which the two width rules disagree: V1 truncates ``32 * 0.6 = 19.2``
#: to 19, V2 rounds it to a multiple of 8 (24), so a stem using the other model's
#: rule is seen.
_MOBILENET_ALPHA = 0.6
_MOBILENET_V1_STEM_FILTERS = 19
_MOBILENET_V2_STEM_FILTERS = 24
#: V2's stem is clipped at 6; the activation input must reach it for the clip to
#: be tested (see ``StemCase.input_scale``).
_MOBILENET_V2_INPUT_SCALE = 50.0


def _mobilenet_original(
    filters: int, kernel_initializer, activation_layer
) -> Original:
    """The pre-migration MobileNet stem (V1 to V4), as literals."""
    conv = keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.L2(_MOBILENET_KERNEL_REG),
        name="stem_conv",
    )
    norm = keras.layers.BatchNormalization(
        momentum=0.99, epsilon=1e-3, name="stem_bn"
    )

    def forward(x, training):
        x = conv(x)
        x = norm(x, training=training)
        return activation_layer(x)

    return Original(
        sublayers={"conv1": conv, "norm1": norm, "act1": activation_layer},
        forward=forward,
    )


def _mobilenet_v1_model() -> MobileNetV1:
    return MobileNetV1(
        num_classes=4,
        width_multiplier=_MOBILENET_ALPHA,
        include_top=False,
        input_shape=(32, 32, 3),
        kernel_regularizer=keras.regularizers.L2(_MOBILENET_KERNEL_REG),
        kernel_initializer=_MOBILENET_V1_KERNEL_INIT,
    )


def _mobilenet_v2_model() -> MobileNetV2:
    return MobileNetV2(
        num_classes=4,
        width_multiplier=_MOBILENET_ALPHA,
        include_top=False,
        input_shape=(32, 32, 3),
        kernel_regularizer=keras.regularizers.L2(_MOBILENET_KERNEL_REG),
        kernel_initializer=_MOBILENET_V2_KERNEL_INIT,
    )


def _mobilenet_v3_model() -> MobileNetV3:
    return MobileNetV3(
        variant="small",
        num_classes=4,
        include_top=False,
        input_shape=(32, 32, 3),
        kernel_regularizer=keras.regularizers.L2(_MOBILENET_KERNEL_REG),
        kernel_initializer=_MOBILENET_V3_KERNEL_INIT,
    )


def _mobilenet_v4_model() -> MobileNetV4:
    return MobileNetV4(
        num_classes=4,
        depths=[1, 1],
        dims=[_MOBILENET_V4_STEM_FILTERS, 32],
        block_types=["IB", "IB"],
        strides=[1, 2],
        include_top=False,
        input_shape=(32, 32, 3),
        kernel_regularizer=keras.regularizers.L2(_MOBILENET_KERNEL_REG),
        kernel_initializer=_MOBILENET_V4_KERNEL_INIT,
    )


MOBILENET_V1 = StemCase(
    case_id="mobilenet-v1-relu",
    make_model=_mobilenet_v1_model,
    make_original=lambda: _mobilenet_original(
        _MOBILENET_V1_STEM_FILTERS,
        _MOBILENET_V1_KERNEL_INIT,
        keras.layers.ReLU(name="conv1_relu"),
    ),
    input_shape=(32, 32, 3),
    expected_output_shape=(16, 16, _MOBILENET_V1_STEM_FILTERS),
)
MOBILENET_V2 = StemCase(
    case_id="mobilenet-v2-relu6",
    make_model=_mobilenet_v2_model,
    make_original=lambda: _mobilenet_original(
        _MOBILENET_V2_STEM_FILTERS,
        _MOBILENET_V2_KERNEL_INIT,
        keras.layers.ReLU(max_value=6, name="conv1_relu6"),
    ),
    input_shape=(32, 32, 3),
    expected_output_shape=(16, 16, _MOBILENET_V2_STEM_FILTERS),
    input_scale=_MOBILENET_V2_INPUT_SCALE,
)
MOBILENET_V3 = StemCase(
    case_id="mobilenet-v3-hard-swish",
    make_model=_mobilenet_v3_model,
    make_original=lambda: _mobilenet_original(
        16, _MOBILENET_V3_KERNEL_INIT, HardSwish(name="stem_hard_swish")
    ),
    input_shape=(32, 32, 3),
    expected_output_shape=(16, 16, 16),
)
MOBILENET_V4 = StemCase(
    case_id="mobilenet-v4-relu",
    make_model=_mobilenet_v4_model,
    make_original=lambda: _mobilenet_original(
        _MOBILENET_V4_STEM_FILTERS,
        _MOBILENET_V4_KERNEL_INIT,
        keras.layers.ReLU(name="stem_relu"),
    ),
    input_shape=(32, 32, 3),
    expected_output_shape=(16, 16, _MOBILENET_V4_STEM_FILTERS),
)

# ---------------------------------------------------------------------
# SqueezeNet V1 and SqueezeNoduleNet V2 (2D path, and the 3D path of V2)
# ---------------------------------------------------------------------

# Original: the files before they were migrated (``git show 7cea79771^:<path>``;
# ``models/vision/squeezenet/squeezenet_v1.py`` and ``squeezenet_v2.py``,
# ``_build_stem``). ``Conv2D(conv1_filters, conv1_kernel, strides=conv1_stride,
# activation="relu", padding="same" if conv1_stride == 1 else "valid",
# kernel_regularizer, kernel_initializer="lecun_uniform")`` with the default
# bias, NO normalization, then (pool index 1 only) ``MaxPooling2D(3, strides=2,
# padding="valid")``. The ReLU is FUSED into the conv (declared in
# ``Original.fused``); ``ConvStem`` runs the same ReLU as a layer of its own
# after the same conv, so the values are identical and the comparison stays at
# ``rtol=0, atol=0``.
#
# The 3D stem of ``SqueezeNoduleNetV2`` (``_build_stem_3d`` of the same revision,
# before it moved onto ``ConvStem(rank=3)``) is the same list with ``Conv3D`` and
# ``MaxPooling3D`` (``name="conv1"`` and ``"maxpool1"``), compared at the same
# ``rtol=0, atol=0``.

_SQUEEZE_KERNEL_REG = 1e-4
#: The models' ``kernel_initializer`` knob initializes the Fire blocks only; the
#: stem uses ``STEM_INITIALIZER`` (``lecun_uniform``). The models are built with
#: a knob value that differs from both, so a stem that started reading the knob
#: is seen.
_SQUEEZE_BLOCK_KERNEL_INIT = "he_uniform"


def _squeezenet_original(
    filters: int, kernel: int, stride: int, pool: bool, rank: int = 2
) -> Original:
    """The pre-migration SqueezeNet V1 / V2 stem, as literals (``rank`` 3: the 3D stem)."""
    conv_class = {2: keras.layers.Conv2D, 3: keras.layers.Conv3D}[rank]
    pool_class = {2: keras.layers.MaxPooling2D, 3: keras.layers.MaxPooling3D}[rank]
    conv = conv_class(
        filters=filters,
        kernel_size=kernel,
        strides=stride,
        activation="relu",
        padding="same" if stride == 1 else "valid",
        kernel_regularizer=keras.regularizers.L2(_SQUEEZE_KERNEL_REG),
        kernel_initializer="lecun_uniform",
        name="conv1",
    )
    maxpool = (
        pool_class(pool_size=3, strides=2, padding="valid", name="maxpool1")
        if pool
        else None
    )

    def forward(x, training):
        x = conv(x)
        return x if maxpool is None else maxpool(x)

    sublayers = {"conv1": conv}
    if maxpool is not None:
        sublayers["pool"] = maxpool
    return Original(sublayers=sublayers, forward=forward, fused=("act1",))


def _squeezenet_model(cls, variant: str, input_shape, **changes):
    """``cls`` built headless on ``variant``, with ``changes`` applied to the variant's table.

    The shipped variants have one pool schedule and one stride each, and the models take a
    variant by NAME. A case that needs another schedule (no stem pool) or stride (the ``same``
    branch) registers a copy of the variant under a temporary key, builds on it and removes it.
    """
    knobs = dict(
        num_classes=4,
        include_top=False,
        input_shape=input_shape,
        kernel_initializer=_SQUEEZE_BLOCK_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_SQUEEZE_KERNEL_REG),
    )
    if not changes:
        return cls(variant=variant, **knobs)
    entry = {**cls.MODEL_VARIANTS[variant], **changes}
    with mock.patch.dict(cls.MODEL_VARIANTS, {"eq_case": entry}):
        return cls(variant="eq_case", **knobs)


def _squeezenet_v1_model(variant: str, pool_indices=None, **changes) -> SqueezeNetV1:
    if pool_indices is not None:
        changes["pool_indices"] = pool_indices
    return _squeezenet_model(SqueezeNetV1, variant, (64, 64, 3), **changes)


def _squeezenet_v2_model(variant: str, pool_indices=None) -> SqueezeNoduleNetV2:
    changes = {} if pool_indices is None else {"pool_indices": pool_indices}
    return _squeezenet_model(SqueezeNoduleNetV2, variant, (64, 64, 1), **changes)


def _squeezenet_v2_3d_model(**changes) -> SqueezeNoduleNetV2:
    return _squeezenet_model(SqueezeNoduleNetV2, "v2_3d", (36, 36, 36, 1), **changes)


# 64 px: V1.0 conv k7/s2 valid -> 29, pool -> 14; V1.1 conv k3/s2 valid -> 31,
# pool -> 15; no pool keeps 29 / 31.
SQUEEZENET_V1_0 = StemCase(
    case_id="squeezenet-v1.0-pool",
    make_model=lambda: _squeezenet_v1_model("1.0"),
    make_original=lambda: _squeezenet_original(96, 7, 2, True),
    input_shape=(64, 64, 3),
    expected_output_shape=(14, 14, 96),
)
SQUEEZENET_V1_0_NOPOOL = StemCase(
    case_id="squeezenet-v1.0-no-pool",
    make_model=lambda: _squeezenet_v1_model("1.0", [4, 8]),
    make_original=lambda: _squeezenet_original(96, 7, 2, False),
    input_shape=(64, 64, 3),
    expected_output_shape=(29, 29, 96),
)
SQUEEZENET_V1_1 = StemCase(
    case_id="squeezenet-v1.1-pool",
    make_model=lambda: _squeezenet_v1_model("1.1"),
    make_original=lambda: _squeezenet_original(64, 3, 2, True),
    input_shape=(64, 64, 3),
    expected_output_shape=(15, 15, 64),
)
SQUEEZENET_V1_1_NOPOOL = StemCase(
    case_id="squeezenet-v1.1-no-pool",
    make_model=lambda: _squeezenet_v1_model("1.1", [3, 5]),
    make_original=lambda: _squeezenet_original(64, 3, 2, False),
    input_shape=(64, 64, 3),
    expected_output_shape=(31, 31, 64),
)
# Stride 1 takes the ``same`` branch, which no shipped variant does.
SQUEEZENET_V1_STRIDE1_SAME = StemCase(
    case_id="squeezenet-v1-stride1-same-pool",
    make_model=lambda: _squeezenet_v1_model("1.1", conv1_stride=1),
    make_original=lambda: _squeezenet_original(64, 3, 1, True),
    input_shape=(64, 64, 3),
    expected_output_shape=(31, 31, 64),
)
SQUEEZENET_V2 = StemCase(
    case_id="squeezenodule-v2-2d-pool",
    make_model=lambda: _squeezenet_v2_model("v2"),
    make_original=lambda: _squeezenet_original(96, 7, 2, True),
    input_shape=(64, 64, 1),
    expected_output_shape=(14, 14, 96),
)
SQUEEZENET_V2_NOPOOL = StemCase(
    case_id="squeezenodule-v2-2d-no-pool",
    make_model=lambda: _squeezenet_v2_model("v1", [4, 8]),
    make_original=lambda: _squeezenet_original(96, 7, 2, False),
    input_shape=(64, 64, 1),
    expected_output_shape=(29, 29, 96),
)
# 36 voxels, the smallest legal extent: conv k7/s2 valid -> 15, pool -> 7; no pool keeps 15;
# stride 1 takes the ``same`` branch (36) and the pool gives 17.
SQUEEZENET_V2_3D = StemCase(
    case_id="squeezenodule-v2-3d-pool",
    make_model=lambda: _squeezenet_v2_3d_model(),
    make_original=lambda: _squeezenet_original(96, 7, 2, True, rank=3),
    input_shape=(36, 36, 36, 1),
    expected_output_shape=(7, 7, 7, 96),
)
SQUEEZENET_V2_3D_NOPOOL = StemCase(
    case_id="squeezenodule-v2-3d-no-pool",
    make_model=lambda: _squeezenet_v2_3d_model(pool_indices=[4, 8]),
    make_original=lambda: _squeezenet_original(96, 7, 2, False, rank=3),
    input_shape=(36, 36, 36, 1),
    expected_output_shape=(15, 15, 15, 96),
)
SQUEEZENET_V2_3D_STRIDE1_SAME = StemCase(
    case_id="squeezenodule-v2-3d-stride1-same-pool",
    make_model=lambda: _squeezenet_v2_3d_model(conv1_stride=1),
    make_original=lambda: _squeezenet_original(96, 7, 1, True, rank=3),
    input_shape=(36, 36, 36, 1),
    expected_output_shape=(17, 17, 17, 96),
)

# ---------------------------------------------------------------------
# VAE encoder stem
# ---------------------------------------------------------------------

# Original: ``git show cad74076e^:src/dl_techniques/models/vision/vae/model.py``
# (``_build_encoder``), the functional initial stem: ``Conv2D(filters[0], 3,
# strides=1, padding="same", use_bias, kernel_initializer, kernel_regularizer)``
# -> (``use_batch_norm`` only) a bare ``BatchNormalization(center=use_bias)``,
# so Keras' defaults momentum 0.99, epsilon 1e-3, scale=True -> ``Activation``
# over ``substitute_exact(activation)`` (D-277): the exact callable for
# ``"gelu"`` / ``"leaky_relu"``, the string itself for any other name. The
# stem is a layer of the functional encoder graph, reached through
# ``vae.get_layer("encoder_stem")``.
_VAE_FILTERS = 8
_VAE_KERNEL_REG = 1e-4
#: Not the VAE's default (``he_normal``) and not ConvStem's (``glorot_uniform``).
_VAE_KERNEL_INIT = "lecun_normal"


def _vae_original(
    use_batch_norm: bool, use_bias: bool, activation, identity_activation: bool = False
) -> Original:
    conv = keras.layers.Conv2D(
        filters=_VAE_FILTERS,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=_VAE_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_VAE_KERNEL_REG),
        name="encoder_stem_conv",
    )
    norm = (
        keras.layers.BatchNormalization(
            momentum=0.99, epsilon=1e-3, center=use_bias, name="encoder_stem_bn"
        )
        if use_batch_norm
        else None
    )
    act = keras.layers.Activation(activation, name="encoder_stem_activation")

    def forward(x, training):
        x = conv(x)
        if norm is not None:
            x = norm(x, training=training)
        return act(x)

    sublayers = {"conv1": conv}
    if norm is not None:
        sublayers["norm1"] = norm
    if not identity_activation:
        sublayers["act1"] = act
    return Original(sublayers=sublayers, forward=forward)


def _vae_model(use_batch_norm: bool, use_bias: bool, activation) -> VAE:
    return VAE(
        latent_dim=8,
        input_shape=(32, 32, 3),
        depths=2,
        steps_per_depth=1,
        filters=[_VAE_FILTERS, 16],
        kernel_initializer=_VAE_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_VAE_KERNEL_REG),
        use_batch_norm=use_batch_norm,
        use_bias=use_bias,
        activation=activation,
    )


def _vae_stem_of(model: VAE) -> ConvStem:
    return model.get_layer("encoder_stem")


def _vae_case(
    case_id: str,
    use_batch_norm: bool,
    use_bias: bool,
    make_activation: Callable[[], object],
    make_original_activation: Callable[[], object],
    identity_activation: bool = False,
) -> StemCase:
    """One case; both activations are zero-argument factories so a Layer-instance
    activation is fresh for the model and for the original."""
    return StemCase(
        case_id=case_id,
        make_model=lambda: _vae_model(use_batch_norm, use_bias, make_activation()),
        make_original=lambda: _vae_original(
            use_batch_norm, use_bias, make_original_activation(), identity_activation
        ),
        input_shape=(32, 32, 3),
        expected_output_shape=(32, 32, _VAE_FILTERS),
        stem_of=_vae_stem_of,
    )


VAE_BN_BIAS = _vae_case(
    "vae-leaky-relu-bn-bias", True, True, lambda: "leaky_relu", lambda: exact_leaky_relu
)
VAE_BN_NOBIAS = _vae_case(
    "vae-leaky-relu-bn-no-bias", True, False, lambda: "leaky_relu", lambda: exact_leaky_relu
)
VAE_NOBN_BIAS = _vae_case(
    "vae-leaky-relu-no-bn-bias", False, True, lambda: "leaky_relu", lambda: exact_leaky_relu
)
VAE_NOBN_NOBIAS = _vae_case(
    "vae-leaky-relu-no-bn-no-bias",
    False,
    False,
    lambda: "leaky_relu",
    lambda: exact_leaky_relu,
)
VAE_GELU = _vae_case(
    "vae-gelu-bn-bias", True, True, lambda: "gelu", lambda: exact_gelu
)
VAE_RELU_STRING = _vae_case(
    "vae-relu-string-bn-bias", True, True, lambda: "relu", lambda: "relu"
)
VAE_LEAKY_RELU_INSTANCE = _vae_case(
    "vae-leaky-relu-instance",
    True,
    True,
    lambda: keras.layers.LeakyReLU(negative_slope=0.1),
    lambda: keras.layers.LeakyReLU(negative_slope=0.1),
)
VAE_LINEAR = _vae_case(
    "vae-linear",
    True,
    True,
    lambda: "linear",
    lambda: "linear",
    identity_activation=True,
)

# ---------------------------------------------------------------------
# CliffordRBFProtoNet patch-embedding stem
# ---------------------------------------------------------------------

# Original: ``src/dl_techniques/models/vision/rbf_protonet/model.py`` at the
# parent of the Step 12 commit (``_build_stem``): ``Conv2D(channels, 3,
# strides=patch_size, padding="same", use_bias=True, kernel_regularizer,
# name="stem_conv")`` with Keras' default glorot_uniform kernel and zeros bias,
# then a bare ``BatchNormalization(momentum=0.9)`` (Keras epsilon 1e-3), no
# activation, no pool. The ``# DECISION plan-2026-09-16T064227-b83f88ad/D-005``
# anchor in that file pins this exact pairing.
_RBF_CHANNELS = 8
_RBF_KERNEL_REG = 1e-4


def _rbf_clifford_original(patch_size: int, regularized: bool) -> Original:
    conv = keras.layers.Conv2D(
        filters=_RBF_CHANNELS,
        kernel_size=3,
        strides=patch_size,
        padding="same",
        use_bias=True,
        kernel_regularizer=(
            keras.regularizers.L2(_RBF_KERNEL_REG) if regularized else None
        ),
        name="stem_conv",
    )
    norm = keras.layers.BatchNormalization(momentum=0.9, name="stem_bn")

    def forward(x, training):
        return norm(conv(x), training=training)

    return Original(sublayers={"conv1": conv, "norm1": norm}, forward=forward)


def _rbf_clifford_model(patch_size: int, regularized: bool) -> CliffordRBFProtoNet:
    return CliffordRBFProtoNet(
        num_classes=4,
        channels=_RBF_CHANNELS,
        depth=1,
        feature_dim=_RBF_CHANNELS,
        patch_size=patch_size,
        kernel_regularizer=(
            keras.regularizers.L2(_RBF_KERNEL_REG) if regularized else None
        ),
    )


def _rbf_clifford_case(patch_size: int, regularized: bool) -> StemCase:
    side = -(-16 // patch_size)  # ceil under ``same`` padding
    return StemCase(
        case_id=f"rbf-clifford-patch{patch_size}-{'l2' if regularized else 'noreg'}",
        make_model=lambda: _rbf_clifford_model(patch_size, regularized),
        make_original=lambda: _rbf_clifford_original(patch_size, regularized),
        input_shape=(16, 16, 3),
        expected_output_shape=(side, side, _RBF_CHANNELS),
    )


RBF_CLIFFORD_P1 = _rbf_clifford_case(1, True)
RBF_CLIFFORD_P2 = _rbf_clifford_case(2, True)
RBF_CLIFFORD_P2_NOREG = _rbf_clifford_case(2, False)
RBF_CLIFFORD_P4 = _rbf_clifford_case(4, True)
# 16 -> ceil(16 / 3) = 6 under ``same`` padding.
RBF_CLIFFORD_P3 = _rbf_clifford_case(3, True)

# ---------------------------------------------------------------------
# CapsNet and CapsNetV2 convolution blocks
# ---------------------------------------------------------------------

# Original: ``src/dl_techniques/models/vision/capsnet/model.py`` and ``model_v2.py``
# at ``be92a74eb`` (``_build_feature_extraction`` / the legacy branch of
# ``_build_stem``): per ``conv_filters`` entry ``Conv2D(filters, 9 if i == 0 else 5,
# strides=1, padding="valid", use_bias=True, kernel_initializer=<one shared
# instance>, kernel_regularizer)``, a ``BatchNormalization()`` (Keras defaults:
# momentum 0.99, epsilon 1e-3) only when ``use_batch_norm``, and a ``ReLU()``. The
# paper has no batch normalization; 1e-3 is the pre-migration Keras default, not a
# reference value, and the norm factory's own default is 1e-6, so a block that did
# not pass its epsilon would differ with no shape symptom. A block is a feature
# trunk stage, not a patch stem: the CapsNet trunk is a list of single-convolution
# ConvStem blocks (``num_convs=2`` cannot reproduce it: one kernel size for both
# convolutions).
_CAPSNET_KERNEL_REG = 1e-4
#: Neither the models' default (``he_normal``) nor ConvStem's (``glorot_uniform``).
_CAPSNET_KERNEL_INIT = "lecun_normal"
_CAPSNET_FILTERS = 8


def _capsnet_block_original(kernel_size: int, batch_norm: bool) -> Original:
    conv = keras.layers.Conv2D(
        filters=_CAPSNET_FILTERS,
        kernel_size=kernel_size,
        strides=1,
        padding="valid",
        kernel_initializer=_CAPSNET_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_CAPSNET_KERNEL_REG),
        name="conv",
    )
    norm = (
        keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name="bn")
        if batch_norm
        else None
    )
    act = keras.layers.ReLU(name="relu")

    def forward(x, training):
        x = conv(x)
        if norm is not None:
            x = norm(x, training=training)
        return act(x)

    sublayers = {"conv1": conv, "act1": act}
    if norm is not None:
        sublayers["norm1"] = norm
    return Original(sublayers=sublayers, forward=forward)


def _capsnet_v1_model(filters=(_CAPSNET_FILTERS, _CAPSNET_FILTERS), batch_norm=True) -> CapsNet:
    return CapsNet(
        num_classes=4,
        conv_filters=filters,
        primary_capsules=4,
        input_shape=(28, 28, 1),
        reconstruction=False,
        use_batch_norm=batch_norm,
        kernel_initializer=_CAPSNET_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_CAPSNET_KERNEL_REG),
    )


def _capsnet_v2_model() -> CapsNetV2:
    return CapsNetV2(
        num_classes=4,
        input_shape=(28, 28, 1),
        stem_type="legacy",
        legacy_conv_filters=[_CAPSNET_FILTERS, _CAPSNET_FILTERS],
        primary_capsules=4,
        kernel_initializer=_CAPSNET_KERNEL_INIT,
        kernel_regularizer=keras.regularizers.L2(_CAPSNET_KERNEL_REG),
    )


def _capsnet_block(index: int):
    return lambda model: model.conv_blocks[index]


def _capsnet_case(
    case_id: str, make_model, index: int, batch_norm: bool = True
) -> StemCase:
    kernel = 9 if index == 0 else 5
    side = 28 if index == 0 else 28 - 8 - 4 * (index - 1)
    out = side - kernel + 1
    channels = 1 if index == 0 else _CAPSNET_FILTERS
    return StemCase(
        case_id=case_id,
        make_model=make_model,
        make_original=lambda: _capsnet_block_original(kernel, batch_norm),
        input_shape=(side, side, channels),
        expected_output_shape=(out, out, _CAPSNET_FILTERS),
        stem_of=_capsnet_block(index),
    )


CAPSNET_V1_BLOCK1 = _capsnet_case("capsnet-v1-block1-k9", _capsnet_v1_model, 0)
CAPSNET_V1_BLOCK2 = _capsnet_case("capsnet-v1-block2-k5", _capsnet_v1_model, 1)
CAPSNET_V1_BLOCK3 = _capsnet_case(
    "capsnet-v1-three-convs-block3-k5",
    lambda: _capsnet_v1_model(filters=(_CAPSNET_FILTERS,) * 3),
    2,
)
CAPSNET_V1_NOBN = _capsnet_case(
    "capsnet-v1-no-bn-block2-k5",
    lambda: _capsnet_v1_model(batch_norm=False),
    1,
    batch_norm=False,
)
CAPSNET_V2_BLOCK1 = _capsnet_case("capsnet-v2-legacy-block1-k9", _capsnet_v2_model, 0)
CAPSNET_V2_BLOCK2 = _capsnet_case("capsnet-v2-legacy-block2-k5", _capsnet_v2_model, 1)


# ---------------------------------------------------------------------
# DeepSORT appearance net (the mars-small128 trunk)
# ---------------------------------------------------------------------

# Original: ``DeepSortAppearanceNet.__init__`` and ``call`` before the migration
# (HEAD d86e9e1ad): two Conv2D(32, 3, same, bias) with a truncated-normal 1e-3
# kernel, zero bias and L2 1e-8, each followed by a BatchNormalization from the norm
# factory (momentum 0.999, epsilon 1e-3: the TF-slim values, not the factory's 1e-6)
# and a functional ``ops.elu`` (no layer of its own, so ``fused``), then
# MaxPooling2D(3, strides=2, same). The model has no initializer knob: the
# truncated normal is a literal of the stem call.
_DEEPSORT_STDDEV = 1e-3
_DEEPSORT_KERNEL_REG = 1e-8


def _deepsort_original() -> Original:
    def conv(name: str) -> keras.layers.Conv2D:
        return keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=_DEEPSORT_STDDEV),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.L2(_DEEPSORT_KERNEL_REG),
            name=name,
        )

    def norm(name: str) -> keras.layers.Layer:
        return create_normalization_layer(
            "batch_norm", momentum=0.999, epsilon=1e-3, name=name
        )

    conv1, norm1, conv2, norm2 = conv("conv1_1"), norm("bn1_1"), conv("conv1_2"), norm("bn1_2")
    pool = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="pool1")

    def forward(x, training):
        x = keras.ops.elu(norm1(conv1(x), training=training))
        x = keras.ops.elu(norm2(conv2(x), training=training))
        return pool(x)

    return Original(
        sublayers={"conv1": conv1, "norm1": norm1, "conv2": conv2, "norm2": norm2, "pool": pool},
        forward=forward,
        fused=("act1", "act2"),
    )


DEEPSORT = StemCase(
    case_id="deepsort-two-convs-elu-pool",
    make_model=DeepSortAppearanceNet,
    make_original=_deepsort_original,
    input_shape=(32, 16, 3),
    expected_output_shape=(16, 8, 32),
)


# ---------------------------------------------------------------------
# SiamFC and DaSiamRPN: stage 1 of the AlexNet-style trunk
# ---------------------------------------------------------------------

# Original: ``SiamFCBackbone`` and ``SiamRPNBackbone`` before the migration (HEAD
# 07363f8f6): ``conv1`` = Conv2D(96 * width, 11, strides 2, valid, bias) with the model's
# ``kernel_initializer``; ``bn1`` from the norm factory with the backbone's momentum and
# epsilon (0.9 and 1e-5, the torch values, when the model builds it) only when
# ``use_batch_norm``; a functional ``ops.relu`` (no layer, so ``fused``); ``pool1`` =
# MaxPooling2D(3, strides 2, valid). Stages two to five are the trunk's body and are not a
# stem. A model without BatchNorm still created ``bn1`` and never built or called it.
#: Neither the knob's default (``glorot_uniform``, which is also ConvStem's) nor
#: ConvStem's, so a stem that dropped the argument is visible.
_SIAM_KERNEL_INIT = "lecun_normal"
_SIAM_FILTERS = 96


def _siam_stage1_original(
    filters: int, batch_norm: bool, momentum: float, epsilon: float
) -> Original:
    conv = keras.layers.Conv2D(
        filters=filters,
        kernel_size=11,
        strides=2,
        padding="valid",
        use_bias=True,
        kernel_initializer=_SIAM_KERNEL_INIT,
        name="conv1",
    )
    norm = (
        create_normalization_layer(
            "batch_norm", momentum=momentum, epsilon=epsilon, name="bn1"
        )
        if batch_norm
        else None
    )
    pool = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="pool1")

    def forward(x, training):
        x = conv(x)
        if norm is not None:
            x = norm(x, training=training)
        return pool(keras.ops.relu(x))

    sublayers = {"conv1": conv, "pool": pool}
    if norm is not None:
        sublayers["norm1"] = norm
    return Original(sublayers=sublayers, forward=forward, fused=("act1",))


def _siamfc_model(batch_norm: bool = True) -> SiamFC:
    return SiamFC(use_batch_norm=batch_norm, kernel_initializer=_SIAM_KERNEL_INIT)


def _siamfc_backbone(momentum: float = 0.9, epsilon: float = 1e-5) -> SiamFCBackbone:
    return SiamFCBackbone(
        bn_momentum=momentum, bn_epsilon=epsilon, kernel_initializer=_SIAM_KERNEL_INIT
    )


def _dasiamrpn_model(variant: str = "otb", batch_norm: bool = True) -> DaSiamRPN:
    return DaSiamRPN.from_variant(
        variant, use_batch_norm=batch_norm, kernel_initializer=_SIAM_KERNEL_INIT
    )


def _siamrpn_backbone(momentum: float = 0.9, epsilon: float = 1e-5) -> SiamRPNBackbone:
    return SiamRPNBackbone(
        width_scale=1, bn_momentum=momentum, bn_epsilon=epsilon,
        kernel_initializer=_SIAM_KERNEL_INIT,
    )


def _through_backbone(model) -> ConvStem:
    return model.backbone.stem


# 47 px: (47 - 11) // 2 + 1 = 19, then a 3x3/2 valid pool gives 9.
def _siam_case(case_id, make_model, filters, batch_norm, momentum, epsilon, stem_of):
    return StemCase(
        case_id=case_id,
        make_model=make_model,
        make_original=lambda: _siam_stage1_original(filters, batch_norm, momentum, epsilon),
        input_shape=(47, 47, 3),
        expected_output_shape=(9, 9, filters),
        stem_of=stem_of,
    )


def _own_stem(layer) -> ConvStem:
    return layer.stem


SIAMFC_BN = _siam_case(
    "siamfc-bn", _siamfc_model, _SIAM_FILTERS, True, 0.9, 1e-5, _through_backbone
)
SIAMFC_NOBN = _siam_case(
    "siamfc-no-bn",
    lambda: _siamfc_model(batch_norm=False),
    _SIAM_FILTERS, False, 0.9, 1e-5, _through_backbone,
)
SIAMFC_BACKBONE_MOMENTUM_EPSILON = _siam_case(
    "siamfc-backbone-momentum-0.8-epsilon-1e-3",
    lambda: _siamfc_backbone(0.8, 1e-3),
    _SIAM_FILTERS, True, 0.8, 1e-3, _own_stem,
)
DASIAMRPN_WIDTH1 = _siam_case(
    "dasiamrpn-otb-width1", _dasiamrpn_model, _SIAM_FILTERS, True, 0.9, 1e-5, _through_backbone
)
DASIAMRPN_WIDTH2 = _siam_case(
    "dasiamrpn-big-width2",
    lambda: _dasiamrpn_model("big"),
    2 * _SIAM_FILTERS, True, 0.9, 1e-5, _through_backbone,
)
DASIAMRPN_NOBN = _siam_case(
    "dasiamrpn-otb-no-bn",
    lambda: _dasiamrpn_model(batch_norm=False),
    _SIAM_FILTERS, False, 0.9, 1e-5, _through_backbone,
)
DASIAMRPN_BACKBONE_MOMENTUM_EPSILON = _siam_case(
    "siamrpn-backbone-momentum-0.8-epsilon-1e-3",
    lambda: _siamrpn_backbone(0.8, 1e-3),
    _SIAM_FILTERS, True, 0.8, 1e-3, _own_stem,
)


#: One row per original variant. Later steps append here.
CASES = [
    RESNET_IMAGENET,
    RESNET_CIFAR,
    CLIFFORD_P1,
    CLIFFORD_P2,
    CLIFFORD_P4,
    CLIFFORD_OTHER,
    CLIFFORD_P2_NOBIAS,
    HIER_P1,
    HIER_P2,
    UCA_ENCODER,
    UCA_ENCODER_NOBIAS,
    CONVUNEXT_GELU_GRN_NOBIAS,
    CONVUNEXT_GELU_LAYERNORM,
    CONVUNEXT_LEAKY_RELU_STRING,
    CONVUNEXT_LEAKY_RELU_INSTANCE,
    CONVUNEXT_LINEAR,
    CONVUNEXT_BATCH_NORM,
    CONVNEXT_V1_VALID,
    CONVNEXT_V1_VALID_NONDIVISIBLE,
    CONVNEXT_V1_SAME_NOBIAS,
    CONVNEXT_V2_VALID,
    CONVNEXT_V2_VALID_NONDIVISIBLE_NOBIAS,
    CONVNEXT_V2_SAME,
    RAD_VALID,
    RAD_VALID_NONDIVISIBLE_NOBIAS,
    RAD_SAME,
    MOBILENET_V1,
    MOBILENET_V2,
    MOBILENET_V3,
    MOBILENET_V4,
    SQUEEZENET_V1_0,
    SQUEEZENET_V1_0_NOPOOL,
    SQUEEZENET_V1_1,
    SQUEEZENET_V1_1_NOPOOL,
    SQUEEZENET_V1_STRIDE1_SAME,
    SQUEEZENET_V2,
    SQUEEZENET_V2_NOPOOL,
    SQUEEZENET_V2_3D,
    SQUEEZENET_V2_3D_NOPOOL,
    SQUEEZENET_V2_3D_STRIDE1_SAME,
    VAE_BN_BIAS,
    VAE_BN_NOBIAS,
    VAE_NOBN_BIAS,
    VAE_NOBN_NOBIAS,
    VAE_GELU,
    VAE_RELU_STRING,
    VAE_LEAKY_RELU_INSTANCE,
    VAE_LINEAR,
    RBF_CLIFFORD_P1,
    RBF_CLIFFORD_P2,
    RBF_CLIFFORD_P2_NOREG,
    RBF_CLIFFORD_P4,
    RBF_CLIFFORD_P3,
    CAPSNET_V1_BLOCK1,
    CAPSNET_V1_BLOCK2,
    CAPSNET_V1_BLOCK3,
    CAPSNET_V1_NOBN,
    CAPSNET_V2_BLOCK1,
    CAPSNET_V2_BLOCK2,
    DEEPSORT,
    SIAMFC_BN,
    SIAMFC_NOBN,
    SIAMFC_BACKBONE_MOMENTUM_EPSILON,
    DASIAMRPN_WIDTH1,
    DASIAMRPN_WIDTH2,
    DASIAMRPN_NOBN,
    DASIAMRPN_BACKBONE_MOMENTUM_EPSILON,
]

#: ``(row id, make a model, a forward input shape)`` for the built-stem test.
BUILT_STEM_ROWS = [
    ("resnet-imagenet", lambda: _resnet_model("imagenet", (64, 64, 3)), (64, 64, 3)),
    ("resnet-cifar", lambda: _resnet_model("cifar", (32, 32, 3)), (32, 32, 3)),
    ("cliffordnet-patch1", lambda: _clifford_model(1), (16, 16, 3)),
    ("cliffordnet-patch2", lambda: _clifford_model(2), (16, 16, 3)),
    ("cliffordnet-patch4", lambda: _clifford_model(4), (16, 16, 3)),
    ("cliffordnet-patch3", lambda: _clifford_model(3), (16, 16, 3)),
    ("hierarchical-patch1", lambda: _hierarchical_model(1), (16, 16, 3)),
    ("hierarchical-patch2", lambda: _hierarchical_model(2), (16, 16, 3)),
    ("uca-encoder", _uca_model, (16, 16, 3)),
    (
        "convunext-stem",
        lambda: _convunext_model("global_response_norm", "gelu", False),
        (16, 16, 3),
    ),
    (
        "convnext-v1",
        lambda: _convnext_family_model(ConvNeXtV1, 4, True),
        (32, 32, 3),
    ),
    (
        "convnext-v2",
        lambda: _convnext_family_model(ConvNeXtV2, 4, True),
        (32, 32, 3),
    ),
    ("rad-convnet", lambda: _convnext_family_model(RADConvNet, 4, True), (32, 32, 3)),
    ("mobilenet-v1", _mobilenet_v1_model, (32, 32, 3)),
    ("mobilenet-v2", _mobilenet_v2_model, (32, 32, 3)),
    ("mobilenet-v3", _mobilenet_v3_model, (32, 32, 3)),
    ("mobilenet-v4", _mobilenet_v4_model, (32, 32, 3)),
    ("squeezenet-v1", lambda: _squeezenet_v1_model("1.0"), (64, 64, 3)),
    ("squeezenodule-v2", lambda: _squeezenet_v2_model("v2"), (64, 64, 1)),
    ("squeezenodule-v2-3d", lambda: _squeezenet_v2_3d_model(), (36, 36, 36, 1)),
    (
        "vae-encoder",
        lambda: _vae_model(True, True, "leaky_relu"),
        (32, 32, 3),
        _vae_stem_of,
    ),
    ("rbf-clifford-protonet", lambda: _rbf_clifford_model(2, True), (16, 16, 3)),
    ("capsnet-v1-block1", _capsnet_v1_model, (28, 28, 1), _capsnet_block(0)),
    ("capsnet-v1-block2", _capsnet_v1_model, (28, 28, 1), _capsnet_block(1)),
    ("capsnet-v2-legacy-block1", _capsnet_v2_model, (28, 28, 1), _capsnet_block(0)),
    ("capsnet-v2-legacy-block2", _capsnet_v2_model, (28, 28, 1), _capsnet_block(1)),
    ("deepsort", DeepSortAppearanceNet, (128, 64, 3)),
    # 91 px is the smallest side the trunk's five valid convolutions leave a feature map at.
    ("siamfc-backbone", lambda: _siamfc_model().backbone, (91, 91, 3)),
    ("dasiamrpn-backbone", lambda: _dasiamrpn_model().backbone, (91, 91, 3)),
]


def _model_stem(model) -> ConvStem:
    return model.stem


# ---------------------------------------------------------------------
# Initializer audit: how each model's initializer knob reaches its stem
# ---------------------------------------------------------------------

# D-020 (pass-2 WARNING 1). ``assert_stem_settings_match_original`` compares the
# stem's initializer with the original's, so it can only see a dropped
# passthrough when the model is built with a value ConvStem would NOT fall back
# to. MobileNetV3's knob defaults to ``glorot_uniform``, which is ConvStem's own
# default: built at its default, a stem without ``kernel_initializer=`` passed
# the check. Statuses:
#
# * ``knob``: the model has a ``kernel_initializer`` constructor parameter that
#   reaches the stem. The harness builds it with a value that differs from the
#   parameter's default AND from ConvStem's default, and the original literal is
#   that value. (Models with a ``bias_initializer`` parameter: the same.)
# * ``knob-not-forwarded``: the parameter exists but the stem deliberately does
#   not read it (the stem has its own constant). The model is built with a
#   different knob value, and the stem must keep the constant.
# * ``hardcoded``: no parameter; the stem call writes a literal that is not
#   ConvStem's default. The original literal is the expectation.
# * ``default-no-knob``: no parameter, and the original used the Keras default,
#   which IS ConvStem's default, so nothing can be made non-default. The reason
#   is the model has no initializer to forward.
INITIALIZER_AUDIT = {
    "ResNet": ("knob", "lecun_normal (default he_normal)"),
    "CliffordNet": (
        "knob",
        "kernel TruncatedNormal(0.02), bias Constant(0.25), L2 kernel and bias",
    ),
    "HierarchicalCliffordNet": (
        "knob",
        "kernel TruncatedNormal(0.02), bias Constant(0.25), L2 kernel and bias",
    ),
    "UcaEncoder": (
        "default-no-knob",
        "no initializer or regularizer parameter; the original Conv2D took "
        "Keras' glorot_uniform, which is ConvStem's default",
    ),
    "ConvUNextStem": ("knob", "he_uniform (default he_normal)"),
    "ConvNeXtV1": ("hardcoded", "STEM_INITIALIZER TruncatedNormal(0.02)"),
    "ConvNeXtV2": ("hardcoded", "STEM_INITIALIZER TruncatedNormal(0.02)"),
    "RADConvNet": ("hardcoded", "STEM_INITIALIZER 'truncated_normal'"),
    "MobileNetV1": ("knob", "he_uniform (default he_normal)"),
    "MobileNetV2": ("knob", "lecun_normal (default he_normal)"),
    "MobileNetV3": ("knob", "he_uniform (default glorot_uniform = ConvStem's)"),
    "MobileNetV4": ("knob", "lecun_normal (default he_normal)"),
    "SqueezeNetV1": (
        "knob-not-forwarded",
        "knob he_uniform reaches the Fire blocks; the stem keeps lecun_uniform",
    ),
    "SqueezeNoduleNetV2": (
        "knob-not-forwarded",
        "knob he_uniform reaches the Fire blocks; the stem keeps lecun_uniform",
    ),
    "VAE": ("knob", "lecun_normal (default he_normal)"),
    "CapsNet": ("knob", "lecun_normal (default he_normal)"),
    "CapsNetV2": ("knob", "lecun_normal (default he_normal)"),
    "DeepSortAppearanceNet": (
        "hardcoded",
        "TruncatedNormal(stddev=1e-3), the reference's init; the stem call writes the literal",
    ),
    "SiamFC": ("knob", "lecun_normal (default glorot_uniform = ConvStem's)"),
    "SiamFCBackbone": ("knob", "lecun_normal (default glorot_uniform = ConvStem's)"),
    "DaSiamRPN": ("knob", "lecun_normal (default glorot_uniform = ConvStem's)"),
    "SiamRPNBackbone": ("knob", "lecun_normal (default glorot_uniform = ConvStem's)"),
    "CliffordRBFProtoNet": (
        "default-no-knob",
        "no initializer parameter; the original Conv2D took Keras' "
        "glorot_uniform / zeros, which are ConvStem's defaults",
    ),
}

_REGULARIZER_KNOBS = {
    "kernel_regularizer": ("kernel_regularizer",),
    "bias_regularizer": ("bias_regularizer",),
}


def _knob_default(cls, knob: str) -> dict:
    """The seed-free signature of ``cls.__init__``'s default for ``knob``."""
    default = inspect.signature(cls.__init__).parameters[knob].default
    return _initializer_signature(keras.initializers.get(default))


def assert_initializer_audit(
    case: StemCase, model=None, audit: Optional[dict] = None
) -> None:
    """Assert the case's model is built so a dropped initializer is visible.

    Interface contract: raises ``AssertionError`` naming the case and the rule
    on any violation (never another type). ``model`` defaults to a fresh
    ``case.make_model()``; ``audit`` defaults to ``INITIALIZER_AUDIT``. Builds
    nothing beyond the model and mutates nothing.
    """
    audit = INITIALIZER_AUDIT if audit is None else audit
    model = case.make_model() if model is None else model
    name = type(model).__name__
    assert name in audit, f"{case.case_id}: {name} has no INITIALIZER_AUDIT row"
    status, _ = audit[name]
    params = inspect.signature(type(model).__init__).parameters
    has_knob = "kernel_initializer" in params
    assert has_knob == (status in ("knob", "knob-not-forwarded")), (
        f"{case.case_id}: the audit says {name} is {status!r} but its constructor "
        f"{'has' if has_knob else 'has no'} kernel_initializer parameter"
    )

    want = _conv_settings(case.make_original().sublayers["conv1"])
    stem_default = {
        knob: _knob_default(ConvStem, knob)
        for knob in ("kernel_initializer", "bias_initializer")
    }

    if status == "hardcoded":
        assert want["kernel_initializer"] != stem_default["kernel_initializer"], (
            f"{case.case_id}: a hardcoded initializer equal to ConvStem's default "
            f"({want['kernel_initializer']['class_name']}) is invisible when dropped"
        )
    elif status == "default-no-knob":
        assert want["kernel_initializer"] == stem_default["kernel_initializer"]
    else:
        value = _initializer_signature(
            keras.initializers.get(model.kernel_initializer)
        )
        assert value != _knob_default(type(model), "kernel_initializer"), (
            f"{case.case_id}: built with {name}'s own default kernel_initializer"
        )
        assert value != stem_default["kernel_initializer"], (
            f"{case.case_id}: built with ConvStem's default kernel_initializer "
            f"({value['class_name']}); a stem that dropped the argument would pass"
        )
        if status == "knob":
            assert want["kernel_initializer"] == value, (
                f"{case.case_id}: the original literal "
                f"{want['kernel_initializer']['class_name']} is not the knob value "
                f"{value['class_name']}"
            )
        else:
            assert want["kernel_initializer"] != value, (
                f"{case.case_id}: the knob value reaches a stem the audit says "
                f"does not read it"
            )
            assert want["kernel_initializer"] != stem_default["kernel_initializer"]

    if "bias_initializer" in params:
        value = _initializer_signature(
            keras.initializers.get(model.bias_initializer)
        )
        assert value != _knob_default(type(model), "bias_initializer")
        assert value != stem_default["bias_initializer"], (
            f"{case.case_id}: built with ConvStem's default bias_initializer"
        )
        assert want["bias_initializer"] == value
    else:
        assert want["bias_initializer"] == stem_default["bias_initializer"], (
            f"{case.case_id}: {name} has no bias_initializer parameter yet the "
            f"original used {want['bias_initializer']['class_name']}"
        )


# ---------------------------------------------------------------------
# The comparison on every real case
# ---------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_the_model_stem_replicates_the_original(case):
    """The stem a model builds equals the pre-migration stem, weight for weight."""
    model = case.make_model()
    assert_stem_matches_original(case, case.stem_of(model))


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_every_case_is_built_so_a_dropped_initializer_is_visible(case):
    """The case's model is built with a non-default initializer (see the audit)."""
    assert_initializer_audit(case)


def test_the_initializer_audit_has_no_stale_row():
    """Every audit row names a model class some case builds, and every model
    class a case builds has a row."""
    built = {type(case.make_model()).__name__ for case in CASES}
    assert built == set(INITIALIZER_AUDIT), (
        f"built but not audited: {sorted(built - set(INITIALIZER_AUDIT))}; "
        f"audited but never built: {sorted(set(INITIALIZER_AUDIT) - built)}"
    )


def test_a_model_with_a_regularizer_knob_is_built_regularized_in_some_case():
    """A model exposing a kernel/bias regularizer knob has a case whose original
    carries one, or a dropped passthrough would fall back to ``None``."""
    regularized = {}
    exposed = {}
    for case in CASES:
        model = case.make_model()
        name = type(model).__name__
        params = inspect.signature(type(model).__init__).parameters
        want = _conv_settings(case.make_original().sublayers["conv1"])
        for setting, knobs in _REGULARIZER_KNOBS.items():
            if any(knob in params for knob in knobs):
                exposed.setdefault(setting, set()).add(name)
                if want[setting] is not None:
                    regularized.setdefault(setting, set()).add(name)
    for setting, names in exposed.items():
        missing = names - regularized.get(setting, set())
        assert not missing, (
            f"{sorted(missing)} expose a {setting} knob but no case builds them "
            f"with a non-None one"
        )
    assert exposed["kernel_regularizer"] and exposed["bias_regularizer"]


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_the_model_stem_draws_and_regularizes_like_the_original(case):
    """Initializers and regularizers of every stem conv equal the original's.

    The test above overwrites every weight with the original's, so it cannot see
    which initializer the stem would have drawn them with. D-005's independent
    draw for ``conv2`` is a seed difference only and is ignored here.
    """
    model = case.make_model()
    assert_stem_settings_match_original(case, case.stem_of(model))


def test_the_imagenet_stem_pool_geometry_is_stride_2_same():
    """Stated outright, not only through values: 3x3, stride 2, ``same``, 224 -> 56."""
    model = RESNET_IMAGENET.make_model()
    pool = model.stem.pool
    assert pool is not None
    assert tuple(pool.pool_size) == (3, 3)
    assert tuple(pool.strides) == (2, 2), (
        f"the stem pool strides are {pool.strides}; the original used 2 "
        f"(stride 3 turns 224 into 37x37 instead of 56x56)"
    )
    assert pool.padding == "same"
    shape = model.stem.compute_output_shape((None, 224, 224, 3))
    assert tuple(shape) == (None, 56, 56, 64)
    out = model.stem(np.zeros((1, 224, 224, 3), "float32"), training=False)
    assert tuple(out.shape) == (1, 56, 56, 64)


def test_the_cifar_stem_has_no_pool_and_keeps_the_resolution():
    model = RESNET_CIFAR.make_model()
    assert model.stem.pool is None
    assert tuple(model.stem.compute_output_shape((None, 32, 32, 3))) == (
        None,
        32,
        32,
        64,
    )


# ---------------------------------------------------------------------
# CliffordNet family: the model's bias initializer / regularizer reach the stem
# ---------------------------------------------------------------------

# The weight transplant of ``assert_stem_matches_original`` overwrites every
# bias, so an initializer dropped on the way to the stem convs is invisible to
# it (the regularization loss does see a dropped regularizer, on a stem that
# has a bias). These read the freshly built weights and the conv attributes.

_SINGLE_CONV_MODELS = [
    ("cliffordnet-patch2", lambda: _clifford_model(2)),
    ("cliffordnet-patch3", lambda: _clifford_model(3)),
    ("hierarchical-patch2", lambda: _hierarchical_model(2)),
]
_TWO_CONV_MODELS = [
    ("cliffordnet-patch1", lambda: _clifford_model(1)),
    ("cliffordnet-patch4", lambda: _clifford_model(4)),
    ("hierarchical-patch1", lambda: _hierarchical_model(1)),
]


@pytest.mark.parametrize(
    "make_model", [m for _, m in _SINGLE_CONV_MODELS], ids=[i for i, _ in _SINGLE_CONV_MODELS]
)
def test_a_single_conv_stem_gets_the_model_bias_initializer_and_regularizer(make_model):
    """The stem conv bias starts at the model's ``bias_initializer`` and carries
    its ``bias_regularizer``: 0.25 everywhere, and a bias penalty in ``losses``."""
    model = make_model()
    model.stem.build((None, 16, 16, 3))
    bias = _np(model.stem.conv1.bias)
    np.testing.assert_array_equal(
        bias, np.full_like(bias, _CLIFFORD_BIAS_INIT)
    )
    assert model.stem.conv1.bias_regularizer is not None
    expected = _CLIFFORD_BIAS_REG * float(np.sum(bias**2))
    bias_penalty = float(_np(model.stem.conv1.bias_regularizer(model.stem.conv1.bias)))
    assert bias_penalty == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "make_model", [m for _, m in _TWO_CONV_MODELS], ids=[i for i, _ in _TWO_CONV_MODELS]
)
def test_a_two_conv_stem_hands_both_convs_the_model_bias_settings(make_model):
    """The two-conv stems are bias-free, so no bias weight exists to read; the
    settings are still the model's on BOTH convs (the original passed them)."""
    model = make_model()
    stem = model.stem
    for conv in (stem.conv1, stem.conv2):
        assert conv.use_bias is False
        init = conv.bias_initializer
        assert isinstance(init, keras.initializers.Constant)
        assert float(init.value) == _CLIFFORD_BIAS_INIT
        assert conv.bias_regularizer is not None
        assert float(conv.bias_regularizer.l2) == pytest.approx(_CLIFFORD_BIAS_REG)


# ---------------------------------------------------------------------
# Planted controls: the comparison must FAIL on each defect class
# ---------------------------------------------------------------------


def _stem_from_config(case: StemCase, **overrides) -> ConvStem:
    """A ConvStem built from the model's own stem config with ``overrides``."""
    config = case.stem_of(case.make_model()).get_config()
    norm_kwargs = dict(config["normalization_kwargs"])
    norm_kwargs.update(overrides.pop("normalization_kwargs", {}))
    config.update(overrides)
    config["normalization_kwargs"] = norm_kwargs
    return ConvStem.from_config(config)


def _norm_applied_twice(stem: ConvStem, x, training):
    """The UcaEncoder defect shape: conv -> norm -> norm -> act -> pool."""
    x = stem.conv1(x)
    x = stem.norm1(x, training=training)
    x = stem.norm1(x, training=training)
    if stem.act1 is not None:
        x = stem.act1(x)
    if stem.pool is not None:
        x = stem.pool(x)
    return x


def test_the_guard_the_harness_sees_a_wrong_pool():
    """Stride 3 / ``valid`` (the defect the first migration introduced) must make
    the comparison fail."""
    stem = _stem_from_config(RESNET_IMAGENET, pool_strides=None, pool_padding="valid")
    with pytest.raises(AssertionError, match="shape"):
        assert_stem_matches_original(RESNET_IMAGENET, stem)


def test_the_guard_the_harness_sees_a_wrong_epsilon():
    """BN epsilon 1e-3 against the original's 1e-6 must make the comparison fail."""
    stem = _stem_from_config(
        RESNET_IMAGENET, normalization_kwargs={"epsilon": 1e-3}
    )
    assert stem.norm1.epsilon == pytest.approx(1e-3)
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(RESNET_IMAGENET, stem)


def test_the_guard_the_harness_sees_a_norm_applied_twice():
    """A norm in the forward twice must make the comparison fail."""
    model = RESNET_IMAGENET.make_model()
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(
            RESNET_IMAGENET, model.stem, forward=_norm_applied_twice
        )


def test_the_guard_the_harness_sees_the_other_models_width_rule():
    """V1 truncates ``32 * 0.6`` to 19; a stem rounded to a multiple of 8 (24) must fail.

    The transplant of the original's 19-wide kernel into a 24-wide conv is refused by Keras
    before any output is compared.
    """
    stem = _stem_from_config(MOBILENET_V1, filters=_MOBILENET_V2_STEM_FILTERS)
    with pytest.raises(ValueError, match="not compatible with provided weight shape"):
        assert_stem_matches_original(MOBILENET_V1, stem)


def test_the_guard_the_harness_sees_a_dropped_relu6_clip():
    """ReLU with no ``max_value=6`` against V2's clipped stem must fail the comparison."""
    stem = _stem_from_config(MOBILENET_V2, activation_kwargs={})
    assert stem.act1.max_value is None
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(MOBILENET_V2, stem)


def test_the_guard_the_relu6_case_needs_inputs_that_reach_the_clip():
    """At unit-scale inputs the activation input never reaches 6, so the same plain
    ReLU stem passes: the case's ``input_scale`` is what makes the clip visible."""
    stem = _stem_from_config(MOBILENET_V2, activation_kwargs={})
    assert_stem_matches_original(replace(MOBILENET_V2, input_scale=1.0), stem)


@pytest.mark.parametrize(
    "case,override,setting",
    [
        # he_normal where the original drew glorot_uniform (UcaEncoder's conv
        # has no initializer argument: the Keras default).
        (UCA_ENCODER, {"kernel_initializer": "he_normal"}, "kernel_initializer"),
        # Same class, other parameter: the original is TruncatedNormal(0.02).
        (
            CLIFFORD_P2,
            {"kernel_initializer": keras.initializers.TruncatedNormal(stddev=0.2)},
            "kernel_initializer",
        ),
        # The original took Constant(0.25); the stem default is zeros.
        (CLIFFORD_P2, {"bias_initializer": "zeros"}, "bias_initializer"),
        # Regularizer of another strength, and a dropped one.
        (
            RESNET_IMAGENET,
            {"kernel_regularizer": keras.regularizers.L2(1e-3)},
            "kernel_regularizer",
        ),
        (RESNET_IMAGENET, {"kernel_regularizer": None}, "kernel_regularizer"),
        # The pass-2 mutant: MobileNetV3's stem call without kernel_initializer
        # falls back to ConvStem's glorot_uniform. At the model's own default
        # (also glorot_uniform) this was indistinguishable.
        (MOBILENET_V1, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        (MOBILENET_V2, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        (MOBILENET_V3, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        (MOBILENET_V4, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        (VAE_BN_BIAS, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        # DeepSORT writes its initializer as a literal: a stem at the default
        # glorot_uniform, or at another truncated-normal width, is a defect.
        (DEEPSORT, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        (
            DEEPSORT,
            {"kernel_initializer": keras.initializers.TruncatedNormal(stddev=1e-2)},
            "kernel_initializer",
        ),
        (DEEPSORT, {"kernel_regularizer": None}, "kernel_regularizer"),
        # The trunk's knob reaches stage 1 as the model's own value.
        (SIAMFC_BN, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
        (DASIAMRPN_WIDTH1, {"kernel_initializer": "glorot_uniform"}, "kernel_initializer"),
    ],
    ids=[
        "kernel-initializer-class",
        "kernel-initializer-parameter",
        "bias-initializer",
        "kernel-regularizer-strength",
        "kernel-regularizer-dropped",
        "mobilenet-v1-initializer-dropped",
        "mobilenet-v2-initializer-dropped",
        "mobilenet-v3-initializer-dropped",
        "mobilenet-v4-initializer-dropped",
        "vae-initializer-dropped",
        "deepsort-initializer-at-the-default",
        "deepsort-initializer-other-width",
        "deepsort-regularizer-dropped",
        "siamfc-initializer-dropped",
        "dasiamrpn-initializer-dropped",
    ],
)
def test_the_guard_the_settings_check_sees_a_wrong_initializer_or_regularizer(
    case, override, setting
):
    """A stem built with a wrong setting must fail the settings comparison,
    naming the setting. The weight-transplant comparison passes on the stems with
    a wrong initializer (it never reads one), which is the gap this closes.
    """
    stem = _stem_from_config(case, **override)
    with pytest.raises(AssertionError, match=setting):
        assert_stem_settings_match_original(case, stem)
    if "initializer" in setting:
        # Every weight is overwritten, so the transplant comparison is blind to
        # an initializer. (It does see a regularizer, through the loss.)
        assert_stem_matches_original(case, stem)


def _mobilenet_v3_at_its_default_initializer() -> MobileNetV3:
    return MobileNetV3(
        variant="small",
        num_classes=4,
        include_top=False,
        input_shape=(32, 32, 3),
        kernel_regularizer=keras.regularizers.L2(_MOBILENET_KERNEL_REG),
    )


def test_the_guard_the_audit_sees_a_model_built_at_convstems_default():
    """The pass-2 case: MobileNetV3 at its own default initializer, which is
    ConvStem's. The audit must refuse it."""
    with pytest.raises(AssertionError, match="own default kernel_initializer"):
        assert_initializer_audit(
            MOBILENET_V3, model=_mobilenet_v3_at_its_default_initializer()
        )


def test_the_guard_the_audit_sees_a_wrong_table_row():
    """A row that contradicts the constructor (a knob called hardcoded, and a
    hardcoded model called a knob) must fail."""
    lie = {**INITIALIZER_AUDIT, "MobileNetV4": ("hardcoded", "wrong")}
    with pytest.raises(AssertionError, match="has kernel_initializer parameter"):
        assert_initializer_audit(MOBILENET_V4, audit=lie)
    lie = {**INITIALIZER_AUDIT, "UcaEncoder": ("knob", "wrong")}
    with pytest.raises(AssertionError, match="has no kernel_initializer parameter"):
        assert_initializer_audit(UCA_ENCODER, audit=lie)
    with pytest.raises(AssertionError, match="no INITIALIZER_AUDIT row"):
        assert_initializer_audit(
            MOBILENET_V4, audit={k: v for k, v in INITIALIZER_AUDIT.items() if k != "MobileNetV4"}
        )


def test_the_guard_a_correct_stem_survives_the_settings_check():
    """Anti-vacuity twin: a stem rebuilt from the model's own config passes, and
    so does the two-conv stem whose conv2 holds a clone with another seed."""
    assert_stem_settings_match_original(
        RESNET_IMAGENET, _stem_from_config(RESNET_IMAGENET)
    )
    stem = CLIFFORD_P1.make_model().stem
    assert stem.conv1.kernel_initializer is not stem.conv2.kernel_initializer
    assert_stem_settings_match_original(CLIFFORD_P1, stem)


def test_the_guard_the_harness_sees_a_capsnet_block_at_the_norm_factorys_epsilon():
    """The paper has no batch norm and the blocks keep the Keras epsilon of 1e-3; the
    norm factory's own default is 1e-6, so a block that did not pass its epsilon is
    1000x off with no shape symptom. The harness must tell the two apart."""
    stem = _stem_from_config(CAPSNET_V1_BLOCK1, normalization_kwargs={"epsilon": 1e-6})
    assert stem.norm1.epsilon == pytest.approx(1e-6)
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(CAPSNET_V1_BLOCK1, stem)
    own = _stem_from_config(CAPSNET_V1_BLOCK1)
    assert own.norm1.epsilon == pytest.approx(1e-3)
    assert_stem_matches_original(CAPSNET_V1_BLOCK1, own)


@pytest.mark.parametrize(
    "override,match",
    [
        # 3x3/2 pool padded "valid": 32 rows become 15, not 16.
        ({"pool_padding": "valid"}, "shape"),
        # The factory's own epsilon is 1e-6 and a torch-style 1e-5 is the usual
        # carry-over: the reference is TF-slim's 1e-3.
        ({"normalization_kwargs": {"epsilon": 1e-5}}, "differs from the original"),
        ({"normalization_kwargs": {"epsilon": 1e-6}}, "differs from the original"),
        # Momentum is invisible at inference; the moving statistics after a
        # training=True pass see it.
        ({"normalization_kwargs": {"momentum": 0.9}}, "after a training=True pass"),
        ({"activation_type": "relu"}, "differs from the original"),
        ({"kernel_regularizer": None}, "regularization loss"),
    ],
    ids=[
        "pool-padding",
        "epsilon-1e-5",
        "epsilon-1e-6",
        "momentum-0.9",
        "relu-for-elu",
        "regularizer-dropped",
    ],
)
def test_the_guard_the_harness_sees_a_deepsort_stem_with_one_wrong_setting(override, match):
    """Each TF-slim value the reference fixes is a defect when it is not kept."""
    stem = _stem_from_config(DEEPSORT, **override)
    with pytest.raises(AssertionError, match=match):
        assert_stem_matches_original(DEEPSORT, stem)
    assert_stem_matches_original(DEEPSORT, _stem_from_config(DEEPSORT))


@pytest.mark.parametrize(
    "case,override,match",
    [
        # A 3x3/2 pool padded "same" turns 19 into 10, not 9.
        (SIAMFC_BN, {"pool_padding": "same"}, "shape"),
        # A "same" convolution keeps 47 at stride 2: 24, not 19.
        (SIAMFC_BN, {"padding": "same"}, "shape"),
        # The factory's own epsilon is 1e-6; the torch value the trunk carries is 1e-5.
        (SIAMFC_BN, {"normalization_kwargs": {"epsilon": 1e-6}}, "differs from the original"),
        # Momentum shows only in the moving statistics after a training=True pass.
        (SIAMFC_BN, {"normalization_kwargs": {"momentum": 0.99}}, "after a training=True pass"),
        (SIAMFC_BN, {"activation_type": "gelu"}, "differs from the original"),
        (SIAMFC_BACKBONE_MOMENTUM_EPSILON, {"normalization_kwargs": {"momentum": 0.9}},
         "after a training=True pass"),
        (DASIAMRPN_WIDTH2, {"filters": _SIAM_FILTERS}, "not compatible"),
        (DASIAMRPN_WIDTH2, {"pool_padding": "same"}, "shape"),
        (DASIAMRPN_WIDTH1, {"normalization_kwargs": {"epsilon": 1e-6}}, "differs from the original"),
    ],
    ids=[
        "siamfc-pool-padding",
        "siamfc-conv-padding",
        "siamfc-epsilon-1e-6",
        "siamfc-momentum-0.99",
        "siamfc-gelu-for-relu",
        "siamfc-backbone-momentum-back-at-the-default",
        "dasiamrpn-width-not-doubled",
        "dasiamrpn-pool-padding",
        "dasiamrpn-epsilon-1e-6",
    ],
)
def test_the_guard_the_harness_sees_a_siamese_stem_with_one_wrong_setting(case, override, match):
    """Each value the trunk's first stage fixes is a defect when it is not kept; the
    stem rebuilt from the model's own config passes the same comparison."""
    stem = _stem_from_config(case, **override)
    exception = ValueError if match == "not compatible" else AssertionError
    with pytest.raises(exception, match=match):
        assert_stem_matches_original(case, stem)
    assert_stem_matches_original(case, _stem_from_config(case))


def test_the_guard_the_harness_sees_a_siamese_norm_applied_twice():
    """A second ``bn1`` in the forward changes the output."""
    model = SIAMFC_BN.make_model()
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(SIAMFC_BN, SIAMFC_BN.stem_of(model), forward=_norm_applied_twice)


def test_the_guard_a_siamese_model_without_batch_norm_has_no_norm_in_its_stem():
    """The no-BN case has no ``norm1`` role: a stem that built one fails the role-set check."""
    stem = _stem_from_config(SIAMFC_BN)  # BatchNorm built, against the original without one
    with pytest.raises(AssertionError, match="roles"):
        assert_stem_matches_original(SIAMFC_NOBN, stem)


def test_the_guard_a_correct_stem_survives_the_same_harness():
    """Anti-vacuity twin: the controls above fail for their defect, not because
    the harness rejects everything. A stem rebuilt from the model's own config
    with no override passes.
    """
    stem = _stem_from_config(RESNET_IMAGENET)
    assert_stem_matches_original(RESNET_IMAGENET, stem)


# ---------------------------------------------------------------------
# training=None
# ---------------------------------------------------------------------

# What ``training=None`` means in Keras 3 (measured on 3.8.0): a top-level call
# has no enclosing call context to inherit a flag from, so ``Layer.__call__``
# leaves ``None`` as ``None`` and ``BatchNormalization.call`` evaluates
# ``if training and self.trainable``: ``None`` is falsy, so the layer normalizes
# with its MOVING statistics, exactly as ``training=False`` does, and updates
# nothing. It is not "unspecified": a model that omits ``training`` runs every BN
# of its stem in inference. The comparison is only meaningful if the arm differs
# from the others, so ``test_training_none_is_inference_for_batch_norm`` pins
# that on the original (None == False != True once the statistics are random),
# and ``test_the_guard_the_harness_sees_none_treated_as_training`` hands the arm a
# stem that turns ``None`` into ``True``, which passes the False and True arms
# and must fail only this one. LayerNorm, GRN and conv-only stems ignore
# ``training``; they stay in the set because a stem that special-cased ``None``
# anywhere would still change the output.

_TRAINING_NONE_BATCH_NORM = [
    RESNET_IMAGENET,
    CLIFFORD_P1,
    MOBILENET_V3,
    VAE_BN_BIAS,
    CONVUNEXT_BATCH_NORM,
]
_TRAINING_NONE_OTHER = [
    UCA_ENCODER,
    CONVNEXT_V1_VALID,
    CONVUNEXT_GELU_LAYERNORM,
    CONVUNEXT_GELU_GRN_NOBIAS,
    SQUEEZENET_V1_1,
]
_TRAINING_NONE_CASES = _TRAINING_NONE_BATCH_NORM + _TRAINING_NONE_OTHER


@pytest.mark.parametrize(
    "case", _TRAINING_NONE_CASES, ids=[c.case_id for c in _TRAINING_NONE_CASES]
)
def test_the_model_stem_replicates_the_original_at_training_none(case):
    """``training=None`` equals the original's, exactly, with random BN statistics
    (the False and True arms run in the same call)."""
    model = case.make_model()
    assert_stem_matches_original(
        case, case.stem_of(model), check_training_none=True
    )


@pytest.mark.parametrize(
    "case",
    _TRAINING_NONE_BATCH_NORM,
    ids=[c.case_id for c in _TRAINING_NONE_BATCH_NORM],
)
def test_training_none_is_inference_for_batch_norm(case):
    """On the original with random statistics: None == False and None != True,
    so the ``training=None`` arm is the inference arm and not a copy of the
    training one (the premise of the comment above, measured)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH,) + tuple(case.input_shape)).astype("float32")
    original = case.make_original()
    original.forward(x, False)
    for layer in original.sublayers.values():
        _randomize(layer, rng)
    out_none = _np(original.forward(x, None))
    out_false = _np(original.forward(x, False))
    out_true = _np(original.forward(x, True))  # last: it moves the statistics
    assert np.array_equal(out_none, out_false), f"{case.case_id}: None != False"
    assert float(np.max(np.abs(out_none - out_true))) > 1e-3, (
        f"{case.case_id}: training=True equals None, the arm cannot tell them apart"
    )


def _none_treated_as_training(stem: ConvStem, x, training):
    """The planted defect: a stem (or model) that turns an omitted ``training``
    into ``True``. Identical to ``stem(x, training=training)`` for False/True."""
    return stem(x, training=True if training is None else training)


def test_the_guard_the_harness_sees_none_treated_as_training():
    """Passes the False and True arms (same output there), fails ``None``."""
    model = RESNET_IMAGENET.make_model()
    assert_stem_matches_original(RESNET_IMAGENET, model.stem, forward=_none_treated_as_training)
    with pytest.raises(AssertionError, match=r"training=None output differs"):
        assert_stem_matches_original(
            RESNET_IMAGENET,
            RESNET_IMAGENET.make_model().stem,
            forward=_none_treated_as_training,
            check_training_none=True,
        )


def test_the_guard_the_none_arm_sees_a_wrong_epsilon():
    """The existing wrong-epsilon control, run through the ``None`` arm as well:
    the stem fails the whole comparison with the arm switched on."""
    stem = _stem_from_config(
        RESNET_IMAGENET, normalization_kwargs={"epsilon": 1e-3}
    )
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(
            RESNET_IMAGENET, stem, check_training_none=True
        )


# ---------------------------------------------------------------------
# Every migrated model runs its stem
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_model,input_shape,stem_of",
    [(row[1], row[2], row[3] if len(row) > 3 else _model_stem) for row in BUILT_STEM_ROWS],
    ids=[row[0] for row in BUILT_STEM_ROWS],
)
def test_every_migrated_model_builds_its_stem(make_model, input_shape, stem_of):
    """After a forward, ``model.stem`` and every sub-layer of it are built.

    After the first migration, ResNet, ConvNeXt and RAD ran their stems through aliases of the
    sub-layers, so the ``ConvStem`` object itself was never called and stayed
    unbuilt (``stem.built is False``) under a green suite.
    """
    model = make_model()
    model(np.zeros((1,) + tuple(input_shape), "float32"), training=False)
    stem = stem_of(model)
    assert stem.built is True, "model.stem was never built: call() bypasses it"
    unbuilt = [
        layer.name
        for layer in stem._flatten_layers(include_self=False)
        if not layer.built
    ]
    assert unbuilt == [], f"unbuilt sub-layers of the stem: {unbuilt}"
    assert len(stem.weights) > 0


# ---------------------------------------------------------------------
# The model applies its stem exactly once
# ---------------------------------------------------------------------

# The harness compares a stem in isolation, so a model that applies the stem (or
# one of its layers) again in its own ``call`` is invisible to it. The first
# migration's UcaEncoder did exactly that: ConvStem already ran conv -> norm, and ``call()``
# still ran ``self.stem_norm(x)`` after it. The counter is the pattern of
# ``test_mambalct.py`` (wrap ``layer.call``), made restoring.


@contextlib.contextmanager
def _counting_calls(layers: Dict[str, keras.layers.Layer]) -> Iterator[Dict[str, int]]:
    """Count the executions of ``layer.call`` for each labelled layer.

    Interface contract: yields ``{label: calls}``, updated live while the block
    runs; on exit (also on error) every wrapper is removed, so the layer is
    exactly as before (``"call" not in vars(layer)`` is asserted). A layer must
    not already carry an instance-level ``call``. Counts executions of the Python
    ``call`` body, so it is meaningful for an eager forward, not for a traced one.
    """
    counts = {label: 0 for label in layers}

    def install(label: str, layer: keras.layers.Layer) -> None:
        original_call = layer.call

        def counting_call(*args, **kwargs):  # type: ignore[no-untyped-def]
            counts[label] += 1
            return original_call(*args, **kwargs)

        layer.call = counting_call

    for label, layer in layers.items():
        assert "call" not in vars(layer), f"{label} is already wrapped"
        install(label, layer)
    try:
        yield counts
    finally:
        for label, layer in layers.items():
            if "call" in vars(layer):
                del layer.call
            assert "call" not in vars(layer), f"{label}: wrapper left installed"


def _stem_call_sites(stem: ConvStem) -> Dict[str, keras.layers.Layer]:
    """The stem and every sub-layer it carries, by label."""
    sites = {"stem": stem}
    sites.update({f"stem.{role}": layer for role, layer in _stem_sublayers(stem).items()})
    return sites


@pytest.mark.parametrize(
    "make_model,input_shape,stem_of",
    [(row[1], row[2], row[3] if len(row) > 3 else _model_stem) for row in BUILT_STEM_ROWS],
    ids=[row[0] for row in BUILT_STEM_ROWS],
)
def test_every_migrated_model_applies_its_stem_exactly_once(
    make_model, input_shape, stem_of
):
    """One forward runs ``ConvStem.call`` and each of its layers exactly once,
    at ``training`` False and True (a branch on ``training`` could hide a second
    application)."""
    model = make_model()
    batch = np.zeros((1,) + tuple(input_shape), "float32")
    model(batch, training=False)  # build: counts below are of a built model
    sites = _stem_call_sites(stem_of(model))
    for training in (False, True):
        with _counting_calls(sites) as counts:
            model(batch, training=training)
        assert counts == {label: 1 for label in sites}, (
            f"training={training}: calls per stem layer in one forward {counts}; "
            f"each must be 1 (the model applies the stem once)"
        )


class _AppliesItsStem(PublicAutoName, keras.layers.Layer):
    """A planted model: applies a shape-preserving ConvStem ``times`` times."""

    def __init__(self, times: int, **kwargs):
        super().__init__(**kwargs)
        self.times = times
        self.stem = ConvStem(filters=3, normalization_type="layer_norm")

    def call(self, inputs, training=None):
        x = inputs
        for _ in range(self.times):
            x = self.stem(x, training=training)
        return x


def test_the_guard_the_counter_reads_two_when_a_model_applies_its_stem_twice():
    """The counter is not blind: 1 for a single application, 2 for a double one,
    and the wrappers are gone afterwards."""
    batch = np.zeros((1, 8, 8, 3), "float32")
    readings = {}
    for times in (1, 2):
        model = _AppliesItsStem(times)
        model(batch, training=False)
        sites = _stem_call_sites(model.stem)
        with _counting_calls(sites) as counts:
            model(batch, training=False)
        readings[times] = dict(counts)
        assert all("call" not in vars(layer) for layer in sites.values())
    assert set(readings[1].values()) == {1}, readings[1]
    assert set(readings[2].values()) == {2}, readings[2]
    assert set(readings[2]) == {"stem", "stem.conv1", "stem.norm1", "stem.act1"}
