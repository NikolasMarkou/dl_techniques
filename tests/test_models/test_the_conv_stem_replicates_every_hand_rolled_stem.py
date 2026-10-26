"""``ConvStem`` replicates every stem it replaced, layer for layer.

HEAD ``20b48aba6`` moved ten hand-rolled stems onto ``ConvStem`` and the move
was not exact: ResNet's ``MaxPooling2D(3, strides=2, padding="same")`` became
a stride-3 ``valid`` pool (224 px -> 37x37 instead of 56x56), and no test saw
it because every comparison in the tree was between the NEW code and itself.
This file compares the stem a model builds against the ORIGINAL layers, rebuilt
inline here from the pre-migration source (``git show 20b48aba6^:<path>``) with
every constructor argument written out as a literal, never read back from the
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
  ``norm1``, ``activation1``, ``conv2``, ``norm2``, ``activation2``, ``pool``;
  omit a role the original had no layer for) and ``forward(x, training)``
  chaining them exactly as the original ``call()`` did.
* Write a ``_<model>_model()`` factory returning a built-able model whose
  ``.stem`` is the ``ConvStem`` under test.
* Append ONE ``StemCase`` to ``CASES`` and, when the model has a ``stem``
  attribute, ONE row to ``BUILT_STEM_ROWS`` (``test_every_migrated_model_
  builds_its_stem``). Nothing else in this file changes.

Planted controls prove the harness can see each defect class it exists for:
a wrong pool, a wrong epsilon and a norm applied twice are each handed to the
SAME comparison and the comparison must FAIL.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations import create_activation_layer
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.vision.cliffordnet.hierarchical import (
    HierarchicalCliffordNet,
)
from dl_techniques.models.vision.cliffordnet.model import CliffordNet
from dl_techniques.models.vision.resnet.model import ResNet
from tests.numerics import true_float32_matmul



@pytest.fixture(autouse=True)
def _true_float32():
    """Scope TF32 off so the exact comparisons are made in true float32 on a
    TF32-capable GPU as well as on CPU (restored, and asserted, on exit)."""
    with true_float32_matmul():
        yield

BATCH = 2

#: Sub-layer attributes of ``ConvStem`` a case may supply, in forward order.
ROLES = ("conv1", "norm1", "activation1", "conv2", "norm2", "activation2", "pool")


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
    """

    sublayers: Dict[str, keras.layers.Layer]
    forward: Callable[[object, Optional[bool]], object]


@dataclass
class StemCase:
    """One original stem variant.

    :param case_id: pytest id.
    :param make_model: returns a fresh model whose ``.stem`` is a ``ConvStem``.
    :param make_original: returns the ``Original`` (fresh layers every call).
    :param input_shape: ``(H, W, C)`` fed to the stem.
    :param expected_output_shape: ``(H', W', C')`` the ORIGINAL stem yields, so
        a geometry defect is named by its shape, not only by its values.
    """

    case_id: str
    make_model: Callable[[], keras.Model]
    make_original: Callable[[], Original]
    input_shape: Tuple[int, int, int]
    expected_output_shape: Tuple[int, int, int]


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
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((BATCH,) + tuple(case.input_shape)).astype("float32")

    original = case.make_original()
    original.forward(x, False)  # builds the original sub-layers
    for layer in original.sublayers.values():
        _randomize(layer, rng)

    # Build the stem BEFORE touching its sub-layers: building one first makes a
    # later ``stem.build`` raise "already built".
    if not stem.built:
        stem.build(x.shape)

    stem_layers = _stem_sublayers(stem)
    assert set(stem_layers) == set(original.sublayers), (
        f"{case.case_id}: the stem's sub-layer roles {sorted(stem_layers)} differ "
        f"from the original's {sorted(original.sublayers)}"
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


# ---------------------------------------------------------------------
# ResNet (imagenet and cifar stems)
# ---------------------------------------------------------------------

# Original: ``git show 20b48aba6^:src/dl_techniques/models/vision/resnet/model.py``
# ``_build_stem`` (Conv2D he_normal, no bias, padding same -> BN via the norm
# factory with momentum 0.9 -> factory ReLU -> MaxPooling2D(3, strides=2,
# padding="same") for the imagenet stem only).


def _resnet_original(stem_type: str) -> Original:
    imagenet = stem_type == "imagenet"
    conv = keras.layers.Conv2D(
        filters=64,
        kernel_size=7 if imagenet else 3,
        strides=2 if imagenet else 1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
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

    sublayers = {"conv1": conv, "norm1": norm, "activation1": act}
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
# ``activation1`` role the ConvStem has.

#: Non-default conv knobs the models are built with, so a dropped one is seen.
_CLIFFORD_KERNEL_REG = 1e-4
_CLIFFORD_BIAS_REG = 3e-5
_CLIFFORD_BIAS_INIT = 0.25


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
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
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
                "activation1": silu,
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
]


# ---------------------------------------------------------------------
# The comparison on every real case
# ---------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_the_model_stem_replicates_the_original(case):
    """The stem a model builds equals the pre-migration stem, weight for weight."""
    model = case.make_model()
    assert_stem_matches_original(case, model.stem)


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
    config = case.make_model().stem.get_config()
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
    if stem.activation1 is not None:
        x = stem.activation1(x)
    if stem.pool is not None:
        x = stem.pool(x)
    return x


def test_control_the_harness_sees_a_wrong_pool():
    """Stride 3 / ``valid`` (the HEAD defect) must make the comparison fail."""
    stem = _stem_from_config(RESNET_IMAGENET, pool_strides=None, pool_padding="valid")
    with pytest.raises(AssertionError, match="shape"):
        assert_stem_matches_original(RESNET_IMAGENET, stem)


def test_control_the_harness_sees_a_wrong_epsilon():
    """BN epsilon 1e-3 against the original's 1e-6 must make the comparison fail."""
    stem = _stem_from_config(
        RESNET_IMAGENET, normalization_kwargs={"epsilon": 1e-3}
    )
    assert stem.norm1.epsilon == pytest.approx(1e-3)
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(RESNET_IMAGENET, stem)


def test_control_the_harness_sees_a_norm_applied_twice():
    """A norm in the forward twice must make the comparison fail."""
    model = RESNET_IMAGENET.make_model()
    with pytest.raises(AssertionError, match="differs from the original"):
        assert_stem_matches_original(
            RESNET_IMAGENET, model.stem, forward=_norm_applied_twice
        )


def test_control_a_correct_stem_survives_the_same_harness():
    """Anti-vacuity twin: the controls above fail for their defect, not because
    the harness rejects everything. A stem rebuilt from the model's own config
    with no override passes.
    """
    stem = _stem_from_config(RESNET_IMAGENET)
    assert_stem_matches_original(RESNET_IMAGENET, stem)


# ---------------------------------------------------------------------
# Every migrated model runs its stem
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_model,input_shape",
    [(row[1], row[2]) for row in BUILT_STEM_ROWS],
    ids=[row[0] for row in BUILT_STEM_ROWS],
)
def test_every_migrated_model_builds_its_stem(make_model, input_shape):
    """After a forward, ``model.stem`` and every sub-layer of it are built.

    HEAD's ResNet, ConvNeXt and RAD ran their stems through aliases of the
    sub-layers, so the ``ConvStem`` object itself was never called and stayed
    unbuilt (``stem.built is False``) under a green suite.
    """
    model = make_model()
    model(np.zeros((1,) + tuple(input_shape), "float32"), training=False)
    assert model.stem.built is True, "model.stem was never built: call() bypasses it"
    unbuilt = [
        layer.name
        for layer in model.stem._flatten_layers(include_self=False)
        if not layer.built
    ]
    assert unbuilt == [], f"unbuilt sub-layers of the stem: {unbuilt}"
    assert len(model.stem.weights) > 0
