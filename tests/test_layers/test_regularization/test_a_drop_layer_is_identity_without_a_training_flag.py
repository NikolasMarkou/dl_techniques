"""A drop-path layer called without a training flag is the identity, as Keras ``Dropout`` is.

Keras ``Dropout.call`` is ``if training and self.rate > 0`` (keras/src/layers/regularization/dropout.py),
timm's ``drop_path`` is ``if drop_prob == 0. or not training`` (timm/layers/drop.py) and
torchvision's ``stochastic_depth`` is ``if not training or p == 0.0``. A direct ``model(x)`` hands
``training=None`` down, so a layer that tested ``training is False`` kept dropping paths on it
(a ConvNeXt V1 ``cifar10`` model moved by 0.093 on its outputs between two identical calls).

Scope: ``StochasticDepth``, ``StochasticGradient`` and ``FractalBlock``'s join (the one other layer
under ``layers/`` that gated a train-only random op on ``training is False``, found by the AST census
of step 10f of plan-2026-11-03T221936-98b866fb), three whole models that contain them, and a source
guard that no layer under ``layers/`` tests ``training is False`` again.
"""

import ast
import pathlib

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.conv_blocks.conv_block import ConvBlock
from dl_techniques.layers.fractal_block import FractalBlock
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.regularization.stochastic_gradient import StochasticGradient
from dl_techniques.models.vision.cliffordnet import CliffordNet, HierarchicalCliffordNet
from dl_techniques.models.vision.convnext.convnext_v1 import create_convnext_v1

LAYERS_DIR = pathlib.Path(__file__).resolve().parents[3] / "src" / "dl_techniques" / "layers"


def _x(rows: int = 256, features: int = 8) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((rows, features)).astype("float32")


def _numpy(t) -> np.ndarray:
    return keras.ops.convert_to_numpy(t)


@pytest.mark.parametrize("cls", [StochasticDepth, StochasticGradient])
@pytest.mark.parametrize("flag", ["absent", None, False])
def test_a_drop_layer_returns_its_input_exactly_unless_training_is_truthy(cls, flag):
    layer = cls(drop_path_rate=0.9)
    x = _x()
    out = layer(x) if flag == "absent" else layer(x, training=flag)
    np.testing.assert_array_equal(_numpy(out), x)


def test_stochastic_gradient_never_severs_a_gradient_without_a_flag():
    """A severed gradient is an all-zero gradient; 60 draws at rate 0.9 sever one with certainty."""
    layer = StochasticGradient(drop_path_rate=0.9)
    x = tf.constant(_x(4, 3))
    for _ in range(60):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(layer(x))
        np.testing.assert_array_equal(tape.gradient(loss, x).numpy(), np.ones((4, 3), "float32"))


def test_stochastic_depth_still_drops_rows_when_training_is_true():
    layer = StochasticDepth(drop_path_rate=0.5)
    x = np.ones((256, 4), "float32")
    out = _numpy(layer(x, training=True))
    row = out[:, 0]
    assert 0 < int((row == 0.0).sum()) < 256, "a rate of 0.5 over 256 rows must drop some and keep some"
    np.testing.assert_allclose(row[row != 0.0], 2.0)


def test_stochastic_gradient_still_severs_gradients_when_training_is_true():
    layer = StochasticGradient(drop_path_rate=0.9)
    x = tf.constant(_x(4, 3))
    severed = 0
    for _ in range(60):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(layer(x, training=True))
        grad = tape.gradient(loss, x)
        severed += int(grad is None or not np.any(grad.numpy()))
    assert severed > 0, "training=True at rate 0.9 must sever the gradient on some of 60 draws"


def _fractal_block(rate: float) -> FractalBlock:
    config = ConvBlock(filters=4, kernel_size=3, strides=1, padding="same", use_pooling=False).get_config()
    return FractalBlock(block_config=config, depth=2, drop_path_rate=rate)


def test_a_fractal_block_called_without_a_flag_equals_the_same_call_with_training_false():
    block = _fractal_block(0.9)
    x = np.random.default_rng(1).standard_normal((64, 6, 6, 3)).astype("float32")
    inference = _numpy(block(x, training=False))
    np.testing.assert_array_equal(_numpy(block(x)), inference)
    np.testing.assert_array_equal(_numpy(block(x, training=None)), inference)


def test_the_fractal_join_is_the_plain_mean_unless_training_is_truthy_and_drops_branches_when_it_is():
    """Called on the join itself: the block's BatchNormalization would make ``training=True`` differ from inference anyway,
    so a whole-block comparison could not tell a live join from a dead one."""
    block = _fractal_block(0.9)
    rng = np.random.default_rng(2)
    deep, shallow = (rng.standard_normal((64, 6, 6, 3)).astype("float32") for _ in range(2))
    mean = (deep + shallow) * np.float32(0.5)
    for flag in ("absent", None, False):
        joined = block._join(deep, shallow) if flag == "absent" else block._join(deep, shallow, training=flag)
        np.testing.assert_array_equal(_numpy(joined), mean)
    first, second = (_numpy(block._join(deep, shallow, training=True)) for _ in range(2))
    assert not np.array_equal(first, mean) and not np.array_equal(first, second), "training=True must drop branches"


def _convnext_cifar10():
    return create_convnext_v1("cifar10", num_classes=10, input_shape=(32, 32, 3), strides=2, drop_path_rate=0.3, dropout_rate=0.1)


def _cliffordnet_nano():
    return CliffordNet.from_variant("nano", num_classes=10, stochastic_depth_rate=0.3)


def _cliffordnet_hier_345():
    return HierarchicalCliffordNet.from_variant("hier_345", num_classes=10, stochastic_depth_rate=0.1)


@pytest.mark.parametrize("build", [_convnext_cifar10, _cliffordnet_nano, _cliffordnet_hier_345],
                         ids=["convnext_v1/cifar10", "cliffordnet/nano", "cliffordnet_hier/hier_345"])
def test_a_model_called_without_a_flag_equals_the_same_call_with_training_false(build):
    model = build()
    x = np.random.default_rng(0).uniform(size=(8, 32, 32, 3)).astype("float32")
    reference = _numpy(model(x, training=False))
    np.testing.assert_array_equal(_numpy(model(x)), reference)
    np.testing.assert_array_equal(_numpy(model(x)), reference)


def _compares_training_to_false(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "training"):
        return False
    return any(isinstance(op, (ast.Is, ast.Eq)) and isinstance(c, ast.Constant) and c.value is False
               for op, c in zip(node.ops, node.comparators))


def test_no_layer_gates_a_train_only_random_op_on_training_being_exactly_false():
    """The predicate ``training is False`` is the defect; ``if not training`` / ``if training`` are the rule."""
    offenders = []
    for path in sorted(LAYERS_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if _compares_training_to_false(node):
                offenders.append(f"{path.relative_to(LAYERS_DIR)}:{node.lineno}")
    assert not offenders, f"`training is False` gates a branch in: {offenders}"
