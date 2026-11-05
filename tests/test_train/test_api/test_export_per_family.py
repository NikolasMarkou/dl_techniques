"""``export`` on every model family, not only ``vit_pico``: stochastic depth, select ops, onnx.

``test_export_roundtrip.py`` pins the formats on one ViT, which has no stochastic layer and
needs no TensorFlow operator the TFLite built-ins lack. This file closes both gaps (review of
iteration 3, concerns 1 and 2). Six families (vit, resnet, convnext_v1, convnext_v2,
cliffordnet, cliffordnet_hier) are built small (16x16 inputs, four classes), with stochastic
depth on at 0.5 and, for the CliffordNet families, ``layer_scale_init=1.0`` so that a live
drop path moves the logits by far more than the tolerance (at the default 1e-5 it moves them
by 5e-6 and no test could see it).

The claims:

* every family exports to all four formats and the artifact, read back by its own runtime,
  gives the Keras model's ``training=False`` logits (``ATOL`` 1e-4; the largest difference
  measured over the run is printed in D-066);
* a ``saved_model`` of a family with stochastic depth is REPEATABLE: two ``serve`` calls are
  equal, and equal Keras (the traced call is ``training=False``; ``model.export`` traced
  ``training=None``, which left drop path live and made ``serve`` random);
* the ``onnx`` arm on the same families fails if the function is traced ``training=True``;
* a model that needs ``Roll`` converts to TFLite only through select ops, the result is
  checked against Keras, and a WARNING says it needs the Flex runtime; a converter that fails
  with select ops gives one readable :class:`ConfigError` and no MLIR dump.

Everything writes under ``tmp_path`` or pytest's tmp factory. Nothing is deleted.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.cliffordnet.hierarchical import create_hierarchical_cliffordnet
from dl_techniques.models.vision.cliffordnet.model import create_cliffordnet
from dl_techniques.models.vision.convnext.convnext_v1 import create_convnext_v1
from dl_techniques.models.vision.convnext.convnext_v2 import create_convnext_v2
from dl_techniques.models.vision.resnet.model import create_resnet
from dl_techniques.models.vision.vit.model import create_vit
from train.api import exporting
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, TFLITE_SIBLING, export

ATOL = 1e-4
SHAPE = (16, 16, 3)
CLASSES = 4

#: name -> (builder, has stochastic depth that is live under ``training=True``)
FAMILIES = {
    "vit": (lambda: create_vit("vit_pico", num_classes=CLASSES, input_shape=SHAPE, patch_size=4), False),
    "resnet": (lambda: create_resnet("resnet18", num_classes=CLASSES, input_shape=SHAPE), False),
    "convnext_v1": (
        lambda: create_convnext_v1(
            "cifar10", num_classes=CLASSES, input_shape=SHAPE, strides=2, drop_path_rate=0.5
        ),
        True,
    ),
    "convnext_v2": (
        lambda: create_convnext_v2(
            "atto", num_classes=CLASSES, input_shape=SHAPE, strides=2, drop_path_rate=0.5
        ),
        True,
    ),
    "cliffordnet": (
        lambda: create_cliffordnet(
            "nano", num_classes=CLASSES, input_shape=SHAPE, stochastic_depth_rate=0.5,
            layer_scale_init=1.0, depth=4,
        ),
        True,
    ),
    "cliffordnet_hier": (
        lambda: create_hierarchical_cliffordnet(
            "hier_345", num_classes=CLASSES, input_shape=SHAPE, stochastic_depth_rate=0.5,
            stage_depths=(1, 2, 1), layer_scale_init=1.0,
        ),
        True,
    ),
}
#: The families whose graph holds ``Roll``, which TFLite's built-ins lack.
NEEDS_SELECT_OPS = ("cliffordnet", "cliffordnet_hier")
NAMES = {"keras": "m.keras", "saved_model": "m_saved_model", "tflite": "m.tflite", "onnx": "m.onnx"}


@pytest.fixture(scope="module")
def batch():
    return np.random.default_rng(0).random((3, *SHAPE)).astype(np.float32)


def wake_the_branches(net):
    """Set the ConvNeXt layer scales and GRN gammas live, the way the CliffordNet rows use ``layer_scale_init=1.0``.

    At their initial values the branches are invisible: V1's layer scale starts at the paper's 1e-6
    (a live drop path moved the logits by 2e-7, plan 98b866fb step 10b) and V2's GRN gamma starts at
    zero, so GRN is the identity (step 10c). A drop path, an export or a runtime that got the branch
    wrong could not show on such a model. The reference and the artifact are both made from the
    woken model.
    """
    for w in net.weights:
        if w.path.endswith("gamma_scale/gamma"):
            w.assign(keras.ops.ones_like(w))
        elif w.path.endswith("global_response_norm/gamma"):
            w.assign(0.5 * keras.ops.ones_like(w))


class Collected(logging.Handler):
    """Collects the library logger's records while an export runs."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class Workshop:
    """Builds each family once and exports each (family, format) once, keeping the log."""

    def __init__(self, root):
        self.root = Path(root)
        self.models = {}
        self.written = {}
        self.records = {}

    def model(self, family):
        if family not in self.models:
            net = FAMILIES[family][0]()
            net(np.zeros((1, *SHAPE), np.float32))
            if family in ("convnext_v1", "convnext_v2"):
                wake_the_branches(net)
            self.models[family] = net
        return self.models[family]

    def artifact(self, family, format):
        key = (family, format)
        if key not in self.written:
            if format == "onnx":
                pytest.importorskip("tf2onnx")
                pytest.importorskip("onnxruntime")
            out = self.root / family / NAMES[format]
            handler = Collected()
            logger = logging.getLogger("dl")
            logger.addHandler(handler)
            try:
                self.written[key] = export(
                    model=self.model(family), out=out, format=format, input_shape=SHAPE
                )
            finally:
                logger.removeHandler(handler)
            self.records[key] = handler.records
        return self.written[key]


@pytest.fixture(scope="module")
def workshop(tmp_path_factory):
    return Workshop(tmp_path_factory.mktemp("families"))


def read_back(format, path, batch):
    """The artifact's output on ``batch``, from the runtime the format is for."""
    if format == "keras":
        return keras.models.load_model(path, compile=False)(batch, training=False)
    if format == "saved_model":
        import tensorflow as tf

        loaded = tf.saved_model.load(str(path))  # the object owns the variables: keep it alive
        return loaded.serve(batch)
    if format == "tflite":
        return exporting._tflite_logits(Path(path).read_bytes(), batch)
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: batch})[0]


def reference(model, batch):
    return np.asarray(model(batch, training=False), np.float64)


# ---------------------------------------------------------------------
# Every family, every format
# ---------------------------------------------------------------------


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_exports_each_format_and_matches_keras(family, format, workshop, batch):
    expected = reference(workshop.model(family), batch)
    assert np.abs(expected).max() > 1e-3  # a model that outputs nothing would match anything
    got = np.asarray(read_back(format, workshop.artifact(family, format), batch), np.float64)
    assert got.shape == expected.shape
    worst = np.abs(got - expected).max()
    assert worst <= ATOL, f"{family} {format}: {worst:.3g}"


@pytest.mark.parametrize("family", FAMILIES)
def test_the_stochastic_families_really_are_stochastic_under_training_true(family, workshop, batch):
    """Control: without this, a repeatable export of a family with no randomness proves nothing."""
    model = workshop.model(family)
    stochastic = FAMILIES[family][1]
    saved = model.get_weights()  # a training=True pass moves batch-norm statistics: put them back
    runs = [np.asarray(model(batch, training=True)) for _ in range(4)]
    model.set_weights(saved)
    spread = max(np.abs(runs[0] - other).max() for other in runs[1:])
    assert (spread > 1e-3) == stochastic, f"{family}: training=True spread {spread:.3g}"


@pytest.mark.parametrize("family", [name for name, (_, live) in FAMILIES.items() if live])
def test_a_saved_model_with_stochastic_depth_serves_the_same_answer_every_time(family, workshop, batch):
    import tensorflow as tf

    path = workshop.artifact(family, "saved_model")
    loaded = tf.saved_model.load(str(path))
    first, second = loaded.serve(batch).numpy(), loaded.serve(batch).numpy()
    np.testing.assert_array_equal(first, second)  # exactly: nothing random is left in the graph
    np.testing.assert_allclose(first, reference(workshop.model(family), batch), atol=ATOL, rtol=0)
    other = np.random.default_rng(4).random((5, *SHAPE)).astype(np.float32)  # the batch stays free
    np.testing.assert_allclose(
        loaded.serve(other).numpy(), reference(workshop.model(family), other), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("family", [name for name, (_, live) in FAMILIES.items() if live])
def test_the_graph_of_a_saved_model_holds_no_random_op(family, workshop):
    from tensorflow.core.protobuf import saved_model_pb2

    proto = saved_model_pb2.SavedModel()
    proto.ParseFromString((workshop.artifact(family, "saved_model") / "saved_model.pb").read_bytes())
    ops = {
        node.op
        for meta in proto.meta_graphs
        for function in meta.graph_def.library.function
        for node in function.node_def
    }
    assert not {op for op in ops if "Random" in op or "Dropout" in op}, sorted(ops)


# ---------------------------------------------------------------------
# TFLite and select operators
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", list(FAMILIES))
def test_tflite_warns_about_the_flex_runtime_exactly_for_the_models_that_need_it(family, workshop):
    workshop.artifact(family, "tflite")
    warnings = [
        r.getMessage() for r in workshop.records[(family, "tflite")] if r.levelno >= logging.WARNING
    ]
    if family in NEEDS_SELECT_OPS:
        assert len(warnings) == 1 and "Flex" in warnings[0] and "select TF operators" in warnings[0], warnings
    else:
        assert warnings == []


def roll_model():
    """A functional model with one ``Roll``, which TFLite's built-in operators lack."""
    inputs = keras.Input((6, 6, 3))
    rolled = keras.layers.Lambda(lambda t: keras.ops.roll(t, 1, axis=1))(inputs)
    outputs = keras.layers.Dense(4)(keras.layers.Flatten()(rolled))
    return keras.Model(inputs, outputs)


def test_a_model_that_needs_select_ops_converts_through_them_and_matches_keras(tmp_path):
    net = roll_model()
    handler = Collected()
    logger = logging.getLogger("dl")
    logger.addHandler(handler)
    try:
        out = export(model=net, out=tmp_path / "r.tflite", format="tflite")
    finally:
        logger.removeHandler(handler)
    # The built-in attempt was made first and failed (info), then the select-ops one succeeded.
    texts = [(r.levelno, r.getMessage()) for r in handler.records]
    assert any(level == logging.INFO and "built-in operators cannot express" in text for level, text in texts)
    assert [text for level, text in texts if level == logging.WARNING] == [
        f"{out} needs select TF operators, so it runs only where the Flex delegate is available "
        "(the tensorflow pip package has it; a plain TFLite runtime does not)"
    ]
    x = np.random.default_rng(3).random((2, 6, 6, 3)).astype(np.float32)
    np.testing.assert_allclose(
        exporting._tflite_logits(out.read_bytes(), x), np.asarray(net(x), np.float64), atol=1e-6, rtol=0
    )
    # Liveness: the file really holds a Flex operator (a built-ins-only file could not).
    assert b"FlexRoll" in out.read_bytes()


def test_a_converter_that_fails_even_with_select_ops_is_one_readable_error_and_a_retry_hint(
    tmp_path, monkeypatch
):
    import tensorflow as tf
    from tensorflow.lite.python.convert_phase import ConverterError

    dump = (
        'Could not translate MLIR to FlatBuffer.<unknown>:0: error: loc(callsite(fused["Roll:", "x"] at fused["StatefulPartitionedCall:", "y"])): '
        "'tf.Roll' op is neither a custom op nor a flex op\n"
        + "\n".join(f"  mlir line {i} %0 = tf.Roll" for i in range(400))
    )

    def refuse():
        raise ConverterError(dump)

    # `from_saved_model` returns a private converter class, so the factory is what is replaced.
    stub = SimpleNamespace(target_spec=SimpleNamespace(), convert=refuse)
    monkeypatch.setattr(tf.lite.TFLiteConverter, "from_saved_model", staticmethod(lambda path: stub))
    out = tmp_path / "r.tflite"
    with pytest.raises(ConfigError) as caught:
        export(model=roll_model(), out=out, format="tflite")
    message = str(caught.value)
    assert "the converter rejected the operator(s) Roll)" in message  # the call wrapper is not an operator
    assert "mlir line" not in message and len(message) < 700 and "\n" not in message
    assert "move it away before trying again" in message
    sibling = tmp_path / ("r.tflite" + TFLITE_SIBLING)
    assert sibling.is_dir() and not out.exists()
    with pytest.raises(ConfigError, match="already exists; export never overwrites or deletes, so move it away"):
        export(model=roll_model(), out=out, format="tflite")


def test_a_select_ops_file_that_disagrees_with_keras_is_not_written(tmp_path, monkeypatch):
    real = exporting._tflite_logits
    monkeypatch.setattr(exporting, "_tflite_logits", lambda blob, batch: real(blob, batch) + 0.5)
    out = tmp_path / "r.tflite"
    with pytest.raises(ConfigError, match=r"differs from the Keras model by 0\.5"):
        export(model=roll_model(), out=out, format="tflite")
    assert not out.exists()


def test_the_documentation_states_the_input_contract_and_the_mixed_precision_onnx_gap():
    doc = exporting.__doc__
    assert "RAW LOGITS" in doc and "(pixels - mean) / std" in doc and "task.json" in doc
    assert "mixed_float16" in doc and "1e-4" in doc and "onnx" in doc
