"""``export`` on the two segmentation families, in every format, at a non-square size (plan 2faf890a step 31, D-081).

``test_export_per_family.py`` pins the formats on the six classifiers. This file is its segmentation
counterpart: the output of a segmentation model is a score map ``(batch, height, width, classes)``, and
the operations that make it (resizes, transposed convolutions, concatenations) are exactly the ones a
converter may translate differently. Measured first (``scratch_step31/logs``): every cell of the table
below exports and matches Keras to about 1e-6, so this file asserts the NUMBER and no cell is a refusal;
the one refusal a segmentation model can meet is the shared bfloat16 ``tflite`` one.

The families are built small and at 32x64 (the sides differ), with their output head scaled so that the
scores have a scale near 1 (AccUNet's untrained logits are 3e-5, and a tolerance of 1e-4 would pass any
file against them):

* every (family, format) exports through ``export(..., record=)`` and the artifact, read by its own
  runtime, gives Keras' ``training=False`` scores to ``ATOL`` (1e-4; the largest difference measured over
  the run is in D-081), with the input's height and width and one score per class;
* two calls of an artifact give the same bytes of answer;
* a ``saved_model`` of ConvUNext, which has stochastic depth that is live under ``training=True`` (a
  control asserts it), serves the same answer every time, matches Keras for another batch size, and holds
  no random op in its graph (``model.export`` traces ``training=None``, which leaves it live);
* ``<out>.task.json`` is the run's version 2 record, byte for byte, and ``tflite`` needs no Flex operator;
* ``python -m train export`` of a segmentation run writes only the path on stdout, and the exported
  ``.keras`` file reloads through ``load_model`` as a ``SegmentationPredictor`` that answers the same masks
  as the run;
* the help of ``export`` says one score per class per pixel.

Everything writes under pytest's tmp factory. Nothing is deleted.
"""

import json
import logging
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.api import cli, exporting, recipes, registry
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, export
from train.api.predictor import (
    RECORD_KEYS, SEGMENTATION, build_segmentation_record, load_model, overrides_of_variant,
)
from train.api.segmentation_predictor import SegmentationPredictor

ATOL = 1e-4
SIZE = (32, 64)  # height != width
SHAPE = (*SIZE, 3)
CLASSES = 3

#: family -> (variant, model_args, the layer that makes the scores, whether stochastic depth is live)
FAMILIES = {
    "convunext": ("tiny", {"drop_path_rate": 0.5}, "final_output", True),
    "accunet": ("base", {"base_filters": 8, "mlfc_iterations": 1}, "output_conv", False),
}
NAMES = {"keras": "m.keras", "saved_model": "m_saved_model", "tflite": "m.tflite", "onnx": "m.onnx"}


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


@pytest.fixture(scope="module")
def batch():
    return np.random.default_rng(0).random((3, *SHAPE)).astype(np.float32)


def build(family):
    """The family's model with its weights moved off the initial ones and its scores at a scale near 1."""
    variant, args, head, _ = FAMILIES[family]
    model = recipes.build_model(registry.MODELS[family], variant, CLASSES, SHAPE, args)
    rng = np.random.default_rng(1)
    model.set_weights([
        w + rng.normal(0.0, 0.02, w.shape).astype(np.float32) if w.ndim > 1 else w for w in model.get_weights()
    ])
    probe = np.random.default_rng(2).random((2, *SHAPE)).astype(np.float32)
    scale = float(np.abs(np.asarray(model(probe, training=False))).max())
    (layer,) = [  # AccUNet names every convolution of its blocks `output_conv`; the head is the one with the classes
        l for l in model._flatten_layers(include_self=False, recursive=True)
        if l.name == head and getattr(l, "filters", None) == CLASSES
    ]
    layer.set_weights([w / scale for w in layer.get_weights()])
    return model


def record_of(family):
    variant, args = FAMILIES[family][:2]
    table = registry.MODELS[family].load_variants()[variant]
    return build_segmentation_record(
        model=f"{family}/{variant}", family=family, variant=variant, class_names=["pet", "background", "border"],
        input_size=SIZE, channels=3, normalize="unit", mean=[0.0] * 3, std=[1.0] * 3,
        variant_overrides=overrides_of_variant(table, args),
    )


class Collected(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class Workshop:
    """Builds each family once and exports each (family, format) once, keeping the log."""

    def __init__(self, root):
        self.root = Path(root)
        self.models, self.written, self.records = {}, {}, {}

    def model(self, family):
        if family not in self.models:
            self.models[family] = build(family)
        return self.models[family]

    def artifact(self, family, format):
        key = (family, format)
        if key not in self.written:
            if format == "onnx":
                pytest.importorskip("tf2onnx")
                pytest.importorskip("onnxruntime")
            handler = Collected()
            logger = logging.getLogger("dl")
            logger.addHandler(handler)
            try:
                self.written[key] = export(
                    model=self.model(family), out=self.root / family / NAMES[format], format=format,
                    input_shape=SHAPE, record=record_of(family),
                )
            finally:
                logger.removeHandler(handler)
            self.records[key] = handler.records
        return self.written[key]


@pytest.fixture(scope="module")
def workshop(world, tmp_path_factory):
    return Workshop(tmp_path_factory.mktemp("seg_export"))


def read_back(format, path, x):
    """The artifact's output on ``x`` from the runtime the format is for."""
    if format == "keras":
        return keras.models.load_model(path, compile=False)(x, training=False)
    if format == "saved_model":
        loaded = tf.saved_model.load(str(path))  # the object owns the variables: keep it alive
        return loaded.serve(x)
    if format == "tflite":
        return exporting._tflite_logits(Path(path).read_bytes(), x)
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def runtime(format, path):
    """A callable that runs the artifact in its own runtime. ONE runtime object answers every call.

    That matters for the randomness check: a freshly loaded SavedModel or interpreter starts its random
    generator from the same state, so two fresh loads give the same draw even when the graph is random.
    """
    if format == "keras":
        model = keras.models.load_model(path, compile=False)
        return lambda x: model(x, training=False)
    if format == "saved_model":
        loaded = tf.saved_model.load(str(path))  # the closure keeps it alive
        return lambda x: loaded.serve(x)
    if format == "tflite":
        interpreter = tf.lite.Interpreter(model_content=Path(path).read_bytes())
        entry = interpreter.get_input_details()[0]

        def call(x):
            interpreter.resize_tensor_input(entry["index"], x.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(entry["index"], x)
            interpreter.invoke()
            return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

        return call
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda x: session.run(None, {session.get_inputs()[0].name: x})[0]


def reference(model, x):
    with tf.device("/CPU:0"):
        return np.asarray(model(x, training=False), np.float64)


# ---------------------------------------------------------------------
# The table: family x format
# ---------------------------------------------------------------------

@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_exports_each_format_and_matches_keras_at_a_non_square_size(family, format, workshop, batch):
    expected = reference(workshop.model(family), batch)
    assert expected.shape == (3, *SIZE, CLASSES)
    assert np.abs(expected).max() > 0.5  # a model that outputs nothing would match anything
    got = np.asarray(read_back(format, workshop.artifact(family, format), batch), np.float64)
    assert got.shape == expected.shape
    worst = np.abs(got - expected).max()
    assert worst <= ATOL, f"{family} {format}: {worst:.3g}"
    assert (got.argmax(-1) == expected.argmax(-1)).mean() > 0.999


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("family", FAMILIES)
def test_an_artifact_answers_the_same_twice(family, format, workshop, batch):
    call = runtime(format, workshop.artifact(family, format))
    first, second = np.asarray(call(batch)), np.asarray(call(batch))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_stochastic_family_really_is_stochastic_under_training_true(family, workshop, batch):
    """Control: without this, a repeatable export of a family with no randomness proves nothing."""
    model = workshop.model(family)
    runs = [np.asarray(model(batch, training=True)) for _ in range(4)]
    spread = max(np.abs(runs[0] - other).max() for other in runs[1:])
    assert (spread > 1e-6) == FAMILIES[family][3], f"{family}: training=True spread {spread:.3g}"


def test_a_saved_model_with_stochastic_depth_serves_the_same_answer_every_time(workshop, batch):
    loaded = tf.saved_model.load(str(workshop.artifact("convunext", "saved_model")))
    first, second = loaded.serve(batch).numpy(), loaded.serve(batch).numpy()
    np.testing.assert_array_equal(first, second)  # exactly: nothing random is left in the graph
    np.testing.assert_allclose(first, reference(workshop.model("convunext"), batch), atol=ATOL, rtol=0)
    other = np.random.default_rng(4).random((5, *SHAPE)).astype(np.float32)  # the batch stays free
    np.testing.assert_allclose(
        loaded.serve(other).numpy(), reference(workshop.model("convunext"), other), atol=ATOL, rtol=0
    )


def test_the_graph_of_a_saved_model_holds_no_random_op(workshop):
    from tensorflow.core.protobuf import saved_model_pb2

    proto = saved_model_pb2.SavedModel()
    proto.ParseFromString((workshop.artifact("convunext", "saved_model") / "saved_model.pb").read_bytes())
    ops = {
        node.op
        for meta in proto.meta_graphs
        for function in meta.graph_def.library.function
        for node in function.node_def
    }
    assert not {op for op in ops if "Random" in op or "Dropout" in op}, sorted(ops)


@pytest.mark.parametrize("family", FAMILIES)
def test_tflite_needs_no_flex_operator_for_a_segmentation_family(family, workshop):
    path = workshop.artifact(family, "tflite")
    warnings = [r.getMessage() for r in workshop.records[(family, "tflite")] if r.levelno >= logging.WARNING]
    assert warnings == [] and b"Flex" not in path.read_bytes()


# ---------------------------------------------------------------------
# The record beside the file
# ---------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_the_record_beside_an_export_is_the_runs_version_2_record(family, workshop):
    written = json.loads(Path(str(workshop.artifact(family, "keras")) + ".task.json").read_text())
    assert written == record_of(family)
    assert list(written)[:len(RECORD_KEYS[SEGMENTATION])] == list(RECORD_KEYS[SEGMENTATION])
    assert list(written)[len(RECORD_KEYS[SEGMENTATION]):] == ["variant_overrides"]
    assert written["schema_version"] == 2 and written["input_size"] == [32, 64] and "input_shape" not in written


def test_a_bfloat16_segmentation_model_is_refused_as_tflite_by_name(world, tmp_path):
    keras.config.set_dtype_policy("mixed_bfloat16")
    try:
        model = recipes.build_model(registry.MODELS["convunext"], "tiny", CLASSES, SHAPE, {})
    finally:
        keras.config.set_dtype_policy("float32")
    assert model.compute_dtype == "bfloat16"
    out = tmp_path / "m.tflite"
    with pytest.raises(ConfigError, match="cannot carry a model that computes in bfloat16"):
        export(model=model, out=out, format="tflite", input_shape=SHAPE)
    assert not out.exists() and not (tmp_path / ("m.tflite" + exporting.TFLITE_SIBLING)).exists()


# ---------------------------------------------------------------------
# The command, on a run directory
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_dir(workshop, tmp_path_factory):
    """A run directory of the ConvUNext fixture: its best model and its record."""
    directory = tmp_path_factory.mktemp("seg_run") / "run"
    directory.mkdir()
    workshop.model("convunext").save(directory / "best_model.keras")
    (directory / "task.json").write_text(json.dumps(record_of("convunext"), indent=2) + "\n")
    return directory


def test_the_command_writes_only_the_path_and_the_exported_keras_file_predicts_the_runs_masks(run_dir, tmp_path, capsys):
    out = tmp_path / "exported.keras"
    assert cli.main(["export", f"model={run_dir}", f"out={out}", "format=keras"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    assert json.loads(Path(str(out) + ".task.json").read_text()) == record_of("convunext")
    image = np.random.default_rng(5).integers(0, 256, (50, 41, 3), dtype=np.uint8)
    from_run = load_model(run_dir).predict(image)
    exported = load_model(out)
    assert isinstance(exported, SegmentationPredictor)
    np.testing.assert_array_equal(exported.predict(image)["masks"][0], from_run["masks"][0])
    assert from_run["masks"][0].shape == (50, 41)


def test_the_command_exports_a_saved_model_at_the_records_input_size(run_dir, tmp_path, capsys):
    out = tmp_path / "exported_sm"
    assert cli.main(["export", f"model={run_dir}", f"out={out}", "format=saved_model"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    x = np.random.default_rng(6).random((2, *SHAPE)).astype(np.float32)
    loaded = tf.saved_model.load(str(out))  # the object owns the variables: keep it alive
    served = loaded.serve(x).numpy()
    assert served.shape == (2, *SIZE, CLASSES)
    with tf.device("/CPU:0"):
        np.testing.assert_allclose(served, np.asarray(load_model(run_dir).model(x, training=False)), atol=ATOL, rtol=0)


def test_the_help_says_one_score_per_class_per_pixel(capsys):
    assert cli.main(["export", "help"]) == 0
    text = capsys.readouterr().out
    assert "one score per class per pixel: (batch, height, width, classes)" in text
    assert "one" in text and "(batch, classes)" in text
