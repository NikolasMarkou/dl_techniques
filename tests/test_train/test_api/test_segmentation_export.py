"""``export`` on the segmentation families, in every format, at a non-square size (plan 2faf890a step 31, D-081).

Plan 98b866fb step 2 (D-012) adds ``resnet_seg`` (``resnet18``) and ``convnext_seg`` (``cifar10``, with a stochastic
depth of 0.5 so that the drop-path controls below cover a ConvNeXt body too) to the table: their rows run through the
same relative tolerance, the same repeatability on one runtime object and the same record check, because the head
they end in (transposed convolutions, a LayerNorm, a 1x1 convolution) and a BatchNorm or LayerNorm backbone are
operations a converter may translate differently from the U-Nets'.

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
  runtime, gives Keras' ``training=False`` scores to the family's band for that format (``EXPORT_BAND``: a share of the largest
  score, measured per family on trained weights and on these fixtures, D-029, which replaces the one relative 1e-4 of D-088), with the
  input's height and width and one score per class;
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

#: THE EXPORT CONTRACT (plan 98b866fb fix 9.3, D-029), one band per family: the largest difference between an artifact (read by its own
#: runtime: ``saved_model``, ``tflite``, ``onnx``) and Keras on the CPU, as a share of the largest Keras score. D-088 had replaced D-081's
#: absolute 1e-4 by a relative 1e-4 for every family on the strength of the AccUNet files then measured (4.9e-5 at worst); the iteration 1
#: gate's trained AccUNet cells then showed 1.4e-4 (saved_model) and 2.5e-4 (tflite), every argmax at 0.99995 or better, while the other
#: three families stayed at 2.1e-6 or less: one number for four families was both too loose for three and too tight for one. Each band is
#: between 2 and 10 times the largest gap measured for its family, in any of the three formats, on trained Oxford-IIIT Pet weights (CPU,
#: 16 real test images, 128 px) and on this file's fixtures, all listed in ``export_gap_evidence.json`` (rows copied from the probes,
#: sha256 of each source); a test holds both sides.
EXPORT_BAND = {
    "convunext": 5e-6,
    "accunet": 1e-3,
    "resnet_seg": 5e-6,
    "convnext_seg": 5e-6,
}
#: The share of argmaxes that agree with Keras' (the worst measured over the trained cells is 0.99995; the fixtures' is 1.0).
EXPORT_ARGMAX = 0.9999
#: Keras itself (the ``keras`` format) and the checks that are not about a format's runtime keep the absolute figure measured at a scale of 1.
ATOL = 1e-4  # at a scale of 1
#: The scale of a trained AccUNet's scores, which `test_a_trained_scale_accunet_matches_in_every_format` builds to.
TRAINED_SCALE = 220.0
SIZE = (32, 64)  # height != width
SHAPE = (*SIZE, 3)
CLASSES = 3

#: family -> (variant, model_args, the layer that makes the scores, whether stochastic depth is live)
FAMILIES = {
    "convunext": ("tiny", {"drop_path_rate": 0.5}, "final_output", True),
    "accunet": ("base", {"base_filters": 8, "mlfc_iterations": 1}, "output_conv", False),
    "resnet_seg": ("resnet18", {}, "seg_head", False),
    "convnext_seg": ("cifar10", {"drop_path_rate": 0.5}, "seg_head", True),
}
#: The families with a live stochastic depth: the controls that need one run over these.
STOCHASTIC = [family for family, row in FAMILIES.items() if row[3]]
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


def build(family, target=1.0):
    """The family's model with its weights moved off the initial ones and its scores at a scale of ``target``."""
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
    layer.set_weights([w * (target / scale) for w in layer.get_weights()])
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

    def model(self, family, target=1.0):
        key = family if target == 1.0 else (family, target)
        if key not in self.models:
            self.models[key] = build(family, target)
        return self.models[key]

    def artifact(self, family, format, target=1.0):
        key = (family, format) if target == 1.0 else (family, format, target)
        if key not in self.written:
            if format == "onnx":
                pytest.importorskip("tf2onnx")
                pytest.importorskip("onnxruntime")
            handler = Collected()
            logger = logging.getLogger("dl")
            logger.addHandler(handler)
            try:
                self.written[key] = export(
                    model=self.model(family, target),
                    out=self.root / (family if target == 1.0 else f"{family}_{target:g}") / NAMES[format], format=format,
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


def gap_limit(family, format, scale):
    """The largest difference the contract allows for ``family`` in ``format`` at the Keras scores' ``scale``."""
    return ATOL if format == "keras" else EXPORT_BAND[family] * scale


def check_within_contract(family, format, got, expected):
    """Assert that ``got`` (an artifact's scores) is inside the family's band of ``expected`` (Keras', CPU) and agrees on the argmax."""
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    assert got.shape == expected.shape, f"{family} {format}: {got.shape} against {expected.shape}"
    scale = float(np.abs(expected).max())
    worst = float(np.abs(got - expected).max())
    limit = gap_limit(family, format, scale)
    assert worst <= limit, f"{family} {format}: {worst:.3g} ({worst / scale:.2g} of the scale) against {limit:.3g}"
    assert (got.argmax(-1) == expected.argmax(-1)).mean() >= EXPORT_ARGMAX, f"{family} {format}: the argmax disagrees"


# ---------------------------------------------------------------------
# The contract against the measurement
# ---------------------------------------------------------------------

EVIDENCE = json.loads((Path(__file__).resolve().parent / "export_gap_evidence.json").read_text())["rows"]


def measured(family, format, kind=None):
    return [row for row in EVIDENCE if row["family"] == family and row["format"] == format and (kind is None or row["kind"].startswith(kind))]


def test_the_evidence_has_trained_weights_and_fixtures_for_every_family_and_format():
    assert set(EXPORT_BAND) == set(FAMILIES)
    for family in EXPORT_BAND:
        for format in ("saved_model", "tflite", "onnx"):
            assert measured(family, format, "trained"), f"no trained-weights row for {family} {format}"
            assert measured(family, format, "fixture"), f"no fixture row for {family} {format}"


@pytest.mark.parametrize("family", FAMILIES)
def test_each_band_covers_every_measured_gap_of_its_family_and_is_within_ten_times_the_largest(family):
    worst = max(row["ratio"] for row in EVIDENCE if row["family"] == family)
    band = EXPORT_BAND[family]
    assert worst * 2 <= band, f"{family}: the band {band:.3g} is under twice the largest measured gap {worst:.3g}"
    assert band <= 10 * worst, f"{family}: the band {band:.3g} is more than ten times the largest measured gap {worst:.3g}"


@pytest.mark.parametrize("family", FAMILIES)
def test_every_measured_argmax_agreement_is_inside_the_contract(family):
    rows = [row for row in EVIDENCE if row["family"] == family]
    assert rows and all(row["argmax_agreement"] >= EXPORT_ARGMAX for row in rows)


def test_accunet_is_the_family_that_needs_a_wider_band_than_the_others():
    """Why there is a table: the three other families are under 3e-6 of the scale on trained weights, AccUNet is two orders above."""
    others = max(row["ratio"] for row in EVIDENCE if row["family"] != "accunet" and row["kind"] == "trained")
    accunet = max(row["ratio"] for row in EVIDENCE if row["family"] == "accunet" and row["kind"] == "trained")
    assert others < 3e-6 and accunet > 100 * others
    assert EXPORT_BAND["accunet"] >= 100 * max(band for family, band in EXPORT_BAND.items() if family != "accunet")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_comparison_passes_just_inside_the_band_and_fails_just_outside_it_for_every_family(family):
    rng = np.random.default_rng(3)
    expected = rng.normal(0, 40.0, (2, 4, 4, 3))
    scale = float(np.abs(expected).max())
    band = EXPORT_BAND[family]
    direction = np.zeros_like(expected)
    direction[0, 0, 0, 0] = 1.0  # one score moved, far from any argmax tie: the argmax does not change at these sizes
    for format in ("saved_model", "tflite", "onnx"):
        check_within_contract(family, format, expected + 0.9 * band * scale * direction, expected)
        with pytest.raises(AssertionError, match="against"):
            check_within_contract(family, format, expected + 1.1 * band * scale * direction, expected)
    check_within_contract(family, "keras", expected + 0.9 * ATOL * direction, expected)
    with pytest.raises(AssertionError, match="against"):
        check_within_contract(family, "keras", expected + 1.1 * ATOL * direction, expected)


def test_the_comparison_refuses_another_shape():
    expected = np.random.default_rng(4).normal(0, 1.0, (1, 4, 4, 3))
    with pytest.raises(AssertionError, match="against"):
        check_within_contract("convunext", "tflite", expected[:, :2], expected)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_comparison_refuses_a_flipped_argmax_that_is_inside_the_band(family):
    """A near tie at one pixel: the scores differ by half a band, the argmax flips, 1 pixel in 32 disagrees."""
    expected = np.random.default_rng(5).normal(0, 40.0, (2, 4, 4, 3))
    scale = float(np.abs(expected).max())
    gap = 0.2 * EXPORT_BAND[family] * scale
    expected[0, 0, 0] = [0.0, -gap, -30.0]  # class 0 wins by `gap`
    got = expected.copy()
    got[0, 0, 0, 1] += 2.5 * gap  # class 1 wins by 1.5 gap, and this moved one score by half a band
    assert np.abs(got - expected).max() < EXPORT_BAND[family] * scale and got[0, 0, 0].argmax() != expected[0, 0, 0].argmax()
    with pytest.raises(AssertionError, match="argmax"):
        check_within_contract(family, "onnx", got, expected)


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
    check_within_contract(family, format, got, expected)


@pytest.mark.parametrize("format", ["saved_model", "tflite", "onnx"])
def test_a_trained_scale_accunet_matches_in_every_format_to_a_relative_tolerance(format, workshop, batch):
    """Review-iter-3 concern 4: D-081's 1e-4 was measured on scores of scale 1. Trained AccUNet scores reach 220.

    The model here is the small fixture with its head lifted to a scale of 220, so the exports are compared
    where a trained model puts them. What is compared: the largest absolute difference between the artifact's
    scores and Keras' (CPU) against AccUNet's ``EXPORT_BAND`` times the largest Keras score, and, for ``tflite`` and
    ``onnx``, that the same difference is above the absolute ``ATOL`` (the control: an absolute tolerance would have refused a
    correct file). The measured numbers on trained weights are in ``export_gap_evidence.json``; the fixture sits far under the band
    (its weights are near-random), the band is the trained cells'.
    """
    if format == "onnx":
        pytest.importorskip("tf2onnx")
        pytest.importorskip("onnxruntime")
    model = workshop.model("accunet", TRAINED_SCALE)
    expected = reference(model, batch)
    scale = float(np.abs(expected).max())
    assert 0.5 * TRAINED_SCALE < scale < 2 * TRAINED_SCALE  # the head was lifted on another probe: the order of magnitude
    got = np.asarray(read_back(format, workshop.artifact("accunet", format, TRAINED_SCALE), batch), np.float64)
    check_within_contract("accunet", format, got, expected)
    worst = float(np.abs(got - expected).max())
    if format != "saved_model":  # the SavedModel runs Keras' own kernels: exactly equal on the CPU
        assert worst > ATOL, f"{format}: {worst:.3g} is under the absolute 1e-4, so this case would not have shown the defect"


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


@pytest.mark.parametrize("family", STOCHASTIC)
def test_a_saved_model_with_stochastic_depth_serves_the_same_answer_every_time(family, workshop, batch):
    loaded = tf.saved_model.load(str(workshop.artifact(family, "saved_model")))
    first, second = loaded.serve(batch).numpy(), loaded.serve(batch).numpy()
    np.testing.assert_array_equal(first, second)  # exactly: nothing random is left in the graph
    np.testing.assert_allclose(first, reference(workshop.model(family), batch), atol=ATOL, rtol=0)
    other = np.random.default_rng(4).random((5, *SHAPE)).astype(np.float32)  # the batch stays free
    np.testing.assert_allclose(
        loaded.serve(other).numpy(), reference(workshop.model(family), other), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("family", STOCHASTIC)
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
    # ``variant_overrides`` is written when a model argument changes a key of the variant's table (ConvUNext's
    # ``drop_path_rate``, AccUNet's ``base_filters``); the backbone families' rows touch no table key.
    table = registry.MODELS[family].load_variants()[FAMILIES[family][0]]
    overridden = bool(overrides_of_variant(table, FAMILIES[family][1]))
    assert overridden is (family in ("convunext", "accunet"))
    assert list(written)[len(RECORD_KEYS[SEGMENTATION]):] == (["variant_overrides"] if overridden else [])
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
