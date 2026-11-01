"""``export``: each format reloads in its own runtime and gives the Keras model's numbers.

The claim under test is numeric: for a fixed batch, the exported artefact's output equals
``model(batch)`` within ``ATOL`` (1e-5; one earlier measurement was 2.4e-7), in all four
formats, each read back by the runtime that format is for (Keras, ``tf.saved_model.load``, the
TFLite interpreter, ONNX Runtime). The model is an untrained ``vit_pico`` with random weights,
so equal outputs are not an artefact of a model that outputs a constant. The difference arm is
made to fail by an export that feeds the model a different input (see the plan's RED proof).

Everything writes under ``tmp_path``. The suite never deletes: an export directory is left for
pytest's own tmp-path rotation.
"""

import hashlib
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.vit.model import create_vit
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, TFLITE_SIBLING, export

ATOL = 1e-5
SHAPE = (32, 32, 3)
CLASSES = 10
NAMES = {"keras": "m.keras", "saved_model": "m_saved_model", "tflite": "m.tflite", "onnx": "m.onnx"}


@pytest.fixture(scope="module")
def model():
    net = create_vit("vit_pico", num_classes=CLASSES, input_shape=SHAPE, patch_size=4)
    net(np.zeros((1, *SHAPE), np.float32))  # build
    return net


@pytest.fixture(scope="module")
def batch():
    return np.random.default_rng(0).random((4, *SHAPE)).astype(np.float32)


@pytest.fixture(scope="module")
def reference(model, batch):
    return np.asarray(model(batch, training=False), np.float64)


def assert_matches(got, reference, label):
    got = np.asarray(got, np.float64)
    assert got.shape == reference.shape, label
    np.testing.assert_allclose(got, reference, atol=ATOL, rtol=0, err_msg=label)
    assert np.abs(reference).max() > 1e-3  # a model that outputs nothing would match anything


def read_back(format, path, batch):
    """The artefact's output on ``batch``, from the runtime the format is for."""
    if format == "keras":
        return keras.models.load_model(path, compile=False)(batch, training=False)
    if format == "saved_model":
        import tensorflow as tf

        loaded = tf.saved_model.load(str(path))  # the object owns the variables: keep it alive
        return loaded.serve(batch)
    if format == "tflite":
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(path))
        entry = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(entry["index"], batch.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(entry["index"], batch)
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: batch})[0]


def digest(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    sha = hashlib.sha256()
    for file in files:
        sha.update(str(file.relative_to(path if path.is_dir() else path.parent)).encode())
        sha.update(file.read_bytes())
    return sha.hexdigest()


# ---------------------------------------------------------------------
# The formats
# ---------------------------------------------------------------------


def test_the_format_list_is_literal():
    assert FORMATS == ("keras", "saved_model", "tflite", "onnx")


@pytest.mark.parametrize("format", FORMATS)
def test_each_format_reloads_and_matches_the_keras_model_on_a_fixed_batch(
    format, model, batch, reference, tmp_path
):
    if format == "onnx":
        pytest.importorskip("tf2onnx")
        pytest.importorskip("onnxruntime")
    out = tmp_path / NAMES[format]
    written = export(model=model, out=out, format=format, input_shape=SHAPE)
    assert written == out and out.exists()
    assert_matches(read_back(format, out, batch), reference, format)


def test_a_different_batch_size_is_served_by_the_free_batch_formats(model, tmp_path):
    other = np.random.default_rng(1).random((7, *SHAPE)).astype(np.float32)
    expected = np.asarray(model(other, training=False), np.float64)
    path = export(model=model, out=tmp_path / "m_saved_model", format="saved_model", input_shape=SHAPE)
    assert_matches(read_back("saved_model", path, other), expected, "batch of 7")
    flat = export(model=model, out=tmp_path / "m.tflite", format="tflite", input_shape=SHAPE)
    assert_matches(read_back("tflite", flat, other), expected, "tflite batch of 7")


def test_the_exported_numbers_are_the_models_not_a_constant(model, batch, tmp_path):
    other = np.random.default_rng(2).random((4, *SHAPE)).astype(np.float32)
    path = export(model=model, out=tmp_path / "m.keras", format="keras")
    first = np.asarray(read_back("keras", path, batch))
    second = np.asarray(read_back("keras", path, other))
    assert np.abs(first - second).max() > 1e-4


def test_a_model_reloaded_from_its_archive_exports_like_the_original(model, batch, reference, tmp_path):
    archive = export(model=model, out=tmp_path / "a.keras", format="keras")
    reloaded = keras.models.load_model(archive, compile=False)
    path = export(model=reloaded, out=tmp_path / "b_saved_model", format="saved_model", input_shape=SHAPE)
    assert_matches(read_back("saved_model", path, batch), reference, "reloaded")


def test_a_functional_model_reports_its_own_input_shape(tmp_path):
    inputs = keras.Input((6, 5, 3))
    outputs = keras.layers.Dense(4)(keras.layers.Flatten()(inputs))
    net = keras.Model(inputs, outputs)
    path = export(model=net, out=tmp_path / "f.tflite", format="tflite")  # no input_shape given
    x = np.random.default_rng(3).random((2, 6, 5, 3)).astype(np.float32)
    assert_matches(read_back("tflite", path, x), np.asarray(net(x), np.float64), "functional")


def test_the_tflite_savedmodel_is_left_beside_the_result_and_never_deleted(model, tmp_path):
    out = export(model=model, out=tmp_path / "m.tflite", format="tflite", input_shape=SHAPE)
    sibling = tmp_path / ("m.tflite" + TFLITE_SIBLING)
    assert out.is_file() and (sibling / "saved_model.pb").is_file()


# ---------------------------------------------------------------------
# onnx without tf2onnx
# ---------------------------------------------------------------------


def test_onnx_is_refused_when_tf2onnx_cannot_be_imported_and_nothing_is_written(model, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tf2onnx", None)  # `import tf2onnx` raises ImportError
    out = tmp_path / "m.onnx"
    with pytest.raises(ConfigError, match=r"'onnx' needs the tf2onnx package.*keras.*saved_model.*tflite"):
        export(model=model, out=out, format="onnx", input_shape=SHAPE)
    assert not out.exists() and list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------


@pytest.mark.parametrize("format", ["keras", "saved_model", "tflite"])
def test_an_existing_out_is_refused_and_left_as_it_is(format, model, tmp_path):
    out = tmp_path / NAMES[format]
    if format == "saved_model":
        out.mkdir()
        (out / "keep.txt").write_text("mine")
    else:
        out.write_text("mine")
    before = digest(out)
    with pytest.raises(ConfigError, match="already exists; export never overwrites"):
        export(model=model, out=out, format=format, input_shape=SHAPE)
    assert digest(out) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [out.name]  # no sibling was started


def test_an_existing_onnx_out_is_refused_and_left_as_it_is(model, tmp_path):
    pytest.importorskip("tf2onnx")
    out = tmp_path / "m.onnx"
    out.write_text("mine")
    with pytest.raises(ConfigError, match="already exists"):
        export(model=model, out=out, format="onnx", input_shape=SHAPE)
    assert out.read_text() == "mine"


def test_a_task_json_that_appears_after_the_check_is_never_overwritten(tmp_path, monkeypatch):
    """Mutant ``xp_record_w`` (review-iter-1 concern 10): the record beside the export is opened
    exclusively, so a ``<out>.task.json`` written by someone else while the model was converting
    is an error and not a truncation."""
    from train.api.run_dir import TASK_SUFFIX

    inputs = keras.Input((6, 5, 3))
    net = keras.Model(inputs, keras.layers.Dense(4)(keras.layers.Flatten()(inputs)))
    out = tmp_path / "m.keras"
    sibling = tmp_path / ("m.keras" + TASK_SUFFIX)
    theirs = "someone else's record"
    real = net.save

    def save_then_they_write(*arguments, **keywords):
        result = real(*arguments, **keywords)
        sibling.write_text(theirs)
        return result

    monkeypatch.setattr(net, "save", save_then_they_write)
    with pytest.raises(FileExistsError):
        export(model=net, out=out, format="keras", record={"schema_version": 1})
    assert sibling.read_text() == theirs


@pytest.mark.parametrize("format", ["keras", "tflite", "onnx"])
def test_a_file_that_appears_after_the_check_and_before_the_write_is_never_overwritten(
    format, tmp_path, monkeypatch
):
    """Mutants ``model.save overwrite on``, ``tflite wb`` and ``onnx wb`` (D-063 anchor).

    ``check_target`` found ``out`` free; the conversion takes seconds, and someone writes the
    path meanwhile. The last step must fail rather than truncate what is there.
    """
    from train.api import exporting

    if format == "onnx":
        pytest.importorskip("tf2onnx")
    inputs = keras.Input((6, 5, 3))
    net = keras.Model(inputs, keras.layers.Dense(4)(keras.layers.Flatten()(inputs)))
    out = tmp_path / NAMES[format]
    theirs = b"someone else's file"

    def appear(real, before=False):
        def wrapper(*arguments, **keywords):
            if before:
                out.write_bytes(theirs)
            result = real(*arguments, **keywords)
            if not before:
                out.write_bytes(theirs)
            return result

        return wrapper

    if format == "keras":  # the save IS the write: the file must be there when it starts
        monkeypatch.setattr(net, "save", appear(net.save, before=True))
    elif format == "tflite":
        monkeypatch.setattr(exporting, "_convert_tflite", appear(exporting._convert_tflite))
    else:
        import tf2onnx

        monkeypatch.setattr(tf2onnx.convert, "from_function", appear(tf2onnx.convert.from_function))
    # Keras asks on stdin before it overwrites a file (an OSError under capture); the two
    # byte writers raise FileExistsError.
    with pytest.raises((FileExistsError, OSError, EOFError)):
        export(model=net, out=out, format=format)
    assert out.read_bytes() == theirs


def test_an_existing_tflite_sibling_is_refused_before_anything_is_written(model, tmp_path):
    sibling = tmp_path / ("m.tflite" + TFLITE_SIBLING)
    sibling.mkdir()
    with pytest.raises(ConfigError, match="already exists"):
        export(model=model, out=tmp_path / "m.tflite", format="tflite", input_shape=SHAPE)
    assert list(tmp_path.iterdir()) == [sibling] and list(sibling.iterdir()) == []


def test_a_second_export_to_the_same_out_is_refused_after_the_first_succeeded(model, tmp_path):
    out = export(model=model, out=tmp_path / "m.keras", format="keras")
    before = digest(out)
    with pytest.raises(ConfigError, match="already exists"):
        export(model=model, out=out, format="keras")
    assert digest(out) == before


@pytest.mark.parametrize("format", ["pb", "ONNX", "", None, 3])
def test_an_unknown_format_is_refused_naming_the_list(format, model, tmp_path):
    with pytest.raises(ConfigError, match=r"expected one of 'keras', 'saved_model', 'tflite', 'onnx'"):
        export(model=model, out=tmp_path / "x", format=format)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "format, name", [("keras", "m.h5"), ("keras", "m"), ("tflite", "m.keras"), ("onnx", "m.tflite")]
)
def test_a_file_format_needs_its_own_suffix(format, name, model, tmp_path):
    with pytest.raises(ConfigError, match=r"writes a file named \*\."):
        export(model=model, out=tmp_path / name, format=format, input_shape=SHAPE)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [None, "model", np.zeros(3)])
def test_something_that_is_not_a_model_is_refused(bad, tmp_path):
    with pytest.raises(ConfigError, match="needs a built keras.Model"):
        export(model=bad, out=tmp_path / "m.keras", format="keras")


def test_a_model_that_was_never_built_is_refused(tmp_path):
    unbuilt = create_vit("vit_pico", num_classes=CLASSES, input_shape=SHAPE, patch_size=4)
    assert not unbuilt.built
    with pytest.raises(ConfigError, match="needs a built keras.Model"):
        export(model=unbuilt, out=tmp_path / "m.keras", format="keras")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("format", ["saved_model", "tflite"])
def test_a_model_with_no_inputs_of_its_own_needs_an_input_shape(format, model, tmp_path):
    with pytest.raises(ConfigError, match="pass input_shape=\\(height, width, channels\\)"):
        export(model=model, out=tmp_path / NAMES[format], format=format)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 0), (32, 32, 3, 1), (32.0, 32, 3), "abc", (True, 32, 3)])
def test_a_bad_input_shape_is_refused(shape, model, tmp_path):
    with pytest.raises(ConfigError, match="input_shape"):
        export(model=model, out=tmp_path / "m_saved_model", format="saved_model", input_shape=shape)
    assert list(tmp_path.iterdir()) == []


def test_the_parent_directory_is_created(model, tmp_path):
    out = export(model=model, out=tmp_path / "a" / "b" / "m.keras", format="keras")
    assert out.is_file()

