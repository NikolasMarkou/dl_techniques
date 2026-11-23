"""The ``pretrain`` record, ``predict`` and ``export`` (plan-2026-11-19T081214-5c6cc105 step 4, U2, D-003).

A pretrain run writes a schema-v2 ``task.json`` (``task: "pretrain"``) and its directory can be predicted with and
exported. Both answer the backbone's patch-token embeddings, float32 ``(N, D)`` per image with
``N = (side / patch)^2``: BEiT's class-token row is dropped so the two families give the same layout. This file pins,
on one real tiny run per family on the fake CIFAR-10 (``et_mim/et_tiny`` at ``num_steps`` 2; ``beit_mim/beit_tiny`` at
one 32-wide layer on a 16-code toy tokenizer), CPU, written under ``tmp_path``:

* **the record**: it passes ``record_problems``; its scaling is the word, mean and std image_classification records for
  the same dataset (Pre-Mortem 4); its tokenizer is the sha256 of the tokenizer file (``null`` for ET); a record missing
  any key is refused; a resume compares the geometry, the scaling, the patch and the tokenizer (``DATA_KEYS``);
  ``load_model`` refuses a record whose size or patch is not the model's, or that names another family or variant;
* **the input** (D-023): the predictor's model input of raw rows is the CIFAR selection stream's batch bit for bit, and
  of Imagenette files the Imagenette evaluation stream's;
* **predict**: one ``.npy`` per image equal to the test's own ``model.backbone`` call on its own standardized input
  (BEiT's row 0 dropped), ``index.json`` last, a path, a PIL image and an array alike;
* **export**: each cell that is not refused equals the predictor within the shared check's tolerance; another
  ``input_shape`` is refused before anything is written; the ``et_mim`` tflite cell is refused by name (D-024) and its
  reason is still true (the converter still misses the shared check on the real backbone).
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from dl_techniques.datasets.vision.classification import get_spec
from train.api import cli
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, SUFFIX, TFLITE_CHECK_RTOL
from train.api.tasks.pretrain import PretrainTrainer
from train.common import pretrain_run

from .test_export_roundtrip import read_back
from .test_pretrain_task import (  # noqa: F401  (``world`` is a fixture)
    BEIT_MODEL, BEIT_SMALL, LOW_FLOOR, TOKENIZER_KEY, make_config, make_tokenizer, world,
)

FAMILIES = ("et_mim", "beit_mim")
SIDE, PATCH = 32, 4
N = (SIDE // PATCH) ** 2
#: The keys of a pretrain record, in the order they are written (Problem Statement 1, D-009).
KEYS = ("schema_version", "task", "model", "family", "variant", "input_size", "channels", "normalize", "mean", "std",
        "patch_size", "tokenizer", "output")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    """One finished 1-epoch run per family (the record is read by each test: a missing one fails the test, not this)."""
    root = tmp_path_factory.mktemp("pretrain_record")
    tokenizer = make_tokenizer(root / "tokenizer.keras")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        PretrainTrainer(make_config(root / "et_mim", epochs=1)).run()
        PretrainTrainer(make_config(root / "beit_mim", epochs=1, model=BEIT_MODEL,
                                    model_args={TOKENIZER_KEY: tokenizer, **BEIT_SMALL})).run()
    return {family: SimpleNamespace(family=family, run=root / family, tokenizer=tokenizer) for family in FAMILIES}


def record_of(run: SimpleNamespace) -> Dict[str, Any]:
    return read_json(run.run / "task.json")


def pictures(count: int, seed: int, shape=(SIDE, SIDE, 3)) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(count)]


def oracle(model: Any, images: List[np.ndarray], record: Dict[str, Any]) -> np.ndarray:
    """The test's own embeddings: bytes / 255 (numpy), a bilinear resize to the trained side where the size differs,
    ``(x - mean) / std`` with the record's statistics, ``model.backbone`` with ``training=False``, BEiT's row 0
    dropped."""
    rows = []
    for image in images:
        unit = image.astype(np.float32) / np.float32(255.0)
        if unit.shape[:2] != (SIDE, SIDE):
            unit = tf.image.resize(unit, (SIDE, SIDE), method="bilinear").numpy()
        rows.append((unit - np.asarray(record["mean"], np.float32)) / np.asarray(record["std"], np.float32))
    tokens = np.asarray(model.backbone(np.stack(rows), training=False))
    return tokens[:, 1:, :] if record["family"] == "beit_mim" else tokens


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_the_record_passes_the_reader_and_holds_what_the_run_trained_with(runs, family):
    from train.api.args import TransformArgs
    from train.api.predictor import OPTIONAL_RECORD_KEYS, RECORD_KEYS, record_problems
    from train.common.classification_data import _stats

    record = record_of(runs[family])
    assert record_problems(record) == []
    assert RECORD_KEYS["pretrain"] == KEYS
    assert tuple(record)[: len(KEYS)] == KEYS and set(record) - set(KEYS) <= set(OPTIONAL_RECORD_KEYS)
    variant = "et_tiny" if family == "et_mim" else "beit_tiny"
    assert (record["schema_version"], record["task"], record["model"], record["family"], record["variant"]) == (
        2, "pretrain", f"{family}/{variant}", family, variant)
    assert (record["input_size"], record["channels"], record["patch_size"]) == ([SIDE, SIDE], 3, PATCH)
    # Pre-Mortem 4: the word, mean and std image_classification records for the same dataset (its default).
    mean, std = _stats(get_spec("cifar10"), 3, TransformArgs().normalize)
    assert record["normalize"] == TransformArgs().normalize == "standardize"
    assert record["mean"] == [float(v) for v in mean] and record["std"] == [float(v) for v in std]
    expected = None if family == "et_mim" else hashlib.sha256(Path(runs[family].tokenizer).read_bytes()).hexdigest()
    assert record["tokenizer"] == expected and record["output"] == "patch_embeddings"


@pytest.mark.parametrize("key", KEYS)
def test_a_record_missing_any_key_is_refused(runs, key):
    from train.api.predictor import record_problems

    record = {k: v for k, v in record_of(runs["et_mim"]).items() if k != key}
    problems = record_problems(record)
    assert problems, f"a record without {key!r} was accepted"
    if key not in ("schema_version", "task"):
        assert problems == [f"missing key(s) {key!r}"]


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"input_size": [32, 30]}, "'input_size'"),
        ({"input_size": [30, 30]}, "'patch_size'"),
        ({"patch_size": 0}, "'patch_size'"),
        ({"channels": 1}, "'channels'"),
        ({"tokenizer": "abc"}, "'tokenizer'"),
        ({"tokenizer": "A" * 64}, "'tokenizer'"),
        ({"output": "logits"}, "'output'"),
        ({"std": [0.2, 0.2]}, "'std'"),
    ],
    ids=["not_square", "patch_does_not_divide", "patch_zero", "grey", "short_hash", "upper_hash", "other_output",
         "std_width"],
)
def test_a_record_of_impossible_values_is_refused_naming_the_key(runs, change, fragment):
    from train.api.predictor import record_problems

    problems = record_problems({**record_of(runs["et_mim"]), **change})
    assert any(fragment in line for line in problems), problems


@pytest.mark.parametrize(
    "key, value",
    [("patch_size", 8), ("input_size", [64, 64]), ("channels", 1), ("normalize", "unit"), ("mean", [0.5, 0.5, 0.5]),
     ("std", [0.5, 0.5, 0.5]), ("tokenizer", "0" * 64)],
)
def test_a_resume_compares_the_geometry_the_scaling_the_patch_and_the_tokenizer(runs, key, value):
    from train.api.predictor import DATA_KEYS, record_differences

    record = record_of(runs["et_mim"])
    assert set(DATA_KEYS["pretrain"]) == {
        "input_size", "channels", "normalize", "mean", "std", "patch_size", "tokenizer"}
    assert [line.split(":")[0] for line in record_differences(record, {**record, key: value})] == [key]


def test_the_output_word_is_how_the_predictor_answers_and_not_a_data_key(runs):
    from train.api.predictor import record_differences

    record = record_of(runs["et_mim"])
    assert record_differences(record, {**record, "output": "something_else"}) == []


@pytest.mark.parametrize(
    "change, fragment",
    [({"input_size": [64, 64]}, "built at"), ({"patch_size": 8}, "16 patches of 8 px"),
     ({"input_size": [16, 16], "patch_size": 2}, "built at")],
    ids=["other_side", "other_patch", "other_side_and_patch"],
)
def test_load_model_refuses_a_record_whose_geometry_is_not_the_models(runs, tmp_path, change, fragment):
    from train.api.predictor import load_model

    copy = tmp_path / "run"
    copy.mkdir()
    (copy / "best_model.keras").write_bytes((runs["et_mim"].run / "best_model.keras").read_bytes())
    (copy / "task.json").write_text(json.dumps({**record_of(runs["et_mim"]), **change}))
    with pytest.raises(ConfigError, match=fragment):
        load_model(copy)


@pytest.mark.parametrize(
    "file_of, named, fragment",
    [("et_mim", "et_mim/et_small", "is not that variant (scale is \"tiny\" in the model, \"small\" in 'et_small')"),
     ("et_mim", "beit_mim/beit_tiny", "holds no layer of that family (its own layers: EnergyTransformerBackbone)"),
     ("beit_mim", "et_mim/et_tiny", "holds no layer of that family (its own layers: BeitModel)")],
    ids=["et_tiny_under_et_small", "et_mim_under_beit_mim", "beit_mim_under_et_mim"],
)
def test_load_model_refuses_a_record_naming_another_family_or_variant(runs, tmp_path, file_of, named, fragment):
    """plan 5c6cc105 D-042 (review R1): the rows' variant tables are module-level, so only ``observe`` reads the file."""
    from train.api.predictor import load_model

    family, variant = named.split("/")
    copy = tmp_path / "run"
    copy.mkdir()
    (copy / "best_model.keras").write_bytes((runs[file_of].run / "best_model.keras").read_bytes())
    record = {**record_of(runs[file_of]), "model": named, "family": family, "variant": variant,
              "tokenizer": None if family == "et_mim" else "ab" * 32}
    (copy / "task.json").write_text(json.dumps(record))
    with pytest.raises(ConfigError) as caught:
        load_model(copy)
    assert f"names the {'variant' if family == file_of else 'family'} " in str(caught.value), str(caught.value)
    assert fragment in str(caught.value) and "the file and the model are not from one run" in str(caught.value)


# ---------------------------------------------------------------------
# The input is the stream's (D-023)
# ---------------------------------------------------------------------


def test_the_model_input_of_raw_cifar_rows_is_the_selection_streams_batch_bit_for_bit(runs):
    from train.api.pretrain_predictor import model_input
    from train.common.classification_data import load_split_arrays
    from train.common.image_streams import load_image_streams

    record = record_of(runs["et_mim"])
    streams = load_image_streams("cifar10", image_size=SIDE, batch_size=6, seed=42, augment=False)
    batch = next(iter(streams.select_ds))[0].numpy()
    # The rows are float32 bytes / 255, so the bytes come back exactly.
    raw = np.rint(load_split_arrays("cifar10", seed=42).x_select[: len(batch)] * 255.0).astype(np.uint8)
    assert len(batch) == 6 and np.array_equal(model_input(list(raw), record), batch)


def test_the_model_input_of_imagenette_files_is_the_imagenette_evaluation_streams_batch(monkeypatch, tmp_path):
    from dl_techniques.datasets.vision.splits import seeded_test_rows
    from tests.classification_fakes import IMAGENETTE_FAKE_TEST, imagenette_fake_records, use_fake_imagenette
    from train.api.predictor import build_pretrain_record
    from train.api.pretrain_predictor import model_input, read_image
    from train.common.image_streams import load_image_streams, stream_statistics

    use_fake_imagenette(monkeypatch)
    streams = load_image_streams("imagenette", image_size=SIDE, batch_size=4, seed=0, augment=False)
    batch = next(iter(streams.test_ds))[0].numpy()
    jpegs, _ = imagenette_fake_records(IMAGENETTE_FAKE_TEST, 2)
    paths = []
    for position, row in enumerate(seeded_test_rows(IMAGENETTE_FAKE_TEST, 0)[: len(batch)]):
        paths.append(tmp_path / f"{position}.jpg")
        paths[-1].write_bytes(jpegs[row])
    mean, std = stream_statistics("imagenette")
    record = build_pretrain_record(model="et_mim/et_tiny", family="et_mim", variant="et_tiny", input_size=(SIDE, SIDE),
                                   channels=3, mean=mean, std=std, patch_size=PATCH, tokenizer=None)
    assert np.array_equal(model_input([read_image(path) for path in paths], record), batch)


# ---------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_predict_writes_one_embedding_per_image_equal_to_the_backbone_on_the_tests_own_input(runs, family, tmp_path):
    images = pictures(3, seed=1) + pictures(1, seed=2, shape=(48, 40, 3))
    folder = tmp_path / "pictures"
    folder.mkdir()
    for position, image in enumerate(images):
        Image.fromarray(image).save(folder / f"p{position}.png")
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={runs[family].run}", f"data={folder}", f"out={out}"]) == 0
    record = record_of(runs[family])
    model = keras.models.load_model(str(runs[family].run / "best_model.keras"), compile=False)
    expected = oracle(model, images, record)
    width = expected.shape[-1]
    assert expected.shape == (4, N, width), "the control: N patch rows, the class row dropped"
    for position in range(4):
        got = np.load(out / f"p{position}.npy")
        assert got.dtype == np.float32 and got.shape == (N, width)
        np.testing.assert_allclose(got, expected[position], rtol=0, atol=1e-6 * max(1.0, float(np.abs(expected).max())))
    index = read_json(out / "index.json")
    assert index["output"] == "patch_embeddings" and index["note"]
    assert index["embeddings"] == [
        {"path": str(folder / f"p{position}.png"), "file": f"p{position}.npy", "shape": [N, width]}
        for position in range(4)
    ]


def test_a_predict_that_fails_half_way_leaves_its_arrays_and_no_index(runs, tmp_path, monkeypatch, capsys):
    import train.api.pretrain_predictor as module

    monkeypatch.setattr(module, "CHUNK", 1)
    folder = tmp_path / "pictures"
    folder.mkdir()
    Image.fromarray(pictures(1, seed=3)[0]).save(folder / "a.png")
    (folder / "b.png").write_bytes(b"not an image")
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={runs['et_mim'].run}", f"data={folder}", f"out={out}"]) == 2
    assert "b.png" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["a.npy"], "the index is written last: a broken answer has none"


def test_a_path_a_pil_image_and_an_array_of_one_picture_give_one_answer(runs, tmp_path):
    from train.api.predictor import load_model
    from train.api.pretrain_predictor import model_input, read_image

    predictor = load_model(runs["beit_mim"].run)
    pixels = pictures(1, seed=4)[0]
    path = tmp_path / "p.png"
    Image.fromarray(pixels).save(path)
    forms = [str(path), Image.fromarray(pixels), pixels, pixels.astype(np.float32) / 255.0]
    inputs = model_input([read_image(form) for form in forms], predictor.record)
    for row in inputs[1:]:
        assert np.array_equal(row, inputs[0]), "the four forms of one picture are one model input"
    answers = predictor.predict(forms)["embeddings"]
    assert len(answers) == 4 and answers[0].shape == (N, BEIT_SMALL["hidden_size"])
    single = predictor.predict(pixels)["embeddings"]
    assert len(single) == 1
    for other in [*answers[1:], single[0]]:
        np.testing.assert_allclose(other, answers[0], rtol=0, atol=1e-6)


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


EXPORTED = [(family, fmt) for family in FAMILIES for fmt in FORMATS if (family, fmt) != ("et_mim", "tflite")]


@pytest.mark.parametrize("family, fmt", EXPORTED, ids=[f"{family}-{fmt}" for family, fmt in EXPORTED])
def test_each_export_that_is_not_refused_equals_the_predictor(runs, family, fmt, tmp_path):
    from train.api.predictor import load_model
    from train.api.pretrain_predictor import model_input

    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    assert cli.main(["export", f"model={runs[family].run}", f"out={out}", f"format={fmt}"]) == 0
    assert read_json(out.with_name(out.name + ".task.json")) == record_of(runs[family])
    predictor = load_model(runs[family].run)
    images = pictures(3, seed=5)
    expected = np.stack(predictor.predict(images)["embeddings"])
    got = np.asarray(read_back(fmt, out, model_input(images, predictor.record)))
    assert got.shape == expected.shape == (3, N, expected.shape[-1])
    scale = max(1.0, float(np.abs(expected).max()))
    assert float(np.abs(got - expected).max()) <= TFLITE_CHECK_RTOL * scale


@pytest.mark.parametrize("fmt", FORMATS)
def test_another_input_shape_is_refused_before_a_file_is_written(runs, fmt, tmp_path):
    from train.api.exporting import export
    from train.api.predictor import load_model

    predictor = load_model(runs["beit_mim"].run)
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    with pytest.raises(ConfigError, match=r"\(48, 48, 3\) is not the trained size \(32, 32, 3\)"):
        export(model=predictor.model, out=out, format=fmt, input_shape=(48, 48, 3), record=predictor.record)
    assert list(tmp_path.iterdir()) == []


def test_the_et_mim_tflite_cell_is_refused_by_name_before_anything_is_written(runs, tmp_path, capsys):
    out = tmp_path / "exported.tflite"
    assert cli.main(["export", f"model={runs['et_mim'].run}", f"out={out}", "format=tflite"]) == 2
    error = capsys.readouterr().err
    assert "'tflite' of a model of the 'et_mim' family is refused" in error and "Hopfield" in error, error
    assert list(tmp_path.iterdir()) == [], "no file, no SavedModel sibling, no task.json"


def test_the_et_mim_tflite_refusal_still_has_its_reason(runs, tmp_path):
    # D-024: the real ET backbone's served function through exporting's own tflite writer, past the refusal: the shared
    # numeric check still refuses the converted file. When it does not, the converter carries the Hopfield update now.
    from train.api import exporting
    from train.api.predictor import load_model
    from train.api.pretrain_predictor import served

    predictor = load_model(runs["et_mim"].run)
    spec = exporting._input_spec(predictor.model, (SIDE, SIDE, 3), rank=3, batch=None)
    out = tmp_path / "et.tflite"
    try:
        exporting._export_tflite(predictor.model, out, spec, served(predictor.model, predictor.record))
    except ConfigError as exc:
        assert "differs from the Keras model by" in str(exc), str(exc)
    else:
        pytest.fail("the et_mim backbone now converts to tflite within the shared check: lift the ('et_mim', 'tflite') "
                    "row of train.api.pretrain_predictor.EXPORT_REFUSALS (D-024)")
    assert not out.exists()
