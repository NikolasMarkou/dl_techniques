"""The ``dino`` record, ``predict`` and ``export`` (plan-2026-11-26T224507-e3381378 step 7, D-028, D-058, D-061 to D-066).

A dino run writes a schema-v2 ``task.json`` (``task: "dino"``, the tenth kind) before ``fit``, and its directory can be predicted with and
exported. The file a run saves is the TRAINING model (a student, an EMA teacher, both with projection heads, taking a view set), so the answer
of both is DERIVED: image in, the student's CLS embedding out (the post-final-norm token, ``embedding_dim`` wide, the vector the k-NN probe
scores; never the projection head's output). Pinned here, on one real run of ``dino_v1/tiny`` (the full 12 blocks, 32 px, patch 8, 2 local crops,
a 64-wide head, one epoch) on the fake CIFAR-10, CPU, written under ``tmp_path``:

* **the record**: it passes ``record_problems`` and equals the record the test builds itself; a record missing any key, or holding an impossible
  value (a non-square size, a patch that does not divide it, another output word, a count that is not a positive int), is refused naming the key;
  a resume compares the geometry, the scaling, the patch and the number of local crops and nothing of the two widths or of how the predictor
  answers (``DATA_KEYS``);
* **load**: ``load_model`` runs the TRAINING model on the rank-5 zero batch of the record (``2 + n_local_crops`` views) and holds the record to the
  model before it is called: a wrong view count is named as one (not as "does not take the input_size it names"), a wrong size, patch,
  embedding width, head width or variant are named, and so is a file that is no dino model;
* **predict**: every embedding equals the test's own forward of the feature model on the image standardized in numpy, whatever the image's size
  or form; it is the CLS width and not the projection head's; a DELTA probe (one patch of a flat background changed) moves the embedding while
  the flat image alone does not move it wherever it sits in a batch, and two different images differ; the CLI writes one ``<stem>.npy`` per image
  and ``index.json`` last;
* **export** (measured, ``scratch_export/measured.md``): each of the four formats gives the embeddings ``predict`` gives and no cell is refused; another
  size is refused before a file is written; the shared tflite check judges embeddings, so a file that returns others is refused; and the ``.keras``
  file holds library layers only, so it loads where the ``train`` package cannot be imported.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from dl_techniques.datasets.vision.classification import get_spec
from train.api import cli, exporting, predictor
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, SUFFIX

from ._dino_world import IMAGE, N_LOCAL, OUT_DIM, PATCH, identity_world, train_real_run  # noqa: F401
from ._tokenizer_world import read_json
from .test_export_roundtrip import read_back

SRC = Path(__file__).resolve().parents[3] / "src"
TASK = "dino"
SIDE, GRID, VIEWS, EMBED = IMAGE, IMAGE // PATCH, 2 + N_LOCAL, 192
#: The keys of a dino record, in the order they are written.
KEYS = ("schema_version", "task", "model", "family", "variant", "input_size", "channels", "normalize", "mean", "std", "patch_size",
        "n_local_crops", "dino_out_dim", "embedding_dim", "output")


@pytest.fixture(scope="module")
def run(identity_world, tmp_path_factory):
    """One finished 1-epoch run of the full ``dino_v1/tiny`` (``_dino_world.train_real_run``), shared by every test; the record is read by each
    test, so a missing one fails the test and not this fixture."""
    out = tmp_path_factory.mktemp("dino_record") / "run"
    return SimpleNamespace(out=out, code=train_real_run(out))


@pytest.fixture(scope="module")
def loaded(run):
    """What ``load_model`` makes of the run."""
    return predictor.load_model(run.out)


@pytest.fixture(scope="module")
def reference(run):
    """The test's OWN feature model: the student of the reloaded training model, cut at the tensor the head reads."""
    student = keras.models.load_model(str(run.out / "best_model.keras"), compile=False).student
    return keras.Model(student.inputs[0], student.head.input)


def record_of(run: SimpleNamespace) -> Dict[str, Any]:
    return read_json(run.out / "task.json")


def pictures(count: int, seed: int, shape: Tuple[int, int, int] = (SIDE, SIDE, 3)) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(count)]


def standardized(image: np.ndarray, record: Dict[str, Any]) -> np.ndarray:
    """The model input of one ``uint8`` image, written out here and not read from the module: bytes / 255, a bilinear resize to the trained side
    where the size differs, ``(x - mean) / std`` with the record's statistics, all in float32."""
    unit = image.astype(np.float32) / np.float32(255.0)
    if unit.shape[:2] != (SIDE, SIDE):
        unit = tf.image.resize(unit, (SIDE, SIDE), method="bilinear").numpy()
    return (unit - np.asarray(record["mean"], np.float32)) / np.asarray(record["std"], np.float32)


def oracle(reference: Any, images: List[np.ndarray], record: Dict[str, Any]) -> np.ndarray:
    """The test's own embeddings: the feature model's output on the numpy-standardized images, ``(n, embedding_dim)``."""
    return np.asarray(reference(np.stack([standardized(image, record) for image in images]), training=False))


def copy_run(run: SimpleNamespace, directory: Path, record: Dict[str, Any]) -> Path:
    """A copy of the run's model file with ``record`` as its ``task.json``: the directory ``load_model`` reads."""
    directory.mkdir()
    shutil.copy(run.out / "best_model.keras", directory / "best_model.keras")
    (directory / "task.json").write_text(json.dumps(record))
    return directory


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


def test_the_record_passes_the_reader_and_is_the_one_the_test_builds_from_what_the_run_trained_with(run, reference):
    from train.api.args import TransformArgs
    from train.api.predictor import OPTIONAL_RECORD_KEYS, RECORD_KEYS, build_dino_record, record_problems
    from train.common.classification_data import _stats

    record = record_of(run)
    assert run.code == 0
    assert record_problems(record) == []
    assert RECORD_KEYS[TASK] == KEYS and tuple(record)[: len(KEYS)] == KEYS and set(record) - set(KEYS) <= set(OPTIONAL_RECORD_KEYS)
    assert (record["schema_version"], record["task"], record["model"], record["family"], record["variant"]) == (
        2, TASK, "dino_v1/tiny", "dino_v1", "tiny")
    assert (record["input_size"], record["channels"], record["patch_size"]) == ([SIDE, SIDE], 3, PATCH)
    assert (record["n_local_crops"], record["dino_out_dim"], record["embedding_dim"]) == (N_LOCAL, OUT_DIM, EMBED)
    assert reference.output_shape[-1] == record["embedding_dim"] != record["dino_out_dim"], "the control: the two widths are not one number"
    mean, std = _stats(get_spec("cifar10"), 3, TransformArgs().normalize)
    assert record["normalize"] == TransformArgs().normalize == "standardize"
    assert record["mean"] == [float(v) for v in mean] and record["std"] == [float(v) for v in std]
    assert record["output"] == "cls_embeddings"
    assert record == build_dino_record(
        model="dino_v1/tiny", family="dino_v1", variant="tiny", input_size=(SIDE, SIDE), channels=3, mean=mean, std=std, patch_size=PATCH,
        n_local_crops=N_LOCAL, dino_out_dim=OUT_DIM, embedding_dim=EMBED, variant_overrides=record.get("variant_overrides"))


@pytest.mark.parametrize("key", KEYS)
def test_a_record_missing_any_key_is_refused(run, key):
    problems = predictor.record_problems({k: v for k, v in record_of(run).items() if k != key})
    assert problems, f"a record without {key!r} was accepted"
    if key not in ("schema_version", "task"):
        assert problems == [f"missing key(s) {key!r}"]


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"input_size": [32, 30]}, "'input_size'"),
        ({"input_size": [30, 30]}, "'patch_size'"),
        ({"patch_size": 0}, "'patch_size'"),
        ({"patch_size": 6}, "'patch_size'"),
        ({"channels": 1}, "'channels'"),
        ({"n_local_crops": 0}, "'n_local_crops'"),
        ({"n_local_crops": 2.0}, "'n_local_crops'"),
        ({"n_local_crops": True}, "'n_local_crops'"),
        ({"dino_out_dim": 0}, "'dino_out_dim'"),
        ({"embedding_dim": -192}, "'embedding_dim'"),
        ({"embedding_dim": "192"}, "'embedding_dim'"),
        ({"output": "reconstructions"}, "'output'"),
        ({"output": "patch_embeddings"}, "'output'"),
        ({"normalize": "unit"}, "'normalize'"),
        ({"std": [0.2, 0.2]}, "'std'"),
        ({"tokenizer": "ab" * 32}, "unknown key"),
    ],
    ids=["not_square", "patch_does_not_divide", "patch_zero", "patch_not_dividing_either", "grey", "no_local_crops", "crops_a_float",
         "crops_a_bool", "no_head", "negative_embedding", "embedding_a_string", "other_output", "the_pretrain_word", "unit_scaling",
         "std_width", "a_pretrain_key"],
)
def test_a_record_of_impossible_values_is_refused_naming_the_key(run, change, fragment):
    problems = predictor.record_problems({**record_of(run), **change})
    assert any(fragment in line for line in problems), problems


def test_a_resume_compares_the_geometry_the_scaling_the_patch_and_the_crops_and_nothing_of_the_widths_or_the_answer(run):
    from train.api.predictor import DATA_KEYS, record_differences

    record = record_of(run)
    assert set(DATA_KEYS[TASK]) == {"input_size", "channels", "normalize", "mean", "std", "patch_size", "n_local_crops"}
    for key, value in [("patch_size", 4), ("input_size", [64, 64]), ("channels", 1), ("normalize", "unit"), ("mean", [0.5, 0.5, 0.5]),
                       ("std", [0.5, 0.5, 0.5]), ("n_local_crops", 6)]:
        assert [line.split(":")[0] for line in record_differences(record, {**record, key: value})] == [key]
    # the widths are the model's (the request comparison holds them) and ``output`` is how the predictor answers, not what the run saw
    assert record_differences(record, {**record, "output": "something_else", "dino_out_dim": 4096, "embedding_dim": 384}) == []


@pytest.mark.parametrize("key, value", [("patch_size", 4), ("mean", [0.5, 0.5, 0.5]), ("n_local_crops", 6)])
def test_a_resume_of_a_run_on_another_patch_other_statistics_or_another_view_count_is_refused_by_the_record(run, tmp_path, key, value):
    from train.api.resume_decisions import check_data_unchanged

    shutil.copy(run.out / "task.json", tmp_path / "task.json")
    data = SimpleNamespace(n_train=1, n_select=1, n_test=1)
    resume = SimpleNamespace(state={"data_counts": {"n_train": 1, "n_select": 1, "n_test": 1}})
    check_data_unchanged(tmp_path, None, record_of(run), data, resume)  # the control: the same record is the same data
    with pytest.raises(ConfigError, match=rf"the data is not the data the run started with \({key}: "):
        check_data_unchanged(tmp_path, None, {**record_of(run), key: value}, data, resume)


# ---------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------


def test_load_model_returns_the_dino_predictor_and_a_predictor_of_another_task_is_refused_for_a_record_of_this_one(run, loaded):
    from train.api.dino_predictor import DinoPredictor
    from train.api.predictor import build_pretrain_record

    assert type(loaded) is DinoPredictor and loaded.record == record_of(run)
    other = build_pretrain_record(model="et_mim/et_tiny", family="et_mim", variant="et_tiny", input_size=(SIDE, SIDE), channels=3,
                                  mean=[0.5] * 3, std=[0.25] * 3, patch_size=PATCH, tokenizer=None)
    with pytest.raises(ConfigError, match=rf"the task is 'pretrain' but this predictor serves '{TASK}' only"):
        DinoPredictor(loaded.model, other)


def test_load_model_runs_the_training_model_on_the_rank_5_zero_batch_of_the_views_and_not_on_one_image(run, monkeypatch):
    seen = []
    real = predictor._zeros_of

    def spy(record):
        zeros = real(record)
        seen.append(zeros.shape)
        return zeros

    monkeypatch.setattr(predictor, "_zeros_of", spy)
    model = predictor.load_model(run.out).model
    assert seen == [(1, VIEWS, SIDE, SIDE, 3)] and predictor.input_shape_of(record_of(run)) == (VIEWS, SIDE, SIDE, 3)
    assert type(model).__name__ == "DINOTrainingModel", "the file a dino run saves is the whole training model"
    assert tuple(model(np.zeros(seen[0], np.float32), training=False).shape) == (2 * VIEWS - 2, 2 * OUT_DIM)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        model(np.zeros((1, SIDE, SIDE, 3), np.float32), training=False)   # the control: one image is not what the training model takes


@pytest.mark.parametrize(
    "change, fragment",
    [({"input_size": [64, 64]}, r"input_size \[64, 64\] and 3 channels but .* is built at \[32, 32, 3\]"),
     ({"patch_size": 4}, r"patch_size 4 \(64 tokens\) but .* cuts the image into 16 tokens"),
     ({"n_local_crops": 3}, r"n_local_crops 3 \(5 views\) but .* was built for 4 views"),
     ({"embedding_dim": 96}, r"embedding_dim 96 but .* gives an embedding of 192")],
    ids=["other_side", "other_patch", "other_view_count", "other_embedding"],
)
def test_load_model_refuses_a_record_that_is_not_the_models_before_the_model_is_called(run, tmp_path, monkeypatch, change, fragment):
    monkeypatch.setattr(predictor, "_run_on_zeros", lambda *args: pytest.fail("the model was called before its record was held to it"))
    copy = copy_run(run, tmp_path / "run", {**record_of(run), **change})
    with pytest.raises(ConfigError, match=fragment) as caught:
        predictor.load_model(copy)
    assert "the file and the model are not from one run" in str(caught.value)


def test_a_record_naming_the_wrong_view_count_gets_the_true_message_and_not_the_one_about_the_input_size(run, tmp_path):
    copy = copy_run(run, tmp_path / "run", {**record_of(run), "n_local_crops": 5})
    with pytest.raises(ConfigError) as caught:
        predictor.load_model(copy)
    text = str(caught.value)
    assert "n_local_crops 5 (7 views) but" in text and "was built for 4 views" in text
    assert "does not take" not in text and "input_size it names" not in text


def test_a_record_naming_another_head_than_the_models_is_refused_by_the_output_check_after_the_call(run, tmp_path):
    copy = copy_run(run, tmp_path / "run", {**record_of(run), "dino_out_dim": 128})
    with pytest.raises(ConfigError, match=rf"outputs \({2 * VIEWS - 2}, {2 * OUT_DIM}\) for one view set of {VIEWS} views but .* a head of 128 outputs"):
        predictor.load_model(copy)


def test_load_model_refuses_a_record_that_names_another_variant_than_the_model_and_loads_the_honest_one(run, tmp_path):
    honest = copy_run(run, tmp_path / "honest", record_of(run))
    assert type(predictor.load_model(honest)).__name__ == "DinoPredictor", "the control: the honest copy loads"
    lying = copy_run(run, tmp_path / "lying", {**record_of(run), "model": "dino_v1/small", "variant": "small", "embedding_dim": 384})
    with pytest.raises(ConfigError, match=r"is not that variant \(embed_dim is 192 in the model, 384 in 'small'"):
        predictor.load_model(lying)


def test_load_model_refuses_a_file_that_is_no_dino_model_by_name(run, tmp_path):
    directory = tmp_path / "other"
    directory.mkdir()
    keras.Sequential([keras.Input((SIDE, SIDE, 3)), keras.layers.Conv2D(3, 3, padding="same")]).save(str(directory / "best_model.keras"))
    (directory / "task.json").write_text(json.dumps(record_of(run)))
    with pytest.raises(ConfigError, match=r"dino_v1"):
        predictor.load_model(directory)


# ---------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------


def test_the_independent_numpy_transform_is_the_one_the_predictor_applies_bit_for_bit(run):
    from train.api.pretrain_predictor import model_input, read_image

    images = pictures(3, seed=1) + pictures(1, seed=2, shape=(48, 40, 3))
    record = record_of(run)
    assert np.array_equal(model_input([read_image(image) for image in images], record), np.stack([standardized(i, record) for i in images]))


def test_predict_gives_one_embedding_per_image_equal_to_the_feature_model_on_the_tests_own_input(run, loaded, reference):
    images = pictures(3, seed=1) + pictures(1, seed=2, shape=(48, 40, 3))
    got = loaded.predict(images)["embeddings"]
    expected = oracle(reference, images, record_of(run))
    assert isinstance(got, list) and len(got) == 4 and expected.shape == (4, EMBED)
    assert len({row.tobytes() for row in expected}) == 4, "the control: four images, four embeddings"
    assert np.abs(expected).max() > 0.1, "the control: the embeddings are not all near zero"
    for position, row in enumerate(got):
        assert row.dtype == np.float32 and row.shape == (EMBED,)
        assert np.allclose(row, expected[position], rtol=0.0, atol=1e-5), (position, float(np.abs(row - expected[position]).max()))


def test_the_embedding_is_the_cls_width_and_not_the_projection_heads_output_and_not_the_teachers(run, loaded, reference):
    record = record_of(run)
    images = pictures(2, seed=9)
    row = loaded.predict(images)["embeddings"][0]
    student = loaded.model.student
    head_output = np.asarray(student(np.stack([standardized(images[0], record)]), training=False))[0]
    assert head_output.shape == (OUT_DIM,) and row.shape == (EMBED,) and EMBED != OUT_DIM == record["dino_out_dim"]
    assert loaded.features.output_shape == (None, EMBED) == reference.output_shape
    assert loaded.features.outputs[0] is student.head.input, "the tensor the head reads, post final norm"
    # the teacher is a separate network of the same shape: its embedding of the same image is another vector once the EMA has moved it
    teacher = keras.Model(loaded.model.teacher.inputs[0], loaded.model.teacher.head.input)
    assert not np.array_equal(np.asarray(teacher(np.stack([standardized(images[0], record)]), training=False))[0], row)


def test_a_path_a_pil_image_and_an_array_of_one_picture_give_one_embedding(run, loaded, reference, tmp_path):
    pixels = pictures(1, seed=4)[0]
    path = tmp_path / "p.png"
    Image.fromarray(pixels).save(path)
    answers = loaded.predict([str(path), Image.fromarray(pixels), pixels, pixels.astype(np.float32) / 255.0])["embeddings"]
    assert len(answers) == 4 and all(a.dtype == np.float32 and a.shape == (EMBED,) for a in answers)
    for other in answers[1:]:  # one model input, so one embedding up to the float rounding of a row's place in a batch (measured below 1e-6)
        assert np.allclose(other, answers[0], rtol=0.0, atol=1e-5), "the four forms of one picture are one model input"
    assert np.allclose(loaded.predict(pixels)["embeddings"][0], answers[0], rtol=0.0, atol=1e-5), "one picture is a list of one"
    assert np.allclose(answers[0], oracle(reference, [pixels], record_of(run))[0], rtol=0.0, atol=1e-5)


def test_a_chunk_of_one_gives_the_answers_of_a_chunk_of_many(loaded, monkeypatch):
    import train.api.dino_predictor as module

    images = pictures(5, seed=6)
    together = loaded.predict(images)["embeddings"]
    monkeypatch.setattr(module, "CHUNK", 2)
    assert all(np.allclose(a, b, rtol=0.0, atol=1e-5) for a, b in zip(together, loaded.predict(images)["embeddings"]))


def flat(color: int) -> np.ndarray:
    return np.full((SIDE, SIDE, 3), color, np.uint8)


def with_patch(color: int, row: int, column: int, value: int) -> np.ndarray:
    image = flat(color)
    image[row * PATCH:(row + 1) * PATCH, column * PATCH:(column + 1) * PATCH] = value
    return image


def test_a_changed_patch_moves_the_embedding_and_the_flat_image_alone_does_not_move_it_wherever_it_sits_in_a_batch(loaded, capsys):
    background = flat(100)
    alone = loaded.predict(background)["embeddings"][0]
    crowd = pictures(5, seed=12)
    within = loaded.predict(crowd[:2] + [background] + crowd[2:])["embeddings"][2]
    noise = float(np.abs(alone - within).max())
    assert noise < 1e-5, f"the same flat image gave two embeddings {noise} apart: the answer depends on its neighbours"
    moved = {}
    for cell in [(0, 0), (1, 2), (2, 1), (3, 3)]:
        changed = loaded.predict(with_patch(100, *cell, 220))["embeddings"][0]
        moved[cell] = float(np.abs(changed - alone).max())
    # the DELTA: one patch of sixteen differs, and every embedding moves by far more than the noise of the flat image
    assert min(moved.values()) > 1e-3 and min(moved.values()) > 100 * max(noise, 1e-7), (moved, noise)
    a, b = (loaded.predict(with_patch(100, *cell, 220))["embeddings"][0] for cell in [(0, 0), (3, 3)])
    assert float(np.abs(a - b).max()) > 1e-3, "the same change at two cells is two embeddings: the model reads where it is"
    first, second = loaded.predict(pictures(2, seed=33))["embeddings"]
    assert float(np.abs(first - second).max()) > 1e-3, "two different images differ"
    with capsys.disabled():
        print(f"\nDELTA PROBE: flat-image noise {noise:.2e}; one patch changed moves the embedding by {min(moved.values()):.3f} to {max(moved.values()):.3f}")


@pytest.mark.parametrize("bad, fragment", [([], "list of images is empty"), (np.zeros((0, SIDE, SIDE, 3), np.uint8), "array of images is empty")])
def test_an_empty_call_is_refused_by_name(loaded, bad, fragment):
    with pytest.raises(ConfigError, match=fragment):
        loaded.predict(bad)


def image_folder(root: Path, count: int = 4) -> List[Path]:
    """``count`` PNGs, the last of another size, one of them in a sub-folder; returns their paths in the order the lister gives them."""
    root.mkdir(exist_ok=True)
    (root / "nested").mkdir(exist_ok=True)
    files = []
    for position, image in enumerate(pictures(count - 1, seed=1) + pictures(1, seed=2, shape=(48, 40, 3))):
        path = (root / "nested" if position == 1 else root) / f"p{position}.png"
        Image.fromarray(image).save(path)
        files.append(path)
    return sorted(files)


def test_predict_through_the_command_line_writes_one_npy_per_image_equal_to_the_oracle_and_the_index_last(run, reference, tmp_path):
    files = image_folder(tmp_path / "pictures")
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={run.out}", f"data={tmp_path / 'pictures'}", f"out={out}"]) == 0
    images = [np.asarray(Image.open(path)) for path in files]
    expected = oracle(reference, images, record_of(run))
    assert len({row.tobytes() for row in expected}) == 4, "the control: four images, four embeddings"
    stems = [path.stem for path in files]
    assert sorted(item.name for item in out.iterdir()) == sorted([f"{stem}.npy" for stem in stems] + ["index.json"])
    for stem, row in zip(stems, expected):
        got = np.load(out / f"{stem}.npy")
        assert got.dtype == np.float32 and got.shape == (EMBED,) and np.allclose(got, row, rtol=0.0, atol=1e-5), stem
    index = read_json(out / "index.json")
    assert list(index) == ["output", "note", "embeddings"] and index["output"] == "cls_embeddings" and "CLS" in index["note"]
    assert index["embeddings"] == [{"path": str(path), "file": f"{path.stem}.npy", "shape": [EMBED]} for path in files]
    assert (out / "index.json").stat().st_mtime_ns >= max((out / f"{stem}.npy").stat().st_mtime_ns for stem in stems), "the index is written last"


def test_the_lister_of_a_dino_run_is_the_image_lister_and_a_folder_without_an_image_is_refused(tmp_path):
    from train.api.predictor import DINO

    files = image_folder(tmp_path / "pictures")
    assert cli.DATA_LISTERS[DINO](tmp_path / "pictures") == files and cli.DATA_LISTERS[DINO](files[0]) == [files[0]]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="holds no image"):
        cli.DATA_LISTERS[DINO](tmp_path / "empty")
    assert set(cli.PREDICT_WRITERS) == set(cli.DATA_LISTERS) >= {DINO}


def test_a_predict_that_fails_half_way_leaves_its_arrays_and_no_index(run, tmp_path, monkeypatch, capsys):
    import train.api.dino_predictor as module

    monkeypatch.setattr(module, "CHUNK", 1)
    folder = tmp_path / "pictures"
    folder.mkdir()
    Image.fromarray(pictures(1, seed=3)[0]).save(folder / "a.png")
    (folder / "b.png").write_bytes(b"not an image")
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={run.out}", f"data={folder}", f"out={out}"]) == 2
    assert "b.png" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["a.npy"], "the index is written last: a broken answer has none"


def test_two_images_with_one_stem_are_refused_before_the_model_is_loaded(run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(predictor, "load_model", lambda path: pytest.fail("the model was loaded"))
    folder = tmp_path / "pictures"
    (folder / "x").mkdir(parents=True)
    (folder / "y").mkdir()
    for sub in ("x", "y"):
        Image.fromarray(pictures(1, seed=3)[0]).save(folder / sub / "same.png")
    assert cli.main(["predict", f"model={run.out}", f"data={folder}", f"out={tmp_path / 'answers'}"]) == 2
    assert "same.png" in capsys.readouterr().err and not (tmp_path / "answers").exists()


def test_predict_help_and_export_help_say_what_a_dino_model_takes_and_gives(capsys):
    assert cli.main(["predict", "help"]) == 0
    text = capsys.readouterr().out
    assert "a dino model a directory of float32 .npy CLS embeddings" in text
    assert cli.main(["export", "help"]) == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "a dino model takes float32 images" in text and "CLS embedding" in text and "(batch, embedding_dim)" in text
    assert "projection head" in text and "no teacher" in text


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(run, tmp_path_factory):
    """Each format written once through the command line, the way a user runs it: ``{format: path}``."""
    root = tmp_path_factory.mktemp("dino_exports")
    paths = {}
    for fmt in FORMATS:
        paths[fmt] = root / f"exported{SUFFIX.get(fmt, '')}"
        assert cli.main(["export", f"model={run.out}", f"out={paths[fmt]}", f"format={fmt}"]) == 0, fmt
    return paths


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_export_gives_the_embeddings_predict_gives_and_the_record_beside_it(run, loaded, exported, fmt):
    from train.api.pretrain_predictor import model_input

    out = exported[fmt]
    assert read_json(out.with_name(out.name + ".task.json")) == record_of(run)
    images = pictures(8, seed=5)
    expected = np.stack(loaded.predict(images)["embeddings"]).astype(np.float64)
    got = np.asarray(read_back(fmt, out, model_input(images, loaded.record)), np.float64)
    assert got.shape == expected.shape == (8, EMBED)
    scale = float(np.abs(expected).max())
    assert scale > 0.1, "the control: the embeddings are not all near zero"
    gap = float(np.abs(got - expected).max())
    assert gap <= 1e-4 * max(1.0, scale), f"{fmt} differs from predict by {gap} (scale {scale})"
    assert len({row.tobytes() for row in got}) == 8, "eight images, eight embeddings"


@pytest.mark.parametrize("fmt", FORMATS)
def test_another_input_shape_is_refused_before_a_file_is_written(run, loaded, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    with pytest.raises(ConfigError, match=rf"\(48, 48, 3\) is not the trained size \({SIDE}, {SIDE}, 3\)"):
        exporting.export(model=loaded.model, out=out, format=fmt, input_shape=(48, 48, 3), record=record_of(run))
    assert list(tmp_path.iterdir()) == []


def test_the_export_rule_has_a_served_model_no_refused_cell_and_the_trained_size_and_the_view_set_is_not_its_shape(run):
    from train.api.dino_predictor import check_export
    from train.api.predictor import EXPORT_RULES

    rule = EXPORT_RULES[TASK]
    record = record_of(run)
    assert rule.served is not None and rule.sample is None and rule.shape_of(record) == (SIDE, SIDE, 3)
    assert predictor.input_shape_of(record) == (VIEWS, SIDE, SIDE, 3) != rule.shape_of(record), "the training model takes views, the file an image"
    for fmt in FORMATS:                                  # every cell of the table is exported (measured), none is refused
        rule.check(record, fmt, (SIDE, SIDE, 3))
        with pytest.raises(ConfigError, match=r"\(48, 32, 3\) is not the trained size"):
            check_export(record, fmt, (48, 32, 3))


def test_the_served_model_is_the_feature_model_of_the_loaded_student_and_its_input_is_one_image(run, loaded):
    served = predictor.EXPORT_RULES[TASK].served(loaded.model, record_of(run))
    batch = np.stack([standardized(image, record_of(run)) for image in pictures(3, seed=8)])
    assert served.input_shape == (None, SIDE, SIDE, 3) and served.output_shape == (None, EMBED)
    assert np.array_equal(np.asarray(served(batch, training=False)), np.asarray(loaded.features(batch, training=False)))


def test_the_shared_tflite_check_judges_the_embeddings_and_a_file_that_returns_others_is_refused(run, loaded, tmp_path, monkeypatch):
    real = exporting._tflite_logits
    monkeypatch.setattr(exporting, "_tflite_logits", lambda blob, batch: np.asarray(real(blob, batch)) + 0.05)
    out = tmp_path / "wrong.tflite"
    with pytest.raises(ConfigError, match=r"the tflite file converted with TensorFlow operators differs from the Keras model by"):
        exporting.export(model=loaded.model, out=out, format="tflite", input_shape=(SIDE, SIDE, 3), record=record_of(run))
    assert not out.exists() and not (tmp_path / "wrong.tflite.task.json").exists()
    monkeypatch.undo()                                   # the control: the same export with the real reader is written
    exporting.export(model=loaded.model, out=tmp_path / "right.tflite", format="tflite", input_shape=(SIDE, SIDE, 3), record=record_of(run))
    assert (tmp_path / "right.tflite").is_file()


def bytes_under(path: Path) -> int:
    return path.stat().st_size if path.is_file() else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def test_the_limit_the_export_note_states_is_still_real_a_savedmodel_carries_the_training_models_weights(loaded, exported, capsys):
    # `exporting._export_saved_model` tracks the model it is given, which is the TRAINING model (the teacher and both projection heads ride in
    # the archive beside the feature graph the endpoint traces). Measured at the default size: the SavedModel 117 MB against 22 MB for the keras, tflite
    # and onnx files. When the archive tracks only the served model this fails: then lift the sentence of `dino_predictor.export_note` and of
    # the help (`cli.EXPORT_NOTES`).
    sizes = {fmt: bytes_under(exported[fmt]) for fmt in FORMATS}
    variables = sum(item.stat().st_size for item in (exported["saved_model"] / "variables").glob("variables.data*"))
    feature = sum(int(np.prod(w.shape)) * 4 for w in loaded.features.weights)
    training = sum(int(np.prod(w.shape)) * 4 for w in loaded.model.weights)
    with capsys.disabled():
        print(f"\nSIZES (bytes): {sizes}; SavedModel variables {variables}; feature model weights {feature}; training model weights {training}")
    assert feature < training and variables > 1.5 * feature, "the SavedModel's variables are more than the feature model's weights"
    assert sizes["saved_model"] > 2 * sizes["keras"] and sizes["onnx"] < sizes["saved_model"], sizes


_CHILD_WITHOUT_TRAIN = """
import importlib.util
import os
import sys

import numpy as np

sys.path = [p for p in sys.path if not os.path.isdir(os.path.join(p, "train"))]
sys.path.insert(0, sys.argv[2])
import keras

assert importlib.util.find_spec("train") is None, "the train package is importable in this process"
model = keras.saving.load_model(sys.argv[1])
np.save(sys.argv[4], np.asarray(model(np.load(sys.argv[3]), training=False)))
"""


def test_an_exported_keras_file_holds_library_layers_only_and_loads_where_the_train_package_cannot_be_imported(run, loaded, exported, tmp_path):
    """The note says so, so this is the proof: the file is the functional feature model of the student (``dl_techniques`` layers), unlike a
    tokenizer's or a forecaster's, which hold a class of ``train``."""
    from train.api.pretrain_predictor import model_input

    only = tmp_path / "only_dl_techniques"
    only.mkdir()
    (only / "dl_techniques").symlink_to(SRC / "dl_techniques", target_is_directory=True)
    script = tmp_path / "child.py"
    script.write_text(_CHILD_WITHOUT_TRAIN)
    images = pictures(3, seed=14)
    np.save(tmp_path / "in.npy", model_input(images, loaded.record))
    done = subprocess.run([sys.executable, str(script), str(exported["keras"]), str(only), str(tmp_path / "in.npy"), str(tmp_path / "out.npy")],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-800:] + done.stderr[-1500:]
    assert np.allclose(np.load(tmp_path / "out.npy"), np.stack(loaded.predict(images)["embeddings"]), rtol=0.0, atol=1e-5)


def test_the_export_note_says_what_goes_in_what_comes_out_and_that_the_file_needs_the_library_only(run):
    note = predictor.EXPORT_RULES[TASK].note(record_of(run))
    for fragment in ("float32", f"(batch, {SIDE}, {SIDE}, 3)", "the trained size only", "(x - mean) / std", "CLS embedding", f"(batch, {EMBED})",
                     "not the projection head's output", "no teacher", "dl_techniques", "train package"):
        assert fragment in note, (fragment, note)
