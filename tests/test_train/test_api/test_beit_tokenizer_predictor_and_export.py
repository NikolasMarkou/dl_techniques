"""The ``beit_tokenizer`` record, ``predict`` and ``export`` (plan-2026-11-25T124419-fa94f59c step 7, D-044 to D-048).

A tokenizer run writes a schema-v2 ``task.json`` (``task: "beit_tokenizer"``, the ninth kind) before ``fit``, and its directory can be
predicted with and exported. Both answer the CODE GRID: one int32 id per cell of ``image_size // patch_size`` on a side, the index of the
codebook entry nearest the cell's encoding, in row-major order. Pinned here, on one real tiny run on the fake CIFAR-10 (64 codes of 8
numbers, 16 channels wide, one residual block, patch 4: an 8 x 8 grid), CPU, written under ``tmp_path``:

* **the record**: it passes ``record_problems`` and equals the record the test builds itself; a record missing any key, or holding an
  impossible value (a non-square size, a patch that does not divide it, another output word, a codebook that is not a positive int), is
  refused naming the key; a resume compares the geometry, the scaling and the patch and nothing of the codebook or of how the predictor
  answers; ``load_model`` refuses, before the model is called, a record whose size, patch or codebook is not the model's, one that names
  another variant than the model, and a model that is no tokenizer;
* **predict**: every grid equals the test's own ``encode_to_indices`` of the image standardized in numpy, whatever the image's size or
  form; a DELTA probe on a tokenizer whose code at a cell depends on that cell's patch alone puts the changed code at exactly the cell
  of the changed patch (a transposed grid fails it); the index is written last;
* **export** (measured, ``scratch_export/measured.md``): each of the four formats gives the grid ``predict`` gives; another size is
  refused before a file is written; the shared tflite check judges codes, so a file that returns another grid is refused; and the limit
  the export note states is TRUE: two codes a float rounding apart are told apart differently by a file than by Keras (an exact
  duplicate takes the lowest code in every format).
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
from tests.classification_fakes import use_identity_cifar10
from train.api import cli, exporting, predictor, recipes, registry
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, SUFFIX
from train.api.tasks import register_builtin_tasks
from train.api.tasks.beit_tokenizer import train_beit_tokenizer

from ._tokenizer_world import TINY_MODEL_ARGS, read_json, request
from .test_export_roundtrip import read_back
from .test_forecast_predictor_and_export import _CHILD_WITHOUT_TRAIN   # one child script: a process in which ``train`` cannot be imported

SRC = Path(__file__).resolve().parents[3] / "src"
TASK = "beit_tokenizer"
SIDE, PATCH, CODES = 32, 4, TINY_MODEL_ARGS["num_embeddings"]
GRID = SIDE // PATCH
#: The keys of a tokenizer record, in the order they are written (D-044).
KEYS = ("schema_version", "task", "model", "family", "variant", "input_size", "channels", "normalize", "mean", "std", "patch_size",
        "num_embeddings", "output")
#: What the tiny run changes of the ``small`` variant's table, which the record carries so that ``load_model`` can hold the model to it.
OVERRIDES = {"hidden_channels": 16, "num_res_blocks": 1, "embedding_dim": 8}
#: Pairs of codebook rows the near-tie tests build (rows ``2k`` and ``2k + 1``).
PAIRS = 16


@pytest.fixture(scope="module")
def world():
    """The identity fake of CIFAR-10 and an empty registry holding every built-in family and task (no real data is read)."""
    with pytest.MonkeyPatch.context() as patch:
        use_identity_cifar10(patch, n_train=200, n_test=40)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(scope="module")
def run(world, tmp_path_factory):
    """One finished 1-epoch run (the record is read by each test: a missing one fails the test, not this fixture) and what ``load_model`` makes of it."""
    out = tmp_path_factory.mktemp("tokenizer_record") / "run"
    train_beit_tokenizer(**request(out, epochs=1, batch_size=16, **{"data.max_samples": None}))
    return SimpleNamespace(out=out, predictor=predictor.load_model(out) if (out / "task.json").exists() else None)


def record_of(run: SimpleNamespace) -> Dict[str, Any]:
    return read_json(run.out / "task.json")


def pictures(count: int, seed: int, shape: Tuple[int, int, int] = (SIDE, SIDE, 3)) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(count)]


def standardized(image: np.ndarray, record: Dict[str, Any]) -> np.ndarray:
    """The model input of one ``uint8`` image, written out here and not read from the module: bytes / 255, a bilinear resize to the trained
    side where the size differs, ``(x - mean) / std`` with the record's statistics, all in float32."""
    unit = image.astype(np.float32) / np.float32(255.0)
    if unit.shape[:2] != (SIDE, SIDE):
        unit = tf.image.resize(unit, (SIDE, SIDE), method="bilinear").numpy()
    return (unit - np.asarray(record["mean"], np.float32)) / np.asarray(record["std"], np.float32)


def oracle(model: Any, images: List[np.ndarray], record: Dict[str, Any]) -> np.ndarray:
    """The test's own code grids: the model's ``encode_to_indices`` of the numpy-standardized images, ``(n, grid, grid)``."""
    return np.asarray(model.encode_to_indices(np.stack([standardized(image, record) for image in images])))


def reloaded(run: SimpleNamespace) -> Any:
    return keras.models.load_model(str(run.out / "best_model.keras"), compile=False)


def save_run(directory: Path, model: Any, record: Dict[str, Any]) -> Path:
    """Write ``best_model.keras`` and ``task.json`` into the new ``directory``: the run directory ``load_model`` reads."""
    directory.mkdir()
    model.save(str(directory / "best_model.keras"))
    (directory / "task.json").write_text(json.dumps(record))
    return directory


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


def test_the_record_passes_the_reader_and_is_the_one_the_test_builds_from_what_the_run_trained_with(run):
    from train.api.args import TransformArgs
    from train.api.predictor import OPTIONAL_RECORD_KEYS, RECORD_KEYS, build_tokenizer_record, record_problems
    from train.common.classification_data import _stats

    record = record_of(run)
    assert record_problems(record) == []
    assert RECORD_KEYS[TASK] == KEYS and tuple(record)[: len(KEYS)] == KEYS and set(record) - set(KEYS) <= set(OPTIONAL_RECORD_KEYS)
    assert (record["schema_version"], record["task"], record["model"], record["family"], record["variant"]) == (
        2, TASK, "vq_vae_rotation/small", "vq_vae_rotation", "small")
    assert (record["input_size"], record["channels"], record["patch_size"], record["num_embeddings"]) == ([SIDE, SIDE], 3, PATCH, CODES)
    # the word, mean and std image_classification records for the same dataset hold (its default)
    mean, std = _stats(get_spec("cifar10"), 3, TransformArgs().normalize)
    assert record["normalize"] == TransformArgs().normalize == "standardize"
    assert record["mean"] == [float(v) for v in mean] and record["std"] == [float(v) for v in std]
    assert record["output"] == "code_ids" and record["variant_overrides"] == OVERRIDES
    assert record == build_tokenizer_record(
        model="vq_vae_rotation/small", family="vq_vae_rotation", variant="small", input_size=(SIDE, SIDE), channels=3, mean=mean,
        std=std, patch_size=PATCH, num_embeddings=CODES, variant_overrides=OVERRIDES)


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
        ({"num_embeddings": 0}, "'num_embeddings'"),
        ({"num_embeddings": 64.0}, "'num_embeddings'"),
        ({"num_embeddings": True}, "'num_embeddings'"),
        ({"output": "reconstructions"}, "'output'"),
        ({"normalize": "unit"}, "'normalize'"),
        ({"std": [0.2, 0.2]}, "'std'"),
        ({"tokenizer": "ab" * 32}, "unknown key"),
    ],
    ids=["not_square", "patch_does_not_divide", "patch_zero", "patch_not_dividing_either", "grey", "no_codes", "codes_a_float",
         "codes_a_bool", "other_output", "unit_scaling", "std_width", "a_pretrain_key"],
)
def test_a_record_of_impossible_values_is_refused_naming_the_key(run, change, fragment):
    problems = predictor.record_problems({**record_of(run), **change})
    assert any(fragment in line for line in problems), problems


def test_a_resume_compares_the_geometry_the_scaling_and_the_patch_and_nothing_of_the_codebook_or_the_answer(run):
    from train.api.predictor import DATA_KEYS, record_differences

    record = record_of(run)
    assert set(DATA_KEYS[TASK]) == {"input_size", "channels", "normalize", "mean", "std", "patch_size"}
    for key, value in [("patch_size", 8), ("input_size", [64, 64]), ("channels", 1), ("normalize", "unit"), ("mean", [0.5, 0.5, 0.5]),
                       ("std", [0.5, 0.5, 0.5])]:
        assert [line.split(":")[0] for line in record_differences(record, {**record, key: value})] == [key]
    # the codebook is the model's (the request comparison holds it) and ``output`` is how the predictor answers, not what the run saw
    assert record_differences(record, {**record, "output": "something_else", "num_embeddings": 128}) == []


@pytest.mark.parametrize("key, value", [("patch_size", 8), ("mean", [0.5, 0.5, 0.5])])
def test_a_resume_of_a_run_on_another_patch_or_other_statistics_is_refused_by_the_record(run, tmp_path, key, value):
    from train.api.resume_decisions import check_data_unchanged

    shutil.copy(run.out / "task.json", tmp_path / "task.json")
    data = SimpleNamespace(n_train=1, n_select=1, n_test=1)
    resume = SimpleNamespace(state={"data_counts": {"n_train": 1, "n_select": 1, "n_test": 1}})
    check_data_unchanged(tmp_path, None, record_of(run), data, resume)  # the control: the same record is the same data
    with pytest.raises(ConfigError, match=rf"the data is not the data the run started with \({key}: "):
        check_data_unchanged(tmp_path, None, {**record_of(run), key: value}, data, resume)


def test_load_model_returns_the_tokenizer_predictor_and_a_predictor_of_another_task_is_refused_for_a_record_of_this_one(run):
    from train.api.predictor import build_pretrain_record
    from train.api.tokenizer_predictor import TokenizerPredictor

    assert type(run.predictor) is TokenizerPredictor and run.predictor.record == record_of(run)
    other = build_pretrain_record(model="et_mim/et_tiny", family="et_mim", variant="et_tiny", input_size=(SIDE, SIDE), channels=3,
                                  mean=[0.5] * 3, std=[0.25] * 3, patch_size=PATCH, tokenizer=None)
    with pytest.raises(ConfigError, match=rf"the task is 'pretrain' but this predictor serves '{TASK}' only"):
        TokenizerPredictor(run.predictor.model, other)


@pytest.mark.parametrize(
    "change, fragment",
    [({"input_size": [64, 64]}, r"input_size \[64, 64\] and 3 channels but .* is built at \[32, 32, 3\]"),
     ({"patch_size": 8}, r"patch_size 8 but .* total stride of 4"),
     ({"num_embeddings": 128}, r"num_embeddings 128 but .* codebook holds 64 codes")],
    ids=["other_side", "other_patch", "other_codebook"],
)
def test_load_model_refuses_a_record_that_is_not_the_models_before_the_model_is_called(run, tmp_path, monkeypatch, change, fragment):
    monkeypatch.setattr(predictor, "_run_on_zeros", lambda *args: pytest.fail("the model was called before its record was held to it"))
    copy = save_run(tmp_path / "run", reloaded(run), {**record_of(run), **change})
    with pytest.raises(ConfigError, match=fragment) as caught:
        predictor.load_model(copy)
    assert "the file and the model are not from one run" in str(caught.value)


def test_load_model_refuses_a_record_that_names_another_variant_than_the_model_and_loads_the_honest_one(run, tmp_path):
    honest = save_run(tmp_path / "honest", reloaded(run), record_of(run))
    assert type(predictor.load_model(honest)).__name__ == "TokenizerPredictor", "the control: the honest copy loads"
    without = {key: value for key, value in record_of(run).items() if key != "variant_overrides"}
    lying = save_run(tmp_path / "lying", reloaded(run), without)
    with pytest.raises(ConfigError, match=r"is not that variant \(hidden_channels is 16 in the model, 128 in 'small'"):
        predictor.load_model(lying)


def test_load_model_refuses_a_file_that_is_no_tokenizer_by_name(run, tmp_path):
    other = keras.Sequential([keras.Input((SIDE, SIDE, 3)), keras.layers.Conv2D(3, 3, padding="same")])
    copy = save_run(tmp_path / "other", other, record_of(run))
    with pytest.raises(ConfigError, match=r"names the family 'vq_vae_rotation' but .* holds no layer of that family"):
        predictor.load_model(copy)


# ---------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------


def test_the_independent_numpy_transform_is_the_one_the_predictor_applies_bit_for_bit(run):
    from train.api.pretrain_predictor import model_input, read_image

    images = pictures(3, seed=1) + pictures(1, seed=2, shape=(48, 40, 3))
    record = record_of(run)
    assert np.array_equal(model_input([read_image(image) for image in images], record), np.stack([standardized(i, record) for i in images]))


def test_predict_writes_one_code_grid_per_image_equal_to_the_models_codes_on_the_tests_own_input(run, tmp_path):
    images = pictures(3, seed=1) + pictures(1, seed=2, shape=(48, 40, 3))
    folder = tmp_path / "pictures"
    folder.mkdir()
    for position, image in enumerate(images):
        Image.fromarray(image).save(folder / f"p{position}.png")
    out = tmp_path / "answers"
    assert cli.main(["predict", f"model={run.out}", f"data={folder}", f"out={out}"]) == 0
    expected = oracle(reloaded(run), images, record_of(run))
    assert expected.shape == (4, GRID, GRID) and len(np.unique(expected)) > 1, "the control: a grid of ids, not one id everywhere"
    assert len({expected[i].tobytes() for i in range(4)}) == 4, "the control: four images, four grids"
    for position in range(4):
        got = np.load(out / f"p{position}.npy")
        assert got.dtype == np.int32 and got.shape == (GRID, GRID) and np.array_equal(got, expected[position]), position
    index = read_json(out / "index.json")
    assert list(index) == ["output", "note", "codes"] and index["output"] == "code_ids" and index["note"]
    assert index["codes"] == [
        {"path": str(folder / f"p{position}.png"), "file": f"p{position}.npy", "shape": [GRID, GRID]} for position in range(4)]


def test_a_path_a_pil_image_and_an_array_of_one_picture_give_one_grid(run, tmp_path):
    pixels = pictures(1, seed=4)[0]
    path = tmp_path / "p.png"
    Image.fromarray(pixels).save(path)
    answers = run.predictor.predict([str(path), Image.fromarray(pixels), pixels, pixels.astype(np.float32) / 255.0])["codes"]
    assert len(answers) == 4 and all(a.dtype == np.int32 and a.shape == (GRID, GRID) for a in answers)
    for other in answers[1:]:
        assert np.array_equal(other, answers[0]), "the four forms of one picture are one model input"
    assert np.array_equal(run.predictor.predict(pixels)["codes"][0], answers[0]), "one picture is a list of one"
    assert np.array_equal(answers[0], oracle(reloaded(run), [pixels], record_of(run))[0])


def test_a_chunk_of_one_gives_the_answers_of_a_chunk_of_many(run, monkeypatch):
    import train.api.tokenizer_predictor as module

    images = pictures(5, seed=6)
    together = run.predictor.predict(images)["codes"]
    monkeypatch.setattr(module, "CHUNK", 2)
    assert all(np.array_equal(a, b) for a, b in zip(together, run.predictor.predict(images)["codes"]))


def local_tokenizer(codes: int = 16, dim: int = 4, decoder_scale: int = PATCH) -> Any:
    """A tokenizer whose code at a cell depends on that cell's patch ALONE: its encoder is one convolution whose kernel is its stride.

    ``decoder_scale`` is the factor its decoder grows the grid by: ``PATCH`` gives an image of the encoder's size, another a reconstruction of
    another size (the model of ``test_a_model_whose_reconstruction...``).
    """
    from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick

    keras.utils.set_random_seed(3)
    image = keras.Input((SIDE, SIDE, 3))
    encoder = keras.Model(image, keras.layers.Conv2D(dim, PATCH, strides=PATCH, padding="same")(image))
    latent = keras.Input((GRID, GRID, dim))
    decoder = keras.Model(latent, keras.layers.Conv2DTranspose(3, decoder_scale, strides=decoder_scale, padding="same")(latent))
    model = VQVAERotationTrick(num_embeddings=codes, embedding_dim=dim, encoder=encoder, decoder=decoder,
                               input_shape=(SIDE, SIDE, 3), downsample_factor=PATCH)
    model.build((None, SIDE, SIDE, 3))
    return model


def test_a_changed_patch_changes_the_code_of_exactly_its_own_cell_and_a_transposed_grid_fails_it(run, tmp_path):
    record = {**record_of(run), "num_embeddings": 16, "variant_overrides": {"embedding_dim": 4}}
    model = local_tokenizer()
    loaded = predictor.load_model(save_run(tmp_path / "local", model, record))

    def code_of(color: int) -> int:
        flat = np.full((SIDE, SIDE, 3), color, np.uint8)
        codes = np.asarray(model.encode_to_indices(standardized(flat, record)[None]))
        assert (codes == codes[0, 0, 0]).all(), "a flat image has one code in every cell"
        return int(codes[0, 0, 0])

    background = 0
    foreground = next((color for color in range(8, 256, 8) if code_of(color) != code_of(background)), None)
    assert foreground is not None, "the control: some colour has another code than the background's"
    for row, column in [(1, 6), (6, 1), (0, 0), (7, 7), (3, 5), (5, 3)]:
        image = np.zeros((SIDE, SIDE, 3), np.uint8)
        image[row * PATCH:(row + 1) * PATCH, column * PATCH:(column + 1) * PATCH] = foreground
        got = loaded.predict(image)["codes"][0]
        expected = np.full((GRID, GRID), code_of(background))
        expected[row, column] = code_of(foreground)
        assert np.array_equal(got, expected), f"patch ({row}, {column}) changed {np.argwhere(got != code_of(background)).tolist()}"
        assert (got != code_of(background)).sum() == 1


def test_a_model_whose_reconstruction_is_not_the_records_size_is_refused_by_the_output_check(run, tmp_path):
    # The config, the stride and the code grid of this model are the record's; only its decoder grows the grid to another size.
    record = {**record_of(run), "num_embeddings": 16, "variant_overrides": {"embedding_dim": 4}}
    copy = save_run(tmp_path / "small_decoder", local_tokenizer(decoder_scale=2), record)
    with pytest.raises(ConfigError, match=r"outputs \(1, 16, 16, 3\) for one 32x32 image but .* reconstruction is \(1, 32, 32, 3\)"):
        predictor.load_model(copy)
    honest = save_run(tmp_path / "honest_decoder", local_tokenizer(), record)
    assert type(predictor.load_model(honest)).__name__ == "TokenizerPredictor", "the control: the same model with a decoder of the record's size loads"


@pytest.mark.parametrize("bad, fragment", [([], "list of images is empty"), (np.zeros((0, SIDE, SIDE, 3), np.uint8), "array of images is empty")])
def test_an_empty_call_is_refused_by_name(run, bad, fragment):
    with pytest.raises(ConfigError, match=fragment):
        run.predictor.predict(bad)


def test_a_predict_that_fails_half_way_leaves_its_arrays_and_no_index(run, tmp_path, monkeypatch, capsys):
    import train.api.tokenizer_predictor as module

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


def test_predict_help_and_export_help_say_what_a_tokenizer_takes_and_gives(capsys):
    assert cli.main(["predict", "help"]) == 0
    assert "code grid" in capsys.readouterr().out
    assert cli.main(["export", "help"]) == 0
    text = capsys.readouterr().out
    assert "a tokenizer takes float32 images" in text and "int32 code ids" in text and "train package" in text


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_export_gives_the_grid_predict_gives_and_the_record_beside_it(run, fmt, tmp_path):
    from train.api.pretrain_predictor import model_input

    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    assert cli.main(["export", f"model={run.out}", f"out={out}", f"format={fmt}"]) == 0
    assert read_json(out.with_name(out.name + ".task.json")) == record_of(run)
    images = pictures(8, seed=5)
    expected = np.stack(run.predictor.predict(images)["codes"])
    got = np.asarray(read_back(fmt, out, model_input(images, run.predictor.record)))
    assert got.dtype == np.int32 and got.shape == expected.shape == (8, GRID, GRID)
    assert np.array_equal(got, expected), f"{int((got != expected).sum())} of {expected.size} positions differ"
    assert len(np.unique(expected)) > 1, "the control: the grids hold more than one code"


@pytest.mark.parametrize("fmt", FORMATS)
def test_another_input_shape_is_refused_before_a_file_is_written(run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    with pytest.raises(ConfigError, match=r"\(48, 48, 3\) is not the trained size \(32, 32, 3\)"):
        exporting.export(model=run.predictor.model, out=out, format=fmt, input_shape=(48, 48, 3), record=record_of(run))
    assert list(tmp_path.iterdir()) == []


def test_the_export_rule_has_a_served_layer_no_refused_cell_and_the_trained_size(run):
    from train.api.predictor import EXPORT_RULES
    from train.api.tokenizer_predictor import check_export

    rule = EXPORT_RULES[TASK]
    assert rule.served is not None and rule.sample is None and rule.shape_of(record_of(run)) == (SIDE, SIDE, 3)
    for fmt in FORMATS:                                  # every cell of the table is exported (16 of 16 measured), none is refused
        rule.check(record_of(run), fmt, (SIDE, SIDE, 3))
        with pytest.raises(ConfigError, match=r"\(48, 32, 3\) is not the trained size"):
            check_export(record_of(run), fmt, (48, 32, 3))


def test_the_served_layer_returns_int32_grids_and_names_its_model_in_its_config(run):
    from train.common.tokenizer_data import TokenizerCodes

    layer = predictor.EXPORT_RULES[TASK].served(run.predictor.model, record_of(run))
    batch = np.stack([standardized(image, record_of(run)) for image in pictures(3, seed=8)])
    codes = np.asarray(layer(batch))
    assert isinstance(layer, TokenizerCodes) and codes.dtype == np.int32 and codes.shape == (3, GRID, GRID)
    assert np.array_equal(codes, np.asarray(run.predictor.model.encode_to_indices(batch)))
    # the weights ride in the archive beside this config (the keras export test loads that archive); the config alone names the model
    config = json.loads(json.dumps(layer.get_config()))
    assert config["model"]["class_name"] == "VQVAERotationTrick" and config["model"]["config"]["num_embeddings"] == CODES


def test_the_shared_tflite_check_judges_the_codes_and_a_file_that_returns_another_grid_is_refused(run, tmp_path, monkeypatch):
    real = exporting._tflite_logits
    monkeypatch.setattr(exporting, "_tflite_logits", lambda blob, batch: np.swapaxes(real(blob, batch), 1, 2))
    out = tmp_path / "wrong.tflite"
    with pytest.raises(ConfigError, match=r"the tflite file converted with TensorFlow operators differs from the Keras model by"):
        exporting.export(model=run.predictor.model, out=out, format="tflite", input_shape=(SIDE, SIDE, 3), record=record_of(run))
    assert not out.exists() and not (tmp_path / "wrong.tflite.task.json").exists()
    monkeypatch.undo()                                   # the control: the same export with the real reader is written
    exporting.export(model=run.predictor.model, out=tmp_path / "right.tflite", format="tflite", input_shape=(SIDE, SIDE, 3),
                     record=record_of(run))
    assert (tmp_path / "right.tflite").is_file()


def test_an_exported_keras_file_does_not_load_in_a_process_where_the_train_package_cannot_be_imported(run, tmp_path):
    """The layer the file holds is a class of ``train`` (``TokenizerCodes``), which the wheel does not ship: the note, the help and the README say the
    file loads only where ``train`` is importable, and this is the failure a user meets (Keras's own sentence, named by the class and the module)."""
    out = tmp_path / "exported.keras"
    assert cli.main(["export", f"model={run.out}", f"out={out}", "format=keras"]) == 0
    only = tmp_path / "only_dl_techniques"
    only.mkdir()
    (only / "dl_techniques").symlink_to(SRC / "dl_techniques", target_is_directory=True)
    script = tmp_path / "child.py"
    script.write_text(_CHILD_WITHOUT_TRAIN)
    done = subprocess.run([sys.executable, str(script), str(out), str(only)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 3, done.stdout[-800:] + done.stderr[-800:]
    assert "TokenizerCodes" in done.stdout and "train.common.tokenizer_data cannot be imported" in done.stdout, done.stdout[-800:]


def near_ties(run: SimpleNamespace, tmp_path: Path, delta: float) -> Tuple[Path, Any]:
    """A copy of the run whose codebook holds PAIRS of rows ``delta`` apart (0: exact duplicates), and its model: the stress a trained
    codebook can approach, where many positions have two codes within float32 rounding of each other."""
    model = reloaded(run)
    book = np.asarray(model.quantizer.embeddings, np.float32).copy()
    rng = np.random.default_rng(7)
    for pair in range(PAIRS):
        direction = rng.standard_normal(book.shape[-1]).astype(np.float32)
        book[0, 2 * pair + 1] = book[0, 2 * pair] + np.float32(delta) * direction / np.linalg.norm(direction)
    model.quantizer.embeddings.assign(book)
    return save_run(tmp_path / f"ties_{delta}", model, record_of(run)), model


def normal_images(count: int) -> np.ndarray:
    return np.random.default_rng(1).standard_normal((count, SIDE, SIDE, 3)).astype(np.float32)


def test_an_exact_tie_takes_the_lowest_code_in_every_format(run, tmp_path, capsys):
    copy, model = near_ties(run, tmp_path, 0.0)
    batch = normal_images(64)
    expected = np.asarray(model.encode_to_indices(batch))
    first, second = np.arange(0, 2 * PAIRS, 2), np.arange(1, 2 * PAIRS, 2)
    assert np.isin(expected, first).sum() > 0, "the control: positions do sit on a duplicated pair"
    assert not np.isin(expected, second).any(), "an exact tie is the lowest index, in Keras"
    with capsys.disabled():
        print(f"\nEXACT TIES: {int(np.isin(expected, first).sum())} of {expected.size} positions sit on a duplicated pair")
    for fmt in FORMATS:
        out = tmp_path / f"exact_{fmt}{SUFFIX.get(fmt, '')}"
        assert cli.main(["export", f"model={copy}", f"out={out}", f"format={fmt}"]) == 0
        got = np.asarray(read_back(fmt, out, batch))
        assert np.array_equal(got, expected), f"{fmt}: {int((got != expected).sum())} of {expected.size} positions differ"


def test_the_limit_the_export_note_states_is_still_real_two_codes_a_rounding_apart_are_told_apart_differently_by_a_file(run, tmp_path, capsys):
    # Measured (scratch_export/measured.md): codes whose distances differ by less than the float32 resolution of the distance (a squared gap
    # under about 1e-4) are picked differently by a SavedModel, a TFLite file and an ONNX file than by Keras, at a few positions in a hundred
    # of a codebook built to have many such pairs, and at none of 32,768 positions of ordinary tokenizers. When a converter becomes exact
    # here this test fails: then lift the sentence of `tokenizer_predictor.export_note` and of the README that states the limit.
    copy, model = near_ties(run, tmp_path, 1e-6)
    batch = normal_images(128)
    expected = np.asarray(model.encode_to_indices(batch))
    out = tmp_path / "ties_saved_model"
    assert cli.main(["export", f"model={copy}", f"out={out}", "format=saved_model"]) == 0
    differ = int((np.asarray(read_back("saved_model", out, batch)) != expected).sum())
    with capsys.disabled():
        print(f"\nNEAR TIES (pairs 1e-6 apart): the SavedModel differs from Keras at {differ} of {expected.size} positions")
    assert differ > 0, "the SavedModel now gives Keras's code at every near tie: the export note's limit is no longer true"
    assert differ < expected.size // 4, "a near tie moves a few positions, not the grid"


def test_the_export_note_says_what_goes_in_what_comes_out_and_the_limit(run):
    note = predictor.EXPORT_RULES[TASK].note(record_of(run))
    for fragment in ("float32", f"(batch, {SIDE}, {SIDE}, 3)", "the trained size only", "(x - mean) / std", "int32 code ids",
                     f"(batch, {GRID}, {GRID})", f"{CODES} codes", "row-major", "within float32 rounding", "train package"):
        assert fragment in note, (fragment, note)
