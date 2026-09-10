"""Guards for ``train.doc_res.common`` -- the config, the tf.data pipeline and
the table-driven loss.

Everything here writes only under ``tmp_path``. The staged corpus on
``/media/arxwn/data0_4tb`` is READ by two opt-in tests that skip when it is
absent, so the suite passes on a machine with no data; nothing under
``results/`` or under the staging volume is created, modified or deleted.

The defect classes these exist to catch
---------------------------------------
1. A SECOND NORMALISATION. bfunet's pipeline carries an explicit invariant that
   ``/255.0`` happens exactly once; the DocRes pipeline reads three files per
   sample instead of two, so there are three places a well-meaning edit could
   add a second divide. ``test_normalisation_happens_exactly_once`` pins the
   VALUE (a 200-valued pixel must arrive as ``200/255``, not ``200/255**2``),
   which is what a source-text count cannot do.
2. A CROP THAT TEARS. The page, its prompt and its ground truth must be cut at
   the same offset. They are decoded into one 9-channel tensor precisely so
   that they cannot disagree, and ``test_the_crop_is_the_same_crop_for_all_
   three`` proves the alignment survives the pipeline rather than trusting the
   construction.
3. A TASK-STRING BRANCH. The loss, the supervised channel count and the
   ground-truth adapter must all come from the ``TASKS`` table. The tests
   fabricate a spec with ``dataclasses.replace`` and assert the behaviour
   follows it; an ``if task == "binarization"`` anywhere would make them pass
   for the shipped table and fail for the fabricated one, and
   ``test_no_task_string_is_compared_against_in_this_module`` catches it
   structurally as well.
4. A QUIET EMPTY CORPUS. ``require_task_data`` must name the REAL reason a task
   cannot be trained. The assertions are on the reason text, not on the mere
   fact that something raised.
5. DOUBLE WEIGHT DECAY. AdamW decays internally; an L2 ``kernel_regularizer``
   on top decays the same parameter twice and inflates the loss.
"""

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from dl_techniques.datasets.document_restoration.tasks import (
    LOSSES,
    LOSS_CATEGORICAL_CROSSENTROPY,
    LOSS_L1,
    TASKS,
    get_task,
    task_names,
)
from train.doc_res import common as C
from train.doc_res import prepare_doc_res_data as prep


# ---------------------------------------------------------------------------
# Fixtures -- a miniature staged corpus under tmp_path
# ---------------------------------------------------------------------------

_PAGE_H, _PAGE_W = 80, 96
_PATCH = 64


def _positional_page(height: int = _PAGE_H, width: int = _PAGE_W) -> np.ndarray:
    """An RGB page whose pixel VALUES encode their coordinates.

    Two crops of this array are equal only if they were taken at the same
    offset, which is what makes the alignment guard non-vacuous.
    """
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack(
        [
            (ys * 3 % 251).astype(np.uint8),
            (xs * 5 % 251).astype(np.uint8),
            ((ys + xs) * 7 % 251).astype(np.uint8),
        ],
        axis=-1,
    )


@pytest.fixture
def staged_root(tmp_path: Path) -> Path:
    """A tmp_path staging volume holding one page of a REAL manifest dataset.

    Uses ``dibco2009`` because ``require_task_data`` consults the shipped
    manifest; a made-up dataset name would never be looked at.
    """
    ds = prep.get_dataset("dibco2009")
    target = prep.dataset_dir(tmp_path, ds)
    (target / "Dataset").mkdir(parents=True)
    (target / "GT").mkdir(parents=True)

    page = _positional_page()
    Image.fromarray(page).save(target / "Dataset" / "P01.png")

    # Ground truth: dark ink on a light background, so the two-class target has
    # both classes present.
    gt = np.full((_PAGE_H, _PAGE_W, 3), 240, dtype=np.uint8)
    gt[:, : _PAGE_W // 2] = 10
    Image.fromarray(gt).save(target / "GT" / "P01_GT.png")

    sidecar = prep.sidecar_path(target, target / "Dataset" / "P01.png", ".png")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page).save(sidecar)
    return tmp_path


@pytest.fixture
def config(staged_root: Path) -> C.DocResTrainingConfig:
    """A small, fast config pointed at the tmp_path corpus."""
    return C.DocResTrainingConfig(
        task="binarization",
        dataset_root=str(staged_root),
        patch_size=_PATCH,
        batch_size=2,
        patches_per_image=2,
        patch_shuffle_buffer=4,
        dataset_shuffle_buffer=4,
        val_split=0.0,
    )


REAL_ROOT = Path(prep.DEFAULT_DATASET_ROOT)
_real_corpus_present = (
    REAL_ROOT / prep.STAGING_DIRNAME / "binarization"
).is_dir()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_a_patch_size_not_divisible_by_eight_is_rejected_at_config_time():
    """The model refuses a non-divisible input; catch it at startup."""
    with pytest.raises(ValueError, match="multiple of 8"):
        C.DocResTrainingConfig(patch_size=100)


def test_a_divisible_patch_size_is_accepted():
    """Anti-vacuity twin: the guard rejects 100 for its remainder, not always."""
    assert C.DocResTrainingConfig(patch_size=96).patch_size == 96


def test_the_config_rejects_an_unknown_task_listing_the_legal_ones():
    with pytest.raises(ValueError) as excinfo:
        C.DocResTrainingConfig(task="unwarping")
    for name in task_names():
        assert name in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"epochs": -1},
        {"steps_per_epoch": 0},
        {"learning_rate": 0.0},
        {"final_learning_rate": 1.0},  # greater than the peak
        {"weight_decay": -1e-4},
        {"val_split": 1.0},
        {"max_train_files": 0},
    ],
)
def test_the_config_rejects_an_out_of_range_knob(kwargs):
    with pytest.raises(ValueError):
        C.DocResTrainingConfig(**kwargs)


def test_every_flag_of_add_common_arguments_reaches_the_config():
    """Flag -> field wiring, driven through the REAL parser and builder.

    A flag that ``config_from_args`` forgets silently does nothing. Each row
    carries a value DIFFERENT from the default, so "forwarded" is
    distinguishable from "never touched".
    """
    import argparse

    parser = C.add_common_arguments(argparse.ArgumentParser())
    rows = [
        ("--task", "deshadowing", "task", "deshadowing"),
        ("--dataset-root", "/tmp/x", "dataset_root", "/tmp/x"),
        ("--model-variant", "restormer_base", "model_variant", "restormer_base"),
        ("--patch-size", "128", "patch_size", 128),
        ("--batch-size", "7", "batch_size", 7),
        ("--epochs", "9", "epochs", 9),
        ("--steps-per-epoch", "11", "steps_per_epoch", 11),
        ("--validation-steps", "13", "validation_steps", 13),
        ("--learning-rate", "0.003", "learning_rate", 0.003),
        ("--final-learning-rate", "0.0001", "final_learning_rate", 0.0001),
        ("--weight-decay", "0.017", "weight_decay", 0.017),
        ("--patches-per-image", "3", "patches_per_image", 3),
        ("--patch-shuffle-buffer", "19", "patch_shuffle_buffer", 19),
        ("--dataset-shuffle-buffer", "23", "dataset_shuffle_buffer", 23),
        ("--val-split", "0.25", "val_split", 0.25),
        ("--max-train-files", "29", "max_train_files", 29),
        ("--seed", "31", "seed", 31),
        ("--patience", "37", "patience", 37),
        ("--output-dir", "/tmp/out", "output_dir", "/tmp/out"),
    ]
    declared = {
        s
        for a in parser._actions
        for s in a.option_strings
        if s not in ("--help", "-h")
    }
    assert declared == {flag for flag, _v, _f, _e in rows}, (
        "every flag needs a row; the table went stale"
    )

    defaults = C.DocResTrainingConfig()
    for flag, value, field, expected in rows:
        cfg = C.config_from_args(parser.parse_args([flag, value]))
        assert getattr(cfg, field) == expected, f"{flag} did not reach {field}"
        assert getattr(defaults, field) != expected, (
            f"{flag}'s row value equals the default; the row cannot fail"
        )


# ---------------------------------------------------------------------------
# Pairing and collection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("P01", "p01"),
        ("P01_GT", "p01"),
        ("7_gt", "7"),
        ("H08_skelGT", "h08"),
        ("H06_estGT", "h06"),
        ("H09-GT", "h09"),
        ("Fontfre_Noisec_TE", "fontfre_clean_te"),
        ("Fontfre_Clean_TE", "fontfre_clean_te"),
    ],
)
def test_pairing_key_folds_every_staged_ground_truth_spelling(stem, expected):
    assert C.pairing_key(stem) == expected


def test_collect_task_triplets_pairs_page_prompt_and_ground_truth(config):
    triplets = C.collect_task_triplets(config)
    assert len(triplets) == 1
    triplet = triplets[0]
    assert triplet.image.name == "P01.png"
    assert triplet.prompt.parent.parts[-2] == prep.PROMPT_DIRNAME
    assert triplet.ground_truth.name == "P01_GT.png"


def test_a_page_whose_ground_truth_has_a_different_size_is_dropped(
        config, staged_root
):
    """The size check is what disambiguates NoisyOffice's doubleresolution copy.

    Replacing the GT with a half-size image must drop the page rather than pair
    it, and the resulting empty worklist must raise rather than train on air.
    """
    ds = prep.get_dataset("dibco2009")
    gt = prep.dataset_dir(staged_root, ds) / "GT" / "P01_GT.png"
    Image.fromarray(np.zeros((_PAGE_H // 2, _PAGE_W // 2, 3), np.uint8)).save(gt)
    with pytest.raises(ValueError, match="none could be paired"):
        C.collect_task_triplets(config)


def test_a_page_with_no_prompt_sidecar_is_dropped(config, staged_root):
    ds = prep.get_dataset("dibco2009")
    target = prep.dataset_dir(staged_root, ds)
    prep.sidecar_path(target, target / "Dataset" / "P01.png", ".png").unlink()
    with pytest.raises(ValueError, match="none could be paired"):
        C.collect_task_triplets(config)


def test_split_triplets_keeps_at_least_one_page_on_each_side():
    fake = [
        C.SampleTriplet(Path(f"{i}.png"), Path(f"{i}p.png"), Path(f"{i}g.png"))
        for i in range(3)
    ]
    cfg = C.DocResTrainingConfig(val_split=0.1)
    train, val = C.split_triplets(fake, cfg)
    assert len(val) == 1 and len(train) == 2
    assert not set(train) & set(val)


def test_split_triplets_holds_nothing_back_when_val_split_is_zero():
    fake = [
        C.SampleTriplet(Path(f"{i}.png"), Path(f"{i}p.png"), Path(f"{i}g.png"))
        for i in range(3)
    ]
    train, val = C.split_triplets(fake, C.DocResTrainingConfig(val_split=0.0))
    assert len(train) == 3 and val == []


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _first_batch(config, spec, triplets):
    ds = C.create_dataset(triplets, config, spec, is_training=True)
    for x, y in ds.take(1):
        return x.numpy(), y.numpy()
    raise AssertionError("the dataset yielded nothing")


def test_the_pipeline_yields_six_input_channels_and_the_task_gt_shape(config):
    """The model's contract: 3 RGB + 3 prompt in, the task's channels out."""
    spec = config.spec
    x, y = _first_batch(config, spec, C.collect_task_triplets(config))
    assert x.shape == (2, _PATCH, _PATCH, 6)
    assert x.shape[-1] == C.N_INPUT_CHANNELS
    assert y.shape == (2, _PATCH, _PATCH, spec.n_supervised_channels)
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_normalisation_happens_exactly_once(tmp_path):
    """A 200-valued pixel must arrive as 200/255, never 200/255**2.

    This is the guard a second ``/255.0`` anywhere in the decode path reds. It
    asserts the VALUE rather than counting divisions in the source, so it fires
    for a second normalisation written in any form.
    """
    page = np.full((16, 16, 3), 200, dtype=np.uint8)
    paths = []
    for name in ("page.png", "prompt.png", "gt.png"):
        path = tmp_path / name
        Image.fromarray(page).save(path)
        paths.append(tf.constant(str(path)))

    cfg = C.DocResTrainingConfig(patch_size=8)
    decoded = C.decode_triplet(*paths, cfg).numpy()

    expected = 200.0 / 255.0
    assert decoded.shape[-1] == C.N_STACKED_CHANNELS
    np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-6)
    # The failure mode being excluded, stated explicitly.
    assert not np.allclose(decoded, expected / 255.0)


def test_the_crop_is_the_same_crop_for_page_prompt_and_ground_truth(config):
    """Alignment, proven on data where an offset would show.

    The page encodes its own coordinates, and the prompt and GT are copies of
    it, so equality after the crop is possible only if all three were cut at
    the same offset.
    """
    triplets = C.collect_task_triplets(config)
    # A fabricated L1 spec so the target is the RAW ground truth rather than a
    # one-hot map -- the comparison needs the pixels, not the classes.
    ds = prep.get_dataset("dibco2009")
    target = prep.dataset_dir(Path(config.dataset_root), ds)
    Image.fromarray(_positional_page()).save(target / "GT" / "P01_GT.png")

    spec = replace(config.spec, loss=LOSS_L1, n_supervised_channels=3)
    x, y = _first_batch(config, spec, triplets)
    rgb, prompt = x[..., :3], x[..., 3:6]
    np.testing.assert_array_equal(rgb, prompt)
    np.testing.assert_array_equal(rgb, y)
    # Anti-vacuity: the crop is not a constant field, so equality means
    # something.
    assert rgb.std() > 0.0


def test_a_degenerate_all_constant_page_is_filtered_out():
    """A flat scan carries no supervision; the filter drops it.

    Asserted on the PREDICATE, not by iterating the pipeline: the dataset
    repeats forever, so a filtered-to-empty pipeline spins rather than failing,
    and a test written that way would hang instead of reporting.
    """
    flat = np.full((4, 4, C.N_STACKED_CHANNELS), 0.5, np.float32)
    assert not bool(C.is_informative_page(tf.constant(flat)))

    informative = flat.copy()
    informative[0, 0, 0] = 0.6
    assert bool(C.is_informative_page(tf.constant(informative)))

    # A constant PAGE is degenerate even when the ground truth varies -- the
    # filter reads the RGB channels only.
    gt_only = flat.copy()
    gt_only[..., C.N_INPUT_CHANNELS:] = np.linspace(
        0.0, 1.0, 4 * 4 * 3
    ).reshape(4, 4, 3)
    assert not bool(C.is_informative_page(tf.constant(gt_only)))


def test_the_pipeline_actually_installs_the_degenerate_page_filter(config):
    """The predicate above is wired in, not merely defined.

    A flat page paired with a valid target must not reach the model; asserted
    by counting the filter op in the pipeline's own graph description rather
    than by draining an infinite dataset.
    """
    triplets = C.collect_task_triplets(config)
    dataset = C.create_dataset(triplets, config, config.spec, is_training=True)
    lineage = []
    node = dataset
    while node is not None:
        lineage.append(type(node).__name__)
        inputs = getattr(node, "_input_dataset", None)
        node = inputs
    assert any("Filter" in name for name in lineage), lineage


def test_a_page_prompt_size_mismatch_raises_instead_of_resizing(tmp_path):
    """Silently resizing a mismatched triplet would hide a pairing defect."""
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "b.png")
    with pytest.raises(ValueError, match="must share one size"):
        C._decode_triplet_numpy(
            str(tmp_path / "a.png").encode(),
            str(tmp_path / "b.png").encode(),
            str(tmp_path / "a.png").encode(),
        )


# ---------------------------------------------------------------------------
# Table-driven loss and ground truth
# ---------------------------------------------------------------------------


def test_every_task_in_the_table_has_a_loss_and_a_ground_truth_adapter():
    """The dispatch dicts cover the table, and nothing beyond it."""
    assert set(C._BASE_LOSS_BUILDERS) == set(LOSSES)
    assert set(C._GROUND_TRUTH_ADAPTERS) == set(LOSSES)
    for name in task_names():
        assert C.build_task_loss(get_task(name)) is not None


@pytest.mark.parametrize("name", list(task_names()))
def test_the_loss_slice_width_comes_from_the_table(name):
    spec = get_task(name)
    loss = C.build_task_loss(spec)
    assert loss.n_channels == spec.n_supervised_channels


def test_changing_the_table_changes_the_loss_and_the_target():
    """The anti-branching guard.

    A fabricated spec -- same task NAME, different table fields -- must change
    the loss object, the slice width and the ground-truth adapter. Any
    ``if task == "binarization"`` would keep the shipped behaviour and red this.
    """
    shipped = get_task("binarization")
    assert shipped.loss == LOSS_CATEGORICAL_CROSSENTROPY
    shipped_loss = C.build_task_loss(shipped)
    assert isinstance(shipped_loss.base_loss, keras.losses.CategoricalCrossentropy)
    assert shipped_loss.n_channels == 2

    fabricated = replace(shipped, loss=LOSS_L1, n_supervised_channels=3)
    other_loss = C.build_task_loss(fabricated)
    assert isinstance(other_loss.base_loss, keras.losses.MeanAbsoluteError)
    assert other_loss.n_channels == 3

    patch = tf.constant(
        np.random.default_rng(0).random((4, 4, C.N_STACKED_CHANNELS)),
        dtype=tf.float32,
    )
    _x_a, y_a = C.split_stacked_sample(patch, shipped)
    _x_b, y_b = C.split_stacked_sample(patch, fabricated)
    assert y_a.shape[-1] == 2 and y_b.shape[-1] == 3
    # The shipped spec produces a ONE-HOT map; the fabricated one the raw GT.
    assert set(np.unique(y_a.numpy())) <= {0.0, 1.0}
    assert not set(np.unique(y_b.numpy())) <= {0.0, 1.0}


def test_the_two_class_target_puts_ink_in_the_declared_channel():
    gt = np.zeros((1, 2, C.N_STACKED_CHANNELS), np.float32)
    dark = 10.0 / C.PIXEL_SCALE
    light = 240.0 / C.PIXEL_SCALE
    gt[0, 0, C.N_INPUT_CHANNELS:] = dark
    gt[0, 1, C.N_INPUT_CHANNELS:] = light

    _x, y = C.split_stacked_sample(tf.constant(gt), get_task("binarization"))
    y = y.numpy()
    # Pinned ABSOLUTELY, not through the constant: reading
    # `BINARY_INK_CLASS_INDEX` here would make the expectation move with the
    # code and the guard could never fail. The inference shim's argmax reads
    # the constant, so the two must agree -- asserted on the next line.
    assert C.BINARY_INK_CLASS_INDEX == 1
    assert y[0, 0, 1] == 1.0 and y[0, 0, 0] == 0.0, "dark pixel must be ink"
    assert y[0, 1, 1] == 0.0 and y[0, 1, 0] == 1.0, "light pixel must be background"
    np.testing.assert_allclose(y.sum(axis=-1), 1.0)


def test_a_cross_entropy_task_that_does_not_supervise_two_channels_raises():
    bad = replace(
        get_task("binarization"),
        loss=LOSS_CATEGORICAL_CROSSENTROPY,
        n_supervised_channels=3,
    )
    with pytest.raises(ValueError, match="exactly 2"):
        C.split_stacked_sample(
            tf.zeros((1, 1, C.N_STACKED_CHANNELS)), bad
        )


def test_the_loss_slices_the_prediction_and_round_trips():
    """The head is 3 channels for every task; the loss sees only its slice."""
    loss = C.build_task_loss(get_task("binarization"))
    y_true = tf.constant([[[[1.0, 0.0]]]])
    a = tf.constant([[[[10.0, -10.0, 0.0]]]])
    b = tf.constant([[[[10.0, -10.0, 999.0]]]])
    assert float(loss(y_true, a)) == float(loss(y_true, b))

    restored = C.SupervisedSliceLoss.from_config(loss.get_config())
    assert restored.n_channels == loss.n_channels
    assert float(restored(y_true, a)) == pytest.approx(float(loss(y_true, a)))


def _task_string_branch_offenders(source_path: Path) -> list:
    """Every task-string COMPARISON or dict KEY in one module.

    Interface contract: the shared instrument behind the repo-wide invariant in
    ``tasks.py``'s module docstring. Takes a path to a Python source file;
    returns a list of human-readable ``"<kind> at line N"`` strings, empty when
    the module is clean. It deliberately does NOT flag a task name used as a
    default, a tuple element or a data value -- ``infer_doc_res.END2END_STAGES``
    is a legitimate ordered tuple of stage names and is not a branch. Never
    raises; a syntactically invalid module would raise from ``ast.parse``.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = set(task_names())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for operand in operands:
                for sub in ast.walk(operand):
                    if isinstance(sub, ast.Constant) and sub.value in names:
                        offenders.append(f"comparison at line {node.lineno}")
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value in names:
                    offenders.append(f"dict key at line {node.lineno}")
    return offenders


def test_no_task_string_is_compared_against_in_this_module():
    """Structural twin of the behavioural guard above.

    A task name may appear as a DEFAULT (``task: str = "binarization"``); it may
    never appear in a comparison or as a dict key, which are the two shapes a
    per-task branch takes.
    """
    offenders = _task_string_branch_offenders(Path(inspect.getfile(C)))
    assert not offenders, (
        "task-string branching inside train/doc_res/common.py: "
        + ", ".join(offenders)
    )


def test_no_task_string_is_compared_against_in_any_shipped_train_doc_res_module():
    """The same rule, over EVERY module in ``src/train/doc_res/``.

    The per-module guards (this file's, and ``test_inference.py``'s text
    backstop) each watched one file, so ``prepare_doc_res_data.py`` and
    ``train_doc_res.py`` -- the two entry points a user actually runs -- were
    unguarded while ``tasks.py`` asserted the rule tree-wide. This closes that
    gap by discovering the modules from the package directory, so a NEW module
    is covered the day it is added rather than the day someone remembers.
    """
    package_dir = Path(inspect.getfile(C)).parent
    modules = sorted(package_dir.glob("*.py"))
    assert len(modules) >= 5, [m.name for m in modules]

    offenders = {
        module.name: found
        for module in modules
        if (found := _task_string_branch_offenders(module))
    }
    assert not offenders, (
        "task-string branching inside src/train/doc_res/: "
        f"{offenders}. Task specificity belongs in the TASKS table; read a "
        "field off the spec instead of comparing a name."
    )


# ---------------------------------------------------------------------------
# The float32 prompt sidecar the pipeline cannot read (D-034)
# ---------------------------------------------------------------------------


def _float32_prompt_spec(name: str = "binarization"):
    """A spec whose prompt sidecars are ``.npy`` arrays, as dewarping's are."""
    return replace(get_task(name), prompt_dtype="float32")


def test_the_shipped_dewarping_task_is_the_reason_this_guard_exists():
    """Anchors the guards below to a real row of the table, not a hypothetical.

    If this ever fails because dewarping became a uint8 task, the two guards
    below stop testing the shipped tree and must be re-aimed.
    """
    assert get_task("dewarping").prompt_dtype == "float32"
    assert C.PROMPT_SIDECAR_SUFFIXES["float32"] == ".npy"


@pytest.mark.parametrize("task", sorted(task_names()))
def test_every_task_declares_a_dtype_with_a_sidecar_suffix(task):
    """The writer's suffix and the reader's suffix come from ONE mapping."""
    assert get_task(task).prompt_dtype in C.PROMPT_SIDECAR_SUFFIXES


def test_a_dtype_with_no_sidecar_suffix_is_rejected_by_the_table_validator():
    """The enforcement behind the pin above, exercised directly.

    A dtype with no entry in the mapping has no on-disk representation, so the
    staging script could not name the file it writes. ``_validate`` refuses it
    at import time, which is why the pin above can only ever fail as a
    collection error on the shipped table.
    """
    import dl_techniques.datasets.document_restoration.tasks as tasks_module

    bad = replace(get_task("binarization"), prompt_dtype="float64")
    with pytest.raises(ValueError, match="prompt_dtype"):
        tasks_module._validate((bad,))


def test_collecting_triplets_for_a_float32_prompt_task_refuses_by_name(tmp_path):
    """A float32 prompt is refused at WORKLIST time, not mid-epoch.

    The decoder opens all three paths with ``PIL.Image.open``, which raises
    ``UnidentifiedImageError`` on an ``.npy`` file from inside a
    ``tf.numpy_function`` -- i.e. after ``fit()`` has started. This turns that
    into a named refusal at startup that says what is missing.
    """
    with pytest.raises(C.UnsupportedPromptDtypeError) as excinfo:
        C.collect_dataset_triplets(tmp_path, _float32_prompt_spec())

    message = str(excinfo.value)
    assert ".npy" in message, message
    for needed in ("mask", "decoder arm", "normalisation"):
        assert needed in message, message


def test_building_a_dataset_for_a_float32_prompt_task_refuses_by_name(config):
    """The second entry point: a hand-built worklist must be refused too."""
    triplets = C.collect_task_triplets(config)
    assert triplets
    with pytest.raises(C.UnsupportedPromptDtypeError):
        C.create_dataset(
            triplets, config, _float32_prompt_spec(), is_training=True
        )


def test_a_uint8_prompt_task_is_not_refused(config):
    """The guard discriminates: every shipped-trainable task still builds."""
    C.require_uint8_prompt_sidecars(get_task("binarization"))
    triplets = C.collect_task_triplets(config)
    assert C.create_dataset(triplets, config, config.spec, is_training=True)


# ---------------------------------------------------------------------------
# require_task_data -- the right reason, per task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task,needles",
    [
        ("dewarping", ["549", "registration", "doc3d"]),
        ("deblurring", ["dead", "tdd"]),
        ("deshadowing", ["drive", "rdd"]),
    ],
)
def test_require_task_data_names_the_real_reason(tmp_path, task, needles):
    """Not "file not found" -- the gate, the dead host, the Drive folder."""
    with pytest.raises(prep.MissingTrainingDataError) as excinfo:
        C.require_task_data(tmp_path, task)
    message = str(excinfo.value).lower()
    for needle in needles:
        assert needle in message, f"{task}: missing {needle!r} in:\n{message}"


def test_require_task_data_says_dir300_is_eval_only():
    """An eval corpus must never look like a training corpus."""
    with pytest.raises(prep.MissingTrainingDataError) as excinfo:
        C.require_task_data(Path("/nonexistent-doc-res-root"), "dewarping")
    assert "eval only" in str(excinfo.value).lower()


def test_require_task_data_returns_the_staged_datasets(staged_root):
    staged = C.require_task_data(staged_root, "binarization")
    assert [d.name for d in staged] == ["dibco2009"]


def test_collect_task_triplets_raises_before_building_an_empty_dataset(tmp_path):
    cfg = C.DocResTrainingConfig(task="deblurring", dataset_root=str(tmp_path))
    with pytest.raises(prep.MissingTrainingDataError):
        C.collect_task_triplets(cfg)


# ---------------------------------------------------------------------------
# Optimizer / model
# ---------------------------------------------------------------------------


def test_weight_decay_is_applied_once_by_the_optimizer_only():
    """AdamW decays internally; an L2 regularizer would decay it again."""
    cfg = C.DocResTrainingConfig(weight_decay=5e-4)
    optimizer = C.build_optimizer(cfg)
    assert isinstance(optimizer, keras.optimizers.AdamW)
    assert float(optimizer.weight_decay) == pytest.approx(5e-4)

    model = C.build_model(cfg)
    offenders = []
    for layer in model._flatten_layers(include_self=False):
        for attribute in (
                "kernel_regularizer", "bias_regularizer", "activity_regularizer"
        ):
            if getattr(layer, attribute, None) is not None:
                offenders.append(f"{layer.name}.{attribute}")
    assert not offenders, (
        "a regularizer plus AdamW's decoupled decay penalises the same weight "
        "twice: " + ", ".join(offenders)
    )


def test_the_schedule_anneals_from_the_peak_to_the_declared_floor():
    cfg = C.DocResTrainingConfig(
        learning_rate=2e-4, final_learning_rate=1e-6, epochs=2, steps_per_epoch=5
    )
    optimizer = C.build_optimizer(cfg)
    # `optimizer.learning_rate` is the CURRENT value; the schedule object
    # itself lives on the private attribute.
    schedule = optimizer._learning_rate
    assert float(schedule(0)) == pytest.approx(2e-4, rel=1e-6)
    assert float(schedule(10)) == pytest.approx(1e-6, rel=1e-3)


def test_the_compiled_model_takes_six_channels_and_emits_three():
    cfg = C.DocResTrainingConfig(patch_size=32)
    model = C.build_model(cfg)
    out = model(tf.zeros((1, 32, 32, C.N_INPUT_CHANNELS)), training=False)
    assert tuple(out.shape) == (1, 32, 32, 3)


def test_train_uses_stock_fit_with_no_custom_train_step():
    """The invariant, asserted rather than assumed."""
    model = C.build_model(C.DocResTrainingConfig())
    assert type(model).train_step is keras.Model.train_step
    assert "def train_step" not in Path(inspect.getfile(C)).read_text()


# ---------------------------------------------------------------------------
# The real staged corpus (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_corpus_present, reason="no staged doc_res corpus on this machine"
)
def test_the_real_binarization_corpus_pairs_and_yields_six_channels():
    cfg = C.DocResTrainingConfig(
        patch_size=64, batch_size=2, patches_per_image=2, max_train_files=8
    )
    triplets = C.collect_task_triplets(cfg)
    assert len(triplets) == 8
    for triplet in triplets:
        assert triplet.image.is_file()
        assert triplet.prompt.is_file()
        assert triplet.ground_truth.is_file()
    x, y = _first_batch(cfg, cfg.spec, triplets)
    assert x.shape == (2, 64, 64, 6)
    assert y.shape == (2, 64, 64, 2)
    np.testing.assert_allclose(y.sum(axis=-1), 1.0)


@pytest.mark.skipif(
    not _real_corpus_present, reason="no staged doc_res corpus on this machine"
)
def test_the_real_corpus_never_pairs_a_page_with_a_different_sized_target():
    """NoisyOffice ships a doubleresolution copy under the same stems."""
    cfg = C.DocResTrainingConfig(max_train_files=60)
    for triplet in C.collect_task_triplets(cfg):
        assert C._image_size(triplet.image) == C._image_size(triplet.ground_truth)
        assert C._image_size(triplet.image) == C._image_size(triplet.prompt)
