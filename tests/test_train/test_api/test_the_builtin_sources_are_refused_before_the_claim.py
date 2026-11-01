"""A missing ImageNet tree or Imagenette record set exits 2 before the run is claimed.

Measured before the change (``findings/imagenet-imagenette.md``): missing Imagenette records
was a ``FileNotFoundError`` traceback and exit 1, a missing devkit the same, a missing ImageNet
root a ``ConfigError`` that came AFTER ``open_run`` ("run directory ... was claimed"). The
check is ``train.api.data_args.check_builtin_sources``, called by ``builtin_facts`` and so by
``describe_data``, and it is ``Path`` tests only.

Four things are pinned.

* Each missing piece is a ``ConfigError`` that names the key and the path looked at; the
  neighbouring complete tree is accepted (a refusal that hits everything is not a check), the
  devkit may sit in ``devkit/data`` as the loader reads it, and an explicit ``devkit_dir`` is
  the one looked at.
* In a child process through the real command line: exit 2, the message on stderr, ``out``
  absent, no TFDS builder or loader used and TensorFlow's devices not created (D-055, the same
  recorder as ``test_no_check_before_setup_gpu_initialises_tensorflow``). In another child, the
  check itself imports neither ``tensorflow_datasets`` nor ``train.common`` (which imports it for
  the whole process once the command line has resolved a recipe, so the first child cannot say
  "not imported").
* The REAL layouts on this machine pass, read only: a false refusal of the real dataset is the
  worst failure of this step. Skipped (with the reason) on a machine that has neither.
* The in-memory datasets are not looked up.

Everything is written under ``tmp_path``; the real trees are only listed.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from train.api import data_args
from train.api.data_args import BuiltinImageData, check_builtin_sources
from train.api.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_IMAGENET = Path("/media/arxwn/data0_4tb/datasets/imagenet2012")
REAL_TFDS = Path("/media/arxwn/data0_4tb/datasets/tensorflow_datasets")


def imagenet_tree(root: Path, *, train=True, val=True, devkit="devkit", meta=True, truth=True) -> Path:
    """Lay out the files the loader looks for (empty ones: only their presence is read)."""
    root.mkdir(parents=True)
    if train:
        (root / "train" / "n01440764").mkdir(parents=True)
    if val:
        (root / "val").mkdir()
    if devkit:
        place = root / devkit
        place.mkdir(parents=True)
        if meta:
            (place / "meta.mat").write_bytes(b"")
        if truth:
            (place / "ILSVRC2012_validation_ground_truth.txt").write_text("")
    return root


def tfds_records(root: Path, *, version="1.0.0", info=True) -> Path:
    """Lay out ``<root>/imagenette/320px-v2/<version>/dataset_info.json``."""
    place = root / "imagenette" / "320px-v2" / version
    place.mkdir(parents=True)
    if info:
        (place / "dataset_info.json").write_text("{}")
    return root


def imagenet(root, **more):
    return BuiltinImageData(dataset="imagenet", imagenet_root=str(root), **more)


# ---------------------------------------------------------------------
# ImageNet
# ---------------------------------------------------------------------


def test_a_complete_tree_is_accepted_in_both_devkit_places_and_with_an_explicit_devkit(tmp_path):
    check_builtin_sources(imagenet(imagenet_tree(tmp_path / "a")))
    check_builtin_sources(imagenet(imagenet_tree(tmp_path / "b", devkit="devkit/data")))
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "meta.mat").write_bytes(b"")
    (other / "ILSVRC2012_validation_ground_truth.txt").write_text("")
    check_builtin_sources(imagenet(imagenet_tree(tmp_path / "c", devkit=None), devkit_dir=str(other)))


def test_a_missing_root_is_refused_with_the_key_and_the_path(tmp_path):
    missing = tmp_path / "nowhere"
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(imagenet(missing))
    assert "'data.imagenet_root'" in str(caught.value) and str(missing) in str(caught.value)
    # The ROOT's own message, not the train/ one that would also name the path (the survivor
    # of the first mutant run: with the root test dropped this test still passed).
    assert f"the ImageNet root {str(missing)!r} is not a directory; point it at" in str(caught.value)


def test_a_root_that_is_a_file_is_refused(tmp_path):
    file = tmp_path / "root.txt"
    file.write_text("x")
    with pytest.raises(ConfigError, match="'data.imagenet_root'.*root .* is not a directory; point it at"):
        check_builtin_sources(imagenet(file))


def test_a_null_root_means_the_loaders_default_and_says_so(tmp_path, monkeypatch):
    from dl_techniques.datasets.vision import imagenet2012

    missing = tmp_path / "default_root"
    monkeypatch.setattr(imagenet2012, "IMAGENET2012_DEFAULT_ROOT", str(missing))
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(BuiltinImageData(dataset="imagenet"))
    assert str(missing) in str(caught.value) and "null: the loader's default" in str(caught.value)


@pytest.mark.parametrize("split", ["train", "val"])
def test_a_root_without_a_split_folder_is_refused_naming_it(tmp_path, split):
    root = imagenet_tree(tmp_path / "r", **{split: False})
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(imagenet(root))
    text = str(caught.value)
    assert "'data.imagenet_root'" in text and str(root / split) in text


@pytest.mark.parametrize(
    "layout, missing",
    [
        ({"devkit": None}, "meta.mat, ILSVRC2012_validation_ground_truth.txt"),
        ({"meta": False}, "meta.mat"),
        ({"truth": False}, "ILSVRC2012_validation_ground_truth.txt"),
    ],
)
def test_a_missing_devkit_file_is_refused_naming_the_file_and_both_places(tmp_path, layout, missing):
    root = imagenet_tree(tmp_path / "r", **layout)
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(imagenet(root))
    text = str(caught.value)
    assert "'data.devkit_dir'" in text and missing in text
    assert str(root / "devkit") in text and str(root / "devkit" / "data") in text
    assert "null: <imagenet_root>/devkit" in text


def test_an_explicit_devkit_dir_is_the_one_looked_at(tmp_path):
    root = imagenet_tree(tmp_path / "r")  # its own devkit is complete
    bad = tmp_path / "empty_devkit"
    bad.mkdir()
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(imagenet(root, devkit_dir=str(bad)))
    assert str(bad) in str(caught.value) and "null:" not in str(caught.value)


# ---------------------------------------------------------------------
# Imagenette
# ---------------------------------------------------------------------


def test_records_under_tfds_data_dir_are_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tfds_records(tmp_path / "tfds")))
    check_builtin_sources(BuiltinImageData(dataset="imagenette"))


def test_the_default_directory_is_read_when_the_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("TFDS_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(BuiltinImageData(dataset="imagenette"))
    assert str(tmp_path / "tensorflow_datasets" / "imagenette" / "320px-v2") in str(caught.value)
    assert "default ~/tensorflow_datasets" in str(caught.value)
    tfds_records(tmp_path / "tensorflow_datasets")
    check_builtin_sources(BuiltinImageData(dataset="imagenette"))


@pytest.mark.parametrize(
    "build",
    [
        lambda root: root.mkdir(),  # nothing under it
        lambda root: tfds_records(root, info=False),  # a version folder with no dataset_info.json
        lambda root: (root / "imagenette" / "full-size-v2" / "1.0.0").mkdir(parents=True),  # the other config
    ],
    ids=["empty", "no_dataset_info", "other_config_only"],
)
def test_records_that_are_not_the_320px_set_are_refused_with_the_key_and_the_path(
    tmp_path, monkeypatch, build
):
    root = tmp_path / "tfds"
    build(root)
    monkeypatch.setenv("TFDS_DATA_DIR", str(root))
    with pytest.raises(ConfigError) as caught:
        check_builtin_sources(BuiltinImageData(dataset="imagenette"))
    text = str(caught.value)
    assert "'data.dataset'" in text and str(root / "imagenette" / "320px-v2") in text
    assert "$TFDS_DATA_DIR" in text and "nothing is downloaded" in text


@pytest.mark.parametrize("dataset", ["mnist", "cifar10", "cifar100"])
def test_an_in_memory_dataset_is_not_looked_up(dataset, monkeypatch, tmp_path):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "empty"))
    check_builtin_sources(BuiltinImageData(dataset=dataset))


def test_builtin_facts_makes_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(ConfigError, match="imagenette"):
        data_args.builtin_facts(BuiltinImageData(dataset="imagenette"), None)
    with pytest.raises(ConfigError, match="'data.imagenet_root'"):
        data_args.builtin_facts(imagenet(tmp_path / "nowhere"), None)


# ---------------------------------------------------------------------
# The real layouts, read only
# ---------------------------------------------------------------------


@pytest.mark.skipif(not REAL_IMAGENET.is_dir(), reason=f"no ImageNet tree at {REAL_IMAGENET}")
def test_the_real_imagenet_tree_passes_by_default_and_with_explicit_paths():
    check_builtin_sources(BuiltinImageData(dataset="imagenet"))
    check_builtin_sources(imagenet(REAL_IMAGENET))
    check_builtin_sources(imagenet(REAL_IMAGENET, devkit_dir=str(REAL_IMAGENET / "devkit")))


@pytest.mark.skipif(not REAL_TFDS.is_dir(), reason=f"no TFDS directory at {REAL_TFDS}")
def test_the_real_imagenette_records_pass_with_tfds_data_dir_set(monkeypatch):
    monkeypatch.setenv("TFDS_DATA_DIR", str(REAL_TFDS))
    check_builtin_sources(BuiltinImageData(dataset="imagenette"))


# ---------------------------------------------------------------------
# Through the command line, in a child
# ---------------------------------------------------------------------

#: ``argv``: the arguments of the command; prints what the child saw as one JSON line.
DRIVER = textwrap.dedent(
    f"""
    import json
    import sys

    sys.path.insert(0, {str(REPO_ROOT)!r})
    from tensorflow.python.eager import context as eager

    import tensorflow_datasets as tfds
    import train.common.gpu as gpu
    from train.api import cli

    # ``train.common`` imports tensorflow_datasets for the whole process, so "not imported" cannot
    # be asserted here (the unit child below asserts it for the check itself); what can be is that
    # no TFDS builder or loader is USED.
    used = []


    def tripwire(*args, **kwargs):
        used.append("tfds")
        raise RuntimeError("TFDS was used")


    tfds.builder = tfds.load = tripwire

    class Stop(Exception):
        pass


    reached = []


    def recorder(*args, **kwargs):
        reached.append("setup_gpu")
        raise Stop


    gpu.setup_gpu = recorder
    try:
        code = cli.main(sys.argv[1:])
    except Stop:
        code = "stopped at setup_gpu"
    print(json.dumps({{
        "code": code,
        "setup_gpu_reached": bool(reached),
        "tfds_used": bool(used),
        "devices_created": eager.context()._context_handle is not None,
    }}))
    """
)


def command(tmp_path: Path, *keys: str, tfds: str = None):
    """Run ``image_classification`` on vit_pico with ``keys`` in a child; return (seen, stderr)."""
    env = {**os.environ, "MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1"}
    if tfds is not None:
        env["TFDS_DATA_DIR"] = tfds
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, "image_classification", "model=vit/vit_pico",
         f"out={out}", "data.dataset=" + keys[0], *keys[1:]],
        capture_output=True, text=True, timeout=600, cwd=tmp_path, env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr, out


UNIT_DRIVER = textwrap.dedent(
    f"""
    import json
    import os
    import sys

    sys.path.insert(0, {str(REPO_ROOT)!r})
    from tensorflow.python.eager import context as eager

    import train.api.data_args as data_args
    from train.api.errors import ConfigError

    before = "tensorflow_datasets" in sys.modules
    refused = []
    for data in (
        data_args.BuiltinImageData(dataset="imagenet", imagenet_root=os.path.join(sys.argv[1], "nowhere")),
        data_args.BuiltinImageData(dataset="imagenette"),
    ):
        try:
            data_args.builtin_facts(data, None)
        except ConfigError as exc:
            refused.append(str(exc))
    print(json.dumps({{
        "refused": len(refused),
        "tfds_before": before,
        "tfds_after": "tensorflow_datasets" in sys.modules,
        "train_common_after": "train.common" in sys.modules,
        "devices_created": eager.context()._context_handle is not None,
    }}))
    """
)


def test_the_check_itself_imports_no_tfds_and_creates_no_device(tmp_path):
    empty = tmp_path / "tfds"
    empty.mkdir()
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1", "TFDS_DATA_DIR": str(empty)}
    done = subprocess.run(
        [sys.executable, "-c", UNIT_DRIVER, str(tmp_path)],
        capture_output=True, text=True, timeout=600, cwd=tmp_path, env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen == {
        "refused": 2, "tfds_before": False, "tfds_after": False,
        "train_common_after": False, "devices_created": False,
    }


@pytest.mark.parametrize(
    "case, build, key",
    [
        ("root_missing", lambda tmp: ("imagenet", f"data.imagenet_root={tmp / 'nowhere'}"), "data.imagenet_root"),
        ("no_train", lambda tmp: ("imagenet", f"data.imagenet_root={imagenet_tree(tmp / 'r', train=False)}"),
         "data.imagenet_root"),
        ("no_devkit", lambda tmp: ("imagenet", f"data.imagenet_root={imagenet_tree(tmp / 'r', devkit=None)}"),
         "data.devkit_dir"),
    ],
)
def test_an_imagenet_mistake_exits_2_before_the_claim_and_before_any_device(tmp_path, case, build, key):
    dataset, root_key = build(tmp_path)
    seen, stderr, out = command(tmp_path, dataset, root_key)
    assert seen["code"] == 2, (seen, stderr[-2000:])
    assert "error:" in stderr and f"'{key}'" in stderr
    assert str(tmp_path) in stderr, "the message names the path looked at"
    assert not out.exists(), "a refusal before the claim leaves no run directory"
    assert seen["setup_gpu_reached"] is False
    assert seen["tfds_used"] is False and seen["devices_created"] is False


def test_missing_imagenette_records_exit_2_before_the_claim_and_before_any_device(tmp_path):
    empty = tmp_path / "tfds"
    empty.mkdir()
    seen, stderr, out = command(tmp_path, "imagenette", tfds=str(empty))
    assert seen["code"] == 2, (seen, stderr[-2000:])
    assert "'data.dataset'" in stderr and str(empty / "imagenette" / "320px-v2") in stderr
    assert not out.exists()
    assert seen["setup_gpu_reached"] is False
    assert seen["tfds_used"] is False and seen["devices_created"] is False


@pytest.mark.skipif(not REAL_TFDS.is_dir(), reason=f"no TFDS directory at {REAL_TFDS}")
def test_the_control_real_records_pass_the_check_and_the_run_is_claimed_up_to_setup_gpu(tmp_path):
    """Reaching ``setup_gpu`` proves the check let a real request by, with no device created."""
    seen, stderr, out = command(tmp_path, "imagenette", tfds=str(REAL_TFDS))
    assert seen["code"] == "stopped at setup_gpu", (seen, stderr[-2000:])
    assert out.exists() and seen["setup_gpu_reached"] is True
    assert seen["tfds_used"] is False and seen["devices_created"] is False
