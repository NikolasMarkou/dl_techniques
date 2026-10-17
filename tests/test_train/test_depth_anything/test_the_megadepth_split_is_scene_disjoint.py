"""The trainer's MegaDepth split is scene-disjoint and spans the dataset's scenes.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.6 (D-129).
Measured at 308a4dfbf on the real dataset (195 scenes, 19 to 3970 pairs
each): ``discover_megadepth_pairs(max_files=11000)``, the trainer's default
``max_train_files + max_val_files``, returned the first 11,000 pairs in scene
order, i.e. scenes ``0000``-``0004`` only, and the trainer split that list at
90%, so its validation pairs (scene ``0004``) came from a scene whose other
images were in training.

The fixture tree mimics that shape: scenes larger than ``11000 / 5`` pairs, so
the first-N rule covers exactly 5 scenes. The trainer's own data path runs
(``train_depth_anything`` with the dataset class recording its file lists and
the model build stopping the run), so the test reads what a run trains on.
"""

from pathlib import Path

import pytest

from train.common import megadepth
from train.depth_anything import train_depth_anything as module

N_SCENES = 12
PAIRS_PER_SCENE = 2400


class _Stop(Exception):
    pass


def _scene(path):
    """{root}/{scene}/dense{N}/{imgs,depths}/{file} -> scene."""
    return Path(path).parts[-4]


def _tree(root, n_scenes=N_SCENES, pairs=PAIRS_PER_SCENE):
    for s in range(n_scenes):
        for kind, suffix in (("imgs", ".jpg"), ("depths", ".h5")):
            directory = root / f"{s:04d}" / "dense0" / kind
            directory.mkdir(parents=True)
            for i in range(pairs):
                (directory / f"{i:05d}{suffix}").touch()
    return root


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return _tree(tmp_path_factory.mktemp("megadepth"))


def _trainer_lists(tree, **overrides):
    seen = {}

    def dataset(rgb_paths, depth_paths, **kwargs):
        seen["train" if kwargs["is_training"] else "val"] = list(rgb_paths)
        return [None]

    def stop(_config):
        raise _Stop()

    config = module.DepthAnythingTrainingConfig(
        megadepth_root=str(tree), experiment_name="split", **overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "MegaDepthDataset", dataset)
        mp.setattr(module, "create_model", stop)
        with pytest.raises(_Stop):
            module.train_depth_anything(config)
    return seen["train"], seen["val"]


def test_no_scene_is_in_both_splits(tree):
    train, val = _trainer_lists(tree)
    train_scenes = {_scene(p) for p in train}
    val_scenes = {_scene(p) for p in val}
    assert val_scenes and train_scenes
    assert not train_scenes & val_scenes, sorted(train_scenes & val_scenes)


def test_a_default_run_covers_more_than_five_scenes(tree):
    config = module.DepthAnythingTrainingConfig()
    train, val = _trainer_lists(tree)
    assert len(train) == config.max_train_files and len(val) == config.max_val_files
    train_scenes = {_scene(p) for p in train}
    assert len(train_scenes) > 5, sorted(train_scenes)
    assert len(train_scenes) + len({_scene(p) for p in val}) == N_SCENES


def test_a_capped_discovery_spreads_its_files_over_every_scene(tree):
    rgb, depth = megadepth.discover_megadepth_pairs(str(tree), max_files=1200)
    counts = {}
    for path in rgb:
        counts[_scene(path)] = counts.get(_scene(path), 0) + 1
    assert counts == {f"{s:04d}": 100 for s in range(N_SCENES)}
    assert [p.rsplit(".", 1)[0].replace("/imgs/", "/") for p in rgb] == [
        p.rsplit(".", 1)[0].replace("/depths/", "/") for p in depth]


def test_the_split_is_deterministic_and_takes_a_tenth_of_the_scenes(tree):
    first = megadepth.split_megadepth_pairs(str(tree), 0.9, 500, 50)
    assert first == megadepth.split_megadepth_pairs(str(tree), 0.9, 500, 50)
    val_scenes = {_scene(p) for p in first[2]}
    assert len(val_scenes) == round(N_SCENES * 0.1)
