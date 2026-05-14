"""Unit tests for ``train.logic.clevr_hans_data`` (E5 / plan_2026-05-14_c95e848c)."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from train.logic.clevr_hans_data import (
    COLOR_OFFSET,
    FEATURE_WIDTH,
    MATERIAL_OFFSET,
    PRESENCE_OFFSET,
    SHAPE_OFFSET,
    SIZE_OFFSET,
    encode_scene_graph,
    encode_scenes,
    find_splits,
    infer_num_classes,
    load_scenes_json,
)


# ---------------------------------------------------------------------
# Synthetic scenes
# ---------------------------------------------------------------------


def _mk_scene(objs, class_id=0, fname="x.png"):
    return {"image_filename": fname, "class_id": class_id, "objects": objs}


def _obj(size="small", color="red", material="rubber", shape="cube"):
    return {"size": size, "color": color, "material": material, "shape": shape}


# ---------------------------------------------------------------------
# encode_scene_graph
# ---------------------------------------------------------------------


class TestEncodeSceneGraph:
    def test_shape_and_dtype(self):
        scene = _mk_scene([_obj(), _obj("large", "blue", "metal", "sphere")])
        arr = encode_scene_graph(scene, max_objects=10)
        assert arr.shape == (10, FEATURE_WIDTH)
        assert arr.dtype == np.float32

    def test_onehot_positions(self):
        scene = _mk_scene([_obj(size="large", color="blue", material="metal", shape="sphere")])
        arr = encode_scene_graph(scene, max_objects=4)
        # size=large -> SIZES[1] -> offset 0 + 1 = 1
        assert arr[0, SIZE_OFFSET + 1] == 1.0
        # color=blue -> COLORS[2]
        assert arr[0, COLOR_OFFSET + 2] == 1.0
        # material=metal -> MATERIALS[1]
        assert arr[0, MATERIAL_OFFSET + 1] == 1.0
        # shape=sphere -> SHAPES[1]
        assert arr[0, SHAPE_OFFSET + 1] == 1.0
        # presence bit
        assert arr[0, PRESENCE_OFFSET] == 1.0
        # Empty slots are zero.
        assert np.all(arr[1:] == 0.0)

    def test_zero_padding(self):
        scene = _mk_scene([_obj(), _obj()])
        arr = encode_scene_graph(scene, max_objects=5)
        assert arr.shape == (5, FEATURE_WIDTH)
        # rows 0 and 1 have data, 2-4 are zero
        assert np.any(arr[0] > 0) and np.any(arr[1] > 0)
        assert np.all(arr[2:] == 0.0)

    def test_truncation_past_max(self):
        objs = [_obj() for _ in range(12)]
        scene = _mk_scene(objs)
        arr = encode_scene_graph(scene, max_objects=10)
        assert arr.shape == (10, FEATURE_WIDTH)
        # All 10 slots should be populated (presence==1 on every row).
        assert np.all(arr[:, PRESENCE_OFFSET] == 1.0)

    def test_unknown_attribute_silent_zero(self):
        scene = _mk_scene([_obj(color="chartreuse")])
        arr = encode_scene_graph(scene, max_objects=2)
        # color block (offsets 3..10) should be all zero for that row.
        assert np.all(arr[0, COLOR_OFFSET:COLOR_OFFSET + 8] == 0.0)
        # But the other attributes still encoded.
        assert arr[0, SIZE_OFFSET + 0] == 1.0  # small
        assert arr[0, PRESENCE_OFFSET] == 1.0


# ---------------------------------------------------------------------
# encode_scenes
# ---------------------------------------------------------------------


class TestEncodeScenes:
    def test_batch_encoding(self):
        scenes = [
            _mk_scene([_obj()], class_id=0),
            _mk_scene([_obj(), _obj("large")], class_id=2),
        ]
        feats, labels = encode_scenes(scenes, max_objects=4)
        assert feats.shape == (2, 4, FEATURE_WIDTH)
        assert feats.dtype == np.float32
        assert labels.tolist() == [0, 2]
        assert labels.dtype == np.int32


# ---------------------------------------------------------------------
# load_scenes_json
# ---------------------------------------------------------------------


class TestLoadScenesJson:
    def test_wrapped_scenes_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            with open(p, "w") as f:
                json.dump({"info": {}, "scenes": [_mk_scene([_obj()])]}, f)
            out = load_scenes_json(p)
            assert isinstance(out, list)
            assert len(out) == 1
            assert out[0]["class_id"] == 0

    def test_bare_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            with open(p, "w") as f:
                json.dump([_mk_scene([_obj()])], f)
            out = load_scenes_json(p)
            assert len(out) == 1

    def test_unrecognized_structure_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            with open(p, "w") as f:
                json.dump({"foo": "bar"}, f)
            with pytest.raises(ValueError):
                load_scenes_json(p)


# ---------------------------------------------------------------------
# find_splits
# ---------------------------------------------------------------------


class TestFindSplits:
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            assert find_splits(td) == {}

    def test_full_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "CLEVR-Hans3")
            for split in ("train", "val", "test"):
                os.makedirs(os.path.join(root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(root, "scenes"), exist_ok=True)
            for split in ("train", "val", "test"):
                with open(os.path.join(root, "scenes", f"CLEVR_Hans3_{split}_scenes.json"), "w") as f:
                    json.dump({"scenes": []}, f)
            splits = find_splits(td)
            assert set(splits.keys()) == {"train", "val", "test"}
            for s in ("train", "val", "test"):
                assert os.path.isdir(splits[s]["images_dir"])
                assert os.path.isfile(splits[s]["scenes_json"])


# ---------------------------------------------------------------------
# infer_num_classes
# ---------------------------------------------------------------------


class TestInferNumClasses:
    def test_three_classes(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            scenes = [
                _mk_scene([_obj()], class_id=0),
                _mk_scene([_obj()], class_id=1),
                _mk_scene([_obj()], class_id=2),
            ]
            with open(p, "w") as f:
                json.dump({"scenes": scenes}, f)
            n = infer_num_classes({"scenes_json": p, "images_dir": td})
            assert n == 3

    def test_no_class_id_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            with open(p, "w") as f:
                json.dump({"scenes": [{"image_filename": "x.png", "objects": []}]}, f)
            n = infer_num_classes({"scenes_json": p, "images_dir": td})
            assert n == 0


# ---------------------------------------------------------------------
# Symbolic dataset construction (no real PNGs needed)
# ---------------------------------------------------------------------


class TestSymbolicDataset:
    def test_build_symbolic_dataset_shapes(self):
        # Lazy import to avoid forcing tf at module load.
        from train.logic.clevr_hans_data import build_symbolic_dataset

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.json")
            scenes = [
                _mk_scene([_obj()], class_id=i % 3)
                for i in range(7)
            ]
            with open(p, "w") as f:
                json.dump({"scenes": scenes}, f)
            ds, n = build_symbolic_dataset(
                {"scenes_json": p, "images_dir": td},
                max_objects=4, batch_size=3,
            )
            assert n == 7
            xb, yb = next(iter(ds))
            assert tuple(xb.shape)[1:] == (4, FEATURE_WIDTH)
            assert tuple(yb.shape)[0] == xb.shape[0]
