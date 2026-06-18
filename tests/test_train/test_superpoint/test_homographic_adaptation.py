"""Unit tests for ``select_weighted_image_paths`` (SuperPoint stage 3 HA).

Covers the per-dataset weighted worklist helper (plan SC5): balanced default
split, explicit weights, determinism, exact total count, undersized-dir
wraparound, ``num_images=None`` (every path once), and the config weight/dir
length-mismatch guard.

The helper is pure (stdlib + numpy + an injectable ``collect_fn``); a stub
``collect_fn`` gives deterministic control over each dir's pool without touching
the filesystem. The return element is verified to be ``(dataset_name, path)``
with ``dataset_name == Path(dir).name``.
"""

from collections import Counter
from pathlib import Path

import pytest

from train.superpoint.homographic_adaptation import (
    select_weighted_image_paths,
    HomographicAdaptationConfig,
)


# ---------------------------------------------------------------------
# collect_fn stubs
# ---------------------------------------------------------------------


def _make_collect_fn(counts_by_name):
    """Build a ``collect_fn(dirs, shuffle_seed=, sort=)`` stub.

    Returns ``[f"{dir}/img{i}.png" for i in range(N)]`` for the single dir in
    ``dirs``, where ``N = counts_by_name[Path(dir).name]``. Deterministic and
    filesystem-free; mirrors the real ``collect_image_paths`` call shape
    (single-element ``dirs`` list, keyword ``shuffle_seed`` / ``sort``).
    """

    def _collect_fn(dirs, shuffle_seed=None, sort=True):
        assert len(dirs) == 1, "helper must call collect_fn one dir at a time"
        d = dirs[0]
        n = counts_by_name[Path(d).name]
        return [f"{d}/img{i}.png" for i in range(n)]

    return _collect_fn


COCO = "/data/COCO/train2017"
DIV2K = "/data/div2k/train"


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestSelectWeightedImagePaths:

    def test_balanced_default_splits_50_50(self):
        collect_fn = _make_collect_fn({"train2017": 1000, "train": 1000})
        result = select_weighted_image_paths(
            [COCO, DIV2K], weights=None, num_images=100, seed=42,
            collect_fn=collect_fn,
        )
        counts = Counter(name for name, _ in result)
        assert counts == {"train2017": 50, "train": 50}

    def test_explicit_weights_90_10(self):
        collect_fn = _make_collect_fn({"train2017": 1000, "train": 1000})
        result = select_weighted_image_paths(
            [COCO, DIV2K], weights=[0.9, 0.1], num_images=100, seed=42,
            collect_fn=collect_fn,
        )
        counts = Counter(name for name, _ in result)
        assert counts == {"train2017": 90, "train": 10}

    def test_deterministic_same_seed(self):
        collect_fn = _make_collect_fn({"train2017": 500, "train": 300})
        a = select_weighted_image_paths(
            [COCO, DIV2K], weights=None, num_images=77, seed=7,
            collect_fn=collect_fn,
        )
        b = select_weighted_image_paths(
            [COCO, DIV2K], weights=None, num_images=77, seed=7,
            collect_fn=collect_fn,
        )
        assert a == b

    @pytest.mark.parametrize(
        "dirs,weights,num,counts",
        [
            ([COCO, DIV2K], [0.7, 0.3], 10, {"train2017": 500, "train": 500}),
            ([COCO, DIV2K], None, 101, {"train2017": 500, "train": 500}),
            (
                ["/d/a", "/d/b", "/d/c"],
                [1 / 3, 1 / 3, 1 / 3],
                10,
                {"a": 200, "b": 200, "c": 200},
            ),
        ],
    )
    def test_total_count_exact(self, dirs, weights, num, counts):
        collect_fn = _make_collect_fn(counts)
        result = select_weighted_image_paths(
            dirs, weights=weights, num_images=num, seed=3, collect_fn=collect_fn,
        )
        assert len(result) == num

    def test_undersized_dir_wraparound(self):
        # DIV2K stub yields only 3 files but its ~50 quota forces wraparound.
        collect_fn = _make_collect_fn({"train2017": 1000, "train": 3})
        result = select_weighted_image_paths(
            [COCO, DIV2K], weights=[0.5, 0.5], num_images=100, seed=11,
            collect_fn=collect_fn,
        )
        # Exact total, no crash.
        assert len(result) == 100
        div2k_paths = [p for name, p in result if name == "train"]
        # 50% quota -> 50 entries, all cycled from the 3 unique stub paths.
        assert len(div2k_paths) == 50
        expected_unique = {f"{DIV2K}/img{i}.png" for i in range(3)}
        assert set(div2k_paths) == expected_unique

    def test_num_images_none_uses_all(self):
        collect_fn = _make_collect_fn({"train2017": 40, "train": 25})
        result = select_weighted_image_paths(
            [COCO, DIV2K], weights=None, num_images=None, seed=1,
            collect_fn=collect_fn,
        )
        paths = [p for _, p in result]
        # Every collected path appears exactly once: no dups, no omissions.
        assert len(paths) == 65
        assert len(set(paths)) == 65
        all_expected = {f"{COCO}/img{i}.png" for i in range(40)}
        all_expected |= {f"{DIV2K}/img{i}.png" for i in range(25)}
        assert set(paths) == all_expected

    def test_return_element_is_name_path_pair(self):
        collect_fn = _make_collect_fn({"train2017": 10, "train": 10})
        result = select_weighted_image_paths(
            [COCO, DIV2K], weights=None, num_images=4, seed=0,
            collect_fn=collect_fn,
        )
        for name, path in result:
            assert name == Path(path).parent.name
            assert name in {"train2017", "train"}
            assert name == Path(COCO).name or name == Path(DIV2K).name

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            HomographicAdaptationConfig(
                image_dirs=["/a", "/b"], dataset_weights=[1.0]
            )
