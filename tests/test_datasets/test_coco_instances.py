"""
Guards for ``COCO2017MultiTaskLoader._build_instances`` -- the per-instance
sibling of ``_build_mask``.

This loader had NO tests before this module (verified by grep over ``tests/``),
and it has other consumers (``src/dl_techniques/callbacks/coco_map_callback.py``
and ``src/train/sam/data.py``; a third, ``train/cliffordnet/train_coco_multitask.py``,
was deleted on 2026-08-10). So the module opens with a CHARACTERIZATION guard
pinning ``_build_mask``'s pre-change output before it asserts anything about the
new method.

The pinned SHA-256 digests were captured by executing the loader at the commit
BEFORE ``_build_instances`` existed, on ``val2017`` at ``image_size=128`` -- see
decision D-039 for the capture command. They are the regression instrument for
"the shared path did not move", which is the risk this extension creates.

Every test skips (loudly, with a reason) when the local COCO copy is absent.
That is a real coverage gap on a machine without the data, and it is declared
rather than papered over.
"""

import hashlib
import os
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("pycocotools", reason="pycocotools not installed")

from dl_techniques.datasets.vision.coco_multitask_local import (  # noqa: E402
    COCO_DEFAULT_ROOT,
    COCO2017MultiTaskLoader,
    COCOMultiTaskConfig,
)

MASK_SIZE = 128
SPLIT = "val2017"
ANNOTATION_FILE = os.path.join(
    COCO_DEFAULT_ROOT, "annotations", f"instances_{SPLIT}.json"
)

#: `_build_mask` output digests captured BEFORE `_build_instances` landed.
#: `image_size=128`, `split=val2017`, `shuffle=False`, first 12 ids in sorted
#: order. Any change to these means the shared path moved.
PRE_CHANGE_BUILD_MASK_SHA256: Dict[int, str] = {
    6818: "339b378589e1c0fb",
    17627: "fd691c91fcc4a866",
    37777: "e63b6fb840b33c42",
    41888: "ed636a24910bab56",
    58636: "de2f256064a0af79",
    87038: "9b8e9e6e21c6664c",
    122745: "2ee54d7d0e6db177",
    143931: "99124c7a7a88054b",
    153299: "aef1e5645a4f0163",
    174482: "448efe56f8c8466b",
    181666: "82a8f08d604f0100",
    184321: "11d90560d676b16a",
}
#: An image whose annotation list is empty after filtering -- the "zero
#: eligible annotations" edge case, found in the data rather than invented.
IMAGE_WITHOUT_INSTANCES = 58636

pytestmark = pytest.mark.skipif(
    not os.path.exists(ANNOTATION_FILE),
    reason=f"local COCO 2017 not found at {ANNOTATION_FILE}",
)


@pytest.fixture(scope="module")
def loader() -> COCO2017MultiTaskLoader:
    """One loader for the whole module -- the JSON load costs ~0.5 s."""
    return COCO2017MultiTaskLoader(
        COCOMultiTaskConfig(
            split=SPLIT,
            image_size=MASK_SIZE,
            batch_size=2,
            max_images=64,
            shuffle=False,
            augment=False,
            workers=1,
            use_multiprocessing=False,
            emit_boxes=True,
        )
    )


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


class TestBuildMaskIsUnchanged:
    """The characterization guard. This extension must not move it."""

    def test_the_semantic_map_is_value_exact_on_twelve_fixed_images(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        moved = {
            image_id: _digest(loader._build_mask(image_id, (MASK_SIZE, MASK_SIZE)))
            for image_id, expected in PRE_CHANGE_BUILD_MASK_SHA256.items()
            if _digest(loader._build_mask(image_id, (MASK_SIZE, MASK_SIZE)))
            != expected
        }
        assert moved == {}, f"_build_mask output moved for {moved}"

    def test_the_pinned_digests_are_not_all_identical(self) -> None:
        """
        Control. If the capture had gone wrong and every digest were the same
        constant, the guard above would pass against almost any implementation.
        """
        assert len(set(PRE_CHANGE_BUILD_MASK_SHA256.values())) == len(
            PRE_CHANGE_BUILD_MASK_SHA256
        )

    def test_the_detection_labels_still_match_the_inline_arithmetic(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        """
        ``_build_detection_labels``' xyxy conversion was extracted into
        ``_normalized_xyxy`` so ``_build_instances`` could reuse it rather than
        re-derive it. This re-implements the ORIGINAL inline arithmetic and
        demands byte equality, so the extraction is proved neutral rather than
        assumed to be.
        """
        max_boxes = loader.config.max_boxes
        for image_id in list(loader.image_ids)[:24]:
            expected = -1.0 * np.ones((max_boxes, 5), dtype=np.float32)
            info = loader.coco.loadImgs(image_id)[0]
            img_h, img_w = float(info["height"]), float(info["width"])
            anns = loader.coco.loadAnns(
                loader.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            )[:max_boxes]
            for index, ann in enumerate(anns):
                class_index = loader.cat_id_to_idx.get(ann["category_id"])
                if class_index is None:
                    continue
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                x1, y1 = max(0.0, x / img_w), max(0.0, y / img_h)
                x2, y2 = min(1.0, (x + w) / img_w), min(1.0, (y + h) / img_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                expected[index, 0] = float(class_index)
                expected[index, 1:] = [x1, y1, x2, y2]
            assert np.array_equal(
                expected, loader._build_detection_labels(image_id)
            ), f"_build_detection_labels moved on image {image_id}"


class TestNormalizedXyxy:
    """
    Direct unit guards for the extracted converter.

    These exist because the corpus-based guard above was MEASURED blind to one
    of them: deliberately deleting the ``min(1.0, ...)`` clip left all 10 tests
    green, since no box in the sampled val2017 range exceeds its image bounds.
    "0 violations over N rows" is worthless when the corpus was swept rather
    than attacked, so the clip is attacked here directly.
    """

    def test_a_box_running_past_the_right_edge_is_clipped(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        box = loader._normalized_xyxy({"bbox": [90.0, 10.0, 50.0, 20.0]}, 100.0, 100.0)
        assert box == (0.9, 0.1, 1.0, 0.3)

    def test_a_box_running_past_the_bottom_edge_is_clipped(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        box = loader._normalized_xyxy({"bbox": [10.0, 80.0, 20.0, 60.0]}, 100.0, 100.0)
        assert box == (0.1, 0.8, 0.3, 1.0)

    @pytest.mark.parametrize(
        "bbox",
        [[10.0, 10.0, 0.0, 20.0], [10.0, 10.0, 20.0, 0.0], [200.0, 10.0, 20.0, 20.0]],
    )
    def test_a_degenerate_box_is_refused(
        self, loader: COCO2017MultiTaskLoader, bbox: List[float]
    ) -> None:
        assert loader._normalized_xyxy({"bbox": bbox}, 100.0, 100.0) is None

    def test_an_ordinary_box_is_untouched(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        """Control: a converter that clipped everything to (0,0,1,1) would pass
        the two clipping tests above."""
        assert loader._normalized_xyxy(
            {"bbox": [10.0, 20.0, 30.0, 40.0]}, 100.0, 100.0
        ) == (0.1, 0.2, 0.4, 0.6)


class TestBuildInstancesIsPerInstance:
    """
    The probe the collapsed "last-painted wins" class map CANNOT satisfy.
    """

    @staticmethod
    def _first_image_with_two_same_class_instances(
        loader: COCO2017MultiTaskLoader,
    ) -> Any:
        for image_id in loader.image_ids:
            records = loader._build_instances(image_id, (MASK_SIZE, MASK_SIZE))
            grouped: Dict[int, List[Dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(record["class_index"], []).append(record)
            for class_index, group in grouped.items():
                if len(group) >= 2:
                    return image_id, class_index, group
        return None

    def test_two_instances_of_one_category_come_back_separately(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        found = self._first_image_with_two_same_class_instances(loader)
        assert found is not None, (
            "no image in the sampled range carries two instances of one "
            "category -- this probe would be vacuous"
        )
        image_id, class_index, group = found
        first, second = group[0]["mask"], group[1]["mask"]
        union = np.clip(first + second, 0.0, 1.0)

        assert not np.array_equal(first, second), "the two masks are identical"
        assert not np.array_equal(union, first)
        assert not np.array_equal(union, second)
        assert group[0]["annotation_id"] != group[1]["annotation_id"]

        # The load-bearing half: the collapsed class map cannot produce either
        # of these. Its slice for this category is the UNION of every instance
        # of it, further eroded wherever a later-painted category overwrote it.
        semantic = loader._build_mask(image_id, (MASK_SIZE, MASK_SIZE))
        collapsed = (semantic == class_index + 1).astype("float32")
        assert not np.array_equal(collapsed, first)
        assert not np.array_equal(collapsed, second)

    def test_every_returned_mask_is_binary_non_empty_and_correctly_shaped(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        total = 0
        for image_id in list(loader.image_ids)[:24]:
            for record in loader._build_instances(
                image_id, (MASK_SIZE, MASK_SIZE)
            ):
                mask = record["mask"]
                assert mask.shape == (MASK_SIZE, MASK_SIZE)
                assert mask.dtype == np.float32
                assert set(np.unique(mask)).issubset({0.0, 1.0})
                assert mask.sum() > 0
                total += 1
        assert total > 20, f"only {total} instances -- sample too small to mean much"

    def test_the_box_bounds_its_own_mask(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        """
        The box comes from COCO's ``bbox`` and the mask from ``annToMask``; if
        the two were ever paired up wrongly this is what would show it.
        """
        checked = 0
        for image_id in list(loader.image_ids)[:24]:
            for record in loader._build_instances(
                image_id, (MASK_SIZE, MASK_SIZE)
            ):
                x1, y1, x2, y2 = record["box"] * MASK_SIZE
                rows = np.flatnonzero(record["mask"].any(axis=1))
                cols = np.flatnonzero(record["mask"].any(axis=0))
                # One pixel of slack per side: the mask is nearest-resized from
                # the original resolution while the box is scaled exactly.
                assert cols[0] >= x1 - 1.5 and cols[-1] <= x2 + 1.5
                assert rows[0] >= y1 - 1.5 and rows[-1] <= y2 + 1.5
                checked += 1
        assert checked > 20

    def test_max_instances_truncates_deterministically(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        image_id = max(
            loader.image_ids,
            key=lambda i: len(loader._build_instances(i, (MASK_SIZE, MASK_SIZE)))
            if i in list(loader.image_ids)[:24]
            else 0,
        )
        full = loader._build_instances(image_id, (MASK_SIZE, MASK_SIZE))
        capped = loader._build_instances(
            image_id, (MASK_SIZE, MASK_SIZE), max_instances=2
        )
        assert len(capped) == min(2, len(full))
        for a, b in zip(capped, full):
            assert a["annotation_id"] == b["annotation_id"]


class TestZeroAnnotationImagesAreExplicit:
    """
    LESSONS: a silent-drop branch under-reports even after it is deleted. The
    drop must be visible in the ARTIFACT, not merely absent from the output.
    """

    def test_an_image_with_no_eligible_annotation_returns_an_empty_list(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        assert (
            loader._build_instances(
                IMAGE_WITHOUT_INSTANCES, (MASK_SIZE, MASK_SIZE)
            )
            == []
        )

    def test_that_drop_is_counted(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        before = loader.instance_drop_counts["images_without_instances"]
        loader._build_instances(IMAGE_WITHOUT_INSTANCES, (MASK_SIZE, MASK_SIZE))
        assert (
            loader.instance_drop_counts["images_without_instances"] == before + 1
        )

    def test_a_normal_image_does_not_increment_that_counter(
        self, loader: COCO2017MultiTaskLoader
    ) -> None:
        """Control: a counter that increments unconditionally counts nothing."""
        populated = next(
            i
            for i in loader.image_ids
            if loader._build_instances(i, (MASK_SIZE, MASK_SIZE))
        )
        before = loader.instance_drop_counts["images_without_instances"]
        loader._build_instances(populated, (MASK_SIZE, MASK_SIZE))
        assert loader.instance_drop_counts["images_without_instances"] == before
