"""COCO 2017 instance annotations rasterized to a semantic mask of 80 thing classes and a background.

``instances_{train,val}2017.json`` holds polygons and run-length regions of the 80 COCO thing classes and
no stuff and no panoptic annotation, so a semantic mask can only say "this thing" or "no thing here":
class 0 is everything that is not an annotated thing (stuff, unannotated things, nothing), classes 1 to 80
are the things in category-id order, and ``255`` is void. This is NOT COCO-Stuff (171 classes) and its mIoU
is not comparable to published COCO-Stuff or panoptic numbers.

The rules (D-018, with the sources there):

* a crowd annotation (``iscrowd=1``) paints void; the COCO evaluation ignores crowd regions the same way;
* a pixel covered by instances of two DIFFERENT classes is void (``torchvision`` references/segmentation
  ``coco_utils`` voids every pixel two instances cover; two instances of one class do not disagree, so
  here they keep the class);
* the result does not depend on the order of the annotations.

Needs numpy; ``pycocotools`` is imported when a region is decoded and is NOT a declared dependency of this
repository (it is installed in the project environment). Never imports TensorFlow.

Interface contract
------------------
``category_table(categories)``
    :param categories: the ``categories`` list of an instances file.
    :return: ``(class_names, index_of)``: ``class_names[0]`` is ``"background"``, then the category names in id
        order; ``index_of`` maps a category id to its class index (1 to 80).
    :raises CocoError: for a repeated id or name.

``annotation_region(segmentation, height, width)``
    :param segmentation: a polygon list, an uncompressed run-length dict or a compressed one (COCO's three forms).
    :return: ``bool`` ``(height, width)``.
    :raises CocoError: if ``pycocotools`` is not installed.

``rasterize(height, width, annotations, index_of)``
    :param annotations: dicts with ``category_id``, ``iscrowd`` and ``segmentation``.
    :return: ``(mask, crowd_pixels, conflict_pixels)``: ``uint8`` ``(height, width)`` and the number of pixels
        crowd regions and class conflicts made void (a pixel in both counts in both).
    :raises CocoError: for a category id that is not in ``index_of``.
"""

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

#: The mask value of a pixel two classes claim, or a crowd region covers.
VOID_INDEX = 255
BACKGROUND = "background"


class CocoError(ValueError):
    """An annotation file or a region that cannot be rasterized; the message says why."""


def category_table(categories: Sequence[Mapping[str, Any]]) -> Tuple[Tuple[str, ...], Dict[int, int]]:
    """Return the class names (background first) and the map from a category id to a class index."""
    ordered = sorted(categories, key=lambda category: category["id"])
    identifiers = [category["id"] for category in ordered]
    names = [category["name"] for category in ordered]
    if len(set(identifiers)) != len(identifiers) or len(set(names)) != len(names) or BACKGROUND in names:
        raise CocoError("the categories hold a repeated id or name (or a category named 'background')")
    return (BACKGROUND, *names), {identifier: index + 1 for index, identifier in enumerate(identifiers)}


def annotation_region(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Return the pixels one annotation covers as a ``bool`` array; see the module contract."""
    try:
        from pycocotools import mask as coco_mask
    except ModuleNotFoundError as exc:
        raise CocoError("pycocotools is not installed; it reads COCO's polygons and run-length regions "
                        "(pip install pycocotools); it is not a declared dependency of this repository") from exc
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        encoded = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))     # parts of one object
    elif isinstance(segmentation["counts"], list):
        encoded = coco_mask.frPyObjects(segmentation, height, width)                     # uncompressed run lengths
    else:
        encoded = segmentation
    return coco_mask.decode(encoded).astype(bool)


def rasterize(height: int, width: int, annotations: Sequence[Mapping[str, Any]],
              index_of: Mapping[int, int]) -> Tuple[np.ndarray, int, int]:
    """Return the semantic mask of one image and its void counts; see the module contract."""
    mask = np.zeros((height, width), dtype=np.uint8)
    conflict = np.zeros((height, width), dtype=bool)
    crowd = np.zeros((height, width), dtype=bool)
    for annotation in annotations:
        region = annotation_region(annotation["segmentation"], height, width)
        if annotation["iscrowd"]:
            crowd |= region
            continue
        if annotation["category_id"] not in index_of:
            raise CocoError(f"category id {annotation['category_id']} is not one of the file's categories")
        label = index_of[annotation["category_id"]]
        # DECISION plan-2026-11-03T221936-98b866fb/D-018
        # Two instances of different classes on one pixel leave it void; one class twice does not. The
        # test is "a label already there that differs", so the outcome does not depend on the order of
        # the annotations. Do NOT paint later over earlier (cocostuffapi and panopticapi paint in file order
        # but RAISE on an overlap, so they are no rule for COCO's thing instances, which overlap), and do
        # NOT void every overlap as torchvision does (a crowd of people overlapping one person is no disagreement).
        conflict |= region & (mask != 0) & (mask != label)
        mask[region & (mask == 0)] = label
    voided = conflict | crowd
    mask[voided] = VOID_INDEX
    return mask, int(crowd.sum()), int(conflict.sum())
