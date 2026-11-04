"""``dl_techniques.datasets.vision.coco_semantic``: COCO instances rasterized to a semantic mask (plan 98b866fb step 6, D-018).

The rules, each on arrays drawn here:

* the class table is background 0 and the categories in id order, whatever the ids (COCO's run 1 to 90 with gaps);
* a polygon (several parts too), an uncompressed run-length region and a compressed one cover their pixels;
* a crowd region is void and wins over an instance under it;
* two instances of DIFFERENT classes on a pixel leave it void, two of one class keep the class;
* the mask does not depend on the order of the annotations;
* no annotation gives an all-background mask; an unknown category is refused; the void counts are exact.
"""

import itertools

import numpy as np
import pytest

from dl_techniques.datasets.vision import coco_semantic as cs

coco_mask = pytest.importorskip("pycocotools.mask")

H, W = 8, 10
INDEX = {1: 1, 3: 2, 90: 3}


def rectangle(x0, y0, x1, y1):
    """The polygon of a rectangle (COCO's x, y order), as one part."""
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def region_of(x0, y0, x1, y1):
    """The pixels a rectangle polygon covers, read back through the library (so tests compare like with like)."""
    return cs.annotation_region(rectangle(x0, y0, x1, y1), H, W)


def thing(category, polygon, crowd=0):
    return {"category_id": category, "iscrowd": crowd, "segmentation": polygon}


def compressed_rle(region):
    rle = coco_mask.encode(np.asfortranarray(region.astype(np.uint8)))
    return {"size": [H, W], "counts": rle["counts"].decode("ascii")}


def uncompressed_rle(region):
    flat = region.T.ravel().astype(np.uint8)                        # run lengths go down the columns, starting with zeros
    changes = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], changes, [flat.size]])).tolist()
    return {"size": [H, W], "counts": runs if flat[0] == 0 else [0, *runs]}


def test_the_class_table_is_background_then_the_categories_in_id_order():
    names, index_of = cs.category_table([{"id": 90, "name": "toaster"}, {"id": 1, "name": "person"}, {"id": 3, "name": "car"}])
    assert names == ("background", "person", "car", "toaster") and index_of == INDEX


def test_a_repeated_id_or_name_or_a_category_called_background_is_refused():
    for categories in ([{"id": 1, "name": "a"}, {"id": 1, "name": "b"}], [{"id": 1, "name": "a"}, {"id": 2, "name": "a"}],
                       [{"id": 1, "name": "background"}]):
        with pytest.raises(cs.CocoError, match="repeated id or name"):
            cs.category_table(categories)


def test_a_polygon_with_two_parts_covers_both():
    parts = rectangle(0, 0, 3, 3) + rectangle(6, 4, 9, 7)
    mask, crowd, conflict = cs.rasterize(H, W, [thing(3, parts)], INDEX)
    expected = cs.annotation_region(parts, H, W)
    assert (mask[expected] == 2).all() and (mask[~expected] == 0).all() and expected.sum() == 18
    assert (crowd, conflict) == (0, 0)


def test_a_rectangle_polygon_covers_the_pixels_between_its_corners():
    region = cs.annotation_region(rectangle(1, 2, 4, 5), H, W)
    rows, columns = np.nonzero(region)
    assert region.sum() == 9 and (rows.min(), rows.max(), columns.min(), columns.max()) == (2, 4, 1, 3)


def test_the_three_forms_of_a_region_cover_the_same_pixels():
    target = region_of(2, 1, 7, 6)
    assert target.sum() > 10
    for segmentation in (rectangle(2, 1, 7, 6), uncompressed_rle(target), compressed_rle(target)):
        assert (cs.annotation_region(segmentation, H, W) == target).all()


def test_no_annotation_is_all_background_and_an_empty_polygon_list_is_nothing():
    mask, crowd, conflict = cs.rasterize(H, W, [], INDEX)
    assert mask.dtype == np.uint8 and mask.shape == (H, W) and not mask.any() and (crowd, conflict) == (0, 0)
    assert not cs.rasterize(H, W, [thing(1, [])], INDEX)[0].any()


def test_two_classes_on_a_pixel_leave_it_void_and_one_class_twice_keeps_it():
    a, b, c = region_of(0, 0, 6, 6), region_of(3, 3, 9, 7), region_of(4, 3, 8, 6)
    mask, crowd, conflict = cs.rasterize(H, W, [thing(1, rectangle(0, 0, 6, 6)), thing(3, rectangle(3, 3, 9, 7))], INDEX)
    overlap = a & b
    assert overlap.sum() > 5
    assert (mask[overlap] == cs.VOID_INDEX).all() and (mask[a & ~b] == 1).all() and (mask[b & ~a] == 2).all()
    assert (crowd, conflict) == (0, int(overlap.sum()))
    # the same class twice: no disagreement
    mask, _, conflict = cs.rasterize(H, W, [thing(1, rectangle(0, 0, 6, 6)), thing(1, rectangle(3, 3, 9, 7))], INDEX)
    assert (mask[a | b] == 1).all() and conflict == 0
    # a third instance inside the overlap of two classes stays void
    mask, _, _ = cs.rasterize(H, W, [thing(1, rectangle(0, 0, 6, 6)), thing(3, rectangle(3, 3, 9, 7)), thing(3, rectangle(4, 3, 8, 6))], INDEX)
    assert (mask[overlap] == cs.VOID_INDEX).all() and c.sum() > 4


def test_a_crowd_region_is_void_and_wins_over_an_instance_under_it():
    crowd_region = region_of(2, 2, 8, 6)
    person = region_of(0, 0, 5, 5)
    for form in (compressed_rle, uncompressed_rle):
        mask, crowd, conflict = cs.rasterize(H, W, [thing(1, rectangle(0, 0, 5, 5)), thing(1, form(crowd_region), crowd=1)], INDEX)
        assert (mask[crowd_region] == cs.VOID_INDEX).all() and (mask[person & ~crowd_region] == 1).all()
        assert (crowd, conflict) == (int(crowd_region.sum()), 0)


def test_the_mask_does_not_depend_on_the_order_of_the_annotations():
    annotations = [thing(1, rectangle(0, 0, 6, 6)), thing(3, rectangle(3, 3, 9, 7)), thing(3, rectangle(1, 1, 4, 4)),
                   thing(1, compressed_rle(region_of(5, 0, 9, 3)), crowd=1)]
    reference = cs.rasterize(H, W, annotations, INDEX)
    for order in itertools.permutations(range(len(annotations))):
        got = cs.rasterize(H, W, [annotations[i] for i in order], INDEX)
        assert np.array_equal(got[0], reference[0]) and got[1:] == reference[1:]


def test_a_category_that_is_not_in_the_file_is_refused():
    with pytest.raises(cs.CocoError, match="category id 7"):
        cs.rasterize(H, W, [thing(7, rectangle(0, 0, 3, 3))], INDEX)
