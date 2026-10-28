"""``train.common.image_text.hold_out_images``: the selection cut of an image-caption train split.

COCO lists about five captions per image, and retrieval is scored one caption per image, so the
cut must be BY IMAGE: a held-out image may not keep any caption in the fit set. The fixture
gives image ``i`` five captions whose token row encodes ``(i, caption index)``, so the identity
of every pair is read straight off the token matrix, never inferred from the function under test.
"""

import numpy as np
import pytest

from train.common.image_text import hold_out_images

N_IMAGES = 40
CAPTIONS = 5


def _split():
    paths, tokens = [], []
    for image in range(N_IMAGES):
        for caption in range(CAPTIONS):
            paths.append(f"/coco/train2017/{image:06d}.jpg")
            tokens.append([image, caption, 7, 7])
    return paths, np.asarray(tokens, dtype=np.int32)


def _images_of(tokens):
    return {int(row[0]) for row in tokens}


def test_held_out_images_keep_no_caption_in_train() -> None:
    paths, tokens = _split()
    train_paths, train_tokens, select_paths, select_tokens = hold_out_images(paths, tokens, 8, 3)
    held = _images_of(select_tokens)
    assert len(held) == 8
    assert not held & _images_of(train_tokens), "a held-out image still has captions in train"
    assert not set(select_paths) & set(train_paths)


def test_selection_has_exactly_one_caption_per_image_and_it_is_the_first_one() -> None:
    paths, tokens = _split()
    _, _, select_paths, select_tokens = hold_out_images(paths, tokens, 8, 3)
    assert len(select_paths) == len(set(select_paths)) == 8
    assert (select_tokens[:, 1] == 0).all(), "the selection caption is not the split's first one"


def test_nothing_is_lost_or_invented() -> None:
    paths, tokens = _split()
    train_paths, train_tokens, select_paths, select_tokens = hold_out_images(paths, tokens, 8, 3)
    assert len(train_paths) == len(train_tokens) == (N_IMAGES - 8) * CAPTIONS
    assert len(select_paths) == len(select_tokens) == 8
    assert _images_of(train_tokens) | _images_of(select_tokens) == set(range(N_IMAGES))
    # every returned pair is a pair of the input: the paths and tokens stay aligned
    for path, row in list(zip(train_paths, train_tokens)) + list(zip(select_paths, select_tokens)):
        assert path == f"/coco/train2017/{int(row[0]):06d}.jpg"


def test_the_cut_is_deterministic_in_the_seed_and_depends_on_it() -> None:
    paths, tokens = _split()
    a = _images_of(hold_out_images(paths, tokens, 8, 3)[3])
    b = _images_of(hold_out_images(paths, tokens, 8, 3)[3])
    c = _images_of(hold_out_images(paths, tokens, 8, 4)[3])
    assert a == b
    assert a != c


def test_a_one_caption_per_image_split_cuts_the_same_way() -> None:
    paths = [f"/cc3m/train/{i:08d}.jpg" for i in range(30)]
    tokens = np.arange(30, dtype=np.int32)[:, None] * np.ones((1, 4), np.int32)
    train_paths, _, select_paths, _ = hold_out_images(paths, tokens, 5, 0)
    assert len(select_paths) == 5 and len(train_paths) == 25
    assert not set(select_paths) & set(train_paths)


@pytest.mark.parametrize("n_select, message", [
    (0, "cannot hold out 0 images from a train split of 40 distinct images \\(200 pairs\\)"),
    (-3, "cannot hold out -3 images"),
    (40, "cannot hold out 40 images from a train split of 40"),
    (400, "Lower --selection-images"),
])
def test_a_bad_selection_size_is_refused_naming_the_numbers(n_select, message) -> None:
    paths, tokens = _split()
    with pytest.raises(ValueError, match=message):
        hold_out_images(paths, tokens, n_select, 0)


def test_misaligned_inputs_are_refused() -> None:
    paths, tokens = _split()
    with pytest.raises(ValueError, match="must align one to one"):
        hold_out_images(paths[:-1], tokens, 8, 0)
