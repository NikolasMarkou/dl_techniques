"""An unlabeled image that cannot be read stops the run instead of becoming zeros.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.6 (D-130). At
308a4dfbf ``UnlabeledImageDataset.__getitem__`` caught every exception from
the image read and appended an all-zero image: a missing or corrupt file
became a uniform grey input that the student and the teacher both read, with
nothing in the log.
"""

import numpy as np
import pytest
from PIL import Image

from train.common.megadepth import UnlabeledImageDataset


def _files(tmp_path):
    good = tmp_path / "good.png"
    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    return str(good), str(bad)


def test_a_corrupt_image_raises_and_names_its_path(tmp_path):
    good, bad = _files(tmp_path)
    data = UnlabeledImageDataset([good, bad], batch_size=2, patch_size=4,
                                 is_training=False, workers=1)
    with pytest.raises(OSError, match="bad.jpg"):
        data[0]


def test_a_missing_image_raises(tmp_path):
    good, _ = _files(tmp_path)
    data = UnlabeledImageDataset([good, str(tmp_path / "gone.png")], batch_size=2,
                                 patch_size=4, is_training=False, workers=1)
    with pytest.raises(OSError, match="gone.png"):
        data[0]


def test_readable_images_still_load(tmp_path):
    good, _ = _files(tmp_path)
    batch = UnlabeledImageDataset([good, good], batch_size=2, patch_size=4,
                                  is_training=False, workers=1)[0]
    np.testing.assert_allclose(batch, 200 / 255.0 * 2.0 - 1.0, rtol=0, atol=1e-6)
