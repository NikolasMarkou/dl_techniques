"""The Depth Anything trainer feeds ImageNet-standardized RGB, the DINOv2 encoder's input.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.5 (D-139). The
official DINOv2 (and Depth Anything) were trained on RGB in ``[0, 1]``
standardized by the ImageNet mean and std (the reference transform's
``NormalizeImage``); ``train.common.megadepth`` emits ``[-1, +1]``. The
datasets standardize when asked (``imagenet_standardize=True``), the trainer
asks for it on every dataset it builds (its own test,
``test_the_decoder_trains_10x_faster_from_a_pretrained_start.py``), and the
depth monitor shows the un-standardized image while predicting on the
standardized one.
"""

import numpy as np
import pytest
from PIL import Image

from dl_techniques.datasets.vision.imagenet2012 import IMAGENET_MEAN, IMAGENET_STD
from train.common.megadepth import (
    MegaDepthDataset,
    UnlabeledImageDataset,
    standardize_imagenet,
)


def test_the_standardization_is_imagenets_on_zero_one_rgb():
    # A pixel equal to the ImageNet mean maps to 0; one std above it to 1.
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    pixels = np.stack([mean * 2 - 1, (mean + std) * 2 - 1]).astype(np.float32)
    np.testing.assert_allclose(standardize_imagenet(pixels), [[0, 0, 0], [1, 1, 1]],
                               rtol=0, atol=1e-6)
    assert standardize_imagenet(pixels).dtype == np.float32


def _image(tmp_path, value):
    path = tmp_path / f"im{value}.png"
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(path)
    return str(path)


@pytest.mark.parametrize("flag", [False, True])
def test_the_unlabeled_dataset_standardizes_only_when_asked(tmp_path, flag):
    data = UnlabeledImageDataset([_image(tmp_path, 200)] * 2, batch_size=2, patch_size=4,
                                 is_training=False, workers=1, imagenet_standardize=flag)
    raw = np.float32(200 / 255 * 2 - 1)
    expected = standardize_imagenet(np.full((3,), raw)) if flag else np.full((3,), raw)
    np.testing.assert_allclose(data[0][0, 0, 0], expected, rtol=0, atol=1e-6)


def test_the_labeled_dataset_standardizes_only_when_asked(monkeypatch):
    rgb = np.random.default_rng(0).uniform(-1, 1, size=(4, 4, 3)).astype(np.float32)
    target = np.ones((4, 4, 2), np.float32)
    monkeypatch.setattr("train.common.megadepth.load_and_process_pair",
                        lambda *a, **k: (rgb.copy(), target.copy()))
    batches = [MegaDepthDataset(["a"] * 2, ["b"] * 2, batch_size=2, patch_size=4, augment=False,
                                is_training=False, workers=1, imagenet_standardize=flag)[0][0]
               for flag in (False, True)]
    np.testing.assert_array_equal(batches[0][0], rgb)
    np.testing.assert_allclose(batches[1][0], standardize_imagenet(rgb), rtol=0, atol=1e-6)


def test_the_depth_monitor_predicts_on_one_and_shows_the_other(tmp_path, monkeypatch):
    from dl_techniques.callbacks import depth_visualization as dv

    shown = []

    class _Axis:  # stands for the figure, the axes grid and each axis
        def imshow(self, img, **kwargs):
            shown.append(np.asarray(img))

        def __getitem__(self, key):
            return self

        def __getattr__(self, name):
            return lambda *a, **k: None

    class _Plt:
        def subplots(self, *a, **k):
            return _Axis(), _Axis()

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(dv, "import_pyplot", lambda: _Plt())
    display = np.full((1, 4, 4, 3), 0.5, np.float32)
    callback = dv.DepthPredictionGridCallback(
        val_rgb=standardize_imagenet(display), val_depth=np.ones((1, 4, 4, 1), np.float32),
        val_mask=np.ones((1, 4, 4, 1), np.float32), output_dir=str(tmp_path), max_samples=1,
        display_rgb=display)
    callback._save_grid(1, np.ones((1, 4, 4, 1), np.float32))
    np.testing.assert_allclose(shown[0], (display[0] + 1) / 2)
