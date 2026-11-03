"""Three open items of the iteration 3 gate, pinned on a CPU fixture (plan 98b866fb step 1, items c to e).

* ``init_scale_warning`` of a BatchNorm segmenter is ``null`` BY DESIGN (D-045): AccUNet's BatchNorm layers
  hold the statistics they were built with at epoch 0, so the inference-mode pass that gives
  ``initial_loss_ratio`` normalizes nothing and says nothing about the first training step. The summary
  must carry the sentence that says so; ConvUNext (LayerNorm only) is judged, so its flag is a bool. The
  contrast is the reason the gate read the AccUNet ``null`` as a difference.
* An EXIF-oriented file and its upright twin that decode to EQUAL arrays must give equal predictions, for
  both U-Nets. A second JPEG generation (re-encoding a rotated picture) changes the bytes, so the
  prediction may move with them; that is a property of the weights, measured in
  ``scratch_step1/exif_stability.json`` and stated in the README, and the test pins only that the
  re-encoded file is a different array (so the equal-array case above is not vacuous).

The tests write only under ``tmp_path``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from train.api.predictor import load_model
from train.api.segmentation_models import create_accunet, create_convunext
from train.api.tasks.image_segmentation import train_image_segmentation
from train.common.classification_run import unset_batch_statistics
from train.common.segmentation_data import image_to_rgb

from . import _segmentation_fixtures as fx
from .test_image_segmentation_task import SIZE, data_of, folders, world  # noqa: F401  (fixtures)

NOT_JUDGED = "`init_scale_warning` is null (not judged)"
ORIENT_6 = 6


@pytest.fixture(scope="module", params=["accunet/base", "convunext/tiny"])
def run(request, world, folders, tmp_path_factory):  # noqa: F811
    """One real CPU fit (1 epoch) per family: ``(model, run directory, summary)``."""
    model = request.param
    out = tmp_path_factory.mktemp("open_items_" + model.replace("/", "_")) / "run"
    result = train_image_segmentation(
        out=str(out), model=model, data=data_of(folders, "clean"), epochs=1, batch_size=8,
        transform_args={"image_size": SIZE},
        analysis_args={"model_analysis": False, "figures": False},
    )
    return model, Path(result.run_dir), json.loads(Path(result.summary_path).read_text())


# ---------------------------------------------------------------------
# init_scale_warning: null for BatchNorm statistics at their initial values
# ---------------------------------------------------------------------

def test_a_batchnorm_segmenter_lists_its_unset_statistics_and_a_layernorm_one_does_not():
    """The mechanism: ``unset_batch_statistics`` names AccUNet's BatchNorm layers on a fresh model."""
    shape = (SIZE, SIZE, 3)
    accunet = unset_batch_statistics(create_accunet(variant="base", num_classes=3, input_shape=shape))
    assert len(accunet) > 10, accunet
    assert all("batch" in name.lower() or "norm" in name.lower() or "bn" in name.lower() for name in accunet), accunet
    # The control: the same function on the LayerNorm-only U-Net finds nothing to withhold a verdict for.
    assert unset_batch_statistics(create_convunext(variant="tiny", num_classes=3, input_shape=shape)) == []


def test_the_summary_says_why_the_verdict_is_withheld_or_gives_one(run):
    model, _, summary = run
    text = " ".join(summary["notes"])
    if model.startswith("accunet"):
        assert summary["init_scale_warning"] is None
        assert NOT_JUDGED in text
        # The sentence names what was measured and what was not, so a null is not read as "fine".
        assert "BatchNorm" in text and "NOT caught here" in text
        assert summary["initial_loss_ratio"] > 0
    else:
        assert summary["init_scale_warning"] is False
        assert NOT_JUDGED not in text


# ---------------------------------------------------------------------
# EXIF orientation
# ---------------------------------------------------------------------

def _photo() -> np.ndarray:
    """A non-square picture with structure at every scale (the prediction depends on the pixels)."""
    rng = np.random.default_rng(7)
    ramp = np.linspace(0, 255, 56, dtype=np.float64)[None, :, None] * np.ones((40, 56, 3))
    return np.clip(ramp + fx.picture(fx.blocks(40, 56, classes=4, seed=3)) * 0.4 + rng.normal(0, 6, (40, 56, 3)),
                   0, 255).astype(np.uint8)


def _exif(orientation: int) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif


def test_equal_arrays_give_equal_predictions_exif_or_not(run, tmp_path):
    _, run_dir, _ = run
    predictor = load_model(run_dir)
    upright = tmp_path / "upright.jpg"
    Image.fromarray(_photo()).save(upright, quality=95)
    array = np.asarray(image_to_rgb(Image.open(upright)))
    # The twin: the DECODED array rotated losslessly and stored with orientation 6 (rotate back by 90 degrees clockwise).
    twin = tmp_path / "twin_exif6.png"
    Image.fromarray(np.ascontiguousarray(np.rot90(array, k=1))).save(twin, exif=_exif(ORIENT_6))
    assert Image.open(twin).size == (array.shape[0], array.shape[1]), "the stored file is the rotated picture"
    decoded = np.asarray(image_to_rgb(Image.open(twin)))
    assert decoded.shape == array.shape and np.array_equal(decoded, array), "the premise: equal decoded arrays"

    one = predictor.predict(str(upright))["masks"][0]
    two = predictor.predict(str(twin))["masks"][0]
    assert one.shape == two.shape == array.shape[:2]
    assert np.array_equal(one, two), f"{(one != two).mean():.4f} of the pixels differ for equal arrays"
    # Batched together the answers stay equal to the singles (the order of a batch moves nothing).
    both = predictor.predict([str(upright), str(twin)])["masks"]
    assert np.array_equal(both[0], one) and np.array_equal(both[1], two)
    # A one-epoch model may predict one class everywhere, which makes equal masks cheap: the SCORES are the finding.
    # ``_logits`` is the predictor's own seam from decoded bytes to the model's output (the masks are built from it).
    # One image per call: inside one batch the CPU kernels round two identical items differently (1e-8), which says
    # nothing about the decoder.
    scores = [np.asarray(predictor._logits([image])[0]) for image in (decoded, array)]
    assert np.array_equal(scores[0], scores[1])
    # The control: an array that really is different gives different scores, so the equality above is a finding.
    other = np.asarray(predictor._logits([np.ascontiguousarray(255 - array)])[0])
    assert not np.array_equal(other, scores[0])


def test_a_reencoded_rotation_is_a_different_array_so_its_prediction_may_differ(run, tmp_path):
    """Re-encoding a rotated picture as a JPEG changes the pixels; the answer's SIZE still follows the EXIF."""
    _, run_dir, _ = run
    predictor = load_model(run_dir)
    upright = tmp_path / "upright.jpg"
    Image.fromarray(_photo()).save(upright, quality=95)
    base = Image.open(upright).convert("RGB")
    again = tmp_path / "again_exif6.jpg"
    base.rotate(90, expand=True).save(again, quality=95, exif=_exif(ORIENT_6))
    array = np.asarray(image_to_rgb(Image.open(upright)))
    rotated_back = np.asarray(image_to_rgb(Image.open(again)))
    assert rotated_back.shape == array.shape
    assert not np.array_equal(rotated_back, array), "a second JPEG generation must change some byte"
    mask = predictor.predict(str(again))["masks"][0]
    assert mask.shape == array.shape[:2]
