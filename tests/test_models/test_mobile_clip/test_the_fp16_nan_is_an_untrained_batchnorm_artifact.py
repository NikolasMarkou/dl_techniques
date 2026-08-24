"""MobileClip v1's fp16 NaN and sub-unit feature norm are NOT model defects.

Batch 4 recorded two CRITICALs against `MobileClipModel` (v1): 1024 NaN of 1024
image features under `mixed_float16`, and `||image_features||_2` far below the
contract of 1.0 at float32. Batch 8 then found the same shape in `yolo12` and
showed it was an artifact: sixty BatchNorm updates took it from 100% NaN to
matching float32.

RE-MEASURED here on variant "b", GPU 1, and the same explanation holds
EXACTLY:

    BN updates   float32 ||f||_2      mixed_float16
    0            0.007333             1024/1024 NaN
    60           1.0                  0 NaN, 1.001142

Both symptoms are the SAME phenomenon -- a randomly initialised BatchNorm ladder
evaluated at `training=False` reads `moving_mean=0, moving_variance=1` against
activations that have neither, and the resulting blow-up overflows float16 while
merely shrinking the normalised projection in float32. The text tower, which has
no BatchNorm, reads 1.0 in every arm and is the built-in control.

**This row is therefore CLOSED as refuted, and this file exists so it stays
closed.** There is nothing to repair in the model. What there IS, is a test
obligation: any assertion about a BatchNorm model's outputs at `training=False`
on an untrained instance is measuring the initialiser, not the architecture.

See decisions.md D-038 (plan-2026-08-19T163559-499b6f0e).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.mobile_clip.mobile_clip_v1 import MobileClipModel

# ---------------------------------------------------------------------

BN_UPDATES = 60
BATCH = 2
SEQ = 77


def _run(policy: str, bn_updates: int):
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy(policy)
    try:
        keras.utils.set_random_seed(1234)
        model = MobileClipModel.from_variant("b")
        size = model.image_config["image_size"]
        rs = np.random.RandomState(0)
        images = rs.rand(BATCH, size, size, 3).astype("float32")
        text = rs.randint(0, 100, size=(BATCH, SEQ)).astype("int32")

        for _ in range(bn_updates):
            model({"image": rs.rand(BATCH, size, size, 3).astype("float32"),
                   "text": text}, training=True)

        out = model({"image": images, "text": text}, training=False)
        return (np.array(out["image_features"]), np.array(out["text_features"]))
    finally:
        keras.mixed_precision.set_global_policy(previous)


@pytest.fixture(scope="module")
def untrained_fp16():
    return _run("mixed_float16", 0)


@pytest.fixture(scope="module")
def warmed_fp16():
    return _run("mixed_float16", BN_UPDATES)


@pytest.fixture(scope="module")
def warmed_fp32():
    return _run("float32", BN_UPDATES)


def test_the_untrained_fp16_arm_is_all_nan(untrained_fp16):
    """The ARTIFACT, pinned as an artifact.

    This is NOT a defect assertion -- it records the reading that was
    mistaken for one, so that a future reader who sees fp16 NaN from an
    untrained BatchNorm model recognises it instead of "fixing" the model.
    If this ever stops being NaN, the artifact's mechanism has changed and
    batch 4's CRITICAL needs re-opening on the new evidence.
    """
    image_features, _ = untrained_fp16
    assert int(np.sum(np.isnan(image_features))) == image_features.size


def test_the_text_tower_is_the_control_and_is_always_unit_norm(untrained_fp16):
    """The text tower has no BatchNorm; it must be clean in the SAME call."""
    _, text_features = untrained_fp16
    assert int(np.sum(np.isnan(text_features))) == 0
    norms = np.linalg.norm(text_features.astype("float64"), axis=-1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-3)


def test_warming_the_batchnorm_removes_the_fp16_nan(warmed_fp16):
    """The refutation: 60 BatchNorm updates and the NaN is gone."""
    image_features, _ = warmed_fp16
    assert int(np.sum(np.isnan(image_features))) == 0, (
        f"{int(np.sum(np.isnan(image_features)))} NaN remain after "
        f"{BN_UPDATES} BatchNorm updates; the fp16 failure is then NOT the "
        f"untrained-BatchNorm artifact and batch 4's CRITICAL stands"
    )


@pytest.mark.parametrize("arm", ["warmed_fp32", "warmed_fp16"])
def test_the_image_features_are_unit_norm_once_the_batchnorm_is_warm(
        arm, request):
    """The second symptom -- the sub-unit norm -- has the same single cause."""
    image_features, _ = request.getfixturevalue(arm)
    norms = np.linalg.norm(image_features.astype("float64"), axis=-1)
    np.testing.assert_allclose(norms, 1.0, rtol=5e-3, err_msg=(
        f"{arm}: image feature norms {norms.tolist()} are not 1.0 after "
        f"{BN_UPDATES} BatchNorm updates. Untrained, this read 0.007333."
    ))
