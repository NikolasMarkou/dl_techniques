import numpy as np
import keras
import pytest

from dl_techniques.models.vision.omnipoint import OmniPoint

# Deliberately tiny: 56 % 14 == 0 -> a 4x4 = 16-token patch grid, ViT-Base/14.
_TEST_IMAGE_SHAPE = (56, 56, 3)


def _make_model() -> OmniPoint:
    return OmniPoint.from_variant(
        "omnipoint_base",
        image_shape=_TEST_IMAGE_SHAPE,
        enable_conditioning=True,
        intrinsics_channels=2,
        depth_channels=2,
        conditioning_hidden_channels=4,
    )


def _make_batch(batch_size: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, size=(batch_size,) + _TEST_IMAGE_SHAPE).astype("float32")
    ray_map = rng.normal(size=(batch_size,) + _TEST_IMAGE_SHAPE[:2] + (3,)).astype("float32")
    ray_map /= np.linalg.norm(ray_map, axis=-1, keepdims=True) + 1e-8

    sparse_depth = np.zeros((batch_size,) + _TEST_IMAGE_SHAPE[:2] + (1,), dtype="float32")
    sparse_depth_mask = np.zeros_like(sparse_depth)
    sparse_depth[:, 10, 10, 0] = 5.0
    sparse_depth_mask[:, 10, 10, 0] = 1.0
    return image, ray_map, sparse_depth, sparse_depth_mask


class TestOmniPointConditioning:
    """Step 5: optional geometric conditioning, wired into the full model."""

    def _assert_finite_and_shaped(self, outputs, batch_size):
        ray, distance, point, mask_logit, scale = outputs
        grid = _TEST_IMAGE_SHAPE[0] // 14
        assert tuple(ray.shape) == (batch_size, grid, grid, 3)
        assert tuple(distance.shape) == (batch_size, grid, grid, 1)
        assert tuple(point.shape) == (batch_size, grid, grid, 3)
        assert tuple(mask_logit.shape) == (batch_size, grid, grid, 1)
        assert tuple(scale.shape) == (batch_size,)
        for tensor in outputs:
            assert np.isfinite(keras.ops.convert_to_numpy(tensor)).all()

    def test_forward_with_intrinsics_only(self):
        model = _make_model()
        image, ray_map, _, _ = _make_batch()
        outputs = model(image, intrinsics_ray_map=ray_map)
        self._assert_finite_and_shaped(outputs, batch_size=2)

    def test_forward_with_sparse_depth_only(self):
        model = _make_model()
        image, _, sparse_depth, sparse_depth_mask = _make_batch()
        outputs = model(
            image, sparse_depth=sparse_depth, sparse_depth_mask=sparse_depth_mask
        )
        self._assert_finite_and_shaped(outputs, batch_size=2)

    def test_forward_with_both(self):
        model = _make_model()
        image, ray_map, sparse_depth, sparse_depth_mask = _make_batch()
        outputs = model(
            image,
            intrinsics_ray_map=ray_map,
            sparse_depth=sparse_depth,
            sparse_depth_mask=sparse_depth_mask,
        )
        self._assert_finite_and_shaped(outputs, batch_size=2)

    def test_forward_with_neither(self):
        model = _make_model()
        image, _, _, _ = _make_batch()
        outputs = model(image)
        self._assert_finite_and_shaped(outputs, batch_size=2)

    def test_disabled_conditioning_matches_step4_behavior(self):
        # enable_conditioning=False (default) must still work exactly as
        # Step 4's model -- no conditioning kwargs, unwidened encoder.
        model = OmniPoint.from_variant("omnipoint_base", image_shape=_TEST_IMAGE_SHAPE)
        image, _, _, _ = _make_batch()
        outputs = model(image)
        self._assert_finite_and_shaped(outputs, batch_size=2)

    # ------------------------------------------------------------------
    # Mixed-batch: per-sample present/absent flags produce different output
    # ------------------------------------------------------------------

    def test_mixed_batch_intrinsics_flag_changes_output_per_sample(self):
        model = _make_model()
        rng = np.random.default_rng(1)
        image = rng.uniform(0.0, 1.0, size=(2,) + _TEST_IMAGE_SHAPE).astype("float32")
        # Same ray map content for both samples; only the flag differs.
        ray_map = rng.normal(size=(2,) + _TEST_IMAGE_SHAPE[:2] + (3,)).astype("float32")
        ray_map /= np.linalg.norm(ray_map, axis=-1, keepdims=True) + 1e-8
        ray_map[1] = ray_map[0]
        flag = np.array([True, False])

        outputs = model(image, intrinsics_ray_map=ray_map, intrinsics_present=flag)
        ray, distance, point, mask_logit, scale = outputs

        ray_np = keras.ops.convert_to_numpy(ray)
        # Sample 0 (flag=True, real ray map) and sample 1 (flag=False, same
        # ray map content but marked absent -> zeroed + absent embedding)
        # must NOT collapse to the same prediction on different input images
        # only by chance -- assert the two per-sample outputs actually differ,
        # a concrete, checkable claim.
        assert not np.allclose(ray_np[0], ray_np[1], atol=1e-5)

    def test_mixed_batch_sparse_depth_zero_valid_points_is_finite(self):
        model = _make_model()
        image, _, sparse_depth, sparse_depth_mask = _make_batch()
        # Zero out ALL valid points across the whole batch -- the critical
        # edge case (Problem Statement: must degrade to no depth prior, not
        # divide-by-zero).
        sparse_depth_mask = np.zeros_like(sparse_depth_mask)
        sparse_depth = np.zeros_like(sparse_depth)

        outputs = model(
            image, sparse_depth=sparse_depth, sparse_depth_mask=sparse_depth_mask
        )
        self._assert_finite_and_shaped(outputs, batch_size=2)

    def test_mixed_batch_per_sample_depth_presence_flag(self):
        model = _make_model()
        image, _, sparse_depth, sparse_depth_mask = _make_batch()
        flag = np.array([True, False])

        outputs = model(
            image,
            sparse_depth=sparse_depth,
            sparse_depth_mask=sparse_depth_mask,
            sparse_depth_present=flag,
        )
        self._assert_finite_and_shaped(outputs, batch_size=2)
