"""Tests for the masked-patch tf.data transform (Energy Transformer MIM).

Every numerical test runs at the REALISTIC size N=196 (image 224, patch 16,
P=768). A toy N is banned here: it hides exactly the class of bug this suite
exists to catch (rounding collapsing the 90/10 split, an off-by-one in the
patch grid).

Covers success criteria C2 (the 90/10 masking rule + the loss-weight scale) and
C6 (I7: patch-order agreement between ``patchify_targets`` and the model's
``PatchEmbedding2D``).
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.vision.masked_patches import (
    make_masked_patch_map_fn,
    patchify_targets,
)
from dl_techniques.layers.embedding.factory import create_embedding_layer

# Realistic ViT-Base/16 @224 geometry. Do not shrink these.
IMAGE_SIZE = 224
PATCH_SIZE = 16
CHANNELS = 3
GRID = IMAGE_SIZE // PATCH_SIZE          # 14
NUM_PATCHES = GRID * GRID                # 196
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS  # 768

MASK_RATIO = 0.5
MASK_TOKEN_FRAC = 0.9
N_LOSS = int(round(MASK_RATIO * NUM_PATCHES))          # 98
N_INPUT = int(round(MASK_TOKEN_FRAC * N_LOSS))         # 88


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260714)


@pytest.fixture(scope="module")
def sample_image(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")


def _sampled_batches(num_batches: int = 64, batch_size: int = 4, seed: int = 7):
    """Run the map fn over `num_batches` batches and return the stacked outputs."""
    images = np.zeros(
        (num_batches * batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype="float32"
    )
    map_fn = make_masked_patch_map_fn(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        mask_ratio=MASK_RATIO,
        mask_token_frac=MASK_TOKEN_FRAC,
        seed=seed,
    )
    ds = (
        tf.data.Dataset.from_tensor_slices(images)
        .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    masks, weights = [], []
    for (_img, input_mask), _targets, loss_weight in ds:
        masks.append(input_mask.numpy())
        weights.append(loss_weight.numpy())
    return np.concatenate(masks, axis=0), np.concatenate(weights, axis=0)


class TestMaskingRule:
    """C2 / I2: the paper's 90/10 rule and the loss-weight scale."""

    @pytest.fixture(scope="class")
    def sampled(self):
        return _sampled_batches()

    def test_shapes_and_dtypes(self, sampled):
        input_mask, loss_weight = sampled
        assert input_mask.shape == (256, NUM_PATCHES)
        assert loss_weight.shape == (256, NUM_PATCHES)
        assert input_mask.dtype == np.bool_
        assert loss_weight.dtype == np.float32

    def test_element_signature(self):
        """The frozen contract: ((image, mask), targets, weights)."""
        map_fn = make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, seed=0)
        ds = tf.data.Dataset.from_tensor_slices(
            np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "float32")
        ).map(map_fn).batch(2)
        (image, input_mask), targets, loss_weight = next(iter(ds))
        assert image.shape == (2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        assert image.dtype == tf.float32
        assert input_mask.shape == (2, NUM_PATCHES)
        assert input_mask.dtype == tf.bool
        assert targets.shape == (2, NUM_PATCHES, PATCH_DIM)
        assert targets.dtype == tf.float32
        assert loss_weight.shape == (2, NUM_PATCHES)
        assert loss_weight.dtype == tf.float32

    def test_exact_counts(self, sampled):
        """|S| == 98 and |input_mask| == 88 on every one of the 256 samples."""
        input_mask, loss_weight = sampled
        loss_set = loss_weight > 0.0
        assert np.all(loss_set.sum(axis=1) == N_LOSS)
        assert np.all(input_mask.sum(axis=1) == N_INPUT)

    def test_input_mask_is_strict_subset_of_loss_set(self, sampled):
        """I2: input_mask ⊊ loss_set. The ~10% un-occluded loss tokens must exist."""
        input_mask, loss_weight = sampled
        loss_set = loss_weight > 0.0
        # subset: no token is masked-in-input without being in the loss set
        assert not np.any(input_mask & ~loss_set)
        # STRICT: every sample keeps n_loss - n_input un-occluded loss tokens
        unoccluded = (loss_set & ~input_mask).sum(axis=1)
        assert np.all(unoccluded == N_LOSS - N_INPUT)
        assert N_LOSS - N_INPUT == 10

    def test_loss_weight_sums_to_N(self, sampled):
        """I3: Σ_i loss_weight_i == N per sample (fp32 -> atol 1e-3)."""
        _input_mask, loss_weight = sampled
        np.testing.assert_allclose(
            loss_weight.sum(axis=1), float(NUM_PATCHES), atol=1e-3
        )

    def test_loss_weight_is_zero_off_the_loss_set(self, sampled):
        """Exactly 0.0 off S (not merely small) and exactly N/n_loss on S."""
        _input_mask, loss_weight = sampled
        nonzero = loss_weight[loss_weight != 0.0]
        assert np.all(loss_weight[loss_weight == 0.0] == 0.0)
        np.testing.assert_allclose(
            nonzero, np.float32(NUM_PATCHES / N_LOSS), rtol=0, atol=0
        )

    def test_masks_differ_across_samples(self, sampled):
        """The selection is per-sample, not one mask broadcast over the batch."""
        input_mask, _loss_weight = sampled
        assert len({m.tobytes() for m in input_mask}) > 200


class TestPatchOrder:
    """C6 / I7: patchify_targets must agree with PatchEmbedding2D token-for-token.

    An identity-kernel Conv2D turns PatchEmbedding2D into a pure patchifier, so
    its output must be elementwise equal to patchify_targets. A mismatch means
    the decoder would be trained against spatially scrambled targets — a failure
    that still shows a descending loss curve.
    """

    @staticmethod
    def _identity_patch_embedding():
        layer = create_embedding_layer(
            "patch_2d",
            patch_size=PATCH_SIZE,
            embed_dim=PATCH_DIM,
            use_bias=True,
        )
        layer.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        # Conv2D kernel is (patch_h, patch_w, C, filters) and contracts over
        # (h, w, c). eye(P).reshape(patch, patch, C, P) sets
        # kernel[h, w, c, d] = 1 iff d == (h * patch + w) * C + c, i.e. the
        # output channel d reads patch element d under an HWC row-major flatten.
        kernel = np.eye(PATCH_DIM, dtype="float32").reshape(
            PATCH_SIZE, PATCH_SIZE, CHANNELS, PATCH_DIM
        )
        bias = np.zeros((PATCH_DIM,), dtype="float32")
        layer.proj.kernel.assign(kernel)
        layer.proj.bias.assign(bias)
        return layer

    def test_identity_kernel_reproduces_patchify_targets(self, sample_image):
        layer = self._identity_patch_embedding()
        batch = sample_image[None, ...]

        embedded = np.asarray(layer(batch))            # (1, 196, 768)
        targets = patchify_targets(batch, PATCH_SIZE).numpy()  # (1, 196, 768)

        assert embedded.shape == (1, NUM_PATCHES, PATCH_DIM)
        assert targets.shape == (1, NUM_PATCHES, PATCH_DIM)
        np.testing.assert_allclose(embedded, targets, atol=1e-5)

    def test_token_i_is_the_row_major_grid_cell_i(self, sample_image):
        """Independent check of token order against a plain numpy slice."""
        targets = patchify_targets(sample_image, PATCH_SIZE).numpy()  # (196, 768)
        assert targets.shape == (NUM_PATCHES, PATCH_DIM)
        for token in (0, 1, GRID, GRID + 1, NUM_PATCHES - 1):
            gh, gw = divmod(token, GRID)
            expected = sample_image[
                gh * PATCH_SIZE:(gh + 1) * PATCH_SIZE,
                gw * PATCH_SIZE:(gw + 1) * PATCH_SIZE,
                :,
            ].reshape(-1)  # numpy C-order == HWC row-major
            np.testing.assert_allclose(targets[token], expected, atol=0)

    def test_unbatched_and_batched_agree(self, sample_image):
        single = patchify_targets(sample_image, PATCH_SIZE).numpy()
        batched = patchify_targets(sample_image[None, ...], PATCH_SIZE).numpy()
        np.testing.assert_allclose(single, batched[0], atol=0)


class TestConstructionErrors:
    """Config errors must be raised EAGERLY, at pipeline-build time."""

    def test_non_divisible_image_size(self):
        with pytest.raises(ValueError, match="divisible"):
            make_masked_patch_map_fn(patch_size=16, image_size=225)

    def test_n_loss_zero(self):
        # 196 patches, ratio 0.002 -> round(0.392) = 0
        with pytest.raises(ValueError, match="n_loss=0"):
            make_masked_patch_map_fn(
                patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, mask_ratio=0.002
            )

    def test_n_input_equals_n_loss(self):
        """The rounding edge that silently deletes the 10% un-occluded signal."""
        # 4 patches (image 32, patch 16), ratio 0.5 -> n_loss=2; frac 0.9 -> round(1.8)=2
        with pytest.raises(ValueError, match="n_input == n_loss"):
            make_masked_patch_map_fn(
                patch_size=16,
                image_size=32,
                mask_ratio=0.5,
                mask_token_frac=MASK_TOKEN_FRAC,
            )

    def test_n_input_zero(self):
        with pytest.raises(ValueError, match="n_input=0"):
            make_masked_patch_map_fn(
                patch_size=PATCH_SIZE,
                image_size=IMAGE_SIZE,
                mask_ratio=MASK_RATIO,
                mask_token_frac=0.001,
            )

    def test_bad_ratios(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, mask_ratio=0.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, mask_ratio=1.5)
        with pytest.raises(ValueError, match="mask_token_frac"):
            make_masked_patch_map_fn(PATCH_SIZE, IMAGE_SIZE, mask_token_frac=1.5)

    def test_patchify_rejects_bad_rank_and_size(self, sample_image):
        with pytest.raises(ValueError, match="rank"):
            patchify_targets(np.zeros((10, 10), "float32"), PATCH_SIZE)
        with pytest.raises(ValueError, match="divisible"):
            patchify_targets(np.zeros((17, 17, 3), "float32"), PATCH_SIZE)
        with pytest.raises(ValueError, match="patch_size"):
            patchify_targets(sample_image, 0)
