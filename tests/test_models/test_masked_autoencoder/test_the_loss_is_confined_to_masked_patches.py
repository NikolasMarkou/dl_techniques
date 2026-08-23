"""The MAE reconstruction loss must be computed on MASKED patches ONLY.

MAE's whole training signal rests on one line of `MaskedAutoencoder.compute_loss`::

    loss = loss * mask_img

Without it the model is trained as a plain autoencoder that can see its own
target, and the masked-prediction task it exists to perform is gone.

MEASURED 2026-08-21 (CPU, ``CUDA_VISIBLE_DEVICES=1``, TF sees no GPU for this
module -- the probe is pure ``keras.ops`` arithmetic and device-independent):
replacing line 394 with ``loss = loss`` left the whole
``tests/test_models/test_masked_autoencoder/`` directory GREEN at **14 passed**.
The term was deletable undetected. This module is the guard that convicts it.

Shape of the probe -- three arms, because a one-armed "unmasked pixels do not
move the loss" assertion is satisfied by a probe that pokes a pixel the loss
never reads for any other reason:

1. perturb the reconstruction inside an UNMASKED patch -> the loss must be
   **exactly** unchanged (delta 0.0);
2. perturb inside a MASKED patch -> the loss must move;
3. negative control: the identical arm-1 perturbation on a model built with
   ``non_mask_value=1.0`` (every patch weighted) MUST move the loss, proving the
   arm-1 location is one the loss can see at all.

The mask/patch geometry is pinned rather than inferred: at 32x32 with
``patch_size=16`` there are 4 patches, ``_reshape_mask_for_loss`` reshapes the
flat mask to a (2, 2) grid in row-major order and nearest-neighbour upsamples,
so patch index ``p`` covers rows ``16*(p//2)`` and columns ``16*(p%2)``.
``PATCH_ROWCOL`` states that and ``test_the_patch_geometry_this_module_assumes``
re-derives it from the model instead of trusting the comment.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.masked_autoencoder.mae import MaskedAutoencoder

from .conftest import tiny_encoder

BATCH, IMAGE_SIZE, PATCH_SIZE, CHANNELS = 1, 32, 16, 3
GRID = IMAGE_SIZE // PATCH_SIZE          # 2
NUM_PATCHES = GRID * GRID                # 4
MASKED_PATCH, UNMASKED_PATCH = 0, 1
SEED = 20260821


def PATCH_ROWCOL(p):
    """Pixel slice of flat patch index `p` under `_reshape_mask_for_loss`."""
    r, c = (p // GRID) * PATCH_SIZE, (p % GRID) * PATCH_SIZE
    return slice(r, r + PATCH_SIZE), slice(c, c + PATCH_SIZE)


def _tiny_encoder():
    """Smallest 16x-downsampling encoder MAE's decoder_depth=4 contract accepts."""
    return tiny_encoder(image_size=IMAGE_SIZE, channels=CHANNELS)


def _mae(non_mask_value=0.0):
    keras.utils.set_random_seed(SEED)
    return MaskedAutoencoder(
        encoder=_tiny_encoder(),
        patch_size=PATCH_SIZE,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        non_mask_value=non_mask_value,
    )


def _fixture():
    """A target, a perfect reconstruction, and a mask with patch 0 masked only."""
    rng = np.random.default_rng(SEED)
    x = rng.random((BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")
    mask = np.zeros((BATCH, NUM_PATCHES), dtype="float32")
    mask[:, MASKED_PATCH] = 1.0
    return x, mask


def _loss(model, x, recon, mask):
    y_pred = {
        "reconstruction": keras.ops.convert_to_tensor(recon),
        "mask": keras.ops.convert_to_tensor(mask),
    }
    return float(keras.ops.convert_to_numpy(
        model.compute_loss(x=keras.ops.convert_to_tensor(x), y_pred=y_pred)
    ))


def _perturbed(x, patch):
    recon = x.copy()
    rs, cs = PATCH_ROWCOL(patch)
    recon[:, rs, cs, :] += 1.0
    return recon


class TestTheLossIsConfinedToMaskedPatches:

    def test_the_patch_geometry_this_module_assumes(self):
        """Re-derive the flat-index -> pixel-block map from the model itself."""
        model = _mae()
        target = keras.ops.zeros((BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        for p in range(NUM_PATCHES):
            mask = np.zeros((BATCH, NUM_PATCHES), dtype="float32")
            mask[:, p] = 1.0
            img = keras.ops.convert_to_numpy(
                model._reshape_mask_for_loss(keras.ops.convert_to_tensor(mask), target)
            )
            expected = np.zeros((BATCH, IMAGE_SIZE, IMAGE_SIZE), dtype="float32")
            rs, cs = PATCH_ROWCOL(p)
            expected[:, rs, cs] = 1.0
            np.testing.assert_array_equal(img, expected, err_msg=f"patch {p}")

    def test_an_unmasked_perturbation_does_not_move_the_loss(self):
        model = _mae()
        x, mask = _fixture()
        base = _loss(model, x, x.copy(), mask)
        moved = _loss(model, x, _perturbed(x, UNMASKED_PATCH), mask)
        assert moved == base == pytest.approx(0.0, abs=1e-12), (
            f"unmasked pixels reached the loss: {base} -> {moved}"
        )

    def test_a_masked_perturbation_does_move_the_loss(self):
        """The twin of the above: without it, a loss stuck at 0 would pass."""
        model = _mae()
        x, mask = _fixture()
        base = _loss(model, x, x.copy(), mask)
        moved = _loss(model, x, _perturbed(x, MASKED_PATCH), mask)
        assert moved > base + 1e-3, f"masked pixels never reached the loss: {moved}"

    def test_the_negative_control_sees_the_unmasked_location(self):
        """`non_mask_value=1.0` weights every patch, so arm 1's poke MUST show."""
        model = _mae(non_mask_value=1.0)
        x, mask = _fixture()
        base = _loss(model, x, x.copy(), mask)
        moved = _loss(model, x, _perturbed(x, UNMASKED_PATCH), mask)
        assert moved > base + 1e-3, (
            "the arm-1 perturbation is invisible even when every patch is "
            f"weighted, so arm 1 proves nothing: {base} -> {moved}"
        )
