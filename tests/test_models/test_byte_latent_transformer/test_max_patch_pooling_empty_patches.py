"""`patch_pooling_method='max'` must not emit a `-1e9` sentinel for empty patches.

F-09/F-85 of the 2026-08-18 deep review, fixed under
``plan-2026-08-18T140459-7991552f/D-039``.

`PatchPooling._max_pooling` masks out-of-patch positions with `-1e9` before the
`ops.max`. For an EMPTY patch the mask is all-`False`, so the sentinel survives
the reduction and the slot becomes ``[-1e9] * hidden_dim``, which the output
`Dense` turns into O(1e9) activations. Empty patches are the NORM, not an edge
case: `DynamicPatcher` always emits `max_patches` slots and fills only as many
as there were entropy crossings.

MEASURED at ``create_blt_model("micro", patch_pooling_method="max")`` on a
16-byte sequence (``max_patches=128``): pre-fix **112 of 128** pooled slots had
``|value| > 1e5`` with an absolute maximum of **2.49e9**; post-fix 0 slots and
an absolute maximum of **3.55**. The occupied slots' values are BIT-UNCHANGED by
the fix -- this is not a rescale, it is a replacement of a constant that should
never have been observable.

The model's *logit* magnitude is NOT an instrument for this defect (4.55 legacy
vs 4.22 fixed, same seed): the `GlobalTransformer`'s LayerNorm re-normalizes the
poisoned stream to O(1). The instrument is the pooled patch tensor itself, which
is why every assertion here reads `PatchPooling`'s output rather than the
model's.
"""

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.blt_blocks import PatchPooling


BATCH, SEQ, HID, OUT_DIM, MAX_PATCHES = 2, 6, 8, 16, 8
OCCUPIED, EMPTY = (0, 1), (2, 3, 4, 5, 6, 7)


def _legacy_max_pooling(self, byte_hiddens, patch_ids, num_patches):
    """The pre-fix `_max_pooling` body, verbatim, for RED-proving the guard."""
    reps = []
    for p in range(num_patches):
        mask = ops.equal(patch_ids, p)
        mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)
        reps.append(ops.max(ops.where(mask_expanded, byte_hiddens, -1e9), axis=1))
    result = ops.stack(reps, axis=1)
    if self.output_projection is not None:
        result = self.output_projection(result)
    return result


def _fixture():
    keras.utils.set_random_seed(0)
    pool = PatchPooling(
        pooling_method="max", output_dim=OUT_DIM, max_patches=MAX_PATCHES
    )
    x = np.random.RandomState(0).randn(BATCH, SEQ, HID).astype("float32")
    patch_ids = np.zeros((BATCH, SEQ), dtype="int32")
    patch_ids[:, SEQ // 2:] = 1  # patches 0 and 1 occupied, 2..7 empty
    return pool, ops.convert_to_tensor(x), ops.convert_to_tensor(patch_ids)


class TestMaxPoolingEmptyPatches:
    """The pooled vector for an empty patch is finite and O(1)."""

    def test_empty_patch_slots_are_finite_and_order_one(self):
        pool, x, patch_ids = _fixture()
        pooled = ops.convert_to_numpy(pool(x, patch_ids))

        assert np.isfinite(pooled).all()
        empty = np.abs(pooled[:, EMPTY, :]).max()
        assert empty < 1e3, (
            f"empty-patch slots reached |{empty:.3e}| -- the -1e9 sentinel "
            f"survived the max reduction"
        )

    def test_occupied_patch_slots_are_order_one(self):
        pool, x, patch_ids = _fixture()
        pooled = ops.convert_to_numpy(pool(x, patch_ids))
        assert np.abs(pooled[:, OCCUPIED, :]).max() < 1e3

    def test_guard_is_red_against_the_legacy_body(self, monkeypatch):
        """RED-proof: the same assertion FAILS with the pre-fix reduction."""
        monkeypatch.setattr(
            PatchPooling, "_max_pooling", _legacy_max_pooling, raising=True
        )
        pool, x, patch_ids = _fixture()
        pooled = ops.convert_to_numpy(pool(x, patch_ids))
        assert np.abs(pooled[:, EMPTY, :]).max() > 1e5, (
            "the legacy body no longer reproduces the defect -- this RED-proof "
            "has stopped being an instrument"
        )

    def test_occupied_slots_are_bit_identical_to_the_legacy_body(self, monkeypatch):
        """The fix touches ONLY the empty slots; occupied ones must not move."""
        pool, x, patch_ids = _fixture()
        fixed = ops.convert_to_numpy(pool(x, patch_ids))
        monkeypatch.setattr(
            PatchPooling, "_max_pooling", _legacy_max_pooling, raising=True
        )
        legacy = ops.convert_to_numpy(pool(x, patch_ids))
        np.testing.assert_array_equal(
            fixed[:, OCCUPIED, :], legacy[:, OCCUPIED, :]
        )


class TestMaxPoolingInsideTheAssembledModel:
    """Empty patches dominate a real `micro` forward pass, so this matters."""

    @staticmethod
    def _pooled_from_micro():
        from dl_techniques.models.byte_latent_transformer.model import (
            create_blt_model,
        )

        keras.utils.set_random_seed(0)
        model = create_blt_model(
            variant="micro",
            vocab_size=256,
            max_sequence_length=64,
            patch_pooling_method="max",
        )
        pools = [
            layer
            for layer in model._flatten_layers(include_self=True)
            if isinstance(layer, PatchPooling)
        ]
        assert len(pools) == 1, f"expected exactly one PatchPooling, got {len(pools)}"
        captured = {}
        inner = pools[0].call

        def spy(*args, **kwargs):
            out = inner(*args, **kwargs)
            captured["pooled"] = ops.convert_to_numpy(out)
            return out

        pools[0].call = spy
        tokens = np.random.RandomState(0).randint(0, 256, (2, 16)).astype("int32")
        model(ops.convert_to_tensor(tokens))
        return captured["pooled"]

    def test_micro_pooled_patches_are_order_one(self):
        pooled = self._pooled_from_micro()
        assert pooled.shape[1] == 128
        assert np.isfinite(pooled).all()
        assert np.abs(pooled).max() < 1e3, (
            f"|pooled|max = {np.abs(pooled).max():.3e}; pre-fix this was 2.49e9"
        )

    def test_micro_really_does_leave_most_patch_slots_empty(self, monkeypatch):
        """The premise: empty patches are the norm at `micro`, not the exception.

        Measured with the legacy body, whose poisoned slots are exactly the
        empty ones -- 112 of 128.
        """
        monkeypatch.setattr(
            PatchPooling, "_max_pooling", _legacy_max_pooling, raising=True
        )
        pooled = self._pooled_from_micro()
        poisoned = (np.abs(pooled).max(axis=2) > 1e5).sum(axis=1)
        assert (poisoned > 100).all(), (
            f"expected >100 empty patch slots per row at micro, got {poisoned}"
        )
