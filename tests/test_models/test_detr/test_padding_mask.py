"""RED proof for C-6: DETR must honour the padding mask it accepts.

`DETR.call` unpacked `(images, padding_mask)` and then passed `None` into the
transformer, so every token in the zero padding of a letterboxed image took
full part in encoder self-attention and decoder cross-attention. `README.md`'s
worked "preprocess a variable-size image" recipe builds exactly such a mask and
calls `model([image_batch, mask_batch])` as though it were honoured.

Two levels are asserted, deliberately:

* **Exactly**, at the transformer: perturbing `src` at masked positions must
  not move the decoder outputs at all. This is the claim the fix actually makes.
* **At the model**, on the INTERIOR of the padded region only. A convolutional
  backbone leaks padding content into every feature cell within one receptive
  field of the boundary, so invariance to *all* padding pixels is not
  achievable (and is not achieved by the reference implementation either).
  Perturbing only pixels a full receptive field clear of the boundary makes the
  masked model's expected delta exactly zero, with an all-valid control arm
  proving the perturbation is visible at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import keras

from dl_techniques.models.detr import DETR, DetrTransformer

HIDDEN_DIM = 32
NUM_HEADS = 4
NUM_QUERIES = 5
NUM_CLASSES = 10
BATCH_SIZE = 2
IMG_SIZE = 64
SEQ_LEN = 16


def _transformer() -> DetrTransformer:
    return DetrTransformer(
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
        num_encoder_layers=2, num_decoder_layers=2, ffn_dim=64, dropout=0.0,
    )


def _stub_backbone() -> keras.Sequential:
    return keras.Sequential([
        keras.layers.Conv2D(HIDDEN_DIM, 3, strides=2, padding="same",
                            activation="relu"),
        keras.layers.Conv2D(HIDDEN_DIM, 3, strides=2, padding="same",
                            activation="relu"),
    ], name="stub_backbone")


def _detr() -> DETR:
    return DETR(
        num_classes=NUM_CLASSES, num_queries=NUM_QUERIES,
        backbone=_stub_backbone(), transformer=_transformer(),
        hidden_dim=HIDDEN_DIM, aux_loss=False,
    )


class TestTransformerHonoursTheKeyMask:
    """Exact invariance, at the level where the fix lives."""

    def _run(self, transformer, src, mask):
        rng = np.random.default_rng(0)
        query_embed = keras.ops.convert_to_tensor(
            rng.normal(size=(NUM_QUERIES, HIDDEN_DIM)).astype("float32"))
        pos_embed = keras.ops.convert_to_tensor(
            rng.normal(size=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)).astype("float32"))
        out = transformer(src, mask, query_embed, pos_embed, training=False)
        return np.asarray(out[-1])

    def test_masked_positions_cannot_move_the_output(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)).astype("float32")
        keep = np.ones((BATCH_SIZE, SEQ_LEN), dtype="float32")
        keep[:, SEQ_LEN // 2:] = 0.0  # second half is "padding"

        transformer = _transformer()
        base = self._run(transformer, keras.ops.convert_to_tensor(src),
                         keras.ops.convert_to_tensor(keep))

        perturbed = src.copy()
        perturbed[:, SEQ_LEN // 2:, :] += 50.0
        after = self._run(transformer, keras.ops.convert_to_tensor(perturbed),
                          keras.ops.convert_to_tensor(keep))

        # MEASURED: the fully-masked path gives exactly 0.0 here. The tolerance
        # is 1e-7 rather than the repo-usual 1e-5 on purpose — with the ENCODER
        # mask alone removed (decoder still masked) this delta is 4.77e-06, so
        # a 1e-5 tolerance would leave the encoder half of the fix untested at
        # this level.
        np.testing.assert_allclose(after, base, atol=1e-7, rtol=0.0)

    def test_unmasked_positions_do_move_the_output(self):
        """Anti-vacuity: the same perturbation on a KEPT position must move it,
        otherwise the assertion above is satisfied by a dead transformer."""
        rng = np.random.default_rng(1)
        src = rng.normal(size=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)).astype("float32")
        keep = np.ones((BATCH_SIZE, SEQ_LEN), dtype="float32")
        keep[:, SEQ_LEN // 2:] = 0.0

        transformer = _transformer()
        base = self._run(transformer, keras.ops.convert_to_tensor(src),
                         keras.ops.convert_to_tensor(keep))

        perturbed = src.copy()
        perturbed[:, 0, :] += 50.0
        after = self._run(transformer, keras.ops.convert_to_tensor(perturbed),
                          keras.ops.convert_to_tensor(keep))

        assert float(np.max(np.abs(after - base))) > 1e-3

    def test_a_none_mask_still_attends_to_everything(self):
        """The mask is optional; `None` must keep the old behaviour."""
        rng = np.random.default_rng(2)
        src = rng.normal(size=(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)).astype("float32")
        transformer = _transformer()
        base = self._run(transformer, keras.ops.convert_to_tensor(src), None)

        perturbed = src.copy()
        perturbed[:, SEQ_LEN // 2:, :] += 50.0
        after = self._run(transformer, keras.ops.convert_to_tensor(perturbed),
                          None)
        assert float(np.max(np.abs(after - base))) > 1e-3


class TestModelPropagatesThePaddingMask:
    def test_padding_content_matters_far_less_when_it_is_masked(self):
        rng = np.random.default_rng(3)
        images = rng.random((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
        pad_from = IMG_SIZE // 2

        true_mask = np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE), dtype=bool)
        true_mask[:, pad_from:, :] = True
        all_valid = np.zeros_like(true_mask)

        # Perturb only the INTERIOR of the padded region, 16 px clear of the
        # boundary. The stub backbone is two stride-2 3x3 convs, receptive
        # field 11 px, so no VALID feature cell can see these pixels — which
        # makes the masked arm's expected delta exactly zero rather than
        # merely small.
        noisy = images.copy()
        start = pad_from + 16
        noisy[:, start:, :, :] = rng.random(
            (BATCH_SIZE, IMG_SIZE - start, IMG_SIZE, 3)).astype("float32") * 5.0

        model = _detr()

        def boxes(x, m):
            return np.asarray(model([x, m], training=False)["pred_boxes"])

        masked_delta = float(np.max(np.abs(
            boxes(noisy, true_mask) - boxes(images, true_mask))))
        unmasked_delta = float(np.max(np.abs(
            boxes(noisy, all_valid) - boxes(images, all_valid))))

        assert unmasked_delta > 1e-4, (
            "control arm is vacuous: the padding content did not move the "
            "unmasked model either"
        )
        assert masked_delta < 1e-6, (
            f"padding content reached the output through attention: "
            f"masked={masked_delta:.3e} (unmasked control {unmasked_delta:.3e})"
        )

    def test_the_mask_reaches_the_transformer_at_feature_resolution(self):
        """The mask is (B, H, W) at IMAGE resolution and the sequence is at
        feature resolution; pin the downsample by recording what the
        transformer is handed."""
        seen = {}
        model = _detr()
        original = model.transformer.call

        def spy(src, mask, *args, **kwargs):
            seen['mask'] = mask
            seen['src'] = src
            return original(src, mask, *args, **kwargs)

        model.transformer.call = spy
        mask = np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE), dtype=bool)
        mask[:, IMG_SIZE // 2:, :] = True
        model([np.random.default_rng(4).random(
            (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)).astype("float32"), mask],
            training=False)
        model.transformer.call = original

        assert seen['mask'] is not None, "the padding mask was discarded"
        seq_len = int(seen['src'].shape[1])
        assert tuple(seen['mask'].shape) == (BATCH_SIZE, seq_len)
        kept = np.asarray(seen['mask'])
        # The stub backbone is stride 4, the bottom half is padding: exactly
        # half of the feature positions must be dropped.
        assert np.isclose(kept.mean(), 0.5), kept.mean()


class TestBoxHeadDepth:
    def test_box_head_is_the_paper_s_three_layer_mlp(self):
        model = _detr()
        dense = [l for l in model.bbox_embed.layers
                 if isinstance(l, keras.layers.Dense)]
        assert len(dense) == 3, [l.name for l in model.bbox_embed.layers]
        assert dense[-1].units == 4
