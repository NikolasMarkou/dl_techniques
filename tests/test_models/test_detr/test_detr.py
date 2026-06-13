"""
Test suite for the DETR (DEtection TRansformer) model implementation.

Covers: DetrTransformer init, forward pass, serialization;
        DETR model init, forward pass, aux_loss variants, and .keras round-trip.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.detr import DETR, DetrTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
NUM_HEADS = 4
NUM_QUERIES = 5
NUM_CLASSES = 10
NUM_ENC_LAYERS = 2
NUM_DEC_LAYERS = 2
FFN_DIM = 64
BATCH_SIZE = 2
IMG_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_backbone():
    return keras.Sequential([
        keras.layers.Conv2D(HIDDEN_DIM, 3, strides=2, padding="same", activation="relu"),
        keras.layers.Conv2D(HIDDEN_DIM, 3, strides=2, padding="same", activation="relu"),
    ], name="stub_backbone")


def _make_tiny_detr(aux_loss=True):
    transformer = DetrTransformer(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=0.0,
    )
    return DETR(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        backbone=_make_stub_backbone(),
        transformer=transformer,
        hidden_dim=HIDDEN_DIM,
        aux_loss=aux_loss,
    )


def _forward_inputs():
    images = keras.random.normal((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
    mask = keras.ops.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE), dtype="bool")
    return images, mask


# ---------------------------------------------------------------------------
# TestDetrTransformerInit
# ---------------------------------------------------------------------------

class TestDetrTransformerInit:
    """DetrTransformer construction and attribute validation."""

    def test_default_params(self):
        t = DetrTransformer()
        assert t.hidden_dim == 256
        assert t.num_heads == 8
        assert t.num_encoder_layers == 6
        assert t.num_decoder_layers == 6
        assert t.ffn_dim == 2048
        assert t.dropout_rate == 0.1
        assert t.activation == "relu"

    def test_custom_params(self):
        t = DetrTransformer(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_ENC_LAYERS,
            num_decoder_layers=NUM_DEC_LAYERS,
            ffn_dim=FFN_DIM,
            dropout=0.0,
        )
        assert t.hidden_dim == HIDDEN_DIM
        assert t.num_heads == NUM_HEADS
        assert t.num_encoder_layers == NUM_ENC_LAYERS
        assert t.num_decoder_layers == NUM_DEC_LAYERS
        assert t.ffn_dim == FFN_DIM
        assert t.dropout_rate == 0.0

    def test_invalid_hidden_dim_not_divisible(self):
        with pytest.raises(ValueError):
            DetrTransformer(hidden_dim=33, num_heads=4)

    def test_invalid_non_positive_dims(self):
        with pytest.raises(ValueError):
            DetrTransformer(hidden_dim=0, num_heads=4)

    def test_invalid_non_positive_layer_counts(self):
        with pytest.raises(ValueError):
            DetrTransformer(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_encoder_layers=0,
            )


# ---------------------------------------------------------------------------
# TestDetrTransformerForward
# ---------------------------------------------------------------------------

class TestDetrTransformerForward:
    """DetrTransformer forward pass and config round-trip."""

    @pytest.fixture
    def transformer(self):
        return DetrTransformer(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_ENC_LAYERS,
            num_decoder_layers=NUM_DEC_LAYERS,
            ffn_dim=FFN_DIM,
            dropout=0.0,
        )

    @pytest.fixture
    def transformer_inputs(self):
        seq_len = 20
        src = keras.random.normal((BATCH_SIZE, seq_len, HIDDEN_DIM))
        query_embed = keras.random.normal((NUM_QUERIES, HIDDEN_DIM))
        pos_embed = keras.random.normal((BATCH_SIZE, seq_len, HIDDEN_DIM))
        return src, query_embed, pos_embed

    def test_forward_pass_shapes(self, transformer, transformer_inputs):
        src, query_embed, pos_embed = transformer_inputs
        outputs = transformer(src, None, query_embed, pos_embed, training=False)
        assert len(outputs) == NUM_DEC_LAYERS
        for out in outputs:
            assert tuple(out.shape) == (BATCH_SIZE, NUM_QUERIES, HIDDEN_DIM)

    def test_finite_outputs(self, transformer, transformer_inputs):
        src, query_embed, pos_embed = transformer_inputs
        outputs = transformer(src, None, query_embed, pos_embed, training=False)
        for out in outputs:
            arr = np.array(out)
            assert not np.any(np.isnan(arr)), "NaN found in transformer output"
            assert not np.any(np.isinf(arr)), "Inf found in transformer output"

    def test_get_config_roundtrip(self, transformer, transformer_inputs):
        src, query_embed, pos_embed = transformer_inputs
        # Call once to build weights
        transformer(src, None, query_embed, pos_embed, training=False)

        cfg = transformer.get_config()
        restored = DetrTransformer.from_config(cfg)

        assert restored.hidden_dim == transformer.hidden_dim
        assert restored.num_heads == transformer.num_heads
        assert restored.num_encoder_layers == transformer.num_encoder_layers
        assert restored.num_decoder_layers == transformer.num_decoder_layers
        assert restored.ffn_dim == transformer.ffn_dim
        assert restored.dropout_rate == transformer.dropout_rate
        assert restored.activation == transformer.activation


# ---------------------------------------------------------------------------
# TestDetrModelInit
# ---------------------------------------------------------------------------

class TestDetrModelInit:
    """DETR model construction and attribute validation."""

    def test_stores_attrs(self):
        model = _make_tiny_detr(aux_loss=True)
        assert model.num_classes == NUM_CLASSES
        assert model.num_queries == NUM_QUERIES
        assert model.hidden_dim == HIDDEN_DIM
        assert model.aux_loss is True

    def test_stores_aux_loss_false(self):
        model = _make_tiny_detr(aux_loss=False)
        assert model.aux_loss is False

    def test_invalid_params_num_classes(self):
        with pytest.raises(ValueError):
            transformer = DetrTransformer(
                hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS,
                ffn_dim=FFN_DIM, dropout=0.0,
            )
            DETR(
                num_classes=0,
                num_queries=NUM_QUERIES,
                backbone=_make_stub_backbone(),
                transformer=transformer,
                hidden_dim=HIDDEN_DIM,
            )

    def test_invalid_params_num_queries(self):
        with pytest.raises(ValueError):
            transformer = DetrTransformer(
                hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS,
                ffn_dim=FFN_DIM, dropout=0.0,
            )
            DETR(
                num_classes=NUM_CLASSES,
                num_queries=0,
                backbone=_make_stub_backbone(),
                transformer=transformer,
                hidden_dim=HIDDEN_DIM,
            )

    def test_invalid_params_hidden_dim(self):
        with pytest.raises(ValueError):
            transformer = DetrTransformer(
                hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS,
                ffn_dim=FFN_DIM, dropout=0.0,
            )
            DETR(
                num_classes=NUM_CLASSES,
                num_queries=NUM_QUERIES,
                backbone=_make_stub_backbone(),
                transformer=transformer,
                hidden_dim=0,
            )


# ---------------------------------------------------------------------------
# TestDetrModelForward
# ---------------------------------------------------------------------------

class TestDetrModelForward:
    """DETR model forward pass shapes, ranges, and aux_loss behaviour."""

    @pytest.fixture
    def model_aux(self):
        return _make_tiny_detr(aux_loss=True)

    @pytest.fixture
    def model_no_aux(self):
        return _make_tiny_detr(aux_loss=False)

    def test_output_keys(self, model_aux):
        images, mask = _forward_inputs()
        out = model_aux([images, mask], training=False)
        assert "pred_logits" in out
        assert "pred_boxes" in out

    def test_output_shapes(self, model_aux):
        images, mask = _forward_inputs()
        out = model_aux([images, mask], training=False)
        assert tuple(out["pred_logits"].shape) == (BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1)
        assert tuple(out["pred_boxes"].shape) == (BATCH_SIZE, NUM_QUERIES, 4)

    def test_pred_boxes_range(self, model_aux):
        images, mask = _forward_inputs()
        out = model_aux([images, mask], training=False)
        boxes = np.array(out["pred_boxes"])
        assert float(boxes.min()) >= 0.0, "pred_boxes below 0"
        assert float(boxes.max()) <= 1.0, "pred_boxes above 1"

    def test_finite_outputs(self, model_aux):
        images, mask = _forward_inputs()
        out = model_aux([images, mask], training=False)
        for key in ("pred_logits", "pred_boxes"):
            arr = np.array(out[key])
            assert not np.any(np.isnan(arr)), f"NaN in {key}"
            assert not np.any(np.isinf(arr)), f"Inf in {key}"

    def test_aux_loss_true(self, model_aux):
        images, mask = _forward_inputs()
        out = model_aux([images, mask], training=False)
        assert "aux_outputs" in out
        assert len(out["aux_outputs"]) == NUM_DEC_LAYERS - 1

    def test_aux_loss_false(self, model_no_aux):
        images, mask = _forward_inputs()
        out = model_no_aux([images, mask], training=False)
        assert "aux_outputs" not in out
        assert set(out.keys()) == {"pred_logits", "pred_boxes"}


# ---------------------------------------------------------------------------
# TestDetrSerialization
# ---------------------------------------------------------------------------

class TestDetrSerialization:
    """.keras save/load round-trip and get_config checks."""

    @pytest.fixture
    def built_model(self):
        model = _make_tiny_detr(aux_loss=True)
        images, mask = _forward_inputs()
        # Trigger build
        model([images, mask], training=False)
        return model

    def test_keras_roundtrip(self, built_model):
        images, mask = _forward_inputs()
        out_before = built_model([images, mask], training=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "detr_test.keras")
            built_model.save(save_path)
            loaded = keras.models.load_model(save_path)
            out_after = loaded([images, mask], training=False)

        np.testing.assert_allclose(
            np.array(out_before["pred_logits"]),
            np.array(out_after["pred_logits"]),
            atol=1e-4,
            err_msg="pred_logits differ after .keras round-trip",
        )
        np.testing.assert_allclose(
            np.array(out_before["pred_boxes"]),
            np.array(out_after["pred_boxes"]),
            atol=1e-4,
            err_msg="pred_boxes differ after .keras round-trip",
        )

    def test_get_config_has_keys(self, built_model):
        cfg = built_model.get_config()
        for key in ("num_classes", "num_queries", "hidden_dim", "aux_loss",
                    "backbone", "transformer"):
            assert key in cfg, f"Missing key '{key}' in get_config()"
