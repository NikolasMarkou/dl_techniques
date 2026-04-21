"""Tests for CliffordNetUNet and create_cliffordnet_depth factory."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.cliffordnet.unet import (
    CliffordNetUNet,
    _ClassificationHeadBlock,
    _SpatialHeadBlock,
    create_cliffordnet_depth,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _tiny_backbone_kwargs():
    """Minimal 3-level backbone used across tests (fast to build)."""
    return dict(
        in_channels=3,
        level_channels=[8, 16, 32],
        level_blocks=[1, 1, 1],
        level_shifts=[[1], [1, 2], [1, 2]],
        cli_mode="full",
        ctx_mode="diff",
        use_global_context=False,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
    )


def _dummy_image(batch: int = 2, h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed=123)
    return rng.standard_normal((batch, h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Head block unit tests
# ---------------------------------------------------------------------------


class TestClassificationHeadBlock:
    def test_forward_shape(self):
        head = _ClassificationHeadBlock(num_classes=10, dropout=0.1, hidden_dim=32)
        x = np.random.randn(2, 4, 4, 16).astype(np.float32)
        y = head(x, training=False)
        assert tuple(y.shape) == (2, 10)

    def test_no_hidden(self):
        head = _ClassificationHeadBlock(num_classes=5)
        x = np.random.randn(2, 4, 4, 16).astype(np.float32)
        y = head(x, training=False)
        assert tuple(y.shape) == (2, 5)

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError):
            _ClassificationHeadBlock(num_classes=0)


class TestSpatialHeadBlock:
    def test_forward_shape(self):
        head = _SpatialHeadBlock(out_channels=1)
        x = np.random.randn(2, 8, 8, 16).astype(np.float32)
        y = head(x, training=False)
        assert tuple(y.shape) == (2, 8, 8, 1)

    def test_with_hidden(self):
        head = _SpatialHeadBlock(out_channels=81, hidden_dim=32)
        x = np.random.randn(2, 8, 8, 16).astype(np.float32)
        y = head(x, training=False)
        assert tuple(y.shape) == (2, 8, 8, 81)


# ---------------------------------------------------------------------------
# CliffordNetUNet — head-less (backbone identity) mode
# ---------------------------------------------------------------------------


class TestHeadlessUNet:
    def test_forward_dict_keys(self):
        model = CliffordNetUNet(**_tiny_backbone_kwargs())
        x = _dummy_image()
        out = model(x, training=False)
        assert isinstance(out, dict)
        expected = {"level_0", "level_1", "level_2", "bottleneck"}
        assert set(out.keys()) == expected

    def test_level_spatial_resolutions(self):
        model = CliffordNetUNet(**_tiny_backbone_kwargs())
        x = _dummy_image(batch=1, h=32, w=32)
        out = model(x, training=False)
        # Level 0 full res, level 1 half, level 2 quarter
        assert tuple(out["level_0"].shape) == (1, 32, 32, 8)
        assert tuple(out["level_1"].shape) == (1, 16, 16, 16)
        assert tuple(out["level_2"].shape) == (1, 8, 8, 32)
        assert tuple(out["bottleneck"].shape) == (1, 8, 8, 32)


# ---------------------------------------------------------------------------
# CliffordNetUNet — classification-only
# ---------------------------------------------------------------------------


class TestClassificationUNet:
    def test_forward_shape(self):
        head_configs = {
            "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 80}
        }
        model = CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())
        x = _dummy_image(batch=2, h=32, w=32)
        out = model(x, training=False)
        assert set(out.keys()) == {"cls"}
        assert tuple(out["cls"].shape) == (2, 80)

    def test_deep_supervision_at_bottleneck_rejected(self):
        head_configs = {
            "cls": {
                "type": "classification",
                "tap": "bottleneck",
                "num_classes": 80,
                "deep_supervision": True,
            }
        }
        with pytest.raises(ValueError, match="deep_supervision"):
            CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())


# ---------------------------------------------------------------------------
# CliffordNetUNet — segmentation with deep supervision
# ---------------------------------------------------------------------------


class TestSegmentationUNet:
    def test_forward_shapes_with_deep_supervision(self):
        head_configs = {
            "seg": {
                "type": "segmentation",
                "tap": 0,
                "num_classes": 81,
                "deep_supervision": True,
            }
        }
        model = CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())
        x = _dummy_image(batch=1, h=32, w=32)
        out = model(x, training=False)
        assert set(out.keys()) == {"seg", "seg_aux_1", "seg_aux_2"}
        assert tuple(out["seg"].shape) == (1, 32, 32, 81)
        assert tuple(out["seg_aux_1"].shape) == (1, 16, 16, 81)
        assert tuple(out["seg_aux_2"].shape) == (1, 8, 8, 81)

    def test_no_deep_supervision(self):
        head_configs = {
            "seg": {
                "type": "segmentation",
                "tap": 0,
                "num_classes": 21,
                "deep_supervision": False,
            }
        }
        model = CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())
        out = model(_dummy_image(batch=1, h=32, w=32), training=False)
        assert set(out.keys()) == {"seg"}
        assert tuple(out["seg"].shape) == (1, 32, 32, 21)


# ---------------------------------------------------------------------------
# CliffordNetUNet — multi-task (classification + segmentation)
# ---------------------------------------------------------------------------


class TestMultiTaskUNet:
    def test_combined_forward(self):
        head_configs = {
            "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 80},
            "seg": {
                "type": "segmentation",
                "tap": 0,
                "num_classes": 81,
                "deep_supervision": True,
            },
        }
        model = CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())
        x = _dummy_image(batch=2, h=32, w=32)
        out = model(x, training=False)
        expected = {"cls", "seg", "seg_aux_1", "seg_aux_2"}
        assert set(out.keys()) == expected
        assert tuple(out["cls"].shape) == (2, 80)
        assert tuple(out["seg"].shape) == (2, 32, 32, 81)


# ---------------------------------------------------------------------------
# Depth factory
# ---------------------------------------------------------------------------


class TestCreateCliffordNetDepth:
    def test_tiny_factory_smoke(self):
        model = create_cliffordnet_depth(
            variant="tiny",
            out_channels=1,
            enable_deep_supervision=True,
        )
        x = _dummy_image(batch=1, h=64, w=64)
        out = model(x, training=False)
        # tiny has 3 levels → aux at levels 1, 2
        assert set(out.keys()) == {"depth", "depth_aux_1", "depth_aux_2"}
        assert tuple(out["depth"].shape) == (1, 64, 64, 1)

    def test_no_deep_supervision(self):
        model = create_cliffordnet_depth(
            variant="tiny", enable_deep_supervision=False
        )
        out = model(_dummy_image(batch=1, h=32, w=32), training=False)
        assert set(out.keys()) == {"depth"}


# ---------------------------------------------------------------------------
# Layer-name stability across head configs (for weight transfer)
# ---------------------------------------------------------------------------


class TestLayerNameStability:
    def _backbone_layer_names(self, model: CliffordNetUNet):
        # All non-head layers (filter out head_* prefixed names)
        return sorted(
            w.name
            for w in model.weights
            if not any(h in w.path for h in ("head_", "head_primary_", "head_aux_"))
        )

    def test_same_backbone_names_across_head_configs(self):
        kwargs = _tiny_backbone_kwargs()
        m_none = CliffordNetUNet(**kwargs)
        m_cls = CliffordNetUNet(
            head_configs={
                "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 10}
            },
            **kwargs,
        )
        m_seg = CliffordNetUNet(
            head_configs={
                "seg": {
                    "type": "segmentation",
                    "tap": 0,
                    "num_classes": 5,
                    "deep_supervision": True,
                }
            },
            **kwargs,
        )
        # Build all three
        x = _dummy_image(batch=1, h=32, w=32)
        _ = m_none(x, training=False)
        _ = m_cls(x, training=False)
        _ = m_seg(x, training=False)

        names_none = self._backbone_layer_names(m_none)
        names_cls = self._backbone_layer_names(m_cls)
        names_seg = self._backbone_layer_names(m_seg)
        assert names_none == names_cls == names_seg


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_classification(self, tmp_path):
        head_configs = {
            "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 10}
        }
        model = CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())
        x = _dummy_image(batch=2, h=32, w=32)
        y1 = model(x, training=False)

        path = os.path.join(str(tmp_path), "cls.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        y2 = reloaded(x, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y1["cls"]),
            keras.ops.convert_to_numpy(y2["cls"]),
            atol=1e-5,
        )

    def test_roundtrip_multitask(self, tmp_path):
        head_configs = {
            "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 10},
            "seg": {
                "type": "segmentation",
                "tap": 0,
                "num_classes": 5,
                "deep_supervision": True,
            },
        }
        model = CliffordNetUNet(head_configs=head_configs, **_tiny_backbone_kwargs())
        x = _dummy_image(batch=1, h=32, w=32)
        y1 = model(x, training=False)

        path = os.path.join(str(tmp_path), "multitask.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        y2 = reloaded(x, training=False)

        assert set(y1.keys()) == set(y2.keys())
        for k in y1.keys():
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(y1[k]),
                keras.ops.convert_to_numpy(y2[k]),
                atol=1e-5,
                err_msg=f"Mismatch on key {k}",
            )

    def test_roundtrip_depth_factory(self, tmp_path):
        model = create_cliffordnet_depth(variant="tiny", enable_deep_supervision=True)
        x = _dummy_image(batch=1, h=32, w=32)
        y1 = model(x, training=False)

        path = os.path.join(str(tmp_path), "depth.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        y2 = reloaded(x, training=False)

        for k in y1.keys():
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(y1[k]),
                keras.ops.convert_to_numpy(y2[k]),
                atol=1e-5,
                err_msg=f"Mismatch on key {k}",
            )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_head_type(self):
        with pytest.raises(ValueError, match="type"):
            CliffordNetUNet(
                head_configs={"x": {"type": "bogus", "tap": 0, "out_channels": 1}},
                **_tiny_backbone_kwargs(),
            )

    def test_invalid_tap_out_of_range(self):
        with pytest.raises(ValueError, match="tap"):
            CliffordNetUNet(
                head_configs={"x": {"type": "segmentation", "tap": 99, "out_channels": 1}},
                **_tiny_backbone_kwargs(),
            )

    def test_pooled_missing_num_classes(self):
        with pytest.raises(ValueError, match="num_classes"):
            CliffordNetUNet(
                head_configs={"x": {"type": "classification", "tap": "bottleneck"}},
                **_tiny_backbone_kwargs(),
            )
