"""Tests for dl_techniques.utils.weight_transfer.

Uses CliffordNetUNet as the real-world fixture since it has verified
layer-name stability across head configurations.
"""

import os

import keras
import numpy as np
import pytest

from dl_techniques.models.cliffordnet.unet import CliffordNetUNet
from dl_techniques.utils.weight_transfer import (
    TransferReport,
    load_weights_from_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_backbone_kwargs():
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


def _build_classifier():
    m = CliffordNetUNet(
        head_configs={
            "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 7}
        },
        **_tiny_backbone_kwargs(),
    )
    m.build((None, 32, 32, 3))
    _ = m(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
    return m


def _build_segmenter():
    m = CliffordNetUNet(
        head_configs={
            "seg": {
                "type": "segmentation", "tap": 0, "num_classes": 5, "deep_supervision": True,
            }
        },
        **_tiny_backbone_kwargs(),
    )
    m.build((None, 32, 32, 3))
    _ = m(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
    return m


def _dup_initial_weights(model):
    """Snapshot every layer's weights before transfer so we can diff."""
    return {l.name: [w.copy() for w in l.get_weights()] for l in model.layers}


def _weights_equal(w1, w2):
    if len(w1) != len(w2):
        return False
    return all(np.array_equal(a, b) for a, b in zip(w1, w2))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestTransferHappyPath:
    def test_loads_backbone_skips_heads(self, tmp_path):
        src = _build_classifier()
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt = _build_segmenter()
        tgt_before = _dup_initial_weights(tgt)

        report = load_weights_from_checkpoint(tgt, ckpt)

        assert isinstance(report, TransferReport)
        assert report.num_loaded > 0, "no backbone layers transferred"
        assert report.num_shape_mismatch == 0
        # Every layer marked loaded should now match the source's weights.
        src_layers = {l.name: l for l in src.layers}
        for name in report.loaded:
            after = tgt.get_layer(name).get_weights()
            src_weights = src_layers[name].get_weights()
            assert _weights_equal(after, src_weights), f"layer {name} didn't transfer"

        # A target backbone layer that was loaded should have different weights
        # from its random-init (unless the random init happened to coincide).
        backbone_layers = [n for n in report.loaded if n.startswith(("stem_", "enc_", "bottleneck_", "dec_"))]
        assert backbone_layers, "expected backbone layers in report"
        changed = sum(
            1 for n in backbone_layers
            if not _weights_equal(tgt.get_layer(n).get_weights(), tgt_before[n])
        )
        assert changed > 0, "transfer didn't change any backbone weights"

    def test_head_layers_untouched(self, tmp_path):
        src = _build_classifier()
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt = _build_segmenter()
        tgt_before = _dup_initial_weights(tgt)

        _ = load_weights_from_checkpoint(tgt, ckpt)

        # Target's seg head layers should still have their original init.
        head_layer_names = [
            n for n in tgt_before if n.startswith("head_")
        ]
        assert head_layer_names, "sanity: expected some head layers"
        for n in head_layer_names:
            before = tgt_before[n]
            after = tgt.get_layer(n).get_weights()
            if not before and not after:
                continue
            assert _weights_equal(before, after), f"head layer {n} was modified"

    def test_report_summary_string(self, tmp_path):
        src = _build_classifier()
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt = _build_segmenter()
        report = load_weights_from_checkpoint(tgt, ckpt)
        s = report.summary_string()
        assert "TransferReport:" in s
        assert "loaded" in s
        assert "skipped_by_prefix" in s


# ---------------------------------------------------------------------------
# Skip prefixes
# ---------------------------------------------------------------------------


class TestSkipPrefixes:
    def test_empty_skip_prefixes_hits_source_head_as_unused(self, tmp_path):
        src = _build_classifier()
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt = _build_segmenter()
        report = load_weights_from_checkpoint(tgt, ckpt, skip_prefixes=())
        # No skip prefixes → no skipped_by_prefix entries.
        assert report.skipped_by_prefix == []
        # Source classification head layer ("head_cls") doesn't exist in the
        # segmentation target, so it lands in unused_in_source.
        assert "head_cls" in report.unused_in_source

    def test_default_skip_prefixes_hide_heads(self, tmp_path):
        src = _build_classifier()
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt = _build_segmenter()
        report = load_weights_from_checkpoint(tgt, ckpt)
        # With default skip_prefixes, any source head layers with weights are
        # added to skipped_by_prefix.
        assert any(n.startswith("head_") for n in report.skipped_by_prefix)


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_strict_raises_on_shape_mismatch(self, tmp_path):
        # Build two models with DIFFERENT channel counts — their shared
        # backbone layer names will have mismatched weight shapes.
        src_kwargs = _tiny_backbone_kwargs()
        src_kwargs["level_channels"] = [8, 16, 32]
        src = CliffordNetUNet(
            head_configs={
                "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 7}
            },
            **src_kwargs,
        )
        src.build((None, 32, 32, 3))
        _ = src(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt_kwargs = _tiny_backbone_kwargs()
        tgt_kwargs["level_channels"] = [16, 32, 64]  # different!
        tgt = CliffordNetUNet(
            head_configs={
                "seg": {"type": "segmentation", "tap": 0, "num_classes": 5}
            },
            **tgt_kwargs,
        )
        tgt.build((None, 32, 32, 3))
        _ = tgt(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)

        with pytest.raises(ValueError, match="[Ss]hape mismatch"):
            load_weights_from_checkpoint(tgt, ckpt, strict=True)

    def test_non_strict_records_shape_mismatch(self, tmp_path):
        src_kwargs = _tiny_backbone_kwargs()
        src_kwargs["level_channels"] = [8, 16, 32]
        src = CliffordNetUNet(
            head_configs={
                "cls": {"type": "classification", "tap": "bottleneck", "num_classes": 7}
            },
            **src_kwargs,
        )
        src.build((None, 32, 32, 3))
        _ = src(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        tgt_kwargs = _tiny_backbone_kwargs()
        tgt_kwargs["level_channels"] = [16, 32, 64]
        tgt = CliffordNetUNet(
            head_configs={
                "seg": {"type": "segmentation", "tap": 0, "num_classes": 5}
            },
            **tgt_kwargs,
        )
        tgt.build((None, 32, 32, 3))
        _ = tgt(np.zeros((1, 32, 32, 3), dtype=np.float32), training=False)

        report = load_weights_from_checkpoint(tgt, ckpt, strict=False)
        assert report.num_shape_mismatch > 0
        # Loaded may be empty (all backbone mismatched).
        # But total target backbone layers should equal loaded + mismatch + missing.


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_nonexistent_checkpoint(self):
        tgt = _build_segmenter()
        with pytest.raises(FileNotFoundError):
            load_weights_from_checkpoint(tgt, "/tmp/does_not_exist_xyz.keras")

    def test_non_keras_extension(self, tmp_path):
        tgt = _build_segmenter()
        bogus = os.path.join(str(tmp_path), "weights.h5")
        with open(bogus, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match=".keras"):
            load_weights_from_checkpoint(tgt, bogus)

    def test_unbuilt_target_raises(self, tmp_path):
        src = _build_classifier()
        ckpt = os.path.join(str(tmp_path), "source.keras")
        src.save(ckpt)

        # Build a target but don't call build() / probe.
        tgt_unbuilt = CliffordNetUNet(
            head_configs={"seg": {"type": "segmentation", "tap": 0, "num_classes": 5}},
            **_tiny_backbone_kwargs(),
        )
        with pytest.raises(ValueError, match="built"):
            load_weights_from_checkpoint(tgt_unbuilt, ckpt)
