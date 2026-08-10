"""Tests for dl_techniques.utils.weight_transfer.

Fixture note (plan-2026-08-10-3649c19e/iter-1/step-4, decisions.md D-012):
these tests used to build ``CliffordNetUNet``, which was deleted with the rest
of the cliffordnet denoiser/UNet surface. They now build small *functional*
Keras models with hand-chosen layer names instead. That is not a downgrade:
``load_weights_from_checkpoint`` matches purely on ``layer.name`` and on
``skip_prefixes``, so the only property of the old fixture the tests ever
exercised was its naming convention (``stem_`` / ``enc_`` / ``bottleneck_`` /
``dec_`` backbone, ``head_<task>`` heads). Naming those layers directly makes
the contract under test explicit and removes a 3-second model build per test.

Do NOT re-point these at a real architecture to make them "realistic" — a real
model reintroduces the exact coupling (a utility test that dies when an
unrelated model is deleted) this rewrite removed.
"""

import os

import keras
import numpy as np
import pytest

from dl_techniques.utils.weight_transfer import (
    TransferReport,
    load_weights_from_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INPUT_DIM = 8


# DECISION plan-2026-08-10T130454-3649c19e/D-012
# The fixture is a hand-named functional model, NOT a real architecture.
# `load_weights_from_checkpoint` dispatches purely on `layer.name` and
# `skip_prefixes` — it never inspects a model's type, topology or config — so a
# real model contributes nothing to these tests beyond a naming convention and a
# multi-second build. Do NOT re-point this at some surviving architecture to make
# it "realistic": that is exactly the coupling that killed the previous fixture
# (`CliffordNetUNet`) and took an unrelated utility's whole test module down with
# it. See decisions.md D-012.
def _build_multihead(
    head_name: str,
    head_units: int,
    widths=(8, 16),
):
    """A tiny encoder/decoder-shaped functional model with pinned layer names.

    The four backbone layers carry the ``stem_`` / ``enc_`` / ``bottleneck_`` /
    ``dec_`` prefixes the transfer helper's callers use, and the single output
    layer carries the ``head_`` prefix that ``skip_prefixes`` defaults to.
    Layer names are IDENTICAL across head configurations and across ``widths``,
    which is what makes name-matched transfer (and shape mismatch) observable.
    """
    inp = keras.Input(shape=(_INPUT_DIM,), name="input")
    x = keras.layers.Dense(widths[0], activation="relu", name="stem_dense")(inp)
    x = keras.layers.Dense(widths[1], activation="relu", name="enc_dense")(x)
    x = keras.layers.Dense(widths[1], activation="relu", name="bottleneck_dense")(x)
    x = keras.layers.Dense(widths[0], activation="relu", name="dec_dense")(x)
    out = keras.layers.Dense(head_units, name=f"head_{head_name}")(x)
    return keras.Model(inp, out, name=f"tiny_{head_name}")


def _build_classifier(widths=(8, 16)):
    return _build_multihead("cls", 7, widths=widths)


def _build_segmenter(widths=(8, 16)):
    return _build_multihead("seg", 5, widths=widths)


def _save(model, tmp_path, name="source.keras"):
    ckpt = os.path.join(str(tmp_path), name)
    model.save(ckpt)
    return ckpt


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
        ckpt = _save(src, tmp_path)

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
        backbone_layers = [
            n for n in report.loaded
            if n.startswith(("stem_", "enc_", "bottleneck_", "dec_"))
        ]
        assert backbone_layers, "expected backbone layers in report"
        changed = sum(
            1 for n in backbone_layers
            if not _weights_equal(tgt.get_layer(n).get_weights(), tgt_before[n])
        )
        assert changed > 0, "transfer didn't change any backbone weights"

    def test_head_layers_untouched(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

        tgt = _build_segmenter()
        tgt_before = _dup_initial_weights(tgt)

        _ = load_weights_from_checkpoint(tgt, ckpt)

        # Target's seg head layers should still have their original init.
        head_layer_names = [n for n in tgt_before if n.startswith("head_")]
        assert head_layer_names, "sanity: expected some head layers"
        for n in head_layer_names:
            before = tgt_before[n]
            after = tgt.get_layer(n).get_weights()
            if not before and not after:
                continue
            assert _weights_equal(before, after), f"head layer {n} was modified"

    def test_report_summary_string(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

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
        ckpt = _save(src, tmp_path)

        tgt = _build_segmenter()
        report = load_weights_from_checkpoint(tgt, ckpt, skip_prefixes=())
        # No skip prefixes → no skipped_by_prefix entries.
        assert report.skipped_by_prefix == []
        # Source classification head layer ("head_cls") doesn't exist in the
        # segmentation target, so it lands in unused_in_source.
        assert "head_cls" in report.unused_in_source

    def test_default_skip_prefixes_hide_heads(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

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
        # Build two models with DIFFERENT layer widths — their shared backbone
        # layer names will have mismatched weight shapes.
        src = _build_classifier(widths=(8, 16))
        ckpt = _save(src, tmp_path)

        tgt = _build_segmenter(widths=(16, 32))  # different!

        with pytest.raises(ValueError, match="[Ss]hape mismatch"):
            load_weights_from_checkpoint(tgt, ckpt, strict=True)

    def test_non_strict_records_shape_mismatch(self, tmp_path):
        src = _build_classifier(widths=(8, 16))
        ckpt = _save(src, tmp_path)

        tgt = _build_segmenter(widths=(16, 32))

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
        ckpt = _save(src, tmp_path)

        # A Sequential with no declared input shape is NOT built until it is
        # called; a functional model always is, so it cannot exercise this path.
        tgt_unbuilt = keras.Sequential(
            [keras.layers.Dense(5, name="head_seg")], name="unbuilt"
        )
        assert not tgt_unbuilt.built, "fixture precondition: target must be unbuilt"

        with pytest.raises(ValueError, match="built"):
            load_weights_from_checkpoint(tgt_unbuilt, ckpt)
