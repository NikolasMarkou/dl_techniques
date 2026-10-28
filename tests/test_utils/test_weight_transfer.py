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

Nested-weight note (iter-2 step 13, review concern 7): the D-012 rewrite was
right to decouple, but the first version narrowed what was covered — five flat
``Dense`` layers exercise ``set_weights`` on single-tensor-pair layers only,
while the old ``CliffordNetUNet`` fixture had COMPOSITE ``model.layers`` whose
``get_weights()`` / ``set_weights()`` flatten several sub-layers into one
ordered list. ``load_weights_from_checkpoint`` dispatches on ``.layers`` and
then does a whole-layer ``set_weights`` (``utils/weight_transfer.py:148-149,
179``), so that ORDERING is part of the contract under test. ``_TwoSubDense``
below restores it without re-coupling to any model package.
"""

import contextlib
import json
import os
import zipfile

import keras
import numpy as np
import pytest

from dl_techniques.utils import weight_transfer
from dl_techniques.utils.weight_transfer import (
    DEFAULT_HEAD_PATTERN,
    DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
    TransferReport,
    VocabChange,
    load_pretrained_checkpoint,
    load_weights_from_checkpoint,
    load_weights_or_raise,
    pretrained_path_or_raise,
    read_checkpoint_config,
    refuse_pretrained,
    vocab_change,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INPUT_DIM = 8


@keras.saving.register_keras_serializable(package="test_weight_transfer")
class _TwoSubDense(keras.layers.Layer):
    """A composite layer: ONE ``model.layers`` entry, TWO weight-bearing subs.

    This is the piece the flat-``Dense`` fixture cannot supply. ``get_weights``
    on this layer returns FOUR tensors — ``inner_a`` kernel/bias then
    ``inner_b`` kernel/bias — flattened in sub-layer creation order, and
    ``set_weights`` re-assigns them positionally. So a source and a target that
    build their sub-layers in different orders have identical layer names,
    identical weight COUNTS and identical shapes (when the widths coincide),
    and ``load_weights_from_checkpoint`` will happily transfer ``inner_a``'s
    weights into ``inner_b``. Nothing in the flat fixture can see that.

    Kept deliberately tiny and package-local: the point is the nesting, not the
    architecture. See the module docstring and decisions.md D-012 / D-030.
    """

    def __init__(self, units_a: int, units_b: int, **kwargs):
        super().__init__(**kwargs)
        self.units_a = units_a
        self.units_b = units_b
        # Creation ORDER is the contract: inner_a's weights precede inner_b's
        # in get_weights() / set_weights().
        self.inner_a = keras.layers.Dense(units_a, name="inner_a")
        self.inner_b = keras.layers.Dense(units_b, name="inner_b")

    def build(self, input_shape):
        self.inner_a.build(input_shape)
        self.inner_b.build((*input_shape[:-1], self.units_a))
        super().build(input_shape)

    def call(self, x):
        return self.inner_b(keras.activations.relu(self.inner_a(x)))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units_b)

    def get_config(self):
        config = super().get_config()
        config.update({"units_a": self.units_a, "units_b": self.units_b})
        return config


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
    input_dim: int = _INPUT_DIM,
    extra_layer: bool = False,
):
    """A tiny encoder/decoder-shaped functional model with pinned layer names.

    The four backbone layers carry the ``stem_`` / ``enc_`` / ``bottleneck_`` /
    ``dec_`` prefixes the transfer helper's callers use, and the single output
    layer carries the ``head_`` prefix that ``skip_prefixes`` defaults to.
    Layer names are IDENTICAL across head configurations and across ``widths``,
    which is what makes name-matched transfer (and shape mismatch) observable.
    """
    inp = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(widths[0], activation="relu", name="stem_dense")(inp)
    x = keras.layers.Dense(widths[1], activation="relu", name="enc_dense")(x)
    if extra_layer:
        x = keras.layers.Dense(widths[1], activation="relu", name="enc_extra_dense")(x)
    # ONE composite layer with two weight-bearing sub-layers, so the whole-layer
    # set_weights() ORDERING contract stays exercised (see module docstring).
    # Its two subs are deliberately SHAPE-IDENTICAL (widths[1] -> widths[1] ->
    # widths[1]): a swap between them is then invisible to every shape and
    # count check, which is exactly the failure mode being guarded.
    x = _TwoSubDense(widths[1], widths[1], name="bottleneck_dense")(x)
    x = keras.layers.Dense(widths[0], activation="relu", name="dec_dense")(x)
    out = keras.layers.Dense(head_units, name=f"head_{head_name}")(x)
    return keras.Model(inp, out, name=f"tiny_{head_name}")


def _build_classifier(widths=(8, 16), **kwargs):
    return _build_multihead("cls", 7, widths=widths, **kwargs)


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

    def test_nested_sublayer_weights_transfer_in_order(self, tmp_path):
        """The composite layer's FOUR weights must land on the right subs.

        `load_weights_from_checkpoint` does a whole-layer `set_weights`, which
        is positional. A flat-Dense-only fixture cannot tell "transferred" from
        "transferred into the wrong sub-layer": both give the right count and
        the right shapes.
        """
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

        tgt = _build_segmenter()
        report = load_weights_from_checkpoint(tgt, ckpt)

        assert "bottleneck_dense" in report.loaded, report.summary_string()

        src_layer = src.get_layer("bottleneck_dense")
        tgt_layer = tgt.get_layer("bottleneck_dense")
        assert len(src_layer.get_weights()) == 4, "fixture must be composite"

        # Per SUB-LAYER, not just per flattened list: comparing the flattened
        # lists would also pass if the two subs' weights had been swapped and
        # then swapped back by an equally wrong read.
        for sub in ("inner_a", "inner_b"):
            got = getattr(tgt_layer, sub).get_weights()
            want = getattr(src_layer, sub).get_weights()
            assert _weights_equal(got, want), (
                f"bottleneck_dense/{sub} did not receive its own source weights; "
                "the nested set_weights ordering is wrong"
            )

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
# allow_unloaded: complete-or-raise (plan-2026-10-11T004932-b83935e1 D-020)
# ---------------------------------------------------------------------------


class TestAllowUnloaded:
    """Every target layer loads, except the named ones, or nothing is copied.

    ``stem_dense`` stands in for a vocabulary-dependent embedding: an
    ``input_dim`` change mismatches it and nothing else.
    """

    def test_an_allowed_mismatch_is_tolerated_even_when_strict(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(input_dim=12)
        stem_before = tgt.get_layer("stem_dense").get_weights()

        report = load_weights_from_checkpoint(
            tgt, ckpt, strict=True, allow_unloaded=("stem_",))

        assert [m[0] for m in report.shape_mismatch] == ["stem_dense"]
        assert _weights_equal(tgt.get_layer("stem_dense").get_weights(), stem_before)
        for name in ("enc_dense", "bottleneck_dense", "dec_dense"):
            assert _weights_equal(
                tgt.get_layer(name).get_weights(),
                src.get_layer(name).get_weights()), name

    def test_a_mismatch_outside_the_allowed_layers_raises_and_copies_nothing(
        self, tmp_path
    ):
        src = _build_classifier(widths=(8, 16))
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(widths=(8, 32))   # stem fits, the rest does not
        before = _dup_initial_weights(tgt)

        with pytest.raises(ValueError, match="enc_dense"):
            load_weights_from_checkpoint(tgt, ckpt, allow_unloaded=("stem_",))

        # Nothing was copied, not even the layer that fits.
        for layer in tgt.layers:
            assert _weights_equal(layer.get_weights(), before[layer.name]), layer.name

    def test_a_target_layer_absent_from_the_source_raises(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(extra_layer=True)

        with pytest.raises(ValueError, match="enc_extra_dense"):
            load_weights_from_checkpoint(tgt, ckpt, allow_unloaded=())

        # Anti-vacuity: the default (None) still reports instead of raising.
        report = load_weights_from_checkpoint(tgt, ckpt)
        assert report.missing_in_source == ["enc_extra_dense"]

    def test_a_source_layer_absent_from_the_target_raises(self, tmp_path):
        """D-031: complete-or-raise holds in both directions.

        A deeper checkpoint used to return normally with its extra layer
        dropped (pinned here as allowed until D-031).
        """
        src = _build_classifier(extra_layer=True)
        ckpt = _save(src, tmp_path)
        tgt = _build_segmenter()
        before = _dup_initial_weights(tgt)

        with pytest.raises(ValueError, match=r"no target layer: \['enc_extra_dense'\]"):
            load_weights_from_checkpoint(tgt, ckpt, allow_unloaded=())
        for layer in tgt.layers:
            assert _weights_equal(layer.get_weights(), before[layer.name]), layer.name

        # Anti-vacuity: the default (None) still reports instead of raising.
        report = load_weights_from_checkpoint(tgt, ckpt)
        assert report.unused_in_source == ["enc_extra_dense"]

    def test_skipped_and_allowed_source_layers_never_raise(self, tmp_path):
        src = _build_classifier(extra_layer=True)
        ckpt = _save(src, tmp_path)
        tgt = _build_segmenter()

        report = load_weights_from_checkpoint(
            tgt, ckpt, allow_unloaded=("enc_extra_",))

        assert report.unused_in_source == ["enc_extra_dense"]
        assert report.skipped_by_prefix == ["head_cls"]


class TestVocabChange:
    """D-031: a vocabulary layer is transferred variable by variable.

    ``stem_dense`` stands in for a layer holding a vocabulary table and
    another variable: an ``input_dim`` change (8 -> 12) reshapes its kernel
    along the "vocabulary" axis and leaves its bias as it was.
    """

    def test_only_the_vocabulary_shaped_variable_keeps_its_init(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(input_dim=12)
        kernel_before = tgt.get_layer("stem_dense").get_weights()[0].copy()

        report = load_weights_from_checkpoint(
            tgt, ckpt, strict=True, allow_unloaded=(),
            vocab_change=VocabChange(("stem_",), _INPUT_DIM, 12, ("stem_dense/kernel",)))

        kernel, bias = tgt.get_layer("stem_dense").get_weights()
        np.testing.assert_array_equal(kernel, kernel_before)
        np.testing.assert_array_equal(bias, src.get_layer("stem_dense").get_weights()[1])
        assert report.vocab_fresh == [tgt.get_layer("stem_dense").weights[0].path]
        assert "stem_dense" in report.loaded and not report.shape_mismatch

    def test_a_difference_other_than_the_vocabulary_is_a_mismatch(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(input_dim=12)

        with pytest.raises(ValueError, match="stem_dense"):
            load_weights_from_checkpoint(
                tgt, ckpt, allow_unloaded=(),
                vocab_change=VocabChange(("stem_",), _INPUT_DIM, 13, ("stem_dense/kernel",)))

    def test_a_layer_not_named_is_never_split(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(input_dim=12)

        with pytest.raises(ValueError, match="stem_dense"):
            load_weights_from_checkpoint(
                tgt, ckpt, allow_unloaded=(),
                vocab_change=VocabChange(("enc_",), _INPUT_DIM, 12, ("stem_dense/kernel",)))

    @pytest.mark.parametrize(
        "tables", [(), ("stem_dense/bias",), ("dense/kernel",)],
        ids=["no_table", "another_variable", "another_layer"])
    def test_a_vocabulary_shaped_variable_that_is_not_a_declared_table_raises(
            self, tmp_path, tables):
        """D-007: the kernel's shape is exactly vocabulary-shaped (8 -> 12 on
        the "vocabulary" axis), yet it is not a declared table, so it is a
        mismatch. Before D-007 the shape alone made it a fresh table."""
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _build_classifier(input_dim=12)

        with pytest.raises(ValueError, match="stem_dense"):
            load_weights_from_checkpoint(
                tgt, ckpt, strict=True, allow_unloaded=(),
                vocab_change=VocabChange(("stem_",), _INPUT_DIM, 12, tables))

    def test_a_table_is_matched_by_whole_path_components(self):
        change = VocabChange(("e",), 64, 96, ("word_embeddings/embeddings",))
        shape_from, shape_to = (64, 8), (96, 8)
        assert change.is_vocab_table(
            "bert/embeddings/word_embeddings/embeddings", shape_from, shape_to)
        assert change.is_vocab_table(
            "word_embeddings/embeddings", shape_from, shape_to)
        # a suffix of the string, not of the path components
        assert not change.is_vocab_table(
            "bert/embeddings/my_word_embeddings/embeddings", shape_from, shape_to)
        # another role, even with the table's exact shapes
        assert not change.is_vocab_table(
            "bert/embeddings/position_embeddings/embeddings", shape_from, shape_to)

    def test_a_declared_table_still_needs_a_vocabulary_shaped_difference(self):
        change = VocabChange(("e",), 64, 96, ("word_embeddings/embeddings",))
        path = "word_embeddings/embeddings"
        assert not change.is_vocab_table(path, (64, 8), (64, 8))
        assert not change.is_vocab_table(path, (64, 8), (96, 16))
        assert not change.is_vocab_table(path, (32, 8), (96, 8))


def _wrapped(inner, extra_layer: bool = False):
    """A functional I/O wrapper whose only non-input layer is *inner*.

    The shape of the models the retired ``create_qwen3_classification`` and
    its siblings returned, which their old archives still hold. With
    ``extra_layer`` it gains a second weighted layer and is no longer a
    single-model wrapper.
    """
    inp = keras.Input(shape=(_INPUT_DIM,), name="input")
    out = inner(inp)
    if extra_layer:
        out = keras.layers.Dense(3, name="post_dense")(out)
    return keras.Model(inp, out, name=f"wrapper_{inner.name}")


class TestSingleModelWrapper:
    """D-027 of plan-2026-10-11T004932-b83935e1: a wrapper is compared by its inner model."""

    def test_a_wrapped_checkpoint_loads_into_a_bare_model(self, tmp_path):
        src = _wrapped(_build_classifier())
        ckpt = _save(src, tmp_path)
        tgt = _build_segmenter()

        report = load_weights_from_checkpoint(tgt, ckpt, allow_unloaded=())

        inner = src.get_layer("tiny_cls")
        assert sorted(report.loaded) == sorted(
            l.name for l in tgt.layers
            if l.get_weights() and not l.name.startswith("head_"))
        for name in report.loaded:
            assert _weights_equal(tgt.get_layer(name).get_weights(),
                                  inner.get_layer(name).get_weights()), name
        assert report.skipped_by_prefix == ["head_cls"]

    def test_a_bare_checkpoint_loads_into_a_wrapped_model(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt = _wrapped(_build_segmenter())

        report = load_weights_from_checkpoint(tgt, ckpt, allow_unloaded=())

        inner = tgt.get_layer("tiny_seg")
        assert report.loaded
        for name in report.loaded:
            assert _weights_equal(inner.get_layer(name).get_weights(),
                                  src.get_layer(name).get_weights()), name

    def test_two_wrappers_with_different_heads_transfer_the_backbone(self, tmp_path):
        """Compared whole, the two wrapped models would be one shape mismatch."""
        src = _wrapped(_build_classifier())
        ckpt = _save(src, tmp_path)
        tgt = _wrapped(_build_segmenter())
        seg_head = [w.copy() for w in
                    tgt.get_layer("tiny_seg").get_layer("head_seg").get_weights()]

        report = load_weights_from_checkpoint(
            tgt, ckpt, strict=True, allow_unloaded=())

        assert report.skipped_by_prefix == ["head_cls"]
        assert _weights_equal(
            tgt.get_layer("tiny_seg").get_layer("head_seg").get_weights(), seg_head)

    def test_a_functional_model_of_one_plain_layer_is_compared_as_is(self, tmp_path):
        """Anti-vacuity: only a nested MODEL is unwrapped, never a plain layer."""
        def one_dense():
            inp = keras.Input(shape=(_INPUT_DIM,), name="input")
            out = keras.layers.Dense(4, name="stem_dense")(inp)
            return keras.Model(inp, out, name="one_dense")

        src = one_dense()
        ckpt = _save(src, tmp_path)
        tgt = one_dense()

        report = load_weights_from_checkpoint(tgt, ckpt, allow_unloaded=())

        assert report.loaded == ["stem_dense"]
        assert _weights_equal(tgt.get_layer("stem_dense").get_weights(),
                              src.get_layer("stem_dense").get_weights())

    def test_a_wrapper_with_a_second_weighted_layer_is_compared_as_is(self, tmp_path):
        """Anti-vacuity: only a wrapper around ONE model is unwrapped."""
        src = _wrapped(_build_classifier(), extra_layer=True)
        ckpt = _save(src, tmp_path)
        tgt = _build_segmenter()

        with pytest.raises(ValueError, match="No overlapping layers"):
            load_weights_from_checkpoint(tgt, ckpt)


class TestCheckpointConfig:
    def test_reads_the_archives_own_config(self, tmp_path):
        ckpt = _save(_build_classifier(), tmp_path)
        config = read_checkpoint_config(ckpt)
        assert config["config"]["name"] == "tiny_cls"
        assert "build_config" in config

    def test_an_unreadable_file_gives_an_empty_dict(self, tmp_path):
        bogus = os.path.join(str(tmp_path), "bogus.keras")
        with open(bogus, "w") as f:
            f.write("not a zip")
        assert read_checkpoint_config(bogus) == {}
        assert read_checkpoint_config(os.path.join(str(tmp_path), "nope.keras")) == {}

    def test_a_vocab_change_is_reported_only_when_recorded(self, tmp_path):
        model = keras.Sequential(
            [keras.Input(shape=(4,), dtype="int32"),
             keras.layers.Embedding(64, 8, name="embedding")],
            name="vocab_holder",
        )
        ckpt = _save(model, tmp_path)
        # A Sequential config records no vocab_size: an unknown vocabulary is
        # not a change, so nothing is exempted.
        assert vocab_change(ckpt, 64, ("embedding",), ("embedding/embeddings",)) is None

    def test_a_vocab_change_follows_the_recorded_vocab_size(self, tmp_path, monkeypatch):
        import dl_techniques.utils.weight_transfer as wt
        monkeypatch.setattr(
            wt, "read_checkpoint_config", lambda p: {"config": {"vocab_size": 64}})
        assert wt.vocab_change(
            "x.keras", 64, ("embedding",), ("embedding/embeddings",)) is None
        assert wt.vocab_change(
            "x.keras", 96, ("embedding",), ("embedding/embeddings",)
        ) == VocabChange(("embedding",), 64, 96, ("embedding/embeddings",))

    def test_a_vocab_size_inside_arch_config_is_the_recorded_one(self, monkeypatch):
        """H-Net keeps its vocabulary in ``arch_config``, not at the top."""
        import dl_techniques.utils.weight_transfer as wt
        monkeypatch.setattr(
            wt, "read_checkpoint_config",
            lambda p: {"config": {"arch_config": {"vocab_size": 256}}})
        assert wt.vocab_change(
            "x.keras", 256, ("embeddings",), ("embeddings/embeddings",)) is None
        assert wt.vocab_change(
            "x.keras", 96, ("embeddings",), ("embeddings/embeddings",)
        ) == VocabChange(("embeddings",), 256, 96, ("embeddings/embeddings",))
        # a top-level vocab_size still wins, and a config with neither records none
        monkeypatch.setattr(
            wt, "read_checkpoint_config",
            lambda p: {"config": {"vocab_size": 64,
                                  "arch_config": {"vocab_size": 256}}})
        assert wt.vocab_change(
            "x.keras", 96, ("embeddings",), ("embeddings/embeddings",)
        ).checkpoint_size == 64
        monkeypatch.setattr(wt, "read_checkpoint_config",
                            lambda p: {"config": {"arch_config": {}}})
        assert wt.vocab_change(
            "x.keras", 96, ("embeddings",), ("embeddings/embeddings",)) is None


def _functional_config(*layer_classes, **top):
    """A ``config.json`` as a functional model's archive writes it."""
    layers = [{"class_name": "InputLayer", "config": {"name": "input_ids"}}]
    layers += [{"class_name": name, "config": dict(cfg)}
               for name, cfg in layer_classes]
    return {"class_name": "Functional", "config": dict(top, layers=layers)}


class TestWrapperCheckpointConfig:
    """D-015 of plan-2026-10-11T170958-dc4d2202: a wrapper records no vocab_size itself."""

    def _vocab_change(self, monkeypatch, archive_config, model_vocab=96):
        import dl_techniques.utils.weight_transfer as wt
        monkeypatch.setattr(wt, "read_checkpoint_config", lambda p: archive_config)
        return wt.vocab_change(
            "x.keras", model_vocab, ("token_embedding",), ("token_embedding/embeddings",))

    def test_the_wrapped_models_vocab_size_is_the_recorded_one(self, monkeypatch):
        archive = _functional_config(("Qwen3", {"vocab_size": 64}))
        assert self._vocab_change(monkeypatch, archive) == VocabChange(
            ("token_embedding",), 64, 96, ("token_embedding/embeddings",))
        assert self._vocab_change(monkeypatch, archive, model_vocab=64) is None

    def test_a_top_level_vocab_size_wins_over_a_nested_one(self, monkeypatch):
        archive = _functional_config(
            ("Qwen3", {"vocab_size": 64}), vocab_size=32)
        change = self._vocab_change(monkeypatch, archive)
        assert change.checkpoint_size == 32

    def test_a_wrapper_around_two_models_records_nothing(self, monkeypatch):
        """Anti-vacuity: only a wrapper around ONE model is read through."""
        archive = _functional_config(
            ("Qwen3", {"vocab_size": 64}), ("Qwen3", {"vocab_size": 64}))
        assert self._vocab_change(monkeypatch, archive) is None

    def test_a_model_of_plain_layers_is_not_read_through(self, monkeypatch):
        archive = _functional_config(("Dense", {"units": 4}))
        assert self._vocab_change(monkeypatch, archive) is None


class TestPretrainedPathOrRaise:
    def test_a_falsy_argument_means_random_init(self):
        for falsy in (False, None, ""):
            assert pretrained_path_or_raise(falsy, "M") is None

    def test_true_is_refused_not_faked(self):
        for truthy in (True, 1):
            with pytest.raises(NotImplementedError, match="not distributed"):
                pretrained_path_or_raise(truthy, "M")

    def test_a_missing_path_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            pretrained_path_or_raise(str(tmp_path / "nope.keras"), "M")

    def test_an_existing_path_is_returned_unchanged(self, tmp_path):
        path = tmp_path / "w.keras"
        path.write_bytes(b"")
        assert pretrained_path_or_raise(str(path), "M") == str(path)


class TestRefusePretrained:
    """The sibling for models with no backbone/head split: a path is refused too."""

    def test_a_falsy_argument_means_random_init(self):
        for falsy in (False, None, ""):
            assert refuse_pretrained(falsy, "M") is None

    def test_true_and_an_existing_path_are_both_refused(self, tmp_path):
        path = tmp_path / "w.keras"
        path.write_bytes(b"")
        for truthy in (True, 1, str(path)):
            with pytest.raises(NotImplementedError,
                               match=r"keras\.models\.load_model\(path\)"):
                refuse_pretrained(truthy, "M")

    def test_the_default_message_is_unchanged(self):
        with pytest.raises(NotImplementedError) as caught:
            refuse_pretrained("w.keras", "M")
        assert str(caught.value) == (
            "Pretrained M weights are not distributed with dl_techniques "
            "(got pretrained='w.keras'), and M has no backbone/head split for a partial "
            "load. Load a local .keras archive with keras.models.load_model(path), or "
            "pass pretrained=False to random-init.")

    def test_how_to_load_replaces_the_advice_and_keeps_the_refusal(self):
        """A model whose archives ``load_model`` refuses says what does work instead."""
        with pytest.raises(NotImplementedError) as caught:
            refuse_pretrained(True, "M", how_to_load="Call model.load_weights(path).")
        assert str(caught.value) == (
            "Pretrained M weights are not distributed with dl_techniques "
            "(got pretrained=True). Call model.load_weights(path).")
        assert "load_model" not in str(caught.value) and "backbone/head" not in str(caught.value)

    def test_how_to_load_changes_nothing_for_a_falsy_argument(self):
        assert refuse_pretrained(False, "M", how_to_load="Call model.load_weights(path).") is None


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


# ---------------------------------------------------------------------------
# load_weights_or_raise -- the whole-file restore path
# ---------------------------------------------------------------------------


class TestLoadWeightsOrRaise:
    """``model.load_weights(path, skip_mismatch=True)`` can restore NOTHING.

    It returns normally when the checkpoint's variable names or shapes do not
    line up with the target: every variable is left at its initialized value and
    nothing is reported. Three sites did exactly that with an unconditional
    ``skip_mismatch=True`` and then logged "Loaded weights from ..." -- the
    whole-file restores in ``GPT2.from_variant`` and ``WaveFieldLLM.from_variant``
    (twice). Two of them have since moved to the complete-or-raise by-name
    transfer (plan-2026-10-11T004932-b83935e1 D-024, D-026); the third is
    ``WaveFieldLLM``'s download branch, unreachable while no weights are hosted.
    ``distilbert/model.py`` was the repo's only implementation of the check;
    ``load_weights_or_raise`` is that check, shared.

    Note both arms are needed. A test that only asserts the raise is passed by a
    function that raises unconditionally, which would break every real load.
    """

    def test_matching_checkpoint_still_loads(self, tmp_path):
        """Anti-vacuity: a real load must succeed and change real variables."""
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

        tgt = _build_classifier()
        changed = load_weights_or_raise(tgt, ckpt)

        assert changed > 0
        assert changed <= len(tgt.weights)
        # And the values really are the source's.
        for s_layer, t_layer in zip(src.layers, tgt.layers):
            assert _weights_equal(s_layer.get_weights(), t_layer.get_weights())

    def test_nonmatching_checkpoint_raises_naming_the_cause(self, tmp_path):
        """A checkpoint that restores nothing must raise, not log success.

        MEASURED, and it is not what the layer names suggest: ``load_weights``
        on a ``.keras`` file matches STRUCTURALLY (by position in the saved
        object graph), not by ``layer.name``. A target with entirely different
        names but a coincidentally matching first-layer shape still restores
        that layer. So the "restores nothing" case is built from SHAPES: every
        layer here is shaped so that nothing in the checkpoint fits, and with
        ``skip_mismatch=True`` every variable is skipped. At HEAD this returned
        normally and the call site logged a successful load.
        """
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

        inp = keras.Input(shape=(_INPUT_DIM,), name="input")
        x = keras.layers.Dense(9, activation="relu", name="totally_other_a")(inp)
        out = keras.layers.Dense(13, name="totally_other_b")(x)
        tgt = keras.Model(inp, out, name="other")

        # R-038 / D-054: every variable being skipped is exactly what this
        # test constructs, and Keras reports it as
        # `A total of N objects could not be loaded`. Asserted, not suppressed.
        with pytest.warns(UserWarning, match="could not be loaded"):
            with pytest.raises(ValueError, match="changed none of this model"):
                load_weights_or_raise(tgt, ckpt, skip_mismatch=True)

    def test_the_raise_is_about_the_changed_count_not_the_exception(self, tmp_path):
        """Pin the *count*, not merely "no exception escaped".

        A load into a model whose variables already hold the checkpoint's values
        changes zero variables and is reported as a failure -- deliberately: this
        function cannot distinguish "already equal" from "never restored", and
        treating the ambiguous case as success is the defect it exists to close.
        """
        src = _build_classifier()
        ckpt = _save(src, tmp_path)

        tgt = _build_classifier()
        assert load_weights_or_raise(tgt, ckpt) > 0

        # Second load into the now-identical model: zero variables change.
        with pytest.raises(ValueError, match="changed none of this model"):
            load_weights_or_raise(tgt, ckpt)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        tgt = _build_classifier()
        with pytest.raises(FileNotFoundError):
            load_weights_or_raise(tgt, os.path.join(str(tmp_path), "nope.keras"))

    def test_unbuilt_model_raises(self, tmp_path):
        src = _build_classifier()
        ckpt = _save(src, tmp_path)
        tgt_unbuilt = keras.Sequential(
            [keras.layers.Dense(5, name="head_seg")], name="unbuilt"
        )
        assert not tgt_unbuilt.built, "fixture precondition: target must be unbuilt"
        with pytest.raises(ValueError, match="built"):
            load_weights_or_raise(tgt_unbuilt, ckpt)


# ---------------------------------------------------------------------------
# load_pretrained_checkpoint -- the models' ``pretrained=<path>`` load
# ---------------------------------------------------------------------------

_BUILD_SHAPE = (_INPUT_DIM,)


@keras.saving.register_keras_serializable(package="test_weight_transfer")
class TinyCheckpointClassifier(keras.Model):
    """A subclassed model of the house shape: lazy build, optional head.

    ``build_shapes`` records every shape ``build`` was called with, so a test can
    say how often and at what shape the loader built it.
    """

    def __init__(
        self,
        include_top: bool = True,
        width: int = 6,
        block_width: int = None,
        num_classes: int = 3,
        extra_block: bool = False,
        activation: str = "relu",
        shifts=(1, 2),
        dropout_rate: float = 0.0,
        kernel_size=3,
        head_config_overrides: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.head_config_overrides = head_config_overrides
        self.activation = activation
        self.shifts = tuple(shifts)
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.width = width
        self.block_width = width if block_width is None else block_width
        self.num_classes = num_classes
        self.extra_block = extra_block
        self.build_shapes = []
        self.stem = keras.layers.Dense(width, activation=activation, name="stem")
        self.block_0 = keras.layers.Dense(
            self.block_width, activation="relu", name="block_0"
        )
        self.block_1 = (
            keras.layers.Dense(self.block_width, activation="relu", name="block_1")
            if extra_block else None
        )
        self.classification_head = (
            keras.layers.Dense(num_classes, name="classification_head")
            if include_top else None
        )

    def build(self, input_shape):
        # No `if self.built: return` guard on purpose: every call is recorded.
        self.build_shapes.append(tuple(input_shape))
        self.stem.build(input_shape)
        self.block_0.build((*input_shape[:-1], self.width))
        if self.block_1 is not None:
            self.block_1.build((*input_shape[:-1], self.block_width))
        if self.classification_head is not None:
            self.classification_head.build((*input_shape[:-1], self.block_width))
        super().build(input_shape)

    def call(self, x):
        x = self.block_0(self.stem(x))
        if self.block_1 is not None:
            x = self.block_1(x)
        return x if self.classification_head is None else self.classification_head(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_top": self.include_top,
                "width": self.width,
                "block_width": self.block_width,
                "num_classes": self.num_classes,
                "extra_block": self.extra_block,
                "activation": self.activation,
                "shifts": self.shifts,
                "dropout_rate": self.dropout_rate,
                "kernel_size": self.kernel_size,
                "head_config_overrides": self.head_config_overrides,
            }
        )
        return config


#: The fixture's config keys that change a layer shape or the layer set: the tests of
#: the LAYER rule pass them as ``ignore_config_keys`` so the layer check is the one
#: that fires. The tests of the config rule leave the default.
_LAYER_RULE_ONLY = (
    *DEFAULT_NON_STRUCTURAL_CONFIG_KEYS, "width", "block_width", "extra_block",
)


def _tiny(include_top=True, width=6, block_width=None, built=True,
          num_classes=3, extra_block=False, activation="relu", shifts=(1, 2),
          dropout_rate=0.0, kernel_size=3, head_config_overrides=None):
    model = TinyCheckpointClassifier(
        include_top=include_top, width=width, block_width=block_width,
        num_classes=num_classes, extra_block=extra_block, activation=activation,
        shifts=shifts, dropout_rate=dropout_rate, kernel_size=kernel_size,
        head_config_overrides=head_config_overrides,
    )
    if built:
        model.build((1, *_BUILD_SHAPE))
    return model


def _by_layer_name(model):
    return {layer.name: layer.get_weights() for layer in model.layers}


class TestLoadPretrainedCheckpoint:
    """One reviewed load for every model's ``pretrained=<path>``.

    The reporting loader returns an empty report when the checkpoint and the
    model share layer names and nothing else, and a model that logged a
    successful load then trains from random weights. The first test shows that
    fact; the rest pin what the shared function does about it.
    """

    def test_the_reporting_loader_alone_restores_nothing_and_returns(self, tmp_path):
        ckpt = _save(_tiny(width=6), tmp_path)
        tgt = _tiny(width=10)
        before = _by_layer_name(tgt)

        report = load_weights_from_checkpoint(
            tgt, ckpt, skip_prefixes=(), strict=False
        )

        assert report.num_loaded == 0
        assert report.num_shape_mismatch == 3
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights)

    def test_tensors_are_restored_and_the_report_names_the_layers(self, tmp_path):
        src = _tiny()
        ckpt = _save(src, tmp_path)
        tgt = _tiny()
        assert not _weights_equal(
            _by_layer_name(src)["stem"], _by_layer_name(tgt)["stem"]
        ), "fixture precondition: two builds start from different weights"

        report = load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert isinstance(report, TransferReport)
        assert sorted(report.loaded) == ["block_0", "classification_head", "stem"]
        for name, weights in _by_layer_name(src).items():
            assert _weights_equal(weights, _by_layer_name(tgt)[name])
        x = np.random.default_rng(0).normal(size=(2, _INPUT_DIM)).astype("float32")
        assert np.array_equal(
            keras.ops.convert_to_numpy(src(x)), keras.ops.convert_to_numpy(tgt(x))
        )

    def test_a_headless_target_loads_from_a_headed_checkpoint(self, tmp_path):
        src = _tiny(include_top=True)
        ckpt = _save(src, tmp_path)
        tgt = _tiny(include_top=False)
        assert tgt.classification_head is None

        report = load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert sorted(report.loaded) == ["block_0", "stem"]
        assert report.unused_in_source == ["classification_head"]
        for name in ("stem", "block_0"):
            assert _weights_equal(
                _by_layer_name(src)[name], _by_layer_name(tgt)[name]
            )

    def test_an_unbuilt_model_is_built_once_at_batch_one(self, tmp_path):
        ckpt = _save(_tiny(), tmp_path)
        tgt = _tiny(built=False)
        assert not tgt.built, "fixture precondition"

        load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert tgt.built
        assert tgt.build_shapes == [(1, *_BUILD_SHAPE)]

    def test_a_built_model_is_not_rebuilt(self, tmp_path):
        ckpt = _save(_tiny(), tmp_path)
        tgt = _tiny()
        variables_before = list(tgt.weights)
        builds_before = list(tgt.build_shapes)

        load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert len(tgt.weights) == len(variables_before)
        assert all(a is b for a, b in zip(tgt.weights, variables_before))
        assert tgt.build_shapes == builds_before

    def test_a_missing_file_raises_file_not_found_unwrapped(self, tmp_path):
        tgt = _tiny(built=False)
        missing = os.path.join(str(tmp_path), "nope.keras")
        with pytest.raises(FileNotFoundError, match="nope.keras"):
            load_pretrained_checkpoint(tgt, missing, _BUILD_SHAPE)
        assert not tgt.built, "a missing file is refused before any build"

    @pytest.mark.parametrize("shape", [(None,), (4, None), ()])
    def test_an_unbuilt_model_without_a_channel_count_is_refused_by_name(self, tmp_path, shape):
        """The build needs a known last axis; the failure used to be the model's own ``%``-on-``None``.

        A channel-less ``input_shape`` reached the first convolution as ``None % groups`` and
        came back as "unsupported operand type(s) for %" wrapped in a load error that named
        neither the argument nor the value.
        """
        ckpt = _save(_tiny(), tmp_path)
        tgt = _tiny(built=False)
        with pytest.raises(ValueError, match="input_shape") as caught:
            load_pretrained_checkpoint(tgt, ckpt, shape)
        assert str(shape) in str(caught.value)
        assert not tgt.built, "the check comes before the build"

    def test_a_built_model_needs_no_channel_count_because_it_is_not_built(self, tmp_path):
        ckpt = _save(_tiny(), tmp_path)
        report = load_pretrained_checkpoint(_tiny(), ckpt, (None,))
        assert report.num_loaded == 3

    def test_a_load_that_restores_nothing_raises_and_names_the_path(self, tmp_path):
        """No layer restored, none of them a head the caller expects to differ."""
        ckpt = _save(_tiny(width=6), tmp_path)
        tgt = _tiny(width=10)
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError, match="source.keras"):
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights)

    def test_a_model_made_only_of_a_mismatched_head_restores_no_layer(self, tmp_path):
        """The zero-layer raise stays reachable: the head rule alone does not catch it."""
        src = keras.Sequential(
            [keras.Input((_INPUT_DIM,)), keras.layers.Dense(3, name="classification_head")]
        )
        ckpt = _save(src, tmp_path)
        tgt = keras.Sequential(
            [keras.Input((_INPUT_DIM,)), keras.layers.Dense(4, name="classification_head")]
        )

        with pytest.raises(ValueError, match="restored no layer") as caught:
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert "source.keras" in str(caught.value)

    def test_a_transfer_failure_raises_value_error_chained(self, tmp_path):
        ckpt = _save(_tiny(width=6), tmp_path)
        tgt = _tiny(width=10)

        with pytest.raises(ValueError, match="Failed to load weights") as caught:
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, skip_mismatch=False,
                ignore_config_keys=_LAYER_RULE_ONLY,
            )

        cause = caught.value.__cause__
        assert isinstance(cause, ValueError)
        assert "Shape mismatch on layer" in str(cause)
        assert "source.keras" in str(caught.value)

    def test_skip_mismatch_keeps_the_layers_that_do_fit(self, tmp_path):
        """Anti-vacuity: the refusal must not turn a head-only difference into a failure."""
        ckpt = _save(_tiny(), tmp_path)
        # Every layer but the head keeps its shape: another class count.
        tgt = _tiny(num_classes=5)

        report = load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert sorted(report.loaded) == ["block_0", "stem"]
        assert [name for name, _, _ in report.shape_mismatch] == ["classification_head"]


class TestAPretrainedLoadRestoresEveryLayerOutsideTheHead:
    """D-040: a wrong-architecture checkpoint is an error, not a report.

    ``resnet50`` into ``resnet18`` restored one layer, reported nine shape
    mismatches and nine missing layers, and returned normally. Only the head
    layers (``head_pattern``) may differ.
    """

    def test_a_trunk_layer_of_another_width_raises_and_leaves_the_model_alone(
        self, tmp_path
    ):
        ckpt = _save(_tiny(), tmp_path)
        tgt = _tiny(block_width=5)
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=_LAYER_RULE_ONLY
            )

        message = str(caught.value)
        assert "source.keras" in message
        assert "block_0" in message, message
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights), f"{name} was written"

    def test_the_error_counts_restored_mismatched_and_missing_layers(self, tmp_path):
        ckpt = _save(_tiny(), tmp_path)
        tgt = _tiny(block_width=5)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=_LAYER_RULE_ONLY
            )

        message = str(caught.value)
        # stem fits; block_0 and the head it feeds do not.
        assert "loaded             : 1 layers" in message, message
        assert "shape_mismatch     : 2 layers" in message, message
        assert "missing_in_source  : 2 layers" in message, message

    def test_a_long_offender_list_names_the_first_layers_and_counts_the_rest(
        self, tmp_path
    ):
        def stack(width):
            return keras.Sequential(
                [keras.Input((_INPUT_DIM,))]
                + [keras.layers.Dense(width, name=f"layer_{i:02d}") for i in range(12)]
            )

        ckpt = _save(stack(8), tmp_path)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(stack(9), ckpt, _BUILD_SHAPE)

        message = str(caught.value)
        assert "layer_00" in message and "layer_07" in message, message
        assert "layer_11" not in message, message
        assert "4 more" in message, message

    def test_a_deeper_checkpoint_raises_instead_of_dropping_its_extra_layer(
        self, tmp_path
    ):
        ckpt = _save(_tiny(extra_block=True), tmp_path)
        tgt = _tiny()

        with pytest.raises(ValueError, match="block_1"):
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=_LAYER_RULE_ONLY
            )

    def test_a_shallower_checkpoint_raises_instead_of_leaving_a_layer_random(
        self, tmp_path
    ):
        ckpt = _save(_tiny(), tmp_path)
        tgt = _tiny(extra_block=True)

        with pytest.raises(ValueError, match="block_1"):
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=_LAYER_RULE_ONLY
            )

    def test_another_class_count_loads_the_trunk_and_skips_the_head(self, tmp_path):
        src = _tiny(num_classes=3)
        ckpt = _save(src, tmp_path)
        tgt = _tiny(num_classes=5)
        head_before = _by_layer_name(tgt)["classification_head"]

        report = load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert sorted(report.loaded) == ["block_0", "stem"]
        assert [name for name, _, _ in report.shape_mismatch] == ["classification_head"]
        for name in ("stem", "block_0"):
            assert _weights_equal(_by_layer_name(src)[name], _by_layer_name(tgt)[name])
        assert _weights_equal(head_before, _by_layer_name(tgt)["classification_head"])

    def test_a_headless_checkpoint_loads_into_a_headed_model_with_a_fresh_head(
        self, tmp_path
    ):
        src = _tiny(include_top=False)
        ckpt = _save(src, tmp_path)
        tgt = _tiny(include_top=True)
        head_before = _by_layer_name(tgt)["classification_head"]

        report = load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert sorted(report.loaded) == ["block_0", "stem"]
        assert report.missing_in_source == ["classification_head"]
        assert _weights_equal(head_before, _by_layer_name(tgt)["classification_head"])

    def test_head_pattern_names_the_layers_that_may_differ(self, tmp_path):
        ckpt = _save(_tiny(num_classes=3), tmp_path)

        tgt = _tiny(num_classes=5)
        with pytest.raises(ValueError, match="classification_head"):
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE, head_pattern=r"(?!)")

        tgt = _tiny(block_width=5)
        report = load_pretrained_checkpoint(
            tgt, ckpt, _BUILD_SHAPE, head_pattern=r"block_0|classification_head",
            ignore_config_keys=_LAYER_RULE_ONLY,
        )
        assert report.loaded == ["stem"]

    def test_the_default_pattern_reads_the_deep_supervision_heads_by_index(self):
        import re

        head = re.compile(DEFAULT_HEAD_PATTERN)
        for name in ("classification_head", "supervision_head_0", "supervision_head_12"):
            assert head.fullmatch(name), name
        for name in ("classification_head_proj", "supervision_head", "supervision_head_x",
                     "my_classification_head", "stem", "supervision_head_0_norm"):
            assert not head.fullmatch(name), name

    def test_a_trunk_layer_that_only_starts_like_a_head_must_still_load(self, tmp_path):
        """A name PREFIX is not a head: ``classification_head_proj`` is trunk."""
        def stack(width):
            return keras.Sequential([
                keras.Input((_INPUT_DIM,)),
                keras.layers.Dense(4, name="stem"),
                keras.layers.Dense(width, name="classification_head_proj"),
                keras.layers.Dense(3, name="classification_head"),
            ])

        ckpt = _save(stack(8), tmp_path)
        tgt = stack(9)
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError, match="classification_head_proj"):
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights), f"{name} was written"

    def test_a_supervision_head_of_another_width_is_a_head(self, tmp_path):
        def stack(width):
            return keras.Sequential([
                keras.Input((_INPUT_DIM,)),
                keras.layers.Dense(4, name="stem"),
                keras.layers.Dense(width, name="supervision_head_2"),
            ])

        report = load_pretrained_checkpoint(
            stack(5), _save(stack(3), tmp_path), _BUILD_SHAPE
        )

        assert report.loaded == ["stem"]
        assert [n for n, _, _ in report.shape_mismatch] == ["supervision_head_2"]

    def test_a_head_kept_at_its_initialization_is_named_in_a_warning(
        self, tmp_path, caplog
    ):
        from dl_techniques.utils.logger import logger as dl_logger

        ckpt = _save(_tiny(num_classes=3), tmp_path)
        tgt = _tiny(num_classes=5)

        dl_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level("WARNING", logger=dl_logger.name):
                load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)
        finally:
            dl_logger.removeHandler(caplog.handler)

        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("classification_head" in m and "initialization" in m for m in warned), warned

    def test_skip_mismatch_false_refuses_a_head_of_another_width_too(self, tmp_path):
        ckpt = _save(_tiny(num_classes=3), tmp_path)
        tgt = _tiny(num_classes=5)
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError, match="classification_head") as caught:
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE, skip_mismatch=False)

        assert isinstance(caught.value.__cause__, ValueError)
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights)


def _rewrite_archive_config(ckpt, edit):
    """Rewrite ``config.json`` of a ``.keras`` archive in place: ``edit(config)``."""
    with zipfile.ZipFile(ckpt) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    config = json.loads(members["config.json"])
    edit(config)
    members["config.json"] = json.dumps(config).encode()
    with zipfile.ZipFile(ckpt, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


class TestAPretrainedLoadRefusesAnotherArchitecture:
    """D-041: equal weight shapes do not make equal architectures.

    The fixture's ``activation`` and ``shifts`` change what the model computes and
    no weight shape, so the layer check alone restores every layer and reports
    success. The archive's own config is the only witness.
    """

    def test_the_layer_check_alone_accepts_another_activation(self, tmp_path):
        """Control: this is the hole, shown with the config keys ignored."""
        ckpt = _save(_tiny(activation="relu"), tmp_path)
        tgt = _tiny(activation="gelu")

        report = load_pretrained_checkpoint(
            tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=(*DEFAULT_NON_STRUCTURAL_CONFIG_KEYS, "activation")
        )

        assert sorted(report.loaded) == ["block_0", "classification_head", "stem"]

    def test_another_activation_with_every_shape_equal_is_refused(self, tmp_path):
        ckpt = _save(_tiny(activation="relu"), tmp_path)
        tgt = _tiny(activation="gelu")
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        message = str(caught.value)
        assert "source.keras" in message
        assert "activation: checkpoint 'relu', model 'gelu'" in message, message
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights), f"{name} was written"

    def test_every_differing_key_is_named_with_both_values(self, tmp_path):
        ckpt = _save(_tiny(activation="relu", shifts=(1, 2)), tmp_path)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(
                _tiny(activation="gelu", shifts=(1, 3)), ckpt, _BUILD_SHAPE
            )

        message = str(caught.value)
        assert "2 key(s)" in message, message
        assert "activation: checkpoint 'relu', model 'gelu'" in message, message
        assert "shifts: checkpoint [1, 2], model [1, 3]" in message, message

    def test_equal_tuples_are_equal(self, tmp_path):
        """A tuple is a tuple on the model side and a list in ``config.json``; both read as one list."""
        ckpt = _save(_tiny(shifts=(1, 2)), tmp_path)

        report = load_pretrained_checkpoint(_tiny(shifts=(1, 2)), ckpt, _BUILD_SHAPE)

        assert report.num_loaded == 3

    def test_a_key_the_caller_ignores_may_differ(self, tmp_path):
        ckpt = _save(_tiny(activation="relu"), tmp_path)
        tgt = _tiny(activation="gelu")

        report = load_pretrained_checkpoint(
            tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=("activation", "name", "dtype", "trainable")
        )

        assert report.num_loaded == 3

    def test_the_default_keys_may_differ_without_a_refusal(self, tmp_path):
        """Another class count and a headless target are not another architecture."""
        ckpt = _save(_tiny(num_classes=3, include_top=True), tmp_path)

        report = load_pretrained_checkpoint(
            _tiny(num_classes=5, include_top=False), ckpt, _BUILD_SHAPE
        )

        assert sorted(report.loaded) == ["block_0", "stem"]

    def test_a_key_missing_from_the_archive_is_unknown_not_different(self, tmp_path):
        """A checkpoint of an older revision lacks arguments added since."""
        ckpt = _save(_tiny(activation="relu"), tmp_path)
        _rewrite_archive_config(ckpt, lambda c: c["config"].pop("activation"))
        assert "activation" not in read_checkpoint_config(ckpt)["config"], "precondition"

        report = load_pretrained_checkpoint(_tiny(activation="gelu"), ckpt, _BUILD_SHAPE)

        assert report.num_loaded == 3

    def test_a_key_missing_from_the_model_is_unknown_not_different(
        self, tmp_path, monkeypatch
    ):
        """The comparison skips a recorded key the model has no argument for.

        The recorded config is replaced here, so Keras still rebuilds the archived
        model from the real one; the next test writes the key into the archive.
        """
        ckpt = _save(_tiny(), tmp_path)
        recorded = read_checkpoint_config(ckpt)
        recorded["config"]["argument_removed_since"] = "x"
        monkeypatch.setattr(weight_transfer, "read_checkpoint_config", lambda path: recorded)

        report = load_pretrained_checkpoint(_tiny(), ckpt, _BUILD_SHAPE)

        assert report.num_loaded == 3

    def test_an_argument_the_class_no_longer_takes_fails_the_load_naming_it(self, tmp_path):
        """A real archive that records a removed argument never reaches the comparison.

        Keras rebuilds the archived model from its own config first, so the load
        raises ``ValueError`` naming the argument (MobileNet's ``weight_decay``
        became ``kernel_regularizer``). The control is the same archive without it.
        """
        ckpt = _save(_tiny(), tmp_path)
        assert load_pretrained_checkpoint(_tiny(), ckpt, _BUILD_SHAPE).num_loaded == 3, "control"

        _rewrite_archive_config(ckpt, lambda c: c["config"].update(argument_removed_since="x"))
        target = _tiny()

        with pytest.raises(ValueError, match="argument_removed_since"):
            load_pretrained_checkpoint(target, ckpt, _BUILD_SHAPE)

    def test_a_graph_model_is_checked_by_layers_only(self, tmp_path):
        """A Sequential config lists its layers, whose generated names differ per build."""
        def stack():
            return keras.Sequential([
                keras.Input((_INPUT_DIM,)),
                keras.layers.Dense(4, name="stem"),
                keras.layers.Dense(3, name="classification_head"),
            ])

        report = load_pretrained_checkpoint(stack(), _save(stack(), tmp_path), _BUILD_SHAPE)

        assert report.num_loaded == 2

    def test_a_config_of_another_class_that_holds_no_such_model_is_checked_by_layers_only(
        self, tmp_path, monkeypatch, caplog
    ):
        """No model of the target's class can be located in the archive's config.

        The recorded config is edited to a class that has none of the model's
        constructor keys and no layer list, so nothing can be compared. The load
        still goes by layers, and says so in a warning rather than skipping silently.
        """
        ckpt = _save(_tiny(activation="relu"), tmp_path)
        recorded = read_checkpoint_config(ckpt)
        recorded["class_name"] = "Functional"
        monkeypatch.setattr(weight_transfer, "read_checkpoint_config", lambda path: recorded)

        with _warnings_logged(caplog) as warned:
            report = load_pretrained_checkpoint(_tiny(activation="gelu"), ckpt, _BUILD_SHAPE)

        assert report.num_loaded == 3
        assert any("configuration was not compared" in m and "Functional" in m for m in warned), warned


@contextlib.contextmanager
def _warnings_logged(caplog):
    """The WARNING messages the library logger emits inside the block (a list)."""
    from dl_techniques.utils.logger import logger as dl_logger

    dl_logger.addHandler(caplog.handler)
    messages = []
    try:
        with caplog.at_level("WARNING", logger=dl_logger.name):
            yield messages
    finally:
        dl_logger.removeHandler(caplog.handler)
        messages.extend(r.getMessage() for r in caplog.records if r.levelname == "WARNING")


def _wrap(model):
    """The shape the repo's trainers save: ``keras.Model(inp, model(inp))``."""
    inp = keras.Input(shape=_BUILD_SHAPE, name="input")
    return keras.Model(inp, model(inp), name="wrapper")


class TestAPretrainedLoadReadsTheConfigOfAWrappedModel:
    """A functional wrapper keeps the wrapped model's config in its archive."""

    def test_a_wrapped_checkpoint_of_another_activation_is_refused(self, tmp_path):
        ckpt = _save(_wrap(_tiny(activation="relu")), tmp_path)
        tgt = _tiny(activation="gelu")
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert "activation: checkpoint 'relu', model 'gelu'" in str(caught.value), caught.value
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights), f"{name} was written"

    def test_a_wrapped_checkpoint_of_the_same_architecture_loads(self, tmp_path):
        src = _tiny(activation="gelu")
        ckpt = _save(_wrap(src), tmp_path)

        report = load_pretrained_checkpoint(_tiny(activation="gelu"), ckpt, _BUILD_SHAPE)

        assert report.num_loaded == 3

    def test_a_wrapper_holding_two_models_of_the_class_cannot_be_compared(
        self, tmp_path, caplog
    ):
        """Which of the two is meant is unknown: layers only, and the load says so."""
        first, second = _tiny(activation="relu"), _tiny(activation="relu")
        inp = keras.Input(shape=_BUILD_SHAPE, name="input")
        wrapper = keras.Model(inp, keras.layers.Add()([first(inp), second(inp)]), name="wrapper")
        ckpt = _save(wrapper, tmp_path)

        with _warnings_logged(caplog) as warned:
            with pytest.raises(ValueError) as caught:
                load_pretrained_checkpoint(_tiny(activation="gelu"), ckpt, _BUILD_SHAPE)

        assert "activation: checkpoint" not in str(caught.value), caught.value
        assert any("configuration was not compared" in m for m in warned), warned

    def test_a_wrapper_around_another_class_is_not_read_as_the_models_config(
        self, tmp_path, caplog
    ):
        inp = keras.Input(shape=(_INPUT_DIM,), name="input")
        plain = keras.Model(inp, keras.layers.Dense(4, name="stem")(inp), name="plain")

        with _warnings_logged(caplog) as warned:
            with pytest.raises(ValueError):
                load_pretrained_checkpoint(_tiny(), _save(plain, tmp_path), _BUILD_SHAPE)

        assert any("configuration was not compared" in m for m in warned), warned


class TestAPretrainedLoadComparesAHeadThatWouldBeCopied:
    """``head_config_overrides`` changes what the head computes and no weight shape.

    The head layer is copied only when every one of its weight shapes fits. The
    configuration is compared exactly then; a head the shapes already exclude
    (another ``num_classes``) is a new head and may be configured as the caller likes.
    """

    MAX = {"pooling_type": "max"}

    def test_a_head_that_computes_another_function_with_equal_shapes_is_refused(self, tmp_path):
        ckpt = _save(_tiny(head_config_overrides=self.MAX), tmp_path)
        tgt = _tiny()
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError) as caught:
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        message = str(caught.value)
        assert "head_config_overrides['pooling_type']: checkpoint 'max', model <default>" in message, message
        assert "classification_head" in message, message
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights), f"{name} was written"

    def test_a_layer_is_a_head_only_when_the_whole_name_matches_the_head_pattern(self, tmp_path):
        """A trunk layer called ``classification_head_proj`` is not a head (the layer rule says so).

        The head comparison reads ``head_pattern`` the same way: a prefix match would take
        every layer whose name merely starts like the head's for the head, and refuse a load
        over a configuration that no head holds.
        """
        ckpt = _save(_tiny(head_config_overrides=self.MAX), tmp_path)

        report = load_pretrained_checkpoint(
            _tiny(), ckpt, _BUILD_SHAPE, head_pattern=r"classification"
        )
        assert sorted(report.loaded) == ["block_0", "classification_head", "stem"]

        with pytest.raises(ValueError, match="head_config_overrides"):
            load_pretrained_checkpoint(
                _tiny(), ckpt, _BUILD_SHAPE, head_pattern=r"classification_head"
            )

    def test_the_same_overrides_on_both_sides_load(self, tmp_path):
        ckpt = _save(_tiny(head_config_overrides=self.MAX), tmp_path)

        report = load_pretrained_checkpoint(
            _tiny(head_config_overrides=dict(self.MAX)), ckpt, _BUILD_SHAPE
        )

        assert sorted(report.loaded) == ["block_0", "classification_head", "stem"]

    def test_another_class_count_makes_a_new_head_that_may_be_configured_freely(self, tmp_path):
        ckpt = _save(_tiny(num_classes=3, head_config_overrides=self.MAX), tmp_path)

        report = load_pretrained_checkpoint(
            _tiny(num_classes=5, head_config_overrides={"pooling_type": "avg"}), ckpt, _BUILD_SHAPE
        )

        assert sorted(report.loaded) == ["block_0", "stem"]

    def test_a_headless_target_has_no_head_to_compare(self, tmp_path):
        ckpt = _save(_tiny(head_config_overrides=self.MAX), tmp_path)

        report = load_pretrained_checkpoint(_tiny(include_top=False), ckpt, _BUILD_SHAPE)

        assert sorted(report.loaded) == ["block_0", "stem"]

    def test_a_headless_checkpoint_has_no_head_to_compare(self, tmp_path):
        ckpt = _save(_tiny(include_top=False), tmp_path)

        report = load_pretrained_checkpoint(
            _tiny(head_config_overrides=self.MAX), ckpt, _BUILD_SHAPE
        )

        assert sorted(report.loaded) == ["block_0", "stem"]

    def test_the_training_only_head_keys_may_differ(self, tmp_path):
        ckpt = _save(
            _tiny(head_config_overrides={"dropout_rate": 0.3, "kernel_initializer": "he_normal"}),
            tmp_path,
        )

        report = load_pretrained_checkpoint(
            _tiny(head_config_overrides={"classifier_dropout_rate": 0.1,
                                         "kernel_regularizer": "l2"}),
            ckpt, _BUILD_SHAPE,
        )

        assert "classification_head" in report.loaded

    def test_the_caller_may_accept_the_other_head_by_ignoring_the_key(self, tmp_path):
        ckpt = _save(_tiny(head_config_overrides=self.MAX), tmp_path)

        report = load_pretrained_checkpoint(
            _tiny(), ckpt, _BUILD_SHAPE,
            ignore_config_keys=(*DEFAULT_NON_STRUCTURAL_CONFIG_KEYS, "head_config_overrides"),
        )

        assert "classification_head" in report.loaded

    def test_the_default_ignore_set_no_longer_holds_the_key(self):
        assert "head_config_overrides" not in DEFAULT_NON_STRUCTURAL_CONFIG_KEYS

    def test_a_key_missing_from_the_archive_is_unknown_not_different(self, tmp_path):
        ckpt = _save(_tiny(head_config_overrides=self.MAX), tmp_path)
        _rewrite_archive_config(ckpt, lambda c: c["config"].pop("head_config_overrides"))
        assert "head_config_overrides" not in read_checkpoint_config(ckpt)["config"], "precondition"

        report = load_pretrained_checkpoint(
            _tiny(head_config_overrides={"pooling_type": "avg"}), ckpt, _BUILD_SHAPE
        )

        assert "classification_head" in report.loaded


class TestAnIntegerAndItsPairAreOneKernel:
    """``kernel_size=7`` and ``kernel_size=(7, 7)`` build the same layers."""

    def test_an_integer_checkpoint_loads_into_the_pair_spelling(self, tmp_path):
        ckpt = _save(_tiny(kernel_size=7), tmp_path)

        assert load_pretrained_checkpoint(_tiny(kernel_size=(7, 7)), ckpt, _BUILD_SHAPE).num_loaded == 3

    def test_a_pair_checkpoint_loads_into_the_integer_spelling(self, tmp_path):
        ckpt = _save(_tiny(kernel_size=(7, 7)), tmp_path)

        assert load_pretrained_checkpoint(_tiny(kernel_size=7), ckpt, _BUILD_SHAPE).num_loaded == 3

    def test_a_pair_of_two_different_sizes_is_another_kernel(self, tmp_path):
        ckpt = _save(_tiny(kernel_size=7), tmp_path)

        with pytest.raises(ValueError, match=r"kernel_size: checkpoint 7, model \[7, 5\]"):
            load_pretrained_checkpoint(_tiny(kernel_size=(7, 5)), ckpt, _BUILD_SHAPE)

    def test_another_integer_is_another_kernel(self, tmp_path):
        ckpt = _save(_tiny(kernel_size=7), tmp_path)

        with pytest.raises(ValueError, match="kernel_size"):
            load_pretrained_checkpoint(_tiny(kernel_size=(5, 5)), ckpt, _BUILD_SHAPE)

    def test_a_bool_is_not_an_integer_spelling(self, tmp_path):
        ckpt = _save(_tiny(kernel_size=True), tmp_path)

        with pytest.raises(ValueError, match="kernel_size"):
            load_pretrained_checkpoint(_tiny(kernel_size=(1, 1)), ckpt, _BUILD_SHAPE)


class TestARefusedLoadCopiesNothing:
    """What "a refused load leaves the model untouched" means for an unbuilt model.

    The configuration is compared before the build; the layer rule needs the built
    layers, so an unbuilt model that fails it has been built (at its own
    initialization) and nothing was copied into it.
    """

    def test_a_configuration_refusal_leaves_an_unbuilt_model_unbuilt(self, tmp_path):
        ckpt = _save(_tiny(activation="relu"), tmp_path)
        tgt = _tiny(activation="gelu", built=False)

        with pytest.raises(ValueError, match="another architecture"):
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert not tgt.built
        assert tgt.build_shapes == []

    def test_a_layer_refusal_builds_an_unbuilt_model_and_copies_nothing(self, tmp_path):
        src = _tiny()
        ckpt = _save(src, tmp_path)
        tgt = _tiny(extra_block=True, built=False)

        with pytest.raises(ValueError, match="block_1"):
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, ignore_config_keys=_LAYER_RULE_ONLY
            )

        assert tgt.built and tgt.build_shapes == [(1, *_BUILD_SHAPE)]
        # `stem` fits the checkpoint's by shape, so a load that went on would have copied it.
        assert not _weights_equal(
            tgt.get_layer("stem").get_weights(), src.get_layer("stem").get_weights()
        )
