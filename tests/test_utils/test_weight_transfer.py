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

import os

import keras
import numpy as np
import pytest

from dl_techniques.utils.weight_transfer import (
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_top = include_top
        self.width = width
        self.block_width = width if block_width is None else block_width
        self.build_shapes = []
        self.stem = keras.layers.Dense(width, activation="relu", name="stem")
        self.block_0 = keras.layers.Dense(
            self.block_width, activation="relu", name="block_0"
        )
        self.classification_head = (
            keras.layers.Dense(3, name="classification_head") if include_top else None
        )

    def build(self, input_shape):
        # No `if self.built: return` guard on purpose: every call is recorded.
        self.build_shapes.append(tuple(input_shape))
        self.stem.build(input_shape)
        self.block_0.build((*input_shape[:-1], self.width))
        if self.classification_head is not None:
            self.classification_head.build((*input_shape[:-1], self.block_width))
        super().build(input_shape)

    def call(self, x):
        x = self.block_0(self.stem(x))
        return x if self.classification_head is None else self.classification_head(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_top": self.include_top,
                "width": self.width,
                "block_width": self.block_width,
            }
        )
        return config


def _tiny(include_top=True, width=6, block_width=None, built=True):
    model = TinyCheckpointClassifier(
        include_top=include_top, width=width, block_width=block_width
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

    def test_a_load_that_restores_nothing_raises_and_names_the_path(self, tmp_path):
        ckpt = _save(_tiny(width=6), tmp_path)
        tgt = _tiny(width=10)
        before = _by_layer_name(tgt)

        with pytest.raises(ValueError, match="restored no layer") as caught:
            load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert "source.keras" in str(caught.value)
        for name, weights in _by_layer_name(tgt).items():
            assert _weights_equal(before[name], weights)

    def test_a_transfer_failure_raises_value_error_chained(self, tmp_path):
        ckpt = _save(_tiny(width=6), tmp_path)
        tgt = _tiny(width=10)

        with pytest.raises(ValueError, match="Failed to load weights") as caught:
            load_pretrained_checkpoint(
                tgt, ckpt, _BUILD_SHAPE, skip_mismatch=False
            )

        cause = caught.value.__cause__
        assert isinstance(cause, ValueError)
        assert "Shape mismatch on layer" in str(cause)
        assert "source.keras" in str(caught.value)

    def test_skip_mismatch_keeps_the_layers_that_do_fit(self, tmp_path):
        """Anti-vacuity: the zero-layer raise must not turn a partial load into a failure."""
        ckpt = _save(_tiny(), tmp_path)
        # The stem keeps its width; the block and the head it feeds do not.
        tgt = _tiny(block_width=5)

        report = load_pretrained_checkpoint(tgt, ckpt, _BUILD_SHAPE)

        assert report.loaded == ["stem"]
        assert sorted(name for name, _, _ in report.shape_mismatch) == [
            "block_0",
            "classification_head",
        ]
