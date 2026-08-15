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
    load_weights_from_checkpoint,
    load_weights_or_raise,
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
    # ONE composite layer with two weight-bearing sub-layers, so the whole-layer
    # set_weights() ORDERING contract stays exercised (see module docstring).
    # Its two subs are deliberately SHAPE-IDENTICAL (widths[1] -> widths[1] ->
    # widths[1]): a swap between them is then invisible to every shape and
    # count check, which is exactly the failure mode being guarded.
    x = _TwoSubDense(widths[1], widths[1], name="bottleneck_dense")(x)
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
    ``skip_mismatch=True`` and then logged "Loaded weights from ..." --
    ``gpt2/gpt2.py:480``, ``wave_field/model.py:745`` and ``:769``.
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
