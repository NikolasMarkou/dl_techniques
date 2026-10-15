"""
Test suite for NanoVLM (vision-language model).

call() consumes a dict {'images': (B,224,224,3), 'text_tokens': (B,T)} and
returns LM logits (B, num_vision_tokens + T, vocab_size). Covers construction via the
create_nanovlm factory, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.vision_language.nano_vlm.model import create_nanovlm, NanoVLM

VOCAB = 256


def _model():
    return create_nanovlm("mini", vocab_size=VOCAB)


def _inputs(batch=2, seq=16):
    rng = np.random.default_rng(0)
    return {
        "images": rng.random((batch, 224, 224, 3)).astype("float32"),
        "text_tokens": rng.integers(0, VOCAB, (batch, seq)).astype("int32"),
    }


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


class TestNanoVLM:

    def test_factory_construction(self):
        model = _model()
        assert isinstance(model, NanoVLM)

    def test_forward_logits(self):
        out = _model()(_inputs(), training=False)
        assert out.shape[0] == 2 and out.shape[-1] == VOCAB
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _inputs()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "nano_vlm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="NanoVLM differs after .keras round-trip")


# --- Ported from test_nanovlm/test_model.py (2026-06-15 build/forward sweep,
#     plan_2026-06-15_b5cec9e4): the 2 unique smoke + weight-tie tests. ---


def test_smoke_build_and_forward():
    """Permanent build+forward smoke test for the nano_vlm family.

    Part of the 2026-06-15 model build/forward sweep. REPORT-ONLY idiom: a
    build/forward break is documented, never silently passed.
    """
    model = create_nanovlm(variant="mini", vocab_size=256)

    b, t = 2, 16
    inputs = {
        "images": np.random.rand(b, 224, 224, 3).astype("float32"),
        "text_tokens": np.random.randint(0, 256, (b, t)).astype("int32"),
    }
    out = model(inputs, training=False)

    # Output may be a single logits tensor, or a dict/tuple/list of tensors.
    if isinstance(out, dict):
        for v in out.values():
            _assert_finite(v)
    elif isinstance(out, (tuple, list)):
        for v in out:
            _assert_finite(v)
    else:
        _assert_finite(out)

    # mini: a 14x14 patch grid, pixel-shuffled 2x2 to 7x7 = 49 tokens, plus
    # CLS = 50 vision positions, then t = 16 text positions; vocab 256.
    logits = np.asarray(out)
    assert model.num_vision_tokens == 50
    assert logits.shape == (b, 66, 256), f"unexpected logits shape {logits.shape}"


def test_shared_embedding_is_really_tied():
    """Prove input/output embeddings are tied at CALL time (plan D-001).

    The tied path produces logits via matmul(x, transpose(word_embeddings.embeddings)).

    1. Positive: model logits == matmul(decoder_hidden_states, transpose(emb)).
    2. Negative: a tied model builds NO output Dense at all -- the
       `shared_output_projection` it used to build was created, saved and
       counted (24,576,000 params on 'base') and never called
       (plan-2026-10-14T062616-81938e81 D-037). No weight path names one.

    Asserted OUTSIDE any try (a real failure must not silently pass).
    """
    model = create_nanovlm(variant="mini", vocab_size=256)

    # Guard must hold for this variant (shared embedding + has word_embeddings).
    assert model.use_shared_embedding is True
    assert model.ties_output_to_embeddings is True
    assert hasattr(model.text_decoder, "word_embeddings")

    b, t = 2, 16
    inputs = {
        "images": np.random.rand(b, 224, 224, 3).astype("float32"),
        "text_tokens": np.random.randint(0, 256, (b, t)).astype("int32"),
    }

    logits = np.asarray(model(inputs, training=False))

    # Positive proof: reconstruct logits from the tied embedding weight directly.
    # Reproduce the call() pipeline up to the decoder's hidden states (no
    # output projection): vision tokens projected, prefixed to the text's
    # word embeddings, one decoder over both.
    vision_tokens = model.modality_projection(
        model.vision_encoder(inputs["images"], training=False), training=False)
    text_embeddings = model.text_decoder.embed_tokens(inputs["text_tokens"])
    joint = keras.ops.concatenate([vision_tokens, text_embeddings], axis=1)
    hidden = model.text_decoder.decode_embeddings(joint, training=False)
    emb = model.text_decoder.word_embeddings.embeddings
    manual = np.asarray(keras.ops.matmul(hidden, keras.ops.transpose(emb)))
    assert np.allclose(logits, manual, atol=1e-4), (
        "shared-path logits do not equal matmul(x, transpose(embeddings)) -> tie not wired"
    )

    # Negative proof: no second vocabulary matrix exists.
    assert model.output_projection is None
    output_paths = [w.path for w in model.weights if "output_projection" in w.path]
    assert output_paths == [], output_paths
    # The only (vocab, ...) or (..., vocab) variable is the word table itself.
    vocab_shaped = [w.path for w in model.weights if 256 in tuple(w.shape)]
    assert vocab_shaped == [model.text_decoder.word_embeddings.embeddings.path], vocab_shaped


def test_an_untied_model_projects_through_its_own_dense():
    """The other branch: untied, the logits come from `output_projection`."""
    model = create_nanovlm(variant="mini", vocab_size=VOCAB, use_shared_embedding=False)
    assert model.ties_output_to_embeddings is False
    x = _inputs(batch=1, seq=4)
    before = keras.ops.convert_to_numpy(model(x, training=False))
    kernel = model.output_projection.kernel
    kernel.assign(keras.ops.zeros_like(kernel))
    after = keras.ops.convert_to_numpy(model(x, training=False))
    assert np.any(before != 0.0)
    np.testing.assert_array_equal(after, np.zeros_like(after))


def test_tied_logits_are_float32_under_mixed_float16():
    """`plan-2026-09-12T123331-28fd855f`'s D-005 completion-fix (task 6) wires
    both tied-embedding call sites (`call()` and `generate()`) through
    `tied_embedding_logits`, which floors a narrower-than-float32 matmul
    result up to float32 for loss-facing numerical stability -- the same
    floor pinned for `gpt2`/`wave_field`/`cliffordnet` elsewhere. This is why
    `nano_vlm` carries an `expected_compute_dtype="float32"` pin in
    `tests/test_models/precision_arm_subjects.py`."""
    previous = keras.mixed_precision.global_policy().name
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = create_nanovlm(variant="mini", vocab_size=256)
        assert model.use_shared_embedding is True

        b, t = 2, 16
        inputs = {
            "images": np.random.rand(b, 224, 224, 3).astype("float32"),
            "text_tokens": np.random.randint(0, 256, (b, t)).astype("int32"),
        }
        logits = model(inputs, training=False)

        assert str(logits.dtype) == "<dtype: 'float32'>", (
            f"expected float32 tied logits under mixed_float16, got {logits.dtype}"
        )
    finally:
        keras.mixed_precision.set_global_policy(previous)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestNanoVLMGradientFlow:
    """Every trainable weight is on the backward graph, with no waiver.

    Until 2026-10-15 this carried a by-design waiver for
    `shared_output_projection/kernel`, a Dense a tied model built and never
    called. The Dense is gone (plan-2026-10-14T062616-81938e81 D-037), so the
    oracle runs with no `expect_zero` at all.
    """

    @pytest.mark.parametrize("tied", [True, False])
    def test_gradients_reach_every_trainable_weight(self, tied):
        model = create_nanovlm("mini", vocab_size=VOCAB, use_shared_embedding=tied)
        x = _inputs()
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0

    def test_the_tied_embedding_is_what_carries_the_output_gradient(self):
        """A tied model has no output kernel, so the vocabulary gradient must
        reach the word embedding table: it is the output matrix too."""
        model = _model()
        x = _inputs()
        model(x, training=False)

        from ..gradient_flow_oracle import gradient_report

        report = gradient_report(model, x)
        embedding_paths = [
            path for path in report
            if "word_embeddings" in path and report[path] is not None
        ]
        assert embedding_paths, "no word_embeddings weight received a gradient"
        assert any(report[path] > 0.0 for path in embedding_paths), (
            "the tied embedding receives no gradient, so nothing carries the "
            "output projection's gradient"
        )
