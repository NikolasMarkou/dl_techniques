"""
Test suite for NanoVLM (vision-language model).

call() consumes a dict {'images': (B,224,224,3), 'text_tokens': (B,T)} and
returns LM logits (B, combined_seq, vocab_size). Covers construction via the
create_nanovlm factory, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.nano_vlm.model import create_nanovlm, NanoVLM

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

    # mini: vision seq (224/16)^2=196 + 1 (cls?) ... + text t=16 -> combined 213; vocab 256.
    logits = np.asarray(out)
    assert logits.shape == (b, 213, 256), f"unexpected logits shape {logits.shape}"


def test_shared_embedding_is_really_tied():
    """Prove input/output embeddings are tied at CALL time (plan D-001).

    The shared path produces logits via matmul(x, transpose(word_embeddings.embeddings)),
    NOT via output_projection. Two independent proofs:

    1. Positive: model logits == matmul(combined_features, transpose(emb)) exactly.
    2. Negative: zeroing output_projection.kernel leaves shared-path logits UNCHANGED
       (proves the Dense kernel is not the logit source). This assertion FAILS if the
       tie were not wired (i.e. if logits still came from output_projection).

    Asserted OUTSIDE any try (a real failure must not silently pass).
    """
    from keras import ops

    model = create_nanovlm(variant="mini", vocab_size=256)

    # Guard must hold for this variant (decoder + shared embedding + has word_embeddings).
    assert model.use_shared_embedding is True
    assert model.text_component_type == "decoder"
    assert hasattr(model.text_component, "word_embeddings")

    b, t = 2, 16
    inputs = {
        "images": np.random.rand(b, 224, 224, 3).astype("float32"),
        "text_tokens": np.random.randint(0, 256, (b, t)).astype("int32"),
    }

    logits_before = np.asarray(model(inputs, training=False))

    # Positive proof: reconstruct logits from the tied embedding weight directly.
    # Reproduce the call() pipeline up to combined_features (no output projection).
    vision_features = model.vision_encoder(inputs["images"], training=False)
    text_features = model.text_component(inputs["text_tokens"], training=False)
    fused = model.fusion_layer([vision_features, text_features], training=False)
    if isinstance(fused, tuple):
        vision_fused, text_fused = fused
        combined = ops.concatenate([vision_fused, text_fused], axis=1)
    else:
        combined = fused
    emb = model.text_component.word_embeddings.embeddings
    manual = np.asarray(ops.matmul(combined, ops.transpose(emb)))
    assert np.allclose(logits_before, manual, atol=1e-4), (
        "shared-path logits do not equal matmul(x, transpose(embeddings)) -> tie not wired"
    )

    # Negative proof: destroy the Dense kernel; shared-path logits must be unchanged.
    kernel_var = model.output_projection.kernel
    kernel_var.assign(keras.ops.zeros_like(kernel_var))
    logits_after = np.asarray(model(inputs, training=False))
    assert np.allclose(logits_before, logits_after, atol=1e-6), (
        "zeroing output_projection.kernel changed logits -> Dense is the logit source, "
        "tie is NOT real (the plan_2026-06-15_39a31d4a/D-002 failure)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
