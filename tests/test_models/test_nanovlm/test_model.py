"""Permanent build+forward smoke test for the nano_vlm family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.

nano_vlm consumes the recently-refactored transformer layers (VisionEncoder /
TextDecoder / fusion) → HIGH regression risk; any break is captured + xfailed.

Factory verified at model.py:689:
``create_nanovlm(variant, vocab_size, fusion_strategy, text_component_type)``;
``variant="mini"`` is the smallest preset (vision img_size=224, patch_size=16,
embed_dim=384). ``call()`` (model.py:467) takes a dict with ``images``
(B, 224, 224, 3) float32 and ``text_tokens`` (B, T) int32, returning LM logits
``(B, combined_seq_len, vocab_size)``.
"""

import numpy as np


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    from dl_techniques.models.nano_vlm.model import create_nanovlm

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
    import keras
    from keras import ops
    from dl_techniques.models.nano_vlm.model import create_nanovlm

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
