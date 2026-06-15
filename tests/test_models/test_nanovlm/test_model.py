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
