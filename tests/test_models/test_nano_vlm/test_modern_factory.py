"""
The `'modern'` NanoVLM row: RMSNorm, SwiGLU, differential attention, RoPE.

Until plan-2026-10-14T062616-81938e81 (D-006, D-037) this configuration was a
second factory, `create_modern_nanovlm(vocab_size, embed_dim, **kw)`, built
from hard-coded dicts that bypassed `MODEL_VARIANTS`. It is now the row
`NanoVLM.MODEL_VARIANTS['modern']`, built by `NanoVLM.from_variant('modern')`;
the factory is deleted. Every behavioural assertion below is the one the old
file made against the old factory.

Nothing in `tests/` called the old factory before this module, which is why
two ghost-API defects survived in it: `vision_config['attention_type'] =
'differential_attention'` (the attention factory registers `'differential'`)
raised inside `_create_vision_encoder` and masked the second defect,
`fusion_config`'s ghost `embed_dim`/`num_heads` keys, which `MultiModalFusion`
rejects at `keras.layers.Layer.__init__`.

The row fixes `img_size=224`, `patch_size=16` and `depth=12`. "Tiny" here
means a narrow model, not a shallow one: the old factory's `embed_dim=64`
(so 64 // 64 = 1 head) is laid over the row with the one-level merge.

Fixing the two ghost-API defects made a THIRD defect reachable, and it is
fixed too: the configuration USED to default to `fusion_strategy=
'tensor_fusion'`, which could not run on this model at all, because
`MultiModalFusion._call_tensor_fusion` concatenates the modalities on the
feature axis and so requires equal sequence lengths, while NanoVLM always
feeds vision (197) and text (T) streams of different length. The default is
`cross_attention`, pinned by
`test_no_argument_factory_forward_passes_to_logits`; requesting
`tensor_fusion` explicitly on mismatched-length inputs raises a named
`ValueError` instead of an opaque backend `ConcatOp` error, pinned by
`test_tensor_fusion_raises_named_error_on_unequal_sequence_lengths`.

That second test replaced a bare `strict=True` xfail. A strict xfail with no
`raises=` cannot tell "still broken for the documented reason" from "now broken
for an unrelated one" -- pre-fix the failure was
`tensorflow...InvalidArgumentError: ConcatOp : Dimension 1 in both shapes must
be equal: shape[0] = [2,197,64] vs. shape[1] = [2,8,64]`, which is not even a
`ValueError`, so the positive `pytest.raises(ValueError, match=...)` form below
is non-vacuous: it does not match the pre-guard failure.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.vision_language.nano_vlm.model import NanoVLM

VOCAB = 64
EMBED_DIM = 64  # -> num_heads = EMBED_DIM // 64 = 1, the old factory's rule


def _modern(**kwargs):
    """The 'modern' row at width EMBED_DIM; nothing else of the row changes."""
    heads = EMBED_DIM // 64
    return NanoVLM.from_variant(
        "modern", vocab_size=VOCAB,
        vision_config={"embed_dim": EMBED_DIM, "num_heads": heads},
        text_config={"embed_dim": EMBED_DIM, "num_heads": heads},
        fusion_config={"dim": EMBED_DIM, "attention_config": {"num_heads": heads}},
        **kwargs,
    )


def _model(fusion_strategy="cross_attention"):
    return _modern(fusion_strategy=fusion_strategy)


def _inputs(batch=2, seq=8):
    rng = np.random.default_rng(0)
    return {
        "images": rng.random((batch, 224, 224, 3)).astype("float32"),
        "text_tokens": rng.integers(0, VOCAB, (batch, seq)).astype("int32"),
    }


class TestTheModernVariant:

    def test_factory_construction(self):
        """The row with only its width narrowed -- no fusion or component
        override, so this is the literal path both defects lived on.

        Pre-fix this raised `ValueError: Unknown attention type:
        differential_attention` (transformer.py:788), then, once the literal was
        corrected, `ValueError: Unrecognized keyword arguments passed to
        MultiModalFusion: {'embed_dim': 64, 'num_heads': 1}`
        (multimodal_fusion.py:150)."""
        model = _modern()
        assert isinstance(model, NanoVLM)

    def test_fusion_layer_uses_the_real_signature(self):
        """The fix must converge on `dim` + `attention_config`, the shape the
        D-002 anchor names, not on a translation layer that strips the ghost
        keys. Pins the resulting layer's identity, not just that it built."""
        model = _modern()
        assert model.fusion_layer.dim == EMBED_DIM
        assert "embed_dim" not in model.fusion_config
        assert "num_heads" not in model.fusion_config
        assert model.fusion_config["attention_config"]["num_heads"] == EMBED_DIM // 64

    def test_forward_logits(self):
        out = _model()(_inputs(), training=False)
        arr = keras.ops.convert_to_numpy(out)
        assert arr.shape[0] == 2 and arr.shape[-1] == VOCAB
        assert not np.any(np.isnan(arr)) and not np.any(np.isinf(arr))

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _inputs()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "modern_nano_vlm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(
            before, after, atol=1e-4,
            err_msg="the modern NanoVLM differs after .keras round-trip")

    def test_no_argument_factory_default_is_cross_attention(self):
        """The default literal itself, independent of any forward pass: the
        row's own `fusion_strategy`, with no override passed.

        Pre-fix this read `'tensor_fusion'` (model.py:836)."""
        assert NanoVLM.MODEL_VARIANTS["modern"]["fusion_config"]["fusion_strategy"] == (
            "cross_attention")
        model = _modern()
        assert model.fusion_config["fusion_strategy"] == "cross_attention"

    def test_no_argument_factory_forward_passes_to_logits(self):
        """No `fusion_strategy` override anywhere -- the path a first-time caller
        takes. This was impossible before the default changed: it died inside a
        `ConcatOp` on vision 197 vs text 8."""
        model = _modern()
        arr = keras.ops.convert_to_numpy(model(_inputs(), training=False))
        assert arr.shape[0] == 2 and arr.shape[-1] == VOCAB
        assert not np.any(np.isnan(arr)) and not np.any(np.isinf(arr))

    def test_tensor_fusion_raises_named_error_on_unequal_sequence_lengths(self):
        """`tensor_fusion` is still selectable, and still cannot fuse vision 197
        against text 8 — but it now says so.

        Formerly a bare strict xfail. Without the guard in
        `MultiModalFusion._call_tensor_fusion` this raises
        `tensorflow...InvalidArgumentError` ("ConcatOp : Dimension 1 in both
        shapes must be equal"), which is NOT a `ValueError` and never mentions a
        sequence length, so neither assertion below can pass vacuously."""
        with pytest.raises(ValueError, match="same sequence length") as excinfo:
            _model("tensor_fusion")(_inputs(), training=False)
        message = str(excinfo.value)
        assert "cross_attention" in message, (
            f"the raise must point at a strategy that works; got: {message}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
