"""
RED-proof for the two ghost-API defects in `create_modern_nanovlm`.

Nothing in `tests/` called this factory before this module, which is why both
defects survived: `vision_config['attention_type'] = 'differential_attention'`
(the attention factory registers `'differential'`) raised inside
`_create_vision_encoder` and masked the second defect, `fusion_config`'s ghost
`embed_dim`/`num_heads` keys, which `MultiModalFusion` rejects at
`keras.layers.Layer.__init__`.

`test_model.py` covers `create_nanovlm` only — a different factory with
different (already correct) variant configs — so these cases are kept here.

`create_modern_nanovlm` hardcodes `img_size=224`, `patch_size=16` and
`depth=12`; only `vocab_size`, `embed_dim` and `fusion_strategy` are callable
knobs, so "tiny" here means a narrow model, not a shallow one.

Fixing the two ghost-API defects made a THIRD defect reachable for the first
time and it is deliberately NOT fixed here: this factory's default
`fusion_strategy='tensor_fusion'` cannot run on this model at all, because
`MultiModalFusion._call_tensor_fusion` concatenates the modalities on the
feature axis and so requires equal sequence lengths, while NanoVLM always
feeds vision (197) and text (T) streams of different length. That is a design
default, not a wrong literal, so it is pinned by `test_default_tensor_fusion_
strategy_is_unrunnable` (strict xfail — it flips to a failure the moment the
default is changed or the strategy is made length-agnostic) and the
construct/forward/round-trip cases run under `cross_attention`, the default
`create_nanovlm` uses.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.nano_vlm.model import create_modern_nanovlm, NanoVLM

VOCAB = 64
EMBED_DIM = 64  # -> num_heads = EMBED_DIM // 64 = 1


def _model(fusion_strategy="cross_attention"):
    return create_modern_nanovlm(vocab_size=VOCAB, embed_dim=EMBED_DIM,
                                 fusion_strategy=fusion_strategy)


def _inputs(batch=2, seq=8):
    rng = np.random.default_rng(0)
    return {
        "images": rng.random((batch, 224, 224, 3)).astype("float32"),
        "text_tokens": rng.integers(0, VOCAB, (batch, seq)).astype("int32"),
    }


class TestCreateModernNanoVLM:

    def test_factory_construction(self):
        """The documented call — no overrides at all, so this is the literal
        path both defects lived on.

        Pre-fix this raised `ValueError: Unknown attention type:
        differential_attention` (transformer.py:788), then, once the literal was
        corrected, `ValueError: Unrecognized keyword arguments passed to
        MultiModalFusion: {'embed_dim': 64, 'num_heads': 1}`
        (multimodal_fusion.py:150)."""
        model = create_modern_nanovlm(vocab_size=VOCAB, embed_dim=EMBED_DIM)
        assert isinstance(model, NanoVLM)

    def test_fusion_layer_uses_the_real_signature(self):
        """The fix must converge on `dim` + `attention_config`, the shape the
        D-002 anchor names, not on a translation layer that strips the ghost
        keys. Pins the resulting layer's identity, not just that it built."""
        model = create_modern_nanovlm(vocab_size=VOCAB, embed_dim=EMBED_DIM)
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
            err_msg="create_modern_nanovlm model differs after .keras round-trip")

    @pytest.mark.xfail(strict=True, reason=(
        "Third defect, newly reachable once the two ghost-API defects were "
        "fixed and NOT repaired here: this factory's default "
        "fusion_strategy='tensor_fusion' concatenates the modalities on the "
        "feature axis (multimodal_fusion.py:958), so it needs equal sequence "
        "lengths, while NanoVLM always feeds vision 197 vs text T. Changing a "
        "default is a design decision, not a wrong-literal fix."))
    def test_default_tensor_fusion_strategy_is_unrunnable(self):
        _model("tensor_fusion")(_inputs(), training=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
