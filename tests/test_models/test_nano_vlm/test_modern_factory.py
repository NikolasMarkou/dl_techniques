"""
The `'modern'` NanoVLM row: RMSNorm, SwiGLU, differential attention,
factorized word embeddings.

Until plan-2026-10-14T062616-81938e81 (D-006, D-037) this configuration was a
second factory, `create_modern_nanovlm(vocab_size, embed_dim, **kw)`, built
from hard-coded dicts that bypassed `MODEL_VARIANTS`. It is now the row
`NanoVLM.MODEL_VARIANTS['modern']`, built by `NanoVLM.from_variant('modern')`;
the factory is deleted.

Nothing in `tests/` called the old factory before this module, which is why
ghost-API defects survived in it: `vision_config['attention_type'] =
'differential_attention'` (the attention factory registers `'differential'`)
raised inside `_create_vision_encoder`. Its fusion-layer defects (ghost
`embed_dim`/`num_heads` keys, a `tensor_fusion` default that could not run on
unequal stream lengths) went with the fusion layer itself in step 9.1
(D-040): the model is now the reference prefix-into-one-causal-decoder layout.

Step 9.1 also found that the row's `text_config['positional_type'] = 'rope'`
was a dead knob: `TextDecoder` builds positional encodings for `'learned'`
and `'sincos'` only, so `'rope'` silently ran the text tower with NO
positional encoding. `TextDecoder` now refuses an unknown `positional_type`
(`tests/test_layers/test_transformers/test_the_decoder_runs_on_precomputed_embeddings.py`),
and the row uses the decoder's default learned positions.

The row fixes `img_size=224`, `patch_size=16` and `depth=12`. "Tiny" here
means a narrow model, not a shallow one: `embed_dim=64` (so 64 // 64 = 1
head) is laid over the row with the one-level merge.
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
        **kwargs,
    )


def _inputs(batch=2, seq=8):
    rng = np.random.default_rng(0)
    return {
        "images": rng.random((batch, 224, 224, 3)).astype("float32"),
        "text_tokens": rng.integers(0, VOCAB, (batch, seq)).astype("int32"),
    }


class TestTheModernVariant:

    def test_factory_construction(self):
        """The row with only its width narrowed.

        Pre-fix this raised `ValueError: Unknown attention type:
        differential_attention` (transformer.py:788)."""
        model = _modern()
        assert isinstance(model, NanoVLM)
        assert model.vision_config["attention_type"] == "differential"
        assert model.text_config["embedding_type"] == "factorized"

    def test_the_text_decoder_carries_learned_positions(self):
        """The row's decoder has a positional table covering the vision prefix
        and the whole text; under the old `'rope'` it had none."""
        assert "positional_type" not in NanoVLM.MODEL_VARIANTS["modern"]["text_config"]
        model = _modern()
        assert model.text_decoder.positional_type == "learned"
        assert model.text_decoder.max_seq_len == model.num_vision_tokens + 1024

    def test_forward_logits(self):
        model = _modern()
        out = model(_inputs(), training=False)
        arr = keras.ops.convert_to_numpy(out)
        assert arr.shape == (2, model.num_vision_tokens + 8, VOCAB)
        assert not np.any(np.isnan(arr)) and not np.any(np.isinf(arr))

    def test_keras_round_trip(self, tmp_path):
        model = _modern()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
