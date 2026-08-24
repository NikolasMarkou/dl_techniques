"""DistilBERT's three defining structural claims, each pinned by an assertion.

The module docstring of ``models/distilbert/model.py`` makes exactly three
structural claims that distinguish this package from ``models/bert/``:

1. **The compression is entirely in the DEPTH.** ``base`` is 6 layers where
   BERT-base is 12, while ``hidden_size`` / ``num_heads`` / ``intermediate_size``
   stay at BERT's 768 / 12 / 3072. The docstring argues this is deliberate: width
   is what a student initialized from the teacher can inherit.
2. **No token type embeddings** — the shared ``BertEmbeddings`` is constructed
   with ``use_token_type_embeddings=False``, because DistilBERT trains on
   single-segment input.
3. **No pooler** — the model emits exactly ``{"last_hidden_state",
   "attention_mask"}``. The pooler served BERT's NSP objective, which
   distillation drops.

MEASURED 2026-08-21, injecting each claim's negation into the real source and
running ``tests/test_models/test_distilbert/`` (baseline **32 passed**):

| Injection | Directory result | Verdict |
|---|---|---|
| ``MODEL_VARIANTS['base']['num_layers'] = 12`` | **32 passed** | claim 1 was **unguarded** |
| ``use_token_type_embeddings=True`` | 23 failed, 5 errors | caught, but as a ``ValueError`` CRASH from the embedding factory ("type_vocab_size is required..."), not as a judgement |
| an extra ``"pooler_output"`` output key | 3 failed | caught, incidentally, by the smoke contract and the single-key-dict predict test |

So one of the three was genuinely deletable, and the other two were held up by
side effects rather than by anything that states the claim. A crash is a fragile
guard: it disappears the moment someone supplies the missing argument. This
module states all three directly.

``test_the_width_is_not_also_compressed`` is the twin of claim 1 that keeps it
from being satisfiable by a model that simply shrank everything: the point of
DistilBERT is that depth alone moved.

Costs no model build for claims 1 -- ``MODEL_VARIANTS`` is a class attribute --
so those arms are device-independent by construction. Claims 2 and 3 build one
tiny model each, seeded, at ``dropout=0.0``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.distilbert.model import DistilBERT

# BERT-base's own geometry, the reference DistilBERT compresses against.
BERT_BASE = {"hidden_size": 768, "num_layers": 12, "num_heads": 12,
             "intermediate_size": 3072}
SEED = 20260821


def _tiny_model():
    keras.utils.set_random_seed(SEED)
    return DistilBERT(
        vocab_size=256, hidden_size=64, num_layers=2, num_heads=4,
        intermediate_size=128, max_position_embeddings=32,
        dropout_rate=0.0, attention_dropout_rate=0.0,
    )


class TestTheThreeStructuralClaims:

    def test_the_depth_is_halved_relative_to_bert_base(self):
        """Claim 1. This is the one that was deletable with 32 tests green."""
        base = DistilBERT.MODEL_VARIANTS["base"]
        assert base["num_layers"] == BERT_BASE["num_layers"] // 2 == 6, (
            f"DistilBERT-base is {base['num_layers']} layers; the package's "
            f"defining claim is half of BERT-base's {BERT_BASE['num_layers']}"
        )

    def test_the_width_is_not_also_compressed(self):
        """Claim 1's twin: depth ALONE moved, so a teacher layer stays copyable."""
        base = DistilBERT.MODEL_VARIANTS["base"]
        for key in ("hidden_size", "num_heads", "intermediate_size"):
            assert base[key] == BERT_BASE[key], (
                f"DistilBERT-base narrowed {key} to {base[key]} (BERT-base: "
                f"{BERT_BASE[key]}); the package claims width is UNCHANGED, "
                "which is what makes teacher layers copyable"
            )

    def test_there_are_no_token_type_embeddings(self):
        """Claim 2, stated rather than left to a constructor crash."""
        model = _tiny_model()
        emb = model.embeddings
        assert emb.use_token_type_embeddings is False, (
            "DistilBERT built BertEmbeddings WITH segment embeddings; it trains "
            "on single-segment input and the paper removes them"
        )
        assert getattr(emb, "token_type_embeddings", None) is None

        paths = [w.path for w in model.weights]
        assert not any("token_type" in p for p in paths), (
            f"a token-type weight reached the model: "
            f"{[p for p in paths if 'token_type' in p]}"
        )

    def test_the_output_is_exactly_two_keys_and_carries_no_pooler(self):
        """Claim 3. `"last_hidden_state" in out` is satisfied by a pooled model."""
        model = _tiny_model()
        ids = keras.ops.convert_to_tensor(
            np.random.default_rng(SEED).integers(0, 256, size=(2, 8)), dtype="int32"
        )
        out = model({"input_ids": ids}, training=False)

        assert set(out) == {"last_hidden_state", "attention_mask"}, (
            f"DistilBERT is a pure foundation model with no head and no pooler; "
            f"it emitted {sorted(out)}"
        )
        assert not hasattr(model, "pooler"), "a pooler sub-layer was attached"
        assert not any("pooler" in w.path for w in model.weights)

        # Control: the two keys carry real content, so the set equality above is
        # not passing over a pair of empty placeholders.
        arr = keras.ops.convert_to_numpy(out["last_hidden_state"])
        assert arr.shape == (2, 8, 64) and bool(np.all(np.isfinite(arr)))
