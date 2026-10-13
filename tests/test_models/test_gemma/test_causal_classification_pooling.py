"""RED proof for C-29 (plan-2026-08-14T233721-d4f9beb2, D-029) — Gemma 3 arm.

The retired `create_gemma3_classification` factory used to default to
``pooling_strategy="cls"``. Gemma 3's blocks are causally masked (sliding-window
and full layers alike), so position 0 attends only to itself and the pooled
vector — hence every logit — was a deterministic function of the FIRST TOKEN ID
ALONE. Nothing raised.

A classifier is now the class itself (``include_top=True`` with a
``TEXT_CLASSIFICATION`` task), whose head pools ``"last"`` unless
``head_config_overrides`` names another ``pooling_type``; the claims below are
pinned on the class.

The qwen arm of this proof lives at
``tests/test_models/test_qwen/test_causal_classification_pooling.py``; the two
are kept beside their own packages rather than merged into one cross-package
module.
"""

import keras
import numpy as np

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.gemma.gemma3 import Gemma3

# A `Gemma3` small enough to build on CPU. `layer_types` is spelled out so the
# probe covers both a sliding-window and a full-attention block; both are
# causal, which is the property under test.
GEMMA3_CONFIG = dict(
    vocab_size=64,
    hidden_size=32,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    ffn_hidden_size=64,
    max_seq_len=16,
    sliding_window_size=4,
    layer_types=["sliding_window", "full_attention"],
    dropout_rate=0.0,
)

IDS = np.array([[5, 7, 9, 11, 13, 2, 0, 0]], dtype="int32")
MASK = np.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype="int32")
NON_FIRST_INDEX = 3
FIRST_INDEX = 0
PAD_INDEX = 7
NEW_TOKEN = 21


#: Three labels, no dropout, so a probe sees the backbone only.
CLASSIFICATION = NLPTaskConfig(
    name="classification", task_type=NLPTaskType.TEXT_CLASSIFICATION,
    num_classes=3, dropout_rate=0.0,
)


def _logits(model, ids):
    return np.asarray(model.predict(
        {"input_ids": ids, "attention_mask": MASK}, verbose=0))


def _mutate(index, token=NEW_TOKEN):
    mutated = IDS.copy()
    mutated[0, index] = token
    return mutated


def _build(strategy=None):
    """The headed class; ``strategy=None`` keeps the class's own default."""
    keras.utils.set_random_seed(0)
    return Gemma3(
        **GEMMA3_CONFIG,
        include_top=True,
        task_config=CLASSIFICATION,
        head_config_overrides=(
            {"pooling_type": strategy} if strategy is not None else None),
    )


class TestGemma3CausalClassificationPooling:
    """The defect, its control, and the shipped default."""

    def test_default_pooling_is_last_not_cls(self):
        default = _build().task_head.pooling_type
        assert default == "last", (
            "the headed Gemma3's default pooling regressed to "
            f"{default!r}; under a causal mask 'cls' pools a position that "
            "attended only to itself (see decisions.md D-029)"
        )

    def test_non_first_token_moves_the_logits_under_the_default(self):
        """THE defect. At HEAD this delta was exactly 0.000e+00."""
        model = _build()
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(NON_FIRST_INDEX))
        delta = float(np.max(np.abs(moved - base)))
        assert delta > 1e-6, (
            f"changing token {NON_FIRST_INDEX} (inside the attention mask) "
            f"moved the per-example logits by {delta:.3e}; the classifier is "
            "not reading the sequence"
        )

    def test_cls_pooling_is_still_blind_to_a_non_first_token(self):
        """ANTI-VACUITY control: `cls` really is first-token-only here."""
        model = _build("cls")
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(NON_FIRST_INDEX))
        assert np.array_equal(moved, base), (
            "'cls' pooling under a causal mask must be bit-identical when a "
            "non-first token changes; if it is not, Gemma 3's causal mask has "
            "been weakened"
        )

    def test_cls_pooling_still_tracks_the_first_token(self):
        """Second anti-vacuity arm: the `cls` probe is not dead by construction."""
        model = _build("cls")
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(FIRST_INDEX))
        delta = float(np.max(np.abs(moved - base)))
        assert delta > 1e-6, (
            "the 'cls' probe measured no movement even from the FIRST token, "
            f"delta {delta:.3e} — the probe itself is broken, so the "
            "zero-delta control above proves nothing"
        )

    def test_padding_is_not_pooled_under_the_default(self):
        """`last` gathers the last KEPT position, not `inputs[:, -1, :]`."""
        model = _build()
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(PAD_INDEX))
        assert np.array_equal(moved, base), (
            f"changing padded token {PAD_INDEX} moved the logits; the pooler "
            "is gathering a padded position instead of the last real one"
        )
