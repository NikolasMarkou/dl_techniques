"""RED proof for C-29 (plan-2026-08-14T233721-d4f9beb2, D-029).

`create_qwen3_classification` and `create_qwen3_next_classification` used to
default to ``pooling_strategy="cls"``. Both backbones are strictly causally
masked, so position 0 attends only to itself: the pooled vector — and therefore
every logit — was a deterministic function of the FIRST TOKEN ID ALONE. Nothing
raised; loss still decreased, accuracy just plateaued at the first-token prior.

Each assertion below pins a per-example logit vector, never a shape and never a
mere "output changed somewhere". The mutated token is always inside the region
kept by ``attention_mask``, because a mutation outside it is invisible for a
reason that has nothing to do with pooling.
"""

import inspect

import keras
import numpy as np
import pytest

from dl_techniques.models.qwen.qwen3 import create_qwen3_classification
from dl_techniques.models.qwen.qwen3_next import (
    create_qwen3_next_classification,
)

# A backbone small enough to build on CPU in a test. `Qwen3` takes a
# `moe_layers` list; `Qwen3Next` does not (its GDN/GA block layout is fixed), so
# the two configs are spelled separately rather than shared and patched.
QWEN3_CONFIG = dict(
    vocab_size=64,
    hidden_size=32,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    moe_layers=[],
    moe_intermediate_size=64,
    max_seq_len=16,
    dropout_rate=0.0,
)
QWEN3_NEXT_CONFIG = {
    k: v for k, v in QWEN3_CONFIG.items() if k != "moe_layers"
}
QWEN3_NEXT_CONFIG["num_experts"] = 2
QWEN3_NEXT_CONFIG["num_experts_per_tok"] = 1

# Eight positions, the last two of which are right padding. Index 3 is the
# token every "non-first" assertion moves: it is strictly inside the kept
# region and it is neither the first nor the last real position.
IDS = np.array([[5, 7, 9, 11, 13, 2, 0, 0]], dtype="int32")
MASK = np.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype="int32")
NON_FIRST_INDEX = 3
FIRST_INDEX = 0
PAD_INDEX = 7
NEW_TOKEN = 21

FACTORIES = [
    pytest.param(create_qwen3_classification, QWEN3_CONFIG, id="qwen3"),
    pytest.param(
        create_qwen3_next_classification, QWEN3_NEXT_CONFIG, id="qwen3_next"
    ),
]


def _logits(model, ids):
    return np.asarray(model.predict([ids, MASK], verbose=0))


def _mutate(index, token=NEW_TOKEN):
    mutated = IDS.copy()
    mutated[0, index] = token
    return mutated


def _build(factory, config, strategy):
    keras.utils.set_random_seed(0)
    return factory(
        config,
        num_labels=3,
        pooling_strategy=strategy,
        classifier_dropout_rate=0.0,
    )


@pytest.mark.parametrize("factory,config", FACTORIES)
class TestCausalClassificationPooling:
    """The defect, its control, and the shipped default."""

    def test_default_pooling_is_last_not_cls(self, factory, config):
        """The shipped default is the whole finding; pin it directly."""
        default = inspect.signature(factory).parameters[
            "pooling_strategy"
        ].default
        assert default == "last", (
            "the classification factory's default pooling_strategy regressed "
            f"to {default!r}; under a causal mask 'cls' pools a position that "
            "attended only to itself (see decisions.md D-029)"
        )

    def test_non_first_token_moves_the_logits_under_the_default(
        self, factory, config
    ):
        """THE defect. At HEAD this delta was exactly 0.000e+00."""
        model = _build(factory, config, "last")
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(NON_FIRST_INDEX))
        delta = float(np.max(np.abs(moved - base)))
        assert delta > 1e-6, (
            f"changing token {NON_FIRST_INDEX} (inside the attention mask) "
            f"moved the per-example logits by {delta:.3e}; the classifier is "
            "not reading the sequence"
        )

    def test_cls_pooling_is_still_blind_to_a_non_first_token(
        self, factory, config
    ):
        """The mechanism, kept explicit: `cls` really is first-token-only.

        This is the ANTI-VACUITY control for the test above. If this one ever
        starts failing, the probe stopped isolating pooling and the assertion
        above no longer proves what it claims.
        """
        model = _build(factory, config, "cls")
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(NON_FIRST_INDEX))
        assert np.array_equal(moved, base), (
            "'cls' pooling under a causal mask must be bit-identical when a "
            "non-first token changes; if it is not, the backbone's causal "
            "mask has been weakened"
        )

    def test_cls_pooling_still_tracks_the_first_token(self, factory, config):
        """Second anti-vacuity arm: the `cls` probe is not dead by construction."""
        model = _build(factory, config, "cls")
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(FIRST_INDEX))
        delta = float(np.max(np.abs(moved - base)))
        assert delta > 1e-6, (
            "the 'cls' probe measured no movement even from the FIRST token, "
            f"delta {delta:.3e} — the probe itself is broken, so the "
            "zero-delta control above proves nothing"
        )

    def test_padding_is_not_pooled_under_the_default(self, factory, config):
        """`last` gathers the last KEPT position, not `inputs[:, -1, :]`."""
        model = _build(factory, config, "last")
        base = _logits(model, IDS)
        moved = _logits(model, _mutate(PAD_INDEX))
        assert np.array_equal(moved, base), (
            f"changing padded token {PAD_INDEX} moved the logits; the pooler "
            "is gathering a padded position instead of the last real one"
        )
