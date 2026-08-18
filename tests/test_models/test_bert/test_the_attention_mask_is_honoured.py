"""BERT's central claim at this level: the attention mask actually masks.

Why this file exists
--------------------
`test_bert.py::test_forward_pass_with_attention_mask` builds a mask, runs the
model and asserts the OUTPUT SHAPE. Replace every attention block with an
identity and that test still passes -- the mask is never shown to change
anything, and a mask that is silently dropped (built, forwarded, and then not
added to the attention logits) is invisible.

The instrument is the both-ways pair from
`test_byte_latent_transformer::test_future_byte_does_not_change_the_past` +
`::test_the_model_still_responds_at_and_after_the_perturbation`. One half alone
is not a guard: "nothing changed" is also what a model that ignores its input
entirely produces.

MEASURED 2026-08-18 (2 layers, 4 heads, hidden 32, sequence 8, positions 5..7
masked out): perturbing the tokens at the masked positions moves the kept
positions by EXACTLY 0.0 and moves the masked positions themselves by 2.242.
Both numbers are load bearing -- the first is the claim, the second is the proof
that the perturbation reached the model at all.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.bert.bert import BERT


SEQ = 8
KEEP = 5  # positions 0..4 are visible, 5..7 are masked out


def _model() -> BERT:
    keras.utils.set_random_seed(2)
    model = BERT(
        vocab_size=100,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    model(keras.ops.zeros((1, SEQ), dtype="int32"), training=False)
    return model


@pytest.fixture(scope="module")
def perturbation():
    """(model, hidden_state_fn, ids, bumped) with the masked tail changed."""
    model = _model()
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 100, size=(2, SEQ)).astype("int32")
    mask = np.ones((2, SEQ), dtype="int32")
    mask[:, KEEP:] = 0
    bumped = ids.copy()
    bumped[:, KEEP:] = (bumped[:, KEEP:] + 13) % 100
    assert not np.array_equal(ids, bumped)

    def hidden(token_ids):
        out = model(
            {
                "input_ids": keras.ops.convert_to_tensor(token_ids),
                "attention_mask": keras.ops.convert_to_tensor(mask),
            },
            training=False,
        )
        return np.asarray(keras.ops.convert_to_numpy(out["last_hidden_state"]))

    return hidden, ids, bumped


class TestAttentionMaskIsHonoured:
    def test_masked_tokens_do_not_reach_the_visible_positions(self, perturbation):
        hidden, ids, bumped = perturbation
        delta = float(
            np.max(np.abs(hidden(ids)[:, :KEEP] - hidden(bumped)[:, :KEEP]))
        )
        # Bit-identical, not "small": an attention weight of exactly zero on a
        # masked key contributes exactly nothing. Measured 0.0.
        assert delta == 0.0, (
            f"changing the tokens at masked positions {KEEP}..{SEQ - 1} moved "
            f"the VISIBLE positions by {delta:.3e}. The attention mask is not "
            f"being applied -- shape-only mask tests cannot see this."
        )

    def test_the_perturbation_reached_the_model_at_all(self, perturbation):
        """The other half of the pair: without this, 0.0 above proves nothing."""
        hidden, ids, bumped = perturbation
        delta = float(
            np.max(np.abs(hidden(ids)[:, KEEP:] - hidden(bumped)[:, KEEP:]))
        )
        # Measured 2.242. A model that ignored its input would score 0.0 here
        # and would still pass the test above.
        assert delta > 1e-3, (
            f"changing the tokens at the masked positions changed nothing even "
            f"AT those positions ({delta:.3e}); the previous test's 0.0 is then "
            f"an artefact of an inert model, not evidence of masking"
        )
