"""F-36 RED proof: exercise NAM's operand derivation THROUGH ``NAMCell.call``.

Every existing number-assembly test bypasses the mechanism it is supposed to
cover. ``test_deterministic_number_assembly`` explicitly declines to assert
values ("We can't guarantee reduction points to the right place (untrained)")
and degenerates into a NaN/Inf check; ``test_deterministic_assembly_exact`` and
``test_deterministic_assembly_large_numbers`` call the module-level
``_assemble_number_from_tokens`` directly with **hand-written** masks for a
single-operator expression. Nothing feeds a two-operator or a parenthesised
expression through ``NAMCell.call``, where ``op_pos = argmax(reduction_weights)``
and the left/right digit masks are DERIVED from it.

What can honestly be pinned on an untrained model
-------------------------------------------------
``op_pos`` is a learned argmax, so the operand pair is not predictable at random
weights — and asserting a specific value would be a test of the initializer, not
of the mechanism. What IS weight-independent is the RULE: whatever position the
cell selects, the operands are the digits strictly left of it and strictly right
of it, each concatenated positionally into ONE number. So each test enumerates
the operand pair the rule yields at every candidate position and asserts:

1. the pair the cell actually produced is one of them (the derivation really is
   the documented split — a dead derivation is not), and
2. **no** candidate position yields the operands that a chained evaluation would
   need. This is the documented-scope half (Assumption A1 / decisions.md D-002):
   NAM is single-operator and integer-only, and ``"1 + 2 * 3"`` cannot reach 7
   at any weights because the digits of the two sub-expressions get concatenated
   rather than reduced.

These tests pin the CURRENT behaviour honestly. They are not a wish; step 17
corrects the documentation to match them.

What these tests DELIBERATELY do not claim
------------------------------------------
MEASURED (dead-component injection, ``op_pos`` forced to a constant 0 so the
argmax over ``reduction_weights`` is dead): all 9 tests below stay GREEN. That
is correct and it is stated here rather than papered over — position 0 is a
legitimate candidate position, so the split rule still holds at it, and the only
way to reject a constant ``op_pos`` would be to assert the specific position an
UNTRAINED argmax happens to select, which tests the initializer rather than the
mechanism. The injection this file does catch is the one that matters for F-35 /
F-36: zeroing the derived digit masks makes 7 of 9 go RED.
"""

import numpy as np
import pytest

from dl_techniques.models.nam.model import NAM
from dl_techniques.models.nam.config import NAMConfig
from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer

MAX_LEN = 16


@pytest.fixture
def tiny_model():
    return NAM(config=NAMConfig(
        hidden_size=32,
        num_heads=4,
        num_tree_layers=1,
        intermediate_size=64,
        memory_size=8,
        num_read_heads=2,
        max_expression_len=MAX_LEN,
        halt_max_steps=4,
        hidden_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    ))


def _candidate_operand_pairs(ids):
    """Every ``(left, right)`` the cell's own split rule can yield.

    Written from the DOCUMENTED rule — digits are token ids 4..13, the operand
    is the positional concatenation of the digits strictly on one side of the
    selected position — not read back out of ``cell.py``.
    """
    digits = [(i, int(t) - 4) for i, t in enumerate(ids) if 4 <= int(t) <= 13]

    def assemble(items):
        v = 0.0
        for _, d in items:
            v = v * 10.0 + d
        return v

    pairs = []
    for pos in range(len(ids)):
        left = assemble([x for x in digits if x[0] < pos])
        right = assemble([x for x in digits if x[0] > pos])
        pairs.append((pos, left, right))
    return pairs


def _observed_pair(model, expression):
    tokenizer = ArithmeticTokenizer(max_len=MAX_LEN)
    ids = tokenizer.encode_batch([expression])
    batch = {"input_ids": ids}
    carry = model.initial_carry(batch)
    _, out = model(carry, batch, training=False)
    left = float(np.asarray(out["step_left_val"]).reshape(-1)[0])
    right = float(np.asarray(out["step_right_val"]).reshape(-1)[0])
    return ids[0], left, right


class TestOperandsComeFromTheDerivedOperatorPosition:
    """The cell's operands must obey its own split rule at some position."""

    @pytest.mark.parametrize(
        "expression", ["1 + 2", "1 + 2 * 3", "( 1 + 2 ) * 3", "12 - 3"]
    )
    def test_the_pair_is_one_the_split_rule_can_produce(
        self, tiny_model, expression
    ):
        ids, left, right = _observed_pair(tiny_model, expression)
        candidates = [(l, r) for _, l, r in _candidate_operand_pairs(ids)]

        assert (left, right) in candidates, (
            f"({left}, {right}) is not a pair the documented left/right split "
            f"can yield for {expression!r}; the operand derivation is not the "
            f"one this test describes. candidates={sorted(set(candidates))}"
        )

    def test_the_rule_is_discriminating(self, tiny_model):
        """Anti-vacuity: the candidate set must not be everything.

        If every conceivable pair were a candidate, the assertion above would
        be unfalsifiable. ``(0.0, 0.0)`` — what a dead mask derivation emits —
        must NOT be in the set.
        """
        ids, _, _ = _observed_pair(tiny_model, "1 + 2 * 3")
        candidates = {(l, r) for _, l, r in _candidate_operand_pairs(ids)}

        assert (0.0, 0.0) not in candidates, (
            "a dead (all-zero) mask derivation would be indistinguishable "
            "from a live one on this expression"
        )
        assert len(candidates) > 1


class TestMultiOperatorExpressionsAreOutOfScope:
    """Digits of a second sub-expression are CONCATENATED, never reduced.

    decisions.md D-002 / Assumption A1: operands are assembled exclusively from
    raw ``token_ids`` and ``NAM.call`` re-reads ``token_ids`` unchanged on every
    ACT step, so step N's result cannot become step N+1's operand at ANY
    weights. These tests are the measurement behind that ruling.
    """

    def test_one_plus_two_times_three_cannot_reach_seven(self, tiny_model):
        ids, left, right = _observed_pair(tiny_model, "1 + 2 * 3")
        candidates = _candidate_operand_pairs(ids)

        # Chained evaluation would need (1, 6) [1 + (2*3)] or (3, 3) [(1+2)*3].
        for want in ((1.0, 6.0), (3.0, 3.0)):
            assert want not in [(l, r) for _, l, r in candidates], (
                f"operands {want} ARE reachable — the multi-step claim this "
                "test rules out may be implementable after all; re-open F-35"
            )

        # And the concatenation is explicit: at the '+' position the right
        # operand is the digits '2' and '3' run together, i.e. 23.
        plus_pos = int(np.argmax(np.asarray(ids) == 14))
        _, l_at_plus, r_at_plus = next(
            c for c in candidates if c[0] == plus_pos
        )
        assert (l_at_plus, r_at_plus) == (1.0, 23.0)

        assert (left, right) in [(l, r) for _, l, r in candidates]

    def test_a_parenthesised_expression_ignores_its_parentheses(
        self, tiny_model
    ):
        ids, left, right = _observed_pair(tiny_model, "( 1 + 2 ) * 3")
        candidates = _candidate_operand_pairs(ids)

        # '(' and ')' are ids 18/19 — not digits, so they neither delimit an
        # operand nor group one. At the '*' the left operand is '1' and '2'
        # concatenated: 12, not the parenthesised 3.
        star_pos = int(np.argmax(np.asarray(ids) == 16))
        _, l_at_star, r_at_star = next(
            c for c in candidates if c[0] == star_pos
        )
        assert (l_at_star, r_at_star) == (12.0, 3.0)
        assert (3.0, 3.0) not in [(l, r) for _, l, r in candidates]
        assert (left, right) in [(l, r) for _, l, r in candidates]


class TestDecimalsAreOutOfScope:
    """``DOT_ID = 20`` is tokenized but has no fractional branch (F-37).

    ``is_digit`` is ``4 <= id <= 13``, so ``"1.5 + 2"`` assembles the left
    operand from the digits '1' and '5' as 15, and no validity flag is raised.
    """

    def test_one_point_five_assembles_as_fifteen(self, tiny_model):
        ids, left, right = _observed_pair(tiny_model, "1.5 + 2")
        candidates = _candidate_operand_pairs(ids)

        plus_pos = int(np.argmax(np.asarray(ids) == 14))
        _, l_at_plus, r_at_plus = next(
            c for c in candidates if c[0] == plus_pos
        )
        assert (l_at_plus, r_at_plus) == (15.0, 2.0), (
            "the decimal point stopped being dropped — F-37 may be fixed; "
            "re-check the documented scope before changing this test"
        )
        assert 1.5 not in [l for _, l, _ in candidates]
        assert (left, right) in [(l, r) for _, l, r in candidates]

    def test_the_dot_really_is_tokenized(self):
        """Control: the scope limit is in ASSEMBLY, not in the tokenizer."""
        tokenizer = ArithmeticTokenizer(max_len=MAX_LEN)
        ids = tokenizer.encode("1.5 + 2")
        assert 20 in ids, "DOT_ID is not tokenized; this test measures nothing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
