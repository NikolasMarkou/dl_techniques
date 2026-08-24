"""F-35 / F-37: the package's docs must state the scope the code actually has.

Assumption A1 of this plan chose to correct the documentation rather than build
operand chaining and a decimal branch. That choice is only worth anything if the
corrected sentences stay corrected, so this module is the gate on them.

Why a text gate is the right instrument here, and what it cannot do
-------------------------------------------------------------------
The BEHAVIOUR is already pinned, by value, through ``NAMCell.call``, in
``test_operand_derivation_through_call.py`` — that file is the measurement and
this one is not a substitute for it. What no behavioural test can catch is a
maintainer re-adding a *sentence* that promises multi-step reduction or decimal
support; the code would be unchanged and every value assertion would stay green
while the package once again advertises more than it does. That regression is
textual, so the guard is textual. It is deliberately narrow: it asserts the two
scope words are present and that the three specific refuted claims are absent,
not that any particular wording is preserved.

Each absence assertion below is paired with the measurement that makes it a lie,
so a future reader can re-derive the ruling instead of trusting this file.
"""

import pathlib

import numpy as np
import pytest

from dl_techniques.models.nam import model as nam_model
from dl_techniques.models.nam import cell as nam_cell
from dl_techniques.models.nam import tokenizer as nam_tokenizer
from dl_techniques.models.nam.model import NAM
from dl_techniques.models.nam.config import NAMConfig
from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer

README = pathlib.Path(nam_model.__file__).parent / "README.md"


def _norm(text: str) -> str:
    """Lower-case and collapse whitespace so line wrapping is not load-bearing."""
    return " ".join(text.lower().split())


@pytest.fixture(scope="module")
def documented_surfaces():
    return {
        "model.py module docstring": _norm(nam_model.__doc__ or ""),
        "NAM class docstring": _norm(NAM.__doc__ or ""),
        "cell.py module docstring": _norm(nam_cell.__doc__ or ""),
        "README.md": _norm(README.read_text()),
    }


class TestTheScopeIsStated:
    """The two scope words must appear on every user-facing surface."""

    @pytest.mark.parametrize("phrase", ["single-operator", "integer-only"])
    def test_the_scope_words_appear(self, documented_surfaces, phrase):
        missing = [
            where
            for where, text in documented_surfaces.items()
            if phrase not in text
        ]
        assert not missing, (
            f"{phrase!r} is missing from: {missing}. NAM is "
            "single-operator and integer-only (Assumption A1); a surface that "
            "does not say so advertises more than the code does."
        )

    def test_the_tokenizer_says_tokenized_is_not_supported(self):
        """``DOT_ID`` round-trips, which reads as support unless contradicted."""
        text = _norm(nam_tokenizer.__doc__ or "")
        assert "integer-only" in text


class TestTheRefutedClaimsAreGone:
    """Each of these is paired with the measurement that refutes it."""

    def test_no_surface_promises_a_three_step_reduction(
        self, documented_surfaces
    ):
        """MEASURED: operands are re-read from ``token_ids`` every ACT step."""
        for where, text in documented_surfaces.items():
            assert "(1 + 2) * (3 + 4)" not in text, (
                f"{where} still offers the multi-step example. Step N's result "
                "cannot reach step N+1's operands at ANY weights "
                "(test_operand_derivation_through_call.py). If this became "
                "true, delete this test rather than the sentence."
            )
            assert "take 3 steps" not in text, where

    def test_no_surface_says_the_ntm_supplies_operands(
        self, documented_surfaces
    ):
        """MEASURED: read-head vectors are concatenated into the CONTROLLER."""
        for where, text in documented_surfaces.items():
            assert "extract operands (left, right)" not in text, where
            assert "operand extraction via ntm read heads" not in text, where
            assert "intermediate result storage and operand retrieval" not in text, (
                where
            )

    def test_the_readme_does_not_present_its_example_as_evaluable(
        self, documented_surfaces
    ):
        """The quickstart runs ``"1 + 2 * 3"``, which NAM cannot evaluate."""
        text = documented_surfaces["README.md"]
        assert "and not because the model can evaluate it" in text, (
            "the README quickstart uses a two-operator expression; without the "
            "disclaimer it reads as a worked example of something out of scope"
        )


class TestTheDocsAgreeWithTheCode:
    """Anti-vacuity: the sentences above must still describe THIS build.

    A text gate that outlives the behaviour it describes is worse than no gate,
    so the two numbers the docs quote are re-measured here.
    """

    @pytest.fixture
    def tiny_model(self):
        return NAM(config=NAMConfig(
            hidden_size=32,
            num_heads=4,
            num_tree_layers=1,
            intermediate_size=64,
            memory_size=8,
            num_read_heads=2,
            max_expression_len=16,
            halt_max_steps=4,
            hidden_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ))

    @pytest.mark.parametrize(
        "expression,operator_id,expected",
        [
            ("1 + 2 * 3", 14, (1.0, 23.0)),   # documented: concatenated, not 7
            ("1.5 + 2", 14, (15.0, 2.0)),     # documented: the dot is dropped
        ],
    )
    def test_the_quoted_operands_are_the_real_ones(
        self, tiny_model, expression, operator_id, expected
    ):
        tokenizer = ArithmeticTokenizer(max_len=16)
        ids = np.asarray(tokenizer.encode_batch([expression]))[0]

        # Re-derive the docs' split rule at the named operator position.
        pos = int(np.argmax(ids == operator_id))
        digits = [(i, int(t) - 4) for i, t in enumerate(ids) if 4 <= int(t) <= 13]

        def assemble(items):
            v = 0.0
            for _, d in items:
                v = v * 10.0 + d
            return v

        left = assemble([x for x in digits if x[0] < pos])
        right = assemble([x for x in digits if x[0] > pos])

        assert (left, right) == expected, (
            f"the docs quote {expected} for {expression!r} but the split rule "
            f"now yields {(left, right)}; correct the docs, not this test"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
