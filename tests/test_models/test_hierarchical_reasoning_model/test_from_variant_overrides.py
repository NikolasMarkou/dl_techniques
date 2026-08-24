"""
`HierarchicalReasoningModel.from_variant` must accept an override of a variant key.

It ended in ``cls(vocab_size=..., seq_len=..., num_puzzle_identifiers=...,
**config, **kwargs)`` with no ``config.update(kwargs)``. Every key in
`MODEL_VARIANTS` is therefore un-overridable:

    TypeError: dl_techniques.models.hierarchical_reasoning_model.model.
    HierarchicalReasoningModel() got multiple values for keyword argument
    'halt_max_steps'

`create_hierarchical_reasoning_model`'s own docstring Sudoku example --
``variant="base", halt_max_steps=16`` -- is exactly this call, so the documented
configuration could not be built. A docstring claim is only testable by
EXECUTING it, so the last test here does.

`micro` is used instead of `base` wherever the assertion does not depend on the
variant, purely for build cost.
"""

import pytest

from dl_techniques.models.hierarchical_reasoning_model.model import (
    HierarchicalReasoningModel,
    create_hierarchical_reasoning_model,
)


class TestHRMFromVariantOverrides:

    def test_a_variant_key_can_be_overridden(self) -> None:
        model = HierarchicalReasoningModel.from_variant(
            "micro", vocab_size=20, seq_len=16, halt_max_steps=16
        )
        assert model.halt_max_steps == 16

    def test_the_unoverridden_keys_still_come_from_the_variant(self) -> None:
        model = HierarchicalReasoningModel.from_variant(
            "micro", vocab_size=20, seq_len=16, halt_max_steps=16
        )
        assert model.embed_dim == HierarchicalReasoningModel.MODEL_VARIANTS["micro"]["embed_dim"]

    def test_the_variant_table_is_not_mutated(self) -> None:
        HierarchicalReasoningModel.from_variant(
            "micro", vocab_size=20, seq_len=16, halt_max_steps=16
        )
        assert HierarchicalReasoningModel.MODEL_VARIANTS["micro"]["halt_max_steps"] == 4

    def test_unknown_variant_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            HierarchicalReasoningModel.from_variant(
                "does-not-exist", vocab_size=20, seq_len=16
            )


class TestTheSudokuDocstringExampleRuns:
    """`create_hierarchical_reasoning_model`'s own documented Sudoku config."""

    def test_the_example_builds_and_the_override_takes_effect(self) -> None:
        model = create_hierarchical_reasoning_model(
            vocab_size=20,      # 0-9 digits + special tokens
            seq_len=81,         # 9x9 grid flattened
            variant="micro",    # 'base' in the docstring; 'micro' for build cost
            halt_max_steps=16,  # For backtracking search
        )
        assert model.halt_max_steps == 16
