"""
`HierarchicalReasoningModel.from_variant` must accept an override of a variant key.

It ended in ``cls(vocab_size=..., seq_len=..., num_puzzle_identifiers=...,
**config, **kwargs)`` with no ``config.update(kwargs)``. Every key in
`MODEL_VARIANTS` is therefore un-overridable:

    TypeError: dl_techniques.models.language.hrm.model.
    HierarchicalReasoningModel() got multiple values for keyword argument
    'halt_max_steps'

`create_hierarchical_reasoning_model`'s own docstring Sudoku example --
``variant="base", halt_max_steps=16`` -- is exactly this call, so the documented
configuration could not be built. A docstring claim is only testable by
EXECUTING it, so the last test here does.

`micro` is used instead of `base` wherever the assertion does not depend on the
variant, purely for build cost.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.language.hrm.model import (
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


class TestThePretrainedArgument:
    """`pretrained`: `True` refuses, a local path restores, a mismatch raises."""

    _SMALL = dict(vocab_size=16, seq_len=8, embed_dim=32, num_puzzle_identifiers=4,
                  h_layers=1, l_layers=1, num_heads=2, halt_max_steps=2)

    def test_pretrained_true_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            create_hierarchical_reasoning_model("micro", pretrained=True, **self._SMALL)

    def test_a_local_checkpoint_is_restored(self, tmp_path) -> None:
        keras.utils.set_random_seed(1)
        source = create_hierarchical_reasoning_model("micro", **self._SMALL)
        path = str(tmp_path / "hrm.weights.h5")
        source.save_weights(path)
        keras.utils.set_random_seed(2)
        target = create_hierarchical_reasoning_model(
            "micro", pretrained=path, **self._SMALL)
        assert len(target.weights) == len(source.weights)
        for a, b in zip(source.weights, target.weights):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))

    def test_a_checkpoint_of_another_width_is_refused(self, tmp_path) -> None:
        path = str(tmp_path / "hrm.weights.h5")
        create_hierarchical_reasoning_model("micro", **self._SMALL).save_weights(path)
        with pytest.raises(ValueError):
            create_hierarchical_reasoning_model(
                "micro", pretrained=path, **dict(self._SMALL, embed_dim=64))


class TestTheSudokuDocstringExampleRuns:
    """`create_hierarchical_reasoning_model`'s own documented Sudoku config."""

    def test_the_example_builds_and_the_override_takes_effect(self) -> None:
        model = create_hierarchical_reasoning_model(
            vocab_size=20,      # 0-9 digits + special tokens
            seq_len=81,         # 9x9 grid flattened
            variant="micro",    # 'small' in the docstring; 'micro' for build cost
            halt_max_steps=16,  # For backtracking search
        )
        assert model.halt_max_steps == 16
