"""Guards on the block registry -- the seam that makes the study extensible.

The registry is the reason adding an arm is one entry rather than a second
copy of the encoder, so its two refusals are load-bearing: an unknown type must
raise, and an undeclared keyword must raise rather than be dropped. Silently
dropping an unrecognized keyword is the design that previously turned a
misspelled ``dropout=`` (against a declared ``dropout_rate``) into a no-op
across the whole repository.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.embeddings_experimental.shared.blocks import (
    BLOCK_REGISTRY,
    CliffordEncoderBlock,
    available_block_types,
    clifford_receptive_field,
    create_encoder_block,
)


class TestRegistrySurface:
    """The registry keys are public API."""

    def test_both_study_arms_are_registered(self):
        assert set(BLOCK_REGISTRY) >= {"transformer", "clifford"}

    def test_available_block_types_is_sorted(self):
        types = available_block_types()
        assert types == sorted(types)
        assert set(types) == set(BLOCK_REGISTRY)

    def test_every_builder_accepts_the_two_shared_arguments(self):
        import inspect

        for name, builder in BLOCK_REGISTRY.items():
            params = inspect.signature(builder).parameters
            assert "hidden_size" in params, name
            assert "name" in params, name


class TestRegistryRefusals:
    """Both refusals, asserted directly."""

    def test_unknown_block_type_raises_and_names_the_alternatives(self):
        with pytest.raises(ValueError, match="Unknown block_type"):
            create_encoder_block("does_not_exist", hidden_size=8, name="b")

    def test_undeclared_keyword_raises_rather_than_being_dropped(self):
        # `num_heads` is meaningful for the transformer builder and meaningless
        # for the Clifford one; passing it to the latter must be an error.
        with pytest.raises(ValueError, match="does not declare"):
            create_encoder_block(
                "clifford", hidden_size=8, name="b", num_heads=4
            )

    def test_the_refusal_names_the_offending_keyword(self):
        with pytest.raises(ValueError, match=r"\['nonsense_knob'\]"):
            create_encoder_block(
                "transformer", hidden_size=8, name="b", nonsense_knob=1
            )

    def test_a_declared_keyword_is_accepted(self):
        block = create_encoder_block(
            "clifford", hidden_size=8, name="b", context_kernel_size=5
        )
        assert isinstance(block, CliffordEncoderBlock)
        assert block.context_kernel_size == 5


class TestBlockCallContract:
    """Every block honours the same call signature and is shape-preserving."""

    @pytest.mark.parametrize(
        "block_type,config",
        [
            ("transformer", {"num_heads": 2, "intermediate_size": 32}),
            ("clifford", {"shifts": [1, 2], "context_kernel_size": 3}),
        ],
    )
    def test_shape_is_preserved_and_mask_is_accepted(self, block_type, config):
        hidden_size, seq_len, batch = 16, 12, 2
        block = create_encoder_block(
            block_type, hidden_size=hidden_size, name="b", **config
        )
        block.build((None, seq_len, hidden_size))

        x = keras.ops.convert_to_tensor(
            np.random.randn(batch, seq_len, hidden_size).astype("float32")
        )
        mask = keras.ops.convert_to_tensor(
            np.ones((batch, seq_len), dtype="int32")
        )
        out = block(x, attention_mask=mask, layer_idx=0, training=False)

        assert tuple(out.shape) == (batch, seq_len, hidden_size)
        assert np.isfinite(keras.ops.convert_to_numpy(out)).all()


class TestCliffordReceptiveField:
    """The span arithmetic, which is a design parameter at character granularity."""

    def test_matches_the_two_convolutions_per_block_formula(self):
        assert clifford_receptive_field(4, 3) == 17
        assert clifford_receptive_field(4, 7) == 49
        assert clifford_receptive_field(6, 7) == 73

    def test_a_unit_kernel_mixes_nothing(self):
        assert clifford_receptive_field(12, 1) == 1

    def test_span_grows_with_both_depth_and_kernel(self):
        base = clifford_receptive_field(4, 3)
        assert clifford_receptive_field(8, 3) > base
        assert clifford_receptive_field(4, 5) > base
