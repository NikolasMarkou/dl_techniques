"""N-10: Qwen3-Next's ``head_dim`` is DECOUPLED upstream and must stay overridable.

The released ``Qwen/Qwen3-Next-80B-A3B-Instruct`` config (re-fetched 2026-08-23)
carries ``"head_dim": 256`` alongside ``"hidden_size": 2048`` and
``"num_attention_heads": 16``, whose quotient is **128**. Before D-204 this model
derived ``head_dim`` from that quotient unconditionally, so the ``80b_a3b``
variant had **half** the released per-head width while calling itself by the
released name -- the same class of defect as the five field values D-112 already
corrected on this row.

The guard has two arms on purpose. Pinning the table value alone would pass
against a constructor that ignores it; pinning the constructor alone would pass
against a table that never sets it.
"""

import keras
import pytest

from dl_techniques.models.qwen.qwen3_next import Qwen3Next

#: Released config, fetched 2026-08-23 from
#: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
_RELEASED_HEAD_DIM = 256
_RELEASED_HIDDEN_SIZE = 2048
_RELEASED_NUM_ATTENTION_HEADS = 16


class TestTheReleasedHeadDimIsDecoupled:

    def test_the_variant_table_carries_the_released_head_dim(self):
        variant = Qwen3Next.MODEL_VARIANTS["80b_a3b"]
        assert variant["head_dim"] == _RELEASED_HEAD_DIM, (
            f"the 80b_a3b row's head_dim is {variant.get('head_dim')}, not the "
            f"released {_RELEASED_HEAD_DIM}."
        )

    def test_the_released_value_is_not_the_quotient(self):
        """If it were, the whole override mechanism would be untested by the row."""
        variant = Qwen3Next.MODEL_VARIANTS["80b_a3b"]
        quotient = variant["hidden_size"] // variant["num_attention_heads"]
        assert variant["hidden_size"] == _RELEASED_HIDDEN_SIZE
        assert variant["num_attention_heads"] == _RELEASED_NUM_ATTENTION_HEADS
        assert quotient == 128
        assert variant["head_dim"] != quotient, (
            "head_dim now equals hidden_size // num_attention_heads, so this "
            "test can no longer tell a decoupled head_dim from a derived one. "
            "Either the released config changed (re-fetch and re-pin) or the "
            "override was quietly removed."
        )

    def test_the_constructor_honours_an_explicit_head_dim(self):
        model = Qwen3Next(
            vocab_size=128,
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            head_dim=32,
            num_experts=1,
            num_experts_per_tok=1,
            max_seq_len=32,
        )
        assert model.head_dim == 32, (
            f"head_dim=32 was passed and the model derived {model.head_dim} "
            "(= hidden_size // num_attention_heads = 16). The argument is being "
            "ignored, which is exactly the pre-D-204 defect."
        )

    def test_omitting_head_dim_still_derives_the_quotient(self):
        """Backward compatibility: every other variant and caller is unchanged."""
        model = Qwen3Next(
            vocab_size=128,
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            num_experts=1,
            num_experts_per_tok=1,
            max_seq_len=32,
        )
        assert model.head_dim == 16

    def test_head_dim_round_trips_through_get_config(self):
        model = Qwen3Next(
            vocab_size=128,
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            head_dim=32,
            num_experts=1,
            num_experts_per_tok=1,
            max_seq_len=32,
        )
        config = model.get_config()
        assert config["head_dim"] == 32, (
            "head_dim is absent from get_config(), so a saved model silently "
            "reverts to the quotient on load."
        )
        rebuilt = Qwen3Next.from_config(
            {k: v for k, v in config.items()
             if k not in ("name", "trainable", "dtype")}
        )
        assert rebuilt.head_dim == 32

    def test_a_decoupled_head_dim_actually_changes_the_weight_shapes(self):
        """Not a stored-attribute check: the value must reach the attention weights."""
        def signature(head_dim):
            keras.utils.set_random_seed(3)
            model = Qwen3Next(
                vocab_size=128, hidden_size=64, num_layers=1,
                num_attention_heads=4, head_dim=head_dim, num_experts=1,
                num_experts_per_tok=1, max_seq_len=32,
            )
            model(keras.ops.zeros((1, 8), dtype="int32"))
            return sorted(tuple(w.shape) for w in model.weights)

        assert signature(16) != signature(32), (
            "changing head_dim left the weight-shape signature identical, so "
            "the value is stored on the model and never reaches the attention "
            "projections. A stored-attribute assertion would have passed here."
        )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_nonpositive_head_dim_raises(self, bad):
        with pytest.raises(ValueError):
            Qwen3Next(
                vocab_size=128, hidden_size=64, num_layers=1,
                num_attention_heads=4, head_dim=bad, num_experts=1,
                num_experts_per_tok=1, max_seq_len=32,
            )
