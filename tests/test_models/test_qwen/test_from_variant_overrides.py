"""
`from_variant` must accept an override of a variant key.

`Qwen3.from_variant` and `Qwen3Next.from_variant` both ended in
``cls(**config, **kwargs)`` with no ``config.update(kwargs)``. Every variant dict
already carries the overridable keys, so ANY override of one collides:

    TypeError: dl_techniques.models.qwen.qwen3.Qwen3() got multiple values for
    keyword argument 'num_layers'

That is the exact call `create_qwen3`'s own docstring advertises
(``create_qwen3("tiny", num_layers=2)``), so the documented use was dead. Six
siblings already do it correctly; `models/gpt2/gpt2.py:464-468` is the one copied
here:

    config = cls.MODEL_VARIANTS[variant].copy()
    config.pop("description", None)
    config.update(kwargs)
    model = cls(**config)

Each test asserts the RESULTING ATTRIBUTE, not merely that nothing raised -- an
override that is accepted and then ignored is the failure mode a
``pytest.raises``-free smoke test cannot see.
"""

import pytest

from dl_techniques.models.qwen.qwen3 import Qwen3, create_qwen3
from dl_techniques.models.qwen.qwen3_next import Qwen3Next


class TestQwen3FromVariantOverrides:

    def test_a_variant_key_can_be_overridden(self) -> None:
        model = Qwen3.from_variant("tiny", num_layers=2)
        assert model.num_layers == 2
        assert len(model.blocks) == 2

    def test_the_unoverridden_keys_still_come_from_the_variant(self) -> None:
        """The override must not wipe the rest of the variant."""
        model = Qwen3.from_variant("tiny", num_layers=2)
        assert model.hidden_size == Qwen3.MODEL_VARIANTS["tiny"]["hidden_size"]

    def test_the_variant_table_is_not_mutated(self) -> None:
        Qwen3.from_variant("tiny", num_layers=2)
        assert Qwen3.MODEL_VARIANTS["tiny"]["num_layers"] == 6

    def test_the_create_qwen3_docstring_example_runs(self) -> None:
        """CONTROL -- this arm PASSED before the fix, deliberately kept.

        `create_qwen3("tiny", num_layers=2)` is verbatim from `create_qwen3`'s
        docstring and it worked all along: that function never calls
        `from_variant`, it reads `MODEL_VARIANTS` itself and already did the
        `config.update(...)` merge (`qwen3.py:818-839`). It is here to localise
        the defect to `from_variant` rather than to the qwen3 package, and to
        stop a future refactor from routing it through the broken path.
        """
        model = create_qwen3("tiny", num_layers=2)
        assert model is not None


class TestQwen3NextFromVariantOverrides:

    def test_a_variant_key_can_be_overridden(self) -> None:
        model = Qwen3Next.from_variant("tiny", num_layers=1)
        assert model.num_layers == 1

    def test_the_unoverridden_keys_still_come_from_the_variant(self) -> None:
        model = Qwen3Next.from_variant("tiny", num_layers=1)
        assert model.hidden_size == Qwen3Next.MODEL_VARIANTS["tiny"]["hidden_size"]

    def test_the_variant_table_is_not_mutated(self) -> None:
        Qwen3Next.from_variant("tiny", num_layers=1)
        assert Qwen3Next.MODEL_VARIANTS["tiny"]["num_layers"] == 3


class TestUnknownVariantStillRaises:
    """The permissive merge must not swallow a bad variant name."""

    @pytest.mark.parametrize("cls", [Qwen3, Qwen3Next])
    def test_unknown_variant(self, cls) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            cls.from_variant("does-not-exist")
