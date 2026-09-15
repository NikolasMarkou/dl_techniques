"""
Test suite for `layers/activations/common.py`'s activation-argument trio.

Covers the completion-fix for `plan-2026-09-15T034909-a7edc8da` step 7.1
(decisions.md D-014): `resolve_activation`'s final `else` branch had lost the
`callable()` validation `keras.activations.get` used to provide. MEASURED
before the fix: `resolve_activation(7)` returned `7` unchanged instead of
raising, deferring the failure to a confusing `TypeError` inside the layer's
`call()` method rather than an immediate `ValueError` at construction time.
"""

import keras
import pytest

from dl_techniques.layers.activations.common import resolve_activation


class TestResolveActivationRejectsNonCallableGarbage:
    """`resolve_activation` must raise for any non-callable, non-None,
    non-string, non-dict input -- matching `keras.activations.get`'s own
    immediate-raise behavior for garbage identifiers."""

    @pytest.mark.parametrize("garbage", [7, 7.0, [1, 2], True])
    def test_raises_value_error_for_non_callable_garbage(self, garbage):
        with pytest.raises(ValueError, match="Could not interpret activation function identifier"):
            resolve_activation(garbage)

    def test_layer_instance_still_raises_its_own_specific_message(self):
        # A `keras.layers.Layer` IS callable, so it must not be caught by the
        # new callable() guard -- it must still reach the more specific
        # Layer-rejection message from `_deserialize_activation`.
        with pytest.raises(ValueError, match="keras Layer instance"):
            resolve_activation(keras.layers.LeakyReLU(0.1))

    def test_valid_inputs_are_unaffected(self):
        assert resolve_activation(None) is keras.activations.linear
        assert resolve_activation("relu") is keras.activations.relu
        assert resolve_activation(keras.activations.silu) is keras.activations.silu
