"""Tests for ``PatchCausalTransformer``
(``dl_techniques.layers.transformers.patch_causal_transformer``).

Also covers the shared skeleton in ``transformers/causal_stack.py``, since this
layer is its most direct consumer and the BLT layers it also serves have no
package of their own.
"""

import os

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.causal_stack import (
    LAYER_NORM_EPSILON,
    build_causal_stack_norm,
    build_causal_transformer_stack,
)
from dl_techniques.layers.transformers.patch_causal_transformer import (
    PatchCausalTransformer,
)

NP_, GDIM = 8, 16
B = 2


# ---------------------------------------------------------------------
# PatchCausalTransformer (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestPatchCausalTransformer:

    def _make(self, **kwargs):
        params = dict(dim=GDIM, depth=1, num_heads=2, max_patches=NP_)
        params.update(kwargs)
        return PatchCausalTransformer(**params)

    def _x(self):
        return keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        )

    def test_forward_pass(self):
        assert tuple(self._make()(self._x()).shape) == (B, NP_, GDIM)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, NP_, GDIM)) == (B, NP_, GDIM)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(NP_, GDIM))
        out = self._make()(inp)
        model = keras.Model(inp, out)
        y0 = model(self._x())
        path = os.path.join(tmp_path, "patch_causal.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"PatchCausalTransformer": PatchCausalTransformer}
        )
        y1 = loaded(self._x())
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_config_round_trip_rebuilds_identically(self):
        layer = self._make(name_prefix='custom_prefix')
        rebuilt = PatchCausalTransformer.from_config(layer.get_config())
        for key in ('dim', 'depth', 'num_heads', 'max_patches',
                    'dropout_rate', 'layer_norm_epsilon', 'name_prefix'):
            assert rebuilt.get_config()[key] == layer.get_config()[key], key

    def test_depth_and_heads_are_structural(self):
        """Depth and head count must reach the weights, not just the config."""
        one = self._make(depth=1)
        three = self._make(depth=3)
        one.build((B, NP_, GDIM))
        three.build((B, NP_, GDIM))
        assert len(three.stack_layers) == 3
        assert len(three.weights) > len(one.weights)

    def test_name_prefix_renames_every_sub_layer(self):
        """Two instances in one model must not collide on sub-layer names."""
        a = self._make(name_prefix='alpha')
        b = self._make(name_prefix='beta')
        a.build((B, NP_, GDIM))
        b.build((B, NP_, GDIM))
        a_names = {w.path for w in a.weights}
        b_names = {w.path for w in b.weights}
        assert a_names and b_names
        assert a_names.isdisjoint(b_names), sorted(a_names & b_names)

    def test_an_over_long_sequence_raises_rather_than_truncating(self):
        """``max_patches`` is a hard ceiling, checked at build.

        The message names ``max_seq_len`` rather than ``max_patches`` because
        the check belongs to ``PositionalEmbedding``, which only knows its own
        parameter; both are the same number here.
        """
        layer = self._make()
        with pytest.raises(ValueError, match="max_seq_len"):
            layer.build((B, NP_ + 1, GDIM))

    def test_the_attention_is_causal(self):
        """Perturbing patch j must not move any output at patch i < j.

        The single-claim guard for the mask: a stack that dropped its causal
        mask would still pass every shape and round-trip test above.
        """
        layer = self._make(depth=2)
        x = np.random.default_rng(1).standard_normal((1, NP_, GDIM)).astype("float32")
        y0 = keras.ops.convert_to_numpy(layer(x, training=False))

        perturbed = x.copy()
        perturbed[0, NP_ - 1] += 10.0
        y1 = keras.ops.convert_to_numpy(
            layer(perturbed, training=False)
        )

        np.testing.assert_allclose(
            y0[:, :-1], y1[:, :-1], rtol=1e-6, atol=1e-6,
            err_msg="an earlier patch's output moved when the LAST patch was "
                    "perturbed, so the attention is not causal",
        )
        assert not np.allclose(y0[:, -1], y1[:, -1]), (
            "the last patch's own output did not move either -- the probe is "
            "not reaching the layer at all"
        )


# ---------------------------------------------------------------------
# causal_stack helpers
# ---------------------------------------------------------------------

class TestCausalStackHelpers:

    def test_the_ffn_defaults_to_four_times_hidden(self):
        stack = build_causal_transformer_stack(
            hidden_size=32, num_heads=2, depth=1, dropout_rate=0.0
        )
        assert int(stack[0].ffn_layer.hidden_dim) == 128, stack[0].ffn_layer.hidden_dim

    def test_an_explicit_intermediate_size_wins(self):
        stack = build_causal_transformer_stack(
            hidden_size=32, num_heads=2, depth=1, dropout_rate=0.0,
            intermediate_size=7,
        )
        assert int(stack[0].ffn_layer.hidden_dim) == 7

    def test_depth_zero_builds_an_empty_stack(self):
        assert build_causal_transformer_stack(
            hidden_size=32, num_heads=2, depth=0, dropout_rate=0.0
        ) == []

    def test_the_norm_helper_always_states_epsilon(self):
        """No caller can forget the argument, which is the whole point.

        MEASURED consequence of the value this pins, on this layer at
        dim=32/depth=2/max_patches=8: 1e-3 -> 1e-6 moves the forward pass by
        max|delta| = 1.7e-03. The constant is therefore load-bearing and is
        pinned per-package by
        ``tests/test_models/test_the_norm_epsilon_provenance_is_stated.py``.
        """
        assert LAYER_NORM_EPSILON == 1e-3
        assert build_causal_stack_norm('n').epsilon == 1e-3
        assert build_causal_stack_norm('n', epsilon=1e-5).epsilon == 1e-5

    def test_the_norm_is_not_the_factory_default(self):
        """Guards the deliberate divergence from the norm factory.

        ``create_normalization_layer`` would give 1e-6 here; this stack is 1e-3
        on purpose (see ``LAYER_NORM_EPSILON``). If someone "fixes" this by
        routing through the factory, this test is what notices.
        """
        from dl_techniques.layers.norms import create_normalization_layer

        ours = build_causal_stack_norm('ours').epsilon
        factory = create_normalization_layer('layer_norm').epsilon
        assert ours != factory, (
            f"the causal stack norm epsilon moved to the factory's {factory}; "
            "that is a 1000x silent numerics change. Read LAYER_NORM_EPSILON."
        )