"""Direct tests for ``FFNExpert`` -- the only expert implementation in ``layers/moe``.

Scope note: ``test_layer.py::TestFFNExpert`` and ``test_gating.py::TestFFNExpert``
already cover the ``mlp``/``swiglu`` happy path, the missing-``type`` raise, the
factory function and a ``.keras`` round trip. This module adds what neither has:
the **full** FFN-type matrix reachable through ``FFNExpert`` (21 registry
entries), the pre-/post-norm combinations, and the dimension-changing
``post_norm`` case whose docstring was corrected by execution in step 8b.

Every entry of ``FFN_CASES`` below was measured against this tree on 2026-08-26,
including the required input rank -- three of the 21 types are rank-restricted
and one (``monarch``) constrains ``output_dim``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.ffn.factory import FFN_REGISTRY
from dl_techniques.layers.moe.experts import FFNExpert, create_expert

D = 10          # input feature width
OUT = 7         # output width, deliberately != D so a pass-through is visible
R2 = (3, D)
R3 = (3, 5, D)
R4 = (2, 4, 4, D)

# (ffn_type, ffn params, input shape, expected output width)
#
# The input shape is part of the case because the FFN registry is not
# rank-uniform: ``mixer`` requires rank 3, ``tversky`` requires rank 2 and
# ``gated_mlp`` is a convolutional block that requires rank 4. MEASURED,
# 2026-08-26 -- ``mixer`` also ignores ``output_dim`` and returns the input
# width, and ``monarch`` requires ``output_dim`` divisible by ``nblocks``.
FFN_CASES = [
    ('mlp', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('swiglu', {'output_dim': OUT, 'hidden_dim': 16}, R2, OUT),
    ('geglu', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('glu', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('reglu', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('bilinear', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('differential', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('residual', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('squared_relu', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('lowrank', {'hidden_dim': 16, 'output_dim': OUT, 'rank': 4}, R2, OUT),
    # orthoglu keeps hidden_dim <= output_dim: its orthogonal initializer warns
    # (and pytest promotes that to an error) when asked for more orthogonal
    # vectors than the latent width admits.
    ('orthoglu', {'hidden_dim': 6, 'output_dim': OUT}, R2, OUT),
    ('gelu_tanh', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('swin_mlp', {'hidden_dim': 16, 'output_dim': OUT}, R2, OUT),
    ('counting', {'output_dim': OUT, 'count_dim': 4}, R2, OUT),
    ('logic', {'output_dim': OUT, 'logic_dim': 4}, R2, OUT),
    ('power_mlp', {'units': OUT}, R2, OUT),
    ('tversky', {'units': OUT, 'num_features': 4}, R2, OUT),
    ('kan', {'features': OUT}, R2, OUT),
    ('monarch', {'hidden_dim': 16, 'output_dim': 8, 'nblocks': 2}, R2, 8),
    ('mixer', {'tokens_mlp_dim': 8, 'channels_mlp_dim': 16}, R3, D),
    ('gated_mlp', {'filters': OUT}, R4, OUT),
]

FFN_IDS = [c[0] for c in FFN_CASES]


def _tensor(shape, seed=0):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(seed).standard_normal(shape).astype('float32')
    )


class TestEveryFFNTypeIsReachableThroughFFNExpert:

    def test_the_case_table_covers_the_whole_registry(self):
        """A new FFN type must be added here, not silently left untested."""
        assert set(FFN_IDS) == set(FFN_REGISTRY)

    @pytest.mark.parametrize("ffn_type,params,shape,width", FFN_CASES, ids=FFN_IDS)
    def test_forward_pass_shape_and_finiteness(self, ffn_type, params, shape, width):
        expert = FFNExpert(ffn_config=dict(type=ffn_type, **params))
        out = expert(_tensor(shape))
        assert tuple(out.shape) == shape[:-1] + (width,)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("ffn_type,params,shape,width", FFN_CASES, ids=FFN_IDS)
    def test_compute_output_shape_matches_the_runtime_shape(
            self, ffn_type, params, shape, width
    ):
        """The symbolic contract must agree with what ``call`` actually returns."""
        expert = FFNExpert(ffn_config=dict(type=ffn_type, **params))
        symbolic = tuple(expert.compute_output_shape(shape))
        runtime = tuple(expert(_tensor(shape)).shape)
        assert symbolic == runtime == shape[:-1] + (width,)

    @pytest.mark.parametrize("ffn_type,params,shape,width", FFN_CASES, ids=FFN_IDS)
    def test_get_config_round_trip(self, ffn_type, params, shape, width):
        expert = FFNExpert(ffn_config=dict(type=ffn_type, **params))
        expert(_tensor(shape))  # build

        rebuilt = FFNExpert.from_config(expert.get_config())
        rebuilt(_tensor(shape))
        rebuilt.set_weights(expert.get_weights())

        x = _tensor(shape, seed=11)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(expert(x)),
            keras.ops.convert_to_numpy(rebuilt(x)),
            rtol=1e-6, atol=1e-6,
        )


class TestNormalizationWrapping:
    """``norm_type`` + ``pre_norm``/``post_norm`` are independent toggles."""

    @staticmethod
    def _expert(**kwargs):
        return FFNExpert(
            ffn_config={'type': 'mlp', 'hidden_dim': 16, 'output_dim': OUT},
            **kwargs,
        )

    @pytest.mark.parametrize("norm_type", ['rms_norm', 'layer_norm'])
    @pytest.mark.parametrize(
        "pre,post", [(True, False), (False, True), (True, True), (False, False)]
    )
    def test_every_combination_builds_and_runs(self, norm_type, pre, post):
        expert = self._expert(norm_type=norm_type, pre_norm=pre, post_norm=post)
        assert (expert.pre_norm is not None) is pre
        assert (expert.post_norm is not None) is post
        out = expert(_tensor(R2))
        assert tuple(out.shape) == (R2[0], OUT)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("pre,post", [(True, False), (False, True), (True, True)])
    def test_norm_type_none_suppresses_both_flags(self, pre, post):
        """With no ``norm_type`` the flags are inert, whatever they say."""
        expert = self._expert(norm_type=None, pre_norm=pre, post_norm=post)
        assert expert.pre_norm is None and expert.post_norm is None

    def test_the_norms_actually_change_the_output(self):
        """Without this, every assertion above would also pass on an expert
        that constructed the norms and then never called them."""
        x = _tensor(R2) * 25.0  # large scale, so normalizing is visible
        plain = self._expert()
        normed = self._expert(norm_type='rms_norm', pre_norm=True)
        plain.build(R2)
        normed.build(R2)
        normed.ffn_block.set_weights(plain.ffn_block.get_weights())
        assert not np.allclose(
            keras.ops.convert_to_numpy(plain(x)),
            keras.ops.convert_to_numpy(normed(x)),
            atol=1e-4,
        )

    def test_norm_config_reaches_the_norm_layer(self):
        expert = self._expert(
            norm_type='layer_norm', norm_config={'epsilon': 0.5}, post_norm=True
        )
        assert expert.pre_norm.epsilon == 0.5
        assert expert.post_norm.epsilon == 0.5


class TestPostNormWithADimensionChangingFFN:
    """Step 8b / F-14: the old docstring claimed a dimension-changing FFN was
    incompatible with ``post_norm``. Execution refuted it -- ``build`` sizes the
    post-norm from ``ffn_block.compute_output_shape``, not from the input.

    These are **regression pins on already-correct behaviour**, not RED proofs
    of a new guard. They do fail if ``build`` is changed to size the post-norm
    from the input shape (verified by mutation, see the plan's step-9 report).
    """

    @staticmethod
    def _expert(post_norm=True):
        return FFNExpert(
            ffn_config={'type': 'mlp', 'hidden_dim': 16, 'output_dim': OUT},
            norm_type='rms_norm',
            pre_norm=True,
            post_norm=post_norm,
        )

    def test_the_two_norms_carry_independently_sized_weights(self):
        expert = self._expert()
        expert.build(R2)
        pre_widths = [tuple(w.shape) for w in expert.pre_norm.weights]
        post_widths = [tuple(w.shape) for w in expert.post_norm.weights]
        assert pre_widths and post_widths
        assert all(s == (D,) for s in pre_widths), pre_widths
        assert all(s == (OUT,) for s in post_widths), post_widths

    def test_forward_pass_is_finite_and_the_declared_width(self):
        out = self._expert()(_tensor(R2))
        assert tuple(out.shape) == (R2[0], OUT)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_post_norm_is_not_a_no_op(self):
        """The pin must distinguish 'post-norm ran' from 'post-norm absent'."""
        with_post = self._expert(post_norm=True)
        without = self._expert(post_norm=False)
        with_post.build(R2)
        without.build(R2)
        without.pre_norm.set_weights(with_post.pre_norm.get_weights())
        without.ffn_block.set_weights(with_post.ffn_block.get_weights())
        x = _tensor(R2, seed=5) * 25.0
        assert not np.allclose(
            keras.ops.convert_to_numpy(with_post(x)),
            keras.ops.convert_to_numpy(without(x)),
            atol=1e-4,
        )

    def test_serialization_survives_the_dimension_change(self):
        expert = self._expert()
        expert(_tensor(R2))
        rebuilt = FFNExpert.from_config(expert.get_config())
        rebuilt(_tensor(R2))
        rebuilt.set_weights(expert.get_weights())
        x = _tensor(R2, seed=7)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(expert(x)),
            keras.ops.convert_to_numpy(rebuilt(x)),
            rtol=1e-6, atol=1e-6,
        )


class TestExpertConstructionErrors:

    def test_unknown_ffn_type_is_rejected(self):
        with pytest.raises(ValueError):
            FFNExpert(ffn_config={'type': 'definitely_not_an_ffn', 'hidden_dim': 4})

    def test_missing_required_ffn_param_is_rejected(self):
        with pytest.raises(ValueError):
            FFNExpert(ffn_config={'type': 'mlp'})  # no hidden_dim / output_dim

    def test_create_expert_only_makes_ffn_experts(self):
        expert = create_expert(
            'ffn', ffn_config={'type': 'mlp', 'hidden_dim': 8, 'output_dim': OUT}
        )
        assert isinstance(expert, FFNExpert)
        with pytest.raises(ValueError, match='Unsupported expert type'):
            create_expert('attention', ffn_config={'type': 'mlp'})

# ---------------------------------------------------------------------
