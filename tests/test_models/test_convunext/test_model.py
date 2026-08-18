"""
Tests for the merged ConvUNext: ``ConvUNextStem`` and ``create_convunext``.

Structure mirrors ``test_bfconvunext_denoiser.py``, with the bias-on and bias-off
arms exercised SYMMETRICALLY (most classes below are parametrized over
``use_bias``) because the merge collapsed two implementations into one builder and
a defect that reaches only one arm is exactly what this suite exists to catch.

Covers:
- ``ConvUNextStem`` lifecycle and its merged ``use_bias`` / ``stem_normalization`` knobs
- ``create_convunext``'s ``use_bias`` threading, argument validation, forward pass,
  gradient flow, training integration and ``.keras`` round trips
- the residual wiring and the drop-path SCHEDULE (encoder ramp, bottleneck ramp,
  decoder "first block carries none"), read out of the FUNCTIONAL graph by layer
  name -- the deleted subclass's ``model.encoder_drop_paths[i][j]`` list indexing
  has no equivalent here
- deep-supervision output COUNT and ORDER
- the optional structural branches (``zero_pad_channels``,
  ``extra_zero_output_channels``, ``final_projection_groups``,
  ``expose_bottleneck``, ``use_laplacian_pyramid`` + ``high_freq_blocks``,
  ``bottleneck_attention_blocks``), each on BOTH arms
- the three ``use_bias=False`` guardrails and the knobs absorbed from the deleted
  ``ConvUNextModel`` (``downsample_pool_type='strided_conv'``, ``include_top``,
  ``output_channels``)

Two conventions this file holds to, both of which cost a vacuous guard earlier in
this plan:

1. Every ``.keras`` round trip is built at NON-DEFAULT parameter values. A round
   trip built at a parameter's default is NOT a ``get_config`` completeness
   instrument: the constructor default silently repairs the dropped key and the
   trip still passes (measured, plan steps 2 and 3).
2. Both forward calls around a round trip pass ``training=False`` EXPLICITLY.
   ``training=None`` is not inference mode and produces deltas that look like
   reinitialized weights (``plans/SYSTEM.md`` framework-behaviour #3).

NOTE (plan steps 4/9): the ConvUNextModel-era classes were removed in step 4
together with the subclassed ``ConvUNextModel`` they instantiated -- with it
deleted they could not even import. Step 9 re-derives that coverage against
``create_convunext``. Two things were DELIBERATELY not re-derived, and the reason
is recorded at the class that replaced them: ``TestCreateInferenceModel`` (the
bespoke helper is deleted; ``utils/deep_supervision`` owns the canonical one and
its own suite tests it) and ``test_include_top_weight_compatibility`` (the
weight-compat contract is not reproducible in a functional graph -- see
``TestIncludeTop.test_include_top_false_does_not_construct_the_final_projection``,
which pins the divergence we actually ship).
"""

import os
import logging
import tempfile
from typing import Dict, Any

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.match_channels import MatchChannels

from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.norms.global_response_norm import (
    GlobalResponseNormalization,
)
from dl_techniques.models.convunext.model import (
    CONVUNEXT_CONFIGS,
    ConvUNextStem,
    SpatialLinearAttention,
    create_convunext,
    create_convunext_variant,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_input_small() -> np.ndarray:
    """Small sample input for quick tests."""
    return np.random.randn(2, 64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_input_medium() -> np.ndarray:
    """Medium sample input for standard tests."""
    return np.random.randn(2, 128, 128, 3).astype(np.float32)


@pytest.fixture
def stem_config() -> Dict[str, Any]:
    """Default configuration for ConvUNextStem."""
    return {
        'filters': 32,
        'kernel_size': 7,
        'use_bias': True,
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': None,
    }


@pytest.fixture
def minimal_model_config() -> Dict[str, Any]:
    """Minimal configuration for fast model tests."""
    return {
        'input_shape': (64, 64, 3),
        'depth': 2,
        'initial_filters': 16,
        'filter_multiplier': 2,
        'blocks_per_level': 1,
        'convnext_version': 'v2',
        'drop_path_rate': 0.0,
        'output_channels': 1,
    }


# ---------------------------------------------------------------------
# ConvUNextStem Tests
# ---------------------------------------------------------------------

class TestConvUNextStem:
    """Test suite for ConvUNextStem layer."""

    def test_instantiation(self, stem_config: Dict[str, Any]) -> None:
        """Test stem can be instantiated with valid config."""
        stem = ConvUNextStem(**stem_config)
        assert stem.filters == stem_config['filters']
        assert stem.kernel_size == stem_config['kernel_size']
        assert stem.use_bias == stem_config['use_bias']

    def test_forward_pass(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test forward pass produces correct output shape."""
        stem = ConvUNextStem(**stem_config)
        output = stem(sample_input_small)

        expected_shape = (
            sample_input_small.shape[0],
            sample_input_small.shape[1],
            sample_input_small.shape[2],
            stem_config['filters'],
        )
        assert output.shape == expected_shape

    def test_build_creates_weights(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test that build() creates expected weights."""
        stem = ConvUNextStem(**stem_config)
        stem(sample_input_small)

        assert stem.built
        assert len(stem.trainable_weights) > 0

    def test_compute_output_shape(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test compute_output_shape matches actual output."""
        stem = ConvUNextStem(**stem_config)

        computed_shape = stem.compute_output_shape(sample_input_small.shape)
        actual_output = stem(sample_input_small)

        assert computed_shape == actual_output.shape

    def test_compute_output_shape_before_build(
        self,
        stem_config: Dict[str, Any]
    ) -> None:
        """Test compute_output_shape works before layer is built."""
        stem = ConvUNextStem(**stem_config)

        input_shape = (None, 64, 64, 3)
        computed_shape = stem.compute_output_shape(input_shape)

        assert computed_shape == (None, 64, 64, stem_config['filters'])

    def test_training_vs_inference(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test layer behaves correctly in training vs inference mode."""
        stem = ConvUNextStem(**stem_config)

        train_output = stem(sample_input_small, training=True)
        infer_output = stem(sample_input_small, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_output),
            keras.ops.convert_to_numpy(infer_output),
            rtol=1e-5, atol=1e-5,
            err_msg="Stem outputs should match in train vs inference"
        )

    def test_get_config_complete(self, stem_config: Dict[str, Any]) -> None:
        """Test get_config returns all constructor arguments."""
        stem = ConvUNextStem(**stem_config)
        config = stem.get_config()

        assert 'filters' in config
        assert 'kernel_size' in config
        assert 'use_bias' in config
        assert 'kernel_initializer' in config
        assert 'kernel_regularizer' in config

    def test_from_config_reconstruction(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = ConvUNextStem(**stem_config)
        original(sample_input_small)

        config = original.get_config()
        reconstructed = ConvUNextStem.from_config(config)

        assert reconstructed.filters == original.filters
        assert reconstructed.use_bias == original.use_bias

    def test_serialization_cycle(
        self,
        stem_config: Dict[str, Any],
        sample_input_small: np.ndarray
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        inputs = keras.Input(shape=sample_input_small.shape[1:])
        outputs = ConvUNextStem(**stem_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input_small)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_stem.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input_small)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Stem outputs should match after serialization"
        )

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_bias_configuration(
        self,
        use_bias: bool,
        sample_input_small: np.ndarray
    ) -> None:
        """Test stem works with and without bias."""
        stem = ConvUNextStem(filters=32, use_bias=use_bias)
        output = stem(sample_input_small)

        assert output.shape[-1] == 32

    def test_different_kernel_sizes(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """``kernel_size`` must reach the stem's convolution kernel.

        Restructured out of ``@parametrize``: one stem per invocation left
        nowhere to compare, so the test asserted ``(2, 64, 64, 32)`` -- which is
        the SAME output shape at 3, 5 and 7 because the stem pads to 'same', so
        the assertion held whether or not the kwarg reached the Conv2D. The
        kernel's spatial extent is a weight axis; measured, the stem holds
        exactly one rank-4 kernel of shape ``(k, k, 3, 32)``.
        """
        kernel_sizes = [3, 5, 7]

        def _build(kernel_size):
            stem = ConvUNextStem(filters=32, kernel_size=kernel_size)
            stem(sample_input_small)
            return stem

        sigs = assert_structural_knob_changes_weights(
            {k: (lambda k=k: _build(k)) for k in kernel_sizes}, knob="kernel_size")
        for k in kernel_sizes:
            # The stem also holds two (1, 1, 1, 32) GRN parameters, which are
            # rank-4 but are not the convolution; the conv is the one kernel
            # whose in-channels axis matches the 3-channel input.
            kernels = [w for w in sigs[k] if len(w) == 4 and w[2] == 3]
            assert kernels == [(k, k, 3, 32)], (
                f"kernel_size={k} produced conv kernels {kernels}"
            )

        for kernel_size in kernel_sizes:
            stem = ConvUNextStem(filters=32, kernel_size=kernel_size)
            output = stem(sample_input_small)

            assert output.shape == (2, 64, 64, 32)


class TestMergedConvUNextStemKnobs:
    """The two knobs the merge added: `use_bias` and `stem_normalization`.

    This class collapses what used to be two same-named classes (the bias-free
    GRN stem in `models/bias_free_denoisers/bfconvunext.py` and the LayerNorm
    stem here), so each knob needs a guard that is proven to fail when the knob
    is ignored.
    """

    def test_stem_use_bias_true_creates_a_bias_vector(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """`use_bias=True` must reach the conv AND allocate a bias WEIGHT.

        Asserting the flag alone is not enough: a layer can report
        `use_bias is True` and still never have allocated the vector.
        """
        stem = ConvUNextStem(filters=8, use_bias=True)
        stem(sample_input_small)

        assert stem.conv.use_bias is True
        assert stem.conv.bias is not None
        n_bias_weights = sum(
            1 for w in stem.conv.weights if w.path.endswith('bias')
        )
        assert n_bias_weights == 1, (
            f"use_bias=True stem allocated {n_bias_weights} bias weights"
        )
        assert int(np.prod(stem.conv.bias.shape)) == 8

        # Control: the bias-free arm allocates none.
        stem_bf = ConvUNextStem(filters=8, use_bias=False)
        stem_bf(sample_input_small)
        assert stem_bf.conv.use_bias is False
        assert stem_bf.conv.bias is None
        assert not any(w.path.endswith('bias') for w in stem_bf.conv.weights)

    def test_stem_normalization_layernorm_selects_layer_norm(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """`stem_normalization` must select the normalization CLASS actually built."""
        stem_ln = ConvUNextStem(filters=8, stem_normalization='layer_norm')
        stem_ln(sample_input_small)
        assert type(stem_ln.norm) is keras.layers.LayerNormalization

        # The default is the bias-free / ConvNeXt-V2 choice, a different class.
        stem_default = ConvUNextStem(filters=8)
        stem_default(sample_input_small)
        assert type(stem_default.norm) is GlobalResponseNormalization
        assert type(stem_default.norm) is not type(stem_ln.norm)

    def test_stem_config_round_trip_at_non_default_knob_values(
        self,
        sample_input_small: np.ndarray
    ) -> None:
        """Full `.keras` round trip built at NON-DEFAULT values for both knobs.

        Built at the defaults, this test would be vacuous: dropping a key from
        `get_config` is silently repaired by the constructor default and the
        round trip still passes (measured in step 2 of this plan). So the stem
        here is `use_bias=False` (default True) and
        `stem_normalization='layer_norm'` (default 'global_response_norm').
        """
        inputs = keras.Input(shape=sample_input_small.shape[1:])
        outputs = ConvUNextStem(
            filters=8,
            use_bias=False,
            stem_normalization='layer_norm',
            name='stem_under_test',
        )(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input_small, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_merged_stem.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_stem = loaded_model.get_layer('stem_under_test')
        assert loaded_stem.use_bias is False
        assert loaded_stem.conv.use_bias is False
        assert loaded_stem.conv.bias is None
        assert loaded_stem.stem_normalization == 'layer_norm'
        assert type(loaded_stem.norm) is keras.layers.LayerNormalization

        loaded_output = loaded_model(sample_input_small, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Merged stem outputs differ after .keras round-trip",
        )



# ---------------------------------------------------------------------
# create_convunext: use_bias threading, drop-path ramp, liveness
# ---------------------------------------------------------------------

# The ONLY layers allowed to stay bias-free when create_convunext is called with
# use_bias=True. Both are deliberate, documented hardcodes (decisions.md D-004):
# the frozen Gabor bank (trainable=False -- a bias on it is meaningless) and
# SpatialLinearAttention's internal LinearAttention + its four Dense projections
# (a bias there breaks the Miyasawa property; plan_2026-07-11_bb4b38b5/D-001).
_HARDCODED_BIAS_FREE_NAMES = frozenset({
    'gabor_stem',
    'query_proj', 'key_proj', 'value_proj', 'output_proj',
})


def _is_hardcoded_bias_free(name: str) -> bool:
    """True for the documented hardcodes only.

    The ``LinearAttention`` sub-layer built by ``SpatialLinearAttention`` is named
    ``<attention block name>_linear``, so it is matched by suffix rather than by a
    literal (its block index is part of the name).
    """
    return name in _HARDCODED_BIAS_FREE_NAMES or name.endswith('_linear')


def _bias_free_layer_names(model: keras.Model) -> set:
    """Names of every sub-layer in ``model`` whose ``use_bias`` is False."""
    return {
        layer.name
        for layer in model._flatten_layers()
        if getattr(layer, 'use_bias', None) is False
    }


def _builder_config() -> Dict[str, Any]:
    """A config exercising ALL FIVE structural site families at once.

    stem (via the gabor projection AND the standard stem), channel-adjust
    (encoder / bottleneck / decoder), ConvNeXt block, supervision head, and the
    final output projection.
    """
    return dict(
        input_shape=(16, 16, 3),
        depth=3,
        initial_filters=8,
        blocks_per_level=1,
        bottleneck_attention_blocks=1,
        bottleneck_attention_heads=2,
        enable_deep_supervision=True,
    )


class TestCreateConvUNextUseBias:
    """``use_bias`` must reach every structural site family, and only those."""

    def test_bias_on_has_strictly_more_parameters(self):
        """Bias-on costs strictly more parameters, and the ONLY layers left
        bias-free are the two documented hardcodes.

        The set assertion is what catches a single site family whose
        ``use_bias=use_bias`` was missed: that layer would show up in the
        bias-on model's bias-free set. A bare ``delta > 0`` cannot see it,
        because the other four families still contribute.
        """
        cfg = _builder_config()
        model_on = create_convunext(use_bias=True, **cfg)
        model_off = create_convunext(use_bias=False, **cfg)

        delta = model_on.count_params() - model_off.count_params()
        assert delta > 0, (
            f"use_bias=True must add parameters, got delta={delta} "
            f"(on={model_on.count_params()}, off={model_off.count_params()})"
        )

        # Magnitude bucket: biases are one scalar per output channel, so the
        # delta is a small but non-negligible fraction of the bias-free count.
        frac = delta / model_off.count_params()
        assert 1e-3 < frac < 1e-1, (
            f"bias parameter delta {delta} is {frac:.4%} of the bias-free count "
            f"{model_off.count_params()} -- outside the expected 0.1%-10% bucket"
        )

        # Every gabor/attention layer bias-free, and NOTHING else. Checked on BOTH
        # stem arms: the standard ConvUNextStem (model_on) and the frozen Gabor stem
        # + its 1x1 projection (model_gabor_on) are mutually exclusive branches, so
        # a single config cannot see a missed use_bias at both.
        gabor_cfg = dict(cfg, use_gabor_stem=True, gabor_filters=4,
                         gabor_kernel_size=5)
        model_gabor_on = create_convunext(use_bias=True, **gabor_cfg)
        for label, built in (('standard stem', model_on),
                             ('gabor stem', model_gabor_on)):
            leftover = {
                name for name in _bias_free_layer_names(built)
                if not _is_hardcoded_bias_free(name)
            }
            assert leftover == set(), (
                f"[{label}] these layers stayed bias-free under use_bias=True, so "
                f"use_bias was NOT threaded to them: {sorted(leftover)}"
            )

    def test_bias_off_leaves_every_site_bias_free(self):
        """The bias-off arm must not have gained a single biased layer."""
        model_off = create_convunext(use_bias=False, **_builder_config())
        biased = [
            layer.name
            for layer in model_off._flatten_layers()
            if getattr(layer, 'use_bias', None) is True
        ]
        assert biased == [], f"use_bias=False built biased layers: {biased}"


class TestCreateConvUNextBottleneckDropPath:
    """The bottleneck drop-path RAMP (plan_2026-07-10_be906be8/D-001)."""

    def test_bottleneck_drop_path_ramps(self):
        """Bottleneck block i carries rate ``drop_path_rate * i / blocks``.

        Read BY LAYER NAME, never by list index: a functional graph has no
        ``.bottleneck_blocks`` attribute, and index order is not a contract.
        Block 0's rate is 0.0, which means NO StochasticDepth layer is created
        at all -- that absence is part of the schedule and is asserted too.
        """
        blocks = 4
        rate = 0.4
        model = create_convunext(
            input_shape=(16, 16, 3), depth=3, initial_filters=8,
            blocks_per_level=blocks, drop_path_rate=rate, use_bias=False,
        )

        with pytest.raises(ValueError):
            model.get_layer(name='bottleneck_convnext_v2_block_0_drop_path')

        observed = []
        for i in range(1, blocks):
            layer = model.get_layer(
                name=f'bottleneck_convnext_v2_block_{i}_drop_path')
            assert isinstance(layer, StochasticDepth)
            observed.append(float(layer.drop_path_rate))

        expected = [rate * i / blocks for i in range(1, blocks)]
        np.testing.assert_allclose(
            observed, expected, rtol=1e-7,
            err_msg=(
                "the bottleneck drop-path schedule is not the LOCAL linear ramp "
                "drop_path_rate * block_idx / blocks_per_level -- a flat "
                f"(constant) rate would read {[rate] * (blocks - 1)}"
            ),
        )
        assert observed == sorted(observed) and observed[0] < observed[-1], (
            f"bottleneck drop-path rates must strictly increase, got {observed}")


class TestCreateConvUNextLiveness:
    """Dead-component probes: the builder must build a real network."""

    def test_output_is_not_the_input(self, sample_input_small):
        """A shape-only assertion cannot see ``keras.Model(inputs, inputs)``.

        ``output_channels == input_channels`` by construction here, so an
        identity model passes every shape check. This asserts VALUES.
        """
        model = create_convunext(
            input_shape=(64, 64, 3), depth=3, initial_filters=8,
            blocks_per_level=1, use_bias=False,
        )
        y = keras.ops.convert_to_numpy(model(sample_input_small, training=False))
        assert y.shape == sample_input_small.shape
        assert not np.allclose(y, sample_input_small, atol=1e-6), (
            "the model returned its input -- the graph is a pass-through")
        assert np.isfinite(y).all()

    def test_forward_pass_reaches_every_named_stage(self):
        """The stem, bottleneck and final projection all exist by NAME."""
        model = create_convunext(
            input_shape=(16, 16, 3), depth=3, initial_filters=8,
            blocks_per_level=1, use_bias=True,
        )
        for name in ('encoder_level_0_stem', 'bottleneck_channel_adjust',
                     'bottleneck_convnext_v2_block_0', 'final_output'):
            assert model.get_layer(name=name) is not None


# ---------------------------------------------------------------------
# Plan step 5: the three use_bias=False guardrails
# ---------------------------------------------------------------------

def _guard_config(**overrides) -> Dict[str, Any]:
    """The smallest config that still builds, for the guardrail tests.

    Deliberately tiny: most of these tests assert on a raise that happens BEFORE
    any layer is constructed, and the few that do build only need the graph to
    exist, not to be representative.
    """
    cfg = dict(
        input_shape=(16, 16, 3),
        depth=2,
        initial_filters=8,
        blocks_per_level=1,
    )
    cfg.update(overrides)
    return cfg


class TestBiasFreeGuardrails:
    """``_validate_bias_free_arguments`` fires on the bias-off arm only.

    Plan invariant I-6 / decisions.md D-006, D-012. The guard is an ALLOWLIST of
    positively homogeneous activations, deliberately narrow, deliberately silent
    on the model's own non-homogeneous ``'gelu'`` defaults, and deliberately
    non-raising for ``block_normalization='layernorm'``.
    """

    # -- (a) final_activation ------------------------------------------------

    def test_bias_free_rejects_non_homogeneous_final_activation(self):
        """A non-homogeneous string ``final_activation`` is a hard error."""
        with pytest.raises(ValueError, match=r"final_activation='sigmoid'"):
            create_convunext(use_bias=False, final_activation='sigmoid',
                             **_guard_config())

    def test_the_final_activation_error_names_the_allowlist(self):
        """The message must be actionable: argument, value AND allowlist."""
        with pytest.raises(ValueError) as excinfo:
            create_convunext(use_bias=False, final_activation='gelu',
                             **_guard_config())
        msg = str(excinfo.value)
        assert 'final_activation' in msg
        assert "'gelu'" in msg
        assert 'leaky_relu' in msg and 'linear' in msg and 'relu' in msg

    @pytest.mark.parametrize('activation', [None, 'linear', 'relu', 'leaky_relu'])
    def test_allowlisted_final_activations_build(self, activation):
        """Every allowlist member must survive the guard on the bias-off arm."""
        model = create_convunext(use_bias=False, final_activation=activation,
                                 **_guard_config())
        assert isinstance(model, keras.Model)

    # -- (b) gabor_activation, scoped to use_gabor_stem -----------------------

    def test_bias_free_rejects_non_homogeneous_gabor_activation(self):
        """With the Gabor stem ON, the activation reaches a layer, so it raises."""
        with pytest.raises(ValueError, match=r"gabor_activation='gelu'"):
            create_convunext(
                use_bias=False, use_gabor_stem=True, gabor_filters=4,
                gabor_activation='gelu', **_guard_config())

    def test_gabor_activation_guard_is_inert_when_gabor_stem_is_off(self):
        """POSITIVE liveness arm: the guard is SCOPED, not blanket.

        With ``use_gabor_stem=False`` the ``gabor_activation`` argument reaches no
        layer at all, so the network really is homogeneous and raising on it would
        be a false positive. This test goes RED if the ``use_gabor_stem`` condition
        is dropped from the guard.
        """
        model = create_convunext(
            use_bias=False, use_gabor_stem=False, gabor_activation='gelu',
            **_guard_config())
        assert isinstance(model, keras.Model)
        gabor_layers = [l for l in model._flatten_layers()
                        if 'gabor' in l.name]
        assert gabor_layers == [], (
            "use_gabor_stem=False must build no Gabor layer -- if one exists the "
            "argument is NOT inert and this test is measuring the wrong thing")

    # -- (c) supervision_norm_center -----------------------------------------

    def test_bias_free_rejects_supervision_norm_center(self):
        """``center=True`` on the supervision LayerNorm is a bias by another name."""
        with pytest.raises(ValueError, match=r"supervision_norm_center=True"):
            create_convunext(
                use_bias=False, enable_deep_supervision=True,
                supervision_norm_center=True, **_guard_config())

    def test_supervision_norm_center_raises_even_without_deep_supervision(self):
        """PINNED EDGE CASE (plan.md 'Edge cases'): the guard's predicate is a PURE
        FUNCTION OF ITS ARGUMENTS.

        With ``enable_deep_supervision=False`` no supervision head is built and
        ``supervision_norm_center`` reaches nothing -- and it STILL raises, because
        the caller stated a contradictory intent. This is a deliberate ruling, not
        an oversight. Do NOT "fix" it by gating the clause on
        ``enable_deep_supervision``; this test exists to stop exactly that edit.
        """
        with pytest.raises(ValueError, match=r"supervision_norm_center=True"):
            create_convunext(
                use_bias=False, enable_deep_supervision=False,
                supervision_norm_center=True, **_guard_config())

    # -- the bias-ON arm is untouched ----------------------------------------

    def test_bias_on_ignores_all_three_guards(self):
        """All three offending values at once must BUILD under ``use_bias=True``.

        Goes RED if the validator is called unconditionally.
        """
        model = create_convunext(
            use_bias=True,
            final_activation='sigmoid',
            use_gabor_stem=True,
            gabor_filters=4,
            gabor_activation='gelu',
            enable_deep_supervision=True,
            supervision_norm_center=True,
            **_guard_config())
        assert isinstance(model, keras.Model)

    # -- the soft guard: warn, never raise -----------------------------------

    def test_layernorm_under_bias_free_warns_but_builds(self, caplog):
        """Raise-vs-warn is a CONTRACT here, not a comment.

        ``'layernorm'`` is the shipped default of BOTH arms, so promoting this to a
        raise would take down every existing bias-free caller. The assertion is
        two-sided on purpose: the warning IS emitted AND the model IS returned.
        """
        with caplog.at_level(logging.WARNING, logger='dl'):
            model = create_convunext(
                use_bias=False, block_normalization='layernorm',
                **_guard_config())
        assert isinstance(model, keras.Model), (
            "block_normalization='layernorm' must BUILD, never raise")
        assert any("block_normalization='layernorm'" in r.getMessage()
                   for r in caplog.records if r.levelno >= logging.WARNING), (
            "no WARNING mentioning block_normalization='layernorm' was emitted; "
            f"records={[r.getMessage() for r in caplog.records]}")

    def test_batchnorm_under_bias_free_does_not_warn(self, caplog):
        """Control for the test above: the homogeneous choice must be silent."""
        with caplog.at_level(logging.WARNING, logger='dl'):
            create_convunext(use_bias=False, block_normalization='batchnorm',
                             **_guard_config())
        assert not any("block_normalization" in r.getMessage()
                       for r in caplog.records if r.levelno >= logging.WARNING)

    # -- callables cannot be checked statically ------------------------------

    def test_callable_final_activation_warns_but_does_not_raise(self, caplog):
        """A callable's homogeneity is a property of code the guard cannot read."""
        with caplog.at_level(logging.WARNING, logger='dl'):
            model = create_convunext(
                use_bias=False, final_activation=keras.activations.relu,
                **_guard_config())
        assert isinstance(model, keras.Model)
        assert any('final_activation' in r.getMessage() and 'callable' in r.getMessage()
                   for r in caplog.records if r.levelno >= logging.WARNING)

    # -- the shipped default must never raise --------------------------------

    def test_default_bias_free_config_builds(self):
        """The DEFAULTS (``final_activation='linear'``, ``gabor_activation=None``)
        are both in the allowlist, so the default bias-free configuration must
        build. A guard that fires on the shipped default is not a guard, it is an
        outage for every bias-free caller in the repo.
        """
        model = create_convunext(use_bias=False, **_guard_config())
        assert isinstance(model, keras.Model)


# ---------------------------------------------------------------------
# Step 6: the knobs absorbed from the deleted ConvUNextModel
# ---------------------------------------------------------------------


def _knob_config() -> Dict[str, Any]:
    """A small config for the absorbed-knob tests.

    ``initial_filters`` (8) is deliberately DIFFERENT from the input channel count
    (3) so that "the decoder feature width" and "the output channel count" are
    distinguishable numbers -- at ``initial_filters=3`` an ``include_top`` defect
    would be invisible.
    """
    return dict(
        input_shape=(16, 16, 3),
        depth=2,
        initial_filters=8,
        blocks_per_level=1,
        drop_path_rate=0.0,
    )


class TestStridedConvDownsample:
    """``downsample_pool_type='strided_conv'`` (absorbed from ConvUNextModel)."""

    def test_strided_conv_downsample_has_learnable_weights(self):
        """RED-proof target for aliasing ``'strided_conv'`` to ``'max'``.

        A ``Conv2D(k=2, s=2)`` and a ``MaxPooling2D(2)`` agree on OUTPUT SHAPE, so
        only the weight count separates them. Asserted at the junction layer itself
        AND at the whole-model parameter count, because a junction that is built
        correctly but never applied would still report its own weights.
        """
        cfg = _knob_config()
        model_conv = create_convunext(downsample_pool_type='strided_conv', **cfg)
        model_pool = create_convunext(downsample_pool_type='max', **cfg)

        junction = model_conv.get_layer(name='encoder_downsample_0')
        assert len(junction.trainable_weights) > 0, (
            "the strided-conv junction must carry learnable weights; a pooling "
            "junction has zero")

        assert model_pool.get_layer(name='encoder_downsample_0').trainable_weights == []
        assert model_conv.count_params() > model_pool.count_params(), (
            f"strided_conv={model_conv.count_params()} must exceed "
            f"max={model_pool.count_params()}")

    def test_strided_conv_is_channel_preserving(self):
        """decisions.md D-013: this junction does NOT widen channels.

        The deleted ``ConvUNextModel._downsample`` fused downsampling with the
        channel widening (``filters=next_level_filters``). Keeping channels is what
        keeps the builder's SEPARATE ``encoder_level_N_channel_adjust`` step valid.
        """
        cfg = _knob_config()
        model = create_convunext(downsample_pool_type='strided_conv', **cfg)
        junction = model.get_layer(name='encoder_downsample_0')

        assert junction.conv.filters == cfg['initial_filters'], (
            "the level-0 junction must preserve 8 channels, not widen to the level-1 "
            f"width 16; got filters={junction.conv.filters}")
        assert model.get_layer(
            name='encoder_level_1_channel_adjust').filters == 16, (
            "the separate channel-adjust step must still do the widening")

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_strided_conv_threads_use_bias(self, use_bias):
        """The strided conv is a real learnable conv, so it takes ``use_bias``.

        Unlike the two ruled-hardcoded bias-free sites (``SpatialLinearAttention``,
        the frozen Gabor bank), this one MUST follow the argument -- asserted on the
        allocated WEIGHT, not only the flag.
        """
        model = create_convunext(
            use_bias=use_bias, downsample_pool_type='strided_conv', **_knob_config())
        conv = model.get_layer(name='encoder_downsample_0').conv

        assert conv.use_bias is use_bias
        assert (conv.bias is not None) is use_bias
        assert len(conv.weights) == (2 if use_bias else 1)

    def test_strided_conv_is_legal_on_the_bias_free_arm(self):
        """A bias-free strided conv is linear, hence homogeneous: NOT guarded.

        The guardrails must stay silent here, and no layer in the built model may
        carry a bias.
        """
        model = create_convunext(
            use_bias=False, downsample_pool_type='strided_conv', **_knob_config())
        biased = [
            layer.name for layer in model._flatten_layers()
            if getattr(layer, 'use_bias', None) is True
        ]
        assert biased == [], f"bias-free strided_conv built biased layers: {biased}"

    def test_invalid_downsample_pool_type_still_raises(self):
        with pytest.raises(ValueError, match='downsample_pool_type'):
            create_convunext(downsample_pool_type='median', **_knob_config())


class TestIncludeTop:
    """``include_top`` (absorbed from ConvUNextModel, with a documented divergence)."""

    def test_include_top_false_output_channels_match_the_decoder_width(self):
        """The headless output is the DECODER FEATURE MAP, not a 3-channel image."""
        cfg = _knob_config()
        model = create_convunext(include_top=False, **cfg)

        assert model.output_shape[-1] == cfg['initial_filters'], (
            f"include_top=False must emit the decoder width "
            f"({cfg['initial_filters']}), got {model.output_shape[-1]}")
        assert model.output_shape[1:3] == (16, 16), (
            "the headless output must stay at full resolution")
        assert model.get_layer(name='decoder_features') is not None

    def test_include_top_false_does_not_construct_the_final_projection(self):
        """DIVERGENCE PIN (decisions.md D-013), not an oversight.

        ``ConvUNextModel`` CONSTRUCTED its final projection under
        ``include_top=False`` and merely skipped applying it, so its headless variant
        still carried the head's weights. A FUNCTIONAL graph cannot reproduce that:
        ``keras.Model(inputs, outputs)`` prunes every layer that is not on a path to
        an output, so a constructed-but-unapplied layer owns no weights and is not
        reachable. This test pins the contract we ACTUALLY ship, so that a future
        reader cannot quietly "restore" a weight-compat guarantee the graph
        cannot hold. Measured, not assumed: an experiment that constructed AND
        applied the projection on a dead branch left this assertion GREEN, because
        Keras pruned the layer out of the model exactly as described.
        """
        model = create_convunext(include_top=False, **_knob_config())

        with pytest.raises(ValueError):
            model.get_layer(name='final_output')

        assert 'final_output' not in {layer.name for layer in model.layers}

    def test_include_top_false_weight_list_is_strictly_smaller(self):
        """The measurable consequence of the divergence above.

        Weights trained with ``include_top=True`` do NOT load into an
        ``include_top=False`` model -- ``set_weights`` raises on the length mismatch.
        Asserting the raise makes the broken contract a tested fact rather than a
        comment.
        """
        cfg = _knob_config()
        model_top = create_convunext(include_top=True, **cfg)
        model_headless = create_convunext(include_top=False, **cfg)

        assert len(model_headless.weights) < len(model_top.weights)
        assert model_headless.count_params() < model_top.count_params()

        with pytest.raises(ValueError):
            model_headless.set_weights(model_top.get_weights())

    def test_include_top_false_rejects_final_projection_groups(self):
        """A silently inert argument is this repo's recorded defect class."""
        with pytest.raises(ValueError, match='final_projection_groups'):
            create_convunext(
                include_top=False, final_projection_groups=2, **_knob_config())

    def test_include_top_true_is_the_default_and_projects(self):
        """CONTROL: the default must still emit the image-shaped output."""
        model = create_convunext(**_knob_config())
        assert model.output_shape[-1] == 3
        assert model.get_layer(name='final_output').filters == 3
        # Step 9's dead-component probe: a lone `Conv2D(3, 1, name='final_output')`
        # satisfies both lines above, so the graph itself has to be asserted.
        _assert_is_a_convunext_graph(model)


class TestOutputChannels:
    """``output_channels`` (absorbed from ConvUNextModel)."""

    def test_output_channels_override(self):
        """RED-proof target for ignoring the argument.

        Pinned at 1 against a 3-channel input, so the override and the default are
        different numbers.
        """
        model = create_convunext(output_channels=1, **_knob_config())

        assert model.output_shape[-1] == 1, (
            f"output_channels=1 must produce 1 output channel, got "
            f"{model.output_shape[-1]} (3 == the input channel count means the "
            f"argument was ignored)")
        assert model.get_layer(name='final_output').filters == 1
        _assert_is_a_convunext_graph(model)

    def test_output_channels_defaults_to_the_input_channel_count(self):
        """CONTROL: the denoiser/autoencoder contract every caller relies on."""
        model = create_convunext(**_knob_config())
        assert model.output_shape[-1] == 3
        _assert_is_a_convunext_graph(model)
        assert create_convunext(
            **dict(_knob_config(), input_shape=(16, 16, 5))
        ).output_shape[-1] == 5

    def test_output_channels_reaches_the_deep_supervision_heads(self):
        """The supervision heads emit ``output_channels`` too, not input channels.

        A test that only reads the primary output would be blind to a supervision
        head still hardcoded to ``input_shape[-1]``.
        """
        model = create_convunext(
            output_channels=1, enable_deep_supervision=True, depth=3,
            **{k: v for k, v in _knob_config().items() if k != 'depth'})

        assert model.get_layer(name='supervision_output_level_1').filters == 1
        for shape in model.output_shape:
            assert shape[-1] == 1, f"a supervision output kept 3 channels: {shape}"

    def test_output_channels_sizes_the_extra_zero_tail(self):
        """``extra_zero_output_channels`` slices the LAST ``output_channels``."""
        model = create_convunext(
            output_channels=1, extra_zero_output_channels=True, **_knob_config())
        assert model.output_shape[-1] == 1
        assert model.get_layer(name='extra_zero_output_pad').target_channels == 9

    @pytest.mark.parametrize('bad', [0, -1, 2.0, '3'])
    def test_invalid_output_channels_raises(self, bad):
        with pytest.raises(ValueError, match='output_channels'):
            create_convunext(output_channels=bad, **_knob_config())


class TestAbsorbedKnobsRoundTrip:
    """All three knobs at once must survive a `.keras` round trip BY VALUE."""

    def test_round_trip_at_non_default_knob_values(self):
        cfg = dict(
            _knob_config(),
            downsample_pool_type='strided_conv',
            include_top=False,
            output_channels=1,
            use_bias=False,
        )
        model = create_convunext(**cfg)
        x = np.random.default_rng(11).normal(size=(2, 16, 16, 3)).astype('float32')
        before = keras.ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'knobs.keras')
            model.save(path)
            reloaded = keras.models.load_model(path)

        after = keras.ops.convert_to_numpy(reloaded(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-6)

        junction = reloaded.get_layer(name='encoder_downsample_0')
        assert junction.pool_type == 'strided_conv'
        assert junction.use_bias is False
        assert junction.conv.bias is None
        assert reloaded.output_shape[-1] == 8


# =====================================================================
# Step 9: coverage re-derived against `create_convunext`
#
# The classes below replace the ConvUNextModel-era classes step 4 had to
# delete. Each names the class it re-derives. Two were deliberately NOT
# re-derived; both reasons are recorded in the module docstring.
# =====================================================================


def _cfg(**overrides) -> Dict[str, Any]:
    """The shared tiny build config, with overrides.

    Reuses :func:`_knob_config` rather than restating its numbers -- the
    ``initial_filters`` (8) != input channels (3) property it documents is
    load-bearing for every test below that distinguishes "the decoder width"
    from "the output channel count".
    """
    return dict(_knob_config(), **overrides)


def _convnext_block_layers(model: keras.Model):
    """Every ConvNeXt block layer in ``model``, in graph order."""
    return [
        layer for layer in model.layers
        if isinstance(layer, (ConvNextV1Block, ConvNextV2Block))
    ]


def _assert_is_a_convunext_graph(
        model: keras.Model, depth: int = 2, blocks_per_level: int = 1) -> None:
    """Assert ``model`` is the ConvUNext graph, not some smaller stand-in.

    WHY THIS EXISTS (step 9's dead-component probe, measured): substituting a
    single ``Conv2D`` for the builder's whole return value left every
    forward-pass, gradient-flow, weight-transfer and training-integration test
    GREEN -- a 1x1 conv is finite, input-dependent, fits, round-trips and
    matches the output shape. Those tests were measuring nothing about the
    ConvUNext graph. Calling this helper is what makes each of them fire.

    :param model: The built model.
    :param depth: The ``depth`` it was built with.
    :param blocks_per_level: The ``blocks_per_level`` it was built with.
    """
    expected_blocks = (2 * depth + 1) * blocks_per_level
    n_blocks = len(_convnext_block_layers(model))
    assert n_blocks == expected_blocks, (
        f'expected {expected_blocks} ConvNeXt blocks (depth={depth}, '
        f'blocks_per_level={blocks_per_level}), found {n_blocks}')
    assert model.get_layer(name='bottleneck_downsample') is not None
    for level in range(depth - 1):
        assert model.get_layer(name=f'encoder_downsample_{level}') is not None
    for level in range(depth):
        assert model.get_layer(name=f'decoder_upsample_{level}') is not None


def _drop_path_rate_by_name(model: keras.Model, name: str):
    """The ``StochasticDepth`` rate at ``name``, or ``None`` if no such layer."""
    try:
        layer = model.get_layer(name=name)
    except ValueError:
        return None
    assert isinstance(layer, StochasticDepth), (
        f"'{name}' is a {type(layer).__name__}, not a StochasticDepth -- the "
        f"drop-path schedule is reaching the wrong object")
    return float(layer.drop_path_rate)


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelInstantiation (4)
# ---------------------------------------------------------------------

class TestCreateConvUNextInstantiation:
    """Construction and argument validation, symmetric across both arms."""

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_default_build(self, use_bias: bool) -> None:
        model = create_convunext(use_bias=use_bias, **_cfg())
        assert isinstance(model, keras.Model)
        assert model.name == 'convunext'
        assert model.output_shape == (None, 16, 16, 3)
        _assert_is_a_convunext_graph(model)

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_custom_parameters_reach_the_graph(self, use_bias: bool) -> None:
        """``depth`` / ``initial_filters`` / ``filter_multiplier`` /
        ``blocks_per_level`` are read back off the BUILT graph, not off a
        config dict -- a functional model keeps no constructor arguments, so an
        argument that never reached a layer is invisible to any other check.
        """
        model = create_convunext(
            use_bias=use_bias, input_shape=(16, 16, 3), depth=3,
            initial_filters=6, filter_multiplier=2.0, blocks_per_level=2,
            drop_path_rate=0.0,
        )
        # depth=3 -> junctions at levels 0,1 plus the bottleneck junction.
        for name in ('encoder_downsample_0', 'encoder_downsample_1',
                     'bottleneck_downsample'):
            assert model.get_layer(name=name) is not None
        with pytest.raises(ValueError):
            model.get_layer(name='encoder_downsample_2')

        assert model.get_layer(name='encoder_level_0_stem').filters == 6
        # filter_multiplier=2.0 -> 6, 12, 24 ...
        assert model.get_layer(name='encoder_level_1_channel_adjust').filters == 12
        assert model.get_layer(name='encoder_level_2_channel_adjust').filters == 24

        # depth*bpl encoder + bpl bottleneck + depth*bpl decoder
        assert len(_convnext_block_layers(model)) == 3 * 2 + 2 + 3 * 2

    @pytest.mark.parametrize('version,block_cls', [
        ('v1', ConvNextV1Block), ('v2', ConvNextV2Block)])
    def test_convnext_version_selects_the_block_class(
            self, version: str, block_cls: type) -> None:
        """Asserted on the built layer TYPE. The two block classes agree on
        every shape, so a shape or name check cannot tell them apart.
        """
        model = create_convunext(**_cfg(convnext_version=version))
        blocks = _convnext_block_layers(model)
        assert blocks, 'no ConvNeXt block was built at all'
        assert all(type(b) is block_cls for b in blocks), (
            f"convnext_version='{version}' built "
            f"{sorted({type(b).__name__ for b in blocks})}")

    @pytest.mark.parametrize('kwargs,exc,match', [
        (dict(depth=1), ValueError, 'depth'),
        (dict(initial_filters=0), ValueError, 'initial_filters'),
        (dict(filter_multiplier=0.5), ValueError, 'filter_multiplier'),
        (dict(blocks_per_level=0), ValueError, 'blocks_per_level'),
        (dict(convnext_version='v3'), ValueError, 'convnext_version'),
        (dict(high_freq_blocks=-1), ValueError, 'high_freq_blocks'),
        (dict(bottleneck_attention_blocks=-1), ValueError,
         'bottleneck_attention_blocks'),
        (dict(input_shape=(16, 16)), TypeError, 'input_shape'),
    ])
    def test_invalid_arguments_raise(self, kwargs, exc, match) -> None:
        with pytest.raises(exc, match=match):
            create_convunext(**_cfg(**kwargs))


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelForwardPass (4) + ComputeOutputShape (3)
# ---------------------------------------------------------------------

class TestForwardPass:
    """Forward behaviour on both arms.

    ``compute_output_shape`` has no functional-graph equivalent as a METHOD
    under test (a ``keras.Model`` inherits Keras's own, which this repo does
    not own), so the three ``TestConvUNextModelComputeOutputShape`` tests are
    re-derived here as ``model.output_shape`` / actual-output agreement -- the
    property they were actually checking.
    """

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_forward_output_is_finite_and_input_dependent(
            self, use_bias: bool) -> None:
        model = create_convunext(use_bias=use_bias, **_cfg())
        rng = np.random.default_rng(3)
        x1 = rng.normal(size=(2, 16, 16, 3)).astype('float32')
        x2 = rng.normal(size=(2, 16, 16, 3)).astype('float32')

        y1 = keras.ops.convert_to_numpy(model(x1, training=False))
        y2 = keras.ops.convert_to_numpy(model(x2, training=False))

        assert y1.shape == (2, 16, 16, 3)
        assert np.isfinite(y1).all()
        assert float(np.max(np.abs(y1 - y2))) > 1e-6, (
            'the output does not depend on the input')
        assert float(np.max(np.abs(y1))) > 0.0, 'the output is identically zero'
        _assert_is_a_convunext_graph(model)

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_the_receptive_field_spans_the_whole_image(
            self, use_bias: bool) -> None:
        """A U-Net's corner output pixel must depend on the OPPOSITE corner.

        This is the one forward-pass assertion that a stand-in cannot fake by
        having the right output shape: for a 1x1 (or any small) convolution the
        measured delta is EXACTLY 0.0, because output pixel (0, 0) reads only a
        local neighbourhood of the input. Passing it requires the encoder's
        downsampling path to actually be present and to actually mix the whole
        image. Measured here at 3.0e-3 (bias-on) / 7.5e-3 (bias-off) against a
        threshold of 1e-5.
        """
        model = create_convunext(use_bias=use_bias, **_cfg())
        x = np.random.default_rng(101).normal(
            size=(1, 16, 16, 3)).astype('float32')
        x_perturbed = x.copy()
        x_perturbed[0, 15, 15, :] += 5.0

        y = keras.ops.convert_to_numpy(model(x, training=False))
        y_perturbed = keras.ops.convert_to_numpy(
            model(x_perturbed, training=False))

        corner_delta = float(np.max(np.abs(y_perturbed - y)[0, 0, 0]))
        assert corner_delta > 1e-5, (
            'perturbing the input at (15, 15) did not change the output at '
            f'(0, 0) (delta={corner_delta}); the receptive field is local, so '
            'the downsampling encoder path is missing or disconnected')

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_declared_output_shape_matches_the_actual_output(
            self, use_bias: bool) -> None:
        model = create_convunext(use_bias=use_bias, **_cfg(output_channels=2))
        x = np.zeros((3, 16, 16, 3), dtype='float32')
        actual = keras.ops.convert_to_numpy(model(x, training=False)).shape
        assert model.output_shape == (None, 16, 16, 2)
        assert actual == (3, 16, 16, 2)
        _assert_is_a_convunext_graph(model)

    def test_variable_batch_sizes(self) -> None:
        model = create_convunext(**_cfg())
        _assert_is_a_convunext_graph(model)
        for batch in (1, 2, 5):
            x = np.zeros((batch, 16, 16, 3), dtype='float32')
            assert model(x, training=False).shape == (batch, 16, 16, 3)

    def test_training_and_inference_agree_when_no_stochastic_layer_is_active(
            self) -> None:
        """CONTROL for the round-trip tests: at ``drop_path_rate=0.0`` and
        ``dropout_rate=0.0`` with the default LayerNorm, the graph carries NO
        train/inference-divergent layer, so the two modes must agree exactly.
        Without this control, an ``allclose`` failure elsewhere could not be
        attributed to serialization rather than to mode.
        """
        model = create_convunext(**_cfg())
        _assert_is_a_convunext_graph(model)
        x = np.random.default_rng(5).normal(size=(2, 16, 16, 3)).astype('float32')
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(model(x, training=True)),
            keras.ops.convert_to_numpy(model(x, training=False)),
            rtol=1e-6, atol=1e-6)

    def test_non_square_input(self) -> None:
        model = create_convunext(**_cfg(input_shape=(16, 32, 3)))
        _assert_is_a_convunext_graph(model)
        x = np.zeros((1, 16, 32, 3), dtype='float32')
        assert model(x, training=False).shape == (1, 16, 32, 3)


class TestGradientFlow:
    """Every trainable weight must receive a gradient.

    A ``None`` gradient is how a disconnected sub-graph (a layer built but not
    applied, an output that never reaches the loss) shows up; a
    forward-pass-only suite is completely blind to it.
    """

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_all_trainable_weights_receive_a_gradient(self, use_bias: bool) -> None:
        model = create_convunext(use_bias=use_bias, **_cfg())
        _assert_is_a_convunext_graph(model)
        x = tf.constant(
            np.random.default_rng(7).normal(size=(2, 16, 16, 3)).astype('float32'))

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True)))
        grads = tape.gradient(loss, model.trainable_variables)

        missing = [
            v.path for v, g in zip(model.trainable_variables, grads) if g is None]
        assert missing == [], f'no gradient reached: {missing}'

        nonzero = sum(
            1 for g in grads if float(tf.reduce_max(tf.abs(g))) > 0.0)
        assert nonzero > 0.5 * len(grads), (
            f'only {nonzero}/{len(grads)} weights got a NONZERO gradient')


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextResidualAndDropPathWiring (4)
# ---------------------------------------------------------------------

class TestResidualAndDropPathWiring:
    """``ConvNextV1Block`` / ``ConvNextV2Block`` are the residual BRANCH only.

    The caller must supply ``x + StochasticDepth(rate)(block(x))``
    (``convnext_v1_block.py:103-121``; anchors
    ``plan-2026-08-11T201945-91938f65/D-002 + D-004`` on
    ``_apply_residual_convnext_block``). The deleted subclass pinned this by
    swapping ``model.encoder_stages[i][j]`` for a zero-output stand-in; a
    functional graph has no such list, so the equivalents below (a) zero the
    branch through its LayerScale ``gamma`` and (b) read the ``Add``'s operands
    out of the graph.
    """

    @staticmethod
    def _zero_every_branch(model: keras.Model) -> int:
        """Force every ConvNeXt residual branch to output exactly zero.

        Each block ends in ``gamma * x`` (``LearnableMultiplier`` named
        ``gamma_scale``), so zeroing that variable annihilates the branch while
        leaving the rest of the graph -- stem, junctions, channel adjusts,
        upsamples, final projection -- untouched. Returns the count zeroed.
        """
        zeroed = 0
        for w in model.weights:
            if 'gamma_scale' in w.path:
                w.assign(keras.ops.zeros_like(w))
                zeroed += 1
        return zeroed

    def test_residual_skip_is_present_at_every_block(self) -> None:
        """BEHAVIOURAL detector, not a name check.

        With every ConvNeXt branch output forced to exactly zero, the ONLY route
        from input to output is the additive skip inside each block loop. If any
        loop is wired ``x = block(x)`` instead, the activation becomes exactly 0
        at that block, every downstream skip is 0, and the output collapses to a
        constant that does not depend on the input at all. This repo has a
        recorded ~1700x signal-collapse defect from exactly that missing
        external residual, and it was invisible to every shape and
        serialization test.
        """
        model = create_convunext(**_cfg(depth=3, blocks_per_level=2))
        rng = np.random.default_rng(13)
        x1 = rng.normal(size=(2, 16, 16, 3)).astype('float32')
        x2 = rng.normal(size=(2, 16, 16, 3)).astype('float32')

        zeroed = self._zero_every_branch(model)
        # depth=3, bpl=2 -> 6 encoder + 2 bottleneck + 6 decoder blocks.
        assert zeroed == 14, f'expected 14 gamma_scale weights, zeroed {zeroed}'

        y1 = keras.ops.convert_to_numpy(model(x1, training=False))
        y2 = keras.ops.convert_to_numpy(model(x2, training=False))

        spread = float(np.max(np.abs(y1 - y2)))
        assert spread > 1e-4, (
            'with every ConvNeXt branch zeroed the output no longer depends on '
            f'the input (max|f(x1) - f(x2)| = {spread}); the residual skip is '
            'missing from at least one block loop')
        # And the signal did not merely survive -- it kept its MAGNITUDE. A
        # residual chain that is present but scaled would pass the check above.
        assert float(np.max(np.abs(y1))) > 1e-3, (
            f'output magnitude collapsed to {float(np.max(np.abs(y1)))}')

    def test_each_residual_add_takes_the_block_input_and_the_block_output(
            self) -> None:
        """STRUCTURAL control for the test above, isolating the operands.

        The behavioural probe proves *some* path survives; this proves the two
        operands of every ``*_residual`` Add are exactly (block input, block
        output) -- catching a residual wired from the wrong tensor, which the
        magnitude probe cannot see.
        """
        model = create_convunext(**_cfg(depth=3, blocks_per_level=2))
        blocks = _convnext_block_layers(model)
        assert blocks

        for block in blocks:
            add = model.get_layer(name=f'{block.name}_residual')
            assert isinstance(add, keras.layers.Add)
            operands = add.input
            assert isinstance(operands, list) and len(operands) == 2
            assert operands[0] is block.input, (
                f'{block.name}: the residual operand is not the block INPUT')
            assert operands[1] is block.output, (
                f'{block.name}: the branch operand is not the block OUTPUT')

    def test_drop_path_schedule_reaches_stochastic_depth(self) -> None:
        """The schedule must land on ``StochasticDepth``, by NAME, with the
        expected RATE -- not on the block's own ``dropout_rate``.

        ``drop_path_rate`` is deliberately non-zero: an all-zeros schedule is
        invariant under shift and reversal, which would make this vacuous.
        """
        rate, depth, bpl = 0.4, 3, 2
        model = create_convunext(
            **_cfg(depth=depth, blocks_per_level=bpl, drop_path_rate=rate))

        def scheduled(level: int, idx: int) -> float:
            return rate * (level * bpl + idx) / (depth * bpl)

        seen = []
        # Encoder: a GLOBAL linear ramp across depth. rate 0.0 means the layer
        # is not created at all -- that absence is part of the schedule.
        for level in range(depth):
            for idx in range(bpl):
                name = (f'encoder_level_{level}_convnext_v2_block_{idx}'
                        f'_drop_path')
                expected = scheduled(level, idx)
                observed = _drop_path_rate_by_name(model, name)
                if expected == 0.0:
                    assert observed is None, (
                        f'{name} exists at rate 0.0; it must not be created')
                else:
                    assert observed == pytest.approx(expected), name
                    seen.append(observed)

        # Bottleneck: a LOCAL ramp restarting at 0.0.
        for idx in range(bpl):
            name = f'bottleneck_convnext_v2_block_{idx}_drop_path'
            expected = rate * idx / bpl
            observed = _drop_path_rate_by_name(model, name)
            if expected == 0.0:
                assert observed is None, f'{name} exists at rate 0.0'
            else:
                assert observed == pytest.approx(expected), name
                seen.append(observed)

        # Decoder: mirrors the encoder ramp, EXCEPT block 0 of every level,
        # which deliberately carries none.
        for level in range(depth):
            for idx in range(bpl):
                name = (f'decoder_level_{level}_convnext_v2_block_{idx}'
                        f'_drop_path')
                expected = 0.0 if idx == 0 else scheduled(level, idx)
                observed = _drop_path_rate_by_name(model, name)
                if expected == 0.0:
                    assert observed is None, (
                        f'{name} exists; decoder block 0 must carry NO '
                        'stochastic depth')
                else:
                    assert observed == pytest.approx(expected), name
                    seen.append(observed)

        assert seen, 'no StochasticDepth layer carried a non-zero rate at all'
        assert max(seen) <= rate, (
            f'a scheduled rate {max(seen)} exceeds drop_path_rate {rate}')
        assert len(set(seen)) > 1, (
            f'the schedule is FLAT ({seen[0]} everywhere), not a ramp')

    def test_blocks_own_dropout_rate_stays_zero(self) -> None:
        """The block's ``dropout_rate`` is ordinary MLP dropout INSIDE the
        inverted bottleneck, not stochastic depth. A schedule that leaked into
        it would still "work" and would still be wrong.
        """
        model = create_convunext(**_cfg(drop_path_rate=0.4))
        blocks = _convnext_block_layers(model)
        assert blocks
        for block in blocks:
            assert block.dropout_rate == 0.0, (
                f'{block.name} carries dropout_rate={block.dropout_rate}; the '
                'drop-path schedule leaked into the block MLP dropout')

    def test_explicit_dropout_rate_is_a_separate_knob(self) -> None:
        """CONTROL for the test above: ``dropout_rate`` DOES reach the blocks
        when asked, so the assertion there is measuring the schedule, not a
        knob that never works.
        """
        model = create_convunext(**_cfg(dropout_rate=0.25, drop_path_rate=0.4))
        blocks = _convnext_block_layers(model)
        assert blocks
        assert all(b.dropout_rate == 0.25 for b in blocks)


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelDeepSupervision (4)
# ---------------------------------------------------------------------

class TestDeepSupervision:
    """Output COUNT and ORDER, plus the ``expose_bottleneck`` tail."""

    def test_disabled_gives_exactly_one_output(self) -> None:
        model = create_convunext(**_cfg(enable_deep_supervision=False))
        assert len(model.outputs) == 1
        assert model.output_shape == (None, 16, 16, 3)
        _assert_is_a_convunext_graph(model)
        assert not [l for l in model.layers if 'supervision' in l.name]

    @pytest.mark.parametrize('depth', [2, 3, 4])
    def test_output_count_is_one_plus_depth_minus_one(self, depth: int) -> None:
        model = create_convunext(
            **_cfg(depth=depth, enable_deep_supervision=True))
        assert len(model.outputs) == depth, (
            f'depth={depth} must give 1 final + {depth - 1} supervision outputs')

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_output_order_is_final_then_shallowest_to_deepest(
            self, use_bias: bool) -> None:
        """ORDER is a contract every deep-supervision loss depends on, and it is
        pinned by TENSOR IDENTITY, not by shape: at ``filter_multiplier=1`` two
        supervision heads could share a shape and a shape check would pass under
        a swap.
        """
        depth = 4
        model = create_convunext(
            use_bias=use_bias,
            **_cfg(depth=depth, enable_deep_supervision=True))

        assert model.outputs[0] is model.get_layer(name='final_output').output
        for i in range(1, depth):
            head = model.get_layer(name=f'supervision_output_level_{i}')
            assert model.outputs[i] is head.output, (
                f'output {i} is not supervision level {i} -- the supervision '
                'outputs are mis-ordered')

        # Resolution decreases monotonically down the list (shallow -> deep).
        heights = [shape[1] for shape in model.output_shape]
        assert heights == [16, 8, 4, 2], heights

    def test_expose_bottleneck_appends_the_bottleneck_last(self) -> None:
        model = create_convunext(
            **_cfg(depth=3, enable_deep_supervision=True, expose_bottleneck=True))
        assert len(model.outputs) == 3 + 1
        assert model.outputs[-1] is model.get_layer(name='bottleneck').output
        # depth=3 -> the bottleneck is at 16 / 2**3.
        assert model.output_shape[-1][1:3] == (2, 2)

    def test_expose_bottleneck_without_deep_supervision(self) -> None:
        model = create_convunext(**_cfg(expose_bottleneck=True))
        assert len(model.outputs) == 2
        assert model.outputs[0] is model.get_layer(name='final_output').output
        assert model.outputs[1] is model.get_layer(name='bottleneck').output

    def test_deep_supervision_with_include_top_false(self) -> None:
        """The primary output becomes the decoder feature map; the supervision
        heads are unaffected and still emit ``output_channels``.
        """
        model = create_convunext(
            **_cfg(depth=3, enable_deep_supervision=True, include_top=False))
        assert model.outputs[0] is model.get_layer(
            name='decoder_features').output
        assert model.output_shape[0][-1] == 8
        assert model.output_shape[1][-1] == 3
        assert model.output_shape[2][-1] == 3


# ---------------------------------------------------------------------
# The optional structural branches, each on BOTH arms
# ---------------------------------------------------------------------

class TestStructuralBranches:
    """``zero_pad_channels``, ``extra_zero_output_channels``,
    ``final_projection_groups``, ``use_laplacian_pyramid`` +
    ``high_freq_blocks``, ``bottleneck_attention_blocks``.

    These are the branches step 4's numeric control exercised against the
    pre-merge build, so each needs at least one guard on each arm -- a branch
    threaded on only one arm is exactly the defect this suite is for.
    """

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_zero_pad_channels_replaces_the_channel_adjust_convs(
            self, use_bias: bool) -> None:
        """The padded path must be PARAMETER-FREE where the conv was, and the
        model must lose parameters as a result -- a layer-name check alone
        cannot see a MatchChannels that was inserted without removing the conv.
        """
        cfg = _cfg(depth=3)
        padded = create_convunext(use_bias=use_bias, zero_pad_channels=True, **cfg)
        conved = create_convunext(use_bias=use_bias, **cfg)

        match = padded.get_layer(name='encoder_level_1_match_channels')
        assert isinstance(match, MatchChannels)
        assert match.trainable_weights == []
        with pytest.raises(ValueError):
            padded.get_layer(name='encoder_level_1_channel_adjust')

        assert conved.get_layer(name='encoder_level_1_channel_adjust') is not None
        assert padded.count_params() < conved.count_params()

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_extra_zero_output_channels_drops_the_learned_projection(
            self, use_bias: bool) -> None:
        model = create_convunext(
            use_bias=use_bias, extra_zero_output_channels=True, **_cfg())
        assert model.get_layer(name='final_output_tail_slice') is not None
        with pytest.raises(ValueError):
            model.get_layer(name='final_output')
        assert model.output_shape[-1] == 3

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_final_projection_groups_reaches_the_final_conv(
            self, use_bias: bool) -> None:
        """Asserted on ``groups`` AND on the parameter count: a grouped 1x1
        conv has strictly fewer kernel parameters than the ungrouped one, and
        both agree on output shape.
        """
        cfg = _cfg(output_channels=4)
        grouped = create_convunext(
            use_bias=use_bias, final_projection_groups=2, **cfg)
        plain = create_convunext(use_bias=use_bias, **cfg)

        assert grouped.get_layer(name='final_output').groups == 2
        assert plain.get_layer(name='final_output').groups == 1
        assert grouped.count_params() < plain.count_params()
        assert grouped.output_shape == plain.output_shape

    def test_final_projection_groups_must_divide_both_channel_counts(self) -> None:
        with pytest.raises(ValueError, match='final_projection_groups'):
            create_convunext(**_cfg(output_channels=3, final_projection_groups=2))

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_laplacian_pyramid_with_high_freq_blocks(self, use_bias: bool) -> None:
        """``high_freq_blocks`` only exists on the pyramid path, and its
        drop-path ramp is LOCAL (restarts at 0.0 per level).
        """
        model = create_convunext(
            use_bias=use_bias, use_laplacian_pyramid=True, high_freq_blocks=2,
            **_cfg(depth=3, drop_path_rate=0.4))

        for level in range(3):
            assert model.get_layer(
                name=f'skip_highfreq_block_{level}_0') is not None
            assert model.get_layer(
                name=f'skip_highfreq_block_{level}_1') is not None
            # Local ramp: hf_idx 0 -> 0.0 (no layer), hf_idx 1 -> rate * 1/2.
            assert _drop_path_rate_by_name(
                model, f'skip_highfreq_block_{level}_0_drop_path') is None
            assert _drop_path_rate_by_name(
                model, f'skip_highfreq_block_{level}_1_drop_path') == pytest.approx(0.2)

        x = np.zeros((1, 16, 16, 3), dtype='float32')
        assert model(x, training=False).shape == (1, 16, 16, 3)

    def test_high_freq_blocks_are_inert_without_the_pyramid(self) -> None:
        """The argument is gated on ``use_laplacian_pyramid`` -- without a
        pyramid there is no high band to process, so it must add NO layer.
        """
        model = create_convunext(**_cfg(high_freq_blocks=2))
        assert not [l for l in model.layers if 'highfreq' in l.name]
        # POSITIVE control: the rest of the graph is still there, so this is an
        # assertion about the ARGUMENT being inert, not about an empty model.
        _assert_is_a_convunext_graph(model)

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_bottleneck_attention_blocks(self, use_bias: bool) -> None:
        """The attention stack is built, is the right CLASS, and its internal
        projections stay bias-free on BOTH arms (a bias there breaks the
        Miyasawa property -- ``plan_2026-07-11_bb4b38b5/D-001``).
        """
        model = create_convunext(
            use_bias=use_bias, bottleneck_attention_blocks=2,
            bottleneck_attention_heads=2, **_cfg(drop_path_rate=0.4))

        block0 = model.get_layer(name='bottleneck_attention_block_0')
        assert isinstance(block0, SpatialLinearAttention)
        assert isinstance(
            model.get_layer(name='bottleneck_attention_add_0'), keras.layers.Add)
        # Local ramp: attn_idx 0 -> 0.0 (no SD layer), attn_idx 1 -> 0.4 * 1/2.
        assert _drop_path_rate_by_name(model, 'bottleneck_attention_sd_0') is None
        assert _drop_path_rate_by_name(
            model, 'bottleneck_attention_sd_1') == pytest.approx(0.2)

        biased = [l.name for l in block0._flatten_layers()
                  if getattr(l, 'use_bias', None) is True]
        assert biased == [], (
            f'SpatialLinearAttention must stay bias-free on both arms: {biased}')

    def test_bottleneck_attention_heads_must_divide_the_bottleneck_width(
            self) -> None:
        with pytest.raises(ValueError, match='divisible'):
            create_convunext(
                **_cfg(bottleneck_attention_blocks=1,
                       bottleneck_attention_heads=7))


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelSerialization (4)
# ---------------------------------------------------------------------

class TestSerialization:
    """``.keras`` round trips, all at NON-DEFAULT values and all with
    ``training=False`` EXPLICIT on both forward calls.
    """

    @staticmethod
    def _round_trip(model: keras.Model, x: np.ndarray):
        before = model(x, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'convunext.keras')
            model.save(path)
            reloaded = keras.models.load_model(path)
        return reloaded, before, reloaded(x, training=False)

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_round_trip_at_non_default_values(self, use_bias: bool) -> None:
        """Nine non-default arguments at once. Built at the DEFAULTS this test
        would be vacuous: a key dropped from a config is silently repaired by
        the constructor default and the trip still passes (measured, step 2).
        """
        model = create_convunext(
            use_bias=use_bias,
            input_shape=(16, 16, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=2,
            convnext_version='v1',
            stem_kernel_size=5,
            block_kernel_size=5,
            drop_path_rate=0.3,
            dropout_rate=0.1,
            downsample_pool_type='average',
            output_channels=2,
            model_name='convunext_round_trip',
        )
        x = np.random.default_rng(17).normal(size=(2, 16, 16, 3)).astype('float32')
        reloaded, before, after = self._round_trip(model, x)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(before),
            keras.ops.convert_to_numpy(after), rtol=1e-6, atol=1e-6,
            err_msg='outputs differ after a .keras round trip')

        assert reloaded.name == 'convunext_round_trip'
        assert reloaded.count_params() == model.count_params()
        assert [l.name for l in reloaded.layers] == [l.name for l in model.layers]
        # The non-default knobs survived, read off the RELOADED graph.
        assert reloaded.get_layer(name='final_output').filters == 2
        assert reloaded.get_layer(name='encoder_level_0_stem').kernel_size == 5
        assert type(_convnext_block_layers(reloaded)[0]) is ConvNextV1Block
        assert all(b.dropout_rate == 0.1
                   for b in _convnext_block_layers(reloaded))
        assert _drop_path_rate_by_name(
            reloaded, 'bottleneck_convnext_v1_block_1_drop_path'
        ) == pytest.approx(0.15)

    def test_round_trip_with_deep_supervision_preserves_count_and_order(
            self) -> None:
        model = create_convunext(
            **_cfg(depth=3, enable_deep_supervision=True, output_channels=2))
        x = np.random.default_rng(19).normal(size=(2, 16, 16, 3)).astype('float32')
        reloaded, before, after = self._round_trip(model, x)

        assert len(reloaded.outputs) == 3
        for b, a in zip(before, after):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(b), keras.ops.convert_to_numpy(a),
                rtol=1e-6, atol=1e-6)
        for i in range(1, 3):
            assert reloaded.outputs[i] is reloaded.get_layer(
                name=f'supervision_output_level_{i}').output

    def test_round_trip_with_the_optional_branches(self) -> None:
        """The Gabor stem, the pyramid + high-freq stack and the bottleneck
        attention all carry custom layers whose registration is what a round
        trip actually tests.
        """
        model = create_convunext(
            use_bias=False, use_gabor_stem=True, gabor_filters=4,
            gabor_kernel_size=5, use_laplacian_pyramid=True, high_freq_blocks=1,
            bottleneck_attention_blocks=1, bottleneck_attention_heads=2,
            **_cfg())
        x = np.random.default_rng(23).normal(size=(2, 16, 16, 3)).astype('float32')
        reloaded, before, after = self._round_trip(model, x)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(before),
            keras.ops.convert_to_numpy(after), rtol=1e-6, atol=1e-6)
        assert isinstance(
            reloaded.get_layer(name='bottleneck_attention_block_0'),
            SpatialLinearAttention)
        assert reloaded.get_layer(name='skip_highfreq_block_0_0') is not None
        assert reloaded.get_layer(name='gabor_stem').trainable is False


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelVariants (4) + TestCreateConvUNextVariant (2)
# ---------------------------------------------------------------------

class TestVariants:
    """``CONVUNEXT_CONFIGS`` and ``create_convunext_variant``.

    ``initial_filters`` is overridden to a tiny value for the four heavier
    variants so the sweep stays a CPU-second test; the variant's own declared
    ``initial_filters`` is asserted separately on ``'tiny'``, where building it
    unmodified is cheap. Overriding it everywhere would have made the sweep
    blind to that field entirely.
    """

    def test_all_five_variants_are_registered(self) -> None:
        assert set(CONVUNEXT_CONFIGS) == {
            'tiny', 'small', 'base', 'large', 'xlarge'}
        for name, cfg in CONVUNEXT_CONFIGS.items():
            assert {'depth', 'initial_filters', 'blocks_per_level',
                    'convnext_version', 'drop_path_rate',
                    'description'} <= set(cfg), name
            # Plan invariant I-3: the SHARED dict must not carry this key, or
            # the bias-on variants flip too. Only the bfconvunext wrapper sets it.
            assert 'block_normalization' not in cfg, name

    @pytest.mark.parametrize('variant', ['tiny', 'small', 'base', 'large', 'xlarge'])
    def test_variant_config_reaches_the_built_graph(self, variant: str) -> None:
        cfg = CONVUNEXT_CONFIGS[variant]
        depth = cfg['depth']
        model = create_convunext_variant(
            variant, input_shape=(32, 32, 3), initial_filters=4,
            filter_multiplier=1.0)

        assert model.name == f"convunext_{variant}_{cfg['convnext_version']}"
        # depth junctions: depth-1 encoder + the bottleneck one.
        assert model.get_layer(name=f'encoder_downsample_{depth - 2}') is not None
        with pytest.raises(ValueError):
            model.get_layer(name=f'encoder_downsample_{depth - 1}')
        assert model.get_layer(name='bottleneck_downsample') is not None

        expected_blocks = (2 * depth + 1) * cfg['blocks_per_level']
        assert len(_convnext_block_layers(model)) == expected_blocks

        block_cls = (ConvNextV2Block if cfg['convnext_version'] == 'v2'
                     else ConvNextV1Block)
        assert type(_convnext_block_layers(model)[0]) is block_cls

        # drop_path_rate reached the schedule (or is genuinely 0 for 'tiny').
        deepest = _drop_path_rate_by_name(
            model,
            f"encoder_level_{depth - 1}_convnext_{cfg['convnext_version']}"
            f"_block_{cfg['blocks_per_level'] - 1}_drop_path")
        if cfg['drop_path_rate'] == 0.0:
            assert deepest is None
        else:
            assert deepest == pytest.approx(cfg['drop_path_rate'] * (
                (depth - 1) * cfg['blocks_per_level']
                + cfg['blocks_per_level'] - 1) / (depth * cfg['blocks_per_level']))

    def test_tiny_variant_uses_its_declared_initial_filters(self) -> None:
        """CONTROL for the sweep above, which overrides this field."""
        model = create_convunext_variant('tiny', input_shape=(32, 32, 3))
        assert model.get_layer(name='encoder_level_0_stem').filters == \
            CONVUNEXT_CONFIGS['tiny']['initial_filters'] == 32

    def test_variant_kwargs_override_the_config(self) -> None:
        model = create_convunext_variant(
            'tiny', input_shape=(32, 32, 3), initial_filters=4,
            filter_multiplier=1.0, output_channels=1)
        assert model.get_layer(name='encoder_level_0_stem').filters == 4
        assert model.output_shape[-1] == 1

    def test_variant_deep_supervision_flag_and_name_suffix(self) -> None:
        model = create_convunext_variant(
            'tiny', input_shape=(32, 32, 3), enable_deep_supervision=True,
            initial_filters=4, filter_multiplier=1.0)
        assert model.name.endswith('_ds')
        assert len(model.outputs) == CONVUNEXT_CONFIGS['tiny']['depth']

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match='Unknown variant'):
            create_convunext_variant('gigantic', input_shape=(32, 32, 3))

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_variant_threads_use_bias(self, use_bias: bool) -> None:
        cfg = CONVUNEXT_CONFIGS['tiny']
        model = create_convunext_variant(
            'tiny', input_shape=(32, 32, 3), initial_filters=4,
            filter_multiplier=1.0, use_bias=use_bias)
        _assert_is_a_convunext_graph(
            model, depth=cfg['depth'], blocks_per_level=cfg['blocks_per_level'])
        assert (model.get_layer(name='final_output').bias is not None) is use_bias
        # Swept over the WHOLE graph, not just the head: a single site that
        # ignored `use_bias` would be invisible to the head assertion above.
        biased = [l.name for l in model._flatten_layers()
                  if getattr(l, 'use_bias', None) is True]
        if use_bias:
            assert biased, 'use_bias=True built no biased layer at all'
        else:
            assert biased == [], f'use_bias=False built biased layers: {biased}'


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelTrainingIntegration (3) + WeightLoading (2)
# ---------------------------------------------------------------------

class TestTrainingIntegration:
    """The model compiles, fits, and its weights actually MOVE."""

    @staticmethod
    def _data(n: int = 4, channels: int = 3):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(n, 16, 16, 3)).astype('float32')
        y = rng.normal(size=(n, 16, 16, channels)).astype('float32')
        return x, y

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_compile_and_fit_moves_the_weights(self, use_bias: bool) -> None:
        """Measured under plain SGD, deliberately.

        Adam's per-parameter normalization compresses the magnitude of a weight
        update to near-uniform, which makes a total-|dW| probe a poor instrument
        (recorded repo lesson). SGD makes the movement legible.
        """
        model = create_convunext(use_bias=use_bias, **_cfg())
        _assert_is_a_convunext_graph(model)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                      loss='mse')
        before = [keras.ops.convert_to_numpy(w) for w in model.trainable_weights]

        x, y = self._data()
        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        after = [keras.ops.convert_to_numpy(w) for w in model.trainable_weights]
        moved = sum(1 for b, a in zip(before, after)
                    if float(np.max(np.abs(a - b))) > 0.0)
        assert moved > 0.5 * len(before), (
            f'only {moved}/{len(before)} trainable weights moved after fit()')
        assert np.isfinite(history.history['loss'][0])

    def test_compile_and_fit_with_deep_supervision(self) -> None:
        model = create_convunext(
            **_cfg(depth=3, enable_deep_supervision=True, output_channels=1))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                      loss=['mse'] * 3)

        x, _ = self._data()
        rng = np.random.default_rng(31)
        targets = [rng.normal(size=(4, s, s, 1)).astype('float32')
                   for s in (16, 8, 4)]
        history = model.fit(x, targets, epochs=1, batch_size=2, verbose=0)
        assert np.isfinite(history.history['loss'][0])

    def test_trained_model_round_trips(self) -> None:
        model = create_convunext(**_cfg())
        _assert_is_a_convunext_graph(model)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                      loss='mse')
        x, y = self._data()
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        before = keras.ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'trained.keras')
            model.save(path)
            reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(x, training=False))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


class TestWeightTransfer:
    """Two builds at the same config are weight-compatible.

    This is what the deleted ``TestConvUNextModelWeightLoading`` was really
    asserting once its ``include_top`` half is removed (that half is now
    ``TestIncludeTop.test_include_top_false_weight_list_is_strictly_smaller``,
    which pins the OPPOSITE, measured contract).
    """

    @pytest.mark.parametrize('use_bias', [True, False])
    def test_set_weights_between_identical_configs(self, use_bias: bool) -> None:
        cfg = _cfg()
        src = create_convunext(use_bias=use_bias, **cfg)
        dst = create_convunext(use_bias=use_bias, **cfg)
        _assert_is_a_convunext_graph(src)
        x = np.random.default_rng(37).normal(size=(2, 16, 16, 3)).astype('float32')

        assert not np.allclose(
            keras.ops.convert_to_numpy(src(x, training=False)),
            keras.ops.convert_to_numpy(dst(x, training=False)), atol=1e-6), (
            'two freshly initialized models already agree; the transfer below '
            'would prove nothing')

        dst.set_weights(src.get_weights())
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(src(x, training=False)),
            keras.ops.convert_to_numpy(dst(x, training=False)),
            rtol=1e-6, atol=1e-6)

    def test_weights_do_not_transfer_across_a_shape_change(self) -> None:
        """CONTROL: the transfer above is not vacuous -- a genuinely different
        graph is rejected rather than silently accepted.
        """
        src = create_convunext(**_cfg())
        dst = create_convunext(**_cfg(initial_filters=16))
        with pytest.raises(ValueError):
            dst.set_weights(src.get_weights())


# ---------------------------------------------------------------------
# Re-derives: TestConvUNextModelEdgeCases (6)
# ---------------------------------------------------------------------

class TestEdgeCases:
    """Boundary configurations, both arms where the argument is arm-sensitive."""

    @pytest.mark.parametrize('channels', [1, 4])
    def test_non_rgb_input_channels(self, channels: int) -> None:
        model = create_convunext(**_cfg(input_shape=(16, 16, channels)))
        _assert_is_a_convunext_graph(model)
        x = np.zeros((1, 16, 16, channels), dtype='float32')
        assert model(x, training=False).shape == (1, 16, 16, channels)

    def test_minimum_depth(self) -> None:
        model = create_convunext(**_cfg(depth=2))
        assert model.get_layer(name='encoder_downsample_0') is not None
        assert model.get_layer(name='bottleneck_downsample') is not None
        x = np.zeros((1, 16, 16, 3), dtype='float32')
        assert model(x, training=False).shape == (1, 16, 16, 3)

    def test_kernel_regularizer_reaches_the_structural_convs(self) -> None:
        """A regularizer that never reached a layer would be a silently inert
        argument -- this repo's recorded defect class.
        """
        reg = keras.regularizers.L2(1e-4)
        model = create_convunext(**_cfg(kernel_regularizer=reg))
        assert model.get_layer(name='final_output').kernel_regularizer is not None
        assert model.get_layer(
            name='encoder_level_0_stem').kernel_regularizer is not None

        # NOTE (measured, not assumed): `model.losses` is NEVER empty here --
        # every ConvNeXt block's LayerScale gamma carries its own hardcoded
        # GAMMA_L2_REGULARIZATION (1e-5), so 5 losses exist at this config even
        # with kernel_regularizer=None. A bare `assert model.losses` would
        # therefore pass with the argument completely ignored. The instrument is
        # the DELTA against the no-regularizer control below.
        baseline = create_convunext(**_cfg())
        assert len(model.losses) > len(baseline.losses), (
            f'kernel_regularizer added no loss term: {len(model.losses)} vs '
            f'{len(baseline.losses)} without it')

    def test_no_regularizer_leaves_only_the_blocks_own_gamma_penalty(self) -> None:
        """CONTROL for the test above, and a pin on what the baseline IS."""
        model = create_convunext(**_cfg())
        assert model.get_layer(name='final_output').kernel_regularizer is None
        assert model.get_layer(
            name='encoder_level_0_stem').kernel_regularizer is None
        # One per ConvNeXt block (their gamma L2 is not configurable here).
        assert len(model.losses) == len(_convnext_block_layers(model))

    @pytest.mark.parametrize('activation', ['sigmoid', 'tanh', 'relu'])
    def test_final_activation_reaches_the_final_conv_on_the_bias_on_arm(
            self, activation: str) -> None:
        """Non-homogeneous final activations are legal on the bias-ON arm (the
        bias-off arm rejects them -- ``TestBiasFreeGuardrails``).
        """
        model = create_convunext(**_cfg(final_activation=activation))
        final = model.get_layer(name='final_output')
        assert keras.activations.serialize(final.activation) == activation

        x = np.random.default_rng(41).normal(
            size=(2, 16, 16, 3)).astype('float32')
        y = keras.ops.convert_to_numpy(model(x, training=False))
        assert np.isfinite(y).all()
        if activation == 'sigmoid':
            assert (y > 0.0).all() and (y < 1.0).all()
        if activation == 'relu':
            assert (y >= 0.0).all()

    def test_stem_normalization_reaches_the_builder(self) -> None:
        """``stem_normalization`` is threaded from ``create_convunext``, not
        only settable on the layer directly (which ``TestMergedConvUNextStemKnobs``
        already covers).
        """
        model = create_convunext(**_cfg(stem_normalization='layer_norm'))
        stem = model.get_layer(name='encoder_level_0_stem')
        assert type(stem.norm) is keras.layers.LayerNormalization

        default = create_convunext(**_cfg())
        assert type(default.get_layer(name='encoder_level_0_stem').norm) is \
            GlobalResponseNormalization

    def test_gabor_stem_replaces_the_convunext_stem(self) -> None:
        """The two stems are mutually exclusive branches -- with the Gabor bank
        on, no ``ConvUNextStem`` is built at all, and the bank is FROZEN.
        """
        model = create_convunext(
            **_cfg(use_gabor_stem=True, gabor_filters=4, gabor_kernel_size=5))
        with pytest.raises(ValueError):
            model.get_layer(name='encoder_level_0_stem')
        gabor = model.get_layer(name='gabor_stem')
        assert gabor.trainable is False
        assert model.get_layer(name='gabor_stem_projection') is not None
