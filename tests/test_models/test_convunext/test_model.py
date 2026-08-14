"""
Tests for the merged ConvUNext: ``ConvUNextStem`` and ``create_convunext``.

Covers:
- ``ConvUNextStem`` lifecycle, its merged ``use_bias`` / ``stem_normalization`` knobs
- ``create_convunext``'s ``use_bias`` threading (bias-on vs bias-off)
- the bottleneck drop-path RAMP
- the builder is not a pass-through of its input

NOTE (plan step 4): the ConvUNextModel-era classes that used to live below
(instantiation, forward pass, include_top, deep supervision, compute_output_shape,
serialization, variants, inference-model, training integration, edge cases, weight
loading, residual/drop-path wiring) were removed together with the subclassed
``ConvUNextModel`` they instantiated -- with it deleted they could not even import.
Plan step 9 rewrites this file from scratch, mirroring
``test_bfconvunext_denoiser.py`` with symmetric bias-on / bias-off arms, and ports
forward the invariants ``TestConvUNextResidualAndDropPathWiring`` pinned as
functional-graph equivalents.
"""

import os
import logging
import tempfile
from typing import Dict, Any

import numpy as np
import pytest
import keras

from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.norms.global_response_norm import (
    GlobalResponseNormalization,
)
from dl_techniques.models.convunext.model import (
    ConvUNextStem,
    create_convunext,
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

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_different_kernel_sizes(
        self,
        kernel_size: int,
        sample_input_small: np.ndarray
    ) -> None:
        """Test stem works with various kernel sizes."""
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

    def test_output_channels_defaults_to_the_input_channel_count(self):
        """CONTROL: the denoiser/autoencoder contract every caller relies on."""
        model = create_convunext(**_knob_config())
        assert model.output_shape[-1] == 3
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
