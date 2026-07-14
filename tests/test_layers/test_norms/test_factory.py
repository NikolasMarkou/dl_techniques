"""
Test suite for the normalization factory (`norms/factory.py`).

Covers:
- construct-all: every NormalizationType key builds a usable keras Layer.
- F1 (plan_2026-06-15_2485b951): validate_normalization_config accepts the
  band/GRN initializer+regularizer params (previously false-rejected).
- create_normalization_from_config round-trip.
- known validation failures still raise.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.layers.norms.factory import (
    create_normalization_layer,
    create_normalization_from_config,
    validate_normalization_config,
    get_normalization_info,
)

ALL_TYPES = [
    "layer_norm", "batch_norm", "rms_norm", "zero_centered_rms_norm",
    "zero_centered_band_rms_norm", "zero_centered_adaptive_band_rms_norm",
    "band_rms", "adaptive_band_rms", "band_logit_norm", "global_response_norm",
    "logit_norm", "max_logit_norm", "decoupled_max_logit", "dml_plus_focal",
    "dml_plus_center", "dynamic_tanh", "energy_layer_norm",
]


class TestFactoryConstruction:
    @pytest.mark.parametrize("norm_type", ALL_TYPES)
    def test_construct_all(self, norm_type):
        layer = create_normalization_layer(norm_type, name=f"n_{norm_type}")
        assert isinstance(layer, keras.layers.Layer)

    def test_info_covers_all_types(self):
        info = get_normalization_info()
        # every factory-dispatchable type except the two Keras built-ins+aliases
        # is described; all 16 keys should be present.
        for t in ALL_TYPES:
            assert t in info, f"{t} missing from get_normalization_info()"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_normalization_layer("does_not_exist")


class TestF1ValidateWhitelist:
    """Pins F1: regularizer/initializer params must be accepted by validate."""

    def test_band_rms_accepts_regularizer(self):
        assert validate_normalization_config(
            "band_rms",
            max_band_width=0.1,
            band_initializer="zeros",
            band_regularizer=keras.regularizers.L2(1e-5),
        )

    def test_adaptive_band_rms_accepts_regularizer(self):
        assert validate_normalization_config(
            "adaptive_band_rms",
            band_regularizer=keras.regularizers.L1(1e-4),
        )

    def test_grn_accepts_regularizers(self):
        assert validate_normalization_config(
            "global_response_norm",
            gamma_regularizer=keras.regularizers.L2(1e-5),
            beta_regularizer=None,
            activity_regularizer=None,
        )

    def test_invalid_param_still_rejected(self):
        with pytest.raises(ValueError):
            validate_normalization_config("dynamic_tanh", epsilon=1e-6)


class TestEnergyLayerNormGammaConstraint:
    """G-03: the validator and the BUILDER must agree about the layer's own signature.

    `plan_2026-07-13_57c9833e`/D-010 added the `gamma_constraint` ctor kwarg to
    `EnergyLayerNorm` (it pins `gamma > 0`, on which the energy-descent guarantee rests)
    but never added it to this registry entry's `parameters` list. Result:
    `create_normalization_layer` ACCEPTED the kwarg while `validate_normalization_config`
    REJECTED it -- so any caller who validated before building was locked out of the one
    constraint the layer's own docstring tells them to use. See decisions.md D-004.
    """

    def test_validate_accepts_gamma_constraint(self):
        """RED at HEAD: ValueError: Invalid parameters for energy_layer_norm."""
        assert validate_normalization_config(
            "energy_layer_norm",
            gamma_constraint=ValueRangeConstraint(min_value=1e-3),
        )

    def test_validated_config_round_trips_onto_the_built_gamma(self):
        """Validation passing is NOT the property that matters -- the constraint ARRIVING is.

        A registry entry could name the param and the builder still drop it on the floor.
        Assert on the BUILT variable, not on the validator's return value.
        """
        constraint = ValueRangeConstraint(min_value=1e-3)
        assert validate_normalization_config(
            "energy_layer_norm", gamma_constraint=constraint
        )

        layer = create_normalization_layer(
            "energy_layer_norm", gamma_constraint=constraint, name="eln_c"
        )
        layer.build((2, 8, 16))

        assert layer.gamma.constraint is constraint, (
            "validate() passed and create() accepted the kwarg, but the built `gamma` "
            "variable carries no constraint -- the parameter was silently dropped."
        )

        # And it must BITE: a negative gamma is clipped back above the floor. `gamma > 0`
        # is what makes the Lagrangian's Hessian PSD (57c9833e/D-010).
        layer.gamma.assign(ops.convert_to_tensor(-5.0, dtype=layer.gamma.dtype))
        clipped = float(ops.convert_to_numpy(layer.gamma.constraint(layer.gamma)))
        assert clipped >= 1e-3, f"the constraint did not clip a negative gamma ({clipped})"


class TestFromConfig:
    def test_from_config_roundtrip(self):
        config = {
            "type": "zero_centered_band_rms_norm",
            "max_band_width": 0.08,
            "epsilon": 1e-6,
            "axis": -1,
        }
        layer = create_normalization_from_config(config)
        x = ops.convert_to_tensor(np.random.randn(4, 16).astype("float32"))
        assert tuple(layer(x).shape) == (4, 16)

    def test_missing_type_raises(self):
        with pytest.raises(KeyError):
            create_normalization_from_config({"max_band_width": 0.1})


# ---------------------------------------------------------------------
# The validator/builder agreement guard.
#
# Root cause this pins: `validate_normalization_config` used to whitelist kwargs from a
# HAND-MAINTAINED list (`get_normalization_info()[t]['parameters']`). Every time someone
# added a constructor argument to a layer and did not also edit that list, the validator
# began rejecting a parameter the builder happily accepts. That happened at least twice
# (F1/plan_2026-06-15_2485b951, then `gamma_constraint` on `energy_layer_norm`), and a
# behavioural audit then found **19** live disagreements across `layer_norm` (7),
# `batch_norm` (9), `global_response_norm` (2 incl. `epsilon`) and `dynamic_tanh`
# (`epsilon`). The whitelist is now DERIVED from the real ctor signature, so the drift is
# unrepresentable — these tests exist to keep it that way.
#
# NOTE ON THE PREDECESSOR GUARD: `test_registry_keys_are_real_ctor_args` in
# tests/test_layers/test_attention/test_attention_factory.py is inert three independent
# ways, any one fatal: (a) it lives in the ATTENTION package and never covered norms at
# all; (b) it only checks `declared - accepted` (PHANTOM drift), so it structurally cannot
# see the MISSING drift that is the actual bug; (c) it `pytest.skip`s on `**kwargs`, and
# 18/18 norm classes take `**kwargs`. Nothing below skips, and both drift directions are
# checked. A guard that skips its subjects is worse than no guard: it looks like coverage.
# ---------------------------------------------------------------------

from dl_techniques.layers.norms.factory import (  # noqa: E402
    _TYPE_TO_CLASS,
    _FACTORY_OWNED_PARAMS,
    _FACTORY_DROPPED_PARAMS,
    _accepted_params,
)

# Params the factory constructs with but then IGNORES (overwrites or discards). The
# validator rejects these ON PURPOSE -- see the comments on those two dicts. "Builds
# without raising" is NOT the same as "has an effect", and conflating the two is what
# made the first draft of this guard demand that `dynamic_tanh` accept a silently-dropped
# `epsilon`. An existing test caught it.
def _ignored_params(norm_type):
    return (_FACTORY_OWNED_PARAMS.get(norm_type, frozenset())
            | _FACTORY_DROPPED_PARAMS.get(norm_type, frozenset()))

# A representative value per parameter name, used to actually BUILD the layer. Only needs
# to be type-plausible; construction is what is under test, not numerics.
_PROBE_VALUES = {
    'axis': -1, 'epsilon': 1e-5, 'eps': 1e-5, 'center': True, 'scale': True,
    'momentum': 0.9, 'synchronized': False, 'rms_scaling': False,
    'use_scale': True, 'use_beta': True, 'use_gamma': True,
    'scale_initializer': 'ones', 'scale_regularizer': None,
    'beta_initializer': 'zeros', 'gamma_initializer': 'ones',
    'beta_regularizer': None, 'gamma_regularizer': None,
    'beta_constraint': None, 'gamma_constraint': None,
    'moving_mean_initializer': 'zeros', 'moving_variance_initializer': 'ones',
    'max_band_width': 0.2, 'band_initializer': 'zeros', 'band_regularizer': None,
    'band_constraint': None, 'temperature': 1.0, 'constant': 1.0,
    'alpha_init_value': 0.5, 'delta_initializer': 'zeros',
    'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
    'kernel_regularizer': None, 'bias_regularizer': None,
    'kernel_constraint': None, 'bias_constraint': None,
    'activity_regularizer': None, 'trainable': True, 'dtype': 'float32',
    'autocast': True, 'name': 'probe',
}


def _params_under_test(norm_type):
    """Every accepted param for a type that we have a probe value for."""
    return sorted(p for p in _accepted_params(norm_type) if p in _PROBE_VALUES)


class TestValidatorAgreesWithBuilder:
    """THE invariant: anything the BUILDER accepts, the VALIDATOR must accept.

    This is the bug, stated directly. It is checked per (type, parameter) rather than in
    aggregate so a failure names the exact parameter that drifted.
    """

    @pytest.mark.parametrize("norm_type", sorted(_TYPE_TO_CLASS))
    def test_every_accepted_param_is_probeable(self, norm_type):
        """The probe table must cover the type, or the tests below are vacuous.

        Without this, adding a ctor param with no probe value would silently shrink the
        test's subject set to nothing and the suite would stay green. That is exactly the
        'guard that skips its subjects' failure mode the predecessor guard shipped.
        """
        accepted = _accepted_params(norm_type)
        uncovered = accepted - set(_PROBE_VALUES)
        assert not uncovered, (
            f"'{norm_type}' accepts {sorted(uncovered)} but the probe table has no value "
            f"for them, so they are UNTESTED. Add them to _PROBE_VALUES."
        )
        assert _params_under_test(norm_type), f"no params under test for '{norm_type}'"

    @pytest.mark.parametrize("norm_type", sorted(_TYPE_TO_CLASS))
    def test_validator_accepts_everything_the_builder_accepts(self, norm_type):
        """MISSING drift — the actual bug. Builds, then validates, each param."""
        disagreements = []
        ignored = _ignored_params(norm_type)
        for param in _params_under_test(norm_type):
            if param in ignored:
                continue  # deliberately rejected -- see test_factory_ignored_params_are_rejected
            kwargs = {param: _PROBE_VALUES[param]}
            try:
                create_normalization_layer(norm_type, **kwargs)
            except Exception:
                continue  # builder rejects it too -> no disagreement, not our concern
            try:
                validate_normalization_config(norm_type, **kwargs)
            except ValueError:
                disagreements.append(param)
        assert not disagreements, (
            f"validate_normalization_config('{norm_type}', ...) REJECTS "
            f"{disagreements}, which create_normalization_layer() accepts. The validator "
            f"and the builder disagree about this layer's own signature."
        )

    @pytest.mark.parametrize("norm_type", sorted(_TYPE_TO_CLASS))
    def test_documented_parameters_are_really_accepted(self, norm_type):
        """PHANTOM drift — a documented param the layer does not take."""
        documented = set(get_normalization_info()[norm_type]['parameters'])
        phantom = documented - _accepted_params(norm_type)
        assert not phantom, (
            f"get_normalization_info()['{norm_type}'] documents {sorted(phantom)}, which "
            f"{_TYPE_TO_CLASS[norm_type].__name__} does not accept."
        )

    @pytest.mark.parametrize("norm_type", sorted(_TYPE_TO_CLASS))
    def test_type_to_class_matches_what_the_builder_returns(self, norm_type):
        """_TYPE_TO_CLASS must not drift from create_normalization_layer's if/elif chain.

        The derived whitelist is only correct if the map names the class the builder
        really instantiates. Without this, the map could rot and the validator would
        derive its whitelist from the wrong class.
        """
        layer = create_normalization_layer(norm_type)
        assert type(layer) is _TYPE_TO_CLASS[norm_type], (
            f"create_normalization_layer('{norm_type}') returns "
            f"{type(layer).__name__}, but _TYPE_TO_CLASS says "
            f"{_TYPE_TO_CLASS[norm_type].__name__}."
        )

    def test_typos_are_still_rejected(self):
        """Deriving the whitelist must not make the validator permissive."""
        with pytest.raises(ValueError, match="Invalid parameters"):
            validate_normalization_config("rms_norm", epsilonn=1e-5)

    @pytest.mark.parametrize(
        "norm_type", sorted(set(_FACTORY_OWNED_PARAMS) | set(_FACTORY_DROPPED_PARAMS))
    )
    def test_factory_ignored_params_are_rejected(self, norm_type):
        """Params the factory OVERWRITES or DISCARDS must still be rejected.

        `model_type` is hard-assigned by the factory for the DML+ variants; `epsilon` is
        popped for `dynamic_tanh`. Both CONSTRUCT fine, so a naive 'the validator must
        accept whatever builds' rule would wrongly accept them -- and then a caller who
        sets the value would never learn it does nothing. Rejecting is CORRECT, not drift.
        """
        for param in _ignored_params(norm_type):
            probe = 'focal' if param == 'model_type' else _PROBE_VALUES[param]
            with pytest.raises(ValueError, match="Invalid parameters"):
                validate_normalization_config(norm_type, **{param: probe})
