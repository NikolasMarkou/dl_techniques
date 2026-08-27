"""Serialization-surface and validation tests for the MoE configuration dataclasses.

Scope note (deliberate, to avoid duplicating coverage that already exists):

* The *validation matrix* for the integer fields (``bool`` rejection, non-int
  rejection, range messages) lives in
  ``test_layer.py::TestIntegerFieldsRejectBool`` -- except the numpy half of it,
  which lives here in ``TestValidatePositiveIntAcceptanceTable`` because numpy
  scalars are the case that motivated widening the predicate. The ``num_experts`` /
  ``top_k <= num_experts`` / ``jitter_noise`` cross-field rules live in
  ``test_layer.py::TestMoEConfigValidation``. This module does **not** repeat
  them; it covers the fields and the surface those two classes leave open.
* The gap this module exists to close is the ``to_dict`` / ``from_dict``
  round-trip. The pre-existing round-trip test
  (``test_layer.py::TestMoEConfigurations::test_config_serialization``) leaves
  10 of the 19 primitive fields at their dataclass defaults, so a field that
  silently reverts to its default on the way through cannot be detected by it.
  Every round-trip test here drives **every** field off its default first and
  asserts that it did.
"""

import dataclasses
import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.moe.config import (
    ExpertConfig,
    GatingConfig,
    MoEConfig,
)

# ---------------------------------------------------------------------
# Non-default fixtures: every field of every dataclass driven off its default
# ---------------------------------------------------------------------

NON_DEFAULT_EXPERT = dict(
    ffn_config={'type': 'geglu', 'hidden_dim': 24, 'output_dim': 12},
    use_bias=False,
    kernel_initializer=keras.initializers.HeNormal(seed=3),
    bias_initializer=keras.initializers.Constant(0.25),
    kernel_regularizer=keras.regularizers.L2(1e-4),
    bias_regularizer=keras.regularizers.L1(2e-4),
    norm_type='rms_norm',
    norm_config={'epsilon': 1e-5},
    pre_norm=False,
    post_norm=True,
)

NON_DEFAULT_GATING = dict(
    gating_type='cosine',
    top_k=3,
    add_noise=False,
    noise_std=0.25,
    temperature=2.5,
    use_bias=True,
    embedding_dim=64,
    learnable_temperature=False,
    num_slots=7,
    aux_loss_weight=0.02,
    z_loss_weight=5e-3,
    norm_type='layer_norm',
    norm_config={'epsilon': 1e-3},
)

NON_DEFAULT_MOE = dict(
    num_experts=6,
    jitter_noise=0.05,
    drop_tokens=False,
    use_residual_connection=False,
)


def _default_of(cls, name):
    """Return the declared default of a dataclass field (calling its factory)."""
    fld = {f.name: f for f in dataclasses.fields(cls)}[name]
    if fld.default is not dataclasses.MISSING:
        return fld.default
    return fld.default_factory()


def _comparable(value):
    """Normalize a field value to something with a meaningful ``==``.

    Keras ``Initializer``/``Regularizer`` objects do not implement value
    equality -- two freshly constructed ``HeNormal()`` instances compare
    unequal -- so they are compared through ``get_config()`` instead.
    """
    if isinstance(value, (keras.initializers.Initializer,
                          keras.regularizers.Regularizer)):
        return (type(value).__name__, value.get_config())
    return value


def _make_config():
    return MoEConfig(
        expert_config=ExpertConfig(**NON_DEFAULT_EXPERT),
        gating_config=GatingConfig(**NON_DEFAULT_GATING),
        **NON_DEFAULT_MOE,
    )


# ---------------------------------------------------------------------


class TestNonDefaultFixturesAreActuallyNonDefault:
    """Guard the guard: if a "non-default" value equals the default, the
    round-trip tests below silently stop being able to fail."""

    @pytest.mark.parametrize("name", sorted(NON_DEFAULT_EXPERT))
    def test_expert_field_differs_from_default(self, name):
        assert _comparable(NON_DEFAULT_EXPERT[name]) != _comparable(
            _default_of(ExpertConfig, name)
        )

    @pytest.mark.parametrize("name", sorted(NON_DEFAULT_GATING))
    def test_gating_field_differs_from_default(self, name):
        assert NON_DEFAULT_GATING[name] != _default_of(GatingConfig, name)

    @pytest.mark.parametrize("name", sorted(NON_DEFAULT_MOE))
    def test_moe_field_differs_from_default(self, name):
        assert NON_DEFAULT_MOE[name] != _default_of(MoEConfig, name)

    def test_every_declared_field_is_covered(self):
        """A field added to a dataclass later must be added to the fixtures too."""
        assert {f.name for f in dataclasses.fields(ExpertConfig)} == set(
            NON_DEFAULT_EXPERT
        )
        assert {f.name for f in dataclasses.fields(GatingConfig)} == set(
            NON_DEFAULT_GATING
        )
        assert {f.name for f in dataclasses.fields(MoEConfig)} == (
            set(NON_DEFAULT_MOE) | {'expert_config', 'gating_config'}
        )


class TestRoundTripPreservesEveryField:
    """``to_dict`` -> ``from_dict`` must preserve every field at a non-default value."""

    @pytest.mark.parametrize("name", sorted(NON_DEFAULT_MOE))
    def test_moe_primitive_field_survives(self, name):
        restored = MoEConfig.from_dict(_make_config().to_dict())
        assert getattr(restored, name) == NON_DEFAULT_MOE[name]

    @pytest.mark.parametrize("name", sorted(NON_DEFAULT_GATING))
    def test_gating_field_survives(self, name):
        restored = MoEConfig.from_dict(_make_config().to_dict())
        assert getattr(restored.gating_config, name) == NON_DEFAULT_GATING[name]

    @pytest.mark.parametrize("name", sorted(NON_DEFAULT_EXPERT))
    def test_expert_field_survives(self, name):
        restored = MoEConfig.from_dict(_make_config().to_dict())
        assert _comparable(getattr(restored.expert_config, name)) == _comparable(
            NON_DEFAULT_EXPERT[name]
        )

    def test_keras_objects_come_back_as_objects_not_dicts(self):
        """``to_dict`` serializes initializers/regularizers; ``from_dict`` must
        deserialize them, not leave the raw ``dict`` on the dataclass."""
        restored = MoEConfig.from_dict(_make_config().to_dict())
        ec = restored.expert_config
        assert isinstance(ec.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(ec.bias_initializer, keras.initializers.Constant)
        assert isinstance(ec.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(ec.bias_regularizer, keras.regularizers.L1)

    def test_round_trip_is_idempotent(self):
        """A second pass through the pair must not drift."""
        once = _make_config().to_dict()
        twice = MoEConfig.from_dict(once).to_dict()
        assert twice['gating_config'] == once['gating_config']
        for key in NON_DEFAULT_MOE:
            assert twice[key] == once[key]

    def test_from_dict_does_not_mutate_its_input(self):
        payload = _make_config().to_dict()
        before = dict(payload)
        MoEConfig.from_dict(payload)
        assert payload.keys() == before.keys()
        assert payload['gating_config'] == before['gating_config']


class TestToDictSurface:
    """``to_dict`` emits exactly the live field set -- no more, no less."""

    def test_top_level_keys_match_the_dataclass(self):
        assert set(_make_config().to_dict()) == {
            f.name for f in dataclasses.fields(MoEConfig)
        }

    def test_gating_keys_match_the_dataclass(self):
        assert set(_make_config().to_dict()['gating_config']) == {
            f.name for f in dataclasses.fields(GatingConfig)
        }

    def test_expert_keys_match_the_dataclass(self):
        assert set(_make_config().to_dict()['expert_config']) == {
            f.name for f in dataclasses.fields(ExpertConfig)
        }

    def test_nested_dicts_are_copies_not_aliases(self):
        """Mutating the serialized payload must not reach back into the config."""
        cfg = _make_config()
        payload = cfg.to_dict()
        payload['gating_config']['top_k'] = 99
        assert cfg.gating_config.top_k == 3


class TestRemovedFieldsAreNotTolerated:
    """C1 / D-014: ``routing_dtype`` and ``capacity_factor`` were removed with
    **no** back-compat shim. A payload naming either must fail loud.

    ``test_layer.py::test_dead_fields_removed_from_config_surface`` covers the
    *constructor*; this covers the deserialization path, which is where a
    stored payload actually arrives.
    """

    def test_from_dict_rejects_routing_dtype(self):
        payload = _make_config().to_dict()
        payload['routing_dtype'] = 'float32'
        with pytest.raises(TypeError, match='routing_dtype'):
            MoEConfig.from_dict(payload)

    def test_from_dict_rejects_capacity_factor(self):
        payload = _make_config().to_dict()
        payload['gating_config']['capacity_factor'] = 1.25
        with pytest.raises(TypeError, match='capacity_factor'):
            MoEConfig.from_dict(payload)

    def test_unknown_top_level_key_is_rejected_too(self):
        """The rejection is the generic dataclass contract, not a special case."""
        payload = _make_config().to_dict()
        payload['not_a_field'] = 1
        with pytest.raises(TypeError):
            MoEConfig.from_dict(payload)

    @pytest.mark.parametrize(
        "legacy_key", ['train_capacity_factor', 'eval_capacity_factor']
    )
    def test_the_two_declared_legacy_keys_are_still_dropped(self, legacy_key):
        """These two, and only these two, are silently dropped -- an explicit,
        pre-existing precedent in ``from_dict``, unchanged by this plan."""
        payload = _make_config().to_dict()
        payload[legacy_key] = 2.0
        restored = MoEConfig.from_dict(payload)
        assert restored.num_experts == NON_DEFAULT_MOE['num_experts']
        assert not hasattr(restored, legacy_key)


class TestExpertConfigValidation:
    """``ExpertConfig`` owns only one rule; both of its branches are covered."""

    def test_empty_ffn_config_gets_the_default_mlp(self):
        cfg = ExpertConfig()
        assert cfg.ffn_config == {
            'type': 'mlp', 'hidden_dim': 2048, 'output_dim': 512
        }

    def test_non_empty_ffn_config_without_type_is_rejected(self):
        with pytest.raises(ValueError, match="must contain 'type'"):
            ExpertConfig(ffn_config={'hidden_dim': 128})

    def test_a_supplied_ffn_config_is_never_overwritten(self):
        cfg = ExpertConfig(ffn_config={'type': 'swiglu', 'output_dim': 8})
        assert cfg.ffn_config == {'type': 'swiglu', 'output_dim': 8}

    def test_ffn_config_reaching_the_default_survives_the_round_trip(self):
        """The default is injected in ``__post_init__``, i.e. on both sides of
        the round trip -- assert it is the *same* dict, not re-defaulted from
        an empty one that happened to match."""
        cfg = MoEConfig(expert_config=ExpertConfig(
            ffn_config={'type': 'mlp', 'hidden_dim': 33, 'output_dim': 11}
        ))
        restored = MoEConfig.from_dict(cfg.to_dict())
        assert restored.expert_config.ffn_config['hidden_dim'] == 33



class TestValidatePositiveIntAcceptanceTable:
    """The full accept/reject table for ``_validate_positive_int``, per field.

    Driven through the PUBLIC constructors, not the private helper, so the table
    pins what a caller actually experiences -- including the coercion, which the
    helper's return value alone would not show.

    The numpy rows are the point of the class: ``np.int64(4)`` is NOT an instance
    of ``int`` (measured), so before this widening every integral numpy scalar --
    a legitimate arrival from ``some_array.shape[-1]`` -- was rejected, while
    ``np.bool_(True)`` must stay rejected.
    """

    # (field name, constructor) for each of the four validated int fields.
    FIELDS = [
        ('top_k', lambda v: GatingConfig(gating_type='linear', top_k=v)),
        ('num_slots', lambda v: GatingConfig(num_slots=v)),
        ('embedding_dim', lambda v: GatingConfig(embedding_dim=v)),
        ('num_experts', lambda v: MoEConfig(num_experts=v)),
    ]
    FIELD_IDS = [f for f, _ in FIELDS]

    ACCEPTED = [
        pytest.param(4, id='python_int'),
        pytest.param(np.int64(4), id='np_int64'),
        pytest.param(np.int32(4), id='np_int32'),
        pytest.param(np.uint8(4), id='np_uint8'),
    ]
    REJECTED = [
        pytest.param(True, id='True'),
        pytest.param(False, id='False'),
        pytest.param(np.bool_(True), id='np_bool_'),
        pytest.param(4.0, id='float_4p0'),
        pytest.param(4.5, id='float_4p5'),
        pytest.param(-1, id='minus_1'),
        pytest.param(0, id='zero'),
        pytest.param("4", id='str_4'),
        pytest.param(None, id='None'),
    ]

    @pytest.mark.parametrize("field,build", FIELDS, ids=FIELD_IDS)
    @pytest.mark.parametrize("value", ACCEPTED)
    def test_accepted_values_construct_and_come_back_as_python_int(
            self, field, build, value
    ):
        cfg = build(value)
        got = getattr(cfg, field)
        assert got == 4
        # The coercion, not merely the acceptance: a stored ``np.int64`` would
        # only fail later, inside ``json.dumps`` at ``model.save()`` time.
        assert type(got) is int, f"{field} kept {type(got).__name__}, not int"

    @pytest.mark.parametrize("field,build", FIELDS, ids=FIELD_IDS)
    @pytest.mark.parametrize("value", REJECTED)
    def test_rejected_values_raise_value_error(self, field, build, value):
        with pytest.raises(ValueError):
            build(value)

    @pytest.mark.parametrize("value", [True, False, np.bool_(True)])
    def test_the_bool_rejection_message_names_bool_not_the_type_name(self, value):
        """``np.bool_`` must take the SAME branch as ``bool``.

        It is rejected today either way -- ``numbers.Integral`` does not admit
        ``np.bool_`` under numpy 2.0.2 -- so only the message distinguishes an
        explicit branch from an accidental one. If numpy ever registers
        ``np.bool_`` as ``Integral``, this assertion is what fails.
        """
        with pytest.raises(ValueError, match=r"got bool \("):
            GatingConfig(gating_type='linear', top_k=value)

    @pytest.mark.parametrize("field,build", FIELDS, ids=FIELD_IDS)
    def test_the_int32_ceiling_is_the_upper_bound(self, field, build):
        assert getattr(build(2 ** 31 - 1), field) == 2 ** 31 - 1
        with pytest.raises(ValueError, match="int32 tensor-dimension ceiling"):
            build(2 ** 31)

    def test_a_numpy_supplied_config_is_json_serializable(self):
        """The reason coercion is mandatory, pinned as an executable claim."""
        cfg = MoEConfig(
            num_experts=np.int64(4),
            gating_config=GatingConfig(gating_type='linear', top_k=np.int64(2)),
        )
        # Raises TypeError("Object of type int64 is not JSON serializable")
        # if either field kept its numpy type.
        json.dumps(cfg.to_dict())


class TestFieldsThatAreDeliberatelyUnvalidated:
    """Documentation pins, LABELLED as such: these are green on arrival.

    They record the *current* accepted surface so that a future change to it is
    a deliberate edit rather than a silent one. They are not RED-provable
    guards, because there is no guard -- that is exactly what they document.
    """

    @pytest.mark.parametrize("weight", [0.0, -1.0, 1e6])
    def test_aux_and_z_loss_weights_take_any_float(self, weight):
        cfg = GatingConfig(aux_loss_weight=weight, z_loss_weight=weight)
        assert cfg.aux_loss_weight == weight and cfg.z_loss_weight == weight

    def test_noise_std_zero_is_accepted(self):
        assert GatingConfig(noise_std=0.0).noise_std == 0.0

    def test_expert_config_wrapper_fields_take_anything(self):
        """The five wrapper-layer fields are documented-inert (see the class
        docstring); nothing validates them."""
        cfg = ExpertConfig(use_bias='not-a-bool', kernel_initializer=object())
        assert cfg.use_bias == 'not-a-bool'

# ---------------------------------------------------------------------
