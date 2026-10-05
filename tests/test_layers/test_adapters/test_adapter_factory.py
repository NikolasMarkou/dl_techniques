"""Tests for the adapter factory's strict-construction contract."""

import pytest

from dl_techniques.layers.adapters.factory import (
    ADAPTER_REGISTRY,
    STRICT_DROPPED_KEY_MARKER,
    assemble_adapter_config,
    create_adapter_from_config,
    create_adapter_layer,
    get_adapter_info,
    get_adapter_requirements,
    list_adapter_types,
    validate_adapter_config,
)
from dl_techniques.layers.adapters.lora import LoRAAdapter

VALID = dict(output_dim=32, rank=4, alpha=8.0, num_adapters=3)


class TestStrictConstruction:
    """A keyword the chosen type does not declare must RAISE, not be dropped."""

    def test_an_undeclared_keyword_raises(self):
        with pytest.raises(ValueError) as exc:
            create_adapter_layer('lora', dropout_rate=0.1, **VALID)
        assert STRICT_DROPPED_KEY_MARKER in str(exc.value)

    def test_the_message_names_the_offending_key_and_the_accepted_set(self):
        with pytest.raises(ValueError) as exc:
            create_adapter_layer('lora', learning_rate=1e-4, **VALID)
        message = str(exc.value)
        assert 'learning_rate' in message
        assert 'rank' in message and 'num_adapters' in message

    def test_the_validator_raises_on_the_same_key_the_builder_does(self):
        """Both entry points must reject it, so neither path silently accepts.

        Removing either copy of the undeclared-key check changes WHICH failure
        fires first (guide §9.2); this is the guard on that duplication
        surviving.
        """
        with pytest.raises(ValueError) as from_validator:
            validate_adapter_config('lora', dropout_rate=0.1, **VALID)
        with pytest.raises(ValueError) as from_builder:
            create_adapter_layer('lora', dropout_rate=0.1, **VALID)
        assert STRICT_DROPPED_KEY_MARKER in str(from_validator.value)
        assert STRICT_DROPPED_KEY_MARKER in str(from_builder.value)

    def test_a_declared_optional_is_accepted(self):
        """The strictness must not reject the type's OWN optional parameters."""
        layer = create_adapter_layer(
            'lora', kernel_initializer='he_normal', **VALID
        )
        assert isinstance(layer, LoRAAdapter)

    def test_keras_base_kwargs_are_not_reported_as_undeclared(self):
        """``name`` comes from ``keras.layers.Layer``, not the type's params."""
        layer = create_adapter_layer('lora', name='adapter_0', **VALID)
        assert layer.name == 'adapter_0'


class TestUnknownAndMissing:
    def test_an_unknown_type_raises_and_names_the_alternatives(self):
        with pytest.raises(ValueError) as exc:
            create_adapter_layer('lokl', **VALID)
        assert 'lora' in str(exc.value)

    def test_a_missing_required_parameter_raises_and_names_it(self):
        incomplete = {k: v for k, v in VALID.items() if k != 'rank'}
        with pytest.raises(ValueError) as exc:
            create_adapter_layer('lora', **incomplete)
        assert 'rank' in str(exc.value)

    @pytest.mark.parametrize('field,value', [
        ('output_dim', 0), ('rank', -1), ('alpha', 0.0),
        ('num_adapters', 0), ('num_adapters', -3),
    ])
    def test_a_non_positive_value_raises_at_the_factory(self, field, value):
        """Validation must not be delegated to the target constructor.

        Guide §9.2 is explicit that "let the constructor notice" is not
        validating. Both surfaces raise, so the guard cannot be satisfied by the
        layer alone.
        """
        with pytest.raises(ValueError):
            validate_adapter_config('lora', **{**VALID, field: value})

    def test_a_bool_is_rejected_where_an_int_is_required(self):
        """``True`` is an ``int`` subclass; ``rank=True`` must not read as 1."""
        with pytest.raises(ValueError, match='positive integer'):
            validate_adapter_config('lora', **{**VALID, 'rank': True})

    def test_a_non_numeric_alpha_raises(self):
        with pytest.raises(ValueError, match='alpha'):
            validate_adapter_config('lora', **{**VALID, 'alpha': 'eight'})


class TestRegistrySurface:
    def test_the_literal_and_the_registry_agree_in_both_directions(self):
        from typing import get_args

        from dl_techniques.layers.adapters.factory import AdapterType

        assert set(get_args(AdapterType)) == set(ADAPTER_REGISTRY)

    def test_the_registry_is_not_empty(self):
        """A parametrized repo-wide guard must assert a non-empty subject set."""
        assert ADAPTER_REGISTRY, "ADAPTER_REGISTRY is empty; nothing is pinned"

    def test_list_adapter_types_is_sorted_and_complete(self):
        listed = list_adapter_types()
        assert listed == sorted(ADAPTER_REGISTRY)
        assert set(listed) == set(ADAPTER_REGISTRY)

    def test_every_registry_class_is_a_keras_layer(self):
        import keras

        for key, info in ADAPTER_REGISTRY.items():
            assert issubclass(info['class'], keras.layers.Layer), key

    def test_every_registry_entry_declares_the_same_keys(self):
        for key, info in ADAPTER_REGISTRY.items():
            assert {'class', 'description', 'required_params',
                    'optional_params', 'use_case'} <= set(info), key

    def test_required_params_are_actually_constructor_parameters(self):
        """A required param the class does not declare is an unsatisfiable call."""
        import inspect

        for key, info in ADAPTER_REGISTRY.items():
            signature = inspect.signature(info['class'].__init__)
            for param in info['required_params']:
                assert param in signature.parameters, f"{key}.{param}"
                assert signature.parameters[param].default is inspect.Parameter.empty, (
                    f"{key}.{param} is listed required but has a default"
                )

    def test_optional_params_defaults_match_the_class_signature(self):
        """A registry that lies about a default is a silent config bug."""
        import inspect

        for key, info in ADAPTER_REGISTRY.items():
            signature = inspect.signature(info['class'].__init__)
            for name, declared in info['optional_params'].items():
                assert name in signature.parameters, f"{key}.{name}"
                actual = signature.parameters[name].default
                if declared is not None:
                    assert actual == declared, (
                        f"{key}.{name}: registry says {declared!r}, "
                        f"the class says {actual!r}"
                    )


class TestRegistryIsolation:
    def test_get_adapter_info_hands_back_a_deep_copy(self):
        """A shallow copy would let one caller corrupt dispatch process-wide."""
        info = get_adapter_info()
        info['lora']['optional_params']['kernel_initializer'] = 'MUTATED'
        info['lora']['required_params'].append('injected')
        fresh = get_adapter_info()
        assert fresh['lora']['optional_params']['kernel_initializer'] == \
            'glorot_uniform'
        assert 'injected' not in fresh['lora']['required_params']

    def test_get_adapter_requirements_is_a_copy_too(self):
        entry = get_adapter_requirements('lora')
        entry['optional_params']['rank'] = 999
        assert 'rank' not in get_adapter_requirements('lora')['optional_params']

    def test_get_adapter_requirements_raises_on_an_unknown_type(self):
        with pytest.raises(ValueError, match='lora'):
            get_adapter_requirements('nope')


class TestFromConfig:
    def test_a_type_keyed_dict_builds_the_layer(self):
        layer = create_adapter_from_config({'type': 'lora', **VALID})
        assert isinstance(layer, LoRAAdapter)
        assert layer.output_dim == 32

    def test_the_input_dict_is_not_mutated(self):
        config = {'type': 'lora', **VALID}
        create_adapter_from_config(config)
        assert config['type'] == 'lora'

    def test_a_missing_type_key_raises_and_lists_the_valid_types(self):
        with pytest.raises(ValueError) as exc:
            create_adapter_from_config(dict(VALID))
        assert 'lora' in str(exc.value)

    def test_a_non_dict_raises(self):
        with pytest.raises(ValueError, match='dictionary'):
            create_adapter_from_config('lora')

    def test_the_strict_check_still_applies_through_the_dict_path(self):
        with pytest.raises(ValueError) as exc:
            create_adapter_from_config(
                {'type': 'lora', 'dropout_rate': 0.1, **VALID}
            )
        assert STRICT_DROPPED_KEY_MARKER in str(exc.value)


class TestAssembleConfig:
    def test_wrapper_only_keys_are_dropped(self):
        merged = assemble_adapter_config(
            'lora', {**VALID, 'dropout_rate': 0.1, 'kernel_size': 3}
        )
        assert 'dropout_rate' not in merged
        assert 'kernel_size' not in merged
        assert merged['rank'] == 4

    def test_caller_args_pass_through_unfiltered(self):
        """An explicit caller request must still reach the strict check."""
        merged = assemble_adapter_config(
            'lora', dict(VALID), caller_args={'dropout_rate': 0.1}
        )
        assert merged['dropout_rate'] == 0.1
        with pytest.raises(ValueError) as exc:
            create_adapter_layer('lora', **merged)
        assert STRICT_DROPPED_KEY_MARKER in str(exc.value)

    def test_caller_args_win_over_the_wrapper_block(self):
        merged = assemble_adapter_config(
            'lora', {**VALID, 'rank': 4}, caller_args={'rank': 16}
        )
        assert merged['rank'] == 16

    def test_a_passthrough_key_survives_even_when_undeclared(self):
        """``name`` is a ``keras.layers.Layer`` base kwarg, not one of the
        type's declared params; a passthrough list is how a wrapper says it means
        it."""
        merged = assemble_adapter_config(
            'lora', {**VALID, 'name': 'my_adapter'}, passthrough=('name',)
        )
        assert merged['name'] == 'my_adapter'

    def test_a_key_absent_from_the_passthrough_list_is_still_dropped(self):
        """TWIN: the passthrough list must not become a blanket exemption."""
        merged = assemble_adapter_config(
            'lora', {**VALID, 'name': 'my_adapter', 'rank2': 3},
            passthrough=('name',),
        )
        assert 'rank2' not in merged
        assert merged['name'] == 'my_adapter'

    def test_an_unknown_type_raises(self):
        with pytest.raises(ValueError, match='lora'):
            assemble_adapter_config('nope', dict(VALID))
