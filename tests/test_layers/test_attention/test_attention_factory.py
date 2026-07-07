"""Test Suite for the Attention Factory.

Covers the previously-untested ``create_attention_layer`` factory surface:

1. Construction of all 27 registered attention types via the factory.
2. Registry integrity — every ``required_params`` / ``optional_params`` key is a
   real constructor argument of the target class (no silently-dropped params).
3. Parameter passthrough for the registry entries completed in plan
   plan_2026-06-14_ab855e7e/F4 (anchor / channel / spatial / tripse1-4) — values
   supplied via the factory must actually reach the instantiated layer.
4. Validation + config helpers.
"""

import inspect
import typing
import pytest

from dl_techniques.layers.attention.factory import (
    ATTENTION_REGISTRY,
    AttentionType,
    create_attention_layer,
    create_attention_from_config,
    validate_attention_config,
    get_attention_info,
    list_attention_types,
    get_attention_requirements,
)

# Minimal required params per registered type (satisfies validate + construction).
MINIMAL_PARAMS = {
    'anchor': {'dim': 64},
    'capsule_routing': {'num_heads': 4},
    'cbam': {'channels': 32},
    'channel': {'channels': 32},
    'differential': {'dim': 64, 'num_heads': 4, 'head_dim': 16},
    'fnet': {},
    'gated': {'dim': 64, 'num_heads': 4},
    'group_query': {'dim': 64, 'num_heads': 4, 'num_kv_heads': 2},
    'hopfield': {'num_heads': 4, 'key_dim': 16},
    'lighthouse': {'dim': 64, 'num_heads': 4},
    'linear': {'dim': 64},
    'mobile_mqa': {'dim': 64},
    'multi_head': {'dim': 64},
    'multi_head_cross': {'dim': 64},
    'multi_head_latent': {'dim': 64, 'num_heads': 4, 'kv_latent_dim': 32},
    'non_local': {'attention_channels': 32},
    'perceiver': {'dim': 64},
    'performer': {'dim': 64},
    'ring': {'dim': 64},
    'rpc': {'dim': 64},
    'shared_weights_cross': {'dim': 64},
    'single_window': {'dim': 64, 'window_size': 7, 'num_heads': 8},
    'spatial': {},
    'tripse1': {},
    'tripse2': {},
    'tripse3': {},
    'tripse4': {},
    'wave_field': {'dim': 64},
    'window': {'dim': 64, 'window_size': 4, 'num_heads': 4},
    'window_zigzag': {'dim': 64, 'window_size': 4, 'num_heads': 4},
}


def _ctor_param_names(cls_or_fn):
    """Return the set of accepted keyword names for a class or factory callable."""
    target = cls_or_fn.__init__ if inspect.isclass(cls_or_fn) else cls_or_fn
    sig = inspect.signature(target)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    names = {
        name for name, p in sig.parameters.items()
        if name not in ('self', 'args', 'kwargs')
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    return names, has_var_kw


class TestRegistryIntegrity:
    """The registry must describe exactly 30 types and stay in sync with classes."""

    def test_registry_has_expected_types(self):
        assert len(ATTENTION_REGISTRY) == 30
        assert len(list_attention_types()) == 30

    def test_literal_members_match_registry_keys(self):
        literal_members = set(typing.get_args(AttentionType))
        assert literal_members == set(ATTENTION_REGISTRY.keys())

    @pytest.mark.parametrize('attn_type', sorted(ATTENTION_REGISTRY.keys()))
    def test_registry_keys_are_real_ctor_args(self, attn_type):
        """No registry param may be silently dropped — each must be a ctor arg.

        Guards against the inverse of the F4 bug (a registry key that the class
        does not accept would raise TypeError at construction).
        """
        info = ATTENTION_REGISTRY[attn_type]
        cls = info['class']
        accepted, has_var_kw = _ctor_param_names(cls)
        if has_var_kw:
            pytest.skip(f"{attn_type} target accepts **kwargs; key check N/A")
        declared = set(info['required_params']) | set(info['optional_params'].keys())
        unknown = declared - accepted - {'name'}
        assert not unknown, (
            f"Registry for '{attn_type}' declares params not accepted by "
            f"{cls.__name__}: {sorted(unknown)}"
        )


class TestConstructAll:
    """Every registered type must construct through the factory."""

    @pytest.mark.parametrize('attn_type', sorted(MINIMAL_PARAMS.keys()))
    def test_construct(self, attn_type):
        layer = create_attention_layer(attn_type, **MINIMAL_PARAMS[attn_type])
        assert layer is not None

    def test_minimal_params_cover_all_registered(self):
        assert set(MINIMAL_PARAMS.keys()) == set(ATTENTION_REGISTRY.keys())


class TestNewlyRegisteredTypesF8:
    """wave_field + single_window (F8) must construct AND build via the factory."""

    def test_wave_field_constructs_and_builds(self):
        layer = create_attention_layer('wave_field', dim=64)
        assert layer is not None
        layer.build((None, 16, 64))
        assert layer.built

    def test_single_window_constructs_and_builds(self):
        layer = create_attention_layer(
            'single_window', dim=64, window_size=7, num_heads=8
        )
        assert layer is not None
        layer.build((None, 49, 64))
        assert layer.built


class TestParamPassthroughF4:
    """Params completed in F4 must actually reach the constructed layer."""

    def test_anchor_head_dim(self):
        layer = create_attention_layer('anchor', dim=64, num_heads=4, head_dim=8)
        assert layer.head_dim == 8

    def test_anchor_probability_type(self):
        layer = create_attention_layer(
            'anchor', dim=64, num_heads=4, probability_type='softmax'
        )
        assert layer.probability_type == 'softmax'

    def test_channel_activation_passthrough(self):
        layer = create_attention_layer(
            'channel', channels=32,
            intermediate_activation_type='gelu',
            gate_activation_type='hard_sigmoid',
        )
        assert layer.intermediate_activation_type == 'gelu'
        assert layer.gate_activation_type == 'hard_sigmoid'

    def test_spatial_gate_activation_passthrough(self):
        layer = create_attention_layer('spatial', gate_activation_type='hard_sigmoid')
        assert layer.gate_activation_type == 'hard_sigmoid'

    @pytest.mark.parametrize('attn_type', ['tripse1', 'tripse2', 'tripse3', 'tripse4'])
    def test_tripse_gate_activation_passthrough(self, attn_type):
        layer = create_attention_layer(attn_type, gate_activation_type='hard_sigmoid')
        assert layer.gate_activation_type == 'hard_sigmoid'

    def test_tripse4_se_reduction_passthrough(self):
        layer = create_attention_layer('tripse4', se_reduction_activation_type='gelu')
        assert layer.se_reduction_activation_type == 'gelu'


class TestParamPassthroughFCT:
    """optional_params completed in FCT must reach the constructed instance.

    Construct-smoke would pass even if the factory silently dropped these
    kwargs (they have defaults); these assertions prove the forwarded value
    lands on the layer rather than being filtered out by ``valid_param_names``.
    """

    def test_multi_head_qk_norm_forwarded(self):
        layer = create_attention_layer(
            'multi_head', dim=64, qk_norm_type='rms_norm'
        )
        assert layer.qk_norm_type == 'rms_norm'

    def test_multi_head_probability_type_forwarded(self):
        layer = create_attention_layer(
            'multi_head', dim=64, probability_type='softmax'
        )
        assert layer.probability_type == 'softmax'

    def test_capsule_routing_qk_norm_forwarded(self):
        layer = create_attention_layer(
            'capsule_routing', num_heads=4,
            probability_type='softmax', qk_norm_type='rms_norm',
        )
        assert layer.qk_norm_type == 'rms_norm'
        assert layer.probability_type == 'softmax'


class TestFactoryHelpers:
    def test_from_config(self):
        layer = create_attention_layer('multi_head', dim=64)
        cfg = {'type': 'multi_head', 'dim': 64}
        assert create_attention_from_config(cfg) is not None
        assert layer is not None

    def test_validate_unknown_type_raises(self):
        with pytest.raises(ValueError):
            validate_attention_config('does_not_exist', dim=64)

    def test_validate_missing_required_raises(self):
        with pytest.raises(ValueError):
            validate_attention_config('group_query', dim=64)  # missing num_kv_heads

    def test_validate_group_query_divisibility(self):
        with pytest.raises(ValueError):
            validate_attention_config(
                'group_query', dim=64, num_heads=5, num_kv_heads=2
            )

    def test_get_attention_info_complete(self):
        info = get_attention_info()
        assert len(info) == 30

    def test_get_requirements_roundtrip(self):
        req = get_attention_requirements('anchor')
        assert 'head_dim' in req['optional_params']
