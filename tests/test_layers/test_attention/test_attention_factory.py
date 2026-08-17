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
    STRICT_DROPPED_KEY_MARKER,
    AttentionType,
    assemble_attention_config,
    create_attention_layer,
    create_attention_from_config,
    validate_attention_config,
    get_attention_info,
    list_attention_types,
    get_attention_requirements,
)

# Minimal required params per registered type (satisfies validate + construction).
MINIMAL_PARAMS = {
    'anchor': {'dim': 64, 'num_heads': 4},
    'beit': {'dim': 64, 'window_size': 4, 'num_heads': 4},
    'capsule_routing': {'num_heads': 4},
    'cbam': {'channels': 32},
    'channel': {'channels': 32},
    'differential': {'dim': 64, 'num_heads': 4, 'head_dim': 16},
    'energy': {'dim': 64},
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
    """The registry must describe exactly 32 types and stay in sync with classes."""

    def test_registry_has_expected_types(self):
        # DECISION plan_2026-07-13_57c9833e/D-003
        # These are EXACT-EQUALITY counts on purpose. When you register a new attention
        # type and this goes RED, DO NOT relax it to `>=` — bump the number. The whole
        # value of this guard is that it forces the registration checklist (Literal +
        # registry + MINIMAL_PARAMS below + __init__ export); a `>=` assertion passes
        # while `MINIMAL_PARAMS` silently drifts out of sync and the new type ships with
        # zero factory coverage. See decisions.md D-003 / LESSONS [I:3].
        assert len(ATTENTION_REGISTRY) == 32
        assert len(list_attention_types()) == 32

    def test_literal_members_match_registry_keys(self):
        literal_members = set(typing.get_args(AttentionType))
        assert literal_members == set(ATTENTION_REGISTRY.keys())

    @pytest.mark.parametrize('attn_type', sorted(ATTENTION_REGISTRY.keys()))
    def test_registry_keys_are_real_ctor_args(self, attn_type):
        """No registry param may be lost — each must be a ctor arg.

        Guards against the inverse of the F4 bug (a registry key that the class
        does not accept would raise TypeError at construction).

        Note on the wording: since 2026-08-17
        (plan-2026-08-17T183311-79c63e38/D-011) an UNDECLARED caller key raises
        rather than being silently dropped, so this direction — a key the registry
        declares but the class cannot take — is the one the raise cannot help with.
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
        assert len(info) == 32

    def test_get_requirements_roundtrip(self):
        req = get_attention_requirements('anchor')
        assert 'head_dim' in req['optional_params']


# ==============================================================================
# Strict dropped-key behaviour (plan-2026-08-17T183311-79c63e38/D-011)
# ==============================================================================

class TestStrictDroppedKeys:
    """`create_attention_layer` REFUSES a key the target type does not declare.

    Before 2026-08-17 the factory filtered `kwargs` against the registry and
    discarded the rest without a word, which is how TRM shipped a
    permutation-equivariant reasoning stack: it handed `max_seq_len`/`rope_theta`
    to `'multi_head'`, which declares no RoPE parameter, and both keys evaporated.

    Every assertion here reads the MESSAGE TEXT, not just the exception type. That
    is deliberate and it is what the tests would otherwise miss: the factory body
    is wrapped in a `try/except (TypeError, ValueError)` that re-raises ANY
    ValueError as a generic "Please verify parameter compatibility" message. A
    `pytest.raises(ValueError)` alone passes just as happily against that wrapper,
    so it cannot tell the strict raise apart from an unrelated construction
    failure, and it would not have caught the raise being placed inside the `try`.
    """

    def test_undeclared_key_raises_and_names_every_dropped_key(self):
        with pytest.raises(ValueError) as excinfo:
            create_attention_layer(
                'multi_head', dim=8, rope_theta=10000.0, max_seq_len=64
            )
        message = str(excinfo.value)
        assert STRICT_DROPPED_KEY_MARKER in message, message
        assert 'rope_theta' in message, message
        assert 'max_seq_len' in message, message
        assert 'multi_head' in message, message
        # the raise must NOT be the outer wrapper's generic re-raise
        assert 'Please verify parameter compatibility' not in message, (
            "the strict raise was swallowed by the factory's outer "
            "`except (TypeError, ValueError)` re-wrap -- it must fire BEFORE the "
            f"`try`. Got: {message}"
        )

    def test_the_same_keys_construct_fine_under_group_query(self):
        """Control: the keys are not intrinsically bad, the TYPE is."""
        layer = create_attention_layer(
            'group_query', dim=8, num_heads=2, num_kv_heads=2,
            rope_theta=10000.0, max_seq_len=64,
        )
        assert layer is not None
        assert layer.rope_theta == 10000.0
        assert layer.max_seq_len == 64

    def test_gated_accepts_num_kv_heads_through_the_factory(self):
        """D-008: declared in the same commit as the raise, or it inverts.

        `GatedAttention.__init__` has always accepted `num_kv_heads`; the registry
        entry did not declare it. Under the raise that turns a supported feature
        into a hard failure, so the declaration is part of this change.
        """
        layer = create_attention_layer(
            'gated', dim=32, num_heads=4, num_kv_heads=2
        )
        assert layer.num_kv_heads == 2

    def test_unknown_type_keeps_its_own_failure_mode(self):
        """The strict check must not intercept an unknown `attention_type`."""
        with pytest.raises(ValueError) as excinfo:
            create_attention_layer('does_not_exist', dim=64, nonsense=1)
        message = str(excinfo.value)
        assert STRICT_DROPPED_KEY_MARKER not in message, message
        assert 'does_not_exist' in message, message

    def test_every_registry_default_survives_its_own_type(self):
        """No entry declares a key its own strict check would then reject."""
        for attn_type, params in sorted(MINIMAL_PARAMS.items()):
            info = ATTENTION_REGISTRY[attn_type]
            declared = set(info['required_params']) | set(info['optional_params'])
            assert set(params) <= declared, (
                f"MINIMAL_PARAMS['{attn_type}'] carries {sorted(set(params) - declared)}, "
                f"which the entry does not declare -- the strict factory would raise"
            )


class TestAssembleAttentionConfig:
    """The wrapper-side pre-filter that makes the raise livable.

    Interface contract lives at the function; these pin the two halves callers
    get wrong: wrapper defaults ARE filtered, caller args are NOT.
    """

    def test_wrapper_defaults_are_filtered(self):
        config = assemble_attention_config(
            'multi_head',
            {'dim': 8, 'num_heads': 2, 'bias_initializer': 'zeros'},
        )
        assert 'bias_initializer' not in config
        assert config == {'dim': 8, 'num_heads': 2}

    def test_caller_args_are_never_filtered(self):
        """A caller typo must survive to the factory, which is what raises."""
        config = assemble_attention_config(
            'multi_head', {'dim': 8, 'num_heads': 2}, {'rope_theta': 1.0}
        )
        assert config['rope_theta'] == 1.0
        with pytest.raises(ValueError) as excinfo:
            create_attention_layer('multi_head', **config)
        assert STRICT_DROPPED_KEY_MARKER in str(excinfo.value)

    def test_caller_args_win_over_wrapper_defaults(self):
        config = assemble_attention_config(
            'multi_head', {'dim': 8, 'num_heads': 2}, {'num_heads': 4}
        )
        assert config['num_heads'] == 4

    def test_inputs_are_not_mutated(self):
        wrapper = {'dim': 8, 'num_heads': 2, 'bias_initializer': 'zeros'}
        caller = {'num_heads': 4}
        assemble_attention_config('multi_head', wrapper, caller)
        assert wrapper == {'dim': 8, 'num_heads': 2, 'bias_initializer': 'zeros'}
        assert caller == {'num_heads': 4}

    def test_unknown_type_raises_naming_the_available_types(self):
        with pytest.raises(ValueError, match="Unknown attention_type"):
            assemble_attention_config('does_not_exist', {'dim': 8})
