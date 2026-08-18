"""
Test suite for the activation factory's strict-drop guard
(`layers/activations/factory.py`).

Covers N-01 of plan-2026-08-18T140459-7991552f:

- an undeclared kwarg is rejected with a NAMED, factory-level message carrying
  ``STRICT_DROPPED_KEY_MARKER``;
- base ``keras.layers.Layer`` kwargs (``trainable``, ``dtype``) are exempted and
  still build -- the non-regression pin, because these DID build before the
  hardening and a strict-to-registry-keys-only check would have broken them;
- the strict-drop marker literal is identical across all five factories that
  define it.

Registry-vs-constructor DRIFT is deliberately NOT re-tested here. It already has a
home: `tests/test_layers/test_factory_registry_drift.py` covers `activations`
(and now `sampling`) in all three directions -- missing, phantom, and mismatched
defaults -- with wrapper resolution this file could not reproduce. A second copy
would be the duplication this plan exists to remove.

WHAT THIS SUITE DOES NOT CLAIM. The plan premise was that
``create_activation_layer``'s ``or k in kwargs`` filter clause let an undeclared
kwarg through SILENTLY. Measured at HEAD, it did not: Keras 3's own
``Layer.__init__`` already raised ``Unrecognized keyword arguments passed to
Mish`` for ``create_activation_layer('mish', bogus_param=1)``. The unreachable
filter was real; the silent construction was not. What changed is the QUALITY and
the ORIGIN of the diagnostic (named type, named keys, the accepted set) plus the
registry-drift door -- so the assertions below pin the message, not a
before/after behavioural difference that does not exist. See decisions.md D-017.
"""

import keras
import pytest

from dl_techniques.layers.activations.factory import (
    ACTIVATION_REGISTRY,
    STRICT_DROPPED_KEY_MARKER,
    _KERAS_BASE_PARAMS,
    create_activation_layer,
    resolve_activation_layer,
)


class TestStrictDroppedKeys:
    """Pins N-01: undeclared kwargs raise a named factory-level error."""

    @pytest.mark.parametrize("activation_type", sorted(ACTIVATION_REGISTRY))
    def test_undeclared_kwarg_raises_with_marker(self, activation_type):
        """Every registered type rejects an undeclared key by the same rule.

        Parameterized over the WHOLE registry rather than a hand-picked sample:
        a check that only visits three types cannot see a registry entry added
        later, which is the drift this guard exists to stop.
        """
        # `match=` is NOT used: the marker literal contains `(s)`, which the
        # regex engine reads as a capture group, so `match=` would silently
        # test for "parameters" and fail against the real message. The sibling
        # factories' suites assert containment for the same reason.
        with pytest.raises(ValueError) as exc:
            create_activation_layer(
                activation_type, definitely_not_an_activation_argument=1
            )
        assert STRICT_DROPPED_KEY_MARKER in str(exc.value), str(exc.value)

    def test_message_names_the_type_the_key_and_the_accepted_set(self):
        """A diagnostic that does not say WHAT is accepted is not a diagnostic.

        This is the whole value of the change (the pre-existing Keras message
        named neither the factory nor the registry), so it is asserted directly
        rather than inferred from the fact that something raised.
        """
        with pytest.raises(ValueError) as exc:
            create_activation_layer('mish', bogus_param=1)
        message = str(exc.value)
        assert "create_activation_layer('mish')" in message
        assert "bogus_param" in message
        assert "Mish" in message
        assert "accepts only" in message

    @pytest.mark.parametrize("base_param, value", [
        ("trainable", False),
        ("dtype", "float32"),
    ])
    def test_keras_base_params_are_exempt_and_still_build(self, base_param, value):
        """NON-REGRESSION pin, not a feature test.

        Both of these built at HEAD (measured). A hardening step that tightened
        the filter to registry keys only would have broken them, turning a
        guard into a regression. If someone later removes the
        `_KERAS_BASE_PARAMS` exemption, this fails.
        """
        layer = create_activation_layer('mish', **{base_param: value})
        assert isinstance(layer, keras.layers.Layer)

    def test_declared_params_still_reach_the_layer(self):
        """The guard must not reject what the registry declares.

        Without this the suite would be satisfied by a factory that rejects
        EVERYTHING -- the classic vacuous-guard shape.
        """
        layer = create_activation_layer('relu_k', k=5)
        assert layer.k == 5

    def test_unknown_activation_type_keeps_its_own_error(self):
        """The strict block must fall through for an unregistered type.

        `ACTIVATION_REGISTRY.get()` returns None there, so the existing
        `validate_activation_config` message (which lists the available types)
        is what a caller sees. Pinned because the strict check runs first and
        could easily have swallowed it.
        """
        with pytest.raises(ValueError, match="Unknown activation type"):
            create_activation_layer('not_a_real_activation')

    def test_resolve_activation_layer_inherits_the_guard(self):
        """`resolve_activation_layer` delegates for registry types.

        It is the wrapper most call sites actually use, so the guard has to be
        reachable through it or 28 of the 28 measured call sites bypass it.
        """
        with pytest.raises(ValueError) as exc:
            resolve_activation_layer('mish', bogus_param=1)
        assert STRICT_DROPPED_KEY_MARKER in str(exc.value), str(exc.value)

    def test_plain_keras_activation_path_is_untouched(self):
        """`resolve_activation_layer('sigmoid')` never enters the factory."""
        layer = resolve_activation_layer('sigmoid')
        assert isinstance(layer, keras.layers.Activation)

    def test_marker_is_identical_across_all_five_factories(self):
        """The five copies of the marker literal must stay in lockstep.

        `STRICT_DROPPED_KEY_MARKER` is defined five times (attention, ffn,
        embedding, activations, sampling) because every candidate shared home is
        either a peer package -- importing one factory from another drags its
        whole layer tree into the importer's graph -- or a new module, which
        this plan has no abstraction budget for. A hand-maintained "keep these
        equal" invariant is a defect; this test is what makes it not one.
        """
        from dl_techniques.layers.attention.factory import (
            STRICT_DROPPED_KEY_MARKER as attention_marker,
        )
        from dl_techniques.layers.embedding.factory import (
            STRICT_DROPPED_KEY_MARKER as embedding_marker,
        )
        from dl_techniques.layers.ffn.factory import (
            STRICT_DROPPED_KEY_MARKER as ffn_marker,
        )
        from dl_techniques.layers.sampling import (
            STRICT_DROPPED_KEY_MARKER as sampling_marker,
        )
        markers = {
            'attention': attention_marker,
            'ffn': ffn_marker,
            'embedding': embedding_marker,
            'activations': STRICT_DROPPED_KEY_MARKER,
            'sampling': sampling_marker,
        }
        assert len(set(markers.values())) == 1, (
            f"the strict-drop marker has drifted between factories: {markers}"
        )

    def test_keras_base_param_set_matches_the_norms_factory(self):
        """`_KERAS_BASE_PARAMS` is duplicated from `norms/factory.py`.

        Same lockstep argument as the marker above, same reason it is gated
        instead of trusted.
        """
        from dl_techniques.layers.norms.factory import (
            _KERAS_BASE_PARAMS as norms_base_params,
        )
        assert _KERAS_BASE_PARAMS == norms_base_params
