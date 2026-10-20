"""
Test suite for the activation factory's strict-drop guard
(`layers/activations/factory.py`).

Covers N-01 of plan-2026-08-18T140459-7991552f:

- an undeclared kwarg is rejected with a NAMED, factory-level message carrying
  ``STRICT_DROPPED_KEY_MARKER``;
- base ``keras.layers.Layer`` kwargs (``trainable``, ``dtype``) are exempted and
  still build -- the non-regression pin, because these DID build before the
  hardening and a strict-to-registry-keys-only check would have broken them;
- the strict-drop marker literal and the Keras base-parameter set are identical
  in every ``src`` module that defines them (found by a walk, not a list).

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

    def test_marker_is_identical_in_every_factory(self):
        """Every copy of the marker literal in ``src`` must stay in lockstep.

        `STRICT_DROPPED_KEY_MARKER` is defined once per strict factory because
        every candidate shared home is either a peer package -- importing one
        factory from another drags its whole layer tree into the importer's
        graph -- or a new module. A hand-maintained "keep these equal"
        invariant is a defect; this test is what makes it not one. It FINDS the
        copies (a module-level assignment in any ``src`` file) instead of
        listing them: the earlier hand list named five and missed the ssm and
        adapters copies (plan-2026-10-14T062616-81938e81 D-200).
        """
        copies = _module_level_literals("STRICT_DROPPED_KEY_MARKER")
        assert len(copies) >= 8, f"the walk found only {sorted(copies)}"
        assert "layers/sequence_pooling/factory.py" in copies
        assert set(copies.values()) == {STRICT_DROPPED_KEY_MARKER}, (
            f"the strict-drop marker has drifted between factories: {copies}"
        )

    def test_keras_base_param_set_is_identical_in_every_factory(self):
        """Every copy of `_KERAS_BASE_PARAMS` in ``src`` is the same set.

        Same lockstep argument as the marker above, same reason it is found
        by a walk instead of trusted to a list.
        """
        copies = _module_level_literals("_KERAS_BASE_PARAMS")
        assert len(copies) >= 5, f"the walk found only {sorted(copies)}"
        assert "layers/sequence_pooling/factory.py" in copies
        assert set(copies.values()) == {_KERAS_BASE_PARAMS}, (
            f"the Keras base-parameter set has drifted between factories: {copies}"
        )


def _module_level_literals(name):
    """``{path under src/dl_techniques: value}`` of every module-level ``name = <literal>``.

    A ``frozenset({...})`` call is read as the frozenset of its literal argument.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "src" / "dl_techniques"
    found = {}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if name not in text:
            continue
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
                continue
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset"):
                literal = frozenset(ast.literal_eval(value.args[0]))
            else:
                literal = ast.literal_eval(value)
            found[path.relative_to(root).as_posix()] = literal
    return found
