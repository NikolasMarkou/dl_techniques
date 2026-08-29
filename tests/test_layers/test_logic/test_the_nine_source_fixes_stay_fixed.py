"""
Guards for the nine source fixes applied on 2026-08-29 (iter-1/step-2).

Each guard names the fix it pins and says what kind of claim it makes.
Three of the nine change observable behaviour and their guards fail
against the pre-fix source: A3 (the aux loss read the deprecated alias),
B4 (``get_config`` omitted a constructor argument) and B6 (the static
shape contract was checked only in ``build``). The rest pin a removal, a
spelling or a registration key, and make no behavioural claim; they are
marked as such in their own docstrings.
"""

import ast
import inspect
import warnings
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.layers.logic.arithmetic_operators import (
    LearnableArithmeticOperator,
)
from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator
from dl_techniques.layers.logic.neural_circuit import (
    CircuitDepthLayer,
    LearnableNeuralCircuit,
)

PACKAGE_DIR = Path(
    inspect.getfile(LearnableLogicOperator)
).parent
SOURCE_FILES = sorted(PACKAGE_DIR.glob("*.py"))
ALL_CLASSES = (
    LearnableLogicOperator,
    LearnableArithmeticOperator,
    CircuitDepthLayer,
    LearnableNeuralCircuit,
)


def _call_bodies(tree):
    """Yield every ``def call`` FunctionDef in a parsed module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            yield node


class TestA2TheDeadLocalIsGone:
    """A2. Removal pin. Makes no behavioural claim: the local was
    assigned and never read, so deleting it cannot change any output."""

    def test_input_rank_appears_nowhere_in_the_package(self):
        hits = [
            f"{p.name}:{i}"
            for p in SOURCE_FILES
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if "input_rank" in line
        ]
        assert hits == []

    def test_no_len_of_a_shape_object_inside_any_call(self):
        """The v2 4.1 family the dead local also belonged to: a Python
        ``len()`` read of ``ops.shape(...)`` inside ``call``."""
        offenders = []
        for path in SOURCE_FILES:
            tree = ast.parse(path.read_text())
            for fn in _call_bodies(tree):
                for node in ast.walk(fn):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "len"
                        and node.args
                    ):
                        continue
                    inner = node.args[0]
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "shape"
                    ):
                        offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == []


class TestA3TheAuxLossReadsTheCanonicalName:
    """A3. Behavioural. This class fails against the pre-fix source:
    ``_maybe_load_balance_loss`` guarded and scaled on the deprecated
    alias, so zeroing the alias switched the loss off."""

    @staticmethod
    def _built_layer(coefficient):
        layer = CircuitDepthLayer(
            num_logic_ops=2,
            num_arithmetic_ops=2,
            gate_entropy_coefficient=coefficient,
        )
        x = keras.ops.convert_to_tensor(
            np.random.RandomState(0).normal(size=(3, 8)).astype("float32")
        )
        layer(x)
        return layer, x

    def test_loss_still_fires_with_the_alias_zeroed(self):
        layer, x = self._built_layer(0.1)
        assert layer.gate_entropy_coefficient == 0.1
        # Zero the deprecated alias only. Nothing should read it.
        layer.load_balance_coefficient = 0.0
        layer(x)
        assert layer.losses, "aux loss disappeared when the alias went to 0"
        assert float(keras.ops.convert_to_numpy(layer.losses[0])) > 0.0

    def test_loss_stops_when_the_canonical_name_is_zeroed(self):
        """The 'something changed' twin: zeroing the canonical name does
        switch the loss off, so the assertion above is not vacuous."""
        layer, x = self._built_layer(0.1)
        layer.gate_entropy_coefficient = 0.0
        layer(x)
        assert layer.losses == []


class TestA4TheMaxAndMinHelpersAreNamedTruthfully:
    """A4. Spelling pin. Makes no behavioural claim: both bodies were
    already ``keras.ops.maximum`` / ``minimum`` and are unchanged."""

    def test_the_soft_names_are_gone_and_the_truthful_ones_exist(self):
        op = LearnableArithmeticOperator(operation_types=["max", "min"])
        assert not hasattr(op, "_soft_max")
        assert not hasattr(op, "_soft_min")
        assert callable(op._elementwise_max)
        assert callable(op._elementwise_min)

    def test_no_source_file_still_says_soft_max_or_soft_min(self):
        text = "\n".join(p.read_text() for p in SOURCE_FILES)
        assert "_soft_max" not in text
        assert "_soft_min" not in text

    def test_the_helpers_are_the_plain_elementwise_ops(self):
        op = LearnableArithmeticOperator(operation_types=["max", "min"])
        rng = np.random.RandomState(1)
        a = keras.ops.convert_to_tensor(rng.normal(size=(4, 6)).astype("float32"))
        b = keras.ops.convert_to_tensor(rng.normal(size=(4, 6)).astype("float32"))
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(op._elementwise_max(a, b)),
            keras.ops.convert_to_numpy(keras.ops.maximum(a, b)),
        )
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(op._elementwise_min(a, b)),
            keras.ops.convert_to_numpy(keras.ops.minimum(a, b)),
        )


class TestB2NoLoggingInsideCall:
    """B2. The AST half is a removal pin. The raise half is behavioural:
    an unknown operation key used to fall through to a warning plus a
    silent identity."""

    def test_no_logger_call_inside_any_call_body(self):
        """AST, not grep: a ``logger.`` inside a docstring is not a
        call."""
        offenders = []
        for path in SOURCE_FILES:
            tree = ast.parse(path.read_text())
            for fn in _call_bodies(tree):
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "logger"
                    ):
                        offenders.append(
                            f"{path.name}:{node.lineno} logger.{node.func.attr}"
                        )
        assert offenders == []

    def test_the_ast_scan_can_see_a_logger_call_it_is_meant_to_catch(self):
        """Anti-vacuity control for the scan above."""
        tree = ast.parse(
            "def call(self, x):\n"
            "    '''logger.warning in a docstring is not a call'''\n"
            "    logger.warning('boom')\n"
            "    return x\n"
        )
        found = [
            node
            for fn in _call_bodies(tree)
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ]
        assert len(found) == 1

    @pytest.mark.parametrize(
        "cls,valid_ops",
        [
            (LearnableLogicOperator, ["and", "or"]),
            (LearnableArithmeticOperator, ["add", "multiply"]),
        ],
    )
    def test_an_unknown_op_raises_instead_of_warning_and_going_identity(
        self, cls, valid_ops
    ):
        # Two operations, not one: a softmax over a size-1 axis warns.
        layer = cls(operation_types=valid_ops)
        x = keras.ops.convert_to_tensor(
            np.random.RandomState(2).uniform(size=(2, 4)).astype("float32")
        )
        layer([x, x])
        # Only reachable by mutating the list after construction, which
        # is exactly the state the old identity fallback hid.
        layer.operation_types = ["not_a_real_operation"]
        with pytest.raises(ValueError, match="Unknown operation type"):
            layer([x, x])


class TestB3TrainingIsForwardedToTheChannelMix:
    """B3. Spelling pin. Makes no behavioural claim: the sub-layer is a
    ``Dense``, which ignores ``training``. It is forwarded because Keras
    3 propagates ``training`` through one mutable call-context slot."""

    def test_the_channel_mix_call_passes_training(self):
        tree = ast.parse(
            (PACKAGE_DIR / "neural_circuit.py").read_text()
        )
        sites = []
        for fn in _call_bodies(tree):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_channel_mix_layer"
                ):
                    sites.append(node)
        assert len(sites) == 1, "expected exactly one channel-mix call site"
        assert "training" in {kw.arg for kw in sites[0].keywords}

    def test_every_sublayer_call_inside_call_forwards_training(self):
        """The rule, not just the one site: any call on a ``self.<x>``
        attribute made from inside ``call`` carries ``training=``."""
        tree = ast.parse((PACKAGE_DIR / "neural_circuit.py").read_text())
        offenders = []
        for fn in _call_bodies(tree):
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr.startswith(("_channel_mix", "layer_norm"))
                ):
                    continue
                if "training" not in {kw.arg for kw in node.keywords}:
                    offenders.append(f"line {node.lineno}: {node.func.attr}")
        assert offenders == []


class TestB4EveryConstructorArgumentIsAConfigKey:
    """B4. Behavioural. Fails against the pre-fix source, where
    ``load_balance_coefficient`` was an ``__init__`` parameter of two
    classes and a ``get_config`` key of neither."""

    @staticmethod
    def _instance(cls):
        if cls is LearnableNeuralCircuit:
            return cls(circuit_depth=2)
        return cls()

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_signature_minus_config_keys_is_empty(self, cls):
        params = set(inspect.signature(cls.__init__).parameters)
        params -= {"self", "kwargs"}
        missing = params - set(self._instance(cls).get_config())
        assert missing == set()

    @pytest.mark.parametrize(
        "cls", (CircuitDepthLayer, LearnableNeuralCircuit),
        ids=lambda c: c.__name__,
    )
    def test_the_alias_key_is_present_and_always_none(self, cls):
        config = self._instance(cls).get_config()
        assert "load_balance_coefficient" in config
        assert config["load_balance_coefficient"] is None

    @pytest.mark.parametrize(
        "cls", (CircuitDepthLayer, LearnableNeuralCircuit),
        ids=lambda c: c.__name__,
    )
    def test_round_trip_from_the_canonical_name_never_warns(self, cls):
        kwargs = {"gate_entropy_coefficient": 0.3}
        if cls is LearnableNeuralCircuit:
            kwargs["circuit_depth"] = 2
        config = cls(**kwargs).get_config()
        assert config["gate_entropy_coefficient"] == 0.3
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            restored = cls.from_config(config)
        assert restored.gate_entropy_coefficient == 0.3

    @pytest.mark.parametrize(
        "cls", (CircuitDepthLayer, LearnableNeuralCircuit),
        ids=lambda c: c.__name__,
    )
    def test_round_trip_from_the_deprecated_name_warns_once_then_never(
        self, cls
    ):
        """The alias warns at construction and never again: the config
        carries the value under the canonical key, so no load
        re-triggers the warning and no load doubles the value."""
        kwargs = {"load_balance_coefficient": 0.25}
        if cls is LearnableNeuralCircuit:
            kwargs["circuit_depth"] = 2
        with pytest.warns(DeprecationWarning, match="load_balance_coefficient"):
            layer = cls(**kwargs)
        config = layer.get_config()
        assert config["gate_entropy_coefficient"] == 0.25
        assert config["load_balance_coefficient"] is None
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            restored = cls.from_config(config)
        assert restored.gate_entropy_coefficient == 0.25
        # And a second round trip is still 0.25, not 0.5.
        assert (
            cls.from_config(restored.get_config()).gate_entropy_coefficient
            == 0.25
        )


class TestB5TheShapeCheckLivesInOnePureHelper:
    """B5. Structural pin plus a behavioural equivalence check across
    the three call sites."""

    @pytest.mark.parametrize(
        "filename", ("logic_operators.py", "arithmetic_operators.py")
    )
    def test_exactly_one_definition_of_the_detection_per_file(self, filename):
        text = (PACKAGE_DIR / filename).read_text()
        assert text.count("is_list_of_shapes = (") == 1
        assert text.count("def _canonical_input_shape(") == 1

    @pytest.mark.parametrize(
        "filename", ("logic_operators.py", "arithmetic_operators.py")
    )
    def test_build_and_compute_output_shape_both_call_the_helper(
        self, filename
    ):
        tree = ast.parse((PACKAGE_DIR / filename).read_text())
        callers = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_canonical_input_shape"
                ):
                    callers.add(node.name)
        assert {"build", "compute_output_shape"} <= callers
        # call() reaches it through _assert_call_shape_contract.
        assert "_assert_call_shape_contract" in callers

    @pytest.mark.parametrize("cls", (LearnableLogicOperator,
                                     LearnableArithmeticOperator),
                             ids=lambda c: c.__name__)
    def test_all_three_sites_agree_on_a_mismatched_pair(self, cls):
        pattern = "Input tensors must have the same shape"
        with pytest.raises(ValueError, match=pattern):
            cls().build([(4, 8), (4, 16)])
        with pytest.raises(ValueError, match=pattern):
            cls().compute_output_shape([(4, 8), (4, 16)])
        rng = np.random.RandomState(3)
        a = keras.ops.convert_to_tensor(rng.uniform(size=(4, 8)).astype("float32"))
        b = keras.ops.convert_to_tensor(rng.uniform(size=(4, 16)).astype("float32"))
        with pytest.raises(ValueError, match=pattern):
            cls()([a, b])


class TestB6TheShapeContractIsRecheckedInCall:
    """B6. Behavioural. Fails against the pre-fix source, where every
    class checked its static shape contract only in ``build`` and so
    checked it once, against whatever shape arrived first."""

    @staticmethod
    def _make(cls):
        if cls is LearnableNeuralCircuit:
            return cls(
                circuit_depth=2,
                num_logic_ops_per_depth=2,
                num_arithmetic_ops_per_depth=2,
                selection_mode="per_channel",
            )
        if cls is CircuitDepthLayer:
            return cls(
                num_logic_ops=2,
                num_arithmetic_ops=2,
                selection_mode="per_channel",
            )
        return cls(selection_mode="per_channel")

    @staticmethod
    def _operands(cls, x):
        """The operator classes reject a single tensor for a binary op."""
        if cls in (LearnableLogicOperator, LearnableArithmeticOperator):
            return [x, x]
        return x

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_calling_at_a_conflicting_width_raises(self, cls):
        layer = self._make(cls)
        rng = np.random.RandomState(4)
        narrow = keras.ops.convert_to_tensor(
            rng.uniform(size=(3, 8)).astype("float32")
        )
        wide = keras.ops.convert_to_tensor(
            rng.uniform(size=(3, 16)).astype("float32")
        )
        layer(self._operands(cls, narrow))
        assert layer.built
        with pytest.raises(ValueError, match="was built for a last axis of"):
            layer(self._operands(cls, wide))

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_calling_again_at_the_built_width_still_works(self, cls):
        """The 'something changed' twin: the contract rejects only the
        conflicting shape, not every second call."""
        layer = self._make(cls)
        rng = np.random.RandomState(5)
        x = keras.ops.convert_to_tensor(
            rng.uniform(size=(3, 8)).astype("float32")
        )
        first = keras.ops.convert_to_numpy(layer(self._operands(cls, x)))
        second = keras.ops.convert_to_numpy(layer(self._operands(cls, x)))
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "cls", (CircuitDepthLayer, LearnableNeuralCircuit),
        ids=lambda c: c.__name__,
    )
    def test_the_rank_contract_is_rechecked_too(self, cls):
        layer = cls() if cls is CircuitDepthLayer else cls(circuit_depth=2)
        rng = np.random.RandomState(6)
        layer(
            keras.ops.convert_to_tensor(
                rng.uniform(size=(3, 8)).astype("float32")
            )
        )
        with pytest.raises(ValueError, match="expects rank >= 2 input"):
            layer(
                keras.ops.convert_to_tensor(
                    rng.uniform(size=(8,)).astype("float32")
                )
            )

    def test_a_channel_mix_dense_also_pins_the_width(self):
        """channel_mix='dense' builds a Dense(C); global mode alone does
        not depend on the width, but that Dense does."""
        layer = CircuitDepthLayer(
            num_logic_ops=2, num_arithmetic_ops=2, channel_mix="dense"
        )
        rng = np.random.RandomState(7)
        layer(
            keras.ops.convert_to_tensor(
                rng.uniform(size=(3, 8)).astype("float32")
            )
        )
        with pytest.raises(ValueError, match="was built for a last axis of"):
            layer(
                keras.ops.convert_to_tensor(
                    rng.uniform(size=(3, 16)).astype("float32")
                )
            )


class TestB7TheRegisteredNamesArePackageQualified:
    """B7. Registration-key pin. No ``.keras`` archive in this
    repository contains any of the four classes (45 archives opened on
    2026-08-29, 41 of them under ``results/``, 0 hits), so the key change
    cannot orphan a checkpoint.

    UPDATED 2026-08-29 by the tree-wide registration migration (``MIGRATIONS.md``).
    This class originally asserted the OPPOSITE of what it asserts now for the
    legacy key: ``test_no_bare_custom_key_survives`` required
    ``Custom>{name}`` to be ABSENT. That was correct only while these four were
    the sole package-qualified sites in an otherwise bare tree -- the absence was
    evidence that the qualification had been applied. Every registered object in
    ``src/`` is now package-qualified AND carries a ``Custom>`` alias, granted
    uniformly by ``register_dl_technique`` so that pre-migration archives keep
    loading. Demanding the alias's absence here would be demanding that these four
    opt out of a tree-wide policy for no reason: nothing else in the tree registers
    any of these four names, so the alias cannot collide, and 0 archives name them
    so it cannot mislead a load either. What is still worth pinning -- and is
    pinned below -- is that the two keys are DISTINCT and resolve to the SAME
    object, which is exactly the property that makes the alias safe.
    """

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_the_class_is_registered_under_dl_techniques_layers(self, cls,
                                                                registration_contract):
        """Was ``test_no_bare_custom_key_survives`` plus an exact single-key pin.

        The exact package string is still asserted here -- these four are the one
        place in the tree where it is genuinely pinned rather than derived,
        because ``dl_techniques.layers`` was chosen by hand (B7) and not by the
        module-path rule.
        """
        registration_contract(cls, expected_package="dl_techniques.layers")

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_exactly_two_keys_and_no_third_claimant(self, cls):
        """The qualified key and its alias, and nothing else, resolve to ``cls``.

        `registration_contract` checks both keys individually; this checks the
        registry from the other direction, which is the only way a THIRD key
        pointing at the same class would show up.
        """
        registry = keras.saving.get_custom_objects()
        keys = sorted(k for k, v in registry.items() if v is cls)
        assert len(keys) == 2, keys
        assert keys[-1] == f"dl_techniques.layers>{cls.__name__}", keys

    def test_every_decorator_in_the_package_names_a_package(self):
        for path in SOURCE_FILES:
            text = path.read_text()
            assert "register_keras_serializable()" not in text
