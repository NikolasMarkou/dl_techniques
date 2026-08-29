"""Shared instrument for the ``layers/logic`` §16.3 test arms.

No ``test_`` prefix, so pytest does not collect this module (§13.7.1).
It owns three things the three §16.3 guard modules share:

1. ``SUBJECTS`` -- the four classes under test, each with a builder, its
   input arity, a fully-live variant (every trainable weight reachable)
   and a "this sub-layer is absent" variant for the §8.3 anti-vacuity
   sibling.
2. ``Parent`` / ``build_through_parent`` -- the §13.2.1 probe, the only
   path that builds a child through another layer's ``call()``.
3. ``FiniteForwardObserver`` -- the §16.3 "``ops.all(ops.isfinite(y))``
   in every forward test" rule, installed once for the whole directory
   by ``conftest.py`` instead of being copied into 83 test bodies.

RED proofs for everything defined here live in the mirrored
``test_*`` modules beside it.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.layers.logic.arithmetic_operators import (
    LearnableArithmeticOperator,
)
from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator
from dl_techniques.layers.logic.neural_circuit import (
    CircuitDepthLayer,
    LearnableNeuralCircuit,
)

#: The four classes the observer wraps and the guards drive.
OBSERVED_CLASSES = (
    LearnableLogicOperator,
    LearnableArithmeticOperator,
    CircuitDepthLayer,
    LearnableNeuralCircuit,
)

#: Feature shape used by every arm. Non-square on purpose: an 8x16 grid
#: fails a transposed stride that a square grid cannot see (§13.2.5).
FEATURE_SHAPE = (8, 16)

#: A non-degenerate initializer for the "every weight is live" variants.
#: The default ``operation_initializer='zeros'`` makes the softmax over
#: the gates exactly uniform, which makes d(loss)/d(temperature)
#: identically 0.0 -- a configuration in which the gradient guard would
#: be structurally unable to observe the temperature at all (§13.2.6).
LIVE_INIT = keras.initializers.RandomNormal(stddev=0.5, seed=11)


class Subject:
    """One class under test, plus every variant the arms need.

    :param name: Class name, used as the pytest parameter id.
    :param make: Builds the default-configuration layer.
    :param arity: Number of input tensors ``call`` takes.
    :param make_live: Builds a variant in which every trainable weight
        receives a non-zero gradient.
    :param absent_scope: A substring of ``w.path`` that ``make_absent``
        must produce no weight for.
    :param make_absent: Builds a variant configured so that the
        ``absent_scope`` sub-layer is never created.
    :param make_present: Builds the twin of ``make_absent``: the
        configuration that DOES create the ``absent_scope`` sub-layer.
        Without it, an "absent" assertion is also satisfied by a class
        that never builds that sub-layer under any configuration.
    """

    def __init__(
            self,
            name: str,
            make: Callable[[], keras.layers.Layer],
            arity: int,
            make_live: Callable[[], keras.layers.Layer],
            absent_scope: str,
            make_absent: Callable[[], keras.layers.Layer],
            make_present: Callable[[], keras.layers.Layer],
    ) -> None:
        self.name = name
        self.make = make
        self.arity = arity
        self.make_live = make_live
        self.absent_scope = absent_scope
        self.make_absent = make_absent
        self.make_present = make_present

    def input_shapes(self) -> List[Tuple[Optional[int], ...]]:
        """Batch-agnostic input shapes, one per operand."""
        return [(None,) + FEATURE_SHAPE for _ in range(self.arity)]

    def inputs(
            self,
            batch: int = 4,
            dtype: str = "float32",
            shape: Optional[Tuple[int, ...]] = None,
    ) -> Any:
        """A deterministic input in ``(0, 1)``, single or list.

        The gates expect inputs in the unit interval, so the draw is
        uniform on ``[0.05, 0.95]`` rather than normal. The seed is
        fixed, so the same call in two arms yields the same bits.
        """
        feature = FEATURE_SHAPE if shape is None else shape
        rng = np.random.default_rng(1234)
        drawn = [
            rng.uniform(0.05, 0.95, size=(batch,) + feature).astype(dtype)
            for _ in range(self.arity)
        ]
        return drawn if self.arity > 1 else drawn[0]

    def model(
            self,
            layer: Optional[keras.layers.Layer] = None,
            shape: Optional[Tuple[int, ...]] = None,
    ) -> keras.Model:
        """A functional model wrapping ``layer`` (default: ``make()``).

        Functional construction rather than ``layer.build(shape)``: a
        direct ``build`` call bypasses the ``StatelessScope`` path a real
        ``fit()`` takes (V-08).

        The model inherits the ambient global dtype policy. There is
        deliberately no ``dtype=`` override here: a per-layer ``dtype``
        does NOT reach the children ``CircuitDepthLayer`` builds in
        ``__init__`` (measured 2026-08-29), so such an override would
        produce a float32 "control" that silently ran two of the four
        subjects in float16. The float32 control is captured at module
        import instead, while the policy is still float32.

        :param layer: Layer to wrap; a fresh ``make()`` when ``None``.
        :param shape: Feature shape override, for the degenerate-length
            sweep.
        """
        feature = FEATURE_SHAPE if shape is None else shape
        built = self.make() if layer is None else layer
        symbolic = [
            keras.Input(shape=feature) for _ in range(self.arity)
        ]
        operand = symbolic if self.arity > 1 else symbolic[0]
        return keras.Model(operand, built(operand))


def _logic() -> LearnableLogicOperator:
    return LearnableLogicOperator(
        operation_types=["and", "or", "xor"], name="unit"
    )


def _logic_live() -> LearnableLogicOperator:
    return LearnableLogicOperator(
        operation_types=["and", "or", "xor"],
        operation_initializer=LIVE_INIT,
        name="unit",
    )


def _logic_absent() -> LearnableLogicOperator:
    return LearnableLogicOperator(
        operation_types=["and", "or", "xor"],
        use_temperature=False,
        name="unit",
    )


def _arith() -> LearnableArithmeticOperator:
    return LearnableArithmeticOperator(
        operation_types=["add", "multiply"], name="unit"
    )


def _arith_live() -> LearnableArithmeticOperator:
    return LearnableArithmeticOperator(
        operation_types=["add", "multiply"],
        operation_initializer=LIVE_INIT,
        name="unit",
    )


def _arith_absent() -> LearnableArithmeticOperator:
    return LearnableArithmeticOperator(
        operation_types=["add", "multiply"],
        use_scaling=False,
        name="unit",
    )


def _depth() -> CircuitDepthLayer:
    return CircuitDepthLayer(
        num_logic_ops=2, num_arithmetic_ops=1, name="unit"
    )


def _depth_live() -> CircuitDepthLayer:
    # circuit_routing='classic' is required: under the default
    # 'output_only' the routing weights are created but never read, so
    # their gradient is None by construction (see the source comment at
    # neural_circuit.CircuitDepthLayer.build).
    return CircuitDepthLayer(
        num_logic_ops=2,
        num_arithmetic_ops=1,
        circuit_routing="classic",
        routing_initializer=LIVE_INIT,
        combination_initializer=LIVE_INIT,
        inner_logic_kwargs={"operation_initializer": LIVE_INIT},
        inner_arithmetic_kwargs={"operation_initializer": LIVE_INIT},
        name="unit",
    )


def _depth_absent() -> CircuitDepthLayer:
    # channel_mix=None is the DEFAULT, so this variant is also the
    # default configuration; its twin below has to opt in.
    return CircuitDepthLayer(
        num_logic_ops=2,
        num_arithmetic_ops=1,
        channel_mix=None,
        name="unit",
    )


def _depth_present() -> CircuitDepthLayer:
    return CircuitDepthLayer(
        num_logic_ops=2,
        num_arithmetic_ops=1,
        channel_mix="dense",
        name="unit",
    )


def _circuit() -> LearnableNeuralCircuit:
    return LearnableNeuralCircuit(
        circuit_depth=2, use_layer_norm=True, name="unit"
    )


def _circuit_live() -> LearnableNeuralCircuit:
    return LearnableNeuralCircuit(
        circuit_depth=2,
        use_layer_norm=True,
        circuit_routing="classic",
        routing_initializer=LIVE_INIT,
        combination_initializer=LIVE_INIT,
        inner_logic_kwargs={"operation_initializer": LIVE_INIT},
        inner_arithmetic_kwargs={"operation_initializer": LIVE_INIT},
        name="unit",
    )


def _circuit_absent() -> LearnableNeuralCircuit:
    return LearnableNeuralCircuit(
        circuit_depth=2, use_layer_norm=False, name="unit"
    )


SUBJECTS: Dict[str, Subject] = {
    s.name: s
    for s in (
        Subject(
            "LearnableLogicOperator", _logic, 2, _logic_live,
            "temperature", _logic_absent, _logic,
        ),
        Subject(
            "LearnableArithmeticOperator", _arith, 2, _arith_live,
            "scaling_factor", _arith_absent, _arith,
        ),
        Subject(
            "CircuitDepthLayer", _depth, 1, _depth_live,
            "channel_mix", _depth_absent, _depth_present,
        ),
        Subject(
            "LearnableNeuralCircuit", _circuit, 1, _circuit_live,
            "layer_norm", _circuit_absent, _circuit,
        ),
    )
}

#: pytest parameter ids, in a fixed order.
SUBJECT_NAMES = tuple(SUBJECTS)


def relative_weight_paths(container: Any) -> List[str]:
    """Weight paths with the root segment stripped, sorted.

    Stripping the root is what lets two separately-constructed
    instances compare equal (§8.3). It only works because every
    sub-layer in this package carries an explicit ``name=``; without
    that, Keras auto-increments and two instances read ``block/w``
    versus ``block_1/w`` at every unnamed level.
    """
    return sorted(w.path.split("/", 1)[-1] for w in container.weights)


def weight_paths_below(layer: Any) -> List[str]:
    """Weight paths relative to ``layer`` itself, sorted.

    ``relative_weight_paths`` strips exactly one root segment, which is
    right for two models but wrong for a layer built inside a parent:
    that one carries an extra level (``parent/unit/w`` versus
    ``unit/w``). This strips everything up to and including the
    layer's own name instead.
    """
    marker = f"{layer.name}/"
    stripped = []
    for weight in layer.weights:
        index = weight.path.find(marker)
        stripped.append(
            weight.path[index + len(marker):]
            if index >= 0 else weight.path
        )
    return sorted(stripped)


class Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` is the only path that builds the
    child (§13.2.1). A direct ``child.build(shape)`` runs outside the
    ``StatelessScope`` and therefore cannot see the trap it exists to
    catch.

    :param child: The layer to build and run.
    """

    def __init__(self, child: keras.layers.Layer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs: Any) -> Any:
        """Delegate straight to the child."""
        return self.child(inputs)


def build_through_parent(
        child: keras.layers.Layer, subject: Subject
) -> keras.layers.Layer:
    """Build ``child`` only via a parent's ``call()``. Returns ``child``.

    :param child: An unbuilt layer.
    :param subject: Supplies the input arity and feature shape.
    :return: The same object, now built.
    """
    parent = Parent(child)
    symbolic = [
        keras.Input(shape=FEATURE_SHAPE) for _ in range(subject.arity)
    ]
    parent(symbolic if subject.arity > 1 else symbolic[0])
    return child


def is_concrete_tensor(value: Any) -> bool:
    """True only for a tensor whose values exist right now.

    A ``KerasTensor`` (functional graph) and a ``tf`` graph tensor
    (inside a ``tf.function`` trace) both carry no values, so reading
    them would raise rather than report a defect.
    """
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, keras.KerasTensor):
        return False
    if not tf.is_tensor(value):
        return False
    return not tf.is_symbolic_tensor(value)


class FiniteForwardObserver:
    """Assert ``ops.all(ops.isfinite(y))`` on every concrete forward
    output of the four classes, for the whole directory.

    This is the §16.3 "in every forward test" item, implemented once as
    an instrument rather than as a line copied into each test body: a
    line copied into 83 bodies is silently absent from the 84th.

    Two counters make the instrument non-vacuous. ``concrete`` counts
    the outputs actually checked; ``symbolic`` counts the ones skipped
    because they carry no values. A test that asserts finiteness while
    ``concrete`` stayed 0 has asserted nothing, and the meta-tests in
    ``test_every_weight_is_reached_through_a_parent.py`` pin both.
    """

    def __init__(self) -> None:
        self.concrete = 0
        self.symbolic = 0
        self._originals: Dict[type, Callable[..., Any]] = {}

    def _wrap(
            self, cls: type, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        observer = self

        @functools.wraps(original)
        def wrapper(layer: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(layer, *args, **kwargs)
            if not is_concrete_tensor(result):
                observer.symbolic += 1
                return result
            observer.concrete += 1
            if not bool(keras.ops.all(keras.ops.isfinite(result))):
                raise AssertionError(
                    f"{cls.__name__}.call produced a non-finite output "
                    f"(dtype {result.dtype}); §16.3 requires every "
                    f"forward pass in this directory to be finite"
                )
            return result

        return wrapper

    def install(self) -> None:
        """Wrap ``call`` on all four classes."""
        for cls in OBSERVED_CLASSES:
            original = cls.call
            self._originals[cls] = original
            cls.call = self._wrap(cls, original)

    def uninstall(self) -> None:
        """Restore every wrapped ``call`` and assert the restoration."""
        for cls, original in self._originals.items():
            cls.call = original
        self._originals.clear()
        for cls in OBSERVED_CLASSES:
            assert not hasattr(cls.call, "__wrapped__"), (
                f"{cls.__name__}.call was left wrapped by the finiteness "
                f"observer; the next test would inherit the instrument"
            )
