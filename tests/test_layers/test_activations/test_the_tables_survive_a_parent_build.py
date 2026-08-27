"""Regression barrier: every `build()` in this package survives being run through a PARENT.

**What this is, honestly.** It is NOT a bug hunt. Measured for this plan (finding G-6):
`grep -c '\\.assign(' src/dl_techniques/layers/activations/*.py` returns **0** across the
whole package -- there is not one live `add_weight(...)` + `.assign(...)`-in-`build()` target
here today, so this module finds nothing at HEAD and is expected to. It exists because the
instrument was missing: **0 of the 17 pre-existing test files in this directory ever built
one of these layers through an enclosing layer**, and a direct `layer.build(shape)` call in a
unit test is *structurally blind* to the failure this module is for (guide rule L-34).

**The failure it stands guard over.** Keras 3 runs `build()` under a
`keras.src.backend.StatelessScope` whenever the layer is built as part of an enclosing
layer's call -- which is how these layers are actually used. Inside that scope a
`variable.assign(...)` is recorded into the scope and then DISCARDED, so a table written
as::

    self.table = self.add_weight(..., initializer='zeros')   # in build()
    self.table.assign(precomputed_values)                    # silently dropped

leaves the model running on ALL ZEROS in every real composition while every direct-`build()`
unit test in the suite stays green. This repo has measured that exact defect at 11 separate
sites elsewhere in the tree, where it left a trend-only N-BEATS predicting EXACTLY ZERO.

**How it is kept honest.**

* The class list is **re-derived by AST at import time** from the source directory, not
  copied into this file. A tenth class that defines `build()` is picked up automatically; if
  the AST walk finds nothing, the parametrization is empty and `test_the_class_set_is_not_empty`
  fails rather than the module passing vacuously. (An inherited count is a floor to re-derive,
  not a ceiling to trust: the count for this package was carried as 8 for most of this plan
  and is actually **9**.)
* Every layer is constructed with an initializer that is **deliberately not zeros**, so
  "the weight is all zeros" is a real signal. With the shipped defaults, `ExpandedActivation.alpha`
  and `RoutingProbabilitiesLayer.bias` both initialize to zeros, and a degeneracy assertion
  against them would be unfailable.
* The comparison is PARENT-BUILT vs DIRECT-BUILT, which is the pairing measured to
  discriminate. Both composed routes -- symbolic `keras.Input` and eager
  `_Parent(child)(tensor)` -- drop an `.assign()` together (measured 0.0 and 0.0 against an
  injected 7.0), so a symbolic-vs-eager comparison would be blind; only the direct
  `child.build(shape)` keeps the assigned value, which is precisely why the direct build
  every other suite here uses is the route that looks healthy.

RED-proven by injecting an `add_weight('zeros') + .assign()` shape into
`DifferentiableStep.build()`; see this plan's `evidence/iter2-step-12-red-proof.txt`.
"""

import ast
import os

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from dl_techniques.layers.activations.differentiable_step import DifferentiableStep
from dl_techniques.layers.activations.expanded_activations import xGELU
from dl_techniques.layers.activations.hard_sigmoid import HardSigmoid
from dl_techniques.layers.activations.hard_swish import HardSwish
from dl_techniques.layers.activations.monotonicity_layer import MonotonicityLayer
from dl_techniques.layers.activations.probability_output import ProbabilityOutput
from dl_techniques.layers.activations.routing_probabilities import (
    RoutingProbabilitiesLayer,
)
from dl_techniques.layers.activations.thresh_max import ThreshMax

_PACKAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "src", "dl_techniques", "layers", "activations",
)

BATCH = 4
WIDTH = 8


def _classes_defining_build() -> list:
    """AST-walk the package and return every class that defines its own `build()`.

    Re-derived at import time rather than hard-coded: the whole point of this
    module is that a NEW `build()` cannot appear without a barrier.

    :return: sorted ``(module_filename, class_name)`` pairs.
    :rtype: list[tuple[str, str]]
    """
    found = []
    for filename in sorted(os.listdir(_PACKAGE_DIR)):
        if not filename.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_PACKAGE_DIR, filename)).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                isinstance(m, ast.FunctionDef) and m.name == "build"
                for m in node.body
            ):
                found.append((filename, node.name))
    return sorted(found)


CLASSES_DEFINING_BUILD = _classes_defining_build()

# Constructor recipes. The KEY is the AST-derived class name; the VALUE builds a
# concrete instance. Every recipe deliberately asks for a NON-ZERO initializer
# wherever the shipped default is `'zeros'`, so the degeneracy assertion below is
# not unfailable. `ExpandedActivation` is abstract -- it defines `build()` and
# `get_config()` but no `call()` -- so its `build()` is exercised through `xGELU`,
# which inherits it verbatim; that substitution is asserted, not assumed.
_RECIPES = {
    "AdaptiveTemperatureSoftmax": lambda: AdaptiveTemperatureSoftmax(),
    "DifferentiableStep": lambda: DifferentiableStep(
        slope_initializer=keras.initializers.Constant(2.5),
        shift_initializer=keras.initializers.Constant(0.75),
    ),
    "ExpandedActivation": lambda: xGELU(
        alpha_initializer=keras.initializers.Constant(0.25)
    ),
    "HardSigmoid": lambda: HardSigmoid(),
    "HardSwish": lambda: HardSwish(),
    "MonotonicityLayer": lambda: MonotonicityLayer(method="cumulative_softplus"),
    "ProbabilityOutput": lambda: ProbabilityOutput(probability_type="softmax"),
    "RoutingProbabilitiesLayer": lambda: RoutingProbabilitiesLayer(
        output_dim=8,
        mode="trainable",
        bias_initializer=keras.initializers.Constant(0.5),
    ),
    "ThreshMax": lambda: ThreshMax(
        trainable_slope=True,
        slope_initializer=keras.initializers.Constant(3.0),
    ),
}


class _Parent(keras.layers.Layer):
    """Minimal enclosing layer whose `call()` invokes the child.

    Test-local, not a shared abstraction: its entire purpose is to move the
    child's `build()` off the direct `layer.build(shape)` path that every other
    test in this directory uses and onto the composed path Keras really takes.

    :param child: the layer under test, already constructed.
    :type child: keras.layers.Layer
    """

    def __init__(self, child: keras.layers.Layer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        """Forward straight through to the child.

        :param inputs: the input tensor.
        :return: the child's output.
        """
        return self.child(inputs)


def _weight_snapshot(layer: keras.layers.Layer) -> dict:
    """Map every weight of ``layer`` to its value as a float64 NumPy array.

    Uses ``layer.weights`` (which includes non-trainable variables) rather than
    ``get_weights()``: this repo has measured a four-order-of-magnitude defect
    that ``get_weights()`` hid and ``trainable_weights`` exposed.

    :param layer: a built layer.
    :type layer: keras.layers.Layer
    :return: ``{weight.path: value}``.
    :rtype: dict[str, numpy.ndarray]
    """
    return {
        w.path: np.asarray(keras.ops.convert_to_numpy(w.value), dtype=np.float64)
        for w in layer.weights
    }


class TestTheTablesSurviveAParentBuild:
    """One case per `build()`-defining class, plus two non-vacuity guards."""

    def test_the_class_set_is_not_empty(self) -> None:
        """The AST walk found classes, and every one of them has a recipe.

        Without this the module can pass by finding nothing -- the exact shape of
        a barrier that silently stops being a barrier.
        """
        assert CLASSES_DEFINING_BUILD, (
            f"the AST walk over {_PACKAGE_DIR} found NO class defining build(); "
            f"every case below would be silently skipped"
        )
        names = {name for _, name in CLASSES_DEFINING_BUILD}
        missing = sorted(names - set(_RECIPES))
        assert not missing, (
            f"these classes define build() but have no construction recipe in "
            f"this module, so they are UNCOVERED: {missing}"
        )
        stale = sorted(set(_RECIPES) - names)
        assert not stale, (
            f"these recipes name classes that no longer define build(): {stale}"
        )

    @pytest.mark.parametrize(
        "class_name",
        [name for _, name in CLASSES_DEFINING_BUILD],
    )
    def test_weights_survive_a_symbolic_parent_build(self, class_name: str) -> None:
        """Build through a parent on the SYMBOLIC path and check the weights.

        `keras.Input` -> `_Parent(child)` -> `keras.Model` is the composition that
        runs `build()` inside a `StatelessScope`. Any value written with
        `.assign()` in `build()` is discarded there, leaving the weight at its
        initializer -- which for the shipped code would be all zeros.
        """
        keras.utils.set_random_seed(0)
        child = _RECIPES[class_name]()

        inputs = keras.Input(shape=(WIDTH,), batch_size=BATCH)
        outputs = _Parent(child)(inputs)
        model = keras.Model(inputs, outputs)

        assert child.built, (
            f"{class_name} was never built by the parent composition, so this "
            f"case asserts nothing"
        )

        symbolic = _weight_snapshot(child)
        for path, value in symbolic.items():
            assert np.all(np.isfinite(value)), (
                f"{class_name} weight {path!r} is non-finite after a symbolic "
                f"parent build"
            )
            assert not np.all(value == 0.0), (
                f"{class_name} weight {path!r} is ENTIRELY ZERO after being built "
                f"through a parent, while its recipe asked for a non-zero "
                f"initializer. This is the StatelessScope signature: a value "
                f"written with `.assign()` inside `build()` is recorded into the "
                f"scope and dropped, and the layer runs on the initializer."
            )

        # The composition must also actually RUN -- a barrier that only inspects
        # variables would miss a build that produced unusable weights.
        x = np.random.default_rng(0).standard_normal((BATCH, WIDTH)).astype("float32")
        y = np.asarray(keras.ops.convert_to_numpy(model(x)), dtype=np.float64)
        assert np.all(np.isfinite(y)), (
            f"{class_name} produced non-finite output through the parent"
        )

    @pytest.mark.parametrize(
        "class_name",
        [name for _, name in CLASSES_DEFINING_BUILD],
    )
    def test_a_parent_build_agrees_with_a_direct_build(self, class_name: str) -> None:
        """A parent-composed build must produce the SAME weight values as `layer.build()`.

        This is the discriminating half, and the comparison it makes is the one
        MEASURED to work rather than the one that sounded right. On Keras 3.8 an
        `.assign()` inside `build()` is dropped on BOTH composed routes -- the
        symbolic `keras.Input` route and the eager `_Parent(child)(tensor)` route
        -- and survives ONLY on a direct `child.build(shape)` call. Measured on an
        injected `add_weight('zeros') + assign(7.0)`: symbolic 0.0, eager 0.0,
        direct 7.0.

        So a symbolic-vs-eager comparison cannot see this failure at all (both
        arms are wrong together), while direct-vs-parent sees it immediately.
        That asymmetry IS guide rule L-34's point restated as a measurement: the
        direct `build()` every other test in this directory uses is the one route
        that looks healthy.
        """
        keras.utils.set_random_seed(0)
        child_parent = _RECIPES[class_name]()
        x = np.random.default_rng(0).standard_normal((BATCH, WIDTH)).astype("float32")
        _Parent(child_parent)(keras.ops.convert_to_tensor(x))

        keras.utils.set_random_seed(0)
        child_direct = _RECIPES[class_name]()
        child_direct.build((BATCH, WIDTH))

        via_parent = _weight_snapshot(child_parent)
        via_direct = _weight_snapshot(child_direct)

        parent_names = sorted(p.rsplit("/", 1)[-1] for p in via_parent)
        direct_names = sorted(p.rsplit("/", 1)[-1] for p in via_direct)
        assert parent_names == direct_names, (
            f"{class_name} created different weights when built through a parent "
            f"than when built directly: {parent_names} vs {direct_names}"
        )

        for (p_path, p_val), (d_path, d_val) in zip(
            sorted(via_parent.items(), key=lambda kv: kv[0].rsplit("/", 1)[-1]),
            sorted(via_direct.items(), key=lambda kv: kv[0].rsplit("/", 1)[-1]),
        ):
            assert p_val.shape == d_val.shape, (
                f"{class_name} weight {p_path!r} has shape {p_val.shape} through "
                f"a parent and {d_val.shape} built directly"
            )
            dev = float(np.max(np.abs(p_val - d_val))) if p_val.size else 0.0
            assert dev == 0.0, (
                f"{class_name} weight {p_path!r} differs by {dev:.6e} between a "
                f"parent-composed build and a direct build. A value written with "
                f"`.assign()` inside `build()` is discarded under the composed "
                f"routes and kept under the direct one -- that is exactly this "
                f"shape, and the direct build is the arm that looks correct."
            )
