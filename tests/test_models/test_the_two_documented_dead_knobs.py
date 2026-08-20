"""Guard: two knobs that are inert, pinned as inert rather than "fixed".

Plan ``plan-2026-08-19T163559-499b6f0e``, step 17.1. Rules ``R-119`` / ``R-120``
(``coshnet``) and ``R-121`` (``kan``).

Both rows are terminal as **documented-default**, and both are pinned here so
that a later sweep cannot quietly change the ruling in either direction.

``coshnet`` / ``ComplexLayer.epsilon``
    A public, validated, serialized constructor argument read by NO computation.
    MEASURED under a same-weights protocol (build A and B at one seed, copy A's
    weights onto B, compare outputs): ``1e-7`` vs ``1e+3`` and ``1e-7`` vs
    ``1e-30`` both move the output by exactly ``0.000000e+00``, while three
    other knobs of the same class move it (``dropout_rate`` ``3.54e-01``,
    ``conv_filters`` ``2.48e-01`` plus a parameter-count change, ``include_top``
    structurally). It is NOT wired up — there is no division in the module for it
    to guard — and it is NOT removed, because ``from_config`` on every saved
    complex layer passes ``epsilon=``.

``kan`` / an unadapted knot grid
    ``create_kan_model`` at the documented defaults returns a CONSTANT FUNCTION
    (output exactly ``1/output_features``, ``std == 0.0``, 0 of 12 live
    gradients). That state is already pinned by
    ``test_models/test_kan/test_model.py``'s ``xfail(strict=True)`` pair and is
    **not re-litigated**. What is asserted here is the new, inspectable state
    that makes the condition readable instead of silent.

See ``decisions.md`` D-052, D-053.
"""

import ast
import inspect

import numpy as np
import pytest
import keras

from dl_techniques.layers import complex_layers
from dl_techniques.layers.complex_layers import ComplexLayer
from dl_techniques.models.coshnet.model import CoShNet
from dl_techniques.models.kan import KAN
from dl_techniques.models.kan.model import create_kan_model


# ---------------------------------------------------------------------------
# coshnet — the epsilon knob is inert, and provably so
# ---------------------------------------------------------------------------

def test_epsilon_is_read_by_exactly_two_ast_nodes_and_neither_computes():
    """The mechanism, asserted rather than described.

    `self.epsilon` appears at exactly two places in the module's AST: the
    assignment in `__init__` and the entry in `get_config`. If a third ever
    appears, the knob has acquired a consumer and this ruling must be revisited
    — which is the point of pinning the count.

    The predicate is AST, deliberately, and the first draft of this test was a
    `source.count("self.epsilon")` that read **4** — because the DECISION
    comment placed at the site names the attribute twice. A text count cannot
    tell a consumer from a comment about the absence of consumers.
    """
    tree = ast.parse(inspect.getsource(complex_layers))
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "epsilon"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]
    assert len(nodes) == 2, (
        f"`self.epsilon` now appears at {len(nodes)} AST sites in "
        "complex_layers.py (expected exactly 2: the __init__ assignment and "
        "the get_config entry). A new site means the knob is no longer inert."
    )
    # One is a Store (the assignment), one is a Load (the get_config read).
    contexts = sorted(type(node.ctx).__name__ for node in nodes)
    assert contexts == ["Load", "Store"], (
        f"expected one Store and one Load, got {contexts} — a second Load is a "
        "computation reading the knob"
    )


def test_epsilon_is_still_validated_and_serialized():
    """Inert is not the same as absent. The contract callers rely on stands."""
    layer = ComplexLayer(epsilon=1e-5)
    assert layer.epsilon == 1e-5
    assert layer.get_config()["epsilon"] == 1e-5
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ComplexLayer(epsilon=0.0)


def _coshnet_output(epsilon, reference=None):
    """Build a CoShNet at `epsilon`, optionally copying `reference`'s weights."""
    keras.utils.set_random_seed(1234)
    model = CoShNet(
        num_classes=4, input_shape=(32, 48, 1), conv_filters=[8, 16],
        dense_units=[16], dropout_rate=0.0, epsilon=epsilon,
    )
    x = np.random.RandomState(0).rand(2, 32, 48, 1).astype("float32")
    model(x, training=False)
    if reference is not None:
        # SAME-WEIGHTS protocol. Rebuilding under a seed does NOT reproduce the
        # weights here (measured: a seeded rebuild reported a spurious 0.023),
        # so the only valid comparison copies them across.
        for dst, src in zip(model.weights, reference.weights):
            dst.assign(src)
    return model, np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))


@pytest.mark.parametrize("other", [1e3, 1e-30])
def test_epsilon_moves_the_coshnet_output_by_exactly_zero(other):
    reference, baseline = _coshnet_output(1e-7)
    _, moved = _coshnet_output(other, reference=reference)
    assert float(np.max(np.abs(baseline - moved))) == 0.0, (
        f"epsilon 1e-7 vs {other} moved the output — the knob is no longer dead "
        "and decisions.md D-053 must be revisited"
    )


def test_a_live_knob_on_the_same_model_does_move_it():
    """LIVENESS. Without this arm, a broken protocol reads as 'all knobs dead'."""
    reference, baseline = _coshnet_output(1e-7)
    keras.utils.set_random_seed(1234)
    other = CoShNet(
        num_classes=4, input_shape=(32, 48, 1), conv_filters=[4, 4],
        dense_units=[16], dropout_rate=0.0, epsilon=1e-7,
    )
    x = np.random.RandomState(0).rand(2, 32, 48, 1).astype("float32")
    moved = np.asarray(keras.ops.convert_to_numpy(other(x, training=False)))
    assert float(np.max(np.abs(baseline - moved))) > 0.0
    assert other.count_params() != reference.count_params()


# ---------------------------------------------------------------------------
# kan — the untrainable default is now readable from the object
# ---------------------------------------------------------------------------

def test_a_fresh_kan_reports_its_grids_as_unadapted():
    model = create_kan_model(variant="micro", input_features=10, output_features=4)
    assert model.grids_adapted is False


def test_update_kan_grids_flips_the_flag():
    """ANTI-VACUITY: a flag that is always False would pass the test above."""
    model = create_kan_model(variant="micro", input_features=10, output_features=4)
    model.update_kan_grids(
        np.random.RandomState(0).randn(64, 10).astype("float32"))
    assert model.grids_adapted is True


def test_the_flag_is_not_set_by_a_plain_forward_pass():
    """A forward pass is not a grid pass, and must not claim to be one."""
    model = create_kan_model(variant="micro", input_features=10, output_features=4)
    model(np.random.RandomState(0).randn(4, 10).astype("float32"), training=False)
    assert model.grids_adapted is False


def test_the_direct_constructor_also_reports_unadapted_grids():
    model = KAN(
        layer_configs=[
            {"features": 8, "grid_size": 5, "activation": "swish"},
            {"features": 4, "grid_size": 4, "activation": "gelu"},
        ],
        input_features=10,
    )
    assert model.grids_adapted is False


def test_the_factory_docstring_states_the_measured_consequence():
    """The warning is part of the public contract, not a comment.

    A caller who reads only `create_kan_model`'s docstring must learn that the
    returned model cannot be trained as-is; that sentence is what this row
    ships, so it is asserted rather than trusted.
    """
    doc = create_kan_model.__doc__
    assert "update_kan_grids" in doc
    assert "cannot be trained as-is" in doc
    assert "grids_adapted" in doc
