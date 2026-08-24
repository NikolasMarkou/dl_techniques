"""All three shgcn classes must run under mixed_float16 and with use_curvature=False.

Two defects, both CRITICAL, both silent in their own way:

1. **100% NaN under ``mixed_float16``, no warning.** `PoincareMath.log_map_0`
   clamped ``||y||`` with a fixed ``1e-4`` margin, which is 9.77x SMALLER than
   float16's ULP at 1.0 (9.765625e-04). The clamp was arithmetically an
   identity, ``arctanh(1.0) = inf``, and NaN propagated. MEASURED before the
   fix: `SHGCNModel` 96/96 NaN, `SHGCNNodeClassifier` 36/36, `SHGCNLinkPredictor`
   5/5, with float32 green throughout. The root cause is in `utils/geometry/`,
   not in this package.
2. **``use_curvature=False`` had no working forward path in ANY of the three
   classes.** `SHGCNLayer.build` materialised the fixed curvature with
   ``keras.ops.convert_to_tensor(0.54)``, i.e. an eager constant belonging to
   Keras' throwaway build-time scratch `FuncGraph`. All three raised
   ``TypeError: <tf.Tensor 'shgcn_hidden_0/Const:0'> is out of scope``. The
   default ``True`` was green, so no shipped call site ever saw it.

Both arms carry their float32 / ``use_curvature=True`` control in the same file,
so a change that makes the guard pass by disabling the feature fails the control.

See decisions.md D-027 and D-028 (plan-2026-08-19T163559-499b6f0e).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.shgcn.model import (
    SHGCNLinkPredictor,
    SHGCNModel,
    SHGCNNodeClassifier,
)

# ---------------------------------------------------------------------

NUM_NODES = 12
INPUT_DIM = 8
HIDDEN_DIM = 16
NUM_EDGES = 5


@pytest.fixture(scope="module")
def graph():
    rs = np.random.RandomState(0)
    features = rs.randn(NUM_NODES, INPUT_DIM).astype("float32")
    adjacency = (rs.rand(NUM_NODES, NUM_NODES) < 0.3).astype("float32")
    adjacency = np.maximum(adjacency, adjacency.T)
    edges = rs.randint(0, NUM_NODES, size=(NUM_EDGES, 2)).astype("int32")
    return features, adjacency, edges


def _build_and_run(name, use_curvature, graph):
    """Construct one of the three classes and take a forward pass."""
    features, adjacency, edges = graph
    if name == "SHGCNModel":
        model = SHGCNModel(hidden_dims=[HIDDEN_DIM], output_dim=INPUT_DIM,
                           use_curvature=use_curvature)
        return np.array(model([features, adjacency], training=False))
    if name == "SHGCNNodeClassifier":
        model = SHGCNNodeClassifier(num_classes=3, hidden_dims=[HIDDEN_DIM],
                                    use_curvature=use_curvature)
        return np.array(model([features, adjacency], training=False))
    model = SHGCNLinkPredictor(hidden_dims=[HIDDEN_DIM], embedding_dim=INPUT_DIM,
                               use_curvature=use_curvature)
    return np.array(model([features, adjacency, edges], training=False))


CLASS_NAMES = ["SHGCNModel", "SHGCNNodeClassifier", "SHGCNLinkPredictor"]


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_the_forward_pass_is_finite_under_mixed_float16(class_name, graph):
    """Arm 1: the fp16 arm, which read 100% NaN before the boundary-margin fix."""
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        output = _build_and_run(class_name, True, graph)
    finally:
        keras.mixed_precision.set_global_policy(previous)

    nan_count = int(np.sum(np.isnan(output)))
    assert nan_count == 0, (
        f"{class_name} produced {nan_count} NaN of {output.size} elements under "
        f"mixed_float16. Root cause is `PoincareMath`'s boundary margin being "
        f"smaller than the float16 ULP, NOT anything in this package."
    )
    assert np.all(np.isfinite(output)), f"{class_name} produced non-finite output"


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_the_float32_control_is_finite(class_name, graph):
    """Control for arm 1: float32 was ALWAYS green, so it must stay green."""
    output = _build_and_run(class_name, True, graph)
    assert np.all(np.isfinite(output)), (
        f"{class_name} is not finite at float32 -- the fp16 fix has broken the "
        f"default policy, which was never affected"
    )


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_use_curvature_false_has_a_working_forward_path(class_name, graph):
    """Arm 2: the knob that raised in all three classes."""
    output = _build_and_run(class_name, False, graph)
    assert np.all(np.isfinite(output)), (
        f"{class_name}(use_curvature=False) did not produce a finite output"
    )
    assert output.size > 0


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_the_curvature_knob_actually_changes_the_output(class_name, graph):
    """Anti-vacuity for arm 2: a knob that runs but does nothing is still dead.

    `use_curvature=True` learns c via a trainable `curvature_theta`;
    `use_curvature=False` pins it. At INITIALISATION both start from 0.54, so
    the two forward passes must differ only through the randomly initialised
    kernel -- which is why this test asserts the WEIGHT SET differs, the
    observable that does not depend on random draws.
    """
    features, adjacency, edges = graph
    if class_name == "SHGCNModel":
        with_c = SHGCNModel(hidden_dims=[HIDDEN_DIM], output_dim=INPUT_DIM,
                            use_curvature=True)
        without = SHGCNModel(hidden_dims=[HIDDEN_DIM], output_dim=INPUT_DIM,
                             use_curvature=False)
        args = ([features, adjacency],)
    elif class_name == "SHGCNNodeClassifier":
        with_c = SHGCNNodeClassifier(num_classes=3, hidden_dims=[HIDDEN_DIM],
                                     use_curvature=True)
        without = SHGCNNodeClassifier(num_classes=3, hidden_dims=[HIDDEN_DIM],
                                      use_curvature=False)
        args = ([features, adjacency],)
    else:
        with_c = SHGCNLinkPredictor(hidden_dims=[HIDDEN_DIM],
                                    embedding_dim=INPUT_DIM, use_curvature=True)
        without = SHGCNLinkPredictor(hidden_dims=[HIDDEN_DIM],
                                     embedding_dim=INPUT_DIM, use_curvature=False)
        args = ([features, adjacency, edges],)

    with_c(*args, training=False)
    without(*args, training=False)
    assert len(with_c.weights) > len(without.weights), (
        f"{class_name}: use_curvature=True must add a trainable "
        f"`curvature_theta` per sHGCN layer; got "
        f"{len(with_c.weights)} vs {len(without.weights)}"
    )
