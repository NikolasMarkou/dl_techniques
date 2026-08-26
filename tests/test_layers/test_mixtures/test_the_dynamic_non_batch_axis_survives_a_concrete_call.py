"""A layer built against a dynamic non-batch axis must survive a concrete call.

Single-claim guard (plan-2026-08-26T061816-c515641a, E1 / F-1).

**The claim.** ``GMMLayer`` / ``KMeansLayer`` built once via
``keras.Input(shape=(None, 8))`` and then *reused* on a concrete eager batch must
return the declared shape carrying the declared VALUES.

**Why the reuse matters.** A layer built directly on concrete data captures
concrete ints in ``original_shape`` and does not reproduce the defect at all. The
failure needs one instance that was built symbolically (``original_shape ==
[None, None, 8]``) and is then called on real data -- i.e. ordinary
``model.fit`` / ``model.predict`` usage for any variable-length sequence model.
An "is it built and called in one go" repro would falsely refute the bug.

**Why the assertions are on values, not shape.** ``base.py:180-203`` records a
previously-shipped bug where a bare reshape stamped a correct SHAPE onto a
wrongly-ordered buffer; every pre-existing test missed it because they used
symmetric dimensions. Every fixture here therefore has NO two dimension lengths
equal -- batch 2, sequence 5, features 8, prototypes 3 -- and every assertion
compares element-wise against ``cluster_axis_oracle``, which computes no
permutation at all.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest


from dl_techniques.layers.mixtures.gmm import GMMLayer
from dl_techniques.layers.mixtures.kmeans import KMeansLayer

from .cluster_axis_oracle import build_cluster_axis_oracle, flat_twin_forward

# Mutually distinct on purpose: batch != seq != features != prototypes. A
# symmetric fixture degenerates the layout defect into a pure transposition.
BATCH, SEQ, FEATURES, PROTOTYPES = 2, 5, 8, 3


def _gmm_case(output_mode: str) -> Tuple[type, Dict[str, Any], Tuple[str, ...]]:
    config = {
        "n_components": PROTOTYPES,
        "temperature": 1.0,
        "isometric_regularizer_strength": 0.0,
        "output_mode": output_mode,
        "covariance_type": "diagonal",
        "mean_initializer": keras.initializers.GlorotNormal(seed=11),
        "log_variance_initializer": keras.initializers.GlorotNormal(seed=12),
    }
    return GMMLayer, config, ("means", "log_variances", "mixture_logits")


def _kmeans_case(output_mode: str) -> Tuple[type, Dict[str, Any], Tuple[str, ...]]:
    config = {
        "n_clusters": PROTOTYPES,
        "temperature": 0.5,
        "output_mode": output_mode,
        "random_seed": 42,
    }
    return KMeansLayer, config, ("centroids",)


CASES = {"gmm": _gmm_case, "kmeans": _kmeans_case}


def _expected(
    x: np.ndarray,
    layer_cls: type,
    config: Dict[str, Any],
    param_names: Tuple[str, ...],
    layer: Any,
    output_mode: str,
) -> np.ndarray:
    """Independent per-slice reference for the declared output layout."""
    return build_cluster_axis_oracle(
        x,
        [-1],
        output_mode,
        PROTOTYPES,
        flat_twin_forward(layer_cls, config, param_names, layer),
    )


def _assert_matches_oracle(y: Any, expected: np.ndarray, what: str) -> None:
    actual = np.asarray(keras.ops.convert_to_numpy(y), dtype=np.float64)
    assert actual.shape == expected.shape, (
        f"{what}: shape {actual.shape} vs oracle {expected.shape}"
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg=(
            f"{what}: values are in the wrong positions -- the leading dims "
            "recovered by _reshape_output do not match the concrete input"
        ),
    )


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
def test_symbolically_built_layer_reused_on_a_concrete_batch(
    case: str, output_mode: str
) -> None:
    """Build the SAME layer instance against ``(None, 8)``, then call it on data."""
    layer_cls, config, param_names = CASES[case](output_mode)
    layer = layer_cls(cluster_axis=-1, **config)

    # Symbolic build only -- original_shape becomes [None, None, 8].
    layer(keras.Input(shape=(None, FEATURES)))
    assert layer.original_shape[1] is None, (
        "fixture broken: the symbolic build did not capture a dynamic non-batch "
        f"axis (original_shape={layer.original_shape})"
    )

    x = np.random.RandomState(7).normal(size=(BATCH, SEQ, FEATURES)).astype("float32")
    expected = _expected(x, layer_cls, config, param_names, layer, output_mode)
    _assert_matches_oracle(layer(x), expected, f"{case}/{output_mode} bare layer")


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
@pytest.mark.parametrize("invocation", ["call", "predict"])
def test_functional_model_over_a_dynamic_sequence_axis(
    case: str, output_mode: str, invocation: str
) -> None:
    """``model(x)`` and ``model.predict(x)`` both go through the same layer.

    F-1 measured two DIFFERENT failure messages for these two entry points, so
    both are pinned rather than assuming one stands in for the other.
    """
    layer_cls, config, param_names = CASES[case](output_mode)
    layer = layer_cls(cluster_axis=-1, **config)

    inputs = keras.Input(shape=(None, FEATURES))
    model = keras.Model(inputs, layer(inputs))

    x = np.random.RandomState(11).normal(size=(BATCH, SEQ, FEATURES)).astype("float32")
    expected = _expected(x, layer_cls, config, param_names, layer, output_mode)

    y = model(x) if invocation == "call" else model.predict(x, verbose=0)
    _assert_matches_oracle(y, expected, f"{case}/{output_mode} model.{invocation}")


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
def test_two_different_concrete_lengths_through_one_built_layer(
    case: str, output_mode: str
) -> None:
    """The point of a dynamic axis: one built layer, several sequence lengths.

    Uses lengths 5 and 9 -- both distinct from batch, features and prototypes --
    so a fix that merely hard-codes one concrete length cannot pass.
    """
    layer_cls, config, param_names = CASES[case](output_mode)
    layer = layer_cls(cluster_axis=-1, **config)

    inputs = keras.Input(shape=(None, FEATURES))
    model = keras.Model(inputs, layer(inputs))

    for seq in (SEQ, 9):
        x = (
            np.random.RandomState(seq)
            .normal(size=(BATCH, seq, FEATURES))
            .astype("float32")
        )
        expected = _expected(x, layer_cls, config, param_names, layer, output_mode)
        _assert_matches_oracle(
            model(x), expected, f"{case}/{output_mode} seq={seq}"
        )


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
def test_multi_axis_cluster_axis_with_a_dynamic_non_feature_axis(
    case: str, output_mode: str
) -> None:
    """The ``len(axes) > 1`` branch shares the static-``original_shape`` premise.

    Rank 4 ``(batch, None, 7, 4)`` clustered over the last TWO axes leaves the
    dynamic axis in ``non_feature_dims[1:]`` -- the same slot that breaks the
    single-axis case -- and additionally exercises
    ``compute_output_shape``'s multi-axis branch. All lengths distinct:
    batch 2, seq 5, then 7 and 4, prototypes 3.
    """
    layer_cls, config, param_names = CASES[case](output_mode)
    layer = layer_cls(cluster_axis=[-2, -1], **config)

    inputs = keras.Input(shape=(None, 7, 4))
    outputs = layer(inputs)
    model = keras.Model(inputs, outputs)

    x = np.random.RandomState(13).normal(size=(BATCH, SEQ, 7, 4)).astype("float32")
    expected = build_cluster_axis_oracle(
        x,
        [-2, -1],
        output_mode,
        PROTOTYPES,
        flat_twin_forward(layer_cls, config, param_names, layer),
    )
    _assert_matches_oracle(
        model(x), expected, f"{case}/{output_mode} multi-axis"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_compute_output_shape_multi_axis_branch_propagates_the_dynamic_axis(
    case: str,
) -> None:
    """A shape function may return ``None``; it may not crash or mis-place ``K``.

    Phase-3 measurement for ``base.py``'s ``len(axes) > 1`` branch: with
    ``cluster_axis=[-2, -1]`` on ``(None, None, 7, 4)`` the two clustered axes
    collapse to one axis of length ``K`` at the position of the lowest clustered
    axis, and the dynamic non-feature axis is propagated as ``None``.
    """
    layer_cls, config, _ = CASES[case]("assignments")
    layer = layer_cls(cluster_axis=[-2, -1], **config)

    assert layer.compute_output_shape((None, None, 7, 4)) == (
        None,
        None,
        PROTOTYPES,
    )
