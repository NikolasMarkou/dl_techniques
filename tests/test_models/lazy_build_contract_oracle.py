"""
Lazy-build contract oracle -- what a missing or partial ``build()`` actually costs
=================================================================================

This module is an *instrument*, not a test suite. Like ``precision_arm_oracle``
and ``smoke_contract_oracle`` beside it, it carries no ``test_`` prefix so
pytest does not collect it. Its own RED proofs live in
``tests/test_models/test_lazy_build_contract_oracle.py``.

The rows it answers
-------------------
Audit rules R-002 (no ``build()``), R-004 / R-021 (``compute_output_shape``
raises on an unbuilt instance), R-070 (``build()`` does not materialize) and
R-071 (``.build(shape)`` then ``count_params() == 0``) were charged against ~40
``models/`` packages, ~110 cells in all. They are one family and they were
charged as a CONTRACT: "the model should be materialized by ``build()``".

Why this oracle does not assert that contract
---------------------------------------------
Because the contract is not what matters, and the plan measured that twice in
opposite directions.

* Batch 2 measured a real ``.keras`` round trip on all 8 transformer packages
  that violate R-002/R-004/R-071 and found 22-50 tensors restored **before the
  first call**. The contract fails; nothing is lost.
* ``BERT`` was the exception (step 17.1, D-049): its missing ``build()``
  silently disabled weight tying, ``CausalLanguageModel.embedding_weights``
  read ``None``, ``use_weight_tying`` reported ``False`` and the round trip
  RAISED. There the contract failure had a consequence, and the fix was a real
  fix -- not an assertion.
* ``SHGCNLinkPredictor`` (batch 7, D-029) was the second: its archive was
  COMPLETE and its LOAD was lossy, because an unbuilt sub-layer's
  ``load_own_variables`` is skipped and the decoder silently reverted to its
  defaults, with no warning.

So the question this oracle asks is **"does the lazy build cost anything?"**,
not "is the model built after ``build()``?". A test that pins the contract would
turn ~110 healthy packages red and would not have caught either real defect,
both of which were found by a VALUE comparison across a save/load cycle.

The protocol
------------
:func:`measure_lazy_build` runs, in order:

1. construct, one forward pass -- the model is now fully materialized;
2. **perturb** every float weight relatively
   (``sigma = max(0.25 * std(w), 1e-3)``), excluding BatchNorm moving
   statistics, which are not restored by the same path and whose perturbation
   drove a prior probe's model to NaN;
3. forward again -- and require the perturbation to have MOVED the output. This
   liveness arm is the whole instrument: a round trip that compares an
   unperturbed model against itself passes over total weight loss, which is
   exactly how ``ScoreBasedNanoVLM``'s own round-trip test passed 3/3 while 464
   of 1,305 tensors were never written (batch 3);
4. ``.save()`` / ``load_model()`` / forward -- and require an **exact** match
   against step 3.

Separately it reports the materialization RATIO (weights after ``build()``
alone, against weights after a call) as a NUMBER, not a verdict, so a package
can pin what it measured and notice when it changes.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

import keras
import numpy as np
from keras import ops


__all__ = [
    "flatten_tensors",
    "perturb_weights",
    "materialization_report",
    "measure_lazy_build",
    "assert_lazy_build_costs_nothing",
]


def flatten_tensors(output: Any) -> List[Any]:
    """Tensor leaves of a model output, in a stable order. See the twin in
    ``precision_arm_oracle`` for why ``keras.tree`` is not used."""
    if isinstance(output, dict):
        items = list(output.values())
    elif isinstance(output, (list, tuple)):
        items = list(output)
    else:
        items = [output]
    return [t for t in items if hasattr(t, "shape") and hasattr(t, "dtype")]


def perturb_weights(model: Any, seed: int = 0) -> int:
    """
    Perturb every float weight of ``model`` relative to its own spread.

    BatchNorm moving statistics are skipped. An ABSOLUTE perturbation was tried
    first in batch 8 and drove ``yolo12`` to NaN before the round trip could
    run; a relative one with a floor is the corrected instrument.

    :param model: A built model.
    :param seed: NumPy seed, so the perturbation is reproducible.
    :return: The number of weights actually perturbed. A caller must check this
        is non-zero -- a perturbation that touched nothing makes every later
        comparison vacuous.
    :rtype: int
    """
    rs = np.random.RandomState(seed)
    touched = 0
    for w in model.weights:
        name = w.path.lower()
        if "moving_mean" in name or "moving_variance" in name:
            continue
        value = np.asarray(ops.convert_to_numpy(w))
        if value.dtype.kind != "f":
            continue
        sigma = max(0.25 * float(value.std()), 1e-3)
        noise = np.asarray(rs.standard_normal(value.shape)).astype(value.dtype)
        w.assign(value + noise * sigma)
        touched += 1
    return touched


def materialization_report(
        build: Callable[[], Any],
        input_shape: Any,
        n_weights_after_call: int,
) -> Dict[str, Any]:
    """
    What ``.build(input_shape)`` alone materializes, as a number.

    ``count_params()`` is called inside a ``try``: ``tabm`` was measured to
    RAISE ``ValueError`` from it after ``.build()``, which is a different (and
    worse) reading than the ``0`` the rest of the family gives, and collapsing
    the two would hide it.

    :return: ``{"n_weights_after_build", "count_params_after_build",
        "n_weights_after_call", "ratio"}``. ``count_params_after_build`` is
        either an int or a string naming the exception type.
    """
    model = build()
    try:
        model.build(input_shape)
        n_after = len(model.weights)
    except Exception as exc:  # noqa: BLE001 -- the exception IS the reading
        return {
            "n_weights_after_build": f"RAISE {type(exc).__name__}",
            "count_params_after_build": "-",
            "n_weights_after_call": n_weights_after_call,
            "ratio": None,
        }
    try:
        params = model.count_params()
    except Exception as exc:  # noqa: BLE001
        params = f"RAISE {type(exc).__name__}"
    return {
        "n_weights_after_build": n_after,
        "count_params_after_build": params,
        "n_weights_after_call": n_weights_after_call,
        "ratio": (n_after / n_weights_after_call) if n_weights_after_call else None,
    }


def measure_lazy_build(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        input_shape: Optional[Any] = None,
        seed: int = 0,
) -> Dict[str, Any]:
    """
    Run the four-step protocol in this module's docstring and report.

    :param build: Zero-argument model factory.
    :param make_inputs: Zero-argument, DETERMINISTIC input factory. It is called
        three times and the three calls must return the same values, or the
        "exact" comparison in step 4 is measuring the inputs.
    :param input_shape: If given, also produce a :func:`materialization_report`.
    :param seed: Forwarded to :func:`perturb_weights`.
    :return: ``{"n_weights", "n_perturbed", "perturb_liveness",
        "roundtrip_max_delta", "per_output_delta", "n_weights_reloaded",
        "materialization"}``.
    """
    keras.utils.set_random_seed(seed)
    model = build()
    inputs = make_inputs()

    clean = [np.asarray(ops.convert_to_numpy(t))
             for t in flatten_tensors(model(inputs, training=False))]
    n_weights = len(model.weights)

    n_perturbed = perturb_weights(model, seed=seed)
    perturbed = [np.asarray(ops.convert_to_numpy(t))
                 for t in flatten_tensors(model(inputs, training=False))]

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "subject.keras")
        model.save(path)
        reloaded = keras.models.load_model(path, compile=False)
        restored = [np.asarray(ops.convert_to_numpy(t))
                    for t in flatten_tensors(reloaded(inputs, training=False))]
        n_weights_reloaded = len(reloaded.weights)

    def _delta(a: List[Any], b: List[Any]) -> List[float]:
        return [float(np.max(np.abs(x.astype("float64") - y.astype("float64"))))
                for x, y in zip(a, b)]

    liveness = _delta(clean, perturbed)
    roundtrip = _delta(perturbed, restored)

    report: Dict[str, Any] = {
        "n_weights": n_weights,
        "n_perturbed": n_perturbed,
        "n_outputs": len(clean),
        "perturb_liveness": max(liveness) if liveness else 0.0,
        "roundtrip_max_delta": max(roundtrip) if roundtrip else None,
        "per_output_delta": roundtrip,
        "n_weights_reloaded": n_weights_reloaded,
    }
    if input_shape is not None:
        keras.utils.set_random_seed(seed)
        report["materialization"] = materialization_report(
            build, input_shape, n_weights
        )
    return report


def assert_lazy_build_costs_nothing(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        input_shape: Optional[Any] = None,
        seed: int = 0,
        atol: float = 0.0,
) -> Dict[str, Any]:
    """
    Assert that a lazily-built model loses nothing across a save/load cycle.

    :param atol: Deliberately ``0.0``. A restored weight is the SAME float or it
        is a different weight; there is no tolerance to spend here, and R-073
        charges the whole tree with never having made this comparison at
        ``atol=0.0``.
    :raises AssertionError: if the perturbation is dead (the comparison would be
        vacuous), if the reload changes the output, or if the weight count moves.
    """
    report = measure_lazy_build(
        build, make_inputs, input_shape=input_shape, seed=seed
    )

    assert report["n_weights"] > 0, "the model materialized no weights at all"
    assert report["n_perturbed"] > 0, (
        "the perturbation touched no weight, so the round-trip comparison "
        "below cannot distinguish a restored model from a fresh one"
    )
    assert report["perturb_liveness"] > 0.0, (
        "perturbing every float weight did not move the output by any amount; "
        "the forward is insensitive to its own weights, so an exact round trip "
        "proves nothing (this is the shape that let ScoreBasedNanoVLM's own "
        "round-trip test pass 3/3 while 464 of 1,305 tensors were never saved)"
    )
    assert report["n_weights_reloaded"] == report["n_weights"], (
        f"the reloaded model has {report['n_weights_reloaded']} weights against "
        f"{report['n_weights']} before the save"
    )
    assert report["roundtrip_max_delta"] is not None
    assert report["roundtrip_max_delta"] <= atol, (
        f"save/load changed the output by {report['roundtrip_max_delta']:.6e} "
        f"(atol={atol}); per output {report['per_output_delta']}. The lazy "
        "build IS costing something -- see D-029 (SHGCNLinkPredictor) and "
        "D-049 (BERT) for the two shapes this takes."
    )
    return report
