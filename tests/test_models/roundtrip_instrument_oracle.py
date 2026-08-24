"""
The three named round-trip instruments -- R-063, R-072, R-073 (and R-135)
=========================================================================

This module is an *instrument*, not a test suite. Like ``precision_arm_oracle``
and ``lazy_build_contract_oracle`` beside it, it carries no ``test_`` prefix so
pytest does not collect it. Its own RED proofs live in
``tests/test_models/test_roundtrip_instrument_oracle.py``; the per-package
family that consumes it is ``test_roundtrip_instrument_family.py``.

The rows it answers
-------------------
Four audit rules, charged against essentially the whole of ``models/``:

* **R-063** (CRITICAL) -- the ``.keras`` round trip compares output VALUES with
  ``rtol=0``. Measured across eight batches: ``rtol=0`` appears in exactly two
  of 73 package test directories.
* **R-072** (HIGH) -- build parity by RELATIVE weight path between two
  instances of the same builder, plus a no-sub-layer-config sibling proving a
  disabled component's weights are ABSENT. Measured **0 of 73**.
* **R-073** (CRITICAL) -- weight-VALUE comparison at ``atol=0.0`` **before the
  loaded model's first call**. Measured absent everywhere.
* **R-135** (CRITICAL) -- all three of the above present, per package. That one
  is a coverage row and is answered by the family's completeness test, not here.

What was MEASURED before this oracle was written, and why it shapes the design
------------------------------------------------------------------------------
A 72-subject sweep (every ``models/`` package that has a model at all) found
the *substance* of R-063/R-073 already passing: **the weight-value delta across
a save/load cycle is exactly ``0.000000e+00`` in every package where the
comparison could be made at all**. What is missing is the instrument, and two
specific things about the instrument are not obvious:

1.  **``w.path.split("/", 1)[-1]`` is NOT a relative path after a reload.** The
    donor's weights are ``res_net/stem_conv/kernel``; the RELOADED model's are
    ``stem_conv/kernel`` -- the top-level model name is absent. Splitting on the
    first ``/`` therefore strips the model name on one side and the first real
    layer name on the other, and the comparison silently pairs nothing. The
    first version of this sweep read "``accunet``: 680 of 680 weights missing
    after reload" and that reading was the instrument, not the model.
    :func:`relative_path` strips the model's OWN name and nothing else.

2.  **Auto-generated layer names drift between two instances of the same
    builder**, because Keras 3's ``auto_name`` counter is global to the process.
    A sub-layer built without an explicit ``name=`` is ``swi_gluffn_6`` in the
    first instance and ``swi_gluffn_14`` in the second. That drift is exactly
    what R-072(a) exists to detect -- it is the observable symptom of "not every
    sub-layer carries an explicit ``name=``" -- so :func:`assert_build_parity`
    does not forgive it. It requires the drift to be covered by an explicit,
    non-stale waiver naming the auto-named layer stems.

Vacuity, and the four mechanisms that have defeated a round-trip comparison here
-------------------------------------------------------------------------------
A round-trip comparison that cannot fail is worse than no comparison. Four
distinct mechanisms have produced one in this repo's history: a zero-initialized
projection, a BatchNorm collapsed at inference, an initializer-seed
coincidence, and a reconstruction whose own std was 4e-05. All four are the same
shape -- **the donor and a fresh model agree, so "restored" and "re-initialized"
are indistinguishable**. Every measurement here therefore perturbs the donor
first and reports:

* ``n_effective`` -- weights that were actually perturbed AND compared. A caller
  must require this to be non-zero.
* ``n_inert`` -- weights that a FRESH instance reproduces exactly, i.e. the ones
  the comparison cannot speak for (integer weights and BatchNorm moving
  statistics are deliberately not perturbed and land here).

``vae`` is the mirror case: it samples at inference, so its *output* comparison
cannot pass at ``atol=0.0`` at all -- the same model called twice differs by
more than any edit does. That is not an exemption from the family; it is why the
output arm takes a ``call_fn`` and ``vae`` is judged on its deterministic
``z_mean`` head while its WEIGHT arm (R-073) stays at ``atol=0.0`` like
everyone else.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
from keras import ops
from keras.src.utils.naming import to_snake_case

from .lazy_build_contract_oracle import flatten_tensors, perturb_weights


__all__ = [
    "relative_path",
    "weight_map",
    "n_colliding_paths",
    "measure_roundtrip",
    "assert_roundtrip_output_values",
    "assert_weights_restored_before_first_call",
    "measure_build_parity",
    "assert_build_parity",
    "assert_disabled_component_has_no_weights",
]


def _model_prefixes(model: Any) -> Tuple[str, ...]:
    """Every leading path segment that means "this model", not a sub-layer.

    Three spellings, and all three occur in ``models/``:

    * ``model.name`` itself;
    * ``to_snake_case(model.name)`` -- ``ACC_UNet`` sets ``name="ACC_UNet"`` and
      its variables come out under ``acc_u_net/``, so the two DISAGREE and a
      literal name match strips nothing;
    * ``to_snake_case(type(model).__name__)`` -- the auto-name of an instance
      that was given no name at all.

    A trailing ``_<n>`` is allowed on each, because that is exactly what the
    global auto-name counter appends to a SECOND instance.
    """
    names = {model.name, to_snake_case(model.name),
             to_snake_case(type(model).__name__)}
    return tuple(sorted(n for n in names if n))


def relative_path(model: Any, weight: Any) -> str:
    """
    ``weight``'s path with ``model``'s OWN prefix stripped, and nothing else.

    See this module's docstring, point 1: a reloaded model's weight paths do not
    carry the model prefix at all, so the ``split("/", 1)[-1]`` spelling the
    rule sketch uses is asymmetric across a save/load boundary -- it strips the
    model name on one side and the first real layer name on the other -- and
    the comparison then pairs nothing.

    :param model: The model owning ``weight``.
    :param weight: A Keras variable with a ``.path``.
    :return: The path relative to the model.
    :rtype: str
    """
    head, _, tail = weight.path.partition("/")
    if not tail:
        return weight.path
    for prefix in _model_prefixes(model):
        if head == prefix or re.fullmatch(re.escape(prefix) + r"_\d+", head):
            return tail
    return weight.path


def weight_map(model: Any) -> List[Tuple[str, np.ndarray]]:
    """
    ``(relative path, value)`` for every weight of ``model``, in weight order.

    A LIST, not a dict, and the difference is not stylistic: a relative path is
    not unique. ``depth_anything`` builds its frozen teacher with
    ``clone_model``, which bakes the STUDENT's name into the clone's variable
    paths, so 46 of its 114 weights share a path with another weight. A dict
    silently collapses those to 68 entries and the comparison then speaks for
    barely half the model. :func:`n_colliding_paths` reports the count instead.
    """
    return [(relative_path(model, w), np.asarray(ops.convert_to_numpy(w)))
            for w in model.weights]


def n_colliding_paths(pairs: Sequence[Tuple[str, Any]]) -> int:
    """How many entries share a relative path with an earlier entry."""
    return len(pairs) - len({path for path, _ in pairs})


def _as_arrays(tensors: Sequence[Any]) -> List[np.ndarray]:
    return [np.asarray(ops.convert_to_numpy(t)) for t in tensors]


def _call(model: Any, inputs: Any, call_fn: Optional[Callable], training: bool) -> Any:
    if call_fn is not None:
        return call_fn(model, inputs, training)
    return model(inputs, training=training)


def measure_roundtrip(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
        training: bool = False,
        seed: int = 0,
) -> Dict[str, Any]:
    """
    One save/load cycle, reported for BOTH the output arm and the weight arm.

    The two arms share a cycle deliberately: they are the same event seen from
    two sides, and running them separately doubles the cost of the family for
    72 packages while allowing the two halves to disagree about which model
    they measured.

    Order matters and is the whole point of R-073: the loaded model's weights
    are read **before anything calls it**. ``call_count_before_weight_read``
    proves that, by counting invocations of the loaded model's own ``call``
    between load and read. A ``build()``-only load path fills a gap with fresh
    random weights on first call, at which point the weight COUNT is identical
    for the correct and the broken variant -- so a comparison made after a
    forward pass is not this comparison.

    :param build: Zero-argument model factory.
    :param make_inputs: Zero-argument DETERMINISTIC input factory.
    :param call_fn: ``(model, inputs, training) -> outputs``, for a model whose
        forward is not ``model(x)``. Same contract as ``precision_arm_oracle``.
    :param training: Forwarded to the call.
    :param seed: Seeds both the build and the perturbation.
    :return: A report dict; see the keys assembled at the end of this function.
    """
    keras.utils.set_random_seed(seed)
    model = build()
    inputs = make_inputs()
    _call(model, inputs, call_fn, training)

    n_perturbed = perturb_weights(model, seed=seed)
    # ORDER: the forward pass runs BEFORE the weight snapshot. A forward at
    # ``training=True`` updates BatchNorm moving statistics, so a snapshot taken
    # first is stale by the time the model is saved -- measured on ``yolo12``,
    # which is judged in training mode (D-065) and read a spurious
    # ``weight_max_delta`` of 3.56e+00 against a true delta of exactly 0.0.
    donor_outputs = _as_arrays(flatten_tensors(_call(model, inputs, call_fn, training)))
    # The model's OWN call-to-call spread, same weights, same inputs. For a
    # deterministic model this is exactly 0.0; for one that samples at inference
    # it is the only honest bound a round-trip comparison can be held to, and
    # it must be MEASURED rather than assumed (see ``vae``).
    self_outputs = _as_arrays(flatten_tensors(_call(model, inputs, call_fn, training)))
    donor_weights = weight_map(model)

    # A FRESH instance at a different seed: any weight it reproduces exactly is
    # a weight the round trip cannot speak for (see "Vacuity" above).
    keras.utils.set_random_seed(seed + 4242)
    fresh = build()
    _call(fresh, make_inputs(), call_fn, training)
    fresh_weights = weight_map(fresh)

    calls: List[int] = [0]

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "subject.keras")
        model.save(path)
        loaded = keras.models.load_model(path, compile=False)

        original_call = loaded.call

        def _counting_call(*args: Any, **kwargs: Any):
            calls[0] += 1
            return original_call(*args, **kwargs)

        loaded.call = _counting_call
        loaded_weights = weight_map(loaded)
        calls_before_read = calls[0]
        loaded_outputs = _as_arrays(
            flatten_tensors(_call(loaded, inputs, call_fn, training)))
        calls_after_forward = calls[0]

    donor_keys = [path for path, _ in donor_weights]
    loaded_keys = [path for path, _ in loaded_weights]
    loaded_by_path = dict(loaded_weights)
    fresh_by_path = dict(fresh_weights)

    if donor_keys == loaded_keys:
        # Same paths in the same order: positional pairing IS path pairing.
        matched_by = "path"
        pairs = [(a, donor, loaded) for (a, donor), (_, loaded)
                 in zip(donor_weights, loaded_weights)]
    elif len(donor_keys) == len(loaded_keys) and all(
            donor.shape == loaded.shape for (_, donor), (_, loaded)
            in zip(donor_weights, loaded_weights)):
        # Auto-name drift across the reload, not weight loss: same count, same
        # shapes, same order. Pair positionally and SAY SO -- the report carries
        # ``n_path_mismatch`` so the family can pin it per package.
        matched_by = "position"
        pairs = [(a, donor, loaded) for (a, donor), (_, loaded)
                 in zip(donor_weights, loaded_weights)]
    else:
        matched_by = "path"
        pairs = [(a, donor, loaded_by_path[a])
                 for a, donor in donor_weights if a in loaded_by_path]

    n_path_mismatch = sum(
        1 for (a, _, _), b in zip(pairs, loaded_keys) if a != b
    ) if matched_by == "position" else 0
    inert = [a for a, donor, _ in pairs
             if a in fresh_by_path
             and fresh_by_path[a].shape == donor.shape
             and float(np.max(np.abs(fresh_by_path[a].astype("float64")
                                     - donor.astype("float64")))) == 0.0]

    weight_deltas = {
        a: float(np.max(np.abs(donor.astype("float64") - loaded.astype("float64"))))
        for a, donor, loaded in pairs
    }
    output_deltas = [
        float(np.max(np.abs(x.astype("float64") - y.astype("float64"))))
        for x, y in zip(donor_outputs, loaded_outputs)
    ]
    self_deltas = [
        float(np.max(np.abs(x.astype("float64") - y.astype("float64"))))
        for x, y in zip(donor_outputs, self_outputs)
    ]

    return {
        "n_weights": len(donor_keys),
        "n_colliding_paths": n_colliding_paths(donor_weights),
        "n_weights_reloaded": len(loaded_keys),
        "n_perturbed": n_perturbed,
        "n_compared": len(pairs),
        "n_unmatched": len(donor_keys) - len(pairs),
        "n_path_mismatch": n_path_mismatch,
        "matched_by": matched_by,
        "n_inert": len(inert),
        "n_effective": len(pairs) - len(inert),
        "inert_sample": sorted(inert)[:5],
        "weight_max_delta": max(weight_deltas.values()) if weight_deltas else None,
        "worst_weight": (max(weight_deltas, key=weight_deltas.get)
                         if weight_deltas else None),
        "call_count_before_weight_read": calls_before_read,
        # The counter's own liveness: if this is not > 0 the counter is not
        # wired to anything and the "before first call" guard above is vacuous.
        "call_count_after_forward": calls_after_forward,
        "n_outputs": len(donor_outputs),
        "output_max_delta": max(output_deltas) if output_deltas else None,
        "per_output_delta": output_deltas,
        "self_max_delta": max(self_deltas) if self_deltas else None,
        "per_output_self_delta": self_deltas,
        "donor_outputs": donor_outputs,
        "loaded_outputs": loaded_outputs,
    }


def assert_roundtrip_output_values(report: Dict[str, Any], *, atol: float = 0.0,
                                   calibrate: bool = False) -> None:
    """
    R-063: the round trip compares output VALUES, with ``rtol=0``.

    ``rtol=0`` is not decoration. ``np.testing.assert_allclose``'s default
    ``rtol=1e-7`` silently contributes to a nominally-``atol`` bound; the rule
    records a measured case where 1.24e-05 of a 1.53e-05 failure came from the
    relative term. The comparison here is a copy-versus-copy comparison, so the
    default ``atol`` is also ``0.0``.

    :param report: A :func:`measure_roundtrip` report.
    :param atol: Absolute tolerance. Leave at ``0.0`` unless the model's forward
        is not deterministic, and say what was measured if it is not.
    :param calibrate: For a model that samples at inference. The bound becomes
        the model's OWN call-to-call spread, measured on the same weights and
        the same inputs, and that spread is asserted to be non-zero -- so
        calibration cannot be switched on for a deterministic model to make a
        real round-trip loss disappear.
    :raises AssertionError: on any difference beyond the bound.
    """
    assert report["n_outputs"] > 0, "the model returned no tensor to compare"
    if calibrate:
        assert report["self_max_delta"] is not None and report["self_max_delta"] > 0.0, (
            "calibrate=True was asked for, but the model called twice with the "
            "same weights and the same inputs gave an IDENTICAL answer "
            f"({report['self_max_delta']}); it is deterministic, so the bound "
            "must be atol=0.0 and this call is hiding a real difference")
        atol = max(atol, report["self_max_delta"])
    for index, (donor, loaded) in enumerate(
            zip(report["donor_outputs"], report["loaded_outputs"])):
        np.testing.assert_allclose(
            donor, loaded, atol=atol, rtol=0,
            err_msg=(f"output {index} changed across a .keras round trip by "
                     f"{report['per_output_delta'][index]:.6e}"),
        )


def assert_weights_restored_before_first_call(
        report: Dict[str, Any], *, atol: float = 0.0,
        allow_positional_match: bool = False,
        expect_path_collisions: int = 0) -> None:
    """
    R-073: every weight VALUE is restored, read BEFORE the loaded model runs.

    :param report: A :func:`measure_roundtrip` report.
    :param atol: Deliberately ``0.0``. Restoration is a copy, not a computation.
    :param allow_positional_match: Permit the pairing to fall back to weight
        ORDER when auto-generated layer names drift across the reload. The
        fallback is safe (same count, same shapes, same order) but it is a
        weaker pairing, so a caller must ask for it explicitly and the family
        pins which packages need it.
    :param expect_path_collisions: The MEASURED number of weights sharing a
        relative path with another weight. Asserted exactly, not as a maximum,
        so a package that grows a new collision fires and a package that
        repairs one fires too.
    :raises AssertionError: if weights are lost, if the comparison is vacuous,
        if the loaded model was called before the read, or on any value change.
    """
    assert report["n_weights"] > 0, "the model materialized no weights at all"
    assert report["n_colliding_paths"] == expect_path_collisions, (
        f"{report['n_colliding_paths']} weight(s) share a relative path with "
        f"another weight, against an expected {expect_path_collisions}; a "
        "path-keyed load (load_weights(by_name), a by-name checkpoint) cannot "
        "address them")
    assert report["n_weights_reloaded"] == report["n_weights"], (
        f"the reloaded model has {report['n_weights_reloaded']} weights against "
        f"{report['n_weights']} in the donor")
    assert report["call_count_after_forward"] > 0, (
        "the loaded model's call counter never incremented, not even for the "
        "forward pass this measurement makes on purpose -- the counter is not "
        "wired to anything, so the 'before first call' assertion below cannot "
        "fail")
    assert report["call_count_before_weight_read"] == 0, (
        "the loaded model was called "
        f"{report['call_count_before_weight_read']} time(s) before its weights "
        "were read -- a build()-only load path fills the gap with fresh random "
        "weights on first call, so this is no longer R-073's comparison")
    assert report["n_perturbed"] > 0, (
        "the perturbation touched no weight, so a restored model and a freshly "
        "initialized one are indistinguishable here")
    assert report["n_effective"] > 0, (
        f"all {report['n_compared']} compared weights are reproduced exactly by "
        f"a FRESH instance ({report['inert_sample']}), so this comparison "
        "cannot fail -- see the four vacuity mechanisms in the module docstring")
    if not allow_positional_match:
        assert report["matched_by"] == "path", (
            f"{report['n_path_mismatch']} weight paths drifted across the "
            "reload, so the pairing fell back to weight ORDER; pass "
            "allow_positional_match=True and pin the count if that is the "
            "measured state of this package")
    assert report["n_unmatched"] == 0, (
        f"{report['n_unmatched']} donor weights have no counterpart in the "
        "reloaded model")
    assert report["weight_max_delta"] is not None
    assert report["weight_max_delta"] <= atol, (
        f"weight {report['worst_weight']!r} came back different by "
        f"{report['weight_max_delta']:.6e} (atol={atol})")


def measure_build_parity(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
        training: bool = False,
        input_shape: Optional[Any] = None,
        seed: int = 0,
) -> Dict[str, Any]:
    """
    R-072(a): relative-weight-path parity, in the two forms that both matter.

    * **lazy vs lazy** -- two instances of the SAME builder, each materialized by
      a forward pass. Any difference is auto-name drift, i.e. a sub-layer built
      without an explicit ``name=`` (module docstring, point 2).
    * **explicit vs lazy** -- one instance materialized by ``.build(shape)``.
      Measured across 72 packages, ``.build()`` alone materializes NOTHING in
      roughly a third of them, so this half reports a ratio and requires only
      that the explicit path set be a SUBSET of the lazy one (no phantom
      weights). Requiring equality would restate the ~110-cell lazy-build
      contract row that D-056 already closed on the consequence.

    :param input_shape: If ``None``, the explicit half is skipped and reported
        as ``"skipped"``.
    :return: ``{"n_lazy", "drift", "explicit"}``; ``drift`` is the sorted list of
        relative paths present in exactly one of the two lazy instances.
    """
    keras.utils.set_random_seed(seed)
    first = build()
    inputs = make_inputs()
    _call(first, inputs, call_fn, training)
    lazy_a = [relative_path(first, w) for w in first.weights]

    keras.utils.set_random_seed(seed)
    second = build()
    _call(second, make_inputs(), call_fn, training)
    lazy_b = [relative_path(second, w) for w in second.weights]

    drift = sorted(set(lazy_a) ^ set(lazy_b))

    explicit: Dict[str, Any] = {"status": "skipped"}
    if input_shape is not None:
        keras.utils.set_random_seed(seed)
        third = build()
        try:
            third.build(input_shape)
        except Exception as exc:  # noqa: BLE001 -- the exception IS the reading
            explicit = {"status": f"RAISE {type(exc).__name__}"}
        else:
            paths = [relative_path(third, w) for w in third.weights]
            explicit = {
                "status": "built",
                "n_explicit": len(paths),
                "ratio": len(paths) / len(lazy_a) if lazy_a else None,
                "not_in_lazy": sorted(set(paths) - set(lazy_a)),
            }

    return {
        "n_lazy": len(lazy_a),
        "n_lazy_unique": len(set(lazy_a)),
        "drift": drift,
        "explicit": explicit,
    }


def assert_build_parity(report: Dict[str, Any], *,
                        autoname_stems: Sequence[str] = (),
                        expect_path_collisions: int = 0) -> None:
    """
    Assert R-072(a) against a :func:`measure_build_parity` report.

    :param autoname_stems: Substrings naming the sub-layer classes this package
        is MEASURED to build without an explicit ``name=``. Every drifting path
        must contain one of them (so a NEW unnamed sub-layer fires), and every
        stem must match at least one drifting path (so a stem left behind by a
        repair fires -- the same waiver discipline the mutable-default family
        uses, which forced twelve deletions in step 19).
    :param expect_path_collisions: The MEASURED number of weights sharing a
        relative path with another weight, asserted exactly. See
        :func:`weight_map` for the one package where this is non-zero.
    :raises AssertionError: on uncovered drift, on a stale stem, or on a weight
        that ``.build()`` materialized and a forward pass does not.
    """
    assert report["n_lazy"] > 0, "the builder materialized no weights"
    collisions = report["n_lazy"] - report["n_lazy_unique"]
    assert collisions == expect_path_collisions, (
        f"{collisions} weight(s) share a relative path with another weight, "
        f"against an expected {expect_path_collisions}; path parity is not a "
        "pairing for them")

    uncovered = [p for p in report["drift"]
                 if not any(stem in p for stem in autoname_stems)]
    assert not uncovered, (
        f"{len(uncovered)} weight path(s) differ between two instances of the "
        f"same builder and are covered by no declared auto-name stem: "
        f"{uncovered[:5]}. Keras 3's auto_name counter is global to the "
        "process, so this is the symptom of a sub-layer built without an "
        "explicit name=.")
    for stem in autoname_stems:
        assert any(stem in p for p in report["drift"]), (
            f"auto-name stem {stem!r} matches no drifting path any more -- the "
            "sub-layer was given an explicit name=; delete the stem in the "
            "same commit as the repair")

    explicit = report["explicit"]
    if explicit.get("status") == "built":
        # The explicitly-built instance is a THIRD instance in the process, so
        # the same global auto-name counter that produced ``drift`` above also
        # renumbers ITS sub-layers. Only paths that no declared stem explains
        # are phantom weights.
        phantom = [p for p in explicit["not_in_lazy"]
                   if not any(stem in p for stem in autoname_stems)]
        assert not phantom, (
            f".build() materialized {len(phantom)} weight(s) a forward pass "
            f"does not: {phantom[:5]}")


def assert_disabled_component_has_no_weights(
        build_with: Callable[[], Any],
        build_without: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        marker: str,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
        training: bool = False,
        seed: int = 0,
) -> Dict[str, Any]:
    """
    R-072(b): the no-sub-layer-config sibling.

    Build parity alone is blind to over-building -- it passes when BOTH paths
    build everything, which is the failure this arm exists for. Here the same
    builder is instantiated twice, once with a component enabled and once with
    it disabled, and the disabled instance must carry NO weight whose relative
    path contains ``marker``.

    :param marker: A substring identifying the component's weights. Asserted to
        match at least one weight in the enabled instance, so a typo in the
        marker cannot make the arm vacuously green.
    :return: ``{"n_with", "n_without", "n_marked_with", "n_marked_without"}``.
    :raises AssertionError: if the marker matches nothing when enabled, or
        anything when disabled, or if disabling changed nothing at all.
    """
    keras.utils.set_random_seed(seed)
    enabled = build_with()
    inputs = make_inputs()
    _call(enabled, inputs, call_fn, training)
    with_paths = [relative_path(enabled, w) for w in enabled.weights]

    keras.utils.set_random_seed(seed)
    disabled = build_without()
    _call(disabled, make_inputs(), call_fn, training)
    without_paths = [relative_path(disabled, w) for w in disabled.weights]

    marked_with = [p for p in with_paths if marker in p]
    marked_without = [p for p in without_paths if marker in p]

    assert marked_with, (
        f"marker {marker!r} matches no weight in the ENABLED instance, so this "
        "arm would pass for any marker at all")
    assert not marked_without, (
        f"the disabled sibling still carries {len(marked_without)} weight(s) "
        f"matching {marker!r}: {marked_without[:5]} -- the component was built "
        "anyway, which is exactly what build parity cannot see")
    assert len(without_paths) < len(with_paths), (
        "disabling the component removed no weight at all "
        f"({len(without_paths)} against {len(with_paths)})")

    return {
        "n_with": len(with_paths),
        "n_without": len(without_paths),
        "n_marked_with": len(marked_with),
        "n_marked_without": len(marked_without),
    }
