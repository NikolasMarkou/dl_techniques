"""The homogeneity instrument of the bias-free denoisers: a static scan and a numeric probe.

A bias-free denoiser is degree-1 homogeneous at inference, ``f(a x) = a f(x)`` for ``a > 0``; the
Miyasawa and Tweedie readings of its residual (and the legacy ``eval_psnr_vs_noise`` and DDNM paths)
rest on that. Neither instrument below proves it alone, so ``train.api`` runs both (D-015):

* :func:`scan_static` reads the layers and weights and lists what can break the property: an additive
  term (a ``bias``, a ``beta``, a stock ``BatchNormalization``) or a scale-invariant normalisation
  (``LayerNormalization``, RMS and group norms) or an activation outside the allowlist. It sees a
  GELU, which the numeric probe cannot (GELU reads 4.0e-5 at init and 1.4e-2 only on trained weights).
* :func:`probe_numeric` reads ``max|f(a x) - a f(x)| / max|a f(x)|`` at non-power-of-two ``a``. It sees
  a planted constant offset and a LayerNorm after :func:`init_probe_clone`, which the scan cannot rank
  by size, and it is the only instrument that reads TRAINED weights.

Interface contract
------------------
``scan_static(model)``
    :param model: a BUILT ``keras.Model``; nested layers and nested functional models are walked.
    :return: ``{"offenders": [{"kind", "layer", "detail"}, ...], "counts": {kind: n}, "layer_scales":
        [weight path, ...], "passed": bool}``. ``kind`` is one of :data:`STATIC_KINDS`. ``layer_scales``
        names the layer-scale weights (path ending ``gamma_scale/gamma``); they are listed, never
        offenders (a scale on a branch is homogeneous), and are what :func:`init_probe_clone` sets to 1.
        ``passed`` is true when ``offenders`` is empty.
    :raises ValueError: for a model that is not built (its weights cannot be read).

``probe_numeric(model, inputs, alphas=PROBE_ALPHAS, tolerance=PROBE_TOLERANCE)``
    :param inputs: an array (a batch the model accepts); scaled by each ``alpha`` as ``float32``.
    :param alphas: positive finite scale factors. Keep them OFF powers of two: float32 scaling by a
        power of two is exact, so such an ``alpha`` reads 0.0 whether or not the network is homogeneous.
    :return: ``{"alphas", "max_rel", "per_alpha", "tolerance", "passed", "tf32"}``. ``max_rel`` is the
        worst of ``per_alpha`` (one ``max|f(a x) - a f(x)| / max|a f(x)|`` per alpha, ``inf`` for an
        identically-zero or non-finite output, so a dead model never reads as homogeneous), ``passed``
        is ``max_rel <= tolerance`` and ``tf32`` is the TF32 state IN FORCE during the reading (always
        ``False``: it is switched off in a scope restored on exit and on raise). A model with a list or
        tuple output is read at index 0 and the report then carries ``"output_index": 0``; the key is
        absent for a single output. The model is called with ``training=False`` (homogeneity is an
        inference property: a batch norm in training mode is degree 0 whichever variant it is).
    :raises ValueError: for an empty or non-positive ``alphas``, a non-positive ``tolerance``, or a
        model that returns a dict.

``init_probe_clone(model, seed=0)``
    :return: a CLONE of ``model`` with the same weights except that every layer scale is 1 and every
        ``running_var`` (or ``moving_variance``) is drawn from ``U(0.5, 2)`` by ``numpy.random.default_rng(seed)``.
        The original is not touched, and every global random state the clone moved is put back (see
        :func:`init_probe_clone`).
    :raises: whatever ``keras.models.clone_model`` raises (it needs a functional or sequential model).

``tf32_off()``
    A context manager: TF32 matmul and convolution off inside, the prior setting restored on exit and on
    raise (never assumed ``True``).

Import rule
-----------
Nothing imports TensorFlow or Keras at import time (``train.api`` imports this module before the GPU is
configured); every function imports what it needs when it is called.
"""
from __future__ import annotations

import contextlib
import math
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.utils.logger import logger

#: The scale factors of the numeric probe: the ones ``applications.bias_free_denoiser.ddnm.homogeneity_error``
#: uses (copied, not imported: ``train.common`` importing ``applications`` inverts the layering, D-017).
#: NONE is a power of two on purpose (D-015): a power-of-two scale is exact in float32, so it reads 0.0 for any network.
#: ``0.3`` is the sensitive one for a constant offset, whose error scales as ``b (1 - a) / (a |f|)``.
PROBE_ALPHAS: Tuple[float, ...] = (0.3, 1.3, 3.7, 6.1)

# DECISION plan-2026-11-10T095307-d483ffab/D-019: the tolerance is 1e-3, NOT the 1e-2 of ddnm.py and of the
# first plan. Measured at step 2 (scratch_measure/measured.md item f, tiny) and again at step 6 with this module
# (scratch_step6/measured_tolerance.md; tiny, small and base, GPU, TF32 off): every clean family and variant reads
# at most 1.0e-5 (trained or not, on the clone or not); a constant output offset of 1e-3 of the output scale
# reads 2.3e-3 (it PASSES 1e-2) and a trained GELU model reads 1.4e-2 at only two of its four alphas. Any value in
# (clean maximum, smallest planted case) separates them; 1e-3 sits 100x above the clean side and 2.3x below the
# smallest planted one. Do NOT go back to 1e-2 "because DDNM uses it": that is a refuse-to-run bar for a sampler,
# this is the instrument that must see a small offset. Do NOT raise it to make a family pass: a clean family
# reading above 3e-4 is a finding about the family (a TF32 leak, a wide model), not a reason to tune; stop and
# measure.
PROBE_TOLERANCE = 1e-3

#: The kinds :func:`scan_static` reports.
STATIC_KINDS: Tuple[str, ...] = (
    "bias",
    "beta",
    "moving_mean",
    "stock_batchnorm",
    "layernorm",
    "rmsnorm",
    "groupnorm",
    "activation",
    "activation_unknown",
)

#: Weight names (the last path component) that are an additive term, and the kind each reports.
_ADDITIVE_WEIGHTS = {"bias": "bias", "beta": "beta", "moving_mean": "moving_mean"}

#: Weight names of a running variance (stock ``moving_variance`` included).
_RUNNING_VAR_NAMES = ("running_var", "moving_variance")

#: The path tail of a layer scale (``<block>/gamma_scale/gamma``). A classifier keyed on ``"layer_scale"``
#: in the path misses it: the layer is named ``gamma_scale``.
_LAYER_SCALE_TAIL = "gamma_scale/gamma"

#: ``running_var`` of the probe clone is drawn from this interval.
_RUNNING_VAR_RANGE = (0.5, 2.0)


# ---------------------------------------------------------------------
# TF32
# ---------------------------------------------------------------------


@contextlib.contextmanager
def tf32_off() -> Iterator[None]:
    """Run the block with TF32 matmul and convolution OFF, then restore the prior setting.

    On a TF32-capable GPU the default setting leaves a floor of about 3e-4 on a deep network (the TF32 ulp),
    which is the size of the tolerance's margin; the reading is only a statement about the model with TF32
    off. The flag is PROCESS-GLOBAL, so the prior value is captured (never assumed ``True``) and put back in
    ``finally``, also when the block raises. On CPU the flag is inert.

    :yield: nothing.
    """
    import tensorflow as tf

    previous = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        yield
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(previous)


# ---------------------------------------------------------------------
# The static scan
# ---------------------------------------------------------------------


def _allowed_activations() -> frozenset:
    """The positively homogeneous activation names: the library's one allowlist, imported at call time."""
    from dl_techniques.models.vision.convunext.model import POSITIVELY_HOMOGENEOUS_ACTIVATIONS

    return frozenset(a for a in POSITIVELY_HOMOGENEOUS_ACTIVATIONS if a is not None)


def _activation_layer_problem(layer: Any, allowed: frozenset) -> Optional[Tuple[str, str]]:
    """``(kind, detail)`` when ``layer`` is an activation layer outside the allowlist, else ``None``.

    An activation layer is a ``keras.layers.Activation`` (judged by the activation it holds) or any layer
    defined in a ``layers.activations`` module (Keras' and this library's). ``LeakyReLU`` and ``PReLU`` are
    positively homogeneous for any slope; ``ReLU`` is unless it clips (``max_value``) or has a ``threshold``.
    """
    import keras

    if isinstance(layer, keras.layers.Activation):
        return _activation_problem(layer.activation, allowed)
    if ".layers.activations" not in type(layer).__module__:
        return None
    name = type(layer).__name__
    if name in ("LeakyReLU", "PReLU"):
        return None
    if name == "ReLU":
        if getattr(layer, "max_value", None) is None and not getattr(layer, "threshold", 0):
            return None
        return "activation", (
            f"ReLU(max_value={getattr(layer, 'max_value', None)}, threshold={getattr(layer, 'threshold', 0)})"
        )
    return "activation", name


def _activation_problem(act: Any, allowed: frozenset) -> Optional[Tuple[str, str]]:
    """``(kind, detail)`` when the activation ``act`` is not known positively homogeneous, else ``None``.

    ``act`` is what a layer holds: ``None``, a name, a Keras activation function, a layer instance or any
    callable. A callable that is neither is ``activation_unknown`` (its homogeneity is a property of code
    this scan cannot read) and fails the scan.
    """
    import keras

    if act is None:
        return None
    if isinstance(act, str):
        return None if act in allowed else ("activation", act)
    if isinstance(act, keras.layers.Layer):
        return _activation_layer_problem(act, allowed)
    name = getattr(act, "__name__", None)
    if name:
        try:
            registered = keras.activations.get(name) is act
        except (ValueError, TypeError):
            registered = False
        if registered:
            return None if name in allowed else ("activation", name)
    return "activation_unknown", repr(act)


def _norm_kind(layer: Any) -> Optional[str]:
    """The normalisation kind a layer reports, or ``None``: scale-invariant norms are degree 0, not 1."""
    import keras

    if isinstance(layer, keras.layers.BatchNormalization):
        return "stock_batchnorm"
    if isinstance(layer, keras.layers.LayerNormalization):
        return "layernorm"
    if isinstance(layer, keras.layers.GroupNormalization):
        return "groupnorm"
    for base in type(layer).__mro__:
        if "LayerNorm" in base.__name__:
            return "layernorm"
        if "RMS" in base.__name__:
            return "rmsnorm"
        if "GroupNorm" in base.__name__:
            return "groupnorm"
    return None


def scan_static(model: Any) -> Dict[str, Any]:
    """List what in ``model`` can break degree-1 homogeneity (see the module docstring for the report)."""
    if not getattr(model, "built", False):
        raise ValueError("scan_static needs a built model: its weights cannot be read before build")
    allowed = _allowed_activations()
    offenders: List[Dict[str, str]] = []
    seen = set()

    def add(kind: str, layer: Any, detail: str) -> None:
        # One row per (kind, layer OBJECT): a use_bias flag and its bias weight are one finding, and two
        # layers that share a name (every ConvNeXt block names its activation ``activation``) are two.
        if (kind, id(layer)) not in seen:
            seen.add((kind, id(layer)))
            offenders.append({"kind": kind, "layer": layer.name, "detail": detail})

    layer_scales: List[str] = []
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            add("bias", layer, "use_bias=True")
        norm = _norm_kind(layer)
        if norm is not None:
            add(norm, layer, type(layer).__name__)
        # an activation layer is judged by its class, any other layer by the ``activation`` it holds
        problem = _activation_layer_problem(layer, allowed)
        if problem is None and hasattr(layer, "activation"):
            problem = _activation_problem(layer.activation, allowed)
        if problem is not None:
            add(problem[0], layer, problem[1])
        # the layer's OWN weights (``_trainable_variables`` and ``_non_trainable_variables``; a layer's
        # ``weights`` also lists its sublayers', which the walk reaches itself)
        for weight in (*layer._trainable_variables, *layer._non_trainable_variables):
            kind = _ADDITIVE_WEIGHTS.get(weight.path.rpartition("/")[2])
            if kind is not None:
                add(kind, layer, weight.path)
            if weight.path.endswith(_LAYER_SCALE_TAIL):
                layer_scales.append(weight.path)

    counts: Dict[str, int] = {}
    for row in offenders:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return {
        "offenders": offenders,
        "counts": counts,
        "layer_scales": layer_scales,
        "passed": not offenders,
    }


# ---------------------------------------------------------------------
# The numeric probe
# ---------------------------------------------------------------------


def _forward(model: Any, x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """``(output as float64, whether a list/tuple output was reduced to index 0)`` at ``training=False``."""
    import keras

    out = model(x, training=False)
    if isinstance(out, dict):
        raise ValueError(
            f"probe_numeric cannot choose among the outputs of a dict ({sorted(out)}): "
            "pass a model with one tensor output or a list whose index 0 is the one to read"
        )
    reduced = isinstance(out, (list, tuple))
    if reduced:
        out = out[0]
    return np.asarray(keras.ops.convert_to_numpy(out), dtype=np.float64), reduced


def probe_numeric(
    model: Any,
    inputs: Any,
    alphas: Sequence[float] = PROBE_ALPHAS,
    tolerance: float = PROBE_TOLERANCE,
) -> Dict[str, Any]:
    """Read ``max|f(a x) - a f(x)| / max|a f(x)|`` at every ``alpha`` (see the module docstring)."""
    alphas = tuple(float(a) for a in alphas)
    if not alphas or not all(math.isfinite(a) and a > 0 for a in alphas):
        raise ValueError(f"alphas must be a non-empty sequence of positive finite numbers, got {alphas!r}")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be positive and finite, got {tolerance!r}")
    import tensorflow as tf

    x = np.asarray(inputs, dtype=np.float32)
    per_alpha: List[float] = []
    with tf32_off():
        in_force = bool(tf.config.experimental.tensor_float_32_execution_enabled())
        y, reduced = _forward(model, x)
        for alpha in alphas:
            y_alpha, _ = _forward(model, np.float32(alpha) * x)
            expected = alpha * y
            scale = float(np.max(np.abs(expected)))
            error = float(np.max(np.abs(y_alpha - expected)))
            value = error / scale if scale > 0 else math.inf
            per_alpha.append(value if math.isfinite(value) else math.inf)
    worst = max(per_alpha)
    report: Dict[str, Any] = {
        "alphas": list(alphas),
        "max_rel": worst,
        "per_alpha": per_alpha,
        "tolerance": float(tolerance),
        "passed": bool(worst <= tolerance),
        "tf32": in_force,
    }
    if reduced:
        report["output_index"] = 0
    return report


# ---------------------------------------------------------------------
# The probe clone
# ---------------------------------------------------------------------


def _seed_generators(model: Any) -> List[Any]:
    """Every ``keras.random.SeedGenerator`` reachable from ``model`` (Keras 3.8: ``Layer._seed_generators``), once each."""
    found: List[Any] = []
    seen = set()
    for layer in model._flatten_layers():
        for generator in getattr(layer, "_seed_generators", ()):
            if id(generator) not in seen:
                seen.add(id(generator))
                found.append(generator)
    return found


def _capture_random_state(model: Any) -> Dict[str, Any]:
    """The random state a clone must not move: the model's generators, Keras' global generator, Python's and NumPy's RNG."""
    import keras

    # Keras 3.8 keeps its global generator in ``global_state`` (private; ``tests/conftest.py`` reads it the same way)
    global_generator = keras.src.backend.common.global_state.get_global_attribute("global_seed_generator")
    numpy_state = np.random.get_state()
    return {
        "generators": [(g, np.array(keras.ops.convert_to_numpy(g.state))) for g in _seed_generators(model)],
        "global": (
            None
            if global_generator is None
            else (global_generator, np.array(keras.ops.convert_to_numpy(global_generator.state)))
        ),
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
    }


def _restore_random_state(state: Dict[str, Any]) -> None:
    """Put back what :func:`_capture_random_state` read."""
    for generator, saved in state["generators"]:
        generator.state.assign(saved)
    if state["global"] is not None:
        generator, saved = state["global"]
        generator.state.assign(saved)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _build_probe_clone(model: Any, seed: int) -> Any:
    """The clone of :func:`init_probe_clone`, WITHOUT the random-state restore (the caller restores)."""
    import keras

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    rng = np.random.default_rng(seed)
    scales = variances = 0
    for weight in clone.weights:
        path = weight.path
        if path.endswith(_LAYER_SCALE_TAIL):
            weight.assign(np.ones(tuple(weight.shape), dtype=np.dtype(weight.dtype)))
            scales += 1
        elif path.rpartition("/")[2] in _RUNNING_VAR_NAMES:
            drawn = rng.uniform(*_RUNNING_VAR_RANGE, size=tuple(weight.shape))
            weight.assign(drawn.astype(np.dtype(weight.dtype)))
            variances += 1
    logger.info(f"homogeneity probe clone: {scales} layer scales set to 1, {variances} running variances redrawn")
    return clone


# DECISION plan-2026-11-10T095307-d483ffab/D-015: the clone is PROBED, the model never is, and the random state
# is restored. Building a clone draws from Python's ``random`` (every layer built without a seed asks
# ``keras.random.SeedGenerator`` for a default one) and can advance a generator; a diagnostic pass that moved the
# dropout stream once shifted a same-seed accuracy by 0.004 (plan-2026-09-18T213948-68dcb72c D-031). Do NOT drop
# the restore because "the weights are unchanged": the run is then no longer reproducible against a run made
# without the probe. Do NOT probe the live model after assigning layer scales to it: the run would train from the
# probe's weights. Guard: ``test_init_probe_clone_leaves_every_random_state_where_it_found_it``.
def init_probe_clone(model: Any, seed: int = 0) -> Any:
    """A clone of ``model`` made readable by :func:`probe_numeric` (see the module docstring for the contract).

    Two things hide a break at the weights a model has at init, and the clone removes both without touching
    the model: a layer scale starts at 1e-4 to 1e-6, which mutes the whole branch it scales, and a
    ``running_var`` starts at 1, which makes a normalisation a constant and so hides a LayerNorm (reads 1.4e-5
    at init and 0.18 on the clone). Every other weight is copied.

    :param model: a built functional or sequential ``keras.Model``.
    :param seed: the seed of the ``running_var`` draw (a local ``numpy`` generator, never the global one).
    :return: the clone.
    """
    state = _capture_random_state(model)
    try:
        return _build_probe_clone(model, seed)
    finally:
        _restore_random_state(state)
