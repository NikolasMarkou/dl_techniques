"""
Attribution + faithfulness helpers for the E3 ``LearnableNeuralCircuit``
experiment.

Plan: plan_2026-05-13_798d3a60.

This module provides three attribution methods that all return a 1-D numpy
array of length ``num_input_bits``:

  - ``circuit_attributions``: in-model gradient × input. The circuit is
    differentiable end-to-end so the gradient of the (scalar) output with
    respect to each input bit, scaled by the input value, is a natural
    "circuit-native" attribution. On a symmetric task such as
    ``parity_k6`` a well-trained circuit will give roughly equal magnitude
    to every input bit — which is the symmetry oracle used by SC6.

  - ``lime_attributions``: ``lime.lime_tabular.LimeTabularExplainer`` on a
    keras model treated as a black-box classifier. Returns the
    feature-importance vector for the predicted class.

  - ``shap_attributions``: ``shap.KernelExplainer`` on the same black-box
    callable, with a small background set; returns SHAP values for the
    predicted class.

Plus the faithfulness/quality metrics defined by the E3 spec:

  - ``suff_comp_aucs``: insertion/deletion AUC under a top-k attribution
    mask. Sufficiency = accuracy when only the top-k features are present
    (others held at ``baseline``). Comprehensiveness = drop in confidence
    when the top-k features are removed.

  - ``sparsity``: normalized entropy of ``|attribution|``. Returns a value
    in ``[0, 1]`` where ``0`` = one-hot (perfectly sparse) and ``1`` =
    uniform mass.

  - ``stability``: mean cosine similarity between the attribution on the
    original input and attributions on its 1-bit-flip neighbours.

Design constraint: all three attribution functions return a vector with a
consistent sign convention (positive => evidence FOR the predicted class)
and a consistent shape so they are directly comparable in the
faithfulness/sparsity/stability metrics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Attribution methods
# ---------------------------------------------------------------------

def _ensure_2d_input(x: np.ndarray) -> np.ndarray:
    """Promote a 1-D input vector to a (1, num_bits) batch."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x


def _predict_class_prob(model: keras.Model, x: np.ndarray) -> Tuple[int, float]:
    """Run model on a single sample and return (predicted_class, prob).

    Works for either binary sigmoid head (shape (B,1)) — class is 0/1
    relative to threshold 0.5 — or a softmax/sigmoid multi-output head.
    For E3 tasks all 3 generators are binary single-output so we treat
    the head as scalar logistic.
    """
    x = _ensure_2d_input(x)
    pred = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    p = float(pred.ravel()[0])
    return (1 if p >= 0.5 else 0), p


def circuit_attributions(
    model: keras.Model,
    x_single: np.ndarray,
    baseline: float = 0.5,
    n_steps: int = 32,
) -> np.ndarray:
    """Integrated-gradient attribution for the circuit head.

    The circuit is differentiable end-to-end, so we use integrated
    gradients along a straight path from ``baseline`` (default 0.5 — the
    "uninformative" mid-point between 0 and 1) to the input. Integrated
    gradients (Sundararajan et al. 2017) satisfy two properties relevant
    here:

      - **Symmetry-preserving**: if the model is invariant under a
        permutation of input dimensions (as a converged parity model is),
        the attribution vector inherits that invariance after averaging
        over inputs. This is exactly the SC6 oracle.
      - **Completeness**: the per-bit attributions sum to
        ``f(x) - f(baseline)`` for the predicted class — useful for
        ranking but not strictly required here.

    Returned shape: ``(num_input_bits,)``. Positive entries push the
    prediction toward the predicted class.
    """
    x = _ensure_2d_input(x_single)
    x_t = tf.constant(x, dtype=tf.float32)
    b_t = tf.constant(np.full_like(x, baseline, dtype=np.float32))
    # Build the path from baseline to x in n_steps points (exclusive of
    # the baseline so 1..n_steps).
    alphas = tf.reshape(tf.linspace(1.0 / n_steps, 1.0, n_steps), (n_steps, 1, 1))
    path = b_t[None, ...] + alphas * (x_t[None, ...] - b_t[None, ...])
    path = tf.reshape(path, (n_steps, x.shape[1]))
    with tf.GradientTape() as tape:
        tape.watch(path)
        out = model(path, training=False)
        scalar = tf.reduce_sum(out)
    grads = tape.gradient(scalar, path)
    avg_grad = tf.reduce_mean(grads, axis=0).numpy().astype(np.float32)
    attribution = avg_grad * (x[0] - baseline)
    # Align sign so positive = evidence for predicted class.
    cls, _ = _predict_class_prob(model, x)
    if cls == 0:
        attribution = -attribution
    return attribution


def lime_attributions(
    model: keras.Model,
    x_single: np.ndarray,
    num_samples: int = 5000,
    feature_names: Optional[Sequence[str]] = None,
    training_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """LIME tabular attribution. Returns ``(num_input_bits,)``.

    For a binary head we treat ``model.predict`` as
    ``[P(class=0), P(class=1)]`` and ask LIME to explain the predicted class.
    """
    import lime
    import lime.lime_tabular  # noqa: F401

    x = _ensure_2d_input(x_single)
    num_bits = int(x.shape[1])
    if training_data is None:
        # 64-sample default background — enough for LIME's discretizer to
        # see both 0s and 1s on each bit without slowing down too much.
        training_data = np.random.RandomState(0).randint(0, 2, size=(64, num_bits)).astype(np.float32)

    def proba_fn(arr: np.ndarray) -> np.ndarray:
        pred = np.asarray(model.predict(arr.astype(np.float32), verbose=0), dtype=np.float32).ravel()
        # Convert sigmoid output to 2-class probability vector.
        return np.stack([1.0 - pred, pred], axis=-1)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names or [f"bit_{i}" for i in range(num_bits)],
        class_names=["0", "1"],
        discretize_continuous=False,
    )
    cls, _ = _predict_class_prob(model, x)
    exp = explainer.explain_instance(
        x[0], proba_fn, num_features=num_bits, num_samples=int(num_samples),
        labels=(cls,),
    )
    importance = np.zeros(num_bits, dtype=np.float32)
    for feat_idx, weight in exp.as_map()[cls]:
        importance[int(feat_idx)] = float(weight)
    return importance


def shap_attributions(
    model: keras.Model,
    x_single: np.ndarray,
    background: Optional[np.ndarray] = None,
    nsamples: int = 256,
) -> np.ndarray:
    """SHAP KernelExplainer attribution. Returns ``(num_input_bits,)``."""
    import shap

    x = _ensure_2d_input(x_single)
    num_bits = int(x.shape[1])
    if background is None:
        background = np.random.RandomState(0).randint(0, 2, size=(16, num_bits)).astype(np.float32)

    def f(arr: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(arr.astype(np.float32), verbose=0), dtype=np.float32).ravel()

    explainer = shap.KernelExplainer(f, background)
    sv = explainer.shap_values(x, nsamples=int(nsamples), silent=True)
    sv = np.asarray(sv, dtype=np.float32).reshape(-1)
    if sv.shape[0] != num_bits:
        # Some shap versions stack class dim — squeeze it.
        sv = sv[:num_bits]
    cls, _ = _predict_class_prob(model, x)
    if cls == 0:
        sv = -sv
    return sv


# ---------------------------------------------------------------------
# Faithfulness / quality metrics
# ---------------------------------------------------------------------

def _topk_mask(attribution: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean mask of the top-k positions by ``|attribution|``."""
    if k <= 0:
        return np.zeros_like(attribution, dtype=bool)
    if k >= attribution.size:
        return np.ones_like(attribution, dtype=bool)
    order = np.argsort(-np.abs(attribution))
    mask = np.zeros_like(attribution, dtype=bool)
    mask[order[:k]] = True
    return mask


def suff_comp_aucs(
    model: keras.Model,
    X: np.ndarray,
    attribution_fn: Callable[[keras.Model, np.ndarray], np.ndarray],
    k_range: Optional[Sequence[int]] = None,
    baseline: float = 0.5,
) -> Dict[str, Any]:
    """Sufficiency and comprehensiveness AUCs.

    For each sample in ``X`` and each ``k`` in ``k_range``:
      - Sufficiency curve point: confidence of the predicted class when
        ONLY the top-k features are retained (others set to ``baseline``).
      - Comprehensiveness curve point: drop in predicted-class confidence
        when the top-k features are REMOVED (set to ``baseline``).

    AUCs are computed by trapezoidal integration over ``k`` normalized to
    ``[0, 1]``. Higher sufficiency AUC = small explanation already enough
    (faithful). Higher comprehensiveness AUC = removing the explanation
    breaks the prediction (faithful).

    Returns a dict with ``suff_auc``, ``comp_auc``, ``k_curves``.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    num_bits = X.shape[1]
    if k_range is None:
        # Default: every k from 0..num_bits step 1 for small bit vectors.
        k_range = list(range(0, num_bits + 1))
    else:
        k_range = list(k_range)

    suff_pts: List[float] = []
    comp_pts: List[float] = []
    per_sample_suff: List[List[float]] = []
    per_sample_comp: List[List[float]] = []

    for x in X:
        cls, p_orig = _predict_class_prob(model, x)
        attr = attribution_fn(model, x)
        suff_curve: List[float] = []
        comp_curve: List[float] = []
        for k in k_range:
            mask = _topk_mask(attr, k)
            # Sufficiency: keep top-k, baseline everywhere else.
            x_suff = np.where(mask, x, baseline).astype(np.float32)
            _, p_suff = _predict_class_prob(model, x_suff)
            # Comprehensiveness: baseline the top-k, keep the rest.
            x_comp = np.where(mask, baseline, x).astype(np.float32)
            _, p_comp = _predict_class_prob(model, x_comp)
            # Always express as class-aligned confidence.
            if cls == 1:
                suff_curve.append(p_suff)
                comp_curve.append(p_orig - p_comp)
            else:
                suff_curve.append(1.0 - p_suff)
                comp_curve.append((1.0 - p_orig) - (1.0 - p_comp))
        per_sample_suff.append(suff_curve)
        per_sample_comp.append(comp_curve)

    suff_curve_mean = np.mean(np.asarray(per_sample_suff, dtype=np.float32), axis=0)
    comp_curve_mean = np.mean(np.asarray(per_sample_comp, dtype=np.float32), axis=0)
    ks_norm = np.asarray(k_range, dtype=np.float32) / max(1, num_bits)
    suff_auc = float(np.trapz(suff_curve_mean, ks_norm))
    comp_auc = float(np.trapz(comp_curve_mean, ks_norm))
    return {
        "suff_auc": suff_auc,
        "comp_auc": comp_auc,
        "k_curves": {
            "k": list(k_range),
            "suff": suff_curve_mean.tolist(),
            "comp": comp_curve_mean.tolist(),
        },
    }


def sparsity(attribution: np.ndarray) -> float:
    """Normalized entropy of ``|attribution|``.

    Returns ``1.0`` for a uniform vector (max entropy => zero sparsity in
    the colloquial sense) and ``0.0`` for a one-hot vector. The naming
    follows information-theoretic sparsity ("how spread is the mass").
    """
    a = np.abs(np.asarray(attribution, dtype=np.float64))
    s = a.sum()
    if s == 0 or a.size <= 1:
        return 0.0
    p = a / s
    # Shannon entropy in nats, normalized to log(N).
    eps = 1e-12
    h = float(-(p * np.log(p + eps)).sum())
    return h / float(np.log(a.size))


def stability(
    model: keras.Model,
    x_single: np.ndarray,
    attribution_fn: Callable[[keras.Model, np.ndarray], np.ndarray],
    num_perturbations: int = 10,
    seed: int = 0,
) -> float:
    """Mean cosine similarity between attribution on ``x_single`` and on
    its 1-bit-flip neighbours.

    For a binary vector, we cycle through up to ``num_perturbations`` bit
    flips chosen at random without replacement (limited by length). For
    ``num_perturbations >= num_bits`` every bit is flipped once.
    Returns a scalar in ``[-1, 1]``; higher = more stable.
    """
    x = _ensure_2d_input(x_single)[0]
    num_bits = int(x.shape[0])
    a0 = attribution_fn(model, x)
    n0 = np.linalg.norm(a0) + 1e-12
    rng = np.random.RandomState(seed)
    n_perturb = min(num_perturbations, num_bits)
    if n_perturb <= 0:
        return float("nan")
    idxs = rng.choice(num_bits, size=n_perturb, replace=False)
    sims: List[float] = []
    for i in idxs:
        x_flip = x.copy()
        x_flip[i] = 1.0 - x_flip[i]
        a = attribution_fn(model, x_flip)
        n = np.linalg.norm(a) + 1e-12
        sims.append(float(np.dot(a, a0) / (n * n0)))
    return float(np.mean(sims))
