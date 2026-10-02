"""LightGlue training loss (per-layer assignment NLL plus token-confidence BCE).

Reference: cvg/glue-factory ``gluefactory/models/matchers/lightglue.py``
(``LightGlue.loss``, ``TokenConfidence.loss``) and
``gluefactory/models/utils/losses.py`` (``NLLLoss``, ``weight_loss``), read on
2026-10-02. What the source does, and what this module does:

1. **Per-layer NLL.** For layer ``i`` with log assignment ``la`` of shape
   ``(M+1, N+1)`` (dustbin = last row and column) a weight matrix ``W`` selects the
   supervised entries: ``W[i, j] = 1`` for a matched pair, ``W[i, N] = 1`` for a
   keypoint of image 0 with label ``-1``, ``W[M, j] = 1`` for a keypoint of image 1
   with label ``-1``. Ignored and padded keypoints (label ``-2``) have no weight.
   ``nll_pos = -sum(la * W_pos) / max(#pos, 1)``;
   ``nll_neg = -sum(la * W_neg) / (max(#neg0, 1) + max(#neg1, 1))``;
   ``nll = b * nll_pos + (1 - b) * nll_neg`` with ``b = nll_balancing`` (0.5).
2. **Layer weighting.** The final layer has weight 1, layer ``i < L - 1`` has weight
   ``gamma ** (L - 1 - i)`` (``gamma > 0``) or ``i + 1`` (``gamma <= 0``); the total
   is ``sum(w_i * nll_i) / sum(w_i)``. The reference default is ``gamma = 1.0``, i.e.
   a plain mean over layers. (The plan said "mean over layers"; this is the same at
   the default and is the general form of the source.)
3. **Token confidence BCE.** For each layer ``i < L - 1`` the target of a keypoint of
   image 0 is ``argmax_j la_final[i, :] == argmax_j la_i[i, :]`` (argmax over ALL
   ``N + 1`` columns, dustbin included), the target of a keypoint of image 1 is the
   same over columns of the dustbin-inclusive row axis; both are computed on detached
   log assignments. The term is ``(mean_tokens BCE0 + mean_tokens BCE1) / 2`` averaged
   over the ``L - 1`` layers and ADDED UNSCALED to the total in training. The
   confidence head itself reads detached descriptors, so it trains only its own head.
   The target does not use ground-truth labels at all.

Divergences from the source, all deliberate:

* the reference has no padding; here padded entries are excluded from the NLL by
  label ``-2``, and from the confidence term by optional ``mask0``/``mask1`` (without
  masks the confidence term assumes no padding, exactly like the reference; with
  padded batches pass the masks, because padded log-assignment entries are 0 and would
  otherwise win the argmax);
* the BCE is computed from the pre-sigmoid LOGITS in the numerically stable form
  ``max(z, 0) - z * t + log1p(exp(-|z|))``, i.e. exactly ``BCEWithLogitsLoss`` as in the
  reference (the model exposes them as ``token_logits0/1``). An earlier version clipped
  the sigmoid output to ``[1e-7, 1 - 1e-7]``, whose gradient is exactly 0 for a logit
  beyond about 16.6 (a confidently wrong head could never recover); the logit form has
  gradient ``sigmoid(z) - t`` everywhere (D-018);
* ``nll_pos`` is taken from ``matches0`` (the reference carries a separate
  ``gt_assignment`` tensor; ``utils.keypoint_matching`` guarantees both agree);
* the focal knob ``gamma_f`` (default 0, unused by ``weight_loss``) is not provided.

Keras interface (probed on Keras 3.8): a ``Loss`` object called directly accepts dict
``y_true``/``y_pred``, but stock ``fit`` with ``compile(loss=obj)`` and a dict ``y``
raises ``TypeError`` (the dict keys are read as output names). Therefore
:meth:`LightGlueLoss.call` takes PACKED tensors: ``y_true`` is the int32 matrix
``(B, M + N)`` from :func:`pack_matches` and ``y_pred`` is ``log_assignments``
``(B, L, M+1, N+1)``. ``Loss.__call__`` casts both to the loss dtype (float32), so
labels arrive as float32 and are cast back to int32 exactly (values far below 2**24).
``call`` can only see ``log_assignments``, so it returns the NLL term alone; the full
objective including the confidence term is :meth:`LightGlueLoss.compute`, meant for
``add_loss`` in a pipeline model (D-001).

All maths is ``keras.ops`` only, float32 inside whatever the input dtype, and
graph-safe. An all-ignored batch yields exactly 0 (not NaN).
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


# DECISION plan-2026-10-02T084508-dd2c07ac/D-018
# The confidence BCE is computed from LOGITS in this stable form. Do NOT go back to
# clipping sigmoid probabilities to [1e-7, 1 - 1e-7]: the gradient is exactly 0 beyond a
# logit of about 16.6, so a confidently wrong head could never recover. Guards:
# tests/test_losses/test_lightglue_loss.py::TestConfidence::
# test_gradient_does_not_die_at_saturation and
# test_saturation_guard_is_red_for_the_clipped_probability_form. See decisions.md D-018.
def _bce_with_logits(logits: Any, target: Any) -> Any:
    """Elementwise ``BCEWithLogits``, stable for any finite logit.

    ``max(z, 0) - z * t + log1p(exp(-|z|))``; the gradient with respect to ``z`` is
    ``sigmoid(z) - t``, never clipped to zero. Module-level so a test can replace it
    with a clipped-probability version and watch the saturation guard go RED.

    :param logits: Float32 logits, any shape.
    :param target: Float32 targets in ``{0, 1}``, same shape.
    :return: Float32 loss, same shape.
    """
    ops = keras.ops
    return ops.relu(logits) - logits * target + ops.log1p(ops.exp(-ops.abs(logits)))


def pack_matches(matches0: Any, matches1: Any) -> Any:
    """Concatenate the two label vectors into one int32 ``(B, M + N)`` matrix.

    :param matches0: ``(B, M)`` ints, ``j`` match, ``-1`` dustbin, ``-2`` ignored.
    :param matches1: ``(B, N)`` ints with the same meaning.
    :return: int32 ``(B, M + N)``; ``M`` is recovered from ``log_assignments`` shape.
    """
    return keras.ops.concatenate(
        [keras.ops.cast(matches0, "int32"), keras.ops.cast(matches1, "int32")], axis=1)


def unpack_matches(packed: Any, m: int) -> Tuple[Any, Any]:
    """Inverse of :func:`pack_matches` given the static ``M``.

    :param packed: ``(B, M + N)`` labels (any numeric dtype; cast to int32).
    :param m: Number of image-0 keypoints.
    :return: ``(matches0 (B, M), matches1 (B, N))`` int32.
    """
    if keras.backend.is_float_dtype(packed.dtype):
        packed = keras.ops.round(packed)
    packed = keras.ops.cast(packed, "int32")
    return packed[:, :m], packed[:, m:]


def _static_dims(log_assignments: Any) -> Tuple[int, int, int]:
    shape = tuple(log_assignments.shape)
    if len(shape) != 4 or any(d is None for d in shape[1:]):
        raise ValueError(
            "log_assignments must be (B, L, M+1, N+1) with static L, M, N, "
            f"got shape {shape}")
    return shape[1], shape[2] - 1, shape[3] - 1


def layer_weights(num_layers: int, gamma: float) -> Any:
    """Per-layer NLL weights of the reference (final layer weight 1).

    :param num_layers: ``L``.
    :param gamma: ``> 0`` gives ``gamma ** (L - 1 - i)``; ``<= 0`` gives ``i + 1`` for
        ``i < L - 1``.
    :return: Python list of ``L`` floats.
    """
    if gamma > 0.0:
        return [float(gamma) ** (num_layers - 1 - i) for i in range(num_layers)]
    return [float(i + 1) for i in range(num_layers - 1)] + [1.0]


def lightglue_nll(
    log_assignments: Any,
    matches0: Any,
    matches1: Any,
    gamma: float = 1.0,
    nll_balancing: float = 0.5,
) -> Any:
    """Layer-weighted balanced assignment NLL, one value per sample.

    Interface contract: ``log_assignments`` ``(B, L, M+1, N+1)`` with static ``L, M,
    N`` (dustbin last); ``matches0`` ``(B, M)`` and ``matches1`` ``(B, N)`` integer
    labels (``j`` match, ``-1`` dustbin, ``-2`` ignored or padded). Returns float32
    ``(B,)``. Entries with zero weight never reach the arithmetic (selected with
    ``where``), so non-finite values there cannot poison the result. A sample with no
    supervised entry yields 0.

    :param log_assignments: Log assignment matrices of all layers.
    :param matches0: Labels of image 0.
    :param matches1: Labels of image 1.
    :param gamma: Layer weighting, see :func:`layer_weights`.
    :param nll_balancing: Weight ``b`` of the matched-pair term, ``1 - b`` for the
        dustbin term.
    :return: Per-sample loss ``(B,)``.
    """
    ops = keras.ops
    num_layers, m, n = _static_dims(log_assignments)
    la = ops.cast(log_assignments, "float32")
    m0 = ops.cast(matches0, "int32")
    m1 = ops.cast(matches1, "int32")

    pos = ops.one_hot(ops.maximum(m0, 0), n, dtype="float32") \
        * ops.cast(m0 >= 0, "float32")[..., None]                       # (B, M, N)
    neg0 = ops.cast(m0 == -1, "float32")                                # (B, M)
    neg1 = ops.cast(m1 == -1, "float32")                                # (B, N)

    zero = ops.zeros_like(la)
    la_pos = ops.where(pos[:, None] > 0.0, la[:, :, :m, :n], zero[:, :, :m, :n])
    la_neg0 = ops.where(neg0[:, None] > 0.0, la[:, :, :m, n], zero[:, :, :m, n])
    la_neg1 = ops.where(neg1[:, None] > 0.0, la[:, :, m, :n], zero[:, :, m, :n])

    num_pos = ops.maximum(ops.sum(pos, axis=(1, 2)), 1.0)               # (B,)
    num_neg0 = ops.maximum(ops.sum(neg0, axis=1), 1.0)
    num_neg1 = ops.maximum(ops.sum(neg1, axis=1), 1.0)

    nll_pos = -ops.sum(la_pos, axis=(2, 3)) / num_pos[:, None]          # (B, L)
    nll_neg = -(ops.sum(la_neg0, axis=2) + ops.sum(la_neg1, axis=2)) \
        / (num_neg0 + num_neg1)[:, None]
    nll = nll_balancing * nll_pos + (1.0 - nll_balancing) * nll_neg     # (B, L)

    weights = layer_weights(num_layers, gamma)
    w = ops.convert_to_tensor(weights, dtype="float32")
    return ops.sum(nll * w[None, :], axis=1) / float(sum(weights))


def lightglue_confidence_loss(
    log_assignments: Any,
    token_logits0: Any,
    token_logits1: Any,
    mask0: Optional[Any] = None,
    mask1: Optional[Any] = None,
) -> Any:
    """Token-confidence BCE-with-logits against "layer i agrees with the final layer".

    Interface contract: ``log_assignments`` ``(B, L, M+1, N+1)``;
    ``token_logits0`` ``(B, L-1, M)`` and ``token_logits1`` ``(B, L-1, N)`` are the
    PRE-sigmoid confidence logits (the model's ``token_logits0/1``; a probability here
    would be a silent error, apply ``log(p / (1 - p))`` first); ``mask0`` ``(B, M)`` / ``mask1`` ``(B, N)`` mark
    real keypoints (default: all real, as in the reference). Targets come from detached
    log assignments, argmax over the dustbin-inclusive axis. Padded columns/rows are
    pushed to a large negative value before the argmax so they cannot win. Returns
    float32 ``(B,)``; 0 when ``L == 1`` or no real keypoint.

    :param log_assignments: Log assignments of all layers.
    :param token_logits0: Confidence logits of image-0 tokens for layers ``0..L-2``.
    :param token_logits1: Confidence logits of image-1 tokens for layers ``0..L-2``.
    :param mask0: Optional real-keypoint mask of image 0.
    :param mask1: Optional real-keypoint mask of image 1.
    :return: Per-sample loss ``(B,)``.
    """
    ops = keras.ops
    num_layers, m, n = _static_dims(log_assignments)
    batch = ops.shape(log_assignments)[0]
    if num_layers < 2:
        return ops.zeros((batch,), dtype="float32")

    la = ops.stop_gradient(ops.cast(log_assignments, "float32"))
    real0 = ops.ones((batch, m), "float32") if mask0 is None \
        else ops.cast(mask0, "float32")
    real1 = ops.ones((batch, n), "float32") if mask1 is None \
        else ops.cast(mask1, "float32")
    # DECISION plan-2026-10-02T084508-dd2c07ac/D-012
    # Padded rows and columns are pushed to -big BEFORE the argmax. Do NOT drop the masks:
    # padded log-assignment entries are 0 (above every real, negative entry), so they would
    # win the argmax and corrupt the target. The reference has no padding; this is our
    # extension. Guard: tests/test_losses/test_lightglue_loss.py::TestConfidence::
    # test_matches_oracle_with_padding and tests/test_train/test_lightglue/test_pipeline.py::
    # test_masks_reach_the_confidence_term. See decisions.md D-012.
    big = 1e9
    # extended masks include the dustbin (always selectable)
    col_ok = ops.concatenate([real1, ops.ones((batch, 1), "float32")], axis=1)  # (B,N+1)
    row_ok = ops.concatenate([real0, ops.ones((batch, 1), "float32")], axis=1)  # (B,M+1)
    la_rows = ops.where(col_ok[:, None, None, :] > 0.0, la, -big)       # for axis -1 argmax
    la_cols = ops.where(row_ok[:, None, :, None] > 0.0, la, -big)       # for axis -2 argmax

    row_arg = ops.argmax(la_rows[:, :, :m, :], axis=-1)                  # (B, L, M)
    col_arg = ops.argmax(la_cols[:, :, :, :n], axis=-2)                  # (B, L, N)
    correct0 = ops.cast(row_arg[:, :-1] == row_arg[:, -1:], "float32")   # (B, L-1, M)
    correct1 = ops.cast(col_arg[:, :-1] == col_arg[:, -1:], "float32")

    def _bce(logit: Any, target: Any, real: Any) -> Any:
        bce = _bce_with_logits(ops.cast(logit, "float32"), target)
        r = real[:, None, :]                                             # (B,1,T)
        denom = ops.maximum(ops.sum(r, axis=-1), 1.0)                    # (B,1)
        return ops.sum(bce * r, axis=-1) / denom                         # (B, L-1)

    per_layer = (_bce(token_logits0, correct0, real0)
                 + _bce(token_logits1, correct1, real1)) / 2.0
    return ops.sum(per_layer, axis=1) / float(num_layers - 1)


@register_dl_technique("dl_techniques.losses.lightglue_loss")
class LightGlueLoss(keras.losses.Loss):
    """LightGlue objective: layer-weighted balanced NLL plus token-confidence BCE.

    **Intent**: supervise every layer's log assignment with the homography-derived
    labels of ``utils.keypoint_matching`` and teach the confidence heads to predict
    whether a layer already agrees with the final one. See the module docstring for
    the exact formulation and the divergences from glue-factory.

    **Two entry points**:

    * :meth:`call` (``compile(loss=LightGlueLoss())`` with stock ``fit``): packed
      ``y_true`` int ``(B, M + N)`` from :func:`pack_matches`, ``y_pred`` =
      ``log_assignments``; returns the NLL term only (the confidence term needs the
      token logits, which are not part of a single ``y_pred`` tensor).
    * :meth:`compute` (``add_loss`` in a pipeline): labels, ``log_assignments`` and
      token confidence logits; returns the full per-sample objective
      ``nll + confidence_weight * confidence``.

    :param gamma: Layer weighting, ``> 0`` gives ``gamma ** (L - 1 - i)``, ``<= 0``
        gives ``i + 1``; final layer weight 1. Default 1.0 (plain mean).
    :param nll_balancing: Weight of the matched-pair term versus the dustbin term.
    :param confidence_weight: Scale of the confidence BCE in :meth:`compute`
        (reference: 1.0; 0 disables it).
    :param name: Loss name.
    :param kwargs: Forwarded to :class:`keras.losses.Loss` (e.g. ``reduction``).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        nll_balancing: float = 0.5,
        confidence_weight: float = 1.0,
        name: str = "lightglue_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if not 0.0 <= nll_balancing <= 1.0:
            raise ValueError(f"nll_balancing must be in [0, 1], got {nll_balancing}")
        if confidence_weight < 0.0:
            raise ValueError(f"confidence_weight must be >= 0, got {confidence_weight}")
        self.gamma = float(gamma)
        self.nll_balancing = float(nll_balancing)
        self.confidence_weight = float(confidence_weight)
        logger.info(
            f"LightGlueLoss initialized (gamma={gamma}, nll_balancing={nll_balancing}, "
            f"confidence_weight={confidence_weight})")

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """NLL term from packed labels.

        :param y_true: ``(B, M + N)`` packed labels (see :func:`pack_matches`); arrives
            as float32 after ``Loss.__call__`` and is cast back to int32.
        :param y_pred: ``log_assignments`` ``(B, L, M+1, N+1)``.
        :return: Per-sample NLL ``(B,)``; the parent reduction yields the scalar.
        """
        _, m, _ = _static_dims(y_pred)
        matches0, matches1 = unpack_matches(y_true, m)
        return lightglue_nll(
            y_pred, matches0, matches1, self.gamma, self.nll_balancing)

    def compute(
        self,
        log_assignments: Any,
        matches0: Any,
        matches1: Any,
        token_logits0: Optional[Any] = None,
        token_logits1: Optional[Any] = None,
        mask0: Optional[Any] = None,
        mask1: Optional[Any] = None,
    ) -> Any:
        """Full objective, per sample.

        :param log_assignments: ``(B, L, M+1, N+1)``.
        :param matches0: ``(B, M)`` int labels.
        :param matches1: ``(B, N)`` int labels.
        :param token_logits0: ``(B, L-1, M)`` pre-sigmoid confidence logits, or None to
            skip the confidence term.
        :param token_logits1: ``(B, L-1, N)`` pre-sigmoid confidence logits, or None.
        :param mask0: Optional real-keypoint mask ``(B, M)`` for the confidence term.
        :param mask1: Optional real-keypoint mask ``(B, N)`` for the confidence term.
        :return: ``(B,)`` float32; reduce with ``keras.ops.mean`` for ``add_loss``.
        """
        total = lightglue_nll(
            log_assignments, matches0, matches1, self.gamma, self.nll_balancing)
        if (self.confidence_weight > 0.0 and token_logits0 is not None
                and token_logits1 is not None):
            total = total + self.confidence_weight * lightglue_confidence_loss(
                log_assignments, token_logits0, token_logits1, mask0, mask1)
        return total

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration."""
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "nll_balancing": self.nll_balancing,
            "confidence_weight": self.confidence_weight,
        })
        return config
