"""LightGlue token confidence: per-keypoint confidence that a layer's match is final.

:class:`MatchTokenConfidence` reads each descriptor with one ``Dense(1)`` and a sigmoid
and predicts whether the match found at this layer already equals the last layer's
match. LightGlue inference uses it for early stopping and pruning, and an auxiliary
loss trains it. Descriptors pass through ``stop_gradient`` first, so that loss trains
only this head and never reshapes the matcher's features.

References:
    - Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). "LightGlue: Local
      Feature Matching at Light Speed". arXiv:2306.13643.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.common import mask_dtype
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.matching.token_confidence")
class MatchTokenConfidence(keras.layers.Layer):
    """Predict per-keypoint match confidence for both images with one shared head.

    Computes ``sigmoid(token_0(stop_gradient(desc)))`` for each image with one set of
    weights. The Dense runs in the compute dtype; its logit is then cast to at least
    float32 (``mask_dtype``) because the confidence is compared against thresholds.
    With ``return_logits=True`` the logits come back too, so a loss can use a stable
    log-sigmoid form (``BCEWithLogits``) instead of a clipped probability whose
    gradient dies when the head saturates. The gradient with respect to ``desc0`` and
    ``desc1`` is exactly zero; ``token_0`` still receives gradient.

    .. code-block:: text

        desc0 [B, M, D]      desc1 [B, N, D]
               │                    │
               ▼                    ▼
        stop_gradient        stop_gradient
               │                    │
               ▼                    ▼
        ┌──────────────────────────────┐
        │ token_0  Dense(1)            │  shared
        └──────────────────────────────┘
               │                    │
               ▼                    ▼
        logit0 [B, M]        logit1 [B, N]
               │                    │
               ▼                    ▼
        sigmoid              sigmoid
               │                    │
               ▼                    ▼
        confidence0 [B, M]   confidence1 [B, N]

    Torch weight mapping (the ``Linear`` weight transposes into the kernel, the bias
    copies; the sigmoid has no weights):

    .. code-block:: text

        torch (token_confidence.i.)  Keras
        token.0.{weight,bias}        token_0.{kernel,bias}

    :param dim: Descriptor width ``D``. Must be positive.
    :type dim: int
    :param **kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar token_0: ``Dense(1)``, named after torch ``token.0``.

    Call arguments:
        desc0: ``(B, M, dim)``.
        desc1: ``(B, N, dim)``.
        return_logits: Python bool, default ``False``. When ``True`` the output is the
            4-tuple ``(confidence0, confidence1, logit0, logit1)``.

    Output:
        ``(confidence0 (B, M), confidence1 (B, N))`` in ``[0, 1]``; a saturated logit
        rounds to exactly 0 or 1.

    Example:

    .. code-block:: python

        conf0, conf1 = MatchTokenConfidence(dim=64)(desc0, desc1)
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        """Validate ``dim`` and create ``token_0``.

        Arguments are documented on the class.

        :raises ValueError: If ``dim`` is not positive.
        """
        super().__init__(**kwargs)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.token_0 = keras.layers.Dense(1, dtype=self.dtype_policy, name="token_0")

    def build(self, desc0_shape: Tuple[Any, ...], desc1_shape: Optional[Tuple[Any, ...]] = None) -> None:
        """Build the Dense head explicitly. Returns at once if already built.

        :param desc0_shape: ``(batch, M, dim)``. An unknown last dimension is accepted.
        :param desc1_shape: ``(batch, N, dim)``. Checked only when given.
        :raises ValueError: If a given descriptor shape is not rank 3 with last dim
            ``dim``.
        """
        if self.built:
            return
        for name, shape in (("desc0", desc0_shape), ("desc1", desc1_shape)):
            if shape is None:
                continue
            if len(shape) != 3 or (shape[-1] is not None and shape[-1] != self.dim):
                raise ValueError(f"{name} must be (batch, num_points, {self.dim}), got {shape}")
        self.token_0.build((None, None, self.dim))
        super().build(desc0_shape)

    def _logit(self, desc: Any) -> Any:
        """Pre-sigmoid logit ``(B, N)`` in at least float32; the input is detached."""
        md = mask_dtype(self.compute_dtype)
        logit = self.token_0(keras.ops.stop_gradient(desc))
        return keras.ops.cast(logit, md)[..., 0]

    def _confidence(self, desc: Any) -> Any:
        return keras.ops.sigmoid(self._logit(desc))

    def call(self, desc0: Any, desc1: Any, return_logits: bool = False) -> Tuple[Any, ...]:
        """Return the two confidence maps, plus the logits on request.

        :param desc0: ``(B, M, dim)``.
        :param desc1: ``(B, N, dim)``.
        :param return_logits: If ``True``, also return the logits.
        :return: ``(conf0 (B, M), conf1 (B, N))``, or
            ``(conf0, conf1, logit0, logit1)``; all in at least float32.
        """
        logit0, logit1 = self._logit(desc0), self._logit(desc1)
        conf0, conf1 = keras.ops.sigmoid(logit0), keras.ops.sigmoid(logit1)
        if return_logits:
            return conf0, conf1, logit0, logit1
        return conf0, conf1

    def compute_output_shape(self, desc0_shape: Tuple[Any, ...],
                             desc1_shape: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Return ``((B, M), (B, N))``; ``desc1_shape`` is required.

        Covers the default call only: with ``return_logits=True`` the call returns
        four tensors, and the two logit shapes are not listed here.
        """
        return tuple(desc0_shape[:2]), tuple(desc1_shape[:2])

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration."""
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

# ---------------------------------------------------------------------
