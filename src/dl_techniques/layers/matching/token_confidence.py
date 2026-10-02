"""
LightGlue token confidence: how sure is this layer about each keypoint's final match.

A single ``Dense(1)`` plus sigmoid reads each (detached) descriptor and predicts whether
the match found at this layer already equals the match of the last layer. It drives the
early-stopping and pruning rules of LightGlue inference and is trained by an auxiliary
loss. The descriptors are passed through ``stop_gradient`` first, so that auxiliary loss
trains only this head and never reshapes the matcher's features.

Weight mapping from the official PyTorch state dict (``token_confidence.i.token.0.*``,
the sigmoid has no weights): ``layer.token_0.{kernel,bias}``, kernel by TRANSPOSE.

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
    """Per-keypoint confidence head shared by both images of a pair.

    Computes ``sigmoid(token_0(stop_gradient(desc)))`` for each image with one set of
    weights. The result is cast to at least float32 (``mask_dtype``) because it is
    compared against thresholds. With ``return_logits=True`` the pre-sigmoid logits are
    returned as well, so a loss can use a numerically stable log-sigmoid form
    (``BCEWithLogits``) instead of a clipped probability whose gradient dies when the
    head saturates.

    :param dim: Descriptor width.
    :type dim: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar token_0: ``Dense(1)``, named after torch ``token.0``.

    Call arguments:
        desc0: ``(B, M, dim)``.
        desc1: ``(B, N, dim)``.
        return_logits: Python bool, default ``False``. When ``True`` the output is the
            4-tuple ``(confidence0, confidence1, logit0, logit1)``; the default 2-tuple
            API is unchanged.

    Output:
        ``(confidence0 (B, M), confidence1 (B, N))`` in ``(0, 1)``. The gradient with
        respect to ``desc0`` and ``desc1`` is exactly zero by construction; the head's own
        kernel and bias still receive gradient.

    :raises ValueError: From ``__init__`` for a non-positive ``dim``; from ``build()``
        for a wrong rank or last dimension.

    Example:

    .. code-block:: python

        conf0, conf1 = MatchTokenConfidence(dim=64)(desc0, desc1)
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.token_0 = keras.layers.Dense(1, dtype=self.dtype_policy, name="token_0")

    def build(self, desc0_shape: Tuple[Any, ...], desc1_shape: Optional[Tuple[Any, ...]] = None) -> None:
        """Build the Dense head explicitly.

        :raises ValueError: If a descriptor shape is not rank 3 with last dim ``dim``.
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
        """Return the two ``(B, N)`` confidence maps, plus the logits on request.

        :param return_logits: If True return ``(conf0, conf1, logit0, logit1)``.
        :return: ``(conf0, conf1)`` or ``(conf0, conf1, logit0, logit1)``.
        """
        logit0, logit1 = self._logit(desc0), self._logit(desc1)
        conf0, conf1 = keras.ops.sigmoid(logit0), keras.ops.sigmoid(logit1)
        if return_logits:
            return conf0, conf1, logit0, logit1
        return conf0, conf1

    def compute_output_shape(self, desc0_shape: Tuple[Any, ...],
                             desc1_shape: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """``((B, M), (B, N))``."""
        return tuple(desc0_shape[:2]), tuple(desc1_shape[:2])

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration."""
        config = super().get_config()
        config.update({"dim": self.dim})
        return config
