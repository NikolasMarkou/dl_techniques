"""Sam3DotProductScoring, SAM 3's open-vocabulary classification head.

There is no class table and no softmax over categories: each decoder query
gets one scalar logit from the scaled dot product of its own projection and
a pooled, projected text-prompt embedding, so swapping the prompt swaps the
detected class. The prompt is optionally refined by a small residual MLP
first, then mean-pooled over its sequence with padding excluded and the
divisor floored at one so an all-padding row cannot divide by zero. Query
and prompt each get their own independent projection before the dot product.

The logit clamp here (default 12.0) is a different, deliberately unrelated
number from the decoder's presence clamp elsewhere in this package; do not
unify them.

References:
    - Ravi et al., 2025. SAM 3: Segment Anything with Concepts.
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision.
"""

import keras
import math
from keras import layers, ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam3.model_misc")
class Sam3DotProductScoring(keras.layers.Layer):
    """Open-vocabulary per-query class logits from a pooled text prompt.

    :param d_model: Width of both the query features and the prompt features.
        Default: ``256``.
    :type d_model: int
    :param d_proj: Width of the shared projection space; also sets the
        ``1/sqrt(d_proj)`` score scale. Default: ``256``.
    :type d_proj: int
    :param use_prompt_mlp: Whether to refine the prompt with the residual MLP
        before pooling. Default: ``True``.
    :type use_prompt_mlp: bool
    :param prompt_mlp_hidden_dim: Hidden width of that MLP. Default: ``2048``.
    :type prompt_mlp_hidden_dim: int
    :param prompt_mlp_dropout_rate: Dropout rate applied after the MLP's activation
        only. Default: ``0.1``.
    :type prompt_mlp_dropout_rate: float
    :param clamp_logits: Whether to clamp the scores. Default: ``True``.
    :type clamp_logits: bool
    :param clamp_max_val: Symmetric clamp bound. Default: ``12.0``.
    :type clamp_max_val: float
    :raises ValueError: If any width is non-positive, if the dropout rate is
        outside ``[0, 1)``, or if ``clamp_max_val`` is non-positive.

    Example:
        >>> import numpy as np
        >>> head = Sam3DotProductScoring(d_model=8, d_proj=4,
        ...                              prompt_mlp_hidden_dim=16)
        >>> queries = np.zeros((2, 5, 8), dtype="float32")
        >>> prompt = np.zeros((2, 6, 8), dtype="float32")
        >>> padding = np.zeros((2, 6), dtype="bool")
        >>> head(queries, prompt, padding).shape
        (2, 5, 1)
    """

    def __init__(
            self, d_model: int = 256, d_proj: int = 256,
            use_prompt_mlp: bool = True, prompt_mlp_hidden_dim: int = 2048,
            prompt_mlp_dropout_rate: float = 0.1, clamp_logits: bool = True,
            clamp_max_val: float = 12.0, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("d_model", d_model), ("d_proj", d_proj),
                            ("prompt_mlp_hidden_dim", prompt_mlp_hidden_dim)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= prompt_mlp_dropout_rate < 1.0:
            raise ValueError(f"prompt_mlp_dropout_rate must be in [0, 1), got "
                             f"{prompt_mlp_dropout_rate}")
        if clamp_max_val <= 0.0:
            raise ValueError(f"clamp_max_val must be positive, got "
                             f"{clamp_max_val}")

        # Store ALL configuration parameters.
        self.d_model = int(d_model)
        self.d_proj = int(d_proj)
        self.use_prompt_mlp = bool(use_prompt_mlp)
        self.prompt_mlp_hidden_dim = int(prompt_mlp_hidden_dim)
        self.prompt_mlp_dropout_rate = float(prompt_mlp_dropout_rate)
        self.clamp_logits = bool(clamp_logits)
        self.clamp_max_val = float(clamp_max_val)
        self.scale = 1.0 / math.sqrt(float(self.d_proj))

        # DECISION plan-2026-08-04T044628-4c240b4c/D-106: keep this plain
        # Dense/Dropout/norm MLP, not the repo's MLPBlock FFN.
        # MLPBlock drops after both dense layers and has no residual/terminal norm; the contract differs from the name. See decisions.md.
        if self.use_prompt_mlp:
            self.prompt_fc1 = layers.Dense(self.prompt_mlp_hidden_dim,
                                           name="prompt_mlp_fc1")
            self.prompt_fc2 = layers.Dense(self.d_model, name="prompt_mlp_fc2")
            self.prompt_drop = layers.Dropout(self.prompt_mlp_dropout_rate,
                                              name="prompt_mlp_dropout")
            self.prompt_norm = layers.LayerNormalization(
                epsilon=1e-5, name="prompt_mlp_norm")
        self.prompt_proj = layers.Dense(self.d_proj, name="prompt_proj")
        self.hs_proj = layers.Dense(self.d_proj, name="hs_proj")

        logger.info(
            f"Sam3DotProductScoring: d_model={self.d_model}, "
            f"d_proj={self.d_proj}, prompt_mlp={self.use_prompt_mlp}, "
            f"clamp={self.clamp_max_val if self.clamp_logits else None}"
        )

    def build(
            self, hs_shape: Tuple[Optional[int], ...],
            prompt_shape: Tuple[Optional[int], ...],
            prompt_padding_mask_shape: Optional[Tuple] = None,
    ) -> None:
        """Build both projections and the optional prompt MLP.

        :param hs_shape: Query shape ``(..., num_queries, d_model)``, rank >= 3.
        :type hs_shape: Tuple[Optional[int], ...]
        :param prompt_shape: Prompt shape ``(batch, seq, d_model)``.
        :type prompt_shape: Tuple[Optional[int], ...]
        :param prompt_padding_mask_shape: Unused; accepted so that the layer can
            be built from its full call signature.
        :type prompt_padding_mask_shape: Optional[Tuple[Optional[int], ...]]
        :raises ValueError: On a wrong rank or a width other than ``d_model``.
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-136: keep this re-entry
        # guard, matching the package's other build() methods.
        # A second build() without it raises on .keras load, when Keras rebuilds from a recorded build config. See decisions.md.
        if self.built:
            return
        if len(hs_shape) < 3:
            raise ValueError(f"queries must have rank >= 3 (..., num_queries, "
                             f"d_model), got {hs_shape}")
        if len(prompt_shape) != 3:
            raise ValueError(f"prompt must have shape (batch, seq, d_model), "
                             f"got {prompt_shape}")
        for name, shape in (("queries", hs_shape), ("prompt", prompt_shape)):
            if shape[-1] is not None and shape[-1] != self.d_model:
                raise ValueError(f"{name} width {shape[-1]} != d_model "
                                 f"{self.d_model}")
        if self.use_prompt_mlp:
            hidden = tuple(prompt_shape[:-1]) + (self.prompt_mlp_hidden_dim,)
            self.prompt_fc1.build(tuple(prompt_shape))
            self.prompt_drop.build(hidden)
            self.prompt_fc2.build(hidden)
            self.prompt_norm.build(tuple(prompt_shape))
        self.prompt_proj.build((prompt_shape[0], self.d_model))
        self.hs_proj.build(tuple(hs_shape))
        super().build(hs_shape)

    # A static method so `Sam3EncoderQuerySelection`'s prompt-conditioned branch
    # can share it without a second, drift-prone copy of the pooling logic.
    @staticmethod
    def masked_mean_pool(
            prompt: keras.KerasTensor,
            prompt_padding_mask: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Mean-pool the prompt over its sequence axis, ignoring padding.

        :param prompt: Prompt features ``(batch, seq, d_model)``.
        :type prompt: keras.KerasTensor
        :param prompt_padding_mask: ``(batch, seq)``, ``True`` at PADDING
            positions. ``None`` means every position is valid.
        :type prompt_padding_mask: Optional[keras.KerasTensor]
        :return: Pooled prompt ``(batch, d_model)``.
        :rtype: keras.KerasTensor
        """
        if prompt_padding_mask is None:
            return ops.mean(prompt, axis=1)
        # DECISION plan-2026-08-04T044628-4c240b4c/D-104: mask polarity is
        # padding-is-True; do not flip it to match the text tower's keep mask.
        # Divisor is floored at one so an all-padding row does not divide by zero. See decisions.md.
        valid = ops.cast(
            ops.logical_not(ops.cast(prompt_padding_mask, "bool")), prompt.dtype
        )
        valid = ops.expand_dims(valid, axis=-1)
        total = ops.sum(prompt * valid, axis=1)
        count = ops.maximum(ops.sum(valid, axis=1), ops.cast(1.0, prompt.dtype))
        return total / count

    def call(
            self, hs: keras.KerasTensor, prompt: keras.KerasTensor,
            prompt_padding_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Score every query against the pooled prompt.

        :param hs: Query features ``(..., num_queries, d_model)``. The leading
            axes are free, so a stack of per-decoder-layer states works
            unchanged.
        :type hs: keras.KerasTensor
        :param prompt: Prompt features ``(batch, seq, d_model)``.
        :type prompt: keras.KerasTensor
        :param prompt_padding_mask: ``(batch, seq)``, ``True`` at PADDING.
        :type prompt_padding_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag; affects the prompt MLP's dropout.
        :type training: Optional[bool]
        :return: Class logits ``(..., num_queries, 1)``.
        :rtype: keras.KerasTensor
        """
        if self.use_prompt_mlp:
            hidden = self.prompt_drop(
                ops.relu(self.prompt_fc1(prompt)), training=training)
            prompt = self.prompt_norm(self.prompt_fc2(hidden) + prompt)

        pooled = self.prompt_proj(self.masked_mean_pool(
            prompt, prompt_padding_mask))
        scores = ops.matmul(
            self.hs_proj(hs), ops.expand_dims(pooled, axis=-1)) * self.scale

        # DECISION plan-2026-08-04T044628-4c240b4c/D-105: this clamp is 12.0,
        # the decoder's presence clamp is 10.0 -- do not unify them.
        # The two are indistinguishable on any probe whose scores never reach (10, 12]. See decisions.md.
        if self.clamp_logits:
            scores = ops.clip(scores, -self.clamp_max_val, self.clamp_max_val)
        return scores

    def compute_output_shape(
            self, hs_shape: Tuple[Optional[int], ...],
            prompt_shape: Optional[Tuple] = None,
            prompt_padding_mask_shape: Optional[Tuple] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return ``hs_shape`` with its width replaced by a single logit.

        :param hs_shape: Query shape ``(..., num_queries, d_model)``.
        :type hs_shape: Tuple[Optional[int], ...]
        :param prompt_shape: Unused.
        :type prompt_shape: Optional[Tuple[Optional[int], ...]]
        :param prompt_padding_mask_shape: Unused.
        :type prompt_padding_mask_shape: Optional[Tuple[Optional[int], ...]]
        :return: Output shape ``(..., num_queries, 1)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(hs_shape[:-1]) + (1,)

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "d_proj": self.d_proj,
            "use_prompt_mlp": self.use_prompt_mlp,
            "prompt_mlp_hidden_dim": self.prompt_mlp_hidden_dim,
            "prompt_mlp_dropout_rate": self.prompt_mlp_dropout_rate,
            "clamp_logits": self.clamp_logits,
            "clamp_max_val": self.clamp_max_val,
        })
        return config
