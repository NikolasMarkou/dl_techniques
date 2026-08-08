"""
SAM 3 Dot-Product Scoring: the open-vocabulary class-score head.
================================================================

:class:`Sam3DotProductScoring` is SAM 3's classification head and deliberately
NOT a fixed-vocabulary classifier: no class table, no softmax over categories.
Each decoder query gets exactly ONE scalar logit -- query and pooled text prompt
projected into a shared space, then their scaled dot product.

Based on:
---------
- Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."
- Radford, A. et al. (2021). CLIP -- the image-text dot-product score
  generalized here to per-query detection logits.

Key Features:
------------
- Open vocabulary: swapping the prompt swaps the "class".
- Masked mean-pool over the prompt sequence, divisor floored at one.
- Two independent projections, one per operand, and a symmetric logit clamp.

Architecture Overview:
---------------------
1. prompt ``(batch, seq, d_model)`` -> optional 2-layer residual prompt MLP with
   a terminal normalization -> masked mean-pool -> ``Dense(d_proj)``.
2. queries ``(..., num_queries, d_model)`` -> a SECOND, INDEPENDENT
   ``Dense(d_proj)``.
3. ``score = clip(queries . prompt / sqrt(d_proj), -clamp_max_val, clamp_max_val)``.
Settled configuration: ``d_model=256``, ``d_proj=256``, a ``256 -> 2048 -> 256``
prompt MLP at dropout ``0.1``, clamp ``12.0``.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM3.model_misc import Sam3DotProductScoring
scorer = Sam3DotProductScoring(d_model=256, d_proj=256, clamp_max_val=12.0)
```

Measured caveats:
----------------
- The two projections are independent; sharing one still produces the right
  shapes and a plausible score, but it is a different model.
- The pool is MASKED and its divisor is floored at one: padding must not
  contribute, and an all-padding row has divisor zero under the naive spelling.
- The clamp bound is a different number from the decoder's and they are NOT to
  be unified; see the anchor on ``clamp_max_val``.
"""

import keras
import math
from keras import layers, ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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
    :param prompt_mlp_dropout: Dropout rate applied after the MLP's activation
        only. Default: ``0.1``.
    :type prompt_mlp_dropout: float
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
            prompt_mlp_dropout: float = 0.1, clamp_logits: bool = True,
            clamp_max_val: float = 12.0, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("d_model", d_model), ("d_proj", d_proj),
                            ("prompt_mlp_hidden_dim", prompt_mlp_hidden_dim)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= prompt_mlp_dropout < 1.0:
            raise ValueError(f"prompt_mlp_dropout must be in [0, 1), got "
                             f"{prompt_mlp_dropout}")
        if clamp_max_val <= 0.0:
            raise ValueError(f"clamp_max_val must be positive, got "
                             f"{clamp_max_val}")

        # Store ALL configuration parameters.
        self.d_model = int(d_model)
        self.d_proj = int(d_proj)
        self.use_prompt_mlp = bool(use_prompt_mlp)
        self.prompt_mlp_hidden_dim = int(prompt_mlp_hidden_dim)
        self.prompt_mlp_dropout = float(prompt_mlp_dropout)
        self.clamp_logits = bool(clamp_logits)
        self.clamp_max_val = float(clamp_max_val)
        self.scale = 1.0 / math.sqrt(float(self.d_proj))

        # DECISION plan-2026-08-04T044628-4c240b4c/D-106
        # The prompt MLP is composed here from plain `Dense`/`Dropout`/norm
        # layers ON PURPOSE. Do NOT "simplify" it to the repo's `MLPBlock` FFN:
        # that layer applies its dropout after BOTH dense layers, while the
        # reference drops only after the activation, and it offers neither the
        # residual add nor the terminal normalization this head needs. The
        # contract differs from the name, so the name is not the verdict --
        # the same failure mode already measured at six assets in this plan.
        # See decisions.md D-106.
        if self.use_prompt_mlp:
            self.prompt_fc1 = layers.Dense(self.prompt_mlp_hidden_dim,
                                           name="prompt_mlp_fc1")
            self.prompt_fc2 = layers.Dense(self.d_model, name="prompt_mlp_fc2")
            self.prompt_drop = layers.Dropout(self.prompt_mlp_dropout,
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
        # DECISION plan-2026-08-04T044628-4c240b4c/D-136
        # Re-entry guard, matching the other seven `build` methods in this
        # package. Do NOT delete it as "defensive style": without it a second
        # `build()` raises `ValueError: You cannot add new elements of state ...
        # to a layer that is already built`, which is exactly what Keras does on
        # `.keras` LOAD when it rebuilds a component from its recorded build
        # config before `build_from_config` runs. `Sam3Image._build_once` hides
        # that from this package's own gate, so the defect is invisible to any
        # composer that copies `Sam3Image`'s wiring but not its helper.
        # See decisions.md D-136.
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

    # A `@staticmethod` because it reads no state, and because it now has TWO
    # owners: this head and `Sam3EncoderQuerySelection`'s prompt-conditioned
    # branch. Both must pool a padded prompt with the SAME polarity and the
    # SAME floored divisor -- a second spelling of these three lines is exactly
    # the duplication D-104's two traps would be re-introduced through. Calling
    # it on an INSTANCE (`self.masked_mean_pool(...)`, as `call` below does)
    # keeps working unchanged.
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
        # DECISION plan-2026-08-04T044628-4c240b4c/D-104
        # The mask polarity is PADDING-IS-TRUE, and the divisor is floored at
        # one. Two traps live in these three lines. (a) The reference's own
        # inline comment on this argument says "1 is valid and 0 is padding"
        # and its code says the opposite (`is_valid = (~prompt_mask)`); the
        # code is what runs, and the tensor reaching it is a key-padding mask
        # built as `(tokens != 0).ne(1)`, i.e. True at padding. Do NOT flip
        # this to a keep predicate to match the causal KEEP mask the text tower
        # takes -- these are two different tensors with two different
        # conventions, and a flip is a silent value defect with no shape
        # symptom. (b) The floor of one is NOT defensive garnish: a row whose
        # every position is padding makes the divisor exactly zero, and the
        # naive `sum / sum` spelling returns NaN for it. See decisions.md D-104.
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

        # DECISION plan-2026-08-04T044628-4c240b4c/D-105
        # This bound is 12.0 and the decoder's presence clamp is 10.0. They are
        # DELIBERATELY different numbers on two different quantities. Do NOT
        # unify them, and do NOT "fix" this one to match the other: the two
        # clamps are indistinguishable on any probe whose scores never reach
        # the interval (10, 12], which is exactly why the guard for this line
        # pins a point inside that interval. See decisions.md D-105.
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
            "prompt_mlp_dropout": self.prompt_mlp_dropout,
            "clamp_logits": self.clamp_logits,
            "clamp_max_val": self.clamp_max_val,
        })
        return config
