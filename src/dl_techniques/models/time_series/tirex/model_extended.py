"""
TiRexExtended and its factory create_tirex_extended, a query-token variant of
TiRexCore whose pooled decoder is replaced by learnable horizon tokens that
give one distinct latent state per forecast step.

TiRexCore collapses its encoded history to a single mean-pooled vector and
projects that one vector to the whole horizon grid, so every forecast step is
decoded from identical evidence. This variant keeps the same front half
(normalization, patch embedding, mixed LSTM/attention blocks) but appends a
learnable weight of shape (1, prediction_length, embed_dim), one token per
forecast step, to the end of the embedded history along the time axis. The
LSTM sub-layer carries state forward into these tokens, and attention lets a
token read specific history positions directly; the final normalization's
last prediction_length states are sliced out (no pooling) and a token-wise
quantile head maps each to its own quantiles. Under attention_type='window' a
query token sees only its own window of the augmented sequence, so long-range
history access falls back onto the recurrent path.

build() is reimplemented rather than inherited: the blocks and output
normalization see a sequence longer than the parent's by prediction_length,
and the head is QuantileSequenceHead rather than the parent's pooled
QuantileHead. It calls keras.Model.build directly rather than
TiRexCore.build, since the parent's version would build sub-layers against
the wrong shapes. predict_quantiles, _forecast and the Forecast contract are
inherited unchanged from TiRexCore.

References:
    - Auer et al., 2025. TiRex: Zero-Shot Forecasting Across Long and Short
      Horizons with Enhanced In-Context Learning.
      (https://arxiv.org/abs/2505.23719)
    - Nie et al., 2023. A Time Series is Worth 64 Words: Long-term Forecasting
      with Transformers. ICLR 2023. (https://arxiv.org/abs/2211.14730)
    - Carion et al., 2020. End-to-End Object Detection with Transformers.
      (https://arxiv.org/abs/2005.12872)
    - Zhou et al., 2021. Informer: Beyond Efficient Transformer for Long Sequence
      Time-Series Forecasting. AAAI 2021. (https://arxiv.org/abs/2012.07436)
    - Wen et al., 2017. A Multi-Horizon Quantile Recurrent Forecaster.
      (https://arxiv.org/abs/1711.11053)
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, List, Any, Dict, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.quantile_head_variable_io import QuantileSequenceHead

from .model import BlockType, DEFAULT_QUANTILES, TiRexCore
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.tirex.model_extended")
class TiRexExtended(TiRexCore):
    """TiRex with a query-token decoder instead of mean-pooling.

    Architecture:

    .. code-block:: text

        history embedded [B, num_patches, embed_dim]
             |
        (append learnable query tokens)
             |
             v
        [B, num_patches + prediction_length, embed_dim]
             |
             v
        ┌──────────────┐
        │ block 1        │  lstm carries history state into query tokens;
        │ ...            │  attention lets a query token read history directly
        │ block N        │
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ output norm    │
        │ slice last     │  -> [B, prediction_length, embed_dim]
        │ prediction_len │
        │ quantile head  │  token-wise, -> [B, prediction_length, Q]
        └──────────────┘

    No pooling: each forecast step keeps its own latent state instead of
    sharing one summary vector.
    """

    def __init__(
            self,
            patch_size: int = 16,
            embed_dim: int = 256,
            num_blocks: int = 6,
            num_heads: int = 8,
            lstm_units: Optional[int] = None,
            ff_dim: Optional[int] = None,
            block_types: Optional[List[BlockType]] = None,
            quantile_levels: List[float] = DEFAULT_QUANTILES,
            prediction_length: int = 32,
            dropout_rate: float = 0.1,
            use_layer_norm: bool = True,
            use_normalization: bool = True,
            attention_window_size: int = 8,
            name: str = "TiRexExtended",
            **kwargs: Any
    ) -> None:
        """Create the parent TiRexCore graph, then the query-token head.

        Every argument mirrors ``TiRexCore``; only the prediction head and
        token handling differ.
        """
        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            lstm_units=lstm_units,
            ff_dim=ff_dim,
            block_types=block_types,
            quantile_levels=quantile_levels,
            prediction_length=prediction_length,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_normalization=use_normalization,
            attention_window_size=attention_window_size,
            name=name,
            **kwargs
        )

        # Learnable query tokens are created in build() — see the D-037 anchor there.
        self.query_tokens = None

        self.quantile_head = QuantileSequenceHead(
            num_quantiles=len(self.quantile_levels),
            # Capped rather than the global dropout_rate directly.
            dropout_rate=min(self.dropout_rate, 0.1),
            enforce_monotonicity=True,
            use_bias=True,
            name="quantile_head"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sublayer for the query-token topology.

        Cannot delegate to ``TiRexCore.build``: the blocks and output norm
        see a sequence longer by ``prediction_length``, and the head is the
        token-wise ``QuantileSequenceHead`` over
        ``(B, prediction_length, embed_dim)`` rather than the parent's
        pooled ``QuantileHead``.

        :param input_shape: Raw input shape ``(batch, seq_len, features)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, features), got "
                f"{len(input_shape)}D input with shape {input_shape}"
            )

        batch_size, seq_len, features = input_shape[0], input_shape[1], input_shape[2]

        # DECISION plan-2026-08-19T163559-499b6f0e/D-037: create query_tokens here,
        # not in __init__ — a weight added before build() sits outside Keras's build scope. See decisions.md.
        if self.query_tokens is None:
            self.query_tokens = self.add_weight(
                shape=(1, self.prediction_length, self.embed_dim),
                initializer="glorot_uniform",
                trainable=True,
                name="query_tokens"
            )

        # call() concatenates the NaN mask onto the feature axis -> 2 * features.
        masked_features = None if features is None else features * 2
        patch_input_shape = (batch_size, seq_len, masked_features)

        self.patch_embedding.build(patch_input_shape)
        embedded_shape = self.patch_embedding.compute_output_shape(patch_input_shape)

        self.input_projection.build(embedded_shape)
        projected_shape = self.input_projection.compute_output_shape(embedded_shape)

        # Append prediction_length query tokens along the time axis.
        num_patches = projected_shape[1]
        augmented_len = (
            None if num_patches is None else num_patches + self.prediction_length
        )
        current_shape = (projected_shape[0], augmented_len, self.embed_dim)

        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        self.output_norm.build(current_shape)

        # Quantile head is token-wise over the sliced query states.
        head_input_shape = (current_shape[0], self.prediction_length, self.embed_dim)
        self.quantile_head.build(head_input_shape)

        # query_tokens is already built via add_weight above; skip TiRexCore.build
        # (different topology) and go straight to keras.Model.build.
        keras.Model.build(self, input_shape)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape: identical rank-3 contract to ``TiRexCore``.

        :param input_shape: Raw input shape ``(batch, seq_len, features)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, prediction_length, len(quantile_levels))``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.prediction_length, len(self.quantile_levels))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Normalize, embed history, append query tokens, then decode token-wise.

        :param inputs: Input tensor, ``[batch, sequence_length, features]`` or
            ``[batch, sequence_length]`` (expanded to 3D).
        :type inputs: keras.KerasTensor
        :param training: Whether dropout runs in training mode.
        :type training: Optional[bool]
        :return: Quantile predictions,
            ``[batch, prediction_length, num_quantiles]``.
        :rtype: keras.KerasTensor
        """
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # Mask before normalization, so NaNs never reach the statistics.
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=inputs.dtype)
        clean_inputs = ops.where(ops.isnan(inputs), ops.zeros_like(inputs), inputs)

        if self.use_normalization:
            valid_count = ops.maximum(ops.sum(nan_mask, axis=1, keepdims=True), 1e-7)
            mean = ops.sum(clean_inputs * nan_mask, axis=1, keepdims=True) / valid_count
            sq_diff = ((clean_inputs - mean) * nan_mask) ** 2
            variance = ops.sum(sq_diff, axis=1, keepdims=True) / valid_count
            std = ops.sqrt(variance)
            std = ops.maximum(std, 1e-7)
            x = (clean_inputs - mean) / std
        else:
            x = clean_inputs
            mean = None
            std = None

        x_with_mask = ops.concatenate([x, nan_mask], axis=-1)

        x_patches = self.patch_embedding(x_with_mask, training=training)
        x_embedded = self.input_projection(x_patches, training=training)

        batch_size = ops.shape(x_embedded)[0]

        # Broadcast the learnable query tokens to the batch and append them
        # along the time axis: (B, num_patches + prediction_length, embed_dim).
        prediction_tokens = ops.broadcast_to(
            self.query_tokens,
            (batch_size, self.prediction_length, self.embed_dim)
        )
        mixed_sequence = ops.concatenate([x_embedded, prediction_tokens], axis=1)

        hidden_states = mixed_sequence
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training)

        hidden_states = self.output_norm(hidden_states, training=training)

        # No pooling: keep only the query-token states, at the sequence end.
        prediction_states = hidden_states[:, -self.prediction_length:, :]

        quantile_predictions = self.quantile_head(prediction_states, training=training)

        if self.use_normalization:
            norm_mean, norm_std = self._get_target_stats(mean, std)
            quantile_predictions = (quantile_predictions * norm_std) + norm_mean

        return quantile_predictions

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration, inherited from ``TiRexCore``.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        return super().get_config()


def create_tirex_extended(
    variant: str = "medium",
    input_length: int = 128,
    prediction_length: int = 32,
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    **kwargs
) -> TiRexExtended:
    """Create a TiRexExtended model from a predefined variant, then build it.

    :param variant: Model variant (``"tiny"``, ``"small"``, ``"medium"``,
        ``"large"``).
    :type variant: str
    :param input_length: Length of input sequences.
    :type input_length: int
    :param prediction_length: Length of the prediction horizon.
    :type prediction_length: int
    :param quantile_levels: Quantile levels to predict.
    :type quantile_levels: List[float]
    :param kwargs: Additional arguments passed to the model constructor.
    :return: A built :class:`TiRexExtended` instance.
    :rtype: TiRexExtended

    Example::

        model = create_tirex_extended("small", input_length=96, prediction_length=24)
        model = create_tirex_extended("large", input_length=256, prediction_length=48)
    """
    model = TiRexExtended.from_variant(
        variant,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        **kwargs
    )

    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex-Extended-{variant.upper()}: input_length={input_length}, "
        f"prediction_length={prediction_length}"
    )

    return model
