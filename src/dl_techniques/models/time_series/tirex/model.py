"""
TiRexCore and its factories create_tirex_model/create_tirex_by_variant, a
patch-token forecaster whose blocks mix LSTM recurrence with self-attention,
decoded in one shot into non-crossing quantiles under reversible
per-instance normalization.

Every window is z-scored along its own time axis before anything else touches
it (statistics per series and per feature, never over the batch), and the
prediction is mapped back with y = q * std + mean. Missing values are zeroed
and excluded from the statistics by dividing by the count of valid steps, and
the validity mask is concatenated onto the feature axis so the encoder can
tell an imputed zero from an observed one. The series is segmented into
patches before encoding, cutting sequence length by patch_size. Each `mixed`
block runs LSTM then attention then a feed-forward layer in series, each
pre-normalized and residual, so attention operates on tokens that already
carry recurrent state and no positional embedding is needed; block_types lets
each block be purely recurrent, purely attentional, or both. attention_type
defaults to 'multi_head' (full self-attention, matching the published TiRex);
'window' restricts each token to attention_window_size neighbors. Decoding is
one-shot: the patch axis is mean-pooled to one summary token, and the head
projects it directly to prediction_length * num_quantiles values with
quantile crossing prevented by construction (Q_i = Q_{i-1} + softplus(r_i)).
For multivariate input, de-normalization uses the statistics of the last
feature, which is the model's convention for the target column.

References:
    - Auer et al., 2025. TiRex: Zero-Shot Forecasting Across Long and Short
      Horizons with Enhanced In-Context Learning.
      (https://arxiv.org/abs/2505.23719)
    - Nie et al., 2023. A Time Series is Worth 64 Words: Long-term Forecasting
      with Transformers. ICLR 2023. (https://arxiv.org/abs/2211.14730)
    - Kim et al., 2022. Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift. ICLR 2022.
    - Beltagy et al., 2020. Longformer: The Long-Document Transformer.
      (https://arxiv.org/abs/2004.05150)
    - Wen et al., 2017. A Multi-Horizon Quantile Recurrent Forecaster.
      (https://arxiv.org/abs/1711.11053)
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, List, Any, Sequence, Tuple, Dict, Literal

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn.residual_block import ResidualBlock
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding1D
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead
from dl_techniques.layers.time_series.mixed_sequential_block import MixedSequentialBlock
from dl_techniques.utils.keras_registration import register_dl_technique

BlockType = Literal['lstm', 'transformer', 'mixed']

# Canonical source list; also exposed as TiRexCore.DEFAULT_QUANTILES and
# imported directly by model_extended.py.
# DECISION plan-2026-08-19T163559-499b6f0e/D-079: keep this a tuple, not a list;
# a mutable list aliased by the class attribute let a caller mutate the shared default in place. See decisions.md.
DEFAULT_QUANTILES: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@register_dl_technique("dl_techniques.models.tirex.model")
class TiRexCore(keras.Model, ForecastMixin):
    """TiRex Core model for probabilistic time series forecasting.

    A hybrid LSTM/Transformer forecaster that emits monotonic quantile
    predictions with reversible per-instance normalization.

    Architecture:

    .. code-block:: text

        input [B, T, F]
             |
             v
        ┌──────────────┐
        │ mask + z-score │  NaN-safe, per series/feature (optional)
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ patch embed    │  [B, T, 2F] -> [B, num_patches, 2*embed_dim]
        │ input proj     │  -> [B, num_patches, embed_dim]
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ block 1        │  lstm / transformer / mixed
        │ ...            │
        │ block N        │
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ output norm    │
        │ mean pool      │  -> [B, 1, embed_dim]
        │ quantile head  │  -> [B, prediction_length, num_quantiles]
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ denormalize    │  (optional)
        └──────────────┘
             |
             v
        output [B, prediction_length, num_quantiles]

    :param patch_size: Size of input patches for tokenization.
    :type patch_size: int
    :param embed_dim: Embedding dimension for all model components.
    :type embed_dim: int
    :param num_blocks: Number of mixed sequential blocks.
    :type num_blocks: int
    :param num_heads: Number of attention heads for transformer components.
    :type num_heads: int
    :param lstm_units: LSTM units per block. Uses ``embed_dim`` if ``None``.
    :type lstm_units: Optional[int]
    :param ff_dim: Feed-forward dimension. Uses ``embed_dim * 4`` if ``None``.
    :type ff_dim: Optional[int]
    :param block_types: Type per block, from ``'lstm'``, ``'transformer'``,
        ``'mixed'``.
    :type block_types: Optional[List[BlockType]]
    :param quantile_levels: Quantile levels to predict.
    :type quantile_levels: Sequence[float]
    :param prediction_length: Length of the prediction horizon.
    :type prediction_length: int
    :param dropout_rate: Dropout rate for regularization.
    :type dropout_rate: float
    :param use_layer_norm: Whether to use layer normalization.
    :type use_layer_norm: bool
    :param use_normalization: Whether to apply reversible per-instance
        normalization to the inputs.
    :type use_normalization: bool
    :param attention_window_size: Window width in tokens, used only when
        ``attention_type='window'``.
    :type attention_window_size: int
    :param attention_type: Attention factory key used by every block.
        ``'multi_head'`` is full self-attention, matching the published TiRex;
        ``'window'`` restricts attention to ``attention_window_size`` tokens.
        Any other key from the attention factory registry is accepted.
    :type attention_type: str
    :param kwargs: Additional arguments for the ``Model`` base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.
        Can also accept 2D tensor which will be expanded to 3D.

    Output shape:
        3D tensor with shape: `(batch_size, prediction_length, num_quantiles)`.

    Example:
        ```python
        # Create TiRex model for time series forecasting
        model = TiRexCore(
            patch_size=16,
            embed_dim=256,
            num_blocks=6,
            prediction_length=32,
            quantile_levels=[0.1, 0.5, 0.9]
        )

        # Mixed block types for different processing stages
        model = TiRexCore(
            patch_size=8,
            embed_dim=128,
            num_blocks=4,
            block_types=['lstm', 'mixed', 'transformer', 'mixed'],
            prediction_length=24
        )
        ```
    """

    # References the module-level list so the value lives in exactly one place.
    DEFAULT_QUANTILES: Tuple[float, ...] = DEFAULT_QUANTILES

    MODEL_VARIANTS = {
        "tiny": {
            "patch_size": 8,
            "embed_dim": 64,
            "num_blocks": 3,
            "num_heads": 4,
            "dropout_rate": 0.1
        },
        "small": {
            "patch_size": 12,
            "embed_dim": 128,
            "num_blocks": 6,
            "num_heads": 8,
            "dropout_rate": 0.1
        },
        "medium": {
            "patch_size": 16,
            "embed_dim": 256,
            "num_blocks": 8,
            "num_heads": 8,
            "dropout_rate": 0.1
        },
        "large": {
            "patch_size": 16,
            "embed_dim": 512,
            "num_blocks": 12,
            "num_heads": 16,
            "dropout_rate": 0.15
        }
    }

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        lstm_units: Optional[int] = None,
        ff_dim: Optional[int] = None,
        block_types: Optional[List[BlockType]] = None,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        prediction_length: int = 32,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_normalization: bool = True,
        attention_window_size: int = 8,
        attention_type: str = 'multi_head',
        name: str = "TiRex",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {prediction_length}")
        if attention_window_size <= 0:
            raise ValueError(f"attention_window_size must be positive, got {attention_window_size}")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_types = block_types if block_types is not None else ['mixed'] * num_blocks
        # Materialize as a list: the default is an immutable tuple (D-079), but
        # get_config() must keep round-tripping the same JSON type it always has.
        self.quantile_levels = list(quantile_levels)
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_normalization = use_normalization
        self.attention_window_size = attention_window_size
        # DECISION plan-2026-08-14T183218-f4c612aa/D-008: no membership check on
        # attention_type here; create_attention_layer already raises eagerly on an unregistered key. Don't add a local whitelist. See decisions.md.
        self.attention_type = attention_type

        if len(self.block_types) != num_blocks:
            raise ValueError(
                f"Length of block_types ({len(self.block_types)}) must match num_blocks ({num_blocks})"
            )

        self.patch_embedding = PatchEmbedding1D(
            patch_size=self.patch_size,
            # Doubled width: the mask is concatenated onto the feature axis.
            embed_dim=self.embed_dim * 2,
            name="patch_embedding"
        )

        self.input_projection = ResidualBlock(
            hidden_dim=self.embed_dim * 2,
            output_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            activation="mish",
            name="input_projection"
        )

        self.blocks = []
        for i, block_type in enumerate(self.block_types):
            # DECISION plan-2026-08-17T183311-79c63e38/D-011: window_size stays
            # wired unconditionally; MixedSequentialBlock scopes it per attention type, not this call site. See decisions.md.
            block = MixedSequentialBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                lstm_units=self.lstm_units,
                ff_dim=self.ff_dim,
                block_type=block_type,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                normalization_type='rms_norm',
                attention_type=self.attention_type,
                ffn_type='geglu',
                activation='mish',
                attention_args={'window_size': self.attention_window_size},
                name=f"block_{i}"
            )
            self.blocks.append(block)

        if self.use_layer_norm:
            self.output_norm = (
                create_normalization_layer(
                    normalization_type='rms_norm',
                    name="output_norm"
                )
            )
        else:
            # keras.layers.Identity is a serializable Keras-3 drop-in for a
            # Lambda identity, which serializes as a non-portable pickled Python callable.
            self.output_norm = keras.layers.Identity(name="output_norm")

        self.quantile_head = QuantileHead(
            num_quantiles=len(self.quantile_levels),
            output_length=self.prediction_length,
            # Capped rather than the global dropout_rate directly.
            dropout_rate=min(self.dropout_rate, 0.1),
            enforce_monotonicity=True,
            use_bias=True,
            flatten_input=True,
            name="quantile_head"
        )

        logger.info(
            f"TiRex model initialized: {num_blocks} blocks, "
            f"embed_dim={embed_dim}, prediction_length={prediction_length}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build every sublayer with shapes threaded from the raw input.

        Explicit per-sublayer builds are required: on a ``.keras`` load, Keras
        restores weights before the first ``call``, so an unbuilt sublayer has
        nowhere for its restored weights to land and silently re-initializes.

        :param input_shape: Raw input shape ``(batch, seq_len, features)``. A
            2D ``(batch, seq_len)`` shape is treated as
            ``(batch, seq_len, 1)``, matching ``call``'s expand-dims path.
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

        # call() concatenates the NaN mask onto the feature axis -> 2 * features.
        masked_features = None if features is None else features * 2
        patch_input_shape = (batch_size, seq_len, masked_features)

        self.patch_embedding.build(patch_input_shape)
        embedded_shape = self.patch_embedding.compute_output_shape(patch_input_shape)

        self.input_projection.build(embedded_shape)
        projected_shape = self.input_projection.compute_output_shape(embedded_shape)

        current_shape = projected_shape
        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        self.output_norm.build(current_shape)

        # Quantile head input is mean-pooled over time.
        pooled_shape = (current_shape[0], 1, current_shape[2])
        self.quantile_head.build(pooled_shape)

        super().build(input_shape)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, matching the rank-3 quantile output of ``call``.

        :param input_shape: Raw input shape ``(batch, seq_len, features)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, prediction_length, len(quantile_levels))``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.prediction_length, len(self.quantile_levels))

    def call(
            self,
            inputs: Union[keras.KerasTensor, np.ndarray],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Normalize, mask, encode, and decode into quantile predictions.

        :param inputs: Input tensor, ``[batch_size, sequence_length, features]``
            or ``[batch_size, sequence_length]`` (expanded to 3D).
        :type inputs: Union[keras.KerasTensor, np.ndarray]
        :param training: Whether dropout runs in training mode.
        :type training: Optional[bool]
        :return: Quantile predictions,
            ``[batch_size, prediction_length, num_quantiles]``.
        :rtype: keras.KerasTensor
        """
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # Mask before normalization, so NaNs never reach the statistics.
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=inputs.dtype)
        clean_inputs = ops.where(ops.isnan(inputs), ops.zeros_like(inputs), inputs)

        if self.use_normalization:
            # NaN-safe mean/std: sum over valid values, divide by valid count.
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

        hidden_states = x_embedded
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training)

        hidden_states = self.output_norm(hidden_states, training=training)
        mean_hidden_states = ops.mean(hidden_states, axis=1, keepdims=True)

        quantile_predictions = self.quantile_head(mean_hidden_states, training=training)

        if self.use_normalization:
            norm_mean, norm_std = self._get_target_stats(mean, std)
            quantile_predictions = (quantile_predictions * norm_std) + norm_mean

        return quantile_predictions

    @staticmethod
    def _get_target_stats(
        mean: keras.KerasTensor,
        std: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Extract the last feature's normalization stats, for broadcasting with quantiles.

        For multivariate input, the target is assumed to be the last feature.

        :param mean: Mean tensor, ``(Batch, 1, Features)``.
        :type mean: keras.KerasTensor
        :param std: Std tensor, ``(Batch, 1, Features)``.
        :type std: keras.KerasTensor
        :return: ``(norm_mean, norm_std)``, each shaped ``(Batch, 1, 1)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        if mean.shape[-1] is not None and mean.shape[-1] > 1:
            norm_mean = mean[:, :, -1:]
            norm_std = std[:, :, -1:]
        else:
            norm_mean = mean
            norm_std = std
        return norm_mean, norm_std

    def predict_quantiles(
            self,
            context: Union[np.ndarray, keras.utils.PyDataset],
            quantile_levels: Optional[List[float]] = None,
            batch_size: int = 32,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map requested quantile levels to output indices, wrapping ``model.predict()``.

        Also extracts the median (0.5 quantile) as a point forecast.

        :param context: Input data — a NumPy array of shape
            ``(batch_size, input_length, features)``, or a
            ``keras.utils.PyDataset`` / ``tf.data.Dataset``.
        :type context: Union[np.ndarray, keras.utils.PyDataset]
        :param quantile_levels: Probabilities to extract (e.g.
            ``[0.1, 0.5, 0.9]``). Returns every trained quantile if ``None``.
            A level absent from training falls back to the closest trained
            level, with a warning.
        :type quantile_levels: Optional[List[float]]
        :param batch_size: Samples per batch during inference.
        :type batch_size: int
        :param kwargs: Forwarded to ``model.predict()`` (e.g. ``verbose``).
        :return: ``(quantile_preds, point_preds)`` — quantile predictions
            ``(batch_size, prediction_length, num_requested_quantiles)`` and
            the median as a point forecast ``(batch_size, prediction_length)``.
        :rtype: Tuple[np.ndarray, np.ndarray]

        Example::

            q_preds, median = model.predict_quantiles(
                context,
                quantile_levels=[0.05, 0.5, 0.95]
            )
        """
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        raw_predictions = self.predict(context, batch_size=batch_size, **kwargs)

        quantile_indices = []
        trained_quantiles_arr = np.array(self.quantile_levels)

        for q in quantile_levels:
            if q in self.quantile_levels:
                idx = self.quantile_levels.index(q)
                quantile_indices.append(idx)
            else:
                closest_idx = int(np.argmin(np.abs(trained_quantiles_arr - q)))
                quantile_indices.append(closest_idx)

                logger.warning(
                    f"Requested quantile {q} not found in trained model "
                    f"{self.quantile_levels}. Using closest match: "
                    f"{self.quantile_levels[closest_idx]}"
                )

        quantile_preds = raw_predictions[:, :, quantile_indices]

        # The median minimizes MAE and is the standard point forecast here.
        if 0.5 in self.quantile_levels:
            median_idx = self.quantile_levels.index(0.5)
        else:
            median_idx = len(self.quantile_levels) // 2
            logger.debug(
                f"Median (0.5) not found in quantiles. Using index {median_idx} "
                f"({self.quantile_levels[median_idx]}) as point forecast."
            )

        mean_preds = raw_predictions[:, :, median_idx]

        return quantile_preds, mean_preds

    def _forecast(
            self,
            x: Union[np.ndarray, keras.utils.PyDataset],
            quantile_levels: Optional[List[float]] = None,
            **kwargs: Any
    ) -> Forecast:
        """Produce a unified :class:`Forecast` by reusing ``predict_quantiles``.

        This is the ``ForecastMixin`` hook. It does NOT reimplement any quantile
        mapping; it delegates to the model's existing ``predict_quantiles`` and
        packs the result into the shared contract.

        :param x: Context window, ``[B, input_length, F]`` (or a dataset).
        :type x: Union[np.ndarray, keras.utils.PyDataset]
        :param quantile_levels: Levels to extract; defaults to
            ``self.quantile_levels``.
        :type quantile_levels: Optional[List[float]]
        :param kwargs: Forwarded to ``predict_quantiles``.
        :return: A :class:`Forecast` with ``point`` shape ``[B, H]`` and
            ``quantiles`` shape ``[B, H, Q]``. TiRex flattens the target
            feature axis, so these shapes pass through unchanged.
        :rtype: Forecast
        """
        levels = quantile_levels if quantile_levels is not None else self.quantile_levels
        quantile_preds, point_preds = self.predict_quantiles(x, levels, **kwargs)
        return Forecast(
            point=np.asarray(point_preds),
            quantiles=np.asarray(quantile_preds),
            quantile_levels=list(levels),
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        prediction_length: int = 32,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        **kwargs
    ) -> "TiRexCore":
        """Create a TiRex model from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"medium"``, ``"large"``.
        :type variant: str
        :param prediction_length: Length of the prediction horizon.
        :type prediction_length: int
        :param quantile_levels: Quantile levels to predict.
        :type quantile_levels: Sequence[float]
        :param kwargs: Additional arguments passed to the constructor;
            take precedence over the variant's defaults.
        :return: The configured model.
        :rtype: TiRexCore
        :raises ValueError: If ``variant`` is not recognized.

        Example::

            model = TiRexCore.from_variant("tiny", prediction_length=24)
            model = TiRexCore.from_variant("large", prediction_length=48)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating TiRex-{variant.upper()} model")

        return cls(
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "lstm_units": self.lstm_units,
            "ff_dim": self.ff_dim,
            "block_types": self.block_types,
            "quantile_levels": self.quantile_levels,
            "prediction_length": self.prediction_length,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "use_normalization": self.use_normalization,
            "attention_window_size": self.attention_window_size,
            "attention_type": self.attention_type,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TiRexCore":
        """Create model from configuration."""
        return cls(**config)


def create_tirex_model(
    input_length: int,
    prediction_length: int = 32,
    patch_size: int = 16,
    embed_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    block_types: Optional[List[str]] = None,
    **kwargs
) -> TiRexCore:
    """Create a TiRex model, then build it against ``input_length``.

    :param input_length: Length of input sequences.
    :type input_length: int
    :param prediction_length: Length of the prediction horizon.
    :type prediction_length: int
    :param patch_size: Size of input patches.
    :type patch_size: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param num_blocks: Number of sequential blocks.
    :type num_blocks: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param quantile_levels: Quantile levels to predict.
    :type quantile_levels: Sequence[float]
    :param block_types: Block type for each layer.
    :type block_types: Optional[List[str]]
    :param kwargs: Additional arguments for :class:`TiRexCore`.
    :return: A built :class:`TiRexCore` instance.
    :rtype: TiRexCore
    """
    model = TiRexCore(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        block_types=block_types,
        quantile_levels=quantile_levels,
        prediction_length=prediction_length,
        **kwargs
    )

    # build() is byte-identical to a dummy forward pass at the same seed (D-078).
    model.build((None, input_length, 1))

    logger.info(
        f"Created TiRex model: input_length={input_length}, "
        f"prediction_length={prediction_length}, embed_dim={embed_dim}"
    )

    return model


def create_tirex_by_variant(
    variant: str = "medium",
    input_length: int = 128,
    prediction_length: int = 32,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    **kwargs
) -> TiRexCore:
    """Create a TiRex model from a predefined variant, then build it.

    :param variant: Model variant (``"tiny"``, ``"small"``, ``"medium"``,
        ``"large"``).
    :type variant: str
    :param input_length: Length of input sequences.
    :type input_length: int
    :param prediction_length: Length of the prediction horizon.
    :type prediction_length: int
    :param quantile_levels: Quantile levels to predict.
    :type quantile_levels: Sequence[float]
    :param kwargs: Additional arguments passed to the model constructor.
    :return: A built :class:`TiRexCore` instance.
    :rtype: TiRexCore

    Example::

        model = create_tirex_by_variant("small", input_length=96, prediction_length=24)
        model = create_tirex_by_variant("large", input_length=256, prediction_length=48)
    """
    model = TiRexCore.from_variant(
        variant,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        **kwargs
    )

    # DECISION plan-2026-08-19T163559-499b6f0e/D-078: build() materializes the
    # model instead of a dummy forward pass; byte-identical at the same seed. See decisions.md.
    model.build((None, input_length, 1))

    logger.info(
        f"Created TiRex-{variant.upper()}: input_length={input_length}, "
        f"prediction_length={prediction_length}"
    )

    return model
