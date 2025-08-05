"""
TiRex - Time Series Forecasting Model for Keras.

This module implements a TiRex-inspired time series forecasting model adapted for Keras,
using a mixed architecture of LSTM and Transformer layers for sequential modeling.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, List, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms import RMSNorm
from ..layers.standard_scaler import StandardScaler
from ..layers.patch_embedding import PatchEmbedding1D
from ..layers.ffn.residual_block import ResidualBlock
from ..layers.time_series.quantile_head import QuantileHead
from ..layers.time_series.mixed_sequential_block import MixedSequentialBlock

# ---------------------------------------------------------------------

DEFAULT_QUANTILES : List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TiRexCore(keras.Model):
    """
    TiRex Core Model for Time Series Forecasting.

    This model implements a TiRex-inspired architecture using mixed sequential blocks
    (LSTM + Transformer) for probabilistic time series forecasting.

    Args:
        patch_size: Integer, size of input patches.
        embed_dim: Integer, embedding dimension.
        num_blocks: Integer, number of mixed sequential blocks.
        num_heads: Integer, number of attention heads.
        lstm_units: Integer, LSTM units per block.
        ff_dim: Integer, feed-forward dimension.
        block_types: List of strings, type for each block ('lstm', 'transformer', 'mixed').
        quantile_levels: List of floats, quantile levels to predict.
        prediction_length: Integer, length of prediction horizon.
        dropout_rate: Float, dropout rate for regularization.
        use_layer_norm: Boolean, whether to use layer normalization.
        **kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
            self,
            patch_size: int = 16,
            embed_dim: int = 256,
            num_blocks: int = 6,
            num_heads: int = 8,
            lstm_units: Optional[int] = None,
            ff_dim: Optional[int] = None,
            block_types: Optional[List[str]] = None,
            quantile_levels: List[float] = DEFAULT_QUANTILES,
            prediction_length: int = 32,
            dropout_rate: float = 0.1,
            use_layer_norm: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_types = block_types if block_types is not None else ['mixed'] * num_blocks
        self.quantile_levels = quantile_levels
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if len(self.block_types) != num_blocks:
            raise ValueError(f"Length of block_types ({len(self.block_types)}) must match num_blocks ({num_blocks})")

        # Initialize components
        self._build_model()

    def _build_model(self):
        """Build the TiRex model components."""

        self.scaler = StandardScaler(name="scaler")
        self.patch_embedding = PatchEmbedding1D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim * 2,  # Include mask information
            name="patch_embedding"
        )
        self.input_projection = ResidualBlock(
            hidden_dim=self.embed_dim * 2,
            output_dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="input_projection"
        )

        self.blocks = []
        for i, block_type in enumerate(self.block_types):
            block = MixedSequentialBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                lstm_units=self.lstm_units,
                ff_dim=self.ff_dim,
                block_type=block_type,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                name=f"block_{i}"
            )
            self.blocks.append(block)

        if self.use_layer_norm:
            self.output_norm = RMSNorm(name="output_norm")
        else:
            self.output_norm = keras.layers.Lambda(lambda x: x, name="output_norm")

        # --- FIX: Instantiate QuantileHead without the removed `hidden_dim` parameter.
        self.quantile_head = QuantileHead(
            num_quantiles=len(self.quantile_levels),
            output_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            name="quantile_head"
        )

        logger.info(f"TiRex model initialized with {self.num_blocks} blocks, "
                    f"embed_dim={self.embed_dim}, prediction_length={self.prediction_length}")

    def call(self, inputs, training=None):
        """
        Forward pass through the TiRex model.

        Args:
            inputs: Input tensor of shape [batch_size, sequence_length, features].
            training: Boolean, whether in training mode.

        Returns:
            Quantile predictions of shape [batch_size, num_quantiles, prediction_length].
        """
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        x = self.scaler(inputs, training=training)

        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=x.dtype)

        x_with_mask = ops.concatenate([x, nan_mask], axis=-1)

        x_patches, attention_mask = self.patch_embedding(x_with_mask, training=training)

        x_embedded = self.input_projection(x_patches, training=training)

        # --- FIX: Pass the correctly generated `attention_mask` to the sequential blocks.
        hidden_states = x_embedded
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training, mask=attention_mask)

        hidden_states = self.output_norm(hidden_states, training=training)

        # Global average pooling across the sequence of patches.
        pooled = ops.mean(hidden_states, axis=1)  # [batch_size, embed_dim]

        quantile_predictions = self.quantile_head(pooled, training=training)

        return quantile_predictions

    def predict_quantiles(
            self,
            context: Union[np.ndarray, 'keras.utils.PyDataset'],
            quantile_levels: Optional[List[float]] = None,
            batch_size: int = 32,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate quantile predictions for time series forecasting.

        Args:
            context: Input time series data.
            quantile_levels: List of quantile levels to return. If None, uses model defaults.
            batch_size: Batch size for prediction.
            **kwargs: Additional arguments passed to predict().

        Returns:
            Tuple of (quantile_predictions, mean_predictions) as numpy arrays.
        """
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        predictions = self.predict(context, batch_size=batch_size, **kwargs)

        quantile_indices = []
        for q in quantile_levels:
            if q in self.quantile_levels:
                quantile_indices.append(self.quantile_levels.index(q))
            else:
                closest_idx = np.argmin(np.abs(np.array(self.quantile_levels) - q))
                quantile_indices.append(closest_idx)
                logger.warning(
                    f"Quantile {q} not in training quantiles, using closest: {self.quantile_levels[closest_idx]}")

        quantile_preds = predictions[:, quantile_indices, :]

        median_idx = self.quantile_levels.index(0.5) if 0.5 in self.quantile_levels else len(self.quantile_levels) // 2
        mean_preds = predictions[:, median_idx, :]

        return quantile_preds, mean_preds

    def get_config(self):
        """Get model configuration."""
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
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------


def create_tirex_model(
        input_length: int,
        prediction_length: int = 32,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        quantile_levels: List[float] = DEFAULT_QUANTILES,
        block_types: Optional[List[str]] = None,
        **kwargs
) -> TiRexCore:
    """
    Create a TiRex model with specified configuration.

    Args:
        input_length: Integer, length of input sequences.
        prediction_length: Integer, length of prediction horizon.
        patch_size: Integer, size of input patches.
        embed_dim: Integer, embedding dimension.
        num_blocks: Integer, number of sequential blocks.
        num_heads: Integer, number of attention heads.
        quantile_levels: List of quantile levels to predict.
        block_types: List of block types for each layer.
        **kwargs: Additional arguments for TiRexCore.

    Returns:
        Compiled TiRexCore model.
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

    # Build the model with a dummy input to initialize weights and shapes.
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    model(dummy_input)

    logger.info(f"Created TiRex model: input_length={input_length}, "
                f"prediction_length={prediction_length}, embed_dim={embed_dim}")

    return model

# ---------------------------------------------------------------------


TIREX_CONFIGS = {
    "small": {
        "patch_size": 8,
        "embed_dim": 64,
        "num_blocks": 3,
        "num_heads": 4,
        "dropout_rate": 0.1
    },
    "medium": {
        "patch_size": 12,
        "embed_dim": 128,
        "num_blocks": 6,
        "num_heads": 8,
        "dropout_rate": 0.1
    },
    "large": {
        "patch_size": 16,
        "embed_dim": 256,
        "num_blocks": 12,
        "num_heads": 16,
        "dropout_rate": 0.15
    }
}

# ---------------------------------------------------------------------


def create_tirex_by_size(
        size: str,
        input_length: int,
        prediction_length: int,
        quantile_levels: list = DEFAULT_QUANTILES,
        **override_params
):
    """
    Create TiRex model using predefined size configurations.

    Args:
        size: String, one of 'small', 'medium', 'large'.
        input_length: Integer, length of input sequences.
        prediction_length: Integer, length of prediction horizon.
        quantile_levels: List of quantile levels to predict.
        **override_params: Additional parameters to override defaults.

    Returns:
        Configured TiRexCore model.
    """

    if size not in TIREX_CONFIGS:
        raise ValueError(f"Size must be one of {list(TIREX_CONFIGS.keys())}")

    config = TIREX_CONFIGS[size].copy()
    config.update(override_params)

    return create_tirex_model(
        input_length=input_length,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        **config
    )

# ---------------------------------------------------------------------