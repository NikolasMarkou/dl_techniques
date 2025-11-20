"""
TiRex - Time Series Forecasting Model for Keras.

This module implements a TiRex-inspired time series forecasting model adapted for Keras,
using a mixed architecture of LSTM and Transformer layers for sequential modeling.

The implementation follows modern Keras 3 patterns and utilizes the dl_techniques
factory systems for consistent component creation.

Based on TiRex architecture principles with enhancements for:
- Configurable mixed sequential blocks (LSTM + Transformer)
- Quantile-based probabilistic forecasting
- Patch-based time series tokenization
- Modern normalization techniques
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, List, Any, Tuple, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.standard_scaler import StandardScaler
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn.residual_block import ResidualBlock
from dl_techniques.layers.time_series.quantile_head import QuantileHead
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding1D
from dl_techniques.layers.time_series.mixed_sequential_block import MixedSequentialBlock

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

BlockType = Literal['lstm', 'transformer', 'mixed']

# Default quantile levels for probabilistic forecasting
DEFAULT_QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TiRexCore(keras.Model):
    """
    TiRex Core Model for Time Series Forecasting.

    This model implements a TiRex-inspired architecture using mixed sequential blocks
    (LSTM + Transformer) for probabilistic time series forecasting. The model follows
    modern Keras 3 patterns and utilizes factory systems for component creation.

    The architecture consists of:
    1. Input scaling and preprocessing
    2. Patch embedding for time series tokenization
    3. Sequential processing blocks (configurable LSTM/Transformer mix)
    4. Quantile prediction head for probabilistic outputs

    Args:
        patch_size: Integer, size of input patches for tokenization.
        embed_dim: Integer, embedding dimension for all model components.
        num_blocks: Integer, number of mixed sequential blocks.
        num_heads: Integer, number of attention heads for transformer components.
        lstm_units: Integer, LSTM units per block. If None, uses embed_dim.
        ff_dim: Integer, feed-forward dimension. If None, uses embed_dim * 4.
        block_types: List of BlockType strings, type for each block ('lstm', 'transformer', 'mixed').
        quantile_levels: List of floats, quantile levels to predict.
        prediction_length: Integer, length of prediction horizon.
        dropout_rate: Float, dropout rate for regularization.
        use_layer_norm: Boolean, whether to use layer normalization.
        **kwargs: Additional keyword arguments for the Model base class.

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

    # Model variant configurations following ConvNeXt V2 pattern
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
        quantile_levels: List[float] = DEFAULT_QUANTILES,
        prediction_length: int = 32,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_normalization: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {prediction_length}")

        # Store configuration
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
        self.use_normalization = use_normalization

        if len(self.block_types) != num_blocks:
            raise ValueError(
                f"Length of block_types ({len(self.block_types)}) must match num_blocks ({num_blocks})"
            )

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
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

        # Create sequential processing blocks
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

        # Output normalization using factory
        if self.use_layer_norm:
            self.output_norm = create_normalization_layer('rms_norm', name="output_norm")
        else:
            self.output_norm = keras.layers.Lambda(lambda x: x, name="output_norm")

        # Quantile prediction head
        self.quantile_head = QuantileHead(
            num_quantiles=len(self.quantile_levels),
            output_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            name="quantile_head"
        )

        logger.info(
            f"TiRex model initialized: {num_blocks} blocks, "
            f"embed_dim={embed_dim}, prediction_length={prediction_length}"
        )

    def call(self, inputs, training=None):
        """
        Forward pass through the TiRex model.

        Args:
            inputs: Input tensor of shape [batch_size, sequence_length, features] or
                   [batch_size, sequence_length] which will be expanded.
            training: Boolean, whether in training mode.

        Returns:
            Quantile predictions of shape [batch_size, prediction_length, num_quantiles].
        """
        # Ensure 3D input
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # 1. CALCULATE STATISTICS & NORMALIZE
        if self.use_normalization:
            # Calculate stats across the time dimension (axis 1)
            mean = ops.mean(inputs, axis=1, keepdims=True)
            std = ops.std(inputs, axis=1, keepdims=True)
            std = ops.maximum(std, 1e-7)  # Prevent division by zero
            x = (inputs - mean) / std
        else:
            x = inputs
            mean = None
            std = None

        # 2. HANDLE MASKING
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=x.dtype)
        x_with_mask = ops.concatenate([x, nan_mask], axis=-1)

        # 3. ENCODE
        x_patches = self.patch_embedding(x_with_mask, training=training)
        x_embedded = self.input_projection(x_patches, training=training)

        # 4. PROCESS
        hidden_states = x_embedded
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training)

        hidden_states = self.output_norm(hidden_states, training=training)
        #pooled = ops.mean(hidden_states, axis=1)
        # --- FIX: FLATTEN INSTEAD OF MEAN POOLING ---
        # Preserves sequence history for the dense head to utilize
        batch_size = ops.shape(hidden_states)[0]
        # Flatten patches: (B, Num_Patches, Dim) -> (B, Num_Patches * Dim)
        pooled = ops.reshape(hidden_states, (batch_size, -1))
        # ---------------------------------------------

        # 5. PREDICT (Normalized Space)
        # Shape: [batch, prediction_length, num_quantiles]
        quantile_predictions = self.quantile_head(pooled, training=training)

        # 6. DENORMALIZE OUTPUT (Reversible Instance Normalization)
        if self.use_normalization:
            # mean/std shape: (Batch, 1, 1) via keepdims=True in step 1
            # predictions shape: (Batch, Time, Quantiles)
            # Broadcasting will apply scale/shift to all time steps and all quantiles
            quantile_predictions = (quantile_predictions * std) + mean

        return quantile_predictions

    def predict_quantiles(
        self,
        context: Union[np.ndarray, keras.utils.PyDataset],
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
            quantile_predictions: [batch, prediction_length, selected_quantiles]
            mean_predictions: [batch, prediction_length]
        """
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        # Predictions shape: [batch, prediction_length, num_quantiles]
        predictions = self.predict(context, batch_size=batch_size, **kwargs)

        # Select requested quantiles
        quantile_indices = []
        for q in quantile_levels:
            if q in self.quantile_levels:
                quantile_indices.append(self.quantile_levels.index(q))
            else:
                closest_idx = np.argmin(np.abs(np.array(self.quantile_levels) - q))
                quantile_indices.append(closest_idx)
                logger.warning(
                    f"Quantile {q} not in training quantiles, using closest: {self.quantile_levels[closest_idx]}"
                )

        # Slicing the last dimension (quantiles)
        quantile_preds = predictions[:, :, quantile_indices]

        # Use median as mean prediction
        median_idx = self.quantile_levels.index(0.5) if 0.5 in self.quantile_levels else len(self.quantile_levels) // 2
        # Shape becomes [batch, prediction_length]
        mean_preds = predictions[:, :, median_idx]

        return quantile_preds, mean_preds

    @classmethod
    def from_variant(
        cls,
        variant: str,
        prediction_length: int = 32,
        quantile_levels: List[float] = DEFAULT_QUANTILES,
        **kwargs
    ) -> "TiRexCore":
        """
        Create a TiRex model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "medium", "large"
            prediction_length: Integer, length of prediction horizon
            quantile_levels: List of quantile levels to predict
            **kwargs: Additional arguments passed to the constructor

        Returns:
            TiRexCore model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Tiny model for quick experiments
            >>> model = TiRexCore.from_variant("tiny", prediction_length=24)
            >>> # Large model for production
            >>> model = TiRexCore.from_variant("large", prediction_length=48)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        # Update config with kwargs (kwargs take precedence)
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
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TiRexCore":
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------
# Factory Functions (following ConvNeXt V2 pattern)
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
        TiRexCore model instance.
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

    # Build the model with a dummy input to initialize weights and shapes
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex model: input_length={input_length}, "
        f"prediction_length={prediction_length}, embed_dim={embed_dim}"
    )

    return model


def create_tirex_by_variant(
    variant: str = "medium",
    input_length: int = 128,
    prediction_length: int = 32,
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    **kwargs
) -> TiRexCore:
    """
    Convenience function to create TiRex models from predefined variants.

    Args:
        variant: String, model variant ("tiny", "small", "medium", "large")
        input_length: Integer, length of input sequences
        prediction_length: Integer, length of prediction horizon
        quantile_levels: List of quantile levels to predict
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        TiRexCore model instance

    Example:
        >>> # Create TiRex-Small for quick experiments
        >>> model = create_tirex_by_variant("small", input_length=96, prediction_length=24)
        >>>
        >>> # Create TiRex-Large for production forecasting
        >>> model = create_tirex_by_variant("large", input_length=256, prediction_length=48)
    """
    model = TiRexCore.from_variant(
        variant,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        **kwargs
    )

    # Build the model with a dummy input
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex-{variant.upper()}: input_length={input_length}, "
        f"prediction_length={prediction_length}"
    )

    return model

# ---------------------------------------------------------------------