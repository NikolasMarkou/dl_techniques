"""
TiRex hybrid architecture for probabilistic time series forecasting,
combining recurrent state modeling with attention-based global context.

This class realizes a "Mixed Sequential" architecture designed to address the
dichotomy between local temporal dynamics and long-range global dependencies
in time series data. It synthesizes the inductive biases of Recurrent Neural
Networks (LSTMs) and Transformers into a unified, configurable pipeline.

Architecture Overview:
    The model processes data through four distinct stages:
    1.  **Reversible Normalization**: Input data is normalized (z-score) per-instance
        to handle non-stationary statistics (shifting mean/variance), a technique
        crucial for robust forecasting across diverse regimes.
    2.  **Patch Tokenization**: The time series is segmented into sub-sequences
        (patches) and projected into an embedding space. This reduces the effective
        sequence length, enabling the processing of long horizons with reduced
        computational complexity relative to point-wise attention.
    3.  **Mixed Sequential Encoding**: A stack of configurable blocks processes
        the tokenized sequence. These blocks can be pure LSTMs, pure Transformers,
        or "Mixed" blocks.
        -   **LSTM Layers** enforce a sequential inductive bias, carrying a running
            state $h_t$ that captures local evolution and short-term causality.
        -   **Transformer Layers** utilize Self-Attention to model global
            dependencies ($Attention(Q, K, V)$), allowing the model to attend to
            distant patches (e.g., yearly seasonality) regardless of their
            position in the sequence.
    4.  **Probabilistic Head**: The decoder projects the encoded representation
        into a distribution of quantiles, approximating the inverse cumulative
        distribution function (quantile function) of the target.

Mathematical Intuition:
    Standard Transformers suffer from a lack of inherent sequential bias, often
    requiring complex positional encodings. LSTMs suffer from the vanishing gradient
    problem over long horizons. TiRex addresses this by applying recurrence *within*
    or *before* attention.

    Mathematically, the mixed block computes:
    $$ H_{local} = \text{LSTM}(X_{patches}) $$
    $$ H_{global} = \text{SelfAttention}(H_{local}) $$

    This structure ensures that the tokens fed into the attention mechanism
    already contain rich, context-aware local state information, stabilizing
    optimization and improving zero-shot generalization capabilities.

References:
    -   **TiRex Architecture**: Auer, A., et al. (2025). "TiRex: Zero-Shot
        Forecasting Across Long and Short Horizons with Enhanced In-Context Learning."
        arXiv preprint arXiv:2505.23719.
    -   **Patching**: Nie, Y., et al. (2023). "A Time Series is Worth 64 Words:
        Long-term Forecasting with Transformers." ICLR.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, List, Any, Tuple, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
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
        name: str = "TiRex",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

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
            activation="gelu",
            name="input_projection"
        )

        # Create sequential processing blocks
        self.blocks = []
        for i, block_type in enumerate(self.block_types):
            # --- DIVERGENCE FROM TIREX: WINDOW ATTENTION INSTEAD OF GLOBAL ATTENTION ---
            block = MixedSequentialBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                lstm_units=self.lstm_units,
                ff_dim=self.ff_dim,
                block_type=block_type,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                normalization_type='rms_norm',
                attention_type='window',
                ffn_type='geglu',
                activation='gelu',
                name=f"block_{i}"
            )
            # ---------------------------------------------
            self.blocks.append(block)

        # Output normalization using factory
        if self.use_layer_norm:
            self.output_norm = (
                create_normalization_layer(
                    normalization_type='rms_norm',
                    name="output_norm"
                )
            )
        else:
            self.output_norm = keras.layers.Lambda(lambda x: x, name="output_norm")

        # Quantile prediction head
        self.quantile_head = QuantileHead(
            num_quantiles=len(self.quantile_levels),
            output_length=self.prediction_length,
            # Hardcode a safe low value, or dividing the global rate
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
        mean_hidden_states = ops.mean(hidden_states, axis=1, keepdims=True)

        # 5. PREDICT (Normalized Space)
        # Shape: [batch, prediction_length, num_quantiles]
        quantile_predictions = self.quantile_head(mean_hidden_states, training=training)

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
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate specific quantile and point forecasts for time series data.

        This method acts as a high-level wrapper around `model.predict()`. It handles
        the complexity of mapping user-requested quantile levels (e.g., 0.95) to the
        specific output indices of the model's neural network head. It also automatically
        extracts the median (0.5 quantile) to serve as a robust point forecast.

        **Shape Logic**:
        The raw model outputs a tensor of shape `(Batch, Time, Trained_Quantiles)`.
        This method slices the last dimension based on the requested `quantile_levels`.

        Args:
            context: Input data.
                - A Numpy array of shape `(batch_size, input_length, features)`.
                - Or a `keras.utils.PyDataset` / `tf.data.Dataset`.
            quantile_levels: List of floats between 0 and 1.
                The specific probabilities to extract (e.g., `[0.1, 0.5, 0.9]`).
                If None, returns all quantiles the model was trained with.
                If a requested quantile was not in the training set, the closest
                available trained quantile will be used (with a warning).
            batch_size: Integer, number of samples per batch during inference.
                Defaults to 32.
            **kwargs: Additional arguments passed directly to `model.predict()`,
                such as `verbose` or `callbacks`.

        Returns:
            A tuple `(quantile_preds, point_preds)`:
            1. **quantile_preds**: Numpy array of shape
               `(batch_size, prediction_length, num_requested_quantiles)`.
               Contains the predicted values for the requested probability levels.
            2. **point_preds**: Numpy array of shape
               `(batch_size, prediction_length)`.
               Contains the median prediction (0.5 quantile), used as the primary
               point forecast.

        Example:
            ```python
            # Train with [0.1, 0.5, 0.9]
            model = TiRexCore(...)

            # Request specific confidence intervals at inference
            # context shape: (100, 168, 1)
            q_preds, median = model.predict_quantiles(
                context,
                quantile_levels=[0.05, 0.5, 0.95] # 0.05/0.95 map to closest (0.1/0.9)
            )

            # q_preds shape: (100, 24, 3)
            # median shape:  (100, 24)
            ```
        """
        # ---------------------------------------------------------------------
        # 1. Setup and Validation
        # ---------------------------------------------------------------------
        # If no specific levels requested, return everything the model knows
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        # ---------------------------------------------------------------------
        # 2. Run Inference
        # ---------------------------------------------------------------------
        # Perform the forward pass.
        # Output Shape: [batch_size, prediction_length, num_trained_quantiles]
        raw_predictions = self.predict(context, batch_size=batch_size, **kwargs)

        # ---------------------------------------------------------------------
        # 3. Map Requested Quantiles to Model Output Indices
        # ---------------------------------------------------------------------
        # We need to find which index in the last dimension corresponds to
        # the requested quantiles (e.g., user asks for 0.5, we find index 2).
        quantile_indices = []
        trained_quantiles_arr = np.array(self.quantile_levels)

        for q in quantile_levels:
            # Case A: Exact match found
            if q in self.quantile_levels:
                idx = self.quantile_levels.index(q)
                quantile_indices.append(idx)
            # Case B: Approximation needed (User asks for 0.95, model has 0.9)
            else:
                # Find index of the smallest absolute difference
                closest_idx = int(np.argmin(np.abs(trained_quantiles_arr - q)))
                quantile_indices.append(closest_idx)

                logger.warning(
                    f"Requested quantile {q} not found in trained model "
                    f"{self.quantile_levels}. Using closest match: "
                    f"{self.quantile_levels[closest_idx]}"
                )

        # ---------------------------------------------------------------------
        # 4. Extract Quantile Predictions
        # ---------------------------------------------------------------------
        # Slice the raw predictions tensor.
        # We select all batches (:), all time steps (:), and specific quantile indices.
        # Result Shape: [batch_size, prediction_length, num_requested_quantiles]
        quantile_preds = raw_predictions[:, :, quantile_indices]

        # ---------------------------------------------------------------------
        # 5. Extract Point Forecast (Median)
        # ---------------------------------------------------------------------
        # The median (0.5) minimizes MAE and is the standard point forecast
        # for quantile regression models.
        if 0.5 in self.quantile_levels:
            median_idx = self.quantile_levels.index(0.5)
        else:
            # Fallback: Use the middle index if strict 0.5 is missing
            median_idx = len(self.quantile_levels) // 2
            logger.debug(
                f"Median (0.5) not found in quantiles. Using index {median_idx} "
                f"({self.quantile_levels[median_idx]}) as point forecast."
            )

        # Slice out the median to get a 2D array.
        # Result Shape: [batch_size, prediction_length]
        mean_preds = raw_predictions[:, :, median_idx]

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

@keras.saving.register_keras_serializable()
class TiRexExtended(TiRexCore):
    """
    TiRex Extended (Query-Based) Architecture.

    This variant differs from the Core architecture in how the prediction representations
    are formed. Instead of Mean Pooling the historical context and projecting it,
    this model appends a sequence of learnable 'Query Tokens' to the embedded
    time series.

    Architecture Changes:
        1.  **Input**: Standard patch embedding of history.
        2.  **Token Augmentation**: `prediction_length` learnable vectors are
            concatenated to the end of the sequence.
        3.  **Processing**: The Mixed Sequential Blocks process the combined sequence
            [History, Queries]. The LSTM flows state from history into queries;
            Attention allows queries to look back at specific historical patches.
        4.  **Output Extraction**: No pooling is performed. The last `prediction_length`
            vectors are sliced from the sequence.
        5.  **Head**: These vectors are projected directly to quantiles.

    This approach allows for more fine-grained control per time-step and aligns
    closer to Decoder-style architectures without autoregressive loop overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---------------------------------------------------------------------
        # 1. Define Learnable Query Tokens
        # ---------------------------------------------------------------------
        # Shape: (1, prediction_length, embed_dim)
        # We use add_weight to create a trainable variable.
        self.prediction_query_tokens = self.add_weight(
            name="prediction_query_tokens",
            shape=(1, self.prediction_length, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
            trainable=True,
            dtype=self.dtype
        )

        # ---------------------------------------------------------------------
        # 2. Re-define Quantile Head for Token-wise prediction
        # ---------------------------------------------------------------------
        # In TiRexCore, the head takes a pooled vector and expands to (T, Q).
        # Here, we feed it (T, D) and want (T, Q).
        # We override the self.quantile_head from the parent.
        self.quantile_head = QuantileHead(
            num_quantiles=len(self.quantile_levels),
            output_length=1,  # Projection is per-token, effectively dense(num_quantiles)
            dropout_rate=min(self.dropout_rate, 0.1),
            enforce_monotonicity=True,
            use_bias=True,
            flatten_input=False,  # CRITICAL: Keep the time dimension (Batch, Pred_Len, Dim)
            name="quantile_head_token_wise"
        )

    def call(self, inputs, training=None):
        """
        Forward pass with Query Token appending.

        Logic:
            [Input (T_in)] -> Embed -> [Emb (T_patch)]
            [Emb (T_patch)] + [Queries (T_pred)] -> [Seq (T_total)]
            [Seq] -> Blocks -> [Hidden (T_total)]
            [Hidden] -> Slice last T_pred -> Head -> Output
        """
        # Ensure 3D input
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)

        # 1. CALCULATE STATISTICS & NORMALIZE (Reversible Norm)
        if self.use_normalization:
            mean = ops.mean(inputs, axis=1, keepdims=True)
            std = ops.std(inputs, axis=1, keepdims=True)
            std = ops.maximum(std, 1e-7)
            x = (inputs - mean) / std
        else:
            x = inputs
            mean = None
            std = None

        # 2. HANDLE MASKING
        nan_mask = ops.logical_not(ops.isnan(inputs))
        nan_mask = ops.cast(nan_mask, dtype=x.dtype)
        x_with_mask = ops.concatenate([x, nan_mask], axis=-1)

        # 3. ENCODE HISTORY
        x_patches = self.patch_embedding(x_with_mask, training=training)
        x_embedded = self.input_projection(x_patches, training=training)
        # x_embedded shape: (Batch, Num_Patches, Embed_Dim)

        # ---------------------------------------------------------------------
        # 4. APPEND PREDICTION TOKENS (The specific variation)
        # ---------------------------------------------------------------------
        batch_size = ops.shape(x_embedded)[0]

        # Tile the query tokens to match batch size
        # tokens shape: (1, Pred_Len, Embed_Dim) -> (Batch, Pred_Len, Embed_Dim)
        tokens = ops.tile(self.prediction_query_tokens, [batch_size, 1, 1])

        # Concatenate along time dimension
        # Shape: (Batch, Num_Patches + Pred_Len, Embed_Dim)
        mixed_sequence = ops.concatenate([x_embedded, tokens], axis=1)

        # ---------------------------------------------------------------------
        # 5. PROCESS SEQUENCE
        # ---------------------------------------------------------------------
        hidden_states = mixed_sequence
        for block in self.blocks:
            # Blocks (LSTM/Attn) handle variable length automatically
            hidden_states = block(hidden_states, training=training)

        hidden_states = self.output_norm(hidden_states, training=training)

        # ---------------------------------------------------------------------
        # 6. EXTRACT PREDICTION PART (No Pooling)
        # ---------------------------------------------------------------------
        # Slice the last 'prediction_length' tokens.
        # Shape: (Batch, Prediction_Length, Embed_Dim)
        prediction_states = hidden_states[:, -self.prediction_length:, :]

        # 7. PREDICT (Normalized Space)
        # The head operates token-wise due to flatten_input=False
        # Output: (Batch, Prediction_Length, Num_Quantiles)
        quantile_predictions = self.quantile_head(prediction_states, training=training)

        # 8. DENORMALIZE OUTPUT
        if self.use_normalization:
            quantile_predictions = (quantile_predictions * std) + mean

        return quantile_predictions


# ---------------------------------------------------------------------
# Factory Function for the Extended Variant
# ---------------------------------------------------------------------

def create_tirex_extended(
    input_length: int,
    prediction_length: int = 32,
    patch_size: int = 16,
    embed_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    block_types: Optional[List[str]] = None,
    **kwargs
) -> TiRexExtended:
    """
    Create a TiRex Extended (Query-Based) model.

    Args:
        input_length: Integer, length of input sequences.
        prediction_length: Integer, length of prediction horizon.
        patch_size: Integer, size of input patches.
        embed_dim: Integer, embedding dimension.
        num_blocks: Integer, number of sequential blocks.
        num_heads: Integer, number of attention heads.
        quantile_levels: List of quantile levels to predict.
        block_types: List of block types for each layer.
        **kwargs: Additional arguments.

    Returns:
        TiRexExtended model instance.
    """
    model = TiRexExtended(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        block_types=block_types,
        quantile_levels=quantile_levels,
        prediction_length=prediction_length,
        **kwargs
    )

    # Initialize weights with dummy input
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex Extended (Token-Augmented): "
        f"input_length={input_length}, prediction_length={prediction_length}"
    )

    return model