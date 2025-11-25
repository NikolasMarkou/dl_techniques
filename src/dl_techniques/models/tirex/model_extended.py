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
from typing import Optional, List, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.quantile_head_variable_io import QuantileSequenceHead

from .model import BlockType, DEFAULT_QUANTILES, TiRexCore

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
            name: str = "TiRexExtended",
            **kwargs: Any
    ) -> None:
        """
        Initialize the TiRexExtended model.

        All arguments mirror TiRexCore, but the internal graph construction
        differs for the prediction head and token handling.
        """
        # Explicitly pass arguments to the parent TiRexCore
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
            name=name,
            **kwargs
        )

        # Quantile prediction head
        self.quantile_head = QuantileSequenceHead(
            num_quantiles=len(self.quantile_levels),
            # Hardcode a safe low value, or dividing the global rate
            dropout_rate=min(self.dropout_rate, 0.1),
            enforce_monotonicity=True,
            use_bias=True,
            name="quantile_head"
        )

    def call(self, inputs, training=None):
        """
        Forward pass with Query Token appending.

        Logic:
            1. Normalize & Mask
            2. Embed History -> [Batch, Patches, Dim]
            3. Append Learned Queries -> [Batch, Patches + Pred_Len, Dim]
            4. Process via Mixed Blocks (LSTM flows history -> queries)
            5. Slice last Pred_Len tokens
            6. Project to Quantiles
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
        # Standard patch embedding from parent class
        x_patches = self.patch_embedding(x_with_mask, training=training)
        x_embedded = self.input_projection(x_patches, training=training)
        # x_embedded shape: (Batch, Num_Patches, Embed_Dim)

        # ---------------------------------------------------------------------
        # 4. APPEND PREDICTION TOKENS
        # ---------------------------------------------------------------------
        batch_size = ops.shape(x_embedded)[0]

        prediction_tokens = (
            ops.zeros(
                shape=(batch_size, self.prediction_length, self.embed_dim),
                dtype=inputs.dtype
            )
        )

        # Concatenate along time dimension (axis 1)
        # New Shape: (Batch, Num_Patches + Prediction_Length, Embed_Dim)
        mixed_sequence = ops.concatenate([x_embedded, prediction_tokens], axis=1)

        # ---------------------------------------------------------------------
        # 5. PROCESS SEQUENCE
        # ---------------------------------------------------------------------
        hidden_states = mixed_sequence

        # Iterate through MixedSequentialBlocks
        # The LSTM component allows information to flow from the history patches
        # into the query tokens. The Attention component allows query tokens
        # to attend back to specific historical events.
        for block in self.blocks:
            hidden_states = block(hidden_states, training=training)

        hidden_states = self.output_norm(hidden_states, training=training)

        # ---------------------------------------------------------------------
        # 6. EXTRACT PREDICTION PART (No Pooling)
        # ---------------------------------------------------------------------
        # We only care about the states of our Query Tokens (the end of the sequence).
        # Slice the last 'prediction_length' steps.
        # Shape: (Batch, Prediction_Length, Embed_Dim)
        prediction_states = hidden_states[:, -self.prediction_length:, :]

        # 7. PREDICT (Normalized Space)
        # The head operates token-wise
        # It projects (Batch, Pred_Len, Dim) -> (Batch, Pred_Len, Num_Quantiles)
        quantile_predictions = self.quantile_head(prediction_states, training=training)

        # 8. DENORMALIZE OUTPUT
        if self.use_normalization:
            # mean/std shape is (Batch, 1, 1). Broadcasting handles the rest.
            quantile_predictions = (quantile_predictions * std) + mean

        return quantile_predictions

    def get_config(self) -> Dict[str, Any]:
        """
        Return config for serialization.

        Since we explicitly accept the same arguments as TiRexCore,
        super().get_config() captures most of what we need.
        """
        return super().get_config()


# ---------------------------------------------------------------------
# Factory Function for the Extended Variant
# ---------------------------------------------------------------------

def create_tirex_extended(
    variant: str = "medium",
    input_length: int = 128,
    prediction_length: int = 32,
    quantile_levels: List[float] = DEFAULT_QUANTILES,
    **kwargs
) -> TiRexExtended:
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
        >>> # Create TiRex-Extended-Small for quick experiments
        >>> model = create_tirex_extended("small", input_length=96, prediction_length=24)
        >>>
        >>> # Create TiRex-Extended-Large for production forecasting
        >>> model = create_tirex_extended("large", input_length=256, prediction_length=48)
    """
    model = TiRexExtended.from_variant(
        variant,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        **kwargs
    )

    # Build the model with a dummy input
    dummy_input = np.zeros((1, input_length, 1), dtype='float32')
    _ = model(dummy_input)

    logger.info(
        f"Created TiRex-Extended-{variant.upper()}: input_length={input_length}, "
        f"prediction_length={prediction_length}"
    )

    return model

# ---------------------------------------------------------------------
