"""
xLSTM (Extended Long Short-Term Memory) Implementation.

This module provides a production-ready implementation of the xLSTM architecture
from the paper "xLSTM: Extended Long Short-Term Memory" (arXiv:2405.04517v2).

The implementation follows the dl_techniques framework standards:
- Uses normalization factory for all normalization layers
- Uses FFN factory for feed-forward networks
- Follows Keras 3 custom layer/model guidelines
- Implements proper serialization and build patterns

References:
    Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory.
    arXiv:2405.04517v2
"""

import keras
from keras import layers, initializers
from typing import Optional, Union, Any, Dict, Literal


from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock, sLSTMBlock

@keras.saving.register_keras_serializable(package="xLSTM")
class xLSTM(keras.Model):
    """
    Complete xLSTM architecture with stacked sLSTM and mLSTM blocks.

    This model implements the xLSTM from Section 2.4 of the paper, creating a
    deep sequence model by residually stacking xLSTM blocks. The architecture
    balances parallelizable matrix memory (mLSTM) with stateful recurrent memory
    (sLSTM) for optimal sequence modeling.

    **Intent**: Provide a complete, configurable xLSTM model for sequence-to-sequence
    tasks, language modeling, and other sequential prediction tasks.

    **Architecture**:
    ```
    Tokens
       ↓
    Embedding
       ↓
    [xLSTM Blocks] × num_layers
       ↓
    Final Normalization
       ↓
    Output Head (Dense)
    ```

    The blocks are distributed according to mlstm_ratio:
    - First num_layers * mlstm_ratio blocks are mLSTM
    - Remaining blocks are sLSTM

    Args:
        vocab_size: Integer, size of the vocabulary.
        embed_dim: Integer, dimensionality of token embeddings.
        num_layers: Integer, total number of xLSTM blocks.
        mlstm_ratio: Float in [0, 1], fraction of layers that should be mLSTM.
            Defaults to 0.5 (balanced architecture).
        mlstm_num_heads: Integer, number of heads for mLSTM blocks. Defaults to 4.
        mlstm_expansion_factor: Integer, expansion factor for mLSTM. Defaults to 2.
        slstm_forget_gate: Literal['sigmoid', 'exp'], sLSTM forget gate activation.
            Defaults to 'sigmoid'.
        ffn_type: String, FFN type for sLSTM blocks. Defaults to 'swiglu'.
        ffn_expansion_factor: Integer, FFN expansion for sLSTM. Defaults to 2.
        normalization_type: String, type of normalization. Defaults to 'layer_norm'.
        normalization_kwargs: Optional dict of normalization kwargs.
        dropout_rate: Float, dropout rate for FFN in sLSTM blocks. Defaults to 0.0.
        embedding_dropout_rate: Float, dropout after embedding. Defaults to 0.0.
        kernel_initializer: Initializer for kernel weights.
        recurrent_initializer: Initializer for recurrent weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        2D integer tensor with shape: `(batch_size, sequence_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, vocab_size)`.

    Example:
        ```python
        # Create xLSTM model for language modeling
        model = xLSTM(
            vocab_size=50000,
            embed_dim=512,
            num_layers=12,
            mlstm_ratio=0.5,
            mlstm_num_heads=8,
            ffn_type='swiglu',
            normalization_type='rms_norm'
        )

        # Compile and train
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate predictions
        tokens = keras.random.randint(0, 50000, shape=(4, 128))
        logits = model(tokens)
        print(logits.shape)  # (4, 128, 50000)
        ```
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        mlstm_ratio: float = 0.5,
        mlstm_num_heads: int = 4,
        mlstm_expansion_factor: int = 2,
        slstm_forget_gate: Literal['sigmoid', 'exp'] = 'sigmoid',
        ffn_type: str = 'swiglu',
        ffn_expansion_factor: int = 2,
        normalization_type: str = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.0,
        embedding_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= mlstm_ratio <= 1:
            raise ValueError(f"mlstm_ratio must be in [0, 1], got {mlstm_ratio}")

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlstm_ratio = mlstm_ratio
        self.mlstm_num_heads = mlstm_num_heads
        self.mlstm_expansion_factor = mlstm_expansion_factor
        self.slstm_forget_gate = slstm_forget_gate
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.dropout_rate = dropout_rate
        self.embedding_dropout_rate = embedding_dropout_rate
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name='embedding',
        )

        # Optional embedding dropout
        if embedding_dropout_rate > 0:
            self.embedding_dropout = layers.Dropout(
                rate=embedding_dropout_rate,
                name='embedding_dropout',
            )
        else:
            self.embedding_dropout = None

        # Create xLSTM blocks
        self.blocks = []
        num_mlstm = int(num_layers * mlstm_ratio)

        for i in range(num_layers):
            if i < num_mlstm:
                # mLSTM block
                block = mLSTMBlock(
                    units=embed_dim,
                    expansion_factor=mlstm_expansion_factor,
                    num_heads=mlstm_num_heads,
                    normalization_type=normalization_type,
                    normalization_kwargs=normalization_kwargs,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f'mlstm_block_{i}',
                )
            else:
                # sLSTM block
                block = sLSTMBlock(
                    units=embed_dim,
                    ffn_type=ffn_type,
                    ffn_expansion_factor=ffn_expansion_factor,
                    normalization_type=normalization_type,
                    normalization_kwargs=normalization_kwargs,
                    forget_gate_activation=slstm_forget_gate,
                    dropout_rate=dropout_rate,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f'slstm_block_{i}',
                )

            self.blocks.append(block)

        # Final normalization
        self.final_norm = create_normalization_layer(
            normalization_type=normalization_type,
            name='final_norm',
            **self.normalization_kwargs
        )

        # Output head
        self.output_head = layers.Dense(
            vocab_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='output_head',
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass through the xLSTM model.

        Args:
            inputs: Integer tensor of token IDs, shape (batch_size, seq_len).
            training: Boolean, whether in training mode.
            mask: Optional mask tensor.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Embedding
        x = self.embedding(inputs, training=training)

        # Embedding dropout
        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x, training=training)

        # Pass through xLSTM blocks
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        # Final normalization
        x = self.final_norm(x, training=training)

        # Output projection
        logits = self.output_head(x, training=training)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the model."""
        config = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'mlstm_ratio': self.mlstm_ratio,
            'mlstm_num_heads': self.mlstm_num_heads,
            'mlstm_expansion_factor': self.mlstm_expansion_factor,
            'slstm_forget_gate': self.slstm_forget_gate,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'dropout_rate': self.dropout_rate,
            'embedding_dropout_rate': self.embedding_dropout_rate,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'name': self.name,
            'trainable': self.trainable,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'xLSTM':
        """Create model from configuration."""
        return cls(**config)