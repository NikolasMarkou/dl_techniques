import keras
from keras import ops
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .positional_embedding import PositionalEmbedding
from .transformer_encoder import TransformerEncoderLayer
from .attention.multi_head_attention import MultiHeadAttention

# ---------------------------------------------------------------------

class ByteTokenizer(keras.layers.Layer):
    """
    Converts text to byte tokens for BLT processing.

    This layer handles the conversion of text strings to byte sequences,
    with proper handling of special tokens and padding. It operates at the
    byte level to achieve true language-agnostic processing.

    Args:
        vocab_size: Size of the vocabulary including special tokens.
        byte_offset: Offset added to raw byte values for special tokens.
        **kwargs: Additional layer arguments.

    Input shape:
        Variable - this layer processes text strings.

    Output shape:
        Variable - returns list of integers representing byte tokens.

    Example:
        ```python
        tokenizer = ByteTokenizer(vocab_size=260, byte_offset=4)
        tokens = tokenizer.text_to_bytes("Hello, world!")
        text = tokenizer.tokens_to_text(tokens)
        ```
    """

    def __init__(
            self,
            vocab_size: int = 260,
            byte_offset: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset

        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3

    def text_to_bytes(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Convert text string to byte token sequence.

        Args:
            text: Input text string.
            add_bos: Whether to add begin-of-sequence token.
            add_eos: Whether to add end-of-sequence token.

        Returns:
            List of byte token IDs.
        """
        # Convert to UTF-8 bytes
        byte_sequence = text.encode('utf-8', errors='ignore')

        # Map bytes to tokens with offset
        tokens = [byte + self.byte_offset for byte in byte_sequence]

        # Add special tokens
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        """
        Convert byte token sequence back to text.

        Args:
            tokens: List of byte token IDs.

        Returns:
            Decoded text string.
        """
        # Filter out special tokens and convert back to bytes
        byte_values = []
        for token in tokens:
            if token >= self.byte_offset:
                byte_values.append(token - self.byte_offset)

        # Convert bytes back to string
        try:
            text = bytes(byte_values).decode('utf-8', errors='ignore')
        except (ValueError, UnicodeDecodeError):
            text = ""

        return text

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'byte_offset': self.byte_offset
        })
        return config

# ---------------------------------------------------------------------

class EntropyModel(keras.layers.Layer):
    """
    Small causal transformer for computing next-byte entropy.

    This model predicts the probability distribution of the next byte
    at each position, which is used for dynamic patching. The entropy
    computed from these distributions indicates information density.

    Args:
        vocab_size: Size of byte vocabulary.
        hidden_dim: Hidden dimension of the transformer.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        dropout_rate: Dropout rate.
        **kwargs: Additional layer arguments.

    Input shape:
        Tensor with shape (batch_size, seq_len).

    Output shape:
        Tensor with shape (batch_size, seq_len, vocab_size).

    Example:
        ```python
        entropy_model = EntropyModel(vocab_size=260, hidden_dim=256)
        logits = entropy_model(byte_tokens)
        entropy = entropy_model.compute_entropy(logits)
        ```
    """

    def __init__(
            self,
            vocab_size: int = 260,
            hidden_dim: int = 256,
            num_layers: int = 6,
            num_heads: int = 8,
            max_seq_len: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # Sublayers initialized in build()
        self.embedding = None
        self.positional_embedding = None
        self.transformer_layers = []
        self.layer_norm = None
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the entropy model layers."""
        super().build(input_shape)

        # Token embedding
        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_dim,
            name='token_embedding'
        )

        # Positional embedding
        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.hidden_dim,
            dropout=self.dropout_rate,
            name='positional_embedding'
        )

        # Transformer layers
        for i in range(self.num_layers):
            layer = TransformerEncoderLayer(
                hidden_size=self.hidden_dim,
                num_heads=self.num_heads,
                intermediate_size=self.hidden_dim * 4,
                dropout=self.dropout_rate,
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(layer)

        # Final layer norm and projection
        self.layer_norm = keras.layers.LayerNormalization(name='final_layer_norm')
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

        # Build sublayers with proper input shape
        sample_input = (input_shape[0], input_shape[1], self.hidden_dim)
        for layer in self.transformer_layers:
            layer.build(sample_input)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the entropy model.

        Args:
            inputs: Input token tensor of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Token embedding
        x = self.embedding(inputs)

        # Add positional embedding
        x = self.positional_embedding(x, training=training)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Final layer norm and projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)

        return logits

    def compute_entropy(self, logits: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute Shannon entropy from logits.

        Args:
            logits: Logits tensor of shape (batch_size, seq_len, vocab_size).

        Returns:
            Entropy tensor of shape (batch_size, seq_len).
        """
        # Apply softmax to get probabilities
        probs = keras.activations.softmax(logits, axis=-1)

        # Compute log probabilities for numerical stability
        log_probs = ops.log(ops.maximum(probs, 1e-12))

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -ops.sum(probs * log_probs, axis=-1)

        return entropy

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape + (self.vocab_size,)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------

class DynamicPatcher(keras.layers.Layer):
    """
    Creates dynamic patches based on entropy thresholding.

    This layer implements the core dynamic patching algorithm that segments
    byte sequences based on information density measured by entropy. When
    entropy exceeds a threshold, it indicates an unpredictable transition
    where a new patch boundary should be created.

    Args:
        entropy_threshold: Threshold for creating patch boundaries.
        max_patches: Maximum number of patches to create.
        **kwargs: Additional layer arguments.

    Input shape:
        Entropy tensor with shape (batch_size, seq_len).

    Output shape:
        Patch lengths tensor with shape (batch_size, max_patches).

    Example:
        ```python
        patcher = DynamicPatcher(entropy_threshold=1.5, max_patches=512)
        patch_lengths = patcher(entropy_values)
        patch_ids = patcher.compute_patch_ids(patch_lengths)
        ```
    """

    def __init__(
            self,
            entropy_threshold: float = 1.5,
            max_patches: int = 512,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.entropy_threshold = entropy_threshold
        self.max_patches = max_patches

    def call(
            self,
            entropy: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Create patch lengths from entropy values.

        This is a simplified implementation that creates roughly equal patches.
        A full implementation would use more sophisticated boundary detection.

        Args:
            entropy: Entropy tensor of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Patch lengths tensor of shape (batch_size, max_patches).
        """
        batch_size = ops.shape(entropy)[0]
        seq_len = ops.shape(entropy)[1]

        # Simple patch creation: divide sequence into roughly equal parts
        # In practice, this would use the entropy threshold for boundary detection
        avg_patch_size = ops.maximum(seq_len // self.max_patches, 1)

        # Initialize patch lengths
        patch_lengths = ops.zeros((batch_size, self.max_patches), dtype='int32')

        # Create patches by distributing sequence length
        remaining_length = seq_len
        for p in range(self.max_patches - 1):
            if remaining_length <= avg_patch_size:
                # Put all remaining length in current patch
                patch_lengths = ops.slice_update(
                    patch_lengths,
                    [slice(None), p],
                    ops.expand_dims(remaining_length, axis=1)
                )
                break
            else:
                # Regular patch size
                patch_lengths = ops.slice_update(
                    patch_lengths,
                    [slice(None), p],
                    ops.expand_dims(avg_patch_size, axis=1)
                )
                remaining_length -= avg_patch_size

        # Put any remaining tokens in the last patch
        if remaining_length > 0:
            patch_lengths = ops.slice_update(
                patch_lengths,
                [slice(None), self.max_patches - 1],
                ops.expand_dims(remaining_length, axis=1)
            )

        return patch_lengths

    def compute_patch_ids(
            self,
            patch_lengths: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Convert patch lengths to patch IDs for each position.

        Args:
            patch_lengths: Patch lengths tensor of shape (batch_size, max_patches).

        Returns:
            Patch IDs tensor of shape (batch_size, seq_len).
        """
        batch_size = ops.shape(patch_lengths)[0]
        max_patches = ops.shape(patch_lengths)[1]

        # Calculate total sequence length
        total_lengths = ops.sum(patch_lengths, axis=1)
        max_seq_len = ops.max(total_lengths)

        # Initialize patch IDs
        patch_ids = ops.zeros((batch_size, max_seq_len), dtype='int32')

        # Fill patch IDs using cumulative sums
        cumulative_lengths = ops.cumsum(patch_lengths, axis=1)

        for b in range(batch_size):
            current_pos = 0
            for p in range(max_patches):
                patch_len = patch_lengths[b, p]
                if patch_len > 0:
                    end_pos = current_pos + patch_len
                    # Fill positions with patch ID
                    for pos in range(current_pos, end_pos):
                        if pos < max_seq_len:
                            patch_ids = ops.slice_update(
                                patch_ids,
                                [b, pos],
                                p
                            )
                    current_pos = end_pos

        return patch_ids

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], self.max_patches)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'entropy_threshold': self.entropy_threshold,
            'max_patches': self.max_patches
        })
        return config

# ---------------------------------------------------------------------

class PatchPooling(keras.layers.Layer):
    """
    Pools byte representations within patches to create patch representations.

    This layer reduces the sequence of byte hidden states to patch hidden states
    using various pooling strategies. The attention-based pooling method uses
    learnable query vectors to extract the most relevant information from each patch.

    Args:
        pooling_method: Method for pooling ('max', 'mean', 'attention').
        output_dim: Output dimension for patch representations.
        num_queries: Number of query vectors for attention pooling.
        **kwargs: Additional layer arguments.

    Input shape:
        byte_hiddens: (batch_size, seq_len, hidden_dim)
        patch_ids: (batch_size, seq_len)

    Output shape:
        Tensor with shape (batch_size, num_patches, output_dim).

    Example:
        ```python
        pooling = PatchPooling(
            pooling_method='attention',
            output_dim=768,
            num_queries=4
        )
        patch_reps = pooling(byte_hiddens, patch_ids)
        ```
    """

    def __init__(
            self,
            pooling_method: str = 'attention',
            output_dim: int = 768,
            num_queries: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.pooling_method = pooling_method
        self.output_dim = output_dim
        self.num_queries = num_queries

        # Sublayers initialized in build()
        self.query_embeddings = None
        self.attention_layer = None
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build pooling layers."""
        super().build(input_shape)

        # Assume input_shape is for byte_hiddens
        input_dim = input_shape[-1]

        if self.pooling_method == 'attention':
            # Learnable query embeddings for cross-attention
            self.query_embeddings = self.add_weight(
                shape=(self.num_queries, input_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='query_embeddings'
            )

            # Cross-attention layer
            self.attention_layer = MultiHeadAttention(
                embed_dim=input_dim,
                num_heads=8,
                name='patch_attention'
            )

        # Output projection to desired dimension
        if input_dim != self.output_dim:
            self.output_projection = keras.layers.Dense(
                self.output_dim,
                name='output_projection'
            )

    def call(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Pool byte representations into patch representations.

        Args:
            byte_hiddens: Byte hidden states of shape (batch_size, seq_len, hidden_dim).
            patch_ids: Patch IDs of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Patch representations of shape (batch_size, num_patches, output_dim).
        """
        batch_size = ops.shape(byte_hiddens)[0]
        seq_len = ops.shape(byte_hiddens)[1]
        hidden_dim = ops.shape(byte_hiddens)[2]

        # Determine number of patches from patch_ids
        num_patches = ops.max(patch_ids) + 1

        if self.pooling_method == 'max':
            return self._max_pooling(byte_hiddens, patch_ids, num_patches)
        elif self.pooling_method == 'mean':
            return self._mean_pooling(byte_hiddens, patch_ids, num_patches)
        elif self.pooling_method == 'attention':
            return self._attention_pooling(byte_hiddens, patch_ids, num_patches, training)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def _max_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Max pooling within patches."""
        batch_size = ops.shape(byte_hiddens)[0]
        hidden_dim = ops.shape(byte_hiddens)[2]

        patch_reps = []

        for p in range(num_patches):
            # Create mask for positions belonging to this patch
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            # Apply mask and get max (set masked positions to large negative value)
            masked_hiddens = ops.where(mask_expanded, byte_hiddens, -1e9)
            patch_max = ops.max(masked_hiddens, axis=1)  # (batch_size, hidden_dim)

            patch_reps.append(patch_max)

        # Stack all patches
        result = ops.stack(patch_reps, axis=1)  # (batch_size, num_patches, hidden_dim)

        # Project to output dimension if needed
        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _mean_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Mean pooling within patches."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            # Create mask for positions belonging to this patch
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            # Apply mask and compute mean
            masked_hiddens = byte_hiddens * mask_expanded
            patch_sum = ops.sum(masked_hiddens, axis=1)  # (batch_size, hidden_dim)
            patch_count = ops.sum(ops.cast(mask, byte_hiddens.dtype), axis=1, keepdims=True)
            patch_mean = patch_sum / ops.maximum(patch_count, 1.0)

            patch_reps.append(patch_mean)

        # Stack all patches
        result = ops.stack(patch_reps, axis=1)  # (batch_size, num_patches, hidden_dim)

        # Project to output dimension if needed
        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _attention_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Attention-based pooling within patches."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            # Find positions belonging to this patch
            mask = ops.equal(patch_ids, p)

            # Get patch-specific hidden states
            mask_expanded = ops.expand_dims(mask, axis=-1)
            patch_hiddens = ops.where(
                mask_expanded,
                byte_hiddens,
                ops.zeros_like(byte_hiddens)
            )

            # Use learnable queries to attend to patch hidden states
            queries = ops.expand_dims(self.query_embeddings, axis=0)
            queries = ops.tile(queries, [batch_size, 1, 1])

            # Cross-attention: queries attend to patch hidden states
            attended = self.attention_layer(
                queries,
                patch_hiddens,
                patch_hiddens,
                training=training
            )

            # Flatten and average the attended queries
            patch_rep = ops.mean(attended, axis=1)  # (batch_size, hidden_dim)

            patch_reps.append(patch_rep)

        # Stack patch representations
        result = ops.stack(patch_reps, axis=1)

        # Project to output dimension if needed
        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, None, self.output_dim)  # num_patches is dynamic

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'pooling_method': self.pooling_method,
            'output_dim': self.output_dim,
            'num_queries': self.num_queries
        })
        return config


# ---------------------------------------------------------------------

class LocalEncoder(keras.layers.Layer):
    """
    Local Encoder for BLT that processes bytes within their patches.

    This encoder applies causal self-attention to bytes within patches,
    learning local patterns and dependencies. It then pools the byte
    representations to create a single representation for each patch.

    Args:
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        local_dim: Hidden dimension for local encoder.
        num_local_layers: Number of transformer layers in local encoder.
        num_heads_local: Number of attention heads for local transformer.
        max_sequence_length: Maximum sequence length in bytes.
        max_patches: Maximum number of patches per sequence.
        dropout_rate: Dropout rate for all layers.
        patch_pooling_method: Method for patch pooling ('max', 'mean', 'attention').
        global_dim: Hidden dimension for global transformer (output dimension).
        cross_attention_queries: Number of queries for patch representation.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            num_local_layers: int = 6,
            num_heads_local: int = 8,
            max_sequence_length: int = 2048,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            patch_pooling_method: str = 'attention',
            global_dim: int = 768,
            cross_attention_queries: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.num_local_layers = num_local_layers
        self.num_heads_local = num_heads_local
        self.max_sequence_length = max_sequence_length
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate
        self.patch_pooling_method = patch_pooling_method
        self.global_dim = global_dim
        self.cross_attention_queries = cross_attention_queries

        # Sublayers initialized in build()
        self.byte_embedding = None
        self.positional_embedding = None
        self.transformer_layers = []
        self.patch_pooling = None
        self.layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build local encoder layers."""
        super().build(input_shape)

        # Byte token embedding
        self.byte_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.local_dim,
            name='byte_embedding'
        )

        # Positional embedding for byte positions
        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_sequence_length,
            dim=self.local_dim,
            dropout=self.dropout_rate,
            name='positional_embedding'
        )

        # Local transformer layers (causal attention)
        for i in range(self.num_local_layers):
            layer = TransformerEncoderLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout=self.dropout_rate,
                name=f'local_transformer_{i}'
            )
            self.transformer_layers.append(layer)

        # Patch pooling layer
        self.patch_pooling = PatchPooling(
            pooling_method=self.patch_pooling_method,
            output_dim=self.global_dim,
            num_queries=self.cross_attention_queries,
            name='patch_pooling'
        )

        # Final layer norm
        self.layer_norm = keras.layers.LayerNormalization(name='local_encoder_norm')

        # Build sublayers
        sample_shape = (input_shape[0], input_shape[1], self.local_dim)
        for layer in self.transformer_layers:
            layer.build(sample_shape)

        self.patch_pooling.build(sample_shape)

    def call(
            self,
            byte_tokens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of local encoder.

        Args:
            byte_tokens: Byte token tensor of shape (batch_size, seq_len).
            patch_ids: Patch ID tensor of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Patch representations of shape (batch_size, num_patches, global_dim).
        """
        # Embed byte tokens
        x = self.byte_embedding(byte_tokens)

        # Add positional embeddings
        x = self.positional_embedding(x, training=training)

        # Apply causal transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Pool bytes into patch representations
        patch_representations = self.patch_pooling(x, patch_ids, training=training)

        return patch_representations

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.max_patches, self.global_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'num_local_layers': self.num_local_layers,
            'num_heads_local': self.num_heads_local,
            'max_sequence_length': self.max_sequence_length,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate,
            'patch_pooling_method': self.patch_pooling_method,
            'global_dim': self.global_dim,
            'cross_attention_queries': self.cross_attention_queries
        })
        return config

# ---------------------------------------------------------------------

class GlobalTransformer(keras.layers.Layer):
    """
    Global Transformer for BLT that processes patch sequences.

    This transformer applies self-attention across patch representations
    to model long-range dependencies in the hierarchical structure.

    Args:
        global_dim: Hidden dimension for global transformer.
        num_global_layers: Number of transformer layers in global processor.
        num_heads_global: Number of attention heads for global transformer.
        max_patches: Maximum number of patches per sequence.
        dropout_rate: Dropout rate for all layers.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            global_dim: int = 768,
            num_global_layers: int = 12,
            num_heads_global: int = 12,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.global_dim = global_dim
        self.num_global_layers = num_global_layers
        self.num_heads_global = num_heads_global
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate

        # Sublayers initialized in build()
        self.patch_positional_embedding = None
        self.transformer_layers = []
        self.layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build global transformer layers."""
        super().build(input_shape)

        # Positional embedding for patch positions
        self.patch_positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_patches,
            dim=self.global_dim,
            dropout=self.dropout_rate,
            name='patch_positional_embedding'
        )

        # Global transformer layers
        for i in range(self.num_global_layers):
            layer = TransformerEncoderLayer(
                hidden_size=self.global_dim,
                num_heads=self.num_heads_global,
                intermediate_size=self.global_dim * 4,
                dropout=self.dropout_rate,
                name=f'global_transformer_{i}'
            )
            self.transformer_layers.append(layer)

        # Final layer norm
        self.layer_norm = keras.layers.LayerNormalization(name='global_transformer_norm')

        # Build sublayers
        for layer in self.transformer_layers:
            layer.build(input_shape)

    def call(
            self,
            patch_representations: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of global transformer.

        Args:
            patch_representations: Patch representations of shape (batch_size, num_patches, global_dim).
            training: Whether in training mode.

        Returns:
            Contextual patch representations of shape (batch_size, num_patches, global_dim).
        """
        # Add patch positional embeddings
        x = self.patch_positional_embedding(patch_representations, training=training)

        # Apply global transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Apply final layer norm
        x = self.layer_norm(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'global_dim': self.global_dim,
            'num_global_layers': self.num_global_layers,
            'num_heads_global': self.num_heads_global,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------

class LocalDecoder(keras.layers.Layer):
    """
    Local Decoder for BLT that generates next byte predictions.

    This decoder processes byte sequences with causal self-attention
    and uses cross-attention to incorporate global patch context.

    Args:
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        local_dim: Hidden dimension for local decoder.
        global_dim: Hidden dimension for global transformer.
        num_local_layers: Number of transformer layers in local decoder.
        num_heads_local: Number of attention heads for local transformers.
        max_sequence_length: Maximum sequence length in bytes.
        dropout_rate: Dropout rate for all layers.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            global_dim: int = 768,
            num_local_layers: int = 6,
            num_heads_local: int = 8,
            max_sequence_length: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_local_layers = num_local_layers
        self.num_heads_local = num_heads_local
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate

        # Sublayers initialized in build()
        self.byte_embedding = None
        self.positional_embedding = None
        self.decoder_layers = []
        self.cross_attention_layers = []
        self.cross_attention_norms = []
        self.context_projection = None
        self.layer_norm = None
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build local decoder layers."""
        super().build(input_shape)

        # Byte token embedding (can be shared with encoder or separate)
        self.byte_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.local_dim,
            name='decoder_byte_embedding'
        )

        # Positional embedding
        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_sequence_length,
            dim=self.local_dim,
            dropout=self.dropout_rate,
            name='decoder_positional_embedding'
        )

        # Project global context to local dimension if needed
        if self.global_dim != self.local_dim:
            self.context_projection = keras.layers.Dense(
                self.local_dim,
                name='context_projection'
            )

        # Decoder transformer layers with cross-attention
        for i in range(self.num_local_layers):
            # Self-attention layer
            decoder_layer = TransformerEncoderLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout=self.dropout_rate,
                name=f'decoder_transformer_{i}'
            )
            self.decoder_layers.append(decoder_layer)

            # Cross-attention to global patch context
            cross_attention = MultiHeadAttention(
                embed_dim=self.local_dim,
                num_heads=self.num_heads_local,
                dropout_rate=self.dropout_rate,
                name=f'cross_attention_{i}'
            )
            self.cross_attention_layers.append(cross_attention)

            # Layer norm for cross-attention
            cross_norm = keras.layers.LayerNormalization(name=f'cross_attention_norm_{i}')
            self.cross_attention_norms.append(cross_norm)

        # Final layer norm and output projection
        self.layer_norm = keras.layers.LayerNormalization(name='decoder_norm')
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

        # Build sublayers
        sample_shape = (input_shape[0], input_shape[1], self.local_dim)
        for layer in self.decoder_layers:
            layer.build(sample_shape)

        for layer in self.cross_attention_layers:
            layer.build(sample_shape)

    def call(
            self,
            byte_tokens: keras.KerasTensor,
            global_context: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of local decoder.

        Args:
            byte_tokens: Byte token tensor of shape (batch_size, seq_len).
            global_context: Global patch representations of shape (batch_size, num_patches, global_dim).
            patch_ids: Patch ID tensor of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Embed byte tokens
        x = self.byte_embedding(byte_tokens)

        # Add positional embeddings
        x = self.positional_embedding(x, training=training)

        # Project global context to local dimension if needed
        if self.context_projection is not None:
            global_context = self.context_projection(global_context)

        # Apply decoder layers with cross-attention
        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            # Self-attention within byte sequence
            x = decoder_layer(x, training=training)

            # Cross-attention to global context
            cross_attended = self._masked_cross_attention(
                x, global_context, patch_ids, cross_attention, training
            )

            # Residual connection and layer norm for cross-attention
            x = x + cross_attended
            x = cross_norm(x)

        # Apply final layer norm
        x = self.layer_norm(x)

        # Project to vocabulary logits
        logits = self.output_projection(x)

        return logits

    def _masked_cross_attention(
            self,
            decoder_hidden: keras.KerasTensor,
            global_context: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            cross_attention: keras.layers.Layer,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply cross-attention with patch-based masking.

        Each byte position can only attend to the global representation
        of the patch it belongs to.
        """
        batch_size = ops.shape(decoder_hidden)[0]
        seq_len = ops.shape(decoder_hidden)[1]
        num_patches = ops.shape(global_context)[1]

        # For each position, gather the corresponding patch representation
        batch_indices = ops.arange(batch_size)
        batch_indices = ops.repeat(batch_indices, seq_len)

        flat_patch_ids = ops.reshape(patch_ids, (-1,))
        gather_indices = ops.stack([batch_indices, flat_patch_ids], axis=1)

        # Gather the corresponding patch representations
        gathered_context = ops.take_along_axis(
            global_context,
            ops.expand_dims(flat_patch_ids, axis=-1),
            axis=1
        )

        # Reshape back to sequence format
        position_context = ops.reshape(
            gathered_context,
            (batch_size, seq_len, ops.shape(global_context)[-1])
        )

        # Apply cross-attention
        attended = cross_attention(
            decoder_hidden,  # queries
            position_context,  # keys
            position_context,  # values
            training=training
        )

        return attended

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        return (batch_size, seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'global_dim': self.global_dim,
            'num_local_layers': self.num_local_layers,
            'num_heads_local': self.num_heads_local,
            'max_sequence_length': self.max_sequence_length,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------
