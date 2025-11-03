import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, List

# ---------------------------------------------------------------------
# Component 1: Byte-Level Tokenizer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ByteTokenizer(keras.layers.Layer):
    """
    A simple, stateless byte-level tokenizer for text processing.

    This layer handles the conversion between raw strings and byte-level token IDs.
    It reserves the first few token IDs for special tokens (PAD, BOS, EOS, UNK)
    and maps the 256 possible byte values to a shifted vocabulary space.

    **Intent**: To provide a simple and language-agnostic tokenization scheme
    that is directly integrated into the Keras layer ecosystem, making it
    serializable with the model.

    **Architecture**:
    - `text_to_bytes`: String -> UTF-8 bytes -> List of integer IDs
    - `tokens_to_text`: List of integer IDs -> UTF-8 bytes -> String

    **Special Tokens**:
    - `pad_token_id = 0`: Padding token.
    - `bos_token_id = 1`: Beginning-of-sequence token.
    - `eos_token_id = 2`: End-of-sequence token.
    - `unk_token_id = 3`: Unknown token (used for decoding errors).

    Args:
        vocab_size (int): The total size of the vocabulary, including special
            tokens and all possible byte values. Defaults to 260 (4 special + 256 bytes).
        byte_offset (int): The offset for byte values to accommodate special
            tokens. The first byte (0x00) will be mapped to this ID. Defaults to 4.
        **kwargs: Additional arguments for the `Layer` base class.

    Note:
        This layer is stateless and does not contain any trainable weights. It
        inherits from `keras.layers.Layer` primarily for serialization purposes.
    """
    def __init__(self, vocab_size: int = 260, byte_offset: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        ## FIX: This layer has no weights and should not be trainable.
        self.trainable = False

    def text_to_bytes(
        self, text: str, add_bos: bool, add_eos: bool
    ) -> List[int]:
        """Converts a string to a list of byte token IDs."""
        tokens = [b + self.byte_offset for b in text.encode("utf-8", "replace")]
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            ## FIX: Use append to add the token to the end, not the beginning.
            tokens.append(self.eos_token_id)
        return tokens

    def tokens_to_text(self, token_ids: List[int]) -> str:
        """Converts a list of byte token IDs back to a string."""
        byte_values = [
            t - self.byte_offset for t in token_ids if t >= self.byte_offset
        ]
        try:
            return bytes(byte_values).decode("utf-8", "replace")
        except (UnicodeDecodeError, ValueError):
            return ""

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "byte_offset": self.byte_offset,
        })
        return config


# ---------------------------------------------------------------------
# Component 2: Hash N-Gram Embeddings
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HashNGramEmbedding(keras.layers.Layer):
    """
    Computes hash n-gram embeddings for byte-level tokens.

    This layer generates embeddings for overlapping n-grams of byte tokens using
    a polynomial hashing trick. It creates separate embedding tables for each
    n-gram size specified and sums their outputs. This allows the model to
    capture subword information without a predefined vocabulary.

    **Intent**: To enrich byte-level representations with local contextual
    information from n-grams, improving the model's ability to understand
    multi-byte characters and common subwords.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len])
       |
       +--> Compute 2-gram hashes -> EmbeddingLookup -> Emb_2
       +--> Compute 3-gram hashes -> EmbeddingLookup -> Emb_3
       +--> ...
       |
    Sum(Emb_2, Emb_3, ...)
       |
    Output(shape=[batch, seq_len, embed_dim])
    ```
    **Mathematical Operation (Hashing)**:
    A polynomial rolling hash is computed for each n-gram window:
    `hash(c1, c2, ..., cn) = (c1*base^(n-1) + ... + cn*base^0) % hash_vocab_size`

    Args:
        hash_vocab_size (int): The size of the vocabulary for each hash
            embedding table (the number of hash buckets). Must be positive.
        embed_dim (int): The dimensionality of the n-gram embedding vectors.
            Must be positive.
        ngram_sizes (List[int]): A list of integers specifying the sizes of
            n-grams to use (e.g., [2, 3] for bigrams and trigrams).
        **kwargs: Additional arguments for the `Layer` base class.
    """
    def __init__(
        self,
        hash_vocab_size: int,
        embed_dim: int,
        ngram_sizes: List[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not (hash_vocab_size > 0 and embed_dim > 0):
            raise ValueError("hash_vocab_size and embed_dim must be positive.")
        if not ngram_sizes or not all(n > 0 for n in ngram_sizes):
            raise ValueError("ngram_sizes must be a non-empty list of positive integers.")

        # Store configuration
        self.hash_vocab_size = hash_vocab_size
        self.embed_dim = embed_dim
        self.ngram_sizes = ngram_sizes

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.hash_embeddings = {
            str(n): keras.layers.Embedding(
                input_dim=self.hash_vocab_size,
                output_dim=self.embed_dim,
                name=f"hash_embedding_{n}gram",
            )
            for n in self.ngram_sizes
        }

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        for n in self.ngram_sizes:
            # The input to the embedding layers is the hashed indices, which
            # have the same shape as the original input sequence.
            self.hash_embeddings[str(n)].build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _compute_ngram_embeddings(
        self,
        inputs: keras.KerasTensor,
        n: int
    ) -> keras.KerasTensor:
        """
        Compute embeddings for a specific n-gram size using a vectorized
        polynomial hashing method compatible with Keras graph mode.
        """
        seq_len = ops.shape(inputs)[1]
        inputs = ops.cast(inputs, dtype='int64')
        base = 257  # A prime larger than the byte vocabulary (256)
        powers = ops.power(base, ops.arange(n - 1, -1, -1, dtype=inputs.dtype))
        padded_input = ops.pad(inputs, [[0, 0], [n - 1, 0]])
        ngram_components = [
            padded_input[:, i : i + seq_len] for i in range(n)
        ]
        stacked_ngrams = ops.stack(ngram_components, axis=-1)
        hash_values = ops.sum(stacked_ngrams * powers, axis=-1)
        hashed_indices = hash_values % self.hash_vocab_size
        ngram_embedding_layer = self.hash_embeddings[str(n)]
        return ngram_embedding_layer(hashed_indices)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Sum the embeddings from all specified n-gram sizes."""
        all_embeddings = [
            self._compute_ngram_embeddings(inputs, n) for n in self.ngram_sizes
        ]
        return ops.sum(ops.stack(all_embeddings), axis=0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hash_vocab_size": self.hash_vocab_size,
            "embed_dim": self.embed_dim,
            "ngram_sizes": self.ngram_sizes,
        })
        return config


# ---------------------------------------------------------------------
# Component 3: Combined Embeddings Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertBltEmbeddings(keras.layers.Layer):
    """
    Combines byte, positional, and optional hash n-gram embeddings.

    This layer is the main input processing stage for a byte-level BERT model.
    It constructs the final input representations by summing three components:
    1.  **Byte Embeddings**: A standard embedding lookup for each byte token ID.
    2.  **Positional Embeddings**: Learned embeddings representing the position
        of each token in the sequence.
    3.  **Hash N-gram Embeddings (Optional)**: Embeddings derived from local
        n-gram context, which are projected and added to the total.

    The combined embeddings are then normalized and passed through dropout.

    **Intent**: To create a rich, contextualized input representation for a
    transformer model directly from byte-level tokens.

    **Architecture**:
    ```
    Input_IDs(shape=[batch, seq_len])
       |
       +-------------> ByteEmbedding --------------+
       |                                           |
       +-------------> PositionalEmbedding --------+--> Add -> LayerNorm -> Dropout -> Output
       | (optional)                                |
       +-------------> HashNGramEmbedding -> Dense? -+
    ```

    Args:
        vocab_size (int): Size of the byte vocabulary (e.g., 260).
        hidden_size (int): The dimensionality of the final output embeddings.
        max_position_embeddings (int): The maximum sequence length the model
            can handle, determining the size of the position embedding table.
        initializer_range (float): The standard deviation for the TruncatedNormal
            initializer used for embedding tables.
        layer_norm_eps (float): Epsilon for the LayerNormalization layer.
        hidden_dropout_prob (float): Dropout probability for the final embeddings.
        use_hash_embeddings (bool): If True, enables the hash n-gram embedding branch.
        normalization_type (str): Type of normalization to use. Currently supports
            "layer_norm".
        hash_vocab_size (Optional[int]): Vocabulary size for hash embeddings.
            Required if `use_hash_embeddings` is True.
        ngram_sizes (Optional[List[int]]): N-gram sizes for hash embeddings.
            Required if `use_hash_embeddings` is True.
        hash_embedding_dim (Optional[int]): Dimension for hash embeddings.
            Required if `use_hash_embeddings` is True.
        **kwargs: Additional arguments for the `Layer` base class.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        initializer_range: float,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
        use_hash_embeddings: bool,
        normalization_type: str,
        hash_vocab_size: Optional[int] = None,
        ngram_sizes: Optional[List[int]] = None,
        hash_embedding_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # --- Validation ---
        if use_hash_embeddings and not all([hash_vocab_size, ngram_sizes, hash_embedding_dim]):
            raise ValueError(
                "If use_hash_embeddings is True, hash_vocab_size, ngram_sizes, "
                "and hash_embedding_dim must be provided."
            )

        # --- Store Configuration ---
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_hash_embeddings = use_hash_embeddings
        self.normalization_type = normalization_type
        self.hash_vocab_size = hash_vocab_size
        self.ngram_sizes = ngram_sizes
        self.hash_embedding_dim = hash_embedding_dim

        # --- CREATE all sub-layers in __init__ ---
        self.tokenizer = ByteTokenizer(vocab_size=vocab_size)

        initializer = keras.initializers.TruncatedNormal(stddev=initializer_range)
        self.byte_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=initializer,
            name="byte_embeddings",
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=max_position_embeddings,
            output_dim=hidden_size,
            embeddings_initializer=initializer,
            name="position_embeddings",
        )

        self.hash_embeddings = None
        self.hash_projection = None
        if use_hash_embeddings:
            self.hash_embeddings = HashNGramEmbedding(
                hash_vocab_size=self.hash_vocab_size,
                embed_dim=self.hash_embedding_dim,
                ngram_sizes=self.ngram_sizes,
                name="hash_embeddings",
            )
            if self.hash_embedding_dim != hidden_size:
                self.hash_projection = keras.layers.Dense(
                    hidden_size, name="hash_projection"
                )

        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm"
        )
        self.dropout = keras.layers.Dropout(hidden_dropout_prob)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers for robust serialization."""
        # The input_shape is (batch_size, seq_len)
        # Build main embedding layers
        self.byte_embeddings.build(input_shape)
        self.position_embeddings.build(input_shape)

        # Build optional hash embedding branch
        if self.use_hash_embeddings:
            self.hash_embeddings.build(input_shape)
            if self.hash_projection:
                # Input to projection is (batch, seq, hash_embedding_dim)
                hash_output_shape = (*input_shape, self.hash_embedding_dim)
                self.hash_projection.build(hash_output_shape)

        # Build final processing layers
        # Input to norm/dropout is (batch, seq, hidden_size)
        final_embedding_shape = (*input_shape, self.hidden_size)
        self.layer_norm.build(final_embedding_shape)
        self.dropout.build(final_embedding_shape)

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        seq_length = ops.shape(input_ids)[1]
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype="int32")

        byte_embeds = self.byte_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = byte_embeds + position_embeds

        if self.use_hash_embeddings:
            hash_embeds = self.hash_embeddings(input_ids)
            if self.hash_projection:
                hash_embeds = self.hash_projection(hash_embeds)
            embeddings += hash_embeds

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def encode_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> keras.KerasTensor:
        """Convenience method to tokenize text for this layer."""
        tokens = self.tokenizer.text_to_bytes(
            text, add_bos=add_special_tokens, add_eos=add_special_tokens
        )
        if max_length:
            tokens = tokens[:max_length]
            padding_length = max_length - len(tokens)
            tokens += [self.tokenizer.pad_token_id] * padding_length
        return ops.array([tokens], dtype="int32")

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        """Convenience method to decode token IDs from this layer."""
        if hasattr(token_ids, 'numpy'):
            token_ids = token_ids.numpy()
        return self.tokenizer.tokens_to_text(token_ids.flatten().tolist())

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "use_hash_embeddings": self.use_hash_embeddings,
            "normalization_type": self.normalization_type,
            "hash_vocab_size": self.hash_vocab_size,
            "ngram_sizes": self.ngram_sizes,
            "hash_embedding_dim": self.hash_embedding_dim,
        })
        return config

# ---------------------------------------------------------------------