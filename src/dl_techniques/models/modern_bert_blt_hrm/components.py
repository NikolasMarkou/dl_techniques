"""
Components for the BertBlt model, including byte-level tokenization,
hash n-gram embeddings, and the combined embeddings layer.
"""

import keras
from keras import ops
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------
# Component 1: Byte-Level Tokenizer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ByteTokenizer(keras.layers.Layer):
    """A simple, stateless byte-level tokenizer."""

    def __init__(self, vocab_size: int = 260, byte_offset: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def text_to_bytes(
            self, text: str, add_bos: bool, add_eos: bool
    ) -> List[int]:
        """Converts a string to a list of byte token IDs."""
        tokens = [b + self.byte_offset for b in text.encode("utf-8", "replace")]
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = [self.eos_token_id] + tokens
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
    """Computes hash n-gram embeddings for byte-level tokens."""

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

        self.hash_vocab_size = hash_vocab_size
        self.embed_dim = embed_dim
        self.ngram_sizes = ngram_sizes

        self.hash_embeddings = {
            str(n): keras.layers.Embedding(
                input_dim=self.hash_vocab_size,
                output_dim=self.embed_dim,
                name=f"hash_embedding_{n}gram",
            )
            for n in self.ngram_sizes
        }

    # =================================================================
    # FIXED: Replaced the Python for-loop with a vectorized implementation
    # that is compatible with Keras graph mode (Symbolic Tensors).
    # =================================================================
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

        # 1. Create powers of the base for the polynomial hash.
        # For n=3, this will be [base^2, base^1, base^0]
        powers = ops.power(base, ops.arange(n - 1, -1, -1, dtype=inputs.dtype))

        # 2. Create the n-grams for every position in a vectorized way.
        # Pad the input at the beginning to handle edge cases.
        padded_input = ops.pad(inputs, [[0, 0], [n - 1, 0]])

        # Create a list of tensors, each shifted by one position.
        ngram_components = [
            padded_input[:, i: i + seq_len] for i in range(n)
        ]

        # Stack them to form n-grams at each position.
        # Shape: (batch_size, seq_len, n)
        stacked_ngrams = ops.stack(ngram_components, axis=-1)

        # 3. Apply the vectorized polynomial hash formula.
        # (batch, seq_len, n) * (n,) -> (batch, seq_len)
        hash_values = ops.sum(stacked_ngrams * powers, axis=-1)

        # 4. Modulo to fit into the hash vocabulary.
        hashed_indices = hash_values % self.hash_vocab_size

        # 5. Look up the embeddings for the hashed indices.
        ngram_embedding_layer = self.hash_embeddings[str(n)]
        return ngram_embedding_layer(hashed_indices)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Sum the embeddings from all specified n-gram sizes."""
        total_embeddings = None
        for n in self.ngram_sizes:
            ngram_embeddings = self._compute_ngram_embeddings(inputs, n)
            if total_embeddings is None:
                total_embeddings = ngram_embeddings
            else:
                total_embeddings += ngram_embeddings
        return total_embeddings

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
class BertBltEmbeddings(keras.layers.Layer):
    """Combines byte, positional, and hash n-gram embeddings."""

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
        # Validation and parameter storage (omitted for brevity)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_hash_embeddings = use_hash_embeddings
        self.hash_embedding_dim = hash_embedding_dim

        self.tokenizer = ByteTokenizer(vocab_size=vocab_size)

        self.byte_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            name="byte_embeddings",
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=max_position_embeddings,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            name="position_embeddings",
        )

        self.hash_embeddings = None
        self.hash_projection = None
        if use_hash_embeddings:
            self.hash_embeddings = HashNGramEmbedding(
                hash_vocab_size=hash_vocab_size,
                embed_dim=hash_embedding_dim,
                ngram_sizes=ngram_sizes,
                name="hash_embeddings",
            )
            if hash_embedding_dim != hidden_size:
                self.hash_projection = keras.layers.Dense(
                    hidden_size, name="hash_projection"
                )

        if normalization_type == "layer_norm":
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=layer_norm_eps, name="layer_norm"
            )
        # Add other norm types if needed

        self.dropout = keras.layers.Dropout(hidden_dropout_prob)

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

        if self.use_hash_embeddings and self.hash_embeddings:
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
        tokens = self.tokenizer.text_to_bytes(
            text, add_bos=add_special_tokens, add_eos=add_special_tokens
        )
        if max_length:
            tokens = tokens[:max_length]
            padding_length = max_length - len(tokens)
            tokens += [self.tokenizer.pad_token_id] * padding_length
        return ops.array([tokens], dtype="int32")

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        if hasattr(token_ids, 'numpy'):
            token_ids = token_ids.numpy()
        return self.tokenizer.tokens_to_text(token_ids.flatten().tolist())

    def get_config(self):
        # ... get_config logic ...
        config = super().get_config()
        # ... update config ...
        return config