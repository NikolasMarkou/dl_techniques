"""
Masked Language Model (MLM) Pre-training Framework
====================================================

A flexible and model-agnostic framework for pre-training NLP foundation models
using the Masked Language Modeling (MLM) objective, as introduced in BERT.
This module is designed to work with any Keras-based transformer encoder.

Based on: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
(Devlin et al., 2018) https://arxiv.org/abs/1810.04805

Usage Examples:
--------------

.. code-block:: python

    import keras
    import tensorflow as tf
    from bert import BERT
    from transformers import BertTokenizer

    # 1. Create a foundation model to be pretrained
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BERT.from_variant("base", vocab_size=tokenizer.vocab_size)

    # 2. Create the MLM pre-training model
    mlm_pretrainer = MaskedLanguageModel(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        mask_ratio=0.15,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=tokenizer.all_special_ids,
    )

    # 3. Compile the model
    mlm_pretrainer.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-5)
    )

    # 4. Start pre-training
    mlm_pretrainer.fit(train_dataset, epochs=5)

    # 5. Extract the pretrained encoder for downstream tasks
    pretrained_encoder = mlm_pretrainer.encoder
    pretrained_encoder.save("pretrained_bert_encoder.keras")

"""

import keras
import tensorflow as tf
from typing import Dict, Any, Optional, Union, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.pretraining")
class MaskedLanguageModel(keras.Model):
    """A model-agnostic Masked Language Modeling (MLM) pre-trainer.

    This model wraps a given encoder (like BERT) and adds the necessary logic
    for MLM pre-training. It performs the following steps:

    1. Dynamically masks input tokens based on the BERT masking strategy.
    2. Processes the masked tokens through the encoder.
    3. Uses an MLM head to predict the original IDs of the masked tokens.
    4. Computes a sparse categorical cross-entropy loss only on the
       masked positions.

    The encoder is treated as a core component, making it easy to save and
    reuse for downstream fine-tuning after pre-training is complete.

    **Masking Strategy (BERT-style):**

    - 15% of tokens are selected for masking.
    - Of these selected tokens:
      - 80% are replaced with [MASK] token
      - 10% are replaced with a random token
      - 10% are left unchanged
    - Special tokens ([CLS], [SEP], [PAD]) are never masked.

    **Architecture:**

    .. code-block:: text

        Input(input_ids, attention_mask, ...) → Dynamic Masking
               ↓
        Encoder(masked_input_ids) → last_hidden_state
               ↓
        MLM Head(Dense → LayerNorm → Dense) → logits
               ↓
        Loss(only on masked positions) → Cross-Entropy

    :param encoder: An instance of a Keras model (e.g., BERT) that acts
        as the token encoder. It must accept a dictionary of inputs
        (`input_ids`, `attention_mask`, etc.) and return a dictionary
        containing the `last_hidden_state`.
    :type encoder: keras.Model
    :param vocab_size: The size of the vocabulary.
    :type vocab_size: int
    :param mask_ratio: The probability of a token being chosen for masking.
        Defaults to 0.15.
    :type mask_ratio: float
    :param mask_token_id: The vocabulary ID for the `[MASK]` token.
    :type mask_token_id: int
    :param random_token_ratio: The probability of replacing a masked token
        with a random token from the vocabulary. Defaults to 0.1.
    :type random_token_ratio: float
    :param unchanged_ratio: The probability of leaving a masked token as is.
        Defaults to 0.1.
    :type unchanged_ratio: float
    :param special_token_ids: A list of special token IDs (e.g., [CLS], [SEP])
        to exclude from masking. Defaults to None.
    :type special_token_ids: Optional[List[int]]
    :param mlm_head_activation: Activation function for the MLM head's
        intermediate layer. Defaults to "gelu".
    :type mlm_head_activation: str
    :param initializer_range: The standard deviation for weight initialization
        in the MLM head. Defaults to 0.02.
    :type initializer_range: float
    :param mlm_head_dropout: Dropout rate for the MLM head. Defaults to 0.1.
    :type mlm_head_dropout: float
    :param layer_norm_eps: Epsilon for LayerNormalization in MLM head.
        Defaults to 1e-12.
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar encoder: The foundational encoder model being pretrained.
    :vartype encoder: keras.Model
    :ivar mlm_dense: Dense layer for MLM head transformation.
    :vartype mlm_dense: keras.layers.Dense
    :ivar mlm_norm: LayerNormalization for MLM head.
    :vartype mlm_norm: keras.layers.LayerNormalization
    :ivar mlm_output: Output dense layer projecting to vocabulary.
    :vartype mlm_output: keras.layers.Dense
    :ivar mlm_dropout: Dropout layer for MLM head regularization.
    :vartype mlm_dropout: keras.layers.Dropout

    Example:
        .. code-block:: python

            # Create encoder and MLM pretrainer
            encoder = BERT.from_variant("base")
            mlm_model = MaskedLanguageModel(
                encoder=encoder,
                vocab_size=30522,
                mask_token_id=103
            )

            # Compile and train
            mlm_model.compile(
                optimizer=keras.optimizers.AdamW(learning_rate=5e-5)
            )
            mlm_model.fit(train_dataset, epochs=5)

            # Extract pretrained encoder
            pretrained_encoder = mlm_model.encoder
            pretrained_encoder.save("pretrained_bert.keras")
    """

    def __init__(
            self,
            encoder: keras.Model,
            vocab_size: int,
            mask_ratio: float = 0.15,
            mask_token_id: int = 103,  # Default for BERT's [MASK]
            random_token_ratio: float = 0.1,
            unchanged_ratio: float = 0.1,
            special_token_ids: Optional[List[int]] = None,
            mlm_head_activation: str = "gelu",
            initializer_range: float = 0.02,
            mlm_head_dropout: float = 0.1,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any,
    ) -> None:
        """Initialize the MaskedLanguageModel.

        :param encoder: The encoder model to pretrain.
        :type encoder: keras.Model
        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param mask_ratio: Fraction of tokens to mask.
        :type mask_ratio: float
        :param mask_token_id: ID for the [MASK] token.
        :type mask_token_id: int
        :param random_token_ratio: Fraction of masked tokens to replace with
            random tokens.
        :type random_token_ratio: float
        :param unchanged_ratio: Fraction of masked tokens to leave unchanged.
        :type unchanged_ratio: float
        :param special_token_ids: List of special token IDs to never mask.
        :type special_token_ids: Optional[List[int]]
        :param mlm_head_activation: Activation for MLM head.
        :type mlm_head_activation: str
        :param initializer_range: Standard deviation for weight initialization.
        :type initializer_range: float
        :param mlm_head_dropout: Dropout rate for MLM head.
        :type mlm_head_dropout: float
        :param layer_norm_eps: Epsilon for layer normalization.
        :type layer_norm_eps: float
        :param kwargs: Additional arguments for keras.Model.
        """
        super().__init__(**kwargs)

        # Validate configuration
        self._validate_config(
            vocab_size, mask_ratio, mask_token_id, random_token_ratio,
            unchanged_ratio, initializer_range, mlm_head_dropout
        )

        # Store all configuration parameters
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.random_token_ratio = random_token_ratio
        self.unchanged_ratio = unchanged_ratio
        self.special_token_ids = special_token_ids or []
        self.mlm_head_activation = mlm_head_activation
        self.initializer_range = initializer_range
        self.mlm_head_dropout = mlm_head_dropout
        self.layer_norm_eps = layer_norm_eps

        # Validate encoder has required attributes
        if not hasattr(self.encoder, "hidden_size"):
            raise ValueError(
                "The provided encoder must have a 'hidden_size' attribute."
            )
        self.hidden_size = self.encoder.hidden_size

        # Create MLM head components (created in __init__, not build)
        self.mlm_dense = keras.layers.Dense(
            self.hidden_size,
            activation=self.mlm_head_activation,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="mlm_dense",
        )

        self.mlm_dropout = keras.layers.Dropout(
            rate=self.mlm_head_dropout,
            name="mlm_dropout"
        )

        self.mlm_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="mlm_norm"
        )

        self.mlm_output = keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="mlm_output",
        )

        logger.info(
            f"Created MaskedLanguageModel: vocab_size={self.vocab_size}, "
            f"mask_ratio={self.mask_ratio}, hidden_size={self.hidden_size}"
        )

    def _validate_config(
            self,
            vocab_size: int,
            mask_ratio: float,
            mask_token_id: int,
            random_token_ratio: float,
            unchanged_ratio: float,
            initializer_range: float,
            mlm_head_dropout: float,
    ) -> None:
        """Validate model configuration parameters.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param mask_ratio: Fraction of tokens to mask.
        :type mask_ratio: float
        :param mask_token_id: ID for the [MASK] token.
        :type mask_token_id: int
        :param random_token_ratio: Fraction to replace with random tokens.
        :type random_token_ratio: float
        :param unchanged_ratio: Fraction to leave unchanged.
        :type unchanged_ratio: float
        :param initializer_range: Standard deviation for initialization.
        :type initializer_range: float
        :param mlm_head_dropout: Dropout rate for MLM head.
        :type mlm_head_dropout: float
        :raises ValueError: If any configuration value is invalid.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not (0.0 < mask_ratio <= 1.0):
            raise ValueError(
                f"mask_ratio must be between 0 and 1, got {mask_ratio}"
            )
        if mask_token_id < 0 or mask_token_id >= vocab_size:
            raise ValueError(
                f"mask_token_id must be in [0, {vocab_size}), got {mask_token_id}"
            )
        if not (0.0 <= random_token_ratio <= 1.0):
            raise ValueError(
                f"random_token_ratio must be between 0 and 1, "
                f"got {random_token_ratio}"
            )
        if not (0.0 <= unchanged_ratio <= 1.0):
            raise ValueError(
                f"unchanged_ratio must be between 0 and 1, "
                f"got {unchanged_ratio}"
            )
        if random_token_ratio + unchanged_ratio > 1.0:
            raise ValueError(
                f"random_token_ratio + unchanged_ratio cannot exceed 1.0, "
                f"got {random_token_ratio + unchanged_ratio}"
            )
        if initializer_range <= 0.0:
            raise ValueError(
                f"initializer_range must be positive, got {initializer_range}"
            )
        if not (0.0 <= mlm_head_dropout < 1.0):
            raise ValueError(
                f"mlm_head_dropout must be between 0 and 1, "
                f"got {mlm_head_dropout}"
            )

    def call(
            self,
            inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
            training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """Forward pass for prediction.

        Note: During inference, this model does not perform masking and simply
        returns logits for all tokens. For pre-training with dynamic masking,
        use `train_step`.

        :param inputs: A dictionary of input tensors, including 'input_ids',
            or a single tensor of input_ids.
        :type inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor]
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Logits over the vocabulary for each input token.
            Shape: (batch_size, seq_len, vocab_size)
        :rtype: keras.KerasTensor
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(inputs, training=training)
        sequence_output = encoder_outputs["last_hidden_state"]

        # Apply MLM head
        logits = self._apply_mlm_head(sequence_output, training=training)

        return logits

    def _apply_mlm_head(
            self,
            sequence_output: keras.KerasTensor,
            training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """Apply the MLM prediction head to encoder outputs.

        :param sequence_output: Output from the encoder.
            Shape: (batch_size, seq_len, hidden_size)
        :type sequence_output: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Logits over vocabulary.
            Shape: (batch_size, seq_len, vocab_size)
        :rtype: keras.KerasTensor
        """
        # Dense transformation with activation
        hidden_states = self.mlm_dense(sequence_output)

        # Apply dropout during training
        hidden_states = self.mlm_dropout(hidden_states, training=training)

        # Layer normalization
        hidden_states = self.mlm_norm(hidden_states)

        # Project to vocabulary
        logits = self.mlm_output(hidden_states)

        return logits

    def _mask_tokens(
            self, inputs: Dict[str, keras.KerasTensor]
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, keras.KerasTensor]:
        """Performs dynamic token masking according to BERT's strategy.

        Masking Strategy:
        - 15% of tokens are selected for masking (excluding special tokens).
        - Of the selected tokens:
          - 80% are replaced with [MASK] token
          - 10% are replaced with a random token
          - 10% are left unchanged

        :param inputs: The original, unmasked input dictionary containing
            'input_ids' and optionally 'attention_mask'.
        :type inputs: Dict[str, keras.KerasTensor]
        :return: A tuple containing:
                 - The new inputs with masked tokens.
                 - The original token IDs (labels).
                 - A boolean mask indicating which tokens were masked.
        :rtype: Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, keras.KerasTensor]
        """
        input_ids = inputs["input_ids"]
        labels = tf.identity(input_ids)

        # Create a boolean mask for special tokens and padding
        special_mask = tf.zeros_like(input_ids, dtype=tf.bool)
        for token_id in self.special_token_ids:
            special_mask = tf.logical_or(special_mask, input_ids == token_id)

        # Exclude padding tokens from being masked
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            padding_mask = attention_mask == 0
            special_mask = tf.logical_or(special_mask, padding_mask)

        # Determine which tokens to mask (15% of non-special tokens)
        mask_probabilities = tf.random.uniform(tf.shape(input_ids), dtype=tf.float32)
        should_mask = (mask_probabilities < self.mask_ratio) & ~special_mask
        masked_indices = tf.where(should_mask)

        # If no tokens are masked, return original inputs
        num_masked = tf.shape(masked_indices)[0]
        if num_masked == 0:
            return inputs, labels, should_mask

        # Decide on the masking strategy for each masked token
        # 80% -> [MASK], 10% -> random token, 10% -> unchanged
        mask_strategy_probs = tf.random.uniform(
            shape=(num_masked,), dtype=tf.float32
        )

        # Calculate thresholds
        mask_threshold = 1.0 - self.random_token_ratio - self.unchanged_ratio
        random_threshold = 1.0 - self.unchanged_ratio

        # 80% -> Replace with [MASK]
        mask_token_mask = mask_strategy_probs < mask_threshold
        mask_token_indices = tf.boolean_mask(masked_indices, mask_token_mask)

        # 10% -> Replace with random token
        random_token_mask = (mask_strategy_probs >= mask_threshold) & (
                mask_strategy_probs < random_threshold
        )
        random_token_indices = tf.boolean_mask(masked_indices, random_token_mask)

        # Generate random tokens for replacement
        num_random = tf.shape(random_token_indices)[0]
        random_tokens = tf.random.uniform(
            shape=(num_random,),
            minval=0,
            maxval=self.vocab_size,
            dtype=tf.int32,
        )

        # 10% -> Leave unchanged (no operation needed)

        # Update input_ids with the masked values
        masked_input_ids = input_ids

        # Apply [MASK] replacements
        if tf.shape(mask_token_indices)[0] > 0:
            mask_values = tf.fill(
                (tf.shape(mask_token_indices)[0],), self.mask_token_id
            )
            masked_input_ids = tf.tensor_scatter_nd_update(
                masked_input_ids, mask_token_indices, mask_values
            )

        # Apply random token replacements
        if tf.shape(random_token_indices)[0] > 0:
            masked_input_ids = tf.tensor_scatter_nd_update(
                masked_input_ids, random_token_indices, random_tokens
            )

        # Create new inputs dictionary with masked input_ids
        new_inputs = inputs.copy()
        new_inputs["input_ids"] = masked_input_ids

        return new_inputs, labels, should_mask

    def train_step(
            self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        """Custom training step for MLM with dynamic masking.

        This method performs:
        1. Dynamic token masking using BERT's strategy.
        2. Forward pass through encoder and MLM head.
        3. Loss computation only on masked positions.
        4. Gradient computation and weight updates.
        5. Metric updates.

        :param data: A batch of data from the training dataset. Can be a
            dictionary of inputs or a tuple of (inputs, labels, sample_weight).
        :type data: Union[Dict[str, keras.KerasTensor], Tuple]
        :return: A dictionary of metrics (loss and any compiled metrics).
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Unpack data
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        # Apply dynamic masking
        masked_inputs, labels, masked_positions = self._mask_tokens(inputs)

        # Forward pass with gradient tape
        with tf.GradientTape() as tape:
            # Get encoder outputs
            encoder_outputs = self.encoder(masked_inputs, training=True)
            sequence_output = encoder_outputs["last_hidden_state"]

            # Apply MLM head
            logits = self._apply_mlm_head(sequence_output, training=True)

            # Compute loss only on masked tokens
            loss = self.compute_loss(
                y=labels,
                y_pred=logits,
                sample_weight=masked_positions,
            )

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        return self.compute_metrics(
            x=None,
            y=labels,
            y_pred=logits,
            sample_weight=masked_positions,
        )

    def compute_loss(
            self,
            x: Optional[keras.KerasTensor] = None,
            y: Optional[keras.KerasTensor] = None,
            y_pred: Optional[keras.KerasTensor] = None,
            sample_weight: Optional[keras.KerasTensor] = None,
            **kwargs: Any,
    ) -> keras.KerasTensor:
        """Computes MLM loss.

        Loss is calculated using sparse categorical cross-entropy, but only for
        the tokens that were masked.

        :param x: Input data (unused for MLM loss).
        :type x: Optional[keras.KerasTensor]
        :param y: The original token IDs (labels).
            Shape: (batch_size, seq_len)
        :type y: Optional[keras.KerasTensor]
        :param y_pred: The predicted logits from the MLM head.
            Shape: (batch_size, seq_len, vocab_size)
        :type y_pred: Optional[keras.KerasTensor]
        :param sample_weight: A boolean mask where `True` indicates a
            masked token. Shape: (batch_size, seq_len)
        :type sample_weight: Optional[keras.KerasTensor]
        :return: The computed MLM loss (scalar).
        :rtype: keras.KerasTensor
        """
        # Create loss function
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

        # Compute per-token loss
        loss = loss_fn(y, y_pred)

        # Apply mask to compute loss only on masked tokens
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=loss.dtype)
            loss = loss * sample_weight

            # Return mean loss over the masked tokens
            num_masked = tf.maximum(tf.reduce_sum(sample_weight), 1.0)
            return tf.reduce_sum(loss) / num_masked
        else:
            # If no mask provided, return mean loss over all tokens
            return tf.reduce_mean(loss)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the model for serialization.

        :return: A dictionary containing all configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "vocab_size": self.vocab_size,
                "mask_ratio": self.mask_ratio,
                "mask_token_id": self.mask_token_id,
                "random_token_ratio": self.random_token_ratio,
                "unchanged_ratio": self.unchanged_ratio,
                "special_token_ids": self.special_token_ids,
                "mlm_head_activation": self.mlm_head_activation,
                "initializer_range": self.initializer_range,
                "mlm_head_dropout": self.mlm_head_dropout,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskedLanguageModel":
        """Creates a model from its configuration.

        :param config: A dictionary containing the model's configuration.
        :type config: Dict[str, Any]
        :return: A new MaskedLanguageModel instance.
        :rtype: MaskedLanguageModel
        """
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)
        return cls(encoder=encoder, **config)


# ---------------------------------------------------------------------
