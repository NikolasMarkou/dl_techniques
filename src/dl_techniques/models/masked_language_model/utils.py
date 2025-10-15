import keras
import tensorflow as tf
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .mlm import MaskedLanguageModel

# ---------------------------------------------------------------------

def visualize_mlm_predictions(
        mlm_model: MaskedLanguageModel,
        inputs: Dict[str, keras.KerasTensor],
        tokenizer: Any,
        num_samples: int = 4,
) -> None:
    """Visualizes the model's ability to fill in masked tokens.

    This utility function demonstrates the MLM model's predictions by:
    1. Applying dynamic masking to input samples.
    2. Getting model predictions for masked positions.
    3. Comparing original, masked, and predicted sequences.

    :param mlm_model: The trained MaskedLanguageModel instance.
    :type mlm_model: MaskedLanguageModel
    :param inputs: A batch of tokenized inputs containing 'input_ids' and
        optionally 'attention_mask'.
    :type inputs: Dict[str, keras.KerasTensor]
    :param tokenizer: The tokenizer used for encoding/decoding. Must have
        a `decode` method that accepts a tensor or list of token IDs.
    :type tokenizer: Any
    :param num_samples: The number of samples to visualize. Defaults to 4.
    :type num_samples: int
    """
    # Apply masking
    masked_inputs, labels, masked_positions = mlm_model._mask_tokens(inputs)

    # Get model predictions
    predictions = mlm_model(masked_inputs, training=False)
    predicted_ids = keras.ops.argmax(predictions, axis=-1)
    predicted_ids = tf.cast(predicted_ids, dtype=tf.int32)

    # Limit to num_samples
    batch_size = tf.shape(labels)[0]
    num_samples = tf.minimum(num_samples, batch_size)

    masked_input_ids = masked_inputs["input_ids"][:num_samples]
    predicted_ids = predicted_ids[:num_samples]
    labels = labels[:num_samples]
    masked_positions = masked_positions[:num_samples]

    # Convert to numpy for easier processing
    masked_input_ids = keras.ops.convert_to_numpy(masked_input_ids)
    predicted_ids = keras.ops.convert_to_numpy(predicted_ids)
    labels = keras.ops.convert_to_numpy(labels)
    masked_positions = keras.ops.convert_to_numpy(masked_positions)

    logger.info("=" * 80)
    logger.info("MLM Prediction Visualization")
    logger.info("=" * 80)

    for i in range(num_samples):
        # Decode sequences
        original_text = tokenizer.decode(
            labels[i], skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        masked_text = tokenizer.decode(
            masked_input_ids[i], skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )

        # Create filled sequence (use predictions for masked positions only)
        filled_ids = tf.where(
            masked_positions[i],
            predicted_ids[i],
            labels[i]
        )
        filled_ids = keras.ops.convert_to_numpy(filled_ids)
        filled_text = tokenizer.decode(
            filled_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )

        logger.info("-" * 80)
        logger.info(f"Sample {i + 1}")
        logger.info(f"Original:     {original_text}")
        logger.info(f"Masked Input: {masked_text}")
        logger.info(f"Prediction:   {filled_text}")
        logger.info("-" * 80)

    logger.info("=" * 80)

# ---------------------------------------------------------------------


def create_mlm_training_model(
        encoder: keras.Model,
        vocab_size: int,
        mask_token_id: int,
        special_token_ids: Optional[List[int]] = None,
        mlm_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
) -> MaskedLanguageModel:
    """Factory function to create a fully configured MLM training model.

    This convenience function creates an MLM model with sensible defaults
    and compiles it with an optimizer.

    :param encoder: The encoder model to pretrain.
    :type encoder: keras.Model
    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param mask_token_id: ID for the [MASK] token.
    :type mask_token_id: int
    :param special_token_ids: List of special token IDs to never mask.
    :type special_token_ids: Optional[List[int]]
    :param mlm_config: Configuration dictionary for MaskedLanguageModel.
        Defaults to standard BERT settings.
    :type mlm_config: Optional[Dict[str, Any]]
    :param optimizer_config: Configuration for the optimizer.
        Defaults to AdamW with learning rate 5e-5.
    :type optimizer_config: Optional[Dict[str, Any]]
    :return: A compiled MaskedLanguageModel ready for training.
    :rtype: MaskedLanguageModel

    Example:
        .. code-block:: python

            from bert import BERT
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            encoder = BERT.from_variant("base", vocab_size=tokenizer.vocab_size)

            mlm_model = create_mlm_training_model(
                encoder=encoder,
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                special_token_ids=tokenizer.all_special_ids,
            )

            # Ready to train
            mlm_model.fit(train_dataset, epochs=5)
    """
    # Default MLM configuration
    default_mlm_config = {
        "mask_ratio": 0.15,
        "random_token_ratio": 0.1,
        "unchanged_ratio": 0.1,
        "mlm_head_activation": "gelu",
        "initializer_range": 0.02,
        "mlm_head_dropout": 0.1,
        "layer_norm_eps": 1e-12,
    }

    # Update with user-provided config
    if mlm_config is not None:
        default_mlm_config.update(mlm_config)

    # Create MLM model
    mlm_model = MaskedLanguageModel(
        encoder=encoder,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
        **default_mlm_config,
    )

    # Default optimizer configuration
    default_optimizer_config = {
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
    }

    # Update with user-provided config
    if optimizer_config is not None:
        default_optimizer_config.update(optimizer_config)

    # Create optimizer
    optimizer = keras.optimizers.AdamW(**default_optimizer_config)

    # Compile the model
    mlm_model.compile(
        optimizer=optimizer,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    logger.info(
        f"Created and compiled MLM training model with "
        f"{mlm_model.encoder.count_params():,} encoder parameters"
    )

    return mlm_model

# ---------------------------------------------------------------------
