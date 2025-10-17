"""FNet Fine-tuning Script for Sentiment Analysis with Tiktoken.

A complete, production-quality training script for fine-tuning a pre-trained
FNet encoder on a downstream sentiment analysis task, featuring advanced
post-training analysis and using TiktokenPreprocessor for tokenization.

This script demonstrates the full fine-tuning workflow:
1. Loading a pre-trained FNet encoder.
2. Loading and preprocessing the IMDB reviews dataset.
3. Implementing a configurable two-stage fine-tuning process:
    a. Stage 1: Train only the classification head with the encoder frozen.
    b. Stage 2: Unfreeze the encoder and fine-tune the entire model.
4. Saving the fine-tuned model for inference.
5. Performing a comprehensive post-training analysis comparing the initial,
   best, and final models on weights, calibration, and training dynamics.
6. Evaluating the final model on sample texts.
"""

import os
import keras
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.analyzer import (
    AnalysisConfig,
    DataInput,
    ModelAnalyzer
)

from dl_techniques.models.fnet import FNet
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.layers.nlp_heads.task_types import NLPTaskType
from dl_techniques.layers.nlp_heads import NLPTaskConfig, create_nlp_head
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class FinetuneConfig:
    """Configuration for FNet Sentiment Analysis fine-tuning."""

    # Model and paths
    pretrained_encoder_path: str = (
        "results/fnet_pretrain/pretrained_fnet_encoder_best.keras"
    )
    save_dir: str = "results/fnet_sentiment_finetune"
    log_dir: str = "results/fnet_sentiment_finetune/logs"
    checkpoint_dir: str = "results/fnet_sentiment_finetune/checkpoints"
    analysis_dir: str = "results/fnet_sentiment_finetune/epoch_analysis"
    full_analysis_dir: str = (
        "results/fnet_sentiment_finetune/full_analysis"
    )

    # Task configuration
    num_classes: int = 2  # IMDB: 0 (negative), 1 (positive)

    # Tokenizer configuration (Tiktoken)
    encoding_name: str = "cl100k_base"
    cls_token_id: int = 100264
    sep_token_id: int = 100265
    pad_token_id: int = 100266
    mask_token_id: int = 100267  # For reference, though not used here

    # Data configuration
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = None
    max_seq_length: int = 128
    batch_size: int = 16

    # --- Two-Stage Fine-tuning Configuration ---
    run_two_stage_finetuning: bool = True
    # Stage 1: Head Training (Encoder Frozen)
    stage1_epochs: int = 5
    stage1_learning_rate: float = 1e-3  # Higher LR for the new head
    # Stage 2: Full Model Fine-tuning (Encoder Unfrozen)
    stage2_epochs: int = 10  # Total epochs will be stage1 + stage2
    stage2_learning_rate: float = 3e-5  # Lower LR for the entire model
    weight_decay: float = 0.01

    # In-Training Analysis
    run_epoch_analysis: bool = True
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 1

    # Post-Training Full Analysis
    run_post_training_analysis: bool = True
    analysis_n_samples: int = 1000  # Samples for calibration analysis


# ---------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------


def create_tokenizer(config: FinetuneConfig) -> TiktokenPreprocessor:
    """Create and configure Tiktoken preprocessor.

    :param config: Fine-tuning configuration.
    :type config: FinetuneConfig
    :return: Configured TiktokenPreprocessor instance.
    :rtype: TiktokenPreprocessor
    """
    logger.info("Initializing TiktokenPreprocessor...")
    preprocessor = TiktokenPreprocessor(
        encoding_name=config.encoding_name,
        max_length=config.max_seq_length,
        cls_token_id=config.cls_token_id,
        sep_token_id=config.sep_token_id,
        pad_token_id=config.pad_token_id,
        mask_token_id=config.mask_token_id,
        truncation=True,
        padding='max_length',
    )
    logger.info(
        f"TiktokenPreprocessor initialized: "
        f"vocab_size={preprocessor.vocab_size}, "
        f"encoding={config.encoding_name}"
    )
    logger.info(
        f"Special tokens: [CLS]={config.cls_token_id}, "
        f"[SEP]={config.sep_token_id}, "
        f"[PAD]={config.pad_token_id}"
    )
    return preprocessor


def load_dataset(
    config: FinetuneConfig,
    split: str = "train"
) -> tf.data.Dataset:
    """Load text dataset with labels from tensorflow-datasets.

    :param config: Fine-tuning configuration.
    :type config: FinetuneConfig
    :param split: Dataset split to load ('train', 'test').
    :type split: str
    :return: TensorFlow dataset.
    :rtype: tf.data.Dataset
    """
    logger.info(
        f"Loading {config.dataset_name} dataset ({split} split)..."
    )
    dataset, _ = tfds.load(
        config.dataset_name,
        split=split,
        as_supervised=True,
        shuffle_files=True,
        with_info=True,
    )
    if config.max_samples is not None and "train" in split:
        dataset = dataset.take(config.max_samples)
        logger.info(
            f"Limited training data to {config.max_samples} samples"
        )
    return dataset


def preprocess_dataset(
    dataset: tf.data.Dataset,
    preprocessor: TiktokenPreprocessor,
    config: FinetuneConfig
) -> tf.data.Dataset:
    """Preprocess text dataset for sentiment analysis fine-tuning.

    :param dataset: Raw text dataset with labels.
    :type dataset: tf.data.Dataset
    :param preprocessor: Tiktoken preprocessor.
    :type preprocessor: TiktokenPreprocessor
    :param config: Fine-tuning configuration.
    :type config: FinetuneConfig
    :return: Preprocessed dataset ready for training.
    :rtype: tf.data.Dataset
    """
    logger.info("Preprocessing dataset for fine-tuning...")

    def tokenize_function(
        text: tf.Tensor,
        label: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tf.Tensor]:
        """Tokenize text and return flat NumPy arrays.

        :param text: Input text tensor.
        :type text: tf.Tensor
        :param label: Label tensor.
        :type label: tf.Tensor
        :return: Tuple of (input_ids, attention_mask, token_type_ids, label).
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, tf.Tensor]
        """
        # Decode bytes to string if necessary
        if isinstance(text, bytes):
            text_str = text.decode('utf-8')
        elif hasattr(text, 'numpy'):
            text_np = text.numpy()
            if isinstance(text_np, bytes):
                text_str = text_np.decode('utf-8')
            else:
                text_str = str(text_np)
        else:
            text_str = str(text)

        # Tokenize using TiktokenPreprocessor
        encoded = preprocessor(text_str, return_tensors='np')

        return (
            encoded['input_ids'][0],
            encoded['attention_mask'][0],
            encoded['token_type_ids'][0],
            label
        )

    dataset = dataset.map(
        lambda text, label: tf.py_function(
            func=tokenize_function,
            inp=[text, label],
            Tout=[tf.int32, tf.int32, tf.int32, tf.int64]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    def structure_to_dict_and_label(
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        token_type_ids: tf.Tensor,
        label: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Restructure flat tensors into model input format.

        :param input_ids: Token IDs tensor.
        :type input_ids: tf.Tensor
        :param attention_mask: Attention mask tensor.
        :type attention_mask: tf.Tensor
        :param token_type_ids: Token type IDs tensor.
        :type token_type_ids: tf.Tensor
        :param label: Label tensor.
        :type label: tf.Tensor
        :return: Tuple of (inputs_dict, label_tensor).
        :rtype: Tuple[Dict[str, tf.Tensor], tf.Tensor]
        """
        inputs = {
            'input_ids': tf.ensure_shape(
                input_ids,
                [config.max_seq_length]
            ),
            'attention_mask': tf.ensure_shape(
                attention_mask,
                [config.max_seq_length]
            ),
            'token_type_ids': tf.ensure_shape(
                token_type_ids,
                [config.max_seq_length]
            ),
        }
        label_tensor = tf.cast(label, dtype=tf.int32)
        label_tensor.set_shape(())
        return inputs, label_tensor

    dataset = dataset.map(
        structure_to_dict_and_label,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.cache().shuffle(1000).batch(
        config.batch_size,
        drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)

    logger.info(
        f"Dataset preprocessed: batch_size={config.batch_size}"
    )
    return dataset


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_sentiment_model(
    config: FinetuneConfig
) -> Tuple[keras.Model, keras.Model]:
    """Create a sentiment analysis model.

    This function loads the pretrained FNet encoder, attaches a
    classification head, and returns both the complete model and a
    reference to the encoder.

    :param config: Fine-tuning configuration.
    :type config: FinetuneConfig
    :return: Tuple of (full model, encoder model).
    :rtype: Tuple[keras.Model, keras.Model]
    :raises IOError: If the pretrained encoder cannot be loaded.
    """
    logger.info("=" * 80)
    logger.info("Creating Sentiment Analysis Model")
    logger.info("=" * 80)

    try:
        logger.info(
            f"Loading pretrained encoder from: "
            f"{config.pretrained_encoder_path}"
        )
        fnet_encoder = keras.models.load_model(
            config.pretrained_encoder_path,
            custom_objects={"FNet": FNet}
        )
        logger.info("Pretrained FNet encoder loaded successfully.")
    except (IOError, OSError) as e:
        logger.error(
            f"Failed to load FNet encoder. Ensure the path is correct. "
            f"Error: {e}"
        )
        raise

    task_config = NLPTaskConfig(
        name="sentiment_analysis",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=config.num_classes,
        dropout_rate=0.1
    )
    head = create_nlp_head(
        task_config=task_config,
        input_dim=fnet_encoder.hidden_size
    )
    logger.info("Sequence classification head created.")

    inputs = {
        "input_ids": keras.Input(
            shape=(None,),
            dtype="int32",
            name="input_ids"
        ),
        "attention_mask": keras.Input(
            shape=(None,),
            dtype="int32",
            name="attention_mask"
        ),
        "token_type_ids": keras.Input(
            shape=(None,),
            dtype="int32",
            name="token_type_ids"
        ),
    }
    encoder_outputs = fnet_encoder(inputs)
    cls_token_output = encoder_outputs["last_hidden_state"][:, 0, :]
    task_outputs = head({"hidden_states": cls_token_output})

    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs['logits'],
        name="fnet_sentiment_analyzer"
    )
    logger.info(
        f"Model created successfully. "
        f"Total parameters: {model.count_params():,}"
    )

    return model, fnet_encoder


# ---------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------


def compile_model(
    model: keras.Model,
    config: FinetuneConfig,
    learning_rate: float
) -> None:
    """Compile the model with a specific learning rate.

    :param model: The Keras model to compile.
    :type model: keras.Model
    :param config: The fine-tuning configuration.
    :type config: FinetuneConfig
    :param learning_rate: The learning rate for the optimizer.
    :type learning_rate: float
    """
    logger.info(f"Compiling model with learning_rate={learning_rate}...")
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")]
    )


def create_callbacks(
    config: FinetuneConfig
) -> List[keras.callbacks.Callback]:
    """Create training callbacks.

    :param config: The fine-tuning configuration.
    :type config: FinetuneConfig
    :return: A list of Keras callbacks.
    :rtype: List[keras.callbacks.Callback]
    """
    logger.info("Creating training callbacks...")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(
        config.checkpoint_dir,
        "best_sentiment_model.keras"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=config.log_dir),
        keras.callbacks.CSVLogger(
            os.path.join(config.save_dir, "finetuning_log.csv")
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    if config.run_epoch_analysis:
        callbacks.append(EpochAnalyzerCallback(
            output_dir=config.analysis_dir,
            start_epoch=config.analysis_start_epoch,
            epoch_frequency=config.analysis_epoch_frequency,
            model_name="FNet-Sentiment-Finetuned-Tiktoken"
        ))
    logger.info(f"Created {len(callbacks)} callbacks.")
    return callbacks


def _merge_histories(
    base_history: keras.callbacks.History,
    new_history: keras.callbacks.History
) -> keras.callbacks.History:
    """Merge a new Keras History object into a base one.

    :param base_history: The base history to merge into.
    :type base_history: keras.callbacks.History
    :param new_history: The new history to merge.
    :type new_history: keras.callbacks.History
    :return: The merged history object.
    :rtype: keras.callbacks.History
    """
    for key, value in new_history.history.items():
        base_history.history.setdefault(key, []).extend(value)
    base_history.epoch.extend(new_history.epoch)
    return base_history


def finetune_sentiment_model(
    config: FinetuneConfig
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Main function for fine-tuning the sentiment analysis model.

    :param config: The fine-tuning configuration.
    :type config: FinetuneConfig
    :return: Tuple of (trained model, training history).
    :rtype: Tuple[keras.Model, keras.callbacks.History]
    """
    logger.info("=" * 80)
    logger.info("FNet Sentiment Analysis Fine-tuning with Tiktoken")
    logger.info("=" * 80)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)
    os.makedirs(config.save_dir, exist_ok=True)

    preprocessor = create_tokenizer(config)
    train_dataset = preprocess_dataset(
        load_dataset(config, "train"),
        preprocessor,
        config
    )
    val_dataset = preprocess_dataset(
        load_dataset(config, "test"),
        preprocessor,
        config
    )
    model, fnet_encoder = create_sentiment_model(config)
    callbacks = create_callbacks(config)

    if config.run_two_stage_finetuning:
        logger.info("=" * 80)
        logger.info(
            f"Starting Stage 1: Head Training for "
            f"{config.stage1_epochs} epochs"
        )
        fnet_encoder.trainable = False
        compile_model(model, config, config.stage1_learning_rate)
        history1 = model.fit(
            train_dataset,
            epochs=config.stage1_epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            verbose=1
        )
        final_history = history1

        logger.info("=" * 80)
        logger.info(
            f"Starting Stage 2: Full Fine-tuning for "
            f"{config.stage2_epochs} epochs"
        )
        fnet_encoder.trainable = True
        compile_model(model, config, config.stage2_learning_rate)
        initial_epoch = final_history.epoch[-1] + 1
        history2 = model.fit(
            train_dataset,
            epochs=initial_epoch + config.stage2_epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            validation_data=val_dataset,
            verbose=1
        )
        final_history = _merge_histories(final_history, history2)
    else:
        logger.info("=" * 80)
        logger.info(
            f"Starting Single-Stage Fine-tuning for "
            f"{config.stage2_epochs} epochs"
        )
        compile_model(model, config, config.stage2_learning_rate)
        final_history = model.fit(
            train_dataset,
            epochs=config.stage2_epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
            verbose=1
        )

    logger.info("=" * 80)
    logger.info("Fine-tuning completed!")
    final_model_path = os.path.join(
        config.save_dir,
        "fnet_sentiment_final_best.keras"
    )
    model.save(final_model_path)
    logger.info(
        f"Saved final model (with best weights) to {final_model_path}"
    )

    history_path = os.path.join(config.save_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(final_history.history, f)
    logger.info(f"Saved training history to {history_path}")

    return model, final_history


# ---------------------------------------------------------------------
# Analysis and Evaluation
# ---------------------------------------------------------------------


def prepare_data_for_analyzer(
    val_dataset: tf.data.Dataset,
    num_samples: int
) -> DataInput:
    """Prepare a subsample of the validation set for the ModelAnalyzer.

    This function extracts a specified number of samples and formats them
    into a dictionary of NumPy arrays for `x_data` and a single NumPy
    array for `y_data`.

    :param val_dataset: The validation `tf.data.Dataset`.
    :type val_dataset: tf.data.Dataset
    :param num_samples: The number of samples to extract.
    :type num_samples: int
    :return: A `DataInput` object ready for the analyzer.
    :rtype: DataInput
    """
    logger.info(
        f"Preparing {num_samples} samples for post-training analysis..."
    )
    val_subset = val_dataset.unbatch().take(num_samples)

    x_batches = []
    y_list = []
    for x, y in val_subset:
        x_batches.append(x)
        y_list.append(y.numpy())

    # Collate the list of dictionaries into a single dictionary
    # of NumPy arrays. This is the format Keras and the ModelAnalyzer
    # expect for multi-input models.
    x_data_dict = {
        key: np.array([d[key].numpy() for d in x_batches])
        for key in x_batches[0].keys()
    }

    return DataInput(x_data=x_data_dict, y_data=np.array(y_list))


def post_training_analysis(config: FinetuneConfig) -> None:
    """Run comprehensive analysis comparing initial, best, and final models.

    :param config: The fine-tuning configuration.
    :type config: FinetuneConfig
    """
    logger.info("=" * 80)
    logger.info("Running Post-Training Comprehensive Analysis")
    logger.info("=" * 80)
    os.makedirs(config.full_analysis_dir, exist_ok=True)

    # 1. Load Models for Comparison
    initial_model, _ = create_sentiment_model(config)
    best_model_path = os.path.join(
        config.checkpoint_dir,
        "best_sentiment_model.keras"
    )
    final_model_path = os.path.join(
        config.save_dir,
        "fnet_sentiment_final_best.keras"
    )

    models_to_analyze = {
        "Initial_Model": initial_model,
        "Best_Model(ValAcc)": keras.models.load_model(
            best_model_path, custom_objects={"FNet": FNet}
        ),
        "Final_Model": keras.models.load_model(
            final_model_path, custom_objects={"FNet": FNet}
        ),
    }
    logger.info(f"Loaded {len(models_to_analyze)} models for analysis.")

    # 2. Load Training History
    history_path = os.path.join(config.save_dir, "training_history.pkl")
    with open(history_path, 'rb') as f:
        history_dict = pickle.load(f)
    training_histories = {
        name: history_dict for name in models_to_analyze.keys()
    }

    # 3. Prepare Data
    preprocessor = create_tokenizer(config)
    val_dataset = preprocess_dataset(
        load_dataset(config, "test"),
        preprocessor,
        config
    )
    analysis_data = prepare_data_for_analyzer(
        val_dataset,
        config.analysis_n_samples
    )

    # 4. Configure and Run Analyzer
    analysis_config = AnalysisConfig(
        analyze_weights=True,
        analyze_spectral=True,
        analyze_calibration=True,
        analyze_training_dynamics=True,
        analyze_information_flow=False,  # Skipped due to multi-input
        verbose=True
    )

    analyzer = ModelAnalyzer(
        models=models_to_analyze,
        training_history=training_histories,
        config=analysis_config,
        output_dir=config.full_analysis_dir
    )

    logger.info("Starting comprehensive analysis...")
    analyzer.analyze(data=analysis_data)
    logger.info(
        f"Comprehensive analysis complete. "
        f"Results saved to: {config.full_analysis_dir}"
    )


def evaluate_model(
    model: keras.Model,
    preprocessor: TiktokenPreprocessor,
    config: FinetuneConfig
) -> None:
    """Evaluate the fine-tuned model on sample sentences.

    :param model: The fine-tuned Keras model.
    :type model: keras.Model
    :param preprocessor: The Tiktoken preprocessor.
    :type preprocessor: TiktokenPreprocessor
    :param config: The fine-tuning configuration.
    :type config: FinetuneConfig
    """
    logger.info("=" * 80)
    logger.info("Final Model Evaluation on Sample Texts")
    logger.info("=" * 80)
    test_texts = [
        "This movie was an absolute masterpiece. "
        "The acting and story were perfect.",
        "I was really disappointed with this film. "
        "It was boring and made no sense.",
        "The movie was okay, not great but not terrible either.",
        "A truly inspiring and heartwarming story that will stay with you.",
    ]
    labels = ["Negative", "Positive"]
    for text in test_texts:
        # Use TiktokenPreprocessor
        inputs = preprocessor.encode(text, return_tensors='tf')

        # Make prediction
        logits = model.predict(inputs, verbose=0)
        probabilities = tf.nn.softmax(logits, axis=-1)[0]
        pred_id = tf.argmax(probabilities).numpy()

        logger.info(f"Text: '{text}'")
        logger.info(
            f" -> Predicted: '{labels[pred_id]}' "
            f"(Confidence: {probabilities[pred_id]:.4f})"
        )


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for FNet fine-tuning script."""
    config = FinetuneConfig()
    logger.info("Fine-tuning Configuration:")
    logger.info(f"  - Tokenizer: TiktokenPreprocessor")
    logger.info(f"  - Encoding: {config.encoding_name}")
    for key, value in vars(config).items():
        if not key.startswith("__") and key not in [
            'encoding_name', 'cls_token_id', 'sep_token_id',
            'pad_token_id', 'mask_token_id'
        ]:
            logger.info(f"  - {key}: {value}")

    model, _ = finetune_sentiment_model(config)

    if config.run_post_training_analysis:
        post_training_analysis(config)

    preprocessor = create_tokenizer(config)
    evaluate_model(model, preprocessor, config)

    logger.info("=" * 80)
    logger.info("Script complete! Model is ready for inference.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()