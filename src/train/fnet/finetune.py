"""FNet Fine-tuning Script for Sentiment Analysis.

Fine-tunes a pre-trained FNet encoder on IMDB sentiment analysis with
optional two-stage training (frozen encoder -> full fine-tuning) and
comprehensive post-training analysis.
"""

import argparse
import os
import pickle

import keras
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple

from train.common import setup_gpu
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    preprocess_classification_dataset,
    create_nlp_callbacks,
)

from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer
from dl_techniques.models.fnet import FNet
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.layers.nlp_heads.task_types import NLPTaskType
from dl_techniques.layers.nlp_heads import NLPTaskConfig, create_nlp_head

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class FinetuneConfig:
    """Configuration for FNet sentiment analysis fine-tuning."""

    # Model and paths
    pretrained_encoder_path: str = "results/fnet_pretrain/pretrained_fnet_encoder_best.keras"
    save_dir: str = "results/fnet_sentiment_finetune"
    full_analysis_dir: str = "results/fnet_sentiment_finetune/full_analysis"

    # Task
    num_classes: int = 2

    # Tokenizer (Tiktoken cl100k_base)
    encoding_name: str = "cl100k_base"
    cls_token_id: int = 100264
    sep_token_id: int = 100265
    pad_token_id: int = 100266
    mask_token_id: int = 100267

    # Data
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = None
    max_seq_length: int = 128
    batch_size: int = 16

    # Two-stage fine-tuning
    run_two_stage_finetuning: bool = True
    stage1_epochs: int = 5
    stage1_learning_rate: float = 1e-3
    stage2_epochs: int = 10
    stage2_learning_rate: float = 3e-5
    weight_decay: float = 0.01

    # Analysis
    run_epoch_analysis: bool = True
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 1
    run_post_training_analysis: bool = True
    analysis_n_samples: int = 1000


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_sentiment_model(config: FinetuneConfig) -> Tuple[keras.Model, keras.Model]:
    """Load pretrained FNet encoder and attach a classification head."""
    logger.info("Creating Sentiment Analysis Model")
    try:
        fnet_encoder = keras.models.load_model(
            config.pretrained_encoder_path, custom_objects={"FNet": FNet},
        )
        logger.info("Pretrained FNet encoder loaded.")
    except (IOError, OSError) as e:
        logger.error(f"Failed to load FNet encoder: {e}")
        raise

    head = create_nlp_head(
        task_config=NLPTaskConfig(
            name="sentiment_analysis",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=config.num_classes, dropout_rate=0.1,
        ),
        input_dim=fnet_encoder.hidden_size,
    )

    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
    }
    encoder_out = fnet_encoder(inputs)
    cls_output = encoder_out["last_hidden_state"][:, 0, :]
    logits = head({"hidden_states": cls_output})['logits']

    model = keras.Model(inputs=inputs, outputs=logits, name="fnet_sentiment_analyzer")
    logger.info(f"Model created: {model.count_params():,} parameters")
    return model, fnet_encoder


# ---------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------


def compile_model(model: keras.Model, config: FinetuneConfig, learning_rate: float):
    """Compile model with AdamW for classification."""
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )
    logger.info(f"Compiled with lr={learning_rate}")


def _merge_histories(base, new):
    """Merge two Keras History objects."""
    for key, value in new.history.items():
        base.history.setdefault(key, []).extend(value)
    base.epoch.extend(new.epoch)
    return base


def finetune_sentiment_model(
    config: FinetuneConfig,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Run sentiment analysis fine-tuning with optional two-stage training."""
    logger.info("=" * 60)
    logger.info("FNet Sentiment Analysis Fine-tuning")
    logger.info("=" * 60)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)
    os.makedirs(config.save_dir, exist_ok=True)

    preprocessor = create_tokenizer(
        config.encoding_name, config.max_seq_length,
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    )
    train_dataset = preprocess_classification_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples, as_supervised=True),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val_dataset = preprocess_classification_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples, as_supervised=True),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    model, fnet_encoder = create_sentiment_model(config)
    callbacks, results_dir = create_nlp_callbacks(
        model_name="FNet-Sentiment-Finetuned-Tiktoken",
        results_dir_prefix="fnet_finetune",
        monitor='val_accuracy',
        patience=3,
        include_analyzer=config.run_epoch_analysis,
        analyzer_epoch_frequency=config.analysis_epoch_frequency,
        analyzer_start_epoch=config.analysis_start_epoch,
    )

    if config.run_two_stage_finetuning:
        # Stage 1: train head only
        logger.info(f"Stage 1: Head Training for {config.stage1_epochs} epochs")
        fnet_encoder.trainable = False
        compile_model(model, config, config.stage1_learning_rate)
        history1 = model.fit(
            train_dataset, epochs=config.stage1_epochs,
            callbacks=callbacks, validation_data=val_dataset, verbose=1,
        )
        final_history = history1

        # Stage 2: full fine-tuning
        logger.info(f"Stage 2: Full Fine-tuning for {config.stage2_epochs} epochs")
        fnet_encoder.trainable = True
        compile_model(model, config, config.stage2_learning_rate)
        initial_epoch = final_history.epoch[-1] + 1
        history2 = model.fit(
            train_dataset, epochs=initial_epoch + config.stage2_epochs,
            initial_epoch=initial_epoch, callbacks=callbacks,
            validation_data=val_dataset, verbose=1,
        )
        final_history = _merge_histories(final_history, history2)
    else:
        logger.info(f"Single-Stage Fine-tuning for {config.stage2_epochs} epochs")
        compile_model(model, config, config.stage2_learning_rate)
        final_history = model.fit(
            train_dataset, epochs=config.stage2_epochs,
            callbacks=callbacks, validation_data=val_dataset, verbose=1,
        )

    logger.info("Fine-tuning completed!")
    final_path = os.path.join(config.save_dir, "fnet_sentiment_final_best.keras")
    model.save(final_path)
    logger.info(f"Saved final model to {final_path}")

    history_path = os.path.join(config.save_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(final_history.history, f)

    return model, final_history


# ---------------------------------------------------------------------
# Analysis and Evaluation
# ---------------------------------------------------------------------


def prepare_data_for_analyzer(val_dataset: tf.data.Dataset, num_samples: int) -> DataInput:
    """Extract samples from validation dataset for ModelAnalyzer."""
    logger.info(f"Preparing {num_samples} samples for analysis...")
    val_subset = val_dataset.unbatch().take(num_samples)
    x_batches, y_list = [], []
    for x, y in val_subset:
        x_batches.append(x)
        y_list.append(y.numpy())
    x_data = {key: np.array([d[key].numpy() for d in x_batches]) for key in x_batches[0].keys()}
    return DataInput(x_data=x_data, y_data=np.array(y_list))


def post_training_analysis(config: FinetuneConfig) -> None:
    """Run comprehensive post-training analysis comparing model snapshots."""
    logger.info("Running Post-Training Analysis")
    os.makedirs(config.full_analysis_dir, exist_ok=True)

    initial_model, _ = create_sentiment_model(config)
    best_path = os.path.join(config.save_dir, "best_sentiment_model.keras")
    final_path = os.path.join(config.save_dir, "fnet_sentiment_final_best.keras")

    models_to_analyze = {
        "Initial_Model": initial_model,
        "Best_Model(ValAcc)": keras.models.load_model(best_path, custom_objects={"FNet": FNet}),
        "Final_Model": keras.models.load_model(final_path, custom_objects={"FNet": FNet}),
    }

    history_path = os.path.join(config.save_dir, "training_history.pkl")
    with open(history_path, 'rb') as f:
        history_dict = pickle.load(f)
    training_histories = {name: history_dict for name in models_to_analyze}

    preprocessor = create_tokenizer(
        config.encoding_name, config.max_seq_length,
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    )
    val_dataset = preprocess_classification_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples, as_supervised=True),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    analysis_data = prepare_data_for_analyzer(val_dataset, config.analysis_n_samples)

    analyzer = ModelAnalyzer(
        models=models_to_analyze,
        training_history=training_histories,
        config=AnalysisConfig(
            analyze_weights=True, analyze_spectral=True,
            analyze_calibration=True, analyze_training_dynamics=True,
            analyze_information_flow=False, verbose=True,
        ),
        output_dir=config.full_analysis_dir,
    )
    analyzer.analyze(data=analysis_data)
    logger.info(f"Analysis complete. Results: {config.full_analysis_dir}")


def evaluate_model(model: keras.Model, preprocessor: TiktokenPreprocessor, config: FinetuneConfig):
    """Evaluate on sample sentences."""
    logger.info("Evaluating on sample texts")
    test_texts = [
        "This movie was an absolute masterpiece. The acting and story were perfect.",
        "I was really disappointed with this film. It was boring and made no sense.",
        "The movie was okay, not great but not terrible either.",
        "A truly inspiring and heartwarming story that will stay with you.",
    ]
    labels = ["Negative", "Positive"]
    for text in test_texts:
        inputs = preprocessor.encode(text, return_tensors='tf')
        logits = model.predict(inputs, verbose=0)
        probs = tf.nn.softmax(logits, axis=-1)[0]
        pred_id = tf.argmax(probs).numpy()
        logger.info(f"'{text}' -> {labels[pred_id]} ({probs[pred_id]:.4f})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for FNet fine-tuning."""
    parser = argparse.ArgumentParser(description="FNet Sentiment Fine-tuning")
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    parser.add_argument('--encoder-path', type=str, default=None, help='Pretrained encoder path')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples')
    parser.add_argument('--no-two-stage', action='store_true', help='Disable two-stage training')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip post-training analysis')
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = FinetuneConfig()
    if args.encoder_path:
        config.pretrained_encoder_path = args.encoder_path
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples
    config.run_two_stage_finetuning = not args.no_two_stage
    config.run_post_training_analysis = not args.skip_analysis

    model, _ = finetune_sentiment_model(config)

    if config.run_post_training_analysis:
        post_training_analysis(config)

    preprocessor = create_tokenizer(
        config.encoding_name, config.max_seq_length,
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    )
    evaluate_model(model, preprocessor, config)
    logger.info("Script complete! Model is ready for inference.")


if __name__ == "__main__":
    main()
