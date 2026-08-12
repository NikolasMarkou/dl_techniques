"""
Sentiment Analysis Inference Tool with Tiktoken.

A command-line tool to load a fine-tuned BERT model and perform sentiment
analysis on user-provided sentences in real-time using TiktokenPreprocessor.

This script loads the model saved by the fine-tuning script and provides an
interactive terminal interface for inference.

Which checkpoint the default points at
--------------------------------------
``--model_path`` defaults to the FINAL model ``train.bert.finetune`` writes at
the end of a run: ``<FinetuneConfig.save_dir>/<sentiment_final_model_filename>``.
Both halves of that path are imported from the fine-tuning script itself rather
than re-spelled here, so the default cannot drift away from what a run writes.

The best-VALIDATION snapshot is a different file: ``ModelCheckpoint`` writes it
to ``<results_dir>/best_model.keras``, where ``results_dir`` is a TIMESTAMPED
directory this script cannot know. Pass it explicitly with ``--model_path`` if
you want the best-val checkpoint instead of the final one.

Usage
-----
    # Run with default model path
    python deploy.py

    # Specify a different model file
    python deploy.py --model_path /path/to/your/model.keras

The tool will load the model and tokenizer, then prompt for sentence input.
Type 'quit' or 'exit' to terminate the program.
"""

import os
import keras
import argparse
import tensorflow as tf
from typing import Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.bert.finetune import FinetuneConfig
from train.common.nlp import sentiment_final_model_filename

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor

# ---------------------------------------------------------------------

# DECISION plan-2026-08-12T123743-e798a9e1/D-021
# Derived from the fine-tuning script's OWN save site, never re-typed. Before
# this, the default was the literal
# `results/bert_sentiment_finetune/checkpoints/best_sentiment_model.keras` --
# a third distinct spelling of a file NOTHING in `src/` has ever written (wrong
# subdirectory AND wrong filename), so `python -m train.bert.deploy` with no
# arguments could only ever print "Model file not found".
# DO NOT re-hard-code this path. See decisions.md D-021.
DEFAULT_MODEL_PATH = os.path.join(
    FinetuneConfig.save_dir, sentiment_final_model_filename("bert")
)

# ---------------------------------------------------------------------

# Suppress TensorFlow logging noise for a cleaner interface
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the inference tool.

    :return: Parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Sentiment Analysis Tool with a fine-tuned "
            "BERT model and TiktokenPreprocessor."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the saved .keras model file. Defaults to the FINAL model "
            "train.bert.finetune writes; the best-val snapshot lives in that "
            "run's timestamped results dir as best_model.keras."
        )
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization (must match training)."
    )
    parser.add_argument(
        "--encoding_name",
        type=str,
        default="cl100k_base",
        help="Tiktoken encoding name (must match training)."
    )
    parser.add_argument(
        "--cls_token_id",
        type=int,
        default=100264,
        help="[CLS] token ID (must match training)."
    )
    parser.add_argument(
        "--sep_token_id",
        type=int,
        default=100265,
        help="[SEP] token ID (must match training)."
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=100266,
        help="[PAD] token ID (must match training)."
    )
    return parser.parse_args()


def load_model_and_preprocessor(
    model_path: str,
    max_length: int,
    encoding_name: str,
    cls_token_id: int,
    sep_token_id: int,
    pad_token_id: int
) -> Tuple[keras.Model, TiktokenPreprocessor]:
    """Load the Keras model and TiktokenPreprocessor.

    :param model_path: Path to the .keras model file.
    :type model_path: str
    :param max_length: Maximum sequence length for tokenization.
    :type max_length: int
    :param encoding_name: Tiktoken encoding name.
    :type encoding_name: str
    :param cls_token_id: [CLS] token ID.
    :type cls_token_id: int
    :param sep_token_id: [SEP] token ID.
    :type sep_token_id: int
    :param pad_token_id: [PAD] token ID.
    :type pad_token_id: int
    :return: Tuple containing the loaded model and preprocessor.
    :rtype: Tuple[keras.Model, TiktokenPreprocessor]
    :raises FileNotFoundError: If the model file does not exist.
    """
    logger.info("Loading resources...")

    # Load Preprocessor
    try:
        preprocessor = TiktokenPreprocessor(
            encoding_name=encoding_name,
            max_length=max_length,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            truncation=True,
            padding='max_length',
        )
        logger.info(
            f"✅ TiktokenPreprocessor loaded "
            f"(encoding={encoding_name}, vocab_size={preprocessor.vocab_size})."
        )
    except Exception as e:
        logger.info(f"❌ Error loading preprocessor: {e}")
        exit(1)

    # Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please ensure you have run the fine-tuning script to generate "
            "the model."
        )

    try:
        model = keras.models.load_model(model_path)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.info(f"❌ Error loading Keras model: {e}")
        logger.info(
            "   This can happen if custom objects were not registered or "
            "if the file is corrupt."
        )
        exit(1)

    logger.info("-" * 50)
    return model, preprocessor


def predict_sentiment(
    text: str,
    model: keras.Model,
    preprocessor: TiktokenPreprocessor
) -> Tuple[str, float]:
    """Preprocess text, run inference, and return the sentiment.

    :param text: The input sentence from the user.
    :type text: str
    :param model: The loaded Keras model.
    :type model: keras.Model
    :param preprocessor: The loaded TiktokenPreprocessor.
    :type preprocessor: TiktokenPreprocessor
    :return: Tuple of (predicted_label, confidence_score).
    :rtype: Tuple[str, float]
    """
    # Define class labels
    labels = ["Negative", "Positive"]

    # Preprocess the input text using TiktokenPreprocessor
    inputs = preprocessor.encode(text, return_tensors='tf')

    # Run prediction (inputs is already a dict of tensors)
    logits = model.predict(inputs, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1)[0]
    predicted_class_id = tf.argmax(probabilities).numpy()

    # Get the results
    predicted_label = labels[predicted_class_id]
    confidence = probabilities[predicted_class_id].numpy()

    return predicted_label, float(confidence)

# ---------------------------------------------------------------------

def main() -> None:
    """Main function to run the interactive sentiment analysis tool."""
    args = parse_arguments()

    try:
        model, preprocessor = load_model_and_preprocessor(
            model_path=args.model_path,
            max_length=args.max_length,
            encoding_name=args.encoding_name,
            cls_token_id=args.cls_token_id,
            sep_token_id=args.sep_token_id,
            pad_token_id=args.pad_token_id
        )
    except FileNotFoundError as e:
        logger.info(f"❌ ERROR: {e}")
        return

    logger.info("🚀 Sentiment Analysis Tool is ready.")
    logger.info("   Enter a sentence for analysis.")
    logger.info("   Type 'quit' or 'exit' to end the session.")
    logger.info("-" * 50)

    # First prediction is often slower due to graph compilation
    logger.info("Warming up the model...")
    predict_sentiment(
        "This is a warm-up sentence.",
        model,
        preprocessor
    )
    logger.info("Model is ready for real-time predictions!")

    while True:
        try:
            user_input = input("\nEnter a sentence > ")

            # Check for exit condition
            if user_input.lower().strip() in ["quit", "exit", "q"]:
                break

            if not user_input.strip():
                continue

            # Get prediction
            label, confidence = predict_sentiment(
                user_input,
                model,
                preprocessor
            )

            # Display result
            emoji = "👍" if label == "Positive" else "👎"
            logger.info(f"  Sentiment: {label} {emoji}")
            logger.info(f"  Confidence: {confidence:.2%}")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            break
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")

    logger.info("👋 Goodbye!")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()