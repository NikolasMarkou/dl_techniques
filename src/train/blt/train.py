import os
import sys
import time
import keras
import random
import numpy as np
from typing import List, Tuple

from dl_techniques.models.blt import (
    create_blt_model,
)
from dl_techniques.utils.logger import logger
from dl_techniques.layers.blt import ByteTokenizer, EntropyModel

def create_sample_training_data(num_samples: int = 200) -> List[str]:
    """
    Create sample training data with diverse text patterns.

    Args:
        num_samples: Number of training examples to generate.

    Returns:
        List of training text samples.
    """
    # Base templates for different types of content
    templates = [
        # Technical content
        "Artificial intelligence systems use machine learning algorithms to process large datasets and extract meaningful patterns.",
        "Deep neural networks consist of multiple layers that transform input data through weighted connections and activation functions.",
        "Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way.",
        "Computer vision algorithms analyze visual data to identify objects, recognize faces, and understand scene context.",
        "Reinforcement learning agents learn optimal behaviors through trial and error interactions with their environment.",

        # Scientific content
        "Scientific research involves hypothesis formation, experimental design, data collection, and statistical analysis of results.",
        "The scientific method provides a systematic approach to understanding natural phenomena through observation and experimentation.",
        "Peer review ensures the quality and reliability of scientific publications through expert evaluation and feedback.",
        "Interdisciplinary collaboration combines expertise from multiple fields to address complex research questions.",
        "Open science practices promote transparency, reproducibility, and accessibility in scientific research.",

        # General knowledge
        "Climate change affects global weather patterns, sea levels, and ecosystem dynamics across the planet.",
        "Renewable energy sources like solar, wind, and hydroelectric power provide sustainable alternatives to fossil fuels.",
        "Biodiversity conservation protects endangered species and maintains healthy ecosystem functioning worldwide.",
        "Urban planning integrates transportation, housing, and infrastructure to create livable and sustainable cities.",
        "Cultural heritage preservation maintains historical sites, traditions, and knowledge for future generations.",

        # Technology topics
        "Cloud computing provides scalable computing resources and services over the internet infrastructure.",
        "Cybersecurity protects digital systems, networks, and data from malicious attacks and unauthorized access.",
        "Blockchain technology enables secure, decentralized transactions without requiring trusted intermediaries.",
        "Internet of Things devices connect everyday objects to networks, enabling smart homes and cities.",
        "Quantum computing leverages quantum mechanical properties to solve certain problems exponentially faster.",

        # Conversational patterns
        "Hello, how are you doing today? I hope you're having a wonderful time.",
        "Thank you for your help with this project. I really appreciate your assistance.",
        "Could you please explain how this process works? I'd like to understand it better.",
        "That's a great question! Let me think about the best way to answer it.",
        "I'm excited to learn more about this topic. It seems really interesting and important.",
    ]

    # Sentence connectors and transitions
    connectors = [
        "Furthermore,", "Moreover,", "Additionally,", "However,", "Nevertheless,",
        "In contrast,", "On the other hand,", "For example,", "As a result,",
        "Therefore,", "Consequently,", "In conclusion,", "To summarize,"
    ]

    # Additional sentence fragments
    fragments = [
        "recent studies have shown that", "researchers have discovered",
        "experts believe that", "according to the latest findings",
        "preliminary results suggest", "the evidence indicates",
        "it is widely accepted that", "scientists have confirmed",
        "data analysis reveals", "experimental results demonstrate"
    ]

    training_samples = []

    for _ in range(num_samples):
        # Choose random combination strategy
        strategy = random.choice(['single', 'combined', 'extended'])

        if strategy == 'single':
            # Use single template
            sample = random.choice(templates)

        elif strategy == 'combined':
            # Combine 2-3 templates with connectors
            num_templates = random.randint(2, 3)
            selected_templates = random.sample(templates, num_templates)

            combined_parts = []
            for i, template in enumerate(selected_templates):
                if i > 0:
                    connector = random.choice(connectors)
                    combined_parts.append(f"{connector} {template.lower()}")
                else:
                    combined_parts.append(template)

            sample = " ".join(combined_parts)

        else:  # extended
            # Extend template with fragments
            base_template = random.choice(templates)
            fragment = random.choice(fragments)
            extension = random.choice(templates[:10])  # Use subset for extension

            sample = f"{base_template} {fragment.title()} {extension.lower()}"

        training_samples.append(sample)

    return training_samples


def prepare_training_data(
        texts: List[str],
        tokenizer: ByteTokenizer,
        max_length: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from text samples.

    Args:
        texts: List of training text samples.
        tokenizer: Byte tokenizer for conversion.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (input_tokens, target_tokens) arrays.
    """
    logger.info(f"Preparing training data from {len(texts)} samples")

    tokenized_samples = []

    for text in texts:
        # Convert text to byte tokens
        tokens = tokenizer.text_to_bytes(text, add_bos=True, add_eos=True)

        # Skip samples that are too short or too long
        if len(tokens) < 10 or len(tokens) > max_length:
            continue

        tokenized_samples.append(tokens)

    logger.info(f"Kept {len(tokenized_samples)} samples after filtering")

    # Pad sequences to consistent length
    padded_sequences = []
    for tokens in tokenized_samples:
        if len(tokens) < max_length:
            # Pad with zeros (pad token)
            padded = tokens + [0] * (max_length - len(tokens))
        else:
            # Truncate if too long
            padded = tokens[:max_length]

        padded_sequences.append(padded)

    # Convert to numpy arrays
    input_tokens = np.array(padded_sequences, dtype=np.int32)

    # Create targets by shifting input left by one position
    target_tokens = np.roll(input_tokens, -1, axis=1)
    target_tokens[:, -1] = 0  # Set last position to pad token

    logger.info(f"Training data shape: {input_tokens.shape}")
    logger.info(f"Token range: [{np.min(input_tokens)}, {np.max(input_tokens)}]")

    return input_tokens, target_tokens


def create_and_train_entropy_model(
        train_x: np.ndarray,
        train_y: np.ndarray,
        vocab_size: int = 260,
        epochs: int = 5
) -> EntropyModel:
    """
    Create and train a simple entropy model for dynamic patching.

    Args:
        train_x: Training input tokens.
        train_y: Training target tokens.
        vocab_size: Size of vocabulary.
        epochs: Number of training epochs.

    Returns:
        Trained entropy model.
    """
    logger.info("Creating and training entropy model...")

    # Create small entropy model
    entropy_model = EntropyModel(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=train_x.shape[1],
        dropout_rate=0.1
    )

    # Compile model
    entropy_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train entropy model
    logger.info("Training entropy model...")
    entropy_history = entropy_model.fit(
        train_x, train_y,
        epochs=epochs,
        batch_size=8,
        validation_split=0.1,
        verbose=1
    )

    logger.info(f"Entropy model training completed. Final loss: {entropy_history.history['loss'][-1]:.4f}")

    return entropy_model


def main():
    """Main training script."""

    logger.info("Starting BLT Training Script")
    logger.info("=" * 50)

    # Configuration parameters
    VOCAB_SIZE = 260
    LOCAL_DIM = 256
    GLOBAL_DIM = 384
    NUM_LOCAL_LAYERS = 3
    NUM_GLOBAL_LAYERS = 4
    NUM_HEADS_LOCAL = 4
    NUM_HEADS_GLOBAL = 6
    MAX_SEQUENCE_LENGTH = 256
    MAX_PATCHES = 64
    ENTROPY_THRESHOLD = 1.3
    CROSS_ATTENTION_QUERIES = 4
    DROPOUT_RATE = 0.1
    PATCH_POOLING_METHOD = 'attention'

    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    # Create output directory
    output_dir = "blt_training_output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Create training data
        logger.info("Step 1: Creating sample training data...")
        training_texts = create_sample_training_data(num_samples=150)

        # Show some examples
        logger.info("Sample training texts:")
        for i, text in enumerate(training_texts[:3]):
            logger.info(f"  {i + 1}: {text[:100]}...")

        # Step 2: Prepare tokenized training data
        logger.info("Step 2: Tokenizing training data...")
        tokenizer = ByteTokenizer(vocab_size=VOCAB_SIZE)

        train_x, train_y = prepare_training_data(
            training_texts,
            tokenizer,
            max_length=MAX_SEQUENCE_LENGTH
        )

        # Step 3: Create and train entropy model
        logger.info("Step 3: Training entropy model...")
        entropy_model = create_and_train_entropy_model(
            train_x, train_y,
            vocab_size=VOCAB_SIZE,
            epochs=3  # Quick training for entropy model
        )

        # Step 4: Create BLT model
        logger.info("Step 4: Creating BLT model...")

        # Create BLT model with trained entropy model
        blt_model = create_blt_model(
            vocab_size=VOCAB_SIZE,
            local_dim=LOCAL_DIM,
            global_dim=GLOBAL_DIM,
            num_local_layers=NUM_LOCAL_LAYERS,
            num_global_layers=NUM_GLOBAL_LAYERS,
            num_heads_local=NUM_HEADS_LOCAL,
            num_heads_global=NUM_HEADS_GLOBAL,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            max_patches=MAX_PATCHES,
            entropy_threshold=ENTROPY_THRESHOLD,
            cross_attention_queries=CROSS_ATTENTION_QUERIES,
            dropout_rate=DROPOUT_RATE,
            patch_pooling_method=PATCH_POOLING_METHOD,
            entropy_model=entropy_model,
            compile_model=True,
            learning_rate=LEARNING_RATE
        )

        logger.info(f"BLT model created with {blt_model.count_params():,} parameters")

        # Step 5: Train BLT model
        logger.info("Step 5: Training BLT model...")

        # Create train/validation split
        split_idx = int(0.9 * len(train_x))
        train_x_split, val_x = train_x[:split_idx], train_x[split_idx:]
        train_y_split, val_y = train_y[:split_idx], train_y[split_idx:]

        logger.info(f"Training samples: {len(train_x_split)}")
        logger.info(f"Validation samples: {len(val_x)}")

        # Training callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, "blt_model_best.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=1,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                verbose=1,
                min_lr=1e-6
            )
        ]

        # Train the model
        start_time = time.time()

        history = blt_model.fit(
            train_x_split, train_y_split,
            validation_data=(val_x, val_y),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Step 6: Save final model
        logger.info("Step 6: Saving trained model...")

        final_model_path = os.path.join(output_dir, "blt_model_final.keras")
        blt_model.save(final_model_path)
        logger.info(f"Model saved to: {final_model_path}")

        # Step 7: Evaluate and demonstrate text generation
        logger.info("Step 7: Testing text generation...")

        test_prompts = [
            "Artificial intelligence",
            "The future of technology",
            "Scientific research shows",
            "Machine learning algorithms"
        ]

        logger.info("Generated text samples:")
        logger.info("-" * 40)

        for prompt in test_prompts:
            try:
                generated = blt_model.generate(
                    prompt=prompt,
                    max_new_tokens=40,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True
                )

                logger.info(f"Prompt: '{prompt}'")
                logger.info(f"Generated: '{generated}'")
                logger.info("-" * 40)

            except Exception as e:
                logger.warning(f"Generation failed for prompt '{prompt}': {e}")

        # Step 8: Display training metrics
        logger.info("Step 8: Training summary...")

        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        best_val_loss = min(history.history['val_loss'])

        logger.info(f"Final training loss: {final_train_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        # Save training history
        history_path = os.path.join(output_dir, "training_history.npz")
        np.savez(history_path, **history.history)
        logger.info(f"Training history saved to: {history_path}")

        logger.info("=" * 50)
        logger.info("BLT Training Script Completed Successfully!")
        logger.info(f"Model files saved in: {output_dir}/")

        return blt_model, history

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def quick_test():
    """Quick test function for debugging."""

    logger.info("Running quick test...")

    # Create minimal data
    tokenizer = ByteTokenizer()
    sample_texts = [
        "Hello world, this is a test.",
        "Machine learning is fascinating.",
        "Natural language processing works well."
    ]

    train_x, train_y = prepare_training_data(sample_texts, tokenizer, max_length=64)

    # Create small model
    model = create_blt_model(
        vocab_size=260,
        local_dim=128,
        global_dim=192,
        num_local_layers=2,
        num_global_layers=2,
        num_heads_local=4,
        num_heads_global=4,
        max_sequence_length=64,
        max_patches=16,
        entropy_threshold=1.5,
        cross_attention_queries=2,
        dropout_rate=0.1,
        patch_pooling_method='attention',
        compile_model=True
    )

    logger.info(f"Test model has {model.count_params()} parameters")

    # Quick training
    model.fit(train_x, train_y, epochs=2, batch_size=1, verbose=1)

    # Test generation
    generated = model.generate("Hello", max_new_tokens=10, temperature=1.0)
    logger.info(f"Generated: '{generated}'")

    logger.info("Quick test completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BLT Model")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick test instead of full training")
    parser.add_argument("--output-dir", type=str, default="blt_training_output",
                        help="Directory to save model outputs")

    args = parser.parse_args()

    try:
        if args.quick_test:
            quick_test()
        else:
            main()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)