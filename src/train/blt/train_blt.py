"""
Byte Latent Transformer (BLT) Training Script

Training pipeline for BLT models with synthetic data generation,
entropy model pre-training, dynamic patching, and text generation evaluation.

Usage:
    python train_blt.py                    # Full training
    python train_blt.py --quick-test       # Quick functionality test
    python train_blt.py --config config.json  # Custom configuration
"""

import sys
import time
import json
import keras
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.byte_latent_transformer.model import create_blt_model
from dl_techniques.layers.blt_blocks import ByteTokenizer, EntropyModel


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class BLTTrainingConfig:
    """Configuration for BLT model training."""

    # Model Architecture
    vocab_size: int = 260
    local_dim: int = 256
    global_dim: int = 384
    num_local_layers: int = 3
    num_global_layers: int = 4
    num_heads_local: int = 4
    num_heads_global: int = 6
    max_sequence_length: int = 256
    max_patches: int = 64
    cross_attention_queries: int = 4
    dropout_rate: float = 0.1
    patch_pooling_method: str = 'attention'

    # Dynamic Patching
    entropy_threshold: float = 1.3
    entropy_model_hidden_dim: int = 128
    entropy_model_layers: int = 4
    entropy_model_heads: int = 4
    entropy_model_epochs: int = 3

    # Training Parameters
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    validation_split: float = 0.1
    early_stopping_patience: int = 3
    lr_reduction_factor: float = 0.5
    lr_reduction_patience: int = 2
    min_learning_rate: float = 1e-6

    # Data Generation
    num_training_samples: int = 150
    max_text_length: int = 256
    data_augmentation: bool = True

    # Generation Parameters
    generation_temperature: float = 0.8
    generation_top_p: float = 0.9
    generation_max_tokens: int = 40

    # Output and Logging
    output_dir: str = "blt_training_output"
    save_best_only: bool = True
    save_training_history: bool = True
    log_generation_examples: bool = True
    checkpoint_frequency: int = 5

    # System
    random_seed: int = 42
    mixed_precision: bool = False
    use_gpu: bool = True

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'BLTTrainingConfig':
        with open(filepath, 'r') as f:
            return cls(**json.load(f))

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'global_dim': self.global_dim,
            'num_local_layers': self.num_local_layers,
            'num_global_layers': self.num_global_layers,
            'num_heads_local': self.num_heads_local,
            'num_heads_global': self.num_heads_global,
            'max_sequence_length': self.max_sequence_length,
            'max_patches': self.max_patches,
            'entropy_threshold': self.entropy_threshold,
            'cross_attention_queries': self.cross_attention_queries,
            'dropout_rate': self.dropout_rate,
            'patch_pooling_method': self.patch_pooling_method
        }


# ---------------------------------------------------------------------
# Configuration Presets
# ---------------------------------------------------------------------

def get_quick_test_config() -> BLTTrainingConfig:
    return BLTTrainingConfig(
        local_dim=128, global_dim=192, num_local_layers=2, num_global_layers=2,
        num_heads_local=4, num_heads_global=4, max_sequence_length=64, max_patches=16,
        cross_attention_queries=2, num_training_samples=10, batch_size=1, epochs=2,
        entropy_model_epochs=1, generation_max_tokens=10, output_dir="blt_quick_test"
    )


def get_small_model_config() -> BLTTrainingConfig:
    return BLTTrainingConfig(
        local_dim=384, global_dim=512, num_local_layers=6, num_global_layers=8,
        num_heads_local=6, num_heads_global=8, max_sequence_length=512, max_patches=128,
        num_training_samples=500, batch_size=8, epochs=20, entropy_model_epochs=5,
        output_dir="blt_small_model"
    )


def get_large_model_config() -> BLTTrainingConfig:
    return BLTTrainingConfig(
        local_dim=768, global_dim=1024, num_local_layers=8, num_global_layers=16,
        num_heads_local=12, num_heads_global=16, max_sequence_length=2048, max_patches=512,
        num_training_samples=1000, batch_size=16, epochs=50, entropy_model_epochs=10,
        learning_rate=5e-5, mixed_precision=True, output_dir="blt_large_model"
    )


# ---------------------------------------------------------------------
# Training Data Generation
# ---------------------------------------------------------------------

class TrainingDataGenerator:
    """Generates diverse synthetic training data for BLT training."""

    def __init__(self, config: BLTTrainingConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

        self.templates = {
            'technical': [
                "Artificial intelligence systems use machine learning algorithms to process large datasets and extract meaningful patterns from complex information structures.",
                "Deep neural networks consist of multiple layers that transform input data through weighted connections and nonlinear activation functions.",
                "Natural language processing enables computers to understand, interpret, and generate human language in a contextually meaningful way.",
                "Computer vision_heads algorithms analyze visual data to identify objects, recognize faces, and understand complex scene relationships.",
                "Reinforcement learning agents learn optimal behaviors through trial and error interactions with dynamic environments.",
            ],
            'scientific': [
                "Scientific research involves hypothesis formation, experimental design, data collection, and rigorous statistical analysis of empirical results.",
                "The scientific method provides a systematic approach to understanding natural phenomena through careful observation and controlled experimentation.",
                "Peer review ensures the quality and reliability of scientific publications through expert evaluation and constructive feedback processes.",
                "Interdisciplinary collaboration combines expertise from multiple fields to address complex research questions that transcend traditional boundaries.",
                "Open science practices promote transparency, reproducibility, and accessibility in scientific research and knowledge dissemination.",
            ],
            'general': [
                "Climate change affects global weather patterns, sea levels, and ecosystem dynamics across diverse planetary environments.",
                "Renewable energy sources like solar, wind, and hydroelectric power provide sustainable alternatives to traditional fossil fuels.",
                "Biodiversity conservation protects endangered species and maintains healthy ecosystem functioning in natural habitats worldwide.",
                "Urban planning integrates transportation, housing, and infrastructure to create livable and sustainable metropolitan areas.",
                "Cultural heritage preservation maintains historical sites, traditions, and indigenous knowledge for future generations.",
            ],
            'conversational': [
                "Hello, how are you doing today? I hope you're having a wonderful and productive time.",
                "Thank you for your help with this challenging project. I really appreciate your dedicated assistance and expertise.",
                "Could you please explain how this complex process works? I'd like to understand it better for future reference.",
                "That's a great question! Let me think about the best way to answer it comprehensively and clearly.",
                "I'm excited to learn more about this fascinating topic. It seems really interesting and potentially important.",
            ]
        }

        self.connectors = [
            "Furthermore,", "Moreover,", "Additionally,", "However,", "Nevertheless,",
            "In contrast,", "On the other hand,", "For example,", "As a result,",
            "Therefore,", "Consequently,", "In conclusion,", "To summarize,",
            "Meanwhile,", "Subsequently,", "Specifically,", "Notably,"
        ]

        self.fragments = [
            "recent studies have shown that", "researchers have discovered",
            "experts believe that", "according to the latest findings",
            "preliminary results suggest", "the evidence indicates",
            "it is widely accepted that", "scientists have confirmed",
            "data analysis reveals", "experimental results demonstrate",
            "investigations have revealed", "analysis suggests that"
        ]

    def generate_samples(self) -> List[str]:
        samples = []
        for _ in range(self.config.num_training_samples):
            strategy = self.rng.choice(['single', 'combined', 'extended', 'question_answer'])
            if strategy == 'single':
                sample = self._generate_single_template()
            elif strategy == 'combined':
                sample = self._generate_combined_template()
            elif strategy == 'extended':
                sample = self._generate_extended_template()
            else:
                sample = self._generate_qa_pair()

            if self.config.data_augmentation:
                sample = self._apply_augmentation(sample)
            samples.append(sample)
        return samples

    def _generate_single_template(self) -> str:
        category = self.rng.choice(list(self.templates.keys()))
        return self.rng.choice(self.templates[category])

    def _generate_combined_template(self) -> str:
        num_templates = self.rng.randint(2, 3)
        categories = self.rng.choices(list(self.templates.keys()), k=num_templates)
        parts = []
        for i, category in enumerate(categories):
            template = self.rng.choice(self.templates[category])
            if i > 0:
                connector = self.rng.choice(self.connectors)
                parts.append(f"{connector} {template.lower()}")
            else:
                parts.append(template)
        return " ".join(parts)

    def _generate_extended_template(self) -> str:
        base_template = self._generate_single_template()
        fragment = self.rng.choice(self.fragments)
        extension = self.rng.choice(self.templates['technical'][:3])
        return f"{base_template} {fragment.title()} {extension.lower()}"

    def _generate_qa_pair(self) -> str:
        questions = [
            "What is the main purpose of", "How does the process of",
            "Why is it important to", "What are the key benefits of", "How can we improve"
        ]
        topics = [
            "machine learning in modern applications", "neural network training and optimization",
            "sustainable energy development worldwide", "scientific research and peer review",
            "biodiversity conservation efforts"
        ]
        question = f"{self.rng.choice(questions)} {self.rng.choice(topics)}?"
        return f"{question} {self._generate_single_template()}"

    def _apply_augmentation(self, text: str) -> str:
        if self.rng.random() < 0.1:
            text = text.replace('.', '!')
        if self.rng.random() < 0.05:
            words = text.split()
            idx = self.rng.randint(0, len(words) - 1)
            if words[idx].islower():
                words[idx] = words[idx].upper()
            text = ' '.join(words)
        return text


# ---------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------

class BLTTrainer:
    """Main training pipeline for BLT models."""

    def __init__(self, config: BLTTrainingConfig):
        self.config = config
        self._setup_environment()
        self.tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
        self.data_generator = TrainingDataGenerator(config)
        self.entropy_model: Optional[EntropyModel] = None
        self.blt_model: Optional[keras.Model] = None
        self.training_history: Optional[keras.callbacks.History] = None

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        config.save_to_file(str(Path(self.config.output_dir) / "config.json"))

    def _setup_environment(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        keras.utils.set_random_seed(self.config.random_seed)

        if self.config.mixed_precision:
            keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision training enabled")

        if self.config.use_gpu:
            pass  # GPU setup handled by main() via setup_gpu(gpu_id)

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Generating {self.config.num_training_samples} training samples...")
        texts = self.data_generator.generate_samples()

        for i, text in enumerate(texts[:3]):
            logger.info(f"  Sample {i + 1}: {text[:100]}...")

        tokenized_samples = []
        for text in texts:
            tokens = self.tokenizer.text_to_bytes(text, add_bos=True, add_eos=True)
            if 10 <= len(tokens) <= self.config.max_text_length:
                tokenized_samples.append(tokens)

        logger.info(f"Kept {len(tokenized_samples)} samples after filtering")

        padded_sequences = []
        for tokens in tokenized_samples:
            if len(tokens) < self.config.max_sequence_length:
                padded = tokens + [0] * (self.config.max_sequence_length - len(tokens))
            else:
                padded = tokens[:self.config.max_sequence_length]
            padded_sequences.append(padded)

        input_tokens = np.array(padded_sequences, dtype=np.int32)
        target_tokens = np.roll(input_tokens, -1, axis=1)
        target_tokens[:, -1] = 0

        logger.info(f"Training data: {input_tokens.shape}, tokens [{np.min(input_tokens)}, {np.max(input_tokens)}]")
        return input_tokens, target_tokens

    def train_entropy_model(self, train_x: np.ndarray, train_y: np.ndarray) -> EntropyModel:
        logger.info("Training entropy model...")
        self.entropy_model = EntropyModel(
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.entropy_model_hidden_dim,
            num_layers=self.config.entropy_model_layers,
            num_heads=self.config.entropy_model_heads,
            max_seq_len=self.config.max_sequence_length,
            dropout_rate=self.config.dropout_rate
        )

        # Wrap Layer in a Model for compile/fit
        entropy_input = keras.Input(shape=(train_x.shape[1],), dtype="int32", name="entropy_input")
        entropy_output = self.entropy_model(entropy_input)
        entropy_train_model = keras.Model(inputs=entropy_input, outputs=entropy_output, name="entropy_train_model")

        entropy_train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        entropy_history = entropy_train_model.fit(
            train_x, train_y,
            epochs=self.config.entropy_model_epochs,
            batch_size=max(self.config.batch_size, 8),
            validation_split=0.1, verbose=1
        )

        logger.info(f"Entropy model final loss: {entropy_history.history['loss'][-1]:.4f}")

        entropy_path = Path(self.config.output_dir) / "entropy_model.keras"
        entropy_train_model.save(str(entropy_path))
        return self.entropy_model

    def create_blt_model(self) -> keras.Model:
        logger.info("Creating BLT model...")
        model_config = self.config.get_model_config()
        model_config['entropy_model'] = self.entropy_model

        self.blt_model = create_blt_model(**model_config)
        self.blt_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        logger.info("BLT model created and compiled")
        return self.blt_model

    def create_callbacks(self) -> List[keras.callbacks.Callback]:
        callbacks = []

        if self.config.save_best_only:
            checkpoint_path = Path(self.config.output_dir) / "blt_model_best.keras"
            callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path), monitor='val_loss',
                save_best_only=True, verbose=1
            ))

        if self.config.checkpoint_frequency > 0:
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / "epoch_{epoch:03d}.keras"),
                save_freq=self.config.checkpoint_frequency * self.config.batch_size, verbose=0
            ))

        if self.config.early_stopping_patience > 0:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.config.early_stopping_patience,
                verbose=1, restore_best_weights=True
            ))

        if self.config.lr_reduction_patience > 0:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=self.config.lr_reduction_factor,
                patience=self.config.lr_reduction_patience, verbose=1,
                min_lr=self.config.min_learning_rate
            ))

        return callbacks

    def train_blt_model(self, train_x: np.ndarray, train_y: np.ndarray) -> keras.callbacks.History:
        logger.info("Training BLT model...")
        split_idx = int((1 - self.config.validation_split) * len(train_x))
        train_x_split, val_x = train_x[:split_idx], train_x[split_idx:]
        train_y_split, val_y = train_y[:split_idx], train_y[split_idx:]

        logger.info(f"Train: {len(train_x_split)}, Validation: {len(val_x)}")

        start_time = time.time()
        self.training_history = self.blt_model.fit(
            train_x_split, train_y_split,
            validation_data=(val_x, val_y),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.create_callbacks(), verbose=1
        )
        logger.info(f"Training completed in {time.time() - start_time:.2f}s")
        return self.training_history

    def evaluate_model(self) -> None:
        if not self.config.log_generation_examples:
            return

        test_prompts = [
            "Artificial intelligence", "The future of technology",
            "Scientific research shows", "Machine learning algorithms",
            "In recent studies", "Deep neural networks"
        ]

        logger.info("Generated text samples:")
        for prompt in test_prompts:
            try:
                generated = self.blt_model.generate(
                    prompt=prompt, max_new_tokens=self.config.generation_max_tokens,
                    temperature=self.config.generation_temperature,
                    top_p=self.config.generation_top_p, do_sample=True
                )
                logger.info(f"  '{prompt}' -> '{generated}'")
            except Exception as e:
                logger.warning(f"Generation failed for '{prompt}': {e}")

    def save_final_artifacts(self) -> None:
        final_model_path = Path(self.config.output_dir) / "blt_model_final.keras"
        self.blt_model.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

        if self.config.save_training_history and self.training_history:
            history_path = Path(self.config.output_dir) / "training_history.npz"
            np.savez(str(history_path), **self.training_history.history)

        self._save_training_summary()

    def _save_training_summary(self) -> None:
        if not self.training_history:
            return

        summary = {
            'config': asdict(self.config),
            'model_parameters': self.blt_model.count_params(),
            'final_train_loss': float(self.training_history.history['loss'][-1]),
            'final_val_loss': float(self.training_history.history['val_loss'][-1]),
            'best_val_loss': float(min(self.training_history.history['val_loss'])),
            'epochs_completed': len(self.training_history.history['loss'])
        }

        summary_path = Path(self.config.output_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Training summary: train_loss={summary['final_train_loss']:.4f}, "
            f"val_loss={summary['final_val_loss']:.4f}, "
            f"best_val_loss={summary['best_val_loss']:.4f}, "
            f"epochs={summary['epochs_completed']}"
        )

    def run_full_training(self) -> Tuple[keras.Model, keras.callbacks.History]:
        logger.info("=" * 60)
        logger.info("BLT Full Training Pipeline")
        logger.info("=" * 60)

        try:
            train_x, train_y = self.prepare_training_data()
            self.train_entropy_model(train_x, train_y)
            self.create_blt_model()
            self.train_blt_model(train_x, train_y)
            self.evaluate_model()
            self.save_final_artifacts()

            logger.info("BLT Training Pipeline Completed Successfully!")
            logger.info(f"Artifacts saved in: {self.config.output_dir}/")
            return self.blt_model, self.training_history

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train BLT Model")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--preset", type=str, choices=['quick', 'small', 'large'],
                        help="Use preset configuration")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")

    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    if args.config:
        config = BLTTrainingConfig.load_from_file(args.config)
    elif args.preset == 'quick':
        config = get_quick_test_config()
    elif args.preset == 'small':
        config = get_small_model_config()
    elif args.preset == 'large':
        config = get_large_model_config()
    else:
        config = BLTTrainingConfig()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    trainer = BLTTrainer(config)
    trainer.run_full_training()


def quick_test() -> None:
    logger.info("Running BLT quick test...")
    config = get_quick_test_config()
    trainer = BLTTrainer(config)
    trainer.run_full_training()
    logger.info("Quick test completed successfully!")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
            quick_test()
        else:
            main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)
