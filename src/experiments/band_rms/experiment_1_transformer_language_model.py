"""
Experiment 1: BandRMS Evaluation in Transformer Language Model

This experiment evaluates BandRMS against RMSNorm, LayerNorm, and no normalization
in a lightweight Transformer architecture for language modeling.

Objectives:
- Compare final validation perplexity across normalization techniques
- Analyze training dynamics and stability
- Evaluate different max_band_width values
- Use Model Analyzer for comprehensive performance analysis
"""

import json
import numpy as np
import keras
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.metrics.perplexity_metric import Perplexity
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder


@keras.saving.register_keras_serializable()
class LightweightTransformer(keras.Model):
    """
    Lightweight Transformer for language modeling experiments.

    Args:
        vocab_size: Size of vocabulary
        seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        normalization_type: Type of normalization ('band_rms', 'rms_norm', 'layer_norm', 'none')
        max_band_width: Maximum band width for BandRMS (only used if normalization_type='band_rms')
    """

    def __init__(
            self,
            vocab_size: int = 10000,
            seq_len: int = 128,
            embed_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 6,
            normalization_type: str = 'rms_norm',
            max_band_width: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.normalization_type = normalization_type
        self.max_band_width = max_band_width

        # Token and position embeddings
        self.token_embedding = keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    hidden_size=embed_dim,
                    num_heads=num_heads,
                    intermediate_size=embed_dim * 4,
                    normalization_type=self._get_norm_type(),
                    normalization_position='pre',
                    dropout_rate=0.1,
                    attention_dropout_rate=0.1,
                    name=f'transformer_layer_{i}'
                )
            )

        # Final normalization
        self.final_norm = self._create_norm_layer('final_norm')

        # Output projection
        self.output_projection = keras.layers.Dense(vocab_size, name='output_projection')

    def _get_norm_type(self) -> str:
        """Convert our normalization type to TransformerLayer format."""
        if self.normalization_type == 'band_rms':
            return 'band_rms'
        elif self.normalization_type == 'rms_norm':
            return 'rms_norm'
        elif self.normalization_type == 'layer_norm':
            return 'layer_norm'
        else:  # none
            return 'layer_norm'  # TransformerLayer will use identity if needed

    def _create_norm_layer(self, name: str) -> keras.layers.Layer:
        """Create normalization layer based on type."""
        if self.normalization_type == 'band_rms':
            return BandRMS(max_band_width=self.max_band_width, name=name)
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(name=name)
        elif self.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(name=name)
        else:  # none
            return keras.layers.Lambda(lambda x: x, name=name)

    def call(self, inputs: keras.KerasTensor, training: bool = None) -> keras.KerasTensor:
        """Forward pass through the transformer."""
        seq_len = keras.ops.shape(inputs)[1]

        # Embeddings
        x = self.token_embedding(inputs)
        x = self.position_embedding(x)

        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        # Final normalization and projection
        x = self.final_norm(x, training=training)
        logits = self.output_projection(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'normalization_type': self.normalization_type,
            'max_band_width': self.max_band_width,
        })
        return config


class TransformerExperiment:
    """
    Experiment runner for BandRMS evaluation in Transformer models.
    """

    def __init__(self, output_dir: str = 'experiments/transformer'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment configurations
        self.configs = {
            'band_rms_0.05': {'normalization_type': 'band_rms', 'max_band_width': 0.05},
            'band_rms_0.1': {'normalization_type': 'band_rms', 'max_band_width': 0.1},
            'band_rms_0.2': {'normalization_type': 'band_rms', 'max_band_width': 0.2},
            'rms_norm': {'normalization_type': 'rms_norm'},
            'layer_norm': {'normalization_type': 'layer_norm'},
            'no_norm': {'normalization_type': 'none'},
        }

        # Training settings
        self.batch_size = 32
        self.seq_len = 128
        self.vocab_size = 8000
        self.epochs = 20
        self.validation_split = 0.1

    def create_synthetic_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic language modeling dataset.

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        logger.info("Creating synthetic language modeling dataset")

        # Generate synthetic sequences that have some structure
        num_samples = 10000

        # Create sequences with some patterns (simple n-gram-like structure)
        sequences = []
        for _ in range(num_samples):
            # Start with random token
            seq = [np.random.randint(1, self.vocab_size)]

            for i in range(1, self.seq_len):
                # Add some structure: next token influenced by previous tokens
                if i > 2:
                    # Pattern-based generation with some randomness
                    pattern = (seq[i - 1] + seq[i - 2]) % (self.vocab_size // 2) + (self.vocab_size // 2)
                    next_token = pattern if np.random.random() < 0.7 else np.random.randint(1, self.vocab_size)
                else:
                    next_token = np.random.randint(1, self.vocab_size)
                seq.append(next_token)

            sequences.append(seq)

        sequences = np.array(sequences, dtype=np.int32)

        # For language modeling: input = seq[:-1], target = seq[1:]
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        logger.info(f"Created dataset with {len(inputs)} sequences")
        logger.info(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

        return inputs, targets

    def create_model(self, config_name: str) -> LightweightTransformer:
        """Create model with specified configuration."""
        config = self.configs[config_name]

        model = LightweightTransformer(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            name=f'transformer_{config_name}',
            **config
        )

        # Build model
        dummy_input = np.zeros((1, self.seq_len - 1), dtype=np.int32)
        _ = model(dummy_input)

        return model

    def setup_training(self, model: keras.Model) -> Tuple[keras.optimizers.Optimizer, Any]:
        """Setup optimizer and learning rate schedule."""
        # Learning rate schedule
        lr_config = {
            "type": "cosine_decay",
            "warmup_steps": 1000,
            "warmup_start_lr": 1e-6,
            "learning_rate": 3e-4,
            "decay_steps": 5000,
            "alpha": 0.1
        }

        # Optimizer configuration
        opt_config = {
            "type": "adamw",
            "beta_1": 0.9,
            "beta_2": 0.999,
            "gradient_clipping_by_norm": 1.0
        }

        lr_schedule = learning_rate_schedule_builder(lr_config)
        optimizer = optimizer_builder(opt_config, lr_schedule)

        return optimizer, lr_schedule

    def train_model(
            self,
            model: keras.Model,
            x_train: np.ndarray,
            y_train: np.ndarray,
            config_name: str
    ) -> keras.callbacks.History:
        """Train a single model configuration."""
        logger.info(f"Training model: {config_name}")

        # Setup training
        optimizer, lr_schedule = self.setup_training(model)

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(),
                Perplexity(from_logits=True, name='perplexity')
            ]
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        # Train model
        history = model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        model_path = self.output_dir / f"{config_name}_model.keras"
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")

        return history

    def analyze_models(self, models: Dict[str, keras.Model], test_data: DataInput) -> Dict[str, Any]:
        """Analyze all models using ModelAnalyzer."""
        logger.info("Running comprehensive model analysis")

        # Setup analyzer configuration
        analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_calibration=True,
            analyze_information_flow=True,
            analyze_training_dynamics=True,
            save_plots=True,
            save_format='png',
            dpi=150,
            plot_style='publication'
        )

        # Prepare training histories
        histories = {}
        for config_name in models.keys():
            history_path = self.output_dir / f"{config_name}_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    histories[config_name] = json.load(f)

        # Run analysis
        analyzer = ModelAnalyzer(
            models=models,
            training_history=histories,
            config=analysis_config,
            output_dir=str(self.output_dir / 'analysis')
        )

        results = analyzer.analyze(data=test_data)

        # Get summary statistics
        summary = analyzer.get_summary_statistics()

        # Save summary
        summary_path = self.output_dir / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Analysis complete. Results saved to {self.output_dir / 'analysis'}")

        return summary

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        logger.info("Starting Transformer BandRMS experiment")

        # Create dataset
        x_data, y_data = self.create_synthetic_dataset()

        # Split into train/test
        split_idx = int(len(x_data) * 0.8)
        x_train, x_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]

        # Store results
        results = {}
        models = {}
        histories = {}

        # Train each configuration
        for config_name in self.configs.keys():
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training configuration: {config_name}")
            logger.info(f"{'=' * 50}")

            try:
                # Create model
                model = self.create_model(config_name)
                models[config_name] = model

                # Train model
                history = self.train_model(model, x_train, y_train, config_name)
                histories[config_name] = history.history

                # Save history
                history_path = self.output_dir / f"{config_name}_history.json"
                with open(history_path, 'w') as f:
                    json.dump(history.history, f, indent=2, default=str)

                # Evaluate on test set
                test_results = model.evaluate(x_test, y_test, verbose=0)
                results[config_name] = {
                    'test_loss': float(test_results[0]),
                    'test_accuracy': float(test_results[1]),
                    'test_perplexity': float(test_results[2]) if len(test_results) > 2 else None,
                    'best_val_loss': float(min(history.history['val_loss'])),
                    'best_val_accuracy': float(max(history.history['val_sparse_categorical_accuracy'])),
                    'epochs_trained': len(history.history['loss'])
                }

                logger.info(f"Results for {config_name}:")
                logger.info(f"  Test Loss: {results[config_name]['test_loss']:.4f}")
                logger.info(f"  Test Accuracy: {results[config_name]['test_accuracy']:.4f}")
                logger.info(f"  Best Val Loss: {results[config_name]['best_val_loss']:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {config_name}: {str(e)}")
                results[config_name] = {'error': str(e)}

        # Prepare test data for analysis
        test_data = DataInput(x_data=x_test, y_data=y_test)

        # Run comprehensive analysis
        analysis_summary = self.analyze_models(models, test_data)

        # Create comparison plots
        self._create_comparison_plots(results, histories)

        # Save final results
        final_results = {
            'training_results': results,
            'analysis_summary': analysis_summary
        }

        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"\nExperiment complete! Results saved to {self.output_dir}")
        return final_results

    def _create_comparison_plots(self, results: Dict[str, Any], histories: Dict[str, Any]):
        """Create comparison plots for the experiment."""

        # Training curves comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        colors = plt.cm.Set3(np.linspace(0, 1, len(histories)))

        for i, (config_name, history) in enumerate(histories.items()):
            if 'error' in results.get(config_name, {}):
                continue

            epochs = range(1, len(history['loss']) + 1)
            color = colors[i]

            # Training loss
            ax1.plot(epochs, history['loss'], color=color, label=config_name, linewidth=2)

            # Validation loss
            ax2.plot(epochs, history['val_loss'], color=color, label=config_name, linewidth=2)

            # Training accuracy
            if 'sparse_categorical_accuracy' in history:
                ax3.plot(epochs, history['sparse_categorical_accuracy'], color=color, label=config_name, linewidth=2)

            # Validation accuracy
            if 'val_sparse_categorical_accuracy' in history:
                ax4.plot(epochs, history['val_sparse_categorical_accuracy'], color=color, label=config_name,
                         linewidth=2)

        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Final performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        configs = []
        test_losses = []
        test_accuracies = []

        for config_name, result in results.items():
            if 'error' not in result:
                configs.append(config_name)
                test_losses.append(result['test_loss'])
                test_accuracies.append(result['test_accuracy'])

        x_pos = np.arange(len(configs))

        # Test loss comparison
        bars1 = ax1.bar(x_pos, test_losses, color=colors[:len(configs)])
        ax1.set_title('Final Test Loss Comparison')
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Test Loss')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, test_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # Test accuracy comparison
        bars2 = ax2.bar(x_pos, test_accuracies, color=colors[:len(configs)])
        ax2.set_title('Final Test Accuracy Comparison')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, test_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Comparison plots saved")


def main():
    """Run the Transformer BandRMS experiment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    # Create and run experiment
    experiment = TransformerExperiment(output_dir='experiments/transformer_bandrms')
    results = experiment.run_experiment()

    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    for config_name, result in results['training_results'].items():
        if 'error' not in result:
            logger.info(f"{config_name}:")
            logger.info(f"  Test Loss: {result['test_loss']:.4f}")
            logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            logger.info(f"  Best Val Loss: {result['best_val_loss']:.4f}")
            logger.info(f"  Epochs Trained: {result['epochs_trained']}")
        else:
            logger.info(f"\n{config_name}: FAILED - {result['error']}")


if __name__ == "__main__":
    main()