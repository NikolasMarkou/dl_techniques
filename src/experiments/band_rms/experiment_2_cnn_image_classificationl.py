"""
Experiment 2: BandRMS Evaluation in CNN for Image Classification

This experiment evaluates BandRMS against RMSNorm, LayerNorm, and BatchNorm
in a lightweight CNN architecture for image classification.

Objectives:
- Compare final validation accuracy across normalization techniques
- Analyze training stability and convergence
- Evaluate different max_band_width values
- Test performance across different batch sizes
- Use Model Analyzer for comprehensive performance analysis
"""

import json
import numpy as np
import keras
import tensorflow as tf
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.squeeze_excitation import SqueezeExcitation
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.utils.datasets.cifar10 import load_and_preprocess_cifar10


@keras.saving.register_keras_serializable()
class LightweightCNN(keras.Model):
    """
    Lightweight CNN for image classification experiments.

    Architecture inspired by ResNet but simplified for quick experiments.

    Args:
        num_classes: Number of output classes
        normalization_type: Type of normalization ('band_rms', 'rms_norm', 'layer_norm', 'batch_norm', 'none')
        max_band_width: Maximum band width for BandRMS (only used if normalization_type='band_rms')
        use_squeeze_excitation: Whether to use Squeeze-and-Excitation blocks
    """

    def __init__(
        self,
        num_classes: int = 10,
        normalization_type: str = 'batch_norm',
        max_band_width: float = 0.1,
        use_squeeze_excitation: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.normalization_type = normalization_type
        self.max_band_width = max_band_width
        self.use_squeeze_excitation = use_squeeze_excitation

        # Initial conv layer
        self.initial_conv = keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)
        self.initial_norm = self._create_norm_layer(64, 'initial_norm')
        self.initial_activation = keras.layers.Activation('relu')
        self.initial_pool = keras.layers.MaxPooling2D(3, strides=2, padding='same')

        # ResNet-style blocks
        self.block1 = self._create_residual_block(64, 'block1')
        self.block2 = self._create_residual_block(128, 'block2', strides=2)
        self.block3 = self._create_residual_block(256, 'block3', strides=2)
        self.block4 = self._create_residual_block(512, 'block4', strides=2)

        # Global pooling and classifier
        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.dropout = keras.layers.Dropout(0.5)
        self.classifier = keras.layers.Dense(num_classes)

    def _create_norm_layer(self, channels: int, name: str) -> keras.layers.Layer:
        """Create normalization layer based on type."""
        if self.normalization_type == 'band_rms':
            return BandRMS(max_band_width=self.max_band_width, axis=-1, name=name)
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(axis=-1, name=name)
        elif self.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(axis=-1, name=name)
        elif self.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(name=name)
        else:  # none
            return keras.layers.Lambda(lambda x: x, name=name)

    def _create_residual_block(
        self,
        filters: int,
        name: str,
        strides: int = 1
    ) -> keras.layers.Layer:
        """Create a residual block with normalization."""

        class ResidualBlock(keras.layers.Layer):
            def __init__(self, parent_model, filters, strides, block_name, **kwargs):
                super().__init__(name=block_name, **kwargs)
                self.parent_model = parent_model
                self.filters = filters
                self.strides = strides

                # Main path
                self.conv1 = keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)
                self.norm1 = parent_model._create_norm_layer(filters, f'{block_name}_norm1')
                self.act1 = keras.layers.Activation('relu')

                self.conv2 = keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)
                self.norm2 = parent_model._create_norm_layer(filters, f'{block_name}_norm2')

                # Squeeze-and-Excitation (optional)
                if parent_model.use_squeeze_excitation:
                    self.se = SqueezeExcitation(reduction_ratio=0.25)
                else:
                    self.se = None

                # Shortcut path
                self.use_projection = strides > 1 or True  # Always use projection for simplicity
                if self.use_projection:
                    self.shortcut_conv = keras.layers.Conv2D(filters, 1, strides=strides, use_bias=False)
                    self.shortcut_norm = parent_model._create_norm_layer(filters, f'{block_name}_shortcut_norm')

                self.final_act = keras.layers.Activation('relu')

            def call(self, inputs, training=None):
                # Main path
                x = self.conv1(inputs)
                x = self.norm1(x, training=training)
                x = self.act1(x)

                x = self.conv2(x)
                x = self.norm2(x, training=training)

                # Squeeze-and-Excitation
                if self.se is not None:
                    x = self.se(x)

                # Shortcut path
                if self.use_projection:
                    shortcut = self.shortcut_conv(inputs)
                    shortcut = self.shortcut_norm(shortcut, training=training)
                else:
                    shortcut = inputs

                # Combine and activate
                x = x + shortcut
                x = self.final_act(x)

                return x

        return ResidualBlock(self, filters, strides, name)

    def call(self, inputs: keras.KerasTensor, training: bool = None) -> keras.KerasTensor:
        """Forward pass through the CNN."""
        # Initial processing
        x = self.initial_conv(inputs)
        x = self.initial_norm(x, training=training)
        x = self.initial_activation(x)
        x = self.initial_pool(x)

        # Residual blocks
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        # Classification head
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        logits = self.classifier(x)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'normalization_type': self.normalization_type,
            'max_band_width': self.max_band_width,
            'use_squeeze_excitation': self.use_squeeze_excitation,
        })
        return config


class CNNExperiment:
    """
    Experiment runner for BandRMS evaluation in CNN models.
    """

    def __init__(self, output_dir: str = 'experiments/cnn'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment configurations
        self.configs = {
            'band_rms_0.05': {'normalization_type': 'band_rms', 'max_band_width': 0.05},
            'band_rms_0.1': {'normalization_type': 'band_rms', 'max_band_width': 0.1},
            'band_rms_0.2': {'normalization_type': 'band_rms', 'max_band_width': 0.2},
            'rms_norm': {'normalization_type': 'rms_norm'},
            'layer_norm': {'normalization_type': 'layer_norm'},
            'batch_norm': {'normalization_type': 'batch_norm'},
            'no_norm': {'normalization_type': 'none'},
        }

        # Training settings
        self.batch_sizes = [32, 64]  # Test different batch sizes
        self.epochs = 30
        self.num_classes = 10
        self.validation_split = 0.1

    def load_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess CIFAR-10 dataset."""
        logger.info("Loading CIFAR-10 dataset")

        try:
            (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()
        except:
            # Fallback to standard Keras dataset
            logger.warning("Using fallback Keras CIFAR-10 dataset")
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

            # Normalize
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0

            # Flatten labels
            y_train = y_train.flatten()
            y_test = y_test.flatten()

        # Use subset for faster experiments
        subset_size = 10000
        indices = np.random.choice(len(x_train), subset_size, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

        logger.info(f"Dataset shapes - Train: {x_train.shape}, Test: {x_test.shape}")
        logger.info(f"Label shapes - Train: {y_train.shape}, Test: {y_test.shape}")

        return (x_train, y_train), (x_test, y_test)

    def create_model(self, config_name: str) -> LightweightCNN:
        """Create model with specified configuration."""
        config = self.configs[config_name]

        model = LightweightCNN(
            num_classes=self.num_classes,
            use_squeeze_excitation=False,
            name=f'cnn_{config_name}',
            **config
        )

        # Build model
        dummy_input = np.zeros((1, 32, 32, 3), dtype=np.float32)
        _ = model(dummy_input)

        return model

    def setup_training(self, model: keras.Model) -> Tuple[keras.optimizers.Optimizer, Any]:
        """Setup optimizer and learning rate schedule."""
        # Learning rate schedule
        lr_config = {
            "type": "cosine_decay",
            "warmup_steps": 500,
            "warmup_start_lr": 1e-5,
            "learning_rate": 1e-3,
            "decay_steps": 3000,
            "alpha": 0.01
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
        config_name: str,
        batch_size: int = 32
    ) -> keras.callbacks.History:
        """Train a single model configuration."""
        logger.info(f"Training model: {config_name} with batch_size={batch_size}")

        # Setup training
        optimizer, lr_schedule = self.setup_training(model)

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )

        # Data augmentation
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6
            )
        ]

        # Prepare training data with augmentation
        def preprocess_train(x, y):
            x = data_augmentation(x, training=True)
            return x, y

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train model
        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        model_path = self.output_dir / f"{config_name}_bs{batch_size}_model.keras"
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")

        return history

    def analyze_models(
        self,
        models: Dict[str, keras.Model],
        test_data: DataInput,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Analyze models using ModelAnalyzer."""
        logger.info(f"Running comprehensive model analysis for batch_size={batch_size}")

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
            history_path = self.output_dir / f"{config_name}_bs{batch_size}_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    histories[config_name] = json.load(f)

        # Run analysis
        analysis_dir = self.output_dir / f'analysis_bs{batch_size}'
        analyzer = ModelAnalyzer(
            models=models,
            training_history=histories,
            config=analysis_config,
            output_dir=str(analysis_dir)
        )

        results = analyzer.analyze(data=test_data)

        # Get summary statistics
        summary = analyzer.get_summary_statistics()

        # Save summary
        summary_path = self.output_dir / f'analysis_summary_bs{batch_size}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Analysis complete. Results saved to {analysis_dir}")

        return summary

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        logger.info("Starting CNN BandRMS experiment")

        # Load dataset
        (x_train, y_train), (x_test, y_test) = self.load_dataset()

        # Store results
        all_results = {}

        # Run experiments for different batch sizes
        for batch_size in self.batch_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiments with batch_size={batch_size}")
            logger.info(f"{'='*60}")

            results = {}
            models = {}
            histories = {}

            # Train each configuration
            for config_name in self.configs.keys():
                logger.info(f"\nTraining configuration: {config_name}")

                try:
                    # Create model
                    model = self.create_model(config_name)
                    models[config_name] = model

                    # Train model
                    history = self.train_model(model, x_train, y_train, config_name, batch_size)
                    histories[config_name] = history.history

                    # Save history
                    history_path = self.output_dir / f"{config_name}_bs{batch_size}_history.json"
                    with open(history_path, 'w') as f:
                        json.dump(history.history, f, indent=2, default=str)

                    # Evaluate on test set
                    test_results = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
                    results[config_name] = {
                        'test_loss': float(test_results[0]),
                        'test_accuracy': float(test_results[1]),
                        'test_top5_accuracy': float(test_results[2]) if len(test_results) > 2 else None,
                        'best_val_loss': float(min(history.history['val_loss'])),
                        'best_val_accuracy': float(max(history.history['val_accuracy'])),
                        'epochs_trained': len(history.history['loss']),
                        'batch_size': batch_size
                    }

                    logger.info(f"Results for {config_name} (bs={batch_size}):")
                    logger.info(f"  Test Loss: {results[config_name]['test_loss']:.4f}")
                    logger.info(f"  Test Accuracy: {results[config_name]['test_accuracy']:.4f}")
                    logger.info(f"  Best Val Loss: {results[config_name]['best_val_loss']:.4f}")

                except Exception as e:
                    logger.error(f"Failed to train {config_name}: {str(e)}")
                    results[config_name] = {'error': str(e), 'batch_size': batch_size}

            # Prepare test data for analysis
            test_data = DataInput(x_data=x_test, y_data=y_test)

            # Run comprehensive analysis
            analysis_summary = self.analyze_models(models, test_data, batch_size)

            # Create comparison plots for this batch size
            self._create_comparison_plots(results, histories, batch_size)

            # Store results for this batch size
            all_results[f'batch_size_{batch_size}'] = {
                'training_results': results,
                'analysis_summary': analysis_summary
            }

        # Create cross-batch-size comparison
        self._create_batch_size_comparison(all_results)

        # Save final results
        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"\nExperiment complete! Results saved to {self.output_dir}")
        return all_results

    def _create_comparison_plots(
        self,
        results: Dict[str, Any],
        histories: Dict[str, Any],
        batch_size: int
    ):
        """Create comparison plots for a specific batch size."""

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
            if 'accuracy' in history:
                ax3.plot(epochs, history['accuracy'], color=color, label=config_name, linewidth=2)

            # Validation accuracy
            if 'val_accuracy' in history:
                ax4.plot(epochs, history['val_accuracy'], color=color, label=config_name, linewidth=2)

        ax1.set_title(f'Training Loss (Batch Size: {batch_size})')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        ax2.set_title(f'Validation Loss (Batch Size: {batch_size})')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        ax3.set_title(f'Training Accuracy (Batch Size: {batch_size})')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.set_title(f'Validation Accuracy (Batch Size: {batch_size})')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_curves_bs{batch_size}.png', dpi=150, bbox_inches='tight')
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
        ax1.set_title(f'Final Test Loss (Batch Size: {batch_size})')
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Test Loss')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, test_losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # Test accuracy comparison
        bars2 = ax2.bar(x_pos, test_accuracies, color=colors[:len(configs)])
        ax2.set_title(f'Final Test Accuracy (Batch Size: {batch_size})')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, test_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'final_performance_bs{batch_size}.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plots saved for batch_size={batch_size}")

    def _create_batch_size_comparison(self, all_results: Dict[str, Any]):
        """Create comparison plots across different batch sizes."""

        # Extract data for batch size comparison
        batch_sizes = []
        configs_data = {}

        for batch_key, batch_results in all_results.items():
            if 'batch_size_' in batch_key:
                batch_size = int(batch_key.split('_')[2])
                batch_sizes.append(batch_size)

                for config_name, result in batch_results['training_results'].items():
                    if 'error' not in result:
                        if config_name not in configs_data:
                            configs_data[config_name] = {'batch_sizes': [], 'accuracies': [], 'losses': []}

                        configs_data[config_name]['batch_sizes'].append(batch_size)
                        configs_data[config_name]['accuracies'].append(result['test_accuracy'])
                        configs_data[config_name]['losses'].append(result['test_loss'])

        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, len(configs_data)))

        for i, (config_name, data) in enumerate(configs_data.items()):
            ax1.plot(data['batch_sizes'], data['accuracies'], 'o-', color=colors[i],
                    label=config_name, linewidth=2, markersize=8)
            ax2.plot(data['batch_sizes'], data['losses'], 'o-', color=colors[i],
                    label=config_name, linewidth=2, markersize=8)

        ax1.set_title('Test Accuracy vs Batch Size')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Test Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(sorted(set(batch_sizes)))

        ax2.set_title('Test Loss vs Batch Size')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Test Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(sorted(set(batch_sizes)))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_size_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Batch size comparison plots saved")


def main():
    """Run the CNN BandRMS experiment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    # Create and run experiment
    experiment = CNNExperiment(output_dir='experiments/cnn_bandrms')
    results = experiment.run_experiment()

    # Print summary
    logger.info("="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)

    for batch_key, batch_results in results.items():
        logger.info(f"{batch_key.upper()}:")
        logger.info("-" * 40)

        for config_name, result in batch_results['training_results'].items():
            if 'error' not in result:
                logger.info(f"{config_name}:")
                logger.info(f"  Test Loss: {result['test_loss']:.4f}")
                logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
                logger.info(f"  Best Val Loss: {result['best_val_loss']:.4f}")
                logger.info(f"  Epochs Trained: {result['epochs_trained']}")
            else:
                logger.info(f"{config_name}: FAILED - {result['error']}")


if __name__ == "__main__":
    main()