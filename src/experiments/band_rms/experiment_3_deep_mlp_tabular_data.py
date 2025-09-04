"""
Experiment 3: BandRMS Evaluation in Deep MLP for Tabular Data

This experiment evaluates BandRMS against RMSNorm, LayerNorm, and no normalization
in very deep MLP architectures for tabular data classification/regression.

Objectives:
- Test training stability in very deep networks (20+ layers)
- Compare gradient flow and activation statistics
- Evaluate learned band parameters and their utilization
- Analyze the ability to train deeper networks than baselines
- Use Model Analyzer for comprehensive performance analysis
"""

import json
import numpy as np
import keras
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder


@keras.saving.register_keras_serializable()
class DeepMLP(keras.Model):
    """
    Very deep MLP for tabular data experiments.

    This model is designed to test the limits of normalization techniques
    by using many layers that would typically cause vanishing/exploding gradients.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        task_type: 'classification' or 'regression'
        normalization_type: Type of normalization ('band_rms', 'rms_norm', 'layer_norm', 'none')
        max_band_width: Maximum band width for BandRMS
        dropout_rate: Dropout rate after each hidden layer
        activation: Activation function for hidden layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 512, 256, 256, 128, 128, 64, 64],
        output_dim: int = 1,
        task_type: str = 'classification',
        normalization_type: str = 'layer_norm',
        max_band_width: float = 0.1,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task_type = task_type
        self.normalization_type = normalization_type
        self.max_band_width = max_band_width
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Store layer references for analysis
        self.dense_layers = []
        self.norm_layers = []
        self.activation_layers = []
        self.dropout_layers = []

        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Dense layer
            dense = keras.layers.Dense(
                hidden_dim,
                activation=None,  # Apply activation after normalization
                kernel_initializer='he_normal',
                name=f'dense_{i}'
            )
            self.dense_layers.append(dense)

            # Normalization layer
            norm = self._create_norm_layer(f'norm_{i}')
            self.norm_layers.append(norm)

            # Activation
            act = keras.layers.Activation(activation, name=f'activation_{i}')
            self.activation_layers.append(act)

            # Dropout
            dropout = keras.layers.Dropout(dropout_rate, name=f'dropout_{i}')
            self.dropout_layers.append(dropout)

        # Output layer
        if task_type == 'classification':
            if output_dim == 1:
                # Binary classification
                self.output_layer = keras.layers.Dense(
                    output_dim,
                    activation='sigmoid',
                    name='output'
                )
            else:
                # Multi-class classification
                self.output_layer = keras.layers.Dense(
                    output_dim,
                    activation='softmax',
                    name='output'
                )
        else:
            # Regression
            self.output_layer = keras.layers.Dense(
                output_dim,
                activation=None,
                name='output'
            )

        # For tracking gradients and activations
        self.gradient_norms = {}
        self.activation_stats = {}

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
        """Forward pass through the deep MLP."""
        x = inputs

        # Store activations for analysis
        activations = []

        # Hidden layers
        for i, (dense, norm, activation, dropout) in enumerate(
            zip(self.dense_layers, self.norm_layers, self.activation_layers, self.dropout_layers)
        ):
            x = dense(x)
            x = norm(x, training=training)
            x = activation(x)

            # Store activation statistics
            if training is not False:  # During training or inference
                activations.append(x)

            x = dropout(x, training=training)

        # Output layer
        output = self.output_layer(x)

        # Store activation statistics (in training mode only)
        if training is not False and hasattr(self, 'activation_stats'):
            self.activation_stats['activations'] = activations

        return output

    def get_band_parameters(self) -> Dict[str, float]:
        """Get learned band parameters from BandRMS layers."""
        if self.normalization_type != 'band_rms':
            return {}

        band_params = {}
        for i, norm_layer in enumerate(self.norm_layers):
            if hasattr(norm_layer, 'band_param'):
                band_params[f'layer_{i}'] = float(keras.ops.convert_to_numpy(norm_layer.band_param))

        return band_params

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'task_type': self.task_type,
            'normalization_type': self.normalization_type,
            'max_band_width': self.max_band_width,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
        })
        return config


class GradientCallback(keras.callbacks.Callback):
    """Callback to monitor gradients during training."""

    def __init__(self, model: DeepMLP, x_val: np.ndarray, y_val: np.ndarray):
        super().__init__()
        self.tracked_model = model
        self.x_val = x_val
        self.y_val = y_val
        self.gradient_norms = []
        self.activation_means = []
        self.activation_stds = []

    def on_epoch_end(self, epoch, logs=None):
        """Compute gradient norms and activation statistics."""
        # Sample a small batch for gradient analysis
        batch_size = min(32, len(self.x_val))
        indices = np.random.choice(len(self.x_val), batch_size, replace=False)
        x_batch = self.x_val[indices]
        y_batch = self.y_val[indices]

        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = self.tracked_model(x_batch, training=True)
            if self.tracked_model.task_type == 'classification':
                if self.tracked_model.output_dim == 1:
                    loss = keras.losses.binary_crossentropy(y_batch, predictions)
                else:
                    loss = keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
            else:
                loss = keras.losses.mean_squared_error(y_batch, predictions)
            loss = keras.ops.mean(loss)

        gradients = tape.gradient(loss, self.tracked_model.trainable_variables)

        # Compute gradient norms for each layer
        layer_gradient_norms = []
        for grad in gradients:
            if grad is not None:
                grad_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(grad)))
                layer_gradient_norms.append(float(keras.ops.convert_to_numpy(grad_norm)))
            else:
                layer_gradient_norms.append(0.0)

        self.gradient_norms.append({
            'epoch': epoch,
            'layer_norms': layer_gradient_norms,
            'mean_norm': np.mean(layer_gradient_norms),
            'max_norm': np.max(layer_gradient_norms)
        })

        # Compute activation statistics
        if hasattr(self.tracked_model, 'activation_stats') and 'activations' in self.tracked_model.activation_stats:
            activations = self.tracked_model.activation_stats['activations']

            layer_means = []
            layer_stds = []

            for activation in activations:
                mean_val = float(keras.ops.convert_to_numpy(keras.ops.mean(activation)))
                std_val = float(keras.ops.convert_to_numpy(keras.ops.std(activation)))
                layer_means.append(mean_val)
                layer_stds.append(std_val)

            self.activation_means.append({
                'epoch': epoch,
                'layer_means': layer_means,
                'overall_mean': np.mean(layer_means)
            })

            self.activation_stds.append({
                'epoch': epoch,
                'layer_stds': layer_stds,
                'overall_std': np.mean(layer_stds)
            })


class MLPExperiment:
    """
    Experiment runner for BandRMS evaluation in deep MLP models.
    """

    def __init__(self, output_dir: str = 'experiments/mlp'):
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
        self.batch_size = 64
        self.epochs = 50
        self.validation_split = 0.2

        # Network architecture settings
        self.depth_variants = [
            {'name': 'shallow', 'hidden_dims': [256, 128, 64]},
            {'name': 'medium', 'hidden_dims': [512, 256, 256, 128, 64]},
            {'name': 'deep', 'hidden_dims': [512, 512, 256, 256, 128, 128, 64, 64]},
            {'name': 'very_deep', 'hidden_dims': [512] * 6 + [256] * 4 + [128] * 4 + [64] * 2},
        ]

    def create_synthetic_dataset(
        self,
        task_type: str = 'classification'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create synthetic tabular dataset."""
        logger.info(f"Creating synthetic {task_type} dataset")

        if task_type == 'classification':
            # Multi-class classification
            X, y = make_classification(
                n_samples=5000,
                n_features=50,
                n_informative=30,
                n_redundant=10,
                n_clusters_per_class=2,
                n_classes=5,
                random_state=42,
                class_sep=0.8
            )
        else:
            # Regression
            X, y = make_regression(
                n_samples=5000,
                n_features=50,
                n_informative=30,
                noise=0.1,
                random_state=42
            )
            # Make target 2D for multi-output regression
            y = np.column_stack([y, y * 0.5 + np.random.normal(0, 0.1, len(y))])

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
        )

        logger.info(f"Dataset created - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Task type: {task_type}, Output shape: {y_train.shape}")

        return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test

    def create_model(
        self,
        config_name: str,
        depth_variant: Dict[str, Any],
        input_dim: int,
        output_dim: int,
        task_type: str
    ) -> DeepMLP:
        """Create model with specified configuration."""
        config = self.configs[config_name]

        model = DeepMLP(
            input_dim=input_dim,
            hidden_dims=depth_variant['hidden_dims'],
            output_dim=output_dim,
            task_type=task_type,
            dropout_rate=0.1,
            activation='relu',
            name=f'mlp_{config_name}_{depth_variant["name"]}',
            **config
        )

        # Build model
        dummy_input = np.zeros((1, input_dim), dtype=np.float32)
        _ = model(dummy_input)

        return model

    def setup_training(self, model: keras.Model, task_type: str) -> Tuple[keras.optimizers.Optimizer, Any]:
        """Setup optimizer and learning rate schedule."""
        # Learning rate schedule
        lr_config = {
            "type": "cosine_decay",
            "warmup_steps": 500,
            "warmup_start_lr": 1e-6,
            "learning_rate": 1e-3,
            "decay_steps": 2000,
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
        model: DeepMLP,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config_name: str,
        depth_name: str,
        task_type: str
    ) -> Tuple[keras.callbacks.History, GradientCallback]:
        """Train a single model configuration."""
        logger.info(f"Training model: {config_name}_{depth_name}")

        # Setup training
        optimizer, lr_schedule = self.setup_training(model, task_type)

        # Compile model
        if task_type == 'classification':
            if model.output_dim == 1:
                loss = keras.losses.BinaryCrossentropy()
                metrics = [keras.metrics.BinaryAccuracy()]
            else:
                loss = keras.losses.SparseCategoricalCrossentropy()
                metrics = [keras.metrics.SparseCategoricalAccuracy()]
        else:
            loss = keras.losses.MeanSquaredError()
            metrics = [keras.metrics.MeanAbsoluteError()]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Setup callbacks
        gradient_callback = GradientCallback(model, X_val, y_val)

        callbacks = [
            gradient_callback,
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        # Train model
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # Save model
            model_path = self.output_dir / f"{config_name}_{depth_name}_model.keras"
            model.save(model_path)
            logger.info(f"Saved model to {model_path}")

        except Exception as e:
            logger.error(f"Training failed for {config_name}_{depth_name}: {str(e)}")
            raise

        return history, gradient_callback

    def analyze_models(
        self,
        models: Dict[str, keras.Model],
        test_data: DataInput,
        depth_name: str
    ) -> Dict[str, Any]:
        """Analyze models using ModelAnalyzer."""
        logger.info(f"Running comprehensive model analysis for {depth_name}")

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
            history_path = self.output_dir / f"{config_name}_{depth_name}_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    histories[config_name] = json.load(f)

        # Run analysis
        analysis_dir = self.output_dir / f'analysis_{depth_name}'
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
        summary_path = self.output_dir / f'analysis_summary_{depth_name}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Analysis complete. Results saved to {analysis_dir}")

        return summary

    def run_experiment(self, task_type: str = 'classification') -> Dict[str, Any]:
        """Run the complete experiment."""
        logger.info(f"Starting Deep MLP BandRMS experiment for {task_type}")

        # Create dataset
        X_train, X_test, y_train, y_test = self.create_synthetic_dataset(task_type)

        # Prepare train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42,
            stratify=y_train if task_type == 'classification' else None
        )

        input_dim = X_train.shape[1]
        if task_type == 'classification':
            output_dim = len(np.unique(y_train)) if len(y_train.shape) == 1 else y_train.shape[1]
        else:
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

        # Store all results
        all_results = {}

        # Run experiments for different depth variants
        for depth_variant in self.depth_variants:
            depth_name = depth_variant['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiments for {depth_name} network")
            logger.info(f"Architecture: {depth_variant['hidden_dims']}")
            logger.info(f"{'='*60}")

            results = {}
            models = {}
            gradient_stats = {}
            band_params = {}

            # Train each configuration
            for config_name in self.configs.keys():
                logger.info(f"\nTraining configuration: {config_name}")

                try:
                    # Create model
                    model = self.create_model(
                        config_name, depth_variant, input_dim, output_dim, task_type
                    )
                    models[config_name] = model

                    # Train model
                    history, gradient_callback = self.train_model(
                        model, X_train, y_train, X_val, y_val, config_name, depth_name, task_type
                    )

                    # Save history
                    history_path = self.output_dir / f"{config_name}_{depth_name}_history.json"
                    with open(history_path, 'w') as f:
                        json.dump(history.history, f, indent=2, default=str)

                    # Save gradient statistics
                    gradient_stats[config_name] = {
                        'gradient_norms': gradient_callback.gradient_norms,
                        'activation_means': gradient_callback.activation_means,
                        'activation_stds': gradient_callback.activation_stds
                    }

                    # Get band parameters (for BandRMS)
                    if hasattr(model, 'get_band_parameters'):
                        band_params[config_name] = model.get_band_parameters()

                    # Evaluate on test set
                    test_results = model.evaluate(X_test, y_test, verbose=0)
                    results[config_name] = {
                        'test_loss': float(test_results[0]),
                        'test_metric': float(test_results[1]) if len(test_results) > 1 else None,
                        'best_val_loss': float(min(history.history['val_loss'])),
                        'epochs_trained': len(history.history['loss']),
                        'depth': len(depth_variant['hidden_dims']),
                        'total_parameters': model.count_params(),
                        'converged': len(history.history['loss']) < self.epochs  # Early stopping triggered
                    }

                    logger.info(f"Results for {config_name}:")
                    logger.info(f"  Test Loss: {results[config_name]['test_loss']:.4f}")
                    logger.info(f"  Test Metric: {results[config_name]['test_metric']:.4f}")
                    logger.info(f"  Converged: {results[config_name]['converged']}")
                    logger.info(f"  Parameters: {results[config_name]['total_parameters']:,}")

                except Exception as e:
                    logger.error(f"Failed to train {config_name}: {str(e)}")
                    results[config_name] = {'error': str(e), 'depth': len(depth_variant['hidden_dims'])}

            # Save gradient statistics
            gradient_path = self.output_dir / f'gradient_stats_{depth_name}.json'
            with open(gradient_path, 'w') as f:
                json.dump(gradient_stats, f, indent=2, default=str)

            # Save band parameters
            band_path = self.output_dir / f'band_parameters_{depth_name}.json'
            with open(band_path, 'w') as f:
                json.dump(band_params, f, indent=2, default=str)

            # Prepare test data for analysis
            test_data = DataInput(x_data=X_test, y_data=y_test)

            # Run comprehensive analysis
            analysis_summary = self.analyze_models(models, test_data, depth_name)

            # Create comparison plots for this depth
            self._create_comparison_plots(results, gradient_stats, depth_name, task_type)

            # Store results for this depth
            all_results[depth_name] = {
                'training_results': results,
                'analysis_summary': analysis_summary,
                'gradient_stats': gradient_stats,
                'band_parameters': band_params
            }

        # Create cross-depth comparison
        self._create_depth_comparison(all_results, task_type)

        # Save final results
        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"\nExperiment complete! Results saved to {self.output_dir}")
        return all_results

    def _create_comparison_plots(
        self,
        results: Dict[str, Any],
        gradient_stats: Dict[str, Any],
        depth_name: str,
        task_type: str
    ):
        """Create comparison plots for a specific depth."""

        # Training stability comparison (gradient norms over time)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        colors = plt.cm.Set3(np.linspace(0, 1, len(gradient_stats)))

        for i, (config_name, stats) in enumerate(gradient_stats.items()):
            if not stats['gradient_norms']:
                continue

            epochs = [item['epoch'] for item in stats['gradient_norms']]
            mean_norms = [item['mean_norm'] for item in stats['gradient_norms']]
            max_norms = [item['max_norm'] for item in stats['gradient_norms']]

            color = colors[i]

            # Mean gradient norms
            ax1.plot(epochs, mean_norms, color=color, label=config_name, linewidth=2)
            ax1.set_title(f'Mean Gradient Norms ({depth_name})')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Gradient Norm')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # Max gradient norms
            ax2.plot(epochs, max_norms, color=color, label=config_name, linewidth=2)
            ax2.set_title(f'Max Gradient Norms ({depth_name})')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Gradient Norm')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

            # Activation means
            if stats['activation_means']:
                overall_means = [item['overall_mean'] for item in stats['activation_means']]
                ax3.plot(epochs, overall_means, color=color, label=config_name, linewidth=2)

            # Activation stds
            if stats['activation_stds']:
                overall_stds = [item['overall_std'] for item in stats['activation_stds']]
                ax4.plot(epochs, overall_stds, color=color, label=config_name, linewidth=2)

        ax3.set_title(f'Mean Activation Values ({depth_name})')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Mean Activation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.set_title(f'Activation Standard Deviations ({depth_name})')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Std Activation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_stability_{depth_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Final performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        configs = []
        test_losses = []
        test_metrics = []
        converged = []

        for config_name, result in results.items():
            if 'error' not in result:
                configs.append(config_name)
                test_losses.append(result['test_loss'])
                test_metrics.append(result['test_metric'] if result['test_metric'] else 0)
                converged.append(result['converged'])

        x_pos = np.arange(len(configs))

        # Test loss comparison
        colors_bar = ['green' if c else 'red' for c in converged]
        bars1 = ax1.bar(x_pos, test_losses, color=colors_bar, alpha=0.7)
        ax1.set_title(f'Final Test Loss ({depth_name})')
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

        # Test metric comparison
        metric_name = 'Accuracy' if task_type == 'classification' else 'MAE'
        bars2 = ax2.bar(x_pos, test_metrics, color=colors_bar, alpha=0.7)
        ax2.set_title(f'Final Test {metric_name} ({depth_name})')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel(f'Test {metric_name}')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, test_metrics):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # Add legend for convergence colors
        ax1.text(0.02, 0.98, 'Green: Converged\nRed: Did not converge',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / f'final_performance_{depth_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plots saved for {depth_name}")

    def _create_depth_comparison(self, all_results: Dict[str, Any], task_type: str):
        """Create comparison plots across different network depths."""

        # Extract data for depth comparison
        depths = []
        configs_data = {}

        for depth_name, depth_results in all_results.items():
            depth_layers = len(self.depth_variants[[d['name'] for d in self.depth_variants].index(depth_name)]['hidden_dims'])
            depths.append(depth_layers)

            for config_name, result in depth_results['training_results'].items():
                if 'error' not in result:
                    if config_name not in configs_data:
                        configs_data[config_name] = {'depths': [], 'losses': [], 'metrics': [], 'converged': []}

                    configs_data[config_name]['depths'].append(depth_layers)
                    configs_data[config_name]['losses'].append(result['test_loss'])
                    configs_data[config_name]['metrics'].append(result['test_metric'] or 0)
                    configs_data[config_name]['converged'].append(result['converged'])

        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        colors = plt.cm.Set3(np.linspace(0, 1, len(configs_data)))

        for i, (config_name, data) in enumerate(configs_data.items()):
            color = colors[i]

            # Test loss vs depth
            ax1.plot(data['depths'], data['losses'], 'o-', color=color,
                    label=config_name, linewidth=2, markersize=8)

            # Test metric vs depth
            metric_name = 'Accuracy' if task_type == 'classification' else 'MAE'
            ax2.plot(data['depths'], data['metrics'], 'o-', color=color,
                    label=config_name, linewidth=2, markersize=8)

            # Convergence rate vs depth
            convergence_rates = [sum(data['converged'][:i+1])/(i+1) for i in range(len(data['converged']))]
            ax3.plot(data['depths'], data['converged'], 'o-', color=color,
                    label=config_name, linewidth=2, markersize=8)

        ax1.set_title('Test Loss vs Network Depth')
        ax1.set_xlabel('Number of Hidden Layers')
        ax1.set_ylabel('Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(sorted(set(depths)))

        ax2.set_title(f'Test {metric_name} vs Network Depth')
        ax2.set_xlabel('Number of Hidden Layers')
        ax2.set_ylabel(f'Test {metric_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(sorted(set(depths)))

        ax3.set_title('Convergence vs Network Depth')
        ax3.set_xlabel('Number of Hidden Layers')
        ax3.set_ylabel('Converged (1=Yes, 0=No)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(sorted(set(depths)))
        ax3.set_ylim(-0.1, 1.1)

        # Band parameter utilization (for BandRMS variants only)
        band_configs = [name for name in configs_data.keys() if 'band_rms' in name]
        if band_configs:
            for depth_name, depth_results in all_results.items():
                depth_layers = len(self.depth_variants[[d['name'] for d in self.depth_variants].index(depth_name)]['hidden_dims'])

                for config_name in band_configs:
                    if config_name in depth_results['band_parameters']:
                        band_params = depth_results['band_parameters'][config_name]
                        if band_params:
                            layer_names = list(band_params.keys())
                            band_values = list(band_params.values())

                            ax4.scatter([depth_layers] * len(band_values), band_values,
                                      label=f'{config_name} @ depth {depth_layers}', alpha=0.6)

        ax4.set_title('Learned Band Parameters vs Depth')
        ax4.set_xlabel('Number of Hidden Layers')
        ax4.set_ylabel('Band Parameter Value')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(sorted(set(depths)))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'depth_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Depth comparison plots saved")


def main():
    """Run the Deep MLP BandRMS experiment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    # Create and run experiment
    experiment = MLPExperiment(output_dir='experiments/mlp_bandrms')
    results = experiment.run_experiment(task_type='classification')

    # Print summary
    logger.info("="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)

    for depth_name, depth_results in results.items():
        logger.info(f"{depth_name.upper()} NETWORK:")
        logger.info("-" * 40)

        for config_name, result in depth_results['training_results'].items():
            if 'error' not in result:
                logger.info(f"{config_name}:")
                logger.info(f"  Test Loss: {result['test_loss']:.4f}")
                logger.info(f"  Test Accuracy: {result['test_metric']:.4f}")
                logger.info(f"  Converged: {result['converged']}")
                logger.info(f"  Parameters: {result['total_parameters']:,}")
                logger.info(f"  Depth: {result['depth']} layers")
            else:
                logger.info(f"{config_name}: FAILED - {result['error']}")

        # Band parameter summary
        if 'band_parameters' in depth_results:
            logger.info("Band Parameter Summary:")
            for config_name, params in depth_results['band_parameters'].items():
                if params:
                    mean_param = np.mean(list(params.values()))
                    logger.info(f"  {config_name}: Mean band param = {mean_param:.4f}")


if __name__ == "__main__":
    main()