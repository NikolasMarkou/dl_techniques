"""
Colored MNIST Spurious Correlation Experiment: GoodhartAwareLoss vs Label Smoothing
==================================================================================

This module implements a comprehensive experiment to evaluate the robustness of different
loss functions against spurious correlations. It uses a Colored MNIST dataset where models
can "game" the training metric by exploiting color instead of learning digit shapes.

The experiment follows the project's standard analysis framework, using modular
components for data generation, training, and analysis.
"""

# ------------------------------------------------------------------------------
# 1. Imports and Dependencies
# ------------------------------------------------------------------------------

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable

import keras
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.goodhart_loss import GoodhartAwareLoss
from dl_techniques.utils.train import TrainingConfig, train_model
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig

# ------------------------------------------------------------------------------
# 2. Experiment Configuration Class
# ------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Unified configuration for the Colored MNIST spurious correlation experiment."""
    # Dataset Configuration
    dataset_name: str = "colored_mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 3)

    # Spurious Correlation Parameters
    train_correlation_strength: float = 0.95
    test_correlation_strength: float = 0.0

    # Model Architecture Parameters
    conv_filters: List[int] = (32, 64)
    dense_units: int = 128
    dropout_rate: float = 0.4
    weight_decay: float = 1e-4
    use_batch_norm: bool = True

    # Training Parameters
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    monitor_metric: str = 'val_accuracy'

    # Loss Functions to Test (all configured for logits)
    loss_functions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'crossentropy': {
            'name': 'Cross-Entropy',
            'loss_fn': lambda: keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        'label_smoothing': {
            'name': 'Label Smoothing',
            'loss_fn': lambda: keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=True)
        },
        'goodhart_aware': {
            'name': 'GoodhartAwareLoss',
            'loss_fn': lambda: GoodhartAwareLoss(entropy_weight=0.15, mi_weight=0.02)
        }
    })

    # Experiment Parameters
    output_dir: Path = Path("results")
    experiment_name: str = "colored_mnist_spurious_correlation"
    random_seed: int = 42

# ------------------------------------------------------------------------------
# 3. Colored MNIST Dataset Generation
# ------------------------------------------------------------------------------
def get_color_palette() -> List[Tuple[int, int, int]]:
    """Defines the color for each digit class."""
    return [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]

def create_colored_mnist_dataset(config: ExperimentConfig) -> Tuple[np.ndarray, ...]:
    """Generates the Colored MNIST dataset with specified correlations."""
    logger.info("🎨 Generating Colored MNIST dataset...")
    np.random.seed(config.random_seed)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    colors = get_color_palette()

    def colorize(images, labels, correlation):
        normalized_images = np.stack([images] * 3, axis=-1) / 255.0
        colored_images = np.zeros_like(normalized_images)
        for i, label in enumerate(labels):
            color_idx = label if np.random.rand() < correlation else np.random.randint(config.num_classes)
            color = np.array(colors[color_idx]) / 255.0
            colored_images[i] = normalized_images[i] * color
        return colored_images

    x_train_col = colorize(x_train, y_train, config.train_correlation_strength)
    x_test_col = colorize(x_test, y_test, config.test_correlation_strength)
    y_train_cat = keras.utils.to_categorical(y_train, config.num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, config.num_classes)
    
    logger.info(f"✅ Dataset generated. Train correlation: {config.train_correlation_strength:.0%}, Test correlation: {config.test_correlation_strength:.0%}")
    return x_train_col, y_train_cat, x_test_col, y_test_cat

# ------------------------------------------------------------------------------
# 4. Model Building Utility
# ------------------------------------------------------------------------------
def build_model(config: ExperimentConfig, loss_fn: Callable, name: str) -> keras.Model:
    """Builds a standard CNN model that outputs logits."""
    inputs = keras.layers.Input(shape=config.input_shape, name=f'{name}_input')
    x = inputs
    for filters in config.conv_filters:
        x = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(config.weight_decay))(x)
        if config.use_batch_norm: x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        if config.dropout_rate > 0: x = keras.layers.Dropout(config.dropout_rate)(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(config.dense_units, activation='relu', kernel_regularizer=keras.regularizers.l2(config.weight_decay))(x)
    if config.dropout_rate > 0: x = keras.layers.Dropout(config.dropout_rate)(x)
    
    logits = keras.layers.Dense(config.num_classes, name='logits')(x)
    model = keras.Model(inputs=inputs, outputs=logits, name=name)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=loss_fn,
        metrics=['accuracy']
    )
    return model

# ------------------------------------------------------------------------------
# 5. Robustness Analyzer
# ------------------------------------------------------------------------------
class RobustnessAnalyzer:
    """A class to handle robustness analysis and visualization."""
    def __init__(self, models: Dict[str, keras.Model], vis_manager: VisualizationManager):
        self.models = models
        self.vis_manager = vis_manager
        self.results = {}

    def analyze(self, x_train, y_train, x_test, y_test) -> Dict[str, Any]:
        """Performs a full robustness analysis on all models."""
        logger.info("🛡️  Starting robustness analysis...")
        for name, model in self.models.items():
            logger.info(f"  -> Analyzing model: {name}")
            train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            
            self.results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'generalization_gap': train_acc - test_acc,
            }
        logger.info("✅ Robustness analysis complete.")
        return self.results

    def plot_comparison(self, output_dir: Path, model_names_map: Dict[str, str]):
        """Plots a comparison of key robustness metrics."""
        metric_names = ['Test Accuracy', 'Generalization Gap']
        model_keys = list(self.results.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        fig.suptitle('Spurious Correlation Robustness Comparison', fontsize=16, weight='bold')

        for i, metric in enumerate(['test_accuracy', 'generalization_gap']):
            ax = axes[i]
            values = [self.results[key][metric] for key in model_keys]
            display_names = [model_names_map[key] for key in model_keys]
            
            bars = ax.bar(display_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title(metric_names[i], fontsize=14)
            ax.set_ylabel('Value', fontsize=12)
            ax.tick_params(axis='x', rotation=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = output_dir / "robustness_summary.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"✅ Robustness comparison plot saved to {save_path}")

# ------------------------------------------------------------------------------
# 6. Experiment Runner
# ------------------------------------------------------------------------------

def run_spurious_correlation_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Runs the complete spurious correlation experiment in the style of the calibration experiment."""
    keras.utils.set_random_seed(config.random_seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting Experiment: {config.experiment_name}")
    logger.info(f"📁 Results will be saved to: {exp_dir}")

    # --- Setup ---
    vis_manager = VisualizationManager(output_dir=exp_dir / "visualizations", config=VisualizationConfig())
    x_train, y_train, x_test, y_test = create_colored_mnist_dataset(config)

    # --- Training Phase ---
    trained_models = {}
    all_histories = {}
    for key, loss_config in config.loss_functions.items():
        logger.info("-" * 80)
        logger.info(f"🧠 Training model: {loss_config['name']} (key: {key})")
        
        model = build_model(config, loss_config['loss_fn'](), name=key)
        logger.info(f"  -> Model built with {model.count_params():,} parameters.")
        
        training_config = TrainingConfig(
            epochs=config.epochs, batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            monitor_metric=config.monitor_metric, model_name=key,
            output_dir=exp_dir / key
        )
        
        history = train_model(model, x_train, y_train, x_test, y_test, training_config)
        trained_models[key] = model
        all_histories[key] = history.history

    # --- Analysis Phase ---
    logger.info("-" * 80)
    logger.info("🔬 Starting Analysis Phase...")

    # Create prediction models with softmax for analysis if needed
    prediction_models = {}
    for name, logit_model in trained_models.items():
        prediction_model = keras.Model(inputs=logit_model.input, outputs=keras.layers.Activation('softmax')(logit_model.output))
        prediction_model.compile(metrics=['accuracy']) # Compile for .evaluate()
        prediction_models[name] = prediction_model

    # Robustness Analysis
    robustness_analyzer = RobustnessAnalyzer(prediction_models, vis_manager)
    robustness_results = robustness_analyzer.analyze(x_train, y_train, x_test, y_test)
    robustness_analyzer.plot_comparison(vis_manager.output_dir, {k: v['name'] for k, v in config.loss_functions.items()})

    # --- Final Results Aggregation ---
    results = {
        'config': config,
        'histories': all_histories,
        'robustness_metrics': robustness_results,
        'exp_dir': exp_dir,
    }
    
    print_summary(results)
    return results

# ------------------------------------------------------------------------------
# 7. Main Execution
# ------------------------------------------------------------------------------

def print_summary(results: Dict[str, Any]):
    """Prints a final, clear summary of the experiment."""
    logger.info("=" * 80)
    logger.info("📋 EXPERIMENT SUMMARY & VERDICT")
    logger.info("=" * 80)
    
    metrics = results.get('robustness_metrics', {})
    config = results.get('config')
    model_names_map = {k: v['name'] for k, v in config.loss_functions.items()}
    
    header = f"{'Metric':<22}" + "".join([f"{model_names_map[k]:<20}" for k in model_names_map])
    logger.info(header)
    logger.info("-" * len(header))
    
    for metric_key in ['test_accuracy', 'generalization_gap']:
        row = f"{metric_key.replace('_', ' ').title():<22}"
        for model_key in model_names_map:
            value = metrics.get(model_key, {}).get(metric_key, float('nan'))
            row += f"{value:<20.4f}"
        logger.info(row)
    logger.info("-" * len(header))
    
    # Verdict Logic
    ls_metrics = metrics.get('label_smoothing')
    gal_metrics = metrics.get('goodhart_aware')
    if ls_metrics and gal_metrics:
        test_acc_diff = gal_metrics['test_accuracy'] - ls_metrics['test_accuracy']
        gap_diff = gal_metrics['generalization_gap'] - ls_metrics['generalization_gap']
        
        logger.info("🔍 VERDICT (GoodhartAwareLoss vs. Label Smoothing):")
        if test_acc_diff > 0.05 and gap_diff < -0.05:
            verdict = "🎉 STRONG SUPPORT: GAL is significantly more robust and generalizes better."
        elif test_acc_diff > 0.02:
            verdict = "✅ POSITIVE: GAL shows a clear improvement in robustness."
        elif test_acc_diff > -0.02:
            verdict = "➖ NEUTRAL: GAL performed similarly to Label Smoothing."
        else:
            verdict = "⚠️  NEGATIVE: Label Smoothing was more robust in this configuration."
        logger.info(f"  -> {verdict}")
    
    logger.info("=" * 80)

# ------------------------------------------------------------------------------

def main():
    """Main execution function."""
    config = ExperimentConfig()
    logger.info("🚀 Colored MNIST Spurious Correlation Experiment")
    logger.info("=" * 80)
    logger.info("⚙️  EXPERIMENT CONFIGURATION:")
    for key, value in config.__dict__.items():
        if key != 'loss_functions':
            logger.info(f"   {key}: {value}")
    logger.info(f"   Loss Functions: {[v['name'] for v in config.loss_functions.values()]}")
    logger.info("=" * 80)
    
    try:
        results = run_spurious_correlation_experiment(config)
        logger.info("✅ Experiment completed successfully!")
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}", exc_info=True)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
