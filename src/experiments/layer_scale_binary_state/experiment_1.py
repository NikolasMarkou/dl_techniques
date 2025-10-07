"""
Complete LayerScale and LearnableMultiplier Feature Selection Experiment
=======================================================================

This experiment demonstrates how the LearnableMultiplier layer with BinaryPreferenceRegularizer
can act as effective feature selectors on a synthetic dataset with many irrelevant features.

The experiment uses the dl_techniques visualization framework for all plots.

Note: This code assumes you have already imported the following custom layers:
- LearnableMultiplier
- BinaryPreferenceRegularizer
"""

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.layers.layer_scale import LearnableMultiplier
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    ModelComparison,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# Custom Data Structures for Feature Selection Visualization
# ============================================================================

@dataclass
class FeatureWeightsData:
    """
    Data structure for feature weights visualization.

    Attributes:
        weights: Array of feature weights
        feature_names: Names/labels for each feature
        important_features: Indices of ground truth important features
        model_name: Name of the model
    """
    weights: np.ndarray
    feature_names: List[str]
    important_features: List[int]
    model_name: str


@dataclass
class DecisionBoundaryData:
    """
    Data structure for decision boundary visualization.

    Attributes:
        X: Feature matrix
        y: Class labels
        feature_indices: Tuple of (feature1_idx, feature2_idx) to visualize
        model: Trained keras model for boundary prediction
        centers: Optional class centers in the feature space
        title: Plot title
    """
    X: np.ndarray
    y: np.ndarray
    feature_indices: Tuple[int, int]
    model: Optional[keras.Model]
    centers: Optional[np.ndarray]
    title: str


# ============================================================================
# Custom Visualization Plugins
# ============================================================================

class FeatureWeightsVisualization(VisualizationPlugin):
    """Visualization plugin for feature weights with importance highlighting."""

    @property
    def name(self) -> str:
        return "feature_weights"

    @property
    def description(self) -> str:
        return "Visualizes learned feature weights with importance highlighting"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (FeatureWeightsData, Dict))

    def create_visualization(
        self,
        data: Any,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Create feature weights bar chart."""
        if isinstance(data, dict):
            # Handle multiple models
            fig, axes = plt.subplots(
                len(data), 1,
                figsize=(self.config.fig_size[0], self.config.fig_size[1] * len(data)),
                dpi=self.config.dpi
            )
            if len(data) == 1:
                axes = [axes]

            for idx, (model_name, weights_data) in enumerate(data.items()):
                self._plot_single_weights(weights_data, axes[idx])

            plt.tight_layout()
            return fig
        else:
            # Handle single model
            if ax is None:
                fig, ax = plt.subplots(figsize=self.config.fig_size, dpi=self.config.dpi)
            else:
                fig = ax.get_figure()

            self._plot_single_weights(data, ax)
            return fig

    def _plot_single_weights(
        self,
        data: FeatureWeightsData,
        ax: plt.Axes
    ) -> None:
        """Plot weights for a single model."""
        weights = data.weights
        feature_names = data.feature_names
        important_features = data.important_features

        # Color bars based on importance
        colors = [
            self.config.color_scheme.primary if i in important_features
            else self.config.color_scheme.secondary
            for i in range(len(weights))
        ]

        bars = ax.bar(feature_names, weights, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(
            f'{data.model_name} Feature Weights',
            fontsize=self.config.title_fontsize
        )
        ax.set_ylabel('Weight Value', fontsize=self.config.label_fontsize)
        ax.tick_params(axis='x', rotation=90)
        ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.7, linewidth=1)
        ax.grid(alpha=0.3, axis='y')

        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=8
            )


class FeatureWeightDistributionVisualization(VisualizationPlugin):
    """Visualization plugin for feature weight distributions."""

    @property
    def name(self) -> str:
        return "feature_weight_distribution"

    @property
    def description(self) -> str:
        return "Visualizes the distribution of feature weights"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (FeatureWeightsData, Dict))

    def create_visualization(
        self,
        data: Any,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Create weight distribution histogram."""
        if isinstance(data, dict):
            # Handle multiple models
            fig, axes = plt.subplots(
                1, len(data),
                figsize=(self.config.fig_size[0] * len(data) / 2, self.config.fig_size[1]),
                dpi=self.config.dpi
            )
            if len(data) == 1:
                axes = [axes]

            for idx, (model_name, weights_data) in enumerate(data.items()):
                self._plot_single_distribution(weights_data, axes[idx])

            plt.tight_layout()
            return fig
        else:
            # Handle single model
            if ax is None:
                fig, ax = plt.subplots(figsize=self.config.fig_size, dpi=self.config.dpi)
            else:
                fig = ax.get_figure()

            self._plot_single_distribution(data, ax)
            return fig

    def _plot_single_distribution(
        self,
        data: FeatureWeightsData,
        ax: plt.Axes
    ) -> None:
        """Plot weight distribution for a single model."""
        weights = data.weights

        ax.hist(
            weights, bins=20,
            color=self.config.color_scheme.primary,
            alpha=0.7, edgecolor='black'
        )
        ax.set_title(
            f'{data.model_name} Weight Distribution',
            fontsize=self.config.title_fontsize
        )
        ax.set_xlabel('Weight Value', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Count', fontsize=self.config.label_fontsize)
        ax.axvline(x=0.5, linestyle='--', color='red', linewidth=2, label='Binary threshold')
        ax.grid(alpha=0.3)
        ax.legend()


class DecisionBoundaryVisualization(VisualizationPlugin):
    """Visualization plugin for decision boundaries in feature space."""

    @property
    def name(self) -> str:
        return "decision_boundary"

    @property
    def description(self) -> str:
        return "Visualizes decision boundaries in 2D feature space"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (DecisionBoundaryData, List))

    def create_visualization(
        self,
        data: Any,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        """Create decision boundary visualization."""
        if isinstance(data, list):
            # Handle multiple boundaries
            n_plots = len(data)
            fig, axes = plt.subplots(
                1, n_plots,
                figsize=(self.config.fig_size[0] * n_plots / 2, self.config.fig_size[1]),
                dpi=self.config.dpi
            )
            if n_plots == 1:
                axes = [axes]

            for idx, boundary_data in enumerate(data):
                self._plot_single_boundary(boundary_data, axes[idx])

            plt.tight_layout()
            return fig
        else:
            # Handle single boundary
            if ax is None:
                fig, ax = plt.subplots(figsize=self.config.fig_size, dpi=self.config.dpi)
            else:
                fig = ax.get_figure()

            self._plot_single_boundary(data, ax)
            return fig

    def _plot_single_boundary(
        self,
        data: DecisionBoundaryData,
        ax: plt.Axes
    ) -> None:
        """Plot a single decision boundary."""
        X = data.X
        y = data.y
        feature_indices = data.feature_indices
        model = data.model
        centers = data.centers

        # Extract the two features to visualize
        X_reduced = X[:, feature_indices]

        # Create meshgrid for decision boundary
        h = 0.1
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot decision boundaries if model is provided
        if model is not None:
            grid = np.c_[xx.ravel(), yy.ravel()]

            # Pad with zeros for other features
            if X.shape[1] > 2:
                grid_full = np.zeros((grid.shape[0], X.shape[1]))
                grid_full[:, feature_indices[0]] = grid[:, 0]
                grid_full[:, feature_indices[1]] = grid[:, 1]
            else:
                grid_full = grid

            # Get predictions
            Z = model.predict(grid_full, verbose=0)
            Z = np.argmax(Z, axis=1)
            Z = Z.reshape(xx.shape)

            # Plot decision boundaries
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        # Plot class centers if provided
        if centers is not None and feature_indices[0] < 2 and feature_indices[1] < 2:
            ax.scatter(
                centers[:, 0], centers[:, 1],
                marker='X', s=200, c='red',
                edgecolors='white', linewidths=2,
                label='Class Centers', zorder=5
            )

        # Plot data points
        scatter = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1],
            c=y, cmap='viridis', s=30, alpha=0.8,
            edgecolors='k', linewidths=0.5
        )

        ax.set_title(data.title, fontsize=self.config.title_fontsize)
        ax.set_xlabel(f'Feature {feature_indices[0]+1}', fontsize=self.config.label_fontsize)
        ax.set_ylabel(f'Feature {feature_indices[1]+1}', fontsize=self.config.label_fontsize)
        ax.grid(alpha=0.3)

        # Add colorbar for classes
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class', fontsize=self.config.label_fontsize)


# ============================================================================
# Data Generation
# ============================================================================

def generate_synthetic_data(
    n_samples: int = 1000,
    n_noise_features: int = 20,
    noise_level: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset with important and noise features.

    Args:
        n_samples: Number of samples to generate
        n_noise_features: Number of irrelevant noise features to add
        noise_level: Standard deviation of noise

    Returns:
        Tuple of (X, y, centers) with features, labels, and class centers
    """
    # Generate two informative features
    X_informative = np.random.randn(n_samples, 2)

    # Generate 4 clusters (classes) in these two features
    centers = np.array([
        [-2, -2],  # Class 0
        [-2, 2],   # Class 1
        [2, -2],   # Class 2
        [2, 2]     # Class 3
    ])

    # Assign each point to nearest center
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        dists = np.sum((X_informative[i] - centers) ** 2, axis=1)
        y[i] = np.argmin(dists)

    # Move points towards their assigned centers
    for i in range(n_samples):
        X_informative[i] = 0.8 * centers[y[i]] + 0.2 * X_informative[i]

    # Add Gaussian noise
    X_informative += noise_level * np.random.randn(n_samples, 2)

    # Generate noise features with no predictive power
    X_noise = np.random.randn(n_samples, n_noise_features)

    # Combine informative and noise features
    X = np.hstack([X_informative, X_noise])

    return X, y, centers


# ============================================================================
# Model Creation
# ============================================================================

def create_model_with_learnable_multiplier(
    input_dim: int,
    num_classes: int
) -> keras.Model:
    """
    Create a model with LearnableMultiplier layer to identify important features.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    # Add the LearnableMultiplier layer for feature selection
    x = LearnableMultiplier(
        multiplier_type="CHANNEL",
        initializer=keras.initializers.Constant(0.5),
        regularizer=BinaryPreferenceRegularizer(multiplier=1.5),
        constraint=None
    )(inputs)

    # Add dense layers
    x = keras.layers.Dense(256, activation='relu', use_bias=False)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu', use_bias=False)(x)
    x = keras.layers.Dense(
        num_classes, activation='softmax',
        kernel_regularizer=SoftOrthonormalConstraintRegularizer(),
        use_bias=False
    )(x)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_model_with_layer_scale(
    input_dim: int,
    num_classes: int
) -> keras.Model:
    """
    Create a model with LearnableMultiplier (without binary preference) layer for comparison.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    # Add the LayerScale layer
    x = LearnableMultiplier(
        multiplier_type="CHANNEL",
        initializer=keras.initializers.Constant(0.5),
        regularizer=None,
        constraint=None
    )(inputs)

    # Add dense layers (same as the other model)
    x = keras.layers.Dense(256, activation='relu', use_bias=False)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu', use_bias=False)(x)
    x = keras.layers.Dense(
        num_classes, activation='softmax',
        kernel_regularizer=SoftOrthonormalConstraintRegularizer(),
        use_bias=False
    )(x)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment() -> Dict[str, Any]:
    """
    Run the full experiment and return results.

    Returns:
        Dictionary with experiment results
    """
    # Initialize visualization manager
    viz_config = PlotConfig(
        style=PlotStyle.SCIENTIFIC,
        color_scheme=ColorScheme(
            primary='#2E86AB',
            secondary='#A23B72',
            accent='#F18F01'
        ),
        title_fontsize=14,
        save_format='png'
    )

    viz_manager = VisualizationManager(
        experiment_name="feature_selection_experiment",
        output_dir="visualizations_output",
        config=viz_config
    )

    # Register visualization plugins
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("performance_radar", PerformanceRadarChart)
    viz_manager.register_template("feature_weights", FeatureWeightsVisualization)
    viz_manager.register_template("feature_weight_distribution", FeatureWeightDistributionVisualization)
    viz_manager.register_template("decision_boundary", DecisionBoundaryVisualization)

    # Generate dataset with 2 important features and 20 noise features
    logger.info("Generating synthetic dataset...")
    X, y, centers = generate_synthetic_data(n_samples=10000, n_noise_features=20)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    logger.info(f"Dataset: {X.shape[0]} samples, {n_features} features ({n_features - 20} important, 20 noise)")
    logger.info(f"Classes: {n_classes}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model with LearnableMultiplier
    logger.info("\nTraining model with LearnableMultiplier...")
    model_lm = create_model_with_learnable_multiplier(n_features, n_classes)

    history_lm = model_lm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Train model with LayerScale
    logger.info("\nTraining model with LearnableMultiplier (no binary preference)...")
    model_ls = create_model_with_layer_scale(n_features, n_classes)

    history_ls = model_ls.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate both models
    logger.info("\nEvaluating models...")
    lm_eval = model_lm.evaluate(X_test, y_test, verbose=0)
    ls_eval = model_ls.evaluate(X_test, y_test, verbose=0)

    logger.info(f"LearnableMultiplier Test Accuracy: {lm_eval[1]:.4f}")
    logger.info(f"LearnableMultiplier (no binary preference) Test Accuracy: {ls_eval[1]:.4f}")

    # Extract feature weights
    lm_layer = model_lm.layers[1]
    lm_weights = lm_layer.get_weights()[0].flatten()

    ls_layer = model_ls.layers[1]
    ls_weights = ls_layer.get_weights()[0].flatten()

    # Create data structures for visualization
    feature_names = [f'Important {i + 1}' if i < 2 else f'Noise {i - 1}' for i in range(n_features)]

    # Training history
    history_data = {
        "LearnableMultiplier": TrainingHistory(
            epochs=list(range(len(history_lm.history['loss']))),
            train_loss=history_lm.history['loss'],
            val_loss=history_lm.history['val_loss'],
            train_metrics={'accuracy': history_lm.history['accuracy']},
            val_metrics={'accuracy': history_lm.history['val_accuracy']}
        ),
        "LearnableMultiplier (no binary pref)": TrainingHistory(
            epochs=list(range(len(history_ls.history['loss']))),
            train_loss=history_ls.history['loss'],
            val_loss=history_ls.history['val_loss'],
            train_metrics={'accuracy': history_ls.history['accuracy']},
            val_metrics={'accuracy': history_ls.history['val_accuracy']}
        )
    }

    # Model comparison
    comparison_data = ModelComparison(
        model_names=["LearnableMultiplier", "LearnableMultiplier (no binary pref)"],
        metrics={
            "LearnableMultiplier": {"accuracy": lm_eval[1]},
            "LearnableMultiplier (no binary pref)": {"accuracy": ls_eval[1]}
        }
    )

    # Feature weights
    lm_weights_data = FeatureWeightsData(
        weights=lm_weights,
        feature_names=feature_names,
        important_features=[0, 1],
        model_name="LearnableMultiplier"
    )

    ls_weights_data = FeatureWeightsData(
        weights=ls_weights,
        feature_names=feature_names,
        important_features=[0, 1],
        model_name="LearnableMultiplier (no binary pref)"
    )

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    # 1. Training curves
    viz_manager.visualize(
        data=history_data,
        plugin_name="training_curves",
        show=False
    )

    # 2. Model comparison
    viz_manager.visualize(
        data=comparison_data,
        plugin_name="model_comparison_bars",
        show=False
    )

    # 3. Feature weights comparison
    viz_manager.visualize(
        data={
            "LearnableMultiplier": lm_weights_data,
            "LearnableMultiplier (no binary pref)": ls_weights_data
        },
        plugin_name="feature_weights",
        show=False
    )

    # 4. Weight distributions
    viz_manager.visualize(
        data={
            "LearnableMultiplier": lm_weights_data,
            "LearnableMultiplier (no binary pref)": ls_weights_data
        },
        plugin_name="feature_weight_distribution",
        show=False
    )

    # 5. Decision boundaries
    X_viz, y_viz, centers_viz = generate_synthetic_data(n_samples=500, n_noise_features=20)

    lm_top_features = tuple(np.argsort(lm_weights)[-2:])
    ls_top_features = tuple(np.argsort(ls_weights)[-2:])

    boundary_data = [
        DecisionBoundaryData(
            X=X_viz, y=y_viz, feature_indices=(0, 1),
            model=None, centers=centers_viz,
            title="True Feature Space (Features 1 & 2)"
        ),
        DecisionBoundaryData(
            X=X_viz, y=y_viz, feature_indices=lm_top_features,
            model=model_lm, centers=None,
            title=f"LearnableMultiplier Selected\n(Features {lm_top_features[0]+1} & {lm_top_features[1]+1})"
        ),
        DecisionBoundaryData(
            X=X_viz, y=y_viz, feature_indices=ls_top_features,
            model=model_ls, centers=None,
            title=f"LearnableMultiplier (no binary pref) Selected\n(Features {ls_top_features[0]+1} & {ls_top_features[1]+1})"
        )
    ]

    viz_manager.visualize(
        data=boundary_data,
        plugin_name="decision_boundary",
        show=False
    )

    # Print feature importance statistics
    important_features = list(range(2))
    lm_important_avg = np.mean(lm_weights[important_features])
    lm_noise_avg = np.mean(lm_weights[2:])
    ls_important_avg = np.mean(ls_weights[important_features])
    ls_noise_avg = np.mean(ls_weights[2:])

    logger.info("\nFeature Importance Analysis:")
    logger.info(f"LearnableMultiplier - Important features avg weight: {lm_important_avg:.4f}")
    logger.info(f"LearnableMultiplier - Noise features avg weight: {lm_noise_avg:.4f}")
    logger.info(f"LearnableMultiplier - Important/Noise ratio: {lm_important_avg / (lm_noise_avg + 1e-10):.4f}")
    logger.info(f"LearnableMultiplier (no binary pref) - Important features avg weight: {ls_important_avg:.4f}")
    logger.info(f"LearnableMultiplier (no binary pref) - Noise features avg weight: {ls_noise_avg:.4f}")
    logger.info(f"LearnableMultiplier (no binary pref) - Important/Noise ratio: {ls_important_avg / (ls_noise_avg + 1e-10):.4f}")

    # Feature selection success analysis
    lm_success = set(lm_top_features) == {0, 1}
    ls_success = set(ls_top_features) == {0, 1}

    logger.info("\nFeature Selection Success:")
    logger.info(f"LearnableMultiplier: {'✓' if lm_success else '✗'} (Selected features: {lm_top_features})")
    logger.info(f"LearnableMultiplier (no binary pref): {'✓' if ls_success else '✗'} (Selected features: {ls_top_features})")

    return {
        "models": {
            "lm": model_lm,
            "ls": model_ls
        },
        "weights": {
            "lm": lm_weights,
            "ls": ls_weights
        },
        "accuracy": {
            "lm": lm_eval[1],
            "ls": ls_eval[1]
        },
        "history": {
            "lm": history_lm.history,
            "ls": history_ls.history
        },
        "viz_manager": viz_manager
    }


def main():
    """Execute the experiment."""
    logger.info("Starting LearnableMultiplier Feature Selection Experiment...")
    logger.info("=" * 80)

    # Run the experiment
    results = run_experiment()

    logger.info("=" * 80)
    logger.info("Experiment completed. Results saved in 'visualizations_output' directory.")

    # Print final summary
    logger.info("\nFinal Performance Summary:")
    logger.info(f"LearnableMultiplier Accuracy: {results['accuracy']['lm']:.4f}")
    logger.info(f"LearnableMultiplier (no binary pref) Accuracy: {results['accuracy']['ls']:.4f}")


if __name__ == "__main__":
    main()