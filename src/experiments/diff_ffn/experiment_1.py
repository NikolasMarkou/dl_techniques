# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import gc
import json
import keras
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Callable
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN
from dl_techniques.utils.convert import convert_numpy_to_python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.utils.visualization_manager import VisualizationManager, VisualizationConfig


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================


@dataclass
class ExperimentConfig:
    """
    Configuration for the DifferentialFFN architecture evaluation experiment.

    This class encapsulates all configurable parameters for comprehensive
    evaluation of the DifferentialFFN layer against baseline architectures,
    including dataset generation, model architecture, training parameters,
    and analysis configuration.
    """

    # --- Dataset Generation Parameters ---
    n_samples: int = 50000  # Total number of samples to generate
    n_features: int = 30  # Total number of input features
    n_informative: int = 12  # Number of informative features (rest are noise)
    n_classes: int = 6  # Number of target classes
    # Factor controlling class separation (higher = easier)
    class_sep: float = 1.5
    noise_level: float = 0.1  # Gaussian noise intensity added to features
    # Strength of positive/negative interactions
    interaction_strength: float = 0.2
    # Random seed for dataset generation reproducibility
    random_state: int = 42

    # --- Model Architecture Parameters ---
    hidden_dim: int = 256  # Hidden layer dimension for all models
    # Number of output classes (derived from n_classes)
    output_classes: int = 6
    dropout_rate: float = 0.25  # Dropout rate for regularization
    use_batch_norm: bool = True  # Enable batch normalization
    l2_reg: float = 1e-4  # L2 regularization strength

    # Activation functions to test with DifferentialFFN
    differential_activations: List[str] = field(default_factory=lambda: [
        'gelu', 'elu', 'relu'
    ])

    # --- Training Configuration ---
    epochs: int = 100  # Maximum training epochs
    batch_size: int = 128  # Training batch size
    learning_rate: float = 0.001  # Adam optimizer learning rate
    early_stopping_patience: int = 15  # Early stopping patience
    # Metric to monitor for early stopping
    monitor_metric: str = 'val_accuracy'
    validation_split: float = 0.2  # Fraction of data for validation

    # Cross-validation parameters
    cv_folds: int = 10  # Number of cross-validation folds
    cv_scoring: str = 'accuracy'  # Scoring metric for cross-validation

    # --- Model Architectures to Compare ---
    # This will be populated programmatically based on differential_activations
    model_builders: Dict[str, Callable] = field(default_factory=dict)

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,  # Analyze weight distribution patterns
        analyze_calibration=True,  # Analyze prediction calibration
        analyze_information_flow=True,  # Analyze activation patterns
        # Skip training dynamics (handled separately)
        analyze_training_dynamics=False,
        calibration_bins=15,  # Bins for calibration analysis
        save_plots=True,  # Save analysis visualizations
        plot_style='publication',  # Professional plot styling
        dpi=300  # High-resolution plots
    ))

    # --- Experiment Output Configuration ---
    output_dir: Path = Path("results")  # Base output directory
    experiment_name: str = "differential_ffn_evaluation"  # Experiment identifier

    # Visualization and analysis options
    analyze_feature_importance: bool = True  # Analyze feature importance
    # Generate decision boundary plots
    visualize_decision_boundaries: bool = True
    # PCA visualization of features
    generate_feature_representations: bool = True
    statistical_testing: bool = True  # Perform statistical significance tests

    def __post_init__(self):
        """Post-initialization setup for derived parameters."""
        self.output_classes = self.n_classes

        # Programmatically build the model builders dictionary
        self.model_builders = {}

        # Add DifferentialFFN variants
        for activation in self.differential_activations:
            self.model_builders[f'DifferentialFFN_{activation.upper()}'] = (
                lambda act=activation: self._build_differential_ffn_model(act)
            )

        # Add baseline models
        self.model_builders.update({
            'Dense_Baseline': self._build_dense_baseline_model,
            'Manual_DualPath': self._build_manual_dual_pathway_model,
        })

    def _build_differential_ffn_model(self, activation: str) -> keras.Model:
        """Build a DifferentialFFN model with specified activation."""
        return build_differential_ffn_model(
            input_dim=self.n_features,
            num_classes=self.output_classes,
            hidden_dim=self.hidden_dim,
            activation=activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            use_batch_norm=self.use_batch_norm
        )

    def _build_dense_baseline_model(self) -> keras.Model:
        """Build dense baseline model."""
        return build_dense_baseline_model(
            input_dim=self.n_features,
            num_classes=self.output_classes,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            use_batch_norm=self.use_batch_norm
        )

    def _build_manual_dual_pathway_model(self) -> keras.Model:
        """Build manual dual pathway model."""
        return build_manual_dual_pathway_model(
            input_dim=self.n_features,
            num_classes=self.output_classes,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            use_batch_norm=self.use_batch_norm
        )


# ==============================================================================
# SYNTHETIC DATASET GENERATION
# ==============================================================================

def generate_differential_dataset(
        config: ExperimentConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset optimized for evaluating DifferentialFFN.

    This function creates a dataset where features have both positive and negative
    correlations with target classes, along with interaction effects that should
    benefit from dual-pathway processing.

    Args:
        config: Experiment configuration containing dataset parameters

    Returns:
        Tuple of (X, y) with features and class labels
    """
    np.random.seed(config.random_state)

    # Generate base feature matrix
    X = np.random.randn(config.n_samples, config.n_features)

    # Create class centers with deliberate positive/negative structure
    centers = np.random.randn(
        config.n_classes, config.n_informative) * config.class_sep

    # Generate class assignments
    y = np.random.randint(0, config.n_classes, size=config.n_samples)

    # Split informative features into positive and negative correlation groups
    pos_features = config.n_informative // 2
    neg_features = config.n_informative - pos_features

    logger.info(
        f"Generating dataset with {pos_features} positive and "
        f"{neg_features} negative features"
    )

    # Apply class-dependent transformations to create differential patterns
    for i in range(config.n_samples):
        class_idx = y[i]

        # Positive correlation features (first half of informative features)
        X[i, :pos_features] += centers[class_idx, :pos_features]

        # Negative correlation features (second half of informative features)
        X[i, pos_features:config.n_informative] -= \
            centers[class_idx, pos_features:]

        # Add interaction effects between positive and negative features
        for p in range(pos_features):
            for n in range(neg_features):
                interaction = (config.interaction_strength *
                               X[i, p] * X[i, pos_features + n] *
                               np.sign(centers[class_idx, p] -
                                       centers[class_idx, pos_features + n]))
                X[i, p] += 0.1 * interaction
                X[i, pos_features + n] -= 0.1 * interaction

    # Add noise to all features
    X += config.noise_level * \
         np.random.randn(config.n_samples, config.n_features)

    # Standardize features
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    logger.info(
        f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, "
        f"{len(np.unique(y))} classes"
    )

    return X, y


# ==============================================================================
# MODEL ARCHITECTURE BUILDERS
# ==============================================================================

def build_differential_ffn_model(
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        activation: str = 'gelu',
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True
) -> keras.Model:
    """
    Build a model using DifferentialFFN with specified configuration.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        activation: Activation function for DifferentialFFN branches
        dropout_rate: Dropout probability
        l2_reg: L2 regularization strength
        use_batch_norm: Whether to use batch normalization

    Returns:
        Compiled Keras model with DifferentialFFN architecture
    """
    inputs = keras.layers.Input(shape=(input_dim,), name='input')

    # Primary DifferentialFFN layer
    x = DifferentialFFN(
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        branch_activation=activation,
        dropout_rate=dropout_rate,
        use_bias=True,
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='differential_ffn_1'
    )(inputs)

    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_1')(x)

    # Secondary processing layer
    x = keras.layers.Dense(
        hidden_dim // 2,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='dense_intermediate'
    )(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_final')(x)

    # Output layer
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='output'
    )(x)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name=f'DifferentialFFN_{activation}')

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    return model


def build_dense_baseline_model(
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True
) -> keras.Model:
    """
    Build a baseline model using standard Dense layers with equivalent parameters.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout_rate: Dropout probability
        l2_reg: L2 regularization strength
        use_batch_norm: Whether to use batch normalization

    Returns:
        Compiled Keras model with standard Dense architecture
    """
    inputs = keras.layers.Input(shape=(input_dim,), name='input')

    # First dense layer - equivalent to DifferentialFFN hidden processing
    x = keras.layers.Dense(
        hidden_dim * 2,  # Match parameter count with DifferentialFFN
        activation='gelu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='dense_1'
    )(inputs)

    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_1')(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_1')(x)

    # Second dense layer
    x = keras.layers.Dense(
        hidden_dim,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='dense_2'
    )(x)

    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_2')(x)

    # Intermediate layer
    x = keras.layers.Dense(
        hidden_dim // 2,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='dense_intermediate'
    )(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_final')(x)

    # Output layer
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='output'
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Dense_Baseline')

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    return model


def build_manual_dual_pathway_model(
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4,
        use_batch_norm: bool = True
) -> keras.Model:
    """
    Build a manual dual-pathway model without using DifferentialFFN.

    This model manually implements a dual-pathway architecture to compare
    against the DifferentialFFN implementation.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension per pathway
        dropout_rate: Dropout probability
        l2_reg: L2 regularization strength
        use_batch_norm: Whether to use batch normalization

    Returns:
        Compiled Keras model with manual dual-pathway architecture
    """
    inputs = keras.layers.Input(shape=(input_dim,), name='input')

    # Positive pathway
    pos_path = keras.layers.Dense(
        hidden_dim,
        activation='gelu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='positive_path_1'
    )(inputs)

    pos_path = keras.layers.Dense(
        hidden_dim,
        activation='sigmoid',  # Different activation for positive path
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='positive_path_2'
    )(pos_path)

    # Negative pathway
    neg_path = keras.layers.Dense(
        hidden_dim,
        activation='gelu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='negative_path_1'
    )(inputs)

    neg_path = keras.layers.Dense(
        hidden_dim,
        activation='sigmoid',  # Matching activation for negative path
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='negative_path_2'
    )(neg_path)

    # Combine pathways with subtraction (like DifferentialFFN)
    x = keras.layers.Subtract(name='pathway_combination')([pos_path, neg_path])

    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_combination')(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_combination')(x)

    # Final processing layer
    x = keras.layers.Dense(
        hidden_dim // 2,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='dense_final'
    )(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_final')(x)

    # Output layer
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='output'
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Manual_DualPath')

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    return model


# ==============================================================================
# VISUALIZATION AND ANALYSIS FUNCTIONS
# ==============================================================================

def visualize_feature_representations(
        models: Dict[str, keras.Model],
        X: np.ndarray,
        y: np.ndarray,
        output_dir: Path,
        vis_manager: VisualizationManager
) -> None:
    """
    Visualize learned feature representations using PCA projection.

    Args:
        models: Dictionary mapping model names to trained models
        X: Input features for visualization
        y: True class labels
        output_dir: Directory to save visualizations
        vis_manager: Visualization manager for plot generation
    """
    logger.info("Generating feature representation visualizations...")

    n_models = len(models)
    fig, axes = plt.subplots(
        2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
    if n_models <= 2:
        axes = axes.reshape(2, -1)

    for idx, (name, model) in enumerate(models.items()):
        row = idx // ((n_models + 1) // 2)
        col = idx % ((n_models + 1) // 2)
        ax = axes[row, col] if n_models > 2 else axes[row] if n_models == 2 else axes[idx]

        # Extract features from the layer before the final output
        feature_model = keras.Model(
            inputs=model.inputs,
            outputs=model.layers[-2].output  # Second to last layer
        )

        # Get feature representations
        features = feature_model.predict(X, verbose=0)

        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features)

        # Create scatter plot
        scatter = ax.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=y, cmap='tab10', alpha=0.7, s=20
        )

        ax.set_title(
            f'{name}\nExplained Var: {pca.explained_variance_ratio_.sum():.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

        # Add colorbar only for the last plot
        if idx == len(models) - 1:
            plt.colorbar(scatter, ax=ax, label='Class')

    # Hide unused subplots
    if n_models % 2 == 1 and n_models > 2:
        axes[1, -1].set_visible(False)

    plt.tight_layout()
    vis_manager.save_figure(fig, 'feature_representations', subdir='analysis')
    plt.close(fig)


def visualize_decision_boundaries(
        models: Dict[str, keras.Model],
        X: np.ndarray,
        y: np.ndarray,
        output_dir: Path,
        vis_manager: VisualizationManager
) -> None:
    """
    Visualize decision boundaries using PCA projection to 2D space.

    Args:
        models: Dictionary mapping model names to trained models
        X: Input features
        y: True class labels
        output_dir: Directory to save visualizations
        vis_manager: Visualization manager for plot generation
    """
    logger.info("Generating decision boundary visualizations...")

    # Project data to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # Create a mesh for decision boundary plotting
    h = 0.02  # Step size in the mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    n_models = len(models)
    fig, axes = plt.subplots(
        2, (n_models + 1) // 2, figsize=(6 * ((n_models + 1) // 2), 12))
    if n_models <= 2:
        axes = axes.reshape(2, -1) if n_models == 2 else [axes]

    for idx, (name, model) in enumerate(models.items()):
        row = idx // ((n_models + 1) // 2)
        col = idx % ((n_models + 1) // 2)
        ax = axes[row][col] if n_models > 2 else axes[row] if n_models == 2 else axes[idx]

        # Create grid points in original feature space
        mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]
        mesh_points = pca.inverse_transform(mesh_points_2d)

        # Predict on mesh points
        Z = model.predict(mesh_points, verbose=0)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='tab10')

        # Plot data points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10',
                             edgecolors='black', linewidth=0.5, s=30)

        ax.set_title(f'{name} Decision Boundary')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    if n_models % 2 == 1 and n_models > 2:
        axes[1][-1].set_visible(False)

    plt.tight_layout()
    vis_manager.save_figure(fig, 'decision_boundaries', subdir='analysis')
    plt.close(fig)


def perform_statistical_analysis(
        results_df: pd.DataFrame,
        config: ExperimentConfig,
        output_dir: Path
) -> Dict[str, Any]:
    """
    Perform statistical analysis of model performance differences.

    Args:
        results_df: DataFrame containing performance results for all models
        config: Experiment configuration
        output_dir: Directory to save analysis results

    Returns:
        Dictionary containing statistical analysis results
    """
    logger.info("Performing statistical significance analysis...")

    stats_results = {}

    # Prepare data for statistical testing
    model_names = results_df['Model'].unique()
    accuracy_data = {}

    for model in model_names:
        model_data = results_df[results_df['Model'] == model]
        accuracy_data[model] = model_data['Accuracy'].values

    # Perform pairwise t-tests
    pairwise_results = {}
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i + 1:]:
            stat, p_value = stats.ttest_rel(
                accuracy_data[model1], accuracy_data[model2])
            effect_size = (np.mean(accuracy_data[model1]) - np.mean(accuracy_data[model2])) / \
                          np.sqrt(
                              (np.var(accuracy_data[model1]) + np.var(accuracy_data[model2])) / 2)

            pairwise_results[f"{model1}_vs_{model2}"] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'significant': p_value < 0.05
            }

    stats_results['pairwise_comparisons'] = pairwise_results

    # Overall ANOVA
    accuracy_values = [accuracy_data[model] for model in model_names]
    f_stat, p_value = stats.f_oneway(*accuracy_values)

    stats_results['anova'] = {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }

    # Summary statistics
    summary_stats = {}
    for model in model_names:
        data = accuracy_data[model]
        summary_stats[model] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data))
        }

    stats_results['summary_statistics'] = summary_stats

    # Before saving to JSON, convert numpy types to Python native types
    stats_results_serializable = convert_numpy_to_python(stats_results)

    # Save statistical results
    stats_file = output_dir / "statistical_analysis.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_results_serializable, f, indent=2)

    return stats_results_serializable


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Execute the complete DifferentialFFN evaluation experiment.

    This function orchestrates the entire experimental pipeline including:
    dataset generation, model training, performance evaluation, model analysis,
    statistical testing, and visualization generation.

    Args:
        config: Experiment configuration specifying all parameters

    Returns:
        Dictionary containing comprehensive experimental results
    """
    # Set random seeds for reproducibility
    keras.utils.set_random_seed(config.random_state)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        config=VisualizationConfig(),
        timestamp_dirs=False
    )

    logger.info("Starting DifferentialFFN Architecture Evaluation Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET GENERATION =====
    logger.info("Generating synthetic differential dataset...")
    X, y = generate_differential_dataset(config)

    # ===== CROSS-VALIDATION EVALUATION =====
    logger.info("Performing cross-validation evaluation...")
    cv_results = {}
    all_models = {}
    all_histories = {}

    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=config.cv_folds,
                          shuffle=True, random_state=config.random_state)

    # Store results for each model and fold
    cv_scores = {name: [] for name in config.model_builders.keys()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Processing fold {fold + 1}/{config.cv_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert integer labels to one-hot for categorical_crossentropy
        y_train_onehot = keras.utils.to_categorical(y_train, num_classes=config.n_classes)
        y_val_onehot = keras.utils.to_categorical(y_val, num_classes=config.n_classes)

        fold_models = {}
        fold_histories = {}

        for model_name, model_builder in config.model_builders.items():
            logger.info(f"Training {model_name} on fold {fold + 1}")

            # Build model for this fold
            model = model_builder()

            # Train model
            history = model.fit(
                X_train, y_train_onehot,  # Use one-hot labels
                validation_data=(X_val, y_val_onehot),  # Use one-hot labels
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor=config.monitor_metric,
                        patience=config.early_stopping_patience,
                        restore_best_weights=True,
                        verbose=0
                    )
                ],
                verbose=0
            )

            # Evaluate model
            val_loss, val_accuracy, val_top3_accuracy = model.evaluate(
                X_val, y_val_onehot, verbose=0)  # Use one-hot labels
            cv_scores[model_name].append(val_accuracy)

            # Store results for the last fold (for analysis)
            if fold == config.cv_folds - 1:
                fold_models[model_name] = model
                fold_histories[model_name] = history.history

        # Store models and histories from last fold for analysis
        if fold == config.cv_folds - 1:
            all_models = fold_models
            all_histories = fold_histories

    # ===== PERFORMANCE ANALYSIS =====
    logger.info("Analyzing cross-validation results...")

    performance_results = {}
    results_data = []

    for model_name, scores in cv_scores.items():
        performance_results[model_name] = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
            'min_accuracy': float(np.min(scores)),
            'max_accuracy': float(np.max(scores)),
            'cv_scores': [float(s) for s in scores]
        }

        # Prepare data for statistical analysis
        for fold, score in enumerate(scores):
            results_data.append({
                'Model': model_name,
                'Fold': fold + 1,
                'Accuracy': score
            })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(experiment_dir / "cv_results.csv", index=False)

    # ===== STATISTICAL ANALYSIS =====
    statistical_results = None
    if config.statistical_testing:
        statistical_results = perform_statistical_analysis(
            results_df, config, experiment_dir)

    # ===== COMPREHENSIVE MODEL ANALYSIS =====
    logger.info("Performing comprehensive model analysis with ModelAnalyzer...")
    model_analysis_results = None

    try:
        # Use a smaller subset for analysis to avoid memory issues
        analysis_indices = np.random.choice(
            len(X), size=min(2000, len(X)), replace=False)
        X_analysis = X[analysis_indices]
        y_analysis = y[analysis_indices]

        # CRITICAL FIX: Convert integer labels to one-hot for calibration analysis
        logger.info(f"Converting labels to one-hot format for calibration analysis")
        y_analysis_onehot = keras.utils.to_categorical(y_analysis, num_classes=config.n_classes)

        logger.info(f"Analysis data shapes: X={X_analysis.shape}, y_onehot={y_analysis_onehot.shape}")

        # Create DataInput object with one-hot labels for calibration analysis
        data_input = DataInput(x_data=X_analysis, y_data=y_analysis_onehot)

        analyzer = ModelAnalyzer(
            models=all_models,
            config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )

        model_analysis_results = analyzer.analyze(data=data_input)

        logger.info("Model analysis completed successfully")

    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)
        # Log additional details for debugging
        logger.error(
            f"Analysis data shapes: X={X_analysis.shape if 'X_analysis' in locals() else 'N/A'}, "
            f"y_original={y_analysis.shape if 'y_analysis' in locals() else 'N/A'}, "
            f"y_onehot={y_analysis_onehot.shape if 'y_analysis_onehot' in locals() else 'N/A'}")

    # ===== VISUALIZATION GENERATION =====
    logger.info("Generating visualizations...")

    # Plot cross-validation results
    vis_manager.plot_history(
        histories=all_histories,
        metrics=['accuracy', 'loss'],
        name='training_history_comparison',
        subdir='training',
        title='DifferentialFFN Architecture Training Comparison'
    )

    # Create performance comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Mean accuracy with error bars
    models = list(performance_results.keys())
    means = [performance_results[m]['mean_accuracy'] for m in models]
    stds = [performance_results[m]['std_accuracy'] for m in models]

    ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title('Cross-Validation Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Box plot of CV scores
    cv_data = [cv_scores[model] for model in models]
    ax2.boxplot(cv_data, tick_labels=models)
    ax2.set_title('Cross-Validation Score Distribution')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    vis_manager.save_figure(fig, 'performance_comparison', subdir='analysis')
    plt.close(fig)

    # Generate feature analysis visualizations
    if config.generate_feature_representations:
        # Use subset for visualization
        viz_indices = np.random.choice(
            len(X), size=min(1000, len(X)), replace=False)
        X_viz = X[viz_indices]
        y_viz = y[viz_indices]

        visualize_feature_representations(
            all_models, X_viz, y_viz, experiment_dir, vis_manager)

    if config.visualize_decision_boundaries:
        # Use smaller subset for decision boundary visualization
        boundary_indices = np.random.choice(
            len(X), size=min(500, len(X)), replace=False)
        X_boundary = X[boundary_indices]
        y_boundary = y[boundary_indices]

        visualize_decision_boundaries(
            all_models, X_boundary, y_boundary, experiment_dir, vis_manager)

    # ===== MEMORY CLEANUP =====
    gc.collect()

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'performance_analysis': performance_results,
        'statistical_analysis': statistical_results,
        'model_analysis': model_analysis_results,
        'training_histories': all_histories,
        'cv_results_dataframe': results_df,
        'config': config,
        'dataset_info': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        }
    }

    # Print comprehensive summary
    print_experiment_summary(results_payload)

    return results_payload


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """
    Print comprehensive experimental results summary.

    Args:
        results: Dictionary containing all experimental results
    """
    logger.info("=" * 80)
    logger.info("DIFFERENTIAL FFN EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # ===== PERFORMANCE RESULTS =====
    if 'performance_analysis' in results and results['performance_analysis']:
        logger.info("CROSS-VALIDATION PERFORMANCE RESULTS:")
        logger.info(
            f"{'Model':<25} {'Mean Acc':<10} {'Std Acc':<10} "
            f"{'Min Acc':<10} {'Max Acc':<10}"
        )
        logger.info("-" * 70)

        # Sort models by mean accuracy for better readability
        sorted_models = sorted(
            results['performance_analysis'].items(),
            key=lambda x: x[1]['mean_accuracy'],
            reverse=True
        )

        for model_name, metrics in sorted_models:
            logger.info(
                f"{model_name:<25} "
                f"{metrics['mean_accuracy']:<10.4f} "
                f"{metrics['std_accuracy']:<10.4f} "
                f"{metrics['min_accuracy']:<10.4f} "
                f"{metrics['max_accuracy']:<10.4f}"
            )

        # Highlight best performing model
        best_model = sorted_models[0]

        logger.info("-" * 70)

        logger.info(
            f"Best Performing Model: {best_model[0]} "
            f"(Mean Accuracy: {best_model[1]['mean_accuracy']:.4f})"
        )

    # ===== STATISTICAL ANALYSIS RESULTS =====
    if 'statistical_analysis' in results and results['statistical_analysis']:
        stats_results = results['statistical_analysis']

        logger.info("STATISTICAL SIGNIFICANCE ANALYSIS:")

        # ANOVA results
        if 'anova' in stats_results:
            anova = stats_results['anova']
            logger.info(
                f"Overall ANOVA: F={anova['f_statistic']:.4f}, "
                f"p={anova['p_value']:.4f}, "
                f"Significant: {anova['significant']}"
            )

        # Significant pairwise comparisons
        if 'pairwise_comparisons' in stats_results:
            significant_pairs = [
                (pair, data) for pair, data in stats_results['pairwise_comparisons'].items()
                if data['significant']
            ]

            if significant_pairs:
                logger.info("Significant Pairwise Comparisons (p < 0.05):")
                for pair, data in significant_pairs:
                    logger.info(
                        f"  {pair}: p={data['p_value']:.4f}, "
                        f"Effect Size={data['effect_size']:.4f}"
                    )
            else:
                logger.info(
                    "No statistically significant differences found "
                    "between models."
                )

    # ===== CALIBRATION ANALYSIS =====
    model_analysis = results.get('model_analysis')
    if model_analysis and hasattr(model_analysis, 'calibration_metrics') and model_analysis.calibration_metrics:
        logger.info("CALIBRATION ANALYSIS:")
        logger.info(
            f"{'Model':<25} {'ECE':<10} {'Brier':<10}")
        logger.info("-" * 45)

        for model_name, cal_metrics in model_analysis.calibration_metrics.items():
            # Use safe access with .get() and provide defaults
            ece = cal_metrics.get('ece', 'N/A')
            brier = cal_metrics.get('brier_score', 'N/A')

            # Format the values appropriately
            ece_str = f"{ece:.4f}" if isinstance(ece, (int, float)) else str(ece)
            brier_str = f"{brier:.4f}" if isinstance(brier, (int, float)) else str(brier)

            logger.info(
                f"{model_name:<25} "
                f"{ece_str:<10} "
                f"{brier_str:<10}"
            )

    # ===== DATASET INFORMATION =====
    if 'dataset_info' in results:
        dataset_info = results['dataset_info']
        logger.info("DATASET CHARACTERISTICS:")
        logger.info(f"  Samples: {dataset_info['n_samples']}")
        logger.info(f"  Features: {dataset_info['n_features']}")
        logger.info(f"  Classes: {dataset_info['n_classes']}")
        logger.info(f"  Class Distribution: {dataset_info['class_distribution']}")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for the DifferentialFFN evaluation experiment.
    """
    logger.info("DifferentialFFN Architecture Evaluation Experiment")
    logger.info("=" * 80)

    # Initialize experiment configuration
    config = ExperimentConfig()

    # Log key configuration parameters
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(
        f"  Dataset: {config.n_samples} samples, {config.n_features} features, "
        f"{config.n_classes} classes"
    )
    logger.info(f"  Architectures: {list(config.model_builders.keys())}")
    logger.info(f"  Cross-validation: {config.cv_folds} folds")
    logger.info(f"  Training: {config.epochs} epochs, batch size {config.batch_size}")
    logger.info("")

    try:
        # Execute experiment
        _ = run_experiment(config)
        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()