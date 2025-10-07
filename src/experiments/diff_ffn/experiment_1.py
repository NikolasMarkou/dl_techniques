"""
Experiment: Evaluating the DifferentialFFN Architecture

This script conducts a comprehensive experiment to evaluate the effectiveness of the
DifferentialFFN layer, a novel feed-forward network architecture. The evaluation is
performed on a synthetic dataset specifically designed to have features with both
positive and negative correlations with the target classes, a scenario where the
DifferentialFFN is hypothesized to excel.

Hypothesis:
-----------
The core hypothesis is that the DifferentialFFN's dual-pathway architecture, which
processes inputs through separate positive and negative branches before combining them,
can more effectively learn and disentangle complex, opposing feature relationships
than standard dense architectures. This should lead to improved classification
performance, better generalization, and more interpretable feature representations.

Experimental Design:
--------------------
- **Dataset**: A synthetic, multi-class classification dataset is generated with a
  controlled number of informative features, noise, and interaction effects. This
  ensures the task has the specific differential characteristics needed to test the
  hypothesis.

- **Model Architectures Compared**:
    1.  **DifferentialFFN**: The primary model under investigation, tested with
        various activation functions (e.g., GELU, ELU, ReLU).
    2.  **Dense Baseline**: A standard multi-layer perceptron (MLP) with a
        comparable number of parameters to the DifferentialFFN, serving as a
        direct baseline.
    3.  **Manual Dual-Path**: An MLP that manually mimics the dual-pathway
        structure of the DifferentialFFN using standard Keras layers, providing a
        comparison for the architectural concept itself.

- **Evaluation Protocol**: A robust stratified K-fold cross-validation methodology
  is employed to evaluate model performance. This minimizes the impact of random
  data splits and provides a reliable estimate of generalization performance.

Workflow:
---------
1.  **Configuration**: All experimental parameters are centralized in the
    `ExperimentConfig` dataclass for easy modification and reproducibility.
2.  **Data Generation**: The `generate_differential_dataset` function creates the
    synthetic dataset based on the specified configuration.
3.  **Cross-Validation**: The script iterates through each fold of the stratified
    K-fold split. For each fold, all model variants are built, trained, and
    evaluated.
4.  **Performance Analysis**: After all folds are complete, the performance
    metrics (e.g., accuracy) are aggregated. Mean, standard deviation, and other
    statistics are calculated for each model.
5.  **Statistical Testing**: Pairwise t-tests and ANOVA are performed on the
    cross-validation scores to determine if the performance differences between
    models are statistically significant.
6.  **Deep-Dive Analysis**: The models from the final fold are subjected to a
    deeper analysis using the `ModelAnalyzer` to inspect weight distributions,
    activation patterns, and prediction calibration (ECE, Brier score).
7.  **Visualization**:
    - The new `VisualizationManager` plots comparative training histories.
    - Custom visualizations are generated for feature representations (via PCA)
      and 2D decision boundaries to provide qualitative insights into how each
      architecture learns to separate the classes.
8.  **Reporting**: A comprehensive summary of all results is printed to the
    console, and all artifacts (plots, logs, data, and statistical results)
    are saved to a unique, timestamped output directory.
"""

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

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN
from dl_techniques.utils.convert import convert_numpy_to_python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    TrainingCurvesVisualization
)

# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the DifferentialFFN architecture evaluation."""
    n_samples: int = 50000
    n_features: int = 30
    n_informative: int = 12
    n_classes: int = 6
    class_sep: float = 1.5
    noise_level: float = 0.1
    interaction_strength: float = 0.2
    random_state: int = 42

    hidden_dim: int = 256
    output_classes: int = 6
    dropout_rate: float = 0.25
    use_batch_norm: bool = True
    l2_reg: float = 1e-4
    differential_activations: List[str] = field(default_factory=lambda: ['gelu', 'elu', 'relu'])

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    monitor_metric: str = 'val_accuracy'
    validation_split: float = 0.2

    cv_folds: int = 10
    cv_scoring: str = 'accuracy'
    model_builders: Dict[str, Callable] = field(default_factory=dict)

    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True, analyze_calibration=True, analyze_information_flow=True,
        analyze_training_dynamics=False, calibration_bins=15, save_plots=True,
        plot_style='publication', dpi=300
    ))

    output_dir: Path = Path("results")
    experiment_name: str = "differential_ffn_evaluation"
    analyze_feature_importance: bool = True
    visualize_decision_boundaries: bool = True
    generate_feature_representations: bool = True
    statistical_testing: bool = True

    def __post_init__(self):
        """Post-initialization setup for derived parameters."""
        self.output_classes = self.n_classes
        self.model_builders = {}
        for activation in self.differential_activations:
            self.model_builders[f'DifferentialFFN_{activation.upper()}'] = (
                lambda act=activation: self._build_differential_ffn_model(act)
            )
        self.model_builders.update({
            'Dense_Baseline': self._build_dense_baseline_model,
            'Manual_DualPath': self._build_manual_dual_pathway_model,
        })

    def _build_differential_ffn_model(self, activation: str) -> keras.Model:
        return build_differential_ffn_model(
            self.n_features, self.output_classes, self.hidden_dim, activation,
            self.dropout_rate, self.l2_reg, self.use_batch_norm
        )

    def _build_dense_baseline_model(self) -> keras.Model:
        return build_dense_baseline_model(
            self.n_features, self.output_classes, self.hidden_dim,
            self.dropout_rate, self.l2_reg, self.use_batch_norm
        )

    def _build_manual_dual_pathway_model(self) -> keras.Model:
        return build_manual_dual_pathway_model(
            self.n_features, self.output_classes, self.hidden_dim,
            self.dropout_rate, self.l2_reg, self.use_batch_norm
        )

# ==============================================================================
# SYNTHETIC DATASET GENERATION
# ==============================================================================

def generate_differential_dataset(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset with positive/negative feature correlations."""
    np.random.seed(config.random_state)
    X = np.random.randn(config.n_samples, config.n_features)
    centers = np.random.randn(config.n_classes, config.n_informative) * config.class_sep
    y = np.random.randint(0, config.n_classes, size=config.n_samples)
    pos_features = config.n_informative // 2
    neg_features = config.n_informative - pos_features
    logger.info(f"Generating dataset with {pos_features} positive and {neg_features} negative features.")

    for i in range(config.n_samples):
        class_idx = y[i]
        X[i, :pos_features] += centers[class_idx, :pos_features]
        X[i, pos_features:config.n_informative] -= centers[class_idx, pos_features:]
        for p in range(pos_features):
            for n in range(neg_features):
                interaction = (config.interaction_strength * X[i, p] * X[i, pos_features + n] *
                               np.sign(centers[class_idx, p] - centers[class_idx, pos_features + n]))
                X[i, p] += 0.1 * interaction
                X[i, pos_features + n] -= 0.1 * interaction

    X += config.noise_level * np.random.randn(config.n_samples, config.n_features)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    logger.info(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes.")
    return X, y

# ==============================================================================
# MODEL ARCHITECTURE BUILDERS
# ==============================================================================

def build_differential_ffn_model(
        input_dim: int, num_classes: int, hidden_dim: int = 256, activation: str = 'gelu',
        dropout_rate: float = 0.3, l2_reg: float = 1e-4, use_batch_norm: bool = True
) -> keras.Model:
    """Build a model using the DifferentialFFN layer."""
    inputs = keras.layers.Input(shape=(input_dim,), name='input')
    x = DifferentialFFN(
        hidden_dim=hidden_dim, output_dim=hidden_dim, branch_activation=activation,
        dropout_rate=dropout_rate, use_bias=True,
        kernel_regularizer=keras.regularizers.L2(l2_reg), name='differential_ffn_1'
    )(inputs)
    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
    x = keras.layers.Dense(
        hidden_dim // 2, activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg), name='dense_intermediate'
    )(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_final')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'DifferentialFFN_{activation}')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model

def build_dense_baseline_model(
        input_dim: int, num_classes: int, hidden_dim: int = 256, dropout_rate: float = 0.3,
        l2_reg: float = 1e-4, use_batch_norm: bool = True
) -> keras.Model:
    """Build a baseline model using standard Dense layers."""
    inputs = keras.layers.Input(shape=(input_dim,), name='input')
    x = keras.layers.Dense(
        hidden_dim * 2, activation='gelu',
        kernel_regularizer=keras.regularizers.L2(l2_reg), name='dense_1'
    )(inputs)
    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = keras.layers.Dense(
        hidden_dim, activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg), name='dense_2'
    )(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_2')(x)
    x = keras.layers.Dense(
        hidden_dim // 2, activation='relu',
        kernel_regularizer=keras.regularizers.L2(l2_reg), name='dense_intermediate'
    )(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_final')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='Dense_Baseline')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model

def build_manual_dual_pathway_model(
        input_dim: int, num_classes: int, hidden_dim: int = 256, dropout_rate: float = 0.3,
        l2_reg: float = 1e-4, use_batch_norm: bool = True
) -> keras.Model:
    """Build a manual dual-pathway model without using DifferentialFFN."""
    inputs = keras.layers.Input(shape=(input_dim,), name='input')
    pos_path = keras.layers.Dense(hidden_dim, activation='gelu', name='positive_path_1')(inputs)
    pos_path = keras.layers.Dense(hidden_dim, activation='sigmoid', name='positive_path_2')(pos_path)
    neg_path = keras.layers.Dense(hidden_dim, activation='gelu', name='negative_path_1')(inputs)
    neg_path = keras.layers.Dense(hidden_dim, activation='sigmoid', name='negative_path_2')(neg_path)
    x = keras.layers.Subtract(name='pathway_combination')([pos_path, neg_path])
    if use_batch_norm:
        x = keras.layers.BatchNormalization(name='batch_norm_combination')(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_combination')(x)
    x = keras.layers.Dense(hidden_dim // 2, activation='relu', name='dense_final')(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name='dropout_final')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='Manual_DualPath')
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model

# ==============================================================================
# VISUALIZATION AND ANALYSIS FUNCTIONS
# ==============================================================================

def visualize_feature_representations(
        models: Dict[str, keras.Model], X: np.ndarray, y: np.ndarray, output_dir: Path
) -> None:
    """Visualize learned feature representations using PCA projection."""
    logger.info("Generating feature representation visualizations...")
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        feature_model = keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features = feature_model.predict(X, verbose=0)
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features)
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=y, cmap='viridis', alpha=0.7, s=15)
        ax.set_title(f'{name}\nExplained Var: {pca.explained_variance_ratio_.sum():.3f}')
        ax.set(xlabel='PC1', ylabel='PC2')
        ax.grid(True, alpha=0.3)

    fig.colorbar(scatter, ax=axes, label='Class', orientation='vertical')
    plt.tight_layout()
    plot_path = output_dir / "analysis"
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path / 'feature_representations.png', dpi=300)
    plt.close(fig)

def visualize_decision_boundaries(
        models: Dict[str, keras.Model], X: np.ndarray, y: np.ndarray, output_dir: Path
) -> None:
    """Visualize 2D decision boundaries using PCA."""
    logger.info("Generating decision boundary visualizations...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(model.predict(mesh_points, verbose=0), axis=1).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='k', s=20)
        ax.set_title(f'{name} Decision Boundary')
        ax.set(xlabel='PC1', ylabel='PC2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "analysis"
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path / 'decision_boundaries.png', dpi=300)
    plt.close(fig)

def perform_statistical_analysis(results_df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Perform and save statistical analysis of model performance."""
    logger.info("Performing statistical significance analysis...")
    stats_results = {}
    model_names = results_df['Model'].unique()
    accuracy_data = {model: results_df[results_df['Model'] == model]['Accuracy'].values for model in model_names}

    pairwise_results = {}
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i + 1:]:
            stat, p_value = stats.ttest_rel(accuracy_data[model1], accuracy_data[model2])
            pairwise_results[f"{model1}_vs_{model2}"] = {'p_value': float(p_value), 'significant': p_value < 0.05}
    stats_results['pairwise_comparisons'] = pairwise_results

    f_stat, p_value = stats.f_oneway(*[accuracy_data[model] for model in model_names])
    stats_results['anova'] = {'f_statistic': float(f_stat), 'p_value': float(p_value), 'significant': p_value < 0.05}

    stats_results_serializable = convert_numpy_to_python(stats_results)
    with open(output_dir / "statistical_analysis.json", 'w') as f:
        json.dump(stats_results_serializable, f, indent=2)
    return stats_results_serializable

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Execute the complete DifferentialFFN evaluation experiment."""
    keras.utils.set_random_seed(config.random_state)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    viz_manager = VisualizationManager(
        output_dir=experiment_dir / "visualizations",
        experiment_name=config.experiment_name
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)

    logger.info("Starting DifferentialFFN Architecture Evaluation Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    logger.info("Generating synthetic differential dataset...")
    X, y = generate_differential_dataset(config)

    logger.info("Performing cross-validation evaluation...")
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    cv_scores = {name: [] for name in config.model_builders.keys()}
    all_models, all_histories = {}, {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Processing fold {fold + 1}/{config.cv_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_onehot = keras.utils.to_categorical(y[train_idx], config.n_classes)
        y_val_onehot = keras.utils.to_categorical(y[val_idx], config.n_classes)

        for name, builder in config.model_builders.items():
            model = builder()
            history = model.fit(
                X_train, y_train_onehot, validation_data=(X_val, y_val_onehot),
                epochs=config.epochs, batch_size=config.batch_size,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor=config.monitor_metric, patience=config.early_stopping_patience,
                    restore_best_weights=True, verbose=0)],
                verbose=0
            )
            _, val_accuracy = model.evaluate(X_val, y_val_onehot, verbose=0)
            cv_scores[name].append(val_accuracy)
            if fold == config.cv_folds - 1:
                all_models[name] = model
                all_histories[name] = history.history

    logger.info("Analyzing cross-validation results...")
    results_data = [{'Model': name, 'Fold': i + 1, 'Accuracy': score}
                    for name, scores in cv_scores.items() for i, score in enumerate(scores)]
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(experiment_dir / "cv_results.csv", index=False)
    performance_results = {name: {'mean_accuracy': np.mean(s), 'std_accuracy': np.std(s)}
                           for name, s in cv_scores.items()}

    statistical_results = perform_statistical_analysis(results_df, experiment_dir) if config.statistical_testing else None

    logger.info("Performing comprehensive model analysis with ModelAnalyzer...")
    model_analysis_results = None
    try:
        indices = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
        X_analysis, y_analysis = X[indices], y[indices]
        y_analysis_onehot = keras.utils.to_categorical(y_analysis, config.n_classes)
        data_input = DataInput(x_data=X_analysis, y_data=y_analysis_onehot)
        analyzer = ModelAnalyzer(
            models=all_models, config=config.analyzer_config,
            output_dir=experiment_dir / "model_analysis"
        )
        model_analysis_results = analyzer.analyze(data=data_input)
        logger.info("Model analysis completed successfully.")
    except Exception as e:
        logger.error(f"Model analysis failed: {e}", exc_info=True)

    logger.info("Generating visualizations...")
    training_histories_data = {
        name: TrainingHistory(
            epochs=list(range(len(h['loss']))), train_loss=h.get('loss'), val_loss=h.get('val_loss'),
            train_metrics={'accuracy': h.get('accuracy')}, val_metrics={'accuracy': h.get('val_accuracy')}
        ) for name, h in all_histories.items()
    }
    viz_manager.visualize(
        data=training_histories_data, plugin_name="training_curves",
        title='DifferentialFFN Architecture Training Comparison'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    models, means, stds = list(performance_results.keys()), \
                          [p['mean_accuracy'] for p in performance_results.values()], \
                          [p['std_accuracy'] for p in performance_results.values()]
    ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    ax1.set(title='Cross-Validation Accuracy Comparison', ylabel='Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax2.boxplot([cv_scores[m] for m in models], tick_labels=models)
    ax2.set(title='Cross-Validation Score Distribution', ylabel='Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(experiment_dir / "visualizations" / "analysis" / "performance_comparison.png", dpi=300)
    plt.close(fig)

    if config.generate_feature_representations or config.visualize_decision_boundaries:
        viz_indices = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
        X_viz, y_viz = X[viz_indices], y[viz_indices]
        if config.generate_feature_representations:
            visualize_feature_representations(all_models, X_viz, y_viz, experiment_dir / "visualizations")
        if config.visualize_decision_boundaries:
            visualize_decision_boundaries(all_models, X_viz, y_viz, experiment_dir / "visualizations")

    gc.collect()
    results_payload = {
        'performance_analysis': performance_results, 'statistical_analysis': statistical_results,
        'model_analysis': model_analysis_results, 'training_histories': all_histories,
        'cv_results_dataframe': results_df.to_dict(), 'config': config.__dict__
    }
    print_experiment_summary(results_payload)
    return results_payload

# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive experimental results summary."""
    logger.info("=" * 80)
    logger.info("DIFFERENTIAL FFN EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if 'performance_analysis' in results:
        logger.info("CROSS-VALIDATION PERFORMANCE RESULTS:")
        logger.info(f"{'Model':<25} {'Mean Acc':<15} {'Std Acc':<15}")
        logger.info("-" * 55)
        sorted_models = sorted(results['performance_analysis'].items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
        for name, metrics in sorted_models:
            logger.info(f"{name:<25} {metrics['mean_accuracy']:.4f} {metrics['std_accuracy']:.4f}")
        best_model = sorted_models[0]
        logger.info(f"Best Performing Model: {best_model[0]} (Mean Accuracy: {best_model[1]['mean_accuracy']:.4f})")

    if 'statistical_analysis' in results and results['statistical_analysis']:
        stats = results['statistical_analysis']
        logger.info("STATISTICAL SIGNIFICANCE ANALYSIS:")
        if 'anova' in stats:
            logger.info(f"Overall ANOVA: p={stats['anova']['p_value']:.4f}, Significant: {stats['anova']['significant']}")
        if 'pairwise_comparisons' in stats:
            sig_pairs = [(p, d) for p, d in stats['pairwise_comparisons'].items() if d['significant']]
            if sig_pairs:
                logger.info("Significant Pairwise Comparisons (p < 0.05):")
                for pair, data in sig_pairs:
                    logger.info(f"  {pair}: p={data['p_value']:.4f}")

    analysis = results.get('model_analysis')
    if analysis and hasattr(analysis, 'calibration_metrics') and analysis.calibration_metrics:
        logger.info("CALIBRATION ANALYSIS (from final fold):")
        logger.info(f"{'Model':<25} {'ECE':<10} {'Brier Score':<15}")
        logger.info("-" * 50)
        for name, metrics in analysis.calibration_metrics.items():
            logger.info(f"{name:<25} {metrics.get('ece', 'N/A'):.4f} {metrics.get('brier_score', 'N/A'):.4f}")

    logger.info("=" * 80)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Main execution function for the DifferentialFFN evaluation experiment."""
    logger.info("DifferentialFFN Architecture Evaluation Experiment")
    logger.info("=" * 80)
    config = ExperimentConfig()
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  Architectures: {list(config.model_builders.keys())}")
    logger.info(f"  Cross-validation: {config.cv_folds} folds")
    try:
        run_experiment(config)
        logger.info("Experiment completed successfully.")
    except Exception as e:
        logger.error(f"Experiment failed with an unhandled exception: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()