"""
Usage Examples and Specialized Visualization Templates
========================================================

Complete examples showing how to use the visualization framework,
plus additional specialized templates for specific use cases.
"""

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any


from .core import (
    VisualizationManager, PlotConfig, PlotStyle, ColorScheme,
    VisualizationPlugin, CompositeVisualization
)
from .training_performance import (
    TrainingHistory, ModelComparison, TrainingCurvesVisualization,
    ModelComparisonBarChart, PerformanceRadarChart, ConvergenceAnalysis,
    OverfittingAnalysis, PerformanceDashboard
)
from .classification import (
    ClassificationResults,
    ConfusionMatrixVisualization, ROCPRCurves,
    ClassificationReportVisualization, PerClassAnalysis,
    ErrorAnalysisDashboard
)
from .data_nn import (
    DataDistributionAnalysis, ClassBalanceVisualization,
    NetworkArchitectureVisualization, ActivationVisualization,
    WeightVisualization, FeatureMapVisualization, GradientVisualization
)


# =============================================================================
# Quick Start Examples
# =============================================================================

def example_basic_usage():
    """Basic usage example of the visualization framework."""

    # 1. Initialize the manager
    viz_manager = VisualizationManager(
        experiment_name="my_experiment",
        output_dir="visualizations"
    )

    # 2. Register templates
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("data_distribution", DataDistributionAnalysis)

    # 3. Create some dummy data
    history = TrainingHistory(
        epochs=list(range(50)),
        train_loss=np.random.exponential(0.5, 50)[::-1],
        val_loss=np.random.exponential(0.5, 50)[::-1] + 0.1,
        train_metrics={'accuracy': np.random.uniform(0.7, 1.0, 50)},
        val_metrics={'accuracy': np.random.uniform(0.6, 0.95, 50)}
    )

    # 4. Create visualization
    viz_manager.visualize(
        data=history,
        plugin_name="training_curves",
        save=True,
        show=False
    )

    print("Basic visualization created successfully!")


def example_custom_configuration():
    """Example with custom configuration."""

    # Custom color scheme
    custom_colors = ColorScheme(
        primary="#2E86AB",
        secondary="#A23B72",
        success="#F18F01",
        warning="#C73E1D",
        palette=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A0572"]
    )

    # Custom plot configuration
    config = PlotConfig(
        fig_size=(14, 8),
        dpi=150,
        style=PlotStyle.PUBLICATION,
        color_scheme=custom_colors,
        title_fontsize=18,
        save_format="pdf"
    )

    # Initialize manager with custom config
    viz_manager = VisualizationManager(
        experiment_name="custom_experiment",
        config=config
    )

    return viz_manager


def example_model_comparison():
    """Example comparing multiple models."""

    # Create manager
    viz_manager = VisualizationManager("model_comparison")

    # Register relevant templates
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart) # FIX: Correct name
    viz_manager.register_template("performance_radar", PerformanceRadarChart)
    viz_manager.register_template("performance_dashboard", PerformanceDashboard)

    # Create comparison data
    comparison = ModelComparison(
        model_names=["ResNet50", "VGG16", "EfficientNet", "MobileNet"],
        metrics={
            "ResNet50": {"accuracy": 0.94, "f1_score": 0.93, "precision": 0.95, "recall": 0.91},
            "VGG16": {"accuracy": 0.91, "f1_score": 0.90, "precision": 0.92, "recall": 0.88},
            "EfficientNet": {"accuracy": 0.95, "f1_score": 0.94, "precision": 0.96, "recall": 0.92},
            "MobileNet": {"accuracy": 0.89, "f1_score": 0.88, "precision": 0.90, "recall": 0.86}
        }
    )

    # Create multiple visualizations
    viz_manager.visualize(comparison, "model_comparison_bars", sort_by="accuracy") # FIX: Correct name
    viz_manager.visualize(comparison, "performance_radar", normalize=True)
    viz_manager.visualize(comparison, "performance_dashboard")

    print("Model comparison visualizations created!")


def example_classification_analysis():
    """Example for classification result analysis."""

    viz_manager = VisualizationManager("classification_analysis")

    # Register classification templates
    viz_manager.register_template("confusion", ConfusionMatrixVisualization)
    viz_manager.register_template("roc_pr", ROCPRCurves)
    viz_manager.register_template("per_class", PerClassAnalysis)
    viz_manager.register_template("error_analysis", ErrorAnalysisDashboard)

    # Generate dummy classification results
    n_samples = 1000
    n_classes = 5

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=100, replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, 100)

    # Generate probability scores
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)

    results = ClassificationResults(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=[f"Class_{i}" for i in range(n_classes)],
        model_name="MyModel"
    )

    # Create visualizations
    viz_manager.visualize(results, "confusion", normalize='true')
    viz_manager.visualize(results, "roc_pr", plot_type='both')
    viz_manager.visualize(results, "per_class")
    viz_manager.visualize(results, "error_analysis")

    print("Classification analysis complete!")


# =============================================================================
# Specialized Visualization Templates
# =============================================================================

class ExperimentComparisonDashboard(CompositeVisualization):
    """Dashboard for comparing multiple experiments."""

    @property
    def name(self) -> str:
        return "experiment_comparison"

    @property
    def description(self) -> str:
        return "Compare multiple experiments in a single dashboard"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and 'experiments' in data

    def create_visualization(
            self,
            data: Dict[str, Any],
            ax: Optional[plt.Axes] = None,
            metrics_to_compare: List[str] = ['loss', 'accuracy'],
            **kwargs
    ) -> plt.Figure:
        """Create experiment comparison dashboard."""

        experiments = data['experiments']
        n_experiments = len(experiments)

        fig = plt.figure(figsize=(16, 10))
        gs = plt.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Training curves comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_comparison(ax1, experiments, 'loss')

        # 2. Final metrics comparison
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_final_metrics(ax2, experiments)

        # 3. Best performance timeline
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_performance_timeline(ax3, experiments)

        # 4. Hyperparameter impact
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_hyperparameter_impact(ax4, experiments)

        # 5. Summary table
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_experiment_summary_table(ax5, experiments)

        plt.suptitle('Experiment Comparison Dashboard', fontsize=16, fontweight='bold')
        return fig

    def _plot_training_comparison(self, ax: plt.Axes, experiments: Dict, metric: str):
        """Plot training curves for all experiments."""

        for idx, (name, exp_data) in enumerate(experiments.items()):
            if 'history' in exp_data:
                color = self.config.color_scheme.get_model_color(name, idx)
                history = exp_data['history']

                if metric in history:
                    ax.plot(history[metric], label=name, color=color,
                            linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} Comparison Across Experiments')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_final_metrics(self, ax: plt.Axes, experiments: Dict):
        """Plot final metrics for each experiment."""

        metrics_data = []
        for name, exp_data in experiments.items():
            if 'final_metrics' in exp_data:
                metrics_data.append({
                    'Experiment': name[:15],
                    **exp_data['final_metrics']
                })

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df.set_index('Experiment').plot(kind='bar', ax=ax, rot=45)
            ax.set_title('Final Metrics Comparison')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_performance_timeline(self, ax: plt.Axes, experiments: Dict):
        """Plot when each experiment achieved best performance."""

        for idx, (name, exp_data) in enumerate(experiments.items()):
            if 'history' in exp_data and 'val_accuracy' in exp_data['history']:
                val_acc = exp_data['history']['val_accuracy']
                best_epoch = np.argmax(val_acc)
                best_value = val_acc[best_epoch]

                color = self.config.color_scheme.get_model_color(name, idx)
                ax.scatter(best_epoch, best_value, s=100, color=color,
                           label=name, alpha=0.7, edgecolors='black')

                # Add annotation
                ax.annotate(name[:10], (best_epoch, best_value),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Validation Accuracy')
        ax.set_title('Performance Timeline')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_hyperparameter_impact(self, ax: plt.Axes, experiments: Dict):
        """Analyze hyperparameter impact on performance."""

        # Extract hyperparameters and performance
        hp_data = []
        for name, exp_data in experiments.items():
            if 'hyperparameters' in exp_data and 'final_metrics' in exp_data:
                hp_data.append({
                    'name': name[:10],
                    'lr': exp_data['hyperparameters'].get('learning_rate', 0),
                    'batch_size': exp_data['hyperparameters'].get('batch_size', 0),
                    'performance': exp_data['final_metrics'].get('accuracy', 0)
                })

        if hp_data:
            # Simple scatter plot of learning rate vs performance
            lrs = [d['lr'] for d in hp_data]
            perfs = [d['performance'] for d in hp_data]

            ax.scatter(lrs, perfs, s=100, alpha=0.6)
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Final Accuracy')
            ax.set_title('Learning Rate Impact')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

            # Add labels
            for d in hp_data:
                ax.annotate(d['name'], (d['lr'], d['performance']),
                            fontsize=7, alpha=0.7)

    def _plot_experiment_summary_table(self, ax: plt.Axes, experiments: Dict):
        """Create summary table of all experiments."""

        table_data = []
        for name, exp_data in experiments.items():
            row = [name[:20]]

            # Add various metrics
            if 'hyperparameters' in exp_data:
                hp = exp_data['hyperparameters']
                row.append(f"{hp.get('learning_rate', 'N/A')}")
                row.append(f"{hp.get('batch_size', 'N/A')}")
            else:
                row.extend(['N/A', 'N/A'])

            if 'final_metrics' in exp_data:
                metrics = exp_data['final_metrics']
                row.append(f"{metrics.get('accuracy', 0):.3f}")
                row.append(f"{metrics.get('loss', 0):.3f}")
            else:
                row.extend(['N/A', 'N/A'])

            if 'training_time' in exp_data:
                row.append(f"{exp_data['training_time']:.1f}s")
            else:
                row.append('N/A')

            table_data.append(row)

        headers = ['Experiment', 'LR', 'Batch', 'Accuracy', 'Loss', 'Time']

        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        ax.axis('off')
        ax.set_title('Experiment Summary', fontsize=12, fontweight='bold')


class LossLandscapeVisualization(VisualizationPlugin):
    """Visualize loss landscape around trained models."""

    @property
    def name(self) -> str:
        return "loss_landscape"

    @property
    def description(self) -> str:
        return "Visualize the loss landscape"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and 'landscape' in data

    def create_visualization(
            self,
            data: Dict[str, Any],
            ax: Optional[plt.Axes] = None,
            plot_type: str = '2d',  # '1d', '2d', '3d'
            **kwargs
    ) -> plt.Figure:
        """Create loss landscape visualization."""

        if plot_type == '1d':
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.get_figure()

            # 1D loss landscape (along a direction)
            if 'alphas' in data and 'losses' in data:
                ax.plot(data['alphas'], data['losses'], 'o-', linewidth=2)
                ax.set_xlabel('α (Distance along direction)')
                ax.set_ylabel('Loss')
                ax.set_title('1D Loss Landscape')

                # Mark minimum
                min_idx = np.argmin(data['losses'])
                ax.scatter(data['alphas'][min_idx], data['losses'][min_idx],
                           color='red', s=100, zorder=5, marker='*')
                ax.grid(True, alpha=0.3)
            return fig

        elif plot_type == '2d':
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
            else:
                fig = ax.get_figure()

            # 2D loss landscape (contour plot)
            if 'X' in data and 'Y' in data and 'Z' in data:
                contour = ax.contour(data['X'], data['Y'], data['Z'],
                                     levels=20, cmap='viridis')
                ax.clabel(contour, inline=True, fontsize=8)

                contourf = ax.contourf(data['X'], data['Y'], data['Z'],
                                       levels=20, cmap='viridis', alpha=0.7)
                plt.colorbar(contourf, ax=ax, label='Loss')

                # Mark the trained model position
                if 'model_pos' in data:
                    ax.scatter(*data['model_pos'], color='red', s=100,
                               marker='*', label='Trained Model')

                ax.set_xlabel('Direction 1')
                ax.set_ylabel('Direction 2')
                ax.set_title('2D Loss Landscape')
                ax.legend()
            return fig

        elif plot_type == '3d':
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            if 'X' in data and 'Y' in data and 'Z' in data:
                surf = ax.plot_surface(data['X'], data['Y'], data['Z'],
                                       cmap='viridis', alpha=0.8)
                plt.colorbar(surf, ax=ax, label='Loss', shrink=0.5)

                ax.set_xlabel('Direction 1')
                ax.set_ylabel('Direction 2')
                ax.set_zlabel('Loss')
                ax.set_title('3D Loss Landscape')
            return fig

        raise ValueError("Invalid plot_type for LossLandscapeVisualization")


class AttentionVisualization(VisualizationPlugin):
    """Visualize attention weights in transformer models."""

    @property
    def name(self) -> str:
        return "attention"

    @property
    def description(self) -> str:
        return "Visualize attention patterns"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and 'attention_weights' in data

    def create_visualization(
            self,
            data: Dict[str, Any],
            ax: Optional[plt.Axes] = None,
            layer_idx: int = 0,
            head_idx: Optional[int] = None,
            **kwargs
    ) -> plt.Figure:
        """Create attention visualization."""

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()

        attention_weights = data['attention_weights']
        tokens = data.get('tokens', None)

        # Get attention for specific layer
        if isinstance(attention_weights, list):
            layer_attention = attention_weights[layer_idx]
        else:
            layer_attention = attention_weights

        # Average over heads if not specified
        if head_idx is None and len(layer_attention.shape) == 4:
            # Shape: (batch, heads, seq_len, seq_len)
            attention_matrix = np.mean(layer_attention[0], axis=0)
            title_suffix = f" (Layer {layer_idx}, Averaged Heads)"
        elif head_idx is not None and len(layer_attention.shape) == 4:
            attention_matrix = layer_attention[0, head_idx]
            title_suffix = f" (Layer {layer_idx}, Head {head_idx})"
        else:
            attention_matrix = layer_attention[0] if len(layer_attention.shape) == 3 else layer_attention
            title_suffix = ""

        # Plot attention heatmap
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='Attention Weight')

        # Set labels
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)

        ax.set_xlabel('Keys/Values')
        ax.set_ylabel('Queries')
        ax.set_title(f'Attention Weights{title_suffix}')

        return fig


class EmbeddingVisualization(VisualizationPlugin):
    """Visualize learned embeddings."""

    @property
    def name(self) -> str:
        return "embeddings"

    @property
    def description(self) -> str:
        return "Visualize learned embedding spaces"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and 'embeddings' in data

    def create_visualization(
            self,
            data: Dict[str, Any],
            ax: Optional[plt.Axes] = None,
            method: str = 'tsne',  # 'tsne', 'pca', 'umap'
            **kwargs
    ) -> plt.Figure:
        """Create embedding visualization."""

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = ax.get_figure()

        embeddings = data['embeddings']
        labels = data.get('labels', None)
        words = data.get('words', None)

        # Dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                # Fallback to PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(embeddings)
                method = 'pca'

        # Scatter plot
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[colors[i]], label=str(label), alpha=0.6, s=30)
            ax.legend()
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       alpha=0.6, s=30)

        # Add word labels if provided
        if words is not None:
            for i, word in enumerate(words[:100]):  # Limit to avoid clutter
                ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                            fontsize=8, alpha=0.7)

        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Embedding Visualization ({method.upper()})')
        ax.grid(True, alpha=0.3)

        return fig


# =============================================================================
# Complete Workflow Example
# =============================================================================

class MLExperimentWorkflow:
    """Complete ML experiment visualization workflow."""

    def __init__(self, experiment_name: str, output_dir: str = "visualizations"):
        """Initialize workflow."""

        self.experiment_name = experiment_name
        self.viz_manager = VisualizationManager(
            experiment_name=experiment_name,
            output_dir=output_dir
        )
        self._register_all_templates()

    def _register_all_templates(self):
        """Register all available templates."""

        # Training templates
        self.viz_manager.register_template("training_curves", TrainingCurvesVisualization)
        self.viz_manager.register_template("convergence_analysis", ConvergenceAnalysis) # FIX: Correct name
        self.viz_manager.register_template("overfitting_analysis", OverfittingAnalysis) # FIX: Correct name

        # Model comparison
        self.viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart) # FIX: Correct name
        self.viz_manager.register_template("performance_radar", PerformanceRadarChart)
        self.viz_manager.register_template("performance_dashboard", PerformanceDashboard)

        # Classification
        self.viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
        self.viz_manager.register_template("roc_pr_curves", ROCPRCurves) # FIX: Correct name
        self.viz_manager.register_template("classification_report", ClassificationReportVisualization)
        self.viz_manager.register_template("per_class", PerClassAnalysis)
        self.viz_manager.register_template("error_analysis", ErrorAnalysisDashboard)

        # Data analysis
        self.viz_manager.register_template("data_distribution", DataDistributionAnalysis)
        self.viz_manager.register_template("class_balance", ClassBalanceVisualization)

        # Neural network
        self.viz_manager.register_template("network_architecture", NetworkArchitectureVisualization) # FIX: Correct name
        self.viz_manager.register_template("activations", ActivationVisualization)
        self.viz_manager.register_template("weights", WeightVisualization)
        self.viz_manager.register_template("feature_maps", FeatureMapVisualization)
        self.viz_manager.register_template("gradients", GradientVisualization)

        # Specialized
        self.viz_manager.register_template("experiment_comparison", ExperimentComparisonDashboard)
        self.viz_manager.register_template("loss_landscape", LossLandscapeVisualization)
        self.viz_manager.register_template("attention", AttentionVisualization)
        self.viz_manager.register_template("embeddings", EmbeddingVisualization)

    def visualize_training(self, history: TrainingHistory, model_name: str = "Model"):
        """Visualize training process."""

        print(f"Creating training visualizations for {model_name}...")

        # Training curves
        self.viz_manager.visualize(history, "training_curves")

        # Convergence analysis
        self.viz_manager.visualize(history, "convergence_analysis")

        # Overfitting detection
        self.viz_manager.visualize(history, "overfitting_analysis")

        print("Training visualizations complete!")

    def visualize_model(self, model: keras.Model):
        """Visualize model architecture and weights."""

        print(f"Creating model visualizations...")

        # Architecture
        self.viz_manager.visualize(model, "network_architecture")

        # Weights
        self.viz_manager.visualize(model, "weights")

        print("Model visualizations complete!")

    def visualize_predictions(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: Optional[np.ndarray] = None,
            class_names: Optional[List[str]] = None
    ):
        """Visualize prediction results."""

        print("Creating prediction visualizations...")

        results = ClassificationResults(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names
        )

        # Confusion matrix
        self.viz_manager.visualize(results, "confusion_matrix")

        # ROC curves (if probabilities available)
        if y_prob is not None:
            self.viz_manager.visualize(results, "roc_pr_curves")

        # Classification report
        self.viz_manager.visualize(results, "classification_report")

        # Per-class analysis
        self.viz_manager.visualize(results, "per_class")

        # Error analysis
        self.viz_manager.visualize(results, "error_analysis")

        print("Prediction visualizations complete!")

    def compare_models(self, models_data: Dict[str, Dict[str, Any]]):
        """Compare multiple models."""

        print("Creating model comparison visualizations...")

        # Extract metrics for comparison
        model_names = list(models_data.keys())
        metrics = {}
        histories = {}

        for name, data in models_data.items():
            if 'metrics' in data:
                metrics[name] = data['metrics']
            if 'history' in data:
                histories[name] = data['history']

        if metrics:
            comparison = ModelComparison(
                model_names=model_names,
                metrics=metrics,
                histories=histories if histories else None
            )

            self.viz_manager.visualize(comparison, "model_comparison_bars")
            self.viz_manager.visualize(comparison, "performance_radar")
            self.viz_manager.visualize(comparison, "performance_dashboard")

        print("Model comparison visualizations complete!")

    def create_full_report(self, all_data: Dict[str, Any]):
        """Create comprehensive experiment report."""

        print("Creating full experiment report...")

        # Save metadata
        self.viz_manager.save_metadata({
            'experiment_name': self.experiment_name,
            'timestamp': str(pd.Timestamp.now()),
            'data_keys': list(all_data.keys())
        })

        # Create visualizations based on available data
        if 'training_history' in all_data:
            self.visualize_training(all_data['training_history'])

        if 'model' in all_data:
            self.visualize_model(all_data['model'])

        if 'predictions' in all_data:
            self.visualize_predictions(**all_data['predictions'])

        if 'model_comparison' in all_data:
            self.compare_models(all_data['model_comparison'])

        print(f"Full report saved to: {self.viz_manager.context.output_dir}")


# =============================================================================
# Main Example
# =============================================================================

def main_example():
    """Complete example demonstrating the visualization framework."""

    print("=" * 60)
    print("ML Visualization Framework - Complete Example")
    print("=" * 60)

    # Initialize workflow
    workflow = MLExperimentWorkflow("complete_example")

    # Generate example data
    np.random.seed(42)

    # 1. Training history
    n_epochs = 100
    history = TrainingHistory(
        epochs=list(range(n_epochs)),
        train_loss=np.exp(-np.linspace(0, 2, n_epochs)) + np.random.normal(0, 0.01, n_epochs),
        val_loss=np.exp(-np.linspace(0, 1.8, n_epochs)) + np.random.normal(0, 0.02, n_epochs),
        train_metrics={
            'accuracy': 1 - np.exp(-np.linspace(0, 3, n_epochs)) + np.random.normal(0, 0.01, n_epochs)
        },
        val_metrics={
            'accuracy': 1 - np.exp(-np.linspace(0, 2.5, n_epochs)) + np.random.normal(0, 0.02, n_epochs)
        }
    )

    # 2. Classification results
    n_samples = 1000
    n_classes = 10
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add realistic errors
    error_mask = np.random.random(n_samples) < 0.15
    y_pred[error_mask] = np.random.randint(0, n_classes, error_mask.sum())

    y_prob = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_prob[i] = np.random.dirichlet(np.ones(n_classes))
        y_prob[i, y_pred[i]] += 0.5  # Boost predicted class probability
        y_prob[i] /= y_prob[i].sum()

    # 3. Create all visualizations
    all_data = {
        'training_history': history,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'class_names': [f'Class_{i}' for i in range(n_classes)]
        }
    }

    workflow.create_full_report(all_data)

    print("\n" + "=" * 60)
    print("Example complete! Check the 'visualizations' directory.")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    main_example()