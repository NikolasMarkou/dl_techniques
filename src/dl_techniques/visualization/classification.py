"""
Classification and Evaluation Visualization Templates
======================================================

Ready-made templates for classification tasks, confusion matrices,
ROC curves, and model evaluation visualizations.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score
)

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .core import VisualizationPlugin, CompositeVisualization, PlotConfig, VisualizationContext


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class ClassificationResults:
    """Container for classification results."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    model_name: Optional[str] = None


@dataclass
class MultiModelClassification:
    """Container for comparing multiple classification models."""

    results: Dict[str, ClassificationResults]
    dataset_name: Optional[str] = None


# ---------------------------------------------------------------------
# Confusion Matrix Templates
# ---------------------------------------------------------------------

class ConfusionMatrixVisualization(VisualizationPlugin):
    """Enhanced confusion matrix visualization."""

    @property
    def name(self) -> str:
        return "confusion_matrix"

    @property
    def description(self) -> str:
        return "Visualize confusion matrices with detailed annotations"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (ClassificationResults, MultiModelClassification))

    def create_visualization(
            self,
            data: Union[ClassificationResults, MultiModelClassification],
            ax: Optional[plt.Axes] = None,
            normalize: str = 'true',  # 'true', 'pred', 'all', None
            show_percentages: bool = True,
            cmap: str = 'Blues',
            **kwargs
    ) -> plt.Figure:
        """
        Create confusion matrix visualization.

        Args:
            data: Classification results
            ax: Optional matplotlib axes to plot on.
            normalize: Normalization mode
            show_percentages: Show percentages in cells
            cmap: Colormap to use
        """
        if isinstance(data, ClassificationResults):
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
            else:
                fig = ax.get_figure()
            self._single_confusion_matrix(ax, data, normalize, show_percentages, cmap)
            plt.tight_layout()
            return fig
        else:
            return self._multi_confusion_matrix(data, normalize, show_percentages, cmap)

    def _single_confusion_matrix(
            self,
            ax: plt.Axes,
            results: ClassificationResults,
            normalize: str,
            show_percentages: bool,
            cmap: str
    ):
        """Create single confusion matrix on a given axis."""
        fig = ax.get_figure()

        # Compute confusion matrix
        cm = confusion_matrix(results.y_true, results.y_pred)

        # Normalize if requested
        if normalize == 'true':
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            title_suffix = ' (Row Normalized)'
        elif normalize == 'pred':
            cm_norm = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-10)
            title_suffix = ' (Column Normalized)'
        elif normalize == 'all':
            cm_norm = cm.astype('float') / (cm.sum() + 1e-10)
            title_suffix = ' (Overall Normalized)'
        else:
            cm_norm = cm.astype('float')
            title_suffix = ''

        # Plot heatmap
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        fig.colorbar(im, ax=ax)

        # Set ticks and labels
        classes = results.class_names if results.class_names else range(len(cm))
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)

        # Add text annotations
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if show_percentages and normalize:
                    text = f'{cm[i, j]}\n({cm_norm[i, j]:.1%})'
                else:
                    text = f'{cm[i, j]}'

                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=9)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        model_name = results.model_name if results.model_name else "Model"
        ax.set_title(f'Confusion Matrix - {model_name}{title_suffix}')

    def _multi_confusion_matrix(
            self,
            data: MultiModelClassification,
            normalize: str,
            show_percentages: bool,
            cmap: str
    ) -> plt.Figure:
        """Create multiple confusion matrices for comparison."""

        n_models = len(data.results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (model_name, results) in enumerate(data.results.items()):
            ax = axes[idx]
            self._single_confusion_matrix(ax, results, normalize, show_percentages, cmap)
            ax.set_title(model_name)  # Override default title with model name

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        dataset_name = data.dataset_name if data.dataset_name else "Dataset"
        plt.suptitle(f'Confusion Matrices Comparison - {dataset_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


# ---------------------------------------------------------------------
# ROC and PR Curve Templates
# ---------------------------------------------------------------------

class ROCPRCurves(VisualizationPlugin):
    """ROC and Precision-Recall curves visualization."""

    @property
    def name(self) -> str:
        return "roc_pr_curves"

    @property
    def description(self) -> str:
        return "Visualize ROC and Precision-Recall curves"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (ClassificationResults, MultiModelClassification))

    def create_visualization(
            self,
            data: Union[ClassificationResults, MultiModelClassification],
            ax: Optional[plt.Axes] = None,
            plot_type: str = 'both',  # 'roc', 'pr', 'both'
            show_thresholds: bool = False,
            **kwargs
    ) -> plt.Figure:
        """Create ROC and/or PR curves."""
        if ax is not None and plot_type == 'both':
            raise ValueError("Cannot plot 'both' ROC and PR curves on a single provided axis.")

        if plot_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            self._plot_roc_curves(axes[0], data, show_thresholds)
            self._plot_pr_curves(axes[1], data, show_thresholds)
        elif plot_type == 'roc':
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.get_figure()
            self._plot_roc_curves(ax, data, show_thresholds)
        else:  # pr
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.get_figure()
            self._plot_pr_curves(ax, data, show_thresholds)

        plt.suptitle('Model Performance Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def _plot_roc_curves(self, ax: plt.Axes, data: Any, show_thresholds: bool):
        """Plot ROC curves."""

        # Convert single result to multi-model format
        if isinstance(data, ClassificationResults):
            results_dict = {data.model_name or "Model": data}
        else:
            results_dict = data.results

        # Plot ROC for each model
        for idx, (model_name, results) in enumerate(results_dict.items()):
            if results.y_prob is None:
                continue

            # Handle multi-class
            if len(results.y_prob.shape) == 2 and results.y_prob.shape[1] > 2:
                # Use micro-average for multi-class
                from sklearn.preprocessing import label_binarize
                n_classes = results.y_prob.shape[1]
                y_true_bin = label_binarize(results.y_true, classes=range(n_classes))
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), results.y_prob.ravel())
            else:
                # Binary classification
                y_prob_pos = results.y_prob[:, 1] if len(results.y_prob.shape) == 2 else results.y_prob
                fpr, tpr, thresholds = roc_curve(results.y_true, y_prob_pos)

            roc_auc = auc(fpr, tpr)

            color = self.config.color_scheme.get_model_color(model_name, idx)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')

            # Mark thresholds if requested
            if show_thresholds and 'thresholds' in locals():
                n_points = min(10, len(thresholds))
                indices = np.linspace(0, len(thresholds) - 1, n_points, dtype=int)
                ax.scatter(fpr[indices], tpr[indices], color=color, s=30, alpha=0.6)
                for i in indices[::2]:  # Show every other threshold
                    ax.annotate(f'{thresholds[i]:.2f}',
                                (fpr[i], tpr[i]), fontsize=7)

        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    def _plot_pr_curves(self, ax: plt.Axes, data: Any, show_thresholds: bool):
        """Plot Precision-Recall curves."""

        # Convert single result to multi-model format
        if isinstance(data, ClassificationResults):
            results_dict = {data.model_name or "Model": data}
        else:
            results_dict = data.results

        for idx, (model_name, results) in enumerate(results_dict.items()):
            if results.y_prob is None:
                continue

            # Handle multi-class
            if len(results.y_prob.shape) == 2 and results.y_prob.shape[1] > 2:
                # Use micro-average for multi-class
                from sklearn.preprocessing import label_binarize
                n_classes = results.y_prob.shape[1]
                y_true_bin = label_binarize(results.y_true, classes=range(n_classes))
                precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), results.y_prob.ravel())
            else:
                # Binary classification
                y_prob_pos = results.y_prob[:, 1] if len(results.y_prob.shape) == 2 else results.y_prob
                precision, recall, thresholds = precision_recall_curve(results.y_true, y_prob_pos)

            pr_auc = auc(recall, precision)

            color = self.config.color_scheme.get_model_color(model_name, idx)
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f'{model_name} (AUC = {pr_auc:.3f})')

            # Mark thresholds if requested
            if show_thresholds and 'thresholds' in locals():
                n_points = min(10, len(thresholds))
                indices = np.linspace(0, len(thresholds) - 1, n_points, dtype=int)
                ax.scatter(recall[indices], precision[indices],
                           color=color, s=30, alpha=0.6)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------
# Classification Report Visualization
# ---------------------------------------------------------------------

class ClassificationReportVisualization(VisualizationPlugin):
    """Visual representation of classification reports."""

    @property
    def name(self) -> str:
        return "classification_report"

    @property
    def description(self) -> str:
        return "Visualize classification reports as heatmaps"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (ClassificationResults, MultiModelClassification))

    def create_visualization(
            self,
            data: Union[ClassificationResults, MultiModelClassification],
            ax: Optional[plt.Axes] = None,
            metrics: List[str] = ['precision', 'recall', 'f1-score'],
            **kwargs
    ) -> plt.Figure:
        """Create classification report visualization."""

        if isinstance(data, ClassificationResults):
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = ax.get_figure()
            self._single_report(ax, data, metrics)
            plt.tight_layout()
            return fig
        else:
            return self._multi_report(data, metrics)

    def _single_report(self, ax: plt.Axes, results: ClassificationResults, metrics: List[str]):
        """Create single classification report on a given axis."""

        # Get classification report as dict
        report = classification_report(
            results.y_true, results.y_pred,
            target_names=results.class_names,
            output_dict=True
        )

        # Convert to dataframe
        df = pd.DataFrame(report).transpose()

        # Filter metrics and classes
        classes = results.class_names if results.class_names else [str(i) for i in range(max(results.y_true) + 1)]
        df_filtered = df.loc[classes, metrics]

        # Plot heatmap
        sns.heatmap(df_filtered.astype(float), annot=True, fmt='.2f',
                    cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': 'Score'})

        ax.set_title(f'Classification Report - {results.model_name or "Model"}')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')

    def _multi_report(self, data: MultiModelClassification, metrics: List[str]) -> plt.Figure:
        """Create comparison of classification reports."""

        n_models = len(data.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        vmin, vmax = 1, 0  # For consistent color scale

        # First pass to get value range
        for results in data.results.values():
            report = classification_report(
                results.y_true, results.y_pred,
                target_names=results.class_names,
                output_dict=True
            )
            df = pd.DataFrame(report).transpose()
            classes = results.class_names if results.class_names else [str(i) for i in range(max(results.y_true) + 1)]
            df_filtered = df.loc[classes, metrics]
            vmin = min(vmin, df_filtered.min().min())
            vmax = max(vmax, df_filtered.max().max())

        # Plot each model's report
        for idx, (model_name, results) in enumerate(data.results.items()):
            ax = axes[idx]
            self._single_report(ax, results, metrics)
            ax.set_title(model_name)
            ax.set_xlabel('Metrics' if idx == 0 else '')
            ax.set_ylabel('Classes' if idx == 0 else '')

        plt.suptitle('Classification Reports Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------
# Per-Class Performance Analysis
# ---------------------------------------------------------------------

class PerClassAnalysis(CompositeVisualization):
    """Detailed per-class performance analysis."""

    @property
    def name(self) -> str:
        return "per_class_analysis"

    @property
    def description(self) -> str:
        return "Analyze performance for each class separately"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (ClassificationResults, MultiModelClassification))

    def __init__(self, config: PlotConfig, context: VisualizationContext):
        super().__init__(config, context)
        self.add_subplot("Class Distribution", self._plot_class_distribution)
        self.add_subplot("Per-Class Accuracy", self._plot_per_class_accuracy)
        self.add_subplot("Class Confusion", self._plot_class_confusion)
        self.add_subplot("Hardest Examples", self._plot_hardest_examples)

    def _plot_class_distribution(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot distribution of classes in predictions vs true."""

        if isinstance(data, ClassificationResults):
            results = data
        else:
            # Use first model for multi-model data
            results = list(data.results.values())[0]

        classes = results.class_names if results.class_names else range(
            max(max(results.y_true), max(results.y_pred)) + 1)

        # Count occurrences
        true_counts = np.bincount(results.y_true)
        pred_counts = np.bincount(results.y_pred)

        x = np.arange(len(classes))
        width = 0.35

        ax.bar(x - width / 2, true_counts[:len(classes)], width, label='True', alpha=0.8)
        ax.bar(x + width / 2, pred_counts[:len(classes)], width, label='Predicted', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_per_class_accuracy(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot accuracy for each class."""

        if isinstance(data, MultiModelClassification):
            # Multiple models
            for idx, (model_name, results) in enumerate(data.results.items()):
                accuracies = self._calculate_per_class_accuracy(results)
                classes = results.class_names if results.class_names else range(len(accuracies))
                color = self.config.color_scheme.get_model_color(model_name, idx)
                ax.plot(range(len(accuracies)), accuracies, 'o-',
                        label=model_name, color=color, linewidth=2)
        else:
            # Single model
            accuracies = self._calculate_per_class_accuracy(data)
            classes = data.class_names if data.class_names else range(len(accuracies))
            ax.bar(range(len(accuracies)), accuracies, alpha=0.8)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha='right')

        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _calculate_per_class_accuracy(self, results: ClassificationResults) -> List[float]:
        """Calculate accuracy for each class."""

        n_classes = max(max(results.y_true), max(results.y_pred)) + 1
        accuracies = []

        for class_id in range(n_classes):
            mask = results.y_true == class_id
            if mask.sum() > 0:
                correct = (results.y_pred[mask] == class_id).sum()
                accuracy = correct / mask.sum()
                accuracies.append(accuracy)
            else:
                accuracies.append(0)

        return accuracies

    def _plot_class_confusion(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot which classes are confused with each other."""

        if isinstance(data, ClassificationResults):
            results = data
        else:
            # Use first model
            results = list(data.results.values())[0]

        # Get confusion matrix
        cm = confusion_matrix(results.y_true, results.y_pred)

        # Remove diagonal (correct predictions)
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        # Find most confused pairs
        confused_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm_no_diag[i, j] > 0:
                    confused_pairs.append((i, j, cm_no_diag[i, j]))

        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        # Plot top confused pairs
        top_n = min(10, len(confused_pairs))
        if top_n > 0:
            pairs_labels = [f'{p[0]}→{p[1]}' for p in confused_pairs[:top_n]]
            values = [p[2] for p in confused_pairs[:top_n]]

            ax.barh(range(top_n), values, alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(pairs_labels)
            ax.set_xlabel('Number of Confusions')
            ax.set_title('Most Confused Class Pairs')
        else:
            ax.text(0.5, 0.5, 'No confusions found',
                    ha='center', va='center', transform=ax.transAxes)

        ax.grid(True, alpha=0.3, axis='x')

    def _plot_hardest_examples(self, ax: plt.Axes, data: Any, **kwargs):
        """Identify hardest examples based on prediction confidence."""

        if isinstance(data, ClassificationResults):
            results = data
        else:
            results = list(data.results.values())[0]

        if results.y_prob is not None:
            # Get maximum probability for each sample
            max_probs = results.y_prob.max(axis=1) if len(results.y_prob.shape) == 2 else results.y_prob

            # Find incorrectly classified samples
            incorrect_mask = results.y_true != results.y_pred

            if incorrect_mask.sum() > 0:
                # Get confidence for incorrect predictions
                incorrect_confidence = max_probs[incorrect_mask]

                # Plot distribution
                ax.hist(incorrect_confidence, bins=20, alpha=0.8, edgecolor='black')
                ax.set_xlabel('Prediction Confidence')
                ax.set_ylabel('Number of Misclassified Samples')
                ax.set_title('Confidence Distribution of Errors')
                ax.axvline(x=incorrect_confidence.mean(), color='r',
                           linestyle='--', label=f'Mean: {incorrect_confidence.mean():.2f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No misclassifications found',
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No probability data available',
                    ha='center', va='center', transform=ax.transAxes)

        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------
# Error Analysis Dashboard
# ---------------------------------------------------------------------

class ErrorAnalysisDashboard(VisualizationPlugin):
    """Comprehensive error analysis dashboard."""

    @property
    def name(self) -> str:
        return "error_analysis"

    @property
    def description(self) -> str:
        return "Detailed analysis of prediction errors"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, ClassificationResults)

    def create_visualization(
            self,
            data: ClassificationResults,
            ax: Optional[plt.Axes] = None,
            show_examples: bool = False,
            x_data: Optional[np.ndarray] = None,
            **kwargs
    ) -> plt.Figure:
        """Create error analysis dashboard."""
        if ax is not None:
            # This complex dashboard cannot render on a single provided axis.
            # It will create its own figure.
            pass

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Error rate by class
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_error_rate_by_class(ax1, data)

        # 2. Confidence analysis for errors
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_error_confidence(ax2, data)

        # 3. Error types
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_error_types(ax3, data)

        # 4. Confusion hotspots
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_confusion_hotspots(ax4, data)

        # 5. Error examples (if data provided)
        if show_examples and x_data is not None:
            ax5_container = fig.add_subplot(gs[2, :])
            self._plot_error_examples(fig, ax5_container, data, x_data)
        else:
            # Summary statistics
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_error_summary(ax5, data)

        model_name = data.model_name if data.model_name else "Model"
        plt.suptitle(f'Error Analysis - {model_name}', fontsize=16, fontweight='bold')

        return fig

    def _plot_error_rate_by_class(self, ax: plt.Axes, data: ClassificationResults):
        """Plot error rate for each class."""

        n_classes = max(max(data.y_true), max(data.y_pred)) + 1
        error_rates = []
        support = []

        for class_id in range(n_classes):
            mask = data.y_true == class_id
            if mask.sum() > 0:
                errors = (data.y_pred[mask] != class_id).sum()
                error_rate = errors / mask.sum()
                error_rates.append(error_rate)
                support.append(mask.sum())
            else:
                error_rates.append(0)
                support.append(0)

        classes = data.class_names if data.class_names else range(n_classes)
        x = np.arange(len(classes))

        # Color bars by error rate
        colors = ['red' if e > 0.5 else 'orange' if e > 0.2 else 'green'
                  for e in error_rates]

        bars = ax.bar(x, error_rates, color=colors, alpha=0.7, edgecolor='black')

        # Add support as text
        for i, (bar, sup) in enumerate(zip(bars, support)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'n={sup}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Rate by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_error_confidence(self, ax: plt.Axes, data: ClassificationResults):
        """Analyze confidence of errors."""

        if data.y_prob is None:
            ax.text(0.5, 0.5, 'No probability data',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Get prediction confidence
        max_probs = data.y_prob.max(axis=1) if len(data.y_prob.shape) == 2 else data.y_prob

        # Split by correct/incorrect
        correct_mask = data.y_true == data.y_pred

        ax.hist([max_probs[correct_mask], max_probs[~correct_mask]],
                bins=20, label=['Correct', 'Incorrect'],
                alpha=0.7, color=['green', 'red'])

        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_error_types(self, ax: plt.Axes, data: ClassificationResults):
        """Categorize types of errors."""

        # Simple categorization based on confidence
        if data.y_prob is not None:
            max_probs = data.y_prob.max(axis=1) if len(data.y_prob.shape) == 2 else data.y_prob
            incorrect_mask = data.y_true != data.y_pred

            if incorrect_mask.sum() > 0:
                incorrect_conf = max_probs[incorrect_mask]

                # Categorize errors
                high_conf_errors = (incorrect_conf > 0.8).sum()
                med_conf_errors = ((incorrect_conf > 0.5) & (incorrect_conf <= 0.8)).sum()
                low_conf_errors = (incorrect_conf <= 0.5).sum()

                categories = ['High Conf\n(>0.8)', 'Med Conf\n(0.5-0.8)', 'Low Conf\n(≤0.5)']
                values = [high_conf_errors, med_conf_errors, low_conf_errors]
                colors = ['darkred', 'orange', 'yellow']

                ax.pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
                ax.set_title('Error Types by Confidence')
            else:
                ax.text(0.5, 0.5, 'No errors found',
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No probability data',
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_confusion_hotspots(self, ax: plt.Axes, data: ClassificationResults):
        """Show hotspots in confusion matrix."""

        cm = confusion_matrix(data.y_true, data.y_pred)

        # Normalize by row (true class)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Set diagonal to 0 to focus on errors
        np.fill_diagonal(cm_norm, 0)

        # Plot heatmap
        im = ax.imshow(cm_norm, cmap='Reds', aspect='auto')

        classes = data.class_names if data.class_names else range(cm.shape[0])
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Error Hotspots')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_error_examples(
            self,
            fig: plt.Figure,
            ax_container: plt.Axes,
            data: ClassificationResults,
            x_data: np.ndarray
    ):
        """Show examples of errors within a container axis."""
        incorrect_mask = data.y_true != data.y_pred
        incorrect_indices = np.where(incorrect_mask)[0]

        ax_container.set_title('Error Examples')
        ax_container.axis('off')

        if len(incorrect_indices) > 0:
            # Show up to 10 examples
            n_examples = min(10, len(incorrect_indices))

            # Create a nested GridSpec within the container axis
            gs_sub = GridSpecFromSubplotSpec(1, n_examples, subplot_spec=ax_container.get_subplotspec())

            for i in range(n_examples):
                idx = incorrect_indices[i]
                ax_sub = fig.add_subplot(gs_sub[i])

                # Assuming image data
                if len(x_data.shape) == 4:  # (N, H, W, C)
                    if x_data.shape[3] == 1:
                        ax_sub.imshow(x_data[idx, :, :, 0], cmap='gray')
                    else:
                        ax_sub.imshow(x_data[idx])
                elif len(x_data.shape) == 3:  # (N, H, W)
                    ax_sub.imshow(x_data[idx], cmap='gray')

                true_label = data.class_names[data.y_true[idx]] if data.class_names else data.y_true[idx]
                pred_label = data.class_names[data.y_pred[idx]] if data.class_names else data.y_pred[idx]

                ax_sub.set_title(f'T:{true_label}\nP:{pred_label}', fontsize=8)
                ax_sub.axis('off')
        else:
            ax_container.text(0.5, 0.5, 'No errors found',
                              ha='center', va='center', transform=ax_container.transAxes)

    def _plot_error_summary(self, ax: plt.Axes, data: ClassificationResults):
        """Show error summary statistics."""

        # Calculate metrics
        total_samples = len(data.y_true)
        total_errors = (data.y_true != data.y_pred).sum()
        accuracy = accuracy_score(data.y_true, data.y_pred)

        # Per-class errors
        n_classes = max(max(data.y_true), max(data.y_pred)) + 1
        class_errors = []
        for class_id in range(n_classes):
            mask = data.y_true == class_id
            if mask.sum() > 0:
                errors = (data.y_pred[mask] != class_id).sum()
                class_errors.append((class_id, errors, mask.sum()))

        # Sort by error count
        class_errors.sort(key=lambda x: x[1], reverse=True)

        # Create summary text
        summary_text = f"""
        ERROR SUMMARY
        =============
        Total Samples: {total_samples}
        Total Errors: {total_errors}
        Error Rate: {total_errors / total_samples:.2%}
        Accuracy: {accuracy:.2%}

        Top Error Classes:
        """

        for i, (class_id, errors, support) in enumerate(class_errors[:5]):
            class_name = data.class_names[class_id] if data.class_names else f"Class {class_id}"
            summary_text += f"\n  {i + 1}. {class_name}: {errors}/{support} errors ({errors / support:.1%})"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', family='monospace')
        ax.axis('off')