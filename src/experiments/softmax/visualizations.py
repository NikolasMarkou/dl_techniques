"""
Enhanced Visualizations for Softmax Decision Boundary Experiment
=============================================================

This module provides additional visualization functions for analyzing model behavior,
decision boundaries, and prediction confidence in classification tasks.

These visualizations can be integrated into the existing Softmax Decision Boundary Experiment
to provide deeper insights into model behavior and performance.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import keras

# Visualization parameters
CONFIDENCE_CMAP = plt.cm.RdYlGn  # Red-Yellow-Green colormap for confidence
BOUNDARY_CMAP = plt.cm.RdBu  # Red-Blue colormap for decision boundaries
FIGURE_SIZE_LARGE = (16, 12)
FIGURE_SIZE_MEDIUM = (12, 10)
FIGURE_SIZE_SMALL = (10, 8)
DPI = 300
SCATTER_POINT_SIZE = 30
GRID_ALPHA = 0.2
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12


class EnhancedVisualizations:
    """Enhanced visualization methods for classification models."""

    @staticmethod
    def visualize_confidence(
            model: keras.Model,
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Model Confidence",
            plot_points: bool = True,
            grid_resolution: float = 0.05,
            fig_size: Tuple[int, int] = FIGURE_SIZE_MEDIUM,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize model prediction confidence across the feature space.

        Args:
            model: Trained Keras model
            X: Feature matrix of shape (n_samples, 2)
            y: True labels of shape (n_samples,)
            title: Plot title
            plot_points: Whether to plot data points
            grid_resolution: Resolution of the grid for confidence visualization
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        # Create meshgrid for visualization
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_resolution),
            np.arange(y_min, y_max, grid_resolution)
        )

        # Get predictions for the grid points
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(grid_points, verbose=0)

        # Get confidence scores
        if predictions.shape[1] > 1:  # Multi-class with softmax
            # Get the highest probability for each point
            confidence = np.max(predictions, axis=1)
            # Get the predicted class
            pred_class = np.argmax(predictions, axis=1)
            # Map confidence to [0, 1] based on the predicted class
            # If class 0, invert confidence so red=class 0, green=class 1
            confidence_mapped = np.where(pred_class == 0, 1 - confidence, confidence)
        else:  # Binary with sigmoid
            # Sigmoid output is already a probability for class 1
            confidence = predictions.flatten()
            # Map to [0, 1] range where 0.5 is the decision boundary
            confidence_mapped = confidence

        # Reshape for plotting
        confidence_mapped = confidence_mapped.reshape(xx.shape)

        # Create plot
        plt.figure(figsize=fig_size)

        # Plot the confidence heatmap
        plt.contourf(xx, yy, confidence_mapped, cmap=CONFIDENCE_CMAP, alpha=0.8, levels=50)

        # Plot the decision boundary as a contour line
        if predictions.shape[1] > 1:  # Multi-class with softmax
            Z = pred_class.reshape(xx.shape)
        else:  # Binary with sigmoid
            Z = (confidence > 0.5).astype(int).reshape(xx.shape)

        plt.contour(xx, yy, Z, colors=['k'], linewidths=2, levels=[0.5])

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Confidence (class 1)', fontsize=FONT_SIZE_LABEL)

        # Plot the original data points if requested
        if plot_points:
            plt.scatter(
                X[:, 0], X[:, 1],
                c=y, cmap=BOUNDARY_CMAP,
                s=SCATTER_POINT_SIZE, edgecolors='k',
                alpha=0.6
            )

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)
        plt.grid(alpha=GRID_ALPHA)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_confusion_matrix(
            model: keras.Model,
            X: np.ndarray,
            y: np.ndarray,
            class_names: List[str] = ['Class 0', 'Class 1'],
            title: str = "Confusion Matrix",
            normalize: bool = True,
            fig_size: Tuple[int, int] = FIGURE_SIZE_SMALL,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the confusion matrix for model predictions.

        Args:
            model: Trained Keras model
            X: Feature matrix
            y: True labels
            class_names: Names of the classes
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        # Get predictions
        y_pred_proba = model.predict(X, verbose=0)

        # Convert to class predictions
        if y_pred_proba.shape[1] > 1:  # Multi-class with softmax
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:  # Binary with sigmoid
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create plot
        plt.figure(figsize=fig_size)

        # Plot confusion matrix
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.ylabel('True Label', fontsize=FONT_SIZE_LABEL)
        plt.xlabel('Predicted Label', fontsize=FONT_SIZE_LABEL)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_model_comparison(
            models: Dict[str, keras.Model],
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Model Comparison",
            grid_resolution: float = 0.05,
            fig_size: Optional[Tuple[int, int]] = None,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create a side-by-side comparison of multiple models' decision boundaries.

        Args:
            models: Dictionary of model names and their Keras model instances
            X: Feature matrix of shape (n_samples, 2)
            y: True labels of shape (n_samples,)
            title: Main plot title
            grid_resolution: Resolution of the grid for visualization
            fig_size: Figure size as (width, height), if None calculated based on number of models
            save_path: Path to save the figure (None to not save)
        """
        n_models = len(models)

        # Calculate figure size if not provided
        if fig_size is None:
            width = min(16, 5 * n_models)  # Limit width to 16 inches
            fig_size = (width, 5)

        # Create meshgrid for visualization
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_resolution),
            np.arange(y_min, y_max, grid_resolution)
        )

        # Create the figure and subplots
        fig, axes = plt.subplots(1, n_models, figsize=fig_size)

        # Handle the case of a single model
        if n_models == 1:
            axes = [axes]

        # Plot each model
        for i, (name, model) in enumerate(models.items()):
            # Get predictions for the grid points
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            predictions = model.predict(grid_points, verbose=0)

            # Get class predictions
            if predictions.shape[1] > 1:  # Multi-class with softmax
                Z = np.argmax(predictions, axis=1)
            else:  # Binary with sigmoid
                Z = (predictions > 0.5).astype(int).flatten()

            # Reshape for plotting
            Z = Z.reshape(xx.shape)

            # Plot decision boundary
            axes[i].contourf(xx, yy, Z, cmap=BOUNDARY_CMAP, alpha=0.8)

            # Plot the original data points
            axes[i].scatter(
                X[:, 0], X[:, 1],
                c=y, cmap=BOUNDARY_CMAP,
                s=SCATTER_POINT_SIZE, edgecolors='k',
                alpha=0.6
            )

            # Set title and labels
            axes[i].set_title(name, fontsize=FONT_SIZE_LABEL)
            axes[i].set_xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)

            # Only show y-axis label for the first subplot
            if i == 0:
                axes[i].set_ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)

            # Add grid
            axes[i].grid(alpha=GRID_ALPHA)

        # Set overall title
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

        # Save figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_roc_curve(
            models: Dict[str, keras.Model],
            X: np.ndarray,
            y: np.ndarray,
            title: str = "ROC Curve Comparison",
            fig_size: Tuple[int, int] = FIGURE_SIZE_SMALL,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize ROC curves for multiple models.

        Args:
            models: Dictionary of model names and their Keras model instances
            X: Feature matrix
            y: True labels
            title: Plot title
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        plt.figure(figsize=fig_size)

        # Plot ROC curve for each model
        for name, model in models.items():
            # Get predictions
            y_pred_proba = model.predict(X, verbose=0)

            # Convert to class 1 probability
            if y_pred_proba.shape[1] > 1:  # Multi-class with softmax
                y_pred_proba_1 = y_pred_proba[:, 1]  # Probability of class 1
            else:  # Binary with sigmoid
                y_pred_proba_1 = y_pred_proba.flatten()

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba_1)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                label=f'{name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('False Positive Rate', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('True Positive Rate', fontsize=FONT_SIZE_LABEL)
        plt.legend(loc='lower right')
        plt.grid(alpha=GRID_ALPHA)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_precision_recall(
            models: Dict[str, keras.Model],
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Precision-Recall Curve Comparison",
            fig_size: Tuple[int, int] = FIGURE_SIZE_SMALL,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize precision-recall curves for multiple models.

        Args:
            models: Dictionary of model names and their Keras model instances
            X: Feature matrix
            y: True labels
            title: Plot title
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        plt.figure(figsize=fig_size)

        # Plot precision-recall curve for each model
        for name, model in models.items():
            # Get predictions
            y_pred_proba = model.predict(X, verbose=0)

            # Convert to class 1 probability
            if y_pred_proba.shape[1] > 1:  # Multi-class with softmax
                y_pred_proba_1 = y_pred_proba[:, 1]  # Probability of class 1
            else:  # Binary with sigmoid
                y_pred_proba_1 = y_pred_proba.flatten()

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y, y_pred_proba_1)
            pr_auc = auc(recall, precision)

            # Plot precision-recall curve
            plt.plot(
                recall, precision,
                label=f'{name} (AUC = {pr_auc:.3f})',
                linewidth=2
            )

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Recall', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Precision', fontsize=FONT_SIZE_LABEL)
        plt.legend(loc='best')
        plt.grid(alpha=GRID_ALPHA)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_feature_space_transformation(
            model: keras.Model,
            X: np.ndarray,
            y: np.ndarray,
            layer_indices: Optional[List[int]] = None,
            title: str = "Feature Space Transformation",
            fig_size: Optional[Tuple[int, int]] = None,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize how the feature space is transformed through the layers of the model.

        Args:
            model: Trained Keras model
            X: Feature matrix of shape (n_samples, 2)
            y: True labels of shape (n_samples,)
            layer_indices: Indices of layers to visualize (None for all layers)
            title: Main plot title
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        # Identify which layers to visualize (only layers with 2D outputs can be visualized)
        if layer_indices is None:
            # Get all intermediate layers that might be interesting to visualize
            layer_indices = []
            for i, layer in enumerate(model.layers):
                # Skip input layer and layers without weights
                if i == 0 or len(layer.weights) == 0:
                    continue

                # Skip non-dense layers like dropout
                if not isinstance(layer, keras.layers.Dense):
                    continue

                layer_indices.append(i)

            # Add the final layer
            layer_indices.append(len(model.layers) - 1)

        # Count how many layers we'll visualize
        n_layers = len(layer_indices) + 1  # +1 for the input

        # Set figure size if not provided
        if fig_size is None:
            # Calculate width based on number of layers to visualize
            width = min(16, 4 * n_layers)  # Limit width to 16 inches
            height = 4
            fig_size = (width, height)

        # Create figure and subplots
        fig, axes = plt.subplots(1, n_layers, figsize=fig_size)

        # Handle case of a single subplot
        if n_layers == 1:
            axes = [axes]

        # Plot the input features
        axes[0].scatter(
            X[:, 0], X[:, 1],
            c=y, cmap=BOUNDARY_CMAP,
            s=SCATTER_POINT_SIZE, edgecolors='k',
            alpha=0.8
        )
        axes[0].set_title('Input Features', fontsize=FONT_SIZE_LABEL)
        axes[0].set_xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
        axes[0].set_ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)
        axes[0].grid(alpha=GRID_ALPHA)

        # Create intermediate models to get layer outputs
        for i, layer_idx in enumerate(layer_indices):
            # Create a model that outputs the layer activations
            layer_model = keras.Model(
                inputs=model.input,
                outputs=model.layers[layer_idx].output
            )

            # Get layer output for the input data
            layer_output = layer_model.predict(X, verbose=0)

            # Check if we can visualize this layer (need 2D output)
            if layer_output.shape[1] < 2:
                # Skip if not enough dimensions to visualize
                axes[i + 1].text(
                    0.5, 0.5,
                    f"Layer {layer_idx}\nOutput dim: {layer_output.shape[1]}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[i + 1].transAxes
                )
                continue

            if layer_output.shape[1] > 2:
                # If more than 2D, use PCA to reduce to 2D for visualization
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                layer_output_2d = pca.fit_transform(layer_output)
                dim_note = f"\nPCA from {layer_output.shape[1]}D"
            else:
                # Already 2D
                layer_output_2d = layer_output
                dim_note = ""

            # Plot the transformed features
            axes[i + 1].scatter(
                layer_output_2d[:, 0], layer_output_2d[:, 1],
                c=y, cmap=BOUNDARY_CMAP,
                s=SCATTER_POINT_SIZE, edgecolors='k',
                alpha=0.8
            )
            axes[i + 1].set_title(f'Layer {layer_idx}{dim_note}', fontsize=FONT_SIZE_LABEL)
            axes[i + 1].set_xlabel('Component 1', fontsize=FONT_SIZE_LABEL)
            axes[i + 1].set_ylabel('Component 2', fontsize=FONT_SIZE_LABEL)
            axes[i + 1].grid(alpha=GRID_ALPHA)

        # Set overall title
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

        # Save figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=DPI)

        plt.show()


def integrate_with_experiment(experiment_class):
    """
    Extend the provided experiment class with enhanced visualizations.

    Args:
        experiment_class: The Experiment class to extend

    Returns:
        Enhanced experiment class with additional visualization methods
    """

    # Add enhanced visualization methods to the experiment class
    class EnhancedExperiment(experiment_class):
        def visualize_model_confidence(self, model_name: Optional[str] = None) -> None:
            """
            Visualize confidence of a model or all models.

            Args:
                model_name: Name of the model to visualize (None for all models)
            """
            if self.results['dataset'] is None:
                print("No dataset available. Run the experiment first.")
                return

            if not self.results['models']:
                print("No models available. Run the experiment first.")
                return

            X = self.results['dataset']['X']
            y = self.results['dataset']['y']

            if model_name is not None:
                # Visualize a specific model
                if model_name not in self.results['models']:
                    print(f"Model '{model_name}' not found.")
                    return

                model = self.results['models'][model_name]
                EnhancedVisualizations.visualize_confidence(
                    model, X, y,
                    title=f"{model_name} Prediction Confidence",
                    save_path=f"{model_name.lower()}_confidence.png"
                )
            else:
                # Visualize all models
                for name, model in self.results['models'].items():
                    EnhancedVisualizations.visualize_confidence(
                        model, X, y,
                        title=f"{name} Prediction Confidence",
                        save_path=f"{name.lower()}_confidence.png"
                    )

        def visualize_confusion_matrices(self) -> None:
            """Visualize confusion matrices for all models."""
            if self.results['dataset'] is None or not self.results['models']:
                print("No dataset or models available. Run the experiment first.")
                return

            X_test = self.results['dataset']['X_test']
            y_test = self.results['dataset']['y_test']

            for name, model in self.results['models'].items():
                EnhancedVisualizations.visualize_confusion_matrix(
                    model, X_test, y_test,
                    title=f"{name} Confusion Matrix",
                    save_path=f"{name.lower()}_confusion_matrix.png"
                )

        def visualize_roc_curves(self) -> None:
            """Visualize ROC curves for all models."""
            if self.results['dataset'] is None or not self.results['models']:
                print("No dataset or models available. Run the experiment first.")
                return

            X_test = self.results['dataset']['X_test']
            y_test = self.results['dataset']['y_test']

            EnhancedVisualizations.visualize_roc_curve(
                self.results['models'], X_test, y_test,
                save_path="roc_curves.png"
            )

        def visualize_precision_recall_curves(self) -> None:
            """Visualize precision-recall curves for all models."""
            if self.results['dataset'] is None or not self.results['models']:
                print("No dataset or models available. Run the experiment first.")
                return

            X_test = self.results['dataset']['X_test']
            y_test = self.results['dataset']['y_test']

            EnhancedVisualizations.visualize_precision_recall(
                self.results['models'], X_test, y_test,
                save_path="precision_recall_curves.png"
            )

        def visualize_models_comparison(self) -> None:
            """Visualize a direct comparison of all model decision boundaries."""
            if self.results['dataset'] is None or not self.results['models']:
                print("No dataset or models available. Run the experiment first.")
                return

            X = self.results['dataset']['X']
            y = self.results['dataset']['y']

            EnhancedVisualizations.visualize_model_comparison(
                self.results['models'], X, y,
                title="Model Decision Boundaries Comparison",
                save_path="model_comparison.png"
            )

        def visualize_feature_transformation(self, model_name: str) -> None:
            """
            Visualize how the feature space is transformed through a model's layers.

            Args:
                model_name: Name of the model to visualize
            """
            if self.results['dataset'] is None:
                print("No dataset available. Run the experiment first.")
                return

            if model_name not in self.results['models']:
                print(f"Model '{model_name}' not found.")
                return

            X = self.results['dataset']['X']
            y = self.results['dataset']['y']
            model = self.results['models'][model_name]

            EnhancedVisualizations.visualize_feature_space_transformation(
                model, X, y,
                title=f"{model_name} Feature Space Transformation",
                save_path=f"{model_name.lower()}_feature_transformation.png"
            )

        def visualize_advanced(self) -> None:
            """Run all advanced visualizations."""
            print("Generating model confidence visualizations...")
            self.visualize_model_confidence()

            print("Generating confusion matrices...")
            self.visualize_confusion_matrices()

            print("Generating ROC curves...")
            self.visualize_roc_curves()

            print("Generating precision-recall curves...")
            self.visualize_precision_recall_curves()

            print("Generating model comparison visualization...")
            self.visualize_models_comparison()

            print("Generating feature transformation visualizations...")
            for name in self.results['models'].keys():
                self.visualize_feature_transformation(name)

            print("All visualizations completed.")

        def visualize_all_enhanced(self) -> None:
            """Run all basic and advanced visualizations."""
            print("Running basic visualizations...")
            self.visualize_all()

            print("Running advanced visualizations...")
            self.visualize_advanced()

    return EnhancedExperiment


def main():
    """Demonstrate usage of enhanced visualizations."""
    print("Enhanced visualizations module loaded.")
    print("To use with your experiment, add the following code:")
    print("\nfrom enhanced_visualizations import integrate_with_experiment")
    print("EnhancedExperiment = integrate_with_experiment(Experiment)")
    print("experiment = EnhancedExperiment()")
    print("...")
    print("# After running your experiment")
    print("experiment.visualize_all_enhanced()")


if __name__ == "__main__":
    main()