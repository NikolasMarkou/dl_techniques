"""
CCNet MNIST Experiment using the Visualization Framework.

This module demonstrates a complete training and analysis workflow for a
Counter-Causal Net (CCNet) on the MNIST dataset, fully integrated with the
visualization framework.

Examples:
    Basic usage::

        experiment = CCNetExperiment(experiment_name="ccnets_mnist")
        orchestrator, trainer = experiment.run(epochs=50)

Attributes:
    logger: Module-level logger for tracking execution progress.
"""

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from typing import Dict, Any, Optional, Tuple, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    CompositeVisualization,
)
from dl_techniques.models.ccnets import (
    CCNetConfig,
    CCNetOrchestrator,
    CCNetTrainer,
    EarlyStoppingCallback,
    wrap_keras_model,
)
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Custom Visualization Plugins for CCNet
# ---------------------------------------------------------------------


class CCNetTrainingHistoryViz(CompositeVisualization):
    """
    Visualization for the complete loss landscape of CCNet training.

    This composite visualization displays fundamental losses, model errors,
    gradient norms, and system convergence metrics across training epochs.

    Attributes:
        name: Identifier for this visualization plugin.
        description: Human-readable description of the visualization.
    """

    @property
    def name(self) -> str:
        """
        Get the name of this visualization plugin.

        Returns:
            str: The plugin identifier.
        """
        return "ccnet_training_history"

    @property
    def description(self) -> str:
        """
        Get the description of this visualization.

        Returns:
            str: Human-readable description.
        """
        return "Plots all CCNet-specific losses, errors, and gradient norms."

    def can_handle(self, data: Any) -> bool:
        """
        Check if this visualization can handle the provided data.

        Args:
            data: Data to be visualized.

        Returns:
            bool: True if data contains required loss keys.
        """
        return isinstance(data, dict) and all(
            k in data for k in ["generation_loss", "explainer_error"]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the CCNet training history visualization.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.add_subplot("Fundamental Losses", self._plot_fundamental_losses)
        self.add_subplot("Model Errors", self._plot_model_errors)
        self.add_subplot("Gradient Norms", self._plot_gradient_norms)
        self.add_subplot("System Convergence", self._plot_system_convergence)

    def _plot_fundamental_losses(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot fundamental loss components.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing loss histories.
            **kwargs: Additional plotting arguments.
        """
        loss_keys: List[str] = [
            "generation_loss",
            "reconstruction_loss",
            "inference_loss"
        ]

        for key in loss_keys:
            if key in data and data[key]:
                ax.plot(
                    data[key],
                    label=key.replace("_loss", "").title(),
                    linewidth=2
                )

        ax.set_title("Fundamental Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_errors(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot derived model errors.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing error histories.
            **kwargs: Additional plotting arguments.
        """
        error_keys: List[str] = [
            "explainer_error",
            "reasoner_error",
            "producer_error"
        ]

        for key in error_keys:
            if key in data and data[key]:
                ax.plot(
                    data[key],
                    label=key.replace("_error", "").title(),
                    linewidth=2
                )

        ax.set_title("Derived Model Errors")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot gradient L2 norms for each model component.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing gradient norm histories.
            **kwargs: Additional plotting arguments.
        """
        grad_keys: List[str] = [
            "explainer_grad_norm",
            "reasoner_grad_norm",
            "producer_grad_norm"
        ]

        for key in grad_keys:
            if key in data and data[key]:
                ax.plot(
                    data[key],
                    label=key.replace("_grad_norm", "").title(),
                    linewidth=2
                )

        ax.set_title("Gradient Norms (L2)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_system_convergence(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot total system error convergence.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing loss histories.
            **kwargs: Additional plotting arguments.
        """
        required_keys: List[str] = [
            "generation_loss",
            "reconstruction_loss",
            "inference_loss"
        ]

        if not all(k in data and data[k] for k in required_keys):
            ax.text(
                0.5, 0.5,
                "Incomplete loss data for convergence plot.",
                ha='center',
                va='center'
            )
            return

        total_error: List[float] = [
            g + r + i
            for g, r, i in zip(
                data["generation_loss"],
                data["reconstruction_loss"],
                data["inference_loss"]
            )
        ]

        ax.plot(total_error, linewidth=3, color="darkblue")
        ax.fill_between(range(len(total_error)), total_error, alpha=0.3)
        ax.set_title("Total System Error Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total System Error")
        ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------

class ReconstructionQualityViz(VisualizationPlugin):
    """
    Visualization comparing original, reconstructed, and generated images.

    This plugin creates a grid showing original images alongside their
    reconstructions, generations, and corresponding error maps.
    """

    @property
    def name(self) -> str:
        """
        Get the name of this visualization plugin.

        Returns:
            str: The plugin identifier.
        """
        return "ccnet_reconstruction_quality"

    @property
    def description(self) -> str:
        """
        Get the description of this visualization.

        Returns:
            str: Human-readable description.
        """
        return "Compares original, reconstructed, generated images and error maps."

    def can_handle(self, data: Any) -> bool:
        """
        Check if this visualization can handle the provided data.

        Args:
            data: Data to be visualized.

        Returns:
            bool: True if data contains required components.
        """
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(
        self,
        data: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Create the reconstruction quality visualization.

        Args:
            data: Dictionary containing orchestrator and test data.
            ax: Optional matplotlib axes (unused).
            **kwargs: Additional visualization parameters.

        Returns:
            plt.Figure: The generated figure.
        """
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        num_samples: int = data.get("num_samples", 5)

        fig, axes = plt.subplots(
            num_samples, 5,
            figsize=(15, 3 * num_samples)
        )
        fig.suptitle(
            "Reconstruction Quality Analysis",
            fontsize=16,
            fontweight="bold"
        )

        for i in range(num_samples):
            # Prepare input data
            x_input: np.ndarray = x_data[i: i + 1]
            y_truth: np.ndarray = keras.utils.to_categorical([y_data[i]], 10)

            # Forward pass through the network
            tensors: Dict[str, np.ndarray] = orchestrator.forward_pass(
                x_input, y_truth, training=False
            )
            y_pred: int = np.argmax(tensors["y_inferred"][0])

            # Display original image
            axes[i, 0].imshow(x_input[0, :, :, 0], cmap="gray")
            axes[i, 0].set_title(f"Original\nLabel: {y_data[i]}")

            # Display reconstructed image
            axes[i, 1].imshow(tensors["x_reconstructed"][0, :, :, 0], cmap="gray")
            color: str = "green" if y_pred == y_data[i] else "red"
            axes[i, 1].set_title(f"Reconstructed\nInferred: {y_pred}", color=color)

            # Display generated image
            axes[i, 2].imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")
            axes[i, 2].set_title(f"Generated\nTrue: {y_data[i]}")

            # Calculate and display reconstruction error
            diff_recon: np.ndarray = np.abs(
                tensors["x_reconstructed"][0, :, :, 0] - x_input[0, :, :, 0]
            )
            axes[i, 3].imshow(diff_recon, cmap="hot")
            axes[i, 3].set_title(f"Recon Error\nMAE: {np.mean(diff_recon):.3f}")

            # Calculate and display generation error
            diff_gen: np.ndarray = np.abs(
                tensors["x_generated"][0, :, :, 0] - x_input[0, :, :, 0]
            )
            axes[i, 4].imshow(diff_gen, cmap="hot")
            axes[i, 4].set_title(f"Gen Error\nMAE: {np.mean(diff_gen):.3f}")

            # Remove axes for all subplots
            for j in range(5):
                axes[i, j].axis("off")

        return fig

# ---------------------------------------------------------------------

class CounterfactualMatrixViz(VisualizationPlugin):
    """
    Visualization generating a matrix of counterfactual digits.

    This plugin creates a grid showing how different source digits would
    appear if they were drawn as different target classes.
    """

    @property
    def name(self) -> str:
        """
        Get the name of this visualization plugin.

        Returns:
            str: The plugin identifier.
        """
        return "ccnet_counterfactual_matrix"

    @property
    def description(self) -> str:
        """
        Get the description of this visualization.

        Returns:
            str: Human-readable description.
        """
        return "Generates a matrix of counterfactual digits."

    def can_handle(self, data: Any) -> bool:
        """
        Check if this visualization can handle the provided data.

        Args:
            data: Data to be visualized.

        Returns:
            bool: True if data contains required components.
        """
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(
        self,
        data: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Create the counterfactual matrix visualization.

        Args:
            data: Dictionary containing orchestrator and test data.
            ax: Optional matplotlib axes (unused).
            **kwargs: Additional visualization parameters.

        Returns:
            plt.Figure: The generated figure.
        """
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        source_digits: List[int] = data.get("source_digits", [0, 1, 3, 5, 7])
        target_digits: List[int] = data.get("target_digits", [0, 2, 4, 6, 8, 9])

        rows: int = len(source_digits)
        cols: int = len(target_digits)

        fig, axes = plt.subplots(
            rows, cols + 1,
            figsize=(cols * 1.5 + 1.5, rows * 1.5)
        )
        fig.suptitle(
            'Counterfactual Generation Matrix\n"What if this digit were drawn as..."',
            fontsize=16,
            fontweight="bold"
        )

        for i, source in enumerate(source_digits):
            # Find an example of the source digit
            idx: int = np.where(y_data == source)[0][0]
            x_source: np.ndarray = x_data[idx: idx + 1]

            # Display source image
            axes[i, 0].imshow(x_source[0, :, :, 0], cmap="gray")
            axes[i, 0].set_ylabel(f"{source} →", rotation=0, size='large', labelpad=20)

            # Generate counterfactuals for each target
            for j, target in enumerate(target_digits):
                y_target: np.ndarray = keras.utils.to_categorical([target], 10)
                x_counter: np.ndarray = orchestrator.counterfactual_generation(
                    x_source, y_target
                )
                axes[i, j + 1].imshow(x_counter[0, :, :, 0], cmap="gray")

                # Add column headers
                if i == 0:
                    axes[i, j + 1].set_title(f"→ {target}", size='large')

                # Highlight diagonal (source == target)
                if source == target:
                    axes[i, j + 1].patch.set_edgecolor("green")
                    axes[i, j + 1].patch.set_linewidth(3)

        # Remove ticks from all subplots
        for axis in axes.flatten():
            axis.set_xticks([])
            axis.set_yticks([])

        return fig

# ---------------------------------------------------------------------

class LatentSpaceAnalysisViz(CompositeVisualization):
    """
    Visualization for latent space analysis using t-SNE and statistics.

    This composite visualization provides multiple views of the learned
    latent representations including t-SNE projections, density plots,
    and activation statistics.
    """

    @property
    def name(self) -> str:
        """
        Get the name of this visualization plugin.

        Returns:
            str: The plugin identifier.
        """
        return "ccnet_latent_space"

    @property
    def description(self) -> str:
        """
        Get the description of this visualization.

        Returns:
            str: Human-readable description.
        """
        return "Visualizes latent space with t-SNE and statistical plots."

    def can_handle(self, data: Any) -> bool:
        """
        Check if this visualization can handle the provided data.

        Args:
            data: Data to be visualized.

        Returns:
            bool: True if data contains required components.
        """
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the latent space analysis visualization.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.latent_vectors: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.latent_2d: Optional[np.ndarray] = None

        self.add_subplot("t-SNE Colored by Class", self._plot_tsne_class)
        self.add_subplot("t-SNE Density", self._plot_tsne_density)
        self.add_subplot("Mean Activation", self._plot_mean_activation)
        self.add_subplot("Latent Norms by Class", self._plot_latent_norms)

    def _prepare_data(self, data: Dict[str, Any]) -> None:
        """
        Prepare latent vectors and compute t-SNE embedding.

        Args:
            data: Dictionary containing orchestrator and test data.
        """
        if self.latent_vectors is not None:
            return

        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        num_samples: int = data.get("num_samples", 1000)

        # Sample data points
        indices: np.ndarray = np.random.choice(
            len(x_data),
            min(num_samples, len(x_data)),
            replace=False
        )

        # Extract latent representations
        latent_vectors: List[np.ndarray] = []
        labels: List[int] = []

        for idx in indices:
            x_sample: np.ndarray = x_data[idx: idx + 1]
            _, e_latent = orchestrator.disentangle_causes(x_sample)
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_data[idx])

        self.latent_vectors = np.array(latent_vectors)
        self.labels = np.array(labels)

        # Compute t-SNE embedding
        logger.info("Computing t-SNE embedding for latent space...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            n_iter=300
        )
        self.latent_2d = tsne.fit_transform(self.latent_vectors)

    def _plot_tsne_class(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot t-SNE projection colored by digit class.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing visualization data.
            **kwargs: Additional plotting arguments.
        """
        self._prepare_data(data)

        scatter = ax.scatter(
            self.latent_2d[:, 0],
            self.latent_2d[:, 1],
            c=self.labels,
            cmap="tab10",
            s=20,
            alpha=0.7
        )

        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Digit")
        ax.grid(True, alpha=0.3)

    def _plot_tsne_density(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot t-SNE projection with density coloring.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing visualization data.
            **kwargs: Additional plotting arguments.
        """
        self._prepare_data(data)

        # Calculate density using Gaussian KDE
        xy: np.ndarray = self.latent_2d.T
        z: np.ndarray = gaussian_kde(xy)(xy)

        scatter = ax.scatter(
            self.latent_2d[:, 0],
            self.latent_2d[:, 1],
            c=z,
            cmap="viridis",
            s=20,
            alpha=0.7
        )

        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Density")
        ax.grid(True, alpha=0.3)

    def _plot_mean_activation(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot mean absolute activation per latent dimension.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing visualization data.
            **kwargs: Additional plotting arguments.
        """
        self._prepare_data(data)

        mean_activations: np.ndarray = np.mean(
            np.abs(self.latent_vectors), axis=0
        )

        ax.bar(
            range(len(mean_activations)),
            mean_activations,
            color="steelblue"
        )

        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |Activation|")
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_latent_norms(
        self,
        ax: plt.Axes,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot distribution of latent vector norms by digit class.

        Args:
            ax: Matplotlib axes for plotting.
            data: Dictionary containing visualization data.
            **kwargs: Additional plotting arguments.
        """
        self._prepare_data(data)

        latent_norms: np.ndarray = np.linalg.norm(self.latent_vectors, axis=1)

        sns.violinplot(
            x=self.labels,
            y=latent_norms,
            ax=ax,
            palette="husl",
            inner="quartile"
        )

        ax.set_xlabel("Digit Class")
        ax.set_ylabel("||E|| (L2 Norm)")
        ax.grid(True, alpha=0.3, axis='y')


# ---------------------------------------------------------------------
# Keras Model Definitions
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MNISTExplainer(keras.Model):
    """
    Explainer network for MNIST CCNet.

    This model encodes input images into latent explanatory factors using
    a variational approach with mean and log-variance outputs.

    Args:
        explanation_dim: Dimension of the latent explanation space.
        l2_regularization: L2 regularization strength for weights.
        **kwargs: Additional arguments passed to keras.Model.
    """

    def __init__(
        self,
        explanation_dim: int = 128,
        l2_regularization: float = 1e-4,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MNIST Explainer.

        Args:
            explanation_dim: Dimension of latent space.
            l2_regularization: L2 regularization coefficient.
            **kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)
        self.explanation_dim = explanation_dim
        self.regularizer = (
            keras.regularizers.L2(l2_regularization)
            if l2_regularization > 0
            else None
        )

        # Convolutional encoder layers
        self.conv1 = keras.layers.Conv2D(
            32, 5, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool1 = keras.layers.MaxPooling2D(2)

        self.conv2 = keras.layers.Conv2D(
            64, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool2 = keras.layers.MaxPooling2D(2)

        self.conv3 = keras.layers.Conv2D(
            128, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool3 = keras.layers.GlobalMaxPool2D()

        # Variational layers
        self.flatten = keras.layers.Flatten()
        self.fc_mu = keras.layers.Dense(
            explanation_dim, name="mu", kernel_regularizer=self.regularizer
        )
        self.fc_log_var = keras.layers.Dense(
            explanation_dim, name="log_var", kernel_regularizer=self.regularizer
        )

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = False
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass through the explainer network.

        Args:
            x: Input image tensor of shape (batch, height, width, channels).
            training: Whether the model is in training mode.

        Returns:
            Tuple[keras.KerasTensor, keras.KerasTensor]:
                Mean and log-variance of latent distribution.
        """
        # Convolutional encoding
        x = self.pool1(self.act1(self.bn1(self.conv1(x), training=training)))
        x = self.pool2(self.act2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.act3(self.bn3(self.conv3(x), training=training)))

        # Flatten and compute distribution parameters
        features: keras.KerasTensor = self.flatten(x)
        mu: keras.KerasTensor = self.fc_mu(features)
        log_var: keras.KerasTensor = self.fc_log_var(features)

        return mu, log_var

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for model serialization.

        Returns:
            Dict[str, Any]: Model configuration.
        """
        config = super().get_config()
        config.update({
            "explanation_dim": self.explanation_dim,
            "l2_regularization": (
                self.regularizer.l2 if self.regularizer else 0.0
            )
        })
        return config


@keras.saving.register_keras_serializable()
class MNISTReasoner(keras.Model):
    """
    Reasoner network for MNIST CCNet.

    This model performs inference from images and latent explanations to
    predict digit classes.

    Args:
        num_classes: Number of output classes.
        explanation_dim: Dimension of the latent explanation space.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization strength for weights.
        **kwargs: Additional arguments passed to keras.Model.
    """

    def __init__(
        self,
        num_classes: int = 10,
        explanation_dim: int = 128,
        dropout_rate: float = 0.2,
        l2_regularization: float = 1e-4,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MNIST Reasoner.

        Args:
            num_classes: Number of classification categories.
            explanation_dim: Dimension of latent space.
            dropout_rate: Dropout probability.
            l2_regularization: L2 regularization coefficient.
            **kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = (
            keras.regularizers.L2(l2_regularization)
            if l2_regularization > 0
            else None
        )

        # Image processing layers
        self.conv1 = keras.layers.Conv2D(
            32, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool1 = keras.layers.MaxPooling2D(2)

        self.conv2 = keras.layers.Conv2D(
            64, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool2 = keras.layers.MaxPooling2D(2)

        self.flatten = keras.layers.Flatten()

        # Combination and classification layers
        self.fc_combine = keras.layers.Dense(
            512, kernel_regularizer=self.regularizer
        )
        self.bn_combine = keras.layers.BatchNormalization()
        self.act_combine = keras.layers.LeakyReLU(negative_slope=0.2)

        self.fc_hidden = keras.layers.Dense(
            256, kernel_regularizer=self.regularizer
        )
        self.bn_hidden = keras.layers.BatchNormalization()
        self.act_hidden = keras.layers.LeakyReLU(negative_slope=0.2)

        self.dropout = keras.layers.Dropout(dropout_rate)
        self.fc_output = keras.layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=self.regularizer
        )

    def call(
        self,
        x: keras.KerasTensor,
        e: keras.KerasTensor,
        training: Optional[bool] = False
    ) -> keras.KerasTensor:
        """
        Forward pass through the reasoner network.

        Args:
            x: Input image tensor of shape (batch, height, width, channels).
            e: Latent explanation tensor of shape (batch, explanation_dim).
            training: Whether the model is in training mode.

        Returns:
            keras.KerasTensor: Predicted class probabilities.
        """
        # Process image features
        img_features = self.pool1(
            self.act1(self.bn1(self.conv1(x), training=training))
        )
        img_features = self.pool2(
            self.act2(self.bn2(self.conv2(img_features), training=training))
        )

        # Combine image features with explanations
        combined = keras.ops.concatenate(
            [self.flatten(img_features), e], axis=-1
        )

        # Classification pipeline
        combined = self.dropout(
            self.act_combine(
                self.bn_combine(self.fc_combine(combined), training=training)
            ),
            training=training
        )

        combined = self.dropout(
            self.act_hidden(
                self.bn_hidden(self.fc_hidden(combined), training=training)
            ),
            training=training
        )

        return self.fc_output(combined)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for model serialization.

        Returns:
            Dict[str, Any]: Model configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "explanation_dim": self.explanation_dim,
            "dropout_rate": self.dropout_rate,
            "l2_regularization": (
                self.regularizer.l2 if self.regularizer else 0.0
            )
        })
        return config


@keras.saving.register_keras_serializable()
class MNISTProducer(keras.Model):
    """
    Producer network for MNIST CCNet.

    This model generates images from class labels and latent explanations
    using style-based generation techniques.

    Args:
        num_classes: Number of output classes.
        explanation_dim: Dimension of the latent explanation space.
        **kwargs: Additional arguments passed to keras.Model.
    """

    def __init__(
        self,
        num_classes: int = 10,
        explanation_dim: int = 16,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MNIST Producer.

        Args:
            num_classes: Number of digit classes.
            explanation_dim: Dimension of latent space.
            **kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim

        # Class embedding
        self.label_embedding = keras.layers.Embedding(num_classes, 512)

        # Content generation
        self.fc_content = keras.layers.Dense(7 * 7 * 128, activation='relu')
        self.reshape_content = keras.layers.Reshape((7, 7, 128))

        # Style transformation
        self.style_transform_1 = keras.layers.Dense(256, activation='relu')
        self.style_transform_2 = keras.layers.Dense(128, activation='relu')

        # Upsampling and convolution layers
        self.up1 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv1 = keras.layers.Conv2D(128, 3, padding="same")
        self.norm1 = keras.layers.BatchNormalization(center=False, scale=False)
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.2)

        self.up2 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = keras.layers.Conv2D(64, 3, padding="same")
        self.norm2 = keras.layers.BatchNormalization(center=False, scale=False)
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.2)

        # Output layer
        self.conv_out = keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )

    def apply_style(
        self,
        content_tensor: keras.KerasTensor,
        style_vector: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply style modulation to content features.

        Args:
            content_tensor: Content feature tensor.
            style_vector: Style parameter vector.

        Returns:
            keras.KerasTensor: Style-modulated features.
        """
        num_channels: int = keras.ops.shape(content_tensor)[-1]
        style_params: keras.KerasTensor = style_vector

        # Split style parameters into scale and bias
        scale: keras.KerasTensor = style_params[:, :num_channels]
        bias: keras.KerasTensor = style_params[:, num_channels:]

        # Reshape for broadcasting
        scale = keras.ops.expand_dims(keras.ops.expand_dims(scale, 1), 1)
        bias = keras.ops.expand_dims(keras.ops.expand_dims(bias, 1), 1)

        return content_tensor * (scale + 1) + bias

    def call(
        self,
        y: keras.KerasTensor,
        e: keras.KerasTensor,
        training: Optional[bool] = False
    ) -> keras.KerasTensor:
        """
        Forward pass through the producer network.

        Args:
            y: One-hot class label tensor of shape (batch, num_classes).
            e: Latent explanation tensor of shape (batch, explanation_dim).
            training: Whether the model is in training mode.

        Returns:
            keras.KerasTensor: Generated image tensor.
        """
        # Generate content from class embedding
        c: keras.KerasTensor = keras.ops.matmul(
            y, self.label_embedding.weights[0]
        )
        c = self.fc_content(c)
        c = self.reshape_content(c)

        # Prepare style vectors
        style1: keras.KerasTensor = self.style_transform_1(e)
        style2: keras.KerasTensor = self.style_transform_2(e)

        # First upsampling block
        x = self.up1(c)
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.apply_style(x, style1)
        x = self.act1(x)

        # Second upsampling block
        x = self.up2(x)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.apply_style(x, style2)
        x = self.act2(x)

        return self.conv_out(x)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for model serialization.

        Returns:
            Dict[str, Any]: Model configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "explanation_dim": self.explanation_dim
        })
        return config


# ---------------------------------------------------------------------
# Helper Functions for Data and Model Setup
# ---------------------------------------------------------------------


def create_mnist_ccnet(
    explanation_dim: int = 16,
    learning_rate: float = 1e-3
) -> CCNetOrchestrator:
    """
    Create and initialize a CCNet for MNIST digit classification.

    Args:
        explanation_dim: Dimension of the latent explanation space.
        learning_rate: Learning rate for all model optimizers.

    Returns:
        CCNetOrchestrator: Configured CCNet orchestrator instance.
    """
    # Model hyperparameters
    dropout_rate: float = 0.2
    l2_regularization: float = 1e-4

    # Initialize model components
    explainer = MNISTExplainer(explanation_dim, l2_regularization)
    reasoner = MNISTReasoner(num_classes=10, explanation_dim=explanation_dim, dropout_rate=dropout_rate, l2_regularization=l2_regularization)
    producer = MNISTProducer(num_classes=10, explanation_dim=explanation_dim)

    # Build models with dummy data to initialize weights
    dummy_image: keras.KerasTensor = keras.ops.zeros((1, 28, 28, 1))
    dummy_label_one_hot: keras.KerasTensor = keras.ops.zeros((1, 10))
    dummy_label_indices: keras.KerasTensor = keras.ops.zeros((1,), dtype="int32")

    # Initialize explainer
    mu, _ = explainer(dummy_image)
    dummy_latent: keras.KerasTensor = mu

    # Initialize producer (build embedding layer first)
    _ = producer.label_embedding(dummy_label_indices)
    _ = producer(dummy_label_one_hot, dummy_latent)

    # Initialize reasoner
    _ = reasoner(dummy_image, dummy_latent)

    # Configure CCNet
    config = CCNetConfig(
        explanation_dim=explanation_dim,
        loss_type="l2",
        learning_rates={
            "explainer": learning_rate,
            "reasoner": learning_rate,
            "producer": learning_rate
        },
        gradient_clip_norm=1.0,
        kl_weight=0.1
    )

    # Create orchestrator
    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=config
    )


def prepare_mnist_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare MNIST dataset for training and validation.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]:
            Training and validation datasets.
    """
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and reshape images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.batch(128).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(128).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# ---------------------------------------------------------------------
# Main Training & Visualization Workflow
# ---------------------------------------------------------------------


class CCNetExperiment:
    """
    Experimental framework for CCNet training and visualization.

    This class orchestrates the complete workflow of training a CCNet model
    on MNIST and generating comprehensive visualizations of the results.

    Args:
        experiment_name: Identifier for this experiment run.
        results_dir: Directory path for saving results.

    Attributes:
        viz_manager: Visualization manager instance.
        x_test: Test set images.
        y_test: Test set labels.
    """

    def __init__(
        self,
        experiment_name: str,
        results_dir: str = "results"
    ) -> None:
        """
        Initialize the CCNet experiment.

        Args:
            experiment_name: Name identifier for the experiment.
            results_dir: Output directory for results.
        """
        self.viz_manager = VisualizationManager(
            experiment_name=experiment_name,
            output_dir=results_dir
        )
        self._register_plugins()
        self.x_test, self.y_test = self._load_test_data()

    def _register_plugins(self) -> None:
        """Register visualization plugins with the manager."""
        plugin_classes: List[type] = [
            CCNetTrainingHistoryViz,
            ReconstructionQualityViz,
            CounterfactualMatrixViz,
            LatentSpaceAnalysisViz,
        ]

        for plugin_class in plugin_classes:
            instance = plugin_class(
                self.viz_manager.config,
                self.viz_manager.context
            )
            self.viz_manager.register_plugin(instance)

    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess test data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Test images and labels.
        """
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
        return x_test, y_test

    def run(self, epochs: int = 20) -> Tuple[CCNetOrchestrator, CCNetTrainer]:
        """
        Execute the complete training and visualization pipeline.

        Args:
            epochs: Number of training epochs.

        Returns:
            Tuple[CCNetOrchestrator, CCNetTrainer]:
                Trained orchestrator and trainer instances.
        """
        # Initialize model
        logger.info("Creating CCNet for MNIST...")
        orchestrator = create_mnist_ccnet(explanation_dim=16)

        # Prepare data
        logger.info("Preparing data...")
        train_dataset, val_dataset = prepare_mnist_data()

        # Setup training
        logger.info("Setting up trainer...")
        trainer = CCNetTrainer(orchestrator)

        # Execute training
        logger.info("Starting training...")
        try:
            trainer.train(
                train_dataset,
                epochs,
                validation_dataset=val_dataset,
                callbacks=[
                    EarlyStoppingCallback(patience=5, error_threshold=1e-4)
                ]
            )
        except StopIteration:
            logger.info("Training stopped early due to convergence.")

        # Generate visualizations
        self.generate_final_report(orchestrator, trainer)

        logger.info(
            f"All visualizations saved to: "
            f"{self.viz_manager.context.output_dir / self.viz_manager.context.experiment_name}"
        )

        return orchestrator, trainer

    def generate_final_report(
        self,
        orchestrator: CCNetOrchestrator,
        trainer: CCNetTrainer
    ) -> None:
        """
        Generate comprehensive visualization report.

        Args:
            orchestrator: Trained CCNet orchestrator.
            trainer: CCNet trainer with history.
        """
        logger.info("=" * 60)
        logger.info("Generating final visualizations...")
        logger.info("=" * 60)

        # Prepare visualization data
        report_data: Dict[str, Any] = {
            'orchestrator': orchestrator,
            'x_data': self.x_test,
            'y_data': self.y_test
        }

        dashboard_recon_data: Dict[str, Any] = {
            **report_data,
            'num_samples': 3
        }

        dashboard_latent_data: Dict[str, Any] = {
            **report_data,
            'num_samples': 500
        }

        # Generate individual visualizations
        self.viz_manager.visualize(
            trainer.history,
            "ccnet_training_history",
            show=False
        )

        self.viz_manager.visualize(
            report_data,
            "ccnet_reconstruction_quality",
            show=False
        )

        self.viz_manager.visualize(
            report_data,
            "ccnet_counterfactual_matrix",
            show=False
        )

        self.viz_manager.visualize(
            report_data,
            "ccnet_latent_space",
            show=False
        )

        # Create composite dashboard
        dashboard_data: Dict[str, Any] = {
            "ccnet_training_history": trainer.history,
            "ccnet_reconstruction_quality": dashboard_recon_data,
            "ccnet_latent_space": dashboard_latent_data,
        }

        self.viz_manager.create_dashboard(dashboard_data, show=False)

        # Save trained models
        logger.info("Saving models...")
        models_dir = self.viz_manager.context.get_save_path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.save_models(str(models_dir / "mnist_ccnet"))
        logger.info(f"Models saved to {models_dir}")


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Create and run experiment
    experiment = CCNetExperiment(experiment_name="ccnets_mnist")
    orchestrator, trainer = experiment.run(epochs=50)