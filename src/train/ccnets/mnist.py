"""
CCNet MNIST Experiment with Centralized Configuration.

This module demonstrates a complete training and analysis workflow for a
Counter-Causal Net (CCNet) on the MNIST dataset, with all configuration
parameters centralized at the top for easy experimentation.

Examples:
    Basic usage::

        # Use default configuration
        config = ExperimentConfig()
        experiment = CCNetExperiment(config)
        orchestrator, trainer = experiment.run()

        # Or customize specific aspects
        config = ExperimentConfig(
            model=ModelConfig(explanation_dim=32),
            training=TrainingConfig(epochs=100),
            data=DataConfig(batch_size=256)
        )
        experiment = CCNetExperiment(config)
        orchestrator, trainer = experiment.run()
"""

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from dataclasses import dataclass, field
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
from dl_techniques.layers.activations.golu import GoLU

# =====================================================================
# CENTRALIZED CONFIGURATION
# =====================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture parameters."""

    # Shared parameters
    explanation_dim: int = 8
    num_classes: int = 10

    # Explainer parameters
    explainer_conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    explainer_conv_kernels: List[int] = field(default_factory=lambda: [5, 3, 3])
    explainer_l2_regularization: float = 1e-4

    # Reasoner parameters
    reasoner_conv_filters: List[int] = field(default_factory=lambda: [32, 64])
    reasoner_conv_kernels: List[int] = field(default_factory=lambda: [5, 3])
    reasoner_dense_units: List[int] = field(default_factory=lambda: [512, 256])
    reasoner_dropout_rate: float = 0.2
    reasoner_l2_regularization: float = 1e-4

    # Producer parameters
    producer_initial_dense_units: int = 256
    producer_initial_spatial_size: int = 7
    producer_initial_channels: int = 128
    producer_conv_filters: List[int] = field(default_factory=lambda: [128, 64])
    producer_style_units: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Basic training parameters
    epochs: int = 20
    learning_rates: Dict[str, float] = field(default_factory=lambda: {
        'explainer': 1e-3,
        'reasoner': 1e-3,
        'producer': 3e-4
    })

    # CCNet-specific parameters
    loss_fn: str = 'huber'  # Options: 'l1', 'l2', 'huber', 'polynomial'
    loss_fn_params: Dict[str, Any] = field(default_factory=dict)
    gradient_clip_norm: Optional[float] = 1.0
    kl_weight: float = 0.1
    dynamic_weighting: bool = True
    use_mixed_precision: bool = False

    # KL annealing
    kl_annealing_epochs: Optional[int] = None  # None to disable

    # Loss weights
    explainer_weights: Dict[str, float] = field(default_factory=lambda: {
        'inference': 1.0,
        'generation': 1.0
    })
    reasoner_weights: Dict[str, float] = field(default_factory=lambda: {
        'reconstruction': 1.0,
        'inference': 1.0
    })
    producer_weights: Dict[str, float] = field(default_factory=lambda: {
        'generation': 1.0,
        'reconstruction': 1.0
    })


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""

    # Data loading
    batch_size: int = 128
    shuffle_buffer_size: int = 60000
    prefetch_buffer_size: int = tf.data.AUTOTUNE

    # Data augmentation
    apply_augmentation: bool = True
    noise_stddev: float = 0.05
    max_translation: int = 2  # pixels


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping callback."""

    enabled: bool = True
    patience: int = 5
    error_threshold: float = 1e-4
    grad_stagnation_threshold: Optional[float] = 1e-3


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    # Output settings
    experiment_name: str = "ccnets_mnist"
    results_dir: str = "results"
    save_figures: bool = True
    show_figures: bool = False

    # Visualization samples
    reconstruction_samples: int = 5
    counterfactual_source_digits: List[int] = field(default_factory=lambda: [0, 1, 3, 5, 7])
    counterfactual_target_digits: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 9])
    latent_space_samples: int = 1000

    # t-SNE parameters
    tsne_perplexity: int = 30
    tsne_max_iter: int = 300
    tsne_random_state: int = 42

    # Dashboard settings
    dashboard_reconstruction_samples: int = 3
    dashboard_latent_samples: int = 500


@dataclass
class ExperimentConfig:
    """Master configuration for the entire experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Logging
    log_interval: int = 10  # Log progress every N batches
    verbose: bool = True


# =====================================================================
# VISUALIZATION PLUGINS (unchanged from original)
# =====================================================================

class CCNetTrainingHistoryViz(CompositeVisualization):
    """
    Visualization for the complete loss landscape of CCNet training.
    """

    @property
    def name(self) -> str:
        return "ccnet_training_history"

    @property
    def description(self) -> str:
        return "Plots all CCNet-specific losses, errors, and gradient norms."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["generation_loss", "explainer_error"]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_subplot("Fundamental Losses", self._plot_fundamental_losses)
        self.add_subplot("Model Errors", self._plot_model_errors)
        self.add_subplot("Gradient Norms", self._plot_gradient_norms)
        self.add_subplot("System Convergence", self._plot_system_convergence)

    def _plot_fundamental_losses(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        loss_keys: List[str] = [
            "generation_loss", "reconstruction_loss", "inference_loss"
        ]
        for key in loss_keys:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_loss", "").title(), linewidth=2)
        ax.set_title("Fundamental Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_errors(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        error_keys: List[str] = [
            "explainer_error", "reasoner_error", "producer_error"
        ]
        for key in error_keys:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_error", "").title(), linewidth=2)
        ax.set_title("Derived Model Errors")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        grad_keys: List[str] = [
            "explainer_grad_norm", "reasoner_grad_norm", "producer_grad_norm"
        ]
        for key in grad_keys:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_grad_norm", "").title(), linewidth=2)
        ax.set_title("Gradient Norms (L2)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_system_convergence(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        required_keys: List[str] = [
            "generation_loss", "reconstruction_loss", "inference_loss"
        ]
        if not all(k in data and data[k] for k in required_keys):
            ax.text(0.5, 0.5, "Incomplete loss data for convergence plot.",
                    ha='center', va='center')
            return

        total_error: List[float] = [
            g + r + i for g, r, i in zip(
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


class ReconstructionQualityViz(VisualizationPlugin):
    """
    Visualization comparing original, reconstructed, and generated images.
    """

    @property
    def name(self) -> str:
        return "ccnet_reconstruction_quality"

    @property
    def description(self) -> str:
        return "Compares original, reconstructed, generated images and error maps."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(
        self, data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs: Any
    ) -> plt.Figure:
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        num_samples: int = data.get("num_samples", 5)

        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
        fig.suptitle("Reconstruction Quality Analysis", fontsize=16, fontweight="bold")

        for i in range(num_samples):
            x_input: np.ndarray = x_data[i: i + 1]
            y_truth: np.ndarray = keras.utils.to_categorical([y_data[i]], 10)

            tensors: Dict[str, np.ndarray] = orchestrator.forward_pass(
                x_input, y_truth, training=False
            )
            y_pred: int = np.argmax(tensors["y_inferred"][0])

            axes[i, 0].imshow(x_input[0, :, :, 0], cmap="gray")
            axes[i, 0].set_title(f"Original\nLabel: {y_data[i]}")

            axes[i, 1].imshow(tensors["x_reconstructed"][0, :, :, 0], cmap="gray")
            color: str = "green" if y_pred == y_data[i] else "red"
            axes[i, 1].set_title(f"Reconstructed\nInferred: {y_pred}", color=color)

            axes[i, 2].imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")
            axes[i, 2].set_title(f"Generated\nTrue: {y_data[i]}")

            diff_recon: np.ndarray = np.abs(
                tensors["x_reconstructed"][0, :, :, 0] - x_input[0, :, :, 0]
            )
            axes[i, 3].imshow(diff_recon, cmap="hot")
            axes[i, 3].set_title(f"Recon Error\nMAE: {np.mean(diff_recon):.3f}")

            diff_gen: np.ndarray = np.abs(
                tensors["x_generated"][0, :, :, 0] - x_input[0, :, :, 0]
            )
            axes[i, 4].imshow(diff_gen, cmap="hot")
            axes[i, 4].set_title(f"Gen Error\nMAE: {np.mean(diff_gen):.3f}")

            for j in range(5):
                axes[i, j].axis("off")

        return fig


class CounterfactualMatrixViz(VisualizationPlugin):
    """
    Visualization generating a matrix of counterfactual digits.
    """

    @property
    def name(self) -> str:
        return "ccnet_counterfactual_matrix"

    @property
    def description(self) -> str:
        return "Generates a matrix of counterfactual digits."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(
        self, data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs: Any
    ) -> plt.Figure:
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        source_digits: List[int] = data.get("source_digits", [0, 1, 3, 5, 7])
        target_digits: List[int] = data.get("target_digits", [0, 2, 4, 6, 8, 9])

        rows: int = len(source_digits)
        cols: int = len(target_digits)

        fig, axes = plt.subplots(
            rows, cols + 1, figsize=(cols * 1.5 + 1.5, rows * 1.5)
        )
        fig.suptitle(
            'Counterfactual Generation Matrix\n"What if this digit were drawn as..."',
            fontsize=16, fontweight="bold"
        )

        for i, source in enumerate(source_digits):
            idx: int = np.where(y_data == source)[0][0]
            x_source: np.ndarray = x_data[idx: idx + 1]

            axes[i, 0].imshow(x_source[0, :, :, 0], cmap="gray")
            axes[i, 0].set_ylabel(f"{source} →", rotation=0, size='large', labelpad=20)

            for j, target in enumerate(target_digits):
                y_target: np.ndarray = keras.utils.to_categorical([target], 10)
                x_counter: np.ndarray = orchestrator.counterfactual_generation(
                    x_source, y_target
                )
                axes[i, j + 1].imshow(x_counter[0, :, :, 0], cmap="gray")

                if i == 0:
                    axes[i, j + 1].set_title(f"→ {target}", size='large')

                if source == target:
                    axes[i, j + 1].patch.set_edgecolor("green")
                    axes[i, j + 1].patch.set_linewidth(3)

        for axis in axes.flatten():
            axis.set_xticks([])
            axis.set_yticks([])

        return fig


class LatentSpaceAnalysisViz(CompositeVisualization):
    """
    Visualization for latent space analysis using t-SNE and statistics.
    """

    @property
    def name(self) -> str:
        return "ccnet_latent_space"

    @property
    def description(self) -> str:
        return "Visualizes latent space with t-SNE and statistical plots."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latent_vectors: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.latent_2d: Optional[np.ndarray] = None
        self.tsne_config: Optional[Dict[str, Any]] = None

        self.add_subplot("t-SNE Colored by Class", self._plot_tsne_class)
        self.add_subplot("t-SNE Density", self._plot_tsne_density)
        self.add_subplot("Mean Activation", self._plot_mean_activation)
        self.add_subplot("Latent Norms by Class", self._plot_latent_norms)

    def _prepare_data(self, data: Dict[str, Any]) -> None:
        if self.latent_vectors is not None:
            return

        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data: np.ndarray = data["x_data"]
        y_data: np.ndarray = data["y_data"]
        num_samples: int = data.get("num_samples", 1000)

        # Extract t-SNE config if provided
        self.tsne_config = data.get("tsne_config", {
            "perplexity": 30,
            "max_iter": 300,
            "random_state": 42
        })

        indices: np.ndarray = np.random.choice(
            len(x_data), min(num_samples, len(x_data)), replace=False
        )

        latent_vectors: List[np.ndarray] = []
        labels: List[int] = []

        for idx in indices:
            x_sample: np.ndarray = x_data[idx: idx + 1]
            _, e_latent = orchestrator.disentangle_causes(x_sample)
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_data[idx])

        self.latent_vectors = np.array(latent_vectors)
        self.labels = np.array(labels)

        logger.info("Computing t-SNE embedding for latent space...")
        tsne = TSNE(
            n_components=2,
            perplexity=self.tsne_config["perplexity"],
            max_iter=self.tsne_config["max_iter"],
            random_state=self.tsne_config["random_state"]
        )
        self.latent_2d = tsne.fit_transform(self.latent_vectors)

    def _plot_tsne_class(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        scatter = ax.scatter(
            self.latent_2d[:, 0], self.latent_2d[:, 1],
            c=self.labels, cmap="tab10", s=20, alpha=0.7
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Digit")
        ax.grid(True, alpha=0.3)

    def _plot_tsne_density(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        xy: np.ndarray = self.latent_2d.T
        z: np.ndarray = gaussian_kde(xy)(xy)
        scatter = ax.scatter(
            self.latent_2d[:, 0], self.latent_2d[:, 1],
            c=z, cmap="viridis", s=20, alpha=0.7
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Density")
        ax.grid(True, alpha=0.3)

    def _plot_mean_activation(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        mean_activations: np.ndarray = np.mean(np.abs(self.latent_vectors), axis=0)
        ax.bar(range(len(mean_activations)), mean_activations, color="steelblue")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |Activation|")
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_latent_norms(
        self, ax: plt.Axes, data: Dict[str, Any], **kwargs: Any
    ) -> None:
        self._prepare_data(data)
        latent_norms: np.ndarray = np.linalg.norm(self.latent_vectors, axis=1)
        sns.violinplot(
            x=self.labels, y=latent_norms, ax=ax,
            palette="husl", inner="quartile"
        )
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("||E|| (L2 Norm)")
        ax.grid(True, alpha=0.3, axis='y')


# =====================================================================
# MODEL DEFINITIONS
# =====================================================================

@keras.saving.register_keras_serializable()
class MNISTExplainer(keras.Model):
    """
    Explainer network for MNIST CCNet.
    """

    def __init__(
        self,
        config: ModelConfig,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MNIST Explainer with configuration.

        Args:
            config: Model configuration object.
            **kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)
        self.config = config
        self.regularizer = (
            keras.regularizers.L2(config.explainer_l2_regularization)
            if config.explainer_l2_regularization > 0 else None
        )

        # Build convolutional layers from config
        self.conv_layers = []
        self.bn_layers = []
        self.act_layers = []
        self.pool_layers = []

        for i, (filters, kernel) in enumerate(zip(
            config.explainer_conv_filters,
            config.explainer_conv_kernels
        )):
            self.conv_layers.append(keras.layers.Conv2D(
                filters, kernel, padding="same",
                kernel_regularizer=self.regularizer,
                name=f"conv_{i+1}"
            ))
            self.bn_layers.append(keras.layers.BatchNormalization(name=f"bn_{i+1}"))
            self.act_layers.append(GoLU(
                name=f"act_{i+1}"
            ))
            # Use GlobalMaxPool for the last layer, MaxPool for others
            if i == len(config.explainer_conv_filters) - 1:
                self.pool_layers.append(keras.layers.GlobalMaxPool2D(name="global_pool"))
            else:
                self.pool_layers.append(keras.layers.MaxPooling2D(2, name=f"pool_{i+1}"))

        # Variational layers
        self.flatten = keras.layers.Flatten()
        self.fc_mu = keras.layers.Dense(
            config.explanation_dim, name="mu",
            kernel_regularizer=self.regularizer
        )
        self.fc_log_var = keras.layers.Dense(
            config.explanation_dim, name="log_var",
            kernel_regularizer=self.regularizer
        )

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = False
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass through the explainer network."""
        # Convolutional encoding
        for conv, bn, act, pool in zip(
            self.conv_layers, self.bn_layers,
            self.act_layers, self.pool_layers
        ):
            x = conv(x)
            x = bn(x, training=training)
            x = act(x)
            x = pool(x)

        # Flatten and compute distribution parameters
        features: keras.KerasTensor = self.flatten(x) if x.shape.rank > 2 else x
        mu: keras.KerasTensor = self.fc_mu(features)
        log_var: keras.KerasTensor = self.fc_log_var(features)

        return mu, log_var

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for model serialization."""
        base_config = super().get_config()
        # Convert dataclass to dict for serialization
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
        }
        base_config.update({"config": config_dict})
        return base_config


@keras.saving.register_keras_serializable()
class MNISTReasoner(keras.Model):
    """
    Reasoner network for MNIST CCNet.
    """

    def __init__(
        self,
        config: ModelConfig,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MNIST Reasoner with configuration.

        Args:
            config: Model configuration object.
            **kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)
        self.config = config
        self.regularizer = (
            keras.regularizers.L2(config.reasoner_l2_regularization)
            if config.reasoner_l2_regularization > 0 else None
        )

        # Image processing layers
        self.conv_layers = []
        self.bn_layers = []
        self.act_layers = []
        self.pool_layers = []

        for i, (filters, kernel) in enumerate(zip(
            config.reasoner_conv_filters,
            config.reasoner_conv_kernels
        )):
            self.conv_layers.append(keras.layers.Conv2D(
                filters, kernel, padding="same",
                kernel_regularizer=self.regularizer,
                name=f"conv_{i+1}"
            ))
            self.bn_layers.append(keras.layers.BatchNormalization(name=f"bn_{i+1}"))
            self.act_layers.append(GoLU(
                name=f"act_{i+1}"
            ))
            self.pool_layers.append(keras.layers.MaxPooling2D(2, name=f"pool_{i+1}"))

        self.flatten = keras.layers.Flatten()

        # Build dense layers from config
        self.fc_layers = []
        self.bn_fc_layers = []
        self.act_fc_layers = []

        for i, units in enumerate(config.reasoner_dense_units):
            self.fc_layers.append(keras.layers.Dense(
                units, kernel_regularizer=self.regularizer,
                name=f"fc_{i+1}"
            ))
            self.bn_fc_layers.append(keras.layers.BatchNormalization(name=f"bn_fc_{i+1}"))
            self.act_fc_layers.append(GoLU(
                name=f"act_fc_{i+1}"
            ))

        self.dropout = keras.layers.Dropout(config.reasoner_dropout_rate)
        self.fc_output = keras.layers.Dense(
            config.num_classes,
            activation="softmax",
            kernel_regularizer=self.regularizer
        )

    def call(
        self,
        x: keras.KerasTensor,
        e: keras.KerasTensor,
        training: Optional[bool] = False
    ) -> keras.KerasTensor:
        """Forward pass through the reasoner network."""
        # Process image features
        img_features = x
        for conv, bn, act, pool in zip(
            self.conv_layers, self.bn_layers,
            self.act_layers, self.pool_layers
        ):
            img_features = conv(img_features)
            img_features = bn(img_features, training=training)
            img_features = act(img_features)
            img_features = pool(img_features)

        # Combine image features with explanations
        combined = keras.ops.concatenate(
            [self.flatten(img_features), e], axis=-1
        )

        # Classification pipeline through dense layers
        for fc, bn, act in zip(
            self.fc_layers, self.bn_fc_layers, self.act_fc_layers
        ):
            combined = fc(combined)
            combined = bn(combined, training=training)
            combined = act(combined)
            combined = self.dropout(combined, training=training)

        return self.fc_output(combined)

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for model serialization."""
        base_config = super().get_config()
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        base_config.update({"config": config_dict})
        return base_config


@keras.saving.register_keras_serializable()
class MNISTProducer(keras.Model):
    """
    Producer network for MNIST CCNet.
    """

    def __init__(
        self,
        config: ModelConfig,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MNIST Producer with configuration.

        Args:
            config: Model configuration object.
            **kwargs: Additional keras.Model arguments.
        """
        super().__init__(**kwargs)
        self.config = config

        # Class embedding
        self.label_embedding = keras.layers.Embedding(
            config.num_classes,
            config.producer_initial_dense_units
        )

        # Content generation
        spatial_units = (
            config.producer_initial_spatial_size *
            config.producer_initial_spatial_size *
            config.producer_initial_channels
        )
        self.fc_content = keras.layers.Dense(spatial_units, activation='relu')
        self.reshape_content = keras.layers.Reshape((
            config.producer_initial_spatial_size,
            config.producer_initial_spatial_size,
            config.producer_initial_channels
        ))

        # Style transformation layers
        self.style_transforms = []
        for units in config.producer_style_units:
            self.style_transforms.append(
                keras.layers.Dense(units, activation='relu')
            )

        # Upsampling and convolution layers
        self.up_layers = []
        self.conv_layers = []
        self.norm_layers = []
        self.act_layers = []

        for i, filters in enumerate(config.producer_conv_filters):
            self.up_layers.append(keras.layers.UpSampling2D(
                size=(2, 2), interpolation="nearest", name=f"up_{i+1}"
            ))
            self.conv_layers.append(keras.layers.Conv2D(
                filters, 3, padding="same", name=f"conv_{i+1}"
            ))
            self.norm_layers.append(keras.layers.BatchNormalization(name=f"norm_{i+1}"))
            self.act_layers.append(GoLU(
                name=f"act_{i+1}"
            ))

        self.conv_out_1 = keras.layers.Conv2D(
            32, 1, padding="same", activation="linear"
        )

        # Output layer
        self.conv_out_2 = keras.layers.Conv2D(
            1, 1, padding="same", activation="sigmoid"
        )

    def apply_style(
        self,
        content_tensor: keras.KerasTensor,
        style_vector: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply style modulation to content features."""
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
        """Forward pass through the producer network."""
        # Generate content from class embedding
        c: keras.KerasTensor = keras.ops.matmul(
            y, self.label_embedding.weights[0]
        )
        c = self.fc_content(c)
        c = self.reshape_content(c)

        # Prepare style vectors
        style_vectors = []
        style_input = e
        for transform in self.style_transforms:
            style_input = transform(style_input)
            style_vectors.append(style_input)

        # Apply upsampling blocks with style modulation
        x = c
        for i, (up, conv, norm, act) in enumerate(zip(
            self.up_layers, self.conv_layers,
            self.norm_layers, self.act_layers
        )):
            x = up(x)
            x = conv(x)
            x = norm(x, training=training)
            if i < len(style_vectors):
                x = self.apply_style(x, style_vectors[i])
            x = act(x)
        x = self.conv_out_1(x)
        return self.conv_out_2(x)

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for model serialization."""
        base_config = super().get_config()
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        base_config.update({"config": config_dict})
        return base_config


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def create_mnist_ccnet(config: ExperimentConfig) -> CCNetOrchestrator:
    """
    Create and initialize a CCNet for MNIST digit classification.

    Args:
        config: Experiment configuration object.

    Returns:
        CCNetOrchestrator: Configured CCNet orchestrator instance.
    """
    # Initialize model components with configuration
    explainer = MNISTExplainer(config.model)
    reasoner = MNISTReasoner(config.model)
    producer = MNISTProducer(config.model)

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

    # Configure CCNet from training config
    ccnet_config = CCNetConfig(
        explanation_dim=config.model.explanation_dim,
        loss_fn=config.training.loss_fn,
        loss_fn_params=config.training.loss_fn_params,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        kl_weight=config.training.kl_weight,
        dynamic_weighting=config.training.dynamic_weighting,
        use_mixed_precision=config.training.use_mixed_precision,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )

    # Create orchestrator
    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config
    )


def prepare_mnist_data(config: DataConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare MNIST dataset for training and validation.

    Args:
        config: Data configuration object.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    img_height, img_width = x_train.shape[1:]

    # Normalize and reshape images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Build augmentation pipeline with Keras layers
    augmentation_layers = []
    if config.apply_augmentation:
        if config.max_translation > 0:
            translation_factor = config.max_translation / img_height
            augmentation_layers.append(
                keras.layers.RandomTranslation(
                    height_factor=translation_factor,
                    width_factor=translation_factor,
                    fill_mode='constant',
                    fill_value=0.0
                )
            )
        if config.noise_stddev > 0:
            augmentation_layers.append(
                keras.layers.GaussianNoise(stddev=config.noise_stddev)
            )

    if augmentation_layers:
        augmentation_pipeline = keras.Sequential(augmentation_layers)
    else:
        augmentation_pipeline = lambda x, training: x

    # Create the training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(config.shuffle_buffer_size)
    train_ds = train_ds.map(
        lambda x, y: (augmentation_pipeline(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(config.batch_size).prefetch(config.prefetch_buffer_size)

    # Create the validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(config.batch_size).prefetch(config.prefetch_buffer_size)

    return train_ds, val_ds


# =====================================================================
# MAIN EXPERIMENT CLASS
# =====================================================================

class CCNetExperiment:
    """
    Experimental framework for CCNet training and visualization.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the CCNet experiment.

        Args:
            config: Experiment configuration object.
        """
        self.config = config
        self.viz_manager = VisualizationManager(
            experiment_name=config.visualization.experiment_name,
            output_dir=config.visualization.results_dir
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
        """Load and preprocess test data."""
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
        return x_test, y_test

    def run(self) -> Tuple[CCNetOrchestrator, CCNetTrainer]:
        """
        Execute the complete training and visualization pipeline.

        Returns:
            Tuple[CCNetOrchestrator, CCNetTrainer]: Trained models and trainer.
        """
        # Initialize model
        logger.info("Creating CCNet for MNIST...")
        orchestrator = create_mnist_ccnet(self.config)

        # Prepare data
        logger.info("Preparing data...")
        train_dataset, val_dataset = prepare_mnist_data(self.config.data)

        # Setup training
        logger.info("Setting up trainer...")
        trainer = CCNetTrainer(
            orchestrator,
            kl_annealing_epochs=self.config.training.kl_annealing_epochs
        )

        # Setup callbacks
        callbacks = []
        if self.config.early_stopping.enabled:
            callbacks.append(EarlyStoppingCallback(
                patience=self.config.early_stopping.patience,
                error_threshold=self.config.early_stopping.error_threshold,
                grad_stagnation_threshold=self.config.early_stopping.grad_stagnation_threshold
            ))

        # Execute training
        logger.info(f"Starting training for {self.config.training.epochs} epochs...")
        try:
            trainer.train(
                train_dataset,
                self.config.training.epochs,
                validation_dataset=val_dataset,
                callbacks=callbacks
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
        """Generate comprehensive visualization report."""
        logger.info("=" * 60)
        logger.info("Generating final visualizations...")
        logger.info("=" * 60)

        # Prepare visualization data with t-SNE config
        tsne_config = {
            "perplexity": self.config.visualization.tsne_perplexity,
            "max_iter": self.config.visualization.tsne_max_iter,
            "random_state": self.config.visualization.tsne_random_state
        }

        report_data: Dict[str, Any] = {
            'orchestrator': orchestrator,
            'x_data': self.x_test,
            'y_data': self.y_test,
            'num_samples': self.config.visualization.reconstruction_samples,
            'source_digits': self.config.visualization.counterfactual_source_digits,
            'target_digits': self.config.visualization.counterfactual_target_digits,
            'tsne_config': tsne_config
        }

        dashboard_recon_data: Dict[str, Any] = {
            'orchestrator': orchestrator,
            'x_data': self.x_test,
            'y_data': self.y_test,
            'num_samples': self.config.visualization.dashboard_reconstruction_samples
        }

        dashboard_latent_data: Dict[str, Any] = {
            'orchestrator': orchestrator,
            'x_data': self.x_test,
            'y_data': self.y_test,
            'num_samples': self.config.visualization.dashboard_latent_samples,
            'tsne_config': tsne_config
        }

        # Generate individual visualizations
        self.viz_manager.visualize(
            trainer.history,
            "ccnet_training_history",
            show=self.config.visualization.show_figures
        )

        self.viz_manager.visualize(
            report_data,
            "ccnet_reconstruction_quality",
            show=self.config.visualization.show_figures
        )

        self.viz_manager.visualize(
            report_data,
            "ccnet_counterfactual_matrix",
            show=self.config.visualization.show_figures
        )

        self.viz_manager.visualize(
            {**report_data, 'num_samples': self.config.visualization.latent_space_samples},
            "ccnet_latent_space",
            show=self.config.visualization.show_figures
        )

        # Create composite dashboard
        dashboard_data: Dict[str, Any] = {
            "ccnet_training_history": trainer.history,
            "ccnet_reconstruction_quality": dashboard_recon_data,
            "ccnet_latent_space": dashboard_latent_data,
        }

        self.viz_manager.create_dashboard(
            dashboard_data,
            show=self.config.visualization.show_figures
        )

        # Save trained models
        logger.info("Saving models...")
        models_dir = self.viz_manager.context.get_save_path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.save_models(str(models_dir / "mnist_ccnet"))
        logger.info(f"Models saved to {models_dir}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Create configuration
    config = ExperimentConfig(
        model=ModelConfig(
            explanation_dim=16,
            explainer_l2_regularization=1e-4,
            reasoner_dropout_rate=0.2
        ),
        training=TrainingConfig(
            epochs=100,
            learning_rates={
                'explainer': 1e-4,
                'reasoner': 1e-4,
                'producer': 3e-3
            },
            loss_fn='l2',
            kl_weight=0.1,
            dynamic_weighting=True
        ),
        data=DataConfig(
            batch_size=128,
            apply_augmentation=True,
            noise_stddev=0.01,
            max_translation=2
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=20,
            error_threshold=1e-5
        ),
        visualization=VisualizationConfig(
            experiment_name="ccnets_mnist",
            reconstruction_samples=5,
            latent_space_samples=1000,
            tsne_perplexity=30
        )
    )

    # Run experiment
    experiment = CCNetExperiment(config)
    orchestrator, trainer = experiment.run()