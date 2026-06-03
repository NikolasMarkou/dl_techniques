"""CCNet MNIST training script (the reference image task).

The MNIST CCNet architecture (`MNISTExplainer/Reasoner/Producer`, the shared
`FiLMLayer/ConvBlock/DenseBlock`, and the `create_mnist_ccnet` factory) now lives
in `dl_techniques.models.ccnets`; this script keeps only the training-side layer:
the training/data/visualization config dataclasses, MNIST data preparation,
visualization plugins, the `CCNetTrainer` driver, and model saving.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.train_mnist --gpu 0 --epochs 50
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

from train.common import setup_gpu, set_seeds
from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    CompositeVisualization,
)
from dl_techniques.models.ccnets import (
    CCNetTrainer,
    EarlyStoppingCallback,
)
from dl_techniques.models.ccnets.architectures.mnist import (
    ModelConfig,
    create_mnist_ccnet,
)
from dl_techniques.utils.logger import logger


# =====================================================================
# CONFIGURATION (training-side; architecture ModelConfig is imported)
# =====================================================================

@dataclass
class TrainingConfig:
    """Training parameters."""
    epochs: int = 50
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4}
    )
    loss_fn: str = 'l2'
    loss_fn_params: Dict[str, Any] = field(default_factory=dict)
    gradient_clip_norm: Optional[float] = 1.0
    dynamic_weighting: bool = False
    use_mixed_precision: bool = False
    kl_annealing_epochs: Optional[int] = 10

    explainer_weights: Dict[str, float] = field(
        default_factory=lambda: {'inference': 1.0, 'generation': 1.0, 'kl_divergence': 0.001}
    )
    reasoner_weights: Dict[str, float] = field(
        default_factory=lambda: {'reconstruction': 1.0, 'inference': 1.0}
    )
    producer_weights: Dict[str, float] = field(
        default_factory=lambda: {'generation': 1.0, 'reconstruction': 1.0}
    )


@dataclass
class DataConfig:
    """Data loading and augmentation parameters."""
    batch_size: int = 128
    shuffle_buffer_size: int = 60000
    prefetch_buffer_size: int = tf.data.AUTOTUNE
    apply_augmentation: bool = True
    noise_stddev: float = 0.05
    max_translation: int = 2


@dataclass
class EarlyStoppingConfig:
    """Early stopping parameters."""
    enabled: bool = True
    patience: int = 10
    error_threshold: float = 1e-4
    grad_stagnation_threshold: Optional[float] = 1e-4


@dataclass
class VisualizationConfig:
    """Visualization parameters."""
    experiment_name: str = "ccnets_mnist_corrected"
    results_dir: str = "results"
    save_figures: bool = True
    show_figures: bool = False
    reconstruction_samples: int = 5
    counterfactual_source_digits: List[int] = field(default_factory=lambda: [0, 1, 3, 5, 7])
    counterfactual_target_digits: List[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 9])
    latent_space_samples: int = 1000
    tsne_perplexity: int = 30
    tsne_max_iter: int = 300
    tsne_random_state: int = 42
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
    log_interval: int = 10
    verbose: bool = True


# =====================================================================
# VISUALIZATION PLUGINS
# =====================================================================

class CCNetTrainingHistoryViz(CompositeVisualization):
    """Plots all CCNet-specific losses, errors, and gradient norms."""

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_subplot("Fundamental Losses", self._plot_fundamental_losses)
        self.add_subplot("Model Errors", self._plot_model_errors)
        self.add_subplot("Gradient Norms", self._plot_gradient_norms)
        self.add_subplot("System Convergence", self._plot_system_convergence)

    def _plot_fundamental_losses(self, ax, data, **kwargs):
        for key in ["generation_loss", "reconstruction_loss", "inference_loss"]:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_loss", "").title(), linewidth=2)
        ax.set_title("Fundamental Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_errors(self, ax, data, **kwargs):
        for key in ["explainer_error", "reasoner_error", "producer_error"]:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_error", "").title(), linewidth=2)
        ax.set_title("Derived Model Errors")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(self, ax, data, **kwargs):
        for key in ["explainer_grad_norm", "reasoner_grad_norm", "producer_grad_norm"]:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_grad_norm", "").title(), linewidth=2)
        ax.set_title("Gradient Norms (L2)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_system_convergence(self, ax, data, **kwargs):
        required = ["generation_loss", "reconstruction_loss", "inference_loss"]
        if not all(k in data and data[k] for k in required):
            ax.text(0.5, 0.5, "Incomplete loss data.", ha='center', va='center')
            return
        total_error = [g + r + i for g, r, i in zip(
            data["generation_loss"], data["reconstruction_loss"], data["inference_loss"]
        )]
        ax.plot(total_error, linewidth=3, color="darkblue")
        ax.fill_between(range(len(total_error)), total_error, alpha=0.3)
        ax.set_title("Total System Error Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total System Error")
        ax.grid(True, alpha=0.3)


class ReconstructionQualityViz(VisualizationPlugin):
    """Compares original, reconstructed, generated images and errors."""

    @property
    def name(self) -> str:
        return "ccnet_reconstruction_quality"

    @property
    def description(self) -> str:
        return "Compares original, reconstructed, generated images and errors."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(
            k in data for k in ["orchestrator", "x_data", "y_data"]
        )

    def create_visualization(self, data, ax=None, **kwargs):
        orchestrator = data["orchestrator"]
        x_data = data["x_data"]
        y_data = data["y_data"]
        num_samples = data.get("num_samples", 5)

        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
        fig.suptitle("Reconstruction Quality Analysis", fontsize=16, fontweight="bold")

        for i in range(num_samples):
            x_input = x_data[i:i + 1]
            y_truth = keras.utils.to_categorical([y_data[i]], 10)
            tensors = orchestrator.forward_pass(x_input, y_truth, training=False)
            y_pred = np.argmax(tensors["y_inferred"][0])

            axes[i, 0].imshow(x_input[0, :, :, 0], cmap="gray")
            axes[i, 0].set_title(f"Original\nLabel: {y_data[i]}")

            axes[i, 1].imshow(tensors["x_reconstructed"][0, :, :, 0], cmap="gray")
            color = "green" if y_pred == y_data[i] else "red"
            axes[i, 1].set_title(f"Reconstructed\nInferred: {y_pred}", color=color)

            axes[i, 2].imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")
            axes[i, 2].set_title(f"Generated\nTrue: {y_data[i]}")

            diff_recon = np.abs(tensors["x_reconstructed"][0, :, :, 0] - x_input[0, :, :, 0])
            axes[i, 3].imshow(diff_recon, cmap="hot")
            axes[i, 3].set_title(f"Recon Error\nMAE: {np.mean(diff_recon):.3f}")

            diff_gen = np.abs(tensors["x_generated"][0, :, :, 0] - x_input[0, :, :, 0])
            axes[i, 4].imshow(diff_gen, cmap="hot")
            axes[i, 4].set_title(f"Gen Error\nMAE: {np.mean(diff_gen):.3f}")

            for j in range(5):
                axes[i, j].axis("off")

        return fig


class CounterfactualMatrixViz(VisualizationPlugin):
    """Generates a matrix of counterfactual digits."""

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

    def create_visualization(self, data, ax=None, **kwargs):
        orchestrator = data["orchestrator"]
        x_data = data["x_data"]
        y_data = data["y_data"]
        source_digits = data.get("source_digits", [0, 1, 3, 5, 7])
        target_digits = data.get("target_digits", [0, 2, 4, 6, 8, 9])

        rows, cols = len(source_digits), len(target_digits)
        fig, axes = plt.subplots(rows, cols + 1, figsize=(cols * 1.5 + 1.5, rows * 1.5))
        fig.suptitle(
            'Counterfactual Generation Matrix\n"What if this digit were drawn as..."',
            fontsize=16, fontweight="bold"
        )

        for i, source in enumerate(source_digits):
            idx = np.where(y_data == source)[0][0]
            x_source = x_data[idx:idx + 1]
            axes[i, 0].imshow(x_source[0, :, :, 0], cmap="gray")
            axes[i, 0].set_ylabel(f"{source} ->", rotation=0, size='large', labelpad=20)

            for j, target in enumerate(target_digits):
                y_target = keras.utils.to_categorical([target], 10)
                x_counter = orchestrator.counterfactual_generation(x_source, y_target)
                axes[i, j + 1].imshow(x_counter[0, :, :, 0], cmap="gray")
                if i == 0:
                    axes[i, j + 1].set_title(f"-> {target}", size='large')
                if source == target:
                    axes[i, j + 1].patch.set_edgecolor("green")
                    axes[i, j + 1].patch.set_linewidth(3)

        for axis in axes.flatten():
            axis.set_xticks([])
            axis.set_yticks([])

        return fig


class LatentSpaceAnalysisViz(CompositeVisualization):
    """Latent space analysis with t-SNE and statistics."""

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_vectors = None
        self.labels = None
        self.latent_2d = None
        self.tsne_config = None
        self.add_subplot("t-SNE Colored by Class", self._plot_tsne_class)
        self.add_subplot("t-SNE Density", self._plot_tsne_density)
        self.add_subplot("Mean Activation", self._plot_mean_activation)
        self.add_subplot("Latent Norms by Class", self._plot_latent_norms)

    def _prepare_data(self, data):
        if self.latent_vectors is not None:
            return

        orchestrator = data["orchestrator"]
        x_data = data["x_data"]
        y_data = data["y_data"]
        num_samples = data.get("num_samples", 1000)
        self.tsne_config = data.get(
            "tsne_config", {"perplexity": 30, "max_iter": 300, "random_state": 42}
        )

        indices = np.random.choice(len(x_data), min(num_samples, len(x_data)), replace=False)
        latent_vectors, labels = [], []
        for idx in indices:
            _, e_latent = orchestrator.disentangle_causes(x_data[idx:idx + 1])
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_data[idx])

        self.latent_vectors = np.array(latent_vectors)
        self.labels = np.array(labels)

        logger.info("Computing t-SNE embedding...")
        tsne = TSNE(
            n_components=2, perplexity=self.tsne_config["perplexity"],
            max_iter=self.tsne_config["max_iter"],
            random_state=self.tsne_config["random_state"]
        )
        self.latent_2d = tsne.fit_transform(self.latent_vectors)

    def _plot_tsne_class(self, ax, data, **kwargs):
        self._prepare_data(data)
        scatter = ax.scatter(
            self.latent_2d[:, 0], self.latent_2d[:, 1],
            c=self.labels, cmap="tab10", s=20, alpha=0.7
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Digit")
        ax.grid(True, alpha=0.3)

    def _plot_tsne_density(self, ax, data, **kwargs):
        self._prepare_data(data)
        xy = self.latent_2d.T
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(
            self.latent_2d[:, 0], self.latent_2d[:, 1],
            c=z, cmap="viridis", s=20, alpha=0.7
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Density")
        ax.grid(True, alpha=0.3)

    def _plot_mean_activation(self, ax, data, **kwargs):
        self._prepare_data(data)
        mean_act = np.mean(np.abs(self.latent_vectors), axis=0)
        ax.bar(range(len(mean_act)), mean_act, color="steelblue")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |Activation|")
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_latent_norms(self, ax, data, **kwargs):
        self._prepare_data(data)
        norms = np.linalg.norm(self.latent_vectors, axis=1)
        sns.violinplot(
            x=self.labels, y=norms, hue=self.labels,
            ax=ax, palette="husl", inner="quartile", legend=False
        )
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("||E|| (L2 Norm)")
        ax.grid(True, alpha=0.3, axis='y')


# =====================================================================
# DATA
# =====================================================================

def prepare_mnist_data(config: DataConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Prepare MNIST dataset for training and validation."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    h, w = x_train.shape[1:]

    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    aug_layers = []
    if config.apply_augmentation:
        if config.max_translation > 0:
            aug_layers.append(keras.layers.RandomTranslation(
                height_factor=config.max_translation / h,
                width_factor=config.max_translation / w,
                fill_mode='constant'
            ))
        if config.noise_stddev > 0:
            aug_layers.append(keras.layers.GaussianNoise(config.noise_stddev))

    aug_pipe = keras.Sequential(aug_layers) if aug_layers else lambda x, **_: x
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(config.shuffle_buffer_size)
    train_ds = train_ds.map(
        lambda x, y: (aug_pipe(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(config.batch_size).prefetch(config.prefetch_buffer_size)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(config.batch_size).prefetch(config.prefetch_buffer_size)

    return train_ds, val_ds


# =====================================================================
# EXPERIMENT
# =====================================================================

class CCNetExperiment:
    """Experimental framework for CCNet training and visualization."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.viz_manager = VisualizationManager(
            experiment_name=config.visualization.experiment_name,
            output_dir=config.visualization.results_dir
        )
        self._register_plugins()
        self.x_test, self.y_test = self._load_test_data()

    def _register_plugins(self) -> None:
        for plugin_class in [
            CCNetTrainingHistoryViz, ReconstructionQualityViz,
            CounterfactualMatrixViz, LatentSpaceAnalysisViz
        ]:
            instance = plugin_class(self.viz_manager.config, self.viz_manager.context)
            self.viz_manager.register_plugin(instance)

    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        (_, _), (x, y) = keras.datasets.mnist.load_data()
        return (x.astype("float32") / 255.0)[..., np.newaxis], y

    def run(self) -> Tuple[Any, CCNetTrainer]:
        """Execute the complete training and visualization pipeline."""
        logger.info("Creating CCNet for MNIST...")
        orchestrator = create_mnist_ccnet(
            self.config.model,
            loss_fn=self.config.training.loss_fn,
            loss_fn_params=self.config.training.loss_fn_params,
            learning_rates=self.config.training.learning_rates,
            gradient_clip_norm=self.config.training.gradient_clip_norm,
            use_mixed_precision=self.config.training.use_mixed_precision,
            explainer_weights=self.config.training.explainer_weights,
            reasoner_weights=self.config.training.reasoner_weights,
            producer_weights=self.config.training.producer_weights,
        )

        logger.info("Preparing data...")
        train_ds, val_ds = prepare_mnist_data(self.config.data)

        trainer = CCNetTrainer(
            orchestrator,
            kl_annealing_epochs=self.config.training.kl_annealing_epochs
        )

        callbacks = []
        if self.config.early_stopping.enabled:
            callbacks.append(EarlyStoppingCallback(
                patience=self.config.early_stopping.patience,
                error_threshold=self.config.early_stopping.error_threshold,
                grad_stagnation_threshold=self.config.early_stopping.grad_stagnation_threshold
            ))

        logger.info(f"Training for {self.config.training.epochs} epochs...")
        try:
            trainer.train(
                train_ds, self.config.training.epochs,
                validation_dataset=val_ds, callbacks=callbacks
            )
        except StopIteration:
            logger.info("Training stopped early due to convergence.")

        self.generate_final_report(orchestrator, trainer)
        logger.info(
            f"Visualizations saved to: "
            f"{self.viz_manager.context.output_dir / self.viz_manager.context.experiment_name}"
        )
        return orchestrator, trainer

    def generate_final_report(self, orchestrator, trainer) -> None:
        """Generate comprehensive visualization report."""
        vis_cfg = self.config.visualization
        tsne_cfg = {
            "perplexity": vis_cfg.tsne_perplexity,
            "max_iter": vis_cfg.tsne_max_iter,
            "random_state": vis_cfg.tsne_random_state
        }
        report_data = {
            'orchestrator': orchestrator,
            'x_data': self.x_test,
            'y_data': self.y_test,
            'num_samples': vis_cfg.reconstruction_samples,
            'source_digits': vis_cfg.counterfactual_source_digits,
            'target_digits': vis_cfg.counterfactual_target_digits,
            'tsne_config': tsne_cfg
        }

        self.viz_manager.visualize(trainer.history, "ccnet_training_history", show=vis_cfg.show_figures)
        self.viz_manager.visualize(report_data, "ccnet_reconstruction_quality", show=vis_cfg.show_figures)
        self.viz_manager.visualize(report_data, "ccnet_counterfactual_matrix", show=vis_cfg.show_figures)
        self.viz_manager.visualize(
            {**report_data, 'num_samples': vis_cfg.latent_space_samples},
            "ccnet_latent_space", show=vis_cfg.show_figures
        )

        models_dir = self.viz_manager.context.get_save_path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.save_models(str(models_dir / "mnist_ccnet"))
        logger.info(f"Models saved to {models_dir}")


# =====================================================================
# CONFIG BUILDER + ENTRY POINT
# =====================================================================

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Map parsed CLI args onto the experiment config dataclasses."""
    config = ExperimentConfig(
        model=ModelConfig(
            explanation_dim=32,
            explainer_l2_regularization=1e-5,
            reasoner_dropout_rate=0.25,
        ),
        training=TrainingConfig(
            epochs=100,
            learning_rates={'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4},
            loss_fn='l2',
            dynamic_weighting=False,
            kl_annealing_epochs=10,
            explainer_weights={'inference': 1.0, 'generation': 1.0, 'kl_divergence': 0.001},
        ),
        data=DataConfig(
            batch_size=128,
            apply_augmentation=True,
            noise_stddev=0.02,
            max_translation=2,
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=True, patience=10,
            error_threshold=1e-4, grad_stagnation_threshold=1e-4,
        ),
        visualization=VisualizationConfig(
            experiment_name="ccnets_mnist",
            reconstruction_samples=5,
            latent_space_samples=2000,
            tsne_perplexity=40,
        ),
    )

    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    if args.smoke:
        config.model.explanation_dim = 8
        config.training.epochs = 1
        config.training.kl_annealing_epochs = 1
        config.data.batch_size = 64
        config.data.shuffle_buffer_size = 1000
        config.early_stopping.enabled = False
        config.visualization.latent_space_samples = 100
        config.visualization.reconstruction_samples = 2
        config.visualization.tsne_perplexity = 5

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CCNet on MNIST.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index.")
    parser.add_argument('--epochs', type=int, default=None, help="Training epochs.")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--smoke', action='store_true',
                        help="Tiny CI-safe run (1 epoch, small dims).")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    set_seeds(args.seed)

    config = build_config(args)
    experiment = CCNetExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
