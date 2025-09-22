"""
CCNet MNIST Experiment using the Visualization Framework
=========================================================

This script demonstrates a complete training and analysis workflow for a
Counter-Causal Net (CCNet) on the MNIST dataset, fully integrated with the
visualization framework.
"""

import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from typing import Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
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
    """Visualizes the complete loss landscape of CCNet training."""

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

    def _plot_fundamental_losses(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        loss_keys = ["generation_loss", "reconstruction_loss", "inference_loss"]
        for key in loss_keys:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_loss", "").title(), linewidth=2)
        ax.set_title("Fundamental Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_errors(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        error_keys = ["explainer_error", "reasoner_error", "producer_error"]
        for key in error_keys:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_error", "").title(), linewidth=2)
        ax.set_title("Derived Model Errors")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        grad_keys = ["explainer_grad_norm", "reasoner_grad_norm", "producer_grad_norm"]
        for key in grad_keys:
            if key in data and data[key]:
                ax.plot(data[key], label=key.replace("_grad_norm", "").title(), linewidth=2)
        ax.set_title("Gradient Norms (L2)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_system_convergence(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        required_keys = ["generation_loss", "reconstruction_loss", "inference_loss"]
        if not all(k in data and data[k] for k in required_keys):
            ax.text(0.5, 0.5, "Incomplete loss data for convergence plot.", ha='center', va='center')
            return

        total_error = [
            g + r + i
            for g, r, i in zip(data["generation_loss"], data["reconstruction_loss"], data["inference_loss"])
        ]
        ax.plot(total_error, linewidth=3, color="darkblue")
        ax.fill_between(range(len(total_error)), total_error, alpha=0.3)
        ax.set_title("Total System Error Convergence")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total System Error")
        ax.grid(True, alpha=0.3)


class ReconstructionQualityViz(VisualizationPlugin):
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

    def create_visualization(self, data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs) -> plt.Figure:
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data, y_data = data["x_data"], data["y_data"]
        num_samples = data.get("num_samples", 5)

        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
        fig.suptitle("Reconstruction Quality Analysis", fontsize=16, fontweight="bold")

        for i in range(num_samples):
            x_input = x_data[i: i + 1]
            y_truth = keras.utils.to_categorical([y_data[i]], 10)
            tensors = orchestrator.forward_pass(x_input, y_truth, training=False)
            y_pred = np.argmax(tensors["y_inferred"][0])

            axes[i, 0].imshow(x_input[0, :, :, 0], cmap="gray")
            axes[i, 1].imshow(tensors["x_reconstructed"][0, :, :, 0], cmap="gray")
            axes[i, 2].imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")

            diff_recon = np.abs(tensors["x_reconstructed"][0, :, :, 0] - x_input[0, :, :, 0])
            axes[i, 3].imshow(diff_recon, cmap="hot")
            diff_gen = np.abs(tensors["x_generated"][0, :, :, 0] - x_input[0, :, :, 0])
            axes[i, 4].imshow(diff_gen, cmap="hot")

            axes[i, 0].set_title(f"Original\nLabel: {y_data[i]}")
            color = "green" if y_pred == y_data[i] else "red"
            axes[i, 1].set_title(f"Reconstructed\nInferred: {y_pred}", color=color)
            axes[i, 2].set_title(f"Generated\nTrue: {y_data[i]}")
            axes[i, 3].set_title(f"Recon Error\nMAE: {np.mean(diff_recon):.3f}")
            axes[i, 4].set_title(f"Gen Error\nMAE: {np.mean(diff_gen):.3f}")

            for j in range(5):
                axes[i, j].axis("off")
        return fig


class CounterfactualMatrixViz(VisualizationPlugin):
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

    def create_visualization(self, data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs) -> plt.Figure:
        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data, y_data = data["x_data"], data["y_data"]
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
            x_source = x_data[idx: idx + 1]
            axes[i, 0].imshow(x_source[0, :, :, 0], cmap="gray")
            axes[i, 0].set_ylabel(f"{source} →", rotation=0, size='large', labelpad=20)

            for j, target in enumerate(target_digits):
                y_target = keras.utils.to_categorical([target], 10)
                x_counter = orchestrator.counterfactual_generation(x_source, y_target)
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
    @property
    def name(self) -> str:
        return "ccnet_latent_space"

    @property
    def description(self) -> str:
        return "Visualizes latent space with t-SNE and statistical plots."

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and all(k in data for k in ["orchestrator", "x_data", "y_data"])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_vectors = None
        self.labels = None
        self.latent_2d = None
        self.add_subplot("t-SNE Colored by Class", self._plot_tsne_class)
        self.add_subplot("t-SNE Density", self._plot_tsne_density)
        self.add_subplot("Mean Activation", self._plot_mean_activation)
        self.add_subplot("Latent Norms by Class", self._plot_latent_norms)

    def _prepare_data(self, data: Dict[str, Any]):
        if self.latent_vectors is not None:
            return

        orchestrator: CCNetOrchestrator = data["orchestrator"]
        x_data, y_data = data["x_data"], data["y_data"]
        num_samples = data.get("num_samples", 1000)

        indices = np.random.choice(len(x_data), min(num_samples, len(x_data)), replace=False)

        latent_vectors, labels = [], []
        for idx in indices:
            x_sample = x_data[idx: idx + 1]
            _, e_latent = orchestrator.disentangle_causes(x_sample)
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_data[idx])

        self.latent_vectors = np.array(latent_vectors)
        self.labels = np.array(labels)

        logger.info("Computing t-SNE embedding for latent space...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
        self.latent_2d = tsne.fit_transform(self.latent_vectors)

    def _plot_tsne_class(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        self._prepare_data(data)
        scatter = ax.scatter(self.latent_2d[:, 0], self.latent_2d[:, 1], c=self.labels, cmap="tab10", s=20, alpha=0.7)
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Digit")
        ax.grid(True, alpha=0.3)

    def _plot_tsne_density(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        self._prepare_data(data)
        xy = self.latent_2d.T
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(self.latent_2d[:, 0], self.latent_2d[:, 1], c=z, cmap="viridis", s=20, alpha=0.7)
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.get_figure().colorbar(scatter, ax=ax, label="Density")
        ax.grid(True, alpha=0.3)

    def _plot_mean_activation(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        self._prepare_data(data)
        mean_activations = np.mean(np.abs(self.latent_vectors), axis=0)
        ax.bar(range(len(mean_activations)), mean_activations, color="steelblue")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |Activation|")
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_latent_norms(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        self._prepare_data(data)
        latent_norms = np.linalg.norm(self.latent_vectors, axis=1)
        sns.violinplot(x=self.labels, y=latent_norms, ax=ax, palette="husl", inner="quartile")
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("||E|| (L2 Norm)")
        ax.grid(True, alpha=0.3, axis='y')


# ---------------------------------------------------------------------
# Keras Model Definitions
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MNISTExplainer(keras.Model):
    def __init__(self, explanation_dim: int = 128, l2_regularization: float = 1e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.explanation_dim = explanation_dim
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None
        self.conv1 = keras.layers.Conv2D(32, 5, padding="same", kernel_regularizer=self.regularizer)
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool1 = keras.layers.MaxPooling2D(2)
        self.conv2 = keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=self.regularizer)
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool2 = keras.layers.MaxPooling2D(2)
        self.conv3 = keras.layers.Conv2D(128, 3, padding="same", kernel_regularizer=self.regularizer)
        self.bn3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool3 = keras.layers.GlobalMaxPool2D()
        self.flatten = keras.layers.Flatten()
        self.fc_mu = keras.layers.Dense(explanation_dim, name="mu", kernel_regularizer=self.regularizer)
        self.fc_log_var = keras.layers.Dense(explanation_dim, name="log_var", kernel_regularizer=self.regularizer)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = False) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        x = self.pool1(self.act1(self.bn1(self.conv1(x), training=training)))
        x = self.pool2(self.act2(self.bn2(self.conv2(x), training=training)))
        x = self.pool3(self.act3(self.bn3(self.conv3(x), training=training)))
        features = self.flatten(x)
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)
        return mu, log_var

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "explanation_dim": self.explanation_dim,
            "l2_regularization": self.regularizer.l2 if self.regularizer else 0.0
        })
        return config


@keras.saving.register_keras_serializable()
class MNISTReasoner(keras.Model):
    def __init__(self, num_classes: int = 10, explanation_dim: int = 128, dropout_rate: float = 0.2,
                 l2_regularization: float = 1e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_classes, self.explanation_dim, self.dropout_rate = num_classes, explanation_dim, dropout_rate
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None
        self.conv1 = keras.layers.Conv2D(32, 3, padding="same", kernel_regularizer=self.regularizer)
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool1 = keras.layers.MaxPooling2D(2)
        self.conv2 = keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=self.regularizer)
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool2 = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.fc_combine = keras.layers.Dense(512, kernel_regularizer=self.regularizer)
        self.bn_combine = keras.layers.BatchNormalization()
        self.act_combine = keras.layers.LeakyReLU(negative_slope=0.2)
        self.fc_hidden = keras.layers.Dense(256, kernel_regularizer=self.regularizer)
        self.bn_hidden = keras.layers.BatchNormalization()
        self.act_hidden = keras.layers.LeakyReLU(negative_slope=0.2)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.fc_output = keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=self.regularizer)

    def call(self, x: keras.KerasTensor, e: keras.KerasTensor, training: Optional[bool] = False) -> keras.KerasTensor:
        img_features = self.pool1(self.act1(self.bn1(self.conv1(x), training=training)))
        img_features = self.pool2(self.act2(self.bn2(self.conv2(img_features), training=training)))
        combined = keras.ops.concatenate([self.flatten(img_features), e], axis=-1)
        combined = self.dropout(self.act_combine(self.bn_combine(self.fc_combine(combined), training=training)),
                                training=training)
        combined = self.dropout(self.act_hidden(self.bn_hidden(self.fc_hidden(combined), training=training)),
                                training=training)
        return self.fc_output(combined)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "explanation_dim": self.explanation_dim,
                       "dropout_rate": self.dropout_rate,
                       "l2_regularization": self.regularizer.l2 if self.regularizer else 0.0})
        return config


@keras.saving.register_keras_serializable()
class MNISTProducer(keras.Model):
    def __init__(self, num_classes: int = 10, explanation_dim: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.label_embedding = keras.layers.Embedding(num_classes, 512)
        self.fc_content = keras.layers.Dense(7 * 7 * 128, activation='relu')
        self.reshape_content = keras.layers.Reshape((7, 7, 128))
        self.style_transform_1 = keras.layers.Dense(256, activation='relu')
        self.style_transform_2 = keras.layers.Dense(128, activation='relu')
        self.up1 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv1 = keras.layers.Conv2D(128, 3, padding="same")
        self.norm1 = keras.layers.BatchNormalization(center=False, scale=False)
        self.act1 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.up2 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = keras.layers.Conv2D(64, 3, padding="same")
        self.norm2 = keras.layers.BatchNormalization(center=False, scale=False)
        self.act2 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.conv_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")

    def apply_style(self, content_tensor, style_vector):
        num_channels = keras.ops.shape(content_tensor)[-1]
        style_params = style_vector
        scale = style_params[:, :num_channels]
        bias = style_params[:, num_channels:]
        scale = keras.ops.expand_dims(keras.ops.expand_dims(scale, 1), 1)
        bias = keras.ops.expand_dims(keras.ops.expand_dims(bias, 1), 1)
        return content_tensor * (scale + 1) + bias

    def call(self, y: keras.KerasTensor, e: keras.KerasTensor, training: Optional[bool] = False) -> keras.KerasTensor:
        # CORRECTED: Use differentiable matmul for soft lookup instead of argmax.
        c = keras.ops.matmul(y, self.label_embedding.weights[0])
        c = self.fc_content(c)
        c = self.reshape_content(c)

        style1 = self.style_transform_1(e)
        style2 = self.style_transform_2(e)

        x = self.up1(c)
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.apply_style(x, style1)
        x = self.act1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.apply_style(x, style2)
        x = self.act2(x)

        return self.conv_out(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "explanation_dim": self.explanation_dim
        })
        return config


# ---------------------------------------------------------------------
# Helper Functions for Data and Model Setup
# ---------------------------------------------------------------------


def create_mnist_ccnet(explanation_dim: int = 16, learning_rate: float = 1e-3) -> CCNetOrchestrator:
    dropout_rate = 0.2
    l2_regularization = 1e-4

    explainer = MNISTExplainer(explanation_dim, l2_regularization)
    reasoner = MNISTReasoner(10, explanation_dim, dropout_rate, l2_regularization)
    producer = MNISTProducer(num_classes=10, explanation_dim=explanation_dim)

    dummy_image = keras.ops.zeros((1, 28, 28, 1))
    dummy_label = keras.ops.zeros((1, 10))
    mu, _ = explainer(dummy_image)
    dummy_latent = mu
    _ = reasoner(dummy_image, dummy_latent)
    _ = producer(dummy_label, dummy_latent)

    config = CCNetConfig(
        explanation_dim=explanation_dim, loss_type="l2",
        learning_rates={"explainer": learning_rate, "reasoner": learning_rate, "producer": learning_rate},
        gradient_clip_norm=1.0,
        kl_weight=0.1
    )
    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=config
    )


def prepare_mnist_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = (d.astype("float32") / 255.0 for d in (x_train, x_test))
    x_train, x_test = (np.expand_dims(d, axis=-1) for d in (x_train, x_test))
    y_train, y_test = (keras.utils.to_categorical(d, 10) for d in (y_train, y_test))
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128).shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


# ---------------------------------------------------------------------
# Main Training & Visualization Workflow
# ---------------------------------------------------------------------


class CCNetExperiment:
    def __init__(self, experiment_name: str, results_dir: str = "results"):
        self.viz_manager = VisualizationManager(experiment_name=experiment_name, output_dir=results_dir)
        self._register_plugins()
        self.x_test, self.y_test = self._load_test_data()

    def _register_plugins(self):
        plugin_classes = [
            CCNetTrainingHistoryViz, ReconstructionQualityViz,
            CounterfactualMatrixViz, LatentSpaceAnalysisViz,
        ]
        for plugin_class in plugin_classes:
            instance = plugin_class(self.viz_manager.config, self.viz_manager.context)
            self.viz_manager.register_plugin(instance)

    def _load_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
        return x_test, y_test

    def run(self, epochs: int = 20) -> Tuple[CCNetOrchestrator, CCNetTrainer]:
        logger.info("Creating CCNet for MNIST...")
        orchestrator = create_mnist_ccnet(explanation_dim=16)
        logger.info("Preparing data...")
        train_dataset, val_dataset = prepare_mnist_data()
        logger.info("Setting up trainer...")
        trainer = CCNetTrainer(orchestrator)
        logger.info("Starting training...")
        try:
            trainer.train(
                train_dataset, epochs, validation_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(patience=5, error_threshold=1e-4)]
            )
        except StopIteration:
            logger.info("Training stopped early due to convergence.")
        self.generate_final_report(orchestrator, trainer)
        logger.info(
            f"All visualizations saved to: {self.viz_manager.context.output_dir / self.viz_manager.context.experiment_name}")
        return orchestrator, trainer

    def generate_final_report(self, orchestrator: CCNetOrchestrator, trainer: CCNetTrainer):
        logger.info("=" * 60)
        logger.info("Generating final visualizations...")
        logger.info("=" * 60)
        report_data = {'orchestrator': orchestrator, 'x_data': self.x_test, 'y_data': self.y_test}
        dashboard_recon_data = {**report_data, 'num_samples': 3}
        dashboard_latent_data = {**report_data, 'num_samples': 500}

        self.viz_manager.visualize(trainer.history, "ccnet_training_history", show=False)
        self.viz_manager.visualize(report_data, "ccnet_reconstruction_quality", show=False)
        self.viz_manager.visualize(report_data, "ccnet_counterfactual_matrix", show=False)
        self.viz_manager.visualize(report_data, "ccnet_latent_space", show=False)

        dashboard_data = {
            "ccnet_training_history": trainer.history,
            "ccnet_reconstruction_quality": dashboard_recon_data,
            "ccnet_latent_space": dashboard_latent_data,
        }
        self.viz_manager.create_dashboard(dashboard_data, show=False)

        logger.info("Saving models...")
        models_dir = self.viz_manager.context.get_save_path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.save_models(str(models_dir / "mnist_ccnet"))
        logger.info(f"Models saved to {models_dir}")


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------


if __name__ == "__main__":
    experiment = CCNetExperiment(experiment_name="ccnets_mnist")
    orchestrator, trainer = experiment.run(epochs=50)