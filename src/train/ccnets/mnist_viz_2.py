from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

from dl_techniques.models.ccnets import (
    CCNetConfig,
    CCNetOrchestrator,
    CCNetTrainer,
    EarlyStoppingCallback,
    wrap_keras_model,
)
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# =============================================================================
# Visualization Utilities
# =============================================================================


class CCNetVisualizer:
    """
    Comprehensive visualization utilities for CCNet analysis.

    This class provides methods for visualizing various aspects of CCNet training
    and performance, including loss curves, reconstructions, counterfactuals,
    style transfer, and latent space analysis.

    Attributes:
        results_dir: Directory to save visualization results.
        timestamp: Timestamp for the current experiment.
        experiment_dir: Directory for the current experiment's results.

    Example:
        >>> visualizer = CCNetVisualizer(results_dir="results")
        >>> visualizer.plot_training_history(trainer.history)
        >>> visualizer.visualize_counterfactual_generation(orchestrator, x_test, y_test)
    """

    def __init__(self, results_dir: str = "results") -> None:
        """
        Initialize visualizer with results directory.

        Args:
            results_dir: Directory to save visualization results.
        """
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"ccnet_mnist_{self.timestamp}"

        # Create directory structure
        self._create_directories()
        logger.info(f"Visualization results will be saved to: {self.experiment_dir}")

    def _create_directories(self) -> None:
        """Create necessary directory structure for saving visualizations."""
        directories = [
            "losses",
            "counterfactuals",
            "style_transfer",
            "reconstructions",
            "latent_space",
            "generation_quality",
        ]

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (self.experiment_dir / directory).mkdir(exist_ok=True)

    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot comprehensive training history with all losses and errors.

        Args:
            history: Training history dictionary containing loss values.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Plot fundamental losses
        loss_configs = [
            (0, 0, "generation_loss", "Generation Loss", "||X_generated - X_input||", "blue"),
            (0, 1, "reconstruction_loss", "Reconstruction Loss",
             "||X_reconstructed - X_input||", "orange"),
            (0, 2, "inference_loss", "Inference Loss",
             "||X_reconstructed - X_generated||", "green"),
        ]

        for row, col, key, title, subtitle, color in loss_configs:
            ax = fig.add_subplot(gs[row, col])
            ax.plot(history[key], label=title.split()[0], linewidth=2, color=color)
            ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot model errors
        error_configs = [
            (1, 0, "explainer_error", "Explainer Error", "Inf + Gen - Rec", "red"),
            (1, 1, "reasoner_error", "Reasoner Error", "Rec + Inf - Gen", "purple"),
            (1, 2, "producer_error", "Producer Error", "Gen + Rec - Inf", "brown"),
        ]

        for row, col, key, title, subtitle, color in error_configs:
            ax = fig.add_subplot(gs[row, col])
            ax.plot(history[key], label=title.split()[0], linewidth=2, color=color)
            ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "CCNet Training History - Complete Loss Landscape",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        save_path = self.experiment_dir / "losses" / "training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Training history saved to {save_path}")

    def plot_convergence_analysis(self, history: Dict[str, List[float]]) -> None:
        """
        Plot convergence analysis showing how losses converge to equilibrium.

        Args:
            history: Training history dictionary containing loss values.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Combined losses plot
        ax = axes[0]
        loss_types = [
            ("generation_loss", "Generation", "blue"),
            ("reconstruction_loss", "Reconstruction", "orange"),
            ("inference_loss", "Inference", "green"),
        ]

        for key, label, color in loss_types:
            ax.plot(history[key], label=label, linewidth=2, alpha=0.8, color=color)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Loss Convergence Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Total system error
        ax = axes[1]
        total_error = self._calculate_total_error(history)

        ax.plot(total_error, linewidth=3, color="darkblue")
        ax.fill_between(range(len(total_error)), total_error, alpha=0.3)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Total System Error", fontsize=12)
        ax.set_title("CCNet System Convergence", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add convergence threshold line
        if total_error:
            convergence_threshold = min(total_error) * 1.1
            ax.axhline(
                y=convergence_threshold,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="Convergence Threshold",
            )
            ax.legend()

        plt.suptitle(
            "CCNet Convergence Analysis", fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        save_path = self.experiment_dir / "losses" / "convergence_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Convergence analysis saved")

    def _calculate_total_error(
        self, history: Dict[str, List[float]]
    ) -> List[float]:
        """Calculate total system error from individual losses."""
        return [
            g + r + i
            for g, r, i in zip(
                history["generation_loss"],
                history["reconstruction_loss"],
                history["inference_loss"],
            )
        ]

    def visualize_reconstruction_quality(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int = 10,
    ) -> None:
        """
        Visualize reconstruction quality comparing original, reconstructed, and generated images.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images of shape (n_samples, 28, 28, 1).
            y_test: Test labels of shape (n_samples,).
            num_samples: Number of samples to visualize.
        """
        fig = plt.figure(figsize=(15, 3 * num_samples))
        gs = gridspec.GridSpec(
            num_samples, 5, figure=fig, wspace=0.3, hspace=0.4
        )

        for i in range(num_samples):
            x_input = x_test[i : i + 1]
            y_truth = keras.utils.to_categorical([y_test[i]], 10)

            # Get all outputs
            tensors = orchestrator.forward_pass(
                x_input, y_truth, training=False
            )

            self._plot_reconstruction_row(
                fig, gs, i, x_input, tensors, y_test[i]
            )

        plt.suptitle(
            "Reconstruction Quality Analysis",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )

        save_path = self.experiment_dir / "reconstructions" / "quality_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Reconstruction quality visualization saved")

    def _plot_reconstruction_row(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        row_idx: int,
        x_input: np.ndarray,
        tensors: Dict[str, tf.Tensor],
        true_label: int,
    ) -> None:
        """Plot a single row in the reconstruction quality visualization."""
        # Original
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(x_input[0, :, :, 0], cmap="gray")
        ax.set_title(f"Original\nLabel: {true_label}", fontsize=10)
        ax.axis("off")

        # Reconstructed
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(tensors["x_reconstructed"][0, :, :, 0], cmap="gray")
        y_pred = np.argmax(tensors["y_inferred"][0])
        color = "green" if y_pred == true_label else "red"
        ax.set_title(f"Reconstructed\nInferred: {y_pred}", fontsize=10, color=color)
        ax.axis("off")

        # Generated
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")
        ax.set_title(f"Generated\nTrue: {true_label}", fontsize=10)
        ax.axis("off")

        # Difference maps
        self._plot_difference_maps(
            fig, gs, row_idx, x_input, tensors
        )

    def _plot_difference_maps(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        row_idx: int,
        x_input: np.ndarray,
        tensors: Dict[str, tf.Tensor],
    ) -> None:
        """Plot difference maps for reconstruction and generation errors."""
        # Reconstruction error
        ax = fig.add_subplot(gs[row_idx, 3])
        diff_recon = np.abs(
            tensors["x_reconstructed"][0, :, :, 0] - x_input[0, :, :, 0]
        )
        ax.imshow(diff_recon, cmap="hot")
        ax.set_title(f"Recon Error\nMAE: {np.mean(diff_recon):.3f}", fontsize=10)
        ax.axis("off")

        # Generation error
        ax = fig.add_subplot(gs[row_idx, 4])
        diff_gen = np.abs(
            tensors["x_generated"][0, :, :, 0] - x_input[0, :, :, 0]
        )
        ax.imshow(diff_gen, cmap="hot")
        ax.set_title(f"Gen Error\nMAE: {np.mean(diff_gen):.3f}", fontsize=10)
        ax.axis("off")

    def visualize_counterfactual_generation(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Visualize counterfactual generation capabilities.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images of shape (n_samples, 28, 28, 1).
            y_test: Test labels of shape (n_samples,).
        """
        self._plot_counterfactual_matrix(orchestrator, x_test, y_test)
        self._plot_counterfactual_sequence(orchestrator, x_test, y_test)

    def _plot_counterfactual_matrix(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Create a grid showing counterfactual transformations."""
        source_digits = [0, 1, 3, 5, 7]
        target_digits = [0, 2, 4, 6, 8, 9]

        fig = plt.figure(
            figsize=(len(target_digits) * 2 + 2, len(source_digits) * 2 + 1)
        )
        gs = gridspec.GridSpec(
            len(source_digits) + 1,
            len(target_digits) + 1,
            figure=fig,
            wspace=0.1,
            hspace=0.1,
        )

        # Headers
        self._add_counterfactual_headers(fig, gs, target_digits)

        # Generate and plot counterfactuals
        for i, source in enumerate(source_digits):
            self._plot_counterfactual_row(
                fig, gs, i, source, target_digits, orchestrator, x_test, y_test
            )

        plt.suptitle(
            'Counterfactual Generation Matrix\n"What if this digit were drawn as..."',
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        save_path = self.experiment_dir / "counterfactuals" / "counterfactual_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Counterfactual matrix saved")

    def _add_counterfactual_headers(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        target_digits: List[int],
    ) -> None:
        """Add column headers for counterfactual matrix."""
        for j, target in enumerate(target_digits):
            ax = fig.add_subplot(gs[0, j + 1])
            ax.text(
                0.5, 0.5, f"→ {target}",
                ha="center", va="center", fontsize=14, fontweight="bold"
            )
            ax.axis("off")

    def _plot_counterfactual_row(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        row_idx: int,
        source_digit: int,
        target_digits: List[int],
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Plot a single row of counterfactual transformations."""
        # Row header (original)
        idx = np.where(y_test == source_digit)[0][0]
        x_source = x_test[idx : idx + 1]

        ax = fig.add_subplot(gs[row_idx + 1, 0])
        ax.imshow(x_source[0, :, :, 0], cmap="gray")
        ax.set_title(f"{source_digit} →", fontsize=12, fontweight="bold", rotation=0)
        ax.axis("off")

        # Generate counterfactuals
        for j, target in enumerate(target_digits):
            y_target = keras.ops.zeros((1, 10))
            y_target = keras.ops.scatter_update(y_target, [[0, target]], [1.0])

            x_counterfactual = orchestrator.counterfactual_generation(
                x_source, y_target
            )

            ax = fig.add_subplot(gs[row_idx + 1, j + 1])
            ax.imshow(x_counterfactual[0, :, :, 0], cmap="gray")

            # Highlight diagonal (same digit)
            if source_digit == target:
                ax.patch.set_edgecolor("green")
                ax.patch.set_linewidth(3)

            ax.axis("off")

    def _plot_counterfactual_sequence(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Plot detailed counterfactual transformation sequence."""
        source_digit = 3
        idx = np.where(y_test == source_digit)[0][2]  # Third instance
        x_source = x_test[idx : idx + 1]

        fig = plt.figure(figsize=(20, 4))
        gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.3)

        # Show endpoints
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(x_source[0, :, :, 0], cmap="gray")
        ax.set_title("Original\n(3)", fontsize=11, fontweight="bold", color="blue")
        ax.axis("off")

        # Generate all counterfactuals
        for target in range(10):
            row = 0 if target < 5 else 1
            col = (target % 5) * 2 + 1

            y_target = keras.ops.zeros((1, 10))
            y_target = keras.ops.scatter_update(y_target, [[0, target]], [1.0])

            x_counter = orchestrator.counterfactual_generation(x_source, y_target)

            ax = fig.add_subplot(gs[row, col : col + 2])
            ax.imshow(x_counter[0, :, :, 0], cmap="gray")
            color = "green" if target == source_digit else "black"
            ax.set_title(f'As "{target}"', fontsize=11, color=color)
            ax.axis("off")

        plt.suptitle(
            "Counterfactual Sequence: Preserving Style Across All Digits",
            fontsize=14,
            fontweight="bold",
            y=1.05,
        )

        save_path = self.experiment_dir / "counterfactuals" / "counterfactual_sequence.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Counterfactual sequence saved")

    def visualize_style_transfer(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Visualize style transfer capabilities.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images of shape (n_samples, 28, 28, 1).
            y_test: Test labels of shape (n_samples,).
        """
        content_digits = [1, 2, 7, 8]
        style_digits = [3, 4, 5, 9]

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(
            len(content_digits),
            len(style_digits) + 2,
            figure=fig,
            wspace=0.2,
            hspace=0.3,
        )

        for i, content_digit in enumerate(content_digits):
            self._plot_style_transfer_row(
                fig, gs, i, content_digit, style_digits,
                orchestrator, x_test, y_test
            )

        plt.suptitle(
            "Style Transfer Matrix: Content × Style",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        save_path = self.experiment_dir / "style_transfer" / "style_transfer_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Style transfer matrix saved")

    def _plot_style_transfer_row(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        row_idx: int,
        content_digit: int,
        style_digits: List[int],
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Plot a single row of style transfer examples."""
        # Content source
        idx_content = np.where(y_test == content_digit)[0][0]
        x_content = x_test[idx_content : idx_content + 1]

        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(x_content[0, :, :, 0], cmap="gray")
        ax.set_title(f"Content\n({content_digit})", fontsize=11, fontweight="bold")
        ax.axis("off")

        # Arrow
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.text(0.5, 0.5, "→", ha="center", va="center", fontsize=20)
        ax.axis("off")

        for j, style_digit in enumerate(style_digits):
            # Style source
            idx_style = np.where(y_test == style_digit)[0][1]
            x_style = x_test[idx_style : idx_style + 1]

            # Perform style transfer
            x_transferred = orchestrator.style_transfer(x_content, x_style)

            ax = fig.add_subplot(gs[row_idx, j + 2])
            ax.imshow(x_transferred[0, :, :, 0], cmap="gray")

            if row_idx == 0:
                # Show style source label above
                ax.set_title(f"Style: {style_digit}", fontsize=10)

            ax.axis("off")

    def visualize_latent_space(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int = 1000,
    ) -> None:
        """
        Visualize the learned latent space using t-SNE.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images of shape (n_samples, 28, 28, 1).
            y_test: Test labels of shape (n_samples,).
            num_samples: Number of samples to visualize.
        """
        # Extract latent representations
        latent_vectors, labels = self._extract_latent_vectors(
            orchestrator, x_test, y_test, num_samples
        )

        # Apply t-SNE
        logger.info("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent_vectors)

        # Create visualizations
        self._plot_latent_space_tsne(latent_2d, labels)
        self._plot_latent_statistics(latent_vectors, labels)

    def _extract_latent_vectors(
        self,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract latent vectors from test samples."""
        num_samples = min(num_samples, len(x_test))
        indices = np.random.choice(len(x_test), num_samples, replace=False)

        latent_vectors = []
        labels = []

        for idx in indices:
            x_sample = x_test[idx : idx + 1]
            _, e_latent = orchestrator.disentangle_causes(x_sample)
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_test[idx])

        return np.array(latent_vectors), np.array(labels)

    def _plot_latent_space_tsne(
        self, latent_2d: np.ndarray, labels: np.ndarray
    ) -> None:
        """Plot t-SNE visualization of latent space."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Colored by digit class
        ax = axes[0]
        scatter = ax.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=labels,
            cmap="tab10",
            s=20,
            alpha=0.6,
        )
        ax.set_title(
            "Latent Space Colored by Digit Class",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        plt.colorbar(scatter, ax=ax, label="Digit")

        # Density plot
        ax = axes[1]
        xy = latent_2d.T
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=z,
            cmap="viridis",
            s=20,
            alpha=0.6,
        )
        ax.set_title("Latent Space Density", fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        plt.colorbar(scatter, ax=ax, label="Density")

        plt.suptitle(
            "CCNet Latent Space Visualization (E vectors)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        save_path = self.experiment_dir / "latent_space" / "tsne_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Latent space visualization saved")

    def _plot_latent_statistics(
        self, latent_vectors: np.ndarray, labels: np.ndarray
    ) -> None:
        """Plot statistics of the latent space."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Latent dimension activation patterns
        ax = axes[0, 0]
        mean_activations = np.mean(np.abs(latent_vectors), axis=0)
        ax.bar(range(len(mean_activations)), mean_activations, color="steelblue")
        ax.set_title(
            "Mean Activation per Latent Dimension",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |Activation|")
        ax.grid(True, alpha=0.3)

        # Variance per dimension
        ax = axes[0, 1]
        var_activations = np.var(latent_vectors, axis=0)
        ax.bar(range(len(var_activations)), var_activations, color="coral")
        ax.set_title(
            "Variance per Latent Dimension", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Variance")
        ax.grid(True, alpha=0.3)

        # Per-class latent norms
        ax = axes[1, 0]
        latent_norms = np.linalg.norm(latent_vectors, axis=1)
        for digit in range(10):
            digit_norms = latent_norms[labels == digit]
            ax.violinplot(
                [digit_norms],
                positions=[digit],
                widths=0.7,
                showmeans=True,
                showmedians=True,
            )
        ax.set_title(
            "Latent Vector Norms by Digit Class",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Digit")
        ax.set_ylabel("||E|| (L2 Norm)")
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis="y")

        # Correlation matrix of top dimensions
        ax = axes[1, 1]
        top_k = 10
        top_dims = np.argsort(mean_activations)[-top_k:]
        corr_matrix = np.corrcoef(latent_vectors[:, top_dims].T)
        im = ax.imshow(
            corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
        )
        ax.set_title(
            f"Correlation Matrix (Top {top_k} Dimensions)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Dimension Index")
        ax.set_ylabel("Dimension Index")
        plt.colorbar(im, ax=ax, label="Correlation")

        plt.suptitle(
            "Latent Space Statistics Analysis",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        save_path = self.experiment_dir / "latent_space" / "latent_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Latent statistics saved")

    def create_summary_report(
        self,
        orchestrator: CCNetOrchestrator,
        trainer: CCNetTrainer,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Create a comprehensive summary report.

        Args:
            orchestrator: Trained CCNet orchestrator.
            trainer: CCNet trainer with history.
            x_test: Test images of shape (n_samples, 28, 28, 1).
            y_test: Test labels of shape (n_samples,).
        """
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Add various summary components
        self._add_training_curves(fig, gs, trainer)
        self._add_final_metrics(fig, gs, trainer)
        self._add_reconstruction_examples(fig, gs, orchestrator, x_test, y_test)
        self._add_counterfactual_examples(fig, gs, orchestrator, x_test, y_test)
        self._add_consistency_check(fig, gs, orchestrator, x_test)
        self._add_model_info(fig, gs, orchestrator, trainer)

        plt.suptitle(
            "CCNet Training & Evaluation Summary Report",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        save_path = self.experiment_dir / "summary_report.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Summary report saved to {save_path}")

    def _add_training_curves(
        self, fig: plt.Figure, gs: gridspec.GridSpec, trainer: CCNetTrainer
    ) -> None:
        """Add training curves to summary report."""
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(trainer.history["generation_loss"], label="Generation", linewidth=2)
        ax.plot(
            trainer.history["reconstruction_loss"],
            label="Reconstruction",
            linewidth=2
        )
        ax.plot(trainer.history["inference_loss"], label="Inference", linewidth=2)
        ax.set_title("Training Progress", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _add_final_metrics(
        self, fig: plt.Figure, gs: gridspec.GridSpec, trainer: CCNetTrainer
    ) -> None:
        """Add final metrics to summary report."""
        ax = fig.add_subplot(gs[0, 2:])

        final_metrics = {
            "Generation": trainer.history["generation_loss"][-1]
            if trainer.history["generation_loss"]
            else 0,
            "Reconstruction": trainer.history["reconstruction_loss"][-1]
            if trainer.history["reconstruction_loss"]
            else 0,
            "Inference": trainer.history["inference_loss"][-1]
            if trainer.history["inference_loss"]
            else 0,
        }

        bars = ax.bar(
            final_metrics.keys(),
            final_metrics.values(),
            color=["blue", "orange", "green"],
        )
        ax.set_title("Final Loss Values", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss")

        for bar, value in zip(bars, final_metrics.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )
        ax.grid(True, alpha=0.3, axis="y")

    def _add_reconstruction_examples(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add reconstruction examples to summary report."""
        ax = fig.add_subplot(gs[1:3, :])
        ax.axis("off")
        ax.text(
            0.5,
            0.98,
            "Reconstruction Examples",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Create sub-grid for examples
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            4, 5, subplot_spec=gs[1:3, :], wspace=0.1, hspace=0.2
        )

        for i in range(4):
            idx = np.random.randint(0, len(x_test))
            x_sample = x_test[idx : idx + 1]
            y_truth = keras.utils.to_categorical([y_test[idx]], 10)

            tensors = orchestrator.forward_pass(x_sample, y_truth, training=False)

            # Show visualizations
            self._add_reconstruction_example_row(
                fig, sub_gs, i, x_sample, tensors
            )

    def _add_reconstruction_example_row(
        self,
        fig: plt.Figure,
        sub_gs: gridspec.GridSpecFromSubplotSpec,
        row_idx: int,
        x_sample: np.ndarray,
        tensors: Dict[str, tf.Tensor],
    ) -> None:
        """Add a single row of reconstruction examples."""
        titles = ["Original", "Reconstructed", "Generated", "Recon Error", "Gen Error"]

        for col_idx, title in enumerate(titles):
            ax_sub = fig.add_subplot(sub_gs[row_idx, col_idx])

            if col_idx == 0:  # Original
                ax_sub.imshow(x_sample[0, :, :, 0], cmap="gray")
            elif col_idx == 1:  # Reconstructed
                ax_sub.imshow(tensors["x_reconstructed"][0, :, :, 0], cmap="gray")
            elif col_idx == 2:  # Generated
                ax_sub.imshow(tensors["x_generated"][0, :, :, 0], cmap="gray")
            elif col_idx == 3:  # Recon Error
                diff = np.abs(
                    tensors["x_reconstructed"][0, :, :, 0] - x_sample[0, :, :, 0]
                )
                ax_sub.imshow(diff, cmap="hot")
            else:  # Gen Error
                diff = np.abs(
                    tensors["x_generated"][0, :, :, 0] - x_sample[0, :, :, 0]
                )
                ax_sub.imshow(diff, cmap="hot")

            if row_idx == 0:
                ax_sub.set_title(title, fontsize=9)
            ax_sub.axis("off")

    def _add_counterfactual_examples(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add counterfactual examples to summary report."""
        ax = fig.add_subplot(gs[3:5, :])
        ax.axis("off")
        ax.text(
            0.5,
            0.98,
            "Counterfactual Generation Examples",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )

        sub_gs2 = gridspec.GridSpecFromSubplotSpec(
            2, 10, subplot_spec=gs[3:5, :], wspace=0.1, hspace=0.2
        )

        # Show counterfactual transformation
        idx_3 = np.where(y_test == 3)[0][0]
        x_3 = x_test[idx_3 : idx_3 + 1]

        for i in range(10):
            row = i // 5
            col = i % 5 * 2

            y_target = keras.ops.zeros((1, 10))
            y_target = keras.ops.scatter_update(y_target, [[0, i]], [1.0])
            x_counter = orchestrator.counterfactual_generation(x_3, y_target)

            ax_sub = fig.add_subplot(sub_gs2[row, col : col + 2])
            ax_sub.imshow(x_counter[0, :, :, 0], cmap="gray")
            ax_sub.set_title(f"{i}", fontsize=9)
            ax_sub.axis("off")

    def _add_consistency_check(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        orchestrator: CCNetOrchestrator,
        x_test: np.ndarray,
    ) -> None:
        """Add consistency check to summary report."""
        ax = fig.add_subplot(gs[5, :2])

        consistency_results = []
        for _ in range(100):
            idx = np.random.randint(0, min(1000, len(x_test)))
            is_consistent = orchestrator.verify_consistency(
                x_test[idx : idx + 1], threshold=0.05
            )
            consistency_results.append(is_consistent)

        consistency_rate = sum(consistency_results) / len(consistency_results)

        ax.pie(
            [consistency_rate, 1 - consistency_rate],
            labels=["Consistent", "Inconsistent"],
            colors=["green", "red"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(
            f"Consistency Check\n({len(consistency_results)} samples)",
            fontsize=12,
            fontweight="bold",
        )

    def _add_model_info(
        self,
        fig: plt.Figure,
        gs: gridspec.GridSpec,
        orchestrator: CCNetOrchestrator,
        trainer: CCNetTrainer,
    ) -> None:
        """Add model information to summary report."""
        ax = fig.add_subplot(gs[5, 2:])
        ax.axis("off")

        final_gen = (
            trainer.history["generation_loss"][-1]
            if trainer.history["generation_loss"]
            else 0
        )
        final_rec = (
            trainer.history["reconstruction_loss"][-1]
            if trainer.history["reconstruction_loss"]
            else 0
        )
        final_inf = (
            trainer.history["inference_loss"][-1]
            if trainer.history["inference_loss"]
            else 0
        )

        info_text = f"""
        Model Configuration:
        • Explanation Dimension: {orchestrator.config.explanation_dim}
        • Loss Type: {orchestrator.config.loss_type}
        • Learning Rates: {orchestrator.config.learning_rates}
        • Gradient Clipping: {orchestrator.config.gradient_clip_norm}
        • Verification Weight: {orchestrator.config.verification_weight}

        Training Summary:
        • Total Epochs: {len(trainer.history['generation_loss'])}
        • Final Gen Loss: {final_gen:.4f}
        • Final Rec Loss: {final_rec:.4f}
        • Final Inf Loss: {final_inf:.4f}
        """

        ax.text(
            0.1,
            0.9,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )


# =============================================================================
# Model Definitions
# =============================================================================


@keras.saving.register_keras_serializable()
class MNISTExplainer(keras.Model):
    """
    Explainer network for MNIST: P(E|X).

    Extracts latent style/context from digit images using a modern CNN architecture.
    The explainer learns to disentangle style factors from digit identity.

    Args:
        explanation_dim: Dimension of the latent explanation vector.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization strength for weights.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, 28, 28, 1)`.

    Output shape:
        2D tensor with shape: `(batch_size, explanation_dim)`.
        L2-normalized latent vectors representing style/context.

    Example:
        >>> explainer = MNISTExplainer(explanation_dim=128, dropout_rate=0.2)
        >>> x = keras.ops.random.normal(shape=(32, 28, 28, 1))
        >>> e_latent = explainer(x, training=True)
        >>> assert e_latent.shape == (32, 128)
    """

    def __init__(
        self,
        explanation_dim: int = 128,
        dropout_rate: float = 0.2,
        l2_regularization: float = 1e-4,
        **kwargs: Any
    ) -> None:
        """Initialize the Explainer network."""
        super().__init__(**kwargs)

        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = (
            keras.regularizers.L2(l2_regularization)
            if l2_regularization > 0
            else None
        )

        # Create all layers in __init__ (following modern Keras patterns)
        # Convolutional blocks
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

        self.conv3 = keras.layers.Conv2D(
            128, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.LeakyReLU(negative_slope=0.2)
        self.pool3 = keras.layers.MaxPooling2D(2)

        # Dense layers
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(dropout_rate)

        self.fc1 = keras.layers.Dense(
            512, kernel_regularizer=self.regularizer
        )
        self.bn_fc1 = keras.layers.BatchNormalization()
        self.act_fc1 = keras.layers.LeakyReLU(negative_slope=0.2)

        self.fc2 = keras.layers.Dense(
            256, kernel_regularizer=self.regularizer
        )
        self.bn_fc2 = keras.layers.BatchNormalization()
        self.act_fc2 = keras.layers.LeakyReLU(negative_slope=0.2)

        self.fc_latent = keras.layers.Dense(
            explanation_dim, kernel_regularizer=self.regularizer
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model layers.

        Args:
            input_shape: Shape of input tensor.
        """
        # Build all sub-layers explicitly (important for serialization)
        super().build(input_shape)

        # Build convolutional layers
        self.conv1.build(input_shape)
        conv1_out = self.conv1.compute_output_shape(input_shape)
        self.bn1.build(conv1_out)
        pool1_out = self.pool1.compute_output_shape(conv1_out)

        self.conv2.build(pool1_out)
        conv2_out = self.conv2.compute_output_shape(pool1_out)
        self.bn2.build(conv2_out)
        pool2_out = self.pool2.compute_output_shape(conv2_out)

        self.conv3.build(pool2_out)
        conv3_out = self.conv3.compute_output_shape(pool2_out)
        self.bn3.build(conv3_out)
        pool3_out = self.pool3.compute_output_shape(conv3_out)

        # Build dense layers
        flatten_out = self.flatten.compute_output_shape(pool3_out)
        self.fc1.build(flatten_out)
        fc1_out = self.fc1.compute_output_shape(flatten_out)
        self.bn_fc1.build(fc1_out)

        self.fc2.build(fc1_out)
        fc2_out = self.fc2.compute_output_shape(fc1_out)
        self.bn_fc2.build(fc2_out)

        self.fc_latent.build(fc2_out)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = False
    ) -> keras.KerasTensor:
        """
        Forward pass through the explainer network.

        Args:
            x: Input tensor of shape (batch_size, 28, 28, 1).
            training: Whether the model is in training mode.

        Returns:
            L2-normalized latent explanation vector.
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)

        # Latent vector generation
        x = self.flatten(x)
        x = self.dropout(x, training=training)

        x = self.fc1(x)
        x = self.bn_fc1(x, training=training)
        x = self.act_fc1(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.bn_fc2(x, training=training)
        x = self.act_fc2(x)

        # Get the raw latent vector
        e_latent_raw = self.fc_latent(x)

        # Normalize to unit length (L2 norm of 1)
        e_latent = tf.math.l2_normalize(e_latent_raw, axis=-1)

        return e_latent

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "explanation_dim": self.explanation_dim,
            "dropout_rate": self.dropout_rate,
            "l2_regularization": (
                self.regularizer.l2 if self.regularizer else 0.0
            ),
        })
        return config


@keras.saving.register_keras_serializable()
class MNISTReasoner(keras.Model):
    """
    Reasoner network for MNIST: P(Y|X,E).

    Performs context-aware digit classification using both the input image
    and the extracted latent explanation vector.

    Args:
        num_classes: Number of output classes (10 for MNIST).
        explanation_dim: Dimension of the latent explanation vector.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization strength for weights.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        - x: 4D tensor with shape `(batch_size, 28, 28, 1)`.
        - e: 2D tensor with shape `(batch_size, explanation_dim)`.

    Output shape:
        2D tensor with shape: `(batch_size, num_classes)`.
        Softmax probabilities over digit classes.

    Example:
        >>> reasoner = MNISTReasoner(num_classes=10, explanation_dim=128)
        >>> x = keras.ops.random.normal(shape=(32, 28, 28, 1))
        >>> e = keras.ops.random.normal(shape=(32, 128))
        >>> y_probs = reasoner(x, e, training=True)
        >>> assert y_probs.shape == (32, 10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        explanation_dim: int = 128,
        dropout_rate: float = 0.2,
        l2_regularization: float = 1e-4,
        **kwargs: Any
    ) -> None:
        """Initialize the Reasoner network."""
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = (
            keras.regularizers.L2(l2_regularization)
            if l2_regularization > 0
            else None
        )

        # Image feature extractor
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

        # Combined processing head
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
            num_classes, activation="softmax", kernel_regularizer=self.regularizer
        )

    def build(self, input_shape: Any) -> None:
        """
        Build the model layers.

        Note: For models with multiple inputs, Keras calls build with
        the full model's input shape, not individual inputs.
        """
        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        e: keras.KerasTensor,
        training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """
        Forward pass through the reasoner network.

        Args:
            x: Input image tensor of shape (batch_size, 28, 28, 1).
            e: Latent explanation tensor of shape (batch_size, explanation_dim).
            training: Whether the model is in training mode.

        Returns:
            Softmax probabilities over digit classes.
        """
        # Extract features from the image
        img_features = self.conv1(x)
        img_features = self.bn1(img_features, training=training)
        img_features = self.act1(img_features)
        img_features = self.pool1(img_features)

        img_features = self.conv2(img_features)
        img_features = self.bn2(img_features, training=training)
        img_features = self.act2(img_features)
        img_features = self.pool2(img_features)

        img_features = self.flatten(img_features)

        # Combine image features with latent explanation
        combined = keras.ops.concatenate([img_features, e], axis=-1)

        # Process combined features
        combined = self.fc_combine(combined)
        combined = self.bn_combine(combined, training=training)
        combined = self.act_combine(combined)
        combined = self.dropout(combined, training=training)

        combined = self.fc_hidden(combined)
        combined = self.bn_hidden(combined, training=training)
        combined = self.act_hidden(combined)
        combined = self.dropout(combined, training=training)

        y_probs = self.fc_output(combined)
        return y_probs

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "explanation_dim": self.explanation_dim,
            "dropout_rate": self.dropout_rate,
            "l2_regularization": (
                self.regularizer.l2 if self.regularizer else 0.0
            ),
        })
        return config


@keras.saving.register_keras_serializable()
class MNISTProducer(keras.Model):
    """
    Producer network for MNIST: P(X|Y,E).

    Generates/reconstructs digit images from label and style information,
    using an improved upsampling architecture to prevent checkerboard artifacts.

    Args:
        num_classes: Number of digit classes (10 for MNIST).
        explanation_dim: Dimension of the latent explanation vector.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization strength for weights.
        **kwargs: Additional arguments for Model base class.

    Input shape:
        - y: 2D tensor with shape `(batch_size, num_classes)`.
        - e: 2D tensor with shape `(batch_size, explanation_dim)`.

    Output shape:
        4D tensor with shape: `(batch_size, 28, 28, 1)`.
        Generated digit images with values in [0, 1].

    Example:
        >>> producer = MNISTProducer(num_classes=10, explanation_dim=128)
        >>> y = keras.ops.one_hot(keras.ops.array([3, 7, 1]), 10)
        >>> e = keras.ops.random.normal(shape=(3, 128))
        >>> x_generated = producer(y, e, training=True)
        >>> assert x_generated.shape == (3, 28, 28, 1)
    """

    def __init__(
        self,
        num_classes: int = 10,
        explanation_dim: int = 128,
        dropout_rate: float = 0.2,
        l2_regularization: float = 1e-4,
        **kwargs: Any
    ) -> None:
        """Initialize the Producer network."""
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = (
            keras.regularizers.L2(l2_regularization)
            if l2_regularization > 0
            else None
        )

        # Initial projection layers
        self.fc1 = keras.layers.Dense(512, kernel_regularizer=self.regularizer)
        self.bn_fc1 = keras.layers.BatchNormalization()
        self.act_fc1 = keras.layers.LeakyReLU(negative_slope=0.2)

        self.fc2 = keras.layers.Dense(
            7 * 7 * 128, kernel_regularizer=self.regularizer
        )
        self.bn_fc2 = keras.layers.BatchNormalization()
        self.act_fc2 = keras.layers.LeakyReLU(negative_slope=0.2)

        self.reshape = keras.layers.Reshape((7, 7, 128))
        self.dropout = keras.layers.Dropout(dropout_rate)

        # Upsampling blocks
        self.up1 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv1 = keras.layers.Conv2D(
            64, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn_up1 = keras.layers.BatchNormalization()
        self.act_up1 = keras.layers.LeakyReLU(negative_slope=0.2)

        self.up2 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = keras.layers.Conv2D(
            32, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn_up2 = keras.layers.BatchNormalization()
        self.act_up2 = keras.layers.LeakyReLU(negative_slope=0.2)

        # Refinement block
        self.conv3 = keras.layers.Conv2D(
            16, 3, padding="same", kernel_regularizer=self.regularizer
        )
        self.bn_up3 = keras.layers.BatchNormalization()
        self.act_up3 = keras.layers.LeakyReLU(negative_slope=0.2)

        # Output layer
        self.conv_out = keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid",
            kernel_regularizer=self.regularizer
        )

    def build(self, input_shape: Any) -> None:
        """
        Build the model layers.

        Note: For models with multiple inputs, Keras calls build with
        the full model's input shape, not individual inputs.
        """
        super().build(input_shape)

    def call(
        self,
        y: keras.KerasTensor,
        e: keras.KerasTensor,
        training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """
        Forward pass through the producer network.

        Args:
            y: One-hot encoded label tensor of shape (batch_size, num_classes).
            e: Latent explanation tensor of shape (batch_size, explanation_dim).
            training: Whether the model is in training mode.

        Returns:
            Generated image tensor of shape (batch_size, 28, 28, 1).
        """
        combined = keras.ops.concatenate([y, e], axis=-1)

        # Project and reshape
        x = self.fc1(combined)
        x = self.bn_fc1(x, training=training)
        x = self.act_fc1(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.bn_fc2(x, training=training)
        x = self.act_fc2(x)
        x = self.reshape(x)

        # Upsample to generate image
        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn_up1(x, training=training)
        x = self.act_up1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn_up2(x, training=training)
        x = self.act_up2(x)

        x = self.conv3(x)
        x = self.bn_up3(x, training=training)
        x = self.act_up3(x)

        x_generated = self.conv_out(x)
        return x_generated

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "explanation_dim": self.explanation_dim,
            "dropout_rate": self.dropout_rate,
            "l2_regularization": (
                self.regularizer.l2 if self.regularizer else 0.0
            ),
        })
        return config


# =============================================================================
# Training Functions
# =============================================================================


def create_mnist_ccnet(
    explanation_dim: int = 128,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.2,
    l2_regularization: float = 1e-4,
) -> CCNetOrchestrator:
    """
    Create a complete CCNet for MNIST.

    Args:
        explanation_dim: Dimension of latent explanation vectors.
        learning_rate: Learning rate for all models.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization strength.

    Returns:
        Configured CCNetOrchestrator ready for training.
    """
    # Create models
    explainer = MNISTExplainer(
        explanation_dim=explanation_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
    )

    reasoner = MNISTReasoner(
        num_classes=10,
        explanation_dim=explanation_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
    )

    producer = MNISTProducer(
        num_classes=10,
        explanation_dim=explanation_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
    )

    # Build models with dummy input
    dummy_image = keras.ops.zeros((1, 28, 28, 1))
    dummy_label = keras.ops.zeros((1, 10))
    dummy_latent = keras.ops.zeros((1, explanation_dim))

    _ = explainer(dummy_image)
    _ = reasoner(dummy_image, dummy_latent)
    _ = producer(dummy_label, dummy_latent)

    # Wrap and configure
    explainer_wrapped = wrap_keras_model(explainer)
    reasoner_wrapped = wrap_keras_model(reasoner)
    producer_wrapped = wrap_keras_model(producer)

    config = CCNetConfig(
        explanation_dim=explanation_dim,
        loss_type="l2",
        learning_rates={
            "explainer": learning_rate,
            "reasoner": learning_rate,
            "producer": learning_rate,
        },
        gradient_clip_norm=1.0,
        use_mixed_precision=False,
        sequential_data=False,
        verification_weight=1.0,
    )

    return CCNetOrchestrator(
        explainer=explainer_wrapped,
        reasoner=reasoner_wrapped,
        producer=producer_wrapped,
        config=config,
    )


def prepare_mnist_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare MNIST dataset for CCNet training.

    Returns:
        Tuple of (train_dataset, val_dataset) ready for training.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.batch(32)
        .shuffle(1000)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def train_mnist_ccnet_with_visualizations(epochs: int = 20) -> Tuple[
    CCNetOrchestrator, CCNetTrainer, CCNetVisualizer
]:
    """
    Main training script with comprehensive visualizations.

    Args:
        epochs: Number of training epochs.

    Returns:
        Tuple of (orchestrator, trainer, visualizer) after training.
    """
    # Initialize visualizer
    visualizer = CCNetVisualizer(results_dir="results")

    logger.info("Creating CCNet for MNIST...")
    orchestrator = create_mnist_ccnet(
        explanation_dim=128,
        learning_rate=1e-3,
        dropout_rate=0.2,
        l2_regularization=1e-4,
    )

    logger.info("Preparing data...")
    train_dataset, val_dataset = prepare_mnist_data()

    # Prepare test data for visualizations
    (x_test, y_test) = keras.datasets.mnist.load_data()[1]
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    logger.info("Setting up trainer...")
    trainer = CCNetTrainer(orchestrator)

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(patience=10, threshold=1e-4)

    # Visualization callback
    def visualization_callback(
        epoch: int, metrics: Dict[str, float], orch: CCNetOrchestrator
    ) -> None:
        """Callback for periodic visualizations during training."""
        if epoch % 5 == 0:
            logger.info(f"--- Epoch {epoch} Visualizations ---")

            # Save intermediate visualizations
            if epoch > 0:
                visualizer.plot_training_history(trainer.history)
                visualizer.visualize_reconstruction_quality(
                    orch, x_test[:100], y_test[:100], num_samples=5
                )

                if epoch % 10 == 0:
                    visualizer.visualize_counterfactual_generation(
                        orch, x_test[:1000], y_test[:1000]
                    )
                    visualizer.visualize_style_transfer(
                        orch, x_test[:1000], y_test[:1000]
                    )

    logger.info("Starting training...")
    try:
        trainer.train(
            train_dataset=train_dataset,
            epochs=epochs,
            validation_dataset=val_dataset,
            callbacks=[early_stopping, visualization_callback],
        )
    except StopIteration:
        logger.info("Training stopped early due to convergence.")

    logger.info("=" * 60)
    logger.info("Training complete! Generating final visualizations...")
    logger.info("=" * 60)

    # Generate comprehensive final visualizations
    visualizer.plot_training_history(trainer.history)
    visualizer.plot_convergence_analysis(trainer.history)
    visualizer.visualize_reconstruction_quality(
        orchestrator, x_test[:500], y_test[:500], num_samples=10
    )
    visualizer.visualize_counterfactual_generation(
        orchestrator, x_test[:1000], y_test[:1000]
    )
    visualizer.visualize_style_transfer(
        orchestrator, x_test[:1000], y_test[:1000]
    )
    visualizer.visualize_latent_space(
        orchestrator, x_test[:2000], y_test[:2000], num_samples=1000
    )
    visualizer.create_summary_report(
        orchestrator, trainer, x_test[:1000], y_test[:1000]
    )

    # Save models
    logger.info("Saving models...")
    save_path = visualizer.experiment_dir / "models"
    save_path.mkdir(exist_ok=True)
    orchestrator.save_models(str(save_path / "mnist_ccnet"))
    logger.info(f"Models saved to {save_path}")

    logger.info(f"All visualizations saved to: {visualizer.experiment_dir}")
    logger.info("CCNet training and visualization complete!")

    return orchestrator, trainer, visualizer


# =============================================================================
# Main Execution
# =============================================================================


if __name__ == "__main__":
    # Run the complete training and visualization pipeline
    orchestrator, trainer, visualizer = train_mnist_ccnet_with_visualizations(
        epochs=20
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Results directory: {visualizer.experiment_dir}")
    logger.info("=" * 60)