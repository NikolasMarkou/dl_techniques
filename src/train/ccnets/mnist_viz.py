"""
Enhanced MNIST CCNet implementation with comprehensive visualizations.
Demonstrates counterfactual generation, style transfer, and causal disentanglement.
"""

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, Dict, List
from pathlib import Path
import seaborn as sns
from datetime import datetime

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.ccnets import (
    CCNetOrchestrator,
    CCNetConfig,
    CCNetTrainer,
    EarlyStoppingCallback,
    wrap_keras_model
)
from dl_techniques.utils.logger import logger

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# Visualization Utilities
# ---------------------------------------------------------------------

class CCNetVisualizer:
    """
    Comprehensive visualization utilities for CCNet analysis.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize visualizer with results directory.

        Args:
            results_dir: Directory to save visualization results.
        """
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"ccnet_mnist_{self.timestamp}"

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "losses").mkdir(exist_ok=True)
        (self.experiment_dir / "counterfactuals").mkdir(exist_ok=True)
        (self.experiment_dir / "style_transfer").mkdir(exist_ok=True)
        (self.experiment_dir / "reconstructions").mkdir(exist_ok=True)
        (self.experiment_dir / "latent_space").mkdir(exist_ok=True)
        (self.experiment_dir / "generation_quality").mkdir(exist_ok=True)

        logger.info(f"Visualization results will be saved to: {self.experiment_dir}")

    def plot_training_history(self, history: Dict[str, List[float]]):
        """
        Plot comprehensive training history with all losses and errors.

        Args:
            history: Training history dictionary.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Plot fundamental losses
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['generation_loss'], label='Generation Loss', linewidth=2)
        ax1.set_title('Generation Loss\n||X_generated - X_input||', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['reconstruction_loss'], label='Reconstruction Loss', linewidth=2, color='orange')
        ax2.set_title('Reconstruction Loss\n||X_reconstructed - X_input||', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(history['inference_loss'], label='Inference Loss', linewidth=2, color='green')
        ax3.set_title('Inference Loss\n||X_reconstructed - X_generated||', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot model errors
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(history['explainer_error'], label='Explainer Error', linewidth=2, color='red')
        ax4.set_title('Explainer Error\nInf + Gen - Rec', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Error')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(history['reasoner_error'], label='Reasoner Error', linewidth=2, color='purple')
        ax5.set_title('Reasoner Error\nRec + Inf - Gen', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Error')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(history['producer_error'], label='Producer Error', linewidth=2, color='brown')
        ax6.set_title('Producer Error\nGen + Rec - Inf', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Error')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('CCNet Training History - Complete Loss Landscape', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "losses" / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training history saved to {self.experiment_dir / 'losses' / 'training_history.png'}")

    def plot_convergence_analysis(self, history: Dict[str, List[float]]):
        """
        Plot convergence analysis showing how losses converge to equilibrium.

        Args:
            history: Training history dictionary.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Combined losses plot
        ax = axes[0]
        ax.plot(history['generation_loss'], label='Generation', linewidth=2, alpha=0.8)
        ax.plot(history['reconstruction_loss'], label='Reconstruction', linewidth=2, alpha=0.8)
        ax.plot(history['inference_loss'], label='Inference', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Convergence Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Total system error
        ax = axes[1]
        total_error = [
            g + r + i
            for g, r, i in zip(
                history['generation_loss'],
                history['reconstruction_loss'],
                history['inference_loss']
            )
        ]
        ax.plot(total_error, linewidth=3, color='darkblue')
        ax.fill_between(range(len(total_error)), total_error, alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total System Error', fontsize=12)
        ax.set_title('CCNet System Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add convergence threshold line
        if len(total_error) > 0:
            convergence_threshold = min(total_error) * 1.1
            ax.axhline(y=convergence_threshold, color='red', linestyle='--',
                       alpha=0.5, label=f'Convergence Threshold')
            ax.legend()

        plt.suptitle('CCNet Convergence Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "losses" / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Convergence analysis saved")

    def visualize_reconstruction_quality(
            self,
            orchestrator: CCNetOrchestrator,
            x_test: np.ndarray,
            y_test: np.ndarray,
            num_samples: int = 10
    ):
        """
        Visualize reconstruction quality comparing original, reconstructed, and generated images.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images.
            y_test: Test labels.
            num_samples: Number of samples to visualize.
        """
        fig = plt.figure(figsize=(15, 3 * num_samples))
        gs = gridspec.GridSpec(num_samples, 5, figure=fig, wspace=0.3, hspace=0.4)

        for i in range(num_samples):
            x_input = x_test[i:i + 1]
            y_truth = keras.utils.to_categorical([y_test[i]], 10)

            # Get all outputs
            tensors = orchestrator.forward_pass(x_input, y_truth, training=False)

            # Extract latent for visualization
            e_latent = tensors['e_latent']

            # Original
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(x_input[0, :, :, 0], cmap='gray')
            ax.set_title(f'Original\nLabel: {y_test[i]}', fontsize=10)
            ax.axis('off')

            # Reconstructed (from inferred label)
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(tensors['x_reconstructed'][0, :, :, 0], cmap='gray')
            y_pred = np.argmax(tensors['y_inferred'][0])
            ax.set_title(f'Reconstructed\nInferred: {y_pred}', fontsize=10,
                         color='green' if y_pred == y_test[i] else 'red')
            ax.axis('off')

            # Generated (from true label)
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(tensors['x_generated'][0, :, :, 0], cmap='gray')
            ax.set_title(f'Generated\nTrue: {y_test[i]}', fontsize=10)
            ax.axis('off')

            # Difference maps
            ax = fig.add_subplot(gs[i, 3])
            diff_recon = np.abs(tensors['x_reconstructed'][0, :, :, 0] - x_input[0, :, :, 0])
            im = ax.imshow(diff_recon, cmap='hot')
            ax.set_title(f'Recon Error\nMAE: {np.mean(diff_recon):.3f}', fontsize=10)
            ax.axis('off')

            ax = fig.add_subplot(gs[i, 4])
            diff_gen = np.abs(tensors['x_generated'][0, :, :, 0] - x_input[0, :, :, 0])
            im = ax.imshow(diff_gen, cmap='hot')
            ax.set_title(f'Gen Error\nMAE: {np.mean(diff_gen):.3f}', fontsize=10)
            ax.axis('off')

        plt.suptitle('Reconstruction Quality Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.savefig(self.experiment_dir / "reconstructions" / "quality_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Reconstruction quality visualization saved")

    def visualize_counterfactual_generation(
            self,
            orchestrator: CCNetOrchestrator,
            x_test: np.ndarray,
            y_test: np.ndarray
    ):
        """
        Visualize counterfactual generation capabilities.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images.
            y_test: Test labels.
        """
        # Create a grid showing counterfactual transformations
        source_digits = [0, 1, 3, 5, 7]  # Source digits
        target_digits = [0, 2, 4, 6, 8, 9]  # Target digits

        fig = plt.figure(figsize=(len(target_digits) * 2 + 2, len(source_digits) * 2 + 1))
        gs = gridspec.GridSpec(len(source_digits) + 1, len(target_digits) + 1,
                               figure=fig, wspace=0.1, hspace=0.1)

        # Headers
        for j, target in enumerate(target_digits):
            ax = fig.add_subplot(gs[0, j + 1])
            ax.text(0.5, 0.5, f'→ {target}', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.axis('off')

        for i, source in enumerate(source_digits):
            # Row header (original)
            idx = np.where(y_test == source)[0][0]
            x_source = x_test[idx:idx + 1]

            ax = fig.add_subplot(gs[i + 1, 0])
            ax.imshow(x_source[0, :, :, 0], cmap='gray')
            ax.set_title(f'{source} →', fontsize=12, fontweight='bold', rotation=0)
            ax.axis('off')

            # Generate counterfactuals
            for j, target in enumerate(target_digits):
                y_target = keras.ops.zeros((1, 10))
                y_target = keras.ops.scatter_update(y_target, [[0, target]], [1.0])

                x_counterfactual = orchestrator.counterfactual_generation(x_source, y_target)

                ax = fig.add_subplot(gs[i + 1, j + 1])
                ax.imshow(x_counterfactual[0, :, :, 0], cmap='gray')

                # Highlight diagonal (same digit)
                if source == target:
                    ax.patch.set_edgecolor('green')
                    ax.patch.set_linewidth(3)

                ax.axis('off')

        plt.suptitle('Counterfactual Generation Matrix\n"What if this digit were drawn as..."',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(self.experiment_dir / "counterfactuals" / "counterfactual_matrix.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Counterfactual matrix saved")

        # Create detailed counterfactual sequence
        self._plot_counterfactual_sequence(orchestrator, x_test, y_test)

    def _plot_counterfactual_sequence(
            self,
            orchestrator: CCNetOrchestrator,
            x_test: np.ndarray,
            y_test: np.ndarray
    ):
        """
        Plot detailed counterfactual transformation sequence.
        """
        # Transform a '3' through all digits while preserving style
        source_digit = 3
        idx = np.where(y_test == source_digit)[0][2]  # Get third instance for variety
        x_source = x_test[idx:idx + 1]

        fig = plt.figure(figsize=(20, 4))
        gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.3)

        # Original in top-left
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(x_source[0, :, :, 0], cmap='gray')
        ax.set_title('Original\n(3)', fontsize=11, fontweight='bold', color='blue')
        ax.axis('off')

        # Generate all counterfactuals
        for target in range(10):
            row = 0 if target < 5 else 1
            col = (target % 5) * 2 + 1

            y_target = keras.ops.zeros((1, 10))
            y_target = keras.ops.scatter_update(y_target, [[0, target]], [1.0])

            x_counter = orchestrator.counterfactual_generation(x_source, y_target)

            ax = fig.add_subplot(gs[row, col:col + 2])
            ax.imshow(x_counter[0, :, :, 0], cmap='gray')
            ax.set_title(f'As "{target}"', fontsize=11,
                         color='green' if target == source_digit else 'black')
            ax.axis('off')

        plt.suptitle('Counterfactual Sequence: Preserving Style Across All Digits',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.savefig(self.experiment_dir / "counterfactuals" / "counterfactual_sequence.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Counterfactual sequence saved")

    def visualize_style_transfer(
            self,
            orchestrator: CCNetOrchestrator,
            x_test: np.ndarray,
            y_test: np.ndarray
    ):
        """
        Visualize style transfer capabilities.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images.
            y_test: Test labels.
        """
        # Select diverse style and content sources
        content_digits = [1, 2, 7, 8]
        style_digits = [3, 4, 5, 9]

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(len(content_digits), len(style_digits) + 2,
                               figure=fig, wspace=0.2, hspace=0.3)

        for i, content_digit in enumerate(content_digits):
            # Content source
            idx_content = np.where(y_test == content_digit)[0][0]
            x_content = x_test[idx_content:idx_content + 1]

            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(x_content[0, :, :, 0], cmap='gray')
            ax.set_title(f'Content\n({content_digit})', fontsize=11, fontweight='bold')
            ax.axis('off')

            # Arrow
            ax = fig.add_subplot(gs[i, 1])
            ax.text(0.5, 0.5, '→', ha='center', va='center', fontsize=20)
            ax.axis('off')

            for j, style_digit in enumerate(style_digits):
                # Style source
                idx_style = np.where(y_test == style_digit)[0][1]  # Use second instance
                x_style = x_test[idx_style:idx_style + 1]

                # Perform style transfer
                x_transferred = orchestrator.style_transfer(x_content, x_style)

                ax = fig.add_subplot(gs[i, j + 2])
                ax.imshow(x_transferred[0, :, :, 0], cmap='gray')

                if i == 0:
                    # Show style source above
                    ax_style = ax.twiny()
                    ax_style.set_xlim(ax.get_xlim())
                    ax_style.set_xticks([0.5])
                    ax_style.set_xticklabels([f'Style: {style_digit}'], fontsize=10)

                ax.axis('off')

        plt.suptitle('Style Transfer Matrix: Content × Style',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(self.experiment_dir / "style_transfer" / "style_transfer_matrix.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Style transfer matrix saved")

    def visualize_latent_space(
            self,
            orchestrator: CCNetOrchestrator,
            x_test: np.ndarray,
            y_test: np.ndarray,
            num_samples: int = 1000
    ):
        """
        Visualize the learned latent space using t-SNE.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images.
            y_test: Test labels.
            num_samples: Number of samples to visualize.
        """
        from sklearn.manifold import TSNE

        # Extract latent representations
        num_samples = min(num_samples, len(x_test))
        indices = np.random.choice(len(x_test), num_samples, replace=False)

        latent_vectors = []
        labels = []

        for idx in indices:
            x_sample = x_test[idx:idx + 1]
            _, e_latent = orchestrator.disentangle_causes(x_sample)
            latent_vectors.append(keras.ops.convert_to_numpy(e_latent[0]))
            labels.append(y_test[idx])

        latent_vectors = np.array(latent_vectors)
        labels = np.array(labels)

        # Apply t-SNE
        logger.info("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent_vectors)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Colored by digit class
        ax = axes[0]
        scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                             c=labels, cmap='tab10', s=20, alpha=0.6)
        ax.set_title('Latent Space Colored by Digit Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=ax, label='Digit')

        # Density plot
        ax = axes[1]
        from scipy.stats import gaussian_kde

        xy = latent_2d.T
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                             c=z, cmap='viridis', s=20, alpha=0.6)
        ax.set_title('Latent Space Density', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=ax, label='Density')

        plt.suptitle('CCNet Latent Space Visualization (E vectors)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "latent_space" / "tsne_visualization.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Latent space visualization saved")

        # Plot latent space statistics
        self._plot_latent_statistics(latent_vectors, labels)

    def _plot_latent_statistics(self, latent_vectors: np.ndarray, labels: np.ndarray):
        """
        Plot statistics of the latent space.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Latent dimension activation patterns
        ax = axes[0, 0]
        mean_activations = np.mean(np.abs(latent_vectors), axis=0)
        ax.bar(range(len(mean_activations)), mean_activations, color='steelblue')
        ax.set_title('Mean Activation per Latent Dimension', fontsize=12, fontweight='bold')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean |Activation|')
        ax.grid(True, alpha=0.3)

        # Variance per dimension
        ax = axes[0, 1]
        var_activations = np.var(latent_vectors, axis=0)
        ax.bar(range(len(var_activations)), var_activations, color='coral')
        ax.set_title('Variance per Latent Dimension', fontsize=12, fontweight='bold')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)

        # Per-class latent norms
        ax = axes[1, 0]
        latent_norms = np.linalg.norm(latent_vectors, axis=1)
        for digit in range(10):
            digit_norms = latent_norms[labels == digit]
            ax.violinplot([digit_norms], positions=[digit], widths=0.7,
                          showmeans=True, showmedians=True)
        ax.set_title('Latent Vector Norms by Digit Class', fontsize=12, fontweight='bold')
        ax.set_xlabel('Digit')
        ax.set_ylabel('||E|| (L2 Norm)')
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis='y')

        # Correlation matrix of top dimensions
        ax = axes[1, 1]
        top_k = 10
        top_dims = np.argsort(mean_activations)[-top_k:]
        corr_matrix = np.corrcoef(latent_vectors[:, top_dims].T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'Correlation Matrix (Top {top_k} Dimensions)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Dimension Index')
        plt.colorbar(im, ax=ax, label='Correlation')

        plt.suptitle('Latent Space Statistics Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "latent_space" / "latent_statistics.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Latent statistics saved")

    def visualize_generation_interpolation(
            self,
            orchestrator: CCNetOrchestrator,
            x_test: np.ndarray,
            y_test: np.ndarray
    ):
        """
        Visualize smooth interpolation in latent space.

        Args:
            orchestrator: Trained CCNet orchestrator.
            x_test: Test images.
            y_test: Test labels.
        """
        # Select two different styles of the same digit
        digit = 7
        indices = np.where(y_test == digit)[0][:2]

        x1 = x_test[indices[0]:indices[0] + 1]
        x2 = x_test[indices[1]:indices[1] + 1]

        # Extract latent representations
        _, e1 = orchestrator.disentangle_causes(x1)
        _, e2 = orchestrator.disentangle_causes(x2)

        # Create interpolation
        num_steps = 10
        interpolations = []

        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(3, num_steps + 2, figure=fig, wspace=0.1, hspace=0.3)

        # Show endpoints
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(x1[0, :, :, 0], cmap='gray')
        ax.set_title('Start', fontsize=11, fontweight='bold')
        ax.axis('off')

        ax = fig.add_subplot(gs[1, num_steps + 1])
        ax.imshow(x2[0, :, :, 0], cmap='gray')
        ax.set_title('End', fontsize=11, fontweight='bold')
        ax.axis('off')

        # Generate interpolations for multiple digits
        for digit_row, target_digit in enumerate([digit, 3, 8]):
            y_target = keras.ops.zeros((1, 10))
            y_target = keras.ops.scatter_update(y_target, [[0, target_digit]], [1.0])

            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                e_interp = (1 - alpha) * e1 + alpha * e2

                # Generate with interpolated latent
                x_interp = orchestrator.producer(y_target, e_interp, training=False)

                ax = fig.add_subplot(gs[digit_row, i + 1])
                ax.imshow(x_interp[0, :, :, 0], cmap='gray')
                if digit_row == 0 and i == 0:
                    ax.set_title(f'{alpha:.1f}', fontsize=9)
                if i == 0:
                    ax.set_ylabel(f'Digit: {target_digit}', fontsize=10)
                ax.axis('off')

        plt.suptitle('Latent Space Interpolation: Smooth Style Transitions',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(self.experiment_dir / "generation_quality" / "interpolation.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Interpolation visualization saved")

    def create_summary_report(
            self,
            orchestrator: CCNetOrchestrator,
            trainer: CCNetTrainer,
            x_test: np.ndarray,
            y_test: np.ndarray
    ):
        """
        Create a comprehensive summary report.

        Args:
            orchestrator: Trained CCNet orchestrator.
            trainer: CCNet trainer with history.
            x_test: Test images.
            y_test: Test labels.
        """
        # Create summary figure
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Training curves
        ax = fig.add_subplot(gs[0, :2])
        ax.plot(trainer.history['generation_loss'], label='Generation', linewidth=2)
        ax.plot(trainer.history['reconstruction_loss'], label='Reconstruction', linewidth=2)
        ax.plot(trainer.history['inference_loss'], label='Inference', linewidth=2)
        ax.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final metrics
        ax = fig.add_subplot(gs[0, 2:])
        final_metrics = {
            'Generation': trainer.history['generation_loss'][-1] if trainer.history['generation_loss'] else 0,
            'Reconstruction': trainer.history['reconstruction_loss'][-1] if trainer.history[
                'reconstruction_loss'] else 0,
            'Inference': trainer.history['inference_loss'][-1] if trainer.history['inference_loss'] else 0
        }
        bars = ax.bar(final_metrics.keys(), final_metrics.values(), color=['blue', 'orange', 'green'])
        ax.set_title('Final Loss Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss')
        for bar, value in zip(bars, final_metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3, axis='y')

        # Example reconstructions
        ax = fig.add_subplot(gs[1:3, :])
        ax.axis('off')
        ax.text(0.5, 0.98, 'Reconstruction Examples', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)

        # Create sub-grid for examples
        sub_gs = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs[1:3, :],
                                                  wspace=0.1, hspace=0.2)

        for i in range(4):
            idx = np.random.randint(0, len(x_test))
            x_sample = x_test[idx:idx + 1]
            y_truth = keras.utils.to_categorical([y_test[idx]], 10)

            tensors = orchestrator.forward_pass(x_sample, y_truth, training=False)

            # Show original
            ax_sub = fig.add_subplot(sub_gs[i, 0])
            ax_sub.imshow(x_sample[0, :, :, 0], cmap='gray')
            if i == 0:
                ax_sub.set_title('Original', fontsize=9)
            ax_sub.axis('off')

            # Show reconstructed
            ax_sub = fig.add_subplot(sub_gs[i, 1])
            ax_sub.imshow(tensors['x_reconstructed'][0, :, :, 0], cmap='gray')
            if i == 0:
                ax_sub.set_title('Reconstructed', fontsize=9)
            ax_sub.axis('off')

            # Show generated
            ax_sub = fig.add_subplot(sub_gs[i, 2])
            ax_sub.imshow(tensors['x_generated'][0, :, :, 0], cmap='gray')
            if i == 0:
                ax_sub.set_title('Generated', fontsize=9)
            ax_sub.axis('off')

            # Show error maps
            ax_sub = fig.add_subplot(sub_gs[i, 3])
            diff = np.abs(tensors['x_reconstructed'][0, :, :, 0] - x_sample[0, :, :, 0])
            ax_sub.imshow(diff, cmap='hot')
            if i == 0:
                ax_sub.set_title('Recon Error', fontsize=9)
            ax_sub.axis('off')

            ax_sub = fig.add_subplot(sub_gs[i, 4])
            diff = np.abs(tensors['x_generated'][0, :, :, 0] - x_sample[0, :, :, 0])
            ax_sub.imshow(diff, cmap='hot')
            if i == 0:
                ax_sub.set_title('Gen Error', fontsize=9)
            ax_sub.axis('off')

        # Counterfactual examples
        ax = fig.add_subplot(gs[3:5, :])
        ax.axis('off')
        ax.text(0.5, 0.98, 'Counterfactual Generation Examples', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)

        sub_gs2 = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs[3:5, :],
                                                   wspace=0.1, hspace=0.2)

        # Show counterfactual transformation
        idx_3 = np.where(y_test == 3)[0][0]
        x_3 = x_test[idx_3:idx_3 + 1]

        for i in range(10):
            row = i // 5
            col = i % 5 * 2

            y_target = keras.ops.zeros((1, 10))
            y_target = keras.ops.scatter_update(y_target, [[0, i]], [1.0])
            x_counter = orchestrator.counterfactual_generation(x_3, y_target)

            ax_sub = fig.add_subplot(sub_gs2[row, col:col + 2])
            ax_sub.imshow(x_counter[0, :, :, 0], cmap='gray')
            ax_sub.set_title(f'{i}', fontsize=9)
            ax_sub.axis('off')

        # Consistency check
        ax = fig.add_subplot(gs[5, :2])
        consistency_results = []
        for _ in range(100):
            idx = np.random.randint(0, min(1000, len(x_test)))
            is_consistent = orchestrator.verify_consistency(x_test[idx:idx + 1], threshold=0.05)
            consistency_results.append(is_consistent)

        consistency_rate = sum(consistency_results) / len(consistency_results)
        ax.pie([consistency_rate, 1 - consistency_rate],
               labels=['Consistent', 'Inconsistent'],
               colors=['green', 'red'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f'Consistency Check\n({len(consistency_results)} samples)',
                     fontsize=12, fontweight='bold')

        # Model info
        ax = fig.add_subplot(gs[5, 2:])
        ax.axis('off')
        info_text = f"""
        Model Configuration:
        • Explanation Dimension: {orchestrator.config.explanation_dim}
        • Loss Type: {orchestrator.config.loss_type}
        • Learning Rates: {orchestrator.config.learning_rates}
        • Gradient Clipping: {orchestrator.config.gradient_clip_norm}
        • Verification Weight: {orchestrator.config.verification_weight}

        Training Summary:
        • Total Epochs: {len(trainer.history['generation_loss'])}
        • Final Gen Loss: {final_metrics['Generation']:.4f}
        • Final Rec Loss: {final_metrics['Reconstruction']:.4f}
        • Final Inf Loss: {final_metrics['Inference']:.4f}
        • Consistency Rate: {consistency_rate:.1%}
        """
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')

        plt.suptitle('CCNet Training & Evaluation Summary Report',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(self.experiment_dir / "summary_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Summary report saved to {self.experiment_dir / 'summary_report.png'}")


# ---------------------------------------------------------------------
# Model Definitions (Same as before, included for completeness)
# ---------------------------------------------------------------------

class MNISTExplainer(keras.Model):
    """
    Explainer network for MNIST: P(E|X)
    Extracts latent style/context from digit images.
    """

    def __init__(
            self,
            explanation_dim: int = 128,
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-4,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None

        # Layers will be built in build() method
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.pool = None
        self.flatten = None
        self.dropout = None
        self.fc1 = None
        self.fc2 = None
        self.fc_latent = None
        self.batch_norm1 = None
        self.batch_norm2 = None

    def build(self, input_shape: Tuple[int, ...]):
        super().build(input_shape)

        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=self.regularizer)
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=self.regularizer)
        self.conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=self.regularizer)
        self.pool = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.fc1 = keras.layers.Dense(512, activation='relu', kernel_regularizer=self.regularizer)
        self.fc2 = keras.layers.Dense(256, activation='relu', kernel_regularizer=self.regularizer)
        self.fc_latent = keras.layers.Dense(self.explanation_dim, kernel_regularizer=self.regularizer)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.batch_norm1(x, training=training)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.batch_norm2(x, training=training)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        e_latent = self.fc_latent(x)
        return e_latent


class MNISTReasoner(keras.Model):
    """
    Reasoner network for MNIST: P(Y|X,E)
    Performs context-aware digit classification.
    """

    def __init__(
            self,
            num_classes: int = 10,
            explanation_dim: int = 128,
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-4,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None

        self.conv1 = None
        self.conv2 = None
        self.pool = None
        self.flatten = None
        self.dropout = None
        self.fc_combine = None
        self.fc_hidden = None
        self.fc_output = None
        self.batch_norm = None

    def build(self, input_shape):
        super().build(input_shape)

        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=self.regularizer)
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=self.regularizer)
        self.pool = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.batch_norm = keras.layers.BatchNormalization()
        self.fc_combine = keras.layers.Dense(512, activation='relu', kernel_regularizer=self.regularizer)
        self.fc_hidden = keras.layers.Dense(256, activation='relu', kernel_regularizer=self.regularizer)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.fc_output = keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=self.regularizer)

    def call(self, x: tf.Tensor, e: tf.Tensor, training: bool = False) -> tf.Tensor:
        img_features = self.conv1(x)
        img_features = self.pool(img_features)
        img_features = self.batch_norm(img_features, training=training)
        img_features = self.conv2(img_features)
        img_features = self.pool(img_features)
        img_features = self.flatten(img_features)
        combined = keras.ops.concatenate([img_features, e], axis=-1)
        combined = self.fc_combine(combined)
        combined = self.dropout(combined, training=training)
        combined = self.fc_hidden(combined)
        combined = self.dropout(combined, training=training)
        y_probs = self.fc_output(combined)
        return y_probs


class MNISTProducer(keras.Model):
    """
    Producer network for MNIST: P(X|Y,E)
    Generates/reconstructs digit images from label and style.
    """

    def __init__(
            self,
            num_classes: int = 10,
            explanation_dim: int = 128,
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-4,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None

        self.fc_input = None
        self.fc_hidden1 = None
        self.fc_hidden2 = None
        self.fc_reshape = None
        self.dropout = None
        self.reshape = None
        self.conv_transpose1 = None
        self.conv_transpose2 = None
        self.conv_transpose3 = None
        self.conv_output = None
        self.batch_norm1 = None
        self.batch_norm2 = None

    def build(self, input_shape):
        super().build(input_shape)

        self.fc_input = keras.layers.Dense(256, activation='relu', kernel_regularizer=self.regularizer)
        self.fc_hidden1 = keras.layers.Dense(512, activation='relu', kernel_regularizer=self.regularizer)
        self.fc_hidden2 = keras.layers.Dense(1024, activation='relu', kernel_regularizer=self.regularizer)
        self.fc_reshape = keras.layers.Dense(7 * 7 * 128, activation='relu', kernel_regularizer=self.regularizer)
        self.reshape = keras.layers.Reshape((7, 7, 128))
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.conv_transpose1 = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu',
                                                            kernel_regularizer=self.regularizer)
        self.conv_transpose2 = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu',
                                                            kernel_regularizer=self.regularizer)
        self.conv_transpose3 = keras.layers.Conv2DTranspose(16, 3, padding='same', activation='relu',
                                                            kernel_regularizer=self.regularizer)
        self.conv_output = keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid',
                                               kernel_regularizer=self.regularizer)

    def call(self, y: tf.Tensor, e: tf.Tensor, training: bool = False) -> tf.Tensor:
        combined = keras.ops.concatenate([y, e], axis=-1)
        x = self.fc_input(combined)
        x = self.dropout(x, training=training)
        x = self.fc_hidden1(x)
        x = self.dropout(x, training=training)
        x = self.fc_hidden2(x)
        x = self.fc_reshape(x)
        x = self.reshape(x)
        x = self.conv_transpose1(x)
        x = self.batch_norm1(x, training=training)
        x = self.conv_transpose2(x)
        x = self.batch_norm2(x, training=training)
        x = self.conv_transpose3(x)
        x_generated = self.conv_output(x)
        return x_generated


# ---------------------------------------------------------------------
# Training Script with Visualizations
# ---------------------------------------------------------------------

def create_mnist_ccnet(
        explanation_dim: int = 128,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.2,
        l2_regularization: float = 1e-4
) -> CCNetOrchestrator:
    """Create a complete CCNet for MNIST."""
    explainer = MNISTExplainer(explanation_dim=explanation_dim, dropout_rate=dropout_rate,
                               l2_regularization=l2_regularization)
    reasoner = MNISTReasoner(num_classes=10, explanation_dim=explanation_dim, dropout_rate=dropout_rate,
                             l2_regularization=l2_regularization)
    producer = MNISTProducer(num_classes=10, explanation_dim=explanation_dim, dropout_rate=dropout_rate,
                             l2_regularization=l2_regularization)

    # Build models with dummy input
    dummy_image = keras.ops.zeros((1, 28, 28, 1))
    dummy_label = keras.ops.zeros((1, 10))
    dummy_latent = keras.ops.zeros((1, explanation_dim))

    explainer(dummy_image)
    reasoner(dummy_image, dummy_latent)
    producer(dummy_label, dummy_latent)

    # Wrap and configure
    explainer_wrapped = wrap_keras_model(explainer)
    reasoner_wrapped = wrap_keras_model(reasoner)
    producer_wrapped = wrap_keras_model(producer)

    config = CCNetConfig(
        explanation_dim=explanation_dim,
        loss_type='l2',
        learning_rates={'explainer': learning_rate, 'reasoner': learning_rate, 'producer': learning_rate},
        gradient_clip_norm=1.0,
        use_mixed_precision=False,
        sequential_data=False,
        verification_weight=1.0
    )

    return CCNetOrchestrator(explainer=explainer_wrapped, reasoner=reasoner_wrapped, producer=producer_wrapped,
                             config=config)


def prepare_mnist_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Prepare MNIST dataset for CCNet training."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(32).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def train_mnist_ccnet_with_visualizations(epochs: int = 20):
    """Main training script with comprehensive visualizations."""

    # Initialize visualizer
    visualizer = CCNetVisualizer(results_dir="results")

    logger.info("Creating CCNet for MNIST...")
    orchestrator = create_mnist_ccnet(
        explanation_dim=128,
        learning_rate=1e-3,
        dropout_rate=0.2,
        l2_regularization=1e-4
    )

    logger.info("Preparing data...")
    train_dataset, val_dataset = prepare_mnist_data()

    # Prepare test data for visualizations
    (x_test, y_test) = keras.datasets.mnist.load_data()[1]
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    logger.info("Setting up trainer...")
    trainer = CCNetTrainer(orchestrator)

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(patience=10, threshold=1e-4)

    # Visualization callback
    def visualization_callback(epoch, metrics, orch):
        if epoch % 5 == 0:
            logger.info(f"\n--- Epoch {epoch} Visualizations ---")

            # Save intermediate visualizations
            if epoch > 0:
                visualizer.plot_training_history(trainer.history)
                visualizer.visualize_reconstruction_quality(orch, x_test[:100], y_test[:100], num_samples=5)

                if epoch % 10 == 0:
                    visualizer.visualize_counterfactual_generation(orch, x_test[:1000], y_test[:1000])
                    visualizer.visualize_style_transfer(orch, x_test[:1000], y_test[:1000])

    logger.info("Starting training...")
    try:
        trainer.train(
            train_dataset=train_dataset,
            epochs=epochs,
            validation_dataset=val_dataset,
            callbacks=[early_stopping, visualization_callback]
        )
    except StopIteration:
        logger.info("Training stopped early due to convergence.")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete! Generating final visualizations...")
    logger.info("=" * 60)

    # Generate comprehensive final visualizations
    visualizer.plot_training_history(trainer.history)
    visualizer.plot_convergence_analysis(trainer.history)
    visualizer.visualize_reconstruction_quality(orchestrator, x_test[:500], y_test[:500], num_samples=10)
    visualizer.visualize_counterfactual_generation(orchestrator, x_test[:1000], y_test[:1000])
    visualizer.visualize_style_transfer(orchestrator, x_test[:1000], y_test[:1000])
    visualizer.visualize_latent_space(orchestrator, x_test[:2000], y_test[:2000], num_samples=1000)
    visualizer.visualize_generation_interpolation(orchestrator, x_test[:1000], y_test[:1000])
    visualizer.create_summary_report(orchestrator, trainer, x_test[:1000], y_test[:1000])

    # Save models
    logger.info("\nSaving models...")
    save_path = visualizer.experiment_dir / "models"
    save_path.mkdir(exist_ok=True)
    orchestrator.save_models(str(save_path / "mnist_ccnet"))
    logger.info(f"Models saved to {save_path}")

    logger.info(f"\nAll visualizations saved to: {visualizer.experiment_dir}")
    logger.info("\nCCNet training and visualization complete!")

    return orchestrator, trainer, visualizer


if __name__ == "__main__":
    orchestrator, trainer, visualizer = train_mnist_ccnet_with_visualizations(epochs=20)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Results directory: {visualizer.experiment_dir}")
    logger.info("=" * 60)