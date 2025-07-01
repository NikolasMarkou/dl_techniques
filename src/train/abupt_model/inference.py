"""
Inference and Evaluation Pipeline for AB-UPT Model.

This module provides:
- Model inference on new data
- Evaluation metrics calculation
- Visualization of results
- Streamline generation and comparison
- Error analysis and reporting

Based on the PyTorch implementation with Keras-specific adaptations.
"""

import os
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

from dl_techniques.utils.logger import logger
from dl_techniques.models.anchored_branched_upt import AnchoredBranchedUPT
from .train import DataPreprocessor, NormalizationStats, CFDDataGenerator


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    surface_pressure_mae: float
    surface_pressure_rmse: float
    surface_wallshear_mae: float
    surface_wallshear_rmse: float
    volume_pressure_mae: float
    volume_pressure_rmse: float
    volume_velocity_mae: float
    volume_velocity_rmse: float
    volume_vorticity_mae: float
    volume_vorticity_rmse: float
    overall_mae: float
    overall_rmse: float


class CFDInference:
    """Inference engine for AB-UPT model."""

    def __init__(
            self,
            model_path: str,
            preprocessor: DataPreprocessor,
            device: str = "auto"
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to saved model
            preprocessor: Data preprocessor instance
            device: Device to run inference on
        """
        self.preprocessor = preprocessor

        # Load model
        logger.info(f"Loading model from {model_path}")
        try:
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    "AnchoredBranchedUPT": AnchoredBranchedUPT,
                }
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run inference on input data.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary of predictions
        """
        # Ensure inputs are tensors
        processed_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, 'shape'):  # Not a tensor
                processed_inputs[key] = ops.array(value, dtype="float32")
            else:
                processed_inputs[key] = value

        # Run prediction
        with keras.backend.eager_scope():
            predictions = self.model(processed_inputs, training=False)

        # Convert to numpy for easier handling
        numpy_predictions = {}
        for key, value in predictions.items():
            numpy_predictions[key] = np.array(value)

        return numpy_predictions

    def predict_batch(self, data_generator: CFDDataGenerator) -> Tuple[List[Dict], List[Dict]]:
        """Run inference on a batch of data.

        Args:
            data_generator: Data generator

        Returns:
            Tuple of (predictions_list, targets_list)
        """
        predictions_list = []
        targets_list = []

        logger.info(f"Running inference on {len(data_generator)} batches...")

        for batch_idx in range(len(data_generator)):
            inputs, targets = data_generator[batch_idx]

            # Run prediction
            predictions = self.predict(inputs)

            predictions_list.append(predictions)
            targets_list.append({k: np.array(v) for k, v in targets.items()})

            if batch_idx % 10 == 0:
                logger.info(f"  Processed {batch_idx}/{len(data_generator)} batches")

        return predictions_list, targets_list

    def denormalize_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Denormalize predictions back to physical units.

        Args:
            predictions: Normalized predictions

        Returns:
            Denormalized predictions
        """
        denormalized = {}

        for key, value in predictions.items():
            if "pressure" in key:
                if "surface" in key:
                    denormalized[key] = np.array(self.preprocessor.denormalize_surface_pressure(value))
                else:  # volume pressure
                    denormalized[key] = np.array(self.preprocessor.denormalize_volume_pressure(value))
            elif "wallshearstress" in key:
                denormalized[key] = np.array(self.preprocessor.denormalize_surface_wallshearstress(value))
            elif "velocity" in key:
                denormalized[key] = np.array(self.preprocessor.denormalize_volume_velocity(value))
            elif "vorticity" in key:
                denormalized[key] = np.array(self.preprocessor.denormalize_volume_vorticity(value))
            else:
                denormalized[key] = value  # Keep as-is if unknown

        return denormalized


class CFDEvaluator:
    """Evaluation utilities for CFD predictions."""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor

    def calculate_metrics(
            self,
            predictions_list: List[Dict],
            targets_list: List[Dict],
            denormalize: bool = True
    ) -> EvaluationMetrics:
        """Calculate evaluation metrics.

        Args:
            predictions_list: List of prediction dictionaries
            targets_list: List of target dictionaries
            denormalize: Whether to denormalize before computing metrics

        Returns:
            Evaluation metrics
        """
        # Collect all predictions and targets
        all_predictions = {}
        all_targets = {}

        # Initialize lists for each quantity
        for key in predictions_list[0].keys():
            all_predictions[key] = []
            all_targets[key] = []

        # Concatenate all batches
        for pred_batch, target_batch in zip(predictions_list, targets_list):
            for key in pred_batch.keys():
                if key in target_batch:
                    pred_data = pred_batch[key]
                    target_data = target_batch[key]

                    if denormalize:
                        # Create temporary inference object for denormalization
                        temp_inference = CFDInference.__new__(CFDInference)
                        temp_inference.preprocessor = self.preprocessor

                        pred_data = temp_inference.denormalize_predictions({key: pred_data})[key]
                        target_data = temp_inference.denormalize_predictions({key: target_data})[key]

                    all_predictions[key].append(pred_data)
                    all_targets[key].append(target_data)

        # Concatenate arrays
        for key in all_predictions.keys():
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
            all_targets[key] = np.concatenate(all_targets[key], axis=0)

        # Calculate metrics
        metrics = {}

        for key in all_predictions.keys():
            pred = all_predictions[key]
            target = all_targets[key]

            # Handle different dimensionalities
            if pred.ndim > 1:
                # Flatten for vector quantities
                pred = pred.reshape(-1)
                target = target.reshape(-1)

            # Calculate MAE and RMSE
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(np.mean((pred - target) ** 2))

            metrics[f"{key}_mae"] = mae
            metrics[f"{key}_rmse"] = rmse

        # Calculate overall metrics
        all_pred_values = np.concatenate([pred.flatten() for pred in all_predictions.values()])
        all_target_values = np.concatenate([target.flatten() for target in all_targets.values()])

        overall_mae = np.mean(np.abs(all_pred_values - all_target_values))
        overall_rmse = np.sqrt(np.mean((all_pred_values - all_target_values) ** 2))

        return EvaluationMetrics(
            surface_pressure_mae=metrics.get("surface_anchor_pressure_mae", 0.0),
            surface_pressure_rmse=metrics.get("surface_anchor_pressure_rmse", 0.0),
            surface_wallshear_mae=metrics.get("surface_anchor_wallshearstress_mae", 0.0),
            surface_wallshear_rmse=metrics.get("surface_anchor_wallshearstress_rmse", 0.0),
            volume_pressure_mae=metrics.get("volume_anchor_totalpcoeff_mae", 0.0),
            volume_pressure_rmse=metrics.get("volume_anchor_totalpcoeff_rmse", 0.0),
            volume_velocity_mae=metrics.get("volume_anchor_velocity_mae", 0.0),
            volume_velocity_rmse=metrics.get("volume_anchor_velocity_rmse", 0.0),
            volume_vorticity_mae=metrics.get("volume_anchor_vorticity_mae", 0.0),
            volume_vorticity_rmse=metrics.get("volume_anchor_vorticity_rmse", 0.0),
            overall_mae=overall_mae,
            overall_rmse=overall_rmse
        )

    def print_metrics(self, metrics: EvaluationMetrics):
        """Print evaluation metrics in a formatted way."""
        logger.info("📊 Evaluation Metrics:")
        logger.info("=" * 50)

        logger.info("Surface Quantities:")
        logger.info(
            f"  Pressure    - MAE: {metrics.surface_pressure_mae:.6f}, RMSE: {metrics.surface_pressure_rmse:.6f}")
        logger.info(
            f"  Wall Shear  - MAE: {metrics.surface_wallshear_mae:.6f}, RMSE: {metrics.surface_wallshear_rmse:.6f}")

        logger.info("\nVolume Quantities:")
        logger.info(f"  Pressure    - MAE: {metrics.volume_pressure_mae:.6f}, RMSE: {metrics.volume_pressure_rmse:.6f}")
        logger.info(f"  Velocity    - MAE: {metrics.volume_velocity_mae:.6f}, RMSE: {metrics.volume_velocity_rmse:.6f}")
        logger.info(
            f"  Vorticity   - MAE: {metrics.volume_vorticity_mae:.6f}, RMSE: {metrics.volume_vorticity_rmse:.6f}")

        logger.info("\nOverall:")
        logger.info(f"  Overall     - MAE: {metrics.overall_mae:.6f}, RMSE: {metrics.overall_rmse:.6f}")
        logger.info("=" * 50)


class CFDVisualizer:
    """Visualization utilities for CFD results."""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor

    def plot_surface_comparison(
            self,
            positions: np.ndarray,
            predictions: Dict[str, np.ndarray],
            targets: Dict[str, np.ndarray],
            quantity: str = "pressure",
            save_path: Optional[str] = None
    ):
        """Plot surface quantity comparison.

        Args:
            positions: Surface positions (N, 3)
            predictions: Predicted quantities
            targets: Target quantities
            quantity: Quantity to plot ("pressure" or "wallshearstress")
            save_path: Path to save the plot
        """
        # Get data
        if quantity == "pressure":
            pred_key = "surface_anchor_pressure"
            target_key = "surface_anchor_pressure"
            pred_data = predictions[pred_key].flatten()
            target_data = targets[target_key].flatten()
            title_prefix = "Surface Pressure"
            cmap = "RdBu_r"
        else:  # wallshearstress
            pred_key = "surface_anchor_wallshearstress"
            target_key = "surface_anchor_wallshearstress"
            # Use magnitude for visualization
            pred_data = np.linalg.norm(predictions[pred_key], axis=1)
            target_data = np.linalg.norm(targets[target_key], axis=1)
            title_prefix = "Wall Shear Stress Magnitude"
            cmap = "viridis"

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Common colormap limits
        vmin = min(pred_data.min(), target_data.min())
        vmax = max(pred_data.max(), target_data.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Target plot
        scatter1 = axes[0].scatter(
            positions[:, 0], positions[:, 1],
            c=target_data, cmap=cmap, norm=norm, s=2, alpha=0.7
        )
        axes[0].set_title(f"{title_prefix} - Ground Truth")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_aspect("equal")

        # Prediction plot
        scatter2 = axes[1].scatter(
            positions[:, 0], positions[:, 1],
            c=pred_data, cmap=cmap, norm=norm, s=2, alpha=0.7
        )
        axes[1].set_title(f"{title_prefix} - Prediction")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].set_aspect("equal")

        # Error plot
        error = np.abs(pred_data - target_data)
        scatter3 = axes[2].scatter(
            positions[:, 0], positions[:, 1],
            c=error, cmap="Reds", s=2, alpha=0.7
        )
        axes[2].set_title(f"{title_prefix} - Absolute Error")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")
        axes[2].set_aspect("equal")

        # Add colorbars
        plt.colorbar(scatter1, ax=axes[0])
        plt.colorbar(scatter2, ax=axes[1])
        plt.colorbar(scatter3, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Surface comparison plot saved to {save_path}")

        plt.show()

    def plot_volume_slice(
            self,
            positions: np.ndarray,
            predictions: Dict[str, np.ndarray],
            targets: Dict[str, np.ndarray],
            quantity: str = "velocity",
            slice_coord: float = 0.0,
            slice_axis: int = 2,
            save_path: Optional[str] = None
    ):
        """Plot volume quantity on a slice.

        Args:
            positions: Volume positions (N, 3)
            predictions: Predicted quantities
            targets: Target quantities
            quantity: Quantity to plot ("velocity", "pressure", or "vorticity")
            slice_coord: Coordinate value for the slice
            slice_axis: Axis for slicing (0=X, 1=Y, 2=Z)
            save_path: Path to save the plot
        """
        # Select points near the slice
        tolerance = 1.0  # Adjust based on your data
        slice_mask = np.abs(positions[:, slice_axis] - slice_coord) < tolerance

        if not np.any(slice_mask):
            logger.warning(f"No points found near slice coordinate {slice_coord} on axis {slice_axis}")
            return

        slice_positions = positions[slice_mask]

        # Get remaining coordinate indices
        coord_indices = [i for i in range(3) if i != slice_axis]
        x_coord = coord_indices[0]
        y_coord = coord_indices[1]

        # Get data
        if quantity == "velocity":
            pred_key = "volume_anchor_velocity"
            target_key = "volume_anchor_velocity"
            pred_data = np.linalg.norm(predictions[pred_key][slice_mask], axis=1)
            target_data = np.linalg.norm(targets[target_key][slice_mask], axis=1)
            title_prefix = "Velocity Magnitude"
            cmap = "plasma"
        elif quantity == "pressure":
            pred_key = "volume_anchor_totalpcoeff"
            target_key = "volume_anchor_totalpcoeff"
            pred_data = predictions[pred_key][slice_mask].flatten()
            target_data = targets[target_key][slice_mask].flatten()
            title_prefix = "Pressure"
            cmap = "RdBu_r"
        else:  # vorticity
            pred_key = "volume_anchor_vorticity"
            target_key = "volume_anchor_vorticity"
            pred_data = np.linalg.norm(predictions[pred_key][slice_mask], axis=1)
            target_data = np.linalg.norm(targets[target_key][slice_mask], axis=1)
            title_prefix = "Vorticity Magnitude"
            cmap = "hot"

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Common colormap limits
        vmin = min(pred_data.min(), target_data.min())
        vmax = max(pred_data.max(), target_data.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Target plot
        scatter1 = axes[0].scatter(
            slice_positions[:, x_coord], slice_positions[:, y_coord],
            c=target_data, cmap=cmap, norm=norm, s=4, alpha=0.7
        )
        axes[0].set_title(f"{title_prefix} - Ground Truth")
        axes[0].set_xlabel(f"Axis {x_coord}")
        axes[0].set_ylabel(f"Axis {y_coord}")

        # Prediction plot
        scatter2 = axes[1].scatter(
            slice_positions[:, x_coord], slice_positions[:, y_coord],
            c=pred_data, cmap=cmap, norm=norm, s=4, alpha=0.7
        )
        axes[1].set_title(f"{title_prefix} - Prediction")
        axes[1].set_xlabel(f"Axis {x_coord}")
        axes[1].set_ylabel(f"Axis {y_coord}")

        # Error plot
        error = np.abs(pred_data - target_data)
        scatter3 = axes[2].scatter(
            slice_positions[:, x_coord], slice_positions[:, y_coord],
            c=error, cmap="Reds", s=4, alpha=0.7
        )
        axes[2].set_title(f"{title_prefix} - Absolute Error")
        axes[2].set_xlabel(f"Axis {x_coord}")
        axes[2].set_ylabel(f"Axis {y_coord}")

        # Add colorbars
        plt.colorbar(scatter1, ax=axes[0])
        plt.colorbar(scatter2, ax=axes[1])
        plt.colorbar(scatter3, ax=axes[2])

        plt.suptitle(f"Volume Slice at {['X', 'Y', 'Z'][slice_axis]}={slice_coord:.1f}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Volume slice plot saved to {save_path}")

        plt.show()

    def plot_error_distribution(
            self,
            predictions_list: List[Dict],
            targets_list: List[Dict],
            save_path: Optional[str] = None
    ):
        """Plot error distributions for all quantities."""
        # Collect errors
        errors = {}

        for pred_batch, target_batch in zip(predictions_list, targets_list):
            for key in pred_batch.keys():
                if key in target_batch:
                    pred = pred_batch[key]
                    target = target_batch[key]

                    # Calculate relative error
                    error = np.abs(pred - target) / (np.abs(target) + 1e-8)

                    if key not in errors:
                        errors[key] = []
                    errors[key].append(error.flatten())

        # Concatenate errors
        for key in errors.keys():
            errors[key] = np.concatenate(errors[key])

        # Create plots
        n_quantities = len(errors)
        fig, axes = plt.subplots(2, (n_quantities + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (key, error_data) in enumerate(errors.items()):
            axes[i].hist(error_data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f"{key.replace('_', ' ').title()}")
            axes[i].set_xlabel("Relative Error")
            axes[i].set_ylabel("Frequency")
            axes[i].set_yscale("log")
            axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(errors), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Error Distributions")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Error distribution plot saved to {save_path}")

        plt.show()


def run_inference_pipeline(
        model_path: str,
        test_data: List[Dict[str, Any]],
        output_dir: str = "inference_results"
):
    """Run complete inference and evaluation pipeline.

    Args:
        model_path: Path to trained model
        test_data: Test dataset
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Setup preprocessing
    stats = NormalizationStats()
    preprocessor = DataPreprocessor(stats)

    # Create test generator
    test_generator = CFDDataGenerator(
        test_data,
        preprocessor,
        batch_size=1,
        num_geometry_points=800,
        num_surface_anchor_points=300,
        num_volume_anchor_points=400,
        num_geometry_supernodes=150,
        use_query_positions=False,  # Simplified for inference
        shuffle=False
    )

    # Initialize inference engine
    inference_engine = CFDInference(model_path, preprocessor)

    # Run inference
    logger.info("🔮 Starting inference...")
    predictions_list, targets_list = inference_engine.predict_batch(test_generator)

    # Initialize evaluator and visualizer
    evaluator = CFDEvaluator(preprocessor)
    visualizer = CFDVisualizer(preprocessor)

    # Calculate metrics
    logger.info("📊 Calculating metrics...")
    metrics = evaluator.calculate_metrics(predictions_list, targets_list, denormalize=True)
    evaluator.print_metrics(metrics)

    # Save metrics
    metrics_dict = {
        "surface_pressure_mae": metrics.surface_pressure_mae,
        "surface_pressure_rmse": metrics.surface_pressure_rmse,
        "surface_wallshear_mae": metrics.surface_wallshear_mae,
        "surface_wallshear_rmse": metrics.surface_wallshear_rmse,
        "volume_pressure_mae": metrics.volume_pressure_mae,
        "volume_pressure_rmse": metrics.volume_pressure_rmse,
        "volume_velocity_mae": metrics.volume_velocity_mae,
        "volume_velocity_rmse": metrics.volume_velocity_rmse,
        "volume_vorticity_mae": metrics.volume_vorticity_mae,
        "volume_vorticity_rmse": metrics.volume_vorticity_rmse,
        "overall_mae": metrics.overall_mae,
        "overall_rmse": metrics.overall_rmse,
    }

    with open(os.path.join(output_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Create visualizations
    logger.info("🎨 Generating visualizations...")

    # Error distribution
    visualizer.plot_error_distribution(
        predictions_list, targets_list,
        save_path=os.path.join(output_dir, "error_distributions.png")
    )

    # Sample comparison plots
    if len(predictions_list) > 0:
        sample_idx = 0
        sample_input, _ = test_generator[sample_idx]

        # Get positions (denormalized)
        surface_positions = np.array(sample_input["surface_anchor_position"][0])
        surface_positions = np.array(preprocessor.denormalize_positions(surface_positions))

        volume_positions = np.array(sample_input["volume_anchor_position"][0])
        volume_positions = np.array(preprocessor.denormalize_positions(volume_positions))

        # Denormalize predictions and targets
        pred_denorm = inference_engine.denormalize_predictions(predictions_list[sample_idx])
        target_denorm = inference_engine.denormalize_predictions(targets_list[sample_idx])

        # Surface plots
        visualizer.plot_surface_comparison(
            surface_positions, pred_denorm, target_denorm,
            quantity="pressure",
            save_path=os.path.join(output_dir, "surface_pressure_comparison.png")
        )

        # Volume plots
        visualizer.plot_volume_slice(
            volume_positions, pred_denorm, target_denorm,
            quantity="velocity", slice_coord=0.0, slice_axis=2,
            save_path=os.path.join(output_dir, "volume_velocity_slice.png")
        )

    logger.info(f"✅ Inference pipeline completed! Results saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    # Example usage
    from abupt_training import create_synthetic_data

    # Create test data
    test_data = create_synthetic_data(num_samples=10)

    # Run inference (assuming model exists)
    model_path = "abupt_checkpoints/best_model.keras"

    if os.path.exists(model_path):
        metrics = run_inference_pipeline(model_path, test_data)
        logger.info("🎉 Inference completed successfully!")
    else:
        logger.warning(f"Model not found at {model_path}. Please train the model first.")