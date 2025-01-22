"""
Enhanced Model Analyzer for CNN visualization and comparison.
Includes activation maps and probability distribution visualization.
"""

import keras
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict, Any, List

from .datasets import MNISTData
from .visualization_manager import VisualizationManager



class ModelAnalyzer:
    """Enhanced analyzer for model comparison with detailed visualization support."""

    def __init__(
            self,
            models: Dict[str, keras.Model],
            vis_manager: VisualizationManager
    ):
        """Initialize analyzer with models and visualization manager.

        Args:
            models: Dictionary of models to analyze
            vis_manager: Visualization manager instance
        """
        self.models = models
        self.vis_manager = vis_manager
        self._setup_activation_models()

    def _setup_activation_models(self) -> None:
        """Create models to extract intermediate activations."""
        self.activation_models = {}

        for model_name, model in self.models.items():
            # Get the last convolutional layer
            conv_layers = [layer for layer in model.layers
                           if isinstance(layer, keras.layers.Conv2D)]
            if conv_layers:
                last_conv = conv_layers[-1]
                self.activation_models[model_name] = keras.Model(
                    inputs=model.input,
                    outputs=[
                        last_conv.output,  # Last conv activation
                        model.output  # Final predictions
                    ]
                )

    def get_activation_and_predictions(
            self,
            image: np.ndarray,
            model_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get activation maps and predictions for a single image.

        Args:
            image: Input image
            model_name: Name of the model to use

        Returns:
            Tuple of (activation_maps, predictions)
        """
        model = self.activation_models[model_name]
        image_batch = np.expand_dims(image, 0)
        activations, predictions = model.predict(image_batch)

        # Average activation maps across channels
        mean_activation = np.mean(activations[0], axis=-1)
        return mean_activation, predictions[0]

    def create_comprehensive_visualization(
            self,
            data: MNISTData,
            sample_digits: Optional[List[int]] = None,
            max_samples_per_digit: int = 3
    ) -> None:
        """Create comprehensive visualization for all digits.

        Args:
            data: MNIST dataset splits
            sample_digits: Optional list of digits to analyze
            max_samples_per_digit: Maximum number of samples per digit
        """
        if sample_digits is None:
            sample_digits = list(range(10))

        n_models = len(self.models)
        n_samples = max_samples_per_digit
        n_digits = len(sample_digits)

        # Create a large figure
        fig = plt.figure(figsize=(20, 2 * n_digits))
        gs = plt.GridSpec(n_digits, n_models * (n_samples + 1))

        # For each digit
        for digit_idx, digit in enumerate(sample_digits):
            # Find examples of this digit
            digit_indices = np.where(np.argmax(data.y_test, axis=1) == digit)[0]
            sample_indices = digit_indices[:max_samples_per_digit]

            # For each model
            for model_idx, (model_name, _) in enumerate(self.models.items()):
                base_col = model_idx * (n_samples + 1)

                # Plot probability distribution
                ax_prob = fig.add_subplot(gs[digit_idx, base_col])

                # Get average predictions for this digit
                avg_predictions = np.zeros(10)
                for idx in sample_indices:
                    _, preds = self.get_activation_and_predictions(
                        data.x_test[idx],
                        model_name
                    )
                    avg_predictions += preds
                avg_predictions /= len(sample_indices)

                # Plot probability distribution
                ax_prob.bar(range(10), avg_predictions)
                ax_prob.set_ylim(0, 1)
                if digit_idx == 0:
                    ax_prob.set_title(f"{model_name}\nProbabilities")
                ax_prob.set_xticks(range(10))
                if digit_idx == n_digits - 1:
                    ax_prob.set_xlabel("Predicted Class")

                # For each sample of this digit
                for sample_num, idx in enumerate(sample_indices, 1):
                    ax = fig.add_subplot(gs[digit_idx, base_col + sample_num])

                    # Get activation map and predictions
                    act_map, predictions = self.get_activation_and_predictions(
                        data.x_test[idx],
                        model_name
                    )

                    # Plot activation map
                    im = ax.imshow(act_map, cmap='viridis')
                    ax.axis('off')

                    # Add prediction confidence as title
                    confidence = predictions[digit]
                    if digit_idx == 0:
                        ax.set_title(f"Act Map\n{confidence:.2f}")
                    else:
                        ax.set_title(f"{confidence:.2f}")

            # Add digit label on the left
            if model_idx == 0:
                ax_prob.set_ylabel(f"Digit {digit}")

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "comprehensive_analysis",
            "model_comparison"
        )

    def analyze_models(
            self,
            data: MNISTData,
            sample_digits: Optional[List[int]] = None,
            max_samples_per_digit: int = 3
    ) -> Dict[str, Any]:
        """Perform comprehensive model analysis.

        Args:
            data: MNIST dataset splits
            sample_digits: Optional list of digits to analyze
            max_samples_per_digit: Maximum samples to show per digit

        Returns:
            Dictionary containing analysis results
        """
        results = {}

        # Model evaluation
        for name, model in self.models.items():
            evaluation = model.evaluate(data.x_test, data.y_test)
            results[name] = dict(zip(model.metrics_names, evaluation))

        # Create comprehensive visualization
        self.create_comprehensive_visualization(
            data,
            sample_digits,
            max_samples_per_digit
        )

        return results