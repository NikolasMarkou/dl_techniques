"""
Enhanced Model Analyzer for CNN Visualization and Probability Distribution Analysis
==================================================================================

This module provides comprehensive analysis and visualization capabilities for comparing
multiple CNN models, with a focus on probability distribution analysis and out-of-distribution
(OOD) confidence detection. The analyzer combines traditional activation visualization with
advanced statistical analysis of model confidence patterns.

CORE FUNCTIONALITY
-----------------
The ModelAnalyzer provides two main analysis workflows:
1. **Activation Visualization**: Traditional CNN layer activation analysis and visualization
2. **Probability Distribution Analysis**: Comprehensive statistical analysis of model confidence patterns

ACTIVATION ANALYSIS FEATURES
---------------------------
- **Activation Map Extraction**: Extracts and visualizes intermediate layer activations
- **Model Comparison**: Side-by-side comparison of activation patterns across models
- **Class-specific Analysis**: Analyzes activation patterns for specific digit classes
- **Confidence Overlay**: Displays prediction confidence alongside activation maps

PROBABILITY DISTRIBUTION ANALYSIS FEATURES
-----------------------------------------
The enhanced analyzer provides deep insights into model confidence patterns through:

**Confidence Metrics Computation:**
- **Max Probability**: Primary confidence measure (highest class probability)
- **Entropy**: Uncertainty quantification (-∑p·log(p))
- **Margin**: Difference between top-2 predictions (confidence gap)
- **Gini Coefficient**: Probability distribution concentration measure

**Comprehensive Visualizations:**

1. **Probability Distributions by Class**
   - Histograms of probability values for each true class
   - Statistical summaries (mean, std) for confidence assessment
   - Identifies classes with poor calibration or unusual confidence patterns

2. **Confidence Analysis with Violin Plots**
   - Distribution shapes of confidence metrics across all classes
   - Comparative analysis between multiple models
   - Reveals systematic differences in uncertainty estimation

3. **Calibration Analysis**
   - Calibration plots comparing predicted confidence vs actual accuracy
   - Confidence histograms for correct vs incorrect predictions
   - Critical for detecting overconfident predictions on wrong classifications

4. **Class-wise Confidence Matrices**
   - Heatmaps showing mean confidence and variability per class combination
   - Identifies systematic confusion patterns and confidence biases
   - Separate visualization for confidence means and standard deviations

5. **Uncertainty Landscapes**
   - Multi-dimensional scatter plots of confidence metrics
   - Confidence vs Entropy plots colored by margin
   - Margin vs Gini coefficient plots colored by confidence
   - Reveals complex relationships between uncertainty measures

OUT-OF-DISTRIBUTION DETECTION CAPABILITIES
-----------------------------------------
The probability analysis is specifically designed to identify OOD samples through:

**Low Confidence Detection:**
- Samples with unusually low maximum probability
- High entropy predictions indicating model confusion
- Low margin between top predictions

**Calibration Assessment:**
- Overconfident predictions on incorrect classifications
- Poor alignment between confidence and accuracy
- Unusual probability distribution shapes

**Anomaly Pattern Recognition:**
- Samples that fall outside normal confidence ranges
- Unusual combinations of confidence metrics
- Class-specific confidence anomalies

USAGE PATTERNS
-------------

**Basic Usage:**
```python
# Initialize with models and visualization manager
analyzer = ModelAnalyzer(models_dict, vis_manager)

# Run complete analysis (includes both activation and probability analysis)
results = analyzer.analyze_models(mnist_data)
```

**Custom Analysis:**
```python
# Run only probability distribution analysis
analyzer.create_probability_distribution_analysis(mnist_data, n_samples=2000)

# Run only activation visualization
analyzer.create_comprehensive_visualization(mnist_data, sample_digits=[0,1,2])
```

**Confidence Metrics Access:**
```python
# Get detailed confidence metrics for further analysis
x_sample = data.x_test[:1000]
predictions = model.predict(x_sample)
confidence_metrics = analyzer._compute_confidence_metrics(predictions)
# Returns: max_probability, entropy, margin, gini_coefficient arrays
```

OUTPUT STRUCTURE
---------------
All visualizations are automatically saved through the VisualizationManager:

```
output_directory/
├── model_comparison/
│   └── comprehensive_analysis.png    # Activation maps and probability bars
└── probability_analysis/
    ├── probability_distributions_by_class.png  # Class-wise probability histograms
    ├── confidence_analysis.png                 # Violin plots of confidence metrics
    ├── calibration_analysis.png               # Calibration curves and confidence distributions
    ├── class_wise_confidence.png              # Confidence matrix heatmaps
    └── uncertainty_landscapes.png             # Multi-metric scatter plots
```

ANALYSIS RESULTS
---------------
The `analyze_models()` method returns a dictionary containing:
- Model evaluation metrics (loss, accuracy) for each model
- All visualizations saved to appropriate subdirectories
- Comprehensive probability analysis plots for OOD detection

DEPENDENCIES
-----------
- **Core**: keras, numpy, matplotlib, seaborn
- **Statistics**: scipy.stats for advanced statistical analysis
- **Calibration**: sklearn.metrics.calibration_curve for reliability assessment
- **Custom**: visualization_manager for consistent plot management
- **Data**: MNISTData class for structured dataset handling

RESEARCH APPLICATIONS
--------------------
This analyzer is designed for:
- **Model Comparison**: Systematic comparison of normalization techniques
- **Confidence Calibration**: Assessing and improving model reliability
- **OOD Detection**: Identifying samples that models handle poorly
- **Uncertainty Quantification**: Understanding model confidence patterns
- **Failure Analysis**: Investigating why models make incorrect predictions
- **Robustness Assessment**: Evaluating model behavior on edge cases

The modular design allows easy extension to other datasets, model architectures,
or additional confidence metrics while maintaining comprehensive visualization capabilities.

Classes:
    ModelAnalyzer: Main analyzer class for CNN model comparison and analysis

Methods:
    analyze_models(): Complete analysis workflow
    create_comprehensive_visualization(): Activation map visualization
    create_probability_distribution_analysis(): Statistical confidence analysis
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Dict, Any, List
from sklearn.calibration import calibration_curve

# ------------------------------------------------------------------------------

from .datasets import MNISTData
from .visualization_manager import VisualizationManager

# ------------------------------------------------------------------------------

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

    def _compute_confidence_metrics(self, probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various confidence metrics from probability distributions.

        Args:
            probabilities: Array of shape (n_samples, n_classes)

        Returns:
            Dictionary containing computed confidence metrics
        """
        # Maximum probability (confidence)
        max_prob = np.max(probabilities, axis=1)

        # Entropy (uncertainty measure)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # Top-2 difference (margin)
        sorted_probs = np.sort(probabilities, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Gini coefficient (concentration measure)
        sorted_probs_desc = np.sort(probabilities, axis=1)[:, ::-1]
        n_classes = probabilities.shape[1]
        gini = 1 - np.sum(sorted_probs_desc ** 2, axis=1)

        return {
            'max_probability': max_prob,
            'entropy': entropy,
            'margin': margin,
            'gini_coefficient': gini
        }

    def create_probability_distribution_analysis(
            self,
            data: MNISTData,
            n_samples: int = 1000
    ) -> None:
        """Create comprehensive probability distribution analysis.

        Args:
            data: MNIST dataset splits
            n_samples: Number of samples to analyze
        """
        # Get predictions from all models
        model_predictions = {}
        model_confidences = {}

        # Sample data for analysis
        indices = np.random.choice(len(data.x_test), min(n_samples, len(data.x_test)), replace=False)
        x_sample = data.x_test[indices]
        y_sample = data.y_test[indices]
        y_true = np.argmax(y_sample, axis=1)

        # Get predictions and confidence metrics for each model
        for model_name, model in self.models.items():
            predictions = model.predict(x_sample)
            model_predictions[model_name] = predictions
            model_confidences[model_name] = self._compute_confidence_metrics(predictions)

        # Create comprehensive visualization
        self._plot_probability_distributions(model_predictions, y_true)
        self._plot_confidence_analysis(model_confidences, y_true)
        self._plot_calibration_analysis(model_predictions, y_true)
        self._plot_class_wise_confidence(model_predictions, y_true)
        self._plot_uncertainty_landscapes(model_confidences)

    def _plot_probability_distributions(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Plot probability distributions for each class and model."""
        n_models = len(model_predictions)
        n_classes = 10

        fig, axes = plt.subplots(n_classes, n_models, figsize=(4 * n_models, 2 * n_classes))
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for class_idx in range(n_classes):
            class_mask = (y_true == class_idx)

            for model_idx, (model_name, predictions) in enumerate(model_predictions.items()):
                ax = axes[class_idx, model_idx]

                if np.sum(class_mask) > 0:
                    class_probs = predictions[class_mask, class_idx]

                    # Plot histogram of probabilities for this class
                    ax.hist(class_probs, bins=30, alpha=0.7, density=True,
                            color=plt.cm.Set3(model_idx))

                    # Add statistics
                    mean_prob = np.mean(class_probs)
                    std_prob = np.std(class_probs)
                    ax.axvline(mean_prob, color='red', linestyle='--', alpha=0.8)
                    ax.set_title(f'Class {class_idx}\n{model_name}\n'
                                 f'μ={mean_prob:.3f}, σ={std_prob:.3f}')

                ax.set_xlim(0, 1)
                if class_idx == n_classes - 1:
                    ax.set_xlabel('Probability')
                if model_idx == 0:
                    ax.set_ylabel('Density')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "probability_distributions_by_class",
            "probability_analysis"
        )

    def _plot_confidence_analysis(
            self,
            model_confidences: Dict[str, Dict[str, np.ndarray]],
            y_true: np.ndarray
    ) -> None:
        """Plot comprehensive confidence analysis."""
        n_models = len(model_confidences)
        metrics = ['max_probability', 'entropy', 'margin', 'gini_coefficient']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            for model_name, confidences in model_confidences.items():
                values = confidences[metric]

                # Create box plot for each class
                class_data = []
                class_labels = []
                for class_idx in range(10):
                    class_mask = (y_true == class_idx)
                    if np.sum(class_mask) > 0:
                        class_data.append(values[class_mask])
                        class_labels.append(f'{class_idx}')

                # Plot violin plot
                positions = np.arange(len(class_data))
                parts = ax.violinplot(class_data, positions=positions, widths=0.8)

                # Color the violin plots
                color = plt.cm.Set3(list(model_confidences.keys()).index(model_name))
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

            ax.set_xlabel('Class')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution by Class')
            ax.set_xticks(range(10))
            ax.set_xticklabels([str(i) for i in range(10)])

        # Add legend
        handles = [plt.Line2D([0], [0], color=plt.cm.Set3(i), lw=4, alpha=0.7)
                   for i, model_name in enumerate(model_confidences.keys())]
        labels = list(model_confidences.keys())
        fig.legend(handles, labels, loc='upper right')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "confidence_analysis",
            "probability_analysis"
        )

    def _plot_calibration_analysis(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Plot calibration analysis for confidence assessment."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Calibration plot
        ax1 = axes[0]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            accuracy = (y_pred == y_true).astype(int)

            fraction_of_positives, mean_predicted_value = calibration_curve(
                accuracy, confidence, n_bins=10, strategy='uniform'
            )

            ax1.plot(mean_predicted_value, fraction_of_positives, 's-',
                     label=model_name, alpha=0.8, linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence histogram for correct vs incorrect predictions
        ax2 = axes[1]
        for model_idx, (model_name, predictions) in enumerate(model_predictions.items()):
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)

            correct_mask = (y_pred == y_true)
            incorrect_mask = ~correct_mask

            color = plt.cm.Set3(model_idx)
            ax2.hist(confidence[correct_mask], bins=30, alpha=0.5,
                     label=f'{model_name} - Correct', color=color, density=True)
            ax2.hist(confidence[incorrect_mask], bins=30, alpha=0.5,
                     label=f'{model_name} - Incorrect', color=color,
                     density=True, linestyle='--', histtype='step', linewidth=2)

        ax2.set_xlabel('Confidence (Max Probability)')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution: Correct vs Incorrect')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "calibration_analysis",
            "probability_analysis"
        )

    def _plot_class_wise_confidence(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Plot class-wise confidence patterns."""
        n_models = len(model_predictions)

        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for model_idx, (model_name, predictions) in enumerate(model_predictions.items()):
            # Confidence matrix heatmap
            ax1 = axes[0, model_idx]
            confidence_matrix = np.zeros((10, 10))

            for true_class in range(10):
                class_mask = (y_true == true_class)
                if np.sum(class_mask) > 0:
                    class_predictions = predictions[class_mask]
                    mean_confidences = np.mean(class_predictions, axis=0)
                    confidence_matrix[true_class] = mean_confidences

            im1 = ax1.imshow(confidence_matrix, cmap='viridis', aspect='auto')
            ax1.set_title(f'{model_name}\nMean Confidence Matrix')
            ax1.set_xlabel('Predicted Class')
            ax1.set_ylabel('True Class')
            ax1.set_xticks(range(10))
            ax1.set_yticks(range(10))
            plt.colorbar(im1, ax=ax1)

            # Confidence spread heatmap
            ax2 = axes[1, model_idx]
            std_matrix = np.zeros((10, 10))

            for true_class in range(10):
                class_mask = (y_true == true_class)
                if np.sum(class_mask) > 0:
                    class_predictions = predictions[class_mask]
                    std_confidences = np.std(class_predictions, axis=0)
                    std_matrix[true_class] = std_confidences

            im2 = ax2.imshow(std_matrix, cmap='plasma', aspect='auto')
            ax2.set_title(f'{model_name}\nConfidence Std Matrix')
            ax2.set_xlabel('Predicted Class')
            ax2.set_ylabel('True Class')
            ax2.set_xticks(range(10))
            ax2.set_yticks(range(10))
            plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "class_wise_confidence",
            "probability_analysis"
        )

    def _plot_uncertainty_landscapes(
            self,
            model_confidences: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Plot uncertainty landscape analysis."""
        n_models = len(model_confidences)

        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for model_idx, (model_name, confidences) in enumerate(model_confidences.items()):
            # Confidence vs Entropy scatter plot
            ax1 = axes[0, model_idx]
            scatter = ax1.scatter(confidences['max_probability'], confidences['entropy'],
                                  alpha=0.6, c=confidences['margin'], cmap='viridis', s=20)
            ax1.set_xlabel('Max Probability (Confidence)')
            ax1.set_ylabel('Entropy (Uncertainty)')
            ax1.set_title(f'{model_name}\nConfidence vs Uncertainty\n(colored by margin)')
            plt.colorbar(scatter, ax=ax1, label='Margin')
            ax1.grid(True, alpha=0.3)

            # Margin vs Gini coefficient
            ax2 = axes[1, model_idx]
            scatter2 = ax2.scatter(confidences['margin'], confidences['gini_coefficient'],
                                   alpha=0.6, c=confidences['max_probability'], cmap='plasma', s=20)
            ax2.set_xlabel('Margin (Top-2 Difference)')
            ax2.set_ylabel('Gini Coefficient')
            ax2.set_title(f'{model_name}\nMargin vs Concentration\n(colored by confidence)')
            plt.colorbar(scatter2, ax=ax2, label='Max Probability')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "uncertainty_landscapes",
            "probability_analysis"
        )

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

        # Create probability distribution analysis
        self.create_probability_distribution_analysis(data)

        return results

# ------------------------------------------------------------------------------
