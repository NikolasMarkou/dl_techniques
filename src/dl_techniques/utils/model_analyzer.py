"""
Enhanced Model Analyzer for Deep Learning Model Visualization and Analysis
=========================================================================

This module provides comprehensive analysis and visualization capabilities for comparing
multiple deep learning models (CNNs, Dense Networks, and other architectures), with a focus
on probability distribution analysis and out-of-distribution (OOD) confidence detection.
The analyzer combines traditional layer activation visualization with advanced statistical
analysis of model confidence patterns and information flow analysis.

ARCHITECTURE COMPATIBILITY
--------------------------
The analyzer is designed to work with ANY Keras model architecture:
- **CNNs**: Full support including activation maps from convolutional layers
- **Dense Networks (MLPs)**: Full support with dense layer visualization
- **Hybrid Architectures**: Automatically adapts to mixed conv/dense models
- **Custom Architectures**: Works with any model having standard Keras layers
- **Transformer/Attention Models**: Compatible with attention layers and normalization

CORE FUNCTIONALITY
-----------------
The ModelAnalyzer provides three main analysis workflows:
1. **Layer Activation Visualization**: Adaptive visualization for any layer type
2. **Probability Distribution Analysis**: Comprehensive statistical analysis of model confidence patterns
3. **Information Flow Analysis**: Deep insights into how information propagates through network layers

ARCHITECTURE COMPATIBILITY
--------------------------
The analyzer is designed to work with ANY Keras model architecture:
- **CNNs**: Full support including activation maps from convolutional layers
- **Dense Networks (MLPs)**: Full support with dense layer visualization
- **Hybrid Architectures**: Automatically adapts to mixed conv/dense models
- **Custom Architectures**: Works with any model having standard Keras layers
- **Transformer/Attention Models**: Compatible with attention layers and normalization

**What Works with Different Architectures:**

| Analysis Type | CNNs | Dense | Hybrid | Custom |
|---------------|------|-------|--------|--------|
| Probability Distribution Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Confidence Metrics | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Model Agreement Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Calibration Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| ROC Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Stability Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Threshold Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Information Flow Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Partial* |
| Layer Activation Visualization | ✅ Conv Maps | ✅ Dense Heatmaps | ✅ Adaptive | ✅ Fallback** |

*Custom architectures: Works with standard layers (Dense, Conv2D, BatchNorm, LayerNorm)
**Fallback: Shows confidence values if no suitable layers found for visualization

EFFICIENCY OPTIMIZATIONS
------------------------
The analyzer is designed for computational efficiency through:
- **Consolidated Inference**: All model predictions are computed once at the beginning and reused
- **Batch Processing**: Noisy samples and stability analysis are processed in batches
- **Pre-computation**: Complex analyses like stability testing pre-compute all required predictions
- **Silent Execution**: All Keras inference is silent with tqdm progress bars for clear status
- **Memory Efficient**: Uses smaller sample sizes for computationally intensive analyses

ACTIVATION ANALYSIS FEATURES
---------------------------
- **Adaptive Layer Visualization**: Automatically selects the best layer for visualization based on architecture
  - CNNs: Uses convolutional layer activation maps
  - Dense Networks: Visualizes dense layer activations as 2D heatmaps
  - Mixed Models: Prioritizes conv layers, falls back to dense layers
- **Multi-Architecture Support**: Works seamlessly with any Keras model type
- **Model Comparison**: Side-by-side comparison of layer patterns across different architectures
- **Class-specific Analysis**: Analyzes layer patterns for specific output classes
- **Confidence Overlay**: Displays prediction confidence alongside layer visualizations

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

6. **Model Agreement Analysis**
   - Agreement matrices showing where models agree/disagree
   - Class-wise disagreement patterns
   - Confidence analysis for agreed vs disagreed predictions
   - Disagreement strength distributions

7. **Statistical Distance Analysis**
   - KL Divergence and Jensen-Shannon divergence matrices between models
   - Per-class distance analysis revealing model behavior differences
   - Distribution distance patterns for systematic comparison

8. **ROC Analysis for Confidence**
   - ROC curves treating confidence as correctness predictor
   - Precision-Recall analysis for confidence quality assessment
   - Optimal threshold determination using Youden's index
   - AUC comparisons across models

9. **Prediction Stability Analysis**
   - Stability analysis across different noise levels (efficiently pre-computed)
   - Confidence change patterns with input perturbations
   - Sample-level stability distributions
   - Correlation analysis between stability and original confidence

10. **Confidence Correlation Analysis**
    - Correlation matrices between different confidence metrics
    - Multi-dimensional relationship visualization
    - Per-model metric dependency analysis

11. **Per-Class Calibration Analysis**
    - Individual calibration curves for each class
    - Class-specific calibration error quantification
    - Binary classification approach for detailed analysis

12. **Multi-Threshold Analysis**
    - Coverage vs accuracy trade-off curves
    - Rejection curve analysis for selective prediction
    - Threshold sensitivity analysis with F1 scores
    - Optimal operating point determination

13. **Information Flow Analysis**
    - Layer-wise activation statistics (magnitude, variance, sparsity, effective dimensionality)
    - Inter-layer correlation analysis revealing information propagation patterns
    - Representational Similarity Analysis (RSA) comparing model representations
    - Information bottleneck analysis using SVD entropy and compression ratios
    - Gradient flow analysis showing how gradients propagate through layers

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

**Model Agreement Patterns:**
- Samples where models strongly disagree (often OOD indicators)
- Consensus confidence analysis
- Systematic disagreement patterns

**Stability Analysis:**
- Samples with unstable predictions under noise
- Confidence degradation patterns
- Correlation between stability and original confidence

USAGE PATTERNS
-------------

**Basic Usage (Works with ANY model architecture):**
```python
# Works with CNNs, Dense networks, or any Keras model
cnn_models = {'cnn_baseline': cnn_model, 'cnn_normalized': cnn_norm_model}
dense_models = {'mlp_baseline': dense_model, 'mlp_dropout': dense_drop_model}
mixed_models = {'cnn': cnn_model, 'dense': dense_model, 'hybrid': hybrid_model}

# Initialize with any model dictionary
analyzer = ModelAnalyzer(mixed_models, vis_manager)

# Run complete analysis (same interface for all architectures)
results = analyzer.analyze_models(dataset)
```

**Architecture-Specific Examples:**
```python
# CNN Models - Full activation map visualization
cnn_analyzer = ModelAnalyzer(cnn_models, vis_manager)
cnn_results = cnn_analyzer.analyze_models(image_data)

# Dense Models - Dense layer heatmap visualization
dense_analyzer = ModelAnalyzer(dense_models, vis_manager)
dense_results = dense_analyzer.analyze_models(tabular_data)

# Mixed Architecture Comparison - Adaptive visualization
mixed_analyzer = ModelAnalyzer({'cnn': cnn, 'transformer': transformer, 'dense': mlp}, vis_manager)
mixed_results = mixed_analyzer.analyze_models(data)
```

**Custom Analysis:**
```python
# Run only probability distribution analysis (works with any architecture)
analyzer.create_probability_distribution_analysis(data, n_samples=2000)

# Run only activation visualization (adapts to architecture automatically)
analyzer.create_comprehensive_visualization(data, sample_digits=[0,1,2])
```

**Confidence Metrics Access:**
```python
# Get detailed confidence metrics for further analysis
x_sample = data.x_test[:1000]
predictions = model.predict(x_sample, verbose=0)
confidence_metrics = analyzer._compute_confidence_metrics(predictions)
# Returns: max_probability, entropy, margin, gini_coefficient arrays
```

COMPUTATIONAL EFFICIENCY
-----------------------
The analyzer is optimized for speed through several strategies:

**Consolidated Inference Phase:**
- All model predictions are computed once at the beginning
- Predictions are reused across all analysis functions
- No redundant inference calls during visualization generation

**Batch Processing:**
- Stability analysis processes samples in batches rather than individually
- Noisy sample generation is batched for efficiency
- Memory-efficient processing of large datasets

**Smart Sampling:**
- Uses appropriate sample sizes for different analyses (1000 for main analysis, 300 for stability)
- Balances statistical power with computational efficiency
- Progress tracking with tqdm for clear status updates

OUTPUT STRUCTURE
---------------
All visualizations are automatically saved through the VisualizationManager:

```
output_directory/
├── model_comparison/
│   └── comprehensive_analysis.png              # Activation maps and probability bars
├── probability_analysis/
│   ├── probability_distributions_by_class.png  # Class-wise probability histograms
│   ├── confidence_analysis.png                 # Violin plots of confidence metrics
│   ├── calibration_analysis.png               # Calibration curves and confidence distributions
│   ├── class_wise_confidence.png              # Confidence matrix heatmaps
│   ├── uncertainty_landscapes.png             # Multi-metric scatter plots
│   ├── model_agreement_analysis.png           # Agreement/disagreement patterns
│   ├── distribution_distances.png             # Statistical distance analysis
│   ├── confidence_roc_analysis.png            # ROC analysis for confidence
│   ├── prediction_stability.png               # Stability analysis results
│   ├── confidence_correlations.png            # Metric correlation analysis
│   ├── per_class_calibration.png              # Class-specific calibration
│   └── threshold_analysis.png                 # Multi-threshold analysis
└── information_flow/
    ├── layer_activation_statistics.png        # Layer-wise activation properties
    ├── layer_correlation_analysis.png         # Inter-layer correlation matrices
    ├── representational_similarity_analysis.png # RSA between models and layers
    ├── information_bottleneck_analysis.png    # Information content and compression
    └── gradient_flow_analysis.png             # Gradient propagation through layers
```

ANALYSIS RESULTS
---------------
The `analyze_models()` method returns a dictionary containing:
- Model evaluation metrics (loss, accuracy) for each model
- All visualizations saved to appropriate subdirectories
- Comprehensive probability analysis plots for OOD detection

DEPENDENCIES
-----------
- **Core**: keras, numpy, matplotlib, seaborn, tqdm
- **Statistics**: scipy.stats for advanced statistical analysis
- **Calibration**: sklearn.metrics.calibration_curve for reliability assessment
- **Custom**: visualization_manager for consistent plot management
- **Data**: MNISTData class for structured dataset handling

RESEARCH APPLICATIONS
--------------------
This analyzer is designed for:
- **Model Comparison**: Systematic comparison across any model architectures (CNN, Dense, Hybrid)
- **Confidence Calibration**: Assessing and improving model reliability across architectures
- **OOD Detection**: Identifying samples that models handle poorly regardless of architecture
- **Uncertainty Quantification**: Understanding model confidence patterns in any network type
- **Failure Analysis**: Investigating why models make incorrect predictions
- **Robustness Assessment**: Evaluating model behavior on edge cases
- **Deployment Planning**: Determining optimal confidence thresholds for any model type
- **Information Flow Analysis**: Understanding how information propagates through network layers
- **Architecture Design**: Insights for designing better network architectures (CNN, Dense, Hybrid)
- **Representation Learning**: Analyzing what different layers learn across various architectures
- **Transfer Learning**: Comparing representations between different model types
- **Architecture Search**: Evaluating different architectural choices systematically

The modular design allows easy extension to other datasets, model architectures,
or additional confidence metrics while maintaining comprehensive visualization capabilities
and computational efficiency.

Classes:
    ModelAnalyzer: Main analyzer class for CNN model comparison and analysis

Methods:
    analyze_models(): Complete analysis workflow with consolidated inference
    create_comprehensive_visualization(): Activation map visualization
    create_probability_distribution_analysis(): Statistical confidence analysis with pre-computed predictions
"""

import keras
import numpy as np
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from typing import Union, Optional, Tuple, Dict, Any, List
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from itertools import combinations


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
        """Create models to extract intermediate activations for any model architecture."""
        self.activation_models = {}
        self.layer_extraction_models = {}

        for model_name, model in self.models.items():
            # For activation visualization - find the best layer to visualize
            best_viz_layer = None

            # Priority: Conv2D > Dense > any layer with meaningful output
            conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
            dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]

            if conv_layers:
                # CNN: Use last conv layer
                best_viz_layer = conv_layers[-1]
            elif dense_layers:
                # Dense network: Use middle dense layer for visualization
                mid_idx = len(dense_layers) // 2
                best_viz_layer = dense_layers[mid_idx]
            else:
                # Other architectures: Use any layer with reasonable output
                for layer in reversed(model.layers):
                    if hasattr(layer, 'output') and layer.output is not None:
                        try:
                            output_shape = layer.output.shape
                            if len(output_shape) >= 2 and output_shape[-1] is not None:
                                best_viz_layer = layer
                                break
                        except:
                            continue

            # Create activation model if we found a suitable layer
            if best_viz_layer is not None:
                try:
                    self.activation_models[model_name] = keras.Model(
                        inputs=model.input,
                        outputs=[
                            best_viz_layer.output,  # Best layer for visualization
                            model.output  # Final predictions
                        ]
                    )
                except:
                    # Fallback: just use model output
                    self.activation_models[model_name] = keras.Model(
                        inputs=model.input,
                        outputs=model.output
                    )

            # Create multi-layer extraction model for information flow analysis
            extraction_layers = []
            layer_names = []

            # Extract from key layers: conv, dense, normalization, attention, etc.
            for layer in model.layers:
                if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                    keras.layers.BatchNormalization, keras.layers.LayerNormalization,
                                    keras.layers.Dropout, keras.layers.Activation)):
                    # Skip certain layers that don't have meaningful outputs
                    if isinstance(layer, (keras.layers.Dropout, keras.layers.Activation)):
                        if layer.name in ['dropout', 'activation']:  # Skip generic named layers
                            continue

                    try:
                        if hasattr(layer, 'output') and layer.output is not None:
                            # Check if output shape is reasonable
                            output_shape = layer.output.shape
                            if len(output_shape) >= 2 and output_shape[-1] is not None:
                                extraction_layers.append(layer.output)
                                layer_names.append(f"{layer.__class__.__name__}_{layer.name}")
                    except:
                        continue

            if extraction_layers:
                try:
                    self.layer_extraction_models[model_name] = {
                        'model': keras.Model(inputs=model.input, outputs=extraction_layers),
                        'layer_names': layer_names
                    }
                except:
                    # If multi-output model creation fails, skip information flow for this model
                    self.layer_extraction_models[model_name] = None

    def get_activation_and_predictions(
            self,
            image: np.ndarray,
            model_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get layer activations and predictions for a single sample (works with any model type).

        Args:
            image: Input sample
            model_name: Name of the model to use

        Returns:
            Tuple of (layer_activations, predictions)
        """
        if model_name not in self.activation_models:
            # Fallback: just get predictions
            model = self.models[model_name]
            sample_batch = np.expand_dims(image, 0)
            predictions = model.predict(sample_batch, verbose=0)
            # Return dummy activation (zeros) and predictions
            dummy_activation = np.zeros((28, 28))  # Default shape
            return dummy_activation, predictions[0]

        model = self.activation_models[model_name]
        sample_batch = np.expand_dims(image, 0)

        try:
            # Try to get both activations and predictions
            outputs = model.predict(sample_batch, verbose=0)

            if isinstance(outputs, list) and len(outputs) == 2:
                # We have both activation and prediction outputs
                layer_output, predictions = outputs

                # Process layer output based on its shape
                if len(layer_output.shape) == 4:
                    # Conv layer: (batch, h, w, channels) -> average across channels
                    processed_activation = np.mean(layer_output[0], axis=-1)
                elif len(layer_output.shape) == 2:
                    # Dense layer: (batch, units) -> reshape to 2D for visualization
                    units = layer_output.shape[1]
                    side_length = int(np.sqrt(units))
                    if side_length * side_length == units:
                        processed_activation = layer_output[0].reshape(side_length, side_length)
                    else:
                        # Create a rectangular visualization
                        cols = min(int(np.sqrt(units)), 32)
                        rows = (units + cols - 1) // cols
                        padded_activation = np.pad(layer_output[0], (0, rows * cols - units), mode='constant')
                        processed_activation = padded_activation.reshape(rows, cols)
                else:
                    # Other shapes: flatten and reshape to square
                    flattened = layer_output[0].flatten()
                    side_length = int(np.sqrt(len(flattened)))
                    if side_length > 0:
                        processed_activation = flattened[:side_length*side_length].reshape(side_length, side_length)
                    else:
                        processed_activation = np.zeros((10, 10))

                return processed_activation, predictions[0]
            else:
                # Only predictions available
                predictions = outputs if not isinstance(outputs, list) else outputs[0]
                dummy_activation = np.zeros((28, 28))
                return dummy_activation, predictions[0] if len(predictions.shape) > 1 else predictions

        except Exception as e:
            # Complete fallback
            model = self.models[model_name]
            predictions = model.predict(sample_batch, verbose=0)
            dummy_activation = np.zeros((28, 28))
            return dummy_activation, predictions[0]

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
        gini = 1 - np.sum(sorted_probs_desc**2, axis=1)

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
        # Sample data for analysis
        indices = np.random.choice(len(data.x_test), min(n_samples, len(data.x_test)), replace=False)
        x_sample = data.x_test[indices]
        y_sample = data.y_test[indices]
        y_true = np.argmax(y_sample, axis=1)

        # CONSOLIDATED INFERENCE PHASE - DO ALL PREDICTIONS AT ONCE
        print("🔮 Performing consolidated model inference (this may take a moment)...")

        # 1. Original predictions for all models
        model_predictions = {}
        model_confidences = {}

        for model_name, model in tqdm(self.models.items(), desc="Original Predictions"):
            predictions = model.predict(x_sample, verbose=0)
            model_predictions[model_name] = predictions
            model_confidences[model_name] = self._compute_confidence_metrics(predictions)

        # 2. Predictions for stability analysis (pre-compute noisy samples)
        print("🔄 Pre-computing predictions for stability analysis...")
        noise_levels = [0.01, 0.05, 0.1]
        stability_predictions = {}

        # Use smaller subset for stability analysis for efficiency
        n_stability_samples = min(300, len(x_sample))
        stability_indices = np.random.choice(len(x_sample), n_stability_samples, replace=False)
        x_stability = x_sample[stability_indices]
        y_stability = y_true[stability_indices]

        for model_name, model in tqdm(self.models.items(), desc="Stability Predictions"):
            stability_predictions[model_name] = {
                'original': model.predict(x_stability, verbose=0),
                'noisy': {}
            }

            # Pre-compute predictions for different noise levels
            for noise_std in noise_levels:
                noisy_x = x_stability + np.random.normal(0, noise_std, x_stability.shape)
                noisy_x = np.clip(noisy_x, 0, 1)
                stability_predictions[model_name]['noisy'][noise_std] = model.predict(noisy_x, verbose=0)

        # 3. Sample-level stability analysis (batch process)
        print("🎯 Computing sample-level stability predictions...")
        sample_stability_data = {}
        first_model_name = list(self.models.keys())[0]
        first_model = self.models[first_model_name]

        # Create batch of noisy samples (10 noise realizations per sample)
        n_realizations = 10
        noise_std = 0.05
        batch_size = min(100, n_stability_samples)  # Process in smaller batches

        sample_stabilities = []
        original_preds_stability = stability_predictions[first_model_name]['original']
        original_classes_stability = np.argmax(original_preds_stability, axis=1)

        for i in tqdm(range(0, n_stability_samples, batch_size), desc="Sample Stability Batches"):
            end_idx = min(i + batch_size, n_stability_samples)
            batch_x = x_stability[i:end_idx]

            # Create all noisy versions for this batch
            noisy_batch = []
            for sample_idx in range(len(batch_x)):
                for _ in range(n_realizations):
                    noisy_sample = batch_x[sample_idx:sample_idx+1] + np.random.normal(0, noise_std, batch_x[sample_idx:sample_idx+1].shape)
                    noisy_sample = np.clip(noisy_sample, 0, 1)
                    noisy_batch.append(noisy_sample[0])

            # Batch predict all noisy samples
            if noisy_batch:
                noisy_batch = np.array(noisy_batch)
                batch_predictions = first_model.predict(noisy_batch, verbose=0)
                batch_classes = np.argmax(batch_predictions, axis=1)

                # Process results for each original sample
                for sample_idx in range(len(batch_x)):
                    original_class = original_classes_stability[i + sample_idx]
                    start_pred = sample_idx * n_realizations
                    end_pred = start_pred + n_realizations
                    sample_pred_classes = batch_classes[start_pred:end_pred]

                    stability = np.mean(sample_pred_classes == original_class)
                    sample_stabilities.append(stability)

        sample_stability_data = {
            'stabilities': np.array(sample_stabilities),
            'original_confidence': np.max(original_preds_stability[:len(sample_stabilities)], axis=1)
        }

        print("✅ All predictions computed! Generating visualizations...")

        # Create comprehensive visualization using pre-computed predictions
        self._plot_probability_distributions(model_predictions, y_true)
        self._plot_confidence_analysis(model_confidences, y_true)
        self._plot_calibration_analysis(model_predictions, y_true)
        self._plot_class_wise_confidence(model_predictions, y_true)
        self._plot_uncertainty_landscapes(model_confidences)

        # New comprehensive analyses using pre-computed predictions
        print("🤝 Analyzing model agreement patterns...")
        self._plot_model_agreement_analysis(model_predictions, y_true)
        print("📏 Computing statistical distribution distances...")
        self._plot_distribution_distances(model_predictions)
        print("📊 Performing ROC analysis for confidence...")
        self._plot_confidence_roc_analysis(model_predictions, y_true)
        print("🔄 Analyzing prediction stability...")
        self._plot_prediction_stability_consolidated(stability_predictions, y_stability, sample_stability_data, noise_levels)
        print("🔗 Analyzing confidence metric correlations...")
        self._plot_confidence_correlations(model_confidences)
        print("🎯 Computing per-class calibration curves...")
        self._plot_per_class_calibration(model_predictions, y_true)
        print("⚖️ Performing threshold analysis...")
        self._plot_threshold_analysis(model_predictions, y_true)

        # Information flow analysis using pre-computed activations
        print("🌊 Analyzing information flow between layers...")
        self._analyze_information_flow(x_sample)

    def _plot_probability_distributions(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Plot probability distributions for each class and model."""
        n_models = len(model_predictions)
        n_classes = 10

        fig, axes = plt.subplots(n_classes, n_models, figsize=(4*n_models, 2*n_classes))
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

        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
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

        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
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

    def _plot_model_agreement_analysis(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Analyze where models agree/disagree - key indicator of OOD samples."""
        model_names = list(model_predictions.keys())
        n_models = len(model_names)

        if n_models < 2:
            return

        # Get predicted classes for each model
        model_pred_classes = {}
        for name, preds in model_predictions.items():
            model_pred_classes[name] = np.argmax(preds, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Agreement matrix
        ax1 = axes[0, 0]
        agreement_matrix = np.zeros((n_models, n_models))
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    agreement = np.mean(model_pred_classes[name1] == model_pred_classes[name2])
                    agreement_matrix[i, j] = agreement
                else:
                    agreement_matrix[i, j] = 1.0

        im1 = ax1.imshow(agreement_matrix, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Model Agreement Matrix')
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels(model_names)
        ax1.set_yticklabels(model_names)
        plt.colorbar(im1, ax=ax1)

        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                ax1.text(j, i, f'{agreement_matrix[i, j]:.3f}',
                        ha='center', va='center', color='white', fontweight='bold')

        # Disagreement analysis by class
        ax2 = axes[0, 1]
        class_disagreement = np.zeros(10)
        for class_idx in range(10):
            class_mask = (y_true == class_idx)
            if np.sum(class_mask) > 0:
                class_preds = [model_pred_classes[name][class_mask] for name in model_names]
                # Calculate pairwise disagreement
                disagreements = []
                for i in range(len(class_preds)):
                    for j in range(i+1, len(class_preds)):
                        disagreement = np.mean(class_preds[i] != class_preds[j])
                        disagreements.append(disagreement)
                class_disagreement[class_idx] = np.mean(disagreements) if disagreements else 0

        ax2.bar(range(10), class_disagreement, color='coral', alpha=0.7)
        ax2.set_title('Model Disagreement by True Class')
        ax2.set_xlabel('True Class')
        ax2.set_ylabel('Average Disagreement Rate')
        ax2.set_xticks(range(10))

        # Confidence when models agree vs disagree
        ax3 = axes[1, 0]
        for model_name, predictions in model_predictions.items():
            confidences = np.max(predictions, axis=1)

            # Find samples where all models agree
            all_preds = [model_pred_classes[name] for name in model_names]
            agreement_mask = np.ones(len(y_true), dtype=bool)
            for i in range(len(all_preds)):
                for j in range(i+1, len(all_preds)):
                    agreement_mask &= (all_preds[i] == all_preds[j])

            ax3.hist(confidences[agreement_mask], bins=30, alpha=0.5,
                    label=f'{model_name} - Agree', density=True)
            ax3.hist(confidences[~agreement_mask], bins=30, alpha=0.5,
                    label=f'{model_name} - Disagree', density=True, histtype='step', linewidth=2)

        ax3.set_title('Confidence: Model Agreement vs Disagreement')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Density')
        ax3.legend()

        # Disagreement strength analysis
        ax4 = axes[1, 1]
        disagreement_strength = np.zeros(len(y_true))
        for i in range(len(y_true)):
            sample_preds = [model_pred_classes[name][i] for name in model_names]
            unique_preds = len(set(sample_preds))
            disagreement_strength[i] = (unique_preds - 1) / (n_models - 1)

        ax4.hist(disagreement_strength, bins=20, alpha=0.7, color='orange')
        ax4.set_title('Distribution of Disagreement Strength')
        ax4.set_xlabel('Disagreement Strength (0=all agree, 1=max disagreement)')
        ax4.set_ylabel('Frequency')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "model_agreement_analysis",
            "probability_analysis"
        )

    def _plot_distribution_distances(
            self,
            model_predictions: Dict[str, np.ndarray]
    ) -> None:
        """Compare probability distributions using statistical distances."""
        model_names = list(model_predictions.keys())
        n_models = len(model_names)

        if n_models < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # KL Divergence matrix
        ax1 = axes[0, 0]
        kl_matrix = np.zeros((n_models, n_models))
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    # Average KL divergence across samples
                    kl_divs = []
                    for k in range(len(model_predictions[name1])):
                        p = model_predictions[name1][k] + 1e-8
                        q = model_predictions[name2][k] + 1e-8
                        kl_div = np.sum(p * np.log(p / q))
                        kl_divs.append(kl_div)
                    kl_matrix[i, j] = np.mean(kl_divs)

        im1 = ax1.imshow(kl_matrix, cmap='plasma')
        ax1.set_title('KL Divergence Matrix')
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels(model_names)
        ax1.set_yticklabels(model_names)
        plt.colorbar(im1, ax=ax1)

        # Jensen-Shannon divergence matrix
        ax2 = axes[0, 1]
        js_matrix = np.zeros((n_models, n_models))
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    js_divs = []
                    for k in range(len(model_predictions[name1])):
                        p = model_predictions[name1][k]
                        q = model_predictions[name2][k]
                        js_div = jensenshannon(p, q)**2
                        js_divs.append(js_div)
                    js_matrix[i, j] = np.mean(js_divs)

        im2 = ax2.imshow(js_matrix, cmap='viridis')
        ax2.set_title('Jensen-Shannon Divergence Matrix')
        ax2.set_xticks(range(n_models))
        ax2.set_yticks(range(n_models))
        ax2.set_xticklabels(model_names)
        ax2.set_yticklabels(model_names)
        plt.colorbar(im2, ax=ax2)

        # Per-class distance analysis
        ax3 = axes[1, 0]
        if n_models >= 2:
            # Take first two models for per-class analysis
            name1, name2 = model_names[0], model_names[1]
            class_distances = []

            for class_idx in range(10):
                # Get average predictions for this class
                pred1_class = np.mean(model_predictions[name1], axis=0)
                pred2_class = np.mean(model_predictions[name2], axis=0)

                js_dist = jensenshannon(pred1_class, pred2_class)**2
                class_distances.append(js_dist)

            ax3.bar(range(10), class_distances, color='teal', alpha=0.7)
            ax3.set_title(f'JS Distance by Class: {name1} vs {name2}')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Jensen-Shannon Distance')
            ax3.set_xticks(range(10))

        # Distance distribution
        ax4 = axes[1, 1]
        all_distances = []
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names[i+1:], i+1):
                distances = []
                for k in range(len(model_predictions[name1])):
                    p = model_predictions[name1][k]
                    q = model_predictions[name2][k]
                    dist = jensenshannon(p, q)**2
                    distances.append(dist)
                all_distances.extend(distances)

        if all_distances:
            ax4.hist(all_distances, bins=50, alpha=0.7, color='purple')
            ax4.set_title('Distribution of JS Distances Between Models')
            ax4.set_xlabel('Jensen-Shannon Distance')
            ax4.set_ylabel('Frequency')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "distribution_distances",
            "probability_analysis"
        )

    def _plot_confidence_roc_analysis(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Treat confidence as binary classifier for correct/incorrect predictions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC curves for confidence predicting correctness
        ax1 = axes[0, 0]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            correctness = (y_pred == y_true).astype(int)

            fpr, tpr, _ = roc_curve(correctness, confidence)
            roc_auc = auc(fpr, tpr)

            ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC: Confidence Predicting Correctness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision-Recall curves
        ax2 = axes[0, 1]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            correctness = (y_pred == y_true).astype(int)

            precision, recall, _ = precision_recall_curve(correctness, confidence)
            pr_auc = auc(recall, precision)

            ax2.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall: Confidence Predicting Correctness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Optimal threshold analysis
        ax3 = axes[1, 0]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            correctness = (y_pred == y_true).astype(int)

            fpr, tpr, thresholds = roc_curve(correctness, confidence)
            # Youden's index for optimal threshold
            youdens_index = tpr - fpr
            optimal_idx = np.argmax(youdens_index)
            optimal_threshold = thresholds[optimal_idx]

            ax3.plot(thresholds, tpr, label=f'{model_name} TPR', alpha=0.7)
            ax3.plot(thresholds, 1-fpr, label=f'{model_name} TNR', alpha=0.7, linestyle='--')
            ax3.axvline(optimal_threshold, color=plt.cm.Set3(list(model_predictions.keys()).index(model_name)),
                       linestyle=':', alpha=0.8, label=f'{model_name} Optimal: {optimal_threshold:.3f}')

        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Rate')
        ax3.set_title('ROC Threshold Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # AUC comparison
        ax4 = axes[1, 1]
        model_aucs = []
        model_labels = []
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            correctness = (y_pred == y_true).astype(int)

            fpr, tpr, _ = roc_curve(correctness, confidence)
            roc_auc = auc(fpr, tpr)
            model_aucs.append(roc_auc)
            model_labels.append(model_name)

        bars = ax4.bar(model_labels, model_aucs, color=plt.cm.Set3(np.arange(len(model_labels))), alpha=0.7)
        ax4.set_title('ROC AUC Comparison')
        ax4.set_ylabel('AUC Score')
        ax4.set_ylim(0, 1)

        # Add value labels on bars
        for bar, auc_val in zip(bars, model_aucs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "confidence_roc_analysis",
            "probability_analysis"
        )

    def _plot_prediction_stability_consolidated(
            self,
            stability_predictions: Dict[str, Dict],
            y_true: np.ndarray,
            sample_stability_data: Dict[str, np.ndarray],
            noise_levels: List[float]
    ) -> None:
        """Analyze prediction stability using pre-computed predictions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Stability vs noise level (using pre-computed predictions)
        ax1 = axes[0, 0]
        for model_name, pred_data in stability_predictions.items():
            original_classes = np.argmax(pred_data['original'], axis=1)

            stabilities = []
            for noise_std in noise_levels:
                noisy_classes = np.argmax(pred_data['noisy'][noise_std], axis=1)
                stability = np.mean(original_classes == noisy_classes)
                stabilities.append(stability)

            ax1.plot(noise_levels, stabilities, 'o-', label=model_name, linewidth=2)

        ax1.set_xlabel('Noise Level (std)')
        ax1.set_ylabel('Prediction Stability')
        ax1.set_title('Prediction Stability vs Noise Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Confidence change with noise (using pre-computed predictions)
        ax2 = axes[0, 1]
        noise_std = 0.05  # Fixed noise level for this analysis
        for model_name, pred_data in stability_predictions.items():
            original_conf = np.max(pred_data['original'], axis=1)
            noisy_conf = np.max(pred_data['noisy'][noise_std], axis=1)
            confidence_change = noisy_conf - original_conf

            ax2.hist(confidence_change, bins=30, alpha=0.6, label=model_name, density=True)

        ax2.set_xlabel('Confidence Change')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Confidence Change with Noise (σ={noise_std})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Sample-level stability distribution (using pre-computed data)
        ax3 = axes[1, 0]
        sample_stabilities = sample_stability_data['stabilities']

        ax3.hist(sample_stabilities, bins=20, alpha=0.7, color='green')
        ax3.set_xlabel('Sample Stability')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Sample-Level Stability')
        ax3.grid(True, alpha=0.3)

        # Stability vs original confidence (using pre-computed data)
        ax4 = axes[1, 1]
        original_conf = sample_stability_data['original_confidence']

        ax4.scatter(original_conf, sample_stabilities, alpha=0.6, color='purple')
        ax4.set_xlabel('Original Confidence')
        ax4.set_ylabel('Stability')
        ax4.set_title('Stability vs Original Confidence')
        ax4.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(original_conf) > 0 and len(sample_stabilities) > 0:
            corr_coef = np.corrcoef(original_conf, sample_stabilities)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}',
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "prediction_stability",
            "probability_analysis"
        )

    def _plot_confidence_correlations(
            self,
            model_confidences: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Analyze relationships between different confidence metrics."""
        metrics = ['max_probability', 'entropy', 'margin', 'gini_coefficient']
        n_models = len(model_confidences)

        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        for model_idx, (model_name, confidences) in enumerate(model_confidences.items()):
            # Correlation matrix
            ax1 = axes[model_idx, 0]
            corr_data = np.array([confidences[metric] for metric in metrics]).T
            corr_matrix = np.corrcoef(corr_data.T)

            im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax1.set_title(f'{model_name}: Confidence Metrics Correlation')
            ax1.set_xticks(range(len(metrics)))
            ax1.set_yticks(range(len(metrics)))
            ax1.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            ax1.set_yticklabels([m.replace('_', '\n') for m in metrics])

            # Add correlation values
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                            fontweight='bold')

            plt.colorbar(im, ax=ax1)

            # Pairwise scatter plot (entropy vs max_probability)
            ax2 = axes[model_idx, 1]
            scatter = ax2.scatter(confidences['max_probability'], confidences['entropy'],
                                alpha=0.6, c=confidences['margin'], cmap='viridis', s=20)
            ax2.set_xlabel('Max Probability')
            ax2.set_ylabel('Entropy')
            ax2.set_title(f'{model_name}: Confidence vs Uncertainty')
            plt.colorbar(scatter, ax=ax2, label='Margin')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "confidence_correlations",
            "probability_analysis"
        )

    def _plot_per_class_calibration(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Detailed calibration analysis per class."""
        n_models = len(model_predictions)
        n_classes = 10

        fig, axes = plt.subplots(n_classes, n_models, figsize=(5*n_models, 2*n_classes))
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for class_idx in range(n_classes):
            class_mask = (y_true == class_idx)

            for model_idx, (model_name, predictions) in enumerate(model_predictions.items()):
                ax = axes[class_idx, model_idx]

                if np.sum(class_mask) > 10:  # Need sufficient samples
                    # Binary classification: this class vs others
                    y_binary = (y_true == class_idx).astype(int)
                    confidence_class = predictions[:, class_idx]

                    try:
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_binary, confidence_class, n_bins=5, strategy='uniform'
                        )

                        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                               linewidth=2, markersize=8, alpha=0.8)
                        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8)

                        # Calculate calibration error
                        cal_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

                        ax.set_title(f'Class {class_idx}\n{model_name}\nCal Error: {cal_error:.3f}')

                    except:
                        ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_title(f'Class {class_idx}\n{model_name}')
                else:
                    ax.text(0.5, 0.5, 'Insufficient\nSamples', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'Class {class_idx}\n{model_name}')

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

                if class_idx == n_classes - 1:
                    ax.set_xlabel('Mean Predicted Probability')
                if model_idx == 0:
                    ax.set_ylabel('Fraction of Positives')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "per_class_calibration",
            "probability_analysis"
        )

    def _plot_threshold_analysis(
            self,
            model_predictions: Dict[str, np.ndarray],
            y_true: np.ndarray
    ) -> None:
        """Comprehensive analysis across confidence thresholds."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        thresholds = np.linspace(0.1, 0.99, 20)

        # Coverage vs Accuracy curves
        ax1 = axes[0, 0]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)

            coverages = []
            accuracies = []

            for threshold in thresholds:
                mask = confidence >= threshold
                if np.sum(mask) > 0:
                    coverage = np.mean(mask)
                    accuracy = np.mean(y_pred[mask] == y_true[mask])
                else:
                    coverage = 0
                    accuracy = 0

                coverages.append(coverage)
                accuracies.append(accuracy)

            ax1.plot(coverages, accuracies, 'o-', label=model_name, linewidth=2)

        ax1.set_xlabel('Coverage (Fraction of Samples)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Coverage vs Accuracy Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rejection curves
        ax2 = axes[0, 1]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)

            rejection_rates = []
            remaining_accuracies = []

            for threshold in thresholds:
                mask = confidence >= threshold
                rejection_rate = 1 - np.mean(mask)

                if np.sum(mask) > 0:
                    remaining_accuracy = np.mean(y_pred[mask] == y_true[mask])
                else:
                    remaining_accuracy = 0

                rejection_rates.append(rejection_rate)
                remaining_accuracies.append(remaining_accuracy)

            ax2.plot(rejection_rates, remaining_accuracies, 's-', label=model_name, linewidth=2)

        ax2.set_xlabel('Rejection Rate')
        ax2.set_ylabel('Remaining Accuracy')
        ax2.set_title('Rejection Curve Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Threshold sensitivity
        ax3 = axes[1, 0]
        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)

            f1_scores = []
            precisions = []
            recalls = []

            for threshold in thresholds:
                mask = confidence >= threshold

                if np.sum(mask) > 0 and np.sum(~mask) > 0:
                    # Treat high confidence as positive class
                    tp = np.sum((y_pred == y_true) & mask)
                    fp = np.sum((y_pred != y_true) & mask)
                    fn = np.sum((y_pred == y_true) & ~mask)

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = recall = f1 = 0

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

            ax3.plot(thresholds, f1_scores, 'o-', label=f'{model_name} F1', linewidth=2)

        ax3.set_xlabel('Confidence Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Threshold Sensitivity Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Optimal operating points
        ax4 = axes[1, 1]
        model_optimal_thresholds = []
        model_labels = []

        for model_name, predictions in model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            correctness = (y_pred == y_true).astype(int)

            # Find threshold that maximizes Youden's index
            fpr, tpr, thresh = roc_curve(correctness, confidence)
            youdens_index = tpr - fpr
            optimal_idx = np.argmax(youdens_index)
            optimal_threshold = thresh[optimal_idx]

            model_optimal_thresholds.append(optimal_threshold)
            model_labels.append(model_name)

        bars = ax4.bar(model_labels, model_optimal_thresholds,
                      color=plt.cm.Set3(np.arange(len(model_labels))), alpha=0.7)
        ax4.set_title('Optimal Confidence Thresholds')
        ax4.set_ylabel('Optimal Threshold')
        ax4.set_ylim(0, 1)

        # Add value labels
        for bar, threshold in zip(bars, model_optimal_thresholds):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{threshold:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "threshold_analysis",
            "probability_analysis"
        )

    def _analyze_information_flow(self, x_sample: np.ndarray) -> None:
        """Comprehensive information flow analysis between layers."""
        # Extract activations from all layers for all models
        print("🔍 Extracting multi-layer activations...")
        model_layer_activations = {}

        # Use smaller sample for computational efficiency
        n_flow_samples = min(200, len(x_sample))
        flow_indices = np.random.choice(len(x_sample), n_flow_samples, replace=False)
        x_flow = x_sample[flow_indices]

        for model_name, extraction_data in tqdm(self.layer_extraction_models.items(), desc="Layer Extraction"):
            if extraction_data:
                layer_outputs = extraction_data['model'].predict(x_flow, verbose=0)
                layer_names = extraction_data['layer_names']

                # Process each layer's activations
                processed_activations = {}
                for i, (layer_output, layer_name) in enumerate(zip(layer_outputs, layer_names)):
                    # Flatten spatial dimensions but keep batch and feature dimensions
                    if len(layer_output.shape) == 4:  # Conv layer: (batch, h, w, channels)
                        # Global average pooling to reduce spatial dimensions
                        flattened = np.mean(layer_output, axis=(1, 2))  # (batch, channels)
                    elif len(layer_output.shape) == 2:  # Dense layer: (batch, units)
                        flattened = layer_output
                    else:
                        # For other shapes, flatten everything except batch dimension
                        flattened = layer_output.reshape(layer_output.shape[0], -1)

                    processed_activations[layer_name] = flattened

                model_layer_activations[model_name] = processed_activations

        # Generate all information flow analyses
        self._plot_layer_activation_statistics(model_layer_activations)
        self._plot_layer_correlation_analysis(model_layer_activations)
        self._plot_representational_similarity_analysis(model_layer_activations)
        self._plot_information_bottleneck_analysis(model_layer_activations)
        self._plot_layer_gradient_flow_analysis(x_flow)

    def _plot_layer_activation_statistics(
            self,
            model_layer_activations: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Plot statistical properties of activations across layers."""
        n_models = len(model_layer_activations)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Activation magnitude evolution
        ax1 = axes[0, 0]
        for model_name, layer_acts in model_layer_activations.items():
            layer_names = list(layer_acts.keys())
            mean_magnitudes = []

            for layer_name, activations in layer_acts.items():
                mean_mag = np.mean(np.abs(activations))
                mean_magnitudes.append(mean_mag)

            ax1.plot(range(len(layer_names)), mean_magnitudes, 'o-',
                    label=model_name, linewidth=2, markersize=6)

        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Mean Activation Magnitude')
        ax1.set_title('Activation Magnitude Evolution Across Layers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Activation variance evolution
        ax2 = axes[0, 1]
        for model_name, layer_acts in model_layer_activations.items():
            variances = []

            for layer_name, activations in layer_acts.items():
                variance = np.var(activations)
                variances.append(variance)

            ax2.plot(range(len(layer_acts)), variances, 's-',
                    label=model_name, linewidth=2, markersize=6)

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Activation Variance')
        ax2.set_title('Activation Variance Evolution Across Layers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Sparsity evolution (fraction of near-zero activations)
        ax3 = axes[1, 0]
        for model_name, layer_acts in model_layer_activations.items():
            sparsities = []

            for layer_name, activations in layer_acts.items():
                # Count activations close to zero (< 0.001)
                sparsity = np.mean(np.abs(activations) < 0.001)
                sparsities.append(sparsity)

            ax3.plot(range(len(layer_acts)), sparsities, '^-',
                    label=model_name, linewidth=2, markersize=6)

        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Sparsity (Fraction Near-Zero)')
        ax3.set_title('Activation Sparsity Evolution Across Layers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Effective dimensionality (using participation ratio)
        ax4 = axes[1, 1]
        for model_name, layer_acts in model_layer_activations.items():
            eff_dims = []

            for layer_name, activations in layer_acts.items():
                # Compute participation ratio as measure of effective dimensionality
                mean_act = np.mean(activations, axis=0)
                var_act = np.var(activations, axis=0)
                # Participation ratio: (sum of variances)^2 / sum of (variances^2)
                if np.sum(var_act**2) > 1e-10:
                    participation_ratio = (np.sum(var_act)**2) / np.sum(var_act**2)
                    eff_dims.append(participation_ratio)
                else:
                    eff_dims.append(0)

            ax4.plot(range(len(layer_acts)), eff_dims, 'd-',
                    label=model_name, linewidth=2, markersize=6)

        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('Effective Dimensionality')
        ax4.set_title('Effective Dimensionality Across Layers')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "layer_activation_statistics",
            "information_flow"
        )

    def _plot_layer_correlation_analysis(
            self,
            model_layer_activations: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Plot correlation analysis between consecutive layers."""
        n_models = len(model_layer_activations)

        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        if n_models == 1:
            axes = [axes]

        for model_idx, (model_name, layer_acts) in enumerate(model_layer_activations.items()):
            ax = axes[model_idx]

            layer_names = list(layer_acts.keys())
            n_layers = len(layer_names)

            if n_layers < 2:
                ax.text(0.5, 0.5, 'Insufficient\nLayers', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{model_name}: Layer Correlations')
                continue

            # Compute correlation matrix between all pairs of layers
            correlation_matrix = np.zeros((n_layers, n_layers))

            for i, layer1_name in enumerate(layer_names):
                for j, layer2_name in enumerate(layer_names):
                    if i != j:
                        acts1 = layer_acts[layer1_name]
                        acts2 = layer_acts[layer2_name]

                        # Compute average pairwise correlation
                        # For different dimensionalities, use mean activations
                        mean1 = np.mean(acts1, axis=1)  # Average across features
                        mean2 = np.mean(acts2, axis=1)  # Average across features

                        correlation = np.corrcoef(mean1, mean2)[0, 1]
                        if not np.isnan(correlation):
                            correlation_matrix[i, j] = correlation
                    else:
                        correlation_matrix[i, j] = 1.0

            im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f'{model_name}: Inter-Layer Correlations')
            ax.set_xticks(range(min(n_layers, 10)))  # Limit ticks for readability
            ax.set_yticks(range(min(n_layers, 10)))

            # Shortened layer names for display
            short_names = [name.split('_')[0][:8] for name in layer_names[:10]]
            ax.set_xticklabels(short_names, rotation=45)
            ax.set_yticklabels(short_names)

            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "layer_correlation_analysis",
            "information_flow"
        )

    def _plot_representational_similarity_analysis(
            self,
            model_layer_activations: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Plot Representational Similarity Analysis (RSA) between models and layers."""
        if len(model_layer_activations) < 2:
            return

        model_names = list(model_layer_activations.keys())

        # Compare same layers across different models
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Get common layer types across models
        common_layer_types = set()
        for model_acts in model_layer_activations.values():
            layer_types = [name.split('_')[0] for name in model_acts.keys()]
            common_layer_types.update(layer_types)

        common_layer_types = list(common_layer_types)[:4]  # Limit to 4 for visualization

        for idx, layer_type in enumerate(common_layer_types):
            if idx >= 4:
                break

            ax = axes[idx // 2, idx % 2]

            # Find layers of this type in each model
            model_layer_data = {}
            for model_name, layer_acts in model_layer_activations.items():
                for layer_name, activations in layer_acts.items():
                    if layer_name.startswith(layer_type):
                        model_layer_data[model_name] = activations
                        break

            if len(model_layer_data) < 2:
                ax.text(0.5, 0.5, f'Insufficient\n{layer_type}\nLayers',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{layer_type} Layer RSA')
                continue

            # Compute representational similarity matrix
            models_list = list(model_layer_data.keys())
            n_models = len(models_list)
            rsa_matrix = np.zeros((n_models, n_models))

            for i, model1 in enumerate(models_list):
                for j, model2 in enumerate(models_list):
                    if i != j:
                        # Compute representational similarity
                        acts1 = model_layer_data[model1]
                        acts2 = model_layer_data[model2]

                        # Use subset of features if dimensions differ
                        min_features = min(acts1.shape[1], acts2.shape[1])
                        acts1_sub = acts1[:, :min_features]
                        acts2_sub = acts2[:, :min_features]

                        # Compute correlation between flattened representations
                        flat1 = acts1_sub.flatten()
                        flat2 = acts2_sub.flatten()

                        similarity = np.corrcoef(flat1, flat2)[0, 1]
                        if not np.isnan(similarity):
                            rsa_matrix[i, j] = similarity
                    else:
                        rsa_matrix[i, j] = 1.0

            im = ax.imshow(rsa_matrix, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'{layer_type} Layer: Model Similarity')
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(models_list)
            ax.set_yticklabels(models_list)

            # Add correlation values
            for i in range(n_models):
                for j in range(n_models):
                    ax.text(j, i, f'{rsa_matrix[i, j]:.3f}',
                           ha='center', va='center',
                           color='white' if rsa_matrix[i, j] < 0.5 else 'black',
                           fontweight='bold')

            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "representational_similarity_analysis",
            "information_flow"
        )

    def _plot_information_bottleneck_analysis(
            self,
            model_layer_activations: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Plot information content analysis across layers."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Information content using singular value decomposition
        ax1 = axes[0, 0]
        for model_name, layer_acts in model_layer_activations.items():
            layer_names = list(layer_acts.keys())
            info_contents = []

            for layer_name, activations in layer_acts.items():
                # Use SVD to estimate information content
                try:
                    U, s, Vt = np.linalg.svd(activations, full_matrices=False)
                    # Information content as sum of normalized singular values
                    normalized_s = s / np.sum(s)
                    info_content = -np.sum(normalized_s * np.log(normalized_s + 1e-10))
                    info_contents.append(info_content)
                except:
                    info_contents.append(0)

            ax1.plot(range(len(layer_names)), info_contents, 'o-',
                    label=model_name, linewidth=2)

        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Information Content (SVD Entropy)')
        ax1.set_title('Information Content Across Layers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Compression ratio (how much information is compressed)
        ax2 = axes[0, 1]
        for model_name, layer_acts in model_layer_activations.items():
            layer_names = list(layer_acts.keys())
            compression_ratios = []

            first_layer_size = None
            for i, (layer_name, activations) in enumerate(layer_acts.items()):
                current_size = np.prod(activations.shape[1:])  # Exclude batch dimension

                if first_layer_size is None:
                    first_layer_size = current_size
                    compression_ratios.append(1.0)
                else:
                    ratio = current_size / first_layer_size
                    compression_ratios.append(ratio)

            ax2.plot(range(len(layer_names)), compression_ratios, 's-',
                    label=model_name, linewidth=2)

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Compression Ratio')
        ax2.set_title('Information Compression Across Layers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Mutual information estimation (simplified)
        ax3 = axes[1, 0]
        for model_name, layer_acts in model_layer_activations.items():
            layer_names = list(layer_acts.keys())
            mutual_infos = []

            layers_list = list(layer_acts.items())
            for i in range(len(layers_list) - 1):
                try:
                    # Simplified mutual information using correlation
                    acts1 = layers_list[i][1]
                    acts2 = layers_list[i + 1][1]

                    # Use mean activations for MI estimation
                    mean1 = np.mean(acts1, axis=1)
                    mean2 = np.mean(acts2, axis=1)

                    # Estimate MI using correlation (simplified)
                    correlation = np.corrcoef(mean1, mean2)[0, 1]
                    if not np.isnan(correlation):
                        mi_estimate = -0.5 * np.log(1 - correlation**2 + 1e-10)
                        mutual_infos.append(mi_estimate)
                    else:
                        mutual_infos.append(0)
                except:
                    mutual_infos.append(0)

            if mutual_infos:
                ax3.plot(range(len(mutual_infos)), mutual_infos, '^-',
                        label=model_name, linewidth=2)

        ax3.set_xlabel('Layer Transition Index')
        ax3.set_ylabel('Estimated Mutual Information')
        ax3.set_title('Information Flow Between Adjacent Layers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Layer complexity (using rank estimation)
        ax4 = axes[1, 1]
        for model_name, layer_acts in model_layer_activations.items():
            layer_names = list(layer_acts.keys())
            complexities = []

            for layer_name, activations in layer_acts.items():
                try:
                    # Estimate effective rank as measure of complexity
                    U, s, Vt = np.linalg.svd(activations, full_matrices=False)
                    # Effective rank using entropy of singular values
                    normalized_s = s / (np.sum(s) + 1e-10)
                    effective_rank = np.exp(-np.sum(normalized_s * np.log(normalized_s + 1e-10)))
                    complexities.append(effective_rank)
                except:
                    complexities.append(0)

            ax4.plot(range(len(layer_names)), complexities, 'd-',
                    label=model_name, linewidth=2)

        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('Effective Rank (Layer Complexity)')
        ax4.set_title('Layer Representational Complexity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "information_bottleneck_analysis",
            "information_flow"
        )

    def _plot_layer_gradient_flow_analysis(self, x_sample: np.ndarray) -> None:
        """Plot gradient flow analysis through layers."""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        # Use smaller sample for gradient computation
        n_grad_samples = min(50, len(x_sample))
        grad_indices = np.random.choice(len(x_sample), n_grad_samples, replace=False)
        x_grad = x_sample[grad_indices]

        for model_idx, (model_name, model) in enumerate(self.models.items()):
            # Gradient magnitude analysis
            ax1 = axes[0, model_idx]
            ax2 = axes[1, model_idx]

            try:
                # Create a simplified gradient computation
                with keras.utils.custom_object_scope({'RMSNorm': lambda: None, 'LogitNorm': lambda: None}):
                    # Get trainable layers
                    trainable_layers = [layer for layer in model.layers if layer.trainable_weights]

                    if not trainable_layers:
                        ax1.text(0.5, 0.5, 'No Trainable\nLayers', ha='center', va='center',
                               transform=ax1.transAxes)
                        ax2.text(0.5, 0.5, 'No Trainable\nLayers', ha='center', va='center',
                               transform=ax2.transAxes)
                        continue

                    # Compute gradients for a few samples
                    layer_grad_norms = []
                    layer_names = []

                    # Use tape to get gradients
                    x_tensor = keras.ops.convert_to_tensor(x_grad[:10])  # Use small batch

                    with keras.utils.tape.GradientTape() as tape:
                        tape.watch(x_tensor)
                        predictions = model(x_tensor, training=False)
                        # Use mean prediction as scalar loss
                        loss = keras.ops.mean(predictions)

                    # Get gradients w.r.t. all trainable weights
                    all_weights = []
                    for layer in trainable_layers:
                        all_weights.extend(layer.trainable_weights)

                    if all_weights:
                        gradients = tape.gradient(loss, all_weights)

                        # Process gradients by layer
                        weight_idx = 0
                        for layer in trainable_layers:
                            layer_grads = []
                            for _ in layer.trainable_weights:
                                if weight_idx < len(gradients) and gradients[weight_idx] is not None:
                                    grad_norm = keras.ops.norm(gradients[weight_idx])
                                    layer_grads.append(float(grad_norm))
                                weight_idx += 1

                            if layer_grads:
                                avg_grad_norm = np.mean(layer_grads)
                                layer_grad_norms.append(avg_grad_norm)
                                layer_names.append(f"{layer.__class__.__name__}_{len(layer_names)}")

                    # Plot gradient magnitudes
                    if layer_grad_norms:
                        ax1.bar(range(len(layer_grad_norms)), layer_grad_norms, alpha=0.7)
                        ax1.set_title(f'{model_name}: Gradient Magnitudes')
                        ax1.set_xlabel('Layer Index')
                        ax1.set_ylabel('Gradient Norm')
                        ax1.set_xticks(range(len(layer_names)))
                        ax1.set_xticklabels([name.split('_')[0][:8] for name in layer_names], rotation=45)

                        # Plot gradient flow (normalized)
                        if len(layer_grad_norms) > 1:
                            normalized_grads = np.array(layer_grad_norms) / (np.max(layer_grad_norms) + 1e-10)
                            ax2.plot(range(len(normalized_grads)), normalized_grads, 'o-', linewidth=2)
                            ax2.set_title(f'{model_name}: Normalized Gradient Flow')
                            ax2.set_xlabel('Layer Index')
                            ax2.set_ylabel('Normalized Gradient')
                            ax2.grid(True, alpha=0.3)
                    else:
                        ax1.text(0.5, 0.5, 'No Gradients\nComputed', ha='center', va='center',
                               transform=ax1.transAxes)
                        ax2.text(0.5, 0.5, 'No Gradients\nComputed', ha='center', va='center',
                               transform=ax2.transAxes)

            except Exception as e:
                # Fallback: show layer weight statistics instead
                ax1.text(0.5, 0.5, f'Gradient Error:\n{str(e)[:50]}...', ha='center', va='center',
                       transform=ax1.transAxes, fontsize=8)

                # Plot layer weight magnitudes as proxy
                weight_norms = []
                layer_names = []
                for layer in model.layers:
                    if layer.trainable_weights:
                        layer_weight_norms = []
                        for weight in layer.trainable_weights:
                            weight_norm = np.linalg.norm(weight.numpy())
                            layer_weight_norms.append(weight_norm)

                        if layer_weight_norms:
                            avg_weight_norm = np.mean(layer_weight_norms)
                            weight_norms.append(avg_weight_norm)
                            layer_names.append(f"{layer.__class__.__name__}_{len(layer_names)}")

                if weight_norms:
                    ax2.bar(range(len(weight_norms)), weight_norms, alpha=0.7, color='orange')
                    ax2.set_title(f'{model_name}: Weight Magnitudes (Proxy)')
                    ax2.set_xlabel('Layer Index')
                    ax2.set_ylabel('Weight Norm')
                    ax2.set_xticks(range(len(layer_names)))
                    ax2.set_xticklabels([name.split('_')[0][:8] for name in layer_names], rotation=45)

        plt.tight_layout()
        self.vis_manager.save_figure(
            fig,
            "gradient_flow_analysis",
            "information_flow"
        )

    def create_comprehensive_visualization(
            self,
            data: MNISTData,
            sample_digits: Optional[List[int]] = None,
            max_samples_per_digit: int = 3
    ) -> None:
        """Create comprehensive visualization for all digits (works with any model architecture).

        Args:
            data: Dataset splits
            sample_digits: Optional list of digits to analyze
            max_samples_per_digit: Maximum number of samples per digit
        """
        if sample_digits is None:
            sample_digits = list(range(10))

        n_models = len(self.models)
        n_samples = max_samples_per_digit
        n_digits = len(sample_digits)

        # Check if we have meaningful activation models
        has_activations = any(model_name in self.activation_models
                            for model_name in self.models.keys())

        if has_activations:
            # Create visualization with activation maps
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

                        # Get layer activation and predictions
                        layer_activation, predictions = self.get_activation_and_predictions(
                            data.x_test[idx],
                            model_name
                        )

                        # Plot layer activation (could be conv activation or dense layer visualization)
                        if layer_activation is not None and layer_activation.size > 0:
                            im = ax.imshow(layer_activation, cmap='viridis')
                            ax.axis('off')

                            # Add prediction confidence as title
                            confidence = predictions[digit]
                            if digit_idx == 0:
                                ax.set_title(f"Layer Act\n{confidence:.2f}")
                            else:
                                ax.set_title(f"{confidence:.2f}")
                        else:
                            # No meaningful activation to show
                            ax.text(0.5, 0.5, f'{predictions[digit]:.3f}',
                                   ha='center', va='center', transform=ax.transAxes,
                                   fontsize=16, fontweight='bold')
                            ax.set_title(f"Conf: {predictions[digit]:.3f}")
                            ax.axis('off')

                # Add digit label on the left
                ax_prob.set_ylabel(f"Digit {digit}")

        else:
            # Simplified visualization without activation maps - just probability bars
            fig, axes = plt.subplots(n_digits, n_models, figsize=(4*n_models, 2*n_digits))
            if n_models == 1:
                axes = axes.reshape(-1, 1)

            for digit_idx, digit in enumerate(sample_digits):
                digit_indices = np.where(np.argmax(data.y_test, axis=1) == digit)[0]
                sample_indices = digit_indices[:max_samples_per_digit]

                for model_idx, (model_name, model) in enumerate(self.models.items()):
                    ax = axes[digit_idx, model_idx]

                    # Get predictions for samples of this digit
                    all_predictions = []
                    confidences = []

                    for idx in sample_indices:
                        sample_batch = np.expand_dims(data.x_test[idx], 0)
                        preds = model.predict(sample_batch, verbose=0)[0]
                        all_predictions.append(preds)
                        confidences.append(preds[digit])

                    if all_predictions:
                        avg_predictions = np.mean(all_predictions, axis=0)
                        avg_confidence = np.mean(confidences)

                        # Plot probability distribution
                        bars = ax.bar(range(10), avg_predictions, alpha=0.7)
                        bars[digit].set_color('red')  # Highlight true class

                        ax.set_ylim(0, 1)
                        ax.set_title(f'{model_name}\nAvg Conf: {avg_confidence:.3f}')
                        ax.set_xticks(range(10))

                        if digit_idx == n_digits - 1:
                            ax.set_xlabel('Predicted Class')
                        if model_idx == 0:
                            ax.set_ylabel(f'Digit {digit}\nProbability')

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
        print("📈 Evaluating model performance...")
        for name, model in tqdm(self.models.items(), desc="Model Evaluation"):
            evaluation = model.evaluate(data.x_test, data.y_test, verbose=0)
            results[name] = dict(zip(model.metrics_names, evaluation))

        # Create comprehensive visualization
        print("🖼️ Creating activation map visualizations...")
        self.create_comprehensive_visualization(
            data,
            sample_digits,
            max_samples_per_digit
        )

        # Create probability distribution analysis
        print("🧠 Starting comprehensive probability distribution analysis...")
        self.create_probability_distribution_analysis(data)

        print("✅ Model analysis completed! All visualizations saved.")
        return results