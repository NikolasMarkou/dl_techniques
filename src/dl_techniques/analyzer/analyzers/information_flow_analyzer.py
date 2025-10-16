"""
Analyze the flow of information and feature dimensionality through layers.

This analyzer diagnoses the health of a neural network by examining the
activations of its intermediate layers. It is designed to detect common
pathologies such as vanishing or exploding gradients, dead neurons, and
information bottlenecks, which can impede a model's ability to learn.

Architecture
------------
The analysis is performed by constructing a temporary Keras "feature
extractor" model. This new model shares the same input as the original model
but is reconfigured to output a list of activation tensors from each
intermediate layer. By performing a single forward pass on a batch of data,
this architecture efficiently captures a snapshot of the entire network's
internal state. The analyzer then processes this sequence of activation
tensors to quantify how information is transformed at each stage of the
network.

Foundational Mathematics
------------------------
The analysis relies on two primary sets of metrics computed from the
activation matrix `A` of each layer, where rows correspond to samples in a
batch and columns correspond to features (neurons):

1.  **Activation Statistics**: Basic statistical moments are used to diagnose
    the health of the signal propagation.
    -   The `mean` and `standard deviation` of activations are tracked across
        layers. Values consistently close to zero suggest a vanishing signal
        (gradient), while very large values suggest an exploding signal.
    -   `Sparsity`, the fraction of near-zero activations, is particularly
        useful for diagnosing "dead neurons," a common issue with ReLU-based
        activations where a neuron ceases to output non-zero values.

2.  **Effective Rank**: This metric quantifies the dimensionality of the
    feature space learned by a layer, providing a more robust measure than
    the classical matrix rank. It is designed to identify "information
    bottlenecks," where a layer compresses the feature representation too
    aggressively, potentially discarding useful information. The calculation
    proceeds as follows:
    -   First, Singular Value Decomposition (SVD) is performed on the
        centered activation matrix `A` to obtain its singular values `{σ_i}`.
    -   These singular values are normalized to form a probability
        distribution, `p_i = σ_i / Σσ_j`. Each `p_i` represents the
        proportion of variance captured by the i-th principal component.
    -   The Shannon entropy of this distribution is calculated:
        `H = -Σ p_i * log(p_i)`.
    -   The effective rank is `exp(H)`.

    Intuitively, `exp(H)` measures the number of "significant" dimensions in
    the activation space. If all singular values are equal (maximum entropy),
    the effective rank equals the true rank. If a few singular values
    dominate (low entropy), the effective rank is low. A sharp drop in
    effective rank between layers is a strong indicator of an information
    bottleneck.

References
----------
1.  Tishby, N., & Zaslavsky, N. (2015). "Deep learning and the information
    bottleneck principle." ITW.
2.  Aghajanian, S., et al. (2020). "Characterizing signal propagation to
    close the performance gap between Caffe and PyTorch." SysML.
3.  Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of
    effective dimensionality." LATS.

"""

import keras
import numpy as np
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseAnalyzer
from ..config import AnalysisConfig
from ..utils import recursively_get_layers
from ..data_types import AnalysisResults, DataInput
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class InformationFlowAnalyzer(BaseAnalyzer):
    """Analyzes information flow through network layers."""

    def __init__(self, models: Dict[str, keras.Model], config: AnalysisConfig):
        """Initialize analyzer and setup extraction models."""
        super().__init__(models, config)
        self.layer_extraction_models = {}
        self._setup_activation_models()

    def requires_data(self) -> bool:
        """Information flow analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze information flow using forward hooks for subclassed model compatibility."""
        logger.info("Analyzing information flow and activations...")

        if data is None:
            raise ValueError("Data is required for information flow analysis")

        x_sample = data.x_data[:min(200, len(data.x_data))]
        self._batch_size = len(x_sample)

        for model_name, model in self.models.items():
            # This check is important for models that haven't been built.
            if not model.built:
                try:
                    # Attempt to build the model with sample data.
                    model(x_sample)
                    logger.info(f"Built subclassed model '{model_name}' for analysis.")
                except Exception as e:
                    logger.error(
                        f"Failed to build model '{model_name}'. Skipping information flow analysis. Error: {e}")
                    continue

            # This dictionary will store the captured activations.
            captured_outputs = {}
            # This list will store the hook handles so we can remove them later.
            hook_handles = []

            try:
                # 1. Get all layers recursively and identify which ones to analyze.
                all_layers = recursively_get_layers(model)
                extraction_layers = self._get_extraction_layers(all_layers)

                if not extraction_layers:
                    logger.warning(f"No suitable layers for information flow analysis in '{model_name}'.")
                    continue

                # 2. Define the hook function that will be attached to each layer.
                def forward_hook(layer, inputs, outputs):
                    # Keras hooks can pass tensors or NumPy arrays. Convert to NumPy.
                    # We use a unique layer name to handle multiple layers with the same class name.
                    captured_outputs[layer.name] = keras.ops.convert_to_numpy(outputs)

                # 3. Register the hook on each target layer.
                for layer in extraction_layers:
                    handle = layer.register_forward_hook(forward_hook)
                    hook_handles.append(handle)

                # 4. Run a forward pass to trigger the hooks.
                model.predict(x_sample, verbose=0)

                # 5. Process the captured activations.
                layer_analysis = {}
                for i, layer in enumerate(extraction_layers):
                    if layer.name in captured_outputs:
                        output_tensor = captured_outputs[layer.name]
                        layer_info = {'name': layer.name, 'type': layer.__class__.__name__}
                        analysis = self._analyze_layer_information(output_tensor, layer_info)
                        layer_analysis[layer.name] = analysis
                    else:
                        logger.debug(f"No activation captured for layer '{layer.name}' in model '{model_name}'.")

                results.information_flow[model_name] = layer_analysis

                # This part can be adapted or simplified if needed. We pass the captured
                # activations and layer info to the detailed analysis function.
                layer_outputs_list = [captured_outputs.get(l.name) for l in extraction_layers]
                layer_info_list = [{'name': l.name, 'type': l.__class__.__name__} for l in extraction_layers]
                self._analyze_key_layer_activations(model_name, layer_outputs_list, layer_info_list, results)

            except Exception as e:
                logger.error(f"Failed to analyze information flow for {model_name}: {e}")
            finally:
                # 6. CRITICAL: Remove all hooks to restore the model to its original state.
                for handle in hook_handles:
                    handle.remove()
                logger.debug(f"Removed {len(hook_handles)} hooks from model '{model_name}'.")

    def _setup_activation_models(self) -> None:
        """Set up models for extracting intermediate activations."""
        for model_name, model in self.models.items():
            if not hasattr(model, 'input') or not hasattr(model, 'layers'):
                logger.warning(
                    f"Model '{model_name}' is likely a subclassed model or not a standard Keras Functional model. "
                    "Automatic information flow analysis is not supported. Skipping."
                )
                self.layer_extraction_models[model_name] = None
                continue

            try:
                extraction_layers = self._get_extraction_layers(model)

                if not extraction_layers:
                    logger.warning(f"No suitable layers found for extraction in {model_name}")
                    self.layer_extraction_models[model_name] = None
                    continue

                layer_outputs = [layer.output for layer in extraction_layers]
                layer_info = [{'name': layer.name, 'type': layer.__class__.__name__} for layer in extraction_layers]

                extraction_model = keras.Model(
                    inputs=model.input,
                    outputs=layer_outputs,
                    name=f"{model_name}_activation_extractor"
                )

                self.layer_extraction_models[model_name] = {
                    'model': extraction_model,
                    'layer_info': layer_info
                }
                logger.info(f"Successfully created activation extraction model for '{model_name}'.")

            except Exception as e:
                logger.error(f"Failed to create activation extraction model for '{model_name}': {e}. "
                             f"This can happen with complex, non-standard model architectures. Skipping.")
                self.layer_extraction_models[model_name] = None

    def _get_extraction_layers(self, layers: List[keras.layers.Layer]) -> List[keras.layers.Layer]:
        """Get a list of layer objects suitable for information flow analysis from a flat list."""
        extraction_layers = []
        for layer in layers:
            # We don't need to check for InputLayer as it won't be in the recursive list.
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                  keras.layers.BatchNormalization, keras.layers.LayerNormalization,
                                  keras.layers.ReLU, keras.layers.PReLU, keras.layers.ELU,
                                  keras.layers.GlobalAveragePooling2D, keras.layers.GlobalMaxPooling2D)):
                # We can analyze any layer that has an output.
                extraction_layers.append(layer)
        return extraction_layers

    def _safely_flatten_activations(self, output: np.ndarray) -> tuple:
        """
        Safely flatten activation tensors using known batch size instead of guessing.

        Args:
            output: Activation tensor from model prediction

        Returns:
            Tuple of (flattened_output, spatial_output_if_applicable)
        """
        if output is None or output.size == 0:
            return None, None

        original_shape = output.shape

        # Case 1: Already 2D (batch, features) - most common for dense layers
        if len(original_shape) == 2:
            return output, None

        # Case 2: 1D tensor (single feature value or flattened)
        elif len(original_shape) == 1:
            # Reshape to (1, features) to ensure batch dimension
            return output.reshape(1, -1), None

        # Case 3: Higher dimensional tensors (conv layers, etc.)
        elif len(original_shape) >= 3:
            expected_batch_size = getattr(self, '_batch_size', None)

            if expected_batch_size is not None and original_shape[0] == expected_batch_size:
                # First dimension matches expected batch size - likely (batch, ...)
                if len(original_shape) == 3:
                    # (batch, seq, features) or (batch, H*W, C)
                    flattened = output.reshape(original_shape[0], -1)
                    spatial_output = output
                elif len(original_shape) == 4:
                    # (batch, H, W, C) or (batch, C, H, W)
                    flattened = output.reshape(original_shape[0], -1)
                    spatial_output = output
                else:
                    # Higher dimensional - flatten all non-batch dimensions
                    flattened = output.reshape(original_shape[0], -1)
                    spatial_output = output
            else:
                if expected_batch_size is None:
                    logger.debug("No batch size available for tensor shape inference")

                # Conservative approach: treat as single sample if dimensions don't match expectations
                if len(original_shape) == 3:
                    # Could be (H, W, C) for single sample
                    flattened = output.reshape(1, -1)
                    spatial_output = output.reshape(1, *original_shape)
                elif len(original_shape) == 4:
                    # If first dim doesn't match batch size, might be (C, H, W, batch) or similar
                    # Try to find which dimension matches batch size
                    batch_dim = None
                    if expected_batch_size is not None:
                        for i, dim_size in enumerate(original_shape):
                            if dim_size == expected_batch_size:
                                batch_dim = i
                                break

                    if batch_dim is not None and batch_dim != 0:
                        # Move batch dimension to front
                        output_reordered = np.moveaxis(output, batch_dim, 0)
                        flattened = output_reordered.reshape(expected_batch_size, -1)
                        spatial_output = output_reordered
                    else:
                        # Fallback: treat as single sample
                        flattened = output.reshape(1, -1)
                        spatial_output = output.reshape(1, *original_shape)
                else:
                    # Very high dimensional - flatten everything, treat as single sample
                    flattened = output.reshape(1, -1)
                    spatial_output = None

            return flattened, spatial_output

        else:
            # 0D tensor (scalar) - shouldn't happen but handle gracefully
            return output.reshape(1, 1), None

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """
        Analysis of layer information content with improved effective rank calculation.
        """

        # Use safe flattening
        output_flat, spatial_output = self._safely_flatten_activations(output)

        if output_flat is None:
            return {'error': 'Invalid or empty activation output'}

        # Compute statistics on safely flattened output
        analysis = {
            'layer_type': layer_info['type'],
            'output_shape': output.shape,
            'mean_activation': float(np.mean(output_flat)),
            'std_activation': float(np.std(output_flat)),
            'sparsity': float(np.mean(np.abs(output_flat) < 1e-5)),
            'positive_ratio': float(np.mean(output_flat > 0)),
        }

        # Add spatial statistics for conv layers if we have spatial structure
        if spatial_output is not None and len(spatial_output.shape) >= 4:
            analysis['spatial_mean'] = float(np.mean(spatial_output))
            analysis['spatial_std'] = float(np.std(spatial_output))
            # Compute variance across channels (last dimension typically)
            if len(spatial_output.shape) == 4:  # (batch, H, W, C)
                channel_means = np.mean(spatial_output, axis=(0, 1, 2))
                analysis['channel_variance'] = float(np.var(channel_means))

        if (output_flat.ndim == 2 and min(output_flat.shape) > 1):
            # SVD is well-defined regardless of whether samples > features
            try:
                # Ensure matrix is centered for SVD
                centered_output = output_flat - np.mean(output_flat, axis=0)
                s = np.linalg.svd(centered_output, compute_uv=False)

                # Prevent log(0) and division by zero
                s_sum = np.sum(s)
                if s_sum > 1e-9:
                    s_normalized = s / s_sum
                    # Add epsilon to prevent log(0)
                    effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-9)))
                    analysis['effective_rank'] = float(effective_rank)
                else:
                    analysis['effective_rank'] = 0.0
            except np.linalg.LinAlgError:
                analysis['effective_rank'] = 0.0
        else:
            # Skip effective rank for inappropriate tensor shapes
            analysis['effective_rank'] = 0.0

        return analysis

    def _analyze_key_layer_activations(self, model_name: str, layer_outputs: List[np.ndarray],
                                     layer_info: List[Dict], results: AnalysisResults) -> None:
        """Analyze activations for key layers in detail."""
        # Find key layers (last conv and middle dense)
        conv_indices = [i for i, info in enumerate(layer_info) if 'Conv' in info['type']]
        dense_indices = [i for i, info in enumerate(layer_info) if 'Dense' in info['type']]

        key_indices = []
        if conv_indices:
            key_indices.append(conv_indices[-1])

        if dense_indices:
            # ==============================================================================
            # COMMENT: The original code arbitrarily chose the second-to-last dense
            # layer (`dense_indices[-2]`), which is not a meaningful heuristic. The
            # current fix uses the middle dense layer, which is better.
            #
            # A further improvement is to avoid the final dense layer (which is often
            # a simple linear projection for classification) and pick a layer
            # from the "body" of the dense block.
            # ==============================================================================

            # Exclude the last dense layer if there are multiple
            candidate_dense_layers = dense_indices[:-1] if len(dense_indices) > 1 else dense_indices

            if candidate_dense_layers:
                # Select the middle layer from the candidates
                middle_dense_layer_index = candidate_dense_layers[len(candidate_dense_layers) // 2]
                key_indices.append(middle_dense_layer_index)

        results.activation_stats[model_name] = {}

        for idx in set(key_indices):
            if idx < len(layer_outputs):
                layer_name = layer_info[idx]['name']
                activations = layer_outputs[idx]

                if not isinstance(activations, np.ndarray):
                    continue

                # Use safe flattening
                flat_acts, spatial_acts = self._safely_flatten_activations(activations)
                if flat_acts is None or flat_acts.size == 0:
                    continue

                flat_acts_1d = flat_acts.flatten()

                results.activation_stats[model_name][layer_name] = {
                    'shape': activations.shape,
                    'mean': float(np.mean(flat_acts_1d)),
                    'std': float(np.std(flat_acts_1d)),
                    'sparsity': float(np.mean(np.abs(flat_acts_1d) < 1e-5)),
                    'positive_ratio': float(np.mean(flat_acts_1d > 0)),
                    'percentiles': {
                        'p25': float(np.percentile(flat_acts_1d, 25)),
                        'p50': float(np.percentile(flat_acts_1d, 50)),
                        'p75': float(np.percentile(flat_acts_1d, 75))
                    },
                    # Only store samples for conv layers with spatial structure
                    'sample_activations': (spatial_acts[:min(10, spatial_acts.shape[0])]
                                         if spatial_acts is not None and len(spatial_acts.shape) >= 4
                                         else None)
                }

# ---------------------------------------------------------------------