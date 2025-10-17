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
    aggressively, potentially discarding useful information.

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

    def requires_data(self) -> bool:
        """Information flow analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze information flow using forward hooks for subclassed model compatibility."""
        logger.info("Analyzing information flow and activations...")

        if data is None:
            raise ValueError("Data is required for information flow analysis")

        x_sample = data.x_data
        # Correctly handle dictionary vs. numpy array for batch size
        if isinstance(x_sample, dict):
            # Take the first key to determine batch size
            first_key = next(iter(x_sample))
            self._batch_size = len(x_sample[first_key])
            # Subsample the dictionary
            x_sample = {k: v[:min(200, self._batch_size)] for k, v in x_sample.items()}
            self._batch_size = len(x_sample[first_key])
        else:
            self._batch_size = len(x_sample)
            x_sample = x_sample[:min(200, self._batch_size)]


        for model_name, model in self.models.items():
            if not model.built:
                try:
                    model.predict(x_sample, verbose=0)
                    logger.info(f"Built model '{model_name}' for analysis.")
                except Exception as e:
                    logger.error(
                        f"Failed to build model '{model_name}'. Skipping information flow analysis. Error: {e}")
                    continue

            captured_outputs = {}
            hook_handles = []

            try:
                all_layers = recursively_get_layers(model)
                extraction_layers = self._get_extraction_layers(all_layers)

                if not extraction_layers:
                    logger.warning(f"No suitable layers for information flow analysis in '{model_name}'.")
                    continue

                def forward_hook(layer, inputs, outputs):
                    captured_outputs[layer.name] = keras.ops.convert_to_numpy(outputs)

                for layer in extraction_layers:
                    handle = layer.register_forward_hook(forward_hook)
                    hook_handles.append(handle)

                model.predict(x_sample, verbose=0)

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

                layer_outputs_list = [captured_outputs.get(l.name) for l in extraction_layers]
                layer_info_list = [{'name': l.name, 'type': l.__class__.__name__} for l in extraction_layers]
                self._analyze_key_layer_activations(model_name, layer_outputs_list, layer_info_list, results)

            except Exception as e:
                logger.error(f"Failed to analyze information flow for {model_name}: {e}", exc_info=True)
            finally:
                for handle in hook_handles:
                    handle.remove()
                logger.debug(f"Removed {len(hook_handles)} hooks from model '{model_name}'.")

    # The rest of the file remains unchanged as it contains the correct logic...
    # (The _setup_activation_models method is now dead code, but we leave it for reference)

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
                extraction_layers = self._get_extraction_layers(model.layers) # Corrected call

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
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                  keras.layers.BatchNormalization, keras.layers.LayerNormalization,
                                  keras.layers.ReLU, keras.layers.PReLU, keras.layers.ELU,
                                  keras.layers.GlobalAveragePooling2D, keras.layers.GlobalMaxPooling2D)):
                extraction_layers.append(layer)
        return extraction_layers

    def _safely_flatten_activations(self, output: np.ndarray) -> tuple:
        """Safely flatten activation tensors using known batch size."""
        if output is None or output.size == 0: return None, None
        original_shape = output.shape
        if len(original_shape) == 2: return output, None
        elif len(original_shape) == 1: return output.reshape(1, -1), None
        elif len(original_shape) >= 3:
            expected_batch_size = getattr(self, '_batch_size', None)
            if expected_batch_size is not None and original_shape[0] == expected_batch_size:
                flattened = output.reshape(original_shape[0], -1)
                return flattened, output
            else:
                flattened = output.reshape(1, -1)
                return flattened, output.reshape(1, *original_shape)
        return output.reshape(1, 1), None

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """Analysis of layer information content with improved effective rank calculation."""
        output_flat, spatial_output = self._safely_flatten_activations(output)
        if output_flat is None: return {'error': 'Invalid or empty activation output'}

        analysis = {
            'layer_type': layer_info['type'], 'output_shape': output.shape,
            'mean_activation': float(np.mean(output_flat)), 'std_activation': float(np.std(output_flat)),
            'sparsity': float(np.mean(np.abs(output_flat) < 1e-5)),
            'positive_ratio': float(np.mean(output_flat > 0)),
        }
        if (output_flat.ndim == 2 and min(output_flat.shape) > 1):
            try:
                centered_output = output_flat - np.mean(output_flat, axis=0)
                s = np.linalg.svd(centered_output, compute_uv=False)
                s_sum = np.sum(s)
                if s_sum > 1e-9:
                    s_normalized = s / s_sum
                    effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-9)))
                    analysis['effective_rank'] = float(effective_rank)
                else:
                    analysis['effective_rank'] = 0.0
            except np.linalg.LinAlgError:
                analysis['effective_rank'] = 0.0
        else:
            analysis['effective_rank'] = 0.0
        return analysis

    def _analyze_key_layer_activations(self, model_name: str, layer_outputs: List[np.ndarray],
                                     layer_info: List[Dict], results: AnalysisResults) -> None:
        """Analyze activations for key layers in detail."""
        conv_indices = [i for i, info in enumerate(layer_info) if 'Conv' in info['type']]
        dense_indices = [i for i, info in enumerate(layer_info) if 'Dense' in info['type']]
        key_indices = []
        if conv_indices: key_indices.append(conv_indices[-1])
        if dense_indices:
            candidate_dense_layers = dense_indices[:-1] if len(dense_indices) > 1 else dense_indices
            if candidate_dense_layers:
                key_indices.append(candidate_dense_layers[len(candidate_dense_layers) // 2])
        results.activation_stats[model_name] = {}
        for idx in set(key_indices):
            if idx < len(layer_outputs) and isinstance(layer_outputs[idx], np.ndarray):
                layer_name, activations = layer_info[idx]['name'], layer_outputs[idx]
                flat_acts, spatial_acts = self._safely_flatten_activations(activations)
                if flat_acts is None or flat_acts.size == 0: continue
                flat_acts_1d = flat_acts.flatten()
                results.activation_stats[model_name][layer_name] = {
                    'shape': activations.shape, 'mean': float(np.mean(flat_acts_1d)),
                    'std': float(np.std(flat_acts_1d)), 'sparsity': float(np.mean(np.abs(flat_acts_1d) < 1e-5)),
                    'positive_ratio': float(np.mean(flat_acts_1d > 0)),
                    'percentiles': {'p25': float(np.percentile(flat_acts_1d, 25)), 'p50': float(np.percentile(flat_acts_1d, 50)), 'p75': float(np.percentile(flat_acts_1d, 75))},
                    'sample_activations': (spatial_acts[:min(10, spatial_acts.shape[0])] if spatial_acts is not None and len(spatial_acts.shape) >= 4 else None)
                }