"""
Information Flow Analysis Module
============================================================================

Analyzes how information flows through the network layers.
"""

import keras
import numpy as np
from typing import Dict, Any, Optional, List

from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from ..config import AnalysisConfig  # Added import for proper type hint
from dl_techniques.utils.logger import logger


class InformationFlowAnalyzer(BaseAnalyzer):
    """Analyzes information flow through network layers."""

    def __init__(self, models: Dict[str, keras.Model], config: AnalysisConfig):  # FIXED: changed Any to AnalysisConfig
        """Initialize analyzer and setup extraction models."""
        super().__init__(models, config)
        self.layer_extraction_models = {}
        self._setup_activation_models()

    def requires_data(self) -> bool:
        """Information flow analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze information flow through network, including activation patterns."""
        logger.info("Analyzing information flow and activations...")

        if data is None:
            raise ValueError("Data is required for information flow analysis")

        # Get sample data
        x_sample = data.x_data[:min(200, len(data.x_data))]

        for model_name, extraction_data in self.layer_extraction_models.items():
            if extraction_data is None:
                # This check now correctly handles unsupported models that were skipped
                logger.warning(f"Skipping information flow for {model_name} as no extraction model could be built.")
                continue

            extraction_model = extraction_data['model']
            layer_info = extraction_data['layer_info']

            try:
                # Get multi-layer outputs
                layer_outputs = extraction_model.predict(x_sample, verbose=0)

                # Handle single output case
                if not isinstance(layer_outputs, list):
                    layer_outputs = [layer_outputs]

                # Analyze each layer
                layer_analysis = {}
                for i, (output, info) in enumerate(zip(layer_outputs, layer_info)):
                    analysis = self._analyze_layer_information(output, info)
                    layer_analysis[info['name']] = analysis

                results.information_flow[model_name] = layer_analysis

                # Store detailed activation stats for key layers
                self._analyze_key_layer_activations(model_name, layer_outputs, layer_info, results)
            except Exception as e:
                logger.error(f"Failed to analyze information flow for {model_name}: {e}")
                continue

    def _setup_activation_models(self) -> None:
        """Set up models for extracting intermediate activations."""
        for model_name, model in self.models.items():
            # First, check if the model is a Functional model with a defined graph.
            # Subclassed models will not have a valid `.input` attribute before being called.
            if not hasattr(model, 'input') or not hasattr(model, 'layers'):
                logger.warning(
                    f"Model '{model_name}' is likely a subclassed model or not a standard Keras Functional model. "
                    "Automatic information flow analysis is not supported. Skipping."
                )
                self.layer_extraction_models[model_name] = None
                continue

            try:
                # Get suitable layers for extraction.
                extraction_layers = self._get_extraction_layers(model)

                if not extraction_layers:
                    logger.warning(f"No suitable layers found for extraction in {model_name}")
                    self.layer_extraction_models[model_name] = None
                    continue

                # Create a new Keras model for efficient multi-layer activation extraction.
                # This is the robust, recommended approach for Functional models.
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

    def _get_extraction_layers(self, model: keras.Model) -> List[keras.layers.Layer]:
        """
        Get a list of layer objects suitable for information flow analysis.
        Returns layer objects themselves, not dictionaries.
        """
        extraction_layers = []
        for layer in model.layers:
            # Skip input layers and other non-op layers
            if isinstance(layer, keras.layers.InputLayer):
                continue

            # Check for common layer types that are suitable for analysis.
            # This ensures we only analyze layers with meaningful activations.
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                  keras.layers.BatchNormalization, keras.layers.LayerNormalization,
                                  keras.layers.ReLU, keras.layers.PReLU, keras.layers.ELU,
                                  keras.layers.GlobalAveragePooling2D, keras.layers.GlobalMaxPooling2D)):
                # Ensure the layer has a valid output tensor in the model's graph
                if hasattr(layer, 'output'):
                    extraction_layers.append(layer)

        return extraction_layers

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """Analysis of layer information content."""
        # This method remains largely the same, but is now more reliable due to
        # the robust activation extraction.

        # Add a check for empty or invalid outputs
        if output is None or output.size == 0:
            return {'error': 'Invalid or empty activation output'}

        # Flatten spatial dimensions if needed
        if len(output.shape) > 2:  # Conv layer or other spatial layer
            # Handle cases with no batch dimension from predict
            if len(output.shape) == 4: # (batch, H, W, C)
                output_flat = np.mean(output, axis=(1, 2))
            elif len(output.shape) == 3: # (H, W, C) - single sample prediction
                output_flat = np.mean(output, axis=(0, 1))
                output_flat = np.expand_dims(output_flat, axis=0) # re-add batch dim
            else: # Fallback
                output_flat = output.reshape(output.shape[0], -1)
            spatial_output = output
        else:
            output_flat = output
            spatial_output = None

        # Compute statistics
        analysis = {
            'layer_type': layer_info['type'],
            'output_shape': output.shape,
            'mean_activation': float(np.mean(output_flat)),
            'std_activation': float(np.std(output_flat)),
            'sparsity': float(np.mean(np.abs(output_flat) < 1e-5)),
            'positive_ratio': float(np.mean(output_flat > 0)),
        }

        # Add spatial statistics for conv layers
        if spatial_output is not None and len(spatial_output.shape) == 4:
            analysis['spatial_mean'] = float(np.mean(spatial_output))
            analysis['spatial_std'] = float(np.std(spatial_output))
            analysis['channel_variance'] = float(np.var(np.mean(spatial_output, axis=(0, 1, 2))))

        # Compute effective rank
        if output_flat.ndim == 2 and output_flat.shape[0] > 1 and output_flat.shape[1] > 1:
            try:
                # Ensure matrix is centered for SVD
                centered_output = output_flat - np.mean(output_flat, axis=0)
                s = np.linalg.svd(centered_output, compute_uv=False)

                # Prevent log(0) and division by zero
                s_sum = np.sum(s)
                if s_sum > 1e-9:
                    s_normalized = s / s_sum
                    # Add epsilon to prevent log(0) for zero-valued singular values
                    effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-9)))
                    analysis['effective_rank'] = float(effective_rank)
                else:
                    analysis['effective_rank'] = 0.0
            except np.linalg.LinAlgError:
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
            key_indices.append(conv_indices[-1])  # Last conv layer
        if dense_indices:
            # Take the one before the final output layer if available, else the first one
            key_indices.append(dense_indices[-2] if len(dense_indices) > 1 else dense_indices[-1])

        results.activation_stats[model_name] = {}

        for idx in set(key_indices): # Use set to avoid duplicates
            if idx < len(layer_outputs):
                layer_name = layer_info[idx]['name']
                activations = layer_outputs[idx]

                # Ensure activations is a numpy array
                if not isinstance(activations, np.ndarray):
                    continue

                flat_acts = activations.flatten()
                if flat_acts.size == 0:
                    continue

                results.activation_stats[model_name][layer_name] = {
                    'shape': activations.shape,
                    'mean': float(np.mean(flat_acts)),
                    'std': float(np.std(flat_acts)),
                    'sparsity': float(np.mean(np.abs(flat_acts) < 1e-5)),
                    'positive_ratio': float(np.mean(flat_acts > 0)),
                    'percentiles': {
                        'p25': float(np.percentile(flat_acts, 25)),
                        'p50': float(np.percentile(flat_acts, 50)),
                        'p75': float(np.percentile(flat_acts, 75))
                    },
                    # Only store samples for conv layers to save space
                    'sample_activations': activations[:min(10, activations.shape[0])] if len(activations.shape) == 4 else None
                }