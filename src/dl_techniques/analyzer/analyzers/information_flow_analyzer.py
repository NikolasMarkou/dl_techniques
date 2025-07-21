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
from ..config import AnalysisConfig
from dl_techniques.utils.logger import logger


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
        """Analyze information flow through network, including activation patterns."""
        logger.info("Analyzing information flow and activations...")

        if data is None:
            raise ValueError("Data is required for information flow analysis")

        # Get sample data
        x_sample = data.x_data[:min(200, len(data.x_data))]

        for model_name, extraction_data in self.layer_extraction_models.items():
            if extraction_data is None:
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

    def _get_extraction_layers(self, model: keras.Model) -> List[keras.layers.Layer]:
        """Get a list of layer objects suitable for information flow analysis."""
        extraction_layers = []
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue

            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                  keras.layers.BatchNormalization, keras.layers.LayerNormalization,
                                  keras.layers.ReLU, keras.layers.PReLU, keras.layers.ELU,
                                  keras.layers.GlobalAveragePooling2D, keras.layers.GlobalMaxPooling2D)):
                if hasattr(layer, 'output'):
                    extraction_layers.append(layer)

        return extraction_layers

    def _safely_flatten_activations(self, output: np.ndarray) -> tuple:
        """
        Safely flatten activation tensors without assuming batch dimension structure.

        FIXED: This addresses the fragile tensor reshaping identified in the code review.
        Instead of assuming output.shape[0] is the batch dimension, we detect the
        tensor structure more carefully.

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

        # Case 2: 1D tensor (single sample, single feature)
        elif len(original_shape) == 1:
            # Reshape to (1, features) to ensure batch dimension
            return output.reshape(1, -1), None

        # Case 3: Higher dimensional tensors (conv layers, etc.)
        elif len(original_shape) >= 3:
            # Try to detect if this has a batch dimension or not
            # Heuristic: if the tensor came from a batch prediction and first dim matches
            # the expected batch size, treat first dim as batch

            # Most conservative approach: compute statistics along appropriate axes
            # without assuming batch structure
            if len(original_shape) == 3:
                # Could be (batch, seq, features) or (H, W, C) for single sample
                # Check if first dimension is reasonable as batch size
                if original_shape[0] <= 512:  # Reasonable batch size
                    # Likely (batch, H*W, C) or (batch, seq, features)
                    flattened = output.reshape(original_shape[0], -1)
                    spatial_output = output if len(original_shape) == 3 else None
                else:
                    # Likely (H, W, C) for single sample
                    # Flatten spatial dimensions, add batch dim
                    flattened = output.reshape(1, -1)
                    spatial_output = output.reshape(1, *original_shape)

            elif len(original_shape) == 4:
                # Most likely (batch, H, W, C) or (batch, C, H, W)
                # Flatten spatial dimensions for each sample in batch
                flattened = output.reshape(original_shape[0], -1)
                spatial_output = output

            elif len(original_shape) == 5:
                # Could be (batch, D, H, W, C) for 3D conv or video
                # Flatten all spatial/temporal dimensions
                flattened = output.reshape(original_shape[0], -1)
                spatial_output = output

            else:
                # Very high dimensional - flatten everything except assume first is batch
                logger.debug(f"High-dimensional tensor with shape {original_shape}, "
                           f"assuming first dimension is batch")
                if original_shape[0] <= 1024:  # Reasonable batch size limit
                    flattened = output.reshape(original_shape[0], -1)
                else:
                    # Treat as single sample with no batch dimension
                    flattened = output.reshape(1, -1)
                spatial_output = None

            return flattened, spatial_output

        else:
            # 0D tensor (scalar) - shouldn't happen but handle gracefully
            return output.reshape(1, 1), None

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """Analysis of layer information content with safe tensor handling."""

        # FIXED: Use safe flattening instead of fragile reshape
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

        # Compute effective rank for 2D flattened output
        if (output_flat.ndim == 2 and output_flat.shape[0] > 1 and
            output_flat.shape[1] > 1 and output_flat.shape[0] <= output_flat.shape[1]):
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
            key_indices.append(conv_indices[-1])  # Last conv layer
        if dense_indices:
            key_indices.append(dense_indices[-2] if len(dense_indices) > 1 else dense_indices[-1])

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