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
from dl_techniques.utils.logger import logger


class InformationFlowAnalyzer(BaseAnalyzer):
    """Analyzes information flow through network layers."""

    def __init__(self, models: Dict[str, keras.Model], config: Any):
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
                logger.warning(f"No extraction data available for model {model_name}")
                continue

            try:
                # Get multi-layer outputs
                layer_outputs = extraction_data['model'].predict(x_sample, verbose=0)
                layer_info = extraction_data['layer_info']

                # Handle single output case
                if not isinstance(layer_outputs, (list, tuple)):
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
            try:
                # Check if model has accessible input
                if not hasattr(model, 'input') or model.input is None:
                    logger.warning(f"Model {model_name} has no accessible input attribute")
                    self.layer_extraction_models[model_name] = None
                    continue

                # Set up multi-layer extraction for information flow
                extraction_layers = self._get_extraction_layers(model)
                if extraction_layers:
                    try:
                        # Validate outputs
                        valid_outputs = []
                        valid_layer_info = []
                        for layer_info in extraction_layers:
                            if layer_info['output'] is not None:
                                valid_outputs.append(layer_info['output'])
                                valid_layer_info.append(layer_info)
                            else:
                                logger.warning(f"Layer {layer_info['name']} has None output")

                        if valid_outputs:
                            self.layer_extraction_models[model_name] = {
                                'model': keras.Model(inputs=model.input, outputs=valid_outputs),
                                'layer_info': valid_layer_info
                            }
                        else:
                            logger.warning(f"No valid layer outputs found for {model_name}")
                            self.layer_extraction_models[model_name] = None

                    except Exception as e:
                        logger.warning(f"Could not create layer extraction model for {model_name}: {e}")
                        self.layer_extraction_models[model_name] = None
                else:
                    logger.warning(f"No suitable layers found for extraction in {model_name}")
                    self.layer_extraction_models[model_name] = None
            except Exception as e:
                logger.error(f"Failed to setup activation models for {model_name}: {e}")
                self.layer_extraction_models[model_name] = None

    def _get_extraction_layers(self, model: keras.Model) -> List[Dict[str, Any]]:
        """Get layers suitable for information flow analysis."""
        extraction_layers = []

        try:
            # Check if model has layers attribute
            if not hasattr(model, 'layers'):
                logger.warning(f"Model does not have 'layers' attribute")
                return extraction_layers

            for layer in model.layers:
                try:
                    # Check for common layer types that are suitable for analysis
                    if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense,
                                          keras.layers.BatchNormalization, keras.layers.LayerNormalization)):
                        if hasattr(layer, 'output') and layer.output is not None:
                            extraction_layers.append({
                                'name': layer.name,
                                'type': layer.__class__.__name__,
                                'output': layer.output
                            })
                except Exception as e:
                    logger.warning(f"Could not process layer {getattr(layer, 'name', 'unknown')}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Failed to extract layers from model: {e}")

        return extraction_layers

    def _analyze_layer_information(self, output: np.ndarray, layer_info: Dict) -> Dict[str, Any]:
        """Analysis of layer information content."""
        # Flatten spatial dimensions if needed
        if len(output.shape) == 4:  # Conv layer
            output_flat = np.mean(output, axis=(1, 2))
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
        if spatial_output is not None:
            analysis['spatial_mean'] = float(np.mean(spatial_output))
            analysis['spatial_std'] = float(np.std(spatial_output))
            analysis['channel_variance'] = float(np.var(np.mean(spatial_output, axis=(0, 1, 2))))

        # Compute effective rank
        if output_flat.shape[1] > 1:
            try:
                _, s, _ = np.linalg.svd(output_flat, full_matrices=False)
                s = s + 1e-10  # Avoid log(0)
                s_normalized = s / np.sum(s)
                effective_rank = np.exp(-np.sum(s_normalized * np.log(s_normalized)))
                analysis['effective_rank'] = float(effective_rank)
            except:
                analysis['effective_rank'] = 0.0

        return analysis

    def _analyze_key_layer_activations(self, model_name: str, layer_outputs: List[np.ndarray],
                                       layer_info: List[Dict], results: AnalysisResults) -> None:
        """Analyze activations for key layers in detail."""
        # Find key layers (last conv and middle dense)
        conv_indices = [i for i, info in enumerate(layer_info) if info['type'] == 'Conv2D']
        dense_indices = [i for i, info in enumerate(layer_info) if info['type'] == 'Dense']

        key_indices = []
        if conv_indices:
            key_indices.append(conv_indices[-1])  # Last conv layer
        if dense_indices and len(dense_indices) > 1:
            key_indices.append(dense_indices[len(dense_indices) // 2])  # Middle dense layer

        results.activation_stats[model_name] = {}

        for idx in key_indices:
            if idx < len(layer_outputs):
                layer_name = layer_info[idx]['name']
                activations = layer_outputs[idx]

                flat_acts = activations.flatten()

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
                    'sample_activations': activations[:10] if len(activations.shape) == 4 else None
                }