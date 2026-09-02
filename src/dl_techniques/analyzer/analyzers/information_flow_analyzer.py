"""
Analyze the flow of information and feature dimensionality through layers.

This analyzer diagnoses the health of a neural network by examining the
activations of its intermediate layers. It is designed to detect common
pathologies such as vanishing or exploding gradients, dead neurons, and
information bottlenecks, which can impede a model's ability to learn.

Architecture
------------
The analysis is performed by temporarily wrapping the ``call`` method of each
selected layer on the layer INSTANCE, so that every wrapped layer records its
own output, and then running one eager forward pass (``model(x, training=
False)``) over a batch of data. A single pass therefore captures a snapshot of
the entire network's internal state, and every wrapper is removed again in a
``finally`` block, leaving the model bit-identical to how it was handed in.
The analyzer then processes this sequence of activation tensors to quantify
how information is transformed at each stage of the network.

Two properties of this mechanism are deliberate and load-bearing:

-   The capture pass must be EAGER. Under ``model.predict(...)`` Keras traces
    the forward function, so a wrapped ``call`` is handed a
    ``SymbolicTensor`` and nothing concrete is captured.
-   A temporary functional "feature extractor" sub-model is NOT used. Slicing
    one requires ``model.input`` and ``layer.output``, and neither exists for
    a subclassed model — a model kind this analyzer explicitly supports, since
    ``recursively_get_layers`` attribute-walks subclassed models to find their
    sublayers.

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
import functools
import itertools
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
        """Initialize the analyzer."""
        super().__init__(models, config)

    def requires_data(self) -> bool:
        """Information flow analysis requires input data."""
        return True

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze information flow by recording each selected layer's own output."""
        logger.info("Analyzing information flow and activations...")

        if data is None:
            raise ValueError("Data is required for information flow analysis")

        x_sample = data.x_data
        max_info_flow_samples = min(self.config.n_samples, 200)
        # Correctly handle dictionary vs. numpy array for batch size
        if isinstance(x_sample, dict):
            # Take the first key to determine batch size
            first_key = next(iter(x_sample))
            self._batch_size = len(x_sample[first_key])
            # Subsample the dictionary
            x_sample = {k: v[:min(max_info_flow_samples, self._batch_size)] for k, v in x_sample.items()}
            self._batch_size = len(x_sample[first_key])
        else:
            # Slice FIRST, then record the batch size, mirroring the dict path above.
            # Recording it before the slice makes `_safely_flatten_activations` reject
            # every rank>=3 activation and report `effective_rank = 0.0`.
            x_sample = x_sample[:min(max_info_flow_samples, len(x_sample))]
            self._batch_size = len(x_sample)


        for model_name, model in self.models.items():
            if not model.built:
                try:
                    model.predict(x_sample, verbose=0)
                    logger.info(f"Built model '{model_name}' for analysis.")
                except Exception as e:
                    logger.error(
                        f"Failed to build model '{model_name}'. Skipping information flow analysis. Error: {e}")
                    continue

            captured_outputs: Dict[str, np.ndarray] = {}
            # layer name -> index of the forward-pass invocation that produced the
            # tensor currently held in `captured_outputs`. See D-021.
            capture_order: Dict[str, int] = {}
            call_counter = itertools.count()
            # (layer, had_own_call, original_call_attribute) for every layer we wrap.
            wrapped: List[tuple] = []

            try:
                all_layers = recursively_get_layers(model)
                extraction_layers = self._get_extraction_layers(all_layers)

                if not extraction_layers:
                    logger.warning(f"No suitable layers for information flow analysis in '{model_name}'.")
                    continue

                for layer in extraction_layers:
                    if any(existing is layer for existing, _, _ in wrapped):
                        # The same layer object can be reached twice by the walk; wrap it once.
                        continue
                    had_own_call = 'call' in layer.__dict__
                    original_call = layer.__dict__.get('call')
                    wrapped.append((layer, had_own_call, original_call))
                    layer.call = self._make_recording_call(
                        layer, captured_outputs, capture_order, call_counter)

                # The pass MUST be eager. Under `model.predict(...)` Keras traces the
                # forward function and the wrapper is handed a SymbolicTensor, so
                # nothing concrete is captured.
                model(x_sample, training=False)

                # DECISION plan-2026-09-01T225724-e79ad4bd/D-021
                # Depth order is the order the layers were INVOKED in, recorded by
                # `recording_call` at forward time. Do NOT iterate `extraction_layers`
                # here and call the result depth: that list comes from
                # `recursively_get_layers`, which reaches a custom Layer's sublayers
                # through a `dir()` sweep and PREPENDS each match, so it is
                # reverse-alphabetical - measured EXACTLY backwards on a block
                # declaring `zeta, alpha, mid` and calling `alpha -> mid -> zeta`.
                # The `capture_index` written into each entry is what the visualizer
                # sorts on, so the order survives any dict rebuild. See decisions.md
                # D-021.
                by_name = {layer.name: layer for layer in extraction_layers}
                ordered_names = sorted(captured_outputs, key=capture_order.__getitem__)

                for layer in extraction_layers:
                    if layer.name not in captured_outputs:
                        logger.debug(f"No activation captured for layer '{layer.name}' in model '{model_name}'.")

                layer_analysis = {}
                for depth, name in enumerate(ordered_names):
                    layer = by_name[name]
                    layer_info = {'name': name, 'type': layer.__class__.__name__}
                    analysis = self._analyze_layer_information(
                        captured_outputs[name], layer_info)
                    analysis['capture_index'] = depth
                    layer_analysis[name] = analysis

                results.information_flow[model_name] = layer_analysis

                layer_outputs_list = [captured_outputs[n] for n in ordered_names]
                layer_info_list = [
                    {'name': n, 'type': by_name[n].__class__.__name__}
                    for n in ordered_names
                ]
                self._analyze_key_layer_activations(model_name, layer_outputs_list, layer_info_list, results)

            # A programming / API-shape error is never swallowed again: this analyzer
            # shipped a call to the PyTorch-only `register_forward_hook`, and the bare
            # `except Exception` that used to stand here turned that hard AttributeError
            # into a per-model log line, so `information_flow` was silently empty on
            # every run. A genuine per-model runtime failure still only skips that model,
            # so one bad model cannot abort a multi-model analysis.
            except (AttributeError, TypeError, NameError, ImportError):
                raise
            except Exception as e:
                logger.error(f"Failed to analyze information flow for {model_name}: {e}", exc_info=True)
                continue
            finally:
                # Runs on every exit path, including the re-raise and the `continue`.
                for layer, had_own_call, original_call in wrapped:
                    if had_own_call:
                        layer.call = original_call
                    else:
                        # Delete the per-instance attribute so the class method is
                        # reachable again; the model must be bit-identical afterwards.
                        layer.__dict__.pop('call', None)
                logger.debug(f"Restored `call` on {len(wrapped)} layers of model '{model_name}'.")

    @staticmethod
    def _make_recording_call(
            layer: keras.layers.Layer,
            store: Dict[str, np.ndarray],
            order: Dict[str, int],
            counter: 'itertools.count',
    ):
        """Build a `call` wrapper that records the layer's output into `store`.

        Args:
            layer: The layer whose bound `call` is being wrapped.
            store: Dict to record into, keyed by layer name. A layer invoked more
                than once in a forward pass (weight sharing) therefore keeps only
                its LAST output; this is deliberate — the per-layer analysis and
                the visualizer are both keyed by layer name.
            order: Dict recording, per layer name, the invocation index of the call
                that produced the tensor currently in `store`. It is rewritten on
                every invocation so it always describes the RETAINED output: a
                weight-shared layer is placed at the depth of its last use, not of
                its first, which is the only placement consistent with `store`.
            counter: Shared monotonically increasing invocation counter for the
                whole forward pass.

        Returns:
            A callable delegating `*args, **kwargs` to the original bound `call`,
            so Keras' `__call__` machinery (training/mask argument routing) is
            unaffected.
        """
        original_call = layer.call

        @functools.wraps(original_call)
        def recording_call(*args, **kwargs):
            outputs = original_call(*args, **kwargs)
            store[layer.name] = keras.ops.convert_to_numpy(outputs)
            order[layer.name] = next(counter)
            return outputs

        return recording_call

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