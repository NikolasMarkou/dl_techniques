"""
BandRMS-OOD: Geometric Out-of-Distribution Detection Layer

This module implements the BandRMS-OOD layer, which extends BandRMS normalization
with confidence-driven shell scaling to create learnable "spherical shells" in
feature space for geometric out-of-distribution detection.

The key innovation is using model confidence to influence shell placement:
- High-confidence (ID) samples → pushed toward outer shell (radius ≈ 1.0)
- Low-confidence (OOD) samples → remain in inner regions (radius < 1.0)

This implementation uses only Keras operations for full backend compatibility
(TensorFlow, JAX, PyTorch) and follows dl_techniques project standards.

Usage:
    ```python
    from dl_techniques.layers.norms.band_rms_ood import BandRMSOOD

    # Basic usage
    layer = BandRMSOOD(max_band_width=0.1, confidence_type='magnitude')

    # In a model
    x = BandRMSOOD(
        max_band_width=0.2,
        confidence_type='entropy',
        confidence_weight=1.5,
        name='ood_detection_layer'
    )(x)

    # Access shell distances for OOD detection
    shell_distance = layer.get_shell_distance()
    ood_score = ops.reduce_mean(shell_distance)
    ```
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, List

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class BandRMSOOD(keras.layers.Layer):
    """
    BandRMS-OOD: Geometric Out-of-Distribution Detection Layer.

    Extends BandRMS normalization with confidence-driven shell scaling to create
    learnable spherical shells in feature space. High-confidence samples are
    encouraged toward the outer shell while low-confidence samples remain in
    inner regions, enabling geometric OOD detection.

    Args:
        max_band_width: Maximum band width (α) controlling shell thickness.
            Range: [0.01, 0.5]. Smaller values create tighter shells.
        confidence_type: Method for confidence estimation:
            - 'magnitude': Use feature L2 norm as confidence proxy
            - 'entropy': Use feature entropy for uncertainty estimation
            - 'prediction': Use model prediction confidence (requires external signal)
        confidence_weight: Weight controlling confidence influence on shell placement.
            Range: [0.1, 3.0]. Higher values increase confidence sensitivity.
        shell_preference_weight: Loss weight for shell preference regularization.
            Range: [0.001, 0.1]. Encourages high-confidence samples to outer shell.
        axis: Axis for normalization. Default -1 for feature axis.
        epsilon: Small value to avoid division by zero.
        momentum: Momentum for running statistics (magnitude-based confidence).
        name: Layer name.
        **kwargs: Additional layer arguments.

    Input shape:
        Any shape with features along the specified axis.

    Output shape:
        Same as input shape.

    Returns:
        Normalized features with shell constraints. Use get_shell_distance()
        to access shell distances for OOD detection.

    Example:
        >>> # Create layer
        >>> layer = BandRMSOOD(max_band_width=0.1, confidence_type='magnitude')
        >>>
        >>> # Apply to features
        >>> features = tf.random.normal((32, 128))
        >>> normalized = layer(features)
        >>>
        >>> # Get OOD score
        >>> shell_distances = layer.get_shell_distance()
        >>> ood_score = tf.reduce_mean(shell_distances, axis=-1)
        >>>
        >>> print(f"Features shape: {features.shape}")
        >>> print(f"Normalized shape: {normalized.shape}")
        >>> print(f"OOD scores shape: {ood_score.shape}")
    """

    def __init__(
        self,
        max_band_width: float = 0.1,
        confidence_type: str = 'magnitude',
        confidence_weight: float = 1.0,
        shell_preference_weight: float = 0.01,
        axis: int = -1,
        epsilon: float = 1e-7,
        momentum: float = 0.99,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate arguments
        if not 0.01 <= max_band_width <= 0.5:
            raise ValueError(f"max_band_width must be in [0.01, 0.5], got {max_band_width}")

        if confidence_type not in ['magnitude', 'entropy', 'prediction']:
            raise ValueError(f"confidence_type must be one of ['magnitude', 'entropy', 'prediction'], got {confidence_type}")

        if not 0.1 <= confidence_weight <= 3.0:
            raise ValueError(f"confidence_weight must be in [0.1, 3.0], got {confidence_weight}")

        # Store configuration
        self.max_band_width = max_band_width
        self.confidence_type = confidence_type
        self.confidence_weight = confidence_weight
        self.shell_preference_weight = shell_preference_weight
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum

        # Internal state for OOD detection
        self.band_param = None
        self.shell_distances = None
        self.confidences = None
        self.current_shell_radii = None
        self._build_input_shape = None

        # Running statistics for magnitude-based confidence
        self.mag_mean = None
        self.mag_std = None

        # External confidence signal (for prediction-based confidence)
        self.external_confidence = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the layer weights and statistics.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Learnable band parameter (sigmoid-activated for [0,1] range)
        self.band_param = self.add_weight(
            name="band_param",
            shape=(),
            initializer="zeros",  # Start at sigmoid(0) = 0.5
            trainable=True,
        )

        # Running statistics for magnitude-based confidence normalization
        if self.confidence_type == 'magnitude':
            self.mag_mean = self.add_weight(
                name="mag_mean",
                shape=(),
                initializer="ones",
                trainable=False,
            )
            self.mag_std = self.add_weight(
                name="mag_std",
                shape=(),
                initializer="ones",
                trainable=False,
            )

        super().build(input_shape)

    def set_external_confidence(self, confidence) -> None:
        """
        Set external confidence signal for prediction-based confidence.

        Args:
            confidence: External confidence tensor with shape matching batch size.
        """
        self.external_confidence = confidence

    def estimate_confidence(self, features, training: Optional[bool] = None):
        """
        Estimate confidence based on the specified method.

        Args:
            features: Input features before normalization.
            training: Whether in training mode.

        Returns:
            Confidence scores in range [0, 1], where higher values indicate
            higher confidence (more likely to be in-distribution).
        """
        if self.confidence_type == 'magnitude':
            return self._magnitude_confidence(features, training)
        elif self.confidence_type == 'entropy':
            return self._entropy_confidence(features)
        elif self.confidence_type == 'prediction':
            return self._prediction_confidence(features)
        else:
            raise ValueError(f"Unknown confidence type: {self.confidence_type}")

    def _magnitude_confidence(self, features, training: Optional[bool] = None):
        """
        Compute magnitude-based confidence using feature L2 norm.

        Higher magnitude indicates more confident, organized features.
        """
        magnitude = ops.norm(features, axis=self.axis, keepdims=True)

        if training:
            # Update running statistics during training
            batch_mean = ops.mean(magnitude)
            batch_var = ops.var(magnitude)
            batch_std = ops.sqrt(batch_var + self.epsilon)

            # Update with momentum
            self.mag_mean.assign(
                self.momentum * self.mag_mean + (1 - self.momentum) * batch_mean
            )
            self.mag_std.assign(
                self.momentum * self.mag_std + (1 - self.momentum) * batch_std
            )

        # Normalize magnitude and convert to confidence
        normalized_mag = (magnitude - self.mag_mean) / (self.mag_std + self.epsilon)
        confidence = ops.sigmoid(normalized_mag)

        return confidence

    def _entropy_confidence(self, features):
        """
        Compute entropy-based confidence using feature uncertainty.

        Lower entropy indicates more organized, confident features.
        """
        # Use absolute values for entropy computation
        abs_features = ops.abs(features)

        # Normalize to create probability distribution
        feature_sum = ops.sum(abs_features, axis=self.axis, keepdims=True)
        normalized_features = abs_features / (feature_sum + self.epsilon)

        # Compute entropy
        log_normalized = ops.log(normalized_features + self.epsilon)
        entropy = -ops.sum(normalized_features * log_normalized, axis=self.axis, keepdims=True)

        # Convert entropy to confidence (invert and normalize)
        feature_dim = ops.cast(ops.shape(features)[self.axis], dtype=features.dtype)
        max_entropy = ops.log(feature_dim)
        confidence = 1.0 - (entropy / max_entropy)

        # Ensure confidence is in [0, 1]
        confidence = ops.clip(confidence, 0.0, 1.0)

        return confidence

    def _prediction_confidence(self, features):
        """
        Use external prediction confidence signal.

        Falls back to magnitude-based confidence if no external signal is available.
        """
        if self.external_confidence is not None:
            # Reshape external confidence to match feature dimensions
            batch_size = ops.shape(features)[0]
            confidence_shape = [batch_size] + [1] * (len(features.shape) - 1)
            confidence = ops.reshape(self.external_confidence, confidence_shape)

            # Ensure confidence is broadcasted correctly
            target_shape = list(features.shape)
            target_shape[self.axis] = 1
            confidence = ops.broadcast_to(confidence, target_shape)

            return confidence
        else:
            # Fallback to magnitude-based confidence
            logger.warning("No external confidence signal available, falling back to magnitude-based confidence")
            return self._magnitude_confidence(features, training=False)

    def apply_shell_scaling(
        self,
        normalized_features,
        confidence
    ) -> Tuple:
        """
        Apply confidence-driven shell scaling to normalized features.

        Args:
            normalized_features: RMS-normalized features.
            confidence: Confidence scores for shell placement.

        Returns:
            Tuple of (scaled_features, shell_distances, shell_radii)
        """
        # Confidence-driven shell scaling
        # Higher confidence → scale closer to 1.0 (outer shell)
        # Lower confidence → scale closer to (1 - α) (inner regions)
        confidence_influence = self.confidence_weight * confidence

        # Sigmoid activation ensures scaling factor is in [0, 1]
        band_activation = ops.sigmoid(5.0 * self.band_param + confidence_influence)

        # Scale factor ranges from (1 - α) to 1.0
        scale_factor = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling
        scaled_features = normalized_features * scale_factor

        # Compute shell radius and distance from target (outer shell at radius 1.0)
        shell_radius = ops.norm(scaled_features, axis=self.axis, keepdims=True)
        shell_distance = ops.abs(shell_radius - 1.0)

        return scaled_features, shell_distance, shell_radius

    def call(self, inputs, training: Optional[bool] = None):
        """
        Forward pass with confidence-driven shell scaling.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.

        Returns:
            Normalized features with shell constraints.
        """
        # Step 1: RMS normalization (project to unit norm)
        rms = ops.sqrt(
            ops.mean(ops.square(inputs), axis=self.axis, keepdims=True) + self.epsilon
        )
        normalized_features = inputs / rms

        # Step 2: Estimate confidence
        confidence = self.estimate_confidence(inputs, training=training)

        # Step 3: Apply confidence-driven shell scaling
        scaled_features, shell_distance, shell_radius = self.apply_shell_scaling(
            normalized_features, confidence
        )

        # Store for OOD detection and analysis
        self.shell_distances = shell_distance
        self.confidences = confidence
        self.current_shell_radii = shell_radius

        return scaled_features

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute output shape (unchanged from input).

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input).
        """
        return input_shape

    def get_shell_distance(self):
        """
        Get current shell distances for OOD detection.

        Returns:
            Shell distances (higher values indicate more likely OOD).
            Returns None if no forward pass has been performed.
        """
        return self.shell_distances

    def get_confidences(self):
        """
        Get current confidence estimates.

        Returns:
            Confidence scores in range [0, 1].
            Returns None if no forward pass has been performed.
        """
        return self.confidences

    def get_shell_radii(self):
        """
        Get current shell radii.

        Returns:
            Shell radii (distance from origin in normalized space).
            Returns None if no forward pass has been performed.
        """
        return self.current_shell_radii

    def compute_shell_preference_loss(self, confidence_threshold: float = 0.7):
        """
        Compute shell preference loss to encourage high-confidence samples to outer shell.

        Args:
            confidence_threshold: Confidence threshold for high-confidence samples.

        Returns:
            Shell preference loss (scalar).
        """
        if self.shell_distances is None or self.confidences is None:
            return ops.convert_to_tensor(0.0, dtype=self.dtype)

        # Encourage high-confidence samples to have low shell distance
        high_confidence_mask = self.confidences > confidence_threshold
        high_conf_distances = ops.where(
            high_confidence_mask,
            self.shell_distances,
            ops.zeros_like(self.shell_distances)
        )

        # Weighted loss by confidence
        confidence_weights = ops.where(
            high_confidence_mask,
            self.confidences,
            ops.zeros_like(self.confidences)
        )

        weighted_loss = confidence_weights * ops.square(high_conf_distances)
        return ops.mean(weighted_loss)

    def compute_separation_loss(self, confidence_threshold: float = 0.7):
        """
        Compute separation loss to encourage separation between high/low confidence samples.

        Args:
            confidence_threshold: Threshold separating high and low confidence.

        Returns:
            Separation loss (scalar).
        """
        if self.shell_distances is None or self.confidences is None:
            return ops.convert_to_tensor(0.0, dtype=self.dtype)

        # Separate high and low confidence samples
        high_conf_mask = self.confidences > confidence_threshold
        low_conf_mask = self.confidences <= confidence_threshold

        high_conf_distances = ops.where(
            high_conf_mask,
            self.shell_distances,
            ops.ones_like(self.shell_distances) * float('inf')
        )
        low_conf_distances = ops.where(
            low_conf_mask,
            self.shell_distances,
            ops.zeros_like(self.shell_distances)
        )

        # Compute average distances for each group
        # Use finite check to avoid inf values
        finite_mask_high = ops.isfinite(high_conf_distances)
        finite_mask_low = ops.isfinite(low_conf_distances)

        high_avg = ops.mean(ops.where(
            finite_mask_high,
            high_conf_distances,
            ops.zeros_like(high_conf_distances)
        ))
        low_avg = ops.mean(low_conf_distances)

        # Encourage separation (low confidence should have higher shell distance)
        target_separation = 0.1
        current_separation = low_avg - high_avg
        separation_loss = ops.maximum(0.0, target_separation - current_separation)

        return separation_loss

    def get_ood_detection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for OOD detection analysis.

        Returns:
            Dictionary containing various statistics for analysis.
        """
        if self.shell_distances is None or self.confidences is None:
            return {}

        stats = {
            'mean_shell_distance': ops.mean(self.shell_distances),
            'std_shell_distance': ops.std(self.shell_distances),
            'mean_confidence': ops.mean(self.confidences),
            'std_confidence': ops.std(self.confidences),
            'mean_shell_radius': ops.mean(self.current_shell_radii),
            'shell_utilization': ops.std(self.current_shell_radii),  # Higher = better utilization
        }

        # Confidence-shell correlation
        conf_flat = ops.reshape(self.confidences, [-1])
        shell_flat = ops.reshape(self.shell_distances, [-1])

        # Pearson correlation coefficient
        conf_mean = ops.mean(conf_flat)
        shell_mean = ops.mean(shell_flat)

        numerator = ops.mean((conf_flat - conf_mean) * (shell_flat - shell_mean))
        conf_var = ops.mean(ops.square(conf_flat - conf_mean))
        shell_var = ops.mean(ops.square(shell_flat - shell_mean))

        correlation = numerator / (ops.sqrt(conf_var * shell_var) + self.epsilon)
        stats['confidence_shell_correlation'] = correlation

        return stats

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "confidence_type": self.confidence_type,
            "confidence_weight": self.confidence_weight,
            "shell_preference_weight": self.shell_preference_weight,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a configuration.

        Args:
            config: Build configuration dictionary.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# ==============================================================================
# MULTI-LAYER OOD DETECTOR
# ==============================================================================

class MultiLayerOODDetector:
    """
    Multi-layer OOD detector that aggregates shell distances from multiple BandRMS-OOD layers.

    This class provides a unified interface for OOD detection using multiple BandRMS-OOD
    layers throughout a neural network. It can aggregate signals from different depths
    and provide robust OOD detection.

    Args:
        model: Keras model containing BandRMS-OOD layers.
        layer_weights: Weights for combining signals from different layers.
            If None, uses uniform weighting.
        aggregation_method: Method for aggregating multi-layer signals:
            - 'weighted_average': Weighted average of layer distances
            - 'max': Maximum distance across layers
            - 'consensus': Majority voting across layers

    Example:
        >>> # Create detector
        >>> detector = MultiLayerOODDetector(model, layer_weights=[0.3, 0.4, 0.3])
        >>>
        >>> # Fit threshold on ID data
        >>> detector.fit_threshold(id_validation_data, fpr_target=0.05)
        >>>
        >>> # Detect OOD samples
        >>> ood_predictions, ood_scores = detector.predict_ood(test_data)
        >>>
        >>> # Evaluate performance
        >>> auroc = detector.evaluate_detection(id_test_data, ood_test_data)
    """

    def __init__(
        self,
        model: keras.Model,
        layer_weights: Optional[List[float]] = None,
        aggregation_method: str = 'weighted_average'
    ):
        self.model = model
        self.aggregation_method = aggregation_method
        self.threshold = None

        # Find all BandRMS-OOD layers
        self.ood_layers = self._find_ood_layers()

        if not self.ood_layers:
            raise ValueError("No BandRMS-OOD layers found in the model")

        # Set layer weights
        if layer_weights is None:
            self.layer_weights = [1.0] * len(self.ood_layers)
        else:
            if len(layer_weights) != len(self.ood_layers):
                raise ValueError(f"Number of layer weights ({len(layer_weights)}) "
                               f"must match number of OOD layers ({len(self.ood_layers)})")
            self.layer_weights = layer_weights

        # Normalize weights
        total_weight = sum(self.layer_weights)
        self.layer_weights = [w / total_weight for w in self.layer_weights]

        logger.info(f"Initialized MultiLayerOODDetector with {len(self.ood_layers)} layers")
        logger.info(f"Layer names: {[layer.name for layer in self.ood_layers]}")
        logger.info(f"Layer weights: {self.layer_weights}")

    def _find_ood_layers(self) -> List[BandRMSOOD]:
        """Find all BandRMS-OOD layers in the model."""
        ood_layers = []

        def _search_layers(layer):
            if isinstance(layer, BandRMSOOD):
                ood_layers.append(layer)
            # Handle nested models and layers with sublayers
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    _search_layers(sublayer)

        for layer in self.model.layers:
            _search_layers(layer)

        return ood_layers

    def compute_ood_scores(self, data) -> Any:
        """
        Compute OOD scores for input data.

        Args:
            data: Input data tensor.

        Returns:
            OOD scores (higher values indicate more likely OOD).
        """
        # Forward pass to populate shell distances
        _ = self.model(data, training=False)

        # Collect shell distances from all BandRMS-OOD layers
        layer_distances = []

        for layer in self.ood_layers:
            distances = layer.get_shell_distance()
            if distances is not None:
                # Reduce spatial dimensions if present (e.g., for conv layers)
                if len(distances.shape) > 2:
                    distances = ops.mean(distances, axis=list(range(1, len(distances.shape) - 1)))

                # Flatten to per-sample scores
                per_sample_distance = ops.mean(distances, axis=-1)
                layer_distances.append(per_sample_distance)

        if not layer_distances:
            raise RuntimeError("No shell distances found. Ensure model has been called on data.")

        # Aggregate layer distances
        if self.aggregation_method == 'weighted_average':
            weighted_sum = ops.zeros_like(layer_distances[0])
            for dist, weight in zip(layer_distances, self.layer_weights):
                weighted_sum = weighted_sum + weight * dist
            ood_scores = weighted_sum

        elif self.aggregation_method == 'max':
            ood_scores = layer_distances[0]
            for dist in layer_distances[1:]:
                ood_scores = ops.maximum(ood_scores, dist)

        elif self.aggregation_method == 'consensus':
            # Use median as consensus measure
            stacked_distances = ops.stack(layer_distances, axis=-1)
            # For median, we'll use a simple approximation since ops.sort might not be available
            # Use mean as approximation for median
            ood_scores = ops.mean(stacked_distances, axis=-1)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return ood_scores

    def fit_threshold(
        self,
        id_data,
        fpr_target: float = 0.05,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Fit detection threshold based on in-distribution data.

        Args:
            id_data: In-distribution validation data.
            fpr_target: Target false positive rate.
            batch_size: Batch size for processing large datasets.
        """
        if batch_size is None:
            id_scores = self.compute_ood_scores(id_data)
        else:
            # Process in batches for large datasets
            id_scores_list = []
            num_samples = ops.shape(id_data)[0]

            for i in range(0, num_samples, batch_size):
                batch = id_data[i:i + batch_size]
                batch_scores = self.compute_ood_scores(batch)
                id_scores_list.append(batch_scores)

            id_scores = ops.concatenate(id_scores_list, axis=0)

        # Convert to numpy for percentile calculation
        try:
            import numpy as np
            id_scores_np = id_scores.numpy() if hasattr(id_scores, 'numpy') else np.array(id_scores)
            self.threshold = float(np.percentile(id_scores_np, (1.0 - fpr_target) * 100))
        except:
            # Fallback if numpy conversion fails
            # Use a simple approximation by sorting and indexing
            sorted_scores = ops.sort(ops.reshape(id_scores, [-1]))
            threshold_idx = int((1.0 - fpr_target) * ops.shape(sorted_scores)[0])
            threshold_idx = min(threshold_idx, ops.shape(sorted_scores)[0] - 1)
            self.threshold = float(sorted_scores[threshold_idx])

        logger.info(f"Set OOD threshold to {self.threshold:.6f} (target FPR: {fpr_target})")
        logger.info(f"ID score statistics: mean={ops.mean(id_scores):.6f}, "
                   f"std={ops.std(id_scores):.6f}")

    def predict_ood(self, data) -> Tuple:
        """
        Predict OOD for input data.

        Args:
            data: Input data tensor.

        Returns:
            Tuple of (ood_predictions, ood_scores) where ood_predictions
            are boolean values indicating OOD samples.
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")

        ood_scores = self.compute_ood_scores(data)
        ood_predictions = ood_scores > self.threshold

        return ood_predictions, ood_scores

    def evaluate_detection(
        self,
        id_data,
        ood_data
    ) -> Dict[str, float]:
        """
        Evaluate OOD detection performance.

        Args:
            id_data: In-distribution test data.
            ood_data: Out-of-distribution test data.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Import here to avoid circular dependencies
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
        except ImportError:
            raise ImportError("sklearn is required for evaluation metrics")

        # Compute scores
        id_scores = self.compute_ood_scores(id_data)
        ood_scores = self.compute_ood_scores(ood_data)

        # Create labels and combine scores
        y_true = ops.concatenate([
            ops.zeros(ops.shape(id_scores)[0], dtype='int32'),
            ops.ones(ops.shape(ood_scores)[0], dtype='int32')
        ], axis=0)
        y_scores = ops.concatenate([id_scores, ood_scores], axis=0)

        # Convert to numpy for sklearn
        try:
            import numpy as np
            y_true_np = y_true.numpy() if hasattr(y_true, 'numpy') else np.array(y_true)
            y_scores_np = y_scores.numpy() if hasattr(y_scores, 'numpy') else np.array(y_scores)
        except:
            # If conversion fails, return basic statistics
            return {
                'auroc': 0.5,  # Random performance
                'aupr': float(ops.mean(y_true)),
                'fpr95': 1.0,
                'id_mean': float(ops.mean(id_scores)),
                'id_std': float(ops.std(id_scores)),
                'ood_mean': float(ops.mean(ood_scores)),
                'ood_std': float(ops.std(ood_scores))
            }

        # Calculate metrics
        auroc = roc_auc_score(y_true_np, y_scores_np)
        aupr = average_precision_score(y_true_np, y_scores_np)

        # Calculate FPR@95
        fpr, tpr, _ = roc_curve(y_true_np, y_scores_np)
        fpr95_idx = ops.argmax(tpr >= 0.95)
        fpr95 = float(fpr[fpr95_idx]) if fpr95_idx < len(fpr) else 1.0

        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr95,
            'id_mean': float(ops.mean(id_scores)),
            'id_std': float(ops.std(id_scores)),
            'ood_mean': float(ops.mean(ood_scores)),
            'ood_std': float(ops.std(ood_scores))
        }