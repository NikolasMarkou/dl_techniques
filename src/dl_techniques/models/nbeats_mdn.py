"""
Probabilistic N-BEATS Model with MDN Integration.

This module combines the N-BEATS architecture with Mixture Density Networks
to provide probabilistic time series forecasting with uncertainty quantification.
"""

import keras
from keras import ops
import numpy as np
from typing import List, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.time_series.nbeats_blocks import (
    GenericBlock, TrendBlock, SeasonalityBlock
)
from dl_techniques.layers.mdn_layer import MDNLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ProbabilisticNBeatsNet(keras.Model):
    """Probabilistic N-BEATS neural network with MDN integration.

    This model combines the structured architecture of N-BEATS with the
    probabilistic capabilities of Mixture Density Networks. Instead of
    producing deterministic forecasts, it outputs mixture parameters that
    define probability distributions over future values.

    Key Features:
        - Maintains N-BEATS interpretable structure (trend, seasonality)
        - Provides uncertainty quantification through mixture distributions
        - Supports multi-modal forecasting scenarios
        - Enables risk assessment and confidence intervals

    Architecture:
        1. N-BEATS blocks process input and generate intermediate forecasts
        2. Block outputs are aggregated into rich feature representations
        3. MDN layer maps features to mixture distribution parameters
        4. Final output: mixture of Gaussians for probabilistic forecasting

    Args:
        input_dim: Integer, dimensionality of input time series (default: 1).
        output_dim: Integer, dimensionality of output time series (default: 1).
        backcast_length: Integer, length of the input time series window.
        forecast_length: Integer, length of the forecast horizon.
        stack_types: List of strings, types of stacks ('generic', 'trend', 'seasonality').
        nb_blocks_per_stack: Integer, number of blocks per stack.
        thetas_dim: List of integers, theta dimensionality for each stack.
        share_weights_in_stack: Boolean, whether to share weights within stacks.
        hidden_layer_units: Integer, number of hidden units in FC layers.
        num_mixtures: Integer, number of Gaussian mixtures for MDN.
        mdn_hidden_units: Integer, number of hidden units in MDN preprocessing.
        aggregation_mode: String, how to aggregate block outputs ('sum', 'concat', 'attention').
        **kwargs: Additional keyword arguments for the Model parent class.

    Example:
        >>> model = ProbabilisticNBeatsNet(
        ...     backcast_length=48,
        ...     forecast_length=12,
        ...     stack_types=['trend', 'seasonality'],
        ...     nb_blocks_per_stack=2,
        ...     thetas_dim=[3, 6],
        ...     num_mixtures=3
        ... )
        >>> model.compile(optimizer='adam', loss=model.mdn_loss)
    """

    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    def __init__(
            self,
            input_dim: int = 1,
            output_dim: int = 1,
            backcast_length: int = 10,
            forecast_length: int = 1,
            stack_types: List[str] = ['trend', 'seasonality'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8],
            share_weights_in_stack: bool = False,
            hidden_layer_units: int = 256,
            num_mixtures: int = 3,
            mdn_hidden_units: int = 128,
            aggregation_mode: str = 'concat',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units
        self.num_mixtures = num_mixtures
        self.mdn_hidden_units = mdn_hidden_units
        self.aggregation_mode = aggregation_mode

        # Validate inputs
        if len(self.stack_types) != len(self.thetas_dim):
            raise ValueError(
                f"Length of stack_types ({len(self.stack_types)}) must match "
                f"length of thetas_dim ({len(self.thetas_dim)})"
            )

        valid_stack_types = {self.GENERIC_BLOCK, self.TREND_BLOCK, self.SEASONALITY_BLOCK}
        for stack_type in self.stack_types:
            if stack_type not in valid_stack_types:
                raise ValueError(f"Invalid stack type: {stack_type}")

        valid_aggregation_modes = {'sum', 'concat', 'attention'}
        if self.aggregation_mode not in valid_aggregation_modes:
            raise ValueError(f"Invalid aggregation mode: {self.aggregation_mode}")

        # Initialize components
        self.blocks: List[List[Any]] = []
        self.feature_aggregator = None
        self.mdn_preprocessor = None
        self.mdn_layer = None
        self.attention_weights = None

        # Build the network
        self._build_network()

    def _build_network(self) -> None:
        """Build the probabilistic N-BEATS network architecture."""
        logger.info(f"Building Probabilistic N-BEATS with {len(self.stack_types)} stacks and {self.num_mixtures} mixtures")

        # Create N-BEATS blocks (same as original)
        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []

            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"

                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=theta_dim,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack,
                        name=block_name
                    )
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=theta_dim,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack,
                        name=block_name
                    )
                elif stack_type == self.SEASONALITY_BLOCK:
                    block = SeasonalityBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=theta_dim,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack,
                        name=block_name
                    )

                stack_blocks.append(block)
            self.blocks.append(stack_blocks)

        # Build aggregation and MDN components
        self._build_aggregation_layer()
        self._build_mdn_components()

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"Probabilistic N-BEATS built with {total_blocks} total blocks and {self.num_mixtures} mixtures")

    def _build_aggregation_layer(self) -> None:
        """Build the feature aggregation layer."""
        total_blocks = sum(len(stack) for stack in self.blocks)

        if self.aggregation_mode == 'concat':
            # Calculate total feature dimension after concatenation
            feature_dim = total_blocks * self.forecast_length
            self.feature_aggregator = keras.layers.Dense(
                self.mdn_hidden_units,
                activation='relu',
                name='forecast_aggregator'
            )
        elif self.aggregation_mode == 'attention':
            # Attention-based aggregation
            self.attention_weights = keras.layers.Dense(
                total_blocks,
                activation='softmax',
                name='attention_weights'
            )
            self.feature_aggregator = keras.layers.Dense(
                self.mdn_hidden_units,
                activation='relu',
                name='attended_features'
            )
        # 'sum' mode doesn't need additional layers

    def _build_mdn_components(self) -> None:
        """Build MDN preprocessing and output layers."""
        # Preprocessing layer for MDN
        self.mdn_preprocessor = keras.layers.Dense(
            self.mdn_hidden_units,
            activation='relu',
            name='mdn_preprocessor'
        )

        # MDN layer for probabilistic output
        self.mdn_layer = MDNLayer(
            output_dimension=self.output_dim,
            num_mixtures=self.num_mixtures,
            name='mdn_output'
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the probabilistic N-BEATS model.

        Args:
            inputs: Input tensor of shape (batch_size, backcast_length) or
                   (batch_size, backcast_length, 1).
            training: Boolean indicating training mode.

        Returns:
            Mixture parameters tensor for probabilistic forecasting.
        """
        # Handle input shapes (same as original N-BEATS)
        if len(inputs.shape) == 3:
            if inputs.shape[-1] == 1:
                inputs = ops.squeeze(inputs, axis=-1)
            else:
                inputs = inputs[..., 0]
        elif len(inputs.shape) == 1:
            inputs = ops.expand_dims(inputs, axis=0)

        batch_size = ops.shape(inputs)[0]

        # Process through N-BEATS blocks
        residual = inputs
        block_forecasts = []  # Collect forecasts from all blocks

        for stack_id, stack_blocks in enumerate(self.blocks):
            for block_id, block in enumerate(stack_blocks):
                # Get backcast and forecast from block
                backcast, forecast = block(residual, training=training)

                # Update residual (standard N-BEATS behavior)
                residual = residual - backcast

                # Store forecast for aggregation
                block_forecasts.append(forecast)

        # Aggregate block forecasts into features
        aggregated_features = self._aggregate_forecasts(block_forecasts, training=training)

        # Process through MDN preprocessing
        mdn_features = self.mdn_preprocessor(aggregated_features, training=training)

        # Generate mixture parameters
        mixture_params = self.mdn_layer(mdn_features, training=training)

        return mixture_params

    def _aggregate_forecasts(
        self,
        block_forecasts: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Aggregate forecasts from all blocks into feature representation.

        Args:
            block_forecasts: List of forecast tensors from all blocks.
            training: Boolean indicating training mode.

        Returns:
            Aggregated feature tensor.
        """
        if self.aggregation_mode == 'sum':
            # Simple summation (closest to original N-BEATS)
            aggregated = ops.stack(block_forecasts, axis=1)  # [batch, num_blocks, forecast_length]
            aggregated = ops.sum(aggregated, axis=1)  # [batch, forecast_length]

        elif self.aggregation_mode == 'concat':
            # Concatenate all forecasts
            aggregated = ops.concatenate(block_forecasts, axis=-1)  # [batch, num_blocks * forecast_length]
            aggregated = self.feature_aggregator(aggregated, training=training)

        elif self.aggregation_mode == 'attention':
            # Attention-weighted combination
            stacked_forecasts = ops.stack(block_forecasts, axis=1)  # [batch, num_blocks, forecast_length]

            # Compute attention weights over blocks
            # Use mean of each forecast as the key for attention
            forecast_means = ops.mean(stacked_forecasts, axis=-1)  # [batch, num_blocks]
            attention_scores = self.attention_weights(forecast_means, training=training)  # [batch, num_blocks]

            # Apply attention weights
            attention_expanded = ops.expand_dims(attention_scores, axis=-1)  # [batch, num_blocks, 1]
            weighted_forecasts = stacked_forecasts * attention_expanded  # [batch, num_blocks, forecast_length]
            aggregated = ops.sum(weighted_forecasts, axis=1)  # [batch, forecast_length]
            aggregated = self.feature_aggregator(aggregated, training=training)

        return aggregated

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the mixture parameters tensor.
        """
        batch_size = input_shape[0]
        # MDN output size: (2 * num_mixtures * output_dim) + num_mixtures
        mdn_output_size = (2 * self.num_mixtures * self.output_dim) + self.num_mixtures
        return (batch_size, mdn_output_size)

    def mdn_loss(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """MDN loss function for training.

        Args:
            y_true: True target values.
            y_pred: Predicted mixture parameters.

        Returns:
            Negative log likelihood loss.
        """
        return self.mdn_layer.loss_func(y_true, y_pred)

    def predict_probabilistic(
        self,
        x: np.ndarray,
        num_samples: int = 100,
        return_components: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate probabilistic predictions with uncertainty quantification.

        Args:
            x: Input data for prediction.
            num_samples: Number of samples to draw from the mixture.
            return_components: Whether to return individual mixture components.

        Returns:
            Dictionary containing:
                - 'point_estimate': Expected value of the mixture
                - 'samples': Samples from the mixture distribution
                - 'total_variance': Total predictive variance
                - 'aleatoric_variance': Data noise variance
                - 'epistemic_variance': Model uncertainty variance
                - 'mixture_params': Raw mixture parameters (if return_components=True)
        """
        # Get mixture parameters
        mixture_params = self.predict(x)

        # Extract components
        mu, sigma, pi_logits = self.mdn_layer.split_mixture_params(mixture_params)
        pi = keras.activations.softmax(pi_logits, axis=-1)

        # Convert to numpy
        mu_np = ops.convert_to_numpy(mu)
        sigma_np = ops.convert_to_numpy(sigma)
        pi_np = ops.convert_to_numpy(pi)

        # Calculate point estimates (weighted mean)
        pi_expanded = np.expand_dims(pi_np, axis=-1)
        point_estimates = np.sum(pi_expanded * mu_np, axis=1)

        # Generate samples
        samples_list = []
        for _ in range(num_samples):
            sample = self.mdn_layer.sample(mixture_params)
            samples_list.append(ops.convert_to_numpy(sample))
        samples = np.stack(samples_list, axis=1)  # [batch, num_samples, output_dim]

        # Calculate uncertainties
        point_expanded = np.expand_dims(point_estimates, axis=1)

        # Aleatoric uncertainty (data noise)
        aleatoric_variance = np.sum(pi_expanded * sigma_np**2, axis=1)

        # Epistemic uncertainty (model uncertainty)
        epistemic_variance = np.sum(pi_expanded * (mu_np - point_expanded)**2, axis=1)

        # Total variance
        total_variance = aleatoric_variance + epistemic_variance

        results = {
            'point_estimate': point_estimates,
            'samples': samples,
            'total_variance': total_variance,
            'aleatoric_variance': aleatoric_variance,
            'epistemic_variance': epistemic_variance,
        }

        if return_components:
            results['mixture_params'] = {
                'mu': mu_np,
                'sigma': sigma_np,
                'pi': pi_np
            }

        return results

    def get_prediction_intervals(
        self,
        x: np.ndarray,
        confidence_levels: List[float] = [0.68, 0.95]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Get prediction intervals for different confidence levels.

        Args:
            x: Input data for prediction.
            confidence_levels: List of confidence levels (e.g., [0.68, 0.95]).

        Returns:
            Dictionary with confidence levels as keys and interval bounds as values.
        """
        from scipy import stats

        predictions = self.predict_probabilistic(x)
        point_est = predictions['point_estimate']
        total_var = predictions['total_variance']

        intervals = {}
        for conf_level in confidence_levels:
            alpha = 1.0 - conf_level
            z_score = stats.norm.ppf(1 - alpha / 2)
            std_dev = np.sqrt(total_var)

            intervals[f'{conf_level:.0%}'] = {
                'lower': point_est - z_score * std_dev,
                'upper': point_est + z_score * std_dev,
                'width': 2 * z_score * std_dev
            }

        return intervals

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'backcast_length': self.backcast_length,
            'forecast_length': self.forecast_length,
            'stack_types': self.stack_types,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'thetas_dim': self.thetas_dim,
            'share_weights_in_stack': self.share_weights_in_stack,
            'hidden_layer_units': self.hidden_layer_units,
            'num_mixtures': self.num_mixtures,
            'mdn_hidden_units': self.mdn_hidden_units,
            'aggregation_mode': self.aggregation_mode,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ProbabilisticNBeatsNet':
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------

def create_probabilistic_nbeats_model(
        config: Optional[Dict[str, Any]] = None,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        learning_rate: float = 0.001,
        **kwargs: Any
) -> ProbabilisticNBeatsNet:
    """Create a compiled Probabilistic N-BEATS model.

    Args:
        config: Configuration dictionary for the model.
        optimizer: Optimizer to use for training.
        learning_rate: Learning rate for the optimizer.
        **kwargs: Additional arguments passed to ProbabilisticNBeatsNet.

    Returns:
        Compiled Probabilistic N-BEATS model.

    Example:
        >>> model = create_probabilistic_nbeats_model(
        ...     config={
        ...         'backcast_length': 96,
        ...         'forecast_length': 24,
        ...         'num_mixtures': 5
        ...     },
        ...     learning_rate=0.001
        ... )
    """
    # Default configuration
    default_config = {
        'backcast_length': 48,
        'forecast_length': 12,
        'stack_types': ['trend', 'seasonality'],
        'nb_blocks_per_stack': 2,
        'thetas_dim': [3, 6],
        'hidden_layer_units': 128,
        'num_mixtures': 3,
        'mdn_hidden_units': 64,
        'aggregation_mode': 'concat',
        'share_weights_in_stack': False,
        'input_dim': 1,
        'output_dim': 1,
    }

    if config is not None:
        default_config.update(config)
    default_config.update(kwargs)

    # Create model
    model = ProbabilisticNBeatsNet(**default_config)

    # Setup optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.get(optimizer)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=model.mdn_loss,
        metrics=['mae']  # Additional metric for monitoring
    )

    logger.info(f"Created Probabilistic N-BEATS with {default_config['num_mixtures']} mixtures")
    logger.info(f"Aggregation mode: {default_config['aggregation_mode']}")

    return model

# ---------------------------------------------------------------------