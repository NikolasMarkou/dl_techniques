"""
Probabilistic N-BEATS Model with MDN Integration.

This module combines the N-BEATS architecture with Mixture Density Networks
to provide probabilistic time series forecasting with uncertainty quantification.
"""

import keras
import numpy as np
from keras import ops
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
    define a joint probability distribution over the entire forecast horizon.

    Key Features:
        - Maintains N-BEATS interpretable structure (trend, seasonality)
        - Provides uncertainty quantification for multi-step forecasts
        - Models a joint distribution over the forecast_length horizon
        - Enables risk assessment and confidence intervals

    Architecture:
        1. N-BEATS blocks process input and generate intermediate forecasts
        2. Block outputs are aggregated into rich feature representations
        3. A final MLP maps features to mixture distribution parameters
        4. The MDNLayer outputs parameters for a mixture of Gaussians, where
           the dimensionality of each Gaussian is equal to `forecast_length`.

    Args:
        backcast_length: Integer, length of the input time series window.
        forecast_length: Integer, length of the forecast horizon. This also
                         defines the output dimensionality of the MDN.
        input_dim: Integer, dimensionality of input time series (default: 1).
        stack_types: List of strings, types of stacks ('generic', 'trend', 'seasonality').
        nb_blocks_per_stack: Integer, number of blocks per stack.
        thetas_dim: List of integers, theta dimensionality for each stack.
        share_weights_in_stack: Boolean, whether to share weights within stacks.
        hidden_layer_units: Integer, number of hidden units in FC layers of blocks.
        num_mixtures: Integer, number of Gaussian mixtures for MDN.
        mdn_hidden_units: Integer, number of hidden units in the final MLP.
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
            backcast_length: int = 10,
            forecast_length: int = 1,
            input_dim: int = 1,
            stack_types: List[str] = ['trend', 'seasonality'],
            nb_blocks_per_stack: int = 3,
            thetas_dim: List[int] = [4, 8],
            share_weights_in_stack: bool = False,
            hidden_layer_units: int = 256,
            num_mixtures: int = 3,
            mdn_hidden_units: int = 128,
            aggregation_mode: str = 'concat',
            diversity_regularizer_strength: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.input_dim = input_dim
        # FIX: The output dimension of the probabilistic model is the forecast_length.
        # This is a critical change to ensure the MDN models a distribution
        # over the entire forecast sequence.
        self.output_dim = forecast_length
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
        self.diversity_regularizer_strength = diversity_regularizer_strength

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

        # Create N-BEATS blocks
        for stack_id, (stack_type, theta_dim) in enumerate(zip(self.stack_types, self.thetas_dim)):
            stack_blocks = []
            for block_id in range(self.nb_blocks_per_stack):
                block_name = f"stack_{stack_id}_block_{block_id}_{stack_type}"
                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(
                        units=self.hidden_layer_units, thetas_dim=theta_dim,
                        backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack, name=block_name
                    )
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(
                        units=self.hidden_layer_units, thetas_dim=theta_dim,
                        backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack, name=block_name
                    )
                else: # SeasonalityBlock
                    block = SeasonalityBlock(
                        units=self.hidden_layer_units, thetas_dim=theta_dim,
                        backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                        share_weights=self.share_weights_in_stack, name=block_name
                    )
                stack_blocks.append(block)
            self.blocks.append(stack_blocks)

        # Build aggregation and MDN components
        self._build_aggregation_layer()
        self._build_mdn_components()

        total_blocks = sum(len(stack) for stack in self.blocks)
        logger.info(f"Probabilistic N-BEATS built with {total_blocks} total blocks and MDN output dim {self.output_dim}")

    def _build_aggregation_layer(self) -> None:
        """Build the feature aggregation layer."""
        # FIX: Create a feature aggregator for all modes for design consistency.
        # This layer processes the aggregated N-BEATS forecasts before the final
        # MDN-specific layers.
        if self.aggregation_mode == 'attention':
            total_blocks = sum(len(stack) for stack in self.blocks)
            self.attention_weights = keras.layers.Dense(
                total_blocks,
                activation='softmax',
                name='attention_weights'
            )

        self.feature_aggregator = keras.layers.Dense(
            self.mdn_hidden_units,
            activation='relu',
            name='feature_aggregator'
        )

    def _build_mdn_components(self) -> None:
        """Build MDN preprocessing and output layers."""
        self.mdn_preprocessor = keras.layers.Dense(
            self.mdn_hidden_units,
            activation='relu',
            name='mdn_preprocessor'
        )

        # FIX: The output_dimension of the MDN layer must be the forecast_length.
        # This is the critical correction.
        self.mdn_layer = MDNLayer(
            output_dimension=self.forecast_length,
            num_mixtures=self.num_mixtures,
            name='mdn_output',
            diversity_regularizer_strength=self.diversity_regularizer_strength
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the probabilistic N-BEATS model."""
        if len(inputs.shape) == 3:
            inputs = ops.squeeze(inputs, axis=-1)
        elif len(inputs.shape) == 1:
            inputs = ops.expand_dims(inputs, axis=0)

        residual = inputs
        block_forecasts = []

        for stack_blocks in self.blocks:
            for block in stack_blocks:
                backcast, forecast = block(residual, training=training)
                residual = residual - backcast
                block_forecasts.append(forecast)

        aggregated_features = self._aggregate_forecasts(block_forecasts, training=training)
        mdn_features = self.mdn_preprocessor(aggregated_features, training=training)
        mixture_params = self.mdn_layer(mdn_features, training=training)

        return mixture_params

    def _aggregate_forecasts(
        self,
        block_forecasts: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Aggregate forecasts from all blocks into a feature representation."""
        if self.aggregation_mode == 'sum':
            aggregated = ops.stack(block_forecasts, axis=1)
            aggregated = ops.sum(aggregated, axis=1)
        elif self.aggregation_mode == 'concat':
            aggregated = ops.concatenate(block_forecasts, axis=-1)
        elif self.aggregation_mode == 'attention':
            stacked_forecasts = ops.stack(block_forecasts, axis=1)
            forecast_means = ops.mean(stacked_forecasts, axis=-1)
            attention_scores = self.attention_weights(forecast_means, training=training)
            attention_expanded = ops.expand_dims(attention_scores, axis=-1)
            weighted_forecasts = stacked_forecasts * attention_expanded
            aggregated = ops.sum(weighted_forecasts, axis=1)

        # FIX: Pass all aggregated outputs through the feature_aggregator
        # for consistent architecture across modes.
        return self.feature_aggregator(aggregated, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model."""
        batch_size = input_shape[0]
        mdn_output_size = (2 * self.num_mixtures * self.output_dim) + self.num_mixtures
        return (batch_size, mdn_output_size)

    def mdn_loss(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """MDN loss function for training."""
        return self.mdn_layer.loss_func(y_true, y_pred)

    def predict_probabilistic(
        self,
        x: np.ndarray,
        num_samples: int = 100,
        return_components: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate probabilistic predictions with uncertainty quantification."""
        mixture_params = self.predict(x)
        mu, sigma, pi_logits = self.mdn_layer.split_mixture_params(mixture_params)
        pi = keras.activations.softmax(pi_logits, axis=-1)

        mu_np = ops.convert_to_numpy(mu)
        sigma_np = ops.convert_to_numpy(sigma)
        pi_np = ops.convert_to_numpy(pi)

        pi_expanded = np.expand_dims(pi_np, axis=-1)
        point_estimates = np.sum(pi_expanded * mu_np, axis=1)

        samples_list = [ops.convert_to_numpy(self.mdn_layer.sample(mixture_params)) for _ in range(num_samples)]
        samples = np.stack(samples_list, axis=1)

        point_expanded = np.expand_dims(point_estimates, axis=1)
        aleatoric_variance = np.sum(pi_expanded * sigma_np**2, axis=1)
        epistemic_variance = np.sum(pi_expanded * (mu_np - point_expanded)**2, axis=1)
        total_variance = aleatoric_variance + epistemic_variance

        results = {
            'point_estimate': point_estimates,
            'samples': samples,
            'total_variance': total_variance,
            'aleatoric_variance': aleatoric_variance,
            'epistemic_variance': epistemic_variance,
        }

        if return_components:
            results['mixture_params'] = {'mu': mu_np, 'sigma': sigma_np, 'pi': pi_np}

        return results

    def get_prediction_intervals(
        self,
        x: np.ndarray,
        confidence_levels: List[float] = [0.68, 0.95]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Get prediction intervals for different confidence levels."""
        from scipy import stats
        predictions = self.predict_probabilistic(x)
        point_est, total_var = predictions['point_estimate'], predictions['total_variance']

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
            'diversity_regularizer_strength': self.diversity_regularizer_strength,
        })
        # Note: 'output_dim' is intentionally omitted as it's derived from 'forecast_length'
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
    """Create a compiled Probabilistic N-BEATS model."""
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
    }

    if config is not None:
        default_config.update(config)
    default_config.update(kwargs)

    model = ProbabilisticNBeatsNet(**default_config)

    if isinstance(optimizer, str):
        opt_class = keras.optimizers.get(optimizer)
        optimizer = opt_class(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=model.mdn_loss)

    logger.info(f"Created Probabilistic N-BEATS with {default_config['num_mixtures']} mixtures")
    logger.info(f"Aggregation mode: {default_config['aggregation_mode']}")

    return model

# ---------------------------------------------------------------------
