"""
This module provides the `TabMModel`, a flexible and powerful Keras Model for tabular
data that integrates various efficient deep ensemble architectures.

The `TabMModel` serves as a high-level orchestrator, combining a preprocessing pipeline
for numerical and categorical features with a configurable MLP backbone that can
operate either as a standard single model or as a high-performance deep ensemble.
It is designed to be a versatile tool for tackling a wide range of tabular data
problems, from simple classification and regression to more complex tasks requiring
robust uncertainty estimation.

Architectural Variants (`arch_type`):

The model's behavior is primarily controlled by the `arch_type` parameter, which
selects from several pre-configured architectures:

-   **`'plain'` (Standard MLP):**
    -   This is a baseline, non-ensemble architecture. It functions as a standard
        Multi-Layer Perceptron, providing a reference point for evaluating the
        benefits of ensembling.

-   **`'tabm'` (Full Efficient Ensemble):**
    -   This is the main ensemble architecture. It uses the `LinearEfficientEnsemble`
        for its hidden layers, which employs a shared kernel with rank-1 perturbations
        to create diversity among the `k` ensemble members in a parameter-efficient way.
    -   The output layer uses `NLinear` to give each of the `k` members its own
        independent prediction head.

-   **`'tabm-mini'` (Minimal Ensemble Adapter):**
    -   This is a more lightweight ensemble variant. It uses a standard MLP backbone
        where all `k` members *share the exact same weights*.
    -   Diversity is introduced *only* at the input layer via the `ScaleEnsemble`
        adapter, which applies a unique, learnable scaling factor to the input features
        for each member. This is a very parameter-efficient way to create an ensemble,
        relying on the idea that slightly different initial inputs will lead the
        shared network down different optimization paths.

-   **`'tabm-packed'` / `'tabm-normal'` etc.:**
    -   These represent other potential variations, for instance, by changing the
        initialization distribution of the `ScaleEnsemble` adapter.

Key Features:

1.  **Integrated Preprocessing:** The model internally handles the one-hot encoding
    of categorical features, simplifying the data pipeline. It seamlessly combines
    numerical and categorical inputs into a single feature vector.

2.  **Batched Ensembling:** All ensemble variants operate in a "batched" or "implicit"
    manner. The `k` ensemble members are represented along an additional dimension
    in the tensors, allowing all members to be trained in parallel within a single
    model forward/backward pass.

3.  **Factory Functions and Helpers:** The module includes several factory functions
    (`create_tabm_plain`, `create_tabm_ensemble`, etc.) to simplify the instantiation
    of different model variants, as well as a helper function to automatically configure
    a model based on the properties of a given dataset.
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.one_hot_encoding import OneHotEncoding
from dl_techniques.layers.tabm_blocks import (
    ScaleEnsemble, NLinear, TabMBackbone)

# ---------------------------------------------------------------------

class TabMModel(keras.Model):
    """TabM (Tabular Model) implementation with various ensemble architectures.

    This model supports multiple architectures including:
    - Plain MLP
    - TabM with efficient ensemble
    - TabM-mini with minimal ensemble adapter
    - TabM-packed with packed ensemble

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        arch_type: Architecture type ('plain', 'tabm', 'tabm-mini', 'tabm-packed', etc.).
        k: Number of ensemble members (required for ensemble variants).
        activation: Activation function.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias.
        share_training_batches: Whether to share training batches across ensemble members.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional model arguments.
    """

    def __init__(
            self,
            n_num_features: int,
            cat_cardinalities: List[int],
            n_classes: Optional[int],
            hidden_dims: List[int],
            arch_type: Literal[
                'plain', 'tabm', 'tabm-mini', 'tabm-packed',
                'tabm-normal', 'tabm-mini-normal'
            ] = 'plain',
            k: Optional[int] = None,
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            share_training_batches: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate arguments
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities

        if arch_type == 'plain':
            assert k is None
            assert share_training_batches, 'Plain architecture must use share_training_batches=True'
        else:
            assert k is not None
            assert k > 0

        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.arch_type = arch_type
        self.k = k
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.share_training_batches = share_training_batches
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate input dimensions
        self.d_num = n_num_features
        self.d_cat = sum(cat_cardinalities)
        self.d_flat = self.d_num + self.d_cat

        # Build layers
        self._build_layers()

    def _build_layers(self) -> None:
        """Build all model layers with proper initialization."""

        # Categorical encoding
        if self.cat_cardinalities:
            self.cat_encoder = OneHotEncoding(self.cat_cardinalities)
        else:
            self.cat_encoder = None

        # Minimal ensemble adapter for tabm-mini variants
        if self.arch_type in ('tabm-mini', 'tabm-mini-normal'):
            init_distribution = 'normal' if self.arch_type == 'tabm-mini-normal' else 'random-signs'
            self.minimal_ensemble_adapter = ScaleEnsemble(
                k=self.k,
                input_dim=self.d_flat,
                init_distribution=init_distribution,
                kernel_regularizer=self.kernel_regularizer
            )
        else:
            self.minimal_ensemble_adapter = None

        # Backbone MLP
        backbone_k = None if self.arch_type == 'plain' else self.k
        self.backbone = TabMBackbone(
            hidden_dims=self.hidden_dims,
            k=backbone_k,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # Output layer
        d_out = 1 if self.n_classes is None else self.n_classes

        if self.arch_type == 'plain':
            self.output_layer = keras.layers.Dense(
                d_out,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='output'
            )
        else:
            self.output_layer = NLinear(
                n=self.k,
                input_dim=self.hidden_dims[-1],
                output_dim=d_out,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )

    def call(
            self,
            inputs: Union[Tuple[Any, Any], Dict[str, Any]],
            training: Optional[bool] = None
    ) -> Any:
        """Forward pass through the TabM model.

        Args:
            inputs: Input data, either as tuple (x_num, x_cat) or dict with keys 'x_num', 'x_cat'.
            training: Training mode flag.

        Returns:
            Model predictions with shape:
            - Plain: (batch_size, 1, n_classes_or_1)
            - Ensemble: (batch_size, k, n_classes_or_1)
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            x_num = inputs.get('x_num')
            x_cat = inputs.get('x_cat')
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            x_num, x_cat = inputs
        else:
            # Single tensor input - assume all numerical
            x_num = inputs
            x_cat = None

        # Process features
        features = []

        # Numerical features
        if x_num is not None and self.n_num_features > 0:
            features.append(x_num)

        # Categorical features
        if x_cat is not None and self.cat_cardinalities:
            cat_encoded = self.cat_encoder(x_cat)
            features.append(cat_encoded)

        # Combine features efficiently
        if len(features) == 0:
            raise ValueError("No valid features provided")
        elif len(features) == 1:
            x = features[0]
        else:
            x = ops.concatenate(features, axis=-1)

        # Handle ensemble dimensions
        if self.k is not None:
            batch_size = ops.shape(x)[0]

            if self.share_training_batches or not training:
                # (B, D) -> (B, K, D) using efficient operations
                x = ops.expand_dims(x, axis=1)  # (B, 1, D)
                x = ops.tile(x, [1, self.k, 1])  # (B, K, D)
            else:
                # (B * K, D) -> (B, K, D)
                # Note: In practice, this requires careful batch preparation
                x = ops.reshape(x, (batch_size // self.k, self.k, -1))

            # Apply minimal ensemble adapter if present
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)

        # Backbone forward pass
        x = self.backbone(x, training=training)

        # Output layer
        x = self.output_layer(x)

        # Adjust output shape for compatibility
        if self.k is None:
            # (B, D) -> (B, 1, D) for consistency with ensemble outputs
            x = ops.expand_dims(x, axis=1)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "n_num_features": self.n_num_features,
            "cat_cardinalities": self.cat_cardinalities,
            "n_classes": self.n_classes,
            "hidden_dims": self.hidden_dims,
            "arch_type": self.arch_type,
            "k": self.k,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "share_training_batches": self.share_training_batches,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TabMModel':
        """Create model from configuration."""
        return cls(**config)


# Factory functions remain unchanged but with enhanced type hints
def create_tabm_model(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: List[int] = [256, 256],
        arch_type: Literal[
            'plain', 'tabm', 'tabm-mini', 'tabm-packed',
            'tabm-normal', 'tabm-mini-normal'
        ] = 'tabm',
        k: Optional[int] = 8,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        share_training_batches: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        **kwargs
) -> TabMModel:
    """Create a TabM model with specified configuration.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        arch_type: Architecture type.
        k: Number of ensemble members.
        activation: Activation function.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias.
        share_training_batches: Whether to share training batches across ensemble members.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        **kwargs: Additional model arguments.

    Returns:
        Configured TabM model.

    Example:
        >>> # Binary classification with numerical and categorical features
        >>> model = create_tabm_model(
        ...     n_num_features=10,
        ...     cat_cardinalities=[5, 3, 8],
        ...     n_classes=2,
        ...     arch_type='tabm',
        ...     k=8
        ... )

        >>> # Regression with only numerical features
        >>> model = create_tabm_model(
        ...     n_num_features=15,
        ...     cat_cardinalities=[],
        ...     n_classes=None,
        ...     arch_type='tabm-mini',
        ...     k=4
        ... )
    """
    return TabMModel(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type=arch_type,
        k=k,
        activation=activation,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        share_training_batches=share_training_batches,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        **kwargs
    )


def create_tabm_plain(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a plain MLP without ensembling.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        Plain MLP model.
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='plain',
        k=None,
        **kwargs
    )


def create_tabm_ensemble(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a TabM model with efficient ensemble.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        k: Number of ensemble members.
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        TabM ensemble model.
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='tabm',
        k=k,
        **kwargs
    )


def create_tabm_mini(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a TabM-mini model with minimal ensemble adapter.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        k: Number of ensemble members.
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        TabM-mini model.
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='tabm-mini',
        k=k,
        **kwargs
    )


def ensemble_predict(
        model: TabMModel,
        x_data: Union[Tuple, Dict, Any],
        method: Literal['mean', 'best', 'greedy'] = 'mean'
) -> np.ndarray:
    """Make predictions using ensemble model with different aggregation methods.

    Args:
        model: Trained TabM model.
        x_data: Input data.
        method: Aggregation method ('mean', 'best', 'greedy').

    Returns:
        Aggregated predictions.
    """
    # Get ensemble predictions
    predictions = model.predict(x_data)  # (batch_size, k, n_outputs)

    if method == 'mean':
        # Simple ensemble averaging using ops for consistency
        return np.mean(predictions, axis=1)

    elif method == 'best':
        # Return predictions from the best single ensemble member
        # Note: This would require validation data to determine the best member
        logger.warning("Best member selection requires validation data. Using mean instead.")
        return np.mean(predictions, axis=1)

    elif method == 'greedy':
        # Greedy ensemble selection (simplified version)
        # Note: Full implementation would require validation data and iterative selection
        logger.warning("Greedy selection requires validation data. Using mean instead.")
        return np.mean(predictions, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")


def create_tabm_for_dataset(
        X_train: np.ndarray,
        y_train: np.ndarray,
        categorical_indices: Optional[List[int]] = None,
        categorical_cardinalities: Optional[List[int]] = None,
        arch_type: str = 'tabm',
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a TabM model configured for a specific dataset.

    Args:
        X_train: Training features.
        y_train: Training labels.
        categorical_indices: Indices of categorical features in X_train.
        categorical_cardinalities: Cardinalities of categorical features.
        arch_type: Architecture type.
        k: Number of ensemble members.
        hidden_dims: Hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        Configured TabM model.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.models.tabm import create_tabm_for_dataset

        >>> # Generate sample data
        >>> X_train = np.random.randn(1000, 15)  # 15 features
        >>> y_train = np.random.randint(0, 3, 1000)  # 3-class classification

        >>> # Assume first 3 features are categorical with cardinalities [5, 3, 8]
        >>> categorical_indices = [0, 1, 2]
        >>> categorical_cardinalities = [5, 3, 8]

        >>> model = create_tabm_for_dataset(
        ...     X_train, y_train,
        ...     categorical_indices=categorical_indices,
        ...     categorical_cardinalities=categorical_cardinalities,
        ...     arch_type='tabm',
        ...     k=8
        ... )
    """
    # Determine problem type
    if len(y_train.shape) == 1:
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            # Binary classification
            n_classes = 2
        elif len(unique_labels) > 2 and np.all(unique_labels == np.arange(len(unique_labels))):
            # Multiclass classification
            n_classes = len(unique_labels)
        else:
            # Regression
            n_classes = None
    else:
        # Multi-output classification
        n_classes = y_train.shape[1]

    # Determine feature splits
    if categorical_indices is None:
        categorical_indices = []
    if categorical_cardinalities is None:
        categorical_cardinalities = []

    n_total_features = X_train.shape[1]
    n_categorical = len(categorical_indices)
    n_numerical = n_total_features - n_categorical

    logger.info(f"Dataset configuration:")
    logger.info(f"  Total features: {n_total_features}")
    logger.info(f"  Numerical features: {n_numerical}")
    logger.info(f"  Categorical features: {n_categorical}")
    logger.info(f"  Problem type: {'Regression' if n_classes is None else f'{n_classes}-class classification'}")

    return create_tabm_model(
        n_num_features=n_numerical,
        cat_cardinalities=categorical_cardinalities,
        n_classes=n_classes,
        arch_type=arch_type,
        k=k,
        hidden_dims=hidden_dims,
        **kwargs
    )