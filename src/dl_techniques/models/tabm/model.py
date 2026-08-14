"""
TabM: a parameter-efficient batched MLP ensemble for tabular data, with integrated
categorical encoding and selectable diversity mechanisms.

Deep ensembles are the reliable way to improve a neural network on tabular data:
train `k` models from different initializations and average them. The improvement is
real but the cost is literal — `k` times the parameters, `k` times the training runs,
`k` times the inference. TabM asks how much of the ensemble's benefit survives if the
members are forced to share almost all of their weights, and answers that most of it
does, provided the shared computation is perturbed per member in the right places.

The mechanism is a rank-1 multiplicative perturbation of a shared kernel. A single
weight matrix `W` is used by every member; member `i` scales the layer's input by a
learned vector `r_i` and its output by a learned vector `s_i`, so

`y_i = s_i * ((r_i * x) W) + b_i`

which is exactly `x (diag(r_i) W diag(s_i)) + b_i` — a distinct effective weight
matrix per member, expressed in `d_in + d_out` parameters instead of
`d_in * d_out`. The members are genuinely different functions, not a shared function
with different readouts, and they cost a vector each. That is the whole idea; the
rest of the architecture is how far the perturbation is pushed.

`arch_type` selects that, and the class attribute `ARCH_SPECS` is the single table
that says what each value builds — every field in it is read by `_create_layers`,
and no two rows are equal, so each name denotes a genuinely different model.

Under `'plain'` there is no ensemble at all: `k` must be `None`, the backbone is an
ordinary MLP and the head an ordinary `Dense` — the baseline the ensemble variants
are measured against. Under `'tabm'` every backbone layer is a
`LinearEfficientEnsemble` with per-member input and output scaling, and the head is
`NLinear`, which is `k` fully independent output matrices; the head is small enough
that sharing it is not worth the loss of diversity where diversity matters most.
`'tabm-normal'` is that same architecture with the scaling vectors drawn from
`N(1, 0.1)` instead of random signs. Random signs is the default because it
guarantees members start at distinct, unit-magnitude perturbations; a normal draw
clusters them near the identity, which is the point of offering both.

`'tabm-packed'` drops the rank-1 trick entirely and gives every backbone layer `k`
independent kernels (`NLinear`). It costs `k` times the backbone parameters and is
the honest deep-ensemble upper bound the efficient variants are measured against —
without it in the table, "efficient ensembling recovers most of the benefit" has no
denominator.

Under `'tabm-mini'` the backbone weights are shared with no per-layer perturbation
and *all* diversity comes from a single `ScaleEnsemble` adapter applied to the input,
one learned scale vector per member — the cheapest possible ensemble, and the
interesting limit case, since every member thereafter computes the same function of a
differently-scaled input. `'tabm-mini-normal'` is the same with the adapter
initialized from a normal distribution instead of random signs.

Ensembling is done by carrying an explicit member axis through the whole network:
tensors are `(batch, k, features)` and every layer is written to broadcast over that
middle axis, so all `k` members are evaluated in one pass of dense kernels rather
than in a Python loop. How the batch reaches that shape is governed by
`share_training_batches`. When true (and always at inference), one batch is tiled `k`
ways so every member sees identical rows — the members differ only through their
perturbations. When false, an incoming batch of `B * k` rows is *reshaped* so each
member gets a disjoint slice, which decorrelates members through data as well as
through weights. That second path requires the caller to have prepared the batch at
`k` times the nominal size; the model does not resample for you, and a batch that is
not a multiple of `k` will not reshape correctly.

Preprocessing is inside the model rather than upstream: numerical features pass
through unchanged and categorical features are one-hot encoded against declared
cardinalities, then concatenated. This keeps the whole pipeline differentiable and
serializable as one object, at the cost of an input width that grows with the sum of
cardinalities — high-cardinality columns are the case where an external embedding
would be preferable.

`call` returns per-member predictions of shape `(batch, k, d_out)`, deliberately
un-aggregated; the `'plain'` path expands to `(batch, 1, d_out)` so the output rank
does not depend on the architecture. Aggregation is a separate, explicit step —
`predict_with_uncertainty` returns `ŷ = (1/k) Σ fᵢ(x)` together with the
across-member standard deviation, and the module-level `ensemble_predict` returns the
mean alone. It is deliberately not folded into `call`, because the training loss
needs the per-member outputs and because the right reduction differs by task. The
spread is a disagreement statistic over whatever `call` emits (logits, for a
classifier), not a calibrated predictive variance.

Six preset variants scale hidden widths and member count together, from `[64, 32]`
with `k=4` up to `[2048, 1024, 512, 256]` with `k=32`. The two smallest default to
`'tabm-mini'` and the rest to `'tabm'`, on the reasoning that at small widths the
per-layer scaling vectors are a large fraction of the parameter budget.

References:
    - Gorishniy et al., 2024. TabM: Advancing Tabular Deep Learning with Parameter-
      Efficient Ensembling. (https://arxiv.org/abs/2410.24210)
    - Lakshminarayanan et al., 2017. Simple and Scalable Predictive Uncertainty
      Estimation using Deep Ensembles. (https://arxiv.org/abs/1612.01474)
    - Wen et al., 2020. BatchEnsemble: An Alternative Approach to Efficient Ensemble
      and Lifelong Learning. (https://arxiv.org/abs/2002.06715)
    - Gorishniy et al., 2021. Revisiting Deep Learning Models for Tabular Data.
      (https://arxiv.org/abs/2106.11959)
    - Grinsztajn et al., 2022. Why do tree-based models still outperform deep
      learning on typical tabular data? (https://arxiv.org/abs/2207.08815)
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
from dl_techniques.layers.tabm_blocks import ScaleEnsemble, NLinear, TabMBackbone

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TabMModel(keras.Model):
    """TabM: Deep Ensemble Architecture for High-Performance Tabular Learning.

    TabM provides state-of-the-art ensemble methods specifically designed for tabular
    data, addressing key challenges through efficient ensemble training, integrated
    preprocessing, and parameter-efficient diversity mechanisms. The architecture
    offers competitive performance with gradient boosting while maintaining neural
    network flexibility and end-to-end differentiability.

    **Intent**: Provide a production-ready tabular learning solution that combines
    the performance benefits of ensemble methods with the efficiency of batched
    training, offering both high accuracy and principled uncertainty estimation
    for critical tabular ML applications.

    **Architecture**:
    Four-stage pipeline integrating preprocessing, ensemble diversity, backbone
    computation, and prediction aggregation:
    - **Preprocessing**: Seamless numerical/categorical feature handling with one-hot encoding
    - **Ensemble Layer**: Architecture-dependent diversity mechanisms (ScaleEnsemble, etc.)
    - **Backbone**: Configurable MLP with ensemble-aware or shared weight strategies
    - **Output**: Standard Dense or NLinear for ensemble-specific prediction heads

    **Component Details**:
    - **Efficient Ensembling**: Batched k-member training without k-fold computational cost
    - **Mixed Data Support**: Integrated numerical passthrough and categorical encoding
    - **Diversity Mechanisms**: Multiple strategies from input-level to full perturbations
    - **Uncertainty Quantification**: `call` emits the un-aggregated member axis and
      quantifies nothing on its own; `predict_with_uncertainty` is the explicit
      opt-in that reduces it to a mean and an across-member spread

    Args:
        n_num_features: Integer, number of numerical input features.
            Must be non-negative. Set to 0 if only categorical features present.
        cat_cardinalities: List of integers, cardinalities for each categorical feature.
            Empty list if no categorical features. Each cardinality must be positive.
        n_classes: Integer or None, number of output classes for classification.
            Set to None for regression tasks. Binary classification should use n_classes=2.
        hidden_dims: List of integers, dimensions for each hidden layer.
            Must be non-empty with all positive values. Defines backbone architecture.
        arch_type: String, architecture variant selector. The row of `ARCH_SPECS`
            it names is what `_create_layers` reads; no two rows are equal.
            - 'plain': Standard MLP, no member axis (k must be None)
            - 'tabm': Efficient ensemble, per-layer rank-1 scaling, random-signs init
            - 'tabm-normal': As 'tabm' but the scaling vectors init from N(1, 0.1)
            - 'tabm-packed': k fully independent backbone kernels (NLinear); costs
              k times the backbone parameters, no rank-1 scaling
            - 'tabm-mini': Shared un-perturbed backbone; all diversity comes from a
              single input-side ScaleEnsemble, random-signs init
            - 'tabm-mini-normal': As 'tabm-mini' but the adapter inits from N(1, 0.1)
        k: Integer or None, number of ensemble members.
            Required for ensemble variants ('tabm', 'tabm-mini', etc.), must be positive.
            Must be None for 'plain' architecture.
        activation: String, activation function name for hidden layers.
            Standard Keras activation names supported.
        dropout_rate: Float, dropout probability applied after each hidden layer.
            Must be between 0 and 1. Set to 0.0 to disable dropout.
        use_bias: Boolean, whether to use bias terms in linear transformations.
        share_training_batches: Boolean, batch sharing strategy for ensemble training.
            If True, all ensemble members see same batch (efficient).
            If False, each member sees different batch subset (more diverse).
        kernel_initializer: Initializer for linear layer weights.
            Can be string name or Initializer instance.
        bias_initializer: Initializer for bias terms.
            Can be string name or Initializer instance.
        kernel_regularizer: Optional regularizer for linear layer kernels.
        bias_regularizer: Optional regularizer for bias terms.
        name: Optional string name for the model.
        **kwargs: Additional keyword arguments for the Model base class.

    Input format:
        Supports multiple input formats for flexibility:
        - Tuple: (x_numerical, x_categorical)
        - Dictionary: {'x_num': x_numerical, 'x_cat': x_categorical}
        - Single tensor: x_numerical (when no categorical features)

    Output format:
        Tensor with shape:
        - Plain: (batch_size, 1, n_classes_or_1) for consistency
        - Ensemble: (batch_size, k, n_classes_or_1) for ensemble predictions

    Attributes:
        Architecture configuration parameters as stored attributes.
        Layer instances: cat_encoder, minimal_ensemble_adapter, backbone, output_layer.

    Raises:
        AssertionError: If n_num_features < 0 or invalid architecture configuration.
        AssertionError: If k requirements don't match arch_type specifications.
        ValueError: If no valid features provided or invalid input format.

    Example:
        ```python
        # Multi-class classification with mixed features
        model = TabMModel(
            n_num_features=10,
            cat_cardinalities=[5, 3, 12],
            n_classes=4,
            hidden_dims=[256, 128],
            arch_type='tabm',
            k=8
        )

        # Regression with minimal ensemble
        model = TabMModel(
            n_num_features=20,
            cat_cardinalities=[],
            n_classes=None,      # Regression
            hidden_dims=[512, 256, 128],
            arch_type='tabm-mini',
            k=4
        )

        # Baseline plain MLP
        model = TabMModel(
            n_num_features=15,
            cat_cardinalities=[8, 4],
            n_classes=2,
            hidden_dims=[128, 64],
            arch_type='plain'    # No ensemble
        )
        ```

    Note:
        For models, Keras automatically handles sub-layer building, so no custom
        build() method is needed. The model can be compiled and trained directly
        after instantiation. `call` returns the raw member axis; use
        `predict_with_uncertainty` (or the module-level `ensemble_predict`) to
        reduce it to a point estimate and a spread.
    """

    # What each arch_type actually builds. Every field here is read by
    # `_create_layers`; an entry that changed nothing would be a lie about an
    # ablation, so there is deliberately no row whose fields duplicate another's.
    #
    #   ensemble_type     - 'efficient' (one shared kernel + rank-1 per-member
    #                       scaling) or 'packed' (k independent kernels)
    #   backbone_scaling  - whether the efficient backbone perturbs per member
    #   backbone_init     - init of the backbone's per-member scaling vectors
    #   adapter_init      - init of the input-side ScaleEnsemble, or None for
    #                       no adapter
    ARCH_SPECS: Dict[str, Dict[str, Any]] = {
        'plain': {
            'ensemble_type': 'efficient',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': None,
        },
        'tabm': {
            'ensemble_type': 'efficient',
            'backbone_scaling': True,
            'backbone_init': 'random-signs',
            'adapter_init': None,
        },
        'tabm-normal': {
            'ensemble_type': 'efficient',
            'backbone_scaling': True,
            'backbone_init': 'normal',
            'adapter_init': None,
        },
        'tabm-packed': {
            'ensemble_type': 'packed',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': None,
        },
        'tabm-mini': {
            'ensemble_type': 'efficient',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': 'random-signs',
        },
        'tabm-mini-normal': {
            'ensemble_type': 'efficient',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': 'normal',
        },
    }

    # Model variant configurations optimized for different dataset scales
    MODEL_VARIANTS = {
        "micro": {
            "hidden_dims": [64, 32],
            "k": 4,
            "arch_type": "tabm-mini",
            "description": "Minimal tabular model - 8K params, small datasets"
        },
        "tiny": {
            "hidden_dims": [128, 64],
            "k": 8,
            "arch_type": "tabm-mini",
            "description": "Lightweight ensemble - 32K params, medium datasets"
        },
        "small": {
            "hidden_dims": [256, 128],
            "k": 8,
            "arch_type": "tabm",
            "description": "Standard configuration - 128K params, most datasets"
        },
        "base": {
            "hidden_dims": [512, 256, 128],
            "k": 8,
            "arch_type": "tabm",
            "description": "High-performance default - 512K params, large datasets"
        },
        "large": {
            "hidden_dims": [1024, 512, 256],
            "k": 16,
            "arch_type": "tabm",
            "description": "Large ensemble - 2M params, very large datasets"
        },
        "xlarge": {
            "hidden_dims": [2048, 1024, 512, 256],
            "k": 32,
            "arch_type": "tabm",
            "description": "XL ensemble - 8M params, massive datasets"
        }
    }

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
            name: Optional[str] = "tabm_model",
            **kwargs: Any
    ) -> None:
        """Initialize the TabM model.

        Args:
            n_num_features: Number of numerical features.
            cat_cardinalities: List of cardinalities for categorical features.
            n_classes: Number of output classes (None for regression).
            hidden_dims: List of hidden layer dimensions.
            arch_type: Architecture variant selector.
            k: Number of ensemble members (required for ensemble variants).
            activation: Activation function for hidden layers.
            dropout_rate: Dropout rate for regularization.
            use_bias: Whether to use bias terms.
            share_training_batches: Batch sharing strategy for ensemble training.
            kernel_initializer: Initializer for weights.
            bias_initializer: Initializer for bias terms.
            kernel_regularizer: Regularizer for weights.
            bias_regularizer: Regularizer for bias terms.
            name: Model name.
            **kwargs: Additional Model arguments.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate arguments
        self._validate_parameters(n_num_features, cat_cardinalities, hidden_dims, arch_type, k, share_training_batches, dropout_rate)

        # Store configuration
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities.copy() if cat_cardinalities else []
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims.copy()
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
        self.d_cat = sum(cat_cardinalities) if cat_cardinalities else 0
        self.d_flat = self.d_num + self.d_cat

        # Create all sub-layers (following modern Keras 3 patterns)
        self._create_layers()

        # Initialize the Model (Keras handles building automatically)
        super().__init__(name=name, **kwargs)

        logger.info(
            f"Initialized TabM model: arch_type={arch_type}, "
            f"k={k}, dims={hidden_dims}, features=({n_num_features} num + {len(cat_cardinalities)} cat)"
        )

    def _validate_parameters(
            self,
            n_num_features: int,
            cat_cardinalities: List[int],
            hidden_dims: List[int],
            arch_type: str,
            k: Optional[int],
            share_training_batches: bool,
            dropout_rate: float
    ) -> None:
        """Validate initialization parameters.

        Args:
            n_num_features: Number of numerical features.
            cat_cardinalities: Categorical feature cardinalities.
            hidden_dims: Hidden layer dimensions.
            arch_type: Architecture type.
            k: Number of ensemble members.
            share_training_batches: Batch sharing flag.
            dropout_rate: Dropout rate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if n_num_features < 0:
            raise ValueError(
                f"n_num_features must be non-negative, got {n_num_features}"
            )
        if not (n_num_features or cat_cardinalities):
            raise ValueError(
                "Must have either numerical or categorical features"
            )

        if cat_cardinalities and not all(c > 0 for c in cat_cardinalities):
            raise ValueError("All cardinalities must be positive")

        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if not all(d > 0 for d in hidden_dims):
            raise ValueError("All hidden dimensions must be positive")

        if arch_type not in self.ARCH_SPECS:
            raise ValueError(
                f"Unknown arch_type {arch_type!r}. Available: "
                f"{sorted(self.ARCH_SPECS)}"
            )

        if arch_type == 'plain':
            if k is not None:
                raise ValueError("Plain architecture must have k=None")
            if not share_training_batches:
                raise ValueError(
                    "Plain architecture must use share_training_batches=True"
                )
        else:
            if k is None:
                raise ValueError(
                    f"Ensemble architecture {arch_type} requires k to be specified"
                )
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")

        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

    def _create_layers(self) -> None:
        """Create all model layers with proper initialization."""
        # Categorical encoding
        if self.cat_cardinalities:
            self.cat_encoder = OneHotEncoding(self.cat_cardinalities)
        else:
            self.cat_encoder = None

        spec = self.ARCH_SPECS[self.arch_type]

        # Minimal ensemble adapter for tabm-mini variants
        if spec['adapter_init'] is not None:
            self.minimal_ensemble_adapter = ScaleEnsemble(
                k=self.k,
                input_dim=self.d_flat,
                init_distribution=spec['adapter_init'],
                kernel_regularizer=self.kernel_regularizer
            )
        else:
            self.minimal_ensemble_adapter = None

        # Backbone MLP
        backbone_k = None if self.arch_type == 'plain' else self.k
        self.backbone = TabMBackbone(
            hidden_dims=self.hidden_dims,
            k=backbone_k,
            ensemble_type=spec['ensemble_type'],
            ensemble_scaling_in=spec['backbone_scaling'],
            ensemble_scaling_out=spec['backbone_scaling'],
            init_distribution=spec['backbone_init'],
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
    ) -> keras.KerasTensor:
        """Forward pass through the TabM model implementing ensemble computation.

        This method handles the complete TabM pipeline:
        1. Input format normalization (tuple/dict to tensors)
        2. Feature preprocessing (numerical passthrough + categorical one-hot)
        3. Feature concatenation into unified representation
        4. Ensemble dimension handling (batched k-member computation)
        5. Ensemble diversity application (architecture-dependent)
        6. Backbone MLP forward pass
        7. Output layer prediction generation

        Args:
            inputs: Input data in multiple supported formats:
                - Tuple: (x_numerical, x_categorical)
                - Dict: {'x_num': x_numerical, 'x_cat': x_categorical}
                - Single tensor: x_numerical (no categorical features)
            training: Boolean training mode flag for dropout/batch norm behavior.

        Returns:
            Predictions with shape:
            - Plain: (batch_size, 1, n_classes_or_1) for consistency
            - Ensemble: (batch_size, k, n_classes_or_1) for k ensemble members

        Raises:
            ValueError: If no valid features provided or invalid input format.
        """
        # Handle different input formats for flexibility
        if isinstance(inputs, dict):
            x_num = inputs.get('x_num')
            x_cat = inputs.get('x_cat')
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            x_num, x_cat = inputs
        else:
            # Single tensor input - assume all numerical
            x_num = inputs
            x_cat = None

        # Process features with integrated preprocessing pipeline
        features = []

        # Numerical features (direct passthrough)
        if x_num is not None and self.n_num_features > 0:
            features.append(x_num)

        # Categorical features (one-hot encoding)
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

        # Handle ensemble dimensions for batched computation
        if self.k is not None:
            batch_size = ops.shape(x)[0]

            if self.share_training_batches or not training:
                # Shared batch strategy: (B, D) -> (B, K, D)
                x = ops.expand_dims(x, axis=1)  # (B, 1, D)
                x = ops.tile(x, [1, self.k, 1])  # (B, K, D)
            else:
                # Independent batch strategy: (B * K, D) -> (B, K, D)
                # Note: Requires careful batch preparation externally
                x = ops.reshape(x, (batch_size // self.k, self.k, -1))

            # Apply minimal ensemble adapter if present (tabm-mini variants)
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)

        # Backbone MLP forward pass (ensemble-aware)
        x = self.backbone(x, training=training)

        # Output layer (architecture-specific prediction heads)
        x = self.output_layer(x)

        # Adjust output shape for consistency across architectures
        if self.k is None:
            # Plain: (B, D) -> (B, 1, D) for consistency with ensemble outputs
            x = ops.expand_dims(x, axis=1)

        return x

    def predict_with_uncertainty(
            self,
            x_data: Union[Tuple[Any, Any], Dict[str, Any], Any],
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate the member axis into a point estimate and a spread.

        This is the explicit opt-in for the aggregation `call` deliberately does
        not perform. `call` returns `(batch, k, d_out)`; this returns the mean
        `ŷ = (1/k) Σ fᵢ(x)` and the across-member standard deviation, each of
        shape `(batch, d_out)`.

        The spread is the ensemble's disagreement, which is an *epistemic*
        signal only. It is computed on whatever `call` emits — for a classifier
        that is logits, not probabilities, so it is not a calibrated predictive
        variance and should not be read as one. For `arch_type='plain'` there is
        one member and the spread is identically zero.

        Args:
            x_data: Input data in any format `call` accepts.
            **kwargs: Forwarded to `keras.Model.predict`.

        Returns:
            Tuple `(mean, std)`, both shaped `(batch_size, d_out)`.
        """
        predictions = self.predict(x_data, **kwargs)
        return np.mean(predictions, axis=1), np.std(predictions, axis=1)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int] = None,
        **kwargs: Any
    ) -> "TabMModel":
        """Create a TabM model from a predefined variant.

        Args:
            variant: String, one of "micro", "tiny", "small", "base", "large", "xlarge"
            n_num_features: Integer, number of numerical features
            cat_cardinalities: List of cardinalities for categorical features
            n_classes: Integer or None, number of output classes (None for regression)
            **kwargs: Additional arguments passed to the constructor

        Returns:
            TabMModel instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Small ensemble for medium datasets
            >>> model = TabMModel.from_variant(
            ...     "small",
            ...     n_num_features=15,
            ...     cat_cardinalities=[5, 3, 8],
            ...     n_classes=3
            ... )
            >>>
            >>> # Large ensemble for big tabular data
            >>> model = TabMModel.from_variant(
            ...     "large",
            ...     n_num_features=100,
            ...     cat_cardinalities=[20, 15, 10, 5],
            ...     n_classes=None  # Regression
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        logger.info(f"Creating TabM-{variant.upper()} model")
        logger.info(f"Configuration: {config['description']}")

        return cls(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes,
            **config,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
        """
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
    def from_config(cls, config: Dict[str, Any]) -> "TabMModel":
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            TabMModel instance.
        """
        # Handle serialized objects
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "bias_initializer" in config and isinstance(config["bias_initializer"], dict):
            config["bias_initializer"] = keras.initializers.deserialize(
                config["bias_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "bias_regularizer" in config and config["bias_regularizer"]:
            config["bias_regularizer"] = keras.regularizers.deserialize(
                config["bias_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with TabM-specific information.

        Args:
            **kwargs: Additional keyword arguments for summary.
        """
        super().summary(**kwargs)
        logger.info("=" * 60)
        logger.info("TabM: Deep Ensemble Tabular Model")
        logger.info("=" * 60)
        logger.info("Architecture Configuration:")
        logger.info(f"  - Architecture type: {self.arch_type}")
        logger.info(f"  - Ensemble members: {self.k if self.k else 'N/A (plain)'}")
        logger.info(f"  - Hidden layers: {self.hidden_dims}")
        logger.info(f"  - Total parameters: {self.count_params():,}")

        logger.info("\nFeature Configuration:")
        logger.info(f"  - Numerical features: {self.n_num_features}")
        logger.info(f"  - Categorical features: {len(self.cat_cardinalities)}")
        logger.info(f"  - Categorical cardinalities: {self.cat_cardinalities}")
        logger.info(f"  - Total input dimension: {self.d_flat}")

        logger.info("\nTask Configuration:")
        task_type = f"{self.n_classes}-class classification" if self.n_classes else "Regression"
        logger.info(f"  - Task type: {task_type}")
        logger.info(f"  - Activation: {self.activation}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Shared batches: {self.share_training_batches}")

        if self.k:
            logger.info("\nEnsemble Benefits:")
            logger.info("  - Improved robustness through ensemble diversity")
            logger.info("  - Uncertainty estimation via prediction variance")
            logger.info("  - Parameter-efficient batched training")

    def __repr__(self) -> str:
        """Return string representation of the model.

        Returns:
            String representation including key parameters.
        """
        return (
            f"TabMModel(arch_type='{self.arch_type}', k={self.k}, "
            f"hidden_dims={self.hidden_dims}, features=({self.n_num_features}, {len(self.cat_cardinalities)}), "
            f"name='{self.name}')"
        )

# ---------------------------------------------------------------------
# Factory functions for easy model creation
# ---------------------------------------------------------------------

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
        **kwargs: Any
) -> TabMModel:
    """Create a TabM model with specified configuration.

    Factory function providing convenient access to TabM architecture with
    sensible defaults optimized for typical tabular learning scenarios.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        arch_type: Architecture variant selector.
        k: Number of ensemble members.
        activation: Activation function for hidden layers.
        dropout_rate: Dropout rate for regularization.
        use_bias: Whether to use bias terms.
        share_training_batches: Batch sharing strategy.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias terms.
        **kwargs: Additional model arguments.

    Returns:
        Configured TabM model ready for compilation and training.

    Example:
        >>> # Binary classification with mixed features
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

# ---------------------------------------------------------------------

def create_tabm_plain(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: List[int] = [256, 256],
        **kwargs: Any
) -> TabMModel:
    """Create a plain MLP baseline without ensembling.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        Plain MLP model for baseline comparison.
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

# ---------------------------------------------------------------------

def create_tabm_ensemble(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs: Any
) -> TabMModel:
    """Create a TabM model with full efficient ensemble.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        k: Number of ensemble members.
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        Full TabM ensemble model with LinearEfficientEnsemble.
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

# ---------------------------------------------------------------------

def create_tabm_mini(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs: Any
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
        TabM-mini model with ScaleEnsemble input diversity.
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

# ---------------------------------------------------------------------

def ensemble_predict(
        model: TabMModel,
        x_data: Union[Tuple, Dict, Any],
        method: Literal['mean', 'best', 'greedy'] = 'mean'
) -> np.ndarray:
    """Make predictions using ensemble model with aggregation.

    Provides different ensemble aggregation strategies for balancing
    performance and computational requirements.

    Args:
        model: Trained TabM model with ensemble capability.
        x_data: Input data in supported format.
        method: Aggregation strategy:
            - 'mean': Simple ensemble averaging (recommended)
            - 'best': Best single member (requires validation data)
            - 'greedy': Greedy selection (requires validation data)

    Returns:
        Aggregated predictions with uncertainty estimates.
        Shape: (batch_size, n_outputs) for final predictions.

    Note:
        'best' and 'greedy' methods require external validation data
        to determine optimal aggregation and currently fall back to 'mean'.
    """
    # Get ensemble predictions: (batch_size, k, n_outputs)
    predictions = model.predict(x_data)

    if method == 'mean':
        # Simple ensemble averaging - most robust approach
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
        raise ValueError(f"Unknown aggregation method: {method}")

# ---------------------------------------------------------------------

def create_tabm_for_dataset(
        X_train: np.ndarray,
        y_train: np.ndarray,
        categorical_indices: Optional[List[int]] = None,
        categorical_cardinalities: Optional[List[int]] = None,
        arch_type: str = 'tabm',
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs: Any
) -> TabMModel:
    """Create a TabM model automatically configured for a specific dataset.

    Analyzes dataset characteristics to determine optimal model configuration,
    including problem type detection and feature organization.

    Args:
        X_train: Training features array with shape (n_samples, n_features).
        y_train: Training labels/targets array.
        categorical_indices: List of indices for categorical features in X_train.
        categorical_cardinalities: List of cardinalities for categorical features.
        arch_type: Architecture variant to use.
        k: Number of ensemble members.
        hidden_dims: Hidden layer architecture.
        **kwargs: Additional model arguments.

    Returns:
        Configured TabM model optimized for the dataset.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.models.tabm.model import create_tabm_for_dataset

        >>> # Generate sample tabular data
        >>> X_train = np.random.randn(1000, 15)  # 15 features
        >>> y_train = np.random.randint(0, 3, 1000)  # 3-class classification

        >>> # Specify categorical features (first 3 columns)
        >>> categorical_indices = [0, 1, 2]
        >>> categorical_cardinalities = [5, 3, 8]

        >>> model = create_tabm_for_dataset(
        ...     X_train, y_train,
        ...     categorical_indices=categorical_indices,
        ...     categorical_cardinalities=categorical_cardinalities,
        ...     arch_type='tabm',
        ...     k=8
        ... )
        >>>
        >>> # Model ready for compilation and training
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    """
    # Determine problem type from target distribution
    if len(y_train.shape) == 1:
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            # Binary classification
            n_classes = 2
            problem_type = "Binary classification"
        elif len(unique_labels) > 2 and np.all(unique_labels == np.arange(len(unique_labels))):
            # Multi-class classification
            n_classes = len(unique_labels)
            problem_type = f"{n_classes}-class classification"
        else:
            # Regression
            n_classes = None
            problem_type = "Regression"
    else:
        # Multi-output classification or regression
        n_classes = y_train.shape[1]
        problem_type = f"Multi-output ({n_classes} outputs)"

    # Determine feature organization
    if categorical_indices is None:
        categorical_indices = []
    if categorical_cardinalities is None:
        categorical_cardinalities = []

    n_total_features = X_train.shape[1]
    n_categorical = len(categorical_indices)
    n_numerical = n_total_features - n_categorical

    # Log dataset analysis
    logger.info("Dataset Analysis for TabM Configuration:")
    logger.info(f"  - Total samples: {X_train.shape[0]:,}")
    logger.info(f"  - Total features: {n_total_features}")
    logger.info(f"  - Numerical features: {n_numerical}")
    logger.info(f"  - Categorical features: {n_categorical}")
    if categorical_cardinalities:
        logger.info(f"  - Categorical cardinalities: {categorical_cardinalities}")
    logger.info(f"  - Problem type: {problem_type}")
    logger.info(f"  - Architecture: {arch_type} with k={k}")

    return create_tabm_model(
        n_num_features=n_numerical,
        cat_cardinalities=categorical_cardinalities,
        n_classes=n_classes,
        arch_type=arch_type,
        k=k,
        hidden_dims=hidden_dims,
        **kwargs
    )

# ---------------------------------------------------------------------