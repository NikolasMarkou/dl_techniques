"""A Radial Basis Function (RBF) layer with center repulsion.

This layer implements a layer of Radial Basis Function units, which are
powerful for function approximation and pattern recognition tasks. Unlike
standard sigmoidal neurons, RBF units respond to localized regions of the
input space, making them effective at learning local features.

Architecture and Mathematical Foundation:
    The core of the RBF layer is a set of units, each with a 'center'
    vector that has the same dimensionality as the input. The activation of
    each unit is determined by the proximity of the input vector to its
    center. This relationship is formalized by the Gaussian RBF function:

    φᵢ(x) = exp(-γᵢ ||x - cᵢ||²)

    Where:
    - `x` is the input vector.
    - `cᵢ` is the center vector of the i-th RBF unit.
    - `γᵢ` is the trainable width (or precision) parameter for the i-th unit.
      It controls the radius of influence, or the "receptiveness," of the
      neuron. A larger gamma results in a more localized, narrower response.
    - `||·||²` denotes the squared Euclidean distance.

    The output of the layer is a vector where each element is the activation
    `φᵢ(x)` from the corresponding RBF unit.

Enhanced Center Repulsion:
    A common issue when training RBF networks with gradient descent is
    "center collapse," where multiple centers converge to the same location,
    leading to redundant units and poor coverage of the input space. This
    implementation mitigates this with an adaptive repulsion mechanism.

    During training, a penalty term is added to the model's loss, which
    activates only when two centers `cᵢ` and `cⱼ` become too close. The
    repulsion potential `V_rep` is defined as:

    V_rep(cᵢ, cⱼ) = α · D · max(0, d_min·(1 + μ) - ||cᵢ - cⱼ||)²

    - The force is proportional to the `repulsion_strength` (α) and is only
      applied when the distance `||cᵢ - cⱼ||` falls below a threshold defined
      by the `min_center_distance` (d_min) and a `safety_margin` (μ).
    - Crucially, the penalty is scaled by the input dimensionality (D). This
      makes the `repulsion_strength` hyperparameter more stable and less
      dependent on the number of features in the input data.

References:
    - Moody, J., & Darken, C. J. (1989). "Fast learning in networks of
      locally-tuned processing units." This is a foundational paper that
      established RBF networks as a competitive architecture, often trained
      with a hybrid approach of unsupervised center selection followed by
      supervised weight training.
    - Bishop, C. M. (1995). "Neural Networks for Pattern Recognition." This
      textbook provides a comprehensive theoretical treatment of RBF
      networks, detailing their properties as universal approximators and
      their connection to techniques like kernel density estimation.
    - Schwenker, F., Kestler, H. A., & Palm, G. (2001). "Three learning
      phases for radial-basis-function networks." This work reviews and
      analyzes different strategies for training RBF networks, including
      fully supervised methods where centers are adapted via gradient
      descent, which is the approach this layer facilitates.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RBFLayer(keras.layers.Layer):
    """
    Radial Basis Function layer with stable center repulsion mechanism.

    This layer implements RBF units with an improved repulsive force mechanism
    between centers to ensure better coverage of the input space and prevent
    center collapse. The implementation includes numerical stability improvements,
    adaptive repulsion strength, and robust parameter initialization.

    Mathematical Foundation:
        For an input x, each RBF unit computes:
            φᵢ(x) = exp(-γᵢ ||x - cᵢ||²)

        Where:
        - cᵢ is the center vector for unit i
        - γᵢ is the width parameter (precision = 1/2σ²)
        - ||·||² denotes squared Euclidean distance

    Enhanced Center Repulsion:
        To prevent center collapse, we use an adaptive repulsion mechanism:
            Vᵣₑₚ(cᵢ, cⱼ) = α * D * max(0, d_min * (1 + μ) - ||cᵢ - cⱼ||)²

        Where:
        - α is the base repulsion strength
        - D is the input dimensionality (for scaling)
        - μ is a safety margin
        - d_min is the minimum desired distance between centers

    Key Features:
    - Stable numerical implementation with bounded exponentials
    - Dimensionality-adaptive repulsion strength scaling
    - Robust parameter initialization and constraints
    - Full serialization support for production deployment
    - Comprehensive error handling and validation

    Args:
        units: Integer, number of RBF units in the layer. Must be positive.
        gamma_init: Float, initial value for the width parameter (1/2σ²).
            Controls the initial spread of RBF functions. Must be positive.
            Defaults to 1.0.
        repulsion_strength: Float, base weight of the repulsion term (α).
            Controls how strongly centers repel each other. Must be non-negative.
            Higher values encourage more spread-out centers. Defaults to 0.1.
        min_center_distance: Float, minimum desired distance between centers.
            Must be positive. Defaults to 1.0.
        center_initializer: String or Initializer, initializer for RBF centers.
            Accepts standard Keras initializer names ('uniform', 'normal', etc.)
            or Initializer instances. Defaults to 'uniform'.
        center_constraint: Optional constraint for center positions.
            Can be used to bound centers within specific regions.
            Defaults to None.
        trainable_gamma: Boolean, whether width parameters are trainable.
            If False, gamma values remain fixed at initialization.
            Defaults to True.
        safety_margin: Float, additional margin for minimum distance calculation.
            Must be non-negative. Defaults to 0.2.
        kernel_regularizer: Optional regularizer for centers.
            Applied to center positions during training. Defaults to None.
        gamma_regularizer: Optional regularizer for width parameters.
            Applied to gamma values during training. Defaults to None.
        **kwargs: Additional Layer base class arguments.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)` or
        3D tensor with shape: `(batch_size, timesteps, input_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, units)` or
        3D tensor with shape: `(batch_size, timesteps, units)`

    Attributes:
        centers: Weight matrix of shape `(units, input_dim)` containing center positions.
        gamma_raw: Raw width parameters of shape `(units,)` before softplus transformation.
        gamma: Effective positive width parameters computed via softplus.

    Example:
        ```python
        # Basic usage
        layer = RBFLayer(units=32)
        inputs = keras.Input(shape=(784,))
        outputs = layer(inputs)

        # Advanced configuration
        layer = RBFLayer(
            units=64,
            gamma_init=2.0,
            repulsion_strength=0.2,
            min_center_distance=1.5,
            center_initializer='normal',
            center_constraint=keras.constraints.UnitNorm(),
            trainable_gamma=True,
            safety_margin=0.3,
            gamma_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a complete model
        inputs = keras.Input(shape=(100,))
        x = RBFLayer(128, repulsion_strength=0.15)(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```

    References:
        - Moody, J., & Darken, C. J. (1989). Fast learning in networks of
          locally-tuned processing units.
        - Schwenker, F., Kestler, H. A., & Palm, G. (2001). Three learning phases
          for radial-basis-function networks.
        - Bishop, C. M. (1995). Neural Networks for Pattern Recognition.

    Raises:
        ValueError: If units is not positive.
        ValueError: If gamma_init is not positive.
        ValueError: If repulsion_strength is negative.
        ValueError: If min_center_distance is not positive.
        ValueError: If safety_margin is negative.
        ValueError: If input has less than 2 dimensions.

    Note:
        The layer automatically adds repulsion loss during training to prevent
        center collapse. This loss is scaled by input dimensionality for
        hyperparameter stability across different datasets.
    """

    def __init__(
        self,
        units: int,
        gamma_init: float = 1.0,
        repulsion_strength: float = 0.1,
        min_center_distance: float = 1.0,
        center_initializer: Union[str, keras.initializers.Initializer] = 'uniform',
        center_constraint: Optional[keras.constraints.Constraint] = None,
        trainable_gamma: bool = True,
        safety_margin: float = 0.2,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        gamma_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the RBF layer with enhanced parameters."""
        super().__init__(**kwargs)

        # Comprehensive input validation
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if gamma_init <= 0:
            raise ValueError(f"gamma_init must be positive, got {gamma_init}")
        if repulsion_strength < 0:
            raise ValueError(f"repulsion_strength must be non-negative, got {repulsion_strength}")
        if min_center_distance <= 0:
            raise ValueError(f"min_center_distance must be positive, got {min_center_distance}")
        if safety_margin < 0:
            raise ValueError(f"safety_margin must be non-negative, got {safety_margin}")

        # Store ALL configuration parameters for serialization
        self.units = units
        self.gamma_init = gamma_init
        self.repulsion_strength = repulsion_strength
        self.min_center_distance = min_center_distance
        self.safety_margin = safety_margin
        self.trainable_gamma = trainable_gamma

        # Process initializers, constraints, and regularizers using standard Keras `get`
        self.center_initializer = keras.initializers.get(center_initializer)
        self.center_constraint = keras.constraints.get(center_constraint)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)

        # Initialize weight attributes - created in build()
        self.centers = None
        self.gamma_raw = None
        self._feature_dim = None  # Store for repulsion scaling

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights with robust initialization.

        Args:
            input_shape: Shape tuple indicating the input shape.

        Raises:
            ValueError: If input_shape has less than 2 dimensions or
                       if last dimension is None.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {len(input_shape)}"
            )

        # Extract and validate feature dimension
        self._feature_dim = input_shape[-1]
        if self._feature_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Create center positions with shape (units, feature_dim)
        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, self._feature_dim),
            initializer=self.center_initializer,
            constraint=self.center_constraint,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Create width parameters using raw values transformed via softplus
        # Initialize raw values such that softplus(raw) = gamma_init
        # Since softplus(x) = log(1 + exp(x)), we need x = log(exp(gamma_init) - 1)
        gamma_raw_init_value = ops.log(
            ops.exp(ops.cast(self.gamma_init, self.compute_dtype)) - 1.0
        )

        self.gamma_raw = self.add_weight(
            name='gamma_raw',
            shape=(self.units,),
            initializer=keras.initializers.Constant(gamma_raw_init_value),
            regularizer=self.gamma_regularizer,
            trainable=self.trainable_gamma,
        )

        # Call parent build
        super().build(input_shape)

    def _compute_pairwise_distances(
        self,
        x: keras.KerasTensor,
        y: keras.KerasTensor,
        epsilon: float = 1e-12
    ) -> keras.KerasTensor:
        """
        Compute pairwise squared Euclidean distances with numerical stability.

        Uses the identity ||x-y||² = ||x||² - 2⟨x,y⟩ + ||y||² for efficiency.

        Args:
            x: Tensor of shape `(n, d)`
            y: Tensor of shape `(m, d)`
            epsilon: Small constant for numerical stability

        Returns:
            Tensor of shape `(n, m)` containing squared distances
        """
        # Compute squared norms
        x_norm = ops.sum(ops.square(x), axis=-1, keepdims=True)  # (n, 1)
        y_norm = ops.sum(ops.square(y), axis=-1, keepdims=True)  # (m, 1)

        # Compute cross term: -2 * x @ y^T
        cross_term = -2.0 * ops.matmul(x, ops.transpose(y))

        # Combine: ||x||² - 2⟨x,y⟩ + ||y||²
        distances_squared = x_norm + cross_term + ops.transpose(y_norm) + epsilon

        # Ensure non-negative (handles numerical errors)
        return ops.maximum(distances_squared, 0.0)

    def _compute_repulsion(self, centers: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute enhanced repulsion regularization term.

        Encourages centers to maintain minimum distance apart, scaled by
        input dimensionality for hyperparameter stability.

        Args:
            centers: Tensor of shape `(units, feature_dim)` representing centers

        Returns:
            Scalar tensor representing repulsion loss to be added
        """
        # Effective minimum distance with safety margin
        effective_min_dist = self.min_center_distance * (1.0 + self.safety_margin)

        # Compute pairwise distances between centers
        squared_distances = self._compute_pairwise_distances(centers, centers)
        distances = ops.sqrt(squared_distances)

        # Compute repulsion potential: max(0, d_min_eff - ||c_i - c_j||)²
        repulsion_potential = ops.square(
            ops.maximum(0.0, effective_min_dist - distances)
        )

        # Mask diagonal elements (self-repulsion)
        mask = 1.0 - ops.eye(self.units, dtype=self.compute_dtype)

        # Mean repulsion over all pairs (excluding self)
        mean_repulsion = ops.mean(repulsion_potential * mask)

        # Scale by dimensionality and strength
        dim_scale = ops.cast(self._feature_dim, self.compute_dtype)
        repulsion_loss = self.repulsion_strength * dim_scale * mean_repulsion

        return repulsion_loss

    @property
    def gamma(self) -> keras.KerasTensor:
        """
        Effective positive gamma values via softplus transformation.

        Returns:
            Tensor of shape `(units,)` with positive width parameters
        """
        return keras.activations.softplus(self.gamma_raw)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with numerical stability and repulsion loss.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim)` or
                   `(batch_size, timesteps, input_dim)`
            training: Boolean indicating training mode

        Returns:
            Output tensor with RBF activations of shape `(batch_size, units)` or
            `(batch_size, timesteps, units)`
        """
        # Compute squared distances from inputs to centers
        distances_squared = self._compute_pairwise_distances(inputs, self.centers)

        # Expand gamma for broadcasting: (1, units)
        gamma_expanded = ops.expand_dims(self.gamma, 0)

        # Compute bounded exponent to prevent overflow
        # Cap at 50.0 to prevent numerical issues with exp()
        bounded_exponent = ops.minimum(gamma_expanded * distances_squared, 50.0)

        # Compute RBF activations: exp(-γ||x-c||²)
        activations_output = ops.exp(-bounded_exponent)

        # Add repulsion loss during training, only if there are multiple centers
        if training and self.units > 1:
            repulsion_loss = self._compute_repulsion(self.centers)
            self.add_loss(repulsion_loss)

        return activations_output

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape based on input shape.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            Output shape with last dimension replaced by units
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return complete layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'gamma_init': self.gamma_init,
            'repulsion_strength': self.repulsion_strength,
            'min_center_distance': self.min_center_distance,
            'safety_margin': self.safety_margin,
            'trainable_gamma': self.trainable_gamma,
            'center_initializer': keras.initializers.serialize(self.center_initializer),
            'center_constraint': keras.constraints.serialize(self.center_constraint),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
        })
        return config

    # Convenience properties for inspection
    @property
    def center_positions(self) -> Optional[keras.KerasTensor]:
        """Get current positions of RBF centers."""
        return self.centers

    @property
    def width_values(self) -> Optional[keras.KerasTensor]:
        """Get current effective width (gamma) values."""
        return self.gamma if self.built else None

# ---------------------------------------------------------------------
