"""
PowerMLP: Complete Reference Guide and Algorithms
===============================================

1. Overview and Core Concepts
----------------------------
PowerMLP is an efficient alternative to Kolmogorov-Arnold Networks (KAN), offering:
- ~40x faster training time
- ~10x fewer FLOPs
- Equal or better performance
- Simpler implementation

Core Algorithm - PowerMLP Layer:
-------------------------------
Algorithm PowerMLPLayer(x, units, k):
    # Main branch - ReLU-k pathway
    main = Dense(x, units)
    main = max(0, main)^k

    # Basis function branch
    basis = x / (1 + exp(-x))  # Basis function
    basis = Dense(basis, units)

    return main + basis

2. Architecture Components
-------------------------
A. ReLU-k Activation:
   - Purpose: Replaces KAN's spline functions
   - Implementation:
     Algorithm ReLU_k(x, k):
         return max(0, x)^k
   - Typical k value: 3
   - Benefits: Efficient computation, strong approximation

B. Basis Function:
   - Purpose: Enhances expressiveness
   - Implementation:
     Algorithm BasisFunction(x):
         return x / (1 + exp(-x))
   - Benefits: Captures complex relationships

C. Full Network Structure:
   Algorithm PowerMLP_Forward(x, hidden_units, k):
       for units in hidden_units[:-1]:
           x = PowerMLPLayer(x, units, k)
       return Dense(x, hidden_units[-1])

3. Training and Optimization
---------------------------
A. Training Loop:
   Algorithm Train_PowerMLP(model, train_data, epochs):
       for epoch in 1 to epochs:
           for batch in train_data:
               pred = PowerMLP_Forward(batch.x)
               loss = ComputeLoss(pred, batch.y)
               grads = ComputeGradients(loss)
               UpdateParameters(grads)

B. Best Practices:
   - Use Adam optimizer
   - Start with k=3
   - Enable mixed precision training
   - Monitor both branch contributions

4. Improvements Over KAN
-----------------------
A. Computational Efficiency:
   - No recursive spline calculations
   - Simpler forward/backward passes
   - Better memory efficiency
   - More stable training dynamics

B. Performance Benefits:
   - Superior task performance
   - Better scalability
   - More stable convergence
   - Easier integration

5. Implementation Details
------------------------
A. Key Parameters:
   - k: Power for ReLU-k (default=3)
   - hidden_units: Layer dimensions
   - use_bias: Whether to use bias terms

B. Forward Pass Flow:
   Algorithm Layer_Forward(x):
       # 1. Split input
       main_branch = x
       basis_branch = x

       # 2. Process branches
       main = Dense(main_branch)
       main = ReLU_k(main)

       basis = BasisFunction(basis_branch)
       basis = Dense(basis)

       # 3. Combine results
       return main + basis

6. Theoretical Properties
------------------------
A. Function Approximation:
   - Superset of KAN functions on bounded intervals
   - Universal approximation capability
   - Better numerical stability

B. Expressiveness:
   ReLU-k captures high-order relationships:
   f(x) = Σ(w_i * ReLU(x)^k + b_i * basis(x))

7. Usage Guidelines
------------------
A. When to Use:
   - Complex function approximation
   - When KAN is too slow
   - Need better scalability
   - Require stable training

B. Architecture Design:
   Algorithm Design_Network(input_dim, output_dim):
       # Example network structure
       return PowerMLP(
           hidden_units=[
               4*input_dim,   # Expansion
               2*input_dim,   # Processing
               output_dim     # Output
           ],
           k=3
       )

8. Performance Optimization
--------------------------
A. Memory Efficiency:
   Algorithm Optimize_Memory():
       # 1. Use gradient checkpointing
       # 2. Enable mixed precision
       # 3. Optimize batch size
       # 4. Clear unused tensors

B. Computation Speed:
   Algorithm Speed_Optimization():
       # 1. Vectorize operations
       # 2. Use efficient matrix ops
       # 3. Batch similar shapes
       # 4. Cache repeated computations

9. Common Issues and Solutions
----------------------------
A. Training Issues:
   - Vanishing gradients with high k
     Solution: Start with k=3, adjust gradually

   - Numerical instability
     Solution: Use mixed precision, gradient clipping

B. Performance Issues:
   - Memory constraints
     Solution: Gradient checkpointing, smaller batches

   - Slow convergence
     Solution: Adjust learning rate, use warmup

10. Code Structure Example
-------------------------
class PowerMLP:
    def __init__(self, hidden_units, k=3):
        self.layers = [
            PowerMLPLayer(units, k)
            for units in hidden_units[:-1]
        ]
        self.output = Dense(hidden_units[-1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

Each PowerMLPLayer combines:
1. Main path: Dense → ReLU-k
2. Basis path: Basis → Dense
3. Addition of both paths

References:
----------
[1] "PowerMLP: An Efficient Version of KAN" (2024)
[2] "KAN: Kolmogorov-Arnold Networks" (2024)
"""

import keras
import tensorflow as tf
from typing import List, Optional, Union, Dict, Any


class ModelConfig:
    """Configuration class for PowerMLP hyperparameters.

    Args:
        hidden_units: List of integers specifying the number of units in each hidden layer
        k: Power for ReLU-k activation function
        weight_decay: L2 regularization factor
        use_bias: Whether to use bias in dense layers
        output_activation: Activation function for the output layer
    """

    def __init__(
            self,
            hidden_units: List[int],
            k: int = 3,
            weight_decay: float = 0.0001,
            use_bias: bool = True,
            output_activation: Optional[str] = None
    ):
        self.hidden_units = hidden_units
        self.k = k
        self.weight_decay = weight_decay
        self.use_bias = use_bias
        self.output_activation = output_activation


class ReLUK(keras.layers.Layer):
    """ReLU-k activation layer implementing f(x) = max(0,x)^k.

    Args:
        k: Power for ReLU function
        **kwargs: Additional layer arguments
    """

    def __init__(self, k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor after ReLU-k activation
        """
        return tf.pow(tf.nn.relu(inputs), self.k)

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer."""
        config = super().get_config()
        config.update({"k": self.k})
        return config


class BasisFunction(keras.layers.Layer):
    """Basis function layer implementing b(x) = x/(1 + e^(-x)).

    This layer implements the basis function branch of PowerMLP.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor after basis function
        """
        return inputs / (1 + tf.exp(-inputs))


class PowerMLPLayer(keras.layers.Layer):
    """Single layer of PowerMLP with ReLU-k activation and basis function.

    Args:
        units: Number of output units
        k: Power for ReLU-k activation
        weight_decay: L2 regularization factor
        use_bias: Whether to use bias in dense layers
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            units: int,
            k: int = 3,
            weight_decay: float = 0.0001,
            use_bias: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.k = k
        self.weight_decay = weight_decay
        self.use_bias = use_bias

        # L2 regularizer for both dense layers
        self.regularizer = keras.regularizers.L2(weight_decay)

        # Initialize layers
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer="he_normal",
            kernel_regularizer=self.regularizer,
            name="main_dense"
        )
        self.relu_k = ReLUK(k)
        self.basis = BasisFunction()
        self.basis_proj = keras.layers.Dense(
            units=units,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=self.regularizer,
            name="basis_dense"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor combining main and basis branches
        """
        # Main branch with ReLU-k
        main = self.dense(inputs)
        main = self.relu_k(main)

        # Basis function branch
        basis = self.basis(inputs)
        basis = self.basis_proj(basis)

        return main + basis

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "weight_decay": self.weight_decay,
            "use_bias": self.use_bias
        })
        return config


class PowerMLP(keras.Model):
    """Full PowerMLP model implementation.

    Args:
        config: ModelConfig instance containing model parameters
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build hidden layers
        self.hidden_layers = []
        for units in config.hidden_units[:-1]:
            self.hidden_layers.append(
                PowerMLPLayer(
                    units=units,
                    k=config.k,
                    weight_decay=config.weight_decay,
                    use_bias=config.use_bias
                )
            )

        # Output layer
        self.output_layer = keras.layers.Dense(
            units=config.hidden_units[-1],
            activation=config.output_activation,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(config.weight_decay),
            use_bias=config.use_bias
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x, training=training)
