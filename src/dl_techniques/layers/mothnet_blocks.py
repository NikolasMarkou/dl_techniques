"""
MothNet: Bio-Mimetic Feature Generation for Few-Shot Learning.

This module provides an implementation of MothNet, a computational
model of the insect olfactory network designed to excel at machine learning
tasks with limited training data. Its primary function is to serve as a
powerful, automatic feature generator that can be prepended to any standard
ML classifier, creating a hybrid "insect cyborg" model with significantly
enhanced performance.

Core Concept: The "Insect Cyborg"
----------------------------------
The central idea is to address the "limited data" problem by leveraging the
remarkable data efficiency of biological neural networks. Moths can learn to
identify new odors from just a few exposures. MothNet mimics the key
architectural principles that enable this rapid learning, extracting rich,
"orthogonal" class-relevant information that conventional ML methods often miss.

When these bio-mimetic features are concatenated with the original data and
fed into a standard classifier (like an SVM or a simple Neural Net), the
resulting "cyborg" model demonstrates a dramatic reduction in test error
(20-60% reported in original research) and a significant decrease in data
requirements (e.g., matching 100-sample performance with only 30 samples).

Key Architectural Components
----------------------------
The MothNet architecture is a feed-forward cascade of three specialized layers,
each inspired by a specific region of the insect brain:

1.  **AntennalLobeLayer (AL)**: Contrast Enhancement
    -   **Biological Analogy**: The Antennal Lobe, the first olfactory
        processing center.
    -   **Mechanism**: Implements competitive inhibition, where neurons suppress
        each other's activity. This creates a "winner-take-more" dynamic that
        sharpens the input signal, enhances contrast, and suppresses noise.

2.  **MushroomBodyLayer (MB)**: High-Dimensional Sparse Coding
    -   **Biological Analogy**: The Mushroom Body, a center for associative
        learning and memory.
    -   **Mechanism**: Projects the sharpened signal from the AL into a much
        higher-dimensional space using sparse, random connections. It then
        enforces sparse firing via a top-k winner-take-all mechanism. This
        transformation untangles complex patterns and creates unique, robust,
        and highly discriminative combinatorial codes for each input.

3.  **HebbianReadoutLayer**: Associative Learning
    -   **Biological Analogy**: Synaptic plasticity in readout neurons.
    -   **Mechanism**: Uses a local, correlation-based Hebbian learning rule
        ("fire together, wire together") instead of backpropagation. Weights
        are strengthened based on the co-occurrence of pre-synaptic (MB) and
        post-synaptic (target class) activity. This forms direct associations
        between the sparse MB codes and their corresponding classes.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AntennalLobeLayer(keras.layers.Layer):
    """Antennal Lobe layer implementing competitive inhibition for contrast enhancement.

    This layer enhances contrast through a "winner-take-more" dynamic that
    sharpens the input signal and suppresses noise, mimicking competitive
    dynamics in the insect antennal lobe. The operation computes
    ``h = W * x + b`` (excitation), ``mu = mean(h)`` (global inhibition),
    then ``y = activation(h - alpha * mu)`` (competitive response), where
    ``alpha`` controls inhibition strength.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input [batch, features]     │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Linear: h = W·x + b        │
        └─────────────┬────────────────┘
                      │
              ┌───────┴───────┐
              │               │
              ▼               ▼
        ┌───────────┐  ┌───────────┐
        │     h     │  │ mu=mean(h)│
        └─────┬─────┘  └─────┬─────┘
              │               │
              ▼               ▼
        ┌──────────────────────────────┐
        │  h_inhibited = h - alpha*mu  │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Activation(h_inhibited)     │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Output [batch, units]       │
        └──────────────────────────────┘

    :param units: Number of output units.
    :type units: int
    :param inhibition_strength: Strength of lateral inhibition (alpha).
        Range [0, 1]. Defaults to 0.5.
    :type inhibition_strength: float
    :param activation: Activation function after inhibition. Defaults to ``'relu'``.
    :type activation: str
    :param kernel_initializer: Initializer for excitatory weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for excitatory weights. Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether to include bias terms. Defaults to True.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        units: int,
        inhibition_strength: float = 0.5,
        activation: str = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.inhibition_strength = inhibition_strength
        self.activation_fn = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create layer weights for excitatory connections.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Create excitatory connection weights
        self.excitatory_weights = self.add_weight(
            name='excitatory_weights',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply competitive inhibition to inputs.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API compatibility.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, units)``.
        :rtype: keras.KerasTensor
        """
        # Excitatory response: linear transformation
        excitation = keras.ops.matmul(inputs, self.excitatory_weights)

        if self.use_bias:
            excitation = excitation + self.bias

        # Global inhibition: subtract scaled mean activity
        mean_activity = keras.ops.mean(excitation, axis=-1, keepdims=True)
        inhibition = mean_activity * self.inhibition_strength

        # Combined competitive response
        output = excitation - inhibition

        # Apply nonlinearity
        output = self.activation_fn(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape for this layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'inhibition_strength': self.inhibition_strength,
            'activation': keras.activations.serialize(self.activation_fn),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_bias': self.use_bias,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MushroomBodyLayer(keras.layers.Layer):
    """Mushroom Body layer implementing high-dimensional sparse random projection.

    This layer projects inputs into a much higher-dimensional space using
    sparse, random connections, then enforces sparse firing through top-k
    winner-take-all selection. The sparse projection
    ``h = W_sparse * x + b`` (with ~10% non-zero weights) expands the
    feature space, and ``output = TopK(activation(h), k)`` where
    ``k = floor(sparsity * units)`` creates a combinatorial code that is
    both robust to noise and highly discriminative, providing massive
    representational capacity (e.g., ``C(4000, 400) ~ 10^459`` patterns).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input from AL [batch, input_dim]│
        └─────────────┬────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  Sparse Random Projection        │
        │  h = W_sparse·x + b             │
        │  (~10% non-zero connections)     │
        │  (input_dim → units, 20-50x)    │
        └─────────────┬────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  Activation(h)                   │
        └─────────────┬────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  Top-K Winner-Take-All           │
        │  keep k = sparsity * units       │
        │  (rest set to 0)                 │
        └─────────────┬────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  Sparse Output [batch, units]    │
        │  (~90% zeros)                    │
        └──────────────────────────────────┘

    :param units: Number of output units (typically 20-50x input dimension).
    :type units: int
    :param sparsity: Fraction of neurons that fire per input. Range (0, 1).
        Defaults to 0.1.
    :type sparsity: float
    :param connection_sparsity: Sparsity of random projection matrix.
        Range (0, 1). Defaults to 0.1.
    :type connection_sparsity: float
    :param activation: Activation function before top-k. Defaults to ``'relu'``.
    :type activation: str
    :param kernel_initializer: Initializer for non-zero projection weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param trainable_projection: Whether projection matrix is trainable.
        Defaults to False.
    :type trainable_projection: bool
    :param use_bias: Whether to include bias terms. Defaults to True.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        units: int,
        sparsity: float = 0.1,
        connection_sparsity: float = 0.1,
        activation: str = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        trainable_projection: bool = False,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.sparsity = sparsity
        self.connection_sparsity = connection_sparsity
        self.activation_fn = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.trainable_projection = trainable_projection
        self.use_bias = use_bias
        self.k = int(units * sparsity)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create sparse random projection matrix.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Initialize full weight matrix
        weights = keras.ops.convert_to_numpy(
            self.kernel_initializer(shape=(input_dim, self.units))
        )

        # Create sparse connection mask (fixed random pattern)
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        self.projection_mask = rng.rand(input_dim, self.units) < self.connection_sparsity

        # Apply sparsity mask
        weights = weights * self.projection_mask

        self.projection_weights = self.add_weight(
            name='projection_weights',
            shape=(input_dim, self.units),
            initializer=keras.initializers.Constant(weights),
            trainable=self.trainable_projection,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply sparse high-dimensional projection with winner-take-all.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API compatibility.
        :type training: Optional[bool]
        :return: Sparse output tensor of shape ``(batch_size, units)``.
        :rtype: keras.KerasTensor
        """
        # Apply sparse random projection
        output = keras.ops.matmul(inputs, self.projection_weights)

        if self.use_bias:
            output = output + self.bias

        # Apply activation
        output = self.activation_fn(output)

        # Enforce sparse firing via top-k winner-take-all
        top_k_values, top_k_indices = keras.ops.top_k(output, k=self.k)

        # Create sparse output (only top-k neurons fire, rest are zero)
        sparse_output = keras.ops.zeros_like(output)

        # Scatter top-k values back to their positions
        batch_size = keras.ops.shape(output)[0]
        batch_indices = keras.ops.repeat(
            keras.ops.arange(batch_size)[:, None],
            repeats=self.k,
            axis=1
        )

        # Stack batch and feature indices for scatter operation
        indices = keras.ops.stack([
            keras.ops.reshape(batch_indices, [-1]),
            keras.ops.reshape(top_k_indices, [-1])
        ], axis=1)

        # Scatter top-k values into sparse tensor
        sparse_output = keras.ops.scatter_nd_update(
            sparse_output,
            indices,
            keras.ops.reshape(top_k_values, [-1])
        )

        return sparse_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape for this layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'sparsity': self.sparsity,
            'connection_sparsity': self.connection_sparsity,
            'activation': keras.activations.serialize(self.activation_fn),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'trainable_projection': self.trainable_projection,
            'use_bias': self.use_bias,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HebbianReadoutLayer(keras.layers.Layer):
    """Hebbian readout layer implementing local correlation-based learning.

    This layer uses Hebbian plasticity ("neurons that fire together, wire
    together") rather than backpropagation for weight updates. The forward
    pass computes ``logits = W * x + b`` (standard linear transformation).
    During training, weights are updated via the Hebbian rule:
    ``dW = alpha * (x^T * y) / batch_size``, where ``x`` are sparse MB
    activations and ``y`` are one-hot target labels. Sparsity in the MB
    codes is critical -- it ensures selective associations between specific
    neuron patterns and their corresponding classes.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  MB Input [batch, mb_units]      │
        └─────────────┬────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  Linear: logits = W·x + b       │
        │  (W updated via Hebbian rule)    │
        └─────────────┬────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │  Class Logits [batch, units]     │
        └──────────────────────────────────┘

        Hebbian Update (training):
        dW = alpha * (x^T · y) / N

    :param units: Number of output units (number of classes).
    :type units: int
    :param learning_rate: Hebbian learning rate (alpha). Defaults to 0.01.
    :type learning_rate: float
    :param kernel_initializer: Initializer for readout weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for readout weights. Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether to include bias terms. Defaults to True.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        units: int,
        learning_rate: float = 0.01,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create readout weights.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self.readout_weights = self.add_weight(
            name='readout_weights',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False,  # Updated via Hebbian rule, not backprop
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,  # Bias can be trained normally or via Hebbian updates
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute readout activations (forward pass).

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API compatibility.
        :type training: Optional[bool]
        :return: Class logits of shape ``(batch_size, units)``.
        :rtype: keras.KerasTensor
        """
        output = keras.ops.matmul(inputs, self.readout_weights)

        if self.use_bias:
            output = output + self.bias

        return output

    def hebbian_update(
        self,
        pre_synaptic: keras.KerasTensor,
        post_synaptic: keras.KerasTensor
    ) -> None:
        """Apply Hebbian weight update rule: ``dW = alpha * (x^T * y) / batch_size``.

        :param pre_synaptic: Pre-synaptic activations (MB output) of shape
            ``(batch_size, input_dim)``.
        :type pre_synaptic: keras.KerasTensor
        :param post_synaptic: Post-synaptic activations (one-hot targets) of shape
            ``(batch_size, units)``.
        :type post_synaptic: keras.KerasTensor
        """
        # Compute batch-averaged outer product: (1/N) · x^T · y
        batch_size = keras.ops.shape(pre_synaptic)[0]
        weight_update = keras.ops.matmul(
            keras.ops.transpose(pre_synaptic),
            post_synaptic
        ) / keras.ops.cast(batch_size, dtype=pre_synaptic.dtype)

        # Apply Hebbian update: W_new = W_old + α·ΔW
        new_weights = self.readout_weights + self.learning_rate * weight_update
        self.readout_weights.assign(new_weights)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape for this layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_bias': self.use_bias,
        })
        return config

# ---------------------------------------------------------------------

