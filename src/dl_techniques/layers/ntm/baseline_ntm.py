"""
Baseline Neural Turing Machine Implementation.

This module provides a production-ready implementation of the Neural Turing
Machine described in Graves et al., 2014, updated for Keras 3 compatibility
with robust serialization and graph-safe operations.

Classes:
    NTMMemory: External memory module with read/write operations.
    NTMReadHead: Read head with content/location addressing.
    NTMWriteHead: Write head with erase/add operations.
    NTMController: Configurable LSTM/GRU/feedforward controller.
    NTMCell: Single timestep NTM cell (RNN compatible).
    NeuralTuringMachine: Complete NTM layer with RNN wrapper.

Functions:
    create_ntm: Factory function to create NTM instances.
"""

import keras
from keras import ops
from keras import layers
from keras import initializers
from typing import Any, Literal

from .ntm_interface import (
    AddressingMode,
    BaseMemory,
    BaseHead,
    BaseController,
    BaseNTM,
    MemoryState,
    HeadState,
    NTMConfig,
    NTMOutput,
    cosine_similarity,
    circular_convolution,
    sharpen_weights,
)

# ---------------------------------------------------------------------
# NTMMemory
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMMemory(BaseMemory):
    """
    Standard NTM Memory Matrix.

    Manages read/write operations on the memory matrix using the
    erase-then-add mechanism from the original NTM paper.

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            epsilon=epsilon,
            **kwargs,
        )

    def initialize_state(self, batch_size: int) -> MemoryState:
        """
        Initialize memory state for a new sequence.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Initial memory state with near-zero memory values.
        :rtype: MemoryState
        """
        memory = ops.ones((batch_size, self.memory_size, self.memory_dim)) * self.epsilon
        usage = ops.zeros((batch_size, self.memory_size))
        return MemoryState(memory=memory, usage=usage)

    def read(
        self,
        memory_state: MemoryState,
        read_weights: Any,
    ) -> Any:
        """
        Read from memory using attention weights.

        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param read_weights: Attention weights of shape (batch, num_slots).
        :type read_weights: Any
        :return: Read vector of shape (batch, memory_dim).
        :rtype: Any
        """
        weights_expanded = ops.expand_dims(read_weights, axis=-1)
        read_vector = ops.sum(memory_state.memory * weights_expanded, axis=1)
        return read_vector

    def write(
        self,
        memory_state: MemoryState,
        write_weights: Any,
        erase_vector: Any,
        add_vector: Any,
    ) -> MemoryState:
        """
        Write to memory using erase-then-add mechanism.

        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param write_weights: Write attention weights of shape (batch, num_slots).
        :type write_weights: Any
        :param erase_vector: Erase vector of shape (batch, memory_dim), values in [0, 1].
        :type erase_vector: Any
        :param add_vector: Add vector of shape (batch, memory_dim).
        :type add_vector: Any
        :return: Updated memory state.
        :rtype: MemoryState
        """
        prev_memory = memory_state.memory

        # Expand dims for broadcasting
        weights_expanded = ops.expand_dims(write_weights, axis=-1)
        erase_expanded = ops.expand_dims(erase_vector, axis=1)
        add_expanded = ops.expand_dims(add_vector, axis=1)

        # Erase: M_t = M_{t-1} * (1 - w_t * e_t)
        erase_matrix = 1.0 - (weights_expanded * erase_expanded)
        erased_memory = prev_memory * erase_matrix

        # Add: M_t = M_{t-1} + w_t * a_t
        add_matrix = weights_expanded * add_expanded
        new_memory = erased_memory + add_matrix

        return MemoryState(
            memory=new_memory,
            usage=memory_state.usage,
            temporal_links=memory_state.temporal_links,
            precedence=memory_state.precedence,
        )

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        return super().get_config()


# ---------------------------------------------------------------------
# NTMReadHead
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMReadHead(BaseHead):
    """
    Standard NTM Read Head.

    Projects controller output to addressing parameters and computes
    attention weights for reading from memory.

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param addressing_mode: Type of addressing mechanism.
    :type addressing_mode: AddressingMode
    :param shift_range: Range of allowed shifts for location addressing.
    :type shift_range: int
    :param kernel_initializer: Initializer for dense layers.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: AddressingMode = AddressingMode.HYBRID,
        shift_range: int = 3,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            epsilon=epsilon,
            **kwargs,
        )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers in __init__ (Golden Rule)
        self.key_dense = layers.Dense(
            memory_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="key",
        )
        self.beta_dense = layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="beta",
        )
        self.gate_dense = layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="gate",
        )
        self.shift_dense = layers.Dense(
            shift_range,
            activation="softmax",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="shift",
        )
        self.gamma_dense = layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="gamma",
        )

    def build(self, input_shape: tuple) -> None:
        """
        Build sub-layers explicitly.

        :param input_shape: Shape of controller output (batch, controller_dim).
        :type input_shape: tuple
        """
        self.key_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        self.gate_dense.build(input_shape)
        self.shift_dense.build(input_shape)
        self.gamma_dense.build(input_shape)
        super().build(input_shape)

    def content_addressing(
        self,
        key: Any,
        beta: Any,
        memory: Any,
    ) -> Any:
        """
        Compute content-based attention weights.

        :param key: Key vector of shape (batch, 1, memory_dim).
        :type key: Any
        :param beta: Key strength of shape (batch, 1).
        :type beta: Any
        :param memory: Memory matrix of shape (batch, num_slots, memory_dim).
        :type memory: Any
        :return: Content weights of shape (batch, num_slots).
        :rtype: Any
        """
        similarity = cosine_similarity(key, memory, epsilon=self.epsilon)
        return ops.softmax(beta * similarity, axis=-1)

    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> tuple[Any, HeadState]:
        """
        Compute attention weights using the full addressing mechanism.

        :param controller_output: Output from controller, shape (batch, controller_dim).
        :type controller_output: Any
        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param prev_weights: Previous attention weights, shape (batch, num_slots).
        :type prev_weights: Any
        :return: Tuple of (new_weights, head_state).
        :rtype: tuple[Any, HeadState]
        """
        # 1. Project controller output to head parameters
        key = self.key_dense(controller_output)
        beta = self.beta_dense(controller_output)
        gate = self.gate_dense(controller_output)
        shift = self.shift_dense(controller_output)
        gamma = self.gamma_dense(controller_output) + 1.0

        # 2. Content Addressing
        key_expanded = ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(
            key_expanded, beta, memory_state.memory
        )

        # 3. Interpolation (Gating)
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Convolutional Shift
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpening
        final_weights = sharpen_weights(shifted_weights, gamma, epsilon=self.epsilon)

        new_state = HeadState(
            weights=final_weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma,
        )

        return final_weights, new_state

    def call(
        self,
        inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass (placeholder for layer compatibility).

        :param inputs: Input tensor.
        :type inputs: Any
        :param kwargs: Additional arguments.
        :return: Input unchanged (heads are called via compute_addressing).
        :rtype: Any
        """
        return inputs

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Compute output shape.

        :param input_shape: Shape of controller output.
        :type input_shape: tuple
        :return: Shape of attention weights.
        :rtype: tuple
        """
        return (input_shape[0], self.memory_size)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTMWriteHead
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMWriteHead(BaseHead):
    """
    Standard NTM Write Head.

    Includes erase and add vectors for memory modification.

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param addressing_mode: Type of addressing mechanism.
    :type addressing_mode: AddressingMode
    :param shift_range: Range of allowed shifts for location addressing.
    :type shift_range: int
    :param kernel_initializer: Initializer for dense layers.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        addressing_mode: AddressingMode = AddressingMode.HYBRID,
        shift_range: int = 3,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            memory_dim=memory_dim,
            addressing_mode=addressing_mode,
            shift_range=shift_range,
            epsilon=epsilon,
            **kwargs,
        )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Addressing parameters
        self.key_dense = layers.Dense(
            memory_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="key",
        )
        self.beta_dense = layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="beta",
        )
        self.gate_dense = layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="gate",
        )
        self.shift_dense = layers.Dense(
            shift_range,
            activation="softmax",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="shift",
        )
        self.gamma_dense = layers.Dense(
            1,
            activation="softplus",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="gamma",
        )

        # Write-specific parameters
        self.erase_dense = layers.Dense(
            memory_dim,
            activation="sigmoid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="erase",
        )
        self.add_dense = layers.Dense(
            memory_dim,
            activation="tanh",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="add",
        )

    def build(self, input_shape: tuple) -> None:
        """
        Build sub-layers explicitly.

        :param input_shape: Shape of controller output (batch, controller_dim).
        :type input_shape: tuple
        """
        self.key_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        self.gate_dense.build(input_shape)
        self.shift_dense.build(input_shape)
        self.gamma_dense.build(input_shape)
        self.erase_dense.build(input_shape)
        self.add_dense.build(input_shape)
        super().build(input_shape)

    def content_addressing(
        self,
        key: Any,
        beta: Any,
        memory: Any,
    ) -> Any:
        """
        Compute content-based attention weights.

        :param key: Key vector of shape (batch, 1, memory_dim).
        :type key: Any
        :param beta: Key strength of shape (batch, 1).
        :type beta: Any
        :param memory: Memory matrix of shape (batch, num_slots, memory_dim).
        :type memory: Any
        :return: Content weights of shape (batch, num_slots).
        :rtype: Any
        """
        similarity = cosine_similarity(key, memory, epsilon=self.epsilon)
        return ops.softmax(beta * similarity, axis=-1)

    def compute_addressing(
        self,
        controller_output: Any,
        memory_state: MemoryState,
        prev_weights: Any,
    ) -> tuple[Any, HeadState]:
        """
        Compute attention weights and write vectors.

        :param controller_output: Output from controller, shape (batch, controller_dim).
        :type controller_output: Any
        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param prev_weights: Previous attention weights, shape (batch, num_slots).
        :type prev_weights: Any
        :return: Tuple of (new_weights, head_state).
        :rtype: tuple[Any, HeadState]
        """
        # 1. Project parameters
        key = self.key_dense(controller_output)
        beta = self.beta_dense(controller_output)
        gate = self.gate_dense(controller_output)
        shift = self.shift_dense(controller_output)
        gamma = self.gamma_dense(controller_output) + 1.0
        erase = self.erase_dense(controller_output)
        add = self.add_dense(controller_output)

        # 2. Content Addressing
        key_expanded = ops.expand_dims(key, axis=1)
        content_weights = self.content_addressing(
            key_expanded, beta, memory_state.memory
        )

        # 3. Interpolation
        gated_weights = gate * content_weights + (1.0 - gate) * prev_weights

        # 4. Shift
        shifted_weights = circular_convolution(gated_weights, shift)

        # 5. Sharpen
        final_weights = sharpen_weights(shifted_weights, gamma, epsilon=self.epsilon)

        new_state = HeadState(
            weights=final_weights,
            key=key,
            beta=beta,
            gate=gate,
            shift=shift,
            gamma=gamma,
            erase_vector=erase,
            add_vector=add,
        )

        return final_weights, new_state

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Compute output shape.

        :param input_shape: Shape of controller output.
        :type input_shape: tuple
        :return: Shape of attention weights.
        :rtype: tuple
        """
        return (input_shape[0], self.memory_size)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTMController
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMController(BaseController):
    """
    Controller network for the Neural Turing Machine.

    Acts as the "CPU" of the NTM, processing external inputs and previous
    read vectors to generate control signals for memory operations.

    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param controller_type: Type of controller ('lstm', 'gru', 'feedforward').
    :type controller_type: str
    :param kernel_initializer: Initializer for dense layers.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        controller_dim: int,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            controller_dim=controller_dim,
            controller_type=controller_type,
            **kwargs,
        )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create core cell in __init__ (Golden Rule)
        if self.controller_type == "lstm":
            self.core = layers.LSTMCell(
                self.controller_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="controller_cell",
            )
        elif self.controller_type == "gru":
            self.core = layers.GRUCell(
                self.controller_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="controller_cell",
            )
        else:
            self.core = layers.Dense(
                self.controller_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="controller_dense",
            )

    def build(self, input_shape: tuple) -> None:
        """
        Build the core layer.

        :param input_shape: Shape of input tensor (batch, input_dim).
        :type input_shape: tuple
        """
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                input_shape = input_shape[0]

        if hasattr(self.core, "build"):
            self.core.build(input_shape)

        super().build(input_shape)

    def initialize_state(self, batch_size: int) -> list[keras.KerasTensor] | None:
        """
        Initialize controller state for a new sequence.

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: List of initial state tensors, or None for feedforward.
        :rtype: list[keras.KerasTensor] | None
        """
        if self.controller_type == "lstm":
            return [
                ops.zeros((batch_size, self.controller_dim)),
                ops.zeros((batch_size, self.controller_dim)),
            ]
        elif self.controller_type == "gru":
            return [ops.zeros((batch_size, self.controller_dim))]
        return None

    def call(
        self,
        inputs: Any,
        state: list[keras.KerasTensor] | None = None,
        training: bool | None = None,
    ) -> tuple[Any, list[keras.KerasTensor]]:
        """
        Process inputs through the controller.

        :param inputs: Input tensor of shape (batch, input_dim).
        :type inputs: Any
        :param state: Previous controller state.
        :type state: list[keras.KerasTensor] | None
        :param training: Whether in training mode.
        :type training: bool | None
        :return: Tuple of (controller_output, new_states).
        :rtype: tuple[Any, list[keras.KerasTensor]]
        """
        if self.controller_type in ["lstm", "gru"]:
            if state is None:
                batch_size = ops.shape(inputs)[0]
                state = self.initialize_state(batch_size)

            output, new_states = self.core(inputs, state, training=training)

            if not isinstance(new_states, list):
                new_states = (
                    list(new_states) if hasattr(new_states, "__iter__") else [new_states]
                )
            return output, new_states
        else:
            output = self.core(inputs, training=training)
            return output, []

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[tuple[int | None, ...], list[tuple[int | None, ...]]]:
        """
        Compute output shape.

        :param input_shape: Shape of input tensor.
        :type input_shape: tuple
        :return: Tuple of (output_shape, state_shapes).
        :rtype: tuple
        """
        batch_size = input_shape[0]
        output_shape = (batch_size, self.controller_dim)

        state_shape = (batch_size, self.controller_dim)
        if self.controller_type == "lstm":
            state_shapes = [state_shape, state_shape]
        elif self.controller_type == "gru":
            state_shapes = [state_shape]
        else:
            state_shapes = []

        return output_shape, state_shapes

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTMCell
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMCell(keras.layers.Layer):
    """
    Core NTM Cell for processing a single timestep.

    This layer implements the recurrent step logic required by keras.layers.RNN.
    It manages a complex state tuple containing controller state, memory matrix,
    read vectors, and attention weights.

    :param config: NTM configuration object or dictionary.
    :type config: NTMConfig | dict[str, Any]
    :param kernel_initializer: Initializer for dense layers.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        config: NTMConfig | dict[str, Any],
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Handle dict config from deserialization
        if isinstance(config, dict):
            self.config = NTMConfig.from_dict(config)
        else:
            self.config = config

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers
        self.memory = NTMMemory(
            self.config.memory_size,
            self.config.memory_dim,
            epsilon=self.config.epsilon,
            name="memory",
        )

        self.controller = NTMController(
            self.config.controller_dim,
            self.config.controller_type,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="controller",
        )

        self.read_heads = [
            NTMReadHead(
                self.config.memory_size,
                self.config.memory_dim,
                self.config.addressing_mode,
                self.config.shift_range,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                epsilon=self.config.epsilon,
                name=f"read_head_{i}",
            )
            for i in range(self.config.num_read_heads)
        ]

        self.write_heads = [
            NTMWriteHead(
                self.config.memory_size,
                self.config.memory_dim,
                self.config.addressing_mode,
                self.config.shift_range,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                epsilon=self.config.epsilon,
                name=f"write_head_{i}",
            )
            for i in range(self.config.num_write_heads)
        ]

        # Pre-calculate state sizes
        self._state_size = self._calculate_state_size()
        self._output_size = self.config.controller_dim + (
            self.config.num_read_heads * self.config.memory_dim
        )

    @property
    def state_size(self) -> list[Any]:
        """Return state sizes for RNN compatibility."""
        return self._state_size

    @property
    def output_size(self) -> int:
        """Return output size for RNN compatibility."""
        return self._output_size

    def _calculate_state_size(self) -> list[Any]:
        """Calculate state sizes for all state components."""
        sizes = []

        # Controller State
        if self.config.controller_type == "lstm":
            sizes.extend([self.config.controller_dim, self.config.controller_dim])
        elif self.config.controller_type == "gru":
            sizes.append(self.config.controller_dim)

        # Memory Matrix
        sizes.append((self.config.memory_size, self.config.memory_dim))

        # Read Vectors
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_dim)

        # Read Weights
        for _ in range(self.config.num_read_heads):
            sizes.append(self.config.memory_size)

        # Write Weights
        for _ in range(self.config.num_write_heads):
            sizes.append(self.config.memory_size)

        return sizes

    def build(self, input_shape: tuple) -> None:
        """
        Build controller and heads with correct shapes.

        :param input_shape: Shape of input tensor (batch, feature_dim).
        :type input_shape: tuple
        """
        feature_dim = input_shape[-1]
        total_read_dim = self.config.num_read_heads * self.config.memory_dim
        controller_input_shape = (None, feature_dim + total_read_dim)

        self.controller.build(controller_input_shape)

        controller_output_shape = (None, self.config.controller_dim)

        for head in self.read_heads:
            head.build(controller_output_shape)

        for head in self.write_heads:
            head.build(controller_output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        states: list[keras.KerasTensor],
        training: bool | None = None,
    ) -> tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        """
        Process one timestep of the NTM.

        :param inputs: Input tensor of shape (batch, input_dim).
        :type inputs: keras.KerasTensor
        :param states: List of state tensors.
        :type states: list[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: bool | None
        :return: Tuple of (output, new_states).
        :rtype: tuple[keras.KerasTensor, list[keras.KerasTensor]]
        """
        # Unpack State
        idx = 0

        if self.config.controller_type == "lstm":
            controller_state = [states[idx], states[idx + 1]]
            idx += 2
        elif self.config.controller_type == "gru":
            controller_state = [states[idx]]
            idx += 1
        else:
            controller_state = None

        memory_val = states[idx]
        idx += 1

        prev_read_vectors = []
        for _ in range(self.config.num_read_heads):
            prev_read_vectors.append(states[idx])
            idx += 1

        prev_read_weights = []
        for _ in range(self.config.num_read_heads):
            prev_read_weights.append(states[idx])
            idx += 1

        prev_write_weights = []
        for _ in range(self.config.num_write_heads):
            prev_write_weights.append(states[idx])
            idx += 1

        memory_state = MemoryState(memory=memory_val)

        # Controller Step
        flat_read_vectors = ops.concatenate(prev_read_vectors, axis=-1)
        controller_input = ops.concatenate([inputs, flat_read_vectors], axis=-1)

        controller_output, new_controller_state = self.controller(
            controller_input,
            state=controller_state,
            training=training,
        )

        # Write Heads
        current_memory_state = memory_state
        new_write_weights = []

        for i, head in enumerate(self.write_heads):
            weights, head_state = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_write_weights[i],
            )
            new_write_weights.append(weights)

            current_memory_state = self.memory.write(
                current_memory_state,
                weights,
                head_state.erase_vector,
                head_state.add_vector,
            )

        # Read Heads
        new_read_weights = []
        new_read_vectors = []

        for i, head in enumerate(self.read_heads):
            weights, _ = head.compute_addressing(
                controller_output,
                current_memory_state,
                prev_read_weights[i],
            )
            new_read_weights.append(weights)

            read_vec = self.memory.read(current_memory_state, weights)
            new_read_vectors.append(read_vec)

        # Pack Output State
        new_states = []

        if self.config.controller_type == "lstm":
            new_states.extend(new_controller_state)
        elif self.config.controller_type == "gru":
            new_states.extend(new_controller_state)

        new_states.append(current_memory_state.memory)
        new_states.extend(new_read_vectors)
        new_states.extend(new_read_weights)
        new_states.extend(new_write_weights)

        # Output
        flat_new_read_vectors = ops.concatenate(new_read_vectors, axis=-1)
        cell_output = ops.concatenate([controller_output, flat_new_read_vectors], axis=-1)

        return cell_output, new_states

    def get_initial_state(
        self,
        inputs: keras.KerasTensor | None = None,
        batch_size: int | None = None,
        dtype: str | None = None,
    ) -> list[keras.KerasTensor]:
        """
        Initialize all states to zero/initial values.

        :param inputs: Optional input tensor to infer batch size.
        :type inputs: keras.KerasTensor | None
        :param batch_size: Batch size for state initialization.
        :type batch_size: int | None
        :param dtype: Data type for states.
        :type dtype: str | None
        :return: List of initial state tensors.
        :rtype: list[keras.KerasTensor]
        """
        if batch_size is None and inputs is not None:
            batch_size = ops.shape(inputs)[0]

        states = []

        # Controller states
        if self.config.controller_type == "lstm":
            states.extend([
                ops.zeros((batch_size, self.config.controller_dim)),
                ops.zeros((batch_size, self.config.controller_dim)),
            ])
        elif self.config.controller_type == "gru":
            states.append(ops.zeros((batch_size, self.config.controller_dim)))

        # Memory
        states.append(
            ops.ones((batch_size, self.config.memory_size, self.config.memory_dim))
            * self.config.epsilon
        )

        # Read Vectors
        for _ in range(self.config.num_read_heads):
            states.append(ops.zeros((batch_size, self.config.memory_dim)))

        # Read Weights (uniform)
        uniform_weight = ops.ones((1, self.config.memory_size)) / self.config.memory_size
        for _ in range(self.config.num_read_heads):
            states.append(ops.broadcast_to(uniform_weight, (batch_size, self.config.memory_size)))

        # Write Weights (uniform)
        for _ in range(self.config.num_write_heads):
            states.append(ops.broadcast_to(uniform_weight, (batch_size, self.config.memory_size)))

        return states

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Compute output shape.

        :param input_shape: Shape of input tensor.
        :type input_shape: tuple
        :return: Output shape.
        :rtype: tuple
        """
        return (input_shape[0], self._output_size)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "config": self.config.to_dict(),
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NTMCell":
        """
        Create layer from configuration.

        :param config: Configuration dictionary.
        :type config: dict[str, Any]
        :return: NTMCell instance.
        :rtype: NTMCell
        """
        return cls(**config)


# ---------------------------------------------------------------------
# NeuralTuringMachine
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="NTM")
class NeuralTuringMachine(BaseNTM):
    """
    Complete Neural Turing Machine Layer.

    Wraps NTMCell in an RNN layer for sequence processing.

    :param config: NTM configuration object or dictionary.
    :type config: NTMConfig | dict[str, Any]
    :param output_dim: Dimension of the output projection.
    :type output_dim: int
    :param return_sequences: Whether to return outputs at all timesteps.
    :type return_sequences: bool
    :param return_state: Whether to return final states.
    :type return_state: bool
    :param kernel_initializer: Initializer for dense layers.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        config: NTMConfig | dict[str, Any],
        output_dim: int,
        return_sequences: bool = True,
        return_state: bool = False,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        # Handle dict config
        if isinstance(config, dict):
            config = NTMConfig.from_dict(config)

        super().__init__(config=config, output_dim=output_dim, **kwargs)

        self.return_sequences = return_sequences
        self.return_state = return_state
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.ntm_cell = NTMCell(
            self.config,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="ntm_cell",
        )

        self.rnn = layers.RNN(
            self.ntm_cell,
            return_sequences=return_sequences,
            return_state=return_state,
            name="ntm_rnn",
        )

        self.output_projection = layers.Dense(
            output_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="output_projection",
        )

    def build(self, input_shape: tuple) -> None:
        """
        Build sub-layers.

        :param input_shape: Shape of input (batch, seq_len, input_dim).
        :type input_shape: tuple
        """
        self.rnn.build(input_shape)

        out_dim = self.ntm_cell.output_size
        if self.return_sequences:
            proj_input_shape = (input_shape[0], input_shape[1], out_dim)
        else:
            proj_input_shape = (input_shape[0], out_dim)
        self.output_projection.build(proj_input_shape)

        super().build(input_shape)

    def initialize_state(
        self,
        batch_size: int,
    ) -> tuple[MemoryState, list[HeadState], Any | None]:
        """
        Initialize all states (placeholder implementation).

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        :return: Empty placeholder states.
        :rtype: tuple
        """
        return MemoryState(memory=None), [], None

    def step(
        self,
        inputs: Any,
        memory_state: MemoryState,
        head_states: list[HeadState],
        controller_state: Any | None,
        training: bool | None = None,
    ) -> NTMOutput:
        """
        Single step (not used - RNN handles internally).

        :param inputs: Input tensor.
        :type inputs: Any
        :param memory_state: Current memory state.
        :type memory_state: MemoryState
        :param head_states: Current head states.
        :type head_states: list[HeadState]
        :param controller_state: Current controller state.
        :type controller_state: Any | None
        :param training: Training mode flag.
        :type training: bool | None
        :return: Empty NTMOutput (not used).
        :rtype: NTMOutput
        """
        return NTMOutput(
            output=inputs,
            memory_state=memory_state,
            head_states=head_states,
            read_vectors=None,
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        initial_state: list[keras.KerasTensor] | None = None,
        training: bool | None = None,
    ) -> keras.KerasTensor | tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        """
        Process input sequence through NTM.

        :param inputs: Input tensor of shape (batch, seq_len, input_dim).
        :type inputs: keras.KerasTensor
        :param initial_state: Optional initial states.
        :type initial_state: list[keras.KerasTensor] | None
        :param training: Training mode flag.
        :type training: bool | None
        :return: Output tensor(s) and optionally final states.
        :rtype: keras.KerasTensor | tuple
        """
        rnn_result = self.rnn(inputs, initial_state=initial_state, training=training)

        if self.return_state:
            rnn_output = rnn_result[0]
            final_states = list(rnn_result[1:])
        else:
            rnn_output = rnn_result
            final_states = None

        output = self.output_projection(rnn_output, training=training)

        if self.return_state:
            return output, final_states
        return output

    def get_memory_state(self) -> MemoryState | None:
        """
        Get the current memory state (not available in wrapped mode).

        :return: None
        :rtype: MemoryState | None
        """
        return None

    def reset_memory(self, batch_size: int) -> None:
        """
        Reset memory (no-op in wrapped mode).

        :param batch_size: Number of sequences in the batch.
        :type batch_size: int
        """
        pass

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Compute output shape.

        :param input_shape: Shape of input (batch, seq_len, input_dim).
        :type input_shape: tuple
        :return: Output shape.
        :rtype: tuple
        """
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if self.return_sequences:
            return (batch_size, seq_len, self.output_dim)
        return (batch_size, self.output_dim)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NeuralTuringMachine":
        """
        Create layer from configuration.

        :param config: Configuration dictionary.
        :type config: dict[str, Any]
        :return: NeuralTuringMachine instance.
        :rtype: NeuralTuringMachine
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------


def create_ntm(
    memory_size: int = 128,
    memory_dim: int = 64,
    output_dim: int = 10,
    controller_dim: int = 256,
    controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
    num_read_heads: int = 1,
    num_write_heads: int = 1,
    shift_range: int = 3,
    return_sequences: bool = True,
    return_state: bool = False,
) -> NeuralTuringMachine:
    """
    Factory function to create a Neural Turing Machine layer.

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param output_dim: Dimension of output.
    :type output_dim: int
    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param controller_type: Type of controller ('lstm', 'gru', 'feedforward').
    :type controller_type: str
    :param num_read_heads: Number of read heads.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads.
    :type num_write_heads: int
    :param shift_range: Range of allowed shifts.
    :type shift_range: int
    :param return_sequences: Whether to return full sequence.
    :type return_sequences: bool
    :param return_state: Whether to return final state.
    :type return_state: bool
    :return: Configured NeuralTuringMachine layer.
    :rtype: NeuralTuringMachine
    """
    config = NTMConfig(
        memory_size=memory_size,
        memory_dim=memory_dim,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
        controller_dim=controller_dim,
        controller_type=controller_type,
        shift_range=shift_range,
    )

    return NeuralTuringMachine(
        config,
        output_dim=output_dim,
        return_sequences=return_sequences,
        return_state=return_state,
    )