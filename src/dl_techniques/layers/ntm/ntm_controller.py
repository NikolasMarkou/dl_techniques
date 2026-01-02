"""
Neural Turing Machine (NTM) Model.

A robust, production-ready implementation of the Neural Turing Machine architecture
adhering to Keras 3 strict serialization and explicit build patterns.

This module integrates the differentiable addressing mechanisms from `base_layers.py`
into a complete, trainable model structure. It features:

1.  **Modular Controller**: configurable LSTM, GRU, or FeedForward controllers.
2.  **Explicit State Management**: Handles the complex state tuple (Controller, Memory,
    Weights, Read Vectors) required for stable RNN execution.
3.  **Variant Configuration**: Pre-defined configurations (tiny, small, base, etc.)
    similar to modern Transformer/ConvNet implementations.
4.  **Weight Compatibility**: Explicit layer creation ensures weights can be loaded
    even if auxiliary features (like return_sequences) change.

Based on: "Neural Turing Machines" (Graves et al., 2014)
"""

import keras
from typing import Any, Literal
from keras import initializers, layers, ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base_layers import DifferentiableSelectCopy

# ---------------------------------------------------------------------
# NTM Configuration
# ---------------------------------------------------------------------


class NTMConfig:
    """
    Configuration for NTM architectures.

    :param memory_size: Number of memory slots (N).
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot (M).
    :type memory_dim: int
    :param num_read_heads: Number of read heads.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads.
    :type num_write_heads: int
    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param controller_type: Type of controller ('lstm', 'gru', 'feedforward').
    :type controller_type: str
    :param shift_range: Range of allowed shifts for location addressing (must be odd).
    :type shift_range: int
    :param use_memory_init: Whether to learn initial memory state.
    :type use_memory_init: bool
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    """

    def __init__(
        self,
        memory_size: int = 128,
        memory_dim: int = 64,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
        controller_dim: int = 256,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        shift_range: int = 3,
        use_memory_init: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if num_read_heads <= 0:
            raise ValueError(f"num_read_heads must be positive, got {num_read_heads}")
        if num_write_heads <= 0:
            raise ValueError(
                f"num_write_heads must be positive, got {num_write_heads}"
            )
        if controller_dim <= 0:
            raise ValueError(f"controller_dim must be positive, got {controller_dim}")
        if controller_type not in ["lstm", "gru", "feedforward"]:
            raise ValueError(
                f"controller_type must be 'lstm', 'gru', or 'feedforward', "
                f"got {controller_type}"
            )
        if shift_range <= 0 or shift_range % 2 == 0:
            raise ValueError(
                f"shift_range must be a positive odd integer, got {shift_range}"
            )

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_dim = controller_dim
        self.controller_type = controller_type
        self.shift_range = shift_range
        self.use_memory_init = use_memory_init
        self.epsilon = epsilon

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "memory_size": self.memory_size,
            "memory_dim": self.memory_dim,
            "num_read_heads": self.num_read_heads,
            "num_write_heads": self.num_write_heads,
            "controller_dim": self.controller_dim,
            "controller_type": self.controller_type,
            "shift_range": self.shift_range,
            "use_memory_init": self.use_memory_init,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "NTMConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# ---------------------------------------------------------------------
# NTM Controller
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMController(keras.layers.Layer):
    """
    Controller network for the Neural Turing Machine.

    Acts as the "CPU" of the NTM, processing external inputs and previous read
    vectors to generate control signals for the memory head and final outputs.

    **Architecture**::

        Input [External Input, Previous Read Vectors]
               ↓
        RNN/Dense Layer (LSTM/GRU/Dense)
               ↓
        Output (to Memory Head and Output Projection)

    :param units: Dimensionality of the controller's hidden state.
    :type units: int
    :param controller_type: One of 'lstm', 'gru', 'feedforward'. Defaults to 'lstm'.
    :type controller_type: str
    :param kernel_initializer: Initializer for weights. Defaults to 'glorot_uniform'.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for biases. Defaults to 'zeros'.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for weights. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        units: int,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if controller_type not in ["lstm", "gru", "feedforward"]:
            raise ValueError(
                f"Invalid controller_type: {controller_type}. "
                f"Must be 'lstm', 'gru', or 'feedforward'."
            )

        self.units = units
        self.controller_type = controller_type
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers in __init__ (Golden Rule)
        if self.controller_type == "lstm":
            self.core = layers.LSTMCell(
                self.units,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="controller_cell",
            )
        elif self.controller_type == "gru":
            self.core = layers.GRUCell(
                self.units,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="controller_cell",
            )
        else:
            self.core = layers.Dense(
                self.units,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="controller_dense",
            )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """
        Explicitly build the core layer.

        :param input_shape: Shape of input tensor (batch, input_dim).
        :type input_shape: tuple
        """
        self.core.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        states: list[keras.KerasTensor] | keras.KerasTensor | None = None,
        training: bool | None = None,
    ) -> tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        """
        Process inputs and update controller state.

        :param inputs: Input tensor of shape (batch, input_dim).
        :type inputs: keras.KerasTensor
        :param states: Previous states for RNN controllers.
        :type states: list[keras.KerasTensor] or None
        :param training: Training mode flag.
        :type training: bool or None
        :return: Tuple of (output, new_states).
        :rtype: tuple[keras.KerasTensor, list[keras.KerasTensor]]
        """
        if self.controller_type in ["lstm", "gru"]:
            # Handle implicit state initialization if None
            if states is None:
                batch_size = ops.shape(inputs)[0]
                states = self.get_initial_state(batch_size)

            output, new_states = self.core(inputs, states, training=training)

            # Ensure new_states is always a list for consistency
            if not isinstance(new_states, list):
                new_states = (
                    list(new_states)
                    if hasattr(new_states, "__iter__")
                    else [new_states]
                )
            return output, new_states
        else:
            output = self.core(inputs, training=training)
            return output, []

    def get_initial_state(
        self,
        batch_size: int | None = None,
    ) -> list[keras.KerasTensor]:
        """
        Get initial state for RNN controllers.

        :param batch_size: Batch size for state initialization.
        :type batch_size: int or None
        :return: List of initial state tensors.
        :rtype: list[keras.KerasTensor]
        """
        if self.controller_type == "lstm":
            return [
                ops.zeros((batch_size, self.units)),
                ops.zeros((batch_size, self.units)),
            ]
        elif self.controller_type == "gru":
            return [ops.zeros((batch_size, self.units))]
        return []

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[tuple[int | None, ...], list[tuple[int | None, ...]]]:
        """
        Compute output shape.

        :param input_shape: Shape of input tensor (batch, input_dim).
        :type input_shape: tuple
        :return: Tuple of (output_shape, state_shapes).
        :rtype: tuple
        """
        batch_size = input_shape[0]
        output_shape = (batch_size, self.units)

        state_shape = (batch_size, self.units)
        state_shapes = []
        if self.controller_type == "lstm":
            state_shapes = [state_shape, state_shape]
        elif self.controller_type == "gru":
            state_shapes = [state_shape]

        return output_shape, state_shapes

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "controller_type": self.controller_type,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# NTM Cell
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="NTM")
class NTMCell(keras.layers.Layer):
    """
    Core NTM Cell orchestrating Controller and Memory interactions per timestep.

    This layer implements the recurrent step logic required by `keras.layers.RNN`.
    It manages a complex state tuple containing:

    1. Controller State (LSTM/GRU hidden states)
    2. Memory Matrix
    3. Previous Read Vectors
    4. Previous Read/Write Weights (for location-based addressing)

    **Logic Flow**::

        1. Input: [External Input, Previous Read Vectors]
        2. Controller processes Input -> Controller Output
        3. Memory Head uses Controller Output + Prev Weights -> New Memory, Read Output, New Weights
        4. Cell Output: Concatenate [Controller Output, Read Output]

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param num_read_heads: Number of read heads. Defaults to 1.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads. Defaults to 1.
    :type num_write_heads: int
    :param controller_type: Type of controller ('lstm', 'gru', 'feedforward').
        Defaults to 'lstm'.
    :type controller_type: str
    :param shift_range: Range of allowed shifts (must be odd). Defaults to 3.
    :type shift_range: int
    :param kernel_initializer: Initializer for weights. Defaults to 'glorot_uniform'.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for biases. Defaults to 'zeros'.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for weights. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param epsilon: Small constant for numerical stability. Defaults to 1e-6.
    :type epsilon: float
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        controller_dim: int,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        shift_range: int = 3,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if controller_dim <= 0:
            raise ValueError(f"controller_dim must be positive, got {controller_dim}")
        if num_read_heads <= 0:
            raise ValueError(f"num_read_heads must be positive, got {num_read_heads}")
        if num_write_heads <= 0:
            raise ValueError(
                f"num_write_heads must be positive, got {num_write_heads}"
            )
        if controller_type not in ["lstm", "gru", "feedforward"]:
            raise ValueError(
                f"controller_type must be 'lstm', 'gru', or 'feedforward', "
                f"got {controller_type}"
            )
        if shift_range <= 0 or shift_range % 2 == 0:
            raise ValueError(
                f"shift_range must be a positive odd integer, got {shift_range}"
            )

        # Store configuration
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_type = controller_type
        self.shift_range = shift_range
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.epsilon = epsilon

        # Create sub-layers in __init__
        self.controller = NTMController(
            units=self.controller_dim,
            controller_type=self.controller_type,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="controller",
        )

        self.memory_interface = DifferentiableSelectCopy(
            memory_size=self.memory_size,
            content_dim=self.memory_dim,
            controller_dim=self.controller_dim,
            num_read_heads=self.num_read_heads,
            num_write_heads=self.num_write_heads,
            num_shifts=self.shift_range,
            use_content_addressing=True,
            use_location_addressing=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="memory_interface",
        )

        # Pre-calculate state sizes for RNN contract
        self._state_size = self._calculate_state_size()
        self._output_size = self.controller_dim + (
            self.num_read_heads * self.memory_dim
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
        """
        Calculate state sizes for all state components.

        State structure:
        1. Controller states (h, c for LSTM; h for GRU; empty for feedforward)
        2. Memory matrix (memory_size, memory_dim)
        3. Read vectors (num_read_heads * memory_dim)
        4. Read weights (num_read_heads * memory_size)
        5. Write weights (num_write_heads * memory_size)
        """
        sizes = []

        # Controller State
        if self.controller_type == "lstm":
            sizes.extend([self.controller_dim, self.controller_dim])
        elif self.controller_type == "gru":
            sizes.append(self.controller_dim)

        # Memory Matrix
        sizes.append((self.memory_size, self.memory_dim))

        # Read Vectors
        for _ in range(self.num_read_heads):
            sizes.append(self.memory_dim)

        # Read Weights
        for _ in range(self.num_read_heads):
            sizes.append(self.memory_size)

        # Write Weights
        for _ in range(self.num_write_heads):
            sizes.append(self.memory_size)

        return sizes

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """
        Build sub-layers.

        :param input_shape: Shape of input tensor (batch, input_dim).
        :type input_shape: tuple
        """
        input_dim = input_shape[-1]
        total_read_dim = self.num_read_heads * self.memory_dim
        controller_input_dim = input_dim + total_read_dim

        self.controller.build((None, controller_input_dim))
        self.memory_interface.build(
            (None, self.memory_size, self.memory_dim)
        )

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
        :type training: bool or None
        :return: Tuple of (output, new_states).
        :rtype: tuple[keras.KerasTensor, list[keras.KerasTensor]]
        """
        # --- 1. Unpack State Tuple ---
        idx = 0

        # Controller States
        if self.controller_type == "lstm":
            ctrl_states = [states[idx], states[idx + 1]]
            idx += 2
        elif self.controller_type == "gru":
            ctrl_states = [states[idx]]
            idx += 1
        else:
            ctrl_states = []

        # Memory Matrix
        memory_matrix = states[idx]
        idx += 1

        # Previous Read Vectors
        prev_read_vectors = []
        for _ in range(self.num_read_heads):
            prev_read_vectors.append(states[idx])
            idx += 1

        # Previous Read Weights
        prev_read_weights = []
        for _ in range(self.num_read_heads):
            prev_read_weights.append(states[idx])
            idx += 1

        # Previous Write Weights
        prev_write_weights = []
        for _ in range(self.num_write_heads):
            prev_write_weights.append(states[idx])
            idx += 1

        # --- 2. Controller Step ---
        flat_read_vectors = ops.concatenate(prev_read_vectors, axis=-1)
        controller_input = ops.concatenate([inputs, flat_read_vectors], axis=-1)

        controller_output, new_ctrl_states = self.controller(
            controller_input,
            states=ctrl_states,
            training=training,
        )

        # --- 3. Memory Step ---
        new_memory, read_output_flat, address_info = self.memory_interface(
            memory=memory_matrix,
            controller_state=controller_output,
            previous_read_weights=prev_read_weights,
            previous_write_weights=prev_write_weights,
            training=training,
        )

        # --- 4. Pack New State ---
        new_states = []

        # Controller states
        if self.controller_type in ["lstm", "gru"]:
            new_states.extend(new_ctrl_states)

        # Memory
        new_states.append(new_memory)

        # Read Vectors (split flat output back into list)
        new_read_vectors = ops.split(
            read_output_flat, self.num_read_heads, axis=-1
        )
        new_states.extend(new_read_vectors)

        # Read/Write Weights
        new_states.extend(address_info["read_weights"])
        new_states.extend(address_info["write_weights"])

        # --- 5. Output ---
        final_output = ops.concatenate([controller_output, read_output_flat], axis=-1)

        return final_output, new_states

    def get_initial_state(
        self,
        inputs: keras.KerasTensor | None = None,
        batch_size: int | None = None,
        dtype: str | None = None,
    ) -> list[keras.KerasTensor]:
        """
        Initialize all states to zero/initial values.

        :param inputs: Optional input tensor to infer batch size.
        :type inputs: keras.KerasTensor or None
        :param batch_size: Batch size for state initialization.
        :type batch_size: int or None
        :param dtype: Data type for states.
        :type dtype: str or None
        :return: List of initial state tensors.
        :rtype: list[keras.KerasTensor]
        """
        if batch_size is None and inputs is not None:
            batch_size = ops.shape(inputs)[0]

        states = []

        # Controller states
        states.extend(self.controller.get_initial_state(batch_size))

        # Memory (batch, size, dim) - small constant initialization
        states.append(
            ops.ones((batch_size, self.memory_size, self.memory_dim)) * self.epsilon
        )

        # Read Vectors - zeros
        for _ in range(self.num_read_heads):
            states.append(ops.zeros((batch_size, self.memory_dim)))

        # Read Weights
        # Use trainable initial weights if available (layer built), else constant fallback
        for head in self.memory_interface.read_heads:
            if hasattr(head, "initial_weights") and head.initial_weights is not None:
                init_w = head.initial_weights
            else:
                init_w = ops.ones((1, self.memory_size)) / self.memory_size

            if dtype is not None:
                init_w = ops.cast(init_w, dtype)
            states.append(ops.broadcast_to(init_w, (batch_size, self.memory_size)))

        # Write Weights
        for head in self.memory_interface.write_heads:
            if hasattr(head, "initial_weights") and head.initial_weights is not None:
                init_w = head.initial_weights
            else:
                init_w = ops.ones((1, self.memory_size)) / self.memory_size

            if dtype is not None:
                init_w = ops.cast(init_w, dtype)
            states.append(ops.broadcast_to(init_w, (batch_size, self.memory_size)))

        return states

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Compute output shape.

        :param input_shape: Shape of input tensor (batch, input_dim).
        :type input_shape: tuple
        :return: Output shape (batch, output_size).
        :rtype: tuple
        """
        return (input_shape[0], self._output_size)

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "memory_dim": self.memory_dim,
                "controller_dim": self.controller_dim,
                "num_read_heads": self.num_read_heads,
                "num_write_heads": self.num_write_heads,
                "controller_type": self.controller_type,
                "shift_range": self.shift_range,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "epsilon": self.epsilon,
            }
        )
        return config


# ---------------------------------------------------------------------
# NTM Layer (RNN Wrapper)
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="NTM")
class NTMLayer(keras.layers.Layer):
    """
    Neural Turing Machine layer wrapping NTMCell with RNN.

    This layer provides a complete NTM that processes sequences and manages
    state automatically.

    **Architecture**::

        Input Sequence (batch, seq_len, input_dim)
               ↓
        RNN(NTMCell)
               ↓
        Output Projection (if output_dim specified)
               ↓
        Output (batch, seq_len, output_dim) or (batch, output_dim)

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param memory_dim: Dimension of each memory slot.
    :type memory_dim: int
    :param controller_dim: Dimension of controller hidden state.
    :type controller_dim: int
    :param output_dim: Output dimension. If None, outputs cell output directly.
    :type output_dim: int or None
    :param num_read_heads: Number of read heads. Defaults to 1.
    :type num_read_heads: int
    :param num_write_heads: Number of write heads. Defaults to 1.
    :type num_write_heads: int
    :param controller_type: Type of controller. Defaults to 'lstm'.
    :type controller_type: str
    :param shift_range: Range of allowed shifts. Defaults to 3.
    :type shift_range: int
    :param return_sequences: Whether to return full sequence. Defaults to True.
    :type return_sequences: bool
    :param return_state: Whether to return final state. Defaults to False.
    :type return_state: bool
    :param kernel_initializer: Initializer for weights. Defaults to 'glorot_uniform'.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for biases. Defaults to 'zeros'.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for weights. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        controller_dim: int,
        output_dim: int | None = None,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
        controller_type: Literal["lstm", "gru", "feedforward"] = "lstm",
        shift_range: int = 3,
        return_sequences: bool = True,
        return_state: bool = False,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim
        self.output_dim = output_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_type = controller_type
        self.shift_range = shift_range
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create NTM Cell
        self.cell = NTMCell(
            memory_size=memory_size,
            memory_dim=memory_dim,
            controller_dim=controller_dim,
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            controller_type=controller_type,
            shift_range=shift_range,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="ntm_cell",
        )

        # RNN wrapper
        self.rnn = layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
            name="ntm_rnn",
        )

        # Output projection (optional)
        self.output_projection: keras.layers.Dense | None = None
        if output_dim is not None:
            self.output_projection = layers.Dense(
                output_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="output_projection",
            )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """
        Build sub-layers.

        :param input_shape: Shape of input (batch, seq_len, input_dim).
        :type input_shape: tuple
        """
        self.rnn.build(input_shape)

        if self.output_projection is not None:
            if self.return_sequences:
                proj_input_shape = (
                    input_shape[0],
                    input_shape[1],
                    self.cell.output_size,
                )
            else:
                proj_input_shape = (input_shape[0], self.cell.output_size)
            self.output_projection.build(proj_input_shape)

        super().build(input_shape)

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
        :type initial_state: list[keras.KerasTensor] or None
        :param training: Training mode flag.
        :type training: bool or None
        :return: Output tensor(s) and optionally final states.
        :rtype: keras.KerasTensor or tuple
        """
        if self.return_state:
            output, *states = self.rnn(
                inputs,
                initial_state=initial_state,
                training=training,
            )
        else:
            output = self.rnn(
                inputs,
                initial_state=initial_state,
                training=training,
            )
            states = None

        if self.output_projection is not None:
            output = self.output_projection(output, training=training)

        if self.return_state:
            return output, states
        return output

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
        batch = input_shape[0]
        seq_len = input_shape[1]

        if self.output_dim is not None:
            out_dim = self.output_dim
        else:
            out_dim = self.cell.output_size

        if self.return_sequences:
            return (batch, seq_len, out_dim)
        return (batch, out_dim)

    def get_config(self) -> dict[str, Any]:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "memory_dim": self.memory_dim,
                "controller_dim": self.controller_dim,
                "output_dim": self.output_dim,
                "num_read_heads": self.num_read_heads,
                "num_write_heads": self.num_write_heads,
                "controller_type": self.controller_type,
                "shift_range": self.shift_range,
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