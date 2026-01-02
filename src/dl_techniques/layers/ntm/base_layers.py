"""
Differentiable Addressing and Memory Layers for Neural Turing Machines.

This module implements the core differentiable addressing mechanisms that form
the foundation for memory-augmented neural networks.

The addressing mechanism combines:
    - Content-based addressing via cosine similarity
    - Location-based addressing via shift convolution
    - Interpolation gating between content and location modes
    - Sharpening to focus attention distributions

Based on: "Neural Turing Machines" (Graves et al., 2014)

Classes:
    DifferentiableAddressingHead: NTM-style addressing head.
    DifferentiableSelectCopy: Read/write operations with multiple heads.
    SimpleSelectCopy: Simplified select-copy mechanism.
"""

import keras
from keras import ops
from typing import Any

from .ntm_interface import (
    cosine_similarity,
    circular_convolution,
)


# ---------------------------------------------------------------------
# DifferentiableAddressingHead
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DifferentiableAddressingHead(keras.layers.Layer):
    """
    Differentiable addressing head implementing NTM-style memory addressing.

    This head computes attention weights over memory locations using a
    combination of content-based and location-based addressing mechanisms.

    **Architecture**::

        controller_state (batch, controller_dim)
               |
               +-> key_proj -> key (batch, content_dim)
               |                    |
               |              cosine_similarity(key, memory)
               |                    |
               +-> beta_proj -> beta * similarity -> softmax -> content_weights
               |
               +-> gate_proj -> gate
               |                 |
               |         gate * content_weights + (1-gate) * prev_weights
               |                 |
               +-> shift_proj -> shift_kernel
               |                    |
               |            circular_convolve(gated_weights, shift_kernel)
               |                    |
               +-> gamma_proj -> gamma
                                  |
                           weights^gamma / sum(weights^gamma)
                                  |
                           output_weights (batch, memory_size)

    :param memory_size: Number of memory slots to address.
    :type memory_size: int
    :param content_dim: Dimensionality of memory content vectors.
    :type content_dim: int
    :param controller_dim: Dimensionality of the controller state.
        If None, inferred from input shape in build().
    :type controller_dim: int | None
    :param num_shifts: Number of shift positions for location-based addressing.
        Must be odd (e.g., 3 for shifts of -1, 0, +1). Defaults to 3.
    :type num_shifts: int
    :param use_content_addressing: Whether to use content-based addressing.
        Defaults to True.
    :type use_content_addressing: bool
    :param use_location_addressing: Whether to use location-based addressing.
        Defaults to True.
    :type use_location_addressing: bool
    :param sharpening_bias: Initial bias for sharpening parameter gamma.
        Defaults to 1.0.
    :type sharpening_bias: float
    :param kernel_initializer: Initializer for dense layer kernels.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layer kernels.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        memory_size: int,
        content_dim: int,
        controller_dim: int | None = None,
        num_shifts: int = 3,
        use_content_addressing: bool = True,
        use_location_addressing: bool = True,
        sharpening_bias: float = 1.0,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_shifts % 2 == 0:
            raise ValueError(
                f"num_shifts must be odd for symmetric shifts, got {num_shifts}"
            )
        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if content_dim <= 0:
            raise ValueError(f"content_dim must be positive, got {content_dim}")

        self.memory_size = memory_size
        self.content_dim = content_dim
        self.controller_dim = controller_dim
        self.num_shifts = num_shifts
        self.use_content_addressing = use_content_addressing
        self.use_location_addressing = use_location_addressing
        self.sharpening_bias = sharpening_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers in __init__ (Golden Rule)
        self.key_proj: keras.layers.Dense | None = None
        self.beta_proj: keras.layers.Dense | None = None
        self.gate_proj: keras.layers.Dense | None = None
        self.shift_proj: keras.layers.Dense | None = None

        if self.use_content_addressing:
            self.key_proj = keras.layers.Dense(
                self.content_dim,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="key_projection",
            )
            self.beta_proj = keras.layers.Dense(
                1,
                activation="softplus",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="beta_projection",
            )

        if self.use_content_addressing and self.use_location_addressing:
            self.gate_proj = keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="interpolation_gate",
            )

        if self.use_location_addressing:
            self.shift_proj = keras.layers.Dense(
                self.num_shifts,
                activation="softmax",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="shift_distribution",
            )

        self.gamma_proj = keras.layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=keras.initializers.Constant(self.sharpening_bias),
            kernel_regularizer=self.kernel_regularizer,
            name="sharpening_gamma",
        )

        # Learnable initial weights (created in build)
        self.initial_weights: keras.Variable | None = None

    def build(self, input_shape: tuple) -> None:
        """
        Build the addressing head components.

        :param input_shape: Shape of input tensor.
        :type input_shape: tuple
        """
        # Determine controller dim
        if self.controller_dim is not None:
            controller_dim = self.controller_dim
        else:
            controller_dim = (
                input_shape[-1] if len(input_shape) == 2 else self.content_dim
            )

        if self.key_proj is not None:
            self.key_proj.build((None, controller_dim))
        if self.beta_proj is not None:
            self.beta_proj.build((None, controller_dim))
        if self.gate_proj is not None:
            self.gate_proj.build((None, controller_dim))
        if self.shift_proj is not None:
            self.shift_proj.build((None, controller_dim))
        self.gamma_proj.build((None, controller_dim))

        self.initial_weights = self.add_weight(
            name="initial_location_weights",
            shape=(1, self.memory_size),
            initializer=keras.initializers.Constant(1.0 / self.memory_size),
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        memory: keras.KerasTensor,
        controller_state: keras.KerasTensor,
        previous_weights: keras.KerasTensor | None = None,
        training: bool | None = None,
    ) -> keras.KerasTensor:
        """
        Compute addressing weights over memory.

        :param memory: Memory tensor of shape (batch, memory_size, content_dim).
        :type memory: keras.KerasTensor
        :param controller_state: Controller output of shape (batch, controller_dim).
        :type controller_state: keras.KerasTensor
        :param previous_weights: Previous attention weights for location addressing.
            Shape (batch, memory_size). Defaults to learned initial weights.
        :type previous_weights: keras.KerasTensor | None
        :param training: Training mode flag.
        :type training: bool | None
        :return: Attention weights of shape (batch, memory_size).
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(memory)[0]

        if previous_weights is None:
            previous_weights = ops.broadcast_to(
                self.initial_weights, (batch_size, self.memory_size)
            )

        # Step 1: Content-based addressing
        if self.use_content_addressing:
            key = self.key_proj(controller_state)
            beta = self.beta_proj(controller_state) + 1.0

            # Use shared utility instead of internal helper
            similarity = cosine_similarity(key, memory)
            content_weights = ops.softmax(beta * similarity, axis=-1)
        else:
            content_weights = previous_weights

        # Step 2: Interpolation gate
        if self.use_content_addressing and self.use_location_addressing:
            gate = self.gate_proj(controller_state)
            gated_weights = gate * content_weights + (1.0 - gate) * previous_weights
        elif self.use_content_addressing:
            gated_weights = content_weights
        else:
            gated_weights = previous_weights

        # Step 3: Location-based shift convolution
        if self.use_location_addressing:
            shift_kernel = self.shift_proj(controller_state)
            # Use shared optimized utility instead of internal helper
            shifted_weights = circular_convolution(gated_weights, shift_kernel)
        else:
            shifted_weights = gated_weights

        # Step 4: Sharpening
        gamma = ops.softplus(self.gamma_proj(controller_state)) + 1.0
        sharpened = ops.power(shifted_weights + 1e-8, gamma)
        weights = sharpened / (ops.sum(sharpened, axis=-1, keepdims=True) + 1e-8)

        return weights

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        """
        Compute output shape.

        :param input_shape: Shape of memory input (batch, memory_size, content_dim).
        :type input_shape: tuple
        :return: Output shape (batch, memory_size).
        :rtype: tuple
        """
        if len(input_shape) == 3:
            return (input_shape[0], self.memory_size)
        return (input_shape[0], self.memory_size)

    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "content_dim": self.content_dim,
                "controller_dim": self.controller_dim,
                "num_shifts": self.num_shifts,
                "use_content_addressing": self.use_content_addressing,
                "use_location_addressing": self.use_location_addressing,
                "sharpening_bias": self.sharpening_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# DifferentiableSelectCopy
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DifferentiableSelectCopy(keras.layers.Layer):
    """
    Differentiable layer for selecting and copying values between memory positions.

    This layer implements the read and write operations of a Neural Turing Machine,
    providing a foundation for differentiable memory-augmented neural networks.

    **Read Operation**::

        read_content = sum(read_weights * memory)

    **Write Operation**::

        memory_new = memory * (1 - w @ erase) + w @ add

    :param memory_size: Number of memory slots.
    :type memory_size: int
    :param content_dim: Dimensionality of content stored in each memory slot.
    :type content_dim: int
    :param controller_dim: Dimensionality of controller state that drives addressing.
    :type controller_dim: int
    :param num_read_heads: Number of parallel read heads. Defaults to 1.
    :type num_read_heads: int
    :param num_write_heads: Number of parallel write heads. Defaults to 1.
    :type num_write_heads: int
    :param num_shifts: Number of shift positions for location addressing.
        Must be odd. Defaults to 3.
    :type num_shifts: int
    :param use_content_addressing: Enable content-based addressing. Defaults to True.
    :type use_content_addressing: bool
    :param use_location_addressing: Enable location-based addressing. Defaults to True.
    :type use_location_addressing: bool
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
        content_dim: int,
        controller_dim: int,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
        num_shifts: int = 3,
        use_content_addressing: bool = True,
        use_location_addressing: bool = True,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {memory_size}")
        if content_dim <= 0:
            raise ValueError(f"content_dim must be positive, got {content_dim}")
        if controller_dim <= 0:
            raise ValueError(f"controller_dim must be positive, got {controller_dim}")
        if num_read_heads <= 0:
            raise ValueError(f"num_read_heads must be positive, got {num_read_heads}")
        if num_write_heads <= 0:
            raise ValueError(
                f"num_write_heads must be positive, got {num_write_heads}"
            )

        self.memory_size = memory_size
        self.content_dim = content_dim
        self.controller_dim = controller_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.num_shifts = num_shifts
        self.use_content_addressing = use_content_addressing
        self.use_location_addressing = use_location_addressing
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create read heads
        self.read_heads: list[DifferentiableAddressingHead] = []
        for i in range(self.num_read_heads):
            head = DifferentiableAddressingHead(
                memory_size=self.memory_size,
                content_dim=self.content_dim,
                controller_dim=self.controller_dim,
                num_shifts=self.num_shifts,
                use_content_addressing=self.use_content_addressing,
                use_location_addressing=self.use_location_addressing,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"read_head_{i}",
            )
            self.read_heads.append(head)

        # Create write heads with erase/add projections
        self.write_heads: list[DifferentiableAddressingHead] = []
        self.erase_projections: list[keras.layers.Dense] = []
        self.add_projections: list[keras.layers.Dense] = []

        for i in range(self.num_write_heads):
            head = DifferentiableAddressingHead(
                memory_size=self.memory_size,
                content_dim=self.content_dim,
                controller_dim=self.controller_dim,
                num_shifts=self.num_shifts,
                use_content_addressing=self.use_content_addressing,
                use_location_addressing=self.use_location_addressing,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"write_head_{i}",
            )
            self.write_heads.append(head)

            erase_proj = keras.layers.Dense(
                self.content_dim,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"erase_proj_{i}",
            )
            self.erase_projections.append(erase_proj)

            add_proj = keras.layers.Dense(
                self.content_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"add_proj_{i}",
            )
            self.add_projections.append(add_proj)

    def build(self, input_shape: tuple) -> None:
        """
        Build read and write heads.

        :param input_shape: Shape of memory input (batch, memory_size, content_dim).
        :type input_shape: tuple
        """
        controller_shape = (None, self.controller_dim)

        for head in self.read_heads:
            head.build(controller_shape)

        for i, head in enumerate(self.write_heads):
            head.build(controller_shape)
            self.erase_projections[i].build(controller_shape)
            self.add_projections[i].build(controller_shape)

        super().build(input_shape)

    def _read(
        self,
        memory: keras.KerasTensor,
        weights: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Read from memory using attention weights.

        :param memory: Memory tensor (batch, memory_size, content_dim).
        :type memory: keras.KerasTensor
        :param weights: Attention weights (batch, memory_size).
        :type weights: keras.KerasTensor
        :return: Read content (batch, content_dim).
        :rtype: keras.KerasTensor
        """
        return ops.einsum("bm,bmd->bd", weights, memory)

    def _write(
        self,
        memory: keras.KerasTensor,
        weights: keras.KerasTensor,
        erase_vector: keras.KerasTensor,
        add_vector: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Write to memory using erase-then-add mechanism.

        :param memory: Memory tensor (batch, memory_size, content_dim).
        :type memory: keras.KerasTensor
        :param weights: Write attention weights (batch, memory_size).
        :type weights: keras.KerasTensor
        :param erase_vector: Erase vector (batch, content_dim), values in [0, 1].
        :type erase_vector: keras.KerasTensor
        :param add_vector: Add vector (batch, content_dim).
        :type add_vector: keras.KerasTensor
        :return: Modified memory (batch, memory_size, content_dim).
        :rtype: keras.KerasTensor
        """
        erase_term = ops.einsum("bm,bd->bmd", weights, erase_vector)
        add_term = ops.einsum("bm,bd->bmd", weights, add_vector)
        memory_erased = memory * (1.0 - erase_term)
        memory_new = memory_erased + add_term
        return memory_new

    def call(
        self,
        memory: keras.KerasTensor,
        controller_state: keras.KerasTensor,
        previous_read_weights: list[keras.KerasTensor] | None = None,
        previous_write_weights: list[keras.KerasTensor] | None = None,
        training: bool | None = None,
    ) -> tuple[keras.KerasTensor, keras.KerasTensor, dict]:
        """
        Perform read and write operations on memory.

        :param memory: Memory tensor of shape (batch, memory_size, content_dim).
        :type memory: keras.KerasTensor
        :param controller_state: Controller output of shape (batch, controller_dim).
        :type controller_state: keras.KerasTensor
        :param previous_read_weights: Previous read attention weights per head.
        :type previous_read_weights: list[keras.KerasTensor] | None
        :param previous_write_weights: Previous write attention weights per head.
        :type previous_write_weights: list[keras.KerasTensor] | None
        :param training: Training mode flag.
        :type training: bool | None
        :return: Tuple of (new_memory, read_contents, state_dict).
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor, dict]
        """
        if previous_read_weights is None:
            previous_read_weights = [None] * self.num_read_heads
        if previous_write_weights is None:
            previous_write_weights = [None] * self.num_write_heads

        # Perform reads
        read_contents = []
        read_weights_list = []
        for i, head in enumerate(self.read_heads):
            weights = head(
                memory,
                controller_state,
                previous_weights=previous_read_weights[i],
                training=training,
            )
            content = self._read(memory, weights)
            read_contents.append(content)
            read_weights_list.append(weights)

        # Perform writes
        write_weights_list = []
        for i, head in enumerate(self.write_heads):
            weights = head(
                memory,
                controller_state,
                previous_weights=previous_write_weights[i],
                training=training,
            )

            erase_vector = self.erase_projections[i](controller_state)
            add_vector = self.add_projections[i](controller_state)

            memory = self._write(memory, weights, erase_vector, add_vector)
            write_weights_list.append(weights)

        # Concatenate read contents from all heads
        if len(read_contents) > 1:
            read_output = ops.concatenate(read_contents, axis=-1)
        else:
            read_output = read_contents[0]

        state_dict = {
            "read_weights": read_weights_list,
            "write_weights": write_weights_list,
        }

        return memory, read_output, state_dict

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[tuple[int | None, ...], tuple[int | None, ...], dict]:
        """
        Compute output shapes.

        :param input_shape: Shape of memory input (batch, memory_size, content_dim).
        :type input_shape: tuple
        :return: Tuple of (memory_shape, read_output_shape, state_dict_shapes).
        :rtype: tuple
        """
        batch = input_shape[0] if len(input_shape) >= 1 else None
        memory_shape = (batch, self.memory_size, self.content_dim)
        read_output_shape = (batch, self.num_read_heads * self.content_dim)

        state_dict_shapes = {
            "read_weights": [
                (batch, self.memory_size) for _ in range(self.num_read_heads)
            ],
            "write_weights": [
                (batch, self.memory_size) for _ in range(self.num_write_heads)
            ],
        }

        return memory_shape, read_output_shape, state_dict_shapes

    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "content_dim": self.content_dim,
                "controller_dim": self.controller_dim,
                "num_read_heads": self.num_read_heads,
                "num_write_heads": self.num_write_heads,
                "num_shifts": self.num_shifts,
                "use_content_addressing": self.use_content_addressing,
                "use_location_addressing": self.use_location_addressing,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# SimpleSelectCopy
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SimpleSelectCopy(keras.layers.Layer):
    """
    Simplified differentiable select-copy layer for learning input-output mappings.

    This is a minimal version that learns to:
        1. Select one position from input via soft attention
        2. Copy the content to one position in output via soft attention

    Unlike the full DifferentiableSelectCopy, this layer uses simple learned
    query vectors without the full NTM addressing machinery.

    :param input_size: Number of input positions to select from.
    :type input_size: int
    :param output_size: Number of output positions to write to.
    :type output_size: int
    :param content_dim: Dimensionality of content vectors.
    :type content_dim: int
    :param num_copies: Number of parallel select-copy operations. Defaults to 1.
    :type num_copies: int
    :param temperature: Softmax temperature for attention sharpness. Defaults to 1.0.
    :type temperature: float
    :param use_content_query: If True, use input content to generate queries.
    :type use_content_query: bool
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
        input_size: int,
        output_size: int,
        content_dim: int,
        num_copies: int = 1,
        temperature: float = 1.0,
        use_content_query: bool = True,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if content_dim <= 0:
            raise ValueError(f"content_dim must be positive, got {content_dim}")
        if num_copies <= 0:
            raise ValueError(f"num_copies must be positive, got {num_copies}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.input_size = input_size
        self.output_size = output_size
        self.content_dim = content_dim
        self.num_copies = num_copies
        self.temperature = temperature
        self.use_content_query = use_content_query
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Create sub-layers
        self.read_query_layers: list[keras.layers.Dense] = []
        self.write_query_layers: list[keras.layers.Dense] = []
        self.content_transforms: list[keras.layers.Dense] = []

        for i in range(self.num_copies):
            if self.use_content_query:
                read_q = keras.layers.Dense(
                    self.content_dim,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"read_query_{i}",
                )
                write_q = keras.layers.Dense(
                    self.content_dim,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"write_query_{i}",
                )
                self.read_query_layers.append(read_q)
                self.write_query_layers.append(write_q)

            transform = keras.layers.Dense(
                self.content_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"content_transform_{i}",
            )
            self.content_transforms.append(transform)

        self.read_key_proj = keras.layers.Dense(
            self.content_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="read_key_projection",
        )

        # Learned query weights (for non-content-query mode) - created in build
        self.read_query_weights: list[keras.Variable] = []
        self.write_query_weights: list[keras.Variable] = []
        self.output_position_embeddings: keras.Variable | None = None

    def build(self, input_shape: tuple) -> None:
        """
        Build selection and placement mechanisms.

        :param input_shape: Shape of input (batch, input_size, content_dim).
        :type input_shape: tuple
        """
        # Build read key projection
        self.read_key_proj.build(input_shape)

        # Build query layers and transforms
        pooled_shape = (None, self.content_dim)
        for i in range(self.num_copies):
            if self.use_content_query:
                self.read_query_layers[i].build(pooled_shape)
                self.write_query_layers[i].build(pooled_shape)
            self.content_transforms[i].build(pooled_shape)

        # Output position embeddings
        self.output_position_embeddings = self.add_weight(
            name="output_position_embeddings",
            shape=(self.output_size, self.content_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # Learned query weights (for non-content-query mode)
        if not self.use_content_query:
            for i in range(self.num_copies):
                read_q = self.add_weight(
                    name=f"read_query_weight_{i}",
                    shape=(1, self.content_dim),
                    initializer=self.kernel_initializer,
                    trainable=True,
                )
                write_q = self.add_weight(
                    name=f"write_query_weight_{i}",
                    shape=(1, self.content_dim),
                    initializer=self.kernel_initializer,
                    trainable=True,
                )
                self.read_query_weights.append(read_q)
                self.write_query_weights.append(write_q)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: bool | None = None,
    ) -> tuple[keras.KerasTensor, dict]:
        """
        Perform select-copy operations.

        :param inputs: Input tensor of shape (batch, input_size, content_dim).
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: bool | None
        :return: Tuple of (output, attention_info).
        :rtype: tuple[keras.KerasTensor, dict]
        """
        batch_size = ops.shape(inputs)[0]

        # Initialize output
        output = ops.zeros((batch_size, self.output_size, self.content_dim))

        # Compute input keys
        input_keys = self.read_key_proj(inputs)

        # Broadcast output position embeddings
        output_keys = ops.broadcast_to(
            self.output_position_embeddings[None, :, :],
            (batch_size, self.output_size, self.content_dim),
        )

        read_weights_list = []
        write_weights_list = []
        scale = ops.sqrt(ops.cast(self.content_dim, "float32"))

        for i in range(self.num_copies):
            # Generate read query
            if self.use_content_query:
                pooled = ops.mean(inputs, axis=1)
                read_query = self.read_query_layers[i](pooled)
                write_query = self.write_query_layers[i](pooled)
            else:
                read_query = ops.broadcast_to(
                    self.read_query_weights[i], (batch_size, self.content_dim)
                )
                write_query = ops.broadcast_to(
                    self.write_query_weights[i], (batch_size, self.content_dim)
                )

            # Compute read attention
            read_scores = ops.einsum("bd,bnd->bn", read_query, input_keys) / scale
            read_weights = ops.softmax(read_scores / self.temperature, axis=-1)
            read_weights_list.append(read_weights)

            # Read content
            read_content = ops.einsum("bn,bnd->bd", read_weights, inputs)

            # Transform content
            transformed_content = self.content_transforms[i](read_content)

            # Compute write attention
            write_scores = ops.einsum("bd,bmd->bm", write_query, output_keys) / scale
            write_weights = ops.softmax(write_scores / self.temperature, axis=-1)
            write_weights_list.append(write_weights)

            # Write to output (additive)
            write_term = ops.einsum("bm,bd->bmd", write_weights, transformed_content)
            output = output + write_term

        attention_info = {
            "read_weights": read_weights_list,
            "write_weights": write_weights_list,
        }

        return output, attention_info

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[tuple[int | None, ...], dict]:
        """
        Compute output shape.

        :param input_shape: Shape of input (batch, input_size, content_dim).
        :type input_shape: tuple
        :return: Tuple of (output_shape, attention_info_shapes).
        :rtype: tuple
        """
        batch = input_shape[0] if len(input_shape) >= 1 else None
        output_shape = (batch, self.output_size, self.content_dim)

        attention_info_shapes = {
            "read_weights": [
                (batch, self.input_size) for _ in range(self.num_copies)
            ],
            "write_weights": [
                (batch, self.output_size) for _ in range(self.num_copies)
            ],
        }

        return output_shape, attention_info_shapes

    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "content_dim": self.content_dim,
                "num_copies": self.num_copies,
                "temperature": self.temperature,
                "use_content_query": self.use_content_query,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config
