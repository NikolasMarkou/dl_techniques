"""
Differentiable Select-Copy Layer for Neural Turing Machine Foundation.

This module implements a fully differentiable layer that can:
1. Select (read) content from an arbitrary position in input memory
2. Copy (write) that content to a learnable position in output memory

The addressing mechanism combines:
- Content-based addressing via cosine similarity
- Location-based addressing via shift convolution
- Interpolation gating between content and location modes
- Sharpening to focus attention distributions

Based on: "Neural Turing Machines" (Graves et al., 2014)
"""

import keras
from keras import ops

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DifferentiableAddressingHead(keras.layers.Layer):
    """
    Differentiable addressing head implementing NTM-style memory addressing.

    This head computes attention weights over memory locations using a combination
    of content-based and location-based addressing mechanisms.

    :param memory_size: Number of memory slots to address.
    :type memory_size: int
    :param content_dim: Dimensionality of memory content vectors.
    :type content_dim: int
    :param num_shifts: Number of shift positions for location-based addressing.
        Must be odd (e.g., 3 for shifts of -1, 0, +1). Defaults to 3.
    :type num_shifts: int
    :param use_content_addressing: Whether to use content-based addressing.
        Defaults to True.
    :type use_content_addressing: bool
    :param use_location_addressing: Whether to use location-based addressing
        (shift convolution). Defaults to True.
    :type use_location_addressing: bool
    :param sharpening_bias: Initial bias for sharpening parameter gamma.
        Higher values produce sharper attention. Defaults to 1.0.
    :type sharpening_bias: float
    :param kernel_initializer: Initializer for dense layer kernels.
        Defaults to "glorot_uniform".
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for dense layer biases.
        Defaults to "zeros".
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layer kernels.
        Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            memory_size: int,
            content_dim: int,
            num_shifts: int = 3,
            use_content_addressing: bool = True,
            use_location_addressing: bool = True,
            sharpening_bias: float = 1.0,
            kernel_initializer: str = "glorot_uniform",
            bias_initializer: str = "zeros",
            kernel_regularizer: keras.regularizers.Regularizer | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if num_shifts % 2 == 0:
            raise ValueError(
                f"num_shifts must be odd for symmetric shifts, got {num_shifts}"
            )

        self.memory_size = memory_size
        self.content_dim = content_dim
        self.num_shifts = num_shifts
        self.use_content_addressing = use_content_addressing
        self.use_location_addressing = use_location_addressing
        self.sharpening_bias = sharpening_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape: tuple) -> None:
        """
        Build the addressing head components.

        :param input_shape: Shape of input tensor (batch, memory_size, content_dim).
        :type input_shape: tuple
        """
        # Key projection for content-based addressing
        if self.use_content_addressing:
            self.key_proj = keras.layers.Dense(
                self.content_dim,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="key_projection",
            )
            # Beta (key strength) - controls sharpness of content addressing
            self.beta_proj = keras.layers.Dense(
                1,
                activation="softplus",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="beta_projection",
            )

        # Interpolation gate (blend content vs location)
        if self.use_content_addressing and self.use_location_addressing:
            self.gate_proj = keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="interpolation_gate",
            )

        # Shift distribution for location-based addressing
        if self.use_location_addressing:
            self.shift_proj = keras.layers.Dense(
                self.num_shifts,
                activation="softmax",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="shift_distribution",
            )

        # Sharpening parameter gamma (must be >= 1)
        self.gamma_proj = keras.layers.Dense(
            1,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=keras.initializers.Constant(self.sharpening_bias),
            kernel_regularizer=self.kernel_regularizer,
            name="sharpening_gamma",
        )

        # Learnable initial location weights (for pure location addressing)
        self.initial_weights = self.add_weight(
            name="initial_location_weights",
            shape=(1, self.memory_size),
            initializer=keras.initializers.Constant(1.0 / self.memory_size),
            trainable=True,
        )

        super().build(input_shape)

    def _cosine_similarity(
            self,
            query: keras.KerasTensor,
            keys: keras.KerasTensor,
            eps: float = 1e-8,
    ) -> keras.KerasTensor:
        """
        Compute cosine similarity between query and keys.

        :param query: Query tensor of shape (batch, content_dim).
        :type query: keras.KerasTensor
        :param keys: Keys tensor of shape (batch, memory_size, content_dim).
        :type keys: keras.KerasTensor
        :param eps: Small constant for numerical stability.
        :type eps: float
        :return: Similarity scores of shape (batch, memory_size).
        :rtype: keras.KerasTensor
        """
        # Normalize query: (batch, content_dim)
        query_norm = query / (ops.norm(query, axis=-1, keepdims=True) + eps)

        # Normalize keys: (batch, memory_size, content_dim)
        keys_norm = keys / (ops.norm(keys, axis=-1, keepdims=True) + eps)

        # Compute dot product: (batch, memory_size)
        similarity = ops.einsum("bd,bmd->bm", query_norm, keys_norm)

        return similarity

    def _circular_convolve(
            self,
            weights: keras.KerasTensor,
            shift_kernel: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Apply circular convolution for location-based shifting.

        :param weights: Attention weights of shape (batch, memory_size).
        :type weights: keras.KerasTensor
        :param shift_kernel: Shift distribution of shape (batch, num_shifts).
        :type shift_kernel: keras.KerasTensor
        :return: Shifted weights of shape (batch, memory_size).
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(weights)[0]
        half_shift = self.num_shifts // 2

        # Build shift matrix for circular convolution
        # For each shift offset, roll weights and weight by shift probability
        shifted_weights = ops.zeros_like(weights)

        for i in range(self.num_shifts):
            shift_offset = i - half_shift  # e.g., -1, 0, +1 for num_shifts=3
            rolled = ops.roll(weights, shift=-shift_offset, axis=-1)
            # Weight by shift probability
            shift_weight = shift_kernel[:, i: i + 1]  # (batch, 1)
            shifted_weights = shifted_weights + rolled * shift_weight

        return shifted_weights

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
            Shape (batch, memory_size). Defaults to uniform or learned initial weights.
        :type previous_weights: keras.KerasTensor or None
        :param training: Training mode flag.
        :type training: bool or None
        :return: Attention weights of shape (batch, memory_size).
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(memory)[0]

        # Initialize previous weights if not provided
        if previous_weights is None:
            previous_weights = ops.broadcast_to(
                self.initial_weights, (batch_size, self.memory_size)
            )

        # Step 1: Content-based addressing
        if self.use_content_addressing:
            # Generate key from controller state
            key = self.key_proj(controller_state)  # (batch, content_dim)
            beta = self.beta_proj(controller_state) + 1.0  # (batch, 1), >= 1

            # Compute content-based weights
            similarity = self._cosine_similarity(key, memory)  # (batch, memory_size)
            content_weights = ops.softmax(beta * similarity, axis=-1)
        else:
            content_weights = previous_weights

        # Step 2: Interpolation gate (blend content and location)
        if self.use_content_addressing and self.use_location_addressing:
            gate = self.gate_proj(controller_state)  # (batch, 1)
            gated_weights = gate * content_weights + (1.0 - gate) * previous_weights
        elif self.use_content_addressing:
            gated_weights = content_weights
        else:
            gated_weights = previous_weights

        # Step 3: Location-based shift convolution
        if self.use_location_addressing:
            shift_kernel = self.shift_proj(controller_state)  # (batch, num_shifts)
            shifted_weights = self._circular_convolve(gated_weights, shift_kernel)
        else:
            shifted_weights = gated_weights

        # Step 4: Sharpening
        gamma = ops.softplus(self.gamma_proj(controller_state)) + 1.0  # (batch, 1), >= 1
        sharpened = ops.power(shifted_weights + 1e-8, gamma)
        weights = sharpened / (ops.sum(sharpened, axis=-1, keepdims=True) + 1e-8)

        return weights

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "memory_size": self.memory_size,
                "content_dim": self.content_dim,
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

@keras.saving.register_keras_serializable()
class DifferentiableSelectCopy(keras.layers.Layer):
    """
    Differentiable layer for selecting and copying values between memory positions.

    This layer implements the read and write operations of a Neural Turing Machine,
    providing a foundation for differentiable memory-augmented neural networks.

    **Read Operation:**
        Uses soft attention to extract a weighted combination of memory content.
        ``read_content = Σ(read_weights * memory)``

    **Write Operation:**
        Uses erase-then-add mechanism to modify memory at attended positions.
        ``memory_new = memory * (1 - w⊗erase) + w⊗add``

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
    :param kernel_initializer: Initializer for dense layers. Defaults to "glorot_uniform".
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.

    Example::

        # Create layer
        layer = DifferentiableSelectCopy(
            memory_size=128,
            content_dim=64,
            controller_dim=256,
            num_read_heads=1,
            num_write_heads=1,
        )

        # Input: memory and controller state
        memory = keras.random.normal((batch, 128, 64))
        controller_state = keras.random.normal((batch, 256))

        # Forward pass
        new_memory, read_content = layer(memory, controller_state)
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
            kernel_initializer: str = "glorot_uniform",
            kernel_regularizer: keras.regularizers.Regularizer | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.memory_size = memory_size
        self.content_dim = content_dim
        self.controller_dim = controller_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.num_shifts = num_shifts
        self.use_content_addressing = use_content_addressing
        self.use_location_addressing = use_location_addressing
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape: tuple) -> None:
        """
        Build read and write heads.

        :param input_shape: Shape of memory input (batch, memory_size, content_dim).
        :type input_shape: tuple
        """
        # Build read heads
        self.read_heads = []
        for i in range(self.num_read_heads):
            head = DifferentiableAddressingHead(
                memory_size=self.memory_size,
                content_dim=self.content_dim,
                num_shifts=self.num_shifts,
                use_content_addressing=self.use_content_addressing,
                use_location_addressing=self.use_location_addressing,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"read_head_{i}",
            )
            self.read_heads.append(head)

        # Build write heads
        self.write_heads = []
        self.erase_projections = []
        self.add_projections = []

        for i in range(self.num_write_heads):
            head = DifferentiableAddressingHead(
                memory_size=self.memory_size,
                content_dim=self.content_dim,
                num_shifts=self.num_shifts,
                use_content_addressing=self.use_content_addressing,
                use_location_addressing=self.use_location_addressing,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"write_head_{i}",
            )
            self.write_heads.append(head)

            # Erase vector projection (sigmoid for values in [0, 1])
            erase_proj = keras.layers.Dense(
                self.content_dim,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"erase_proj_{i}",
            )
            self.erase_projections.append(erase_proj)

            # Add vector projection
            add_proj = keras.layers.Dense(
                self.content_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"add_proj_{i}",
            )
            self.add_projections.append(add_proj)

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
        # Weighted sum over memory positions
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
        # Compute erase term: w ⊗ e gives (batch, memory_size, content_dim)
        erase_term = ops.einsum("bm,bd->bmd", weights, erase_vector)

        # Compute add term: w ⊗ a gives (batch, memory_size, content_dim)
        add_term = ops.einsum("bm,bd->bmd", weights, add_vector)

        # Apply erase-then-add: M_new = M * (1 - erase_term) + add_term
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
            List of tensors with shape (batch, memory_size). Defaults to None.
        :type previous_read_weights: list[keras.KerasTensor] or None
        :param previous_write_weights: Previous write attention weights per head.
            List of tensors with shape (batch, memory_size). Defaults to None.
        :type previous_write_weights: list[keras.KerasTensor] or None
        :param training: Training mode flag.
        :type training: bool or None
        :return: Tuple of (new_memory, read_contents, state_dict) where:
            - new_memory: Modified memory (batch, memory_size, content_dim)
            - read_contents: Concatenated read vectors (batch, num_read_heads * content_dim)
            - state_dict: Dictionary with 'read_weights' and 'write_weights' lists
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor, dict]
        """
        # Initialize previous weights if not provided
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

            # Generate erase and add vectors from controller state
            erase_vector = self.erase_projections[i](controller_state)
            add_vector = self.add_projections[i](controller_state)

            # Write to memory
            memory = self._write(memory, weights, erase_vector, add_vector)
            write_weights_list.append(weights)

        # Concatenate read contents from all heads
        if len(read_contents) > 1:
            read_output = ops.concatenate(read_contents, axis=-1)
        else:
            read_output = read_contents[0]

        # Return state for recurrent usage
        state_dict = {
            "read_weights": read_weights_list,
            "write_weights": write_weights_list,
        }

        return memory, read_output, state_dict

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
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
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

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
    :param temperature: Softmax temperature for attention sharpness.
        Lower values produce sharper attention. Defaults to 1.0.
    :type temperature: float
    :param use_content_query: If True, use input content to generate queries.
        If False, use purely learned position queries. Defaults to True.
    :type use_content_query: bool
    :param kernel_initializer: Initializer for dense layers. Defaults to "glorot_uniform".
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for dense layers. Defaults to None.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.

    Example::

        layer = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=32,
            num_copies=2,
        )

        # Input sequence
        x = keras.random.normal((batch, 10, 32))

        # Output: same shape with content copied between positions
        output, attention_info = layer(x)
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            content_dim: int,
            num_copies: int = 1,
            temperature: float = 1.0,
            use_content_query: bool = True,
            kernel_initializer: str = "glorot_uniform",
            kernel_regularizer: keras.regularizers.Regularizer | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.content_dim = content_dim
        self.num_copies = num_copies
        self.temperature = temperature
        self.use_content_query = use_content_query
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape: tuple) -> None:
        """
        Build selection and placement mechanisms.

        :param input_shape: Shape of input (batch, input_size, content_dim).
        :type input_shape: tuple
        """
        # For each copy operation, we need:
        # 1. Read query generator (which input position to select)
        # 2. Write query generator (which output position to write)

        self.read_queries = []
        self.write_queries = []
        self.content_transforms = []

        for i in range(self.num_copies):
            if self.use_content_query:
                # Learn to generate queries from pooled input content
                read_q = keras.layers.Dense(
                    self.content_dim,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"read_query_{i}",
                )
                write_q = keras.layers.Dense(
                    self.content_dim,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"write_query_{i}",
                )
            else:
                # Pure learned query vectors
                read_q = self.add_weight(
                    name=f"read_query_{i}",
                    shape=(1, self.content_dim),
                    initializer=self.kernel_initializer,
                    trainable=True,
                )
                write_q = self.add_weight(
                    name=f"write_query_{i}",
                    shape=(1, self.content_dim),
                    initializer=self.kernel_initializer,
                    trainable=True,
                )

            self.read_queries.append(read_q)
            self.write_queries.append(write_q)

            # Optional content transform before writing
            transform = keras.layers.Dense(
                self.content_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"content_transform_{i}",
            )
            self.content_transforms.append(transform)

        # Key projections for attention (shared across copies)
        self.read_key_proj = keras.layers.Dense(
            self.content_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="read_key_projection",
        )

        # For write attention, we need position embeddings for output
        self.output_position_embeddings = self.add_weight(
            name="output_position_embeddings",
            shape=(self.output_size, self.content_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )

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
        :type training: bool or None
        :return: Tuple of (output, attention_info) where:
            - output: Output tensor (batch, output_size, content_dim)
            - attention_info: Dict with 'read_weights' and 'write_weights' lists
        :rtype: tuple[keras.KerasTensor, dict]
        """
        batch_size = ops.shape(inputs)[0]

        # Start with zeros output
        output = ops.zeros((batch_size, self.output_size, self.content_dim))

        # Compute input keys once
        input_keys = self.read_key_proj(inputs)  # (batch, input_size, content_dim)

        # Output positions for write attention
        output_keys = ops.broadcast_to(
            self.output_position_embeddings[None, :, :],
            (batch_size, self.output_size, self.content_dim),
        )

        read_weights_list = []
        write_weights_list = []

        for i in range(self.num_copies):
            # Generate read query
            if self.use_content_query:
                # Pool input and generate query
                pooled = ops.mean(inputs, axis=1)  # (batch, content_dim)
                read_query = self.read_queries[i](pooled)  # (batch, content_dim)
                write_query = self.write_queries[i](pooled)  # (batch, content_dim)
            else:
                # Use learned query vectors
                read_query = ops.broadcast_to(
                    self.read_queries[i], (batch_size, self.content_dim)
                )
                write_query = ops.broadcast_to(
                    self.write_queries[i], (batch_size, self.content_dim)
                )

            # Compute read attention (which input to select)
            read_scores = ops.einsum(
                "bd,bnd->bn", read_query, input_keys
            ) / ops.sqrt(ops.cast(self.content_dim, "float32"))
            read_weights = ops.softmax(read_scores / self.temperature, axis=-1)
            read_weights_list.append(read_weights)

            # Read content
            read_content = ops.einsum(
                "bn,bnd->bd", read_weights, inputs
            )  # (batch, content_dim)

            # Transform content
            transformed_content = self.content_transforms[i](read_content)

            # Compute write attention (which output position to write to)
            write_scores = ops.einsum(
                "bd,bmd->bm", write_query, output_keys
            ) / ops.sqrt(ops.cast(self.content_dim, "float32"))
            write_weights = ops.softmax(write_scores / self.temperature, axis=-1)
            write_weights_list.append(write_weights)

            # Write to output (additive)
            write_term = ops.einsum(
                "bm,bd->bmd", write_weights, transformed_content
            )
            output = output + write_term

        attention_info = {
            "read_weights": read_weights_list,
            "write_weights": write_weights_list,
        }

        return output, attention_info

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
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
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

# ---------------------------------------------------------------------
