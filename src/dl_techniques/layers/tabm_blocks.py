"""
Deep Ensembling is a powerful technique for improving model robustness, accuracy, and
uncertainty estimation. The standard approach involves training multiple independent
models and averaging their predictions. However, this can be computationally expensive
and slow. This module implements several building blocks for creating "implicit" or
"batched" ensembles, where all `k` ensemble members are represented and trained
simultaneously within a single model, using an additional dimension.

The key to this module's efficiency is its clever use of weight sharing and
specialized tensor operations (`einsum`) to perform parallel computations across all
ensemble members with minimal overhead.

The module contains the following key components:

1.  **`ScaleEnsemble` (Learnable Member Scaling):**
    -   A simple but powerful layer that applies a learnable, per-feature scaling
        factor to each ensemble member.
    -   This allows the model to learn the relative importance of different features
        for each individual member of the ensemble, providing a simple mechanism for
        each member to specialize.

2.  **`LinearEfficientEnsemble` (The Core Ensemble Layer):**
    -   This is the workhorse of the module. It implements an efficient linear
        transformation for `k` ensemble members.
    -   **Weight Sharing:** It uses a single, shared `kernel` (weight matrix) for the
        main linear projection across all ensemble members. This is a massive saving
        in parameters compared to having `k` separate weight matrices.
    -   **Rank-1 Perturbations:** To allow each ensemble member to learn a unique
        function, it applies learnable, rank-1 scaling factors (`r` for input, `s` for
        output) to the shared kernel. This is equivalent to applying a unique diagonal
        matrix transformation to the input and output for each member, providing
        diversity without the full cost of independent weights.

3.  **`NLinear` (Fully Independent Parallel Layers):**
    -   This layer provides an alternative to the efficient, weight-sharing approach.
        It implements `n` truly independent linear layers that are processed in parallel.
    -   It uses a single weight tensor of shape `(n, input_dim, output_dim)` and
        `einsum` to perform `n` independent matrix multiplications in one operation.
    -   This is useful for the final output heads of an ensemble, where each member needs
        its own independent classifier.

4.  **`MLPBlock` and `TabMBackbone` (High-Level Abstractions):**
    -   These are convenience layers that assemble the lower-level components into
        standard MLP blocks and a full MLP backbone.
    -   They can operate in either a "plain" mode (a single model) or an "ensemble"
        mode (with `k` members) by simply setting the `k` parameter, making it easy
        to switch between standard and ensemble architectures.
    -   In ensemble mode `ensemble_type` chooses which of the two mechanisms above
        realizes the members: `'efficient'` (shared kernel + rank-1 scaling) or
        `'packed'` (`NLinear`, `k` independent kernels). The packed form is what
        the efficient form is supposed to approximate, so it belongs in the same
        switch rather than only in the output head.

A note on the scaling-vector initialization, since it decides whether the members
are different functions at all: `LinearEfficientEnsemble` and `ScaleEnsemble` take
an `init_distribution` of `'random-signs'` (draw from {-1, +1}, the paper's choice),
`'normal'` (N(1, 0.1)) or `'ones'`. Under `'ones'` every member's effective weight
matrix is identical at initialization — a legitimate setting for a variant whose
diversity comes from elsewhere, and a silent degeneracy if it is not.
"""

import keras
from keras import ops
from typing import Dict, List, Literal, Optional, Tuple, Union, Any
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------

EnsembleInitDistribution = Literal['ones', 'normal', 'random-signs']


@keras.saving.register_keras_serializable()
class RandomSigns(keras.initializers.Initializer):
    """
    Draw each element uniformly from :math:`\\{-1, +1\\}`.

    This is the initializer the TabM paper uses for the per-member scaling
    vectors. It is the only one of the three options that guarantees every
    ensemble member starts at a distinct, non-degenerate, unit-magnitude
    perturbation of the shared kernel: a normal draw clusters members near the
    mean, and a constant draw makes them identical.

    :param seed: Optional seed for reproducible draws.
    :type seed: int or None
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        # Delegate the draw to a stock initializer rather than calling
        # `keras.random.*` directly: inside `add_weight` the global seed
        # generator has no variable to update and the direct call raises.
        self._uniform = keras.initializers.RandomUniform(
            minval=-1.0, maxval=1.0, seed=seed
        )

    def __call__(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        dtype = dtype or keras.config.floatx()
        u = self._uniform(shape, dtype=dtype)
        return ops.where(u >= 0.0, ops.ones_like(u), -ops.ones_like(u))

    def get_config(self) -> Dict[str, Any]:
        return {"seed": self.seed}


def _ensemble_scaling_initializer(
        init_distribution: EnsembleInitDistribution
) -> keras.initializers.Initializer:
    """
    Resolve an ensemble scaling-vector initializer by name.

    :param init_distribution: One of ``'ones'``, ``'normal'``, ``'random-signs'``.
    :type init_distribution: str

    :return: The matching initializer instance.
    :rtype: keras.initializers.Initializer

    :raises ValueError: If ``init_distribution`` is not one of the three names.
    """
    if init_distribution == 'ones':
        return keras.initializers.Ones()
    if init_distribution == 'normal':
        return keras.initializers.RandomNormal(mean=1.0, stddev=0.1)
    if init_distribution == 'random-signs':
        return RandomSigns()
    raise ValueError(
        f"init_distribution must be one of 'ones', 'normal', 'random-signs'; "
        f"got {init_distribution!r}"
    )

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ScaleEnsemble(keras.layers.Layer):
    """
    Learnable per-feature scaling for ensemble members.

    This layer applies a learnable, per-feature scaling factor to each of ``k``
    ensemble members, allowing each member to specialize by learning the
    relative importance of different features. The operation is
    ``output = input * weight`` with broadcasting over the batch dimension.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────┐
        │  Input [B, K, D]            │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │  Element-wise multiply      │
        │  with weight [K, D]         │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │  Output [B, K, D]           │
        └─────────────────────────────┘

    :param k: Number of ensemble members.
    :type k: int
    :param input_dim: Input feature dimension.
    :type input_dim: int
    :param init_distribution: Initialization distribution for the scaling weights
        (``'ones'``, ``'normal'`` or ``'random-signs'``). ``'random-signs'`` draws
        from :math:`\\{-1, +1\\}` and ``'normal'`` from :math:`\\mathcal{N}(1, 0.1)`.
    :type init_distribution: str
    :param kernel_regularizer: Optional regularizer for scaling weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            k: int,
            input_dim: int,
            init_distribution: EnsembleInitDistribution = 'normal',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.input_dim = input_dim
        self.init_distribution = init_distribution
        # The scaling weights have exactly one job — differ per member — so the
        # initializer is fully determined by ``init_distribution``; there is no
        # separate ``kernel_initializer`` knob to contradict it.
        self.kernel_initializer = _ensemble_scaling_initializer(init_distribution)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the scaling weights with proper initialization."""
        self.weight = self.add_weight(
            shape=(self.k, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='ensemble_weight'
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply ensemble scaling to inputs.

            :param inputs: Input tensor of shape (batch_size, k, input_dim).
            :type inputs: keras.KerasTensor

            :return: Scaled tensor of shape (batch_size, k, input_dim).
            :rtype: keras.KerasTensor
        """
        # Efficient broadcasting: inputs (B, K, D) * weight (K, D) -> (B, K, D)
        return ops.multiply(inputs, ops.expand_dims(self.weight, axis=0))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "k": self.k,
            "input_dim": self.input_dim,
            "init_distribution": self.init_distribution,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LinearEfficientEnsemble(keras.layers.Layer):
    """
    Efficient ensemble linear layer with rank-1 perturbations.

    This layer performs a shared linear transformation across ``k`` ensemble
    members, with optional learnable input scaling (``r``) and output scaling
    (``s``) vectors per member. The result is equivalent to applying unique
    diagonal transformations to the shared kernel for each member, providing
    ensemble diversity without the cost of ``k`` independent weight matrices.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, K, D_in]              │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Opt. input scaling: x * r[K,D]  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Shared matmul: x @ W            │
        │  W [D_in, units] (shared)        │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Opt. output scaling: x * s[K,U] │
        │  + opt. bias [K, units]          │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, K, units]            │
        └──────────────────────────────────┘

    :param units: Output dimension.
    :type units: int
    :param k: Number of ensemble members.
    :type k: int
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param ensemble_scaling_in: Whether to use input scaling.
    :type ensemble_scaling_in: bool
    :param ensemble_scaling_out: Whether to use output scaling.
    :type ensemble_scaling_out: bool
    :param init_distribution: How the per-member scaling vectors ``r`` and ``s``
        are initialized. ``'random-signs'`` draws from :math:`\\{-1, +1\\}` (the
        paper's default: members start as distinct, unit-magnitude sign patterns),
        ``'normal'`` draws from :math:`\\mathcal{N}(1, 0.1)`, and ``'ones'``
        starts every member at the identity perturbation — under which all ``k``
        members share one effective weight matrix at initialization.
    :type init_distribution: str
    :param kernel_initializer: Initializer for the main weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            units: int,
            k: int,
            use_bias: bool = True,
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.k = k
        self.use_bias = use_bias
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        self.scaling_initializer = _ensemble_scaling_initializer(init_distribution)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the ensemble linear layer weights."""
        input_dim = input_shape[-1]

        # Main weight matrix shared across ensemble members
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='kernel'
        )

        # Input scaling weights
        if self.ensemble_scaling_in:
            self.r = self.add_weight(
                shape=(self.k, input_dim),
                initializer=self.scaling_initializer,
                trainable=True,
                name='input_scaling'
            )

        # Output scaling weights
        if self.ensemble_scaling_out:
            self.s = self.add_weight(
                shape=(self.k, self.units),
                initializer=self.scaling_initializer,
                trainable=True,
                name='output_scaling'
            )

        # Bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.k, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name='bias'
            )

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass through efficient ensemble layer.

            :param inputs: Input tensor of shape (batch_size, k, input_dim).
            :type inputs: keras.KerasTensor

            :return: Output tensor of shape (batch_size, k, units).
            :rtype: keras.KerasTensor
        """
        x = inputs

        # Apply input scaling if enabled
        if self.ensemble_scaling_in:
            x = ops.multiply(x, ops.expand_dims(self.r, axis=0))

        # Apply main linear transformation efficiently
        # Use einsum for better performance and clarity
        x = ops.einsum('bki,iu->bku', x, self.kernel)

        # Apply output scaling if enabled
        if self.ensemble_scaling_out:
            x = ops.multiply(x, ops.expand_dims(self.s, axis=0))

        # Add bias if enabled
        if self.use_bias:
            x = ops.add(x, ops.expand_dims(self.bias, axis=0))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "use_bias": self.use_bias,
            "ensemble_scaling_in": self.ensemble_scaling_in,
            "ensemble_scaling_out": self.ensemble_scaling_out,
            "init_distribution": self.init_distribution,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NLinear(keras.layers.Layer):
    """
    N fully independent parallel linear layers using einsum.

    This layer implements ``n`` truly independent linear layers processed in
    parallel via a single weight tensor of shape ``(n, input_dim, output_dim)``
    and ``einsum``. Useful for final output heads of an ensemble where each
    member needs its own independent classifier.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, N, D_in]             │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  einsum('bni,nio->bno',         │
        │         input, kernels)         │
        │  kernels [N, D_in, D_out]       │
        │  + opt. bias [N, D_out]         │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, N, D_out]           │
        └─────────────────────────────────┘

    :param n: Number of parallel linear layers.
    :type n: int
    :param input_dim: Input dimension, or ``None`` to infer it from the input
        shape at build time.
    :type input_dim: int
    :param output_dim: Output dimension per linear layer.
    :type output_dim: int
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            n: int,
            input_dim: Optional[int],
            output_dim: int,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n = n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the parallel linear layer weights."""
        # ``input_dim=None`` defers the fan-in to build time, which is what lets
        # a packed MLPBlock construct its NLinear in __init__ (where the input
        # width is not yet known) instead of lazily inside build().
        if self.input_dim is None:
            self.input_dim = input_shape[-1]

        self.kernels = self.add_weight(
            shape=(self.n, self.input_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='kernels'
        )

        if self.use_bias:
            self.biases = self.add_weight(
                shape=(self.n, self.output_dim),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name='biases'
            )

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass through N parallel linear layers.

            :param inputs: Input tensor of shape (batch_size, n, input_dim).
            :type inputs: keras.KerasTensor

            :return: Output tensor of shape (batch_size, n, output_dim).
            :rtype: keras.KerasTensor
        """
        # Efficient parallel matrix multiplication using einsum
        outputs = ops.einsum('bni,nio->bno', inputs, self.kernels)

        if self.use_bias:
            outputs = ops.add(outputs, ops.expand_dims(self.biases, axis=0))

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.n, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "n": self.n,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

# Registered under an explicit package: the bare class name collides with
# `layers/ffn/mlp.py::MLPBlock`, and whichever module imported last used to win
# the global registry key, so a saved TabM deserialized into the FFN block.
@keras.saving.register_keras_serializable(package="dl_techniques.tabm")
class MLPBlock(keras.layers.Layer):
    """
    MLP block with optional efficient ensemble support.

    This layer implements a single MLP block (linear + activation + optional
    dropout) that can operate in plain mode (single model) or ensemble mode
    (with ``k`` members using ``LinearEfficientEnsemble``).

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, (K,) D]              │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Linear (Dense or Ensemble)     │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Activation                     │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Optional Dropout               │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, (K,) units]         │
        └─────────────────────────────────┘

    :param units: Number of units in the hidden layer.
    :type units: int
    :param k: Number of ensemble members (None for plain MLP).
    :type k: int or None
    :param ensemble_type: How the ``k`` members are realized when ``k`` is not
        ``None``. ``'efficient'`` uses a :class:`LinearEfficientEnsemble` (one
        shared kernel plus rank-1 per-member scaling); ``'packed'`` uses an
        :class:`NLinear`, i.e. ``k`` fully independent kernels, which costs ``k``
        times the backbone parameters and is the honest deep-ensemble baseline.
    :type ensemble_type: str
    :param ensemble_scaling_in: Whether the efficient ensemble applies per-member
        input scaling. Ignored when ``ensemble_type='packed'`` or ``k is None``.
    :type ensemble_scaling_in: bool
    :param ensemble_scaling_out: Whether the efficient ensemble applies per-member
        output scaling. Ignored when ``ensemble_type='packed'`` or ``k is None``.
    :type ensemble_scaling_out: bool
    :param init_distribution: Initialization of the per-member scaling vectors
        (see :class:`LinearEfficientEnsemble`).
    :type init_distribution: str
    :param activation: Activation function.
    :type activation: str
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any

    :raises ValueError: If ``ensemble_type`` is not ``'efficient'`` or ``'packed'``.
    """

    def __init__(
            self,
            units: int,
            k: Optional[int] = None,
            ensemble_type: Literal['efficient', 'packed'] = 'efficient',
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if ensemble_type not in ('efficient', 'packed'):
            raise ValueError(
                f"ensemble_type must be 'efficient' or 'packed'; got {ensemble_type!r}"
            )
        self.units = units
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        self.activation = keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE sub-layers in __init__ (units/k are config-known) so weights
        # are reliably created/restored across serialization.
        if self.k is None:
            # Plain linear layer
            self.linear = keras.layers.Dense(
                self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )
        elif self.ensemble_type == 'packed':
            # k fully independent kernels; fan-in resolved at build time.
            self.linear = NLinear(
                n=self.k,
                input_dim=None,
                output_dim=self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )
        else:
            # Efficient ensemble layer
            self.linear = LinearEfficientEnsemble(
                self.units,
                self.k,
                use_bias=self.use_bias,
                ensemble_scaling_in=self.ensemble_scaling_in,
                ensemble_scaling_out=self.ensemble_scaling_out,
                init_distribution=self.init_distribution,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the MLP block layers."""
        # Explicitly build sub-layers for robust serialization
        self.linear.build(input_shape)
        linear_output_shape = self.linear.compute_output_shape(input_shape)

        if self.dropout is not None:
            self.dropout.build(linear_output_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass through MLP block.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Training mode flag.
            :type training: bool or None

            :return: Output tensor after linear transformation, activation, and dropout.
            :rtype: keras.KerasTensor
        """
        x = self.linear(inputs)
        x = self.activation(x)

        if self.dropout_rate > 0 and self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        if self.k is None:
            return (input_shape[0], self.units)
        else:
            return (input_shape[0], self.k, self.units)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "ensemble_type": self.ensemble_type,
            "ensemble_scaling_in": self.ensemble_scaling_in,
            "ensemble_scaling_out": self.ensemble_scaling_out,
            "init_distribution": self.init_distribution,
            "activation": keras.activations.serialize(self.activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TabMBackbone(keras.layers.Layer):
    """
    TabM backbone MLP with optional ensemble support.

    This layer stacks multiple ``MLPBlock`` layers to form a complete backbone.
    It can operate in plain mode (single model) or ensemble mode by setting the
    ``k`` parameter.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, (K,) D]              │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  MLPBlock(hidden_dims[0])       │
        ├─────────────────────────────────┤
        │  MLPBlock(hidden_dims[1])       │
        ├─────────────────────────────────┤
        │  ...                            │
        ├─────────────────────────────────┤
        │  MLPBlock(hidden_dims[-1])      │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, (K,) hidden[-1]]    │
        └─────────────────────────────────┘

    :param hidden_dims: List of hidden layer dimensions.
    :type hidden_dims: list[int]
    :param k: Number of ensemble members (None for plain MLP).
    :type k: int or None
    :param ensemble_type: ``'efficient'`` or ``'packed'`` (see :class:`MLPBlock`).
    :type ensemble_type: str
    :param ensemble_scaling_in: Per-member input scaling in the efficient ensemble.
    :type ensemble_scaling_in: bool
    :param ensemble_scaling_out: Per-member output scaling in the efficient ensemble.
    :type ensemble_scaling_out: bool
    :param init_distribution: Initialization of the per-member scaling vectors.
    :type init_distribution: str
    :param activation: Activation function.
    :type activation: str
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param use_bias: Whether to use bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional layer arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            hidden_dims: List[int],
            k: Optional[int] = None,
            ensemble_type: Literal['efficient', 'packed'] = 'efficient',
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            init_distribution: EnsembleInitDistribution = 'random-signs',
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.k = k
        self.ensemble_type = ensemble_type
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.init_distribution = init_distribution
        self.activation = deserialize_activation(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all MLP blocks in __init__ (hidden_dims are config-known) so
        # weights are reliably created/restored across serialization.
        self.blocks = [
            MLPBlock(
                units=units,
                k=self.k,
                ensemble_type=self.ensemble_type,
                ensemble_scaling_in=self.ensemble_scaling_in,
                ensemble_scaling_out=self.ensemble_scaling_out,
                init_distribution=self.init_distribution,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'block_{i}'
            )
            for i, units in enumerate(self.hidden_dims)
        ]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the backbone MLP blocks in computational order."""
        current_shape = input_shape
        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass through backbone MLP.

            :param inputs: Input tensor.
            :type inputs: keras.KerasTensor
            :param training: Training mode flag.
            :type training: bool or None

            :return: Output tensor after passing through all MLP blocks.
            :rtype: keras.KerasTensor
        """
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        shape = input_shape
        for block in self.blocks:
            shape = block.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "k": self.k,
            "ensemble_type": self.ensemble_type,
            "ensemble_scaling_in": self.ensemble_scaling_in,
            "ensemble_scaling_out": self.ensemble_scaling_out,
            "init_distribution": self.init_distribution,
            "activation": serialize_activation(self.activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

