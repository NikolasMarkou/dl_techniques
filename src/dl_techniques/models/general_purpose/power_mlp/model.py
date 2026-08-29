"""
PowerMLP, a dual-branch feedforward network that replaces KAN's B-spline bases with
ReLU-k activations, with configurable power, dropout and batch normalization.

Kolmogorov-Arnold Networks move the nonlinearity from the nodes to the edges: each
connection carries its own learned univariate function, parameterized as a B-spline
over a grid. That is what gives KAN its expressiveness per parameter, and it is
also what makes it slow -- every forward pass must locate each input in the spline
grid and evaluate the basis polynomials there, an irregular, memory-bound
computation that does not reduce to a matrix multiply. PowerMLP starts from the
observation that the *shape* KAN buys with splines can be approximated by a fixed
nonlinearity of the right order combined with a linear map, recovering dense
GEMM-shaped compute.

The layer is two branches summed:

`y = ReLU_k(W_main x + b) + W_basis * swish(x)`

The main branch is a dense projection followed by `ReLU_k(z) = max(0, z)^k`. Raising
ReLU to an integer power `k` makes the branch piecewise-polynomial of degree `k`
rather than piecewise-linear, so a single layer can bend where a ReLU layer would
need several to approximate the same curvature -- the same degree-of-freedom KAN
gets from its splines, but as an elementwise power on an already-projected vector.
The branch ordering matters and is easy to state backwards: the dense map comes
*first* and the power is applied to its output, so `k` acts on learned features
rather than on raw inputs.

The basis branch applies `swish(x) = x * sigmoid(x)` to the *input* and then
projects it. Swish is smooth, non-monotonic and unbounded above, which makes it a
complementary shape to `ReLU_k`: it is nonzero for negative inputs, where `ReLU_k`
is identically zero and its gradient vanishes. Summing the two means a unit is
never fully dead -- whatever the main branch gates off, the basis branch still
passes a signal and a gradient. The basis projection is deliberately bias-free;
both branches carrying a bias would be redundant, and the main branch already has
one.

Note that the basis branch is a *stateless* activation followed by a linear map. It
adds no learned nonlinearity of its own, unlike a KAN edge function; the learning
in that branch lives entirely in `W_basis`. This is the trade PowerMLP makes, and
it is why it is faster rather than merely cheaper.

The model stacks these layers according to `hidden_units`, which is read as
`[input_dim, hidden_1, ..., hidden_n, output_dim]` -- the first entry describes the
expected input width rather than creating a layer, and the last entry sizes the
output. Optional batch normalization and dropout are applied after each hidden
layer, in that order. The output layer is a plain `Dense`, not a PowerMLP layer:
the final map needs an arbitrary activation (softmax, sigmoid, or none for
regression), and `ReLU_k` on the logits would clamp them non-negative and destroy
the parameterization every downstream loss expects.

`k` is a fixed integer hyperparameter, validated as such at construction -- it is
not learned. The preset variants raise it with model size (2 for micro, 3 through
base, 4 for the two largest), since higher-degree units are only worth their
conditioning cost when there is enough width to use them. Large `k` sharpens the
activation's gradient near the origin and grows its outputs fast, which is why
batch normalization becomes worth enabling as `k` rises.

References:
    - Liu et al., 2024. KAN: Kolmogorov-Arnold Networks.
      (https://arxiv.org/abs/2404.19756)
    - Ramachandran et al., 2017. Searching for Activation Functions.
      (https://arxiv.org/abs/1710.05941)
    - Ioffe & Szegedy, 2015. Batch Normalization: Accelerating Deep Network
      Training by Reducing Internal Covariate Shift.
      (https://arxiv.org/abs/1502.03167)
"""


import os
import keras
from typing import List, Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.power_mlp_layer import PowerMLPLayer
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.power_mlp.model")
class PowerMLP(keras.Model):
    """PowerMLP: a dual-branch MLP that trades KAN's splines for ReLU-k.

    A stack of :class:`PowerMLPLayer` instances, each summing a main branch
    (dense projection then ``ReLU_k(z) = max(0, z)^k``) with a basis branch
    (``swish`` on the INPUT, then a bias-free projection). The power makes a
    layer piecewise-polynomial of degree ``k`` rather than piecewise-linear,
    recovering the curvature-per-layer KAN gets from B-splines while keeping
    the compute dense and GEMM-shaped. The two branches are complementary: the
    basis branch is nonzero exactly where ``ReLU_k`` is identically zero, so a
    unit is never fully dead.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, input_dim]                │
        │  input_dim = hidden_units[0]         │
        │  (that entry DESCRIBES the input;    │
        │   it does not create a layer)        │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  PowerMLPLayer(hidden_units[1])      │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  [BatchNormalization]  (optional)    │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  [Dropout]             (optional)    │
        │  order is FIXED: norm, then dropout  │
        └───────────────┬──────────────────────┘
                        ▼
                       ...
                        ▼
        ┌──────────────────────────────────────┐
        │  PowerMLPLayer(hidden_units[-2])     │
        │  → [BatchNorm] → [Dropout]           │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Dense(hidden_units[-1],             │
        │        activation=output_activation) │
        │  a PLAIN Dense, NOT a PowerMLPLayer: │
        │  ReLU_k on the logits would clamp    │
        │  them non-negative and destroy the   │
        │  parameterization every loss expects │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, output_dim]              │
        └──────────────────────────────────────┘

    **Layer internals (the dual branch):**

    .. code-block:: text

                       x  [B, in]
                       │
            ┌──────────┴───────────┐
            ▼                      ▼
        ┌───────────────┐   ┌───────────────────┐
        │ MAIN branch   │   │ BASIS branch      │
        │               │   │                   │
        │ Dense(units,  │   │ swish(x)          │
        │   use_bias)   │   │  = x·sigmoid(x)   │
        │      ▼        │   │ applied to the    │
        │ ReLU_k:       │   │ INPUT, not to a   │
        │   max(0,z)^k  │   │ projection        │
        │               │   │      ▼            │
        │ the dense map │   │ Dense(units,      │
        │ comes FIRST;  │   │   NO BIAS)        │
        │ k acts on     │   │                   │
        │ LEARNED       │   │ stateless act +   │
        │ features      │   │ linear map: all   │
        └───────┬───────┘   │ learning is in    │
                │           │ W_basis           │
                │           └─────────┬─────────┘
                └────────► (+) ◄──────┘
                           ▼
                        y [B, units]

    **Why the two branches are complementary:**

    .. code-block:: text

              ReLU_k (k=3)              swish
        │        ╱                │      ╱
        │       ╱                 │    ╱
        │      ╱                  │  ╱
        ├─────●──────             ├─╲╱──────
        │  ZERO here              │  nonzero for x < 0
        │  gradient VANISHES      │  gradient survives

        summing them means whatever the main branch gates
        off, the basis branch still passes a signal AND a
        gradient -- a unit is never fully dead.

    **Variants:**

    .. code-block:: text

        variant   hidden_units (inner)        k
        micro     [32, 16]                    2
        tiny      [64, 32]                    3
        small     [128, 64, 32]               3
        base      [256, 128, 64]              3
        large     [512, 256, 128]             4
        xlarge    [1024, 512, 256, 128]       4

        from_variant wraps these as
            [input_dim] + inner + [num_classes]

        k rises with size: higher-degree units are only worth
        their conditioning cost when there is width to use
        them, and large k is why batch_normalization becomes
        worth enabling.

    :param hidden_units: Layer sizes read as
        ``[input_dim, hidden_1, ..., hidden_n, output_dim]``. The FIRST entry
        is the expected input width and creates no layer; the LAST sizes the
        output ``Dense``. Must have at least two elements and all values
        positive. For example ``[784, 128, 64, 10]`` gives two hidden
        PowerMLP layers.
    :type hidden_units: List[int]
    :param k: Power exponent of the ``ReLU-k`` activation in the main branch.
        Must be a positive INTEGER; it is a fixed hyperparameter, not learned.
        Recommended range 2-5; higher values may need batch normalization and
        gradient clipping. Defaults to 3.
    :type k: int
    :param kernel_initializer: Initializer for every kernel. Defaults to
        ``"he_normal"``, appropriate for ReLU-like activations.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for every bias. Defaults to
        ``"zeros"``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels. Defaults to
        None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for biases. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param use_bias: Whether the main branch's dense layers carry a bias. The
        basis branch NEVER uses one, by design. Defaults to True.
    :type use_bias: bool
    :param output_activation: Activation for the final ``Dense``. ``None``
        (default) gives a linear output for regression or for a loss consuming
        logits; ``'softmax'`` for multi-class, ``'sigmoid'`` for binary.
    :type output_activation: Optional[Union[str, callable]]
    :param dropout_rate: Dropout applied after each hidden layer, in
        ``[0, 1]``. 0.0 disables it. Defaults to 0.0.
    :type dropout_rate: float
    :param batch_normalization: Whether to apply batch normalization after each
        hidden layer, BEFORE dropout. Recommended when ``k > 3``. Defaults to
        False.
    :type batch_normalization: bool
    :param name: Model name. Defaults to ``"power_mlp"``.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If ``hidden_units`` has fewer than two elements or
        contains a non-positive value, if ``k`` is not positive, or if
        ``dropout_rate`` leaves ``[0, 1]``.
    :raises TypeError: If ``k`` is not an integer.

    Input shape:
        N-D tensor ``(batch_size, ..., input_dim)``, where ``input_dim``
        matches ``hidden_units[0]``. Typically 2D.

    Output shape:
        N-D tensor ``(batch_size, ..., output_dim)``, where ``output_dim`` is
        ``hidden_units[-1]``.

    :ivar hidden_layers: The :class:`PowerMLPLayer` stack.
    :vartype hidden_layers: List[PowerMLPLayer]
    :ivar dropout_layers: Per-hidden-layer ``Dropout``, or ``None`` entries.
    :vartype dropout_layers: List[Optional[keras.layers.Dropout]]
    :ivar batch_norm_layers: Per-hidden-layer ``BatchNormalization``, or
        ``None`` entries.
    :vartype batch_norm_layers: List[Optional[keras.layers.BatchNormalization]]
    :ivar output_layer: The final plain ``Dense``.
    :vartype output_layer: keras.layers.Dense

    Example:
        .. code-block:: python

            # Classification, input_dim = 784 (MNIST)
            model = PowerMLP(
                hidden_units=[784, 128, 64, 10],
                k=3,
                dropout_rate=0.2,
                output_activation="softmax"
            )
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Regression with batch normalization
            model = PowerMLP(
                hidden_units=[100, 256, 128, 1],
                k=4,
                batch_normalization=True,
                output_activation=None
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # From a variant, for CIFAR-10 (flattened, input_dim = 3072)
            model = PowerMLP.from_variant("base", num_classes=10, input_dim=3072)

    Note:
        All sub-layers are created in ``__init__``. ``build()`` materializes
        them by tracing ``call()`` on symbolic inputs, so an explicit
        ``model.build(shape)`` -- and the ``build_from_config`` step of
        ``.keras`` deserialization -- leave the model actually built rather
        than merely marked built.
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {"hidden_units": [32, 16], "k": 2},
        "tiny": {"hidden_units": [64, 32], "k": 3},
        "small": {"hidden_units": [128, 64, 32], "k": 3},
        "base": {"hidden_units": [256, 128, 64], "k": 3},
        "large": {"hidden_units": [512, 256, 128], "k": 4},
        "xlarge": {"hidden_units": [1024, 512, 256, 128], "k": 4},
    }

    def __init__(
        self,
        hidden_units: List[int],
        k: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_bias: bool = True,
        output_activation: Optional[Union[str, callable]] = None,
        dropout_rate: float = 0.0,
        batch_normalization: bool = False,
        name: Optional[str] = "power_mlp",
        **kwargs: Any
    ) -> None:
        """Initialize the model and create every sub-layer.

        Sub-layers are instantiated here but not yet built; materialization
        happens in :meth:`build`.

        :param hidden_units: Layer sizes including input and output.
        :type hidden_units: List[int]
        :param k: Power for the ``ReLU-k`` activation.
        :type k: int
        :param kernel_initializer: Initializer for kernels.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Initializer for biases.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for kernels.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_regularizer: Regularizer for biases.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param use_bias: Whether the main branch uses a bias.
        :type use_bias: bool
        :param output_activation: Activation for the output layer.
        :type output_activation: Optional[Union[str, callable]]
        :param dropout_rate: Dropout rate after each hidden layer.
        :type dropout_rate: float
        :param batch_normalization: Whether to use batch normalization.
        :type batch_normalization: bool
        :param name: Model name.
        :type name: Optional[str]
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any parameter is invalid.
        :raises TypeError: If ``k`` is not an integer.
        """
        super().__init__(name=name, **kwargs)

        # Validate parameters
        self._validate_parameters(hidden_units, k, dropout_rate)

        # Store configuration parameters for serialization
        self.hidden_units = list(hidden_units)  # Make a copy
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias
        self.output_activation = keras.activations.get(output_activation)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        # CREATE all sub-layers in __init__ (Golden Rule)
        self.hidden_layers: List[PowerMLPLayer] = []
        self.dropout_layers: List[Optional[keras.layers.Dropout]] = []
        self.batch_norm_layers: List[Optional[keras.layers.BatchNormalization]] = []

        # Create hidden layers
        self._create_hidden_layers()

        # Create output layer
        self._create_output_layer()

        logger.info(
            f"Initialized PowerMLP model '{self.name}' with architecture "
            f"{self.hidden_units}, k={self.k}, dropout={self.dropout_rate}, "
            f"batch_norm={self.batch_normalization}"
        )

    def _validate_parameters(
        self,
        hidden_units: List[int],
        k: int,
        dropout_rate: float
    ) -> None:
        """Validate the initialization parameters with descriptive errors.

        :param hidden_units: List of layer sizes.
        :type hidden_units: List[int]
        :param k: Power for the ``ReLU-k`` activation.
        :type k: int
        :param dropout_rate: Dropout rate.
        :type dropout_rate: float
        :raises ValueError: If ``hidden_units`` is too short or non-positive,
            if ``k`` is not positive, or if ``dropout_rate`` leaves ``[0, 1]``.
        :raises TypeError: If ``k`` is not an integer.
        """
        if not hidden_units or len(hidden_units) < 2:
            raise ValueError(
                "hidden_units must contain at least an input and output size, "
                f"got {len(hidden_units)} elements"
            )
        if any(units <= 0 for units in hidden_units):
            raise ValueError(
                f"All hidden_units must be positive, got {hidden_units}"
            )
        if not isinstance(k, int):
            raise TypeError(
                f"k must be an integer, got type {type(k).__name__}"
            )
        if k <= 0:
            raise ValueError(
                f"k must be a positive integer, got {k}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

    def _create_hidden_layers(self) -> None:
        """Create the hidden PowerMLP layers with optional dropout and norm.

        The hidden layers correspond to ``hidden_units[1:-1]``; the first entry
        describes the input and the last sizes the output layer. Optional
        regularizers are appended as ``None`` placeholders when disabled, so the
        three lists stay index-aligned.
        """
        # Hidden layers: hidden_units[1:-1]
        for i, units in enumerate(self.hidden_units[1:-1]):
            # PowerMLP layer
            power_mlp_layer = PowerMLPLayer(
                units=units,
                k=self.k,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
                name=f"powermlp_hidden_{i + 1}"
            )
            self.hidden_layers.append(power_mlp_layer)

            # Batch normalization layer (optional)
            if self.batch_normalization:
                bn_layer = keras.layers.BatchNormalization(
                    name=f"batch_norm_{i + 1}"
                )
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)

            # Dropout layer (optional)
            if self.dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=self.dropout_rate,
                    name=f"dropout_{i + 1}"
                )
                self.dropout_layers.append(dropout_layer)
            else:
                self.dropout_layers.append(None)

    def _create_output_layer(self) -> None:
        """Create the output layer, a plain ``Dense`` sized by ``hidden_units[-1]``.

        Deliberately NOT a :class:`PowerMLPLayer`: the final map needs an
        arbitrary activation, and ``ReLU_k`` on the logits would clamp them
        non-negative.
        """
        output_units = self.hidden_units[-1]

        # Use regular Dense layer for output to allow flexible activation
        self.output_layer = keras.layers.Dense(
            units=output_units,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            name="output_layer"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method PowerMLP inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        :type input_shape: Any
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the hidden stack and the output layer.

        :param inputs: Input tensor of shape ``(batch_size, ..., input_dim)``,
            where ``input_dim`` matches ``hidden_units[0]``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode, which decides
            whether dropout and batch-norm statistics are active. ``None`` lets
            Keras infer it from the execution context.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, ..., output_dim)``.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Pass through hidden layers with optional regularization
        for i, layer in enumerate(self.hidden_layers):
            # PowerMLP layer
            x = layer(x, training=training)

            # Optional batch normalization
            if self.batch_normalization and self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i](x, training=training)

            # Optional dropout
            if self.dropout_rate > 0.0 and self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x, training=training)

        # Output layer
        outputs = self.output_layer(x, training=training)

        return outputs

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, replacing the last axis with the output width.

        :param input_shape: Shape tuple of the input, typically
            ``(batch_size, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape with the final axis set to
            ``hidden_units[-1]``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape[:-1] + (self.hidden_units[-1],)

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        Initializers, regularizers and the output activation are stored in
        their serialized forms.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "k": self.k,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "use_bias": self.use_bias,
            "output_activation": keras.activations.serialize(self.output_activation),
            "dropout_rate": self.dropout_rate,
            "batch_normalization": self.batch_normalization,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PowerMLP":
        """Create a model instance from its configuration.

        Deserializes the initializers, regularizers and activation from their
        stored forms before constructing.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: PowerMLP instance reconstructed from the configuration.
        :rtype: PowerMLP
        """
        # Deserialize complex objects
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
        if "output_activation" in config and isinstance(config["output_activation"], dict):
            config["output_activation"] = keras.activations.deserialize(
                config["output_activation"]
            )

        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int,
        input_dim: int,
        **kwargs: Any
    ) -> "PowerMLP":
        """Create a PowerMLP model from a predefined variant.

        The variant's inner width list is wrapped as
        ``[input_dim] + inner + [num_classes]``.

        :param variant: One of ``"micro"``, ``"tiny"``, ``"small"``,
            ``"base"``, ``"large"``, ``"xlarge"``.
        :type variant: str
        :param num_classes: Output dimension: the number of classes for
            classification, or typically 1 for regression.
        :type num_classes: int
        :param input_dim: Input feature dimension; must match the data (e.g.
            784 for flattened MNIST, 3072 for flattened CIFAR-10).
        :type input_dim: int
        :param kwargs: Additional arguments overriding the variant's defaults,
            such as ``dropout_rate``, ``batch_normalization``,
            ``output_activation`` or ``k``. **``hidden_units`` is REFUSED by
            name** with a ``ValueError``: this method builds the variant's own
            architecture, and a caller-supplied list was silently discarded
            before 2026-08-15. Use ``PowerMLP(hidden_units=..., k=...)``
            directly for that.
        :type kwargs: Any
        :return: PowerMLP instance configured for the variant.
        :rtype: PowerMLP
        :raises ValueError: If ``variant`` is not recognized, or if
            ``hidden_units`` is passed in ``kwargs``.

        Example:
            >>> # CIFAR-10 model (flattened 32x32x3 = 3072)
            >>> model = PowerMLP.from_variant("base", num_classes=10, input_dim=3072)
            >>>
            >>> # MNIST model (flattened 28x28 = 784)
            >>> model = PowerMLP.from_variant("small", num_classes=10, input_dim=784)
            >>>
            >>> # Custom regression model with k override
            >>> model = PowerMLP.from_variant(
            ...     "large",
            ...     num_classes=1,
            ...     input_dim=100,
            ...     k=5,
            ...     batch_normalization=True
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # Start with variant defaults
        config = cls.MODEL_VARIANTS[variant].copy()

        # Allow kwargs to override variant defaults
        config.update(kwargs)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-054
        # A caller-supplied `hidden_units` used to be accepted by
        # `config.update(kwargs)` and then UNCONDITIONALLY overwritten by the
        # variant's own list on the next line -- silently discarded. Refuse it
        # by name instead. Do NOT make it win: `from_variant` exists to build
        # the variant's architecture, and the value it would have to honour is
        # the INNER list (without the `[input_dim] + ... + [num_classes]`
        # wrapping this line adds), which no caller can be expected to guess.
        # `PowerMLP(hidden_units=...)` is the direct constructor for that.
        # See decisions.md D-054.
        if "hidden_units" in kwargs:
            raise ValueError(
                f"from_variant('{variant}') builds the variant's own "
                f"hidden_units and cannot honour a caller-supplied one "
                f"(got {kwargs['hidden_units']!r}); it was silently discarded "
                f"before 2026-08-15. Use PowerMLP(hidden_units=..., k=...) "
                f"directly for a custom architecture."
            )

        # Construct the full hidden_units list
        base_hidden_units = cls.MODEL_VARIANTS[variant]["hidden_units"]
        config["hidden_units"] = [input_dim] + base_hidden_units + [num_classes]

        logger.info(f"Creating PowerMLP-{variant.upper()} model")
        logger.info(f"Architecture: {config['hidden_units']}")

        return cls(**config)

    def save_model(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = "keras"
    ) -> None:
        """Save the model, creating the parent directory if needed.

        An UNBUILT model is REFUSED rather than written: ``self.save()`` would
        produce a syntactically valid ``.keras`` archive holding ZERO weights,
        and ``load_model()`` would hand it back as a zero-weight model.

        :param filepath: Destination path; should end with ``.keras``.
        :type filepath: str
        :param overwrite: Whether to overwrite an existing file. Defaults to
            True.
        :type overwrite: bool
        :param save_format: Accepted for backwards compatibility and IGNORED.
            Keras 3 selects the format from the file extension;
            ``saving_api.save_model`` pops this kwarg and never forwards it.
        :type save_format: str
        :raises ValueError: If the model has not been built. Unlike
            ``CapsNet``, ``PowerMLP.__init__`` takes no input shape, so this
            method has nothing to build from.
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # DECISION plan-2026-08-22T035419-a11304c8/D-053
        # Do NOT restore a silent `self.save()` on an unbuilt model. MEASURED
        # 2026-08-22: `PowerMLP(hidden_units=[8,4]).save(path)` writes a
        # 9,997-byte archive whose reload has `built=False` and
        # `len(trainable_weights) == 0`, against 2 for the same model built.
        # The only signal was a UserWarning, and the two tests on this path
        # asserted only that the file existed and that loading returned an
        # object. See decisions.md D-053.
        if not self.built:
            raise ValueError(
                "Cannot save an unbuilt PowerMLP: the archive would contain "
                "zero weights. Call the model on a batch, or call "
                "`model.build((None, input_dim))`, before saving."
            )

        # `save_format` is deliberately NOT forwarded: Keras 3 picks the format
        # from the extension and `saving_api.save_model` discards the kwarg.
        self.save(filepath, overwrite=overwrite)
        logger.info(f"PowerMLP model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "PowerMLP":
        """Load a saved PowerMLP model.

        :param filepath: Path to the saved model file.
        :type filepath: str
        :return: Loaded model, ready for inference or continued training.
        :rtype: PowerMLP
        """
        # Note: both classes are registered via @register_dl_technique --
        # `dl_techniques.models.power_mlp.model>PowerMLP` and
        # `dl_techniques.layers.ffn.power_mlp_layer>PowerMLPLayer` (resolved
        # 2026-08-29 with keras.saving.get_registered_name) -- so custom_objects
        # may not be strictly necessary, but we include them for robustness.
        custom_objects = {
            "PowerMLP": cls,
            "PowerMLPLayer": PowerMLPLayer,
        }

        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"PowerMLP model loaded from {filepath}")
        return model

    def summary(self, **kwargs: Any) -> None:
        """Print the model summary plus PowerMLP-specific configuration.

        Builds the model first if it is not already built, inferring the input
        width from ``hidden_units[0]``.

        :param kwargs: Additional arguments passed to ``keras.Model.summary``,
            such as ``line_length``, ``positions`` or ``print_fn``.
        :type kwargs: Any
        """
        # Build the model first if it hasn't been built
        if not self.built:
            # We need an input shape to build. We can infer from hidden_units.
            input_dim = self.hidden_units[0]
            self.build((None, input_dim))

        # Print standard Keras summary
        super().summary(**kwargs)

        # Print PowerMLP-specific configuration
        logger.info("\nPowerMLP Configuration:")
        logger.info(f"  - Architecture (input→hidden→output): {self.hidden_units}")
        logger.info(f"  - ReLU-k power: {self.k}")
        logger.info(f"  - Total parameters: {self.count_params():,}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Batch normalization: {self.batch_normalization}")
        logger.info(f"  - Output activation: {keras.activations.serialize(self.output_layer.activation)}")
        logger.info(f"  - Use bias: {self.use_bias}")

    def __repr__(self) -> str:
        """Return a concise string representation for debugging and logging.

        :return: Representation including the key parameters.
        :rtype: str
        """
        return (
            f"PowerMLP(hidden_units={self.hidden_units}, k={self.k}, "
            f"dropout_rate={self.dropout_rate}, name='{self.name}')"
        )


# ---------------------------------------------------------------------
# Helper functions to create and compile PowerMLP models
# ---------------------------------------------------------------------

def create_power_mlp(
    hidden_units: List[int],
    k: int = 3,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    loss: Optional[Union[str, keras.losses.Loss]] = None,
    metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
    **kwargs: Any
) -> PowerMLP:
    """Create and compile a PowerMLP model in one call.

    **Default-loss derivation:**

    .. code-block:: text

        loss=None  →  the loss FOLLOWS output_activation:

            output_activation=None (default)  →  CategoricalCrossentropy(
                                                     from_logits=True)
            output_activation='softmax'       →  CategoricalCrossentropy(
            output_activation='sigmoid'              from_logits=False)

        a fixed `loss="categorical_crossentropy"` string would compile
        with from_logits=False against the LINEAR default, feeding
        unnormalized real values to a cross-entropy that renormalizes
        by output/sum(output) and clips -- finite, meaningless, silent.
        See the D-053 anchor.

    :param hidden_units: Layer sizes as
        ``[input_dim, hidden_1, ..., hidden_n, output_dim]``.
    :type hidden_units: List[int]
    :param k: Power for the ``ReLU-k`` activation. Defaults to 3.
    :type k: int
    :param optimizer: Optimizer name or instance. When a string,
        ``learning_rate`` is applied to it. Defaults to ``"adam"``.
    :type optimizer: Union[str, keras.optimizers.Optimizer]
    :param learning_rate: Learning rate, applied when ``optimizer`` is a
        string. Defaults to 0.001.
    :type learning_rate: float
    :param loss: Loss name or instance. ``None`` (the default) DERIVES a
        categorical cross-entropy whose ``from_logits`` matches the model's
        actual ``output_activation``: ``True`` for the default linear output,
        ``False`` for softmax or sigmoid. Pass a value to override.
    :type loss: Optional[Union[str, keras.losses.Loss]]
    :param metrics: Metric names or instances. ``None`` resolves to
        ``['accuracy']``.
    :type metrics: Optional[List[Union[str, keras.metrics.Metric]]]
    :param kwargs: Additional arguments for the :class:`PowerMLP` constructor,
        such as ``dropout_rate``, ``batch_normalization`` or
        ``output_activation``.
    :type kwargs: Any
    :return: Compiled PowerMLP model, ready for ``model.fit()``.
    :rtype: PowerMLP

    Example:
        >>> # The documented default: linear output, cross-entropy on LOGITS.
        >>> model = create_power_mlp(hidden_units=[784, 128, 64, 10], k=3)
        >>> model.loss.from_logits
        True
        >>>
        >>> # Softmax output: the derived loss follows it.
        >>> model = create_power_mlp(
        ...     hidden_units=[784, 128, 64, 10],
        ...     output_activation='softmax',
        ...     dropout_rate=0.2,
        ... )
        >>> model.loss.from_logits
        False
        >>> model.fit(x_train, y_train, epochs=10, validation_split=0.2)
    """
    # Create model
    model = PowerMLP(hidden_units=hidden_units, k=k, **kwargs)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-053
    # DERIVE the default loss from the model's ACTUAL output activation. DO NOT
    # restore the `loss="categorical_crossentropy"` default: `PowerMLP` defaults
    # to `output_activation=None` (linear), and a string loss compiles with
    # `from_logits=False`, so the documented example fed UNNORMALIZED
    # real-valued outputs to a cross-entropy that renormalizes by
    # `output/sum(output)` and clips. With mixed-sign activations that
    # denominator can approach zero and negatives clip to `epsilon`: finite,
    # meaningless, no error, and it trains happily on the wrong objective. Do
    # NOT "fix" it by defaulting `output_activation` to softmax instead --
    # that would silently change the OUTPUT of every existing caller that
    # relies on the linear default, whereas this changes only the loss of a
    # caller who passed neither. Both sibling factories
    # (`create_power_mlp_regressor`, `create_power_mlp_binary_classifier`) pass
    # `loss=` explicitly and are unaffected.
    # See decisions.md D-053.
    if loss is None:
        emits_probabilities = (
            model.output_activation is keras.activations.softmax
            or model.output_activation is keras.activations.sigmoid
        )
        loss = keras.losses.CategoricalCrossentropy(
            from_logits=not emits_probabilities
        )
        logger.info(
            f"create_power_mlp: no loss given; derived "
            f"CategoricalCrossentropy(from_logits={not emits_probabilities}) "
            f"from output_activation="
            f"{keras.activations.serialize(model.output_activation)!r}."
        )

    # Handle optimizer
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
    if hasattr(optimizer, 'learning_rate'):
        optimizer.learning_rate = learning_rate

    # Default metrics
    if metrics is None:
        metrics = ['accuracy']

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(
        f"Created and compiled PowerMLP model with architecture "
        f"{hidden_units[1:-1]} (hidden layers)"
    )
    return model


def create_power_mlp_regressor(
    hidden_units: List[int],
    k: int = 3,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs: Any
) -> PowerMLP:
    """Create and compile a PowerMLP for regression.

    Pins ``loss='mse'``, ``metrics=['mae', 'mse']`` and a linear output, so the
    default-loss derivation in :func:`create_power_mlp` does not apply.

    :param hidden_units: Layer sizes; the last element is typically 1 for
        single-target regression, or N for multi-target.
    :type hidden_units: List[int]
    :param k: Power for the ``ReLU-k`` activation. Defaults to 3.
    :type k: int
    :param optimizer: Optimizer name or instance. Defaults to ``"adam"``.
    :type optimizer: Union[str, keras.optimizers.Optimizer]
    :param learning_rate: Learning rate. Defaults to 0.001.
    :type learning_rate: float
    :param kwargs: Additional arguments for the :class:`PowerMLP` constructor.
    :type kwargs: Any
    :return: Compiled model configured for regression.
    :rtype: PowerMLP

    Example:
        >>> model = create_power_mlp_regressor(
        ...     hidden_units=[100, 256, 128, 1],
        ...     k=4,
        ...     learning_rate=0.001,
        ...     batch_normalization=True
        ... )
        >>> model.fit(x_train, y_train, epochs=50)
    """
    return create_power_mlp(
        hidden_units=hidden_units,
        k=k,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss="mse",
        metrics=["mae", "mse"],
        output_activation=None,  # Linear output for regression
        **kwargs
    )


def create_power_mlp_binary_classifier(
    hidden_units: List[int],
    k: int = 3,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs: Any
) -> PowerMLP:
    """Create and compile a PowerMLP for binary classification.

    Pins ``loss='binary_crossentropy'``, a sigmoid output and
    accuracy/precision/recall metrics, so the default-loss derivation in
    :func:`create_power_mlp` does not apply.

    :param hidden_units: Layer sizes; the last element should be 1. A different
        value logs a warning rather than raising.
    :type hidden_units: List[int]
    :param k: Power for the ``ReLU-k`` activation. Defaults to 3.
    :type k: int
    :param optimizer: Optimizer name or instance. Defaults to ``"adam"``.
    :type optimizer: Union[str, keras.optimizers.Optimizer]
    :param learning_rate: Learning rate. Defaults to 0.001.
    :type learning_rate: float
    :param kwargs: Additional arguments for the :class:`PowerMLP` constructor.
    :type kwargs: Any
    :return: Compiled model configured for binary classification.
    :rtype: PowerMLP

    Example:
        >>> model = create_power_mlp_binary_classifier(
        ...     hidden_units=[200, 512, 256, 128, 1],
        ...     k=3,
        ...     dropout_rate=0.3,
        ...     learning_rate=0.0005
        ... )
        >>> model.fit(x_train, y_train, epochs=20)
    """
    # Ensure output is configured for binary classification
    if hidden_units[-1] != 1:
        logger.warning(
            f"For binary classification, output should be 1 unit, got {hidden_units[-1]}. "
            "Consider adjusting hidden_units to end with 1."
        )

    return create_power_mlp(
        hidden_units=hidden_units,
        k=k,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
        output_activation="sigmoid",
        **kwargs
    )

# ---------------------------------------------------------------------
