"""
Configurable convolutional stem for vision models.

``ConvStem`` wires one or two ``Conv2D`` layers to an optional normalization
and an optional activation, then an optional pooling layer. Normalization and
activation are chosen by name through the ``norms`` and ``activations``
factories, so a stem is stated by constructor arguments alone.

Architecture (single conv, the default):

.. code-block:: text

    input [B, H, W, C_in]
          │
          ▼
    Conv2D(filters, kernel, strides)
          │
          ▼
    Normalization (optional)
          │
          ▼
    Activation (optional)
          │
          ▼
    Pooling (max / avg, optional)
          │
          ▼
    output [B, H', W', filters]

Architecture (two convs, ``num_convs=2``):

.. code-block:: text

    input
          │
          ▼
    Conv2D → [Normalization] → [Activation]
          │
          ▼
    Conv2D → [Normalization] → [Activation] → [Pooling] → output

Scope and conventions:

* The stem is 2D only. ``Conv3D`` / ``MaxPooling3D`` are not supported, and
  there is no ``rank`` option (D-004 of plan-2026-10-26T125959-7af851e5): a
  second 3D stem would have to appear before that abstraction pays for itself.
* The normalization and the activation must preserve the shape of their
  input, so only the convolutions and the pool move the output shape.
* An activation string is resolved as written. ``"leaky_relu"`` therefore
  builds the stock Keras form, not the exact form that
  :func:`dl_techniques.layers.activations.common.resolve_activation` returns
  (D-277). A caller that wants the exact form passes a callable, for example
  ``substitute_exact("leaky_relu")``; ``get_config`` writes it back as the
  string.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import keras

from ..norms import create_normalization_layer
from ..activations import resolve_activation_layer
from ..activations.common import activation_spec, serialize_activation
from ..activations.factory import ACTIVATION_REGISTRY
from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

def _canonical_activation(activation: Any) -> Any:
    """Canonicalize the ``activation_type`` argument for storage.

    Interface contract: returns ``None``, a string, a callable, or a ``dict``
    that serializes a ``keras.layers.Layer`` (what ``get_config`` writes back,
    so a rebuilt stem holds the same kind of value as a fresh one). A Layer
    INSTANCE is stored as that dict, not as the instance: keeping the instance
    as an attribute would track it as a sub-layer of the stem, an unbuilt one.

    ``activation_spec`` is not used for a Layer dict because it hands a Layer
    dict to ``keras.activations.deserialize``, which returns the bare class
    name (a string), silently turning ``LeakyReLU(0.1)`` into ``"LeakyReLU"``.

    :param activation: String, ``None``, callable, Layer instance, or a dict
        (serialized function or serialized Layer).
    :return: The canonical value described above.
    """
    if isinstance(activation, keras.layers.Layer):
        return keras.saving.serialize_keras_object(activation)
    if isinstance(activation, dict) and activation.get("class_name") != "function":
        return dict(activation)
    return activation_spec(activation)


@register_dl_technique("dl_techniques.layers.conv_blocks.stem")
class ConvStem(keras.layers.Layer):
    """Configurable convolutional stem: Conv → [norm] → [activation] → [pool].

    Supports single-conv (default) and two-conv stems. The two-conv form
    places normalization (+ optional activation) between the convolutions.
    Every sub-layer is created in ``__init__`` and built by an explicit
    ``build()``, in forward order.

    The two convolutions draw their weights independently: the second one is
    given ``clone_initializer`` copies of the kernel and bias initializers.
    A seeded initializer is the exception and replays by contract.

    :param filters: Number of convolutional filters (output channels).
    :type filters: int
    :param kernel_size: Size of the convolutional kernel.
    :type kernel_size: int or tuple[int, int]
    :param strides: Convolution strides. Applied to every conv layer.
    :type strides: int or tuple[int, int]
    :param padding: Padding mode of the convolutions (``'same'`` or
        ``'valid'``).
    :type padding: str
    :param use_bias: Whether the convolution(s) carry a bias term.
    :type use_bias: bool
    :param num_convs: Number of convolutional layers (1 or 2).
    :type num_convs: int
    :param conv2_filters: Filters of the second convolution. Defaults to
        ``filters``.
    :type conv2_filters: int or None
    :param normalization_type: Registered normalization type passed to
        :func:`dl_techniques.layers.norms.create_normalization_layer`, or
        ``None`` for no normalization (the factory has no such key, so ``None``
        is handled here). The factory's epsilon default is ``1e-6``, not
        Keras' ``1e-3``: pass ``epsilon`` in ``normalization_kwargs`` to
        restate a reference value.
    :type normalization_type: str or None
    :param normalization_kwargs: Extra keyword arguments forwarded to the
        normalization layer (for example ``{'momentum': 0.9, 'epsilon': 1e-3}``
        for BatchNormalization). Refused when ``normalization_type`` is
        ``None``.
    :type normalization_kwargs: dict or None
    :param activation_type: An ``ACTIVATION_REGISTRY`` key, a plain Keras
        activation name (``'relu6'``, ``'tanh'``), a callable, a
        ``keras.layers.Layer`` instance, or ``None``. ``'linear'`` and ``None``
        build no activation layer. A Layer instance is cloned at each
        activation site, so two sites never share one set of weights.
    :type activation_type: str or Callable or keras.layers.Layer or None
    :param activation_kwargs: Extra keyword arguments for the activation
        layer. Only a registry key accepts them: ``resolve_activation_layer``
        drops them silently otherwise, so any other activation raises.
    :type activation_kwargs: dict or None
    :param final_activation: Whether the second convolution of a two-conv stem
        is followed by an activation.
    :type final_activation: bool
    :param use_pooling: Whether to append a pooling layer after the last norm /
        activation.
    :type use_pooling: bool
    :param pool_type: Type of pooling (``'max'`` or ``'avg'``).
    :type pool_type: str
    :param pool_size: Size of the pooling window.
    :type pool_size: int or tuple[int, int]
    :param pool_strides: Pooling strides. ``None`` keeps the Keras default
        (stride equals ``pool_size``).
    :type pool_strides: int or tuple[int, int] or None
    :param pool_padding: Pooling padding (``'valid'`` or ``'same'``).
    :type pool_padding: str
    :param kernel_initializer: Initializer for the convolution kernel(s).
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for the convolution bias(es).
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for the convolution kernel(s).
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Regularizer for the convolution bias(es).
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param name: Layer name. Sub-layers are named ``{name}_conv1`` /
        ``{name}_conv2``, ``{name}_norm1`` / ``{name}_norm2``,
        ``{name}_act1`` / ``{name}_act2`` and ``{name}_pool``.
    :type name: str or None
    :param kwargs: Additional arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``filters`` or ``conv2_filters`` is not positive,
        ``num_convs`` is not 1 or 2, ``padding`` / ``pool_type`` /
        ``pool_padding`` is invalid, ``normalization_kwargs`` is given with no
        normalization, or ``activation_kwargs`` is given for an activation
        that is not a registry key.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: str = "same",
        use_bias: bool = True,
        num_convs: int = 1,
        conv2_filters: Optional[int] = None,
        normalization_type: Optional[str] = "batch_norm",
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        activation_type: Union[str, Callable, keras.layers.Layer, None] = "relu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        final_activation: bool = True,
        use_pooling: bool = False,
        pool_type: Literal["max", "avg"] = "max",
        pool_size: Union[int, Tuple[int, int]] = 2,
        pool_strides: Optional[Union[int, Tuple[int, int]]] = None,
        pool_padding: Literal["valid", "same"] = "valid",
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConvStem with the given configuration."""
        super().__init__(name=name, **kwargs)

        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if num_convs not in (1, 2):
            raise ValueError(f"num_convs must be 1 or 2, got {num_convs}")
        if padding not in ("same", "valid"):
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")
        if pool_type not in ("max", "avg"):
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type}")
        if pool_padding not in ("valid", "same"):
            raise ValueError(
                f"pool_padding must be 'valid' or 'same', got {pool_padding}"
            )
        if num_convs == 2 and conv2_filters is not None and conv2_filters <= 0:
            raise ValueError(f"conv2_filters must be positive, got {conv2_filters}")
        if normalization_type is None and normalization_kwargs:
            raise ValueError(
                "normalization_kwargs was given but normalization_type is None, "
                f"so there is no normalization layer to receive "
                f"{dict(normalization_kwargs)}"
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.num_convs = num_convs
        self.conv2_filters = conv2_filters if conv2_filters is not None else filters
        self.normalization_type = normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})
        self.activation_type = _canonical_activation(activation_type)
        self.activation_kwargs = dict(activation_kwargs or {})
        self.final_activation = final_activation
        self.use_pooling = use_pooling
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        if self.activation_kwargs and not (
            isinstance(self.activation_type, str)
            and self.activation_type in ACTIVATION_REGISTRY
        ):
            raise ValueError(
                "activation_kwargs is only accepted for an ACTIVATION_REGISTRY "
                f"key; activation_type={activation_type!r} would drop "
                f"{self.activation_kwargs} silently"
            )

        # -----------------------------------------------------------------
        # First convolution
        # -----------------------------------------------------------------
        conv1_name = f"{self.name}_conv1" if self.name else "conv1"
        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name=conv1_name,
        )
        self.norm1: Optional[keras.layers.Layer] = self._make_norm("norm1")
        self.activation1: Optional[keras.layers.Layer] = self._make_activation("act1")

        # -----------------------------------------------------------------
        # Second convolution (optional)
        # -----------------------------------------------------------------
        self.conv2: Optional[keras.layers.Conv2D] = None
        self.norm2: Optional[keras.layers.Layer] = None
        self.activation2: Optional[keras.layers.Layer] = None

        if num_convs == 2:
            conv2_name = f"{self.name}_conv2" if self.name else "conv2"
            # DECISION plan-2026-10-26T125959-7af851e5/D-005
            # conv2 gets CLONES of the kernel and bias initializers. Do NOT hand
            # it self.kernel_initializer: a seedless initializer instance
            # replays one sample at every weight it is given, even at a
            # different shape (the shorter draw is the longer one's prefix,
            # initializers/clone.py), so conv2's kernel would be conv1's draw.
            # The pre-migration CliffordNet stems shared the instance; the
            # replacement is deliberate, a departure from that draw at a fixed
            # seed, not a regression. Guard:
            # tests/test_layers/test_conv_blocks/test_stem.py
            # (TestTheTwoConvsDrawIndependently, with a planted control).
            self.conv2 = keras.layers.Conv2D(
                filters=self.conv2_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=conv2_name,
            )
            self.norm2 = self._make_norm("norm2")
            if final_activation:
                self.activation2 = self._make_activation("act2")

        # -----------------------------------------------------------------
        # Optional pooling
        # -----------------------------------------------------------------
        self.pool: Optional[keras.layers.Layer] = None
        if use_pooling:
            pool_name = f"{self.name}_pool" if self.name else "pool"
            pool_cls = (
                keras.layers.MaxPooling2D
                if pool_type == "max"
                else keras.layers.AveragePooling2D
            )
            self.pool = pool_cls(
                pool_size=pool_size,
                strides=pool_strides,
                padding=pool_padding,
                name=pool_name,
            )

    def _make_norm(self, suffix: str) -> Optional[keras.layers.Layer]:
        """Build one normalization layer, or ``None`` when there is none."""
        if self.normalization_type is None:
            return None
        return create_normalization_layer(
            self.normalization_type,
            name=f"{self.name}_{suffix}" if self.name else suffix,
            **self.normalization_kwargs,
        )

    def _make_activation(self, suffix: str) -> Optional[keras.layers.Layer]:
        """Build one activation layer, or ``None`` for ``'linear'`` / ``None``.

        A serialized Layer is deserialized afresh at every site, so two sites
        never share one instance (and one set of weights).
        """
        spec = self.activation_type
        if spec is None or (isinstance(spec, str) and spec == "linear"):
            return None
        name = f"{self.name}_{suffix}" if self.name else suffix
        if isinstance(spec, dict):
            layer_config = dict(spec)
            layer_config["config"] = {**spec["config"], "name": name}
            return keras.layers.deserialize(layer_config)
        return resolve_activation_layer(spec, name=name, **self.activation_kwargs)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sub-layer explicitly, in forward order.

        Normalization and activation preserve shape, so the running shape only
        changes at a convolution.
        """
        self.conv1.build(input_shape)
        shape = self.conv1.compute_output_shape(input_shape)
        if self.norm1 is not None:
            self.norm1.build(shape)
        if self.activation1 is not None:
            self.activation1.build(shape)

        if self.conv2 is not None:
            self.conv2.build(shape)
            shape = self.conv2.compute_output_shape(shape)
            if self.norm2 is not None:
                self.norm2.build(shape)
            if self.activation2 is not None:
                self.activation2.build(shape)

        if self.pool is not None:
            self.pool.build(shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the stem.

        :param inputs: Input tensor ``(B, H, W, C_in)``.
        :param training: Whether in training mode (affects normalization).
        :return: Output tensor.
        """
        x = self.conv1(inputs)
        if self.norm1 is not None:
            x = self.norm1(x, training=training)
        if self.activation1 is not None:
            x = self.activation1(x)

        if self.conv2 is not None:
            x = self.conv2(x)
            if self.norm2 is not None:
                x = self.norm2(x, training=training)
            if self.activation2 is not None:
                x = self.activation2(x)

        if self.pool is not None:
            x = self.pool(x)

        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape; spatial dims may be ``None``."""
        shape = self.conv1.compute_output_shape(input_shape)
        if self.conv2 is not None:
            shape = self.conv2.compute_output_shape(shape)
        if self.pool is not None:
            shape = self.pool.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization (every constructor argument)."""
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "num_convs": self.num_convs,
                "conv2_filters": self.conv2_filters,
                "normalization_type": self.normalization_type,
                "normalization_kwargs": self.normalization_kwargs,
                "activation_type": serialize_activation(self.activation_type),
                "activation_kwargs": self.activation_kwargs,
                "final_activation": self.final_activation,
                "use_pooling": self.use_pooling,
                "pool_type": self.pool_type,
                "pool_size": self.pool_size,
                "pool_strides": self.pool_strides,
                "pool_padding": self.pool_padding,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                )
                if self.kernel_regularizer
                else None,
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                )
                if self.bias_regularizer
                else None,
            }
        )
        return config


# ---------------------------------------------------------------------
