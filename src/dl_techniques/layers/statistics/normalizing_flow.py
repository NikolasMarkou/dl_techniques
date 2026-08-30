"""
Conditional normalizing flows built from affine coupling layers.

A normalizing flow turns a simple base distribution into a complicated one by
pushing samples through a chain of invertible maps. This module implements the
Real NVP affine coupling variant, conditioned on an external context vector, so
it models ``p(y | context)`` rather than a fixed ``p(y)``.

Two classes ship here:

- ``AffineCouplingLayer`` is one invertible step. It splits its input in two,
  leaves one half alone, then rescales and shifts the other half. The scale and
  shift come from a small network fed the untouched half plus the context.
- ``NormalizingFlowLayer`` stacks ``num_flow_steps`` of those, and adds the
  exact log-likelihood loss and a sampler.

Why coupling layers. The map has to be invertible, and its Jacobian determinant
has to be cheap. Leaving half the input untouched makes the Jacobian
triangular, so its determinant is just the product of the scale factors. That
costs ``O(input_dim)`` work instead of ``O(input_dim^3)``.

Unlike a mixture density network, a flow makes no parametric assumption about
the target shape. It can fit multimodal, skewed and bounded distributions, and
it still reports an exact likelihood rather than a bound.

**Mathematical Foundation:**

The flow composes ``K`` invertible maps ``f_1 ... f_K``::

    y = f_K(f_{K-1}(... f_1(z) ...)),      z ~ N(0, I)

Change of variables gives the exact log-likelihood. The Jacobian term is
SUBTRACTED when it is the forward map's::

    log p(y) = log N(z; 0, I) - sum_i log|det(df_i / dz_{i-1})|

``inverse`` reports the log-determinant of the ``y -> z`` map it performs,
which is the negative of that, so ``loss_func`` adds what ``inverse`` returns.

For one affine coupling step, with the input split into ``z_a`` and ``z_b``::

    y_a = z_a
    y_b = z_b * s(z_a, context) + t(z_a, context)
    log|det(dy/dz)| = sum(log(s))        forward
    log|det(dz/dy)| = -sum(log(s))       inverse, and what inverse() returns

Sampling runs the chain forward, ``z`` to ``y``. Density estimation runs it
backward, ``y`` to ``z``. Both directions use the same weights.
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, Optional, Tuple, Any, List, Union
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

EPSILON_CONSTANT = 1e-6

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.normalizing_flow")
class AffineCouplingLayer(keras.layers.Layer):
    """
    One invertible affine coupling step, conditioned on an external context.

    The input is cut at ``split_dim = input_dim // 2``. The first half, ``z_a``,
    passes through untouched. The second half, ``z_b``, is rescaled and shifted
    by ``y_b = z_b * s + t``, where ``s`` and ``t`` come from
    ``transformation_net`` applied to ``z_a`` and the context. Because ``z_a``
    is unchanged, the same ``s`` and ``t`` can be recovered in either direction,
    which is what makes the step invertible.

    The Jacobian is triangular, so the forward map has
    ``log|det(dy/dz)| = sum(log(s))`` and the inverse has ``-sum(log(s))``.
    ``inverse`` returns the latter, the log-determinant of the map it performs.
    That is ``O(input_dim)`` work.

    Set ``reverse=True`` and the layer rotates the input left by ``split_dim``
    before splitting, so the OTHER half gets transformed, then rotates right by
    ``split_dim`` on the way out to restore the original ordering. A stack that
    alternates the flag transforms every dimension eventually.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────┐
        │ z (batch, input_dim)                             │
        │ context (batch, context_dim)                     │
        └─────────────────────────┬────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │ _apply_split_and_reverse                         │
        │ identity, or rotate left by split_dim            │
        └─────────────────────────┬────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │ affine coupling, see Coupling Internals below    │
        │ transformation_net owns every weight here        │
        └─────────────────────────┬────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │ _undo_split_and_reverse                          │
        │ identity, or rotate right by split_dim           │
        └─────────────────────────┬────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │ y (batch, input_dim)                             │
        └──────────────────────────────────────────────────┘

    **Coupling Internals:**

    .. code-block:: text

                          z (batch, input_dim)
                                     │
                 ┌───────────────────┴──────────────────┐
                 ▼                                      ▼
        ┌────────────────────────────┐  ┌────────────────────────────┐
        │ z_a = z[..., :split_dim]   │  │ z_b = z[..., split_dim:]   │
        │ static, never transformed  │  │ this is what gets changed  │
        │ (batch, split_dim)         │  │ (batch, dim_to_transform)  │
        └─────────────┬──────────────┘  └─────────────┬──────────────┘
                      ▼                               │
        ┌────────────────────────────┐                │
        │ concatenate([z_a, context])│                │
        │ (batch, split_dim + ctx)   │                │
        └─────────────┬──────────────┘                │
                      ▼                               │
        ┌────────────────────────────┐                │
        │ transformation_net         │                │
        │ Dense -> Dense -> Dense    │                │
        │ (batch, 2*dim_to_transform)│                │
        └─────────────┬──────────────┘                │
                      ▼                               │
        ┌────────────────────────────┐                │
        │ log_s, t = split(params)   │                │
        │ s = exp(tanh(log_s))       │                │
        │  or exp(clip(log_s,-10,10))│                │
        └─────────────┬──────────────┘                │
                      │ s, t                          │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │ y_b = z_b * s + t             │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │ y = concatenate([z_a, y_b])   │
                      │ (batch, input_dim)            │
                      └───────────────────────────────┘

    The final Dense emits ``dim_to_transform * 2`` values: the log-scale for
    every transformed dimension, then the shift for every transformed
    dimension. ``dim_to_transform`` is ``input_dim - split_dim``.

    **Forward vs Inverse:**

    .. code-block:: text

             forward()  z ─► y               inverse()  y ─► z
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ _apply_split_and_reverse │    │ _apply_split_and_reverse │
        └────────────┬─────────────┘    └────────────┬─────────────┘
                     ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ z_a, z_b = split(z)      │    │ y_a, y_b = split(y)      │
        └────────────┬─────────────┘    └────────────┬─────────────┘
                     ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ s, t = net(z_a, ctx)     │◄══►│ s, t = net(y_a, ctx)     │
        │   SHARED WEIGHTS         │    │   SHARED WEIGHTS         │
        └────────────┬─────────────┘    └────────────┬─────────────┘
                     ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ y_b = z_b * s + t        │    │ z_b = (y_b - t) / s      │
        └────────────┬─────────────┘    └────────────┬─────────────┘
                     ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ y = concat([z_a, y_b])   │    │ z = concat([y_a, z_b])   │
        └────────────┬─────────────┘    └────────────┬─────────────┘
                     ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ _undo_split_and_reverse  │    │ _undo_split_and_reverse  │
        │ returns y                │    │ returns (z, ldj)         │
        │                          │    │ ldj = sum(log(s), -1)    │
        └──────────────────────────┘    └──────────────────────────┘

    ``y_a`` and ``z_a`` are the same tensor, because the static half is never
    touched. That is why one ``transformation_net`` call serves both
    directions, and why swapping the two formulas is the classic bug here:
    dividing in ``forward`` and multiplying in ``inverse`` still round-trips,
    but the log-determinant then has the wrong sign and the likelihood is wrong.

    :param input_dim: Dimensionality of the input data. Must be >= 2.
    :type input_dim: int
    :param context_dim: Dimensionality of the conditioning context. Must be >= 1.
    :type context_dim: int
    :param hidden_units: Width of the two hidden Dense layers in
        ``transformation_net``. Must be >= 1. Defaults to 64.
    :type hidden_units: int
    :param reverse: Whether to rotate the input by ``split_dim`` before
        splitting, so the other half is the transformed one. Defaults to False.
    :type reverse: bool
    :param activation: Activation for the two hidden Dense layers. Defaults to
        "relu". Stored through ``keras.activations.get`` so a callable
        round-trips.
    :type activation: str | callable
    :param use_tanh_stabilization: If True, ``s = exp(tanh(log_s))``, which caps
        the scale in ``[exp(-1), exp(1)]``. If False, ``s = exp(clip(log_s,
        -10, 10))``, which allows a much larger range. Defaults to True.
    :type use_tanh_stabilization: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar split_dim: Index the input is cut at, ``input_dim // 2``.
    :vartype split_dim: int
    :ivar transformation_net: The conditioner. Predicts the log-scale and the
        shift, and holds all of this layer's weights.
    :vartype transformation_net: keras.Sequential

    :raises ValueError: If ``input_dim`` is less than 2, if ``context_dim`` is
        less than 1, or if ``hidden_units`` is less than 1.

    Input shape:
        Two tensors, ``[data, context]``, of shape ``(batch_size, input_dim)``
        and ``(batch_size, context_dim)``.

    Output shape:
        ``forward`` returns ``(batch_size, input_dim)``. ``inverse`` returns
        that plus a ``(batch_size,)`` log-determinant.

    Example:
        >>> layer = AffineCouplingLayer(input_dim=4, context_dim=3)
        >>> layer.build([(None, 4), (None, 3)])
        >>> y = layer.forward(z, context)
        >>> z_back, log_det = layer.inverse(y, context)

    Note:
        The layer round-trips at every ``input_dim >= 2`` and both ``reverse``
        settings. The exit rotation is the inverse of the entry rotation: the
        entry rotates the last axis left by ``split_dim``, the exit rotates it
        right by the same amount. Measured at ``input_dim`` in 2 through 7,
        ``context_dim=3``, batch 8: ``inverse(forward(z))`` differs from ``z``
        by ~1e-07 float32 round-off in every case. At an even ``input_dim`` the
        two rotations move by the same number of positions and produce the same
        tensor, so even-dimension outputs are bit-identical to those of the
        earlier code that reused the left rotation on both sides.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_units: int = 64,
        reverse: bool = False,
        activation: Union[str, callable] = "relu",
        use_tanh_stabilization: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the AffineCouplingLayer.

        See the class docstring for the parameters. ``transformation_net`` is
        created here, per the Keras 3 pattern, and built in ``build``.

        :raises ValueError: If ``input_dim`` is less than 2, if ``context_dim``
            is less than 1, or if ``hidden_units`` is less than 1.
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if input_dim < 2:
            raise ValueError(f"input_dim must be >= 2 to allow splitting, got {input_dim}")
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if hidden_units < 1:
            raise ValueError(f"hidden_units must be >= 1, got {hidden_units}")

        # Store configuration parameters
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_units = hidden_units
        self.reverse = reverse
        # Normalize activation to a callable/object so a callable passed in
        # (e.g. keras.activations.relu) round-trips via keras.activations.serialize.
        self.activation = keras.activations.get(activation)
        self.use_tanh_stabilization = use_tanh_stabilization

        # Compute splitting dimension
        self.split_dim = input_dim // 2
        dim_to_transform = self.input_dim - self.split_dim

        # CREATE transformation network in __init__ (modern Keras 3 pattern).
        # It sees the unchanged half plus the context, and its last Dense emits
        # dim_to_transform * 2 values: one log-scale and one shift for every
        # transformed dimension.
        net_input_size = self.split_dim + self.context_dim

        self.transformation_net = keras.Sequential([
            keras.layers.Dense(
                self.hidden_units,
                activation=self.activation,
                name="dense_1"
            ),
            keras.layers.Dense(
                self.hidden_units,
                activation=self.activation,
                name="dense_2"
            ),
            keras.layers.Dense(
                dim_to_transform * 2,
                activation=None,
                name="output_dense"
            )
        ], name="transformation_net")

    def build(self, input_shapes: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer and its transformation network.

        :param input_shapes: List of two shape tuples for ``[data, context]``.
        :type input_shapes: list[tuple[int | None, ...]]
        :raises ValueError: If ``input_shapes`` is not two shape tuples.
        """
        # Functional API may pass the two-element container of shapes as a tuple
        # (e.g. ((None, d), (None, c))); accept it by normalizing to a list. A
        # bare shape tuple like (None, 6) is NOT two shapes and stays invalid.
        if isinstance(input_shapes, tuple) and len(input_shapes) == 2 \
                and all(isinstance(s, (list, tuple)) for s in input_shapes):
            input_shapes = list(input_shapes)
        if not isinstance(input_shapes, list):
            raise ValueError(
                f"input_shapes must be a list of two shape tuples, got a "
                f"{type(input_shapes).__name__}: {input_shapes}"
            )
        if len(input_shapes) != 2:
            raise ValueError(
                f"input_shapes must be a list of two shape tuples, got "
                f"{len(input_shapes)}: {input_shapes}"
            )

        # Input size for the transformation network
        net_input_size = self.split_dim + self.context_dim

        # BUILD the transformation network (critical for serialization)
        self.transformation_net.build((None, net_input_size))

        # Always call parent build at the end
        super().build(input_shapes)

    def _apply_split_and_reverse(self, tensor: keras.KerasTensor) -> keras.KerasTensor:
        """Rotate the last axis left by ``split_dim`` when ``reverse`` is set.

        This is what makes a stack alternate which half gets transformed. With
        ``reverse=False`` it is the identity.

        :param tensor: Tensor whose last axis is ``input_dim`` wide.
        :type tensor: keras.KerasTensor
        :return: The rotated tensor, or ``tensor`` itself.
        :rtype: keras.KerasTensor
        """
        if self.reverse:
            return ops.concatenate([
                tensor[..., self.split_dim:],
                tensor[..., :self.split_dim]
            ], axis=-1)
        return tensor

    def _undo_split_and_reverse(self, tensor: keras.KerasTensor) -> keras.KerasTensor:
        """Rotate the last axis right by ``split_dim`` when ``reverse`` is set.

        This is the exact inverse of ``_apply_split_and_reverse`` for any
        ``input_dim``. At an even ``input_dim`` the two rotate by the same
        amount and produce the same tensor. With ``reverse=False`` it is the
        identity.

        :param tensor: Tensor whose last axis is ``input_dim`` wide.
        :type tensor: keras.KerasTensor
        :return: The rotated tensor, or ``tensor`` itself.
        :rtype: keras.KerasTensor
        """
        if self.reverse:
            return ops.concatenate([
                tensor[..., -self.split_dim:],
                tensor[..., :-self.split_dim]
            ], axis=-1)
        return tensor

    def _compute_scale_and_shift(
        self,
        static_part: keras.KerasTensor,
        context: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Predict the scale and shift from the static half and the context.

        Both ``forward`` and ``inverse`` call this with the same tensor, since
        the static half is identical on both sides of the transformation.

        :param static_part: The unchanged part of the input.
        :type static_part: keras.KerasTensor
        :param context: The conditioning context vector.
        :type context: keras.KerasTensor
        :return: Tuple of ``(scale, shift)`` parameters.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Concatenate static part and context for transformation network input
        net_input = ops.concatenate([static_part, context], axis=-1)

        # Get transformation parameters from network
        params = self.transformation_net(net_input)

        # Split into scale (log) and shift parameters
        dim_to_transform = self.input_dim - self.split_dim
        log_s = params[..., :dim_to_transform]
        t = params[..., dim_to_transform:]

        # Compute scale with numerical stabilization
        if self.use_tanh_stabilization:
            # Tanh keeps values in reasonable range but limits scale factor range
            s = ops.exp(ops.tanh(log_s))
        else:
            # Clipping prevents overflow but allows larger scale factors
            s = ops.exp(ops.clip(log_s, -10.0, 10.0))

        return s, t

    def forward(
        self,
        z: keras.KerasTensor,
        context: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Map latent ``z`` to data ``y``. This is the sampling direction.

        No log-determinant is returned here; only ``inverse`` needs it.

        :param z: Input from base distribution, shape ``(batch_size, input_dim)``.
        :type z: keras.KerasTensor
        :param context: Conditioning context, shape ``(batch_size, context_dim)``.
        :type context: keras.KerasTensor
        :return: Transformed tensor y, shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
        """
        # Apply permutation if this layer reverses the split
        z = self._apply_split_and_reverse(z)

        # z_a is the static half and is passed through untouched.
        # z_b is the half the affine transformation acts on.
        z_a = z[..., :self.split_dim]
        z_b = z[..., self.split_dim:]

        # Compute transformation parameters from static part and context
        s, t = self._compute_scale_and_shift(z_a, context)

        # Apply affine transformation to dynamic part
        y_b = z_b * s + t
        y = ops.concatenate([z_a, y_b], axis=-1)

        # Undo the entry rotation, restoring the original ordering.
        y = self._undo_split_and_reverse(y)

        return y

    def inverse(
        self,
        y: keras.KerasTensor,
        context: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Map data ``y`` back to latent ``z``. This is the likelihood direction.

        Returns ``log|det(dz/dy)| = -sum(log(s))`` alongside ``z`` -- the
        log-determinant of the map THIS method performs, not of the forward
        map. ``loss_func`` adds it directly.

        :param y: Transformed data tensor, shape ``(batch_size, input_dim)``.
        :type y: keras.KerasTensor
        :param context: Conditioning context, shape ``(batch_size, context_dim)``.
        :type context: keras.KerasTensor
        :return: Tuple of ``(z, log_det_jacobian)``, shaped
            ``(batch_size, input_dim)`` and ``(batch_size,)``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Apply permutation if this layer reverses the split
        y = self._apply_split_and_reverse(y)

        # y_a is the static half. It equals forward()'s z_a, which is why the
        # same conditioner output can be recovered here.
        # y_b is the half that forward() scaled and shifted.
        y_a = y[..., :self.split_dim]
        y_b = y[..., self.split_dim:]

        # Compute transformation parameters from static part and context
        s, t = self._compute_scale_and_shift(y_a, context)

        # Apply inverse transformation to dynamic part.
        # s = exp(...) is strictly positive, so no epsilon is needed; adding one
        # to only the inverse/log-det (but not the forward `y_b = z_b*s + t`)
        # would break exact invertibility for small s. Use plain s everywhere.
        z_b = (y_b - t) / s
        z = ops.concatenate([y_a, z_b], axis=-1)

        # Undo the entry rotation, restoring the original ordering.
        z = self._undo_split_and_reverse(z)

        # Log-determinant of the map THIS method performs, y -> z.
        # z_b = (y_b - t) / s, so dz/dy is diag(1/s) on the transformed half and
        # identity on the static half, giving -sum(log s). The rotations are
        # permutations and contribute nothing. Reporting +sum(log s) here (the
        # FORWARD map's value) is what made loss_func's NLL wrong.
        log_det_jacobian = -ops.sum(ops.log(s), axis=-1)

        return z, log_det_jacobian

    def compute_output_shape(
        self,
        input_shapes: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, which equals the data input shape.

        :param input_shapes: List of input shapes ``[data_shape, context_shape]``.
        :type input_shapes: list[tuple[int | None, ...]]
        :return: Output shape tuple.
        :rtype: tuple[int | None, ...]
        """
        data_shape = input_shapes[0]
        return data_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
            "hidden_units": self.hidden_units,
            "reverse": self.reverse,
            # Serialize the activation so a callable (e.g. keras.activations.relu)
            # round-trips as JSON; keras.activations.get rebuilds it on load.
            "activation": keras.activations.serialize(self.activation),
            "use_tanh_stabilization": self.use_tanh_stabilization,
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.normalizing_flow")
class NormalizingFlowLayer(keras.layers.Layer):
    """
    A stack of affine coupling layers that learns ``p(y | context)``.

    The layer holds ``num_flow_steps`` ``AffineCouplingLayer`` instances and
    runs them in one of two directions. ``call`` runs them backwards, mapping
    observed data ``y`` to the latent ``z`` and accumulating the total
    log-determinant; that is what ``loss_func`` turns into an exact negative
    log-likelihood. ``sample`` runs them forwards, mapping fresh Gaussian noise
    to new data.

    Coupling layer ``i`` is built with ``reverse=(i % 2 == 1)``. Even-indexed
    layers transform the second half of the vector; odd-indexed layers rotate
    first, so they transform the first half. Without that alternation the first
    ``split_dim`` dimensions would pass through every layer untouched and the
    stack would only ever be a fancy function of them.

    **Architecture Overview:**

    .. code-block:: text

          sample(): z ─► y                 call(): y ─► z
          runs 0, 1, ... K-1               runs K-1, ... 1, 0

            ┌────────────────────────────────────────────┐
            │ coupling_layers[0]      reverse=False      │
          ▼ │ z_a = first half   z_b = second half       │ ▲
            │ the SECOND half is scaled and shifted      │
            └────────────────────────────────────────────┘
            ┌────────────────────────────────────────────┐
            │ coupling_layers[1]      reverse=True       │
          ▼ │ rotated: halves swap roles                 │ ▲
            │ the FIRST half is scaled and shifted       │
            └────────────────────────────────────────────┘
                          ... alternating ...
            ┌────────────────────────────────────────────┐
            │ coupling_layers[num_flow_steps - 1]        │
          ▼ │ reverse = (i % 2 == 1)                     │ ▲
            └────────────────────────────────────────────┘

    Every layer owns its own weights; nothing is shared between steps. Only the
    traversal order differs between the two directions.

    **Entry Points:**

    .. code-block:: text

        call([y, context])              sample(n, context)
                   │                               │
                   ▼                               ▼
        ┌──────────────────────┐        ┌──────────────────────┐
        │ inverse() through    │        │ z ~ N(0, I)          │
        │ reversed(couplings)  │        │ (batch, n, out_dim)  │
        │ sum the ldj terms    │        │ reshape to 2D, tile  │
        └──────────┬───────────┘        │ context to match     │
                   ▼                    └──────────┬───────────┘
          z          (batch, out_dim)              ▼
          total_ldj  (batch,)           ┌──────────────────────┐
                   │                    │ forward() through    │
                   ▼                    │ couplings in order   │
        ┌──────────────────────┐        └──────────┬───────────┘
        │ loss_func(y, y_pred) │                   ▼
        │ -mean(log p(y))      │        y  (batch, n, out_dim)
        └──────────┬───────────┘
                   ▼
             scalar loss

    ``loss_func`` is a plain method, not a Keras loss object. Feed it the tuple
    ``call`` returned. ``y_true`` is accepted for signature compatibility and
    is not read.

    :param output_dimension: Dimensionality of the target distribution. Must be
        >= 2. Stored as ``self.output_dim``.
    :type output_dimension: int
    :param num_flow_steps: Number of coupling layers to stack. Must be >= 1.
    :type num_flow_steps: int
    :param context_dim: Dimensionality of the conditioning context. Must be >= 1.
    :type context_dim: int
    :param hidden_units_coupling: Hidden width inside each coupling layer's
        conditioner. Must be >= 1. Defaults to 64.
    :type hidden_units_coupling: int
    :param activation: Activation for the coupling conditioners. Defaults to
        "relu".
    :type activation: str | callable
    :param use_tanh_stabilization: Passed straight to every coupling layer.
        Defaults to True.
    :type use_tanh_stabilization: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar coupling_layers: The stack, in forward order. Index 0 is applied
        first by ``sample`` and last by ``call``.
    :vartype coupling_layers: list[AffineCouplingLayer]

    :raises ValueError: If ``output_dimension`` is less than 2, if
        ``num_flow_steps`` is less than 1, if ``context_dim`` is less than 1,
        or if ``hidden_units_coupling`` is less than 1.

    Input shape:
        A list of two tensors, ``[data, context]``, of shape
        ``(batch_size, output_dimension)`` and ``(batch_size, context_dim)``.

    Output shape:
        ``call`` returns ``(batch_size, output_dimension)`` and
        ``(batch_size,)``. ``sample`` returns
        ``(batch_size, num_samples, output_dimension)``. ``loss_func`` returns
        a scalar.

    Example:
        >>> flow = NormalizingFlowLayer(4, num_flow_steps=4, context_dim=8)
        >>> z, ldj = flow([y, context])
        >>> loss = flow.loss_func(y, (z, ldj))
        >>> draws = flow.sample(100, context)

    Note:
        Any ``output_dimension >= 2`` round-trips, odd or even, at any
        ``num_flow_steps``. The odd-indexed coupling layers run with
        ``reverse=True``, and each undoes its own entry rotation with the
        inverse rotation on exit. Measured at ``output_dimension`` in 3 through
        7 and ``num_flow_steps`` in 2 through 4, ``context_dim=3``, batch 8: a
        tensor round-tripped through ``call`` and forward again differs by
        ~1e-07 float32 round-off.
    """

    def __init__(
        self,
        output_dimension: int,
        num_flow_steps: int,
        context_dim: int,
        hidden_units_coupling: int = 64,
        activation: Union[str, callable] = "relu",
        use_tanh_stabilization: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the NormalizingFlowLayer.

        See the class docstring for the parameters. The coupling layers are
        created here, per the Keras 3 pattern, and built in ``build``.

        :raises ValueError: If ``output_dimension`` is less than 2, if
            ``num_flow_steps`` is less than 1, if ``context_dim`` is less than
            1, or if ``hidden_units_coupling`` is less than 1.
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if output_dimension < 2:
            raise ValueError(
                f"output_dimension must be >= 2 to allow splitting, got "
                f"{output_dimension}"
            )
        if num_flow_steps < 1:
            raise ValueError(f"num_flow_steps must be >= 1, got {num_flow_steps}")
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if hidden_units_coupling < 1:
            raise ValueError(
                f"hidden_units_coupling must be >= 1, got {hidden_units_coupling}"
            )

        # Store configuration parameters
        self.output_dim = output_dimension
        self.num_flow_steps = num_flow_steps
        self.context_dim = context_dim
        self.hidden_units_coupling = hidden_units_coupling
        # Normalize activation to a callable/object so a callable passed in
        # (e.g. keras.activations.relu) round-trips via keras.activations.serialize.
        self.activation = keras.activations.get(activation)
        self.use_tanh_stabilization = use_tanh_stabilization

        # CREATE coupling layers in __init__ (modern Keras 3 pattern).
        # reverse=(i % 2 == 1) alternates which half each layer transforms, so
        # every dimension is eventually transformed instead of only the second
        # half.
        self.coupling_layers = []
        for i in range(self.num_flow_steps):
            layer = AffineCouplingLayer(
                input_dim=self.output_dim,
                context_dim=self.context_dim,
                hidden_units=self.hidden_units_coupling,
                reverse=(i % 2 == 1),
                activation=self.activation,
                use_tanh_stabilization=self.use_tanh_stabilization,
                name=f"affine_coupling_{i}"
            )
            self.coupling_layers.append(layer)

    def build(self, input_shapes: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer and all coupling layers.

        :param input_shapes: List of two shape tuples for ``[data, context]``.
        :type input_shapes: list[tuple[int | None, ...]]
        :raises ValueError: If ``input_shapes`` is not two shape tuples.
        """
        # Functional API may pass the two-element container of shapes as a tuple
        # (e.g. ((None, d), (None, c))); accept it by normalizing to a list. A
        # bare shape tuple like (None, 4) is NOT two shapes and stays invalid.
        if isinstance(input_shapes, tuple) and len(input_shapes) == 2 \
                and all(isinstance(s, (list, tuple)) for s in input_shapes):
            input_shapes = list(input_shapes)
        if not isinstance(input_shapes, list):
            raise ValueError(
                f"input_shapes must be a list of two shape tuples, got a "
                f"{type(input_shapes).__name__}: {input_shapes}"
            )
        if len(input_shapes) != 2:
            raise ValueError(
                f"input_shapes must be a list of two shape tuples, got "
                f"{len(input_shapes)}: {input_shapes}"
            )

        # BUILD all coupling layers (critical for serialization)
        for layer in self.coupling_layers:
            layer.build(input_shapes)

        # Always call parent build at the end
        super().build(input_shapes)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Map data ``y`` to latent ``z``, accumulating the log-determinant.

        The coupling layers run in reverse index order, which is the inverse of
        the order ``sample`` uses. Pass the returned tuple to ``loss_func``.

        :param inputs: List of ``[data, context]`` tensors.
        :type inputs: list[keras.KerasTensor]
        :param training: Boolean for training mode. Not read; the layer behaves
            the same either way.
        :type training: bool | None
        :return: Tuple of ``(z, total_log_det_jacobian)``, shaped
            ``(batch_size, output_dimension)`` and ``(batch_size,)``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        :raises ValueError: If ``inputs`` does not hold exactly two tensors.
        """
        if len(inputs) != 2:
            raise ValueError(
                f"Expected exactly 2 inputs [data, context], got {len(inputs)}"
            )

        y, context = inputs

        # Initialize log-determinant accumulator
        batch_size = ops.shape(y)[0]
        total_log_det_jacobian = ops.zeros(batch_size, dtype=y.dtype)

        # Apply inverse transformations in reverse order (y → z)
        z = y
        for layer in reversed(self.coupling_layers):
            z, ldj = layer.inverse(z, context)
            total_log_det_jacobian += ldj

        return z, total_log_det_jacobian

    def loss_func(
        self,
        y_true: keras.KerasTensor,
        y_pred: Tuple[keras.KerasTensor, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Compute the exact negative log-likelihood, averaged over the batch.

        :param y_true: Accepted for signature compatibility. Not read.
        :type y_true: keras.KerasTensor
        :param y_pred: Tuple of ``(z, total_log_det_jacobian)`` from ``call``.
        :type y_pred: tuple[keras.KerasTensor, keras.KerasTensor]
        :return: Scalar negative log-likelihood loss.
        :rtype: keras.KerasTensor
        """
        z, total_log_det_jacobian = y_pred

        # Log-probability under base distribution (standard multivariate normal)
        # log π(z) = -0.5 * [d*log(2π) + ||z||²]
        log_prob_z = -0.5 * (
            self.output_dim * ops.log(2 * np.pi) +
            ops.sum(z ** 2, axis=-1)
        )

        # Change of variables. total_log_det_jacobian is log|det(dz/dy)|,
        # the log-determinant of the y -> z map call() performed, so it is
        # ADDED here. It equals -sum(log s); adding +sum(log s) instead is the
        # CD-10 defect and makes p(y) integrate to ~4.8 rather than 1.
        log_prob_y = log_prob_z + total_log_det_jacobian

        # Return negative log-likelihood for minimization
        return -ops.mean(log_prob_y)

    def sample(
        self,
        num_samples: int,
        context: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Draw samples from the learned conditional distribution.

        Draws ``num_samples`` Gaussian vectors per row of ``context`` and runs
        them forward through the stack in index order.

        :param num_samples: Number of samples per context row. Must be >= 1.
        :type num_samples: int
        :param context: Conditioning context, shape ``(batch_size, context_dim)``.
        :type context: keras.KerasTensor
        :return: Samples of shape
            ``(batch_size, num_samples, output_dimension)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``num_samples`` is less than 1.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        batch_size = ops.shape(context)[0]

        # Sample from base distribution (standard multivariate normal)
        z = keras.random.normal(
            shape=(batch_size, num_samples, self.output_dim),
            dtype=context.dtype
        )

        # Reshape z for efficient batch processing through layers
        # (batch, num_samples, output_dim) -> (batch * num_samples, output_dim)
        y_flat = ops.reshape(z, (-1, self.output_dim))

        # Prepare context to match the flattened z
        # context: (batch, context_dim) -> (batch, 1, context_dim)
        context_expanded = ops.expand_dims(context, 1)
        # -> (batch, num_samples, context_dim)
        context_tiled = ops.repeat(context_expanded, num_samples, axis=1)
        # -> (batch * num_samples, context_dim)
        context_flat = ops.reshape(context_tiled, (-1, self.context_dim))

        # Apply forward transformations (z → y)
        for layer in self.coupling_layers:
            y_flat = layer.forward(y_flat, context_flat)

        # Reshape back to the original sample structure: (batch, num_samples, output_dim)
        y = ops.reshape(y_flat, (batch_size, num_samples, self.output_dim))

        return y

    def compute_output_shape(
        self,
        input_shapes: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes for ``(z, log_det_jacobian)``.

        :param input_shapes: List of input shapes ``[data_shape, context_shape]``.
        :type input_shapes: list[tuple[int | None, ...]]
        :return: Tuple of the data shape and the ``(batch_size,)`` ldj shape.
        :rtype: tuple[tuple[int | None, ...], tuple[int | None, ...]]
        """
        data_shape = input_shapes[0]
        batch_size = data_shape[0] if data_shape else None
        ldj_shape = (batch_size,)
        return data_shape, ldj_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dimension": self.output_dim,
            "num_flow_steps": self.num_flow_steps,
            "context_dim": self.context_dim,
            "hidden_units_coupling": self.hidden_units_coupling,
            # Serialize the activation so a callable (e.g. keras.activations.relu)
            # round-trips as JSON; keras.activations.get rebuilds it on load.
            "activation": keras.activations.serialize(self.activation),
            "use_tanh_stabilization": self.use_tanh_stabilization,
        })
        return config


# ---------------------------------------------------------------------
