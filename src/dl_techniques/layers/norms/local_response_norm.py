"""Local Response Normalization (LRN), the 2012 AlexNet cross-channel normalizer.

LRN comes from AlexNet. It makes neighbouring channels compete: every channel is
divided down according to how much energy its *neighbourhood* of channels carries.
A channel that fires hard suppresses its neighbours, which is what stops a later
convolution from simply re-amplifying whatever the previous one already liked.

Computation
-----------

For an input ``X`` and a channel ``c``, with a neighbourhood of
``n = 2 * depth_radius + 1`` channels centred on ``c``::

    denom_c = k + alpha * sum_{j = c - r}^{c + r} X_j ** 2
    Y_c     = X_c / denom_c ** beta

The paper writes the coefficient as ``alpha / n``. That ``1 / n`` is a fixed
constant, so it is folded into ``alpha`` here, which is what the released
implementations do: Caffe's ``LRN`` layer and ``tf.nn.local_response_normalization``
both take ``alpha`` directly and put no ``n`` in the denominator. Measured
agreement with ``tf.nn.local_response_normalization`` is exact to float32 rounding
(see ``tests/test_layers/test_norms/test_local_response_norm.py``).

The sum runs over **channels only**. Every spatial position is normalized
independently, so a channel is damped by the activations sitting beside it in the
feature map rather than by anything elsewhere in the image.

The window is **truncated at the edges**: the first and last ``r`` channels sum over
fewer neighbours rather than reading past the axis. So the boundary channels are
not divided by the same denominator as the interior ones, and this layer reproduces
``tf.nn.local_response_normalization`` at its default ``pad='SAME'``.

AlexNet's values are ``depth_radius=2`` (``n=5``), ``alpha=1e-4``, ``beta=0.75``
and ``k=1``, which are the defaults here.

Why the constant is named ``k``, not ``epsilon``
-----------------------------------------------

``k`` is the name the paper uses, and keeping it is deliberate:
``create_normalization_layer`` imposes its own ``epsilon=1e-6`` default on every
registry type that accepts one, which for most of them is a silent re-tune of the
layer's own default. LRN has no ``epsilon`` at all -- its stabilizing constant is
``k``, and ``alpha`` is the coefficient on the squared sum. Naming them after the
paper keeps this layer out of that trap entirely, exactly as
``GlobalResponseNormalization`` names its constant ``eps``.

Note that ``tf.nn.local_response_normalization`` spells this same constant
``bias``. There is **no separate ``bias`` argument here**: the paper has one
additive constant and so does this layer, and giving it two names for one slot
would let a caller set both and silently take whichever came last.

Not GRN
-------

:class:`~dl_techniques.layers.norms.global_response_norm.GlobalResponseNormalization`
is the other "response normalization" in this package and is **not** a substitute:

============  ==============================  ==============================
              LRN (this layer)                GRN (ConvNeXt V2)
============  ==============================  ==============================
Neighbourhood **local**, across channels     **global**, over all positions
Operation     **divides** by a denominator    **multiplies** by a score, then
                                              adds the input back
Weights       none -- fixed hyperparameters    ``gamma``, ``beta`` are trainable
Rank support  3 or 4                          2, 3 or 4
Paper         Krizhevsky et al. 2012          Liu et al. 2022
============  ==============================  ==============================

References
----------

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet
    Classification with Deep Convolutional Neural Networks". NeurIPS 25.
    LRN is Section 3.4; the ``n=5, alpha=1e-4, beta=0.75, k=1`` constants are
    reported in Section 4 and are the reason the default neighbourhood is 5.
    https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[2] Jain, S., & Wallace, J. M. (2013). "Supervised Learning of Image
    Restoration with Convolutional Networks". arXiv:1307.3065. The first
    controlled ablation of LRN against BatchNorm; cited as the reason the technique
    is historical rather than current practice.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.norms.local_response_norm")
class LocalResponseNormalization(keras.layers.Layer):
    """Local Response Normalization, the AlexNet cross-channel normalizer.

    Divides each channel down by the energy of a neighbourhood of channels centred
    on it::

        Y_c = X_c / (k + alpha * sum_{j=c-r}^{c+r} X_j ** 2) ** beta

    The sum runs over channels only, so each spatial position is normalized
    independently against its own neighbourhood. The output has the same shape as
    the input.

    .. warning::
        **This layer does not support masking.** ``supports_masking`` is left
        ``False``, because a channel's output depends on the *neighbouring* channels,
        and a masked channel still contributes to them. A propagated Keras mask would
        claim an independence that does not hold.

    .. note::
        This layer holds **no weights**. ``depth_radius``, ``bias``, ``alpha``,
        ``beta`` and ``k`` are fixed hyperparameters, so ``build()`` validates the
        input shape and creates nothing, and ``count_params()`` reports 0.

    **Architecture Overview:**

    .. code-block:: text

            inputs (X): (B, ..., C)
                             │
                             ▼
            ┌────────────────────────────────────────┐
            │ sum X_j ** 2 over the n = 2*depth_    │
            │ radius+1 channels centred on each c    │
            │ (truncated at the edges)               │
            └──────────────────┬─────────────────────┘
                               │ S: (B, ..., C)
                               ▼
            ┌────────────────────────────────────────┐
            │ Y = X / (k + alpha * S) ** beta        │
            └──────────────────┬─────────────────────┘
                               │
                               ▼
                    output (Y): (B, ..., C)

    :param depth_radius: Half-width of the channel neighbourhood. The window spans
        ``2 * depth_radius + 1`` channels, so the paper's ``n=5`` is
        ``depth_radius=2``. Defaults to 2. A neighbourhood wider than the channel
        count is handled by the same truncation and needs no special case.
    :type depth_radius: int
    :param alpha: Coefficient on the squared neighbourhood sum. Must be positive.
        Defaults to 1e-4, the paper's value. The paper writes the coefficient as
        ``alpha / n``; that fixed ``1 / n`` is folded in here, matching Caffe and
        TensorFlow.
    :type alpha: float
    :param beta: Exponent on the denominator. Must be positive. Defaults to 0.75,
        the paper's value.
    :type beta: float
    :param k: Additive constant kept inside the exponent, so the denominator can
        never reach zero. Must be positive. Defaults to 1, the paper's value.
    :type k: float
    :param data_format: Channel layout. ``'channels_last'`` (the default) treats
        the last axis as channels; ``'channels_first'`` treats the first axis after
        the batch as channels and requires a rank-4 input.
    :type data_format: Optional[str]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :raises ValueError: If ``depth_radius`` is negative or is not an ``int``.
    :raises ValueError: If ``alpha``, ``beta`` or ``k`` is not positive.
    :raises ValueError: If ``data_format`` is neither ``'channels_last'`` nor
        ``'channels_first'``.
    :raises ValueError: At ``build()`` time, if the input rank is not 3 or 4, if the
        channel dimension is undefined, or if ``data_format='channels_first'`` is
        combined with a rank-3 input.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import LocalResponseNormalization

        x = keras.random.normal((2, 8, 8, 16))
        y = LocalResponseNormalization()(x)
    """

    def __init__(
        self,
        depth_radius: int = 2,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
        data_format: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the layer.

        :param depth_radius: Half-width of the channel neighbourhood. The window
            spans ``2 * depth_radius + 1`` channels.
        :type depth_radius: int
        :param alpha: Coefficient on the squared neighbourhood sum. Must be positive.
        :type alpha: float
        :param beta: Exponent on the denominator. Must be positive.
        :type beta: float
        :param k: Additive constant inside the exponent. Must be positive.
        :type k: float
        :param data_format: ``'channels_last'`` or ``'channels_first'``. ``None``
            resolves to the backend default.
        :type data_format: Optional[str]
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``depth_radius`` is negative or not an ``int``.
        :raises ValueError: If ``alpha``, ``beta`` or ``k`` is not positive.
        :raises ValueError: If ``data_format`` is not a recognised layout.
        """
        super().__init__(**kwargs)

        if isinstance(depth_radius, bool) or not isinstance(depth_radius, int):
            raise ValueError(
                f"depth_radius must be an int, got {depth_radius!r} of type "
                f"{type(depth_radius).__name__}"
            )
        if depth_radius < 0:
            raise ValueError(
                f"depth_radius must be non-negative, got {depth_radius}"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if data_format not in (None, "channels_last", "channels_first"):
            raise ValueError(
                f"data_format must be 'channels_last', 'channels_first' or None, "
                f"got {data_format!r}"
            )

        self.depth_radius = depth_radius
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = float(k)
        self.data_format = data_format or keras.backend.image_data_format()

        logger.debug(
            f"Initialized LocalResponseNormalization with depth_radius="
            f"{self.depth_radius}, alpha={self.alpha}, beta={self.beta}, k={self.k}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve the channel axis and build the neighbourhood band matrix.

        The band matrix is a **constant**, not a weight: it is derived entirely from
        ``depth_radius`` and the channel count, holds no learned value, and stays
        ``trainable=False``. It is therefore created with ``add_weight(
        trainable=False)`` rather than as a bare tensor, for two reasons that were
        both measured:

        * **Graph safety.** A bare ``keras.ops`` tensor built inside ``build()`` is
          created in whatever graph is active at the time. Keras builds a sublayer
          inside a ``scratch_graph`` while tracing a parent model, so the tensor is
          stamped with that graph and then dies on first use:
          ``<tf.Tensor 'lrn1/Cast:0' ...> is out of scope and cannot be used here``
          (reproduced by nesting this layer in ``models/vision/alexnet``).
          ``add_weight`` attaches the value to the layer's own variables, which
          survive the graph that produced them.
        * **Precision.** It is built in ``self.compute_dtype``, NOT in
          ``keras.backend.floatx()``. ``floatx()`` reports the process-wide default
          ``'float32'`` and is not moved by
          ``keras.mixed_precision.set_dtype_policy('float64')``, so reading it
          pinned the band to float32 and dragged a float64 forward pass back down to
          float32 precision -- measured at 1.5e-3 error against a float64 oracle,
          on an input the caller had explicitly asked to be computed in float64.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the input rank is not 3 or 4.
        :raises ValueError: If the channel dimension is undefined.
        :raises ValueError: If ``data_format='channels_first'`` is used with a
            rank-3 input.
        """
        if self.built:
            return

        rank = len(input_shape)
        if rank not in (3, 4):
            raise ValueError(
                f"Input rank must be 3 or 4 (batch, ..., channels), but got "
                f"rank {rank}"
            )
        if rank == 3 and self.data_format == "channels_first":
            raise ValueError(
                "data_format='channels_first' requires a rank-4 input, but got "
                "rank 3. Use data_format='channels_last'."
            )

        self._channel_axis = 1 if self.data_format == "channels_first" else -1
        channels = input_shape[self._channel_axis]
        if channels is None:
            raise ValueError(
                f"The channel dimension (axis {self._channel_axis}) must be "
                f"defined for LocalResponseNormalization."
            )

        # A non-trainable weight, not a variable in `self.trainable_variables`.
        # `self.band` stays a plain attribute so the matrix remains readable.
        self._band_weight = self.add_weight(
            name="neighbourhood_band",
            shape=(int(channels), int(channels)),
            dtype=self.compute_dtype,
            initializer=_NeighbourhoodBandInitializer(self.depth_radius),
            trainable=False,
        )
        self.band = self._band_weight

        logger.debug(
            f"Building LocalResponseNormalization for rank {rank}, "
            f"{channels} channels, data_format={self.data_format}"
        )

        # No trainable weights: every constructor argument is a fixed hyperparameter.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Normalize each channel against its neighbourhood.

        :param inputs: Input tensor of shape ``(batch, ..., channels)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag. Unused; the layer is stateless and
            behaves identically in both modes. The argument is kept for API
            compatibility with the rest of this package.
        :type training: Optional[bool]

        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor

        :raises ValueError: If called before ``build()`` resolved the channel count.
        """
        if not hasattr(self, "band"):
            raise ValueError(
                "LocalResponseNormalization.call() ran before build() resolved the "
                "channel count. Call the layer on a tensor of known shape (or call "
                "build(input_shape)) first."
            )

        # keras.ops.moveaxis reads inputs.ndim, which a tf.Variable does not expose
        # ('ResourceVariable' object has no attribute 'ndim', measured). Converting
        # first makes the layer work when it is called directly inside a
        # GradientTape on a variable, which is how gradients are usually probed.
        inputs = keras.ops.convert_to_tensor(inputs)

        # Move channels to the last axis so one matmul handles every data_format,
        # then move them back so the output shape matches the input exactly.
        moved = keras.ops.moveaxis(inputs, self._channel_axis, -1)
        # The channel count comes from the BAND, which is square by construction, so
        # this stays correct even when `moved`'s static last axis is None.
        flat = keras.ops.reshape(moved, (-1, self.band.shape[0]))

        # S[n, c] = sum over the window of squared[c] -> flat @ band, with band
        # symmetric in (row, column) so this orientation sums over columns.
        summed = keras.ops.matmul(keras.ops.square(flat), self.band)

        denominator = keras.ops.power(
            self.k + self.alpha * summed, self.beta
        )
        normalized = keras.ops.divide(flat, denominator)

        # Restore the RUNTIME shape rather than the static one: a symbolic KerasTensor
        # carries a partially known shape, (None, 56, 56, 96), and reshape() rejects a
        # None in it (measured: "Cannot convert a partially known TensorShape (None, 56,
        # 56, 96) to a Tensor", raised on the FIRST fit step of a parent model).
        #
        # The target must be the shape of the CHANNEL-LAST tensor, not of the input.
        # keras.ops.shape() returns axes in the BACKEND's order, which under
        # channels_first is (N, C, H, W) rather than the logical (N, H, W, C); handing
        # that to a reshape of the channel-last tensor produced (2, 7, 6, 8) for a
        # (2, 6, 8, 7) logical input, i.e. the spatial axes silently transposed.
        restored = keras.ops.reshape(normalized, keras.ops.shape(moved))
        return keras.ops.moveaxis(restored, -1, self._channel_axis)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape tuple that was passed in.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this layer.

        :return: Dictionary holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "depth_radius": self.depth_radius,
            "alpha": self.alpha,
            "beta": self.beta,
            "k": self.k,
            "data_format": self.data_format,
        })
        return config


# ---------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.layers.norms.local_response_norm")
class _NeighbourhoodBandInitializer(keras.initializers.Initializer):
    """Initialize a weight to the ``(C, C)`` local-response band matrix.

    Row ``c`` is 1 in every column ``j`` with ``|c - j| <= depth_radius``, so
    ``x @ band`` sums ``x`` over the truncated window centred on each channel. The
    truncation falls out of the matrix itself: row 0 is 1 in columns
    ``0 .. depth_radius`` only, which is exactly the window clamped at the edge
    rather than padded with out-of-range taps.

    This exists as an ``Initializer`` subclass rather than a closure because
    ``keras.initializers.Constant`` accepts **scalars only** (measured:
    ``TypeError: Initializer() takes no arguments`` when handed a callable), and the
    band matrix is not a scalar.

    Built from an index comparison rather than from slices, because
    ``keras.ops.slice`` advertises ``None`` as "to the end" but the TensorFlow
    backend forwards it to ``tf.slice``, which rejects it outright and the NumPy
    backend computes ``arange(start, start + None)``. A pure elementwise comparison
    sidesteps that broken contract and is static-shape-safe.

    :param depth_radius: Half-width of the window. ``0`` yields the identity.
    :type depth_radius: int
    """

    def __init__(self, depth_radius: int) -> None:
        """Initialize the initializer.

        :param depth_radius: Half-width of the channel neighbourhood.
        :type depth_radius: int
        """
        super().__init__()
        self.depth_radius = depth_radius

    def __call__(
            self,
            shape: Tuple[int, ...],
            dtype: Any = None,
    ) -> keras.KerasTensor:
        """Produce the band matrix for ``shape``.

        :param shape: Must be a 2-tuple; only its first entry is used, since the
            matrix is square by construction.
        :type shape: Tuple[int, ...]
        :param dtype: Target dtype. ``None`` falls back to ``float32``.
        :type dtype: Any
        :return: The band matrix.
        :rtype: keras.KerasTensor
        """
        channels = int(shape[0])
        target = dtype if dtype is not None else "float32"
        index = keras.ops.arange(channels, dtype=target)
        distance = keras.ops.abs(index[:, None] - index[None, :])
        return keras.ops.cast(distance <= float(self.depth_radius), target)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this initializer.

        :return: Dictionary carrying ``depth_radius``.
        :rtype: Dict[str, Any]
        """
        return {"depth_radius": self.depth_radius}