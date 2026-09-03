"""
Mixture Density Network (MDN) layer.

An MDN replaces a network's single point output with the parameters of a
Gaussian mixture. For one input the layer emits three things: the component
means mu, the component standard deviations sigma, and the mixture weights pi.
The predicted density is::

    p(y|x) = sum_i pi_i(x) * N(y | mu_i(x), sigma_i(x))

That lets the model say "the answer is either here or there" instead of
averaging the two into a value it never expects to see.

**What this version adds over a textbook MDN:**

- Each of the three heads gets its own Dense -> BatchNormalization ->
  activation path, so the three parameter types can learn different features.
- Optional diversity regularization pushes the component means apart, which is
  the lever against component collapse.
- Sigma is floored at ``min_sigma``, so no component can claim zero variance.
- ``use_bias`` switches off the Dense biases and the BatchNormalization centers
  together, for bias-free setups.
- The negative log-likelihood runs in log space, so it does not underflow at
  large ``output_dimension`` and needs no epsilon clamp.

**Uses:**

- Time series forecasting with uncertainty
- Control systems with multiple possible outcomes
- Robotics and reinforcement learning
- Inverse problems
- Financial modeling with risk assessment

References:
    - Bishop, C. M. (1994). Mixture Density Networks.
    - Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.
    - Ha, D., & Schmidhuber, J. (2018). World Models.
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Default floor for sigma. Keeps a component from claiming zero variance.
MIN_SIGMA_DEFAULT = 1e-3

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.statistics.mdn_layer")
class MDNLayer(keras.layers.Layer):
    """Mixture Density Network layer with three separate parameter heads.

    The layer turns one input vector into the parameters of a Gaussian mixture
    over the target space::

        p(y|x) = sum_i pi_i(x) * N(y | mu_i(x), sigma_i(x))

    There are ``num_mixtures`` components and the target has
    ``output_dimension`` dimensions. Each component is axis-aligned, so there is
    one sigma per component per output dimension.

    The three parameter groups do not share a path. Means, standard deviations
    and mixture weights each get their own Dense layer, an optional
    BatchNormalization and an activation, then their own final projection. All
    three read the same input tensor and nothing else.

    A Keras layer returns one tensor, so the three projections are concatenated
    on the last axis. ``split_mixture_params`` takes that tensor apart again.

    Set ``diversity_regularizer_strength`` above 0 and the layer adds
    ``exp(-||mu_i - mu_j||^2)`` over every distinct component pair as a training
    loss. That pushes the means apart and is the lever against component
    collapse.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────────┐
        │ inputs                                          (batch, input_dim) │
        └──────────┬───────────────────────┬───────────────────────┬──────────┘
                   ▼                       ▼                       ▼
        ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
        │ intermediate_mu     │ │ intermediate_sigma  │ │ intermediate_pi     │
        │   _dense            │ │   _dense            │ │   _dense            │
        │ intermediate_mu_bn  │ │ intermediate_sigma  │ │ intermediate_pi_bn  │
        │   (optional)        │ │   _bn   (optional)  │ │   (optional)        │
        │ activation          │ │ activation          │ │ activation          │
        └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                   │ (B, inter)            │ (B, inter)            │ (B, inter)
                   ▼                       ▼                       ▼
        ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
        │ mdn_mus             │ │ mdn_sigmas          │ │ mdn_pi              │
        │ Dense, linear       │ │ Dense               │ │ Dense, NO activation│
        │                     │ │ softplus + min_sigma│ │ emits raw logits    │
        └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                   │ [B, M*D]              │ [B, M*D]              │ [B, M]
                   └───────────────────────┼───────────────────────┘
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │ ops.concatenate([mu, sigma, pi], axis=-1) │
                     │ [B, 2*M*D + M]                            │
                     └───────────────────────────────────────────┘

    ``B`` is the batch, ``M`` is ``num_mixtures``, ``D`` is ``output_dimension``
    and ``inter`` is ``intermediate_units``. The three towers are independent:
    they share no weights, only the input tensor. The BatchNormalization stages
    exist only when ``use_batch_norm`` is True.

    One branch is not drawn, because it produces no output tensor. When
    ``diversity_regularizer_strength`` is above 0 and ``training`` is True,
    ``mu_output`` also feeds ``_compute_diversity_loss`` and the resulting
    scalar goes to ``add_loss``.

    **Output Layout:**

    .. code-block:: text

        The single output tensor packs all three groups on axis -1.
        Getting these offsets wrong is the classic bug in this layer.

        0              M*D          2*M*D     2*M*D + M
        ├────────────────┼──────────────┼─────────────┤
        │       mu       │    sigma     │     pi      │
        │   M*D values   │  M*D values  │  M values   │
        └───────┬────────┴───────┬──────┴──────┬──────┘
                ▼                ▼             ▼
          -> [B, M, D]     -> [B, M, D]  [B, M] as-is

        split_mixture_params cuts it back apart:

            mu_end    = num_mix * output_dim
            sigma_end = mu_end + num_mix * output_dim
            out_mu    = y_pred[..., :mu_end]
            out_sigma = y_pred[..., mu_end:sigma_end]
            out_pi    = y_pred[..., sigma_end:]

    Only ``mu`` and ``sigma`` are reshaped to ``[B, M, D]``. ``pi`` stays
    two-dimensional at ``[B, M]``, because there is one mixture weight per
    component, not one per output dimension. Every consumer of ``pi`` in this
    module relies on that.

    :param output_dimension: Number of dimensions in the target space. Must be
        positive.
    :type output_dimension: int
    :param num_mixtures: Number of Gaussian components. Must be positive.
    :type num_mixtures: int
    :param use_bias: Whether the Dense layers get bias vectors. Also sets
        ``center`` on the BatchNormalization layers, so a bias-free layer stays
        bias-free end to end. Defaults to "True".
    :type use_bias: bool
    :param diversity_regularizer_strength: Weight on the pairwise repulsion
        between component means. 0.0 disables it. Defaults to 0.0.
    :type diversity_regularizer_strength: float
    :param intermediate_units: Width of the three intermediate Dense layers.
        Must be positive. Defaults to 32.
    :type intermediate_units: int
    :param use_batch_norm: Whether each head gets a BatchNormalization between
        its Dense layer and its activation. Defaults to "True".
    :type use_batch_norm: bool
    :param intermediate_activation: Activation applied after each intermediate
        stage. Stored through ``deserialize_activation``, so a callable
        round-trips. Defaults to "relu".
    :type intermediate_activation: str
    :param kernel_initializer: Initializer for the Dense kernels. Each Dense
        gets its own clone, never the shared instance. Defaults to
        "glorot_normal".
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for the Dense biases, cloned the same
        way. Defaults to "zeros".
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for the Dense kernels. Defaults to
        ``L2(1e-5)``.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param bias_regularizer: Regularizer for the Dense biases. Defaults to
        ``L2(1e-6)``.
    :type bias_regularizer: keras.regularizers.Regularizer | None
    :param min_sigma: Floor on every standard deviation. Added after the
        softplus in ``mdn_sigmas``, and applied again in ``loss_func`` and
        ``sample``. Defaults to 1e-3.
    :type min_sigma: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``output_dimension``, ``num_mixtures`` or
        ``intermediate_units`` is not positive, or if
        ``diversity_regularizer_strength`` is negative.

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``.

    Output shape:
        2D tensor of shape
        ``(batch_size, 2 * num_mixtures * output_dimension + num_mixtures)``.

    Example:
        >>> mdn = MDNLayer(output_dimension=2, num_mixtures=5)
        >>> params = mdn(features)
        >>> mu, sigma, pi_logits = mdn.split_mixture_params(params)
        >>> nll = mdn.loss_func(y_true, params)

    Note:
        ``mdn_pi`` emits raw logits. Apply exactly one softmax or log_softmax to
        that slice. A second one changes the mixture, because
        ``softmax(softplus(z))`` is not ``softmax(z)``.

    Note:
        The diversity loss is added only when ``training`` is exactly ``True``.
        Passing ``training=None`` skips it.
    """

    def __init__(
        self,
        output_dimension: int,
        num_mixtures: int,
        use_bias: bool = True,
        diversity_regularizer_strength: float = 0.0,
        intermediate_units: int = 32,
        use_batch_norm: bool = True,
        intermediate_activation: str = "relu",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
        bias_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-6),
        min_sigma: float = MIN_SIGMA_DEFAULT,
        **kwargs: Any
    ) -> None:
        """Initialize the MDN layer and create all six Dense sub-layers.

        The sub-layers are created here and built in ``build``. See the class
        docstring for the parameters.

        :raises ValueError: If ``output_dimension``, ``num_mixtures`` or
            ``intermediate_units`` is not positive, or if
            ``diversity_regularizer_strength`` is negative.
        """
        super().__init__(**kwargs)

        # === Parameter Validation ===
        if output_dimension <= 0:
            raise ValueError(f"output_dimension must be positive, got {output_dimension}")
        if num_mixtures <= 0:
            raise ValueError(f"num_mixtures must be positive, got {num_mixtures}")
        if diversity_regularizer_strength < 0:
            raise ValueError(f"diversity_regularizer_strength must be non-negative, got {diversity_regularizer_strength}")
        if intermediate_units <= 0:
            raise ValueError(f"intermediate_units must be positive, got {intermediate_units}")

        # === Store Configuration ===
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.use_bias = use_bias
        self.diversity_regularizer_strength = diversity_regularizer_strength
        self.intermediate_units = intermediate_units
        self.use_batch_norm = use_batch_norm
        self.intermediate_activation = deserialize_activation(intermediate_activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.min_sigma = min_sigma

        # === CREATE all sub-layers (unbuilt) ===
        # This follows the "Create vs. Build" golden rule.

        # --- Intermediate processing layers ---
        # DECISION plan-2026-08-22T035419-a11304c8/D-200: every Dense below gets
        # its own clone_initializer(...). Do NOT pass the shared instance back
        # in: MEASURED, mdn_mus.kernel == mdn_sigmas.kernel and all three
        # intermediate heads came out bit-identical (max|delta| = 0.0) at init.
        # See decisions.md D-200.
        self.intermediate_mu_dense = keras.layers.Dense(
            self.intermediate_units, use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer), bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='intermediate_mu_dense'
        )
        self.intermediate_sigma_dense = keras.layers.Dense(
            self.intermediate_units, use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer), bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='intermediate_sigma_dense'
        )
        self.intermediate_pi_dense = keras.layers.Dense(
            self.intermediate_units, use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer), bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='intermediate_pi_dense'
        )

        if self.use_batch_norm:
            self.intermediate_mu_bn = keras.layers.BatchNormalization(name='intermediate_mu_bn', center=self.use_bias)
            self.intermediate_sigma_bn = keras.layers.BatchNormalization(name='intermediate_sigma_bn', center=self.use_bias)
            self.intermediate_pi_bn = keras.layers.BatchNormalization(name='intermediate_pi_bn', center=self.use_bias)
        else:
            self.intermediate_mu_bn = None
            self.intermediate_sigma_bn = None
            self.intermediate_pi_bn = None

        # --- Final output layers ---
        self.mdn_mus = keras.layers.Dense(
            self.num_mix * self.output_dim, use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer), bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='mdn_mus'
        )
        self.mdn_sigmas = keras.layers.Dense(
            self.num_mix * self.output_dim, use_bias=self.use_bias,
            activation=lambda x: keras.activations.softplus(x) + self.min_sigma,
            kernel_initializer=clone_initializer(self.kernel_initializer), bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='mdn_sigmas'
        )
        # DECISION plan_2026-06-08_a5f40f4f/D-004: mdn_pi takes NO activation and
        # emits raw logits. Do NOT re-add softplus or softmax here. Every pi
        # consumer (loss_func, sample, get_point_estimate, get_uncertainty,
        # check_component_diversity) applies exactly one softmax or log_softmax
        # to this slice; a softplus here compressed the logits and degraded pi.
        self.mdn_pi = keras.layers.Dense(
            self.num_mix, use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer), bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='mdn_pi'
        )

        logger.info(f"Initialized MDN layer with {num_mixtures} mixtures and {output_dimension}D output")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly.

        The three intermediate Dense layers take the raw input. Everything after
        them takes ``(batch, intermediate_units)``, so the later builds use that
        shape instead of ``input_shape``. Building explicitly keeps the weights
        materialized on a ``.keras`` reload.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        # Build intermediate dense layers, which take the primary input
        self.intermediate_mu_dense.build(input_shape)
        self.intermediate_sigma_dense.build(input_shape)
        self.intermediate_pi_dense.build(input_shape)

        # The subsequent layers operate on the intermediate dimension
        intermediate_shape = (input_shape[0], self.intermediate_units)

        # Build batch norm layers if they exist
        if self.use_batch_norm:
            self.intermediate_mu_bn.build(intermediate_shape)
            self.intermediate_sigma_bn.build(intermediate_shape)
            self.intermediate_pi_bn.build(intermediate_shape)

        # Build the final output layers
        self.mdn_mus.build(intermediate_shape)
        self.mdn_sigmas.build(intermediate_shape)
        self.mdn_pi.build(intermediate_shape)

        logger.debug(f"MDN layer built with input shape: {input_shape}")

        # Always call the parent's build() method at the end (MUST be last)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the three heads and concatenate their outputs.

        The mu, sigma and pi paths are independent and run in any order. Only
        the diversity loss depends on ``training``: it is added when
        ``diversity_regularizer_strength`` is above 0 and ``training`` is
        exactly ``True``. The returned tensor is the same either way.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, forwarded to the sub-layers and
            gating the diversity loss.
        :type training: bool | None
        :return: Mixture parameters packed as ``[mu, sigma, pi]`` on axis -1,
            of shape ``(batch_size, 2 * num_mix * output_dim + num_mix)``. See
            the Output Layout diagram in the class docstring.
        :rtype: keras.KerasTensor
        """
        # === Process MU (Means) Path ===
        mu_intermediate = self.intermediate_mu_dense(inputs, training=training)
        if self.use_batch_norm:
            mu_intermediate = self.intermediate_mu_bn(mu_intermediate, training=training)
        mu_intermediate = keras.activations.get(self.intermediate_activation)(mu_intermediate)
        mu_output = self.mdn_mus(mu_intermediate, training=training)

        # === Process SIGMA (Standard Deviations) Path ===
        sigma_intermediate = self.intermediate_sigma_dense(inputs, training=training)
        if self.use_batch_norm:
            sigma_intermediate = self.intermediate_sigma_bn(sigma_intermediate, training=training)
        sigma_intermediate = keras.activations.get(self.intermediate_activation)(sigma_intermediate)
        sigma_output = self.mdn_sigmas(sigma_intermediate, training=training)

        # === Process PI (Mixture Weights) Path ===
        pi_intermediate = self.intermediate_pi_dense(inputs, training=training)
        if self.use_batch_norm:
            pi_intermediate = self.intermediate_pi_bn(pi_intermediate, training=training)
        pi_intermediate = keras.activations.get(self.intermediate_activation)(pi_intermediate)
        pi_output = self.mdn_pi(pi_intermediate, training=training)

        # === Diversity Regularization ===
        if self.diversity_regularizer_strength > 0.0 and training is True:
            diversity_loss = self._compute_diversity_loss(mu_output)
            self.add_loss(diversity_loss)

        # === Concatenate Output ===
        # ops.concatenate, not keras.layers.concatenate: the layer function
        # creates a new graph node on every call. The order stays
        # [mu, sigma, pi] on axis -1, which is what split_mixture_params and
        # compute_output_shape assume.
        return ops.concatenate(
            [mu_output, sigma_output, pi_output],
            axis=-1
        )

    def _compute_diversity_loss(
        self,
        mu_output: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Penalize component means that sit on top of each other.

        Reshapes the flat means to ``[B, M, D]`` and takes the squared distance
        between every pair of components. ``exp(-distance)`` turns that into a
        penalty: close components score near 1, far ones score near 0. The
        diagonal is zeroed by a mask, so a component is never compared with
        itself.

        The mean runs over the whole ``M`` by ``M`` matrix, zeroed diagonal
        included, so the divisor is ``M * M`` and not the pair count.

        Returns an exact 0.0 when there is only one component, since there is
        no pair to separate.

        :param mu_output: Mean outputs of shape
            ``(batch_size, num_mix * output_dim)``.
        :type mu_output: keras.KerasTensor
        :return: Scalar loss, already scaled by
            ``diversity_regularizer_strength``.
        :rtype: keras.KerasTensor
        """
        if self.num_mix <= 1:
            return ops.cast(0.0, dtype=mu_output.dtype)

        batch_size = ops.shape(mu_output)[0]
        mus = ops.reshape(mu_output, [batch_size, self.num_mix, self.output_dim])
        mus_expanded_1 = ops.expand_dims(mus, axis=2)
        mus_expanded_2 = ops.expand_dims(mus, axis=1)
        pairwise_distances = ops.sum(ops.square(mus_expanded_1 - mus_expanded_2), axis=-1)
        mask = 1.0 - ops.eye(self.num_mix, dtype=pairwise_distances.dtype)
        diversity_loss = ops.mean(ops.exp(-pairwise_distances) * mask)

        return self.diversity_regularizer_strength * diversity_loss

    def _param_offsets(self) -> Tuple[int, int, int]:
        """Return the packed-output layout: ``(mu_end, sigma_end, total_width)``.

        The one place the packed layout is computed. ``compute_output_shape``
        takes the width and ``split_mixture_params`` takes the two slice
        boundaries, so a change to the packing cannot land in only one of them.

        The layout is ``[mu | sigma | pi]``: ``num_mix * output_dim`` means,
        the same number of standard deviations, then ``num_mix`` mixture logits.

        :return: ``(mu_end, sigma_end, total_width)`` along the last axis.
        :rtype: tuple[int, int, int]
        """
        block = self.num_mix * self.output_dim
        mu_end = block
        sigma_end = mu_end + block
        return mu_end, sigma_end, sigma_end + self.num_mix

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        The last axis holds ``2 * output_dim * num_mix + num_mix`` values: the
        means, then the standard deviations, then the mixture logits. Leading
        axes pass through unchanged.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Same shape with the last axis replaced by the packed
            parameter width.
        :rtype: tuple[int | None, ...]
        """
        _, _, output_size = self._param_offsets()
        return tuple(list(input_shape)[:-1] + [output_size])

    def split_mixture_params(
            self,
            y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Split the packed output back into mu, sigma and pi.

        The slice offsets are drawn in the class docstring's Output Layout
        diagram. ``mu`` and ``sigma`` are reshaped to ``[B, M, D]``. ``pi`` is
        left at ``[B, M]`` and still holds raw logits, not probabilities.

        :param y_pred: Packed prediction tensor of shape
            ``(batch_size, 2 * num_mix * output_dim + num_mix)``.
        :type y_pred: keras.KerasTensor
        :return: Tuple of ``(mu, sigma, pi_logits)`` with shapes
            ``[B, M, D]``, ``[B, M, D]`` and ``[B, M]``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        mu_end, sigma_end, _ = self._param_offsets()

        out_mu = y_pred[..., :mu_end]
        out_sigma = y_pred[..., mu_end:sigma_end]
        out_pi = y_pred[..., sigma_end:]

        batch_size = ops.shape(y_pred)[0]
        out_mu = ops.reshape(out_mu, [batch_size, self.num_mix, self.output_dim])
        out_sigma = ops.reshape(out_sigma, [batch_size, self.num_mix, self.output_dim])

        return out_mu, out_sigma, out_pi

    def loss_func(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Negative log-likelihood of the targets under the predicted mixture.

        Computes ``L = -mean_b log(sum_i pi_i N(y | mu_i, sigma_i))``. The whole
        reduction runs in log space, via ``log_softmax`` on the pi logits and
        ``logsumexp`` over the mixture axis. A prob-space product then sum
        underflows at large ``output_dim`` and needs an epsilon clamp; this
        does not.

        Shapes through the computation, with ``B`` the batch, ``M``
        ``num_mixtures`` and ``D`` ``output_dimension``::

            y_true (reshaped)          [B, D]
            out_mu, out_sigma          [B, M, D]
            out_pi                     [B, M]      logits
            log_mix_weights            [B, M]
            y_true_expanded            [B, 1, D]
            log_component              [B, M, D]   before the sum
            log_component              [B, M]      after the sum over D
            log_prob                   [B]
            loss                       scalar

        Sigma is floored at ``min_sigma`` again here. ``mdn_sigmas`` already
        applies that floor, but this method also accepts caller-supplied raw
        parameters.

        :param y_true: Ground truth targets. Reshaped to ``[-1, output_dim]``,
            so any leading-axis layout is accepted.
        :type y_true: keras.KerasTensor
        :param y_pred: Packed prediction parameters from ``call``.
        :type y_pred: keras.KerasTensor
        :return: Scalar loss, averaged over the batch.
        :rtype: keras.KerasTensor
        """
        y_true = ops.reshape(y_true, [-1, self.output_dim])
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

        # Floor sigma. The sigma Dense already applies softplus + min_sigma;
        # this covers caller-supplied raw params.
        out_sigma = ops.maximum(out_sigma, self.min_sigma)

        # Everything below stays in log space. out_pi is treated as logits.
        # See the shape chain in this method's docstring.
        log_mix_weights = keras.activations.log_softmax(out_pi, axis=-1)

        y_true_expanded = ops.expand_dims(y_true, 1)

        # log N(y|mu,sigma) = -0.5*log(2pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
        log_2pi = ops.log(ops.cast(2.0 * np.pi, out_mu.dtype))
        z = (y_true_expanded - out_mu) / out_sigma
        log_component = -0.5 * log_2pi - ops.log(out_sigma) - 0.5 * ops.square(z)

        # Sum over the output dimensions. The dimensions are independent
        # Gaussians, so their log-densities add.
        log_component = ops.sum(log_component, axis=-1)

        # Add the log weights, then reduce the mixture axis with logsumexp.
        log_prob = ops.logsumexp(log_mix_weights + log_component, axis=-1)
        loss = -ops.mean(log_prob)

        return loss

    def sample(
            self,
            y_pred: keras.KerasTensor,
            temperature: float = 1.0,
            seed: Optional[int] = None
    ) -> keras.KerasTensor:
        """Draw one sample per row by ancestral sampling.

        Two steps. First pick a component: the pi logits go through a softmax
        and a Gumbel-max draw selects one index per row. Then draw from that
        component's Gaussian, ``mu + sigma * normal()``. The selection is done
        with a one-hot mask and a sum, so it stays a tensor op.

        ``temperature`` divides the pi logits before the softmax. Below 1.0 the
        draw concentrates on the dominant component; above 1.0 it spreads out.
        It scales which component is picked, not how wide that component is.

        The output shape does not depend on ``temperature`` or ``seed``. Both
        branches return ``(batch_size, output_dim)``.

        :param y_pred: Packed prediction parameters from ``call``.
        :type y_pred: keras.KerasTensor
        :param temperature: Divisor applied to the mixture logits before the
            softmax. Defaults to 1.0, which leaves them alone.
        :type temperature: float
        :param seed: Seed for reproducible sampling. Both draws are derived
            from it: the Gumbel draw uses ``seed``, the Gaussian draw uses
            ``seed + 1``. Defaults to "None", which is nondeterministic.
        :type seed: int | None
        :return: Sampled values of shape ``(batch_size, output_dim)``.
        :rtype: keras.KerasTensor
        """
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)
        out_sigma = ops.maximum(out_sigma, self.min_sigma)

        if temperature != 1.0:
            out_pi = out_pi / temperature

        # DECISION plan_2026-06-09_be55db55/D-004: `seed` must reach BOTH
        # keras.random draws. Do NOT drop the parameter or fall back to a global
        # RNG: MDNModel.sample passes `seed + i` per sample, and that value used
        # to be computed and discarded, which made `seed=` a silent no-op. The
        # Gaussian draw uses `seed + 1` so it does not alias the Gumbel stream.
        pi_seed = seed
        normal_seed = None if seed is None else seed + 1

        mix_weights = keras.activations.softmax(out_pi, axis=-1)
        gumbel_noise = -ops.log(-ops.log(
            keras.random.uniform(ops.shape(out_pi), seed=pi_seed)))
        selected_logits = ops.log(mix_weights + keras.config.epsilon()) + gumbel_noise
        selected_components = ops.argmax(selected_logits, axis=-1)

        one_hot = ops.one_hot(selected_components, num_classes=self.num_mix)
        one_hot_expanded = ops.expand_dims(one_hot, -1)

        selected_mu = ops.sum(out_mu * one_hot_expanded, axis=1)
        selected_sigma = ops.sum(out_sigma * one_hot_expanded, axis=1)

        epsilon = keras.random.normal(ops.shape(selected_mu), seed=normal_seed)
        samples = selected_mu + selected_sigma * epsilon

        return samples

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        Every constructor argument is round-tripped. The activation goes
        through ``serialize_activation``, so a plain callable survives a
        ``.keras`` save and reload.

        :return: Configuration dictionary accepted by the constructor.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "use_bias": self.use_bias,
            "diversity_regularizer_strength": self.diversity_regularizer_strength,
            "intermediate_units": self.intermediate_units,
            "use_batch_norm": self.use_batch_norm,
            "intermediate_activation": serialize_activation(self.intermediate_activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "min_sigma": self.min_sigma,
        })
        return config


# ---------------------------------------------------------------------
# Utility functions for analysing a trained MDN.
# Each one runs model.predict and then unpacks the result with the layer's
# own split_mixture_params, so none of them assumes an offset of its own.
# ---------------------------------------------------------------------


def get_point_estimate(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> np.ndarray:
    """Compute the mixture mean ``E[y|x] = sum_i pi_i(x) mu_i(x)``.

    Softmaxes the pi logits, weights the component means by them and sums over
    the mixture axis.

    This is the mean of a multi-modal distribution, so it can land in a region
    the model considers unlikely. With two well-separated modes it returns the
    midpoint between them. Use ``MDNLayer.sample`` when a plausible value
    matters more than the average.

    :param model: Trained model whose output is an ``MDNLayer`` output.
    :type model: keras.Model
    :param x_data: Input data to predict on.
    :type x_data: np.ndarray
    :param mdn_layer: The ``MDNLayer`` instance from that model, used to unpack
        the prediction.
    :type mdn_layer: MDNLayer
    :return: Point estimates of shape ``(batch_size, output_dim)``.
    :rtype: np.ndarray
    """
    y_pred = model.predict(x_data)
    mu, _, pi_logits = mdn_layer.split_mixture_params(y_pred)
    pi = keras.activations.softmax(pi_logits, axis=-1)

    mu_np = ops.convert_to_numpy(mu)
    pi_np = ops.convert_to_numpy(pi)

    pi_expanded = np.expand_dims(pi_np, axis=-1)
    weighted_mu = mu_np * pi_expanded
    point_estimates = np.sum(weighted_mu, axis=1)

    return point_estimates


def get_uncertainty(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer,
    point_estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the predictive variance into its two parts.

    The law of total variance splits it in two::

        aleatoric = sum_i pi_i * sigma_i^2
        epistemic = sum_i pi_i * (mu_i - mean)^2
        total     = aleatoric + epistemic

    The aleatoric term is the average width of the components: noise the model
    believes is in the data. The epistemic term is how far apart the components
    sit: disagreement between the modes. Only the total and the aleatoric part
    are returned; subtract them to get the epistemic part.

    ``point_estimates`` must come from ``get_point_estimate`` on the SAME
    ``x_data``. Passing estimates from other inputs inflates the epistemic term
    without raising anything.

    :param model: Trained model whose output is an ``MDNLayer`` output.
    :type model: keras.Model
    :param x_data: Input data to predict on.
    :type x_data: np.ndarray
    :param mdn_layer: The ``MDNLayer`` instance from that model, used to unpack
        the prediction.
    :type mdn_layer: MDNLayer
    :param point_estimates: Mixture means for the same inputs, of shape
        ``(batch_size, output_dim)``.
    :type point_estimates: np.ndarray
    :return: Tuple of ``(total_variance, aleatoric_variance)``, each of shape
        ``(batch_size, output_dim)``.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    y_pred = model.predict(x_data)
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    mu_np = ops.convert_to_numpy(mu)
    sigma_np = ops.convert_to_numpy(sigma)
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    pi_expanded = np.expand_dims(pi_np, axis=-1)
    point_expanded = np.expand_dims(point_estimates, axis=1)

    # Aleatoric part: the weighted average of the component variances.
    aleatoric_variance = np.sum(pi_expanded * sigma_np ** 2, axis=1)

    # Epistemic part: the weighted spread of the component means around the
    # mixture mean.
    squared_diff = (mu_np - point_expanded) ** 2
    epistemic_variance = np.sum(pi_expanded * squared_diff, axis=1)

    total_variance = aleatoric_variance + epistemic_variance

    return total_variance, aleatoric_variance


def get_prediction_intervals(
    point_estimates: np.ndarray,
    total_variance: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Build symmetric Gaussian intervals ``mu +/- z * sigma``.

    Takes the z-score for ``confidence_level`` from ``scipy.stats.norm`` and
    applies it to the square root of ``total_variance``.

    The interval is Gaussian, but the predicted distribution is a mixture and
    generally is not. On a multi-modal prediction this interval is centred on a
    point between the modes and can cover a region the model rates as unlikely.
    Treat it as a rough band, not as a calibrated quantile.

    :param point_estimates: Interval centres, of shape
        ``(batch_size, output_dim)``.
    :type point_estimates: np.ndarray
    :param total_variance: Total predictive variance for the same inputs.
    :type total_variance: np.ndarray
    :param confidence_level: Two-sided coverage, between 0 and 1. Defaults to
        0.95.
    :type confidence_level: float
    :return: Tuple of ``(lower_bound, upper_bound)``, each the shape of
        ``point_estimates``.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ImportError: If ``scipy`` is not installed.
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("`scipy` is required to calculate prediction intervals. Please install it with `pip install scipy`.")

    alpha = 1.0 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    std_dev = np.sqrt(total_variance)

    lower_bound = point_estimates - z_score * std_dev
    upper_bound = point_estimates + z_score * std_dev

    return lower_bound, upper_bound


def check_component_diversity(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> Dict[str, Any]:
    """Report how far apart the mixture components are.

    Measures the Euclidean distance between the means of every distinct
    component pair, then summarises those distances along with the sigmas and
    the mixture weights. A mean separation near zero with one weight near 1 is
    the signature of component collapse: the mixture has become one Gaussian.

    With ``num_mix == 1`` there is no pair, and the separation statistics are
    reported as 0.0 rather than raising.

    Returned keys:

    - ``mean_component_separation`` -- scalar, average pairwise distance.
    - ``std_component_separation`` -- scalar, spread of those distances.
    - ``mean_sigma_values`` -- scalar, average sigma over every component and
      output dimension.
    - ``mean_mixture_weights`` -- array of shape ``(num_mix,)``, the average
      weight per component.
    - ``std_mixture_weights`` -- array of shape ``(num_mix,)``.

    :param model: Trained model whose output is an ``MDNLayer`` output.
    :type model: keras.Model
    :param x_data: Sample inputs to analyse.
    :type x_data: np.ndarray
    :param mdn_layer: The ``MDNLayer`` instance from that model, used to unpack
        the prediction and to read ``num_mix``.
    :type mdn_layer: MDNLayer
    :return: Diversity metrics keyed as listed above.
    :rtype: dict[str, Any]
    """
    y_pred = model.predict(x_data)
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    mu_np = ops.convert_to_numpy(mu)
    sigma_np = ops.convert_to_numpy(sigma)
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    num_mix = mdn_layer.num_mix
    component_distances = []
    if num_mix > 1:
        for i in range(num_mix):
            for j in range(i + 1, num_mix):
                distances = np.linalg.norm(mu_np[:, i, :] - mu_np[:, j, :], axis=-1)
                component_distances.append(distances)
        component_distances = np.array(component_distances)
    else:
        component_distances = np.array([0.0])


    return {
        "mean_component_separation": np.mean(component_distances),
        "std_component_separation": np.std(component_distances),
        "mean_sigma_values": np.mean(sigma_np),
        "mean_mixture_weights": np.mean(pi_np, axis=0),
        "std_mixture_weights": np.std(pi_np, axis=0)
    }