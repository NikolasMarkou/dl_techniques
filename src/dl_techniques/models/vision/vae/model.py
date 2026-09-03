"""
Residual convolutional variational autoencoder with selectable Gaussian,
hypersphere or von Mises-Fisher latent geometry.

`VAE` optimizes the evidence lower bound
`L = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))` using the reparameterization
trick, so gradients flow through the sampled latent. `sampling_type` switches
the latent geometry, and each choice changes three things together: the
second encoder head, the sampler, and the KL term. `"gaussian"` is the
standard diagonal posterior; `"hypersphere"` reduces the head to one radius
log-variance and drops the direction term (an implicit uniform-sphere prior,
not a full S-VAE); `"vmf"` uses a strictly positive concentration `kappa` and
an analytic von Mises-Fisher KL. The `z_log_var` output slot is reused across
all three and holds a different quantity in each.

Reconstruction is binary crossentropy on `[0, 1]`-range inputs, reduced by a
mean over pixels, while the Gaussian KL is a sum over latents. This means
`kl_loss_weight` is not the literature's `beta` — the actual value optimized
is `beta = kl_loss_weight * prod(input_shape)`, exposed as `effective_kl_beta`.
The vmf mode disables XLA on every compile path (direct `compile()`, the
factories, and reload) because its sampler is not XLA-compatible.

References:
    - Kingma & Welling, 2013. Auto-Encoding Variational Bayes.
      (https://arxiv.org/abs/1312.6114)
    - Rezende et al., 2014. Stochastic Backpropagation and Approximate Inference
      in Deep Generative Models. (https://arxiv.org/abs/1401.4082)
    - Higgins et al., 2017. beta-VAE: Learning Basic Visual Concepts with a
      Constrained Variational Framework. ICLR 2017.
    - Davidson et al., 2018. Hyperspherical Variational Auto-Encoders.
      (https://arxiv.org/abs/1804.00891)
    - Bowman et al., 2015. Generating Sentences from a Continuous Space.
      (https://arxiv.org/abs/1511.06349)
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
"""

import keras
import tensorflow as tf
from keras import layers, ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.sampling import create_sampling_layer, vmf_kl_divergence
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Supported VAE sampling modes
# ---------------------------------------------------------------------

VALID_SAMPLING_TYPES = (
    "gaussian",
    "hypersphere",
    "vmf",
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.vae.model")
class VAE(keras.Model):
    """Residual convolutional variational autoencoder.

    Architecture:

    .. code-block:: text

        input [H, W, C]
              │
        ┌─────▼─────┐
        │ stem conv  │
        └─────┬─────┘
              ▼
        ┌───────────────────────┐
        │ encoder stage x depths │  downsample + residual blocks
        └─────────┬─────────────┘
                   ▼
        ┌────────────────────┐
        │ global avg pool     │
        └─────────┬──────────┘
                   ▼
        ┌────────────────────┐      ┌────────────────────┐
        │ z_mean [B, latent]  │      │ z_log_var head      │  shape/meaning
        └─────────┬──────────┘      └─────────┬──────────┘  depends on mode
                   └───────────┬───────────────┘
                               ▼
                    ┌────────────────────┐
                    │ sampling (reparam)  │  gaussian / hypersphere / vmf
                    └─────────┬──────────┘
                               ▼
                    ┌────────────────────┐
                    │ decoder projection  │
                    └─────────┬──────────┘
                               ▼
        ┌───────────────────────┐
        │ decoder stage x depths │  upsample + residual blocks
        └─────────┬─────────────┘
                   ▼
              reconstruction [H, W, C]

    Sampling modes:

    .. code-block:: text

        mode          z_log_var head        KL term
        gaussian      [B, latent] logvar     sum over latents, N(0, I)
        hypersphere   [B, 1] radius logvar   1-D radius KL, no direction term
        vmf           [B, 1] kappa (>0)      analytic vMF-to-uniform KL

    Variants (``MODEL_VARIANTS``, used by :meth:`from_variant`):

    .. code-block:: text

        name     depths  steps  filters              latent  kl_weight
        micro    2       1      [16, 32]              32     0.01
        small    2       1      [32, 64]              64     0.01
        medium   3       1      [32, 64, 128]         128    0.005
        large    3       2      [64, 128, 256]        256    0.005
        xlarge   4       2      [64, 128, 256, 512]   512    0.001

    :param latent_dim: Dimensionality of the latent space.
    :type latent_dim: int
    :param input_shape: Shape of input images, ``(H, W, C)``.
    :type input_shape: Tuple[int, int, int]
    :param depths: Number of stages in the encoder/decoder.
    :type depths: int
    :param steps_per_depth: Residual blocks per stage.
    :type steps_per_depth: int
    :param filters: Filter count for each stage.
    :type filters: Optional[List[int]]
    :param kl_loss_weight: Weight for the KL term. Not the literature `beta`;
        see the module docstring and :attr:`effective_kl_beta`.
    :type kl_loss_weight: float
    :param sampling_type: One of ``"gaussian"``, ``"hypersphere"``, ``"vmf"``.
        ``"hypersphere_faithful"`` is a deprecated alias of ``"hypersphere"``;
        ``"hypersphere_controlled"`` was removed and now raises ``ValueError``.
    :type sampling_type: str
    :param kernel_initializer: Weight initializer for convolutional layers.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for convolutional weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_batch_norm: Whether to use batch normalization.
    :type use_batch_norm: bool
    :param use_bias: Whether to use bias terms.
    :type use_bias: bool
    :param dropout_rate: Dropout rate in the residual blocks.
    :type dropout_rate: float
    :param activation: Activation used throughout the encoder/decoder.
    :type activation: str
    :param final_activation: Activation on the reconstruction layer.
    :type final_activation: str
    :param name: Model name.
    :type name: Optional[str]
    :param kwargs: Forwarded to ``keras.Model``.

    :Example:

    >>> model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=64)
    >>> model = VAE(
    ...     latent_dim=128,
    ...     input_shape=(64, 64, 3),
    ...     depths=3,
    ...     filters=[32, 64, 128],
    ...     kl_loss_weight=0.01,
    ... )
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {
            "depths": 2,
            "steps_per_depth": 1,
            "filters": [16, 32],
            "default_latent_dim": 32,
            "kl_loss_weight": 0.01,
        },
        "small": {
            "depths": 2,
            "steps_per_depth": 1,
            "filters": [32, 64],
            "default_latent_dim": 64,
            "kl_loss_weight": 0.01,
        },
        "medium": {
            "depths": 3,
            "steps_per_depth": 1,
            "filters": [32, 64, 128],
            "default_latent_dim": 128,
            "kl_loss_weight": 0.005,
        },
        "large": {
            "depths": 3,
            "steps_per_depth": 2,
            "filters": [64, 128, 256],
            "default_latent_dim": 256,
            "kl_loss_weight": 0.005,
        },
        "xlarge": {
            "depths": 4,
            "steps_per_depth": 2,
            "filters": [64, 128, 256, 512],
            "default_latent_dim": 512,
            "kl_loss_weight": 0.001,
        },
    }

    # Architecture constants
    DEFAULT_ACTIVATION = "leaky_relu"
    DEFAULT_FINAL_ACTIVATION = "sigmoid"
    DEFAULT_INITIALIZER = "he_normal"

    def __init__(
        self,
        latent_dim: int,
        input_shape: Tuple[int, int, int],
        depths: int = 2,
        steps_per_depth: int = 1,
        filters: Optional[List[int]] = None,
        kl_loss_weight: float = 0.01,
        sampling_type: str = "gaussian",
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = True,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "leaky_relu",
        final_activation: str = "sigmoid",
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        # Validate inputs
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if depths <= 0:
            raise ValueError(f"depths must be positive, got {depths}")
        if steps_per_depth <= 0:
            raise ValueError(
                f"steps_per_depth must be positive, got {steps_per_depth}"
            )
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # Back-compat: map the renamed legacy value through BEFORE validation /
        # storage so from_config()/load_model() on a legacy checkpoint (whose
        # stored sampling_type is "hypersphere_faithful") deserializes cleanly
        # and self.sampling_type reports the current name "hypersphere".
        if sampling_type == "hypersphere_faithful":
            logger.warning(
                "sampling_type 'hypersphere_faithful' is deprecated; "
                "use 'hypersphere'."
            )
            sampling_type = "hypersphere"
        if sampling_type == "hypersphere_controlled":
            raise ValueError(
                "sampling_type 'hypersphere_controlled' was removed (dropped "
                "negative-control arm). Use 'gaussian' or 'hypersphere'."
            )
        if sampling_type not in VALID_SAMPLING_TYPES:
            raise ValueError(
                f"sampling_type must be one of {list(VALID_SAMPLING_TYPES)}, "
                f"got {sampling_type!r}"
            )

        # Set default filters if not provided
        if filters is None:
            filters = [32 * (2**i) for i in range(depths)]

        if len(filters) != depths:
            raise ValueError(
                f"Filters array length {len(filters)} must equal depths {depths}"
            )

        # Store configuration
        self.latent_dim = latent_dim
        self._input_shape = input_shape
        self.depths = depths
        self.steps_per_depth = steps_per_depth
        self.filters = filters
        self.kl_loss_weight = kl_loss_weight
        self.sampling_type = sampling_type
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)
        self.final_activation = deserialize_activation(final_activation)

        # Validate input dimensions
        height, width, channels = input_shape
        if height < 8 or width < 8:
            raise ValueError(
                f"Input dimensions must be at least 8x8, got {height}x{width}"
            )

        # Initialize metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        # Build the model using functional API
        inputs = keras.Input(shape=input_shape, name="input")
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, name=name or "vae", **kwargs)

        # Schedulable KL weight, read by train_step/test_step in place of the
        # python float self.kl_loss_weight; a warmup callback can ramp it from
        # 0 over the first epochs. Default equals kl_loss_weight.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-011: dtype="float32" alone does
        # NOT stop Keras autocasting this scalar to compute_dtype in call/train_step.
        # autocast=False is required or mixed_float16 raises on the loss add. See decisions.md.
        self.kl_weight = self.add_weight(
            name="kl_weight",
            shape=(),
            initializer=keras.initializers.Constant(float(kl_loss_weight)),
            trainable=False,
            dtype="float32",
            autocast=False,
        )

        # Create a reusable decoder model from the main graph. This allows
        # self.decode() to reuse the trained decoder weights.
        decoder_input = self.get_layer("vae_sampling").output
        decoder_output = self.output["reconstruction"]
        self.decoder = keras.Model(decoder_input, decoder_output, name="decoder")

        logger.info(
            f"Created VAE model for input {input_shape} with latent_dim={latent_dim}, "
            f"depths={depths}, filters={filters}"
        )

    def compile(self, *args, **kwargs):
        """Compile the model, disabling XLA for the vmf sampler on every path.

        :param args: Forwarded to ``keras.Model.compile``.
        :param kwargs: Forwarded to ``keras.Model.compile``; ``jit_compile``
            is overridden to ``False`` when ``sampling_type == "vmf"``.
        """
        # DECISION plan_2026-06-04_6196678d/D-009: vMF's sampler has no XLA-GPU
        # kernel; force jit_compile=False on every compile path (including
        # load_model recompile), overriding any caller-passed value. See decisions.md.
        if getattr(self, "sampling_type", None) == "vmf":
            kwargs["jit_compile"] = False
        return super().compile(*args, **kwargs)

    def compile_from_config(self, config):
        """Recompile from a saved config, routed through :meth:`compile`.

        Overriding ``compile()`` makes Keras call this method on
        ``load_model()`` instead of skipping recompilation, so the vmf
        ``jit_compile=False`` override survives a reload.

        :param config: Serialized compile config, as produced by Keras.
        :return: This model, recompiled.
        :rtype: VAE
        """
        config = keras.saving.deserialize_keras_object(config)
        self.compile(**config)
        # DECISION plan-2026-08-22T035419-a11304c8/D-014: these two lines are Keras'
        # own Trainer.compile_from_config tail; omitting them silently drops the
        # saved optimizer state on reload (measured: 122 vars -> 2). See decisions.md.
        if hasattr(self, "optimizer") and self.built:
            self.optimizer.build(self.trainable_variables)
        return self

    def _build_model(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """Build the complete VAE model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Dictionary containing all VAE outputs
        """
        # Build encoder
        z_mean, z_log_var = self._build_encoder(inputs)

        # DECISION plan_2026-06-04_d4ef81f1/D-004: the sampler layer is named
        # "vae_sampling" in EVERY mode. self.decoder is extracted by this exact
        # layer name (see __init__: self.get_layer("vae_sampling").output), and
        # HypersphereSampling emits [B, latent_dim] just like Sampling, so the
        # extraction is shape-safe for both modes. Do NOT rename the sampler
        # per-mode or branch the decoder-extraction line: that would add surface
        # to the one code path the whole decode()/sample() API depends on. See
        # decisions.md D-004.
        if self.sampling_type == "gaussian":
            # Baseline diagonal-Gaussian reparameterization over [B, D].
            z = create_sampling_layer("gaussian", name="vae_sampling")(
                [z_mean, z_log_var]
            )
        elif self.sampling_type == "vmf":
            # z_log_var is the strictly-positive [B, 1] concentration kappa head
            # (softplus). VMFSampling L2-normalizes z_mean internally onto the
            # unit sphere -- do NOT pre-normalize z_mean here.
            z = create_sampling_layer("vmf", name="vae_sampling")(
                [z_mean, z_log_var]
            )
        else:  # hypersphere
            # z_log_var is already the dedicated [B, 1] radius log-variance head.
            z = create_sampling_layer("hypersphere", name="vae_sampling")(
                [z_mean, z_log_var]
            )

        # Build decoder
        reconstruction = self._build_decoder(z)

        # NOTE: for sampling_type == "vmf" the "z_log_var" slot carries the
        # strictly-positive concentration kappa[B, 1] (NOT a log-variance);
        # for "hypersphere" it is the radius log-variance[B, 1]; for "gaussian"
        # it is the diagonal log-variance[B, latent_dim]. The slot is reused
        # (shape-only contract; see I7 / create_vae assertion).
        return {
            "z": z,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction,
        }

    def _build_encoder(
        self, inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Build the encoder network with ResNet blocks.

        Args:
            inputs: Input tensor

        Returns:
            Tuple of (z_mean, z_log_var) tensors
        """
        x = inputs

        # Initial conv layer
        x = layers.Conv2D(
            filters=self.filters[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_stem_conv",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(center=self.use_bias, name="encoder_stem_bn")(
                x
            )
        x = layers.Activation(self.activation, name="encoder_stem_activation")(x)

        # Encoder blocks with downsampling
        for depth in range(self.depths):
            x = self._build_encoder_stage(x, depth)

        # Global pooling and latent projection
        x = layers.GlobalAveragePooling2D(name="encoder_global_pool")(x)

        # Latent space projection
        z_mean = layers.Dense(
            units=self.latent_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer="zeros",
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_z_mean",
        )(x)

        # The second latent head emits the full latent_dim for the gaussian mode
        # (diagonal-Gaussian posterior), but a single scalar per sample [B, 1]
        # for hypersphere (radius-shell log-variance) and vmf (vMF concentration
        # kappa). For gaussian/hypersphere the head is a raw (possibly negative)
        # log-variance with bias Constant(-2.0) so the shell / variance starts
        # thin. For vmf the head must be STRICTLY POSITIVE (kappa > 0), so a
        # softplus is applied. The bias is initialized HIGH (Constant(12.0)) and
        # the kernel to zeros so kappa STARTS at softplus(12) ~= 12 (an
        # informative concentration) and is PREDICTABLE at init (not swamped by
        # W.h). The same kappa tensor flows BOTH into VMFSampling AND into the
        # vmf KL (vmf_kl_divergence) -- they must agree, so this single head is
        # the sole source of kappa. The head still learns per-sample kappa after
        # init (the zeros kernel only fixes the t=0 value).
        if self.sampling_type == "vmf":
            # DECISION plan_2026-06-04_6196678d/D-007: higher init kappa (~12) +
            # zeros kernel breaks the posterior-collapse trap (z informative from
            # step 0); see decisions.md D-006/D-007. Do NOT revert to
            # bias="zeros" (softplus(0)~=0.69 -> uniform latent -> decoder
            # ignores z -> kappa driven to 0 -> recon stalls at the data mean).
            kappa_raw = layers.Dense(
                units=1,
                use_bias=self.use_bias,
                kernel_initializer="zeros",
                bias_initializer=keras.initializers.Constant(12.0),
                kernel_regularizer=self.kernel_regularizer,
                name="encoder_kappa",
            )(x)
            # Softplus -> strictly positive concentration kappa[B, 1]. This is the
            # value carried in the "z_log_var" output-dict slot for vmf (it is the
            # concentration kappa, NOT a log-variance; see get_config / I7).
            z_log_var = layers.Activation(
                "softplus", name="encoder_kappa_softplus"
            )(kappa_raw)
            return z_mean, z_log_var

        log_var_units = 1 if self.sampling_type == "hypersphere" else self.latent_dim
        log_var_name = (
            "encoder_radius_log_var"
            if self.sampling_type == "hypersphere"
            else "encoder_z_log_var"
        )

        z_log_var = layers.Dense(
            units=log_var_units,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=keras.initializers.Constant(
                -2.0
            ),  # Initialize to small variance
            kernel_regularizer=self.kernel_regularizer,
            name=log_var_name,
        )(x)

        return z_mean, z_log_var

    def _build_encoder_stage(
        self, x: keras.KerasTensor, stage_idx: int
    ) -> keras.KerasTensor:
        """Build a single encoder stage with downsampling and residual blocks.

        Args:
            x: Input tensor
            stage_idx: Index of the stage

        Returns:
            Output tensor from the stage
        """
        num_filters = self.filters[stage_idx]

        # Downsampling layer
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=2,
            strides=2,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"encoder_downsample_{stage_idx}",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(
                center=self.use_bias, name=f"encoder_downsample_bn_{stage_idx}"
            )(x)
        x = layers.Activation(
            self.activation, name=f"encoder_downsample_activation_{stage_idx}"
        )(x)

        # Residual blocks
        for step in range(self.steps_per_depth):
            x = self._build_residual_block(
                x, num_filters, f"encoder_{stage_idx}_{step}"
            )

        return x

    def _build_residual_block(
        self, x: keras.KerasTensor, filters: int, prefix: str
    ) -> keras.KerasTensor:
        """Build a residual block.

        Args:
            x: Input tensor
            filters: Number of filters
            prefix: Name prefix for layers

        Returns:
            Output tensor with residual connection
        """
        residual = x

        # First convolution
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{prefix}_conv_1",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(center=self.use_bias, name=f"{prefix}_bn_1")(
                x
            )
        x = layers.Activation(self.activation, name=f"{prefix}_activation_1")(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name=f"{prefix}_dropout")(x)

        # Second convolution
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{prefix}_conv_2",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(center=self.use_bias, name=f"{prefix}_bn_2")(
                x
            )

        # Residual connection
        x = layers.Add(name=f"{prefix}_add")([x, residual])
        x = layers.Activation(self.activation, name=f"{prefix}_activation_final")(x)

        return x

    def _build_decoder(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Build the decoder network with ResNet blocks.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed image tensor
        """
        # Calculate feature map size after all downsampling
        height, width, channels = self._input_shape
        feature_height = height // (2**self.depths)
        feature_width = width // (2**self.depths)

        # Ensure minimum size
        feature_height = max(feature_height, 1)
        feature_width = max(feature_width, 1)

        # Project latent to feature map
        x = layers.Dense(
            units=feature_height * feature_width * self.filters[-1],
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="decoder_projection",
        )(z)

        x = layers.Reshape(
            (feature_height, feature_width, self.filters[-1]), name="decoder_reshape"
        )(x)

        # Decoder stages with upsampling
        for depth in range(self.depths - 1, -1, -1):
            x = self._build_decoder_stage(x, depth)

        # Final output layer
        x = layers.Conv2D(
            filters=channels,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=self.final_activation,
            use_bias=self.use_bias,
            kernel_regularizer=keras.regularizers.L1(1e-6),
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer="zeros",
            name="decoder_output",
        )(x)

        # Ensure exact shape matching
        if x.shape[1:] != self._input_shape:
            # Resize to exact input shape if needed
            target_height, target_width = self._input_shape[:2]
            x = layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation="bilinear",
                name="decoder_resize",
            )(x)

        return x

    def _build_decoder_stage(
        self, x: keras.KerasTensor, stage_idx: int
    ) -> keras.KerasTensor:
        """Build a single decoder stage with upsampling and residual blocks.

        Args:
            x: Input tensor
            stage_idx: Index of the stage

        Returns:
            Output tensor from the stage
        """
        num_filters = self.filters[stage_idx]

        # Upsampling layer
        x = layers.UpSampling2D(
            size=(2, 2), interpolation="nearest", name=f"decoder_upsample_{stage_idx}"
        )(x)

        # Convolution after upsampling
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"decoder_conv_{stage_idx}",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(
                center=self.use_bias, name=f"decoder_bn_{stage_idx}"
            )(x)
        x = layers.Activation(self.activation, name=f"decoder_activation_{stage_idx}")(
            x
        )

        # Residual blocks
        for step in range(self.steps_per_depth):
            x = self._build_residual_block(
                x, num_filters, f"decoder_{stage_idx}_{step}"
            )

        return x

    @classmethod
    def from_variant(
        cls,
        variant: str,
        input_shape: Tuple[int, int, int],
        latent_dim: Optional[int] = None,
        **kwargs,
    ) -> "VAE":
        """Create a VAE model from a predefined variant.

        Args:
            variant: String, one of "micro", "small", "medium", "large", "xlarge"
            input_shape: Tuple, input image shape (H, W, C)
            latent_dim: Integer, latent dimension. If None, uses variant default
            **kwargs: Additional arguments passed to the constructor

        Returns:
            VAE model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # MNIST VAE
            >>> model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=64)
            >>> # CIFAR-10 VAE
            >>> model = VAE.from_variant("medium", input_shape=(32, 32, 3), latent_dim=128)
            >>> # High-resolution VAE
            >>> model = VAE.from_variant("large", input_shape=(128, 128, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        # Use provided latent_dim or variant default
        if latent_dim is None:
            latent_dim = config["default_latent_dim"]

        logger.info(f"Creating VAE-{variant.upper()} model")
        logger.info(f"Input shape: {input_shape}, Latent dim: {latent_dim}")

        # Let an explicit caller-supplied kl_loss_weight (e.g. train_vae's
        # --kl-loss-weight for the vMF beta-sweep) override the variant default;
        # otherwise fall back to the variant's configured weight.
        # Variant values are DEFAULTS — an explicit caller kwarg (e.g. train_vae's
        # --depths / --steps-per-depth / --filters) overrides them. depths and
        # filters must stay consistent (filters length == depths), so they move
        # together: fall back to the variant pair only when the caller overrides
        # NEITHER. If the caller passes depths alone, the constructor
        # auto-generates filters = [32*2**i for i in range(depths)]; if filters
        # alone, depths is derived from its length. (DECISION plan_2026-06-05_56b39171/D-004)
        kwargs.setdefault("kl_loss_weight", config["kl_loss_weight"])
        kwargs.setdefault("steps_per_depth", config["steps_per_depth"])
        if "depths" not in kwargs and "filters" not in kwargs:
            kwargs["depths"] = config["depths"]
            kwargs["filters"] = config["filters"]
        elif "filters" in kwargs and "depths" not in kwargs:
            kwargs["depths"] = len(kwargs["filters"])

        return cls(
            latent_dim=latent_dim,
            input_shape=input_shape,
            **kwargs,
        )

    @property
    def effective_kl_beta(self) -> float:
        """The beta this model actually optimizes, in sum-over-pixels ELBO units.

        `kl_loss_weight` is NOT the `beta` of the standard ELBO. This model's
        reconstruction term is a **mean** over the `prod(input_shape)` pixels while
        its Gaussian KL is a **sum** over the latent axis, so dividing the standard
        `sum_pixels(BCE) + beta * sum_latents(KL)` through by the pixel count shows
        that `kl_loss_weight == beta / prod(input_shape)`. The `beta` a reader of
        the literature means is therefore this property, not the constructor
        argument -- and because the pixel count is in it, the SAME nominal
        `kl_loss_weight` is a different regularization strength at a different
        input resolution.

        Returns:
            `kl_loss_weight * prod(input_shape)`.
        """
        pixels = 1
        for dim in self._input_shape:
            pixels *= int(dim)
        return float(self.kl_loss_weight) * pixels

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return metrics tracked by the model."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(
        self, inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode inputs to latent parameters.

        Args:
            inputs: Input tensor to encode

        Returns:
            Tuple of (z_mean, z_log_var) tensors
        """
        outputs = self(inputs, training=False)
        return outputs["z_mean"], outputs["z_log_var"]

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode latent samples to reconstructions.

        Args:
            z: Latent tensor to decode

        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)

    def _sample_prior(self, num_samples: int) -> keras.KerasTensor:
        """Draw latent samples from this mode's TRUE prior.

        Args:
            num_samples: Number of latent vectors to draw

        Returns:
            Latent tensor of shape ``(num_samples, latent_dim)``
        """
        # DECISION plan_2026-06-04_7ff8ea8b/D-001: hypersphere modes were trained
        # with a uniform-on-sphere latent of the layer radius, so their prior is
        # NOT N(0, I). Drawing N(0, I) here (the old behavior) decodes the wrong
        # prior and makes the contribution look broken. Branch on sampling_type:
        # gaussian -> N(0, I); hypersphere_* -> Marsaglia uniform-on-sphere * radius.
        if self.sampling_type == "gaussian":
            return keras.random.normal(shape=(num_samples, self.latent_dim))

        if self.sampling_type == "vmf":
            # The vMF prior (kappa = 0) IS exactly the uniform distribution on the
            # unit sphere S^{D-1}. Marsaglia draw at radius 1.0; VMFSampling has no
            # .radius attribute (vMF is unit-sphere by definition), so do NOT look
            # it up here.
            g = keras.random.normal(shape=(num_samples, self.latent_dim))
            norm = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(g), axis=-1, keepdims=True)
            )
            return g / keras.ops.maximum(norm, 1e-12)

        # Marsaglia/Muller: Gaussian draw, L2-normalize per row onto unit sphere,
        # scale by the layer radius. Zero-row degenerate case is floored the same
        # way HypersphereSampling.call does (ops.maximum(norm, eps)).
        radius = self.get_layer("vae_sampling").radius
        g = keras.random.normal(shape=(num_samples, self.latent_dim))
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(g), axis=-1, keepdims=True)
        )
        u = g / keras.ops.maximum(norm, 1e-12)
        return radius * u

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the latent space.

        Decodes latents drawn from this mode's TRUE prior (N(0, I) for gaussian,
        uniform-on-sphere of the layer radius for hypersphere modes).

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples tensor
        """
        z = self._sample_prior(num_samples)
        return self.decode(z)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step with VAE losses.

        Args:
            data: Training data (can be tuple or single tensor)

        Returns:
            Dictionary of loss values
        """
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Validate input shape
        if x.shape[1:] != self._input_shape:
            logger.warning(
                f"Input shape {x.shape} doesn't match expected {self._input_shape}"
            )

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)
            reconstruction = outputs["reconstruction"]

            # Validate reconstruction shape
            if reconstruction.shape != x.shape:
                raise ValueError(
                    f"Reconstruction shape {reconstruction.shape} "
                    f"doesn't match input {x.shape}"
                )

            # Compute losses
            reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
            kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])

            # Total loss (kl_weight is the schedulable warmup weight; == the
            # ctor kl_loss_weight unless a warmup callback is ramping it).
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

            # Add regularization losses
            # DECISION plan-2026-08-19T163559-499b6f0e/D-011
            # `add_loss` terms carry `compute_dtype` (float16 under
            # `mixed_float16`) while the running total is float32. Cast the
            # AUX SUM UP; never reduce the objective in half precision.
            if self.losses:
                total_loss += ops.cast(ops.sum(self.losses), total_loss.dtype)

            # DECISION plan-2026-08-19T163559-499b6f0e/D-089
            # `scale_loss` MUST be INSIDE the tape, and the SCALED value is what
            # `tape.gradient` differentiates; the UNSCALED loss stays what is
            # reported. Do NOT "simplify" this back to a gradient of the unscaled
            # loss: under `mixed_float16` Keras wraps the optimizer in a
            # `LossScaleOptimizer` whose `apply()` DIVIDES every gradient by
            # `dynamic_scale` (2**15 initially) UNCONDITIONALLY, so omitting the call
            # divides the WHOLE weight update by the loss scale, with no warning of
            # any kind. In float32 it is a provable no-op -- `Optimizer.scale_loss`
            # returns its argument unless `loss_scale_factor` is set. MEASURED here
            # (SGD lr=0.1, 5 steps, total |dW| over TRAINABLE weights, GPU 1):
            # BEFORE f32 9.680747e+01 vs fp16 1.549095e-03, ratio 6.249e+04.
            # See decisions.md D-089; same ruling at `masked_autoencoder/mae.py`
            # (D-036).
            scaled_loss = self.optimizer.scale_loss(total_loss)

        # Compute and clip gradients
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(scaled_loss, trainable_weights)
        # DECISION plan-2026-08-19T163559-499b6f0e/D-089
        # The elementwise clip must be expressed IN THE SCALED DOMAIN, because
        # `gradients` above are gradients of the SCALED loss and the optimizer
        # divides them by the same factor afterwards. Do NOT write a bare
        # `ops.clip(grad, -1.0, 1.0)` here: with a `LossScaleOptimizer` in play
        # that saturates EVERY gradient component at 1.0 and the subsequent
        # unscale turns the whole update into lr/32768 -- MEASURED, the
        # per-element |dW| collapsed to exactly 3.051758e-06 == 0.1 * 2**-15 and
        # the fp16/float32 ratio read 64.8 with the `scale_loss` call already
        # in place. `Optimizer.scale_loss(1.0)` returns the CURRENT loss scale,
        # and returns exactly 1.0 for a plain optimizer, so this restores the
        # documented "clip the true gradient at 1.0" semantics in both regimes
        # using only the same public API the scaling itself uses.
        clip_limit = self.optimizer.scale_loss(1.0)
        gradients = [
            ops.clip(grad, -clip_limit, clip_limit) if grad is not None else None
            for grad in gradients
        ]

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom test step with VAE losses.

        Args:
            data: Test data (can be tuple or single tensor)

        Returns:
            Dictionary of loss values
        """
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass
        outputs = self(x, training=False)
        reconstruction = outputs["reconstruction"]

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
        kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])
        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        # Add regularization losses
        # DECISION plan-2026-08-19T163559-499b6f0e/D-011 (see `train_step`).
        if self.losses:
            total_loss += ops.cast(ops.sum(self.losses), total_loss.dtype)

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def _compute_reconstruction_loss(
        self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute reconstruction loss with numerical stability.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Reconstruction loss value
        """
        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-011
        # float32 is a NUMERICS requirement here, not plumbing. The clip below
        # is the module's stated defence against `log(0)`, and in float16 it
        # DOES NOT WORK in either direction: `1e-7` is far below the smallest
        # normal float16 (6.10e-5) and `1.0 - 1e-7` rounds to exactly `1.0`, so
        # the upper clamp is a no-op and `binary_crossentropy` reaches
        # `log(0) = -inf`. Casting also removes the
        # `AddV2 float16 vs float32` raise at `train_step`'s
        # `reconstruction_loss + self.kl_weight * kl_loss` (step 5.8).
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")

        # Flatten for loss computation
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        # Clip predictions to avoid log(0)
        y_pred_clipped = ops.clip(y_pred_flat, 1e-7, 1.0 - 1e-7)

        # DECISION plan-2026-08-18T140459-7991552f/D-028: this reduction is a MEAN
        # over pixels while `_compute_kl_loss`'s gaussian branch SUMS over latents,
        # so `kl_loss_weight` is `beta / prod(input_shape)`, not `beta`. That is
        # KNOWN and MEASURED (micro/small 7.84, medium/large 3.92, xlarge 0.784 at
        # 28x28x1; 30.72 / 15.36 / 3.072 at 32x32x3 -- 3.92x from resolution alone),
        # and it is deliberately LEFT AS IS rather than switched to a sum. Do not
        # "fix" it in passing. Switching to a sum multiplies the whole objective by
        # `prod(input_shape)` (784-3072x) and changes what every shipped
        # `MODEL_VARIANTS` weight and every saved config MEANS; keeping the mean but
        # dividing the KL by the pixel count needs new per-variant defaults, and no
        # resolution-independent default can reproduce today's behaviour at more
        # than one resolution -- the two goals are mutually exclusive. Choosing a
        # re-tuned target requires training runs this repair could not do. What IS
        # fixed is the confusion: `effective_kl_beta` computes the real beta,
        # `summary()` logs it, the module docstring states the convention, and
        # `TestVAEEffectiveBeta` pins the arithmetic so a silent change is caught.
        # See decisions.md D-028.
        #
        # Binary crossentropy for better numerical stability
        reconstruction_loss = ops.mean(
            keras.losses.binary_crossentropy(y_true_flat, y_pred_clipped)
        )

        return reconstruction_loss

    def _compute_kl_loss(
        self, z_mean: keras.KerasTensor, z_log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute KL divergence loss with numerical stability.

        Args:
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution

        Returns:
            KL divergence loss value
        """
        # DECISION plan_2026-06-04_d4ef81f1/D-003: for hypersphere the Gaussian KL
        # is the WRONG prior for a sphere. It is REPLACED by a simplified
        # radius-variance KL (the 1-D Gaussian-Gaussian KL on the radius noise):
        # kl = mean(0.5 * (exp(rlv) - rlv - 1)). There is NO direction-KL term
        # (the direction has an implicit uniform-sphere prior; the radius mean is
        # fixed at 1.0 by the sampler). Do NOT "restore" the Gaussian KL here and
        # do NOT derive a full vMF S-VAE KL: this simplified regularizer is
        # user-locked scope and is float32-stable under [-20, 20] clipping. See
        # decisions.md D-003.
        # DECISION plan_2026-06-04_6196678d/D-003: vMF closed-form KL (depends on
        # kappa + latent_dim only); reuses the verified vmf_kl_divergence helper.
        # z_log_var carries the strictly-positive concentration kappa[B, 1] (NOT a
        # log-variance) -- do NOT clip/exp it or substitute the Gaussian/radius KL;
        # vmf_kl_divergence is the orchestrator-verified analytic vMF->uniform KL
        # (per-row >= 0). See decisions.md D-003.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-011
        # The whole KL runs in float32 regardless of `compute_dtype`, and that
        # is a NUMERICS requirement, not plumbing tidiness. Every branch below
        # clips its log-variance to [-20, 20] and then exponentiates:
        # `exp(20) == 4.85e8`, which is finite in float32 and **+inf in
        # float16** (max 65504). Under `mixed_float16` the inputs arrive as
        # float16, so without this cast the clip's own stated safety margin is
        # a lie and the KL can silently become `inf`/`nan`. It also removes the
        # `AddV2 float32 vs float16` raise at `train_step`'s
        # `reconstruction_loss + self.kl_weight * kl_loss` (the
        # `binary_crossentropy` reconstruction term is already float32), which
        # is what made every VAE unrunnable under mixed precision (step 5.8).
        # Do NOT instead cast the reconstruction term DOWN to float16.
        z_mean = ops.cast(z_mean, "float32")
        z_log_var = ops.cast(z_log_var, "float32")

        if self.sampling_type == "vmf":
            kl_loss = ops.mean(vmf_kl_divergence(z_log_var, self.latent_dim))
            return kl_loss

        if self.sampling_type == "hypersphere":
            rlv_clip = ops.clip(z_log_var, -20.0, 20.0)
            kl_loss = ops.mean(0.5 * (ops.exp(rlv_clip) - rlv_clip - 1.0))
            return kl_loss

        # gaussian: standard diagonal-Gaussian KL.
        # Clip log variance to prevent numerical issues
        z_log_var_clipped = ops.clip(z_log_var, -20.0, 20.0)

        # Compute KL divergence: KL(q||p) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * ops.sum(
            1.0 + z_log_var_clipped - ops.square(z_mean) - ops.exp(z_log_var_clipped),
            axis=1,
        )

        # Take mean across batch
        kl_loss = ops.mean(kl_loss)

        return kl_loss

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "input_shape": self._input_shape,
            "depths": self.depths,
            "steps_per_depth": self.steps_per_depth,
            "filters": self.filters,
            "kl_loss_weight": self.kl_loss_weight,
            "sampling_type": self.sampling_type,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "activation": serialize_activation(self.activation),
            "final_activation": serialize_activation(self.final_activation),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VAE":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            VAE model instance
        """
        # Deserialize complex objects
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        # Convert input_shape from list back to tuple
        if "input_shape" in config and isinstance(config["input_shape"], list):
            config["input_shape"] = tuple(config["input_shape"])

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info("VAE configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Latent dimension: {self.latent_dim}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Steps per depth: {self.steps_per_depth}")
        logger.info(f"  - Filters: {self.filters}")
        logger.info(f"  - KL loss weight: {self.kl_loss_weight}")
        logger.info(
            f"  - Effective beta (sum-over-pixels ELBO): "
            f"{self.effective_kl_beta:.4g}"
        )
        logger.info(f"  - Total parameters: {self.count_params():,}")


# ---------------------------------------------------------------------


def create_vae(
    input_shape: Tuple[int, int, int],
    latent_dim: int,
    variant: str = "small",
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs,
) -> VAE:
    """Convenience function to create and compile VAE models.

    Args:
        input_shape: Tuple representing (height, width, channels) of input
        latent_dim: Integer, dimensionality of the latent space
        variant: String, model variant ("micro", "small", "medium", "large", "xlarge")
        optimizer: String name or optimizer instance. Default is "adam"
        learning_rate: Float, learning rate for optimizer. Default is 0.001
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Compiled VAE model ready for training

    Example:
        >>> # MNIST VAE
        >>> model = create_vae(input_shape=(28, 28, 1), latent_dim=64, variant="small")
        >>>
        >>> # CIFAR-10 VAE
        >>> model = create_vae(input_shape=(32, 32, 3), latent_dim=128, variant="medium")
        >>>
        >>> # Custom learning rate
        >>> model = create_vae(
        ...     input_shape=(64, 64, 3),
        ...     latent_dim=256,
        ...     variant="large",
        ...     learning_rate=0.0005
        ... )
    """
    # Create the model
    model = VAE.from_variant(
        variant=variant, input_shape=input_shape, latent_dim=latent_dim, **kwargs
    )

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # DECISION plan_2026-06-04_6196678d/D-005: disable XLA jit for the vmf
    # sampler. VMFSampling uses keras.random.beta, which lowers to
    # StatelessRandomGammaV3 -- an op with NO XLA_GPU_JIT kernel in TF 2.18
    # (tf2xla conversion fails under the GPU multi_step_on_iterator path). Do
    # NOT try to make the Wood/Ulrich rejection sampler XLA-clean or globally
    # force jit_compile=False (gaussian/hypersphere keep XLA). See decisions.md D-005.
    jit_compile = False if getattr(model, "sampling_type", None) == "vmf" else "auto"
    model.compile(optimizer=optimizer_instance, jit_compile=jit_compile)

    # DECISION plan-2026-08-19T163559-499b6f0e/D-078: this factory does NOT
    # self-test. It used to run `keras.random.uniform((2,) + input_shape)`
    # through the model and `assert` on three output shapes. Three reasons that
    # was wrong, and do NOT restore it:
    #   1. `assert` is stripped by `python -O`, so the "validation" was already
    #      absent in exactly the deployment that most needs it.
    #   2. It ran a full forward pass on every construction -- cost paid by
    #      every caller, including the ones that only want the compiled shell.
    #      (It ALSO drew from the global seed stream via `keras.random.uniform`,
    #      but do not reach for that as the test: weight initialization draws
    #      from the same stream, so a seed-stream comparison cannot isolate the
    #      factory's own draw. That confound killed the first version of the
    #      test below; the shipped one counts `VAE.call` invocations instead.)
    #   3. It is a test. It now lives in
    #      `tests/test_models/test_vae/test_model.py::TestCreateVaeOutputShapes`,
    #      where it runs over all three sampling types instead of whichever one
    #      the caller happened to ask for.
    logger.info(f"Created VAE-{variant.upper()} for input shape {input_shape}")
    logger.info(f"Latent dim: {latent_dim}, Parameters: {model.count_params():,}")

    return model


def create_vae_from_config(
    config: Dict[str, Any],
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
) -> VAE:
    """Create VAE from configuration dictionary.

    Args:
        config: Configuration dictionary containing VAE parameters
        optimizer: String name or optimizer instance
        learning_rate: Float, learning rate for optimizer

    Returns:
        Compiled VAE model

    Example:
        >>> config = {
        ...     "latent_dim": 128,
        ...     "input_shape": (64, 64, 3),
        ...     "depths": 3,
        ...     "filters": [32, 64, 128],
        ...     "kl_loss_weight": 0.01
        ... }
        >>> model = create_vae_from_config(config)
    """
    # Create the model
    model = VAE(**config)

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile the model (vmf opts out of XLA jit; see D-005 in create_vae).
    jit_compile = False if getattr(model, "sampling_type", None) == "vmf" else "auto"
    model.compile(optimizer=optimizer_instance, jit_compile=jit_compile)

    logger.info(f"Created VAE from config with latent_dim={config['latent_dim']}")

    return model
