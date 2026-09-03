"""CapsNet, a capsule network with dynamic routing and an optional reconstruction decoder.

A stack of Conv2D layers extracts features, a primary capsule layer groups them into
short vectors, and a routing capsule layer refines those vectors by iterative agreement
between capsules instead of max-pooling. A capsule's output length stands for the
probability that its class is present, and its orientation encodes pose. An optional
decoder reconstructs the input image from the winning capsule, which regularizes
training. The model exposes standard Keras `compile`/`fit`, with the margin loss and
reconstruction loss implemented directly in `train_step` and `test_step`.

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017).
      Dynamic routing between capsules. In Advances in Neural
      Information Processing Systems (pp. 3856-3866).
"""

import os
import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import length
from dl_techniques.metrics.capsule_accuracy import CapsuleAccuracy
from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss
from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.capsnet.model")
class CapsNet(keras.Model):
    """Capsule network with dynamic routing and an optional reconstruction decoder.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
          |
          v
        Conv2D stack (conv_filters)      -> feature map
          |
          v
        PrimaryCapsule                   -> [B, N_p, D_p]
          |
          v
        RoutingCapsule (dynamic routing) -> digit_caps [B, num_classes, D_d]
          |
          +--> length(digit_caps) -> class probabilities
          |
          '--> mask by true/predicted class -> Decoder (optional) -> reconstruction

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param routing_iterations: Number of dynamic-routing iterations between the primary and digit capsules.
    :type routing_iterations: int
    :param conv_filters: Filter counts for the convolutional feature-extraction stack.
    :type conv_filters: Sequence[int]
    :param primary_capsules: Number of primary capsules.
    :type primary_capsules: int
    :param primary_capsule_dim: Dimension of each primary capsule vector.
    :type primary_capsule_dim: int
    :param digit_capsule_dim: Dimension of each digit (class) capsule vector.
    :type digit_capsule_dim: int
    :param reconstruction: Whether to build the reconstruction decoder.
    :type reconstruction: bool
    :param input_shape: Shape of input images ``(height, width, channels)``. Needed at construction time only if `reconstruction` is True and the decoder should be built eagerly.
    :type input_shape: Optional[Tuple[int, int, int]]
    :param decoder_architecture: Hidden layer sizes for the reconstruction decoder.
    :type decoder_architecture: Sequence[int]
    :param kernel_initializer: Initializer for convolutional and dense weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for convolutional and dense weights.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param use_batch_norm: Whether to apply batch normalization after each convolution.
    :type use_batch_norm: bool
    :param positive_margin: Positive margin ``m+`` in the margin loss.
    :type positive_margin: float
    :param negative_margin: Negative margin ``m-`` in the margin loss.
    :type negative_margin: float
    :param downweight: Downweight factor ``lambda`` for the negative-class term in the margin loss.
    :type downweight: float
    :param reconstruction_weight: Weight applied to the reconstruction loss term.
    :type reconstruction_weight: float
    :param name: Optional model name.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments passed to `keras.Model`.
    :raises ValueError: If any parameter is invalid or inconsistent.
    """

    def __init__(
        self,
        num_classes: int,
        routing_iterations: int = 3,
        conv_filters: Sequence[int] = (256, 256),
        primary_capsules: int = 32,
        primary_capsule_dim: int = 8,
        digit_capsule_dim: int = 16,
        reconstruction: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        decoder_architecture: Sequence[int] = (512, 1024),
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        positive_margin: float = 0.9,
        negative_margin: float = 0.1,
        downweight: float = 0.5,
        reconstruction_weight: float = 0.01,
        name: Optional[str] = "capsnet",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        self._validate_parameters(
            num_classes, routing_iterations, primary_capsules,
            primary_capsule_dim, digit_capsule_dim, reconstruction, input_shape
        )

        self.num_classes = num_classes
        self.routing_iterations = routing_iterations
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: use list(...), not .copy() —
        # conv_filters defaults to a tuple, which has no .copy(). See decisions.md.
        self.conv_filters = list(conv_filters)
        self.primary_capsules = primary_capsules
        self.primary_capsule_dim = primary_capsule_dim
        self.digit_capsule_dim = digit_capsule_dim
        self.reconstruction = reconstruction
        self._input_shape = input_shape
        self.decoder_architecture = list(decoder_architecture)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = self._process_regularizer(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.downweight = downweight
        self.reconstruction_weight = reconstruction_weight

        # Metric names `_update_metrics` has already warned about, so a
        # per-step warning does not flood a multi-hour run's log.
        self._skipped_metric_names = set()

        self.conv_layers = []
        self.batch_norm_layers = []
        self.activation_layers = []
        self.primary_caps = None
        self.digit_caps = None
        self.decoder = None

        self._layers_built = False

        # Sub-layers are created here in __init__; neither helper reads input_shape.
        self._build_feature_extraction()
        self._build_capsule_layers()

        # DECISION plan-2026-08-18T073231-52a93f8c/D-007: the decoder's Dense width
        # depends on input_shape, so it is created here only when input_shape is known; otherwise build() creates it. See decisions.md.
        if self.reconstruction and self._input_shape is not None:
            self._build_decoder()

    def _validate_parameters(
        self,
        num_classes: int,
        routing_iterations: int,
        primary_capsules: int,
        primary_capsule_dim: int,
        digit_capsule_dim: int,
        reconstruction: bool,
        input_shape: Optional[Tuple[int, int, int]]
    ) -> None:
        """Validate initialization parameters."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")
        if primary_capsules <= 0:
            raise ValueError(f"primary_capsules must be positive, got {primary_capsules}")
        if primary_capsule_dim <= 0:
            raise ValueError(f"primary_capsule_dim must be positive, got {primary_capsule_dim}")
        if digit_capsule_dim <= 0:
            raise ValueError(f"digit_capsule_dim must be positive, got {digit_capsule_dim}")
        if reconstruction and input_shape is None:
            logger.warning(
                "Reconstruction enabled but input_shape not provided. "
                "Decoder will be created during build if input shape is available."
            )

    def _process_regularizer(
        self,
        regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    ) -> Optional[keras.regularizers.Regularizer]:
        """Process and validate regularizer parameter."""
        if regularizer is None:
            return None
        if isinstance(regularizer, str):
            regularizer_map = {
                "l1": keras.regularizers.L1(0.01),
                "l2": keras.regularizers.L2(0.01),
                "l1_l2": keras.regularizers.L1L2(l1=0.01, l2=0.01)
            }
            if regularizer.lower() in regularizer_map:
                return regularizer_map[regularizer.lower()]
            else:
                return keras.regularizers.get(regularizer)
        return keras.regularizers.get(regularizer)

    def build(self, input_shape: Tuple[Optional[int], int, int, int]) -> None:
        """Validate the input shape, create the decoder if still needed, and build every sub-layer.

        :param input_shape: 4D input shape ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :raises ValueError: If `input_shape` is not 4D.
        """
        if self._layers_built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape [batch, height, width, channels], got {input_shape}"
            )

        logger.info(f"Building CapsNet with input shape: {input_shape}")

        # Store input shape for reconstruction if not provided during init
        if self.reconstruction and self._input_shape is None:
            self._input_shape = tuple(input_shape[1:])

        # DECISION plan-2026-08-18T073231-52a93f8c/D-007: reached only when input_shape
        # was not supplied to __init__; the decoder is None guard stops it re-creating an already-made decoder. See decisions.md.
        if self.reconstruction and self._input_shape is not None and self.decoder is None:
            self._build_decoder()

        self._build_sublayer_tree(input_shape)

        self._layers_built = True
        super().build(input_shape)

    def _build_sublayer_tree(self, input_shape: Tuple[Optional[int], int, int, int]) -> None:
        """Build every sub-layer explicitly, threading shapes in dependency order.

        :param input_shape: 4D input shape ``(batch, height, width, channels)``.
        :type input_shape: tuple
        """
        # DECISION plan-2026-08-18T073231-52a93f8c/D-006: these .build() calls are
        # required. CapsNet overrides build(), which disables Keras' build-by-run fallback, so a loaded model reloaded with 0 weights without them. See decisions.md.
        shape = tuple(input_shape)

        for i in range(len(self.conv_layers)):
            conv_layer = self.conv_layers[i]
            conv_layer.build(shape)
            shape = conv_layer.compute_output_shape(shape)

            bn_layer = self.batch_norm_layers[i]
            if bn_layer is not None:
                bn_layer.build(shape)
                shape = bn_layer.compute_output_shape(shape)

            activation_layer = self.activation_layers[i]
            activation_layer.build(shape)
            shape = activation_layer.compute_output_shape(shape)

        self.primary_caps.build(shape)
        shape = self.primary_caps.compute_output_shape(shape)

        self.digit_caps.build(shape)

        if self.decoder is not None:
            # `_reconstruct` flattens the masked digit capsules before the
            # decoder, so the decoder never sees `digit_caps`' own output shape.
            self.decoder.build(
                (input_shape[0], self.num_classes * self.digit_capsule_dim)
            )

    def _build_feature_extraction(self) -> None:
        """Build convolutional feature extraction layers."""
        for i, filters in enumerate(self.conv_filters):
            # First layer uses a 9x9 kernel, the rest use 5x5.
            conv_layer = keras.layers.Conv2D(
                filters=filters,
                kernel_size=9 if i == 0 else 5,
                strides=1,
                padding="valid",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"conv_{i+1}"
            )
            self.conv_layers.append(conv_layer)

            if self.use_batch_norm:
                bn_layer = keras.layers.BatchNormalization(name=f"bn_{i+1}")
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)

            activation_layer = keras.layers.ReLU(name=f"relu_{i+1}")
            self.activation_layers.append(activation_layer)

    def _build_capsule_layers(self) -> None:
        """Build primary and routing capsule layers."""
        self.primary_caps = PrimaryCapsule(
            num_capsules=self.primary_capsules,
            dim_capsules=self.primary_capsule_dim,
            kernel_size=9,
            strides=2,
            padding="valid",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="primary_caps"
        )

        self.digit_caps = RoutingCapsule(
            num_capsules=self.num_classes,
            dim_capsules=self.digit_capsule_dim,
            routing_iterations=self.routing_iterations,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="digit_caps"
        )

    def _build_decoder(self) -> None:
        """Build reconstruction decoder network."""
        if self._input_shape is None:
            raise ValueError("Cannot build decoder without input_shape")

        decoder_layers = []

        for i, units in enumerate(self.decoder_architecture):
            decoder_layers.append(
                keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"decoder_hidden_{i+1}"
                )
            )

        flattened_size = int(self._input_shape[0] * self._input_shape[1] * self._input_shape[2])

        decoder_layers.append(
            keras.layers.Dense(
                units=flattened_size,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="decoder_output"
            )
        )

        decoder_layers.append(
            keras.layers.Reshape(
                target_shape=self._input_shape,
                name="decoder_reshape"
            )
        )

        self.decoder = keras.Sequential(decoder_layers, name="reconstruction_decoder")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Run the forward pass.

        :param inputs: Input images, shape ``[B, H, W, C]``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :param mask: One-hot labels used to select which capsule the decoder reconstructs from. Falls back to the predicted class when omitted.
        :type mask: Optional[keras.KerasTensor]
        :return: Dict with ``digit_caps``, ``length`` (class probabilities), and ``reconstructed`` (if reconstruction is enabled).
        :rtype: Dict[str, keras.KerasTensor]
        """
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected 4D input [batch, height, width, channels], got shape {inputs.shape}")

        x = inputs
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_batch_norm and self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i](x, training=training)
            x = self.activation_layers[i](x)

        primary_caps_output = self.primary_caps(x)

        digit_caps_output = self.digit_caps(primary_caps_output)

        # Calculate capsule lengths (class probabilities)
        lengths = length(digit_caps_output)

        results = {
            "digit_caps": digit_caps_output,
            "length": lengths
        }

        if self.reconstruction and self.decoder is not None:
            reconstructed = self._reconstruct(digit_caps_output, lengths, mask)
            results["reconstructed"] = reconstructed

        return results

    def _reconstruct(
        self,
        digit_caps: keras.KerasTensor,
        lengths: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Reconstruct the input from the masked digit capsules.

        :param digit_caps: Digit capsule output, shape ``[B, num_classes, digit_capsule_dim]``.
        :type digit_caps: keras.KerasTensor
        :param lengths: Capsule lengths (class probabilities), shape ``[B, num_classes]``.
        :type lengths: keras.KerasTensor
        :param mask: One-hot class mask to reconstruct from. Falls back to the predicted class when omitted.
        :type mask: Optional[keras.KerasTensor]
        :return: Reconstructed image, shape ``_input_shape``.
        :rtype: keras.KerasTensor
        :raises ValueError: If `mask`'s last dimension does not equal `num_classes`.
        """
        if mask is not None:
            if mask.shape[-1] != self.num_classes:
                raise ValueError(
                    f"Mask shape mismatch. Expected last dimension {self.num_classes}, "
                    f"got {mask.shape[-1]}"
                )
            # Provided mask is one-hot encoded labels.
            reconstruction_mask = mask
        else:
            # Falls back to the predicted class.
            reconstruction_mask = ops.one_hot(ops.argmax(lengths, axis=1), num_classes=self.num_classes)

        masked_caps = ops.multiply(digit_caps, ops.expand_dims(reconstruction_mask, -1))

        decoder_input = ops.reshape(masked_caps, (-1, self.num_classes * self.digit_capsule_dim))

        return self.decoder(decoder_input)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-057: catch is narrowed to
    # ValueError/TypeError/InvalidArgumentError, never a bare except, which silently swallowed KeyboardInterrupt/SystemExit and dropped mismatched metrics. See decisions.md.
    def _update_metrics(self, y: Any, outputs: Dict[str, Any]) -> None:
        """Update every compiled metric, warning about ones that cannot take the data.

        A metric that raises ``ValueError``/``TypeError``/``tf.errors.InvalidArgumentError``
        (wrong arity, shape, or dtype) is skipped with a one-time warning naming it. Any
        other exception propagates.

        :param y: The batch's labels, one-hot encoded.
        :type y: Any
        :param outputs: The dict this model's ``call`` returns; ``CapsuleAccuracy`` consumes it whole, every other metric gets ``outputs["length"]``.
        :type outputs: Dict[str, Any]
        :return: None. Metrics are updated in place.
        :rtype: None
        """
        for metric in self.metrics:
            # DECISION plan-2026-08-17T183311-79c63e38/D-021: skip the loss tracker by
            # identity, not name — it silently accepted (y, lengths) as (values, sample_weight) and accumulated a garbage mean. See decisions.md.
            if metric is getattr(self, "_loss_tracker", None):
                continue
            if isinstance(metric, CapsuleAccuracy):
                metric.update_state(y, outputs)
                continue
            try:
                metric.update_state(y, outputs["length"])
            except (ValueError, TypeError, tf.errors.InvalidArgumentError) as error:
                if metric.name not in self._skipped_metric_names:
                    self._skipped_metric_names.add(metric.name)
                    # self.metrics yields Keras' CompileMetrics wrapper, not the
                    # individual metrics, so name the contents or the warning points at 'compile_metrics'.
                    contained = [
                        inner.name
                        for inner in getattr(metric, "metrics", []) or []
                    ]
                    detail = f" containing {contained}" if contained else ""
                    logger.warning(
                        f"CapsNet: metric '{metric.name}' "
                        f"({type(metric).__name__}){detail} cannot consume "
                        f"(y, outputs['length']) and is being SKIPPED for the "
                        f"whole run -- it will be missing from the training "
                        f"logs entirely. Underlying error: {error}"
                    )

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Run one training step: forward pass, margin + reconstruction loss, gradient update.

        :param data: ``(x, y)`` batch, `y` one-hot encoded.
        :type data: Tuple[tf.Tensor, tf.Tensor]
        :return: Dict of metric results plus ``loss``, ``margin_loss``, and ``reconstruction_loss``.
        :rtype: Dict[str, tf.Tensor]
        """
        x, y = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True, mask=y)

            margin_loss_value = ops.mean(capsule_margin_loss(
                outputs["length"],
                y,
                self.downweight,
                self.positive_margin,
                self.negative_margin
            ))

            total_loss = margin_loss_value
            reconstruction_loss_value = ops.convert_to_tensor(0.0, dtype=total_loss.dtype)

            if self.reconstruction and "reconstructed" in outputs:
                # DECISION plan-2026-08-19T163559-499b6f0e/D-011: cast the prediction up
                # to x's dtype, never cast x down — under mixed_float16 the reverse raised a Sub dtype TypeError. See decisions.md.
                reconstruction_loss_value = ops.mean(ops.square(
                    x - ops.cast(outputs["reconstructed"], x.dtype)
                ))
                total_loss += ops.cast(
                    self.reconstruction_weight * reconstruction_loss_value,
                    total_loss.dtype,
                )

            # DECISION plan-2026-08-19T163559-499b6f0e/D-011: add_loss terms carry
            # compute_dtype, so cast the summed auxiliary losses up too. See decisions.md.
            if self.losses:
                total_loss += ops.cast(ops.sum(self.losses), total_loss.dtype)

            # DECISION plan-2026-08-19T163559-499b6f0e/D-089: scale_loss must stay inside
            # the tape and gradient must differentiate the scaled value — under mixed_float16 skipping it divides the whole update by ~2**15. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(total_loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self._update_metrics(y, outputs)
        # Feed the loss tracker that _update_metrics deliberately skips.
        loss_tracker = getattr(self, "_loss_tracker", None)
        if loss_tracker is not None:
            loss_tracker.update_state(total_loss)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.result()

        results.update({
            "loss": total_loss,
            "margin_loss": margin_loss_value,
            "reconstruction_loss": reconstruction_loss_value
        })

        return results

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Run one evaluation step: forward pass on the predicted class, margin + reconstruction loss.

        Evaluation runs the decoder on the predicted class, not the true one, so the
        reconstruction reported by ``evaluate()`` is reachable at inference time.

        :param data: ``(x, y)`` batch, `y` one-hot encoded.
        :type data: Tuple[tf.Tensor, tf.Tensor]
        :return: Dict of metric results plus ``loss``, ``margin_loss``, and ``reconstruction_loss``.
        :rtype: Dict[str, tf.Tensor]
        """
        x, y = data

        # DECISION plan-2026-08-17T183311-79c63e38/D-021: no mask=y here — masking by the
        # true label made evaluate()'s reconstruction loss optimistic and unreachable from inference. See decisions.md.
        outputs = self(x, training=False)

        margin_loss_value = ops.mean(capsule_margin_loss(
            outputs["length"],
            y,
            self.downweight,
            self.positive_margin,
            self.negative_margin
        ))

        total_loss = margin_loss_value
        reconstruction_loss_value = ops.convert_to_tensor(0.0, dtype=total_loss.dtype)

        if self.reconstruction and "reconstructed" in outputs:
            reconstruction_loss_value = ops.mean(ops.square(x - outputs["reconstructed"]))
            total_loss += self.reconstruction_weight * reconstruction_loss_value

        if self.losses:
            total_loss += ops.sum(self.losses)

        self._update_metrics(y, outputs)
        # Feed the loss tracker that _update_metrics deliberately skips.
        loss_tracker = getattr(self, "_loss_tracker", None)
        if loss_tracker is not None:
            loss_tracker.update_state(total_loss)

        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.result()

        results.update({
            "loss": total_loss,
            "margin_loss": margin_loss_value,
            "reconstruction_loss": reconstruction_loss_value
        })

        return results

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Config dict with every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "routing_iterations": self.routing_iterations,
            "conv_filters": self.conv_filters,
            "primary_capsules": self.primary_capsules,
            "primary_capsule_dim": self.primary_capsule_dim,
            "digit_capsule_dim": self.digit_capsule_dim,
            "reconstruction": self.reconstruction,
            "input_shape": self._input_shape,
            "decoder_architecture": self.decoder_architecture,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "positive_margin": self.positive_margin,
            "negative_margin": self.negative_margin,
            "downweight": self.downweight,
            "reconstruction_weight": self.reconstruction_weight
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CapsNet":
        """Build a model from a config dict, deserializing initializer and regularizer entries.

        :param config: Config dict as returned by `get_config`.
        :type config: Dict[str, Any]
        :return: A new `CapsNet` instance.
        :rtype: CapsNet
        """
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def save_model(
        self,
        filepath: str,
        overwrite: bool = True,
    ) -> None:
        """Build the model if needed, then save it to a file.

        `filepath` must end in ``.keras`` (preferred) or ``.h5``; Keras 3 selects the
        format from the extension. An unbuilt model is built first from the `input_shape`
        given to `__init__`, since saving an unbuilt model would otherwise write an
        archive with zero weights.

        :param filepath: Destination path, ending in ``.keras`` or ``.h5``.
        :type filepath: str
        :param overwrite: Whether to overwrite an existing file at `filepath`.
        :type overwrite: bool
        :raises ValueError: If the model is unbuilt and was constructed without an `input_shape`, so there is nothing to build from.
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # DECISION plan-2026-08-22T035419-a11304c8/D-053: build before save — an unbuilt
        # model.save() writes a syntactically valid archive with 0 weights, only a UserWarning flags it. See decisions.md.
        if not self.built:
            if self._input_shape is None:
                raise ValueError(
                    "Cannot save an unbuilt CapsNet that was constructed "
                    "without `input_shape`: the archive would contain zero "
                    "weights. Call the model on a batch, or call "
                    "`model.build((None, height, width, channels))`, first."
                )
            self.build((None, *self._input_shape))

        self.save(filepath, overwrite=overwrite)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "CapsNet":
        """Load a saved CapsNet model, registering its custom objects.

        :param filepath: Path to a ``.keras`` or ``.h5`` archive.
        :type filepath: str
        :return: The loaded model.
        :rtype: CapsNet
        """
        custom_objects = {
            "CapsNet": cls,
            "PrimaryCapsule": PrimaryCapsule,
            "RoutingCapsule": RoutingCapsule,
            "capsule_margin_loss": capsule_margin_loss,
            "length": length,
            "CapsuleAccuracy": CapsuleAccuracy
        }

        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"Model loaded from {filepath}")
        return model

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)
        logger.info(f"CapsNet Configuration:")
        logger.info(f"  - Classes: {self.num_classes}")
        logger.info(f"  - Routing iterations: {self.routing_iterations}")
        logger.info(f"  - Conv filters: {self.conv_filters}")
        logger.info(f"  - Primary capsules: {self.primary_capsules} x {self.primary_capsule_dim}D")
        logger.info(f"  - Digit capsules: {self.num_classes} x {self.digit_capsule_dim}D")
        logger.info(f"  - Reconstruction: {self.reconstruction}")
        logger.info(f"  - Batch normalization: {self.use_batch_norm}")
        logger.info(f"  - Margins: +{self.positive_margin}, -{self.negative_margin}")
        logger.info(f"  - Downweight: {self.downweight}")
        logger.info(f"  - Reconstruction weight: {self.reconstruction_weight}")


# ---------------------------------------------------------------------

def create_capsnet(
    num_classes: int,
    input_shape: Tuple[int, int, int],
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs
) -> CapsNet:
    """Create and compile a CapsNet model.

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Shape of input images.
    :type input_shape: Tuple[int, int, int]
    :param optimizer: Optimizer name or instance.
    :type optimizer: Union[str, keras.optimizers.Optimizer]
    :param learning_rate: Learning rate, applied when `optimizer` is given as a name.
    :type learning_rate: float
    :param kwargs: Additional keyword arguments passed to `CapsNet`.
    :return: A compiled `CapsNet` model.
    :rtype: CapsNet
    """
    model = CapsNet(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
        optimizer.learning_rate = learning_rate

    # Loss is None: train_step/test_step compute the margin and reconstruction losses directly.
    model.compile(
        optimizer=optimizer,
        loss=None,
        metrics=[CapsuleAccuracy()]
    )

    return model

# ---------------------------------------------------------------------
