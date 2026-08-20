"""
Implementation of the Capsule Network
architecture that works with standard Keras training workflows (compile/fit).
The custom training logic is integrated into the model's train_step and test_step methods.

Architecture Overview:
    1. Feature Extraction: Conv2D layers for initial feature extraction
    2. Primary Capsules: Convert conventional CNN features to capsule format
    3. Routing Capsules: Final capsule layer with dynamic routing
    4. Decoder Network (optional): For reconstruction-based regularization

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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsNet(keras.Model):
    """Keras-compliant Capsule Network model.

    This model implements the full CapsNet architecture with integrated training logic
    that works seamlessly with Keras compile/fit workflow.

    Args:
        num_classes: Number of output classes.
        routing_iterations: Number of routing iterations for capsule routing.
        conv_filters: List of filter numbers for convolutional layers.
        primary_capsules: Number of primary capsules.
        primary_capsule_dim: Dimension of primary capsule vectors.
        digit_capsule_dim: Dimension of digit/class capsule vectors.
        reconstruction: Whether to include reconstruction network.
        input_shape: Shape of input images (height, width, channels).
        decoder_architecture: List of hidden layer sizes for decoder network.
        kernel_initializer: Initializer for convolutional weights.
        kernel_regularizer: Regularizer for convolutional weights.
        use_batch_norm: Whether to use batch normalization after convolutions.
        positive_margin: Positive margin for margin loss (m^+).
        negative_margin: Negative margin for margin loss (m^-).
        downweight: Downweight parameter for negative class loss (λ).
        reconstruction_weight: Weight for reconstruction loss component.
        name: Optional name for the model.
        **kwargs: Additional keyword arguments for the base Model class.

    Raises:
        ValueError: If any parameter is invalid or inconsistent.
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
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: `list(...)`, not
        # `.copy()`. The default is now a TUPLE (R-009 S1), and a tuple has no
        # `.copy()`; `list()` accepts both and keeps the stored attribute -- and
        # therefore `get_config`'s JSON type -- exactly what it always was.
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

        # Guide 1.1 Golden Rule: sub-layers are CREATED here, in `__init__`.
        # Neither helper reads `input_shape` -- both are pure functions of the
        # config stored above -- so neither has any reason to wait for
        # `build()`. Only the `.build()` calls themselves are deferred, to
        # `_build_sublayer_tree`.
        self._build_feature_extraction()
        self._build_capsule_layers()

        # DECISION plan-2026-08-18T073231-52a93f8c/D-007
        # DELIBERATE, BOUNDED DEVIATION from the Golden Rule: the decoder is
        # the ONE sub-layer that cannot always be created here. Its final
        # `Dense` width is `prod(input_shape[1:])` and its `Reshape` target IS
        # `input_shape[1:]` -- both are functions of the input shape, which is
        # legitimately unknown until `build()` when the caller did not pass
        # `input_shape=` to `__init__` (a supported, documented construction:
        # `_validate_parameters` only WARNS in that case). So: create it
        # eagerly HERE whenever `input_shape` was supplied, and only fall back
        # to creating it in `build()` when it was not. Do NOT "simplify" this
        # by moving decoder creation wholly back into `build()` -- that
        # re-introduces create-in-build for the common, fully-specified case
        # for no gain. Do NOT move it wholly into `__init__` either -- it would
        # crash `CapsNet(reconstruction=True)` with no `input_shape`.
        # See decisions.md D-007.
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
        """Build the model layers based on input shape.

        Per guide 1.1, sub-layer CREATION happens in `__init__`; this method
        only (a) validates the input shape, (b) captures `_input_shape`, (c)
        creates the decoder in the one case `__init__` could not (see D-007),
        and (d) builds the sub-layer tree (see D-006).
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

        # DECISION plan-2026-08-18T073231-52a93f8c/D-007
        # The residual half of the decoder deviation: reached ONLY when
        # `input_shape` was not supplied to `__init__`, so the decoder's output
        # width was genuinely unknowable there. The `self.decoder is None`
        # guard is what keeps this from re-creating (and thereby discarding)
        # the decoder `__init__` already made. See decisions.md D-007.
        if self.reconstruction and self._input_shape is not None and self.decoder is None:
            self._build_decoder()

        self._build_sublayer_tree(input_shape)

        self._layers_built = True
        super().build(input_shape)

    def _build_sublayer_tree(self, input_shape: Tuple[Optional[int], int, int, int]) -> None:
        """Explicitly build every sub-layer, threading shapes in dependency order.

        # DECISION plan-2026-08-18T073231-52a93f8c/D-006
        These `.build()` calls are LOAD-BEARING. Do NOT delete them as
        "redundant because `call()` builds them anyway" -- that is exactly the
        defect this replaced. Mechanism (measured 2026-08-18): Keras'
        `Model.build_from_config` only falls back to build-by-run
        (`_build_by_run_for_single_pos_arg`, an actual forward pass that builds
        the whole sub-layer tree) when `keras.src.utils.python_utils.is_default(
        self.build)` is True. `CapsNet` OVERRIDES `build`, so `is_default` is
        False and that fallback is disabled; Keras instead calls `self.build(
        input_shape)` inside a bare `try/except: pass`. The previous override
        only CREATED the sub-layer objects and never built them, so after
        `load_model` every sub-layer was still `built=False` with no variables
        for the saved arrays to be written into -- and the swallowing
        `try/except` meant nothing raised or warned. The loss was therefore
        SILENT: layer paths, weight shapes and parameter totals all matched the
        donor while every restored kernel was FRESH. The only instrument that
        sees it is `len(model.weights)` BEFORE the first `call()` (it was 0;
        after the first call, lazy build makes it 16 either way, so a post-call
        count cannot distinguish fixed from broken). See decisions.md D-006.

        QUALIFICATION -- this mechanism is NECESSARY BUT NOT SUFFICIENT, and it
        is NOT a general law that "overriding `build` loses your weights".
        Counterexample in this same repo: `models/pft_sr/model.py::PFTSR` also
        overrides `keras.Model.build`, also creates every sub-layer inside
        `build()`, and never calls `.build()` on any of them -- yet it
        round-trips with 22 weights BEFORE the first `call()` and an output
        delta of exactly 0.0. MEASURED here why: `PFTSR.build` ends with a
        CONCRETE DUMMY FORWARD (`self.call(keras.ops.zeros((1,) +
        input_shape[1:]))`, `pft_sr/model.py:318-320`), which materializes the
        whole sub-layer tree. Disabling only that dummy forward drops it to 0
        weights. So the discriminating property is whether `build()`
        MATERIALIZES the sub-layer tree at all -- by explicit `.build()` calls
        as here, or by a forward pass as there -- not whether `build` is
        overridden. Do not cite D-006 as evidence that some other model with an
        overridden `build` is broken; measure `len(model.weights)` before the
        first `call()` on that model.

        Args:
            input_shape: 4D input shape `(batch, height, width, channels)`.
        """
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
            conv_layer = keras.layers.Conv2D(
                filters=filters,
                kernel_size=9 if i == 0 else 5,  # First layer uses 9x9, others use 5x5
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
        """Forward pass through the capsule network."""
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
        """Perform reconstruction using the decoder network."""
        if mask is not None:
            if mask.shape[-1] != self.num_classes:
                raise ValueError(
                    f"Mask shape mismatch. Expected last dimension {self.num_classes}, "
                    f"got {mask.shape[-1]}"
                )
            # Use provided mask (one-hot encoded labels)
            reconstruction_mask = mask
        else:
            # Use predicted classes for reconstruction
            reconstruction_mask = ops.one_hot(ops.argmax(lengths, axis=1), num_classes=self.num_classes)

        masked_caps = ops.multiply(digit_caps, ops.expand_dims(reconstruction_mask, -1))

        decoder_input = ops.reshape(masked_caps, (-1, self.num_classes * self.digit_capsule_dim))

        return self.decoder(decoder_input)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-057
    # A bare `except: continue` here dropped ANY metric whose signature or shape
    # did not match `(y, outputs["length"])` from training AND from the returned
    # logs, silently and with no warning -- a user's metric could simply never
    # appear, and `except:` also swallowed `KeyboardInterrupt` and
    # `SystemExit`. It is narrowed to the shape/dtype family a wrong-arity
    # metric actually raises, and it WARNS once per metric name so the drop is
    # visible in the log. DO NOT re-widen it to a bare `except`, and DO NOT
    # convert it to a raise: `CapsNet` returns a DICT and a stock metric like
    # `keras.metrics.Accuracy` legitimately wants the `length` head, so some
    # mismatch is expected and must not abort a multi-hour run.
    # See decisions.md D-057.
    def _update_metrics(self, y: Any, outputs: Dict[str, Any]) -> None:
        """Update every compiled metric, warning about ones that cannot take the data.

        Args:
            y: The batch's labels, one-hot encoded.
            outputs: The dict this model's ``call`` returns; ``CapsuleAccuracy``
                consumes it whole, every other metric gets ``outputs["length"]``.

        Returns:
            None -- metrics are updated in place.

        Failure mode: a metric that raises ``ValueError``/``TypeError``/
        ``tf.errors.InvalidArgumentError`` (wrong arity, wrong shape, wrong
        dtype) is SKIPPED with a one-time WARNING naming it. Anything else
        propagates.
        """
        for metric in self.metrics:
            # DECISION plan-2026-08-17T183311-79c63e38/D-021
            # Skip the loss tracker. Keras' `Trainer.metrics` yields
            # `self._loss_tracker` (a `keras.metrics.Mean` named "loss") FIRST,
            # and `Mean.update_state(values, sample_weight)` accepts
            # `(y_onehot, lengths)` without complaint because both are
            # (B, num_classes) -- so this loop silently accumulated
            # mean(y * lengths) into the loss tracker. No exception ever fired,
            # and it stayed invisible only because `results.update({"loss":
            # total_loss})` overwrites the reported value afterwards;
            # `model.metrics[0].result()` was garbage. Do NOT replace this with
            # a name check: the tracker is identified by identity, not by name.
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
                    # MEASURED: `self.metrics` yields Keras' `CompileMetrics`
                    # WRAPPER, not the individual metrics, so one misshaped
                    # metric takes the whole container down with it. Name the
                    # contents, or the warning points at 'compile_metrics' and
                    # the user cannot tell which of their metrics is at fault.
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
        """Custom training step with margin loss and reconstruction loss."""
        x, y = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True, mask=y)

            margin_loss_value = ops.mean(capsule_margin_loss(
                outputs["length"],  # y_pred
                y,                  # y_true
                self.downweight,
                self.positive_margin,
                self.negative_margin
            ))

            total_loss = margin_loss_value
            reconstruction_loss_value = ops.convert_to_tensor(0.0, dtype=total_loss.dtype)

            if self.reconstruction and "reconstructed" in outputs:
                # DECISION plan-2026-08-19T163559-499b6f0e/D-011
                # Reduce in float32. `x` is float32 dataset data while
                # `outputs["reconstructed"]` carries `compute_dtype`; under
                # `mixed_float16` this subtraction raised
                # `TypeError: Input 'y' of 'Sub' ...` (step 5.8). Cast the
                # PREDICTION UP -- never cast the data down, a squared-error
                # mean accumulated in float16 underflows on small residuals.
                reconstruction_loss_value = ops.mean(ops.square(
                    x - ops.cast(outputs["reconstructed"], x.dtype)
                ))
                total_loss += ops.cast(
                    self.reconstruction_weight * reconstruction_loss_value,
                    total_loss.dtype,
                )

            # DECISION plan-2026-08-19T163559-499b6f0e/D-011
            # `add_loss` terms carry `compute_dtype`; cast the AUX SUM UP.
            if self.losses:
                total_loss += ops.cast(ops.sum(self.losses), total_loss.dtype)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self._update_metrics(y, outputs)
        # The tracker `_update_metrics` deliberately skips is fed HERE, with the
        # quantity it is named for.
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
        """Custom test step with margin loss and reconstruction loss.

        Evaluation runs the decoder on the PREDICTED class, not the true one:
        the reconstruction reported by ``evaluate()`` must be reachable at
        inference time.
        """
        x, y = data

        # DECISION plan-2026-08-17T183311-79c63e38/D-021
        # No `mask=y` here. `_reconstruct` takes `reconstruction_mask = mask`
        # whenever mask is not None, so passing the labels teacher-forced the
        # decoder and made `evaluate()`'s reconstruction loss optimistic by
        # construction -- the inference branch (argmax over the capsule lengths)
        # was unreachable from this method. `train_step` DOES pass `mask=y`, and
        # that is correct: masking by the true class during training is the
        # paper's own recipe (Sabour et al. 2017 § 4.1). Do NOT "restore
        # symmetry" between the two steps. See decisions.md D-021.
        outputs = self(x, training=False)

        margin_loss_value = ops.mean(capsule_margin_loss(
            outputs["length"],  # y_pred
            y,                  # y_true
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
        # The tracker `_update_metrics` deliberately skips is fed HERE, with the
        # quantity it is named for.
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
        """Get model configuration for serialization."""
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
        """Create model from configuration."""
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
        """Save the model to a file.

        `filepath` must end in `.keras` (preferred) or `.h5`; Keras 3 selects
        the format from the extension. The `save_format` parameter this method
        used to forward was removed in Keras 3 and RAISES for any path Keras
        cannot classify, so passing it turned an "unknown extension" error into
        a deprecation error naming the wrong cause.
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save model
        self.save(filepath, overwrite=overwrite)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "CapsNet":
        """Load a saved model."""
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

    Args:
        num_classes: Number of output classes.
        input_shape: Shape of input images.
        optimizer: Optimizer name or instance.
        learning_rate: Learning rate for optimizer.
        **kwargs: Additional arguments for CapsNet.

    Returns:
        Compiled CapsNet model.
    """
    model = CapsNet(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    # Handle optimizer
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
        optimizer.learning_rate = learning_rate

    # Compile model with dummy loss (we handle loss in train_step/test_step)
    model.compile(
        optimizer=optimizer,
        loss=None,  # We handle loss computation in train_step/test_step
        metrics=[CapsuleAccuracy()]
    )

    return model

# ---------------------------------------------------------------------
