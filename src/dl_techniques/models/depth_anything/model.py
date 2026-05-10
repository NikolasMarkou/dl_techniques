"""Depth Anything Implementation in Keras.

This module implements the Depth Anything model architecture as described in the paper.
Key components:
1. Feature Alignment Loss for semantic prior transfer
2. Affine-Invariant Loss for multi-dataset training
3. Strong augmentation pipeline for unlabeled data
4. DINOv2 encoder with DPT decoder architecture

Key Features:
- Uses large-scale unlabeled data (62M images) for better generalization
- Implements challenging student model training with strong perturbations
- Inherits semantic priors from pre-trained encoders
- Supports fine-tuning on specific datasets
- State-of-the-art results on NYUv2 and KITTI benchmarks

Example:
    >>> config = ModelConfig(encoder_type='vit_l')
    >>> model = create_depth_anything(config)
    >>> model.compile(
    ...     optimizer=keras.optimizers.AdamW(learning_rate=5e-6),
    ...     loss_weights={'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1}
    ... )
    >>> # Training would require proper data pipeline
    >>> # model.fit([x_labeled, x_unlabeled], y_labeled, epochs=100)

Note:
    The implementation follows Keras best practices and includes proper
    regularization, initialization, and normalization techniques.
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Tuple, Optional, Union, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.strong_augmentation import StrongAugmentation
from dl_techniques.losses.affine_invariant_loss import AffineInvariantLoss
from dl_techniques.losses.feature_alignment_loss import FeatureAlignmentLoss
from dl_techniques.models.vit.model import ViT

from .components import DPTDecoder

# Map depth_anything encoder_type slugs to ViT scale names.
_VIT_SCALE_MAP: Dict[str, str] = {
    "vit_s": "small",
    "vit_b": "base",
    "vit_l": "large",
}
_VIT_PATCH_SIZE: int = 16

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DepthAnything(keras.Model):
    """Depth Anything model implementation.

    Implements the complete Depth Anything architecture for monocular depth estimation.
    The model combines a feature encoder (placeholder for DINOv2) with a DPT decoder
    to produce dense depth predictions from RGB images.

    The architecture includes:
    - Feature encoder for extracting multi-scale representations
    - DPT decoder for dense prediction
    - Optional feature alignment with frozen encoder
    - Strong augmentation pipeline for robust training

    Args:
        encoder_type: String, type of ViT encoder to use.
            Supported values: ['vit_s', 'vit_b', 'vit_l'].
            Defaults to 'vit_l'.
        input_shape: Tuple of integers, input image shape as (height, width, channels).
            Defaults to (384, 384, 3).
        decoder_dims: List of integers, dimensions for decoder layers.
            Defaults to [256, 128, 64, 32].
        output_channels: Integer, number of output channels for depth prediction.
            Defaults to 1.
        kernel_initializer: String or Initializer, initializer for convolutional kernels.
            Defaults to "he_normal".
        kernel_regularizer: Regularizer or None, regularizer for convolutional kernels.
            Defaults to None.
        loss_weights: Dict of strings to floats, weights for different loss components.
            Keys: 'labeled', 'unlabeled', 'feature'.
            Defaults to {'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1}.
        cutmix_prob: Float, probability of applying CutMix augmentation.
            Defaults to 0.5.
        color_jitter_strength: Float, strength of color jittering augmentation.
            Defaults to 0.2.
        use_feature_alignment: Boolean, whether to use feature alignment loss.
            Defaults to True.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 3)`
        Or tuple of two 4D tensors for training with labeled/unlabeled data.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, output_channels)`

    Returns:
        A 4D tensor representing predicted depth maps.

    Raises:
        ValueError: If unsupported encoder type is specified.

    Example:
        >>> model = DepthAnything(
        ...     encoder_type='vit_l',
        ...     input_shape=(384, 384, 3),
        ...     decoder_dims=[256, 128, 64, 32]
        ... )
        >>> x = keras.random.normal([2, 384, 384, 3])
        >>> depth = model(x)
        >>> print(depth.shape)
        (2, 384, 384, 1)
    """

    def __init__(
        self,
        encoder_type: str = 'vit_l',
        image_shape: Tuple[int, int, int] = (384, 384, 3),
        decoder_dims: Optional[List[int]] = None,
        output_channels: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        cutmix_prob: float = 0.5,
        color_jitter_strength: float = 0.2,
        use_feature_alignment: bool = True,
        encoder_kind: str = 'real',
        enable_semi_supervised: bool = False,
        encoder: Optional[keras.Model] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Back-compat alias: legacy `input_shape=` kwarg maps to `image_shape`.
        if input_shape is not None:
            logger.info(
                "DepthAnything: 'input_shape' kwarg is deprecated; use 'image_shape'. "
                "Forwarding the value."
            )
            image_shape = input_shape

        # Validate encoder type
        self.supported_encoders = ['vit_s', 'vit_b', 'vit_l']
        if encoder_type not in self.supported_encoders:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. "
                f"Supported types: {self.supported_encoders}"
            )
        if encoder_kind not in ('real', 'placeholder'):
            raise ValueError(
                f"Unsupported encoder_kind: {encoder_kind}. Choose 'real' or 'placeholder'."
            )

        # Store configuration parameters
        self.encoder_type = encoder_type
        self.image_shape = tuple(image_shape)
        # Keep legacy attribute name for any external code reading it.
        self.input_shape_param = self.image_shape
        self.decoder_dims = decoder_dims if decoder_dims is not None else [256, 128, 64, 32]
        self.output_channels = output_channels
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.loss_weights = loss_weights if loss_weights is not None else {
            'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1
        }
        self.cutmix_prob = cutmix_prob
        self.color_jitter_strength = color_jitter_strength
        self.use_feature_alignment = use_feature_alignment
        self.encoder_kind = encoder_kind
        self.enable_semi_supervised = bool(enable_semi_supervised)

        # Encoder geometry: stride from patch_size for real ViT, 32 for placeholder
        # (initial stride-2 conv + 3 maxpools across 4 stages — last stage no pool).
        if self.encoder_kind == 'real':
            self.encoder_stride = _VIT_PATCH_SIZE  # 16
        else:
            # Placeholder Conv encoder: initial stride-2 conv + initial stride-2
            # maxpool => /4, then 3 stride-2 maxpools across stages 0..2
            # (stage 3 has no pool) => /8 ⇒ total stride 32.
            self.encoder_stride = 32
        self.encoder_h = self.image_shape[0] // self.encoder_stride
        self.encoder_w = self.image_shape[1] // self.encoder_stride

        # If an encoder was supplied (typically by `from_config` after
        # deserialization), accept it directly so its saved topology + weights
        # survive the load. Otherwise build() will create one fresh.
        self.encoder: Optional[keras.Model] = encoder
        self.encoder_embed_dim: Optional[int] = None
        if self.encoder_kind == 'real':
            scale = _VIT_SCALE_MAP[self.encoder_type]
            self.encoder_embed_dim = ViT.SCALE_CONFIGS[scale][0]

        # Other components — initialized in build().
        self.decoder: Optional[keras.layers.Layer] = None
        self.frozen_encoder: Optional[keras.Model] = None
        self.augmentation: Optional[keras.layers.Layer] = None

        logger.info(
            f"Initialized DepthAnything (encoder_type={encoder_type}, "
            f"encoder_kind={encoder_kind}, image_shape={self.image_shape}, "
            f"semi_supervised={self.enable_semi_supervised})"
        )

    def build(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) -> None:
        """Build the model components.

        Args:
            input_shape: Shape of input tensor(s).
        """

        # Construct the encoder if not already provided via from_config (which
        # passes a deserialized sub-Model directly). Building lazily inside
        # build() keeps the inner sub-Model under DepthAnything's tracking only
        # for fresh instantiations; for loaded models the encoder is already a
        # deserialized keras.Model with the saved topology + weights.
        if self.encoder is None:
            if self.encoder_kind == 'real':
                scale = _VIT_SCALE_MAP[self.encoder_type]
                self.encoder = ViT(
                    input_shape=self.image_shape,
                    scale=scale,
                    patch_size=_VIT_PATCH_SIZE,
                    include_top=False,
                    pooling=None,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'encoder_{self.encoder_type}_real',
                )
            else:
                self.encoder = self._create_placeholder_encoder(trainable=True)

        # Decoder: pass upsample_factor so the spatial output matches image_shape.
        # For real ViT (stride=16) with len(decoder_dims)>=4, upsample_factor=16 is
        # representable as 4 stages of 2x. For placeholder (stride=16 here) ditto.
        upsample_factor = self.encoder_stride
        self.decoder = DPTDecoder(
            dims=self.decoder_dims,
            output_channels=self.output_channels,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            upsample_factor=upsample_factor,
            name='dpt_decoder',
        )

        # Frozen weight-shared teacher. Order of operations matters:
        #   1) ensure the student encoder is built (so it has weights to copy).
        #   2) clone topology and force-build the clone.
        #   3) copy weights student → teacher and freeze.
        # Wrapped in try/except — if cloning fails on an exotic subclass we
        # disable feature alignment for the run rather than crash the model.
        if self.use_feature_alignment:
            try:
                dummy = keras.ops.zeros((1,) + tuple(self.image_shape))
                _ = self.encoder(dummy, training=False)
                self.frozen_encoder = keras.models.clone_model(self.encoder)
                _ = self.frozen_encoder(dummy, training=False)
                self.frozen_encoder.set_weights(self.encoder.get_weights())
                self.frozen_encoder.trainable = False
            except Exception as exc:  # pragma: no cover — diagnostic path
                logger.warning(
                    f"DepthAnything: clone_model(encoder) failed ({exc!r}); "
                    "disabling feature alignment for this run."
                )
                self.frozen_encoder = None
                self.use_feature_alignment = False

        # Strong augmentation pipeline (always available — module-level import).
        self.augmentation = StrongAugmentation(
            cutmix_prob=self.cutmix_prob,
            color_jitter_strength=self.color_jitter_strength,
            name='strong_augmentation',
        )

        super().build(input_shape)

    def update_teacher_ema(self, decay: float = 0.999) -> None:
        """Update the frozen teacher encoder via EMA over the student weights.

        Intended to be called from a Keras callback per training step. No-op when
        feature alignment is disabled or the frozen encoder was not built.

        Args:
            decay: EMA decay factor in ``[0,1]``. Higher values → slower update.
        """
        if self.frozen_encoder is None or not self.use_feature_alignment:
            return
        student_w = self.encoder.get_weights()
        teacher_w = self.frozen_encoder.get_weights()
        if len(student_w) != len(teacher_w):
            logger.warning(
                "update_teacher_ema: student/teacher weight counts differ; skipping."
            )
            return
        new_w = [decay * t + (1.0 - decay) * s for t, s in zip(teacher_w, student_w)]
        self.frozen_encoder.set_weights(new_w)

    def _features_to_spatial(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Convert a ViT-style ``(B, N+1, D)`` sequence into ``(B, h, w, D)``.

        Drops the CLS token and reshapes using the encoder geometry derived from
        ``image_shape`` and ``patch_size``. 4-D inputs are returned unchanged.
        """
        if len(x.shape) == 4:
            return x
        # (B, N+1, D) → drop CLS → (B, N, D) → reshape (B, h, w, D)
        x = x[:, 1:, :]
        d = self.encoder_embed_dim or x.shape[-1]
        return ops.reshape(x, (-1, self.encoder_h, self.encoder_w, d))

    def _create_placeholder_encoder(self, trainable: bool = True) -> keras.Model:
        """Create the placeholder Conv-BN-ReLU encoder (legacy mode).

        Used when ``encoder_kind='placeholder'``. For ``encoder_kind='real'``
        the actual ViT backbone is constructed eagerly in ``__init__``.

        Args:
            trainable: Boolean indicating whether the encoder should be trainable.

        Returns:
            Encoder model instance.
        """
        inputs = keras.layers.Input(shape=self.image_shape, name='encoder_input')

        # Initial convolution with proper initialization and regularization
        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False,
            name='initial_conv'
        )(inputs)
        x = keras.layers.BatchNormalization(name='initial_bn')(x)
        x = keras.layers.ReLU(name='initial_relu')(x)
        x = keras.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding='same',
            name='initial_pool'
        )(x)

        # Progressive feature extraction blocks
        dims = [64, 128, 256, 512]
        for i, dim in enumerate(dims):
            # First conv block
            x = keras.layers.Conv2D(
                filters=dim,
                kernel_size=3,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
                name=f'conv_block_{i}_1'
            )(x)
            x = keras.layers.BatchNormalization(name=f'bn_block_{i}_1')(x)
            x = keras.layers.ReLU(name=f'relu_block_{i}_1')(x)

            # Second conv block
            x = keras.layers.Conv2D(
                filters=dim,
                kernel_size=3,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
                name=f'conv_block_{i}_2'
            )(x)
            x = keras.layers.BatchNormalization(name=f'bn_block_{i}_2')(x)
            x = keras.layers.ReLU(name=f'relu_block_{i}_2')(x)

            # Downsample (except for last block to maintain spatial resolution)
            if i < len(dims) - 1:
                x = keras.layers.MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    padding='same',
                    name=f'pool_block_{i}'
                )(x)

        # Feature projection layer
        features = keras.layers.Conv2D(
            filters=self.decoder_dims[0],  # Match decoder input
            kernel_size=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False,
            name='feature_projection'
        )(x)

        encoder = keras.Model(
            inputs=inputs,
            outputs=features,
            name=f'encoder_{self.encoder_type}'
        )
        encoder.trainable = trainable

        return encoder

    def call(
        self,
        inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, 3)
                or tuple of (labeled, unlabeled) tensors for training.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Predicted depth maps with shape (batch_size, height, width, output_channels).
        """
        # Handle both single input and tuple input for training
        if isinstance(inputs, tuple):
            x_labeled, x_unlabeled = inputs
            # For simplicity, process labeled data in forward pass
            # Complex training logic would be handled in train_step
            x = x_labeled
        else:
            x = inputs

        # Apply augmentation during training. (`augmentation` is created in
        # build(); tests/users may set it to None to bypass strong aug.)
        if training and self.augmentation is not None:
            x = self.augmentation(x, training=training)

        # Extract features. ViT returns (B, N+1, D); placeholder returns 4-D.
        features = self.encoder(x, training=training)

        # Reshape sequence features to spatial 4-D before the decoder.
        features = self._features_to_spatial(features)

        # Decode features to depth.
        depth = self.decoder(features, training=training)

        return depth

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: Optional[keras.losses.Loss] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ) -> None:
        """Configure the model for training.

        Args:
            optimizer: Keras optimizer instance.
            loss: Primary loss function. If None, uses mean squared error.
            loss_weights: Optional custom loss weights to override defaults.
            **kwargs: Additional arguments passed to parent compile method.
        """
        # Set default loss if none provided
        if loss is None:
            loss = keras.losses.MeanSquaredError()

        super().compile(optimizer=optimizer, loss=loss, **kwargs)

        # Update loss weights if provided. Specialized loss instances are NOT
        # stored on `self` — that previously dead state caused get_config drift.
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        logger.info(f"Compiled DepthAnything with loss weights: {self.loss_weights}")

    def train_step(self, data: Any) -> Dict[str, keras.KerasTensor]:
        """Execute one training step.

        Two input shapes are accepted:

        * ``(x, y)`` — labeled-only path (default).
        * ``((x_lab, x_unlab), y_lab)`` — semi-supervised path. Active only
          when ``self.enable_semi_supervised`` is True AND
          ``self.use_feature_alignment`` is True. Adds a Feature-Alignment
          Loss term computed on unlabeled features against the
          weight-shared frozen teacher.

        Args:
            data: Training data batch.

        Returns:
            Dictionary containing loss metrics.
        """
        x, y = data
        # Detect semi-supervised tuple-of-inputs.
        if (
            self.enable_semi_supervised
            and isinstance(x, (tuple, list))
            and len(x) == 2
        ):
            x_lab, x_unlab = x[0], x[1]
        else:
            x_lab, x_unlab = x, None

        with tf.GradientTape() as tape:
            # Forward pass on labeled batch.
            y_pred = self(x_lab, training=True)
            # DECISION plan_2026-05-10_44694bc9/D-003: Keras-3 canonical train_step
            # — replaces deprecated compiled-loss / compiled-metrics calls.
            # See dl_techniques/models/masked_language_model/mlm.py:309-343.
            loss = self.compute_loss(x=x_lab, y=y, y_pred=y_pred)
            loss = loss * self.loss_weights.get('labeled', 1.0)

            # Optional Feature-Alignment-Loss on unlabeled batch.
            if (
                x_unlab is not None
                and self.use_feature_alignment
                and self.frozen_encoder is not None
            ):
                feat_student = self.encoder(x_unlab, training=True)
                feat_teacher = self.frozen_encoder(x_unlab, training=False)
                # Pool to (B, D). ViT seq output is (B, N+1, D); drop CLS.
                if len(feat_student.shape) == 4:
                    feat_student = ops.mean(feat_student, axis=[1, 2])
                    feat_teacher = ops.mean(feat_teacher, axis=[1, 2])
                elif len(feat_student.shape) == 3:
                    feat_student = ops.mean(feat_student[:, 1:, :], axis=1)
                    feat_teacher = ops.mean(feat_teacher[:, 1:, :], axis=1)
                fal = FeatureAlignmentLoss()(feat_teacher, feat_student)
                loss = loss + self.loss_weights.get('feature', 0.1) * fal

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update user-defined metrics (Keras 3 auto-tracks the loss in self.metrics).
        for m in self.metrics:
            if m.name != "loss":
                m.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "encoder_type": self.encoder_type,
            "image_shape": self.image_shape,
            "decoder_dims": self.decoder_dims,
            "output_channels": self.output_channels,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "loss_weights": self.loss_weights,
            "cutmix_prob": self.cutmix_prob,
            "color_jitter_strength": self.color_jitter_strength,
            "use_feature_alignment": self.use_feature_alignment,
            "encoder_kind": self.encoder_kind,
            "enable_semi_supervised": self.enable_semi_supervised,
            # Serialize the encoder sub-Model so save/load round-trips both
            # topology and weights through `.keras` archives. Mirrors the
            # MaskedLanguageModel pattern in mlm.py.
            "encoder": (
                keras.saving.serialize_keras_object(self.encoder)
                if self.encoder is not None
                else None
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DepthAnything':
        """Create model from configuration.

        Accepts both ``image_shape`` (current key) and ``input_shape``
        (legacy key from pre-bd098beb saved configs) for back-compat.

        Args:
            config: Dictionary containing model configuration.

        Returns:
            DepthAnything model instance.
        """
        cfg = dict(config)
        # Deserialize initializer/regularizer if present as serialized dicts.
        if isinstance(cfg.get("kernel_initializer"), dict):
            cfg["kernel_initializer"] = keras.initializers.deserialize(
                cfg["kernel_initializer"]
            )
        if isinstance(cfg.get("kernel_regularizer"), dict):
            cfg["kernel_regularizer"] = keras.regularizers.deserialize(
                cfg["kernel_regularizer"]
            )
        # Deserialize encoder sub-Model when present.
        enc_cfg = cfg.pop("encoder", None)
        if enc_cfg is not None:
            cfg["encoder"] = keras.saving.deserialize_keras_object(enc_cfg)
        return cls(**cfg)

    # ------------------------------------------------------------------
    # Save / load delegation for nested sub-Models.
    # ------------------------------------------------------------------
    # DECISION plan_2026-05-10_bd098beb/D-004
    # Keras 3 walks weight paths inside `.keras` archives via attribute
    # tracking on the outer `keras.Model` subclass. When `self.encoder`
    # is itself a Functional/subclassed `keras.Model` (here, ViT), the
    # path mapping for its inner FFN/attention Dense kernels can drift
    # between save and load — 55/172 weights round-trip with
    # re-initialised values (forward diff ≈ 1-2.8). The MLM serialization
    # pattern fixes topology round-trip but not weight-path round-trip.
    # The canonical Keras-3 fix is to override `save_own_variables` /
    # `load_own_variables` and persist the full ordered weight list of
    # each sub-Model into a deterministic keyed slot in the store. This
    # bypasses Keras' path-walking for these sub-Models entirely.
    def save_own_variables(self, store: Any) -> None:  # type: ignore[override]
        """Persist all of DepthAnything's variables in one flat store.

        The default Keras 3 implementation only persists ``self``'s own
        direct variables and lets the framework recurse into children. For
        ViT-as-encoder that recursion has been observed to drop kernel
        arrays during load (see D-004). We instead serialize the full,
        ordered ``self.weights`` list under flat numeric keys at the
        DepthAnything level. ``self.weights`` already includes every
        variable of every nested layer (encoder, frozen_encoder, decoder,
        augmentation), so this is one canonical, path-free record.
        """
        all_vars = list(self.weights)
        for i, v in enumerate(all_vars):
            store[str(i)] = keras.ops.convert_to_numpy(v)

    def load_own_variables(self, store: Any) -> None:  # type: ignore[override]
        """Restore all of DepthAnything's variables from the flat store.

        Mirrors :meth:`save_own_variables` — assigns ``self.weights[i]``
        from ``store[str(i)]`` in deterministic order. If sub-layers
        haven't been built yet (Keras 3 may call ``load_own_variables``
        before recursing into children), force-build by running a
        single dummy forward pass under the saved ``image_shape`` so
        ``self.weights`` matches what was written at save time.
        """
        if not self.built or any(
            sub is not None and not sub.built
            for sub in (self.encoder, self.frozen_encoder, self.decoder)
        ):
            dummy = keras.ops.zeros((1,) + tuple(self.image_shape))
            _ = self(dummy, training=False)

        all_vars = list(self.weights)
        n_store = len(store.keys()) if hasattr(store, "keys") else len(all_vars)
        if n_store != len(all_vars):
            raise ValueError(
                f"DepthAnything.load_own_variables: store has {n_store} "
                f"entries but model has {len(all_vars)} weights."
            )
        for i, v in enumerate(all_vars):
            v.assign(store[str(i)])

# ---------------------------------------------------------------------

def create_depth_anything(
    encoder_type: str = 'vit_l',
    image_shape: Tuple[int, int, int] = (384, 384, 3),
    decoder_dims: Optional[List[int]] = None,
    output_channels: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    cutmix_prob: float = 0.5,
    color_jitter_strength: float = 0.2,
    use_feature_alignment: bool = True,
    encoder_kind: str = 'real',
    enable_semi_supervised: bool = False,
    input_shape: Optional[Tuple[int, int, int]] = None,
) -> DepthAnything:
    """Create and build Depth Anything model instance.

    Args:
        encoder_type: String, type of ViT encoder to use.
            Supported values: ['vit_s', 'vit_b', 'vit_l'].
            Defaults to 'vit_l'.
        input_shape: Tuple of integers, input image shape as (height, width, channels).
            Defaults to (384, 384, 3).
        decoder_dims: List of integers, dimensions for decoder layers.
            Defaults to [256, 128, 64, 32].
        output_channels: Integer, number of output channels for depth prediction.
            Defaults to 1.
        kernel_initializer: String or Initializer, initializer for convolutional kernels.
            Defaults to "he_normal".
        kernel_regularizer: Regularizer or None, regularizer for convolutional kernels.
            Defaults to None.
        loss_weights: Dict of strings to floats, weights for different loss components.
            Keys: 'labeled', 'unlabeled', 'feature'.
            Defaults to {'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1}.
        cutmix_prob: Float, probability of applying CutMix augmentation.
            Defaults to 0.5.
        color_jitter_strength: Float, strength of color jittering augmentation.
            Defaults to 0.2.
        use_feature_alignment: Boolean, whether to use feature alignment loss.
            Defaults to True.

    Returns:
        Configured and built DepthAnything model instance.

    Raises:
        ValueError: If unsupported encoder type is specified.

    Example:
        >>> model = create_depth_anything(
        ...     encoder_type='vit_l',
        ...     input_shape=(384, 384, 3),
        ...     kernel_regularizer=keras.regularizers.L2(0.01)
        ... )
        >>> model.compile(
        ...     optimizer=keras.optimizers.AdamW(learning_rate=5e-6),
        ...     loss=keras.losses.MeanSquaredError()
        ... )
    """
    logger.info(f"Creating DepthAnything model with encoder: {encoder_type}")

    # Resolve image_shape (legacy 'input_shape' alias).
    if input_shape is not None:
        image_shape = input_shape

    # Create model with specified configuration
    model = DepthAnything(
        encoder_type=encoder_type,
        image_shape=image_shape,
        decoder_dims=decoder_dims,
        output_channels=output_channels,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        loss_weights=loss_weights,
        cutmix_prob=cutmix_prob,
        color_jitter_strength=color_jitter_strength,
        use_feature_alignment=use_feature_alignment,
        encoder_kind=encoder_kind,
        enable_semi_supervised=enable_semi_supervised,
    )

    # Build model with dummy input to initialize all components
    dummy_input = keras.random.normal([1] + list(image_shape))
    _ = model(dummy_input)

    logger.info("Successfully created and built DepthAnything model")

    return model

# ---------------------------------------------------------------------
