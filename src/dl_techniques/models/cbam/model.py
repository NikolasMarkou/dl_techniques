"""
Attention-augmented CNN backbone built on the Convolutional Block Attention Module.

This model embodies the principle of sequential attention refinement, a design
paradigm that treats feature recalibration as two separable questions applied in
order rather than one joint reweighting. The core idea is that a convolutional
feature map is over-complete in two independent senses: not every channel is
informative for a given input, and not every spatial location is informative
either. Asking both questions at once requires a 3-D attention tensor whose cost
scales with `C x H x W`; asking them one at a time factorizes the problem into a
`C`-dimensional and an `H x W`-dimensional decision, which is dramatically
cheaper and, empirically, more effective.

CBAM composes the two in sequence:

`F' = M_c(F) (*) F`
`F'' = M_s(F') (*) F'`

where `(*)` is element-wise multiplication with broadcasting. Channel attention
`M_c` pools each feature map to a scalar (both average and max pooling, whose
descriptors are complementary), passes them through a shared bottleneck MLP with
reduction ratio `r`, and sums the results before a sigmoid. The bottleneck is what
forces the module to learn inter-channel relationships rather than per-channel
gains. Spatial attention `M_s` pools across the channel axis instead, producing a
two-channel descriptor that a single large convolution (kernel size 7 by default)
maps to a spatial mask; the large receptive field matters here because deciding
whether a location is salient requires context beyond that location. Both masks
pass through a sigmoid, so attention can only attenuate or preserve, never invert
or amplify beyond the original magnitude.

Architecturally, the backbone is a stack of uniform stages, one per entry in
`dims`:

`Conv2D(dim, 3x3, relu) -> BatchNormalization -> CBAM(dim) -> MaxPooling2D(2x2)`

Placing CBAM after normalization and before pooling is deliberate: attention
operates on normalized activations, and downsampling then acts on an already
refined map, so the pooling decision is made over features whose salience has
been accounted for. The head is a global average pool followed by a softmax
`Dense`, and is omitted entirely when `include_top=False`, in which case the
model returns the final stage's feature maps and serves as a backbone for
detection, segmentation, or transfer learning.

Three preset variants trade capacity against cost: tiny (`[64, 128]`), small
(`[64, 128, 256]`), and base (`[128, 256, 512]`). Because each stage halves the
spatial resolution, depth is bounded below by input size; a 32x32 input supports
at most five stages before the feature map degenerates.

The implementation follows the Keras 3 subclassed-model contract. All sub-layers
are instantiated in `__init__` so that they exist before the first call and are
therefore captured correctly by serialization; no custom `build()` is required
because Keras builds sub-layers on the first invocation. `get_config` carries
every constructor argument, and `from_config` deserializes the initializer and
regularizer objects, so a saved model round-trips without reconstruction code.
No pretrained weights are distributed with this package. `pretrained=True`
raises `NotImplementedError` rather than warning and returning a randomly
initialized model, which is a deliberate choice: the previous behaviour held a
table of unreachable weight URLs and swallowed the download failure, making an
unavailable checkpoint silently indistinguishable from a successful load.
`from_variant(pretrained="<path>.keras")` loads a local checkpoint and
tolerates a classifier shape mismatch by name-matching and skipping the head,
which is the common case when fine-tuning an ImageNet checkpoint onto a
different label set.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module. ECCV 2018.
      (https://arxiv.org/abs/1807.06521)
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
      (https://arxiv.org/abs/1709.01507)
    - Park et al., 2018. BAM: Bottleneck Attention Module.
      (https://arxiv.org/abs/1807.06514)
    - Ioffe and Szegedy, 2015. Batch Normalization: Accelerating Deep Network
      Training by Reducing Internal Covariate Shift.
      (https://arxiv.org/abs/1502.03167)

"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.attention.convolutional_block_attention import CBAM
from dl_techniques.utils.model_build import materialize_sublayers


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CBAMNet(keras.Model):
    """
    CNN model with CBAM attention for image classification.

    This model integrates CBAM (Convolutional Block Attention Module) after
    convolutional blocks to refine feature maps by focusing on salient channels
    and spatial regions. It provides a flexible, configurable architecture with
    proper serialization support for production deployment.

    **Intent**: Provide a robust attention-augmented CNN architecture that
    demonstrates modern Keras 3 model implementation patterns while delivering
    strong performance through spatial and channel attention mechanisms.

    **Architecture**:
    ```
    ┌─────────────────────────────────┐
    │   Input(shape=input_shape)      │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │ Stage i (for each dim in dims): │
    │   Conv2D(dim, 3, 'relu')        │
    │   BatchNormalization()          │
    │   CBAM(channels=dim)            │
    │   MaxPooling2D(2x2)             │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │    GlobalAveragePooling2D()     │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │  Dense(num_classes, 'softmax')  │
    │    ← (if include_top=True)      │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │           Output                │
    └─────────────────────────────────┘
    ```

    **Mathematical Operations**:
    - Convolution: output = activation(W * input + b)
    - CBAM Attention: output = Ms(Mc(F) ⊗ F) ⊗ (Mc(F) ⊗ F)
      where Mc is channel attention, Ms is spatial attention, ⊗ is element-wise multiplication
    - Softmax Classification: P(class_i) = exp(z_i) / Σ_j exp(z_j)

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if `include_top=True`. Must be positive.
        dims: List of integers, channel dimensions for each stage.
            Each value creates one Conv-BN-CBAM-Pool stage. Must not be empty
            and all values must be positive. `None` (default) resolves to
            [64, 128].
        attention_ratio: Integer, reduction ratio for channel attention MLP
            in CBAM blocks. Controls the compression of channel dimension in
            the attention mechanism. Must be positive. Default: 8.
        attention_kernel_size: Integer, kernel size for spatial attention
            convolution in CBAM blocks. Must be positive and odd. Default: 7.
        kernel_initializer: Initializer for Conv2D and Dense layer kernels.
            Can be a string identifier or Initializer instance. Default: 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for Conv2D and Dense kernels.
            Can be a string identifier or Regularizer instance. Default: None.
        include_top: Boolean, whether to include the classification head.
            If False, returns feature maps from final stage. Default: True.
        input_shape: Optional tuple, input shape excluding batch dimension,
            e.g., (height, width, channels). If None, shape is inferred on first call.
        **kwargs: Additional arguments for Model base class (e.g., name).

    Input shape:
        4D tensor with shape `(batch_size, height, width, channels)`.

    Output shape:
        - If `include_top=True`: 2D tensor `(batch_size, num_classes)` with class probabilities.
        - If `include_top=False`: 4D tensor `(batch_size, H', W', dims[-1])` with feature maps.

    Attributes:
        stages: List of lists, each inner list contains layers for one stage
            (Conv2D, BatchNormalization, CBAM, MaxPooling2D).
        head: List of layers for classification head (GlobalAveragePooling2D, Dense).
            Empty if `include_top=False`.

    Raises:
        ValueError: If `num_classes` <= 0 when `include_top=True`.
        ValueError: If `dims` is empty or contains non-positive values.
        ValueError: If `attention_ratio` <= 0.
        ValueError: If `attention_kernel_size` <= 0.

    Example:
        ```python
        # CIFAR-10 classifier with tiny variant
        model = CBAMNet(num_classes=10, dims=[64, 128], input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # ImageNet feature extractor with base variant
        model = CBAMNet(
            dims=[128, 256, 512],
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Custom architecture with regularization
        model = CBAMNet(
            num_classes=100,
            dims=[64, 128, 256, 512],
            attention_ratio=16,
            kernel_regularizer='l2',
            input_shape=(128, 128, 3)
        )
        ```

    Note:
        As a Model subclass, Keras automatically handles building of sub-layers
        on the first call. No custom build() method is needed. All sub-layers
        are created in __init__() for proper serialization support.
    """

    MODEL_VARIANTS: Dict[str, Dict[str, List[int]]] = {
        "tiny": {"dims": [64, 128]},
        "small": {"dims": [64, 128, 256]},
        "base": {"dims": [128, 256, 512]},
    }

    def __init__(
        self,
        num_classes: int = 1000,
        dims: Optional[List[int]] = None,
        attention_ratio: int = 8,
        attention_kernel_size: int = 7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        include_top: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize CBAMNet model with specified architecture and configuration."""
        super().__init__(**kwargs)

        if dims is None:
            dims = [64, 128]

        if include_top and num_classes <= 0:
            raise ValueError(f"num_classes must be positive when include_top=True, got {num_classes}")
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"dims must be a non-empty list of positive integers, got {dims}")
        if attention_ratio <= 0:
            raise ValueError(f"attention_ratio must be positive, got {attention_ratio}")
        if attention_kernel_size <= 0:
            raise ValueError(f"attention_kernel_size must be positive, got {attention_kernel_size}")

        self.num_classes = num_classes
        self.dims = dims
        self.attention_ratio = attention_ratio
        self.attention_kernel_size = attention_kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer) if kernel_regularizer else None
        self.include_top = include_top
        self.input_shape_arg = input_shape

        # Sub-layers are created here (not in build) so they exist before the
        # first call and are captured by serialization.
        self.stages: List[List[keras.layers.Layer]] = []
        for i, dim in enumerate(self.dims):
            stage_layers = [
                keras.layers.Conv2D(
                    filters=dim,
                    kernel_size=3,
                    activation='relu',
                    padding='same',
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"stage_{i}_conv"
                ),
                # DECISION plan-2026-08-22T035419-a11304c8/D-111
                # This BatchNorm keeps Keras' documented defaults
                # (momentum=0.99, epsilon=1e-3) and that is a RULING, not an
                # oversight. R-083 asks that a numeric trace to a named external
                # reference; here there is none to trace to. Woo et al. 2018
                # (CBAM) specifies an attention MODULE, not a backbone -- the
                # paper's experiments bolt CBAM onto ResNet/WideResNet/MobileNet
                # and inherit each host's BatchNorm settings, so "the CBAM
                # momentum" does not exist as a published quantity. This stage
                # stack is this repository's own demonstration backbone, so the
                # framework default is the honest provenance. Do NOT copy
                # MobileNet's `REFERENCE_BN_MOMENTUM` here: that constant is
                # traced to the TF Model Garden MobileNet backbone and means
                # nothing outside it. If you host CBAM inside a real backbone,
                # take that backbone's value.
                # See decisions.md D-111.
                keras.layers.BatchNormalization(
                    momentum=0.99,
                    epsilon=1e-3,
                    name=f"stage_{i}_bn",
                ),
                CBAM(
                    channels=dim,
                    ratio=self.attention_ratio,
                    kernel_size=self.attention_kernel_size,
                    name=f"stage_{i}_cbam"
                ),
                keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"stage_{i}_pool")
            ]
            self.stages.append(stage_layers)

        # Build classification head (if requested)
        self.head: List[keras.layers.Layer] = []
        if self.include_top:
            self.head.append(
                keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
            )
            if self.num_classes > 0:
                self.head.append(
                    keras.layers.Dense(
                        units=self.num_classes,
                        activation='softmax',
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name="classifier"
                    )
                )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method CBAMNet inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        Args:
            input_shape: Shape (or nest of shapes) of the input to ``call``.
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
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            training: Boolean or None, whether the model is in training mode.
                Affects BatchNormalization and Dropout layers.

        Returns:
            Output tensor. Shape depends on `include_top`:
            - If True: (batch_size, num_classes) with class probabilities
            - If False: (batch_size, H', W', dims[-1]) with feature maps
        """
        x = inputs

        # Pass through all stages
        for stage_layers in self.stages:
            for layer in stage_layers:
                x = layer(x, training=training)

        # Pass through classification head if present
        if self.include_top:
            for layer in self.head:
                x = layer(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns ALL constructor parameters needed to reconstruct the model.
        This is called during model.save() and must include every parameter
        that was passed to __init__().

        Returns:
            Dictionary containing all model configuration parameters.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "dims": self.dims,
            "attention_ratio": self.attention_ratio,
            "attention_kernel_size": self.attention_kernel_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "include_top": self.include_top,
            "input_shape": self.input_shape_arg,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CBAMNet":
        """
        Create model instance from configuration dictionary.

        This is called during model loading (keras.models.load_model).
        It deserializes configuration objects like initializers and regularizers
        back into their proper types before passing to __init__().

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New CBAMNet instance with the specified configuration.
        """
        # Deserialize initializers and regularizers from their serialized form
        if "kernel_initializer" in config:
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        if "kernel_regularizer" in config:
            config["kernel_regularizer"] = keras.regularizers.deserialize(config["kernel_regularizer"])

        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        **kwargs: Any
    ) -> "CBAMNet":
        """
        Create a CBAMNet model from a predefined variant.

        This convenience method allows creating standard model configurations
        using simple variant names, optionally with pretrained weights.

        Args:
            variant: Model variant name. One of: "tiny", "small", "base".
            num_classes: Number of output classes. Default: 1000 (ImageNet).
            input_shape: Input shape tuple (height, width, channels).
                Default: (224, 224, 3).
            pretrained: If a string path, loads weights from that path. If
                True, raises NotImplementedError -- no public CBAMNet weights
                ship with dl_techniques. Default: False (random
                initialization).
            weights_dataset: Dataset identifier for pretrained weights.
                Default: "imagenet". Used only if pretrained=True.
            **kwargs: Additional arguments passed to the model constructor
                (e.g., attention_ratio, kernel_regularizer).

        Returns:
            CBAMNet model instance with the specified variant configuration.

        Raises:
            ValueError: If variant name is not recognized.
            NotImplementedError: If pretrained is True.

        Example:
            ```python
            # Create tiny variant for CIFAR-10
            model = CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

            # Create small variant as feature extractor
            model = CBAMNet.from_variant("small", include_top=False)

            # Load from local weights
            model = CBAMNet.from_variant("tiny", pretrained="path/to/weights.keras")
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            available = list(cls.MODEL_VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {available}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127
        # House style (`wave_field/model.py`): copy the preset, drop the
        # metadata key, then `config.update(kwargs)`. Do NOT go back to
        # splatting named preset fields alongside `**kwargs` -- every
        # documented override of one of those fields raised
        # `TypeError: got multiple values for keyword argument`
        # (MEASURED at all six sites). The `.copy()` is NOT optional and
        # NOT cosmetic: `config.update(kwargs)` on the shared
        # `MODEL_VARIANTS[variant]` dict would permanently poison the
        # class-level table for every later caller. See decisions.md D-127.
        variant_config = cls.MODEL_VARIANTS[variant].copy()
        variant_config.pop("description", None)
        variant_config.update(kwargs)

        model = cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **variant_config
        )

        if pretrained:
            if isinstance(pretrained, str):
                weights_path = pretrained
                logger.info(f"Loading weights from {weights_path}...")
            else:
                weights_path = cls._download_weights(variant, weights_dataset)

            # The ImageNet head is 1000-wide; a different num_classes means the
            # classifier weights must be skipped rather than refused.
            include_top = kwargs.get("include_top", True)
            skip_mismatch = include_top and (num_classes != 1000)

            # `load_weights_from_checkpoint` rather than
            # `model.load_weights(..., by_name=True)`: Keras 3.8's
            # `Model.load_weights` rejects `by_name` outright
            # (`ValueError: Invalid keyword arguments: {'by_name': True}`), so
            # the local-path route never worked. The failure was invisible
            # because the call sat inside a `try/except` that logged a warning,
            # which is the same swallow that made `pretrained=True` return an
            # untrained model. Layer-by-layer transfer is the repo's canonical
            # replacement.
            if not model.built:
                model(keras.ops.zeros((1,) + tuple(model.input_shape_arg or (32, 32, 3))))
            report = load_weights_from_checkpoint(
                target=model,
                ckpt_path=weights_path,
                skip_prefixes=("classifier",) if skip_mismatch else (),
                strict=not skip_mismatch,
            )
            logger.info(report.summary_string())

        return model

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs on
    # a non-existent host; `from_variant` caught the download failure, logged a
    # warning and continued with random initialization, so `pretrained=True`
    # silently returned an untrained model.
    # Do NOT reinstate a warn-and-return branch here or in `from_variant`. No
    # public CBAMNet weights are distributed with dl_techniques; pass a local
    # path via `pretrained="/path/to/file.keras"` or use `pretrained=False`.
    @staticmethod
    def _download_weights(variant: str, dataset: str = "imagenet") -> str:
        """
        Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public CBAMNet weights ship with dl_techniques.
        Always raises. Kept so `pretrained=True` fails loudly instead of
        silently returning a randomly-initialized model.

        Args:
            variant: Model variant name (unused).
            dataset: Dataset identifier (unused).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained CBAMNet weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: CBAMNet.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

# ---------------------------------------------------------------------

def create_cbam_net(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
    pretrained: Union[bool, str] = False,
    **kwargs: Any
) -> CBAMNet:
    """
    Convenience function to create a CBAMNet model.

    This is a simple wrapper around CBAMNet.from_variant() for more
    concise model creation in scripts and experiments.

    Args:
        variant: Model variant ("tiny", "small", "base"). Default: "tiny".
        num_classes: Number of output classes. Default: 1000.
        input_shape: Input shape tuple (height, width, channels).
            Default: (224, 224, 3).
        pretrained: If True, loads pretrained weights. If string,
            loads from the specified path. Default: False.
        **kwargs: Additional arguments for the model constructor.

    Returns:
        CBAMNet model instance.

    Example:
        ```python
        # Quick model creation for CIFAR-10
        model = create_cbam_net("tiny", num_classes=10, input_shape=(32, 32, 3))

        # Warm-start from a local checkpoint (`pretrained=True` raises
        # NotImplementedError -- no public CBAMNet weights are distributed)
        model = create_cbam_net("base", pretrained="/path/to/weights.keras")
        ```
    """
    return CBAMNet.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        **kwargs
    )

# ---------------------------------------------------------------------