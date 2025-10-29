"""
CBAMNet Model Implementation with Pretrained Support
==================================================

A complete implementation of a CNN architecture featuring the Convolutional
Block Attention Module (CBAM), structured following modern Keras 3 best
practices for robust and serializable models.

Based on: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
https://arxiv.org/abs/1807.06521

Model Variants:
--------------
- CBAMNet-T (Tiny): [64, 128] dims
- CBAMNet-S (Small): [64, 128, 256] dims
- CBAMNet-B (Base): [128, 256, 512] dims

Usage Examples:
-------------
```python
# Create a model for CIFAR-10
model = CBAMNet(num_classes=10, dims=[64, 128], input_shape=(32, 32, 3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Create from variant
model = CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

# Load pretrained model as a feature extractor
model = CBAMNet.from_variant("small", include_top=False)

# Load from a local weights file
model = CBAMNet(num_classes=10, dims=[64, 128])
model.load_weights("path/to/cbamnet_weights.keras")
```
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
from dl_techniques.layers.attention.convolutional_block_attention import CBAM


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
        Input(shape=input_shape)
               ↓
    -----------------------------------
    | Stage i (for each dim in dims): |
    |   Conv2D(dim, 3, 'relu')        |
    |   BatchNormalization()          |
    |   CBAM(channels=dim)            |
    |   MaxPooling2D(2x2)             |
    -----------------------------------
               ↓
        GlobalAveragePooling2D()
               ↓
        Dense(num_classes, 'softmax') ← (if include_top=True)
               ↓
             Output
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
            and all values must be positive. Default: [64, 128].
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

    # --- Predefined Variants Configuration ---
    MODEL_VARIANTS: Dict[str, Dict[str, List[int]]] = {
        "tiny": {"dims": [64, 128]},
        "small": {"dims": [64, 128, 256]},
        "base": {"dims": [128, 256, 512]},
    }

    PRETRAINED_WEIGHTS: Dict[str, Dict[str, str]] = {
        "tiny": {"imagenet": "https://example.com/cbamnet_tiny_imagenet.keras"},
        "small": {"imagenet": "https://example.com/cbamnet_small_imagenet.keras"},
        "base": {"imagenet": "https://example.com/cbamnet_base_imagenet.keras"},
    }

    def __init__(
        self,
        num_classes: int = 1000,
        dims: List[int] = [64, 128],
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

        # --- Input Validation ---
        if include_top and num_classes <= 0:
            raise ValueError(f"num_classes must be positive when include_top=True, got {num_classes}")
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"dims must be a non-empty list of positive integers, got {dims}")
        if attention_ratio <= 0:
            raise ValueError(f"attention_ratio must be positive, got {attention_ratio}")
        if attention_kernel_size <= 0:
            raise ValueError(f"attention_kernel_size must be positive, got {attention_kernel_size}")

        # --- Store Configuration ---
        # These will be used in get_config() for serialization
        self.num_classes = num_classes
        self.dims = dims
        self.attention_ratio = attention_ratio
        self.attention_kernel_size = attention_kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer) if kernel_regularizer else None
        self.include_top = include_top
        self.input_shape_arg = input_shape

        # --- CREATE All Sub-layers in __init__ ---
        # Following modern Keras 3 pattern: instantiate all layers during initialization
        # Keras will handle building them automatically on first call

        # Build stages: Conv -> BN -> CBAM -> Pool
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
                keras.layers.BatchNormalization(name=f"stage_{i}_bn"),
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
            pretrained: If True, downloads and loads pretrained weights from
                the default URL. If a string path, loads weights from that path.
                Default: False (random initialization).
            weights_dataset: Dataset identifier for pretrained weights.
                Default: "imagenet". Used only if pretrained=True.
            **kwargs: Additional arguments passed to the model constructor
                (e.g., attention_ratio, kernel_regularizer).

        Returns:
            CBAMNet model instance with the specified variant configuration.

        Raises:
            ValueError: If variant name is not recognized.

        Example:
            ```python
            # Create tiny variant for CIFAR-10
            model = CBAMNet.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

            # Create base variant with pretrained ImageNet weights
            model = CBAMNet.from_variant("base", pretrained=True, num_classes=1000)

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

        # Get variant configuration
        variant_config = cls.MODEL_VARIANTS[variant]

        # Create model with variant dims and user-provided parameters
        model = cls(
            num_classes=num_classes,
            dims=variant_config["dims"],
            input_shape=input_shape,
            **kwargs
        )

        # Load pretrained weights if requested
        if pretrained:
            if isinstance(pretrained, str):
                # Load from provided path
                weights_path = pretrained
                print(f"Loading weights from {weights_path}...")
            else:
                # Download from URL
                try:
                    weights_path = cls._download_weights(variant, weights_dataset)
                    print(f"Downloaded weights from {weights_dataset}...")
                except Exception as e:
                    print(
                        f"Warning: Failed to download pretrained weights: {e}. "
                        "Model will be randomly initialized."
                    )
                    weights_path = None

            if weights_path:
                # Handle classifier mismatch when num_classes differs
                include_top = kwargs.get("include_top", True)
                pretrained_classes = 1000  # Assume ImageNet default
                skip_mismatch = include_top and (num_classes != pretrained_classes)

                try:
                    model.load_weights(
                        weights_path,
                        skip_mismatch=skip_mismatch,
                        by_name=True
                    )
                    suffix = " (classifier layer skipped)" if skip_mismatch else ""
                    print(f"✓ Loaded weights successfully{suffix}.")
                except Exception as e:
                    print(f"Warning: Failed to load weights: {e}")

        return model

    @staticmethod
    def _download_weights(variant: str, dataset: str = "imagenet") -> str:
        """
        Download pretrained weights from a URL.

        Args:
            variant: Model variant ("tiny", "small", "base").
            dataset: Dataset identifier (e.g., "imagenet").

        Returns:
            Local path to downloaded weights file.

        Raises:
            ValueError: If no weights URL is available for the variant/dataset combination.
        """
        if variant not in CBAMNet.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'"
            )

        if dataset not in CBAMNet.PRETRAINED_WEIGHTS[variant]:
            available_datasets = list(CBAMNet.PRETRAINED_WEIGHTS[variant].keys())
            raise ValueError(
                f"No pretrained weights for variant '{variant}' on dataset '{dataset}'. "
                f"Available datasets: {available_datasets}"
            )

        url = CBAMNet.PRETRAINED_WEIGHTS[variant][dataset]
        print(f"Downloading weights for CBAMNet-{variant} from {dataset}...")

        weights_path = keras.utils.get_file(
            fname=f"cbamnet_{variant}_{dataset}.keras",
            origin=url,
            cache_subdir="models/cbamnet"
        )

        return weights_path

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

        # With pretrained weights
        model = create_cbam_net("base", pretrained=True)
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