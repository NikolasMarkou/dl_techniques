# DL-Techniques Project Knowledge Base

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Development Environment](#development-environment)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Available Components](#available-components)
    - [Constraints](#constraints)
    - [Initializers](#initializers)
    - [Layers](#layers)
        - [Core Layers](#core-layers)
        - [Activations](#activations)
        - [Feed-Forward Networks (FFN)](#feed-forward-networks-ffn)
        - [Logic Gates](#logic-gates)
        - [Normalization Layers](#normalization-layers)
    - [Losses](#losses)
    - [Metrics](#metrics)
    - [Models](#models)
    - [Optimization](#optimization)
    - [Regularizers](#regularizers)
    - [Utilities](#utilities)
        - [Core Utilities](#core-utilities)
        - [Dataset Utilities](#dataset-utilities)
    - [WeightWatcher](#weightwatcher)
6. [Training Techniques](#training-techniques)
7. [Best Practices](#best-practices)
8. [Technical Specifications](#technical-specifications)

---

## Project Overview

DL-Techniques is a comprehensive deep learning library built on **Keras 3.8.0** and **TensorFlow 2.18.0**, providing advanced neural network components including custom layers, models, losses, metrics, initializers, regularizers, and optimization utilities. The library emphasizes:

- **Backend-agnostic implementation** using `keras.ops` for cross-platform compatibility (TensorFlow, JAX, PyTorch).
- **Proper serialization** with `@keras.saving.register_keras_serializable()` to ensure models can be saved and loaded reliably.
- **Comprehensive testing** for reliability and correctness.
- **Modern architectures** including Vision Transformers (ViT), ConvNeXt, Capsule Networks (CapsNet), Self-Organizing Maps (SOM), and efficient mobile networks like MobileNetV4.
- **Advanced techniques** such as Differential Attention, Kolmogorov-Arnold Networks (KAN), Matrix Product States (MPS), and Shearlet Transforms for specialized applications.
- **Extensive analysis and visualization tools**, including an integrated WeightWatcher module for deep spectral analysis of weight matrices.

## Project Structure

### Detailed Directory Tree
The project is organized into a modular structure to separate concerns and improve maintainability.

```
└── src/
    ├── dl_techniques/
    │   ├── constraints/          # Custom weight constraints
    │   ├── initializers/         # Custom weight initializers
    │   ├── layers/               # Custom Keras layer implementations
    │   │   ├── activations/      # Specialized activation function layers
    │   │   ├── ffn/              # Feed-forward network variants (gMLP, SwiGLU, etc.)
    │   │   ├── logic/            # Fuzzy and standard logic gate layers
    │   │   └── norms/            # Advanced normalization layers
    │   ├── losses/               # Custom loss functions for various tasks
    │   ├── metrics/              # Custom metrics for model evaluation
    │   ├── models/               # Complete model architectures (CapsNet, ViT, etc.)
    │   ├── optimization/         # Optimizers and learning rate schedules
    │   ├── regularizers/         # Custom regularization functions
    │   ├── utils/                # General-purpose utility functions
    │   │   └── datasets/         # Dataset loading and preprocessing utilities
    │   └── weightwatcher/        # Tools for deep weight matrix analysis
    ├── experiments/              # Scripts for experimental validation and testing
    └── train/                    # Training scripts for specific models
```

## Development Environment

### Dependencies
```
numpy~=2.0.2
pytest~=8.3.4
pytest-cov
matplotlib~=3.10.0
scikit-learn~=1.6.1
keras~=3.8.0
tensorflow==2.18.0
tqdm
seaborn~=0.13.2
scipy~=1.15.1
setuptools~=65.5.1
pandas~=2.2.3
```

### Key Requirements
- Python 3.11
- Keras 3.8.0 (prefer Keras-only code over TensorFlow primitives)
- TensorFlow 2.18.0 backend
- Model files saved in `.keras` format
- Type hinting throughout
- Sphinx-compliant docstrings

### Logging
Use the project logger instead of print statements:
```python
from dl_techniques.utils.logger import logger

# Logger implementation:
import logging

LOGGER_FORMAT = \
    "%(asctime)s %(levelname)-4s %(filename)s:%(funcName)s:%(lineno)s] " \
    "%(message)s"

logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
logging.getLogger("dl").setLevel(logging.INFO)
logger = logging.getLogger("dl")

# Usage:
logger.info("Training started")
logger.debug("Detailed information")
logger.warning("Potential issue")
```

## Architecture Guidelines

### Documentation Style Examples
The project follows comprehensive documentation patterns as shown in these Keras layer examples:

#### Conv2D Documentation Pattern
```python
class Conv2D(BaseConv):
    """2D convolution layer.

    This layer creates a convolution kernel that is convolved with the layer
    input over a 2D spatial (or temporal) dimension (height and width) to
    produce a tensor of outputs. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`, it is
    applied to the outputs as well.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
        data_format: string, either `"channels_last"` or `"channels_first"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        groups: A positive int specifying the number of groups.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function for the kernel.
        bias_constraint: Optional projection function for the bias.

    Input shape:
        - If `data_format="channels_last"`:
            A 4D tensor with shape: `(batch_size, height, width, channels)`
        - If `data_format="channels_first"`:
            A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:
        - If `data_format="channels_last"`:
            A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
        - If `data_format="channels_first"`:
            A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

    Returns:
        A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:
        >>> x = np.random.rand(4, 10, 10, 128)
        >>> y = keras.layers.Conv2D(32, 3, activation='relu')(x)
        >>> print(y.shape)
        (4, 8, 8, 32)
    """
```

### Layer Design Principles
1. **Initialization (`__init__`)**:
   - Call `super().__init__(**kwargs)` to handle base layer parameters.
   - Store configuration as instance attributes after validation.
   - Initialize weights to `None`; they will be created in `build`.
   - Use type hints and comprehensive docstrings.
   - Make regularizers and initializers customizable.

2. **Building (`build`)**:
   - Create weights lazily in `build()` once the input shape is known.
   - Initialize complex sublayers within `build()`.
   - Store `self._build_input_shape = input_shape` for serialization.
   - Use `self.add_weight()` for creating trainable parameters.
   - Call `super().build(input_shape)` at the end of the method.

3. **Forward Computation (`call`)**:
   - Implement the core logic in the `call(inputs, training=None)` method.
   - Use `keras.ops` for all tensor operations to ensure backend compatibility.
   - Propagate the `training` parameter to sublayers that behave differently during training and inference (e.g., `Dropout`, `BatchNormalization`).

4. **Serialization (`get_config`, etc.)**:
   - Implement `get_config()` to return a serializable dictionary of constructor parameters.
   - Implement `get_build_config()` to return the `input_shape` needed for rebuilding.
   - Implement `build_from_config(config)` to correctly reconstruct the layer from its build configuration.
   - Decorate the class with `@keras.saving.register_keras_serializable()` for custom objects.

## Available Components

### Constraints
-   #### `ValueRangeConstraint`
    -   **File**: `src/dl_techniques/constraints/value_range_constraint.py`
    -   **Type**: Keras Constraint
    -   **Description**: Constrains weights to a specified range `[min_value, max_value]`. Useful for preventing vanishing/exploding weights, ensuring numerical stability, or enforcing architectural requirements like non-negative weights.
    -   **Key Parameters (`__init__`)**:
        -   `min_value` (float): The minimum allowed value for the weights.
        -   `max_value` (Optional[float]): The maximum allowed value. If `None`, only a minimum constraint is applied.
        -   `clip_gradients` (bool): A legacy parameter, kept for API compatibility. The clipping is inherent to the constraint's operation.
    -   **Core Logic (`call` method)**:
        1.  Applies `ops.maximum(weights, self.min_value)` to enforce the lower bound.
        2.  If `max_value` is specified, it subsequently applies `ops.minimum(constrained, self.max_value)` to enforce the upper bound.

### Initializers
-   #### `HaarWaveletInitializer`
    -   **File**: `src/dl_techniques/initializers/haar_wavelet_initializer.py`
    -   **Type**: Keras Initializer
    -   **Description**: Generates 2D Haar wavelet decomposition filters (LL, LH, HL, HH). These filters are useful for multi-resolution analysis and feature extraction in computer vision, particularly in wavelet-based CNNs. The initializer creates an orthonormal basis that preserves energy.
    -   **Key Parameters (`__init__`)**:
        -   `scale` (float): A scaling factor for the wavelet coefficients. Must be positive.
    -   **Core Logic (`call` method)**:
        1.  Defines the four standard 2D Haar wavelet patterns as a `numpy` array.
        2.  Scales these patterns by the `self.scale` factor.
        3.  Initializes a weight kernel of the requested `shape`.
        4.  Cycles through the four patterns, populating the output channels of the kernel for each input channel. This ensures that a `DepthwiseConv2D` layer with a `channel_multiplier` of 4 will apply the full set of Haar filters to each input channel.
        5.  Converts the final `numpy` kernel to a tensor using `ops.convert_to_tensor`.

-   #### `HeOrthonormalInitializer`
    -   **File**: `src/dl_techniques/initializers/he_orthonormal_initializer.py`
    -   **Type**: Keras Initializer
    -   **Description**: Combines He normal initialization with QR decomposition to produce a set of orthonormal vectors. This leverages the variance-scaling benefits of He initialization (suitable for ReLU-like activations) while ensuring the geometric properties of an orthogonal basis.
    -   **Key Parameters (`__init__`)**:
        -   `seed` (Optional[int]): A random seed for reproducibility.
    -   **Core Logic (`call` method)**:
        1.  Validates that the requested `shape` is 2D and that the number of vectors does not exceed the feature dimensionality.
        2.  Generates an initial matrix using `keras.initializers.HeNormal`.
        3.  Transposes the matrix to shape `(feature_dims, n_clusters)`.
        4.  Applies QR decomposition using `ops.linalg.qr`. The resulting `Q` matrix has orthonormal columns.
        5.  For deterministic output, it applies a sign correction to `Q` based on the signs of the diagonal elements of the `R` matrix.
        6.  Transposes the corrected `Q` matrix back to `(n_clusters, feature_dims)` so that the rows are orthonormal vectors.

-   #### `OrthonormalInitializer`
    -   **File**: `src/dl_techniques/initializers/orthonormal_initializer.py`
    -   **Type**: Keras Initializer
    -   **Description**: Generates a set of orthonormal vectors (orthogonal and unit length) using QR decomposition on a random matrix. This is useful for initializing layers where decorrelated features are desired, such as clustering centroids or weights in deep networks to mitigate gradient issues.
    -   **Key Parameters (`__init__`)**:
        -   `seed` (Optional[int]): A random seed for reproducibility.
    -   **Core Logic (`call` method)**:
        1.  Validates that the requested `shape` is 2D and that the number of vectors does not exceed the feature dimensionality.
        2.  Generates a random square matrix of shape `(feature_dims, feature_dims)` using `numpy.random.RandomState`.
        3.  Converts the numpy matrix to a tensor using `ops.convert_to_tensor`.
        4.  Computes the QR decomposition using `ops.linalg.qr`.
        5.  Applies a sign correction based on the first row of the `Q` matrix to ensure deterministic output for a given seed.
        6.  Extracts the first `n_clusters` rows from the corrected `Q` matrix to get the final set of orthonormal vectors.

### Layers

#### Core Layers
-   #### `AdaptiveSoftmax`
    -   **File**: `src/dl_techniques/layers/adaptive_softmax.py`
    -   **Type**: Keras Layer
    -   **Description**: An enhanced softmax layer that dynamically adjusts its temperature based on the Shannon entropy of the input logits' distribution. This helps to maintain sharp, focused probability distributions, especially for large output spaces, mitigating the "dispersion effect" of standard softmax.
    -   **Key Parameters (`__init__`)**:
        -   `min_temp`, `max_temp` (float): The allowable range for the adaptive temperature.
        -   `entropy_threshold` (float): The entropy value above which temperature adaptation is applied.
        -   `polynomial_coeffs` (List[float]): Coefficients for the polynomial function that maps entropy to temperature.
    -   **Core Logic (`call` method)**:
        1.  Computes an initial probability distribution `p` using `ops.nn.softmax`.
        2.  Calculates the Shannon entropy `H = -Σ p_i * log(p_i)` of this distribution.
        3.  If `H` is above `entropy_threshold`, it computes a new temperature `T` by evaluating a polynomial function of `H` and clamping the result.
        4.  If `H` is below the threshold, `T` is set to 1.0 (standard softmax behavior).
        5.  The final output is computed as `ops.nn.softmax(logits / T)`.

-   #### `AdaptiveMultiHeadAttention`
    -   **File**: `src/dl_techniques/layers/adaptive_softmax_mha.py`
    -   **Type**: Keras Layer
    -   **Description**: A multi-head attention layer that replaces the standard softmax for score normalization with the `AdaptiveTemperatureSoftmax` layer. This allows the attention mechanism to dynamically sharpen or soften its focus based on the context provided by the attention scores.
    -   **Key Parameters (`__init__`)**: Inherits all `MultiHeadAttention` parameters and adds those from `AdaptiveTemperatureSoftmax` (`min_temp`, `max_temp`, etc.).
    -   **Core Logic (`call` method)**:
        1.  Standard MHA projections for Q, K, V.
        2.  Computes raw attention scores: `scores = matmul(Q, K^T) / sqrt(d_k)`.
        3.  Applies the attention mask by setting masked positions to a large negative number.
        4.  Instead of `ops.softmax`, it passes the scores through the internal `_adaptive_softmax` layer instance.
        5.  The resulting probabilities are used to compute the weighted sum of the V vectors: `output = matmul(probabilities, V)`.

-   #### `BatchConditionalOutputLayer`
    -   **File**: `src/dl_techniques/layers/batch_conditional_layer.py`
    -   **Type**: Keras Layer
    -   **Description**: A layer that performs batch-wise conditional selection. For each item in a batch, it outputs an `inference` tensor if the corresponding `ground_truth` tensor is all zeros; otherwise, it outputs the `ground_truth` tensor. This is useful in scenarios like teacher forcing or conditional data mixing.
    -   **Key Parameters (`__init__`)**: Standard Keras layer parameters.
    -   **Core Logic (`call` method)**:
        1.  Takes a list of two tensors: `[ground_truth, inference]`.
        2.  Checks for all-zero entries in the `ground_truth` tensor along all axes except the batch axis using `tf.reduce_all(tf.equal(ground_truth, 0), axis=reduction_axes)`.
        3.  Creates a boolean mask from this check.
        4.  Uses `tf.where(mask, inference, ground_truth)` to select the output for each batch item.

-   #### `BitLinear`
    -   **File**: `src/dl_techniques/layers/bitlinear_layer.py`
    -   **Type**: Keras Layer
    -   **Description**: An implementation of a 1.58-bit Linear layer, as proposed in "The Era of 1-bit LLMs." It uses quantization-aware training to simulate low-bit weights and activations.
    -   **Key Parameters (`__init__`)**:
        -   `bit_config` (`BitLinearConfig`): A dataclass holding parameters for dimensions, quantization ranges, and measurement methods (e.g., `AbsMax`, `AbsMedian`).
    -   **Core Logic (`call` method)**:
        1.  **Input Quantization**: Scales the input tensor using a measure function (e.g., `abs_max`) and quantizes it using a chosen strategy (e.g., `round_clamp`).
        2.  **Weight Quantization**: Performs the same scaling and quantization on the layer's kernel weights. This is done on-the-fly.
        3.  **Linear Transformation**: Performs `tf.matmul` using the quantized inputs and quantized weights.
        4.  **Rescaling**: Divides the output by the product of the input and weight scaling factors to return the result to the original scale.
        5.  **Gradient Flow**: Uses a Straight-Through Estimator (STE) within the quantization functions (`round_clamp`, `sample`) to allow gradients to pass through the non-differentiable rounding operations.

-   #### `Canny`
    -   **File**: `src/dl_techniques/layers/canny.py`
    -   **Type**: TensorFlow Module (not a Keras Layer)
    -   **Description**: A TensorFlow implementation of the Canny edge detection algorithm. It's structured as a module that can be called like a function. It is not a trainable layer but a fixed feature extractor.
    -   **Key Parameters (`__init__`)**:
        -   `sigma`: Standard deviation for the initial Gaussian blur.
        -   `threshold_min`, `threshold_max`: Thresholds for hysteresis edge tracking.
    -   **Core Logic (`call` method)**:
        1.  **Noise Reduction**: Applies Gaussian smoothing using `tf.nn.convolution` with a pre-computed Gaussian kernel.
        2.  **Gradient Calculation**: Computes gradients using Sobel operators, also via `tf.nn.convolution`. It then calculates gradient magnitude (`ops.sqrt`) and direction (`ops.atan2`).
        3.  **Non-Maximum Suppression**: Thins edges by keeping only local maxima in the gradient direction.
        4.  **Double Thresholding**: Identifies strong and weak edge pixels based on `threshold_max` and `threshold_min`.
        5.  **Hysteresis Edge Tracking**: Connects weak edges to strong edges using `tf.nn.dilation2d` iteratively in a `tf.while_loop`.

-   #### `Capsules` (`SquashLayer`, `PrimaryCapsule`, `RoutingCapsule`, `CapsuleBlock`)
    -   **File**: `src/dl_techniques/layers/capsules.py`
    -   **Type**: Keras Layer
    -   **Description**: A collection of layers for building Capsule Networks (CapsNets).
        -   `SquashLayer`: A non-linear activation function for capsules that normalizes vector lengths to be between 0 and 1, preserving their orientation. It uses `ops.sum(ops.square(...))` to compute norms and scales the vector.
        -   `PrimaryCapsule`: A layer that converts traditional CNN feature maps into capsule vectors. It uses a `Conv2D` layer followed by `ops.reshape`.
        -   `RoutingCapsule`: Implements the "dynamic routing by agreement" algorithm. It iteratively updates routing logits `b` to determine connections between capsule layers. Core operations include `ops.matmul` for transformations, `keras.activations.softmax` for routing coefficients `c`, and dot products to measure agreement.
        -   `CapsuleBlock`: A convenient wrapper that combines a `RoutingCapsule` with optional `Dropout` and `LayerNormalization`.
    -   **Core Logic (`RoutingCapsule.call`)**:
        1.  Initializes routing logits `b` to zeros.
        2.  Repeats the following for `routing_iterations`:
            a.  Computes routing coefficients `c = softmax(b)`.
            b.  Calculates weighted sum of input predictions `s = Σ c_ij * û_j|i`.
            c.  Applies the `squash` activation to get output vectors `v`.
            d.  Updates logits `b` based on agreement: `b += v · û`.

-   #### `Capsules for Hierarchical Vision Transformers`
    -   **File**: `src/dl_techniques/layers/capsules_hierarchical_vision_transformers.py`
    -   **Type**: Keras Layer / Model
    -   **Description**: An experimental stem for Vision Transformers that replaces standard patch processing with a hierarchical structure of `CapsuleBlock` layers. It processes patches at increasing scales (2x2 -> 4x4 -> ...) using capsule routing at each stage.
    -   **Key Parameters (`__init__`)**: `embed_dim`, `img_size`, `patch_size`.
    -   **Core Logic (`HierarchicalCapsuleStem.call`)**:
        1.  An initial `Conv2D` with a large stride (4x4) creates the first level of "patches."
        2.  The output is reshaped and passed through a `CapsuleBlock`.
        3.  The result is reshaped back to a spatial format and downsampled using `AveragePooling2D`.
        4.  This process (capsule processing -> pooling) is repeated to build up hierarchical representations before feeding into a standard transformer body.

-   #### `CLAHE`
    -   **File**: `src/dl_techniques/layers/clahe.py`
    -   **Type**: Keras Layer
    -   **Description**: Implements Contrast Limited Adaptive Histogram Equalization. It enhances local image contrast by applying histogram equalization to tiles of an image. The "contrast limiting" part clips the histogram at a certain value to prevent noise amplification. This is a non-trainable feature enhancement layer.
    -   **Key Parameters (`__init__`)**: `clip_limit`, `n_bins`, `tile_size`.
    -   **Core Logic (`call` method)**:
        1.  The `call` method iterates through tiles of the input image.
        2.  For each tile, `_process_tile` is called.
        3.  `_process_tile` computes a histogram using `tf.histogram_fixed_width`.
        4.  The histogram is clipped based on `clip_limit`.
        5.  The clipped value excess is redistributed uniformly.
        6.  A cumulative distribution function (CDF) is computed using `tf.cumsum`.
        7.  The CDF is normalized and used as a lookup table to map the original pixel values to their enhanced values using `tf.gather`.

-   #### `Complex Layers` (`ComplexConv2D`, `ComplexDense`, `ComplexReLU`)
    -   **File**: `src/dl_techniques/layers/complex_layers.py`
    -   **Type**: Keras Layer
    -   **Description**: A suite of layers for building complex-valued neural networks. They operate on complex-numbered tensors by splitting them into real and imaginary parts for computation.
        -   `ComplexConv2D`: Implements complex convolution. It uses four real `tf.nn.conv2d` operations to compute the real and imaginary parts of the output, based on the formula `(a+bi)*(c+di) = (ac-bd) + i(ad+bc)`.
        -   `ComplexDense`: Implements a complex fully-connected layer using `tf.matmul` on the split real and imaginary parts.
        -   `ComplexReLU`: Applies the ReLU function independently to the real and imaginary parts of the input tensor.
        -   `_init_complex_weights`: A custom initializer that uses a Rayleigh distribution for magnitude and a uniform distribution for phase, providing better-conditioned weights for complex-valued networks.

-   #### `ConditionalOutput`
    -   **File**: `src/dl_techniques/layers/conditional_output.py`
    -   **Type**: Keras Layer
    -   **Description**: A synonym or alternate implementation of `BatchConditionalOutputLayer`. It selects its output based on whether the `ground_truth` input tensor is all zeros.
    -   **Core Logic (`call` method)**: Same as `BatchConditionalOutputLayer`, using `tf.reduce_all` and `tf.where`.

-   #### `Conv2DBuilder`
    -   **File**: `src/dl_techniques/layers/conv2d_builder.py`
    -   **Type**: Utility Function
    -   **Description**: A factory function, not a layer itself. It constructs a sequence of Keras layers representing a standard "convolution block." This includes a convolution layer (`Conv2D`, `DepthwiseConv2D`, etc.), optional batch/layer normalization, a custom activation, and optional dropout. It provides a consistent way to build common convolutional patterns.
    -   **Key Parameters**: `conv_params`, `bn_params`, `ln_params`, `dropout_params`, `conv_type`.

-   #### `ConvNeXtV1Block` & `ConvNeXtV2Block`
    -   **Files**: `src/dl_techniques/layers/convnext_v1_block.py`, `src/dl_techniques/layers/convnext_v2_block.py`
    -   **Type**: Keras Layer
    -   **Description**: Implements the core blocks of the ConvNeXt architecture. These blocks use an inverted bottleneck design.
        -   **V1 Block**: `DepthwiseConv2D` -> `LayerNorm` -> `Conv2D` (expansion) -> `Activation` (GELU) -> `Conv2D` (projection) -> Optional `LearnableMultiplier` (gamma scaling).
        -   **V2 Block**: Similar to V1, but inserts a `GlobalResponseNormalization` (GRN) layer after the GELU activation to enhance inter-channel feature competition.
    -   **Core Logic (`call` method)**: A sequential application of the layers described above, with a residual connection added at the end.

-   #### `ConvolutionalBlockAttentionModule (CBAM)`
    -   **File**: `src/dl_techniques/layers/convolutional_block_attention_module.py`
    -   **Type**: Keras Layer
    -   **Description**: An attention module for CNNs that infers attention maps along two separate dimensions: channel and spatial.
        -   `ChannelAttention`: Uses both `tf.reduce_mean` (average pooling) and `tf.reduce_max` (max pooling) across spatial dimensions to create two feature descriptors. These are passed through a shared MLP to produce a channel attention map.
        -   `SpatialAttention`: Concatenates the results of average and max pooling across the channel dimension and passes them through a `Conv2D` layer to generate a spatial attention map.
        -   `CBAM`: Applies `ChannelAttention` first, then `SpatialAttention` sequentially to the input feature map.

-   #### `ConvolutionalTransformerBlock`
    -   **File**: `src/dl_techniques/layers/convolutional_transformer.py`
    -   **Type**: Keras Layer
    -   **Description**: A hybrid block that combines self-attention with convolutional layers for Q, K, V projections. This allows it to operate directly on 2D feature maps.
    -   **Core Logic (`call` method)**:
        1.  Input is passed through a pre-normalization `LayerNormalization`.
        2.  The normalized input is flattened using `tf.reshape` for the attention mechanism.
        3.  `MultiHeadAttention` is applied.
        4.  The output is reshaped back to its 2D spatial format.
        5.  A residual connection is added.
        6.  This is followed by another pre-norm and an MLP block (implemented with `Conv2D` layers).

-   #### `Downsample` & `Upsample`
    -   **Files**: `src/dl_techniques/layers/downsample.py`, `src/dl_techniques/layers/upsample.py`
    -   **Type**: Utility Functions
    -   **Description**: Factory functions that provide various strategies for downsampling and upsampling feature maps in CNNs. They are not layers themselves but construct and return Keras layers.
        -   `downsample`: Supports methods like strided `Conv2D`, `MaxPooling2D`, and combinations.
        -   `upsample`: Supports `Conv2DTranspose`, `UpSampling2D` (with 'bilinear' or 'nearest' interpolation), and combinations with subsequent `Conv2D` layers for feature refinement.

-   #### `DyTLayer (DynamicTanh)`
    -   **File**: `src/dl_techniques/layers/dyt_layer.py`
    -   **Type**: Keras Layer
    -   **Description**: An alternative to `LayerNormalization` proposed in "Transformers without Normalization." It's a simple, learnable, element-wise activation.
    -   **Core Logic (`call` method)**:
        1.  Computes `y = tanh(α * x)`, where `α` is a learnable scalar parameter.
        2.  Applies a learnable affine transformation: `output = y * weight + bias`.
        3.  The `weight` and `bias` are learnable vectors that are broadcast to match the input shape, providing per-feature scaling and shifting.

-   #### `GaussianFilter` & `LaplacianFilter`
    -   **Files**: `src/dl_techniques/layers/gaussian_filter.py`, `src/dl_techniques/layers/laplacian_filter.py`
    -   **Type**: Keras Layer
    -   **Description**: Implementations of classical image processing filters as non-trainable Keras layers.
        -   `GaussianFilter`: Applies a Gaussian blur using a `depthwise_conv` with a pre-computed Gaussian kernel. The kernel is generated using `numpy` and converted to a tensor.
        -   `LaplacianFilter`: Approximates the Laplacian of Gaussian (LoG) operator. The basic version computes the difference between the input and its Gaussian-blurred version (`inputs - blurred`). The `AdvancedLaplacianFilter` can also use a pre-computed LoG kernel or a discrete Laplacian kernel for direct convolution.

-   #### `GradientRoutingLayer` (`SelectiveGradientMask`)
    -   **Files**: `src/dl_techniques/layers/gradient_routing_layer.py`, `src/dl_techniques/layers/selective_gradient_layer.py`
    -   **Type**: Keras Layer
    -   **Description**: A layer designed to control gradient flow during backpropagation. It takes two inputs: a `signal` and a binary `mask`.
    -   **Core Logic (`call` method)**:
        1.  During inference (`training=False`), it acts as an identity function, passing the `signal` through unchanged.
        2.  During training (`training=True`), it splits the computation into two paths:
            -   Path 1: `tf.stop_gradient(signal) * mask`. Gradients are blocked for the masked parts of the signal.
            -   Path 2: `signal * (1.0 - mask)`. Gradients flow normally for the unmasked parts.
        3.  The final output is the sum of these two paths, which is numerically identical to the original signal but has a modified gradient.

-   #### `IO Preparation`
    -   **File**: `src/dl_techniques/layers/io_preparation.py`
    -   **Type**: Utility Functions
    -   **Description**: A set of utility functions for tensor normalization, denormalization, and clipping. These are standard preprocessing operations encapsulated in functions.
        -   `normalize_tensor`: Scales a tensor from a source range to a target range using `(x - src_min) / (src_max - src_min) * (tgt_max - tgt_min) + tgt_min`.
        -   `denormalize_tensor`: Reverses the normalization process.
        -   `clip_tensor`: Uses `tf.clip_by_value` to constrain tensor values.

-   #### `LayerScale` (`LearnableMultiplier`)
    -   **File**: `src/dl_techniques/layers/layer_scale.py`
    -   **Type**: Keras Layer
    -   **Description**: A layer that applies a learnable scaling factor (`gamma`) to its input. It can be configured to have a single global scalar or a per-channel vector of scaling factors. This is often used in modern transformer and ConvNet blocks to scale the output of a residual branch before it's added back to the shortcut path.
    -   **Key Parameters (`__init__`)**:
        -   `multiplier_type`: 'GLOBAL' (scalar `gamma`) or 'CHANNEL' (vector `gamma`).
    -   **Core Logic (`call` method)**:
        1.  Performs a simple element-wise multiplication: `ops.multiply(inputs, self.gamma)`.
        2.  Broadcasting handles the application of the scalar or vector gamma to the N-D input tensor.

-   #### `MixedSequentialBlock`
    -   **File**: `src/dl_techniques/layers/mixed_sequential_block.py`
    -   **Type**: Keras Layer
    -   **Description**: A hybrid block inspired by TiRex that can operate as an LSTM, a Transformer, or a mix of both for sequence processing.
    -   **Core Logic (`call` method)**:
        1.  If `block_type` is 'lstm' or 'mixed', it processes the input through an `LSTM` layer and adds the result back as a residual.
        2.  If `block_type` is 'transformer' or 'mixed', it processes the (potentially LSTM-modified) input through a `MultiHeadAttention` layer and adds the result as a residual.
        3.  Finally, it passes the result through a standard two-layer feed-forward network, also with a residual connection.

-   #### `YOLOv12 Layers` (`ConvBlock`, `AreaAttention`, `AttentionBlock`, `Bottleneck`, etc.)
    -   **Files**: `src/dl_techniques/layers/yolo12.py`, `src/dl_techniques/layers/yolo12_heads.py`
    -   **Type**: Keras Layers
    -   **Description**: A collection of specialized building blocks for the YOLOv12 architecture.
        -   `ConvBlock`: The standard unit of `Conv2D` -> `BatchNormalization` -> `Activation('silu')`.
        -   `AreaAttention`: A custom self-attention mechanism that can operate globally or on local "areas" of a feature map by reshaping the input tensor before attention.
        -   `AttentionBlock`: Combines `AreaAttention` with a small MLP in a residual block.
        -   `Bottleneck`: A classic residual block with two `ConvBlock`s.
        -   `C3k2Block`: A CSP-inspired block that splits the input, processes one path through `Bottleneck` layers, and concatenates the result.
        -   `A2C2fBlock`: An ELAN-inspired block that processes input through sequential `AttentionBlock` pairs and progressively concatenates the outputs.
        -   **Heads**: `YOLOv12DetectionHead`, `YOLOv12SegmentationHead`, and `YOLOv12ClassificationHead` are specialized output modules that take multi-scale features and produce predictions for their respective tasks.

-   #### `RoPE (RotaryPositionEmbedding)`
    -   **File**: `src/dl_techniques/layers/rope.py`
    -   **Type**: Keras Layer
    -   **Description**: Implements Rotary Position Embeddings. Instead of adding positional encodings, RoPE rotates query and key vectors based on their absolute position. The key property is that the dot product between two rotated vectors depends only on their *relative* position.
    -   **Core Logic (`call` and `_apply_rope`)**:
        1.  Pre-computes sine and cosine tables for all positions up to `max_seq_len` in the `build` method.
        2.  Splits the input head dimension into two parts: one to be rotated (`x_rope`) and one to be passed through unchanged (`x_pass`).
        3.  The `x_rope` part is treated as a sequence of complex numbers (by pairing consecutive dimensions).
        4.  Applies rotation using the complex number multiplication formula: `x' = x * e^(imθ)`, which translates to `x_real' = x_real*cos - x_imag*sin` and `x_imag' = x_real*sin + x_imag*cos`.
        5.  Concatenates the rotated part with the pass-through part.

-   #### `Swin Layers` (`SwinTransformerBlock`, `WindowAttention`)
    -   **Files**: `src/dl_techniques/layers/swin_transformer_block.py`, `src/dl_techniques/layers/window_attention.py`
    -   **Type**: Keras Layers
    -   **Description**: Implements the core components of the Swin Transformer.
        -   `WindowAttention`: Performs standard multi-head self-attention, but with a crucial addition: a learnable **relative position bias**. This bias is added to the `QK^T` attention scores before the softmax, allowing the model to learn spatial relationships within each window. The bias is retrieved from a learnable table using pre-computed relative position indices.
        -   `SwinTransformerBlock`: Orchestrates the window attention.
            1.  Optionally performs a cyclic shift on the input feature map using `ops.roll`.
            2.  Partitions the feature map into non-overlapping windows using `window_partition`.
            3.  Reshapes the windows into a batch of sequences.
            4.  Applies `WindowAttention`.
            5.  Reverses the windowing and cyclic shift using `window_reverse` and `ops.roll`.
            6.  Includes a standard MLP block with residual connections and layer normalization.

#### Activations
-   **File**: `src/dl_techniques/layers/activations/`
-   **Description**: Provides a suite of standard and novel activation functions.
    -   `basis_function.py`: `BasisFunction` - Implements `f(x) = x / (1 + exp(-x))`, which is mathematically equivalent to Swish/SiLU (`x * sigmoid(x)`). Used in `PowerMLP`.
    -   `explanded_activations.py`: Implements `xGELU`, `xSiLU`, etc., which are variants of standard activations with a learnable parameter `alpha` to control the gating range, making them more flexible.
    -   `mish.py`: Implements `Mish` (`x * tanh(softplus(x))`) and `SaturatedMish`, which clips the Mish function at a certain threshold `alpha` to prevent excessively large activations.
    -   `relu_k.py`: `ReLUK` - Implements `f(x) = max(0, x)^k`, a powered version of ReLU.

#### Feed-Forward Networks (FFN)
-   **File**: `src/dl_techniques/layers/ffn/`
-   **Description**: Contains various implementations of the feed-forward network block found in transformers.
    -   `diff_ffn.py` & `diff_ffn2.py`: `DifferentialFFN` - A dual-pathway network with separate "positive" and "negative" branches. The output is a (normalized) difference between the two, designed to model excitatory and inhibitory signals.
    -   `gated_mlp.py`: `GatedMLP` - An MLP variant where the input is split into two paths. One path is passed through an activation to form a "gate" which is then element-wise multiplied with the other path.
    -   `glu_ffn.py`: `GLUFFN` - Implements Gated Linear Unit variants (GEGLU, SwiGLU, etc.). Similar to gMLP, it uses a gating mechanism but is typically implemented with `Dense` layers.
    -   `mlp.py`: `MLPBlock` - A standard two-layer MLP block (Dense -> Activation -> Dense) used in many transformer architectures.
    -   `residual_block.py`: `ResidualBlock` - A simple two-layer MLP with a residual connection around it.
    -   `swin_mlp.py`: `SwinMLP` - The MLP block specifically for Swin Transformers, typically using GELU activation.

#### Logic Gates
-   **File**: `src/dl_techniques/layers/logic/`
-   **Description**: Implements differentiable logic gates using various fuzzy logic systems (Łukasiewicz, Gödel, Product).
    -   `logic_operations.py`: Contains the core mathematical functions for `AND`, `OR`, `NOT`, `IMPLIES`, etc., for each logic system. For example, Łukasiewicz AND is `max(0, x + y - 1)`, while Product AND is `x * y`.
    -   `logic_gates.py`: `AdvancedLogicGateLayer` - A base class that takes a logical operation function and applies it to inputs, handling trainable weights and biases.
    -   `fuzzy_gates.py`: Provides concrete Keras layer implementations for each gate (e.g., `FuzzyANDGateLayer`, `FuzzyORGateLayer`) that inherit from `AdvancedLogicGateLayer`.

#### Normalization Layers
-   **File**: `src/dl_techniques/layers/norms/`
-   **Description**: A collection of advanced normalization layers.
    -   `rms_norm.py`: `RMSNorm` - Root Mean Square Normalization. Normalizes by the L2 norm of the feature vector (`x / sqrt(mean(x^2))`) without centering (no mean subtraction). Simpler and faster than LayerNorm.
    -   `logit_norm.py`: `LogitNorm` - Normalizes the final logits of a classifier by their L2 norm, scaled by a temperature parameter: `logits / (norm(logits) * temp)`. Helps in model calibration.
    -   `band_rms.py`, `band_logit_norm.py`: `BandRMS`, `BandLogitNorm` - Experimental layers that constrain the output norm to a "thick shell" or band `[1-α, 1]` instead of a unit sphere, using a learnable parameter.
    -   `global_response_norm.py`: `GlobalResponseNormalization` - Normalizes features for a given channel based on the global response (L2 norm) across all channels.
    -   `max_logit_norm.py`: `DecoupledMaxLogit` - A normalization for OOD detection that decouples a logit into its magnitude (L2 norm) and direction (cosine similarity).

### Losses
-   #### `AnyLoss` Framework
    -   **File**: `src/dl_techniques/losses/any_loss.py`
    -   **Type**: Keras Loss
    -   **Description**: A framework for turning any confusion matrix-based metric (like F1-score, Balanced Accuracy) into a differentiable loss function.
    -   **Core Logic (`compute_confusion_matrix`)**:
        1.  It first transforms the model's probabilistic outputs (`y_pred`) into near-binary values using an `ApproximationFunction`.
        2.  It then computes differentiable versions of TP, TN, FP, FN using element-wise multiplication: `TP = sum(y_true * y_approx)`, `FN = sum(y_true * (1 - y_approx))`, etc.
        3.  Subclasses like `F1Loss` then use these differentiable components to calculate the metric, which is then inverted to form a loss (e.g., `1 - F1_score`).

-   #### `BrierScoreLoss` & `SpiegelhalterZLoss`
    -   **File**: `src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py`
    -   **Type**: Keras Loss
    -   **Description**: Loss functions that directly optimize for model calibration.
        -   `BrierScoreLoss`: Computes the mean squared error between predicted probabilities and one-hot true labels: `(1/N) * Σ(p_i - o_i)²`.
        -   `SpiegelhalterZLoss`: A differentiable version of the Z-test statistic, which penalizes systematic bias in probability predictions.

-   #### `CapsuleMarginLoss`
    -   **File**: `src/dl_techniques/losses/capsule_margin_loss.py`
    -   **Type**: Keras Loss
    -   **Description**: The margin loss for Capsule Networks. It uses separate penalties for positive and negative classes based on the length of the output capsule vectors.
    -   **Core Logic (`call` method)**:
        1.  **Positive Loss**: For the correct class `k`, the loss is `T_k * max(0, m⁺ - ||v_k||)²`. This penalizes the model if the capsule length `||v_k||` is less than the positive margin `m⁺`.
        2.  **Negative Loss**: For incorrect classes, the loss is `λ * (1 - T_k) * max(0, ||v_k|| - m⁻)²`. This penalizes the model if capsule lengths are greater than the negative margin `m⁻`.
        3.  The total loss is the sum over all classes.

-   #### `CLIPContrastiveLoss`
    -   **File**: `src/dl_techniques/losses/clip_contrastive_loss.py`
    -   **Type**: Keras Loss
    -   **Description**: Implements the symmetric contrastive loss for training CLIP models. It works on a similarity matrix between image and text embeddings.
    -   **Core Logic (`call` method)**:
        1.  Takes a dictionary `y_pred` containing two similarity matrices: `logits_per_image` and `logits_per_text`. These are (N, N) matrices for a batch of N pairs.
        2.  Creates the ground truth labels, which are simply the diagonal indices `[0, 1, ..., N-1]`.
        3.  Calculates the image-to-text loss using `keras.losses.sparse_categorical_crossentropy` on `logits_per_image` and the labels.
        4.  Calculates the text-to-image loss similarly using `logits_per_text`.
        5.  The final loss is the average of the two directional losses.

-   #### `ClusteringLoss`
    -   **File**: `src/dl_techniques/losses/clustering_loss.py`
    -   **Type**: Keras Loss
    -   **Description**: A loss function for deep clustering. It encourages both compactness of clusters and balanced cluster sizes.
    -   **Core Logic (`call` method)**:
        1.  **Distance Term**: Computes the mean squared error between soft assignments `y_pred` and hard assignments `y_true`, encouraging high-confidence assignments.
        2.  **Distribution Term**: Computes the mean of `y_pred` across the batch to get the cluster size distribution. It then penalizes the squared difference between this distribution and a uniform distribution, encouraging clusters of similar size.
        3.  The final loss is a weighted sum of these two terms.

-   #### `YOLOv12MultiTaskLoss`
    -   **File**: `src/dl_techniques/losses/yolo12_multitask_loss.py`
    -   **Type**: Keras Loss
    -   **Description**: An orchestrator loss for the YOLOv12 multi-task model. It's the single loss function passed to `model.compile`.
    -   **Core Logic (`call` method)**:
        1.  It inspects the shape of the incoming `y_pred` tensor.
        2.  Based on the tensor's rank (2D, 3D, or 4D), it infers the task (classification, detection, or segmentation).
        3.  It then calls the appropriate internal, task-specific loss function (`YOLOv12ObjectDetectionLoss`, `DiceFocalSegmentationLoss`, etc.) on `y_true` and `y_pred`.
        4.  It applies either fixed or learnable uncertainty weights to the returned task loss.

### Metrics
-   #### `CLIPAccuracy`
    -   **File**: `src/dl_techniques/metrics/clip_accuracy.py`
    -   **Type**: Keras Metric
    -   **Description**: Calculates top-k retrieval accuracy for CLIP models.
    -   **Core Logic (`update_state` method)**:
        1.  Takes the two similarity matrices (`logits_per_image`, `logits_per_text`) as input.
        2.  The ground truth is the diagonal `[0, 1, ..., N-1]`.
        3.  For `logits_per_image`, it finds the `top_k` predictions for each row (image) and checks if the correct text index is present.
        4.  It does the same for `logits_per_text` (text-to-image accuracy).
        5.  The final result is the average accuracy across both directions.

### Models
-   #### `CapsNet`
    -   **File**: `src/dl_techniques/models/capsnet.py`
    -   **Type**: Keras Model
    -   **Description**: A complete implementation of the Capsule Network architecture. It integrates the custom `train_step` and `test_step` methods to handle the composite margin loss and optional reconstruction loss, making it fully compatible with the standard Keras `fit` workflow.
-   #### `CLIPModel`
    -   **File**: `src/dl_techniques/models/clip.py`
    -   **Type**: Keras Model
    -   **Description**: A dual-encoder architecture for learning joint image-text embeddings. It contains a `VisionTransformer` and a `TextTransformer`. The `call` method encodes both inputs, normalizes the embeddings, and computes the dot-product similarity matrix.
-   #### `ConvNeXtV1` & `ConvNeXtV2`
    -   **Files**: `src/dl_techniques/models/convnext_v1.py`, `src/dl_techniques/models/convnext_v2.py`
    -   **Type**: Keras Model
    -   **Description**: Full model implementations of the ConvNeXt architectures, composed of their respective `ConvNextV1Block` or `ConvNextV2Block` layers. They include a stem, multiple stages with downsampling, and a classification head.
-   #### `CoShNet`
    -   **File**: `src/dl_techniques/models/coshnet.py`
    -   **Type**: Keras Model
    -   **Description**: A hybrid model that uses a fixed `ShearletTransform` as a feature extraction front-end, followed by a series of learnable `ComplexConv2D` and `ComplexDense` layers.
-   #### `DepthAnything`
    -   **File**: `src/dl_techniques/models/depth_anything.py`
    -   **Type**: Keras Model
    -   **Description**: An implementation of the Depth Anything model for monocular depth estimation. It uses a DINOv2 encoder and a DPT decoder and is trained with a custom `train_step` that handles a combination of labeled loss, unlabeled loss (using a teacher model), and a feature alignment loss against a frozen encoder.
-   #### `HolographicMPSNet`
    -   **File**: `src/dl_techniques/models/holographic_mps_net.py`
    -   **Type**: Keras Model
    -   **Description**: An experimental encoder-decoder model inspired by the holographic principle. It uses `MPSLayer`s and an `EntropyRegularizer`. The key idea is a multi-branch decoder where each branch is regularized to have a different target entropy, forcing them to learn features at different scales (global vs. local).
-   #### `MDNModel`
    -   **File**: `src/dl_techniques/models/mdn_model.py`
    -   **Type**: Keras Model
    -   **Description**: A complete Mixture Density Network model. It consists of a standard feature extraction MLP/RNN followed by an `MDNLayer`. The class also includes helper methods for `sample`ing from the predicted distribution and calculating `predict_with_uncertainty`.
-   #### `MobileNetV4`
    -   **File**: `src/dl_techniques/models/mobilenet_v4.py`
    -   **Type**: Keras Model
    -   **Description**: Implements the MobileNetV4 architecture, composed of `UIB` (Universal Inverted Bottleneck) blocks and an optional `MobileMQA` attention layer in the final stage.
-   #### `PowerMLP`
    -   **File**: `src/dl_techniques/models/power_mlp.py`
    -   **Type**: Keras Model
    -   **Description**: A full model built by stacking multiple `PowerMLPLayer`s. It serves as an efficient alternative to KANs.
-   #### `SCUNet`
    -   **File**: `src/dl_techniques/models/scunet.py`
    -   **Type**: Keras Model
    -   **Description**: A U-Net architecture that uses `SwinConvBlock`s in its encoder and decoder paths, combining the strengths of convolutions and windowed attention for image restoration.
-   #### `SOMMemory`
    -   **File**: `src/dl_techniques/models/som_memory.py`
    -   **Type**: Keras Model
    -   **Description**: A Self-Organizing Map (SOM) model. It wraps the `SOM2dLayer` and implements a custom `train_som` method, as SOMs are not trained via backpropagation. It also includes methods for visualization and using the trained map for classification.
-   #### `TiRexCore`
    -   **File**: `src/dl_techniques/models/tirex.py`
    -   **Type**: Keras Model
    -   **Description**: A time-series forecasting model composed of an `PatchEmbedding1d` layer, a series of `MixedSequentialBlock`s, and a `QuantileHead` for probabilistic forecasting.
-   #### `VAE`
    -   **File**: `src/dl_techniques/models/vae.py`
    -   **Type**: Keras Model
    -   **Description**: A convolutional Variational Autoencoder. It has a custom `train_step` that computes and combines the reconstruction loss (MSE) and the KL divergence loss between the latent distribution and a standard normal prior.
-   #### `YOLOv12MultiTask`
    -   **File**: `src/dl_techniques/models/yolo12_multitask.py`
    -   **Type**: Keras Model
    -   **Description**: The main multi-task model. It uses the Keras Functional API to combine a shared `YOLOv12FeatureExtractor` with one or more task-specific heads (`YOLOv12DetectionHead`, etc.). The outputs are structured as a dictionary with keys corresponding to the task names.

### Optimization
-   #### `optimizer_builder` & `schedule_builder`
    -   **File**: `src/dl_techniques/optimization/optimizer.py`
    -   **Type**: Factory Functions
    -   **Description**: These functions provide a configuration-driven way to create Keras optimizers (`Adam`, `RMSprop`, etc.) and learning rate schedules (`CosineDecay`, `ExponentialDecay`). The `schedule_builder` can wrap any base schedule with a `WarmupSchedule`.

### Regularizers
-   #### `BinaryPreferenceRegularizer`
    -   **File**: `src/dl_techniques/regularizers/binary_preference.py`
    -   **Type**: Keras Regularizer
    -   **Description**: Encourages weights to take on binary values (0 or 1) by applying a polynomial cost function `y = (1 - ((x-0.5)^2)/0.25)^2` that has minima at 0 and 1.
-   #### `EntropyRegularizer`
    -   **File**: `src/dl_techniques/regularizers/entropy_regularizer.py`
    -   **Type**: Keras Regularizer
    -   **Description**: Penalizes the deviation of a weight matrix's normalized entropy from a target value. It computes the Shannon entropy of the softmax-normalized absolute weights and compares it to a target.
-   #### `SoftOrthogonal` & `SoftOrthonormal` Regularizers
    -   **File**: `src/dl_techniques/regularizers/soft_orthogonal.py`
    -   **Type**: Keras Regularizer
    -   **Description**: These regularizers encourage weight matrices to be orthogonal or orthonormal.
        -   `SoftOrthogonal`: Penalizes the Frobenius norm of the off-diagonal elements of the Gram matrix `W^T * W`.
        -   `SoftOrthonormal`: Penalizes the Frobenius norm of `W^T * W - I`, enforcing both orthogonality and unit norm columns.
-   #### `SRIPRegularizer`
    -   **File**: `src/dl_techniques/regularizers/srip.py`
    -   **Type**: Keras Regularizer
    -   **Description**: Enforces approximate orthonormality by minimizing the *spectral norm* (largest singular value) of `W^T * W - I`. The spectral norm is approximated using power iteration.
-   #### `TriStatePreferenceRegularizer`
    -   **File**: `src/dl_techniques/regularizers/tri_state_preference.py`
    -   **Type**: Keras Regularizer
    -   **Description**: Encourages weights to converge to -1, 0, or 1 using a 6th-order polynomial cost function with minima at these three points.

### Utilities
-   **Core Utilities**:
    -   `analyzer.py`: A `ModelAnalyzer` class for comprehensive evaluation of calibration, weights, activations, and information flow.
    -   `bounding_box.py`: Functions for IoU calculations (`bbox_iou`) and Non-Maximum Suppression (`bbox_nms`).
    -   `calibration_metrics.py`: Functions to compute ECE, MCE, Brier score, and other calibration metrics.
    -   `tensors.py`: Low-level tensor manipulation functions like `gram_matrix` and `reshape_to_2d`.
    -   `visualization.py` & `visualization_manager.py`: Tools for creating and managing plots.
-   **Dataset Utilities**:
    -   `common.py`: Defines a common `Dataset` NamedTuple.
    -   `mnist.py`, `cifar10.py`: Standard functions to load and preprocess these datasets.
    -   `sut.py`, `sut_tf.py`, `coco.py`: Advanced, patch-based data loaders for the SUT-Crack and COCO datasets, designed for multi-task learning. They handle parsing annotations, sampling patches, and creating `tf.data.Dataset` objects.
    -   `class_balancer.py`: Implements strategies like undersampling and oversampling to handle class imbalance at the data level.

### WeightWatcher
-   **File**: `src/dl_techniques/weightwatcher/`
-   **Description**: A powerful diagnostic tool for analyzing the spectral properties of weight matrices.
    -   `weightwatcher.py`: The main `WeightWatcher` class that orchestrates the analysis. It computes the Eigenvalue Spectral Density (ESD) of layer weight matrices.
    -   `metrics.py`: Contains functions to `fit_powerlaw` to the ESD, and calculate metrics like `alpha`, `stable_rank`, `matrix_entropy`, and `concentration_score`.
    -   `analyzer.py`: Provides high-level functions like `analyze_model` and `compare_models` that use the `WeightWatcher` class to generate comprehensive reports and visualizations.

## Training Techniques

### Differential Learning Rates
Multiple approaches for layer-specific learning rates:

1.  **Optimizer Learning Rate Multipliers**: `optimizer.learning_rate_multipliers = {w.name: multiplier for w in weights}`
2.  **Multiple Optimizers**: `model.compile(optimizer={'backbone': Adam(lr=1e-4), 'head': Adam(lr=1e-3)})`
3.  **Custom Training Loop**: Manually apply gradients with different optimizers.
4.  **Layer-wise Decay Pattern**: Common in transfer learning, where earlier layers have lower learning rates.

### Advanced Training Features
-   **Gradient Accumulation**: For simulating large batch sizes.
-   **Mixed Precision**: FP16 training support via Keras policies.
-   **Discriminative Fine-Tuning**: Using custom callbacks or multiple optimizers.
-   **Deep Supervision**: Applying losses at multiple stages of a network, often seen in U-Net architectures.

## Best Practices

### Code Style
1.  **Type Hints**: Use throughout for clarity (`def forward(self, x: tf.Tensor) -> tf.Tensor:`).
2.  **Docstrings**: Use Sphinx-compliant documentation with `Args`, `Returns`, and `Raises` sections.
3.  **Logging**: Use the project's central `logger` instead of `print`.

### Layer Development
1.  **Keras Operations**: Prefer `keras.ops` over backend-specific ops (`tf.`, `torch.`).
2.  **Serialization**: Always implement `get_config`, `get_build_config`, and `build_from_config`, and use the `@keras.saving.register_keras_serializable()` decorator.
3.  **Testing**: Write comprehensive tests covering initialization, building, forward pass, serialization, and gradient flow.
4.  **Shape Handling**: Convert tensor shapes to lists for manipulation (e.g., `list(input_shape)`) and return new shapes as tuples.
5.  **Build Pattern**: Initialize sublayers and weights in `build()`, not `__init__()`, to support deferred building.

## Technical Specifications

### Keras Operations
The project standardizes on `keras.ops` for all tensor manipulations to ensure backend-agnostic compatibility. This includes:

-   **Core**: `cast`, `cond`, `shape`, `stop_gradient`, `while_loop`, `slice`, `scatter`.
-   **NumPy API (`keras.ops.numpy`)**: `arange`, `ones`, `zeros`, `linspace`, `meshgrid`, `concatenate`, `expand_dims`, `reshape`, `transpose`, `stack`, `add`, `subtract`, `multiply`, `divide`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `maximum`, `minimum`, `clip`, `sum`, `mean`, `std`, `max`, `min`, `argmax`, `argmin`, `dot`, `matmul`, `einsum`.
-   **NN API (`keras.ops.nn`)**: `relu`, `sigmoid`, `softmax`, `log_softmax`, `gelu`, `silu`, `conv`, `max_pool`, `average_pool`, `one_hot`.
-   **Linalg API (`keras.ops.linalg`)**: `norm`, `qr`, `svd`.
-   **Image API (`keras.ops.image`)**: `resize`.

### Model Persistence
-   **Format**: Models must be saved in the `.keras` format.
-   **Custom Objects**: When loading models with custom components, a `custom_objects` dictionary must be passed to `keras.models.load_model`.
    ```python
    model = keras.models.load_model(
        "model.keras",
        custom_objects={"CustomLayer": CustomLayer, "CustomLoss": CustomLoss}
    )
    ```