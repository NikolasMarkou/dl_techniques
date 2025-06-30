# DL-Techniques Project Knowledge Base

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Development Environment](#development-environment)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Creating Custom Layers](#creating-custom-layers)
6. [Testing Guidelines](#testing-guidelines)
7. [Available Components](#available-components)
8. [Training Techniques](#training-techniques)
9. [Best Practices](#best-practices)
10. [Technical Specifications](#technical-specifications)

---

## 1. Project Overview

DL-Techniques is a comprehensive deep learning library built on Keras 3.8.0 and TensorFlow 2.18.0, providing advanced neural network components including custom layers, models, losses, metrics, initializers, regularizers, and optimization utilities. The library emphasizes:

- **Backend-agnostic implementation** using Keras operations
- **Proper serialization** for production deployment
- **Comprehensive testing** for reliability
- **Modern architectures** including Vision Transformers, ConvNeXt, Capsule Networks, and more
- **Advanced techniques** such as differential attention, KAN networks, and shearlet transforms

## 2. Project Structure

### Detailed Directory Tree
```
.
└── src/
    ├── dl_techniques/
    │   ├── constraints/          # Custom weight constraints
    │   ├── initializers/         # Custom weight initializers
    │   ├── layers/               # Custom layer implementations
    │   │   ├── activations/      # Activation function layers
    │   │   ├── ffn/              # Feed-forward network variants
    │   │   ├── logic/            # Logic-based neural network layers
    │   │   └── norms/            # Normalization layers
    │   ├── losses/               # Custom loss functions
    │   ├── metrics/              # Custom evaluation metrics
    │   ├── models/               # Complete model architectures
    │   ├── optimization/         # Optimizers and learning rate schedules
    │   ├── regularizers/         # Custom weight regularizers
    │   ├── utils/                # Utility functions
    │   │   └── datasets/         # Dataset loading and processing utilities
    │   ├── visualization/        # Visualization tools (deprecated, see utils)
    │   └── weightwatcher/        # Weight matrix analysis tools
    ├── experiments/              # Experimental implementations and studies
    │   ├── coupled_rms_norm/
    │   ├── diff_ffn/
    │   ├── goodhart/
    │   ├── kmeans/
    │   ├── layer_scale_binary_state/
    │   ├── mdn/
    │   ├── mish/
    │   ├── orthoblock/
    │   ├── rbf/
    │   ├── regularizer_binary/
    │   ├── regularizer_tri_state/
    │   ├── rms_norm/
    │   ├── softmax/
    │   └── som/
    └── train/                    # Training scripts for various models
        ├── capsnet/
        ├── mobilenet_v4/
        ├── power_mlp/
        ├── tirex/
        ├── vae/
        ├── yolo12/
        └── yolo12_pavement_defect/
```

## 3. Development Environment

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

logging.basicConfig(level=logging.INFO,
                    format=LOGGER_FORMAT)
logging.getLogger("dl").setLevel(logging.INFO)
logger = logging.getLogger("dl")

# Usage:
logger.info("Training started")
logger.debug("Detailed information")
logger.warning("Potential issue")
```

## 4. Architecture Guidelines

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

1. **Initialization**
   - Call `super().__init__(**kwargs)` to handle base layer parameters
   - Store configuration as instance attributes
   - Initialize weights to `None`
   - Use type hints and comprehensive docstrings
   - Make regularizers and initializers customizable through `__init__`

2. **Building**
   - Create weights lazily in `build()` when input shape is known
   - Initialize complex sublayers in `build()`
   - Store `_build_input_shape` for serialization
   - Use `self.add_weight()` for weight creation
   - Call `super().build(input_shape)` at the end

3. **Forward Computation**
   - Implement logic in `call()` method
   - Use `keras.ops` for backend compatibility
   - Include `training` parameter and propagate to sublayers
   - Handle different backends gracefully

4. **Serialization**
   - Implement `get_config()` for constructor parameters
   - Implement `get_build_config()` for build information
   - Implement `build_from_config()` for reconstruction
   - Register with `@keras.saving.register_keras_serializable()`

### Example Layer Structure
```python
import keras
from keras import ops
from typing import Optional, Union, Any, Dict

@keras.saving.register_keras_serializable()
class CustomLayer(keras.layers.Layer):
    """Custom layer with proper structure.
    
    Args:
        units: Integer, dimensionality of output space.
        activation: Activation function to use.
        use_bias: Boolean, whether to use bias.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        
        # Initialize to None
        self.kernel = None
        self.bias = None
        self._build_input_shape = None
        
    def build(self, input_shape):
        """Build the layer."""
        self._build_input_shape = input_shape
        
        # Create weights
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
            
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass."""
        outputs = ops.matmul(inputs, self.kernel)
        
        if self.use_bias:
            outputs = outputs + self.bias
            
        if self.activation is not None:
            outputs = self.activation(outputs)
            
        return outputs
        
    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.units])
        
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config
        
    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}
        
    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
```

## 5. Creating Custom Layers

### Key Requirements

1. **Backend Compatibility**: Use `keras.ops` instead of backend-specific operations
2. **Shape Handling**: Convert shapes to lists for manipulation, return as tuples
3. **Browser Storage**: Never use localStorage or sessionStorage in artifacts
4. **Weight Management**: Create weights with meaningful names and proper initializers
5. **Training Mode**: Always handle and propagate the `training` parameter

### Common Pitfalls to Avoid

1. **Shape Serialization Errors**
   ```python
   # Bad
   def compute_output_shape(self, input_shape):
       return input_shape[:-1] + (self.units,)  # Can fail
   
   # Good
   def compute_output_shape(self, input_shape):
       input_shape_list = list(input_shape)
       return tuple(input_shape_list[:-1] + [self.units])
   ```

2. **Missing Build Configuration**
   ```python
   # Always store and restore build information
   def build(self, input_shape):
       self._build_input_shape = input_shape
       # ... rest of build
   
   def get_build_config(self):
       return {"input_shape": self._build_input_shape}
   ```

3. **Improper Sublayer Handling**
   ```python
   # Create sublayers in build, not __init__
   def build(self, input_shape):
       self.dense = keras.layers.Dense(64)
       self.dense.build(input_shape)
       super().build(input_shape)
   ```

## 6. Testing Guidelines

### Test Structure
```python
import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

class TestCustomLayer:
    """Test suite for CustomLayer."""
    
    @pytest.fixture
    def input_tensor(self):
        """Create test input tensor."""
        return tf.random.normal([4, 32, 32, 64])
        
    @pytest.fixture
    def layer_instance(self):
        """Create default layer instance."""
        return CustomLayer(filters=32)
```

### Essential Test Categories

1. **Initialization Tests**: Verify default and custom parameter handling
2. **Build Process Tests**: Ensure proper weight creation
3. **Output Shape Tests**: Validate shape computation
4. **Forward Pass Tests**: Check computation correctness
5. **Serialization Tests**: Verify config save/load
6. **Model Integration Tests**: Test within model context
7. **Save/Load Tests**: Ensure model persistence works
8. **Gradient Flow Tests**: Verify backpropagation
9. **Edge Case Tests**: Test numerical stability
10. **Training Tests**: Verify training behavior

### Example Test
```python
def test_serialization(self):
    """Test layer serialization."""
    # Create and build layer
    original_layer = CustomLayer(units=64)
    original_layer.build((None, 32))
    
    # Test data
    x = tf.random.normal((2, 32))
    original_output = original_layer(x)
    
    # Serialize and recreate
    config = original_layer.get_config()
    build_config = original_layer.get_build_config()
    
    new_layer = CustomLayer.from_config(config)
    new_layer.build_from_config(build_config)
    
    # Verify outputs match
    new_output = new_layer(x)
    assert tf.reduce_all(tf.equal(original_output, new_output))
```

## 7. Available Components

### Models
- **CapsNet**: Capsule Network with optional reconstruction decoder (`capsnet.py`)
- **CLIP**: Contrastive Language-Image Pre-training model (`clip.py`)
- **ConvNeXtV1/V2**: Modern ConvNet architectures (`convnext_v1.py`, `convnext_v2.py`)
- **CoShNet**: Hybrid Shearlet-CNN model (`coshnet.py`)
- **DepthAnything**: Monocular depth estimation (`depth_anything.py`)
- **HolographicMPSNet**: Experimental MPS-based architecture (`holographic_mps_net.py`)
- **MDNModel**: Mixture Density Network for probabilistic regression (`mdn_model.py`)
- **MobileNetV4**: Efficient mobile architecture (`mobilenet_v4.py`)
- **PowerMLP**: Efficient KAN-alternative (`power_mlp.py`)
- **SCUNet**: Swin-Conv-UNet for image restoration (`scunet.py`)
- **SOMMemory**: Self-Organizing Map as an associative memory (`som_memory.py`)
- **TiRex**: Bidirectional sequence processor for time-series (`tirex.py`)
- **VAE**: Variational Autoencoder (`vae.py`)
- **YOLOv12**: Feature extractor and multi-task model (`yolo12_feature_extractor.py`, `yolo12_multitask.py`)

### Key Layers

#### Attention Mechanisms
- **AdaptiveMultiHeadAttention**: Dynamic temperature adjustment (`adaptive_softmax_mha.py`)
- **ConvolutionalBlockAttentionModule (CBAM)**: Channel and spatial attention (`convolutional_block_attention_module.py`)
- **DifferentialAttention**: Noise-cancelling dual-pathway attention (`differential_attention.py`)
- **GroupQueryAttention (GQA)**: Efficient multi-query attention (`group_query_attention.py`)
- **HopfieldAttention**: Modern Hopfield networks as attention (`hopfield_attention.py`)
- **MobileMQA**: Mobile-optimized multi-query attention (`mobile_mqa.py`)
- **NonLocalAttention**: Captures long-range dependencies (`non_local_attention.py`)
- **SparseAttention**: Enforces sparsity in attention weights (`sparse_attention.py`)
- **WindowAttention**: Local window attention for Swin Transformers (`window_attention.py`)

#### Vision Transformers & Related Components
- **VisionTransformer**: Standard ViT layer block (`vision_transformer.py`)
- **ConvolutionalTransformer**: Hybrid Conv-Attention block (`convolutional_transformer.py`)
- **HierarchicalMLPStem**: Patch-independent stem for ViTs (`hierarchical_mlp_stem.py`)
- **HierarchicalVisionTransformers**: Full ViT model with hMLP stem (`hierarchical_vision_transformers.py`)
- **PatchEmbedding**: Image to patch sequence conversion (`patch_embedding.py`)
- **PositionalEmbedding**: Learnable positional encodings (`positional_embedding.py`)
- **RotaryPositionEmbedding (RoPE)**: Rotational positional embeddings (`rope.py`)
- **SwinTransformerBlock**: Shifted window attention block (`swin_transformer_block.py`)

#### Convolutional Variants & Image Processing
- **BitLinear**: 1.58-bit Linear Layer (`bitlinear_layer.py`)
- **Canny**: Canny edge detection layer (`canny.py`)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization (`clahe.py`)
- **ComplexConv2D**: Complex-valued convolution (`complex_layers.py`)
- **Conv2DBuilder**: Wrapper for creating complex conv blocks (`conv2d_builder.py`)
- **ConvNeXtV1/V2Block**: Modern ConvNet blocks (`convnext_v1_block.py`, `convnext_v2_block.py`)
- **ConvolutionalKAN**: Kolmogorov-Arnold Network Convolution (`convolutional_kan.py`)
- **Downsample/Upsample**: Various downsampling/upsampling strategies (`downsample.py`, `upsample.py`)
- **GaussianFilter** & **LaplacianFilter**: Classic image filters (`gaussian_filter.py`, `laplacian_filter.py`)
- **SwinConvBlock**: Swin + Conv hybrid block (`swin_conv_block.py`)
- **ShearletTransform**: Multi-scale analysis layer (`shearlet_transform.py`)
- **UIB (Universal Inverted Bottleneck)**: Core block for MobileNetV4 (`universal_inverted_bottleneck.py`)
- **YOLOv12 Blocks**: `ConvBlock`, `A2C2fBlock`, `C3k2Block` for YOLO architecture (`yolo12.py`)

#### Feed-Forward Network (FFN) Variants
- **MLPBlock**: Standard transformer FFN (`ffn/mlp.py`)
- **DiffFFN**: Differential FFN with dual pathways (`ffn/diff_ffn.py`, `ffn/diff_ffn2.py`)
- **GatedMLP**: Gated MLP from "Pay Attention to MLPs" (`ffn/gated_mlp.py`)
- **GLUFFN**: Gated Linear Unit FFN (`ffn/glu_ffn.py`)
- **SwiGLUFFN**: Swish Gated Linear Unit FFN (`ffn/swiglu_ffn.py`)
- **SwinMLP**: FFN for Swin Transformers (`ffn/swin_mlp.py`)
- **ResidualBlock**: Simple residual FFN block (`ffn/residual_block.py`)

#### Normalization Layers
- **RMSNorm**: Root Mean Square normalization (`norms/rms_norm.py`)
- **BandRMS**: Constrained RMS norm (`norms/band_rms.py`)
- **GlobalResponseNormalization (GRN)**: From ConvNeXt V2 (`norms/global_response_norm.py`)
- **LogitNorm**: Output logit normalization (`norms/logit_norm.py`)
- **BandLogitNorm**: Constrained logit normalization (`norms/band_logit_norm.py`)
- **MaxLogitNorm**: For OOD detection (`norms/max_logit_norm.py`)

#### Activations
- **Mish/SaturatedMish**: Self-regularizing non-monotonic activations (`activations/mish.py`)
- **ReLUK**: Powered ReLU activation (`activations/relu_k.py`)
- **BasisFunction**: Self-gated activation from PowerMLP (`activations/basis_function.py`)
- **ExpandedActivations**: GELU, SiLU, and their expanded-range variants (`activations/explanded_activations.py`)
- **AdaptiveSoftmax**: Softmax with adaptive temperature (`adaptive_softmax.py`)

#### Novel & Specialized Layers
- **Capsule Layers**: `PrimaryCapsule`, `RoutingCapsule` (`capsules.py`)
- **KAN**: Kolmogorov-Arnold Network Layer (`kan.py`)
- **MPSLayer**: Matrix Product States (`mps_layer.py`)
- **PowerMLPLayer**: Efficient KAN-alternative (`power_mlp_layer.py`)
- **MDNLayer**: Mixture Density Network layer (`mdn_layer.py`)
- **RBF**: Radial Basis Function layer (`rbf.py`)
- **SOM2dLayer**: Self-Organizing Map (`som_2d_layer.py`)
- **Fuzzy Logic Gates**: `FuzzyANDGateLayer`, `FuzzyORGateLayer`, etc. (`logic/fuzzy_gates.py`)
- **LayerScale**: Learnable per-channel scaling (`layer_scale.py`)
- **StochasticDepth**: Drops residual paths during training (`stochastic_depth.py`)
- **Sampling**: Reparameterization trick for VAEs (`sampling.py`)

### Losses
- **AnyLoss Framework**: Converts classification metrics into loss functions (`any_loss.py`)
- **BrierSpiegelhalterZTestLoss**: Calibration-focused losses (`brier_spiegelhalters_ztest_loss.py`)
- **CapsuleMarginLoss**: Margin loss for capsule networks (`capsule_margin_loss.py`)
- **CLIPContrastiveLoss**: CLIP training loss (`clip_contrastive_loss.py`)
- **ClusteringLoss**: Deep clustering loss (`clustering_loss.py`)
- **GoodhartAwareLoss**: Robust, information-theoretic loss (`goodhart_loss.py`)
- **SegmentationLoss**: Composite loss functions for segmentation (`segmentation_loss.py`)
- **YOLOv12MultiTaskLoss**: Orchestrates losses for detection, segmentation, and classification (`yolo12_multitask_loss.py`)

### Regularizers
- **Binary/TriStatePreference**: Encourages weights towards binary/ternary values (`binary_preference.py`, `tri_state_preference.py`)
- **EntropyRegularizer**: Encourages specific entropy profiles in weight matrices (`entropy_regularizer.py`)
- **SoftOrthogonal/Orthonormal**: Enforces orthogonality on weights (`soft_orthogonal.py`)
- **SRIP**: Spectral Restricted Isometry Property regularizer (`srip.py`)

### Initializers & Constraints
- **HaarWaveletInitializer**: Initializes conv kernels with Haar wavelet filters (`haar_wavelet_initializer.py`)
- **He/OrthonormalInitializer**: Combines He initialization with QR decomposition (`he_orthonormal_initializer.py`, `orthonormal_initializer.py`)
- **ValueRangeConstraint**: Clips weights to a specified min/max range (`value_range_constraint.py`)

## 8. Training Techniques

### Differential Learning Rates

Multiple approaches for layer-specific learning rates:

1. **Optimizer Learning Rate Multipliers**
   ```python
   optimizer.learning_rate_multipliers = {
       w.name: multiplier for w in weights
   }
   ```

2. **Multiple Optimizers**
   ```python
   model.compile(
       optimizer={
           'backbone': keras.optimizers.Adam(lr=0.0001),
           'head': keras.optimizers.Adam(lr=0.001)
       }
   )
   ```

3. **Custom Training Loop**
   ```python
   @tf.function
   def train_step(images, labels):
       with tf.GradientTape() as tape:
           predictions = model(images, training=True)
           loss = loss_function(labels, predictions)
       
       gradients = tape.gradient(loss, model.trainable_variables)
       # Apply different optimizers to different variables
   ```

4. **Layer-wise Decay Pattern**
   - Earlier layers: lower learning rates
   - Later layers: higher learning rates
   - Common in transfer learning

### Advanced Training Features
- **Gradient Accumulation**: For large batch training
- **Mixed Precision**: FP16 training support
- **Discriminative Fine-Tuning**: Via custom callbacks
- **Deep Supervision**: Multi-stage loss application

## 9. Best Practices

### Code Style
1. **Type Hints**: Use throughout for clarity
   ```python
   def forward(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
   ```

2. **Docstrings**: Sphinx-compliant documentation
   ```python
   """Brief description.
   
   Args:
       param: Description of parameter.
       
   Returns:
       Description of return value.
       
   Raises:
       ExceptionType: When this occurs.
   """
   ```

3. **Logging**: Use project logger, not print
   ```python
   from dl_techniques.utils.logger import logger
   logger.info("Processing batch")
   ```

### Layer Development
1. **Keras Operations**: Prefer `keras.ops` over backend-specific ops
2. **Serialization**: Always implement all three serialization methods
3. **Testing**: Write comprehensive tests for each component
4. **Shape Handling**: Be consistent with list/tuple conversions
5. **Build Pattern**: Initialize sublayers in `build()`, not `__init__()`

### Performance
1. **Lazy Weight Creation**: Only create weights when needed
2. **Backend Agnostic**: Ensure compatibility across backends
3. **Memory Efficiency**: Clean up resources properly
4. **Numerical Stability**: Test with extreme values

## 10. Technical Specifications

### Available Keras Operations (Detailed)

The project uses Keras 3.8.0 operations for backend compatibility. Here are the key operations with their signatures:

**Core Operations**
- `cast(x, dtype)` - Cast tensor to different dtype
- `cond(pred, true_fn, false_fn)` - Conditional execution
- `convert_to_tensor(x, dtype=None)` - Convert to tensor
- `fori_loop(lower, upper, body_fun, init_val)` - For loop
- `shape(x)` - Get tensor shape
- `scatter(indices, values, shape)` - Scatter operation
- `slice(inputs, start_indices, slice_sizes)` - Slice tensor
- `stop_gradient(variable)` - Stop gradient flow
- `while_loop(cond, body, loop_vars, maximum_iterations=None)` - While loop

**NumPy API** (`keras.ops.numpy`)

*Array Creation:*
- `arange(start, stop=None, step=1, dtype=None)`
- `array(object, dtype=None)`
- `eye(N, M=None, k=0, dtype=None)`
- `ones(shape, dtype=None)` / `zeros(shape, dtype=None)`
- `full(shape, fill_value, dtype=None)`
- `linspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)`
- `meshgrid(*xi, indexing="xy")`

*Array Manipulation:*
- `broadcast_to(x, shape)`
- `concatenate(tensors, axis=0)`
- `expand_dims(x, axis)`
- `pad(x, pad_width, mode="constant")`
- `reshape(x, newshape)`
- `transpose(x, axes=None)`
- `stack(tensors, axis=0)` / `unstack(x, axis=0)`

*Mathematical Operations:*
- `add(x1, x2)`, `subtract(x1, x2)`, `multiply(x1, x2)`, `divide(x1, x2)`
- `exp(x)`, `log(x)`, `sqrt(x)`, `square(x)`
- `sin(x)`, `cos(x)`, `tan(x)`, `sinh(x)`, `cosh(x)`, `tanh(x)`
- `maximum(x1, x2)`, `minimum(x1, x2)`
- `clip(x, a_min, a_max)`

*Reductions:*
- `sum(x, axis=None, keepdims=False, dtype=None)`
- `mean(x, axis=None, keepdims=False, dtype=None)`
- `std(x, axis=None, keepdims=False, ddof=0, dtype=None)`
- `max(x, axis=None, keepdims=False)` / `min(x, axis=None, keepdims=False)`
- `argmax(x, axis=None)` / `argmin(x, axis=None)`

*Linear Algebra:*
- `dot(a, b)` / `matmul(x1, x2)`
- `einsum(subscripts, *operands, **kwargs)`
- `tensordot(a, b, axes=2)`

**Neural Network Ops** (`keras.ops.nn`)

*Activations:*
- `relu(x)`, `sigmoid(x)`, `tanh(x)`
- `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)`
- `elu(x, alpha=1.0)`, `selu(x)`
- `leaky_relu(x, negative_slope=0.2)`
- `gelu(x, approximate=True)`
- `silu(x)` / `swish(x)`

*Convolutions & Pooling:*
- `conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1)`
- `conv_transpose(inputs, kernel, strides, padding="valid", output_padding=None, data_format=None, dilation_rate=1)`
- `depthwise_conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1)`
- `max_pool(inputs, pool_size, strides=None, padding="valid", data_format=None)`
- `average_pool(inputs, pool_size, strides=None, padding="valid", data_format=None)`

*Normalization & Utilities:*
- `batch_normalization(x, mean, variance, axis=-1, offset=None, scale=None, epsilon=1e-3)`
- `one_hot(x, num_classes, axis=-1, dtype="float32")`
- `moments(x, axes, keepdims=False, synchronized=False)`

*Loss Functions:*
- `binary_crossentropy(y_true, y_pred, from_logits=False)`
- `categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)`
- `sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)`

**Linear Algebra** (`keras.ops.linalg`)
- `cholesky(x)`, `inv(x)`, `det(x)`
- `norm(x, ord=None, axis=None, keepdims=False)`
- `qr(x, mode="reduced")`, `svd(x, full_matrices=True, compute_uv=True)`
- `solve(a, b)`

**Image Operations** (`keras.ops.image`)
- `resize(image, size, interpolation="bilinear", antialias=False, data_format=None)`
- `affine_transform(image, transform, interpolation="bilinear", fill_mode="constant", fill_value=0, data_format=None)`

### Model Persistence
- Format: `.keras` files
- Custom objects: Always provide when loading
  ```python
  model = keras.models.load_model(
      "model.keras",
      custom_objects={"CustomLayer": CustomLayer}
  )
  ```