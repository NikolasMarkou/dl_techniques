# Layers

Individual neural network layers and building blocks

**252 modules in this category**

## Activations

### layers.activations

*📁 File: `src/dl_techniques/layers/activations/__init__.py`*

### layers.activations.adaptive_softmax
An adaptive softmax with entropy-based temperature scaling.

**Classes:**

- `AdaptiveTemperatureSoftmax` - Keras Layer
  Adaptive Temperature Softmax layer with entropy-based temperature adaptation.
  ```python
  AdaptiveTemperatureSoftmax(min_temp: float = 0.1, max_temp: float = 1.0, entropy_threshold: float = 0.5, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/adaptive_softmax.py`*

### layers.activations.basis_function
Swish activation function, used as a non-linear basis.

**Classes:**

- `BasisFunction` - Keras Layer
  Basis function layer implementing the Swish activation: b(x) = x / (1 + e^(-x)).
  ```python
  BasisFunction(**kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/basis_function.py`*

### layers.activations.differentiable_step
A learnable, differentiable approximation of a step function.

**Classes:**

- `DifferentiableStep` - Keras Layer
  A learnable, differentiable step function, configurable for scalar or per-axis operation.
  ```python
  DifferentiableStep(axis: Optional[int] = -1, slope_initializer: Union[str, keras.initializers.Initializer] = 'ones', shift_initializer: Union[str, keras.initializers.Initializer] = 'zeros', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/differentiable_step.py`*

### layers.activations.expanded_activations
==========================================

**Classes:**

- `BaseActivation` - Keras Layer
  Base class for all custom activation functions.
  ```python
  BaseActivation(trainable: bool = True, name: Optional[str] = None, dtype: Optional[Union[str, keras.ops.dtype]] = None, ...)
  ```
- `GELU`
- `SiLU`
- `ExpandedActivation`
- `xATLU`
- `xGELU`
- `xSiLU`
- `EluPlusOne`

**Functions:** `elu_plus_one_plus_epsilon`, `get_activation`, `get_config`, `call`, `call` (and 6 more)

*📁 File: `src/dl_techniques/layers/activations/expanded_activations.py`*

### layers.activations.factory
Activation Layer Factory for dl_techniques Framework

**Functions:** `get_activation_info`, `validate_activation_config`, `create_activation_layer`, `create_activation_from_config`

*📁 File: `src/dl_techniques/layers/activations/factory.py`*

### layers.activations.golu
A Gompertz Linear Unit (GoLU) activation function.

**Classes:**

- `GoLU` - Keras Layer
  Gompertz Linear Unit (GoLU) activation function layer.
  ```python
  GoLU(alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/golu.py`*

### layers.activations.hard_sigmoid
A computationally efficient, piecewise linear sigmoid approximation.

**Classes:**

- `HardSigmoid` - Keras Layer
  Hard-sigmoid activation function for efficient sigmoid approximation.
  ```python
  HardSigmoid(**kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/hard_sigmoid.py`*

### layers.activations.hard_swish
HardSwish activation, a computationally efficient Swish variant.

**Classes:**

- `HardSwish` - Keras Layer
  Hard-swish activation function for efficient mobile-optimized neural networks.
  ```python
  HardSwish(**kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/hard_swish.py`*

### layers.activations.mish
Mish self-regularized, non-monotonic activation function.

**Classes:**

- `Mish` - Keras Layer
  Mish activation function layer.
  ```python
  Mish(**kwargs)
  ```

- `SaturatedMish` - Keras Layer
  SaturatedMish activation function with continuous transition at alpha.
  ```python
  SaturatedMish(alpha: float = 3.0, beta: float = 0.5, **kwargs)
  ```

**Functions:** `mish`, `saturated_mish`, `call`, `compute_output_shape`, `get_config` (and 3 more)

*📁 File: `src/dl_techniques/layers/activations/mish.py`*

### layers.activations.monotonicity_layer
Monotonicity enforcement layer for neural networks.

**Classes:**

- `MonotonicityLayer` - Keras Layer
  Enforces monotonic (non-decreasing) constraints on predictions.
  ```python
  MonotonicityLayer(method: MonotonicityMethod = 'cumulative_softplus', axis: int = -1, min_spacing: Optional[float] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/monotonicity_layer.py`*

### layers.activations.probability_output
Unified Probability Output Layer.

**Classes:**

- `ProbabilityOutput` - Keras Layer
  Unified wrapper for probability output layers.
  ```python
  ProbabilityOutput(probability_type: ProbabilityType = 'softmax', type_config: Optional[Dict[str, Any]] = None, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config` (and 2 more)

*📁 File: `src/dl_techniques/layers/activations/probability_output.py`*

### layers.activations.relu_k
A powered ReLU activation function, `f(x) = max(0, x)^k`.

**Classes:**

- `ReLUK` - Keras Layer
  ReLU-k activation layer implementing f(x) = max(0, x)^k.
  ```python
  ReLUK(k: int = 3, **kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/relu_k.py`*

### layers.activations.routing_probabilities
Deterministic, parameter-free routing tree for classification.

**Classes:**

- `RoutingProbabilitiesLayer` - Keras Layer
  Non-trainable hierarchical routing layer for probabilistic classification.
  ```python
  RoutingProbabilitiesLayer(output_dim: Optional[int] = None, axis: int = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/routing_probabilities.py`*

### layers.activations.routing_probabilities_hierarchical
Trainable hierarchical routing tree for large-scale classification.

**Classes:**

- `HierarchicalRoutingLayer` - Keras Layer
  Trainable hierarchical routing layer for probabilistic classification.
  ```python
  HierarchicalRoutingLayer(output_dim: int, axis: int = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/routing_probabilities_hierarchical.py`*

### layers.activations.sparsemax
Projects a vector of logits onto the probability simplex for sparse outputs.

**Classes:**

- `Sparsemax` - Keras Layer
  Sparsemax activation function layer for sparse probability distributions.
  ```python
  Sparsemax(axis: int = -1, **kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/sparsemax.py`*

### layers.activations.squash
Vector squashing non-linearity for Capsule Networks.

**Classes:**

- `SquashLayer` - Keras Layer
  Applies squashing non-linearity to vectors (capsules).
  ```python
  SquashLayer(axis: int = -1, epsilon: Optional[float] = None, **kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/squash.py`*

### layers.activations.thresh_max
A sparse softmax variant using differentiable confidence thresholding.

**Classes:**

- `ThreshMax` - Keras Layer
  ThreshMax activation layer with learnable sparsity.
  ```python
  ThreshMax(axis: int = -1, slope: float = 10.0, epsilon: float = 1e-12, ...)
  ```

**Functions:** `thresh_max`, `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/activations/thresh_max.py`*

## Anchor_Generator

### layers.anchor_generator
Generate a multi-scale grid of anchor points for object detection.

**Classes:**

- `AnchorGenerator` - Keras Layer
  Anchor generator layer for YOLOv12 object detection.
  ```python
  AnchorGenerator(input_image_shape: Tuple[int, int], strides_config: Optional[List[int]] = None, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `total_anchor_points`, `get_config`

*📁 File: `src/dl_techniques/layers/anchor_generator.py`*

## Attention

### layers.attention
Attention Layers Module.

*📁 File: `src/dl_techniques/layers/attention/__init__.py`*

### layers.attention.anchor_attention
A hierarchical, memory-efficient anchor-based attention layer.

**Classes:**

- `AnchorAttention` - Keras Layer
  Hierarchical attention mechanism with anchor-based information bottleneck.
  ```python
  AnchorAttention(dim: int, num_heads: int, head_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/anchor_attention.py`*

### layers.attention.capsule_routing_attention
Capsule-based dynamic routing mechanism.

**Classes:**

- `CapsuleRoutingSelfAttention` - Keras Layer
  Capsule Routing Self-Attention mechanism from Capsule-Transformer.
  ```python
  CapsuleRoutingSelfAttention(num_heads: int, key_dim: Optional[int] = None, value_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/capsule_routing_attention.py`*

### layers.attention.channel_attention
Channel-wise attention weights for convolutional feature maps.

**Classes:**

- `ChannelAttention` - Keras Layer
  Channel attention module of CBAM (Convolutional Block Attention Module).
  ```python
  ChannelAttention(channels: int, ratio: int = 8, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/channel_attention.py`*

### layers.attention.convolutional_block_attention
Convolutional Block Attention Module (CBAM), a lightweight and

**Classes:**

- `CBAM` - Keras Layer
  Convolutional Block Attention Module for feature refinement.
  ```python
  CBAM(channels: int, ratio: int = 8, kernel_size: int = 7, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/convolutional_block_attention.py`*

### layers.attention.differential_attention
Differential Multi-Head Attention Implementation

**Classes:**

- `DifferentialMultiHeadAttention` - Keras Layer
  Differential multi-head attention mechanism.
  ```python
  DifferentialMultiHeadAttention(dim: int, num_heads: int, head_dim: int, ...)
  ```

**Functions:** `build`, `get_lambda`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/differential_attention.py`*

### layers.attention.factory
Attention Layer Factory

**Functions:** `get_attention_info`, `validate_attention_config`, `create_attention_layer`, `create_attention_from_config`, `list_attention_types` (and 1 more)

*📁 File: `src/dl_techniques/layers/attention/factory.py`*

### layers.attention.fnet_fourier_transform
Mix tokens using a parameter-free 2D Discrete Fourier Transform.

**Classes:**

- `FNetFourierTransform` - Keras Layer
  FNet Fourier Transform layer that replaces self-attention with parameter-free mixing.
  ```python
  FNetFourierTransform(implementation: Literal['matrix', 'fft'] = 'matrix', normalize_dft: bool = True, epsilon: float = 1e-12, ...)
  ```

**Functions:** `build`, `call`, `compute_mask`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/fnet_fourier_transform.py`*

### layers.attention.gated_attention
A gated multi-head attention with rotary position embeddings.

**Classes:**

- `GatedAttention` - Keras Layer
  Gated Attention layer with normalization, partial RoPE, and output gating.
  ```python
  GatedAttention(dim: int, num_heads: int, head_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `scaled_dot_product_attention`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/gated_attention.py`*

### layers.attention.group_query_attention
Grouped Query Attention (GQA) Implementation with Rotary Position Embeddings

**Classes:**

- `GroupedQueryAttention` - Keras Layer
  Grouped Query Attention layer with optional rotary position embeddings.
  ```python
  GroupedQueryAttention(dim: int, num_heads: int, num_kv_heads: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/group_query_attention.py`*

### layers.attention.hopfield_attention
Modern Hopfield Network Layer with Iterative Updates.

**Classes:**

- `HopfieldAttention` - Keras Layer
  Modern Hopfield layer implementation as described in 'Hopfield Networks is All You Need'.
  ```python
  HopfieldAttention(num_heads: int, key_dim: int, value_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/hopfield_attention.py`*

### layers.attention.mobile_mqa
An efficient multi-query attention mechanism for mobile devices.

**Classes:**
- `MobileMQA`

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/mobile_mqa.py`*

### layers.attention.multi_head_attention
Pairwise relationships between elements in a sequence.

**Classes:**

- `MultiHeadAttention` - Keras Layer
  Multi-Head Self-Attention mechanism with comprehensive masking support.
  ```python
  MultiHeadAttention(dim: int, num_heads: int = 8, dropout_rate: float = 0.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/multi_head_attention.py`*

### layers.attention.multi_head_cross_attention
A unified multi-head attention with adaptive temperature and optional hierarchical routing

**Classes:**

- `MultiHeadCrossAttention` - Keras Layer
  Unified, highly configurable multi-head attention layer with advanced features.
  ```python
  MultiHeadCrossAttention(dim: int, num_heads: int = 8, dropout_rate: float = 0.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/multi_head_cross_attention.py`*

### layers.attention.multi_head_latent_attention
Multi-Head Latent Attention (MLA) Layer.

**Classes:**

- `MultiHeadLatentAttention` - Keras Layer
  Multi-Head Latent Attention (MLA) as proposed in DeepSeek-V2.
  ```python
  MultiHeadLatentAttention(dim: int, num_heads: int, kv_latent_dim: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/multi_head_latent_attention.py`*

### layers.attention.non_local_attention
A self-attention mechanism for computer vision_heads tasks,

**Classes:**

- `NonLocalAttention` - Keras Layer
  Non-local Self Attention Layer for computer vision_heads tasks.
  ```python
  NonLocalAttention(attention_channels: int, kernel_size: Union[int, Tuple[int, int]] = (7, 7), use_bias: bool = False, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/non_local_attention.py`*

### layers.attention.perceiver_attention
Asymmetric cross-attention from the Perceiver architecture.

**Classes:**

- `PerceiverAttention` - Keras Layer
  Cross-attention mechanism from the Perceiver architecture with robust serialization.
  ```python
  PerceiverAttention(dim: int, num_heads: int = 8, dropout_rate: float = 0.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/perceiver_attention.py`*

### layers.attention.performer_attention
Approximates softmax attention with linear complexity using random features.

**Classes:**

- `PerformerAttention` - Keras Layer
  Performer attention layer with linear complexity via FAVOR+ approximation.
  ```python
  PerformerAttention(dim: int, num_heads: int = 8, nb_features: int = 256, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/performer_attention.py`*

### layers.attention.progressive_focused_attention
Progressive Focused Attention (PFA) Module for PFT-SR.

**Classes:**

- `ProgressiveFocusedAttention` - Keras Layer
  Progressive Focused Attention mechanism with windowed self-attention.
  ```python
  ProgressiveFocusedAttention(dim: int, num_heads: int, window_size: int = 8, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `from_config`, `compute_output_shape`

*📁 File: `src/dl_techniques/layers/attention/progressive_focused_attention.py`*

### layers.attention.ring_attention
Exact attention for long sequences via blockwise processing.

**Classes:**

- `RingAttention` - Keras Layer
  Ring Attention layer with blockwise processing for extremely long sequences.
  ```python
  RingAttention(dim: int, num_heads: int = 8, block_size: int = 512, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/ring_attention.py`*

### layers.attention.rpc_attention
A robust attention mechanism via Principal Component Pursuit.

**Classes:**

- `RPCAttention` - Keras Layer
  Robust Principal Components Attention layer.
  ```python
  RPCAttention(dim: int, num_heads: int = 8, lambda_sparse: float = 0.1, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/rpc_attention.py`*

### layers.attention.shared_weights_cross_attention
A parameter-efficient, bidirectional cross-attention mechanism.

**Classes:**

- `SharedWeightsCrossAttention` - Keras Layer
  Cross-attention between different modalities with shared weights.
  ```python
  SharedWeightsCrossAttention(dim: int, num_heads: int = 8, dropout_rate: float = 0.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/shared_weights_cross_attention.py`*

### layers.attention.single_window_attention

**Classes:**

- `SingleWindowAttention` - Keras Layer
  Unified multi-head self-attention for a single window.
  ```python
  SingleWindowAttention(dim: int, window_size: int, num_heads: int, ...)
  ```

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/single_window_attention.py`*

### layers.attention.spatial_attention
A spatial attention map for convolutional feature maps.

**Classes:**

- `SpatialAttention` - Keras Layer
  Spatial attention module of CBAM (Convolutional Block Attention Module).
  ```python
  SpatialAttention(kernel_size: int = 7, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', kernel_regularizer: Optional[keras.regularizers.Regularizer] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/attention/spatial_attention.py`*

### layers.attention.tripse_attention
TripSE: Triplet Squeeze and Excitation Attention Block.

**Classes:**

- `TripletAttentionBranch` - Keras Layer
  Single branch of Triplet Attention mechanism.
  ```python
  TripletAttentionBranch(kernel_size: int = 7, permute_pattern: Tuple[int, int, int] = (0, 1, 2), use_bias: bool = False, ...)
  ```

- `TripSE1` - Keras Layer
  TripSE1: Triplet Attention with Post-Fusion Squeeze-and-Excitation.
  ```python
  TripSE1(reduction_ratio: float = 0.0625, kernel_size: int = 7, use_bias: bool = False, ...)
  ```

- `TripSE2` - Keras Layer
  TripSE2: Pre-Process Squeeze-and-Excitation.
  ```python
  TripSE2(reduction_ratio: float = 0.0625, kernel_size: int = 7, use_bias: bool = False, ...)
  ```

- `TripSE3` - Keras Layer
  TripSE3: Parallel Squeeze-and-Excitation.
  ```python
  TripSE3(reduction_ratio: float = 0.0625, kernel_size: int = 7, use_bias: bool = False, ...)
  ```

- `TripSE4` - Keras Layer
  TripSE4: Hybrid 3D Attention with Affine Fusion.
  ```python
  TripSE4(reduction_ratio: float = 0.0625, kernel_size: int = 7, use_bias: bool = False, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 15 more)

*📁 File: `src/dl_techniques/layers/attention/tripse_attention.py`*

### layers.attention.window_attention
Unified windowed multi-head self-attention for sequence processing.

**Classes:**

- `WindowAttention` - Keras Layer
  Unified window-based multi-head self-attention layer.
  ```python
  WindowAttention(dim: int, window_size: int, num_heads: int, ...)
  ```

**Functions:** `create_grid_window_attention`, `create_zigzag_window_attention`, `create_kan_key_window_attention`, `create_adaptive_softmax_window_attention`, `build` (and 4 more)

*📁 File: `src/dl_techniques/layers/attention/window_attention.py`*

## Bias_Free_Conv1D

### layers.bias_free_conv1d
Bias-Free 1D Convolutional Layer

**Classes:**

- `BiasFreeConv1D` - Keras Layer
  Bias-free 1D convolutional layer with batch normalization and activation.
  ```python
  BiasFreeConv1D(filters: int, kernel_size: int = 3, activation: Union[str, callable] = 'relu', ...)
  ```

- `BiasFreeResidualBlock1D` - Keras Layer
  Bias-free residual block for ResNet-style architecture with 1D convolutions.
  ```python
  BiasFreeResidualBlock1D(filters: int, kernel_size: int = 3, activation: Union[str, callable] = 'relu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 3 more)

*📁 File: `src/dl_techniques/layers/bias_free_conv1d.py`*

## Bias_Free_Conv2D

### layers.bias_free_conv2d
Bias-Free 2D Convolutional Layer

**Classes:**

- `BiasFreeConv2D` - Keras Layer
  Bias-free 2D convolutional layer with batch normalization and activation.
  ```python
  BiasFreeConv2D(filters: int, kernel_size: Union[int, Tuple[int, int]] = 3, activation: Union[str, callable] = 'relu', ...)
  ```

- `BiasFreeResidualBlock` - Keras Layer
  Bias-free residual block for ResNet-style architecture with 2D convolutions.
  ```python
  BiasFreeResidualBlock(filters: int, kernel_size: Union[int, Tuple[int, int]] = 3, activation: Union[str, callable] = 'relu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 3 more)

*📁 File: `src/dl_techniques/layers/bias_free_conv2d.py`*

## Bitlinear_Layer

### layers.bitlinear_layer
A bit-quantized linear layer for efficient inference.

**Classes:**

- `BitLinear` - Keras Layer
  Bit-aware linear layer for quantization-aware training.
  ```python
  BitLinear(units: int, weight_bits: Union[float, int, Tuple[float, float]] = 1.58, activation_bits: Union[float, int, Tuple[float, float]] = 8, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/bitlinear_layer.py`*

## Blt_Blocks

### layers.blt_blocks
Byte Latent Transformer (BLT) Core Layer Components

**Classes:**

- `ByteTokenizer` - Keras Layer
  Converts text to byte tokens for BLT processing.
  ```python
  ByteTokenizer(vocab_size: int = 260, byte_offset: int = 4, **kwargs)
  ```

- `EntropyModel` - Keras Layer
  Small causal transformer for computing next-byte entropy.
  ```python
  EntropyModel(vocab_size: int = 260, hidden_dim: int = 256, num_layers: int = 6, ...)
  ```

- `DynamicPatcher` - Keras Layer
  Creates dynamic patches based on entropy thresholding.
  ```python
  DynamicPatcher(entropy_threshold: float = 1.5, max_patches: int = 512, **kwargs)
  ```

- `PatchPooling` - Keras Layer
  Pools byte representations within patches to create patch representations.
  ```python
  PatchPooling(pooling_method: str = 'attention', output_dim: int = 768, num_queries: int = 4, ...)
  ```

- `LocalEncoder` - Keras Layer
  Local Encoder for BLT that processes bytes within their patches.
  ```python
  LocalEncoder(vocab_size: int = 260, local_dim: int = 512, num_local_layers: int = 6, ...)
  ```

- `GlobalTransformer` - Keras Layer
  Global Transformer for BLT that processes patch sequences.
  ```python
  GlobalTransformer(global_dim: int = 768, num_global_layers: int = 12, num_heads_global: int = 12, ...)
  ```

- `LocalDecoder` - Keras Layer
  Local Decoder for BLT that generates next byte predictions.
  ```python
  LocalDecoder(vocab_size: int = 260, local_dim: int = 512, global_dim: int = 768, ...)
  ```

**Functions:** `text_to_bytes`, `tokens_to_text`, `get_config`, `build`, `call` (and 23 more)

*📁 File: `src/dl_techniques/layers/blt_blocks.py`*

## Blt_Core

### layers.blt_core
Fuse entropy-based byte processing with iterative hierarchical reasoning.

**Classes:**

- `ByteLatentReasoningCore` - Keras Layer
  Core hierarchical reasoning model that operates on dynamic byte patches.
  ```python
  ByteLatentReasoningCore(vocab_size: int, seq_len: int, embed_dim: int, ...)
  ```

**Functions:** `build`, `empty_carry`, `reset_carry`, `call`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/blt_core.py`*

## Canny

### layers.canny

**Classes:**

- `Canny` - Keras Layer
  Keras implementation of the Canny edge detection algorithm.
  ```python
  Canny(sigma: float = 0.8, threshold_min: int = 50, threshold_max: int = 80, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `compute_output_shape`, `loop_cond` (and 1 more)

*📁 File: `src/dl_techniques/layers/canny.py`*

## Capsules

### layers.capsules
# Capsule Networks (CapsNet) Implementation

**Classes:**

- `PrimaryCapsule` - Keras Layer
  Primary Capsule Layer implementation.
  ```python
  PrimaryCapsule(num_capsules: int, dim_capsules: int, kernel_size: Union[int, Tuple[int, int]], ...)
  ```

- `RoutingCapsule` - Keras Layer
  Capsule layer with dynamic routing between capsules.
  ```python
  RoutingCapsule(num_capsules: int, dim_capsules: int, routing_iterations: int = 3, ...)
  ```

- `CapsuleBlock` - Keras Layer
  A complete capsule block with optional dropout and normalization.
  ```python
  CapsuleBlock(num_capsules: int, dim_capsules: int, routing_iterations: int = 3, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 7 more)

*📁 File: `src/dl_techniques/layers/capsules.py`*

## Clahe

### layers.clahe
local image contrast using a trainable CLAHE algorithm.

**Classes:**

- `CLAHE` - Keras Layer
  Contrast Limited Adaptive Histogram Equalization (CLAHE) layer.
  ```python
  CLAHE(clip_limit: float = 4.0, n_bins: int = 256, tile_size: int = 16, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `compute_output_shape`

*📁 File: `src/dl_techniques/layers/clahe.py`*

## Complex_Layers

### layers.complex_layers
Complex-Valued Neural Network Layers Implementation

**Classes:**

- `ComplexLayer` - Keras Layer
  Base class for complex-valued layers.
  ```python
  ComplexLayer(epsilon: float = 1e-07, kernel_regularizer: Optional[keras.regularizers.Regularizer] = None, kernel_initializer: Optional[keras.initializers.Initializer] = None, ...)
  ```

- `ComplexConv2D` - Keras Layer
  Complex-valued 2D convolution layer with improved numerical stability.
  ```python
  ComplexConv2D(filters: int, kernel_size: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]] = 1, ...)
  ```

- `ComplexDense` - Keras Layer
  Complex-valued dense layer with improved initialization.
  ```python
  ComplexDense(units: int, **kwargs)
  ```

- `ComplexReLU` - Keras Layer
  Complex ReLU activation function.
  ```python
  ComplexReLU(**kwargs)
  ```

- `ComplexAveragePooling2D` - Keras Layer
  Complex-valued 2D average pooling layer.
  ```python
  ComplexAveragePooling2D(pool_size: Union[int, Tuple[int, int]] = (2, 2), strides: Optional[Union[int, Tuple[int, int]]] = None, padding: str = 'VALID', ...)
  ```

- `ComplexDropout` - Keras Layer
  Complex-valued dropout layer for regularization.
  ```python
  ComplexDropout(rate: float, **kwargs)
  ```

- `ComplexGlobalAveragePooling2D` - Keras Layer
  Complex-valued global 2D average pooling layer.
  ```python
  ComplexGlobalAveragePooling2D(keepdims: bool = False, **kwargs)
  ```

**Functions:** `get_config`, `build`, `call`, `compute_output_shape`, `get_config` (and 16 more)

*📁 File: `src/dl_techniques/layers/complex_layers.py`*

## Conditional_Output_Layer

### layers.conditional_output_layer
Selectively route tensors for conditional training or data imputation.

**Classes:**

- `ConditionalOutputLayer` - Keras Layer
  A custom layer for conditional output selection based on ground truth values.
  ```python
  ConditionalOutputLayer(**kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/conditional_output_layer.py`*

## Conv2D_Builder

### layers.conv2d_builder
Advanced Layer and Activation Function Wrappers

**Classes:**
- `ConvType`

**Functions:** `activation_wrapper`, `multiscales_generator_fn`, `conv2d_wrapper`, `multiscale_fn`, `from_string` (and 1 more)

*📁 File: `src/dl_techniques/layers/conv2d_builder.py`*

## Convnext_V1_Block

### layers.convnext_v1_block
ConvNext Block Implementation

**Classes:**

- `ConvNextV1Block` - Keras Layer
  Implementation of ConvNext block with modern best practices.
  ```python
  ConvNextV1Block(kernel_size: Union[int, Tuple[int, int]], filters: int, activation: Union[str, keras.layers.Activation] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/convnext_v1_block.py`*

## Convnext_V2_Block

### layers.convnext_v2_block
ConvNextV2 Block Implementation

**Classes:**

- `ConvNextV2Block` - Keras Layer
  Implementation of ConvNextV2 block with modern best practices.
  ```python
  ConvNextV2Block(kernel_size: Union[int, Tuple[int, int]], filters: int, activation: Union[str, keras.layers.Activation] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/convnext_v2_block.py`*

## Convolutional_Kan

### layers.convolutional_kan
Convolutional Kolmogorov-Arnold Networks Implementation

**Classes:**

- `KANvolution` - Keras Layer
  Kolmogorov-Arnold Network convolution layer with learnable B-spline activations.
  ```python
  KANvolution(filters: int, kernel_size: Union[int, Tuple[int, int]], grid_size: int = 16, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/convolutional_kan.py`*

## Core

### layers

*📁 File: `src/dl_techniques/layers/__init__.py`*

## Depthwise_Separable_Block

### layers.depthwise_separable_block
Depthwise separable convolution block, a core of MobileNet.

**Classes:**

- `DepthwiseSeparableBlock` - Keras Layer
  Configurable depthwise separable convolution block.
  ```python
  DepthwiseSeparableBlock(filters: int, depthwise_kernel_size: Union[int, Tuple[int, int]] = 3, stride: int = 1, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/depthwise_separable_block.py`*

## Downsample

### layers.downsample
Downsampling Module for Neural Networks.

**Functions:** `downsample`

*📁 File: `src/dl_techniques/layers/downsample.py`*

## Dynamic_Conv2D

### layers.dynamic_conv2d
2D convolution with input-dependent dynamic kernel aggregation.

**Classes:**

- `DynamicConv2D` - Keras Layer
  Dynamic 2D Convolution with Attention over Convolution Kernels.
  ```python
  DynamicConv2D(filters: int, kernel_size: Union[int, Tuple[int, int]], num_kernels: int = 4, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `conv_output_length`

*📁 File: `src/dl_techniques/layers/dynamic_conv2d.py`*

## Embedding

### layers.embedding

*📁 File: `src/dl_techniques/layers/embedding/__init__.py`*

### layers.embedding.bert_embeddings
Construct the composite input embeddings for BERT-style models.

**Classes:**

- `BertEmbeddings` - Keras Layer
  BERT embeddings layer combining word, position, and token type embeddings.
  ```python
  BertEmbeddings(vocab_size: int, hidden_size: int, max_position_embeddings: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/embedding/bert_embeddings.py`*

### layers.embedding.continuous_rope_embedding
Continuous, multi-dimensional rotary position embeddings (RoPE).

**Classes:**

- `ContinuousRoPE` - Keras Layer
  Continuous Rotary Position Embedding for variable positions.
  ```python
  ContinuousRoPE(dim: int, ndim: int, max_wavelength: float = 10000.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/continuous_rope_embedding.py`*

### layers.embedding.continuous_sin_cos_embedding
Generates continuous, multi-dimensional positional embeddings using sinusoids.

**Classes:**

- `ContinuousSinCosEmbed` - Keras Layer
  Continuous coordinate embedding using sine and cosine functions.
  ```python
  ContinuousSinCosEmbed(dim: int, ndim: int, max_wavelength: float = 10000.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py`*

### layers.embedding.dual_rotary_position_embedding
A dual-configuration Rotary Position Embedding (RoPE).

**Classes:**

- `DualRotaryPositionEmbedding` - Keras Layer
  Dual Rotary Position Embedding layer for Gemma3-style attention mechanisms.
  ```python
  DualRotaryPositionEmbedding(head_dim: int, max_seq_len: int, global_theta_base: float = 1000000.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/dual_rotary_position_embedding.py`*

### layers.embedding.factory
Embedding Layer Factory for dl_techniques Framework

**Functions:** `get_embedding_info`, `validate_embedding_config`, `create_embedding_layer`, `create_embedding_from_config`

*📁 File: `src/dl_techniques/layers/embedding/factory.py`*

### layers.embedding.modern_bert_embeddings

**Classes:**

- `ModernBertEmbeddings` - Keras Layer
  Computes embeddings for ModernBERT from token and type IDs.
  ```python
  ModernBertEmbeddings(vocab_size: int, hidden_size: int, type_vocab_size: int, ...)
  ```

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/modern_bert_embeddings.py`*

### layers.embedding.patch_embedding
Convert a 2D image into a sequence of flattened patch embeddings.

**Classes:**

- `PatchEmbedding2D` - Keras Layer
  2D Image to Patch Embedding Layer.
  ```python
  PatchEmbedding2D(patch_size: Union[int, Tuple[int, int]], embed_dim: int, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal', ...)
  ```

- `PatchEmbedding1D` - Keras Layer
  Patch embedding layer for time series data.
  ```python
  PatchEmbedding1D(patch_size: int, embed_dim: int, stride: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 3 more)

*📁 File: `src/dl_techniques/layers/embedding/patch_embedding.py`*

### layers.embedding.positional_embedding
Inject positional information into a sequence using learnable embeddings.

**Classes:**

- `PositionalEmbedding` - Keras Layer
  Learned positional embedding layer with enhanced stability.
  ```python
  PositionalEmbedding(max_seq_len: int, dim: int, dropout_rate: float = 0.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/positional_embedding.py`*

### layers.embedding.positional_embedding_sine_2d
Generate 2D sinusoidal positional encodings for image-like inputs.

**Classes:**

- `PositionEmbeddingSine2D` - Keras Layer
  Generates 2D sinusoidal positional encodings for image-like feature maps.
  ```python
  PositionEmbeddingSine2D(num_pos_feats: int = 64, temperature: float = 10000.0, normalize: bool = True, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py`*

### layers.embedding.rotary_position_embedding
Applies rotary embeddings to inject relative positional information.

**Classes:**

- `RotaryPositionEmbedding` - Keras Layer
  Rotary Position Embedding layer for transformer attention mechanisms.
  ```python
  RotaryPositionEmbedding(head_dim: int, max_seq_len: int, rope_theta: float = 10000.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/embedding/rotary_position_embedding.py`*

## Eomt_Mask

### layers.eomt_mask
Generate instance segmentation predictions from transformer query tokens.

**Classes:**

- `EomtMask` - Keras Layer
  Configurable mask prediction module for Encoder-only Mask Transformer (EoMT).
  ```python
  EomtMask(num_classes: int, hidden_dims: Optional[List[int]] = None, mask_dim: int = 256, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/eomt_mask.py`*

## Experimental

### layers.experimental

*📁 File: `src/dl_techniques/layers/experimental/__init__.py`*

### layers.experimental.band_rms_ood
BandRMS-OOD: Geometric Out-of-Distribution Detection Layer

**Classes:**

- `BandRMSOOD` - Keras Layer
  BandRMS-OOD: Geometric Out-of-Distribution Detection Layer.
  ```python
  BandRMSOOD(max_band_width: float = 0.1, confidence_type: Literal['magnitude', 'entropy', 'prediction'] = 'magnitude', confidence_weight: float = 1.0, ...)
  ```
- `MultiLayerOODDetector`

**Functions:** `build`, `set_external_confidence`, `estimate_confidence`, `apply_shell_scaling`, `call` (and 12 more)

*📁 File: `src/dl_techniques/layers/experimental/band_rms_ood.py`*

### layers.experimental.contextual_counter_ffn

**Classes:**

- `ContextualCounterFFN` - Keras Layer
  Feed-forward network that modulates sequences through contextual counting.
  ```python
  ContextualCounterFFN(output_dim: int, count_dim: int, counting_scope: Literal['global', 'causal', 'bidirectional'] = 'bidirectional', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/experimental/contextual_counter_ffn.py`*

### layers.experimental.contextual_memory
Contextual Memory Bank Implementation for Keras 3.x

**Classes:**
- `MemoryBankConfig`

- `KeyValueMemoryStore` - Keras Layer
  Key-Value Memory Store for long-term associations.
  ```python
  KeyValueMemoryStore(num_slots: int, memory_dim: int, key_dim: int, ...)
  ```

- `GraphNeuralNetworkLayer` - Keras Layer
  Complete configurable Graph Neural Network for concept relationship modeling.
  ```python
  GraphNeuralNetworkLayer(concept_dim: int, num_layers: int = 3, message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gcn', ...)
  ```

- `TemporalContextEncoder` - Keras Layer
  Temporal Context Encoder using modern Transformer architecture.
  ```python
  TemporalContextEncoder(temporal_dim: int, num_heads: int = 8, num_layers: int = 6, ...)
  ```

- `ContextualMemoryBank` - Keras Layer
  Contextual Memory Bank integrating KV memory, GNN, and temporal encoding.
  ```python
  ContextualMemoryBank(config: Optional[MemoryBankConfig] = None, **kwargs)
  ```

**Functions:** `normalize_adjacency_matrix`, `create_contextual_memory_model`, `build`, `call`, `compute_output_shape` (and 15 more)

*📁 File: `src/dl_techniques/layers/experimental/contextual_memory.py`*

### layers.experimental.field_embeddings
Holonomic Field Embeddings - Geometric Safety Through Lie Group Theory.

**Classes:**

- `LieGroupEmbedding` - Keras Layer
  Maps token IDs to rotation matrices in SO(n) using Lie algebra exponential map.
  ```python
  LieGroupEmbedding(vocab_size: int, embed_dim: int, use_expm: bool = True, ...)
  ```

- `HolonomicPathIntegrator` - Keras Layer
  Computes the path-ordered integral of rotation matrices along a sequence.
  ```python
  HolonomicPathIntegrator(return_sequences: bool = True, name: Optional[str] = None, **kwargs)
  ```

- `ManifoldStressMonitor` - Keras Layer
  Measures geometric "stress" in semantic trajectories for anomaly detection.
  ```python
  ManifoldStressMonitor(aggregation: str = 'mean', epsilon: float = 1e-08, name: Optional[str] = None, ...)
  ```

- `HolonomicFieldProjection` - Keras Layer
  Projects final rotation matrix state to a feature vector for downstream tasks.
  ```python
  HolonomicFieldProjection(projection_type: str = 'reference', reference_vector: Optional[Union[List[float], np.ndarray]] = None, name: Optional[str] = None, ...)
  ```

**Functions:** `build_holonomic_field_model`, `verify_rotation_matrix`, `visualize_stress_trajectory`, `build`, `call` (and 14 more)

*📁 File: `src/dl_techniques/layers/experimental/field_embeddings.py`*

### layers.experimental.graph_mann

**Classes:**

- `GraphMannLayer` - Keras Layer
  Graph Memory-Augmented Neural Network (GMANN) layer based on NTM principles.
  ```python
  GraphMannLayer(num_memory_nodes: int, memory_dim: int, controller_units: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/experimental/graph_mann.py`*

### layers.experimental.hierarchical_evidence_llm
Hierarchical Evidence Support System for LLM Token Generation

**Classes:**

- `EvidenceEncoder` - Keras Layer
  Multi-level evidence encoder for token generation support.
  ```python
  EvidenceEncoder(embed_dim: int = 768, local_window: int = 32, num_heads: int = 12, ...)
  ```

- `HierarchicalEvidenceAggregator` - Keras Layer
  Hierarchical aggregator for evidence-based token generation support.
  ```python
  HierarchicalEvidenceAggregator(embed_dim: int = 768, num_levels: int = 4, pooling_sizes: List[int] = [1, 4, 16, 64], ...)
  ```

- `SupportEmbeddingLayer` - Keras Layer
  Support embedding layer for evidence-based token generation.
  ```python
  SupportEmbeddingLayer(vocab_size: int, embed_dim: int = 768, support_dim: int = 256, ...)
  ```

- `EvidenceSupportedTokenGeneration` - Keras Layer
  Complete evidence-supported token generation system.
  ```python
  EvidenceSupportedTokenGeneration(vocab_size: int, embed_dim: int = 768, max_seq_len: int = 512, ...)
  ```

**Functions:** `analyze_evidence_support`, `create_evidence_supported_language_model`, `call`, `get_config`, `call` (and 5 more)

*📁 File: `src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py`*

### layers.experimental.hierarchical_memory_system
Hierarchical Memory System using Multiple Self-Organizing Map Layers.

**Classes:**

- `HierarchicalMemorySystem` - Keras Layer
  Hierarchical memory system using multiple Self-Organizing Map layers.
  ```python
  HierarchicalMemorySystem(input_dim: int, levels: int = 3, grid_dimensions: int = 2, ...)
  ```

**Functions:** `build`, `call`, `get_level_weights`, `get_all_weights`, `get_grid_shapes` (and 2 more)

*📁 File: `src/dl_techniques/layers/experimental/hierarchical_memory_system.py`*

### layers.experimental.mst_correlation_filter
A high-performance, graph-based structural regularizer for Keras 3.

**Classes:**

- `SystemicGraphFilter` - Keras Layer
  A principled, graph-based filter for correlation matrices.
  ```python
  SystemicGraphFilter(top_k_neighbors: int = 2, n_propagation_steps: int = 3, distance_metric: str = 'sqrt', ...)
  ```
- `StructuredAttention`

**Functions:** `build`, `get_config`, `call`, `compute_output_shape`, `build` (and 1 more)

*📁 File: `src/dl_techniques/layers/experimental/mst_correlation_filter.py`*

## Ffn

### layers.ffn

*📁 File: `src/dl_techniques/layers/ffn/__init__.py`*

### layers.ffn.counting_ffn
A Feed-Forward Network that learns to count features in a sequence.

**Classes:**

- `CountingFFN` - Keras Layer
  A Feed-Forward Network that learns to count events in a sequence.
  ```python
  CountingFFN(output_dim: int, count_dim: int, counting_scope: Literal['global', 'local', 'causal'] = 'local', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/counting_ffn.py`*

### layers.ffn.diff_ffn
A dual-pathway feed-forward network for differential processing.

**Classes:**

- `DifferentialFFN` - Keras Layer
  Differential Feed-Forward Network layer implementing dual-pathway processing.
  ```python
  DifferentialFFN(hidden_dim: int, output_dim: int, branch_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/diff_ffn.py`*

### layers.ffn.factory
A Factory Method design pattern to provide a single,

**Functions:** `get_ffn_info`, `validate_ffn_config`, `create_ffn_layer`, `create_ffn_from_config`

*📁 File: `src/dl_techniques/layers/ffn/factory.py`*

### layers.ffn.gated_mlp
A spatially-gated MLP block as an alternative to self-attention.

**Classes:**

- `GatedMLP` - Keras Layer
  A Gated MLP layer implementation using 1x1 convolutions.
  ```python
  GatedMLP(filters: int, use_bias: bool = True, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/gated_mlp.py`*

### layers.ffn.geglu_ffn
A Gated Linear Unit Feed-Forward Network with GELU activation.

**Classes:**

- `GeGLUFFN` - Keras Layer
  GELU Gated Linear Unit Feed-Forward Network (GeGLU).
  ```python
  GeGLUFFN(hidden_dim: int, output_dim: int, activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/geglu_ffn.py`*

### layers.ffn.glu_ffn
A Gated Linear Unit feed-forward network.

**Classes:**

- `GLUFFN` - Keras Layer
  Gated Linear Unit Feed Forward Network as described in "GLU Variants Improve Transformer".
  ```python
  GLUFFN(hidden_dim: int, output_dim: int, activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'swish', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/glu_ffn.py`*

### layers.ffn.logic_ffn
A feed-forward network that performs soft logical reasoning.

**Classes:**

- `LogicFFN` - Keras Layer
  Logic-based Feed-Forward Network using learnable soft logic operations.
  ```python
  LogicFFN(output_dim: int, logic_dim: int, use_bias: bool = True, ...)
  ```

**Functions:** `create_logic_ffn_standard`, `create_logic_ffn_regularized`, `build`, `call`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/ffn/logic_ffn.py`*

### layers.ffn.mlp
A position-wise Feed-Forward Network from the Transformer.

**Classes:**

- `MLPBlock` - Keras Layer
  MLP block used in Transformers.
  ```python
  MLPBlock(hidden_dim: int, output_dim: int, activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/mlp.py`*

### layers.ffn.orthoglu_ffn
An orthogonally-regularized Gated Linear Unit FFN.

**Classes:**

- `OrthoGLUFFN` - Keras Layer
  Orthogonally-Regularized Gated Linear Unit Feed-Forward Network.
  ```python
  OrthoGLUFFN(hidden_dim: int, output_dim: int, activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/orthoglu_ffn.py`*

### layers.ffn.power_mlp_layer
A dual-branch MLP for enhanced function approximation.

**Classes:**

- `PowerMLPLayer` - Keras Layer
  PowerMLP layer with dual-branch architecture for enhanced expressiveness.
  ```python
  PowerMLPLayer(units: int, k: int = 3, kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/power_mlp_layer.py`*

### layers.ffn.residual_block
A residual block with a learnable projection shortcut.

**Classes:**

- `ResidualBlock` - Keras Layer
  Residual block with linear transformations and configurable activation.
  ```python
  ResidualBlock(hidden_dim: int, output_dim: int, dropout_rate: float = 0.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/residual_block.py`*

### layers.ffn.swiglu_ffn
A SwiGLU feed-forward network, a high-performance FFN variant.

**Classes:**

- `SwiGLUFFN` - Keras Layer
  SwiGLU Feed-Forward Network with gating mechanism.
  ```python
  SwiGLUFFN(output_dim: int, ffn_expansion_factor: int = 4, ffn_multiple_of: int = 256, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `num_parameters`

*📁 File: `src/dl_techniques/layers/ffn/swiglu_ffn.py`*

### layers.ffn.swin_mlp
The MLP block from the Swin Transformer architecture.

**Classes:**

- `SwinMLP` - Keras Layer
  MLP module for Swin Transformer with configurable activation and regularization.
  ```python
  SwinMLP(hidden_dim: int, use_bias: bool = True, output_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/ffn/swin_mlp.py`*

## Fft_Layers

### layers.fft_layers

**Classes:**

- `FFTLayer` - Keras Layer
  Applies 2D Fast Fourier Transform and outputs concatenated real/imag parts.
  ```python
  FFTLayer(**kwargs)
  ```

- `IFFTLayer` - Keras Layer
  Applies 2D Inverse FFT to concatenated real/imag parts.
  ```python
  IFFTLayer(**kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `call`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/fft_layers.py`*

## Film

### layers.film
Feature-wise Linear Modulation (FiLM) Layer

**Classes:**

- `FiLMLayer` - Keras Layer
  Highly configurable Feature-wise Linear Modulation (FiLM) Layer.
  ```python
  FiLMLayer(gamma_units: Optional[int] = None, beta_units: Optional[int] = None, gamma_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'tanh', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/film.py`*

## Fnet_Encoder_Block

### layers.fnet_encoder_block
An FNet block using Fourier Transforms for token mixing.

**Classes:**

- `FNetEncoderBlock` - Keras Layer
  Complete FNet encoder block with Fourier mixing and feed-forward components using factory patterns.
  ```python
  FNetEncoderBlock(intermediate_dim: Optional[int] = None, dropout_rate: float = 0.1, fourier_config: Optional[Dict[str, Any]] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_mask`, `compute_output_shape`, `get_config` (and 1 more)

*📁 File: `src/dl_techniques/layers/fnet_encoder_block.py`*

## Fractal_Block

### layers.fractal_block
A recursive fractal block from the FractalNet architecture.

**Classes:**

- `FractalBlock` - Keras Layer
  Recursive fractal block implementing the fractal expansion rule for FractalNet.
  ```python
  FractalBlock(block_config: Dict[str, Any], depth: int = 1, drop_path_rate: float = 0.15, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/fractal_block.py`*

## Fusion

### layers.fusion

*📁 File: `src/dl_techniques/layers/fusion/__init__.py`*

### layers.fusion.multimodal_fusion
A unified framework for multi-modal information fusion.

**Classes:**

- `MultiModalFusion` - Keras Layer
  General-purpose configurable multi-modal fusion layer.
  ```python
  MultiModalFusion(dim: int = 768, fusion_strategy: FusionStrategy = 'cross_attention', num_fusion_layers: int = 1, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/fusion/multimodal_fusion.py`*

## Gated_Delta_Net

### layers.gated_delta_net
A Gated Delta Network, a linear-time transformer variant.

**Classes:**

- `GatedDeltaNet` - Keras Layer
  Gated DeltaNet layer combining delta rule updates with adaptive gating mechanism.
  ```python
  GatedDeltaNet(dim: int, num_heads: int, max_seq_len: int, ...)
  ```

**Functions:** `build`, `delta_rule_update`, `call`, `compute_output_shape`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/layers/gated_delta_net.py`*

## Gaussian_Filter

### layers.gaussian_filter
2D Gaussian blur using a depthwise convolution.

**Classes:**

- `GaussianFilter` - Keras Layer
  Applies Gaussian blur filter to input images.
  ```python
  GaussianFilter(kernel_size: Tuple[int, int] = (5, 5), strides: Union[Tuple[int, int], List[int]] = (1, 1), sigma: Union[float, Tuple[float, float]] = -1, ...)
  ```

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/layers/gaussian_filter.py`*

## Gaussian_Pyramid

### layers.gaussian_pyramid
Construct a multi-scale Gaussian pyramid representation of an image.

**Classes:**

- `GaussianPyramid` - Keras Layer
  Gaussian Pyramid layer for multi-scale image representation.
  ```python
  GaussianPyramid(levels: int = 3, kernel_size: Tuple[int, int] = (5, 5), sigma: Union[float, Tuple[float, float]] = -1, ...)
  ```

**Functions:** `gaussian_pyramid`, `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/gaussian_pyramid.py`*

## Geometric

### layers.geometric

*📁 File: `src/dl_techniques/layers/geometric/__init__.py`*

### layers.geometric.fields
Holonomic Field Layers for Deep Learning.

**Functions:** `validate_field_config`, `create_field_layer`, `create_field_layer_from_config`, `get_field_layer_info`

*📁 File: `src/dl_techniques/layers/geometric/fields/__init__.py`*

### layers.geometric.fields.connection_layer
Connection Layer.

**Classes:**

- `ConnectionLayer` - Keras Layer
  Computes the gauge connection from field representations.
  ```python
  ConnectionLayer(hidden_dim: int, connection_dim: Optional[int] = None, connection_type: ConnectionType = 'yang_mills', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/fields/connection_layer.py`*

### layers.geometric.fields.field_embedding
Field Embedding Layer.

**Classes:**

- `FieldEmbedding` - Keras Layer
  Field Embedding layer that maps tokens to fields with curvature.
  ```python
  FieldEmbedding(vocab_size: int, embed_dim: int, curvature_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/fields/field_embedding.py`*

### layers.geometric.fields.gauge_invariant_attention
Gauge-Invariant Attention Layer.

**Classes:**

- `GaugeInvariantAttention` - Keras Layer
  Attention mechanism that respects gauge invariance.
  ```python
  GaugeInvariantAttention(hidden_dim: int, num_heads: int = 8, key_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/fields/gauge_invariant_attention.py`*

### layers.geometric.fields.holonomic_transformer
Holonomic Transformer Layer.

**Classes:**

- `FieldNormalization` - Keras Layer
  Field-aware normalization that respects curvature.
  ```python
  FieldNormalization(epsilon: float = 1e-06, use_curvature_scaling: bool = True, center: bool = True, ...)
  ```

- `HolonomicTransformerLayer` - Keras Layer
  Complete Holonomic Transformer Layer.
  ```python
  HolonomicTransformerLayer(hidden_dim: int, num_heads: int = 8, ffn_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `build`, `call` (and 2 more)

*📁 File: `src/dl_techniques/layers/geometric/fields/holonomic_transformer.py`*

### layers.geometric.fields.holonomy_layer
Holonomy Layer

**Classes:**

- `HolonomyLayer` - Keras Layer
  Computes holonomy (path-ordered exponential around loops).
  ```python
  HolonomyLayer(hidden_dim: int, loop_sizes: List[int] = [2, 4, 8], loop_type: LoopType = 'rectangular', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/fields/holonomy_layer.py`*

### layers.geometric.fields.manifold_stress
Manifold Stress Layer

**Classes:**

- `ManifoldStressLayer` - Keras Layer
  Computes manifold stress for anomaly and adversarial detection.
  ```python
  ManifoldStressLayer(hidden_dim: int, stress_types: List[str] = ['curvature', 'connection', 'combined'], stress_threshold: float = 0.5, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/fields/manifold_stress.py`*

### layers.geometric.fields.parallel_transport
Parallel Transport Layer

**Classes:**

- `ParallelTransportLayer` - Keras Layer
  Parallel transport of vectors along paths using the gauge connection.
  ```python
  ParallelTransportLayer(transport_dim: int, num_steps: int = 10, transport_method: TransportMethod = 'iterative', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/fields/parallel_transport.py`*

### layers.geometric.point_cloud_autoencoder

**Classes:**

- `PointCloudAutoencoder` - Keras Layer
  Modified DGCNN-based autoencoder for point cloud feature extraction.
  ```python
  PointCloudAutoencoder(k_neighbors: int = 20, **kwargs)
  ```

- `CorrespondenceNetwork` - Keras Layer
  Augmented regression network to estimate point-to-GMM correspondences.
  ```python
  CorrespondenceNetwork(num_gaussians: int, **kwargs)
  ```

**Functions:** `build`, `call`, `get_config`, `build`, `call` (and 1 more)

*📁 File: `src/dl_techniques/layers/geometric/point_cloud_autoencoder.py`*

### layers.geometric.supernode_pooling

**Classes:**

- `SupernodePooling` - Keras Layer
  Supernode pooling layer with message passing for point clouds.
  ```python
  SupernodePooling(hidden_dim: int, ndim: int, radius: Optional[float] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/geometric/supernode_pooling.py`*

## Global_Sum_Pool_2D

### layers.global_sum_pool_2d
Pool features globally by summing over spatial dimensions.

**Classes:**

- `GlobalSumPooling2D` - Keras Layer
  Global sum pooling operation for 2D spatial data.
  ```python
  GlobalSumPooling2D(keepdims: bool = False, data_format: Optional[str] = None, **kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/global_sum_pool_2d.py`*

## Graphs

### layers.graphs

*📁 File: `src/dl_techniques/layers/graphs/__init__.py`*

### layers.graphs.entity_graph_refinement
Learn a dynamic, sparse, directed graph of entity relationships.

**Classes:**

- `EntityGraphRefinement` - Keras Layer
  Entity-Graph Refinement Component for learning hierarchical relationships in embedding space.
  ```python
  EntityGraphRefinement(max_entities: int, entity_dim: int, num_refinement_steps: int = 3, ...)
  ```

**Functions:** `get_graph_statistics`, `extract_hierarchies`, `build`, `call`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/graphs/entity_graph_refinement.py`*

### layers.graphs.fermi_diract_decoder
Fermi-Dirac Decoder for Link Prediction.

**Classes:**

- `FermiDiracDecoder` - Keras Layer
  Fermi-Dirac decoder for edge probability prediction using Euclidean distances.
  ```python
  FermiDiracDecoder(r_initializer: Union[str, keras.initializers.Initializer] = None, t_initializer: Union[str, keras.initializers.Initializer] = None, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/graphs/fermi_diract_decoder.py`*

### layers.graphs.graph_neural_network
A configurable multi-paradigm Graph Neural Network.

**Classes:**

- `GraphNeuralNetworkLayer` - Keras Layer
  Complete configurable Graph Neural Network for concept relationship modeling.
  ```python
  GraphNeuralNetworkLayer(concept_dim: int, num_layers: int = 3, message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gcn', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/graphs/graph_neural_network.py`*

### layers.graphs.relational_graph_transformer_blocks
Relational Graph Transformer (RELGT) Building Blocks.

**Classes:**

- `LightweightGNNLayer` - Keras Layer
  Lightweight Graph Convolutional Network layer for structural encoding.
  ```python
  LightweightGNNLayer(units: int, activation: Optional[Union[str, Callable]] = 'relu', kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', ...)
  ```

- `RELGTTokenEncoder` - Keras Layer
  Multi-element tokenization encoder for heterogeneous graph nodes.
  ```python
  RELGTTokenEncoder(embedding_dim: int, num_node_types: int, max_hops: int = 2, ...)
  ```

- `RELGTTransformerBlock` - Keras Layer
  Hybrid local-global Transformer block for relational graph processing.
  ```python
  RELGTTransformerBlock(embedding_dim: int, num_heads: int, num_global_centroids: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 7 more)

*📁 File: `src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py`*

### layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer
Simplified Hyperbolic Graph Convolutional Neural Network Layer.

**Classes:**

- `SHGCNLayer` - Keras Layer
  Simplified Hyperbolic Graph Convolutional Layer.
  ```python
  SHGCNLayer(units: int, activation: Union[str, callable] = 'relu', use_bias: bool = True, ...)
  ```

**Functions:** `build`, `curvature`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py`*

## Haar_Wavelet_Decomposition

### layers.haar_wavelet_decomposition
Haar Wavelet Decomposition Layer supporting multi-dimensional inputs.

**Classes:**

- `HaarWaveletDecomposition` - Keras Layer
  Performs Haar Discrete Wavelet Transform (DWT) decomposition.
  ```python
  HaarWaveletDecomposition(num_levels: int = 3, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/haar_wavelet_decomposition.py`*

## Hanc_Block

### layers.hanc_block
Model long-range dependencies using hierarchical context aggregation.

**Classes:**

- `HANCBlock` - Keras Layer
  Hierarchical Aggregation of Neighborhood Context (HANC) Block.
  ```python
  HANCBlock(filters: int, input_channels: int, k: int = 3, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/hanc_block.py`*

## Hanc_Layer

### layers.hanc_layer
Approximate self-attention by aggregating hierarchical neighborhood context.

**Classes:**

- `HANCLayer` - Keras Layer
  Hierarchical Aggregation of Neighborhood Context (HANC) Layer.
  ```python
  HANCLayer(in_channels: int, out_channels: int, k: int = 3, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/hanc_layer.py`*

## Hierarchical_Mlp_Stem

### layers.hierarchical_mlp_stem
A hierarchical, non-overlapping convolutional stem for ViTs.

**Classes:**

- `HierarchicalMLPStem` - Keras Layer
  Hierarchical MLP stem for Vision Transformers with patch-independent processing.
  ```python
  HierarchicalMLPStem(embed_dim: int = 768, img_size: Tuple[int, int] = (224, 224), patch_size: Tuple[int, int] = (16, 16), ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/hierarchical_mlp_stem.py`*

## Inverted_Residual_Block

### layers.inverted_residual_block

**Classes:**
- `InvertedResidualBlock`

**Functions:** `get_config`

*📁 File: `src/dl_techniques/layers/inverted_residual_block.py`*

## Io_Preparation

### layers.io_preparation
Tensor normalization and clipping layers.

**Classes:**

- `ClipLayer` - Keras Layer
  Layer that clips tensor values to a specified range.
  ```python
  ClipLayer(clip_min: float, clip_max: float, **kwargs)
  ```

- `NormalizationLayer` - Keras Layer
  Layer that normalizes tensor values from source range to target range.
  ```python
  NormalizationLayer(source_min: float = 0.0, source_max: float = 255.0, target_min: float = -0.5, ...)
  ```

- `DenormalizationLayer` - Keras Layer
  Layer that denormalizes tensor values from source range to target range.
  ```python
  DenormalizationLayer(source_min: float = -0.5, source_max: float = 0.5, target_min: float = 0.0, ...)
  ```

- `TensorPreprocessingLayer` - Keras Layer
  Composite preprocessing layer combining normalization and clipping operations.
  ```python
  TensorPreprocessingLayer(source_min: float = 0.0, source_max: float = 255.0, target_min: float = -0.5, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `call`, `compute_output_shape` (and 8 more)

*📁 File: `src/dl_techniques/layers/io_preparation.py`*

## Kan_Linear

### layers.kan_linear
Kolmogorov-Arnold Network (KAN) linear layer.

**Classes:**

- `KANLinear` - Keras Layer
  Kolmogorov-Arnold Network (KAN) linear layer with learnable activation functions.
  ```python
  KANLinear(features: int, grid_size: int = 5, spline_order: int = 3, ...)
  ```

**Functions:** `build`, `call`, `update_grid_from_samples`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/kan_linear.py`*

## Kmeans

### layers.kmeans
A differentiable K-means clustering layer for deep networks.

**Classes:**

- `KMeansLayer` - Keras Layer
  A differentiable K-means layer with momentum and centroid repulsion.
  ```python
  KMeansLayer(n_clusters: int, temperature: float = 0.1, momentum: float = 0.9, ...)
  ```

**Functions:** `build`, `compute_output_shape`, `call`, `get_config`, `cluster_centers` (and 1 more)

*📁 File: `src/dl_techniques/layers/kmeans.py`*

## Laplacian_Filter

### layers.laplacian_filter
This module provides Keras layers for applying Laplacian filters to image data,

**Classes:**

- `LaplacianFilter` - Keras Layer
  Laplacian filter layer that detects edges by approximating the second derivative.
  ```python
  LaplacianFilter(kernel_size: Tuple[int, int] = (5, 5), strides: Union[Tuple[int, int], List[int]] = (1, 1), sigma: Optional[Union[float, Tuple[float, float]]] = 1.0, ...)
  ```

- `AdvancedLaplacianFilter` - Keras Layer
  Advanced Laplacian filter with multiple implementation options.
  ```python
  AdvancedLaplacianFilter(method: Literal['dog', 'log', 'kernel'] = 'dog', kernel_size: Tuple[int, int] = (5, 5), strides: Union[Tuple[int, int], List[int]] = (1, 1), ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 3 more)

*📁 File: `src/dl_techniques/layers/laplacian_filter.py`*

## Layer_Scale

### layers.layer_scale
This module provides a specialized Keras layer, `LearnableMultiplier`, for implementing

**Classes:**
- `MultiplierType`

- `LearnableMultiplier` - Keras Layer
  Layer implementing learnable element-wise multipliers for adaptive feature scaling.
  ```python
  LearnableMultiplier(multiplier_type: Union[MultiplierType, str] = MultiplierType.CHANNEL, initializer: Union[str, keras.initializers.Initializer] = 'ones', regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None, ...)
  ```

**Functions:** `from_string`, `to_string`, `build`, `call`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/layer_scale.py`*

## Logic

### layers.logic

*📁 File: `src/dl_techniques/layers/logic/__init__.py`*

### layers.logic.arithmetic_operators
A differentiable, learnable arithmetic operator.

**Classes:**

- `LearnableArithmeticOperator` - Keras Layer
  A learnable arithmetic operator that can perform various arithmetic operations.
  ```python
  LearnableArithmeticOperator(operation_types: Optional[List[str]] = None, use_temperature: bool = True, temperature_init: float = 1.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/logic/arithmetic_operators.py`*

### layers.logic.logic_operators
A differentiable operator that learns logical functions.

**Classes:**

- `LearnableLogicOperator` - Keras Layer
  A learnable logic operator that can perform various logical operations.
  ```python
  LearnableLogicOperator(operation_types: Optional[List[str]] = None, use_temperature: bool = True, temperature_init: float = 1.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/logic/logic_operators.py`*

### layers.logic.neural_circuit
A parallelized, learnable computational block for a neural circuit.

**Classes:**

- `CircuitDepthLayer` - Keras Layer
  A single depth layer of the neural circuit.
  ```python
  CircuitDepthLayer(num_logic_ops: int = 2, num_arithmetic_ops: int = 2, use_residual: bool = True, ...)
  ```

- `LearnableNeuralCircuit` - Keras Layer
  A learnable neural circuit with configurable depth and parallel operators.
  ```python
  LearnableNeuralCircuit(circuit_depth: int = 3, num_logic_ops_per_depth: int = 2, num_arithmetic_ops_per_depth: int = 2, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 3 more)

*📁 File: `src/dl_techniques/layers/logic/neural_circuit.py`*

## Memory

### layers.memory

*📁 File: `src/dl_techniques/layers/memory/__init__.py`*

### layers.memory.mann
Memory-Augmented Neural Network (MANN) based on

**Classes:**

- `MannLayer` - Keras Layer
  Memory-Augmented Neural Network (MANN) layer based on Neural Turing Machines.
  ```python
  MannLayer(memory_locations: int, memory_dim: int, controller_units: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/memory/mann.py`*

### layers.memory.som_2d_layer
Self-Organizing Map (SOM) 2D layer implementation.

**Classes:**

- `SOM2dLayer` - Keras Layer
  2D Self-Organizing Map (SOM) layer for competitive learning and topological data organization.
  ```python
  SOM2dLayer(map_size: Tuple[int, int], input_dim: int, initial_learning_rate: float = 0.1, ...)
  ```

**Functions:** `get_weights_as_grid`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/memory/som_2d_layer.py`*

### layers.memory.som_nd_layer
N-Dimensional Self-Organizing Map (SOM) layer implementation for Keras.

**Classes:**

- `SOMLayer` - Keras Layer
  N-Dimensional Self-Organizing Map (SOM) layer implementation for Keras.
  ```python
  SOMLayer(grid_shape: Tuple[int, ...], input_dim: int, initial_learning_rate: float = 0.1, ...)
  ```

**Functions:** `build`, `call`, `get_weights_map`, `compute_output_shape`, `get_config` (and 3 more)

*📁 File: `src/dl_techniques/layers/memory/som_nd_layer.py`*

### layers.memory.som_nd_soft_layer
Differentiable Soft Self-Organizing Map (Soft SOM) Layer.

**Classes:**

- `SoftSOMLayer` - Keras Layer
  Differentiable Soft Self-Organizing Map layer for end-to-end training.
  ```python
  SoftSOMLayer(grid_shape: Tuple[int, ...], input_dim: int, temperature: float = 1.0, ...)
  ```

**Functions:** `build`, `call`, `get_weights_map`, `get_soft_assignments`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/memory/som_nd_soft_layer.py`*

## Mobile_One_Block

### layers.mobile_one_block
MobileOne block using structural reparameterization.

**Classes:**

- `MobileOneBlock` - Keras Layer
  MobileOne building block with structural reparameterization.
  ```python
  MobileOneBlock(out_channels: int, kernel_size: int, stride: int = 1, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/mobile_one_block.py`*

## Modality_Projection

### layers.modality_projection
Projects visual features into a language model's embedding space.

**Classes:**

- `ModalityProjection` - Keras Layer
  Modality projection layer for nanoVLM.
  ```python
  ModalityProjection(input_dim: int, output_dim: int, scale_factor: int = 2, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `get_build_config` (and 1 more)

*📁 File: `src/dl_techniques/layers/modality_projection.py`*

## Moe

### layers.moe

*📁 File: `src/dl_techniques/layers/moe/__init__.py`*

### layers.moe.config
Configuration classes for Mixture of Experts (MoE) models.

**Classes:**
- `ExpertConfig`
- `GatingConfig`
- `MoEConfig`

**Functions:** `to_dict`, `from_dict`

*📁 File: `src/dl_techniques/layers/moe/config.py`*

### layers.moe.experts
Expert network implementations for Mixture of Experts (MoE) models.

**Classes:**

- `BaseExpert` - Keras Layer
  Abstract base class for MoE expert networks.
  ```python
  BaseExpert(name: Optional[str] = None, **kwargs)
  ```
- `FFNExpert`

**Functions:** `create_expert`, `call`, `compute_output_shape`, `get_build_config`, `build_from_config` (and 4 more)

*📁 File: `src/dl_techniques/layers/moe/experts.py`*

### layers.moe.gating
Gating network implementations for Mixture of Experts (MoE) models.

**Classes:**

- `BaseGating` - Keras Layer
  Abstract base class for MoE gating networks.
  ```python
  BaseGating(num_experts: int, name: Optional[str] = None, **kwargs)
  ```
- `LinearGating`
- `CosineGating`
- `SoftMoEGating`

**Functions:** `compute_auxiliary_loss`, `compute_z_loss`, `create_gating`, `call`, `get_config` (and 9 more)

*📁 File: `src/dl_techniques/layers/moe/gating.py`*

### layers.moe.integration
Integration utilities for MoE module with dl_techniques framework.

**Classes:**
- `MoETrainingConfig`
- `MoEOptimizerBuilder`

**Functions:** `build_moe_optimizer`

*📁 File: `src/dl_techniques/layers/moe/integration.py`*

### layers.moe.layer
Main Mixture of Experts (MoE) layer implementation.

**Classes:**

- `MixtureOfExperts` - Keras Layer
  Mixture of Experts (MoE) layer for sparse neural networks using FFN experts.
  ```python
  MixtureOfExperts(config: MoEConfig, **kwargs)
  ```

**Functions:** `create_ffn_moe`, `build`, `call`, `compute_output_shape`, `get_expert_utilization` (and 2 more)

*📁 File: `src/dl_techniques/layers/moe/layer.py`*

## Mothnet_Blocks

### layers.mothnet_blocks
MothNet: Bio-Mimetic Feature Generation for Few-Shot Learning.

**Classes:**

- `AntennalLobeLayer` - Keras Layer
  Antennal Lobe layer implementing competitive inhibition for contrast enhancement.
  ```python
  AntennalLobeLayer(units: int, inhibition_strength: float = 0.5, activation: str = 'relu', ...)
  ```

- `MushroomBodyLayer` - Keras Layer
  Mushroom Body layer implementing high-dimensional sparse random projection.
  ```python
  MushroomBodyLayer(units: int, sparsity: float = 0.1, connection_sparsity: float = 0.1, ...)
  ```

- `HebbianReadoutLayer` - Keras Layer
  Hebbian readout layer implementing local correlation-based learning.
  ```python
  HebbianReadoutLayer(units: int, learning_rate: float = 0.01, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 8 more)

*📁 File: `src/dl_techniques/layers/mothnet_blocks.py`*

## Mps_Layer

### layers.mps_layer
A layer based on the Matrix Product State tensor network.

**Classes:**

- `MPSLayer` - Keras Layer
  Matrix Product State inspired layer for tensor decomposition.
  ```python
  MPSLayer(output_dim: int, bond_dim: int = 16, use_bias: bool = True, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/mps_layer.py`*

## Multi_Level_Feature_Compilation

### layers.multi_level_feature_compilation
Multi Level Feature Compilation (MLFC) Layer for Cross-Scale Feature Fusion.

**Classes:**

- `MLFCLayer` - Keras Layer
  Multi Level Feature Compilation (MLFC) Layer.
  ```python
  MLFCLayer(channels_list: List[int], num_iterations: int = 1, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/multi_level_feature_compilation.py`*

## Neuro_Grid

### layers.neuro_grid
Implements a differentiable, n-dimensional memory with probabilistic addressing.

**Classes:**

- `NeuroGrid` - Keras Layer
  NeuroGrid: Differentiable N-Dimensional Memory Lattice with Probabilistic Addressing for Transformers.
  ```python
  NeuroGrid(grid_shape: Union[List[int], Tuple[int, ...]], latent_dim: int, use_bias: bool = False, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `get_grid_weights` (and 8 more)

*📁 File: `src/dl_techniques/layers/neuro_grid.py`*

## Nlp_Heads

### layers.nlp_heads

*📁 File: `src/dl_techniques/layers/nlp_heads/__init__.py`*

### layers.nlp_heads.factory
NLP Task Head Factory

**Classes:**

- `BaseNLPHead` - Keras Layer
  Base class for all NLP task heads.
  ```python
  BaseNLPHead(task_config: NLPTaskConfig, input_dim: int, normalization_type: NormalizationType = 'layer_norm', ...)
  ```
- `TextClassificationHead`
- `TokenClassificationHead`
- `QuestionAnsweringHead`
- `TextSimilarityHead`
- `TextGenerationHead`
- `MultipleChoiceHead`

- `MultiTaskNLPHead` - Keras Layer
  Multi-task head that combines multiple task-specific NLP heads.
  ```python
  MultiTaskNLPHead(task_configs: Dict[str, NLPTaskConfig], shared_input_dim: int, use_task_specific_projections: bool = False, ...)
  ```
- `NLPHeadConfiguration`

**Functions:** `get_head_class`, `create_nlp_head`, `create_multi_task_nlp_head`, `build`, `compute_output_shape` (and 27 more)

*📁 File: `src/dl_techniques/layers/nlp_heads/factory.py`*

### layers.nlp_heads.task_types
NLP Task Types and Configuration

**Classes:**
- `NLPTaskType`
- `NLPTaskConfig`
- `NLPTaskConfiguration`
- `CommonNLPTaskConfigurations`

**Functions:** `all_tasks`, `get_task_categories`, `get_compatible_tasks`, `get_output_types`, `get_input_requirements` (and 8 more)

*📁 File: `src/dl_techniques/layers/nlp_heads/task_types.py`*

## Norms

### layers.norms

*📁 File: `src/dl_techniques/layers/norms/__init__.py`*

### layers.norms.adaptive_band_rms
Adaptive BandRMS Layer: RMS Normalization with Log-Transformed RMS-Statistics-Based Scaling.

**Classes:**

- `AdaptiveBandRMS` - Keras Layer
  Adaptive Root Mean Square Normalization with log-transformed RMS scaling.
  ```python
  AdaptiveBandRMS(max_band_width: float = 0.1, axis: Union[int, Tuple[int, ...]] = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/adaptive_band_rms.py`*

### layers.norms.band_logit_norm
BandLogitNorm Layer Implementation

**Classes:**

- `BandLogitNorm` - Keras Layer
  Band-constrained logit normalization layer.
  ```python
  BandLogitNorm(max_band_width: float = 0.01, axis: int = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/norms/band_logit_norm.py`*

### layers.norms.band_rms
BandRMS Layer: RMS Normalization within a Learnable Spherical Shell.

**Classes:**

- `BandRMS` - Keras Layer
  Root Mean Square Normalization layer with bounded RMS constraints.
  ```python
  BandRMS(max_band_width: float = 0.1, axis: Union[int, Tuple[int, ...]] = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/band_rms.py`*

### layers.norms.dynamic_tanh
Dynamic Tanh (DyT) Layer Implementation for Keras 3.x

**Classes:**

- `DynamicTanh` - Keras Layer
  Dynamic Tanh (DyT) layer as described in "Transformers without Normalization".
  ```python
  DynamicTanh(axis: Union[int, List[int]] = -1, alpha_init_value: float = 0.5, kernel_initializer: Union[str, initializers.Initializer] = 'ones', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/dynamic_tanh.py`*

### layers.norms.factory
Normalization Layer Factory Utility for dl_techniques Framework.

**Functions:** `create_normalization_layer`, `get_normalization_info`, `validate_normalization_config`, `create_normalization_from_config`

*📁 File: `src/dl_techniques/layers/norms/factory.py`*

### layers.norms.global_response_norm
Global Response Normalization (GRN) Layer Implementation

**Classes:**

- `GlobalResponseNormalization` - Keras Layer
  Global Response Normalization (GRN) layer supporting 2D, 3D, and 4D inputs.
  ```python
  GlobalResponseNormalization(eps: float = 1e-06, gamma_initializer: Union[str, keras.initializers.Initializer] = 'ones', beta_initializer: Union[str, keras.initializers.Initializer] = 'zeros', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/global_response_norm.py`*

### layers.norms.logit_norm
LogitNorm Layer for Classification Tasks

**Classes:**

- `LogitNorm` - Keras Layer
  LogitNorm layer for classification tasks.
  ```python
  LogitNorm(temperature: float = 0.04, axis: int = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/logit_norm.py`*

### layers.norms.max_logit_norm
MaxLogit Normalization Implementations for Out-of-Distribution Detection.

**Classes:**

- `MaxLogitNorm` - Keras Layer
  Basic MaxLogit normalization layer for out-of-distribution detection.
  ```python
  MaxLogitNorm(axis: int = -1, epsilon: float = 1e-07, **kwargs)
  ```

- `DecoupledMaxLogit` - Keras Layer
  Decoupled MaxLogit (DML) normalization layer.
  ```python
  DecoupledMaxLogit(constant: float = 1.0, axis: int = -1, epsilon: float = 1e-07, ...)
  ```

- `DMLPlus` - Keras Layer
  DML+ implementation for separate focal and center models.
  ```python
  DMLPlus(model_type: Literal['focal', 'center'], axis: int = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `call`, `compute_output_shape` (and 4 more)

*📁 File: `src/dl_techniques/layers/norms/max_logit_norm.py`*

### layers.norms.rms_norm
Root Mean Square Normalization Layer for Deep Neural Networks

**Classes:**

- `RMSNorm` - Keras Layer
  Root Mean Square Normalization layer for stabilized training in deep networks.
  ```python
  RMSNorm(axis: Union[int, Tuple[int, ...]] = -1, epsilon: float = 1e-06, use_scale: bool = True, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/rms_norm.py`*

### layers.norms.zero_centered_band_rms_norm
Zero-Centered Band RMS Normalization Layer for Enhanced Training Stability

**Classes:**

- `ZeroCenteredBandRMSNorm` - Keras Layer
  Zero-Centered Root Mean Square Normalization with learnable band constraints.
  ```python
  ZeroCenteredBandRMSNorm(max_band_width: float = 0.1, axis: Union[int, Tuple[int, ...]] = -1, epsilon: float = 1e-07, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py`*

### layers.norms.zero_centered_rms_norm
Zero-Centered Root Mean Square Normalization Layer for Deep Neural Networks

**Classes:**

- `ZeroCenteredRMSNorm` - Keras Layer
  Zero-Centered Root Mean Square Normalization layer for enhanced training stability.
  ```python
  ZeroCenteredRMSNorm(axis: Union[int, Tuple[int, ...]] = -1, epsilon: float = 1e-06, use_scale: bool = True, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/norms/zero_centered_rms_norm.py`*

## Ntm

### layers.ntm
Neural Turing Machine (NTM) Package.

*📁 File: `src/dl_techniques/layers/ntm/__init__.py`*

### layers.ntm.base_layers
Differentiable Addressing and Memory Layers for Neural Turing Machines.

**Classes:**

- `DifferentiableAddressingHead` - Keras Layer
  Differentiable addressing head implementing NTM-style memory addressing.
  ```python
  DifferentiableAddressingHead(memory_size: int, content_dim: int, controller_dim: int | None = None, ...)
  ```

- `DifferentiableSelectCopy` - Keras Layer
  Differentiable layer for selecting and copying values between memory positions.
  ```python
  DifferentiableSelectCopy(memory_size: int, content_dim: int, controller_dim: int, ...)
  ```

- `SimpleSelectCopy` - Keras Layer
  Simplified differentiable select-copy layer for learning input-output mappings.
  ```python
  SimpleSelectCopy(input_size: int, output_size: int, content_dim: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 7 more)

*📁 File: `src/dl_techniques/layers/ntm/base_layers.py`*

### layers.ntm.baseline_ntm
Baseline Neural Turing Machine Implementation.

**Classes:**
- `NTMMemory`
- `NTMReadHead`
- `NTMWriteHead`
- `NTMController`

- `NTMCell` - Keras Layer
  Core NTM Cell for processing a single timestep.
  ```python
  NTMCell(config: NTMConfig | dict[str, Any], kernel_initializer: str | keras.initializers.Initializer = 'glorot_uniform', bias_initializer: str | keras.initializers.Initializer = 'zeros', ...)
  ```
- `NeuralTuringMachine`

**Functions:** `create_ntm`, `initialize_state`, `read`, `write`, `get_config` (and 33 more)

*📁 File: `src/dl_techniques/layers/ntm/baseline_ntm.py`*

### layers.ntm.ntm_interface
Neural Turing Machine (NTM) Interface Module.

**Classes:**
- `AddressingMode`
- `MemoryAccessType`
- `MemoryState`
- `HeadState`
- `NTMOutput`
- `NTMConfig`

- `BaseMemory` - Keras Layer
  Abstract base class for memory modules.
  ```python
  BaseMemory(memory_size: int, memory_dim: int, epsilon: float = 1e-06, ...)
  ```

- `BaseHead` - Keras Layer
  Abstract base class for read and write heads.
  ```python
  BaseHead(memory_size: int, memory_dim: int, addressing_mode: AddressingMode = AddressingMode.HYBRID, ...)
  ```

- `BaseController` - Keras Layer
  Abstract base class for controller networks.
  ```python
  BaseController(controller_dim: int, controller_type: Literal['lstm', 'gru', 'feedforward'] = 'lstm', **kwargs)
  ```

- `BaseNTM` - Keras Layer
  Abstract base class for Neural Turing Machine architectures.
  ```python
  BaseNTM(config: NTMConfig, output_dim: int | None = None, **kwargs)
  ```

**Functions:** `cosine_similarity`, `circular_convolution`, `sharpen_weights`, `clone`, `to_dict` (and 19 more)

*📁 File: `src/dl_techniques/layers/ntm/ntm_interface.py`*

## One_Hot_Encoding

### layers.one_hot_encoding
This module provides a `OneHotEncoding` layer, a custom Keras layer that performs

**Classes:**

- `OneHotEncoding` - Keras Layer
  One-hot encoding layer for categorical features with enhanced efficiency.
  ```python
  OneHotEncoding(cardinalities: List[int], **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/one_hot_encoding.py`*

## Orthoblock

### layers.orthoblock
Learn decorrelated and gated features through a structured pipeline.

**Classes:**

- `OrthoBlock` - Keras Layer
  Structured feature learning block with orthogonal regularization and constrained scaling.
  ```python
  OrthoBlock(units: int, activation: Optional[Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]]] = None, use_bias: bool = True, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/orthoblock.py`*

## Patch_Merging

### layers.patch_merging
Downsample feature maps by merging patches to create a hierarchical representation.

**Classes:**

- `PatchMerging` - Keras Layer
  Patch merging layer for hierarchical downsampling in Swin Transformer architectures.
  ```python
  PatchMerging(dim: int, use_bias: bool = False, kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/patch_merging.py`*

## Physics

### layers.physics

*📁 File: `src/dl_techniques/layers/physics/__init__.py`*

### layers.physics.approximate_lagrange_layer

**Classes:**

- `ApproximatedLNNLayer` - Keras Layer
  Gradient-tape-free approximation of Lagrangian Neural Network dynamics.
  ```python
  ApproximatedLNNLayer(hidden_dims: List[int], activation: str = 'softplus', **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/physics/approximate_lagrange_layer.py`*

### layers.physics.lagrange_layer

**Classes:**

- `LagrangianNeuralNetworkLayer` - Keras Layer
  Physics-informed layer modeling system dynamics through learned Lagrangian mechanics.
  ```python
  LagrangianNeuralNetworkLayer(hidden_dims: List[int], activation: str = 'softplus', **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/physics/lagrange_layer.py`*

## Pixel_Shuffle

### layers.pixel_shuffle
Pixel Shuffle Layer Implementation for Vision Transformers.

**Classes:**

- `PixelShuffle` - Keras Layer
  Pixel shuffle operation for reducing spatial tokens in vision_heads transformers.
  ```python
  PixelShuffle(scale_factor: int = 2, validate_spatial_dims: bool = True, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/pixel_shuffle.py`*

## Radial_Basis_Function

### layers.radial_basis_function
A Radial Basis Function (RBF) layer with center repulsion.

**Classes:**

- `RBFLayer` - Keras Layer
  Radial Basis Function layer with stable center repulsion mechanism.
  ```python
  RBFLayer(units: int, gamma_init: float = 1.0, repulsion_strength: float = 0.1, ...)
  ```

**Functions:** `build`, `gamma`, `call`, `compute_output_shape`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/layers/radial_basis_function.py`*

## Random_Fourier_Features

### layers.random_fourier_features
A Random Fourier Features (RFF) mapping to approximate kernel methods.

**Classes:**

- `RFFKernelLayer` - Keras Layer
  Random Fourier Features layer for efficient kernel approximation.
  ```python
  RFFKernelLayer(input_dim: int, output_dim: Optional[int] = None, n_features: int = 1000, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/random_fourier_features.py`*

## Reasoning

### layers.reasoning

*📁 File: `src/dl_techniques/layers/reasoning/__init__.py`*

### layers.reasoning.hrm_reasoning_core
A stateful, recurrent engine for the Hierarchical Reasoning Model (HRM).

**Classes:**

- `HierarchicalReasoningCore` - Keras Layer
  Stateful hierarchical reasoning core for complex multi-step reasoning tasks.
  ```python
  HierarchicalReasoningCore(vocab_size: int, seq_len: int, embed_dim: int, ...)
  ```

**Functions:** `build`, `empty_carry`, `reset_carry`, `call`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/reasoning/hrm_reasoning_core.py`*

### layers.reasoning.hrm_reasoning_module
This module defines the HierarchicalReasoningModule, a composite Keras layer that

**Classes:**

- `HierarchicalReasoningModule` - Keras Layer
  Configurable multi-layer reasoning module with input injection.
  ```python
  HierarchicalReasoningModule(num_layers: int, embed_dim: int, num_heads: int = 8, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/reasoning/hrm_reasoning_module.py`*

### layers.reasoning.hrm_sparse_puzzle_embedding
This module defines the SparsePuzzleEmbedding layer.

**Classes:**

- `SparsePuzzleEmbedding` - Keras Layer
  Sparse embedding layer optimized for large-scale puzzle identifier lookups with training efficiency.
  ```python
  SparsePuzzleEmbedding(num_embeddings: int, embedding_dim: int, batch_size: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/reasoning/hrm_sparse_puzzle_embedding.py`*

## Repmixer_Block

### layers.repmixer_block
A RepMixer block, an efficient feature-mixing architecture.

**Classes:**

- `RepMixerBlock` - Keras Layer
  RepMixer block for efficient feature mixing in vision_heads models.
  ```python
  RepMixerBlock(dim: int, kernel_size: int = 3, expansion_ratio: float = 4.0, ...)
  ```

- `ConvolutionalStem` - Keras Layer
  Convolutional stem for FastVLM using MobileOne blocks.
  ```python
  ConvolutionalStem(out_channels: int, use_se: bool = False, activation: Union[str, callable] = 'gelu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 5 more)

*📁 File: `src/dl_techniques/layers/repmixer_block.py`*

## Res_Path

### layers.res_path
A residual path to bridge the semantic gap in U-Net skip connections.

**Classes:**

- `ResPath` - Keras Layer
  Residual Path layer for improving skip connections in U-Net architectures.
  ```python
  ResPath(channels: int, num_blocks: int, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/res_path.py`*

## Restricted_Boltzmann_Machine

### layers.restricted_boltzmann_machine
Restricted Boltzmann Machine (RBM).

**Classes:**

- `RestrictedBoltzmannMachine` - Keras Layer
  Restricted Boltzmann Machine (RBM) layer for unsupervised feature learning.
  ```python
  RestrictedBoltzmannMachine(n_hidden: int, learning_rate: float = 0.01, n_gibbs_steps: int = 1, ...)
  ```

**Functions:** `build`, `call`, `sample_hidden_given_visible`, `sample_visible_given_hidden`, `gibbs_sampling_step` (and 4 more)

*📁 File: `src/dl_techniques/layers/restricted_boltzmann_machine.py`*

## Rigid_Simplex_Layer

### layers.rigid_simplex_layer
Rigid Simplex Layer with learnable rotation and bounded scaling.

**Classes:**

- `RigidSimplexLayer` - Keras Layer
  Projects inputs onto a fixed Simplex structure with learnable rotation and scaling.
  ```python
  RigidSimplexLayer(units: int, scale_min: float = 0.5, scale_max: float = 2.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/rigid_simplex_layer.py`*

## Router

### layers.router
A dynamic routing mechanism for conditional computation.

**Classes:**

- `RouterLayer` - Keras Layer
  Wraps a TransformerLayer with a Dr.LLM-style dynamic routing mechanism.
  ```python
  RouterLayer(transformer_layer: TransformerLayer, router_bottleneck_dim: int = 128, num_windows: int = 8, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/router.py`*

## Sampling

### layers.sampling
Sample from a latent Normal distribution using the reparameterization trick.

**Classes:**

- `Sampling` - Keras Layer
  Uses reparameterization trick to sample from a Normal distribution.
  ```python
  Sampling(seed: Optional[int] = None, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/sampling.py`*

## Selective_Gradient_Mask

### layers.selective_gradient_mask
Selectively mask gradients during backpropagation without altering the forward pass.

**Classes:**

- `SelectiveGradientMask` - Keras Layer
  Layer that selectively stops gradients based on a binary mask.
  ```python
  SelectiveGradientMask(name: Optional[str] = None, dtype: Optional[str] = None, **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/layers/selective_gradient_mask.py`*

## Sequence_Pooling

### layers.sequence_pooling
A unified and configurable pooling layer for sequence data.

**Classes:**

- `AttentionPooling` - Keras Layer
  Attention-based pooling that learns to weight sequence elements.
  ```python
  AttentionPooling(hidden_dim: int = 256, num_heads: int = 1, dropout_rate: float = 0.0, ...)
  ```

- `WeightedPooling` - Keras Layer
  Learnable weighted pooling with position-specific weights.
  ```python
  WeightedPooling(max_seq_len: int = 512, dropout_rate: float = 0.0, temperature: float = 1.0, ...)
  ```

- `SequencePooling` - Keras Layer
  Highly configurable pooling layer for sequence data.
  ```python
  SequencePooling(strategy: Union[PoolingStrategy, List[PoolingStrategy]] = 'mean', exclude_positions: Optional[List[int]] = None, aggregation_method: AggregationMethod = 'concat', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 7 more)

*📁 File: `src/dl_techniques/layers/sequence_pooling.py`*

## Shearlet_Transform

### layers.shearlet_transform
Decompose an image into multi-scale, multi-directional components.

**Classes:**

- `ShearletTransform` - Keras Layer
  Multi-scale, multi-directional shearlet transform layer for enhanced time-frequency analysis.
  ```python
  ShearletTransform(scales: int = 4, directions: int = 8, alpha: float = 0.5, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `smooth_transition`

*📁 File: `src/dl_techniques/layers/shearlet_transform.py`*

## Sparse_Autoencoder

### layers.sparse_autoencoder
Sparse Autoencoder (SAE) Module

**Classes:**

- `SparseAutoencoder` - Keras Layer
  Sparse Autoencoder layer with multiple sparsity enforcement variants.
  ```python
  SparseAutoencoder(d_input: int, d_latent: int, variant: SAEVariant = 'topk', ...)
  ```

**Functions:** `create_sparse_autoencoder`, `build`, `encode`, `decode`, `call` (and 3 more)

*📁 File: `src/dl_techniques/layers/sparse_autoencoder.py`*

## Spatial_Layer

### layers.spatial_layer
Inject explicit spatial coordinate information into feature maps.

**Classes:**

- `SpatialLayer` - Keras Layer
  Spatial coordinate grid generator for injecting positional information into models.
  ```python
  SpatialLayer(resolution: Tuple[int, int] = (4, 4), resize_method: Literal['nearest', 'bilinear'] = 'nearest', **kwargs)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/spatial_layer.py`*

## Squeeze_Excitation

### layers.squeeze_excitation
Implement a Squeeze-and-Excitation block for channel-wise feature recalibration.

**Classes:**

- `SqueezeExcitation` - Keras Layer
  Squeeze-and-Excitation block for channel-wise feature recalibration.
  ```python
  SqueezeExcitation(reduction_ratio: float = 0.25, activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu', use_bias: bool = False, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/squeeze_excitation.py`*

## Standard_Blocks

### layers.standard_blocks
Configurable Building Blocks for Deep Learning Architectures.

**Classes:**

- `ConvBlock` - Keras Layer
  Configurable convolutional block with normalization, activation, and optional pooling.
  ```python
  ConvBlock(filters: int, kernel_size: Union[int, Tuple[int, int]] = 3, strides: Union[int, Tuple[int, int]] = 1, ...)
  ```

- `DenseBlock` - Keras Layer
  Configurable dense block with normalization, activation, and optional dropout.
  ```python
  DenseBlock(units: int, normalization_type: Optional[str] = 'layer_norm', activation_type: str = 'relu', ...)
  ```

- `ResidualDenseBlock` - Keras Layer
  Dense block with residual connection and configurable normalization/activation.
  ```python
  ResidualDenseBlock(units: Optional[int] = None, normalization_type: Optional[str] = 'layer_norm', activation_type: str = 'relu', ...)
  ```

- `BasicBlock` - Keras Layer
  Basic ResNet block with two 3x3 convolutions.
  ```python
  BasicBlock(filters: int, stride: int = 1, use_projection: bool = False, ...)
  ```

- `BottleneckBlock` - Keras Layer
  Bottleneck ResNet block with 1x1 → 3x3 → 1x1 convolutions.
  ```python
  BottleneckBlock(filters: int, stride: int = 1, use_projection: bool = False, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 15 more)

*📁 File: `src/dl_techniques/layers/standard_blocks.py`*

## Statistics

### layers.statistics

*📁 File: `src/dl_techniques/layers/statistics/__init__.py`*

### layers.statistics.deep_kernel_pca
Deep Kernel Principal Component Analysis (DKPCA) algorithm.

**Classes:**

- `DeepKernelPCA` - Keras Layer
  Deep Kernel Principal Component Analysis layer for multi-level feature extraction.
  ```python
  DeepKernelPCA(num_levels: int = 3, components_per_level: Optional[List[int]] = None, kernel_type: Union[str, List[str]] = 'rbf', ...)
  ```

**Functions:** `build`, `compute_kernel_matrix`, `extract_components`, `call`, `compute_output_shape` (and 2 more)

*📁 File: `src/dl_techniques/layers/statistics/deep_kernel_pca.py`*

### layers.statistics.invertible_kernel_pca
An invertible Kernel PCA using Random Fourier Features.

**Classes:**

- `InvertibleKernelPCA` - Keras Layer
  Invertible Kernel PCA layer using Random Fourier Features approximation.
  ```python
  InvertibleKernelPCA(n_components: Optional[int] = None, n_random_features: int = 256, kernel_type: Literal['rbf', 'laplacian', 'cauchy'] = 'rbf', ...)
  ```

- `InvertibleKernelPCADenoiser` - Keras Layer
  Denoising layer based on Invertible Kernel PCA.
  ```python
  InvertibleKernelPCADenoiser(n_components: Union[int, float] = 0.95, n_random_features: int = 512, kernel_type: str = 'rbf', ...)
  ```

**Functions:** `build`, `compute_random_features`, `update_pca_components`, `call`, `transform` (and 9 more)

*📁 File: `src/dl_techniques/layers/statistics/invertible_kernel_pca.py`*

### layers.statistics.mdn_layer
Mixture Density Network (MDN) Layer with Intermediate Processing

**Classes:**

- `MDNLayer` - Keras Layer
  Mixture Density Network Layer with separated processing paths.
  ```python
  MDNLayer(output_dimension: int, num_mixtures: int, use_bias: bool = True, ...)
  ```

**Functions:** `get_point_estimate`, `get_uncertainty`, `get_prediction_intervals`, `check_component_diversity`, `build` (and 6 more)

*📁 File: `src/dl_techniques/layers/statistics/mdn_layer.py`*

### layers.statistics.moving_std
This module provides a `MovingStd` layer that applies a 2D moving standard deviation

**Classes:**

- `MovingStd` - Keras Layer
  Applies a 2D moving standard deviation filter to input images for texture analysis.
  ```python
  MovingStd(pool_size: Tuple[int, int] = (3, 3), strides: Union[Tuple[int, int], List[int]] = (1, 1), padding: str = 'same', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/statistics/moving_std.py`*

### layers.statistics.normalizing_flow
Normalizing Flow Layer for Conditional Density Estimation using Keras 3.

**Classes:**

- `AffineCouplingLayer` - Keras Layer
  Affine coupling transformation layer for normalizing flows with conditional context.
  ```python
  AffineCouplingLayer(input_dim: int, context_dim: int, hidden_units: int = 64, ...)
  ```

- `NormalizingFlowLayer` - Keras Layer
  Conditional normalizing flow layer using stacked affine coupling transformations.
  ```python
  NormalizingFlowLayer(output_dimension: int, num_flow_steps: int, context_dim: int, ...)
  ```

**Functions:** `build`, `forward`, `inverse`, `compute_output_shape`, `get_config` (and 6 more)

*📁 File: `src/dl_techniques/layers/statistics/normalizing_flow.py`*

### layers.statistics.residual_acf
Residual Autocorrelation Function (ACF) analysis and regularization layer.

**Classes:**

- `ResidualACFLayer` - Keras Layer
  Residual Autocorrelation Function analysis and regularization layer for time series models.
  ```python
  ResidualACFLayer(max_lag: int = 40, regularization_weight: Optional[float] = None, target_lags: Optional[List[int]] = None, ...)
  ```
- `ACFMonitorCallback`

**Functions:** `build`, `compute_acf`, `call`, `get_acf_summary`, `compute_output_shape` (and 2 more)

*📁 File: `src/dl_techniques/layers/statistics/residual_acf.py`*

### layers.statistics.scaler
Unified Scaler Layer - A Comprehensive Normalization Solution.

**Classes:**

- `UnifiedScaler` - Keras Layer
  Unified normalization layer combining RevIN and StandardScaler capabilities.
  ```python
  UnifiedScaler(num_features: Optional[int] = None, axis: Union[int, Tuple[int, ...]] = -1, eps: float = 1e-05, ...)
  ```

**Functions:** `build`, `call`, `inverse_transform`, `denormalize`, `reset_stats` (and 3 more)

*📁 File: `src/dl_techniques/layers/statistics/scaler.py`*

## Stochastic_Depth

### layers.stochastic_depth
Stochastic Depth is a regularization method primarily used in very deep neural networks,

**Classes:**

- `StochasticDepth` - Keras Layer
  Implements Stochastic Depth for deep networks.
  ```python
  StochasticDepth(drop_path_rate: float = 0.5, **kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/stochastic_depth.py`*

## Stochastic_Gradient

### layers.stochastic_gradient
This module implements the Stochastic Gradient regularization technique.

**Classes:**

- `StochasticGradient` - Keras Layer
  Implements Stochastic Gradient dropping for deep networks.
  ```python
  StochasticGradient(drop_path_rate: float = 0.5, **kwargs)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/stochastic_gradient.py`*

## Strong_Augmentation

### layers.strong_augmentation
Apply strong data augmentations for consistency-based regularization.

**Classes:**

- `StrongAugmentation` - Keras Layer
  Strong augmentation layer for unlabeled data.
  ```python
  StrongAugmentation(cutmix_prob: float = 0.5, cutmix_ratio_range: Tuple[float, float] = (0.1, 0.5), color_jitter_strength: float = 0.2, ...)
  ```

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/layers/strong_augmentation.py`*

## Tabm_Blocks

### layers.tabm_blocks
Deep Ensembling is a powerful technique for improving model robustness, accuracy, and

**Classes:**

- `ScaleEnsemble` - Keras Layer
  Enhanced ensemble adapter with learnable scaling weights.
  ```python
  ScaleEnsemble(k: int, input_dim: int, init_distribution: Literal['normal', 'random-signs'] = 'normal', ...)
  ```

- `LinearEfficientEnsemble` - Keras Layer
  Efficient ensemble linear layer with separate input/output scaling.
  ```python
  LinearEfficientEnsemble(units: int, k: int, use_bias: bool = True, ...)
  ```

- `NLinear` - Keras Layer
  N parallel linear layers for ensemble output with enhanced efficiency.
  ```python
  NLinear(n: int, input_dim: int, output_dim: int, ...)
  ```

- `MLPBlock` - Keras Layer
  MLP block with efficient ensemble support and enhanced configurability.
  ```python
  MLPBlock(units: int, k: Optional[int] = None, activation: str = 'relu', ...)
  ```

- `TabMBackbone` - Keras Layer
  TabM backbone MLP with ensemble support and proper layer management.
  ```python
  TabMBackbone(hidden_dims: List[int], k: Optional[int] = None, activation: str = 'relu', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 15 more)

*📁 File: `src/dl_techniques/layers/tabm_blocks.py`*

## Time_Series

### layers.time_series
Time Series Layers Module.

*📁 File: `src/dl_techniques/layers/time_series/__init__.py`*

### layers.time_series.adaptive_lag_attention
A context-aware, gated attention mechanism for autoregression.

**Classes:**

- `AdaptiveLagAttentionLayer` - Keras Layer
  Advanced attention layer for dynamically weighting temporal lags with gating control.
  ```python
  AdaptiveLagAttentionLayer(num_lags: int, kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform', bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/adaptive_lag_attention.py`*

### layers.time_series.deepar_blocks
DeepAR Custom Layers.

**Classes:**

- `ScaleLayer` - Keras Layer
  Applies item-dependent scaling to inputs and inverse scaling to outputs.
  ```python
  ScaleLayer(scale_per_sample: bool = True, epsilon: float = 1.0, **kwargs)
  ```

- `GaussianLikelihoodHead` - Keras Layer
  Computes Gaussian likelihood parameters (mean, std) from hidden states.
  ```python
  GaussianLikelihoodHead(units: int = 1, **kwargs)
  ```

- `NegativeBinomialLikelihoodHead` - Keras Layer
  Computes Negative Binomial likelihood parameters (mu, alpha) from hidden states.
  ```python
  NegativeBinomialLikelihoodHead(units: int = 1, **kwargs)
  ```

- `DeepARCell` - Keras Layer
  Autoregressive recurrent cell for DeepAR.
  ```python
  DeepARCell(units: int, dropout: float = 0.0, recurrent_dropout: float = 0.0, ...)
  ```

**Functions:** `call`, `get_config`, `build`, `call`, `compute_output_shape` (and 9 more)

*📁 File: `src/dl_techniques/layers/time_series/deepar_blocks.py`*

### layers.time_series.ema_layer
Adaptive EMA Slope Filter Layer.

**Classes:**

- `ExponentialMovingAverage` - Keras Layer
  Computes Exponential Moving Average over time series data.
  ```python
  ExponentialMovingAverage(period: int = 25, adjust: bool = True, **kwargs)
  ```

- `EMASlopeFilter` - Keras Layer
  Computes EMA slope and generates trading signals based on slope thresholds.
  ```python
  EMASlopeFilter(ema_period: int = 25, lookback_period: int = 25, upper_threshold: float = 15.0, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `build`, `call` (and 2 more)

*📁 File: `src/dl_techniques/layers/time_series/ema_layer.py`*

### layers.time_series.forecasting_layers
Forecasting Layers based on Valeriy Manokhin's Scientific Framework.

**Classes:**

- `NaiveResidual` - Keras Layer
  Structural implementation of the Naive Benchmark Principle.
  ```python
  NaiveResidual(forecast_length: int, name: Optional[str] = None, **kwargs)
  ```

- `ForecastabilityGate` - Keras Layer
  Learnable gate for weighing deep predictions versus naive forecasts.
  ```python
  ForecastabilityGate(hidden_units: int = 16, activation: str = 'relu', kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform', ...)
  ```

- `ConformalQuantileHead` - Keras Layer
  Output layer designed for Conformalized Quantile Regression (CQR).
  ```python
  ConformalQuantileHead(forecast_length: int, output_dim: int, kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform', ...)
  ```

**Functions:** `create_manokhin_compliant_model`, `call`, `get_config`, `build`, `call` (and 6 more)

*📁 File: `src/dl_techniques/layers/time_series/forecasting_layers.py`*

### layers.time_series.mixed_sequential_block
A hybrid sequential block combining recurrent and attention mechanisms.

**Classes:**

- `MixedSequentialBlock` - Keras Layer
  Mixed sequential block combining LSTM and self-attention mechanisms for time series processing.
  ```python
  MixedSequentialBlock(embed_dim: int, num_heads: int = 8, lstm_units: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/mixed_sequential_block.py`*

### layers.time_series.nbeats_blocks
The foundational block of the N-BEATS architecture.

**Classes:**

- `NBeatsBlock` - Keras Layer
  Enhanced N-BEATS block layer with performance optimizations and modern Keras 3 compliance.
  ```python
  NBeatsBlock(units: int, thetas_dim: int, backcast_length: int, ...)
  ```
- `GenericBlock`
- `TrendBlock`
- `SeasonalityBlock`

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 5 more)

*📁 File: `src/dl_techniques/layers/time_series/nbeats_blocks.py`*

### layers.time_series.nbeatsx_blocks

**Classes:**
- `ExogenousBlock`

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/nbeatsx_blocks.py`*

### layers.time_series.prism_blocks
PRISM: Partitioned Representations for Iterative Sequence Modeling.

**Classes:**

- `FrequencyBandStatistics` - Keras Layer
  Computes summary statistics for frequency bands.
  ```python
  FrequencyBandStatistics(epsilon: float = 1e-06, **kwargs)
  ```

- `FrequencyBandRouter` - Keras Layer
  Learnable router for computing frequency band importance weights.
  ```python
  FrequencyBandRouter(hidden_dim: int = 64, temperature: float = 1.0, dropout_rate: float = 0.1, ...)
  ```

- `PRISMNode` - Keras Layer
  Single PRISM node combining wavelet decomposition and adaptive weighting.
  ```python
  PRISMNode(num_wavelet_levels: int = 3, router_hidden_dim: int = 64, router_temperature: float = 1.0, ...)
  ```

- `PRISMTimeTree` - Keras Layer
  Hierarchical time decomposition with PRISM nodes at each level.
  ```python
  PRISMTimeTree(tree_depth: int = 2, overlap_ratio: float = 0.25, num_wavelet_levels: int = 3, ...)
  ```

- `PRISMLayer` - Keras Layer
  Main PRISM layer combining hierarchical time-frequency decomposition.
  ```python
  PRISMLayer(tree_depth: int = 2, overlap_ratio: float = 0.25, num_wavelet_levels: int = 3, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `build`, `call` (and 16 more)

*📁 File: `src/dl_techniques/layers/time_series/prism_blocks.py`*

### layers.time_series.quantile_head_fixed_io
A quantile prediction head for probabilistic forecasting.

**Classes:**

- `QuantileHead` - Keras Layer
  Quantile prediction head for probabilistic time series forecasting.
  ```python
  QuantileHead(num_quantiles: int, output_length: int, dropout_rate: float = 0.1, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/quantile_head_fixed_io.py`*

### layers.time_series.quantile_head_variable_io
A sequence-to-sequence quantile prediction head for probabilistic forecasting.

**Classes:**

- `QuantileSequenceHead` - Keras Layer
  Sequence-wise quantile prediction head for probabilistic time series forecasting.
  ```python
  QuantileSequenceHead(num_quantiles: int, dropout_rate: float = 0.1, use_bias: bool = True, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/quantile_head_variable_io.py`*

### layers.time_series.temporal_convolutional_network

**Classes:**

- `TemporalBlock` - Keras Layer
  A single residual block for the Temporal Convolutional Network.
  ```python
  TemporalBlock(filters: int, kernel_size: int, dilation_rate: int, ...)
  ```

- `TemporalConvNet` - Keras Layer
  Temporal Convolutional Network (TCN) Encoder.
  ```python
  TemporalConvNet(filters: int, kernel_size: int = 2, num_levels: int = 4, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `call`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/temporal_convolutional_network.py`*

### layers.time_series.temporal_fusion
Fuse contextual and autoregressive forecasts with a dynamic gating mechanism.

**Classes:**

- `TemporalFusionLayer` - Keras Layer
  A layer that fuses a context-based forecast with an attention-based autoregressive forecast.
  ```python
  TemporalFusionLayer(output_dim: int, num_lags: int, project_lags: bool = False, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/time_series/temporal_fusion.py`*

### layers.time_series.xlstm_blocks
xLSTM (Extended Long Short-Term Memory) Implementation.

**Classes:**

- `sLSTMCell` - Keras Layer
  Scalar LSTM (sLSTM) Cell with exponential gating and normalizer state.
  ```python
  sLSTMCell(units: int, forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid', kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform', ...)
  ```

- `sLSTMLayer` - Keras Layer
  Scalar LSTM (sLSTM) layer for processing sequences.
  ```python
  sLSTMLayer(units: int, forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid', return_sequences: bool = True, ...)
  ```

- `mLSTMCell` - Keras Layer
  Matrix LSTM (mLSTM) Cell with matrix memory and covariance update rule.
  ```python
  mLSTMCell(units: int, num_heads: int = 1, key_dim: Optional[int] = None, ...)
  ```

- `mLSTMLayer` - Keras Layer
  Matrix LSTM (mLSTM) layer for processing sequences.
  ```python
  mLSTMLayer(units: int, num_heads: int = 1, key_dim: Optional[int] = None, ...)
  ```

- `sLSTMBlock` - Keras Layer
  sLSTM residual block with post-normalization architecture.
  ```python
  sLSTMBlock(units: int, ffn_type: str = 'swiglu', ffn_expansion_factor: int = 2, ...)
  ```

- `mLSTMBlock` - Keras Layer
  mLSTM residual block with pre-up-projection architecture.
  ```python
  mLSTMBlock(units: int, expansion_factor: int = 2, num_heads: int = 1, ...)
  ```

**Functions:** `build`, `call`, `get_initial_state`, `get_config`, `build` (and 17 more)

*📁 File: `src/dl_techniques/layers/time_series/xlstm_blocks.py`*

## Tokenizers

### layers.tokenizers

*📁 File: `src/dl_techniques/layers/tokenizers/__init__.py`*

### layers.tokenizers.bpe
Custom Byte-Pair Encoding (BPE) Tokenizer implementation for Keras 3.x.

**Classes:**

- `BPETokenizer` - Keras Layer
  Byte-Pair Encoding (BPE) tokenizer layer for Keras 3.x.
  ```python
  BPETokenizer(vocab_dict: Optional[Dict[str, int]] = None, merges: Optional[List[Tuple[str, str]]] = None, max_length: int = 512, ...)
  ```

- `TokenEmbedding` - Keras Layer
  Token embedding layer that converts token IDs to dense vectors.
  ```python
  TokenEmbedding(vocab_size: int, embedding_dim: int, mask_zero: bool = True, ...)
  ```

**Functions:** `train_bpe`, `create_bpe_pipeline`, `call`, `tokenize_texts`, `decode_tokens` (and 5 more)

*📁 File: `src/dl_techniques/layers/tokenizers/bpe.py`*

## Transformers

### layers.transformers
Core Transformer Blocks for Building Advanced Models.

*📁 File: `src/dl_techniques/layers/transformers/__init__.py`*

### layers.transformers.eomt_transformer
A Transformer encoder layer for joint patch-query processing.

**Classes:**

- `EomtTransformer` - Keras Layer
  Configurable Encoder-only Mask Transformer layer for vision_heads segmentation.
  ```python
  EomtTransformer(hidden_size: int, num_heads: int = 8, intermediate_size: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/transformers/eomt_transformer.py`*

### layers.transformers.free_transformer
Free Transformer Layer with integrated Variational Autoencoder components.

**Classes:**

- `BinaryMapper` - Keras Layer
  Samples one-hot vectors from bit logits with gradient pass-through.
  ```python
  BinaryMapper(num_bits: int, **kwargs)
  ```

- `FreeTransformerLayer` - Keras Layer
  A Transformer layer extended with the Free Transformer C-VAE architecture.
  ```python
  FreeTransformerLayer(hidden_size: int, num_heads: int, intermediate_size: int, ...)
  ```

**Functions:** `compute_kl_divergence_uniform_prior`, `compute_free_bits_kl_loss`, `call`, `compute_output_shape`, `get_config` (and 4 more)

*📁 File: `src/dl_techniques/layers/transformers/free_transformer.py`*

### layers.transformers.perceiver_transformer
Perceiver-style Transformer block with decoupled cross-attention.

**Classes:**

- `PerceiverTransformerLayer` - Keras Layer
  Complete Perceiver transformer block with cross-attention.
  ```python
  PerceiverTransformerLayer(dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/transformers/perceiver_transformer.py`*

### layers.transformers.progressive_focused_transformer
Progressive Focused Transformer (PFT) Block Module.

**Classes:**

- `PFTBlock` - Keras Layer
  Progressive Focused Transformer Block.
  ```python
  PFTBlock(dim: int, num_heads: int, window_size: int = 8, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `from_config`, `compute_output_shape`

*📁 File: `src/dl_techniques/layers/transformers/progressive_focused_transformer.py`*

### layers.transformers.swin_conv_block
SwinConvBlock: A hybrid Keras layer that synergistically combines the strengths

**Classes:**

- `SwinConvBlock` - Keras Layer
  Hybrid Swin-Conv block combining transformer and convolutional paths in parallel.
  ```python
  SwinConvBlock(conv_dim: int, trans_dim: int, head_dim: int = 32, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/transformers/swin_conv_block.py`*

### layers.transformers.swin_transformer_block
Swin Transformer Block Implementation

**Classes:**

- `SwinTransformerBlock` - Keras Layer
  Swin Transformer Block with windowed multi-head self-attention.
  ```python
  SwinTransformerBlock(dim: int, num_heads: int, window_size: int = 8, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/transformers/swin_transformer_block.py`*

### layers.transformers.text_decoder
A configurable, Transformer-based text decoder stack.

**Classes:**

- `TextDecoder` - Keras Layer
  General-purpose configurable text decoder built upon a stack of TransformerLayers.
  ```python
  TextDecoder(vocab_size: int, embed_dim: int, depth: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/transformers/text_decoder.py`*

### layers.transformers.text_encoder
A highly configurable, Transformer-based text encoder.

**Classes:**

- `TextEncoder` - Keras Layer
  General purpose configurable text encoder using factory-based components.
  ```python
  TextEncoder(vocab_size: int, embed_dim: int, depth: int = 12, ...)
  ```

**Functions:** `create_text_encoder`, `create_bert_encoder`, `create_roberta_encoder`, `create_modern_encoder`, `create_efficient_encoder` (and 6 more)

*📁 File: `src/dl_techniques/layers/transformers/text_encoder.py`*

### layers.transformers.transformer
Foundational building block of a Transformer network, implementing a highly

**Classes:**

- `TransformerLayer` - Keras Layer
  Generic transformer layer with configurable attention, FFN, and normalization.
  ```python
  TransformerLayer(hidden_size: int, num_heads: int, intermediate_size: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/transformers/transformer.py`*

### layers.transformers.vision_encoder
A configurable, general-purpose Vision Transformer encoder.

**Classes:**

- `VisionEncoder` - Keras Layer
  General purpose configurable vision_heads encoder using factory-based components.
  ```python
  VisionEncoder(img_size: int = 224, patch_size: int = 16, embed_dim: int = 768, ...)
  ```

**Functions:** `create_vision_encoder`, `create_vit_encoder`, `create_siglip_encoder`, `build`, `call` (and 5 more)

*📁 File: `src/dl_techniques/layers/transformers/vision_encoder.py`*

## Tversky_Projection

### layers.tversky_projection
A projection layer based on Tversky's contrast model of similarity.

**Classes:**

- `TverskyProjectionLayer` - Keras Layer
  A projection layer based on a differentiable Tversky similarity model.
  ```python
  TverskyProjectionLayer(units: int, num_features: int, intersection_reduction: Literal['product', 'min', 'mean'] = 'product', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/tversky_projection.py`*

## Universal_Inverted_Bottleneck

### layers.universal_inverted_bottleneck
A `Universal Inverted Bottleneck` (UIB), a highly flexible

**Classes:**

- `UniversalInvertedBottleneck` - Keras Layer
  Universal Inverted Bottleneck (UIB) - A highly configurable building block for efficient CNNs.
  ```python
  UniversalInvertedBottleneck(filters: int, expansion_factor: int = 4, expanded_channels: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/layers/universal_inverted_bottleneck.py`*

## Upsample

### layers.upsample
A neural network upsampling block using various strategies.

**Functions:** `upsample`

*📁 File: `src/dl_techniques/layers/upsample.py`*

## Vector_Quantizer

### layers.vector_quantizer

**Classes:**

- `VectorQuantizer` - Keras Layer
  Vector Quantization layer for discrete latent representations.
  ```python
  VectorQuantizer(num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, ...)
  ```

**Functions:** `build`, `call`, `get_codebook_indices`, `quantize_from_indices`, `compute_output_shape` (and 1 more)

*📁 File: `src/dl_techniques/layers/vector_quantizer.py`*

## Vision_Heads

### layers.vision_heads

*📁 File: `src/dl_techniques/layers/vision_heads/__init__.py`*

### layers.vision_heads.factory
Vision Task Head Network Factory

**Classes:**

- `BaseVisionHead` - Keras Layer
  Base class for all vision_heads task heads.
  ```python
  BaseVisionHead(hidden_dim: int = 256, normalization_type: NormalizationType = 'layer_norm', activation_type: ActivationType = 'gelu', ...)
  ```
- `DetectionHead`
- `SegmentationHead`
- `DepthEstimationHead`
- `ClassificationHead`
- `InstanceSegmentationHead`

- `MultiTaskHead` - Keras Layer
  Multi-task head that combines multiple task-specific heads.
  ```python
  MultiTaskHead(task_configs: Dict[str, Dict[str, Any]], shared_backbone_dim: int = 256, use_task_specific_attention: bool = True, ...)
  ```
- `HeadConfiguration`
- `EnhancementHead`

**Functions:** `create_vision_head`, `create_enhancement_head`, `create_multi_task_head`, `build`, `get_config` (and 17 more)

*📁 File: `src/dl_techniques/layers/vision_heads/factory.py`*

### layers.vision_heads.task_types

**Classes:**
- `TaskType`
- `TaskConfiguration`
- `CommonTaskConfigurations`

**Functions:** `parse_task_list`, `get_task_suggestions`, `validate_task_combination`, `all_tasks`, `get_task_categories` (and 29 more)

*📁 File: `src/dl_techniques/layers/vision_heads/task_types.py`*

## Vlm_Heads

### layers.vlm_heads

*📁 File: `src/dl_techniques/layers/vlm_heads/__init__.py`*

### layers.vlm_heads.factory
VLM Task Head Factory

**Classes:**

- `BaseVLMHead` - Keras Layer
  Base class for all VLM task heads, using an advanced fusion module.
  ```python
  BaseVLMHead(task_config: VLMTaskConfig, vision_dim: int = 768, text_dim: int = 768, ...)
  ```

- `ImageCaptioningHead` - Keras Layer
  An autoregressive decoder head for generating text conditioned on vision features.
  ```python
  ImageCaptioningHead(task_config: VLMTaskConfig, vision_dim: int = 768, text_dim: int = 768, ...)
  ```

- `VQAHead` - Keras Layer
  A multimodal fusion and classification head for Visual Question Answering.
  ```python
  VQAHead(task_config: VLMTaskConfig, vision_dim: int = 768, text_dim: int = 768, ...)
  ```
- `VisualGroundingHead`
- `ImageTextMatchingHead`

- `MultiTaskVLMHead` - Keras Layer
  Multi-task head combining multiple VLM task-specific heads.
  ```python
  MultiTaskVLMHead(task_configs: Dict[str, VLMTaskConfig], shared_vision_dim: int = 768, shared_text_dim: int = 768, ...)
  ```

**Functions:** `get_head_class`, `create_vlm_head`, `create_multi_task_vlm_head`, `build`, `get_config` (and 9 more)

*📁 File: `src/dl_techniques/layers/vlm_heads/factory.py`*

### layers.vlm_heads.task_types
VLM Task Types and Configuration

**Classes:**
- `VLMTaskType`
- `VLMTaskConfig`
- `VLMTaskConfiguration`

**Functions:** `all_tasks`, `get_task_categories`, `from_string`, `tasks`, `has_task` (and 2 more)

*📁 File: `src/dl_techniques/layers/vlm_heads/task_types.py`*

## Yolo12_Blocks

### layers.yolo12_blocks
YOLOv12 Core Building Blocks.

**Classes:**

- `ConvBlock` - Keras Layer
  Standard Convolution Block with BatchNorm and SiLU activation.
  ```python
  ConvBlock(filters: int, kernel_size: int = 3, strides: int = 1, ...)
  ```

- `AreaAttention` - Keras Layer
  Area Attention mechanism for YOLOv12.
  ```python
  AreaAttention(dim: int, num_heads: int = 8, area: int = 1, ...)
  ```

- `AttentionBlock` - Keras Layer
  Attention Block with Area Attention and MLP.
  ```python
  AttentionBlock(dim: int, num_heads: int = 8, mlp_ratio: float = 1.2, ...)
  ```

- `Bottleneck` - Keras Layer
  Standard Bottleneck block with optional residual connection.
  ```python
  Bottleneck(filters: int, shortcut: bool = True, kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal', ...)
  ```

- `C3k2Block` - Keras Layer
  CSP-like block with 2 convolutions and Bottleneck layers.
  ```python
  C3k2Block(filters: int, n: int = 1, shortcut: bool = True, ...)
  ```

- `A2C2fBlock` - Keras Layer
  Attention-enhanced R-ELAN block with progressive feature extraction.
  ```python
  A2C2fBlock(filters: int, n: int = 1, area: int = 1, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 19 more)

*📁 File: `src/dl_techniques/layers/yolo12_blocks.py`*

## Yolo12_Heads

### layers.yolo12_heads
YOLOv12 Task-Specific Heads for Multi-Task Learning.

**Classes:**

- `YOLOv12DetectionHead` - Keras Layer
  YOLOv12 Detection Head with separate classification and regression branches.
  ```python
  YOLOv12DetectionHead(num_classes: int = 80, reg_max: int = 16, bbox_channels: Optional[List[int]] = None, ...)
  ```

- `YOLOv12SegmentationHead` - Keras Layer
  Segmentation head for YOLOv12 multitask learning.
  ```python
  YOLOv12SegmentationHead(num_classes: int = 1, intermediate_filters: List[int] = [128, 64, 32, 16], target_size: Optional[Tuple[int, int]] = None, ...)
  ```

- `YOLOv12ClassificationHead` - Keras Layer
  Classification head for YOLOv12 multitask learning.
  ```python
  YOLOv12ClassificationHead(num_classes: int = 1, hidden_dims: List[int] = [512, 256], pooling_types: List[str] = ['avg', 'max'], ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 7 more)

*📁 File: `src/dl_techniques/layers/yolo12_heads.py`*