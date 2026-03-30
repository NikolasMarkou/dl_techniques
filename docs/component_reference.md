# Component Reference

All public components in the DL-Techniques library organized by type.

## Classes (806)

### Analyzer Classes

#### `AnalysisConfig`
**Module:** `analyzer.config`

Configuration for all analysis types.

*📁 src/dl_techniques/analyzer/config.py:15*

#### `AnalysisResults`
**Module:** `analyzer.data_types`

Container for all analysis results.

*📁 src/dl_techniques/analyzer/data_types.py:50*

#### `BaseAnalyzer`
**Module:** `analyzer.analyzers.base`

Abstract base class for all analyzers.

*Inherits from: `ABC`*

*📁 src/dl_techniques/analyzer/analyzers/base.py:14*

#### `BaseVisualizer`
**Module:** `analyzer.visualizers.base`

Abstract base class for all visualizers with centralized legend management.

*Inherits from: `ABC`*

*📁 src/dl_techniques/analyzer/visualizers/base.py:42*

#### `CalibrationAnalyzer`
**Module:** `analyzer.analyzers.calibration_analyzer`

Analyzes model confidence and calibration.

*Inherits from: `BaseAnalyzer`*

*📁 src/dl_techniques/analyzer/analyzers/calibration_analyzer.py:92*

#### `CalibrationVisualizer`
**Module:** `analyzer.visualizers.calibration_visualizer`

Creates calibration analysis visualizations with centralized legend.

*Inherits from: `BaseVisualizer`*

*📁 src/dl_techniques/analyzer/visualizers/calibration_visualizer.py:77*

#### `DataInput`
**Module:** `analyzer.data_types`

Structured data input type.

*Inherits from: `NamedTuple`*

*📁 src/dl_techniques/analyzer/data_types.py:12*

#### `DataSampler`
**Module:** `analyzer.utils`

Helper class to handle robust data sampling from various input formats.

*📁 src/dl_techniques/analyzer/utils.py:24*

#### `InformationFlowAnalyzer`
**Module:** `analyzer.analyzers.information_flow_analyzer`

Analyzes information flow through network layers.

*Inherits from: `BaseAnalyzer`*

*📁 src/dl_techniques/analyzer/analyzers/information_flow_analyzer.py:59*

#### `InformationFlowVisualizer`
**Module:** `analyzer.visualizers.information_flow_visualizer`

Creates information flow visualizations with centralized legend.

*Inherits from: `BaseVisualizer`*

*📁 src/dl_techniques/analyzer/visualizers/information_flow_visualizer.py:77*

#### `LayerType`
**Module:** `analyzer.constants`

Enum for supported layer types for spectral analysis

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/analyzer/constants.py:45*

#### `MetricNames`
**Module:** `analyzer.constants`

Class holding the standard names of metrics used in spectral analysis

*📁 src/dl_techniques/analyzer/constants.py:70*

#### `ModelAnalyzer`
**Module:** `analyzer.model_analyzer`

Model analyzer with training dynamics and improved visualizations.

*📁 src/dl_techniques/analyzer/model_analyzer.py:207*

#### `SmoothingMethod`
**Module:** `analyzer.constants`

Enum for SVD smoothing methods

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/analyzer/constants.py:57*

#### `SpectralAnalyzer`
**Module:** `analyzer.analyzers.spectral_analyzer`

Performs spectral analysis of model weights (WeightWatcher).

*Inherits from: `BaseAnalyzer`*

*📁 src/dl_techniques/analyzer/analyzers/spectral_analyzer.py:104*

#### `SpectralVisualizer`
**Module:** `analyzer.visualizers.spectral_visualizer`

Creates visualizations for spectral analysis results.

*Inherits from: `BaseVisualizer`*

*📁 src/dl_techniques/analyzer/visualizers/spectral_visualizer.py:29*

#### `StatusCode`
**Module:** `analyzer.constants`

Enum for spectral analysis status codes

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/analyzer/constants.py:63*

#### `SummaryVisualizer`
**Module:** `analyzer.visualizers.summary_visualizer`

Creates comprehensive summary dashboard visualization with centralized legend.

*Inherits from: `BaseVisualizer`*

*📁 src/dl_techniques/analyzer/visualizers/summary_visualizer.py:88*

#### `TrainingDynamicsAnalyzer`
**Module:** `analyzer.analyzers.training_dynamics_analyzer`

Analyzes training dynamics from history.

*Inherits from: `BaseAnalyzer`*

*📁 src/dl_techniques/analyzer/analyzers/training_dynamics_analyzer.py:85*

#### `TrainingDynamicsVisualizer`
**Module:** `analyzer.visualizers.training_dynamics_visualizer`

Creates training dynamics visualizations with centralized legend.

*Inherits from: `BaseVisualizer`*

*📁 src/dl_techniques/analyzer/visualizers/training_dynamics_visualizer.py:70*

#### `TrainingMetrics`
**Module:** `analyzer.data_types`

Container for computed training metrics.

*📁 src/dl_techniques/analyzer/data_types.py:29*

#### `WeightAnalyzer`
**Module:** `analyzer.analyzers.weight_analyzer`

Analyzes weight distributions and statistics.

*Inherits from: `BaseAnalyzer`*

*📁 src/dl_techniques/analyzer/analyzers/weight_analyzer.py:88*

#### `WeightVisualizer`
**Module:** `analyzer.visualizers.weight_visualizer`

Creates weight analysis visualizations with centralized legend.

*Inherits from: `BaseVisualizer`*

*📁 src/dl_techniques/analyzer/visualizers/weight_visualizer.py:76*

### Constraints Classes

#### `ValueRangeConstraint`
**Module:** `constraints.value_range_constraint`

Constrains weights to be within specified minimum and maximum values.

*Inherits from: `keras.constraints.Constraint`*

*📁 src/dl_techniques/constraints/value_range_constraint.py:57*

### Initializers Classes

#### `HaarWaveletInitializer`
**Module:** `initializers.haar_wavelet_initializer`

Haar wavelet initializer for convolutional layers.

*Inherits from: `keras.initializers.Initializer`*

*📁 src/dl_techniques/initializers/haar_wavelet_initializer.py:66*

#### `HeOrthonormalInitializer`
**Module:** `initializers.he_orthonormal_initializer`

Custom initializer that applies He normal initialization followed by orthonormalization.

*Inherits from: `keras.initializers.Initializer`*

*📁 src/dl_techniques/initializers/he_orthonormal_initializer.py:63*

#### `OrthogonalHypersphereInitializer`
**Module:** `initializers.hypersphere_orthogonal_initializer`

Orthogonal hypersphere weight initializer with mathematical dimensionality constraints.

*Inherits from: `keras.initializers.Initializer`*

*📁 src/dl_techniques/initializers/hypersphere_orthogonal_initializer.py:64*

#### `OrthonormalInitializer`
**Module:** `initializers.orthonormal_initializer`

Custom initializer for orthonormal vectors using QR decomposition.

*Inherits from: `keras.initializers.Initializer`*

*📁 src/dl_techniques/initializers/orthonormal_initializer.py:63*

### Layers Classes

#### `A2C2fBlock`
**Module:** `layers.yolo12_blocks`

Attention-enhanced R-ELAN block with progressive feature extraction.

**Constructor Arguments:**
```python
A2C2fBlock(
    filters: int,
    n: int = 1,
    area: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_blocks.py:1013*

#### `ACFMonitorCallback`
**Module:** `layers.statistics.residual_acf`

Callback to monitor and log ACF statistics during training.

*Inherits from: `keras.callbacks.Callback`*

*📁 src/dl_techniques/layers/statistics/residual_acf.py:496*

#### `AdaptiveBandRMS`
**Module:** `layers.norms.adaptive_band_rms`

Adaptive Root Mean Square Normalization with log-transformed RMS scaling.

**Constructor Arguments:**
```python
AdaptiveBandRMS(
    max_band_width: float = 0.1,
    axis: Union[int, Tuple[int, ...]] = -1,
    epsilon: float = 1e-07,
    band_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    band_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/adaptive_band_rms.py:41*

#### `AdaptiveLagAttentionLayer`
**Module:** `layers.time_series.adaptive_lag_attention`

Advanced attention layer for dynamically weighting temporal lags with gating control.

**Constructor Arguments:**
```python
AdaptiveLagAttentionLayer(
    num_lags: int,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/adaptive_lag_attention.py:80*

#### `AdaptiveTemperatureSoftmax`
**Module:** `layers.activations.adaptive_softmax`

Adaptive Temperature Softmax layer with entropy-based temperature adaptation.

**Constructor Arguments:**
```python
AdaptiveTemperatureSoftmax(
    min_temp: float = 0.1,
    max_temp: float = 1.0,
    entropy_threshold: float = 0.5,
    eps: Optional[float] = None,
    polynomial_coeffs: Optional[List[float]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/adaptive_softmax.py:73*

#### `AddressingMode`
**Module:** `layers.ntm.ntm_interface`

Enumeration of addressing mechanism types.

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:36*

#### `AdvancedLaplacianFilter`
**Module:** `layers.laplacian_filter`

Advanced Laplacian filter with multiple implementation options.

**Constructor Arguments:**
```python
AdvancedLaplacianFilter(
    method: Literal['dog', 'log', 'kernel'] = 'dog',
    kernel_size: Tuple[int, int] = (5, 5),
    strides: Union[Tuple[int, int], List[int]] = (1, 1),
    sigma: Union[float, Tuple[float, float]] = 1.0,
    scale_factor: float = 1.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/laplacian_filter.py:239*

#### `AffineCouplingLayer`
**Module:** `layers.statistics.normalizing_flow`

Affine coupling transformation layer for normalizing flows with conditional context.

**Constructor Arguments:**
```python
AffineCouplingLayer(
    input_dim: int,
    context_dim: int,
    hidden_units: int = 64,
    reverse: bool = False,
    activation: Union[str, callable] = 'relu',
    use_tanh_stabilization: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:60*

#### `AnchorAttention`
**Module:** `layers.attention.anchor_attention`

Hierarchical attention mechanism with anchor-based information bottleneck.

**Constructor Arguments:**
```python
AnchorAttention(
    dim: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    probability_type: str = 'softmax',
    probability_config: Optional[Dict[str, Any]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/anchor_attention.py:61*

#### `AnchorGenerator`
**Module:** `layers.anchor_generator`

Anchor generator layer for YOLOv12 object detection.

**Constructor Arguments:**
```python
AnchorGenerator(
    input_image_shape: Tuple[int, int],
    strides_config: Optional[List[int]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/anchor_generator.py:82*

#### `AntennalLobeLayer`
**Module:** `layers.mothnet_blocks`

Antennal Lobe layer implementing competitive inhibition for contrast enhancement.

**Constructor Arguments:**
```python
AntennalLobeLayer(
    units: int,
    inhibition_strength: float = 0.5,
    activation: str = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/mothnet_blocks.py:62*

#### `ApproximatedLNNLayer`
**Module:** `layers.physics.approximate_lagrange_layer`

Gradient-tape-free approximation of Lagrangian Neural Network dynamics.

**Constructor Arguments:**
```python
ApproximatedLNNLayer(
    hidden_dims: List[int],
    activation: str = 'softplus',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/physics/approximate_lagrange_layer.py:7*

#### `AreaAttention`
**Module:** `layers.yolo12_blocks`

Area Attention mechanism for YOLOv12.

**Constructor Arguments:**
```python
AreaAttention(
    dim: int,
    num_heads: int = 8,
    area: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_blocks.py:243*

#### `AttentionBlock`
**Module:** `layers.yolo12_blocks`

Attention Block with Area Attention and MLP.

**Constructor Arguments:**
```python
AttentionBlock(
    dim: int,
    num_heads: int = 8,
    mlp_ratio: float = 1.2,
    area: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_blocks.py:507*

#### `AttentionPooling`
**Module:** `layers.sequence_pooling`

Attention-based pooling that learns to weight sequence elements.

**Constructor Arguments:**
```python
AttentionPooling(
    hidden_dim: int = 256,
    num_heads: int = 1,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    temperature: float = 1.0,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/sequence_pooling.py:108*

#### `BPETokenizer`
**Module:** `layers.tokenizers.bpe`

Byte-Pair Encoding (BPE) tokenizer layer for Keras 3.x.

**Constructor Arguments:**
```python
BPETokenizer(
    vocab_dict: Optional[Dict[str, int]] = None,
    merges: Optional[List[Tuple[str, str]]] = None,
    max_length: int = 512,
    pad_token: str = '<pad>',
    unk_token: str = '<unk>',
    eos_token: str = '<eos>',
    do_lower_case: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tokenizers/bpe.py:123*

#### `BandLogitNorm`
**Module:** `layers.norms.band_logit_norm`

Band-constrained logit normalization layer.

**Constructor Arguments:**
```python
BandLogitNorm(
    max_band_width: float = 0.01,
    axis: int = -1,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/band_logit_norm.py:26*

#### `BandRMS`
**Module:** `layers.norms.band_rms`

Root Mean Square Normalization layer with bounded RMS constraints.

**Constructor Arguments:**
```python
BandRMS(
    max_band_width: float = 0.1,
    axis: Union[int, Tuple[int, ...]] = -1,
    epsilon: float = 1e-07,
    band_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    band_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/band_rms.py:59*

#### `BandRMSOOD`
**Module:** `layers.experimental.band_rms_ood`

BandRMS-OOD: Geometric Out-of-Distribution Detection Layer.

**Constructor Arguments:**
```python
BandRMSOOD(
    max_band_width: float = 0.1,
    confidence_type: Literal['magnitude', 'entropy', 'prediction'] = 'magnitude',
    confidence_weight: float = 1.0,
    shell_preference_weight: float = 0.01,
    axis: int = -1,
    epsilon: float = 1e-07,
    momentum: float = 0.99,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:50*

#### `BaseActivation`
**Module:** `layers.activations.expanded_activations`

Base class for all custom activation functions.

**Constructor Arguments:**
```python
BaseActivation(
    trainable: bool = True,
    name: Optional[str] = None,
    dtype: Optional[Union[str, keras.ops.dtype]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:127*

#### `BaseController`
**Module:** `layers.ntm.ntm_interface`

Abstract base class for controller networks.

**Constructor Arguments:**
```python
BaseController(
    controller_dim: int,
    controller_type: Literal['lstm', 'gru', 'feedforward'] = 'lstm',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`, `ABC`*

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:534*

#### `BaseExpert`
**Module:** `layers.moe.experts`

Abstract base class for MoE expert networks.

**Constructor Arguments:**
```python
BaseExpert(
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`, `ABC`*

*📁 src/dl_techniques/layers/moe/experts.py:22*

#### `BaseGating`
**Module:** `layers.moe.gating`

Abstract base class for MoE gating networks.

**Constructor Arguments:**
```python
BaseGating(
    num_experts: int,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`, `ABC`*

*📁 src/dl_techniques/layers/moe/gating.py:17*

#### `BaseHead`
**Module:** `layers.ntm.ntm_interface`

Abstract base class for read and write heads.

**Constructor Arguments:**
```python
BaseHead(
    memory_size: int,
    memory_dim: int,
    addressing_mode: AddressingMode = AddressingMode.HYBRID,
    shift_range: int = 3,
    epsilon: float = 1e-06,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`, `ABC`*

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:421*

#### `BaseMemory`
**Module:** `layers.ntm.ntm_interface`

Abstract base class for memory modules.

**Constructor Arguments:**
```python
BaseMemory(
    memory_size: int,
    memory_dim: int,
    epsilon: float = 1e-06,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`, `ABC`*

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:310*

#### `BaseNLPHead`
**Module:** `layers.nlp_heads.factory`

Base class for all NLP task heads.

**Constructor Arguments:**
```python
BaseNLPHead(
    task_config: NLPTaskConfig,
    input_dim: int,
    normalization_type: NormalizationType = 'layer_norm',
    activation_type: ActivationType = 'gelu',
    use_pooling: bool = True,
    pooling_type: Literal['mean', 'max', 'cls', 'attention'] = 'cls',
    use_intermediate: bool = True,
    intermediate_size: Optional[int] = None,
    use_task_attention: bool = False,
    attention_type: AttentionType = 'multi_head',
    use_ffn: bool = False,
    ffn_type: FFNType = 'mlp',
    ffn_expansion_factor: int = 4,
    initializer_range: float = 0.02,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:29*

#### `BaseNTM`
**Module:** `layers.ntm.ntm_interface`

Abstract base class for Neural Turing Machine architectures.

**Constructor Arguments:**
```python
BaseNTM(
    config: NTMConfig,
    output_dim: int | None = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`, `ABC`*

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:617*

#### `BaseVLMHead`
**Module:** `layers.vlm_heads.factory`

Base class for all VLM task heads, using an advanced fusion module.

**Constructor Arguments:**
```python
BaseVLMHead(
    task_config: VLMTaskConfig,
    vision_dim: int = 768,
    text_dim: int = 768,
    fusion_strategy: FusionStrategy = 'cross_attention',
    fusion_config: Optional[Dict[str, Any]] = None,
    normalization_type: NormalizationType = 'layer_norm',
    activation_type: ActivationType = 'gelu',
    use_post_fusion_ffn: bool = True,
    ffn_type: FFNType = 'mlp',
    ffn_expansion_factor: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/vlm_heads/factory.py:31*

#### `BaseVisionHead`
**Module:** `layers.vision_heads.factory`

Base class for all vision_heads task heads.

**Constructor Arguments:**
```python
BaseVisionHead(
    hidden_dim: int = 256,
    normalization_type: NormalizationType = 'layer_norm',
    activation_type: ActivationType = 'gelu',
    dropout_rate: float = 0.1,
    use_attention: bool = False,
    attention_type: AttentionType = 'multi_head',
    use_ffn: bool = True,
    ffn_type: FFNType = 'mlp',
    ffn_expansion_factor: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:29*

#### `BasicBlock`
**Module:** `layers.standard_blocks`

Basic ResNet block with two 3x3 convolutions.

**Constructor Arguments:**
```python
BasicBlock(
    filters: int,
    stride: int = 1,
    use_projection: bool = False,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    normalization_type: str = 'batch_norm',
    activation_type: str = 'relu',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/standard_blocks.py:843*

#### `BasisFunction`
**Module:** `layers.activations.basis_function`

Basis function layer implementing the Swish activation: b(x) = x / (1 + e^(-x)).

**Constructor Arguments:**
```python
BasisFunction(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/basis_function.py:59*

#### `BertEmbeddings`
**Module:** `layers.embedding.bert_embeddings`

BERT embeddings layer combining word, position, and token type embeddings.

**Constructor Arguments:**
```python
BertEmbeddings(
    vocab_size: int,
    hidden_size: int,
    max_position_embeddings: int,
    type_vocab_size: int,
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-08,
    dropout_rate: float = 0.0,
    normalization_type: str = 'layer_norm',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/bert_embeddings.py:80*

#### `BiasFreeConv1D`
**Module:** `layers.bias_free_conv1d`

Bias-free 1D convolutional layer with batch normalization and activation.

**Constructor Arguments:**
```python
BiasFreeConv1D(
    filters: int,
    kernel_size: int = 3,
    activation: Union[str, callable] = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    use_batch_norm: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/bias_free_conv1d.py:25*

#### `BiasFreeConv2D`
**Module:** `layers.bias_free_conv2d`

Bias-free 2D convolutional layer with batch normalization and activation.

**Constructor Arguments:**
```python
BiasFreeConv2D(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    activation: Union[str, callable] = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    use_batch_norm: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/bias_free_conv2d.py:25*

#### `BiasFreeResidualBlock`
**Module:** `layers.bias_free_conv2d`

Bias-free residual block for ResNet-style architecture with 2D convolutions.

**Constructor Arguments:**
```python
BiasFreeResidualBlock(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    activation: Union[str, callable] = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    use_batch_norm: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/bias_free_conv2d.py:289*

#### `BiasFreeResidualBlock1D`
**Module:** `layers.bias_free_conv1d`

Bias-free residual block for ResNet-style architecture with 1D convolutions.

**Constructor Arguments:**
```python
BiasFreeResidualBlock1D(
    filters: int,
    kernel_size: int = 3,
    activation: Union[str, callable] = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/bias_free_conv1d.py:269*

#### `BinaryMapper`
**Module:** `layers.transformers.free_transformer`

Samples one-hot vectors from bit logits with gradient pass-through.

**Constructor Arguments:**
```python
BinaryMapper(
    num_bits: int,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/free_transformer.py:72*

#### `BitLinear`
**Module:** `layers.bitlinear_layer`

Bit-aware linear layer for quantization-aware training.

**Constructor Arguments:**
```python
BitLinear(
    units: int,
    weight_bits: Union[float, int, Tuple[float, float]] = 1.58,
    activation_bits: Union[float, int, Tuple[float, float]] = 8,
    weight_scale_method: str = 'abs_median',
    activation_scale_method: str = 'abs_max',
    quantization_method: str = 'round_clip',
    use_bias: bool = True,
    use_input_norm: bool = False,
    ste_lambda: float = 1.0,
    epsilon: float = 1e-05,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/bitlinear_layer.py:69*

#### `Bottleneck`
**Module:** `layers.yolo12_blocks`

Standard Bottleneck block with optional residual connection.

**Constructor Arguments:**
```python
Bottleneck(
    filters: int,
    shortcut: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_blocks.py:684*

#### `BottleneckBlock`
**Module:** `layers.standard_blocks`

Bottleneck ResNet block with 1x1 → 3x3 → 1x1 convolutions.

**Constructor Arguments:**
```python
BottleneckBlock(
    filters: int,
    stride: int = 1,
    use_projection: bool = False,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    normalization_type: str = 'batch_norm',
    activation_type: str = 'relu',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/standard_blocks.py:1084*

#### `ByteLatentReasoningCore`
**Module:** `layers.blt_core`

Core hierarchical reasoning model that operates on dynamic byte patches.

**Constructor Arguments:**
```python
ByteLatentReasoningCore(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    local_dim: int = 512,
    global_dim: int = 768,
    max_patches: int = 512,
    num_puzzle_identifiers: int = 1000,
    puzzle_emb_dim: int = 512,
    batch_size: int = 32,
    h_layers: int = 4,
    l_layers: int = 4,
    h_cycles: int = 2,
    l_cycles: int = 2,
    num_heads: int = 8,
    entropy_threshold: float = 1.5,
    pos_encodings: str = 'rope',
    rope_theta: float = 10000.0,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    embeddings_initializer: Union[str, keras.initializers.Initializer] = 'truncated_normal',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_core.py:101*

#### `ByteTokenizer`
**Module:** `layers.blt_blocks`

Converts text to byte tokens for BLT processing.

**Constructor Arguments:**
```python
ByteTokenizer(
    vocab_size: int = 260,
    byte_offset: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:232*

#### `C3k2Block`
**Module:** `layers.yolo12_blocks`

CSP-like block with 2 convolutions and Bottleneck layers.

**Constructor Arguments:**
```python
C3k2Block(
    filters: int,
    n: int = 1,
    shortcut: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_blocks.py:829*

#### `CBAM`
**Module:** `layers.attention.convolutional_block_attention`

Convolutional Block Attention Module for feature refinement.

**Constructor Arguments:**
```python
CBAM(
    channels: int,
    ratio: int = 8,
    kernel_size: int = 7,
    channel_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    spatial_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    channel_kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    spatial_kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    channel_use_bias: bool = False,
    spatial_use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/convolutional_block_attention.py:64*

#### `CLAHE`
**Module:** `layers.clahe`

Contrast Limited Adaptive Histogram Equalization (CLAHE) layer.

**Constructor Arguments:**
```python
CLAHE(
    clip_limit: float = 4.0,
    n_bins: int = 256,
    tile_size: int = 16,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_constraint: Optional[keras.constraints.Constraint] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/clahe.py:62*

#### `Canny`
**Module:** `layers.canny`

Keras implementation of the Canny edge detection algorithm.

**Constructor Arguments:**
```python
Canny(
    sigma: float = 0.8,
    threshold_min: int = 50,
    threshold_max: int = 80,
    tracking_connection: int = 5,
    tracking_iterations: int = 3,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/canny.py:9*

#### `CapsuleBlock`
**Module:** `layers.capsules`

A complete capsule block with optional dropout and normalization.

**Constructor Arguments:**
```python
CapsuleBlock(
    num_capsules: int,
    dim_capsules: int,
    routing_iterations: int = 3,
    dropout_rate: float = 0.0,
    use_layer_norm: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    squash_axis: int = -2,
    squash_epsilon: Optional[float] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/capsules.py:660*

#### `CapsuleRoutingSelfAttention`
**Module:** `layers.attention.capsule_routing_attention`

Capsule Routing Self-Attention mechanism from Capsule-Transformer.

**Constructor Arguments:**
```python
CapsuleRoutingSelfAttention(
    num_heads: int,
    key_dim: Optional[int] = None,
    value_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    activity_regularizer: Optional[regularizers.Regularizer] = None,
    routing_iterations: int = 3,
    use_vertical_routing: bool = True,
    use_horizontal_routing: bool = True,
    use_positional_routing: bool = True,
    epsilon: float = 1e-08,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/capsule_routing_attention.py:77*

#### `ChannelAttention`
**Module:** `layers.attention.channel_attention`

Channel attention module of CBAM (Convolutional Block Attention Module).

**Constructor Arguments:**
```python
ChannelAttention(
    channels: int,
    ratio: int = 8,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/channel_attention.py:69*

#### `CircuitDepthLayer`
**Module:** `layers.logic.neural_circuit`

A single depth layer of the neural circuit.

**Constructor Arguments:**
```python
CircuitDepthLayer(
    num_logic_ops: int = 2,
    num_arithmetic_ops: int = 2,
    use_residual: bool = True,
    logic_op_types: Optional[List[str]] = None,
    arithmetic_op_types: Optional[List[str]] = None,
    routing_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    combination_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/logic/neural_circuit.py:81*

#### `ClassificationHead`
**Module:** `layers.vision_heads.factory`

Classification head for image-level classification.

*Inherits from: `BaseVisionHead`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:508*

#### `ClipLayer`
**Module:** `layers.io_preparation`

Layer that clips tensor values to a specified range.

**Constructor Arguments:**
```python
ClipLayer(
    clip_min: float,
    clip_max: float,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/io_preparation.py:15*

#### `CommonNLPTaskConfigurations`
**Module:** `layers.nlp_heads.task_types`

Predefined common NLP task configurations.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:528*

#### `CommonTaskConfigurations`
**Module:** `layers.vision_heads.task_types`

Predefined common task configurations for convenience.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:730*

#### `ComplexAveragePooling2D`
**Module:** `layers.complex_layers`

Complex-valued 2D average pooling layer.

**Constructor Arguments:**
```python
ComplexAveragePooling2D(
    pool_size: Union[int, Tuple[int, int]] = (2, 2),
    strides: Optional[Union[int, Tuple[int, int]]] = None,
    padding: str = 'VALID',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/complex_layers.py:714*

#### `ComplexConv2D`
**Module:** `layers.complex_layers`

Complex-valued 2D convolution layer with improved numerical stability.

**Constructor Arguments:**
```python
ComplexConv2D(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'SAME',
    **kwargs
)
```

*Inherits from: `ComplexLayer`*

*📁 src/dl_techniques/layers/complex_layers.py:179*

#### `ComplexDense`
**Module:** `layers.complex_layers`

Complex-valued dense layer with improved initialization.

**Constructor Arguments:**
```python
ComplexDense(
    units: int,
    **kwargs
)
```

*Inherits from: `ComplexLayer`*

*📁 src/dl_techniques/layers/complex_layers.py:416*

#### `ComplexDropout`
**Module:** `layers.complex_layers`

Complex-valued dropout layer for regularization.

**Constructor Arguments:**
```python
ComplexDropout(
    rate: float,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/complex_layers.py:894*

#### `ComplexGlobalAveragePooling2D`
**Module:** `layers.complex_layers`

Complex-valued global 2D average pooling layer.

**Constructor Arguments:**
```python
ComplexGlobalAveragePooling2D(
    keepdims: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/complex_layers.py:1024*

#### `ComplexLayer`
**Module:** `layers.complex_layers`

Base class for complex-valued layers.

**Constructor Arguments:**
```python
ComplexLayer(
    epsilon: float = 1e-07,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_initializer: Optional[keras.initializers.Initializer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/complex_layers.py:65*

#### `ComplexReLU`
**Module:** `layers.complex_layers`

Complex ReLU activation function.

**Constructor Arguments:**
```python
ComplexReLU(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/complex_layers.py:605*

#### `ConditionalOutputLayer`
**Module:** `layers.conditional_output_layer`

A custom layer for conditional output selection based on ground truth values.

**Constructor Arguments:**
```python
ConditionalOutputLayer(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/conditional_output_layer.py:73*

#### `ConformalQuantileHead`
**Module:** `layers.time_series.forecasting_layers`

Output layer designed for Conformalized Quantile Regression (CQR).

**Constructor Arguments:**
```python
ConformalQuantileHead(
    forecast_length: int,
    output_dim: int,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:276*

#### `ConnectionLayer`
**Module:** `layers.geometric.fields.connection_layer`

Computes the gauge connection from field representations.

**Constructor Arguments:**
```python
ConnectionLayer(
    hidden_dim: int,
    connection_dim: Optional[int] = None,
    connection_type: ConnectionType = 'yang_mills',
    num_generators: int = 8,
    use_metric: bool = True,
    antisymmetric: bool = True,
    connection_regularization: float = 0.001,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/connection_layer.py:36*

#### `ContextualCounterFFN`
**Module:** `layers.experimental.contextual_counter_ffn`

Feed-forward network that modulates sequences through contextual counting.

**Constructor Arguments:**
```python
ContextualCounterFFN(
    output_dim: int,
    count_dim: int,
    counting_scope: Literal['global', 'causal', 'bidirectional'] = 'bidirectional',
    residual_mode: Literal['blend', 'project', 'gate_only'] = 'blend',
    activation: Union[str, callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/contextual_counter_ffn.py:14*

#### `ContextualMemoryBank`
**Module:** `layers.experimental.contextual_memory`

Contextual Memory Bank integrating KV memory, GNN, and temporal encoding.

**Constructor Arguments:**
```python
ContextualMemoryBank(
    config: Optional[MemoryBankConfig] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:693*

#### `ContinuousRoPE`
**Module:** `layers.embedding.continuous_rope_embedding`

Continuous Rotary Position Embedding for variable positions.

**Constructor Arguments:**
```python
ContinuousRoPE(
    dim: int,
    ndim: int,
    max_wavelength: float = 10000.0,
    assert_positive: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/continuous_rope_embedding.py:75*

#### `ContinuousSinCosEmbed`
**Module:** `layers.embedding.continuous_sin_cos_embedding`

Continuous coordinate embedding using sine and cosine functions.

**Constructor Arguments:**
```python
ContinuousSinCosEmbed(
    dim: int,
    ndim: int,
    max_wavelength: float = 10000.0,
    assert_positive: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py:83*

#### `ConvBlock`
**Module:** `layers.yolo12_blocks`

Standard Convolution Block with BatchNorm and SiLU activation.

**Constructor Arguments:**
```python
ConvBlock(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = 'same',
    groups: int = 1,
    activation: bool = True,
    use_bias: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_blocks.py:55*

#### `ConvBlock`
**Module:** `layers.standard_blocks`

Configurable convolutional block with normalization, activation, and optional pooling.

**Constructor Arguments:**
```python
ConvBlock(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'same',
    normalization_type: str = 'batch_norm',
    activation_type: str = 'relu',
    dropout_rate: float = 0.0,
    use_pooling: bool = False,
    pool_size: Union[int, Tuple[int, int]] = 2,
    pool_type: Literal['max', 'avg'] = 'max',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    activation_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/standard_blocks.py:157*

#### `ConvNextV1Block`
**Module:** `layers.convnext_v1_block`

Implementation of ConvNext block with modern best practices.

**Constructor Arguments:**
```python
ConvNextV1Block(
    kernel_size: Union[int, Tuple[int, int]],
    filters: int,
    activation: Union[str, keras.layers.Activation] = 'gelu',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    dropout_rate: Optional[float] = 0.0,
    spatial_dropout_rate: Optional[float] = 0.0,
    use_gamma: bool = True,
    use_softorthonormal_regularizer: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/convnext_v1_block.py:54*

#### `ConvNextV2Block`
**Module:** `layers.convnext_v2_block`

Implementation of ConvNextV2 block with modern best practices.

**Constructor Arguments:**
```python
ConvNextV2Block(
    kernel_size: Union[int, Tuple[int, int]],
    filters: int,
    activation: Union[str, keras.layers.Activation] = 'gelu',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    dropout_rate: Optional[float] = 0.0,
    spatial_dropout_rate: Optional[float] = 0.0,
    use_gamma: bool = True,
    use_softorthonormal_regularizer: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/convnext_v2_block.py:60*

#### `ConvType`
**Module:** `layers.conv2d_builder`

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/conv2d_builder.py:217*

#### `ConvolutionalStem`
**Module:** `layers.repmixer_block`

Convolutional stem for FastVLM using MobileOne blocks.

**Constructor Arguments:**
```python
ConvolutionalStem(
    out_channels: int,
    use_se: bool = False,
    activation: Union[str, callable] = 'gelu',
    kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/repmixer_block.py:408*

#### `CorrespondenceNetwork`
**Module:** `layers.geometric.point_cloud_autoencoder`

Augmented regression network to estimate point-to-GMM correspondences.

**Constructor Arguments:**
```python
CorrespondenceNetwork(
    num_gaussians: int,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:295*

#### `CosineGating`
**Module:** `layers.moe.gating`

Cosine similarity-based gating network.

*Inherits from: `BaseGating`*

*📁 src/dl_techniques/layers/moe/gating.py:265*

#### `CountingFFN`
**Module:** `layers.ffn.counting_ffn`

A Feed-Forward Network that learns to count events in a sequence.

**Constructor Arguments:**
```python
CountingFFN(
    output_dim: int,
    count_dim: int,
    counting_scope: Literal['global', 'local', 'causal'] = 'local',
    activation: Union[str, callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/counting_ffn.py:98*

#### `DMLPlus`
**Module:** `layers.norms.max_logit_norm`

DML+ implementation for separate focal and center models.

**Constructor Arguments:**
```python
DMLPlus(
    model_type: Literal['focal', 'center'],
    axis: int = -1,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:447*

#### `DecoupledMaxLogit`
**Module:** `layers.norms.max_logit_norm`

Decoupled MaxLogit (DML) normalization layer.

**Constructor Arguments:**
```python
DecoupledMaxLogit(
    constant: float = 1.0,
    axis: int = -1,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:224*

#### `DeepARCell`
**Module:** `layers.time_series.deepar_blocks`

Autoregressive recurrent cell for DeepAR.

**Constructor Arguments:**
```python
DeepARCell(
    units: int,
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:362*

#### `DeepKernelPCA`
**Module:** `layers.statistics.deep_kernel_pca`

Deep Kernel Principal Component Analysis layer for multi-level feature extraction.

**Constructor Arguments:**
```python
DeepKernelPCA(
    num_levels: int = 3,
    components_per_level: Optional[List[int]] = None,
    kernel_type: Union[str, List[str]] = 'rbf',
    kernel_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    regularization_lambda: float = 0.01,
    coupling_strength: float = 0.5,
    use_backward_coupling: bool = True,
    center_kernel: bool = True,
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    projection_regularizer: Optional[regularizers.Regularizer] = None,
    coupling_regularizer: Optional[regularizers.Regularizer] = None,
    trainable_kernels: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:61*

#### `DenormalizationLayer`
**Module:** `layers.io_preparation`

Layer that denormalizes tensor values from source range to target range.

**Constructor Arguments:**
```python
DenormalizationLayer(
    source_min: float = -0.5,
    source_max: float = 0.5,
    target_min: float = 0.0,
    target_max: float = 255.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/io_preparation.py:255*

#### `DenseBlock`
**Module:** `layers.standard_blocks`

Configurable dense block with normalization, activation, and optional dropout.

**Constructor Arguments:**
```python
DenseBlock(
    units: int,
    normalization_type: Optional[str] = 'layer_norm',
    activation_type: str = 'relu',
    dropout_rate: float = 0.0,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_constraint: Optional[keras.constraints.Constraint] = None,
    bias_constraint: Optional[keras.constraints.Constraint] = None,
    use_bias: bool = True,
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    activation_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/standard_blocks.py:403*

#### `DepthEstimationHead`
**Module:** `layers.vision_heads.factory`

Depth estimation head for predicting depth maps.

*Inherits from: `BaseVisionHead`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:389*

#### `DepthwiseSeparableBlock`
**Module:** `layers.depthwise_separable_block`

Configurable depthwise separable convolution block.

**Constructor Arguments:**
```python
DepthwiseSeparableBlock(
    filters: int,
    depthwise_kernel_size: Union[int, Tuple[int, int]] = 3,
    stride: int = 1,
    block_id: int = 0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    normalization_type: str = 'batch_norm',
    activation_type: str = 'relu',
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    activation_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/depthwise_separable_block.py:76*

#### `DetectionHead`
**Module:** `layers.vision_heads.factory`

Detection head for object detection tasks.

*Inherits from: `BaseVisionHead`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:159*

#### `DifferentiableAddressingHead`
**Module:** `layers.ntm.base_layers`

Differentiable addressing head implementing NTM-style memory addressing.

**Constructor Arguments:**
```python
DifferentiableAddressingHead(
    memory_size: int,
    content_dim: int,
    controller_dim: int | None = None,
    num_shifts: int = 3,
    use_content_addressing: bool = True,
    use_location_addressing: bool = True,
    sharpening_bias: float = 1.0,
    kernel_initializer: str | keras.initializers.Initializer = 'glorot_uniform',
    bias_initializer: str | keras.initializers.Initializer = 'zeros',
    kernel_regularizer: keras.regularizers.Regularizer | None = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ntm/base_layers.py:37*

#### `DifferentiableSelectCopy`
**Module:** `layers.ntm.base_layers`

Differentiable layer for selecting and copying values between memory positions.

**Constructor Arguments:**
```python
DifferentiableSelectCopy(
    memory_size: int,
    content_dim: int,
    controller_dim: int,
    num_read_heads: int = 1,
    num_write_heads: int = 1,
    num_shifts: int = 3,
    use_content_addressing: bool = True,
    use_location_addressing: bool = True,
    kernel_initializer: str | keras.initializers.Initializer = 'glorot_uniform',
    bias_initializer: str | keras.initializers.Initializer = 'zeros',
    kernel_regularizer: keras.regularizers.Regularizer | None = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ntm/base_layers.py:335*

#### `DifferentiableStep`
**Module:** `layers.activations.differentiable_step`

A learnable, differentiable step function, configurable for scalar or per-axis operation.

**Constructor Arguments:**
```python
DifferentiableStep(
    axis: Optional[int] = -1,
    slope_initializer: Union[str, keras.initializers.Initializer] = 'ones',
    shift_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    shift_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = keras.regularizers.L2(0.001),
    shift_constraint: Optional[Union[str, keras.constraints.Constraint]] = ValueRangeConstraint(min_value=-1, max_value=+1),
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/differentiable_step.py:66*

#### `DifferentialFFN`
**Module:** `layers.ffn.diff_ffn`

Differential Feed-Forward Network layer implementing dual-pathway processing.

**Constructor Arguments:**
```python
DifferentialFFN(
    hidden_dim: int,
    output_dim: int,
    branch_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    gate_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'sigmoid',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/diff_ffn.py:101*

#### `DifferentialMultiHeadAttention`
**Module:** `layers.attention.differential_attention`

Differential multi-head attention mechanism.

**Constructor Arguments:**
```python
DifferentialMultiHeadAttention(
    dim: int,
    num_heads: int,
    head_dim: int,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    lambda_init: float = 0.8,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/differential_attention.py:76*

#### `DualRotaryPositionEmbedding`
**Module:** `layers.embedding.dual_rotary_position_embedding`

Dual Rotary Position Embedding layer for Gemma3-style attention mechanisms.

**Constructor Arguments:**
```python
DualRotaryPositionEmbedding(
    head_dim: int,
    max_seq_len: int,
    global_theta_base: float = 1000000.0,
    local_theta_base: float = 10000.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/dual_rotary_position_embedding.py:79*

#### `DynamicConv2D`
**Module:** `layers.dynamic_conv2d`

Dynamic 2D Convolution with Attention over Convolution Kernels.

**Constructor Arguments:**
```python
DynamicConv2D(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    num_kernels: int = 4,
    temperature: float = 30.0,
    attention_reduction_ratio: int = 4,
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'valid',
    dilation_rate: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    activation: Optional[Union[str, keras.layers.Activation]] = None,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_constraint: Optional[keras.constraints.Constraint] = None,
    bias_constraint: Optional[keras.constraints.Constraint] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/dynamic_conv2d.py:71*

#### `DynamicPatcher`
**Module:** `layers.blt_blocks`

Creates dynamic patches based on entropy thresholding.

**Constructor Arguments:**
```python
DynamicPatcher(
    entropy_threshold: float = 1.5,
    max_patches: int = 512,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:539*

#### `DynamicTanh`
**Module:** `layers.norms.dynamic_tanh`

Dynamic Tanh (DyT) layer as described in "Transformers without Normalization".

**Constructor Arguments:**
```python
DynamicTanh(
    axis: Union[int, List[int]] = -1,
    alpha_init_value: float = 0.5,
    kernel_initializer: Union[str, initializers.Initializer] = 'ones',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    kernel_constraint: Optional[constraints.Constraint] = None,
    bias_constraint: Optional[constraints.Constraint] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/dynamic_tanh.py:28*

#### `EMASlopeFilter`
**Module:** `layers.time_series.ema_layer`

Computes EMA slope and generates trading signals based on slope thresholds.

**Constructor Arguments:**
```python
EMASlopeFilter(
    ema_period: int = 25,
    lookback_period: int = 25,
    upper_threshold: float = 15.0,
    lower_threshold: float = -15.0,
    output_mode: str = 'all',
    adjust_ema: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/ema_layer.py:174*

#### `EluPlusOne`
**Module:** `layers.activations.expanded_activations`

Enhanced ELU activation layer to ensure positive values.

*Inherits from: `BaseActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:551*

#### `EnhancementHead`
**Module:** `layers.vision_heads.factory`

*Inherits from: `BaseVisionHead`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:901*

#### `EntityGraphRefinement`
**Module:** `layers.graphs.entity_graph_refinement`

Entity-Graph Refinement Component for learning hierarchical relationships in embedding space.

**Constructor Arguments:**
```python
EntityGraphRefinement(
    max_entities: int,
    entity_dim: int,
    num_refinement_steps: int = 3,
    initial_density: float = 0.8,
    attention_heads: int = 8,
    dropout_rate: float = 0.1,
    refinement_activation: str = 'gelu',
    entity_activity_threshold: float = 0.1,
    use_positional_encoding: bool = True,
    max_sequence_length: int = 1000,
    regularization_weight: float = 0.01,
    activity_regularization_target: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:96*

#### `EntropyModel`
**Module:** `layers.blt_blocks`

Small causal transformer for computing next-byte entropy.

**Constructor Arguments:**
```python
EntropyModel(
    vocab_size: int = 260,
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    max_seq_len: int = 2048,
    dropout_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:347*

#### `EomtMask`
**Module:** `layers.eomt_mask`

Configurable mask prediction module for Encoder-only Mask Transformer (EoMT).

**Constructor Arguments:**
```python
EomtMask(
    num_classes: int,
    hidden_dims: Optional[List[int]] = None,
    mask_dim: int = 256,
    class_mlp_dims: Optional[List[int]] = None,
    use_class_norm: bool = False,
    use_mask_norm: bool = False,
    normalization_type: NormalizationType = 'layer_norm',
    normalization_args: Optional[Dict[str, Any]] = None,
    mlp_activation: Union[str, keras.layers.Activation] = 'relu',
    mlp_dropout_rate: float = 0.0,
    use_bias: bool = True,
    mask_temperature: float = 1.0,
    learnable_temperature: bool = False,
    class_activation: Optional[Union[str, keras.layers.Activation]] = None,
    mask_activation: Optional[Union[str, keras.layers.Activation]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    kernel_constraint: Optional[constraints.Constraint] = None,
    bias_constraint: Optional[constraints.Constraint] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/eomt_mask.py:76*

#### `EomtTransformer`
**Module:** `layers.transformers.eomt_transformer`

Configurable Encoder-only Mask Transformer layer for vision_heads segmentation.

**Constructor Arguments:**
```python
EomtTransformer(
    hidden_size: int,
    num_heads: int = 8,
    intermediate_size: Optional[int] = None,
    attention_type: str = 'multi_head',
    attention_args: Optional[Dict[str, Any]] = None,
    normalization_type: NormalizationType = 'layer_norm',
    normalization_position: Literal['pre', 'post'] = 'pre',
    attention_norm_args: Optional[Dict[str, Any]] = None,
    ffn_norm_args: Optional[Dict[str, Any]] = None,
    ffn_type: FFNType = 'mlp',
    ffn_args: Optional[Dict[str, Any]] = None,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    use_stochastic_depth: bool = False,
    stochastic_depth_rate: float = 0.1,
    activation: Union[str, keras.layers.Activation] = 'gelu',
    use_bias: bool = True,
    use_masked_attention: bool = False,
    mask_probability: float = 1.0,
    mask_annealing_steps: int = 0,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/eomt_transformer.py:73*

#### `EvidenceEncoder`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Multi-level evidence encoder for token generation support.

**Constructor Arguments:**
```python
EvidenceEncoder(
    embed_dim: int = 768,
    local_window: int = 32,
    num_heads: int = 12,
    dropout_rate: float = 0.1,
    use_external: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:22*

#### `EvidenceSupportedTokenGeneration`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Complete evidence-supported token generation system.

**Constructor Arguments:**
```python
EvidenceSupportedTokenGeneration(
    vocab_size: int,
    embed_dim: int = 768,
    max_seq_len: int = 512,
    num_heads: int = 12,
    num_evidence_levels: int = 4,
    support_dim: int = 256,
    dropout_rate: float = 0.1,
    use_external_evidence: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:618*

#### `ExogenousBlock`
**Module:** `layers.time_series.nbeatsx_blocks`

N-BEATSx Exogenous Block.

*Inherits from: `NBeatsBlock`*

*📁 src/dl_techniques/layers/time_series/nbeatsx_blocks.py:14*

#### `ExpandedActivation`
**Module:** `layers.activations.expanded_activations`

Base class for expanded gating range activation functions.

*Inherits from: `BaseActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:268*

#### `ExpertConfig`
**Module:** `layers.moe.config`

Simplified configuration for FFN expert networks in MoE models.

*📁 src/dl_techniques/layers/moe/config.py:15*

#### `ExponentialMovingAverage`
**Module:** `layers.time_series.ema_layer`

Computes Exponential Moving Average over time series data.

**Constructor Arguments:**
```python
ExponentialMovingAverage(
    period: int = 25,
    adjust: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/ema_layer.py:24*

#### `FFNExpert`
**Module:** `layers.moe.experts`

Feed-Forward Network expert for MoE layers using dl_techniques FFN factory.

*Inherits from: `BaseExpert`*

*📁 src/dl_techniques/layers/moe/experts.py:82*

#### `FFTLayer`
**Module:** `layers.fft_layers`

Applies 2D Fast Fourier Transform and outputs concatenated real/imag parts.

**Constructor Arguments:**
```python
FFTLayer(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/fft_layers.py:8*

#### `FNetEncoderBlock`
**Module:** `layers.fnet_encoder_block`

Complete FNet encoder block with Fourier mixing and feed-forward components using factory patterns.

**Constructor Arguments:**
```python
FNetEncoderBlock(
    intermediate_dim: Optional[int] = None,
    dropout_rate: float = 0.1,
    fourier_config: Optional[Dict[str, Any]] = None,
    normalization_type: NormalizationType = 'layer_norm',
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    ffn_type: FFNType = 'mlp',
    ffn_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/fnet_encoder_block.py:71*

#### `FNetFourierTransform`
**Module:** `layers.attention.fnet_fourier_transform`

FNet Fourier Transform layer that replaces self-attention with parameter-free mixing.

**Constructor Arguments:**
```python
FNetFourierTransform(
    implementation: Literal['matrix', 'fft'] = 'matrix',
    normalize_dft: bool = True,
    epsilon: float = 1e-12,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/fnet_fourier_transform.py:58*

#### `FermiDiracDecoder`
**Module:** `layers.graphs.fermi_diract_decoder`

Fermi-Dirac decoder for edge probability prediction using Euclidean distances.

**Constructor Arguments:**
```python
FermiDiracDecoder(
    r_initializer: Union[str, keras.initializers.Initializer] = None,
    t_initializer: Union[str, keras.initializers.Initializer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/fermi_diract_decoder.py:17*

#### `FiLMLayer`
**Module:** `layers.film`

Highly configurable Feature-wise Linear Modulation (FiLM) Layer.

**Constructor Arguments:**
```python
FiLMLayer(
    gamma_units: Optional[int] = None,
    beta_units: Optional[int] = None,
    gamma_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'tanh',
    beta_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'linear',
    use_bias: bool = True,
    scale_factor: float = 1.0,
    projection_dropout: float = 0.0,
    use_layer_norm: bool = False,
    gamma_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    beta_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    gamma_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    beta_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    gamma_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
    beta_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
    modulation_mode: Literal['multiplicative', 'additive', 'both'] = 'both',
    epsilon: float = 1e-08,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/film.py:77*

#### `FieldEmbedding`
**Module:** `layers.geometric.fields.field_embedding`

Field Embedding layer that maps tokens to fields with curvature.

**Constructor Arguments:**
```python
FieldEmbedding(
    vocab_size: int,
    embed_dim: int,
    curvature_dim: Optional[int] = None,
    curvature_type: CurvatureType = 'ricci',
    curvature_scale: float = 0.1,
    curvature_regularization: float = 0.01,
    embed_initializer: Union[str, initializers.Initializer] = 'uniform',
    curvature_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    embed_regularizer: Optional[regularizers.Regularizer] = None,
    curvature_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/field_embedding.py:30*

#### `FieldNormalization`
**Module:** `layers.geometric.fields.holonomic_transformer`

Field-aware normalization that respects curvature.

**Constructor Arguments:**
```python
FieldNormalization(
    epsilon: float = 1e-06,
    use_curvature_scaling: bool = True,
    center: bool = True,
    scale: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:38*

#### `ForecastabilityGate`
**Module:** `layers.time_series.forecasting_layers`

Learnable gate for weighing deep predictions versus naive forecasts.

**Constructor Arguments:**
```python
ForecastabilityGate(
    hidden_units: int = 16,
    activation: str = 'relu',
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:124*

#### `FractalBlock`
**Module:** `layers.fractal_block`

Recursive fractal block implementing the fractal expansion rule for FractalNet.

**Constructor Arguments:**
```python
FractalBlock(
    block_config: Dict[str, Any],
    depth: int = 1,
    drop_path_rate: float = 0.15,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/fractal_block.py:66*

#### `FreeTransformerLayer`
**Module:** `layers.transformers.free_transformer`

A Transformer layer extended with the Free Transformer C-VAE architecture.

**Constructor Arguments:**
```python
FreeTransformerLayer(
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    use_free_transformer: bool = False,
    num_latent_bits: int = 16,
    encoder_attention_type: AttentionType = 'multi_head',
    encoder_ffn_type: FFNType = 'swiglu',
    encoder_attention_args: Optional[Dict[str, Any]] = None,
    encoder_ffn_args: Optional[Dict[str, Any]] = None,
    encoder_normalization_type: NormalizationType = 'rms_norm',
    **kwargs
)
```

*Inherits from: `TransformerLayer`*

*📁 src/dl_techniques/layers/transformers/free_transformer.py:271*

#### `FrequencyBandRouter`
**Module:** `layers.time_series.prism_blocks`

Learnable router for computing frequency band importance weights.

**Constructor Arguments:**
```python
FrequencyBandRouter(
    hidden_dim: int = 64,
    temperature: float = 1.0,
    dropout_rate: float = 0.1,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:140*

#### `FrequencyBandStatistics`
**Module:** `layers.time_series.prism_blocks`

Computes summary statistics for frequency bands.

**Constructor Arguments:**
```python
FrequencyBandStatistics(
    epsilon: float = 1e-06,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:29*

#### `GELU`
**Module:** `layers.activations.expanded_activations`

Gaussian Error Linear Unit (GELU) activation function.

*Inherits from: `BaseActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:187*

#### `GLUFFN`
**Module:** `layers.ffn.glu_ffn`

Gated Linear Unit Feed Forward Network as described in "GLU Variants Improve Transformer".

**Constructor Arguments:**
```python
GLUFFN(
    hidden_dim: int,
    output_dim: int,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'swish',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/glu_ffn.py:85*

#### `GatedAttention`
**Module:** `layers.attention.gated_attention`

Gated Attention layer with normalization, partial RoPE, and output gating.

**Constructor Arguments:**
```python
GatedAttention(
    dim: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    max_seq_len: int = 4096,
    rope_percentage: float = 0.5,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/gated_attention.py:83*

#### `GatedDeltaNet`
**Module:** `layers.gated_delta_net`

Gated DeltaNet layer combining delta rule updates with adaptive gating mechanism. This layer is normally input length agnostic, however due to limitations of tensorflow framework we have to define a hard top limit named max_seq_len

**Constructor Arguments:**
```python
GatedDeltaNet(
    dim: int,
    num_heads: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
    conv_kernel_size: int = 4,
    dropout_rate: float = 0.0,
    activation: Union[str, Callable] = 'silu',
    normalization_type: NormalizationType = 'zero_centered_rms_norm',
    q_norm_args: Optional[Dict[str, Any]] = None,
    k_norm_args: Optional[Dict[str, Any]] = None,
    v_norm_args: Optional[Dict[str, Any]] = None,
    ffn_type: Optional[FFNType] = None,
    ffn_args: Optional[Dict[str, Any]] = None,
    intermediate_size: Optional[int] = None,
    use_bias: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/gated_delta_net.py:79*

#### `GatedMLP`
**Module:** `layers.ffn.gated_mlp`

A Gated MLP layer implementation using 1x1 convolutions.

**Constructor Arguments:**
```python
GatedMLP(
    filters: int,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    attention_activation: Literal['relu', 'gelu', 'swish', 'silu', 'linear'] = 'relu',
    output_activation: Literal['relu', 'gelu', 'swish', 'silu', 'linear'] = 'linear',
    data_format: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/gated_mlp.py:80*

#### `GatingConfig`
**Module:** `layers.moe.config`

Configuration for MoE gating networks (routers).

*📁 src/dl_techniques/layers/moe/config.py:99*

#### `GaugeInvariantAttention`
**Module:** `layers.geometric.fields.gauge_invariant_attention`

Attention mechanism that respects gauge invariance.

**Constructor Arguments:**
```python
GaugeInvariantAttention(
    hidden_dim: int,
    num_heads: int = 8,
    key_dim: Optional[int] = None,
    attention_metric: AttentionMetric = 'hybrid',
    use_curvature_gating: bool = True,
    use_parallel_transport: bool = True,
    dropout_rate: float = 0.0,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/gauge_invariant_attention.py:35*

#### `GaussianFilter`
**Module:** `layers.gaussian_filter`

Applies Gaussian blur filter to input images.

**Constructor Arguments:**
```python
GaussianFilter(
    kernel_size: Tuple[int, int] = (5, 5),
    strides: Union[Tuple[int, int], List[int]] = (1, 1),
    sigma: Union[float, Tuple[float, float]] = -1,
    padding: str = 'same',
    data_format: Optional[str] = None,
    trainable: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/gaussian_filter.py:61*

#### `GaussianLikelihoodHead`
**Module:** `layers.time_series.deepar_blocks`

Computes Gaussian likelihood parameters (mean, std) from hidden states.

**Constructor Arguments:**
```python
GaussianLikelihoodHead(
    units: int = 1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:128*

#### `GaussianPyramid`
**Module:** `layers.gaussian_pyramid`

Gaussian Pyramid layer for multi-scale image representation.

**Constructor Arguments:**
```python
GaussianPyramid(
    levels: int = 3,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: Union[float, Tuple[float, float]] = -1,
    scale_factor: int = 2,
    padding: str = 'same',
    data_format: Optional[str] = None,
    trainable: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/gaussian_pyramid.py:59*

#### `GeGLUFFN`
**Module:** `layers.ffn.geglu_ffn`

GELU Gated Linear Unit Feed-Forward Network (GeGLU).

**Constructor Arguments:**
```python
GeGLUFFN(
    hidden_dim: int,
    output_dim: int,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/geglu_ffn.py:92*

#### `GenericBlock`
**Module:** `layers.time_series.nbeats_blocks`

Generic N-BEATS block with learnable linear transformations for flexible pattern modeling.

*Inherits from: `NBeatsBlock`*

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:483*

#### `GlobalResponseNormalization`
**Module:** `layers.norms.global_response_norm`

Global Response Normalization (GRN) layer supporting 2D, 3D, and 4D inputs.

**Constructor Arguments:**
```python
GlobalResponseNormalization(
    eps: float = 1e-06,
    gamma_initializer: Union[str, keras.initializers.Initializer] = 'ones',
    beta_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/global_response_norm.py:57*

#### `GlobalSumPooling2D`
**Module:** `layers.global_sum_pool_2d`

Global sum pooling operation for 2D spatial data.

**Constructor Arguments:**
```python
GlobalSumPooling2D(
    keepdims: bool = False,
    data_format: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/global_sum_pool_2d.py:54*

#### `GlobalTransformer`
**Module:** `layers.blt_blocks`

Global Transformer for BLT that processes patch sequences.

**Constructor Arguments:**
```python
GlobalTransformer(
    global_dim: int = 768,
    num_global_layers: int = 12,
    num_heads_global: int = 12,
    max_patches: int = 512,
    dropout_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:1163*

#### `GoLU`
**Module:** `layers.activations.golu`

Gompertz Linear Unit (GoLU) activation function layer.

**Constructor Arguments:**
```python
GoLU(
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/golu.py:68*

#### `GraphMannLayer`
**Module:** `layers.experimental.graph_mann`

Graph Memory-Augmented Neural Network (GMANN) layer based on NTM principles.

**Constructor Arguments:**
```python
GraphMannLayer(
    num_memory_nodes: int,
    memory_dim: int,
    controller_units: int,
    num_read_heads: int,
    num_write_heads: int,
    controller_type: Literal['lstm', 'gru'] = 'lstm',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/graph_mann.py:6*

#### `GraphNeuralNetworkLayer`
**Module:** `layers.experimental.contextual_memory`

Complete configurable Graph Neural Network for concept relationship modeling.

**Constructor Arguments:**
```python
GraphNeuralNetworkLayer(
    concept_dim: int,
    num_layers: int = 3,
    message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gcn',
    aggregation: Literal['mean', 'max', 'attention', 'sum'] = 'attention',
    normalization: Literal['none', 'batch', 'layer', 'rms'] = 'layer',
    activation: str = 'relu',
    dropout_rate: float = 0.1,
    use_residual: bool = True,
    use_layer_norm: bool = True,
    num_attention_heads: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:259*

#### `GraphNeuralNetworkLayer`
**Module:** `layers.graphs.graph_neural_network`

Complete configurable Graph Neural Network for concept relationship modeling.

**Constructor Arguments:**
```python
GraphNeuralNetworkLayer(
    concept_dim: int,
    num_layers: int = 3,
    message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gcn',
    aggregation: Literal['mean', 'max', 'attention', 'sum', 'none'] = 'attention',
    normalization: Literal['none', 'batch', 'layer', 'rms'] = 'layer',
    activation: Union[str, Callable] = 'relu',
    dropout_rate: float = 0.1,
    use_residual: bool = True,
    num_attention_heads: int = 4,
    epsilon: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/graph_neural_network.py:97*

#### `GroupedQueryAttention`
**Module:** `layers.attention.group_query_attention`

Grouped Query Attention layer with optional rotary position embeddings.

**Constructor Arguments:**
```python
GroupedQueryAttention(
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int = 2048,
    dropout_rate: float = 0.0,
    rope_percentage: float = 1.0,
    rope_theta: float = 10000.0,
    use_bias: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/group_query_attention.py:63*

#### `HANCBlock`
**Module:** `layers.hanc_block`

Hierarchical Aggregation of Neighborhood Context (HANC) Block.

**Constructor Arguments:**
```python
HANCBlock(
    filters: int,
    input_channels: int,
    k: int = 3,
    inv_factor: int = 3,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/hanc_block.py:75*

#### `HANCLayer`
**Module:** `layers.hanc_layer`

Hierarchical Aggregation of Neighborhood Context (HANC) Layer.

**Constructor Arguments:**
```python
HANCLayer(
    in_channels: int,
    out_channels: int,
    k: int = 3,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/hanc_layer.py:62*

#### `HaarWaveletDecomposition`
**Module:** `layers.haar_wavelet_decomposition`

Performs Haar Discrete Wavelet Transform (DWT) decomposition.

**Constructor Arguments:**
```python
HaarWaveletDecomposition(
    num_levels: int = 3,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/haar_wavelet_decomposition.py:15*

#### `HardSigmoid`
**Module:** `layers.activations.hard_sigmoid`

Hard-sigmoid activation function for efficient sigmoid approximation.

**Constructor Arguments:**
```python
HardSigmoid(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/hard_sigmoid.py:54*

#### `HardSwish`
**Module:** `layers.activations.hard_swish`

Hard-swish activation function for efficient mobile-optimized neural networks.

**Constructor Arguments:**
```python
HardSwish(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/hard_swish.py:57*

#### `HeadConfiguration`
**Module:** `layers.vision_heads.factory`

Configuration helper for vision_heads heads.

*📁 src/dl_techniques/layers/vision_heads/factory.py:1037*

#### `HeadState`
**Module:** `layers.ntm.ntm_interface`

Represents the state of a read or write head.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:128*

#### `HebbianReadoutLayer`
**Module:** `layers.mothnet_blocks`

Hebbian readout layer implementing local correlation-based learning.

**Constructor Arguments:**
```python
HebbianReadoutLayer(
    units: int,
    learning_rate: float = 0.01,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/mothnet_blocks.py:571*

#### `HierarchicalEvidenceAggregator`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Hierarchical aggregator for evidence-based token generation support.

**Constructor Arguments:**
```python
HierarchicalEvidenceAggregator(
    embed_dim: int = 768,
    num_levels: int = 4,
    pooling_sizes: List[int] = [1, 4, 16, 64],
    num_heads: int = 8,
    dropout_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:257*

#### `HierarchicalMLPStem`
**Module:** `layers.hierarchical_mlp_stem`

Hierarchical MLP stem for Vision Transformers with patch-independent processing.

**Constructor Arguments:**
```python
HierarchicalMLPStem(
    embed_dim: int = 768,
    img_size: Tuple[int, int] = (224, 224),
    patch_size: Tuple[int, int] = (16, 16),
    in_channels: int = 3,
    norm_layer: Literal['batch', 'layer'] = 'batch',
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/hierarchical_mlp_stem.py:61*

#### `HierarchicalMemorySystem`
**Module:** `layers.experimental.hierarchical_memory_system`

Hierarchical memory system using multiple Self-Organizing Map layers.

**Constructor Arguments:**
```python
HierarchicalMemorySystem(
    input_dim: int,
    levels: int = 3,
    grid_dimensions: int = 2,
    base_grid_size: int = 5,
    grid_expansion_factor: float = 2.0,
    initial_learning_rate: float = 0.1,
    decay_function: Optional[Callable] = None,
    sigma: float = 1.0,
    neighborhood_function: str = 'gaussian',
    weights_initializer: Union[str, keras.initializers.Initializer] = 'he_uniform',
    regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:62*

#### `HierarchicalReasoningCore`
**Module:** `layers.reasoning.hrm_reasoning_core`

Stateful hierarchical reasoning core for complex multi-step reasoning tasks.

**Constructor Arguments:**
```python
HierarchicalReasoningCore(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_puzzle_identifiers: int,
    puzzle_emb_dim: int = 0,
    batch_size: int = 32,
    h_layers: int = 4,
    l_layers: int = 4,
    h_cycles: int = 2,
    l_cycles: int = 2,
    num_heads: int = 8,
    ffn_expansion_factor: int = 4,
    pos_encodings: Literal['rope', 'learned'] = 'rope',
    rope_theta: float = 10000.0,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    embeddings_initializer: Union[str, keras.initializers.Initializer] = 'truncated_normal',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:81*

#### `HierarchicalReasoningModule`
**Module:** `layers.reasoning.hrm_reasoning_module`

Configurable multi-layer reasoning module with input injection.

**Constructor Arguments:**
```python
HierarchicalReasoningModule(
    num_layers: int,
    embed_dim: int,
    num_heads: int = 8,
    ffn_expansion_factor: int = 4,
    attention_type: AttentionType = 'multi_head',
    normalization_type: NormalizationType = 'rms_norm',
    normalization_position: NormalizationPositionType = 'post',
    ffn_type: FFNType = 'swiglu',
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_module.py:57*

#### `HierarchicalRoutingLayer`
**Module:** `layers.activations.routing_probabilities_hierarchical`

Trainable hierarchical routing layer for probabilistic classification.

**Constructor Arguments:**
```python
HierarchicalRoutingLayer(
    output_dim: int,
    axis: int = -1,
    epsilon: float = 1e-07,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/routing_probabilities_hierarchical.py:103*

#### `HolonomicFieldProjection`
**Module:** `layers.experimental.field_embeddings`

Projects final rotation matrix state to a feature vector for downstream tasks.

**Constructor Arguments:**
```python
HolonomicFieldProjection(
    projection_type: str = 'reference',
    reference_vector: Optional[Union[List[float], np.ndarray]] = None,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:640*

#### `HolonomicPathIntegrator`
**Module:** `layers.experimental.field_embeddings`

Computes the path-ordered integral of rotation matrices along a sequence.

**Constructor Arguments:**
```python
HolonomicPathIntegrator(
    return_sequences: bool = True,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:288*

#### `HolonomicTransformerLayer`
**Module:** `layers.geometric.fields.holonomic_transformer`

Complete Holonomic Transformer Layer.

**Constructor Arguments:**
```python
HolonomicTransformerLayer(
    hidden_dim: int,
    num_heads: int = 8,
    ffn_dim: Optional[int] = None,
    curvature_type: str = 'ricci',
    connection_type: str = 'yang_mills',
    attention_metric: str = 'hybrid',
    use_holonomy_features: bool = True,
    use_anomaly_detection: bool = True,
    dropout_rate: float = 0.1,
    normalization_type: NormalizationType = 'field_norm',
    activation: str = 'gelu',
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:167*

#### `HolonomyLayer`
**Module:** `layers.geometric.fields.holonomy_layer`

Computes holonomy (path-ordered exponential around loops).

**Constructor Arguments:**
```python
HolonomyLayer(
    hidden_dim: int,
    loop_sizes: List[int] = [2, 4, 8],
    loop_type: LoopType = 'rectangular',
    num_loops: int = 4,
    use_trace: bool = True,
    holonomy_regularization: float = 0.001,
    kernel_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/holonomy_layer.py:34*

#### `HopfieldAttention`
**Module:** `layers.attention.hopfield_attention`

Modern Hopfield layer implementation as described in 'Hopfield Networks is All You Need'.

**Constructor Arguments:**
```python
HopfieldAttention(
    num_heads: int,
    key_dim: int,
    value_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    normalize_patterns: bool = True,
    update_steps_max: int = 0,
    update_steps_eps: float = 0.0001,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/hopfield_attention.py:68*

#### `IFFTLayer`
**Module:** `layers.fft_layers`

Applies 2D Inverse FFT to concatenated real/imag parts.

**Constructor Arguments:**
```python
IFFTLayer(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/fft_layers.py:126*

#### `ImageCaptioningHead`
**Module:** `layers.vlm_heads.factory`

An autoregressive decoder head for generating text conditioned on vision features.

**Constructor Arguments:**
```python
ImageCaptioningHead(
    task_config: VLMTaskConfig,
    vision_dim: int = 768,
    text_dim: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    ffn_type: FFNType = 'swiglu',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/vlm_heads/factory.py:140*

#### `ImageTextMatchingHead`
**Module:** `layers.vlm_heads.factory`

A projection head for contrastive image-text alignment and fine-grained matching.

*Inherits from: `BaseVLMHead`*

*📁 src/dl_techniques/layers/vlm_heads/factory.py:464*

#### `InstanceSegmentationHead`
**Module:** `layers.vision_heads.factory`

Instance segmentation head combining detection and segmentation.

*Inherits from: `BaseVisionHead`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:614*

#### `InvertedResidualBlock`
**Module:** `layers.inverted_residual_block`

Inverted residual block, the core building block for MobileNetV2.

*Inherits from: `UniversalInvertedBottleneck`*

*📁 src/dl_techniques/layers/inverted_residual_block.py:13*

#### `InvertibleKernelPCA`
**Module:** `layers.statistics.invertible_kernel_pca`

Invertible Kernel PCA layer using Random Fourier Features approximation.

**Constructor Arguments:**
```python
InvertibleKernelPCA(
    n_components: Optional[int] = None,
    n_random_features: int = 256,
    kernel_type: Literal['rbf', 'laplacian', 'cauchy'] = 'rbf',
    gamma: Optional[float] = None,
    center_features: bool = True,
    whiten: bool = False,
    regularization: float = 1e-06,
    random_seed: Optional[int] = None,
    trainable_frequencies: bool = False,
    use_bias: bool = True,
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:77*

#### `InvertibleKernelPCADenoiser`
**Module:** `layers.statistics.invertible_kernel_pca`

Denoising layer based on Invertible Kernel PCA.

**Constructor Arguments:**
```python
InvertibleKernelPCADenoiser(
    n_components: Union[int, float] = 0.95,
    n_random_features: int = 512,
    kernel_type: str = 'rbf',
    gamma: Optional[float] = None,
    adaptive_components: bool = False,
    noise_estimation: Literal['mad', 'std'] = 'mad',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:657*

#### `KANLinear`
**Module:** `layers.kan_linear`

Kolmogorov-Arnold Network (KAN) linear layer with learnable activation functions.

**Constructor Arguments:**
```python
KANLinear(
    features: int,
    grid_size: int = 5,
    spline_order: int = 3,
    grid_range: Tuple[float, float] = (-2.0, 2.0),
    activation: Union[str, Callable] = 'swish',
    base_trainable: bool = True,
    spline_trainable: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/kan_linear.py:56*

#### `KANvolution`
**Module:** `layers.convolutional_kan`

Kolmogorov-Arnold Network convolution layer with learnable B-spline activations.

**Constructor Arguments:**
```python
KANvolution(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    grid_size: int = 16,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: str = 'same',
    dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
    activation: Optional[Union[str, Callable]] = None,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/convolutional_kan.py:54*

#### `KMeansLayer`
**Module:** `layers.kmeans`

A differentiable K-means layer with momentum and centroid repulsion.

**Constructor Arguments:**
```python
KMeansLayer(
    n_clusters: int,
    temperature: float = 0.1,
    momentum: float = 0.9,
    centroid_lr: float = 0.1,
    repulsion_strength: float = 0.1,
    min_distance: float = 1.0,
    output_mode: OutputMode = 'assignments',
    cluster_axis: Axis = -1,
    centroid_initializer: Union[str, keras.initializers.Initializer] = 'orthonormal',
    centroid_regularizer: Optional[keras.regularizers.Regularizer] = None,
    random_seed: Optional[int] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/kmeans.py:100*

#### `KeyValueMemoryStore`
**Module:** `layers.experimental.contextual_memory`

Key-Value Memory Store for long-term associations.

**Constructor Arguments:**
```python
KeyValueMemoryStore(
    num_slots: int,
    memory_dim: int,
    key_dim: int,
    temperature: float = 1.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:124*

#### `LagrangianNeuralNetworkLayer`
**Module:** `layers.physics.lagrange_layer`

Physics-informed layer modeling system dynamics through learned Lagrangian mechanics.

**Constructor Arguments:**
```python
LagrangianNeuralNetworkLayer(
    hidden_dims: List[int],
    activation: str = 'softplus',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/physics/lagrange_layer.py:8*

#### `LaplacianFilter`
**Module:** `layers.laplacian_filter`

Laplacian filter layer that detects edges by approximating the second derivative.

**Constructor Arguments:**
```python
LaplacianFilter(
    kernel_size: Tuple[int, int] = (5, 5),
    strides: Union[Tuple[int, int], List[int]] = (1, 1),
    sigma: Optional[Union[float, Tuple[float, float]]] = 1.0,
    scale_factor: float = 1.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/laplacian_filter.py:55*

#### `LearnableArithmeticOperator`
**Module:** `layers.logic.arithmetic_operators`

A learnable arithmetic operator that can perform various arithmetic operations.

**Constructor Arguments:**
```python
LearnableArithmeticOperator(
    operation_types: Optional[List[str]] = None,
    use_temperature: bool = True,
    temperature_init: float = 1.0,
    use_scaling: bool = True,
    scaling_init: float = 1.0,
    operation_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    temperature_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
    scaling_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
    epsilon: float = 1e-07,
    power_clip_range: Tuple[float, float] = (1e-07, 10.0),
    exponent_clip_range: Tuple[float, float] = (-2.0, 2.0),
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/logic/arithmetic_operators.py:79*

#### `LearnableLogicOperator`
**Module:** `layers.logic.logic_operators`

A learnable logic operator that can perform various logical operations.

**Constructor Arguments:**
```python
LearnableLogicOperator(
    operation_types: Optional[List[str]] = None,
    use_temperature: bool = True,
    temperature_init: float = 1.0,
    operation_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    temperature_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/logic/logic_operators.py:100*

#### `LearnableMultiplier`
**Module:** `layers.layer_scale`

Layer implementing learnable element-wise multipliers for adaptive feature scaling.

**Constructor Arguments:**
```python
LearnableMultiplier(
    multiplier_type: Union[MultiplierType, str] = MultiplierType.CHANNEL,
    initializer: Union[str, keras.initializers.Initializer] = 'ones',
    regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    constraint: Optional[Union[str, keras.constraints.Constraint]] = 'non_neg',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/layer_scale.py:107*

#### `LearnableNeuralCircuit`
**Module:** `layers.logic.neural_circuit`

A learnable neural circuit with configurable depth and parallel operators.

**Constructor Arguments:**
```python
LearnableNeuralCircuit(
    circuit_depth: int = 3,
    num_logic_ops_per_depth: int = 2,
    num_arithmetic_ops_per_depth: int = 2,
    use_residual: bool = False,
    use_layer_norm: bool = False,
    logic_op_types: Optional[List[str]] = None,
    arithmetic_op_types: Optional[List[str]] = None,
    routing_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    combination_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/logic/neural_circuit.py:320*

#### `LieGroupEmbedding`
**Module:** `layers.experimental.field_embeddings`

Maps token IDs to rotation matrices in SO(n) using Lie algebra exponential map.

**Constructor Arguments:**
```python
LieGroupEmbedding(
    vocab_size: int,
    embed_dim: int,
    use_expm: bool = True,
    max_norm: Optional[float] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:41*

#### `LightweightGNNLayer`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Lightweight Graph Convolutional Network layer for structural encoding.

**Constructor Arguments:**
```python
LightweightGNNLayer(
    units: int,
    activation: Optional[Union[str, Callable]] = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:38*

#### `LinearEfficientEnsemble`
**Module:** `layers.tabm_blocks`

Efficient ensemble linear layer with separate input/output scaling.

**Constructor Arguments:**
```python
LinearEfficientEnsemble(
    units: int,
    k: int,
    use_bias: bool = True,
    ensemble_scaling_in: bool = True,
    ensemble_scaling_out: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tabm_blocks.py:139*

#### `LinearGating`
**Module:** `layers.moe.gating`

Linear gating network with optional noise and top-k selection.

*Inherits from: `BaseGating`*

*📁 src/dl_techniques/layers/moe/gating.py:77*

#### `LocalDecoder`
**Module:** `layers.blt_blocks`

Local Decoder for BLT that generates next byte predictions.

**Constructor Arguments:**
```python
LocalDecoder(
    vocab_size: int = 260,
    local_dim: int = 512,
    global_dim: int = 768,
    num_local_layers: int = 6,
    num_heads_local: int = 8,
    max_sequence_length: int = 2048,
    dropout_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:1295*

#### `LocalEncoder`
**Module:** `layers.blt_blocks`

Local Encoder for BLT that processes bytes within their patches.

**Constructor Arguments:**
```python
LocalEncoder(
    vocab_size: int = 260,
    local_dim: int = 512,
    num_local_layers: int = 6,
    num_heads_local: int = 8,
    max_sequence_length: int = 2048,
    max_patches: int = 512,
    dropout_rate: float = 0.1,
    patch_pooling_method: str = 'attention',
    global_dim: int = 768,
    cross_attention_queries: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:978*

#### `LogicFFN`
**Module:** `layers.ffn.logic_ffn`

Logic-based Feed-Forward Network using learnable soft logic operations.

**Constructor Arguments:**
```python
LogicFFN(
    output_dim: int,
    logic_dim: int,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    temperature: float = 1.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:92*

#### `LogitNorm`
**Module:** `layers.norms.logit_norm`

LogitNorm layer for classification tasks.

**Constructor Arguments:**
```python
LogitNorm(
    temperature: float = 0.04,
    axis: int = -1,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/logit_norm.py:45*

#### `MDNLayer`
**Module:** `layers.statistics.mdn_layer`

Mixture Density Network Layer with separated processing paths.

**Constructor Arguments:**
```python
MDNLayer(
    output_dimension: int,
    num_mixtures: int,
    use_bias: bool = True,
    diversity_regularizer_strength: float = 0.0,
    intermediate_units: int = 32,
    use_batch_norm: bool = True,
    intermediate_activation: str = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-05),
    bias_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-06),
    min_sigma: float = MIN_SIGMA_DEFAULT,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:59*

#### `MLFCLayer`
**Module:** `layers.multi_level_feature_compilation`

Multi Level Feature Compilation (MLFC) Layer.

**Constructor Arguments:**
```python
MLFCLayer(
    channels_list: List[int],
    num_iterations: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/multi_level_feature_compilation.py:92*

#### `MLPBlock`
**Module:** `layers.tabm_blocks`

MLP block with efficient ensemble support and enhanced configurability.

**Constructor Arguments:**
```python
MLPBlock(
    units: int,
    k: Optional[int] = None,
    activation: str = 'relu',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tabm_blocks.py:375*

#### `MLPBlock`
**Module:** `layers.ffn.mlp`

MLP block used in Transformers.

**Constructor Arguments:**
```python
MLPBlock(
    hidden_dim: int,
    output_dim: int,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/mlp.py:75*

#### `MPSLayer`
**Module:** `layers.mps_layer`

Matrix Product State inspired layer for tensor decomposition.

**Constructor Arguments:**
```python
MPSLayer(
    output_dim: int,
    bond_dim: int = 16,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/mps_layer.py:88*

#### `ManifoldStressLayer`
**Module:** `layers.geometric.fields.manifold_stress`

Computes manifold stress for anomaly and adversarial detection.

**Constructor Arguments:**
```python
ManifoldStressLayer(
    hidden_dim: int,
    stress_types: List[str] = ['curvature', 'connection', 'combined'],
    stress_threshold: float = 0.5,
    use_learnable_baseline: bool = True,
    return_components: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/manifold_stress.py:36*

#### `ManifoldStressMonitor`
**Module:** `layers.experimental.field_embeddings`

Measures geometric "stress" in semantic trajectories for anomaly detection.

**Constructor Arguments:**
```python
ManifoldStressMonitor(
    aggregation: str = 'mean',
    epsilon: float = 1e-08,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:458*

#### `MannLayer`
**Module:** `layers.memory.mann`

Memory-Augmented Neural Network (MANN) layer based on Neural Turing Machines.

**Constructor Arguments:**
```python
MannLayer(
    memory_locations: int,
    memory_dim: int,
    controller_units: int,
    num_read_heads: int,
    num_write_heads: int,
    controller_type: Literal['lstm', 'gru'] = 'lstm',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/memory/mann.py:115*

#### `MaxLogitNorm`
**Module:** `layers.norms.max_logit_norm`

Basic MaxLogit normalization layer for out-of-distribution detection.

**Constructor Arguments:**
```python
MaxLogitNorm(
    axis: int = -1,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:42*

#### `MemoryAccessType`
**Module:** `layers.ntm.ntm_interface`

Enumeration of memory access types.

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:56*

#### `MemoryBankConfig`
**Module:** `layers.experimental.contextual_memory`

Configuration for the Contextual Memory Bank.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:83*

#### `MemoryState`
**Module:** `layers.ntm.ntm_interface`

Represents the state of external memory.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:76*

#### `Mish`
**Module:** `layers.activations.mish`

Mish activation function layer.

**Constructor Arguments:**
```python
Mish(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/mish.py:123*

#### `MixedSequentialBlock`
**Module:** `layers.time_series.mixed_sequential_block`

Mixed sequential block combining LSTM and self-attention mechanisms for time series processing.

**Constructor Arguments:**
```python
MixedSequentialBlock(
    embed_dim: int,
    num_heads: int = 8,
    lstm_units: Optional[int] = None,
    ff_dim: Optional[int] = None,
    block_type: BlockType = 'mixed',
    dropout_rate: float = 0.1,
    use_layer_norm: bool = True,
    normalization_type: NormalizationType = 'rms_norm',
    attention_type: AttentionType = 'multi_head',
    ffn_type: FFNType = 'mlp',
    activation: Union[str, Callable] = 'relu',
    normalization_args: Optional[Dict[str, Any]] = None,
    attention_args: Optional[Dict[str, Any]] = None,
    ffn_args: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/mixed_sequential_block.py:81*

#### `MixtureOfExperts`
**Module:** `layers.moe.layer`

Mixture of Experts (MoE) layer for sparse neural networks using FFN experts.

**Constructor Arguments:**
```python
MixtureOfExperts(
    config: MoEConfig,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/moe/layer.py:28*

#### `MoEConfig`
**Module:** `layers.moe.config`

Complete configuration for Mixture of Experts models focused on FFN experts.

*📁 src/dl_techniques/layers/moe/config.py:171*

#### `MoEOptimizerBuilder`
**Module:** `layers.moe.integration`

Builder for creating optimizers optimized for MoE training with FFN experts.

*📁 src/dl_techniques/layers/moe/integration.py:74*

#### `MoETrainingConfig`
**Module:** `layers.moe.integration`

Training configuration specifically optimized for MoE models.

*📁 src/dl_techniques/layers/moe/integration.py:27*

#### `MobileMQA`
**Module:** `layers.attention.mobile_mqa`

Mobile Multi-Query Attention (MobileMQA) block.

*Inherits from: `GroupedQueryAttention`*

*📁 src/dl_techniques/layers/attention/mobile_mqa.py:38*

#### `MobileOneBlock`
**Module:** `layers.mobile_one_block`

MobileOne building block with structural reparameterization.

**Constructor Arguments:**
```python
MobileOneBlock(
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: str = 'same',
    use_se: bool = False,
    num_conv_branches: int = 1,
    activation: Union[str, callable] = 'gelu',
    kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/mobile_one_block.py:66*

#### `ModalityProjection`
**Module:** `layers.modality_projection`

Modality projection layer for nanoVLM.

**Constructor Arguments:**
```python
ModalityProjection(
    input_dim: int,
    output_dim: int,
    scale_factor: int = 2,
    use_gelu: bool = True,
    use_layer_norm: bool = True,
    projection_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    projection_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    projection_kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    projection_bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/modality_projection.py:74*

#### `ModernBertEmbeddings`
**Module:** `layers.embedding.modern_bert_embeddings`

Computes embeddings for ModernBERT from token and type IDs.

**Constructor Arguments:**
```python
ModernBertEmbeddings(
    vocab_size: int,
    hidden_size: int,
    type_vocab_size: int,
    initializer_range: float,
    layer_norm_eps: float,
    dropout_rate: float,
    use_bias: bool,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/modern_bert_embeddings.py:9*

#### `MonotonicityLayer`
**Module:** `layers.activations.monotonicity_layer`

Enforces monotonic (non-decreasing) constraints on predictions.

**Constructor Arguments:**
```python
MonotonicityLayer(
    method: MonotonicityMethod = 'cumulative_softplus',
    axis: int = -1,
    min_spacing: Optional[float] = None,
    max_spacing: Optional[float] = None,
    value_range: Optional[Tuple[float, float]] = None,
    clip_inputs: Optional[bool] = None,
    input_clip_range: Tuple[float, float] = (-20.0, 20.0),
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/monotonicity_layer.py:61*

#### `MovingStd`
**Module:** `layers.statistics.moving_std`

Applies a 2D moving standard deviation filter to input images for texture analysis.

**Constructor Arguments:**
```python
MovingStd(
    pool_size: Tuple[int, int] = (3, 3),
    strides: Union[Tuple[int, int], List[int]] = (1, 1),
    padding: str = 'same',
    data_format: Optional[str] = None,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/moving_std.py:36*

#### `MultiHeadAttention`
**Module:** `layers.attention.multi_head_attention`

Multi-Head Self-Attention mechanism with comprehensive masking support.

**Constructor Arguments:**
```python
MultiHeadAttention(
    dim: int,
    num_heads: int = 8,
    dropout_rate: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/multi_head_attention.py:60*

#### `MultiHeadCrossAttention`
**Module:** `layers.attention.multi_head_cross_attention`

Unified, highly configurable multi-head attention layer with advanced features.

**Constructor Arguments:**
```python
MultiHeadCrossAttention(
    dim: int,
    num_heads: int = 8,
    dropout_rate: float = 0.0,
    shared_qk_projections: bool = False,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_hierarchical_routing: bool = False,
    use_adaptive_softmax: bool = False,
    adaptive_softmax_config: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/multi_head_cross_attention.py:79*

#### `MultiHeadLatentAttention`
**Module:** `layers.attention.multi_head_latent_attention`

Multi-Head Latent Attention (MLA) as proposed in DeepSeek-V2.

**Constructor Arguments:**
```python
MultiHeadLatentAttention(
    dim: int,
    num_heads: int,
    kv_latent_dim: int,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    q_latent_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    max_seq_len: int = 4096,
    rope_theta: float = 10000.0,
    rope_percentage: float = 1.0,
    normalization_type: str = 'rms_norm',
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/multi_head_latent_attention.py:32*

#### `MultiLayerOODDetector`
**Module:** `layers.experimental.band_rms_ood`

Multi-layer OOD detector that aggregates shell distances from multiple BandRMS-OOD layers.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:617*

#### `MultiModalFusion`
**Module:** `layers.fusion.multimodal_fusion`

General-purpose configurable multi-modal fusion layer.

**Constructor Arguments:**
```python
MultiModalFusion(
    dim: int = 768,
    fusion_strategy: FusionStrategy = 'cross_attention',
    num_fusion_layers: int = 1,
    attention_config: Optional[Dict[str, Any]] = None,
    ffn_type: FFNType = 'mlp',
    ffn_config: Optional[Dict[str, Any]] = None,
    norm_type: NormalizationType = 'layer_norm',
    norm_config: Optional[Dict[str, Any]] = None,
    num_tensor_projections: int = 8,
    dropout_rate: float = 0.1,
    use_residual: bool = True,
    activation: Union[str, Callable] = 'gelu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/fusion/multimodal_fusion.py:55*

#### `MultiTaskHead`
**Module:** `layers.vision_heads.factory`

Multi-task head that combines multiple task-specific heads.

**Constructor Arguments:**
```python
MultiTaskHead(
    task_configs: Dict[str, Dict[str, Any]],
    shared_backbone_dim: int = 256,
    use_task_specific_attention: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:720*

#### `MultiTaskNLPHead`
**Module:** `layers.nlp_heads.factory`

Multi-task head that combines multiple task-specific NLP heads.

**Constructor Arguments:**
```python
MultiTaskNLPHead(
    task_configs: Dict[str, NLPTaskConfig],
    shared_input_dim: int,
    use_task_specific_projections: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1104*

#### `MultiTaskVLMHead`
**Module:** `layers.vlm_heads.factory`

Multi-task head combining multiple VLM task-specific heads.

**Constructor Arguments:**
```python
MultiTaskVLMHead(
    task_configs: Dict[str, VLMTaskConfig],
    shared_vision_dim: int = 768,
    shared_text_dim: int = 768,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/vlm_heads/factory.py:571*

#### `MultipleChoiceHead`
**Module:** `layers.nlp_heads.factory`

Head for multiple choice tasks.

*Inherits from: `BaseNLPHead`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:993*

#### `MultiplierType`
**Module:** `layers.layer_scale`

Enumeration for multiplier types.

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/layer_scale.py:56*

#### `MushroomBodyLayer`
**Module:** `layers.mothnet_blocks`

Mushroom Body layer implementing high-dimensional sparse random projection.

**Constructor Arguments:**
```python
MushroomBodyLayer(
    units: int,
    sparsity: float = 0.1,
    connection_sparsity: float = 0.1,
    activation: str = 'relu',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    trainable_projection: bool = False,
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/mothnet_blocks.py:284*

#### `NBeatsBlock`
**Module:** `layers.time_series.nbeats_blocks`

Enhanced N-BEATS block layer with performance optimizations and modern Keras 3 compliance.

**Constructor Arguments:**
```python
NBeatsBlock(
    units: int,
    thetas_dim: int,
    backcast_length: int,
    forecast_length: int,
    input_dim: int = 1,
    output_dim: int = 1,
    share_weights: bool = False,
    dropout_rate: float = 0.0,
    activation: Union[str, callable] = 'relu',
    use_bias: bool = False,
    use_normalization: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    theta_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    theta_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:77*

#### `NLPHeadConfiguration`
**Module:** `layers.nlp_heads.factory`

Configuration helper for NLP heads.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1448*

#### `NLPTaskConfig`
**Module:** `layers.nlp_heads.task_types`

Configuration for a specific NLP task.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:356*

#### `NLPTaskConfiguration`
**Module:** `layers.nlp_heads.task_types`

Configuration helper for managing task combinations in NLP multi-task models.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:425*

#### `NLPTaskType`
**Module:** `layers.nlp_heads.task_types`

Enumeration of supported NLP tasks for multi-task models.

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:15*

#### `NLinear`
**Module:** `layers.tabm_blocks`

N parallel linear layers for ensemble output with enhanced efficiency.

**Constructor Arguments:**
```python
NLinear(
    n: int,
    input_dim: int,
    output_dim: int,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tabm_blocks.py:276*

#### `NTMCell`
**Module:** `layers.ntm.baseline_ntm`

Core NTM Cell for processing a single timestep.

**Constructor Arguments:**
```python
NTMCell(
    config: NTMConfig | dict[str, Any],
    kernel_initializer: str | keras.initializers.Initializer = 'glorot_uniform',
    bias_initializer: str | keras.initializers.Initializer = 'zeros',
    kernel_regularizer: keras.regularizers.Regularizer | None = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:822*

#### `NTMConfig`
**Module:** `layers.ntm.ntm_interface`

Configuration for NTM architectures.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:200*

#### `NTMController`
**Module:** `layers.ntm.baseline_ntm`

Controller network for the Neural Turing Machine.

*Inherits from: `BaseController`*

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:637*

#### `NTMMemory`
**Module:** `layers.ntm.baseline_ntm`

Standard NTM Memory Matrix.

*Inherits from: `BaseMemory`*

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:47*

#### `NTMOutput`
**Module:** `layers.ntm.ntm_interface`

Output structure for NTM forward pass.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:167*

#### `NTMReadHead`
**Module:** `layers.ntm.baseline_ntm`

Standard NTM Read Head.

*Inherits from: `BaseHead`*

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:168*

#### `NTMWriteHead`
**Module:** `layers.ntm.baseline_ntm`

Standard NTM Write Head.

*Inherits from: `BaseHead`*

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:399*

#### `NaiveResidual`
**Module:** `layers.time_series.forecasting_layers`

Structural implementation of the Naive Benchmark Principle.

**Constructor Arguments:**
```python
NaiveResidual(
    forecast_length: int,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:27*

#### `NegativeBinomialLikelihoodHead`
**Module:** `layers.time_series.deepar_blocks`

Computes Negative Binomial likelihood parameters (mu, alpha) from hidden states.

**Constructor Arguments:**
```python
NegativeBinomialLikelihoodHead(
    units: int = 1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:242*

#### `NeuralTuringMachine`
**Module:** `layers.ntm.baseline_ntm`

Complete Neural Turing Machine Layer.

*Inherits from: `BaseNTM`*

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1193*

#### `NeuroGrid`
**Module:** `layers.neuro_grid`

NeuroGrid: Differentiable N-Dimensional Memory Lattice with Probabilistic Addressing for Transformers.

**Constructor Arguments:**
```python
NeuroGrid(
    grid_shape: Union[List[int], Tuple[int, ...]],
    latent_dim: int,
    use_bias: bool = False,
    temperature: float = 1.0,
    learnable_temperature: bool = False,
    entropy_regularizer_strength: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    grid_initializer: Union[str, keras.initializers.Initializer] = OrthogonalHypersphereInitializer(),
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    grid_regularizer: Optional[keras.regularizers.Regularizer] = SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.001),
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/neuro_grid.py:94*

#### `NonLocalAttention`
**Module:** `layers.attention.non_local_attention`

Non-local Self Attention Layer for computer vision_heads tasks.

**Constructor Arguments:**
```python
NonLocalAttention(
    attention_channels: int,
    kernel_size: Union[int, Tuple[int, int]] = (7, 7),
    use_bias: bool = False,
    normalization: Optional[Literal['batch', 'layer']] = 'batch',
    intermediate_activation: Union[str, callable] = 'relu',
    output_activation: Union[str, callable] = 'linear',
    output_channels: int = -1,
    dropout_rate: float = 0.0,
    attention_mode: Literal['gaussian', 'dot_product'] = 'gaussian',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/non_local_attention.py:65*

#### `NormalizationLayer`
**Module:** `layers.io_preparation`

Layer that normalizes tensor values from source range to target range.

**Constructor Arguments:**
```python
NormalizationLayer(
    source_min: float = 0.0,
    source_max: float = 255.0,
    target_min: float = -0.5,
    target_max: float = 0.5,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/io_preparation.py:118*

#### `NormalizingFlowLayer`
**Module:** `layers.statistics.normalizing_flow`

Conditional normalizing flow layer using stacked affine coupling transformations.

**Constructor Arguments:**
```python
NormalizingFlowLayer(
    output_dimension: int,
    num_flow_steps: int,
    context_dim: int,
    hidden_units_coupling: int = 64,
    activation: Union[str, callable] = 'relu',
    use_tanh_stabilization: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:414*

#### `OneHotEncoding`
**Module:** `layers.one_hot_encoding`

One-hot encoding layer for categorical features with enhanced efficiency.

**Constructor Arguments:**
```python
OneHotEncoding(
    cardinalities: List[int],
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/one_hot_encoding.py:54*

#### `OrthoBlock`
**Module:** `layers.orthoblock`

Structured feature learning block with orthogonal regularization and constrained scaling.

**Constructor Arguments:**
```python
OrthoBlock(
    units: int,
    activation: Optional[Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]]] = None,
    use_bias: bool = True,
    ortho_reg_factor: float = 0.01,
    kernel_initializer: Union[str, keras.initializers.Initializer] = OrthogonalHypersphereInitializer(),
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    scale_initial_value: float = 0.5,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/orthoblock.py:76*

#### `OrthoGLUFFN`
**Module:** `layers.ffn.orthoglu_ffn`

Orthogonally-Regularized Gated Linear Unit Feed-Forward Network.

**Constructor Arguments:**
```python
OrthoGLUFFN(
    hidden_dim: int,
    output_dim: int,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    ortho_reg_factor: Union[float, Tuple[float, float]] = 1.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/orthoglu_ffn.py:96*

#### `PFTBlock`
**Module:** `layers.transformers.progressive_focused_transformer`

Progressive Focused Transformer Block.

**Constructor Arguments:**
```python
PFTBlock(
    dim: int,
    num_heads: int,
    window_size: int = 8,
    shift_size: int = 0,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    attention_dropout: float = 0.0,
    projection_dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    norm_type: NormalizationType = 'layer_norm',
    norm_kwargs: Optional[Dict[str, Any]] = None,
    ffn_type: FFNType = 'mlp',
    ffn_kwargs: Optional[Dict[str, Any]] = None,
    ffn_activation: str = 'gelu',
    use_lepe: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/progressive_focused_transformer.py:72*

#### `PRISMLayer`
**Module:** `layers.time_series.prism_blocks`

Main PRISM layer combining hierarchical time-frequency decomposition.

**Constructor Arguments:**
```python
PRISMLayer(
    tree_depth: int = 2,
    overlap_ratio: float = 0.25,
    num_wavelet_levels: int = 3,
    router_hidden_dim: int = 64,
    router_temperature: float = 1.0,
    dropout_rate: float = 0.1,
    use_residual: bool = True,
    use_output_norm: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:895*

#### `PRISMNode`
**Module:** `layers.time_series.prism_blocks`

Single PRISM node combining wavelet decomposition and adaptive weighting.

**Constructor Arguments:**
```python
PRISMNode(
    num_wavelet_levels: int = 3,
    router_hidden_dim: int = 64,
    router_temperature: float = 1.0,
    dropout_rate: float = 0.1,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:318*

#### `PRISMTimeTree`
**Module:** `layers.time_series.prism_blocks`

Hierarchical time decomposition with PRISM nodes at each level.

**Constructor Arguments:**
```python
PRISMTimeTree(
    tree_depth: int = 2,
    overlap_ratio: float = 0.25,
    num_wavelet_levels: int = 3,
    router_hidden_dim: int = 64,
    router_temperature: float = 1.0,
    dropout_rate: float = 0.1,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:542*

#### `ParallelTransportLayer`
**Module:** `layers.geometric.fields.parallel_transport`

Parallel transport of vectors along paths using the gauge connection.

**Constructor Arguments:**
```python
ParallelTransportLayer(
    transport_dim: int,
    num_steps: int = 10,
    transport_method: TransportMethod = 'iterative',
    step_size: float = 0.1,
    use_adaptive_steps: bool = False,
    transport_regularization: float = 0.0,
    kernel_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/fields/parallel_transport.py:33*

#### `PatchEmbedding1D`
**Module:** `layers.embedding.patch_embedding`

Patch embedding layer for time series data.

**Constructor Arguments:**
```python
PatchEmbedding1D(
    patch_size: int,
    embed_dim: int,
    stride: Optional[int] = None,
    padding: str = 'causal',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:286*

#### `PatchEmbedding2D`
**Module:** `layers.embedding.patch_embedding`

2D Image to Patch Embedding Layer.

**Constructor Arguments:**
```python
PatchEmbedding2D(
    patch_size: Union[int, Tuple[int, int]],
    embed_dim: int,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activation: Optional[Union[str, callable]] = 'linear',
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:74*

#### `PatchMerging`
**Module:** `layers.patch_merging`

Patch merging layer for hierarchical downsampling in Swin Transformer architectures.

**Constructor Arguments:**
```python
PatchMerging(
    dim: int,
    use_bias: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/patch_merging.py:50*

#### `PatchPooling`
**Module:** `layers.blt_blocks`

Pools byte representations within patches to create patch representations.

**Constructor Arguments:**
```python
PatchPooling(
    pooling_method: str = 'attention',
    output_dim: int = 768,
    num_queries: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/blt_blocks.py:711*

#### `PerceiverAttention`
**Module:** `layers.attention.perceiver_attention`

Cross-attention mechanism from the Perceiver architecture with robust serialization.

**Constructor Arguments:**
```python
PerceiverAttention(
    dim: int,
    num_heads: int = 8,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/perceiver_attention.py:78*

#### `PerceiverTransformerLayer`
**Module:** `layers.transformers.perceiver_transformer`

Complete Perceiver transformer block with cross-attention.

**Constructor Arguments:**
```python
PerceiverTransformerLayer(
    dim: int,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    activation: Union[str, callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/perceiver_transformer.py:75*

#### `PerformerAttention`
**Module:** `layers.attention.performer_attention`

Performer attention layer with linear complexity via FAVOR+ approximation.

**Constructor Arguments:**
```python
PerformerAttention(
    dim: int,
    num_heads: int = 8,
    nb_features: int = 256,
    ortho_scaling: float = 0.0,
    causal: bool = False,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/performer_attention.py:78*

#### `PixelShuffle`
**Module:** `layers.pixel_shuffle`

Pixel shuffle operation for reducing spatial tokens in vision_heads transformers.

**Constructor Arguments:**
```python
PixelShuffle(
    scale_factor: int = 2,
    validate_spatial_dims: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/pixel_shuffle.py:59*

#### `PointCloudAutoencoder`
**Module:** `layers.geometric.point_cloud_autoencoder`

Modified DGCNN-based autoencoder for point cloud feature extraction.

**Constructor Arguments:**
```python
PointCloudAutoencoder(
    k_neighbors: int = 20,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:8*

#### `PositionEmbeddingSine2D`
**Module:** `layers.embedding.positional_embedding_sine_2d`

Generates 2D sinusoidal positional encodings for image-like feature maps.

**Constructor Arguments:**
```python
PositionEmbeddingSine2D(
    num_pos_feats: int = 64,
    temperature: float = 10000.0,
    normalize: bool = True,
    scale: float = 2 * math.pi,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py:64*

#### `PositionalEmbedding`
**Module:** `layers.embedding.positional_embedding`

Learned positional embedding layer with enhanced stability.

**Constructor Arguments:**
```python
PositionalEmbedding(
    max_seq_len: int,
    dim: int,
    dropout_rate: float = 0.0,
    pos_initializer: Union[str, keras.initializers.Initializer] = 'truncated_normal',
    scale: float = 0.02,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/positional_embedding.py:76*

#### `PowerMLPLayer`
**Module:** `layers.ffn.power_mlp_layer`

PowerMLP layer with dual-branch architecture for enhanced expressiveness.

**Constructor Arguments:**
```python
PowerMLPLayer(
    units: int,
    k: int = 3,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/power_mlp_layer.py:95*

#### `PrimaryCapsule`
**Module:** `layers.capsules`

Primary Capsule Layer implementation.

**Constructor Arguments:**
```python
PrimaryCapsule(
    num_capsules: int,
    dim_capsules: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]] = 1,
    padding: str = 'valid',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    squash_axis: int = -1,
    squash_epsilon: Optional[float] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/capsules.py:60*

#### `ProbabilityOutput`
**Module:** `layers.activations.probability_output`

Unified wrapper for probability output layers.

**Constructor Arguments:**
```python
ProbabilityOutput(
    probability_type: ProbabilityType = 'softmax',
    type_config: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/probability_output.py:55*

#### `ProgressiveFocusedAttention`
**Module:** `layers.attention.progressive_focused_attention`

Progressive Focused Attention mechanism with windowed self-attention.

**Constructor Arguments:**
```python
ProgressiveFocusedAttention(
    dim: int,
    num_heads: int,
    window_size: int = 8,
    shift_size: int = 0,
    top_k: Optional[int] = None,
    sparsity_threshold: float = 0.0,
    sparsity_mode: SparsityMode = 'none',
    qkv_bias: bool = True,
    attention_dropout: float = 0.0,
    projection_dropout: float = 0.0,
    use_lepe: bool = True,
    lepe_kernel_size: int = 3,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/progressive_focused_attention.py:48*

#### `QuantileHead`
**Module:** `layers.time_series.quantile_head_fixed_io`

Quantile prediction head for probabilistic time series forecasting.

**Constructor Arguments:**
```python
QuantileHead(
    num_quantiles: int,
    output_length: int,
    dropout_rate: float = 0.1,
    use_bias: bool = True,
    flatten_input: bool = False,
    enforce_monotonicity: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/quantile_head_fixed_io.py:45*

#### `QuantileSequenceHead`
**Module:** `layers.time_series.quantile_head_variable_io`

Sequence-wise quantile prediction head for probabilistic time series forecasting.

**Constructor Arguments:**
```python
QuantileSequenceHead(
    num_quantiles: int,
    dropout_rate: float = 0.1,
    use_bias: bool = True,
    enforce_monotonicity: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_constraint: Optional[keras.constraints.Constraint] = None,
    bias_constraint: Optional[keras.constraints.Constraint] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/quantile_head_variable_io.py:76*

#### `QuestionAnsweringHead`
**Module:** `layers.nlp_heads.factory`

Head for extractive question answering.

*Inherits from: `BaseNLPHead`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:573*

#### `RBFLayer`
**Module:** `layers.radial_basis_function`

Radial Basis Function layer with stable center repulsion mechanism.

**Constructor Arguments:**
```python
RBFLayer(
    units: int,
    gamma_init: float = 1.0,
    repulsion_strength: float = 0.1,
    min_center_distance: float = 1.0,
    center_initializer: Union[str, keras.initializers.Initializer] = 'uniform',
    center_constraint: Optional[keras.constraints.Constraint] = None,
    trainable_gamma: bool = True,
    safety_margin: float = 0.2,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    gamma_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/radial_basis_function.py:54*

#### `RELGTTokenEncoder`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Multi-element tokenization encoder for heterogeneous graph nodes.

**Constructor Arguments:**
```python
RELGTTokenEncoder(
    embedding_dim: int,
    num_node_types: int,
    max_hops: int = 2,
    gnn_pe_dim: int = 32,
    gnn_pe_layers: int = 2,
    dropout_rate: float = 0.1,
    normalization_type: str = 'layer_norm',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:260*

#### `RELGTTransformerBlock`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Hybrid local-global Transformer block for relational graph processing.

**Constructor Arguments:**
```python
RELGTTransformerBlock(
    embedding_dim: int,
    num_heads: int,
    num_global_centroids: int,
    ffn_dim: int,
    dropout_rate: float = 0.1,
    ffn_type: str = 'mlp',
    normalization_type: str = 'layer_norm',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:611*

#### `RFFKernelLayer`
**Module:** `layers.random_fourier_features`

Random Fourier Features layer for efficient kernel approximation.

**Constructor Arguments:**
```python
RFFKernelLayer(
    input_dim: int,
    output_dim: Optional[int] = None,
    n_features: int = 1000,
    gamma: float = 1.0,
    use_bias: bool = True,
    activation: Optional[Union[str, callable]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    kernel_constraint: Optional[constraints.Constraint] = None,
    bias_constraint: Optional[constraints.Constraint] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/random_fourier_features.py:82*

#### `RMSNorm`
**Module:** `layers.norms.rms_norm`

Root Mean Square Normalization layer for stabilized training in deep networks.

**Constructor Arguments:**
```python
RMSNorm(
    axis: Union[int, Tuple[int, ...]] = -1,
    epsilon: float = 1e-06,
    use_scale: bool = True,
    scale_initializer: Union[str, keras.initializers.Initializer] = 'ones',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/rms_norm.py:54*

#### `RPCAttention`
**Module:** `layers.attention.rpc_attention`

Robust Principal Components Attention layer.

**Constructor Arguments:**
```python
RPCAttention(
    dim: int,
    num_heads: int = 8,
    lambda_sparse: float = 0.1,
    max_pcp_iter: int = 10,
    svd_threshold: float = 1.0,
    qkv_bias: bool = False,
    dropout_rate: float = 0.0,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/rpc_attention.py:70*

#### `ReLUK`
**Module:** `layers.activations.relu_k`

ReLU-k activation layer implementing f(x) = max(0, x)^k.

**Constructor Arguments:**
```python
ReLUK(
    k: int = 3,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/relu_k.py:75*

#### `RepMixerBlock`
**Module:** `layers.repmixer_block`

RepMixer block for efficient feature mixing in vision_heads models.

**Constructor Arguments:**
```python
RepMixerBlock(
    dim: int,
    kernel_size: int = 3,
    expansion_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    activation: Union[str, callable] = 'gelu',
    use_layer_norm: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/repmixer_block.py:94*

#### `ResPath`
**Module:** `layers.res_path`

Residual Path layer for improving skip connections in U-Net architectures.

**Constructor Arguments:**
```python
ResPath(
    channels: int,
    num_blocks: int,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/res_path.py:86*

#### `ResidualACFLayer`
**Module:** `layers.statistics.residual_acf`

Residual Autocorrelation Function analysis and regularization layer for time series models.

**Constructor Arguments:**
```python
ResidualACFLayer(
    max_lag: int = 40,
    regularization_weight: Optional[float] = None,
    target_lags: Optional[List[int]] = None,
    acf_threshold: float = 0.1,
    use_absolute_acf: bool = True,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/residual_acf.py:50*

#### `ResidualBlock`
**Module:** `layers.ffn.residual_block`

Residual block with linear transformations and configurable activation.

**Constructor Arguments:**
```python
ResidualBlock(
    hidden_dim: int,
    output_dim: int,
    dropout_rate: float = 0.0,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/residual_block.py:73*

#### `ResidualDenseBlock`
**Module:** `layers.standard_blocks`

Dense block with residual connection and configurable normalization/activation.

**Constructor Arguments:**
```python
ResidualDenseBlock(
    units: Optional[int] = None,
    normalization_type: Optional[str] = 'layer_norm',
    activation_type: str = 'relu',
    dropout_rate: float = 0.0,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    use_bias: bool = True,
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    activation_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/standard_blocks.py:640*

#### `RestrictedBoltzmannMachine`
**Module:** `layers.restricted_boltzmann_machine`

Restricted Boltzmann Machine (RBM) layer for unsupervised feature learning.

**Constructor Arguments:**
```python
RestrictedBoltzmannMachine(
    n_hidden: int,
    learning_rate: float = 0.01,
    n_gibbs_steps: int = 1,
    visible_unit_type: str = 'binary',
    use_bias: bool = True,
    kernel_initializer: str = 'glorot_uniform',
    visible_bias_initializer: str = 'zeros',
    hidden_bias_initializer: str = 'zeros',
    kernel_regularizer: Optional[Any] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:83*

#### `RigidSimplexLayer`
**Module:** `layers.rigid_simplex_layer`

Projects inputs onto a fixed Simplex structure with learnable rotation and scaling.

**Constructor Arguments:**
```python
RigidSimplexLayer(
    units: int,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    orthogonality_penalty: float = 0.0001,
    rotation_initializer: Union[str, initializers.Initializer] = 'identity',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/rigid_simplex_layer.py:129*

#### `RingAttention`
**Module:** `layers.attention.ring_attention`

Ring Attention layer with blockwise processing for extremely long sequences.

**Constructor Arguments:**
```python
RingAttention(
    dim: int,
    num_heads: int = 8,
    block_size: int = 512,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/ring_attention.py:77*

#### `RotaryPositionEmbedding`
**Module:** `layers.embedding.rotary_position_embedding`

Rotary Position Embedding layer for transformer attention mechanisms.

**Constructor Arguments:**
```python
RotaryPositionEmbedding(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float = 10000.0,
    rope_percentage: float = 0.5,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/embedding/rotary_position_embedding.py:82*

#### `RouterLayer`
**Module:** `layers.router`

Wraps a TransformerLayer with a Dr.LLM-style dynamic routing mechanism.

**Constructor Arguments:**
```python
RouterLayer(
    transformer_layer: TransformerLayer,
    router_bottleneck_dim: int = 128,
    num_windows: int = 8,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/router.py:72*

#### `RoutingCapsule`
**Module:** `layers.capsules`

Capsule layer with dynamic routing between capsules.

**Constructor Arguments:**
```python
RoutingCapsule(
    num_capsules: int,
    dim_capsules: int,
    routing_iterations: int = 3,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    squash_axis: int = -2,
    squash_epsilon: Optional[float] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/capsules.py:324*

#### `RoutingProbabilitiesLayer`
**Module:** `layers.activations.routing_probabilities`

Non-trainable hierarchical routing layer for probabilistic classification.

**Constructor Arguments:**
```python
RoutingProbabilitiesLayer(
    output_dim: Optional[int] = None,
    axis: int = -1,
    epsilon: float = 1e-07,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/routing_probabilities.py:143*

#### `SHGCNLayer`
**Module:** `layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer`

Simplified Hyperbolic Graph Convolutional Layer.

**Constructor Arguments:**
```python
SHGCNLayer(
    units: int,
    activation: Union[str, callable] = 'relu',
    use_bias: bool = True,
    use_curvature: bool = True,
    dropout_rate: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py:39*

#### `SOM2dLayer`
**Module:** `layers.memory.som_2d_layer`

2D Self-Organizing Map (SOM) layer for competitive learning and topological data organization.

**Constructor Arguments:**
```python
SOM2dLayer(
    map_size: Tuple[int, int],
    input_dim: int,
    initial_learning_rate: float = 0.1,
    decay_function: Optional[Callable] = None,
    sigma: float = 1.0,
    neighborhood_function: str = 'gaussian',
    weights_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    regularizer: Optional[keras.regularizers.Regularizer] = None,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `SOMLayer`*

*📁 src/dl_techniques/layers/memory/som_2d_layer.py:141*

#### `SOMLayer`
**Module:** `layers.memory.som_nd_layer`

N-Dimensional Self-Organizing Map (SOM) layer implementation for Keras.

**Constructor Arguments:**
```python
SOMLayer(
    grid_shape: Tuple[int, ...],
    input_dim: int,
    initial_learning_rate: float = 0.1,
    decay_function: Optional[Callable] = None,
    sigma: float = 1.0,
    neighborhood_function: str = 'gaussian',
    weights_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
    regularizer: Optional[keras.regularizers.Regularizer] = None,
    name: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:141*

#### `Sampling`
**Module:** `layers.sampling`

Uses reparameterization trick to sample from a Normal distribution.

**Constructor Arguments:**
```python
Sampling(
    seed: Optional[int] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/sampling.py:58*

#### `SaturatedMish`
**Module:** `layers.activations.mish`

SaturatedMish activation function with continuous transition at alpha.

**Constructor Arguments:**
```python
SaturatedMish(
    alpha: float = 3.0,
    beta: float = 0.5,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/mish.py:257*

#### `ScaleEnsemble`
**Module:** `layers.tabm_blocks`

Enhanced ensemble adapter with learnable scaling weights.

**Constructor Arguments:**
```python
ScaleEnsemble(
    k: int,
    input_dim: int,
    init_distribution: Literal['normal', 'random-signs'] = 'normal',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'ones',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tabm_blocks.py:56*

#### `ScaleLayer`
**Module:** `layers.time_series.deepar_blocks`

Applies item-dependent scaling to inputs and inverse scaling to outputs.

**Constructor Arguments:**
```python
ScaleLayer(
    scale_per_sample: bool = True,
    epsilon: float = 1.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:21*

#### `SeasonalityBlock`
**Module:** `layers.time_series.nbeats_blocks`

Seasonality N-BEATS block with corrected Fourier basis functions for periodic patterns.

*Inherits from: `NBeatsBlock`*

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:812*

#### `SegmentationHead`
**Module:** `layers.vision_heads.factory`

Segmentation head for semantic segmentation tasks.

*Inherits from: `BaseVisionHead`*

*📁 src/dl_techniques/layers/vision_heads/factory.py:266*

#### `SelectiveGradientMask`
**Module:** `layers.selective_gradient_mask`

Layer that selectively stops gradients based on a binary mask.

**Constructor Arguments:**
```python
SelectiveGradientMask(
    name: Optional[str] = None,
    dtype: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/selective_gradient_mask.py:69*

#### `SequencePooling`
**Module:** `layers.sequence_pooling`

Highly configurable pooling layer for sequence data.

**Constructor Arguments:**
```python
SequencePooling(
    strategy: Union[PoolingStrategy, List[PoolingStrategy]] = 'mean',
    exclude_positions: Optional[List[int]] = None,
    aggregation_method: AggregationMethod = 'concat',
    attention_hidden_dim: int = 256,
    attention_num_heads: int = 1,
    attention_dropout: float = 0.0,
    weighted_max_seq_len: int = 512,
    top_k: int = 10,
    temperature: float = 1.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/sequence_pooling.py:420*

#### `SharedWeightsCrossAttention`
**Module:** `layers.attention.shared_weights_cross_attention`

Cross-attention between different modalities with shared weights.

**Constructor Arguments:**
```python
SharedWeightsCrossAttention(
    dim: int,
    num_heads: int = 8,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/shared_weights_cross_attention.py:82*

#### `ShearletTransform`
**Module:** `layers.shearlet_transform`

Multi-scale, multi-directional shearlet transform layer for enhanced time-frequency analysis.

**Constructor Arguments:**
```python
ShearletTransform(
    scales: int = 4,
    directions: int = 8,
    alpha: float = 0.5,
    high_freq: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/shearlet_transform.py:58*

#### `SiLU`
**Module:** `layers.activations.expanded_activations`

Sigmoid Linear Unit (SiLU) activation function.

*Inherits from: `BaseActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:230*

#### `SimpleSelectCopy`
**Module:** `layers.ntm.base_layers`

Simplified differentiable select-copy layer for learning input-output mappings.

**Constructor Arguments:**
```python
SimpleSelectCopy(
    input_size: int,
    output_size: int,
    content_dim: int,
    num_copies: int = 1,
    temperature: float = 1.0,
    use_content_query: bool = True,
    kernel_initializer: str | keras.initializers.Initializer = 'glorot_uniform',
    bias_initializer: str | keras.initializers.Initializer = 'zeros',
    kernel_regularizer: keras.regularizers.Regularizer | None = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ntm/base_layers.py:672*

#### `SingleWindowAttention`
**Module:** `layers.attention.single_window_attention`

Unified multi-head self-attention for a single window.

**Constructor Arguments:**
```python
SingleWindowAttention(
    dim: int,
    window_size: int,
    num_heads: int,
    attention_mode: str = 'linear',
    normalization: str = 'softmax',
    use_relative_position_bias: bool = True,
    qkv_bias: bool = True,
    qk_scale: Optional[float] = None,
    dropout_rate: float = 0.0,
    proj_bias: bool = True,
    kan_grid_size: int = 5,
    kan_spline_order: int = 3,
    kan_activation: str = 'swish',
    adaptive_softmax_config: Optional[Dict[str, Any]] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/single_window_attention.py:10*

#### `SoftMoEGating`
**Module:** `layers.moe.gating`

SoftMoE gating that creates soft input slots for experts.

*Inherits from: `BaseGating`*

*📁 src/dl_techniques/layers/moe/gating.py:451*

#### `SoftSOMLayer`
**Module:** `layers.memory.som_nd_soft_layer`

Differentiable Soft Self-Organizing Map layer for end-to-end training.

**Constructor Arguments:**
```python
SoftSOMLayer(
    grid_shape: Tuple[int, ...],
    input_dim: int,
    temperature: float = 1.0,
    use_per_dimension_softmax: bool = True,
    use_reconstruction_loss: bool = True,
    reconstruction_weight: float = 1.0,
    topological_weight: float = 0.1,
    sharpness_weight: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-05),
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:294*

#### `SparseAutoencoder`
**Module:** `layers.sparse_autoencoder`

Sparse Autoencoder layer with multiple sparsity enforcement variants.

**Constructor Arguments:**
```python
SparseAutoencoder(
    d_input: int,
    d_latent: int,
    variant: SAEVariant = 'topk',
    k: Optional[int] = 32,
    l1_coefficient: float = 0.001,
    l0_coefficient: float = 0.001,
    tied_weights: bool = False,
    normalize_decoder: bool = True,
    use_pre_encoder_bias: bool = True,
    aux_k: Optional[int] = 256,
    aux_coefficient: float = 1 / 32,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/sparse_autoencoder.py:42*

#### `SparsePuzzleEmbedding`
**Module:** `layers.reasoning.hrm_sparse_puzzle_embedding`

Sparse embedding layer optimized for large-scale puzzle identifier lookups with training efficiency.

**Constructor Arguments:**
```python
SparsePuzzleEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    embeddings_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/reasoning/hrm_sparse_puzzle_embedding.py:45*

#### `Sparsemax`
**Module:** `layers.activations.sparsemax`

Sparsemax activation function layer for sparse probability distributions.

**Constructor Arguments:**
```python
Sparsemax(
    axis: int = -1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/sparsemax.py:58*

#### `SpatialAttention`
**Module:** `layers.attention.spatial_attention`

Spatial attention module of CBAM (Convolutional Block Attention Module).

**Constructor Arguments:**
```python
SpatialAttention(
    kernel_size: int = 7,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/spatial_attention.py:67*

#### `SpatialLayer`
**Module:** `layers.spatial_layer`

Spatial coordinate grid generator for injecting positional information into models.

**Constructor Arguments:**
```python
SpatialLayer(
    resolution: Tuple[int, int] = (4, 4),
    resize_method: Literal['nearest', 'bilinear'] = 'nearest',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/spatial_layer.py:56*

#### `SquashLayer`
**Module:** `layers.activations.squash`

Applies squashing non-linearity to vectors (capsules).

**Constructor Arguments:**
```python
SquashLayer(
    axis: int = -1,
    epsilon: Optional[float] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/squash.py:63*

#### `SqueezeExcitation`
**Module:** `layers.squeeze_excitation`

Squeeze-and-Excitation block for channel-wise feature recalibration.

**Constructor Arguments:**
```python
SqueezeExcitation(
    reduction_ratio: float = 0.25,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu',
    use_bias: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_normal',
    kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/squeeze_excitation.py:73*

#### `StochasticDepth`
**Module:** `layers.stochastic_depth`

Implements Stochastic Depth for deep networks.

**Constructor Arguments:**
```python
StochasticDepth(
    drop_path_rate: float = 0.5,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/stochastic_depth.py:56*

#### `StochasticGradient`
**Module:** `layers.stochastic_gradient`

Implements Stochastic Gradient dropping for deep networks.

**Constructor Arguments:**
```python
StochasticGradient(
    drop_path_rate: float = 0.5,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/stochastic_gradient.py:51*

#### `StrongAugmentation`
**Module:** `layers.strong_augmentation`

Strong augmentation layer for unlabeled data.

**Constructor Arguments:**
```python
StrongAugmentation(
    cutmix_prob: float = 0.5,
    cutmix_ratio_range: Tuple[float, float] = (0.1, 0.5),
    color_jitter_strength: float = 0.2,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/strong_augmentation.py:63*

#### `StructuredAttention`
**Module:** `layers.experimental.mst_correlation_filter`

Multi-Head Attention layer regularized by a SystemicGraphFilter.

*Inherits from: `keras.layers.MultiHeadAttention`*

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:379*

#### `SupernodePooling`
**Module:** `layers.geometric.supernode_pooling`

Supernode pooling layer with message passing for point clouds.

**Constructor Arguments:**
```python
SupernodePooling(
    hidden_dim: int,
    ndim: int,
    radius: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    max_neighbors: int = 32,
    mode: str = 'relpos',
    activation: Union[str, callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/geometric/supernode_pooling.py:15*

#### `SupportEmbeddingLayer`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Support embedding layer for evidence-based token generation.

**Constructor Arguments:**
```python
SupportEmbeddingLayer(
    vocab_size: int,
    embed_dim: int = 768,
    support_dim: int = 256,
    num_reasoning_steps: int = 4,
    dropout_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:436*

#### `SwiGLUFFN`
**Module:** `layers.ffn.swiglu_ffn`

SwiGLU Feed-Forward Network with gating mechanism.

**Constructor Arguments:**
```python
SwiGLUFFN(
    output_dim: int,
    ffn_expansion_factor: int = 4,
    ffn_multiple_of: int = 256,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/swiglu_ffn.py:81*

#### `SwinConvBlock`
**Module:** `layers.transformers.swin_conv_block`

Hybrid Swin-Conv block combining transformer and convolutional paths in parallel.

**Constructor Arguments:**
```python
SwinConvBlock(
    conv_dim: int,
    trans_dim: int,
    head_dim: int = 32,
    window_size: int = 8,
    drop_path_rate: float = 0.0,
    block_type: str = 'W',
    input_resolution: Optional[int] = None,
    mlp_ratio: float = 4.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/swin_conv_block.py:27*

#### `SwinMLP`
**Module:** `layers.ffn.swin_mlp`

MLP module for Swin Transformer with configurable activation and regularization.

**Constructor Arguments:**
```python
SwinMLP(
    hidden_dim: int,
    use_bias: bool = True,
    output_dim: Optional[int] = None,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    dropout_rate: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/ffn/swin_mlp.py:77*

#### `SwinTransformerBlock`
**Module:** `layers.transformers.swin_transformer_block`

Swin Transformer Block with windowed multi-head self-attention.

**Constructor Arguments:**
```python
SwinTransformerBlock(
    dim: int,
    num_heads: int,
    window_size: int = 8,
    shift_size: int = 0,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    stochastic_depth_rate: float = 0.0,
    activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/swin_transformer_block.py:133*

#### `SystemicGraphFilter`
**Module:** `layers.experimental.mst_correlation_filter`

A principled, graph-based filter for correlation matrices.

**Constructor Arguments:**
```python
SystemicGraphFilter(
    top_k_neighbors: int = 2,
    n_propagation_steps: int = 3,
    distance_metric: str = 'sqrt',
    initial_temperature: float = 0.1,
    learnable_temperature: bool = True,
    epsilon: float = 1e-08,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:47*

#### `TabMBackbone`
**Module:** `layers.tabm_blocks`

TabM backbone MLP with ensemble support and proper layer management.

**Constructor Arguments:**
```python
TabMBackbone(
    hidden_dims: List[int],
    k: Optional[int] = None,
    activation: str = 'relu',
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tabm_blocks.py:496*

#### `TaskConfiguration`
**Module:** `layers.vision_heads.task_types`

Configuration helper for managing task combinations in multi-task models.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:498*

#### `TaskType`
**Module:** `layers.vision_heads.task_types`

Enumeration of supported computer vision_heads tasks for multi-task models.

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/vision_heads/task_types.py:8*

#### `TemporalBlock`
**Module:** `layers.time_series.temporal_convolutional_network`

A single residual block for the Temporal Convolutional Network.

**Constructor Arguments:**
```python
TemporalBlock(
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float = 0.0,
    activation: str = 'relu',
    kernel_initializer: str = 'he_normal',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:7*

#### `TemporalContextEncoder`
**Module:** `layers.experimental.contextual_memory`

Temporal Context Encoder using modern Transformer architecture.

**Constructor Arguments:**
```python
TemporalContextEncoder(
    temporal_dim: int,
    num_heads: int = 8,
    num_layers: int = 6,
    max_sequence_length: int = 128,
    dropout_rate: float = 0.1,
    normalization_type: str = 'layer_norm',
    ffn_type: str = 'mlp',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:540*

#### `TemporalConvNet`
**Module:** `layers.time_series.temporal_convolutional_network`

Temporal Convolutional Network (TCN) Encoder.

**Constructor Arguments:**
```python
TemporalConvNet(
    filters: int,
    kernel_size: int = 2,
    num_levels: int = 4,
    dropout_rate: float = 0.0,
    activation: str = 'relu',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:87*

#### `TemporalFusionLayer`
**Module:** `layers.time_series.temporal_fusion`

A layer that fuses a context-based forecast with an attention-based autoregressive forecast.

**Constructor Arguments:**
```python
TemporalFusionLayer(
    output_dim: int,
    num_lags: int,
    project_lags: bool = False,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/temporal_fusion.py:84*

#### `TensorPreprocessingLayer`
**Module:** `layers.io_preparation`

Composite preprocessing layer combining normalization and clipping operations.

**Constructor Arguments:**
```python
TensorPreprocessingLayer(
    source_min: float = 0.0,
    source_max: float = 255.0,
    target_min: float = -0.5,
    target_max: float = 0.5,
    enable_final_clipping: bool = False,
    final_clip_min: float = -1.0,
    final_clip_max: float = 1.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/io_preparation.py:389*

#### `TextClassificationHead`
**Module:** `layers.nlp_heads.factory`

Head for text classification tasks.

*Inherits from: `BaseNLPHead`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:340*

#### `TextDecoder`
**Module:** `layers.transformers.text_decoder`

General-purpose configurable text decoder built upon a stack of TransformerLayers.

**Constructor Arguments:**
```python
TextDecoder(
    vocab_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    max_seq_len: int = 512,
    embedding_type: EmbeddingType = 'learned',
    positional_type: PositionalType = 'learned',
    attention_type: AttentionType = 'multi_head',
    normalization_type: NormalizationType = 'layer_norm',
    normalization_position: NormalizationPositionType = 'post',
    ffn_type: FFNType = 'mlp',
    stochastic_depth_rate: float = 0.0,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-12,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/text_decoder.py:108*

#### `TextEncoder`
**Module:** `layers.transformers.text_encoder`

General purpose configurable text encoder using factory-based components.

**Constructor Arguments:**
```python
TextEncoder(
    vocab_size: int,
    embed_dim: int,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    max_seq_len: int = 512,
    embedding_type: EmbeddingType = 'learned',
    positional_type: PositionalType = 'learned',
    attention_type: AttentionType = 'multi_head',
    normalization_type: NormalizationType = 'layer_norm',
    normalization_position: NormalizationPositionType = 'post',
    ffn_type: FFNType = 'mlp',
    use_token_type_embedding: bool = False,
    type_vocab_size: int = 2,
    use_cls_token: bool = False,
    output_mode: PoolingStrategy = 'none',
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    embed_dropout_rate: float = 0.1,
    stochastic_depth_rate: float = 0.0,
    activation: Union[str, Callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-12,
    rope_theta: float = 10000.0,
    rope_percentage: float = 1.0,
    attention_args: Optional[Dict[str, Any]] = None,
    norm_args: Optional[Dict[str, Any]] = None,
    ffn_args: Optional[Dict[str, Any]] = None,
    embedding_args: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/text_encoder.py:114*

#### `TextGenerationHead`
**Module:** `layers.nlp_heads.factory`

Head for text generation tasks.

*Inherits from: `BaseNLPHead`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:895*

#### `TextSimilarityHead`
**Module:** `layers.nlp_heads.factory`

Head for text similarity and semantic matching tasks.

*Inherits from: `BaseNLPHead`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:693*

#### `ThreshMax`
**Module:** `layers.activations.thresh_max`

ThreshMax activation layer with learnable sparsity.

**Constructor Arguments:**
```python
ThreshMax(
    axis: int = -1,
    slope: float = 10.0,
    epsilon: float = 1e-12,
    trainable_slope: bool = False,
    slope_initializer: Union[str, initializers.Initializer] = 'ones',
    slope_regularizer: Optional[Union[str, regularizers.Regularizer]] = L2_custom(-0.0001),
    slope_constraint: Optional[Union[str, constraints.Constraint]] = ValueRangeConstraint(1.0, 50.0),
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/activations/thresh_max.py:61*

#### `TokenClassificationHead`
**Module:** `layers.nlp_heads.factory`

Head for token-level classification tasks.

*Inherits from: `BaseNLPHead`*

*📁 src/dl_techniques/layers/nlp_heads/factory.py:448*

#### `TokenEmbedding`
**Module:** `layers.tokenizers.bpe`

Token embedding layer that converts token IDs to dense vectors.

**Constructor Arguments:**
```python
TokenEmbedding(
    vocab_size: int,
    embedding_dim: int,
    mask_zero: bool = True,
    embeddings_initializer: str = 'uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/tokenizers/bpe.py:387*

#### `TransformerLayer`
**Module:** `layers.transformers.transformer`

Generic transformer layer with configurable attention, FFN, and normalization.

**Constructor Arguments:**
```python
TransformerLayer(
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    attention_type: AttentionType = 'multi_head',
    attention_args: Optional[Dict[str, Any]] = None,
    normalization_type: NormalizationType = 'layer_norm',
    normalization_position: NormalizationPositionType = 'post',
    attention_norm_args: Optional[Dict[str, Any]] = None,
    ffn_norm_args: Optional[Dict[str, Any]] = None,
    ffn_type: FFNType = 'mlp',
    ffn_args: Optional[Dict[str, Any]] = None,
    moe_config: Optional[Union[MoEConfig, Dict[str, Any]]] = None,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    use_stochastic_depth: bool = False,
    stochastic_depth_rate: float = 0.1,
    activation: Union[str, Callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    window_size: int = 8,
    n_kv_head: Optional[int] = None,
    lambda_init: float = 0.8,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/transformer.py:93*

#### `TrendBlock`
**Module:** `layers.time_series.nbeats_blocks`

Trend N-BEATS block with polynomial basis functions for modeling trending behavior.

*Inherits from: `NBeatsBlock`*

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:614*

#### `TripSE1`
**Module:** `layers.attention.tripse_attention`

TripSE1: Triplet Attention with Post-Fusion Squeeze-and-Excitation.

**Constructor Arguments:**
```python
TripSE1(
    reduction_ratio: float = 0.0625,
    kernel_size: int = 7,
    use_bias: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: Optional[Any] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/attention/tripse_attention.py:178*

#### `TripSE2`
**Module:** `layers.attention.tripse_attention`

TripSE2: Pre-Process Squeeze-and-Excitation.

**Constructor Arguments:**
```python
TripSE2(
    reduction_ratio: float = 0.0625,
    kernel_size: int = 7,
    use_bias: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: Optional[Any] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/attention/tripse_attention.py:281*

#### `TripSE3`
**Module:** `layers.attention.tripse_attention`

TripSE3: Parallel Squeeze-and-Excitation.

**Constructor Arguments:**
```python
TripSE3(
    reduction_ratio: float = 0.0625,
    kernel_size: int = 7,
    use_bias: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: Optional[Any] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/attention/tripse_attention.py:410*

#### `TripSE4`
**Module:** `layers.attention.tripse_attention`

TripSE4: Hybrid 3D Attention with Affine Fusion.

**Constructor Arguments:**
```python
TripSE4(
    reduction_ratio: float = 0.0625,
    kernel_size: int = 7,
    use_bias: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: Optional[Any] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/attention/tripse_attention.py:620*

#### `TripletAttentionBranch`
**Module:** `layers.attention.tripse_attention`

Single branch of Triplet Attention mechanism.

**Constructor Arguments:**
```python
TripletAttentionBranch(
    kernel_size: int = 7,
    permute_pattern: Tuple[int, int, int] = (0, 1, 2),
    use_bias: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: Optional[Any] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/attention/tripse_attention.py:32*

#### `TverskyProjectionLayer`
**Module:** `layers.tversky_projection`

A projection layer based on a differentiable Tversky similarity model.

**Constructor Arguments:**
```python
TverskyProjectionLayer(
    units: int,
    num_features: int,
    intersection_reduction: Literal['product', 'min', 'mean'] = 'product',
    difference_reduction: Literal['ignorematch', 'subtractmatch'] = 'subtractmatch',
    prototype_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    feature_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    contrast_initializer: Union[str, initializers.Initializer] = 'ones',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/tversky_projection.py:74*

#### `UnifiedScaler`
**Module:** `layers.statistics.scaler`

Unified normalization layer combining RevIN and StandardScaler capabilities.

**Constructor Arguments:**
```python
UnifiedScaler(
    num_features: Optional[int] = None,
    axis: Union[int, Tuple[int, ...]] = -1,
    eps: float = 1e-05,
    affine: bool = False,
    affine_weight_initializer: Union[str, keras.initializers.Initializer] = 'ones',
    affine_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    nan_replacement: float = 0.0,
    store_stats: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/statistics/scaler.py:99*

#### `UniversalInvertedBottleneck`
**Module:** `layers.universal_inverted_bottleneck`

Universal Inverted Bottleneck (UIB) - A highly configurable building block for efficient CNNs.

**Constructor Arguments:**
```python
UniversalInvertedBottleneck(
    filters: int,
    expansion_factor: int = 4,
    expanded_channels: Optional[int] = None,
    stride: int = 1,
    kernel_size: int = 3,
    use_dw1: bool = True,
    use_dw2: bool = False,
    activation_type: ActivationType = 'relu',
    activation_args: Optional[Dict[str, Any]] = None,
    normalization_type: NormalizationType = 'batch_norm',
    normalization_args: Optional[Dict[str, Any]] = None,
    dropout_rate: float = 0.0,
    use_squeeze_excitation: bool = False,
    se_ratio: float = 0.25,
    se_activation: str = 'relu',
    use_bias: bool = False,
    padding: str = 'same',
    block_type: str = 'UIB',
    kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
    depthwise_initializer: Union[str, initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    depthwise_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/universal_inverted_bottleneck.py:81*

#### `VLMTaskConfig`
**Module:** `layers.vlm_heads.task_types`

Configuration for a specific VLM task.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:205*

#### `VLMTaskConfiguration`
**Module:** `layers.vlm_heads.task_types`

Configuration helper for managing task combinations in VLM multi-task models.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:277*

#### `VLMTaskType`
**Module:** `layers.vlm_heads.task_types`

Enumeration of supported VLM tasks for multi-modal models.

*Inherits from: `Enum`*

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:17*

#### `VQAHead`
**Module:** `layers.vlm_heads.factory`

A multimodal fusion and classification head for Visual Question Answering.

**Constructor Arguments:**
```python
VQAHead(
    task_config: VLMTaskConfig,
    vision_dim: int = 768,
    text_dim: int = 768,
    hidden_dims: List[int] = [512, 256],
    pooling_strategy: str = 'attention',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/vlm_heads/factory.py:279*

#### `VectorQuantizer`
**Module:** `layers.vector_quantizer`

Vector Quantization layer for discrete latent representations.

**Constructor Arguments:**
```python
VectorQuantizer(
    num_embeddings: int,
    embedding_dim: int,
    commitment_cost: float = 0.25,
    initializer: Union[str, initializers.Initializer] = 'uniform',
    use_ema: bool = False,
    ema_decay: float = 0.99,
    epsilon: float = 1e-05,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/layers/vector_quantizer.py:9*

#### `VisionEncoder`
**Module:** `layers.transformers.vision_encoder`

General purpose configurable vision_heads encoder using factory-based components.

**Constructor Arguments:**
```python
VisionEncoder(
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    patch_embed_type: PatchEmbedType = 'linear',
    attention_type: AttentionType = 'multi_head',
    normalization_type: NormalizationType = 'layer_norm',
    normalization_position: NormalizationPositionType = 'post',
    ffn_type: FFNType = 'mlp',
    use_cls_token: bool = True,
    output_mode: PoolingStrategy = 'cls',
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    pos_dropout_rate: float = 0.0,
    stochastic_depth_rate: float = 0.0,
    activation: Union[str, Callable] = 'gelu',
    use_bias: bool = True,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[regularizers.Regularizer] = None,
    bias_regularizer: Optional[regularizers.Regularizer] = None,
    attention_args: Optional[Dict[str, Any]] = None,
    norm_args: Optional[Dict[str, Any]] = None,
    ffn_args: Optional[Dict[str, Any]] = None,
    patch_embed_args: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:99*

#### `VisualGroundingHead`
**Module:** `layers.vlm_heads.factory`

Head for visual grounding tasks.

*Inherits from: `BaseVLMHead`*

*📁 src/dl_techniques/layers/vlm_heads/factory.py:402*

#### `WeightedPooling`
**Module:** `layers.sequence_pooling`

Learnable weighted pooling with position-specific weights.

**Constructor Arguments:**
```python
WeightedPooling(
    max_seq_len: int = 512,
    dropout_rate: float = 0.0,
    temperature: float = 1.0,
    initializer: Union[str, initializers.Initializer] = 'ones',
    regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/sequence_pooling.py:283*

#### `WindowAttention`
**Module:** `layers.attention.window_attention`

Unified window-based multi-head self-attention layer.

**Constructor Arguments:**
```python
WindowAttention(
    dim: int,
    window_size: int,
    num_heads: int,
    partition_mode: Literal['grid', 'zigzag'] = 'grid',
    attention_mode: Literal['linear', 'kan_key'] = 'linear',
    normalization: Literal['softmax', 'adaptive_softmax', 'hierarchical_routing'] = 'softmax',
    use_relative_position_bias: bool = True,
    qkv_bias: bool = True,
    qk_scale: Optional[float] = None,
    dropout_rate: float = 0.0,
    proj_bias: bool = True,
    kan_grid_size: int = 5,
    kan_spline_order: int = 3,
    kan_activation: str = 'swish',
    adaptive_softmax_config: Optional[Dict[str, Any]] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/attention/window_attention.py:104*

#### `YOLOv12ClassificationHead`
**Module:** `layers.yolo12_heads`

Classification head for YOLOv12 multitask learning.

**Constructor Arguments:**
```python
YOLOv12ClassificationHead(
    num_classes: int = 1,
    hidden_dims: List[int] = [512, 256],
    pooling_types: List[str] = ['avg', 'max'],
    use_attention: bool = True,
    dropout_rate: float = 0.3,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_heads.py:827*

#### `YOLOv12DetectionHead`
**Module:** `layers.yolo12_heads`

YOLOv12 Detection Head with separate classification and regression branches.

**Constructor Arguments:**
```python
YOLOv12DetectionHead(
    num_classes: int = 80,
    reg_max: int = 16,
    bbox_channels: Optional[List[int]] = None,
    cls_channels: Optional[List[int]] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_heads.py:68*

#### `YOLOv12SegmentationHead`
**Module:** `layers.yolo12_heads`

Segmentation head for YOLOv12 multitask learning.

**Constructor Arguments:**
```python
YOLOv12SegmentationHead(
    num_classes: int = 1,
    intermediate_filters: List[int] = [128, 64, 32, 16],
    target_size: Optional[Tuple[int, int]] = None,
    use_attention: bool = True,
    dropout_rate: float = 0.1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/yolo12_heads.py:381*

#### `ZeroCenteredBandRMSNorm`
**Module:** `layers.norms.zero_centered_band_rms_norm`

Zero-Centered Root Mean Square Normalization with learnable band constraints.

**Constructor Arguments:**
```python
ZeroCenteredBandRMSNorm(
    max_band_width: float = 0.1,
    axis: Union[int, Tuple[int, ...]] = -1,
    epsilon: float = 1e-07,
    band_initializer: Union[str, initializers.Initializer] = 'zeros',
    band_regularizer: Optional[regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py:54*

#### `ZeroCenteredRMSNorm`
**Module:** `layers.norms.zero_centered_rms_norm`

Zero-Centered Root Mean Square Normalization layer for enhanced training stability.

**Constructor Arguments:**
```python
ZeroCenteredRMSNorm(
    axis: Union[int, Tuple[int, ...]] = -1,
    epsilon: float = 1e-06,
    use_scale: bool = True,
    scale_initializer: Union[str, initializers.Initializer] = 'ones',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/norms/zero_centered_rms_norm.py:57*

#### `mLSTMBlock`
**Module:** `layers.time_series.xlstm_blocks`

mLSTM residual block with pre-up-projection architecture.

**Constructor Arguments:**
```python
mLSTMBlock(
    units: int,
    expansion_factor: int = 2,
    num_heads: int = 1,
    conv_kernel_size: int = 4,
    normalization_type: str = 'layer_norm',
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1100*

#### `mLSTMCell`
**Module:** `layers.time_series.xlstm_blocks`

Matrix LSTM (mLSTM) Cell with matrix memory and covariance update rule.

**Constructor Arguments:**
```python
mLSTMCell(
    units: int,
    num_heads: int = 1,
    key_dim: Optional[int] = None,
    value_dim: Optional[int] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:469*

#### `mLSTMLayer`
**Module:** `layers.time_series.xlstm_blocks`

Matrix LSTM (mLSTM) layer for processing sequences.

**Constructor Arguments:**
```python
mLSTMLayer(
    units: int,
    num_heads: int = 1,
    key_dim: Optional[int] = None,
    value_dim: Optional[int] = None,
    return_sequences: bool = True,
    return_state: bool = False,
    go_backwards: bool = False,
    stateful: bool = False,
    unroll: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:757*

#### `sLSTMBlock`
**Module:** `layers.time_series.xlstm_blocks`

sLSTM residual block with post-normalization architecture.

**Constructor Arguments:**
```python
sLSTMBlock(
    units: int,
    ffn_type: str = 'swiglu',
    ffn_expansion_factor: int = 2,
    normalization_type: str = 'layer_norm',
    normalization_kwargs: Optional[Dict[str, Any]] = None,
    forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid',
    dropout_rate: float = 0.0,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:904*

#### `sLSTMCell`
**Module:** `layers.time_series.xlstm_blocks`

Scalar LSTM (sLSTM) Cell with exponential gating and normalizer state.

**Constructor Arguments:**
```python
sLSTMCell(
    units: int,
    forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid',
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:33*

#### `sLSTMLayer`
**Module:** `layers.time_series.xlstm_blocks`

Scalar LSTM (sLSTM) layer for processing sequences.

**Constructor Arguments:**
```python
sLSTMLayer(
    units: int,
    forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid',
    return_sequences: bool = True,
    return_state: bool = False,
    go_backwards: bool = False,
    stateful: bool = False,
    unroll: bool = False,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:294*

#### `xATLU`
**Module:** `layers.activations.expanded_activations`

Expanded ArcTan Linear Unit activation function.

*Inherits from: `ExpandedActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:361*

#### `xGELU`
**Module:** `layers.activations.expanded_activations`

Expanded Gaussian Error Linear Unit (xGELU) activation function.

*Inherits from: `ExpandedActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:415*

#### `xSiLU`
**Module:** `layers.activations.expanded_activations`

Expanded Sigmoid Linear Unit (xSiLU) activation function.

*Inherits from: `ExpandedActivation`*

*📁 src/dl_techniques/layers/activations/expanded_activations.py:469*

### Losses Classes

#### `AccuracyLoss`
**Module:** `losses.any_loss`

Loss function that optimizes accuracy.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:384*

#### `AdaptiveSigLIPLoss`
**Module:** `losses.siglip_contrastive_loss`

Adaptive SigLIP Loss with dynamic temperature scaling.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:173*

#### `AffineInvariantLoss`
**Module:** `losses.affine_invariant_loss`

Affine-invariant loss for scale-invariant depth prediction.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/affine_invariant_loss.py:67*

#### `AnyLoss`
**Module:** `losses.any_loss`

Base class for all confusion matrix-based losses in the AnyLoss framework.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/any_loss.py:220*

#### `ApproximationFunction`
**Module:** `losses.any_loss`

Approximation function for transforming sigmoid outputs to near-binary values.

**Constructor Arguments:**
```python
ApproximationFunction(
    amplifying_scale: float = 73.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/losses/any_loss.py:120*

#### `BalancedAccuracyLoss`
**Module:** `losses.any_loss`

Loss function that optimizes balanced accuracy.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:936*

#### `BrierScoreLoss`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Differentiable Brier Score loss function for model calibration.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:69*

#### `BrierScoreMetric`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Brier Score metric for monitoring calibration during training.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:375*

#### `CLIPContrastiveLoss`
**Module:** `losses.clip_contrastive_loss`

CLIP Symmetric Contrastive Loss with configurable temperature and smoothing.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/clip_contrastive_loss.py:28*

#### `CapsuleMarginLoss`
**Module:** `losses.capsule_margin_loss`

Margin loss function for Capsule Networks.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/capsule_margin_loss.py:99*

#### `ChamferLoss`
**Module:** `losses.chamfer_loss`

Computes the Chamfer distance between two point clouds.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/chamfer_loss.py:5*

#### `CharbonnierLoss`
**Module:** `losses.image_restoration_loss`

Charbonnier Loss (Robust L1 Loss).

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/image_restoration_loss.py:111*

#### `ClassificationFocalLoss`
**Module:** `losses.yolo12_multitask_loss`

Internal Focal Loss for image-level classification.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:762*

#### `ClusteringLoss`
**Module:** `losses.clustering_loss`

Custom loss function for clustering quality.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/clustering_loss.py:109*

#### `ClusteringMetrics`
**Module:** `losses.clustering_loss`

Container for clustering quality metrics.

*📁 src/dl_techniques/losses/clustering_loss.py:89*

#### `ClusteringMetricsCallback`
**Module:** `losses.clustering_loss`

Callback to monitor clustering quality metrics during training.

*Inherits from: `keras.callbacks.Callback`*

*📁 src/dl_techniques/losses/clustering_loss.py:221*

#### `CohenKappaLoss`
**Module:** `losses.any_loss`

Loss function that optimizes Cohen's Kappa statistic.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1313*

#### `CombinedCalibrationLoss`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Combined loss function using both Brier Score and Spiegelhalter's Z-test.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:269*

#### `DINOLoss`
**Module:** `losses.dino_loss`

DINO consistency loss for self-supervised learning with momentum-based center.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/dino_loss.py:69*

#### `DarkIRCompositeLoss`
**Module:** `losses.image_restoration_loss`

Composite Loss for DarkIR training.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/image_restoration_loss.py:599*

#### `DecoupledInformationLoss`
**Module:** `losses.decoupled_information_loss`

A decoupled information-theoretic loss combining cross-entropy with regularization.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/decoupled_information_loss.py:98*

#### `DiceFocalSegmentationLoss`
**Module:** `losses.yolo12_multitask_loss`

Internal combined Dice and Focal Loss for segmentation.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:578*

#### `DiceLoss`
**Module:** `losses.any_loss`

Loss function that optimizes Dice coefficient.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1511*

#### `DiceLossPerChannel`
**Module:** `losses.multi_labels_loss`

Dice Loss applied per channel for multi-label segmentation.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/multi_labels_loss.py:295*

#### `EdgeLoss`
**Module:** `losses.image_restoration_loss`

Edge Loss via Laplacian of Gaussian approximation.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/image_restoration_loss.py:264*

#### `EnhanceLoss`
**Module:** `losses.image_restoration_loss`

Deep Supervision Loss for intermediate feature outputs.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/image_restoration_loss.py:506*

#### `F1Loss`
**Module:** `losses.any_loss`

Loss function that optimizes F1 score.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:723*

#### `FBetaLoss`
**Module:** `losses.any_loss`

Loss function that optimizes F-beta score.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:811*

#### `FeatureAlignmentLoss`
**Module:** `losses.feature_alignment_loss`

Feature alignment loss for semantic prior transfer.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/feature_alignment_loss.py:76*

#### `FocalTverskyLoss`
**Module:** `losses.any_loss`

Focal Tversky Loss for hard example mining.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1731*

#### `FocalUncertaintyLoss`
**Module:** `losses.focal_uncertainty_loss`

Focal Loss with uncertainty regularization for robust classification.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/focal_uncertainty_loss.py:89*

#### `FrequencyLoss`
**Module:** `losses.image_restoration_loss`

Frequency Loss using FFT amplitude comparison.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/image_restoration_loss.py:168*

#### `GeometricMeanLoss`
**Module:** `losses.any_loss`

Loss function that optimizes the geometric mean of sensitivity and specificity.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1026*

#### `GoodhartAwareLoss`
**Module:** `losses.goodhart_loss`

An information-theoretic loss combining cross-entropy with regularization.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/goodhart_loss.py:70*

#### `HRMLoss`
**Module:** `losses.hrm_loss`

Combined loss function for Hierarchical Reasoning Model.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/hrm_loss.py:120*

#### `HuberLoss`
**Module:** `losses.huber_loss`

Huber loss for robust time series forecasting.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/huber_loss.py:58*

#### `HybridContrastiveLoss`
**Module:** `losses.siglip_contrastive_loss`

Hybrid loss combining SigLIP with score-based objectives.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:283*

#### `IoULoss`
**Module:** `losses.any_loss`

Loss function that optimizes IoU (Intersection over Union / Jaccard Index).

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1428*

#### `KoLeoLoss`
**Module:** `losses.dino_loss`

Kozachenko-Leonenko entropic regularizer for uniform distribution on unit sphere.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/dino_loss.py:453*

#### `LossConfig`
**Module:** `losses.segmentation_loss`

Configuration for loss function parameters.

*📁 src/dl_techniques/losses/segmentation_loss.py:31*

#### `MASELoss`
**Module:** `losses.mase_loss`

Mean Absolute Scaled Error (MASE) loss.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/mase_loss.py:56*

#### `MCCLoss`
**Module:** `losses.any_loss`

Loss function that optimizes Matthews Correlation Coefficient.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1209*

#### `MQLoss`
**Module:** `losses.quantile_loss`

Mean Quantile Loss for probabilistic forecasting.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/quantile_loss.py:58*

#### `NanoVLMLoss`
**Module:** `losses.nano_vlm_loss`

Autoregressive language modeling loss for vision_heads-language training.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/nano_vlm_loss.py:77*

#### `PerChannelBinaryLoss`
**Module:** `losses.multi_labels_loss`

Wrapper that applies a binary loss function independently per channel.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/multi_labels_loss.py:28*

#### `PrecisionLoss`
**Module:** `losses.any_loss`

Loss function that optimizes precision.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:470*

#### `QuantileLoss`
**Module:** `losses.quantile_loss`

Vectorized Mean Quantile Loss for probabilistic forecasting.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/quantile_loss.py:112*

#### `RecallLoss`
**Module:** `losses.any_loss`

Loss function that optimizes recall (sensitivity).

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:553*

#### `SMAPELoss`
**Module:** `losses.smape_loss`

Symmetric Mean Absolute Percentage Error (SMAPE) loss.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/smape_loss.py:56*

#### `SegmentationLosses`
**Module:** `losses.segmentation_loss`

Implementation of various segmentation loss functions.

*📁 src/dl_techniques/losses/segmentation_loss.py:58*

#### `SigLIPContrastiveLoss`
**Module:** `losses.siglip_contrastive_loss`

SigLIP Contrastive Loss Function.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:68*

#### `SparsemaxLoss`
**Module:** `losses.sparsemax_loss`

Sparsemax loss function for training with sparse probability distributions.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/sparsemax_loss.py:58*

#### `SpecificityLoss`
**Module:** `losses.any_loss`

Loss function that optimizes specificity (true negative rate).

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:636*

#### `SpiegelhalterZLoss`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Differentiable Spiegelhalter's Z-test loss function for model calibration.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:156*

#### `SpiegelhalterZMetric`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Spiegelhalter Z-statistic metric for monitoring calibration during training.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:463*

#### `StableMaxCrossEntropy`
**Module:** `losses.hrm_loss`

Stable max cross entropy loss as used in the original HRM.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/hrm_loss.py:40*

#### `TabMLoss`
**Module:** `losses.tabm_loss`

Custom loss for TabM ensemble training.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/tabm_loss.py:14*

#### `TverskyLoss`
**Module:** `losses.any_loss`

Loss function that optimizes Tversky Index.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1597*

#### `VGGLoss`
**Module:** `losses.image_restoration_loss`

Perceptual Loss using VGG19 features.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/image_restoration_loss.py:382*

#### `WassersteinDivergence`
**Module:** `losses.wasserstein_loss`

Wasserstein divergence loss for comparing distributions.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/wasserstein_loss.py:432*

#### `WassersteinGradientPenaltyLoss`
**Module:** `losses.wasserstein_loss`

Wasserstein loss with gradient penalty for improved training stability.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/wasserstein_loss.py:305*

#### `WassersteinLoss`
**Module:** `losses.wasserstein_loss`

Wasserstein loss for GANs.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/wasserstein_loss.py:211*

#### `WeightedBinaryFocalLoss`
**Module:** `losses.multi_labels_loss`

Binary Focal Loss with class weighting for imbalanced segmentation.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/multi_labels_loss.py:195*

#### `WeightedCrossEntropyWithAnyLoss`
**Module:** `losses.any_loss`

Combines weighted binary cross-entropy with any AnyLoss-based metric.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1877*

#### `WrappedLoss`
**Module:** `losses.segmentation_loss`

Wrapper class to make segmentation losses compatible with Keras.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/segmentation_loss.py:554*

#### `YOLOv12MultiTaskLoss`
**Module:** `losses.yolo12_multitask_loss`

A single, "smart" loss function for YOLOv12 multi-task models.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:838*

#### `YOLOv12ObjectDetectionLoss`
**Module:** `losses.yolo12_multitask_loss`

Internal loss for YOLOv12 object detection.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:51*

#### `YoudenJLoss`
**Module:** `losses.any_loss`

Loss function that optimizes Youden's J statistic.

*Inherits from: `AnyLoss`*

*📁 src/dl_techniques/losses/any_loss.py:1112*

#### `iBOTPatchLoss`
**Module:** `losses.dino_loss`

iBOT masked patch prediction loss for self-supervised learning.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/losses/dino_loss.py:264*

### Metrics Classes

#### `CLIPAccuracy`
**Module:** `metrics.clip_accuracy`

Accuracy metric for CLIP contrastive learning.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/clip_accuracy.py:185*

#### `CLIPRecallAtK`
**Module:** `metrics.clip_accuracy`

Recall@K metric for CLIP contrastive learning.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/clip_accuracy.py:431*

#### `CapsuleAccuracy`
**Module:** `metrics.capsule_accuracy`

Custom accuracy metric for capsule networks based on capsule lengths.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/capsule_accuracy.py:8*

#### `HRMMetrics`
**Module:** `metrics.hrm_metrics`

Metrics for Hierarchical Reasoning Model.

*📁 src/dl_techniques/metrics/hrm_metrics.py:13*

#### `MultiLabelMetrics`
**Module:** `metrics.multi_label_metrics`

Comprehensive multi-label classification metrics.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/multi_label_metrics.py:8*

#### `Perplexity`
**Module:** `metrics.perplexity_metric`

Perplexity metric for language modeling tasks.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/perplexity_metric.py:165*

#### `PsnrMetric`
**Module:** `metrics.psnr_metric`

PSNR metric that evaluates only the primary output for multi-output models.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/psnr_metric.py:7*

#### `SMAPE`
**Module:** `metrics.time_series_metrics`

Symmetric Mean Absolute Percentage Error metric.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/metrics/time_series_metrics.py:13*

### Models Classes

#### `AccUNet`
**Module:** `models.accunet.model`

ACC-UNet: A Completely Convolutional UNet model for the 2020s.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/accunet/model.py:68*

#### `AdaptiveDivergenceStrategy`
**Module:** `models.ccnets.control`

An adaptive strategy that throttles the Reasoner if it converges significantly faster than the other modules. This version is graph-compatible, using tf.Variables for state.

*Inherits from: `ConvergenceControlStrategy`*

*📁 src/dl_techniques/models/ccnets/control.py:45*

#### `AdaptiveEMASlopeFilterModel`
**Module:** `models.adaptive_ema.model`

Model wrapper for adaptive EMA slope filtering with learnable thresholds.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/adaptive_ema/model.py:31*

#### `AttentionBlock`
**Module:** `models.fastvlm.components`

Attention block with vision_heads-specific adaptations.

**Constructor Arguments:**
```python
AttentionBlock(
    dim: int,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    attention_type: str = 'multi_head_attention',
    normalization_position: str = 'pre',
    dropout_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init: float = 0.0001,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/fastvlm/components.py:15*

#### `AttentionPoolingLayer`
**Module:** `models.fftnet.components`

Two-layer attention pooling for creating global descriptors with learned weights.

**Constructor Arguments:**
```python
AttentionPoolingLayer(
    hidden_dim: int = 256,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:98*

#### `AudioMaskingStrategy`
**Module:** `models.jepa.utilities`

Masking strategy adapted for audio spectrograms (A-JEPA).

*📁 src/dl_techniques/models/jepa/utilities.py:445*

#### `BERT`
**Module:** `models.bert.bert`

BERT (Bidirectional Encoder Representations from Transformers) model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/bert/bert.py:78*

#### `ByteLatentTransformer`
**Module:** `models.byte_latent_transformer.model`

Complete Byte Latent Transformer model with hierarchical processing architecture.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:56*

#### `ByteTokenizer`
**Module:** `models.modern_bert.components`

A simple, stateless byte-level tokenizer for text processing.

**Constructor Arguments:**
```python
ByteTokenizer(
    vocab_size: int = 260,
    byte_offset: int = 4,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/modern_bert/components.py:10*

#### `CBAMNet`
**Module:** `models.cbam.model`

CNN model with CBAM attention for image classification.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/cbam/model.py:49*

#### `CCNetConfig`
**Module:** `models.ccnets.base`

Configuration for CCNet framework.

*📁 src/dl_techniques/models/ccnets/base.py:32*

#### `CCNetLosses`
**Module:** `models.ccnets.base`

Container for the three fundamental CCNet losses.

*📁 src/dl_techniques/models/ccnets/base.py:82*

#### `CCNetModelErrors`
**Module:** `models.ccnets.base`

Container for the three model-specific error signals.

*📁 src/dl_techniques/models/ccnets/base.py:107*

#### `CCNetModule`
**Module:** `models.ccnets.base`

Protocol defining the interface for CCNet modules. Any model that implements this protocol can be used in the framework.

*Inherits from: `Protocol`*

*📁 src/dl_techniques/models/ccnets/base.py:9*

#### `CCNetOrchestrator`
**Module:** `models.ccnets.orchestrators`

Main orchestrator for Causal Cooperative Networks. Manages the three neural modules and their cooperative training.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:17*

#### `CCNetTrainer`
**Module:** `models.ccnets.trainer`

High-level trainer for CCNet models with built-in callbacks, dynamic weighting, KL annealing, and advanced monitoring.

*📁 src/dl_techniques/models/ccnets/trainer.py:16*

#### `CLIP`
**Module:** `models.clip.model`

CLIP model with integrated vision and text encoders.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/clip/model.py:70*

#### `CapsNet`
**Module:** `models.capsnet.model`

Keras-compliant Capsule Network model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/capsnet/model.py:38*

#### `CausalLanguageModel`
**Module:** `models.masked_language_model.clm`

A model-agnostic Causal Language Modeling (CLM) pre-trainer.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/masked_language_model/clm.py:21*

#### `CoShNet`
**Module:** `models.coshnet.model`

Refined Complex Shearlet Network (CoShNet) implementation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/coshnet/model.py:78*

#### `ComplexConv1DLayer`
**Module:** `models.fftnet.components`

One-dimensional complex convolution with circular padding.

**Constructor Arguments:**
```python
ComplexConv1DLayer(
    kernel_size: int,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:438*

#### `ComplexInterpolationLayer`
**Module:** `models.fftnet.components`

Complex tensor interpolation along the last dimension using bicubic interpolation.

**Constructor Arguments:**
```python
ComplexInterpolationLayer(
    size: int,
    mode: str = 'cubic',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:345*

#### `ComplexModReLULayer`
**Module:** `models.fftnet.components`

Complex modReLU activation with learnable bias.

**Constructor Arguments:**
```python
ComplexModReLULayer(
    num_features: int,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:258*

#### `ConditionalDenoiser`
**Module:** `models.nano_vlm_world_model.denoisers`

Conditional denoiser network that learns score functions.

**Constructor Arguments:**
```python
ConditionalDenoiser(
    data_dim: int,
    condition_dim: int,
    hidden_dim: int = 512,
    num_layers: int = 6,
    dropout_rate: float = 0.1,
    use_self_attention: bool = True,
    num_attention_heads: int = 8,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:98*

#### `ConvDecoder`
**Module:** `models.masked_autoencoder.conv_decoder`

Convolutional decoder for MAE reconstruction.

**Constructor Arguments:**
```python
ConvDecoder(
    decoder_dims: List[int] = [512, 256, 128, 64],
    output_channels: int = 3,
    kernel_size: int = 3,
    activation: str = 'gelu',
    use_batch_norm: bool = True,
    final_activation: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/masked_autoencoder/conv_decoder.py:22*

#### `ConvNeXtV1`
**Module:** `models.convnext.convnext_v1`

ConvNeXt V1 model implementation with pretrained support

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/convnext/convnext_v1.py:54*

#### `ConvNeXtV2`
**Module:** `models.convnext.convnext_v2`

ConvNeXt V2 model implementation with pretrained support.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/convnext/convnext_v2.py:57*

#### `ConvUNextModel`
**Module:** `models.convunext.model`

ConvUNext Model: Modern U-Net with ConvNeXt-inspired blocks and deep supervision.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/convunext/model.py:248*

#### `ConvUNextStem`
**Module:** `models.bias_free_denoisers.bfconvunext`

ConvUNext stem block for initial feature extraction using bias-free design.

**Constructor Arguments:**
```python
ConvUNextStem(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 7,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:58*

#### `ConvUNextStem`
**Module:** `models.convunext.model`

ConvUNext stem block for initial feature extraction.

**Constructor Arguments:**
```python
ConvUNextStem(
    filters: int,
    kernel_size: Union[int, Tuple[int, int]] = 7,
    use_bias: bool = True,
    kernel_initializer: str = 'he_normal',
    kernel_regularizer: Optional[str] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/convunext/model.py:39*

#### `ConvergenceControlStrategy`
**Module:** `models.ccnets.control`

Abstract base class for Reasoner training control strategies.

*Inherits from: `ABC`*

*📁 src/dl_techniques/models/ccnets/control.py:5*

#### `DCTPoolingLayer`
**Module:** `models.fftnet.components`

DCT-based pooling for gate descriptor creation.

**Constructor Arguments:**
```python
DCTPoolingLayer(
    dct_components: int = 64,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:180*

#### `DETR`
**Module:** `models.detr.model`

The complete DETR model for end-to-end object detection.

*Inherits from: `models.Model`*

*📁 src/dl_techniques/models/detr/model.py:472*

#### `DINOHead`
**Module:** `models.dino.dino_v1`

DINO projection head for self-supervised learning.

**Constructor Arguments:**
```python
DINOHead(
    in_dim: int,
    out_dim: int,
    use_bn: bool = False,
    norm_last_layer: bool = True,
    nlayers: int = 3,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
    normalization_type: str = 'batch_norm',
    activation: str = 'gelu',
    dropout_rate: float = 0.0,
    kernel_initializer: str = 'truncated_normal',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/dino/dino_v1.py:89*

#### `DINOv1`
**Module:** `models.dino.dino_v1`

DINO Vision Transformer model for self-supervised learning.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/dino/dino_v1.py:333*

#### `DINOv2`
**Module:** `models.dino.dino_v2`

Complete DINOv2 Model with classification head following modern Keras 3 patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/dino/dino_v2.py:913*

#### `DINOv2Block`
**Module:** `models.dino.dino_v2`

DINOv2 Transformer Block with LearnableMultiplier scaling and configurable components.

**Constructor Arguments:**
```python
DINOv2Block(
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    attention_type: str = 'multi_head_attention',
    ffn_type: str = 'mlp',
    normalization_type: str = 'layer_norm',
    qkv_bias: bool = True,
    proj_bias: bool = True,
    ffn_bias: bool = True,
    stochastic_depth_rate: float = 0.0,
    init_values: Optional[float] = None,
    attention_dropout: float = 0.0,
    ffn_dropout: float = 0.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/dino/dino_v2.py:68*

#### `DINOv2VisionTransformer`
**Module:** `models.dino.dino_v2`

DINOv2 Vision Transformer backbone implementation following Modern Keras 3 patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/dino/dino_v2.py:342*

#### `DINOv3`
**Module:** `models.dino.dino_v3`

DINOv3 Vision Transformer Model Implementation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/dino/dino_v3.py:66*

#### `DPTDecoder`
**Module:** `models.depth_anything.components`

DPT (Dense Prediction Transformer) decoder.

**Constructor Arguments:**
```python
DPTDecoder(
    dims: Optional[List[int]] = None,
    output_channels: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    use_bias: bool = False,
    activation: Union[str, callable] = 'relu',
    output_activation: Union[str, callable] = 'sigmoid',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/depth_anything/components.py:51*

#### `DarkIRDecoderBlock`
**Module:** `models.darkir.model`

Decoder Block (DBlock) for DarkIR with dual SimpleGate and FFN structure.

**Constructor Arguments:**
```python
DarkIRDecoderBlock(
    channels: int,
    dw_expand: int = 2,
    ffn_expand: int = 2,
    dilations: List[int] = None,
    extra_depth_wise: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/darkir/model.py:908*

#### `DarkIREncoderBlock`
**Module:** `models.darkir.model`

Encoder Block (EBlock) for DarkIR with parallel dilated branches and FreMLP.

**Constructor Arguments:**
```python
DarkIREncoderBlock(
    channels: int,
    dw_expand: int = 2,
    dilations: List[int] = None,
    extra_depth_wise: bool = False,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/darkir/model.py:562*

#### `DeepAR`
**Module:** `models.deepar.model`

DeepAR: Probabilistic forecasting with autoregressive recurrent networks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/deepar/model.py:34*

#### `DenoisingScoreMatchingLoss`
**Module:** `models.nano_vlm_world_model.train`

Denoising Score Matching loss for learning score functions.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:18*

#### `DenseConditioningInjection`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Inject dense conditioning features into target features.

**Constructor Arguments:**
```python
DenseConditioningInjection(
    method: str = 'film',
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    name: str = 'dense_injection',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:111*

#### `DepthAnything`
**Module:** `models.depth_anything.model`

Depth Anything model implementation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/depth_anything/model.py:52*

#### `DetrDecoderLayer`
**Module:** `models.detr.model`

A single DETR Transformer Decoder Layer with pre-normalization.

**Constructor Arguments:**
```python
DetrDecoderLayer(
    hidden_dim: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float = 0.1,
    activation: str = 'relu',
    normalization_type: str = 'layer_norm',
    ffn_type: str = 'mlp',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/detr/model.py:260*

#### `DetrTransformer`
**Module:** `models.detr.model`

DETR Transformer combining encoder and decoder stacks.

**Constructor Arguments:**
```python
DetrTransformer(
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    ffn_dim: int = 2048,
    dropout: float = 0.1,
    activation: str = 'relu',
    normalization_type: NormalizationType = 'layer_norm',
    ffn_type: FFNType = 'mlp',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/detr/model.py:69*

#### `DiffusionScheduler`
**Module:** `models.nano_vlm_world_model.scheduler`

Base diffusion scheduler for score-based models.

**Constructor Arguments:**
```python
DiffusionScheduler(
    num_timesteps: int = 1000,
    beta_schedule: Literal['linear', 'cosine', 'quadratic'] = 'linear',
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    clip_sample: bool = True,
    prediction_type: Literal['epsilon', 'sample', 'v_prediction'] = 'epsilon',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:20*

#### `DilatedBranch`
**Module:** `models.darkir.model`

A single branch of dilated depthwise convolution for multi-scale context.

**Constructor Arguments:**
```python
DilatedBranch(
    channels: int,
    expansion: int = 1,
    dilation: int = 1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/darkir/model.py:398*

#### `DiscreteConditioningInjection`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Inject discrete conditioning (embeddings) into target features.

**Constructor Arguments:**
```python
DiscreteConditioningInjection(
    method: str = 'spatial_broadcast',
    projected_channels: Optional[int] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    name: str = 'discrete_injection',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:238*

#### `DistilBERT`
**Module:** `models.distilbert.model`

DistilBERT (Distilled BERT) model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/distilbert/model.py:276*

#### `DistilBertEmbeddings`
**Module:** `models.distilbert.model`

Embeddings layer for DistilBERT.

**Constructor Arguments:**
```python
DistilBertEmbeddings(
    vocab_size: int,
    hidden_size: int,
    max_position_embeddings: int = 512,
    sinusoidal_pos_embds: bool = False,
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-12,
    dropout_rate: float = 0.1,
    normalization_type: str = 'layer_norm',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/distilbert/model.py:82*

#### `Downsample`
**Module:** `models.pw_fnet.model`

Trainable downsampling layer using strided convolution.

**Constructor Arguments:**
```python
Downsample(
    dim: int,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/pw_fnet/model.py:449*

#### `DummyDataset`
**Module:** `models.nano_vlm_world_model.train`

*Inherits from: `keras.utils.Sequence`*

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:452*

#### `EarlyStoppingCallback`
**Module:** `models.ccnets.utils`

Early stopping with dual conditions: convergence of model errors and stagnation of gradient flow, indicating that learning has ceased.

*📁 src/dl_techniques/models/ccnets/utils.py:14*

#### `FFTMixer`
**Module:** `models.fftnet.model`

Adaptive spectral filtering layer implementing the core FFTNet mechanism.

**Constructor Arguments:**
```python
FFTMixer(
    embed_dim: int,
    mlp_hidden_dim: int = 256,
    dropout_p: float = 0.0,
    use_bias_in_modrelu: bool = True,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/model.py:73*

#### `FFTNet`
**Module:** `models.fftnet.model`

FFTNet (Adaptive Spectral Filtering) foundation model for vision tasks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/fftnet/model.py:342*

#### `FFTNetBlock`
**Module:** `models.fftnet.model`

Complete Transformer-style block using FFTMixer for token mixing.

**Constructor Arguments:**
```python
FFTNetBlock(
    embed_dim: int,
    mlp_hidden_dim: int = 256,
    ffn_ratio: int = 4,
    dropout_p: float = 0.0,
    ffn_type: str = 'mlp',
    normalization_type: str = 'layer_norm',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/model.py:239*

#### `FNet`
**Module:** `models.fnet.model`

FNet (Fourier Transform-based Neural Network) model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/fnet/model.py:73*

#### `FastVLM`
**Module:** `models.fastvlm.model`

FastVLM: A fast hybrid vision_heads model combining efficient convolutions and transformers.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/fastvlm/model.py:26*

#### `FireModule`
**Module:** `models.squeezenet.squeezenet_v1`

Fire module - the fundamental building block of SqueezeNet.

**Constructor Arguments:**
```python
FireModule(
    s1x1: int,
    e1x1: int,
    e3x3: int,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:79*

#### `FractalNet`
**Module:** `models.fractalnet.model`

FractalNet model implementation using modern Keras 3 patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/fractalnet/model.py:67*

#### `FreMLP`
**Module:** `models.darkir.model`

Frequency MLP: Processes features in the frequency domain for global modeling.

**Constructor Arguments:**
```python
FreMLP(
    channels: int,
    expansion: int = 2,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/darkir/model.py:191*

#### `Gemma3`
**Module:** `models.gemma.gemma3`

Gemma 3 Language Model with dual normalization and mixed attention patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/gemma/gemma3.py:24*

#### `Gemma3TransformerBlock`
**Module:** `models.gemma.components`

Gemma 3 Transformer Block with a dual normalization pattern.

**Constructor Arguments:**
```python
Gemma3TransformerBlock(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    ffn_hidden_size: int,
    max_seq_len: int = 32768,
    attention_type: Literal['sliding_window', 'full_attention'] = 'full_attention',
    sliding_window_size: int = 512,
    dropout_rate: float = 0.0,
    use_bias: bool = False,
    norm_eps: float = 1e-06,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/gemma/components.py:25*

#### `GroupAttention`
**Module:** `models.tree_transformer.model`

Hierarchical group attention for Tree Transformer.

**Constructor Arguments:**
```python
GroupAttention(
    hidden_size: int,
    normalization_type: str = 'layer_norm',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/tree_transformer/model.py:201*

#### `HashNGramEmbedding`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Hash-based n-gram embedding layer for enhanced byte representations.

**Constructor Arguments:**
```python
HashNGramEmbedding(
    hash_vocab_size: int,
    embed_dim: int,
    ngram_sizes: List[int] = [3, 4, 5, 6, 7, 8],
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:191*

#### `HashNGramEmbedding`
**Module:** `models.modern_bert.components`

Computes hash n-gram embeddings for byte-level tokens.

**Constructor Arguments:**
```python
HashNGramEmbedding(
    hash_vocab_size: int,
    embed_dim: int,
    ngram_sizes: List[int],
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/modern_bert/components.py:90*

#### `HierarchicalReasoningModel`
**Module:** `models.hierarchical_reasoning_model.model`

Hierarchical Reasoning Model with Adaptive Computation Time.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:166*

#### `HuberLoss`
**Module:** `models.ccnets.losses`

Huber loss function (smooth L1).

*Inherits from: `LossFunction`*

*📁 src/dl_techniques/models/ccnets/losses.py:43*

#### `ImageEncoderViT`
**Module:** `models.sam.image_encoder`

The Vision Transformer (ViT) Image Encoder for SAM.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/sam/image_encoder.py:726*

#### `ImageProjectionHead`
**Module:** `models.mobile_clip.components`

Projects image feature maps into a fixed-size embedding.

**Constructor Arguments:**
```python
ImageProjectionHead(
    projection_dim: int,
    dropout_rate: float = 0.0,
    activation: Optional[Union[str, Callable]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/mobile_clip/components.py:19*

#### `JEPAConfig`
**Module:** `models.jepa.config`

Comprehensive configuration for Joint Embedding Predictive Architecture models.

*📁 src/dl_techniques/models/jepa/config.py:15*

#### `JEPAEncoder`
**Module:** `models.jepa.encoder`

JEPA Encoder using Vision Transformer architecture with modern optimizations.

**Constructor Arguments:**
```python
JEPAEncoder(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    patch_size: Union[int, Tuple[int, ...]] = 16,
    img_size: Tuple[int, ...] = (224, 224),
    variant: str = 'image',
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    layer_scale_init: float = 0.0001,
    use_layer_scale: bool = True,
    use_gradient_checkpointing: bool = False,
    activation: str = 'gelu',
    norm_type: str = 'layer_norm',
    kernel_initializer: Union[str, initializers.Initializer] = 'truncated_normal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/jepa/encoder.py:139*

#### `JEPAMaskingStrategy`
**Module:** `models.jepa.utilities`

Advanced masking strategy for JEPA training with semantic block-based approach.

*📁 src/dl_techniques/models/jepa/utilities.py:18*

#### `JEPAPatchEmbedding`
**Module:** `models.jepa.encoder`

Advanced patch embedding layer for JEPA with support for different modalities.

**Constructor Arguments:**
```python
JEPAPatchEmbedding(
    patch_size: Union[int, Tuple[int, ...]],
    embed_dim: int,
    img_size: Tuple[int, ...],
    variant: str = 'image',
    kernel_initializer: Union[str, initializers.Initializer] = 'truncated_normal',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/jepa/encoder.py:18*

#### `JEPAPredictor`
**Module:** `models.jepa.encoder`

JEPA Predictor network for masked token prediction.

**Constructor Arguments:**
```python
JEPAPredictor(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    activation: str = 'gelu',
    norm_type: str = 'layer_norm',
    kernel_initializer: Union[str, initializers.Initializer] = 'truncated_normal',
    bias_initializer: Union[str, initializers.Initializer] = 'zeros',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/jepa/encoder.py:356*

#### `JointDenoiser`
**Module:** `models.nano_vlm_world_model.denoisers`

Joint denoiser for simultaneous vision and text denoising.

**Constructor Arguments:**
```python
JointDenoiser(
    vision_dim: int,
    text_dim: int,
    hidden_dim: int = 1024,
    num_layers: int = 16,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:413*

#### `KAN`
**Module:** `models.kan.model`

Modern Kolmogorov-Arnold Network model using Keras 3 functional API patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/kan/model.py:42*

#### `KerasModelWrapper`
**Module:** `models.ccnets.utils`

*📁 src/dl_techniques/models/ccnets/utils.py:98*

#### `L1Loss`
**Module:** `models.ccnets.losses`

L1 (Mean Absolute Error) loss function.

*Inherits from: `LossFunction`*

*📁 src/dl_techniques/models/ccnets/losses.py:25*

#### `L2Loss`
**Module:** `models.ccnets.losses`

L2 (Mean Squared Error) loss function.

*Inherits from: `LossFunction`*

*📁 src/dl_techniques/models/ccnets/losses.py:34*

#### `LatentGMMRegistration`
**Module:** `models.latent_gmm_registration.model`

Robust Semi-Supervised Point Cloud Registration via Latent GMM.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:15*

#### `LossFunction`
**Module:** `models.ccnets.losses`

Abstract base class for loss functions used in CCNet.

*Inherits from: `ABC`*

*📁 src/dl_techniques/models/ccnets/losses.py:6*

#### `MDNModel`
**Module:** `models.mdn.model`

A complete Mixture Density Network model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mdn/model.py:99*

#### `Mamba`
**Module:** `models.mamba.mamba_v1`

Mamba (v1) foundation model for efficient sequence modeling.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mamba/mamba_v1.py:125*

#### `Mamba2`
**Module:** `models.mamba.mamba_v2`

Mamba v2 foundation model for efficient sequence modeling.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mamba/mamba_v2.py:14*

#### `Mamba2Layer`
**Module:** `models.mamba.components_v2`

Core Mamba v2 selective state space model layer.

**Constructor Arguments:**
```python
Mamba2Layer(
    d_model: int,
    d_state: int = 128,
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 64,
    ngroups: int = 1,
    d_ssm: Optional[int] = None,
    rmsnorm: bool = True,
    norm_epsilon: float = 1e-05,
    norm_before_gate: bool = True,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init_floor: float = 0.0001,
    bias: bool = False,
    conv_bias: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/mamba/components_v2.py:10*

#### `Mamba2ResidualBlock`
**Module:** `models.mamba.components_v2`

Residual block wrapping a Mamba2Layer with pre-normalization.

**Constructor Arguments:**
```python
Mamba2ResidualBlock(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    d_ssm: int,
    norm_epsilon: float = 1e-05,
    rmsnorm: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/mamba/components_v2.py:320*

#### `MambaLayer`
**Module:** `models.mamba.components`

Core Mamba selective state space model layer.

**Constructor Arguments:**
```python
MambaLayer(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dt_rank: Union[str, int] = 'auto',
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init: str = 'random',
    dt_scale: float = 1.0,
    dt_init_floor: float = 0.0001,
    conv_bias: bool = True,
    use_bias: bool = False,
    layer_idx: Optional[int] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/mamba/components.py:15*

#### `MambaResidualBlock`
**Module:** `models.mamba.components`

Residual block wrapping a MambaLayer with pre-normalization.

**Constructor Arguments:**
```python
MambaResidualBlock(
    d_model: int,
    norm_epsilon: float = 1e-05,
    mamba_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/mamba/components.py:550*

#### `MaskDecoder`
**Module:** `models.sam.mask_decoder`

Predicts segmentation masks from image and prompt embeddings using a transformer.

**Constructor Arguments:**
```python
MaskDecoder(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/sam/mask_decoder.py:121*

#### `MaskedAutoencoder`
**Module:** `models.masked_autoencoder.mae`

Masked Autoencoder (MAE) model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:29*

#### `MaskedLanguageModel`
**Module:** `models.masked_language_model.mlm`

A model-agnostic Masked Language Modeling (MLM) pre-trainer.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/masked_language_model/mlm.py:65*

#### `MeanPoolingLayer`
**Module:** `models.fftnet.components`

Simple mean pooling over the sequence dimension.

**Constructor Arguments:**
```python
MeanPoolingLayer(
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:48*

#### `MemoryIntegrationLayer`
**Module:** `models.qwen.qwen3_mega`

Layer that integrates MANN memory and GNN entity graph with transformer hidden states.

**Constructor Arguments:**
```python
MemoryIntegrationLayer(
    hidden_size: int,
    memory_dim: int,
    entity_dim: int,
    num_attention_heads: int = 8,
    dropout_rate: float = 0.1,
    use_memory_write: bool = True,
    use_gnn_update: bool = True,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:39*

#### `MiniVec2VecAligner`
**Module:** `models.mini_vec2vec.model`

Keras implementation of the mini-vec2vec unsupervised alignment algorithm.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mini_vec2vec/model.py:22*

#### `MobileClipImageEncoder`
**Module:** `models.mobile_clip.components`

MobileClip Image Encoder combining a backbone and a projection head.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mobile_clip/components.py:165*

#### `MobileClipModel`
**Module:** `models.mobile_clip.mobile_clip_v1`

Mobile CLIP Model combining image and text encoders with variant support.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:17*

#### `MobileClipTextEncoder`
**Module:** `models.mobile_clip.components`

MobileClip Text Encoder using a stack of Transformer layers.

**Constructor Arguments:**
```python
MobileClipTextEncoder(
    vocab_size: int,
    max_seq_len: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    intermediate_size: int,
    projection_dim: int,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    use_causal_mask: bool = True,
    embed_scale: Optional[float] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/mobile_clip/components.py:301*

#### `MobileNetV1`
**Module:** `models.mobilenet.mobilenet_v1`

MobileNetV1 model implementation with depthwise separable convolutions.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:56*

#### `MobileNetV2`
**Module:** `models.mobilenet.mobilenet_v2`

MobileNetV2 classification model built with Universal Inverted Bottleneck blocks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mobilenet/mobilenet_v2.py:66*

#### `MobileNetV3`
**Module:** `models.mobilenet.mobilenet_v3`

MobileNetV3 model implemented with Universal Inverted Bottleneck blocks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:65*

#### `MobileNetV4`
**Module:** `models.mobilenet.mobilenet_v4`

MobileNetV4 model implementation with Universal Inverted Bottleneck blocks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:72*

#### `ModernBERT`
**Module:** `models.modern_bert.modern_bert`

ModernBERT (A Modern Bidirectional Encoder) foundation model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:107*

#### `ModernBertBLT`
**Module:** `models.modern_bert.modern_bert_blt`

A Modern Bidirectional Encoder with Byte Latent Transformer (BLT) features.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:61*

#### `ModernBertBltEmbeddings`
**Module:** `models.modern_bert.components`

Combines byte, positional, and optional hash n-gram embeddings.

**Constructor Arguments:**
```python
ModernBertBltEmbeddings(
    vocab_size: int,
    hidden_size: int,
    max_position_embeddings: int,
    initializer_range: float,
    layer_norm_eps: float,
    hidden_dropout_prob: float,
    use_hash_embeddings: bool,
    normalization_type: str,
    hash_vocab_size: Optional[int] = None,
    ngram_sizes: Optional[List[int]] = None,
    hash_embedding_dim: Optional[int] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/modern_bert/components.py:215*

#### `MothNet`
**Module:** `models.mothnet.model`

Complete MothNet architecture combining AL, MB, and Hebbian readout layers.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/mothnet/model.py:94*

#### `NBeatsNet`
**Module:** `models.nbeats.nbeats`

Neural Basis Expansion Analysis for Time Series (N-BEATS) forecasting model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/nbeats/nbeats.py:75*

#### `NBeatsXNet`
**Module:** `models.nbeats.nbeatsx`

N-BEATSx: Neural Basis Expansion Analysis with Exogenous Variables.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/nbeats/nbeatsx.py:16*

#### `NTMModel`
**Module:** `models.ntm.model`

Neural Turing Machine Model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/ntm/model.py:34*

#### `NTMMultiTask`
**Module:** `models.ntm.model_multitask`

A Neural Turing Machine wrapper for Multi-Task Learning.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/ntm/model_multitask.py:15*

#### `NanoVLM`
**Module:** `models.nano_vlm.model`

NanoVLM: Modern Compact Vision-Language Model using existing dl-techniques components.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/nano_vlm/model.py:116*

#### `PFTSR`
**Module:** `models.pft_sr.model`

Progressive Focused Transformer for Single Image Super-Resolution.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/pft_sr/model.py:19*

#### `PRISMModel`
**Module:** `models.prism.model`

Complete PRISM model for time series forecasting.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/prism/model.py:88*

#### `PW_FNet`
**Module:** `models.pw_fnet.model`

Complete Pyramid Wavelet-Fourier Network (PW-FNet) model for image restoration.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/pw_fnet/model.py:628*

#### `PW_FNet_Block`
**Module:** `models.pw_fnet.model`

Pyramid Wavelet-Fourier Network (PW-FNet) building block with configurable components.

**Constructor Arguments:**
```python
PW_FNet_Block(
    dim: int,
    ffn_expansion_factor: float = 2.0,
    normalization_type: str = 'layer_norm',
    norm1_kwargs: Optional[Dict[str, Any]] = None,
    norm2_kwargs: Optional[Dict[str, Any]] = None,
    use_spatial_ffn: bool = True,
    ffn_type: Optional[str] = None,
    ffn_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/pw_fnet/model.py:36*

#### `PatchEmbedding`
**Module:** `models.sam.image_encoder`

Image to Patch Embedding Layer.

**Constructor Arguments:**
```python
PatchEmbedding(
    patch_size: Union[int, Tuple[int, int]] = 16,
    embed_dim: int = 768,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/sam/image_encoder.py:117*

#### `PatchMasking`
**Module:** `models.masked_autoencoder.patch_masking`

Layer for creating patches and applying random masking.

**Constructor Arguments:**
```python
PatchMasking(
    patch_size: int = 16,
    mask_ratio: float = 0.75,
    mask_value: Union[str, float] = 'learnable',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/masked_autoencoder/patch_masking.py:12*

#### `PolynomialLoss`
**Module:** `models.ccnets.losses`

A generalized loss function that computes the mean of the absolute error raised to a power 'p'. This provides a flexible way to control the system's aversion to large errors.

*Inherits from: `LossFunction`*

*📁 src/dl_techniques/models/ccnets/losses.py:65*

#### `PositionEmbeddingRandom`
**Module:** `models.sam.prompt_encoder`

Positional encoding using random spatial frequencies.

**Constructor Arguments:**
```python
PositionEmbeddingRandom(
    num_pos_feats: int = 64,
    scale: float = 1.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/sam/prompt_encoder.py:88*

#### `PositionalEncoding`
**Module:** `models.tree_transformer.model`

Injects sinusoidal positional encoding into input embeddings.

**Constructor Arguments:**
```python
PositionalEncoding(
    hidden_size: int,
    dropout_rate: float,
    max_len: int = 5000,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/tree_transformer/model.py:84*

#### `PowerMLP`
**Module:** `models.power_mlp.model`

PowerMLP model: Efficient alternative to Kolmogorov-Arnold Networks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/power_mlp/model.py:134*

#### `PromptEncoder`
**Module:** `models.sam.prompt_encoder`

Encodes prompts (points, boxes, masks) for the SAM mask decoder.

**Constructor Arguments:**
```python
PromptEncoder(
    embed_dim: int,
    image_embedding_size: Tuple[int, int],
    input_image_size: Tuple[int, int],
    mask_in_chans: int = 16,
    normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
    activation: str = 'gelu',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/sam/prompt_encoder.py:240*

#### `Qwen3`
**Module:** `models.qwen.qwen3`

Qwen3 model with standard transformer architecture and optional MoE layers.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/qwen/qwen3.py:32*

#### `Qwen3EmbeddingLayer`
**Module:** `models.qwen.qwen3_embeddings`

Keras implementation of the Qwen3 Text Embedding model using factory components.

**Constructor Arguments:**
```python
Qwen3EmbeddingLayer(
    vocab_size: int,
    hidden_size: int = 1024,
    num_layers: int = 12,
    num_heads: int = 16,
    intermediate_size: int = 2816,
    max_seq_len: int = 8192,
    normalize: bool = True,
    truncate_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    ffn_type: str = 'swiglu',
    normalization_type: str = 'rms_norm',
    attention_type: str = 'multi_head',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:86*

#### `Qwen3EmbeddingModel`
**Module:** `models.qwen.qwen3_embeddings`

High-level Keras Model for Qwen3 Text Embedding.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:586*

#### `Qwen3MEGA`
**Module:** `models.qwen.qwen3_mega`

Qwen3 model enhanced with Memory-Augmented Neural Networks and Graph Neural Networks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:237*

#### `Qwen3Next`
**Module:** `models.qwen.qwen3_next`

Qwen3 Next (Mixture of Experts) model with correct architecture.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/qwen/qwen3_next.py:28*

#### `Qwen3NextBlock`
**Module:** `models.qwen.components`

Qwen3 Next transformer block implementing the exact architectural pattern.

**Constructor Arguments:**
```python
Qwen3NextBlock(
    dim: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    max_seq_len: int = 4096,
    moe_config: Optional[Any] = None,
    normalization_type: str = 'zero_centered_rms_norm',
    norm_eps: float = 1e-06,
    dropout_rate: float = 0.0,
    use_stochastic_depth: bool = False,
    stochastic_depth_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/qwen/components.py:31*

#### `Qwen3RerankerLayer`
**Module:** `models.qwen.qwen3_embeddings`

Keras implementation of the Qwen3 Reranker using factory components.

**Constructor Arguments:**
```python
Qwen3RerankerLayer(
    vocab_size: int,
    hidden_size: int = 1024,
    num_layers: int = 12,
    num_heads: int = 16,
    intermediate_size: int = 2816,
    max_seq_len: int = 8192,
    dropout_rate: float = 0.0,
    ffn_type: str = 'swiglu',
    normalization_type: str = 'rms_norm',
    attention_type: str = 'multi_head',
    yes_token_id: int = 9891,
    no_token_id: int = 2201,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:333*

#### `Qwen3RerankerModel`
**Module:** `models.qwen.qwen3_embeddings`

High-level Keras Model for Qwen3 Text Reranking.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:713*

#### `Qwen3SOM`
**Module:** `models.qwen.qwen3_som`

Qwen3 language model with Self-Organizing Map memory integration.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/qwen/qwen3_som.py:62*

#### `RELGT`
**Module:** `models.relgt.model`

Complete Relational Graph Transformer model for multi-table relational data.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/relgt/model.py:28*

#### `ReasoningByteBERT`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

ReasoningByteBERT: Combining ByteBERT with Hierarchical Reasoning and ACT.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:814*

#### `ReasoningByteBertConfig`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Configuration for ReasoningByteBERT combining ByteBERT and HRM features.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:62*

#### `ReasoningByteCore`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Core reasoning engine combining byte-level processing with hierarchical reasoning.

**Constructor Arguments:**
```python
ReasoningByteCore(
    config: ReasoningByteBertConfig,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:518*

#### `ReasoningByteEmbeddings`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

ReasoningByte embeddings combining byte-level processing with puzzle context.

**Constructor Arguments:**
```python
ReasoningByteEmbeddings(
    config: ReasoningByteBertConfig,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:309*

#### `ResNet`
**Module:** `models.resnet.model`

ResNet model implementation with pretrained support and deep supervision.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/resnet/model.py:69*

#### `SAM`
**Module:** `models.sam.model`

Segment Anything Model (SAM) - A foundation model for image segmentation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/sam/model.py:141*

#### `SCUNet`
**Module:** `models.scunet.model`

Swin-Conv-UNet for image restoration tasks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/scunet/model.py:16*

#### `SHGCNLinkPredictor`
**Module:** `models.shgcn.model`

Complete link prediction model with sHGCN backbone and Fermi-Dirac decoder.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/shgcn/model.py:400*

#### `SHGCNModel`
**Module:** `models.shgcn.model`

Multi-layer Simplified Hyperbolic Graph Convolutional Neural Network.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/shgcn/model.py:26*

#### `SHGCNNodeClassifier`
**Module:** `models.shgcn.model`

Complete node classification model with sHGCN backbone and classification head.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/shgcn/model.py:254*

#### `SOMModel`
**Module:** `models.som.model`

Self-Organizing Map model implementing associative memory and topological learning.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/som/model.py:88*

#### `ScoreBasedNanoVLM`
**Module:** `models.nano_vlm_world_model.model`

Score-Based nanoVLM: A Navigable Vision-Language World Model.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:33*

#### `ScoreVLMTrainer`
**Module:** `models.nano_vlm_world_model.train`

Custom trainer for Score-Based nanoVLM.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:189*

#### `SequentialCCNetOrchestrator`
**Module:** `models.ccnets.orchestrators`

Extended orchestrator for sequential data. Implements reverse causality for the Producer via sequence reversal.

*Inherits from: `CCNetOrchestrator`*

*📁 src/dl_techniques/models/ccnets/orchestrators.py:407*

#### `SigLIPVisionTransformer`
**Module:** `models.vit_siglip.model`

SigLIP Vision Transformer model with factory-based component creation and modern Keras 3 patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/vit_siglip/model.py:41*

#### `SimpleGate`
**Module:** `models.darkir.model`

SimpleGate: Element-wise multiplicative gating without learnable parameters.

**Constructor Arguments:**
```python
SimpleGate(
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/darkir/model.py:92*

#### `SimplifiedFireModule`
**Module:** `models.squeezenet.squeezenet_v2`

Simplified Fire module - the core building block of SqueezeNodule-Net.

**Constructor Arguments:**
```python
SimplifiedFireModule(
    s1x1: int,
    e3x3: int,
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:77*

#### `SpectreBlock`
**Module:** `models.fftnet.components`

Complete Transformer-style block using SpectreMultiHead for token mixing.

**Constructor Arguments:**
```python
SpectreBlock(
    embed_dim: int,
    num_heads: int,
    fft_size: int,
    mlp_ratio: int = 4,
    memory_size: int = 0,
    ffn_type: str = 'mlp',
    normalization_type: str = 'layer_norm',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:915*

#### `SpectreHead`
**Module:** `models.fftnet.components`

Frequency-domain token mixer for a single attention head.

**Constructor Arguments:**
```python
SpectreHead(
    embed_dim: int,
    fft_size: int,
    num_groups: int = 4,
    num_buckets: Optional[int] = None,
    d_gate: int = 256,
    use_toeplitz: bool = False,
    toeplitz_bw: int = 4,
    dropout_p: float = 0.0,
    pooling_type: Literal['dct', 'attention', 'mean'] = 'dct',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:552*

#### `SpectreMultiHead`
**Module:** `models.fftnet.components`

Multi-head Spectre layer combining multiple SpectreHead instances.

**Constructor Arguments:**
```python
SpectreMultiHead(
    embed_dim: int,
    num_heads: int,
    fft_size: int,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/fftnet/components.py:792*

#### `SqueezeNetV1`
**Module:** `models.squeezenet.squeezenet_v1`

SqueezeNet V1 model implementation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:219*

#### `SqueezeNoduleNetV2`
**Module:** `models.squeezenet.squeezenet_v2`

SqueezeNodule-Net V2 model implementation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:198*

#### `StaticThresholdStrategy`
**Module:** `models.ccnets.control`

The default strategy: trains the Reasoner as long as its accuracy is below a fixed threshold. This encapsulates the original behavior.

*Inherits from: `ConvergenceControlStrategy`*

*📁 src/dl_techniques/models/ccnets/control.py:27*

#### `SwinTransformer`
**Module:** `models.swin_transformer.model`

Hierarchical Vision Transformer using shifted windows for efficient image classification.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/swin_transformer/model.py:56*

#### `TRM`
**Module:** `models.tiny_recursive_model.model`

Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/tiny_recursive_model/model.py:62*

#### `TRMInner`
**Module:** `models.tiny_recursive_model.components`

The inner computational core of the TRM model.

**Constructor Arguments:**
```python
TRMInner(
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    expansion: float,
    seq_len: int,
    puzzle_emb_len: int = 16,
    h_layers: int = 2,
    l_layers: int = 2,
    rope_theta: float = 10000.0,
    attention_type: AttentionType = 'multi_head',
    ffn_type: FFNType = 'swiglu',
    normalization_type: NormalizationType = 'rms_norm',
    normalization_position: NormalizationPositionType = 'post',
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:296*

#### `TRMReasoningModule`
**Module:** `models.tiny_recursive_model.components`

A module that stacks multiple TransformerLayers for deep reasoning.

**Constructor Arguments:**
```python
TRMReasoningModule(
    hidden_size: int,
    num_heads: int,
    expansion: float,
    num_layers: int,
    seq_len: int,
    puzzle_emb_len: int = 16,
    rope_theta: float = 10000.0,
    attention_type: AttentionType = 'multi_head',
    ffn_type: FFNType = 'swiglu',
    normalization_type: NormalizationType = 'rms_norm',
    normalization_position: NormalizationPositionType = 'post',
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:47*

#### `TabMModel`
**Module:** `models.tabm.model`

TabM: Deep Ensemble Architecture for High-Performance Tabular Learning.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/tabm/model.py:222*

#### `TextDenoiser`
**Module:** `models.nano_vlm_world_model.denoisers`

Denoiser for text embeddings conditioned on images.

**Constructor Arguments:**
```python
TextDenoiser(
    text_dim: int,
    vision_dim: int,
    num_layers: int = 12,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:340*

#### `TiRexCore`
**Module:** `models.tirex.model`

TiRex Core Model for Time Series Forecasting.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/tirex/model.py:83*

#### `TiRexExtended`
**Module:** `models.tirex.model_extended`

TiRex Extended (Query-Based) Architecture.

*Inherits from: `TiRexCore`*

*📁 src/dl_techniques/models/tirex/model_extended.py:72*

#### `TimestepEmbedding`
**Module:** `models.nano_vlm_world_model.denoisers`

Sinusoidal timestep embedding for diffusion models.

**Constructor Arguments:**
```python
TimestepEmbedding(
    embedding_dim: int,
    max_period: int = 10000,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:22*

#### `TreeMHA`
**Module:** `models.tree_transformer.model`

Multi-Head Attention modulated by Tree Transformer group probabilities.

**Constructor Arguments:**
```python
TreeMHA(
    num_heads: int,
    hidden_size: int,
    attention_dropout_rate: float = 0.1,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/tree_transformer/model.py:378*

#### `TreeTransformer`
**Module:** `models.tree_transformer.model`

Tree Transformer model for grammar induction and language modeling.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/tree_transformer/model.py:711*

#### `TreeTransformerBlock`
**Module:** `models.tree_transformer.model`

Single block of the Tree Transformer encoder.

**Constructor Arguments:**
```python
TreeTransformerBlock(
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    hidden_dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    normalization_type: NormalizationType = 'layer_norm',
    ffn_type: FFNType = 'mlp',
    hidden_act: str = 'gelu',
    layer_norm_eps: float = 1e-12,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/tree_transformer/model.py:528*

#### `TwoWayAttentionBlock`
**Module:** `models.sam.transformer`

A transformer block with four layers for bidirectional attention.

**Constructor Arguments:**
```python
TwoWayAttentionBlock(
    embedding_dim: int,
    num_heads: int,
    mlp_dim: int = 2048,
    skip_first_layer_pe: bool = False,
    normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
    activation: str = 'relu',
    attention_dropout: float = 0.0,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/sam/transformer.py:114*

#### `TwoWayTransformer`
**Module:** `models.sam.transformer`

A two-way transformer decoder for joint refinement of queries and image features.

**Constructor Arguments:**
```python
TwoWayTransformer(
    depth: int,
    embedding_dim: int,
    num_heads: int,
    mlp_dim: int = 2048,
    normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
    activation: str = 'relu',
    attention_dropout: float = 0.0,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/sam/transformer.py:444*

#### `Upsample`
**Module:** `models.pw_fnet.model`

Trainable upsampling layer using transposed convolution.

**Constructor Arguments:**
```python
Upsample(
    dim: int,
    **kwargs
)
```

*Inherits from: `keras.layers.Layer`*

*📁 src/dl_techniques/models/pw_fnet/model.py:536*

#### `VAE`
**Module:** `models.vae.model`

ResNet-based Variational Autoencoder using modern Keras 3 patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/vae/model.py:77*

#### `VLMDenoisingLoss`
**Module:** `models.nano_vlm_world_model.train`

Combined loss for vision-language denoising.

*Inherits from: `keras.losses.Loss`*

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:101*

#### `VQVAEModel`
**Module:** `models.vq_vae.model`

Complete VQ-VAE model combining encoder, quantizer, and decoder.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/vq_vae/model.py:85*

#### `ViT`
**Module:** `models.vit.model`

Vision Transformer model with factory-based component creation and modern Keras 3 patterns.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/vit/model.py:35*

#### `ViTBlock`
**Module:** `models.sam.image_encoder`

Transformer Block for the Vision Transformer with Windowing Support.

**Constructor Arguments:**
```python
ViTBlock(
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    use_rel_pos: bool = False,
    window_size: int = 0,
    input_size: Optional[Tuple[int, int]] = None,
    normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
    ffn_type: Literal['mlp', 'swiglu', 'geglu', 'glu'] = 'mlp',
    activation: str = 'gelu',
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/sam/image_encoder.py:467*

#### `ViTHMLP`
**Module:** `models.vit_hmlp.model`

Vision Transformer with Hierarchical MLP Stem using factory-based component creation.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/vit_hmlp/model.py:78*

#### `VideoMaskingStrategy`
**Module:** `models.jepa.utilities`

Extended masking strategy for video data (V-JEPA).

*Inherits from: `JEPAMaskingStrategy`*

*📁 src/dl_techniques/models/jepa/utilities.py:396*

#### `VisionDenoiser`
**Module:** `models.nano_vlm_world_model.denoisers`

Denoiser for image data conditioned on text.

**Constructor Arguments:**
```python
VisionDenoiser(
    vision_config: Dict[str, Any],
    text_dim: int,
    num_layers: int = 12,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:265*

#### `WindowedAttentionWithRelPos`
**Module:** `models.sam.image_encoder`

Multi-Head Self-Attention with optional Relative Positional Embeddings and Windowing.

**Constructor Arguments:**
```python
WindowedAttentionWithRelPos(
    dim: int,
    num_heads: int = 8,
    qkv_bias: bool = True,
    use_rel_pos: bool = False,
    input_size: Optional[Tuple[int, int]] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/models/sam/image_encoder.py:233*

#### `YOLOv12FeatureExtractor`
**Module:** `models.yolo12.feature_extractor`

YOLOv12 Feature Extractor (Backbone + Neck).

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:26*

#### `YOLOv12MultiTask`
**Module:** `models.yolo12.multitask`

YOLOv12 Multi-Task Learning Model using Named Outputs (Functional API).

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/yolo12/multitask.py:104*

#### `xLSTM`
**Module:** `models.xlstm.model`

Complete xLSTM architecture with stacked sLSTM and mLSTM blocks.

*Inherits from: `keras.Model`*

*📁 src/dl_techniques/models/xlstm/model.py:72*

### Optimization Classes

#### `DatasetBuilder`
**Module:** `optimization.train_vision.framework`

Abstract base class for creating training and validation datasets.

*Inherits from: `ABC`*

*📁 src/dl_techniques/optimization/train_vision/framework.py:335*

#### `EnhancedVisualizationCallback`
**Module:** `optimization.train_vision.framework`

Enhanced callback for creating comprehensive visualizations during training.

*Inherits from: `keras.callbacks.Callback`*

*📁 src/dl_techniques/optimization/train_vision/framework.py:398*

#### `Muon`
**Module:** `optimization.muon_optimizer`

Muon (MomentUm Orthogonalized by Newton-schulz) Optimizer.

*Inherits from: `keras.optimizers.Optimizer`*

*📁 src/dl_techniques/optimization/muon_optimizer.py:62*

#### `OptimizerType`
**Module:** `optimization.optimizer`

Enumeration of available optimizer types.

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/optimization/optimizer.py:58*

#### `ScheduleType`
**Module:** `optimization.schedule`

Enumeration of available learning rate schedule types.

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/optimization/schedule.py:50*

#### `ScheduleType`
**Module:** `optimization.deep_supervision`

Enumeration of available deep supervision schedule types.

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/optimization/deep_supervision.py:52*

#### `ScheduleType`
**Module:** `optimization.optimizer`

Enumeration of available learning rate schedule types.

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/optimization/optimizer.py:51*

#### `SledEvolutionType`
**Module:** `optimization.sled_supervision`

Enumeration of available SLED algorithm versions.

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/optimization/sled_supervision.py:73*

#### `SledLogitsProcessor`
**Module:** `optimization.sled_supervision`

A callable class that applies the SLED algorithm to a set of logits using the Keras API.

*📁 src/dl_techniques/optimization/sled_supervision.py:105*

#### `TrainingConfig`
**Module:** `optimization.train_vision.framework`

Centralized configuration for training vision models.

*📁 src/dl_techniques/optimization/train_vision/framework.py:79*

#### `TrainingPipeline`
**Module:** `optimization.train_vision.framework`

Orchestrates end-to-end model training with visualization and analysis.

*📁 src/dl_techniques/optimization/train_vision/framework.py:644*

#### `WarmupSchedule`
**Module:** `optimization.warmup_schedule`

Learning rate schedule with linear warmup followed by primary schedule.

*Inherits from: `keras.optimizers.schedules.LearningRateSchedule`*

*📁 src/dl_techniques/optimization/warmup_schedule.py:69*

### Regularizers Classes

#### `BinaryPreferenceRegularizer`
**Module:** `regularizers.binary_preference`

A regularizer that encourages weights to move towards binary values (0 or 1).

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/binary_preference.py:85*

#### `EntropyRegularizer`
**Module:** `regularizers.entropy_regularizer`

Custom regularizer that promotes entropy-based structure in neural network weights.

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/entropy_regularizer.py:103*

#### `L2_custom`
**Module:** `regularizers.l2_custom`

A regularizer that applies a L2 regularization penalty but also allows negative l2 (forces the weights to increase)

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/l2_custom.py:57*

#### `SRIPRegularizer`
**Module:** `regularizers.srip`

Spectral Restricted Isometry Property (SRIP) regularizer.

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/srip.py:66*

#### `SoftOrthogonalConstraintRegularizer`
**Module:** `regularizers.soft_orthogonal`

Implements soft orthogonality constraint regularization.

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/soft_orthogonal.py:115*

#### `SoftOrthonormalConstraintRegularizer`
**Module:** `regularizers.soft_orthogonal`

Implements soft orthonormality constraint regularization.

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/soft_orthogonal.py:300*

#### `TriStatePreferenceRegularizer`
**Module:** `regularizers.tri_state_preference`

A regularizer that encourages weights to converge to -1, 0, or 1.

*Inherits from: `keras.regularizers.Regularizer`*

*📁 src/dl_techniques/regularizers/tri_state_preference.py:71*

### Utils Classes

#### `ARCAccuracyMetric`
**Module:** `datasets.arc.arc_keras`

Custom accuracy metric for ARC datasets.

*Inherits from: `keras.metrics.Metric`*

*📁 src/dl_techniques/datasets/arc/arc_keras.py:518*

#### `ARCDataAugmenter`
**Module:** `datasets.arc.arc_converters`

Advanced data augmentation for ARC datasets.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:445*

#### `ARCDatasetAnalyzer`
**Module:** `datasets.arc.arc_utilities`

Utility class for analyzing ARC datasets.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:362*

#### `ARCDatasetLoader`
**Module:** `datasets.arc.arc_utilities`

Utility class for loading ARC datasets in HRM format.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:142*

#### `ARCDatasetMerger`
**Module:** `datasets.arc.arc_converters`

Utility for merging multiple ARC datasets.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:376*

#### `ARCDatasetSplitter`
**Module:** `datasets.arc.arc_converters`

Utility for splitting ARC datasets into train/validation/test sets.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:567*

#### `ARCDatasetStats`
**Module:** `datasets.arc.arc_utilities`

Statistics about an ARC dataset.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:124*

#### `ARCDatasetValidator`
**Module:** `datasets.arc.arc_utilities`

Utility class for validating ARC datasets.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:733*

#### `ARCDatasetVisualizer`
**Module:** `datasets.arc.arc_utilities`

Utility class for visualizing ARC datasets.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:493*

#### `ARCExample`
**Module:** `datasets.arc.arc_utilities`

Represents a single ARC example with input and output grids.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:92*

#### `ARCFormatConverter`
**Module:** `datasets.arc.arc_converters`

Converter between different ARC data formats.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:133*

#### `ARCGridDecoder`
**Module:** `datasets.arc.arc_keras`

Custom Keras layer for decoding ARC grid sequences.

**Constructor Arguments:**
```python
ARCGridDecoder(
    max_grid_size: int = 30,
    pad_token: int = 0,
    eos_token: int = 1,
    color_offset: int = 2,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/datasets/arc/arc_keras.py:192*

#### `ARCGridEncoder`
**Module:** `datasets.arc.arc_keras`

Custom Keras layer for encoding 2D grids to sequences.

**Constructor Arguments:**
```python
ARCGridEncoder(
    max_grid_size: int = 30,
    pad_token: int = 0,
    eos_token: int = 1,
    color_offset: int = 2,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/datasets/arc/arc_keras.py:291*

#### `ARCPuzzle`
**Module:** `datasets.arc.arc_utilities`

Represents an ARC puzzle with multiple examples.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:109*

#### `ARCPuzzleEmbedding`
**Module:** `datasets.arc.arc_keras`

Custom embedding layer for ARC puzzle identifiers.

**Constructor Arguments:**
```python
ARCPuzzleEmbedding(
    num_puzzles: int,
    embedding_dim: int,
    mask_zero: bool = True,
    embeddings_initializer: str = 'uniform',
    embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
    **kwargs
)
```

*Inherits from: `layers.Layer`*

*📁 src/dl_techniques/datasets/arc/arc_keras.py:399*

#### `ARCSequence`
**Module:** `datasets.arc.arc_keras`

Keras Sequence for loading ARC data efficiently.

*Inherits from: `keras.utils.Sequence`*

*📁 src/dl_techniques/datasets/arc/arc_keras.py:64*

#### `ARCTaskData`
**Module:** `datasets.arc.arc_converters`

Represents an ARC task in standard JSON format.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:98*

#### `Alignment`
**Module:** `utils.alignment.alignment`

High-level API for computing representation alignment.

*📁 src/dl_techniques/utils/alignment/alignment.py:18*

#### `AlignmentLogger`
**Module:** `utils.alignment.alignment`

Logger for tracking alignment scores during training.

*📁 src/dl_techniques/utils/alignment/alignment.py:372*

#### `AlignmentMetrics`
**Module:** `utils.alignment.metrics`

Collection of alignment metrics for neural network representations.

*📁 src/dl_techniques/utils/alignment/metrics.py:22*

#### `AugmentationConfig`
**Module:** `datasets.vision.coco`

Configuration for data augmentation parameters.

*📁 src/dl_techniques/datasets/vision/coco.py:67*

#### `AugmentationConfig`
**Module:** `datasets.arc.arc_converters`

Configuration for data augmentation.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:114*

#### `BaseTimeSeriesDataset`
**Module:** `datasets.time_series.base`

Abstract base class for time series dataset loaders.

*Inherits from: `ABC`*

*📁 src/dl_techniques/datasets/time_series/base.py:41*

#### `BaseTokenizer`
**Module:** `datasets.vqa_dataset`

Abstract base class for tokenizers.

*Inherits from: `ABC`*

*📁 src/dl_techniques/datasets/vqa_dataset.py:24*

#### `BoundingBox`
**Module:** `datasets.sut`

Bounding box representation with enhanced functionality.

*📁 src/dl_techniques/datasets/sut.py:45*

#### `CIFAR100DatasetBuilder`
**Module:** `datasets.vision.common`

Dataset builder for CIFAR-100 fine-grained image classification.

*Inherits from: `DatasetBuilder`*

*📁 src/dl_techniques/datasets/vision/common.py:330*

#### `CIFAR10DatasetBuilder`
**Module:** `datasets.vision.common`

Dataset builder for CIFAR-10 image classification.

*Inherits from: `DatasetBuilder`*

*📁 src/dl_techniques/datasets/vision/common.py:192*

#### `COCODatasetBuilder`
**Module:** `datasets.vision.coco`

Enhanced COCO Dataset Builder with robust preprocessing pipeline.

*📁 src/dl_techniques/datasets/vision/coco.py:218*

#### `ClassificationAggregator`
**Module:** `datasets.patch_transforms`

Aggregates patch-level classification results.

*📁 src/dl_techniques/datasets/patch_transforms.py:445*

#### `ConformalForecaster`
**Module:** `utils.conformal_forecaster`

Model-agnostic wrapper for Inductive Conformal Prediction (ICP).

*📁 src/dl_techniques/utils/conformal_forecaster.py:20*

#### `CoordinateTransformer`
**Module:** `datasets.patch_transforms`

Handles coordinate transformations between patch and full image space.

*📁 src/dl_techniques/datasets/patch_transforms.py:84*

#### `CorruptionSeverity`
**Module:** `utils.corruption`

Enumeration for corruption severity levels.

*Inherits from: `Enum`*

*📁 src/dl_techniques/utils/corruption.py:16*

#### `CorruptionType`
**Module:** `utils.corruption`

Enumeration for available corruption types.

*Inherits from: `Enum`*

*📁 src/dl_techniques/utils/corruption.py:36*

#### `DatasetConfig`
**Module:** `datasets.vision.coco`

Configuration for dataset class mapping and filtering.

*📁 src/dl_techniques/datasets/vision/coco.py:93*

#### `DatasetGenerator`
**Module:** `datasets.simple_2d`

Generator for various synthetic 2D datasets that challenge classification algorithms.

*📁 src/dl_techniques/datasets/simple_2d.py:75*

#### `DatasetSplits`
**Module:** `datasets.time_series.config`

Container for train/validation/test dataset splits.

*📁 src/dl_techniques/datasets/time_series/config.py:127*

#### `DatasetType`
**Module:** `datasets.simple_2d`

Enumeration of available dataset types.

*Inherits from: `Enum`*

*📁 src/dl_techniques/datasets/simple_2d.py:40*

#### `DetectionResult`
**Module:** `datasets.patch_transforms`

Single detection result with confidence.

*📁 src/dl_techniques/datasets/patch_transforms.py:59*

#### `EpochAnalyzerCallback`
**Module:** `callbacks.analyzer_callback`

A Keras callback to run ModelAnalyzer at the end of specified epochs.

*Inherits from: `keras.callbacks.Callback`*

*📁 src/dl_techniques/callbacks/analyzer_callback.py:29*

#### `FavoritaDataset`
**Module:** `datasets.time_series.favorita`

Loader for Favorita Grocery Sales Forecasting Dataset.

*Inherits from: `BaseTimeSeriesDataset`*

*📁 src/dl_techniques/datasets/time_series/favorita.py:62*

#### `ForecastabilityAssessor`
**Module:** `utils.forecastability_analyzer`

Implements the Forecastability Assessment Framework.

*📁 src/dl_techniques/utils/forecastability_analyzer.py:18*

#### `FullImageInference`
**Module:** `utils.inference`

Full image inference engine for YOLOv12 multi-task model.

*📁 src/dl_techniques/utils/inference.py:55*

#### `ImageAnnotation`
**Module:** `datasets.sut`

Complete annotation for a single image with validation.

*📁 src/dl_techniques/datasets/sut.py:93*

#### `InferenceConfig`
**Module:** `utils.inference`

Configuration for full image inference.

*📁 src/dl_techniques/utils/inference.py:41*

#### `InferenceProfiler`
**Module:** `utils.inference`

Performance profiler for inference operations.

*📁 src/dl_techniques/utils/inference.py:481*

#### `LongHorizonDataset`
**Module:** `datasets.time_series.long_horizon`

Loader for Long-Horizon Forecasting Benchmark Datasets.

*Inherits from: `BaseTimeSeriesDataset`*

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:139*

#### `M4Dataset`
**Module:** `datasets.time_series.m4`

Loader for M4 Competition Dataset.

*Inherits from: `BaseTimeSeriesDataset`*

*📁 src/dl_techniques/datasets/time_series/m4.py:108*

#### `MNISTDatasetBuilder`
**Module:** `datasets.vision.common`

Dataset builder for MNIST handwritten digits.

*Inherits from: `DatasetBuilder`*

*📁 src/dl_techniques/datasets/vision/common.py:40*

#### `MaskConfig`
**Module:** `utils.masking.factory`

Configuration for mask creation.

*📁 src/dl_techniques/utils/masking/factory.py:71*

#### `MaskFactory`
**Module:** `utils.masking.factory`

Factory class for creating various types of masks.

*📁 src/dl_techniques/utils/masking/factory.py:163*

#### `MaskType`
**Module:** `utils.masking.factory`

Enumeration of available mask types.

*Inherits from: `str`, `Enum`*

*📁 src/dl_techniques/utils/masking/factory.py:52*

#### `NonMaximumSuppression`
**Module:** `datasets.patch_transforms`

Non-Maximum Suppression for overlapping detections.

*📁 src/dl_techniques/datasets/patch_transforms.py:269*

#### `NormalizationConfig`
**Module:** `datasets.time_series.config`

Configuration for time series normalization.

*📁 src/dl_techniques/datasets/time_series/config.py:204*

#### `NormalizationMethod`
**Module:** `datasets.time_series.normalizer`

Enumeration of available normalization methods.

*Inherits from: `Enum`*

*📁 src/dl_techniques/datasets/time_series/normalizer.py:23*

#### `OptimizedSUTDataset`
**Module:** `datasets.sut`

Highly optimized TensorFlow-native dataset for SUT-Crack patch-based learning.

*📁 src/dl_techniques/datasets/sut.py:803*

#### `PatchGridGenerator`
**Module:** `datasets.patch_transforms`

Generates grid of patches for sliding window inference.

*📁 src/dl_techniques/datasets/patch_transforms.py:168*

#### `PatchInfo`
**Module:** `datasets.patch_transforms`

Information about a patch location in the full image.

*📁 src/dl_techniques/datasets/patch_transforms.py:30*

#### `PatchPrediction`
**Module:** `datasets.patch_transforms`

Predictions from a single patch.

*📁 src/dl_techniques/datasets/patch_transforms.py:75*

#### `PipelineConfig`
**Module:** `datasets.time_series.config`

Configuration for tf.data pipeline construction.

*📁 src/dl_techniques/datasets/time_series/config.py:163*

#### `PoincareMath`
**Module:** `utils.geometry.poincare_math`

Stateless utility class for Poincaré Ball hyperbolic geometry operations.

*📁 src/dl_techniques/utils/geometry/poincare_math.py:31*

#### `ResultAggregator`
**Module:** `datasets.patch_transforms`

Main class for aggregating all multi-task results.

*📁 src/dl_techniques/datasets/patch_transforms.py:501*

#### `SegmentationStitcher`
**Module:** `datasets.patch_transforms`

Stitches segmentation patches into full image masks.

*📁 src/dl_techniques/datasets/patch_transforms.py:348*

#### `SimpleCharTokenizer`
**Module:** `datasets.vqa_dataset`

Simple character-level tokenizer for demonstration purposes.

*Inherits from: `BaseTokenizer`*

*📁 src/dl_techniques/datasets/vqa_dataset.py:53*

#### `TabularDataProcessor`
**Module:** `datasets.tabular`

Preprocessor for tabular data compatible with TabM models.

*📁 src/dl_techniques/datasets/tabular.py:13*

#### `TensorFlowNativePatchSampler`
**Module:** `datasets.sut`

TensorFlow-native patch sampler with vectorized operations.

*📁 src/dl_techniques/datasets/sut.py:202*

#### `TiktokenPreprocessor`
**Module:** `utils.tokenizer`

Callable preprocessor adapting Tiktoken for BERT-style model inputs.

*📁 src/dl_techniques/utils/tokenizer.py:86*

#### `TimeSeriesConfig`
**Module:** `datasets.time_series.config`

Configuration for Time Series Datasets.

*📁 src/dl_techniques/datasets/time_series/config.py:26*

#### `TimeSeriesGenerator`
**Module:** `datasets.time_series.generator`

Generator for diverse time series patterns for machine learning experiments.

*📁 src/dl_techniques/datasets/time_series/generator.py:287*

#### `TimeSeriesGeneratorConfig`
**Module:** `datasets.time_series.generator`

Configuration class for time series generation.

*📁 src/dl_techniques/datasets/time_series/generator.py:234*

#### `TimeSeriesNormalizer`
**Module:** `datasets.time_series.normalizer`

Normalizer for time series data with proper scaling and inverse scaling.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:42*

#### `TrainingConfig`
**Module:** `utils.train`

Configuration class for model training parameters.

*📁 src/dl_techniques/utils/train.py:17*

#### `UniversalDatasetLoader`
**Module:** `datasets.universal_dataset_loader`

A generic utility to stream data from Hugging Face datasets.

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:62*

#### `VQADataProcessor`
**Module:** `datasets.vqa_dataset`

Data processor for Vision Question Answering datasets.

*📁 src/dl_techniques/datasets/vqa_dataset.py:96*

#### `VQADataSequence`
**Module:** `datasets.vqa_dataset`

Keras Sequence for VQA data loading and batching.

*Inherits from: `keras.utils.Sequence`*

*📁 src/dl_techniques/datasets/vqa_dataset.py:364*

#### `VisualizationConfig`
**Module:** `utils.visualization_manager`

Configuration for visualization parameters.

*📁 src/dl_techniques/utils/visualization_manager.py:29*

#### `VisualizationManager`
**Module:** `utils.visualization_manager`

Manager for handling visualization creation and saving.

*📁 src/dl_techniques/utils/visualization_manager.py:56*

#### `WindowConfig`
**Module:** `datasets.time_series.config`

Configuration for sliding window generation.

*📁 src/dl_techniques/datasets/time_series/config.py:82*

### Visualization Classes

#### `ActivationData`
**Module:** `visualization.data_nn`

Container for neural network activation data.

*📁 src/dl_techniques/visualization/data_nn.py:57*

#### `ActivationVisualization`
**Module:** `visualization.data_nn`

Visualize neural network activations.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:785*

#### `ClassBalanceVisualization`
**Module:** `visualization.data_nn`

Visualize class balance and imbalance in datasets.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:472*

#### `ClassificationReportVisualization`
**Module:** `visualization.classification`

Visual representation of classification reports.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/classification.py:361*

#### `ClassificationResults`
**Module:** `visualization.classification`

Container for classification results.

*📁 src/dl_techniques/visualization/classification.py:39*

#### `ColorScheme`
**Module:** `visualization.core`

Color scheme configuration for visualizations.

*📁 src/dl_techniques/visualization/core.py:50*

#### `CompositeVisualization`
**Module:** `visualization.core`

Base class for composite visualizations that combine multiple plots.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/core.py:311*

#### `ConfusionMatrixVisualization`
**Module:** `visualization.classification`

Enhanced confusion matrix visualization.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/classification.py:61*

#### `ConvergenceAnalysis`
**Module:** `visualization.training_performance`

Comprehensive convergence analysis visualization.

*Inherits from: `CompositeVisualization`*

*📁 src/dl_techniques/visualization/training_performance.py:412*

#### `DataDistributionAnalysis`
**Module:** `visualization.data_nn`

Comprehensive data distribution analysis.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:358*

#### `DatasetInfo`
**Module:** `visualization.data_nn`

Container for dataset information.

*📁 src/dl_techniques/visualization/data_nn.py:44*

#### `ErrorAnalysisDashboard`
**Module:** `visualization.classification`

Comprehensive error analysis dashboard.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/classification.py:637*

#### `FeatureMapVisualization`
**Module:** `visualization.data_nn`

Visualize convolutional feature maps.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:1123*

#### `ForecastVisualization`
**Module:** `visualization.time_series`

Comprehensive visualization for time series forecasts.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/time_series.py:51*

#### `GenericMatrixVisualization`
**Module:** `visualization.data_nn`

Visualize a generic 2D matrix as a heatmap.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:1403*

#### `GradientData`
**Module:** `visualization.data_nn`

Container for neural network gradient data.

*📁 src/dl_techniques/visualization/data_nn.py:75*

#### `GradientTopologyData`
**Module:** `visualization.data_nn`

Container for gradient topology data.

*📁 src/dl_techniques/visualization/data_nn.py:84*

#### `GradientTopologyVisualization`
**Module:** `visualization.data_nn`

Visualize gradient flow through the model's topology as a heatmap.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:1332*

#### `GradientTopologyVisualizer`
**Module:** `visualization.data_nn`

Visualizes model topology and gradient flow as a heatmap.

*📁 src/dl_techniques/visualization/data_nn.py:112*

#### `GradientVisualization`
**Module:** `visualization.data_nn`

Visualize gradients during training.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:1200*

#### `ImageComparisonVisualization`
**Module:** `visualization.data_nn`

Compare multiple images side-by-side.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:1476*

#### `ImageData`
**Module:** `visualization.data_nn`

Container for comparing multiple images.

*📁 src/dl_techniques/visualization/data_nn.py:102*

#### `LearningRateScheduleVisualization`
**Module:** `visualization.training_performance`

Visualization for learning rate schedules.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/training_performance.py:184*

#### `MatrixData`
**Module:** `visualization.data_nn`

Container for a generic 2D matrix for visualization.

*📁 src/dl_techniques/visualization/data_nn.py:91*

#### `ModelComparison`
**Module:** `visualization.training_performance`

Container for comparing multiple models.

*📁 src/dl_techniques/visualization/training_performance.py:46*

#### `ModelComparisonBarChart`
**Module:** `visualization.training_performance`

Bar chart comparison of model metrics.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/training_performance.py:259*

#### `MultiModelClassification`
**Module:** `visualization.classification`

Container for comparing multiple classification models.

*📁 src/dl_techniques/visualization/classification.py:50*

#### `MultiModelRegression`
**Module:** `visualization.regression`

Container for comparing multiple regression models.

*📁 src/dl_techniques/visualization/regression.py:49*

#### `NetworkArchitectureVisualization`
**Module:** `visualization.data_nn`

Visualize neural network architecture.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:633*

#### `OverfittingAnalysis`
**Module:** `visualization.training_performance`

Visualization for detecting and analyzing overfitting.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/training_performance.py:547*

#### `PerClassAnalysis`
**Module:** `visualization.classification`

Detailed per-class performance analysis.

*Inherits from: `CompositeVisualization`*

*📁 src/dl_techniques/visualization/classification.py:461*

#### `PerformanceDashboard`
**Module:** `visualization.training_performance`

Comprehensive performance dashboard.

*Inherits from: `CompositeVisualization`*

*📁 src/dl_techniques/visualization/training_performance.py:686*

#### `PerformanceRadarChart`
**Module:** `visualization.training_performance`

Radar chart for multi-metric model comparison.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/training_performance.py:339*

#### `PlotConfig`
**Module:** `visualization.core`

Configuration for plot appearance and behavior.

*📁 src/dl_techniques/visualization/core.py:79*

#### `PlotStyle`
**Module:** `visualization.core`

Available plot styles.

*Inherits from: `Enum`*

*📁 src/dl_techniques/visualization/core.py:40*

#### `PredictionErrorVisualization`
**Module:** `visualization.regression`

Visualize Predicted vs. Actual values.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/regression.py:60*

#### `QQPlotVisualization`
**Module:** `visualization.regression`

Q-Q Plot to check for normality of residuals.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/regression.py:233*

#### `ROCPRCurves`
**Module:** `visualization.classification`

ROC and Precision-Recall curves visualization.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/classification.py:209*

#### `RegressionEvaluationDashboard`
**Module:** `visualization.regression`

Comprehensive regression analysis dashboard.

*Inherits from: `CompositeVisualization`*

*📁 src/dl_techniques/visualization/regression.py:280*

#### `RegressionResults`
**Module:** `visualization.regression`

Container for regression evaluation results.

*📁 src/dl_techniques/visualization/regression.py:34*

#### `ResidualDistributionVisualization`
**Module:** `visualization.regression`

Visualize the distribution of residuals.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/regression.py:187*

#### `ResidualsPlotVisualization`
**Module:** `visualization.regression`

Visualize residuals against predicted values.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/regression.py:126*

#### `TimeSeriesEvaluationResults`
**Module:** `visualization.time_series`

Container for time series forecasting evaluation results.

*📁 src/dl_techniques/visualization/time_series.py:26*

#### `TrainingCurvesVisualization`
**Module:** `visualization.training_performance`

Visualization for training curves with multiple metrics.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/training_performance.py:59*

#### `TrainingHistory`
**Module:** `visualization.training_performance`

Container for training history data.

*📁 src/dl_techniques/visualization/training_performance.py:26*

#### `VisualizationContext`
**Module:** `visualization.core`

Context information for visualization generation.

*📁 src/dl_techniques/visualization/core.py:198*

#### `VisualizationManager`
**Module:** `visualization.core`

Central manager for all visualizations.

*📁 src/dl_techniques/visualization/core.py:382*

#### `VisualizationPlugin`
**Module:** `visualization.core`

Abstract base class for visualization plugins.

*Inherits from: `ABC`*

*📁 src/dl_techniques/visualization/core.py:221*

#### `WeightData`
**Module:** `visualization.data_nn`

Container for neural network weight data.

*📁 src/dl_techniques/visualization/data_nn.py:66*

#### `WeightVisualization`
**Module:** `visualization.data_nn`

Visualize neural network weights.

*Inherits from: `VisualizationPlugin`*

*📁 src/dl_techniques/visualization/data_nn.py:908*

## Functions (3274)

### Analyzer Functions

#### `add_non_serializable_field(self, field_name)`
**Module:** `analyzer.data_types`

Add a field to the non-serializable set.

*📁 src/dl_techniques/analyzer/data_types.py:106*

#### `analyze(self, data, analysis_types)`
**Module:** `analyzer.model_analyzer`

Run comprehensive or selected analyses on models.

*📁 src/dl_techniques/analyzer/model_analyzer.py:347*

#### `analyze(self, results, data, cache)`
**Module:** `analyzer.analyzers.base`

Perform the analysis and update results.

*📁 src/dl_techniques/analyzer/analyzers/base.py:30*

#### `analyze(self, results, data, cache)`
**Module:** `analyzer.analyzers.spectral_analyzer`

Perform comprehensive spectral analysis on all models.

*📁 src/dl_techniques/analyzer/analyzers/spectral_analyzer.py:116*

#### `analyze(self, results, data, cache)`
**Module:** `analyzer.analyzers.training_dynamics_analyzer`

Analyze training history to understand how models learned.

*📁 src/dl_techniques/analyzer/analyzers/training_dynamics_analyzer.py:92*

#### `analyze(self, results, data, cache)`
**Module:** `analyzer.analyzers.information_flow_analyzer`

Analyze information flow using forward hooks for subclassed model compatibility.

*📁 src/dl_techniques/analyzer/analyzers/information_flow_analyzer.py:71*

#### `analyze(self, results, data, cache)`
**Module:** `analyzer.analyzers.calibration_analyzer`

Analyze model confidence and calibration with consolidated metric storage.

*📁 src/dl_techniques/analyzer/analyzers/calibration_analyzer.py:99*

#### `analyze(self, results, data, cache)`
**Module:** `analyzer.analyzers.weight_analyzer`

Analyze weight distributions with improved visualizations.

*📁 src/dl_techniques/analyzer/analyzers/weight_analyzer.py:95*

#### `calculate_concentration_metrics(weight_matrix, num_eigenvectors)`
**Module:** `analyzer.spectral_metrics`

Calculate comprehensive concentration metrics for a weight matrix.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:621*

#### `calculate_dominance_ratio(evals)`
**Module:** `analyzer.spectral_metrics`

Calculate the ratio of the largest eigenvalue to the sum of all other eigenvalues.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:413*

#### `calculate_gini_coefficient(evals)`
**Module:** `analyzer.spectral_metrics`

Calculate the Gini coefficient of the eigenvalue distribution.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:383*

#### `calculate_glorot_normalization_factor(N, M, rf)`
**Module:** `analyzer.spectral_metrics`

Calculate the Glorot normalization factor for weight initialization.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:772*

#### `calculate_matrix_entropy(singular_values, N)`
**Module:** `analyzer.spectral_metrics`

Calculate the matrix entropy from singular values.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:287*

#### `calculate_participation_ratio(vector)`
**Module:** `analyzer.spectral_metrics`

Calculate the Participation Ratio (PR) of a vector.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:437*

#### `calculate_spectral_metrics(evals, alpha)`
**Module:** `analyzer.spectral_metrics`

Calculate various spectral metrics from eigenvalues.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:345*

#### `compute_adaptive_ece(y_true, y_prob, n_bins)`
**Module:** `analyzer.calibration_metrics`

Compute Adaptive Expected Calibration Error (AECE) using equal-mass bins.

*📁 src/dl_techniques/analyzer/calibration_metrics.py:135*

#### `compute_brier_score(y_true_onehot, y_prob)`
**Module:** `analyzer.calibration_metrics`

Compute Brier Score for multiclass probabilistic predictions.

*📁 src/dl_techniques/analyzer/calibration_metrics.py:267*

#### `compute_brier_score_decomposition(y_true, y_prob, n_bins)`
**Module:** `analyzer.calibration_metrics`

Decompose the Brier Score into Reliability, Resolution, and Uncertainty.

*📁 src/dl_techniques/analyzer/calibration_metrics.py:296*

#### `compute_detX_constraint(evals)`
**Module:** `analyzer.spectral_metrics`

Identify the number of eigenvalues necessary to satisfy det(X) = 1.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:722*

#### `compute_ece(y_true, y_prob, n_bins)`
**Module:** `analyzer.calibration_metrics`

Compute Expected Calibration Error (ECE) using equal-width bins.

*📁 src/dl_techniques/analyzer/calibration_metrics.py:97*

#### `compute_eigenvalues(weight_matrices, N, M, n_comp, normalize)`
**Module:** `analyzer.spectral_metrics`

Compute the eigenvalues for all weight matrices combined.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:90*

#### `compute_mce(y_true, y_prob, n_bins)`
**Module:** `analyzer.calibration_metrics`

Compute Maximum Calibration Error (MCE).

*📁 src/dl_techniques/analyzer/calibration_metrics.py:187*

#### `compute_prediction_entropy_stats(y_prob)`
**Module:** `analyzer.calibration_metrics`

Compute prediction entropy statistics to measure model uncertainty.

*📁 src/dl_techniques/analyzer/calibration_metrics.py:352*

#### `compute_reliability_data(y_true, y_prob, n_bins)`
**Module:** `analyzer.calibration_metrics`

Compute data for reliability diagram visualization.

*📁 src/dl_techniques/analyzer/calibration_metrics.py:223*

#### `compute_weight_statistics(model, include_layers)`
**Module:** `analyzer.spectral_utils`

Computes basic statistics for weights in model layers.

*📁 src/dl_techniques/analyzer/spectral_utils.py:311*

#### `convert_numpy(obj)`
**Module:** `analyzer.model_analyzer`

Recursively convert numpy types and pandas DataFrames to JSON-serializable formats.

*📁 src/dl_techniques/analyzer/model_analyzer.py:658*

#### `create_pareto_analysis(self, save_plot)`
**Module:** `analyzer.model_analyzer`

Create Pareto front analysis for hyperparameter sweep scenarios.

*📁 src/dl_techniques/analyzer/model_analyzer.py:804*

#### `create_smoothed_model(self, model_name, method, percent, save_path)`
**Module:** `analyzer.model_analyzer`

Create a smoothed version of a model using SVD truncation.

*📁 src/dl_techniques/analyzer/model_analyzer.py:937*

#### `create_summary_dashboard(self)`
**Module:** `analyzer.model_analyzer`

Create summary dashboard visualization.

*📁 src/dl_techniques/analyzer/model_analyzer.py:582*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.base`

Create all visualizations for this analyzer.

*📁 src/dl_techniques/analyzer/visualizers/base.py:193*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.summary_visualizer`

Create a comprehensive summary dashboard with training insights and single legend.

*📁 src/dl_techniques/analyzer/visualizers/summary_visualizer.py:101*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.weight_visualizer`

Create unified Weight Learning Journey visualization with single legend.

*📁 src/dl_techniques/analyzer/visualizers/weight_visualizer.py:79*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.information_flow_visualizer`

Create information flow visualizations with single legend.

*📁 src/dl_techniques/analyzer/visualizers/information_flow_visualizer.py:80*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.calibration_visualizer`

Create unified confidence and calibration visualizations with single legend.

*📁 src/dl_techniques/analyzer/visualizers/calibration_visualizer.py:80*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.training_dynamics_visualizer`

Create comprehensive training dynamics visualizations with single legend.

*📁 src/dl_techniques/analyzer/visualizers/training_dynamics_visualizer.py:73*

#### `create_visualizations(self)`
**Module:** `analyzer.visualizers.spectral_visualizer`

Main entry point for generating all spectral visualizations.

*📁 src/dl_techniques/analyzer/visualizers/spectral_visualizer.py:38*

#### `create_weight_visualization(model, layer_index, figsize, cmap)`
**Module:** `analyzer.spectral_utils`

Creates a visualization of weight matrices for a specific layer.

*📁 src/dl_techniques/analyzer/spectral_utils.py:233*

#### `extract_conv_filters(layer, filter_indices)`
**Module:** `analyzer.spectral_utils`

Extracts convolutional filters from a convolutional layer.

*📁 src/dl_techniques/analyzer/spectral_utils.py:369*

#### `find_critical_weights(weight_matrix, eigenvectors, eigenvalues, threshold)`
**Module:** `analyzer.spectral_metrics`

Find individual weights that contribute most to top eigenvectors.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:564*

#### `find_metric_in_history(history, patterns, exclude_prefixes)`
**Module:** `analyzer.utils`

Robustly find a metric in training history by checking multiple possible names.

*📁 src/dl_techniques/analyzer/utils.py:255*

#### `find_model_metric(model_metrics, metric_keys, default)`
**Module:** `analyzer.utils`

Helper function to find a metric value from model metrics with fallback chain.

*📁 src/dl_techniques/analyzer/utils.py:354*

#### `find_pareto_front(costs1, costs2)`
**Module:** `analyzer.utils`

Find indices of Pareto optimal points (maximizing both objectives).

*📁 src/dl_techniques/analyzer/utils.py:384*

#### `fit_powerlaw(evals, xmin)`
**Module:** `analyzer.spectral_metrics`

Fit eigenvalues to a power-law distribution using a robust xmin search.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:175*

#### `forward_hook(layer, inputs, outputs)`
**Module:** `analyzer.analyzers.information_flow_analyzer`

*📁 src/dl_techniques/analyzer/analyzers/information_flow_analyzer.py:114*

#### `from_object(cls, data)`
**Module:** `analyzer.data_types`

Create from object with x_test and y_test attributes.

*📁 src/dl_techniques/analyzer/data_types.py:23*

#### `from_tuple(cls, data)`
**Module:** `analyzer.data_types`

Create from tuple.

*📁 src/dl_techniques/analyzer/data_types.py:18*

#### `get_figure_size(self, scale)`
**Module:** `analyzer.config`

Get figure size with optional scaling.

*📁 src/dl_techniques/analyzer/config.py:86*

#### `get_layer_weights_and_bias(layer)`
**Module:** `analyzer.spectral_utils`

Extract weights and biases from a Keras layer.

*📁 src/dl_techniques/analyzer/spectral_utils.py:116*

#### `get_serializable_dict(self)`
**Module:** `analyzer.data_types`

Get a dictionary representation excluding non-serializable fields.

*📁 src/dl_techniques/analyzer/data_types.py:110*

#### `get_summary_statistics(self)`
**Module:** `analyzer.model_analyzer`

Get summary statistics of the analysis.

*📁 src/dl_techniques/analyzer/model_analyzer.py:717*

#### `get_top_eigenvectors(weight_matrix, k, method)`
**Module:** `analyzer.spectral_metrics`

Calculate the top k left singular vectors of a weight matrix.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:471*

#### `get_weight_matrices(weights, layer_type)`
**Module:** `analyzer.spectral_utils`

Extract weight matrices from a layer's weights.

*📁 src/dl_techniques/analyzer/spectral_utils.py:167*

#### `infer_layer_type(layer)`
**Module:** `analyzer.spectral_utils`

Determine the layer type for a given Keras layer.

*📁 src/dl_techniques/analyzer/spectral_utils.py:65*

#### `jensen_shannon_distance(p, q)`
**Module:** `analyzer.spectral_metrics`

Calculate Jensen-Shannon distance between two distributions.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:673*

#### `lighten_color(color, factor)`
**Module:** `analyzer.utils`

Lighten a color by interpolating towards white.

*📁 src/dl_techniques/analyzer/utils.py:377*

#### `normalize_metric(values, higher_better)`
**Module:** `analyzer.utils`

Normalize metric values to 0-1 range.

*📁 src/dl_techniques/analyzer/utils.py:412*

#### `recursively_get_layers(layer_or_model)`
**Module:** `analyzer.utils`

Recursively traverses a Keras model or layer to get a flat list of all layers.

*📁 src/dl_techniques/analyzer/utils.py:487*

#### `requires_data(self)`
**Module:** `analyzer.analyzers.base`

Check if this analyzer requires input data.

*📁 src/dl_techniques/analyzer/analyzers/base.py:43*

#### `requires_data(self)`
**Module:** `analyzer.analyzers.spectral_analyzer`

Spectral analysis is data-independent.

*📁 src/dl_techniques/analyzer/analyzers/spectral_analyzer.py:112*

#### `requires_data(self)`
**Module:** `analyzer.analyzers.training_dynamics_analyzer`

Training dynamics analysis doesn't require input data.

*📁 src/dl_techniques/analyzer/analyzers/training_dynamics_analyzer.py:88*

#### `requires_data(self)`
**Module:** `analyzer.analyzers.information_flow_analyzer`

Information flow analysis requires input data.

*📁 src/dl_techniques/analyzer/analyzers/information_flow_analyzer.py:67*

#### `requires_data(self)`
**Module:** `analyzer.analyzers.calibration_analyzer`

Calibration analysis requires input data.

*📁 src/dl_techniques/analyzer/analyzers/calibration_analyzer.py:95*

#### `requires_data(self)`
**Module:** `analyzer.analyzers.weight_analyzer`

Weight analysis doesn't require input data.

*📁 src/dl_techniques/analyzer/analyzers/weight_analyzer.py:91*

#### `rescale_eigenvalues(evals)`
**Module:** `analyzer.spectral_metrics`

Rescale eigenvalues by their L2 norm to sqrt(N).

*📁 src/dl_techniques/analyzer/spectral_metrics.py:707*

#### `safe_set_xticklabels(ax, labels, rotation, max_labels)`
**Module:** `analyzer.utils`

Safely set x-tick labels with proper handling.

*📁 src/dl_techniques/analyzer/utils.py:214*

#### `safe_tight_layout(fig)`
**Module:** `analyzer.utils`

Safely apply tight_layout with error handling.

*📁 src/dl_techniques/analyzer/utils.py:229*

#### `sample(data, n_samples)`
**Module:** `analyzer.utils`

Sample a subset of data from the input, handling various formats.

*📁 src/dl_techniques/analyzer/utils.py:36*

#### `save_results(self, filename)`
**Module:** `analyzer.model_analyzer`

Save analysis results to JSON file.

*📁 src/dl_techniques/analyzer/model_analyzer.py:594*

#### `setup_plotting_style(self)`
**Module:** `analyzer.config`

Set up matplotlib style based on configuration.

*📁 src/dl_techniques/analyzer/config.py:90*

#### `smooth_curve(values, window_size)`
**Module:** `analyzer.utils`

Apply smoothing to a curve using a moving average.

*📁 src/dl_techniques/analyzer/utils.py:241*

#### `smooth_matrix(W, n_comp)`
**Module:** `analyzer.spectral_metrics`

Apply SVD smoothing to a weight matrix by zeroing out small singular values.

*📁 src/dl_techniques/analyzer/spectral_metrics.py:743*

#### `truncate_model_name(name, max_len, filler)`
**Module:** `analyzer.utils`

Truncates a string by replacing middle characters with a filler.

*📁 src/dl_techniques/analyzer/utils.py:472*

#### `validate_training_history(history)`
**Module:** `analyzer.utils`

Validate training history and return a report of potential issues.

*📁 src/dl_techniques/analyzer/utils.py:439*

### Constraints Functions

#### `from_config(cls, config)`
**Module:** `constraints.value_range_constraint`

Creates a constraint from its configuration dictionary.

*📁 src/dl_techniques/constraints/value_range_constraint.py:165*

#### `get_config(self)`
**Module:** `constraints.value_range_constraint`

Return the configuration of the constraint for serialization.

*📁 src/dl_techniques/constraints/value_range_constraint.py:149*

### Initializers Functions

#### `create_haar_depthwise_conv2d(input_shape, channel_multiplier, scale, use_bias, kernel_regularizer, trainable, name)`
**Module:** `initializers.haar_wavelet_initializer`

Create a Haar wavelet depthwise convolution layer.

*📁 src/dl_techniques/initializers/haar_wavelet_initializer.py:212*

#### `from_config(cls, config)`
**Module:** `initializers.haar_wavelet_initializer`

Create initializer from configuration.

*📁 src/dl_techniques/initializers/haar_wavelet_initializer.py:197*

#### `from_config(cls, config)`
**Module:** `initializers.he_orthonormal_initializer`

Create an initializer from its configuration.

*📁 src/dl_techniques/initializers/he_orthonormal_initializer.py:295*

#### `from_config(cls, config)`
**Module:** `initializers.orthonormal_initializer`

Create an initializer from its configuration.

*📁 src/dl_techniques/initializers/orthonormal_initializer.py:310*

#### `get_config(self)`
**Module:** `initializers.haar_wavelet_initializer`

Get configuration for serialization.

*📁 src/dl_techniques/initializers/haar_wavelet_initializer.py:183*

#### `get_config(self)`
**Module:** `initializers.he_orthonormal_initializer`

Get the configuration of the initializer.

*📁 src/dl_techniques/initializers/he_orthonormal_initializer.py:279*

#### `get_config(self)`
**Module:** `initializers.hypersphere_orthogonal_initializer`

Get initializer configuration for serialization.

*📁 src/dl_techniques/initializers/hypersphere_orthogonal_initializer.py:270*

#### `get_config(self)`
**Module:** `initializers.orthonormal_initializer`

Get the configuration of the initializer.

*📁 src/dl_techniques/initializers/orthonormal_initializer.py:294*

### Layers Functions

#### `activation_wrapper(activation)`
**Module:** `layers.conv2d_builder`

Creates and returns a Keras activation layer based on the specified activation type.

*📁 src/dl_techniques/layers/conv2d_builder.py:67*

#### `all_tasks(cls)`
**Module:** `layers.nlp_heads.task_types`

Get all available task types.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:131*

#### `all_tasks(cls)`
**Module:** `layers.vision_heads.task_types`

Get all available task types.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:140*

#### `all_tasks(cls)`
**Module:** `layers.vlm_heads.task_types`

Get all available task types.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:98*

#### `analyze_evidence_support(outputs, input_tokens, vocab)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Analyze evidence support for generated tokens for the first item in a batch.

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:831*

#### `apply_shell_scaling(self, normalized_features, confidence)`
**Module:** `layers.experimental.band_rms_ood`

Apply confidence-driven shell scaling to normalized features.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:364*

#### `body(i, state, outputs)`
**Module:** `layers.gated_delta_net`

*📁 src/dl_techniques/layers/gated_delta_net.py:426*

#### `build(self, input_shape)`
**Module:** `layers.mobile_one_block`

Create weights and build sub-layers.

*📁 src/dl_techniques/layers/mobile_one_block.py:260*

#### `build(self, input_shape)`
**Module:** `layers.depthwise_separable_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/depthwise_separable_block.py:284*

#### `build(self, input_shape)`
**Module:** `layers.sampling`

Build the layer and validate input shapes.

*📁 src/dl_techniques/layers/sampling.py:154*

#### `build(self, input_shape)`
**Module:** `layers.hanc_layer`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/hanc_layer.py:222*

#### `build(self, input_shape)`
**Module:** `layers.laplacian_filter`

Build the layer and explicitly build the Gaussian filter.

*📁 src/dl_techniques/layers/laplacian_filter.py:162*

#### `build(self, input_shape)`
**Module:** `layers.laplacian_filter`

Build the layer and initialize components based on the selected method.

*📁 src/dl_techniques/layers/laplacian_filter.py:400*

#### `build(self, input_shape)`
**Module:** `layers.multi_level_feature_compilation`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/multi_level_feature_compilation.py:263*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/yolo12_blocks.py:168*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Build the attention components and all sub-layers.

*📁 src/dl_techniques/layers/yolo12_blocks.py:355*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Build the attention block components and all sub-layers.

*📁 src/dl_techniques/layers/yolo12_blocks.py:608*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Build the bottleneck components and all sub-layers.

*📁 src/dl_techniques/layers/yolo12_blocks.py:759*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Build the C3k2 block components and all sub-layers.

*📁 src/dl_techniques/layers/yolo12_blocks.py:927*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Build the A2C2f block components and all sub-layers.

*📁 src/dl_techniques/layers/yolo12_blocks.py:1120*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_heads`

Build detection head branches for each input scale.

*📁 src/dl_techniques/layers/yolo12_heads.py:177*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_heads`

Build segmentation head with progressive upsampling to full resolution.

*📁 src/dl_techniques/layers/yolo12_heads.py:514*

#### `build(self, input_shape)`
**Module:** `layers.yolo12_heads`

Build classification head.

*📁 src/dl_techniques/layers/yolo12_heads.py:968*

#### `build(self, input_shape)`
**Module:** `layers.radial_basis_function`

Create layer weights.

*📁 src/dl_techniques/layers/radial_basis_function.py:139*

#### `build(self, input_shape)`
**Module:** `layers.spatial_layer`

Creates the normalized coordinate grid during layer building.

*📁 src/dl_techniques/layers/spatial_layer.py:182*

#### `build(self, input_shape)`
**Module:** `layers.one_hot_encoding`

Build the layer by computing cumulative cardinalities for efficient indexing.

*📁 src/dl_techniques/layers/one_hot_encoding.py:75*

#### `build(self, input_shape)`
**Module:** `layers.neuro_grid`

Create the grid weights, temperature parameter, and build projection layers.

*📁 src/dl_techniques/layers/neuro_grid.py:550*

#### `build(self, input_shape)`
**Module:** `layers.gaussian_filter`

Build the Gaussian kernel weights based on input shape.

*📁 src/dl_techniques/layers/gaussian_filter.py:154*

#### `build(self, input_shape)`
**Module:** `layers.dynamic_conv2d`

Build the layer by creating all sub-layers.

*📁 src/dl_techniques/layers/dynamic_conv2d.py:297*

#### `build(self, input_shape)`
**Module:** `layers.shearlet_transform`

Build the layer and create shearlet filter bank.

*📁 src/dl_techniques/layers/shearlet_transform.py:153*

#### `build(self, input_shape)`
**Module:** `layers.squeeze_excitation`

Build the layer and create all sub-layers.

*📁 src/dl_techniques/layers/squeeze_excitation.py:214*

#### `build(self, input_shape)`
**Module:** `layers.standard_blocks`

Build sub-layers explicitly for proper serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:327*

#### `build(self, input_shape)`
**Module:** `layers.standard_blocks`

Build sub-layers explicitly for proper serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:565*

#### `build(self, input_shape)`
**Module:** `layers.standard_blocks`

Build sub-layers with input-dependent configuration.

*📁 src/dl_techniques/layers/standard_blocks.py:729*

#### `build(self, input_shape)`
**Module:** `layers.standard_blocks`

Build sub-layers explicitly for proper serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:997*

#### `build(self, input_shape)`
**Module:** `layers.standard_blocks`

Build sub-layers explicitly for proper serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:1269*

#### `build(self, input_shape)`
**Module:** `layers.sequence_pooling`

Build attention layers based on input shape.

*📁 src/dl_techniques/layers/sequence_pooling.py:191*

#### `build(self, input_shape)`
**Module:** `layers.sequence_pooling`

Build position weights.

*📁 src/dl_techniques/layers/sequence_pooling.py:349*

#### `build(self, input_shape)`
**Module:** `layers.sequence_pooling`

Build the layer and all sub-layers.

*📁 src/dl_techniques/layers/sequence_pooling.py:560*

#### `build(self, input_shape)`
**Module:** `layers.blt_core`

Build the model components.

*📁 src/dl_techniques/layers/blt_core.py:216*

#### `build(self, input_shape)`
**Module:** `layers.fnet_encoder_block`

Build encoder block and all sub-layers with proper shape inference.

*📁 src/dl_techniques/layers/fnet_encoder_block.py:183*

#### `build(self, input_shape)`
**Module:** `layers.complex_layers`

Create the layer's weights.

*📁 src/dl_techniques/layers/complex_layers.py:290*

#### `build(self, input_shape)`
**Module:** `layers.complex_layers`

Create the layer's weights.

*📁 src/dl_techniques/layers/complex_layers.py:512*

#### `build(self, input_shape)`
**Module:** `layers.anchor_generator`

Create the layer's anchor and stride weights.

*📁 src/dl_techniques/layers/anchor_generator.py:187*

#### `build(self, input_shape)`
**Module:** `layers.haar_wavelet_decomposition`

Build the layer by determining input dimensionality.

*📁 src/dl_techniques/layers/haar_wavelet_decomposition.py:100*

#### `build(self, input_shape)`
**Module:** `layers.tabm_blocks`

Build the scaling weights with proper initialization.

*📁 src/dl_techniques/layers/tabm_blocks.py:98*

#### `build(self, input_shape)`
**Module:** `layers.tabm_blocks`

Build the ensemble linear layer weights.

*📁 src/dl_techniques/layers/tabm_blocks.py:182*

#### `build(self, input_shape)`
**Module:** `layers.tabm_blocks`

Build the parallel linear layer weights.

*📁 src/dl_techniques/layers/tabm_blocks.py:316*

#### `build(self, input_shape)`
**Module:** `layers.tabm_blocks`

Build the MLP block layers.

*📁 src/dl_techniques/layers/tabm_blocks.py:422*

#### `build(self, input_shape)`
**Module:** `layers.tabm_blocks`

Build the backbone MLP blocks.

*📁 src/dl_techniques/layers/tabm_blocks.py:542*

#### `build(self, input_shape)`
**Module:** `layers.kan_linear`

Create layer weights and B-spline grid based on input shape.

*📁 src/dl_techniques/layers/kan_linear.py:200*

#### `build(self, input_shape)`
**Module:** `layers.convnext_v2_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/convnext_v2_block.py:341*

#### `build(self, input_shape)`
**Module:** `layers.io_preparation`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/io_preparation.py:500*

#### `build(self, input_shape)`
**Module:** `layers.convnext_v1_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/convnext_v1_block.py:318*

#### `build(self, input_shape)`
**Module:** `layers.orthoblock`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/orthoblock.py:275*

#### `build(self, input_shape)`
**Module:** `layers.pixel_shuffle`

Validate input shape and build the layer.

*📁 src/dl_techniques/layers/pixel_shuffle.py:129*

#### `build(self, input_shape)`
**Module:** `layers.res_path`

Build the residual blocks and all sub-layers.

*📁 src/dl_techniques/layers/res_path.py:188*

#### `build(self, input_shape)`
**Module:** `layers.mps_layer`

Build the layer weights based on input shape.

*📁 src/dl_techniques/layers/mps_layer.py:249*

#### `build(self, input_shape)`
**Module:** `layers.film`

Build the layer's weights and sub-layers.

*📁 src/dl_techniques/layers/film.py:261*

#### `build(self, input_shape)`
**Module:** `layers.hanc_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/hanc_block.py:272*

#### `build(self, input_shape)`
**Module:** `layers.repmixer_block`

Build the layer and all sub-layers.

*📁 src/dl_techniques/layers/repmixer_block.py:327*

#### `build(self, input_shape)`
**Module:** `layers.repmixer_block`

Build all stem blocks.

*📁 src/dl_techniques/layers/repmixer_block.py:547*

#### `build(self, input_shape)`
**Module:** `layers.bitlinear_layer`

Build the layer weights based on input shape.

*📁 src/dl_techniques/layers/bitlinear_layer.py:391*

#### `build(self, input_shape)`
**Module:** `layers.gaussian_pyramid`

Build all Gaussian filters with appropriate input shapes.

*📁 src/dl_techniques/layers/gaussian_pyramid.py:205*

#### `build(self, input_shape)`
**Module:** `layers.bias_free_conv1d`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:162*

#### `build(self, input_shape)`
**Module:** `layers.bias_free_conv1d`

Build the residual block components.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:407*

#### `build(self, input_shape)`
**Module:** `layers.vector_quantizer`

Create the embedding codebook and EMA variables if needed.

*📁 src/dl_techniques/layers/vector_quantizer.py:160*

#### `build(self, input_shape)`
**Module:** `layers.restricted_boltzmann_machine`

Create RBM weight variables based on input shape.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:217*

#### `build(self, input_shape)`
**Module:** `layers.convolutional_kan`

Create the layer's weights including B-spline control points and combination weights.

*📁 src/dl_techniques/layers/convolutional_kan.py:274*

#### `build(self, input_shape)`
**Module:** `layers.kmeans`

Build the layer weights.

*📁 src/dl_techniques/layers/kmeans.py:280*

#### `build(self, input_shape)`
**Module:** `layers.capsules`

Build the layer based on input shape.

*📁 src/dl_techniques/layers/capsules.py:214*

#### `build(self, input_shape)`
**Module:** `layers.capsules`

Build layer weights based on input shape.

*📁 src/dl_techniques/layers/capsules.py:503*

#### `build(self, input_shape)`
**Module:** `layers.capsules`

Build the layer based on input shape.

*📁 src/dl_techniques/layers/capsules.py:840*

#### `build(self, input_shape)`
**Module:** `layers.gated_delta_net`

Build the layer and all sub-layers.

*📁 src/dl_techniques/layers/gated_delta_net.py:355*

#### `build(self, input_shape)`
**Module:** `layers.bias_free_conv2d`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:182*

#### `build(self, input_shape)`
**Module:** `layers.bias_free_conv2d`

Build the residual block components.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:447*

#### `build(self, input_shape)`
**Module:** `layers.mothnet_blocks`

Create layer weights for excitatory connections.

*📁 src/dl_techniques/layers/mothnet_blocks.py:170*

#### `build(self, input_shape)`
**Module:** `layers.mothnet_blocks`

Create sparse random projection matrix.

*📁 src/dl_techniques/layers/mothnet_blocks.py:426*

#### `build(self, input_shape)`
**Module:** `layers.mothnet_blocks`

Create readout weights.

*📁 src/dl_techniques/layers/mothnet_blocks.py:711*

#### `build(self, input_shape)`
**Module:** `layers.selective_gradient_mask`

Build the layer by validating input shapes.

*📁 src/dl_techniques/layers/selective_gradient_mask.py:128*

#### `build(self, input_shape)`
**Module:** `layers.clahe`

Create the layer's trainable weights.

*📁 src/dl_techniques/layers/clahe.py:184*

#### `build(self, input_shape)`
**Module:** `layers.tversky_projection`

Create the layer's weights: prototypes, features, and contrast params.

*📁 src/dl_techniques/layers/tversky_projection.py:180*

#### `build(self, input_shape)`
**Module:** `layers.hierarchical_mlp_stem`

Build the layer and all its sub-layers dynamically.

*📁 src/dl_techniques/layers/hierarchical_mlp_stem.py:214*

#### `build(self, input_shape)`
**Module:** `layers.layer_scale`

Create the layer's trainable multiplier weights.

*📁 src/dl_techniques/layers/layer_scale.py:219*

#### `build(self, input_shape)`
**Module:** `layers.blt_blocks`

Build the entropy model layers.

*📁 src/dl_techniques/layers/blt_blocks.py:443*

#### `build(self, input_shape)`
**Module:** `layers.blt_blocks`

Build pooling layers.

*📁 src/dl_techniques/layers/blt_blocks.py:781*

#### `build(self, input_shape)`
**Module:** `layers.blt_blocks`

Build local encoder layers.

*📁 src/dl_techniques/layers/blt_blocks.py:1076*

#### `build(self, input_shape)`
**Module:** `layers.blt_blocks`

Build global transformer layers.

*📁 src/dl_techniques/layers/blt_blocks.py:1232*

#### `build(self, input_shape)`
**Module:** `layers.blt_blocks`

Build local decoder layers.

*📁 src/dl_techniques/layers/blt_blocks.py:1412*

#### `build(self, input_shape)`
**Module:** `layers.random_fourier_features`

Create the layer's weights including random features and output projection.

*📁 src/dl_techniques/layers/random_fourier_features.py:264*

#### `build(self, input_shape)`
**Module:** `layers.modality_projection`

Build the modality projection components.

*📁 src/dl_techniques/layers/modality_projection.py:188*

#### `build(self, input_shape)`
**Module:** `layers.sparse_autoencoder`

Build the layer weights.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:235*

#### `build(self, input_shape)`
**Module:** `layers.eomt_mask`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/eomt_mask.py:389*

#### `build(self, input_shape)`
**Module:** `layers.canny`

Create the layer's non-trainable weights (kernels).

*📁 src/dl_techniques/layers/canny.py:179*

#### `build(self, input_shape)`
**Module:** `layers.universal_inverted_bottleneck`

Build the layer's weights and sub-layers.

*📁 src/dl_techniques/layers/universal_inverted_bottleneck.py:424*

#### `build(self, input_shape)`
**Module:** `layers.fractal_block`

Build the FractalBlock and all its sub-layers.

*📁 src/dl_techniques/layers/fractal_block.py:294*

#### `build(self, input_shape)`
**Module:** `layers.rigid_simplex_layer`

Create the layer's weights.

*📁 src/dl_techniques/layers/rigid_simplex_layer.py:278*

#### `build(self, input_shape)`
**Module:** `layers.router`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/router.py:234*

#### `build(self, input_shape)`
**Module:** `layers.transformers.progressive_focused_transformer`

Build layer components.

*📁 src/dl_techniques/layers/transformers/progressive_focused_transformer.py:398*

#### `build(self, input_shape)`
**Module:** `layers.transformers.text_encoder`

Build the text encoder and all its sub-layers.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:642*

#### `build(self, input_shape)`
**Module:** `layers.transformers.text_decoder`

Build the layer and all sub-layers.

*📁 src/dl_techniques/layers/transformers/text_decoder.py:362*

#### `build(self, input_shape)`
**Module:** `layers.transformers.eomt_transformer`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/transformers/eomt_transformer.py:316*

#### `build(self, input_shape)`
**Module:** `layers.transformers.swin_transformer_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/transformers/swin_transformer_block.py:429*

#### `build(self, input_shape)`
**Module:** `layers.transformers.free_transformer`

Build all sub-layers including encoder components if enabled.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:557*

#### `build(self, input_shape)`
**Module:** `layers.transformers.perceiver_transformer`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/transformers/perceiver_transformer.py:247*

#### `build(self, input_shape)`
**Module:** `layers.transformers.swin_conv_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/transformers/swin_conv_block.py:520*

#### `build(self, input_shape)`
**Module:** `layers.transformers.vision_encoder`

Build the vision_heads encoder and all its sub-layers.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:592*

#### `build(self, input_shape)`
**Module:** `layers.transformers.transformer`

Build all sub-layers with appropriate shapes.

*📁 src/dl_techniques/layers/transformers/transformer.py:527*

#### `build(self, input_shape)`
**Module:** `layers.experimental.hierarchical_memory_system`

Build the hierarchical memory system by building all SOM layers.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:249*

#### `build(self, input_shape)`
**Module:** `layers.experimental.field_embeddings`

Create the Lie algebra weight variables.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:137*

#### `build(self, input_shape)`
**Module:** `layers.experimental.field_embeddings`

Create reference vector variable if needed.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:724*

#### `build(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Build the memory store weights.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:193*

#### `build(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Build GNN layers.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:413*

#### `build(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Build temporal encoder layers.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:633*

#### `build(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Build all components of the memory bank.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:793*

#### `build(self, input_shape)`
**Module:** `layers.experimental.mst_correlation_filter`

Build the layer and create trainable weights.

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:146*

#### `build(self, query_shape, value_shape, key_shape)`
**Module:** `layers.experimental.mst_correlation_filter`

Build the layer and initialize sublayers.

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:446*

#### `build(self, input_shape)`
**Module:** `layers.experimental.band_rms_ood`

Build the layer and create weight variables.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:189*

#### `build(self, input_shape)`
**Module:** `layers.experimental.contextual_counter_ffn`

Build the layer and all sub-layers with proper shape inference.

*📁 src/dl_techniques/layers/experimental/contextual_counter_ffn.py:235*

#### `build(self, input_shape)`
**Module:** `layers.experimental.graph_mann`

Create the layer's weights and build sub-layers.

*📁 src/dl_techniques/layers/experimental/graph_mann.py:176*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:231*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:364*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:482*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:603*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:733*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:920*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1013*

#### `build(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1154*

#### `build(self, input_shape)`
**Module:** `layers.time_series.ema_layer`

Build the layer.

*📁 src/dl_techniques/layers/time_series/ema_layer.py:301*

#### `build(self, input_shape)`
**Module:** `layers.time_series.nbeats_blocks`

Build the layer weights and explicitly build all sub-layers.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:324*

#### `build(self, input_shape)`
**Module:** `layers.time_series.nbeats_blocks`

Build the generic block and its sub-layers.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:558*

#### `build(self, input_shape)`
**Module:** `layers.time_series.nbeats_blocks`

Build the trend block with corrected basis functions.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:674*

#### `build(self, input_shape)`
**Module:** `layers.time_series.nbeats_blocks`

Build the seasonality block with corrected basis functions.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:864*

#### `build(self, input_shape)`
**Module:** `layers.time_series.adaptive_lag_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/time_series/adaptive_lag_attention.py:241*

#### `build(self, input_shape)`
**Module:** `layers.time_series.quantile_head_fixed_io`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/time_series/quantile_head_fixed_io.py:151*

#### `build(self, input_shape)`
**Module:** `layers.time_series.deepar_blocks`

Build the layer by explicitly building sub-layers.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:196*

#### `build(self, input_shape)`
**Module:** `layers.time_series.deepar_blocks`

Build the layer by explicitly building sub-layers.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:315*

#### `build(self, input_shape)`
**Module:** `layers.time_series.deepar_blocks`

Build the cell by building the LSTM sub-layer.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:438*

#### `build(self, input_shape)`
**Module:** `layers.time_series.quantile_head_variable_io`

Build the layer and initialize all sub-layer weights.

*📁 src/dl_techniques/layers/time_series/quantile_head_variable_io.py:242*

#### `build(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

Build the cell's weight matrices.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:167*

#### `build(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

Build the RNN layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:407*

#### `build(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

Build the cell's weight matrices.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:578*

#### `build(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

Build the RNN layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:850*

#### `build(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

Build all sub-layers.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1035*

#### `build(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

Build all sub-layers.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1251*

#### `build(self, input_shape)`
**Module:** `layers.time_series.nbeatsx_blocks`

*📁 src/dl_techniques/layers/time_series/nbeatsx_blocks.py:89*

#### `build(self, input_shape)`
**Module:** `layers.time_series.forecasting_layers`

Build the complexity analyzer sub-network.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:188*

#### `build(self, input_shape)`
**Module:** `layers.time_series.forecasting_layers`

Build the projection layer and calibration score.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:359*

#### `build(self, input_shape)`
**Module:** `layers.time_series.temporal_fusion`

Build the layer weights and sublayers based on input shape.

*📁 src/dl_techniques/layers/time_series/temporal_fusion.py:278*

#### `build(self, input_shape)`
**Module:** `layers.time_series.mixed_sequential_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/time_series/mixed_sequential_block.py:441*

#### `build(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Build the layer.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:223*

#### `build(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Build the layer.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:395*

#### `build(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Build all PRISM nodes.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:641*

#### `build(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Build the layer.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:1001*

#### `build(self, input_shape)`
**Module:** `layers.time_series.temporal_convolutional_network`

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:57*

#### `build(self, input_shape)`
**Module:** `layers.physics.approximate_lagrange_layer`

Build all internal MLP sub-layers with proper output dimensions.

*📁 src/dl_techniques/layers/physics/approximate_lagrange_layer.py:176*

#### `build(self, input_shape)`
**Module:** `layers.physics.lagrange_layer`

Build the internal MLP sub-layer with explicit shape information.

*📁 src/dl_techniques/layers/physics/lagrange_layer.py:140*

#### `build(self, input_shape)`
**Module:** `layers.activations.thresh_max`

Create the layer weights if trainable_slope is True.

*📁 src/dl_techniques/layers/activations/thresh_max.py:117*

#### `build(self, input_shape)`
**Module:** `layers.activations.differentiable_step`

Create the layer's trainable weights based on the `axis` configuration.

*📁 src/dl_techniques/layers/activations/differentiable_step.py:175*

#### `build(self, input_shape)`
**Module:** `layers.activations.expanded_activations`

Create the trainable parameter `alpha` for the expanded activation.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:318*

#### `build(self, input_shape)`
**Module:** `layers.activations.probability_output`

Build the internal strategy layer.

*📁 src/dl_techniques/layers/activations/probability_output.py:237*

#### `build(self, input_shape)`
**Module:** `layers.activations.routing_probabilities`

Build the layer by computing output dimensions and weight patterns.

*📁 src/dl_techniques/layers/activations/routing_probabilities.py:282*

#### `build(self, input_shape)`
**Module:** `layers.activations.adaptive_softmax`

Build the layer - validates input shape.

*📁 src/dl_techniques/layers/activations/adaptive_softmax.py:183*

#### `build(self, input_shape)`
**Module:** `layers.activations.monotonicity_layer`

Build the layer.

*📁 src/dl_techniques/layers/activations/monotonicity_layer.py:258*

#### `build(self, input_shape)`
**Module:** `layers.activations.routing_probabilities_hierarchical`

Build the layer by creating trainable weights and tree structure.

*📁 src/dl_techniques/layers/activations/routing_probabilities_hierarchical.py:203*

#### `build(self, input_shape)`
**Module:** `layers.memory.mann`

Create the layer's own weights and build sub-layers.

*📁 src/dl_techniques/layers/memory/mann.py:270*

#### `build(self, input_shape)`
**Module:** `layers.memory.som_nd_soft_layer`

Build the Soft SOM layer by creating trainable weight parameters.

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:472*

#### `build(self, input_shape)`
**Module:** `layers.memory.som_nd_layer`

Build the SOM layer by initializing the weight vectors.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:301*

#### `build(self, input_shape)`
**Module:** `layers.vision_heads.factory`

Build the layer.

*📁 src/dl_techniques/layers/vision_heads/factory.py:133*

#### `build(self, input_shape)`
**Module:** `layers.vlm_heads.factory`

Builds the layer.

*📁 src/dl_techniques/layers/vlm_heads/factory.py:111*

#### `build(self, input_shape)`
**Module:** `layers.geometric.point_cloud_autoencoder`

Build the layer by creating shape-dependent sub-layers.

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:147*

#### `build(self, input_shape)`
**Module:** `layers.geometric.point_cloud_autoencoder`

Build the MLP with the correct input dimension.

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:395*

#### `build(self, input_shape)`
**Module:** `layers.geometric.supernode_pooling`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/geometric/supernode_pooling.py:216*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.parallel_transport`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/parallel_transport.py:101*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.holonomy_layer`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/holonomy_layer.py:116*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.gauge_invariant_attention`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/gauge_invariant_attention.py:116*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.connection_layer`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/connection_layer.py:123*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.manifold_stress`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/manifold_stress.py:111*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:69*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Build the layer.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:360*

#### `build(self, input_shape)`
**Module:** `layers.geometric.fields.field_embedding`

Build the layer weights.

*📁 src/dl_techniques/layers/geometric/fields/field_embedding.py:138*

#### `build(self, input_shape)`
**Module:** `layers.norms.zero_centered_rms_norm`

Create the layer's own weights.

*📁 src/dl_techniques/layers/norms/zero_centered_rms_norm.py:284*

#### `build(self, input_shape)`
**Module:** `layers.norms.adaptive_band_rms`

Build the layer and create sub-layers with proper parameter sizing.

*📁 src/dl_techniques/layers/norms/adaptive_band_rms.py:278*

#### `build(self, input_shape)`
**Module:** `layers.norms.band_rms`

Create the layer's own weights.

*📁 src/dl_techniques/layers/norms/band_rms.py:212*

#### `build(self, input_shape)`
**Module:** `layers.norms.global_response_norm`

Create the layer's weights, adapted for 2D, 3D, or 4D inputs.

*📁 src/dl_techniques/layers/norms/global_response_norm.py:179*

#### `build(self, input_shape)`
**Module:** `layers.norms.band_logit_norm`

Build the layer by initializing the LayerNormalization sublayer.

*📁 src/dl_techniques/layers/norms/band_logit_norm.py:99*

#### `build(self, input_shape)`
**Module:** `layers.norms.rms_norm`

Create the layer's own weights.

*📁 src/dl_techniques/layers/norms/rms_norm.py:251*

#### `build(self, input_shape)`
**Module:** `layers.norms.zero_centered_band_rms_norm`

Create the layer's own weights.

*📁 src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py:327*

#### `build(self, input_shape)`
**Module:** `layers.norms.dynamic_tanh`

Create the layer's learnable parameters.

*📁 src/dl_techniques/layers/norms/dynamic_tanh.py:133*

#### `build(self, input_shape)`
**Module:** `layers.logic.arithmetic_operators`

Build the layer weights.

*📁 src/dl_techniques/layers/logic/arithmetic_operators.py:232*

#### `build(self, input_shape)`
**Module:** `layers.logic.logic_operators`

Build the layer weights.

*📁 src/dl_techniques/layers/logic/logic_operators.py:210*

#### `build(self, input_shape)`
**Module:** `layers.logic.neural_circuit`

Build the layer components.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:175*

#### `build(self, input_shape)`
**Module:** `layers.logic.neural_circuit`

Build the neural circuit layers.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:422*

#### `build(self, input_shape)`
**Module:** `layers.moe.layer`

Build the MoE layer and all its sub-layers.

*📁 src/dl_techniques/layers/moe/layer.py:184*

#### `build(self, input_shape)`
**Module:** `layers.moe.experts`

Build the FFN expert using the factory system.

*📁 src/dl_techniques/layers/moe/experts.py:157*

#### `build(self, input_shape)`
**Module:** `layers.moe.gating`

Build the linear gating layers.

*📁 src/dl_techniques/layers/moe/gating.py:164*

#### `build(self, input_shape)`
**Module:** `layers.moe.gating`

Build the cosine gating layers.

*📁 src/dl_techniques/layers/moe/gating.py:339*

#### `build(self, input_shape)`
**Module:** `layers.moe.gating`

Build the SoftMoE gating layers.

*📁 src/dl_techniques/layers/moe/gating.py:508*

#### `build(self, input_shape)`
**Module:** `layers.fusion.multimodal_fusion`

Create and build all sublayers based on input shapes.

*📁 src/dl_techniques/layers/fusion/multimodal_fusion.py:204*

#### `build(self, input_shape)`
**Module:** `layers.statistics.moving_std`

Build the layer and its internal average pooling component.

*📁 src/dl_techniques/layers/statistics/moving_std.py:225*

#### `build(self, input_shape)`
**Module:** `layers.statistics.residual_acf`

Build the layer and validate input shapes.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:248*

#### `build(self, input_shapes)`
**Module:** `layers.statistics.normalizing_flow`

Build the layer and its transformation network.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:235*

#### `build(self, input_shapes)`
**Module:** `layers.statistics.normalizing_flow`

Build the layer and all coupling layers.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:623*

#### `build(self, input_shape)`
**Module:** `layers.statistics.invertible_kernel_pca`

Create Random Fourier Features and PCA projection weights.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:266*

#### `build(self, input_shape)`
**Module:** `layers.statistics.invertible_kernel_pca`

Build the denoising layer.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:729*

#### `build(self, input_shape)`
**Module:** `layers.statistics.scaler`

Create the layer's weights and validate input shape.

*📁 src/dl_techniques/layers/statistics/scaler.py:314*

#### `build(self, input_shape)`
**Module:** `layers.statistics.deep_kernel_pca`

Create weights for multi-level kernel PCA.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:255*

#### `build(self, input_shape)`
**Module:** `layers.statistics.mdn_layer`

Build the layer's weights and sub-layers.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:238*

#### `build(self, input_shape)`
**Module:** `layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer`

Create layer weights based on input shape.

*📁 src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py:175*

#### `build(self, input_shape)`
**Module:** `layers.graphs.graph_neural_network`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/graphs/graph_neural_network.py:385*

#### `build(self, input_shape)`
**Module:** `layers.graphs.entity_graph_refinement`

Build all sublayers and weights for the entity-graph refinement component.

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:324*

#### `build(self, input_shape)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Create the layer's learnable weights.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:156*

#### `build(self, input_shape)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Build all sub-layers.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:479*

#### `build(self, input_shape)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Build sub-layers and create global centroid weights.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:847*

#### `build(self, input_shape)`
**Module:** `layers.graphs.fermi_diract_decoder`

Create learnable parameters.

*📁 src/dl_techniques/layers/graphs/fermi_diract_decoder.py:137*

#### `build(self, input_shape)`
**Module:** `layers.attention.channel_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/channel_attention.py:201*

#### `build(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

Build the branch layers.

*📁 src/dl_techniques/layers/attention/tripse_attention.py:87*

#### `build(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:245*

#### `build(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:335*

#### `build(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:470*

#### `build(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:567*

#### `build(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:687*

#### `build(self, input_shape)`
**Module:** `layers.attention.rpc_attention`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/attention/rpc_attention.py:294*

#### `build(self, input_shape)`
**Module:** `layers.attention.multi_head_attention`

Build the layer by creating weight variables and building sub-layers.

*📁 src/dl_techniques/layers/attention/multi_head_attention.py:190*

#### `build(self, input_shape)`
**Module:** `layers.attention.shared_weights_cross_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/shared_weights_cross_attention.py:232*

#### `build(self, input_shape)`
**Module:** `layers.attention.single_window_attention`

Build the layer's weights.

*📁 src/dl_techniques/layers/attention/single_window_attention.py:166*

#### `build(self, input_shape)`
**Module:** `layers.attention.ring_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/ring_attention.py:361*

#### `build(self, input_shape)`
**Module:** `layers.attention.perceiver_attention`

Build the layer by creating weight variables and building sub-layers.

*📁 src/dl_techniques/layers/attention/perceiver_attention.py:238*

#### `build(self, input_shape)`
**Module:** `layers.attention.gated_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/gated_attention.py:392*

#### `build(self, input_shape)`
**Module:** `layers.attention.spatial_attention`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/attention/spatial_attention.py:165*

#### `build(self, input_shape)`
**Module:** `layers.attention.performer_attention`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/attention/performer_attention.py:288*

#### `build(self, input_shape)`
**Module:** `layers.attention.multi_head_cross_attention`

Build the layer by creating weight variables and building sub-layers.

*📁 src/dl_techniques/layers/attention/multi_head_cross_attention.py:330*

#### `build(self, input_shape)`
**Module:** `layers.attention.fnet_fourier_transform`

Create and cache DFT matrices for efficient computation.

*📁 src/dl_techniques/layers/attention/fnet_fourier_transform.py:146*

#### `build(self, input_shape)`
**Module:** `layers.attention.window_attention`

Build layer and precompute zigzag indices if needed.

*📁 src/dl_techniques/layers/attention/window_attention.py:321*

#### `build(self, input_shape)`
**Module:** `layers.attention.capsule_routing_attention`

Build the layer and create all sub-components.

*📁 src/dl_techniques/layers/attention/capsule_routing_attention.py:296*

#### `build(self, input_shape)`
**Module:** `layers.attention.anchor_attention`

Build the layer and all sub-layers.

*📁 src/dl_techniques/layers/attention/anchor_attention.py:381*

#### `build(self, input_shape)`
**Module:** `layers.attention.differential_attention`

Build the layer and create the lambda parameter weight.

*📁 src/dl_techniques/layers/attention/differential_attention.py:269*

#### `build(self, input_shape)`
**Module:** `layers.attention.hopfield_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/hopfield_attention.py:279*

#### `build(self, input_shape)`
**Module:** `layers.attention.multi_head_latent_attention`

Build the layer and all sub-layers.

*📁 src/dl_techniques/layers/attention/multi_head_latent_attention.py:411*

#### `build(self, input_shape)`
**Module:** `layers.attention.group_query_attention`

Build the layer and all its sub-layers. Explicitly builds sub-layers for robust serialization.

*📁 src/dl_techniques/layers/attention/group_query_attention.py:255*

#### `build(self, input_shape)`
**Module:** `layers.attention.mobile_mqa`

Build the layer. Calls super().build() for GQA weights, then adds MobileMQA-specific components (downsample, lambda).

*📁 src/dl_techniques/layers/attention/mobile_mqa.py:116*

#### `build(self, input_shape)`
**Module:** `layers.attention.convolutional_block_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/convolutional_block_attention.py:220*

#### `build(self, input_shape)`
**Module:** `layers.attention.non_local_attention`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/attention/non_local_attention.py:364*

#### `build(self, input_shape)`
**Module:** `layers.attention.progressive_focused_attention`

Build layer weights and sub-layers.

*📁 src/dl_techniques/layers/attention/progressive_focused_attention.py:324*

#### `build(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Build sub-layers explicitly.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:257*

#### `build(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Build sub-layers explicitly.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:505*

#### `build(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Build the core layer.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:703*

#### `build(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Build controller and heads with correct shapes.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:951*

#### `build(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Build sub-layers.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1262*

#### `build(self, input_shape)`
**Module:** `layers.ntm.base_layers`

Build the addressing head components.

*📁 src/dl_techniques/layers/ntm/base_layers.py:186*

#### `build(self, input_shape)`
**Module:** `layers.ntm.base_layers`

Build read and write heads.

*📁 src/dl_techniques/layers/ntm/base_layers.py:474*

#### `build(self, input_shape)`
**Module:** `layers.ntm.base_layers`

Build selection and placement mechanisms.

*📁 src/dl_techniques/layers/ntm/base_layers.py:786*

#### `build(self, input_shape)`
**Module:** `layers.ffn.residual_block`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/residual_block.py:246*

#### `build(self, input_shape)`
**Module:** `layers.ffn.counting_ffn`

Build the Counting FFN and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/counting_ffn.py:289*

#### `build(self, input_shape)`
**Module:** `layers.ffn.power_mlp_layer`

Build the layer weights and initialize sublayers.

*📁 src/dl_techniques/layers/ffn/power_mlp_layer.py:262*

#### `build(self, input_shape)`
**Module:** `layers.ffn.diff_ffn`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/diff_ffn.py:342*

#### `build(self, input_shape)`
**Module:** `layers.ffn.gated_mlp`

Build the GatedMLP layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/gated_mlp.py:293*

#### `build(self, input_shape)`
**Module:** `layers.ffn.swiglu_ffn`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/swiglu_ffn.py:315*

#### `build(self, input_shape)`
**Module:** `layers.ffn.swin_mlp`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/swin_mlp.py:240*

#### `build(self, input_shape)`
**Module:** `layers.ffn.mlp`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/mlp.py:248*

#### `build(self, input_shape)`
**Module:** `layers.ffn.glu_ffn`

Build the layer and all its sub-layers for robust serialization.

*📁 src/dl_techniques/layers/ffn/glu_ffn.py:304*

#### `build(self, input_shape)`
**Module:** `layers.ffn.orthoglu_ffn`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/orthoglu_ffn.py:221*

#### `build(self, input_shape)`
**Module:** `layers.ffn.geglu_ffn`

Create weights for all sub-layers.

*📁 src/dl_techniques/layers/ffn/geglu_ffn.py:276*

#### `build(self, input_shape)`
**Module:** `layers.ffn.logic_ffn`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:242*

#### `build(self, input_shape)`
**Module:** `layers.embedding.patch_embedding`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:182*

#### `build(self, input_shape)`
**Module:** `layers.embedding.patch_embedding`

Build the layer and its sub-layers.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:383*

#### `build(self, input_shape)`
**Module:** `layers.embedding.continuous_rope_embedding`

Create the layer's frequency weights.

*📁 src/dl_techniques/layers/embedding/continuous_rope_embedding.py:176*

#### `build(self, input_shape)`
**Module:** `layers.embedding.modern_bert_embeddings`

Creates the weights for the embedding, norm, and dropout layers.

*📁 src/dl_techniques/layers/embedding/modern_bert_embeddings.py:109*

#### `build(self, input_shape)`
**Module:** `layers.embedding.positional_embedding`

Build the layer by creating weights and building sub-layers.

*📁 src/dl_techniques/layers/embedding/positional_embedding.py:180*

#### `build(self, input_shape)`
**Module:** `layers.embedding.bert_embeddings`

Build the embeddings layer by explicitly building all sub-layers.

*📁 src/dl_techniques/layers/embedding/bert_embeddings.py:257*

#### `build(self, input_shape)`
**Module:** `layers.embedding.continuous_sin_cos_embedding`

Create the layer's frequency weights.

*📁 src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py:197*

#### `build(self, input_shape)`
**Module:** `layers.embedding.dual_rotary_position_embedding`

Create cos/sin lookup tables for both global and local RoPE configurations.

*📁 src/dl_techniques/layers/embedding/dual_rotary_position_embedding.py:223*

#### `build(self, input_shape)`
**Module:** `layers.embedding.rotary_position_embedding`

Create the layer's cos/sin lookup tables.

*📁 src/dl_techniques/layers/embedding/rotary_position_embedding.py:218*

#### `build(self, input_shape)`
**Module:** `layers.reasoning.hrm_reasoning_module`

Build the module and all its internal TransformerLayer sub-layers.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_module.py:160*

#### `build(self, input_shape)`
**Module:** `layers.reasoning.hrm_sparse_puzzle_embedding`

Create the embedding weights and caching variables.

*📁 src/dl_techniques/layers/reasoning/hrm_sparse_puzzle_embedding.py:174*

#### `build(self, input_shape)`
**Module:** `layers.reasoning.hrm_reasoning_core`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:439*

#### `build_from_config(self, config)`
**Module:** `layers.modality_projection`

Build the layer from a config created with get_build_config.

*📁 src/dl_techniques/layers/modality_projection.py:307*

#### `build_from_config(self, config)`
**Module:** `layers.memory.som_nd_layer`

Build the layer from a configuration.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:571*

#### `build_from_config(self, config)`
**Module:** `layers.moe.experts`

Build the expert from configuration.

*📁 src/dl_techniques/layers/moe/experts.py:75*

#### `build_holonomic_field_model(vocab_size, embed_dim, seq_len, num_classes, use_stress_loss, stress_weight, use_expm, projection_type, name)`
**Module:** `layers.experimental.field_embeddings`

Build a complete holonomic field embedding model for sequence classification.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:848*

#### `build_moe_optimizer(self, model, config)`
**Module:** `layers.moe.integration`

Build an optimizer optimized for MoE training with FFN experts.

*📁 src/dl_techniques/layers/moe/integration.py:96*

#### `calibrate(self, calibration_scores, alpha)`
**Module:** `layers.time_series.forecasting_layers`

Update the conformal calibration score Q based on calibration data.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:418*

#### `call(self, inputs, training)`
**Module:** `layers.mobile_one_block`

Forward pass through the block.

*📁 src/dl_techniques/layers/mobile_one_block.py:287*

#### `call(self, inputs, training)`
**Module:** `layers.depthwise_separable_block`

Forward pass through the depthwise separable block.

*📁 src/dl_techniques/layers/depthwise_separable_block.py:336*

#### `call(self, inputs, training)`
**Module:** `layers.sampling`

Apply reparameterization trick to sample from Normal distribution.

*📁 src/dl_techniques/layers/sampling.py:198*

#### `call(self, inputs, training)`
**Module:** `layers.hanc_layer`

Forward pass computation.

*📁 src/dl_techniques/layers/hanc_layer.py:261*

#### `call(self, inputs, training)`
**Module:** `layers.laplacian_filter`

Apply the Laplacian filter to the input tensor.

*📁 src/dl_techniques/layers/laplacian_filter.py:175*

#### `call(self, inputs, training)`
**Module:** `layers.laplacian_filter`

Apply the Laplacian filter to the input tensor.

*📁 src/dl_techniques/layers/laplacian_filter.py:427*

#### `call(self, inputs, training)`
**Module:** `layers.multi_level_feature_compilation`

Forward pass computation.

*📁 src/dl_techniques/layers/multi_level_feature_compilation.py:320*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_blocks`

Forward pass through the convolution block.

*📁 src/dl_techniques/layers/yolo12_blocks.py:187*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_blocks`

Forward pass through area attention.

*📁 src/dl_techniques/layers/yolo12_blocks.py:370*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_blocks`

Forward pass through attention block.

*📁 src/dl_techniques/layers/yolo12_blocks.py:625*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_blocks`

Forward pass through bottleneck.

*📁 src/dl_techniques/layers/yolo12_blocks.py:775*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_blocks`

Forward pass through C3k2 block.

*📁 src/dl_techniques/layers/yolo12_blocks.py:955*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_blocks`

Forward pass through A2C2f block.

*📁 src/dl_techniques/layers/yolo12_blocks.py:1153*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_heads`

Forward pass through detection head.

*📁 src/dl_techniques/layers/yolo12_heads.py:290*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_heads`

Forward pass through segmentation head.

*📁 src/dl_techniques/layers/yolo12_heads.py:706*

#### `call(self, inputs, training)`
**Module:** `layers.yolo12_heads`

Forward pass through classification head.

*📁 src/dl_techniques/layers/yolo12_heads.py:1058*

#### `call(self, inputs, training)`
**Module:** `layers.radial_basis_function`

Forward pass of the RBF Layer.

*📁 src/dl_techniques/layers/radial_basis_function.py:243*

#### `call(self, inputs, training)`
**Module:** `layers.spatial_layer`

Forward pass that dynamically resizes coordinate grid to match input dimensions.

*📁 src/dl_techniques/layers/spatial_layer.py:230*

#### `call(self, inputs)`
**Module:** `layers.one_hot_encoding`

Apply one-hot encoding to categorical inputs.

*📁 src/dl_techniques/layers/one_hot_encoding.py:84*

#### `call(self, inputs, training)`
**Module:** `layers.neuro_grid`

Forward pass: project input to probabilities and perform soft lookup.

*📁 src/dl_techniques/layers/neuro_grid.py:599*

#### `call(self, inputs, training)`
**Module:** `layers.gaussian_filter`

Apply the Gaussian filter to the input tensor.

*📁 src/dl_techniques/layers/gaussian_filter.py:191*

#### `call(self, inputs, training)`
**Module:** `layers.dynamic_conv2d`

Forward pass with dynamic kernel aggregation.

*📁 src/dl_techniques/layers/dynamic_conv2d.py:393*

#### `call(self, inputs, training)`
**Module:** `layers.shearlet_transform`

Apply shearlet transform to input images using Keras Ops.

*📁 src/dl_techniques/layers/shearlet_transform.py:322*

#### `call(self, inputs, training)`
**Module:** `layers.squeeze_excitation`

Forward pass of the SE block.

*📁 src/dl_techniques/layers/squeeze_excitation.py:302*

#### `call(self, inputs, training)`
**Module:** `layers.standard_blocks`

Forward pass through the convolutional block.

*📁 src/dl_techniques/layers/standard_blocks.py:350*

#### `call(self, inputs, training)`
**Module:** `layers.standard_blocks`

Forward pass through the dense block.

*📁 src/dl_techniques/layers/standard_blocks.py:588*

#### `call(self, inputs, training)`
**Module:** `layers.standard_blocks`

Forward pass through the residual dense block.

*📁 src/dl_techniques/layers/standard_blocks.py:795*

#### `call(self, inputs, training)`
**Module:** `layers.standard_blocks`

Forward pass of the basic block.

*📁 src/dl_techniques/layers/standard_blocks.py:1033*

#### `call(self, inputs, training)`
**Module:** `layers.standard_blocks`

Forward pass of the bottleneck block.

*📁 src/dl_techniques/layers/standard_blocks.py:1313*

#### `call(self, inputs, mask, training)`
**Module:** `layers.sequence_pooling`

Apply attention-based pooling.

*📁 src/dl_techniques/layers/sequence_pooling.py:215*

#### `call(self, inputs, mask, training)`
**Module:** `layers.sequence_pooling`

Apply weighted pooling.

*📁 src/dl_techniques/layers/sequence_pooling.py:368*

#### `call(self, inputs, mask, training)`
**Module:** `layers.sequence_pooling`

Apply pooling strategy to inputs.

*📁 src/dl_techniques/layers/sequence_pooling.py:757*

#### `call(self, carry, inputs, training)`
**Module:** `layers.blt_core`

Forward pass through the byte latent hierarchical reasoning core.

*📁 src/dl_techniques/layers/blt_core.py:466*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.fnet_encoder_block`

Forward pass through complete FNet encoder block.

*📁 src/dl_techniques/layers/fnet_encoder_block.py:252*

#### `call(self, inputs, training)`
**Module:** `layers.conditional_output_layer`

Forward pass of the layer.

*📁 src/dl_techniques/layers/conditional_output_layer.py:137*

#### `call(self, inputs, training)`
**Module:** `layers.complex_layers`

Apply complex convolution to input tensor.

*📁 src/dl_techniques/layers/complex_layers.py:325*

#### `call(self, inputs, training)`
**Module:** `layers.complex_layers`

Apply complex dense transformation.

*📁 src/dl_techniques/layers/complex_layers.py:542*

#### `call(self, inputs, training)`
**Module:** `layers.complex_layers`

Apply complex ReLU activation.

*📁 src/dl_techniques/layers/complex_layers.py:670*

#### `call(self, inputs, training)`
**Module:** `layers.complex_layers`

Apply complex average pooling.

*📁 src/dl_techniques/layers/complex_layers.py:808*

#### `call(self, inputs, training)`
**Module:** `layers.complex_layers`

Apply complex dropout.

*📁 src/dl_techniques/layers/complex_layers.py:970*

#### `call(self, inputs, training)`
**Module:** `layers.complex_layers`

Apply complex global average pooling.

*📁 src/dl_techniques/layers/complex_layers.py:1092*

#### `call(self, inputs, training)`
**Module:** `layers.anchor_generator`

Forward pass returning batch-tiled anchors and strides.

*📁 src/dl_techniques/layers/anchor_generator.py:215*

#### `call(self, inputs, training)`
**Module:** `layers.haar_wavelet_decomposition`

Apply Haar DWT decomposition.

*📁 src/dl_techniques/layers/haar_wavelet_decomposition.py:127*

#### `call(self, inputs)`
**Module:** `layers.tabm_blocks`

Apply ensemble scaling to inputs.

*📁 src/dl_techniques/layers/tabm_blocks.py:109*

#### `call(self, inputs)`
**Module:** `layers.tabm_blocks`

Forward pass through efficient ensemble layer.

*📁 src/dl_techniques/layers/tabm_blocks.py:225*

#### `call(self, inputs)`
**Module:** `layers.tabm_blocks`

Forward pass through N parallel linear layers.

*📁 src/dl_techniques/layers/tabm_blocks.py:337*

#### `call(self, inputs, training)`
**Module:** `layers.tabm_blocks`

Forward pass through MLP block.

*📁 src/dl_techniques/layers/tabm_blocks.py:453*

#### `call(self, inputs, training)`
**Module:** `layers.tabm_blocks`

Forward pass through backbone MLP.

*📁 src/dl_techniques/layers/tabm_blocks.py:562*

#### `call(self, inputs, training)`
**Module:** `layers.kan_linear`

Forward pass: compute output using learned activation functions.

*📁 src/dl_techniques/layers/kan_linear.py:357*

#### `call(self, inputs, training)`
**Module:** `layers.convnext_v2_block`

Forward pass of the ConvNextV2 block.

*📁 src/dl_techniques/layers/convnext_v2_block.py:394*

#### `call(self, inputs, training)`
**Module:** `layers.io_preparation`

Apply clipping to input tensor.

*📁 src/dl_techniques/layers/io_preparation.py:91*

#### `call(self, inputs, training)`
**Module:** `layers.io_preparation`

Apply normalization to input tensor.

*📁 src/dl_techniques/layers/io_preparation.py:219*

#### `call(self, inputs, training)`
**Module:** `layers.io_preparation`

Apply denormalization to input tensor.

*📁 src/dl_techniques/layers/io_preparation.py:353*

#### `call(self, inputs, training)`
**Module:** `layers.io_preparation`

Forward pass through sub-layers.

*📁 src/dl_techniques/layers/io_preparation.py:516*

#### `call(self, inputs, training)`
**Module:** `layers.convnext_v1_block`

Forward pass of the ConvNext block.

*📁 src/dl_techniques/layers/convnext_v1_block.py:368*

#### `call(self, inputs, training)`
**Module:** `layers.patch_merging`

Forward pass of patch merging operation.

*📁 src/dl_techniques/layers/patch_merging.py:180*

#### `call(self, inputs)`
**Module:** `layers.fft_layers`

Apply 2D FFT to input tensor.

*📁 src/dl_techniques/layers/fft_layers.py:62*

#### `call(self, inputs)`
**Module:** `layers.fft_layers`

Apply 2D IFFT to input tensor and extract real part.

*📁 src/dl_techniques/layers/fft_layers.py:178*

#### `call(self, inputs, training)`
**Module:** `layers.orthoblock`

Forward computation through the orthogonal block pipeline.

*📁 src/dl_techniques/layers/orthoblock.py:301*

#### `call(self, inputs, training)`
**Module:** `layers.pixel_shuffle`

Apply pixel shuffle operation.

*📁 src/dl_techniques/layers/pixel_shuffle.py:177*

#### `call(self, inputs, training)`
**Module:** `layers.res_path`

Forward pass through the series of residual blocks.

*📁 src/dl_techniques/layers/res_path.py:207*

#### `call(self, inputs, training)`
**Module:** `layers.mps_layer`

Forward pass implementing MPS tensor contraction.

*📁 src/dl_techniques/layers/mps_layer.py:311*

#### `call(self, inputs, training)`
**Module:** `layers.film`

Apply the configurable FiLM transformation.

*📁 src/dl_techniques/layers/film.py:357*

#### `call(self, inputs, training)`
**Module:** `layers.hanc_block`

Forward pass computation.

*📁 src/dl_techniques/layers/hanc_block.py:324*

#### `call(self, inputs, training)`
**Module:** `layers.repmixer_block`

Forward pass through RepMixer block.

*📁 src/dl_techniques/layers/repmixer_block.py:355*

#### `call(self, inputs, training)`
**Module:** `layers.repmixer_block`

Forward pass through stem blocks.

*📁 src/dl_techniques/layers/repmixer_block.py:557*

#### `call(self, inputs, training)`
**Module:** `layers.bitlinear_layer`

Perform quantized linear transformation.

*📁 src/dl_techniques/layers/bitlinear_layer.py:442*

#### `call(self, inputs, training)`
**Module:** `layers.stochastic_gradient`

Forward pass of the layer.

*📁 src/dl_techniques/layers/stochastic_gradient.py:129*

#### `call(self, inputs, training)`
**Module:** `layers.gaussian_pyramid`

Apply Gaussian pyramid decomposition.

*📁 src/dl_techniques/layers/gaussian_pyramid.py:275*

#### `call(self, inputs, training)`
**Module:** `layers.bias_free_conv1d`

Forward computation through the bias-free 1D convolution layer.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:199*

#### `call(self, inputs, training)`
**Module:** `layers.bias_free_conv1d`

Forward pass through the residual block.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:457*

#### `call(self, inputs, training)`
**Module:** `layers.vector_quantizer`

Quantize inputs using nearest neighbor lookup in embedding space.

*📁 src/dl_techniques/layers/vector_quantizer.py:205*

#### `call(self, inputs, training)`
**Module:** `layers.restricted_boltzmann_machine`

Forward pass: compute hidden representation given visible units.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:261*

#### `call(self, inputs, training)`
**Module:** `layers.convolutional_kan`

Forward pass applying KAN transformation followed by convolution.

*📁 src/dl_techniques/layers/convolutional_kan.py:378*

#### `call(self, inputs, training)`
**Module:** `layers.kmeans`

Forward pass of the layer.

*📁 src/dl_techniques/layers/kmeans.py:596*

#### `call(self, inputs, training)`
**Module:** `layers.capsules`

Forward pass for primary capsule layer.

*📁 src/dl_techniques/layers/capsules.py:245*

#### `call(self, inputs, training)`
**Module:** `layers.capsules`

Forward pass implementing dynamic routing between capsules.

*📁 src/dl_techniques/layers/capsules.py:546*

#### `call(self, inputs, training)`
**Module:** `layers.capsules`

Forward pass through the capsule block.

*📁 src/dl_techniques/layers/capsules.py:863*

#### `call(self, inputs, training)`
**Module:** `layers.gated_delta_net`

Forward pass through the Gated DeltaNet layer.

*📁 src/dl_techniques/layers/gated_delta_net.py:455*

#### `call(self, inputs, training)`
**Module:** `layers.bias_free_conv2d`

Forward computation through the bias-free 2D convolution layer.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:219*

#### `call(self, inputs, training)`
**Module:** `layers.bias_free_conv2d`

Forward pass through the residual block.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:497*

#### `call(self, inputs, training)`
**Module:** `layers.mothnet_blocks`

Apply competitive inhibition to inputs.

*📁 src/dl_techniques/layers/mothnet_blocks.py:204*

#### `call(self, inputs, training)`
**Module:** `layers.mothnet_blocks`

Apply sparse high-dimensional projection with winner-take-all.

*📁 src/dl_techniques/layers/mothnet_blocks.py:470*

#### `call(self, inputs, training)`
**Module:** `layers.mothnet_blocks`

Compute readout activations (forward pass).

*📁 src/dl_techniques/layers/mothnet_blocks.py:744*

#### `call(self, inputs, training)`
**Module:** `layers.selective_gradient_mask`

Apply selective gradient masking.

*📁 src/dl_techniques/layers/selective_gradient_mask.py:165*

#### `call(self, inputs, training)`
**Module:** `layers.strong_augmentation`

Apply strong augmentations to input images.

*📁 src/dl_techniques/layers/strong_augmentation.py:88*

#### `call(self, inputs)`
**Module:** `layers.clahe`

Apply CLAHE to the input tensor.

*📁 src/dl_techniques/layers/clahe.py:233*

#### `call(self, inputs)`
**Module:** `layers.tversky_projection`

Forward pass computation of Tversky similarity.

*📁 src/dl_techniques/layers/tversky_projection.py:218*

#### `call(self, inputs, training)`
**Module:** `layers.stochastic_depth`

Forward pass of the layer.

*📁 src/dl_techniques/layers/stochastic_depth.py:138*

#### `call(self, inputs, training)`
**Module:** `layers.hierarchical_mlp_stem`

Apply hierarchical MLP stem to input images.

*📁 src/dl_techniques/layers/hierarchical_mlp_stem.py:234*

#### `call(self, inputs, training)`
**Module:** `layers.layer_scale`

Apply the learnable multipliers to inputs.

*📁 src/dl_techniques/layers/layer_scale.py:259*

#### `call(self, inputs, training)`
**Module:** `layers.blt_blocks`

Forward pass of the entropy model.

*📁 src/dl_techniques/layers/blt_blocks.py:467*

#### `call(self, entropy, training)`
**Module:** `layers.blt_blocks`

Create patch lengths from entropy values.

*📁 src/dl_techniques/layers/blt_blocks.py:593*

#### `call(self, byte_hiddens, patch_ids, training)`
**Module:** `layers.blt_blocks`

Pool byte representations into patch representations.

*📁 src/dl_techniques/layers/blt_blocks.py:811*

#### `call(self, byte_tokens, patch_ids, training)`
**Module:** `layers.blt_blocks`

Forward pass of local encoder.

*📁 src/dl_techniques/layers/blt_blocks.py:1103*

#### `call(self, patch_representations, training)`
**Module:** `layers.blt_blocks`

Forward pass of global transformer.

*📁 src/dl_techniques/layers/blt_blocks.py:1249*

#### `call(self, byte_tokens, global_context, patch_ids, training)`
**Module:** `layers.blt_blocks`

Forward pass of local decoder.

*📁 src/dl_techniques/layers/blt_blocks.py:1455*

#### `call(self, inputs, training)`
**Module:** `layers.random_fourier_features`

Apply Random Fourier Features transformation.

*📁 src/dl_techniques/layers/random_fourier_features.py:323*

#### `call(self, inputs, training)`
**Module:** `layers.modality_projection`

Apply modality projection.

*📁 src/dl_techniques/layers/modality_projection.py:224*

#### `call(self, inputs, training, return_latents)`
**Module:** `layers.sparse_autoencoder`

Forward pass through the Sparse Autoencoder.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:759*

#### `call(self, inputs, training)`
**Module:** `layers.eomt_mask`

Forward pass through the mask module.

*📁 src/dl_techniques/layers/eomt_mask.py:421*

#### `call(self, inputs)`
**Module:** `layers.canny`

Perform Canny edge detection on the input image tensor.

*📁 src/dl_techniques/layers/canny.py:213*

#### `call(self, inputs, training)`
**Module:** `layers.universal_inverted_bottleneck`

Forward pass of the UIB block.

*📁 src/dl_techniques/layers/universal_inverted_bottleneck.py:506*

#### `call(self, inputs, training)`
**Module:** `layers.global_sum_pool_2d`

Forward pass computation.

*📁 src/dl_techniques/layers/global_sum_pool_2d.py:149*

#### `call(self, inputs, training)`
**Module:** `layers.fractal_block`

Forward pass through the FractalBlock.

*📁 src/dl_techniques/layers/fractal_block.py:332*

#### `call(self, inputs, training)`
**Module:** `layers.rigid_simplex_layer`

Forward pass computation.

*📁 src/dl_techniques/layers/rigid_simplex_layer.py:330*

#### `call(self, inputs, attention_mask, layer_idx, layer_decision, training)`
**Module:** `layers.router`

Forward pass of the RouterLayer.

*📁 src/dl_techniques/layers/router.py:260*

#### `call(self, inputs, training)`
**Module:** `layers.transformers.progressive_focused_transformer`

Forward pass of PFT block.

*📁 src/dl_techniques/layers/transformers/progressive_focused_transformer.py:642*

#### `call(self, inputs, token_type_ids, attention_mask, training)`
**Module:** `layers.transformers.text_encoder`

Forward pass through the text encoder.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:724*

#### `call(self, input_ids, attention_mask, training)`
**Module:** `layers.transformers.text_decoder`

Forward pass through the text decoder.

*📁 src/dl_techniques/layers/transformers/text_decoder.py:405*

#### `call(self, inputs, mask, training)`
**Module:** `layers.transformers.eomt_transformer`

Forward pass through the EoMT transformer layer.

*📁 src/dl_techniques/layers/transformers/eomt_transformer.py:437*

#### `call(self, x, training)`
**Module:** `layers.transformers.swin_transformer_block`

Forward pass of the SwinTransformerBlock layer.

*📁 src/dl_techniques/layers/transformers/swin_transformer_block.py:481*

#### `call(self, bit_logits, training)`
**Module:** `layers.transformers.free_transformer`

Forward pass: sample one-hot vectors from bit logits.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:171*

#### `call(self, inputs, attention_mask, layer_idx, training)`
**Module:** `layers.transformers.free_transformer`

Forward pass of the Free Transformer layer.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:698*

#### `call(self, query_input, kv_input, training)`
**Module:** `layers.transformers.perceiver_transformer`

Apply Perceiver block processing.

*📁 src/dl_techniques/layers/transformers/perceiver_transformer.py:301*

#### `call(self, x, training)`
**Module:** `layers.transformers.swin_conv_block`

Forward pass implementing the split-transform-merge paradigm with residuals.

*📁 src/dl_techniques/layers/transformers/swin_conv_block.py:585*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.transformers.vision_encoder`

Forward pass through the vision_heads encoder.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:662*

#### `call(self, inputs, attention_mask, layer_idx, training)`
**Module:** `layers.transformers.transformer`

Forward pass of the transformer layer.

*📁 src/dl_techniques/layers/transformers/transformer.py:563*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Encode evidence at multiple levels.

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:134*

#### `call(self, evidence, training)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Aggregate evidence hierarchically.

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:366*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Generate support embeddings for token generation.

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:558*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Generate tokens with hierarchical evidence support.

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:758*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.hierarchical_memory_system`

Forward pass through the hierarchical memory system.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:272*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.field_embeddings`

Transform token indices to rotation matrices via exponential map.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:158*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.field_embeddings`

Compute path-ordered integral of rotation matrices.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:363*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.field_embeddings`

Compute manifold stress for rotation matrix trajectories.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:544*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.field_embeddings`

Project rotation matrix to feature vector.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:761*

#### `call(self, query, training)`
**Module:** `layers.experimental.contextual_memory`

Retrieve memories based on query.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:212*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.contextual_memory`

Process concept graph.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:439*

#### `call(self, sequence, training)`
**Module:** `layers.experimental.contextual_memory`

Encode temporal sequence.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:645*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.contextual_memory`

Process inputs through the contextual memory bank.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:818*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.mst_correlation_filter`

Execute forward pass of the layer.

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:309*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.band_rms_ood`

Forward pass with confidence-driven shell scaling.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:399*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.contextual_counter_ffn`

Forward pass implementing sense-aggregate-transform-modulate protocol.

*📁 src/dl_techniques/layers/experimental/contextual_counter_ffn.py:288*

#### `call(self, inputs, training)`
**Module:** `layers.experimental.graph_mann`

Forward pass computation over a sequence.

*📁 src/dl_techniques/layers/experimental/graph_mann.py:260*

#### `call(self, inputs, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through classification head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:393*

#### `call(self, inputs, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through token classification head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:521*

#### `call(self, inputs, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through QA head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:639*

#### `call(self, inputs, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through similarity head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:786*

#### `call(self, inputs, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through generation head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:952*

#### `call(self, inputs, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through multiple choice head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1043*

#### `call(self, inputs, task_name, training)`
**Module:** `layers.nlp_heads.factory`

Forward pass through multi-task head.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1213*

#### `call(self, inputs)`
**Module:** `layers.time_series.ema_layer`

Compute EMA over the time dimension.

*📁 src/dl_techniques/layers/time_series/ema_layer.py:86*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.ema_layer`

Compute EMA, slope, and trading signals.

*📁 src/dl_techniques/layers/time_series/ema_layer.py:311*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.nbeats_blocks`

Forward pass with performance optimizations.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:344*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.adaptive_lag_attention`

Forward pass of the layer.

*📁 src/dl_techniques/layers/time_series/adaptive_lag_attention.py:283*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.quantile_head_fixed_io`

Predict quantiles from the input feature vector.

*📁 src/dl_techniques/layers/time_series/quantile_head_fixed_io.py:189*

#### `call(self, inputs, scale, inverse)`
**Module:** `layers.time_series.deepar_blocks`

Apply scaling or inverse scaling.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:87*

#### `call(self, inputs)`
**Module:** `layers.time_series.deepar_blocks`

Compute Gaussian parameters.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:202*

#### `call(self, inputs)`
**Module:** `layers.time_series.deepar_blocks`

Compute Negative Binomial parameters.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:321*

#### `call(self, inputs, states, training)`
**Module:** `layers.time_series.deepar_blocks`

Process one time step.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:443*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.quantile_head_variable_io`

Predict quantiles for each time step in the input sequence.

*📁 src/dl_techniques/layers/time_series/quantile_head_variable_io.py:285*

#### `call(self, inputs, states, training)`
**Module:** `layers.time_series.xlstm_blocks`

Forward pass for a single timestep.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:202*

#### `call(self, inputs, mask, training, initial_state)`
**Module:** `layers.time_series.xlstm_blocks`

Forward pass through the RNN layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:412*

#### `call(self, inputs, states, training)`
**Module:** `layers.time_series.xlstm_blocks`

Forward pass for a single timestep.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:620*

#### `call(self, inputs, mask, training, initial_state)`
**Module:** `layers.time_series.xlstm_blocks`

Forward pass through the RNN layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:855*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.xlstm_blocks`

Forward pass through the block.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1050*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.xlstm_blocks`

Forward pass through the block.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1273*

#### `call(self, inputs, training, exogenous_inputs)`
**Module:** `layers.time_series.nbeatsx_blocks`

Forward pass for Exogenous Block.

*📁 src/dl_techniques/layers/time_series/nbeatsx_blocks.py:100*

#### `call(self, inputs, network_output)`
**Module:** `layers.time_series.forecasting_layers`

Combine network prediction with naive baseline.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:81*

#### `call(self, inputs, deep_forecast, naive_forecast, training)`
**Module:** `layers.time_series.forecasting_layers`

Compute gated combination of deep and naive forecasts.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:224*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.forecasting_layers`

Project inputs to quantile predictions.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:388*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.temporal_fusion`

Forward pass through the temporal fusion mechanism.

*📁 src/dl_techniques/layers/time_series/temporal_fusion.py:329*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.mixed_sequential_block`

Forward pass dispatching to the correct block type.

*📁 src/dl_techniques/layers/time_series/mixed_sequential_block.py:556*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.prism_blocks`

Compute statistics for the input frequency band.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:65*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.prism_blocks`

Compute importance weights for frequency bands.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:248*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.prism_blocks`

Process input through wavelet decomposition and weighted reconstruction.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:465*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.prism_blocks`

Process input through the hierarchical time tree.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:809*

#### `call(self, inputs, training, mask)`
**Module:** `layers.time_series.prism_blocks`

Apply PRISM processing.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:1012*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.temporal_convolutional_network`

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:64*

#### `call(self, inputs, training)`
**Module:** `layers.time_series.temporal_convolutional_network`

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:123*

#### `call(self, inputs)`
**Module:** `layers.physics.approximate_lagrange_layer`

Forward pass computing accelerations from approximated components.

*📁 src/dl_techniques/layers/physics/approximate_lagrange_layer.py:232*

#### `call(self, inputs)`
**Module:** `layers.physics.lagrange_layer`

Forward pass computing accelerations from learned Lagrangian.

*📁 src/dl_techniques/layers/physics/lagrange_layer.py:179*

#### `call(self, inputs, training)`
**Module:** `layers.activations.relu_k`

Forward pass of the ReLU-k activation.

*📁 src/dl_techniques/layers/activations/relu_k.py:178*

#### `call(self, inputs, training)`
**Module:** `layers.activations.hard_sigmoid`

Apply hard-sigmoid activation to inputs.

*📁 src/dl_techniques/layers/activations/hard_sigmoid.py:148*

#### `call(self, inputs, training)`
**Module:** `layers.activations.thresh_max`

Apply ThreshMax activation to inputs.

*📁 src/dl_techniques/layers/activations/thresh_max.py:212*

#### `call(self, inputs, training)`
**Module:** `layers.activations.basis_function`

Forward pass of the basis function activation.

*📁 src/dl_techniques/layers/activations/basis_function.py:168*

#### `call(self, inputs, training)`
**Module:** `layers.activations.differentiable_step`

Forward pass computation applying the soft step function. Broadcasting handles both scalar and per-axis cases automatically.

*📁 src/dl_techniques/layers/activations/differentiable_step.py:228*

#### `call(self, inputs)`
**Module:** `layers.activations.expanded_activations`

Apply the GELU activation function to the inputs.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:209*

#### `call(self, inputs)`
**Module:** `layers.activations.expanded_activations`

Apply the SiLU activation function to the inputs.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:247*

#### `call(self, inputs)`
**Module:** `layers.activations.expanded_activations`

Apply the xATLU activation function to the inputs.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:393*

#### `call(self, inputs)`
**Module:** `layers.activations.expanded_activations`

Apply the xGELU activation function to the inputs.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:447*

#### `call(self, inputs)`
**Module:** `layers.activations.expanded_activations`

Apply the xSiLU activation function to the inputs.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:501*

#### `call(self, inputs)`
**Module:** `layers.activations.expanded_activations`

Apply the ELU+1+ε activation function to the inputs.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:568*

#### `call(self, inputs, training, mask)`
**Module:** `layers.activations.probability_output`

Forward pass delegating to the selected strategy.

*📁 src/dl_techniques/layers/activations/probability_output.py:250*

#### `call(self, inputs, training)`
**Module:** `layers.activations.routing_probabilities`

Define the forward pass logic of the layer.

*📁 src/dl_techniques/layers/activations/routing_probabilities.py:396*

#### `call(self, inputs, training)`
**Module:** `layers.activations.adaptive_softmax`

Apply adaptive temperature softmax to input logits.

*📁 src/dl_techniques/layers/activations/adaptive_softmax.py:285*

#### `call(self, inputs, training)`
**Module:** `layers.activations.monotonicity_layer`

Apply monotonicity constraint to inputs.

*📁 src/dl_techniques/layers/activations/monotonicity_layer.py:291*

#### `call(self, inputs, training)`
**Module:** `layers.activations.mish`

Forward pass computation.

*📁 src/dl_techniques/layers/activations/mish.py:199*

#### `call(self, inputs, training)`
**Module:** `layers.activations.mish`

Forward pass computation.

*📁 src/dl_techniques/layers/activations/mish.py:367*

#### `call(self, inputs, training)`
**Module:** `layers.activations.sparsemax`

Apply sparsemax activation to input logits.

*📁 src/dl_techniques/layers/activations/sparsemax.py:86*

#### `call(self, inputs, training)`
**Module:** `layers.activations.squash`

Apply squashing non-linearity to input vectors.

*📁 src/dl_techniques/layers/activations/squash.py:160*

#### `call(self, inputs, training)`
**Module:** `layers.activations.routing_probabilities_hierarchical`

Define the forward pass logic of the layer.

*📁 src/dl_techniques/layers/activations/routing_probabilities_hierarchical.py:262*

#### `call(self, inputs, training)`
**Module:** `layers.activations.hard_swish`

Apply hard-swish activation to inputs.

*📁 src/dl_techniques/layers/activations/hard_swish.py:175*

#### `call(self, inputs)`
**Module:** `layers.activations.golu`

Forward pass computation for the GoLU activation.

*📁 src/dl_techniques/layers/activations/golu.py:158*

#### `call(self, inputs, training)`
**Module:** `layers.memory.mann`

Forward pass computation over a sequence.

*📁 src/dl_techniques/layers/memory/mann.py:342*

#### `call(self, inputs, training)`
**Module:** `layers.memory.som_nd_soft_layer`

Forward pass implementing soft competitive learning.

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:539*

#### `call(self, inputs, training)`
**Module:** `layers.memory.som_nd_layer`

Forward pass for the SOM layer.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:379*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

Forward pass through detection head.

*📁 src/dl_techniques/layers/vision_heads/factory.py:223*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

Forward pass through segmentation head.

*📁 src/dl_techniques/layers/vision_heads/factory.py:335*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

Forward pass through depth head.

*📁 src/dl_techniques/layers/vision_heads/factory.py:453*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

Forward pass through classification head.

*📁 src/dl_techniques/layers/vision_heads/factory.py:567*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

Forward pass through instance segmentation head.

*📁 src/dl_techniques/layers/vision_heads/factory.py:673*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

Forward pass through all task heads.

*📁 src/dl_techniques/layers/vision_heads/factory.py:772*

#### `call(self, inputs, training)`
**Module:** `layers.vision_heads.factory`

*📁 src/dl_techniques/layers/vision_heads/factory.py:934*

#### `call(self, inputs, training)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:226*

#### `call(self, inputs, training)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:347*

#### `call(self, inputs, training)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:415*

#### `call(self, inputs, training)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:512*

#### `call(self, inputs, task_name, training)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:609*

#### `call(self, inputs)`
**Module:** `layers.geometric.point_cloud_autoencoder`

Forward pass through the autoencoder for both point clouds.

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:256*

#### `call(self, inputs)`
**Module:** `layers.geometric.point_cloud_autoencoder`

Compute soft point-to-GMM component assignments.

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:416*

#### `call(self, inputs, training)`
**Module:** `layers.geometric.supernode_pooling`

Apply supernode pooling with message passing.

*📁 src/dl_techniques/layers/geometric/supernode_pooling.py:264*

#### `call(self, inputs, training, tangent)`
**Module:** `layers.geometric.fields.parallel_transport`

Perform parallel transport of vectors.

*📁 src/dl_techniques/layers/geometric/fields/parallel_transport.py:268*

#### `call(self, inputs, training)`
**Module:** `layers.geometric.fields.holonomy_layer`

Compute holonomy features for the input field.

*📁 src/dl_techniques/layers/geometric/fields/holonomy_layer.py:323*

#### `call(self, inputs, training, attention_mask)`
**Module:** `layers.geometric.fields.gauge_invariant_attention`

Compute gauge-invariant attention.

*📁 src/dl_techniques/layers/geometric/fields/gauge_invariant_attention.py:395*

#### `call(self, inputs, training)`
**Module:** `layers.geometric.fields.connection_layer`

Compute the connection from input field.

*📁 src/dl_techniques/layers/geometric/fields/connection_layer.py:333*

#### `call(self, inputs, training)`
**Module:** `layers.geometric.fields.manifold_stress`

Compute manifold stress and anomaly mask.

*📁 src/dl_techniques/layers/geometric/fields/manifold_stress.py:360*

#### `call(self, inputs, training)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Apply field-aware normalization.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:102*

#### `call(self, inputs, training, attention_mask, return_attention_weights)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Forward pass through the holonomic transformer layer.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:399*

#### `call(self, inputs, training)`
**Module:** `layers.geometric.fields.field_embedding`

Forward pass: embed tokens as fields with curvature.

*📁 src/dl_techniques/layers/geometric/fields/field_embedding.py:274*

#### `call(self, inputs, training)`
**Module:** `layers.norms.zero_centered_rms_norm`

Apply Zero-Centered RMS normalization to inputs.

*📁 src/dl_techniques/layers/norms/zero_centered_rms_norm.py:334*

#### `call(self, inputs, training)`
**Module:** `layers.norms.adaptive_band_rms`

Apply adaptive RMS normalization with log-transformed statistics.

*📁 src/dl_techniques/layers/norms/adaptive_band_rms.py:368*

#### `call(self, inputs, training)`
**Module:** `layers.norms.logit_norm`

Apply logit normalization to inputs.

*📁 src/dl_techniques/layers/norms/logit_norm.py:218*

#### `call(self, inputs, training)`
**Module:** `layers.norms.band_rms`

Apply constrained RMS normalization.

*📁 src/dl_techniques/layers/norms/band_rms.py:237*

#### `call(self, inputs, training)`
**Module:** `layers.norms.global_response_norm`

Apply global response normalization to the input tensor.

*📁 src/dl_techniques/layers/norms/global_response_norm.py:224*

#### `call(self, inputs, training)`
**Module:** `layers.norms.band_logit_norm`

Apply constrained RMS normalization.

*📁 src/dl_techniques/layers/norms/band_logit_norm.py:122*

#### `call(self, inputs, training)`
**Module:** `layers.norms.max_logit_norm`

Apply MaxLogit normalization.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:165*

#### `call(self, inputs, training)`
**Module:** `layers.norms.max_logit_norm`

Apply decoupled MaxLogit normalization.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:363*

#### `call(self, inputs, training)`
**Module:** `layers.norms.max_logit_norm`

Apply DML+ normalization based on model type.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:587*

#### `call(self, inputs, training)`
**Module:** `layers.norms.rms_norm`

Apply RMS normalization to inputs.

*📁 src/dl_techniques/layers/norms/rms_norm.py:301*

#### `call(self, inputs, training)`
**Module:** `layers.norms.zero_centered_band_rms_norm`

Apply Zero-Centered Band RMS normalization to inputs.

*📁 src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py:353*

#### `call(self, inputs, training)`
**Module:** `layers.norms.dynamic_tanh`

Forward computation: weight * tanh(alpha * inputs) + bias.

*📁 src/dl_techniques/layers/norms/dynamic_tanh.py:191*

#### `call(self, inputs, training)`
**Module:** `layers.logic.arithmetic_operators`

Forward pass through the arithmetic operator.

*📁 src/dl_techniques/layers/logic/arithmetic_operators.py:342*

#### `call(self, inputs, training)`
**Module:** `layers.logic.logic_operators`

Forward pass through the logic operator.

*📁 src/dl_techniques/layers/logic/logic_operators.py:330*

#### `call(self, inputs, training)`
**Module:** `layers.logic.neural_circuit`

Forward pass through the circuit depth layer.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:226*

#### `call(self, inputs, training)`
**Module:** `layers.logic.neural_circuit`

Forward pass through the neural circuit.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:462*

#### `call(self, inputs, training)`
**Module:** `layers.moe.layer`

Forward pass through the MoE layer.

*📁 src/dl_techniques/layers/moe/layer.py:251*

#### `call(self, inputs, training)`
**Module:** `layers.moe.experts`

Forward computation for the expert.

*📁 src/dl_techniques/layers/moe/experts.py:45*

#### `call(self, inputs, training)`
**Module:** `layers.moe.experts`

Forward pass through the FFN expert.

*📁 src/dl_techniques/layers/moe/experts.py:178*

#### `call(self, inputs, training)`
**Module:** `layers.moe.gating`

Compute gating scores and routing information.

*📁 src/dl_techniques/layers/moe/gating.py:47*

#### `call(self, inputs, training)`
**Module:** `layers.moe.gating`

Forward pass through the linear gating network.

*📁 src/dl_techniques/layers/moe/gating.py:174*

#### `call(self, inputs, training)`
**Module:** `layers.moe.gating`

Forward pass through the cosine gating network.

*📁 src/dl_techniques/layers/moe/gating.py:362*

#### `call(self, inputs, training)`
**Module:** `layers.moe.gating`

Forward pass through the SoftMoE gating network.

*📁 src/dl_techniques/layers/moe/gating.py:527*

#### `call(self, inputs, training)`
**Module:** `layers.fusion.multimodal_fusion`

Apply the fusion strategy to combine multiple modalities.

*📁 src/dl_techniques/layers/fusion/multimodal_fusion.py:596*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.moving_std`

Apply the moving standard deviation filter to the input tensor.

*📁 src/dl_techniques/layers/statistics/moving_std.py:253*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.residual_acf`

Forward pass: compute ACF statistics and optional regularization loss.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:339*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.normalizing_flow`

Inverse transformation y → z for likelihood computation during training.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:643*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.invertible_kernel_pca`

Forward pass: transform inputs to principal components.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:466*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.invertible_kernel_pca`

Denoise inputs using ikPCA.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:783*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.scaler`

Apply normalization to inputs.

*📁 src/dl_techniques/layers/statistics/scaler.py:413*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.deep_kernel_pca`

Forward pass through hierarchical kernel PCA levels with coupled optimization.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:534*

#### `call(self, inputs, training)`
**Module:** `layers.statistics.mdn_layer`

Forward pass of the layer with separate processing paths.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:267*

#### `call(self, inputs, training)`
**Module:** `layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer`

Forward pass implementing Equation 14.

*📁 src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py:238*

#### `call(self, inputs, training)`
**Module:** `layers.graphs.graph_neural_network`

Process concept graph through GNN layers.

*📁 src/dl_techniques/layers/graphs/graph_neural_network.py:435*

#### `call(self, embeddings, training)`
**Module:** `layers.graphs.entity_graph_refinement`

Forward pass through the complete entity-graph refinement pipeline.

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:687*

#### `call(self, inputs, training)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Forward pass of the GNN layer.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:179*

#### `call(self, inputs, training)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Forward pass for multi-element tokenization.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:512*

#### `call(self, inputs, training)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Forward pass for hybrid local-global processing.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:895*

#### `call(self, inputs, training)`
**Module:** `layers.graphs.fermi_diract_decoder`

Compute edge probabilities from embedding pairs.

*📁 src/dl_techniques/layers/graphs/fermi_diract_decoder.py:180*

#### `call(self, inputs, training)`
**Module:** `layers.attention.channel_attention`

Apply channel attention to input tensor.

*📁 src/dl_techniques/layers/attention/channel_attention.py:241*

#### `call(self, inputs, training)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:119*

#### `call(self, inputs, training)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:252*

#### `call(self, inputs, training)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:356*

#### `call(self, inputs, training)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:484*

#### `call(self, inputs, training)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:596*

#### `call(self, inputs, training)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:703*

#### `call(self, inputs, mask, training, return_attention_scores)`
**Module:** `layers.attention.rpc_attention`

Apply RPC attention to inputs.

*📁 src/dl_techniques/layers/attention/rpc_attention.py:485*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.multi_head_attention`

Forward pass through self-attention mechanism.

*📁 src/dl_techniques/layers/attention/multi_head_attention.py:216*

#### `call(self, inputs, split_sizes, attention_mask, training)`
**Module:** `layers.attention.shared_weights_cross_attention`

Apply shared weights cross-attention.

*📁 src/dl_techniques/layers/attention/shared_weights_cross_attention.py:257*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.single_window_attention`

Forward pass for the unified single window attention.

*📁 src/dl_techniques/layers/attention/single_window_attention.py:209*

#### `call(self, inputs, training, attention_mask, return_attention_weights)`
**Module:** `layers.attention.ring_attention`

Apply ring attention with blockwise processing.

*📁 src/dl_techniques/layers/attention/ring_attention.py:386*

#### `call(self, query_input, kv_input, attention_mask, training)`
**Module:** `layers.attention.perceiver_attention`

Apply Perceiver cross-attention.

*📁 src/dl_techniques/layers/attention/perceiver_attention.py:279*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.gated_attention`

Forward pass through the gated attention layer.

*📁 src/dl_techniques/layers/attention/gated_attention.py:513*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.spatial_attention`

Apply spatial attention to input tensor.

*📁 src/dl_techniques/layers/attention/spatial_attention.py:185*

#### `call(self, inputs, training, return_attention_scores)`
**Module:** `layers.attention.performer_attention`

Apply Performer attention to inputs.

*📁 src/dl_techniques/layers/attention/performer_attention.py:449*

#### `call(self, query_input, kv_input, attention_mask, training)`
**Module:** `layers.attention.multi_head_cross_attention`

Forward pass through multi-head attention with optional masking and adaptive softmax or hierarchical routing.

*📁 src/dl_techniques/layers/attention/multi_head_cross_attention.py:424*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.fnet_fourier_transform`

Apply 2D Fourier Transform for token mixing.

*📁 src/dl_techniques/layers/attention/fnet_fourier_transform.py:228*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.window_attention`

*📁 src/dl_techniques/layers/attention/window_attention.py:358*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.capsule_routing_attention`

Forward pass of capsule routing self-attention.

*📁 src/dl_techniques/layers/attention/capsule_routing_attention.py:412*

#### `call(self, x, num_anchor_tokens, training)`
**Module:** `layers.attention.anchor_attention`

Apply anchor-based attention to input tensor.

*📁 src/dl_techniques/layers/attention/anchor_attention.py:423*

#### `call(self, inputs, attention_mask, layer_idx, training)`
**Module:** `layers.attention.differential_attention`

Apply differential attention mechanism.

*📁 src/dl_techniques/layers/attention/differential_attention.py:343*

#### `call(self, inputs, attention_mask, return_attention_scores, training)`
**Module:** `layers.attention.hopfield_attention`

Forward pass of the Hopfield attention layer.

*📁 src/dl_techniques/layers/attention/hopfield_attention.py:402*

#### `call(self, query_input, kv_input, attention_mask, training)`
**Module:** `layers.attention.multi_head_latent_attention`

Forward pass through the Multi-Head Latent Attention layer.

*📁 src/dl_techniques/layers/attention/multi_head_latent_attention.py:506*

#### `call(self, inputs, training, attention_mask, return_attention_weights)`
**Module:** `layers.attention.group_query_attention`

Apply grouped query attention. Supports 3D (B, S, D) and 4D (B, H, W, D) inputs.

*📁 src/dl_techniques/layers/attention/group_query_attention.py:287*

#### `call(self, inputs, training, attention_mask, return_attention_weights)`
**Module:** `layers.attention.mobile_mqa`

Forward pass of MobileMQA.

*📁 src/dl_techniques/layers/attention/mobile_mqa.py:147*

#### `call(self, inputs, training)`
**Module:** `layers.attention.convolutional_block_attention`

Apply CBAM attention to input tensor.

*📁 src/dl_techniques/layers/attention/convolutional_block_attention.py:238*

#### `call(self, inputs, attention_mask, training)`
**Module:** `layers.attention.non_local_attention`

Apply non-local attention to input features.

*📁 src/dl_techniques/layers/attention/non_local_attention.py:423*

#### `call(self, x, prev_attn_map, training)`
**Module:** `layers.attention.progressive_focused_attention`

Forward pass of Progressive Focused Attention.

*📁 src/dl_techniques/layers/attention/progressive_focused_attention.py:762*

#### `call(self, inputs, training)`
**Module:** `layers.tokenizers.bpe`

Forward pass placeholder - not implemented for string processing.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:296*

#### `call(self, inputs, training)`
**Module:** `layers.tokenizers.bpe`

Forward pass of the embedding layer.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:426*

#### `call(self, inputs)`
**Module:** `layers.ntm.baseline_ntm`

Forward pass (placeholder for layer compatibility).

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:343*

#### `call(self, inputs, state, training)`
**Module:** `layers.ntm.baseline_ntm`

Process inputs through the controller.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:737*

#### `call(self, inputs, states, training)`
**Module:** `layers.ntm.baseline_ntm`

Process one timestep of the NTM.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:974*

#### `call(self, inputs, initial_state, training)`
**Module:** `layers.ntm.baseline_ntm`

Process input sequence through NTM.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1325*

#### `call(self, inputs, state, training)`
**Module:** `layers.ntm.ntm_interface`

Process inputs through the controller.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:580*

#### `call(self, inputs, initial_state, training, return_sequences, return_state)`
**Module:** `layers.ntm.ntm_interface`

Process a sequence through the NTM.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:698*

#### `call(self, memory, controller_state, previous_weights, training)`
**Module:** `layers.ntm.base_layers`

Compute addressing weights over memory.

*📁 src/dl_techniques/layers/ntm/base_layers.py:220*

#### `call(self, memory, controller_state, previous_read_weights, previous_write_weights, training)`
**Module:** `layers.ntm.base_layers`

Perform read and write operations on memory.

*📁 src/dl_techniques/layers/ntm/base_layers.py:537*

#### `call(self, inputs, training)`
**Module:** `layers.ntm.base_layers`

Perform select-copy operations.

*📁 src/dl_techniques/layers/ntm/base_layers.py:832*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.residual_block`

Forward pass with residual connection.

*📁 src/dl_techniques/layers/ffn/residual_block.py:277*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.counting_ffn`

Forward pass computation.

*📁 src/dl_techniques/layers/ffn/counting_ffn.py:337*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.power_mlp_layer`

Forward pass implementing the dual-branch PowerMLP architecture.

*📁 src/dl_techniques/layers/ffn/power_mlp_layer.py:288*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.diff_ffn`

Forward pass through the Differential FFN layer.

*📁 src/dl_techniques/layers/ffn/diff_ffn.py:370*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.gated_mlp`

Forward pass for the GatedMLP layer.

*📁 src/dl_techniques/layers/ffn/gated_mlp.py:320*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.swiglu_ffn`

Apply SwiGLU feed-forward transformation.

*📁 src/dl_techniques/layers/ffn/swiglu_ffn.py:340*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.swin_mlp`

Forward pass of the SwinMLP layer.

*📁 src/dl_techniques/layers/ffn/swin_mlp.py:291*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.mlp`

Apply the MLP block to input tensors.

*📁 src/dl_techniques/layers/ffn/mlp.py:276*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.glu_ffn`

Forward pass implementing the GLU gating mechanism.

*📁 src/dl_techniques/layers/ffn/glu_ffn.py:334*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.orthoglu_ffn`

Forward pass for the OrthoGLU FFN.

*📁 src/dl_techniques/layers/ffn/orthoglu_ffn.py:241*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.geglu_ffn`

Forward pass for the GeGLU FFN.

*📁 src/dl_techniques/layers/ffn/geglu_ffn.py:300*

#### `call(self, inputs, training)`
**Module:** `layers.ffn.logic_ffn`

Forward pass through the logic FFN.

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:288*

#### `call(self, inputs, training)`
**Module:** `layers.embedding.patch_embedding`

Forward pass of the layer.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:210*

#### `call(self, inputs, training)`
**Module:** `layers.embedding.patch_embedding`

Convert inputs to patches and embed them.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:402*

#### `call(self, coords, training)`
**Module:** `layers.embedding.continuous_rope_embedding`

Generate continuous RoPE frequencies.

*📁 src/dl_techniques/layers/embedding/continuous_rope_embedding.py:214*

#### `call(self, input_ids, token_type_ids, training)`
**Module:** `layers.embedding.modern_bert_embeddings`

Computes the final embedding vectors.

*📁 src/dl_techniques/layers/embedding/modern_bert_embeddings.py:122*

#### `call(self, inputs, training)`
**Module:** `layers.embedding.positional_embedding`

Add positional embeddings to input tensor.

*📁 src/dl_techniques/layers/embedding/positional_embedding.py:214*

#### `call(self, input_ids, token_type_ids, position_ids, training)`
**Module:** `layers.embedding.bert_embeddings`

Apply embeddings to input tokens.

*📁 src/dl_techniques/layers/embedding/bert_embeddings.py:293*

#### `call(self, coords, training)`
**Module:** `layers.embedding.continuous_sin_cos_embedding`

Forward computation with sinusoidal embedding.

*📁 src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py:235*

#### `call(self, inputs, rope_type, training)`
**Module:** `layers.embedding.dual_rotary_position_embedding`

Apply dual rotary position embedding to input tensor.

*📁 src/dl_techniques/layers/embedding/dual_rotary_position_embedding.py:328*

#### `call(self, inputs, training)`
**Module:** `layers.embedding.rotary_position_embedding`

Apply rotary position embedding to input tensor.

*📁 src/dl_techniques/layers/embedding/rotary_position_embedding.py:313*

#### `call(self, inputs, mask)`
**Module:** `layers.embedding.positional_embedding_sine_2d`

Forward pass to generate positional encodings.

*📁 src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py:165*

#### `call(self, inputs, mask, training)`
**Module:** `layers.reasoning.hrm_reasoning_module`

Forward pass with input injection and sequential refinement.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_module.py:178*

#### `call(self, inputs, training)`
**Module:** `layers.reasoning.hrm_sparse_puzzle_embedding`

Forward pass through sparse embedding with mode-dependent behavior.

*📁 src/dl_techniques/layers/reasoning/hrm_sparse_puzzle_embedding.py:218*

#### `call(self, carry, inputs, training)`
**Module:** `layers.reasoning.hrm_reasoning_core`

Forward pass through the hierarchical reasoning core.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:587*

#### `center_positions(self)`
**Module:** `layers.radial_basis_function`

Get current positions of RBF centers.

*📁 src/dl_techniques/layers/radial_basis_function.py:329*

#### `check_component_diversity(model, x_data, mdn_layer)`
**Module:** `layers.statistics.mdn_layer`

Analyzes the diversity of mixture components for trained MDN.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:571*

#### `circular_convolution(weights, shift)`
**Module:** `layers.ntm.ntm_interface`

Perform circular convolution for location-based addressing.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:874*

#### `clone(self)`
**Module:** `layers.ntm.ntm_interface`

Create a shallow copy of the memory state.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:109*

#### `cluster_centers(self)`
**Module:** `layers.kmeans`

Get current cluster centers.

*📁 src/dl_techniques/layers/kmeans.py:667*

#### `compute_acf(self, residuals)`
**Module:** `layers.statistics.residual_acf`

Compute autocorrelation function of residuals using efficient Keras operations.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:272*

#### `compute_addressing(self, controller_output, memory_state, prev_weights)`
**Module:** `layers.ntm.baseline_ntm`

Compute attention weights using the full addressing mechanism.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:292*

#### `compute_addressing(self, controller_output, memory_state, prev_weights)`
**Module:** `layers.ntm.baseline_ntm`

Compute attention weights and write vectors.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:542*

#### `compute_addressing(self, controller_output, memory_state, prev_weights)`
**Module:** `layers.ntm.ntm_interface`

Compute attention weights using the addressing mechanism.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:473*

#### `compute_auxiliary_loss(expert_weights, gate_probs, num_experts, aux_loss_weight)`
**Module:** `layers.moe.gating`

Compute auxiliary load balancing loss for MoE training.

*📁 src/dl_techniques/layers/moe/gating.py:591*

#### `compute_entropy(self, logits)`
**Module:** `layers.blt_blocks`

Compute Shannon entropy from logits.

*📁 src/dl_techniques/layers/blt_blocks.py:498*

#### `compute_free_bits_kl_loss(bit_logits, num_bits, kappa, reduction)`
**Module:** `layers.transformers.free_transformer`

Compute KL divergence loss with free bits thresholding.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:1077*

#### `compute_input_quality(self, inputs)`
**Module:** `layers.neuro_grid`

Compute comprehensive quality measures for input data based on probabilistic addressing behavior.

*📁 src/dl_techniques/layers/neuro_grid.py:1005*

#### `compute_kernel_matrix(self, x, level)`
**Module:** `layers.statistics.deep_kernel_pca`

Compute kernel matrix for a given level with numerical stability.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:408*

#### `compute_kl_divergence_uniform_prior(bit_logits, num_bits, axis)`
**Module:** `layers.transformers.free_transformer`

Compute KL divergence between encoder posterior Q(Z|S) and uniform prior P(Z).

*📁 src/dl_techniques/layers/transformers/free_transformer.py:985*

#### `compute_mask(self, inputs, mask)`
**Module:** `layers.fnet_encoder_block`

Propagate the input mask unchanged.

*📁 src/dl_techniques/layers/fnet_encoder_block.py:286*

#### `compute_mask(self, inputs, mask)`
**Module:** `layers.attention.fnet_fourier_transform`

Propagate the input mask.

*📁 src/dl_techniques/layers/attention/fnet_fourier_transform.py:263*

#### `compute_ood_scores(self, data)`
**Module:** `layers.experimental.band_rms_ood`

Compute OOD scores for input data.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:700*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.mobile_one_block`

Compute output shape.

*📁 src/dl_techniques/layers/mobile_one_block.py:317*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.depthwise_separable_block`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/depthwise_separable_block.py:364*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.sampling`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/sampling.py:242*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.hanc_layer`

Compute output shape.

*📁 src/dl_techniques/layers/hanc_layer.py:318*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.laplacian_filter`

Compute the output shape.

*📁 src/dl_techniques/layers/laplacian_filter.py:204*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.laplacian_filter`

Compute the output shape.

*📁 src/dl_techniques/layers/laplacian_filter.py:458*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.multi_level_feature_compilation`

Compute output shapes.

*📁 src/dl_techniques/layers/multi_level_feature_compilation.py:405*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/yolo12_blocks.py:207*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/yolo12_blocks.py:474*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/yolo12_blocks.py:650*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/yolo12_blocks.py:797*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/yolo12_blocks.py:980*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/yolo12_blocks.py:1182*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_heads`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/yolo12_heads.py:336*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_heads`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/yolo12_heads.py:778*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.yolo12_heads`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/yolo12_heads.py:1107*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.radial_basis_function`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/radial_basis_function.py:287*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.spatial_layer`

Computes the output shape of the layer.

*📁 src/dl_techniques/layers/spatial_layer.py:273*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.one_hot_encoding`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/one_hot_encoding.py:120*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.neuro_grid`

Compute output shape for both 2D and 3D inputs.

*📁 src/dl_techniques/layers/neuro_grid.py:760*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.dynamic_conv2d`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/dynamic_conv2d.py:458*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.shearlet_transform`

Compute output shape.

*📁 src/dl_techniques/layers/shearlet_transform.py:411*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.squeeze_excitation`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/squeeze_excitation.py:370*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.standard_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/standard_blocks.py:368*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.standard_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/standard_blocks.py:606*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.standard_blocks`

Compute output shape (same as input for residual connection).

*📁 src/dl_techniques/layers/standard_blocks.py:815*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.standard_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/standard_blocks.py:1060*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.standard_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/standard_blocks.py:1344*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.sequence_pooling`

Compute output shape.

*📁 src/dl_techniques/layers/sequence_pooling.py:262*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.sequence_pooling`

Compute output shape.

*📁 src/dl_techniques/layers/sequence_pooling.py:398*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.sequence_pooling`

Compute output shape based on pooling strategy.

*📁 src/dl_techniques/layers/sequence_pooling.py:812*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_core`

Compute output shapes.

*📁 src/dl_techniques/layers/blt_core.py:564*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.fnet_encoder_block`

Encoder block preserves input shape.

*📁 src/dl_techniques/layers/fnet_encoder_block.py:294*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.conditional_output_layer`

Compute the layer's output shape.

*📁 src/dl_techniques/layers/conditional_output_layer.py:195*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.complex_layers`

Compute the output shape of the convolution.

*📁 src/dl_techniques/layers/complex_layers.py:370*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.complex_layers`

Compute the output shape of the dense layer.

*📁 src/dl_techniques/layers/complex_layers.py:575*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.complex_layers`

Compute the output shape (same as input for activation).

*📁 src/dl_techniques/layers/complex_layers.py:690*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.complex_layers`

Compute the output shape of the pooling layer.

*📁 src/dl_techniques/layers/complex_layers.py:848*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.complex_layers`

Compute the output shape (same as input for dropout).

*📁 src/dl_techniques/layers/complex_layers.py:996*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.complex_layers`

Compute the output shape of the global pooling layer.

*📁 src/dl_techniques/layers/complex_layers.py:1119*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.anchor_generator`

Compute output shapes for anchors and strides tensors.

*📁 src/dl_techniques/layers/anchor_generator.py:243*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.haar_wavelet_decomposition`

Compute output shapes for all frequency bands.

*📁 src/dl_techniques/layers/haar_wavelet_decomposition.py:339*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tabm_blocks`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tabm_blocks.py:121*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tabm_blocks`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tabm_blocks.py:254*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tabm_blocks`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tabm_blocks.py:354*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tabm_blocks`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tabm_blocks.py:471*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tabm_blocks`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tabm_blocks.py:577*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.kan_linear`

Compute output shape from input shape.

*📁 src/dl_techniques/layers/kan_linear.py:451*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.convnext_v2_block`

Compute the output shape of the ConvNextV2 block.

*📁 src/dl_techniques/layers/convnext_v2_block.py:436*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.io_preparation`

Output shape is identical to input shape.

*📁 src/dl_techniques/layers/io_preparation.py:99*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.io_preparation`

Output shape is identical to input shape.

*📁 src/dl_techniques/layers/io_preparation.py:234*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.io_preparation`

Output shape is identical to input shape.

*📁 src/dl_techniques/layers/io_preparation.py:368*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.io_preparation`

Output shape is identical to input shape.

*📁 src/dl_techniques/layers/io_preparation.py:529*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.convnext_v1_block`

Compute the output shape of the ConvNext block.

*📁 src/dl_techniques/layers/convnext_v1_block.py:407*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.patch_merging`

Compute output shape for shape inference.

*📁 src/dl_techniques/layers/patch_merging.py:219*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.fft_layers`

Compute output shape (channels are doubled).

*📁 src/dl_techniques/layers/fft_layers.py:93*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.fft_layers`

Compute output shape (channels are halved).

*📁 src/dl_techniques/layers/fft_layers.py:204*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.orthoblock`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/orthoblock.py:338*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.pixel_shuffle`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/pixel_shuffle.py:238*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.res_path`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/res_path.py:231*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.mps_layer`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/mps_layer.py:381*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.film`

Compute output shape (same as content tensor).

*📁 src/dl_techniques/layers/film.py:421*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.hanc_block`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/hanc_block.py:367*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.repmixer_block`

Output shape is identical to input shape.

*📁 src/dl_techniques/layers/repmixer_block.py:381*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.repmixer_block`

Compute output shape through all blocks.

*📁 src/dl_techniques/layers/repmixer_block.py:584*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.bitlinear_layer`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/bitlinear_layer.py:494*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.stochastic_gradient`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/stochastic_gradient.py:163*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.gaussian_pyramid`

Compute the output shapes for all pyramid levels.

*📁 src/dl_techniques/layers/gaussian_pyramid.py:305*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.bias_free_conv1d`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:229*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.bias_free_conv1d`

Compute output shape.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:492*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.vector_quantizer`

Compute output shape (same as input shape).

*📁 src/dl_techniques/layers/vector_quantizer.py:451*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.restricted_boltzmann_machine`

Compute output shape for forward pass.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:604*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.convolutional_kan`

Compute output shape based on input shape and layer parameters.

*📁 src/dl_techniques/layers/convolutional_kan.py:425*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.kmeans`

Compute shape of layer output.

*📁 src/dl_techniques/layers/kmeans.py:401*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.capsules`

Compute output shape based on input shape.

*📁 src/dl_techniques/layers/capsules.py:278*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.capsules`

Compute output shape based on input shape.

*📁 src/dl_techniques/layers/capsules.py:626*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.capsules`

Compute output shape based on input shape.

*📁 src/dl_techniques/layers/capsules.py:889*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.gated_delta_net`

Compute the output shape given input shape.

*📁 src/dl_techniques/layers/gated_delta_net.py:497*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.bias_free_conv2d`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:249*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.bias_free_conv2d`

Compute output shape.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:532*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.mothnet_blocks`

Compute output shape for this layer.

*📁 src/dl_techniques/layers/mothnet_blocks.py:242*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.mothnet_blocks`

Compute output shape for this layer.

*📁 src/dl_techniques/layers/mothnet_blocks.py:529*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.mothnet_blocks`

Compute output shape for this layer.

*📁 src/dl_techniques/layers/mothnet_blocks.py:808*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.selective_gradient_mask`

Compute the output shape.

*📁 src/dl_techniques/layers/selective_gradient_mask.py:223*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.clahe`

The output shape is the same as the input shape.

*📁 src/dl_techniques/layers/clahe.py:283*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tversky_projection`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tversky_projection.py:291*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.stochastic_depth`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/stochastic_depth.py:193*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.hierarchical_mlp_stem`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/hierarchical_mlp_stem.py:254*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.layer_scale`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/layer_scale.py:281*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/blt_blocks.py:519*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/blt_blocks.py:695*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/blt_blocks.py:959*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/blt_blocks.py:1138*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/blt_blocks.py:1276*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.blt_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/blt_blocks.py:1556*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.random_fourier_features`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/random_fourier_features.py:361*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.modality_projection`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/modality_projection.py:257*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.sparse_autoencoder`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:880*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.eomt_mask`

Compute output shapes.

*📁 src/dl_techniques/layers/eomt_mask.py:497*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.canny`

The output shape is the same as the input shape.

*📁 src/dl_techniques/layers/canny.py:330*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.universal_inverted_bottleneck`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/universal_inverted_bottleneck.py:554*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.global_sum_pool_2d`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/global_sum_pool_2d.py:175*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.fractal_block`

Compute output shape of the FractalBlock.

*📁 src/dl_techniques/layers/fractal_block.py:371*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.rigid_simplex_layer`

Compute output shape from input shape.

*📁 src/dl_techniques/layers/rigid_simplex_layer.py:369*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.router`

Computes the output shape of the layer.

*📁 src/dl_techniques/layers/router.py:340*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.progressive_focused_transformer`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/transformers/progressive_focused_transformer.py:866*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.text_encoder`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:900*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.text_decoder`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/transformers/text_decoder.py:487*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.eomt_transformer`

Compute output shape (same as input).

*📁 src/dl_techniques/layers/transformers/eomt_transformer.py:474*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.swin_transformer_block`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/transformers/swin_transformer_block.py:587*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.free_transformer`

Compute output shape: replace last dimension with 2^num_bits.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:243*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.free_transformer`

Compute output shape(s) of the layer.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:932*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.perceiver_transformer`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/transformers/perceiver_transformer.py:350*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.swin_conv_block`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/transformers/swin_conv_block.py:681*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.vision_encoder`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:756*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.transformers.transformer`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/transformers/transformer.py:635*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.hierarchical_memory_system`

Compute the output shape of the hierarchical memory system.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:341*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.field_embeddings`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:220*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.field_embeddings`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:421*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.field_embeddings`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:606*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.field_embeddings`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:806*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Compute output shape.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:238*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Compute output shape.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:513*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Compute output shape.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:672*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.contextual_memory`

Compute output shapes.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:890*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.mst_correlation_filter`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:359*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.band_rms_ood`

Compute output shape (unchanged from input).

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:435*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.contextual_counter_ffn`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/experimental/contextual_counter_ffn.py:368*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.experimental.graph_mann`

Compute output shape.

*📁 src/dl_techniques/layers/experimental/graph_mann.py:412*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:272*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:381*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:503*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:625*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:762*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:941*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1028*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.nlp_heads.factory`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1188*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.ema_layer`

Compute output shape.

*📁 src/dl_techniques/layers/time_series/ema_layer.py:144*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.ema_layer`

Compute output shape(s).

*📁 src/dl_techniques/layers/time_series/ema_layer.py:392*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.nbeats_blocks`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:430*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.adaptive_lag_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/time_series/adaptive_lag_attention.py:338*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.quantile_head_fixed_io`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/time_series/quantile_head_fixed_io.py:254*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.deepar_blocks`

Compute output shapes for both mu and sigma.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:223*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.deepar_blocks`

Compute output shapes for both mu and alpha.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:343*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.quantile_head_variable_io`

Compute the output shape of the layer given an input shape.

*📁 src/dl_techniques/layers/time_series/quantile_head_variable_io.py:352*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:438*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.xlstm_blocks`

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:870*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.temporal_fusion`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/time_series/temporal_fusion.py:380*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.mixed_sequential_block`

Output shape is the same as the input shape.

*📁 src/dl_techniques/layers/time_series/mixed_sequential_block.py:573*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:110*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:283*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:509*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:860*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.time_series.prism_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:1046*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.physics.approximate_lagrange_layer`

Compute output shape for accelerations.

*📁 src/dl_techniques/layers/physics/approximate_lagrange_layer.py:293*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.physics.lagrange_layer`

Compute output shape for accelerations.

*📁 src/dl_techniques/layers/physics/lagrange_layer.py:252*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.relu_k`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/relu_k.py:208*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.hard_sigmoid`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/activations/hard_sigmoid.py:169*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.thresh_max`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/thresh_max.py:228*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.basis_function`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/basis_function.py:196*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.differentiable_step`

Compute the output shape of the layer. Since this is an element-wise operation, the output shape is identical to the input shape.

*📁 src/dl_techniques/layers/activations/differentiable_step.py:240*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.probability_output`

Compute output shape based on strategy.

*📁 src/dl_techniques/layers/activations/probability_output.py:275*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.routing_probabilities`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/routing_probabilities.py:691*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.adaptive_softmax`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/activations/adaptive_softmax.py:319*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.monotonicity_layer`

Output shape is the same as input shape.

*📁 src/dl_techniques/layers/activations/monotonicity_layer.py:570*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.mish`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/mish.py:220*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.mish`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/mish.py:393*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.sparsemax`

Compute output shape (same as input shape).

*📁 src/dl_techniques/layers/activations/sparsemax.py:238*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.squash`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/squash.py:203*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.routing_probabilities_hierarchical`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/activations/routing_probabilities_hierarchical.py:409*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.hard_swish`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/activations/hard_swish.py:202*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.activations.golu`

Returns the output shape, which is identical to the input shape.

*📁 src/dl_techniques/layers/activations/golu.py:172*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.memory.mann`

Compute output shape.

*📁 src/dl_techniques/layers/memory/mann.py:455*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.memory.som_nd_soft_layer`

Compute output tensor shape.

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:853*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.memory.som_nd_layer`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:512*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.supernode_pooling`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/geometric/supernode_pooling.py:424*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.parallel_transport`

Compute output shape.

*📁 src/dl_techniques/layers/geometric/fields/parallel_transport.py:327*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.holonomy_layer`

Compute output shape.

*📁 src/dl_techniques/layers/geometric/fields/holonomy_layer.py:405*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.gauge_invariant_attention`

Compute output shape.

*📁 src/dl_techniques/layers/geometric/fields/gauge_invariant_attention.py:522*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.connection_layer`

Compute the output shape.

*📁 src/dl_techniques/layers/geometric/fields/connection_layer.py:401*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.manifold_stress`

Compute output shapes.

*📁 src/dl_techniques/layers/geometric/fields/manifold_stress.py:443*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Compute output shape.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:484*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.geometric.fields.field_embedding`

Compute the output shapes.

*📁 src/dl_techniques/layers/geometric/fields/field_embedding.py:304*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.zero_centered_rms_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/zero_centered_rms_norm.py:391*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.adaptive_band_rms`

Compute output shape (same as input).

*📁 src/dl_techniques/layers/norms/adaptive_band_rms.py:423*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.logit_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/logit_norm.py:248*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.band_rms`

Compute the shape of output tensor.

*📁 src/dl_techniques/layers/norms/band_rms.py:293*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.global_response_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/global_response_norm.py:264*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.band_logit_norm`

Compute the shape of the output tensor.

*📁 src/dl_techniques/layers/norms/band_logit_norm.py:167*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.max_logit_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:193*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.max_logit_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:404*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.max_logit_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:623*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.rms_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/rms_norm.py:348*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.zero_centered_band_rms_norm`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py:424*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.norms.dynamic_tanh`

Compute output shape (same as input shape).

*📁 src/dl_techniques/layers/norms/dynamic_tanh.py:232*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.logic.arithmetic_operators`

Compute output shape.

*📁 src/dl_techniques/layers/logic/arithmetic_operators.py:422*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.logic.logic_operators`

Compute output shape.

*📁 src/dl_techniques/layers/logic/logic_operators.py:406*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.logic.neural_circuit`

Compute output shape.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:287*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.logic.neural_circuit`

Compute output shape.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:490*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.moe.layer`

Compute the output shape of the MoE layer.

*📁 src/dl_techniques/layers/moe/layer.py:484*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.moe.experts`

Compute the output shape of the expert.

*📁 src/dl_techniques/layers/moe/experts.py:59*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.moe.experts`

Compute output shape by delegating to the FFN block.

*📁 src/dl_techniques/layers/moe/experts.py:182*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.fusion.multimodal_fusion`

Compute the output shape for given input shapes.

*📁 src/dl_techniques/layers/fusion/multimodal_fusion.py:938*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.statistics.moving_std`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/statistics/moving_std.py:289*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.statistics.residual_acf`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:458*

#### `compute_output_shape(self, input_shapes)`
**Module:** `layers.statistics.normalizing_flow`

Compute output shape (same as data input shape).

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:375*

#### `compute_output_shape(self, input_shapes)`
**Module:** `layers.statistics.normalizing_flow`

Compute output shapes for the tuple (z, log_det_jacobian).

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:771*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.statistics.invertible_kernel_pca`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:631*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.statistics.scaler`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/statistics/scaler.py:609*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.statistics.deep_kernel_pca`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:647*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.statistics.mdn_layer`

Computes the output shape of the layer.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:332*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py:301*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.graph_neural_network`

Compute output shape based on aggregation type.

*📁 src/dl_techniques/layers/graphs/graph_neural_network.py:537*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.entity_graph_refinement`

Compute output tensor shapes for the three returned tensors.

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:777*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:228*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:575*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Compute output shape.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:952*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.graphs.fermi_diract_decoder`

Compute output shape given input shapes.

*📁 src/dl_techniques/layers/graphs/fermi_diract_decoder.py:216*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.channel_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/channel_attention.py:287*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:161*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:264*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.rpc_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/rpc_attention.py:555*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.multi_head_attention`

Compute output shape - same as input shape for self-attention.

*📁 src/dl_techniques/layers/attention/multi_head_attention.py:235*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.shared_weights_cross_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/shared_weights_cross_attention.py:415*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.ring_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/ring_attention.py:589*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.perceiver_attention`

Compute the output shape - same as query input shape.

*📁 src/dl_techniques/layers/attention/perceiver_attention.py:311*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.gated_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/gated_attention.py:579*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.spatial_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/spatial_attention.py:221*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.performer_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/performer_attention.py:521*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.multi_head_cross_attention`

Compute output shape - returns query input shape.

*📁 src/dl_techniques/layers/attention/multi_head_cross_attention.py:579*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.fnet_fourier_transform`

Fourier transform preserves input shape.

*📁 src/dl_techniques/layers/attention/fnet_fourier_transform.py:271*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.window_attention`

Compute the output shape, which is identical to the input shape.

*📁 src/dl_techniques/layers/attention/window_attention.py:480*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.capsule_routing_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/capsule_routing_attention.py:701*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.anchor_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/anchor_attention.py:607*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.differential_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/differential_attention.py:402*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.hopfield_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/hopfield_attention.py:520*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.multi_head_latent_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/multi_head_latent_attention.py:762*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.group_query_attention`

*📁 src/dl_techniques/layers/attention/group_query_attention.py:386*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.convolutional_block_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/convolutional_block_attention.py:274*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.non_local_attention`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/attention/non_local_attention.py:486*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.attention.progressive_focused_attention`

Compute output shape of the layer.

*📁 src/dl_techniques/layers/attention/progressive_focused_attention.py:1049*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tokenizers.bpe`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:354*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.tokenizers.bpe`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:440*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:359*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:597*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:771*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1140*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.baseline_ntm`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1376*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.ntm_interface`

Compute output shape of the NTM layer.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:780*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.base_layers`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/base_layers.py:284*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.base_layers`

Compute output shapes.

*📁 src/dl_techniques/layers/ntm/base_layers.py:609*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ntm.base_layers`

Compute output shape.

*📁 src/dl_techniques/layers/ntm/base_layers.py:906*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.residual_block`

Compute output shape.

*📁 src/dl_techniques/layers/ffn/residual_block.py:306*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.counting_ffn`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/counting_ffn.py:402*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.power_mlp_layer`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/power_mlp_layer.py:321*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.diff_ffn`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/diff_ffn.py:416*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.gated_mlp`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/gated_mlp.py:358*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.swiglu_ffn`

Compute output shape (same as input shape for FFN).

*📁 src/dl_techniques/layers/ffn/swiglu_ffn.py:384*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.swin_mlp`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/swin_mlp.py:327*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.mlp`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/mlp.py:307*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.glu_ffn`

Compute output shape transformation.

*📁 src/dl_techniques/layers/ffn/glu_ffn.py:365*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.orthoglu_ffn`

Computes the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/orthoglu_ffn.py:259*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.geglu_ffn`

Computes the output shape of the layer.

*📁 src/dl_techniques/layers/ffn/geglu_ffn.py:331*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.ffn.logic_ffn`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:344*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.patch_embedding`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:237*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.patch_embedding`

Compute output shape after patch embedding.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:421*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.continuous_rope_embedding`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/embedding/continuous_rope_embedding.py:271*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.positional_embedding`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/embedding/positional_embedding.py:248*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.bert_embeddings`

Compute output shape given input shape.

*📁 src/dl_techniques/layers/embedding/bert_embeddings.py:343*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.continuous_sin_cos_embedding`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py:299*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.dual_rotary_position_embedding`

Compute output shape - identical to input shape.

*📁 src/dl_techniques/layers/embedding/dual_rotary_position_embedding.py:411*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.rotary_position_embedding`

Compute output shape - identical to input shape.

*📁 src/dl_techniques/layers/embedding/rotary_position_embedding.py:410*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.embedding.positional_embedding_sine_2d`

Compute the output shape of the layer.

*📁 src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py:201*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.reasoning.hrm_reasoning_module`

Compute output shape, which is the shape of the `hidden_states` input.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_module.py:206*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.reasoning.hrm_sparse_puzzle_embedding`

Compute output shape by appending embedding dimension.

*📁 src/dl_techniques/layers/reasoning/hrm_sparse_puzzle_embedding.py:259*

#### `compute_output_shape(self, input_shape)`
**Module:** `layers.reasoning.hrm_reasoning_core`

Compute output shapes for new_carry and outputs.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:651*

#### `compute_patch_ids(self, patch_lengths)`
**Module:** `layers.blt_blocks`

Convert patch lengths to patch IDs for each position.

*📁 src/dl_techniques/layers/blt_blocks.py:651*

#### `compute_random_features(self, inputs)`
**Module:** `layers.statistics.invertible_kernel_pca`

Compute Random Fourier Features for the input.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:392*

#### `compute_reconstruction_error(self, inputs)`
**Module:** `layers.statistics.invertible_kernel_pca`

Compute reconstruction error for the inputs.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:607*

#### `compute_separation_loss(self, confidence_threshold)`
**Module:** `layers.experimental.band_rms_ood`

Compute separation loss to encourage separation between high/low confidence samples.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:508*

#### `compute_shell_preference_loss(self, confidence_threshold)`
**Module:** `layers.experimental.band_rms_ood`

Compute shell preference loss to encourage high-confidence samples to outer shell.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:477*

#### `compute_z_loss(gate_logits, z_loss_weight)`
**Module:** `layers.moe.gating`

Compute router z-loss for entropy regularization.

*📁 src/dl_techniques/layers/moe/gating.py:630*

#### `condition(i, state, outputs)`
**Module:** `layers.gated_delta_net`

*📁 src/dl_techniques/layers/gated_delta_net.py:423*

#### `content_addressing(self, key, beta, memory)`
**Module:** `layers.ntm.baseline_ntm`

Compute content-based attention weights.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:271*

#### `content_addressing(self, key, beta, memory)`
**Module:** `layers.ntm.baseline_ntm`

Compute content-based attention weights.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:521*

#### `content_addressing(self, key, beta, memory)`
**Module:** `layers.ntm.ntm_interface`

Compute content-based attention weights.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:494*

#### `contrastive_divergence(self, visible_data)`
**Module:** `layers.restricted_boltzmann_machine`

Train RBM using Contrastive Divergence algorithm.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:443*

#### `conv2d_wrapper(input_layer, conv_params, bn_params, ln_params, dropout_params, dropout_2d_params, conv_type)`
**Module:** `layers.conv2d_builder`

Creates a wrapped convolution layer with optional normalization, activation, and regularization.

*📁 src/dl_techniques/layers/conv2d_builder.py:247*

#### `conv_output_length(input_length, kernel_size, padding, stride, dilation)`
**Module:** `layers.dynamic_conv2d`

Calculate output length for convolution dimension.

*📁 src/dl_techniques/layers/dynamic_conv2d.py:479*

#### `cosine_similarity(query, keys, epsilon)`
**Module:** `layers.ntm.ntm_interface`

Compute cosine similarity between query and keys.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:836*

#### `create_activation_from_config(config)`
**Module:** `layers.activations.factory`

Create an activation layer from a configuration dictionary.

*📁 src/dl_techniques/layers/activations/factory.py:563*

#### `create_activation_layer(activation_type, name)`
**Module:** `layers.activations.factory`

Factory function for creating activation layers with a unified interface.

*📁 src/dl_techniques/layers/activations/factory.py:458*

#### `create_adaptive_softmax_window_attention(dim, window_size, num_heads, partition_mode)`
**Module:** `layers.attention.window_attention`

Creates a window attention layer with adaptive temperature softmax.

*📁 src/dl_techniques/layers/attention/window_attention.py:677*

#### `create_attention_from_config(config)`
**Module:** `layers.attention.factory`

Create an attention layer from a configuration dictionary.

*📁 src/dl_techniques/layers/attention/factory.py:999*

#### `create_attention_layer(attention_type, name)`
**Module:** `layers.attention.factory`

Factory function for creating attention layers with unified interface and validation.

*📁 src/dl_techniques/layers/attention/factory.py:872*

#### `create_bert_encoder(vocab_size, embed_dim, depth, num_heads, max_seq_len)`
**Module:** `layers.transformers.text_encoder`

Create BERT-style encoder configuration.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:1078*

#### `create_bpe_pipeline(texts, vocab_size, embedding_dim, max_length, do_lower_case, min_frequency)`
**Module:** `layers.tokenizers.bpe`

Create a complete BPE tokenization and embedding pipeline.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:469*

#### `create_contextual_memory_model(config, include_downstream_modules)`
**Module:** `layers.experimental.contextual_memory`

Create a complete model with Contextual Memory Bank.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:936*

#### `create_efficient_encoder(vocab_size, embed_dim, depth, num_heads, max_seq_len)`
**Module:** `layers.transformers.text_encoder`

Create efficient encoder for resource-constrained environments.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:1165*

#### `create_embedding_from_config(config)`
**Module:** `layers.embedding.factory`

Create an embedding layer from a configuration dictionary.

*📁 src/dl_techniques/layers/embedding/factory.py:371*

#### `create_embedding_layer(embedding_type, name)`
**Module:** `layers.embedding.factory`

Factory function for creating embedding layers with a unified interface.

*📁 src/dl_techniques/layers/embedding/factory.py:269*

#### `create_enhancement_head(task_type)`
**Module:** `layers.vision_heads.factory`

Create enhancement-specific heads (denoising, super-resolution, etc.).

*📁 src/dl_techniques/layers/vision_heads/factory.py:890*

#### `create_evidence_supported_language_model(vocab_size, embed_dim, max_seq_len)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

Create a complete evidence-supported language model.

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:902*

#### `create_expert(expert_type)`
**Module:** `layers.moe.experts`

Factory function to create FFN expert networks.

*📁 src/dl_techniques/layers/moe/experts.py:210*

#### `create_ffn_from_config(config)`
**Module:** `layers.ffn.factory`

Create FFN layer from configuration dictionary.

*📁 src/dl_techniques/layers/ffn/factory.py:532*

#### `create_ffn_layer(ffn_type, name)`
**Module:** `layers.ffn.factory`

Factory function for creating FFN layers with unified interface.

*📁 src/dl_techniques/layers/ffn/factory.py:420*

#### `create_ffn_moe(num_experts, ffn_config, top_k, gating_type, aux_loss_weight)`
**Module:** `layers.moe.layer`

Convenience function to create FFN-based MoE layers.

*📁 src/dl_techniques/layers/moe/layer.py:533*

#### `create_field_layer(layer_type, name)`
**Module:** `layers.geometric.fields`

Factory function to create field-based layers.

*📁 src/dl_techniques/layers/geometric/fields/__init__.py:185*

#### `create_field_layer_from_config(config)`
**Module:** `layers.geometric.fields`

Create a field layer from a configuration dictionary.

*📁 src/dl_techniques/layers/geometric/fields/__init__.py:260*

#### `create_gating(gating_type, num_experts)`
**Module:** `layers.moe.gating`

Factory function to create gating networks.

*📁 src/dl_techniques/layers/moe/gating.py:657*

#### `create_grid_window_attention(dim, window_size, num_heads)`
**Module:** `layers.attention.window_attention`

Creates a standard spatial window attention layer (Swin-style).

*📁 src/dl_techniques/layers/attention/window_attention.py:560*

#### `create_kan_key_window_attention(dim, window_size, num_heads, partition_mode)`
**Module:** `layers.attention.window_attention`

Creates a window attention layer with a non-linear KAN Key projection.

*📁 src/dl_techniques/layers/attention/window_attention.py:635*

#### `create_logic_ffn_regularized(output_dim, logic_dim, l2_reg)`
**Module:** `layers.ffn.logic_ffn`

Create a LogicFFN layer with L2 regularization.

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:404*

#### `create_logic_ffn_standard(output_dim, logic_dim)`
**Module:** `layers.ffn.logic_ffn`

Create a standard LogicFFN layer with recommended settings.

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:386*

#### `create_manokhin_compliant_model(input_shape, forecast_length, hidden_units, gate_hidden_units, gate_activation)`
**Module:** `layers.time_series.forecasting_layers`

Create a model that structurally enforces Manokhin's forecasting principles.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:514*

#### `create_modern_encoder(vocab_size, embed_dim, depth, num_heads, max_seq_len)`
**Module:** `layers.transformers.text_encoder`

Create modern encoder with advanced components.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:1136*

#### `create_multi_task_head(task_configuration)`
**Module:** `layers.vision_heads.factory`

Create a multi-task head from task configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:961*

#### `create_multi_task_nlp_head(task_configs, input_dim)`
**Module:** `layers.nlp_heads.factory`

Create a multi-task NLP head from task configurations.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1407*

#### `create_multi_task_vlm_head(task_configs)`
**Module:** `layers.vlm_heads.factory`

Creates a multi-task VLM head from task configurations.

*📁 src/dl_techniques/layers/vlm_heads/factory.py:679*

#### `create_nlp_head(task_config, input_dim)`
**Module:** `layers.nlp_heads.factory`

Factory function to create NLP task heads.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1358*

#### `create_normalization_from_config(config)`
**Module:** `layers.norms.factory`

Create a normalization layer from a configuration dictionary.

*📁 src/dl_techniques/layers/norms/factory.py:465*

#### `create_normalization_layer(normalization_type, name, epsilon)`
**Module:** `layers.norms.factory`

Create a normalization layer based on the specified type with customizable parameters.

*📁 src/dl_techniques/layers/norms/factory.py:43*

#### `create_ntm(memory_size, memory_dim, output_dim, controller_dim, controller_type, num_read_heads, num_write_heads, shift_range, return_sequences, return_state)`
**Module:** `layers.ntm.baseline_ntm`

Factory function to create a Neural Turing Machine layer.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1434*

#### `create_roberta_encoder(vocab_size, embed_dim, depth, num_heads, max_seq_len)`
**Module:** `layers.transformers.text_encoder`

Create RoBERTa-style encoder configuration.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:1107*

#### `create_siglip_encoder(img_size, patch_size, embed_dim, depth, num_heads)`
**Module:** `layers.transformers.vision_encoder`

Create SigLIP-style encoder configuration.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:947*

#### `create_sparse_autoencoder(d_input, d_latent, variant, expansion_factor)`
**Module:** `layers.sparse_autoencoder`

Factory function to create Sparse Autoencoder with common configurations.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:959*

#### `create_text_encoder(vocab_size, embed_dim, depth, num_heads, max_seq_len, embedding_type, positional_type, attention_type, normalization_type, normalization_position, ffn_type)`
**Module:** `layers.transformers.text_encoder`

Factory function to create a TextEncoder with validated parameters.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:981*

#### `create_vision_encoder(img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, patch_embed_type, attention_type, normalization_type, normalization_position, ffn_type, use_cls_token, output_mode, dropout)`
**Module:** `layers.transformers.vision_encoder`

Factory function to create a VisionEncoder with validated parameters.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:819*

#### `create_vision_head(task_type)`
**Module:** `layers.vision_heads.factory`

Factory function to create vision_heads task heads.

*📁 src/dl_techniques/layers/vision_heads/factory.py:809*

#### `create_vit_encoder(img_size, patch_size, embed_dim, depth, num_heads)`
**Module:** `layers.transformers.vision_encoder`

Create standard ViT encoder configuration.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:922*

#### `create_vlm_head(task_config)`
**Module:** `layers.vlm_heads.factory`

Factory function to create VLM task heads.

*📁 src/dl_techniques/layers/vlm_heads/factory.py:658*

#### `create_zigzag_window_attention(dim, window_size, num_heads)`
**Module:** `layers.attention.window_attention`

Creates a window attention layer with zigzag partitioning.

*📁 src/dl_techniques/layers/attention/window_attention.py:597*

#### `curvature(self)`
**Module:** `layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer`

Get current curvature value c > 0.

*📁 src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py:229*

#### `decode(self, latents)`
**Module:** `layers.sparse_autoencoder`

Decode sparse latents back to input space.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:657*

#### `decode_tokens(self, token_ids)`
**Module:** `layers.tokenizers.bpe`

Decode token IDs back to text.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:329*

#### `delta_rule_update(self, q, k, v, alpha, beta, training)`
**Module:** `layers.gated_delta_net`

Apply gated delta rule update using `keras.ops.while_loop`.

*📁 src/dl_techniques/layers/gated_delta_net.py:403*

#### `denormalize(self, scaled_inputs)`
**Module:** `layers.statistics.scaler`

Apply denormalization (alias for inverse_transform).

*📁 src/dl_techniques/layers/statistics/scaler.py:520*

#### `do_interpolate()`
**Module:** `layers.time_series.prism_blocks`

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:428*

#### `downsample(input_layer, downsample_type, conv_params, bn_params, ln_params)`
**Module:** `layers.downsample`

Applies downsampling operation to the input layer based on specified strategy.

*📁 src/dl_techniques/layers/downsample.py:53*

#### `elu_plus_one_plus_epsilon(x)`
**Module:** `layers.activations.expanded_activations`

Enhanced ELU activation to ensure positive values for rate parameters.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:522*

#### `empty_carry(self, batch_size)`
**Module:** `layers.blt_core`

Create empty carry state for byte-level reasoning.

*📁 src/dl_techniques/layers/blt_core.py:423*

#### `empty_carry(self, batch_size)`
**Module:** `layers.reasoning.hrm_reasoning_core`

Create empty carry state for initialization.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:537*

#### `encode(self, inputs, training)`
**Module:** `layers.sparse_autoencoder`

Encode inputs to pre-activation latent space.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:362*

#### `estimate_confidence(self, features, training)`
**Module:** `layers.experimental.band_rms_ood`

Estimate confidence based on the specified method.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:234*

#### `estimate_noise_level(self, inputs)`
**Module:** `layers.statistics.invertible_kernel_pca`

Estimate noise level in the input.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:755*

#### `evaluate_detection(self, id_data, ood_data)`
**Module:** `layers.experimental.band_rms_ood`

Evaluate OOD detection performance.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:818*

#### `extract_components(self, kernel_matrix, projection_matrix, eigenvalues, num_components, training)`
**Module:** `layers.statistics.deep_kernel_pca`

Extract principal components using kernel PCA with eigendecomposition.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:489*

#### `extract_hierarchies(graph, entity_mask, threshold)`
**Module:** `layers.graphs.entity_graph_refinement`

Extract hierarchical relationships from the learned entity graph.

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:893*

#### `filter_by_quality_threshold(self, inputs, quality_threshold, quality_measure)`
**Module:** `layers.neuro_grid`

Intelligently filter and partition input data based on configurable quality thresholds.

*📁 src/dl_techniques/layers/neuro_grid.py:1403*

#### `find_best_matching_units(self, inputs)`
**Module:** `layers.neuro_grid`

Find Best Matching Units (BMUs) for given inputs in SOM terminology.

*📁 src/dl_techniques/layers/neuro_grid.py:927*

#### `fit_threshold(self, id_data, fpr_target, batch_size)`
**Module:** `layers.experimental.band_rms_ood`

Fit detection threshold based on in-distribution data.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:754*

#### `fit_transform(self, inputs)`
**Module:** `layers.statistics.invertible_kernel_pca`

Fit the model and transform inputs in one step.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:588*

#### `forward(self, z, context)`
**Module:** `layers.statistics.normalizing_flow`

Forward transformation z → y for sampling.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:302*

#### `from_config(cls, config)`
**Module:** `layers.fnet_encoder_block`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/fnet_encoder_block.py:313*

#### `from_config(cls, config)`
**Module:** `layers.haar_wavelet_decomposition`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/haar_wavelet_decomposition.py:486*

#### `from_config(cls, config)`
**Module:** `layers.convnext_v2_block`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/convnext_v2_block.py:487*

#### `from_config(cls, config)`
**Module:** `layers.convnext_v1_block`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/convnext_v1_block.py:458*

#### `from_config(cls, config)`
**Module:** `layers.bitlinear_layer`

Create layer instance from configuration.

*📁 src/dl_techniques/layers/bitlinear_layer.py:538*

#### `from_config(cls, config)`
**Module:** `layers.selective_gradient_mask`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/selective_gradient_mask.py:267*

#### `from_config(cls, config)`
**Module:** `layers.sparse_autoencoder`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:928*

#### `from_config(cls, config)`
**Module:** `layers.rigid_simplex_layer`

Create layer from configuration.

*📁 src/dl_techniques/layers/rigid_simplex_layer.py:404*

#### `from_config(cls, config)`
**Module:** `layers.router`

Creates a layer from its config, properly deserializing sub-layers.

*📁 src/dl_techniques/layers/router.py:364*

#### `from_config(cls, config)`
**Module:** `layers.transformers.progressive_focused_transformer`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/transformers/progressive_focused_transformer.py:846*

#### `from_config(cls, config)`
**Module:** `layers.experimental.field_embeddings`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:261*

#### `from_config(cls, config)`
**Module:** `layers.experimental.contextual_memory`

Create layer from configuration.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:929*

#### `from_config(cls, config)`
**Module:** `layers.nlp_heads.factory`

Create layer from configuration.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:315*

#### `from_config(cls, config)`
**Module:** `layers.nlp_heads.factory`

Create layer from configuration.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1287*

#### `from_config(cls, config)`
**Module:** `layers.activations.probability_output`

Reconstruct the layer from configuration.

*📁 src/dl_techniques/layers/activations/probability_output.py:302*

#### `from_config(cls, config)`
**Module:** `layers.memory.som_2d_layer`

Create layer instance from configuration dictionary.

*📁 src/dl_techniques/layers/memory/som_2d_layer.py:399*

#### `from_config(cls, config)`
**Module:** `layers.memory.som_nd_layer`

Create a layer from its configuration.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:584*

#### `from_config(cls, config)`
**Module:** `layers.norms.band_logit_norm`

Create a layer instance from its configuration.

*📁 src/dl_techniques/layers/norms/band_logit_norm.py:194*

#### `from_config(cls, config)`
**Module:** `layers.moe.layer`

Create layer from configuration.

*📁 src/dl_techniques/layers/moe/layer.py:526*

#### `from_config(cls, config)`
**Module:** `layers.fusion.multimodal_fusion`

Create layer from configuration.

*📁 src/dl_techniques/layers/fusion/multimodal_fusion.py:988*

#### `from_config(cls, config)`
**Module:** `layers.attention.window_attention`

Create a layer from its configuration.

*📁 src/dl_techniques/layers/attention/window_attention.py:523*

#### `from_config(cls, config)`
**Module:** `layers.attention.progressive_focused_attention`

Create layer from configuration dictionary.

*📁 src/dl_techniques/layers/attention/progressive_focused_attention.py:1025*

#### `from_config(cls, config)`
**Module:** `layers.ntm.baseline_ntm`

Create layer from configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1175*

#### `from_config(cls, config)`
**Module:** `layers.ntm.baseline_ntm`

Create layer from configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1417*

#### `from_config(cls, config)`
**Module:** `layers.ntm.ntm_interface`

Create layer from configuration.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:815*

#### `from_config(cls, config)`
**Module:** `layers.embedding.bert_embeddings`

Create layer from configuration.

*📁 src/dl_techniques/layers/embedding/bert_embeddings.py:363*

#### `from_dict(cls, config_dict, validate_compatibility)`
**Module:** `layers.vision_heads.task_types`

Create TaskConfiguration from dictionary.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:660*

#### `from_dict(cls, config_dict)`
**Module:** `layers.moe.config`

Create configuration from dictionary.

*📁 src/dl_techniques/layers/moe/config.py:263*

#### `from_dict(cls, config_dict)`
**Module:** `layers.ntm.ntm_interface`

Create configuration from dictionary.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:290*

#### `from_string(type_str)`
**Module:** `layers.layer_scale`

Convert string to MultiplierType enum.

*📁 src/dl_techniques/layers/layer_scale.py:63*

#### `from_string(type_str)`
**Module:** `layers.conv2d_builder`

*📁 src/dl_techniques/layers/conv2d_builder.py:227*

#### `from_string(cls, task_str)`
**Module:** `layers.nlp_heads.task_types`

Create NLPTaskType from string value.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:331*

#### `from_string(cls, task_str)`
**Module:** `layers.vision_heads.task_types`

Create TaskType from string value.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:386*

#### `from_string(cls, task_str)`
**Module:** `layers.vlm_heads.task_types`

Create VLMTaskType from string value.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:180*

#### `from_strings(cls, task_strs)`
**Module:** `layers.vision_heads.task_types`

Create list of TaskTypes from list of strings.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:415*

#### `from_strings(cls, task_strings, validate_compatibility)`
**Module:** `layers.vision_heads.task_types`

Create TaskConfiguration from list of task name strings.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:689*

#### `gamma(self)`
**Module:** `layers.radial_basis_function`

Effective positive gamma values via softplus transformation.

*📁 src/dl_techniques/layers/radial_basis_function.py:186*

#### `gaussian_pyramid(inputs, levels, kernel_size, sigma, scale_factor, padding, data_format, name)`
**Module:** `layers.gaussian_pyramid`

Functional interface for Gaussian pyramid decomposition.

*📁 src/dl_techniques/layers/gaussian_pyramid.py:347*

#### `get_acf_summary(self)`
**Module:** `layers.statistics.residual_acf`

Get comprehensive summary statistics of the most recent ACF computation.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:407*

#### `get_activation(activation_name)`
**Module:** `layers.activations.expanded_activations`

Factory function to create an activation layer by name.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:588*

#### `get_activation_info()`
**Module:** `layers.activations.factory`

Get comprehensive information about all available activation layer types.

*📁 src/dl_techniques/layers/activations/factory.py:315*

#### `get_addressing_probabilities(self, inputs)`
**Module:** `layers.neuro_grid`

Get the addressing probabilities for analysis and interpretation.

*📁 src/dl_techniques/layers/neuro_grid.py:816*

#### `get_all_configurations(cls)`
**Module:** `layers.vision_heads.task_types`

Get all predefined configurations.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:802*

#### `get_all_weights(self)`
**Module:** `layers.experimental.hierarchical_memory_system`

Get weight maps for all hierarchy levels.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:323*

#### `get_attention_info()`
**Module:** `layers.attention.factory`

Retrieve comprehensive information about all available attention layer types.

*📁 src/dl_techniques/layers/attention/factory.py:724*

#### `get_attention_requirements(attention_type)`
**Module:** `layers.attention.factory`

Get parameter requirements for a specific attention layer type.

*📁 src/dl_techniques/layers/attention/factory.py:1096*

#### `get_build_config(self)`
**Module:** `layers.modality_projection`

Get the config needed to build the layer from a config.

*📁 src/dl_techniques/layers/modality_projection.py:295*

#### `get_build_config(self)`
**Module:** `layers.memory.som_nd_layer`

Get build configuration for the layer.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:558*

#### `get_build_config(self)`
**Module:** `layers.moe.experts`

Get build configuration for serialization.

*📁 src/dl_techniques/layers/moe/experts.py:71*

#### `get_category(self)`
**Module:** `layers.vision_heads.task_types`

Get the category this task belongs to.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:453*

#### `get_cls_features(self, inputs, training)`
**Module:** `layers.transformers.vision_encoder`

Extract CLS token features for classification tasks.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:685*

#### `get_codebook_indices(self, inputs)`
**Module:** `layers.vector_quantizer`

Get discrete codebook indices for given inputs.

*📁 src/dl_techniques/layers/vector_quantizer.py:342*

#### `get_compatible_tasks(cls, task)`
**Module:** `layers.nlp_heads.task_types`

Get tasks that are commonly combined with the given task.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:195*

#### `get_compatible_tasks(cls, task)`
**Module:** `layers.vision_heads.task_types`

Get tasks that are commonly combined with the given task.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:232*

#### `get_confidences(self)`
**Module:** `layers.experimental.band_rms_ood`

Get current confidence estimates.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:457*

#### `get_config(self)`
**Module:** `layers.mobile_one_block`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/mobile_one_block.py:337*

#### `get_config(self)`
**Module:** `layers.depthwise_separable_block`

Return configuration for serialization.

*📁 src/dl_techniques/layers/depthwise_separable_block.py:395*

#### `get_config(self)`
**Module:** `layers.sampling`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/sampling.py:258*

#### `get_config(self)`
**Module:** `layers.hanc_layer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/hanc_layer.py:325*

#### `get_config(self)`
**Module:** `layers.laplacian_filter`

Return the config dictionary for the layer.

*📁 src/dl_techniques/layers/laplacian_filter.py:215*

#### `get_config(self)`
**Module:** `layers.laplacian_filter`

Return the config dictionary for the layer.

*📁 src/dl_techniques/layers/laplacian_filter.py:475*

#### `get_config(self)`
**Module:** `layers.multi_level_feature_compilation`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/multi_level_feature_compilation.py:421*

#### `get_config(self)`
**Module:** `layers.yolo12_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_blocks.py:218*

#### `get_config(self)`
**Module:** `layers.yolo12_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_blocks.py:487*

#### `get_config(self)`
**Module:** `layers.yolo12_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_blocks.py:663*

#### `get_config(self)`
**Module:** `layers.yolo12_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_blocks.py:810*

#### `get_config(self)`
**Module:** `layers.yolo12_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_blocks.py:993*

#### `get_config(self)`
**Module:** `layers.yolo12_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_blocks.py:1195*

#### `get_config(self)`
**Module:** `layers.yolo12_heads`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_heads.py:359*

#### `get_config(self)`
**Module:** `layers.yolo12_heads`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_heads.py:804*

#### `get_config(self)`
**Module:** `layers.yolo12_heads`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/yolo12_heads.py:1122*

#### `get_config(self)`
**Module:** `layers.radial_basis_function`

Returns the config of the layer.

*📁 src/dl_techniques/layers/radial_basis_function.py:304*

#### `get_config(self)`
**Module:** `layers.spatial_layer`

Returns the configuration dictionary for serialization.

*📁 src/dl_techniques/layers/spatial_layer.py:289*

#### `get_config(self)`
**Module:** `layers.one_hot_encoding`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/one_hot_encoding.py:124*

#### `get_config(self)`
**Module:** `layers.neuro_grid`

Return configuration for serialization.

*📁 src/dl_techniques/layers/neuro_grid.py:781*

#### `get_config(self)`
**Module:** `layers.gaussian_filter`

Return the configuration for serialization.

*📁 src/dl_techniques/layers/gaussian_filter.py:214*

#### `get_config(self)`
**Module:** `layers.dynamic_conv2d`

Return configuration for serialization.

*📁 src/dl_techniques/layers/dynamic_conv2d.py:500*

#### `get_config(self)`
**Module:** `layers.shearlet_transform`

Serialization configuration.

*📁 src/dl_techniques/layers/shearlet_transform.py:433*

#### `get_config(self)`
**Module:** `layers.squeeze_excitation`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/squeeze_excitation.py:382*

#### `get_config(self)`
**Module:** `layers.standard_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:377*

#### `get_config(self)`
**Module:** `layers.standard_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:612*

#### `get_config(self)`
**Module:** `layers.standard_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:821*

#### `get_config(self)`
**Module:** `layers.standard_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:1068*

#### `get_config(self)`
**Module:** `layers.standard_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/standard_blocks.py:1353*

#### `get_config(self)`
**Module:** `layers.sequence_pooling`

Return configuration for serialization.

*📁 src/dl_techniques/layers/sequence_pooling.py:266*

#### `get_config(self)`
**Module:** `layers.sequence_pooling`

Return configuration for serialization.

*📁 src/dl_techniques/layers/sequence_pooling.py:402*

#### `get_config(self)`
**Module:** `layers.sequence_pooling`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/sequence_pooling.py:850*

#### `get_config(self)`
**Module:** `layers.blt_core`

Get layer configuration.

*📁 src/dl_techniques/layers/blt_core.py:575*

#### `get_config(self)`
**Module:** `layers.fnet_encoder_block`

Return complete configuration for serialization.

*📁 src/dl_techniques/layers/fnet_encoder_block.py:298*

#### `get_config(self)`
**Module:** `layers.conditional_output_layer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/conditional_output_layer.py:227*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:160*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:397*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:589*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:702*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:876*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:1008*

#### `get_config(self)`
**Module:** `layers.complex_layers`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/complex_layers.py:1137*

#### `get_config(self)`
**Module:** `layers.anchor_generator`

Return configuration for serialization.

*📁 src/dl_techniques/layers/anchor_generator.py:268*

#### `get_config(self)`
**Module:** `layers.haar_wavelet_decomposition`

Return configuration for serialization.

*📁 src/dl_techniques/layers/haar_wavelet_decomposition.py:472*

#### `get_config(self)`
**Module:** `layers.tabm_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/tabm_blocks.py:125*

#### `get_config(self)`
**Module:** `layers.tabm_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/tabm_blocks.py:258*

#### `get_config(self)`
**Module:** `layers.tabm_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/tabm_blocks.py:358*

#### `get_config(self)`
**Module:** `layers.tabm_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/tabm_blocks.py:478*

#### `get_config(self)`
**Module:** `layers.tabm_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/tabm_blocks.py:584*

#### `get_config(self)`
**Module:** `layers.inverted_residual_block`

Return configuration for serialization.

*📁 src/dl_techniques/layers/inverted_residual_block.py:134*

#### `get_config(self)`
**Module:** `layers.kan_linear`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/kan_linear.py:459*

#### `get_config(self)`
**Module:** `layers.convnext_v2_block`

Return configuration for serialization.

*📁 src/dl_techniques/layers/convnext_v2_block.py:463*

#### `get_config(self)`
**Module:** `layers.io_preparation`

Return configuration for serialization.

*📁 src/dl_techniques/layers/io_preparation.py:106*

#### `get_config(self)`
**Module:** `layers.io_preparation`

Return configuration for serialization.

*📁 src/dl_techniques/layers/io_preparation.py:241*

#### `get_config(self)`
**Module:** `layers.io_preparation`

Return configuration for serialization.

*📁 src/dl_techniques/layers/io_preparation.py:375*

#### `get_config(self)`
**Module:** `layers.io_preparation`

Return configuration for serialization.

*📁 src/dl_techniques/layers/io_preparation.py:536*

#### `get_config(self)`
**Module:** `layers.convnext_v1_block`

Return configuration for serialization.

*📁 src/dl_techniques/layers/convnext_v1_block.py:434*

#### `get_config(self)`
**Module:** `layers.patch_merging`

Return configuration for serialization.

*📁 src/dl_techniques/layers/patch_merging.py:230*

#### `get_config(self)`
**Module:** `layers.fft_layers`

Return configuration for serialization.

*📁 src/dl_techniques/layers/fft_layers.py:113*

#### `get_config(self)`
**Module:** `layers.fft_layers`

Return configuration for serialization.

*📁 src/dl_techniques/layers/fft_layers.py:228*

#### `get_config(self)`
**Module:** `layers.orthoblock`

Return the layer's configuration for serialization.

*📁 src/dl_techniques/layers/orthoblock.py:356*

#### `get_config(self)`
**Module:** `layers.pixel_shuffle`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/pixel_shuffle.py:265*

#### `get_config(self)`
**Module:** `layers.res_path`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/res_path.py:235*

#### `get_config(self)`
**Module:** `layers.mps_layer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/mps_layer.py:399*

#### `get_config(self)`
**Module:** `layers.film`

Get the configuration dictionary for layer serialization.

*📁 src/dl_techniques/layers/film.py:429*

#### `get_config(self)`
**Module:** `layers.hanc_block`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/hanc_block.py:374*

#### `get_config(self)`
**Module:** `layers.repmixer_block`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/repmixer_block.py:388*

#### `get_config(self)`
**Module:** `layers.repmixer_block`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/repmixer_block.py:594*

#### `get_config(self)`
**Module:** `layers.bitlinear_layer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/bitlinear_layer.py:510*

#### `get_config(self)`
**Module:** `layers.stochastic_gradient`

Return the config dictionary for layer serialization.

*📁 src/dl_techniques/layers/stochastic_gradient.py:178*

#### `get_config(self)`
**Module:** `layers.gaussian_pyramid`

Return the configuration for serialization.

*📁 src/dl_techniques/layers/gaussian_pyramid.py:324*

#### `get_config(self)`
**Module:** `layers.bias_free_conv1d`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:243*

#### `get_config(self)`
**Module:** `layers.bias_free_conv1d`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/bias_free_conv1d.py:506*

#### `get_config(self)`
**Module:** `layers.vector_quantizer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/vector_quantizer.py:466*

#### `get_config(self)`
**Module:** `layers.restricted_boltzmann_machine`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:621*

#### `get_config(self)`
**Module:** `layers.convolutional_kan`

Return configuration for serialization.

*📁 src/dl_techniques/layers/convolutional_kan.py:466*

#### `get_config(self)`
**Module:** `layers.kmeans`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/kmeans.py:642*

#### `get_config(self)`
**Module:** `layers.capsules`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/capsules.py:299*

#### `get_config(self)`
**Module:** `layers.capsules`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/capsules.py:637*

#### `get_config(self)`
**Module:** `layers.capsules`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/capsules.py:900*

#### `get_config(self)`
**Module:** `layers.gated_delta_net`

Return configuration for serialization.

*📁 src/dl_techniques/layers/gated_delta_net.py:503*

#### `get_config(self)`
**Module:** `layers.bias_free_conv2d`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:263*

#### `get_config(self)`
**Module:** `layers.bias_free_conv2d`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/bias_free_conv2d.py:546*

#### `get_config(self)`
**Module:** `layers.mothnet_blocks`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/mothnet_blocks.py:260*

#### `get_config(self)`
**Module:** `layers.mothnet_blocks`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/mothnet_blocks.py:547*

#### `get_config(self)`
**Module:** `layers.mothnet_blocks`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/mothnet_blocks.py:826*

#### `get_config(self)`
**Module:** `layers.selective_gradient_mask`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/selective_gradient_mask.py:255*

#### `get_config(self)`
**Module:** `layers.strong_augmentation`

Get layer configuration.

*📁 src/dl_techniques/layers/strong_augmentation.py:216*

#### `get_config(self)`
**Module:** `layers.clahe`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/clahe.py:270*

#### `get_config(self)`
**Module:** `layers.tversky_projection`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/tversky_projection.py:297*

#### `get_config(self)`
**Module:** `layers.stochastic_depth`

Return the config dictionary for layer serialization.

*📁 src/dl_techniques/layers/stochastic_depth.py:208*

#### `get_config(self)`
**Module:** `layers.hierarchical_mlp_stem`

Return configuration for serialization.

*📁 src/dl_techniques/layers/hierarchical_mlp_stem.py:262*

#### `get_config(self)`
**Module:** `layers.layer_scale`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/layer_scale.py:293*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:335*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:523*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:699*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:964*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:1143*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:1280*

#### `get_config(self)`
**Module:** `layers.blt_blocks`

Return layer configuration.

*📁 src/dl_techniques/layers/blt_blocks.py:1567*

#### `get_config(self)`
**Module:** `layers.random_fourier_features`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/random_fourier_features.py:375*

#### `get_config(self)`
**Module:** `layers.modality_projection`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/modality_projection.py:275*

#### `get_config(self)`
**Module:** `layers.sparse_autoencoder`

Return the configuration of the layer.

*📁 src/dl_techniques/layers/sparse_autoencoder.py:899*

#### `get_config(self)`
**Module:** `layers.eomt_mask`

Return configuration for serialization.

*📁 src/dl_techniques/layers/eomt_mask.py:518*

#### `get_config(self)`
**Module:** `layers.canny`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/canny.py:318*

#### `get_config(self)`
**Module:** `layers.universal_inverted_bottleneck`

Return the layer's configuration for serialization.

*📁 src/dl_techniques/layers/universal_inverted_bottleneck.py:570*

#### `get_config(self)`
**Module:** `layers.global_sum_pool_2d`

Get the layer configuration for serialization.

*📁 src/dl_techniques/layers/global_sum_pool_2d.py:209*

#### `get_config(self)`
**Module:** `layers.fractal_block`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/fractal_block.py:394*

#### `get_config(self)`
**Module:** `layers.rigid_simplex_layer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/rigid_simplex_layer.py:386*

#### `get_config(self)`
**Module:** `layers.router`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/router.py:349*

#### `get_config(self)`
**Module:** `layers.transformers.progressive_focused_transformer`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/transformers/progressive_focused_transformer.py:810*

#### `get_config(self)`
**Module:** `layers.transformers.text_encoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:927*

#### `get_config(self)`
**Module:** `layers.transformers.text_decoder`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/transformers/text_decoder.py:491*

#### `get_config(self)`
**Module:** `layers.transformers.eomt_transformer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/transformers/eomt_transformer.py:483*

#### `get_config(self)`
**Module:** `layers.transformers.swin_transformer_block`

Return configuration for serialization.

*📁 src/dl_techniques/layers/transformers/swin_transformer_block.py:606*

#### `get_config(self)`
**Module:** `layers.transformers.free_transformer`

Serialize layer configuration.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:258*

#### `get_config(self)`
**Module:** `layers.transformers.free_transformer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/transformers/free_transformer.py:960*

#### `get_config(self)`
**Module:** `layers.transformers.perceiver_transformer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/transformers/perceiver_transformer.py:360*

#### `get_config(self)`
**Module:** `layers.transformers.swin_conv_block`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/transformers/swin_conv_block.py:701*

#### `get_config(self)`
**Module:** `layers.transformers.vision_encoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:772*

#### `get_config(self)`
**Module:** `layers.transformers.transformer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/transformers/transformer.py:650*

#### `get_config(self)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:244*

#### `get_config(self)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:423*

#### `get_config(self)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:605*

#### `get_config(self)`
**Module:** `layers.experimental.hierarchical_evidence_llm`

*📁 src/dl_techniques/layers/experimental/hierarchical_evidence_llm.py:814*

#### `get_config(self)`
**Module:** `layers.experimental.hierarchical_memory_system`

Get configuration for the hierarchical memory system.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:372*

#### `get_config(self)`
**Module:** `layers.experimental.field_embeddings`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:236*

#### `get_config(self)`
**Module:** `layers.experimental.field_embeddings`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:441*

#### `get_config(self)`
**Module:** `layers.experimental.field_embeddings`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:622*

#### `get_config(self)`
**Module:** `layers.experimental.field_embeddings`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:827*

#### `get_config(self)`
**Module:** `layers.experimental.contextual_memory`

Get layer configuration.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:242*

#### `get_config(self)`
**Module:** `layers.experimental.contextual_memory`

Get layer configuration.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:520*

#### `get_config(self)`
**Module:** `layers.experimental.contextual_memory`

Get layer configuration.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:676*

#### `get_config(self)`
**Module:** `layers.experimental.contextual_memory`

Get layer configuration.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:903*

#### `get_config(self)`
**Module:** `layers.experimental.mst_correlation_filter`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:169*

#### `get_config(self)`
**Module:** `layers.experimental.mst_correlation_filter`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/experimental/mst_correlation_filter.py:550*

#### `get_config(self)`
**Module:** `layers.experimental.band_rms_ood`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:592*

#### `get_config(self)`
**Module:** `layers.experimental.contextual_counter_ffn`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/experimental/contextual_counter_ffn.py:383*

#### `get_config(self)`
**Module:** `layers.experimental.graph_mann`

Return configuration for serialization.

*📁 src/dl_techniques/layers/experimental/graph_mann.py:422*

#### `get_config(self)`
**Module:** `layers.nlp_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:282*

#### `get_config(self)`
**Module:** `layers.nlp_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:880*

#### `get_config(self)`
**Module:** `layers.nlp_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1266*

#### `get_config(self)`
**Module:** `layers.time_series.ema_layer`

Return layer configuration.

*📁 src/dl_techniques/layers/time_series/ema_layer.py:157*

#### `get_config(self)`
**Module:** `layers.time_series.ema_layer`

Return layer configuration.

*📁 src/dl_techniques/layers/time_series/ema_layer.py:422*

#### `get_config(self)`
**Module:** `layers.time_series.nbeats_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:451*

#### `get_config(self)`
**Module:** `layers.time_series.nbeats_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:595*

#### `get_config(self)`
**Module:** `layers.time_series.nbeats_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:794*

#### `get_config(self)`
**Module:** `layers.time_series.nbeats_blocks`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/time_series/nbeats_blocks.py:1019*

#### `get_config(self)`
**Module:** `layers.time_series.adaptive_lag_attention`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/time_series/adaptive_lag_attention.py:366*

#### `get_config(self)`
**Module:** `layers.time_series.quantile_head_fixed_io`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/quantile_head_fixed_io.py:259*

#### `get_config(self)`
**Module:** `layers.time_series.deepar_blocks`

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:118*

#### `get_config(self)`
**Module:** `layers.time_series.deepar_blocks`

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:233*

#### `get_config(self)`
**Module:** `layers.time_series.deepar_blocks`

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:353*

#### `get_config(self)`
**Module:** `layers.time_series.deepar_blocks`

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:470*

#### `get_config(self)`
**Module:** `layers.time_series.quantile_head_variable_io`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/time_series/quantile_head_variable_io.py:381*

#### `get_config(self)`
**Module:** `layers.time_series.xlstm_blocks`

Return the configuration of the cell.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:276*

#### `get_config(self)`
**Module:** `layers.time_series.xlstm_blocks`

Return the configuration of the layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:441*

#### `get_config(self)`
**Module:** `layers.time_series.xlstm_blocks`

Return the configuration of the cell.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:738*

#### `get_config(self)`
**Module:** `layers.time_series.xlstm_blocks`

Return the configuration of the layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:873*

#### `get_config(self)`
**Module:** `layers.time_series.xlstm_blocks`

Return the configuration of the layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1071*

#### `get_config(self)`
**Module:** `layers.time_series.xlstm_blocks`

Return the configuration of the layer.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:1303*

#### `get_config(self)`
**Module:** `layers.time_series.nbeatsx_blocks`

*📁 src/dl_techniques/layers/time_series/nbeatsx_blocks.py:189*

#### `get_config(self)`
**Module:** `layers.time_series.forecasting_layers`

Serialization configuration.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:113*

#### `get_config(self)`
**Module:** `layers.time_series.forecasting_layers`

Serialization configuration.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:262*

#### `get_config(self)`
**Module:** `layers.time_series.forecasting_layers`

Serialization configuration.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:501*

#### `get_config(self)`
**Module:** `layers.time_series.temporal_fusion`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/time_series/temporal_fusion.py:403*

#### `get_config(self)`
**Module:** `layers.time_series.mixed_sequential_block`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/mixed_sequential_block.py:577*

#### `get_config(self)`
**Module:** `layers.time_series.prism_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:126*

#### `get_config(self)`
**Module:** `layers.time_series.prism_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:300*

#### `get_config(self)`
**Module:** `layers.time_series.prism_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:523*

#### `get_config(self)`
**Module:** `layers.time_series.prism_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:874*

#### `get_config(self)`
**Module:** `layers.time_series.prism_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:1060*

#### `get_config(self)`
**Module:** `layers.time_series.temporal_convolutional_network`

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:73*

#### `get_config(self)`
**Module:** `layers.time_series.temporal_convolutional_network`

*📁 src/dl_techniques/layers/time_series/temporal_convolutional_network.py:129*

#### `get_config(self)`
**Module:** `layers.physics.approximate_lagrange_layer`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/physics/approximate_lagrange_layer.py:309*

#### `get_config(self)`
**Module:** `layers.physics.lagrange_layer`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/physics/lagrange_layer.py:268*

#### `get_config(self)`
**Module:** `layers.activations.relu_k`

Get the layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/relu_k.py:225*

#### `get_config(self)`
**Module:** `layers.activations.hard_sigmoid`

Return configuration for serialization.

*📁 src/dl_techniques/layers/activations/hard_sigmoid.py:186*

#### `get_config(self)`
**Module:** `layers.activations.thresh_max`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/thresh_max.py:242*

#### `get_config(self)`
**Module:** `layers.activations.basis_function`

Get the layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/basis_function.py:215*

#### `get_config(self)`
**Module:** `layers.activations.differentiable_step`

Return the layer's configuration for serialization.

*📁 src/dl_techniques/layers/activations/differentiable_step.py:248*

#### `get_config(self)`
**Module:** `layers.activations.expanded_activations`

Returns the configuration of the layer.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:167*

#### `get_config(self)`
**Module:** `layers.activations.expanded_activations`

Returns the configuration of the expanded activation layer.

*📁 src/dl_techniques/layers/activations/expanded_activations.py:338*

#### `get_config(self)`
**Module:** `layers.activations.probability_output`

Return configuration for serialization.

*📁 src/dl_techniques/layers/activations/probability_output.py:287*

#### `get_config(self)`
**Module:** `layers.activations.routing_probabilities`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/activations/routing_probabilities.py:722*

#### `get_config(self)`
**Module:** `layers.activations.adaptive_softmax`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/adaptive_softmax.py:331*

#### `get_config(self)`
**Module:** `layers.activations.monotonicity_layer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/activations/monotonicity_layer.py:582*

#### `get_config(self)`
**Module:** `layers.activations.mish`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/mish.py:236*

#### `get_config(self)`
**Module:** `layers.activations.mish`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/mish.py:409*

#### `get_config(self)`
**Module:** `layers.activations.sparsemax`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/sparsemax.py:253*

#### `get_config(self)`
**Module:** `layers.activations.squash`

Get the layer configuration for serialization.

*📁 src/dl_techniques/layers/activations/squash.py:220*

#### `get_config(self)`
**Module:** `layers.activations.routing_probabilities_hierarchical`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/activations/routing_probabilities_hierarchical.py:429*

#### `get_config(self)`
**Module:** `layers.activations.hard_swish`

Return configuration for serialization.

*📁 src/dl_techniques/layers/activations/hard_swish.py:220*

#### `get_config(self)`
**Module:** `layers.activations.golu`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/activations/golu.py:176*

#### `get_config(self)`
**Module:** `layers.memory.mann`

Return configuration for serialization.

*📁 src/dl_techniques/layers/memory/mann.py:463*

#### `get_config(self)`
**Module:** `layers.memory.som_2d_layer`

Return layer configuration for serialization and model saving.

*📁 src/dl_techniques/layers/memory/som_2d_layer.py:353*

#### `get_config(self)`
**Module:** `layers.memory.som_nd_soft_layer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:867*

#### `get_config(self)`
**Module:** `layers.memory.som_nd_layer`

Get configuration for the layer.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:537*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:137*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:250*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:373*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:491*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:598*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:704*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

Get layer configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:794*

#### `get_config(self)`
**Module:** `layers.vision_heads.factory`

*📁 src/dl_techniques/layers/vision_heads/factory.py:947*

#### `get_config(self)`
**Module:** `layers.vlm_heads.factory`

Gets layer configuration.

*📁 src/dl_techniques/layers/vlm_heads/factory.py:115*

#### `get_config(self)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:258*

#### `get_config(self)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:382*

#### `get_config(self)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:554*

#### `get_config(self)`
**Module:** `layers.vlm_heads.factory`

*📁 src/dl_techniques/layers/vlm_heads/factory.py:625*

#### `get_config(self)`
**Module:** `layers.geometric.point_cloud_autoencoder`

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:288*

#### `get_config(self)`
**Module:** `layers.geometric.point_cloud_autoencoder`

*📁 src/dl_techniques/layers/geometric/point_cloud_autoencoder.py:459*

#### `get_config(self)`
**Module:** `layers.geometric.supernode_pooling`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/supernode_pooling.py:433*

#### `get_config(self)`
**Module:** `layers.geometric.fields.parallel_transport`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/parallel_transport.py:343*

#### `get_config(self)`
**Module:** `layers.geometric.fields.holonomy_layer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/holonomy_layer.py:424*

#### `get_config(self)`
**Module:** `layers.geometric.fields.gauge_invariant_attention`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/gauge_invariant_attention.py:545*

#### `get_config(self)`
**Module:** `layers.geometric.fields.connection_layer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/connection_layer.py:424*

#### `get_config(self)`
**Module:** `layers.geometric.fields.manifold_stress`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/manifold_stress.py:473*

#### `get_config(self)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:154*

#### `get_config(self)`
**Module:** `layers.geometric.fields.holonomic_transformer`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/holonomic_transformer.py:504*

#### `get_config(self)`
**Module:** `layers.geometric.fields.field_embedding`

Return configuration for serialization.

*📁 src/dl_techniques/layers/geometric/fields/field_embedding.py:333*

#### `get_config(self)`
**Module:** `layers.norms.zero_centered_rms_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/zero_centered_rms_norm.py:403*

#### `get_config(self)`
**Module:** `layers.norms.adaptive_band_rms`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/adaptive_band_rms.py:430*

#### `get_config(self)`
**Module:** `layers.norms.logit_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/logit_norm.py:260*

#### `get_config(self)`
**Module:** `layers.norms.band_rms`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/band_rms.py:305*

#### `get_config(self)`
**Module:** `layers.norms.global_response_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/global_response_norm.py:276*

#### `get_config(self)`
**Module:** `layers.norms.band_logit_norm`

Return the layer configuration for serialization.

*📁 src/dl_techniques/layers/norms/band_logit_norm.py:178*

#### `get_config(self)`
**Module:** `layers.norms.max_logit_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:205*

#### `get_config(self)`
**Module:** `layers.norms.max_logit_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:427*

#### `get_config(self)`
**Module:** `layers.norms.max_logit_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/max_logit_norm.py:649*

#### `get_config(self)`
**Module:** `layers.norms.rms_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/rms_norm.py:360*

#### `get_config(self)`
**Module:** `layers.norms.zero_centered_band_rms_norm`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py:436*

#### `get_config(self)`
**Module:** `layers.norms.dynamic_tanh`

Return configuration for serialization.

*📁 src/dl_techniques/layers/norms/dynamic_tanh.py:239*

#### `get_config(self)`
**Module:** `layers.logic.arithmetic_operators`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/logic/arithmetic_operators.py:439*

#### `get_config(self)`
**Module:** `layers.logic.logic_operators`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/logic/logic_operators.py:428*

#### `get_config(self)`
**Module:** `layers.logic.neural_circuit`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:298*

#### `get_config(self)`
**Module:** `layers.logic.neural_circuit`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/logic/neural_circuit.py:501*

#### `get_config(self)`
**Module:** `layers.moe.layer`

Get configuration for serialization.

*📁 src/dl_techniques/layers/moe/layer.py:517*

#### `get_config(self)`
**Module:** `layers.moe.experts`

Get configuration for serialization.

*📁 src/dl_techniques/layers/moe/experts.py:201*

#### `get_config(self)`
**Module:** `layers.moe.gating`

Get configuration for serialization.

*📁 src/dl_techniques/layers/moe/gating.py:67*

#### `get_config(self)`
**Module:** `layers.moe.gating`

Get configuration for serialization.

*📁 src/dl_techniques/layers/moe/gating.py:249*

#### `get_config(self)`
**Module:** `layers.moe.gating`

Get configuration for serialization.

*📁 src/dl_techniques/layers/moe/gating.py:436*

#### `get_config(self)`
**Module:** `layers.moe.gating`

Get configuration for serialization.

*📁 src/dl_techniques/layers/moe/gating.py:580*

#### `get_config(self)`
**Module:** `layers.fusion.multimodal_fusion`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/fusion/multimodal_fusion.py:960*

#### `get_config(self)`
**Module:** `layers.statistics.moving_std`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/statistics/moving_std.py:318*

#### `get_config(self)`
**Module:** `layers.statistics.residual_acf`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:474*

#### `get_config(self)`
**Module:** `layers.statistics.normalizing_flow`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:391*

#### `get_config(self)`
**Module:** `layers.statistics.normalizing_flow`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:789*

#### `get_config(self)`
**Module:** `layers.statistics.invertible_kernel_pca`

Return configuration for serialization.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:636*

#### `get_config(self)`
**Module:** `layers.statistics.invertible_kernel_pca`

Return configuration for serialization.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:815*

#### `get_config(self)`
**Module:** `layers.statistics.scaler`

Return the layer configuration for serialization.

*📁 src/dl_techniques/layers/statistics/scaler.py:624*

#### `get_config(self)`
**Module:** `layers.statistics.deep_kernel_pca`

Return configuration for serialization.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:669*

#### `get_config(self)`
**Module:** `layers.statistics.mdn_layer`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:414*

#### `get_config(self)`
**Module:** `layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py:317*

#### `get_config(self)`
**Module:** `layers.graphs.graph_neural_network`

Return configuration for serialization.

*📁 src/dl_techniques/layers/graphs/graph_neural_network.py:551*

#### `get_config(self)`
**Module:** `layers.graphs.entity_graph_refinement`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:796*

#### `get_config(self)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:244*

#### `get_config(self)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:591*

#### `get_config(self)`
**Module:** `layers.graphs.relational_graph_transformer_blocks`

Return configuration for serialization.

*📁 src/dl_techniques/layers/graphs/relational_graph_transformer_blocks.py:969*

#### `get_config(self)`
**Module:** `layers.graphs.fermi_diract_decoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/graphs/fermi_diract_decoder.py:232*

#### `get_config(self)`
**Module:** `layers.attention.channel_attention`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/channel_attention.py:302*

#### `get_config(self)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:164*

#### `get_config(self)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:267*

#### `get_config(self)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:396*

#### `get_config(self)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:524*

#### `get_config(self)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:606*

#### `get_config(self)`
**Module:** `layers.attention.tripse_attention`

*📁 src/dl_techniques/layers/attention/tripse_attention.py:753*

#### `get_config(self)`
**Module:** `layers.attention.rpc_attention`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/attention/rpc_attention.py:567*

#### `get_config(self)`
**Module:** `layers.attention.multi_head_attention`

Return configuration for serialization - includes ALL constructor parameters.

*📁 src/dl_techniques/layers/attention/multi_head_attention.py:242*

#### `get_config(self)`
**Module:** `layers.attention.shared_weights_cross_attention`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/shared_weights_cross_attention.py:419*

#### `get_config(self)`
**Module:** `layers.attention.single_window_attention`

Serialize the layer's configuration.

*📁 src/dl_techniques/layers/attention/single_window_attention.py:311*

#### `get_config(self)`
**Module:** `layers.attention.ring_attention`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/ring_attention.py:601*

#### `get_config(self)`
**Module:** `layers.attention.perceiver_attention`

Return configuration for serialization - includes ALL constructor parameters.

*📁 src/dl_techniques/layers/attention/perceiver_attention.py:329*

#### `get_config(self)`
**Module:** `layers.attention.gated_attention`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/gated_attention.py:591*

#### `get_config(self)`
**Module:** `layers.attention.spatial_attention`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/spatial_attention.py:240*

#### `get_config(self)`
**Module:** `layers.attention.performer_attention`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/attention/performer_attention.py:533*

#### `get_config(self)`
**Module:** `layers.attention.multi_head_cross_attention`

Return configuration for serialization - includes ALL constructor parameters.

*📁 src/dl_techniques/layers/attention/multi_head_cross_attention.py:594*

#### `get_config(self)`
**Module:** `layers.attention.fnet_fourier_transform`

Return configuration for serialization.

*📁 src/dl_techniques/layers/attention/fnet_fourier_transform.py:275*

#### `get_config(self)`
**Module:** `layers.attention.window_attention`

Serialize the layer's configuration.

*📁 src/dl_techniques/layers/attention/window_attention.py:486*

#### `get_config(self)`
**Module:** `layers.attention.capsule_routing_attention`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/layers/attention/capsule_routing_attention.py:718*

#### `get_config(self)`
**Module:** `layers.attention.anchor_attention`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/attention/anchor_attention.py:622*

#### `get_config(self)`
**Module:** `layers.attention.differential_attention`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/differential_attention.py:415*

#### `get_config(self)`
**Module:** `layers.attention.hopfield_attention`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/attention/hopfield_attention.py:546*

#### `get_config(self)`
**Module:** `layers.attention.multi_head_latent_attention`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/attention/multi_head_latent_attention.py:789*

#### `get_config(self)`
**Module:** `layers.attention.group_query_attention`

*📁 src/dl_techniques/layers/attention/group_query_attention.py:389*

#### `get_config(self)`
**Module:** `layers.attention.mobile_mqa`

Return config with MobileMQA specific parameters.

*📁 src/dl_techniques/layers/attention/mobile_mqa.py:229*

#### `get_config(self)`
**Module:** `layers.attention.convolutional_block_attention`

Return configuration for serialization.

*📁 src/dl_techniques/layers/attention/convolutional_block_attention.py:286*

#### `get_config(self)`
**Module:** `layers.attention.non_local_attention`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/non_local_attention.py:501*

#### `get_config(self)`
**Module:** `layers.attention.progressive_focused_attention`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/attention/progressive_focused_attention.py:991*

#### `get_config(self)`
**Module:** `layers.tokenizers.bpe`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:366*

#### `get_config(self)`
**Module:** `layers.tokenizers.bpe`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:452*

#### `get_config(self)`
**Module:** `layers.ntm.baseline_ntm`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:152*

#### `get_config(self)`
**Module:** `layers.ntm.baseline_ntm`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:373*

#### `get_config(self)`
**Module:** `layers.ntm.baseline_ntm`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:611*

#### `get_config(self)`
**Module:** `layers.ntm.baseline_ntm`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:796*

#### `get_config(self)`
**Module:** `layers.ntm.baseline_ntm`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1154*

#### `get_config(self)`
**Module:** `layers.ntm.baseline_ntm`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1395*

#### `get_config(self)`
**Module:** `layers.ntm.ntm_interface`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:403*

#### `get_config(self)`
**Module:** `layers.ntm.ntm_interface`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:514*

#### `get_config(self)`
**Module:** `layers.ntm.ntm_interface`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:600*

#### `get_config(self)`
**Module:** `layers.ntm.ntm_interface`

Get layer configuration.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:798*

#### `get_config(self)`
**Module:** `layers.ntm.base_layers`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/ntm/base_layers.py:300*

#### `get_config(self)`
**Module:** `layers.ntm.base_layers`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/ntm/base_layers.py:636*

#### `get_config(self)`
**Module:** `layers.ntm.base_layers`

Return layer configuration for serialization.

*📁 src/dl_techniques/layers/ntm/base_layers.py:932*

#### `get_config(self)`
**Module:** `layers.ffn.residual_block`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/residual_block.py:320*

#### `get_config(self)`
**Module:** `layers.ffn.counting_ffn`

Returns the layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/counting_ffn.py:414*

#### `get_config(self)`
**Module:** `layers.ffn.power_mlp_layer`

Get the layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/power_mlp_layer.py:342*

#### `get_config(self)`
**Module:** `layers.ffn.diff_ffn`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/diff_ffn.py:430*

#### `get_config(self)`
**Module:** `layers.ffn.gated_mlp`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/gated_mlp.py:375*

#### `get_config(self)`
**Module:** `layers.ffn.swiglu_ffn`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/swiglu_ffn.py:396*

#### `get_config(self)`
**Module:** `layers.ffn.swin_mlp`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/swin_mlp.py:347*

#### `get_config(self)`
**Module:** `layers.ffn.mlp`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/mlp.py:324*

#### `get_config(self)`
**Module:** `layers.ffn.glu_ffn`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/glu_ffn.py:379*

#### `get_config(self)`
**Module:** `layers.ffn.orthoglu_ffn`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/layers/ffn/orthoglu_ffn.py:267*

#### `get_config(self)`
**Module:** `layers.ffn.geglu_ffn`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/layers/ffn/geglu_ffn.py:345*

#### `get_config(self)`
**Module:** `layers.ffn.logic_ffn`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/ffn/logic_ffn.py:359*

#### `get_config(self)`
**Module:** `layers.embedding.patch_embedding`

Return the configuration of the layer for serialization.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:262*

#### `get_config(self)`
**Module:** `layers.embedding.patch_embedding`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/embedding/patch_embedding.py:448*

#### `get_config(self)`
**Module:** `layers.embedding.continuous_rope_embedding`

Return configuration for serialization.

*📁 src/dl_techniques/layers/embedding/continuous_rope_embedding.py:284*

#### `get_config(self)`
**Module:** `layers.embedding.modern_bert_embeddings`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/layers/embedding/modern_bert_embeddings.py:154*

#### `get_config(self)`
**Module:** `layers.embedding.positional_embedding`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/embedding/positional_embedding.py:259*

#### `get_config(self)`
**Module:** `layers.embedding.bert_embeddings`

Get layer configuration for serialization.

*📁 src/dl_techniques/layers/embedding/bert_embeddings.py:347*

#### `get_config(self)`
**Module:** `layers.embedding.continuous_sin_cos_embedding`

Return configuration for serialization.

*📁 src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py:312*

#### `get_config(self)`
**Module:** `layers.embedding.dual_rotary_position_embedding`

Return configuration for serialization.

*📁 src/dl_techniques/layers/embedding/dual_rotary_position_embedding.py:423*

#### `get_config(self)`
**Module:** `layers.embedding.rotary_position_embedding`

Return configuration for serialization.

*📁 src/dl_techniques/layers/embedding/rotary_position_embedding.py:422*

#### `get_config(self)`
**Module:** `layers.embedding.positional_embedding_sine_2d`

Return configuration for serialization.

*📁 src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py:206*

#### `get_config(self)`
**Module:** `layers.reasoning.hrm_reasoning_module`

Return configuration for serialization.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_module.py:210*

#### `get_config(self)`
**Module:** `layers.reasoning.hrm_sparse_puzzle_embedding`

Return configuration for serialization.

*📁 src/dl_techniques/layers/reasoning/hrm_sparse_puzzle_embedding.py:271*

#### `get_config(self)`
**Module:** `layers.reasoning.hrm_reasoning_core`

Return configuration for serialization.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:681*

#### `get_configurations_by_complexity(cls)`
**Module:** `layers.vision_heads.task_types`

Get configurations organized by complexity level.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:847*

#### `get_current_temperature(self)`
**Module:** `layers.neuro_grid`

Get the current temperature value.

*📁 src/dl_techniques/layers/neuro_grid.py:990*

#### `get_default_config(task_type)`
**Module:** `layers.nlp_heads.factory`

Get default configuration for a task type.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1452*

#### `get_default_config(task_type)`
**Module:** `layers.vision_heads.factory`

Get default configuration for a task type.

*📁 src/dl_techniques/layers/vision_heads/factory.py:1045*

#### `get_efficient_config(task_type)`
**Module:** `layers.vision_heads.factory`

Get efficient (lightweight) configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:1102*

#### `get_embedding_info()`
**Module:** `layers.embedding.factory`

Get comprehensive information about all available embedding layer types.

*📁 src/dl_techniques/layers/embedding/factory.py:143*

#### `get_enabled_tasks(self)`
**Module:** `layers.vision_heads.task_types`

Get list of enabled tasks in a consistent order.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:611*

#### `get_expert_utilization(self)`
**Module:** `layers.moe.layer`

Get statistics about expert utilization and configuration.

*📁 src/dl_techniques/layers/moe/layer.py:496*

#### `get_explained_variance_ratio(self)`
**Module:** `layers.statistics.deep_kernel_pca`

Get the explained variance ratio for each level.

*📁 src/dl_techniques/layers/statistics/deep_kernel_pca.py:653*

#### `get_ffn_info()`
**Module:** `layers.ffn.factory`

Get comprehensive information about all available FFN types.

*📁 src/dl_techniques/layers/ffn/factory.py:292*

#### `get_field_layer_info()`
**Module:** `layers.geometric.fields`

Get information about all available field layer types.

*📁 src/dl_techniques/layers/geometric/fields/__init__.py:284*

#### `get_graph_statistics(entities, graph, entity_mask, threshold)`
**Module:** `layers.graphs.entity_graph_refinement`

Compute comprehensive statistics for an extracted entity graph.

*📁 src/dl_techniques/layers/graphs/entity_graph_refinement.py:827*

#### `get_grid_shapes(self)`
**Module:** `layers.experimental.hierarchical_memory_system`

Get the grid shapes for all hierarchy levels.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:332*

#### `get_grid_utilization(self, inputs)`
**Module:** `layers.neuro_grid`

Compute grid utilization statistics for understanding memory usage patterns.

*📁 src/dl_techniques/layers/neuro_grid.py:881*

#### `get_grid_weights(self)`
**Module:** `layers.neuro_grid`

Get the current grid weights for analysis or visualization.

*📁 src/dl_techniques/layers/neuro_grid.py:802*

#### `get_head_class(task_type)`
**Module:** `layers.nlp_heads.factory`

Get the appropriate head class for a task type.

*📁 src/dl_techniques/layers/nlp_heads/factory.py:1311*

#### `get_head_class(task_type)`
**Module:** `layers.vlm_heads.factory`

Gets the appropriate head class for a VLM task type.

*📁 src/dl_techniques/layers/vlm_heads/factory.py:645*

#### `get_high_performance_config(task_type)`
**Module:** `layers.vision_heads.factory`

Get high-performance configuration.

*📁 src/dl_techniques/layers/vision_heads/factory.py:1117*

#### `get_initial_state(self, batch_size)`
**Module:** `layers.time_series.deepar_blocks`

Get initial state for the cell.

*📁 src/dl_techniques/layers/time_series/deepar_blocks.py:463*

#### `get_initial_state(self, batch_size)`
**Module:** `layers.time_series.xlstm_blocks`

Get initial states for the cell.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:256*

#### `get_initial_state(self, batch_size)`
**Module:** `layers.time_series.xlstm_blocks`

Get initial states for the cell.

*📁 src/dl_techniques/layers/time_series/xlstm_blocks.py:727*

#### `get_initial_state(self, inputs, batch_size, dtype)`
**Module:** `layers.ntm.baseline_ntm`

Initialize all states to zero/initial values.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1087*

#### `get_input_requirements(cls, task)`
**Module:** `layers.nlp_heads.task_types`

Get input requirements for a task.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:293*

#### `get_lambda(self, layer_idx)`
**Module:** `layers.attention.differential_attention`

Compute the lambda value with layer-dependent adaptation.

*📁 src/dl_techniques/layers/attention/differential_attention.py:312*

#### `get_level_weights(self, level)`
**Module:** `layers.experimental.hierarchical_memory_system`

Get the weight map for a specific hierarchy level.

*📁 src/dl_techniques/layers/experimental/hierarchical_memory_system.py:305*

#### `get_max_sequence_length(self)`
**Module:** `layers.nlp_heads.task_types`

Get the maximum sequence length required across all tasks.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:503*

#### `get_memory_state(self)`
**Module:** `layers.experimental.contextual_memory`

Get current memory state for analysis or visualization.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:879*

#### `get_memory_state(self)`
**Module:** `layers.ntm.baseline_ntm`

Get the current memory state (not available in wrapped mode).

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1358*

#### `get_memory_state(self)`
**Module:** `layers.ntm.ntm_interface`

Get the current memory state.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:761*

#### `get_normalization_info()`
**Module:** `layers.norms.factory`

Get information about all supported normalization types and their parameters.

*📁 src/dl_techniques/layers/norms/factory.py:281*

#### `get_ood_detection_stats(self)`
**Module:** `layers.experimental.band_rms_ood`

Get comprehensive statistics for OOD detection analysis.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:555*

#### `get_output_specifications(self)`
**Module:** `layers.vision_heads.task_types`

Get output specifications for all enabled tasks.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:638*

#### `get_output_types(cls, task)`
**Module:** `layers.nlp_heads.task_types`

Get the expected output types for a given task.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:243*

#### `get_output_types(cls, task)`
**Module:** `layers.vision_heads.task_types`

Get the expected output types for a given task.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:300*

#### `get_patch_features(self, inputs, training)`
**Module:** `layers.transformers.vision_encoder`

Extract patch token features for dense prediction tasks.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:709*

#### `get_point_estimate(model, x_data, mdn_layer)`
**Module:** `layers.statistics.mdn_layer`

Calculate point estimates from MDN outputs as the weighted average of means.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:439*

#### `get_pooled_features(self, inputs, pooling_mode, token_type_ids, attention_mask, training)`
**Module:** `layers.transformers.text_encoder`

Get pooled features with specified pooling strategy.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:863*

#### `get_prediction_intervals(point_estimates, total_variance, confidence_level)`
**Module:** `layers.statistics.mdn_layer`

Calculate prediction intervals from MDN outputs.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:529*

#### `get_quality_statistics(self, inputs)`
**Module:** `layers.neuro_grid`

Compute comprehensive batch-level statistical summaries for all input quality measures.

*📁 src/dl_techniques/layers/neuro_grid.py:1249*

#### `get_sequence_features(self, inputs, token_type_ids, attention_mask, training)`
**Module:** `layers.transformers.text_encoder`

Get full sequence features regardless of output_mode.

*📁 src/dl_techniques/layers/transformers/text_encoder.py:828*

#### `get_shell_distance(self)`
**Module:** `layers.experimental.band_rms_ood`

Get current shell distances for OOD detection.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:447*

#### `get_shell_radii(self)`
**Module:** `layers.experimental.band_rms_ood`

Get current shell radii.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:467*

#### `get_soft_assignments(self, inputs)`
**Module:** `layers.memory.som_nd_soft_layer`

Get soft assignment probabilities for given inputs.

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:837*

#### `get_spatial_features(self, inputs, training)`
**Module:** `layers.transformers.vision_encoder`

Get spatial features reshaped for dense prediction tasks.

*📁 src/dl_techniques/layers/transformers/vision_encoder.py:730*

#### `get_stats(self)`
**Module:** `layers.statistics.scaler`

Get the currently stored persistent statistics.

*📁 src/dl_techniques/layers/statistics/scaler.py:579*

#### `get_task_categories(cls)`
**Module:** `layers.nlp_heads.task_types`

Get tasks organized by categories.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:136*

#### `get_task_categories(cls)`
**Module:** `layers.vision_heads.task_types`

Get tasks organized by categories.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:155*

#### `get_task_categories(cls)`
**Module:** `layers.vlm_heads.task_types`

Get tasks organized by categories.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:103*

#### `get_task_names(self)`
**Module:** `layers.vision_heads.task_types`

Get list of enabled task names as strings.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:617*

#### `get_task_suggestions(base_task, max_suggestions)`
**Module:** `layers.vision_heads.task_types`

Get task suggestions that work well with a base task.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:921*

#### `get_tasks_by_category(self)`
**Module:** `layers.vision_heads.task_types`

Get enabled tasks organized by category.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:621*

#### `get_uncertainty(model, x_data, mdn_layer, point_estimates)`
**Module:** `layers.statistics.mdn_layer`

Calculate and decompose predictive uncertainty from an MDN.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:476*

#### `get_weights_as_grid(self)`
**Module:** `layers.memory.som_2d_layer`

Get the current neuron weights organized as a 2D grid for visualization.

*📁 src/dl_techniques/layers/memory/som_2d_layer.py:313*

#### `get_weights_map(self)`
**Module:** `layers.memory.som_nd_soft_layer`

Get the learned prototype weight map.

*📁 src/dl_techniques/layers/memory/som_nd_soft_layer.py:822*

#### `get_weights_map(self)`
**Module:** `layers.memory.som_nd_layer`

Get the current weights organized as an N-dimensional map.

*📁 src/dl_techniques/layers/memory/som_nd_layer.py:500*

#### `gibbs_sampling_step(self, visible)`
**Module:** `layers.restricted_boltzmann_machine`

Perform one step of Gibbs sampling: v -> h -> v'.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:412*

#### `has_classification(self)`
**Module:** `layers.vision_heads.task_types`

Check if classification task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:572*

#### `has_denoising(self)`
**Module:** `layers.vision_heads.task_types`

Check if denoising task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:595*

#### `has_depth_estimation(self)`
**Module:** `layers.vision_heads.task_types`

Check if depth estimation task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:577*

#### `has_detection(self)`
**Module:** `layers.vision_heads.task_types`

Check if detection task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:564*

#### `has_instance_segmentation(self)`
**Module:** `layers.vision_heads.task_types`

Check if instance segmentation task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:586*

#### `has_panoptic_segmentation(self)`
**Module:** `layers.vision_heads.task_types`

Check if panoptic segmentation task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:590*

#### `has_segmentation(self)`
**Module:** `layers.vision_heads.task_types`

Check if segmentation task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:568*

#### `has_super_resolution(self)`
**Module:** `layers.vision_heads.task_types`

Check if super resolution task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:599*

#### `has_surface_normals(self)`
**Module:** `layers.vision_heads.task_types`

Check if surface normals estimation task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:581*

#### `has_task(self, task)`
**Module:** `layers.nlp_heads.task_types`

Check if a specific task is enabled.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:468*

#### `has_task(self, task)`
**Module:** `layers.vision_heads.task_types`

Check if a specific task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:559*

#### `has_task(self, task)`
**Module:** `layers.vlm_heads.task_types`

Check if a specific task is enabled.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:295*

#### `hebbian_update(self, pre_synaptic, post_synaptic)`
**Module:** `layers.mothnet_blocks`

Apply Hebbian weight update rule: ΔW = α · (x^T · y) / batch_size.

*📁 src/dl_techniques/layers/mothnet_blocks.py:771*

#### `initialize_state(self, batch_size)`
**Module:** `layers.ntm.baseline_ntm`

Initialize memory state for a new sequence.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:77*

#### `initialize_state(self, batch_size)`
**Module:** `layers.ntm.baseline_ntm`

Initialize controller state for a new sequence.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:719*

#### `initialize_state(self, batch_size)`
**Module:** `layers.ntm.baseline_ntm`

Initialize all states (placeholder implementation).

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1280*

#### `initialize_state(self, batch_size)`
**Module:** `layers.ntm.ntm_interface`

Initialize memory state for a new sequence.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:350*

#### `initialize_state(self, batch_size)`
**Module:** `layers.ntm.ntm_interface`

Initialize controller state for a new sequence.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:568*

#### `initialize_state(self, batch_size)`
**Module:** `layers.ntm.ntm_interface`

Initialize all states for a new sequence.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:657*

#### `inverse(self, y, context)`
**Module:** `layers.statistics.normalizing_flow`

Inverse transformation y → z with log-determinant for likelihood computation.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:336*

#### `inverse_transform(self, components)`
**Module:** `layers.statistics.invertible_kernel_pca`

Reconstruct original data from principal components.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:517*

#### `inverse_transform(self, scaled_inputs)`
**Module:** `layers.statistics.scaler`

Transform normalized data back to original scale.

*📁 src/dl_techniques/layers/statistics/scaler.py:470*

#### `is_compatible_with(self, other)`
**Module:** `layers.vision_heads.task_types`

Check if this task is compatible with another task.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:470*

#### `is_multi_task(self)`
**Module:** `layers.vision_heads.task_types`

Check if multiple tasks are enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:607*

#### `is_single_task(self)`
**Module:** `layers.vision_heads.task_types`

Check if only one task is enabled.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:603*

#### `list_attention_types()`
**Module:** `layers.attention.factory`

Get a list of all supported attention layer types.

*📁 src/dl_techniques/layers/attention/factory.py:1069*

#### `loop_body(current, has_changed)`
**Module:** `layers.canny`

*📁 src/dl_techniques/layers/canny.py:301*

#### `loop_cond(current, has_changed)`
**Module:** `layers.canny`

*📁 src/dl_techniques/layers/canny.py:298*

#### `loss_func(self, y_true, y_pred)`
**Module:** `layers.statistics.normalizing_flow`

Compute negative log-likelihood loss for maximum likelihood training.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:685*

#### `loss_func(self, y_true, y_pred)`
**Module:** `layers.statistics.mdn_layer`

MDN loss function using negative log-likelihood.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:358*

#### `mish(inputs)`
**Module:** `layers.activations.mish`

Compute the Mish activation function.

*📁 src/dl_techniques/layers/activations/mish.py:69*

#### `multiscale_fn(n)`
**Module:** `layers.conv2d_builder`

*📁 src/dl_techniques/layers/conv2d_builder.py:160*

#### `multiscales_generator_fn(shape, no_scales, kernel_size, use_max_pool, clip_values, round_values, normalize_values, concrete_functions, jit_compile)`
**Module:** `layers.conv2d_builder`

*📁 src/dl_techniques/layers/conv2d_builder.py:150*

#### `no_interpolate()`
**Module:** `layers.time_series.prism_blocks`

*📁 src/dl_techniques/layers/time_series/prism_blocks.py:456*

#### `normalize_adjacency_matrix(adjacency, normalization)`
**Module:** `layers.experimental.contextual_memory`

Normalize adjacency matrix for stable GNN training.

*📁 src/dl_techniques/layers/experimental/contextual_memory.py:29*

#### `num_parameters(self)`
**Module:** `layers.ffn.swiglu_ffn`

Get the total number of parameters in the layer.

*📁 src/dl_techniques/layers/ffn/swiglu_ffn.py:419*

#### `on_train_batch_end(self, batch, logs)`
**Module:** `layers.statistics.residual_acf`

Log ACF statistics at specified intervals during training.

*📁 src/dl_techniques/layers/statistics/residual_acf.py:542*

#### `output_size(self)`
**Module:** `layers.ntm.baseline_ntm`

Return output size for RNN compatibility.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:920*

#### `parse_task_list(tasks, validate_compatibility)`
**Module:** `layers.vision_heads.task_types`

Parse various task input formats into TaskConfiguration.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:879*

#### `predict_intervals(self, inputs)`
**Module:** `layers.time_series.forecasting_layers`

Compute calibrated prediction intervals during inference.

*📁 src/dl_techniques/layers/time_series/forecasting_layers.py:462*

#### `predict_ood(self, data)`
**Module:** `layers.experimental.band_rms_ood`

Predict OOD for input data.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:799*

#### `probability_type(self)`
**Module:** `layers.activations.probability_output`

Return the configured probability type.

*📁 src/dl_techniques/layers/activations/probability_output.py:315*

#### `quantize_from_indices(self, indices)`
**Module:** `layers.vector_quantizer`

Convert discrete indices back to continuous embeddings.

*📁 src/dl_techniques/layers/vector_quantizer.py:396*

#### `read(self, memory_state, read_weights)`
**Module:** `layers.ntm.baseline_ntm`

Read from memory using attention weights.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:90*

#### `read(self, memory_state, read_weights)`
**Module:** `layers.ntm.ntm_interface`

Read from memory using attention weights.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:362*

#### `reconstruct(self, visible, n_steps)`
**Module:** `layers.restricted_boltzmann_machine`

Reconstruct visible units through Gibbs sampling.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:581*

#### `reparameterize(self)`
**Module:** `layers.repmixer_block`

Reparameterize all blocks in the stem for efficient inference.

*📁 src/dl_techniques/layers/repmixer_block.py:568*

#### `requires_generation(self)`
**Module:** `layers.nlp_heads.task_types`

Check if any task requires generation capabilities.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:493*

#### `requires_generation(self)`
**Module:** `layers.vlm_heads.task_types`

Check if any task requires text generation capabilities.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:299*

#### `requires_sequence_pair(self)`
**Module:** `layers.nlp_heads.task_types`

Check if any task requires sequence pair inputs.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:483*

#### `requires_token_level(self)`
**Module:** `layers.nlp_heads.task_types`

Check if any task requires token-level processing.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:472*

#### `reset_carry(self, reset_flag, carry)`
**Module:** `layers.blt_core`

Reset carry state for halted sequences.

*📁 src/dl_techniques/layers/blt_core.py:432*

#### `reset_carry(self, reset_flag, carry)`
**Module:** `layers.reasoning.hrm_reasoning_core`

Reset carry state for halted sequences.

*📁 src/dl_techniques/layers/reasoning/hrm_reasoning_core.py:552*

#### `reset_centroids(self, new_centroids)`
**Module:** `layers.kmeans`

Reset centroids to new values or reinitialize.

*📁 src/dl_techniques/layers/kmeans.py:676*

#### `reset_memory(self, batch_size)`
**Module:** `layers.ntm.baseline_ntm`

Reset memory (no-op in wrapped mode).

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1367*

#### `reset_memory(self, batch_size)`
**Module:** `layers.ntm.ntm_interface`

Reset memory to initial state.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:771*

#### `reset_reparameterization(self)`
**Module:** `layers.repmixer_block`

Reset all blocks to training mode.

*📁 src/dl_techniques/layers/repmixer_block.py:578*

#### `reset_stats(self)`
**Module:** `layers.statistics.scaler`

Reset all stored statistics to initial values.

*📁 src/dl_techniques/layers/statistics/scaler.py:547*

#### `sample(self, num_samples, context)`
**Module:** `layers.statistics.normalizing_flow`

Generate samples from the learned conditional distribution.

*📁 src/dl_techniques/layers/statistics/normalizing_flow.py:718*

#### `sample(self, y_pred, temperature)`
**Module:** `layers.statistics.mdn_layer`

Samples from the predicted mixture distribution.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:386*

#### `sample_hidden_given_visible(self, visible, sample)`
**Module:** `layers.restricted_boltzmann_machine`

Sample or compute hidden unit activations given visible units.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:354*

#### `sample_visible_given_hidden(self, hidden, sample)`
**Module:** `layers.restricted_boltzmann_machine`

Sample or compute visible unit activations given hidden units.

*📁 src/dl_techniques/layers/restricted_boltzmann_machine.py:378*

#### `saturated_mish(inputs, alpha, beta, mish_at_alpha)`
**Module:** `layers.activations.mish`

Compute the Saturated Mish activation function.

*📁 src/dl_techniques/layers/activations/mish.py:89*

#### `scaled_dot_product_attention(self, q, k, v, attention_mask, training)`
**Module:** `layers.attention.gated_attention`

Compute scaled dot-product attention.

*📁 src/dl_techniques/layers/attention/gated_attention.py:447*

#### `set_external_confidence(self, confidence)`
**Module:** `layers.experimental.band_rms_ood`

Set external confidence signal for prediction-based confidence.

*📁 src/dl_techniques/layers/experimental/band_rms_ood.py:225*

#### `set_temperature(self, new_temperature)`
**Module:** `layers.neuro_grid`

Update the temperature parameter for dynamic control during training.

*📁 src/dl_techniques/layers/neuro_grid.py:972*

#### `sharpen_weights(weights, gamma, epsilon)`
**Module:** `layers.ntm.ntm_interface`

Sharpen attention weights using gamma parameter.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:920*

#### `smooth_transition(t)`
**Module:** `layers.shearlet_transform`

*📁 src/dl_techniques/layers/shearlet_transform.py:273*

#### `split_mixture_params(self, y_pred)`
**Module:** `layers.statistics.mdn_layer`

Splits the concatenated network output into parameter tensors.

*📁 src/dl_techniques/layers/statistics/mdn_layer.py:340*

#### `state_size(self)`
**Module:** `layers.ntm.baseline_ntm`

Return state sizes for RNN compatibility.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:915*

#### `step(state, current_rotation)`
**Module:** `layers.experimental.field_embeddings`

Apply current rotation to accumulated state via matrix multiplication.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:396*

#### `step(self, inputs, memory_state, head_states, controller_state, training)`
**Module:** `layers.ntm.baseline_ntm`

Single step (not used - RNN handles internally).

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:1294*

#### `step(self, inputs, memory_state, head_states, controller_state, training)`
**Module:** `layers.ntm.ntm_interface`

Perform a single timestep of the NTM.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:672*

#### `tasks(self)`
**Module:** `layers.nlp_heads.task_types`

Get the set of enabled tasks.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:464*

#### `tasks(self)`
**Module:** `layers.vision_heads.task_types`

Get the set of enabled tasks.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:555*

#### `tasks(self)`
**Module:** `layers.vlm_heads.task_types`

Get the set of enabled tasks.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:291*

#### `text_to_bytes(self, text, add_bos, add_eos)`
**Module:** `layers.blt_blocks`

Convert text string to byte token sequence.

*📁 src/dl_techniques/layers/blt_blocks.py:285*

#### `thresh_max(x, axis, slope, epsilon)`
**Module:** `layers.activations.thresh_max`

Functional interface for ThreshMax activation.

*📁 src/dl_techniques/layers/activations/thresh_max.py:276*

#### `to_dict(self)`
**Module:** `layers.nlp_heads.task_types`

Convert configuration to dictionary for serialization.

*📁 src/dl_techniques/layers/nlp_heads/task_types.py:515*

#### `to_dict(self)`
**Module:** `layers.vision_heads.task_types`

Convert configuration to dictionary for serialization.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:647*

#### `to_dict(self)`
**Module:** `layers.vlm_heads.task_types`

Convert configuration to dictionary for serialization.

*📁 src/dl_techniques/layers/vlm_heads/task_types.py:312*

#### `to_dict(self)`
**Module:** `layers.moe.config`

Convert configuration to dictionary for serialization.

*📁 src/dl_techniques/layers/moe/config.py:248*

#### `to_dict(self)`
**Module:** `layers.ntm.ntm_interface`

Convert configuration to dictionary for serialization.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:268*

#### `to_string(self)`
**Module:** `layers.layer_scale`

Convert enum to string representation.

*📁 src/dl_techniques/layers/layer_scale.py:93*

#### `to_string(self)`
**Module:** `layers.conv2d_builder`

*📁 src/dl_techniques/layers/conv2d_builder.py:240*

#### `to_strings(cls, tasks)`
**Module:** `layers.vision_heads.task_types`

Convert list of TaskTypes to list of strings.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:436*

#### `tokenize_texts(self, texts)`
**Module:** `layers.tokenizers.bpe`

Tokenize a list of text strings (preprocessing method).

*📁 src/dl_techniques/layers/tokenizers/bpe.py:317*

#### `tokens_to_text(self, tokens)`
**Module:** `layers.blt_blocks`

Convert byte token sequence back to text.

*📁 src/dl_techniques/layers/blt_blocks.py:311*

#### `total_anchor_points(self)`
**Module:** `layers.anchor_generator`

Calculate total number of anchor points across all stride levels.

*📁 src/dl_techniques/layers/anchor_generator.py:257*

#### `train_bpe(texts, vocab_size, min_frequency, do_lower_case, handle_punctuation)`
**Module:** `layers.tokenizers.bpe`

Train BPE tokenizer on a corpus of texts with optimized performance.

*📁 src/dl_techniques/layers/tokenizers/bpe.py:23*

#### `transform(self, inputs)`
**Module:** `layers.statistics.invertible_kernel_pca`

Transform inputs to principal components (alias for call).

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:502*

#### `type_config(self)`
**Module:** `layers.activations.probability_output`

Return the type-specific configuration.

*📁 src/dl_techniques/layers/activations/probability_output.py:320*

#### `update_grid_from_samples(self, x)`
**Module:** `layers.kan_linear`

Update B-spline grid based on input data statistics.

*📁 src/dl_techniques/layers/kan_linear.py:404*

#### `update_pca_components(self, rff_features, training)`
**Module:** `layers.statistics.invertible_kernel_pca`

Update PCA components using eigendecomposition of RFF features.

*📁 src/dl_techniques/layers/statistics/invertible_kernel_pca.py:420*

#### `upsample(input_layer, upsample_type, conv_params, bn_params, ln_params)`
**Module:** `layers.upsample`

Applies upsampling operation to the input layer based on specified strategy.

*📁 src/dl_techniques/layers/upsample.py:102*

#### `validate_activation_config(activation_type)`
**Module:** `layers.activations.factory`

Validate activation layer configuration parameters.

*📁 src/dl_techniques/layers/activations/factory.py:331*

#### `validate_attention_config(attention_type)`
**Module:** `layers.attention.factory`

Validate attention layer configuration parameters against type requirements.

*📁 src/dl_techniques/layers/attention/factory.py:760*

#### `validate_embedding_config(embedding_type)`
**Module:** `layers.embedding.factory`

Validate embedding configuration parameters.

*📁 src/dl_techniques/layers/embedding/factory.py:165*

#### `validate_ffn_config(ffn_type)`
**Module:** `layers.ffn.factory`

Validate FFN configuration parameters.

*📁 src/dl_techniques/layers/ffn/factory.py:317*

#### `validate_field_config(layer_type)`
**Module:** `layers.geometric.fields`

Validate configuration for a field layer type.

*📁 src/dl_techniques/layers/geometric/fields/__init__.py:156*

#### `validate_normalization_config(normalization_type)`
**Module:** `layers.norms.factory`

Validate normalization configuration parameters.

*📁 src/dl_techniques/layers/norms/factory.py:378*

#### `validate_task_combination(tasks)`
**Module:** `layers.vision_heads.task_types`

Validate if a combination of tasks is reasonable.

*📁 src/dl_techniques/layers/vision_heads/task_types.py:940*

#### `verify_rotation_matrix(rotation, tolerance)`
**Module:** `layers.experimental.field_embeddings`

Verify that a matrix is a valid rotation (orthogonal with det=1).

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:988*

#### `visualize_stress_trajectory(stress_values, threshold)`
**Module:** `layers.experimental.field_embeddings`

Visualize stress values over a batch to identify anomalies.

*📁 src/dl_techniques/layers/experimental/field_embeddings.py:1035*

#### `width_values(self)`
**Module:** `layers.radial_basis_function`

Get current effective width (gamma) values.

*📁 src/dl_techniques/layers/radial_basis_function.py:334*

#### `write(self, memory_state, write_weights, erase_vector, add_vector)`
**Module:** `layers.ntm.baseline_ntm`

Write to memory using erase-then-add mechanism.

*📁 src/dl_techniques/layers/ntm/baseline_ntm.py:109*

#### `write(self, memory_state, write_weights, erase_vector, add_vector)`
**Module:** `layers.ntm.ntm_interface`

Write to memory using erase and add operations.

*📁 src/dl_techniques/layers/ntm/ntm_interface.py:380*

### Losses Functions

#### `analyze_decoupled_information_loss(loss_fn, y_true, y_pred)`
**Module:** `losses.decoupled_information_loss`

Analyze individual components of the DecoupledInformationLoss for debugging.

*📁 src/dl_techniques/losses/decoupled_information_loss.py:301*

#### `analyze_focal_uncertainty_loss(loss_fn, y_true, y_pred)`
**Module:** `losses.focal_uncertainty_loss`

Analyze individual components of the FocalUncertaintyLoss for debugging.

*📁 src/dl_techniques/losses/focal_uncertainty_loss.py:365*

#### `analyze_loss_components(loss_fn, y_true, y_pred)`
**Module:** `losses.goodhart_loss`

Analyzes individual components of the GoodhartAwareLoss for debugging.

*📁 src/dl_techniques/losses/goodhart_loss.py:284*

#### `analyze_margin_loss_components(loss_fn, y_true, y_pred)`
**Module:** `losses.capsule_margin_loss`

Analyzes individual components of the CapsuleMarginLoss for debugging.

*📁 src/dl_techniques/losses/capsule_margin_loss.py:230*

#### `approximate_distance_transform(mask)`
**Module:** `losses.segmentation_loss`

Approximate distance transform using morphological operations.

*📁 src/dl_techniques/losses/segmentation_loss.py:486*

#### `boundary_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Boundary loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:406*

#### `call(self, y_true, y_pred)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Compute the Brier Score loss.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:119*

#### `call(self, y_true, y_pred)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Compute the Spiegelhalter Z-test loss.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:210*

#### `call(self, y_true, y_pred)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Compute the combined calibration loss.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:342*

#### `call(self, y_true, y_pred)`
**Module:** `losses.hrm_loss`

Compute stable max cross entropy loss.

*📁 src/dl_techniques/losses/hrm_loss.py:71*

#### `call(self, y_true, y_pred)`
**Module:** `losses.hrm_loss`

Compute combined HRM loss.

*📁 src/dl_techniques/losses/hrm_loss.py:161*

#### `call(self, y_true, y_pred)`
**Module:** `losses.goodhart_loss`

Computes the Goodhart-aware loss for a batch.

*📁 src/dl_techniques/losses/goodhart_loss.py:164*

#### `call(self, y_true, y_pred)`
**Module:** `losses.capsule_margin_loss`

Computes the margin loss for capsule networks.

*📁 src/dl_techniques/losses/capsule_margin_loss.py:188*

#### `call(self, y_true, y_pred)`
**Module:** `losses.affine_invariant_loss`

Compute affine-invariant depth loss.

*📁 src/dl_techniques/losses/affine_invariant_loss.py:96*

#### `call(self, y_true, y_pred)`
**Module:** `losses.decoupled_information_loss`

Compute the decoupled information-theoretic loss for a batch.

*📁 src/dl_techniques/losses/decoupled_information_loss.py:184*

#### `call(self, y_true, y_pred)`
**Module:** `losses.huber_loss`

Compute Huber loss.

*📁 src/dl_techniques/losses/huber_loss.py:80*

#### `call(self, y_true, y_pred)`
**Module:** `losses.image_restoration_loss`

Compute Charbonnier loss.

*📁 src/dl_techniques/losses/image_restoration_loss.py:140*

#### `call(self, y_true, y_pred)`
**Module:** `losses.image_restoration_loss`

Compute frequency loss via FFT amplitude comparison.

*📁 src/dl_techniques/losses/image_restoration_loss.py:204*

#### `call(self, y_true, y_pred)`
**Module:** `losses.image_restoration_loss`

Compute edge loss via Laplacian of Gaussian.

*📁 src/dl_techniques/losses/image_restoration_loss.py:343*

#### `call(self, y_true, y_pred)`
**Module:** `losses.image_restoration_loss`

Compute perceptual loss using VGG19 features.

*📁 src/dl_techniques/losses/image_restoration_loss.py:463*

#### `call(self, y_true, y_pred)`
**Module:** `losses.image_restoration_loss`

Compute enhance loss for deep supervision.

*📁 src/dl_techniques/losses/image_restoration_loss.py:545*

#### `call(self, y_true, y_pred)`
**Module:** `losses.image_restoration_loss`

Compute composite loss.

*📁 src/dl_techniques/losses/image_restoration_loss.py:650*

#### `call(self, y_true, y_pred)`
**Module:** `losses.clip_contrastive_loss`

Compute symmetric contrastive loss for a batch.

*📁 src/dl_techniques/losses/clip_contrastive_loss.py:415*

#### `call(self, y_true, y_pred)`
**Module:** `losses.siglip_contrastive_loss`

Compute SigLIP contrastive loss.

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:113*

#### `call(self, y_true, y_pred)`
**Module:** `losses.siglip_contrastive_loss`

Compute adaptive SigLIP loss with dynamic temperature.

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:220*

#### `call(self, y_true, y_pred)`
**Module:** `losses.siglip_contrastive_loss`

Compute hybrid contrastive + score matching loss.

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:325*

#### `call(self, y_true, y_pred)`
**Module:** `losses.focal_uncertainty_loss`

Compute the focal loss with uncertainty regularization.

*📁 src/dl_techniques/losses/focal_uncertainty_loss.py:182*

#### `call(self, inputs, training)`
**Module:** `losses.any_loss`

Apply the approximation function to transform probabilities to near-binary values.

*📁 src/dl_techniques/losses/any_loss.py:171*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Abstract method that should be implemented by subclasses.

*📁 src/dl_techniques/losses/any_loss.py:340*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the accuracy loss.

*📁 src/dl_techniques/losses/any_loss.py:447*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the precision loss.

*📁 src/dl_techniques/losses/any_loss.py:531*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the recall loss.

*📁 src/dl_techniques/losses/any_loss.py:614*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the specificity loss.

*📁 src/dl_techniques/losses/any_loss.py:697*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the F1 loss.

*📁 src/dl_techniques/losses/any_loss.py:787*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the F-beta loss.

*📁 src/dl_techniques/losses/any_loss.py:893*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the balanced accuracy loss.

*📁 src/dl_techniques/losses/any_loss.py:1000*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the geometric mean loss.

*📁 src/dl_techniques/losses/any_loss.py:1086*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the Youden's J loss.

*📁 src/dl_techniques/losses/any_loss.py:1174*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the MCC loss.

*📁 src/dl_techniques/losses/any_loss.py:1273*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the Cohen's Kappa loss.

*📁 src/dl_techniques/losses/any_loss.py:1379*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the IoU loss.

*📁 src/dl_techniques/losses/any_loss.py:1488*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the Dice loss.

*📁 src/dl_techniques/losses/any_loss.py:1571*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the Tversky loss.

*📁 src/dl_techniques/losses/any_loss.py:1688*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the Focal Tversky loss.

*📁 src/dl_techniques/losses/any_loss.py:1824*

#### `call(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute the combined loss.

*📁 src/dl_techniques/losses/any_loss.py:1984*

#### `call(self, y_true, y_pred)`
**Module:** `losses.mase_loss`

Compute MASE loss.

*📁 src/dl_techniques/losses/mase_loss.py:89*

#### `call(self, y_true, y_pred)`
**Module:** `losses.smape_loss`

Compute SMAPE loss.

*📁 src/dl_techniques/losses/smape_loss.py:80*

#### `call(self, y_true, y_pred)`
**Module:** `losses.tabm_loss`

Compute loss for TabM ensemble predictions.

*📁 src/dl_techniques/losses/tabm_loss.py:34*

#### `call(self, y_true, y_pred)`
**Module:** `losses.clustering_loss`

Compute the clustering loss.

*📁 src/dl_techniques/losses/clustering_loss.py:168*

#### `call(self, y_true, y_pred)`
**Module:** `losses.wasserstein_loss`

Compute the Wasserstein loss.

*📁 src/dl_techniques/losses/wasserstein_loss.py:258*

#### `call(self, y_true, y_pred)`
**Module:** `losses.wasserstein_loss`

Compute the Wasserstein loss with gradient penalty.

*📁 src/dl_techniques/losses/wasserstein_loss.py:346*

#### `call(self, y_true, y_pred)`
**Module:** `losses.wasserstein_loss`

Compute the Wasserstein divergence.

*📁 src/dl_techniques/losses/wasserstein_loss.py:459*

#### `call(self, y_true, y_pred)`
**Module:** `losses.sparsemax_loss`

Compute sparsemax loss per sample.

*📁 src/dl_techniques/losses/sparsemax_loss.py:226*

#### `call(self, y_true, y_pred)`
**Module:** `losses.dino_loss`

Compute DINO loss between teacher and student outputs.

*📁 src/dl_techniques/losses/dino_loss.py:187*

#### `call(self, y_true, y_pred, mask)`
**Module:** `losses.dino_loss`

Compute iBOT loss on masked patches only.

*📁 src/dl_techniques/losses/dino_loss.py:377*

#### `call(self, y_true, y_pred)`
**Module:** `losses.dino_loss`

Compute KoLeo regularization loss.

*📁 src/dl_techniques/losses/dino_loss.py:542*

#### `call(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Call the wrapped loss function.

*📁 src/dl_techniques/losses/segmentation_loss.py:567*

#### `call(self, y_true, y_pred)`
**Module:** `losses.nano_vlm_loss`

Compute the autoregressive language modeling loss.

*📁 src/dl_techniques/losses/nano_vlm_loss.py:151*

#### `call(self, y_true, y_pred)`
**Module:** `losses.multi_labels_loss`

Compute loss independently per channel with adaptive weighting.

*📁 src/dl_techniques/losses/multi_labels_loss.py:83*

#### `call(self, y_true, y_pred)`
**Module:** `losses.multi_labels_loss`

Compute focal loss element-wise.

*📁 src/dl_techniques/losses/multi_labels_loss.py:238*

#### `call(self, y_true, y_pred)`
**Module:** `losses.multi_labels_loss`

Compute Dice loss.

*📁 src/dl_techniques/losses/multi_labels_loss.py:329*

#### `call(self, y_true, y_pred)`
**Module:** `losses.chamfer_loss`

Computes the Chamfer distance between two point clouds.

*📁 src/dl_techniques/losses/chamfer_loss.py:39*

#### `call(self, y_true, y_pred)`
**Module:** `losses.feature_alignment_loss`

Compute feature alignment loss.

*📁 src/dl_techniques/losses/feature_alignment_loss.py:106*

#### `call(self, y_true, y_pred)`
**Module:** `losses.yolo12_multitask_loss`

Compute the YOLOv12 detection loss.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:364*

#### `call(self, y_true, y_pred)`
**Module:** `losses.yolo12_multitask_loss`

Compute the combined Dice and Focal loss.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:698*

#### `call(self, y_true, y_pred)`
**Module:** `losses.yolo12_multitask_loss`

Compute the focal loss for classification.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:796*

#### `call(self, y_true, y_pred)`
**Module:** `losses.yolo12_multitask_loss`

Calculates the loss for a single task output.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1005*

#### `call(self, y_true, y_pred)`
**Module:** `losses.quantile_loss`

Compute quantile loss.

*📁 src/dl_techniques/losses/quantile_loss.py:81*

#### `call(self, y_true, y_pred)`
**Module:** `losses.quantile_loss`

Compute vectorized quantile loss.

*📁 src/dl_techniques/losses/quantile_loss.py:136*

#### `capsule_margin_loss(y_pred, y_true, downweight, positive_margin, negative_margin)`
**Module:** `losses.capsule_margin_loss`

*📁 src/dl_techniques/losses/capsule_margin_loss.py:68*

#### `combo_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Combo loss (combination of Dice and Cross-Entropy).

*📁 src/dl_techniques/losses/segmentation_loss.py:379*

#### `compute_boundary_map(mask)`
**Module:** `losses.segmentation_loss`

Compute boundary map using difference-based edge detection.

*📁 src/dl_techniques/losses/segmentation_loss.py:431*

#### `compute_box_and_dfl_only()`
**Module:** `losses.yolo12_multitask_loss`

A nested function to compute losses that only apply to foreground anchors. This is wrapped in an ops.cond to avoid running on batches with no positive assignments.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:453*

#### `compute_clustering_metrics(data, assignments)`
**Module:** `losses.clustering_loss`

Compute various clustering quality metrics.

*📁 src/dl_techniques/losses/clustering_loss.py:369*

#### `compute_confusion_matrix(self, y_true, y_pred)`
**Module:** `losses.any_loss`

Compute differentiable confusion matrix entries.

*📁 src/dl_techniques/losses/any_loss.py:298*

#### `compute_gradient_penalty(critic, real_samples, fake_samples, lambda_gp)`
**Module:** `losses.wasserstein_loss`

Compute gradient penalty for WGAN-GP.

*📁 src/dl_techniques/losses/wasserstein_loss.py:377*

#### `compute_losses()`
**Module:** `losses.yolo12_multitask_loss`

Compute all loss components for a batch of images with ground truth. This function is called only when at least one ground truth box exists in the batch. It performs the following steps: 1.  Assigns ground truth boxes to the most suitable anchor points. 2.  Calculates classification loss for all assigned anchors. 3.  Calculates bounding box (IoU) and Distribution Focal Loss (DFL) only for the positive anchor assignments (foreground).

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:413*

#### `compute_output_shape(self, input_shape)`
**Module:** `losses.focal_uncertainty_loss`

Compute the output shape of the loss.

*📁 src/dl_techniques/losses/focal_uncertainty_loss.py:279*

#### `compute_output_shape(self, input_shape)`
**Module:** `losses.any_loss`

Compute the output shape of the layer.

*📁 src/dl_techniques/losses/any_loss.py:190*

#### `create_adaptive_siglip_loss(initial_temperature, target_entropy)`
**Module:** `losses.siglip_contrastive_loss`

Create adaptive SigLIP loss with dynamic temperature.

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:390*

#### `create_hrm_loss(lm_loss_type, q_loss_weight, ignore_index)`
**Module:** `losses.hrm_loss`

Create HRM loss function.

*📁 src/dl_techniques/losses/hrm_loss.py:233*

#### `create_hybrid_loss(siglip_weight, score_weight)`
**Module:** `losses.siglip_contrastive_loss`

Create hybrid loss combining SigLIP with score matching.

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:403*

#### `create_loss_function(loss_name, config)`
**Module:** `losses.segmentation_loss`

Create a Keras loss function from the specified loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:525*

#### `create_multilabel_segmentation_loss(loss_type, alpha, gamma, channel_weights, smooth)`
**Module:** `losses.multi_labels_loss`

Factory function to create appropriate loss for multi-label segmentation.

*📁 src/dl_techniques/losses/multi_labels_loss.py:381*

#### `create_siglip_loss(temperature, use_learnable_temperature)`
**Module:** `losses.siglip_contrastive_loss`

Create standard SigLIP contrastive loss.

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:377*

#### `create_wgan_gp_losses(lambda_gp)`
**Module:** `losses.wasserstein_loss`

Create critic and generator losses for WGAN-GP.

*📁 src/dl_techniques/losses/wasserstein_loss.py:524*

#### `create_wgan_losses(lambda_gp)`
**Module:** `losses.wasserstein_loss`

Create critic and generator losses for WGAN.

*📁 src/dl_techniques/losses/wasserstein_loss.py:502*

#### `create_yolov12_coco_loss(input_shape)`
**Module:** `losses.yolo12_multitask_loss`

Create loss function specifically for COCO pretraining.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1203*

#### `create_yolov12_crack_loss(input_shape)`
**Module:** `losses.yolo12_multitask_loss`

Create loss function specifically for crack detection fine-tuning.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1226*

#### `create_yolov12_multitask_loss(tasks, num_detection_classes, num_segmentation_classes, num_classification_classes, num_classes, input_shape)`
**Module:** `losses.yolo12_multitask_loss`

Factory function to create the YOLOv12MultiTaskLoss instance.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1145*

#### `cross_entropy_loss(self, y_true, y_pred, weights)`
**Module:** `losses.segmentation_loss`

Implement weighted cross-entropy loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:136*

#### `dice_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Dice loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:170*

#### `focal_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Focal loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:204*

#### `focal_tversky_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Focal Tversky loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:274*

#### `from_config(cls, config)`
**Module:** `losses.focal_uncertainty_loss`

Create a FocalUncertaintyLoss instance from a configuration dictionary.

*📁 src/dl_techniques/losses/focal_uncertainty_loss.py:338*

#### `from_config(cls, config)`
**Module:** `losses.any_loss`

Create an instance from config dictionary.

*📁 src/dl_techniques/losses/any_loss.py:2026*

#### `from_config(cls, config)`
**Module:** `losses.nano_vlm_loss`

Create loss function from configuration.

*📁 src/dl_techniques/losses/nano_vlm_loss.py:264*

#### `from_config(cls, config)`
**Module:** `losses.multi_labels_loss`

*📁 src/dl_techniques/losses/multi_labels_loss.py:186*

#### `from_config(cls, config)`
**Module:** `losses.yolo12_multitask_loss`

Creates a loss instance from its config.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1128*

#### `get_config(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:142*

#### `get_config(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:252*

#### `get_config(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:357*

#### `get_config(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Get metric configuration for serialization.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:449*

#### `get_config(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Get metric configuration for serialization.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:542*

#### `get_config(self)`
**Module:** `losses.hrm_loss`

Get loss configuration.

*📁 src/dl_techniques/losses/hrm_loss.py:106*

#### `get_config(self)`
**Module:** `losses.hrm_loss`

Get loss configuration.

*📁 src/dl_techniques/losses/hrm_loss.py:221*

#### `get_config(self)`
**Module:** `losses.goodhart_loss`

Returns the configuration dictionary for serialization.

*📁 src/dl_techniques/losses/goodhart_loss.py:266*

#### `get_config(self)`
**Module:** `losses.capsule_margin_loss`

Returns the configuration dictionary for serialization.

*📁 src/dl_techniques/losses/capsule_margin_loss.py:214*

#### `get_config(self)`
**Module:** `losses.affine_invariant_loss`

Returns the configuration of the loss function.

*📁 src/dl_techniques/losses/affine_invariant_loss.py:156*

#### `get_config(self)`
**Module:** `losses.decoupled_information_loss`

Return the configuration dictionary for serialization.

*📁 src/dl_techniques/losses/decoupled_information_loss.py:281*

#### `get_config(self)`
**Module:** `losses.huber_loss`

Get loss configuration.

*📁 src/dl_techniques/losses/huber_loss.py:107*

#### `get_config(self)`
**Module:** `losses.image_restoration_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/image_restoration_loss.py:159*

#### `get_config(self)`
**Module:** `losses.image_restoration_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/image_restoration_loss.py:252*

#### `get_config(self)`
**Module:** `losses.image_restoration_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/image_restoration_loss.py:370*

#### `get_config(self)`
**Module:** `losses.image_restoration_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/image_restoration_loss.py:497*

#### `get_config(self)`
**Module:** `losses.image_restoration_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/image_restoration_loss.py:586*

#### `get_config(self)`
**Module:** `losses.image_restoration_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/image_restoration_loss.py:679*

#### `get_config(self)`
**Module:** `losses.clip_contrastive_loss`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/losses/clip_contrastive_loss.py:633*

#### `get_config(self)`
**Module:** `losses.siglip_contrastive_loss`

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:164*

#### `get_config(self)`
**Module:** `losses.siglip_contrastive_loss`

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:271*

#### `get_config(self)`
**Module:** `losses.siglip_contrastive_loss`

*📁 src/dl_techniques/losses/siglip_contrastive_loss.py:364*

#### `get_config(self)`
**Module:** `losses.focal_uncertainty_loss`

Return the configuration dictionary for serialization.

*📁 src/dl_techniques/losses/focal_uncertainty_loss.py:291*

#### `get_config(self)`
**Module:** `losses.any_loss`

Get layer configuration for serialization.

*📁 src/dl_techniques/losses/any_loss.py:205*

#### `get_config(self)`
**Module:** `losses.any_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/any_loss.py:362*

#### `get_config(self)`
**Module:** `losses.any_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/any_loss.py:917*

#### `get_config(self)`
**Module:** `losses.any_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/any_loss.py:1716*

#### `get_config(self)`
**Module:** `losses.any_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/any_loss.py:1854*

#### `get_config(self)`
**Module:** `losses.any_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/any_loss.py:2010*

#### `get_config(self)`
**Module:** `losses.mase_loss`

Get loss configuration.

*📁 src/dl_techniques/losses/mase_loss.py:132*

#### `get_config(self)`
**Module:** `losses.smape_loss`

Get loss configuration.

*📁 src/dl_techniques/losses/smape_loss.py:108*

#### `get_config(self)`
**Module:** `losses.tabm_loss`

*📁 src/dl_techniques/losses/tabm_loss.py:60*

#### `get_config(self)`
**Module:** `losses.clustering_loss`

Get the configuration of the loss function.

*📁 src/dl_techniques/losses/clustering_loss.py:205*

#### `get_config(self)`
**Module:** `losses.wasserstein_loss`

Get the configuration of the loss.

*📁 src/dl_techniques/losses/wasserstein_loss.py:290*

#### `get_config(self)`
**Module:** `losses.wasserstein_loss`

Get the configuration of the loss.

*📁 src/dl_techniques/losses/wasserstein_loss.py:362*

#### `get_config(self)`
**Module:** `losses.wasserstein_loss`

Get the configuration of the loss.

*📁 src/dl_techniques/losses/wasserstein_loss.py:486*

#### `get_config(self)`
**Module:** `losses.sparsemax_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/sparsemax_loss.py:271*

#### `get_config(self)`
**Module:** `losses.dino_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/dino_loss.py:250*

#### `get_config(self)`
**Module:** `losses.dino_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/dino_loss.py:439*

#### `get_config(self)`
**Module:** `losses.dino_loss`

Return configuration for serialization.

*📁 src/dl_techniques/losses/dino_loss.py:584*

#### `get_config(self)`
**Module:** `losses.segmentation_loss`

Get configuration for serialization.

*📁 src/dl_techniques/losses/segmentation_loss.py:579*

#### `get_config(self)`
**Module:** `losses.nano_vlm_loss`

Get the loss function configuration.

*📁 src/dl_techniques/losses/nano_vlm_loss.py:249*

#### `get_config(self)`
**Module:** `losses.multi_labels_loss`

*📁 src/dl_techniques/losses/multi_labels_loss.py:170*

#### `get_config(self)`
**Module:** `losses.multi_labels_loss`

*📁 src/dl_techniques/losses/multi_labels_loss.py:283*

#### `get_config(self)`
**Module:** `losses.multi_labels_loss`

*📁 src/dl_techniques/losses/multi_labels_loss.py:374*

#### `get_config(self)`
**Module:** `losses.feature_alignment_loss`

Returns the configuration of the loss function.

*📁 src/dl_techniques/losses/feature_alignment_loss.py:141*

#### `get_config(self)`
**Module:** `losses.yolo12_multitask_loss`

Returns the serializable config of the loss.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:556*

#### `get_config(self)`
**Module:** `losses.yolo12_multitask_loss`

Returns the serializable config of the loss.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:742*

#### `get_config(self)`
**Module:** `losses.yolo12_multitask_loss`

Returns the serializable config of the loss.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:817*

#### `get_config(self)`
**Module:** `losses.yolo12_multitask_loss`

Returns the serializable config of the loss.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1102*

#### `get_config(self)`
**Module:** `losses.quantile_loss`

Get loss configuration.

*📁 src/dl_techniques/losses/quantile_loss.py:101*

#### `get_config(self)`
**Module:** `losses.quantile_loss`

Get loss configuration for serialization.

*📁 src/dl_techniques/losses/quantile_loss.py:170*

#### `get_individual_losses(self)`
**Module:** `losses.yolo12_multitask_loss`

Provides API compatibility for the callback.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1093*

#### `get_loss(name)`
**Module:** `losses.any_loss`

Factory function to get a loss by name.

*📁 src/dl_techniques/losses/any_loss.py:2160*

#### `get_task_weights(self)`
**Module:** `losses.yolo12_multitask_loss`

Returns the current weights for each task, for callback logging.

*📁 src/dl_techniques/losses/yolo12_multitask_loss.py:1059*

#### `hausdorff_distance_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement an approximation of Hausdorff Distance loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:462*

#### `lovasz_grad(gt_sorted)`
**Module:** `losses.segmentation_loss`

Compute Lovász gradient.

*📁 src/dl_techniques/losses/segmentation_loss.py:336*

#### `lovasz_softmax_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Lovász-Softmax loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:314*

#### `mase_metric(seasonal_periods)`
**Module:** `losses.mase_loss`

Factory function for a MASE metric for use with `model.compile()`.

*📁 src/dl_techniques/losses/mase_loss.py:141*

#### `metric(y_true, y_pred)`
**Module:** `losses.mase_loss`

Computes the Mean Absolute Scaled Error metric.

*📁 src/dl_techniques/losses/mase_loss.py:153*

#### `on_epoch_end(self, epoch, logs)`
**Module:** `losses.clustering_loss`

Compute metrics at the end of each epoch.

*📁 src/dl_techniques/losses/clustering_loss.py:263*

#### `reset_state(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Reset the metric state.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:444*

#### `reset_state(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Reset the metric state.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:537*

#### `result(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Compute the final metric result.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:431*

#### `result(self)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Compute the final metric result.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:521*

#### `smape_metric(y_true, y_pred)`
**Module:** `losses.smape_loss`

SMAPE metric function for use with `model.compile()`.

*📁 src/dl_techniques/losses/smape_loss.py:118*

#### `temperature_value(self)`
**Module:** `losses.clip_contrastive_loss`

Get the current temperature parameter value.

*📁 src/dl_techniques/losses/clip_contrastive_loss.py:691*

#### `tversky_loss(self, y_true, y_pred)`
**Module:** `losses.segmentation_loss`

Implement Tversky loss.

*📁 src/dl_techniques/losses/segmentation_loss.py:236*

#### `update_center(self, teacher_logits)`
**Module:** `losses.dino_loss`

Update center vector using exponential moving average of teacher logits.

*📁 src/dl_techniques/losses/dino_loss.py:218*

#### `update_center(self, teacher_patch_logits)`
**Module:** `losses.dino_loss`

Update center using all teacher patch tokens.

*📁 src/dl_techniques/losses/dino_loss.py:416*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Update the metric state.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:405*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `losses.brier_spiegelhalters_ztest_loss`

Update the metric state.

*📁 src/dl_techniques/losses/brier_spiegelhalters_ztest_loss.py:493*

#### `update_temperature(self, new_temperature)`
**Module:** `losses.clip_contrastive_loss`

Update the temperature parameter dynamically during training.

*📁 src/dl_techniques/losses/clip_contrastive_loss.py:710*

#### `weighted_bce_fn(y_true, y_pred)`
**Module:** `losses.multi_labels_loss`

*📁 src/dl_techniques/losses/multi_labels_loss.py:430*

### Metrics Functions

#### `calculate_comprehensive_metrics(y_true, y_pred, backcast)`
**Module:** `metrics.time_series_metrics`

Calculate comprehensive time series forecasting metrics.

*📁 src/dl_techniques/metrics/time_series_metrics.py:88*

#### `compute_f1(self)`
**Module:** `metrics.multi_label_metrics`

Compute per-class F1 score.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:146*

#### `compute_precision(self)`
**Module:** `metrics.multi_label_metrics`

Compute per-class precision.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:124*

#### `compute_recall(self)`
**Module:** `metrics.multi_label_metrics`

Compute per-class recall.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:135*

#### `get_config(self)`
**Module:** `metrics.clip_accuracy`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/metrics/clip_accuracy.py:408*

#### `get_config(self)`
**Module:** `metrics.clip_accuracy`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/metrics/clip_accuracy.py:688*

#### `get_config(self)`
**Module:** `metrics.multi_label_metrics`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:174*

#### `get_config(self)`
**Module:** `metrics.perplexity_metric`

Returns the serializable config of the metric.

*📁 src/dl_techniques/metrics/perplexity_metric.py:283*

#### `perplexity(y_true, y_pred, from_logits, ignore_class)`
**Module:** `metrics.perplexity_metric`

Functional interface for computing perplexity.

*📁 src/dl_techniques/metrics/perplexity_metric.py:295*

#### `reset_state(self)`
**Module:** `metrics.capsule_accuracy`

*📁 src/dl_techniques/metrics/capsule_accuracy.py:44*

#### `reset_state(self)`
**Module:** `metrics.psnr_metric`

Reset metric state for new epoch or evaluation period.

*📁 src/dl_techniques/metrics/psnr_metric.py:63*

#### `reset_state(self)`
**Module:** `metrics.hrm_metrics`

Reset all metrics.

*📁 src/dl_techniques/metrics/hrm_metrics.py:97*

#### `reset_state(self)`
**Module:** `metrics.clip_accuracy`

Reset metric state for new epoch or evaluation.

*📁 src/dl_techniques/metrics/clip_accuracy.py:398*

#### `reset_state(self)`
**Module:** `metrics.clip_accuracy`

Reset metric state for new epoch or evaluation.

*📁 src/dl_techniques/metrics/clip_accuracy.py:678*

#### `reset_state(self)`
**Module:** `metrics.multi_label_metrics`

Reset all accumulated statistics to zero.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:166*

#### `reset_state(self)`
**Module:** `metrics.time_series_metrics`

Reset the metric state.

*📁 src/dl_techniques/metrics/time_series_metrics.py:77*

#### `reset_state(self)`
**Module:** `metrics.perplexity_metric`

Resets all metric state variables.

*📁 src/dl_techniques/metrics/perplexity_metric.py:278*

#### `result(self)`
**Module:** `metrics.capsule_accuracy`

*📁 src/dl_techniques/metrics/capsule_accuracy.py:41*

#### `result(self)`
**Module:** `metrics.psnr_metric`

Compute the mean PSNR across all processed samples.

*📁 src/dl_techniques/metrics/psnr_metric.py:59*

#### `result(self)`
**Module:** `metrics.hrm_metrics`

Get current metric results.

*📁 src/dl_techniques/metrics/hrm_metrics.py:88*

#### `result(self)`
**Module:** `metrics.clip_accuracy`

Compute current accuracy value.

*📁 src/dl_techniques/metrics/clip_accuracy.py:386*

#### `result(self)`
**Module:** `metrics.clip_accuracy`

Compute current recall@k value.

*📁 src/dl_techniques/metrics/clip_accuracy.py:666*

#### `result(self)`
**Module:** `metrics.multi_label_metrics`

Compute macro-averaged F1 score.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:157*

#### `result(self)`
**Module:** `metrics.time_series_metrics`

Compute the current metric value.

*📁 src/dl_techniques/metrics/time_series_metrics.py:67*

#### `result(self)`
**Module:** `metrics.perplexity_metric`

Computes and returns the metric result.

*📁 src/dl_techniques/metrics/perplexity_metric.py:265*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.capsule_accuracy`

Update accuracy state based on capsule lengths.

*📁 src/dl_techniques/metrics/capsule_accuracy.py:16*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.psnr_metric`

Update PSNR state using only the primary output.

*📁 src/dl_techniques/metrics/psnr_metric.py:28*

#### `update_state(self, y_true, y_pred)`
**Module:** `metrics.hrm_metrics`

Update metric states.

*📁 src/dl_techniques/metrics/hrm_metrics.py:32*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.clip_accuracy`

Update metric state with batch predictions.

*📁 src/dl_techniques/metrics/clip_accuracy.py:313*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.clip_accuracy`

Update metric state with batch predictions.

*📁 src/dl_techniques/metrics/clip_accuracy.py:577*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.multi_label_metrics`

Accumulate confusion matrix statistics.

*📁 src/dl_techniques/metrics/multi_label_metrics.py:68*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.time_series_metrics`

Update the metric state with new predictions.

*📁 src/dl_techniques/metrics/time_series_metrics.py:34*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `metrics.perplexity_metric`

Updates the metric state.

*📁 src/dl_techniques/metrics/perplexity_metric.py:209*

### Models Functions

#### `add_noise(self, original_samples, noise, timesteps)`
**Module:** `models.nano_vlm_world_model.scheduler`

Forward diffusion: Add noise to samples according to q(x_t | x_0).

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:131*

#### `add_pos_embed(x_with_cls)`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:651*

#### `align(self, XA, XB, approx_clusters, approx_runs, approx_neighbors, refine1_iterations, refine1_sample_size, refine1_neighbors, refine2_clusters, smoothing_alpha)`
**Module:** `models.mini_vec2vec.model`

Execute the full mini-vec2vec alignment pipeline (Algorithm 1).

*📁 src/dl_techniques/models/mini_vec2vec/model.py:392*

#### `analyze_som_clustering(model, input_dataset, max_samples)`
**Module:** `models.qwen.qwen3_som`

Analyze clustering quality of SOM layers.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:987*

#### `apply_mask_after_stem(stem, images, mask)`
**Module:** `models.vit_hmlp.model`

Process images with hMLP stem then apply masking.

*📁 src/dl_techniques/models/vit_hmlp/model.py:968*

#### `apply_masks(args)`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:614*

#### `body(t, h, ys)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:204*

#### `body(t, h, ys)`
**Module:** `models.mamba.components`

Single timestep of SSM computation.

*📁 src/dl_techniques/models/mamba/components.py:400*

#### `build(self, input_shape)`
**Module:** `models.nano_vlm.model`

Build the NanoVLM and all its sub-layers.

*📁 src/dl_techniques/models/nano_vlm/model.py:391*

#### `build(self, input_shape)`
**Module:** `models.depth_anything.model`

Build the model components.

*📁 src/dl_techniques/models/depth_anything/model.py:165*

#### `build(self, input_shape)`
**Module:** `models.depth_anything.components`

Build decoder layers based on input shape.

*📁 src/dl_techniques/models/depth_anything/components.py:121*

#### `build(self, input_shape)`
**Module:** `models.nano_vlm_world_model.model`

Build all components.

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:171*

#### `build(self, input_shape)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Build layer based on input shapes.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:150*

#### `build(self, input_shape)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Build layer based on input shapes.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:276*

#### `build(self, input_shape)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Build the stem layers.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:92*

#### `build(self, input_shape)`
**Module:** `models.mdn.model`

Build the model with the given input shape.

*📁 src/dl_techniques/models/mdn/model.py:233*

#### `build(self, input_shape)`
**Module:** `models.adaptive_ema.model`

Build the model and create threshold variables.

*📁 src/dl_techniques/models/adaptive_ema/model.py:200*

#### `build(self, input_shape)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:146*

#### `build(self, input_shape)`
**Module:** `models.mamba.components`

Create layer weights and build sub-layers.

*📁 src/dl_techniques/models/mamba/components.py:245*

#### `build(self, input_shape)`
**Module:** `models.mamba.components`

Build sub-layers.

*📁 src/dl_techniques/models/mamba/components.py:660*

#### `build(self, input_shape)`
**Module:** `models.byte_latent_transformer.model`

Build the model and all sub-layers.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:293*

#### `build(self, input_shape)`
**Module:** `models.convnext.convnext_v2`

Builds the model and its layers.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:347*

#### `build(self, input_shape)`
**Module:** `models.convnext.convnext_v1`

Builds the model and its layers.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:330*

#### `build(self, input_shape)`
**Module:** `models.detr.model`

Build encoder and decoder layers.

*📁 src/dl_techniques/models/detr/model.py:179*

#### `build(self, input_shape)`
**Module:** `models.detr.model`

Build all sub-layers.

*📁 src/dl_techniques/models/detr/model.py:364*

#### `build(self, input_shape)`
**Module:** `models.fftnet.model`

Build the layer by creating frequency-dependent parameters.

*📁 src/dl_techniques/models/fftnet/model.py:143*

#### `build(self, input_shape)`
**Module:** `models.fftnet.model`

Build sub-layers.

*📁 src/dl_techniques/models/fftnet/model.py:296*

#### `build(self, input_shape)`
**Module:** `models.fftnet.model`

Build the model by creating learnable parameters.

*📁 src/dl_techniques/models/fftnet/model.py:553*

#### `build(self, input_shape)`
**Module:** `models.fftnet.components`

Build sub-layers explicitly for robust serialization.

*📁 src/dl_techniques/models/fftnet/components.py:139*

#### `build(self, input_shape)`
**Module:** `models.fftnet.components`

Create the learnable bias parameter.

*📁 src/dl_techniques/models/fftnet/components.py:294*

#### `build(self, input_shape)`
**Module:** `models.fftnet.components`

Create the complex kernel weight.

*📁 src/dl_techniques/models/fftnet/components.py:475*

#### `build(self, input_shape)`
**Module:** `models.fftnet.components`

Build sub-layers explicitly for robust serialization.

*📁 src/dl_techniques/models/fftnet/components.py:677*

#### `build(self, input_shape)`
**Module:** `models.fftnet.components`

Build sub-layers explicitly for robust serialization.

*📁 src/dl_techniques/models/fftnet/components.py:859*

#### `build(self, input_shape)`
**Module:** `models.fftnet.components`

Build sub-layers and create optional spectral memory.

*📁 src/dl_techniques/models/fftnet/components.py:1008*

#### `build(self, input_shape)`
**Module:** `models.gemma.components`

Build the layer and all its sub-layers. CRITICAL: This method explicitly builds each sub-layer to ensure proper weight variable creation before weight restoration during serialization.

*📁 src/dl_techniques/models/gemma/components.py:162*

#### `build(self, input_shape)`
**Module:** `models.mini_vec2vec.model`

Create the transformation matrix W.

*📁 src/dl_techniques/models/mini_vec2vec/model.py:118*

#### `build(self, input_shape)`
**Module:** `models.dino.dino_v1`

Build the DINO head layers.

*📁 src/dl_techniques/models/dino/dino_v1.py:179*

#### `build(self, input_shape)`
**Module:** `models.dino.dino_v2`

Build the transformer block with all sub-components.

*📁 src/dl_techniques/models/dino/dino_v2.py:253*

#### `build(self, input_shape)`
**Module:** `models.vit_siglip.model`

Build the model and all its sub-layers.

*📁 src/dl_techniques/models/vit_siglip/model.py:456*

#### `build(self, input_shape)`
**Module:** `models.prism.model`

Build all model components.

*📁 src/dl_techniques/models/prism/model.py:342*

#### `build(self, input_shape)`
**Module:** `models.masked_language_model.clm`

Builds the model and initializes the output head/weight tying.

*📁 src/dl_techniques/models/masked_language_model/clm.py:120*

#### `build(self, input_shape)`
**Module:** `models.tiny_recursive_model.model`

Build the model and its inner layer.

*📁 src/dl_techniques/models/tiny_recursive_model/model.py:181*

#### `build(self, input_shape)`
**Module:** `models.tiny_recursive_model.components`

Build all constituent TransformerLayer instances.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:204*

#### `build(self, input_shape)`
**Module:** `models.tiny_recursive_model.components`

Build sub-layers and create initial state weights.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:465*

#### `build(self, input_shape)`
**Module:** `models.sam.prompt_encoder`

Creates the random projection matrix.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:143*

#### `build(self, input_shape)`
**Module:** `models.sam.prompt_encoder`

Builds all sub-layers.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:380*

#### `build(self, input_shape)`
**Module:** `models.sam.image_encoder`

Builds the sub-layer.

*📁 src/dl_techniques/models/sam/image_encoder.py:175*

#### `build(self, input_shape)`
**Module:** `models.sam.image_encoder`

Creates the layer's weights and builds its sub-layers.

*📁 src/dl_techniques/models/sam/image_encoder.py:301*

#### `build(self, input_shape)`
**Module:** `models.sam.image_encoder`

Builds all sub-layers.

*📁 src/dl_techniques/models/sam/image_encoder.py:583*

#### `build(self, input_shape)`
**Module:** `models.sam.image_encoder`

Creates the model's own weights.

*📁 src/dl_techniques/models/sam/image_encoder.py:853*

#### `build(self, input_shape)`
**Module:** `models.sam.mask_decoder`

Builds all sub-layers.

*📁 src/dl_techniques/models/sam/mask_decoder.py:302*

#### `build(self, input_shape)`
**Module:** `models.sam.transformer`

Builds all sub-layers.

*📁 src/dl_techniques/models/sam/transformer.py:283*

#### `build(self, input_shape)`
**Module:** `models.sam.transformer`

Builds all sub-layers.

*📁 src/dl_techniques/models/sam/transformer.py:601*

#### `build(self, input_shape)`
**Module:** `models.masked_autoencoder.patch_masking`

*📁 src/dl_techniques/models/masked_autoencoder/patch_masking.py:38*

#### `build(self, input_shape)`
**Module:** `models.masked_autoencoder.conv_decoder`

Build decoder sub-layers explicitly using the input shape.

*📁 src/dl_techniques/models/masked_autoencoder/conv_decoder.py:140*

#### `build(self, input_shape)`
**Module:** `models.som.model`

Build the model by initializing the SOM layer.

*📁 src/dl_techniques/models/som/model.py:301*

#### `build(self, input_shape)`
**Module:** `models.clip.model`

Create the model's own weights.

*📁 src/dl_techniques/models/clip/model.py:450*

#### `build(self, input_shape)`
**Module:** `models.tree_transformer.model`

Creates the non-trainable positional encoding matrix.

*📁 src/dl_techniques/models/tree_transformer/model.py:159*

#### `build(self, input_shape)`
**Module:** `models.tree_transformer.model`

Builds the sub-layers, which is critical for serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:277*

#### `build(self, input_shape)`
**Module:** `models.tree_transformer.model`

Builds all sub-layers for robust serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:454*

#### `build(self, input_shape)`
**Module:** `models.tree_transformer.model`

Builds all sub-layers explicitly for robust serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:639*

#### `build(self, input_shape)`
**Module:** `models.yolo12.feature_extractor`

Build the feature extractor layers.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:109*

#### `build(self, input_shape)`
**Module:** `models.pft_sr.model`

Build model layers.

*📁 src/dl_techniques/models/pft_sr/model.py:132*

#### `build(self, input_shape)`
**Module:** `models.pw_fnet.model`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/models/pw_fnet/model.py:279*

#### `build(self, input_shape)`
**Module:** `models.pw_fnet.model`

Build the internal convolution layer.

*📁 src/dl_techniques/models/pw_fnet/model.py:490*

#### `build(self, input_shape)`
**Module:** `models.pw_fnet.model`

Build the internal transposed convolution layer.

*📁 src/dl_techniques/models/pw_fnet/model.py:577*

#### `build(self, input_shape)`
**Module:** `models.convunext.model`

Build layer by creating weights for sub-layers.

*📁 src/dl_techniques/models/convunext/model.py:137*

#### `build(self, input_shape)`
**Module:** `models.convunext.model`

Build model by creating weights for all sub-layers.

*📁 src/dl_techniques/models/convunext/model.py:746*

#### `build(self, input_shape)`
**Module:** `models.capsnet.model`

Build the model layers based on input shape.

*📁 src/dl_techniques/models/capsnet/model.py:174*

#### `build(self, input_shape)`
**Module:** `models.ntm.model`

Build the model layers with explicit shapes.

*📁 src/dl_techniques/models/ntm/model.py:151*

#### `build(self, input_shape)`
**Module:** `models.ntm.model_multitask`

Build the model and its sub-layers.

*📁 src/dl_techniques/models/ntm/model_multitask.py:71*

#### `build(self, input_shape)`
**Module:** `models.darkir.model`

Build sub-layers for proper serialization.

*📁 src/dl_techniques/models/darkir/model.py:313*

#### `build(self, input_shape)`
**Module:** `models.darkir.model`

Build the convolution layer.

*📁 src/dl_techniques/models/darkir/model.py:507*

#### `build(self, input_shape)`
**Module:** `models.darkir.model`

Build all sub-layers for proper serialization.

*📁 src/dl_techniques/models/darkir/model.py:771*

#### `build(self, input_shape)`
**Module:** `models.darkir.model`

Build all sub-layers for proper serialization.

*📁 src/dl_techniques/models/darkir/model.py:1147*

#### `build(self, input_shape)`
**Module:** `models.mothnet.model`

Create and build all sub-layers.

*📁 src/dl_techniques/models/mothnet/model.py:292*

#### `build(self, input_shape)`
**Module:** `models.nbeats.nbeatsx`

*📁 src/dl_techniques/models/nbeats/nbeatsx.py:141*

#### `build(self, input_shape)`
**Module:** `models.nbeats.nbeats`

Build the model and all its sub-layers.

*📁 src/dl_techniques/models/nbeats/nbeats.py:358*

#### `build(self, input_shape)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Build hash embedding tables.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:219*

#### `build(self, input_shape)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Build all embedding sublayers.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:354*

#### `build(self, input_shape)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Build all reasoning components.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:558*

#### `build(self, input_shape)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Build the complete ReasoningByteBERT model.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:850*

#### `build(self, input_shape)`
**Module:** `models.modern_bert.components`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/models/modern_bert/components.py:156*

#### `build(self, input_shape)`
**Module:** `models.modern_bert.components`

Build the layer and all its sub-layers for robust serialization.

*📁 src/dl_techniques/models/modern_bert/components.py:335*

#### `build(self, input_shape)`
**Module:** `models.vit_hmlp.model`

Build the model and all its sub-layers.

*📁 src/dl_techniques/models/vit_hmlp/model.py:491*

#### `build(self, input_shape)`
**Module:** `models.fastvlm.components`

Build the attention block and extract spatial dimensions.

*📁 src/dl_techniques/models/fastvlm/components.py:179*

#### `build(self, input_shape)`
**Module:** `models.qwen.qwen3_embeddings`

Build all sub-layers with proper shapes.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:231*

#### `build(self, input_shape)`
**Module:** `models.qwen.qwen3_embeddings`

Build all sub-layers with proper shapes.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:475*

#### `build(self, input_shape)`
**Module:** `models.qwen.qwen3_mega`

Build all sub-layers explicitly.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:149*

#### `build(self, input_shape)`
**Module:** `models.qwen.components`

Build all sub-layers for robust serialization.

*📁 src/dl_techniques/models/qwen/components.py:282*

#### `build(self, input_shape)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Build the model and its sub-components.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:249*

#### `build(self, input_shape)`
**Module:** `models.mobile_clip.components`

Build the layer and all its sub-layers.

*📁 src/dl_techniques/models/mobile_clip/components.py:113*

#### `build(self, input_shape)`
**Module:** `models.mobile_clip.components`

Explicitly build sub-layers to ensure deterministic serialization.

*📁 src/dl_techniques/models/mobile_clip/components.py:260*

#### `build(self, input_shape)`
**Module:** `models.mobile_clip.components`

Create weights and build all sub-layers.

*📁 src/dl_techniques/models/mobile_clip/components.py:430*

#### `build(self, input_shape)`
**Module:** `models.vit.model`

Build the model and all its sub-layers.

*📁 src/dl_techniques/models/vit/model.py:401*

#### `build(self, input_shape)`
**Module:** `models.squeezenet.squeezenet_v2`

Build the Simplified Fire module by building all sub-layers.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:151*

#### `build(self, input_shape)`
**Module:** `models.squeezenet.squeezenet_v1`

Build the Fire module by building all sub-layers.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:162*

#### `build_from_config(self, config)`
**Module:** `models.depth_anything.components`

Build layer from configuration.

*📁 src/dl_techniques/models/depth_anything/components.py:252*

#### `build_from_config(self, config)`
**Module:** `models.mdn.model`

Build the model from a build configuration.

*📁 src/dl_techniques/models/mdn/model.py:682*

#### `build_from_config(self, config)`
**Module:** `models.yolo12.feature_extractor`

Build model from configuration.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:351*

#### `call(self, inputs, training)`
**Module:** `models.nano_vlm.model`

Forward pass through NanoVLM.

*📁 src/dl_techniques/models/nano_vlm/model.py:467*

#### `call(self, inputs, training)`
**Module:** `models.mobilenet.mobilenet_v1`

Forward pass of the MobileNetV1 model.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:213*

#### `call(self, x, training)`
**Module:** `models.mobilenet.mobilenet_v4`

Forward pass of the MobileNetV4 model.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:260*

#### `call(self, inputs, training)`
**Module:** `models.mobilenet.mobilenet_v2`

Forward pass of the MobileNetV2 model.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v2.py:264*

#### `call(self, x, training)`
**Module:** `models.mobilenet.mobilenet_v3`

Forward pass through the model.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:289*

#### `call(self, inputs, training)`
**Module:** `models.depth_anything.model`

Forward pass through the model.

*📁 src/dl_techniques/models/depth_anything/model.py:329*

#### `call(self, inputs, training)`
**Module:** `models.depth_anything.components`

Forward pass through decoder.

*📁 src/dl_techniques/models/depth_anything/components.py:174*

#### `call(self, y_true, y_pred)`
**Module:** `models.nano_vlm_world_model.train`

Compute DSM loss.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:67*

#### `call(self, y_true, y_pred)`
**Module:** `models.nano_vlm_world_model.train`

Compute combined VLM loss.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:133*

#### `call(self, timesteps)`
**Module:** `models.nano_vlm_world_model.denoisers`

Compute sinusoidal timestep embeddings.

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:58*

#### `call(self, noisy_data, condition, timesteps, training)`
**Module:** `models.nano_vlm_world_model.denoisers`

Denoise data conditioned on context and timestep.

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:185*

#### `call(self, noisy_vision, text_features, timesteps, training)`
**Module:** `models.nano_vlm_world_model.denoisers`

Denoise vision features conditioned on text.

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:308*

#### `call(self, noisy_text, vision_features, timesteps, training)`
**Module:** `models.nano_vlm_world_model.denoisers`

Denoise text embeddings conditioned on vision.

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:381*

#### `call(self, noisy_vision, noisy_text, timesteps, training)`
**Module:** `models.nano_vlm_world_model.denoisers`

Jointly denoise vision and text features.

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:496*

#### `call(self, inputs, training)`
**Module:** `models.nano_vlm_world_model.model`

Forward pass for training.

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:191*

#### `call(self, inputs, training)`
**Module:** `models.tabm.model`

Forward pass through the TabM model implementing ensemble computation.

*📁 src/dl_techniques/models/tabm/model.py:565*

#### `call(self, inputs)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Inject conditioning features.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:181*

#### `call(self, inputs)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Inject discrete conditioning.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:299*

#### `call(self, inputs, training)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Forward pass.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:110*

#### `call(self, inputs, training)`
**Module:** `models.mdn.model`

Forward pass of the model.

*📁 src/dl_techniques/models/mdn/model.py:335*

#### `call(self, inputs, training)`
**Module:** `models.adaptive_ema.model`

Compute EMA, slope, and trading signals with potentially learnable thresholds.

*📁 src/dl_techniques/models/adaptive_ema/model.py:238*

#### `call(self, hidden_states, training)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:239*

#### `call(self, hidden_states, residual, training)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:364*

#### `call(self, inputs, training)`
**Module:** `models.mamba.mamba_v2`

*📁 src/dl_techniques/models/mamba/mamba_v2.py:105*

#### `call(self, inputs, training)`
**Module:** `models.mamba.mamba_v1`

Forward pass through the Mamba model.

*📁 src/dl_techniques/models/mamba/mamba_v1.py:353*

#### `call(self, hidden_states, training)`
**Module:** `models.mamba.components`

Forward pass through the Mamba layer.

*📁 src/dl_techniques/models/mamba/components.py:443*

#### `call(self, hidden_states, residual, training)`
**Module:** `models.mamba.components`

Forward pass through the residual block.

*📁 src/dl_techniques/models/mamba/components.py:673*

#### `call(self, inputs, training)`
**Module:** `models.byte_latent_transformer.model`

Forward pass of the BLT model.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:321*

#### `call(self, inputs, training)`
**Module:** `models.jepa.encoder`

Apply patch embedding to inputs.

*📁 src/dl_techniques/models/jepa/encoder.py:94*

#### `call(self, inputs, training, return_all_tokens)`
**Module:** `models.jepa.encoder`

Forward pass through JEPA encoder.

*📁 src/dl_techniques/models/jepa/encoder.py:276*

#### `call(self, context_tokens, mask_positions, training)`
**Module:** `models.jepa.encoder`

Predict masked token representations.

*📁 src/dl_techniques/models/jepa/encoder.py:450*

#### `call(self, inputs, training)`
**Module:** `models.convnext.convnext_v2`

Defines the forward pass of the model.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:361*

#### `call(self, inputs, training)`
**Module:** `models.convnext.convnext_v1`

Defines the forward pass of the model.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:344*

#### `call(self, src, mask, query_embed, pos_embed, training)`
**Module:** `models.detr.model`

Forward pass through encoder and decoder.

*📁 src/dl_techniques/models/detr/model.py:195*

#### `call(self, tgt, memory, query_pos, pos_embed, training)`
**Module:** `models.detr.model`

Forward pass through decoder layer.

*📁 src/dl_techniques/models/detr/model.py:378*

#### `call(self, inputs, training)`
**Module:** `models.detr.model`

Forward pass through DETR model.

*📁 src/dl_techniques/models/detr/model.py:552*

#### `call(self, inputs, attention_mask, token_type_ids, position_ids, training)`
**Module:** `models.fnet.model`

Forward pass of the FNet foundation model.

*📁 src/dl_techniques/models/fnet/model.py:384*

#### `call(self, inputs, training)`
**Module:** `models.latent_gmm_registration.model`

Forward pass of the model.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:159*

#### `call(self, inputs, training)`
**Module:** `models.tirex.model`

Forward pass through the TiRex model.

*📁 src/dl_techniques/models/tirex/model.py:284*

#### `call(self, inputs, training)`
**Module:** `models.tirex.model_extended`

Forward pass with Query Token appending.

*📁 src/dl_techniques/models/tirex/model_extended.py:147*

#### `call(self, inputs, training)`
**Module:** `models.fftnet.model`

Forward pass implementing adaptive spectral filtering.

*📁 src/dl_techniques/models/fftnet/model.py:171*

#### `call(self, inputs, training)`
**Module:** `models.fftnet.model`

Forward pass through the FFTNet block.

*📁 src/dl_techniques/models/fftnet/model.py:305*

#### `call(self, inputs, training)`
**Module:** `models.fftnet.model`

Forward pass of the FFTNet foundation model.

*📁 src/dl_techniques/models/fftnet/model.py:573*

#### `call(self, inputs)`
**Module:** `models.fftnet.components`

Apply mean pooling across sequence dimension.

*📁 src/dl_techniques/models/fftnet/components.py:76*

#### `call(self, inputs)`
**Module:** `models.fftnet.components`

Apply attention-based pooling.

*📁 src/dl_techniques/models/fftnet/components.py:153*

#### `call(self, inputs)`
**Module:** `models.fftnet.components`

Apply DCT-based pooling.

*📁 src/dl_techniques/models/fftnet/components.py:221*

#### `call(self, inputs)`
**Module:** `models.fftnet.components`

Apply complex modReLU activation.

*📁 src/dl_techniques/models/fftnet/components.py:311*

#### `call(self, inputs)`
**Module:** `models.fftnet.components`

Apply complex interpolation.

*📁 src/dl_techniques/models/fftnet/components.py:395*

#### `call(self, inputs)`
**Module:** `models.fftnet.components`

Apply complex 1D convolution with circular padding.

*📁 src/dl_techniques/models/fftnet/components.py:493*

#### `call(self, inputs, training)`
**Module:** `models.fftnet.components`

Forward pass through the Spectre head.

*📁 src/dl_techniques/models/fftnet/components.py:706*

#### `call(self, inputs, training)`
**Module:** `models.fftnet.components`

Forward pass through multi-head Spectre layer.

*📁 src/dl_techniques/models/fftnet/components.py:873*

#### `call(self, inputs, training)`
**Module:** `models.fftnet.components`

Forward pass through the Spectre block.

*📁 src/dl_techniques/models/fftnet/components.py:1035*

#### `call(self, inputs, attention_mask, training)`
**Module:** `models.gemma.gemma3`

Forward pass of the Gemma3 model.

*📁 src/dl_techniques/models/gemma/gemma3.py:340*

#### `call(self, inputs, attention_mask, training)`
**Module:** `models.gemma.components`

Forward pass through the transformer block.

*📁 src/dl_techniques/models/gemma/components.py:198*

#### `call(self, inputs, training)`
**Module:** `models.mini_vec2vec.model`

Apply the learned linear transformation W to input embeddings.

*📁 src/dl_techniques/models/mini_vec2vec/model.py:146*

#### `call(self, x, training)`
**Module:** `models.scunet.model`

Forward pass of the SCUNet model.

*📁 src/dl_techniques/models/scunet/model.py:293*

#### `call(self, inputs, training)`
**Module:** `models.dino.dino_v1`

Forward pass of the DINO head.

*📁 src/dl_techniques/models/dino/dino_v1.py:281*

#### `call(self, inputs, training)`
**Module:** `models.dino.dino_v2`

Forward pass of the transformer block.

*📁 src/dl_techniques/models/dino/dino_v2.py:283*

#### `call(self, inputs, training)`
**Module:** `models.vit_siglip.model`

Forward pass through the SigLIP Vision Transformer.

*📁 src/dl_techniques/models/vit_siglip/model.py:508*

#### `call(self, inputs, training)`
**Module:** `models.prism.model`

Generate forecasts from context window.

*📁 src/dl_techniques/models/prism/model.py:377*

#### `call(self, inputs, training)`
**Module:** `models.masked_language_model.mlm`

Forward pass for prediction.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:265*

#### `call(self, inputs, training)`
**Module:** `models.masked_language_model.clm`

Forward pass for prediction/generation.

*📁 src/dl_techniques/models/masked_language_model/clm.py:168*

#### `call(self, carry, batch, training)`
**Module:** `models.tiny_recursive_model.model`

Perform one step of the ACT reasoning process.

*📁 src/dl_techniques/models/tiny_recursive_model/model.py:236*

#### `call(self, hidden_states, input_injection, training)`
**Module:** `models.tiny_recursive_model.components`

Perform the forward pass through the stack of TransformerLayers.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:221*

#### `call(self, carry, data, training)`
**Module:** `models.tiny_recursive_model.components`

Perform a single inner reasoning step.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:506*

#### `call(self, size)`
**Module:** `models.sam.prompt_encoder`

Generate a grid of positional encodings for a given spatial size.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:177*

#### `call(self, points, boxes, masks, training)`
**Module:** `models.sam.prompt_encoder`

Encode prompts into sparse and dense embeddings.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:511*

#### `call(self, x)`
**Module:** `models.sam.image_encoder`

Forward pass for patch embedding.

*📁 src/dl_techniques/models/sam/image_encoder.py:189*

#### `call(self, x)`
**Module:** `models.sam.image_encoder`

Forward pass for attention.

*📁 src/dl_techniques/models/sam/image_encoder.py:329*

#### `call(self, x)`
**Module:** `models.sam.image_encoder`

Forward pass for the ViT block.

*📁 src/dl_techniques/models/sam/image_encoder.py:601*

#### `call(self, x, training)`
**Module:** `models.sam.image_encoder`

Forward pass for the image encoder.

*📁 src/dl_techniques/models/sam/image_encoder.py:872*

#### `call(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, training)`
**Module:** `models.sam.mask_decoder`

Forward pass for mask prediction.

*📁 src/dl_techniques/models/sam/mask_decoder.py:337*

#### `call(self, inputs, training, multimask_output)`
**Module:** `models.sam.model`

Forward pass through the SAM model.

*📁 src/dl_techniques/models/sam/model.py:282*

#### `call(self, queries, keys, query_pe, key_pe, training)`
**Module:** `models.sam.transformer`

Forward pass through the two-way attention block.

*📁 src/dl_techniques/models/sam/transformer.py:325*

#### `call(self, image_embedding, image_pe, point_embedding, training)`
**Module:** `models.sam.transformer`

Forward pass through the two-way transformer.

*📁 src/dl_techniques/models/sam/transformer.py:626*

#### `call(self, inputs, training)`
**Module:** `models.masked_autoencoder.mae`

Forward pass: Mask -> Encode -> Decode.

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:151*

#### `call(self, inputs, training)`
**Module:** `models.masked_autoencoder.patch_masking`

*📁 src/dl_techniques/models/masked_autoencoder/patch_masking.py:84*

#### `call(self, inputs, training)`
**Module:** `models.masked_autoencoder.conv_decoder`

Decode features to reconstruct image.

*📁 src/dl_techniques/models/masked_autoencoder/conv_decoder.py:170*

#### `call(self, inputs, training)`
**Module:** `models.som.model`

Forward pass computing Best Matching Units and quantization errors.

*📁 src/dl_techniques/models/som/model.py:320*

#### `call(self, inputs, training)`
**Module:** `models.hierarchical_reasoning_model.model`

Forward pass through the model.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:555*

#### `call(self, inputs, training)`
**Module:** `models.clip.model`

Forward pass of the CLIP model.

*📁 src/dl_techniques/models/clip/model.py:615*

#### `call(self, x, training)`
**Module:** `models.tree_transformer.model`

Adds positional encodings to the input tensor.

*📁 src/dl_techniques/models/tree_transformer/model.py:179*

#### `call(self, inputs, training)`
**Module:** `models.tree_transformer.model`

Computes group attention probabilities.

*📁 src/dl_techniques/models/tree_transformer/model.py:285*

#### `call(self, inputs, training)`
**Module:** `models.tree_transformer.model`

Forward pass for tree-modulated multi-head attention.

*📁 src/dl_techniques/models/tree_transformer/model.py:470*

#### `call(self, inputs, training)`
**Module:** `models.tree_transformer.model`

Forward pass for the Tree Transformer block.

*📁 src/dl_techniques/models/tree_transformer/model.py:656*

#### `call(self, inputs, training)`
**Module:** `models.tree_transformer.model`

Forward pass of the TreeTransformer model.

*📁 src/dl_techniques/models/tree_transformer/model.py:963*

#### `call(self, inputs, training)`
**Module:** `models.yolo12.feature_extractor`

Forward pass through feature extractor.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:258*

#### `call(self, inputs, training)`
**Module:** `models.shgcn.model`

Forward pass through all sHGCN layers.

*📁 src/dl_techniques/models/shgcn/model.py:200*

#### `call(self, inputs, training)`
**Module:** `models.shgcn.model`

Forward pass for node classification.

*📁 src/dl_techniques/models/shgcn/model.py:362*

#### `call(self, inputs, training)`
**Module:** `models.shgcn.model`

Forward pass for link prediction.

*📁 src/dl_techniques/models/shgcn/model.py:517*

#### `call(self, input_ids, position_ids, training)`
**Module:** `models.distilbert.model`

Forward pass of the embeddings layer.

*📁 src/dl_techniques/models/distilbert/model.py:217*

#### `call(self, inputs, attention_mask, position_ids, training)`
**Module:** `models.distilbert.model`

Forward pass of the DistilBERT foundation model.

*📁 src/dl_techniques/models/distilbert/model.py:635*

#### `call(self, inputs, training)`
**Module:** `models.pft_sr.model`

Forward pass of PFT-SR.

*📁 src/dl_techniques/models/pft_sr/model.py:316*

#### `call(self, inputs, training)`
**Module:** `models.pw_fnet.model`

Forward pass through the PW-FNet block.

*📁 src/dl_techniques/models/pw_fnet/model.py:370*

#### `call(self, inputs)`
**Module:** `models.pw_fnet.model`

Apply downsampling to input tensor.

*📁 src/dl_techniques/models/pw_fnet/model.py:496*

#### `call(self, inputs)`
**Module:** `models.pw_fnet.model`

Apply upsampling to input tensor.

*📁 src/dl_techniques/models/pw_fnet/model.py:583*

#### `call(self, inputs, training)`
**Module:** `models.pw_fnet.model`

Forward pass through the PW-FNet model.

*📁 src/dl_techniques/models/pw_fnet/model.py:881*

#### `call(self, inputs, training)`
**Module:** `models.convunext.model`

Forward pass of the stem layer.

*📁 src/dl_techniques/models/convunext/model.py:154*

#### `call(self, inputs, training)`
**Module:** `models.convunext.model`

Forward pass of ConvUNext model.

*📁 src/dl_techniques/models/convunext/model.py:916*

#### `call(self, inputs, training)`
**Module:** `models.relgt.model`

Forward pass through the complete RELGT model.

*📁 src/dl_techniques/models/relgt/model.py:194*

#### `call(self, inputs, training)`
**Module:** `models.vq_vae.model`

Forward pass through VQ-VAE: encode, quantize, decode.

*📁 src/dl_techniques/models/vq_vae/model.py:281*

#### `call(self, inputs, training)`
**Module:** `models.resnet.model`

Forward pass of the model.

*📁 src/dl_techniques/models/resnet/model.py:369*

#### `call(self, inputs, attention_mask, token_type_ids, position_ids, training)`
**Module:** `models.bert.bert`

Forward pass of the BERT foundation model.

*📁 src/dl_techniques/models/bert/bert.py:464*

#### `call(self, inputs, training, mask)`
**Module:** `models.capsnet.model`

Forward pass through the capsule network.

*📁 src/dl_techniques/models/capsnet/model.py:298*

#### `call(self, inputs, training, mask)`
**Module:** `models.xlstm.model`

Forward pass through the xLSTM model.

*📁 src/dl_techniques/models/xlstm/model.py:289*

#### `call(self, inputs, training)`
**Module:** `models.power_mlp.model`

Forward pass for the PowerMLP model.

*📁 src/dl_techniques/models/power_mlp/model.py:490*

#### `call(self, inputs, training)`
**Module:** `models.ntm.model`

Forward pass.

*📁 src/dl_techniques/models/ntm/model.py:173*

#### `call(self, inputs)`
**Module:** `models.ntm.model_multitask`

Forward pass.

*📁 src/dl_techniques/models/ntm/model_multitask.py:100*

#### `call(self, x)`
**Module:** `models.darkir.model`

Apply gating operation by splitting and multiplying.

*📁 src/dl_techniques/models/darkir/model.py:151*

#### `call(self, x)`
**Module:** `models.darkir.model`

Forward pass: FFT -> Process Magnitude -> IFFT.

*📁 src/dl_techniques/models/darkir/model.py:331*

#### `call(self, x)`
**Module:** `models.darkir.model`

Apply dilated depthwise convolution.

*📁 src/dl_techniques/models/darkir/model.py:520*

#### `call(self, inputs, training)`
**Module:** `models.darkir.model`

Forward pass with dual residual paths.

*📁 src/dl_techniques/models/darkir/model.py:827*

#### `call(self, inputs, training)`
**Module:** `models.darkir.model`

Forward pass with dual residual paths.

*📁 src/dl_techniques/models/darkir/model.py:1208*

#### `call(self, inputs, training)`
**Module:** `models.mothnet.model`

Forward pass through MothNet (AL → MB → Readout).

*📁 src/dl_techniques/models/mothnet/model.py:343*

#### `call(self, inputs, training, return_samples)`
**Module:** `models.deepar.model`

Forward pass through DeepAR.

*📁 src/dl_techniques/models/deepar/model.py:244*

#### `call(self, inputs, training)`
**Module:** `models.accunet.model`

Forward pass computation.

*📁 src/dl_techniques/models/accunet/model.py:368*

#### `call(self, inputs, training)`
**Module:** `models.cbam.model`

Forward pass through the model.

*📁 src/dl_techniques/models/cbam/model.py:254*

#### `call(self, inputs, training)`
**Module:** `models.nbeats.nbeatsx`

Args: inputs: Dictionary containing: - 'target_history': (B, Backcast, 1) - 'exog_history': (B, Backcast, ExogDim) - 'exog_forecast': (B, Forecast, ExogDim)

*📁 src/dl_techniques/models/nbeats/nbeatsx.py:152*

#### `call(self, inputs, training)`
**Module:** `models.nbeats.nbeats`

Forward pass through the N-BEATS network.

*📁 src/dl_techniques/models/nbeats/nbeats.py:394*

#### `call(self, inputs)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Compute hash n-gram embeddings for byte sequence.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:237*

#### `call(self, token_ids, puzzle_ids, training)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create combined embeddings from tokens and puzzle context.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:446*

#### `call(self, carry, embeddings, byte_tokens, training)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Perform hierarchical reasoning on byte-level representations.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:717*

#### `call(self, inputs, training, return_dict)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Forward pass through ReasoningByteBERT.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:899*

#### `call(self, inputs, attention_mask, token_type_ids, training)`
**Module:** `models.modern_bert.modern_bert`

Forward pass of the ModernBERT foundation model.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:383*

#### `call(self, inputs, attention_mask, training)`
**Module:** `models.modern_bert.modern_bert_blt`

Forward pass of the ModernBertBLT model.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:236*

#### `call(self, inputs)`
**Module:** `models.modern_bert.components`

Sum the embeddings from all specified n-gram sizes.

*📁 src/dl_techniques/models/modern_bert/components.py:193*

#### `call(self, input_ids, position_ids, training)`
**Module:** `models.modern_bert.components`

*📁 src/dl_techniques/models/modern_bert/components.py:358*

#### `call(self, inputs, training)`
**Module:** `models.vit_hmlp.model`

Forward pass through the Vision Transformer with hMLP stem.

*📁 src/dl_techniques/models/vit_hmlp/model.py:544*

#### `call(self, inputs, training)`
**Module:** `models.fastvlm.model`

Forward pass through FastVLM.

*📁 src/dl_techniques/models/fastvlm/model.py:412*

#### `call(self, inputs, training)`
**Module:** `models.fastvlm.components`

Forward pass through attention block.

*📁 src/dl_techniques/models/fastvlm/components.py:209*

#### `call(self, inputs, training)`
**Module:** `models.qwen.qwen3_embeddings`

Forward pass to compute embeddings.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:257*

#### `call(self, inputs, training)`
**Module:** `models.qwen.qwen3_embeddings`

Forward pass to compute relevance scores.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:503*

#### `call(self, inputs, training)`
**Module:** `models.qwen.qwen3_embeddings`

The base forward pass expects tokenized inputs.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:683*

#### `call(self, inputs, training)`
**Module:** `models.qwen.qwen3_embeddings`

The base forward pass expects tokenized, pre-formatted inputs.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:809*

#### `call(self, inputs, attention_mask, training, return_dict)`
**Module:** `models.qwen.qwen3`

Forward pass of the Qwen3 model.

*📁 src/dl_techniques/models/qwen/qwen3.py:359*

#### `call(self, inputs, training)`
**Module:** `models.qwen.qwen3_mega`

Integrate memory and entity information with hidden states.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:173*

#### `call(self, inputs, attention_mask, entity_adjacency, training, return_dict)`
**Module:** `models.qwen.qwen3_mega`

Forward pass of Qwen3-MEGA model.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:626*

#### `call(self, inputs, attention_mask, training, return_dict)`
**Module:** `models.qwen.qwen3_next`

Forward pass of the Qwen3 Next model.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:319*

#### `call(self, inputs, attention_mask, training, return_dict, return_som_assignments)`
**Module:** `models.qwen.qwen3_som`

Forward pass of the Qwen3-SOM model.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:496*

#### `call(self, inputs, attention_mask, training)`
**Module:** `models.qwen.components`

Forward pass through the Qwen3Next block.

*📁 src/dl_techniques/models/qwen/components.py:313*

#### `call(self, inputs, training)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Forward pass for the MobileClip model.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:285*

#### `call(self, inputs, training)`
**Module:** `models.mobile_clip.components`

Forward pass through sub-layers.

*📁 src/dl_techniques/models/mobile_clip/components.py:130*

#### `call(self, inputs, training)`
**Module:** `models.mobile_clip.components`

Forward pass through the backbone and projection head.

*📁 src/dl_techniques/models/mobile_clip/components.py:273*

#### `call(self, text_tokens, training)`
**Module:** `models.mobile_clip.components`

Forward pass for text encoding.

*📁 src/dl_techniques/models/mobile_clip/components.py:456*

#### `call(self, inputs, training)`
**Module:** `models.vit.model`

Forward pass through the Vision Transformer.

*📁 src/dl_techniques/models/vit/model.py:452*

#### `call(self, inputs, training)`
**Module:** `models.squeezenet.squeezenet_v2`

Forward pass through the Simplified Fire module.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:164*

#### `call(self, inputs, training)`
**Module:** `models.squeezenet.squeezenet_v1`

Forward pass through the Fire module.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:181*

#### `compile(self, optimizer, loss, loss_weights)`
**Module:** `models.depth_anything.model`

Configure the model for training.

*📁 src/dl_techniques/models/depth_anything/model.py:366*

#### `compile(self, optimizer, metrics)`
**Module:** `models.mdn.model`

Configure the model for training.

*📁 src/dl_techniques/models/mdn/model.py:588*

#### `compute_gmm_params(points, gamma)`
**Module:** `models.latent_gmm_registration.model`

Compute GMM parameters from soft point-to-component assignments.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:404*

#### `compute_loss(self, x, y, y_pred, sample_weight)`
**Module:** `models.masked_language_model.mlm`

Computes MLM loss on masked positions.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:380*

#### `compute_loss(self, x, y, y_pred, sample_weight)`
**Module:** `models.masked_language_model.clm`

*📁 src/dl_techniques/models/masked_language_model/clm.py:268*

#### `compute_loss(self, x, y, y_pred, sample_weight)`
**Module:** `models.masked_autoencoder.mae`

Compute reconstruction loss only on masked patches.

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:183*

#### `compute_losses(self, tensors)`
**Module:** `models.ccnets.orchestrators`

Compute the three fundamental CCNet losses.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:139*

#### `compute_mean_cosine_similarity(XA_aligned, XB_true)`
**Module:** `models.mini_vec2vec.example_alignment`

Calculate mean cosine similarity between aligned pairs.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:100*

#### `compute_model_errors(self, losses, tensors)`
**Module:** `models.ccnets.orchestrators`

Compute model-specific error signals using the stable, additive protocol as specified in the canonical documentation (FOUNDATION.md). This replaces the unstable subtractive protocol.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:159*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.nano_vlm.model`

Compute output shape given input shape.

*📁 src/dl_techniques/models/nano_vlm/model.py:636*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.depth_anything.components`

Compute the output shape of the layer.

*📁 src/dl_techniques/models/depth_anything/components.py:206*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Compute output shape.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:117*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.mdn.model`

Compute the output shape of the model.

*📁 src/dl_techniques/models/mdn/model.py:753*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.jepa.encoder`

Compute output shape.

*📁 src/dl_techniques/models/jepa/encoder.py:120*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.jepa.encoder`

Compute output shape.

*📁 src/dl_techniques/models/jepa/encoder.py:324*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.jepa.encoder`

Compute output shape.

*📁 src/dl_techniques/models/jepa/encoder.py:490*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.convnext.convnext_v2`

Compute the output shape of the model.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:634*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.model`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fftnet/model.py:218*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.model`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fftnet/model.py:319*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Compute output shape.

*📁 src/dl_techniques/models/fftnet/components.py:88*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Compute output shape.

*📁 src/dl_techniques/models/fftnet/components.py:168*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Compute output shape.

*📁 src/dl_techniques/models/fftnet/components.py:242*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fftnet/components.py:333*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Compute output shape.

*📁 src/dl_techniques/models/fftnet/components.py:426*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fftnet/components.py:536*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Compute output shapes for both outputs.

*📁 src/dl_techniques/models/fftnet/components.py:767*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fftnet/components.py:898*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fftnet.components`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fftnet/components.py:1058*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.dino.dino_v2`

Compute the output shape of the layer.

*📁 src/dl_techniques/models/dino/dino_v2.py:315*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.vit_siglip.model`

Compute output shape.

*📁 src/dl_techniques/models/vit_siglip/model.py:621*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.prism.model`

Compute output shape based on configuration.

*📁 src/dl_techniques/models/prism/model.py:513*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.tiny_recursive_model.components`

Compute the output shape of the layer.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:247*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.sam.prompt_encoder`

Compute output shapes for sparse and dense embeddings.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:592*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.sam.image_encoder`

Compute output shape of the layer.

*📁 src/dl_techniques/models/sam/image_encoder.py:202*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.sam.image_encoder`

Compute output shape of the layer.

*📁 src/dl_techniques/models/sam/image_encoder.py:436*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.sam.image_encoder`

Compute output shape of the layer.

*📁 src/dl_techniques/models/sam/image_encoder.py:690*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.sam.mask_decoder`

Compute output shapes for masks and IoU predictions.

*📁 src/dl_techniques/models/sam/mask_decoder.py:479*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.sam.model`

Compute output shapes given input shapes.

*📁 src/dl_techniques/models/sam/model.py:610*

#### `compute_output_shape(self, query_shape, key_shape)`
**Module:** `models.sam.transformer`

Compute output shapes.

*📁 src/dl_techniques/models/sam/transformer.py:405*

#### `compute_output_shape(self, image_shape, point_shape)`
**Module:** `models.sam.transformer`

Compute output shapes.

*📁 src/dl_techniques/models/sam/transformer.py:686*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.masked_autoencoder.mae`

Compute output shapes for all model outputs.

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:130*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.tree_transformer.model`

Computes the output shape of the layer.

*📁 src/dl_techniques/models/tree_transformer/model.py:950*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.yolo12.feature_extractor`

Compute output shapes for the three feature maps.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:311*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.pw_fnet.model`

Compute output shape (same as input shape).

*📁 src/dl_techniques/models/pw_fnet/model.py:408*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.pw_fnet.model`

Compute output shape after downsampling.

*📁 src/dl_techniques/models/pw_fnet/model.py:508*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.pw_fnet.model`

Compute output shape after upsampling.

*📁 src/dl_techniques/models/pw_fnet/model.py:595*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.convunext.model`

Compute output shape for given input shape.

*📁 src/dl_techniques/models/convunext/model.py:178*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.convunext.model`

Compute output shape(s) for the model.

*📁 src/dl_techniques/models/convunext/model.py:836*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.power_mlp.model`

Compute the output shape of the model.

*📁 src/dl_techniques/models/power_mlp/model.py:533*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.ntm.model`

Compute output shape based on configuration.

*📁 src/dl_techniques/models/ntm/model.py:191*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.ntm.model_multitask`

Compute output shape based on input shapes.

*📁 src/dl_techniques/models/ntm/model_multitask.py:136*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.darkir.model`

Compute output shape (channels are halved).

*📁 src/dl_techniques/models/darkir/model.py:165*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.darkir.model`

Compute output shape (same as input).

*📁 src/dl_techniques/models/darkir/model.py:369*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.darkir.model`

Compute output shape.

*📁 src/dl_techniques/models/darkir/model.py:532*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.darkir.model`

Compute output shape (same as input).

*📁 src/dl_techniques/models/darkir/model.py:877*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.darkir.model`

Compute output shape (same as input).

*📁 src/dl_techniques/models/darkir/model.py:1264*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.nbeats.nbeats`

Compute output shapes for forecast and residual.

*📁 src/dl_techniques/models/nbeats/nbeats.py:530*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.vit_hmlp.model`

Compute output shape.

*📁 src/dl_techniques/models/vit_hmlp/model.py:600*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.fastvlm.components`

Output shape is identical to input shape.

*📁 src/dl_techniques/models/fastvlm/components.py:236*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.qwen.components`

Compute output shape - identical to input shape.

*📁 src/dl_techniques/models/qwen/components.py:379*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.mobile_clip.components`

Compute the output shape.

*📁 src/dl_techniques/models/mobile_clip/components.py:144*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.mobile_clip.components`

Compute the output shape.

*📁 src/dl_techniques/models/mobile_clip/components.py:487*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.vit.model`

Compute output shape.

*📁 src/dl_techniques/models/vit/model.py:511*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.squeezenet.squeezenet_v2`

Compute output shape of Simplified Fire module.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:178*

#### `compute_output_shape(self, input_shape)`
**Module:** `models.squeezenet.squeezenet_v1`

Compute output shape of Fire module.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:199*

#### `compute_output_spec(self, inputs, attention_mask, training)`
**Module:** `models.gemma.components`

Infers the output shape and dtype for the functional API.

*📁 src/dl_techniques/models/gemma/components.py:180*

#### `compute_reasoner_grads()`
**Module:** `models.ccnets.orchestrators`

*📁 src/dl_techniques/models/ccnets/orchestrators.py:251*

#### `compute_rigid_transform(mu_source, pi_source, mu_target, pi_target)`
**Module:** `models.latent_gmm_registration.model`

Compute optimal rigid transformation between GMM means.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:457*

#### `compute_scale(self, target, conditioning_length)`
**Module:** `models.deepar.model`

Compute scale factor for each time series.

*📁 src/dl_techniques/models/deepar/model.py:220*

#### `compute_score_field(self, vision_features, text_features, timestep)`
**Module:** `models.nano_vlm_world_model.model`

Compute the joint score field ∇ log p(image, text) at a point.

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:432*

#### `compute_top1_accuracy(XA_aligned, XB_true)`
**Module:** `models.mini_vec2vec.example_alignment`

Calculate Top-1 retrieval accuracy.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:75*

#### `compute_transformation_error(learned_W, ground_truth_W)`
**Module:** `models.mini_vec2vec.example_alignment`

Compute Frobenius norm error between learned and ground truth W.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:126*

#### `cond(t, h, ys)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:201*

#### `condition(t, h, ys)`
**Module:** `models.mamba.components`

Loop condition: continue while t < seq_len.

*📁 src/dl_techniques/models/mamba/components.py:395*

#### `counterfactual_generation(self, x_reference, y_target)`
**Module:** `models.ccnets.orchestrators`

Generate counterfactual observations.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:338*

#### `create_acc_unet(input_channels, num_classes, base_filters, mlfc_iterations, input_shape)`
**Module:** `models.accunet.model`

Create ACC-UNet model with functional API.

*📁 src/dl_techniques/models/accunet/model.py:461*

#### `create_acc_unet_binary(input_channels, input_shape, base_filters, mlfc_iterations)`
**Module:** `models.accunet.model`

Create ACC-UNet model for binary segmentation.

*📁 src/dl_techniques/models/accunet/model.py:524*

#### `create_acc_unet_multiclass(input_channels, num_classes, input_shape, base_filters, mlfc_iterations)`
**Module:** `models.accunet.model`

Create ACC-UNet model for multi-class segmentation.

*📁 src/dl_techniques/models/accunet/model.py:571*

#### `create_bert_with_head(bert_variant, task_config, pretrained, weights_dataset, cache_dir, bert_config_overrides, head_config_overrides)`
**Module:** `models.bert.bert`

Factory function to create a BERT model with a task-specific head.

*📁 src/dl_techniques/models/bert/bert.py:846*

#### `create_bfcnn_denoiser(input_shape, num_blocks, filters, initial_kernel_size, kernel_size, activation, final_activation, kernel_initializer, kernel_regularizer, model_name)`
**Module:** `models.bias_free_denoisers.bfcnn`

Create a bias-free CNN model for image denoising using ResNet architecture.

*📁 src/dl_techniques/models/bias_free_denoisers/bfcnn.py:59*

#### `create_bfcnn_variant(variant, input_shape)`
**Module:** `models.bias_free_denoisers.bfcnn`

Create a BFCNN model with a specific variant configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfcnn.py:179*

#### `create_bfunet_denoiser(input_shape, depth, initial_filters, filter_multiplier, blocks_per_level, kernel_size, initial_kernel_size, activation, final_activation, kernel_initializer, kernel_regularizer, use_residual_blocks, enable_deep_supervision, model_name)`
**Module:** `models.bias_free_denoisers.bfunet`

Create a bias-free U-Net model with optional deep supervision.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet.py:105*

#### `create_bfunet_variant(variant, input_shape, enable_deep_supervision, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.bias_free_denoisers.bfunet`

Create a bias-free U-Net model with a specific variant configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet.py:583*

#### `create_blt_model(variant, vocab_size, max_sequence_length, entropy_threshold)`
**Module:** `models.byte_latent_transformer.model`

Create a BLT model with the specified variant and configuration.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:688*

#### `create_capsnet(num_classes, input_shape, optimizer, learning_rate)`
**Module:** `models.capsnet.model`

Create and compile a CapsNet model.

*📁 src/dl_techniques/models/capsnet/model.py:572*

#### `create_cbam_net(variant, num_classes, input_shape, pretrained)`
**Module:** `models.cbam.model`

Convenience function to create a CBAMNet model.

*📁 src/dl_techniques/models/cbam/model.py:477*

#### `create_class_conditional_bfunet(image_shape, num_classes, enable_cfg_training)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Create bias-free U-Net for class-conditional image generation/denoising.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:845*

#### `create_clip_model()`
**Module:** `models.clip.model`

Convenience function to create a CLIP model.

*📁 src/dl_techniques/models/clip/model.py:772*

#### `create_clip_variant(variant)`
**Module:** `models.clip.model`

Convenience function to create a CLIP model from a predefined variant.

*📁 src/dl_techniques/models/clip/model.py:801*

#### `create_conditional_bfunet_denoiser(input_shape, num_classes, depth, initial_filters, filter_multiplier, blocks_per_level, kernel_size, initial_kernel_size, activation, final_activation, kernel_initializer, kernel_regularizer, use_residual_blocks, enable_deep_supervision, class_embedding_dim, class_injection_method, enable_cfg_training, model_name)`
**Module:** `models.bias_free_denoisers.bfunet_conditional`

Create a class-conditional bias-free U-Net model with optional deep supervision.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional.py:34*

#### `create_conditional_bfunet_variant(variant, input_shape, num_classes, enable_deep_supervision, enable_cfg_training)`
**Module:** `models.bias_free_denoisers.bfunet_conditional`

Create a conditional bias-free U-Net with a specific variant configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional.py:508*

#### `create_convnext_v1(variant, num_classes, input_shape, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.convnext.convnext_v1`

Convenience function to create ConvNeXt V1 models.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:686*

#### `create_convnext_v2(variant, num_classes, input_shape, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.convnext.convnext_v2`

Convenience function to create ConvNeXt V2 models.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:742*

#### `create_convunext_denoiser(input_shape, depth, initial_filters, filter_multiplier, blocks_per_level, convnext_version, stem_kernel_size, block_kernel_size, drop_path_rate, final_activation, kernel_initializer, kernel_regularizer, enable_deep_supervision, model_name)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Create a ConvUNext model using existing ConvNeXt V1/V2 blocks with bias-free configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:183*

#### `create_convunext_variant(variant, input_shape, enable_deep_supervision)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Create a ConvUNext model with a specific variant configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:576*

#### `create_convunext_variant(variant, input_shape, include_top, enable_deep_supervision, output_channels, use_bias)`
**Module:** `models.convunext.model`

Factory function to create ConvUNext model from variant name.

*📁 src/dl_techniques/models/convunext/model.py:1226*

#### `create_coshnet(variant, num_classes, input_shape)`
**Module:** `models.coshnet.model`

Convenience function to create CoShNet models with predefined configurations.

*📁 src/dl_techniques/models/coshnet/model.py:642*

#### `create_cyborg_features(mothnet, x_data)`
**Module:** `models.mothnet.model`

Create augmented 'cyborg' features by concatenating original features with MothNet outputs.

*📁 src/dl_techniques/models/mothnet/model.py:560*

#### `create_darkir_model(img_channels, width, middle_blk_num_enc, middle_blk_num_dec, enc_blk_nums, dec_blk_nums, dilations, extra_depth_wise, use_side_loss)`
**Module:** `models.darkir.model`

Create the DarkIR model for low-light image restoration.

*📁 src/dl_techniques/models/darkir/model.py:1295*

#### `create_dense_conditioning_encoder(input_layer, num_levels, base_filters, encoder_type, kernel_initializer, kernel_regularizer, activation, use_batch_norm)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Create multi-scale feature encoder for dense conditioning signals.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:41*

#### `create_depth_anything(encoder_type, input_shape, decoder_dims, output_channels, kernel_initializer, kernel_regularizer, loss_weights, cutmix_prob, color_jitter_strength, use_feature_alignment)`
**Module:** `models.depth_anything.model`

Create and build Depth Anything model instance.

*📁 src/dl_techniques/models/depth_anything/model.py:474*

#### `create_depth_estimation_bfunet(depth_shape, rgb_shape)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Create bias-free U-Net for monocular depth estimation.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:809*

#### `create_detr(num_classes, num_queries, backbone_name, backbone_trainable, hidden_dim, num_heads, num_encoder_layers, num_decoder_layers, ffn_dim, dropout, aux_loss, activation, normalization_type, ffn_type)`
**Module:** `models.detr.model`

Convenience factory to create a DETR model with a specified backbone.

*📁 src/dl_techniques/models/detr/model.py:645*

#### `create_dino_teacher_student_pair(variant, teacher_temp, student_temp, patch_size, input_shape, dino_out_dim)`
**Module:** `models.dino.dino_v1`

Create teacher-student pair for DINO self-supervised learning.

*📁 src/dl_techniques/models/dino/dino_v1.py:852*

#### `create_dino_v1(variant, num_classes, patch_size, input_shape, include_top, include_projection_head, dino_out_dim)`
**Module:** `models.dino.dino_v1`

Convenience function to create DINO Vision Transformer models.

*📁 src/dl_techniques/models/dino/dino_v1.py:794*

#### `create_dino_v2(variant, image_size, patch_size, num_register_tokens, init_values, stochastic_depth_rate, ffn_type, num_classes, include_top, input_shape, pretrained)`
**Module:** `models.dino.dino_v2`

Factory function to create DINOv2 model variants with sensible defaults.

*📁 src/dl_techniques/models/dino/dino_v2.py:1147*

#### `create_dino_v3(variant, image_size, num_classes, include_top, pretrained)`
**Module:** `models.dino.dino_v3`

A factory function to create DINOv3 models.

*📁 src/dl_techniques/models/dino/dino_v3.py:438*

#### `create_distilbert_with_head(distilbert_variant, task_config, pretrained, weights_dataset, cache_dir, distilbert_config_overrides, head_config_overrides)`
**Module:** `models.distilbert.model`

Factory function to create a DistilBERT model with a task-specific head.

*📁 src/dl_techniques/models/distilbert/model.py:1015*

#### `create_fast_reasoning_byte_bert()`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create fast ReasoningByteBERT configuration for quick inference.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1134*

#### `create_fftnet(variant, image_size, patch_size)`
**Module:** `models.fftnet.model`

Create FFTNet foundation model with preset configuration.

*📁 src/dl_techniques/models/fftnet/model.py:814*

#### `create_fftnet_classifier(variant, num_classes, image_size, patch_size)`
**Module:** `models.fftnet.model`

Convenience function to create FFTNet classification model.

*📁 src/dl_techniques/models/fftnet/model.py:851*

#### `create_fftnet_with_head(fftnet_variant, task_type, num_classes, image_size, patch_size, fftnet_config_overrides, head_config_overrides)`
**Module:** `models.fftnet.model`

Factory function to create a complete FFTNet model with a task-specific head.

*📁 src/dl_techniques/models/fftnet/model.py:693*

#### `create_fnet_with_head(fnet_variant, task_config, pretrained, weights_dataset, cache_dir, fnet_config_overrides, head_config_overrides, sequence_length)`
**Module:** `models.fnet.model`

Factory function to create an FNet model with a task-specific head.

*📁 src/dl_techniques/models/fnet/model.py:751*

#### `create_fractal_net(variant, num_classes, input_shape, optimizer, learning_rate, loss, metrics)`
**Module:** `models.fractalnet.model`

Convenience function to create and compile FractalNet models.

*📁 src/dl_techniques/models/fractalnet/model.py:436*

#### `create_gemma3(config_or_variant, task_type)`
**Module:** `models.gemma.gemma3`

High-level factory to create Gemma3 models for common tasks.

*📁 src/dl_techniques/models/gemma/gemma3.py:503*

#### `create_gemma3_classification(config, num_labels, pooling_strategy, classifier_dropout)`
**Module:** `models.gemma.gemma3`

Creates a Gemma3 model for sequence classification tasks.

*📁 src/dl_techniques/models/gemma/gemma3.py:430*

#### `create_gemma3_generation(config)`
**Module:** `models.gemma.gemma3`

Creates a Gemma3 model for text generation tasks.

*📁 src/dl_techniques/models/gemma/gemma3.py:407*

#### `create_hierarchical_reasoning_model(vocab_size, seq_len, embed_dim, num_puzzle_identifiers, variant, optimizer, learning_rate)`
**Module:** `models.hierarchical_reasoning_model.model`

Create and optionally compile a Hierarchical Reasoning Model.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:860*

#### `create_inference_model_from_training_model(training_model)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Create a single-output inference model from a multi-output training model.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:684*

#### `create_inference_model_from_training_model(training_model)`
**Module:** `models.bias_free_denoisers.bfunet`

Create a single-output inference model from a multi-output training model.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet.py:780*

#### `create_inference_model_from_training_model(training_model, disable_deep_supervision)`
**Module:** `models.convunext.model`

Create inference model from training model.

*📁 src/dl_techniques/models/convunext/model.py:1296*

#### `create_inference_model_from_training_model(training_model)`
**Module:** `models.resnet.model`

Create a single-output inference model from a multi-output training model.

*📁 src/dl_techniques/models/resnet/model.py:731*

#### `create_inputs_with_masking(batch_size, image_size, patch_size, mask_ratio)`
**Module:** `models.vit_hmlp.model`

Create masked input images and corresponding mask for self-supervised learning.

*📁 src/dl_techniques/models/vit_hmlp/model.py:898*

#### `create_kan_model(variant, input_features, output_features, output_activation, pretrained, weights_dataset, weights_input_features, cache_dir)`
**Module:** `models.kan.model`

Helper to create a standard KAN model configuration.

*📁 src/dl_techniques/models/kan/model.py:522*

#### `create_mae_model(encoder, patch_size, mask_ratio, decoder_dims, input_shape)`
**Module:** `models.masked_autoencoder.utils`

Convenience factory to create an MAE model.

*📁 src/dl_techniques/models/masked_autoencoder/utils.py:13*

#### `create_mamba_with_head(mamba_variant, task_config, pretrained, mamba_config_overrides, head_config_overrides)`
**Module:** `models.mamba.mamba_v1`

Factory function to create a Mamba model with a task-specific head.

*📁 src/dl_techniques/models/mamba/mamba_v1.py:546*

#### `create_mini_vec2vec_aligner(embedding_dim)`
**Module:** `models.mini_vec2vec.model`

Factory function to create a MiniVec2VecAligner model.

*📁 src/dl_techniques/models/mini_vec2vec/model.py:530*

#### `create_mlm_training_model(encoder, vocab_size, mask_token_id, special_token_ids, mlm_config, optimizer_config)`
**Module:** `models.masked_language_model.utils`

Factory function to create a fully configured MLM training model.

*📁 src/dl_techniques/models/masked_language_model/utils.py:99*

#### `create_mobile_clip_model(variant, pretrained)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Convenience function to create Mobile CLIP models.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:367*

#### `create_mobilenetv1(variant, num_classes, input_shape, pretrained)`
**Module:** `models.mobilenet.mobilenet_v1`

Convenience function to create MobileNetV1 models.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:341*

#### `create_mobilenetv2(variant, num_classes, input_shape, pretrained)`
**Module:** `models.mobilenet.mobilenet_v2`

Convenience function to create MobileNetV2 models.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v2.py:348*

#### `create_mobilenetv3(variant, num_classes, input_shape, width_multiplier, pretrained)`
**Module:** `models.mobilenet.mobilenet_v3`

Convenience function to create MobileNetV3 models.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:392*

#### `create_mobilenetv4(variant, num_classes, input_shape, width_multiplier, pretrained)`
**Module:** `models.mobilenet.mobilenet_v4`

Convenience function to create MobileNetV4 models.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:497*

#### `create_modern_bert_blt_with_head(bert_variant, task_config, bert_config_overrides, head_config_overrides)`
**Module:** `models.modern_bert.modern_bert_blt`

Factory function to create a complete ModernBertBLT model with a task-specific head.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:326*

#### `create_modern_bert_with_head(bert_variant, task_config, pretrained, weights_dataset, cache_dir, bert_config_overrides, head_config_overrides)`
**Module:** `models.modern_bert.modern_bert`

Factory function to create a ModernBERT model with a task-specific head.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:598*

#### `create_modern_nanovlm(vocab_size, embed_dim)`
**Module:** `models.nano_vlm.model`

Create NanoVLM with modern architectural components.

*📁 src/dl_techniques/models/nano_vlm/model.py:777*

#### `create_nanovlm(variant, vocab_size, fusion_strategy, text_component_type)`
**Module:** `models.nano_vlm.model`

Factory function to create NanoVLM with predefined configurations.

*📁 src/dl_techniques/models/nano_vlm/model.py:689*

#### `create_nbeats_model(backcast_length, forecast_length, stack_types, nb_blocks_per_stack, thetas_dim, hidden_layer_units, activation, use_normalization, dropout_rate, reconstruction_weight, input_dim, output_dim)`
**Module:** `models.nbeats.nbeats`

Create an N-BEATS model instance with optimal defaults.

*📁 src/dl_techniques/models/nbeats/nbeats.py:619*

#### `create_nbeatsx_model(backcast_length, forecast_length, exogenous_dim, stack_types)`
**Module:** `models.nbeats.nbeatsx`

Factory for NBEATSx Model.

*📁 src/dl_techniques/models/nbeats/nbeatsx.py:240*

#### `create_ntm_variant(variant, input_shape, output_dim, return_sequences)`
**Module:** `models.ntm.model`

Factory function to create an NTM model variant.

*📁 src/dl_techniques/models/ntm/model.py:274*

#### `create_pft_sr(scale, variant)`
**Module:** `models.pft_sr.model`

Factory function to create PFT-SR models with predefined configurations.

*📁 src/dl_techniques/models/pft_sr/model.py:376*

#### `create_power_mlp(hidden_units, k, optimizer, learning_rate, loss, metrics)`
**Module:** `models.power_mlp.model`

Create and compile a PowerMLP model.

*📁 src/dl_techniques/models/power_mlp/model.py:783*

#### `create_power_mlp_binary_classifier(hidden_units, k, optimizer, learning_rate)`
**Module:** `models.power_mlp.model`

Create and compile a PowerMLP model for binary classification.

*📁 src/dl_techniques/models/power_mlp/model.py:899*

#### `create_power_mlp_regressor(hidden_units, k, optimizer, learning_rate)`
**Module:** `models.power_mlp.model`

Create and compile a PowerMLP model for regression tasks.

*📁 src/dl_techniques/models/power_mlp/model.py:853*

#### `create_qwen3(config_or_variant, task_type)`
**Module:** `models.qwen.qwen3`

High-level factory to create Qwen3 models for common tasks.

*📁 src/dl_techniques/models/qwen/qwen3.py:633*

#### `create_qwen3_classification(config, num_labels, pooling_strategy, classifier_dropout)`
**Module:** `models.qwen.qwen3`

Create a Qwen3 model for sequence classification tasks.

*📁 src/dl_techniques/models/qwen/qwen3.py:546*

#### `create_qwen3_generation(config)`
**Module:** `models.qwen.qwen3`

Create a Qwen3 model optimized for text generation tasks.

*📁 src/dl_techniques/models/qwen/qwen3.py:506*

#### `create_qwen3_mega(variant, memory_size, entity_graph_size)`
**Module:** `models.qwen.qwen3_mega`

Factory function to create Qwen3-MEGA models with preset configurations.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:770*

#### `create_qwen3_next(config_or_variant, task_type)`
**Module:** `models.qwen.qwen3_next`

High-level factory to create Qwen3 Next models for common tasks.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:597*

#### `create_qwen3_next_classification(config, num_labels, pooling_strategy, classifier_dropout)`
**Module:** `models.qwen.qwen3_next`

Create a Qwen3 Next model for sequence classification tasks.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:508*

#### `create_qwen3_next_generation(config)`
**Module:** `models.qwen.qwen3_next`

Create a Qwen3 Next model optimized for text generation tasks.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:468*

#### `create_qwen3som(config_or_variant, task_type)`
**Module:** `models.qwen.qwen3_som`

High-level factory to create Qwen3-SOM models for common tasks.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:858*

#### `create_qwen3som_classification(config, num_labels, pooling_strategy, classifier_dropout)`
**Module:** `models.qwen.qwen3_som`

Create a Qwen3-SOM model for sequence classification tasks.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:781*

#### `create_qwen3som_generation(config)`
**Module:** `models.qwen.qwen3_som`

Create a Qwen3-SOM model optimized for text generation tasks.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:738*

#### `create_reasoning_byte_bert_base()`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create base ReasoningByteBERT configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1074*

#### `create_reasoning_byte_bert_for_reasoning_tasks(config, num_puzzle_types)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create ReasoningByteBERT optimized for reasoning tasks.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1107*

#### `create_reasoning_byte_bert_large()`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create large ReasoningByteBERT configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1090*

#### `create_relgt_model(output_dim, problem_type, model_size)`
**Module:** `models.relgt.model`

Factory function to create RELGT models with predefined configurations.

*📁 src/dl_techniques/models/relgt/model.py:260*

#### `create_resnet(variant, num_classes, input_shape, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.resnet.model`

Convenience function to create ResNet models.

*📁 src/dl_techniques/models/resnet/model.py:771*

#### `create_score_based_nanovlm(variant, mode, vocab_size)`
**Module:** `models.nano_vlm_world_model.model`

Create a score-based nanoVLM with predefined configurations.

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:548*

#### `create_semantic_depth_bfunet(depth_shape, rgb_shape, num_classes)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Create bias-free U-Net for semantic-aware depth estimation.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:885*

#### `create_siglip_vision_transformer(input_shape, num_classes, scale, patch_size, include_top, pooling, dropout_rate, attention_dropout_rate, pos_dropout_rate, kernel_initializer, kernel_regularizer, bias_initializer, bias_regularizer, normalization_type, normalization_position, ffn_type, activation)`
**Module:** `models.vit_siglip.model`

Create a SigLIP Vision Transformer model with specified configuration.

*📁 src/dl_techniques/models/vit_siglip/model.py:741*

#### `create_squeezenet_v1(variant, num_classes, input_shape, weights)`
**Module:** `models.squeezenet.squeezenet_v1`

Convenience function to create SqueezeNet V1 models.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:615*

#### `create_squeezenodule_net_v2(variant, num_classes, input_shape, weights)`
**Module:** `models.squeezenet.squeezenet_v2`

Convenience function to create SqueezeNodule-Net V2 models.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:645*

#### `create_swin_transformer(variant, num_classes, input_shape, pretrained)`
**Module:** `models.swin_transformer.model`

Factory function to create Swin Transformer models with validation.

*📁 src/dl_techniques/models/swin_transformer/model.py:627*

#### `create_tabm_ensemble(n_num_features, cat_cardinalities, n_classes, k, hidden_dims)`
**Module:** `models.tabm.model`

Create a TabM model with full efficient ensemble.

*📁 src/dl_techniques/models/tabm/model.py:932*

#### `create_tabm_for_dataset(X_train, y_train, categorical_indices, categorical_cardinalities, arch_type, k, hidden_dims)`
**Module:** `models.tabm.model`

Create a TabM model automatically configured for a specific dataset.

*📁 src/dl_techniques/models/tabm/model.py:1048*

#### `create_tabm_mini(n_num_features, cat_cardinalities, n_classes, k, hidden_dims)`
**Module:** `models.tabm.model`

Create a TabM-mini model with minimal ensemble adapter.

*📁 src/dl_techniques/models/tabm/model.py:965*

#### `create_tabm_model(n_num_features, cat_cardinalities, n_classes, hidden_dims, arch_type, k, activation, dropout_rate, use_bias, share_training_batches, kernel_initializer, bias_initializer)`
**Module:** `models.tabm.model`

Create a TabM model with specified configuration.

*📁 src/dl_techniques/models/tabm/model.py:823*

#### `create_tabm_plain(n_num_features, cat_cardinalities, n_classes, hidden_dims)`
**Module:** `models.tabm.model`

Create a plain MLP baseline without ensembling.

*📁 src/dl_techniques/models/tabm/model.py:901*

#### `create_tirex_by_variant(variant, input_length, prediction_length, quantile_levels)`
**Module:** `models.tirex.model`

Convenience function to create TiRex models from predefined variants.

*📁 src/dl_techniques/models/tirex/model.py:613*

#### `create_tirex_extended(variant, input_length, prediction_length, quantile_levels)`
**Module:** `models.tirex.model_extended`

Convenience function to create TiRex models from predefined variants.

*📁 src/dl_techniques/models/tirex/model_extended.py:249*

#### `create_tirex_model(input_length, prediction_length, patch_size, embed_dim, num_blocks, num_heads, quantile_levels, block_types)`
**Module:** `models.tirex.model`

Create a TiRex model with specified configuration.

*📁 src/dl_techniques/models/tirex/model.py:562*

#### `create_tree_transformer_with_head(tree_transformer_variant, task_config, pretrained, weights_dataset, cache_dir, encoder_config_overrides, head_config_overrides)`
**Module:** `models.tree_transformer.model`

Factory function to create a Tree Transformer model with a task-specific head.

*📁 src/dl_techniques/models/tree_transformer/model.py:1175*

#### `create_unified_conditional_bfunet(target_shape, dense_conditioning_shape, num_classes, depth, initial_filters, filter_multiplier, blocks_per_level, kernel_size, activation, final_activation, kernel_initializer, kernel_regularizer, use_residual_blocks, use_batch_norm, dense_conditioning_encoder_filters, dense_injection_method, class_embedding_dim, discrete_injection_method, enable_cfg_training, enable_deep_supervision, model_name)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Create unified conditional bias-free U-Net supporting multiple conditioning modalities.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:366*

#### `create_vae(input_shape, latent_dim, variant, optimizer, learning_rate)`
**Module:** `models.vae.model`

Convenience function to create and compile VAE models.

*📁 src/dl_techniques/models/vae/model.py:881*

#### `create_vae_from_config(config, optimizer, learning_rate)`
**Module:** `models.vae.model`

Create VAE from configuration dictionary.

*📁 src/dl_techniques/models/vae/model.py:956*

#### `create_vision_transformer(input_shape, num_classes, scale, patch_size, include_top, pooling, dropout_rate, attention_dropout_rate, pos_dropout_rate, kernel_initializer, kernel_regularizer, bias_initializer, bias_regularizer, normalization_type, normalization_position, ffn_type, activation)`
**Module:** `models.vit.model`

Create a Vision Transformer model with specified configuration.

*📁 src/dl_techniques/models/vit/model.py:630*

#### `create_vit_hmlp(input_shape, num_classes, scale, patch_size, include_top, pooling, dropout_rate, attention_dropout_rate, pos_dropout_rate, stem_norm_layer, kernel_initializer, kernel_regularizer, bias_initializer, bias_regularizer, normalization_type, normalization_position, ffn_type, activation, use_stochastic_depth, stochastic_depth_rate)`
**Module:** `models.vit_hmlp.model`

Create a Vision Transformer with Hierarchical MLP stem.

*📁 src/dl_techniques/models/vit_hmlp/model.py:729*

#### `create_yolov12_feature_extractor(input_shape, scale)`
**Module:** `models.yolo12.feature_extractor`

Create a YOLOv12 feature extractor with specified configuration.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:363*

#### `create_yolov12_multitask(num_detection_classes, num_segmentation_classes, num_classification_classes, num_classes, input_shape, scale, tasks)`
**Module:** `models.yolo12.multitask`

Create YOLOv12 multi-task model with specified tasks and class counts.

*📁 src/dl_techniques/models/yolo12/multitask.py:417*

#### `decode(self, latents)`
**Module:** `models.vq_vae.model`

Decode latent representations to reconstructed outputs.

*📁 src/dl_techniques/models/vq_vae/model.py:443*

#### `decode(self, z)`
**Module:** `models.vae.model`

Decode latent samples to reconstructions.

*📁 src/dl_techniques/models/vae/model.py:626*

#### `decode_from_indices(self, indices)`
**Module:** `models.vq_vae.model`

Decode discrete codebook indices to reconstructed outputs.

*📁 src/dl_techniques/models/vq_vae/model.py:479*

#### `decode_tokens(self, token_ids)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Decode byte tokens back to text.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1051*

#### `decode_tokens(self, token_ids)`
**Module:** `models.modern_bert.modern_bert_blt`

Decodes a tensor of byte token IDs back into a string.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:291*

#### `decode_tokens(self, token_ids)`
**Module:** `models.modern_bert.components`

Convenience method to decode token IDs from this layer.

*📁 src/dl_techniques/models/modern_bert/components.py:398*

#### `disentangle_causes(self, x_input)`
**Module:** `models.ccnets.orchestrators`

Disentangle the explicit and latent causes of an observation.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:367*

#### `empty_carry(self, batch_size)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create empty carry state for reasoning.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:685*

#### `encode(self, inputs)`
**Module:** `models.vq_vae.model`

Encode inputs to continuous latent representations.

*📁 src/dl_techniques/models/vq_vae/model.py:417*

#### `encode(self, inputs)`
**Module:** `models.vae.model`

Encode inputs to latent parameters.

*📁 src/dl_techniques/models/vae/model.py:612*

#### `encode_image(self, images, training)`
**Module:** `models.clip.model`

Encode images to the shared embedding space.

*📁 src/dl_techniques/models/clip/model.py:510*

#### `encode_image(self, image, normalize, training)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Encode images to embedding vectors.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:257*

#### `encode_text(self, text_ids, training)`
**Module:** `models.clip.model`

Encode text to the shared embedding space.

*📁 src/dl_techniques/models/clip/model.py:560*

#### `encode_text(self, text, max_length)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Encode text to byte tokens.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1034*

#### `encode_text(self, text, max_length, add_special_tokens)`
**Module:** `models.modern_bert.modern_bert_blt`

Encodes a string of text into byte token IDs.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:282*

#### `encode_text(self, text, max_length, add_special_tokens)`
**Module:** `models.modern_bert.components`

Convenience method to tokenize text for this layer.

*📁 src/dl_techniques/models/modern_bert/components.py:382*

#### `encode_text(self, text, normalize, training)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Encode text tokens to embedding vectors.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:271*

#### `encode_to_indices(self, inputs)`
**Module:** `models.vq_vae.model`

Encode inputs directly to discrete codebook indices.

*📁 src/dl_techniques/models/vq_vae/model.py:455*

#### `ensemble_predict(model, x_data, method)`
**Module:** `models.tabm.model`

Make predictions using ensemble model with aggregation.

*📁 src/dl_techniques/models/tabm/model.py:998*

#### `evaluate(self, x_input, y_truth)`
**Module:** `models.ccnets.orchestrators`

Evaluate the CCNet without training.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:317*

#### `evaluate_alignment(aligner, XA_eval, XB_eval, ground_truth_W, stage)`
**Module:** `models.mini_vec2vec.example_alignment`

Evaluate alignment quality with multiple metrics.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:148*

#### `example_training()`
**Module:** `models.nano_vlm_world_model.train`

Example of training a score-based VLM.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:440*

#### `extract_cls_features(backbone_output)`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:1052*

#### `extract_features(self, inputs, training)`
**Module:** `models.yolo12.multitask`

Extract shared features without applying task heads.

*📁 src/dl_techniques/models/yolo12/multitask.py:380*

#### `extract_features(self, inputs)`
**Module:** `models.mothnet.model`

Extract MothNet features (readout activations) for "cyborg" augmentation.

*📁 src/dl_techniques/models/mothnet/model.py:374*

#### `extract_features(self, inputs, training)`
**Module:** `models.fastvlm.model`

Extract intermediate feature maps from all stages.

*📁 src/dl_techniques/models/fastvlm/model.py:442*

#### `extract_mb_features(self, inputs)`
**Module:** `models.mothnet.model`

Extract Mushroom Body activations (sparse codes before readout).

*📁 src/dl_techniques/models/mothnet/model.py:393*

#### `fit_class_prototypes(self, x_train, y_train)`
**Module:** `models.som.model`

Learn class-to-grid mappings by finding representative BMU for each class.

*📁 src/dl_techniques/models/som/model.py:468*

#### `forward_pass(self, x_input, y_truth, training)`
**Module:** `models.ccnets.orchestrators`

Perform a complete forward pass with selective gradient blocking to enforce the causal credit assignment protocol.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:88*

#### `forward_pass(self, x_input, y_truth, training)`
**Module:** `models.ccnets.orchestrators`

Forward pass with sequential data handling and correct index conversion.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:429*

#### `forward_with_coords(self, coords_input, image_size)`
**Module:** `models.sam.prompt_encoder`

Encode explicit coordinates (e.g., point or box coordinates).

*📁 src/dl_techniques/models/sam/prompt_encoder.py:200*

#### `from_config(cls, config)`
**Module:** `models.coshnet.model`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/coshnet/model.py:593*

#### `from_config(cls, config)`
**Module:** `models.mobilenet.mobilenet_v1`

Create model from configuration.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:306*

#### `from_config(cls, config)`
**Module:** `models.mobilenet.mobilenet_v4`

Create model from configuration.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:452*

#### `from_config(cls, config)`
**Module:** `models.depth_anything.model`

Create model from configuration.

*📁 src/dl_techniques/models/depth_anything/model.py:461*

#### `from_config(cls, config)`
**Module:** `models.tabm.model`

Create model from configuration.

*📁 src/dl_techniques/models/tabm/model.py:743*

#### `from_config(cls, config)`
**Module:** `models.mdn.model`

Create a model from its configuration.

*📁 src/dl_techniques/models/mdn/model.py:695*

#### `from_config(cls, config)`
**Module:** `models.mamba.mamba_v1`

Create model instance from configuration.

*📁 src/dl_techniques/models/mamba/mamba_v1.py:509*

#### `from_config(cls, config)`
**Module:** `models.byte_latent_transformer.model`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:656*

#### `from_config(cls, config)`
**Module:** `models.fractalnet.model`

Create model from configuration.

*📁 src/dl_techniques/models/fractalnet/model.py:394*

#### `from_config(cls, config)`
**Module:** `models.convnext.convnext_v2`

Create model from configuration.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:699*

#### `from_config(cls, config)`
**Module:** `models.convnext.convnext_v1`

Create model from configuration.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:643*

#### `from_config(cls, config)`
**Module:** `models.detr.model`

Deserialize model from configuration.

*📁 src/dl_techniques/models/detr/model.py:631*

#### `from_config(cls, config)`
**Module:** `models.fnet.model`

Create a model instance from its configuration.

*📁 src/dl_techniques/models/fnet/model.py:707*

#### `from_config(cls, config)`
**Module:** `models.latent_gmm_registration.model`

Create model from configuration.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:392*

#### `from_config(cls, config)`
**Module:** `models.tirex.model`

Create model from configuration.

*📁 src/dl_techniques/models/tirex/model.py:553*

#### `from_config(cls, config)`
**Module:** `models.fftnet.model`

Create model from configuration.

*📁 src/dl_techniques/models/fftnet/model.py:672*

#### `from_config(cls, config)`
**Module:** `models.dino.dino_v3`

Creates a model from its configuration.

*📁 src/dl_techniques/models/dino/dino_v3.py:433*

#### `from_config(cls, config)`
**Module:** `models.dino.dino_v1`

Create model from configuration.

*📁 src/dl_techniques/models/dino/dino_v1.py:764*

#### `from_config(cls, config)`
**Module:** `models.prism.model`

Create model from configuration.

*📁 src/dl_techniques/models/prism/model.py:574*

#### `from_config(cls, config)`
**Module:** `models.masked_language_model.mlm`

Creates a model from its configuration.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:423*

#### `from_config(cls, config)`
**Module:** `models.masked_language_model.clm`

*📁 src/dl_techniques/models/masked_language_model/clm.py:302*

#### `from_config(cls, config)`
**Module:** `models.sam.mask_decoder`

Creates a MaskDecoder from its config.

*📁 src/dl_techniques/models/sam/mask_decoder.py:524*

#### `from_config(cls, config)`
**Module:** `models.sam.model`

Creates a SAM model from a configuration dictionary.

*📁 src/dl_techniques/models/sam/model.py:586*

#### `from_config(cls, config)`
**Module:** `models.masked_autoencoder.mae`

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:346*

#### `from_config(cls, config)`
**Module:** `models.som.model`

Create model instance from configuration dictionary.

*📁 src/dl_techniques/models/som/model.py:1227*

#### `from_config(cls, config)`
**Module:** `models.hierarchical_reasoning_model.model`

Create model from configuration.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:795*

#### `from_config(cls, config)`
**Module:** `models.clip.model`

Create model from configuration.

*📁 src/dl_techniques/models/clip/model.py:711*

#### `from_config(cls, config)`
**Module:** `models.kan.model`

*📁 src/dl_techniques/models/kan/model.py:515*

#### `from_config(cls, config)`
**Module:** `models.tree_transformer.model`

Creates a model instance from its configuration.

*📁 src/dl_techniques/models/tree_transformer/model.py:1148*

#### `from_config(cls, config)`
**Module:** `models.yolo12.feature_extractor`

Create model from configuration.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:357*

#### `from_config(cls, config)`
**Module:** `models.yolo12.multitask`

Create model from configuration.

*📁 src/dl_techniques/models/yolo12/multitask.py:321*

#### `from_config(cls, config)`
**Module:** `models.distilbert.model`

Create a model instance from its configuration.

*📁 src/dl_techniques/models/distilbert/model.py:963*

#### `from_config(cls, config)`
**Module:** `models.convunext.model`

Create layer from configuration dictionary.

*📁 src/dl_techniques/models/convunext/model.py:217*

#### `from_config(cls, config)`
**Module:** `models.convunext.model`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/convunext/model.py:1067*

#### `from_config(cls, config)`
**Module:** `models.relgt.model`

Create model from configuration.

*📁 src/dl_techniques/models/relgt/model.py:254*

#### `from_config(cls, config)`
**Module:** `models.vq_vae.model`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/vq_vae/model.py:528*

#### `from_config(cls, config)`
**Module:** `models.resnet.model`

Create model from configuration.

*📁 src/dl_techniques/models/resnet/model.py:673*

#### `from_config(cls, config)`
**Module:** `models.bert.bert`

Create a model instance from its configuration.

*📁 src/dl_techniques/models/bert/bert.py:798*

#### `from_config(cls, config)`
**Module:** `models.capsnet.model`

Create model from configuration.

*📁 src/dl_techniques/models/capsnet/model.py:508*

#### `from_config(cls, config)`
**Module:** `models.xlstm.model`

Create model from configuration.

*📁 src/dl_techniques/models/xlstm/model.py:359*

#### `from_config(cls, config)`
**Module:** `models.power_mlp.model`

Create PowerMLP model from configuration dictionary.

*📁 src/dl_techniques/models/power_mlp/model.py:578*

#### `from_config(cls, config)`
**Module:** `models.ntm.model`

Deserialize configuration.

*📁 src/dl_techniques/models/ntm/model.py:216*

#### `from_config(cls, config)`
**Module:** `models.ntm.model_multitask`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/ntm/model_multitask.py:163*

#### `from_config(cls, config)`
**Module:** `models.mothnet.model`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/mothnet/model.py:541*

#### `from_config(cls, config)`
**Module:** `models.swin_transformer.model`

Create model from configuration dictionary.

*📁 src/dl_techniques/models/swin_transformer/model.py:573*

#### `from_config(cls, config)`
**Module:** `models.accunet.model`

Create model from configuration.

*📁 src/dl_techniques/models/accunet/model.py:453*

#### `from_config(cls, config)`
**Module:** `models.cbam.model`

Create model instance from configuration dictionary.

*📁 src/dl_techniques/models/cbam/model.py:311*

#### `from_config(cls, config)`
**Module:** `models.nbeats.nbeats`

Create model instance from configuration.

*📁 src/dl_techniques/models/nbeats/nbeats.py:584*

#### `from_config(cls, config)`
**Module:** `models.vae.model`

Create model from configuration.

*📁 src/dl_techniques/models/vae/model.py:838*

#### `from_config(cls, config)`
**Module:** `models.modern_bert.modern_bert`

Create a model instance from its configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:575*

#### `from_config(cls, config)`
**Module:** `models.fastvlm.model`

Create model from configuration.

*📁 src/dl_techniques/models/fastvlm/model.py:561*

#### `from_config(cls, config)`
**Module:** `models.qwen.qwen3`

Create model from configuration.

*📁 src/dl_techniques/models/qwen/qwen3.py:468*

#### `from_config(cls, config)`
**Module:** `models.qwen.qwen3_mega`

Create model from configuration.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:761*

#### `from_config(cls, config)`
**Module:** `models.qwen.qwen3_next`

Create model from configuration.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:426*

#### `from_config(cls, config)`
**Module:** `models.qwen.qwen3_som`

Create model from configuration.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:701*

#### `from_config(cls, config)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Create model from configuration.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:344*

#### `from_config(cls, config)`
**Module:** `models.squeezenet.squeezenet_v2`

Create model from configuration.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:587*

#### `from_config(cls, config)`
**Module:** `models.squeezenet.squeezenet_v1`

Create model from configuration.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:569*

#### `from_dict(cls, config_dict)`
**Module:** `models.jepa.config`

Create configuration from dictionary.

*📁 src/dl_techniques/models/jepa/config.py:266*

#### `from_dict(cls, config_dict)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Create configuration from dictionary.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:184*

#### `from_layer_sizes(cls, layer_sizes, grid_size, spline_order, activation, final_activation)`
**Module:** `models.kan.model`

Create a KAN by defining a list of node counts per layer.

*📁 src/dl_techniques/models/kan/model.py:409*

#### `from_preset(cls, preset)`
**Module:** `models.jepa.config`

Create configuration from preset.

*📁 src/dl_techniques/models/jepa/config.py:145*

#### `from_preset(cls, preset, context_len, forecast_len, num_features)`
**Module:** `models.prism.model`

Create model from a predefined preset ('tiny', 'small', 'base', 'large').

*📁 src/dl_techniques/models/prism/model.py:524*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.coshnet.model`

Create a CoShNet model from a predefined variant.

*📁 src/dl_techniques/models/coshnet/model.py:507*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.mobilenet.mobilenet_v1`

Create a MobileNetV1 model from a predefined variant.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:242*

#### `from_variant(cls, variant, num_classes, input_shape, width_multiplier)`
**Module:** `models.mobilenet.mobilenet_v4`

Create a MobileNetV4 model from a predefined variant.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:368*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.mobilenet.mobilenet_v2`

Create a MobileNetV2 model from a predefined variant string.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v2.py:286*

#### `from_variant(cls, variant, num_classes, input_shape, width_multiplier)`
**Module:** `models.mobilenet.mobilenet_v3`

Create a MobileNetV3 model from a predefined variant.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:316*

#### `from_variant(cls, variant, n_num_features, cat_cardinalities, n_classes)`
**Module:** `models.tabm.model`

Create a TabM model from a predefined variant.

*📁 src/dl_techniques/models/tabm/model.py:658*

#### `from_variant(cls, variant, vocab_size)`
**Module:** `models.mamba.mamba_v2`

*📁 src/dl_techniques/models/mamba/mamba_v2.py:131*

#### `from_variant(cls, variant, vocab_size, pretrained)`
**Module:** `models.mamba.mamba_v1`

Create a Mamba model from a predefined variant.

*📁 src/dl_techniques/models/mamba/mamba_v1.py:400*

#### `from_variant(cls, variant, vocab_size, max_sequence_length, entropy_threshold)`
**Module:** `models.byte_latent_transformer.model`

Create a BLT model from a predefined variant configuration.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:573*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.fractalnet.model`

Create a FractalNet model from a predefined variant.

*📁 src/dl_techniques/models/fractalnet/model.py:316*

#### `from_variant(cls, variant, num_classes, input_shape, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.convnext.convnext_v2`

Create a ConvNeXt V2 model from a predefined variant.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:502*

#### `from_variant(cls, variant, num_classes, input_shape, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.convnext.convnext_v1`

Create a ConvNeXt model from a predefined variant.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:485*

#### `from_variant(cls, variant, pretrained, weights_dataset, cache_dir)`
**Module:** `models.fnet.model`

Create an FNet model from a predefined variant.

*📁 src/dl_techniques/models/fnet/model.py:554*

#### `from_variant(cls, variant, prediction_length, quantile_levels)`
**Module:** `models.tirex.model`

Create a TiRex model from a predefined variant.

*📁 src/dl_techniques/models/tirex/model.py:486*

#### `from_variant(cls, variant)`
**Module:** `models.fftnet.model`

Create an FFTNet model from a predefined variant.

*📁 src/dl_techniques/models/fftnet/model.py:624*

#### `from_variant(cls, variant)`
**Module:** `models.gemma.gemma3`

Create a Gemma3 model from a predefined variant.

*📁 src/dl_techniques/models/gemma/gemma3.py:368*

#### `from_variant(cls, variant, image_size, num_classes, include_top)`
**Module:** `models.dino.dino_v3`

Creates a DINOv3 model from a predefined variant.

*📁 src/dl_techniques/models/dino/dino_v3.py:370*

#### `from_variant(cls, variant, num_classes, patch_size, input_shape)`
**Module:** `models.dino.dino_v1`

Create a DINO model from a predefined variant.

*📁 src/dl_techniques/models/dino/dino_v1.py:674*

#### `from_variant(cls, variant, image_size, patch_size, num_register_tokens, init_values, stochastic_depth_rate, input_shape)`
**Module:** `models.dino.dino_v2`

Create DINOv2 Vision Transformer from predefined variant.

*📁 src/dl_techniques/models/dino/dino_v2.py:819*

#### `from_variant(cls, variant, image_size, patch_size, num_classes, include_top, input_shape)`
**Module:** `models.dino.dino_v2`

Create DINOv2 model from predefined variant.

*📁 src/dl_techniques/models/dino/dino_v2.py:1077*

#### `from_variant(cls, variant)`
**Module:** `models.sam.model`

Create a SAM model from a predefined variant configuration.

*📁 src/dl_techniques/models/sam/model.py:435*

#### `from_variant(cls, variant, vocab_size, seq_len, num_puzzle_identifiers)`
**Module:** `models.hierarchical_reasoning_model.model`

Create a Hierarchical Reasoning Model from a predefined variant.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:708*

#### `from_variant(cls, variant)`
**Module:** `models.clip.model`

Create a CLIP model from a predefined variant.

*📁 src/dl_techniques/models/clip/model.py:724*

#### `from_variant(cls, variant, input_features, output_features, output_activation, pretrained, weights_dataset, weights_input_features, cache_dir, override_config)`
**Module:** `models.kan.model`

Factory method to create KAN models from standard presets.

*📁 src/dl_techniques/models/kan/model.py:285*

#### `from_variant(cls, variant, pretrained, weights_dataset, cache_dir)`
**Module:** `models.tree_transformer.model`

Creates a TreeTransformer model from a predefined variant.

*📁 src/dl_techniques/models/tree_transformer/model.py:1070*

#### `from_variant(cls, variant, pretrained, weights_dataset, cache_dir)`
**Module:** `models.distilbert.model`

Create a DistilBERT model from a predefined variant.

*📁 src/dl_techniques/models/distilbert/model.py:812*

#### `from_variant(cls, variant, input_shape, include_top, enable_deep_supervision, output_channels, use_bias)`
**Module:** `models.convunext.model`

Create model from predefined variant configuration.

*📁 src/dl_techniques/models/convunext/model.py:1131*

#### `from_variant(cls, variant, num_classes, input_shape, pretrained, weights_dataset, weights_input_shape, cache_dir)`
**Module:** `models.resnet.model`

Create a ResNet model from a predefined variant.

*📁 src/dl_techniques/models/resnet/model.py:535*

#### `from_variant(cls, variant, pretrained, weights_dataset, cache_dir)`
**Module:** `models.bert.bert`

Create a BERT model from a predefined variant.

*📁 src/dl_techniques/models/bert/bert.py:644*

#### `from_variant(cls, variant, num_classes, input_dim)`
**Module:** `models.power_mlp.model`

Create a PowerMLP model from a predefined variant.

*📁 src/dl_techniques/models/power_mlp/model.py:616*

#### `from_variant(cls, variant, input_shape, output_dim, return_sequences)`
**Module:** `models.ntm.model`

Create NTM model from a predefined variant.

*📁 src/dl_techniques/models/ntm/model.py:224*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.swin_transformer.model`

Create Swin Transformer from a predefined variant configuration.

*📁 src/dl_techniques/models/swin_transformer/model.py:508*

#### `from_variant(cls, variant, num_classes, input_shape, pretrained, weights_dataset)`
**Module:** `models.cbam.model`

Create a CBAMNet model from a predefined variant.

*📁 src/dl_techniques/models/cbam/model.py:334*

#### `from_variant(cls, variant, input_shape, latent_dim)`
**Module:** `models.vae.model`

Create a VAE model from a predefined variant.

*📁 src/dl_techniques/models/vae/model.py:549*

#### `from_variant(cls, variant, pretrained, weights_dataset, cache_dir)`
**Module:** `models.modern_bert.modern_bert`

Create a ModernBERT model from a predefined variant.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:503*

#### `from_variant(cls, variant)`
**Module:** `models.modern_bert.modern_bert_blt`

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:269*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.fastvlm.model`

Create a FastVLM model from a predefined variant.

*📁 src/dl_techniques/models/fastvlm/model.py:484*

#### `from_variant(cls, variant)`
**Module:** `models.qwen.qwen3`

Create a Qwen3 model from a predefined variant.

*📁 src/dl_techniques/models/qwen/qwen3.py:413*

#### `from_variant(cls, variant)`
**Module:** `models.qwen.qwen3_next`

Create a Qwen3 Next model from a predefined variant.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:373*

#### `from_variant(cls, variant)`
**Module:** `models.qwen.qwen3_som`

Create a Qwen3-SOM model from a predefined variant.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:630*

#### `from_variant(cls, variant)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Create a Mobile CLIP model from a predefined variant.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:312*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.squeezenet.squeezenet_v2`

Create a SqueezeNodule-Net model from a predefined variant.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:525*

#### `from_variant(cls, variant, num_classes, input_shape)`
**Module:** `models.squeezenet.squeezenet_v1`

Create a SqueezeNet model from a predefined variant.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:508*

#### `gaussian_loss(y_true, y_pred)`
**Module:** `models.deepar.model`

Gaussian negative log-likelihood loss.

*📁 src/dl_techniques/models/deepar/model.py:509*

#### `generate(self, images, prompt_tokens, max_length, temperature, top_k, eos_token_id)`
**Module:** `models.nano_vlm.model`

Generate text autoregressively given images and prompt.

*📁 src/dl_techniques/models/nano_vlm/model.py:538*

#### `generate(self, prompt, max_new_tokens, temperature, top_p, top_k, do_sample)`
**Module:** `models.byte_latent_transformer.model`

Generate text autoregressively using the BLT model.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:446*

#### `generate_audio_masks(self, batch_size)`
**Module:** `models.jepa.utilities`

Generate time-frequency aware masks for audio spectrograms.

*📁 src/dl_techniques/models/jepa/utilities.py:477*

#### `generate_from_image(self, vision_features, num_inference_steps, max_length, guidance_scale)`
**Module:** `models.nano_vlm_world_model.model`

Generate text from images via latent diffusion (Protocol 2).

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:363*

#### `generate_from_text(self, text_features, num_inference_steps, guidance_scale, generator)`
**Module:** `models.nano_vlm_world_model.model`

Generate images from text via reverse diffusion (Protocol 1).

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:281*

#### `generate_masks(self, batch_size)`
**Module:** `models.jepa.utilities`

Generate context and target masks for a batch.

*📁 src/dl_techniques/models/jepa/utilities.py:282*

#### `generate_synthetic_data(n_samples, n_eval, embed_dim, seed)`
**Module:** `models.mini_vec2vec.example_alignment`

Generate synthetic aligned embedding spaces for testing.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:23*

#### `generate_video_masks(self, batch_size)`
**Module:** `models.jepa.utilities`

Generate spatiotemporally consistent masks for video.

*📁 src/dl_techniques/models/jepa/utilities.py:409*

#### `get_architecture_summary(self)`
**Module:** `models.kan.model`

Returns a formatted string summarizing the KAN architecture details.

*📁 src/dl_techniques/models/kan/model.py:448*

#### `get_build_config(self)`
**Module:** `models.depth_anything.components`

Get build configuration for serialization.

*📁 src/dl_techniques/models/depth_anything/components.py:242*

#### `get_build_config(self)`
**Module:** `models.mdn.model`

Get the build configuration for serialization.

*📁 src/dl_techniques/models/mdn/model.py:668*

#### `get_build_config(self)`
**Module:** `models.yolo12.feature_extractor`

Get build configuration for serialization.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:345*

#### `get_class_counts(self)`
**Module:** `models.yolo12.multitask`

Get the number of classes for each task.

*📁 src/dl_techniques/models/yolo12/multitask.py:402*

#### `get_cls_token(self, features)`
**Module:** `models.vit_siglip.model`

Extract CLS token from vision_heads features for classification tasks.

*📁 src/dl_techniques/models/vit_siglip/model.py:574*

#### `get_config(self)`
**Module:** `models.coshnet.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/coshnet/model.py:561*

#### `get_config(self)`
**Module:** `models.nano_vlm.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/nano_vlm/model.py:661*

#### `get_config(self)`
**Module:** `models.mobilenet.mobilenet_v1`

Get model configuration for serialization.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:291*

#### `get_config(self)`
**Module:** `models.mobilenet.mobilenet_v4`

Get model configuration for serialization.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:426*

#### `get_config(self)`
**Module:** `models.mobilenet.mobilenet_v2`

Return configuration for serialization.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v2.py:307*

#### `get_config(self)`
**Module:** `models.mobilenet.mobilenet_v3`

Get model configuration for serialization.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:352*

#### `get_config(self)`
**Module:** `models.depth_anything.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/depth_anything/model.py:439*

#### `get_config(self)`
**Module:** `models.depth_anything.components`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/depth_anything/components.py:224*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.scheduler`

Get layer configuration.

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:359*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.train`

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:90*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.train`

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:179*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.denoisers`

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:88*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.denoisers`

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:250*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.denoisers`

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:329*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.denoisers`

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:402*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.denoisers`

*📁 src/dl_techniques/models/nano_vlm_world_model/denoisers.py:564*

#### `get_config(self)`
**Module:** `models.nano_vlm_world_model.model`

Get model configuration.

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:532*

#### `get_config(self)`
**Module:** `models.tabm.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/tabm/model.py:717*

#### `get_config(self)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Get layer configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:223*

#### `get_config(self)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Get layer configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:346*

#### `get_config(self)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Get layer configuration.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:121*

#### `get_config(self)`
**Module:** `models.mdn.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/mdn/model.py:644*

#### `get_config(self)`
**Module:** `models.adaptive_ema.model`

Return model configuration.

*📁 src/dl_techniques/models/adaptive_ema/model.py:334*

#### `get_config(self)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:303*

#### `get_config(self)`
**Module:** `models.mamba.components_v2`

*📁 src/dl_techniques/models/mamba/components_v2.py:375*

#### `get_config(self)`
**Module:** `models.mamba.mamba_v2`

*📁 src/dl_techniques/models/mamba/mamba_v2.py:142*

#### `get_config(self)`
**Module:** `models.mamba.mamba_v1`

Return model configuration for serialization.

*📁 src/dl_techniques/models/mamba/mamba_v1.py:487*

#### `get_config(self)`
**Module:** `models.mamba.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/mamba/components.py:523*

#### `get_config(self)`
**Module:** `models.mamba.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/mamba/components.py:704*

#### `get_config(self)`
**Module:** `models.byte_latent_transformer.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:629*

#### `get_config(self)`
**Module:** `models.jepa.encoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/jepa/encoder.py:125*

#### `get_config(self)`
**Module:** `models.jepa.encoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/jepa/encoder.py:331*

#### `get_config(self)`
**Module:** `models.jepa.encoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/jepa/encoder.py:500*

#### `get_config(self)`
**Module:** `models.fractalnet.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/fractalnet/model.py:367*

#### `get_config(self)`
**Module:** `models.convnext.convnext_v2`

Get model configuration for serialization.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:672*

#### `get_config(self)`
**Module:** `models.convnext.convnext_v1`

Get model configuration for serialization.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:616*

#### `get_config(self)`
**Module:** `models.detr.model`

*📁 src/dl_techniques/models/detr/model.py:243*

#### `get_config(self)`
**Module:** `models.detr.model`

*📁 src/dl_techniques/models/detr/model.py:452*

#### `get_config(self)`
**Module:** `models.detr.model`

*📁 src/dl_techniques/models/detr/model.py:618*

#### `get_config(self)`
**Module:** `models.fnet.model`

Return the model's configuration for serialization.

*📁 src/dl_techniques/models/fnet/model.py:677*

#### `get_config(self)`
**Module:** `models.latent_gmm_registration.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:375*

#### `get_config(self)`
**Module:** `models.tirex.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/tirex/model.py:533*

#### `get_config(self)`
**Module:** `models.tirex.model_extended`

Return config for serialization.

*📁 src/dl_techniques/models/tirex/model_extended.py:235*

#### `get_config(self)`
**Module:** `models.fftnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/model.py:222*

#### `get_config(self)`
**Module:** `models.fftnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/model.py:323*

#### `get_config(self)`
**Module:** `models.fftnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/model.py:655*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:92*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:172*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:246*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:337*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:430*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:540*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:774*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:902*

#### `get_config(self)`
**Module:** `models.fftnet.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/fftnet/components.py:1062*

#### `get_config(self)`
**Module:** `models.gemma.gemma3`

Return configuration for serialization.

*📁 src/dl_techniques/models/gemma/gemma3.py:384*

#### `get_config(self)`
**Module:** `models.gemma.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/gemma/components.py:252*

#### `get_config(self)`
**Module:** `models.mini_vec2vec.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/mini_vec2vec/model.py:516*

#### `get_config(self)`
**Module:** `models.scunet.model`

Returns the model configuration for serialization.

*📁 src/dl_techniques/models/scunet/model.py:338*

#### `get_config(self)`
**Module:** `models.dino.dino_v3`

Returns the model's configuration for serialization.

*📁 src/dl_techniques/models/dino/dino_v3.py:408*

#### `get_config(self)`
**Module:** `models.dino.dino_v1`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/dino/dino_v1.py:313*

#### `get_config(self)`
**Module:** `models.dino.dino_v1`

Get model configuration for serialization.

*📁 src/dl_techniques/models/dino/dino_v1.py:734*

#### `get_config(self)`
**Module:** `models.dino.dino_v2`

Get the configuration of the layer.

*📁 src/dl_techniques/models/dino/dino_v2.py:319*

#### `get_config(self)`
**Module:** `models.dino.dino_v2`

Get model configuration.

*📁 src/dl_techniques/models/dino/dino_v2.py:868*

#### `get_config(self)`
**Module:** `models.dino.dino_v2`

Get model configuration.

*📁 src/dl_techniques/models/dino/dino_v2.py:1122*

#### `get_config(self)`
**Module:** `models.vit_siglip.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/vit_siglip/model.py:644*

#### `get_config(self)`
**Module:** `models.prism.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/prism/model.py:548*

#### `get_config(self)`
**Module:** `models.masked_language_model.mlm`

Returns the configuration of the model for serialization.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:402*

#### `get_config(self)`
**Module:** `models.masked_language_model.clm`

*📁 src/dl_techniques/models/masked_language_model/clm.py:289*

#### `get_config(self)`
**Module:** `models.tiny_recursive_model.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/tiny_recursive_model/model.py:358*

#### `get_config(self)`
**Module:** `models.tiny_recursive_model.components`

Return the configuration for serialization.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:264*

#### `get_config(self)`
**Module:** `models.tiny_recursive_model.components`

Return the configuration for serialization.

*📁 src/dl_techniques/models/tiny_recursive_model/components.py:558*

#### `get_config(self)`
**Module:** `models.sam.prompt_encoder`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:223*

#### `get_config(self)`
**Module:** `models.sam.prompt_encoder`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:616*

#### `get_config(self)`
**Module:** `models.sam.image_encoder`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/image_encoder.py:217*

#### `get_config(self)`
**Module:** `models.sam.image_encoder`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/image_encoder.py:448*

#### `get_config(self)`
**Module:** `models.sam.image_encoder`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/image_encoder.py:702*

#### `get_config(self)`
**Module:** `models.sam.image_encoder`

Returns the configuration of the model for serialization.

*📁 src/dl_techniques/models/sam/image_encoder.py:895*

#### `get_config(self)`
**Module:** `models.sam.mask_decoder`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/mask_decoder.py:499*

#### `get_config(self)`
**Module:** `models.sam.model`

Returns the configuration of the model for serialization.

*📁 src/dl_techniques/models/sam/model.py:563*

#### `get_config(self)`
**Module:** `models.sam.transformer`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/transformer.py:423*

#### `get_config(self)`
**Module:** `models.sam.transformer`

Returns the configuration of the layer for serialization.

*📁 src/dl_techniques/models/sam/transformer.py:718*

#### `get_config(self)`
**Module:** `models.masked_autoencoder.mae`

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:330*

#### `get_config(self)`
**Module:** `models.masked_autoencoder.patch_masking`

*📁 src/dl_techniques/models/masked_autoencoder/patch_masking.py:138*

#### `get_config(self)`
**Module:** `models.masked_autoencoder.conv_decoder`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/masked_autoencoder/conv_decoder.py:203*

#### `get_config(self)`
**Module:** `models.som.model`

Return configuration dictionary for model serialization.

*📁 src/dl_techniques/models/som/model.py:1191*

#### `get_config(self)`
**Module:** `models.hierarchical_reasoning_model.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:761*

#### `get_config(self)`
**Module:** `models.clip.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/clip/model.py:680*

#### `get_config(self)`
**Module:** `models.kan.model`

*📁 src/dl_techniques/models/kan/model.py:506*

#### `get_config(self)`
**Module:** `models.tree_transformer.model`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:187*

#### `get_config(self)`
**Module:** `models.tree_transformer.model`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:365*

#### `get_config(self)`
**Module:** `models.tree_transformer.model`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:514*

#### `get_config(self)`
**Module:** `models.tree_transformer.model`

Returns the layer's configuration for serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:686*

#### `get_config(self)`
**Module:** `models.tree_transformer.model`

Returns the model's configuration for serialization.

*📁 src/dl_techniques/models/tree_transformer/model.py:1125*

#### `get_config(self)`
**Module:** `models.yolo12.feature_extractor`

Get model configuration for serialization.

*📁 src/dl_techniques/models/yolo12/feature_extractor.py:335*

#### `get_config(self)`
**Module:** `models.yolo12.multitask`

Get model configuration for serialization.

*📁 src/dl_techniques/models/yolo12/multitask.py:296*

#### `get_config(self)`
**Module:** `models.shgcn.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/shgcn/model.py:228*

#### `get_config(self)`
**Module:** `models.shgcn.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/shgcn/model.py:385*

#### `get_config(self)`
**Module:** `models.shgcn.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/shgcn/model.py:553*

#### `get_config(self)`
**Module:** `models.distilbert.model`

Return layer configuration for serialization.

*📁 src/dl_techniques/models/distilbert/model.py:259*

#### `get_config(self)`
**Module:** `models.distilbert.model`

Return the model's configuration for serialization.

*📁 src/dl_techniques/models/distilbert/model.py:932*

#### `get_config(self)`
**Module:** `models.pft_sr.model`

Return model configuration.

*📁 src/dl_techniques/models/pft_sr/model.py:354*

#### `get_config(self)`
**Module:** `models.pw_fnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/pw_fnet/model.py:423*

#### `get_config(self)`
**Module:** `models.pw_fnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/pw_fnet/model.py:523*

#### `get_config(self)`
**Module:** `models.pw_fnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/pw_fnet/model.py:611*

#### `get_config(self)`
**Module:** `models.pw_fnet.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/pw_fnet/model.py:953*

#### `get_config(self)`
**Module:** `models.convunext.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/convunext/model.py:197*

#### `get_config(self)`
**Module:** `models.convunext.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/convunext/model.py:1035*

#### `get_config(self)`
**Module:** `models.relgt.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/relgt/model.py:234*

#### `get_config(self)`
**Module:** `models.vq_vae.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/vq_vae/model.py:504*

#### `get_config(self)`
**Module:** `models.resnet.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/resnet/model.py:654*

#### `get_config(self)`
**Module:** `models.bert.bert`

Return the model's configuration for serialization.

*📁 src/dl_techniques/models/bert/bert.py:765*

#### `get_config(self)`
**Module:** `models.capsnet.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/capsnet/model.py:484*

#### `get_config(self)`
**Module:** `models.xlstm.model`

Return the configuration of the model.

*📁 src/dl_techniques/models/xlstm/model.py:325*

#### `get_config(self)`
**Module:** `models.power_mlp.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/power_mlp/model.py:550*

#### `get_config(self)`
**Module:** `models.ntm.model`

Serialize configuration.

*📁 src/dl_techniques/models/ntm/model.py:203*

#### `get_config(self)`
**Module:** `models.ntm.model_multitask`

Return configuration for serialization.

*📁 src/dl_techniques/models/ntm/model_multitask.py:150*

#### `get_config(self)`
**Module:** `models.darkir.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/darkir/model.py:182*

#### `get_config(self)`
**Module:** `models.darkir.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/darkir/model.py:384*

#### `get_config(self)`
**Module:** `models.darkir.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/darkir/model.py:547*

#### `get_config(self)`
**Module:** `models.darkir.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/darkir/model.py:892*

#### `get_config(self)`
**Module:** `models.darkir.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/darkir/model.py:1279*

#### `get_config(self)`
**Module:** `models.mothnet.model`

Return model configuration for serialization.

*📁 src/dl_techniques/models/mothnet/model.py:517*

#### `get_config(self)`
**Module:** `models.deepar.model`

*📁 src/dl_techniques/models/deepar/model.py:568*

#### `get_config(self)`
**Module:** `models.swin_transformer.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/swin_transformer/model.py:547*

#### `get_config(self)`
**Module:** `models.accunet.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/accunet/model.py:432*

#### `get_config(self)`
**Module:** `models.cbam.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/cbam/model.py:286*

#### `get_config(self)`
**Module:** `models.nbeats.nbeatsx`

*📁 src/dl_techniques/models/nbeats/nbeatsx.py:220*

#### `get_config(self)`
**Module:** `models.nbeats.nbeats`

Return complete configuration for serialization.

*📁 src/dl_techniques/models/nbeats/nbeats.py:551*

#### `get_config(self)`
**Module:** `models.vae.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/vae/model.py:812*

#### `get_config(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Get layer configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:296*

#### `get_config(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Get layer configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:510*

#### `get_config(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Get layer configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:806*

#### `get_config(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Get model configuration.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:1063*

#### `get_config(self)`
**Module:** `models.modern_bert.modern_bert`

Return the model's configuration for serialization.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:553*

#### `get_config(self)`
**Module:** `models.modern_bert.modern_bert_blt`

Serializes the model's configuration for saving.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt.py:295*

#### `get_config(self)`
**Module:** `models.modern_bert.components`

*📁 src/dl_techniques/models/modern_bert/components.py:76*

#### `get_config(self)`
**Module:** `models.modern_bert.components`

*📁 src/dl_techniques/models/modern_bert/components.py:200*

#### `get_config(self)`
**Module:** `models.modern_bert.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/modern_bert/components.py:404*

#### `get_config(self)`
**Module:** `models.vit_hmlp.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/vit_hmlp/model.py:623*

#### `get_config(self)`
**Module:** `models.fastvlm.model`

Get model configuration for serialization.

*📁 src/dl_techniques/models/fastvlm/model.py:538*

#### `get_config(self)`
**Module:** `models.fastvlm.components`

Get layer configuration for serialization.

*📁 src/dl_techniques/models/fastvlm/components.py:243*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_embeddings`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:311*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_embeddings`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:564*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_embeddings`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:691*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_embeddings`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_embeddings.py:817*

#### `get_config(self)`
**Module:** `models.qwen.qwen3`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3.py:442*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_mega`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:220*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_mega`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_mega.py:725*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_next`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:402*

#### `get_config(self)`
**Module:** `models.qwen.qwen3_som`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:668*

#### `get_config(self)`
**Module:** `models.qwen.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/qwen/components.py:383*

#### `get_config(self)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Get model configuration for serialization.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:331*

#### `get_config(self)`
**Module:** `models.mobile_clip.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/mobile_clip/components.py:151*

#### `get_config(self)`
**Module:** `models.mobile_clip.components`

Return model configuration for serialization.

*📁 src/dl_techniques/models/mobile_clip/components.py:282*

#### `get_config(self)`
**Module:** `models.mobile_clip.components`

Return configuration for serialization.

*📁 src/dl_techniques/models/mobile_clip/components.py:494*

#### `get_config(self)`
**Module:** `models.vit.model`

Return configuration for serialization.

*📁 src/dl_techniques/models/vit/model.py:534*

#### `get_config(self)`
**Module:** `models.squeezenet.squeezenet_v2`

Return configuration for serialization.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:184*

#### `get_config(self)`
**Module:** `models.squeezenet.squeezenet_v2`

Get model configuration for serialization.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:571*

#### `get_config(self)`
**Module:** `models.squeezenet.squeezenet_v1`

Return configuration for serialization.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:205*

#### `get_config(self)`
**Module:** `models.squeezenet.squeezenet_v1`

Get model configuration for serialization.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:553*

#### `get_dense_pe(self)`
**Module:** `models.sam.prompt_encoder`

Get dense positional encoding grid.

*📁 src/dl_techniques/models/sam/prompt_encoder.py:406*

#### `get_dim_at_level(dim, level)`
**Module:** `models.convunext.model`

*📁 src/dl_techniques/models/convunext/model.py:870*

#### `get_enabled_task_names(self)`
**Module:** `models.yolo12.multitask`

Get list of enabled task names as strings.

*📁 src/dl_techniques/models/yolo12/multitask.py:359*

#### `get_enabled_tasks(self)`
**Module:** `models.yolo12.multitask`

Get list of enabled tasks.

*📁 src/dl_techniques/models/yolo12/multitask.py:350*

#### `get_encoder_config(self)`
**Module:** `models.jepa.config`

Get configuration for encoder components.

*📁 src/dl_techniques/models/jepa/config.py:270*

#### `get_feature_extractor(self)`
**Module:** `models.vit_siglip.model`

Get a feature extractor version of this model.

*📁 src/dl_techniques/models/vit_siglip/model.py:672*

#### `get_feature_extractor(self)`
**Module:** `models.yolo12.multitask`

Get the shared feature extractor.

*📁 src/dl_techniques/models/yolo12/multitask.py:325*

#### `get_feature_extractor(self)`
**Module:** `models.vit_hmlp.model`

Get a feature extractor version of this model.

*📁 src/dl_techniques/models/vit_hmlp/model.py:654*

#### `get_feature_extractor(self)`
**Module:** `models.vit.model`

Get a feature extractor version of this model.

*📁 src/dl_techniques/models/vit/model.py:562*

#### `get_last_selfattention(self, inputs)`
**Module:** `models.dino.dino_v3`

Extracts attention weights from the last transformer layer's [CLS] token. Useful for visualization as shown in the DINO paper.

*📁 src/dl_techniques/models/dino/dino_v3.py:333*

#### `get_last_selfattention(self, inputs)`
**Module:** `models.dino.dino_v1`

Get attention weights from the last transformer layer.

*📁 src/dl_techniques/models/dino/dino_v1.py:624*

#### `get_mask_statistics(self, context_mask, target_mask)`
**Module:** `models.jepa.utilities`

Compute masking statistics for monitoring.

*📁 src/dl_techniques/models/jepa/utilities.py:363*

#### `get_model_for_inference(self)`
**Module:** `models.nano_vlm_world_model.train`

Get model for inference (EMA if available).

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:350*

#### `get_model_output_info(model)`
**Module:** `models.bias_free_denoisers.bfconvunext`

Get information about model outputs for deep supervision models.

*📁 src/dl_techniques/models/bias_free_denoisers/bfconvunext.py:645*

#### `get_model_output_info(model)`
**Module:** `models.bias_free_denoisers.bfunet`

Get information about model outputs for deep supervision models.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet.py:743*

#### `get_model_output_info(model)`
**Module:** `models.resnet.model`

Get information about model outputs for deep supervision models.

*📁 src/dl_techniques/models/resnet/model.py:694*

#### `get_patch_tokens(self, features)`
**Module:** `models.vit_siglip.model`

Extract patch tokens from vision_heads features for dense prediction tasks.

*📁 src/dl_techniques/models/vit_siglip/model.py:586*

#### `get_predictor_config(self)`
**Module:** `models.jepa.config`

Get configuration for predictor component.

*📁 src/dl_techniques/models/jepa/config.py:289*

#### `get_score_from_noise(self, noise_pred, timesteps, x_t)`
**Module:** `models.nano_vlm_world_model.scheduler`

Convert predicted noise to score function ∇_x log p(x_t).

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:220*

#### `get_som_prototypes(self)`
**Module:** `models.qwen.qwen3_som`

Get learned SOM prototype vectors from all memory layers.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:608*

#### `get_spatial_features(self, features)`
**Module:** `models.vit_siglip.model`

Reshape patch tokens back to spatial format for dense tasks.

*📁 src/dl_techniques/models/vit_siglip/model.py:598*

#### `get_velocity(self, sample, noise, timesteps)`
**Module:** `models.nano_vlm_world_model.scheduler`

Compute velocity prediction v_t for the v-prediction objective.

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:161*

#### `get_vocab_size(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Get vocabulary size from ByteTokenizer.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:506*

#### `has_task(self, task)`
**Module:** `models.yolo12.multitask`

Check if a specific task is enabled.

*📁 src/dl_techniques/models/yolo12/multitask.py:368*

#### `identity_fn()`
**Module:** `models.convunext.model`

Return x unchanged.

*📁 src/dl_techniques/models/convunext/model.py:998*

#### `increment_counter()`
**Module:** `models.ccnets.control`

*📁 src/dl_techniques/models/ccnets/control.py:96*

#### `inference_output()`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:809*

#### `initial_carry(self, batch)`
**Module:** `models.tiny_recursive_model.model`

Create the initial state for the ACT loop.

*📁 src/dl_techniques/models/tiny_recursive_model/model.py:198*

#### `initial_carry(self, batch)`
**Module:** `models.hierarchical_reasoning_model.model`

Initialize carry state for a batch.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:532*

#### `initial_carry(self, batch)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Initialize carry state for a batch.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:883*

#### `inject_class_conditioning(x, class_emb, level, stage)`
**Module:** `models.bias_free_denoisers.bfunet_conditional`

Inject class conditioning into feature maps.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional.py:167*

#### `inject_conditioning(features, level, stage)`
**Module:** `models.bias_free_denoisers.bfunet_conditional_unified`

Inject all available conditioning signals into features.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py:535*

#### `insert_register_tokens(args)`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:671*

#### `interpolate_pos_embed(x_input)`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:688*

#### `load_model(cls, filepath)`
**Module:** `models.capsnet.model`

Load a saved model.

*📁 src/dl_techniques/models/capsnet/model.py:539*

#### `load_model(cls, filepath)`
**Module:** `models.power_mlp.model`

Load a saved PowerMLP model.

*📁 src/dl_techniques/models/power_mlp/model.py:712*

#### `load_models(self, base_path)`
**Module:** `models.ccnets.orchestrators`

Load all three models from disk.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:398*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.convnext.convnext_v2`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:397*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.convnext.convnext_v1`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:380*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.fnet.model`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/fnet/model.py:442*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.kan.model`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/kan/model.py:210*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.tree_transformer.model`

Loads pretrained weights into the model.

*📁 src/dl_techniques/models/tree_transformer/model.py:1010*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.distilbert.model`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/distilbert/model.py:693*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.resnet.model`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/resnet/model.py:443*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.bert.bert`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/bert/bert.py:525*

#### `load_pretrained_weights(self, weights_path, skip_mismatch, by_name)`
**Module:** `models.modern_bert.modern_bert`

Load pretrained weights into the model.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:440*

#### `load_pretrained_weights_into_model(model, weights_path, skip_mismatch, by_name)`
**Module:** `models.bias_free_denoisers.bfunet`

Load pretrained weights into a BFUNet model.

*📁 src/dl_techniques/models/bias_free_denoisers/bfunet.py:515*

#### `make_divisible(value, divisor)`
**Module:** `models.mobilenet.mobilenet_v3`

Make value divisible by divisor.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:193*

#### `make_zero_grads()`
**Module:** `models.ccnets.orchestrators`

*📁 src/dl_techniques/models/ccnets/orchestrators.py:254*

#### `metrics(self)`
**Module:** `models.masked_language_model.mlm`

List of metrics for Keras to track.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:215*

#### `metrics(self)`
**Module:** `models.masked_language_model.clm`

*📁 src/dl_techniques/models/masked_language_model/clm.py:90*

#### `metrics(self)`
**Module:** `models.masked_autoencoder.mae`

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:311*

#### `metrics(self)`
**Module:** `models.vq_vae.model`

List of metrics tracked by the model.

*📁 src/dl_techniques/models/vq_vae/model.py:404*

#### `metrics(self)`
**Module:** `models.vae.model`

Return metrics tracked by the model.

*📁 src/dl_techniques/models/vae/model.py:604*

#### `navigate_semantic_space(self, start_vision, start_text, target_text, num_steps, step_size)`
**Module:** `models.nano_vlm_world_model.model`

Navigate from one point to another in semantic space (Protocol 3).

*📁 src/dl_techniques/models/nano_vlm_world_model/model.py:473*

#### `negative_binomial_loss(y_true, y_pred)`
**Module:** `models.deepar.model`

Negative Binomial negative log-likelihood loss.

*📁 src/dl_techniques/models/deepar/model.py:535*

#### `postprocess_masks(self, masks, input_size, original_size)`
**Module:** `models.sam.model`

Postprocess predicted masks to match original image size.

*📁 src/dl_techniques/models/sam/model.py:388*

#### `predict_class(self, x_test)`
**Module:** `models.som.model`

Classify samples using fitted class prototypes and topological similarity.

*📁 src/dl_techniques/models/som/model.py:556*

#### `predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, training)`
**Module:** `models.sam.mask_decoder`

Generate mask predictions and IoU estimates.

*📁 src/dl_techniques/models/sam/mask_decoder.py:388*

#### `predict_quantiles(self, context, quantile_levels, batch_size)`
**Module:** `models.tirex.model`

Generate specific quantile and point forecasts for time series data.

*📁 src/dl_techniques/models/tirex/model.py:358*

#### `predict_quantiles(self, context, quantile_levels, batch_size)`
**Module:** `models.prism.model`

Generate specific quantile and point forecasts for time series data.

*📁 src/dl_techniques/models/prism/model.py:441*

#### `predict_start_from_noise(self, x_t, t, noise)`
**Module:** `models.nano_vlm_world_model.scheduler`

Predict x_0 from x_t and predicted noise ε using Miyasawa's theorem.

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:188*

#### `predict_step(self, data)`
**Module:** `models.deepar.model`

Override predict_step to use sampling mode.

*📁 src/dl_techniques/models/deepar/model.py:503*

#### `predict_with_uncertainty(self, inputs, confidence_level)`
**Module:** `models.mdn.model`

Generate predictions with comprehensive uncertainty estimates.

*📁 src/dl_techniques/models/mdn/model.py:461*

#### `preprocess(self, x)`
**Module:** `models.sam.model`

Preprocess input image for the encoder.

*📁 src/dl_techniques/models/sam/model.py:359*

#### `quantize_latents(self, latents)`
**Module:** `models.vq_vae.model`

Quantize continuous latents to discrete representations.

*📁 src/dl_techniques/models/vq_vae/model.py:429*

#### `reset_carry(self, reset_flag, carry)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Reset carry state for halted sequences.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:692*

#### `reset_counter()`
**Module:** `models.ccnets.control`

*📁 src/dl_techniques/models/ccnets/control.py:99*

#### `reset_metrics(self)`
**Module:** `models.nano_vlm_world_model.train`

Reset all metrics.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:344*

#### `resize_fn()`
**Module:** `models.convunext.model`

Resize x to match skip dimensions.

*📁 src/dl_techniques/models/convunext/model.py:990*

#### `reverse_sequence(self, sequence)`
**Module:** `models.ccnets.orchestrators`

Reverse a sequence along the time dimension (axis 1).

*📁 src/dl_techniques/models/ccnets/orchestrators.py:425*

#### `run_alignment_example(n_samples, n_eval, embed_dim, approx_clusters, approx_runs, approx_neighbors, refine1_iterations, refine1_sample_size, refine1_neighbors, refine2_clusters, smoothing_alpha, seed)`
**Module:** `models.mini_vec2vec.example_alignment`

Run complete alignment example with evaluation and testing.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:246*

#### `sample(self, inputs, num_samples, temperature, seed)`
**Module:** `models.mdn.model`

Generate samples from the predicted distribution.

*📁 src/dl_techniques/models/mdn/model.py:381*

#### `sample(self, num_samples)`
**Module:** `models.vae.model`

Generate samples from the latent space.

*📁 src/dl_techniques/models/vae/model.py:637*

#### `save(self, filepath)`
**Module:** `models.ccnets.base`

Save the model.

*📁 src/dl_techniques/models/ccnets/base.py:24*

#### `save(self, filepath)`
**Module:** `models.ccnets.utils`

*📁 src/dl_techniques/models/ccnets/utils.py:109*

#### `save(self, filepath)`
**Module:** `models.mdn.model`

Save the model to a file.

*📁 src/dl_techniques/models/mdn/model.py:722*

#### `save_model(self, filepath, overwrite, save_format)`
**Module:** `models.capsnet.model`

Save the model to a file.

*📁 src/dl_techniques/models/capsnet/model.py:522*

#### `save_model(self, filepath, overwrite, save_format)`
**Module:** `models.power_mlp.model`

Save the model to a file.

*📁 src/dl_techniques/models/power_mlp/model.py:685*

#### `save_models(self, base_path)`
**Module:** `models.ccnets.orchestrators`

Save all three models to disk.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:392*

#### `set_curriculum_stage(self, stage)`
**Module:** `models.jepa.utilities`

Set current curriculum learning stage.

*📁 src/dl_techniques/models/jepa/utilities.py:473*

#### `set_difficulty(self, progress)`
**Module:** `models.jepa.utilities`

Set curriculum learning difficulty.

*📁 src/dl_techniques/models/jepa/utilities.py:83*

#### `should_train_reasoner(self, metrics)`
**Module:** `models.ccnets.control`

Determine if the Reasoner should be trained in the current step.

*📁 src/dl_techniques/models/ccnets/control.py:14*

#### `should_train_reasoner(self, metrics)`
**Module:** `models.ccnets.control`

Decision is based solely on batch accuracy.

*📁 src/dl_techniques/models/ccnets/control.py:40*

#### `should_train_reasoner(self, metrics)`
**Module:** `models.ccnets.control`

Decision logic written with pure TensorFlow ops to be graph-compatible.

*📁 src/dl_techniques/models/ccnets/control.py:82*

#### `split_outputs(args)`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:785*

#### `step(self, model_output, timestep, sample, generator)`
**Module:** `models.nano_vlm_world_model.scheduler`

Reverse diffusion step: predict x_{t-1} from x_t and model output.

*📁 src/dl_techniques/models/nano_vlm_world_model/scheduler.py:248*

#### `style_transfer(self, x_content, x_style)`
**Module:** `models.ccnets.orchestrators`

Perform style transfer between observations.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:351*

#### `summary(self)`
**Module:** `models.coshnet.model`

Print model summary with additional CoShNet-specific information.

*📁 src/dl_techniques/models/coshnet/model.py:616*

#### `summary(self)`
**Module:** `models.mobilenet.mobilenet_v1`

Print model summary with additional information.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v1.py:315*

#### `summary(self)`
**Module:** `models.mobilenet.mobilenet_v4`

Print model summary with additional information.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v4.py:463*

#### `summary(self)`
**Module:** `models.mobilenet.mobilenet_v2`

Print model summary with additional information.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v2.py:321*

#### `summary(self)`
**Module:** `models.mobilenet.mobilenet_v3`

Print model summary with additional information.

*📁 src/dl_techniques/models/mobilenet/mobilenet_v3.py:367*

#### `summary(self)`
**Module:** `models.tabm.model`

Print model summary with TabM-specific information.

*📁 src/dl_techniques/models/tabm/model.py:772*

#### `summary(self)`
**Module:** `models.mamba.mamba_v1`

Print model summary with Mamba-specific information.

*📁 src/dl_techniques/models/mamba/mamba_v1.py:520*

#### `summary(self)`
**Module:** `models.byte_latent_transformer.model`

Print model summary with BLT-specific information.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:668*

#### `summary(self)`
**Module:** `models.fractalnet.model`

Print model summary with additional information.

*📁 src/dl_techniques/models/fractalnet/model.py:415*

#### `summary(self)`
**Module:** `models.convnext.convnext_v2`

Print model summary with additional information.

*📁 src/dl_techniques/models/convnext/convnext_v2.py:716*

#### `summary(self)`
**Module:** `models.convnext.convnext_v1`

Print model summary with additional information.

*📁 src/dl_techniques/models/convnext/convnext_v1.py:660*

#### `summary(self)`
**Module:** `models.fnet.model`

Print the model summary with additional FNet-specific information.

*📁 src/dl_techniques/models/fnet/model.py:717*

#### `summary(self)`
**Module:** `models.fftnet.model`

Print model summary with additional FFTNet-specific information.

*📁 src/dl_techniques/models/fftnet/model.py:676*

#### `summary(self)`
**Module:** `models.dino.dino_v1`

Print model summary with additional information.

*📁 src/dl_techniques/models/dino/dino_v1.py:768*

#### `summary(self)`
**Module:** `models.dino.dino_v2`

Print model summary with additional information.

*📁 src/dl_techniques/models/dino/dino_v2.py:893*

#### `summary(self)`
**Module:** `models.dino.dino_v2`

Print model summary with additional information.

*📁 src/dl_techniques/models/dino/dino_v2.py:1132*

#### `summary(self)`
**Module:** `models.hierarchical_reasoning_model.model`

Print model summary with additional HRM information.

*📁 src/dl_techniques/models/hierarchical_reasoning_model/model.py:824*

#### `summary(self)`
**Module:** `models.kan.model`

*📁 src/dl_techniques/models/kan/model.py:502*

#### `summary(self)`
**Module:** `models.tree_transformer.model`

Prints the model summary with additional configuration details.

*📁 src/dl_techniques/models/tree_transformer/model.py:1152*

#### `summary(self)`
**Module:** `models.distilbert.model`

Print the model summary with additional DistilBERT-specific information.

*📁 src/dl_techniques/models/distilbert/model.py:973*

#### `summary(self)`
**Module:** `models.bert.bert`

Print the model summary with additional BERT-specific information.

*📁 src/dl_techniques/models/bert/bert.py:808*

#### `summary(self)`
**Module:** `models.capsnet.model`

Print model summary with additional information.

*📁 src/dl_techniques/models/capsnet/model.py:554*

#### `summary(self)`
**Module:** `models.power_mlp.model`

Print model summary with additional PowerMLP-specific information.

*📁 src/dl_techniques/models/power_mlp/model.py:733*

#### `summary(self)`
**Module:** `models.swin_transformer.model`

Print model summary with Swin Transformer specific information.

*📁 src/dl_techniques/models/swin_transformer/model.py:595*

#### `summary(self)`
**Module:** `models.vae.model`

Print model summary with additional information.

*📁 src/dl_techniques/models/vae/model.py:863*

#### `summary(self)`
**Module:** `models.modern_bert.modern_bert`

Print the model summary with additional ModernBERT-specific information.

*📁 src/dl_techniques/models/modern_bert/modern_bert.py:579*

#### `summary(self)`
**Module:** `models.fastvlm.model`

Print model summary with additional information.

*📁 src/dl_techniques/models/fastvlm/model.py:578*

#### `summary(self)`
**Module:** `models.qwen.qwen3`

Print model summary with additional Qwen3-specific information.

*📁 src/dl_techniques/models/qwen/qwen3.py:472*

#### `summary(self)`
**Module:** `models.qwen.qwen3_next`

Print model summary with additional Qwen3-specific information.

*📁 src/dl_techniques/models/qwen/qwen3_next.py:430*

#### `summary(self)`
**Module:** `models.qwen.qwen3_som`

Print model summary with SOM-specific information.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:705*

#### `summary(self)`
**Module:** `models.mobile_clip.mobile_clip_v1`

Print model summary with additional information.

*📁 src/dl_techniques/models/mobile_clip/mobile_clip_v1.py:348*

#### `summary_detailed(self)`
**Module:** `models.vit_siglip.model`

Print detailed model summary with architecture information.

*📁 src/dl_techniques/models/vit_siglip/model.py:703*

#### `summary_detailed(self)`
**Module:** `models.vit_hmlp.model`

Print detailed model summary with architecture information.

*📁 src/dl_techniques/models/vit_hmlp/model.py:688*

#### `summary_detailed(self)`
**Module:** `models.vit.model`

Print detailed model summary with architecture information.

*📁 src/dl_techniques/models/vit/model.py:593*

#### `summary_with_details(self)`
**Module:** `models.squeezenet.squeezenet_v2`

Print detailed model summary with configuration information.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v2.py:607*

#### `summary_with_details(self)`
**Module:** `models.squeezenet.squeezenet_v1`

Print detailed model summary with configuration information.

*📁 src/dl_techniques/models/squeezenet/squeezenet_v1.py:589*

#### `test_serialization(aligner, XA_eval, save_dir)`
**Module:** `models.mini_vec2vec.example_alignment`

Test model serialization and deserialization.

*📁 src/dl_techniques/models/mini_vec2vec/example_alignment.py:192*

#### `test_step(self, data)`
**Module:** `models.latent_gmm_registration.model`

Custom test step with semi-supervised loss evaluation.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:304*

#### `test_step(self, data)`
**Module:** `models.masked_language_model.mlm`

Custom validation step for MLM with dynamic masking.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:345*

#### `test_step(self, data)`
**Module:** `models.masked_language_model.clm`

*📁 src/dl_techniques/models/masked_language_model/clm.py:247*

#### `test_step(self, data)`
**Module:** `models.masked_autoencoder.mae`

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:294*

#### `test_step(self, data)`
**Module:** `models.vq_vae.model`

Custom test step for evaluation.

*📁 src/dl_techniques/models/vq_vae/model.py:360*

#### `test_step(self, data)`
**Module:** `models.capsnet.model`

Custom test step with margin loss and reconstruction loss.

*📁 src/dl_techniques/models/capsnet/model.py:430*

#### `test_step(self, data)`
**Module:** `models.vae.model`

Custom test step with VAE losses.

*📁 src/dl_techniques/models/vae/model.py:715*

#### `text_to_bytes(self, text, add_bos, add_eos)`
**Module:** `models.modern_bert.components`

Converts a string to a list of byte token IDs.

*📁 src/dl_techniques/models/modern_bert/components.py:54*

#### `to_dict(self)`
**Module:** `models.ccnets.base`

Convert losses to dictionary of scalars.

*📁 src/dl_techniques/models/ccnets/base.py:95*

#### `to_dict(self)`
**Module:** `models.ccnets.base`

Convert errors to dictionary of scalars.

*📁 src/dl_techniques/models/ccnets/base.py:120*

#### `to_dict(self)`
**Module:** `models.jepa.config`

Convert configuration to dictionary.

*📁 src/dl_techniques/models/jepa/config.py:258*

#### `to_dict(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Convert configuration to dictionary.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:179*

#### `tokens_to_text(self, token_ids)`
**Module:** `models.modern_bert.components`

Converts a list of byte token IDs back to a string.

*📁 src/dl_techniques/models/modern_bert/components.py:66*

#### `train(self, train_dataset, epochs, validation_dataset, callbacks)`
**Module:** `models.ccnets.trainer`

Train the CCNet for multiple epochs.

*📁 src/dl_techniques/models/ccnets/trainer.py:46*

#### `train(self, x_train, epochs, batch_size, shuffle, verbose)`
**Module:** `models.som.model`

Train the SOM to organize input data into a topological memory structure.

*📁 src/dl_techniques/models/som/model.py:347*

#### `train_hebbian(self, x, y, epochs, batch_size, verbose)`
**Module:** `models.mothnet.model`

Train the model using Hebbian learning.

*📁 src/dl_techniques/models/mothnet/model.py:413*

#### `train_score_vlm(model, train_dataset, epochs, optimizer_config, checkpoint_dir, log_frequency)`
**Module:** `models.nano_vlm_world_model.train`

Main training loop for Score-Based nanoVLM.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:355*

#### `train_step(self, data)`
**Module:** `models.depth_anything.model`

Execute one training step.

*📁 src/dl_techniques/models/depth_anything/model.py:402*

#### `train_step(self, images, text_tokens)`
**Module:** `models.nano_vlm_world_model.train`

Single training step with DSM.

*📁 src/dl_techniques/models/nano_vlm_world_model/train.py:241*

#### `train_step(self, x_input, y_truth)`
**Module:** `models.ccnets.orchestrators`

Perform a single training step with causal credit assignment. This implementation is compiled to a static graph for performance.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:205*

#### `train_step(self, data)`
**Module:** `models.byte_latent_transformer.model`

Custom training step for BLT.

*📁 src/dl_techniques/models/byte_latent_transformer/model.py:372*

#### `train_step(self, data)`
**Module:** `models.latent_gmm_registration.model`

Custom training step with semi-supervised loss.

*📁 src/dl_techniques/models/latent_gmm_registration/model.py:218*

#### `train_step(self, data)`
**Module:** `models.masked_language_model.mlm`

Custom training step for MLM with dynamic masking.

*📁 src/dl_techniques/models/masked_language_model/mlm.py:309*

#### `train_step(self, data)`
**Module:** `models.masked_language_model.clm`

*📁 src/dl_techniques/models/masked_language_model/clm.py:221*

#### `train_step(self, data)`
**Module:** `models.masked_autoencoder.mae`

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:272*

#### `train_step(self, data)`
**Module:** `models.vq_vae.model`

Custom training step that computes VQ-VAE losses.

*📁 src/dl_techniques/models/vq_vae/model.py:307*

#### `train_step(self, data)`
**Module:** `models.capsnet.model`

Custom training step with margin loss and reconstruction loss.

*📁 src/dl_techniques/models/capsnet/model.py:368*

#### `train_step(self, data)`
**Module:** `models.vae.model`

Custom training step with VAE losses.

*📁 src/dl_techniques/models/vae/model.py:649*

#### `trainable_variables(self)`
**Module:** `models.ccnets.base`

Get trainable variables of the module.

*📁 src/dl_techniques/models/ccnets/base.py:20*

#### `trainable_variables(self)`
**Module:** `models.ccnets.utils`

*📁 src/dl_techniques/models/ccnets/utils.py:106*

#### `training_output()`
**Module:** `models.dino.dino_v2`

*📁 src/dl_techniques/models/dino/dino_v2.py:800*

#### `update_kan_grids(self, x_data)`
**Module:** `models.kan.model`

Update the B-spline grids of all KANLinear layers using the provided data.

*📁 src/dl_techniques/models/kan/model.py:170*

#### `update_state(self, metrics)`
**Module:** `models.ccnets.control`

Update the internal state of the strategy with the latest metrics.

*📁 src/dl_techniques/models/ccnets/control.py:9*

#### `update_state(self, metrics)`
**Module:** `models.ccnets.control`

This strategy is stateless and does not require updates.

*📁 src/dl_techniques/models/ccnets/control.py:36*

#### `update_state(self, metrics)`
**Module:** `models.ccnets.control`

Update the EMAs. This method should be called from an EAGER context (e.g., the Python training loop in the Trainer).

*📁 src/dl_techniques/models/ccnets/control.py:72*

#### `validate(self)`
**Module:** `models.modern_bert.modern_bert_blt_hrm`

Validate configuration parameters.

*📁 src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:167*

#### `verify_consistency(self, x_input, threshold)`
**Module:** `models.ccnets.orchestrators`

Verify the internal consistency of the CCNet's reasoning.

*📁 src/dl_techniques/models/ccnets/orchestrators.py:379*

#### `visualize(self, image, return_arrays)`
**Module:** `models.masked_autoencoder.mae`

Visualize a single image reconstruction.

*📁 src/dl_techniques/models/masked_autoencoder/mae.py:314*

#### `visualize_class_distribution(self, x_data, y_data, figsize, cmap, alpha, marker_size, save_path)`
**Module:** `models.som.model`

Visualize how different classes distribute across the SOM grid topology.

*📁 src/dl_techniques/models/som/model.py:735*

#### `visualize_grid(self, figsize, cmap, save_path)`
**Module:** `models.som.model`

Visualize the learned SOM grid showing neuron prototype memories.

*📁 src/dl_techniques/models/som/model.py:640*

#### `visualize_hit_histogram(self, x_data, figsize, cmap, log_scale, save_path)`
**Module:** `models.som.model`

Visualize activation frequency across the SOM grid (hit histogram).

*📁 src/dl_techniques/models/som/model.py:950*

#### `visualize_masks(self, context_mask, target_mask, sample_idx)`
**Module:** `models.jepa.utilities`

Visualize masks for debugging and analysis.

*📁 src/dl_techniques/models/jepa/utilities.py:327*

#### `visualize_memory_recall(self, test_sample, n_similar, x_train, y_train, figsize, cmap, save_path)`
**Module:** `models.som.model`

Demonstrate associative memory recall for a query sample.

*📁 src/dl_techniques/models/som/model.py:1037*

#### `visualize_mlm_predictions(mlm_model, inputs, tokenizer, num_samples)`
**Module:** `models.masked_language_model.utils`

Visualizes the model's ability to fill in masked tokens.

*📁 src/dl_techniques/models/masked_language_model/utils.py:14*

#### `visualize_reconstruction(mae, images, num_samples)`
**Module:** `models.masked_autoencoder.utils`

Visualize MAE reconstructions for multiple images.

*📁 src/dl_techniques/models/masked_autoencoder/utils.py:48*

#### `visualize_som_assignments(model, input_ids, attention_mask, layer_name)`
**Module:** `models.qwen.qwen3_som`

Visualize SOM assignments for given inputs.

*📁 src/dl_techniques/models/qwen/qwen3_som.py:936*

#### `visualize_u_matrix(self, figsize, cmap, save_path)`
**Module:** `models.som.model`

Visualize the Unified Distance Matrix (U-Matrix) revealing cluster boundaries.

*📁 src/dl_techniques/models/som/model.py:858*

#### `wrap_keras_model(model)`
**Module:** `models.ccnets.utils`

Wrap a Keras model to comply with CCNetModule protocol.

*📁 src/dl_techniques/models/ccnets/utils.py:87*

### Optimization Functions

#### `build(self, var_list)`
**Module:** `optimization.muon_optimizer`

Initialize optimizer state variables.

*📁 src/dl_techniques/optimization/muon_optimizer.py:136*

#### `build(self)`
**Module:** `optimization.train_vision.framework`

Construct training and validation datasets.

*📁 src/dl_techniques/optimization/train_vision/framework.py:357*

#### `config_from_args(args)`
**Module:** `optimization.train_vision.framework`

Create TrainingConfig from command-line arguments.

*📁 src/dl_techniques/optimization/train_vision/framework.py:1432*

#### `constant_equal_schedule(progress, no_outputs)`
**Module:** `optimization.deep_supervision`

Equal weighting for all outputs regardless of training progress.

*📁 src/dl_techniques/optimization/deep_supervision.py:237*

#### `constant_high_to_low_schedule(progress, no_outputs)`
**Module:** `optimization.deep_supervision`

Constant weighting favoring deeper (lower resolution) outputs.

*📁 src/dl_techniques/optimization/deep_supervision.py:315*

#### `constant_low_to_high_schedule(progress, no_outputs)`
**Module:** `optimization.deep_supervision`

Constant weighting favoring higher resolution (shallower) outputs.

*📁 src/dl_techniques/optimization/deep_supervision.py:275*

#### `cosine_annealing_schedule(progress, no_outputs, frequency, final_ratio)`
**Module:** `optimization.deep_supervision`

Cyclical weight patterns using cosine functions with annealing.

*📁 src/dl_techniques/optimization/deep_supervision.py:580*

#### `create_argument_parser()`
**Module:** `optimization.train_vision.framework`

Create argument parser for command-line configuration.

*📁 src/dl_techniques/optimization/train_vision/framework.py:1275*

#### `curriculum_schedule(progress, no_outputs, max_active_outputs, activation_strategy)`
**Module:** `optimization.deep_supervision`

Progressive curriculum learning with gradual output activation.

*📁 src/dl_techniques/optimization/deep_supervision.py:658*

#### `custom_sigmoid_low_to_high_schedule(progress, no_outputs, k, x0, transition_point)`
**Module:** `optimization.deep_supervision`

Sigmoid-based transition from deep to shallow layer focus.

*📁 src/dl_techniques/optimization/deep_supervision.py:451*

#### `false_fn()`
**Module:** `optimization.muon_optimizer`

*📁 src/dl_techniques/optimization/muon_optimizer.py:239*

#### `from_config(cls, config)`
**Module:** `optimization.muon_optimizer`

Create optimizer from configuration.

*📁 src/dl_techniques/optimization/muon_optimizer.py:411*

#### `from_config(cls, config)`
**Module:** `optimization.warmup_schedule`

Create WarmupSchedule instance from configuration dictionary.

*📁 src/dl_techniques/optimization/warmup_schedule.py:205*

#### `get_class_names(self)`
**Module:** `optimization.train_vision.framework`

Get class names for the dataset.

*📁 src/dl_techniques/optimization/train_vision/framework.py:384*

#### `get_config(self)`
**Module:** `optimization.muon_optimizer`

Return optimizer configuration.

*📁 src/dl_techniques/optimization/muon_optimizer.py:389*

#### `get_config(self)`
**Module:** `optimization.warmup_schedule`

Return configuration dictionary for serialization.

*📁 src/dl_techniques/optimization/warmup_schedule.py:186*

#### `get_history_data(self)`
**Module:** `optimization.train_vision.framework`

Get the collected history data.

*📁 src/dl_techniques/optimization/train_vision/framework.py:623*

#### `get_test_data(self)`
**Module:** `optimization.train_vision.framework`

Get test data for model analysis.

*📁 src/dl_techniques/optimization/train_vision/framework.py:375*

#### `learning_rate_schedule_builder(config)`
**Module:** `optimization.optimizer`

Build a learning rate schedule from configuration.

*📁 src/dl_techniques/optimization/optimizer.py:108*

#### `linear_low_to_high_schedule(progress, no_outputs)`
**Module:** `optimization.deep_supervision`

Linear transition from focusing on deep to shallow layer outputs.

*📁 src/dl_techniques/optimization/deep_supervision.py:355*

#### `load(cls, file_path)`
**Module:** `optimization.train_vision.framework`

Load configuration from JSON file.

*📁 src/dl_techniques/optimization/train_vision/framework.py:315*

#### `non_linear_low_to_high_schedule(progress, no_outputs)`
**Module:** `optimization.deep_supervision`

Non-linear (quadratic) transition from deep to shallow layer focus.

*📁 src/dl_techniques/optimization/deep_supervision.py:400*

#### `on_epoch_end(self, epoch, logs)`
**Module:** `optimization.train_vision.framework`

Called at the end of each epoch to update and visualize metrics.

*📁 src/dl_techniques/optimization/train_vision/framework.py:454*

#### `on_train_end(self, logs)`
**Module:** `optimization.train_vision.framework`

Create final visualizations when training ends.

*📁 src/dl_techniques/optimization/train_vision/framework.py:619*

#### `optimizer_builder(config, lr_schedule)`
**Module:** `optimization.optimizer`

Build and configure a Keras optimizer from configuration dictionary.

*📁 src/dl_techniques/optimization/optimizer.py:218*

#### `primary_fn()`
**Module:** `optimization.warmup_schedule`

*📁 src/dl_techniques/optimization/warmup_schedule.py:173*

#### `process_wide(X)`
**Module:** `optimization.muon_optimizer`

Process matrix assuming it's wide (cols >= rows).

*📁 src/dl_techniques/optimization/muon_optimizer.py:217*

#### `run(self, model_builder, dataset_builder, custom_callbacks)`
**Module:** `optimization.train_vision.framework`

Execute the complete training pipeline.

*📁 src/dl_techniques/optimization/train_vision/framework.py:1140*

#### `save(self, file_path)`
**Module:** `optimization.train_vision.framework`

Save configuration to JSON file.

*📁 src/dl_techniques/optimization/train_vision/framework.py:303*

#### `scale_by_scale_low_to_high_schedule(progress, no_outputs)`
**Module:** `optimization.deep_supervision`

Progressive activation of outputs from deep to shallow, one at a time.

*📁 src/dl_techniques/optimization/deep_supervision.py:523*

#### `schedule_builder(config)`
**Module:** `optimization.schedule`

Build a learning rate schedule with optional warmup from configuration.

*📁 src/dl_techniques/optimization/schedule.py:62*

#### `schedule_builder(config, no_outputs, invert_order)`
**Module:** `optimization.deep_supervision`

Build a deep supervision weight scheduler from configuration.

*📁 src/dl_techniques/optimization/deep_supervision.py:68*

#### `sled_builder(config)`
**Module:** `optimization.sled_supervision`

Builds a SledLogitsProcessor from a configuration dictionary.

*📁 src/dl_techniques/optimization/sled_supervision.py:224*

#### `step_wise_schedule(progress, no_outputs, threshold)`
**Module:** `optimization.deep_supervision`

Step-wise transition with hard cutoff to shallowest layer.

*📁 src/dl_techniques/optimization/deep_supervision.py:744*

#### `to_optimizer_config(self)`
**Module:** `optimization.train_vision.framework`

Convert training config to optimizer builder config.

*📁 src/dl_techniques/optimization/train_vision/framework.py:254*

#### `to_schedule_config(self, total_steps)`
**Module:** `optimization.train_vision.framework`

Convert training config to schedule builder config.

*📁 src/dl_techniques/optimization/train_vision/framework.py:220*

#### `true_fn()`
**Module:** `optimization.muon_optimizer`

*📁 src/dl_techniques/optimization/muon_optimizer.py:233*

#### `update_step(self, gradient, variable, learning_rate)`
**Module:** `optimization.muon_optimizer`

Apply a single optimization step.

*📁 src/dl_techniques/optimization/muon_optimizer.py:246*

#### `warmup_fn()`
**Module:** `optimization.warmup_schedule`

*📁 src/dl_techniques/optimization/warmup_schedule.py:167*

### Regularizers Functions

#### `create_binary_preference_regularizer(multiplier, scale)`
**Module:** `regularizers.binary_preference`

Factory function to create binary preference regularizers.

*📁 src/dl_techniques/regularizers/binary_preference.py:249*

#### `create_entropy_regularizer(strength, target_entropy, mode)`
**Module:** `regularizers.entropy_regularizer`

Factory function to create entropy regularizers with common configurations.

*📁 src/dl_techniques/regularizers/entropy_regularizer.py:287*

#### `create_srip_regularizer(lambda_init, power_iterations, epsilon, lambda_schedule)`
**Module:** `regularizers.srip`

Factory function to create a SRIP regularizer instance.

*📁 src/dl_techniques/regularizers/srip.py:382*

#### `current_lambda(self)`
**Module:** `regularizers.srip`

Current regularization strength.

*📁 src/dl_techniques/regularizers/srip.py:140*

#### `from_config(cls, config)`
**Module:** `regularizers.srip`

Create regularizer instance from configuration dictionary.

*📁 src/dl_techniques/regularizers/srip.py:353*

#### `from_config(cls, config)`
**Module:** `regularizers.tri_state_preference`

Create a regularizer instance from configuration dictionary.

*📁 src/dl_techniques/regularizers/tri_state_preference.py:162*

#### `get_config(self)`
**Module:** `regularizers.l2_custom`

*📁 src/dl_techniques/regularizers/l2_custom.py:81*

#### `get_config(self)`
**Module:** `regularizers.srip`

Return the configuration of the regularizer.

*📁 src/dl_techniques/regularizers/srip.py:339*

#### `get_config(self)`
**Module:** `regularizers.soft_orthogonal`

Get regularizer configuration for serialization.

*📁 src/dl_techniques/regularizers/soft_orthogonal.py:273*

#### `get_config(self)`
**Module:** `regularizers.soft_orthogonal`

Get regularizer configuration for serialization.

*📁 src/dl_techniques/regularizers/soft_orthogonal.py:455*

#### `get_config(self)`
**Module:** `regularizers.tri_state_preference`

Return the configuration of the regularizer.

*📁 src/dl_techniques/regularizers/tri_state_preference.py:150*

#### `get_config(self)`
**Module:** `regularizers.entropy_regularizer`

Get the regularizer configuration for serialization.

*📁 src/dl_techniques/regularizers/entropy_regularizer.py:259*

#### `get_config(self)`
**Module:** `regularizers.binary_preference`

Return the configuration of the regularizer for serialization.

*📁 src/dl_techniques/regularizers/binary_preference.py:226*

#### `update_lambda(self, epoch)`
**Module:** `regularizers.srip`

Update lambda value based on current epoch.

*📁 src/dl_techniques/regularizers/srip.py:321*

#### `validate_float_arg(value, name)`
**Module:** `regularizers.l2_custom`

check penalty number availability, raise ValueError if failed.

*📁 src/dl_techniques/regularizers/l2_custom.py:42*

### Utils Functions

#### `abs_max(inputs, keepdim)`
**Module:** `utils.scaling`

Compute the absolute maximum value of the input tensor.

*📁 src/dl_techniques/utils/scaling.py:111*

#### `abs_mean(inputs, keepdim)`
**Module:** `utils.scaling`

Compute the absolute mean value of the input tensor.

*📁 src/dl_techniques/utils/scaling.py:134*

#### `abs_median(inputs, keepdim)`
**Module:** `utils.scaling`

Compute the absolute median value of the input tensor.

*📁 src/dl_techniques/utils/scaling.py:204*

#### `add_features(inputs_targets, time_feat)`
**Module:** `datasets.time_series.pipeline`

*📁 src/dl_techniques/datasets/time_series/pipeline.py:357*

#### `add_time_features(dataset, time_indices, feature_extractors)`
**Module:** `datasets.time_series.pipeline`

Add time-based features to an existing dataset.

*📁 src/dl_techniques/datasets/time_series/pipeline.py:325*

#### `adjust_bboxes()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:690*

#### `aggregate_patch_results(patch_predictions, image_width, image_height)`
**Module:** `datasets.patch_transforms`

Convenience function to aggregate patch results.

*📁 src/dl_techniques/datasets/patch_transforms.py:607*

#### `aggregate_predictions(self, patch_predictions, image_width, image_height)`
**Module:** `datasets.patch_transforms`

Aggregate all patch predictions into final results.

*📁 src/dl_techniques/datasets/patch_transforms.py:526*

#### `aggregate_scores(patch_predictions, method)`
**Module:** `datasets.patch_transforms`

Aggregate patch classification scores into image-level prediction.

*📁 src/dl_techniques/datasets/patch_transforms.py:449*

#### `all_valid()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:519*

#### `analyze_puzzle_complexity(self, puzzle)`
**Module:** `datasets.arc.arc_utilities`

Analyze the complexity of a specific puzzle.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:451*

#### `apply_brightness(image, severity)`
**Module:** `utils.corruption`

Apply brightness corruption.

*📁 src/dl_techniques/utils/corruption.py:480*

#### `apply_contrast(image, severity)`
**Module:** `utils.corruption`

Apply contrast corruption.

*📁 src/dl_techniques/utils/corruption.py:505*

#### `apply_corruption(image, corruption_type, severity)`
**Module:** `utils.corruption`

Apply a specific corruption to an image.

*📁 src/dl_techniques/utils/corruption.py:604*

#### `apply_gaussian_blur(image, severity)`
**Module:** `utils.corruption`

Apply Gaussian blur corruption.

*📁 src/dl_techniques/utils/corruption.py:258*

#### `apply_gaussian_noise(image, severity)`
**Module:** `utils.corruption`

Apply Gaussian noise corruption.

*📁 src/dl_techniques/utils/corruption.py:145*

#### `apply_impulse_noise(image, severity)`
**Module:** `utils.corruption`

Apply impulse (salt and pepper) noise corruption.

*📁 src/dl_techniques/utils/corruption.py:170*

#### `apply_mask(inputs, mask, mask_value, mask_type)`
**Module:** `utils.masking.factory`

Apply a mask to inputs (attention logits or segmentation predictions).

*📁 src/dl_techniques/utils/masking/factory.py:741*

#### `apply_mlm_masking(input_ids, attention_mask, vocab_size, mask_ratio, mask_token_id, special_token_ids, random_token_ratio, unchanged_ratio)`
**Module:** `utils.masking.strategies`

Performs dynamic token masking and corruption according to BERT's strategy.

*📁 src/dl_techniques/utils/masking/strategies.py:21*

#### `apply_motion_blur(image, severity)`
**Module:** `utils.corruption`

Apply motion blur corruption using convolution.

*📁 src/dl_techniques/utils/corruption.py:282*

#### `apply_nms(cls, detections, iou_threshold, score_threshold)`
**Module:** `datasets.patch_transforms`

Apply Non-Maximum Suppression to detection results.

*📁 src/dl_techniques/datasets/patch_transforms.py:300*

#### `apply_pixelate(image, severity)`
**Module:** `utils.corruption`

Apply pixelate corruption using pooling operations.

*📁 src/dl_techniques/utils/corruption.py:353*

#### `apply_quantize(image, severity)`
**Module:** `utils.corruption`

Apply quantization (bit-depth reduction) corruption.

*📁 src/dl_techniques/utils/corruption.py:450*

#### `apply_saturate(image, severity)`
**Module:** `utils.corruption`

Apply saturation corruption.

*📁 src/dl_techniques/utils/corruption.py:531*

#### `apply_shot_noise(image, severity)`
**Module:** `utils.corruption`

Apply shot noise (Poisson-like noise) corruption.

*📁 src/dl_techniques/utils/corruption.py:213*

#### `area(self)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:75*

#### `assess_forecastability(time_series, season_length, auto_pe)`
**Module:** `utils.forecastability_analyzer`

Complete forecastability assessment pipeline.

*📁 src/dl_techniques/utils/forecastability_analyzer.py:517*

#### `augment_series(self, series, augmentations)`
**Module:** `datasets.time_series.generator`

Apply a sequence of augmentations to a time series.

*📁 src/dl_techniques/datasets/time_series/generator.py:1113*

#### `augment_task(self, task, config)`
**Module:** `datasets.arc.arc_converters`

Generate augmented versions of a task.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:463*

#### `auto_permutation_entropy(time_series, max_dim, max_delay)`
**Module:** `utils.forecastability_analyzer`

Calculate PE with automatic parameter selection.

*📁 src/dl_techniques/utils/forecastability_analyzer.py:272*

#### `available_methods(self)`
**Module:** `datasets.time_series.normalizer`

List available normalization methods.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:111*

#### `batch_encode(self, texts, return_tensors)`
**Module:** `utils.tokenizer`

Batch encode multiple texts (alias for __call__ with list input).

*📁 src/dl_techniques/utils/tokenizer.py:371*

#### `batch_generator(data, batch_size)`
**Module:** `utils.alignment.utils`

Generate batches from data.

*📁 src/dl_techniques/utils/alignment/utils.py:479*

#### `bbox_iou(box1, box2, xywh, GIoU, DIoU, CIoU, eps)`
**Module:** `utils.bounding_box`

Calculate IoU, GIoU, DIoU, or CIoU for batches of bounding boxes.

*📁 src/dl_techniques/utils/bounding_box.py:14*

#### `bbox_nms(boxes, scores, iou_threshold, score_threshold, max_outputs, xywh)`
**Module:** `utils.bounding_box`

Perform Non-Maximum Suppression (NMS) on bounding boxes.

*📁 src/dl_techniques/utils/bounding_box.py:413*

#### `body(i, current_canvas)`
**Module:** `datasets.vision.coco`

*📁 src/dl_techniques/datasets/vision/coco.py:470*

#### `build(self)`
**Module:** `datasets.vision.common`

Build MNIST training and validation datasets.

*📁 src/dl_techniques/datasets/vision/common.py:84*

#### `build(self)`
**Module:** `datasets.vision.common`

Build CIFAR-10 training and validation datasets.

*📁 src/dl_techniques/datasets/vision/common.py:221*

#### `build(self)`
**Module:** `datasets.vision.common`

Build CIFAR-100 training and validation datasets.

*📁 src/dl_techniques/datasets/vision/common.py:372*

#### `build(self, input_shape)`
**Module:** `datasets.arc.arc_keras`

Build the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:225*

#### `build(self, input_shape)`
**Module:** `datasets.arc.arc_keras`

Build the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:323*

#### `build(self, input_shape)`
**Module:** `datasets.arc.arc_keras`

Build the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:434*

#### `build_3d_scene_understanding()`
**Module:** `applications.geometric.examples`

3D Scene Understanding with object detection and relationships.

*📁 src/dl_techniques/applications/geometric/examples.py:84*

#### `build_climate_model()`
**Module:** `applications.geometric.examples`

Climate/Weather Prediction Model with spatial transformers.

*📁 src/dl_techniques/applications/geometric/examples.py:471*

#### `build_manipulation_planner()`
**Module:** `applications.geometric.examples`

Robot Manipulation Planning with spatial reasoning.

*📁 src/dl_techniques/applications/geometric/examples.py:348*

#### `build_molecular_property_predictor()`
**Module:** `applications.geometric.examples`

Molecular Property Prediction using spatial molecular graphs.

*📁 src/dl_techniques/applications/geometric/examples.py:293*

#### `build_multi_sensor_fusion()`
**Module:** `applications.geometric.examples`

Multi-Sensor Fusion for Autonomous Systems.

*📁 src/dl_techniques/applications/geometric/examples.py:181*

#### `build_physics_simulator(grid_size)`
**Module:** `applications.geometric.examples`

Physics Simulation Network (CFD, Weather, Molecular Dynamics).

*📁 src/dl_techniques/applications/geometric/examples.py:243*

#### `build_point_cloud_classifier(num_points, num_classes)`
**Module:** `applications.geometric.examples`

Point Cloud Classification (like ModelNet40).

*📁 src/dl_techniques/applications/geometric/examples.py:24*

#### `build_point_cloud_segmentation(num_points, num_classes)`
**Module:** `applications.geometric.examples`

Point Cloud Segmentation (like ShapeNet parts).

*📁 src/dl_techniques/applications/geometric/examples.py:56*

#### `build_procedural_world_generator()`
**Module:** `applications.geometric.examples`

Procedural World Generation for games and simulations.

*📁 src/dl_techniques/applications/geometric/examples.py:533*

#### `build_slam_system()`
**Module:** `applications.geometric.examples`

Simultaneous Localization and Mapping with spatial transformers.

*📁 src/dl_techniques/applications/geometric/examples.py:408*

#### `build_vision_language_model(max_text_len, image_patches)`
**Module:** `applications.geometric.examples`

Vision-Language Model (like CLIP but with spatial reasoning).

*📁 src/dl_techniques/applications/geometric/examples.py:137*

#### `calculate_naive_baselines(time_series, season_length, n_folds, horizon)`
**Module:** `utils.forecastability_analyzer`

Compute naive benchmarks using time series cross-validation.

*📁 src/dl_techniques/utils/forecastability_analyzer.py:318*

#### `calibrate(self, x_calib, y_calib)`
**Module:** `utils.conformal_forecaster`

Compute conformity scores on calibration set.

*📁 src/dl_techniques/utils/conformal_forecaster.py:332*

#### `call(self, inputs, training)`
**Module:** `datasets.arc.arc_keras`

Forward pass of the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:242*

#### `call(self, inputs, training)`
**Module:** `datasets.arc.arc_keras`

Forward pass of the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:336*

#### `call(self, inputs, training)`
**Module:** `datasets.arc.arc_keras`

Forward pass of the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:452*

#### `center_x(self)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:67*

#### `center_x(self)`
**Module:** `datasets.patch_transforms`

*📁 src/dl_techniques/datasets/patch_transforms.py:49*

#### `center_y(self)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:71*

#### `center_y(self)`
**Module:** `datasets.patch_transforms`

*📁 src/dl_techniques/datasets/patch_transforms.py:53*

#### `check_avoidance()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:497*

#### `cka(feats_a, feats_b, kernel_metric, rbf_sigma, unbiased)`
**Module:** `utils.alignment.metrics`

Centered Kernel Alignment (CKA).

*📁 src/dl_techniques/utils/alignment/metrics.py:194*

#### `cknna(feats_a, feats_b, topk, distance_agnostic, unbiased)`
**Module:** `utils.alignment.metrics`

Centered Kernel Alignment with Nearest Neighbor Attention (CKNNA).

*📁 src/dl_techniques/utils/alignment/metrics.py:350*

#### `clean_cache(root_dir, dataset_name)`
**Module:** `datasets.time_series.utils`

Remove cached files.

*📁 src/dl_techniques/datasets/time_series/utils.py:147*

#### `coco_default(cls)`
**Module:** `datasets.vision.coco`

Create default COCO configuration.

*📁 src/dl_techniques/datasets/vision/coco.py:126*

#### `collage(images_batch)`
**Module:** `utils.visualization`

Create a collage of image from a batch

*📁 src/dl_techniques/utils/visualization.py:19*

#### `combine_masks()`
**Module:** `utils.masking.factory`

Combine multiple masks using logical operations.

*📁 src/dl_techniques/utils/masking/factory.py:801*

#### `compare_images(self, images, titles, name, subdir, cmap)`
**Module:** `utils.visualization_manager`

Compare multiple images side by side.

*📁 src/dl_techniques/utils/visualization_manager.py:266*

#### `compute_alignment_matrix(self, features_list_a, features_list_b)`
**Module:** `utils.alignment.alignment`

Compute alignment matrix for multiple models.

*📁 src/dl_techniques/utils/alignment/alignment.py:226*

#### `compute_alignment_matrix(x_feat_list, y_feat_list, metric, topk, normalize, precise)`
**Module:** `utils.alignment.utils`

Compute alignment matrix between all pairs of feature sets.

*📁 src/dl_techniques/utils/alignment/utils.py:124*

#### `compute_dataset_statistics(self, split, subset)`
**Module:** `datasets.arc.arc_utilities`

Compute comprehensive statistics for a dataset split.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:379*

#### `compute_iou(bbox1, bbox2)`
**Module:** `datasets.patch_transforms`

Compute Intersection over Union (IoU) of two bounding boxes.

*📁 src/dl_techniques/datasets/patch_transforms.py:273*

#### `compute_mask(self, inputs, mask)`
**Module:** `datasets.arc.arc_keras`

Compute mask for the layer.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:471*

#### `compute_output_shape(self, input_shape)`
**Module:** `datasets.arc.arc_keras`

Compute output shape.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:262*

#### `compute_output_shape(self, input_shape)`
**Module:** `datasets.arc.arc_keras`

Compute output shape.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:370*

#### `compute_output_shape(self, input_shape)`
**Module:** `datasets.arc.arc_keras`

Compute output shape.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:488*

#### `compute_pairwise_alignment(self, features_a, features_b, return_layer_indices)`
**Module:** `utils.alignment.alignment`

Compute alignment between two feature sets.

*📁 src/dl_techniques/utils/alignment/alignment.py:180*

#### `compute_prediction_entropy(y_pred, temperature, epsilon)`
**Module:** `utils.tensors`

Compute the entropy of predictions for calibration analysis.

*📁 src/dl_techniques/utils/tensors.py:269*

#### `compute_score(x_feats, y_feats, metric, topk, normalize)`
**Module:** `utils.alignment.utils`

Find best alignment score across layer combinations.

*📁 src/dl_techniques/utils/alignment/utils.py:44*

#### `compute_statistics(features)`
**Module:** `utils.alignment.utils`

Compute statistics of features.

*📁 src/dl_techniques/utils/alignment/utils.py:453*

#### `cond(i, _)`
**Module:** `datasets.vision.coco`

*📁 src/dl_techniques/datasets/vision/coco.py:467*

#### `convert_numpy_to_python(obj)`
**Module:** `utils.convert`

Recursively convert numpy types to Python native types for JSON serialization.

*📁 src/dl_techniques/utils/convert.py:5*

#### `count_available_files(directories, extensions, max_files)`
**Module:** `utils.filesystem`

Count available files without loading them into memory. Used for logging and steps_per_epoch calculation.

*📁 src/dl_techniques/utils/filesystem.py:12*

#### `create_alignment_filename(output_dir, dataset, modelset, modality_x, pool_x, prompt_x, modality_y, pool_y, prompt_y, metric, topk)`
**Module:** `utils.alignment.utils`

Create standardized filename for alignment results.

*📁 src/dl_techniques/utils/alignment/utils.py:400*

#### `create_arc_data_generator(dataset_path, split, batch_size, shuffle, normalize_inputs)`
**Module:** `datasets.arc.arc_keras`

Create an ARC data generator for Keras training.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:625*

#### `create_avoidance_zones()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:469*

#### `create_banded_mask(seq_len, band_width, dtype)`
**Module:** `utils.masking.factory`

Create a banded attention mask.

*📁 src/dl_techniques/utils/masking/factory.py:373*

#### `create_block_diagonal_mask(seq_len, block_size, dtype)`
**Module:** `utils.masking.factory`

Create a block diagonal attention mask.

*📁 src/dl_techniques/utils/masking/factory.py:289*

#### `create_causal_mask(size)`
**Module:** `utils.tensors`

Create causal mask for attention.

*📁 src/dl_techniques/utils/tensors.py:171*

#### `create_causal_mask(seq_len, dtype)`
**Module:** `utils.masking.factory`

Create a causal (lower triangular) attention mask.

*📁 src/dl_techniques/utils/masking/factory.py:172*

#### `create_coco_dataset(img_size, batch_size, use_detection, use_segmentation, segmentation_classes, augment_data, class_names)`
**Module:** `datasets.vision.coco`

Factory function to create a ready-to-use COCO dataset pipeline.

*📁 src/dl_techniques/datasets/vision/coco.py:688*

#### `create_dataset_builder(dataset_name, config, use_rgb)`
**Module:** `datasets.vision.common`

Factory function to create the appropriate dataset builder.

*📁 src/dl_techniques/datasets/vision/common.py:482*

#### `create_datasets(self)`
**Module:** `datasets.vision.coco`

Create the final optimized training and validation datasets.

*📁 src/dl_techniques/datasets/vision/coco.py:610*

#### `create_dummy_coco_dataset(num_samples, img_size, num_classes, max_boxes, min_boxes, segmentation_classes)`
**Module:** `datasets.vision.coco`

Create a dummy COCO-style dataset for testing and development.

*📁 src/dl_techniques/datasets/vision/coco.py:139*

#### `create_feature_filename(output_dir, dataset, subset, model_name, pool, prompt, caption_idx)`
**Module:** `utils.alignment.utils`

Create standardized filename for features.

*📁 src/dl_techniques/utils/alignment/utils.py:357*

#### `create_figure(self, size)`
**Module:** `utils.visualization_manager`

Create a new figure with proper settings.

*📁 src/dl_techniques/utils/visualization_manager.py:149*

#### `create_global_local_masks(seq_len, sliding_window, dtype)`
**Module:** `utils.masking.factory`

Create masks for combined global and local attention patterns.

*📁 src/dl_techniques/utils/masking/factory.py:249*

#### `create_grid()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:310*

#### `create_inference_engine(model_path, config)`
**Module:** `utils.inference`

Create inference engine from saved model.

*📁 src/dl_techniques/utils/inference.py:584*

#### `create_instance_separation_mask(mask_predictions, separation_threshold, dtype)`
**Module:** `utils.masking.factory`

Create masks to enforce separation between instance predictions.

*📁 src/dl_techniques/utils/masking/factory.py:568*

#### `create_mask(mask_type, config)`
**Module:** `utils.masking.factory`

Universal interface for creating masks.

*📁 src/dl_techniques/utils/masking/factory.py:605*

#### `create_padding_mask(padding_mask, dtype)`
**Module:** `utils.masking.factory`

Create 2D attention mask from 1D padding mask.

*📁 src/dl_techniques/utils/masking/factory.py:415*

#### `create_patch_grid(image_width, image_height, patch_size, overlap)`
**Module:** `datasets.patch_transforms`

Convenience function to create patch grid.

*📁 src/dl_techniques/datasets/patch_transforms.py:595*

#### `create_puzzle_grid_comparison(self, puzzles, max_puzzles, save_path)`
**Module:** `datasets.arc.arc_utilities`

Create a grid comparison of multiple puzzles.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:649*

#### `create_query_interaction_mask(num_queries, interaction_type, hierarchy_levels, dtype)`
**Module:** `utils.masking.factory`

Create masks for controlling interactions between queries.

*📁 src/dl_techniques/utils/masking/factory.py:518*

#### `create_random_graph(num_nodes, edge_probability, add_self_loops, symmetric)`
**Module:** `utils.graphs`

Create a random graph for testing or synthetic experiments.

*📁 src/dl_techniques/utils/graphs.py:201*

#### `create_random_mask(seq_len, mask_probability, seed, dtype)`
**Module:** `utils.masking.factory`

Create a random attention mask.

*📁 src/dl_techniques/utils/masking/factory.py:331*

#### `create_simple_arc_model(vocab_size, seq_len, embed_dim, num_layers, num_heads)`
**Module:** `datasets.arc.arc_keras`

Create a simple transformer model for ARC tasks.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:653*

#### `create_sliding_window_mask(seq_len, window_size, dtype)`
**Module:** `utils.masking.factory`

Create a sliding window attention mask with causal constraint.

*📁 src/dl_techniques/utils/masking/factory.py:203*

#### `create_sliding_windows(data, window_size, horizon, stride)`
**Module:** `datasets.time_series.pipeline`

Create sliding windows from a numpy array using efficient memory views.

*📁 src/dl_techniques/datasets/time_series/pipeline.py:23*

#### `create_spatial_mask(height, width, attention_regions, mask_mode, dtype)`
**Module:** `utils.masking.factory`

Create spatial attention mask for image regions.

*📁 src/dl_techniques/utils/masking/factory.py:480*

#### `create_sut_crack_dataset(data_dir)`
**Module:** `datasets.sut`

Convenience function to create optimized SUT-Crack dataset.

*📁 src/dl_techniques/datasets/sut.py:1283*

#### `create_tf_dataset(self, batch_size, shuffle, repeat, prefetch_buffer, subset, cache, num_parallel_calls)`
**Module:** `datasets.sut`

Create an optimized TensorFlow dataset with high-performance ordering.

*📁 src/dl_techniques/datasets/sut.py:1023*

#### `create_train_val_test_datasets(train_df, val_df, test_df, window_config, pipeline_config)`
**Module:** `datasets.time_series.pipeline`

Create train, validation, and test tf.data.Datasets from DataFrames.

*📁 src/dl_techniques/datasets/time_series/pipeline.py:266*

#### `create_valid_query_mask(num_queries, valid_queries, dtype)`
**Module:** `utils.masking.factory`

Create a mask for valid object queries in segmentation.

*📁 src/dl_techniques/utils/masking/factory.py:444*

#### `create_vqa_dataset(data_samples, processor, batch_size, shuffle, conversation_format)`
**Module:** `datasets.vqa_dataset`

Create VQA dataset from samples.

*📁 src/dl_techniques/datasets/vqa_dataset.py:492*

#### `cycle_knn(feats_a, feats_b, topk)`
**Module:** `utils.alignment.metrics`

Cycle KNN: A->B nearest neighbors, then query B->A nearest neighbors.

*📁 src/dl_techniques/utils/alignment/metrics.py:70*

#### `dataset(self)`
**Module:** `datasets.universal_dataset_loader`

Get the underlying Hugging Face dataset object.

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:184*

#### `decode(self, token_ids)`
**Module:** `datasets.vqa_dataset`

Decode token IDs to text.

*📁 src/dl_techniques/datasets/vqa_dataset.py:40*

#### `decode(self, token_ids)`
**Module:** `datasets.vqa_dataset`

Decode token IDs back to text.

*📁 src/dl_techniques/datasets/vqa_dataset.py:81*

#### `decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces)`
**Module:** `utils.tokenizer`

Decode token IDs back to text.

*📁 src/dl_techniques/utils/tokenizer.py:447*

#### `demonstrate_applications()`
**Module:** `applications.geometric.examples`

Demonstrate building and using these applications.

*📁 src/dl_techniques/applications/geometric/examples.py:598*

#### `denormalize_bbox(normalized_bbox, image_width, image_height)`
**Module:** `datasets.patch_transforms`

Convert normalized bounding box back to pixel coordinates.

*📁 src/dl_techniques/datasets/patch_transforms.py:153*

#### `depthwise_gaussian_kernel(channels, kernel_size, nsig, dtype)`
**Module:** `utils.tensors`

Create a depthwise Gaussian kernel.

*📁 src/dl_techniques/utils/tensors.py:234*

#### `dihedral_transform(arr, tid)`
**Module:** `datasets.arc.arc_converters`

8 dihedral symmetries by rotate, flip and mirror

*📁 src/dl_techniques/datasets/arc/arc_converters.py:67*

#### `download(self)`
**Module:** `datasets.time_series.base`

Download the raw dataset files.

*📁 src/dl_techniques/datasets/time_series/base.py:77*

#### `download(self, group)`
**Module:** `datasets.time_series.m4`

Download M4 dataset files.

*📁 src/dl_techniques/datasets/time_series/m4.py:170*

#### `download(self, group)`
**Module:** `datasets.time_series.long_horizon`

Download the long horizon datasets.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:196*

#### `download(self)`
**Module:** `datasets.time_series.favorita`

Download and extract dataset files.

*📁 src/dl_techniques/datasets/time_series/favorita.py:95*

#### `download_file(directory, source_url, decompress, filename, chunk_size, timeout, progress_callback)`
**Module:** `datasets.time_series.utils`

Download a file from a URL with progress tracking.

*📁 src/dl_techniques/datasets/time_series/utils.py:61*

#### `draw_figure_to_buffer(fig, dpi)`
**Module:** `utils.visualization`

draw figure into numpy buffer

*📁 src/dl_techniques/utils/visualization.py:47*

#### `edit_distance_knn(feats_a, feats_b, topk)`
**Module:** `utils.alignment.metrics`

Edit distance between k-nearest neighbor orderings.

*📁 src/dl_techniques/utils/alignment/metrics.py:314*

#### `empty_mask()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:258*

#### `empty_mask()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:665*

#### `encode(self, text)`
**Module:** `datasets.vqa_dataset`

Encode text to token IDs.

*📁 src/dl_techniques/datasets/vqa_dataset.py:28*

#### `encode(self, text)`
**Module:** `datasets.vqa_dataset`

Encode text using character-level tokenization.

*📁 src/dl_techniques/datasets/vqa_dataset.py:65*

#### `encode(self, text, return_tensors)`
**Module:** `utils.tokenizer`

Encode a single text (alias for __call__ with string input).

*📁 src/dl_techniques/utils/tokenizer.py:403*

#### `ensure_directory(path)`
**Module:** `datasets.time_series.utils`

Ensure directory exists.

*📁 src/dl_techniques/datasets/time_series/utils.py:129*

#### `evaluate_coverage(self, x_test, y_test, per_horizon)`
**Module:** `utils.conformal_forecaster`

Evaluate empirical coverage probability.

*📁 src/dl_techniques/utils/conformal_forecaster.py:571*

#### `exp_map_0(self, v, c)`
**Module:** `utils.geometry.poincare_math`

Exponential map at the origin: Tangent space (Euclidean) → Manifold (Hyperbolic).

*📁 src/dl_techniques/utils/geometry/poincare_math.py:141*

#### `extract_file(filepath, directory, remove_archive)`
**Module:** `datasets.time_series.utils`

Extract a compressed archive to a specified directory.

*📁 src/dl_techniques/datasets/time_series/utils.py:23*

#### `extract_layer_features(model, inputs, layer_names, batch_size)`
**Module:** `utils.alignment.utils`

Extract features from multiple layers of a Keras model.

*📁 src/dl_techniques/utils/alignment/utils.py:214*

#### `extract_mask()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:626*

#### `extract_patch(self, image, patch_info)`
**Module:** `datasets.patch_transforms`

Extract a patch from the full image.

*📁 src/dl_techniques/datasets/patch_transforms.py:237*

#### `extract_patches()`
**Module:** `datasets.sut`

Extract patches when centers are available.

*📁 src/dl_techniques/datasets/sut.py:569*

#### `extract_single_patch(i)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:572*

#### `fit(self, X, column_names)`
**Module:** `datasets.tabular`

Fit preprocessing transformations.

*📁 src/dl_techniques/datasets/tabular.py:42*

#### `fit(self, data)`
**Module:** `datasets.time_series.normalizer`

Fit the normalizer to the data.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:131*

#### `fit_transform(self, X, column_names)`
**Module:** `datasets.tabular`

Fit and transform data in one step.

*📁 src/dl_techniques/datasets/tabular.py:171*

#### `fit_transform(self, data)`
**Module:** `datasets.time_series.normalizer`

Fit to data, then transform it.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:412*

#### `forecast_value_added(time_series, model_predictions, season_length, metric)`
**Module:** `utils.forecastability_analyzer`

Calculate Forecast Value Added (FVA) for model comparison.

*📁 src/dl_techniques/utils/forecastability_analyzer.py:414*

#### `from_class_names(cls, class_names)`
**Module:** `datasets.vision.coco`

Create config from list of class names.

*📁 src/dl_techniques/datasets/vision/coco.py:116*

#### `from_models(reference_models, data, layer_names, metric, topk, normalize, batch_size)`
**Module:** `utils.alignment.alignment`

Create alignment scorer from Keras models.

*📁 src/dl_techniques/utils/alignment/alignment.py:323*

#### `from_xml(cls, xml_path, image_path, mask_path)`
**Module:** `datasets.sut`

Create annotation from a Pascal VOC XML file.

*📁 src/dl_techniques/datasets/sut.py:133*

#### `full_image_to_patch_bbox(full_bbox, patch_info)`
**Module:** `datasets.patch_transforms`

Transform bounding box from full image coordinates to patch coordinates.

*📁 src/dl_techniques/datasets/patch_transforms.py:113*

#### `gaussian_kernel(kernel_size, nsig)`
**Module:** `utils.tensors`

Build a 2D Gaussian kernel array.

*📁 src/dl_techniques/utils/tensors.py:206*

#### `gaussian_probability(y, mu, sigma)`
**Module:** `utils.tensors`

Compute Gaussian probability density using Keras operations.

*📁 src/dl_techniques/utils/tensors.py:431*

#### `gen()`
**Module:** `datasets.time_series.pipeline`

*📁 src/dl_techniques/datasets/time_series/pipeline.py:227*

#### `generate_all_patterns(self)`
**Module:** `datasets.time_series.generator`

Generate all available time series patterns.

*📁 src/dl_techniques/datasets/time_series/generator.py:1023*

#### `generate_centers()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:290*

#### `generate_checker(n_samples, n_classes, noise_level, random_state)`
**Module:** `datasets.simple_2d`

Generate a checkerboard dataset.

*📁 src/dl_techniques/datasets/simple_2d.py:469*

#### `generate_circles(n_samples, noise_level, factor, random_state)`
**Module:** `datasets.simple_2d`

Generate a concentric circles dataset.

*📁 src/dl_techniques/datasets/simple_2d.py:210*

#### `generate_clusters(n_samples, centers, n_features, cluster_std, center_box, random_state, return_centers)`
**Module:** `datasets.simple_2d`

Generate Gaussian clusters dataset.

*📁 src/dl_techniques/datasets/simple_2d.py:236*

#### `generate_custom_pattern(self, pattern_type)`
**Module:** `datasets.time_series.generator`

Generate a custom time series pattern with specified parameters.

*📁 src/dl_techniques/datasets/time_series/generator.py:1064*

#### `generate_dataset(dataset_type, n_samples, noise_level, random_state, return_centers)`
**Module:** `datasets.simple_2d`

Generate a dataset of the specified type.

*📁 src/dl_techniques/datasets/simple_2d.py:79*

#### `generate_gaussian_quantiles(n_samples, n_classes, random_state)`
**Module:** `datasets.simple_2d`

Generate a dataset with concentric normal distributions separated by quantiles.

*📁 src/dl_techniques/datasets/simple_2d.py:387*

#### `generate_mixture(n_samples, centers, n_classes, cluster_std, random_state)`
**Module:** `datasets.simple_2d`

Generate a dataset with overlapping clusters that belongs to different classes.

*📁 src/dl_techniques/datasets/simple_2d.py:413*

#### `generate_moons(n_samples, noise_level, random_state)`
**Module:** `datasets.simple_2d`

Generate a moons dataset (two interleaving half-circles).

*📁 src/dl_techniques/datasets/simple_2d.py:187*

#### `generate_patches(self, image_width, image_height)`
**Module:** `datasets.patch_transforms`

Generate list of patch locations covering the full image.

*📁 src/dl_techniques/datasets/patch_transforms.py:183*

#### `generate_random_pattern(self, category)`
**Module:** `datasets.time_series.generator`

Generate a random time series pattern, optionally from a specific category.

*📁 src/dl_techniques/datasets/time_series/generator.py:1038*

#### `generate_spiral(n_samples, noise_level, random_state)`
**Module:** `datasets.simple_2d`

Generate intertwined spirals dataset.

*📁 src/dl_techniques/datasets/simple_2d.py:335*

#### `generate_task_data(self, task_name)`
**Module:** `datasets.time_series.generator`

Generate time series data for a specific task.

*📁 src/dl_techniques/datasets/time_series/generator.py:1003*

#### `generate_xor(n_samples, noise_level, random_state)`
**Module:** `datasets.simple_2d`

Generate an XOR-like dataset (points in opposite quadrants belong to the same class).

*📁 src/dl_techniques/datasets/simple_2d.py:285*

#### `generator()`
**Module:** `datasets.vision.coco`

*📁 src/dl_techniques/datasets/vision/coco.py:158*

#### `generator_factory()`
**Module:** `datasets.universal_dataset_loader`

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:379*

#### `get_all_corruption_types()`
**Module:** `utils.corruption`

Get a list of all available corruption types.

*📁 src/dl_techniques/utils/corruption.py:625*

#### `get_all_severity_levels()`
**Module:** `utils.corruption`

Get a list of all available severity levels.

*📁 src/dl_techniques/utils/corruption.py:636*

#### `get_cache_path(root_dir, dataset_name, filename)`
**Module:** `datasets.time_series.utils`

Get standardized cache path.

*📁 src/dl_techniques/datasets/time_series/utils.py:136*

#### `get_class_names(self)`
**Module:** `datasets.vision.common`

Get class names for visualization.

*📁 src/dl_techniques/datasets/vision/common.py:178*

#### `get_class_names(self)`
**Module:** `datasets.vision.common`

Get class names for visualization.

*📁 src/dl_techniques/datasets/vision/common.py:316*

#### `get_class_names(self)`
**Module:** `datasets.vision.common`

Get class names for visualization.

*📁 src/dl_techniques/datasets/vision/common.py:468*

#### `get_config(self, group)`
**Module:** `datasets.time_series.base`

Get the configuration for a specific dataset group.

*📁 src/dl_techniques/datasets/time_series/base.py:111*

#### `get_config(self)`
**Module:** `datasets.arc.arc_keras`

Get layer configuration.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:274*

#### `get_config(self)`
**Module:** `datasets.arc.arc_keras`

Get layer configuration.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:382*

#### `get_config(self)`
**Module:** `datasets.arc.arc_keras`

Get layer configuration.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:500*

#### `get_config(self)`
**Module:** `datasets.arc.arc_keras`

Get metric configuration.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:609*

#### `get_corruption_function(corruption_type)`
**Module:** `utils.corruption`

Get the corruption function for a given corruption type.

*📁 src/dl_techniques/utils/corruption.py:563*

#### `get_dataset_info(self)`
**Module:** `datasets.sut`

Get comprehensive dataset statistics - same as original.

*📁 src/dl_techniques/datasets/sut.py:1238*

#### `get_dataset_info(dataset)`
**Module:** `datasets.time_series.pipeline`

Get information about a tf.data.Dataset.

*📁 src/dl_techniques/datasets/time_series/pipeline.py:391*

#### `get_dataset_info(dataset_name, use_rgb)`
**Module:** `datasets.vision.common`

Get metadata information for a dataset.

*📁 src/dl_techniques/datasets/vision/common.py:537*

#### `get_dataset_info(self)`
**Module:** `datasets.vision.coco`

Get dataset configuration information.

*📁 src/dl_techniques/datasets/vision/coco.py:668*

#### `get_diagnostics(self)`
**Module:** `utils.conformal_forecaster`

Get comprehensive diagnostic information.

*📁 src/dl_techniques/utils/conformal_forecaster.py:708*

#### `get_efficiency_metrics(self, x_test, per_horizon)`
**Module:** `utils.conformal_forecaster`

Compute efficiency metrics (interval sharpness).

*📁 src/dl_techniques/utils/conformal_forecaster.py:643*

#### `get_generator(self, transform_fn, filter_fn, columns, skip_errors)`
**Module:** `datasets.universal_dataset_loader`

Create a Python generator yielding processed data items.

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:206*

#### `get_info(self, group)`
**Module:** `datasets.time_series.base`

Get detailed information about a dataset group.

*📁 src/dl_techniques/datasets/time_series/base.py:321*

#### `get_m4_horizon(group)`
**Module:** `datasets.time_series.m4`

Get the standard forecast horizon for an M4 group.

*📁 src/dl_techniques/datasets/time_series/m4.py:474*

#### `get_m4_seasonality(group)`
**Module:** `datasets.time_series.m4`

Get the seasonality period for an M4 group.

*📁 src/dl_techniques/datasets/time_series/m4.py:492*

#### `get_mask_info()`
**Module:** `utils.masking.factory`

Get information about all available mask types.

*📁 src/dl_techniques/utils/masking/factory.py:929*

#### `get_naive2_forecasts(self, group)`
**Module:** `datasets.time_series.m4`

Get Naive2 baseline forecasts for comparison.

*📁 src/dl_techniques/datasets/time_series/m4.py:425*

#### `get_save_path(self, name, subdir)`
**Module:** `utils.visualization_manager`

Get full save path for a visualization.

*📁 src/dl_techniques/utils/visualization_manager.py:97*

#### `get_scores(self)`
**Module:** `utils.alignment.alignment`

Get all logged scores.

*📁 src/dl_techniques/utils/alignment/alignment.py:448*

#### `get_special_token_ids(encoding_name)`
**Module:** `utils.tokenizer`

Get valid special token IDs for a Tiktoken encoding.

*📁 src/dl_techniques/utils/tokenizer.py:42*

#### `get_splits(self, group, normalize, multivariate)`
**Module:** `datasets.time_series.long_horizon`

Load and split a dataset into train/val/test sets.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:428*

#### `get_statistics(self)`
**Module:** `datasets.time_series.normalizer`

Get the fitted statistics.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:416*

#### `get_supported_metrics(self)`
**Module:** `utils.alignment.alignment`

Get list of supported metrics.

*📁 src/dl_techniques/utils/alignment/alignment.py:313*

#### `get_task_categories(self)`
**Module:** `datasets.time_series.generator`

Get list of all task categories.

*📁 src/dl_techniques/datasets/time_series/generator.py:981*

#### `get_task_names(self)`
**Module:** `datasets.time_series.generator`

Get list of all available task names.

*📁 src/dl_techniques/datasets/time_series/generator.py:973*

#### `get_tasks_by_category(self, category)`
**Module:** `datasets.time_series.generator`

Get all task names belonging to a specific category.

*📁 src/dl_techniques/datasets/time_series/generator.py:989*

#### `get_test_data(self)`
**Module:** `datasets.vision.common`

Get test data for analysis and visualization.

*📁 src/dl_techniques/datasets/vision/common.py:167*

#### `get_test_data(self)`
**Module:** `datasets.vision.common`

Get test data for analysis and visualization.

*📁 src/dl_techniques/datasets/vision/common.py:305*

#### `get_test_data(self)`
**Module:** `datasets.vision.common`

Get test data for analysis and visualization.

*📁 src/dl_techniques/datasets/vision/common.py:457*

#### `gram_matrix(weights)`
**Module:** `utils.tensors`

Compute the Gram matrix (W * W^T) with improved numerical stability.

*📁 src/dl_techniques/utils/tensors.py:74*

#### `grid_to_sequence(self, grid, apply_translation)`
**Module:** `datasets.arc.arc_converters`

Convert a 2D grid to a flattened sequence with padding and EOS tokens.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:194*

#### `h_flip_fn()`
**Module:** `datasets.vision.coco`

*📁 src/dl_techniques/datasets/vision/coco.py:528*

#### `height(self)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:63*

#### `height(self)`
**Module:** `datasets.patch_transforms`

*📁 src/dl_techniques/datasets/patch_transforms.py:45*

#### `image_file_generator(directories, extensions, max_files, patches_per_image)`
**Module:** `utils.filesystem`

Generator that yields image file paths on-the-fly without storing them in memory.

*📁 src/dl_techniques/utils/filesystem.py:54*

#### `inverse_dihedral_transform(arr, tid)`
**Module:** `datasets.arc.arc_converters`

Apply inverse dihedral transformation

*📁 src/dl_techniques/datasets/arc/arc_converters.py:90*

#### `inverse_transform(self, data)`
**Module:** `datasets.time_series.normalizer`

Inverse transform normalized data back to original scale.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:315*

#### `inverse_transform_quantile_uniform(self, data)`
**Module:** `datasets.time_series.normalizer`

Helper for inverse quantile transform.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:380*

#### `json_to_task_data(self, json_data, task_id)`
**Module:** `datasets.arc.arc_converters`

Convert JSON data to ARCTaskData format.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:153*

#### `lcs_knn(feats_a, feats_b, topk)`
**Module:** `utils.alignment.metrics`

Longest Common Subsequence (LCS) of k-nearest neighbors.

*📁 src/dl_techniques/utils/alignment/metrics.py:165*

#### `length(vectors)`
**Module:** `utils.tensors`

Compute length of capsule vectors.

*📁 src/dl_techniques/utils/tensors.py:463*

#### `list_groups(self)`
**Module:** `datasets.time_series.base`

List all available dataset groups.

*📁 src/dl_techniques/datasets/time_series/base.py:128*

#### `load(self, group, cache)`
**Module:** `datasets.time_series.base`

Load the dataset into pandas DataFrames.

*📁 src/dl_techniques/datasets/time_series/base.py:90*

#### `load(self, group, cache, include_test)`
**Module:** `datasets.time_series.m4`

Load a specific M4 frequency group.

*📁 src/dl_techniques/datasets/time_series/m4.py:199*

#### `load(self, group, cache, multivariate)`
**Module:** `datasets.time_series.long_horizon`

Load a specific long-horizon dataset.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:232*

#### `load(self, group, cache)`
**Module:** `datasets.time_series.favorita`

Load the Favorita dataset using chunked processing.

*📁 src/dl_techniques/datasets/time_series/favorita.py:121*

#### `load_cauldron_from_json(json_path)`
**Module:** `datasets.vqa_dataset`

Load Cauldron dataset from JSON file.

*📁 src/dl_techniques/datasets/vqa_dataset.py:562*

#### `load_cauldron_sample()`
**Module:** `datasets.vqa_dataset`

Load sample data in Cauldron format for testing.

*📁 src/dl_techniques/datasets/vqa_dataset.py:524*

#### `load_ecl(root_dir, multivariate)`
**Module:** `datasets.time_series.long_horizon`

Load ECL (Electricity) dataset.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:523*

#### `load_ett(variant, root_dir, multivariate)`
**Module:** `datasets.time_series.long_horizon`

Load ETT (Electricity Transformer Temperature) dataset.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:471*

#### `load_favorita(group, root_dir)`
**Module:** `datasets.time_series.favorita`

Convenience function to load Favorita data.

*📁 src/dl_techniques/datasets/time_series/favorita.py:310*

#### `load_features(load_path)`
**Module:** `utils.alignment.utils`

Load features from disk.

*📁 src/dl_techniques/utils/alignment/utils.py:300*

#### `load_identifiers_map(self)`
**Module:** `datasets.arc.arc_utilities`

Load the puzzle identifiers mapping.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:199*

#### `load_imagenet(split, batch_size, image_size, shuffle_buffer_size, prefetch_buffer_size, cache, data_dir)`
**Module:** `datasets.vision.imagenet`

Load ImageNet dataset using TensorFlow Datasets.

*📁 src/dl_techniques/datasets/vision/imagenet.py:9*

#### `load_m4(group, root_dir, include_test)`
**Module:** `datasets.time_series.m4`

Load M4 dataset with default settings.

*📁 src/dl_techniques/datasets/time_series/m4.py:448*

#### `load_mask()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:246*

#### `load_puzzles(self, split, subset, include_augmented)`
**Module:** `datasets.arc.arc_utilities`

Load puzzles as structured objects.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:251*

#### `load_raw(self, group)`
**Module:** `datasets.time_series.long_horizon`

Load raw CSV data without preprocessing.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:398*

#### `load_raw(self, filename)`
**Module:** `datasets.time_series.favorita`

Load a raw CSV file directly.

*📁 src/dl_techniques/datasets/time_series/favorita.py:301*

#### `load_split_data(self, split, subset)`
**Module:** `datasets.arc.arc_utilities`

Load raw data arrays for a specific split and subset.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:228*

#### `load_split_metadata(self, split)`
**Module:** `datasets.arc.arc_utilities`

Load metadata for a specific split.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:214*

#### `load_tasks_from_directory(self, directory_path)`
**Module:** `datasets.arc.arc_converters`

Load all tasks from a directory of JSON files.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:170*

#### `load_train_test_split(self, group)`
**Module:** `datasets.time_series.m4`

Load M4 data with explicit train/test split.

*📁 src/dl_techniques/datasets/time_series/m4.py:371*

#### `load_weather(root_dir, multivariate)`
**Module:** `datasets.time_series.long_horizon`

Load Weather dataset.

*📁 src/dl_techniques/datasets/time_series/long_horizon.py:505*

#### `log_map_0(self, y, c)`
**Module:** `utils.geometry.poincare_math`

Logarithmic map at the origin: Manifold (Hyperbolic) → Tangent space (Euclidean).

*📁 src/dl_techniques/utils/geometry/poincare_math.py:180*

#### `make_tf_dataset(df, window_size, horizon, batch_size, shuffle, target_col, id_col, time_col, feature_cols, pipeline_config)`
**Module:** `datasets.time_series.pipeline`

Convert a pandas DataFrame to a tf.data.Dataset using a memory-efficient generator.

*📁 src/dl_techniques/datasets/time_series/pipeline.py:120*

#### `make_tf_dataset_from_arrays(x_data, y_data, pipeline_config)`
**Module:** `datasets.time_series.pipeline`

Create a tf.data.Dataset from pre-computed arrays.

*📁 src/dl_techniques/datasets/time_series/pipeline.py:210*

#### `measure(metric)`
**Module:** `utils.alignment.metrics`

Compute alignment using the specified metric.

*📁 src/dl_techniques/utils/alignment/metrics.py:42*

#### `merge_task_lists(self, task_lists, deduplicate)`
**Module:** `datasets.arc.arc_converters`

Merge multiple lists of tasks.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:388*

#### `mobius_add(self, x, y, c)`
**Module:** `utils.geometry.poincare_math`

Möbius addition: Hyperbolic equivalent of vector addition in Euclidean space.

*📁 src/dl_techniques/utils/geometry/poincare_math.py:222*

#### `mutual_knn(feats_a, feats_b, topk)`
**Module:** `utils.alignment.metrics`

Mutual KNN: Intersection of nearest neighbors from both representations.

*📁 src/dl_techniques/utils/alignment/metrics.py:108*

#### `no_avoidance_zones()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:479*

#### `no_bboxes()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:729*

#### `no_bboxes_available()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:738*

#### `no_negative()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:542*

#### `no_positive()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:440*

#### `normalize_adjacency_row(adj_matrix, add_self_loops)`
**Module:** `utils.graphs`

Row normalization of adjacency matrix: D^{-1} A.

*📁 src/dl_techniques/utils/graphs.py:93*

#### `normalize_adjacency_symmetric(adj_matrix, add_self_loops)`
**Module:** `utils.graphs`

Symmetric normalization of adjacency matrix: D^{-1/2} A D^{-1/2}.

*📁 src/dl_techniques/utils/graphs.py:16*

#### `normalize_bbox(bbox, image_width, image_height)`
**Module:** `datasets.patch_transforms`

Normalize bounding box coordinates to [0, 1] range.

*📁 src/dl_techniques/datasets/patch_transforms.py:138*

#### `normalize_features(feats, axis)`
**Module:** `utils.alignment.utils`

L2-normalize features.

*📁 src/dl_techniques/utils/alignment/utils.py:194*

#### `num_classes(self)`
**Module:** `datasets.vision.coco`

Get number of classes.

*📁 src/dl_techniques/datasets/vision/coco.py:106*

#### `on_epoch_end(self, epoch, logs)`
**Module:** `callbacks.analyzer_callback`

Runs the model analysis at the end of an epoch.

*📁 src/dl_techniques/callbacks/analyzer_callback.py:89*

#### `on_epoch_end(self)`
**Module:** `datasets.vqa_dataset`

Shuffle indices at the end of each epoch if shuffle is enabled.

*📁 src/dl_techniques/datasets/vqa_dataset.py:484*

#### `on_epoch_end(self)`
**Module:** `datasets.arc.arc_keras`

Shuffle indices at the end of each epoch.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:186*

#### `on_epoch_end(self, epoch, model, logs)`
**Module:** `utils.alignment.alignment`

Compute and log alignment score.

*📁 src/dl_techniques/utils/alignment/alignment.py:404*

#### `patch_to_full_image_bbox(patch_bbox, patch_info)`
**Module:** `datasets.patch_transforms`

Transform bounding box from patch coordinates to full image coordinates.

*📁 src/dl_techniques/datasets/patch_transforms.py:88*

#### `permutation_entropy(time_series, embedding_dim, delay)`
**Module:** `utils.forecastability_analyzer`

Calculate Permutation Entropy to measure complexity.

*📁 src/dl_techniques/utils/forecastability_analyzer.py:188*

#### `plot_confusion_matrices(models, x_test, y_test, class_names, output_path, figsize, cmap)`
**Module:** `utils.visualization`

Plot confusion matrices for multiple models side by side.

*📁 src/dl_techniques/utils/visualization.py:71*

#### `plot_confusion_matrices_comparison(self, y_true, model_predictions, name, subdir, figsize, normalize, class_names)`
**Module:** `utils.visualization_manager`

Plot confusion matrices for multiple models side by side for comparison.

*📁 src/dl_techniques/utils/visualization_manager.py:303*

#### `plot_dataset_statistics(self, stats, save_path)`
**Module:** `datasets.arc.arc_utilities`

Plot dataset statistics.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:582*

#### `plot_history(self, histories, metrics, name, subdir, title)`
**Module:** `utils.visualization_manager`

Plot training history metrics.

*📁 src/dl_techniques/utils/visualization_manager.py:208*

#### `plot_matrix(self, matrix, title, xlabel, ylabel, name, subdir, annot, fmt, cmap)`
**Module:** `utils.visualization_manager`

Plot and save a matrix visualization (e.g., confusion matrix, correlation matrix).

*📁 src/dl_techniques/utils/visualization_manager.py:163*

#### `plot_puzzle(self, puzzle, max_examples, figsize, save_path)`
**Module:** `datasets.arc.arc_utilities`

Plot a complete puzzle with all its examples.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:510*

#### `plot_scores(self, save_path)`
**Module:** `utils.alignment.alignment`

Plot alignment scores over training.

*📁 src/dl_techniques/utils/alignment/alignment.py:457*

#### `pool_features(features, pool_type, axis)`
**Module:** `utils.alignment.utils`

Pool features along specified axis.

*📁 src/dl_techniques/utils/alignment/utils.py:329*

#### `power_iteration(matrix, iterations, epsilon)`
**Module:** `utils.tensors`

Compute spectral norm using power iteration.

*📁 src/dl_techniques/utils/tensors.py:128*

#### `predict(self, x_test)`
**Module:** `utils.conformal_forecaster`

Generate prediction intervals with validity guarantee.

*📁 src/dl_techniques/utils/conformal_forecaster.py:437*

#### `predict_batch_images(self, image_paths, output_dir, save_visualizations)`
**Module:** `utils.inference`

Run inference on a batch of images.

*📁 src/dl_techniques/utils/inference.py:295*

#### `predict_image(self, image, return_patches, progress_callback)`
**Module:** `utils.inference`

Run inference on a full image.

*📁 src/dl_techniques/utils/inference.py:109*

#### `prepare_arrays(self, df, window_config)`
**Module:** `datasets.time_series.base`

Convert a DataFrame to windowed numpy arrays.

*📁 src/dl_techniques/datasets/time_series/base.py:244*

#### `prepare_features(feats, q, exact)`
**Module:** `utils.alignment.utils`

Prepare features by removing outliers and converting to tensor.

*📁 src/dl_techniques/utils/alignment/utils.py:12*

#### `preprocess(image, label)`
**Module:** `datasets.vision.imagenet`

Preprocess image and label.

*📁 src/dl_techniques/datasets/vision/imagenet.py:52*

#### `preprocess_features(features, normalize)`
**Module:** `utils.graphs`

Preprocess node features for graph neural networks.

*📁 src/dl_techniques/utils/graphs.py:271*

#### `preprocess_image(self, image_path)`
**Module:** `datasets.vqa_dataset`

Preprocess image for nanoVLM input.

*📁 src/dl_techniques/datasets/vqa_dataset.py:217*

#### `preprocess_text(self, question, answer, conversation_format)`
**Module:** `datasets.vqa_dataset`

Preprocess text for training or inference.

*📁 src/dl_techniques/datasets/vqa_dataset.py:246*

#### `process_bboxes()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:679*

#### `process_negative_sampling()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:467*

#### `process_pipeline(ds, is_training)`
**Module:** `datasets.vision.coco`

*📁 src/dl_techniques/datasets/vision/coco.py:623*

#### `process_positive_sampling()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:387*

#### `profile_inference(self, inference_engine, test_images, output_file)`
**Module:** `utils.inference`

Profile inference performance on test images.

*📁 src/dl_techniques/utils/inference.py:488*

#### `project(self, x, c)`
**Module:** `utils.geometry.poincare_math`

Project points onto the Poincaré ball to ensure valid hyperbolic coordinates.

*📁 src/dl_techniques/utils/geometry/poincare_math.py:104*

#### `range_from_bits(bits)`
**Module:** `utils.scaling`

Compute the range of values based on the number of bits.

*📁 src/dl_techniques/utils/scaling.py:4*

#### `rayleigh(shape, scale, dtype, seed, name)`
**Module:** `utils.random`

Generates a tensor of positive reals drawn from a Rayleigh distribution.

*📁 src/dl_techniques/utils/random.py:6*

#### `read_and_melt(filepath)`
**Module:** `datasets.time_series.m4`

Read M4 CSV and convert to long format.

*📁 src/dl_techniques/datasets/time_series/m4.py:311*

#### `remove_outliers(feats, q, exact, max_threshold)`
**Module:** `utils.alignment.metrics`

Remove outliers by clipping to quantile.

*📁 src/dl_techniques/utils/alignment/metrics.py:748*

#### `remove_outliers(feats, q, exact, max_threshold)`
**Module:** `utils.alignment.metrics`

Simplified version without tensorflow_probability.

*📁 src/dl_techniques/utils/alignment/metrics.py:809*

#### `reset_state(self)`
**Module:** `datasets.arc.arc_keras`

Reset metric state.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:604*

#### `reshape_to_2d(weights, name)`
**Module:** `utils.tensors`

Reshape weight tensor to 2D matrix for regularization computations.

*📁 src/dl_techniques/utils/tensors.py:15*

#### `result(self)`
**Module:** `datasets.arc.arc_keras`

Compute the final metric result.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:595*

#### `return_empty()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:287*

#### `return_empty()`
**Module:** `datasets.sut`

Return empty tensors when no centers.

*📁 src/dl_techniques/datasets/sut.py:765*

#### `return_empty_negative()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:464*

#### `return_empty_positive()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:384*

#### `round_clamp(inputs, value_range, lambda_)`
**Module:** `utils.scaling`

Round the tensor and clamp it to the specified range using straight-through estimator.

*📁 src/dl_techniques/utils/scaling.py:31*

#### `safe_divide(x, y, eps)`
**Module:** `utils.tensors`

Safe division with epsilon to prevent div by zero.

*📁 src/dl_techniques/utils/tensors.py:186*

#### `safe_norm(self, x, axis, keepdims)`
**Module:** `utils.geometry.poincare_math`

Compute Euclidean norm with gradient-safe handling of zero vectors.

*📁 src/dl_techniques/utils/geometry/poincare_math.py:76*

#### `sample(inputs, value_range, lambda_)`
**Module:** `utils.scaling`

Sample a discrete tensor from the input tensor using straight-through estimator.

*📁 src/dl_techniques/utils/scaling.py:67*

#### `sample_negative()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:450*

#### `sample_negative_edges(num_nodes, positive_edges, num_samples, max_attempts)`
**Module:** `utils.graphs`

Sample negative edges (non-existent edges) for link prediction training.

*📁 src/dl_techniques/utils/graphs.py:322*

#### `sample_positive()`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:370*

#### `save_features(features, save_path, metadata)`
**Module:** `utils.alignment.utils`

Save features to disk.

*📁 src/dl_techniques/utils/alignment/utils.py:266*

#### `save_figure(self, fig, name, subdir, close_fig)`
**Module:** `utils.visualization_manager`

Save figure with proper configuration.

*📁 src/dl_techniques/utils/visualization_manager.py:115*

#### `scale(inputs, value_range, measure_fn, keepdim, eps)`
**Module:** `utils.scaling`

Scale the input tensor based on the measure and range.

*📁 src/dl_techniques/utils/scaling.py:226*

#### `score(self, features, return_layer_indices)`
**Module:** `utils.alignment.alignment`

Score features against reference features.

*📁 src/dl_techniques/utils/alignment/alignment.py:97*

#### `sequence_to_grid(self, sequence)`
**Module:** `datasets.arc.arc_converters`

Convert a flattened sequence back to a 2D grid.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:237*

#### `set_reference_features(self, reference_features)`
**Module:** `utils.alignment.alignment`

Set or update reference features.

*📁 src/dl_techniques/utils/alignment/alignment.py:286*

#### `sparse_to_tf_sparse(sparse_matrix)`
**Module:** `utils.graphs`

Convert scipy sparse matrix to TensorFlow SparseTensor.

*📁 src/dl_techniques/utils/graphs.py:150*

#### `split_annotations(annotations, split_ratio)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:1001*

#### `split_data(self, df, config, normalize, norm_config)`
**Module:** `datasets.time_series.base`

Split a DataFrame into train/validation/test sets.

*📁 src/dl_techniques/datasets/time_series/base.py:137*

#### `split_tasks(self, tasks, train_ratio, val_ratio, test_ratio, ensure_no_augmented_leakage)`
**Module:** `datasets.arc.arc_converters`

Split tasks into train/validation/test sets.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:585*

#### `stitch_patches(self, patch_predictions, image_width, image_height)`
**Module:** `datasets.patch_transforms`

Stitch segmentation patches into full image mask.

*📁 src/dl_techniques/datasets/patch_transforms.py:360*

#### `supports_perfect_inverse(self)`
**Module:** `datasets.time_series.normalizer`

Check if current method supports perfect reconstruction.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:116*

#### `svcca(feats_a, feats_b, cca_dim)`
**Module:** `utils.alignment.metrics`

Singular Vector Canonical Correlation Analysis (SVCCA).

*📁 src/dl_techniques/utils/alignment/metrics.py:260*

#### `task_to_hrm_format(self, task, augmentation_config)`
**Module:** `datasets.arc.arc_converters`

Convert ARCTaskData to HRM dataset format.

*📁 src/dl_techniques/datasets/arc/arc_converters.py:267*

#### `to_dict(self)`
**Module:** `datasets.patch_transforms`

*📁 src/dl_techniques/datasets/patch_transforms.py:65*

#### `to_normalized_tensor(self, image_width, image_height)`
**Module:** `datasets.sut`

Convert to normalized tensor coordinates.

*📁 src/dl_techniques/datasets/sut.py:82*

#### `to_tensor(self)`
**Module:** `datasets.sut`

Convert to TensorFlow tensor [xmin, ymin, xmax, ymax].

*📁 src/dl_techniques/datasets/sut.py:78*

#### `to_tensor_dict(self)`
**Module:** `datasets.sut`

Convert annotation to TensorFlow tensors.

*📁 src/dl_techniques/datasets/sut.py:114*

#### `to_tf_dataset(self, output_signature, batch_size, transform_fn, filter_fn, columns, skip_errors, drop_remainder, prefetch_buffer, enable_auto_sharding)`
**Module:** `datasets.universal_dataset_loader`

Convert the Hugging Face dataset to an optimized TensorFlow Dataset.

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:299*

#### `to_tf_dataset_tuple(self, output_signature, batch_size, transform_fn, filter_fn, columns, skip_errors, drop_remainder, prefetch_buffer, enable_auto_sharding)`
**Module:** `datasets.universal_dataset_loader`

Convert the dataset to a TensorFlow Dataset yielding tuples.

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:417*

#### `train_model(model, x_train, y_train, x_test, y_test, config)`
**Module:** `utils.train`

Train a Keras model with configurable parameters and callbacks.

*📁 src/dl_techniques/utils/train.py:44*

#### `training_tips()`
**Module:** `applications.geometric.examples`

Tips for training models with spatial layers.

*📁 src/dl_techniques/applications/geometric/examples.py:648*

#### `transform(self, X)`
**Module:** `datasets.tabular`

Transform data into numerical and categorical arrays.

*📁 src/dl_techniques/datasets/tabular.py:104*

#### `transform(self, data)`
**Module:** `datasets.time_series.normalizer`

Transform data using fitted parameters.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:216*

#### `transform_quantile_uniform(self, data)`
**Module:** `datasets.time_series.normalizer`

Helper for quantile transform.

*📁 src/dl_techniques/datasets/time_series/normalizer.py:285*

#### `tuple_generator()`
**Module:** `datasets.universal_dataset_loader`

Generate tuples from the underlying generator.

*📁 src/dl_techniques/datasets/universal_dataset_loader.py:481*

#### `unbiased_cka()`
**Module:** `utils.alignment.metrics`

Unbiased Centered Kernel Alignment.

*📁 src/dl_techniques/utils/alignment/metrics.py:243*

#### `update(self, x_new, y_new)`
**Module:** `utils.conformal_forecaster`

Online update with new observations (adaptive conformal).

*📁 src/dl_techniques/utils/conformal_forecaster.py:489*

#### `update_state(self, y_true, y_pred, sample_weight)`
**Module:** `datasets.arc.arc_keras`

Update metric state.

*📁 src/dl_techniques/datasets/arc/arc_keras.py:548*

#### `v_flip_fn()`
**Module:** `datasets.vision.coco`

*📁 src/dl_techniques/datasets/vision/coco.py:558*

#### `validate_dataframe_schema(df, required_columns, dataset_name)`
**Module:** `datasets.time_series.utils`

Check if DataFrame contains required columns.

*📁 src/dl_techniques/datasets/time_series/utils.py:185*

#### `validate_orthonormality(vectors, rtol, atol)`
**Module:** `utils.tensors`

Validates that a set of vectors is orthonormal using the Keras backend.

*📁 src/dl_techniques/utils/tensors.py:301*

#### `validate_rayleigh_samples(samples, scale, significance_level)`
**Module:** `utils.random`

Validates generated Rayleigh samples using statistical tests.

*📁 src/dl_techniques/utils/random.py:93*

#### `validate_split(self, split, subset)`
**Module:** `datasets.arc.arc_utilities`

Validate a specific dataset split.

*📁 src/dl_techniques/datasets/arc/arc_utilities.py:750*

#### `visualize_dataset(X, y, title, filename, show, centers)`
**Module:** `datasets.simple_2d`

Visualize a 2D classification dataset.

*📁 src/dl_techniques/datasets/simple_2d.py:513*

#### `visualize_mask(mask, title, figsize, cmap, save_path, show)`
**Module:** `utils.masking.factory`

Visualize a mask using matplotlib.

*📁 src/dl_techniques/utils/masking/factory.py:847*

#### `vocab_size(self)`
**Module:** `utils.tokenizer`

Get the vocabulary size of the underlying Tiktoken encoding.

*📁 src/dl_techniques/utils/tokenizer.py:434*

#### `width(self)`
**Module:** `datasets.sut`

*📁 src/dl_techniques/datasets/sut.py:59*

#### `width(self)`
**Module:** `datasets.patch_transforms`

*📁 src/dl_techniques/datasets/patch_transforms.py:41*

#### `window_partition(x, window_size)`
**Module:** `utils.tensors`

Partition feature map into non-overlapping windows.

*📁 src/dl_techniques/utils/tensors.py:393*

#### `window_reverse(windows, window_size, H, W)`
**Module:** `utils.tensors`

Reverse window partitioning back to feature map.

*📁 src/dl_techniques/utils/tensors.py:411*

#### `wt_x_w_normalize(weights)`
**Module:** `utils.tensors`

Compute normalized Gram matrix (W^T * W) with improved numerical stability.

*📁 src/dl_techniques/utils/tensors.py:96*

### Visualization Functions

#### `add_subplot(self, name, plot_func)`
**Module:** `visualization.core`

Add a subplot to this composite visualization.

*📁 src/dl_techniques/visualization/core.py:318*

#### `adjust_lightness(color, amount)`
**Module:** `visualization.core`

Adjust the lightness of a color.

*📁 src/dl_techniques/visualization/core.py:646*

#### `can_handle(self, data)`
**Module:** `visualization.time_series`

*📁 src/dl_techniques/visualization/time_series.py:67*

#### `can_handle(self, data)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:71*

#### `can_handle(self, data)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:137*

#### `can_handle(self, data)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:198*

#### `can_handle(self, data)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:244*

#### `can_handle(self, data)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:291*

#### `can_handle(self, data)`
**Module:** `visualization.core`

Check if this plugin can handle the given data.

*📁 src/dl_techniques/visualization/core.py:248*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:70*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

Check if this plugin can handle the given data.

*📁 src/dl_techniques/visualization/training_performance.py:195*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:270*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:350*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:423*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:558*

#### `can_handle(self, data)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:697*

#### `can_handle(self, data)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:72*

#### `can_handle(self, data)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:220*

#### `can_handle(self, data)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:372*

#### `can_handle(self, data)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:472*

#### `can_handle(self, data)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:648*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:369*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:483*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:644*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:796*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:919*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1134*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1211*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1345*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1414*

#### `can_handle(self, data)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1487*

#### `create_color_palette(n_colors, palette_type)`
**Module:** `visualization.core`

Create a color palette with n colors.

*📁 src/dl_techniques/visualization/core.py:641*

#### `create_dashboard(self, data, layout, save, show)`
**Module:** `visualization.core`

Create a dashboard with multiple visualizations.

*📁 src/dl_techniques/visualization/core.py:505*

#### `create_gradient_heatmap(self, gradients, title, cmap, figsize, show_colorbar, log_scale, save_path)`
**Module:** `visualization.data_nn`

Create a heatmap visualization of gradient topology.

*📁 src/dl_techniques/visualization/data_nn.py:268*

#### `create_plugin_from_template(self, template_name)`
**Module:** `visualization.core`

Create a plugin instance from a template.

*📁 src/dl_techniques/visualization/core.py:429*

#### `create_visualization(self, data, ax, num_samples, plot_type)`
**Module:** `visualization.time_series`

Create the forecast visualization.

*📁 src/dl_techniques/visualization/time_series.py:70*

#### `create_visualization(self, data, ax, show_identity)`
**Module:** `visualization.regression`

Create prediction error visualization.

*📁 src/dl_techniques/visualization/regression.py:74*

#### `create_visualization(self, data, ax, lowess)`
**Module:** `visualization.regression`

Create residuals plot.

*📁 src/dl_techniques/visualization/regression.py:140*

#### `create_visualization(self, data, ax, bins)`
**Module:** `visualization.regression`

Create residual distribution plot.

*📁 src/dl_techniques/visualization/regression.py:201*

#### `create_visualization(self, data, ax)`
**Module:** `visualization.regression`

Create Q-Q plot.

*📁 src/dl_techniques/visualization/regression.py:247*

#### `create_visualization(self, data)`
**Module:** `visualization.regression`

Overridden to add a metrics table to the dashboard.

*📁 src/dl_techniques/visualization/regression.py:321*

#### `create_visualization(self, data, ax)`
**Module:** `visualization.core`

Create the visualization.

*📁 src/dl_techniques/visualization/core.py:261*

#### `create_visualization(self, data, ax, layout, default_cols)`
**Module:** `visualization.core`

Create a composite visualization with multiple subplots.

*📁 src/dl_techniques/visualization/core.py:322*

#### `create_visualization(self, data, metrics_to_plot, smooth_factor, show_best_epoch)`
**Module:** `visualization.training_performance`

Create training curves visualization.

*📁 src/dl_techniques/visualization/training_performance.py:73*

#### `create_visualization(self, data, show_phases, phase_boundaries)`
**Module:** `visualization.training_performance`

Create learning rate schedule visualization.

*📁 src/dl_techniques/visualization/training_performance.py:218*

#### `create_visualization(self, data, metrics_to_show, sort_by, show_values)`
**Module:** `visualization.training_performance`

Create model comparison bar chart.

*📁 src/dl_techniques/visualization/training_performance.py:273*

#### `create_visualization(self, data, metrics_to_show, normalize)`
**Module:** `visualization.training_performance`

Create radar chart comparison.

*📁 src/dl_techniques/visualization/training_performance.py:353*

#### `create_visualization(self, data, patience)`
**Module:** `visualization.training_performance`

Create overfitting analysis visualization.

*📁 src/dl_techniques/visualization/training_performance.py:561*

#### `create_visualization(self, data, metric_to_display)`
**Module:** `visualization.training_performance`

Create comprehensive performance dashboard.

*📁 src/dl_techniques/visualization/training_performance.py:703*

#### `create_visualization(self, data, ax, normalize, show_percentages, show_counts, cmap)`
**Module:** `visualization.classification`

Create confusion matrix visualization.

*📁 src/dl_techniques/visualization/classification.py:75*

#### `create_visualization(self, data, ax, plot_type, show_thresholds)`
**Module:** `visualization.classification`

Create ROC and/or PR curves.

*📁 src/dl_techniques/visualization/classification.py:223*

#### `create_visualization(self, data, ax, metrics)`
**Module:** `visualization.classification`

Create classification report visualization.

*📁 src/dl_techniques/visualization/classification.py:375*

#### `create_visualization(self, data, ax, show_examples, x_data)`
**Module:** `visualization.classification`

Create error analysis dashboard.

*📁 src/dl_techniques/visualization/classification.py:651*

#### `create_visualization(self, data, features_to_plot, plot_type)`
**Module:** `visualization.data_nn`

Create data distribution visualization.

*📁 src/dl_techniques/visualization/data_nn.py:372*

#### `create_visualization(self, data, show_percentages)`
**Module:** `visualization.data_nn`

Create class balance visualization.

*📁 src/dl_techniques/visualization/data_nn.py:486*

#### `create_visualization(self, data, show_params, show_shapes, orientation)`
**Module:** `visualization.data_nn`

Create network architecture visualization.

*📁 src/dl_techniques/visualization/data_nn.py:647*

#### `create_visualization(self, data, layers_to_show, plot_type)`
**Module:** `visualization.data_nn`

Create activation visualization.

*📁 src/dl_techniques/visualization/data_nn.py:799*

#### `create_visualization(self, data, layers_to_show, plot_type)`
**Module:** `visualization.data_nn`

Create weight visualization.

*📁 src/dl_techniques/visualization/data_nn.py:922*

#### `create_visualization(self, data, sample_idx, layers_to_show, max_features)`
**Module:** `visualization.data_nn`

Create feature map visualization.

*📁 src/dl_techniques/visualization/data_nn.py:1137*

#### `create_visualization(self, data, plot_type)`
**Module:** `visualization.data_nn`

Create gradient visualization.

*📁 src/dl_techniques/visualization/data_nn.py:1214*

#### `create_visualization(self, data, ax)`
**Module:** `visualization.data_nn`

Create a gradient topology heatmap.

*📁 src/dl_techniques/visualization/data_nn.py:1348*

#### `create_visualization(self, data, ax)`
**Module:** `visualization.data_nn`

Create a heatmap visualization for a 2D matrix.

*📁 src/dl_techniques/visualization/data_nn.py:1418*

#### `create_visualization(self, data, ax)`
**Module:** `visualization.data_nn`

Create a side-by-side comparison of images.

*📁 src/dl_techniques/visualization/data_nn.py:1494*

#### `description(self)`
**Module:** `visualization.time_series`

*📁 src/dl_techniques/visualization/time_series.py:64*

#### `description(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:68*

#### `description(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:134*

#### `description(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:195*

#### `description(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:241*

#### `description(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:288*

#### `description(self)`
**Module:** `visualization.core`

Return a description of this visualization plugin.

*📁 src/dl_techniques/visualization/core.py:243*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:67*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:192*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:267*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:347*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:420*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:555*

#### `description(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:694*

#### `description(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:69*

#### `description(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:217*

#### `description(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:369*

#### `description(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:469*

#### `description(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:645*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:366*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:480*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:641*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:793*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:916*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1131*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1208*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1342*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1411*

#### `description(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1484*

#### `get_available_plugins(self)`
**Module:** `visualization.core`

Get list of available plugin names.

*📁 src/dl_techniques/visualization/core.py:611*

#### `get_available_templates(self)`
**Module:** `visualization.core`

Get list of available template names.

*📁 src/dl_techniques/visualization/core.py:615*

#### `get_model_color(self, model_name, index)`
**Module:** `visualization.core`

Get color for a specific model.

*📁 src/dl_techniques/visualization/core.py:71*

#### `get_save_path(self, filename, subdir)`
**Module:** `visualization.core`

Get full save path for a file.

*📁 src/dl_techniques/visualization/core.py:206*

#### `get_statistics(self, gradients)`
**Module:** `visualization.data_nn`

Compute statistics about the gradient topology.

*📁 src/dl_techniques/visualization/data_nn.py:322*

#### `get_style_params(self)`
**Module:** `visualization.core`

Get matplotlib rcParams for the configured style.

*📁 src/dl_techniques/visualization/core.py:120*

#### `name(self)`
**Module:** `visualization.time_series`

*📁 src/dl_techniques/visualization/time_series.py:60*

#### `name(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:64*

#### `name(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:130*

#### `name(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:191*

#### `name(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:237*

#### `name(self)`
**Module:** `visualization.regression`

*📁 src/dl_techniques/visualization/regression.py:284*

#### `name(self)`
**Module:** `visualization.core`

Return the name of this visualization plugin.

*📁 src/dl_techniques/visualization/core.py:237*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:63*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:188*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:263*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:343*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:416*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:551*

#### `name(self)`
**Module:** `visualization.training_performance`

*📁 src/dl_techniques/visualization/training_performance.py:690*

#### `name(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:65*

#### `name(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:213*

#### `name(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:365*

#### `name(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:465*

#### `name(self)`
**Module:** `visualization.classification`

*📁 src/dl_techniques/visualization/classification.py:641*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:362*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:476*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:637*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:789*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:912*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1127*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1204*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1338*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1407*

#### `name(self)`
**Module:** `visualization.data_nn`

*📁 src/dl_techniques/visualization/data_nn.py:1480*

#### `register_plugin(self, plugin)`
**Module:** `visualization.core`

Register a visualization plugin.

*📁 src/dl_techniques/visualization/core.py:419*

#### `register_template(self, name, template_class)`
**Module:** `visualization.core`

Register a visualization template class.

*📁 src/dl_techniques/visualization/core.py:424*

#### `save_figure(self, fig, name, subdir)`
**Module:** `visualization.core`

Save a figure with proper configuration.

*📁 src/dl_techniques/visualization/core.py:280*

#### `save_metadata(self, metadata)`
**Module:** `visualization.core`

Save metadata about the visualizations.

*📁 src/dl_techniques/visualization/core.py:619*

#### `setup_logging(level)`
**Module:** `visualization.core`

Setup logging configuration.

*📁 src/dl_techniques/visualization/core.py:632*

#### `style_context(self)`
**Module:** `visualization.core`

Context manager for temporarily applying style settings.

*📁 src/dl_techniques/visualization/core.py:182*

#### `style_context(self)`
**Module:** `visualization.core`

Context manager for applying this manager's style settings.

*📁 src/dl_techniques/visualization/core.py:438*

#### `visualize(self, data, plugin_name, save, show, filename)`
**Module:** `visualization.core`

Create a visualization for the given data.

*📁 src/dl_techniques/visualization/core.py:443*