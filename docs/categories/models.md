# Models

Complete model architectures ready for training

**183 modules in this category**

## Accunet

### models.accunet

*📁 File: `src/dl_techniques/models/accunet/__init__.py`*

### models.accunet.model
ACC-UNet: A Completely Convolutional UNet model for the 2020s.

**Classes:**
- `AccUNet`

**Functions:** `create_acc_unet`, `create_acc_unet_binary`, `create_acc_unet_multiclass`, `call`, `get_config` (and 1 more)

*📁 File: `src/dl_techniques/models/accunet/model.py`*

## Adaptive_Ema

### models.adaptive_ema

*📁 File: `src/dl_techniques/models/adaptive_ema/__init__.py`*

### models.adaptive_ema.model
Adaptive EMA Slope Filter Layer.

**Classes:**
- `AdaptiveEMASlopeFilterModel`

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/adaptive_ema/model.py`*

## Bert

### models.bert

*📁 File: `src/dl_techniques/models/bert/__init__.py`*

### models.bert.bert
BERT Model Implementation with Pretrained Support

**Classes:**
- `BERT`

**Functions:** `create_bert_with_head`, `call`, `load_pretrained_weights`, `from_variant`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/bert/bert.py`*

## Bias_Free_Denoisers

### models.bias_free_denoisers

*📁 File: `src/dl_techniques/models/bias_free_denoisers/__init__.py`*

### models.bias_free_denoisers.bfcnn
Bias-Free CNN Denoiser Model with Variants

**Functions:** `create_bfcnn_denoiser`, `create_bfcnn_variant`

*📁 File: `src/dl_techniques/models/bias_free_denoisers/bfcnn.py`*

### models.bias_free_denoisers.bfconvunext
ConvUNext: Modern Bias-Free U-Net with ConvNeXt-Inspired Architecture

**Classes:**

- `ConvUNextStem` - Keras Layer
  ConvUNext stem block for initial feature extraction using bias-free design.
  ```python
  ConvUNextStem(filters: int, kernel_size: Union[int, Tuple[int, int]] = 7, kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal', ...)
  ```

**Functions:** `create_convunext_denoiser`, `create_convunext_variant`, `get_model_output_info`, `create_inference_model_from_training_model`, `build` (and 3 more)

*📁 File: `src/dl_techniques/models/bias_free_denoisers/bfconvunext.py`*

### models.bias_free_denoisers.bfunet
Bias-Free U-Net Model with Deep Supervision and Variants

**Functions:** `create_bfunet_denoiser`, `load_pretrained_weights_into_model`, `create_bfunet_variant`, `get_model_output_info`, `create_inference_model_from_training_model`

*📁 File: `src/dl_techniques/models/bias_free_denoisers/bfunet.py`*

### models.bias_free_denoisers.bfunet_conditional
Conditional Bias-Free U-Net Model with Deep Supervision

**Functions:** `create_conditional_bfunet_denoiser`, `create_conditional_bfunet_variant`, `inject_class_conditioning`

*📁 File: `src/dl_techniques/models/bias_free_denoisers/bfunet_conditional.py`*

### models.bias_free_denoisers.bfunet_conditional_unified
Unified Conditional Bias-Free U-Net Model

**Classes:**

- `DenseConditioningInjection` - Keras Layer
  Inject dense conditioning features into target features.
  ```python
  DenseConditioningInjection(method: str = 'film', kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal', kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None, ...)
  ```

- `DiscreteConditioningInjection` - Keras Layer
  Inject discrete conditioning (embeddings) into target features.
  ```python
  DiscreteConditioningInjection(method: str = 'spatial_broadcast', projected_channels: Optional[int] = None, kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal', ...)
  ```

**Functions:** `create_dense_conditioning_encoder`, `create_unified_conditional_bfunet`, `create_depth_estimation_bfunet`, `create_class_conditional_bfunet`, `create_semantic_depth_bfunet` (and 7 more)

*📁 File: `src/dl_techniques/models/bias_free_denoisers/bfunet_conditional_unified.py`*

## Byte_Latent_Transformer

### models.byte_latent_transformer

*📁 File: `src/dl_techniques/models/byte_latent_transformer/__init__.py`*

### models.byte_latent_transformer.model
Byte Latent Transformer (BLT): Patches Scale Better Than Tokens

**Classes:**
- `ByteLatentTransformer`

**Functions:** `create_blt_model`, `build`, `call`, `train_step`, `generate` (and 4 more)

*📁 File: `src/dl_techniques/models/byte_latent_transformer/model.py`*

## Capsnet

### models.capsnet

*📁 File: `src/dl_techniques/models/capsnet/__init__.py`*

### models.capsnet.model
Implementation of the Capsule Network

**Classes:**
- `CapsNet`

**Functions:** `create_capsnet`, `build`, `call`, `train_step`, `test_step` (and 5 more)

*📁 File: `src/dl_techniques/models/capsnet/model.py`*

## Cbam

### models.cbam

*📁 File: `src/dl_techniques/models/cbam/__init__.py`*

### models.cbam.model
CBAMNet Model Implementation with Pretrained Support

**Classes:**
- `CBAMNet`

**Functions:** `create_cbam_net`, `call`, `get_config`, `from_config`, `from_variant`

*📁 File: `src/dl_techniques/models/cbam/model.py`*

## Ccnets

### models.ccnets

*📁 File: `src/dl_techniques/models/ccnets/__init__.py`*

### models.ccnets.base

**Classes:**
- `CCNetModule`
- `CCNetConfig`
- `CCNetLosses`
- `CCNetModelErrors`

**Functions:** `trainable_variables`, `save`, `to_dict`, `to_dict`

*📁 File: `src/dl_techniques/models/ccnets/base.py`*

### models.ccnets.control

**Classes:**
- `ConvergenceControlStrategy`
- `StaticThresholdStrategy`
- `AdaptiveDivergenceStrategy`

**Functions:** `update_state`, `should_train_reasoner`, `update_state`, `should_train_reasoner`, `update_state` (and 3 more)

*📁 File: `src/dl_techniques/models/ccnets/control.py`*

### models.ccnets.losses

**Classes:**
- `LossFunction`
- `L1Loss`
- `L2Loss`
- `HuberLoss`
- `PolynomialLoss`

*📁 File: `src/dl_techniques/models/ccnets/losses.py`*

### models.ccnets.orchestrators

**Classes:**
- `CCNetOrchestrator`
- `SequentialCCNetOrchestrator`

**Functions:** `forward_pass`, `compute_losses`, `compute_model_errors`, `train_step`, `evaluate` (and 10 more)

*📁 File: `src/dl_techniques/models/ccnets/orchestrators.py`*

### models.ccnets.trainer

**Classes:**
- `CCNetTrainer`

**Functions:** `train`

*📁 File: `src/dl_techniques/models/ccnets/trainer.py`*

### models.ccnets.utils

**Classes:**
- `EarlyStoppingCallback`
- `KerasModelWrapper`

**Functions:** `wrap_keras_model`, `trainable_variables`, `save`

*📁 File: `src/dl_techniques/models/ccnets/utils.py`*

## Clip

### models.clip

*📁 File: `src/dl_techniques/models/clip/__init__.py`*

### models.clip.model
CLIP (Contrastive Language-Image Pre-training) Model Implementation.

**Classes:**
- `CLIP`

**Functions:** `create_clip_model`, `create_clip_variant`, `build`, `encode_image`, `encode_text` (and 4 more)

*📁 File: `src/dl_techniques/models/clip/model.py`*

## Convnext

### models.convnext

*📁 File: `src/dl_techniques/models/convnext/__init__.py`*

### models.convnext.convnext_v1
ConvNeXt V1 Model Implementation with Pretrained Support

**Classes:**
- `ConvNeXtV1`

**Functions:** `create_convnext_v1`, `build`, `call`, `load_pretrained_weights`, `from_variant` (and 3 more)

*📁 File: `src/dl_techniques/models/convnext/convnext_v1.py`*

### models.convnext.convnext_v2
ConvNeXt V2 Model Implementation with Pretrained Support

**Classes:**
- `ConvNeXtV2`

**Functions:** `create_convnext_v2`, `build`, `call`, `load_pretrained_weights`, `from_variant` (and 4 more)

*📁 File: `src/dl_techniques/models/convnext/convnext_v2.py`*

## Convunext

### models.convunext

*📁 File: `src/dl_techniques/models/convunext/__init__.py`*

### models.convunext.model
ConvUNext Model: Modern U-Net with ConvNeXt-Inspired Architecture

**Classes:**

- `ConvUNextStem` - Keras Layer
  ConvUNext stem block for initial feature extraction.
  ```python
  ConvUNextStem(filters: int, kernel_size: Union[int, Tuple[int, int]] = 7, use_bias: bool = True, ...)
  ```
- `ConvUNextModel`

**Functions:** `create_convunext_variant`, `create_inference_model_from_training_model`, `build`, `call`, `compute_output_shape` (and 11 more)

*📁 File: `src/dl_techniques/models/convunext/model.py`*

## Core

### models

*📁 File: `src/dl_techniques/models/__init__.py`*

## Coshnet

### models.coshnet

*📁 File: `src/dl_techniques/models/coshnet/__init__.py`*

### models.coshnet.model
CoShNet (Complex Shearlet Network) Implementation

**Classes:**
- `CoShNet`

**Functions:** `create_coshnet`, `from_variant`, `get_config`, `from_config`, `summary`

*📁 File: `src/dl_techniques/models/coshnet/model.py`*

## Darkir

### models.darkir

*📁 File: `src/dl_techniques/models/darkir/__init__.py`*

### models.darkir.model
DarkIR: Robust Low-Light Image Restoration Network

**Classes:**

- `SimpleGate` - Keras Layer
  SimpleGate: Element-wise multiplicative gating without learnable parameters.
  ```python
  SimpleGate(**kwargs)
  ```

- `FreMLP` - Keras Layer
  Frequency MLP: Processes features in the frequency domain for global modeling.
  ```python
  FreMLP(channels: int, expansion: int = 2, **kwargs)
  ```

- `DilatedBranch` - Keras Layer
  A single branch of dilated depthwise convolution for multi-scale context.
  ```python
  DilatedBranch(channels: int, expansion: int = 1, dilation: int = 1, ...)
  ```

- `DarkIREncoderBlock` - Keras Layer
  Encoder Block (EBlock) for DarkIR with parallel dilated branches and FreMLP.
  ```python
  DarkIREncoderBlock(channels: int, dw_expand: int = 2, dilations: List[int] = None, ...)
  ```

- `DarkIRDecoderBlock` - Keras Layer
  Decoder Block (DBlock) for DarkIR with dual SimpleGate and FFN structure.
  ```python
  DarkIRDecoderBlock(channels: int, dw_expand: int = 2, ffn_expand: int = 2, ...)
  ```

**Functions:** `create_darkir_model`, `call`, `compute_output_shape`, `get_config`, `build` (and 15 more)

*📁 File: `src/dl_techniques/models/darkir/model.py`*

## Deepar

### models.deepar

*📁 File: `src/dl_techniques/models/deepar/__init__.py`*

### models.deepar.model
DeepAR Probabilistic Forecasting Model.

**Classes:**
- `DeepAR`

**Functions:** `compute_scale`, `call`, `predict_step`, `gaussian_loss`, `negative_binomial_loss` (and 1 more)

*📁 File: `src/dl_techniques/models/deepar/model.py`*

## Depth_Anything

### models.depth_anything

*📁 File: `src/dl_techniques/models/depth_anything/__init__.py`*

### models.depth_anything.components
This module provides a `DPTDecoder` layer, which implements the decoder component of

**Classes:**

- `DPTDecoder` - Keras Layer
  DPT (Dense Prediction Transformer) decoder.
  ```python
  DPTDecoder(dims: Optional[List[int]] = None, output_channels: int = 1, kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal', ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `get_build_config` (and 1 more)

*📁 File: `src/dl_techniques/models/depth_anything/components.py`*

### models.depth_anything.model
Depth Anything Implementation in Keras.

**Classes:**
- `DepthAnything`

**Functions:** `create_depth_anything`, `build`, `call`, `compile`, `train_step` (and 2 more)

*📁 File: `src/dl_techniques/models/depth_anything/model.py`*

## Detr

### models.detr

*📁 File: `src/dl_techniques/models/detr/__init__.py`*

### models.detr.model
DETR (DEtection TRansformer) Model Implementation in Keras 3

**Classes:**

- `DetrTransformer` - Keras Layer
  DETR Transformer combining encoder and decoder stacks.
  ```python
  DetrTransformer(hidden_dim: int = 256, num_heads: int = 8, num_encoder_layers: int = 6, ...)
  ```

- `DetrDecoderLayer` - Keras Layer
  A single DETR Transformer Decoder Layer with pre-normalization.
  ```python
  DetrDecoderLayer(hidden_dim: int, num_heads: int, ffn_dim: int, ...)
  ```
- `DETR`

**Functions:** `create_detr`, `build`, `call`, `get_config`, `build` (and 5 more)

*📁 File: `src/dl_techniques/models/detr/model.py`*

## Dino

### models.dino

*📁 File: `src/dl_techniques/models/dino/__init__.py`*

### models.dino.dino_v1
DINO (DIstillation with NO labels) Vision Transformer Implementation

**Classes:**

- `DINOHead` - Keras Layer
  DINO projection head for self-supervised learning.
  ```python
  DINOHead(in_dim: int, out_dim: int, use_bn: bool = False, ...)
  ```
- `DINOv1`

**Functions:** `create_dino_v1`, `create_dino_teacher_student_pair`, `build`, `call`, `get_config` (and 5 more)

*📁 File: `src/dl_techniques/models/dino/dino_v1.py`*

### models.dino.dino_v2
DINOv2 Vision Transformer Implementation - Modern Keras 3 Patterns

**Classes:**

- `DINOv2Block` - Keras Layer
  DINOv2 Transformer Block with LearnableMultiplier scaling and configurable components.
  ```python
  DINOv2Block(dim: int, num_heads: int, mlp_ratio: float = 4.0, ...)
  ```
- `DINOv2VisionTransformer`
- `DINOv2`

**Functions:** `create_dino_v2`, `build`, `call`, `compute_output_shape`, `get_config` (and 14 more)

*📁 File: `src/dl_techniques/models/dino/dino_v2.py`*

### models.dino.dino_v3
DINOv3 Model Implementation

**Classes:**
- `DINOv3`

**Functions:** `create_dino_v3`, `get_last_selfattention`, `from_variant`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/models/dino/dino_v3.py`*

## Distilbert

### models.distilbert

*📁 File: `src/dl_techniques/models/distilbert/__init__.py`*

### models.distilbert.model
DistilBERT Model Implementation with Pretrained Support

**Classes:**

- `DistilBertEmbeddings` - Keras Layer
  Embeddings layer for DistilBERT.
  ```python
  DistilBertEmbeddings(vocab_size: int, hidden_size: int, max_position_embeddings: int = 512, ...)
  ```
- `DistilBERT`

**Functions:** `create_distilbert_with_head`, `call`, `get_config`, `call`, `load_pretrained_weights` (and 4 more)

*📁 File: `src/dl_techniques/models/distilbert/model.py`*

## Fastvlm

### models.fastvlm

*📁 File: `src/dl_techniques/models/fastvlm/__init__.py`*

### models.fastvlm.components

**Classes:**

- `AttentionBlock` - Keras Layer
  Attention block with vision_heads-specific adaptations.
  ```python
  AttentionBlock(dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/models/fastvlm/components.py`*

### models.fastvlm.model
FastVLM Model Implementation

**Classes:**
- `FastVLM`

**Functions:** `call`, `extract_features`, `from_variant`, `get_config`, `from_config` (and 1 more)

*📁 File: `src/dl_techniques/models/fastvlm/model.py`*

## Fftnet

### models.fftnet

*📁 File: `src/dl_techniques/models/fftnet/__init__.py`*

### models.fftnet.components
FFTNet (Spectre)

**Classes:**

- `MeanPoolingLayer` - Keras Layer
  Simple mean pooling over the sequence dimension.
  ```python
  MeanPoolingLayer(**kwargs)
  ```

- `AttentionPoolingLayer` - Keras Layer
  Two-layer attention pooling for creating global descriptors with learned weights.
  ```python
  AttentionPoolingLayer(hidden_dim: int = 256, **kwargs)
  ```

- `DCTPoolingLayer` - Keras Layer
  DCT-based pooling for gate descriptor creation.
  ```python
  DCTPoolingLayer(dct_components: int = 64, **kwargs)
  ```

- `ComplexModReLULayer` - Keras Layer
  Complex modReLU activation with learnable bias.
  ```python
  ComplexModReLULayer(num_features: int, **kwargs)
  ```

- `ComplexInterpolationLayer` - Keras Layer
  Complex tensor interpolation along the last dimension using bicubic interpolation.
  ```python
  ComplexInterpolationLayer(size: int, mode: str = 'cubic', **kwargs)
  ```

- `ComplexConv1DLayer` - Keras Layer
  One-dimensional complex convolution with circular padding.
  ```python
  ComplexConv1DLayer(kernel_size: int, **kwargs)
  ```

- `SpectreHead` - Keras Layer
  Frequency-domain token mixer for a single attention head.
  ```python
  SpectreHead(embed_dim: int, fft_size: int, num_groups: int = 4, ...)
  ```

- `SpectreMultiHead` - Keras Layer
  Multi-head Spectre layer combining multiple SpectreHead instances.
  ```python
  SpectreMultiHead(embed_dim: int, num_heads: int, fft_size: int, ...)
  ```

- `SpectreBlock` - Keras Layer
  Complete Transformer-style block using SpectreMultiHead for token mixing.
  ```python
  SpectreBlock(embed_dim: int, num_heads: int, fft_size: int, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `build`, `call` (and 28 more)

*📁 File: `src/dl_techniques/models/fftnet/components.py`*

### models.fftnet.model
FFTNet Foundation Model Implementation

**Classes:**

- `FFTMixer` - Keras Layer
  Adaptive spectral filtering layer implementing the core FFTNet mechanism.
  ```python
  FFTMixer(embed_dim: int, mlp_hidden_dim: int = 256, dropout_p: float = 0.0, ...)
  ```

- `FFTNetBlock` - Keras Layer
  Complete Transformer-style block using FFTMixer for token mixing.
  ```python
  FFTNetBlock(embed_dim: int, mlp_hidden_dim: int = 256, ffn_ratio: int = 4, ...)
  ```
- `FFTNet`

**Functions:** `create_fftnet_with_head`, `create_fftnet`, `create_fftnet_classifier`, `build`, `call` (and 12 more)

*📁 File: `src/dl_techniques/models/fftnet/model.py`*

## Fnet

### models.fnet

*📁 File: `src/dl_techniques/models/fnet/__init__.py`*

### models.fnet.model
FNet Model Implementation with Pretrained Support

**Classes:**
- `FNet`

**Functions:** `create_fnet_with_head`, `call`, `load_pretrained_weights`, `from_variant`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/fnet/model.py`*

## Fractalnet

### models.fractalnet

*📁 File: `src/dl_techniques/models/fractalnet/__init__.py`*

### models.fractalnet.model
FractalNet Model Implementation

**Classes:**
- `FractalNet`

**Functions:** `create_fractal_net`, `from_variant`, `get_config`, `from_config`, `summary`

*📁 File: `src/dl_techniques/models/fractalnet/model.py`*

## Gemma

### models.gemma

*📁 File: `src/dl_techniques/models/gemma/__init__.py`*

### models.gemma.components
This module implements the Gemma 3 Transformer Block, a fundamental component of

**Classes:**

- `Gemma3TransformerBlock` - Keras Layer
  Gemma 3 Transformer Block with a dual normalization pattern.
  ```python
  Gemma3TransformerBlock(hidden_size: int, num_attention_heads: int, num_key_value_heads: int, ...)
  ```

**Functions:** `build`, `compute_output_spec`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/gemma/components.py`*

### models.gemma.gemma3
A complete implementation of the Gemma 3 architecture following Modern Keras 3

**Classes:**
- `Gemma3`

**Functions:** `create_gemma3_generation`, `create_gemma3_classification`, `create_gemma3`, `call`, `from_variant` (and 1 more)

*📁 File: `src/dl_techniques/models/gemma/gemma3.py`*

## Hierarchical_Reasoning_Model

### models.hierarchical_reasoning_model

*📁 File: `src/dl_techniques/models/hierarchical_reasoning_model/__init__.py`*

### models.hierarchical_reasoning_model.model
Hierarchical Reasoning Model: Adaptive Computation Time with Multi-Level Reasoning

**Classes:**
- `HierarchicalReasoningModel`

**Functions:** `create_hierarchical_reasoning_model`, `initial_carry`, `call`, `from_variant`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/hierarchical_reasoning_model/model.py`*

## Jepa

### models.jepa

*📁 File: `src/dl_techniques/models/jepa/__init__.py`*

### models.jepa.config
JEPA Configuration Module for dl-techniques Framework.

**Classes:**
- `JEPAConfig`

**Functions:** `from_preset`, `to_dict`, `from_dict`, `get_encoder_config`, `get_predictor_config`

*📁 File: `src/dl_techniques/models/jepa/config.py`*

### models.jepa.encoder
JEPA Encoder Implementation using Vision Transformer Architecture.

**Classes:**

- `JEPAPatchEmbedding` - Keras Layer
  Advanced patch embedding layer for JEPA with support for different modalities.
  ```python
  JEPAPatchEmbedding(patch_size: Union[int, Tuple[int, ...]], embed_dim: int, img_size: Tuple[int, ...], ...)
  ```

- `JEPAEncoder` - Keras Layer
  JEPA Encoder using Vision Transformer architecture with modern optimizations.
  ```python
  JEPAEncoder(embed_dim: int, depth: int, num_heads: int, ...)
  ```

- `JEPAPredictor` - Keras Layer
  JEPA Predictor network for masked token prediction.
  ```python
  JEPAPredictor(embed_dim: int, depth: int, num_heads: int, ...)
  ```

**Functions:** `call`, `compute_output_shape`, `get_config`, `call`, `compute_output_shape` (and 4 more)

*📁 File: `src/dl_techniques/models/jepa/encoder.py`*

### models.jepa.utilities
JEPA Masking Utilities for Semantic Block-Based Masking.

**Classes:**
- `JEPAMaskingStrategy`
- `VideoMaskingStrategy`
- `AudioMaskingStrategy`

**Functions:** `set_difficulty`, `generate_masks`, `visualize_masks`, `get_mask_statistics`, `generate_video_masks` (and 2 more)

*📁 File: `src/dl_techniques/models/jepa/utilities.py`*

## Kan

### models.kan

*📁 File: `src/dl_techniques/models/kan/__init__.py`*

### models.kan.model
Kolmogorov-Arnold Network (KAN)

**Classes:**
- `KAN`

**Functions:** `create_kan_model`, `update_kan_grids`, `load_pretrained_weights`, `from_variant`, `from_layer_sizes` (and 4 more)

*📁 File: `src/dl_techniques/models/kan/model.py`*

## Latent_Gmm_Registration

### models.latent_gmm_registration

*📁 File: `src/dl_techniques/models/latent_gmm_registration/__init__.py`*

### models.latent_gmm_registration.model

**Classes:**
- `LatentGMMRegistration`

**Functions:** `compute_gmm_params`, `compute_rigid_transform`, `call`, `train_step`, `test_step` (and 2 more)

*📁 File: `src/dl_techniques/models/latent_gmm_registration/model.py`*

## Mamba

### models.mamba

*📁 File: `src/dl_techniques/models/mamba/__init__.py`*

### models.mamba.components

**Classes:**

- `MambaLayer` - Keras Layer
  Core Mamba selective state space model layer.
  ```python
  MambaLayer(d_model: int, d_state: int = 16, d_conv: int = 4, ...)
  ```

- `MambaResidualBlock` - Keras Layer
  Residual block wrapping a MambaLayer with pre-normalization.
  ```python
  MambaResidualBlock(d_model: int, norm_epsilon: float = 1e-05, mamba_kwargs: Optional[Dict[str, Any]] = None, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `build`, `call` (and 3 more)

*📁 File: `src/dl_techniques/models/mamba/components.py`*

### models.mamba.components_v2

**Classes:**

- `Mamba2Layer` - Keras Layer
  Core Mamba v2 selective state space model layer.
  ```python
  Mamba2Layer(d_model: int, d_state: int = 128, d_conv: int = 4, ...)
  ```

- `Mamba2ResidualBlock` - Keras Layer
  Residual block wrapping a Mamba2Layer with pre-normalization.
  ```python
  Mamba2ResidualBlock(d_model: int, d_state: int, d_conv: int, ...)
  ```

**Functions:** `build`, `call`, `get_config`, `call`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/mamba/components_v2.py`*

### models.mamba.mamba_v1
Mamba State Space Model Implementation

**Classes:**
- `Mamba`

**Functions:** `create_mamba_with_head`, `call`, `from_variant`, `get_config`, `from_config` (and 1 more)

*📁 File: `src/dl_techniques/models/mamba/mamba_v1.py`*

### models.mamba.mamba_v2

**Classes:**
- `Mamba2`

**Functions:** `call`, `from_variant`, `get_config`

*📁 File: `src/dl_techniques/models/mamba/mamba_v2.py`*

## Masked_Autoencoder

### models.masked_autoencoder

*📁 File: `src/dl_techniques/models/masked_autoencoder/__init__.py`*

### models.masked_autoencoder.conv_decoder
A lightweight convolutional decoder for image reconstruction.

**Classes:**

- `ConvDecoder` - Keras Layer
  Convolutional decoder for MAE reconstruction.
  ```python
  ConvDecoder(decoder_dims: List[int] = [512, 256, 128, 64], output_channels: int = 3, kernel_size: int = 3, ...)
  ```

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/masked_autoencoder/conv_decoder.py`*

### models.masked_autoencoder.mae
Masked Autoencoder (MAE) Framework

**Classes:**
- `MaskedAutoencoder`

**Functions:** `compute_output_shape`, `call`, `compute_loss`, `train_step`, `test_step` (and 4 more)

*📁 File: `src/dl_techniques/models/masked_autoencoder/mae.py`*

### models.masked_autoencoder.patch_masking
A patch-based random masking strategy for self-supervised learning.

**Classes:**

- `PatchMasking` - Keras Layer
  Layer for creating patches and applying random masking.
  ```python
  PatchMasking(patch_size: int = 16, mask_ratio: float = 0.75, mask_value: Union[str, float] = 'learnable', ...)
  ```

**Functions:** `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/masked_autoencoder/patch_masking.py`*

### models.masked_autoencoder.utils

**Functions:** `create_mae_model`, `visualize_reconstruction`

*📁 File: `src/dl_techniques/models/masked_autoencoder/utils.py`*

## Masked_Language_Model

### models.masked_language_model

*📁 File: `src/dl_techniques/models/masked_language_model/__init__.py`*

### models.masked_language_model.clm
Causal Language Model (CLM) Pre-training Framework

**Classes:**
- `CausalLanguageModel`

**Functions:** `metrics`, `build`, `call`, `train_step`, `test_step` (and 3 more)

*📁 File: `src/dl_techniques/models/masked_language_model/clm.py`*

### models.masked_language_model.mlm
Masked Language Model (MLM) Pre-training Framework

**Classes:**
- `MaskedLanguageModel`

**Functions:** `metrics`, `call`, `train_step`, `test_step`, `compute_loss` (and 2 more)

*📁 File: `src/dl_techniques/models/masked_language_model/mlm.py`*

### models.masked_language_model.utils

**Functions:** `visualize_mlm_predictions`, `create_mlm_training_model`

*📁 File: `src/dl_techniques/models/masked_language_model/utils.py`*

## Mdn

### models.mdn

*📁 File: `src/dl_techniques/models/mdn/__init__.py`*

### models.mdn.model
Mixture Density Network (MDN) Model.

**Classes:**
- `MDNModel`

**Functions:** `build`, `call`, `sample`, `predict_with_uncertainty`, `compile` (and 6 more)

*📁 File: `src/dl_techniques/models/mdn/model.py`*

## Mini_Vec2Vec

### models.mini_vec2vec
Mini-Vec2Vec: Unsupervised Embedding Space Alignment.

*📁 File: `src/dl_techniques/models/mini_vec2vec/__init__.py`*

### models.mini_vec2vec.example_alignment
Example script demonstrating MiniVec2VecAligner usage.

**Functions:** `generate_synthetic_data`, `compute_top1_accuracy`, `compute_mean_cosine_similarity`, `compute_transformation_error`, `evaluate_alignment` (and 2 more)

*📁 File: `src/dl_techniques/models/mini_vec2vec/example_alignment.py`*

### models.mini_vec2vec.model
Mini-Vec2Vec Alignment Model for Unsupervised Embedding Space Alignment.

**Classes:**
- `MiniVec2VecAligner`

**Functions:** `create_mini_vec2vec_aligner`, `build`, `call`, `align`, `get_config`

*📁 File: `src/dl_techniques/models/mini_vec2vec/model.py`*

## Mobile_Clip

### models.mobile_clip

*📁 File: `src/dl_techniques/models/mobile_clip/__init__.py`*

### models.mobile_clip.components

**Classes:**

- `ImageProjectionHead` - Keras Layer
  Projects image feature maps into a fixed-size embedding.
  ```python
  ImageProjectionHead(projection_dim: int, dropout_rate: float = 0.0, activation: Optional[Union[str, Callable]] = None, ...)
  ```
- `MobileClipImageEncoder`

- `MobileClipTextEncoder` - Keras Layer
  MobileClip Text Encoder using a stack of Transformer layers.
  ```python
  MobileClipTextEncoder(vocab_size: int, max_seq_len: int, embed_dim: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 6 more)

*📁 File: `src/dl_techniques/models/mobile_clip/components.py`*

### models.mobile_clip.mobile_clip_v1

**Classes:**
- `MobileClipModel`

**Functions:** `create_mobile_clip_model`, `build`, `encode_image`, `encode_text`, `call` (and 4 more)

*📁 File: `src/dl_techniques/models/mobile_clip/mobile_clip_v1.py`*

### models.mobile_clip.mobile_clip_v2

*📁 File: `src/dl_techniques/models/mobile_clip/mobile_clip_v2.py`*

## Mobilenet

### models.mobilenet

*📁 File: `src/dl_techniques/models/mobilenet/__init__.py`*

### models.mobilenet.mobilenet_v1
MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Applications

**Classes:**
- `MobileNetV1`

**Functions:** `create_mobilenetv1`, `call`, `from_variant`, `get_config`, `from_config` (and 1 more)

*📁 File: `src/dl_techniques/models/mobilenet/mobilenet_v1.py`*

### models.mobilenet.mobilenet_v2
MobileNetV2: Inverted Residuals and Linear Bottlenecks

**Classes:**
- `MobileNetV2`

**Functions:** `create_mobilenetv2`, `call`, `from_variant`, `get_config`, `summary`

*📁 File: `src/dl_techniques/models/mobilenet/mobilenet_v2.py`*

### models.mobilenet.mobilenet_v3
MobileNetV3: Efficient Mobile Networks with Hardware-Aware NAS

**Classes:**
- `MobileNetV3`

**Functions:** `create_mobilenetv3`, `call`, `from_variant`, `get_config`, `summary` (and 1 more)

*📁 File: `src/dl_techniques/models/mobilenet/mobilenet_v3.py`*

### models.mobilenet.mobilenet_v4
MobileNetV4: Universal and Efficient Neural Networks for Mobile Applications

**Classes:**
- `MobileNetV4`

**Functions:** `create_mobilenetv4`, `call`, `from_variant`, `get_config`, `from_config` (and 1 more)

*📁 File: `src/dl_techniques/models/mobilenet/mobilenet_v4.py`*

## Modern_Bert

### models.modern_bert

*📁 File: `src/dl_techniques/models/modern_bert/__init__.py`*

### models.modern_bert.components

**Classes:**

- `ByteTokenizer` - Keras Layer
  A simple, stateless byte-level tokenizer for text processing.
  ```python
  ByteTokenizer(vocab_size: int = 260, byte_offset: int = 4, **kwargs)
  ```

- `HashNGramEmbedding` - Keras Layer
  Computes hash n-gram embeddings for byte-level tokens.
  ```python
  HashNGramEmbedding(hash_vocab_size: int, embed_dim: int, ngram_sizes: List[int], ...)
  ```

- `ModernBertBltEmbeddings` - Keras Layer
  Combines byte, positional, and optional hash n-gram embeddings.
  ```python
  ModernBertBltEmbeddings(vocab_size: int, hidden_size: int, max_position_embeddings: int, ...)
  ```

**Functions:** `text_to_bytes`, `tokens_to_text`, `get_config`, `build`, `call` (and 6 more)

*📁 File: `src/dl_techniques/models/modern_bert/components.py`*

### models.modern_bert.modern_bert
ModernBERT: A High-Performance BERT Successor

**Classes:**
- `ModernBERT`

**Functions:** `create_modern_bert_with_head`, `call`, `load_pretrained_weights`, `from_variant`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/modern_bert/modern_bert.py`*

### models.modern_bert.modern_bert_blt
ModernBertBLT: A Modern BERT with Byte Latent Transformer Features

**Classes:**
- `ModernBertBLT`

**Functions:** `create_modern_bert_blt_with_head`, `call`, `from_variant`, `encode_text`, `decode_tokens` (and 1 more)

*📁 File: `src/dl_techniques/models/modern_bert/modern_bert_blt.py`*

### models.modern_bert.modern_bert_blt_hrm
ReasoningByteBERT: Combining ByteBERT with Hierarchical Reasoning

**Classes:**
- `ReasoningByteBertConfig`

- `HashNGramEmbedding` - Keras Layer
  Hash-based n-gram embedding layer for enhanced byte representations.
  ```python
  HashNGramEmbedding(hash_vocab_size: int, embed_dim: int, ngram_sizes: List[int] = [3, 4, 5, 6, 7, 8], ...)
  ```

- `ReasoningByteEmbeddings` - Keras Layer
  ReasoningByte embeddings combining byte-level processing with puzzle context.
  ```python
  ReasoningByteEmbeddings(config: ReasoningByteBertConfig, **kwargs)
  ```

- `ReasoningByteCore` - Keras Layer
  Core reasoning engine combining byte-level processing with hierarchical reasoning.
  ```python
  ReasoningByteCore(config: ReasoningByteBertConfig, **kwargs)
  ```
- `ReasoningByteBERT`

**Functions:** `create_reasoning_byte_bert_base`, `create_reasoning_byte_bert_large`, `create_reasoning_byte_bert_for_reasoning_tasks`, `create_fast_reasoning_byte_bert`, `validate` (and 20 more)

*📁 File: `src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py`*

## Mothnet

### models.mothnet

*📁 File: `src/dl_techniques/models/mothnet/__init__.py`*

### models.mothnet.model
MothNet: Bio-Mimetic Feature Generation for Few-Shot Learning.

**Classes:**
- `MothNet`

**Functions:** `create_cyborg_features`, `build`, `call`, `extract_features`, `extract_mb_features` (and 3 more)

*📁 File: `src/dl_techniques/models/mothnet/model.py`*

## Nano_Vlm

### models.nano_vlm

*📁 File: `src/dl_techniques/models/nano_vlm/__init__.py`*

### models.nano_vlm.model
NanoVLM: Compact Vision-Language Model - Modern Implementation

**Classes:**
- `NanoVLM`

**Functions:** `create_nanovlm`, `create_modern_nanovlm`, `build`, `call`, `generate` (and 2 more)

*📁 File: `src/dl_techniques/models/nano_vlm/model.py`*

## Nano_Vlm_World_Model

### models.nano_vlm_world_model

*📁 File: `src/dl_techniques/models/nano_vlm_world_model/__init__.py`*

### models.nano_vlm_world_model.denoisers
Denoiser Networks for Score-Based nanoVLM

**Classes:**

- `TimestepEmbedding` - Keras Layer
  Sinusoidal timestep embedding for diffusion models.
  ```python
  TimestepEmbedding(embedding_dim: int, max_period: int = 10000, **kwargs)
  ```

- `ConditionalDenoiser` - Keras Layer
  Conditional denoiser network that learns score functions.
  ```python
  ConditionalDenoiser(data_dim: int, condition_dim: int, hidden_dim: int = 512, ...)
  ```

- `VisionDenoiser` - Keras Layer
  Denoiser for image data conditioned on text.
  ```python
  VisionDenoiser(vision_config: Dict[str, Any], text_dim: int, num_layers: int = 12, ...)
  ```

- `TextDenoiser` - Keras Layer
  Denoiser for text embeddings conditioned on images.
  ```python
  TextDenoiser(text_dim: int, vision_dim: int, num_layers: int = 12, ...)
  ```

- `JointDenoiser` - Keras Layer
  Joint denoiser for simultaneous vision and text denoising.
  ```python
  JointDenoiser(vision_dim: int, text_dim: int, hidden_dim: int = 1024, ...)
  ```

**Functions:** `call`, `get_config`, `call`, `get_config`, `call` (and 5 more)

*📁 File: `src/dl_techniques/models/nano_vlm_world_model/denoisers.py`*

### models.nano_vlm_world_model.model
Score-Based nanoVLM: Navigable World Model Architecture

**Classes:**
- `ScoreBasedNanoVLM`

**Functions:** `create_score_based_nanovlm`, `build`, `call`, `generate_from_text`, `generate_from_image` (and 3 more)

*📁 File: `src/dl_techniques/models/nano_vlm_world_model/model.py`*

### models.nano_vlm_world_model.scheduler
Diffusion Schedulers for Score-Based nanoVLM

**Classes:**

- `DiffusionScheduler` - Keras Layer
  Base diffusion scheduler for score-based models.
  ```python
  DiffusionScheduler(num_timesteps: int = 1000, beta_schedule: Literal['linear', 'cosine', 'quadratic'] = 'linear', beta_start: float = 0.0001, ...)
  ```

**Functions:** `add_noise`, `get_velocity`, `predict_start_from_noise`, `get_score_from_noise`, `step` (and 1 more)

*📁 File: `src/dl_techniques/models/nano_vlm_world_model/scheduler.py`*

### models.nano_vlm_world_model.train
Training Infrastructure for Score-Based nanoVLM

**Classes:**
- `DenoisingScoreMatchingLoss`
- `VLMDenoisingLoss`
- `ScoreVLMTrainer`
- `DummyDataset`

**Functions:** `train_score_vlm`, `example_training`, `call`, `get_config`, `call` (and 4 more)

*📁 File: `src/dl_techniques/models/nano_vlm_world_model/train.py`*

## Nbeats

### models.nbeats

*📁 File: `src/dl_techniques/models/nbeats/__init__.py`*

### models.nbeats.nbeats
This model provides a pure deep learning architecture for univariate and

**Classes:**
- `NBeatsNet`

**Functions:** `create_nbeats_model`, `build`, `call`, `compute_output_shape`, `get_config` (and 1 more)

*📁 File: `src/dl_techniques/models/nbeats/nbeats.py`*

### models.nbeats.nbeatsx

**Classes:**
- `NBeatsXNet`

**Functions:** `create_nbeatsx_model`, `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/nbeats/nbeatsx.py`*

## Ntm

### models.ntm

*📁 File: `src/dl_techniques/models/ntm/__init__.py`*

### models.ntm.model
Neural Turing Machine (NTM) Model Wrapper.

**Classes:**
- `NTMModel`

**Functions:** `create_ntm_variant`, `build`, `call`, `compute_output_shape`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/ntm/model.py`*

### models.ntm.model_multitask

**Classes:**
- `NTMMultiTask`

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/models/ntm/model_multitask.py`*

## Pft_Sr

### models.pft_sr

*📁 File: `src/dl_techniques/models/pft_sr/__init__.py`*

### models.pft_sr.model
Progressive Focused Transformer for Single Image Super-Resolution (PFT-SR).

**Classes:**
- `PFTSR`

**Functions:** `create_pft_sr`, `build`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/pft_sr/model.py`*

## Power_Mlp

### models.power_mlp

*📁 File: `src/dl_techniques/models/power_mlp/__init__.py`*

### models.power_mlp.model
PowerMLP Model: Efficient Alternative to Kolmogorov-Arnold Networks

**Classes:**
- `PowerMLP`

**Functions:** `create_power_mlp`, `create_power_mlp_regressor`, `create_power_mlp_binary_classifier`, `call`, `compute_output_shape` (and 6 more)

*📁 File: `src/dl_techniques/models/power_mlp/model.py`*

## Prism

### models.prism

*📁 File: `src/dl_techniques/models/prism/__init__.py`*

### models.prism.model
PRISM: Partitioned Representation for Iterative Sequence Modeling.

**Classes:**
- `PRISMModel`

**Functions:** `build`, `call`, `predict_quantiles`, `compute_output_shape`, `from_preset` (and 2 more)

*📁 File: `src/dl_techniques/models/prism/model.py`*

## Pw_Fnet

### models.pw_fnet

*📁 File: `src/dl_techniques/models/pw_fnet/__init__.py`*

### models.pw_fnet.model
PW-FNet: Pyramid Wavelet-Fourier Network for Image Restoration.

**Classes:**

- `PW_FNet_Block` - Keras Layer
  Pyramid Wavelet-Fourier Network (PW-FNet) building block with configurable components.
  ```python
  PW_FNet_Block(dim: int, ffn_expansion_factor: float = 2.0, normalization_type: str = 'layer_norm', ...)
  ```

- `Downsample` - Keras Layer
  Trainable downsampling layer using strided convolution.
  ```python
  Downsample(dim: int, **kwargs)
  ```

- `Upsample` - Keras Layer
  Trainable upsampling layer using transposed convolution.
  ```python
  Upsample(dim: int, **kwargs)
  ```
- `PW_FNet`

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 9 more)

*📁 File: `src/dl_techniques/models/pw_fnet/model.py`*

## Qwen

### models.qwen

*📁 File: `src/dl_techniques/models/qwen/__init__.py`*

### models.qwen.components
Qwen3 Next Model Implementation

**Classes:**

- `Qwen3NextBlock` - Keras Layer
  Qwen3 Next transformer block implementing the exact architectural pattern.
  ```python
  Qwen3NextBlock(dim: int, num_heads: int, head_dim: Optional[int] = None, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`

*📁 File: `src/dl_techniques/models/qwen/components.py`*

### models.qwen.qwen3
Qwen3 Model Implementation

**Classes:**
- `Qwen3`

**Functions:** `create_qwen3_generation`, `create_qwen3_classification`, `create_qwen3`, `call`, `from_variant` (and 3 more)

*📁 File: `src/dl_techniques/models/qwen/qwen3.py`*

### models.qwen.qwen3_embeddings
Qwen3 text embedding and reranking.

**Classes:**

- `Qwen3EmbeddingLayer` - Keras Layer
  Keras implementation of the Qwen3 Text Embedding model using factory components.
  ```python
  Qwen3EmbeddingLayer(vocab_size: int, hidden_size: int = 1024, num_layers: int = 12, ...)
  ```

- `Qwen3RerankerLayer` - Keras Layer
  Keras implementation of the Qwen3 Reranker using factory components.
  ```python
  Qwen3RerankerLayer(vocab_size: int, hidden_size: int = 1024, num_layers: int = 12, ...)
  ```
- `Qwen3EmbeddingModel`
- `Qwen3RerankerModel`

**Functions:** `build`, `call`, `get_config`, `build`, `call` (and 5 more)

*📁 File: `src/dl_techniques/models/qwen/qwen3_embeddings.py`*

### models.qwen.qwen3_mega
Qwen3-MEGA: Memory-Enhanced Graph-Augmented Language Model

**Classes:**

- `MemoryIntegrationLayer` - Keras Layer
  Layer that integrates MANN memory and GNN entity graph with transformer hidden states.
  ```python
  MemoryIntegrationLayer(hidden_size: int, memory_dim: int, entity_dim: int, ...)
  ```
- `Qwen3MEGA`

**Functions:** `create_qwen3_mega`, `build`, `call`, `get_config`, `call` (and 2 more)

*📁 File: `src/dl_techniques/models/qwen/qwen3_mega.py`*

### models.qwen.qwen3_next
Qwen3 Next Model

**Classes:**
- `Qwen3Next`

**Functions:** `create_qwen3_next_generation`, `create_qwen3_next_classification`, `create_qwen3_next`, `call`, `from_variant` (and 3 more)

*📁 File: `src/dl_techniques/models/qwen/qwen3_next.py`*

### models.qwen.qwen3_omni

*📁 File: `src/dl_techniques/models/qwen/qwen3_omni.py`*

### models.qwen.qwen3_som
Qwen3-SOM: Language Model with Self-Organizing Map Memory Integration

**Classes:**
- `Qwen3SOM`

**Functions:** `create_qwen3som_generation`, `create_qwen3som_classification`, `create_qwen3som`, `visualize_som_assignments`, `analyze_som_clustering` (and 6 more)

*📁 File: `src/dl_techniques/models/qwen/qwen3_som.py`*

## Relgt

### models.relgt

*📁 File: `src/dl_techniques/models/relgt/__init__.py`*

### models.relgt.model
Modern Relational Graph Transformer (RELGT) Implementation

**Classes:**
- `RELGT`

**Functions:** `create_relgt_model`, `call`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/models/relgt/model.py`*

## Resnet

### models.resnet

*📁 File: `src/dl_techniques/models/resnet/__init__.py`*

### models.resnet.model
ResNet Model Implementation with Pretrained Support and Deep Supervision

**Classes:**
- `ResNet`

**Functions:** `get_model_output_info`, `create_inference_model_from_training_model`, `create_resnet`, `call`, `load_pretrained_weights` (and 3 more)

*📁 File: `src/dl_techniques/models/resnet/model.py`*

## Sam

### models.sam

*📁 File: `src/dl_techniques/models/sam/__init__.py`*

### models.sam.image_encoder
SAM Image Encoder (ViT) Implementation

**Classes:**

- `PatchEmbedding` - Keras Layer
  Image to Patch Embedding Layer.
  ```python
  PatchEmbedding(patch_size: Union[int, Tuple[int, int]] = 16, embed_dim: int = 768, **kwargs)
  ```

- `WindowedAttentionWithRelPos` - Keras Layer
  Multi-Head Self-Attention with optional Relative Positional Embeddings and Windowing.
  ```python
  WindowedAttentionWithRelPos(dim: int, num_heads: int = 8, qkv_bias: bool = True, ...)
  ```

- `ViTBlock` - Keras Layer
  Transformer Block for the Vision Transformer with Windowing Support.
  ```python
  ViTBlock(dim: int, num_heads: int, mlp_ratio: float = 4.0, ...)
  ```
- `ImageEncoderViT`

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 10 more)

*📁 File: `src/dl_techniques/models/sam/image_encoder.py`*

### models.sam.mask_decoder
SAM Mask Decoder Implementation

**Classes:**

- `MaskDecoder` - Keras Layer
  Predicts segmentation masks from image and prompt embeddings using a transformer.
  ```python
  MaskDecoder(**kwargs)
  ```

**Functions:** `build`, `call`, `predict_masks`, `compute_output_shape`, `get_config` (and 1 more)

*📁 File: `src/dl_techniques/models/sam/mask_decoder.py`*

### models.sam.model
Segment Anything Model (SAM)

**Classes:**
- `SAM`

**Functions:** `call`, `preprocess`, `postprocess_masks`, `from_variant`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/sam/model.py`*

### models.sam.prompt_encoder
SAM Prompt Encoder

**Classes:**

- `PositionEmbeddingRandom` - Keras Layer
  Positional encoding using random spatial frequencies.
  ```python
  PositionEmbeddingRandom(num_pos_feats: int = 64, scale: float = 1.0, **kwargs)
  ```

- `PromptEncoder` - Keras Layer
  Encodes prompts (points, boxes, masks) for the SAM mask decoder.
  ```python
  PromptEncoder(embed_dim: int, image_embedding_size: Tuple[int, int], input_image_size: Tuple[int, int], ...)
  ```

**Functions:** `build`, `call`, `forward_with_coords`, `get_config`, `build` (and 4 more)

*📁 File: `src/dl_techniques/models/sam/prompt_encoder.py`*

### models.sam.transformer
SAM Two-Way Transformer Implementation

**Classes:**

- `TwoWayAttentionBlock` - Keras Layer
  A transformer block with four layers for bidirectional attention.
  ```python
  TwoWayAttentionBlock(embedding_dim: int, num_heads: int, mlp_dim: int = 2048, ...)
  ```

- `TwoWayTransformer` - Keras Layer
  A two-way transformer decoder for joint refinement of queries and image features.
  ```python
  TwoWayTransformer(depth: int, embedding_dim: int, num_heads: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 3 more)

*📁 File: `src/dl_techniques/models/sam/transformer.py`*

## Scunet

### models.scunet

*📁 File: `src/dl_techniques/models/scunet/__init__.py`*

### models.scunet.model

**Classes:**
- `SCUNet`

**Functions:** `call`, `get_config`

*📁 File: `src/dl_techniques/models/scunet/model.py`*

## Shgcn

### models.shgcn

*📁 File: `src/dl_techniques/models/shgcn/__init__.py`*

### models.shgcn.model
Complete Simplified Hyperbolic Graph Convolutional Neural Network Model.

**Classes:**
- `SHGCNModel`
- `SHGCNNodeClassifier`
- `SHGCNLinkPredictor`

**Functions:** `call`, `get_config`, `call`, `get_config`, `call` (and 1 more)

*📁 File: `src/dl_techniques/models/shgcn/model.py`*

## Som

### models.som

*📁 File: `src/dl_techniques/models/som/__init__.py`*

### models.som.model
Self-Organizing Map for topological memory and pattern organization.

**Classes:**
- `SOMModel`

**Functions:** `build`, `call`, `train`, `fit_class_prototypes`, `predict_class` (and 7 more)

*📁 File: `src/dl_techniques/models/som/model.py`*

## Squeezenet

### models.squeezenet

*📁 File: `src/dl_techniques/models/squeezenet/__init__.py`*

### models.squeezenet.squeezenet_v1
SqueezeNet architecture for efficient classification.

**Classes:**

- `FireModule` - Keras Layer
  Fire module - the fundamental building block of SqueezeNet.
  ```python
  FireModule(s1x1: int, e1x1: int, e3x3: int, ...)
  ```
- `SqueezeNetV1`

**Functions:** `create_squeezenet_v1`, `build`, `call`, `compute_output_shape`, `get_config` (and 4 more)

*📁 File: `src/dl_techniques/models/squeezenet/squeezenet_v1.py`*

### models.squeezenet.squeezenet_v2
SqueezeNodule-Net architecture for medical imaging.

**Classes:**

- `SimplifiedFireModule` - Keras Layer
  Simplified Fire module - the core building block of SqueezeNodule-Net.
  ```python
  SimplifiedFireModule(s1x1: int, e3x3: int, kernel_regularizer: Optional[keras.regularizers.Regularizer] = None, ...)
  ```
- `SqueezeNoduleNetV2`

**Functions:** `create_squeezenodule_net_v2`, `build`, `call`, `compute_output_shape`, `get_config` (and 4 more)

*📁 File: `src/dl_techniques/models/squeezenet/squeezenet_v2.py`*

## Swin_Transformer

### models.swin_transformer

*📁 File: `src/dl_techniques/models/swin_transformer/__init__.py`*

### models.swin_transformer.model
Swin Transformer Model Implementation

**Classes:**
- `SwinTransformer`

**Functions:** `create_swin_transformer`, `from_variant`, `get_config`, `from_config`, `summary`

*📁 File: `src/dl_techniques/models/swin_transformer/model.py`*

## Tabm

### models.tabm

*📁 File: `src/dl_techniques/models/tabm/__init__.py`*

### models.tabm.model
TabM: Deep Ensemble Architecture for High-Performance Tabular Learning

**Classes:**
- `TabMModel`

**Functions:** `create_tabm_model`, `create_tabm_plain`, `create_tabm_ensemble`, `create_tabm_mini`, `ensemble_predict` (and 6 more)

*📁 File: `src/dl_techniques/models/tabm/model.py`*

## Tiny_Recursive_Model

### models.tiny_recursive_model

*📁 File: `src/dl_techniques/models/tiny_recursive_model/__init__.py`*

### models.tiny_recursive_model.components
Core reasoning modules for the Tiny Recursive Model (TRM).

**Classes:**

- `TRMReasoningModule` - Keras Layer
  A module that stacks multiple TransformerLayers for deep reasoning.
  ```python
  TRMReasoningModule(hidden_size: int, num_heads: int, expansion: float, ...)
  ```

- `TRMInner` - Keras Layer
  The inner computational core of the TRM model.
  ```python
  TRMInner(vocab_size: int, hidden_size: int, num_heads: int, ...)
  ```

**Functions:** `build`, `call`, `compute_output_shape`, `get_config`, `build` (and 2 more)

*📁 File: `src/dl_techniques/models/tiny_recursive_model/components.py`*

### models.tiny_recursive_model.model
Tiny Recursive Model (TRM) with Adaptive Computation Time (ACT).

**Classes:**
- `TRM`

**Functions:** `build`, `initial_carry`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/tiny_recursive_model/model.py`*

## Tirex

### models.tirex

*📁 File: `src/dl_techniques/models/tirex/__init__.py`*

### models.tirex.model
TiRex hybrid architecture for probabilistic time series forecasting,

**Classes:**
- `TiRexCore`

**Functions:** `create_tirex_model`, `create_tirex_by_variant`, `call`, `predict_quantiles`, `from_variant` (and 2 more)

*📁 File: `src/dl_techniques/models/tirex/model.py`*

### models.tirex.model_extended
TiRex hybrid architecture for probabilistic time series forecasting,

**Classes:**
- `TiRexExtended`

**Functions:** `create_tirex_extended`, `call`, `get_config`

*📁 File: `src/dl_techniques/models/tirex/model_extended.py`*

## Tree_Transformer

### models.tree_transformer

*📁 File: `src/dl_techniques/models/tree_transformer/__init__.py`*

### models.tree_transformer.model
Tree Transformer: Grammar Induction with Hierarchical Attention

**Classes:**

- `PositionalEncoding` - Keras Layer
  Injects sinusoidal positional encoding into input embeddings.
  ```python
  PositionalEncoding(hidden_size: int, dropout_rate: float, max_len: int = 5000, ...)
  ```

- `GroupAttention` - Keras Layer
  Hierarchical group attention for Tree Transformer.
  ```python
  GroupAttention(hidden_size: int, normalization_type: str = 'layer_norm', **kwargs)
  ```

- `TreeMHA` - Keras Layer
  Multi-Head Attention modulated by Tree Transformer group probabilities.
  ```python
  TreeMHA(num_heads: int, hidden_size: int, attention_dropout_rate: float = 0.1, ...)
  ```

- `TreeTransformerBlock` - Keras Layer
  Single block of the Tree Transformer encoder.
  ```python
  TreeTransformerBlock(hidden_size: int, num_heads: int, intermediate_size: int, ...)
  ```
- `TreeTransformer`

**Functions:** `create_tree_transformer_with_head`, `build`, `call`, `get_config`, `build` (and 15 more)

*📁 File: `src/dl_techniques/models/tree_transformer/model.py`*

## Vae

### models.vae

*📁 File: `src/dl_techniques/models/vae/__init__.py`*

### models.vae.model
Variational Autoencoder (VAE) Model Implementation

**Classes:**
- `VAE`

**Functions:** `create_vae`, `create_vae_from_config`, `from_variant`, `metrics`, `encode` (and 7 more)

*📁 File: `src/dl_techniques/models/vae/model.py`*

## Vit

### models.vit

*📁 File: `src/dl_techniques/models/vit/__init__.py`*

### models.vit.model
Vision Transformer (ViT) Model Implementation

**Classes:**
- `ViT`

**Functions:** `create_vision_transformer`, `build`, `call`, `compute_output_shape`, `get_config` (and 2 more)

*📁 File: `src/dl_techniques/models/vit/model.py`*

## Vit_Hmlp

### models.vit_hmlp

*📁 File: `src/dl_techniques/models/vit_hmlp/__init__.py`*

### models.vit_hmlp.model
Vision Transformer with Hierarchical MLP Stem - Modern Implementation

**Classes:**
- `ViTHMLP`

**Functions:** `create_vit_hmlp`, `create_inputs_with_masking`, `apply_mask_after_stem`, `build`, `call` (and 4 more)

*📁 File: `src/dl_techniques/models/vit_hmlp/model.py`*

## Vit_Siglip

### models.vit_siglip

*📁 File: `src/dl_techniques/models/vit_siglip/__init__.py`*

### models.vit_siglip.model
SigLIP Vision Transformer Model Implementation

**Classes:**
- `SigLIPVisionTransformer`

**Functions:** `create_siglip_vision_transformer`, `build`, `call`, `get_cls_token`, `get_patch_tokens` (and 5 more)

*📁 File: `src/dl_techniques/models/vit_siglip/model.py`*

## Vq_Vae

### models.vq_vae

*📁 File: `src/dl_techniques/models/vq_vae/__init__.py`*

### models.vq_vae.model
Vector Quantised Variational AutoEncoder (VQ-VAE) Implementation.

**Classes:**
- `VQVAEModel`

**Functions:** `call`, `train_step`, `test_step`, `metrics`, `encode` (and 6 more)

*📁 File: `src/dl_techniques/models/vq_vae/model.py`*

## Xlstm

### models.xlstm

*📁 File: `src/dl_techniques/models/xlstm/__init__.py`*

### models.xlstm.model
extended Long Short-Term Memory (xLSTM) architecture.

**Classes:**
- `xLSTM`

**Functions:** `call`, `get_config`, `from_config`

*📁 File: `src/dl_techniques/models/xlstm/model.py`*

## Yolo12

### models.yolo12

*📁 File: `src/dl_techniques/models/yolo12/__init__.py`*

### models.yolo12.feature_extractor
YOLOv12 Feature Extractor Implementation

**Classes:**
- `YOLOv12FeatureExtractor`

**Functions:** `create_yolov12_feature_extractor`, `build`, `call`, `compute_output_shape`, `get_config` (and 3 more)

*📁 File: `src/dl_techniques/models/yolo12/feature_extractor.py`*

### models.yolo12.multitask
YOLOv12 Multi-Task Learning Model Implementation

**Classes:**
- `YOLOv12MultiTask`

**Functions:** `create_yolov12_multitask`, `get_config`, `from_config`, `get_feature_extractor`, `get_enabled_tasks` (and 4 more)

*📁 File: `src/dl_techniques/models/yolo12/multitask.py`*