# VLM Task Heads: Integration Guide

## 1. Overview & Philosophy

The `dl_techniques.layers.vlm_heads` module provides a unified, model-agnostic system for attaching task-specific heads to any Visual Language Model (VLM) foundation model.

**Core Philosophy:** Decouple the *multi-modal encoders* from the *task-specific head*.
*   **Foundation Model Encoders**: (e.g., CLIP, BLIP, Flamingo) are responsible for producing rich, contextualized feature representations from visual and textual inputs.
*   **VLM Task Head**: Is responsible for fusing these uni-modal features and transforming them into task-specific predictions (captions, answer logits, bounding boxes, similarity scores).

This separation allows you to build complex, multi-task VLMs by combining a shared set of encoders with a collection of lightweight, specialized heads.

### Architecture: Two Types of Heads

The framework provides two distinct categories of heads, offering a balance between specialization and flexibility:

1.  **Standalone Task Architectures**: For tasks like `ImageCaptioning` and `VQA`, the heads are complete, self-contained Keras Layers with their own internal logic (e.g., a full Transformer decoder or a custom MLP classifier).
2.  **Flexible Fusion-Based Heads**: For tasks like `ImageTextMatching` and `VisualGrounding`, the heads are built upon a powerful and highly configurable `MultiModalFusion` base layer. Their behavior is primarily controlled by selecting a `fusion_strategy`.

```
Vision Input → Vision Encoder → Vision Features ↘
                                                  Task Head → Task Output
Text Input   → Text Encoder   → Text Features   ↗
```

## 2. Core Concepts: The Input/Output Contract

All VLM heads expect inputs in a dictionary format. The specific keys required depend on the task.

**Input Contract:**
```python
# A dictionary of feature tensors from the foundation encoders.
head_inputs = {
    # Vision features, can be pooled or a sequence of patches/regions
    # Shape: [Batch, Dim] or [Batch, Num_Patches, Dim]
    'vision_features': vision_tensor,

    # Text features, can be pooled or a sequence of tokens
    # Shape: [Batch, Dim] or [Batch, Seq_Len, Dim]
    'text_features': text_tensor,

    # Alias for VQA for clarity
    'question_features': text_tensor
}
```

**Output Contract:**
Heads always return a dictionary containing task-specific tensors.

| Task | Output Dictionary |
|:---|:---|
| **Image Captioning** | `{'logits': [B, Seq, Vocab], 'hidden_states': [B, Seq, Dim]}` |
| **VQA** | `{'answer_logits': [B, Num_Answers]}` |
| **Visual Grounding** | `{'bbox': [B, 4], 'confidence': [B, Regions], 'grounded_features': [B, Dim]}` |
| **Matching** | `{'logits': [B, B], 'match_score': [B], 'vision_embeddings': [B, ProjDim], ...}` |

---

## 3. Catalog of Task Heads

The factory function `create_vlm_head` automatically selects the correct head class based on the `VLMTaskType`.

### 3.1. Image Captioning
*   **Task Type**: `IMAGE_CAPTIONING`
*   **Head Class**: `ImageCaptioningHead` (Standalone Architecture)
*   **Description**: A full, autoregressive Transformer decoder that generates a text sequence conditioned on vision features.
*   **Inputs**: `vision_features` `[B, Patches, Dim]`, `text_features` (embedded partial caption) `[B, Seq, Dim]`.
*   **Key Config**: Requires `vocab_size`. Configured with `num_layers`, `num_heads`.

```python
config = VLMTaskConfig(name="coco_caption", task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=50257)
head = create_vlm_head(config, vision_dim=768, text_dim=768, num_layers=6, num_heads=12)
```

### 3.2. Visual Question Answering (VQA)
*   **Task Type**: `VISUAL_QUESTION_ANSWERING`
*   **Head Class**: `VQAHead` (Standalone Architecture)
*   **Description**: A custom classification model that pools vision and text features and passes them through an MLP to predict an answer from a fixed vocabulary.
*   **Inputs**: `vision_features` `[B, Patches, Dim]`, `question_features` `[B, Seq, Dim]`.
*   **Key Config**: Requires `num_classes`. Configured with `pooling_strategy`, `hidden_dims`.

```python
config = VLMTaskConfig(name="vqa_v2", task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING, num_classes=3129)
head = create_vlm_head(config, vision_dim=768, text_dim=768, pooling_strategy="attention", hidden_dims=[512, 256])
```

### 3.3. Visual Grounding
*   **Task Type**: `VISUAL_GROUNDING`
*   **Head Class**: `VisualGroundingHead` (Fusion-Based)
*   **Description**: Fuses a text query with each visual region feature, scores each region for relevance, and regresses a bounding box from the top-scoring region's fused features.
*   **Inputs**: `vision_features` (unpooled regions) `[B, Regions, Dim]`, `text_features` `[B, Seq, Dim]`.
*   **Key Config**: Configured with `fusion_strategy` and `fusion_config`. Defaults to `'gated'` fusion.

```python
config = VLMTaskConfig(name="refcoco", task_type=VLMTaskType.VISUAL_GROUNDING)
head = create_vlm_head(config, vision_dim=768, text_dim=768, fusion_strategy='gated')
```

### 3.4. Image-Text Matching
*   **Task Type**: `IMAGE_TEXT_MATCHING`
*   **Head Class**: `ImageTextMatchingHead` (Fusion-Based)
*   **Description**: A dual-purpose head. It projects features for CLIP-style contrastive alignment and separately fuses features to produce a fine-grained matching score for a single image-text pair.
*   **Inputs**: `vision_features` `[B, Dim]`, `text_features` `[B, Dim]` (expects pooled features).
*   **Key Config**: Configured with `projection_dim`, `temperature`, and `fusion_strategy`.

```python
config = VLMTaskConfig(name="retrieval", task_type=VLMTaskType.IMAGE_TEXT_MATCHING)
head = create_vlm_head(config, vision_dim=768, text_dim=768, projection_dim=256, fusion_strategy='concatenation')
```
---

## 4. Configuration Reference

### 4.1. Standalone Head Configurations
These heads have unique parameters for their specialized architectures.

| Head Class | Option | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `ImageCaptioningHead`| `num_layers` | `int` | `6` | Number of decoder layers in the Transformer. |
| | `num_heads` | `int` | `12`| Number of attention heads in each decoder layer. |
| | `ffn_type` | `Literal[str]`| `'swiglu'`| FFN type for decoder blocks. See FFN table below. |
| `VQAHead` | `pooling_strategy`| `Literal[str]`| `'attention'`| How to pool vision/text sequences. Options: `'mean'`, `'max'`, `'attention'`. |
| | `hidden_dims` | `List[int]` | `[512, 256]`| A list of hidden layer sizes for the final classifier MLP. |

### 4.2. Fusion-Based Head Configurations (`BaseVLMHead`)
These options, passed to `create_vlm_head`, configure the `MultiModalFusion` layer and post-fusion blocks used by heads like `VisualGroundingHead` and `ImageTextMatchingHead`.

| Option | Type | Default | Description |
|:---|:---|:---|:---|
| `vision_dim` | `int` | `768` | Dimension of the input vision features. |
| `text_dim` | `int` | `768` | Dimension of the input text features. |
| `fusion_strategy` | `FusionStrategy`| `'cross_attention'` | The core fusion mechanism. Options include:<br>`'cross_attention'`, `'concatenation'`, `'addition'`, `'multiplication'`, `'gated'`, `'bilinear'`, `'tensor_fusion'`. |
| `fusion_config` | `Dict` | `{}` | A dictionary of fine-grained parameters for the chosen `fusion_strategy`. |
| `normalization_type`| `NormalizationType`| `'layer_norm'`| Normalization after fusion. Options include:<br>`'layer_norm'`, `'rms_norm'`, `'batch_norm'`, `'zero_centered_rms_norm'`, etc. |
| `activation_type` | `ActivationType`| `'gelu'` | Activation for post-fusion layers. Options include:<br>`'gelu'`, `'relu'`, `'silu'`, `'swish'`, `'mish'`, etc. |
| `use_post_fusion_ffn`| `bool` | `True` | Adds a Transformer-style FFN block after fusion. |
| `ffn_type` | `FFNType` | `'mlp'` | Type of post-fusion FFN. Options include:<br>`'mlp'`, `'swiglu'`, `'geglu'`, `'gated_mlp'`, `'differential'`, `'residual'`, etc. |
| `ffn_expansion_factor`| `int`| `4`| Hidden dimension expansion factor for the post-fusion FFN. |

---

## 5. Deep Dive: Fusion Strategies

The choice of `fusion_strategy` for fusion-based heads is critical and task-dependent.

*   **`'cross_attention'`**: The most powerful and expressive strategy. One modality (e.g., text) queries the other (e.g., vision), allowing for deep, token-by-token reasoning. Ideal for tasks requiring fine-grained alignment and understanding.
*   **`'concatenation'`**: Simple, fast, and effective. It concatenates the pooled vision and text vectors. Excellent for matching and retrieval tasks where the subsequent MLP can learn the interactions.
*   **`'gated'`**: Learns a dynamic "gate" to weigh the importance of each modality before combining them. Useful when one modality might be more informative than the other depending on the sample (e.g., visual grounding).
*   **`'bilinear'`**: Models second-order interactions between features. More powerful than simple concatenation for capturing subtle relationships.
*   **`'tensor_fusion'`**: The most complex strategy, modeling high-order interactions via an outer product. Computationally expensive but can capture very complex correlations.

---

## 6. Multi-Task Integration Example

The `create_multi_task_vlm_head` factory simplifies building complex VLMs with a shared backbone. Crucially, it allows you to pass task-specific configurations for each head type.

```python
from dl_techniques.layers.vlm_heads import create_multi_task_vlm_head, VLMTaskConfig, VLMTaskType

# Step 1: Define multiple VLM task configs
task_configs = {
    'caption': VLMTaskConfig(name='caption', task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=50000),
    'vqa': VLMTaskConfig(name='vqa', task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING, num_classes=3000),
    'matching': VLMTaskConfig(name='matching', task_type=VLMTaskType.IMAGE_TEXT_MATCHING)
}

# Step 2: Create multi-task head with task-specific overrides
multi_head = create_multi_task_vlm_head(
    task_configs=task_configs,
    # Shared kwargs for all heads
    shared_vision_dim=768,
    shared_text_dim=768,
    # Pass parameters for specific head types via this dictionary
    task_specific_kwargs={
        'caption': {'num_layers': 4, 'num_heads': 8}, # Config for ImageCaptioningHead
        'vqa': {'pooling_strategy': 'mean', 'hidden_dims': [512]}, # Config for VQAHead
        'matching': {
            'projection_dim': 256, # Specific to ImageTextMatchingHead
            'fusion_strategy': 'concatenation' # Config for the BaseVLMHead part
        }
    }
)

# Step 3: Integrate into a model
# ... (model definition as in previous examples)
```

---

## 7. Training & Best Practices

### Loss Functions
Each VLM task requires a specific loss function.

*   **Captioning**: `SparseCategoricalCrossentropy` on token logits, often with label smoothing.
*   **VQA**: `SparseCategoricalCrossentropy` on answer logits for classification over a fixed answer set.
*   **Grounding**: A combination of a bounding box regression loss (e.g., GIoU loss) and a classification/scoring loss (e.g., `BinaryCrossentropy` on confidences).
*   **Matching (Contrastive)**: InfoNCE loss applied to the `logits` (temperature-scaled similarity matrix).
*   **Matching (Fine-grained)**: `BinaryCrossentropy` on the `match_score` for binary classification.

### Best Practices
1.  **Start with Pretrained Encoders**: Always initialize your vision and text encoders with pretrained weights (e.g., from CLIP, ViT, BERT) for faster convergence and better performance.
2.  **Gradual Unfreezing**: For the first few epochs, train only the newly added VLM heads while keeping the foundation encoders frozen. Then, unfreeze the encoders and fine-tune the entire model at a lower learning rate.
3.  **Data Balancing**: In a multi-task setting, carefully balance the data sampling ratio across tasks to prevent one task from dominating the training process.
4.  **Feature Projection**: If your vision and text encoders produce features of different dimensions, add a `keras.layers.Dense` projection layer to map one or both to a common dimension *before* passing them to the VLM head.