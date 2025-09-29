# VLM Task Heads Integration Guide

## Overview

This guide demonstrates how to integrate VLM (Visual Language Model) task heads with multi-modal foundation models using the `dl_techniques` framework. The task heads provide a model-agnostic interface that can work with any VLM foundation model (CLIP, BLIP, Flamingo, LLaVA, CogVLM, etc.).

This guide reflects the updated architecture where task heads are either powerful, self-contained modules (for tasks like Captioning and VQA) or flexible layers built upon a highly configurable `MultiModalFusion` component (for tasks like Matching and Grounding).

## Architecture Overview

```
Vision Input → Vision Encoder → Vision Features ↘
                                                  Task Head → Task Output
Text Input   → Text Encoder   → Text Features   ↗
```

-   **Vision Encoder**: Processes visual input (images/videos).
-   **Text Encoder**: Processes textual input (questions, captions, descriptions).
-   **Task Head**: A specialized layer that processes vision and text features for a specific VLM task.
    -   Some heads (e.g., `ImageTextMatchingHead`) use a powerful, centralized **`MultiModalFusion`** layer internally to combine representations using a variety of configurable strategies.
    -   Other heads (e.g., `ImageCaptioningHead`, `VQAHead`) implement a complete, task-specific architecture.
-   **Task Outputs**: Task-specific predictions.

## Core Concepts

### 1. Input/Output Contract

All VLM heads expect inputs in a dictionary format. The exact keys depend on the task.

```python
# General input format
inputs = {
    'vision_features': vision_tensor,    # [batch, patches/height/width, dim] or [batch, dim]
    'text_features': text_tensor,        # [batch, seq_len, dim] or [batch, dim]
    # 'question_features' is often used for VQA for clarity
}
```

### 2. Task Head Outputs

Each head produces a dictionary with task-specific outputs:

```python
# Image captioning
outputs = {
    'logits': tensor,        # [batch_size, seq_len, vocab_size]
    'hidden_states': tensor  # [batch_size, seq_len, hidden_dim]
}

# Visual question answering
outputs = {
    'answer_logits': tensor, # [batch_size, num_answers]
}

# Visual grounding
outputs = {
    'bbox': tensor,          # [batch_size, 4] - sigmoid applied, format [x1, y1, x2, y2]
    'confidence': tensor,    # [batch_size, num_regions]
    'grounded_features': tensor
}

# Image-text matching
outputs = {
    'similarity_matrix': tensor,  # [batch, batch] raw cosine similarities
    'logits': tensor,             # [batch, batch] temperature-scaled similarities for loss
    'match_score': tensor,        # [batch] fine-grained match score (0-1) for a single pair
    'vision_embeddings': tensor,  # [batch, projection_dim] L2-normalized vision embeddings
    'text_embeddings': tensor     # [batch, projection_dim] L2-normalized text embeddings
}
```

## Integration Examples

### Example 1: CLIP with Image Captioning Head

The `ImageCaptioningHead` is a full, standalone Transformer decoder.

```python
import keras
from dl_techniques.models.clip import CLIP
from dl_techniques.vlm.heads import create_vlm_head, VLMTaskConfig, VLMTaskType

# Step 1: Create CLIP foundation model
clip = CLIP(...) # Assume CLIP model is defined

# Step 2: Create captioning head configuration
caption_config = VLMTaskConfig(
    name="caption",
    task_type=VLMTaskType.IMAGE_CAPTIONING,
    vocab_size=49408,
    hidden_size=768
)

# Create the head. Note: This standalone head has its own parameters.
caption_head = create_vlm_head(
    task_config=caption_config,
    vision_dim=768,
    text_dim=768,
    num_layers=6,  # Specific to ImageCaptioningHead
    num_heads=12,
)

# Step 3: Build complete model
@keras.saving.register_keras_serializable()
class ImageCaptioner(keras.Model):
    def __init__(self, clip, caption_head, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.caption_head = caption_head

    def call(self, inputs, training=None):
        vision_features = self.clip.encode_image(inputs['images'], training=training)
        # The head expects pre-embedded text features for the partial caption
        text_features = self.clip.text_encoder(inputs['partial_caption_ids'], training=training)

        head_inputs = {
            'vision_features': vision_features,
            'text_features': text_features
        }
        
        outputs = self.caption_head(head_inputs, training=training)
        return outputs

# Usage
captioner = ImageCaptioner(clip, caption_head)
inputs = {
    'images': keras.random.uniform((8, 224, 224, 3)),
    'partial_caption_ids': keras.random.randint(0, 49408, (8, 20))
}
outputs = captioner(inputs)
# outputs['logits'] shape: (8, 20, 49408)
```

### Example 2: BLIP with Visual Question Answering Head

The `VQAHead` is a standalone classification model with its own pooling and MLP logic.

```python
from dl_techniques.models.blip import BLIP
from dl_techniques.vlm.heads import create_vlm_head, VLMTaskConfig, VLMTaskType

# Step 1: Create BLIP model
blip = BLIP(...) # Assume BLIP model is defined

# Step 2: Create VQA head
vqa_config = VLMTaskConfig(
    name="vqa",
    task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
    num_classes=3129,  # VQA v2 answer vocabulary
)

# This head has its own specific parameters like 'pooling_strategy'
vqa_head = create_vlm_head(
    task_config=vqa_config,
    vision_dim=768,
    text_dim=768,
    pooling_strategy="attention",
    hidden_dims=[512, 256]
)

# Step 3: Build VQA model
@keras.saving.register_keras_serializable()
class VisualQA(keras.Model):
    def __init__(self, blip, vqa_head, **kwargs):
        super().__init__(**kwargs)
        self.blip = blip
        self.vqa_head = vqa_head

    def call(self, inputs, training=None):
        vision_features, text_features = self.blip(
            images=inputs['images'], text=inputs['questions'], training=training
        )
        head_inputs = {
            'vision_features': vision_features,
            'question_features': text_features
        }
        outputs = self.vqa_head(head_inputs, training=training)
        return outputs

# Usage
vqa_model = VisualQA(blip, vqa_head)
inputs = {
    'images': keras.random.uniform((4, 384, 384, 3)),
    'questions': keras.random.randint(0, 30522, (4, 20))
}
outputs = vqa_model(inputs)
# outputs['answer_logits'] shape: (4, 3129)
```

### Example 3: Multi-Task VLM

```python
from dl_techniques.vlm.heads import create_multi_task_vlm_head

# Step 1: Define multiple VLM tasks
task_configs = {
    'caption': VLMTaskConfig(name='caption', task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=50000),
    'vqa': VLMTaskConfig(name='vqa', task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING, num_classes=3000),
    'matching': VLMTaskConfig(name='matching', task_type=VLMTaskType.IMAGE_TEXT_MATCHING)
}

# Step 2: Create multi-task head with task-specific overrides
multi_head = create_multi_task_vlm_head(
    task_configs=task_configs,
    shared_vision_dim=768,
    shared_text_dim=768,
    # Pass task-specific parameters via this dictionary
    task_specific_kwargs={
        'caption': {'num_layers': 4, 'num_heads': 8},
        'vqa': {'pooling_strategy': 'mean', 'hidden_dims': [512]},
        'matching': {
            'projection_dim': 256,
            'fusion_strategy': 'concatenation' # 'matching' head uses fusion
        }
    }
)

# Step 3: Build unified VLM
@keras.saving.register_keras_serializable()
class UnifiedVLM(keras.Model):
    def __init__(self, vision_encoder, text_encoder, multi_head, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multi_head = multi_head

    def call(self, inputs, task_name=None, training=None):
        vision_features = self.vision_encoder(inputs['images'], training=training)
        text_features = self.text_encoder(inputs['text'], training=training)
        
        head_inputs = {
            'vision_features': vision_features,
            'text_features': text_features,
            'question_features': text_features # Alias for VQA head
        }
        
        outputs = self.multi_head(head_inputs, task_name=task_name, training=training)
        return outputs

# Usage
# unified_model = UnifiedVLM(...)
# all_outputs = unified_model(inputs)
# all_outputs['vqa'] uses mean pooling.
# all_outputs['matching'] uses concatenation fusion.
```

## Configuring Task Heads

The framework includes two types of heads, providing a balance of specialization and flexibility.

### 1. Standalone Heads

Heads like `ImageCaptioningHead` and `VQAHead` have a self-contained, task-optimized architecture. They are configured with their own specific parameters.

-   **`ImageCaptioningHead` Config**: `num_layers`, `num_heads`, `ffn_type`.
-   **`VQAHead` Config**: `pooling_strategy` ("mean", "max", "attention"), `hidden_dims`.

### 2. Fusion-Based Heads

Heads like `ImageTextMatchingHead` and `VisualGroundingHead` are built on `BaseVLMHead` and use the powerful, centralized `MultiModalFusion` layer. You control their behavior by passing `fusion_strategy` and `fusion_config` arguments.

#### Available Fusion Strategies (`fusion_strategy`)

-   `'cross_attention'`: Deep, iterative fusion. Best for complex reasoning.
-   `'concatenation'`: Simple and efficient. Excellent for matching and retrieval.
-   `'addition'` / `'multiplication'`: Lightweight element-wise operations.
-   `'gated'`: Learns to weigh the importance of each modality.
-   `'bilinear'`: Models fine-grained, second-order interactions.
-   `'tensor_fusion'`: Models high-order interactions. Computationally intensive.

#### Configuring Fusion (`fusion_config`)

The `fusion_config` dictionary allows fine-grained control over the chosen strategy for fusion-based heads.

```python
# Example: Creating a powerful matching head with deep cross-attention for its scoring part
matching_head = create_vlm_head(
    task_config=matching_config,
    vision_dim=768, text_dim=768,
    projection_dim=256, # Specific to matching head
    fusion_strategy='cross_attention', # For the match_score part
    fusion_config={
        'num_fusion_layers': 4,
        'attention_config': {'num_heads': 12},
        'ffn_config': {'hidden_dim': 3072},
    }
)
```

## Training Considerations

### Loss Functions

Different VLM tasks require appropriate loss functions.

```python
# Captioning - Cross entropy with label smoothing
def captioning_loss(y_true, y_pred):
    return keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred['logits'], from_logits=True, label_smoothing=0.1
    )

# VQA - Sparse cross entropy for classification over a fixed answer set
def vqa_loss(y_true, y_pred):
    return keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred['answer_logits'], from_logits=True
    )

# Grounding - IoU loss for bounding boxes
def grounding_loss(y_true, y_pred):
    # Assumes y_true and y_pred['bbox'] are [x1, y1, x2, y2]
    # ... IoU calculation logic ...
    iou_loss = 1.0 - iou
    return iou_loss

# Contrastive loss for matching (InfoNCE style)
def contrastive_loss(y_true, y_pred):
    # y_pred['logits'] contains the temperature-scaled similarity matrix
    # y_true can be ignored here; the labels are identity matrix
    logits = y_pred['logits']
    labels = keras.ops.eye(keras.ops.shape(logits)[0])
    # Symmetrical loss
    loss_i2t = keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
    loss_t2i = keras.losses.categorical_crossentropy(keras.ops.transpose(labels), keras.ops.transpose(logits), from_logits=True)
    return (keras.ops.mean(loss_i2t) + keras.ops.mean(loss_t2i)) / 2
```

### Multi-Task Training

```python
class MultiTaskVLMLoss(keras.losses.Loss):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {
            'caption': 1.0, 'vqa': 1.0, 'matching': 1.0
        }
    
    def call(self, y_true, y_pred):
        total_loss = 0
        for task, weight in self.task_weights.items():
            if task in y_pred:
                task_loss = self.compute_task_loss(task, y_true[task], y_pred[task])
                total_loss += weight * task_loss
        return total_loss
    
    def compute_task_loss(self, task, y_true, y_pred):
        if task == 'caption': return captioning_loss(y_true, y_pred)
        elif task == 'vqa': return vqa_loss(y_true, y_pred)
        elif task == 'matching': return contrastive_loss(y_true, y_pred)
        return 0.0
```

## Performance Optimization

### Mixed Precision Training

```python
# Enable mixed precision for faster training and reduced memory usage
keras.mixed_precision.set_global_policy('mixed_float16')

# Ensure output layers are float32 for numerical stability
# This is handled internally in Keras; layers with softmax/crossentropy
# are typically kept in float32. For manual control:
class StableOutputModel(keras.Model):
    def __init__(self, vlm_model):
        super().__init__()
        self.vlm_model = vlm_model
        # The output layer of the head can be explicitly cast if needed
        self.vlm_model.multi_head.head_vqa.output_layer.dtype_policy = 'float32'
    
    def call(self, inputs):
        return self.vlm_model(inputs)
```

## Troubleshooting

### Common Issues

1.  **Feature Dimension Mismatch**
    ```python
    # Error: Vision (1024) and text (768) dims don't match.
    # Solution: Manually project features to a common dimension *before* the head.
    # This is the most robust solution that works for all head types.
    vision_proj = keras.layers.Dense(768, name="vision_projection")(vision_features)
    head_inputs = {'vision_features': vision_proj, 'text_features': text_features}
    
    # Now you can use any head with matching 768 dims.
    outputs = some_head(head_inputs)
    ```

2.  **Memory Issues with Large Models**
    ```python
    # Solution: Use gradient checkpointing on expensive layers
    # This can be applied to encoders or large custom layers.
    class CheckpointedCLIP(CLIP):
        @keras.utils.gradient_checkpointing
        def call(self, inputs, training=None):
            return super().call(inputs, training)
    ```

3.  **Slow Convergence**
    ```python
    # Solution: Use proper initialization and a learning rate scheduler with warmup
    optimizer = keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4, decay_steps=10000, warmup_steps=1000
        ),
        weight_decay=0.01
    )
    ```

## Best Practices

1.  **Start with Pretrained Models**: Use pretrained vision and text encoders for better performance and faster convergence.
2.  **Gradual Unfreezing**: Train the newly added heads first with encoders frozen, then gradually unfreeze and fine-tune the encoder backbones at a lower learning rate.
3.  **Task-Specific Data Ratio**: Carefully balance the data sampling ratio across tasks in a multi-task setting.
4.  **Monitor Cross-Modal Alignment**: During training of matching tasks, track the similarity of vision and text embeddings for matched pairs to ensure alignment.
5.  **Use Appropriate Evaluation Metrics**: BLEU/CIDEr for captioning, VQA-Acc for VQA, IoU for grounding.

## Summary

The VLM Task Head system provides:

1.  **Model Agnostic**: Works with any VLM foundation model.
2.  **Task Specialized**: Provides both highly optimized standalone architectures and flexible fusion-based heads.
3.  **Configurable**: Simple, high-level APIs for configuring complex behaviors.
4.  **Production Ready**: Full serialization and deployment support.
5.  **Multi-Task Support**: Unified interface for building complex VLM systems.

Key integration steps:
1.  Choose a VLM foundation model and get its feature outputs.
2.  Select the appropriate VLM task head using `create_vlm_head`.
3.  Configure the head with its specific parameters (e.g., `num_layers`, `pooling_strategy`, or `fusion_strategy`).
4.  Connect encoder outputs to the head's input dictionary.
5.  Process the task-specific outputs for your application.
