# VLM Task Heads Integration Guide

## Overview

This guide demonstrates how to integrate VLM (Visual Language Model) task heads with multi-modal foundation models using the `dl_techniques` framework. The task heads provide a model-agnostic interface that can work with any VLM foundation model (CLIP, BLIP, Flamingo, LLaVA, CogVLM, etc.).

## Architecture Overview

```
Vision Input → Vision Encoder → Vision Features ↘
                                                  Multi-Modal Fusion → Task Head → Task Output
Text Input   → Text Encoder   → Text Features   ↗
```

- **Vision Encoder**: Processes visual input (images/videos)
- **Text Encoder**: Processes textual input (questions, captions, descriptions)
- **Multi-Modal Fusion**: Combines vision and text representations
- **Task Head**: Transforms fused features for specific VLM tasks
- **Task Outputs**: Task-specific predictions (captions, answers, bounding boxes)

## Core Concepts

### 1. Input/Output Contract

All VLM heads expect inputs in dictionary format:

```python
# Standard input format
inputs = {
    'vision_features': vision_tensor,    # [batch, patches/height/width, dim] or [batch, dim]
    'text_features': text_tensor,        # [batch, seq_len, dim] or [batch, dim]
    'attention_mask': mask_tensor,       # Optional attention mask
    'additional_info': {...}             # Task-specific additional inputs
}
```

### 2. Task Head Outputs

Each head produces a dictionary with task-specific outputs:

```python
# Image captioning
outputs = {
    'logits': tensor,        # [batch_size, seq_len, vocab_size]
    'caption': string        # Generated caption text
}

# Visual question answering
outputs = {
    'answer': string,        # Answer text
    'answer_logits': tensor, # [batch_size, num_answers]
    'answer_type': string    # yes/no, number, other
}

# Visual grounding
outputs = {
    'bbox': tensor,          # [batch_size, 4] - [x1, y1, x2, y2]
    'confidence': tensor,    # [batch_size]
    'grounded_features': tensor
}

# Image-text matching
outputs = {
    'similarity_matrix': tensor,  # [batch_size, batch_size]
    'match_score': tensor,        # [batch_size]
    'vision_embeddings': tensor,  # [batch_size, dim]
    'text_embeddings': tensor     # [batch_size, dim]
}
```

## Integration Examples

### Example 1: CLIP with Image Captioning Head

```python
import keras
from dl_techniques.models.clip import CLIP
from dl_techniques.vlm.heads import create_vlm_head, VLMTaskConfig, VLMTaskType

# Step 1: Create CLIP foundation model
clip = CLIP(
    vision_width=768,
    vision_layers=12,
    vision_patch_size=32,
    image_size=224,
    text_width=512,
    text_layers=12,
    text_heads=8,
    text_vocab_size=49408,
    text_max_length=77
)

# Step 2: Create captioning head
caption_config = VLMTaskConfig(
    name="caption",
    task_type=VLMTaskType.IMAGE_CAPTIONING,
    vocab_size=49408,
    vision_hidden_size=768,
    text_hidden_size=512,
    fusion_type='attention'
)

caption_head = create_vlm_head(
    task_config=caption_config,
    vision_dim=768,
    text_dim=512,
    use_cross_attention=True,
    use_ffn=True
)

# Step 3: Build complete model
@keras.saving.register_keras_serializable()
class ImageCaptioner(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.caption_head = caption_head
    
    def call(self, inputs, training=None):
        # Extract features with CLIP
        vision_features = self.clip.encode_image(inputs['images'], training=training)
        
        # Prepare inputs for captioning head
        head_inputs = {
            'vision_features': vision_features,
            'text_input': inputs.get('partial_caption', None)  # For teacher forcing
        }
        
        # Generate caption
        outputs = self.caption_head(head_inputs, training=training)
        
        # Decode if not training
        if not training and 'text_input' not in inputs:
            outputs['caption'] = self.decode_caption(outputs['logits'])
        
        return outputs
    
    def decode_caption(self, logits):
        # Simple greedy decoding
        token_ids = keras.ops.argmax(logits, axis=-1)
        # Convert to text (implementation depends on tokenizer)
        return self.tokenizer.decode(token_ids)

# Usage
captioner = ImageCaptioner()
inputs = {
    'images': keras.random.uniform((8, 224, 224, 3))
}
outputs = captioner(inputs)
# outputs['logits'] shape: (8, seq_len, vocab_size)
```

### Example 2: BLIP with Visual Question Answering

```python
from dl_techniques.models.blip import BLIP
from dl_techniques.vlm.heads import VQAHead

# Step 1: Create BLIP model
blip = BLIP(
    vision_config={
        'image_size': 384,
        'patch_size': 16,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12
    },
    text_config={
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'max_length': 512
    }
)

# Step 2: Create VQA head
vqa_config = VLMTaskConfig(
    name="vqa",
    task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
    vocab_size=30522,
    num_classes=3129,  # VQA v2 answer vocabulary
    fusion_type='attention'
)

vqa_head = VQAHead(
    task_config=vqa_config,
    vision_dim=768,
    text_dim=768
)

# Step 3: Build VQA model
@keras.saving.register_keras_serializable()
class VisualQA(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blip = blip
        self.vqa_head = vqa_head
    
    def call(self, inputs, training=None):
        # Extract multi-modal features
        vision_features, text_features = self.blip(
            images=inputs['images'],
            text=inputs['questions'],
            training=training
        )
        
        # VQA head
        head_inputs = {
            'vision_features': vision_features,
            'question_features': text_features
        }
        
        outputs = self.vqa_head(head_inputs, training=training)
        return outputs

# Usage
vqa_model = VisualQA()
inputs = {
    'images': keras.random.uniform((4, 384, 384, 3)),
    'questions': keras.random.randint(0, 30522, (4, 20))  # Tokenized questions
}
outputs = vqa_model(inputs)
# outputs['answer_type'] shape: (4, 3)
# outputs['answer_logits'] shape: (4, 3129)
```

### Example 3: Flamingo-style with Visual Grounding

```python
from dl_techniques.vlm.heads import VisualGroundingHead

# Step 1: Create Flamingo-style architecture
class FlamingoLike(keras.Model):
    def __init__(self):
        super().__init__()
        # Vision encoder (e.g., NFNet)
        self.vision_encoder = self._build_vision_encoder()
        # Language model with cross-attention
        self.language_model = self._build_language_model()
        # Perceiver resampler
        self.perceiver = self._build_perceiver()
    
    def _build_vision_encoder(self):
        # Your vision encoder
        return keras.Sequential([...])
    
    def _build_language_model(self):
        # Your language model with cross-attention gates
        return keras.Sequential([...])
    
    def _build_perceiver(self):
        # Perceiver resampler for vision features
        return keras.Sequential([...])

# Step 2: Create grounding head
grounding_config = VLMTaskConfig(
    name="grounding",
    task_type=VLMTaskType.VISUAL_GROUNDING,
    fusion_type='attention',
    use_cross_attention=True
)

grounding_head = VisualGroundingHead(
    task_config=grounding_config,
    vision_dim=1024,
    text_dim=1024
)

# Step 3: Build grounding model
@keras.saving.register_keras_serializable()
class VisualGrounder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flamingo = FlamingoLike()
        self.grounding_head = grounding_head
    
    def call(self, inputs, training=None):
        # Extract features
        vision_features = self.flamingo.vision_encoder(inputs['images'], training=training)
        vision_features = self.flamingo.perceiver(vision_features, training=training)
        
        text_features = self.flamingo.language_model(
            inputs['referring_expression'],
            vision_features=vision_features,
            training=training
        )
        
        # Grounding
        head_inputs = {
            'vision_features': vision_features,
            'text_features': text_features
        }
        
        outputs = self.grounding_head(head_inputs, training=training)
        return outputs

# Usage
grounder = VisualGrounder()
inputs = {
    'images': keras.random.uniform((4, 384, 384, 3)),
    'referring_expression': keras.random.randint(0, 50000, (4, 15))
}
outputs = grounder(inputs)
# outputs['bbox'] shape: (4, 4)  # Normalized coordinates
# outputs['confidence'] shape: (4, num_regions)
```

### Example 4: Multi-Modal Matching with Contrastive Learning

```python
from dl_techniques.vlm.heads import ImageTextMatchingHead

# Step 1: Create dual-encoder architecture
class DualEncoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.vision_encoder = self._build_vision_encoder()
        self.text_encoder = self._build_text_encoder()
    
    def _build_vision_encoder(self):
        # ViT or CNN encoder
        return keras.Sequential([...])
    
    def _build_text_encoder(self):
        # BERT or similar
        return keras.Sequential([...])

# Step 2: Create matching head
matching_config = VLMTaskConfig(
    name="matching",
    task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
    fusion_type='concat'
)

matching_head = ImageTextMatchingHead(
    task_config=matching_config,
    vision_dim=768,
    text_dim=768
)

# Step 3: Build retrieval model
@keras.saving.register_keras_serializable()
class CrossModalRetrieval(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoders = DualEncoder()
        self.matching_head = matching_head
    
    def call(self, inputs, training=None):
        # Encode both modalities
        vision_features = self.encoders.vision_encoder(inputs['images'], training=training)
        text_features = self.encoders.text_encoder(inputs['texts'], training=training)
        
        # Compute matching scores
        head_inputs = {
            'vision_features': vision_features,
            'text_features': text_features
        }
        
        outputs = self.matching_head(head_inputs, training=training)
        return outputs
    
    def compute_loss(self, outputs):
        # Contrastive loss using similarity matrix
        similarity = outputs['similarity_matrix']
        labels = keras.ops.eye(keras.ops.shape(similarity)[0])
        
        loss = keras.losses.categorical_crossentropy(
            labels,
            keras.ops.softmax(similarity, axis=-1),
            from_logits=False
        )
        return keras.ops.mean(loss)

# Usage
retrieval_model = CrossModalRetrieval()
inputs = {
    'images': keras.random.uniform((16, 224, 224, 3)),
    'texts': keras.random.randint(0, 30000, (16, 77))
}
outputs = retrieval_model(inputs)
# outputs['similarity_matrix'] shape: (16, 16)
# outputs['match_score'] shape: (16,)
```

### Example 5: Dense Captioning with Region Proposals

```python
from dl_techniques.vlm.heads import DenseCaptioningHead

# Step 1: Create model with spatial features
dense_caption_config = VLMTaskConfig(
    name="dense_caption",
    task_type=VLMTaskType.DENSE_CAPTIONING,
    vocab_size=50000
)

dense_caption_head = DenseCaptioningHead(
    task_config=dense_caption_config,
    vision_dim=1024,
    text_dim=768
)

# Step 2: Build dense captioning model
@keras.saving.register_keras_serializable()
class DenseCaptioner(keras.Model):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone  # Must output spatial features
        self.dense_head = dense_caption_head
    
    def call(self, inputs, training=None):
        # Get spatial features from backbone
        features = self.backbone(inputs['images'], training=training)
        
        # Ensure features are spatial (not pooled)
        if len(keras.ops.shape(features)) == 2:
            # Reshape if needed
            features = keras.ops.reshape(features, (-1, 49, 1024))  # 7x7 patches
        
        # Dense captioning
        head_inputs = {
            'vision_features': features
        }
        
        outputs = self.dense_head(head_inputs, training=training)
        return outputs

# Usage
backbone = create_vision_backbone('resnet50')  # Your choice
dense_model = DenseCaptioner(backbone)
outputs = dense_model({'images': keras.random.uniform((4, 224, 224, 3))})
# outputs['region_bboxes'] shape: (4, 10, 4)  # Top 10 regions
# outputs['caption_logits'] shape: (4, 10, seq_len, vocab_size)
```

### Example 6: Visual Dialogue System

```python
from dl_techniques.vlm.heads import VisualDialogueHead

# Step 1: Create dialogue head
dialogue_config = VLMTaskConfig(
    name="dialogue",
    task_type=VLMTaskType.VISUAL_DIALOGUE,
    vocab_size=30000,
    fusion_type='gated'
)

dialogue_head = VisualDialogueHead(
    task_config=dialogue_config,
    vision_dim=768,
    text_dim=768
)

# Step 2: Build dialogue model
@keras.saving.register_keras_serializable()
class VisualDialogueSystem(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder = self._build_vision_encoder()
        self.text_encoder = self._build_text_encoder()
        self.dialogue_head = dialogue_head
    
    def _build_vision_encoder(self):
        # Your vision encoder
        return keras.Sequential([...])
    
    def _build_text_encoder(self):
        # Your text encoder with dialogue history
        return keras.Sequential([...])
    
    def call(self, inputs, training=None):
        # Encode image (only once per dialogue)
        vision_features = self.vision_encoder(inputs['image'], training=training)
        
        # Encode dialogue history and current utterance
        dialogue_features = self.text_encoder(
            inputs['dialogue_history'],
            training=training
        )
        current_features = self.text_encoder(
            inputs['current_utterance'],
            training=training
        )
        
        # Generate response
        head_inputs = {
            'vision_features': vision_features,
            'dialogue_history': dialogue_features,
            'current_utterance': current_features
        }
        
        if 'candidate_responses' in inputs:
            # Ranking mode
            head_inputs['candidate_responses'] = self.text_encoder(
                inputs['candidate_responses'],
                training=training
            )
        
        outputs = self.dialogue_head(head_inputs, training=training)
        return outputs

# Usage
dialogue_model = VisualDialogueSystem()
inputs = {
    'image': keras.random.uniform((1, 224, 224, 3)),
    'dialogue_history': keras.random.randint(0, 30000, (1, 5, 50)),  # 5 turns
    'current_utterance': keras.random.randint(0, 30000, (1, 20))
}
outputs = dialogue_model(inputs)
# outputs['response_logits'] shape: (1, seq_len, vocab_size)
```

### Example 7: Multi-Task VLM

```python
from dl_techniques.vlm.heads import create_multi_task_vlm_head

# Step 1: Define multiple VLM tasks
task_configs = {
    'caption': VLMTaskConfig(
        name='caption',
        task_type=VLMTaskType.IMAGE_CAPTIONING,
        vocab_size=50000
    ),
    'vqa': VLMTaskConfig(
        name='vqa',
        task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
        num_classes=3000
    ),
    'grounding': VLMTaskConfig(
        name='grounding',
        task_type=VLMTaskType.VISUAL_GROUNDING
    ),
    'matching': VLMTaskConfig(
        name='matching',
        task_type=VLMTaskType.IMAGE_TEXT_MATCHING
    )
}

# Step 2: Create multi-task head
multi_head = create_multi_task_vlm_head(
    task_configs=task_configs,
    vision_dim=768,
    text_dim=768,
    use_task_specific_projections=True
)

# Step 3: Build unified VLM
@keras.saving.register_keras_serializable()
class UnifiedVLM(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision_encoder = self._build_vision_encoder()
        self.text_encoder = self._build_text_encoder()
        self.multi_head = multi_head
    
    def _build_vision_encoder(self):
        # Shared vision encoder
        return keras.Sequential([...])
    
    def _build_text_encoder(self):
        # Shared text encoder
        return keras.Sequential([...])
    
    def call(self, inputs, task_name=None, training=None):
        # Extract shared features
        vision_features = self.vision_encoder(inputs['images'], training=training)
        
        text_inputs = inputs.get('text', inputs.get('questions', inputs.get('captions')))
        text_features = self.text_encoder(text_inputs, training=training)
        
        # Prepare head inputs
        head_inputs = {
            'vision_features': vision_features,
            'text_features': text_features
        }
        
        # Add task-specific inputs
        if 'question_features' in inputs:
            head_inputs['question_features'] = inputs['question_features']
        
        # Run specific task or all tasks
        outputs = self.multi_head(head_inputs, task_name=task_name, training=training)
        return outputs

# Usage
unified_model = UnifiedVLM()

# Run specific task
caption_outputs = unified_model(inputs, task_name='caption')

# Run all tasks
all_outputs = unified_model(inputs)
# all_outputs['caption']['logits']
# all_outputs['vqa']['answer_logits']
# all_outputs['grounding']['bbox']
# all_outputs['matching']['similarity_matrix']
```

## Advanced Configurations

### Multi-Modal Fusion Strategies

Different fusion strategies for different tasks:

```python
# Attention-based fusion (best for complex interactions)
attention_fusion = MultiModalFusion(
    fusion_type='attention',
    hidden_dim=768,
    use_layer_norm=True
)

# Gated fusion (good for selective combination)
gated_fusion = MultiModalFusion(
    fusion_type='gated',
    hidden_dim=768,
    dropout_rate=0.1
)

# Simple concatenation (efficient for matching tasks)
concat_fusion = MultiModalFusion(
    fusion_type='concat',
    hidden_dim=1536  # vision_dim + text_dim
)

# Element-wise operations (memory efficient)
multiply_fusion = MultiModalFusion(
    fusion_type='multiply',
    hidden_dim=768
)
```

### Cross-Modal Attention Options

```python
# Standard multi-head cross-attention
cross_attention_head = create_vlm_head(
    config,
    use_cross_attention=True,
    attention_type='multi_head'
)

# Differential attention for fine-grained alignment
differential_head = create_vlm_head(
    config,
    use_cross_attention=True,
    attention_type='differential'
)

# Sparse attention for efficiency
sparse_head = create_vlm_head(
    config,
    use_cross_attention=True,
    attention_type='sparse'
)
```

## Training Considerations

### Loss Functions

Different VLM tasks require appropriate loss functions:

```python
# Captioning - Cross entropy with label smoothing
def captioning_loss(y_true, y_pred):
    return keras.losses.sparse_categorical_crossentropy(
        y_true,
        y_pred['logits'],
        from_logits=True,
        label_smoothing=0.1
    )

# VQA - Multiple losses for different answer types
def vqa_loss(y_true, y_pred):
    # Answer classification loss
    answer_loss = keras.losses.sparse_categorical_crossentropy(
        y_true['answer_label'],
        y_pred['answer_logits'],
        from_logits=True
    )
    
    # Answer type loss
    type_loss = keras.losses.categorical_crossentropy(
        y_true['answer_type'],
        y_pred['answer_type'],
        from_logits=False
    )
    
    return answer_loss + 0.5 * type_loss

# Grounding - IoU loss for bounding boxes
def grounding_loss(y_true, y_pred):
    bbox_true = y_true['bbox']
    bbox_pred = y_pred['bbox']
    
    # Compute IoU
    intersection = compute_intersection(bbox_true, bbox_pred)
    union = compute_union(bbox_true, bbox_pred)
    iou = intersection / (union + 1e-7)
    
    return 1.0 - iou

# Contrastive loss for matching
def contrastive_loss(y_true, y_pred):
    similarity = y_pred['similarity_matrix']
    temperature = 0.07
    
    # Normalized temperature-scaled cross-entropy loss
    labels = keras.ops.eye(keras.ops.shape(similarity)[0])
    loss = keras.losses.categorical_crossentropy(
        labels,
        similarity / temperature,
        from_logits=True
    )
    return keras.ops.mean(loss)
```

### Multi-Task Training

```python
class MultiTaskVLMLoss(keras.losses.Loss):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {
            'caption': 1.0,
            'vqa': 1.0,
            'grounding': 1.0,
            'matching': 1.0
        }
    
    def call(self, y_true, y_pred):
        total_loss = 0
        
        for task, weight in self.task_weights.items():
            if task in y_pred:
                task_loss = self.compute_task_loss(task, y_true[task], y_pred[task])
                total_loss += weight * task_loss
        
        return total_loss
    
    def compute_task_loss(self, task, y_true, y_pred):
        if task == 'caption':
            return captioning_loss(y_true, y_pred)
        elif task == 'vqa':
            return vqa_loss(y_true, y_pred)
        elif task == 'grounding':
            return grounding_loss(y_true, y_pred)
        elif task == 'matching':
            return contrastive_loss(y_true, y_pred)
```

### Data Augmentation for VLM

```python
def augment_vlm_data(images, texts, task_type):
    # Vision augmentation
    images = keras.preprocessing.image.random_flip(images, 'horizontal')
    images = keras.preprocessing.image.random_brightness(images, 0.2)
    
    # Text augmentation (task-specific)
    if task_type == VLMTaskType.IMAGE_CAPTIONING:
        # Random masking for masked language modeling
        texts = random_mask_tokens(texts, mask_prob=0.15)
    elif task_type == VLMTaskType.VISUAL_GROUNDING:
        # Synonym replacement for robustness
        texts = synonym_replacement(texts)
    
    # Cross-modal augmentation
    if task_type == VLMTaskType.IMAGE_TEXT_MATCHING:
        # Random negative sampling
        images, texts, labels = create_negative_pairs(images, texts)
    
    return images, texts
```

## Performance Optimization

### Mixed Precision Training

```python
# Enable mixed precision for faster training
keras.mixed_precision.set_global_policy('mixed_float16')

class StableVLMHead(BaseVLMHead):
    def call(self, inputs, training=None):
        outputs = super().call(inputs, training)
        
        # Cast outputs to float32 for numerical stability
        for key in outputs:
            if keras.backend.is_tensor(outputs[key]):
                outputs[key] = keras.ops.cast(outputs[key], 'float32')
        
        return outputs
```

### Efficient Attention

```python
# Use Flash Attention for long sequences
def create_efficient_vlm_head(config):
    return create_vlm_head(
        config,
        attention_type='flash',  # If available
        use_gradient_checkpointing=True,
        max_sequence_length=512
    )
```

## Deployment Considerations

### Model Export

```python
# Export to ONNX for deployment
def export_vlm_to_onnx(model, save_path):
    dummy_inputs = {
        'images': keras.random.uniform((1, 224, 224, 3)),
        'text': keras.random.randint(0, 30000, (1, 77))
    }
    
    # Trace and export
    keras.export.onnx.export(
        model,
        save_path,
        input_signature=dummy_inputs
    )
```

### Quantization

```python
# INT8 quantization for edge deployment
def quantize_vlm_model(model):
    converter = keras.quantization.Converter()
    converter.optimizations = [keras.optimizations.DEFAULT]
    converter.target_spec.supported_types = [keras.DType.int8]
    
    quantized_model = converter.convert(model)
    return quantized_model
```

## Troubleshooting

### Common Issues

1. **Feature Dimension Mismatch**
```python
# Error: Vision and text dimensions don't match
# Solution: Use projections
head = create_vlm_head(
    config,
    vision_dim=1024,  # From vision encoder
    text_dim=768,     # From text encoder
    fusion_type='attention'  # Handles different dimensions
)
```

2. **Memory Issues with Large Models**
```python
# Solution: Use gradient checkpointing
@keras.utils.gradient_checkpointing
def call(self, inputs, training=None):
    return super().call(inputs, training)
```

3. **Slow Convergence**
```python
# Solution: Use proper initialization and warmup
optimizer = keras.optimizers.AdamW(
    learning_rate=keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=5e-5,
        first_decay_steps=1000,
        t_mul=2.0,
        alpha=0.1
    ),
    weight_decay=0.01
)
```

## Best Practices

1. **Start with Pretrained Models**: Use pretrained vision and text encoders
2. **Gradual Unfreezing**: Train heads first, then fine-tune encoders
3. **Task-Specific Data Ratio**: Balance data across tasks in multi-task setting
4. **Monitor Cross-Modal Alignment**: Track embedding similarities
5. **Use Appropriate Evaluation Metrics**: BLEU for captioning, accuracy for VQA, IoU for grounding

## Summary

The VLM Task Head system provides:

1. **Model Agnostic**: Works with any VLM foundation model
2. **Task Specialized**: Optimized architectures for each VLM task
3. **Flexible Fusion**: Multiple strategies for combining modalities
4. **Production Ready**: Full serialization and deployment support
5. **Multi-Task Support**: Unified interface for complex VLM systems

Key integration steps:
1. Choose VLM foundation model (CLIP, BLIP, Flamingo, etc.)
2. Select appropriate VLM task head
3. Configure multi-modal fusion strategy
4. Connect encoder outputs to head inputs
5. Process task outputs for your application

The heads handle the complexity of multi-modal interactions while maintaining a clean, consistent interface across all VLM tasks.