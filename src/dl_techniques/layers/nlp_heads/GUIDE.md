# NLP Task Heads Integration Guide

## Overview

This guide demonstrates how to integrate NLP task heads with foundation models using the `dl_techniques` framework. The task heads provide a model-agnostic interface that can work with any NLP foundation model (BERT, ModernBERT, Qwen3, GPT-style models, etc.).

## Architecture Overview

```
Foundation Model → Hidden States → Task Head → Task Outputs
       ↓                ↓              ↓            ↓
   (BERT, GPT)    [B, L, D]      (NLP Head)   {logits, scores}
```

- **Foundation Model**: Produces contextualized representations
- **Hidden States**: Shape `[batch_size, sequence_length, hidden_dim]`
- **Task Head**: Transforms representations for specific tasks
- **Task Outputs**: Task-specific predictions

## Core Concepts

### 1. Input/Output Contract

All NLP heads expect inputs in one of these formats:

```python
# Format 1: Direct tensor
hidden_states = foundation_model(input_ids)  # [batch_size, seq_len, hidden_dim]

# Format 2: Dictionary with metadata
inputs = {
    'hidden_states': hidden_states,      # Required
    'attention_mask': attention_mask,    # Optional, for masking padded tokens
    'pooler_output': pooler_output       # Optional, for sentence-level tasks
}
```

### 2. Task Head Outputs

Each head produces a dictionary with task-specific outputs:

```python
# Classification tasks
outputs = {
    'logits': tensor,        # [batch_size, num_classes]
    'probabilities': tensor  # [batch_size, num_classes] (softmax applied)
}

# Token classification
outputs = {
    'logits': tensor,        # [batch_size, seq_len, num_classes]
    'predictions': tensor    # [batch_size, seq_len] (argmax)
}

# Question answering
outputs = {
    'start_logits': tensor,  # [batch_size, seq_len]
    'end_logits': tensor     # [batch_size, seq_len]
}

# Similarity tasks
outputs = {
    'similarity_score': tensor,  # [batch_size]
    'embeddings': tensor         # [batch_size, hidden_dim]
}
```

## Integration Examples

### Example 1: BERT with Classification Head

```python
import keras
from dl_techniques.models.bert import BERT
from dl_techniques.nlp.heads import create_nlp_head, NLPTaskConfig, NLPTaskType

# Step 1: Create foundation model
bert = BERT(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    add_pooling_layer=True  # Important for classification
)

# Step 2: Create task head
sentiment_config = NLPTaskConfig(
    name="sentiment",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3,  # negative, neutral, positive
    dropout_rate=0.1
)

sentiment_head = create_nlp_head(
    task_config=sentiment_config,
    input_dim=768,  # Must match BERT hidden_size
    pooling_type='cls',  # Use [CLS] token
    use_intermediate=True,
    intermediate_size=256
)

# Step 3: Build complete model
@keras.saving.register_keras_serializable()
class SentimentAnalyzer(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert
        self.head = sentiment_head
    
    def call(self, inputs, training=None):
        # inputs should have 'input_ids', 'attention_mask', 'token_type_ids'
        bert_outputs = self.bert(inputs, training=training)
        sequence_output, pooled_output = bert_outputs
        
        # Pass to head
        head_inputs = {
            'hidden_states': pooled_output,  # Use pooled for classification
            'attention_mask': inputs.get('attention_mask')
        }
        
        return self.head(head_inputs, training=training)

# Usage
model = SentimentAnalyzer()
inputs = {
    'input_ids': keras.ops.ones((32, 128), dtype='int32'),
    'attention_mask': keras.ops.ones((32, 128)),
    'token_type_ids': keras.ops.zeros((32, 128), dtype='int32')
}
outputs = model(inputs)
# outputs['logits'] shape: (32, 3)
# outputs['probabilities'] shape: (32, 3)
```

### Example 2: ModernBERT with NER Head

```python
from dl_techniques.models.modern_bert import ModernBERT
from dl_techniques.nlp.heads import TokenClassificationHead

# Step 1: Create ModernBERT
modern_bert = ModernBERT(
    vocab_size=50265,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
    rope_percentage=0.5  # Modern features
)

# Step 2: Create NER head
ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9,  # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
    dropout_rate=0.1,
    use_crf=False  # Set True for CRF layer
)

ner_head = create_nlp_head(
    task_config=ner_config,
    input_dim=768,
    use_intermediate=True,
    use_task_attention=True,  # Add attention for better token classification
    attention_type='multi_head'
)

# Step 3: Build NER model
@keras.saving.register_keras_serializable()
class NERModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert = modern_bert
        self.head = ner_head
    
    def call(self, inputs, training=None):
        # Get sequence output (not pooled)
        bert_outputs = self.bert(inputs, training=training)
        sequence_output = bert_outputs[0] if isinstance(bert_outputs, tuple) else bert_outputs
        
        # Pass sequence output to token classification head
        head_inputs = {
            'hidden_states': sequence_output,
            'attention_mask': inputs.get('attention_mask')
        }
        
        return self.head(head_inputs, training=training)

# Usage
ner_model = NERModel()
outputs = ner_model(inputs)
# outputs['logits'] shape: (32, 128, 9)
# outputs['predictions'] shape: (32, 128)
```

### Example 3: Qwen3 with Question Answering

```python
from dl_techniques.models.qwen3 import Qwen3
from dl_techniques.nlp.heads import QuestionAnsweringHead

# Step 1: Create Qwen3 model
qwen3 = Qwen3(
    vocab_size=151936,
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,  # GQA
    intermediate_size=4096,
    max_position_embeddings=2048,
    rope_theta=10000.0
)

# Step 2: Create QA head
qa_config = NLPTaskConfig(
    name="squad",
    task_type=NLPTaskType.QUESTION_ANSWERING,
    dropout_rate=0.1
)

qa_head = create_nlp_head(
    task_config=qa_config,
    input_dim=1024,
    use_intermediate=True,
    intermediate_size=512,
    use_task_attention=True  # Helps with span selection
)

# Step 3: Build QA model
@keras.saving.register_keras_serializable()
class QAModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qwen = qwen3
        self.head = qa_head
    
    def call(self, inputs, training=None):
        # Process through Qwen3
        qwen_outputs = self.qwen(inputs, training=training)
        sequence_output = qwen_outputs[0] if isinstance(qwen_outputs, tuple) else qwen_outputs
        
        # QA head needs sequence output
        head_inputs = {
            'hidden_states': sequence_output,
            'attention_mask': inputs.get('attention_mask')
        }
        
        outputs = self.head(head_inputs, training=training)
        
        # Post-process to get answer spans
        start_positions = keras.ops.argmax(outputs['start_logits'], axis=-1)
        end_positions = keras.ops.argmax(outputs['end_logits'], axis=-1)
        
        outputs['answer_spans'] = keras.ops.stack([start_positions, end_positions], axis=-1)
        
        return outputs

# Usage
qa_model = QAModel()
outputs = qa_model(inputs)
# outputs['start_logits'] shape: (32, 128)
# outputs['end_logits'] shape: (32, 128)
# outputs['answer_spans'] shape: (32, 2)
```

### Example 4: Multi-Task Learning with Shared Encoder

```python
from dl_techniques.nlp.heads import create_multi_task_nlp_head

# Step 1: Create shared encoder (using BERT)
shared_bert = BERT(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    add_pooling_layer=True
)

# Step 2: Define multiple tasks
task_configs = {
    'sentiment': NLPTaskConfig(
        name='sentiment',
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=3
    ),
    'ner': NLPTaskConfig(
        name='ner',
        task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        num_classes=9
    ),
    'similarity': NLPTaskConfig(
        name='similarity',
        task_type=NLPTaskType.TEXT_SIMILARITY
    )
}

# Step 3: Create multi-task head
multi_head = create_multi_task_nlp_head(
    task_configs=task_configs,
    input_dim=768,
    use_task_specific_projections=True  # Task-specific transformations
)

# Step 4: Build multi-task model
@keras.saving.register_keras_serializable()
class MultiTaskModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = shared_bert
        self.heads = multi_head
    
    def call(self, inputs, task_name=None, training=None):
        # Get BERT outputs
        bert_outputs = self.encoder(inputs, training=training)
        sequence_output, pooled_output = bert_outputs
        
        # Prepare inputs for heads
        head_inputs = {
            'hidden_states': sequence_output,  # Heads will pool if needed
            'attention_mask': inputs.get('attention_mask')
        }
        
        # Run specific task or all tasks
        return self.heads(head_inputs, task_name=task_name, training=training)

# Usage
multi_model = MultiTaskModel()

# Run single task
sentiment_outputs = multi_model(inputs, task_name='sentiment')

# Run all tasks
all_outputs = multi_model(inputs)
# all_outputs['sentiment']['logits'] shape: (32, 3)
# all_outputs['ner']['logits'] shape: (32, 128, 9)
# all_outputs['similarity']['embeddings'] shape: (32, 768)
```

### Example 5: Text Similarity with Sentence-Pair Input

```python
from dl_techniques.nlp.heads import TextSimilarityHead

# Create similarity head with cosine similarity
similarity_config = NLPTaskConfig(
    name="sts",
    task_type=NLPTaskType.TEXT_SIMILARITY,
    dropout_rate=0.1
)

similarity_head = TextSimilarityHead(
    task_config=similarity_config,
    input_dim=768,
    output_embeddings=True,
    similarity_function='cosine',  # or 'dot', 'learned'
    use_intermediate=True
)

@keras.saving.register_keras_serializable()
class SentenceSimilarityModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = shared_bert
        self.head = similarity_head
    
    def encode_sentence(self, inputs, training=None):
        """Encode a single sentence to embedding."""
        bert_outputs = self.encoder(inputs, training=training)
        _, pooled_output = bert_outputs
        return pooled_output
    
    def call(self, inputs, training=None):
        # Inputs should contain sentence1 and sentence2
        if 'sentence1' in inputs and 'sentence2' in inputs:
            # Encode both sentences
            emb1 = self.encode_sentence(inputs['sentence1'], training)
            emb2 = self.encode_sentence(inputs['sentence2'], training)
            
            # Compute similarity
            return self.head((emb1, emb2), training=training)
        else:
            # Single sentence - return embedding
            emb = self.encode_sentence(inputs, training)
            return self.head(emb, training=training)

# Usage - pairwise similarity
sim_model = SentenceSimilarityModel()
pair_inputs = {
    'sentence1': {
        'input_ids': keras.ops.ones((32, 128), dtype='int32'),
        'attention_mask': keras.ops.ones((32, 128))
    },
    'sentence2': {
        'input_ids': keras.ops.ones((32, 128), dtype='int32'),
        'attention_mask': keras.ops.ones((32, 128))
    }
}
outputs = sim_model(pair_inputs)
# outputs['similarity_score'] shape: (32,)
# outputs['embeddings_1'] shape: (32, 768)
# outputs['embeddings_2'] shape: (32, 768)
```

## Advanced Configurations

### Pooling Strategies

Different tasks benefit from different pooling strategies:

```python
# CLS pooling (BERT-style) - good for classification
cls_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    pooling_type='cls'  # Uses first token
)

# Mean pooling - good for similarity
mean_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    pooling_type='mean'  # Average all tokens
)

# Max pooling - captures most salient features
max_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    pooling_type='max'
)

# Attention pooling - learns to weight tokens
attention_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    pooling_type='attention'  # Learnable attention weights
)
```

### FFN and Attention Options

Enhance heads with additional processing:

```python
# Add task-specific attention
enhanced_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    use_task_attention=True,
    attention_type='multi_head',  # or 'cbam', 'differential'
    use_ffn=True,
    ffn_type='swiglu',  # or 'mlp', 'glu', 'geglu'
    ffn_expansion_factor=4
)
```

### Normalization Options

Different normalization strategies:

```python
# Standard layer normalization
standard_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    normalization_type='layer_norm'
)

# RMS normalization (faster)
rms_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    normalization_type='rms_norm'
)

# Band RMS (more stable)
band_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    normalization_type='band_rms'
)
```

## Training Considerations

### Loss Functions

Different tasks require different loss functions:

```python
# Classification
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Token classification
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'  # Apply mask for padding
)

# Question answering
def qa_loss(y_true, y_pred):
    start_loss = keras.losses.sparse_categorical_crossentropy(
        y_true['start_positions'], y_pred['start_logits'], from_logits=True
    )
    end_loss = keras.losses.sparse_categorical_crossentropy(
        y_true['end_positions'], y_pred['end_logits'], from_logits=True
    )
    return (start_loss + end_loss) / 2

# Similarity (if training with labels)
loss_fn = keras.losses.MeanSquaredError()  # For continuous scores
```

### Masking Padded Tokens

For token-level tasks, properly mask padded positions:

```python
def masked_loss(y_true, y_pred, mask):
    loss = keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    mask = keras.ops.cast(mask, loss.dtype)
    loss = loss * mask
    return keras.ops.sum(loss) / keras.ops.maximum(keras.ops.sum(mask), 1.0)
```

### Fine-tuning Strategies

```python
# Freeze foundation model initially
foundation_model.trainable = False
head.trainable = True

# Train head for a few epochs
model.fit(train_data, epochs=3)

# Unfreeze and fine-tune with lower learning rate
foundation_model.trainable = True
optimizer = keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(train_data, epochs=10)
```

## Performance Optimization

### Efficient Configurations

For faster inference:

```python
# Minimal configuration
efficient_config = {
    'use_intermediate': False,  # Skip intermediate layer
    'use_task_attention': False,  # No extra attention
    'use_ffn': False,  # No FFN block
    'dropout_rate': 0.0  # No dropout for inference
}

efficient_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    **efficient_config
)
```

### High-Performance Configuration

For best accuracy:

```python
# Maximum configuration
hp_config = {
    'use_intermediate': True,
    'intermediate_size': 512,
    'use_task_attention': True,
    'attention_type': 'differential',
    'use_ffn': True,
    'ffn_type': 'swiglu',
    'ffn_expansion_factor': 8,
    'normalization_type': 'zero_centered_rms_norm'
}

hp_head = create_nlp_head(
    task_config=config,
    input_dim=768,
    **hp_config
)
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**
```python
# Error: Input dimension doesn't match head expectation
# Solution: Ensure input_dim matches foundation model's hidden_size
head = create_nlp_head(
    task_config=config,
    input_dim=foundation_model.config.hidden_size  # Use model's actual size
)
```

2. **Missing Pooling Layer**
```python
# Error: BERT doesn't output pooled representation
# Solution: Ensure add_pooling_layer=True for classification tasks
bert = BERT(..., add_pooling_layer=True)
```

3. **Wrong Input Format**
```python
# Error: Head expects dictionary but got tensor
# Solution: Wrap tensor in dictionary
hidden_states = model(inputs)
head_inputs = {'hidden_states': hidden_states}
outputs = head(head_inputs)
```

## Summary

The NLP task head system provides:

1. **Model Agnostic**: Works with any foundation model
2. **Task Specific**: Optimized for each NLP task
3. **Configurable**: Extensive customization options
4. **Production Ready**: Full serialization and type safety
5. **Efficient**: Options for both speed and accuracy

Key integration steps:
1. Choose foundation model (BERT, ModernBERT, Qwen3, etc.)
2. Create task configuration
3. Initialize task head with matching dimensions
4. Connect model outputs to head inputs
5. Process head outputs for your application

The heads handle the complexity of task-specific transformations while maintaining a clean interface with foundation models.