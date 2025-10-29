# NLP Task Heads

## 1. Overview & Philosophy

The `dl_techniques.layers.nlp_heads` module provides a unified, model-agnostic system for attaching task-specific "heads" to any NLP foundation model.

**Core Philosophy:** Decouple the *encoder* (foundation model) from the *decoder* (task head).
*   **Foundation Model**: (e.g., BERT, RoBERTa, T5 Encoder, GPT, custom Transformers) is solely responsible for producing rich, contextualized hidden states from raw text.
*   **Task Head**: Is solely responsible for transforming those hidden states into task-specific predictions (logits, scores, embeddings).

This separation allows you to reuse the same foundation model backbone for dozens of different tasks simultaneously, switching only the lightweight head at the end.

### The Integration Contract

All heads strictly adhere to the following input/output contract, ensuring compatibility across different foundation models:

**Input Contract:**
Heads accept a generic dictionary. The only required key is `'hidden_states'`, representing the sequence output of the foundation model.

```python
head_inputs = {
    # REQUIRED: The full sequence of hidden states from the encoder.
    # Shape: [Batch_Size, Sequence_Length, Hidden_Dimension]
    'hidden_states': encoder_outputs,

    # OPTIONAL: Used by some heads for masking (pooling, attention).
    # Shape: [Batch_Size, Sequence_Length]
    'attention_mask': mask_tensor
}
```

**Output Contract:**
Heads always return a dictionary containing task-specific tensors (e.g., `'logits'`, `'probabilities'`, `'embeddings'`).

---

## 2. Quick Start: The Universal Pattern

Regardless of the specific task, the integration pattern is always the same:

```python
import keras
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig, NLPTaskType

# 1. Load ANY foundation model (generic placeholder here)
foundation_model = load_my_foundation_model(...)
hidden_dim = foundation_model.config.hidden_size  # e.g., 768, 1024

# 2. Define generic task configuration
task_config = NLPTaskConfig(
    name="my_task",
    task_type=NLPTaskType.TEXT_CLASSIFICATION,
    num_classes=2,
)

# 3. Create the head, telling it the input dimension it should expect
head = create_nlp_head(task_config=task_config, input_dim=hidden_dim)

# 4. Stitch them together (Functional API example)
inputs = keras.Input(shape=(None,), dtype="int32")
# Assume foundation model returns a dict with 'last_hidden_state'
encoder_outputs = foundation_model(inputs)

head_inputs = {
    'hidden_states': encoder_outputs['last_hidden_state'],
    # Pass mask if your foundation model uses one
    'attention_mask': inputs_mask if exists else None
}
task_outputs = head(head_inputs)

model = keras.Model(inputs=inputs, outputs=task_outputs)
```

---

## 3. Catalog of Task Heads

Different tasks require different specialized architectures. The factory automatically selects the correct class based on the `NLPTaskType`.

### 3.1. Sequence Classification
*   **Task Types**: `TEXT_CLASSIFICATION`, `SENTIMENT_ANALYSIS`, `INTENT_CLASSIFICATION`, etc.
*   **Supported Inputs**: `[B, Seq, Dim]` (will be pooled) or `[B, Dim]` (already pooled).
*   **Outputs**: `logits` `[B, NumClasses]`, `probabilities` `[B, NumClasses]`.
*   **Key Config**: Requires `num_classes` in `NLPTaskConfig`.

```python
config = NLPTaskConfig(name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3)
head = create_nlp_head(config, input_dim=768, pooling_type='cls') # Standard BERT-style pooling
```

### 3.2. Token Classification
*   **Task Types**: `NAMED_ENTITY_RECOGNITION`, `PART_OF_SPEECH_TAGGING`, `TOKEN_CLASSIFICATION`.
*   **Supported Inputs**: Must be full sequence `[B, Seq, Dim]`.
*   **Outputs**: `logits` `[B, Seq, NumClasses]`, `predictions` `[B, Seq]` (argmax indices).
*   **Key Config**: Requires `num_classes`.

```python
config = NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9)
# Typically uses full sequence, so no pooling needed.
head = create_nlp_head(config, input_dim=768)
```

### 3.3. Question Answering (Extractive)
*   **Task Types**: `QUESTION_ANSWERING`, `SPAN_EXTRACTION`.
*   **Supported Inputs**: Full sequence `[B, Seq, Dim]`.
*   **Outputs**: `start_logits` `[B, Seq]`, `end_logits` `[B, Seq]`.

```python
config = NLPTaskConfig(name="squad", task_type=NLPTaskType.QUESTION_ANSWERING)
head = create_nlp_head(config, input_dim=768)
```

### 3.4. Text Similarity & Embeddings
*   **Task Types**: `TEXT_SIMILARITY`, `PARAPHRASE_DETECTION`.
*   **Supported Inputs**:
    *   *Single Input* `[B, Seq, Dim]`: Returns embeddings.
    *   *Pair Tuple* `([B, Seq, Dim], [B, Seq, Dim])`: Returns similarity score between pairs.
*   **Outputs**: `embeddings` `[B, Dim]`, `similarity_score` `[B]` (if pair input).
*   **Key Config**: `similarity_function` ('cosine', 'dot', 'learned').

```python
config = NLPTaskConfig(name="sts", task_type=NLPTaskType.TEXT_SIMILARITY)
head = create_nlp_head(config, input_dim=768, similarity_function='cosine')

# Usage 1: Get Embeddings
embeds = head({'hidden_states': seq1})['embeddings']

# Usage 2: Get Similarity
score = head((seq1, seq2))['similarity_score']
```

### 3.5. Text Generation (Language Modeling)
*   **Task Types**: `TEXT_GENERATION`, `MASKED_LANGUAGE_MODELING`.
*   **Supported Inputs**: Full sequence `[B, Seq, Dim]`.
*   **Outputs**: `logits` `[B, Seq, VocabSize]`.
*   **Key Config**: Requires `vocabulary_size` in `NLPTaskConfig`.

```python
config = NLPTaskConfig(name="mlm", task_type=NLPTaskType.MASKED_LANGUAGE_MODELING, vocabulary_size=30522)
head = create_nlp_head(config, input_dim=768)
```

### 3.6. Multiple Choice
*   **Task Types**: `MULTIPLE_CHOICE`.
*   **Supported Inputs**: 4D Tensor `[B, NumChoices, Seq, Dim]`.
*   **Outputs**: `logits` `[B, NumChoices]`, `probabilities` `[B, NumChoices]`.

```python
config = NLPTaskConfig(name="swag", task_type=NLPTaskType.MULTIPLE_CHOICE)
# Head automatically handles reshaping the 4D input to pool each choice independently
head = create_nlp_head(config, input_dim=768, pooling_type='cls')
```

---

## 4. Configuration Reference

The system is highly configurable. You can inject intermediate layers, advanced attention mechanisms, or specialized pooling into *any* head without changing the foundation model.

### 4.1. Generic Head Options (`create_nlp_head` kwargs)
These options apply to almost all head types.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | **Required** | The hidden dimension size of the foundation model's output. |
| `normalization_type`| `Literal[str]`| `'layer_norm'`| Normalization layer. Options include:<br>`'layer_norm'`, `'rms_norm'`, `'batch_norm'`, `'zero_centered_rms_norm'`, `'band_rms'`, `'logit_norm'`, etc. |
| `activation_type` | `Literal[str]`| `'gelu'` | Activation function for intermediate layers. Options include:<br>`'gelu'`, `'relu'`, `'silu'`, `'swish'`, `'mish'`, etc. |
| `use_pooling` | `bool`| varies | Whether to pool the sequence into a single vector. |
| `pooling_type` | `Literal[str]` | `'cls'` | Strategy if pooling is enabled. Options:<br>`'cls'`, `'mean'`, `'max'`, `'attention'`. |
| `use_intermediate`| `bool`| `True` | Adds a Dense block before the final task projection. |
| `intermediate_size` | `int` | `input_dim` | Size of the intermediate dense block. |
| `use_task_attention`| `bool`| `False` | Adds a self-attention layer specific to this task. |
| `attention_type` | `Literal[str]` | `'multi_head'`| Type of task attention. Options include:<br>`'multi_head'`, `'cbam'`, `'differential'`, `'group_query'`, `'perceiver'`, `'window'`, etc. |
| `use_ffn` | `bool`| `False` | Adds a Transformer-style FFN block to the head. |
| `ffn_type` | `Literal[str]` | `'mlp'` | Type of FFN. Options include:<br>`'mlp'`, `'swiglu'`, `'geglu'`, `'gated_mlp'`, `'differential'`, `'residual'`, `'swin_mlp'`, etc. |
| `dropout_rate` | `float`| `0.1` | Dropout applied within the head components. |

### 4.2. Task Configuration (`NLPTaskConfig`)
Used to define the high-level properties of the task itself.

| Field | Description |
| :--- | :--- |
| `name` | Unique identifier for the task (e.g., "ner_v2"). |
| `task_type` | Enum from `NLPTaskType` determining the head architecture. |
| `num_classes` | Required for all classification tasks. |
| `vocabulary_size`| Required for generation/MLM tasks. |
| `dropout_rate` | Task-specific dropout override. |
| `loss_weight` | Useful for multi-task training loops (weight of this task's loss). |

---

## 5. Deep Dive: Advanced Features

### 5.1. Pooling Strategies
Foundation models output sequences. For tasks requiring a single prediction per sequence (like sentiment analysis), you must "pool" this sequence.

*   **`'cls'`**: Uses the first token's vector. Standard for BERT-family models trained with a special `[CLS]` token.
*   **`'mean'`**: Averages all token vectors. Often better for sentence similarity as it captures diffuse information. respects `attention_mask` if provided.
*   **`'max'`**: Takes the maximum value across the sequence. Good for detecting sparse features (e.g., is a specific keyword present?).
*   **`'attention'`**: Uses a lightweight, learnable attention mechanism to weight tokens before summing. Best performance but slightly more parameters.

### 5.2. Task-Specific Attention
Sometimes the foundation model's universal representations aren't quite perfectly aligned with a niche task.

Enabling `use_task_attention=True` inserts a full Transformer self-attention layer *inside the head*. This allows the head to re-contextualize the generic hidden states specifically for its task before making a prediction, often improving performance on complex tasks like Question Answering.

### 5.3. Multi-Task Heads
The `MultiTaskNLPHead` manages multiple sub-heads. It's a single layer that can route inputs to all its sub-heads simultaneously or just one.

```python
from dl_techniques.layers.nlp_heads import create_multi_task_nlp_head

# ... (define task_configs and input_dim)
multi_head = create_multi_task_nlp_head(task_configs, input_dim)

# Can be called in two ways:
outputs = multi_head(inputs) # Runs ALL tasks, returns dict of dicts: {'ner': {...}, 'sentiment': {...}}
outputs = multi_head(inputs, task_name='ner') # Runs ONLY 'ner' task, returns dict: {...}
```

It also supports `use_task_specific_projections=True`, which adds a dedicated Dense layer for each task *before* the main head logic, allowing standard-sized foundation models to adapt to tasks requiring different feature dimensions.