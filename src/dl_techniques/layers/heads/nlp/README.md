# NLP Task Heads

## 1. Overview & Philosophy

The `dl_techniques.layers.heads.nlp` module provides a unified, model-agnostic system for attaching task-specific "heads" to any NLP foundation model.

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
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig, NLPTaskType

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

Different tasks require different specialized architectures. The factory selects the class from
`NLP_HEAD_DISPATCH` (`nlp/task_types.py`), the one table from task type to head: 24 of the 37
`NLPTaskType` members have a head and the other 13 raise `ValueError`. `NLPTaskType.get_output_types(task)`
names exactly the keys that head returns, and raises for a task type with no head.

Every head that pools honours the `'attention_mask'` in its input dict, and so does the task
attention of the token and QA heads. A mask for an input that is already pooled is refused (there
is nothing to mask).

### 3.1. Sequence Classification
*   **Task Types**: `TEXT_CLASSIFICATION`, `SENTIMENT_ANALYSIS`, `EMOTION_DETECTION`, `INTENT_CLASSIFICATION`, `TOPIC_CLASSIFICATION`, `SPAM_DETECTION`, `NATURAL_LANGUAGE_INFERENCE`, and the regression types `TEXT_REGRESSION`, `READABILITY_SCORING`, `QUALITY_SCORING` (set `num_classes=1` and read `logits`; `probabilities` of one class is constantly 1).
*   **Supported Inputs**: `[B, Seq, Dim]` (will be pooled) or `[B, Dim]` (already pooled). With `use_pooling=False` only the pooled form is accepted.
*   **Outputs**: `logits` `[B, NumClasses]` (pre-softmax: compile a loss on them with `from_logits=True`), `probabilities` `[B, NumClasses]` (`softmax(logits)`).
*   **Key Config**: Requires `num_classes` in `NLPTaskConfig` (every task type above; the config refuses one without it).
*   **No task attention**: the features are pooled to rank 2 before any stage runs, so `use_task_attention=True` raises `ValueError`.

```python
config = NLPTaskConfig(name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3)
head = create_nlp_head(config, input_dim=768, pooling_type='cls') # Standard BERT-style pooling
```

### 3.2. Token Classification
*   **Task Types**: `NAMED_ENTITY_RECOGNITION`, `PART_OF_SPEECH_TAGGING`, `TOKEN_CLASSIFICATION`, `SEQUENCE_LABELING`.
*   **Supported Inputs**: Must be full sequence `[B, Seq, Dim]`.
*   **Outputs**: `logits` `[B, Seq, NumClasses]`, `predictions` `[B, Seq]` (int32 argmax indices; omitted when `use_crf=True`, which builds no CRF layer and leaves decoding to the caller).
*   **Key Config**: Requires `num_classes`. The head never pools: `use_pooling=True` raises.

```python
config = NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9)
# Typically uses full sequence, so no pooling needed.
head = create_nlp_head(config, input_dim=768)
```

### 3.3. Question Answering (Extractive)
*   **Task Types**: `QUESTION_ANSWERING`, `SPAN_EXTRACTION`.
*   **Supported Inputs**: Full sequence `[B, Seq, Dim]`; a masked position gets the dtype's mask sentinel so an argmax cannot land on padding.
*   **Outputs**: `start_logits` `[B, Seq]`, `end_logits` `[B, Seq]`.
*   **No FFN stage**: `use_ffn=True` raises. The head never pools.

```python
config = NLPTaskConfig(name="squad", task_type=NLPTaskType.QUESTION_ANSWERING)
head = create_nlp_head(config, input_dim=768)
```

### 3.4. Text Similarity & Embeddings
*   **Task Types**: `TEXT_SIMILARITY`, `PARAPHRASE_DETECTION`, `DUPLICATE_DETECTION`.
*   **Supported Inputs**:
    *   *Single Input* `[B, Seq, Dim]` or `[B, Dim]`, or a dict with `attention_mask`: Returns embeddings.
    *   *Pair* (a tuple or a list, which is what a functional model passes) of two such inputs: Returns the similarity score. Each side is pooled under its own mask.
*   **Outputs**: `embeddings` `[B, hidden_size]`; for a pair `similarity_score` `[B]`, plus `embeddings_1` and `embeddings_2` with `output_embeddings=True`.
*   **Key Config**: `similarity_function` ('cosine', 'dot', 'learned'; anything else raises). No task attention.

```python
config = NLPTaskConfig(name="sts", task_type=NLPTaskType.TEXT_SIMILARITY)
head = create_nlp_head(config, input_dim=768, similarity_function='cosine')

# Usage 1: Get Embeddings
embeds = head({'hidden_states': seq1})['embeddings']

# Usage 2: Get Similarity, each side under its own padding mask
score = head(({'hidden_states': seq1, 'attention_mask': mask1},
              {'hidden_states': seq2, 'attention_mask': mask2}))['similarity_score']
```

### 3.5. Text Generation (Language Modeling)
*   **Task Types**: `TEXT_GENERATION`, `MASKED_LANGUAGE_MODELING`, `TEXT_SUMMARIZATION`, `TEXT_COMPLETION`.
*   **Supported Inputs**: Full sequence `[B, Seq, Dim]`.
*   **Outputs**: `logits` `[B, Seq, VocabSize]`.
*   **Key Config**: `vocabulary_size` in `NLPTaskConfig` (32000 when unset). No task attention; the head never pools.

```python
config = NLPTaskConfig(name="mlm", task_type=NLPTaskType.MASKED_LANGUAGE_MODELING, vocabulary_size=30522)
head = create_nlp_head(config, input_dim=768)
```

### 3.6. Multiple Choice
*   **Task Types**: `MULTIPLE_CHOICE`.
*   **Supported Inputs**: 4D Tensor `[B, NumChoices, Seq, Dim]`, with an optional `attention_mask` `[B, NumChoices, Seq]`, or pooled choices `[B, NumChoices, Dim]` (no mask).
*   **Outputs**: `logits` `[B, NumChoices]`, `probabilities` `[B, NumChoices]`.
*   **No task attention and no FFN stage**: both flags raise.

```python
config = NLPTaskConfig(name="swag", task_type=NLPTaskType.MULTIPLE_CHOICE)
# The choice axis is folded into the batch, so each choice is pooled under its own mask row
head = create_nlp_head(config, input_dim=768, pooling_type='cls')
out = head({'hidden_states': choices, 'attention_mask': choice_mask})
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
| `use_pooling` | `bool`| `True` (`False` for token, QA, generation) | Whether to pool the sequence into a single vector. The three heads that never pool refuse `True`. |
| `pooling_type` | `Literal[str]` | `'cls'` | Strategy if pooling is enabled. Options:<br>`'cls'`, `'mean'`, `'max'`, `'last'` (the last unmasked position; the choice for a causal backbone), `'attention'`. Anything else raises at construction, for every head. `'attention'` is the head's own inline `Dense(1, tanh)` scorer, not `SequencePooling('attention')` (a different mechanism and weight set). |
| `use_intermediate`| `bool`| `True` | Adds a Dense block (to `hidden_size`) before the final task projection. |
| `intermediate_size` | `int` | `input_dim` | Size of the intermediate dense block when `task_config.hidden_size` is unset. |
| `use_task_attention`| `bool`| `False` | Adds a self-attention layer specific to this task. Only the token and QA heads run one; the other four raise. |
| `attention_type` | `Optional[Literal[str]]` | `None`| Type of task attention; `None` means `'multi_head'` with `use_task_attention` on. With it off, any value raises `ValueError` and `None` is stored (D-211). Options include:<br>`'multi_head'`, `'cbam'`, `'differential'`, `'group_query'`, `'perceiver'`, `'window'`, etc. |
| `use_ffn` | `bool`| `False` | Adds a Transformer-style FFN block to the head (emits `hidden_size`). The QA and multiple-choice heads run none and raise. |
| `ffn_type` | `Optional[Literal[str]]` | `None` | Type of FFN; `None` means `'mlp'` with `use_ffn` on. With it off, any value raises `ValueError` and `None` is stored. Options include:<br>`'mlp'`, `'swiglu'`, `'geglu'`, `'gated_mlp'`, `'differential'`, `'residual'`, `'swin_mlp'`, etc. |
| `ffn_expansion_factor` | `Optional[int]` | `None` | FFN inner width is `hidden_size * ffn_expansion_factor`; `None` means 4 with `use_ffn` on. With it off, any value raises and `None` is stored. |

**Widths.** Task attention and the FFN run at the width the features have when they arrive:
`hidden_size` after the intermediate block, `input_dim` without it. The output layer is built at the
width `call` feeds it (`hidden_size` after the intermediate block or the FFN, else `input_dim`), so
every accepted combination of the flags builds and runs. The dropout rate lives in `NLPTaskConfig`.

### 4.2. Task Configuration (`NLPTaskConfig`)
Used to define the high-level properties of the task itself.

These seven fields are all of them; each is read by some head. `__post_init__` parses a string
`task_type`, validates every value, and applies the one requirement of the head its task type is
routed to. Unknown fields (the deleted `num_labels`, `max_length`, `loss_weight`,
`label_smoothing`, `use_attention_pooling`, `beam_size`, `temperature`) raise `TypeError`.

| Field | Description |
| :--- | :--- |
| `name` | Unique identifier for the task (e.g., "ner_v2"); a non-empty string. |
| `task_type` | `NLPTaskType` member, or its string value (`"sentiment_analysis"`), on every path including a dict passed to `create_nlp_head`. |
| `num_classes` | Required for every task type routed to `TextClassificationHead` or `TokenClassificationHead`; a positive integer wherever set (0, -1, a bool or a float raise `ValueError`). |
| `hidden_size` | Working width of the head; unset means `intermediate_size`, else `input_dim`. |
| `use_crf` | Token head only: emit `logits` alone, for a CRF the caller adds. |
| `vocabulary_size`| Required for generation/MLM tasks. Defaults to 32000 when unset. On a language model that owns the head (`include_top=True`) it must equal the model's `vocab_size`, else `ValueError`; a dict `task_config` that leaves it out gets the model's. Migration: a `.keras` archive saved before 2026-10-11 whose stored generation `task_config` kept 32000 on a model of another vocabulary no longer loads; set the stored value to the model's `vocab_size` (or delete the key) in its `config.json`, or rebuild and re-save. |
| `dropout_rate` | Dropout inside the head, in `[0, 1)`. |

---

## 5. Deep Dive: Advanced Features

### 5.1. Pooling Strategies
Foundation models output sequences. For tasks requiring a single prediction per sequence (like sentiment analysis), you must "pool" this sequence.

*   **`'cls'`**: Uses the first token's vector. Standard for BERT-family models trained with a special `[CLS]` token.
*   **`'mean'`**: Averages the unmasked token vectors. Often better for sentence similarity as it captures diffuse information.
*   **`'last'`**: Uses the last unmasked position. The right choice for a causal backbone, where position 0 has seen only itself.
*   **`'max'`**: Takes the maximum value across the sequence. Good for detecting sparse features (e.g., is a specific keyword present?).
*   **`'attention'`**: Uses a lightweight, learnable attention mechanism to weight tokens before summing. Best performance but slightly more parameters.

### 5.2. Task-Specific Attention
Sometimes the foundation model's universal representations aren't quite perfectly aligned with a niche task.

Enabling `use_task_attention=True` inserts a self-attention layer *inside the head*, after the intermediate block. This allows the head to re-contextualize the generic hidden states specifically for its task before making a prediction. Only the two heads that keep the sequence axis run it, the token-classification and question-answering heads; the other four refuse the flag. A dict input's `'attention_mask'` reaches it, so a real token never attends over padding (since 2026-10-15, D-072; it used to attend over every position). Without a mask the call is unchanged. An `attention_type` whose layer takes no `attention_mask` (`'performer'`; `'linear'` accepts `mask=` and discards it) refuses a mask with `ValueError` rather than attending over padding.

### 5.3. Multi-Task Heads
The `MultiTaskNLPHead` manages multiple sub-heads. It's a single layer that can route inputs to all its sub-heads simultaneously or just one.

```python
from dl_techniques.layers.heads.nlp import create_multi_task_nlp_head

# ... (define task_configs and input_dim)
multi_head = create_multi_task_nlp_head(task_configs, input_dim)

# Can be called in two ways:
outputs = multi_head(inputs) # Runs ALL tasks, returns dict of dicts: {'ner': {...}, 'sentiment': {...}}
outputs = multi_head(inputs, task_name='ner') # Runs ONLY 'ner' task, returns dict: {...}
```

`task_name` is keyword-only (the second positional slot is `training`), and the symbolic call and
`compute_output_shape(..., task_name=...)` follow it. An empty task set is refused, and so is a
list or tuple (the two mean the same) with two configs of one name; any type other than a list,
tuple or dict raises `ValueError`. The wrapper and its heads keep their own copies of the task
configs, and `get_config` holds every field of each.

It also supports `use_task_specific_projections=True`, which adds a dedicated Dense layer for each task *before* the main head logic, allowing standard-sized foundation models to adapt to tasks requiring different feature dimensions.

### 5.4. Serialization
Every head saves every constructor argument in `get_config`, with `task_config` as a plain dict of
all its fields, and is rebuilt by Keras' own `from_config` (the heads accept that dict). Every head
variant round-trips through `.keras` by value
(`tests/test_layers/test_heads/test_every_nlp_head_survives_a_keras_round_trip.py`). Editing an
`NLPTaskConfig` after building a head changes neither the head nor its config: heads copy it.
Archives of NLP heads saved before 2026-10-15 store `loss_weight`, `label_smoothing` and
`use_attention_pooling` in their `task_config`, which this version refuses with `TypeError`; they
do not load and are not supported.