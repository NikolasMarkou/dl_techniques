# BERT: Bidirectional Encoder Representations from Transformers

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of **BERT** (Devlin et al., 2019) in **Keras 3**. The `BERT` class is a pure
encoder: it turns token IDs into contextual embeddings and knows nothing about downstream tasks.
Task heads live in `dl_techniques.layers.heads.nlp` and are attached by the
`create_bert_with_head` factory.

> **No pretrained weights are distributed by this library.**
> `BERT.from_variant("base", pretrained=True)` raises `NotImplementedError`, and so does the
> `weights_dataset=` argument's download path (that argument is unreachable). To warm-start, pass
> a local file: `pretrained="path/to/checkpoint.keras"`. To get a random-initialized model, omit
> `pretrained` or pass `pretrained=False`. Every example below uses a route that runs.

---

## 1. Overview: What is BERT and Why It Matters

BERT pre-trains deep **bidirectional** representations from unlabeled text. Earlier models were
either context-free (Word2Vec, GloVe: one static vector per word, so the river "bank" and the
financial "bank" share a vector) or unidirectional (an RNN sees only the left context), or
shallowly bidirectional (ELMo concatenates two independently trained LSTMs).

BERT instead runs a Transformer **encoder**, where self-attention lets every token see every other
token from the first layer onward. Its Masked Language Model objective is what makes that
trainable: mask ~15% of tokens and predict them, which forces the model to use left and right
context at once.

What this implementation adds:

| Property | Detail |
|:---|:---|
| Foundation model design | `BERT` is a pure encoder, reusable across tasks and pre-training runs. |
| Variants | `from_variant("tiny"/"small"/"base"/"large")`, each traced to a public checkpoint config. |
| Serializable | Composite `keras.Model`; full `.keras` round-trip with no `custom_objects`. |
| Configurable | Normalization type and position, FFN type, attention type are all constructor arguments. |

---

## 2. The Problem BERT Solves

Word meaning is fixed by its neighbours, and the useful neighbours are on both sides:

```
"He went to the bank to ..."          -> a left-to-right model guesses "withdraw money"
"He went to the bank to sit by the river."  -> the disambiguating evidence is to the RIGHT
```

A unidirectional model cannot revisit its assumption. BERT's two design choices remove the
constraint:

1. **Transformer encoder.** Self-attention is not sequential. All positions are read at once, so
   information flows in both directions from layer 1.
2. **Masked Language Model.** Predicting a masked token is only solvable using both sides of it,
   so bidirectionality is not merely permitted, it is required by the training objective.

---

## 3. How BERT Works: Core Concepts

The model maps a sequence of token IDs to a sequence of contextual vectors.

```
Input (Token IDs) --> BertEmbeddings   (token + position + segment, LayerNorm, dropout)
                          |
                      TransformerLayer  x N   (self-attention + FFN, residual + norm)
                          |
                      last_hidden_state  (B, seq_len, hidden_size)
                          |
                      Task-specific head
```

**Step 1: input representation.** A tokenizer produces `input_ids` (WordPiece IDs), an
`attention_mask` (1 for real tokens, 0 for padding) and `token_type_ids` (segment IDs, for
sentence-pair tasks). `BertEmbeddings` sums three embedding tables (word, position, segment), then
applies LayerNorm and dropout.

**Step 2: bidirectional encoding.** Each `TransformerLayer` applies multi-head self-attention and
a position-wise FFN, each wrapped in a residual connection and normalization.

**Step 3: projection.** Sentence-level tasks read the `[CLS]` vector; token-level tasks (NER) read
the whole sequence. Either way a small head projects to the label space.

> **Padding is not inferred.** `pad_token_id` is stored and serialized but never read, and no
> attention mask is derived from it. Without an explicit `attention_mask`, padding is fully
> attended to.

---

## 4. Architecture Deep Dive

### 4.1 `BertEmbeddings`

Three embedding tables (words, positions, token types). The output is their element-wise sum,
normalized and dropped out. This gives the encoder identity, position and segment membership in a
single vector before any attention runs.

### 4.2 `TransformerLayer`

The repeated encoder block: multi-head self-attention, then a two-layer MLP applied per position,
each sub-layer wrapped in residual + normalization. `normalization_position` selects Post-Norm
(`"post"`, the original paper and this class's default) or Pre-Norm (`"pre"`, more stable for very
deep stacks).

The default activation is `"gelu_tanh"`, not `"gelu"`. Keras' `"gelu"` is the exact erf form,
while the original BERT release computes the tanh approximation. The gap is
`max|erf - tanh| = 4.732e-04` per call and compounds over 12 to 24 layers, so this is an
inference-changing choice, not a spelling.

### 4.3 Task-specific heads

The encoder is deliberately separate from heads. `create_bert_with_head` is the intended
integration point; §9 Pattern 2 shows the manual wiring, including the one rename that everyone
gets wrong.

---

## 5. Quick Start Guide

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

Build a sentiment model end to end:

```python
import keras
import numpy as np

from dl_techniques.models.language.bert import create_bert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

# 1. Define the downstream task
sentiment_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3,
)

# 2. Create the encoder + head. Swap "tiny" for "base" once you have the compute.
model = create_bert_with_head(
    bert_variant="tiny",
    task_config=sentiment_config,
    pretrained=False,  # or a local ".keras" path; True raises NotImplementedError
)

# 3. Compile. The head's classifier is a Dense with NO activation, so the model
#    emits LOGITS and from_logits=True is not optional.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 4. Dummy data. A real application feeds a BERT tokenizer here.
BATCH_SIZE, SEQ_LEN = 4, 128
dummy_inputs = {
    "input_ids": np.random.randint(0, 30522, size=(BATCH_SIZE, SEQ_LEN)),
    "attention_mask": np.ones((BATCH_SIZE, SEQ_LEN), dtype="int32"),
    "token_type_ids": np.zeros((BATCH_SIZE, SEQ_LEN), dtype="int32"),
}
model.fit(dummy_inputs, np.random.randint(0, 3, size=(BATCH_SIZE,)), epochs=1)

predictions = model.predict(dummy_inputs)
print(predictions.shape)  # (4, 3) -- a bare tensor, not a dict
```

Two contracts worth reading before you debug anything:

- **Output.** `create_bert_with_head` returns a **bare tensor** whenever the head produces one
  informative tensor. Keys that are pure functions of `logits` (`probabilities = softmax(logits)`,
  `predictions = argmax(logits)`) are dropped: no loss can consume them, and a dict output cannot
  be compiled with a string loss or any metric. A head with genuinely independent outputs keeps
  its dict, and `QuestionAnsweringHead` returns `{"start_logits", "end_logits"}`, so there you
  compile with a per-key `loss` dict.
- **Input: all three keys are required.** The Functional wrapper declares a `keras.Input` for each
  of `input_ids`, `attention_mask`, `token_type_ids`, and Keras matches your data dict exactly.
  Omitting `token_type_ids` raises `ValueError: Missing data for input "token_type_ids"`; pass
  `np.zeros_like(input_ids)` for single-segment tasks. The bare encoder is looser:
  `BERT.from_variant("tiny")({"input_ids": ids})` forwards fine and returns
  `['last_hidden_state', 'attention_mask']` (§9 Pattern 1).

---

## 6. Component Reference

| Name | Kind | Purpose |
|:---|:---|:---|
| `BERT` | `keras.Model` | The encoder. Returns `{"last_hidden_state", "attention_mask"}`. |
| `BERT.from_variant` | classmethod | Instantiate a named configuration, optionally from a local checkpoint. |
| `create_bert` | factory | Build an encoder from explicit hyperparameters. |
| `create_bert_with_head` | factory | End-to-end model: encoder + task head, wired and named. |

```python
from dl_techniques.models.language.bert import BERT

encoder = BERT.from_variant("base")                             # random init
encoder = BERT.from_variant("base", pretrained="./bert.keras")  # local checkpoint
custom  = BERT.from_variant("small", vocab_size=50000)          # override any config key
```

Overriding `vocab_size` while loading a checkpoint is supported: the loader runs with
`skip_mismatch=True` and logs the embedding layer it skipped.

---

## 7. Configuration & Model Variants

Every row is traced to a released public checkpoint config. `base`/`large` come from Devlin et al.
2019; `small`/`tiny` come from Turc et al. 2019, *Well-Read Students Learn Better*
(https://arxiv.org/abs/1908.08962), which released a checkpoint for every `(L, H)` pair.

| Variant | Hidden Size | Layers | Heads | Intermediate | Parameters (measured) | Upstream config |
|:---:|:---:|:---:|:---:|:---:|---:|:---|
| **`tiny`** | 256 | 4 | 4 | 1024 | 11,104,768 | `google/bert_uncased_L-4_H-256_A-4` |
| **`small`**| 512 | 6 | 8 | 2048 | 34,805,760 | `google/bert_uncased_L-6_H-512_A-8` |
| **`base`** | 768 | 12 | 12 | 3072 | 108,891,648 | `bert-base-uncased` |
| **`large`** | 1024| 24 | 16 | 4096 | 334,092,288 | `bert-large-uncased` |

The Parameters column is measured against the shipped code, not quoted from a paper. Re-derive it
rather than trusting the table:

```python
from dl_techniques.models.language.bert import BERT

for variant in ["tiny", "small", "base", "large"]:
    model = BERT.from_variant(variant)
    model.build((1, 16))          # subclassed: no weights exist until built
    print(variant, f"{model.count_params():,}")
```

These are **encoder-only** counts at this package's defaults (`vocab_size=30522`,
`max_position_embeddings=512`, `type_vocab_size=2`). This `BERT` owns no pooler and no task head,
so they are not comparable to a published `BertModel` total, which includes a pooler `Dense`
(768x768 + 768 = 590,592 parameters at `base`). Add the head's parameters when sizing a
`create_bert_with_head(...)` model.

---

## 8. Comprehensive Usage Examples

### Example 1: Named Entity Recognition (token-level classification)

```python
import keras

from dl_techniques.models.language.bert import create_bert_with_head
from dl_techniques.layers.heads.nlp.task_types import NLPTaskConfig, NLPTaskType

ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9,  # O, B-PER, I-PER, B-LOC, I-LOC, ...
)

# Pass a local ".keras" path to `pretrained` to warm-start from a checkpoint.
ner_model = create_bert_with_head(
    bert_variant="tiny",
    task_config=ner_config,
    pretrained=False,
)

# `TokenClassificationHead` emits `logits` plus a `predictions` entry that is exactly
# argmax(logits, axis=-1); the factory drops that derived duplicate, so the output is
# the bare logits tensor of shape (batch, seq_len, num_classes). That is what lets you
# compile a token-level objective directly.
ner_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: BERT as a feature extractor

```python
import keras
from dl_techniques.models.language.bert import BERT

bert_encoder = BERT.from_variant("tiny", pretrained=False)

# `name=` is not decoration: an unnamed keras.Input inside a dict gets auto-named
# `keras_tensor_N`, which then becomes the data key the model demands. Omitting it
# emits a UserWarning and breaks fit/predict on a dict.
inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
    "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
}
sequence_output = bert_encoder(inputs)["last_hidden_state"]  # (batch, seq_len, hidden)

downstream = keras.layers.Bidirectional(
    keras.layers.LSTM(128, return_sequences=True)
)(sequence_output)
outputs = keras.layers.Dense(9, activation="softmax")(downstream)

feature_model = keras.Model(inputs, outputs)
```

### Pattern 2: Multi-task fine-tuning

Share one encoder across several heads.

> **The heads do not read `last_hidden_state`.** Every head in `layers/heads/nlp/` reads
> `inputs['hidden_states']`, while `BERT` emits `last_hidden_state`. You must build the rename
> yourself (step 4 below), exactly as `create_bert_with_head` does internally. Feeding the
> encoder's dict straight into a head builds a graph that looks fine and raises **at forward
> time**, because the heads' `build()` and `compute_output_shape()` use `.get(..., default)` and
> never touch the missing key. Construction succeeds, then `predict()` raises
> `KeyError: hidden_states`. Pinned by
> `tests/test_models/test_bert/test_the_head_wiring_needs_the_rename.py`.

```python
import keras
import numpy as np
from dl_techniques.models.language.bert import BERT
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig, NLPTaskType, TextClassificationHead, TokenClassificationHead
)

# 1. One shared encoder ("base" needs a local checkpoint path or pretrained=False)
bert_encoder = BERT.from_variant("tiny", pretrained=False)
bert_encoder.trainable = True

# 2. Inputs -- same shape as Pattern 1, `name=` included
inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
    "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
}

# 3. Shared features: keys are {"last_hidden_state", "attention_mask"}
encoder_outputs = bert_encoder(inputs)

# 4. THE RENAME. Heads read "hidden_states"; the encoder emits "last_hidden_state".
#    Without this line the graph still builds and predict() raises KeyError.
head_inputs = {
    "hidden_states": encoder_outputs["last_hidden_state"],
    "attention_mask": encoder_outputs["attention_mask"],
}

# 5. Two heads. They take a `task_config` and an `input_dim`, not a bare `num_classes`.
sentiment_head = TextClassificationHead(
    task_config=NLPTaskConfig(
        name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=2
    ),
    input_dim=bert_encoder.hidden_size,
    name="sentiment",
)
ner_head = TokenClassificationHead(
    task_config=NLPTaskConfig(
        name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9
    ),
    input_dim=bert_encoder.hidden_size,
    name="ner",
)

# 6. A head returns a DICT -- the classification head {"logits", "probabilities"}, the
#    token head {"logits", "predictions"} -- so select the tensor you train on.
sentiment_output = sentiment_head(head_inputs)
ner_output = ner_head(head_inputs)

multi_task_model = keras.Model(
    inputs=inputs,
    outputs={"sentiment": sentiment_output["logits"], "ner": ner_output["logits"]},
)

ids = np.random.randint(0, 100, (2, 16)).astype("int32")
predictions = multi_task_model.predict(
    {"input_ids": ids, "attention_mask": np.ones_like(ids),
     "token_type_ids": np.zeros_like(ids)},
    verbose=0,
)
print({key: value.shape for key, value in predictions.items()})
# {'sentiment': (2, 2), 'ner': (2, 16, 9)}  -- MEASURED, this snippet was run
```

---

## 10. Performance Optimization

```python
import keras
from dl_techniques.models.language.bert import BERT

keras.mixed_precision.set_global_policy("mixed_float16")  # BEFORE constructing
model = BERT.from_variant("base")
```

Pass `jit_compile=True` to `compile()` for XLA: a compilation warmup that pays back on long runs.

---

## 11. Training and Best Practices

- **Differential learning rates.** Small for the pretrained body (2e-5 to 5e-5), larger for the
  freshly initialized head, so the body does not forget while the head adapts.
- **Sequence length.** Attention is O(n^2). Truncate to 512 tokens; for longer documents use a
  sliding window or a long-context architecture.
- **Tokenizer fidelity.** When fine-tuning a checkpoint, reuse the exact vocabulary, WordPiece
  rules and special tokens (`[CLS]`, `[SEP]`) it was pre-trained with.

---

## 12. Serialization & Deployment

`BERT` and its sub-layers are registered for the `.keras` format via `register_dl_technique(...)`
from `dl_techniques.utils.keras_registration`. `BERT` registers as
`dl_techniques.models.bert.model>BERT` (the `language/` family directory is stripped, because a
family is a filing decision and not a namespace), with a legacy `Custom>BERT` alias also bound.

```python
import keras

model.save("my_bert_ner_model.keras")
loaded = keras.models.load_model("my_bert_ner_model.keras")  # no custom_objects needed
```

---

## 13. Testing & Validation

```python
import numpy as np
from dl_techniques.models.language.bert import BERT

def test_model_creation_all_variants():
    for variant in BERT.MODEL_VARIANTS:      # ['large', 'base', 'small', 'tiny']
        assert BERT.from_variant(variant) is not None

def test_forward_pass_shape():
    model = BERT.from_variant("tiny")
    output = model({
        "input_ids": np.random.randint(0, 30522, (4, 64)),
        "attention_mask": np.ones((4, 64), dtype="int32"),
    })
    assert output["last_hidden_state"].shape == (4, 64, model.hidden_size)
```

The package's own suite is `tests/test_models/test_bert/`.

---

## 14. Troubleshooting & FAQs

- **`NotImplementedError` on `pretrained=True`.** Expected. No weights are distributed. Pass a
  local `.keras` path, or `pretrained=False`.
- **`KeyError: 'hidden_states'` at predict time.** You fed the encoder's dict straight to a head.
  Add the rename from §9 Pattern 2.
- **`ValueError: Missing data for input "token_type_ids"`.** `create_bert_with_head` requires all
  three input keys. Pass `np.zeros_like(input_ids)` for single-segment tasks.
- **OOM during training.** Attention is quadratic in sequence length. Reduce batch size, truncate
  harder, accumulate gradients, drop to a smaller variant, or enable mixed precision.
- **Shape mismatch when loading a checkpoint.** Expected if you changed `vocab_size`. The loader
  uses `skip_mismatch=True` and logs each skipped layer.
- **A `UserWarning` about input names, then a broken `fit`.** Your `keras.Input`s inside a dict
  are missing `name=`. See §9 Pattern 1.

**What is `[CLS]` for?** A special token prepended to every sequence. Because self-attention lets
it read the whole input, its final hidden state acts as an aggregate sequence representation, and
is conventionally the input to a sentence-level classifier.

---

## 15. Technical Details

- **Bidirectional self-attention.** Each token attends to all others, so long-range connections
  exist from layer 1 rather than being carried through a recurrence.
- **Masked Language Model.** ~15% of tokens are replaced with `[MASK]` and predicted. The
  objective is only solvable with both-side context.
- **Pre-Norm vs Post-Norm.** The paper uses Post-Norm (Sublayer, Add, Norm); Pre-Norm (Norm,
  Sublayer, Add) trains more stably at depth. `normalization_position` selects either; this class
  defaults to `"post"` for checkpoint compatibility.

---

## 16. Citation

```bibtex
@inproceedings{devlin2019bert,
  title={{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019},
  publisher={Association for Computational Linguistics}
}
```

The `tiny` and `small` configurations come from Turc et al., *Well-Read Students Learn Better: On
the Importance of Pre-training Compact Models*, arXiv:1908.08962 (2019).
