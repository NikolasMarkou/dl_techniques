# DistilBERT: A Distilled Version of BERT

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of the **DistilBERT** *architecture* in **Keras 3**, based on *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"* by Sanh et al.

> **No trained weights ship with this package and none can be downloaded.** `pretrained=True` raises `NotImplementedError`. Loading a local file you saved yourself *does* work: `pretrained="<path>.keras"`, or `keras.models.load_model(path)`, which is simpler. Everything below describes an architecture you must train yourself.

The published DistilBERT checkpoint retains roughly **97% of BERT's GLUE score** while being **40% smaller and 60% faster**. Those are the paper's numbers for the paper's checkpoint; nothing in this repo reproduces or measures them.

---

## 1. Overview: Smaller, Faster, Cheaper

**DistilBERT** is a Transformer encoder trained by **knowledge distillation**: a compact student is trained to reproduce the behaviour of a larger teacher. It is a general-purpose language representation model like BERT, with a smaller architectural footprint.

This implementation provides the encoder as a **foundation model**. It takes token IDs and returns a sequence of contextual vectors, decoupled from task-specific heads.

Four things worth knowing about this implementation:

1.  **Pure encoder.** The `DistilBERT` class carries no head and no pooler. Heads are attached by `create_distilbert_with_head`.
2.  **Weight loading works, downloading does not.** `keras.models.load_model(path)` restores a file you saved; `load_pretrained_weights(path)` transplants weights into a model you configured yourself. Only the `pretrained=True` download route raises (§8).
3.  **Keras 3 native.** A `.keras` round trip with no `custom_objects` reproduces the forward output exactly (measured max abs diff `0.0`). Only the TensorFlow backend is exercised here.
4.  **Shared embedding stage.** There is no DistilBERT-private embedding class; the model builds the same `BertEmbeddings` layer that `models/language/bert/` and `models/language/fnet/` use (§4.1).

---

## 2. The Problem DistilBERT Solves

BERT-base is 110M+ parameters. That cost shows up as inference latency measured in hundreds of milliseconds, as a memory footprint that needs high-VRAM GPUs to hold gradients and optimizer state, and as an energy bill for both training and serving. On a phone or under a real-time latency budget, the model is simply too big.

The obvious fix, training a small model from scratch on the same data, works badly: small models underfit the raw objective. Distillation avoids that by changing the target rather than the data.

```
Teacher (BERT-base)  ──►  soft target probabilities
                                   │
                                   ▼  distillation loss
Student (DistilBERT) ──►  predicted probabilities
```

The teacher's full output distribution carries more information than a one-hot label: it says which wrong answers were nearly right. Training against that signal lets the student generalize better than it could from raw data alone, which is what makes the smaller capacity affordable. DistilBERT keeps BERT's width and halves its depth, then recovers most of the accuracy from the teacher.

---

## 3. How DistilBERT Works

The architecture is BERT with two pieces removed: **token type (segment) embeddings** and the **pooler**.

```
┌──────────────────────────────────────────────────────────────────┐
│                  DistilBERT Foundation Model                     │
│                                                                  │
│ Input (Token IDs) ───►┌────────────────┐                         │
│                       │ BertEmbeddings │ (Token + Position;      │
│                       │    (shared)    │  token types disabled)  │
│                       └────────┬───────┘                         │
│                       ┌────────▼───────────┐                     │
│                       │ TransformerLayer   │ (Repeated N times)  │
│                       │(Self-Attention,FFN)│                     │
│                       └────────┬───────────┘                     │
│                       ┌────────▼───────────┐                     │
│                       │Output Hidden States│ (B, seq_len, D)     │
│                       └────────┬───────────┘                     │
│                       ┌────────▼───────────┐                     │
│                       │ Task-Specific Head │ (e.g., Classifier)  │
│                       └────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

The data flow, in three steps:

1.  **Input preparation.** A WordPiece tokenizer produces `input_ids` and an `attention_mask` (`1` = keep, `0` = pad). No token type IDs are required. The mask is not optional in practice: nothing infers one for you (§11).
2.  **Embedding + encoding.** Word and position embeddings are summed, normalized and dropped out, then passed through `num_layers` transformer blocks. `base` uses 6 blocks where BERT-base uses 12.
3.  **Projection.** The final hidden states `(B, seq_len, D)` go to a task head. There is no pooler, so for sentence-level tasks take index `0` of `last_hidden_state` yourself.

---

## 4. Architecture Deep Dive

### 4.1 Embedding Stage: the shared `BertEmbeddings`

There is **no DistilBERT-private embedding class**. `_build_architecture` calls

```python
create_embedding_layer(
    'bert_embeddings',
    ...,
    type_vocab_size=None,
    use_token_type_embeddings=False,
    position_embedding_type='sinusoidal' if sinusoidal_pos_embds else 'learned',
    mask_zero=False,
    layer_norm_eps=layer_norm_eps,
    normalization_type=normalization_type,
)
```

so the layer is `dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings`. Three kwargs carry DistilBERT's whole delta from BERT, and each differs from that layer's default. Do not drop them:

-   **`use_token_type_embeddings=False`**: no segment embeddings. The input is one continuous sequence and no `token_type_embeddings` weight is allocated. `type_vocab_size=None` follows from it.
-   **`mask_zero=False`**: DistilBERT threads an explicit `attention_mask` into every `TransformerLayer`, and this flag records that the embedding stage is not meant to supply a second one. Measured caveat, so the claim does not rot: `BertEmbeddings` never *propagates* a Keras mask at **either** setting. `supports_masking` is `False`, it defines no `compute_mask`, and the inner `Embedding`'s mask is dropped at the `word_embeds + position_embeds` sum. The forward output is bit-identical (max abs diff `0.0`) with `mask_zero` `True` vs `False`. The flag's only observable effects today are `get_config()['mask_zero']` and `model.embeddings.word_embeddings.mask_zero`; it is passed explicitly so that omitting it cannot silently flip the model to BERT's `True` if the layer ever gains mask propagation.
-   **`position_embedding_type`**: learned by default, fixed **sinusoidal** with `sinusoidal_pos_embds=True`.

Consequence worth knowing: `normalization_type` is validated by `BertEmbeddings`, which accepts exactly `layer_norm`, `rms_norm`, `band_rms`, `batch_norm`. Any other value raises `ValueError` at construction from the embedding stage, even a value a `TransformerLayer` alone would accept.

### 4.2 `TransformerLayer`

The core block is identical to BERT's, but DistilBERT uses half the depth (6 layers for `base` instead of 12). In the paper the student was initialized by taking one of every two layers from the teacher.

### 4.3 Task-Specific Heads

The output width matches BERT (`hidden_size=768` for `base`), so DistilBERT is compatible with the same task heads.

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First DistilBERT Model

A sentiment classifier on a randomly initialized encoder.

```python
import keras
import numpy as np

from dl_techniques.models.language.distilbert import create_distilbert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

# 1. Define the downstream task
sentiment_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3
)

# 2. Create a DistilBERT model with a sentiment head (random weights).
#    Swap "tiny" for "base" once you have the compute.
model = create_distilbert_with_head(
    distilbert_variant="tiny",
    task_config=sentiment_config,
    pretrained=False  # True raises NotImplementedError -- see section 8
)

# 3. Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Dummy usage (note: no token_type_ids needed)
BATCH_SIZE, SEQ_LEN = 4, 128
dummy_inputs = {
    "input_ids": np.random.randint(0, 30522, size=(BATCH_SIZE, SEQ_LEN)),
    "attention_mask": np.ones((BATCH_SIZE, SEQ_LEN), dtype="int32")
}
# A classification head returns a DICT, not one tensor. Measured:
# {'logits': (4, 3), 'probabilities': (4, 3)}
outputs = model.predict(dummy_inputs, verbose=0)
print({k: v.shape for k, v in outputs.items()})
```

---

## 6. Component Reference

| Symbol | What it is |
|---|---|
| `DistilBERT` | The `keras.Model` encoder. `from_variant(name)` builds a named configuration. |
| `DistilBERT.load_pretrained_weights(path)` | Transplants weights from a `.keras` file into an existing model. |
| `create_distilbert_with_head(...)` | End-to-end factory: builds the inputs (no token types), the encoder and the head, and wires them together. |

```python
from dl_techniques.models.language.distilbert import DistilBERT

model = DistilBERT.from_variant("base")  # randomly initialized
model = DistilBERT.from_variant("tiny")  # edge/mobile
```

---

## 7. Configuration & Model Variants

| Variant | Hidden Size | Layers | Heads | Parameters | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`tiny`** | 256 | 2 | 4 | 9,524,736 | Ultra-lightweight, mobile/IoT |
| **`small`**| 512 | 4 | 8 | 28,499,968 | Fast CPU inference |
| **`base`** | 768 | 6 | 12 | 66,362,880 | General purpose |

Parameter counts are measured with `count_params()` on a built model at the default `vocab_size=30522` and `max_position_embeddings=512`; they move with either of those. There is no `large` variant: the point of the architecture is reduction.

### Constrained configuration values

-   `normalization_type`: one of `layer_norm`, `rms_norm`, `band_rms`, `batch_norm`. Anything else raises `ValueError` at construction (§4.1).
-   `sinusoidal_pos_embds`: `False` (learned) or `True` (fixed sinusoidal). Both verified under `float32` and `mixed_float16` (§10).
-   `pad_token_id`: stored and serialized, never read (§11).

---

## 8. Comprehensive Usage Examples

### Example 1: What `pretrained=` actually does

**`pretrained=True` raises `NotImplementedError`.** No public DistilBERT checkpoint ships with `dl_techniques`, and `_download_weights` says so rather than attempting a fetch that would fail silently.

**`pretrained="<path>.keras"` works.** `from_variant` forwards the path to `load_pretrained_weights`, which loads into **this** configuration:

| Route | Result |
|---|---|
| `from_variant(pretrained="<file>.keras")` on a fresh model | Works. The model is built with a dummy pass first, then every variable is restored; verified by value against the saved model (max abs diff `0.0`, against `5.2` for a fresh control). |
| `model(...)` first, then `load_pretrained_weights("<file>.keras")` | Works. Same result on an already-built model. |
| `load_pretrained_weights(..., skip_mismatch=True)` with a different `vocab_size`/`max_position_embeddings` | Partial by design: the two embedding tables are skipped, the encoder stack is restored. The call logs how many variables actually changed value so a partial load cannot look total. |
| A load that restores **nothing** | Raises `ValueError`. Otherwise `skip_mismatch=True` would make a total non-load indistinguishable from success. |
| `keras.models.load_model("<file>.keras")` | Works, and is simpler: architecture and weights together, no config to match. All weights restored identically (`len(model.weights)` is 28 `tiny` / 52 `small` / 76 `base`). |

Rule of thumb: `keras.models.load_model(path)` to get a saved model back as-is; `load_pretrained_weights(path)` to transplant weights into a model you configured yourself.

```python
import keras
from dl_techniques.models.language.distilbert import DistilBERT

model = DistilBERT.from_variant("tiny")      # randomly initialized
model.save("./distilbert_weights.keras")

# Both of these restore the saved weights exactly:
reloaded = keras.models.load_model("./distilbert_weights.keras")
transplanted = DistilBERT.from_variant("tiny", pretrained="./distilbert_weights.keras")

# This is the one route that does not work:
#   DistilBERT.from_variant("tiny", pretrained=True)  -> NotImplementedError
```

### Example 2: NER (Token Classification)

```python
from dl_techniques.models.language.distilbert import create_distilbert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9
)

ner_model = create_distilbert_with_head(
    distilbert_variant="tiny",
    task_config=ner_config,
)
# Output (measured): {'logits': (batch, seq_len, 9), 'predictions': (batch, seq_len)}
# -- a token-classification head returns different keys from the sentiment
# head in section 5. Always inspect the dict.
```

---

## 9. Advanced Usage Patterns

### Pattern 1: DistilBERT as a Feature Extractor

`name=` on each `keras.Input` is not decoration. An unnamed `Input` inside a dict is auto-named `keras_tensor_N`, which then becomes the data key the model expects.

```python
import keras
from dl_techniques.models.language.distilbert import DistilBERT

encoder = DistilBERT.from_variant("tiny")

inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
}
features = encoder(inputs)["last_hidden_state"]  # (batch, seq_len, hidden_size)

x = keras.layers.LSTM(128)(features)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

---

## 10. Performance Optimization

The published DistilBERT is reported at ~60% faster than BERT-base at 6 layers instead of 12; this repo measures no latency. Two knobs are verified to *work* (not to be faster, no timing was taken):

1.  **XLA compilation**: `model.compile(..., jit_compile=True)`, verified end to end through `create_distilbert_with_head` and `predict`.
2.  **Mixed precision**:
    ```python
    keras.mixed_precision.set_global_policy('mixed_float16')
    model = DistilBERT.from_variant("base")
    ```
    Verified for **both** position-embedding modes: `sinusoidal_pos_embds=False` and `True` each forward-pass to a finite `float16` output. The shared `BertEmbeddings` casts its sin/cos table to the dtype of the tensor it is summed with.

---

## 11. Training and Best Practices

### Fine-Tuning Strategy

DistilBERT is shallower than BERT and more sensitive to aggressive learning rates. Use 2e-5 to 5e-5 for fine-tuning, higher (5e-5 to 1e-4) only when training from scratch. It usually converges in fewer epochs than BERT.

### Input Representation

Use a BERT-compatible WordPiece tokenizer. `token_type_ids` are **not** used: a `"token_type_ids"` key inside the input dict is silently ignored (measured: the forward pass succeeds and returns the usual `last_hidden_state` / `attention_mask`), but passing `token_type_ids=` as a **keyword argument** raises `TypeError`, because `call()` does not declare it.

### Padding is your responsibility (`pad_token_id` is advisory)

`pad_token_id` is stored on the model and written into `get_config()`, and **that is all it does**. Nothing reads it; no attention mask is derived from it anywhere in the package.

-   Call the model without an `attention_mask` and padding tokens are attended to exactly like real tokens. Measured on a batch whose second half is `pad_token_id`, the masked and unmasked forward passes differ by max abs `3.6e-2` at a small config, which is proof both that the mask does work and that none is inferred for you.
-   Always pass `attention_mask` (`1` = keep, `0` = pad), as a dict key or as the `attention_mask=` argument.
-   This matches upstream HuggingFace DistilBERT, which also defaults the mask to all-ones. Deriving one here was considered and rejected: it would silently change the output of every mask-less forward pass written against this model so far.

---

## 12. Serialization & Deployment

Fully serializable via Keras 3. Verified: a save/load round trip with **no** `custom_objects` reproduces the `training=False` forward output exactly (max abs diff `0.0`).

```python
model = create_distilbert_with_head(...)
model.save('distilbert_sentiment.keras')

# Reload without custom object dictionaries
loaded = keras.models.load_model('distilbert_sentiment.keras')
```

---

## 13. Testing & Validation

```python
import numpy as np
from dl_techniques.models.language.distilbert import DistilBERT

def test_forward_pass():
    model = DistilBERT.from_variant("small")
    inputs = {
        "input_ids": np.random.randint(0, 1000, (2, 10)),
        "attention_mask": np.ones((2, 10), dtype="int32")
    }
    out = model(inputs, training=False)
    assert tuple(out["last_hidden_state"].shape) == (2, 10, 512)
```

The package's own suite lives in `tests/test_models/test_distilbert/`.

---

## 14. Troubleshooting & FAQs

-   **`pretrained=True` raises `NotImplementedError`.** Intended. No checkpoint ships here and there is no HuggingFace converter. Train your own, save with `model.save(path)`, reload with `keras.models.load_model(path)` (§8).
-   **A mask-less forward pass gives odd results on padded batches.** Nothing derives a mask from `pad_token_id`. Pass `attention_mask` explicitly (§11).
-   **`TypeError: unexpected keyword argument 'token_type_ids'`.** DistilBERT has no segment embeddings. Drop the argument; a dict key of that name is ignored rather than raising.
-   **`ValueError` at construction from the embedding stage.** `normalization_type` must be one of the four values `BertEmbeddings` accepts (§4.1).
-   **No pooler output.** DistilBERT dropped Next Sentence Prediction and with it the pooler. Take index `0` of `last_hidden_state` for sentence-level tasks.
-   **BERT weights do not transfer directly.** The depth differs (6 vs 12) and the matrices are not 1:1 mappable without the paper's layer-selection procedure.

---

## 15. Technical Details

Differences from BERT:

1.  **Layers** reduced from 12 to 6 (in `base`).
2.  **Token type embeddings** removed.
3.  **Pooler** removed.
4.  **Training objective**: a triple loss, combining masked-language-modeling cross-entropy, a distillation loss against the teacher's soft targets, and a cosine-embedding loss aligning student and teacher hidden states. This repo implements the architecture, not the distillation procedure.

---

## 16. Citation

```bibtex
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```
