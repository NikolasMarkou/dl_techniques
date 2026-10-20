# FNet: Mixing Tokens with Fourier Transforms

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of the **FNet** architecture in **Keras 3**, based on ["FNet: Mixing Tokens with Fourier Transforms"](https://arxiv.org/abs/2105.03824) by Lee-Thorp et al. (2021).

This architecture is a **pure encoder**, separating token mixing from task-specific heads, which suits pre-training, fine-tuning and multi-task workflows.

The `FNet` class is a pure encoder by default (`include_top=False`). Pass `include_top=True` with a `task_config` and the same class also owns one task head from `dl_techniques.layers.heads.nlp`, named `task_head`; every head weight lives under `task_head/`, so an encoder trained under one head is reused, unchanged, under another (§7). The knobs and contracts are those of `models/language/bert`.

> **No trained weights ship with this package.** `pretrained=True` raises `NotImplementedError`; pass a local `.keras` path instead. Everything below describes an architecture you train yourself.

---

## 1. Overview: What is FNet and Why It Matters

**FNet** is a Transformer-like architecture that replaces self-attention with an unparameterized **2D Fourier Transform**. It shows that for many NLP tasks the content-aware mixing of self-attention can be replaced by a deterministic, far cheaper operation.

Each encoder block mixes tokens by applying an FFT along the **sequence** dimension, then another along the **hidden** dimension, and keeping the real part.

### Key Innovations

1.  **Parameter-free token mixing.** The Fourier Transform has **zero** learnable weights, cutting the memory footprint of the mixing stage to nothing.
2.  **Complexity.** Token mixing drops from `O(L²)` to `O(L log L)`.
3.  **Accuracy for the price.** The paper reports 92-97% of BERT's GLUE accuracy while training up to 80% faster. Those are the paper's numbers, not measurements of this code.
4.  **Drop-in block.** `FNetEncoderBlock` substitutes directly for a standard Transformer encoder block.

The trade is concrete. Classifying a 4096-token document with self-attention costs roughly `4096²` ≈ 16.7M pairwise comparisons per head per layer; the FFT path costs on the order of `4096 · log 4096` ≈ 49k operations. That is what makes long-sequence domains (genomics, legal documents, whole papers) and low-latency edge deployment tractable.

---

## 2. The Problem FNet Solves

Self-attention is `O(L² · H)`: doubling the sequence length quadruples both compute and memory. Going from 512 to 4096 tokens is a 64x increase. In practice that means slow iterations on long text, attention matrices that consume GBs of VRAM, and models that cap out at 512 or 1024 tokens.

FNet's premise is that **token mixing** is what attention is really for, and mixing does not need learnable weights. Replacing attention with a fixed Fourier Transform still lets every token influence every other token, in quasi-linear time. The burden of content-based, non-linear reasoning shifts entirely onto the feed-forward network.

What you give up is **adaptivity**. Self-attention computes its mixing weights from the input; FNet's mixing is static and identical for every example. The FFN has to make up the difference, which is why FNet benefits from stronger FFN variants (SwiGLU, GEGLU) more than a standard Transformer does.

---

## 3. How FNet Works: Core Concepts

An FNet model is a stack of `FNetEncoderBlock` layers. The block mirrors a Transformer block and replaces only the first sub-layer.

```text
┌─────────────────┐      ┌───────────────────────────────┐
│  Token Mixing   │      │        Channel Mixing         │
│    (2D FFT)     ├──────►      (Feed-Forward Net)       │
│  parameter-free │      │  learns content-based         │
│                 │      │  features, position-wise      │
└─────────────────┘      └───────────────────────────────┘
```

### The Fourier Transform for token mixing

For an input `X` of shape `(seq_len, hidden_dim)`:

1.  **Sequence FFT**: `FFT(X, axis=0)`. Every output element becomes a linear combination of all input tokens.
2.  **Hidden FFT**: `FFT(X, axis=1)`. Mixes within each token's feature vector.
3.  **Real part**: the imaginary component is discarded.

$$Y = \Re(\mathcal{F}_{seq}(\mathcal{F}_{h}(X)))$$

The second FFT is not essential to the idea; the authors found it gave a small consistent gain by encouraging richer within-token interaction.

### The complete data flow

```text
STEP 1: Embedding
─────────────────
Input IDs (B, L)
    ├─► Sum(Token + Position + Type Embeddings)
    └─► LayerNorm -> Dropout -> (B, L, H)

STEP 2: FNet Encoder Stack (Repeated N times)
─────────────────────────────────────────────
Input (B, L, H)
    ├─► Fourier Transform (Mixing)
    ├─► Residual + Norm
    ├─► Feed-Forward Network
    └─► Residual + Norm

STEP 3: Task-Specific Head
──────────────────────────
Final Hidden State (B, L, H)
    ├─► Extract [CLS] or Pool
    └─► Dense Layer -> Logits
```

---

## 4. Architecture Deep Dive

### 4.1 Embedding Layer (`BertEmbeddings`)

FNet uses the standard BERT-style embedding stage, so existing tokenizers work unchanged: a learnable token table, learnable position embeddings, and token type (segment) embeddings.

Position embeddings matter more here than in BERT. The FFT mixes positions symmetrically and supplies no ordering information of its own, so the position embeddings are the model's only source of word order.

### 4.2 FNet Encoder Block (`FNetEncoderBlock`)

The core contribution, at `dl_techniques.layers.fnet_encoder_block.FNetEncoderBlock`:

```text
Input: (B, L, H)
  ▼
┌──────────────────────────┐
│  FNetFourierTransform    │  <- mixing sub-layer, no parameters
└──────────────────────────┘
  ▼  Add & Norm  (residual)
┌──────────────────────────┐
│ Feed-Forward Network     │  <- knowledge sub-layer
│ (MLP / SwiGLU / GEGLU)   │
└──────────────────────────┘
  ▼  Add & Norm  (residual)
Output: (B, L, H)
```

### 4.3 Sequence length must be static

The Fourier mixer needs a known sequence length at build time. A `keras.Input(shape=(None,))` fails with

```
ValueError: Sequence length and hidden dimension must be known at build time.
```

Always give an explicit length: call the model on a concrete batch, call `model.build((None, L))`, or give your `keras.Input` a concrete `shape=(L,)`. This is the single most common way to get an FNet model to refuse to build.

The length is fixed by the first build, with or without a task head: every block's DFT matrices are sized by it, so a model built at `L` raises `InvalidArgumentError` on a batch of any other length. Build one model per length bucket. A `pretrained=<path>` load builds an unbuilt model at the length its checkpoint was built at (§7).

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.18 numpy
```

### Your First FNet Model

A sentiment classifier on the `tiny` configuration.

```python
import keras
import numpy as np

from dl_techniques.models.language.fnet.model import create_fnet, FNet
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

# 1. Define the task configuration
task_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=2
)

# 2. Instantiate via the factory. Use 'base' for real work.
model = create_fnet(
    "tiny",
    include_top=True,
    task_config=task_config,
    pretrained=False,     # True raises NotImplementedError -- section 7
)

# 3. The model returns bare logits (probabilities are derived from them and
#    dropped), so it compiles with a plain loss.
model.compile(
    optimizer='adamw',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Mock training step. The first batch fixes the sequence length (section 4.3).
dummy_inputs = {
    "input_ids": np.random.randint(0, FNet.DEFAULT_VOCAB_SIZE, (4, 128)),
    "attention_mask": np.ones((4, 128), dtype="int32"),
}
dummy_labels = np.array([0, 1, 0, 1])

loss, acc = model.train_on_batch(dummy_inputs, dummy_labels)
logits = model.predict(dummy_inputs, verbose=0)
logits.shape  # (4, 2) -- measured
```

Two contracts worth reading before you debug anything:

- **Output.** With `include_top=True` the model returns a **bare tensor** whenever the head produces one informative tensor: `probabilities` and `predictions` are pure functions of `logits` and are dropped, so the model compiles with a plain loss. `QuestionAnsweringHead` keeps its dict, `{"start_logits", "end_logits"}`. The rule is `layers/heads/nlp/model_head.py::finalize_head_outputs`.
- **Construction.** `include_top=True` without a `task_config` raises `ValueError`, and so does a `task_config` or `head_config_overrides` passed with `include_top=False`. `MULTIPLE_CHOICE` is refused. Only `input_ids` is required at call time; the head's pooling sees the attention mask, cast to the hidden-state dtype (all ones when omitted).

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`FNet`** | `...models.language.fnet.model.FNet` | The foundation encoder. Outputs `{"last_hidden_state", "attention_mask"}`; with `include_top=True`, the task head's output. |
| **`FNetEncoderBlock`** | `dl_techniques.layers.fnet_encoder_block` | The encoder block, usable outside the standard FNet topology. |
| **`create_fnet`** | `...models.language.fnet.model` | Module-level wrapper around `FNet.from_variant`, with or without a head. |
| **`build_model_task_head`** | `...layers.heads.nlp` | Builds the in-model `task_head` (via `create_nlp_head`). |

```python
from dl_techniques.models.language.fnet.model import FNet
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock

encoder = FNet.from_variant("base", pretrained=False, vocab_size=30522)

block = FNetEncoderBlock(
    intermediate_dim=3072,
    dropout_rate=0.1,
    normalization_type='rms_norm',
    ffn_type='swiglu',
)
```

`FNet` takes a tensor of `input_ids`, or a dict whose only required key is `input_ids`; `attention_mask`, `token_type_ids` and `position_ids` are optional.

---

## 7. Configuration & Model Variants

| Variant | Hidden Size | Layers | FFN Size | Parameters | Use Case |
| :--- | :---: |:---: |:---: |:---: |:--- |
| **`large`** | 1024 | 24 | 4096 | 283,674,624 | Maximum capacity |
| **`base`** | 768 | 12 | 3072 | 94,705,152 | General purpose |
| **`small`** | 512 | 6 | 2048 | 31,650,816 | Edge / speed |
| **`tiny`** | 256 | 4 | 512 | 9,527,808 | Mobile / embedded |

Parameter counts are measured with `count_params()` on a built model at the default `vocab_size=30522` and `max_position_embeddings=512`; they move with either of those. Start with `base`; move to `small` if inference latency is critical.

**Pretrained weights.** `pretrained=True` raises `NotImplementedError`: no FNet checkpoint is distributed with `dl_techniques`. Pass `pretrained="<path>.keras"` to load a file you saved yourself, or use `keras.models.load_model(path)`.

A `pretrained=<path>` load transfers **the encoder only**, layer by layer and by name, and never the checkpoint's `task_head`. So the file may hold a bare encoder or an `FNet` trained under any head, and the new model's head (if any) keeps its fresh initialization; a whole model, head included, comes back with `keras.models.load_model`. Call the source model once before saving it, and the new model is built at the sequence length the checkpoint was built at: each block carries DFT matrices sized by that length, so a block can only be copied at the same length.

Every encoder layer must load, and every checkpoint layer must find its place, or the call raises: a checkpoint of another hidden size, depth (deeper or shallower), width or sequence length is refused, never half-loaded or truncated. On a vocabulary change (the checkpoint's own `vocab_size` differs from the model's) only the word-embedding table keeps its fresh initialization; the position and token-type tables and the embedding LayerNorm still load, or the call raises.

```python
import numpy as np
from dl_techniques.models.language.fnet.model import create_fnet
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

batch = {"input_ids": np.random.randint(1, 1000, (2, 64))}
clf = create_fnet("tiny", include_top=True, task_config=NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3))
clf(batch)                      # builds at length 64
clf.save("fnet_clf.keras")      # ... after training under the classifier

ner = create_fnet("tiny", include_top=True, pretrained="fnet_clf.keras",
                  task_config=NLPTaskConfig(
                      name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
                      num_classes=9))
ner(batch).shape                # (2, 64, 9): same encoder weights, fresh NER head
```

---

## 8. Comprehensive Usage Examples

### Example 1: Pure Feature Extraction

Produce `[CLS]` embeddings for clustering or a downstream non-neural model. Note the concrete `shape=(128,)`: a `None` length will not build (§4.3).

```python
import keras
from dl_techniques.models.language.fnet.model import FNet

fnet_encoder = FNet.from_variant("tiny", pretrained=False)

inputs = {
    "input_ids": keras.Input(shape=(128,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(128,), dtype="int32", name="attention_mask"),
}

outputs = fnet_encoder(inputs)
cls_embedding = outputs["last_hidden_state"][:, 0, :]   # (batch, hidden_size)

extractor = keras.Model(inputs, cls_embedding)
```

### Example 2: Named Entity Recognition

```python
from dl_techniques.models.language.fnet.model import create_fnet
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9
)

ner_model = create_fnet(
    "tiny",
    include_top=True,
    task_config=ner_config,
    pretrained=False,
)
ner_model.build((None, 256))   # fixes the sequence length (section 4.3)
# Measured output: bare logits (batch, 256, 9). The token head's `predictions`
# is argmax(logits) and is dropped, as `probabilities` is for the sentiment head.
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Modernizing the Architecture

You are not stuck with the 2021 block. Constructor kwargs pass down to `FNetEncoderBlock`, so RMSNorm and SwiGLU are available.

```python
from dl_techniques.models.language.fnet.model import FNet

modern_fnet = FNet.from_variant(
    "tiny",
    normalization_type="rms_norm",
    normalization_position="pre",  # x = input + branch(Norm(input)), plus a stack-final norm
    ffn_type="swiglu",
)
```

### Pattern 2: Multi-Task Learning

For one head, `include_top=True` is enough (§5). For several heads at once, use the headless encoder: one instance can feed several heads. The heads read a `"hidden_states"` key; the encoder emits `"last_hidden_state"`, so the rename is required, not cosmetic.

```python
import keras
from dl_techniques.models.language.fnet.model import FNet
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head

shared_encoder = FNet.from_variant("tiny")

inputs = {
    "input_ids": keras.Input(shape=(128,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(128,), dtype="int32", name="attention_mask"),
}
encoder_out = shared_encoder(inputs)

# THE RENAME. Without it the head raises KeyError: 'hidden_states'.
head_inputs = {
    "hidden_states": encoder_out["last_hidden_state"],
    "attention_mask": encoder_out["attention_mask"],
}

hidden = shared_encoder.hidden_size
sentiment_head = create_nlp_head(
    NLPTaskConfig(name="sent", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=2),
    input_dim=hidden,
)
ner_head = create_nlp_head(
    NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9),
    input_dim=hidden,
)

# Each head returns a dict; select the tensor you train on.
model = keras.Model(
    inputs=inputs,
    outputs={
        "sent": sentiment_head(head_inputs)["logits"],
        "ner": ner_head(head_inputs)["logits"],
    },
)
model.compile(
    optimizer="adamw",
    loss={
        "sent": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "ner": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    },
)
```

---

## 10. Performance Optimization

### Mixed Precision

FNet is compute-bound by the FFN layers, so reduced precision helps there.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = FNet.from_variant("base")   # picks up the global policy
```

### XLA Compilation

With static sequence lengths, XLA suits FNet well: `model.compile(..., jit_compile=True)`, or wrap a custom step in `@tf.function(jit_compile=True)`.

---

## 11. Training and Best Practices

1.  **Fixed sequence lengths**: required at build time (§4.3), and powers of two suit the FFT best. Pad or truncate to fixed buckets (128, 256, 512) rather than letting shapes vary, which triggers recompilation.
2.  **Optimizer**: `AdamW`.
3.  **Learning rate**: standard Transformer schedules: linear decay with ~10% warmup, peak `1e-4` to `5e-5` for fine-tuning.
4.  **Batch size**: FNet uses less memory than BERT at equal length, so larger batches usually fit.

---

## 12. Serialization & Deployment

Fully compliant with Keras 3 serialization; no `custom_objects` needed on load.

```python
model.save('my_fnet_model.keras')
loaded_model = keras.models.load_model('my_fnet_model.keras')
```

---

## 13. Testing & Validation

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_fnet/ -q
```

The suite covers shape inference, serialization round trips and numerical stability.

---

## 14. Troubleshooting & FAQs

-   **`ValueError: Sequence length and hidden dimension must be known at build time.`** The Fourier mixer needs a static length. Call the model on a concrete batch, call `model.build((None, L))`, or give `keras.Input` a concrete `shape=(L,)` (§4.3).
-   **`InvalidArgumentError` from `FNetFourierTransform.call()`.** The batch's length differs from the one the model was built at (§4.3).
-   **`ValueError: include_top=True needs a task_config`**, or **`... only take effect with include_top=True`.** The head knobs travel together: `include_top=True` with a `task_config`, or none of them (§5).
-   **`KeyError: 'hidden_states'`.** Heads read `hidden_states`; the encoder emits `last_hidden_state`. Rename it (§9, Pattern 2).
-   **`NotImplementedError` from `pretrained=True`.** No checkpoint ships here. Pass a local `.keras` path instead (§7).
-   **Accuracy 2-3% below an attention model.** Expected: FNet trades adaptivity for speed. It often wins in wall-clock time while needing more steps to settle.
-   **Can I use FNet as a decoder?** No. The Fourier Transform is global and bidirectional and cannot be causally masked. FNet is an encoder only.

---

## 15. Technical Details

### Complexity

Let $L$ be sequence length and $H$ hidden size.

| Component | Complexity | Scaling in $L$ |
| :--- | :--- | :--- |
| **Self-Attention** | $O(L^2 \cdot H)$ | Quadratic |
| **FNet mixing (FFT)** | $O(L \log L \cdot H)$ | Quasi-linear |
| **Feed-Forward** | $O(L \cdot H^2)$ | Linear |

### The role of the FFN

FNet's mixing layer is linear and static, so the feed-forward network carries the entire burden of non-linear, content-specific learning. That is why FNet gains more from GLU-family FFNs than a standard Transformer does.

---

## 16. Citation

```bibtex
@article{lee2021fnet,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```
