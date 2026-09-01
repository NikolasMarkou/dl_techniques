# GPT-2: Generative Pre-trained Transformer 2

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **GPT-2** decoder-only language model, based on *Language Models
are Unsupervised Multitask Learners* (Radford et al., 2019).

The class is a deliberately **thin wrapper**. The whole transformer stack (token and positional
embeddings, embedding norm and dropout, causal self-attention blocks, final normalization) lives
in the library's shared `TextDecoder` layer, which is itself built out of the library's factories.
`GPT2` adds the language-modeling head on top and nothing else.

> **No pretrained weights are distributed by this library.**
> `GPT2.from_variant("small", pretrained=True)` raises `NotImplementedError`, so a caller who asks
> for trained weights never silently receives an untrained model. To load weights, pass a local
> file path (`pretrained="path/to/checkpoint.keras"`); omit `pretrained` for random init. Train
> your own with `src/train/gpt2/pretrain.py` (§11).

---

## 1. Overview: What GPT-2 is and Why It Matters

GPT-2 is a **decoder-only autoregressive transformer**. It reads a sequence of token IDs and, at
every position, predicts the next token using only the tokens to its left. Trained at scale on
unlabeled text, that single objective produces a model that continues prose, answers questions,
and fine-tunes onto downstream tasks with no architectural change.

What this implementation is:

1. **A thin wrapper.** `GPT2` builds one sub-layer, `TextDecoder`
   (`dl_techniques/layers/transformers/text_decoder.py`), plus a single untied `Dense` projection
   when `tie_word_embeddings=False`. That is the entire model surface; §15 says why.
2. **Reuse all the way down.** `TextDecoder` composes `TransformerLayer`,
   `create_normalization_layer` and the shared causal/padding mask builders in
   `dl_techniques.utils.masking`, so improvements to them land here with no code change.
3. **Serializable.** Registered as `dl_techniques.models.gpt2.gpt2>GPT2`; `get_config()`
   round-trips all twelve constructor arguments, and a `.keras` save/load needs no
   `custom_objects` (§12). The `gpt2_gelu` activation registers as
   `dl_techniques.models.gpt2.gpt2>gpt2_gelu`.
4. **Configurable attention and FFN.** `attention_type` and `ffn_type` pass straight through to
   `TextDecoder`'s factories, so the classic recipe is a default, not a hard-coding.

This package is the repo's reference decoder-only LM. `models/language/wave_field/` mirrors its
public surface so both slot into the same pipeline, and `src/train/wave_field/` reuses
`src/train/gpt2/`'s scaffold through the shared `ClmPretrainConfig`.

---

## 2. The Problem GPT-2 Solves

Labeled NLP data is scarce and task-specific; raw text is effectively unlimited. Next-token
prediction turns raw text into supervision: every position is its own training example, and the
label is the token that came next. That only works if the model cannot see the future, and GPT-2's
answer is a **causal mask**: position `i` may attend to positions `<= i` only.

```
Attention visibility (4 tokens)          x = may attend, . = masked out

        t0    t1    t2    t3
  t0 [  x     .     .     .  ]
  t1 [  x     x     .     .  ]
  t2 [  x     x     x     .  ]
  t3 [  x     x     x     x  ]
```

In this implementation the mask is **constructed and applied inside `TextDecoder`** (via
`dl_techniques.utils.masking`), not in `gpt2.py`. Any padding mask you supply is combined with it,
never substituted for it.

---

## 3. How GPT-2 Works: Core Concepts

```
Input IDs (B, N)   [+ optional attention_mask (B, N)]
  -> TextDecoder
       |- token embedding + learned positional embedding
       |- embedding LayerNorm -> dropout
       |- depth x [ LayerNorm -> causal self-attention -> residual
       |            LayerNorm -> FFN                   -> residual ]   (pre-norm)
       |- final LayerNorm
  -> hidden_states (B, N, D)
  -> LM head:  tied -> hidden @ E^T  |  untied -> Dense(V, no bias)
  -> {"logits": (B, N, V), "last_hidden_state": (B, N, D)}
```

The model accepts either a bare tensor of IDs or a dict `{"input_ids", "attention_mask"}`; a dict
without `input_ids` raises `ValueError`. Training loss is
`MaskedCausalLMLoss(logits[:, :-1], input_ids[:, 1:])`, and **the shift is done by the caller**,
not by the model.

### Differences from BERT

| | BERT (`models/language/bert/`) | GPT-2 (here) |
|---|---|---|
| Attention | bidirectional | causal |
| Norm position | post-norm | pre-norm |
| Objective | masked LM | next-token prediction |
| Segment (token type) embeddings | yes | no |
| Output head | task heads | vocabulary projection, tied by default |

---

## 4. Architecture Deep Dive

### 4.1 `TextDecoder`, everything except the head

`GPT2._build_architecture` constructs exactly one `TextDecoder` named `"decoder"`, configured with
`embedding_type="learned"`, `positional_type="learned"`, `normalization_type="layer_norm"` and
`normalization_position="pre"`. `attention_type`, `ffn_type`, both dropout rates,
`initializer_range` and `layer_norm_eps` are forwarded from the GPT-2 constructor. Weight paths
therefore all live under `decoder/...`.

### 4.2 The LM head

- **Tied (default).** `self.lm_head is None` and `call()` computes
  `ops.matmul(hidden, ops.transpose(self.decoder.word_embeddings.embeddings))`. This is the
  original GPT-2 recipe and saves a `vocab_size x embed_dim` parameter block.
- **Untied.** A `keras.layers.Dense(vocab_size, use_bias=False)` named `lm_head`, initialized with
  `TruncatedNormal(stddev=initializer_range)`. Untying is the modern preference at
  multi-billion-parameter scale; the kernel is `(D, V)` while the embedding table is `(V, D)`, so
  they are genuinely separate variables, not a transposed view.

This `Dense` is the **only** layer `gpt2.py` builds directly, deliberately; see §15.

### 4.3 Validation

`_validate_config` (a `@staticmethod`) raises `ValueError` for a non-positive `vocab_size`,
`embed_dim`, `depth` or `num_heads`, for `embed_dim % num_heads != 0`, and for a dropout rate
outside `[0, 1]`, all in `__init__` before any layer is constructed.

---

## 5. Quick Start Guide

The model ships with the library; no dependency beyond the repo's own environment (`.venv`,
Keras 3 / TensorFlow 2.18).

```python
import numpy as np
from dl_techniques.models.language.gpt2 import create_gpt2

# A small, fast-to-build configuration. Swap "tiny" for "small"/"medium"/
# "large"/"xl" once you have the compute for them.
model = create_gpt2("tiny", vocab_size=512)

input_ids = np.random.randint(0, 512, size=(2, 64)).astype("int32")
outputs = model(input_ids)

print(outputs["logits"].shape)             # (2, 64, 512)
print(outputs["last_hidden_state"].shape)  # (2, 64, 256)
```

---

## 6. Component Reference

`dl_techniques.models.language.gpt2` exports exactly `GPT2` and `create_gpt2` via `__all__`. That
surface is pinned by a test (§13).

**`GPT2`** is the `keras.Model` subclass. It returns a **dict** with `logits` and
`last_hidden_state`, never a bare tensor, so a dict-keyed `compile(loss={"logits": ...})` works
unchanged. Key methods: `from_variant`, `call`, `compute_output_shape`, `get_config`, and the
static `_download_weights` (always raises). **`create_gpt2(...)`** wraps `GPT2.from_variant`,
mirroring `create_bert` / `create_resnet`; `vocab_size` is injected only when not `None`.

```python
from dl_techniques.models.language.gpt2 import GPT2, create_gpt2

model = GPT2.from_variant("small")                       # named variant
model = GPT2.from_variant("tiny", dropout_rate=0.1)      # variant + override
model = GPT2(vocab_size=50257, embed_dim=512, depth=6, num_heads=8)
model = create_gpt2("tiny", vocab_size=200)              # factory + vocab override
```

---

## 7. Configuration & Model Variants

`GPT2.MODEL_VARIANTS` (each entry also carries a `"description"`, which `from_variant` pops before
construction):

| Variant | `embed_dim` | `depth` | `num_heads` | `max_seq_len` |
|:---:|:---:|:---:|:---:|:---:|
| `tiny` | 256 | 4 | 4 | 512 |
| `small` | 768 | 12 | 12 | 1024 |
| `medium` | 1024 | 24 | 16 | 1024 |
| `large` | 1280 | 36 | 20 | 1024 |
| `xl` | 1600 | 48 | 25 | 1024 |

The variants set architecture only. `vocab_size` is **not** part of a variant and always falls
back to `DEFAULT_VOCAB_SIZE` unless you override it. Re-derive this table with
`for name, cfg in GPT2.MODEL_VARIANTS.items(): print(name, cfg)` rather than trusting it.

Constructor arguments not covered by a variant:

| Argument | Default | Notes |
|---|---|---|
| `vocab_size` | `100277` | Tiktoken `cl100k_base`. Must match your tokenizer. |
| `dropout_rate` | `0.0` | Embedding and residual paths. |
| `attention_dropout_rate` | `0.0` | Attention weights. |
| `initializer_range` | `0.02` | `TruncatedNormal` stddev. |
| `layer_norm_eps` | `1e-5` | |
| `attention_type` | `"multi_head"` | Forwarded to `TextDecoder`'s attention factory. |
| `ffn_type` | `"mlp"` | Forwarded to `TextDecoder`'s FFN factory. |
| `tie_word_embeddings` | `True` | See §4.2. |

---

## 8. Comprehensive Usage Examples

### Example 1: dict input with a padding mask

```python
import numpy as np
from dl_techniques.models.language.gpt2 import GPT2

model = GPT2.from_variant("tiny", vocab_size=512, dropout_rate=0.1)

# 'input_ids' is required, 'attention_mask' is optional (1 = attend, 0 = padding).
batch = {
    "input_ids": np.random.randint(0, 512, size=(2, 16)).astype("int32"),
    "attention_mask": np.concatenate(
        [np.ones((2, 12), "int32"), np.zeros((2, 4), "int32")], axis=1
    ),
}
print(model(batch, training=False)["logits"].shape)  # (2, 16, 512)

# A missing 'input_ids' key is a hard error, not a silent default.
try:
    model({"attention_mask": batch["attention_mask"]})
except ValueError as e:
    print(f"ValueError: {e}")
```

### Example 2: tied vs untied LM head

```python
import numpy as np
from dl_techniques.models.language.gpt2 import GPT2

tied = GPT2(vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
untied = GPT2(
    vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32,
    tie_word_embeddings=False,
)

print(tied.lm_head)          # None -- logits = hidden @ E.T
print(untied.lm_head.name)   # 'lm_head' -- an independent Dense(vocab, no bias)

ids = np.random.randint(0, 512, size=(1, 8)).astype("int32")
print(tied(ids)["logits"].shape, untied(ids)["logits"].shape)  # both (1, 8, 512)
```

### Example 3: the weights contract, in full

```python
from dl_techniques.models.language.gpt2 import GPT2, create_gpt2

# pretrained=True is a hard error: this library distributes no GPT-2 weights.
for call in (
    lambda: GPT2.from_variant("tiny", pretrained=True),
    lambda: create_gpt2("tiny", pretrained=True),
):
    try:
        call()
    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")

# A path that does not exist is also a hard error.
try:
    GPT2.from_variant("tiny", pretrained="/no/such/checkpoint.keras")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
```

An existing path is loaded with `skip_mismatch=True` after a dummy forward pass builds the model,
through `utils.weight_transfer.load_weights_or_raise`. That helper counts variables whose value
actually changed and raises when the count is **zero**, so a checkpoint whose names or shapes do
not match cannot restore nothing and return normally. Its limit: the guard fires only on a
*completely* empty load. A **partial** restore (say 9 of 30 variables) logs a warning and returns
normally, which under `skip_mismatch=True` is possible. Check the logged restored-variable
count.

---

## 9. Advanced Usage Patterns

### Pattern 1: train on your own data

```python
import keras
import numpy as np
from dl_techniques.models.language.gpt2 import GPT2
from dl_techniques.losses import MaskedCausalLMLoss

model = GPT2(vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)

# The model returns a dict, so the loss is keyed on the "logits" output.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, clipnorm=1.0),
    loss={"logits": MaskedCausalLMLoss()},
)

# Causal LM targets are the inputs shifted by one position. The CALLER shifts.
ids = np.random.randint(0, 512, size=(8, 33)).astype("int32")
history = model.fit(
    {"input_ids": ids[:, :-1]},
    {"logits": ids[:, 1:]},
    batch_size=4, epochs=1, verbose=0,
)
print(sorted(history.history.keys()))  # ['loss']
```

### Pattern 2: GPT-2 as a feature extractor

`last_hidden_state` is the pre-projection representation. Take it and ignore `logits` when you
want features rather than a distribution over the vocabulary. Nothing needs to be disabled: the
tied head costs no parameters.

### Pattern 3: text generation

`GPT2` has **no** `generate()` method; sampling is not part of the model. The repo's sampling loop
lives in `src/train/common/generation_probe.py` (`GenerationProbeCallback`), wired in as a
callback by the pre-training script, and `src/dl_techniques/models/common/power_sampling/`
provides inference-time sampling for any causal LM.

---

## 10. Performance Optimization

```python
import keras
import numpy as np
from dl_techniques.models.language.gpt2 import GPT2

keras.mixed_precision.set_global_policy("mixed_float16")
model = GPT2.from_variant("tiny", vocab_size=512)
out = model(np.random.randint(0, 512, (2, 32)).astype("int32"))
print(out["logits"].dtype)  # float16
keras.mixed_precision.set_global_policy("float32")
```

- **Sequence length is architectural.** Attention cost is quadratic in `max_seq_len`, which also
  fixes the positional table's size, so a model built at 1024 cannot be fed 2048 positions.
- **Reduce `batch_size` before `depth`**: the activations dominate.
- `tie_word_embeddings=True` (the default) removes a `vocab x embed_dim` parameter block.
- The trainers pass `clipnorm=1.0` to `AdamW`; keep it when you write your own loop.

---

## 11. Training and Best Practices

Three runnable pipelines ship under `src/train/gpt2/`:

| Script | Purpose |
|---|---|
| `pretrain.py` | Causal-LM pre-training (TFDS or HuggingFace text sources). |
| `pretrain_so.py` | Same, plus soft-orthonormal regularization on the weight matrices. |
| `finetune.py` | Domain fine-tuning of an existing checkpoint. |

`--help` on each is the source of truth for its flags, not this file.

```bash
# A short pre-training run on GPU 1
MPLBACKEND=Agg .venv/bin/python -m train.gpt2.pretrain \
    --gpu 1 --variant tiny --dataset-source tfds \
    --dataset-name imdb_reviews --max-samples 64 \
    --epochs 1 --batch-size 2 --max-seq-length 32
```

- **The tokenizer defines `vocab_size`.** A mismatch is not an error, it is silently wrong
  output. The class default matches Tiktoken `cl100k_base`; the pre-training script chooses its
  own tokenizer and sizes the model to it.
- **Runs write to repo-root `results/`**, never `src/results/`, with `config.json` and
  `training_history.json` alongside the checkpoints.
- **Warmup + AdamW with `clipnorm=1.0`** is the shipped recipe (`train/common/nlp.py`).
- Fine-tuning starts from a local `.keras` checkpoint you produced; there is no public one.

---

## 12. Serialization & Deployment

```python
import keras
import numpy as np
from dl_techniques.models.language.gpt2 import GPT2

model = GPT2(vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
ids = np.random.randint(0, 512, size=(1, 8)).astype("int32")
before = model(ids)["logits"]

model.save("gpt2_demo.keras")
restored = keras.models.load_model("gpt2_demo.keras")  # no custom_objects needed
after = restored(ids)["logits"]

print(float(np.max(np.abs(np.array(before) - np.array(after)))))  # 0.0
```

`get_config()` stores all twelve constructor arguments; `from_config` is Keras's default, which
resolves to `cls(**config)` for this subclassed model.

---

## 13. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_gpt2/ tests/test_train/test_gpt2/ -q
```

The model suite covers initialization and validation
errors, forward shapes for tensor and dict inputs, weight tying **and** the untied head, causal
masking (a future token must not move an earlier position's logits), `get_config` / `.keras` round
trips, every named variant, gradient flow, the `pretrained=True` contract, and the exact `__all__`
surface. `tests/test_train/test_gpt2/` holds the trainer-side contract tests. Quote *passed with
collected*: a suite that collects almost nothing also reports green.

---

## 14. Troubleshooting & FAQs

- **`NotImplementedError: Pretrained GPT-2 weights are not distributed...`** Working as designed.
  Pass a local checkpoint path or `pretrained=False`.
- **`ValueError: embed_dim (X) must be divisible by num_heads (Y)`** Raised at construction by
  `_validate_config`. Pick head counts that divide the hidden size.
- **`ValueError: Dictionary input must contain 'input_ids' key`** Dict inputs are keyed exactly
  `input_ids` (required) and `attention_mask` (optional).
- **Out-of-memory.** Reduce `batch_size` first, then `max_seq_len` (quadratic), then the variant.
  Mixed precision (§10) is usually the cheapest win.
- **Generated text is nonsense.** Check that the tokenizer used at inference is the one used at
  training, and that `vocab_size` matches it exactly.

**Why is there no `generate()` on the model?** Sampling is a decoding policy, not architecture.
Keeping it in `train/common/generation_probe.py` and `models/common/power_sampling/` means a
change of sampler does not touch the model class.

**Tie or untie the embeddings?** Tied matches the original GPT-2 and is the default. Untied is the
modern choice at very large scale. Both are tested; switch with one flag.

**Where is the causal mask?** Inside `TextDecoder`, built by `dl_techniques.utils.masking` and
combined with your padding mask. `gpt2.py` contains no masking code at all.

---

## 15. Technical Details

### The factory-adoption audit

The repo's layer-reuse policy (`models/CLAUDE.md`) says to reach for a factory before hand-rolling
a layer. This package was audited against it and found to have **zero adoptable sites**:

- Everything except the head is delegated to `TextDecoder`, which already uses `TransformerLayer`,
  `create_normalization_layer` and the shared mask builders. GPT-2 reaches the factories
  **through** it, at the right depth.
- The one directly-constructed layer is the untied `Dense(vocab_size, use_bias=False)` LM head. No
  factory covers a bare vocabulary projection (it is not an attention, FFN, normalization,
  embedding or activation primitive), so there is nothing to adopt.

So `grep -rn "create_.*_layer" src/dl_techniques/models/language/gpt2/ --include="*.py"` returns
0, and that is **not** evidence of a problem: it measures the wrong thing. The question is whether
the model reaches the factories at some level, and it does, through `TextDecoder`. Keep the
`--include="*.py"` or this section is counted as a hit.

### Pre-norm

Each block normalizes *before* its sub-layer and adds the residual after
(`normalization_position="pre"`). This is what GPT-2 switched to, and it is why a final
`LayerNorm` is needed after the last block: without it the residual stream would leave the stack
unnormalized.

### Weight tying and the transpose

With tying, logits are `hidden @ E^T` where `E` is the `(V, D)` token embedding table: one
variable serving two roles. With `tie_word_embeddings=False` the head kernel is `(D, V)`, a
*different* variable of transposed shape, which is why an accidental alias between the two is
shape-illegal at any `vocab_size != embed_dim`.

### Output dict

`call()` always returns `{"logits", "last_hidden_state"}`. Keeping `last_hidden_state` costs
nothing (the head consumes that tensor anyway) and makes the model usable as an encoder without a
second forward pass.

---

## 16. Citation

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2019}
}
```
