# Masked Language Model (MLM) Pre-training Framework

A model-agnostic pre-training wrapper for the masked language modelling objective,
plus a causal counterpart. Point it at any encoder that satisfies a two-item contract
and it handles corruption, the masked loss, and the metrics.

---

## 1. Overview: What This Package Is and Why It Matters

`MaskedLanguageModel` wraps an encoder you supply, corrupts its input on the fly, and
trains it to reconstruct the corrupted positions. When training finishes you throw the head
away and keep the encoder, which is now a general-purpose feature extractor. The package
also ships `CausalLanguageModel` (the next-token counterpart), a factory, and a prediction
visualizer; see § 6.

Two properties do most of the work. The wrapper is **model-agnostic**: any encoder
exposing `hidden_size` and returning `{"last_hidden_state": ...}` works. And masking is
**dynamic**, drawn fresh inside `train_step`, so the same example is corrupted differently
every epoch. That is RoBERTa's improvement over BERT's static masking, and it costs
nothing in storage.

---

## 2. The Problem MLM Solves

Autoregressive factorization conditions each token only on its left context. That is
efficient supervision, but it forces every representation to be one-sided: the vector at
position `t` has never seen `x_{t+1}`.

Masked language modelling removes that constraint by changing the *objective* rather than
the architecture. Delete a random subset `S` of positions and maximize
`sum_{i in S} log p(x_i | x_notS)`. Because the target is held out of the input, attention
can be fully bidirectional without the prediction becoming trivial.

The price is supervision density: only selected positions produce signal, so a sequence
yields about `mask_ratio * T` targets instead of `T`. That is why `mask_ratio` is a tuned
quantity and not a free parameter.

---

## 3. How MLM Works: Core Concepts

### 3.1 The 80/10/10 corruption split

The corruption is not a plain deletion, and the reason is a train/deploy mismatch: the
`[MASK]` id is an artifact of pre-training and never appears downstream. Conditioning
entirely on it would give a representation that is only correct in the presence of a token
the encoder will never meet again. So of the selected positions:

| Fraction | Treatment | Why |
|---|---|---|
| 0.8 (`1 - random - unchanged`) | replaced with `[MASK]` | the primary reconstruction signal |
| 0.1 (`random_token_ratio`) | replaced with a uniform vocabulary id | stops the model assuming the observed id is correct |
| 0.1 (`unchanged_ratio`) | left verbatim | the important one, see below |

All three groups are scored. The unchanged group matters most: since the model cannot
tell a scored-but-unchanged position from an ordinary one, it must keep a predictive
representation at *every* position rather than only where a mask token flags one.

Selection and corruption are independent per-token uniform draws, not quotas.
15/80/10/10 are expectations; the number of masked positions varies batch to batch.

### 3.2 What is never masked

Positions whose id appears in `special_token_ids` are excluded by value equality.
Padding is excluded **only when an `attention_mask` is supplied**. Call this without
one and pad positions are eligible for masking and are scored like any other token.

### 3.3 The loss

Labels are the full uncorrupted id tensor, so the per-position cross-entropy is dense and
the restriction to masked positions happens in the reduction. The boolean selection mask is
passed as `sample_weight`, giving `loss = sum(CE * mask) / max(sum(mask), 1)`.

Unselected positions are multiplied by exactly zero, so they contribute neither loss nor
gradient. Normalizing by the number of *selected* tokens rather than by sequence length
makes the loss scale independent of how many tokens the Bernoulli draw happened to pick.
The `max(..., 1)` floor keeps a batch that selected nothing finite instead of `0/0`.

### 3.4 Data flow

```
input_ids  ->  dynamic masking (per step)  ->  encoder  ->  last_hidden_state (B, T, H)
           ->  MLM head: Dense(H, gelu) -> Dropout -> LayerNorm -> Dense(vocab_size)
           ->  masked cross-entropy against the ORIGINAL ids
```

---

## 4. Architecture Deep Dive

### 4.1 The encoder contract

Two requirements, both checked: a `hidden_size` attribute (a missing one raises
`ValueError` at construction), and a `call()` returning a mapping containing
`last_hidden_state` of shape `(B, T, H)`. Nothing else is assumed.

### 4.2 The prediction head

`Dense(hidden_size, gelu) -> Dropout -> LayerNorm -> Dense(vocab_size)`.

Two deliberate divergences from BERT's head:

- **The output projection is not tied to the input embedding table.** The wrapper cannot
  assume an arbitrary encoder exposes an embedding matrix, or under what attribute, so
  tying would make the model-agnostic contract conditional on encoder internals. The cost
  is an extra `hidden_size * vocab_size` parameters and the regularization tying provides.
  `CausalLanguageModel` *does* attempt tying heuristically, falling back to an untied
  `Dense`.
- **Dropout sits between the activation and the LayerNorm.** BERT places LayerNorm
  directly after the activation and uses no dropout in the head.

### 4.3 Backend and metrics

`train_step` and `test_step` are hand-written over `tf.GradientTape`, so this model is
**TensorFlow-backend only**.

The `metrics` property returns two internal trackers named `loss` and `accuracy`, plus
whatever you passed to `compile(metrics=...)`. A compiled metric whose *name* collides with
either is dropped, because the step's return dict is keyed by name; it logs a warning
rather than vanishing silently. Name yours something else, for example `mlm_accuracy`.

Validation masking is dynamic too, so `val_loss` carries the noise of a fresh corruption
draw: an epoch-to-epoch comparison mixes model change with sampling noise.

### 4.4 `call()` does no masking

`call()` runs the encoder on the inputs as given and returns logits for every position,
which is what makes `predict()` usable for scoring a sequence you masked yourself.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.language.bert.model import BERT
from dl_techniques.models.language.masked_language_model import MaskedLanguageModel

VOCAB, SEQ = 500, 32

# 1. Any encoder with `hidden_size` + a "last_hidden_state" output.
encoder = BERT.from_variant("tiny", vocab_size=VOCAB, max_position_embeddings=SEQ)

# 2. Wrap it.
mlm_model = MaskedLanguageModel(
    encoder=encoder,
    vocab_size=VOCAB,
    mask_token_id=4,
    special_token_ids=[0, 1, 2, 3],
)

# 3. Compile. Do NOT name a metric "accuracy": it collides with the internal
#    tracker and gets dropped with a warning.
mlm_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="mlm_accuracy")],
)

# 4. Train. The dataset yields UNMASKED ids; masking happens in train_step.
rng = np.random.default_rng(0)
ids = rng.integers(5, VOCAB, (8, SEQ)).astype("int32")
attention_mask = np.ones((8, SEQ), dtype="int32")
train_dataset = tf.data.Dataset.from_tensor_slices(
    {"input_ids": ids, "attention_mask": attention_mask}
).batch(4)

history = mlm_model.fit(train_dataset, epochs=1, verbose=0)
print(sorted(history.history))          # ['accuracy', 'loss', 'mlm_accuracy']

mlm_model.encoder.save("encoder.keras")  # 5. keep the encoder, drop the head
```

With a real tokenizer, `vocab_size=tokenizer.vocab_size`,
`mask_token_id=tokenizer.mask_token_id` and
`special_token_ids=tokenizer.all_special_ids`.

---

## 6. Component Reference

| Symbol | Role |
|---|---|
| `MaskedLanguageModel` | The MLM pre-trainer. |
| `CausalLanguageModel` | Next-token pre-trainer; first arg is `backbone`, not `encoder`. |
| `create_mlm_training_model` | Factory; returns a **compiled** model. |
| `visualize_mlm_predictions` | Prints decoded original / masked / reconstruction. |

All four import from `dl_techniques.models.language.masked_language_model`.

```python
MaskedLanguageModel(
    encoder,                        # keras.Model, required
    vocab_size,                     # int, required
    mask_ratio=0.15,
    mask_token_id=103,              # BERT's [MASK]
    random_token_ratio=0.1,
    unchanged_ratio=0.1,
    special_token_ids=None,         # Optional[List[int]]
    mlm_head_activation="gelu",
    initializer_range=0.02,
    mlm_head_dropout_rate=0.1,
    layer_norm_eps=1e-12,
)
```

Validated at construction: `vocab_size > 0`, `0 < mask_ratio <= 1`,
`0 <= mask_token_id < vocab_size`, `random_token_ratio + unchanged_ratio <= 1`,
`initializer_range > 0`, `0 <= mlm_head_dropout_rate < 1`.

`CausalLanguageModel(backbone, vocab_size, initializer_range=0.02, tie_weights=True,
verify_causality=True, causality_tolerance=0.0)`. Its causality probe runs two forward
passes over ids differing at one position and raises `ValueError` if any earlier position
moved, so a bidirectional backbone fails loudly instead of training to a spectacular,
meaningless loss.

---

## 7. Configuration & Model Variants

The wrapper has no variants of its own; they come from the encoder. For
`BERT.from_variant` the real key set is `["large", "base", "small", "tiny"]`:

| Variant | Layers | Hidden | Heads | Intermediate |
|---|---|---|---|---|
| `tiny` | 4 | 256 | 4 | 1024 |
| `small` | 6 | 512 | 8 | 2048 |
| `base` | 12 | 768 | 12 | 3072 |
| `large` | 24 | 1024 | 16 | 4096 |

Masking presets:

| Setting | `mask_ratio` | `random_token_ratio` | `unchanged_ratio` | `mlm_head_dropout_rate` |
|---|---|---|---|---|
| Standard BERT pre-training | 0.15 | 0.1 | 0.1 | 0.1 |
| Low-resource | 0.20 | 0.05 | 0.05 | 0.15 |
| Domain adaptation | 0.10 | 0.05 | 0.05 | 0.05 |

These are constructor arguments. Setting `mlm_head_dropout_rate` on an already-built model
does nothing: the `Dropout` layer was created with the old rate.

---

## 8. Comprehensive Usage Examples

### Example 1: the factory

`create_mlm_training_model` returns a model already compiled with AdamW. The encoder must
be **built** first: the factory logs the encoder's parameter count, and `count_params()`
raises on an unbuilt layer.

```python
from dl_techniques.models.language.bert.model import BERT
from dl_techniques.models.language.masked_language_model import (
    create_mlm_training_model,
)

encoder = BERT.from_variant("tiny", vocab_size=500, max_position_embeddings=32)
encoder({"input_ids": ids[:1], "attention_mask": attention_mask[:1]})  # build it

mlm_model = create_mlm_training_model(
    encoder=encoder,
    vocab_size=500,
    mask_token_id=4,
    special_token_ids=[0, 1, 2, 3],
    mlm_config={"mask_ratio": 0.15},
    optimizer_config={"learning_rate": 5e-5, "weight_decay": 0.01},
)
mlm_model.fit(train_dataset, epochs=1, verbose=0)
```

The factory compiles **no** metrics on purpose: the accuracy it used to add was named
`"accuracy"`, collided with the internal tracker, and was silently dropped. Masked accuracy
is already reported under that key.

### Example 2: a custom encoder

```python
import keras

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.models.language.masked_language_model import MaskedLanguageModel


# The package string is the defining module's dotted path: `my_project.<module>` for your
# own code, `dl_techniques.<module.path>` for code inside this repo. Never a bare
# `@keras.saving.register_keras_serializable()`.
@register_dl_technique("my_project.custom_encoder")
class CustomEncoder(keras.Model):
    def __init__(self, vocab_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size          # REQUIRED attribute
        self.embedding = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size, activation="gelu")

    def call(self, inputs, training=None):
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        hidden = self.dense(self.embedding(input_ids), training=training)
        return {"last_hidden_state": hidden}    # REQUIRED key

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size,
                       "hidden_size": self.hidden_size})
        return config


mlm_model = MaskedLanguageModel(
    encoder=CustomEncoder(vocab_size=500, hidden_size=64),
    vocab_size=500,
    mask_token_id=4,
)
```

### Example 3: inspecting predictions

`visualize_mlm_predictions(mlm_model, inputs=batch, tokenizer=tokenizer, num_samples=4)`
logs one block per sample: the original text, the masked input, and the reconstruction with
predictions substituted at the masked positions only. `tokenizer` needs only a `decode`
method.

---

## 9. Advanced Usage Patterns

### Pattern 1: fine-tuning the pre-trained encoder

```python
import keras

encoder = keras.models.load_model("encoder.keras")
encoder.trainable = True

# `name=` is not decoration. An unnamed keras.Input inside a dict gets an
# auto-generated key, which then becomes the key `fit`/`predict` demand.
inputs = {
    "input_ids": keras.Input(shape=(32,), dtype="int32", name="input_ids"),
    "attention_mask": keras.Input(shape=(32,), dtype="int32", name="attention_mask"),
}

sequence_output = encoder(inputs)["last_hidden_state"]
cls_token = sequence_output[:, 0, :]                 # [CLS], for sequence tasks
logits = keras.layers.Dense(3)(keras.layers.Dropout(0.1)(cls_token))

classifier = keras.Model(inputs, logits)
classifier.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

For token-level tasks (NER, tagging) keep the full `sequence_output` instead of slicing.

### Pattern 2: using the repo's NLP heads

The heads read `hidden_states`, but the encoder emits `last_hidden_state`. **Rename the
key** or `predict()` raises `KeyError: 'hidden_states'`:

```python
from dl_techniques.layers.heads.nlp import (
    create_nlp_head, NLPTaskConfig, NLPTaskType,
)

task_config = NLPTaskConfig(
    name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3
)
head = create_nlp_head(task_config=task_config, input_dim=encoder.hidden_size)

encoder_outputs = dict(encoder(inputs))
encoder_outputs["hidden_states"] = encoder_outputs.pop("last_hidden_state")
task_outputs = head(encoder_outputs)   # {"logits", "probabilities"}

sentiment_model = keras.Model(inputs, task_outputs["logits"])
```

### Pattern 3: continual pre-training

```python
mlm_model = keras.models.load_model("checkpoints/mlm_model_03.keras")
mlm_model.mask_ratio = 0.10   # read per step, so this DOES take effect
mlm_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5))
mlm_model.fit(domain_dataset, epochs=5, initial_epoch=3)
```

---

## 10. Performance Optimization

Learning-rate schedule and optimizer, built from config dicts:

```python
from dl_techniques.optimization import (
    optimizer_builder, learning_rate_schedule_builder,
)

total_steps = num_epochs * steps_per_epoch

lr_schedule = learning_rate_schedule_builder({
    "type": "cosine_decay",   # also: exponential_decay, cosine_decay_restarts
    "learning_rate": 5e-5,
    "decay_steps": total_steps,
    "warmup_steps": int(0.1 * total_steps),
    "warmup_start_lr": 1e-8,
    "alpha": 1e-6,
})
optimizer = optimizer_builder(
    {"type": "adamw", "weight_decay": 0.01, "gradient_clipping_by_norm": 1.0},
    lr_schedule,
)
mlm_model.compile(optimizer=optimizer)
```

For mixed precision call `keras.mixed_precision.set_global_policy("mixed_float16")`
**before** constructing the model. `train_step` calls `optimizer.scale_loss(loss)` inside
the tape, so scaling is handled; do not pass `loss_scale=` to `compile`, it is not a Keras
3 parameter.

Pre-tokenize offline, then `.cache().shuffle(...).batch(...).prefetch(tf.data.AUTOTUNE)`.
Corruption is regenerated per step inside the model, so caching tokenized ids does not make
the masking static.

---

## 11. Training and Best Practices

- 15% masking is the tuned default. Raising it gives more targets per sequence but less
  context to predict from.
- Warm up for roughly 10% of total steps. MLM is unstable without warmup.
- Clip gradients by norm (1.0 is standard, 0.5 if the loss spikes).
- Track `accuracy` (masked accuracy). It rises slowly. Flat accuracy across a long run
  usually means `vocab_size`, `mask_token_id` or `special_token_ids` disagree with the
  tokenizer, not that the model is too small.
- Validation loss is noisy by construction (dynamic masking). Compare trends, not epochs.

---

## 12. Serialization & Deployment

```python
mlm_model.encoder.save("encoder.keras")   # what you want for downstream tasks
mlm_model.save("mlm_full.keras")          # wrapper + head, to resume pre-training

encoder = keras.models.load_model("encoder.keras")   # no custom_objects needed
```

Use `.keras`, not `.h5`. Every class here is registered through `register_dl_technique`, so
no `custom_objects` dictionary is required, provided a custom encoder of yours is
registered too.

---

## 13. Troubleshooting

| Symptom | Cause |
|---|---|
| `ValueError: The provided encoder must have a 'hidden_size' attribute.` | The encoder contract in § 4.1. Add the attribute. |
| A compiled metric never appears in the logs | Its name collides with `loss` or `accuracy`. Rename it; a warning is logged. |
| `ValueError: ... count_params ... isn't built` from `create_mlm_training_model` | Call the encoder once on a batch before passing it to the factory. |
| `KeyError: 'hidden_states'` from an NLP head | Rename `last_hidden_state` to `hidden_states` (§ 9, Pattern 2). |
| `CausalLanguageModel` raises about a future leak | The backbone is bidirectional. That is the probe working, not a bug. |
| Padding tokens are masked and scored | You did not pass `attention_mask`. Padding is excluded only when it is present. |
| Loss is `NaN` | Lower the learning rate to 1e-5, clip gradients to 0.5, check for out-of-range token ids. |

---

## 14. Technical Details

- **Backend:** TensorFlow only. `train_step` uses `tf.GradientTape` directly.
- **Masking implementation:** `dl_techniques.utils.masking.strategies.apply_mlm_masking`.
- **Registration key:** `dl_techniques.models.masked_language_model.mlm>MaskedLanguageModel`.
- **Loss scaling:** the *unscaled* loss is reported while the scaled one is differentiated.
  Under `mixed_float16` Keras wraps the optimizer in a `LossScaleOptimizer` whose `apply()`
  divides every gradient by the dynamic scale unconditionally, so omitting `scale_loss`
  would silently divide the whole update.

---


## 15. References

1. Devlin et al., 2018. *BERT: Pre-training of Deep Bidirectional Transformers for
   Language Understanding.* https://arxiv.org/abs/1810.04805
2. Liu et al., 2019. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
   https://arxiv.org/abs/1907.11692 (dynamic masking)
3. Press and Wolf, 2017. *Using the Output Embedding to Improve Language Models.*
   https://arxiv.org/abs/1608.05859 (weight tying)
4. Taylor, 1953. *"Cloze Procedure": A New Tool for Measuring Readability.*
   Journalism Quarterly.
