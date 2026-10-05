# Language family — training backbones with CLM/MLM heads

Text and byte sequence models live here (encoders, decoders, SSMs, reasoning
stacks). The member list lives in `__init__.py`'s docstring; this file is
the training-head usage guide. The heads themselves live in
`models/common/masked_language_model/` — model-agnostic wrappers around any
backbone, which is why they are not under `language/`. Import from the leaf:

```python
from dl_techniques.models.common.masked_language_model import (
    MaskedLanguageModel, CausalLanguageModel, create_mlm_training_model,
)
```

## Which head for which backbone

| Backbone | Head | Why |
|---|---|---|
| Bidirectional encoder (BERT, ModernBERT, FNet, tree_transformer) | `MaskedLanguageModel` | scores masked positions only |
| Causal decoder (GPT-2, Qwen, Gemma, Mamba, WaveField, HNet, Zamba2…) | `CausalLanguageModel` | next-token prediction, one pass |

Never CLM on a bidirectional backbone: `build` runs a future-leak probe
(two forwards differing at one position must leave earlier states
unchanged) and raises `ValueError` otherwise. Pass
`verify_causality=False` only to skip a check you have run elsewhere.

## MLM: `create_mlm_training_model(encoder, vocab_size, mask_token_id, …)`

The factory builds **and compiles** (AdamW 5e-5) with BERT defaults
(`mask_ratio=0.15`, `random/unchanged=0.1`, `layer_norm_eps=1e-12`);
override via `mlm_config=` / `optimizer_config=`. Masking happens
**inside** `train_step`, so the dataset is plain tokenized text
(`train.common.nlp.preprocess_mlm_dataset`). Requirements: the encoder
exposes `hidden_size`; pass `special_token_ids` so they are never masked.
Do NOT add `metrics=["accuracy"]` — the name collides with the model's own
tracker and gets dropped. Full flag table: the leaf `README.md`.

## CLM: `CausalLanguageModel(backbone, vocab_size, …)`

Backbone contract: a `hidden_size` attribute and a mapping with
`last_hidden_state`; the output head ties to the backbone embeddings by
default (`tie_weights=True`; needs `get_embedding_matrix()` or a known
attribute name). `train_step`/`test_step` shift internally
(`x = input_ids[:, :-1]`, `y = input_ids[:, 1:]`); `call` scores as given,
so it doubles as the generation path. Flags that change the contract:

| Flag | When to pass it |
|---|---|
| `skip_head=True` | backbone already returns logits (plain tensor); `hidden_size` not required |
| `output_key=` | `skip_head` backbone returning a mapping (e.g. `{"logits": …}`) |
| `pre_shifted=True` | batches already `(input_ids, labels)` shifted upstream (`preprocess_clm_packed_dataset`); also forces `loss_weights=None`. Passing it otherwise trains every position against the wrong target |
| `loss_fn=` | replace CE wholesale (focal, label smoothing); delegates entirely, no stacking |
| `aggregate_backbone_losses=True` | backbone uses `add_loss` in `call()` (HNet boundary term) |
| `causality_probe_plain_tensor=True` | backbone `call()` takes a plain tensor, not the `{input_ids, attention_mask}` dict — without it the probe silently degrades to a warning and nothing was checked |

## Trainer wiring (Pattern 3)

`src/train/*/{pretrain,finetune}.py` + `train.common.nlp`: `create_tokenizer`,
`load_text_dataset` / `load_wikipedia_train_val`, `preprocess_{mlm,clm}_dataset`
(deliberately no `.cache()`), `estimate_clm_steps_per_epoch` (never roll a
local estimator), `create_warmup_lr_schedule`. Byte-level models (`vocab_size=256`)
cannot use the tiktoken pipeline — use `dl_techniques.datasets.byte_lm`.
`train_step` uses `tf.GradientTape` directly: TF backend only.

## Tests

`tests/test_models/test_masked_language_model/` (forward, masking, locator,
serialization) plus per-trainer `tests/test_train/test_<trainer>/`. Pitfalls
with RED proofs there: double-shift, probe degradation, metric-name clash.
