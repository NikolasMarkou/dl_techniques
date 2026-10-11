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
| Causal decoder (GPT-2, Qwen, Gemma, Mamba, WaveField, HNet, Zamba2, TopoLM…) | `CausalLanguageModel` | next-token prediction, one pass |

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
| `aggregate_backbone_losses=True` | backbone uses `add_loss` in `call()` (HNet boundary term; **TopoLM**'s spatial-smoothness taps, where it is the difference between training the objective and computing it every step and discarding it — measured 15.31 aggregated against 5.36 not on identical weights) |
| `causality_probe_plain_tensor=True` | backbone `call()` takes a plain tensor, not the `{input_ids, attention_mask}` dict — without it the probe silently degrades to a warning and nothing was checked |

## Task heads: `include_top` (fine-tuning, not pretraining)

BERT, DistilBERT, ModernBERT, FNet, TreeTransformer, GPT2, Mamba, Mamba2, TopoLM,
WaveFieldLLM, Qwen3, Qwen3Next and Gemma3 can own one task head, the ResNet way:

- `include_top=False` (default): no head; `call()` returns what it always did,
  which is what the CLM/MLM wrappers above consume.
- `include_top=True` needs a `task_config` (`NLPTaskConfig` or its dict);
  `head_config_overrides` go to the head. A `task_config` or overrides without
  `include_top=True` raise `ValueError`, as does `MULTIPLE_CHOICE`.
- The head is one `layers/heads/nlp` head named `task_head` (`TASK_HEAD_NAME`
  in `layers/heads/nlp/model_head.py`), fed the final hidden states and the
  padding mask. `call()` returns bare logits for a one-output head, a dict for
  a multi-output one (QA). Causal backbones pool `"last"` by default;
  `head_config_overrides={"pooling_type": ...}` changes that.
- Build through `create_<model>(variant, include_top=True, task_config=...)`,
  which only forwards to `from_variant`; the old functional head wrappers are
  gone. `create_qwen3_classification`, `create_qwen3_next_classification` and
  `create_gemma3_classification` remain, as wrappers over the in-model head.

Reusing a backbone under a new head:

- At construction: `pretrained="<path>.keras"`, headed or headless, restores
  the backbone only; the checkpoint's `task_head` is never copied and a new
  head keeps its fresh init. Available on the first ten models above (not
  Qwen3, Qwen3Next, Gemma3).
- The load rule, one for all ten classes, headed and headless: every
  backbone layer loads, or the call raises (a checkpoint of another depth or
  width is refused, never half-restored). The checkpoint's `task_head` is
  never copied. The vocabulary layers may stay at their fresh init only when
  the checkpoint's own `vocab_size` (read from the archive by
  `vocab_layers_if_changed`, never a class default) differs from the model's.
  An untied `lm_head` is a top, so it may also stay fresh when the checkpoint
  holds none (one saved with a task head). Pinned for every class except
  TreeTransformer by
  `tests/test_models/test_a_language_checkpoint_loads_completely_or_raises.py`
  (TreeTransformer: its own suite).

  | Class | Vocabulary layers | Route |
  |---|---|---|
  | BERT, ModernBERT, FNet | `embeddings` | by name |
  | DistilBERT | `embeddings` | whole file, strict, when headless with the checkpoint's vocabulary; else by name |
  | TreeTransformer | `embedding`, `lm_head` | by name |
  | GPT2 | none: a vocabulary change raises | by name |
  | Mamba, Mamba2 | `embedding` | whole file, strict, when headless with the checkpoint's vocabulary; else by name |
  | TopoLM | `word_embeddings`, `lm_head` | by name |
  | WaveFieldLLM | `token_embeddings`, `lm_head` | by name |

  GPT2 refuses a vocabulary change because its `decoder` is ONE layer holding
  the token embedding, so the embedding cannot stay fresh on its own. The
  whole-file routes are strict, measured to raise on a deeper or wider model
  and to ignore a checkpoint's `task_head`. The by-name routes need a `.keras`
  file.
- On a built model: `load_weights_from_checkpoint(model, path,
  skip_prefixes=(TASK_HEAD_NAME,), strict=True, allow_unloaded=())` from
  `dl_techniques.utils.weight_transfer` (add the class's vocabulary layers
  to `allow_unloaded` for a vocabulary change). Without `strict=True` and
  `allow_unloaded`, a backbone that does not fit loads nothing and returns
  normally. Never pass `skip_prefixes=()` across tasks: a head of equal shape
  is copied silently.

## Trainer wiring (Pattern 3)

`src/train/*/{pretrain,finetune}.py` + `train.common.nlp`: `create_tokenizer`,
`load_text_dataset` / `load_wikipedia_train_val`, `preprocess_{mlm,clm}_dataset`
(deliberately no `.cache()`), `estimate_clm_steps_per_epoch` (never roll a
local estimator), `create_warmup_lr_schedule`. Byte-level models (`vocab_size=256`)
cannot use the tiktoken pipeline — use `dl_techniques.datasets.byte_lm`.
`train_step` uses `tf.GradientTape` directly: TF backend only.

## Backbones that cannot use the shared transformer stack

Two members ship a decoder block of their own rather than composing
`layers/transformers/`, for the same structural reason: the shared
`TransformerLayer` adds each residual branch straight into the stream, so in the
pre-norm path the branch tensor is a local that stochastic depth and layer scale
overwrite before the add — **it is gone by the time `call` returns**. A backbone
whose objective needs that tensor *before* normalization and *before* the residual
add cannot reach it: no flag, no hook, no `activity_regularizer`.

- `topolm/` — `TopoLMBlock`, whose two taps carry `SpatialSmoothness`. It composes
  `create_attention_layer` / `create_ffn_layer` / `create_normalization_layer`
  rather than reimplementing them, so the reuse order (factories first, a bespoke
  layer last) is unchanged. The three factory calls pass **positional string
  literals** for their types on purpose: the repo's dropped-param sweep resolves a
  literal type statically and cannot resolve a variable one, and its ceiling is
  explicitly not to be raised to absorb a new site.
- `hnet/` — same constraint, different reason (learned chunk boundaries).

If you are adding a third, budget for it: a per-layer config knob is unreachable
through `TextDecoder` (only `gpt2/` uses it), and `CausalLanguageModel` must be
built with `aggregate_backbone_losses=True`.

## Tests

`tests/test_models/test_masked_language_model/` (forward, masking, locator,
serialization) plus per-trainer `tests/test_train/test_<trainer>/`. Pitfalls
with RED proofs there: double-shift, probe degradation, metric-name clash.

In `test_topolm/` the load-bearing claims are pinned by measurement rather than
by shape: a smooth field scores 0.172 and the same field against a negated
distance prior scores 0.826 (the sign-discriminating probe), `alpha = 0` holds a
byte-identical weight set *and* identical permutations, and `build()`
materialises the tree so a reload has somewhere to land.

