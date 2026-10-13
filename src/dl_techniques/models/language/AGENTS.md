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
- The `task_config` is checked at construction: a classification task's
  `num_classes` must be a positive integer (0, -1 or a bool raise
  `ValueError`), and a generation-type head's `vocabulary_size` must equal the
  model's vocabulary. An archive saved with a stored generation `task_config`
  that kept `vocabulary_size` 32000 on a model of another vocabulary no longer
  loads; the migration is in `layers/heads/AGENTS.md`.
- The head is one `layers/heads/nlp` head named `task_head` (`TASK_HEAD_NAME`
  in `layers/heads/nlp/model_head.py`), fed the final hidden states and the
  padding mask. `call()` returns bare logits for a one-output head, a dict for
  a multi-output one (QA). Causal backbones pool `"last"` by default;
  `head_config_overrides={"pooling_type": ...}` changes that.
- All 15 factories (these 13, `create_zamba2` and `create_hnet`) are uniform:
  `create_X(variant, pretrained=False, **kwargs)` only delegates to
  `X.from_variant` and returns the class itself, and the head comes from
  `include_top` + `task_config` + `head_config_overrides` in `__init__`,
  never from the factory. Some keep a named argument that is only
  forwarded: `vocab_size` (required by Mamba and Mamba2) and
  `weights_dataset` / `cache_dir` (BERT, DistilBERT, ModernBERT, FNet,
  TreeTransformer). For BERT, TreeTransformer, GPT2 and WaveFieldLLM,
  `vocab_size` is an ordinary `**kwargs` override laid over the variant (omit
  it to keep the variant's, or the class default; `vocab_size=None` is no
  longer accepted). Build a headed model with
  `create_<model>(variant, include_top=True, task_config=...)`; there is no
  per-task factory and no functional head wrapper.
- Zamba2 and HNet own a head too, with one difference: headless, `call()` returns the
  LOGITS tensor (not a hidden-state dict; their CLM wrappers use `skip_head=True`), and
  that does not change. With `include_top=True` the final-norm output (HNet: `hidden`, the
  backbone output, one vector per input position) feeds the head (`'last'` pooling) and
  the logits are not computed; HNet builds no `lm_head` then and keeps its ratio loss
  (`len(model.losses) == 1` in both tops). Input may be the ids or a dict with
  `attention_mask` (HNet: its `padding_mask`, which reaches the backbone too; right
  padding only, the Mamba-2 mixer takes no mask).

Which mask the head pools under, when the caller passes none (the backbone's
own attention is unchanged):

| Classes | Head mask without a caller mask | Where a caller's mask goes | Default pooling |
|---|---|---|---|
| BERT, DistilBERT, ModernBERT, FNet (bidirectional encoders) | all ones, the same mask the encoder returns: padding is pooled under `mean` / `max` / `attention` | `attention_mask`: the encoder's attention and the head | `cls` |
| TreeTransformer | `input_ids != pad_token_id` | `attention_mask`: the encoder's attention and the head | `cls` |
| GPT2, Mamba, Mamba2, TopoLM, WaveFieldLLM, Qwen3, Qwen3Next, Gemma3 (causal decoders) | `input_ids != pad_token_id` if a pad id was given, else all ones | the head only; the decoder's own attention sees just the mask passed to it | `last` |
| Zamba2 | `input_ids != pad_token_id` if a pad id was given, else all ones | `attention_mask` (dict input) reaches the head only: neither the Mamba-2 mixer nor the shared attention takes a padding mask | `last` |
| HNet | same rule | `padding_mask` (or the dict's `attention_mask`) reaches the backbone AND the head. RIGHT padding only: right-padded rows match the unpadded row to 1e-7 (measured), interior or left padding changes later positions because the Mamba-2 mixer takes no mask | `last` |

A caller's `attention_mask` always wins. The decoders' `pad_token_id` defaults to
`None`, which means no padding (the all-ones mask): id 0 is a real token in the
GPT2 and Qwen vocabularies (cl100k and Qwen BPE; `!` in both tiktoken encodings),
and the old default 0 dropped it from the pooled positions. A headed decoder fed a
padded batch must therefore pass `attention_mask` or be built with the tokenizer's
pad id, else the head pools the padding (under `last`, it reads a pad position).
An explicit int in an old archive's config is preserved. Headless decoders never
read the value. `tests/test_models/test_a_decoder_assumes_no_padding_token.py`
pins the default and the all-ones behaviour for these eight classes.

Reusing a backbone under a new head:

- At construction: `pretrained="<path>.keras"`, headed or headless, restores
  the backbone only; the checkpoint's `task_head` is never copied and a new
  head keeps its fresh init. Available on all 15 classes, through
  `create_<model>(variant, pretrained=...)` or `from_variant`. For Qwen3,
  Qwen3Next and Gemma3 `pretrained=True` raises `NotImplementedError`, like
  GPT2's, and an old archive of the two-input wrapper their factories used to
  return still loads into the class through the same rule.
- The load rule, one for all 15 classes, headed and headless, enforced in
  one place (`load_backbone_from_checkpoint` in
  `dl_techniques.utils.weight_transfer`, which every class calls): every
  backbone layer loads, and every backbone layer of the checkpoint finds a
  home, or the call raises. A checkpoint of another depth (deeper OR
  shallower) or width is refused, never half-restored or truncated. The
  checkpoint's `task_head` is never copied. On a vocabulary change (the
  checkpoint's own `vocab_size`, read from the archive, never a class
  default, differs from the model's) only the variables shaped by the
  vocabulary keep their fresh init, listed in a WARNING log; every other
  variable of the vocabulary layers (position and token-type tables,
  LayerNorm, the rest of GPT2's decoder) loads or the call raises. An untied
  `lm_head` (TreeTransformer: `lm_head_projection`) is a top: it may be left
  out on either side only when one side carries a task head instead (the
  checkpoint's config or the model says `include_top`) or the vocabulary
  changed. So a tied headless checkpoint does not fill an untied model, and
  an untied `lm_head` is not dropped into a tied one. Pinned for every class
  by `tests/test_models/test_a_language_checkpoint_loads_completely_or_raises.py`
  (deeper and wider targets, a deeper checkpoint, vocabulary plus position
  change, tied and untied tops, and the legitimate routes).

  | Class | Vocabulary layers | Top |
  |---|---|---|
  | BERT, DistilBERT, FNet | `embeddings` (word, position, token-type tables, LayerNorm) | none |
  | ModernBERT | `embeddings` | none |
  | TreeTransformer | `embedding` | `lm_head_projection` |
  | GPT2 | `decoder` (the token table inside the one-layer decoder) | `lm_head` when untied |
  | Mamba, Mamba2 | `embedding` | none |
  | Zamba2 | `embedding` | none (tied) |
  | HNet | `embeddings` (vocabulary read from `arch_config`) | `lm_head` when untied |
  | TopoLM | `word_embeddings` | `lm_head` when untied |
  | WaveFieldLLM | `token_embeddings` | `lm_head` when untied |
  | Qwen3, Qwen3Next, Gemma3 | `token_embedding` | `lm_head` (always untied) |

  Every route copies by layer name, so it needs a `.keras` file.
- On a built model: `load_backbone_from_checkpoint(model, path,
  head_prefixes=(TASK_HEAD_NAME,), vocab_layers=..., tables=..., tops=...)`, with the
  class's row above. `tables` are the variable-path suffixes of the token table
  (`word_embeddings/embeddings`, `embedding/embeddings`, ...) and of an untied
  `lm_head/kernel`: on a vocabulary change only those stay fresh, so a resized
  position table raises (plan-2026-10-11T170958-dc4d2202 D-007). The lower-level call is
  `load_weights_from_checkpoint(model, path,
  skip_prefixes=(TASK_HEAD_NAME,), strict=True, allow_unloaded=())`
  (add `"lm_head"` to `allow_unloaded` when one side is headed). Without
  `strict=True` and `allow_unloaded`, a backbone that does not fit loads
  nothing and returns normally. Never pass `skip_prefixes=()` across tasks: a head of equal shape
  is copied silently. An old archive of the retired Qwen3, Qwen3Next or Gemma3
  two-input wrapper still loads into the class: a functional model whose only
  layer besides its inputs is a model is compared through that inner model's
  layers.

## Sequence-length guard (`utils/sequence_length_guard.py`)

A model with a real position limit refuses an over-long sequence cleanly on every
path: a statically known length raises `SequenceLengthError` (a `ValueError`)
from `build` before any variable exists; a length known only at run time (`predict` after two lengths,
`jit_compile=True`) fails a graph assertion plus a zero-sum term that survives
XLA, where a gather-based position table would otherwise return finite garbage;
and the refusal drops Keras's cached `predict`/`train`/`test` step functions so
the next legal call still works. A `CausalLanguageModel` or `MaskedLanguageModel`
around a guarded backbone caches step functions of its own: its `build` catches
`SequenceLengthError` from the backbone, drops them and re-raises, so the wrapper
stays unbuilt and its next legal call works too (any other backbone build error
keeps the CLM's logged-warning path); pinned by
`tests/test_models/test_masked_language_model/test_a_refused_backbone_leaves_the_wrapper_usable.py`.
An old `.keras` archive whose recorded build shape is over-long (saved before
the guard) still loads, with every weight and a logged warning, and the loaded
model refuses an over-long call: Keras rebuilds from the recorded shape inside a
bare `try/except` and marks the model built anyway, so a refusal there used to
load it with no weights. Every guarded class except BLT lists
`RecordedBuildShapeMixin` first among its bases (one line); BLT has its own
`build_from_config`. `TextDecoder` (GPT2) honours the same load flag.
Usage and the functions: `utils/AGENTS.md`. Pinned per class by
`tests/test_models/test_a_position_limit_is_refused_cleanly.py`: one `Subject`
row per class, which goes RED when its guard line is removed. For the three
conditionally guarded classes (ModernBERT, Zamba2, HNet) that row covers only
the guarded configuration; a `NO_LIMIT` row with the condition false covers the
other direction and goes RED when the guard is made unconditional.

| Status | Classes |
|---|---|
| Guarded (limit attribute) | BERT, DistilBERT, FNet (`max_position_embeddings`), TreeTransformer (`max_len`), GPT2, TopoLM, WaveFieldLLM, Qwen3, Qwen3Next, Gemma3 (`max_seq_len`) |
| Guarded only for some configurations | ModernBERT (only when a global-attention layer exists: `num_layers >= global_attention_interval`), Zamba2 (only with a `g` entry in `layer_mapping`, the one RoPE table), HNet (only when stage 0 holds an attention block; inner stages see at most `max_chunks` positions and truncate silently, which is out of scope) |
| Guarded, own load override | BLT (`max_sequence_length`): its own `build_from_config` of an old archive whose recorded shape is over-long logs a warning and builds, never refuses (every other guarded class does the same through `RecordedBuildShapeMixin`) |
| No position limit, not guarded | Mamba, Mamba2 (and the unguarded configurations above): they serve any length, finite, eagerly and under `jit_compile=True`; the test's `NO_LIMIT` arm goes RED if a guard is wired unconditionally |

A model that builds its sub-layers by hand (BERT, TopoLM, Zamba2, HNet, BLT) also
calls `refuse_static_sequence_length` first in `build`; one whose `build` traces
`call` through `materialize_sublayers` needs only the `call` line.

## Packages that do not own a task head

Not every member is a token model a task head applies to. These keep their own
contract, on purpose; none was changed by the head work.

| Package | Why it has no `include_top` / `task_config` |
|---|---|
| `hrm`, `trm`, `ptrm` | Carry-based ACT puzzle solvers: the `lm_head` IS the task (per-cell prediction) and is trained jointly with the Q-halting head; the state is a recurrence, not a backbone a classification head would read. Their trainers use bespoke carry loops |
| `hrm`, `ptrm` factories | Uniform: `create_hierarchical_reasoning_model(variant, vocab_size, seq_len, ..., pretrained=False, **kwargs)` and `create_ptrm(variant, vocab_size, seq_len, puzzle_emb_len=16, pretrained=False, **kwargs)` only delegate to `from_variant`, which returns a built, UNCOMPILED model (`pretrained=True` raises `NotImplementedError`, since no HRM or PTRM weights are distributed; a path restores a same-shape checkpoint strictly). HRM's old compile is the caller's: `loss={"logits": StableMaxCrossEntropy()}` trains the LM head only, `src/train/hrm/train_hrm.py` trains the Q head. `trm` has no variant table and keeps `create_trm(vocab_size, hidden_size, ...)` |
| `mini_vec2vec` | Not a sequence model: embeddings in, one fitted matrix `W` (numpy `align()`), nothing to put a head on |
| `mini_vec2vec` factory | `create_mini_vec2vec_aligner(embedding_dim, **kwargs)` takes no `variant`: the aligner has no named sizes, and models/AGENTS.md forbids inventing a `MODEL_VARIANTS` table |
| `colbert` | Output is per-token masked L2-normalised embeddings scored by MaxSim; every `layers/heads/nlp` head pools to one vector, which would destroy late interaction. Its encoder is a `BERT` (so the backbone side of reuse is already met) |
| `colbert` factories | `create_colbert`, `create_colbert_v1` and `create_colbert_v2` all have the uniform shape and build the same network; the `_v1`/`_v2` names only record which training recipe the caller targets |
| `blt` | Byte LM with its own entropy model, patcher, `train_step` and `generate`; logits come from an untied `Dense` inside `LocalDecoder`. The per-byte hidden state exists inside `LocalDecoder`, which the whole-layer load rule compares as one unit, so a head would need a layer change |
| `blt` factory | `create_blt_model(variant, vocab_size=260, max_sequence_length=2048, entropy_threshold=1.5, **kwargs)` delegates to `from_variant` but has no `pretrained`: nothing in `blt/` loads weights (no `from_variant` argument, no load method), so the argument would have no route to reach; a load route is its own design, not a factory edit |
| `qwen` (`Qwen3EmbeddingModel`) | An embedding tower, not a head host: last-non-pad pooling, optional MRL truncation and L2 normalisation ARE the output. Its own `TransformerLayer` trunk, not `Qwen3`; no variant table, no `create_*` factory. No `layers/heads/nlp` head has the truncate/normalise options (`TextSimilarityHead` adds a norm, an intermediate block and a biased `Dense`) |
| `qwen` (`Qwen3RerankerModel`) | The score is `softmax([no, yes])` over the LM head's last-token logits, so the LM head IS the task; a `TextClassificationHead(2)` would be a different, freshly initialised model |

The 15 task-head classes (BERT, DistilBERT, ModernBERT, FNet, TreeTransformer,
GPT2, Mamba, Mamba2, TopoLM, WaveFieldLLM, Qwen3, Qwen3Next, Gemma3, Zamba2,
HNet) are the ones that hold the same-weights head-swap contract; the exempt
table above is the complete list of the rest of `language/`.

## Trainer wiring (Pattern 3)

`src/train/*/{pretrain,finetune}.py` + `train.common.nlp`: `create_tokenizer`,
`load_text_dataset` / `load_wikipedia_train_val`, `preprocess_{mlm,clm}_dataset`
(deliberately no `.cache()`), `estimate_clm_steps_per_epoch` (never roll a
local estimator), `create_warmup_lr_schedule`. Byte-level models (`vocab_size=256`)
cannot use the tiktoken pipeline — use `dl_techniques.datasets.byte_lm`.
`train_step` uses `tf.GradientTape` directly: TF backend only.

Fine-tuning a headless encoder checkpoint under a sentiment head (`src/train/{bert,
fnet,tree_transformer}/finetune.py`, `src/train/bert/wikipedia/finetune.py`) goes
through `train.common.nlp.build_sentiment_model_from_encoder(model_cls,
encoder_path, num_classes)`: it reads the archive's own config, rebuilds the class
with `include_top=True` and restores the backbone through the shared load rule, so
a missing, unreadable or incompatible checkpoint RAISES (there is no random-weights
fallback). `set_backbone_trainable(model, False)` leaves only `task_head` trainable
(stage 1); `set_backbone_trainable(model, True)` unfreezes every layer (stage 2),
by loop, because `model.trainable = True` after a per-layer freeze is a no-op. The
saved artifact is the model class itself with a head named `task_head`.

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

