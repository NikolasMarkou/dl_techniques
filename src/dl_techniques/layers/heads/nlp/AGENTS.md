# heads/nlp

NLP task heads: layers that turn a foundation model's hidden states into task predictions, plus the
config types and factories that pick one. Heads take hidden states, not tokens, so one head works on
BERT, GPT or T5 output. The parent `../AGENTS.md` owns the cross-domain rules; this file covers what
is specific to `nlp/`. `README.md` is the user-facing tour.

## Layout

One module per head, a shared base, and thin factories. Dependencies run downward only.

| File | Holds |
|---|---|
| `task_types.py` | `NLPTaskType`, `NLPTaskConfig` (`coerce`, `to_dict`), `NLP_HEAD_DISPATCH` (task type to head class **name**, 24 rows), presets. Imports no head |
| `utils.py` | `_split_inputs`, `_is_pair`, `_apply_per_row`, the pooling-strategy tuples, `NLP_STAGE_KNOB_DEFAULTS` |
| `base.py` | `BaseNLPHead`: validation, `norm`, `dropout`, optional pooler / `intermediate` / `task_attention` / `ffn`, `_pool_input`, `_pool_sequence`, `_apply_task_attention`. No `call` |
| `text_classification.py`, `token_classification.py`, `question_answering.py`, `text_similarity.py`, `text_generation.py`, `multiple_choice.py` | one leaf head each, subclassing `BaseNLPHead` |
| `multitask.py` | `MultiTaskNLPHead`: owns one sub-head per task, routes by `task_name` |
| `dispatch.py` | `_HEAD_CLASSES` (class name to class) and `get_head_class(task_type)`, which resolves through `../validation.lookup_head` |
| `factory.py` | `create_nlp_head(task_config, input_dim, **kwargs)`, `create_multi_task_nlp_head(task_configs, input_dim, **kwargs)` |
| `model_head.py` | helpers for a language model that owns its head (`build_model_task_head`, `finalize_head_outputs`, ...); see `../AGENTS.md` |
| `__init__.py` | the curated public surface (20 names, `__all__`) |

Import order is `task_types` / `utils` -> `base` -> leaf heads -> `dispatch` -> `multitask` ->
`factory`. `task_types.py` cannot import a head, which is why the dispatch table holds class names.
`factory.py` was one 2800-line file until 2026-10-27; the code moved verbatim.

## Rules specific to this package

- **Registration follows the defining module**: `@register_dl_technique("dl_techniques.layers.heads.nlp.<module>")`,
  e.g. `...nlp.base`, `...nlp.text_classification`. Move a class and you change its key.
  `tests/test_layers/test_heads/test_the_nlp_head_classes_register_under_their_own_module.py` pins it.
  Archives saved before the split (`...nlp.factory>X`) do not load.
- **Import classes from the package or their module, never from `factory`.** `factory.py` defines
  only the two `create_*` functions and happens to bind `get_head_class` because `create_nlp_head`
  uses it; the home of `get_head_class` is `dispatch.py`.
- **A head owns what its `call` runs.** `_POOLS`, `_APPLIES_TASK_ATTENTION` and `_APPLIES_FFN` on each
  subclass say which stages it runs, and `__init__` refuses a flag for a stage it never runs (D-062).
  A new head sets all three honestly.
- **Stage knobs** (`attention_type`, `ffn_type`, `ffn_expansion_factor`) default to `None`, resolve to
  `'multi_head'` / `'mlp'` / 4 with their stage on, and raise with it off (D-211), via
  `../validation.resolve_stage_knobs`.
- **Pooling.** `cls`, `mean`, `max`, `last` go through the shared `SequencePooling`; `'attention'` stays
  inline as `Dense(1, tanh)` because `SequencePooling('attention')` is a different mechanism with
  different weights (D-002). Causal backbones need `pooling_type='last'` (D-023).
- **Masks.** A dict input's `attention_mask` reaches pooling and task attention. Rank-2 versus rank-3
  input goes through `../validation.pool_or_passthrough`. Masked scores use `../masking.fill_masked`
  (a `keras.ops.where` selection), never `scores + (1 - mask) * -1e9` (fp16 `0 * -inf` is NaN on the
  kept positions). `tests/test_the_mask_sentinel_population_is_closed.py`
  censuses this across `src/`.
- **Configs are copied** (`NLPTaskConfig.coerce`), so editing a caller's config after construction
  changes neither the head nor its `get_config` (D-065). `get_config` writes `task_config` as a dict;
  there is no `from_config` override.
- **`create_multi_task_nlp_head`** keys a list or tuple through `../validation.key_by_name`.
- **No silent dispatch fallback.** A task type with no row raises `ValueError`; do not restore a default.
  To add a task: implement or choose a head, add its row to `NLP_HEAD_DISPATCH` and its class to
  `_HEAD_CLASSES`.
- Logits out: heads return pre-activation `logits`; `probabilities` / `predictions` are derived entries
  that `finalize_head_outputs` can drop.
- `DECISION ...` anchor comments record why a non-obvious choice was made. Supersede them with a dated
  note; never delete them.

## Adding a head

1. New module `nlp/<name>.py`, subclass `BaseNLPHead`, `@register_dl_technique("dl_techniques.layers.heads.nlp.<name>")`.
2. Set the three `_POOLS` / `_APPLIES_*` class flags; build the output layer at `self._output_width()`.
3. Implement `build` (call `super().build` first, then the output layer), `compute_output_shape`
   (works unbuilt, same keys as `call`), `call`.
4. Add the class to `dispatch._HEAD_CLASSES`, a row to `NLP_HEAD_DISPATCH`, an export in `__init__.py`.
5. Tests in `tests/test_layers/test_heads/`: a `.keras` round trip by value (extend
   `test_every_nlp_head_survives_a_keras_round_trip.py`), shapes (`test_every_nlp_head_reports_the_shapes_it_returns.py`),
   padding masks, and a knob test per constructor argument.

## Tests

`tests/test_layers/test_heads/test_nlp*.py` plus the `test_the_nlp_*` and `test_every_nlp_*` files.
Two repo-level pins name files here: `tests/test_layers/test_norms/test_the_norm_factory_family_is_pinned.py`
(the one dynamic `create_normalization_layer` site is `nlp/base.py`) and
`tests/test_the_head_factories_and_finetune_scripts_bind_no_dead_names.py` (lists every `nlp/*.py`).
Scope runs to `tests/test_layers/test_heads/`; do not run GPU jobs in parallel.
