# heads

Task-head layers for `dl_techniques`, organized by domain. Single merged
package consolidating the formerly-separate `nlp_heads/`, `vision_heads/`, and
`vlm_heads/` packages into `nlp/`, `vision/`, `vlm/` sub-packages, each keeping
its own `factory.py` + `task_types.py` + `README.md`. Relocated via `git mv`;
every layer-class name is preserved verbatim, so the registry keys in existing
`.keras` archives still resolve. Whether such an archive LOADS is a separate
question: the vision and VLM head layouts and output contracts changed on
2026-10-14/15, and `README.md` / `vision/README.md` / `vlm/README.md` say what
loads, what fails and what loads with changed semantics.

**Registration (current).** These classes register through
`@register_dl_technique("dl_techniques.layers.heads.<domain>.factory")`, from
`dl_techniques.utils.keras_registration`:

```python
import keras
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class EnhancementHead(BaseVisionHead):
    ...
```

`keras.saving.get_registered_name(EnhancementHead)` resolves to
`dl_techniques.layers.heads.vision.factory>EnhancementHead`. The package
string is the defining module's dotted path; under `dl_techniques.models` the 12 family
directories and the 4 subfamily containers (`image_restoration`, `keypoints`,
`super_resolution`, `sam`) are stripped, but nothing is stripped under `layers/`.

**Never a bare `@keras.saving.register_keras_serializable()`**: its key `Custom>ClassName` is
independent of `__module__`, so two same-named classes claim one slot and the last import
silently wins. The helper additionally binds `Custom>ClassName` as a legacy alias to the same
object — which is why the class keys of checkpoints written before the registration migration (and before
this package merge) still resolve.

## Layers (24 classes, grouped by domain)

> The count lives HERE and nowhere else in this file -- it was previously
> restated in two places and both said 21 while the list below already had 22
> (24 since 2026-10-15, when `SurfaceNormalsHead` and `OpticalFlowHead` were added).
> Re-derive rather than trust it:
>
> ```bash
> grep -hoP '^class \K\w+Head\b' \
>   src/dl_techniques/layers/heads/{nlp,vision,vlm}/factory.py | sort -u | wc -l
> ```
>
> The `\b` keeps a class whose name only contains `Head` out of the count: it
> excluded `NLPHeadConfiguration`, a lookup helper rather than a layer, until
> that helper was deleted on 2026-10-15 (D-064).
> Breakdown: 8 NLP + 10 vision + 6 VLM. The private `_DenseVectorHeadBase`
> (the shared body of the normals and flow heads) does not match the pattern.

**NLP** (`nlp/factory.py`) — `BaseNLPHead`, `TextClassificationHead`,
`TokenClassificationHead`, `QuestionAnsweringHead`, `TextSimilarityHead`,
`TextGenerationHead`, `MultipleChoiceHead`, `MultiTaskNLPHead`.

**Vision** (`vision/factory.py`) — `BaseVisionHead`, `DetectionHead`,
`SegmentationHead`, `DepthEstimationHead`, `SurfaceNormalsHead`,
`OpticalFlowHead`, `ClassificationHead`, `InstanceSegmentationHead`,
`EnhancementHead`, `MultiTaskHead`.

**VLM** (`vlm/factory.py`) — `BaseVLMHead`, `ImageCaptioningHead`, `VQAHead`,
`VisualGroundingHead`, `ImageTextMatchingHead`, `MultiTaskVLMHead`.

## Factory
- `factory.py` — `create_head(domain, *args, **kwargs)`: thin dispatch facade
  over the three single-head factories (`'nlp'|'vision'|'vlm'`); raises
  `ValueError` on an unknown domain. No signature unification — each domain
  keeps its native calling convention, forwarded verbatim (D-004). It has no
  production caller: models build their heads through
  `nlp/model_head.build_model_task_head` and `vision/factory.build_model_vision_head`.
  It is kept as the documented library entry point, with tests
  (plan-2026-10-14T062616-81938e81 D-067).
- `nlp/task_types.py` — `NLP_HEAD_DISPATCH`, the one task-type to head-class
  table (24 rows, head classes by name because `factory.py` imports this
  module); `get_head_class`, `NLPTaskConfig`'s checks, `get_output_types` and
  `model_head`'s generation set read it (D-064).
- `nlp/factory.py` — `create_nlp_head(task_config, input_dim, ...)`,
  `create_multi_task_nlp_head(...)`, `get_head_class(...)`.
- `nlp/model_head.py` — helpers for a language model that owns its task head
  (`include_top=True`): `validate_task_head_knobs(include_top, task_config,
  head_config_overrides)` (the two knob `ValueError`s, one source),
  `build_model_task_head(task_config, input_dim, ..., vocab_size=None)`
  (dict or `NLPTaskConfig`; refuses `MULTIPLE_CHOICE`; `default_pooling` only
  when the overrides set no `pooling_type`; head named `task_head`;
  `vocab_size` is the model's own vocabulary: a generation-type head
  (`TEXT_GENERATION`, `MASKED_LANGUAGE_MODELING`, `TEXT_COMPLETION`,
  `TEXT_SUMMARIZATION`) fills an unset `vocabulary_size` of a dict config from
  it and raises `ValueError` on a different explicit value, an `NLPTaskConfig`
  that leaves the 32000 default unset on a model of another vocabulary raises
  too, other task types ignore it, `None` checks nothing; every language model
  passes it),
  `task_config_to_dict(task_config, vocab_size=None)` for the model's
  `get_config`, and
  `finalize_head_outputs(...)` (drop `probabilities`/`predictions` beside
  `logits`, unwrap a single-key dict to a bare tensor), and
  `head_attention_mask(input_ids, attention_mask, pad_token_id, dtype)` (the
  caller's mask, else `input_ids != pad_token_id`, else all ones when `pad_token_id` is `None`,
  cast to `dtype`; for causal
  backbones that run without a mask but pool `'last'`), and
  `head_output_shape(task_head, input_shape, hidden_size)` plus
  `ids_shape_of(input_shape)` (the shapes `finalize_head_outputs` would give,
  from shapes alone, so a headed model's `compute_output_shape` reports what
  `call` returns). All eight helpers, `TASK_HEAD_NAME` (`"task_head"`, the
  head's layer name) included, are exported from `nlp/__init__.py` (20 names),
  and that is the one surface the 15 language models and `train/common/nlp.py`
  import them from; nothing in `src/` imports `nlp.model_head` directly
  (`tests/test_models/test_the_language_models_import_one_head_surface.py`).
  **Migration of stored archives (since 2026-10-11).** The vocabulary check
  also runs when a language model is rebuilt from its config, so a `.keras`
  archive saved earlier with a generation-type `task_config` whose stored
  `vocabulary_size` is 32000 (the `NLPTaskConfig` default) on a model of
  another vocabulary now fails `load_model` / `from_config` with `ValueError`
  (`The 'text_generation' head would emit logits over vocabulary_size=32000,
  but the model's vocabulary is ...`). Fix the archive: set the stored
  `task_config.vocabulary_size` in its `config.json` to the model's
  `vocab_size` (or delete the key, and the model's vocabulary fills it), or
  rebuild the model with a corrected `task_config` and save it again.
- `vision/factory.py` — `VISION_HEAD_DISPATCH` (the one task-type to
  head-class table every vision helper reads), `create_vision_head(task_type,
  ...)`, `create_enhancement_head(...)`, `create_multi_task_head(...)`,
  `HeadConfiguration` (presets for the 10 task types with a head; the rest
  raise), `merge_head_config_overrides(...)` and
  `build_model_vision_head(task_type, head_config, head_config_overrides)`,
  the one call every preset-based vision head is built with (21 sites;
  DETR, MambaLCT and SuperPoint call `create_vision_head` with full configs).
- `vlm/factory.py` — `VLM_HEAD_DISPATCH` (the one task-type to head-class
  table, 4 rows), `create_vlm_head(task_config, ...)`,
  `create_multi_task_vlm_head(...)`. **No model, trainer or application
  constructs a VLM head today**; the callers are `layers/heads` and the tests
  (`vlm/README.md` § 1 has the grep).
- `validation.py` — the two checks the domains share: `positive_int(value,
  name, owner)` (every vision count, `VQAHead.num_heads`, the
  `VLMTaskConfig` and `NLPTaskConfig` counts, the NLP heads' `input_dim` and
  `intermediate_size`) and `static_rank(x, name, owner)` (a head that
  branches on rank reads it once, from the input's static shape, and refuses
  an unknown rank by name; the VLM and NLP heads use it, and
  `test_the_heads_read_a_static_rank.py` keeps `len(ops.shape(...))` out of
  all three factories).
- `task_types.py` — aggregator re-exporting `NLPTaskType`, `VisionTaskType`,
  `VLMTaskType`, plus configs/helpers (`NLPTaskConfig`,
  `VLMTaskConfig`, `TaskConfiguration`, `parse_task_list`,
  `CommonTaskConfigurations`). Multi-task heads keep domain-specific
  `task_configs` shapes and are NOT routed through `create_head`.

## Conventions
- **No silent fallback in task dispatch.** `get_head_class()` in `nlp/` and `vlm/` must
  **raise `ValueError`** for a task type with no implemented head — never substitute a
  default. Both used to end with `head_mapping.get(task_type, <SomeDefault>)`, which meant:
  - **NLP**: 13 of 37 `NLPTaskType` members (`MACHINE_TRANSLATION`, `DIALOGUE_GENERATION`,
    `RELATION_EXTRACTION`, `DEPENDENCY_PARSING`, `COREFERENCE_RESOLUTION`, …) silently
    returned a `TextClassificationHead`. It builds, trains and emits plausible numbers — a
    translation task quietly became a classifier, and nothing ever failed.
  - **VLM**: 41 of 47 `VLMTaskType` members silently returned a bare `BaseVLMHead`, which
    has **no `call()`** — so the factory returned an object that constructed fine and died
    on first use with `NotImplementedError`, naming the base class rather than the task.
  `vision/` already got this right (`raise ValueError(f"Unsupported task type: …")`); the
  other two now match it. **`BaseVLMHead` is not a usable head**: it refuses direct
  construction with `TypeError`, and `VLM_HEAD_DISPATCH` holds no placeholder row for it.
  Only 4 VLM tasks have a real head today (captioning, VQA, visual grounding, image-text
  matching). To add one: implement the head and add it to the table — do not restore the
  fallback. Pinned by `tests/test_layers/test_heads/test_head_dispatch_no_silent_fallback.py`
  and `test_the_vlm_dispatch_names_exactly_four_heads.py`.
- **VLM padding masks.** A VLM head reads optional `vision_mask` / `text_mask` /
  `question_mask` keys (`(batch, length)`, 1 = real) next to the features they describe, and
  every pooling and attention over a sequence honours them. The means and the max go through
  `SequencePooling`, which resolves a kernel initializer only for a strategy that owns a
  kernel, so a mean/max pooler draws no random seed (D-055, D-087); no mask means
  bit-identical outputs and fixed-seed weights to the pre-mask heads.
  Pinned by `test_the_vlm_heads_honour_padding_masks.py`.
- **`VLMTaskConfig` holds only read fields.** Six fields; a head refuses a set `vocab_size`
  (captioning only) or `num_classes` (VQA only) it does not read, and `__post_init__` fills in
  nothing (D-058). Pinned by `test_the_vlm_task_config_holds_only_live_fields.py`.
- **`NLPTaskConfig` holds only read fields** (D-064). Seven fields; `num_labels`, `max_length`,
  `loss_weight`, `label_smoothing`, `use_attention_pooling`, `beam_size` and `temperature` are
  deleted. `__post_init__` parses a string `task_type`, validates every value, requires
  `num_classes` for each task type `NLP_HEAD_DISPATCH` routes to a classification or token head,
  and fills `vocabulary_size=32000` only for one routed to `TextGenerationHead`. Pinned by
  `test_the_nlp_task_config_holds_only_live_fields.py`.
- **NLP padding masks** (D-063). `TextSimilarityHead` (both input forms; each side of a pair is a
  dict with its own `attention_mask`, as a tuple or a list) and `MultipleChoiceHead` (a
  `(batch, num_choices, seq)` mask, folded with the choices) pool under the caller's mask; a mask
  for an already pooled input is refused. Pinned by `test_the_nlp_heads_honour_padding_masks.py`.
  The token and QA heads' task attention attends under the same mask (D-072,
  `BaseNLPHead._apply_task_attention`); an attention type that takes no `attention_mask` refuses
  one. Pinned by `test_the_nlp_task_attention_ignores_padding.py`.
- **Vision attention layout** (D-071). A vision head's attention stage runs in the layout its
  type takes (`_SPATIAL_ATTENTION_TYPES` in `vision/factory.py` lists the map-only ones); a map is
  flattened for a sequence type and a sequence head's tokens are placed on `patch_grid_size` for a
  spatial type. Pinned, with the table re-measured, by
  `test_a_vision_head_runs_attention_in_its_own_layout.py`.
- **`sequence_pooling` reuse (NLP).** `BaseNLPHead` pooling for the four strategies
  in `_DELEGATED_POOLING_STRATEGIES` -- `cls`/`mean`/`max`/`last` -- delegates to the shared `dl_techniques.layers.sequence_pooling.SequencePooling`
  layer (built in `__init__`/`build`, Golden Rule). The `attention` strategy
  stays inline (`Dense(1, tanh)` direct-score) — `SequencePooling('attention')`
  uses a different `AttentionPooling` mechanism + weight set, so delegating it
  would change values AND break checkpoint serialization. This is the **partial
  delegation** (D-002); do NOT route the `attention` branch through
  `SequencePooling`.
- **`VisionTaskType` has no alias.** Vision's generically-named task enum was
  renamed to `VisionTaskType`; the back-compat alias for the old name was
  deleted on 2026-10-15 (no in-tree importer, no legacy names per the owner).
  Do not re-add it; `tests/test_layers/test_heads/test_the_vision_task_enum_has_no_alias.py`
  pins its absence.
- **Logits out.** No head applies an activation to a class-like score: every
  classification, detection, segmentation, instance-mask, VQA, captioning,
  grounding `'confidence'` and matching `'match_score'` output is a
  pre-activation logit. The only squashed outputs are range maps for
  regression targets: the progressive `DepthEstimationHead` sigmoid and the
  grounding `'bbox'` sigmoid. Do not add an activation to a class-like
  output; the consumer's loss takes `from_logits=True`. `README.md` §
  Output contract is the authoritative list.
- **A head owns what its `call` runs, and nothing else.** A sub-layer that
  `call` never reaches is not created, and an argument `call` can never read
  is refused with `ValueError`: the vision heads (D-020: no common
  norm/Dropout, `ClassificationHead` refuses `use_ffn`, `EnhancementHead`
  refuses attention/FFN/dropout), `DepthEstimationHead`'s two decoders, each of
  which refuses the other's arguments (D-030), the arguments no layer of a
  given mode reads (D-074: `DetectionHead(use_regression=False)` refuses
  `bbox_dims`, the DPT decoder and the explicit `SegmentationHead` refuse
  `hidden_dim` unless the attention or FFN stage reads it; each stores
  `None`), and the VLM heads (D-032:
  `BaseVLMHead` owns only `fusion`; the post-fusion norm and FFN and their four
  arguments live on `ImageTextMatchingHead`; no
  `vision_dim`/`text_dim`/`activation_type`), and the NLP heads (D-062: a head
  refuses `use_task_attention`, `use_ffn` or `use_pooling` for a stage its
  `call` never runs, e.g. `TextClassificationHead` pools to rank 2 first, so it
  refuses task attention; each stage is built at the width `call` feeds it).
  The `BaseVisionHead` attention/FFN settings follow the same rule (D-210):
  `attention_type`, `ffn_type` and `ffn_expansion_factor` default to `None`,
  resolve to `'multi_head'`/`'mlp'`/4 when their stage is on, and raise when
  it is off; the presets and `build_model_vision_head` drop a preset knob
  whose stage ends up off. The NLP `pooling_type` of a head that never pools
  (validated, not read; a language model passes its default pooling to every
  head) is the flag-gated exception: accepted, read only when the stage runs. Check it by calling: count each
  sub-layer's `call` over one forward
  (`test_vlm.py::TestEveryVLMHeadOwnsOnlyWhatItRuns`). Do not check it by
  weights: a sub-layer created and never called is never built and holds none.
- **Every head round-trips by value.** `tests/test_layers/test_heads/test_every_vision_head_survives_a_keras_round_trip.py`
  saves and reloads every vision head variant and compares every weight by
  path; `test_vlm.py::TestVLMHeadFamilyExplicitBuild` does the same for the
  VLM heads by output, and `test_every_nlp_head_survives_a_keras_round_trip.py`
  for the NLP heads by both (D-065: every constructor field in `get_config`, no
  `from_config` override, so a caller's config dict is never modified; the
  `task_heads`/`task_projections` dicts and the `similarity_layers` list were
  measured to round-trip). Keep sub-layers in flat lists or named attributes (Keras does not save
  a layer held only in a nested list), name every sub-head, and keep
  `get_config` JSON-serializable (no `Enum` members).
- **`EnhancementHead` is module-scope**, not a closure-local class inside
  `create_enhancement_head()`. Do NOT re-nest it. Its primary key is
  `dl_techniques.layers.heads.vision.factory>EnhancementHead`, and the legacy alias the helper
  mints is keyed on the bare class NAME — `keras.saving.get_registered_object("Custom>EnhancementHead")`
  returns this class. Renaming it would drop that alias and break every pre-existing archive, so
  the name stays EXACTLY `EnhancementHead`.
- **No caller-dict mutation, no shared config objects.** `MultiTaskHead._create_task_heads()` copies each
  per-task config dict before `pop('task_type')` (it used to mutate the caller's
  dict and break round-trips). The NLP heads copy every `NLPTaskConfig` they are
  given (`NLPTaskConfig.coerce`), so editing a config after construction changes
  neither a head nor its `get_config` (D-065).
- **Serialization-stable class names.** Every decorator here is
  `@register_dl_technique("dl_techniques.layers.heads.<domain>.factory")`, so each class carries a
  package-qualified key plus the legacy `Custom>ClassName` alias. The class names must stay
  verbatim, because that alias is keyed on the bare name. Sub-layers are created in
  `__init__`/`build`; `keras.ops` only; `dl_techniques.utils.logger` only.
- Public API: `from dl_techniques.layers.heads import create_head` (or per-domain
  `from dl_techniques.layers.heads.{nlp,vision,vlm} import ...`).
- Tests: `tests/test_layers/test_heads/`.
