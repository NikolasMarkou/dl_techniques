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

## Layers (22 classes, grouped by domain)

> The count lives HERE and nowhere else in this file -- it was previously
> restated in two places and both said 21 while the list below already had 22.
> Re-derive rather than trust it:
>
> ```bash
> grep -hoP '^class \K\w+Head\b' \
>   src/dl_techniques/layers/heads/{nlp,vision,vlm}/factory.py | sort -u | wc -l
> ```
>
> The `\b` matters: without it the pattern also matches `NLPHead` inside
> `NLPHeadConfiguration`, which is a config dataclass, not a layer, and yields 23.
> Breakdown: 8 NLP + 8 vision + 6 VLM.

**NLP** (`nlp/factory.py`) — `BaseNLPHead`, `TextClassificationHead`,
`TokenClassificationHead`, `QuestionAnsweringHead`, `TextSimilarityHead`,
`TextGenerationHead`, `MultipleChoiceHead`, `MultiTaskNLPHead`.

**Vision** (`vision/factory.py`) — `BaseVisionHead`, `DetectionHead`,
`SegmentationHead`, `DepthEstimationHead`, `ClassificationHead`,
`InstanceSegmentationHead`, `EnhancementHead`, `MultiTaskHead`.

**VLM** (`vlm/factory.py`) — `BaseVLMHead`, `ImageCaptioningHead`, `VQAHead`,
`VisualGroundingHead`, `ImageTextMatchingHead`, `MultiTaskVLMHead`.

## Factory
- `factory.py` — `create_head(domain, *args, **kwargs)`: thin dispatch facade
  over the three single-head factories (`'nlp'|'vision'|'vlm'`); raises
  `ValueError` on an unknown domain. No signature unification — each domain
  keeps its native calling convention, forwarded verbatim (D-004).
- `nlp/factory.py` — `create_nlp_head(task_config, input_dim, ...)`,
  `create_multi_task_nlp_head(...)`, `NLPHeadConfiguration`.
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
  `call` returns). Both shape helpers are shared by the 15 language models but
  not exported from `nlp/__init__.py`; import them from `nlp.model_head`.
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
- `vision/factory.py` — `create_vision_head(task_type, ...)`,
  `create_enhancement_head(...)`, `create_multi_task_head(...)`,
  `HeadConfiguration`.
- `vlm/factory.py` — `create_vlm_head(task_config, ...)`,
  `create_multi_task_vlm_head(...)`.
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
  other two now match it. **`BaseVLMHead` is not a usable head** and must never be returned
  from dispatch, including for the entries once commented `# Placeholder`. Only 4 VLM tasks
  have a real head today (captioning, VQA, visual grounding, image-text matching). To add
  one: implement the head and map it — do not restore the fallback.
  Pinned by `tests/test_layers/test_heads/test_head_dispatch_no_silent_fallback.py`.
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
  which refuses the other's arguments (D-030), and the VLM heads (D-032:
  `BaseVLMHead` owns only `fusion`; the post-fusion norm and FFN and their four
  arguments live on `ImageTextMatchingHead`; no
  `vision_dim`/`text_dim`/`activation_type`). The `BaseVisionHead`
  attention/FFN settings are the one flag-gated exception: accepted with
  `use_attention`/`use_ffn` off, read only when on. Check it by calling: count each
  sub-layer's `call` over one forward
  (`test_vlm.py::TestEveryVLMHeadOwnsOnlyWhatItRuns`). Do not check it by
  weights: a sub-layer created and never called is never built and holds none.
- **The vision and VLM heads round-trip by value.** `tests/test_layers/test_heads/test_every_vision_head_survives_a_keras_round_trip.py`
  saves and reloads every vision head variant and compares every weight by
  path; `test_vlm.py::TestVLMHeadFamilyExplicitBuild` does the same for the
  VLM heads by output. Keep sub-layers in flat lists or named attributes (Keras does not save
  a layer held only in a nested list), name every sub-head, and keep
  `get_config` JSON-serializable (no `Enum` members).
- **`EnhancementHead` is module-scope**, not a closure-local class inside
  `create_enhancement_head()`. Do NOT re-nest it. Its primary key is
  `dl_techniques.layers.heads.vision.factory>EnhancementHead`, and the legacy alias the helper
  mints is keyed on the bare class NAME — `keras.saving.get_registered_object("Custom>EnhancementHead")`
  returns this class. Renaming it would drop that alias and break every pre-existing archive, so
  the name stays EXACTLY `EnhancementHead`.
- **No caller-dict mutation.** `MultiTaskHead._create_task_heads()` copies each
  per-task config dict before `pop('task_type')` (it used to mutate the caller's
  dict and break round-trips).
- **Serialization-stable class names.** Every decorator here is
  `@register_dl_technique("dl_techniques.layers.heads.<domain>.factory")`, so each class carries a
  package-qualified key plus the legacy `Custom>ClassName` alias. The class names must stay
  verbatim, because that alias is keyed on the bare name. Sub-layers are created in
  `__init__`/`build`; `keras.ops` only; `dl_techniques.utils.logger` only.
- Public API: `from dl_techniques.layers.heads import create_head` (or per-domain
  `from dl_techniques.layers.heads.{nlp,vision,vlm} import ...`).
- Tests: `tests/test_layers/test_heads/`.
