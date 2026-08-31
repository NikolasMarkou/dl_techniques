# heads

Task-head layers for `dl_techniques`, organized by domain. Single merged
package consolidating the formerly-separate `nlp_heads/`, `vision_heads/`, and
`vlm_heads/` packages into `nlp/`, `vision/`, `vlm/` sub-packages, each keeping
its own `factory.py` + `task_types.py` + `README.md`. Relocated via `git mv`;
every layer-class name is preserved verbatim, so existing `.keras` checkpoints
stay loadable.

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
`dl_techniques.layers.heads.vision.factory>EnhancementHead` (verified 2026-08-29). The package
string is the defining module's dotted path; under `dl_techniques.models` the 12 family
directories and the 4 subfamily containers (`image_restoration`, `keypoints`,
`super_resolution`, `sam`) are stripped, but nothing is stripped under `layers/`.

**Never a bare `@keras.saving.register_keras_serializable()`**: its key `Custom>ClassName` is
independent of `__module__`, so two same-named classes claim one slot and the last import
silently wins. The helper additionally binds `Custom>ClassName` as a legacy alias to the same
object — which is why the checkpoints written before 2026-08-29 (and before this package merge)
still load. `MIGRATIONS.md` at the repo root is the record.

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
- `vision/factory.py` — `create_vision_head(task_type, ...)`,
  `create_enhancement_head(...)`, `create_multi_task_head(...)`,
  `HeadConfiguration`.
- `vlm/factory.py` — `create_vlm_head(task_config, ...)`,
  `create_multi_task_vlm_head(...)`.
- `task_types.py` — aggregator re-exporting `NLPTaskType`, `VisionTaskType`,
  `TaskType` (alias), `VLMTaskType`, plus configs/helpers (`NLPTaskConfig`,
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
- **`VisionTaskType` / `TaskType` alias.** Vision's generically-named `TaskType`
  was renamed to `VisionTaskType`; a module-level `TaskType = VisionTaskType`
  alias is kept (and re-exported) as a back-compat safety net (D-003).
- **`EnhancementHead` module-scope.** Lifted out of `create_enhancement_head()`
  (was a closure-local registered class). Class name kept EXACTLY
  `EnhancementHead` so `Custom>EnhancementHead` registration is unchanged. Do
  NOT re-nest it inside the factory.
  *Superseded 2026-08-29: the decorator is now
  `@register_dl_technique("dl_techniques.layers.heads.vision.factory")`, so the primary key is
  `dl_techniques.layers.heads.vision.factory>EnhancementHead`. **The rule below it is
  unchanged and still binding**: the legacy alias the helper mints is keyed on the bare class
  NAME, and `keras.saving.get_registered_object("Custom>EnhancementHead")` still returns this
  class (measured 2026-08-29). Renaming the class would drop that alias and break every
  pre-existing archive exactly as before, so the name stays EXACTLY `EnhancementHead`.*
- **No caller-dict mutation.** `MultiTaskHead._create_task_heads()` copies each
  per-task config dict before `pop('task_type')` (it used to mutate the caller's
  dict and break round-trips).
- **Serialization-stable class names.** All 22 names are verbatim. Sub-layers
  created in `__init__`/`build`, `keras.ops` only, `dl_techniques.utils.logger`
  only.
  *This bullet once also required "no `package=` on any decorator"; that half was
  superseded 2026-08-29 and has been dropped. Every decorator here is now
  `@register_dl_technique("dl_techniques.layers.heads.<domain>.factory")` and each class
  carries a package-qualified key. It cost no archive because the helper also mints the legacy
  `Custom>ClassName` alias (`MIGRATIONS.md`). The **class names** are still verbatim and must
  stay so: the alias is keyed on the bare name.*
- Public API: `from dl_techniques.layers.heads import create_head` (or per-domain
  `from dl_techniques.layers.heads.{nlp,vision,vlm} import ...`).
- Tests: `tests/test_layers/test_heads/`.
