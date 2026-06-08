# heads

Task-head layers for `dl_techniques`, organized by domain. Single merged
package consolidating the formerly-separate `nlp_heads/`, `vision_heads/`, and
`vlm_heads/` packages into `nlp/`, `vision/`, `vlm/` sub-packages, each keeping
its own `factory.py` + `task_types.py` + `README.md`. Relocated via `git mv`;
all 21 layer-class names are preserved verbatim, so existing `.keras`
checkpoints stay loadable (bare `@register_keras_serializable()` registers as
`Custom>ClassName`, independent of `__module__`).

## Layers (21 classes, grouped by domain)

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
- **`sequence_pooling` reuse (NLP).** `BaseNLPHead` pooling for `cls`/`mean`/`max`
  delegates to the shared `dl_techniques.layers.sequence_pooling.SequencePooling`
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
- **No caller-dict mutation.** `MultiTaskHead._create_task_heads()` copies each
  per-task config dict before `pop('task_type')` (it used to mutate the caller's
  dict and break round-trips).
- **Serialization-stable class names.** All 21 names are verbatim; no `package=`
  added to any decorator. Sub-layers created in `__init__`/`build`, `keras.ops`
  only, `dl_techniques.utils.logger` only.
- Public API: `from dl_techniques.layers.heads import create_head` (or per-domain
  `from dl_techniques.layers.heads.{nlp,vision,vlm} import ...`).
- Tests: `tests/test_layers/test_heads/`.
