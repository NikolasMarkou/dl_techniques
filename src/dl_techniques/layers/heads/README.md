# Task Heads (`dl_techniques.layers.heads`)

A single, model-agnostic system for attaching task-specific output **heads** to
any foundation backbone — NLP, vision, or vision-language. This package merges
the formerly-separate `nlp_heads`, `vision_heads`, and `vlm_heads` packages into
three domain sub-packages under one import surface.

**Core philosophy:** decouple the *encoder* (foundation model) from the
*decoder* (task head). The backbone produces rich contextual features; the head
transforms them into task-specific predictions (logits, scores, masks,
embeddings, generated tokens). The same backbone can drive many heads.

## Layout

```
heads/
├── __init__.py        # union re-export of all 3 domains + create_head
├── factory.py         # create_head(domain, ...) dispatch facade
├── task_types.py      # aggregator: NLP/Vision/VLM task-type enums + configs
├── nlp/               # NLP heads     (factory.py, task_types.py, README.md)
├── vision/            # vision heads  (factory.py, task_types.py, README.md)
└── vlm/               # VLM heads     (factory.py, task_types.py, README.md)
```

See the per-domain READMEs for the full input/output contracts and head
catalogues: [`nlp/README.md`](nlp/README.md),
[`vision/README.md`](vision/README.md), [`vlm/README.md`](vlm/README.md).

## Quick start

```python
from dl_techniques.layers.heads import create_head
from dl_techniques.layers.heads.task_types import (
    NLPTaskConfig, NLPTaskType, VisionTaskType, VLMTaskConfig, VLMTaskType,
)

# NLP: task_config + input_dim
nlp = create_head(
    "nlp",
    task_config=NLPTaskConfig(name="sentiment",
                              task_type=NLPTaskType.SENTIMENT_ANALYSIS,
                              num_classes=3),
    input_dim=768,
)

# Vision: task_type (+ head kwargs)
vis = create_head("vision", VisionTaskType.CLASSIFICATION, num_classes=1000)

# VLM: task_config (+ vision_dim/text_dim/...)
vlm = create_head(
    "vlm",
    task_config=VLMTaskConfig(name="caption",
                              task_type=VLMTaskType.IMAGE_CAPTIONING,
                              vocab_size=50000),
    vision_dim=768, text_dim=768,
)
```

`create_head(domain, ...)` is a **thin dispatcher**: each domain keeps its own
native calling convention and the remaining args are forwarded verbatim (no
signature unification). It covers the single-head factories only — build
multi-task heads via the domain functions directly
(`create_multi_task_nlp_head`, `create_multi_task_head`,
`create_multi_task_vlm_head`).

You can also import per domain:

```python
from dl_techniques.layers.heads.nlp import create_nlp_head, TextClassificationHead
from dl_techniques.layers.heads.vision import create_vision_head, VisionTaskType
from dl_techniques.layers.heads.vlm import create_vlm_head, VLMTaskConfig
```

## Domain summary

| Domain | Heads | Task-type vocabulary |
|--------|-------|----------------------|
| **nlp** | `TextClassificationHead`, `TokenClassificationHead`, `QuestionAnsweringHead`, `TextSimilarityHead`, `TextGenerationHead`, `MultipleChoiceHead`, `MultiTaskNLPHead` | `NLPTaskType` / `NLPTaskConfig` |
| **vision** | `DetectionHead`, `SegmentationHead`, `DepthEstimationHead`, `ClassificationHead`, `InstanceSegmentationHead`, `EnhancementHead`, `MultiTaskHead` | `VisionTaskType` (`TaskType` alias) / `TaskConfiguration` |
| **vlm** | `ImageCaptioningHead`, `VQAHead`, `VisualGroundingHead`, `ImageTextMatchingHead`, `MultiTaskVLMHead` | `VLMTaskType` / `VLMTaskConfig` |

## Notes

- **Serialization-stable.** All 21 layer-class names are preserved verbatim; the
  package was relocated via `git mv`, so existing `.keras` checkpoints stay
  loadable (bare `@register_keras_serializable()` registers as
  `Custom>ClassName`, independent of module path).
- **NLP pooling reuse.** `BaseNLPHead` delegates `cls`/`mean`/`max` pooling to
  the shared `SequencePooling` layer; the learnable `attention` pooling stays
  inline (a distinct mechanism + weight set). See `CLAUDE.md` (D-002).
- **`VisionTaskType` / `TaskType` alias.** Vision's task enum was renamed to
  `VisionTaskType`; a `TaskType` alias is retained for back-compat.
