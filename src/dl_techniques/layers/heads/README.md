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
| **vision** | `DetectionHead`, `SegmentationHead`, `DepthEstimationHead`, `ClassificationHead`, `InstanceSegmentationHead`, `EnhancementHead`, `MultiTaskHead` | `VisionTaskType` / `TaskConfiguration` |
| **vlm** | `ImageCaptioningHead`, `VQAHead`, `VisualGroundingHead`, `ImageTextMatchingHead`, `MultiTaskVLMHead` | `VLMTaskType` / `VLMTaskConfig` |

## Output contract: logits

No head in this package applies an activation to a class-like score. Every such output is a
pre-activation logit: classification `'logits'` (NLP and vision), the NLP token, span,
multiple-choice and generation logits, detection `'classifications'`, the segmentation map,
`'instance_masks'`, VQA `'answer_logits'`, captioning `'logits'`, grounding `'confidence'` and
matching `'match_score'`. Compile with `from_logits=True`, or apply `softmax`/`sigmoid` yourself.
Where a head also returns `'probabilities'` (NLP and vision classification, NLP multiple choice), it
is `softmax('logits')`, computed in the head's compute dtype. Two outputs are squashed by design,
because they are range maps for regression targets, not class scores: the progressive
`DepthEstimationHead`'s sigmoid and the grounding `'bbox'` sigmoid (the DPT depth decoder applies
its `dpt_output_activation`, `'linear'` by default). The per-domain READMEs give each head's keys.

## Notes

- **Serialization-stable names.** All 22 layer-class names are preserved verbatim; the
  package was relocated via `git mv`, so the registry keys of existing `.keras` checkpoints still
  resolve. A name resolving is not the archive loading: the vision and VLM head layouts changed on
  2026-10-14/15 (see the vision and VLM READMEs), and archives saved before then either fail to
  load or load with logits where they held probabilities. The
  classes register through `@register_dl_technique("dl_techniques.layers.heads.<domain>.factory")`
  (from `dl_techniques.utils.keras_registration`), so
  `keras.saving.get_registered_name(EnhancementHead)` is
  `dl_techniques.layers.heads.vision.factory>EnhancementHead`. The
  helper additionally binds the legacy `Custom>ClassName` as an alias to the same object —
  and that alias is keyed on the **bare class name**, which is why the names must stay
  verbatim and why a `git mv` costs nothing. See `AGENTS.md` in this directory.
- **NLP pooling reuse.** `BaseNLPHead` delegates `cls`/`mean`/`max`/`last` pooling to
  the shared `SequencePooling` layer; the learnable `attention` pooling stays
  inline (a distinct mechanism + weight set). See `AGENTS.md`.
- **`VisionTaskType` has no alias.** Vision's task enum was renamed from a generic name to
  `VisionTaskType`; the old name is gone, so import `VisionTaskType`.
