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

# VLM: task_config (+ head kwargs; every feature width comes from the inputs)
vlm = create_head(
    "vlm",
    task_config=VLMTaskConfig(name="caption",
                              task_type=VLMTaskType.IMAGE_CAPTIONING,
                              vocab_size=50000),
    num_layers=2,
)
```

`create_head(domain, ...)` is a **thin dispatcher**: each domain keeps its own
native calling convention and the remaining args are forwarded verbatim (no
signature unification). It covers the single-head factories only — build
multi-task heads via the domain functions directly
(`create_multi_task_nlp_head`, `create_multi_task_head`,
`create_multi_task_vlm_head`). No model calls it: the language models build
their heads with `nlp.model_head.build_model_task_head` and the vision models
with `vision.factory.build_model_vision_head`. It stays as the library entry
point for code outside this repository's models.

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
| **vision** | `DetectionHead`, `SegmentationHead`, `DepthEstimationHead`, `SurfaceNormalsHead`, `OpticalFlowHead`, `ClassificationHead`, `InstanceSegmentationHead`, `EnhancementHead`, `MultiTaskHead` | `VisionTaskType` / `TaskConfiguration` |
| **vlm** | `ImageCaptioningHead`, `VQAHead`, `VisualGroundingHead`, `ImageTextMatchingHead`, `MultiTaskVLMHead` | `VLMTaskType` / `VLMTaskConfig` |

**Consumers.** The vision heads are built by the `models/vision` packages and FastVLM, and the NLP
heads by the headed language models. **No model, trainer or application constructs a VLM head
today**; the VLM heads are a library whose only callers are this package and its tests
(`vlm/README.md` § 1 has the grep that says so).

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
its `dpt_output_activation`, `'linear'` by default; DepthAnything passes `'relu'`, the reference
head's final ReLU). Surface normals are L2-normalized to unit length and optical flow is linear and
signed. The per-domain READMEs give each head's keys.

Every head owns only the sub-layers its `call` runs. Arguments that a head's `call`
can never read are refused with `ValueError`: the VLM heads take no `vision_dim`/`text_dim` (widths
come from the inputs) and no `activation_type`, the post-fusion options belong to
`ImageTextMatchingHead` alone, `DepthEstimationHead`'s two decoders each refuse the other's
arguments, and an NLP head refuses `use_task_attention`, `use_ffn` or `use_pooling` for a stage
its `call` never runs (`nlp/README.md` § 4.1 has the table). The vision heads' attention and FFN settings (`attention_type`, `ffn_type`, ...) are
still accepted when `use_attention`/`use_ffn` is off, and are read only when it is on. The
exception is `dropout_rate` on the depth, normals and flow heads, whose decoders take no dropout:
there it defaults to 0.0 and a positive rate is refused unless attention or the FFN is on.

## Notes

- **Serialization-stable names.** All 22 layer-class names of the merge are preserved verbatim (two vision heads were added on 2026-10-15); the
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
- **VLM padding masks.** The VLM heads take optional `vision_mask` / `text_mask` /
  `question_mask` keep masks in their input dict and honour them in every pooling and
  attention; their means and max also go through `SequencePooling`. See `vlm/README.md` § 2.
- **NLP padding masks.** Every NLP head that pools reads `'attention_mask'` from its input
  dict: classification, both forms of `TextSimilarityHead` (each side of a pair carries its
  own) and `MultipleChoiceHead` (a `(batch, num_choices, seq)` mask). See `nlp/README.md` § 3.
- **`VisionTaskType` has no alias.** Vision's task enum was renamed from a generic name to
  `VisionTaskType`; the old name is gone, so import `VisionTaskType`.
