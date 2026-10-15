# Heads Review — `src/dl_techniques/layers/heads/` (2026-10-15)

Method: static analysis + targeted CPU-only runtime verification. No code changed during review. All 626 head tests collected; `test_nlp.py` + `test_vision.py` + `test_vlm.py` (261 tests) pass.

Package: 22 head layers across NLP (8), Vision (8), VLM (6), plus `factory.py` facade (`create_head`) and `task_types.py` aggregator.

## Executive summary

Architecture is sound: no-silent-fallback dispatch, logits-out convention, serialization guards. 7 high-value bugs + multiple gaps affect correctness, usability, API consistency.

## 1. NLP (`nlp/factory.py`, `model_head.py`, `task_types.py`)

Heads: `BaseNLPHead`, `TextClassificationHead`, `TokenClassificationHead`, `QuestionAnsweringHead`, `TextSimilarityHead`, `TextGenerationHead`, `MultipleChoiceHead`, `MultiTaskNLPHead` (not a `BaseNLPHead`). 24/37 `NLPTaskType` mapped, 13 raise `ValueError` — no silent fallback (pinned by `test_head_dispatch_no_silent_fallback.py`).

### Verified bugs

- **NLP-1 Classifier width mismatch** (`factory.py:521,747,807-818`): reproduced. `TextClassificationHead(task_config hidden_size=512, input_dim=768, use_pooling=True, use_task_attention=True, use_intermediate=False)` builds classifier at 512 but `call` pools to 768 (attention never runs post-pool). Error: `expected axis -1 ... value 512, but received shape (2, 768)`.
- **NLP-2 Mask ignored in `TextSimilarityHead` pair mode** (`factory.py:1513-1521`): reproduced. Identical scores with/without `attention_mask` — dict inputs stripped to tensors, `_pool_sequence(seq)` called without mask.
- **NLP-3 Mask ignored in `MultipleChoiceHead`** (`factory.py:1960-2022`): `call` has no `attention_mask` param, dict path only extracts `hidden_states`.
- **NLP-4 `MultipleChoiceHead` intermediate reshape fragility** (`factory.py:2010-2013`): works with `use_intermediate=False`; flatten/reshape path breaks on symbolic shapes.
- **NLP-5 `MultiTaskNLPHead` outer dict aliased** (`factory.py:2124`): `self.task_configs = task_configs` stores caller ref.
- **NLP-6 `create_nlp_head` dict path rejects string `task_type`** (`factory.py:2507`): `NLPTaskConfig(**dict)` fails; `model_head.py:109` accepts via `from_string`. Inconsistent.
- **NLP-7 `TEXT_REGRESSION` passes config then dies in head** (`task_types.py:552-574`, `factory.py:716`): `__post_init__` validates only 5 tasks; 7 more routed to `TextClassificationHead` unvalidated.

### Serialization gaps

- `BaseNLPHead.get_config:573` saves 10/14 fields (drops `num_labels,max_length,beam_size,temperature`).
- `MultiTaskNLPHead.get_config:2303` saves 6/task (also drops `loss_weight,label_smoothing,use_crf,use_attention_pooling`).
- `task_heads/task_projections` plain dicts (`factory.py:2129`), `similarity_layers` list (`factory.py:1403`) — violates flat-list/named-attr round-trip rule. No NLP round-trip test (vision has one).
- `from_config:615,2333` mutates caller dict via `pop`.

### Validation gaps

- Bad `pooling_type` builds nothing, fails only at first `call:478`. Validate in `__init__`.
- `NLPHeadConfiguration.get_default_config:2601` returns `dropout_rate,use_crf,vocabulary_size` — none are `BaseNLPHead.__init__` params. Dead helper.
- `len(ops.shape(h))==3` at `807,1519,1992` — fragile under tracing; use `ops.ndim`.
- `BaseNLPHead` not exported from `nlp/__init__.py` / top `__init__.py` (asymmetry).

## 2. Vision (`vision/factory.py`, `task_types.py`)

Heads: `BaseVisionHead`, `DetectionHead`, `SegmentationHead`, `DepthEstimationHead`, `ClassificationHead`, `InstanceSegmentationHead`, `EnhancementHead`, `MultiTaskHead`. 10/37 dispatched, 27 raise. Round-trip by value pinned (`test_every_vision_head_survives_a_keras_round_trip.py`).

- **VIS-1 `ClassificationHead pooling_type` never validated**: invalid string builds no pooler (`2214`), `call:2361` raises `AttributeError`.
- **VIS-2 Dead `dropout_rate`**: progressive `ConvBlock:1644` + instance mask `ConvBlock:2596` omit it.
- **VIS-3 Unvalidated counts**: `scale_factor`, `num_classes/num_anchors/bbox_dims`, `output_channels`, `num_instances`; tiny `hidden_dim` → `filters=0`.
- **VIS-4 `create_enhancement_head:3377` no `from_string`**: string misses `scale_factor=2` default.
- **VIS-5 Reuse `output_channels` collision** (`3359,3363`): caller value → `TypeError` duplicate kwarg.
- **VIS-6 Missing per-task/`shared` key returns `None`** (`3180,3212,3241`): should raise.
- **VIS-7 Outer `task_configs` aliased** (`3105,3459`).
- Semantic: `KEYPOINT_DETECTION→DetectionHead:3365` returns `classifications/regressions` but `get_output_types:375` documents `keypoints/visibility/scores`. `SURFACE_NORMALS/OPTICAL_FLOW→Depth` inherit sigmoid range — signed flow/normals squeezed positive, needs linear mode. `MultiTaskHead` accepts 5/10; presets name unbuildable tasks. `HeadConfiguration` covers 5 + base. Segmentation skips require identical H/W, fails in `concatenate:1262` without message.

## 3. VLM (`vlm/factory.py`, `task_types.py`)

Heads: `BaseVLMHead` (no `call`), `ImageCaptioningHead`, `VQAHead`, `VisualGroundingHead`, `ImageTextMatchingHead`, `MultiTaskVLMHead`. 4/47 real, 2 explicitly rejected, 41 raise.

- **VLM-1 VQA hardcoded `num_heads=8` no divisibility guard** (`1106-1112`; captioning has one at `692`). Verified: `hidden_size=769` fails late in attention factory.
- **VLM-2 `len(ops.shape(x))`** (`1467,1921`): fragile under tracing.
- **VLM-3 Zero mask plumbing**: captioning cross-attn `892`, VQA pool `1204`, grounding mean `1471`, ITM means `1919` ignore padding.
- **VLM-4 `compute_output_shape`**: VQA reads only vision batch `1234`; grounding `grounded_features` reports `(B,hidden_dim)` `1527` not fusion width; multi-task no `task_name` `2281`.
- **VLM-5 `temperature` excluded from `get_config:1981`**: ctor value lost.
- **VLM-6 Dict path rejects string `task_type`** (`2436`).
- **VLM-7 List path collapses duplicates** (`2481`; `VLMTaskConfiguration:553` rejects).
- **VLM-8 `call(inputs,task_name=None,training=None)` steals positional `training`** (`2252`).
- **VLM-9 11 dead `VLMTaskConfig` fields** (`task_types.py:431-447`): heads want `fusion_strategy`/`pooling_strategy`; `__post_init__` fills `num_classes` for 3 tasks with no head, never VQA.

## 4. Consumers

| Entry | Prod sites |
|---|---|
| `create_head(domain,...)` | 0 (tests/docs only) |
| `create_vision_head` | 25 (`models/vision/*` + `fastvlm/model.py:570`) |
| `build_model_task_head` | 15 (one per headed language model) |
| `create_nlp_head` direct | 0 |
| `create_vlm_head` / multi-task / enhancement | 0; dead import `resnet/model.py:59` |

Friction: vision 4-step boilerplate ×20 (`get_default_config→update→merge_head_config_overrides→create`, e.g. `resnet/model.py:446-469`); dict-output trap (`['logits']` forgotten, cf. `fastvlm/model.py:597`); two detection systems share one enum (generic vs `YOLOv12*Head` at `yolo12/multitask.py:46`); NLP 15 models import non-identical helper subsets; VLM unintegrated (0 models import it); trainer `train/common/nlp.py:695` hard-codes `SENTIMENT_ANALYSIS` rebuild + `TASK_HEAD_NAME` attr; `applications/` zero consumers.

## 5. Test gaps

NLP: no save/load round-trip; no `compute_output_shape`-vs-`call` contract; no mask-correctness for similarity/multiple-choice; no multi-task serialization-loss test; no dict-string-`task_type` or bad-`pooling_type` test. Vision/VLM already strong on round-trip, shape agreement, logits, ownership.

## 6. Priority fixes

P0: validate `pooling_type`/counts/`scale_factor`; thread `dropout_rate` into depth-progressive + instance-mask; fix classifier width logic; pass `attention_mask` in similarity + multiple-choice; copy outer multi-task dict + raise on missing key.
P1: VLM early width-check; VQA divisibility guard; enhancement string handling; multi-task tuple; `output_channels` pop; `HeadConfiguration` reject-or-map; one-call vision helper; export `BaseNLPHead`.
P2: NLP round-trip + mask tests; decide `create_head` fate; wire/mark VLM.
