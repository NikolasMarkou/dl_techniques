# heads/vision

Vision task heads: detection, segmentation, depth, surface normals, optical flow, classification,
instance segmentation, enhancement and a multi-task wrapper. The parent `../AGENTS.md` owns the
cross-domain rules; this file covers `vision/`. `README.md` is the user-facing tour.

## Layout

`factory.py` is a re-export layer over per-concern modules (split out of one 1100-line file; names and
export surface unchanged). Edit the module that defines the thing, not `factory.py`.

| File | Holds |
|---|---|
| `task_types.py` | `VisionTaskType` (37 members, `coerce`), `TaskConfiguration`, `CommonTaskConfigurations`, `parse_task_list` |
| `utils.py` | `STAGE_KNOB_DEFAULTS`, `_drop_off_stage_knobs`, `_refuse_common_layers_call_never_runs`, `_validated_block_dims`, `_require_hidden_dim_at_least`, `_refuse_dropout_no_layer_applies`, attention-layout tables |
| `decoder.py` | the progressive decoder (`_create_/_build_/_run_progressive_decoder`) shared by segmentation and depth |
| `base.py` | `BaseVisionHead`: optional attention and FFN stages. No `call`; not a usable head |
| `detection.py`, `segmentation.py`, `classification.py`, `instance_segmentation.py`, `depth.py` | one task head each |
| `dense_vector.py` | `_DenseVectorHeadBase`, `SurfaceNormalsHead`, `OpticalFlowHead` |
| `enhancement.py` | `EnhancementHead` (module scope; never nest it in a function) |
| `multitask.py` | `MultiTaskHead`, `create_multi_task_head` (a sequence is keyed by `../validation.key_by_name`) |
| `dispatch.py` | `VISION_HEAD_DISPATCH` (10 of 37 task types), `create_vision_head`, `create_enhancement_head`; lookups go through `../validation.lookup_head` |
| `config.py` | `HeadConfiguration` presets, `merge_head_config_overrides`, `build_model_vision_head` (the call all 21 preset-based model sites use) |
| `factory.py` | re-exports of the above |

## Rules specific to this package

- **Registration** follows the defining module: `@register_dl_technique("dl_techniques.layers.heads.vision.<module>")`
  (`...vision.base>BaseVisionHead`, `...vision.detection>DetectionHead`, ...). Moving a class changes its
  key; `test_the_head_classes_register_under_their_own_module.py` pins it. `factory.py` stays a pure
  re-export facade because 25 model files and many tests import from it.
- **`EnhancementHead` stays module scope**, and its class name stays exactly `EnhancementHead`.
- **A head owns what its `call` runs.** `ClassificationHead` refuses `use_ffn`; `EnhancementHead` refuses
  attention, FFN and dropout; `DetectionHead(use_regression=False)` refuses `bbox_dims`. Flag refusal
  goes through `_refuse_common_layers_call_never_runs` and runs BEFORE `super().__init__`.
- **Stage knobs** (`attention_type`, `ffn_type`, `ffn_expansion_factor`) default to `None`, resolve when
  their stage is on and raise when it is off (D-210), through `../validation.resolve_stage_knobs`. The
  presets and `build_model_vision_head` drop a preset knob whose stage ends up off.
- **Attention layout** (D-071): a head runs attention in the layout its type takes; map-only types are in
  `_SPATIAL_ATTENTION_TYPES`; an unrunnable type is refused with a reason from `_UNRUNNABLE_ATTENTION_TYPES`.
- **Logits out**: every class-like output is pre-activation. Only range maps squash (progressive depth
  sigmoid, grounding `bbox`).
- `create_vision_head` raises `ValueError` for the 27 task types without a head; there is no fallback.
  `VisionTaskType` has no alias for its old name and `tests/.../test_the_vision_task_enum_has_no_alias.py`
  keeps it that way.
- `MultiTaskHead._create_task_heads` copies each per-task dict before `pop('task_type')`;
  `get_config` stores `task_type` as its string value (an Enum is not JSON).

## Tests

`tests/test_layers/test_heads/test_vision.py` plus the `test_*vision*` files, the normals/flow,
enhancement and one-call-helper files. `test_every_vision_head_survives_a_keras_round_trip.py` compares
every weight by path. `tests/test_the_head_factories_and_finetune_scripts_bind_no_dead_names.py` lists
every vision submodule (not `factory.py`, which is only re-exports and would fail its own unused-import
check). Scope runs to `tests/test_layers/test_heads/`.
