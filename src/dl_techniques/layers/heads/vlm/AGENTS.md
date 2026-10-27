# heads/vlm

Vision-language task heads: layers that take image and text features together. The parent
`../AGENTS.md` owns the cross-domain rules; this file covers `vlm/`. `README.md` is the user-facing
tour. **No model, trainer or application constructs a VLM head today**; the callers are `layers/heads`
and the tests.

## Layout

One module per head, a shared base, and thin factories. Dependencies run downward only.

| File | Holds |
|---|---|
| `task_types.py` | `VLMTaskType` (47 members), `VLMTaskConfig` (six read fields; `to_dict`, `coerce`), `VLMTaskConfiguration` |
| `utils.py` | `_accepted_constructor_kwargs`, `_ffn_width_kwargs`, `_check_task_config_fields`, `_pool_sequence`, `_MASK_KEYS`, `_PER_HEAD_CONFIG_FIELDS`, the Keras-base and wrapper-owned kwarg sets |
| `base.py` | `BaseVLMHead`: owns only `fusion`; no `call`, refuses direct construction |
| `image_captioning.py`, `vqa.py`, `visual_grounding.py`, `image_text_matching.py` | one head each (`ImageCaptioningHead` and `VQAHead` derive from `keras.layers.Layer`; the other two from `BaseVLMHead`) |
| `dispatch.py` | `VLM_HEAD_DISPATCH` (4 rows) and `get_head_class` |
| `multitask.py` | `MultiTaskVLMHead`: one sub-head per task, three-tier kwarg routing |
| `factory.py` | `create_vlm_head(task_config, **kwargs)`, `create_multi_task_vlm_head(task_configs, **kwargs)` |
| `__init__.py` | the curated public surface (11 names) and the package-level prose |

Import order is `task_types` -> `utils` -> `base` -> leaf heads -> `dispatch` -> `multitask` -> `factory`.
`factory.py` was one 2949-line file until 2026-10-27; the code moved verbatim.

## Rules specific to this package

- **Only four task types have a head** (captioning, VQA, visual grounding, image-text matching).
  `get_head_class` goes through `../validation.lookup_head` and raises `ValueError` for the other 43;
  there is no placeholder row. `BaseVLMHead` is not a head.
- **Registration follows the defining module** (`...heads.vlm.vqa>VQAHead`, ...), pinned for all 24
  heads by `tests/test_layers/test_heads/test_the_head_classes_register_under_their_own_module.py`.
- **Patch where a name is used.** `_ffn_width_kwargs` is bound by name in both `image_captioning.py` and
  `image_text_matching.py`; a test that replaces it must replace it in each of those modules, not in
  `utils.py`.
- **A head owns what its `call` runs** (D-032). A head refuses a `VLMTaskConfig` field it does not read
  (`vocab_size` is captioning's, `num_classes` is VQA's); `__post_init__` fills nothing in (D-058).
- **Padding masks** are optional dict keys next to the features they describe (`vision_mask`,
  `text_mask`, `question_mask`; `(batch, length)`, 1 = real). Pooling goes through
  `../validation.pool_or_passthrough`: a mask for a rank-2 (already pooled) input raises. Mean and max
  build `SequencePooling` directly, which draws no random seed (D-055, D-087). Masked region scores use
  `../masking.fill_masked`.
- **Input keys differ by head.** `VQAHead` reads `question_features`; the other three read
  `text_features`. A multi-task wrapper mixing VQA with another task needs both keys.
- **Multi-task kwarg routing is three-tiered**: wrapper-owned (`task_config`, passing it raises),
  `task_specific_kwargs` (strict, keyed by task name), and shared kwargs routed best-effort to each head
  that accepts them. A key no head accepts raises.
- **Config objects**: `VLMTaskConfig.coerce` returns an instance unchanged (it does not copy, unlike
  `NLPTaskConfig.coerce`); `to_dict` writes `task_type` as its string value.
- Logits out: `confidence` and `match_score` are pre-activation; only the grounding `bbox` is squashed.

## Tests

`tests/test_layers/test_heads/test_vlm.py` and the `test_the_vlm_*` / `test_every_vlm_*` files.
`test_the_vlm_dispatch_names_exactly_four_heads.py` pins the table;
`test_the_vlm_heads_honour_padding_masks.py` pins the masks. Repo-level pins that name files here:
`test_the_norm_factory_family_is_pinned.py` (`image_text_matching.py` one dynamic site,
`image_captioning.py` three literal), `test_the_raw_gelu_and_leaky_relu_calls_cannot_grow.py`
(`vqa.py`), the FFN construction-site sweep in `tests/test_layers/test_ffn/test_factory.py`, and the
dead-names guard. Scope runs to `tests/test_layers/test_heads/`.
