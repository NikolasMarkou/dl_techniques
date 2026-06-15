# Model Build / Forward-Inference Sweep — Full Family Status Report

**Date:** 2026-06-15
**Plan:** `plans/plan_2026-06-15_b5cec9e4`
**Scope:** REPORT-ONLY. Zero edits to any file under `src/dl_techniques/models/` or `src/dl_techniques/layers/`.

## 1. Summary

This sweep gave the 20 model families that had **zero forward-pass coverage** (16 with no test dir, 4 with 0-byte stub files) a **permanent pytest smoke test** that builds the smallest documented variant and runs one `training=False` forward pass with a NaN/Inf guard. For the remaining already-covered families the sweep **cites the existing test suite as evidence and does NOT re-run it** (the full suite is ~1.5h; re-running 50+ suites has low marginal signal — per the user's cite-not-rerun decision D-001). No model or layer code was fixed: every build/forward break is **documented**, encoded as `pytest.mark.xfail(strict=False)` / `pytest.mark.skip` with the captured error, and routed to the `## Backlog` (Section 3) for a future dedicated fix plan. The scoped suite for all 20 new tests is **green** (only passed / xfailed / skipped — zero red).

**Headline counts (20 newly-tested gap families):**

| Disposition | Count | Families |
|---|---|---|
| **PASS** (build + forward OK) | **9** | masked_autoencoder, relgt, byte_latent_transformer, distilbert, fftnet (vision path only), coshnet, yolo12, mini_vec2vec, tabm |
| **XFAIL** (real breakage, captured) | **10** | latent_gmm_registration, shgcn, hierarchical_reasoning_model, nano_vlm, darkir, dino_v1, pft_sr, swin_transformer, mothnet, nano_vlm_world_model |
| **SKIP** (no top-level model) | **1** | jepa |

Risk-cluster regression spot-check (transformer sub-package was heavily refactored 2026-06-14/15): the existing suites for `clip`, `mobile_clip`, `fastvlm`, `video_jepa` were **re-run** this plan — **162 passed, 4 skipped, 0 failed** (314s, GPU1). No regression from the transformer refactor. `nano_vlm` was the only refactor consumer that broke, and it is in the gap list (XFAIL).

> **Count note.** Prior findings cited "71 families". An exhaustive `ls src/dl_techniques/models/` shows **69 standalone family directories + 1 `time_series` container that expands to 7 sub-families** (adaptive_ema, deepar, mdn, nbeats, prism, tirex, xlstm) = **76 family rows** below. Every directory gets a row. The "51 already-covered" headline in the plan corresponds to the non-gap set; the exact covered count is **56** standalone-or-sub families once `time_series` is expanded. The 20-gap figure is exact and unchanged.

---

## 2. Full Family Status Table

Legend — **Evidence**: `new smoke test (this plan)` = built+forwarded here; `existing test suite` = cited test dir, NOT re-run (except the 4 re-run risk-cluster families, flagged); `known-broken note` = documented breakage.

| # | Family | Build | Forward | Evidence | Notes |
|---|---|---|---|---|---|
| 1 | accunet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_accunet/` |
| 2 | bert | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_bert/` |
| 3 | bias_free_denoisers | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_bias_free_denoisers/` |
| 4 | byte_latent_transformer | PASS | PASS | new smoke test (this plan) | `create_blt_model`; real sig `(variant, vocab_size, max_sequence_length, ...)` |
| 5 | capsnet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_capsnet/` |
| 6 | cbam | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_cbam/` |
| 7 | ccnets | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_ccnets/` |
| 8 | cliffordnet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_cliffordnet/` |
| 9 | clip | PASS (re-run) | PASS (re-run) | existing test suite (re-run step 5) | `tests/test_models/test_clip/`; risk-cluster — re-run, no regression |
| 10 | convnext | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_convnext/` |
| 11 | convnext_patch_vae | assumed-PASS | assumed-PASS | existing test suite | `test_convnext_patch_vae/` + `test_convnext_patch_vae_v2/` |
| 12 | convunext | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_convunext/` |
| 13 | coshnet | PASS | PASS | new smoke test (this plan) | filled 0-byte stub `test_coshnet/test_model.py` |
| 14 | darkir | PASS | PASS | smoke test (FIXED plan_2026-06-15_00924f53) | was: `keras.layers` has no `DepthToSpace`. FIXED: added `PixelShuffle2D` layer + cascade fixes (skip channels, `_add_list` for missing `ops.add_n`). |
| 15 | depth_anything | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_depth_anything/` (forces CPU) |
| 16 | detr | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_detr/` — **FUNCTIONAL** (21 tests pass); SYSTEM.md line 245 "broken" is STALE |
| 17 | dino (v1/v2/v3) | XFAIL | **BROKEN** | new smoke test (this plan) / known-broken note | `RuntimeError: forgot to call super().__init__()` in `DINOv1.__init__` (ctor-order; affects v2/v3 too). `.keras` round-trip separately known-broken |
| 18 | distilbert | PASS | PASS | new smoke test (this plan) | raw `DistilBERT` ctor smallest config |
| 19 | fastvlm | PASS (re-run) | PASS (re-run) | existing test suite (re-run step 5) | `tests/test_models/test_fastvlm/`; risk-cluster — re-run, no regression |
| 20 | fftnet | PASS (vision) | PASS (vision) | new smoke test (this plan) | vision stack only; **SpectreHead NOT tested** — separately triple-dead |
| 21 | fnet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_fnet/` |
| 22 | fractalnet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_fractalnet/` |
| 23 | gemma | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_gemma/` |
| 24 | gpt2 | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_gpt2/` |
| 25 | hierarchical_reasoning_model | XFAIL | **BROKEN** | new smoke test (this plan) | `ValueError: convert value (None)` — None propagates to tensor conversion in `HierarchicalReasoningModel.call` |
| 26 | ideogram4 | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_ideogram4/` |
| 27 | jepa | SKIP | SKIP | new smoke test (this plan) / known-broken note | no top-level `keras.Model`; `JEPAEncoder`/`JEPAPredictor` are layers-only |
| 28 | kan | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_kan/` |
| 29 | latent_gmm_registration | PASS | PASS | smoke test (FIXED plan_2026-06-15_00924f53) | was: `keras.ops` has no `get_graph_feature`. FIXED: implemented `_get_graph_feature` DGCNN kNN helper + cascade fixes in `compute_rigid_transform` (rank-4 weight broadcast, `tf.linalg.svd` return-order). |
| 30 | lewm | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_lewm.py` (file, not dir) |
| 31 | mamba | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_mamba/` |
| 32 | masked_autoencoder | PASS | PASS | new smoke test (this plan) | `create_mae_model(encoder, 16, 0.75, (32,32,3))` w/ tiny in-test encoder |
| 33 | masked_language_model | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_masked_language_model/` |
| 34 | memory_bank | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_memory_bank/` |
| 35 | mini_vec2vec | PASS | PASS | new smoke test (this plan) | `create_mini_vec2vec_aligner(128)` |
| 36 | mobile_clip | PASS (re-run) | PASS (re-run) | existing test suite (re-run step 5) | `tests/test_models/test_mobile_clip/`; risk-cluster — re-run, no regression. `mobile_clip_v2.py` is 0 bytes (v1 only) |
| 37 | mobilenet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_mobilenet/` |
| 38 | modern_bert | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_modern_bert/`; `modern_bert_blt_hrm` GHOST reportedly functional (commit 124d464b) — not exercised here |
| 39 | mothnet | PASS | PASS | smoke test (FIXED plan_2026-06-15_00924f53) | was: `keras.ops` has no `scatter_nd_update`. FIXED: `scatter_nd_update` -> `scatter_update` (one-token). |
| 40 | nam | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_nam/` |
| 41 | nano_vlm | XFAIL | **BROKEN** | new smoke test (this plan) | `Unrecognized keyword arguments passed to MultiModalFusion: {'embed_dim','num_heads'}` (transformers-refactor signature drift); filled `test_nanovlm/test_model.py` stub |
| 42 | nano_vlm_world_model | XFAIL | **BROKEN** | new smoke test (this plan) / known-broken note | `keras.random.uniform requires a floating point dtype. Received: int32` (timestep sampling in `ScoreBasedNanoVLM.call`); GHOST still broken despite commit 1b61a381 |
| 43 | ntm | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_ntm/` |
| 44 | pft_sr | XFAIL | **BROKEN** | new smoke test (this plan) | `Unrecognized keyword arguments passed to PFTBlock: {'drop_path': 0.0}` (never exercised since import fix fec455f7) |
| 45 | power_mlp | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_powermlp/` |
| 46 | pw_fnet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_pw_fnet/` |
| 47 | qwen | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_qwen/` |
| 48 | relgt | PASS | PASS | new smoke test (this plan) | `create_relgt_model(2)`, dict input |
| 49 | resnet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_resnet/` |
| 50 | sam | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_sam/` |
| 51 | scunet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_scunet/` |
| 52 | sd3_mmdit | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_sd3_mmdit/` |
| 53 | shgcn | XFAIL | **BROKEN** | new smoke test (this plan) | `SHGCNLayer.call` requires a SparseTensor adjacency; dense float32 rejected (TypeError) |
| 54 | som | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_som/` |
| 55 | squeezenet | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_squeezenet/` |
| 56 | swin_transformer | XFAIL | **BROKEN** | new smoke test (this plan) | `SwinTransformerBlock expects 4D (batch,h,w,c), got (None,64,96)` (model flattens patches before a 4D-expecting block); filled stub |
| 57 | tabm | PASS | PASS | new smoke test (this plan) | `create_tabm_mini`; real sig `(n_num_features, cat_cardinalities, n_classes, k=8, ...)` |
| 58 | thera | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_thera/` |
| 59 | time_series/adaptive_ema | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_adaptive_ema/` |
| 60 | time_series/deepar | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_deepar/` |
| 61 | time_series/mdn | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_mdn/` |
| 62 | time_series/nbeats | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_nbeats/` |
| 63 | time_series/prism | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_prism/` |
| 64 | time_series/tirex | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_tirex/` |
| 65 | time_series/xlstm | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_xlstm/` |
| 66 | tiny_recursive_model | assumed-PASS | assumed-PASS | existing test suite | `test_tiny_recursive_model/` + `test_trm/` (both non-empty) |
| 67 | tree_transformer | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_tree_transformer/` |
| 68 | vae | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_vae/` |
| 69 | video_jepa | PASS (re-run) | PASS (re-run) | existing test suite (re-run step 5) | `tests/test_models/test_video_jepa/`; risk-cluster — re-run, no regression |
| 70 | vit | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_vit/` |
| 71 | vit_hmlp | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_vit_hmlp/` |
| 72 | vit_siglip | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_vit_siglip/` |
| 73 | vq_vae | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_vq_vae/` |
| 74 | vq_vae_rotation | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_vq_vae_rotation/` |
| 75 | wave_field_llm | assumed-PASS | assumed-PASS | existing test suite | `tests/test_models/test_wave_field_llm/` |
| 76 | yolo12 | PASS | PASS | new smoke test (this plan) | `create_yolov12_feature_extractor()` |

**Non-forward-breaking known-broken items (NOT regressions, do NOT block any forward pass):**
- **dino `.keras` round-trip** — `DINOHead.build()` appends sublayers without calling their `.build()`; `load_model` finds unbuilt children (`dino_v1.py:179`). Forward is ALSO broken here for a separate reason (ctor-order, row 17), so dino is fully XFAIL this plan.
- **fftnet SpectreHead** — separately triple-dead (`tf.signal.rfft(axis=)` TypeError + absent `ops.complex`); explicitly NOT exercised (out of scope, dedicated plan). The fftnet **vision** path PASSES (row 20).
- **DETR** — SYSTEM.md line 245 ("pre-existing broken, no test suite") is **STALE**; commit 072df479 rewrote DETR and its 21-test suite passes.

---

## 3. Backlog

Each item is a candidate for a future dedicated fix plan. Grouped by likely root-cause class. Format: **family** — error / root cause — *source to investigate* — fix hint.

### Class A — Missing `keras.ops` / `keras.layers` symbol (dead-on-forward) — ✅ ALL RESOLVED in plan_2026-06-15_00924f53
1. ~~**latent_gmm_registration** — `keras.ops` has no `get_graph_feature`.~~ **RESOLVED**: implemented `_get_graph_feature` DGCNN kNN helper (`layers/geometric/point_cloud_autoencoder.py`) + cascade fixes in `compute_rigid_transform`. Smoke test passes.
2. ~~**mothnet** — `keras.ops` has no `scatter_nd_update`.~~ **RESOLVED**: `scatter_nd_update` → `scatter_update` (`layers/mothnet_blocks.py:407`). Smoke test passes.
3. ~~**darkir** — `keras.layers` has no `DepthToSpace`.~~ **RESOLVED**: added serializable `PixelShuffle2D` (`layers/pixel_unshuffle.py`) + cascade fixes (skip channels; `keras.ops.add_n` also absent → `_add_list` helper). Smoke test passes. NOTE for future: `keras.ops.depth_to_space` and `keras.ops.add_n` do NOT exist in Keras 3.8 (the original fix hint above was wrong on `depth_to_space`).

### Class B — Construction / signature drift (kwargs no longer accepted by a refactored sub-layer)
4. **nano_vlm** — `Unrecognized keyword arguments passed to MultiModalFusion: {'embed_dim','num_heads'}` (transformers-refactor signature drift). *`models/nano_vlm/model.py:102-105` consumers; `MultiModalFusion` ctor.* Fix hint: update the call site to the post-refactor `MultiModalFusion` signature.
5. **pft_sr** — `Unrecognized keyword arguments passed to PFTBlock: {'drop_path': 0.0}` (never exercised since import fix fec455f7). *`models/pft_sr/model.py` PFTBlock instantiation.* Fix hint: drop/rename `drop_path` to the kwarg `PFTBlock` now expects (likely `drop_path_rate`), or thread it through.
6. **dino_v1 (also v2/v3)** — `RuntimeError: you forgot to call super().__init__() as the first statement` in `DINOv1.__init__` (construction-order bug). *`models/dino/dino_v1.py` (and v2/v3) `__init__`.* Fix hint: move `super().__init__(...)` to the first statement before any attribute assignment. Note: `.keras` round-trip is a separate known break in `DINOHead.build()` (`dino_v1.py:179`).

### Class C — Input-contract bug (model wiring feeds the wrong tensor shape/type to a block)
7. **shgcn** — `SHGCNLayer.call` requires a SparseTensor adjacency; a dense float32 adjacency is rejected (TypeError). *`models/shgcn/model.py` → `SHGCNLayer.call`.* Fix hint: accept/convert dense adjacency, or document the sparse-input contract and have the smoke test build a SparseTensor.
8. **swin_transformer** — `SwinTransformerBlock expects 4D input (batch,h,w,c), got (None,64,96)`; the model flattens patches to 3D before a 4D-expecting block. *`models/swin_transformer/model.py` patch-embed → block wiring.* Fix hint: keep/restore the (B,H,W,C) layout into the block, or reshape 3D→4D at the block boundary.
9. **hierarchical_reasoning_model** — `ValueError: Attempt to convert a value (None)...`; a `None` propagates into a tensor conversion. *`models/hierarchical_reasoning_model/model.py` → `HierarchicalReasoningModel.call`.* Fix hint: trace the optional input/state that is `None` at inference (likely `puzzle_ids` or a carried hidden state) and supply a default.
10. **nano_vlm_world_model** — `keras.random.uniform requires a floating point dtype. Received: int32` (timestep sampling). *`models/nano_vlm_world_model/model.py` → `ScoreBasedNanoVLM.call`.* Fix hint: sample timesteps with a float dtype then cast to int, or use `keras.random.randint`. GHOST: still broken despite commit 1b61a381.

### Class D — No top-level model (cannot forward without bespoke assembly)
11. **jepa** — no top-level `keras.Model`; only `JEPAEncoder` / `JEPAPredictor` layers. *`models/jepa/encoder.py`.* Fix hint: add a thin `keras.Model` wrapper / factory that assembles encoder+predictor (or document it as a sub-package of `video_jepa` and remove the smoke-test expectation). Disposition: `skip`, not `xfail`.

### Out-of-scope / separately-tracked (not counted in the 11 above)
- **fftnet/SpectreHead** — triple-dead: `tf.signal.rfft(axis=)` TypeError + absent `ops.complex` (`models/fftnet/components.py:746,754,775`). Dedicated plan; NOT exercised this sweep. fftnet vision path is healthy.

---

## 4. Coverage Note

After this sweep, **every** model family directory under `src/dl_techniques/models/` has at least one forward-pass test reference: the 56 already-covered families (cited existing test suites, of which `clip`/`mobile_clip`/`fastvlm`/`video_jepa` were re-run with zero regression) plus the 20 newly-tested gap families. The important caveat: **10 of the 20 new tests are `xfail`-documented-broken and 1 is `skip`**, so a green scoped suite for the new files does **not** mean those 11 families forward correctly — it means their breakage is now permanently captured (with the exact error) and routed to the Section 3 backlog. The 9 new PASS families now have durable build+forward regression coverage they previously lacked.

No `src/dl_techniques/models/` or `src/dl_techniques/layers/` code was modified by this plan (HARD report-only invariant; `git diff` against `main` over those paths is empty).
