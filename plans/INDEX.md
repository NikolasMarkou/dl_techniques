# Plan Index
*Topic-to-directory mapping. Updated on close. Survives sliding window trim.*

| Plan | Date | Goal | Key Topics |
|------|------|------|------------|
| plan_2026-04-24_1c5ae010 | 2026-04-24 | Refine `src/train/cliffordnet/train_clip.py`: |  |
| plan_2026-04-24_e4c8ebab | 2026-04-24 | Refactor `CliffordCLIP`'s vision tower into a true hierarchi |  |
| plan_2026-04-24_cf1a9ab7 | 2026-04-24 | Build a CIFAR-100 experiment that trains 5 hierarchical/mult |  |
| plan_2026-04-29_b6dbc601 | 2026-04-29 | Add `CliffordNetBlockDS` to `src/dl_techniques/layers/geomet |  |
| plan_2026-05-04_38e259bf | 2026-05-04 | Merge `RoutingProbabilitiesLayer` (deterministic, parameter- | layer-internals.md, usage-sites.md, api-design.md |
| plan_2026-05-04_1b2810b6 | 2026-05-04 | Create a CliffordNetLM variant that uses `RoutingProbabiliti | lm-structure.md, loss-compatibility.md, routing-cost-and-modes.md |
| plan_2026-05-05_60c5be7d | 2026-05-05 | Fix the deep-review findings on `src/dl_techniques/layers/ac | code-and-line-refs.md |
| plan_2026-05-05_0eac2c81 | 2026-05-05 | Verify each issue raised in the Phase-5 review of `clifford_ | f-001, f-002, f-003 |
| plan_2026-05-06_13a2df9e | 2026-05-06 | Add `CausalCliffordNetBlockDSv2` to `src/dl_techniques/layer | f-001 scope-and-callers.md, f-002 causality-mechanics.md, f-003 dsv2-merge-points.md |
| plan_2026-05-06_82749628 | 2026-05-06 | Build `CliffordNetLMUNet` causal U-Net language model on top | causal-blocks-api.md, upsampling-causality.md, lm-and-train-mirror.md |
| plan_2026-05-07_c6dd7cc1 | 2026-05-07 | Audit-confirmed fixes to the Wikipedia + HF NLP CLM dataset  | 01-pipeline-map.md, 02-issue-catalog.md, 03-design-precedents.md |
| plan_2026-05-07_47199c68 | 2026-05-07 | Fix WaveFieldAttention V3.6 review issues — apply real fixes | f1: rfft return type empirically verified, f2: existing test suite all green (62/62), f3: test contracts locking in current behavior |
| plan_2026-05-07_a73304d4 | 2026-05-07 | Apply 4 real V3.7 review optimizations; reject the false rff |  |
| plan_2026-05-07_1519e34f | 2026-05-07 | Build a decoder-only language model using `WaveFieldAttentio | f-001 gpt2.py architecture & attention wiring, f-002 src/train/gpt2/pretrain.py conventions, f-003 wavefieldattention call signature & integration |
| plan_2026-05-07_08aaf818 | 2026-05-07 | Apply the two-part tiktoken decode hardening fix from commit |  |
| plan_2026-05-07_3f461682 | 2026-05-07 | Integrate richer LLM evaluation metrics (perplexity, bits-pe | f-001 llm trainer inventory & current metric setup, "accuracy", f-002 existing metrics infrastructure |
| plan_2026-05-07_824e5687 | 2026-05-07 | No goal |  |
| plan_2026-05-08_146ae899 | 2026-05-08 | Build a Neural Memory-Augmented Transformer for `WaveFieldLL |  |
| plan_2026-05-09_0f39a086 | 2026-05-09 | Fix every finding from the deep code review of `src/dl_techn |  |
| plan_2026-05-10_44694bc9 | 2026-05-10 | Deep review of `src/dl_techniques/models/depth_anything/`, t |  |
| plan_2026-05-10_bd098beb | 2026-05-10 | Fix every remaining issue documented in `src/dl_techniques/m |  |
| plan_2026-05-10_54e6e303 | 2026-05-10 | Close every remaining OPEN item in `src/dl_techniques/models |  |
| plan_2026-05-10_bdb2c84d | 2026-05-10 | Two coupled deliverables: |  |
| plan_2026-05-10_17633038 | 2026-05-10 | Fix the `WrappedLoss` save/load round-trip bug **AND** promo |  |
| plan_2026-05-10_e6309bd5 | 2026-05-10 | Fix the two correctness bugs in `src/dl_techniques/models/ti |  |
| plan_2026-05-11_3c3ed037 | 2026-05-11 | Two coupled deliverables in one iteration: |  |
| plan_2026-05-11_0a5779e8 | 2026-05-11 | Refactor `src/dl_techniques/models/tree_transformer/` to mir |  |
| plan_2026-05-11_9357982a | 2026-05-11 | Refactor `src/dl_techniques/models/bert/` to mirror the stru | iter-n/step-m |
