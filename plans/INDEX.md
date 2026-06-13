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
| plan_2026-05-11_a9e8e6f6 | 2026-05-11 | Refactor `src/dl_techniques/models/gpt2/` to follow the `res |  |
| plan_2026-05-11_0090b0b8 | 2026-05-11 | Refactor `src/dl_techniques/models/cliffordnet/` to match th |  |
| plan_2026-05-11_46ecfa0b | 2026-05-11 | Split `src/dl_techniques/models/tree_transformer/model.py` i |  |
| plan_2026-05-12_f2d29729 | 2026-05-12 | Fix the train/val preprocessing divergence in `src/train/vit |  |
| plan_2026-05-12_94b9fab5 | 2026-05-12 | Produce three deliverables for `src/dl_techniques/models/ada |  |
| plan_2026-05-12_86f14c6e | 2026-05-12 | Fix `dl_techniques.models.adaptive_ema` per prior code revie |  |
| plan_2026-05-12_5f0e087c | 2026-05-12 | Finish adaptive_ema: (1) I-3 vectorize `ExponentialMovingAve |  |
| plan_2026-05-12_ebb5fac5 | 2026-05-12 | Comprehensive review of prism model — produce sibling-style  |  |
| plan_2026-05-12_995a621a | 2026-05-12 | Make the tree encoder a participant in the DFSA reduction-sc |  |
| plan_2026-05-12_e9584ff4 | 2026-05-12 | Add BLT-style tokenization-free byte-level language modeling | 0, 260)` (offset +4). output: plain `(b, t, 260)` logits (not dict-keyed). has dynamic entropy patcher (`entropymodel + dynamicpatcher`), `localencoder` cross-attn pooling, `globaltransformer`, `localdecoder` cross-attn back to bytes. `__init__.py` is empty. existing `src/train/blt/train_blt.py` uses synthetic data, multi-stage trainer, does not touch `train.common.nlp`. no public tests for blt. `bytetokenizer` at `dl_techniques/layers/blt_blocks.py:232` is a python-side text↔byte helper, not a tf.data pipeline. |
| f-002 | current cliffordnetlmrouting integration surface | `findings/lm-routing-current.md` | token-id-keyed lm: `vocab_size=50261` (tiktoken gpt2 + 4 specials), `max_seq_length=512`, `channels∈{128..768}` is the feature dim d (not clifford algebraic dim). 3 embedding strategies (`hce`/`albert`/`dense`), causalcliffordnetblock×depth stack on 4-d tensors, head = `routingprobabilitieslayer(output_dim=vocab_size, mode={trainable,deterministic})` producing **probabilities** in `eps, 1-eps |
| plan_2026-05-12_632605aa | 2026-05-12 | Create a bidirectional CliffordNet U-Net embedding model pac | "last_hidden_state" |
| plan_2026-05-12_6a2cd5b3 | 2026-05-12 | Create `src/train/cliffordnet/wikipedia/pretrain.py` (+ `__i |  |
| plan_2026-05-12_13c70aed | 2026-05-12 | Add Matryoshka Representation Learning (MRL) to the LM head  | :, :w, "logits", "logits_w128", ... |
| plan_2026-05-13_03176394 | 2026-05-13 | Move `tversky_projection.py` and `kan_linear.py` into `src/d |  |
| plan_2026-05-13_16ac1621 | 2026-05-13 | Compile exhaustive per-task ML metrics reference (`src/train |  |
| plan_2026-05-13_8e866056 | 2026-05-13 | Fix what can be fixed, delete the rest, among broken src/tra |  |
| plan_2026-05-13_a1c9a52d | 2026-05-13 | Merge `src/dl_techniques/layers/memory/` and `src/dl_techniq |  |
| plan_2026-05-13_a40908e7 | 2026-05-13 | Deep review of src/dl_techniques/layers/memory/ — analysis o |  |
| plan_2026-05-13_8c1dc6fd | 2026-05-13 | Fix all bug-grade and high-value low-risk issues from the de |  |
| plan_2026-05-13_2aaad563 | 2026-05-13 | Deep comprehensive review of `src/dl_techniques/layers/logic |  |
| plan_2026-05-13_e52a5ac8 | 2026-05-13 | Double-check the prior `layers/logic/` review and apply only |  |
| plan_2026-05-13_a2b0f17b | 2026-05-13 | Implement every actionable item from the latest deep review, | prior-work-audit |
| plan_2026-05-13_3a2f1d23 | 2026-05-13 | Address the residual epistemic review of `src/dl_techniques/ |  |
| plan_2026-05-13_e33114da | 2026-05-13 | Implement the post-rewrite review fixes for `src/dl_techniqu | f1: hamacher or boundary bug, f2: gumbel softmax leaks into inference, f3: risky_stack guard misses residual-only case |
| plan_2026-05-13_d256b568 | 2026-05-13 | Create a model + training script under `src/train/logic/` th |  |
| plan_2026-05-13_25774a34 | 2026-05-13 | Run a real benchmark + interpretability study on `LearnableN |  |
| plan_2026-05-13_798d3a60 | 2026-05-13 | Implement E1 (Petersen-DLGN-style replication on MNIST + CIF |  |
| plan_2026-05-14_e26eede2 | 2026-05-14 | Implement E4 (UCI Monks-1/2/3 rule recovery) + low-data 11-b |  |
| plan_2026-05-14_c95e848c | 2026-05-14 | Implement E5 (CLEVR-Hans3 visual reasoning) for `LearnableNe | 'add','max','min' |
| plan_2026-05-14_9c6387a3 | 2026-05-14 | Multi-seed robustness sweep over E1 (MNIST+CIFAR-10), E3 (3  |  |
| plan_2026-05-17_413eae7d | 2026-05-17 | Implement Token Superposition Training (TST) as a model-agno |  |
| plan_2026-05-17_7ed2d007 | 2026-05-17 | Recover the deleted training output for the crashed Clifford | crash-root-cause.md |
| plan_2026-05-18_74a935a2 | 2026-05-18 | Extend the existing `src/train/rms_variants_train/` harness  |  |
| plan_2026-05-18_c7f1947d | 2026-05-18 | Create `ZeroCenteredAdaptiveBandRMS` layer combining zero-ce |  |
| plan_2026-05-18_63121227 | 2026-05-18 | Design — but do not execute — a comprehensive Phase 3 of the |  |
| plan_2026-05-18_b10fc418 | 2026-05-18 | Implement Rotation Trick VQ-VAE (Fifty et al. ICLR 2025) as  |  |
| plan_2026-05-18_d3655b1e | 2026-05-18 | Relocate `src/dl_techniques/layers/adaln_zero.py` to `src/dl |  |
| plan_2026-05-18_e1f12eab | 2026-05-18 | Refine the existing 8-norm RMS-variants harness at `src/trai |  |
| plan_2026-05-18_6776f8ba | 2026-05-18 | Refine the existing 8-norm `src/train/rms_variants_train/` P |  |
| plan_2026-05-19_39a6a454 | 2026-05-19 | Land Patches 1 + 2 from the BurstDP reuse-review. |  |
| plan_2026-05-19_e3f21fdb | 2026-05-19 | Train BurstDP-small on COCO 2017 via `src/train/burst_dp/tra |  |
| plan_2026-05-19_6f397a37 | 2026-05-19 | Full-purge the depth component from the `burst_dp` package + | 01-depth-touchpoints, 02-viz-callback-design, 03-constraints |
| plan_2026-05-19_64f2a17b | 2026-05-19 | Add DIV2K and VGG-Face2 as fidelity-only training datasets t |  |
| plan_2026-05-19_b225c8df | 2026-05-19 | Expose existing aux-distortion knobs through the burst-dp tr |  |
| plan_2026-05-20_b8f8df89 | 2026-05-20 | Diagnose and fix BurstDP underfitting on DIV2K (recon PSNR s |  |
| plan_2026-05-22_de5197c2 | 2026-05-22 | Implement all actionable LeWM audit findings: the two confir |  |
| plan_2026-05-23_692fd5e5 | 2026-05-23 | Fix 8 LeWM deep-review issues in one focused pass. |  |
| plan_2026-05-23_c573e591 | 2026-05-23 | No goal |  |
| plan_2026-05-23_0b664700 | 2026-05-23 | Refactor Video-JEPA from degenerate single-horizon t+1 next- |  |
| plan_2026-05-23_15151c75 | 2026-05-23 | Switch `VideoJEPA` from a live target encoder (where `z_targ |  |
| plan_2026-05-24_ca745a6c | 2026-05-24 | Close the two remaining concerns surfaced after iter-2 CLOSE |  |
| plan_2026-05-24_aebd4cbb | 2026-05-24 | Add video_jepa observability tooling and run ONE diagnostic  |  |
| plan_2026-05-25_853605c1 | 2026-05-25 | Rewrite `src/dl_techniques/regularizers/sigreg.py` to follow |  |
| plan_2026-05-25_fb57d478 | 2026-05-25 | Build a **resolution-agnostic ConvNeXt-based variational aut |  |
| plan_2026-05-25_8faec5b6 | 2026-05-25 | Rewrite `src/dl_techniques/models/convnext_patch_vae/` to fu |  |
| plan_2026-05-25_74f0eac9 | 2026-05-25 | Deep review of `src/dl_techniques/models/convnext_patch_vae/ |  |
| plan_2026-05-25_e3a309ec | 2026-05-25 | Deep review of convnext_patch_vae model and training code; e |  |
| plan_2026-05-25_a8325e3f | 2026-05-25 | Implement ConvNeXtPatchVAE collapse-prevention fixes from an |  |
| plan_2026-05-26_b11b0e90 | 2026-05-26 | Extend `train_convnext_patch_vae.py` to support ADE20K and C |  |
| plan_2026-05-26_d8c33dca | 2026-05-26 | No goal |  |
| plan_2026-05-26_d7a342f2 | 2026-05-26 | No goal |  |
| plan_2026-05-26_5abf5af3 | 2026-05-26 | No goal |  |
| plan_2026-05-26_0f3c5913 | 2026-05-26 | Implement critical fixes from augmentation pipeline audit (H |  |
| plan_2026-05-27_1a9e3221 | 2026-05-27 | Fix 6 confirmed bugs in ConvNeXtPatchVAE: 3 critical (H4, H1 |  |
| plan_2026-05-27_dee954c6 | 2026-05-27 | Implement the two-level hierarchical ConvNeXtPatchVAE archit | bug-fixes-verified.md, current-architecture.md, hierarchical-design-notes.md |
| plan_2026-05-27_c3184aea | 2026-05-27 | Replace the implicit `p(z_l2) = N(0, I)` in `HierarchicalCon | touch-points.md, math-and-init.md, risks.md |
| plan_2026-05-27_68c7fcd6 | 2026-05-27 | Add SGLD optimizer at `src/dl_techniques/optimization/sgld_o |  |
| plan_2026-05-27_84f6180d | 2026-05-27 | Bring all CLAUDE.md files into consistency with the current  |  |
| plan_2026-05-27_4a444b14 | 2026-05-27 | No goal |  |
| plan_2026-05-27_75849a91 | 2026-05-27 | Port `src/train/convnext_patch_vae_v2/` to use cliffordnet b |  |
| plan_2026-05-28_15256fe3 | 2026-05-28 | In `src/train/convnext_patch_vae/` (V1 — corrected from V2 i | scope-clarification.md, hierarchical-viz-gap.md, annealing-config-redundancy.md |
| plan_2026-05-29_4538aa62 | 2026-05-29 | Implement **Polar Weight Normalization** (a Keras 3 layer th |  |
| plan_2026-05-29_8da9ba37 | 2026-05-29 | Implement `OrthogonalButterfly`: a learnable, exactly-orthog |  |
| plan_2026-05-29_8246cd14 | 2026-05-29 | Build a lightweight anomaly-detection application under `src |  |
| plan_2026-05-29_f1605e5a | 2026-05-29 | Fix the 8 confirmed defects from the epistemic-deconstructor |  |
| plan_2026-05-30_979e95fa | 2026-05-30 | Fully remove the hierarchical ConvNeXtPatchVAE: delete the m |  |
| plan_2026-05-31_76981d58 | 2026-05-31 | Deliver the CODE action items from the CliffordNet-CLIP feas |  |
| plan_2026-05-31_3ec26e00 | 2026-05-31 | Make `node /home/arxwn/.claude/skills/iterative-planner/scri |  |
| plan_2026-05-31_42743977 | 2026-05-31 | Execute the design in `docs/20260530_cliffordnet_clip.md` en |  |
| plan_2026-06-02_30721a0f | 2026-06-02 | Extract copy-pasted functionality from `src/train/` trainers |  |
| plan_2026-06-02_35651564 | 2026-06-02 | Continue the `train/common` deduplication: land the six user |  |
| plan_2026-06-02_cc4d4e14 | 2026-06-02 | Consolidate the four largest remaining REAL code duplication |  |
| plan_2026-06-02_8422b85d | 2026-06-02 | Move `src/train/benchmarks/` (5 markdown benchmark-reference |  |
| plan_2026-06-02_e3da3ff9 | 2026-06-02 | Move `src/dl_techniques/layers/neuro_grid.py` into `src/dl_t |  |
| plan_2026-06-02_2a0b8192 | 2026-06-02 | Absorb the richer reference content from `src/dl_techniques/ |  |
| plan_2026-06-02_da7698bc | 2026-06-02 | Merge `src/dl_techniques/layers/norms/polar_weight_norm.md`  |  |
| plan_2026-06-03_943569ad | 2026-06-03 | Replace the hand-rolled drop_path implementation — `keras.la |  |
| plan_2026-06-03_bf1e592d | 2026-06-03 | Make the ConvNeXt block regularizer's `stochastic_mode` (`de |  |
| plan_2026-06-03_9e82787d | 2026-06-03 | Bring the `analyzer` spectral subsystem (`spectral_metrics.p |  |
| plan_2026-06-03_bc986e52 | 2026-06-03 | Bring `src/dl_techniques/analyzer/` spectral code (`spectral |  |
| plan_2026-06-03_5c8c6d19 | 2026-06-03 | Consolidate `src/train/ccnets/` so its documentation and cod |  |
| plan_2026-06-03_da3a2bbb | 2026-06-03 | Convert the single-file module `src/dl_techniques/layers/seq |  |
| plan_2026-06-04_a114f829 | 2026-06-04 | Add a new Keras 3 layer class `HypersphereSampling` as a SIB |  |
| plan_2026-06-04_d4ef81f1 | 2026-06-04 | Add a registry-driven sampling-layer factory to `layers/samp |  |
| plan_2026-06-04_7ff8ea8b | 2026-06-04 | Deliver a defensible, evidence-backed answer to the question |  |
| plan_2026-06-04_6196678d | 2026-06-04 | Implement a TRUE von Mises-Fisher Spherical VAE (Davidson et |  |
| plan_2026-06-05_56b39171 | 2026-06-05 | Enable `cifar100` as a dataset choice in the VAE trainer (`s |  |
| plan_2026-06-06_38aa045e | 2026-06-06 | Add a serializable per-patch von Mises-Fisher (vMF) sampling |  |
| plan_2026-06-07_8b718ac4 | 2026-06-07 | convnext_patch_vae: comprehensive review of architecture, vM |  |
| plan_2026-06-08_e3917bd5 | 2026-06-08 | Add a library-grade, two-level hierarchical variant to the ` |  |
| plan_2026-06-08_91b27275 | 2026-06-08 | Rewrite `src/applications/anomaly_detection/` so the anomaly |  |
| plan_2026-06-08_57a975d1 | 2026-06-08 | Create a new layer sub-module `src/dl_techniques/layers/mixt |  |
| plan_2026-06-08_8b32ca51 | 2026-06-08 | Consolidate the three sibling task-head packages |  |
| plan_2026-06-08_aaefc92b | 2026-06-08 | Fix the 3 named pre-existing latent bugs in `src/dl_techniqu |  |
| plan_2026-06-08_a5f40f4f | 2026-06-08 | Deep, comprehensive correctness pass over all 7 layer files  |  |
| plan_2026-06-09_be55db55 | 2026-06-09 | A four-item correctness pass: |  |
| plan_2026-06-09_a3c7304c | 2026-06-09 | Finish normalizing the four synthetic time-series trainers ( |  |
| plan_2026-06-09_e0e96220 | 2026-06-09 | Train every time-series model in `src/train/` one-by-one on  |  |
| plan_2026-06-09_49c73926 | 2026-06-09 | Fix the two broken time-series trainers (`nbeats/train_nbeat |  |
| plan_2026-06-10_f361b14c | 2026-06-10 | move src/dl_techniques/models/{tirex,prism,nbeats,mndn} unde |  |
| plan_2026-06-10_7ba2471a | 2026-06-10 | Move `src/dl_techniques/models/deepar` and `src/dl_technique |  |
| plan_2026-06-10_8b2431d9 | 2026-06-10 | Create `src/train/time_series/` and move the five time-serie |  |
| plan_2026-06-10_7a0e42b1 | 2026-06-10 | Move `src/dl_techniques/models/xlstm/` to `src/dl_techniques |  |
| plan_2026-06-10_39646d39 | 2026-06-10 | Build a thin, shared forecasting contract and the metrics/vi |  |
| plan_2026-06-10_31eed970 | 2026-06-10 | Bring the four remaining time-series trainers — **nbeats**,  |  |
| plan_2026-06-10_721a80b5 | 2026-06-10 | Make the Multi-Task MDN trainer produce a genuine multi-step |  |
| plan_2026-06-10_7036cab1 | 2026-06-10 | Ship a complete, runnable Pattern-2 training pipeline for th |  |
| plan_2026-06-10_c6197fb1 | 2026-06-10 | Add the LAST missing piece of the unified Forecast contract: |  |
| plan_2026-06-11_50891da1 | 2026-06-11 | Fix the xLSTMForecaster trainer fit() NaN by giving `mLSTMCe |  |
| plan_2026-06-11_fe7401f4 | 2026-06-11 | Normalize all 7 model families under `src/dl_techniques/mode |  |
| plan_2026-06-11_5f49f080 | 2026-06-11 | Give `TemporalConvNet` and its `TemporalBlock` |  |
| plan_2026-06-11_84296249 | 2026-06-11 | Normalize the 7 production time-series trainers (`mdn`, `nbe |  |
| plan_2026-06-11_49671f7a | 2026-06-11 | Normalize the 8 time-series trainers under `src/train/time_s |  |
| plan_2026-06-11_92e06228 | 2026-06-11 | Run all 7 active time series trainers for 100 epochs each, s |  |
| plan_2026-06-11_f662207d | 2026-06-11 | Bring every THERA custom **Layer** and **Model** into full c |  |
| plan_2026-06-12_bda3e5b5 | 2026-06-12 | Apply the WARNING-tier fixes from a completed THERA code rev |  |
| plan_2026-06-12_f8843c4f | 2026-06-12 | Resolve the three carry-forward caveats from the THERA revie |  |
| plan_2026-06-12_6cc7c378 | 2026-06-12 | Add a new `KANInitializer` (Rigas et al. 2026 init schemes:  |  |
| plan_2026-06-12_59a18a10 | 2026-06-12 | Make every new custom `keras.layers.Layer` / `keras.Model` / |  |
| plan_2026-06-12_7af1504c | 2026-06-12 | Register the newly-created Ideogram4 EMBEDDING layers into t |  |
| plan_2026-06-12_dfce0712 | 2026-06-12 | Port the MiniDiffusion SD3-style text-to-image stack from Py |  |
| plan_2026-06-12_0bb1729b | 2026-06-12 | Make more of the repo's out-of-submodule transformer/attenti |  |
| plan_2026-06-13_e7b5704d | 2026-06-13 | Bring all custom layers/models into conformance with `resear |  |
| plan_2026-06-13_28f0b453 | 2026-06-13 | Fix all confirmed bugs in `src/dl_techniques/models/detr/mod |  |
| plan_2026-06-13_250487cb | 2026-06-13 | Fix all confirmed Keras guide violations in `src/dl_techniqu |  |
| plan_2026-06-13_5b933e7f | 2026-06-13 | Implement the VSGD (Variational Stochastic Gradient Descent) |  |
