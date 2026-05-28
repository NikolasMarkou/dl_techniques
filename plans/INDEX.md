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
