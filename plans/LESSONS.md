# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines — rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Process

- **"Remove X" can mean two different things.** When a user asks to remove a multi-stage / multi-mode surface, distinguish between (a) the *capability* is unwanted and (b) the *naming or layering* is confusing. EXPLORE should classify each piece before PLAN drafts deletion steps. Skipping this step cost one PLAN -> PLAN cycle on the cliffordnet/train_clip plan.
- **Doc updates belong in the same plan as the code change they describe.** Treating README/CLAUDE.md updates as out-of-scope follow-ups means the plan is technically "complete" while the user-visible behavior (CLI flags, examples) is still wrong. If a doc is the only public-facing reference to flags or APIs being changed, include the doc update as a plan step from the start.
- **Static grep gates can substitute for runtime probes — but only for purely mechanical surface changes.** Renames that have no internal recursion and no out-of-file callers are safe to verify with `grep -n`. Anything that touches behavior (loss math, optimizer wiring, dataset paths) needs an actual runtime smoke test.
- **PLAN to PLAN cycles are normal and not a failure.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE.
- **For pure additive layer/class work, EXPLORE→PLAN→EXECUTE→REFLECT can run in a single session with a single iteration.** The whole protocol fits cleanly when the change is non-destructive (sibling class, no signature changes to existing public surface) and tests can be scoped to one file. No PIVOT, no fix attempts needed. Don't over-engineer the protocol for simple additions — but always still write findings, plan, decisions, verification.

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite is ~1.5h and is the pre-push hook. Scope pytest to changed modules only.
- **Use `MPLBACKEND=Agg` for any training-script invocation.** Headless servers crash on the default matplotlib backend.
- **Use `dl_techniques.utils.logger` only — no `print` calls in library or training code.** This is a hard convention; checks should grep for `print(` in modified files.
- **Single GPU jobs only.** Never spawn parallel training runs (memory + driver contention on the dev box).
- **`train_clip.py`-class scripts**: a `--synthetic --max-train-samples 64 --max-val-samples 32 --epochs 1 --batch-size 8` smoke run is the cheap way to verify CLI parsing + dataset wiring + one full epoch end-to-end without external downloads. Use `--skip-pretrain` (or equivalent) on scripts with optional pretraining helpers to keep the smoke loop fast.
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken.** Use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint` for name-based transfer from `.keras` checkpoints.
- **AdamW double weight decay is a real footgun.** Never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)` — pick one. Also: exclude `logit_scale` from AdamW weight decay in CLIP-style training, otherwise the temperature decays to zero and the contrastive signal dies silently.
- **`CliffordNetBlock` is dim-preserving — it does not downsample.** Any hierarchical CliffordNet must introduce explicit inter-stage downsamplers (strided conv / avgpool+1x1 / Swin PatchMerging / depthwise-sep strided). A single-shape model is the only isotropic variant. **As of plan_2026-04-29_b6dbc601, `CliffordNetBlockDS` is available — same file — that integrates downsampling INTO the block via a strided 7x7 DW context conv plus matched-stride avg/max pool on the skip path. Use it for hierarchical stages instead of an external downsampler when the block boundary is also the downsampling point.**
- **CliffordNet does not tolerate aggressive patchify stems.** patch=4 stems collapse spatial structure before any CliffordNetBlock runs, costing 6+ pp val_acc on CIFAR-100 (V5 vs V1). Keep stem stride ≤2 before the first block stack.
- **On CIFAR-100 with ~3M params at 100 epochs, downsampler choice is ≈ noise.** Within a 3-stage 64→128→256 hierarchy, strided-conv / avgpool+1×1 / Swin patch-merging converge within <0.7 pp. Pick the simplest (avgpool+1×1) unless you have a reason.
- **GPU 1 (RTX 4070 12GB) caps CliffordNet training at batch 64 for the hierarchical variants.** Batch 128 fits V0-isotropic but V4 (4-stage 512-ch) OOMs. Halve batch when switching from GPU 0.
- **For shape-preserving Clifford block variants: pool BEFORE the stream split.** The geometric product is element-wise on (z_det, z_ctx); their spatial+channel dims must match. When introducing strided / pooled paths, downsample `x_norm` once and feed both streams from the pooled tensor — never pool only one stream. Use a separate pool instance for the residual skip on `x_prev` (cleaner serialization than reusing one layer).

## Layer testing patterns (dl_techniques)

- **Test class structure for new layers: mirror existing sibling classes in the same file.** `tests/test_layers/test_geometric/test_clifford_block.py` is a strong reference — pytest fixtures for params, classes per layer, tests cover init/build/shape/cli_modes/numerical_stability/serialization/save_load/gradient_flow/stacking. Copying that structure when adding a sibling class keeps the test surface uniform.
- **`test_residual_identity_at_init`** (gamma init ≈ 0 → output ≈ input) is the cheapest sanity check that residual wiring is correct. For downsampling variants, the reference is `pool(input)` instead of `input`.
- **For pool-choice variants (avg vs max), verify they produce different outputs** with `layer_scale_init` near zero so the residual dominates. Otherwise random init noise drowns out the pool difference.
