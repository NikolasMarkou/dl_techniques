# Decision Log
*Append-only. Never edit past entries.*

*Cross-plan context: see plans/FINDINGS.md, plans/DECISIONS.md, and plans/LESSONS.md*

## 2026-04-21T10:15:00Z — PLAN iter-1: chosen approach

**Decision**: Port LeWM as a new model package under `src/dl_techniques/models/lewm/`, reusing the existing `ViT` model for the encoder backbone, writing new custom layers for the parts that have no analogue (`AdaLNZeroConditionalBlock`, `SIGRegLayer`), and a thin dataset module supporting both synthetic smoke-test data and a PushT HDF5 skeleton.

**At the cost of**: 13 new files (above the default 3-file complexity budget), ~1200 net LOC added, two non-trivial new custom layers that must be tested carefully for serialization. We accept this cost because the port is inherently a new-model task — there is no shortcut. We pay the complexity up front, bounded by tests + serialization round-trip + AdaLN-zero identity check.

**Alternatives rejected**:
- *Extend existing `models/jepa/`* — that package is masked I/V-JEPA, not action-conditioned; name collision would confuse users. Use distinct `models/lewm/`.
- *Extend `layers/film.py` to 6-way modulation* — AdaLN-zero is a different pattern (gated residual + zero-init). Writing a dedicated layer is cleaner than overloading FiLM.
- *EMA target encoder* — upstream LeWM does not use EMA (target encoder = live encoder, gradient flows). We match upstream. Noted as future ablation.
- *Rewrite ViT-tiny inline* — wasteful; existing ViT with `include_top=False, pooling='cls', scale='tiny', patch_size=14` is a drop-in fit.

**Assumptions that could invalidate**: A1 (ViT CLS pooling returns (B, 192) cleanly), A4 (no EMA target), A6 (BatchNorm safe under fit). Tracked in plan.md Assumptions table.

## 2026-04-21T10:15:01Z — D-001 anchor: target encoder is live, not EMA

**Decision**: In `LeWM.call`, the target embedding for the JEPA loss comes from the **same live encoder** as the context embedding. Gradient flows through the target path (no `stop_gradient`). This matches upstream `/tmp/lewm_source/jepa.py:29-45` where `self.encoder` is called once per forward and no `.detach()` appears before the loss.

**Why this matters for future changes**: Many JEPA variants (I-JEPA, V-JEPA, BYOL-style) use an EMA target encoder with stop-gradient. A future reader of this code might "helpfully" add stop-gradient thinking it's a bug. It is not. Upstream LeWM is the ablation where the target encoder is live.

**Anchor location**: Inline `# DECISION D-001` comment in `models/lewm/model.py` at the target-emb computation site.

## 2026-04-21 — D-002 anchor: MLPProjector uses LayerNorm, not BatchNorm

**Decision**: `MLPProjector` (models/lewm/projector.py) uses `LayerNormalization` on the hidden activation, not `BatchNormalization`.

**Why**: Plan.md Step 4 described "Linear → BatchNorm → GELU → Linear" (matching findings #1 summary). Re-reading the actual upstream class `/tmp/lewm_source/module.py:159-172`, `MLP.__init__` defaults `norm_fn=nn.LayerNorm`, not BatchNorm. The JEPA wiring uses the default. We match upstream truth (code), not the plan's paraphrase. This also avoids the BN-batch-of-1 failure mode flagged in plan.md Pre-Mortem Scenario 2, so assumption A6 is moot here.

**Impact**: No downstream change — the projector's external shape contract is identical. Loss curves may differ slightly vs the hypothetical-BN version, but the smoke test just needs finite loss.

**Anchor location**: inline docstring in `models/lewm/projector.py` explaining the LayerNorm default and the reason.

## 2026-04-21T13:11:00Z — REFLECT iter-1: all criteria PASS

**Outcome**: All 6 success criteria (C1–C6) PASS on first attempt across 10 plan steps with zero fix attempts. No surprises, no failed falsification signals, no scope drift.

**Evidence**:
- `pytest tests/test_layers/test_adaln_zero.py tests/test_regularizers/test_sigreg.py tests/test_models/test_lewm.py -vvv` → 11/11 passed in 20.53s (CPU).
- `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm --synthetic --batch-size=2 --epochs=1 --steps-per-epoch=2` → exit 0, final loss 0.8782 (finite), `training_log.csv` + `final_model.keras` + `last.keras` written under `results/lewm_20260421_130931/`.

**Simplification Checks**: all 6 clean. 13 files added was within the plan's declared budget (justified: full-model port), and every abstraction maps 1:1 to an upstream class.

**Devil's advocate**: the smoke defaults downscale (56×56, depth=2, num_proj=64). Full-spec LeWM at 224×224 / depth=6 / num_proj=1024 is not exercised here. Any scale-only issue would not invalidate the port but would require tuning on a full run — explicitly out of scope for this plan.

**Recommendation**: CLOSE. Follow-up work (real HDF5, full-scale training, eval.py / MPC) should become new plans.
