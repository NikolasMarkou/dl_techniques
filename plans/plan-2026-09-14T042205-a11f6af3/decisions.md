# Decision Log
*Plan: plan-2026-09-14T042205-a11f6af3*

**python-software.md**: consulted — applicable (`## B. Python architecture patterns`, specifically
the guidance on preserving an existing internal computation contract while relocating where a value
is computed, and `## A. Software-design models`'s treatment of a numerically-equivalent refactor as
a distinct risk class from a behavior-changing one — used to frame Step 2/3's "same computation,
different graph shape" split and the correspondingly stronger correctness-proof requirement).

## D-001 | EXPLORE → PLAN | 2026-09-14
**Context**: The prior plan (`plan-2026-09-13T165751-bc5433cb`) shipped a `tf.recompute_grad` wrap
around `MambaLayer._selective_scan` that measurably helps (v1: batch=2->4) but plateaus well below
the original batch=8 target, and separately found that disabling XLA reaches batch=8 for both v1
and v2 at a 23-31x wall-clock/step cost the user declined. This plan's own EXPLORE phase confirmed a
second, structurally distinct lever exists for v1 specifically: `_selective_scan` precomputes two
full-sequence `(batch, d_inner, seq_len, d_state)` tensors (`deltaA`, `deltaB_u`) before its
`while_loop` starts, and this precompute has a CPU-verified-equivalent per-timestep form that could
be moved inside the loop body, mirroring an existing in-file precedent (`C[:, :, t]`).
**Decision**: Attempt a genuine computation-graph change (not just a checkpointing wrap) that
eliminates the forward-only precompute tensors entirely, GATED by a cheap standalone diagnostic
(isolated-layer peak-memory A/B under both `jit_compile` states, plus a `num_layers` slope sweep)
run BEFORE any production-file rewrite — per `plans/LESSONS.md`'s standing rule to verify a
mechanism actually engaged, not just that a number moved the right direction, and per this exact
plan's own prior-plan precedent of a `tf.recompute_grad` fix whose measured benefit collapsed under
XLA fusion.
**Trade-off**: Attempt a genuine computation-graph change (not just a checkpointing wrap) to
eliminate the forward-only precompute tensors **at the cost of** a more invasive rewrite — one that
touches the actual computed values' code path, not just backward timing — which needs a stronger
numerical-identity proof (forward AND gradient equivalence, not just a shape/serialization check)
and is explicitly gated by an early falsification diagnostic before committing to the full rewrite,
meaning this plan may ship zero production-code changes if the diagnostic falsifies the premise.
**Reasoning**: The alternative (skip chunking, wait for a v2-style `d_state` decision, or silently
adopt the declined XLA-off trade-off) either leaves v1's memory ceiling unimproved or reopens a
trade-off the user already explicitly declined. Chunking is the one lever in this plan's scope that
is a pure internal optimization with no disclosed cost IF its premise holds — worth the gated
investigation. v2's `d_state` lever is deliberately NOT pursued here (see below) because it is a
capacity/quality trade-off of the same shape as the declined XLA-off decision, and this plan's scope
(set by the orchestrator) is v1-chunking only.

## D-002 | PLAN | 2026-09-14
**Context**: v2 (`Mamba2Layer._ssm_scan`) has no analogous full-sequence precompute to chunk
(`findings/v2-chunking-design.md`, a confirmed negative result from a full-method read, not an
assumption) — its only lever is `d_state` reduction, a capacity/quality trade-off, unmeasured for
quality impact.
**Decision**: Do not implement any v2 code change in this plan. Record v2's `d_state` reduction as
a NAMED, DEFERRED follow-up in Success Criteria and the final documentation step (Step 6), for a
future plan or an explicit user decision — never silently dropped.
**Trade-off**: Keeping this plan's scope clean and honest about what it did and did not attempt
**at the cost of** leaving v2's OOM ceiling (batch <=2, unimproved) unaddressed for another
iteration.
**Reasoning**: This mirrors house convention (`plans/LESSONS.md`, the prior plan's own XLA-off
naming) against silently dropping investigated-but-not-pursued options, and keeps a real
capacity/quality trade-off decision with the user rather than deciding it unilaterally inside an
autonomous plan.

## Step 1 raw measurement data | EXECUTE | 2026-09-14

Standalone scratchpad diagnostic (not `src/`/`tests/`), a faithful reproduction of
`_selective_scan`'s math (CPU-verified `np.allclose` True against the shipped precompute form
before any GPU run), stacked into an independent `ScanBlock`/`ScanStack` at
`(batch=8, seq_len=128, d_inner=1536, d_state=16)` — the `130m` variant's real constants
(`d_model=768`, `num_layers=24`, `expand=2` -> `d_inner=1536`, `d_state=16` default, confirmed by
reading `mamba_v1.py`'s `MODEL_VARIANTS["130m"]` before running). Script:
`diagnostic_v1_chunking.py`, scratchpad-only, not committed. Every run confirmed the clean
`Created device ... with 10157 MB memory` TF init line; `nvidia-smi -i 1` read `18 MiB used`
before AND after every single subprocess invocation (GPU1 genuinely idle throughout, no
contamination); a stray CPU-only `pytest -q` process (not mine, not GPU-touching, confirmed by
`memory.used` staying flat) appeared partway through from an unrelated source and was ignored per
that confirmation. One measurement per process (never reused across configs) to avoid `peak`
counter contamination from undead prior-config variables.

**Diagnostic 1 — isolated A/B, num_layers=2, recompute_grad=False:**

| jit | mode | forward-only peak (MB) | fwd+bwd peak (MB) |
|---|---|---|---|
| auto | precompute | 348.77 | 1539.09 |
| auto | chunked | 132.83 | 991.25 |
| off | precompute | 511.47 | 1680.19 |
| off | chunked | 145.28 | 665.06 |

Saving (precompute -> chunked), forward-only: auto 61.9% ((348.77-132.83)/348.77), off 71.6%
((511.47-145.28)/511.47) — ratio auto/off = 0.86.
Saving, forward+backward: auto 35.6% ((1539.09-991.25)/1539.09), off 60.4%
((1680.19-665.06)/1680.19) — ratio auto/off = 0.59.

Both ratios are within the same order of magnitude (0.59-0.86), NOT a collapse resembling the
prior plan's measured `tf.recompute_grad` result (10.4% vs 43.3%, ratio 0.24, a 4.2x gap). **Verdict
gate (Pre-Mortem STOP-IF #1): PASS** — chunking's XLA-fusion-immunity reasoning is NOT falsified
by this isolated-layer measurement.

**Diagnostic 2 — num_layers slope sweep, jit=auto, backward=True, recompute_grad=True (the shipped
default wrap, held constant across both arms):**

| num_layers | precompute peak (MB) | chunked peak (MB) | absolute saving (MB) | relative saving | per-layer saving (MB) |
|---|---|---|---|---|---|
| 4 | 2739.04 | 2079.48 | 659.56 | 24.1% | 164.9 |
| 8 | 5116.17 | 4249.82 | 866.35 | 16.9% | 108.3 |
| 16 | 9664.39 | 8580.66 | 1083.73 | 11.2% | 67.7 |
| 24 | OOM (attempted 12,899,463,272 B) | OOM (attempted 11,465,781,768 B) | ~1,433,681,504 B (~1.34 GiB) attempted-alloc delta | ~11.1% of the failed-attempt size | n/a (both OOM) |

**Surprise, reported honestly rather than rounded toward the design doc's prediction**: the
`findings/v1-chunking-design.md` "Memory Assessment" section predicted a roughly CONSTANT
per-layer forward-only saving (~192 MiB/layer) independent of `num_layers`, reasoning that this is
a first-pass-forward saving, not an N-1-layers-freed recompute_grad artifact. The MEASURED per-layer
saving instead nearly halves at each doubling of `num_layers` (164.9 -> 108.3 -> 67.7 MB/layer,
n=4/8/16) and the relative saving keeps shrinking (24.1% -> 16.9% -> 11.2%), converging toward the
~11.1% relative gap visible in the two OOM attempted-allocation sizes at n=24. This is NOT the
specific mechanism Pre-Mortem STOP-IF #1 targets (that STOP-IF is about the `jit_compile="auto"`
vs `off` RATIO collapsing, which did NOT happen — see Diagnostic 1) — it is instead exactly the
generalization risk named in Pre-Mortem STOP-IF #2 ("small-scale diagnostic does not generalize to
real 24-layer scale"), showing up ALREADY inside this diagnostic's own slope sweep, before Step 4
even runs. Likely mechanism (not verified further here): `tf.recompute_grad`'s own backward
recomputation increasingly dominates peak memory as `num_layers` grows (more stacked
recompute-boundary crossings retaining their own state), so chunking's fixed-size forward-tensor
elimination becomes a shrinking fraction of a growing total peak.

**Diagnostic 3 — combined arm**: Diagnostic 2's `chunked=True, recompute_grad=True` rows above ARE
the combined arm (chunking + the existing `tf.recompute_grad` wrap together, exactly what shipping
this fix would produce) — no separate 4th configuration was run. A planned extra data point
(`recompute_grad=False` at `num_layers=24`, both chunked and precompute) was DELIBERATELY SKIPPED:
given `num_layers=24, recompute_grad=True` already OOMs for BOTH arms, dropping `recompute_grad`
(which retains strictly MORE backward state, not less) would predictably OOM even harder in both
arms with zero decisive new information — overcomplicating the diagnostic per the plan's own
allowance to skip a diagnostic that "doesn't add decisive information."

**Overall verdict**: Pre-Mortem STOP-IF #1 (the specific gate this step exists to test) does NOT
fire — proceed to Step 2 is the recommendation on THIS gate alone. However, the diminishing-returns
finding above is a live, ALREADY-OBSERVED instance of Pre-Mortem STOP-IF #2's risk class, and Step
4's full-scale measurement should not assume the ~24-62% small-scale forward-only savings will
transfer to anything close to that magnitude at the real 24-layer, `recompute_grad=True` production
configuration — the trend in this diagnostic's own data suggests a real but much smaller
(single-digit-to-low-double-digit percent) saving at full scale, likely enough to shift the ceiling
but not guaranteed to reach batch=8. Step 4 must report the actual number rather than assume it.

## Step 4 raw measurement data | EXECUTE | 2026-09-14

Real harness (`profile_v1.py`, reused unmodified from EXPLORE), real `CausalLanguageModel` +
`Mamba.from_variant("130m")` (24 layers, real wiring, not the isolated `ScanBlock` toy stack from
Step 1), `seq_len=128`, `jit_compile="auto"` (production regime, via `model.fit(...,
steps_per_epoch=1)`), against the SHIPPED chunked `components.py` (commits `d9014c3e0`/`d83d4acdb`,
no further code changes made in this step). `CUDA_VISIBLE_DEVICES=1` only. `nvidia-smi -i 1` read
`18 MiB used` before AND after every single run (5 runs total: batch={4,8,6,5}, plus the idle
checks); every run's log confirmed the clean `Created device ... with 10157 MB memory` TF init
line. Foreground/synchronous, one run at a time, no parallel jobs.

| batch | outcome | peak / attempted-alloc | notes |
|---|---|---|---|
| 4 (pre-chunking baseline, from `findings/clean-baseline-and-dstate-check.md`) | succeeds | 9.497 GiB peak | reference point, not re-run this step |
| 4 (chunked, this step) | succeeds | 7.7695 GiB peak | `after-1-train_step` report |
| 8 (pre-chunking baseline, from `findings/clean-baseline-and-dstate-check.md`) | OOMs | 14.8 GiB attempted | reference point, not re-run this step |
| 8 (chunked, this step) | OOMs | 9,871.5 MiB / 9.8715 GiB attempted (10,599,452,976 B), against a 10,157 MB / 9.919 GiB pool with 1.43 GiB already in-use at failure time | clean TF init line confirmed; process aborted post-OOM with a `bfc_allocator.cc:811` internal check-fail during cleanup — a known TF/XLA post-OOM crash artifact, not a measurement contamination (the `ResourceExhaustedError` and attempted-alloc size were already captured cleanly before the abort) |
| 6 (chunked, this step, exploratory — not in plan's required {4,8} set) | OOMs | small residual allocation (9,437,184 B) fails after near-total pool exhaustion — a fragmentation-tail OOM, not one dominant oversized tensor; `after-forward-only` reported 1.2126 GiB peak before `fit()` failed | XLA's `while` fusion materializes a `f32[128,6,1536,16]` (seq, batch, d_inner, d_state) buffer per the error dump — i.e., XLA IS reconstituting a full-sequence-shaped intermediate for at least one op inside the loop, a partial version of Pre-Mortem #1's named risk (loop-invariant code motion re-fusing something precompute-shaped), though not enough to erase the measured savings (see below) |
| 5 (chunked, this step, exploratory) | succeeds | 8.9565 GiB peak | `after-1-train_step` report; this is the new ceiling |

**Computed improvement vs the pre-chunking baseline** (both numbers measured, not rounded toward
Step 1's small-scale prediction):
- Batch=4: peak dropped 9.497 -> 7.7695 GiB, a savings of 1.7275 GiB (18.19%).
- Batch=8 (both still OOM): attempted-allocation size dropped 14.8 -> 9.8715 GiB, a reduction of
  4.9285 GiB (33.30% smaller attempted allocation) — much larger than Step 1's n=24 isolated-toy
  slope-sweep prediction of ~11.1% (attempted-alloc gap between arms at n=24 in the standalone
  `ScanBlock` diagnostic). The real full model with real embedding/head/optimizer overhead shows
  roughly 3x the relative saving the isolated diagnostic predicted.
- New ceiling: batch=5 succeeds (8.9565 GiB peak, 5.69% below the ORIGINAL batch=4 baseline's 9.497
  GiB), batch=6 OOMs. The target `batch=8` is NOT reached. This is the plan's disjunctive "partial
  success" outcome — the ceiling moved from batch=4 to batch=5 (a one-step, not four-step,
  improvement over the pre-chunking ceiling), while the batch=8 shortfall itself shrank
  substantially (attempted-alloc gap fell from a ~4.9 GiB shortfall to ~1.4-2.0 GiB, depending on
  whether the already-in-use 1.43 GiB is counted against the 9.919 GiB pool or not).

**Pre-Mortem STOP-IF #2 verdict**: FIRES, but in the surprising/positive direction, not the
feared one. The clause reads "disagrees by more than ~2x from what Step 1's slope-sweep
extrapolation would predict, OR shows no improvement at all over the current batch=4 ceiling" — the
second disjunct is false (batch=4's peak dropped 18.19%, and the ceiling did move, to batch=5), but
the first disjunct is TRUE: the batch=8 measured improvement (33.30% attempted-alloc reduction) is
roughly 3x Step 1's own n=24 slope-sweep prediction (~11.1%), which is itself a ">~2x disagreement"
by the stated test — just in the helpful direction (real-scale saving is BIGGER than predicted, not
smaller or absent). Reporting this plainly rather than declining to flag it because the surprise is
welcome: the STOP-IF's literal trigger condition is met, so this is escalated to REFLECT/the
orchestrator to decide next steps, per the plan's explicit instruction not to spin either kind of
surprise and not to attempt further code changes in this step to chase a bigger number. No
production code was touched in this step.

**Surprise**: batch=6's OOM traceback shows XLA's `while` loop fusion still materializes at least
one `f32[seq, batch, d_inner, d_state]`-shaped buffer (`128,6,1536,16`) — i.e., a partial
re-fusion of a precompute-shaped intermediate inside the loop, echoing (in miniature) Pre-Mortem
#1's named "XLA reconstructs the removed precompute via loop-invariant code motion" risk. This did
NOT collapse the measured saving (batch=8's attempted-alloc still dropped 33%), so it does not
retroactively fail Step 1's STOP-IF #1 gate (which tested the auto-vs-off RATIO, not zero
re-fusion) — but it is a partial, real-scale confirmation that the "XLA-fusion-immune" reasoning in
`findings/v1-chunking-design.md` was optimistic in degree (chunking reduces, but does not fully
eliminate, XLA's tendency to materialize sequence-shaped intermediates under `jit_compile="auto"`).

## D-003 | EXECUTE | 2026-09-14
**Anchor-Refs**: `src/dl_techniques/models/language/mamba/components.py:480` (docstring pointer),
`:518` (`# DECISION plan-2026-09-14T042205-a11f6af3/D-003` anchor in `body()`).

**Context**: `_selective_scan`'s two full-sequence `deltaA`/`deltaB_u` precompute tensors
(`(batch, d_inner, seq_len, d_state)` each, materialized before the `while_loop` started) were
identified in EXPLORE as a genuine, root-cause-level memory lever for v1 specifically — distinct
from the prior plan's `tf.recompute_grad` checkpointing fix, which addresses backward-pass
retention, not this forward-only allocation. D-001/D-002 record the decision to attempt this,
gated by an early falsification diagnostic, with v2 explicitly out of scope.

**Implemented** (Step 2, commit `d9014c3e0`): `deltaA_t`/`deltaB_u_t` are now computed per-timestep
inside the `while_loop` `body()`, sliced from the already-cast `delta`/`u`/`B` tensors (preserving
D-044's dtype discipline), mirroring the existing `C[:, :, t]` slicing precedent in the same loop.
The two full-sequence precompute tensors are eliminated entirely — this is a computation-graph
change (what is computed, not just when), not an additional checkpointing wrap. It composes with,
and does not replace, the existing `tf.recompute_grad` wrap from the prior plan.

**Correctness** (Step 3, commit `d83d4acdb`): forward pass is bit-exact against the old
full-precompute form (`max_abs_diff = 0.0`, verified on a real `MambaLayer`'s actual sublayer
outputs, not a CPU toy tensor). Gradients agree within a tolerance derived from
`tests/numerics.py::reassociation_atol()` (~1e-13 absolute on `A_log`) — this is float32
reduction-order noise from `tf.recompute_grad`'s backward re-execution visiting the same einsum
contractions in a different order, not a numerical defect; the tolerance was re-derived from the
noise source, not hand-loosened to pass. 194/194 tests pass across the full `test_mamba/` suite
(Step 5, commit `8094ad6b7`), zero regressions.

**Measured outcome** (Step 4, commit `e172f7cbe`, real `CausalLanguageModel` +
`Mamba.from_variant("130m")`, 24 layers, `seq_len=128`, `jit_compile="auto"` production regime, 12GB
GPU):

| batch | pre-chunking (prior plan's baseline) | chunked (this plan) | change |
|---|---|---|---|
| 4 | succeeds, 9.497 GiB peak | succeeds, 7.7695 GiB peak | -18.19% peak memory |
| 5 | (not separately measured pre-chunking; batch=4 was the pre-chunking ceiling) | succeeds, 8.9565 GiB peak | **new ceiling** |
| 6 | — | OOMs (fragmentation-tail; XLA still re-fuses a `f32[128,6,1536,16]` buffer inside the loop) | — |
| 8 | OOMs, 14.8 GiB attempted | OOMs, 9.8715 GiB attempted | -33.30% attempted allocation, still OOMs |

**The original batch=8/seq=128 target is NOT reached** — stated plainly, not rounded up. The ceiling
moved by exactly one batch step (4->5), not to the target. This holds even though the batch=8
attempted-allocation shortfall shrank substantially (33.30%, roughly 3x Step 1's own small-scale
`num_layers=24` isolated-diagnostic prediction of ~11.1% — a real, positive surprise, not a
shortfall; Pre-Mortem STOP-IF #2 fired in the helpful direction, escalated and accepted per Step 4's
raw data).

**Combined history across both plans** (for a reader who only sees this plan's own delta): v1's
ceiling has moved batch=2 (original) -> batch=4 (prior plan's `tf.recompute_grad` wrap) -> batch=5
(this plan's chunking). Two independent, additive levers, two partial improvements, target still
not reached.

**Decision**: Ship the chunked `_selective_scan` — it is a pure internal computation-graph
optimization with bit-exact forward numerics and gradient-correctness re-verified, no behavior
change, no new abstraction, and a real (if partial) memory improvement, additive to the existing
`tf.recompute_grad` wrap.

**Trade-off**: A more invasive computation-graph change — one that touches the actual computed
values' code path (what is computed), not just backward timing (when it is recomputed) — **at the
cost of** requiring a stronger correctness proof (forward-numerics-identical AND
gradient-identical tests, not just a serialization/shape check) than the prior plan's
checkpointing-only fix needed. This is a cost paid once, in review/verification burden, not a
recurring runtime or quality cost — unlike the prior plan's declined XLA-off option, which traded
memory for a 23-31x recurring wall-clock/step penalty. Worth paying: the verification cost is
already discharged (Step 3, 194/194 tests), and the resulting fix has no disclosed downside beyond
that one-time proof burden.

**v2's `d_state` reduction remains a NAMED, DEFERRED follow-up**, not attempted by this plan
(out of scope per D-002, the orchestrator's own scope decision) — v2 (`Mamba2Layer._ssm_scan`) has
no analogous full-sequence precompute to chunk (confirmed negative result, `findings/v2-chunking-design.md`);
its only available lever is a capacity/quality trade-off (reducing `d_state`) requiring its own
future user decision, exactly as the prior plan's declined XLA-off option was named-and-deferred
before this plan investigated one of its two named follow-ups. v2's OOM ceiling (batch<=2 at
`seq_len=128`) remains unimproved by either this plan or the prior one.

**Reasoning**: A negative-adjacent partial result ("batch=5, not batch=8") is reported honestly
per `plans/LESSONS.md`'s standing rule that a negative or partial result is a deliverable when
it is localized and the real numbers are on record — not rounded toward the original target, and
not silently omitted. Both the shipped fix's real benefit and its real shortfall are stated in the
same entry so a future reader (or a future plan targeting v1's remaining gap, or v2's `d_state`
lever) has the full, honest picture rather than just this plan's own delta.

## Step 5 raw result | EXECUTE | 2026-09-14

Confirmed the current file list under `tests/test_models/test_mamba/` before running anything
(`ls`): exactly the 10 files plan.md names — `test_build_and_reload.py`, `test_components.py`,
`test_dt_proj_survives_stateless_build.py`, `test_head_pooling.py`, `test_mamba2_forwarded_knobs.py`,
`test_mamba2_group_routing.py`, `test_mamba_v1.py`, `test_mamba_v2.py`,
`test_norm_before_gate_is_reachable.py`, `test_norm_identity_and_build_idempotence.py` (plus
`__init__.py`) — nothing missing, nothing extra, no assumption needed.

Ran `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_models/test_mamba/ -vvv` (CPU
only, per this step's own command, not GPU-gated). Result: **194 passed, 1 warning (unrelated
`distutils` deprecation notice), 0 failed, in 128.30s**, exit code 0. This includes both new test
classes added by Step 3 (`TestMambaLayerChunkedScanForwardAndGradientNumerics`'s 3 tests, and the
pre-existing `TestMambaLayerCheckpointedScanGradients`'s 3 tests) alongside the full pre-existing
v1/v2/norm/build-idempotence/export suites — zero regressions anywhere in the package, not just in
the file directly touched by chunking.

Re-confirmed (not just trusted from plan.md) the zamba2/hnet zero-dependency assumption via
`grep -n "Mamba" src/dl_techniques/models/language/zamba2/layers.py
src/dl_techniques/models/language/hnet/components.py`: every match in both files is either a
docstring/comment prose reference to "Mamba-2"/"Mamba2" or an actual import/usage of
`Mamba2Layer`/`Mamba2ResidualBlock` from `mamba.components_v2` — zero occurrences of the bare
`MambaLayer` class name or any import from `mamba.components` (v1) in either file. This MATCHES
plan.md's assertion exactly (Success Criteria row 7, Verification Strategy row 7): the plan's
scoping assumption holds, no scope-drift finding to escalate, and the reduced regression scope for
zamba2/hnet (no separate suite re-run required) stands as originally reasoned in Step 5's own text
and this plan's Assumptions section.

No code changes were made in this step (confirmed: `git status` shows no changes under `src/` or
`tests/`). No surprises beyond the ones already recorded in Steps 1 and 4's raw-data sections above;
this step's own result is a clean, unsurprising confirmation of both the regression scope and the
v1-only import-graph boundary.
