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
