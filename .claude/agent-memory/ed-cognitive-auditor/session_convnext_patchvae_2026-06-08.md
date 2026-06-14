---
name: session-convnext-patchvae-2026-06-08
description: ConvNeXtPatchVAE P0->P4 full audit — correlated evidence; architecture-exists/quality-works conflation; C17 scope discharge; LR over-precision; MC false precision on analogical inputs; level naming inversion; PP2 persistently open
metadata:
  type: project
---

Session: analysis_2026-06-08_3ec50266. P0->P1: MODERATE CONCERN, S1 re-trigger recommended. P2: LOW CONCERN, LR magnitude corrections required. P3/P4: MODERATE CONCERN, 5 issues.

**Key traps found (P0-P2):**

1. CORRELATED-EVIDENCE OVER-COUNTING: obs_001 (static weight shapes), obs_002 (KL/recon flatness on untrained model), obs_003 (GRN output stability on untrained model) all measure the same underlying invariant — "weights are Hp*Wp-independent." Treated as independent LRs; all are conditional on obs_001. Corrected H1 posterior ~0.72 not 0.80.

2. ARCHITECTURE-EXISTS != QUALITY-WORKS (H3): H3 is a generative-quality claim. Phase_1 suggested LR=2.5 support from architectural inspection of deleted model. Deleted model's ops being convolutional proves feasibility (H2 evidence), not that the trained model will recover aggregate-posterior coverage (H3 claim). HIGH risk — must not let H3 exceed 0.50 on architectural evidence alone.

3. MOTIVATED SCOPE DISCHARGE: C17 (single-resolution training distribution) confirmed as out-of-S driver in D-001 but received zero hypotheses.json update and no predictions_pending entry. GRN probe (obs_003) answered shape question only, not trained-scale calibration drift question. H6 posterior should be ~0.55 not 0.455 until C17 is tested on a trained checkpoint.

4. GRN PROBE TOO NARROW: Used untrained model + i.i.d. N(0,1) input. The outsider steelman (phase_0_7.md M4) explicitly warned about trained gamma/beta calibration at training resolution. The probe did not address this; "GRN BENIGN" conclusion overstated.

**Pattern match with prior sessions:** S1 trigger suppressed via D-001 rationalization appeared also in LLM separation session 2026-06-04. This is a recurring pattern — administrative scope expansion via decisions.md being used to discharge H_S_prime without empirical falsification.

**P2 traps (LOW CONCERN overall, 4 LR calibration issues):**

5. SINGLE-CHECKPOINT OVER-REFUTATION (H7): LR=0.2 from one checkpoint, one dataset. Same-source control confirms resolution-causation but cannot isolate miscalibration vs. legitimate content-scale sensitivity at different patch grids. Correct LR to 0.3 (H7 posterior 0.23 not 0.167).

6. H6 PARTIAL DOUBLE-COUNTING: P1 exit already partially credited C17 as open (LR=1.7 audit correction). P2 LR=2.0 for the confirming obs_006 measurement double-counts the C17 credit. Additionally, obs_006 confirms drift EXISTS but does not isolate training-recipe baking vs. legitimate content-scale sensitivity. Correct LR to 1.5 (H6 posterior ~0.58 not 0.648).

7. H3 CROSS-ARCHITECTURE EXTRAPOLATION: The 97% conv-k4 result and 61-75% external diffusion recovery are from EXTERNAL PRIOR EXPERIMENTS, not the conditional-KL hierarchy. LR=3.0 is too strong for "mechanism confirmed, in-model efficacy pending." Correct LR to 2.0 (H3 posterior ~0.69 not 0.738). Recurring trap from P1 — same architecture-exists/quality-works pattern.

8. H8/H11 ORTHOGONALITY ASSERTED (metric argument, not causal experiment): structure_gap metric blindness to within-patch content is used to claim causal independence of axes A and B. CMD-2 acknowledges weak A-B coupling but LRs 2.5/2.0 don't reflect this caveat. Correct to LR=1.8/1.5.

**Recurring pattern confirmed P1-P2:** H3 architecture-exists/quality-works conflation persisted from P1 into P2 despite the P1 audit correction. The update mechanism partially worked (H3 held back below 0.75) but the cross-architecture extrapolation trap recurred in the LR magnitude.

**P3/P4 traps (MODERATE CONCERN, 5 issues):**

9. EMERGENCE EXTRAPOLATION PRESENTED AS MEASUREMENT (P4 HIGH): The z2 drift numbers (z2 kappa ratio 0.608 at r=2x, 1.24x MORE drift than z1, 31% relative miscalibration, [43%,66%] improvement interval) are derived by applying LAW1 (fit on the FINE encoder, image-resolution sweep) to the COARSE z2 encoder as if they share the same drift law. This is an analytic extrapolation assumption, not a measurement. The "1.24x MORE drift" figure is also arithmetically inconsistent with the cited kappa ratios (0.795 and 0.608 give either 1.91x in absolute degradation or 1.308x in ratio — not 1.24x). Phase 5 must label all z2 emergence numbers as extrapolated.

10. MC FALSE PRECISION (P4 MODERATE): N=50,000 MC produces [43%,66%] CI with ~0.1pp sampling error, but the input bounds (improvement_fraction = Uniform(0.47, 0.75)) rest on cross-architecture analogies from external priors, and the "50% transfer" coupling parameter has no empirical grounding. The tight MC CI manufactures precision that the epistemic uncertainty in inputs does not support. Coarsen the design expectation to ~30-75% in Phase 5.

11. C17 HIERARCHICAL SUB-PROBLEM NOT DISTINGUISHED (P4 MODERATE): The analysis characterizes C17 as "multi-res training fixes calibration drift" but the coarse encoder's structural r_eff=2 offset (a consequence of two-encoder design, not deployment-time resolution variation) may not be fixed by flat multi-res jitter unless both encoder paths see the same effective resolution range explicitly. C17b (hierarchical coarse-encoder structural offset) is open. Phase 5 must not present multi-res training as fully resolving C17.

12. LEVEL NAMING INVERSION (P4 HIGH — implementation risk): Phase 4 TASK 1 labels z1=fine, z2=coarse. Phase 4 TASK 5 labels L1=coarse, L2=fine, with z1 production yielding (B, Hp/2, Wp/2) = coarse. The two conventions are inverted within the same document. Will cause implementation bugs if not resolved before Phase 5.

13. PP2 PERSISTENTLY OPEN (P1-P4 MODERATE-HIGH): H9 has posterior=0.30 and evidence=[] across all four phases. H3 efficacy (the primary delivery commitment registered in D-007) has zero trained-model evidence. The Phase 5 design report must label H3 as "mechanism confirmed, efficacy unconfirmed (PP2 outstanding)."

**Confirming/disconfirming ratio pattern (ALL PHASES):** P1: 4:1 (flagged), P2: 5:1 (flagged), P3+P4: 6:1 (borderline RED FLAG). Sustained 3-phase confirming-dominated trail without a genuinely disconfirming empirical result for any lead hypothesis (H1, H2, H3 mechanism).
