---
name: session-cliffordclip-2026-05-31
description: CliffordCLIP feasibility session — trap patterns observed across P0, P1, and P2 audits
metadata:
  type: project
---

Session dir: analyses/analysis_2026-05-31_1133d05f

**Recurring trap: RUNNABLE != USEFUL conflation.**
Phase 1 collected 7 structural/mechanical confirming evidence items for H1 and 0 quality-relevant items. The LR 4.0 strong-confirm for H1 applies only to structural feasibility; utility feasibility is entirely unconfirmed. This trap is latent across any session where the subject is a model architecture rather than a trained model -- "instantiates + tests pass" is always confirmable early and always structurally insufficient for "useful."

**Why:** The task statement said "create a useful CLIP." H1's statement included "runnable" but the utility sub-claim was never disaggregated. Phase 1 boundary mapping by design targets I/O and composability, not quality -- so the phase structure itself creates this blind spot unless the hypothesis is explicitly split.

**How to apply:** In any future session where a hypothesis combines a structural claim ("runnable/complete") with a quality claim ("useful/effective"), audit whether evidence has been collected for BOTH sub-claims. Flag if quality-claim evidence count is 0.

---

**H4 adversarial discharge pattern: shape/serial easy, correctness hard.**
The H4 adversarial hypothesis was refuted with LR 0.15 after checking only instantiation, serialization, and shape-passing. The harder sub-claims (mathematical correctness of the geometric product, causal leakage in CausalCliffordNetBlock, gradient flow under contrastive loss) were not scheduled. The D-005 U4 discharge was mechanically complete but substantively shallow.

**Why:** The D-003 mandate listed concrete sub-tasks (grep tests, instantiate, round-trip) that all targeted the easiest-to-verify aspects. The adversarial hypothesis was effectively directed at the most confirmable sub-claims.

**How to apply:** When auditing an adversarial hypothesis discharge, check whether the probes were designed to challenge the HARDEST sub-claim, not just the most accessible one. An adversarial hypothesis that was "strongly disconfirmed" by shape-passing alone is a red flag.

---

**C18/C20 as persistent unweighted risks.**
Raw alt-text (C18) and no pretrained backbone (C20) were confirmed in Phase 1 but never received posterior updates on any hypothesis. They are cited as constraint labels but absent from the causal graph. This is a recurring pattern: named constraints that don't get formalized as causal nodes drift to zero influence on posteriors.

**Why:** Phase 1 boundary mapping records observations but the hypothesis-update step was deferred (posteriors in hypotheses.json still equal priors). By the time posteriors are committed, the mechanical-confirmation evidence has accumulated and quality-risk evidence has not.

**How to apply:** Track whether any named constraints (C-series) that affect the "useful" reading of a hypothesis are absent from the causal graph going into Phase 2. Flag them in the out-of-frame report so they get causal-node status in Phase 2, not just footnote status.

---

**P2 pattern: posterior ledger diverges from prose narrative.**
In P2, D-009 explicitly listed "H7(NULL) confirm LR 2.0" as an update to apply, but hypotheses.json H7 posterior remained at 0.50 (the prior). EXP-3's mechanistic finding was correctly hedged in prose (init-only, R1 loop dormant) but the hedge only exists in text -- the ledger was not updated. This creates a systematic risk: a narrative can accumulate informal strength without ever passing through the Bayesian gate, making later phases implicitly over-weighted toward the prose conclusion.

**Why:** The phase template records "suggested LR" in the phase output file but the actual posterior application is a separate orchestrator step. If that step is deferred, the gap grows.

**How to apply:** At the start of each audit, check whether ALL "suggested LR" entries from the previous phase's hypothesis digest have been applied to hypotheses.json posteriors. Any that are still at their pre-phase prior value are unapplied updates -- flag them before the next phase proceeds.

---

**P2 pattern: H8 'graceful degradation' over-read when the baseline is itself weak.**
EXP-1 confirmed graceful degradation (Clifford head at gamma=0 == plain CLIP anchor). The analysis correctly notes this also bounds the upside. However, neither phase_2.md nor D-009 hedged that the 'plain CLIP' floor is from-scratch on CC3M without a pretrained backbone (C20), which may produce very weak absolute R@K. "Downside bounded" was stated without bounding where the floor actually is in absolute terms.

**Why:** The H8 confirmation is structurally valid but the phrase "downside bounded" is misleading in isolation because it implies a floor that is at minimum acceptable. The floor quality is itself an open question until training.

**How to apply:** Whenever a hypothesis confirmation amounts to "worst case = baseline X," check whether baseline X has itself been bounded in absolute terms. If not, the confirmation only bounds the relative ordering (this >= X) and says nothing about whether X is acceptable. Flag if the language implies absolute safety.

---

**P2 pattern: C20 backbone-init entered causal graph but A/B design lacks signal-floor pre-condition.**
C20 (from-scratch backbone) is now a causal node in the DAG (progress from P1 where it was only a constraint label). The sensitivity ranking correctly identifies it as the #2 headwind and notes H7 becomes unobservable if C20 fails. However, ACTION-B (the H7 A/B run) was not conditioned on first verifying that plain-CLIP from-scratch clears a minimum useful-signal threshold. If the from-scratch run produces near-zero R@K, the A/B comparison is uninformative about Clifford specifically.

**How to apply:** For any A/B experiment that tests a head/architecture modification, check that the experimental design includes a pre-condition confirming the control arm (plain baseline) reaches a non-trivial signal floor. Without this, a negative A/B result is ambiguous between "Clifford doesn't help" and "from-scratch baseline is too weak to observe any signal."

---

**P3 pattern: per-epoch speed measured but epochs-to-useful is the unknown — "wall-clock disconfirmed" is anchoring on the fast number.**
Phase 3 measured 1,176 img/sec on GPU1 vs the C19 Phase 0.7 assumption of ~200 img/sec (~6x faster). The analysis declared C19 "disconfirmed" (LR=0.2 applied to H_scope) and labeled wall-clock "not the binding constraint." But the epoch count range (15-300, a 20x spread) is the actual dominant unknown. The 0.2-4.6 day range is computed as (per-epoch speed) x (epoch estimate), and the epoch estimate has no published precedent for fully-from-scratch CLIP on 3M noisy pairs. The "under 1 day / one weekend" framing anchors on the optimistic 15-30 epoch scenario.

**Why:** The per-epoch measurement is concrete and easy to communicate; the epoch-count uncertainty is abstract and harder to communicate. Fast per-step times make compute sound solved, which creates an optimism anchor. The two uncertainties (speed, epoch count) are treated asymmetrically: speed gets a measurement, epoch count gets an analytic table with no bottom-line uncertainty propagation to the wall-clock claim.

**How to apply:** Whenever a wall-clock estimate combines a measured component (steps/epoch) with an analytic component (epochs-to-useful), flag if the stated range is presented as a measured result. The range is only as narrow as the less-certain component. If epochs-to-useful has a 20x uncertainty, the wall-clock range has a 20x uncertainty -- not a 5x range.

---

**P3 pattern: double-parametric data verdict (0.5x discount AND 3M floor) treated as binary fact.**
Phase 3 data-sufficiency section applies a 0.5x caption quality discount (cited as "Li et al. 2022 quality ablations," unverified in session files) and a 3M floor threshold (described as "research_scout threshold," also parametric). Both parameters are llm_parametric. The conclusion "BELOW THRESHOLD" appears in the exit-gate checklist as a binary pass/fail, losing the double-parametric uncertainty. The verdict is directionally robust but the precision implied by the binary label is not warranted.

**How to apply:** When a sizing verdict depends on two or more llm_parametric thresholds, the exit-gate label should carry an analytic qualifier (e.g., "LIKELY BELOW THRESHOLD (analytic)") rather than a bare binary. Binary verdicts are appropriate only when at least one threshold is a measured or published-and-cited number.

---

**P3 pattern: H1 composite hypothesis dilutes a genuine quality-floor disconfirm.**
H1 combines H1.a ("runnable after fixes") and H1.b ("fix-and-train -> useful R@K"). H1.a is near-confirmed by Phase 1-3 mechanical evidence. H1.b is threatened by C20 (from-scratch), C18 (noisy captions), and H9 (RF ceiling) — and has no published analog at this scale. D-010 asked for a genuine H1 disconfirm in Phase 3; Phase 3 applied LR=0.75 (mild disconfirm), which is technically correct but understated because the composite posterior dilutes the quality-specific disconfirm with the structural confidence. H1 went from 0.529 to 0.458 — the quality-sub-claim posterior if tracked separately would be much lower.

**How to apply:** When a hypothesis combines a structural feasibility sub-claim (confirmable by code inspection) with a quality/utility sub-claim (requires training), flag if the evidence trail mixes sub-claim types. The composite posterior will systematically overstate quality confidence as long as structural evidence accumulates faster. Recommend sub-claim split when the structural sub-claim is near-confirmed and the quality sub-claim remains open.
