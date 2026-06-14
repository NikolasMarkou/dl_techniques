---
name: session-llm-separation-2026-06-04
description: P1 audit of analysis_2026-06-04_10057064 (LLM memory/intelligence separation); key traps found, S1 trigger issued
metadata:
  type: project
---

Session: analysis_2026-06-04_10057064
Topic: Decoupling parametric knowledge (memory) from reasoning in transformer LLMs.
Audit phase: Post-P1 (boundary mapping)
Overall assessment: MODERATE CONCERN trending HIGH

Key traps found:

1. HYPOTHESIS STAGING FAILURE (HIGH): Phase 0.7 scope auditor recommended seeding H_SCOPE_1..6. Only 3 of 6 were absorbed. H_SCOPE_4 (optimization regime / NTK vs muP) and H_SCOPE_6 (tokenizer / representation geometry / superposition) absent from hypotheses.json. Both are mechanistically distinct from existing hypotheses and causally upstream of the surgical-edit mechanisms (H3a). S1 trigger issued.

2. TRIGGER AVOIDANCE RATIONALIZATION (MODERATE): D-001 deferred the H9 ([H_S_prime] spectral entanglement) update explicitly to avoid firing the S1 scope-reopen trigger. The stated rationale was "scope was already expanded at P0.7" — but the expansion was incomplete (H_SCOPE_4/6 missing). The deferral suppressed the very mechanism that would have caught the incomplete expansion.

3. SINGLE-SOURCE CORPUS BIAS (MODERATE): All spectral criticality evidence (alpha~2 / HTSR) traces to a single lab document (research/2026_intelligence.md authored by the same lab). H9 posterior is systematically inflated if this document overstates the HTSR claim relative to the wider literature. Recommend external citation check before Phase 2 causal graph.

4. ANCHORING ON H1 AS BEST-CASE (MODERATE): H1 RAG/kNN-LM rated EXCELLENT hides a structural coupling: base LM has parametric contamination from pretraining on the same knowledge it retrieves. This is noted as a caveat but not incorporated as a disconfirm. "Zero weight perturbation" is being conflated with "clean separation of effective memory."

5. H_IC NARRATIVE SMOOTHING (LOW-MODERATE): H_IC (in-context/KV-cache) rated EXCELLENT on grounds of "no weight perturbation." Eviction is indiscriminate (FIFO), attention dilution begins before hard limit, O(L·d²) cost at scale. Operational plug-and-play quality is GOOD, not EXCELLENT.

Recurring pattern across sessions:
- Easy sub-claims used to discharge hypotheses before the hard sub-claims are tested (compare CliffordCLIP: "code runs" = model useful). Here: "no weight perturbation" = clean memory separation.
- Scope auditor recommendations silently absorbed partially; remainder dropped without logged rationale.

Out-of-Frame Report issued: YES (H_SCOPE_4, H_SCOPE_6)
S1 trigger recommended: YES

Audit file written to: analyses/analysis_2026-06-04_10057064/phase_outputs/cognitive_audit_p1.md
