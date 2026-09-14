# Changelog
*Append-only per-edit ledger. One line per file edit. Owner: ip-executor (writes). Reader: ip-reviewer at REFLECT.*
*Field order: `UTC | iter-N/step-M[.K] | commit | path | op | radius | D-NNN-or-dash | reason`. Field shapes are defined once, in `CHANGELOG_SPEC` (scripts/schema.mjs) — read the spec, not a copy.*
*See references/blast-radius.md for radius scoring. Decision-ref optional — `-` means no `# DECISION` anchor governs this edit.*
2026-09-14T05:00:16Z | iter-1/step-1 | uncommitted | plans/plan-2026-09-14T042205-a11f6af3/checkpoints/cp-000-iter1.md | CREATE(+18) | radius:LOW(0) | - | nuclear fallback checkpoint before iter-1 EXECUTE
2026-09-14T05:00:16Z | iter-1/step-1 | uncommitted | plans/plan-2026-09-14T042205-a11f6af3/decisions.md | EDIT(+64,-0) | radius:MED(5) | - | append Step 1 raw GPU diagnostic data (verdict gate PASS)
2026-09-14T06:10:00Z | iter-1/step-2 | d9014c3e0 | src/dl_techniques/models/language/mamba/components.py | EDIT(+31,-11) | radius:MED(4) | D-003 | chunk deltaA/deltaB_u precompute into while_loop body per-step
2026-09-14T07:05:00Z | iter-1/step-3 | uncommitted | tests/test_models/test_mamba/test_mamba_v1.py | EDIT(+183,-0) | radius:MED(5) | D-003 | add forward-numerics and per-weight gradient-correctness tests for chunked scan
2026-09-14T05:28:32Z | iter-1/step-4 | uncommitted | plans/plan-2026-09-14T042205-a11f6af3/decisions.md | EDIT(+40,-0) | radius:MED(4) | - | append Step 4 full-scale measurement data (batch=4/5/6/8, STOP-IF #2 fires positively)
2026-09-14T08:35:00Z | iter-1/step-5 | uncommitted | plans/plan-2026-09-14T042205-a11f6af3/decisions.md | EDIT(+27,-0) | radius:MED(4) | - | record Step 5 full-suite run (194 passed) and zamba2/hnet zero-dependency re-confirmation
2026-09-14T08:35:00Z | iter-1/step-5 | uncommitted | plans/plan-2026-09-14T042205-a11f6af3/progress.md | EDIT(+4,-2) | radius:LOW(1) | - | mark step 5 complete in progress.md
2026-09-14T09:00:00Z | iter-1/step-6 | uncommitted | src/dl_techniques/models/language/mamba/components.py | EDIT(+9,-2) | radius:MED(3) | D-003 | document measured chunking outcome (batch=4->5, batch=8 not reached)
2026-09-14T09:00:00Z | iter-1/step-6 | uncommitted | plans/plan-2026-09-14T042205-a11f6af3/decisions.md | EDIT(+65,-0) | radius:MED(4) | D-003 | write final D-003 decisions.md entry with anchor-refs backlink
2026-09-14T09:40:00Z | iter-1/step-6.1 | d440ec6cc | src/dl_techniques/models/language/mamba/components.py | EDIT(+3,-1) | radius:LOW(2) | - | mark first docstring note superseded by the batch=5 ceiling
2026-09-14T09:41:00Z | iter-1/step-6.1 | d440ec6cc | tests/test_models/test_mamba/test_mamba_v1.py | EDIT(+29,-6) | radius:LOW(2) | D-005 | fix vacuous per-weight gradient tolerance for near-zero-magnitude weights
