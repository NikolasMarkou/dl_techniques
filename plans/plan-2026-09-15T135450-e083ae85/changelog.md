# Changelog
*Append-only per-edit ledger. One line per file edit. Owner: ip-executor (writes). Reader: ip-reviewer at REFLECT.*
*Field order: `UTC | iter-N/step-M[.K] | commit | path | op | radius | D-NNN-or-dash | reason`. Field shapes are defined once, in `CHANGELOG_SPEC` (scripts/schema.mjs) — read the spec, not a copy.*
*See references/blast-radius.md for radius scoring. Decision-ref optional — `-` means no `# DECISION` anchor governs this edit.*

2026-09-15T14:13:55Z | iter-1/step-1 | 58cd36ceb | src/dl_techniques/layers/tabular/tabm_mlp_block.py | EDIT(+32,-0) | radius:LOW(1) | D-001 | strengthen Idiom-E KEEP exception documentation
2026-09-15T14:30:00Z | iter-1/step-2 | uncommitted | plans/plan-2026-09-15T135450-e083ae85/decisions.md | EDIT(+7,-0) | radius:MED(3) | D-008 | re-confirm Idiom D no-simplification, verification-only, no source change
2026-09-15T14:20:53Z | iter-1/step-3a | e84105bf7 | src/dl_techniques/layers/fusion/multimodal_fusion.py | EDIT(+18,-0) | radius:LOW(0) | D-003 | document raw keras.activations.get as intentional
