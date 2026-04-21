# Checkpoint cp-000 — iter-1 nuclear fallback

**Created**: 2026-04-21
**Purpose**: Pre-EXECUTE snapshot. Restore point if iter-1 goes off the rails.

## Git State (restore point)
- Branch: `main`
- Commit HEAD before EXECUTE: `f7f1e812cbf59caa4d9403f5807608583247a572`
  (subject: `[iter-1/step-8] Detection viz + training_curves detection group + CSV fix`)
- Working tree: clean w.r.t. `src/` and `tests/` for video-jepa scope (no video-jepa files exist yet). Unrelated pre-existing local modifications (CLAUDE.md, research/*, etc.) are not part of this plan and will not be touched.

## Restore procedure
```bash
# Hard reset only src/ + tests/ areas we are about to modify (none currently exist, so restore = delete newly-created files):
rm -rf src/dl_techniques/models/video_jepa/
rm -f src/dl_techniques/datasets/synthetic_drone_video.py
rm -rf src/train/video_jepa/
rm -rf tests/test_models/test_video_jepa/
# Then:
git checkout f7f1e81 -- .   # if any tracked file was accidentally modified
```

## Scope-at-risk
- 11 new files (model package, dataset, training, tests)
- 0 existing-file modifications (per plan.md § Files To Modify)

## Nuclear option
If iter-1 bloats >2× scope or Pre-Mortem P1–P4 fires and a PIVOT isn't sufficient, revert with the procedure above and return to PLAN.
