# Checkpoint cp-000-iter1

## Git State
- Commit hash (restore point BEFORE any changes in this iteration): `8493641`
- Branch: `main`
- Uncommitted tracked modifications (pre-existing, unrelated to this plan):
  - `M CLAUDE.md`
  - `M research/2026_clifford_vlm.md`
  - `M src/train/cliffordnet/train_coco_multitask.py`
- Untracked files present (pre-existing): `.claude/`, `.idea/`, `PLAN_CONTINUE.md`, `research/papers/cliffordnet_extensions/*.{aux,out,pdf}`, `results/`, `src/dl_techniques/callbacks/coco_multitask_visualization.py`

## Purpose
Nuclear fallback for iteration 1 of LeWM port. Restore via:
```
git reset --hard 8493641
# re-apply pre-existing unstaged mods if needed
```

## Scope covered
All work in this plan (Steps 1-10). LeWM has 13 new files, no edits to existing files, so revert is a clean `git clean -fd src/dl_techniques/{layers/adaln_zero.py,regularizers/sigreg.py,models/lewm,datasets/pusht_hdf5.py} src/train/lewm tests/test_layers/test_adaln_zero.py tests/test_models/test_lewm.py` plus `git reset --hard 8493641`.

## Created
2026-04-21 (iteration 1, pre-Step-1)
