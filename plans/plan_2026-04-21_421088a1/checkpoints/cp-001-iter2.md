# Checkpoint cp-001 — iter-2 restore point

**Created**: 2026-04-21
**Purpose**: Pre-iter-2 snapshot. Restore point if iter-2 goes off the rails.
  cp-000-iter1.md remains the nuclear-fallback (pre-iter-1 state).

## Git State (restore point)
- Branch: `main`
- Commit HEAD at iter-2 start: `1c596fdef85488e0af661a9e7d113e1e151f712b`
  (subject: `Add missing tests/test_callbacks/__init__.py`)
- Working tree: plan/docs modifications present (CLAUDE.md, plans/*.md, research/*)
  but all iter-1 src/ + tests/ for `video_jepa/` are committed and clean as of `1c596fd`.
- Known iter-1 tests green: 29/29 in `tests/test_models/test_video_jepa/`.

## Scope-at-risk (iter-2)
- 1 new file: `src/dl_techniques/models/video_jepa/masking.py`
- 4 modified files:
  - `src/dl_techniques/models/video_jepa/config.py`
  - `src/dl_techniques/models/video_jepa/model.py`
  - `src/train/video_jepa/train_video_jepa.py`
  - `tests/test_models/test_video_jepa/test_video_jepa.py`

## Restore procedure (iter-2 partial revert — keeps iter-1)
```bash
# Revert just video_jepa scope back to iter-1 end-state (1c596fd):
git checkout 1c596fd -- \
  src/dl_techniques/models/video_jepa/ \
  src/train/video_jepa/ \
  tests/test_models/test_video_jepa/
# Remove any new files created during iter-2:
rm -f src/dl_techniques/models/video_jepa/masking.py
```

## Nuclear option
If iter-2 bloats >2× scope or P5–P8 fires AND a partial revert is insufficient,
defer to cp-000-iter1.md procedure to revert iter-1 + iter-2 entirely.

## Success criteria reminder (iter-2)
C8 mask cardinality · C9 per-sample independence · C10 mask loss finite ·
C11 disabled-matches-iter1 regression · C12 serialize-with-masking ·
C13 all 29 iter-1 tests still green · C14 smoke run: both losses decrease.

## Autonomy Leash
2 fix attempts max per failing step. P5–P8 trigger REFLECT, not silent workarounds.
