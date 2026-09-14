# Checkpoint 000 (iteration 1)

## Created: Before any EXECUTE-phase changes for plan-2026-09-14T042205-a11f6af3 (nuclear fallback)
## Git State: commit 36127802f  ← commit BEFORE these changes (restore point)
## Files That Will Change:
- (Step 1 is measurement-only/scratchpad-only — no `src/`/`tests/` files change in this step)
- Later steps (2-6, conditional on Step 1's verdict gate) would modify:
  - src/dl_techniques/models/language/mamba/components.py (modify)
  - tests/test_models/test_mamba/test_mamba_v1.py (modify)
  - plans/plan-2026-09-14T042205-a11f6af3/decisions.md (append)

## Lockfiles snapshotted:
- none (no package manager touched)

## Rollback:
git checkout 36127802f -- src/dl_techniques/models/language/mamba/components.py tests/test_models/test_mamba/test_mamba_v1.py
# No lockfile/manifest touched by this plan; no reinstall step required.
