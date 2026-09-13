"""Pattern-3 (subword CLM) trainer for the Qwen3Next model package.

See ``train_qwen3_next.py`` for the entry point and ``common.py`` for the
config/pipeline it drives. **Scope: `Qwen3Next` only.** A new sibling
package to ``src/train/qwen/`` rather than a colocated addition -- that
package's own D-001 anchor (``plans/plan-2026-09-12T173329-e20362c4/
decisions.md``) forbids importing ``qwen3_next.py`` from it. See
``plans/plan-2026-09-13T073704-245ab5d5/decisions.md`` D-003.
"""
