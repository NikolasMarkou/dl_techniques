"""Pattern-3 (subword CLM) trainer for the Mamba-2 model package.

See ``train_mamba.py`` for the entry point and ``common.py`` for the
config/pipeline it drives. Targets ``Mamba2`` (v2) only -- see
``plans/plan-2026-09-12T173329-e20362c4/decisions.md`` D-002 for why
``mamba_v1``'s ``Mamba`` has no trainer here.

ADDENDUM 2026-09-13, plan-2026-09-13T073704-245ab5d5/D-002: SUPERSEDED. This
package now also ships a Mamba-1 trainer, colocated here rather than in a new
top-level package: see ``train_mamba_v1.py`` (entry point) and
``common_v1.py`` (config/pipeline). D-002 above correctly reflected the state
of the world when written; it did not rule that a v1 trainer could never be
added, only that it did not exist yet. See this plan's ``decisions.md`` D-002.
"""
