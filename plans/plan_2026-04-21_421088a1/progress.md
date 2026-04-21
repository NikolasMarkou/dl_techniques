# Progress

## Completed (iter-1 — retained, not reverted)
- EXPLORE: 3 findings indexed (clifford primitives, LeWM reusable assets, positional + infrastructure).
- PLAN v1: 7 design decisions locked (D-001..D-007), full file list, hardest-first verification order.
- EXECUTE iter-1 steps 1–9: model package, dataset, training script, 29-test suite, 2-epoch smoke run with monotone-decreasing loss (3.7679 → 3.3212) on GPU 1 (RTX 4070).
- REFLECT iter-1: 7/7 success criteria PASS, 4/4 STOP-IF triggers quiet, 8/8 assumptions held. `final_model.keras` round-trips.
- PIVOT (2026-04-21): planned-extension pivot to add V-JEPA tube masking (D-003 follow-up). All iter-1 code retained. Decision logged in decisions.md (not a failure pivot).

## In Progress (iter-2)
- REFLECT — verification.md written. Awaiting user decision on C14 partial.

## Completed (iter-2)
- cp-001-iter2.md created (restore point: `1c596fd`).
- Step 1 — config extended; 4 new TestConfig tests.
- Step 2 — `TubeMaskGenerator` + 7 hardest-first tests.
- Step 3 — `VideoJEPA` extended (mask token + dual-loss `call`).
- Step 4 — 4 new `TestVideoJEPAIter2` tests.
- Step 5 — per-loss metric trackers + 4 new CLI flags on
  `train_video_jepa.py`.
- Step 6 — 2-epoch smoke on GPU 1 (RTX 4070), final_model.keras saves
  and round-trips; mask_loss 0.5018→0.4906, total 2.267→2.229,
  next_frame_loss 7e-4→1.6e-3 (partial C14 — see verification.md).

## Remaining (iter-2)
- Step 4 — extend test suite: mask-loss finiteness, serialization-with-new-fields, iter-1 regression guard (causality, AdaLN-id, SIGReg, forward, save/load, streaming still pass)
- Step 5 — extend `train_video_jepa.py` to log both loss components
- Step 6 — smoke training run on GPU 1, verify both losses finite + decreasing

## Blocked
*Nothing.* Plan v2 must be approved before EXECUTE begins.
