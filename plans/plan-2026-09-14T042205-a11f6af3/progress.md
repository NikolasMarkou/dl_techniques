# Progress

## Completed
- [x] EXPLORE: v1 has a genuine chunking lever (~4.5 GiB potential); v2 has none (honest negative); clean baseline reconfirmed.
- [x] PLAN: plan.md written and approved — chunk v1's precompute, gated by an early falsification diagnostic, v2 out of scope.
- [x] Step 1 (commit `bc0326ac0`): diagnostic gate PASSED (auto/off ratio 0.59-0.86, no collapse). New finding: saving shrinks with num_layers (24.1%→11.2% at n=4/8/16, both OOM at n=24 with only ~11% gap) — design doc's "constant per-layer saving" prediction is wrong. Step 4 is now load-bearing, not a formality.

- [x] Step 2 (commit `d9014c3e0`): implemented chunked scan. Bit-exact equivalence verified (max_abs_diff=0.0) against a real MambaLayer's actual sublayer outputs. Scoped suite: 56 passed.
- [x] Step 3 (commit `d83d4acdb`): permanent forward-numerics (bit-exact) + gradient-correctness (reassociation_atol-derived tolerance) tests added. 59/59 tests pass.

- [x] Step 4 (commit `e172f7cbe`, plan-dir only, no production code touched): full-scale measurement at variant="130m". New ceiling batch=5 (was batch=4); batch=8 target NOT reached. Batch=4 peak 9.497->7.7695 GiB (18.19% saved); batch=8 attempted-alloc 14.8->9.8715 GiB (33.30% smaller, still OOMs). Pre-Mortem STOP-IF #2 FIRES (measured saving ~3x Step 1's slope-sweep prediction) — in the positive direction, not "no improvement". Escalated to REFLECT/orchestrator per plan's own instruction not to chase a bigger number with further code changes in this step.

- [x] Step 5 (plan-dir only, no production/test code changed): full `tests/test_models/test_mamba/`
  suite run (all 10 files confirmed via `ls` first) — 194 passed, 0 failed. Re-confirmed via grep
  that zamba2/hnet reference only `Mamba2Layer`/`Mamba2ResidualBlock` (v2), zero `MambaLayer` (v1)
  references — matches plan.md's assertion, no scope-drift found.

## In Progress
- [ ] Step 6: document final outcome, name v2's d_state as deferred follow-up

## Blocked
*Nothing currently.*
