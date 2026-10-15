# Matching layers review — 2026-10-15

Scope: `src/dl_techniques/layers/matching/` CPU-only static review + verification.
Files: `learned_fourier_rotary.py`, `lightglue_blocks.py`, `match_assignment.py`
(+`filter_matches`), `token_confidence.py`, `__init__.py`.
Consumers checked: `models/vision/keypoints/lightglue/model.py`,
`losses/lightglue_loss.py`, `metrics/keypoint_matching.py`,
`train/lightglue/pipeline.py`, oracle `tests/lightglue_reference_numpy.py`.
No code changed during review. Verification runs used
`CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python` (CPU).

## Summary

Faithful LightGlue transcription. Interleaved rotary, `(H, Dh, 3)` qkv layout,
`Dh**-0.25` cross scale, erf-GELU, `1e-5` LayerNorm, double log-softmax +
dustbin, mutual `filter_matches` all match the oracle. Masking/dtype extensions
are deliberate and documented. No weight-mapping bug found.

Double-check overturned the two P0 portability claims and narrowed the rest
(see Verdicts). Remaining items are P1/P2.

## Verdicts (double-check)

| # | Claim | Verdict |
|---|-------|---------|
| 1 | `match_assignment.py:267-269` dynamic slice breaks JAX/XLA | Refuted as stated; narrower bug confirmed (below) |
| 2 | `learned_fourier_rotary.py:76-77` tensor unpack breaks | Refuted |
| 3 | Non-canonical `build`/`compute_output_shape` | Confirmed, P2 |
| 4 | `build` rejects `None` last-dim in 4 files | Narrowed to 1 file |
| 5 | `return_logits` shape mismatch | Confirmed with nuance |
| 6 | `sim` unmasked | Confirmed, documented |
| 7 | Rescue-vs-zero deviation | Confirmed, documented |

### 1. `filter_matches` symbolic-`None` (narrowed)

Eager CPU: `keras.ops.shape(tensor)` returns Python `tuple[int]`, so slicing
with `shape[1]-1` and `arange(m_pts)` works. No backend failure evidenced.

Real bug: on a symbolic `KerasTensor` it returns `(None, None, None)`, so
`match_assignment.py:268` raises `TypeError: unsupported operand ... 'NoneType'`.
Verified: `filter_matches(Input((None,None)))` fails; `filter_matches(MA-output)`
with variable points fails. `LightGlue` subclassed path is unaffected: a spy on
`filter_matches` never fires during Functional wrapping (shape via
`compute_output_shape`), and variable-point `predict` with `(5,7)` succeeds.
Impact: breaks direct Functional composition, not the model.

### 2. `rotate_half` unpack (withdrawn)

`shape` is a static tuple; `(*shape[:-1], ...)` works. Variable-point
Functional build + `predict` of `LightGlueSelfBlock` succeed.

### 3. Non-canonical signatures (P2)

`lightglue_blocks.py:262,405,309,477`, `match_assignment.py:162,223`,
`token_confidence.py:77,113` use multi-arg
`build(x0_shape, x1_shape, ...)` / `compute_output_shape(...)` instead of a
single `input_shape`. Verified: `CrossBlock.compute_output_shape((None,5,32),
None)` raises `TypeError`; `MatchAssignment` variable build yields
`(None,None,None)`, losing the `+1`. `LightGlue` avoids it via explicit
`build()` (`model.py:437-441`) plus subclassed execution. Breaks generic
tooling only.

### 4. `None` last-dim guard (1 file)

`MatchAssignment`, `MatchTokenConfidence`, blocks all have
`shape[-1] is not None` guards; `MA.build((None,5,None))` succeeds. Only
`LearnedFourierRotaryEncoding.build` (`learned_fourier_rotary.py:204`) lacks it
and raises on `(None,None,None)`. Low severity (width always 2/4).

### 5. `token_confidence` `return_logits` (P2)

Subclassed runtime returns the 4-tuple correctly. Standalone Functional
`t(a,b,return_logits=True)` silently yields 2 outputs (verified) because
`compute_output_shape` hardcodes the 2-tuple. `_confidence`
(`token_confidence.py:98-99`) is dead code; `call` inlines `sigmoid`.

### 6. `sim` unmasked

Verified `sim` identical with/without masks; padded `scores` correctly `0`
with shape `(1,4,5)`. Documented (`match_assignment.py:136-139`). Misuse risk
only; model discards it (`model.py:500`).

### 7. Rescue vs reference zero

All-padded `SelfBlock` output verified finite but non-zero, matching the
documented deviation from the oracle's `nan_to_num`/skip
(`tests/lightglue_reference_numpy.py:199-200,241-251`). Isolated by masks;
consumers pass them (`model.py:499-500`, `pipeline.py:293-300`).

## Consumers — checked, correct

- `model.py:472-474` shares one `self_blocks[i]` across images (correct).
- Mask completion `model.py:487-490`, `_run_layer` sharing, pruning
  `take(freqs,keep,axis=3)` correct for `(2,B,1,N,Dh)`.
- `pipeline.py:293-295` forwards masks to `objective.compute` (D-012).
- `metrics/keypoint_matching.py:50-52` correctly warns `compile()` cannot
  supply masks.

## Opportunities

- `matching/__init__.py` is 0-byte; consider docstring-only statement of the
  deliberate exports-nothing shape (cf. `fusion`).
- Document fp16 `attn @ v` matmul (`lightglue_blocks.py:304,468-471`) vs the
  pinned float64 GELU constant.
- Missing tests: empty-image, one-sided mask, `return_logits` Functional shape,
  per-block standalone save/load, symbolic-`None` `filter_matches` guard.
