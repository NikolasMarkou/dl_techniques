# MambaLCT trainer

Pattern-4 clip trainer for `models/vision/mambalct/` (mirrors `train/siamfc/`).

## Data

`build_mambalct_clip_example` (`datasets/vision/tracking.py`) reframes a
still image + box as a clip: centered template + `clip_length` search
frames with center jitter (`max_shift_ratio`, train only) and photometric
jitter, boxes in normalized search-crop `(cx, cy, w, h)`, scores 1.0 when
the jittered center stays in-crop. COCO yields still-image pseudo-clips
(same frame repeated with jitter) — no video corpus ships with this repo,
so true motion must come from a video dataset added later; the clip builder
already takes any `(image, box)` source.

## Losses

`scores`: stock binary cross-entropy. `boxes`: `MambaLCTBoxLoss`
(`l1_weight=5`, `giou_weight=2`, paper Eq. 10). Single AdamW
(`learning_rate=2e-4`); the paper's 10x head LR is deferred.

## Context regime (read before drawing inference conclusions)

Training always zero-initializes the context (`updated_context` has no
auxiliary loss; the context module learns through the `enhanced` frame
path only) on short clips (`--clip-length`, default 2 — the paper's
training clip length, Li et al., 2024, §4.1). Carrying `updated_context`
across clips/frames is therefore inference-only and unevaluated: do not
credit validation numbers with long-horizon memory they never trained
under.

Memory scales with `batch_size × clip_length` (the encoder ingests
`(B*T, S, S, 3)` per step) with no accumulation knob; raising
`clip_length` toward video-realistic values multiplies encoder memory.

## Flags

Notable: `--clip-length 2` (paper train), `--max-shift-ratio 0.1`,
`--variant mambalct-256|mambalct-384` (must agree with
`--template-size/--search-size`), `--data-source synthetic|coco`
(COCO needs `--steps-per-epoch/--val-steps`).
