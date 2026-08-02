# DINO Self-Supervised Pretraining: The Measurement Record

*Date: 2026-08-02. Status: measurement record, git-tracked on purpose.*

---

## 0. What this file is

This is the git-tracked home for the DINO self-supervised-pretraining measurements produced by two
planning sessions in this repository: `plan-2026-08-01T105809-dc0c402e` (which built
`src/dl_techniques/models/dino/`, `src/dl_techniques/losses/dino_loss.py` and
`src/train/dino/`) and `plan-2026-08-01T195746-12a1f2db` (which produced the measurements below).

Both of those plans' working directories live under `plans/`, and every training run they produced
lives under `results/`. **Both directories are gitignored.** A fresh checkout of this repository
therefore contains the code but none of the evidence for how the code's defaults were chosen. This
file is that evidence. If a number in `src/dl_techniques/models/dino/README.md`, in
`src/train/dino/train_dino.py`'s docstring, or in `src/train/dino/knn_eval.py`'s docstring needs a
provenance, it is here.

Nothing in this file is reproducible from a checkout alone. Each figure below was measured once.
Every training-run figure was measured at ONE scale (smoke) on ONE GPU (GPU 1, RTX 4070); the two
probe-derived sections (the sha1 determinism table and the crop-geometry table in section 4) were
measured on CPU with no model built. Each section states its own scope.

**Reading rule used throughout.** Every k-NN number is quoted beside its own control: a
zero-optimizer-step, random-initialization k-NN evaluation written by
`KNNEvalCallback.on_train_begin` into that run's own `random_init_control.json`, before `fit()`
takes its first gradient step. There are no bare deltas anywhere in this file. A "delta" always
means "this run's endpoint minus THIS run's own random-init control".

---

## Table of contents

1. [The headline: improved config vs shipped default](#1-the-headline-improved-config-vs-shipped-default)
2. [The exact configuration of each arm](#2-the-exact-configuration-of-each-arm)
3. [P1: diagnosing the early k-NN dip](#3-p1-diagnosing-the-early-k-nn-dip)
4. [P2: reproducibility and crop geometry](#4-p2-reproducibility-and-crop-geometry)
5. [All nine confounds](#5-all-nine-confounds)
6. [Defects found in the measuring instruments themselves](#6-defects-found-in-the-measuring-instruments-themselves)
7. [The cost model](#7-the-cost-model)
8. [Open questions](#8-open-questions)
9. [Provenance](#9-provenance)

---

## 1. The headline: improved config vs shipped default

**Scope of this whole section**: 2 arms x 2 seeds x 60 epochs, smoke scale (`tiny` backbone,
96 px, `dino_out_dim=4096`, 4 local crops, batch 32), imagenette, fully controlled
(bit-identical) training stream, GPU 1 (RTX 4070, 12 GB), runs strictly serial. Nothing in this
section is a claim about any other scale, dataset, backbone or batch size.

### The pre-committed decision rule, verbatim

> Primary endpoint `dino_knn_top1_k20`;
> `delta = mean(last 3 evaluated epochs) - mean(that run's own zero-step control)`;
> **IMPROVED** if `delta >= 0.04` for BOTH seeds, **NOT IMPROVED** if `<= 0.02` for both,
> **INCONCLUSIVE** otherwise, including seed disagreement.

The rule was fixed before the runs were launched. The baseline arm is reported under the identical
rule, as the comparison the rule alone cannot make.

### The result

| arm | seed | own control | endpoint (mean last 3 evaluated epochs) | delta | verdict |
|---|---|---|---|---|---|
| **improved** | 42 | 0.2900 | 0.4326 | **+0.1426** | **IMPROVED** |
| **improved** | 1337 | 0.2969 | 0.4661 | **+0.1693** | (both seeds >3x the 0.04 threshold) |
| **baseline** (the shipped default at the time of measurement) | 42 | 0.2900 | 0.3363 | **+0.0462** | **INCONCLUSIVE** |
| **baseline** | 1337 | 0.2969 | 0.3285 | **+0.0316** | (one seed clears 0.04, one does not) |

The secondary endpoint `dino_knn_top1_k10` agrees, at the same 2 arms x 2 seeds x 60 epochs:
improved **+0.1364** (s42) and **+0.1517** (s1337); baseline **+0.0488** (s42) and **+0.0296**
(s1337). The k10 verdicts fall the same way as k20 under the same rule.

**The baseline arm's INCONCLUSIVE is a real result and is not softened here.** After 60 epochs at
smoke scale, on a controlled stream, the configuration this repository shipped could not be shown
to beat its own random-init control by the pre-committed rule at both seeds. That is what gives the
improved arm's verdict its meaning.

### The arm-to-arm difference is NOT pre-registered

`improved - baseline` at matched seeds is **+0.0964** (s42) and **+0.1377** (s1337). This
comparison was not part of the pre-committed rule. It is descriptive only, and no verdict is
attached to it.

### Health diagnostics on the four runs

No STOP trigger fired on any of the four 60-epoch runs: 60/60 epochs completed, `halted_early=false`
on all four, the collapse flag never fired. Maximum feature mean-cosine was 0.52 and 0.48 for the
two baseline runs and 0.32 and 0.28 for the two improved runs, against a 0.95 trigger. Minimum
teacher `entropy_norm` across the four runs was 0.974 / 0.980 and 0.756 / 0.747, against a 0.10
trigger. (The source report lists the entropy pairs without arm labels; by the ordering it uses for
the cosine figures in the same sentence, the first pair is the baseline arm. That labelling is an
inference, not something the artifact states.)

---

## 2. The exact configuration of each arm

The improved arm differs from the baseline arm in **exactly two config keys**:

| key | baseline (shipped default at measurement time) | improved |
|---|---|---|
| `ema_warmup_steps` | `0` | `295` |
| `teacher_temp_final` | `0.07` | `0.04` |

`teacher_temp` was `0.04` in BOTH arms — that is already the shipped dataclass default, so the
improved arm's invocation string `--teacher-temp 0.04 --teacher-temp-final 0.04
--ema-warmup-steps 295` names three flags but changes only two values.

Both arms carried, identically:

```
--seed-training-stream --stateless-augmentation
--knn-bank-batches 64 --knn-query-batches 32
--max-steps 100000
--knn-eval-every 4
--random-init-repeats 2
--early-stopping-patience 70
```

at smoke scale (`tiny` / 96 px / `dino_out_dim=4096` / 4 local crops / batch 32), on GPU 1.
`--max-steps 100000` and the explicit `64`/`32` k-NN bank sizes are not decoration: see
[section 6](#6-defects-found-in-the-measuring-instruments-themselves) for the two `--smoke`
presets they exist to defeat, and `--early-stopping-patience 70` exists so a 60-epoch horizon is
actually run to its end rather than stopped early.

### Why `295` is not a portable constant

`295` is `num_train // batch_size` for the imagenette training split at `batch_size=32`. It is
**one epoch at that dataset and that batch size only**. At `batch_size=64` the same literal would
mean roughly two epochs; on a different dataset it means whatever that dataset's cardinality makes
it.

For this reason the repository does **not** ship the literal `295`. It ships the same behaviour
epoch-denominated, as `TrainingConfig.ema_warmup_epochs = 1.0`, resolved against the run's real
`steps_per_epoch` at callback-construction time, with `ema_warmup_steps` retained as an
absolute-step override that wins when it is non-zero. At the measured dataset and batch size the
two express the identical number of steps; on any other dataset or batch size the epoch-denominated
default is the one that transfers and the literal is the one that silently does not.

---

## 3. P1: diagnosing the early k-NN dip

**The phenomenon.** At smoke scale, a DINO run's first evaluated k-NN reading sits *below* its own
random-init control: the model gets measurably worse before it gets better. A prior brief named two
suspects. Neither is the mechanism.

**Scope of this whole section**: 8 epochs per run, smoke scale, imagenette, GPU 1, 2 seeds
(42, 1337). Everything here is measured at the *early* part of training only, and none of it is a
claim about the 60-epoch endpoint in section 1.

### The three-candidate verdict

| candidate | flag tested | verdict |
|---|---|---|
| centering-EMA lag | `--center-momentum 0.0` | **CLEARED** — effect **-0.0063** in the dip matrix |
| teacher-TEMPERATURE warmup | `--teacher-temp 0.04 --teacher-temp-final 0.04` | **structurally cannot cause it** — inert at the epoch-0 endpoint by construction; returned an **exactly-zero** null at both seeds on the controlled stream |
| teacher-WEIGHT EMA warmup (absent from the brief) | `--ema-warmup-steps 295` | **IMPLICATED** in both matrices: dip matrix **+0.0445**, confirm matrix **+0.0498** |

### Dip matrix: 4 arms x 2 seeds, 8 epochs, UNSEEDED stream

Endpoint = `dino_knn_top1_k20` at the first evaluated epoch minus that run's own control; effect =
`mean_seeds(arm) - mean_seeds(A)`.

| arm | mean dip | effect vs A | verdict |
|---|---|---|---|
| A baseline | -0.0601 | — | REFERENCE |
| B no temp warmup | -0.0371 | +0.0229 | AMBIGUOUS |
| C no center lag | -0.0664 | **-0.0063** | **CLEARED** |
| D EMA warmup | -0.0156 | **+0.0445** | **IMPLICATED** |

The dip reproduces: on this matrix the baseline arm sits about 0.060 below its own random-init
control at both seeds.

**But this matrix measured its own null, and the null is 0.0400 wide against a 0.04 threshold.**
Arm B is provably inert at the epoch-0 endpoint: `create_teacher_temp_callback` assigns via
`on_epoch_begin`, and `linear_ema_schedule` returns `teacher_temp` itself at epoch 0 for both the
`0.04 -> 0.07` and the `0.04 -> 0.04` schedule, first differing at `on_epoch_begin(1)`. So arm B's
epoch-0 reading is a NULL sample by construction, and the individual fixed-seed pair
`B/s42 - A/s42 = +0.0400` is a null sample that reaches the IMPLICATED threshold. Arm D cleared
that threshold by 0.0045. **The dip matrix alone does not resolve this**, which is why the confirm
matrix below exists.

Also recorded from the dip matrix, on an **exploratory** (not pre-registered) endpoint — mean of
the last 3 evaluated epochs minus own control: A `-0.0379`, B `+0.0138`, C `-0.0272`, D `+0.0173`;
i.e. effects vs A of B **+0.0518**, D **+0.0552**, C `+0.0107`. On that endpoint B and D are the
only arms rising above their own random-init control, at both seeds. This is exploratory: it was
not pre-registered and it is read at an endpoint where arm B is live rather than inert.

### Confirm matrix: 4 arms x 2 seeds, 8 epochs, FULLY CONTROLLED stream

The instrument check first. Arm B' is inert at epoch 0, so `A' - B'` at a fixed seed must be
exactly zero if the stream is genuinely controlled:

| seed | A' epoch-0 k20 | B' epoch-0 k20 | delta |
|---|---|---|---|
| 42 | 0.2294921875 | 0.2294921875 | **0.0** |
| 1337 | 0.234375 | 0.234375 | **0.0** |

The null went to exactly zero at both seeds. That is what makes the effects below readable at all.

| arm | effect vs A' (pre-registered endpoint) | verdict | exploratory (last-3) |
|---|---|---|---|
| B' no temp warmup | **0.0000** at both seeds | **CLEARED** | +0.0387 |
| D' EMA warmup | **+0.0498** (per-seed: +0.0625 at s42, +0.0371 at s1337) | **IMPLICATED** | **+0.0550** |
| E' `--source-image-size 224` | +0.0024 | NO-DIFFERENCE | +0.0028 |

Note the teacher-temperature arm's exploratory effect **shrank from +0.0518 (unseeded dip matrix)
to +0.0387 (controlled confirm matrix)** once the training stream was controlled. Its
pre-registered verdict is CLEARED and its exploratory verdict is ambiguous. That asymmetry matters
in [section 8](#8-open-questions).

### `--ema-warmup-steps N` is a SUPERSET of the mechanism it is named for

`--ema-warmup-steps 295` does two things at once:

1. it freezes the teacher-weight EMA for the first 295 optimizer steps
   (`TeacherEMACallback.on_train_batch_end` increments a counter and returns early, so
   `update_teacher_ema` is genuinely never called during the warmup); **and**
2. it re-bases the cosine EMA ramp 295 steps later, because the post-warmup step index restarts at
   0 while `total_steps` is unchanged.

Every claim about arm D or D' is a claim about **that pair**, not about freezing alone. Freeze and
ramp-re-basing have never been separated by any measurement in this repository.

### The dip is reduced, not abolished, and the reduction is SEED-DEPENDENT

At the 60-epoch improved arm (section 1's config), the epoch-0 reading relative to that run's own
control is:

| seed | improved arm epoch-0 vs own control | baseline arm epoch-0 vs own control |
|---|---|---|
| 42 | **+0.0020** (above control: no dip at this seed) | -0.0605 |
| 1337 | **-0.0273** (below control: a real dip remains at this seed) | -0.0625 |

The correct statement is: **the improved configuration removes the dip at seed 42 and roughly
halves it at seed 1337**, and improves the 60-epoch endpoint decisively at both. An earlier draft
of the source report wrote "with no dip at all" and quantified that over both seeds; it was false
at seed 1337 and was caught by adversarial review. Do not restate it in the quantified-over-both
form.

---

## 4. P2: reproducibility and crop geometry

### Both reproducibility flags are required together

**Scope**: a 2-process, **CPU-only** probe. Each process built the real
`train_dino.build_dataset` output and took the sha1 of the first 3 batches. No model was
constructed; no GPU was involved. Total cost about 6 CPU-minutes.

| flags | run 1 sha1 | run 2 sha1 | identical? |
|---|---|---|---|
| `--seed-training-stream` alone | `49bf308a` | `4fbd3a6e` | **no** (batch-0 mean 0.0999388 vs 0.0002119) |
| `--stateless-augmentation` alone | `e70e2ad2` | `5f968198` | **no** |
| **both** | `2ce84c18` | `2ce84c18` | **yes**, on all 3 batches, to every printed digit of mean and std |

`--seed` alone does not make two DINO runs comparable, and neither flag is redundant.
`--seed-training-stream` reaches the TFDS training-file interleave;
`--stateless-augmentation` reaches the multi-crop augmentation RNG. The residual unseeded source
that `--seed-training-stream` alone leaves behind is the augmentation RNG:
`make_multi_crop_map_fn` seeded ONE shared `tf.random.Generator` **stream**, and
`build_raw_image_dataset` maps it under `num_parallel_calls=AUTOTUNE`, so which element receives
which slice of that stream varies from process to process.

**What this probe rules out, precisely.** For the pipeline it measured, the bit-identity of the
both-flags cell rules out the `tf.data` element `.shuffle()` and `options.deterministic` as
residual sources of nondeterminism. **It says nothing about cuDNN kernel nondeterminism**, because
the probe never ran on a GPU and never executed a kernel. A stronger claim was written down at one
point and is wrong; do not reintroduce it.

### Crop geometry: what `--source-image-size` actually changes

**Scope**: measured through the SHIPPED `_random_resized_crop` (the achieved crop box was captured
by wrapping `tf.image.crop_to_bounding_box` in the module namespace, rather than re-deriving the
geometry), N=2000 draws per cell, `out_size=96`, `local_scale=(0.05, 0.4)`, `ratio=(3/4, 4/3)`.
Upsample factor = `out_size / sqrt(crop_w * crop_h)`.

| source | local crop side (px) | mean upsample | worst | % of local views upsampled |
|---|---|---|---|---|
| 96 px | 21.4-61.0 (mean 44.1) | **2.35x** | 4.50x | **100%** |
| 224 px | 50.1-141.9 (mean 102.9) | **1.006** | 1.92x | **39%** |

Global views at a 224 px source: 0.53x mean, never above 1.0.

A prediction that 224 px would drive the mean local-crop upsample **below** 1.0 was **falsified as
stated**: the measured mean is **1.006**, marginally *above* 1.0. The direction and magnitude of
the effect were right (a 2.3x reduction in mean interpolation, worst case 4.50x -> 1.92x); the
specific quantified claim was not. The 96 px figures reproduce the inherited 2.33x / 4.50x
docstring numbers to within rounding, so those inherited numbers are sound.

### The k-NN effect of `--source-image-size 224`: none measurable

Arm E' isolated exactly one differing config key (`source_image_size: None -> 224`) at 8 epochs,
smoke scale, controlled stream, 2 seeds. Effect **+0.0024** on the pre-registered endpoint,
**+0.0028** on the exploratory last-3 endpoint, and **+0.0044** at a separate final-epoch endpoint
where the signs disagree. Verdict: **NO DIFFERENCE**.

Read that correctly. It means: *a 2.3x reduction in mean local-crop interpolation buys no
measurable k-NN delta at this scale, at 8 epochs, at 2 seeds*. It does **not** mean crop resolution
is irrelevant. 224 px mitigates the thumbnail-crop defect; it does not eliminate it, since 39% of
local views are still upsamples at that source size.

---

## 5. All nine confounds

Reproduced from the source report, in full, because a confound that lives only in a gitignored
directory does not ship.

1. **The dip matrix ran on an UNSEEDED stream.** Its own null, measured for free from a
   structurally inert arm, reached **0.0400** — the decision rule's own threshold. The dip matrix
   is strictly-less-confounded than what preceded it, not unconfounded.
2. **The improved arm changes TWO things at once** (`ema_warmup_steps` and `teacher_temp_final`).
   A positive 60-epoch result does not apportion credit between them. Separation exists only at 8
   epochs.
3. **`improved - baseline` is NOT a pre-registered endpoint.** It is descriptive only.
4. **n = 2 seeds.** No confidence interval is claimed anywhere in this file or in its sources.
5. **One scale only** — smoke: `tiny` / 96 px / `dino_out_dim=4096` / 4 local crops / batch 32.
   **Paper scale was never run, because it OOMs**: `small` / 224 px / `out_dim=65536` at
   `batch_size=32` requested a single 10.13 GB allocation against 10001 MiB usable and the process
   was aborted by the BFC allocator. An inherited "15-25x" cost extrapolation to paper scale is
   RETIRED, not refined — there is no comparable step time to divide.
6. **The confirm and long matrices are not batch-comparable to the dip matrix or to the earlier
   `results/dino_smoke_step12/` artifact**, because `--stateless-augmentation` changes what each
   batch *contains*, not merely its order. Mitigated by the per-run random-init controls: every arm
   is read against its own control, so each matrix is internally comparable across its own arms,
   which is the only comparison its verdicts make.
7. **The EMA-warmup flag is a superset** (freeze + cosine-ramp re-basing), never separated.
8. **The probe is a k-NN classifier over a 2048-image memory bank and a 1024-image query set**, not
   linear-probe accuracy. It is a proxy for representation quality, not the standard benchmark.
9. **The control's within-run range is exactly 0.0 and is NOT a noise estimate.** The k-NN bank is
   `.cache()`d, so the `--random-init-repeats` repeats replay the identical bank. This is recorded
   in each run's `random_init_control.json` as `repeats_are_independent: false`. The genuine noise
   band lives across seeds and across processes only.

One free cross-process noise datum, for calibration: two identical-config, identical-`--seed 42`
calibration runs in two separate processes differed by **0.0020** in their epoch-0
`knn_top1_k20` — an order of magnitude under the rule's 0.02 CLEARED threshold.

---

## 6. Defects found in the measuring instruments themselves

These are recorded because each one silently produced, or nearly produced, a wrong number.

**(i) The verdict reader reported the WRONG verdict.** The mechanical reader keyed its results dict
on `seed` alone. That was correct while the measurement phase was 1 arm x 2 seeds. The phase was
later changed to 2 arms x 2 seeds and the reader was not re-verified: baseline rows were silently
overwritten by improved rows (identical seed keys) and all four deltas were pooled into one list as
though they were four seeds of one arm. Because `baseline/s1337 = +0.0316` fails the `>= 0.04`
test, the pooled judgement returned **INCONCLUSIVE for a phase in which both arms are internally
consistent**. Fixed by keying on `(arm, seed)`; no run was re-executed and no threshold was moved.
**It was caught only because a hand-check contradicted the machine.** The session that produced it
spent nine steps insisting that verdicts be computed mechanically rather than by eye, and the
mechanical computation was the thing that was wrong. The argument that survives is for keeping
both, not for trusting either alone.

**(ii) The k-NN bank was a DIFFERENT ESTIMATOR under `--smoke`.** `--smoke` leaves the k-NN bank
and query batch counts at 16 / 8 (512 images), while every k-NN figure in this file — including
the 0.2754-0.2949 control band and both control values in section 1 — was measured at
64 / 32 (2048 images). A 4x smaller bank is a *different estimator*, not merely a noisier one. Any
run intended to be comparable to the numbers here must pass
`--knn-bank-batches 64 --knn-query-batches 32` explicitly.

**(iii) `--smoke` silently sets `max_steps=5`.** A `--smoke` "epoch" is 5 of its 295 real training
steps. Any run intended to be comparable must pass `--max-steps 100000` explicitly. Tellingly, the
pre-existing `results/dino_smoke_step12/config.json` already recorded `max_steps=100000` — an
earlier run needed the same workaround and nobody wrote it down.

**Honourable mention 1 — the control's repeats are not independent.** The k-NN memory bank is
`.cache()`d, so `--random-init-repeats N` replays the same bank N times and the within-run repeat
range is exactly 0.0 **by construction**. This was RED-proven by hardcoding
`independent = True` and watching the guard fire. It is recorded per run as
`repeats_are_independent: false`. It must never be reported as a noise estimate.

**Honourable mention 2 — a vacuity injection went RED for the wrong reason.** A deliberate +0.5
perturbation, injected to prove a new set of assertions could actually fail, was caught by
`sync_teacher_to_student`'s own pre-existing trailing sweep rather than by the new assertions. The
test went red, and the thing under test was still unproven.

---

## 7. The cost model

**Scope**: smoke scale, imagenette, GPU 1 (RTX 4070, 12 GB), one process at a time.

| quantity | measured |
|---|---|
| end-to-end step time, real trainer | **0.200 s/step** |
| compute-only step time, synthetic batch, 20 `train_on_batch` after warmup | **0.0526 s/step** |
| peak GPU memory, smoke scale | **1534 MiB** |
| pure training epoch | 59.0 s |
| one k-NN eval at bank 64 / query 32 | 35.7 s |
| fixed per-run overhead | 155.5 s |

**The model is INPUT-BOUND at smoke scale.** End-to-end costs 3.8x the pure GPU compute: JPEG
decode plus multi-crop augmentation on the CPU dominates. A budget taken from a compute-only probe
alone would have been about 4x optimistic. The two real-trainer measurements were a 1-epoch and a
2-epoch run at identical config, differenced to isolate the marginal epoch with no assumption about
overhead — an mtime delta cannot separate per-run overhead from per-epoch cost, and an earlier
inherited estimate that did exactly that was both ~7-10% high and wrongly decomposed.

The 1534 MiB peak corroborates an independently-obtained 1518.6 MiB figure in the trainer docstring
to within 1%.

**Paper scale does not fit.** `small` / 224 px / `out_dim=65536` at `batch_size=32` requested a
single 10.13 GB allocation against 10001 MiB usable and was aborted by the allocator. "Does not
fit" is a stronger and cheaper result than a step-time ratio would have been, and it is why the
inherited "15-25x" extrapolation is retired rather than refined: there is no paper-scale step time
to divide by.

---

## 8. Open questions

Each of these is a question this repository has NOT answered. They are written here because
`plans/` is gitignored and a question that lives only there disappears with the plan directory.

### 8.1 Attribution: the improved arm changes two things

The 60-epoch IMPROVED verdict in section 1 belongs to the PAIR
(`ema_warmup_steps 0 -> 295`, `teacher_temp_final 0.07 -> 0.04`). The two mechanisms were
separated at 8 epochs and never at 60. A positive joint result does not apportion credit.

**Concretely what would settle it**: 3 arms x 2 seeds x 60 epochs — EMA-only, temp-only, both —
read against the already-existing 60-epoch baseline runs. At the measured 0.200 s/step end-to-end
that is roughly 7 hours of serial GPU time.

### 8.2 The `--ema-warmup-steps` superset was never separated

`--ema-warmup-steps N` freezes the teacher EMA for N steps **and** re-bases the cosine EMA ramp N
steps later (the post-warmup step index restarts at 0 while `total_steps` is unchanged). No
measurement has separated the two. A freeze-without-re-basing option on `TeacherEMACallback` would
separate them.

That option is **deliberately not shipped**. `TeacherEMACallback`
(`src/dl_techniques/models/depth_anything/teacher_ema.py`) is shared with
`src/train/depth_anything/`, and adding an unmeasured knob to shared code for an experiment nobody
has scheduled is speculative surface. The question is recorded here so the option survives without
the code existing. Before any claim of the form "freezing the teacher early helps" is made, this
must be separated.

### 8.3 The dip endpoint is structurally biased toward the EMA arm

The epoch-0 dip endpoint asks "how far below its own random-init control does the model fall after
one epoch". Freezing the teacher for exactly epoch 0 means that **any** intervention which causes
the model to learn less during epoch 0 scores as "less dip". The endpoint cannot distinguish "fixed
the dip" from "did less". This is a property of the endpoint, not of the flag, and it applies to
the `+0.0445` / `+0.0498` figures in section 3. The 60-epoch endpoint in section 1 is not subject
to it.

### 8.4 `teacher_temp_final = 0.04` has weaker evidence than the pair it shipped in

Stated plainly, because the repository now ships this value as a default:

- It was **CLEARED** at the pre-registered epoch-0 endpoint, where it is structurally inert.
- It was **ambiguous** on the exploratory last-3 endpoint.
- Its exploratory effect **shrank from +0.0518 to +0.0387** once the training stream was
  controlled.
- It was shipped into the improved arm despite the confirm matrix's own verdict label saying it
  "must be confirmed, not shipped".
- No temp-only arm has ever been run at 60 epochs.
- Setting `teacher_temp_final` equal to `teacher_temp` makes `teacher_temp_warmup_epochs` an INERT
  knob at defaults, and departs from the DINO paper's `0.04 -> 0.07` recipe.

The repository ships it anyway, by explicit user ruling. The reason is not that 0.04 is better —
nothing measured says that. The reason is that shipping only the EMA half would produce a default
configuration **that nobody has ever run at any depth**, which is a worse epistemic position than
either measured arm.

### 8.5 Paper scale is entirely unmeasured

Nothing in this file extrapolates to paper scale. Paper scale OOMs at `batch_size=32` on 12 GB and
needs either a smaller batch or gradient accumulation before any measurement can be taken there.

### 8.6 n = 2 seeds

Every verdict in this file rests on two seeds. No confidence interval is claimed anywhere. The
within-run control repeats are not an independent noise estimate (confound 9); the only genuine
noise data are the two-seed spread and the single 0.0020 cross-process datum in section 5.

---

## 9. Provenance

| item | value |
|---|---|
| plan that built `models/dino/`, `losses/dino_loss.py`, `src/train/dino/` | `plan-2026-08-01T105809-dc0c402e` |
| plan that produced every measurement in this file | `plan-2026-08-01T195746-12a1f2db` |
| commit range of the measurement plan | `355c70d5..3bca113d` (10 commits) |
| plan that promoted these numbers into this git-tracked file | `plan-2026-08-02T132301-93deeae2` |
| hardware | GPU 1, RTX 4070, 12 GB; all training runs strictly serial |
| dataset | imagenette (TFDS), training split |
| raw artifacts | `results/*/` and `plans/*/` — **both gitignored**, not recoverable from a checkout |

The training-run directories, the per-run `random_init_control.json` files, the verdict JSONs, the
sha1 determinism probe output and the crop-geometry probe output all live under those two
gitignored trees. If they have been cleaned up, this file is the record.
