# train/nam — Neural Arithmetic Module

Trains models that evaluate arithmetic expression strings (`"(3 + 5) * 2"`) by
repeatedly reducing one sub-expression at a time. Model code lives in
`src/dl_techniques/models/neural_computer/nam/` (`NAM`, `NAMConfig`,
`ArithmeticTokenizer`); the DFSA variants are defined in this directory, in
`train_dfsa.py`.

## Read this before spending GPU hours

**The DFSA arithmetic path has zero learned parameters.** In `train_dfsa.py`,
operator identity (`token_id - 14`), PEMDAS + parenthesis precedence, adjacent
operand masking, digit-to-number assembly (`digit x 10^pos`), the four
arithmetic ops, and re-tokenization between steps are all hardcoded. A
randomly-initialised `DifferentiableFSA` already evaluates expressions exactly.
Training it changes the tree encoder, whose output the arithmetic path does not
consume. If you want a correct calculator, run `eval_dfsa.py` against an
untrained model and stop.

Training `train_dfsa.py` / `train_dfsa_ste.py` is only useful if you are working
on the *learned* pieces: the tree encoder, the STE gradient path, or the
learned residual scorer (`train_dfsa_ste.py`, gated by `--alpha-max`).

`train_nam.py` trains the older, fully-learned `NAM` model. It does learn, but
scalar number regression through a `Dense(D, 1)` head is the known bottleneck —
accuracy degrades sharply as operand digit count grows.

## Scripts

| Script | What it does |
|---|---|
| `train_nam.py` | Multi-task training of the learned `NAM` model (number, operator, reduction, halting, result heads). |
| `train_dfsa.py` | Trains `DifferentiableFSA` — deterministic arithmetic, learned tree encoder. |
| `train_dfsa_ste.py` | `DifferentiableFSA` with `use_ste=True` + learned residual scorer, ramped in after a phase-1 warmup. |
| `eval_dfsa.py` | Evaluation suite: single-op, 2-op and 3-op flat, parenthesised, PEMDAS-vs-paren, edge cases. Prints per-group pass rates. No training. |
| `test_extreme.py` | Stress cases: big operands, deep nesting. No training. |
| `create_dataset.py` | Writes expression dataset files to disk. Nothing else reads them — the trainers generate batches on the fly via `data_generator.py`. |
| `data_generator.py` | 13 curriculum difficulty levels (8 single-op, 3 multi-op, 2 parenthesised). Library module, not runnable. |

## Run

```bash
# Evaluate the deterministic DFSA (no checkpoint needed)
MPLBACKEND=Agg .venv/bin/python -m train.nam.eval_dfsa --gpu 0

# Train the DFSA tree encoder
MPLBACKEND=Agg .venv/bin/python -m train.nam.train_dfsa \
    --steps 20000 --batch-size 64 --act-steps 4 --gpu 0

# Train the STE variant
MPLBACKEND=Agg .venv/bin/python -m train.nam.train_dfsa_ste \
    --steps 20000 --phase1-steps 2000 --alpha-max 0.1 --gpu 0

# Train the learned NAM with the smooth curriculum
MPLBACKEND=Agg .venv/bin/python -m train.nam.train_nam \
    --variant small --curriculum --steps 100000 --gpu 0
```

## CLI — `train_nam.py`

| Flag | Default | Notes |
|---|---|---|
| `--variant` | `tiny` | `tiny`, `small`, `base`. |
| `--phase` | `phase_1` | `phase_1`..`phase_5`. Ignored when `--curriculum` is set. |
| `--curriculum` | off | Smooth curriculum; difficulty rises over training while easier levels stay mixed in. Overrides `--phase`, `--min-val`, `--max-val`. |
| `--curriculum-cap` | `0.8` | Caps curriculum progress. At 1.0 the sampler puts ~67% of mass on the three hardest levels and operator accuracy regresses late in training. |
| `--steps` | `10000` | |
| `--batch-size` | `64` | |
| `--min-val` / `--max-val` | phase config | Operand range override. |
| `--lr` | `1e-4` | |
| `--weight-decay` | `1e-5` | |
| `--clip-norm` | `10.0` | Global norm clip — this is why loss weights interact. |
| `--warmup-steps` | `1000` | |
| `--act-steps` | model config | ACT depth override. Use 2-4 for early phases. |
| `--ponder-cost` | `0.01` | |
| `--result-loss-weight` | `1.0` | |
| `--valid-loss-weight` | `0.5` | |
| `--w-number` | `0.5` | Number-extraction loss weight. Keep low; large values dominate the global clip and starve the operator and reduction heads. |
| `--number-loss-type` | `log_mse` | `log_mse`, `rel_err`, `combined`. `rel_err` alone has weak gradient at large magnitudes. |
| `--number-loss-delta` | `0.1` | Huber delta; used only for `rel_err` / `combined`. |
| `--w-operator` | `3.0` | Operator CE weight. |
| `--w-reduction` | `5.0` | Reduction-target CE weight. Reduction must converge first — everything downstream depends on it. |
| `--w-halt` | `0.5` | ACT halting BCE weight. `0.0` gives the head no gradient at all. |
| `--log-interval` | `100` | |
| `--save-interval` | `2000` | |
| `--eval-interval` | `1000` | Digit-accuracy matrix eval cadence. |
| `--log-grad-norms` | off | Per-sub-skill gradient norms. ~5x backward cost. Debug only. |
| `--checkpoint` | none | Resume path. Step and best-loss are restored from a JSON sidecar next to the weights. |
| `--save-dir` | `results` | Run root. |
| `--gpu` | none | GPU index. |

## CLI — `train_dfsa.py`

| Flag | Default | Notes |
|---|---|---|
| `--hidden-size` | `64` | |
| `--num-tree-layers` | `2` | Tree transformer blocks. |
| `--num-heads` | `4` | |
| `--max-len` | `64` | Token budget per expression. |
| `--act-steps` | `1` | Reduction steps. 1 = single-op only, 4 = multi-op. |
| `--steps` | `5000` | |
| `--batch-size` | `64` | |
| `--lr` | `1e-4` | |
| `--weight-decay` | `1e-5` | |
| `--clip-norm` | `10.0` | |
| `--warmup-steps` | `500` | |
| `--w-operator` | `3.0` | |
| `--w-reduction` | `20.0` | |
| `--result-loss-weight` | `1.0` | |
| `--curriculum-cap` | `0.8` | |
| `--multiop-start-step` | `0` | Staged training: only single-op levels before this step. |
| `--log-interval` | `100` | |
| `--eval-interval` | `1000` | |
| `--save-interval` | `2000` | |
| `--save-dir` | `results` | |
| `--gpu` | none | |

## CLI — `train_dfsa_ste.py`

Same as `train_dfsa.py` except: no `--w-reduction`, no `--multiop-start-step`,
defaults `--act-steps 4`, `--steps 20000`, `--log-interval 200`,
`--eval-interval 2000`, `--save-interval 5000`, plus:

| Flag | Default | Notes |
|---|---|---|
| `--phase1-steps` | `5000` | Steps training `op_classifier` only. 0 to skip. |
| `--alpha-max` | `0.1` | Hard cap on the residual-scorer mixing weight. `alpha=0` makes the forward pass bit-identical to `train_dfsa.py`. |
| `--alpha-warmup-steps` | `5000` | Linear ramp of alpha, 0 -> `alpha-max`, after phase 1. |
| `--w-consistency` | `0.5` | Weight of the residual/PEMDAS KL consistency loss. |

## CLI — `eval_dfsa.py` and `test_extreme.py`

Both take `--checkpoint`, `--hidden-size`, `--num-tree-layers`, `--num-heads`,
`--max-len`, `--act-steps`, `--gpu`. `eval_dfsa.py` also takes `--save-dir`
(default `results/nam/dfsa_paren_iter1`) and, when `--checkpoint` is omitted,
globs the newest `**/checkpoints/*.h5` under it. `test_extreme.py` also takes
`--use-ste` and `--use-learned-residual`. Note both default to `--gpu 1`.

Model geometry flags must match the checkpoint you are loading. `eval_dfsa.py`
defaults (`hidden 256 / 3 layers / 8 heads / max-len 128`) do **not** match
`train_dfsa.py` defaults (`64 / 2 / 4 / 64`).

## CLI — `create_dataset.py`

`--output-dir`, `--samples-per-phase`, `--div-zero-samples`, `--seed`.

## What lands on disk

```
<save-dir>/nam_<variant>-<phase>_<timestamp>/     # train_nam.py
  checkpoints/            step_NNNNNN.weights.h5, best.weights.h5, final + JSON sidecars
  digit_matrix/           digit_matrix_step_NNNNNN.csv
  metrics.json
  config.json

<save-dir>/dfsa_<timestamp>/                      # train_dfsa.py
  checkpoints/            step_NNNNNN.weights.h5, final.weights.h5
  metrics.json
  config.json

<save-dir>/dfsa_ste_<timestamp>/                  # train_dfsa_ste.py
  checkpoints/            *.weights.h5
  dfsa_ste_final.weights.h5
```

`--save-dir` is `results` by default and the path is used as given, so launch
from the repo root or pass an absolute path — otherwise you get a second
`results/` tree next to wherever you started.

## Gotchas

- `--clip-norm` is a **global** norm clip, so the loss weights compete. Raising
  one weight suppresses learning in every other head.
- Reduction targeting has to converge before number extraction and operator
  classification can learn anything; that is why `--w-reduction` is an order of
  magnitude above the others in `train_dfsa.py`.
- `data_generator.py` levels 11-12 are the parenthesised ones. They only appear
  once the curriculum progresses far enough, so a short run never sees them.
- Companion note: `PARENTHESIS.md` in this directory records the three failed
  attempts at learned parenthesis handling.
