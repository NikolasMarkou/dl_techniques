# Topographic VAE trainer

Production training pipeline for
`dl_techniques.models.vision.topographic_vae` (Keller & Welling, NeurIPS 2021).

```
MPLBACKEND=Agg .venv/bin/python -m train.topographic_vae.train_topographic_vae --help
```

`MPLBACKEND=Agg` is mandatory — matplotlib's interactive backend crashes headless. This
trainer is **Pattern 7** in `src/train/AGENTS.md`: a generator over unlabelled sequences
whose validation signal is the objective and whose real output is a set of paper columns
rather than a single monitor.

---

## What a run produces

`results/<experiment_name>/`, per the repo-wide contract:

| File | What |
|---|---|
| `config.json` | the run configuration; round-trips back into a config that rebuilds the same model |
| `run.log` | the `dl` logger, tee'd for the **whole** run |
| `training_log.csv` | per-epoch losses from `create_callbacks` |
| `best_model.keras` | best `val_loss` checkpoint |
| `final_model.keras` | the model after training |
| `training_history.json` | the same history, machine-readable |
| `results_summary.json` | every column the paper's tables are read from |
| `visualizations/` | capsule traversals, the topographic map, training curves |
| `model_analysis/` | `run_data_free_analysis` output |

`results/` is gitignored and **unrecoverable** — nothing in a test may write into it.

---

## The numbers in `results_summary.json`

`test_evaluation` carries the paper's four columns:

- **`log_likelihood`** — IWAE estimate of `log p(x)` in nats, summed over pixels and
  sequence, over `--likelihood-samples` importance draws. An **upper bound**: it tightens as
  the sample count grows.
- **`equivariance_error`** — Eq. 13, a *smoothness* measure. Low for an **invariant**
  representation too, so it is never read as equivariance on its own.
- **`capcorr`** — Eq. 15/16, the equivariance measurement. `1.0` is perfectly equivariant.
- **`capcorr_per_capsule`** — the same metric per capsule, which makes the pooled number's
  "all capsules roll together" assumption checkable.

`effective_beta` is reported **next to** `kl_loss_weight` rather than conflated with it: under
the default mean-over-pixels reduction the two differ by the pixel count.

---

## Flags worth knowing

| Flag | Default | Why |
|---|---|---|
| `--dataset` | `mnist` | also fixes the frame size and the available transforms. `--variant` must agree with it; the config refuses a mismatch rather than building a 28×28 decoder over 64×64 frames |
| `--transform` | `rotation` | the single factor the sequences vary. Validated against the *selected* dataset — `--dataset dsprites` offers no `rotation`, so it needs `--transform orientation` |
| `--l-preset` | `third` | `L` as a fraction of `S`; `third` resolves to `L = 6` on MNIST's `S = 18`, the paper's best-equivariance setting |
| `--coherence-window` | unset | explicit `L`, overrides the preset. `None` is the sentinel that lets the preset resolve |
| `--temporal-coherence` | `shifting` | `stationary` is the **BubbleVAE** baseline |
| `--no-variance-variables` | off | drops `u`: the **plain-VAE** baseline. Pairs with `--l-preset none`, since a non-zero `L` needs the energy it removes |
| `--learning-rate` / `--momentum` | `1e-4` / `0.9` | the paper trains with **SGD**, not Adam — which is why the default rate is `1e-4` and not the `1e-3` that would suit Adam |
| `--num-train-sequences` | 4096 | the three splits use seeds `s`, `s+10_000`, `s+20_000`, so no sequence appears in two of them |

The full flag → config-field contract is guarded by
`tests/test_train/test_topographic_vae/test_cli_contract.py`, which drives the real parser and
the real config builder: a flag advertised by `--help` and dropped by `config_from_args` turns
that suite red.

---

## Two baselines, two flags, one model

The paper's ablation is a pair of CLI invocations rather than a separate trainer:

```bash
# the model
--temporal-coherence shifting --l-preset third

# BubbleVAE: same construction, roll removed
--temporal-coherence stationary --l-preset third

# plain VAE: no u, hence no topography, hence L = 0
--no-variance-variables --l-preset none
```

Each run's `baseline` key in the summary says which row of Table 1 it is, so a directory can
be attributed without reading its flags.

---

## Reproducibility

The **data** is seeded (three splits at `s`, `s+10_000`, `s+20_000`) and the **evaluation**
is seeded (the model's `sampling_seed` is pinned for the evaluation's duration and released
after, so a second run in the same process still trains with a free sampler).

The **training** is not bit-reproducible, and that is the design: the ELBO's
reparameterization draws must be unbiased, and `keras.random` with `seed=None` is the only
way to get that. Pinning them would make every batch reuse one noise draw — a biased
gradient rather than a repeatable one.

---

## Data

`dl_techniques/datasets/vision/transform_sequences.py` owns the sequence semantics:
MNIST over rotation / colour / scale at `S = 18`, dSprites over orientation / x / y / scale
at `S = 15`. The warps are pure NumPy — `gen_image_ops.image_projective_transform_v3` was
**measured** non-deterministic in this build — and the dSprites `.npz` is cached under
`/media/arxwn/data0_4tb/datasets/dsprites/` (`--dsprites-cache`). Never delete that cache.

---

## Tests

```
pytest tests/test_train/test_topographic_vae/ -v
```

| File | Covers |
|---|---|
| `test_cli_contract.py` | every flag reaches its config field, with the shared driver |
| `test_config_validation.py` | preset resolution, baseline naming, the split seeds, and every field rejected before a GPU is touched |
| `test_evaluate_model.py` | the four evaluation columns, and the point that `equivariance_error` and `capcorr` disagree |
| `test_train_run.py` | one real end-to-end run and the run-directory contract |