# Plan Summary — LeWM PyTorch → Keras 3 Port

**Plan ID**: `plan_2026-04-21_8416bc0b`
**Date**: 2026-04-21
**Outcome**: CLOSED (success, single iteration, zero fix attempts)

## Goal
Port LeWM (JEPA-based action-conditioned world model) from PyTorch to the
`dl_techniques` Keras 3 framework. Deliver a runnable smoke-test training
script on synthetic data, a round-trip-serializable `keras.Model`, and a thin
HDF5 loader skeleton matching the PushT schema.

## Outcome

All 6 success criteria (C1–C6) PASS on first pass. 10 plan steps executed
sequentially; no steps revisited, no autonomy-leash hits, no pivots.

- `pytest tests/test_layers/test_adaln_zero.py tests/test_regularizers/test_sigreg.py tests/test_models/test_lewm.py -vvv` → 11/11 passed (20.53 s, CPU).
- `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm --synthetic --batch-size=2 --epochs=1 --steps-per-epoch=2` → exit 0, final loss 0.8782 (finite) on RTX 4070. `training_log.csv` + `final_model.keras` (~77 MB) + `last.keras` written under `results/lewm_20260421_130931/`.

## Deliverables

13 new files, zero modifications to existing files.

**Library layers / regularizers**
- `src/dl_techniques/layers/adaln_zero.py` — `AdaLNZeroConditionalBlock` (zero-init identity verified).
- `src/dl_techniques/regularizers/sigreg.py` — `SIGRegLayer` (stateless random-projection characteristic-function regularizer).

**Model package `src/dl_techniques/models/lewm/`**
- `__init__.py`
- `config.py` — `LeWMConfig` dataclass (embed_dim=192, depth=6, heads=16, sigreg_weight=0.09, …).
- `embedder.py` — `ActionEmbedder` (Conv1D k=1 + 2-layer SiLU MLP).
- `projector.py` — `MLPProjector` (Dense → LayerNorm → GELU → Dense). **D-002**: LayerNorm, not BatchNorm — matches upstream `/tmp/lewm_source/module.py:159-172`.
- `predictor.py` — `ARPredictor` (learned pos embedding + AdaLN-zero stack + causal attention).
- `model.py` — `LeWM` keras.Model with `encode`, `predict`, `rollout`, `call`, `get_config` / `from_config`. **D-001**: live target encoder (no EMA, no stop-gradient) — matches upstream `/tmp/lewm_source/jepa.py:29-45`.

**Data + training**
- `src/dl_techniques/datasets/pusht_hdf5.py` — `SyntheticLeWMDataset` (smoke) + `PushTHDF5Dataset` (skeleton, h5py, NaN→0 on actions, ImageNet normalization).
- `src/train/lewm/__init__.py`
- `src/train/lewm/train_lewm.py` — argparse, `CSVLogger` + `TerminateOnNaN` + `ModelCheckpoint`, standard `model.fit`; loss via `add_loss` inside `call`.

**Tests**
- `tests/test_layers/test_adaln_zero.py` — build + identity-at-init + save/load round-trip.
- `tests/test_regularizers/test_sigreg.py` — scalar finite output.
- `tests/test_models/test_lewm.py` — shapes + rollout + serialization + AdaLN identity at init.

## Key Decisions

- **D-001** (anchored `models/lewm/model.py:177`): target encoder is live, not EMA. Gradients flow through both context and target paths. Matches upstream; do not "fix" by adding stop-gradient.
- **D-002** (anchored `models/lewm/projector.py:11`): `MLPProjector` uses `LayerNormalization`, not `BatchNormalization`. Plan paraphrase was wrong; upstream code defaults to LayerNorm. Side-benefit: dodges BN-batch-of-1 failure mode.

## Commit History (iter-1)

| Step | Commit  | Description                                   |
|------|---------|-----------------------------------------------|
| 1    | 866eb82 | Scaffold lewm model + training package        |
| 2    | 0db2b16 | AdaLNZeroConditionalBlock layer + test        |
| 3    | 9f15dfc | SIGRegLayer + test                            |
| 4    | 8812989 | ActionEmbedder + MLPProjector                 |
| 5    | 7c1cfeb | ARPredictor                                   |
| 6    | 67d4c61 | LeWMConfig + LeWM keras.Model                 |
| 7    | 02887bf | Synthetic + HDF5 dataset module               |
| 8    | 754e0dc | train_lewm.py smoke-test training script      |
| 9    | 36fca55 | Unit tests (test_lewm.py)                     |
| 10   | —       | Smoke-test execution + REFLECT                |

Checkpoint: `cp-000-iter1.md` → pre-change commit `8493641`.

## Not Verified / Out of Scope (follow-up plans)

- **Real PushT HDF5 data** — loader is a skeleton; no local data available.
- **Full-resolution convergence run** (224×224, depth=6, sigreg_num_proj=1024). Smoke test uses 56×56 / depth=2 / num_proj=64 for CPU-friendly wall time.
- **`eval.py` / MPC / planning** — depends on `stable_worldmodel`.
- **bf16 / multi-GPU**.
- **LR schedule beyond default AdamW**.
- **EMA target encoder variant** — upstream does not use; noted as future ablation.

## Lessons Learned (feeds `plans/LESSONS.md`)

- **JEPA variants ≠ JEPA variant** — `models/jepa/` (masked I/V-JEPA) and `models/lewm/` (action-conditioned) share lineage but not architecture. Always make new model packages for new architectures rather than extending a close-but-wrong neighbour.
- **Plan paraphrase can drift from source code** — Step 4 planned "Linear → BatchNorm → GELU → Linear". Upstream `module.py:159-172` defaults to LayerNorm. Always re-read actual source at implementation time; don't trust the plan's paraphrase of findings.
- **AdaLN-zero identity-at-init is a cheap invariant test** — verifying `block(x, c) ≈ x` within `atol=1e-6` after build (before any training) catches zero-init bugs immediately.
- **`add_loss` inside `call` composes with `model.fit`** — no custom training loop needed for JEPA-style multi-term losses in Keras 3.8. Just call `self.add_loss(scalar_tensor)` and the reported loss in the CSV log includes it automatically.
- **Stateless random projections in a Keras layer** — `keras.random.normal` inside `call` is fine; non-determinism across forward passes matches upstream SIGReg and is not a bug.
- **13-file plan closed in a single iteration with zero fix attempts** — this works when (a) EXPLORE produces a complete file inventory, (b) PLAN steps are ordered so dependencies are satisfied, (c) the smoke-test command is pre-written in the plan, (d) serialization round-trip is a first-class criterion, not an afterthought.
