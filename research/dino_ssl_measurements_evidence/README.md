# DINO SSL measurement evidence (60-epoch A/B, 2 arms x 2 seeds)

The `config.json` + `training_log.csv` of the four runs behind the DINO 60-epoch
measurement, produced by plan `plan-2026-08-01T195746-12a1f2db`. Copied verbatim
(sha256-verified, 26,719 bytes total) out of that plan's `results/` run directories.

**These files are the ONLY surviving evidence for the reproduction claim** in
`src/train/dino/train_dino.py`'s Usage docstring and in
`research/2026_dino_ssl_measurements.md`. `results/` is unconditionally gitignored
(`git ls-files results/` = 0), so before this directory existed a single
`rm -rf results/` silently orphaned a published number.

The `best_model.keras` / `final_model.keras` checkpoints from the same run
directories (~7.7 GB) are deliberately NOT tracked. Do not edit these files --
they are a record, not a document.

## Re-deriving the published k20 endpoint

The improved arm's headline endpoint is the mean of the last three EVALUATED
epochs (cadence 4 -> epochs 48/52/56; intervening rows carry the literal string
`nan`, so a "non-empty cell" filter silently returns `nan`):

```bash
.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('research/dino_ssl_measurements_evidence/long_improved_s42/training_log.csv')
e = d.dropna(subset=['dino_knn_top1_k20']).tail(3)
print(list(e['epoch']), repr(e['dino_knn_top1_k20'].mean()))
"
# [48, 52, 56] np.float64(0.4326171875)
```
