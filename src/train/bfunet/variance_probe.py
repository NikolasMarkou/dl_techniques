"""Production-scale variance / gradient-flow probe: learned 1x1-conv channel
projection vs weightless zero-pad ``MatchChannels``.

Motivation
----------
The bfunet denoiser's ``--zero-pad-channels`` / ``--extra-zero-output-channels``
flags replace the *default* learned 1x1 Conv2D channel projections with weightless
zero-pad / slice ``MatchChannels`` maps. An epistemic-deconstructor analysis
(``analyses/analysis_2026-07-02_eb904fae/summary.md``) established that at
CONVERGENCE the two are quality-neutral (equal loss, ~6k fewer params). BUT the
design's ACTUAL goal was training STABILITY: the original 1x1-conv path showed
gradient-flow issues and high run-to-run variance, and the zero-init padded
channels were intended as "scratch pads" the network writes into. That variance
claim could not be tested on a toy proxy (both variants were too stable). This
script tests it where it matters: the REAL model + REAL data pipeline + multiple
seeds, measuring run-to-run variance, loss-trajectory roughness, and gradient-norm
stability — NOT just the mean endpoint.

What it does
------------
For each ``--seed`` and each of two conditions (``baseline`` = flag OFF, learned
1x1 conv; ``variant`` = flag ON, weightless MatchChannels), it:
  1. seeds Python/NumPy/TF (``train.common.set_seeds``),
  2. builds the REAL denoiser via ``train_convunext_denoiser.build_model``,
  3. streams the REAL COCO+DIV2K ``(noisy, clean)`` patch pipeline
     (``train.bfunet.common``) at a FIXED noise sigma (curriculum disabled so the
     noise distribution is constant across steps/runs — isolates the architectural
     variance),
  4. trains ``--steps`` steps with AdamW (+ grad clipping), a custom loop that
     records the pre-clip global gradient norm each logged step,
  5. scores eval PSNR/MSE every ``--log-every`` steps on ONE fixed val batch
     (identical clean patches + identical noise across all runs/conditions).

Then it aggregates ACROSS seeds per condition and reports the contrasts that answer
the design question:
  - final-metric across-seed std / CV      (run-to-run variance)
  - loss-trajectory roughness              (smoothness)
  - gradient-norm CV                       (gradient-flow steadiness)

The flag under test defaults to ``zero_pad_channels`` (the 1x1-vs-zeropad question);
``--compare extra_zero_output_channels`` toggles the output-path flag instead.

Scale note
----------
Defaults (variant=small, patch=128, 5 seeds, 1500 steps) are a representative
compromise that finishes in ~20-40 min on a 12 GB GPU. The toy-proxy lesson from
the analysis is that TOO-small/TOO-stable a setup cannot reveal a variance
difference — so run at your production ``--variant base --patch-size 256`` (fewer
seeds if GPU-bound) when you want the definitive answer. Nothing here is toy: it is
the production model and data, only the step budget is bounded.

Usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.variance_probe \\
        --variant small --patch-size 128 --seeds 5 --steps 1500 --sigma 0.1 --gpu 1

    # Production scale (heavier):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.variance_probe \\
        --variant base --patch-size 256 --batch-size 4 --seeds 3 --steps 3000 --gpu 1
"""

import os
# Enable GPU memory growth via env BEFORE TF initializes devices (module imports below
# touch TF, after which tf.config.set_memory_growth would raise "cannot be modified after
# initialized"). This is the robust way to get memory growth in a script whose imports
# init the GPU before main() runs.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import glob
import json
import time
import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Dict, Optional

import numpy as np
import keras
import tensorflow as tf

from train.common import setup_gpu, set_seeds
from dl_techniques.utils.logger import logger

from train.bfunet import common as common
from train.bfunet.common import (
    collect_training_paths, create_dataset, make_curriculum_noise_fn,
    build_fixed_val_batch, _mean_psnr, DATA_MIN, DATA_MAX,
)
from train.bfunet.train_convunext_denoiser import TrainingConfig, build_model


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _collect_val_paths(config: TrainingConfig) -> List[str]:
    """Glob a handful of validation image paths from the config's val dirs."""
    paths: List[str] = []
    for d in config.val_image_dirs:
        for ext in config.image_extensions:
            paths.extend(sorted(glob.glob(os.path.join(d, f"*{ext}")))[:64])
        if paths:
            break
    return paths


def _fixed_eval_batch(config: TrainingConfig, sigma: float, n: int = 8):
    """Build ONE fixed (noisy, clean) eval batch, identical across all runs.

    Clean patches come from the real val set; the noise is drawn once with a fixed
    numpy seed so every run/condition is scored on byte-identical inputs.
    """
    val_paths = _collect_val_paths(config)
    clean = build_fixed_val_batch(val_paths, config, n=n)
    if clean is None:
        logger.warning("No val patches found; falling back to a synthetic eval batch.")
        rng = np.random.default_rng(12345)
        clean = tf.constant(
            np.clip(rng.uniform(DATA_MIN, DATA_MAX, (n, config.patch_size, config.patch_size,
                                                     config.channels)),
                    DATA_MIN, DATA_MAX).astype(np.float32)
        )
    rng = np.random.default_rng(777)
    noise = rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    noisy = tf.clip_by_value(clean + noise, DATA_MIN, DATA_MAX)
    return tf.constant(noisy), tf.constant(clean)


def _make_config(args, zero_pad: bool, extra_zero: bool, seed: int) -> TrainingConfig:
    """A production TrainingConfig with the noise curriculum FROZEN to a fixed sigma."""
    return TrainingConfig(
        variant=args.variant,
        convnext_version=args.convnext_version,
        use_gabor_stem=not args.no_gabor_stem,
        use_laplacian_pyramid=args.laplacian_pyramid,
        zero_pad_channels=zero_pad,
        extra_zero_output_channels=extra_zero,
        block_normalization=args.block_normalization,
        patch_size=args.patch_size,
        channels=args.channels,
        batch_size=args.batch_size,
        patches_per_image=args.patches_per_image,
        max_train_files=args.max_train_files,
        seed=seed,
        # Freeze the curriculum to an (effectively) constant sigma so the noise
        # distribution is identical every step and every run (isolates architecture-driven
        # variance). The config validator requires sigma_max_end > noise_sigma_min, so use a
        # negligibly-narrow band [0.999*sigma, sigma] rather than an exact point.
        noise_type="additive",
        noise_sigma_min=args.sigma * 0.999,
        sigma_max_start=args.sigma,
        sigma_max_end=args.sigma,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        # keep the run lean; these do not affect the streaming train pipeline shape
        epochs=1,
        steps_per_epoch=args.steps,
        augment_data=True,
    )


def _run_one(args, condition: str, zero_pad: bool, extra_zero: bool, seed: int,
             eval_noisy, eval_clean) -> Dict:
    """Train one (condition, seed) run and return its trajectory + summary metrics."""
    set_seeds(seed)
    config = _make_config(args, zero_pad, extra_zero, seed)
    model = build_model(config)

    # "Free Gabor": the factory always builds the Gabor stem FROZEN (trainable=False).
    # Flip it to trainable so the Gabor-initialized depthwise bank is learned. Done before
    # the training loop so model.trainable_variables (read each step) picks up the kernel.
    if args.trainable_gabor:
        flipped = 0
        for layer in model._flatten_layers():
            if layer.name == "gabor_stem":
                layer.trainable = True
                flipped += 1
        logger.info(f"[{condition} seed={seed}] free-Gabor: set {flipped} gabor_stem layer(s) trainable")

    optimizer = keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clipnorm=args.gradient_clipping,
    )

    sigma_var = tf.Variable(args.sigma, dtype=tf.float32, trainable=False)
    noise_fn = make_curriculum_noise_fn(config, sigma_var)
    train_paths = collect_training_paths(config)
    ds = create_dataset(train_paths, config, noise_fn, is_training=True)

    @tf.function
    def train_step(noisy, clean):
        with tf.GradientTape() as tape:
            pred = model(noisy, training=True)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            loss = tf.reduce_mean(tf.square(pred - clean))
        gv = model.trainable_variables
        grads = tape.gradient(loss, gv)
        pairs = [(g, v) for g, v in zip(grads, gv) if g is not None]
        gnorm = tf.linalg.global_norm([g for g, _ in pairs])  # pre-clip grad-flow health
        optimizer.apply_gradients(pairs)
        return loss, gnorm

    @tf.function
    def eval_mse():
        pred = model(eval_noisy, training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return tf.reduce_mean(tf.square(pred - eval_clean))

    eval_traj, gnorm_traj = [], []
    t0 = time.time()
    step = 0
    for noisy, clean in ds:
        _, gnorm = train_step(noisy, clean)
        if step % args.log_every == 0 or step == args.steps - 1:
            eval_traj.append((step, float(eval_mse().numpy())))
            gnorm_traj.append((step, float(gnorm.numpy())))
        step += 1
        if step >= args.steps:
            break
    wall = time.time() - t0

    ev = [v for _, v in eval_traj]
    gn = [v for _, v in gnorm_traj]
    log_ev = [np.log(max(v, 1e-12)) for v in ev]
    roughness = float(np.std(np.diff(log_ev))) if len(log_ev) > 1 else 0.0
    final_mse = ev[-1]
    final_psnr = _mean_psnr(
        model(eval_noisy, training=False)[0] if isinstance(model(eval_noisy, training=False), (list, tuple))
        else model(eval_noisy, training=False), eval_clean)

    logger.info(
        f"[{condition} seed={seed}] final_mse={final_mse:.6f} final_psnr={final_psnr:.3f}dB "
        f"roughness={roughness:.4f} gnorm_cv={np.std(gn)/(np.mean(gn)+1e-12):.4f} "
        f"params={model.count_params()} wall={wall:.0f}s"
    )
    return {
        "condition": condition, "seed": seed, "params": int(model.count_params()),
        "final_mse": final_mse, "final_psnr": final_psnr,
        "eval_traj": eval_traj, "gnorm_traj": gnorm_traj,
        "roughness_logmse": roughness,
        "gnorm_mean": float(np.mean(gn)), "gnorm_std": float(np.std(gn)),
        "gnorm_cv": float(np.std(gn) / (np.mean(gn) + 1e-12)),
        "wall_s": wall,
    }


def _summarize(runs: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for cond in ("baseline", "variant"):
        rs = [r for r in runs if r["condition"] == cond]
        if not rs:
            continue
        fm = [r["final_mse"] for r in rs]
        fp = [r["final_psnr"] for r in rs]
        std = pstdev(fm) if len(fm) > 1 else 0.0
        out[cond] = {
            "n_seeds": len(rs),
            "final_mse_mean": mean(fm), "final_mse_std": std,
            "final_mse_cv": std / (mean(fm) + 1e-12),
            "final_psnr_mean": mean(fp),
            "final_psnr_std": pstdev(fp) if len(fp) > 1 else 0.0,
            "roughness_mean": mean(r["roughness_logmse"] for r in rs),
            "gnorm_cv_mean": mean(r["gnorm_cv"] for r in rs),
            "params": rs[0]["params"],
        }
    return out


def _contrasts(summary: Dict) -> Dict:
    if "baseline" not in summary or "variant" not in summary:
        return {}
    b, v = summary["baseline"], summary["variant"]
    def ratio(a, c):
        return a / c if c else float("inf")
    return {
        "final_mse_std_ratio_var_over_base": ratio(v["final_mse_std"], b["final_mse_std"]),
        "final_mse_cv_ratio_var_over_base": ratio(v["final_mse_cv"], b["final_mse_cv"]),
        "roughness_ratio_var_over_base": ratio(v["roughness_mean"], b["roughness_mean"]),
        "gnorm_cv_ratio_var_over_base": ratio(v["gnorm_cv_mean"], b["gnorm_cv_mean"]),
        "final_mse_mean_ratio_var_over_base": ratio(v["final_mse_mean"], b["final_mse_mean"]),
        "param_delta_var_minus_base": v["params"] - b["params"],
        "interpretation": (
            "ratio < 1 => the weightless variant is MORE stable / steadier than the "
            "1x1-conv baseline on that axis; > 1 => less stable. gnorm_cv_ratio < 1 and "
            "final_mse_std_ratio < 1 together would support the variance-reduction design goal."
        ),
    }


def _plot(runs: List[Dict], out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        logger.warning(f"matplotlib unavailable, skipping plot: {e}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"baseline": "tab:blue", "variant": "tab:orange"}
    for cond in ("baseline", "variant"):
        rs = [r for r in runs if r["condition"] == cond]
        for r in rs:
            steps = [s for s, _ in r["eval_traj"]]
            ax1.plot(steps, [v for _, v in r["eval_traj"]], color=colors[cond],
                     alpha=0.5, lw=1)
            ax2.plot([s for s, _ in r["gnorm_traj"]], [v for _, v in r["gnorm_traj"]],
                     color=colors[cond], alpha=0.5, lw=1)
    for ax, title, ylab in ((ax1, "Eval MSE trajectory (per seed)", "eval MSE"),
                            (ax2, "Gradient-norm trajectory (per seed)", "global grad norm")):
        ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.set_yscale("log")
    handles = [plt.Line2D([0], [0], color=colors[c], label=lbl) for c, lbl in
               (("baseline", "baseline (1x1 conv)"), ("variant", "zero-pad / MatchChannels"))]
    ax1.legend(handles=handles)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    logger.info(f"Saved plot: {out_png}")


def _write_report(summary: Dict, contrasts: Dict, args, out_md: Path):
    lines = ["# MatchChannels variance / gradient-flow probe", ""]
    lines.append(f"- flag under test: `{args.compare}` (baseline=OFF/1x1-conv, variant=ON/weightless)")
    lines.append(f"- variant={args.variant} patch={args.patch_size} sigma={args.sigma} "
                 f"steps={args.steps} seeds={args.seeds} batch={args.batch_size}")
    lines.append(f"- laplacian_pyramid={args.laplacian_pyramid} free_gabor={args.trainable_gabor} "
                 f"(applied to BOTH conditions; only `{args.compare}` differs)")
    lines.append("")
    lines.append("| metric | baseline (1x1) | variant (zero-pad) |")
    lines.append("|--------|----------------|--------------------|")
    if summary:
        b = summary.get("baseline", {}); v = summary.get("variant", {})
        def row(name, key, fmt="{:.6f}"):
            lines.append(f"| {name} | {fmt.format(b.get(key, float('nan')))} | "
                         f"{fmt.format(v.get(key, float('nan')))} |")
        row("final MSE mean", "final_mse_mean")
        row("final MSE across-seed std", "final_mse_std")
        row("final MSE CV", "final_mse_cv", "{:.4f}")
        row("final PSNR mean (dB)", "final_psnr_mean", "{:.3f}")
        row("trajectory roughness", "roughness_mean", "{:.4f}")
        row("gradient-norm CV", "gnorm_cv_mean", "{:.4f}")
        row("params", "params", "{:d}")
    lines.append("")
    lines.append("## Contrasts (variant / baseline; <1 => variant more stable)")
    for k, val in contrasts.items():
        if isinstance(val, float):
            lines.append(f"- {k}: {val:.3f}")
        elif isinstance(val, int):
            lines.append(f"- {k}: {val}")
    lines.append("")
    lines.append(contrasts.get("interpretation", ""))
    out_md.write_text("\n".join(lines))
    logger.info(f"Saved report: {out_md}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed variance / gradient-flow probe: 1x1-conv vs weightless "
                    "MatchChannels channel projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--compare", choices=["zero_pad_channels", "extra_zero_output_channels"],
                   default="zero_pad_channels",
                   help="Which flag to toggle ON for the 'variant' condition.")
    p.add_argument("--variant", default="small", help="Model size variant (tiny/small/base/...).")
    p.add_argument("--convnext-version", choices=["v1", "v2"], default="v1")
    p.add_argument("--no-gabor-stem", action="store_true")
    p.add_argument("--block-normalization", choices=["layernorm", "batchnorm"], default="batchnorm")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patches-per-image", type=int, default=4)
    p.add_argument("--max-train-files", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..seeds-1).")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=None,
                   help="If set, steps = epochs * steps-per-epoch (overrides --steps).")
    p.add_argument("--steps-per-epoch", type=int, default=500,
                   help="Steps per epoch when --epochs is used.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--laplacian-pyramid", action="store_true",
                   help="Use the Laplacian-pyramid downsample/skip path in BOTH conditions.")
    p.add_argument("--trainable-gabor", action="store_true",
                   help="Train the (otherwise frozen) Gabor stem in BOTH conditions ('free Gabor').")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Fixed noise sigma on the [0,1] domain (curriculum frozen). "
                        "Unchanged by the domain migration: peak-to-peak width is 1.0 "
                        "in both the old and the new domain.")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.004)
    p.add_argument("--gradient-clipping", type=float, default=1.0)
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: results/variance_probe_<compare>_<variant>).")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_arguments()
    # Resolve epoch spec -> steps (epochs override --steps).
    if args.epochs is not None:
        args.steps = args.epochs * args.steps_per_epoch
    setup_gpu(gpu_id=args.gpu)

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/variance_probe_{args.compare}_{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the fixed eval batch ONCE (needs a throwaway config for patch size/dirs).
    eval_cfg = _make_config(args, zero_pad=False, extra_zero=False, seed=0)
    eval_noisy, eval_clean = _fixed_eval_batch(eval_cfg, args.sigma, n=8)

    zp_flag = args.compare == "zero_pad_channels"
    ez_flag = args.compare == "extra_zero_output_channels"

    logger.info(
        f"Variance probe: compare={args.compare}, variant={args.variant}, patch={args.patch_size}, "
        f"seeds={args.seeds}, steps={args.steps}, sigma={args.sigma}. Output -> {out_dir}"
    )

    runs: List[Dict] = []
    for seed in range(args.seeds):
        # baseline = both flags OFF (learned 1x1 conv); variant = the tested flag ON.
        runs.append(_run_one(args, "baseline", False, False, seed, eval_noisy, eval_clean))
        json.dump({"runs": runs}, open(out_dir / "variance_probe_result.json", "w"), indent=2)
        runs.append(_run_one(args, "variant",
                             zp_flag, ez_flag, seed, eval_noisy, eval_clean))
        json.dump({"runs": runs}, open(out_dir / "variance_probe_result.json", "w"), indent=2)

    summary = _summarize(runs)
    contrasts = _contrasts(summary)
    json.dump({"args": vars(args), "runs": runs, "summary": summary, "contrasts": contrasts},
              open(out_dir / "variance_probe_result.json", "w"), indent=2)
    _plot(runs, out_dir / "variance_probe_trajectories.png")
    _write_report(summary, contrasts, args, out_dir / "variance_probe_report.md")

    logger.info("=== CONTRASTS (variant/baseline; <1 => variant more stable) ===")
    for k, val in contrasts.items():
        if isinstance(val, (int, float)):
            logger.info(f"  {k}: {val}")
    logger.info("Variance probe complete.")


if __name__ == "__main__":
    main()
