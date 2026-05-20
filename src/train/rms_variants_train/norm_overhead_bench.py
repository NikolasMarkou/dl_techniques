"""Per-norm compute-overhead benchmark (Refinement A, plan_2026-05-18_6776f8ba).

# DECISION plan_2026-05-18_6776f8ba/D-001:
#   This bench is a standalone reader of the existing
#   ``dl_techniques.layers.norms.factory.create_normalization_layer``. It does NOT
#   add any new layer abstractions, NOT modify any library code, and NOT extend
#   ``NORM_VARIANTS``. Its single purpose is to surface a per-norm compute /
#   parameter / peak-memory snapshot so the Phase 3 overall-recommendation rules
#   in ``report.py`` (Refinement B) can disqualify a norm whose accuracy gain is
#   paid for by > 1.5x step-time at no statistical benefit. Cost: HIGH-LEVERAGE
#   data row at NO library-side risk.

Run standalone (CPU is fine; the bench enables fp16 in mixed_precision when
``--mixed-fp16`` is asked but does not require a GPU):

    cd src && CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m \
        train.rms_variants_train.norm_overhead_bench --out /tmp/overhead.csv

Emits a CSV with one row per norm in ``NORM_VARIANTS``:

    norm, params, mean_step_ms_fp32, mean_step_ms_fp16,
    peak_mem_mb_fp32, peak_mem_mb_fp16

Peak-memory readings rely on ``tf.config.experimental.get_memory_info('GPU:0')``
when a GPU is visible; on CPU or older TF builds the column is recorded as NaN
(per Failure-Modes table in plan.md). The bench never aborts on telemetry
absence — it logs a WARNING and emits NaN.

Plan: ``plans/plan_2026-05-18_6776f8ba`` (step 2; SC2; D-001 anchor).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BATCH: int = 64
DEFAULT_FEATURES: int = 512
DEFAULT_WARMUP: int = 50
DEFAULT_ITERS: int = 1000


# ---------------------------------------------------------------------------
# Per-norm builder
# ---------------------------------------------------------------------------

def _build_layer(norm_type: str):
    """Construct a single normalization layer instance via the central factory.

    Uses ``build_norm_kwargs`` so per-variant quirks (no ``use_scale`` on
    AdaptiveBandRMS; disabled default L2 on BandRMS) are honoured without
    duplicating the dispatcher. The bench measures the *default* configuration
    of each norm (use_scale=True, max_band_width=0.1, epsilon=1e-6).
    """
    # Local imports keep CLI import cost low when --help is requested.
    from dl_techniques.layers.norms.factory import create_normalization_layer
    from train.rms_variants_train.config import build_norm_kwargs

    kwargs = build_norm_kwargs(norm_type)
    return create_normalization_layer(norm_type, **kwargs)


def _count_params(layer) -> int:
    """Count trainable + non-trainable parameters AFTER the layer is built."""
    return int(sum(int(np.prod(w.shape)) for w in layer.weights))


# ---------------------------------------------------------------------------
# Peak memory helper (safe on CPU + older TF)
# ---------------------------------------------------------------------------

def _peak_mem_mb_or_nan() -> float:
    """Return current peak GPU memory in MB, or NaN if unavailable.

    Wrapped in try/except per plan.md Failure-Modes row: TF version skew can
    remove this API; running on CPU returns nothing meaningful either.
    """
    import tensorflow as tf
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return float(info["peak"]) / (1024.0 * 1024.0)
    except Exception as exc:  # noqa: BLE001 — telemetry must NEVER abort the bench
        logger.warning(f"peak_mem unavailable ({type(exc).__name__}): {exc}")
        return float("nan")


def _reset_peak_mem() -> None:
    import tensorflow as tf
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Single norm benchmark
# ---------------------------------------------------------------------------

def _time_norm(
    norm_type: str,
    *,
    batch: int,
    features: int,
    iters: int,
    warmup: int,
    dtype: str,
) -> Tuple[float, float, int]:
    """Time a forward+backward pass for ``norm_type`` at the given dtype.

    Returns ``(mean_step_ms, peak_mem_mb, params)``.

    ``dtype`` is one of ``"fp32"`` or ``"fp16"``. fp16 enables Keras mixed
    precision globally (``mixed_float16`` policy) for the duration of the call.
    """
    import tensorflow as tf
    import keras

    # Set mixed-precision policy LOCALLY around this measurement. Saving and
    # restoring the previous global policy keeps the bench safe to call
    # repeatedly with alternating dtypes.
    prev_policy = keras.mixed_precision.global_policy()
    try:
        if dtype == "fp16":
            keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            keras.mixed_precision.set_global_policy("float32")

        layer = _build_layer(norm_type)
        # Force build so weight count is meaningful.
        sample = tf.random.normal((batch, features), dtype=tf.float32)
        _ = layer(sample)
        params = _count_params(layer)

        _reset_peak_mem()

        @tf.function(reduce_retracing=True)
        def step(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = layer(x)
                loss = tf.reduce_mean(tf.square(y))
            grads = tape.gradient(loss, layer.trainable_weights or [x])
            # Touch the gradients so TF actually computes them in tf.function.
            if grads is None or len(grads) == 0:
                return loss
            # Cast the gradient touch-term to loss dtype: under mixed_float16
            # `loss` is fp16 while `grads[0]` may be fp32 (AddV2 dtype clash).
            return loss + tf.cast(tf.reduce_sum(grads[0]), loss.dtype) * 0.0

        # Warmup
        for _ in range(max(1, warmup)):
            _ = step(sample)

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = step(sample)
        elapsed_s = time.perf_counter() - t0

        mean_ms = (elapsed_s / max(1, iters)) * 1000.0
        peak_mb = _peak_mem_mb_or_nan()
        return mean_ms, peak_mb, params
    finally:
        keras.mixed_precision.set_global_policy(prev_policy)


# ---------------------------------------------------------------------------
# Public entry point (also used by tests)
# ---------------------------------------------------------------------------

def run_bench(
    norm_types: Tuple[str, ...],
    *,
    batch: int = DEFAULT_BATCH,
    features: int = DEFAULT_FEATURES,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    include_fp16: bool = True,
) -> List[Dict[str, float]]:
    """Run the bench across all ``norm_types`` and return a list of row-dicts.

    Each row matches the CSV header:
        norm, params, mean_step_ms_fp32, mean_step_ms_fp16,
        peak_mem_mb_fp32, peak_mem_mb_fp16

    fp16 columns are recorded as NaN when ``include_fp16=False``.
    """
    rows: List[Dict[str, float]] = []
    for nt in norm_types:
        logger.info(f"benching norm={nt} (fp32)")
        ms32, mem32, params = _time_norm(
            nt, batch=batch, features=features, iters=iters, warmup=warmup, dtype="fp32"
        )
        if include_fp16:
            logger.info(f"benching norm={nt} (fp16)")
            ms16, mem16, _ = _time_norm(
                nt, batch=batch, features=features, iters=iters, warmup=warmup, dtype="fp16"
            )
        else:
            ms16 = float("nan")
            mem16 = float("nan")
        rows.append({
            "norm": nt,
            "params": params,
            "mean_step_ms_fp32": ms32,
            "mean_step_ms_fp16": ms16,
            "peak_mem_mb_fp32": mem32,
            "peak_mem_mb_fp16": mem16,
        })
    return rows


def write_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    """Write the bench rows to ``out_path`` with the canonical column order."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fieldnames = [
        "norm",
        "params",
        "mean_step_ms_fp32",
        "mean_step_ms_fp16",
        "peak_mem_mb_fp32",
        "peak_mem_mb_fp16",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-norm compute-overhead bench.")
    p.add_argument("--out", type=str, default="overhead.csv", help="Output CSV path.")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--features", type=int, default=DEFAULT_FEATURES)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip the fp16 mixed-precision pass (fp16 columns will be NaN).",
    )
    p.add_argument(
        "--norms",
        type=str,
        default="",
        help="Comma-separated norm-types; default = canonical NORM_VARIANTS (all 8).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from train.rms_variants_train.config import NORM_VARIANTS

    if args.norms.strip():
        norm_types = tuple(s.strip() for s in args.norms.split(",") if s.strip())
    else:
        norm_types = NORM_VARIANTS

    logger.info(
        f"norm_overhead_bench: batch={args.batch} features={args.features} "
        f"iters={args.iters} warmup={args.warmup} fp16={not args.no_fp16}"
    )
    rows = run_bench(
        norm_types,
        batch=args.batch,
        features=args.features,
        iters=args.iters,
        warmup=args.warmup,
        include_fp16=not args.no_fp16,
    )
    write_csv(rows, args.out)
    logger.info(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
