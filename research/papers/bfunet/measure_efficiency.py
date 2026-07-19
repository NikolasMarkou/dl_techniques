"""One-off: measure params / FLOPs / peak-memory / latency for the [0,1] ConvUNeXt denoiser.

Produces the efficiency table data for ``bfunet.tex`` (tab:efficiency) for the checkpoint
``results/20260715_convunext_denoiser/best_model.keras``.

What it measures
----------------
1. **Params**: total via ``count_params()``, cross-checked against ``--expected-params``
   (a warn-only sanity gate, default = the default checkpoint's count); trainable vs non-trainable
   (the Gabor stem is frozen, so a non-trivial non-trainable slice is expected).
2. **FLOPs** at 256x256x3 and 512x512x3, via the legacy v1 profiler on a variables-frozen
   concrete function (``tf.function`` -> ``get_concrete_function`` ->
   ``convert_variables_to_constants_v2`` -> ``tf.compat.v1.profiler.profile(...
   float_operation())``). The forward pass runs with ``training=False`` so StochasticDepth
   reduces to identity. Reported as raw FLOPs, GFLOPs, and FLOPs-per-megapixel; the 256->512
   ratio is checked to be ~4x (linear in pixel count) as a sanity gate.
   FALLBACK (plan Pre-Mortem #1): if the v1 profiler raises, returns ``total_float_ops == 0``,
   or the 256->512 ratio is not ~4x, we do NOT invent a number -- we recompute FLOPs by a
   manual per-layer sum (Conv2D: ``2*Hout*Wout*Cout*Cin*Kh*Kw``; DepthwiseConv2D:
   ``2*Hout*Wout*Cin*Kh*Kw``; Dense: ``2*in*out``) and stamp ``method="manual_conv_sum"`` in
   the JSON (else ``method="v1profiler"``).
3. **Peak GPU memory** for a single forward pass at each resolution, via
   ``tf.config.experimental.reset_memory_stats`` + ``get_memory_info("GPU:0")["peak"]``
   (pattern from ``src/train/rms_variants_train/norm_overhead_bench.py``). Reported in MiB.
4. **Latency**: single-image forward-pass wall-time (warm up 3 iters, median of 20). ms.

Run::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        research/papers/bfunet/measure_efficiency.py \\
        [--checkpoint PATH] [--output PATH] [--gpu N] [--expected-params N]

``--checkpoint`` defaults to the module constant below, so an argument-less call
behaves exactly as it always has. The checkpoint actually loaded is recorded in the
output JSON's ``checkpoint`` field -- always read that field, never assume the default.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import keras

from train.common import setup_gpu, set_seeds
from train.bfunet.eval_psnr_vs_noise import load_denoiser, _to_flexible_input

# --- Fixed protocol --------------------------------------------------------------------
CHECKPOINT = "results/20260715_convunext_denoiser/best_model.keras"
SEED = 42
CHANNELS = 3
RESOLUTIONS: List[int] = [256, 512]
LATENCY_WARMUP = 3
LATENCY_ITERS = 20
OUT_JSON = Path(__file__).resolve().parent / "efficiency_results.json"

# Warn-only sanity gate for the DEFAULT checkpoint above. It is checkpoint-specific, so
# pass --expected-params when measuring a different model (or the warn is meaningless).
EXPECTED_TOTAL_PARAMS = 740_784
FOURX_TOLERANCE = 0.30  # accept ratio in [4*(1-tol), 4*(1+tol)] as "~4x linear-in-pixels"


# --- Parameter counts ------------------------------------------------------------------

def _param_counts(model: keras.Model) -> Dict[str, int]:
    """Total / trainable / non-trainable parameter counts."""
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    non_trainable = int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))
    return {
        "total": int(model.count_params()),
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


# --- FLOPs: primary (v1 profiler on a frozen concrete function) ------------------------

def _flops_v1profiler(model: keras.Model, h: int, w: int) -> int:
    """FLOPs via the TF1 profiler on a variables-frozen concrete function.

    Returns ``total_float_ops`` (>= 0). Raises on any incompatibility so the caller can
    fall back to the manual per-layer sum (plan Pre-Mortem #1).
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    @tf.function
    def forward(x):
        return model(x, training=False)  # training=False -> StochasticDepth == identity

    concrete = forward.get_concrete_function(
        tf.TensorSpec([1, h, w, CHANNELS], tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"  # do not dump the per-node table to stdout
        prof = tf.compat.v1.profiler.profile(graph, options=opts)
    return int(prof.total_float_ops) if prof is not None else 0


# --- FLOPs: fallback (manual per-layer sum) --------------------------------------------

def _fixed_input_clone(model: keras.Model, h: int, w: int) -> keras.Model:
    """Rebuild the functional model with a static ``[1, h, w, C]`` input so every conv/dense
    layer exposes a concrete ``output.shape`` for the manual FLOP sum."""
    cfg = model.get_config()
    for layer in cfg.get("layers", []):
        if layer.get("class_name") == "InputLayer":
            lc = layer["config"]
            key = "batch_shape" if "batch_shape" in lc else (
                "batch_input_shape" if "batch_input_shape" in lc else None)
            if key and lc.get(key) and len(lc[key]) == 4:
                b = list(lc[key])
                lc[key] = [1, h, w, b[3]]
    fixed = keras.Model.from_config(cfg)
    fixed.set_weights(model.get_weights())
    return fixed


def _flops_manual(model: keras.Model, h: int, w: int) -> int:
    """Manual FLOP sum over Conv2D / DepthwiseConv2D / Dense (2 * MACs each).

    Recurses into nested sub-models. Uses each layer's static ``output.shape`` (available on
    the fixed-input clone). This is the planned fallback, NOT a hand-invented number.
    """
    fixed = _fixed_input_clone(model, h, w)
    total = 0

    def _out_shape(lyr) -> Optional[Tuple]:
        try:
            return tuple(lyr.output.shape)
        except Exception:  # noqa: BLE001 - layer without a resolved output is skipped
            return None

    def _in_channels(lyr) -> Optional[int]:
        try:
            return int(lyr.input.shape[-1])
        except Exception:  # noqa: BLE001
            return None

    def walk(layers) -> None:
        nonlocal total
        for lyr in layers:
            sub = getattr(lyr, "layers", None)
            if sub:
                walk(sub)
                continue
            cls = lyr.__class__.__name__
            osh = _out_shape(lyr)
            if osh is None:
                continue
            if cls == "Conv2D":
                _, oh, ow, cout = osh
                kh, kw = lyr.kernel_size
                cin = _in_channels(lyr)
                if None in (oh, ow, cout, cin):
                    continue
                total += 2 * int(oh) * int(ow) * int(cout) * int(cin) * int(kh) * int(kw)
            elif cls == "DepthwiseConv2D":
                _, oh, ow, cout = osh
                kh, kw = lyr.kernel_size
                if None in (oh, ow, cout):
                    continue
                total += 2 * int(oh) * int(ow) * int(cout) * int(kh) * int(kw)
            elif cls == "Dense":
                cout = osh[-1]
                cin = _in_channels(lyr)
                if None in (cout, cin):
                    continue
                # collapse any leading spatial/token dims (batch excluded)
                spatial = 1
                for d in osh[1:-1]:
                    if d is not None:
                        spatial *= int(d)
                total += 2 * int(cin) * int(cout) * spatial

    walk(fixed.layers)
    return total


def _measure_flops(model: keras.Model) -> Dict:
    """FLOPs at every RESOLUTION with the v1 profiler, falling back to the manual sum if the
    profiler is unavailable / returns 0 / does not scale ~4x. Returns a dict with the method
    flag and per-resolution FLOP counts + FLOPs-per-megapixel + the 256->512 ratio."""
    method = "v1profiler"
    per_res: Dict[int, int] = {}

    try:
        for res in RESOLUTIONS:
            f = _flops_v1profiler(model, res, res)
            if f <= 0:
                raise RuntimeError(f"v1 profiler returned total_float_ops={f} at {res}")
            per_res[res] = f
        ratio = per_res[512] / per_res[256]
        if not (4.0 * (1 - FOURX_TOLERANCE) <= ratio <= 4.0 * (1 + FOURX_TOLERANCE)):
            raise RuntimeError(
                f"v1 profiler FLOPs do not scale ~4x (512/256 ratio={ratio:.3f})"
            )
    except Exception as exc:  # noqa: BLE001 - any profiler incompatibility -> planned fallback
        print(f"[flops] v1 profiler unusable ({type(exc).__name__}: {exc}); "
              f"falling back to manual per-layer sum")
        method = "manual_conv_sum"
        per_res = {res: _flops_manual(model, res, res) for res in RESOLUTIONS}

    out = {"method": method, "per_resolution": {}}
    for res in RESOLUTIONS:
        flops = per_res[res]
        megapixels = (res * res) / 1.0e6
        out["per_resolution"][str(res)] = {
            "flops": int(flops),
            "gflops": flops / 1.0e9,
            "flops_per_megapixel": flops / megapixels,
        }
    out["ratio_512_over_256"] = per_res[512] / per_res[256]
    return out


# --- Peak memory + latency -------------------------------------------------------------

def _reset_peak_mem() -> None:
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:  # noqa: BLE001
        pass


def _peak_mem_mib() -> float:
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return float(info["peak"]) / (1024.0 * 1024.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[mem] peak unavailable ({type(exc).__name__}: {exc})")
        return float("nan")


def _sync(out) -> None:
    """Force eager execution to complete so latency/peak-mem reflect the real forward pass."""
    t = out[0] if isinstance(out, (list, tuple)) else out
    _ = np.asarray(t)


def _measure_mem_latency(model: keras.Model, res: int) -> Dict:
    x = tf.zeros([1, res, res, CHANNELS], dtype=tf.float32)

    # Warmup (also builds any lazy state and primes the XLA/cuDNN kernels).
    for _ in range(LATENCY_WARMUP):
        _sync(model(x, training=False))

    _reset_peak_mem()
    _sync(model(x, training=False))
    peak = _peak_mem_mib()

    times_ms: List[float] = []
    for _ in range(LATENCY_ITERS):
        t0 = time.perf_counter()
        _sync(model(x, training=False))
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "peak_mem_mib": peak,
        "latency_ms_median": float(np.median(times_ms)),
        "latency_ms_min": float(np.min(times_ms)),
        "latency_ms_max": float(np.max(times_ms)),
        "latency_iters": LATENCY_ITERS,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure params/FLOPs/memory/latency")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
                        help="Path to the saved .keras denoiser.")
    parser.add_argument("--output", type=str, default=str(OUT_JSON),
                        help="Path of the results JSON to write.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id for setup_gpu (default 0).")
    parser.add_argument("--expected-params", type=int, default=EXPECTED_TOTAL_PARAMS,
                        help="Warn-only expected total param count for --checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint
    out_json = Path(args.output)

    setup_gpu(gpu_id=args.gpu)
    set_seeds(SEED)

    model = _to_flexible_input(load_denoiser(checkpoint))

    params = _param_counts(model)
    print(f"params: total={params['total']:,}  trainable={params['trainable']:,}  "
          f"non_trainable={params['non_trainable']:,}")
    if params["total"] != args.expected_params:
        print(f"[warn] total params {params['total']:,} != expected "
              f"{args.expected_params:,}")

    flops = _measure_flops(model)
    print(f"flops method: {flops['method']}  (512/256 ratio={flops['ratio_512_over_256']:.3f})")
    for res in RESOLUTIONS:
        r = flops["per_resolution"][str(res)]
        print(f"  {res}x{res}: {r['gflops']:.3f} GFLOPs  "
              f"({r['flops_per_megapixel'] / 1.0e9:.3f} GFLOPs/megapixel)")

    mem_lat: Dict[str, Dict] = {}
    for res in RESOLUTIONS:
        stats = _measure_mem_latency(model, res)
        mem_lat[str(res)] = stats
        print(f"  {res}x{res}: peak={stats['peak_mem_mib']:.1f} MiB  "
              f"latency(med)={stats['latency_ms_median']:.2f} ms")

    payload = {
        # The checkpoint ACTUALLY loaded (not the module default) -- the paper must key
        # off this field to know which model produced these numbers.
        "checkpoint": checkpoint,
        "seed": SEED,
        "channels": CHANNELS,
        "resolutions": RESOLUTIONS,
        "params": params,
        "flops": flops,
        "mem_latency": mem_lat,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_json}")
    print(f"checkpoint recorded in JSON: {checkpoint}")


if __name__ == "__main__":
    main()
