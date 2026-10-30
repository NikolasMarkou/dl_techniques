"""Experiment 2: MeanShift Attention against multi_head attention in a small ViT, with an ablation grid.

The model is a ``VisionEncoder`` (32x32 input, patch 4, width 192, depth 6, 3 heads, MLP
ratio 4, pre-norm, no CLS token, mean output) followed by a ``Dense`` logits layer. Only the
attention layer differs between configs. Parameters are reported, not matched: tied Q, K and
V make the MeanShift stack about 16 % smaller than multi_head at the same width and depth.

The configs are the table :data:`CONFIGS`: ``mha`` (multi_head) and 16 MeanShift Attention
configs, the product of blurring {off, on}, steps {1, 3}, bandwidth mode {per_head,
per_token} and kernel {gaussian, vmf}. Every vMF config starts its bandwidth at 0.3.
:data:`CONTROL_CONFIGS` holds configs that are not part of that 17-cell grid: ``mha_oinit``
is ``mha`` with an independent initializer for the output projection (``mha`` replays one
draw for ``qkv`` and ``proj``, the MeanShift projections are independent), reachable only
through ``--mode main --configs mha_oinit``.

A cell trains with the shared recipe and scores its final-epoch weights. The row carries the
parameter count, the loss history, top-1 on the selection split (a held-out slice of TRAIN,
the only split a ranking may read) and on the test split, throughput in images per second
measured on the test split after training, and the learned bandwidths (per layer and head
for ``per_head``; mean, min and max per layer over 512 selection images for ``per_token``,
read through the layer's own ``_bandwidth`` helper).

Driver modes (``--mode``), one gated subprocess per cell::

    smoke  mha and the default MeanShift config (blurring, 3 steps, per_head, gaussian), 1 seed
    grid   mha and the 16 MeanShift configs, 1 seed, cheapest first
    main   mha and the configs named by --configs (a grid or control id), 3 seeds

    MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \\
        --mode grid --out-dir results/mean_shift/exp2_grid --epochs 10

The worker (``--worker``, started by the driver) runs one cell. The cell ``arm`` is the
attention type (``multi_head`` or ``mean_shift``) and ``config`` is a key of ``CONFIGS``.
Run from the repository root so a relative ``--out-dir`` lands in the repo-root ``results/``.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from dl_techniques.utils.logger import logger
from train.mean_shift import common

EXPERIMENT = "exp2"
WORKER_MODULE = "train.mean_shift.exp2_mean_shift_attention"
REFERENCE_CONFIG = "mha"
DEFAULT_MEAN_SHIFT_CONFIG = "blur_s3_per_head_gaussian"
MODES = ("smoke", "grid", "main")
PROBE_IMAGES = 512
VMF_BANDWIDTH_INIT = 0.3
NOT_APPLICABLE = "not applicable: the model has no MeanShiftAttention layer"


def _grid_id(blurring: bool, steps: int, mode: str, kernel: str) -> str:
    return f"{'blur' if blurring else 'noblur'}_s{steps}_{mode}_{kernel}"


def _grid_config(blurring: bool, steps: int, mode: str, kernel: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "num_steps": steps, "blurring": blurring, "bandwidth_mode": mode, "kernel": kernel}
    if kernel == "vmf":
        # DECISION plan-2026-10-30T025213-f75ab4e9/D-006
        # Every vMF config starts at bandwidth_init=0.3: the layer default is nearly uniform on unit vectors; do NOT drop it.
        args["bandwidth_init"] = VMF_BANDWIDTH_INIT
    return {"attention_type": "mean_shift", "attention_args": args}


def _build_configs() -> Dict[str, Dict[str, Any]]:
    table: Dict[str, Dict[str, Any]] = {
        REFERENCE_CONFIG: {"attention_type": "multi_head", "attention_args": {}}}
    for steps in (1, 3):
        for blurring in (False, True):
            for mode in ("per_head", "per_token"):
                for kernel in ("gaussian", "vmf"):
                    table[_grid_id(blurring, steps, mode, kernel)] = _grid_config(
                        blurring, steps, mode, kernel)
    return table


#: Config id -> ``{"attention_type", "attention_args"}``, cheapest first (1 step, no blurring).
#: These 17 are the grid; ``mha`` is exactly as its stored rows were run (empty attention_args).
CONFIGS: Dict[str, Dict[str, Any]] = _build_configs()

#: Control configs, outside the grid (same shape as ``CONFIGS``); ``--mode main --configs`` only.
CONTROL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mha_oinit": {
        "attention_type": "multi_head",
        # DECISION plan-2026-10-30T025213-f75ab4e9/D-011
        # mha replays one seeded initializer instance for qkv and proj; a string here builds a separate proj instance. Do NOT edit mha itself (its rows are stored).
        "attention_args": {"output_kernel_initializer": "glorot_uniform"}},
}

#: Every config id a cell may name: the grid and the controls.
ALL_CONFIGS: Dict[str, Dict[str, Any]] = {**CONFIGS, **CONTROL_CONFIGS}


def build_model(config_id: str, num_classes: int):
    """The small ViT of ``config_id`` with a logits ``Dense`` head (the same backbone for all)."""
    import keras
    from dl_techniques.layers.transformers.vision_encoder import VisionEncoder
    spec = ALL_CONFIGS[config_id]
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-006
    # VisionEncoder, not ViT: ViT hardcodes multi_head attention, so the arms could not differ; do NOT swap it in.
    encoder = VisionEncoder(
        img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
        attention_type=spec["attention_type"], attention_args=dict(spec["attention_args"]),
        use_cls_token=False, output_mode="mean", normalization_position="pre", name="encoder")
    inputs = keras.Input((32, 32, 3))
    logits = keras.layers.Dense(num_classes, name="logits")(encoder(inputs))
    return keras.Model(inputs, logits, name=f"exp2_{config_id}")


def attention_layers(model) -> List[Any]:
    """The attention layer of every transformer block of ``model``, in depth order."""
    return [block.attention for block in model.get_layer("encoder").transformer_layers]


# ---------------------------------------------------------------------
# learned bandwidths


def record_bandwidths(layers: Sequence[Any], forward: Callable[[], Any]) -> List[np.ndarray]:
    """Run ``forward()`` once and return the bandwidth each layer's own helper produced.

    ``MeanShiftAttention._bandwidth`` is wrapped on the instances for the duration of the
    call, so the values are exactly those the forward pass used (clip included), whatever
    the bandwidth mode. The wrapper is removed afterwards, also when ``forward`` raises.

    Returns:
        One array per layer: ``(1, H, 1, 1)`` for ``per_head``, ``(B, H, N, 1)`` for
        ``per_token``, ``()`` for ``global``.

    Raises:
        RuntimeError: a layer's helper was not called exactly once by ``forward``.
    """
    captured: List[List[np.ndarray]] = [[] for _ in layers]
    for layer, store in zip(layers, captured):
        original = layer._bandwidth

        def recording(tokens, acc, _original=original, _store=store):
            value = _original(tokens, acc)
            _store.append(np.asarray(value))
            return value

        layer._bandwidth = recording
    try:
        forward()
    finally:
        for layer in layers:
            del layer._bandwidth
    counts = [len(store) for store in captured]
    if counts != [1] * len(layers):
        raise RuntimeError(f"expected one _bandwidth call per layer, got {counts}")
    return [store[0] for store in captured]


def summarize_bandwidth(values: np.ndarray, layer: Any) -> Dict[str, Any]:
    """Mean, min, max, per-head mean and clip-range status of one layer's bandwidths."""
    values = np.asarray(values, dtype=np.float64)
    per_head = values.mean(axis=(0, 2, 3)) if values.ndim == 4 else values.reshape(1)
    return {
        "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
        "per_head_mean": [float(v) for v in per_head],
        "min_bandwidth": float(layer.min_bandwidth), "max_bandwidth": float(layer.max_bandwidth),
        "inside_clip_range": bool(
            values.min() > layer.min_bandwidth and values.max() < layer.max_bandwidth),
    }


def learned_bandwidths(model, probe: np.ndarray) -> Dict[str, Any]:
    """The learned bandwidths of every MeanShift layer of ``model`` on the images ``probe``."""
    from dl_techniques.layers.attention.mean_shift_attention import MeanShiftAttention
    layers = [layer for layer in attention_layers(model) if isinstance(layer, MeanShiftAttention)]
    if not layers:
        return {"status": NOT_APPLICABLE, "layers": []}
    values = record_bandwidths(layers, lambda: model(probe, training=False))
    return {
        "status": "extracted", "mode": layers[0].bandwidth_mode, "probe_images": int(len(probe)),
        "layers": [{"layer": index, **summarize_bandwidth(v, layer)}
                   for index, (v, layer) in enumerate(zip(values, layers))],
    }


def measure_throughput(model, x: np.ndarray, batch_size: int = 256) -> float:
    """Images per second of ``predict`` over ``x``, after one untimed pass that traces it."""
    model.predict(x, batch_size=batch_size, verbose=0)
    started = time.perf_counter()
    model.predict(x, batch_size=batch_size, verbose=0)
    return float(len(x) / (time.perf_counter() - started))


# ---------------------------------------------------------------------
# arguments and cells


def parse_worker_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2 worker: train and score one cell.")
    common.add_worker_arguments(parser)
    common.add_run_arguments(parser, default_dataset="cifar100")
    return parser.parse_args(argv)


def parse_driver_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2 driver: MeanShift Attention against multi_head in a small ViT. "
            "Runs one gated subprocess per cell; the cells call this module with --worker."),
        epilog="GPU 0 only, and only while idle (common.run_sweep); --cpu is for tests.")
    common.add_sweep_arguments(parser)
    common.add_run_arguments(parser, default_dataset="cifar100")
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="smoke: mha and the default config; grid: all 17 configs; "
                             "main: mha and --configs")
    parser.add_argument("--configs", nargs="*", default=[],
                        help="main mode: the config ids to compare with mha (MeanShift grid "
                             f"ids or a control id: {', '.join(CONTROL_CONFIGS)})")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds (default: 0 for smoke and grid, 0 1 2 for main)")
    args = parser.parse_args(argv)
    if args.mode == "main":
        if not args.configs:
            parser.error("--mode main needs --configs (the config ids to compare with mha)")
    elif args.configs:
        parser.error("--configs only applies to --mode main")
    unknown = [c for c in args.configs if c not in ALL_CONFIGS or c == REFERENCE_CONFIG]
    if unknown:
        parser.error(f"--configs must be config ids other than {REFERENCE_CONFIG}, got {unknown}")
    if args.seeds is None:
        args.seeds = [0, 1, 2] if args.mode == "main" else [0]
    return args


def config_ids(args: argparse.Namespace) -> List[str]:
    """The config ids of the driver mode, the reference first."""
    if args.mode == "smoke":
        return [REFERENCE_CONFIG, DEFAULT_MEAN_SHIFT_CONFIG]
    if args.mode == "grid":
        return list(CONFIGS)
    return [REFERENCE_CONFIG, *args.configs]


def build_cells(args: argparse.Namespace) -> List[common.CellSpec]:
    """One cell per (seed, config), seed-major."""
    extra = common.run_extra_args(args)
    return [common.CellSpec(EXPERIMENT, ALL_CONFIGS[config]["attention_type"], config, seed, extra)
            for seed in args.seeds for config in config_ids(args)]


# ---------------------------------------------------------------------
# worker and entry point


def run_worker(args: argparse.Namespace) -> int:
    """Train one cell and write its ``result.json``; exceptions propagate (exit non-zero)."""
    from train.common import set_seeds, setup_gpu
    spec = ALL_CONFIGS[args.config]
    if args.arm != spec["attention_type"]:
        raise ValueError(f"config {args.config!r} is a {spec['attention_type']} config, "
                         f"not arm {args.arm!r}")
    setup_gpu(None)
    set_seeds(args.seed)
    recipe = common.Recipe.for_epochs(args.epochs)
    data = common.load_cell_data(args.dataset, recipe, args.seed, args.max_samples)
    model = build_model(args.config, data.num_classes)
    logger.info(f"[exp2] {args.config} seed {args.seed}: {model.count_params():,} parameters, "
                f"{recipe.epochs} epochs")
    history = common.fit_cell(model, data, recipe)

    arrays = data.arrays
    x_select, y_select = common.selection_split(data)
    select_accuracy = common.top1(model, x_select, y_select)
    test_accuracy = common.top1(model, arrays.x_test, arrays.y_test)
    metrics = {
        "experiment": EXPERIMENT, "attention_type": args.arm, "config": args.config,
        "config_args": dict(spec["attention_args"]), "dataset": args.dataset,
        "epochs": args.epochs, "max_samples": args.max_samples, "seed": args.seed,
        "select_accuracy": select_accuracy, "test_accuracy": test_accuracy,
        "throughput_images_per_s": measure_throughput(model, arrays.x_test),
        "bandwidths": learned_bandwidths(model, x_select[:PROBE_IMAGES]),
        "n_select": int(len(x_select)), "n_test": int(len(arrays.x_test)),
        "dtype_policy": "float32",
    }
    result = common.build_result(
        history, params=model.count_params(), metrics=metrics, devices=common.physical_devices())
    common.write_result(Path(args.cell_dir), result)
    logger.info(f"[exp2] {args.config} seed {args.seed}: select {select_accuracy:.4f} "
                f"test {test_accuracy:.4f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Worker when ``--worker`` is given, otherwise the sweep driver; returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return run_worker(parse_worker_arguments(argv))
    args = parse_driver_arguments(argv)
    return common.run_driver(
        args, build_cells(args), worker_module=WORKER_MODULE,
        recipe=common.Recipe.for_epochs(args.epochs), cell_timeout_s=args.cell_timeout_s,
        resolved={"experiment": EXPERIMENT, "mode": args.mode, "dataset": args.dataset,
                  "epochs": args.epochs, "max_samples": args.max_samples,
                  "configs": config_ids(args), "seeds": list(args.seeds)})


if __name__ == "__main__":
    sys.exit(main())
