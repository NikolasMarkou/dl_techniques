"""Depth-vs-gradient stochastic_mode comparison driver for ConvNeXt.

Trains the production ConvNeXt trainer once with stochastic_mode='depth' and
once with 'gradient' under an identical seed, then emits a side-by-side
comparison (comparison.md + loss/metric curves) via train.common.compare_runs
so the better regularizer can be chosen.

Each mode gets an explicit ``--experiment-name`` (``<stem>_<mode>``, one shared
timestamp per invocation), so the driver reads ``<output-dir>/<name>/results_summary.json``
directly instead of guessing which directory a run created. A name that already holds a
run is refused by the trainer, so a rerun never overwrites an earlier comparison.

Run:
    .venv/bin/python -m train.convnext.run_stochastic_comparison \
        --model v1 --dataset cifar10 --variant cifar10 --epochs 50 --gpu 0
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# The driver only ORCHESTRATES training subprocesses (each gets its own GPU via
# --gpu / a hard-set CUDA_VISIBLE_DEVICES in the child env) and runs a CPU-side
# comparison at the end. It must NOT hold a GPU context: a second TF context on
# the training GPU fragments/starves the trainer's XLA allocator and can SIGABRT
# it (observed as `Check failed: h != kInvalidChunkHandle`). Force the driver
# process CPU-only BEFORE TF is imported below; the child env re-enables the GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from train.common.args import resolved_run_dir
from train.common.compare_runs import compare_runs
from train.common.run_io import default_experiment_name
from dl_techniques.utils.logger import logger


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serial depth-vs-gradient stochastic_mode comparison for ConvNeXt."
    )
    parser.add_argument('--model', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--variant', type=str, default='cifar10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index given to each training subprocess as '
                             'CUDA_VISIBLE_DEVICES (the driver itself stays CPU-only); '
                             'a negative value hides every GPU (CPU training).')
    parser.add_argument('--strides', type=int, default=None,
                        help='Stem + inter-stage downsample stride; forwarded only when '
                             'given, so the trainer default (2: 16,8 feature maps for the '
                             'cifar10 variant and 16,8,4,2 for the 4-stage variants on '
                             '32x32) is the one source. 4 builds fine but is spatially '
                             'degenerate on 32x32 (8,2 or 8,2,1,1).')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='Depthwise kernel size (forwarded to the trainer).')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output root of the two runs and the comparison; a relative '
                             'path is anchored at the repo root by the trainer.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap the train pool and test set (smoke runs of the driver); '
                             'forwarded only when given.')
    parser.add_argument('--modes', type=str, nargs=2, default=['depth', 'gradient'])
    return parser


def run_summary_path(output_dir: str, experiment_name: str) -> Path:
    """Where the trainer writes ``results_summary.json`` for an explicit run name.

    Uses the trainer's own ``resolved_run_dir`` (relative roots anchor at the repo
    root), so driver and trainer cannot disagree about the location.
    """
    run_dir = resolved_run_dir(
        argparse.Namespace(output_dir=output_dir, experiment_name=experiment_name))
    return Path(run_dir) / 'results_summary.json'


def run_comparison(args: argparse.Namespace) -> None:
    target_module = f'train.convnext.train_convnext_{args.model}'
    repo_root = os.getcwd()
    stem = default_experiment_name(
        'convnext', args.model, 'stochastic', args.dataset, args.variant)
    run_dirs = {}

    for mode in args.modes:
        experiment_name = f'{stem}_{mode}'
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu >= 0 else ''
        env['MPLBACKEND'] = 'Agg'

        cmd = [
            sys.executable, '-m', target_module,
            '--stochastic-mode', mode,
            '--seed', str(args.seed),
            '--dataset', args.dataset,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--variant', args.variant,
            '--kernel-size', str(args.kernel_size),
            '--output-dir', args.output_dir,
            '--experiment-name', experiment_name,
        ]
        if args.strides is not None:
            cmd += ['--strides', str(args.strides)]
        if args.max_samples is not None:
            cmd += ['--max-samples', str(args.max_samples)]

        logger.info(f"Launching ConvNeXt {args.model} training for mode={mode}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, cwd=repo_root)

        if result.returncode != 0:
            logger.error(
                f"Training subprocess for mode={mode} exited with code {result.returncode}"
            )
            raise SystemExit(
                f"Training subprocess for mode={mode} failed with code {result.returncode}"
            )

        summary_path = run_summary_path(args.output_dir, experiment_name)
        if not summary_path.is_file():
            raise SystemExit(f"Missing {summary_path} after the mode={mode} run")
        summary = json.loads(summary_path.read_text())
        if summary.get('status') != 'ok':
            raise SystemExit(f"mode={mode} summary status is {summary.get('status')!r}")
        run_dirs[mode] = summary_path.parent
        logger.info(
            f"mode={mode}: {summary_path.parent} test accuracy (best weights) "
            f"{summary['test_metrics_best'].get('accuracy')}"
        )

    comparison_dir = resolved_run_dir(argparse.Namespace(
        output_dir=args.output_dir, experiment_name=f'{stem}_compare'))
    out = compare_runs(
        str(run_dirs[args.modes[0]]),
        str(run_dirs[args.modes[1]]),
        labels=(args.modes[0], args.modes[1]),
        output_dir=str(comparison_dir),
    )
    logger.info(f"Comparison written to {out}")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_comparison(args)


if __name__ == '__main__':
    main()
