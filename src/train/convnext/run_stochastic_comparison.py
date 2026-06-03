"""Depth-vs-gradient stochastic_mode comparison driver for ConvNeXt.

Trains the production ConvNeXt trainer once with stochastic_mode='depth' and
once with 'gradient' under an identical seed, then emits a side-by-side
comparison (comparison.md + loss/metric curves) via train.common.compare_runs
so the better regularizer can be chosen.

Run:
    .venv/bin/python -m train.convnext.run_stochastic_comparison \
        --model v1 --dataset cifar10 --variant cifar10 --epochs 50 --gpu 1
"""

import os
import sys
import argparse
import subprocess

# The driver only ORCHESTRATES training subprocesses (each gets its own GPU via
# --gpu / a hard-set CUDA_VISIBLE_DEVICES in the child env) and runs a CPU-side
# comparison at the end. It must NOT hold a GPU context: a second TF context on
# the training GPU fragments/starves the trainer's XLA allocator and can SIGABRT
# it (observed as `Check failed: h != kInvalidChunkHandle`). Force the driver
# process CPU-only BEFORE TF is imported below; the child env re-enables the GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from train.common.compare_runs import compare_runs
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
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--strides', type=int, default=4,
                        help='Stem + inter-stage downsample stride. Use 2 for the '
                             '4-stage ImageNet variants (tiny/small/base/...) on '
                             '32x32 CIFAR; the default 4 collapses spatial dims and '
                             'crashes (only the 2-stage cifar10 variant tolerates 4).')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='Depthwise kernel size (forwarded to the trainer).')
    parser.add_argument('--modes', type=str, nargs=2, default=['depth', 'gradient'])
    return parser


def run_comparison(args: argparse.Namespace) -> None:
    os.makedirs('results', exist_ok=True)

    target_module = f'train.convnext.train_convnext_{args.model}'
    repo_root = os.getcwd()
    run_dirs = {}

    for mode in args.modes:
        before = {
            d for d in os.listdir('results')
            if os.path.isdir(os.path.join('results', d))
        }

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        env['MPLBACKEND'] = 'Agg'

        cmd = [
            sys.executable, '-m', target_module,
            '--stochastic-mode', mode,
            '--seed', str(args.seed),
            '--dataset', args.dataset,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--variant', args.variant,
            '--strides', str(args.strides),
            '--kernel-size', str(args.kernel_size),
        ]

        logger.info(f"Launching ConvNeXt {args.model} training for mode={mode}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, cwd=repo_root)

        if result.returncode != 0:
            logger.error(
                f"Training subprocess for mode={mode} exited with code {result.returncode}"
            )
            raise SystemExit(
                f"Training subprocess for mode={mode} failed with code {result.returncode}"
            )

        after = {
            d for d in os.listdir('results')
            if os.path.isdir(os.path.join('results', d))
        }
        new = after - before
        if len(new) != 1:
            raise SystemExit(
                f"Expected exactly 1 new results dir for mode={mode}, found {sorted(new)}"
            )
        run_dirs[mode] = os.path.join('results', new.pop())
        logger.info(f"Resolved results dir for mode={mode}: {run_dirs[mode]}")

    out = compare_runs(
        run_dirs[args.modes[0]],
        run_dirs[args.modes[1]],
        labels=(args.modes[0], args.modes[1]),
        output_dir=os.path.join(
            'results', f'convnext_stochastic_compare_{args.model}_{args.dataset}'
        ),
    )
    logger.info(f"Comparison written to {out}")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_comparison(args)


if __name__ == '__main__':
    main()
