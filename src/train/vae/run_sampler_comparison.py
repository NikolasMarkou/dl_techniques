"""Two-arm VAE sampler comparison driver (gaussian vs hypersphere).

Trains the production VAE trainer once per sampler mode under an identical seed
(``gaussian``, ``hypersphere``), then emits a side-by-side comparison via
``train.common.compare_runs`` — the hypersphere arm against the ``gaussian``
baseline — so the sampler's effect can be read off ``comparison.md`` +
loss/metric curves.

CPU-only driver rationale:
    This module only ORCHESTRATES training subprocesses (each child gets its own
    GPU via ``--gpu`` / a hard-set ``CUDA_VISIBLE_DEVICES`` in its env) and runs a
    CPU-side comparison (pandas / matplotlib) at the end. It must NOT hold a GPU
    context: a second TF context on the training GPU fragments/starves the
    trainer's XLA allocator and can SIGABRT it (observed elsewhere as
    ``Check failed: h != kInvalidChunkHandle``). The driver process is forced
    CPU-only BELOW (``CUDA_VISIBLE_DEVICES=''``) BEFORE any TF/keras import; each
    child subprocess re-enables the GPU via its own env.

Arms run STRICTLY SERIALLY (never in parallel) on a single GPU (default GPU1).

Run::

    .venv/bin/python -m train.vae.run_sampler_comparison --dataset mnist --gpu 1
    .venv/bin/python -m train.vae.run_sampler_comparison --smoke --gpu 1
"""

import os
import sys
import argparse
import subprocess

# Force the driver process CPU-only BEFORE TF is imported below (the driver only
# orchestrates subprocesses + a CPU-side compare_runs; a GPU context here would
# starve the child trainer's XLA allocator). Each child re-enables the GPU via
# its own env (see env['CUDA_VISIBLE_DEVICES'] below).
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from train.common.compare_runs import compare_runs
from dl_techniques.utils.logger import logger

# Fixed, ordered arm list. gaussian is the baseline the hypersphere arm is
# compared against.
ARMS = ["gaussian", "hypersphere"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serial 2-arm VAE sampler comparison "
                    "(gaussian vs hypersphere)."
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset for every arm (default mnist — smaller/faster '
                             'than cifar10 for the A/B).')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU id for the child trainers (default 1, per GPU1 pref). '
                             'The driver itself stays CPU-only.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Optional epoch override forwarded to each arm '
                             '(default None = use the trainer default).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Shared seed for every arm (reproducible comparison).')
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Passthrough --smoke to each arm (1 epoch on a small '
                             'train subset; fast end-to-end check).')
    return parser


def run_comparison(args: argparse.Namespace) -> None:
    os.makedirs('results', exist_ok=True)

    repo_root = os.getcwd()
    run_dirs = {}

    for arm in ARMS:
        before = {
            d for d in os.listdir('results')
            if os.path.isdir(os.path.join('results', d))
        }

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        env['MPLBACKEND'] = 'Agg'

        cmd = [
            sys.executable, '-m', 'train.vae.train_vae',
            '--sampler', arm,
            '--dataset', args.dataset,
            '--seed', str(args.seed),
            '--gpu', str(args.gpu),
            '--no-epoch-analyzer',
        ]
        if args.smoke:
            cmd.append('--smoke')
        if args.epochs is not None:
            cmd += ['--epochs', str(args.epochs)]

        logger.info(f"Launching VAE training arm={arm}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, cwd=repo_root)

        if result.returncode != 0:
            logger.error(f"arm {arm} exited with code {result.returncode}")
            raise SystemExit(f"arm {arm} failed with code {result.returncode}")

        after = {
            d for d in os.listdir('results')
            if os.path.isdir(os.path.join('results', d))
        }
        new = after - before
        if len(new) != 1:
            raise SystemExit(
                f"Expected exactly 1 new results dir for arm={arm}, found {sorted(new)}"
            )
        run_dirs[arm] = os.path.join('results', new.pop())
        logger.info(f"Finished arm={arm}; resolved results dir: {run_dirs[arm]}")

    base_out = os.path.join('results', f'vae_sampler_compare_{args.dataset}')

    hypersphere_out = compare_runs(
        run_dirs['gaussian'],
        run_dirs['hypersphere'],
        labels=('gaussian', 'hypersphere'),
        output_dir=os.path.join(base_out, 'gaussian_vs_hypersphere'),
    )
    logger.info(f"Comparison (gaussian vs hypersphere) written to {hypersphere_out}")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_comparison(args)


if __name__ == '__main__':
    main()
