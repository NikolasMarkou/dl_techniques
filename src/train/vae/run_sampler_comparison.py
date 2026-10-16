"""Two-arm VAE sampler comparison driver (gaussian vs hypersphere).

Trains the production VAE trainer once per sampler mode under an identical seed
(``gaussian``, ``hypersphere``), then emits a side-by-side comparison via
``train.common.compare_runs`` — the hypersphere arm against the ``gaussian``
baseline — so the sampler's effect can be read off ``comparison.md`` +
loss/metric curves.

CPU-only driver rationale:
    This module only ORCHESTRATES training subprocesses (each child gets its
    GPU through ``CUDA_VISIBLE_DEVICES`` in its env, resolved from ``--gpu``) and runs a
    CPU-side comparison (pandas / matplotlib) at the end. It must NOT hold a GPU
    context: a second TF context on the training GPU fragments/starves the
    trainer's XLA allocator and can SIGABRT it (observed elsewhere as
    ``Check failed: h != kInvalidChunkHandle``). ``main`` forces the driver
    process CPU-only (``hide_gpus``) before TF lists any device; importing the
    module changes nothing. Each child subprocess re-enables the GPU via its
    own env.

GPU selection (D-110/D-116): ``--gpu N`` counts within the
``CUDA_VISIBLE_DEVICES`` the driver inherited, or names a physical GPU when
none was exported. Without ``--gpu``, an exported value is used as is, and
physical GPU 1 is the default otherwise. The child is not given ``--gpu``: its
list already holds the one selected device.

Arms run STRICTLY SERIALLY (never in parallel) on a single GPU (default GPU1).

Run::

    .venv/bin/python -m train.vae.run_sampler_comparison --dataset mnist --gpu 1
    .venv/bin/python -m train.vae.run_sampler_comparison --smoke --gpu 1
"""

import os
import sys
import argparse
import subprocess
from typing import Optional

from train.common.gpu import child_visible_devices, env_with_visible_devices, hide_gpus
from train.common.compare_runs import compare_runs
from dl_techniques.utils.logger import logger

# Physical GPU used when neither --gpu nor CUDA_VISIBLE_DEVICES is given.
DEFAULT_GPU = 1

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
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index for the child trainers; counts within an '
                             'exported CUDA_VISIBLE_DEVICES, else names a physical '
                             'GPU. Default: the exported list as is, or physical '
                             'GPU 1 when none is exported. The driver itself stays '
                             'CPU-only.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Optional epoch override forwarded to each arm '
                             '(default None = use the trainer default).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Shared seed for every arm (reproducible comparison).')
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Passthrough --smoke to each arm (1 epoch on a small '
                             'train subset; fast end-to-end check).')
    return parser


def run_comparison(
        args: argparse.Namespace,
        inherited_visible_devices: Optional[str],
) -> None:
    """Train both arms as children, then compare them.

    :param args: Parsed driver arguments.
    :param inherited_visible_devices: The ``CUDA_VISIBLE_DEVICES`` the driver
        was started with (``None`` if unset); resolves the children's device.
    """
    os.makedirs('results', exist_ok=True)

    repo_root = os.getcwd()
    run_dirs = {}
    # Resolved once, before the first arm (an index outside the inherited list
    # is refused here).
    # DECISION plan-2026-10-14T062616-81938e81/D-116
    # The child gets the resolved device and NO --gpu of its own; do NOT pass
    # --gpu again: the child's setup_gpu would index its one-entry list and raise.
    visible_devices = child_visible_devices(
        args.gpu, inherited_visible_devices, DEFAULT_GPU)
    logger.info(f"children get CUDA_VISIBLE_DEVICES={visible_devices!r}")

    for arm in ARMS:
        before = {
            d for d in os.listdir('results')
            if os.path.isdir(os.path.join('results', d))
        }

        env = env_with_visible_devices(visible_devices)
        env['MPLBACKEND'] = 'Agg'

        cmd = [
            sys.executable, '-m', 'train.vae.train_vae',
            '--sampler', arm,
            '--dataset', args.dataset,
            '--seed', str(args.seed),
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
    # CPU-only BEFORE TF lists a device (the driver only orchestrates
    # subprocesses + a CPU-side compare_runs; a GPU context here would starve
    # the child trainer's XLA allocator). Here, not at import (D-118): an
    # import-time pin hid every GPU from any importer for good.
    inherited_visible_devices = hide_gpus()
    parser = build_argument_parser()
    args = parser.parse_args()
    run_comparison(args, inherited_visible_devices)


if __name__ == '__main__':
    main()
