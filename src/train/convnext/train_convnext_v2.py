"""Train ConvNeXt V2 on MNIST / CIFAR-10 / CIFAR-100.

A thin wrapper: everything (CLI, data, schedule, callbacks, artifacts) lives in
``train.convnext.common``, shared with the ConvNeXt V1 trainer. This file only
fixes the model family. ``--variant`` offers exactly the keys of
``ConvNeXtV2.MODEL_VARIANTS``.

Run (repo root)::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v2 \
        --dataset cifar10 --variant cifar10 --epochs 5

Each run writes ``results/<experiment_name>/`` (``config.json``, ``training_log.csv``,
``training_history.json``, ``best_model.keras``, ``final_model.keras``,
``results_summary.json``, ``run.log``, ``visualizations/``, ``model_analysis/``); a reused
``--experiment-name`` is refused. ``--help`` allocates no GPU and no run directory.
"""

from typing import Optional, Sequence

from train.convnext.common import main as run_convnext


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse ``argv`` (``None`` reads ``sys.argv[1:]``) and train ConvNeXt V2."""
    run_convnext("v2", argv)


if __name__ == "__main__":
    main()
