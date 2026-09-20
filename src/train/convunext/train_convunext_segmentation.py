"""Train ConvUNext as a 3-class semantic segmenter on Oxford-IIIT Pet.

A thin wrapper: the CLI, data, model, callbacks and artifacts live in
``train.convunext.common``. The bias-free ConvUNeXt denoiser is a different trainer
(``train.bfunet.train_convunext_denoiser``).

Run (repo root, the TFDS cache is read with ``download=False``)::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.convunext.train_convunext_segmentation \
        --variant tiny --image-size 128 --epochs 2 --batch-size 16 --max-samples 1600

Each run writes ``results/<experiment_name>/`` (``config.json``, ``run.log``,
``training_log.csv``, ``training_history.json``, ``best_model.keras``,
``final_model.keras``, ``results_summary.json``, ``visualizations/``, ``model_analysis/``); a
reused ``--experiment-name`` is refused. ``--help`` allocates no GPU and no run directory.
The layout and every summary key are documented in ``src/train/convunext/README.md``.
"""

from typing import Optional, Sequence

from train.convunext.common import main as run_convunext_segmentation


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse ``argv`` (``None`` reads ``sys.argv[1:]``) and train the segmenter."""
    run_convunext_segmentation(argv)


if __name__ == "__main__":
    main()
