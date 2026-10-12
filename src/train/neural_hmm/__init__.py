"""Neural HMM Trainer.

Public API mirrors the ``train/`` convention: the trainer script and data module
are re-exported. Internal callers may still import from the submodules directly.

Examples
--------
Run a training job::

    MPLBACKEND=Agg python -m train.neural_hmm.train_neural_hmm \\
        --dataset gaussian_hmm --n_samples 500 --n_states 4 --epochs 50 \\
        --output-dir /tmp/nhmm_run

References
----------
- Tang et al., 2020. Neural HMMs for regime-switching time-series. arXiv:2006.00694.
- Rabiner, 1989. A Tutorial on Hidden Markov Models. Proceedings of the IEEE.
"""

from train.neural_hmm.train_neural_hmm import train, create_argument_parser, main
from train.neural_hmm.data import (
    available_generators,
    generate_dataset,
    set_global_seed,
)

__all__ = [
    "train",
    "create_argument_parser",
    "main",
    "available_generators",
    "generate_dataset",
    "set_global_seed",
]