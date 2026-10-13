"""Neural HMM trainer: sequences from a known HMM, scored against the truth.

Run it as a module::

    python -m train.neural_hmm.train_neural_hmm --help

``train_neural_hmm`` holds the CLI and the run; ``data`` the ground-truth HMM, the
sampler and the numpy references. This file imports neither: importing the script here
would make ``python -m`` warn that the module was already imported.
"""
