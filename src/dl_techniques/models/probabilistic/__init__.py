"""Probabilistic models — latent-variable models with exact, tractable inference.

- `neural_hmm/` — Neural Hidden Markov Model: exact forward-algorithm likelihood,
  Viterbi decoding and state posteriors over neural transitions and emissions;
  generative or CRF

Import from the leaf package, not from here — this family package carries no re-exports
by design (the reasoning is written out in `models/vision/__init__.py`):

    from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM
"""
