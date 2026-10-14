"""Neural Hidden Markov Model: exact HMM inference over neural transitions and emissions.

The public API is the model class and its factory. The layers are importable from their
modules (``.encoders``, ``.transitions``, ``.emissions``) and the inference functions
from ``.inference``.

Every class registers under ``dl_techniques.models.neural_hmm.<module>``, so a ``.keras``
archive loads without ``custom_objects``.

Example::

    from dl_techniques.models.probabilistic.neural_hmm import create_neural_hmm

    model = create_neural_hmm("neural-hmm", n_states=4, obs_type="gaussian", obs_dim=2)
    model.compile(optimizer="adam")
    model.fit({"obs": obs, "mask": mask}, None, epochs=10)
    path, score = model.viterbi({"obs": obs, "mask": mask})

References:
    - Tran, Bisk, Vaswani, Marcu and Knight, 2016. Unsupervised Neural Hidden Markov
      Models. arXiv:1609.09007.
    - Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected Applications in
      Speech Recognition. Proceedings of the IEEE 77(2), pp. 257-286.
"""

from .model import NeuralHMM, create_neural_hmm

__all__ = [
    "NeuralHMM",
    "create_neural_hmm",
]
