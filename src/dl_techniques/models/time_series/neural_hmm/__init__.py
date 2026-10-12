"""Neural Hidden Markov Model — configurable, factory-driven.

Public API mirrors the ``models/time_series/mdn`` template: the model class and
factory functions are re-exported. Internal callers may still import from the
submodules (``.model`` / ``.encoders`` / ``.transitions`` / ``.emissions`` /
``.inference`` / ``.loss``) directly.

The package registers itself as
``dl_techniques.models.neural_hmm.model`` via
``@register_dl_technique`` (Keras 3 serialisation, no ``custom_objects``
needed on load).

Examples
--------
Build from a variant preset:

    >>> from dl_techniques.models.time_series.neural_hmm import from_variant
    >>> model = from_variant("neural-emission-hmm", n_states=8, obs_dim=16)

Build from a full config dict:

    >>> from dl_techniques.models.time_series.neural_hmm.model import NeuralHMM
    >>> model = NeuralHMM({
...     "mode": "generative",
...     "encoder": "lstm",
...     "emission_context": "none",
...     "transition": "static",
...     "topology": "ergodic",
...     "obs_type": "gaussian",
...     "hidden": 128,
...     "state_emb_dim": 32,
...     "n_states": 8,
...     "obs_dim": 16,
... })

References
----------
- Rabiner, 1989. A Tutorial on Hidden Markov Models. Proceedings of the IEEE.
- Tang et al., 2020. Neural HMMs for regime-switching. arXiv:2006.00694.
"""

from .model import NeuralHMM, from_variant, MODEL_VARIANTS

__all__ = [
    "NeuralHMM",
    "from_variant",
    "MODEL_VARIANTS",
]