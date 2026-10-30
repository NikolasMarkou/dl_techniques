"""Public API of the time-series heads sub-package.

The heads take hidden states from a forecasting backbone and turn them into
a forecast. Nothing here knows which backbone produced them: every input width
comes from the shape handed to ``build``, except the Timestral head, which
takes ``d_model`` because it owns a residual projection of that width.

Five names are exported, one registered head class per module. A sixth class,
the private ``_LikelihoodHead``, is the shared base of the two likelihood
heads and is not registered. There is no factory or dispatcher: the
forecasting models build their heads directly, so import the class you want by
name.

Classes
-------
* ``QuantileHead`` -- fixed-horizon, non-crossing quantile forecast from a
  flattened feature vector (``quantile_head_fixed_io``).
* ``QuantileSequenceHead`` -- the same for a sequence of any length, one
  quantile vector per time step (``quantile_head_variable_io``).
* ``GaussianLikelihoodHead`` -- mean and standard deviation of a Gaussian
  (``likelihood_heads``).
* ``NegativeBinomialLikelihoodHead`` -- mean and dispersion of a negative
  binomial, for overdispersed counts (``likelihood_heads``).
* ``TimestralQuantileHead`` -- maps each decoder token to ``patch_size``
  non-crossing quantile vectors (``timestral_quantile_head``).

Example::

    >>> from dl_techniques.layers.heads.time_series import GaussianLikelihoodHead
    >>> mu, sigma = GaussianLikelihoodHead(units=1)(hidden_states)
"""

from .quantile_head_fixed_io import QuantileHead
from .quantile_head_variable_io import QuantileSequenceHead
from .likelihood_heads import (
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
)
from .timestral_quantile_head import TimestralQuantileHead

__all__ = [
    "QuantileHead",
    "QuantileSequenceHead",
    "GaussianLikelihoodHead",
    "NegativeBinomialLikelihoodHead",
    "TimestralQuantileHead",
]
