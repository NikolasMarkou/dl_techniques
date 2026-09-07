"""Random-sign initializer for ensemble scaling vectors.

Provides :class:`RandomSigns`, which fills a weight of any shape with values
drawn uniformly from ``{-1, +1}``. Use it where every entry must be a distinct,
unit-magnitude perturbation rather than a small random one. The TabM
batched-ensemble layers (``ScaleEnsemble``, ``LinearEfficientEnsemble``) use it
for their per-member scaling vectors under ``init_distribution='random-signs'``.
"""

import keras
from typing import Optional, Any, Dict, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# DECISION plan-2026-09-07T095804-b821967f/D-003: no `legacy_packages=` here.
# This class moved out of `layers/tabular/tabm_blocks.py`, so its registration
# key changed outright. Do NOT add
# `legacy_packages=("dl_techniques.layers.tabular.tabm_blocks",)` to "keep old
# checkpoints loading" -- a repo-wide scan found ZERO tabm `.keras` archives, so
# the alias would name a population of nothing and would have to be retired later.
# See decisions.md D-003.
@register_dl_technique("dl_techniques.initializers.random_signs")
class RandomSigns(keras.initializers.Initializer):
    """Draw each element uniformly from :math:`\\{-1, +1\\}`.

    **Construction:**

    .. code-block:: text

        requested shape
              │
              ▼
        ┌───────────────────────┐
        │ RandomUniform(-1, +1) │   (a stock Keras initializer)
        └───────────┬───────────┘
                    │  u
                    ▼
             ┌─────────────┐
             │ u >= 0 ? +1 │
             │        : -1 │
             └──────┬──────┘
                    ▼
             +/- 1, requested shape

    This is the initializer the TabM paper uses for the per-member scaling
    vectors. It is the only one of the three ``init_distribution`` options that
    guarantees every ensemble member starts at a distinct, non-degenerate,
    unit-magnitude perturbation of the shared kernel: a normal draw clusters
    members near the mean, and a constant draw makes them identical.

    The draw goes through a stock ``RandomUniform`` rather than a direct
    ``keras.random.uniform`` call because this initializer runs inside
    ``add_weight``, where there is no seed-generator variable for
    ``keras.random.*`` to update.

    :param seed: Optional seed for reproducible draws.
    :type seed: int or None

    :ivar seed: The seed as passed by the caller.
    :vartype seed: int or None

    Example:
        >>> init = RandomSigns(seed=0)
        >>> w = init((4, 8))
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Store the seed and build the backing uniform initializer.

        :param seed: Optional seed for reproducible draws.
        :type seed: int or None
        """
        self.seed = seed
        # Draws through a stock initializer: inside `add_weight` there is no
        # seed-generator variable for a direct `keras.random.*` call to update.
        self._uniform = keras.initializers.RandomUniform(
            minval=-1.0, maxval=1.0, seed=seed
        )

    def __call__(
            self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None
    ) -> Any:
        """Draw a tensor of independent random signs.

        :param shape: Requested shape. Any rank is accepted.
        :type shape: tuple of int
        :param dtype: Data type of the result. ``None`` means
            ``keras.config.floatx()``.
        :type dtype: str or None
        :return: A tensor of the requested shape holding only ``-1`` and ``+1``.
        :rtype: tensor
        """
        dtype = dtype or keras.config.floatx()
        u = self._uniform(shape, dtype=dtype)
        return keras.ops.where(
            u >= 0.0, keras.ops.ones_like(u), -keras.ops.ones_like(u)
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``seed``.
        :rtype: dict
        """
        return {"seed": self.seed}

# ---------------------------------------------------------------------
