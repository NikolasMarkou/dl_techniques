"""Spectrum-to-Signal Principle (SSP): diversity first, then amplify what is correct.

SSP is a two-phase contract for a supervised-then-reinforcement pipeline, introduced
as the organising principle of VibeThinker-1.5B (Xu et al., 2025, Weibo):

* the **Spectrum phase** scores a candidate by how BROAD its solution repertoire is
  -- ``Pass@K``, coverage rather than first-guess accuracy -- selects the
  diversity-maximising candidate per subdomain, and consolidates those specialists
  into one model by weighted parameter fusion;
* the **Signal phase** weights each group of rollouts by how UNCERTAIN the policy
  was on it, and applies that weight to the group-relative advantage.

The claim tying them together is that a broad-spectrum checkpoint is a better
precondition for the second phase than a sharp one, because the second phase can only
amplify a signal that already exists somewhere in the candidate space.

What is deliberately general here
---------------------------------
Neither half is bound to language models, to GRPO, or to reinforcement learning at
all. The Spectrum half takes any ``(n_problems, n_samples)`` outcome matrix -- unit
tests, rubric scores, a verifier's pass/fail -- and any set of fusable checkpoints.
The Signal half produces a per-group weight vector that multiplies *whatever*
advantage you already have: a group-relative policy advantage, an RLOO baseline, or
a per-example curriculum weight in ordinary supervised training. That second use --
:func:`spectrum_sampling_weights`, which reuses :func:`max_entropy_weight` verbatim
so the two criteria cannot drift apart -- is what makes the principle usable with no
RL machinery in the picture.

Two corrections to the source, both documented at their call sites
------------------------------------------------------------------
* The report's printed "max-entropy deviation" is not a KL divergence; its first
  denominator should be ``p0``, not ``1 - p_c``. :func:`max_entropy_deviation`
  implements the Bernoulli KL, which is zero exactly at ``p_c = p_0`` and is what the
  report's own prose describes.
* The report also credits MGPO with incentivising low-probability correct traces.
  No formula for that appears anywhere in it, so it is **not** implemented here.

Scope
-----
Nothing in this package has been trained, benchmarked or A/B-tested in this
repository. Every function is arithmetic whose behaviour is pinned by tests; whether
the principle improves a downstream metric is an empirical question this repository
has not answered, and no number here should be read as evidence either way. There is
no SSP trainer: the Signal phase needs a rollout sampler and a verifier, and this
repository has neither.

Modules
-------
- :mod:`~dl_techniques.optimization.ssp.spectrum` -- pool profiling, per-subdomain
  specialist selection, coverage-to-sampler weights
- :mod:`~dl_techniques.optimization.ssp.fusion` -- weight schemes, linear and
  task-arithmetic fusion, greedy soup
- :mod:`~dl_techniques.optimization.ssp.signal` -- binary entropy, the max-entropy
  deviation and weight, group-relative advantages, the clipped surrogate, and
  :class:`MGPOObjective` as a Keras loss
- :mod:`~dl_techniques.optimization.ssp.config` -- :class:`SSPStrategyConfig`,
  :class:`SSPStrategy` and the :func:`ssp_builder` factory

``pass@k`` itself lives in :mod:`dl_techniques.metrics.pass_at_k` and is imported
here rather than duplicated: it is a metric, and one implementation of it is the point.

Example:
    >>> import numpy as np
    >>> from dl_techniques.optimization.ssp import ssp_builder
    >>> ssp = ssp_builder({"type": "ssp_v1", "config": {"enable": True,
    ...                                                "pass_at_k": 8}})
    >>> # A 3-problem x 8-sample outcome matrix, scored for coverage.
    >>> outcomes = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    ...                      [1, 1, 1, 1, 0, 0, 0, 0],
    ...                      [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    >>> profile = ssp.profile(outcomes)
    >>> round(profile["pass_at_1"], 4)     # mean single-sample success rate
    0.125
    >>> profile["spectrum_gain"] if "spectrum_gain" in profile else "n/a"
    'n/a'
"""

from .config import SSPStrategy, SSPStrategyConfig, SSPType, ssp_builder
from .fusion import (
    FUSION_MODES,
    FUSION_WEIGHT_SCHEMES,
    fusion_weights_from_scores,
    fuse_specialists,
    greedy_soup,
    score_softmax_weights,
    uniform_weights,
)
from .signal import (
    MGPOObjective,
    TOKEN_REDUCTIONS,
    ZERO_VARIANCE_POLICIES,
    binary_entropy,
    group_relative_advantages,
    group_success_rate,
    max_entropy_deviation,
    max_entropy_weight,
    mgpo_advantages,
    mgpo_surrogate,
    pack_mgpo_targets,
    unpack_mgpo_targets,
)
from .spectrum import (
    SAMPLING_MODES,
    select_specialists,
    spectrum_profile,
    spectrum_sampling_weights,
)

__all__ = [
    # factory + config
    "ssp_builder",
    "SSPStrategy",
    "SSPStrategyConfig",
    "SSPType",
    # Spectrum phase
    "spectrum_profile",
    "select_specialists",
    "spectrum_sampling_weights",
    "SAMPLING_MODES",
    # fusion
    "fuse_specialists",
    "fusion_weights_from_scores",
    "uniform_weights",
    "score_softmax_weights",
    "greedy_soup",
    "FUSION_MODES",
    "FUSION_WEIGHT_SCHEMES",
    # Signal phase
    "binary_entropy",
    "max_entropy_deviation",
    "max_entropy_weight",
    "group_success_rate",
    "group_relative_advantages",
    "mgpo_advantages",
    "mgpo_surrogate",
    "pack_mgpo_targets",
    "unpack_mgpo_targets",
    "MGPOObjective",
    "TOKEN_REDUCTIONS",
    "ZERO_VARIANCE_POLICIES",
]
