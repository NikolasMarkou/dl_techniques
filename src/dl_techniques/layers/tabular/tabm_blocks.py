"""TabM-style batched-ensemble building blocks: ``ScaleEnsemble``, ``LinearEfficientEnsemble``,
``NLinear``, and the ``TabMMLPBlock`` / ``TabMBackbone`` layers that assemble them into MLPs.

Every symbol named above now lives in its own sibling module; this file only
re-exports them so the tree stays importable while the split is in progress.
Import the leaf module, not this one.
"""

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tabular._ensemble_scaling import EnsembleInitDistribution
from dl_techniques.layers.tabular.linear_efficient_ensemble import LinearEfficientEnsemble
from dl_techniques.layers.tabular.nlinear import NLinear
from dl_techniques.layers.tabular.scale_ensemble import ScaleEnsemble
from dl_techniques.layers.tabular.tabm_backbone import TabMBackbone
from dl_techniques.layers.tabular.tabm_mlp_block import TabMMLPBlock

# ---------------------------------------------------------------------
