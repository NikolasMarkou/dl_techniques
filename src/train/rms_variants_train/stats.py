"""Statistics surface for the rms_variants_train sweep.

This module used to carry its own copy of the four helpers below. That copy was
character-identical to ``train.logic.multiseed_stats`` -- its docstring said as
much ("behavioural contract is byte-equivalent") -- so the repo maintained the
same four functions twice.

Both copies now live in :mod:`train.common.stats`. This module remains as the
package's named statistics surface so existing imports
(``from train.rms_variants_train.stats import ...``) keep working, and re-exports
them unchanged.
"""

from train.common.stats import (
    ArrayLike,
    bootstrap_ci,
    format_mean_std,
    mean_std,
    paired_permutation_test,
)

__all__ = [
    "mean_std",
    "bootstrap_ci",
    "paired_permutation_test",
    "format_mean_std",
    "ArrayLike",
]
