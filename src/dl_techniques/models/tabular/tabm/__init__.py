"""TabM (tabular multi-head ensemble) public API.

Re-exports the model class and its factory functions. Internal callers may still
import from ``dl_techniques.models.tabular.tabm.model`` directly.

``TabMLoss`` is not re-exported here. It is a loss, not a model, and lives in
``dl_techniques/losses/tabm_loss.py``::

    from dl_techniques.losses.tabm_loss import TabMLoss
"""

from .model import (
    TabMModel,
    create_tabm_model,
    create_tabm_plain,
    create_tabm_ensemble,
    create_tabm_mini,
    create_tabm_for_dataset,
    ensemble_predict,
)

__all__ = [
    'TabMModel',
    'create_tabm_model',
    'create_tabm_plain',
    'create_tabm_ensemble',
    'create_tabm_mini',
    'create_tabm_for_dataset',
    'ensemble_predict',
]
