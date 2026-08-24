"""TabM (tabular multi-head ensemble) public API.

Re-exports the model class and its factory functions. Internal callers may still
import from ``dl_techniques.models.tabm.model`` directly.

Deliberately NOT re-exported: ``TabMLoss``. It is a loss, not a model, and it
lives in ``dl_techniques/losses/tabm_loss.py``; passing it through here gave one
class two import paths and left ``models/tabm`` advertising a surface it does not
own. Take it from its canonical home::

    from dl_techniques.losses.tabm_loss import TabMLoss

Do not re-add a pass-through here. See ``dl_techniques/models/resnet/__init__.py``
and ``dl_techniques/models/vit/__init__.py`` for the same removal.
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
