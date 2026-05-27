"""Training callbacks for CliffordNetPatchVAEV2 — re-exported from v2.

Both v2 callbacks only depend on attributes that exist on
:class:`CliffordNetPatchVAEV2` with the same name and meaning:

- ``BetaAnnealingCallback`` mutates ``self.model._beta_kl`` (our model
  has this; see ``model.py`` cached scalar weights).
- ``MaskedReconViz`` reads ``self.model.config.mae_mask_ratio`` and calls
  ``self.model(samples, training=False)['reconstruction']`` (both
  contracts are preserved).

Re-exporting from v2 keeps a single source of truth.
"""

from train.convnext_patch_vae_v2.callbacks import (  # noqa: F401
    BetaAnnealingCallback,
    MaskedReconViz,
)
