"""ConvNeXtPatchVAE v2 — multi-task pretraining backbone.

Extends V1 with LPIPS perceptual loss, SimMIM-style MAE masked-recon,
attention-pool classification head, bilinear-upsample segmentation head,
and an `xl` preset. V1 (``dl_techniques.models.convnext_patch_vae``) is
unchanged.

Per repo convention, this ``__init__.py`` is intentionally minimal —
import the public symbols directly from their submodules:

    from dl_techniques.models.convnext_patch_vae_v2.config import (
        ConvNeXtPatchVAEV2Config,
    )
    from dl_techniques.models.convnext_patch_vae_v2.model import (
        ConvNeXtPatchVAEV2,
        create_convnext_patch_vae_v2,
    )
"""
