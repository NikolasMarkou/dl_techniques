"""CliffordNetPatchVAE v2 — hierarchical Clifford-block patch-level VAE.

Sibling of ``dl_techniques.models.convnext_patch_vae_v2``. The ConvNeXt
v2 backbone is replaced by a hierarchical stack of
:class:`CliffordNetBlock`s interleaved with stride-2
:class:`CliffordNetBlockDSv2` transitions. All other v2 features (MAE
mask, SIGReg, KL, LPIPS, classification head, segmentation head, custom
``train_step``) are preserved.

Per repo convention, this ``__init__.py`` is intentionally minimal —
import the public symbols directly from their submodules:

    from dl_techniques.models.cliffordnet_patch_vae_v2.config import (
        CliffordNetPatchVAEV2Config,
        PRESETS,
    )
    from dl_techniques.models.cliffordnet_patch_vae_v2.model import (
        CliffordNetPatchVAEV2,
        create_cliffordnet_patch_vae_v2,
    )
"""
