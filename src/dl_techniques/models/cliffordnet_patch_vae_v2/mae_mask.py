"""MAE mask utilities — re-exported from
``dl_techniques.models.convnext_patch_vae_v2.mae_mask``.

The masking pipeline (``generate_patch_mask``, ``apply_mask_with_token``,
``upsample_mask_to_pixels``) is model-shape-agnostic — it only depends
on the post-stem patch grid, which is identical between
ConvNeXtPatchVAEV2 and CliffordNetPatchVAEV2. Re-exporting keeps the
single source of truth in v2.
"""

from dl_techniques.models.convnext_patch_vae_v2.mae_mask import (  # noqa: F401
    apply_mask_with_token,
    generate_patch_mask,
    upsample_mask_to_pixels,
)
