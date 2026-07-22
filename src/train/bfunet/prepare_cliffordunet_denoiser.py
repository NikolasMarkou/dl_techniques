"""Non-training script: construct and save an initialized CliffordUNet denoiser
matching ``results/20260720_convunext_denoiser_hfb3``'s shared/architecture config.

This script does NOT call ``train.bfunet.common.train()`` or ``model.fit()``. It
builds an untrained (freshly-initialized) bias-free, degree-1-homogeneous Clifford
U-Net denoiser (``train_cliffordunet_denoiser.build_model``) using the same
shared/architecture ``TrainingConfig`` values as the hfb3 ConvNeXt run
(``results/20260720_convunext_denoiser_hfb3/config.json``), runs the reused
``verify_bias_free`` homogeneity check, and persists ``config.json`` plus the
untrained model to a new ``results/20260722_cliffordunet_denoiser_hfb3/``
directory -- for downstream use (e.g. architecture comparison, capacity
inspection) without spending any training compute.

Part of plan-2026-07-22-2c1fb044 Part B (see decisions.md D-002 for why this is a
standalone hardcoded-config script rather than a ``--prepare-only`` flag on the
existing trainer).
"""

import dataclasses
import json

from dl_techniques.utils.logger import logger
from train.bfunet.common import save_config_json
from train.bfunet.train_cliffordunet_denoiser import (
    TrainingConfig,
    build_model,
    verify_bias_free,
)

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

# Deliberately dated/tagged to mirror the hfb3 ConvNeXt run this script matches.
RESULTS_DIR = "results/20260722_cliffordunet_denoiser_hfb3"
# Deliberately NOT "best_model.keras"/"final_model.keras" -- this model was never
# trained, so those names (which imply a completed/selected training run) would be
# misleading provenance.
MODEL_FILENAME = "initial_model.keras"


# ---------------------------------------------------------------------
# CONFIG CONSTRUCTION
# ---------------------------------------------------------------------


def build_config() -> TrainingConfig:
    """Construct the ``TrainingConfig`` matching hfb3's shared/architecture fields.

    ``shifts``/``cli_mode``/``ctx_mode``/``layer_scale_init`` are deliberately left
    at ``TrainingConfig``'s own dataclass defaults (not passed here) -- D-002 /
    user's explicit choice. ``block_activation``/``block_activation_alpha``/
    ``block_normalization`` are not passed either: Clifford's block internals are
    pinned inside ``create_cliffordunet_denoiser`` and are not accepted the same
    way ConvNeXt's factory accepts them.
    """
    return TrainingConfig(
        variant="base",
        patch_size=256,
        channels=3,
        initial_filters=66,
        filter_multiplier=1.0,
        depth=2,
        blocks_per_level=3,
        final_projection_groups=1,
        use_gabor_stem=True,
        gabor_filters=22,
        gabor_kernel_size=11,
        gabor_activation=None,
        gabor_stem_projection=True,
        use_laplacian_pyramid=True,
        zero_pad_channels=True,
        high_freq_blocks=3,
        downsample_pool_type="max",
        enable_deep_supervision=False,
        expose_bottleneck=False,
        experiment_name="20260722_cliffordunet_denoiser_hfb3",
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    config = build_config()
    logger.info(
        f"Constructed CliffordUNet prepare-only config: {config.experiment_name}"
    )
    print(json.dumps(dataclasses.asdict(config), indent=2, default=str))

    model = build_model(config)
    logger.info(
        f"Built CliffordUNet denoiser model '{model.name}' "
        f"({model.count_params()} params)."
    )
    verify_bias_free(model)
    logger.info(
        "prepare_cliffordunet_denoiser: model build + verify_bias_free completed "
        "successfully (no training, no save)."
    )

    # TODO(step 10): save_config_json(config, RESULTS_DIR, "config.json") and
    #   model.save(os.path.join(RESULTS_DIR, MODEL_FILENAME)). No common.train()/
    #   .fit() call -- build + save only.


if __name__ == "__main__":
    main()
