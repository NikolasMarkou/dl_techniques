"""CliffordNetPatchEncoderV2 — hierarchical CliffordNet encoder.

Stem (Conv k=patch_size, s=patch_size) → LayerNorm →
[optional MAE mask at patch grid] →
for each stage i in 0..N-1:
    stage_depths[i] × CliffordNetBlock(channels=stage_dims[i], shifts=stage_shifts[i])
    if i < N-1:
        CliffordNetBlockDSv2(channels=stage_dims[i], out_channels=stage_dims[i+1], strides=2)
→ mu_head (1x1 Conv, normal init), log_var_head (1x1 Conv, zero init).

DECISION plan_2026-05-27_75849a91/D-002: CliffordNetBlock owns its own
residual via GatedGeometricResidual — the block-stack loop must NOT add
an outer ``residual + ...`` skip (would double-apply the residual).

DECISION plan_2026-05-27_75849a91/D-001: this file is a SIBLING of
``convnext_patch_vae_v2/encoder.py``; v2 stays untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import ops

from dl_techniques.layers.geometric.clifford_block import (
    CliffordNetBlock,
    CliffordNetBlockDSv2,
)

from .mae_mask import apply_mask_with_token, generate_patch_mask


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordNetPatchEncoderV2(keras.layers.Layer):
    """Hierarchical CliffordNet patch encoder.

    Args:
        patch_size: Stem stride / kernel.
        stage_dims: Channel width per stage.
        stage_depths: ``CliffordNetBlock`` count per stage.
        stage_shifts: Per-stage shift schedules handed to each block.
        downsample_kind: Pool kind used by transition blocks.
        downsample_kernel_size: DW conv kernel inside transition blocks.
        cli_mode: Algebraic components for every block.
        ctx_mode: Context-stream mode for every block.
        use_global_context_in_last_stage: When True, last-stage blocks
            enable the GAP-based global branch.
        layer_scale_init: Initial GGR LayerScale gamma.
        block_drop_path_rate: Max DropPath rate (linear schedule across
            entire encoder block stack — transitions excluded).
        latent_dim: Per-cell latent channel width.
        mae_mask_ratio: SimMIM mask ratio applied to post-stem feature map.
        mae_mask_seed: Optional reproducibility seed.

    Input shape:
        ``(B, H, W, C)`` with ``H % patch_size == 0`` and divisibility by
        ``2**(num_stages-1)`` after the stem.

    Output shape:
        ``output_pre_bottleneck=False``: ``(mu, log_var)`` each
            ``(B, Hb, Wb, latent_dim)`` where ``Hb = H/patch_size /
            2**(num_stages-1)``.
        ``output_pre_bottleneck=True``: ``(mu, log_var, pre_bottleneck,
            mask)`` — ``pre_bottleneck`` is ``(B, Hb, Wb, stage_dims[-1])``,
            ``mask`` is ``(B, Hp, Wp, 1)`` at the PATCH-grid resolution
            (NOT the bottleneck) or ``None``.
    """

    def __init__(
        self,
        patch_size: int,
        stage_dims: List[int],
        stage_depths: List[int],
        stage_shifts: List[List[int]],
        downsample_kind: str = "blur",
        downsample_kernel_size: int = 7,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context_in_last_stage: bool = True,
        layer_scale_init: float = 1e-5,
        block_drop_path_rate: float = 0.0,
        latent_dim: int = 16,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        mae_mask_ratio: float = 0.0,
        mae_mask_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if len(stage_dims) < 1:
            raise ValueError("stage_dims must be non-empty.")
        if len(stage_depths) != len(stage_dims):
            raise ValueError(
                "len(stage_depths) must equal len(stage_dims)."
            )
        if len(stage_shifts) != len(stage_dims):
            raise ValueError(
                "len(stage_shifts) must equal len(stage_dims)."
            )
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if not (0.0 <= mae_mask_ratio <= 0.95):
            raise ValueError(
                f"mae_mask_ratio must be in [0.0, 0.95], got {mae_mask_ratio}"
            )

        self.patch_size = patch_size
        self.stage_dims = list(stage_dims)
        self.stage_depths = list(stage_depths)
        self.stage_shifts = [list(s) for s in stage_shifts]
        self.downsample_kind = downsample_kind
        self.downsample_kernel_size = downsample_kernel_size
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context_in_last_stage = use_global_context_in_last_stage
        self.layer_scale_init = layer_scale_init
        self.block_drop_path_rate = block_drop_path_rate
        self.latent_dim = latent_dim
        self.kernel_regularizer = kernel_regularizer
        self.mae_mask_ratio = mae_mask_ratio
        self.mae_mask_seed = mae_mask_seed

        self._num_stages = len(self.stage_dims)
        self._last_stage = self._num_stages - 1

        # --- Linear drop-path schedule over the ENTIRE block stack
        #     (transitions excluded; they do their own residual handling).
        total_blocks = int(sum(self.stage_depths))
        if total_blocks <= 1 or self.block_drop_path_rate <= 0.0:
            drop_path_rates = [0.0] * total_blocks
        else:
            drop_path_rates = [
                float(self.block_drop_path_rate * i / (total_blocks - 1))
                for i in range(total_blocks)
            ]

        # --- Stem ---
        self.stem = keras.layers.Conv2D(
            filters=self.stage_dims[0],
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_regularizer=kernel_regularizer,
            name="stem",
        )
        self.stem_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="stem_norm"
        )

        # --- Stages: list-of-lists for blocks; parallel list for transitions.
        # `self.stage_blocks[i][j]` is the j-th CliffordNetBlock at stage i.
        # `self.transitions[i]` is the CliffordNetBlockDSv2 BETWEEN stage i
        # and stage i+1; defined for i in [0, last_stage).
        self.stage_blocks: List[List[CliffordNetBlock]] = []
        self.transitions: List[CliffordNetBlockDSv2] = []
        block_idx = 0
        for i in range(self._num_stages):
            blocks_i: List[CliffordNetBlock] = []
            use_global = bool(
                self.use_global_context_in_last_stage and i == self._last_stage
            )
            for j in range(self.stage_depths[i]):
                blocks_i.append(
                    CliffordNetBlock(
                        channels=self.stage_dims[i],
                        shifts=self.stage_shifts[i],
                        cli_mode=self.cli_mode,
                        ctx_mode=self.ctx_mode,
                        use_global_context=use_global,
                        layer_scale_init=self.layer_scale_init,
                        drop_path_rate=drop_path_rates[block_idx],
                        kernel_regularizer=kernel_regularizer,
                        # zero_centered_rms_norm is the CliffordNetBlock
                        # default; keep it explicit so config round-trips.
                        normalization_type="zero_centered_rms_norm",
                        name=f"stage{i}_block{j}",
                    )
                )
                block_idx += 1
            self.stage_blocks.append(blocks_i)

            if i < self._last_stage:
                self.transitions.append(
                    CliffordNetBlockDSv2(
                        channels=self.stage_dims[i],
                        out_channels=self.stage_dims[i + 1],
                        shifts=self.stage_shifts[i],
                        cli_mode=self.cli_mode,
                        # DSv2 supports {"diff","abs","pyramid_diff"};
                        # plain ctx_mode passes through unchanged.
                        ctx_mode=self.ctx_mode,
                        kernel_size=self.downsample_kernel_size,
                        strides=2,
                        stream_pool=self.downsample_kind,
                        skip_pool=self.downsample_kind,
                        layer_scale_init=self.layer_scale_init,
                        # DropPath off in transitions — they are not part of
                        # the regular block-depth schedule and downsamplers
                        # are typically left un-stochasticized.
                        drop_path_rate=0.0,
                        kernel_regularizer=kernel_regularizer,
                        name=f"transition_{i}_to_{i + 1}",
                    )
                )

        # --- mu / log_var 1x1 heads (consume bottleneck features) ---
        # log_var zero-init (V1 D-003 convention preserved).
        self.mu_head = keras.layers.Conv2D(
            filters=latent_dim,
            kernel_size=1,
            padding="valid",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            kernel_regularizer=kernel_regularizer,
            name="mu_head",
        )
        self.log_var_head = keras.layers.Conv2D(
            filters=latent_dim,
            kernel_size=1,
            padding="valid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=kernel_regularizer,
            name="log_var_head",
        )

        # Lazily-built — only when mae_mask_ratio > 0.
        self.mask_token: Optional[keras.Variable] = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"CliffordNetPatchEncoderV2 expects 4D input (B,H,W,C), got "
                f"{input_shape}"
            )
        b, h, w, _ = input_shape
        ds_factor = 2 ** (self._num_stages - 1)
        if h is not None:
            if h % self.patch_size != 0:
                raise ValueError(
                    f"input H ({h}) must be divisible by patch_size "
                    f"({self.patch_size})."
                )
            hp = h // self.patch_size
            if hp % ds_factor != 0:
                raise ValueError(
                    f"patch-grid H ({hp}) must be divisible by "
                    f"2**(num_stages-1)={ds_factor}."
                )
        if w is not None:
            if w % self.patch_size != 0:
                raise ValueError(
                    f"input W ({w}) must be divisible by patch_size "
                    f"({self.patch_size})."
                )

        self.stem.build(input_shape)
        hp = None if h is None else h // self.patch_size
        wp = None if w is None else w // self.patch_size
        post_stem_shape = (b, hp, wp, self.stage_dims[0])
        self.stem_norm.build(post_stem_shape)

        # Stage builds + transitions.
        cur_shape = post_stem_shape
        for i in range(self._num_stages):
            for blk in self.stage_blocks[i]:
                blk.build(cur_shape)
            if i < self._last_stage:
                trans = self.transitions[i]
                trans.build(cur_shape)
                cur_shape = trans.compute_output_shape(cur_shape)

        # Heads consume the bottleneck feature map.
        self.mu_head.build(cur_shape)
        self.log_var_head.build(cur_shape)

        if self.mae_mask_ratio > 0.0:
            self.mask_token = self.add_weight(
                shape=(1, 1, 1, self.stage_dims[0]),
                initializer="zeros",
                trainable=True,
                name="mask_token",
            )

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        output_pre_bottleneck: bool = False,
    ):
        x = self.stem(inputs)
        x = self.stem_norm(x, training=training)

        # MAE mask is applied at the PATCH grid (post-stem, pre-stages).
        # The mask is returned at this resolution so the recon loss can
        # upsample it to pixel space with `upsample_mask_to_pixels`.
        mask: Optional[keras.KerasTensor] = None
        if (
            self.mae_mask_ratio > 0.0
            and training is True
            and self.mask_token is not None
        ):
            shape = ops.shape(x)
            batch_size = shape[0]
            hp = shape[1]
            wp = shape[2]
            mask = generate_patch_mask(
                batch_size=batch_size,
                hp=hp,
                wp=wp,
                ratio=self.mae_mask_ratio,
                seed=self.mae_mask_seed,
            )
            x = apply_mask_with_token(x, mask, self.mask_token)

        # DECISION plan_2026-05-27_75849a91/D-002: CliffordNetBlock owns its
        # residual; we DO NOT add an outer skip here. Compare with
        # convnext_patch_vae_v2/encoder.py:239-243 which does add one for
        # ConvNextV2Block.
        for i in range(self._num_stages):
            for blk in self.stage_blocks[i]:
                x = blk(x, training=training)
            if i < self._last_stage:
                x = self.transitions[i](x, training=training)

        pre_bottleneck = x  # (B, Hb, Wb, stage_dims[-1])

        mu = self.mu_head(x)
        log_var = self.log_var_head(x)

        if output_pre_bottleneck:
            return mu, log_var, pre_bottleneck, mask
        return mu, log_var

    # ------------------------------------------------------------------
    # compute_output_shape
    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")
        b, h, w, _ = input_shape
        hp = None if h is None else h // self.patch_size
        wp = None if w is None else w // self.patch_size
        ds = 2 ** (self._num_stages - 1)
        hb = None if hp is None else hp // ds
        wb = None if wp is None else wp // ds
        latent_shape = (b, hb, wb, self.latent_dim)
        return latent_shape, latent_shape

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "stage_dims": list(self.stage_dims),
                "stage_depths": list(self.stage_depths),
                "stage_shifts": [list(s) for s in self.stage_shifts],
                "downsample_kind": self.downsample_kind,
                "downsample_kernel_size": self.downsample_kernel_size,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context_in_last_stage": (
                    self.use_global_context_in_last_stage
                ),
                "layer_scale_init": self.layer_scale_init,
                "block_drop_path_rate": self.block_drop_path_rate,
                "latent_dim": self.latent_dim,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "mae_mask_ratio": self.mae_mask_ratio,
                "mae_mask_seed": self.mae_mask_seed,
            }
        )
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any]
    ) -> "CliffordNetPatchEncoderV2":
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)
