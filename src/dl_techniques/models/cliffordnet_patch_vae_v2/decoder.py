"""CliffordNetPatchDecoderV2 — hierarchical CliffordNet decoder (mirror of the encoder).

Latent ``(B, Hb, Wb, latent_dim)`` →
proj_in (1x1 Conv to stage_dims[-1]) →
for stage i from N-1 down to 0:
    stage_depths[i] × CliffordNetBlock(channels=stage_dims[i])
    if i > 0:
        UpSampling2D(2, bilinear) + Conv2D(stage_dims[i-1], 1x1) + LayerNorm
→ pre_head_norm → Conv2DTranspose(img_channels, k=patch_size, s=patch_size).

DECISION plan_2026-05-27_75849a91/D-002: CliffordNetBlock has internal
residual; the per-stage block loop does NOT add an outer skip.

DECISION plan_2026-05-27_75849a91/D-001: all-new decoder code; the
ConvNeXt v2 path re-exports the V1 decoder, which is flat
(no upsampling stages) and therefore not reusable for the hierarchical
CliffordNet variant.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import keras

from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordNetPatchDecoderV2(keras.layers.Layer):
    """Hierarchical CliffordNet decoder.

    Args:
        patch_size: Final ConvTranspose stride / kernel; pixel up-factor
            after all CliffordNet stages.
        stage_dims: Channel widths per stage in ENCODER order. The
            decoder iterates in reverse.
        stage_depths: Block counts per stage in encoder order.
        stage_shifts: Per-stage shifts in encoder order.
        cli_mode, ctx_mode, use_global_context_in_last_stage,
        layer_scale_init, block_drop_path_rate: forwarded to each block.
            ``use_global_context_in_last_stage`` enables the GAP branch
            in the DEEPEST decoder stage (== encoder's last stage), for
            symmetry.
        img_channels: Pixel output channel count.
        kernel_regularizer: Optional regularizer.

    Input shape: ``(B, Hb, Wb, latent_dim)``.
    Output shape: ``(B, Hb * 2**(N-1) * patch_size, Wb * 2**(N-1) *
        patch_size, img_channels)`` raw logits.
    """

    def __init__(
        self,
        patch_size: int,
        stage_dims: List[int],
        stage_depths: List[int],
        stage_shifts: List[List[int]],
        latent_dim: int,
        img_channels: int = 3,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context_in_last_stage: bool = True,
        layer_scale_init: float = 1e-5,
        block_drop_path_rate: float = 0.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if len(stage_dims) < 1:
            raise ValueError("stage_dims must be non-empty.")
        if len(stage_depths) != len(stage_dims) or len(stage_shifts) != len(stage_dims):
            raise ValueError(
                "stage_dims / stage_depths / stage_shifts length mismatch."
            )
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if img_channels <= 0:
            raise ValueError(
                f"img_channels must be positive, got {img_channels}"
            )

        self.patch_size = patch_size
        self.stage_dims = list(stage_dims)
        self.stage_depths = list(stage_depths)
        self.stage_shifts = [list(s) for s in stage_shifts]
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context_in_last_stage = use_global_context_in_last_stage
        self.layer_scale_init = layer_scale_init
        self.block_drop_path_rate = block_drop_path_rate
        self.kernel_regularizer = kernel_regularizer

        self._num_stages = len(self.stage_dims)
        self._last_stage = self._num_stages - 1  # encoder-order index of deepest

        # Linear drop-path across the entire decoder block stack.
        total_blocks = int(sum(self.stage_depths))
        if total_blocks <= 1 or self.block_drop_path_rate <= 0.0:
            drop_path_rates = [0.0] * total_blocks
        else:
            drop_path_rates = [
                float(self.block_drop_path_rate * i / (total_blocks - 1))
                for i in range(total_blocks)
            ]

        # --- Project latent → stage_dims[-1] at the bottleneck. ---
        self.proj_in = keras.layers.Conv2D(
            filters=self.stage_dims[-1],
            kernel_size=1,
            padding="valid",
            kernel_regularizer=kernel_regularizer,
            name="proj_in",
        )

        # --- Stages, FLAT list (decoder order = deepest first).
        # See encoder.py: list-of-lists breaks Keras save/load tracking.
        # `self._stage_starts[k]` indexes the first block of decoder
        # stage k (k=0 deepest, k=N-1 shallowest).
        self.blocks: List[CliffordNetBlock] = []
        self._stage_starts: List[int] = []
        self.up_layers: List[keras.layers.Layer] = []
        self.up_proj_layers: List[keras.layers.Layer] = []
        self.up_norm_layers: List[keras.layers.Layer] = []

        block_idx = 0
        for k in range(self._num_stages):
            self._stage_starts.append(block_idx)
            i = self._last_stage - k  # encoder-order stage index
            use_global = bool(
                self.use_global_context_in_last_stage and i == self._last_stage
            )
            for j in range(self.stage_depths[i]):
                self.blocks.append(
                    CliffordNetBlock(
                        channels=self.stage_dims[i],
                        shifts=self.stage_shifts[i],
                        cli_mode=self.cli_mode,
                        ctx_mode=self.ctx_mode,
                        use_global_context=use_global,
                        layer_scale_init=self.layer_scale_init,
                        drop_path_rate=drop_path_rates[block_idx],
                        kernel_regularizer=kernel_regularizer,
                        normalization_type="zero_centered_rms_norm",
                        name=f"dec_stage{k}_block{j}",
                    )
                )
                block_idx += 1

            # Upsampling transition between decoder stage k and k+1
            # (i.e. from encoder index i to i-1).
            if k < self._last_stage:
                next_i = i - 1
                self.up_layers.append(
                    keras.layers.UpSampling2D(
                        size=(2, 2),
                        interpolation="bilinear",
                        name=f"dec_up_{k}",
                    )
                )
                self.up_proj_layers.append(
                    keras.layers.Conv2D(
                        filters=self.stage_dims[next_i],
                        kernel_size=1,
                        padding="valid",
                        kernel_regularizer=kernel_regularizer,
                        name=f"dec_up_proj_{k}",
                    )
                )
                self.up_norm_layers.append(
                    keras.layers.LayerNormalization(
                        epsilon=1e-6, name=f"dec_up_norm_{k}"
                    )
                )

        # --- Final head: pre-head norm + ConvTranspose to pixels. ---
        self.pre_head_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="pre_head_norm"
        )
        self.head = keras.layers.Conv2DTranspose(
            filters=img_channels,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            activation=None,
            kernel_regularizer=kernel_regularizer,
            name="head",
        )

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(
                f"CliffordNetPatchDecoderV2 expects 4D input (B,Hb,Wb,latent), "
                f"got {input_shape}"
            )
        b, hb, wb, c = input_shape
        if c is not None and c != self.latent_dim:
            raise ValueError(
                f"input channel dim ({c}) does not match latent_dim "
                f"({self.latent_dim})."
            )

        # Project latent → bottleneck embed.
        self.proj_in.build(input_shape)
        cur_shape = (b, hb, wb, self.stage_dims[-1])

        for k in range(self._num_stages):
            i = self._last_stage - k
            start = self._stage_starts[k]
            end = start + self.stage_depths[i]
            for blk in self.blocks[start:end]:
                blk.build(cur_shape)
            if k < self._last_stage:
                next_i = i - 1
                up = self.up_layers[k]
                up_proj = self.up_proj_layers[k]
                up_norm = self.up_norm_layers[k]
                up.build(cur_shape)
                up_out_shape = up.compute_output_shape(cur_shape)
                up_proj.build(up_out_shape)
                next_shape = (
                    up_out_shape[0],
                    up_out_shape[1],
                    up_out_shape[2],
                    self.stage_dims[next_i],
                )
                up_norm.build(next_shape)
                cur_shape = next_shape

        # Final norm + ConvTranspose.
        self.pre_head_norm.build(cur_shape)
        self.head.build(cur_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x = self.proj_in(inputs)

        # DECISION plan_2026-05-27_75849a91/D-002: CliffordNetBlock owns
        # its residual; the per-stage block loop is bare.
        for k in range(self._num_stages):
            i = self._last_stage - k
            start = self._stage_starts[k]
            end = start + self.stage_depths[i]
            for blk in self.blocks[start:end]:
                x = blk(x, training=training)
            if k < self._last_stage:
                x = self.up_layers[k](x)
                x = self.up_proj_layers[k](x)
                x = self.up_norm_layers[k](x, training=training)

        x = self.pre_head_norm(x, training=training)
        x = self.head(x)
        return x

    # ------------------------------------------------------------------
    # compute_output_shape
    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")
        b, hb, wb, _ = input_shape
        up_factor = 2 ** (self._num_stages - 1) * self.patch_size
        h = None if hb is None else hb * up_factor
        w = None if wb is None else wb * up_factor
        return (b, h, w, self.img_channels)

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
                "latent_dim": self.latent_dim,
                "img_channels": self.img_channels,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context_in_last_stage": (
                    self.use_global_context_in_last_stage
                ),
                "layer_scale_init": self.layer_scale_init,
                "block_drop_path_rate": self.block_drop_path_rate,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any]
    ) -> "CliffordNetPatchDecoderV2":
        config = dict(config)
        reg = config.get("kernel_regularizer")
        if isinstance(reg, dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(reg)
        return cls(**config)
