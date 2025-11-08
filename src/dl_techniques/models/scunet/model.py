import keras
import numpy as np
from keras import ops
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.swin_conv_block import SwinConvBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SCUNet(keras.Model):
    """Swin-Conv-UNet for image restoration tasks.

    A U-Net architecture that combines Swin Transformer blocks with convolutional
    operations for effective image restoration.

    Args:
        in_nc: Number of input channels. Defaults to 3.
        config: Configuration list specifying number of blocks per stage.
               Defaults to [4, 4, 4, 4, 4, 4, 4].
        dim: Base dimension for feature channels. Defaults to 64.
        head_dim: Dimension of each attention head. Defaults to 32.
        window_size: Size of attention window. Defaults to 8.
        drop_path_rate: Maximum stochastic depth rate. Defaults to 0.0.
        input_resolution: Expected input resolution for optimization. Defaults to 256.
        **kwargs: Additional keyword arguments for Model base class.
    """

    def __init__(
            self,
            in_nc: int = 3,
            config: List[int] = None,
            dim: int = 64,
            head_dim: int = 32,
            window_size: int = 8,
            drop_path_rate: float = 0.0,
            input_resolution: int = 256,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if config is None:
            config = [4, 4, 4, 4, 4, 4, 4]

        self.in_nc = in_nc
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate
        self.input_resolution = input_resolution

        logger.info(f"Initializing SCUNet with config: {config}, dim: {dim}, "
                    f"window_size: {window_size}, input_resolution: {input_resolution}")

        # Calculate drop path rates for each layer
        dpr = [float(x) for x in np.linspace(0, drop_path_rate, sum(config))]

        # Build network components
        self._build_network(dpr)

    def _build_network(self, dpr: List[float]) -> None:
        """Build the network architecture.

        Args:
            dpr: List of drop path rates for each layer.
        """
        # Head
        self.m_head = keras.layers.Conv2D(
            self.dim,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="head"
        )

        # Encoder blocks
        begin = 0

        # Stage 1 (down1)
        self.m_down1 = self._create_stage_blocks(
            num_blocks=self.config[0],
            conv_dim=self.dim // 2,
            trans_dim=self.dim // 2,
            dpr=dpr[begin:begin + self.config[0]],
            input_res=self.input_resolution,
            stage_name="down1"
        )
        self.m_down1.append(
            keras.layers.Conv2D(
                2 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="down1_downsample"
            )
        )

        begin += self.config[0]

        # Stage 2 (down2)
        self.m_down2 = self._create_stage_blocks(
            num_blocks=self.config[1],
            conv_dim=self.dim,
            trans_dim=self.dim,
            dpr=dpr[begin:begin + self.config[1]],
            input_res=self.input_resolution // 2,
            stage_name="down2"
        )
        self.m_down2.append(
            keras.layers.Conv2D(
                4 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="down2_downsample"
            )
        )

        begin += self.config[1]

        # Stage 3 (down3)
        self.m_down3 = self._create_stage_blocks(
            num_blocks=self.config[2],
            conv_dim=2 * self.dim,
            trans_dim=2 * self.dim,
            dpr=dpr[begin:begin + self.config[2]],
            input_res=self.input_resolution // 4,
            stage_name="down3"
        )
        self.m_down3.append(
            keras.layers.Conv2D(
                8 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="down3_downsample"
            )
        )

        begin += self.config[2]

        # Bottleneck
        self.m_body = self._create_stage_blocks(
            num_blocks=self.config[3],
            conv_dim=4 * self.dim,
            trans_dim=4 * self.dim,
            dpr=dpr[begin:begin + self.config[3]],
            input_res=self.input_resolution // 8,
            stage_name="body"
        )

        begin += self.config[3]

        # Decoder blocks
        # Stage 4 (up3)
        self.m_up3 = [
            keras.layers.Conv2DTranspose(
                4 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="up3_upsample"
            )
        ]
        self.m_up3.extend(
            self._create_stage_blocks(
                num_blocks=self.config[4],
                conv_dim=2 * self.dim,
                trans_dim=2 * self.dim,
                dpr=dpr[begin:begin + self.config[4]],
                input_res=self.input_resolution // 4,
                stage_name="up3"
            )
        )

        begin += self.config[4]

        # Stage 5 (up2)
        self.m_up2 = [
            keras.layers.Conv2DTranspose(
                2 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="up2_upsample"
            )
        ]
        self.m_up2.extend(
            self._create_stage_blocks(
                num_blocks=self.config[5],
                conv_dim=self.dim,
                trans_dim=self.dim,
                dpr=dpr[begin:begin + self.config[5]],
                input_res=self.input_resolution // 2,
                stage_name="up2"
            )
        )

        begin += self.config[5]

        # Stage 6 (up1)
        self.m_up1 = [
            keras.layers.Conv2DTranspose(
                self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="up1_upsample"
            )
        ]
        self.m_up1.extend(
            self._create_stage_blocks(
                num_blocks=self.config[6],
                conv_dim=self.dim // 2,
                trans_dim=self.dim // 2,
                dpr=dpr[begin:begin + self.config[6]],
                input_res=self.input_resolution,
                stage_name="up1"
            )
        )

        # Tail
        self.m_tail = keras.layers.Conv2D(
            self.in_nc,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="tail"
        )

        # Convert lists to Sequential models
        self.m_down1 = keras.Sequential(self.m_down1, name="down1")
        self.m_down2 = keras.Sequential(self.m_down2, name="down2")
        self.m_down3 = keras.Sequential(self.m_down3, name="down3")
        self.m_body = keras.Sequential(self.m_body, name="body")
        self.m_up3 = keras.Sequential(self.m_up3, name="up3")
        self.m_up2 = keras.Sequential(self.m_up2, name="up2")
        self.m_up1 = keras.Sequential(self.m_up1, name="up1")

    def _create_stage_blocks(
            self,
            num_blocks: int,
            conv_dim: int,
            trans_dim: int,
            dpr: List[float],
            input_res: int,
            stage_name: str
    ) -> List[keras.layers.Layer]:
        """Create blocks for a stage.

        Args:
            num_blocks: Number of blocks in the stage.
            conv_dim: Convolutional dimension.
            trans_dim: Transformer dimension.
            dpr: Drop path rates for blocks in this stage.
            input_res: Input resolution for this stage.
            stage_name: Name of the stage for block naming.

        Returns:
            List of layer blocks for the stage.
        """
        blocks = []
        for i in range(num_blocks):
            block_type = "W" if i % 2 == 0 else "SW"
            blocks.append(
                SwinConvBlock(
                    conv_dim=conv_dim,
                    trans_dim=trans_dim,
                    head_dim=self.head_dim,
                    window_size=self.window_size,
                    drop_path_rate=dpr[i],
                    block_type=block_type,
                    input_resolution=input_res,
                    name=f"{stage_name}_block_{i}"
                )
            )
        return blocks

    def call(self,
             x: keras.KerasTensor,
             training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the SCUNet model.

        Args:
            x: Input tensor of shape (B, H, W, C).
            training: Boolean indicating whether the model should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        h, w = ops.shape(x)[1], ops.shape(x)[2]

        # Padding to ensure divisibility by 64 (2^6 for 6 downsampling steps)
        padding_bottom = int(np.ceil(h / 64) * 64 - h)
        padding_right = int(np.ceil(w / 64) * 64 - w)

        if padding_bottom > 0 or padding_right > 0:
            paddings = [[0, 0], [0, padding_bottom], [0, padding_right], [0, 0]]
            x = ops.pad(x, paddings, mode="REFLECT")

        # Encoder path with skip connections
        x1 = self.m_head(x)
        x2 = self.m_down1(x1, training=training)
        x3 = self.m_down2(x2, training=training)
        x4 = self.m_down3(x3, training=training)

        # Bottleneck
        x = self.m_body(x4, training=training)

        # Decoder path with skip connections
        x = self.m_up3(x + x4, training=training)
        x = self.m_up2(x + x3, training=training)
        x = self.m_up1(x + x2, training=training)
        x = self.m_tail(x + x1)

        # Remove padding
        if padding_bottom > 0 or padding_right > 0:
            x = x[:, :h, :w, :]

        return x

    def get_config(self) -> Dict[str, Any]:
        """Returns the model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "in_nc": self.in_nc,
            "config": self.config,
            "dim": self.dim,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "drop_path_rate": self.drop_path_rate,
            "input_resolution": self.input_resolution,
        })
        return config

# ---------------------------------------------------------------------
