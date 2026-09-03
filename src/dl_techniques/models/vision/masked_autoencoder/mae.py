"""``MaskedAutoencoder``, which turns any convolutional feature extractor into a self-supervised reconstruction model.

Unlike the original MAE, masking happens in pixel space, not by dropping
tokens: `PatchMasking` substitutes each masked patch's value into the full
image before the encoder ever sees it. This keeps the wrapper
encoder-agnostic (any model mapping an image to a feature map fits) but
gives up MAE's training-speed advantage, since the encoder still processes
every pixel. The reconstruction loss is computed only on masked patches,
so copying visible pixels earns nothing.

The encoder must return a 4-D `(B, H', W', C)` feature map, and its total
downsampling factor must equal `2 ** len(decoder_dims)` since `ConvDecoder`
upsamples 2x per entry — the constructor raises `ValueError` on a
mismatch. Training uses a hand-written `train_step`/`test_step` with
`tf.GradientTape` rather than stock `fit()`, so compiled losses, compiled
metrics, and `sample_weight` are bypassed, and the path is
TensorFlow-specific.

References:
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377)
    - Xie et al., 2021. SimMIM: A Simple Framework for Masked Image Modeling
      (masking in the input rather than by token dropping).
      (https://arxiv.org/abs/2111.09886)
    - Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
      Masked Autoencoders. (https://arxiv.org/abs/2301.00808)
    - Bao et al., 2021. BEiT: BERT Pre-Training of Image Transformers.
      (https://arxiv.org/abs/2106.08254)
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .conv_decoder import ConvDecoder
from .patch_masking import PatchMasking
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.masked_autoencoder.mae")
class MaskedAutoencoder(keras.Model):
    """Mask random patches of an image and train to reconstruct them.

    Architecture:

    .. code-block:: text

        image  [B, H, W, C]
           |
           v
        PatchMasking  (pixel-space; full-size output)
           |  masked_images [B, H, W, C]
           v
        encoder  (caller-supplied, any 4-D feature extractor)
           |  [B, H', W', C']
           v
        ConvDecoder  (2x upsample per decoder_dims entry)
           |
           v
        reconstruction  [B, H, W, C]

    Loss is MSE between target and reconstruction, computed on masked
    patches only, optionally after per-patch normalization
    (``norm_pix_loss``).

    :param encoder: A `keras.Model` feature extractor; must return a 4-D
        `(B, H', W', C)` feature map.
    :param patch_size: Size of the square patches masking operates on.
    :param mask_ratio: Fraction of patches to mask, in [0, 1].
    :param decoder_dims: Decoder channel widths, one per upsample stage.
        Auto-derived from the encoder's channel count when `None`.
    :param decoder_depth: Number of decoder stages when `decoder_dims` is `None`.
    :param norm_pix_loss: Normalize each target patch before the MSE.
    :param mask_value: `"learnable"`, `"zero"`, `"noise"`, or a constant float.
    :param input_shape: Input image shape `(H, W, C)`.
    :param non_mask_value: Loss weight floor applied to unmasked pixels.
    """

    def __init__(
        self,
        encoder: keras.Model,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dims: Optional[List[int]] = None,
        decoder_depth: int = 4,
        norm_pix_loss: bool = False,
        mask_value: Union[str, float] = "learnable",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        non_mask_value: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(encoder, keras.Model):
            raise TypeError("encoder must be a keras.Model instance.")

        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.decoder_dims = decoder_dims
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.mask_value = mask_value
        # Normalize to a tuple: on .keras deserialization input_shape comes back
        # as a list, and `(None,) + input_shape` would raise (tuple + list).
        self.input_shape_config = tuple(input_shape)
        self.non_mask_value = non_mask_value

        if not self.encoder.built:
            self.encoder.build((None,) + self.input_shape_config)

        encoder_output_shape = self.encoder.compute_output_shape(
            (None,) + self.input_shape_config)

        # A deep-supervision encoder reports a list of shapes; take the first as the main feature map.
        if isinstance(encoder_output_shape, list):
            main_shape = encoder_output_shape[0]
        else:
            main_shape = encoder_output_shape

        if len(main_shape) != 4:
            raise ValueError(
                f"Encoder main output must be 4D tensor (B, H, W, C). "
                f"Got: {main_shape} (Full output: {encoder_output_shape})"
            )

        self.encoder_channels = main_shape[-1]

        self._resolve_decoder_dims()
        self._validate_scale_contract(main_shape)

        self.masking = PatchMasking(
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            name="patch_masking"
        )

        self.decoder = self._create_decoder()

        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    def _resolve_decoder_dims(self) -> None:
        """Auto-configure `decoder_dims` when it was left unspecified.

        Split out of `_create_decoder` so the effective list — and therefore the
        decoder's `2 ** len(decoder_dims)` upsampling factor — exists before the
        scale contract is checked and before any sub-layer is constructed.
        """
        if self.decoder_dims is None:
            # Gradually reduce from the encoder's channel count.
            decoder_dims = []
            current_dim = self.encoder_channels
            for _ in range(self.decoder_depth):
                current_dim = max(current_dim // 2, 64)
                decoder_dims.append(current_dim)
            self.decoder_dims = decoder_dims

    def _validate_scale_contract(
        self,
        main_shape: Tuple[Optional[int], ...]
    ) -> None:
        """Check the encoder's downsampling against the decoder's upsampling.

        `ConvDecoder` upsamples exactly 2x per entry in `decoder_dims`, so the
        encoder must downsample by exactly `2 ** len(decoder_dims)` for the
        reconstruction to come back out at the input resolution. A mismatch is
        not caught anywhere downstream: `call()` succeeds, and only
        `compute_loss` fails, as a broadcast error between two spatial shapes
        that names neither the encoder nor the decoder.
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-035: compare resolved spatial sizes, not a downsampling ratio.
        # A ratio comparison would accept a 33x33->8x8 encoder against a decoder that emits 128x128. See decisions.md.
        upsample_factor = 2 ** len(self.decoder_dims)
        input_h, input_w = self.input_shape_config[0], self.input_shape_config[1]
        encoder_h, encoder_w = main_shape[1], main_shape[2]

        if encoder_h is None or encoder_w is None:
            # A dynamic encoder feature map cannot be checked statically; the
            # 4-D check above is all this constructor can promise.
            return

        decoded_h = encoder_h * upsample_factor
        decoded_w = encoder_w * upsample_factor
        if decoded_h == input_h and decoded_w == input_w:
            return

        ratio = input_h // encoder_h if encoder_h else 0
        is_power_of_two = ratio > 0 and ratio & (ratio - 1) == 0
        suggestion = (
            f"decoder_depth={int(np.log2(ratio))}"
            if is_power_of_two and encoder_h * ratio == input_h
            else "an encoder whose downsampling factor is a power of two"
        )

        raise ValueError(
            f"Encoder/decoder scale mismatch. The encoder maps "
            f"{input_h}x{input_w} to a {encoder_h}x{encoder_w} feature map, "
            f"but the decoder upsamples by {upsample_factor}x "
            f"({len(self.decoder_dims)} decoder_dims entries, 2x each), so the "
            f"reconstruction would be {decoded_h}x{decoded_w} against a "
            f"{input_h}x{input_w} target. Use {suggestion} (or pass a "
            f"decoder_dims list of that length), or change the encoder's "
            f"strides so it downsamples by exactly {upsample_factor}x."
        )

    def _create_decoder(self) -> ConvDecoder:
        """Create the decoder over the already-resolved `decoder_dims`."""
        return ConvDecoder(
            decoder_dims=self.decoder_dims,
            output_channels=self.input_shape_config[-1],
            name="conv_decoder"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Materialize every sub-layer from the input shape alone.

        :param input_shape: Shape tuple `(batch, height, width, channels)`.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-048: build must materialize masking and decoder itself.
        # The constructor's encoder.build() only reads the encoder's output shape; it reaches neither PatchMasking nor ConvDecoder. See decisions.md.
        if self.built:
            return

        resolved = tuple(input_shape)
        if len(resolved) == 4 and any(d is None for d in resolved[1:]):
            resolved = (resolved[0],) + self.input_shape_config

        self.masking.build(resolved)

        if not self.encoder.built:
            self.encoder.build(resolved)

        encoder_output_shape = self.encoder.compute_output_shape(resolved)
        if isinstance(encoder_output_shape, list):
            encoder_output_shape = encoder_output_shape[0]

        self.decoder.build(encoder_output_shape)

        super().build(input_shape)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute output shapes for all model outputs.

        :param input_shape: Shape tuple `(batch, height, width, channels)`.
        :return: Dict of output name to shape tuple.
        """
        batch_size = input_shape[0]
        height, width, channels = input_shape[1:]

        if height is None: height = self.input_shape_config[0]
        if width is None: width = self.input_shape_config[1]

        num_patches = (height // self.patch_size) * (width // self.patch_size)

        return {
            "reconstruction": (batch_size, height, width, channels),
            "mask": (batch_size, num_patches),
            "masked_input": (batch_size, height, width, channels),
            "encoded": self.encoder.compute_output_shape(input_shape)
        }

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Run mask, encode, decode and return all intermediate outputs.

        :param inputs: Input images, shape `(B, H, W, C)`.
        :param training: Passed to masking, encoder and decoder.
        :return: Dict with `reconstruction`, `mask`, `masked_input`, `encoded`.
        """
        masked_images, mask, _ = self.masking(inputs, training=training)

        # PatchMasking returns float32; cast to the compute dtype so it matches the encoder under mixed precision.
        policy = keras.mixed_precision.dtype_policy()
        if getattr(policy, "name", "") == "mixed_float16":
             masked_images = keras.ops.cast(masked_images, "float16")
        elif self.compute_dtype:
             masked_images = keras.ops.cast(masked_images, self.compute_dtype)

        encoded = self.encoder(masked_images, training=training)
        reconstruction = self.decoder(encoded, training=training)

        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "masked_input": masked_images,
            "encoded": encoded
        }

    def compute_loss(
        self,
        x: keras.KerasTensor,
        y: Optional[keras.KerasTensor] = None,
        y_pred: Optional[Dict[str, keras.KerasTensor]] = None,
        sample_weight: Optional[keras.KerasTensor] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Compute reconstruction loss only on masked patches.

        :param x: Target images, shape `(B, H, W, C)`.
        :param y: Unused; kept for the Keras `compute_loss` signature.
        :param y_pred: Dict output of `call`, with `reconstruction` and `mask`.
        :param sample_weight: Unused.
        :return: Scalar mean loss, or 0.0 when `y_pred` is `None`.
        """
        if y_pred is None:
            return keras.ops.convert_to_tensor(0.0)

        # float32 avoids mixed-precision mismatches between a float16 output and a float32 input.
        target = keras.ops.cast(x, "float32")
        reconstruction = keras.ops.cast(y_pred["reconstruction"], "float32")
        mask = keras.ops.cast(y_pred["mask"], "float32")

        if self.norm_pix_loss:
            target_patches = self._extract_patches_for_loss(target)
            mean = keras.ops.mean(target_patches, axis=-1, keepdims=True)
            var = keras.ops.var(target_patches, axis=-1, keepdims=True)
            target_normalized = (target_patches - mean) / keras.ops.sqrt(var + 1e-6)
            target = self._reconstruct_patches_for_loss(target_normalized)

        # MSE Loss
        loss = keras.ops.square(target - reconstruction)
        loss = keras.ops.mean(loss, axis=-1)  # [batch, H, W]

        # Reshape mask to match spatial dimensions
        mask_img = self._reshape_mask_for_loss(mask, target)
        mask_img = keras.ops.maximum(mask_img, self.non_mask_value)

        # Apply mask: Loss = 0 for unmasked pixels
        loss = loss * mask_img

        # Normalize by number of masked elements
        num_masked = keras.ops.sum(mask, axis=-1) + 1e-6  # [batch]

        # Sum over spatial dims, then divide by num_masked patches * patch_pixels
        # Note: mask_img is 1s and 0s.
        loss_sum = keras.ops.sum(loss, axis=[1, 2]) # [batch]

        # Adjust denominator: num_masked is patches, we need pixels
        pixels_per_patch = self.patch_size * self.patch_size
        loss = loss_sum / (num_masked * pixels_per_patch)

        return keras.ops.mean(loss) # Global mean

    def _extract_patches_for_loss(self, images: keras.KerasTensor) -> keras.KerasTensor:
        """Helper to extract patches for pixel normalization."""
        # Implementation assumes fixed patch size logic
        B = keras.ops.shape(images)[0]
        H, W, C = self.input_shape_config
        P = self.patch_size

        # [B, H//P, P, W//P, P, C]
        patches = keras.ops.reshape(images, (B, H // P, P, W // P, P, C))
        # [B, H//P, W//P, P, P, C] -> [B, N_patches, P*P*C]
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))
        return keras.ops.reshape(patches, (B, -1, P * P * C))

    def _reconstruct_patches_for_loss(self, patches: keras.KerasTensor) -> keras.KerasTensor:
        """Helper to reverse patch extraction."""
        B = keras.ops.shape(patches)[0]
        H, W, C = self.input_shape_config
        P = self.patch_size

        # [B, H//P, W//P, P, P, C]
        patches = keras.ops.reshape(patches, (B, H//P, W//P, P, P, C))
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))
        return keras.ops.reshape(patches, (B, H, W, C))

    def _reshape_mask_for_loss(self, mask: keras.KerasTensor, target: keras.KerasTensor) -> keras.KerasTensor:
        """Expand a `(B, num_patches)` mask to pixel resolution `(B, H, W)`."""
        B = keras.ops.shape(mask)[0]
        H, W = keras.ops.shape(target)[1], keras.ops.shape(target)[2]
        P = self.patch_size

        mask_grid = keras.ops.reshape(mask, (B, H // P, W // P))

        # Nearest-neighbor upsample from patch grid to pixel resolution.
        mask_img = keras.ops.repeat(mask_grid, P, axis=1)
        mask_img = keras.ops.repeat(mask_img, P, axis=2)

        return mask_img

    def train_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, float]:
        """Run one training step with a hand-written gradient tape.

        :param data: A batch, or a tuple whose first element is the batch.
        :return: Dict with `loss` and `reconstruction_loss`, both epoch means.
        """
        # TensorFlow-specific: this train_step is not backend-agnostic.
        import tensorflow as tf

        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=None, y_pred=y_pred)
            # DECISION plan-2026-08-19T163559-499b6f0e/D-036: scale_loss must stay inside the tape; do not simplify to tape.gradient(loss, ...).
            # Under mixed_float16 the LossScaleOptimizer divides every gradient by dynamic_scale unconditionally, so skipping this divides the whole update. MEASURED: float32 |dW|=2.507e+02 vs mixed_float16 2.850e-02 without it. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(loss)

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.reconstruction_loss_tracker.update_state(loss)
        # DECISION plan-2026-08-19T163559-499b6f0e/D-133: both keys must report the same epoch-mean tracker value.
        # Reporting a raw last-batch loss under "loss" would disagree with the epoch-mean "reconstruction_loss" for no visible reason. See decisions.md.
        return {
            "loss": self.reconstruction_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    def test_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, float]:
        """Run one validation step, masking the input as in training.

        :param data: A batch, or a tuple whose first element is the batch.
        :return: Dict with `loss` and `reconstruction_loss`, both epoch means.
        """
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # training=True so masking is applied; PatchMasking returns an all-zero mask otherwise.
        y_pred = self(x, training=True)
        loss = self.compute_loss(x=x, y=None, y_pred=y_pred)

        self.reconstruction_loss_tracker.update_state(loss)
        # DECISION plan-2026-08-19T163559-499b6f0e/D-133: both keys must report the same epoch-mean tracker value. See decisions.md.
        return {
            "loss": self.reconstruction_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return [self.reconstruction_loss_tracker]

    def visualize(self, image: np.ndarray, return_arrays: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one image through mask, encode, decode for inspection.

        :param image: A single image `(H, W, C)` or a batch of one `(1, H, W, C)`.
        :param return_arrays: Convert outputs to numpy arrays and drop the batch axis.
        :return: `(image, masked, reconstructed)`.
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        outputs = self(image, training=True)
        masked = outputs["masked_input"]
        reconstructed = outputs["reconstruction"]

        if return_arrays:
            image = keras.ops.convert_to_numpy(image[0])
            masked = keras.ops.convert_to_numpy(masked[0])
            reconstructed = keras.ops.convert_to_numpy(reconstructed[0])

        return image, masked, reconstructed

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "decoder_dims": self.decoder_dims,
            "decoder_depth": self.decoder_depth,
            "norm_pix_loss": self.norm_pix_loss,
            "mask_value": self.mask_value,
            "input_shape": self.input_shape_config,
            "non_mask_value": self.non_mask_value
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskedAutoencoder":
        # Deserialize the nested encoder before reconstructing the model.
        config["encoder"] = keras.saving.deserialize_keras_object(config["encoder"])
        return cls(**config)