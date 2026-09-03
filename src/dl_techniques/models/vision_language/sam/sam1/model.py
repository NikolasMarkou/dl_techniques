"""
Promptable image segmentation, built by :class:`SAM`, from an image encoder,
a prompt encoder and a mask decoder.

A prompt (a click, a box) is ambiguous: it can mean the object, the region
around it, or a detail inside it. Instead of predicting one mask, the model
predicts a small fixed set (three, plus a single-mask option) with a
separate IoU head scoring each one, turning ambiguity into a ranking problem
for the caller rather than an averaged, blurred regression target. Compute
is asymmetric: the ViT image encoder runs once per image and costs 90M-637M
parameters depending on variant, while the prompt encoder and mask decoder
together cost about 4.06M at every variant, so each subsequent prompt only
re-runs the cheap tail.

`preprocess` only pads to the encoder's input size and raises on an
oversize image rather than resizing it -- resizing happens before the model,
via `resize_longest_side`, so prompt coordinates and image stay in one
frame. `binarize_masks` defaults to True, matching reference SAM's own
output contract; the returned `'masks'` are then `uint8` and not
differentiable, so a trainer supervises `'low_res_logits'` instead.
`get_build_config`/`build_from_config` run a full dummy forward on load
because the ViT and two-way-transformer sublayers build lazily on first
call, and skipping that dummy forward would silently drop part of a
restored checkpoint's weights.

This package ships architecture only: `SAM.from_variant` returns randomly
initialized weights, with no checkpoint loading and no accuracy claim.

References:
    - Kirillov et al., 2023. Segment Anything.
      (https://arxiv.org/abs/2304.02643)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Ha et al., 2016. HyperNetworks.
      (https://arxiv.org/abs/1609.09106)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
from keras import ops
from typing import Tuple, List, Any, Dict, Optional, Literal, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .image_encoder import ImageEncoderViT
from .transformer import DEFAULT_ATTENTION_DROPOUT_RATE, TwoWayTransformer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam1.model")
class SAM(keras.Model):
    """
    A promptable segmentation model: a ViT image encoder, a prompt encoder,
    and a hypernetwork mask decoder.

    Architecture:

    .. code-block:: text

        image [B, H, W, 3]
              │
              ▼
        preprocess (normalize, pad)
              │
              ▼
        ImageEncoderViT ──► image_embeddings [B, h, w, 256]
                                     │
        points/boxes/masks ─► PromptEncoder ─► sparse, dense embeddings
                                     │                │
                                     ▼                ▼
                              MaskDecoder (image_embeddings, image_pe,
                                           sparse, dense)
                                     │
                                     ▼
                        postprocess_masks ──► masks, iou, low_res_logits

    :param image_encoder: ViT encoder that turns an image into feature
        embeddings.
    :type image_encoder: ImageEncoderViT
    :param prompt_encoder: Encodes user prompts into embeddings.
    :type prompt_encoder: PromptEncoder
    :param mask_decoder: Predicts masks from image and prompt embeddings.
    :type mask_decoder: MaskDecoder
    :param pixel_mean: Per-channel mean for image normalization, RGB order.
        Defaults to ImageNet means.
    :type pixel_mean: Sequence[float]
    :param pixel_std: Per-channel standard deviation for image
        normalization, RGB order. Defaults to ImageNet stds.
    :type pixel_std: Sequence[float]
    :param mask_threshold: Threshold for converting mask logits to binary
        masks. Defaults to 0.0.
    :type mask_threshold: float
    :param image_format: Expected input color format. Only ``'RGB'`` is
        supported.
    :type image_format: str
    :param binarize_masks: Controls the ``'masks'`` output only. At True
        (the default, matching reference SAM), ``'masks'`` is a thresholded
        ``uint8`` mask and not differentiable. At False it carries float
        logits and is differentiable. ``'low_res_logits'`` is the training
        target either way.
    :type binarize_masks: bool
    :param kwargs: Additional arguments for the Model base class.
    :ivar image_encoder: The ViT image encoder.
    :ivar prompt_encoder: The prompt encoder.
    :ivar mask_decoder: The mask decoder.

    Input shape (in call):
        Dictionary with the following keys:
        - 'image': Required tensor of shape (batch_size, H, W, 3)
        - 'points': Optional tuple of (coords, labels) where:
            - coords: Shape (batch_size, num_points, 2) with (x, y) coordinates
            - labels: Shape (batch_size, num_points) with point labels
        - 'boxes': Optional tensor of shape (batch_size, num_boxes, 4) with
            (x1, y1, x2, y2) coordinates
        - 'masks': Optional tensor of shape (batch_size, 1, mask_h, mask_w)
        - 'original_size': Required tuple of (height, width) for the original
            image size before any preprocessing

    Output shape:
        Dictionary with the following keys:
        - 'masks': Shape (batch_size, num_masks, H, W). Binary `uint8` masks at
            the default `binarize_masks=True`; float logits at `False`.
        - 'iou_predictions': Quality scores of shape (batch_size, num_masks)
        - 'low_res_logits': Low-resolution mask logits of shape
            (batch_size, num_masks, H/4, W/4)

    Training contract:
        **`low_res_logits` is THE training target.** It is the only mask output
        that is differentiable at every setting, and it is what reference SAM
        supervises (upsampling to full resolution costs memory and buys no
        signal). At the default `binarize_masks=True` the `'masks'` output is
        cast to `uint8`, so differentiating it returns `None` for **every**
        trainable variable -- a trainer that supervises `'masks'` trains
        nothing and reports no error. Set `binarize_masks=False` if a
        full-resolution differentiable mask is genuinely required.

        Independently of that flag, this model's dict output **cannot be
        trained with stock `compile()`/`fit()` on keras 3.8.0**: given a dict
        `y_pred`, `CompileLoss.build` broadcasts one `Loss` across every leaf of
        the structure and dies with a `KeyError`. A trainer therefore needs a
        thin single-tensor wrapper model that returns `low_res_logits` alone.
        No such wrapper ships in this package yet.

    Example:
        ```python
        # Create model from variant
        model = SAM.from_variant('vit_b')

        # Prepare inputs
        image = keras.random.normal(shape=(1, 1024, 1024, 3))
        points = (
            keras.ops.convert_to_tensor([[500.0, 500.0]]),
            keras.ops.convert_to_tensor([[1]])
        )

        # Get predictions
        outputs = model({
            'image': image,
            'points': points,
            'original_size': (1024, 1024)
        })

        # Access results
        masks = outputs['masks']  # Binary masks
        iou_scores = outputs['iou_predictions']  # Quality scores
        ```

    Note:
        The model expects images in RGB format with values in [0, 255] range.
        The image encoder processes the full image once, making subsequent
        predictions with different prompts very efficient.
    """

    #: Public-name registry of the three published SAM 1 checkpoint geometries
    #: (models/CLAUDE.md Axis 2). Hoisted verbatim out of ``from_variant``'s body
    #: on 2026-08-19: the table was a local ``configs`` dict, so nothing outside
    #: the method could enumerate the variants -- ``getattr(SAM, "MODEL_VARIANTS")``
    #: raised ``AttributeError``, the same failure mode ``fastvit`` had.
    #:
    #: Every value is a kwarg of :class:`ImageEncoderViT`; the prompt-encoder and
    #: mask-decoder geometry is IDENTICAL across all three variants and therefore
    #: stays in ``from_variant`` rather than being restated three times here.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "vit_h": {
            "encoder_embed_dim": 1280,
            "encoder_depth": 32,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [7, 15, 23, 31],
            "dropout_rate": DEFAULT_ATTENTION_DROPOUT_RATE,
        },
        "vit_l": {
            "encoder_embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [5, 11, 17, 23],
            "dropout_rate": DEFAULT_ATTENTION_DROPOUT_RATE,
        },
        "vit_b": {
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "encoder_global_attn_indexes": [2, 5, 8, 11],
            "dropout_rate": DEFAULT_ATTENTION_DROPOUT_RATE,
        },
    }

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: Sequence[float] = (123.675, 116.28, 103.53),
        pixel_std: Sequence[float] = (58.395, 57.12, 57.375),
        mask_threshold: float = 0.0,
        image_format: str = "RGB",
        binarize_masks: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if len(pixel_mean) != 3:
            raise ValueError(f"pixel_mean must have 3 values (RGB), got {len(pixel_mean)}")
        if len(pixel_std) != 3:
            raise ValueError(f"pixel_std must have 3 values (RGB), got {len(pixel_std)}")
        if image_format != "RGB":
            raise ValueError(f"Only 'RGB' image format is supported, got '{image_format}'")

        # Store all configuration parameters
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # DECISION plan-2026-08-03T191222-1d751f81/D-019: no class-level
        # mask_threshold/image_format defaults above __init__ -- TF's
        # __setattr__ short-circuits an assignment identical to the class default, so a class-level pair silently shadows the instance dict. See decisions.md.
        self.mask_threshold = mask_threshold
        self.image_format = image_format
        self.binarize_masks = bool(binarize_masks)

        # Convert normalization parameters to tensors
        self.pixel_mean = ops.array(pixel_mean, dtype="float32")
        self.pixel_std = ops.array(pixel_std, dtype="float32")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: store as list, not
        # the constructor's tuple default -- get_config has always emitted a list. See decisions.md.
        self._pixel_mean_list = list(pixel_mean)
        self._pixel_std_list = list(pixel_std)

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Explicitly build the three sub-models.

        :param input_shape: Not used; ``call`` takes a dict input.
        :type input_shape: Optional[Any]

        .. note::
           DECISION plan_2026-06-15_e6a0391c/D-008: without this override,
           Keras auto-traces ``call()`` with a single symbolic tensor and hits
           the dict membership check, emitting a spurious warning. See decisions.md.
        """
        if not self.image_encoder.built:
            img = self.image_encoder.img_size
            self.image_encoder.build((None, img, img, 3))
        if not self.prompt_encoder.built:
            self.prompt_encoder.build(None)
        if not self.mask_decoder.built:
            self.mask_decoder.build(None)
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        """
        Return a build config so Keras invokes ``build_from_config`` on load,
        or None if the model was never built.

        :return: ``{"img_size": ...}`` if built, else None.
        :rtype: Optional[Dict[str, Any]]

        .. note::
           DECISION plan_2026-06-16_6e8c78a3/D-011: a non-empty dict is what
           makes Keras call ``build_from_config``. The lazily-built ViT and
           two-way-transformer sublayers materialize only ~92 of ~124 weights
           from the static build chain alone, so skipping this dropped ~30% of a
           restored checkpoint's weights. Gated on ``self.built`` -- an unbuilt
           model has no weights in the archive, so forcing a build at load
           would raise. See decisions.md.
        """
        if not self.built:
            return None
        return {"img_size": int(self.image_encoder.img_size)}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Run the static build chain plus a dummy forward pass so every
        lazily-built sublayer materializes its variables before Keras
        restores saved weights.

        :param config: Build config from :meth:`get_build_config`.
        :type config: Dict[str, Any]

        .. note::
           DECISION plan-2026-08-03T191222-1d751f81/D-018: the dummy forward
           cannot be replaced by the static build chain alone -- that chain
           materializes only 138 of 202 weights on the reduced fixture, so the
           other 64 come back random with no error. See decisions.md.
        """
        img_size = int(config.get("img_size") or self.image_encoder.img_size)
        if not self.built:
            self.build(None)
        # Materialize lazily-built attention/transformer sublayers via a dummy
        # forward (image-only; no prompts needed to trigger the full graph).
        dummy_inputs = {
            "image": ops.zeros((1, img_size, img_size, 3), dtype="float32"),
            "original_size": ops.convert_to_tensor((img_size, img_size)),
        }
        self(dummy_inputs, multimask_output=True)

    def call(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
        multimask_output: bool = True
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the full segmentation pipeline: preprocess, encode, decode,
        postprocess.

        :param inputs: Dict with ``'image'`` (required,
            ``(batch_size, H, W, 3)``), ``'points'`` (optional, ``(coords,
            labels)``), ``'boxes'`` (optional, ``(batch_size, num_boxes,
            4)``), ``'masks'`` (optional, ``(batch_size, 1, mask_h,
            mask_w)``), and ``'original_size'`` (required, ``(height,
            width)``).
        :type inputs: Dict[str, Any]
        :param training: Whether the model runs in training mode.
        :type training: Optional[bool]
        :param multimask_output: If True, predict multiple masks (usually
            3); if False, predict the single best mask. Defaults to True.
        :type multimask_output: bool
        :return: Dict with ``'masks'`` (thresholded ``uint8`` and
            non-differentiable at the default ``binarize_masks=True``, float
            logits otherwise), ``'iou_predictions'``, and
            ``'low_res_logits'`` -- the training target, differentiable at
            either ``binarize_masks`` setting.
        :rtype: Dict[str, keras.KerasTensor]

        .. note::
           This dict is an inference contract, not a ``fit()`` contract. On
           keras 3.8.0 a dict ``y_pred`` cannot be trained with stock
           ``compile()``/``fit()``; a trainer needs a wrapper model that
           emits ``low_res_logits`` alone.
        """
        if 'image' not in inputs:
            raise ValueError("Input dictionary must contain 'image' key")
        if 'original_size' not in inputs:
            raise ValueError("Input dictionary must contain 'original_size' key")

        image = inputs['image']

        input_image_shape = ops.shape(image)[1:3]
        image = self.preprocess(image)
        image_embeddings = self.image_encoder(image, training=training)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=inputs.get("points"),
            boxes=inputs.get("boxes"),
            masks=inputs.get("masks"),
            training=training
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            training=training,
        )

        masks = self.postprocess_masks(
            low_res_masks,
            input_image_shape,
            inputs["original_size"]
        )

        # DECISION plan-2026-08-03T191222-1d751f81/D-011: the uint8 cast is
        # gradient-dead, but stays the default to match reference SAM's own
        # output contract; low_res_logits is the training target either way. See decisions.md.
        if self.binarize_masks:
            masks = ops.cast(masks > self.mask_threshold, dtype='uint8')

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

    def preprocess(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Normalize with ImageNet statistics and pad to the encoder's input
        size.

        :param x: Image in ``[0, 255]``, shape ``(batch_size, H, W, 3)``.
        :type x: keras.KerasTensor
        :return: Preprocessed image, shape
            ``(batch_size, img_size, img_size, 3)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If either spatial extent of ``x`` exceeds the
            encoder's ``img_size``. This method only pads; it never resizes.
            Apply
            ``dl_techniques.models.vision_language.sam.sam1.resize_longest_side``
            first, as reference SAM's ``ResizeLongestSide`` transform does.
        """
        img_size = self.image_encoder.img_size

        # DECISION plan-2026-08-03T191222-1d751f81/D-005: refuse an oversize
        # image here, at the point of violation -- otherwise pad_h/pad_w go
        # negative and ops.pad raises an uncatchable-as-ValueError OpError. See decisions.md.
        static_shape = tuple(x.shape)
        if len(static_shape) == 4:
            for axis_name, dim in (
                ("height", static_shape[1]),
                ("width", static_shape[2]),
            ):
                if dim is not None and int(dim) > img_size:
                    raise ValueError(
                        f"SAM.preprocess pads to the encoder size and cannot "
                        f"shrink an image: input {axis_name}={int(dim)} exceeds "
                        f"image_encoder.img_size={img_size} (input shape "
                        f"{static_shape}). Apply "
                        f"`dl_techniques.models.vision_language.sam.sam1.resize_longest_side(image, "
                        f"{img_size})` before the model, as reference SAM's "
                        f"ResizeLongestSide transform does, and rescale the "
                        f"prompt coordinates by the same factor."
                    )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-063: pixel_mean/pixel_std
        # are fixed float32 constants, so they must be cast to the tensor's
        # dtype at every use, not once in __init__ -- the dtype policy can change after construction. See decisions.md.
        x = ops.cast(x, self.compute_dtype)
        x = (x - ops.cast(self.pixel_mean, x.dtype)) / ops.cast(
            self.pixel_std, x.dtype)

        # Pad to encoder size
        h, w = ops.shape(x)[1], ops.shape(x)[2]
        pad_h = self.image_encoder.img_size - h
        pad_w = self.image_encoder.img_size - w

        # Add padding to bottom and right
        x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        return x

    def postprocess_masks(
        self,
        masks: keras.KerasTensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> keras.KerasTensor:
        """
        Upscale to encoder size, crop off padding, then resize to the
        original image size.

        :param masks: Low-resolution mask logits from the decoder, shape
            ``(batch_size, num_masks, H_low, W_low)``.
        :type masks: keras.KerasTensor
        :param input_size: Input image size before padding, ``(H, W)``.
        :type input_size: Tuple[int, int]
        :param original_size: Original image size before any preprocessing,
            ``(H, W)``.
        :type original_size: Tuple[int, int]
        :return: Masks at original resolution, shape
            ``(batch_size, num_masks, original_H, original_W)``.
        :rtype: keras.KerasTensor
        """
        masks = ops.image.resize(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            interpolation="bilinear",
            data_format="channels_first"
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = ops.image.resize(
            masks,
            original_size,
            interpolation="bilinear",
            data_format="channels_first"
        )

        return masks

    # DECISION plan-2026-08-23T091307-9a110062/D-601: dropout_rate is derived,
    # not a stored __init__/get_config field -- a second copy on the outer model
    # could silently disagree with the transformer's own stored rate. See decisions.md.
    @property
    def dropout_rate(self) -> float:
        """Attention-dropout rate actually in force on the mask decoder.

        :return: The rate carried by ``mask_decoder.transformer``.
        :rtype: float
        """
        return float(self.mask_decoder.transformer.attention_dropout_rate)

    @classmethod
    def from_variant(
        cls,
        variant: Literal['vit_b', 'vit_l', 'vit_h'],
        **kwargs: Any
    ) -> 'SAM':
        """
        Build a SAM model from a named size preset (``vit_b``/``vit_l``/``vit_h``).

        :param variant: ``'vit_b'`` (768 dim, 12 layers, ~90M params),
            ``'vit_l'`` (1024 dim, 24 layers, ~300M params), or ``'vit_h'``
            (1280 dim, 32 layers, ~630M params).
        :type variant: Literal['vit_b', 'vit_l', 'vit_h']
        :param kwargs: Additional arguments passed to the SAM constructor
            (e.g. ``mask_threshold``, ``pixel_mean``). One key is
            intercepted rather than forwarded: ``dropout_rate`` reaches
            every ``Dropout`` in the mask decoder's ``TwoWayTransformer``.
            It defaults to ``DEFAULT_ATTENTION_DROPOUT_RATE`` (0.0), so
            omitting it reproduces the shipped model bit for bit.
        :return: A configured :class:`SAM` model instance.
        :rtype: SAM
        :raises ValueError: If ``variant`` is not supported, or if
            ``dropout_rate`` is outside ``[0.0, 1.0)``.

        Example:
            ```python
            # Create different model sizes
            model_base = SAM.from_variant('vit_b')
            model_large = SAM.from_variant('vit_l')
            model_huge = SAM.from_variant('vit_h')

            # Create with custom settings
            model = SAM.from_variant(
                'vit_b',
                mask_threshold=0.5,
                pixel_mean=[120.0, 115.0, 100.0]
            )
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant: '{variant}'. "
                f"Supported variants are: {sorted(cls.MODEL_VARIANTS)}"
            )

        config = cls.MODEL_VARIANTS[variant]

        # DECISION plan-2026-08-23T091307-9a110062/D-601: dropout_rate is
        # popped from kwargs here, not forwarded to cls(...), since it
        # configures a sub-layer rather than SAM.__init__. See decisions.md.
        dropout_rate = kwargs.pop("dropout_rate", None)
        if dropout_rate is None:
            dropout_rate = config["dropout_rate"]
        dropout_rate = float(dropout_rate)
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}"
            )

        # Common configuration across all variants
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        # Create image encoder (ViT)
        image_encoder = ImageEncoderViT(
            img_size=image_size,
            patch_size=vit_patch_size,
            embed_dim=config["encoder_embed_dim"],
            depth=config["encoder_depth"],
            num_heads=config["encoder_num_heads"],
            mlp_ratio=4.0,
            out_chans=prompt_embed_dim,
            qkv_bias=True,
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=config["encoder_global_attn_indexes"],
        )

        # Create prompt encoder
        prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )

        # DECISION plan-2026-08-23T091307-9a110062/D-601: this is the only
        # path by which a caller-chosen dropout rate reaches the 7 Dropout
        # layers this transformer builds -- dropping it silently makes the knob dead at its 0.0 default. See decisions.md.
        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            num_heads=8,
            mlp_dim=2048,
            attention_dropout_rate=dropout_rate,
        )

        # Create mask decoder
        mask_decoder = MaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=transformer,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # Create and return SAM model
        return cls(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Serialize all sub-models and configuration parameters for full
        reconstruction via :meth:`from_config`.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "image_encoder": keras.layers.serialize(self.image_encoder),
            "prompt_encoder": keras.layers.serialize(self.prompt_encoder),
            "mask_decoder": keras.layers.serialize(self.mask_decoder),
            "pixel_mean": self._pixel_mean_list,
            "pixel_std": self._pixel_std_list,
            "mask_threshold": self.mask_threshold,
            "image_format": self.image_format,
            "binarize_masks": self.binarize_masks,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SAM':
        """
        Deserialize a :class:`SAM` model, reconstructing its sub-models.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: Reconstructed :class:`SAM` instance.
        :rtype: SAM
        """
        image_encoder_config = config.pop("image_encoder")
        prompt_encoder_config = config.pop("prompt_encoder")
        mask_decoder_config = config.pop("mask_decoder")

        config["image_encoder"] = keras.layers.deserialize(image_encoder_config)
        config["prompt_encoder"] = keras.layers.deserialize(prompt_encoder_config)
        config["mask_decoder"] = keras.layers.deserialize(mask_decoder_config)

        return cls(**config)

    def compute_output_shape(
        self,
        input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Compute output shapes given input shapes.

        :param input_shape: Dictionary of input shapes.
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: Dictionary of output shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch_size = input_shape.get('image', (None,))[0]
        # Unknown until runtime: original size and mask count (depends on
        # multimask_output).
        original_h, original_w = None, None
        num_masks = None

        return {
            'masks': (batch_size, num_masks, original_h, original_w),
            'iou_predictions': (batch_size, num_masks),
            'low_res_logits': (batch_size, num_masks, None, None),
        }

# ---------------------------------------------------------------------
