"""
Promptable segmentation with a heavy image encoder and a near-free mask decoder.

The problem SAM 1 solves is not "segment the object" but "segment *an* object,
given whatever the user points at" -- and a prompt is intrinsically ambiguous.
A single click on a shirt is a valid request for the shirt, for the person
wearing it, or for the checked pattern on its sleeve, and no amount of encoder
capacity can decide which one was meant. The design resolves this by refusing
to: the decoder emits a small fixed set of masks (three, plus a single-mask
token used when the caller says the prompt is unambiguous) and a separate head
predicts each one's IoU against the mask it is trying to be. Ambiguity becomes
a *ranking* problem the caller can settle, instead of an ill-posed regression
whose loss averages incompatible answers into a blur.

The second idea is an asymmetry in where the compute lives. The image encoder is
a full ViT and costs 89.7M to 637.0M parameters depending on variant; the prompt
encoder and mask decoder together cost about 4.06M **at every variant**, because
both run at a fixed `prompt_embed_dim=256` regardless of encoder width. The
image is encoded once into a `(B, img_size/16, img_size/16, 256)` grid, and each
subsequent click re-runs only the 4M-parameter tail. That split is what makes
interactive use viable at all, and it is why the variant table below touches the
encoder alone.

The image encoder is a plain, non-hierarchical ViT: windowed attention with a
learnable relative position bias everywhere, interrupted by full-grid global
blocks at four indices spread evenly through the depth, then a stride-1 neck of
a 1x1 and a 3x3 convolution that changes only the channel count to 256. The
grid is fixed by the patch embedding alone -- the neck does not resample. The
prompt encoder maps points, box corners and mask hints into that same 256-wide
space through a random-Gaussian Fourier positional encoding that is *shared*
with `get_dense_pe()`, so a prompt coordinate and an image position are
expressed in one frame rather than two that must be kept in sync. Points and
box corners additionally pick up one of four learned type embeddings, with
`not_a_point_embed` for padding rows and `no_mask_embed` standing in when no
mask prompt is given.

The mask decoder is a two-way transformer of depth 2: tokens attend to the
image and the image attends back to the tokens, with the positional encoding
re-added at every attention rather than once at the input so geometry survives
the residual updates. The head that finally produces a mask is a hypernetwork,
not a convolution -- a per-mask MLP emits a weight *vector* which is dotted
against the 4x-upscaled feature map. That is what keeps a per-prompt mask head
essentially free: the prompt-dependent part is a vector, and the expensive
spatial part is shared.

Two places in this file behave in ways a reader would otherwise guess wrong.
`preprocess` only PADS -- it normalizes and pads to the encoder size, and raises
`ValueError` on an oversize image rather than resizing it. Resizing here would
be silently wrong, because reference SAM resizes the raw image *before* the
model precisely so prompt coordinates can be rescaled by the same factor; a
hidden resize inside the model would leave every prompt in the wrong frame.
Callers apply `resize_longest_side` first. And `get_build_config` /
`build_from_config` deliberately run a full-resolution dummy forward pass on
every load: the ViT blocks and the two-way transformer build their attention
and FFN sublayers lazily, so the static `build()` chain alone materializes only
138 of the reduced fixture's 202 weights, Keras restores those 138, and the
other 64 are created fresh and random on the first real call -- a mask drift of
order 1-2 absolute with no error and no warning. A weight *count* cannot detect
this (both the correct and the broken build report 202/201/321,862 after any
forward pass), which is why the guards compare index-aligned weight values.

Two behavioural choices are worth stating as choices. `binarize_masks` defaults
to `True`, which casts the full-resolution `masks` output to `uint8` and makes
it gradient-dead -- differentiating it returns `None` for every trainable
variable, silently. It stays the default because it is reference SAM's own
output contract; `low_res_logits` is the training target at either setting, and
`binarize_masks=False` is the escape hatch when a differentiable full-resolution
mask is genuinely needed. Separately, `SAM.call` cannot be traced: it always
ends in `postprocess_masks`, whose `ops.image.resize` raises under graph mode
regardless of which key the caller reads. `SAMTrainingModel` in
`training_model.py` is the `fit()` path -- it drives the submodules directly and
never reaches the postprocess.

This package is architecture only, and that is deliberate rather than
unfinished. `SAM.from_variant('vit_b')` returns a randomly initialized model;
no official Meta checkpoint has ever been loaded here, no key-mapping layer
exists, and no accuracy or segmentation-quality claim is made anywhere in the
package. The parameter counts quoted below are this package's own
`count_params()` measurements at `img_size=1024`, not reference-PyTorch quotes,
and this implementation's layout deviations move them: encoder 89,670,912 /
308,278,272 / 637,026,048 for `vit_b` / `vit_l` / `vit_h`, over a
variant-independent 6,476 (prompt encoder) + 4,058,340 (mask decoder). Only
`vit_b` and a reduced fixture are ever forward-passed by the test suite; the two
larger variants are constructed and parameter-counted only.

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
from .transformer import TwoWayTransformer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM(keras.Model):
    """
    Segment Anything Model (SAM) - A foundation model for image segmentation.

    SAM is a promptable segmentation system that can generate high-quality object
    masks from various types of prompts (points, boxes, or masks). It consists of
    three main components:

    1. **Image Encoder**: A Vision Transformer (ViT) that processes the input
       image once to produce image embeddings.
    2. **Prompt Encoder**: Encodes various prompt types (points, boxes, masks)
       into embedding space.
    3. **Mask Decoder**: A lightweight transformer decoder that combines image
       and prompt embeddings to predict segmentation masks.

    **Intent**: To provide a unified interface for promptable segmentation that
    can be used for interactive segmentation, automatic mask generation, or as
    a component in larger vision pipelines.

    **Key Features**:
    - Supports multiple prompt types (points, boxes, masks)
    - Can predict single or multiple mask proposals
    - Provides mask quality scores (predicted IoU)
    - Fully serializable with complete state preservation
    - Pre-configured variants for different compute budgets

    Args:
        image_encoder: ImageEncoderViT instance, processes input images into
            feature embeddings.
        prompt_encoder: PromptEncoder instance, encodes user prompts into
            embeddings.
        mask_decoder: MaskDecoder instance, predicts masks from image and
            prompt embeddings.
        pixel_mean: List of floats, mean values for image normalization (RGB order).
            Defaults to ImageNet means [123.675, 116.28, 103.53].
        pixel_std: List of floats, standard deviation for image normalization (RGB order).
            Defaults to ImageNet stds [58.395, 57.12, 57.375].
        mask_threshold: Float, threshold for converting mask logits to binary masks.
            Defaults to 0.0.
        image_format: String, expected color format of input images. Currently
            only 'RGB' is supported. Defaults to 'RGB'.
        binarize_masks: Boolean, controls the `'masks'` output only. At `True`
            (the default, and reference SAM's own contract) `'masks'` is the
            thresholded `uint8` mask, which is **gradient-dead**. At `False`
            `'masks'` carries the full-resolution float logits and is
            differentiable. `'low_res_logits'` is unaffected and is the
            training target in both cases. Defaults to True.
        **kwargs: Additional arguments for the Model base class.

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

    Attributes:
        image_encoder: The ViT image encoder.
        prompt_encoder: The prompt encoder.
        mask_decoder: The mask decoder.
        pixel_mean: Image normalization mean.
        pixel_std: Image normalization standard deviation.
        mask_threshold: Threshold for binary mask conversion.
        image_format: Expected image format.
        binarize_masks: Whether the `'masks'` output is thresholded to `uint8`.

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
        },
        "vit_l": {
            "encoder_embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "encoder_global_attn_indexes": [5, 11, 17, 23],
        },
        "vit_b": {
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "encoder_global_attn_indexes": [2, 5, 8, 11],
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
        # DECISION plan-2026-08-03T191222-1d751f81/D-019: do NOT re-add
        # class-level `mask_threshold` / `image_format` defaults above
        # `__init__`. They were NOT the dead shadowed pair F-13 described:
        # TF's `KerasAutoTrackable.__setattr__` short-circuits
        # `if getattr(self, name) is value: return`, and BOTH defaults are
        # identical objects to the class-level ones (`"RGB"` is interned; the
        # `0.0` constants are deduped), so at the defaults these two lines set
        # NOTHING and the class attributes were the live storage. Measured:
        # with the pair present, `'mask_threshold' in instance.__dict__` was
        # False. Removing them makes the assignment actually reach the instance
        # dict; the resolved values are unchanged.
        self.mask_threshold = mask_threshold
        self.image_format = image_format
        self.binarize_masks = bool(binarize_masks)

        # Convert normalization parameters to tensors
        self.pixel_mean = ops.array(pixel_mean, dtype="float32")
        self.pixel_std = ops.array(pixel_std, dtype="float32")

        # Store as Python lists for serialization
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: the DEFAULT is a
        # tuple (R-009 S1) and the STORED attribute is a list. Keeping the
        # store as `list(...)` is what makes the conversion invisible: it is
        # the type `get_config` has always emitted, so a saved config's JSON
        # shape and every `== [..]` assertion in the suites are unchanged.
        self._pixel_mean_list = list(pixel_mean)
        self._pixel_std_list = list(pixel_std)

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Explicitly build the three sub-models.

        DECISION plan_2026-06-15_e6a0391c/D-008: `call()` takes a dict input
        (`inputs['image']`). Without a `build()` override, Keras tries to
        auto-trace `call()` with a single symbolic tensor, hits the dict
        membership check (`'image' not in inputs`), and emits a spurious
        "iterating over a symbolic tf.Tensor" UserWarning on first call. Defining
        `build()` makes Keras call it instead of tracing, and explicitly building
        the sub-models ensures their weights exist before any weight load
        (deserialization robustness). Each sub-build is idempotent and
        input-shape-independent, so the guards keep this safe to call repeatedly.
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
        Returns a build config so Keras invokes `build_from_config` on load.

        DECISION plan_2026-06-16_6e8c78a3/D-011: returning a non-empty dict is
        what makes Keras call `build_from_config` during deserialization. The
        image-encoder ViT blocks and the mask-decoder two-way transformer build
        their attention/FFN/norm sublayers LAZILY on the first `call()`, so the
        explicit `build()` chain alone materializes only ~92 of ~124 weights.
        Without a full materialization before weight restore, the lazily-built
        weights are created fresh (random) on first call and the saved values are
        silently dropped -> a save/load round-trip produced a model with ~30%
        mask drift. We pin `img_size` so `build_from_config` can run a dummy
        forward to materialize the complete weight set.

        Gated on `self.built`: a model saved UNBUILT (no forward pass yet) has no
        weights in the archive, so forcing a full build at load would raise
        "expected N variables, received 0". Returning None there preserves the
        stock unbuilt-save / unbuilt-load behavior.
        """
        if not self.built:
            return None
        return {"img_size": int(self.image_encoder.img_size)}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Fully materializes the weight set at load time (see D-011).

        Runs the static `build()` chain plus a single dummy forward pass so that
        every lazily-built sublayer (ViT blocks, two-way transformer, mask-decoder
        heads) creates its variables BEFORE Keras restores the saved weights.

        Cost, measured (F-11):
            The dummy forward is a full-resolution forward pass at the encoder's
            own `img_size`, plus a `(1, num_masks, img_size, img_size)`
            postprocess, on EVERY `load_model`. On the reduced test fixture
            (`img_size=256`, 321,862 params) `keras.models.load_model` takes
            roughly **0.82-0.86 s** steady-state (median ~0.84 s over 15 loads in
            3 processes; the first load in a process runs ~1.1 s from warm-up).
            The cost scales with the encoder, so a `vit_h` load pays a 1024x1024
            forward through 630M parameters. Wall-clock is deliberately NOT
            asserted anywhere: it is not reproducible across processes.
            Dropping the dummy forward saves ~0.29 s at fixture size (median
            0.54 s) and is REJECTED -- see the DECISION anchor below.
        """
        # DECISION plan-2026-08-03T191222-1d751f81/D-018: do NOT replace this
        # dummy forward with the `self.build(None)` chain alone, however
        # wasteful the full-resolution pass looks. Measured on the reduced
        # fixture: the build chain alone materializes only 138 of 202 weights at
        # load time, so Keras restores 138 and the remaining 64 are created
        # FRESH (random) on the first real `call()` -- a `low_res_logits` drift
        # of order 1-2 absolute, with no error and no warning. The weight COUNT
        # cannot see it: sampled after any forward pass, both variants report
        # 202/201/321,862 identically. Any future optimization here must be
        # judged by (a) `len(restored.weights)` sampled BEFORE the first call and
        # (b) index-aligned weight VALUES, never by a post-forward count.
        # Pinned by `TestBuildFromConfigLoadCost` in test_correctness.py.
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
        Forward pass through the SAM model.

        Args:
            inputs: Dictionary containing:
                - 'image': Required, shape (batch_size, H, W, 3)
                - 'points': Optional, tuple of (coords, labels)
                - 'boxes': Optional, shape (batch_size, num_boxes, 4)
                - 'masks': Optional, shape (batch_size, 1, mask_h, mask_w)
                - 'original_size': Required, tuple of (height, width)
            training: Optional boolean for training mode.
            multimask_output: Boolean, if True predicts multiple masks
                (usually 3), if False predicts single best mask. Defaults to True.

        Returns:
            Dictionary containing:
            - 'masks': Full-resolution masks. At the default
              `binarize_masks=True` these are thresholded `uint8` masks and are
              **not differentiable** -- differentiating this key returns `None`
              for every trainable variable. At `binarize_masks=False` they are
              float logits and carry gradient.
            - 'iou_predictions': Predicted IoU scores for each mask.
            - 'low_res_logits': Low-resolution mask logits. **This is the
              training target** -- it is differentiable at both settings of
              `binarize_masks` and is the output reference SAM supervises.

        Note:
            This dict is an inference contract, not a `fit()` contract. On
            keras 3.8.0 a dict `y_pred` cannot be trained with stock
            `compile()`/`fit()`: `CompileLoss.build` broadcasts a single `Loss`
            across every leaf of the structure and raises `KeyError`. A trainer
            must wrap this model in a single-tensor model that emits
            `low_res_logits` alone.
        """
        # Validate inputs
        if 'image' not in inputs:
            raise ValueError("Input dictionary must contain 'image' key")
        if 'original_size' not in inputs:
            raise ValueError("Input dictionary must contain 'original_size' key")

        image = inputs['image']  # (B, H, W, C)

        # Store input image shape for postprocessing
        input_image_shape = ops.shape(image)[1:3]

        # Step 1: Preprocess image (normalize and pad to encoder size)
        image = self.preprocess(image)

        # Step 2: Encode image to get image embeddings
        image_embeddings = self.image_encoder(image, training=training)

        # Step 3: Encode prompts (points, boxes, masks)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=inputs.get("points"),
            boxes=inputs.get("boxes"),
            masks=inputs.get("masks"),
            training=training
        )

        # Step 4: Decode masks from image and prompt embeddings
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            training=training,
        )

        # Step 5: Postprocess masks (upscale and threshold)
        masks = self.postprocess_masks(
            low_res_masks,
            input_image_shape,
            inputs["original_size"]
        )

        # DECISION plan-2026-08-03T191222-1d751f81/D-011: the `uint8` cast is
        # GRADIENT-DEAD -- differentiating `outputs['masks']` yields `None` for
        # EVERY trainable variable, so a trainer supervising this key trains
        # nothing and says nothing. It stays the DEFAULT because it is reference
        # SAM's own output contract; `binarize_masks=False` is the escape hatch.
        # Do NOT "simplify" this back to an unconditional cast, and do NOT
        # instead add a fourth `masks_logits` output key: that widens a dict
        # `y_pred` which already cannot be consumed by stock `compile()`/`fit()`
        # on keras 3.8.0, and pays a full-resolution tensor on every forward.
        # Flipping the default to float logits was also rejected (it breaks the
        # pre-existing binary-mask assertions and diverges from reference SAM).
        # `low_res_logits` is THE training target either way. See D-002.
        if self.binarize_masks:
            masks = ops.cast(masks > self.mask_threshold, dtype='uint8')

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

    def preprocess(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Preprocess input image for the encoder.

        Performs normalization using ImageNet statistics and pads the image
        to match the encoder's expected input size.

        Args:
            x: Input image tensor of shape (batch_size, H, W, 3) with values
                in [0, 255] range.

        Returns:
            Preprocessed image of shape (batch_size, img_size, img_size, 3)
            where img_size is the encoder's expected size (typically 1024).

        Raises:
            ValueError: If either spatial extent of `x` exceeds the encoder's
                `img_size`. This method only PADS; it never resizes. Apply
                `dl_techniques.models.SAM.SAM1.resize_longest_side` first, exactly as
                reference SAM's `ResizeLongestSide` transform does.
        """
        img_size = self.image_encoder.img_size

        # DECISION plan-2026-08-03T191222-1d751f81/D-005: refuse an oversize
        # image HERE, at the point where the contract is violated. Without this,
        # `pad_h`/`pad_w` below go negative and `ops.pad` raises
        # `InvalidArgumentError` -- which is an `OpError`, NOT a `ValueError`
        # (verified by execution: its MRO is OpError -> Exception), so a caller
        # cannot catch it as an input-validation error and the message names
        # neither the offending size nor the remedy. Do NOT "fix" this by
        # clamping the pad to zero (silently crops the image) and do NOT resize
        # inside `preprocess` (reference SAM resizes the raw image BEFORE the
        # model so prompt coordinates can be rescaled in the same frame; a
        # hidden resize here would leave every prompt in the wrong frame).
        # The static shape is used, not `ops.shape`, so the check is a plain
        # Python branch that neither traces into the graph nor fires on an
        # unknown-at-trace-time extent.
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
                        f"`dl_techniques.models.SAM.SAM1.resize_longest_side(image, "
                        f"{img_size})` before the model, as reference SAM's "
                        f"ResizeLongestSide transform does, and rescale the "
                        f"prompt coordinates by the same factor."
                    )

        # Normalize using ImageNet statistics.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-063
        # `pixel_mean` / `pixel_std` are built once in `__init__` as HARD
        # float32 constants (`ops.array(..., dtype="float32")`), so they must
        # be cast to the tensor's dtype at every use. Without the cast, the
        # line immediately below raised `InvalidArgumentError: cannot compute
        # Sub as input #1 was expected to be a half tensor but is a float
        # tensor` on ANY `mixed_float16` forward -- MEASURED at HEAD on the
        # reduced fixture, with the float32 control green. Do NOT fix this by
        # making `__init__` build the constants in the compute dtype: the
        # policy can change after construction, and a float16 `123.675` is not
        # exactly `123.675`. See decisions.md D-063.
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
        Postprocess predicted masks to match original image size.

        The mask decoder outputs low-resolution masks which need to be:
        1. Upscaled to encoder input size
        2. Cropped to remove padding
        3. Scaled to original image size

        Args:
            masks: Low-resolution mask logits from decoder, shape
                (batch_size, num_masks, H_low, W_low).
            input_size: Size of input image before padding, tuple of (H, W).
            original_size: Original image size before any preprocessing,
                tuple of (H, W).

        Returns:
            Masks at original image resolution, shape
            (batch_size, num_masks, original_H, original_W).
        """
        # Step 1: Upscale to encoder input size
        masks = ops.image.resize(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            interpolation="bilinear",
            data_format="channels_first"
        )

        # Step 2: Remove padding by cropping to input size
        masks = masks[..., :input_size[0], :input_size[1]]

        # Step 3: Scale to original image size
        masks = ops.image.resize(
            masks,
            original_size,
            interpolation="bilinear",
            data_format="channels_first"
        )

        return masks

    @classmethod
    def from_variant(
        cls,
        variant: Literal['vit_b', 'vit_l', 'vit_h'],
        **kwargs: Any
    ) -> 'SAM':
        """
        Create a SAM model from a predefined variant configuration.

        This factory method provides easy access to standard SAM architectures
        with different capacity/compute tradeoffs:

        - **vit_b** (Base): Fastest, lowest memory, good quality
        - **vit_l** (Large): Balanced speed/quality
        - **vit_h** (Huge): Best quality, highest resource requirements

        Args:
            variant: String, model variant name. Options are:
                - 'vit_b': Base model (768 dim, 12 layers, ~90M params)
                - 'vit_l': Large model (1024 dim, 24 layers, ~300M params)
                - 'vit_h': Huge model (1280 dim, 32 layers, ~630M params)
            **kwargs: Additional arguments to pass to SAM constructor
                (e.g., mask_threshold, pixel_mean, pixel_std).

        Returns:
            Configured SAM model instance.

        Raises:
            ValueError: If variant is not one of the supported options.

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

        # DECISION plan-2026-08-22T035419-a11304c8/D-091
        # `attention_dropout_rate` is deliberately NOT threaded here, and this is
        # the opposite ruling to SAM 2's (D-090) for one measured reason: this
        # transformer's `attention_dropout_rate` default is 0.0, so the rate that
        # `from_variant` cannot reach is a rate that does NOTHING. MEASURED
        # 2026-08-22 on `vit_b`: 0 live `keras.layers.Dropout`, 7
        # `MultiHeadAttention` all at `dropout=0.0`. SAM 2 got a knob because
        # its unreachable default was 0.1 -- regularization no caller could
        # switch off. Adding one here would buy a variant-table key, a
        # `get_config` surface and two guards, all to make a no-op
        # configurable. COST, stated so it is not rediscovered as a bug: a
        # caller who wants SAM 1 attention dropout must build the transformer
        # themselves -- `SAM(mask_decoder=MaskDecoder(transformer=
        # TwoWayTransformer(..., attention_dropout_rate=r), ...), ...)` -- rather
        # than passing it to `from_variant`. See decisions.md D-091.
        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            num_heads=8,
            mlp_dim=2048,
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
        Returns the configuration of the model for serialization.

        This method serializes all sub-models and configuration parameters,
        enabling full model reconstruction via `from_config`.

        Returns:
            Configuration dictionary containing all model parameters.
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
        Creates a SAM model from a configuration dictionary.

        This method deserializes the model from a configuration, reconstructing
        all sub-models and restoring their weights.

        Args:
            config: Configuration dictionary from `get_config()`.

        Returns:
            Reconstructed SAM model instance.
        """
        # Deserialize sub-models
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

        Args:
            input_shape: Dictionary of input shapes.

        Returns:
            Dictionary of output shapes.
        """
        batch_size = input_shape.get('image', (None,))[0]
        original_h, original_w = None, None  # Unknown until runtime

        # Number of masks depends on multimask_output setting
        # Default is 3 for multimask, 1 for single mask
        num_masks = None  # Variable

        return {
            'masks': (batch_size, num_masks, original_h, original_w),
            'iou_predictions': (batch_size, num_masks),
            'low_res_logits': (batch_size, num_masks, None, None),
        }

# ---------------------------------------------------------------------
