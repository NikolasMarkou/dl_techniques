"""
SAM 1 Mask Decoder: output tokens, a two-way transformer and a hypernetwork.
============================================================================

:class:`MaskDecoder` consumes the image embedding and the prompt embeddings and
emits low-resolution mask logits plus one predicted IoU per mask. The masks are
not produced by a convolutional head: a per-mask MLP emits a weight VECTOR
which is dotted against the upscaled feature map, so the head is a hypernetwork.

Based on:
---------
- Kirillov, A. et al. (2023). "Segment Anything." https://arxiv.org/abs/2304.02643

Key Features:
------------
- ``num_multimask_outputs + 1`` learnable mask tokens -- index 0 is the
  single-mask output returned at ``multimask_output=False``, the rest are the
  proposals -- plus one IoU token, concatenated in front of the sparse prompts.
- Output upscaling by 4x via transposed convolutions; the mask logits are the
  matrix product of the upscaled map with the hypernetwork vectors.
- Every MLP head is ``iou_head_depth`` Dense layers total, hidden layers
  activated by ``mlp_activation`` and the final layer linear so it can emit
  signed logits.

Architecture Overview:
---------------------
1. ``image_embeddings + dense_prompt_embeddings`` -> source ``(B, H, W, C)``.
2. ``[iou_token, mask_tokens, sparse_prompts]`` -> tokens ``(B, 1+N+M, C)``.
3. -> :class:`TwoWayTransformer` -> updated tokens and updated source.
4. -> source upscaled 4x to ``(B, 4H, 4W, C/8)``; each mask token -> its own
   MLP -> ``(B, N, C/8)``.
5. -> matmul -> mask logits ``(B, N, 4H, 4W)``; IoU token -> MLP -> ``(B, N)``.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM1.mask_decoder import MaskDecoder
from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
decoder = MaskDecoder(transformer_dim=256, num_multimask_outputs=3,
                      transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                                    num_heads=8, mlp_dim=2048))
masks, iou = decoder(image_embeddings=feat, image_pe=pe,
                     sparse_prompt_embeddings=sparse,
                     dense_prompt_embeddings=dense, multimask_output=True)
```

Measured caveats:
----------------
- **``activation`` and ``mlp_activation`` are two SEPARATE knobs and their
  defaults deliberately differ** -- ``'gelu'`` for ``output_upscaling`` only,
  ``'relu'`` for every non-final layer of the hypernetwork and IoU heads. This
  mirrors reference SAM, whose ``MaskDecoder(activation=...)`` reaches the
  upscaler alone while ``MLP`` hardcodes ``F.relu``. Do NOT collapse them into
  one knob "so the decoder agrees with itself": either single choice makes one
  of the two halves wrong, and the error is a value drift with no shape symptom.
- **``iou_head_depth`` drives BOTH the IoU head and every hypernetwork MLP.**
  Reference SAM hardcodes 3 for the hypernetwork and applies the knob to the
  IoU head only, so this is a deliberate deviation reachable only at a
  non-default depth. At the default 3 the two agree exactly.
- **``sparse_prompt_embeddings`` must carry a batch of 1 or exactly
  ``batch_size``.** Anything in between raises ``ValueError``; before that
  guard existed an intermediate value tiled into an interleaved ``[a, b, a,
  b]`` order and scored every image against the wrong prompt with no error at
  all.
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformer import TwoWayTransformer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------


def _build_mlp_head(
    *,
    num_layers: int,
    hidden_dim: int,
    output_dim: int,
    activation: str,
    dense_name_template: str,
    name: str,
) -> keras.Sequential:
    """
    Build reference SAM's ``MLP``: ``num_layers`` Dense layers, last one linear.

    Interface contract (shared by the hypernetwork MLPs and the IoU head):

    Args:
        num_layers: Total number of ``Dense`` layers, i.e. reference SAM's
            ``MLP(..., num_layers=N)``. ``N`` layers means ``N - 1`` hidden
            layers of width ``hidden_dim`` followed by one ``output_dim``
            layer. Must be ``>= 1``; callers validate.
        hidden_dim: Width of every layer except the last.
        output_dim: Width of the last layer.
        activation: Activation applied to every layer EXCEPT the last, which is
            always linear (reference SAM applies ``relu`` to all but the final
            ``Linear``, so the head can emit signed logits).
        dense_name_template: Naming pattern for the sub-layers, containing a
            single ``{n}`` placeholder filled with the 1-based layer index --
            e.g. ``"iou_dense{n}"`` or ``"hyper_dense{n}_0"``. Chosen so that
            at ``num_layers == 2`` the names are byte-identical to the ones
            this package shipped before the depth knob was made live.
        name: Name of the returned ``Sequential``.

    Returns:
        An UNBUILT ``keras.Sequential`` of ``num_layers`` ``Dense`` layers.

    Raises:
        ValueError: never directly; ``num_layers < 1`` yields an empty
            ``Sequential``, which is why every caller validates the depth first.
    """
    dense_layers = []
    for index in range(num_layers):
        is_last = index == num_layers - 1
        dense_layers.append(
            keras.layers.Dense(
                output_dim if is_last else hidden_dim,
                activation=None if is_last else activation,
                name=dense_name_template.format(n=index + 1),
            )
        )
    return keras.Sequential(dense_layers, name=name)


@keras.saving.register_keras_serializable()
class MaskDecoder(keras.layers.Layer):
    """
    Predicts segmentation masks from image and prompt embeddings using a transformer.

    This layer is the final component of the Segment Anything Model (SAM), responsible
    for generating mask predictions from encoded image features and user prompts. It
    uses a two-way transformer to jointly refine prompt and image embeddings, followed
    by upscaling and dynamic mask prediction.

    **Intent**: To provide a flexible, high-quality mask decoder that can generate
    multiple mask proposals with quality estimates, supporting both single-mask and
    multi-mask prediction modes.

    **Architecture**:
    The decoder processes inputs through several stages:
    1. Combines image embeddings with dense prompt embeddings
    2. Prepends learnable output tokens (IoU + mask tokens) to sparse prompts
    3. Runs two-way transformer to update both tokens and image features
    4. Upscales image features 4x using transposed convolutions
    5. Uses hypernetwork MLPs to generate mask-specific parameters
    6. Produces final masks via dynamic convolution (matrix multiplication)
    7. Predicts mask quality (IoU) from IoU token

    Args:
        transformer_dim: Integer, the embedding dimension used by the transformer.
            Must match the dimension of input embeddings.
        transformer: TwoWayTransformer instance, the core transformer for joint
            refinement of prompts and image features.
        num_multimask_outputs: Integer, number of mask predictions to generate
            beyond the single output mask. Defaults to 3. Total masks =
            num_multimask_outputs + 1.
        iou_head_depth: Integer, total number of Dense layers in EACH MLP head
            -- the IoU prediction head and every hypernetwork MLP. Matches
            reference SAM's ``MLP(..., num_layers=N)``: ``N - 1`` hidden layers
            plus one linear output layer. Must be positive. Defaults to 3,
            which is reference SAM's value for both heads.
        iou_head_hidden_dim: Integer, hidden dimension of the IoU prediction head.
            Defaults to 256.
        normalization_type: String, type of normalization to use in upscaling module.
            Supports 'layer_norm', 'rms_norm', 'batch_norm'. Defaults to 'layer_norm'.
        activation: String, activation applied inside the ``output_upscaling``
            module ONLY. Defaults to 'gelu', which is reference SAM's
            ``MaskDecoder(activation=nn.GELU)`` default -- there, ``activation``
            is passed to ``output_upscaling`` and to nothing else.
        mlp_activation: String, activation applied to every non-final layer of
            the hypernetwork MLPs and the IoU head. Defaults to 'relu', which
            reference SAM hardcodes inside ``MLP`` (``F.relu``). It is a
            separate knob from ``activation`` because reference SAM makes the
            two halves DIFFER; see the D-024 anchor below.
        **kwargs: Additional arguments for the Layer base class.

    Input shape (in call):
        - image_embeddings: Shape (batch_size, H, W, transformer_dim)
        - image_pe: Shape (batch_size, H, W, transformer_dim), positional encoding
        - sparse_prompt_embeddings: Shape (batch_size, num_sparse, transformer_dim)
        - dense_prompt_embeddings: Shape (batch_size, H, W, transformer_dim)
        - multimask_output: Boolean, whether to return multiple masks or single mask

    Output shape:
        Tuple of two tensors:
        - masks: Shape (batch_size, num_masks, H*4, W*4) where num_masks is either
            num_multimask_outputs (if multimask_output=True) or 1 (if False)
        - iou_predictions: Shape (batch_size, num_masks) with predicted IoU scores

    Attributes:
        iou_token: Embedding layer for the IoU prediction token.
        mask_tokens: Embedding layer for mask prediction tokens.
        output_upscaling: Sequential model for 4x upsampling of image features.
        output_hypernetworks_mlps: List of MLP heads, one per mask token.
        iou_prediction_head: MLP for predicting mask quality (IoU).

    Example:
        ```python
        # Create decoder
        from .transformer import TwoWayTransformer

        transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8)
        decoder = MaskDecoder(transformer_dim=256, transformer=transformer)

        # Generate masks
        masks, iou_pred = decoder(
            image_embeddings=image_emb,
            image_pe=pos_encoding,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=True
        )
        ```

    Note:
        The transformer is passed as a parameter to allow flexible transformer
        architectures while maintaining the decoder's structure. The transformer
        must implement a compatible interface returning (tokens, source) tuple.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'gelu',
        mlp_activation: str = 'relu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if transformer_dim <= 0:
            raise ValueError(f"transformer_dim must be positive, got {transformer_dim}")
        if num_multimask_outputs <= 0:
            raise ValueError(f"num_multimask_outputs must be positive, got {num_multimask_outputs}")
        if iou_head_depth <= 0:
            raise ValueError(
                f"iou_head_depth must be positive, got {iou_head_depth}. It is the "
                f"TOTAL number of Dense layers per MLP head (reference SAM uses 3); "
                f"a non-positive value would build an empty Sequential that passes "
                f"the token straight through and only fails later as a shape error."
            )
        if iou_head_hidden_dim <= 0:
            raise ValueError(f"iou_head_hidden_dim must be positive, got {iou_head_hidden_dim}")

        # Store all configuration parameters
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.normalization_type = normalization_type
        # DECISION plan-2026-08-03T191222-1d751f81/D-024
        # These are TWO knobs on purpose. Do NOT "simplify" them back into one
        # shared `activation`, and do NOT make them default to the same value
        # "so the decoder agrees with itself" -- reference SAM deliberately
        # makes them differ: `MaskDecoder.activation` defaults to `nn.GELU` and
        # is passed ONLY to `output_upscaling`, while the hypernetwork/IoU heads
        # hardcode `F.relu` inside `MLP`. One shared knob cannot be correct for
        # both halves; routing a single value to both is what made this package
        # trade a wrong-MLP deviation for a wrong-upscaler one. See D-024.
        self.activation = deserialize_activation(activation)
        self.mlp_activation = deserialize_activation(mlp_activation)

        # Calculate number of mask tokens (multi-mask outputs + 1 single output)
        self.num_mask_tokens = num_multimask_outputs + 1

        # CREATE all sub-layers in __init__

        # Learnable output tokens
        self.iou_token = layers.Embedding(1, transformer_dim, name="iou_token")
        self.mask_tokens = layers.Embedding(
            self.num_mask_tokens,
            transformer_dim,
            name="mask_tokens"
        )

        # Output upscaling network: 4x upsampling (2x -> 2x)
        # Input: (B, H, W, transformer_dim) -> Output: (B, H*4, W*4, transformer_dim//8)
        self.output_upscaling = keras.Sequential([
            keras.layers.Conv2DTranspose(
                transformer_dim // 4,
                kernel_size=2,
                strides=2,
                name="upsample_conv1"
            ),
            create_normalization_layer(normalization_type, name="upsample_norm1"),
            keras.layers.Activation(activation, name="upsample_act1"),
            keras.layers.Conv2DTranspose(
                transformer_dim // 8,
                kernel_size=2,
                strides=2,
                name="upsample_conv2"
            ),
            keras.layers.Activation(activation, name="upsample_act2"),
        ], name="output_upscaling")

        # Hypernetwork MLPs: one for each mask token.
        # Each MLP transforms a mask token embedding into the parameters of the
        # final dynamic convolution. Reference SAM:
        #   MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        # i.e. (iou_head_depth - 1) hidden layers at transformer_dim, then one
        # linear layer at transformer_dim // 8.
        #
        # DECISION plan-2026-08-03T191222-1d751f81/D-010
        # Do NOT hardcode 3 here (nor 2, which is what shipped before). Reference
        # SAM hardcodes 3 for the hypernetwork and uses iou_head_depth only for
        # the IoU head; this package instead drives BOTH from iou_head_depth, so
        # the knob is observably live and the two heads cannot silently diverge.
        # A hardcoded depth passes every count/shape assertion at the default and
        # is indistinguishable from a dead knob -- which is exactly the defect
        # (F-5) this replaced. See decisions.md D-010.
        self.output_hypernetworks_mlps = []
        for i in range(self.num_mask_tokens):
            self.output_hypernetworks_mlps.append(
                _build_mlp_head(
                    num_layers=self.iou_head_depth,
                    hidden_dim=transformer_dim,
                    output_dim=transformer_dim // 8,
                    activation=mlp_activation,
                    dense_name_template=f"hyper_dense{{n}}_{i}",
                    name=f"hypernetwork_mlp_{i}",
                )
            )

        # IoU prediction head
        # Predicts a mask quality score for each mask token. Reference SAM:
        #   MLP(transformer_dim, iou_head_hidden_dim, num_mask_tokens, iou_head_depth)
        self.iou_prediction_head = _build_mlp_head(
            num_layers=self.iou_head_depth,
            hidden_dim=self.iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            activation=mlp_activation,
            dense_name_template="iou_dense{n}",
            name="iou_prediction_head",
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Builds all sub-layers.

        Following the "Create vs. Build" principle, we explicitly build all
        sub-layers to ensure their weights are created before the model attempts
        to load any saved weights during deserialization.

        Args:
            input_shape: Optional shape tuple (not used for this layer).
        """
        # Build embedding layers
        self.iou_token.build((None,))
        self.mask_tokens.build((None,))

        # The transformer is deliberately NOT built here: it builds its own
        # sublayers lazily on the first call, from the actual query/key shapes.
        # (A no-op `if hasattr(self.transformer, 'build'): pass` used to sit
        # here and read as if it did something.)

        # Build upscaling network
        # Input shape: (batch, H, W, transformer_dim)
        self.output_upscaling.build((None, None, None, self.transformer_dim))

        # Build hypernetwork MLPs
        # Input shape: (batch, transformer_dim)
        for mlp in self.output_hypernetworks_mlps:
            mlp.build((None, self.transformer_dim))

        # Build IoU prediction head
        # Input shape: (batch, transformer_dim)
        self.iou_prediction_head.build((None, self.transformer_dim))

        super().build(input_shape)

    def call(
        self,
        image_embeddings: keras.KerasTensor,
        image_pe: keras.KerasTensor,
        sparse_prompt_embeddings: keras.KerasTensor,
        dense_prompt_embeddings: keras.KerasTensor,
        multimask_output: bool,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass for mask prediction.

        Args:
            image_embeddings: Image features from encoder, shape
                (batch_size, H, W, transformer_dim).
            image_pe: Positional encoding for image features, shape
                (batch_size, H, W, transformer_dim).
            sparse_prompt_embeddings: Encoded sparse prompts (points/boxes), shape
                (batch_size, num_sparse, transformer_dim).
            dense_prompt_embeddings: Encoded dense prompts (masks), shape
                (batch_size, H, W, transformer_dim).
            multimask_output: Boolean, if True returns num_multimask_outputs masks,
                if False returns single best mask.
            training: Optional boolean for training mode.

        Returns:
            Tuple of (masks, iou_predictions):
            - masks: Shape (batch_size, num_masks, H*4, W*4)
            - iou_predictions: Shape (batch_size, num_masks)
        """
        # Predict all masks
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            training=training
        )

        # Select output masks based on mode
        if multimask_output:
            # Return multiple mask predictions (skip the single-mask output)
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            # Return only the single best mask
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: keras.KerasTensor,
        image_pe: keras.KerasTensor,
        sparse_prompt_embeddings: keras.KerasTensor,
        dense_prompt_embeddings: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Generate mask predictions and IoU estimates.

        This is the core prediction logic that:
        1. Prepares output tokens (IoU + mask tokens)
        2. Runs two-way transformer to refine tokens and image features
        3. Upscales image features
        4. Generates masks via hypernetwork dynamic convolution
        5. Predicts IoU scores

        Args:
            image_embeddings: Image features, shape (batch_size, H, W, transformer_dim).
            image_pe: Positional encoding, shape (batch_size, H, W, transformer_dim).
            sparse_prompt_embeddings: Sparse prompts, shape (batch_size, num_sparse, transformer_dim).
            dense_prompt_embeddings: Dense prompts, shape (batch_size, H, W, transformer_dim).
            training: Optional boolean for training mode.

        Returns:
            Tuple of (masks, iou_predictions):
            - masks: All mask logits, shape (batch_size, num_mask_tokens, H*4, W*4)
            - iou_predictions: Quality scores, shape (batch_size, num_mask_tokens)

        Raises:
            ValueError: If `sparse_prompt_embeddings`'s batch dimension is
                neither 1 (a shared prompt set) nor exactly the image batch
                size (one prompt set per image). Any other value is either an
                impossible tile or an order-scrambling one.
        """
        # Concatenate IoU token and mask tokens: shape (num_mask_tokens + 1, transformer_dim)
        output_tokens = ops.concatenate(
            [self.iou_token.weights[0], self.mask_tokens.weights[0]],
            axis=0
        )
        # Expand and broadcast to batch size: (batch_size, num_mask_tokens + 1, transformer_dim)
        output_tokens = ops.expand_dims(output_tokens, 0)
        # Derive the batch size from image_embeddings, NOT sparse_prompt_embeddings.
        # The prompt encoder returns a batch-1 sparse tensor when prompts are
        # absent/shared while the image stack is batched (B>1); reading the batch
        # from the prompts made output_tokens (1, ...) mismatch the (B, ...) image
        # keys and crashed the two-way transformer cross-attention.
        batch_size = ops.shape(image_embeddings)[0]
        output_tokens = ops.broadcast_to(
            output_tokens,
            (batch_size, ops.shape(output_tokens)[1], ops.shape(output_tokens)[2])
        )

        # Tile sparse prompt embeddings up to the image batch size so the
        # concat (and downstream cross-attention) batch dims agree.
        sparse_batch = ops.shape(sparse_prompt_embeddings)[0]

        # DECISION plan-2026-08-03T191222-1d751f81/D-015: the tile factor below
        # is INTEGER DIVISION, and only two sparse batch sizes are meaningful:
        # 1 (a shared prompt broadcast over the image batch) and `batch_size`
        # (one prompt set per image). Everything else is silently wrong rather
        # than merely unsupported. Measured: B=1 with 3 prompt rows gives a tile
        # factor of 0 and an opaque `InvalidArgumentError`; B=4 with 2 prompt
        # sets tiles to [a, b, a, b] -- NOT [a, a, b, b] -- so every image is
        # scored against the wrong prompt with no error at all. Do NOT "fix"
        # this by switching the tile to `ops.repeat` (that would make the
        # B=4/2 case silently *plausible* instead of refused, and the correct
        # per-image pairing is the caller's contract, not a guess this layer
        # gets to make), and do NOT ceil/clamp the factor.
        # The check reads STATIC shapes so it is a plain Python branch: under
        # tracing with an unknown batch dim it is skipped rather than traced.
        static_batch = image_embeddings.shape[0]
        static_sparse = sparse_prompt_embeddings.shape[0]
        if (
            static_batch is not None
            and static_sparse is not None
            and static_sparse not in (1, static_batch)
        ):
            raise ValueError(
                f"MaskDecoder cannot tile {static_sparse} sparse prompt rows "
                f"onto an image batch of {static_batch}: sparse_batch must be "
                f"1 (one prompt set shared by every image) or exactly "
                f"batch_size={static_batch} (one prompt set per image). Got "
                f"sparse_prompt_embeddings batch {static_sparse} vs "
                f"image_embeddings batch {static_batch}. Tile or split the "
                f"prompts before calling the decoder."
            )

        sparse_prompt_embeddings = ops.tile(
            sparse_prompt_embeddings,
            [batch_size // sparse_batch, 1, 1]
        )

        # Concatenate output tokens with sparse prompt embeddings
        # Shape: (batch_size, num_mask_tokens + 1 + num_sparse, transformer_dim)
        tokens = ops.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)

        # Prepare source (image) input for transformer
        # Add dense prompt embeddings to image embeddings
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        B, H, W, C = ops.shape(src)

        # Run two-way transformer
        # Returns refined tokens and refined image features
        hs, src_out = self.transformer(src, pos_src, tokens, training=training)

        # Extract specific tokens
        iou_token_out = hs[:, 0, :]  # IoU token: (batch_size, transformer_dim)
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # Mask tokens: (batch_size, num_mask_tokens, transformer_dim)

        # Reshape and upscale image features
        src_out = ops.reshape(src_out, (B, H, W, C))
        upscaled_embedding = self.output_upscaling(src_out, training=training)  # (B, H*4, W*4, C//8)

        # Generate mask-specific parameters using hypernetwork MLPs
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :], training=training)
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)  # (batch_size, num_mask_tokens, C//8)

        # Generate masks via dynamic convolution (matrix multiplication)
        # Flatten upscaled embeddings spatially
        B, H_up, W_up, C_up = ops.shape(upscaled_embedding)
        upscaled_embedding_flat = ops.reshape(
            upscaled_embedding,
            (B, H_up * W_up, C_up)
        )  # (batch_size, H*W, C//8)

        # Matrix multiply: (batch_size, num_mask_tokens, C//8) @ (batch_size, C//8, H*W)
        #                -> (batch_size, num_mask_tokens, H*W)
        masks = hyper_in @ ops.transpose(upscaled_embedding_flat, (0, 2, 1))
        masks = ops.reshape(masks, (B, self.num_mask_tokens, H_up, W_up))

        # Predict IoU scores for each mask
        iou_pred = self.iou_prediction_head(iou_token_out, training=training)

        return masks, iou_pred

    def compute_output_shape(
        self,
        input_shape: Optional[Tuple[Optional[int], ...]] = None
    ) -> Tuple[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]],
               Tuple[Optional[int], Optional[int]]]:
        """
        Compute output shapes for masks and IoU predictions.

        Args:
            input_shape: Not used for this layer.

        Returns:
            Tuple of (mask_shape, iou_shape):
            - mask_shape: (batch_size, num_mask_tokens, H*4, W*4)
            - iou_shape: (batch_size, num_mask_tokens)
        """
        mask_shape = (None, self.num_mask_tokens, None, None)
        iou_shape = (None, self.num_mask_tokens)
        return mask_shape, iou_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Note: The transformer is handled separately by Keras serialization
        since it's a layer passed as a parameter.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "transformer_dim": self.transformer_dim,
            "num_multimask_outputs": self.num_multimask_outputs,
            "iou_head_depth": self.iou_head_depth,
            "iou_head_hidden_dim": self.iou_head_hidden_dim,
            "normalization_type": self.normalization_type,
            "activation": serialize_activation(self.activation),
            "mlp_activation": serialize_activation(self.mlp_activation),
            "transformer": keras.layers.serialize(self.transformer),
        })
        # Note: self.transformer is passed in __init__ as a layer,
        # so Keras automatically handles its serialization
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskDecoder":
        """Creates a MaskDecoder from its config."""
        config["transformer"] = keras.layers.deserialize(config.pop("transformer"))
        return cls(**config)