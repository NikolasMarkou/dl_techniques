"""
The mask decoder, built by :class:`MaskDecoder`: output tokens, a two-way
transformer, and a hypernetwork head.

It consumes the image embedding and the prompt embeddings and emits
low-resolution mask logits plus one predicted IoU per mask. The masks are
not produced by a convolutional head: a per-mask MLP emits a weight vector
that is dotted against the upscaled feature map, so the head is a
hypernetwork rather than a decoder in the usual convolutional sense.

``activation`` and ``mlp_activation`` are separate knobs, matching reference
SAM: ``'gelu'`` reaches only the output-upscaling convolutions, ``'relu'``
reaches every non-final layer of the hypernetwork and IoU heads. Collapsing
them into one shared activation makes one of the two halves wrong.
``sparse_prompt_embeddings`` must carry a batch of 1 or exactly the image
batch size; any other value raises rather than silently tiling into a
scrambled pairing.

References:
    - Kirillov et al., 2023. Segment Anything. (https://arxiv.org/abs/2304.02643)
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
from dl_techniques.utils.keras_registration import register_dl_technique

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
    Build reference SAM's MLP head: ``num_layers`` Dense layers, last one
    linear, shared by the hypernetwork MLPs and the IoU head.

    :param num_layers: Total number of Dense layers -- ``N - 1`` hidden
        layers of width ``hidden_dim`` followed by one ``output_dim`` layer.
        Must be at least 1; callers validate.
    :type num_layers: int
    :param hidden_dim: Width of every layer except the last.
    :type hidden_dim: int
    :param output_dim: Width of the last layer.
    :type output_dim: int
    :param activation: Activation applied to every layer except the last,
        which is always linear so the head can emit signed logits.
    :type activation: str
    :param dense_name_template: Naming pattern for the sub-layers, with one
        ``{n}`` placeholder for the 1-based layer index, e.g.
        ``"iou_dense{n}"``.
    :type dense_name_template: str
    :param name: Name of the returned ``Sequential``.
    :type name: str
    :return: An unbuilt :class:`keras.Sequential` of ``num_layers`` Dense
        layers.
    :rtype: keras.Sequential
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


@register_dl_technique("dl_techniques.models.sam1.mask_decoder")
class MaskDecoder(keras.layers.Layer):
    """
    Predict segmentation masks and per-mask IoU scores from image and prompt
    embeddings.

    Architecture:

    .. code-block:: text

        image_embeddings + dense_prompt_embeddings -> src [B, H, W, C]
        [iou_token, mask_tokens, sparse_prompts] -> tokens [B, 1+N+M, C]
                       │
                       ▼
              TwoWayTransformer
                       │
              ┌────────┴────────┐
              ▼                 ▼
        iou_token_out     mask_tokens_out, src_out
              │                 │
              ▼                 ▼
        iou_prediction_head  output_upscaling (4x)
              │                 │
              ▼                 ▼
        iou [B, N]        hypernetwork_mlps -> hyper_in [B, N, C/8]
                                 │
                                 ▼
                    hyper_in @ upscaled_flat -> masks [B, N, 4H, 4W]

    :param transformer_dim: Embedding dimension used by the transformer.
        Must match the dimension of input embeddings.
    :type transformer_dim: int
    :param transformer: The two-way transformer that jointly refines prompt
        and image embeddings.
    :type transformer: TwoWayTransformer
    :param num_multimask_outputs: Number of mask predictions beyond the
        single output mask. Defaults to 3; total masks =
        ``num_multimask_outputs + 1``.
    :type num_multimask_outputs: int
    :param iou_head_depth: Total Dense layers in each MLP head -- the IoU
        head and every hypernetwork MLP. Must be positive. Defaults to 3.
    :type iou_head_depth: int
    :param iou_head_hidden_dim: Hidden dimension of the IoU head. Defaults
        to 256.
    :type iou_head_hidden_dim: int
    :param normalization_type: Normalization used in the upscaling module.
        Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param activation: Activation inside ``output_upscaling`` only. Defaults
        to ``'gelu'``.
    :type activation: str
    :param mlp_activation: Activation applied to every non-final layer of
        the hypernetwork MLPs and the IoU head. Defaults to ``'relu'``.
    :type mlp_activation: str
    :param kwargs: Additional arguments for the Layer base class.
    :ivar iou_token: Embedding for the IoU prediction token.
    :ivar mask_tokens: Embedding for the mask prediction tokens.
    :ivar output_upscaling: Sequential model that upsamples image features 4x.
    :ivar output_hypernetworks_mlps: One MLP head per mask token.
    :ivar iou_prediction_head: MLP predicting mask quality.

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

    Example:
        ```python
        from .transformer import TwoWayTransformer

        transformer = TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8)
        decoder = MaskDecoder(transformer_dim=256, transformer=transformer)

        masks, iou_pred = decoder(
            image_embeddings=image_emb,
            image_pe=pos_encoding,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=True
        )
        ```
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
        # DECISION plan-2026-08-03T191222-1d751f81/D-024: activation and
        # mlp_activation stay separate knobs, matching reference SAM's differing
        # defaults for output_upscaling vs the hypernetwork/IoU heads. See decisions.md.
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

        # Hypernetwork MLPs: one per mask token, each mapping a mask token
        # embedding to the parameters of the final dynamic convolution.
        # DECISION plan-2026-08-03T191222-1d751f81/D-010: depth comes from
        # iou_head_depth here, not a hardcoded 3 -- a hardcoded depth would be
        # indistinguishable from a dead knob at the default. See decisions.md.
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
        Build every sub-layer explicitly so weights exist before
        deserialization restores them.

        :param input_shape: Not used by this layer.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
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
        Predict masks and select the single-mask or multi-mask output.

        :param image_embeddings: Image features, shape
            ``(batch_size, H, W, transformer_dim)``.
        :param image_pe: Positional encoding, same shape as
            ``image_embeddings``.
        :param sparse_prompt_embeddings: Encoded sparse prompts, shape
            ``(batch_size, num_sparse, transformer_dim)``.
        :param dense_prompt_embeddings: Encoded dense prompts, same shape as
            ``image_embeddings``.
        :param multimask_output: If True, return ``num_multimask_outputs``
            masks; if False, return the single best mask.
        :type multimask_output: bool
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]
        :return: ``(masks, iou_predictions)`` -- masks of shape
            ``(batch_size, num_masks, H*4, W*4)`` and IoU scores of shape
            ``(batch_size, num_masks)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
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
        Run the token preparation, two-way transformer, upscaling and
        hypernetwork stages that produce every mask and IoU score.

        :param image_embeddings: Image features, shape
            ``(batch_size, H, W, transformer_dim)``.
        :param image_pe: Positional encoding, same shape as
            ``image_embeddings``.
        :param sparse_prompt_embeddings: Sparse prompts, shape
            ``(batch_size, num_sparse, transformer_dim)``.
        :param dense_prompt_embeddings: Dense prompts, same shape as
            ``image_embeddings``.
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]
        :return: ``(masks, iou_predictions)`` -- all mask logits, shape
            ``(batch_size, num_mask_tokens, H*4, W*4)``, and quality scores,
            shape ``(batch_size, num_mask_tokens)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        :raises ValueError: If ``sparse_prompt_embeddings``'s batch dimension
            is neither 1 (a shared prompt set) nor exactly the image batch
            size (one prompt set per image).
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

        # DECISION plan-2026-08-03T191222-1d751f81/D-015: only sparse batch
        # sizes of 1 or batch_size are valid for the tile below; anything else
        # tiles into a scrambled [a, b, a, b] pairing with no error. See decisions.md.
        # Reads static shapes, so this is a plain Python branch skipped under tracing.
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

        # iou_token_out is (batch_size, transformer_dim); mask_tokens_out is
        # (batch_size, num_mask_tokens, transformer_dim).
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # upscaled_embedding is (B, H*4, W*4, C//8).
        src_out = ops.reshape(src_out, (B, H, W, C))
        upscaled_embedding = self.output_upscaling(src_out, training=training)

        # hyper_in is (batch_size, num_mask_tokens, C//8).
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :], training=training)
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)

        # Flatten upscaled embeddings spatially to (batch_size, H*W, C//8),
        # then matmul with hyper_in to get (batch_size, num_mask_tokens, H*W).
        B, H_up, W_up, C_up = ops.shape(upscaled_embedding)
        upscaled_embedding_flat = ops.reshape(
            upscaled_embedding,
            (B, H_up * W_up, C_up)
        )
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

        :param input_shape: Not used by this layer.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(mask_shape, iou_shape)`` -- masks of shape
            ``(batch_size, num_mask_tokens, H*4, W*4)`` and IoU scores of
            shape ``(batch_size, num_mask_tokens)``.
        :rtype: Tuple[tuple, tuple]
        """
        mask_shape = (None, self.num_mask_tokens, None, None)
        iou_shape = (None, self.num_mask_tokens)
        return mask_shape, iou_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration, including the serialized
        transformer sub-layer.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
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
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskDecoder":
        """
        Build a :class:`MaskDecoder` from a config dict, deserializing the
        transformer sub-layer.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`MaskDecoder` instance.
        :rtype: MaskDecoder
        """
        config["transformer"] = keras.layers.deserialize(config.pop("transformer"))
        return cls(**config)