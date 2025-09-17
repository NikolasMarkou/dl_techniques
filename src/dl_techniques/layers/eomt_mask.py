"""Generate instance segmentation predictions from transformer query tokens.

This layer implements the prediction heads for a query-based segmentation
architecture, translating learned object queries into class labels and spatial
masks. It is designed as the final component in models like the Encoder-only
Mask Transformer (EoMT), bridging the gap between the transformer's abstract
feature representations and the concrete, pixel-level segmentation task.

The core design philosophy is to decouple the tasks of classification ("what")
and localization ("where") into two parallel prediction heads, both operating
on the same set of input query tokens. Each query represents a hypothesis for
a single object instance in the image.

Architectural and Mathematical Foundations:
The module takes two inputs: a set of N query tokens and a dense, pixel-level
feature map from an image encoder (e.g., a Vision Transformer).

1.  **Classification Head**: Each of the N query tokens, which is a vector
    `q` in R^D, is passed through a simple linear layer. This produces a
    logit vector for C classes, directly predicting the object category for
    that query.
        `class_logits = Linear(q)`

2.  **Mask Head**: This head generates the spatial mask and is the more
    intricate component. The process involves two steps:
    a.  First, each query token `q` is transformed by a multi-layer
        perceptron (MLP) into a specialized "mask embedding" `m` in R^D'.
        This MLP allows the model to learn a representation of the query
        that is optimized for the spatial localization task.
    b.  Second, the final mask is produced by computing the similarity between
        this single mask embedding `m` and the feature vector of *every*
        pixel in the encoder's output feature map. This similarity is
        calculated via a dot product. For a pixel feature map `P` of shape
        (H, W, D'), the mask logit at position (i, j) is:
        `mask_logit[i, j] = m @ P[i, j]^T`

The intuition behind this operation is that the mask embedding `m` acts as a
learned "template" or "filter" for a specific object instance. The dot
product effectively "slides" this template across the entire image feature map,
producing a high activation at spatial locations whose features are similar
to the object's learned representation. This yields a per-query segmentation
mask of shape (H, W), localizing the object instance in the image.

This query-based, dual-head approach is foundational to many modern object
detection and segmentation models, enabling end-to-end training without the
need for hand-crafted components like anchor boxes or non-maximum suppression.

References:
    - Carion et al. "End-to-End Object Detection with Transformers" (DETR).
      The pioneering work that introduced the object query and prediction head
      architecture for object detection.
      https://arxiv.org/abs/2005.12872

    - Li et al. "Your ViT is Secretly a Segmentation Model". This work
      adapts and applies query-based mechanisms specifically for segmentation
      tasks with Vision Transformer backbones, providing the direct context
      for this module.
      https://arxiv.org/abs/2312.02113
"""

import keras
from typing import Optional, Any, Tuple, Union, Dict
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class EomtMask(keras.layers.Layer):
    """
    Mask prediction module for Encoder-only Mask Transformer (EoMT).

    This module processes query tokens and pixel-level features to generate both
    class predictions and spatial mask predictions for instance segmentation.
    It implements the dual-head architecture where each query predicts both
    what class it represents and where that class appears spatially.

    **Intent**: Provide the prediction heads for EoMT that convert learned query
    representations into actionable segmentation outputs, bridging between the
    transformer's learned representations and the final task requirements.

    **Architecture**:
    ```
    Query Tokens [batch, num_queries, embed_dim]
           ↓
    ┌─────────────────┬─────────────────┐
    │   Class Head    │   Mask Head     │
    │                 │                 │
    │  Dense(classes) │  MLP(mask_dim)  │
    │       ↓         │       ↓         │
    │ Class Logits    │ Mask Embeddings │
    └─────────────────┴─────────────────┘
                              ↓
    Pixel Features [batch, H, W, embed_dim]
                              ↓
              Dot Product: embeddings @ pixels^T
                              ↓
                   Mask Logits [batch, queries, H, W]

    Final Output: (Class Logits, Mask Logits)
    ```

    **Mathematical Operations**:
    1. **Class Prediction**: cls_logits = Linear(query_tokens)
    2. **Mask Embedding**: mask_emb = MLP(query_tokens)
    3. **Mask Prediction**: mask_logits = mask_emb @ pixel_features^T

    The mask prediction uses learned similarity between query embeddings and
    pixel-level features to determine spatial extent of each predicted class.

    Args:
        num_classes: Integer, number of classes to predict. Must be positive.
            This determines the size of the classification output.
        hidden_dim: Integer, hidden dimension for the mask prediction MLP.
            Controls the capacity of the mask embedding network. Defaults to 256.
        mask_dim: Integer, final dimension of mask embeddings before dot product
            with pixel features. Should match pixel feature dimension for optimal
            performance. Defaults to 256.
        kernel_initializer: String or initializer, weight initialization for
            all dense layers. Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, bias initialization for
            all dense layers. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for dense layer weights.
        bias_regularizer: Optional regularizer for dense layer biases.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tuple of two tensors:
        - query_tokens: `(batch_size, num_queries, embed_dim)`
        - pixel_features: `(batch_size, height, width, embed_dim)`

    Output shape:
        Tuple of two tensors:
        - class_logits: `(batch_size, num_queries, num_classes)`
        - mask_logits: `(batch_size, num_queries, height, width)`

    Attributes:
        class_head: Dense layer for class prediction.
        mask_mlp: Sequential MLP for mask embedding generation.

    Example:
        ```python
        # Basic usage for segmentation
        module = MaskModule(num_classes=80, hidden_dim=256, mask_dim=256)

        # Prepare inputs
        query_tokens = keras.Input(shape=(100, 768))  # 100 queries, 768 dim
        pixel_features = keras.Input(shape=(14, 14, 768))  # 14x14 pixel grid

        # Get predictions
        class_logits, mask_logits = module([query_tokens, pixel_features])

        # class_logits: (batch, 100, 80)
        # mask_logits: (batch, 100, 14, 14)

        # Custom configuration
        module = MaskModule(
            num_classes=21,  # PASCAL VOC classes
            hidden_dim=512,  # Larger capacity
            mask_dim=512,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a complete model
        queries = keras.Input(shape=(100, 768))
        pixels = keras.Input(shape=(16, 16, 768))

        cls_pred, mask_pred = module([queries, pixels])

        model = keras.Model([queries, pixels], [cls_pred, mask_pred])
        model.compile(
            optimizer='adamw',
            loss={
                'class_logits': 'sparse_categorical_crossentropy',
                'mask_logits': 'binary_crossentropy'
            }
        )
        ```

    References:
        - "Your ViT is Secretly a Segmentation Model": https://arxiv.org/abs/2312.02113
        - "DETR: End-to-End Object Detection": https://arxiv.org/abs/2005.12872

    Note:
        The mask_dim should typically match the embed_dim of pixel_features for
        optimal dot product computation. The mask prediction creates a similarity
        map between each query and all spatial locations in the image.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mask_dim <= 0:
            raise ValueError(f"mask_dim must be positive, got {mask_dim}")

        # Store configuration
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # Class prediction head
        self.class_head = layers.Dense(
            self.num_classes,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="class_head"
        )

        # Mask embedding MLP
        self.mask_mlp = keras.Sequential([
            layers.Dense(
                self.hidden_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mask_mlp_1"
            ),
            layers.Dense(
                self.hidden_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mask_mlp_2"
            ),
            layers.Dense(
                self.mask_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mask_mlp_3"
            )
        ], name="mask_mlp")

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer and all its sub-layers.

        Args:
            input_shape: Tuple of (query_shape, pixel_shape)
        """
        query_shape, pixel_shape = input_shape

        # Build class head with query shape
        self.class_head.build(query_shape)

        # Build mask MLP with query shape
        self.mask_mlp.build(query_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass through the mask module.

        Args:
            inputs: Tuple of (query_tokens, pixel_features)
                - query_tokens: [batch, num_queries, embed_dim]
                - pixel_features: [batch, H, W, embed_dim]
            training: Boolean indicating training mode

        Returns:
            Tuple of (class_logits, mask_logits)
            - class_logits: [batch, num_queries, num_classes]
            - mask_logits: [batch, num_queries, H, W]
        """
        query_tokens, pixel_features = inputs

        # Get shapes
        batch_size = ops.shape(query_tokens)[0]
        num_queries = ops.shape(query_tokens)[1]
        height = ops.shape(pixel_features)[1]
        width = ops.shape(pixel_features)[2]
        embed_dim = ops.shape(pixel_features)[3]

        # Predict class logits
        class_logits = self.class_head(query_tokens, training=training)

        # Predict mask embeddings
        mask_embeddings = self.mask_mlp(query_tokens, training=training)  # [batch, num_queries, mask_dim]

        # Compute mask logits via dot product with pixel features
        # Reshape pixel features for efficient computation
        pixel_features_flat = ops.reshape(
            pixel_features, [batch_size, height * width, embed_dim]
        )  # [batch, H*W, embed_dim]

        # Efficient dot product: [batch, queries, mask_dim] @ [batch, mask_dim, H*W]
        # First transpose pixel features: [batch, H*W, embed_dim] -> [batch, embed_dim, H*W]
        pixel_features_transposed = ops.transpose(pixel_features_flat, [0, 2, 1])

        # Compute dot product: [batch, queries, mask_dim] @ [batch, mask_dim, H*W] -> [batch, queries, H*W]
        mask_logits_flat = ops.matmul(mask_embeddings, pixel_features_transposed)

        # Reshape back to spatial dimensions
        mask_logits = ops.reshape(
            mask_logits_flat, [batch_size, num_queries, height, width]
        )

        return class_logits, mask_logits

    def compute_output_shape(
        self,
        input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes."""
        query_shape, pixel_shape = input_shape
        batch_size = query_shape[0]
        num_queries = query_shape[1]
        height, width = pixel_shape[1], pixel_shape[2]

        class_shape = (batch_size, num_queries, self.num_classes)
        mask_shape = (batch_size, num_queries, height, width)

        return class_shape, mask_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "mask_dim": self.mask_dim,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------