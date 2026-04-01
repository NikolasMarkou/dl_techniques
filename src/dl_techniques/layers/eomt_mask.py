"""
Generate instance segmentation predictions from transformer query tokens.

This layer implements the prediction heads for a query-based segmentation
architecture, translating learned object queries into class labels and spatial
masks. Each query represents a hypothesis for a single object instance, and
the module decouples classification ("what") and localization ("where") into
two parallel prediction heads operating on the same query tokens.

The classification head uses a linear layer (with optional MLP) on query tokens
to produce class logits: class_logits = Linear(q). The mask head transforms
queries through an MLP into mask embeddings, then computes dot-product similarity
with every pixel in the encoder's feature map: mask_logit[i,j] = m @ P[i,j]^T,
producing per-query segmentation masks.

References:
    - Carion et al. "End-to-End Object Detection with Transformers" (DETR).
      https://arxiv.org/abs/2005.12872
    - Li et al. "Your ViT is Secretly a Segmentation Model".
      https://arxiv.org/abs/2312.02113
"""


import keras
from typing import Optional, Any, Tuple, Union, Dict, List
from keras import ops, layers, initializers, regularizers, activations, constraints

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class EomtMask(keras.layers.Layer):
    """
    Configurable mask prediction module for Encoder-only Mask Transformer (EoMT).

    Processes query tokens and pixel-level features to generate both class
    predictions and spatial mask predictions for instance segmentation. The class
    head produces logits via cls_logits = ClassHead(norm(query_tokens)), while the
    mask head generates embeddings via mask_emb = MaskMLP(norm(query_tokens)) and
    computes mask logits as mask_logits = mask_emb @ pixel_features^T, optionally
    scaled by a temperature parameter.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────────────┐
        │  Query Tokens [batch, num_queries, embed_dim]      │
        └───────────────────┬────────────────────────────────┘
                            ▼
              ┌─────────────┴─────────────┐
              ▼                           ▼
        ┌───────────────┐         ┌───────────────┐
        │  Class Head   │         │  Mask Head    │
        │  ┌─────────┐  │         │  ┌─────────┐  │
        │  │Norm(opt)│  │         │  │Norm(opt)│  │
        │  └────┬────┘  │         │  └────┬────┘  │
        │       ▼       │         │       ▼       │
        │  ┌─────────┐  │         │  ┌─────────┐  │
        │  │MLP(opt) │  │         │  │MLP Net  │  │
        │  └────┬────┘  │         │  └────┬────┘  │
        │       ▼       │         │       ▼       │
        │  ┌─────────┐  │         │  ┌─────────┐  │
        │  │Dense(C) │  │         │  │Mask Proj│  │
        │  └────┬────┘  │         │  └────┬────┘  │
        └───────┼───────┘         └───────┼───────┘
                ▼                         ▼
        Class Logits          Mask Embeddings
        [B, Q, classes]       [B, Q, mask_dim]
                                      │
                                      ▼
                              ┌───────────────┐
                              │  Dot Product   │◀── Pixel Features
                              │  m @ P^T       │    [B, H, W, D]
                              └───────┬───────┘
                                      ▼
                              Mask Logits
                              [B, Q, H, W]

    :param num_classes: Number of classes to predict. Must be positive.
    :type num_classes: int
    :param hidden_dims: Hidden dimensions for mask MLP layers. If None, defaults
        to [256, 256]. Empty list means direct projection.
    :type hidden_dims: Optional[List[int]]
    :param mask_dim: Final dimension of mask embeddings. Should typically match
        pixel feature dimension. Defaults to 256.
    :type mask_dim: int
    :param class_mlp_dims: Optional hidden dimensions for class MLP.
    :type class_mlp_dims: Optional[List[int]]
    :param use_class_norm: Whether to normalize query tokens before class head.
        Defaults to False.
    :type use_class_norm: bool
    :param use_mask_norm: Whether to normalize query tokens before mask head.
        Defaults to False.
    :type use_mask_norm: bool
    :param normalization_type: Type of normalization from factory. Defaults to
        'layer_norm'.
    :type normalization_type: str
    :param normalization_args: Optional arguments for normalization layers.
    :type normalization_args: Optional[Dict[str, Any]]
    :param mlp_activation: Activation for MLP hidden layers. Defaults to 'relu'.
    :type mlp_activation: Union[str, keras.layers.Activation]
    :param mlp_dropout_rate: Dropout rate for MLP layers. Defaults to 0.0.
    :type mlp_dropout_rate: float
    :param use_bias: Whether to use bias in dense layers. Defaults to True.
    :type use_bias: bool
    :param mask_temperature: Temperature scaling for mask logits. Defaults to 1.0.
    :type mask_temperature: float
    :param learnable_temperature: Whether mask temperature is learnable.
        Defaults to False.
    :type learnable_temperature: bool
    :param class_activation: Optional activation for class logits. Defaults to None.
    :type class_activation: Optional[Union[str, keras.layers.Activation]]
    :param mask_activation: Optional activation for mask logits. Defaults to None.
    :type mask_activation: Optional[Union[str, keras.layers.Activation]]
    :param kernel_initializer: Initializer for dense layer kernels. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for dense layer biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kernel_constraint: Optional constraint for kernel weights.
    :type kernel_constraint: Optional[constraints.Constraint]
    :param bias_constraint: Optional constraint for bias weights.
    :type bias_constraint: Optional[constraints.Constraint]
    :param kwargs: Additional keyword arguments for Layer base class.
    """

    def __init__(
            self,
            num_classes: int,
            hidden_dims: Optional[List[int]] = None,
            mask_dim: int = 256,
            class_mlp_dims: Optional[List[int]] = None,
            use_class_norm: bool = False,
            use_mask_norm: bool = False,
            normalization_type: NormalizationType = 'layer_norm',
            normalization_args: Optional[Dict[str, Any]] = None,
            mlp_activation: Union[str, keras.layers.Activation] = 'relu',
            mlp_dropout_rate: float = 0.0,
            use_bias: bool = True,
            mask_temperature: float = 1.0,
            learnable_temperature: bool = False,
            class_activation: Optional[Union[str, keras.layers.Activation]] = None,
            mask_activation: Optional[Union[str, keras.layers.Activation]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            kernel_constraint: Optional[constraints.Constraint] = None,
            bias_constraint: Optional[constraints.Constraint] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if mask_dim <= 0:
            raise ValueError(f"mask_dim must be positive, got {mask_dim}")
        if not (0.0 <= mlp_dropout_rate <= 1.0):
            raise ValueError(f"mlp_dropout_rate must be between 0 and 1, got {mlp_dropout_rate}")
        if mask_temperature <= 0:
            raise ValueError(f"mask_temperature must be positive, got {mask_temperature}")

        # Store configuration
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 256]
        self.mask_dim = mask_dim
        self.class_mlp_dims = class_mlp_dims or []
        self.use_class_norm = use_class_norm
        self.use_mask_norm = use_mask_norm
        self.normalization_type = normalization_type
        self.normalization_args = normalization_args or {}
        self.mlp_activation = activations.get(mlp_activation)
        self.mlp_dropout_rate = mlp_dropout_rate
        self.use_bias = use_bias
        self.mask_temperature = mask_temperature
        self.learnable_temperature = learnable_temperature
        self.class_activation = activations.get(class_activation) if class_activation else None
        self.mask_activation = activations.get(mask_activation) if mask_activation else None
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # CREATE all sub-layers in __init__

        # Optional normalization layers
        if self.use_class_norm:
            self.class_norm = create_normalization_layer(
                self.normalization_type,
                name="class_norm",
                **self.normalization_args
            )
        else:
            self.class_norm = None

        if self.use_mask_norm:
            self.mask_norm = create_normalization_layer(
                self.normalization_type,
                name="mask_norm",
                **self.normalization_args
            )
        else:
            self.mask_norm = None

        # Class prediction head
        self.class_mlp = self._build_mlp(
            self.class_mlp_dims,
            name_prefix="class_mlp"
        ) if self.class_mlp_dims else None

        self.class_head = layers.Dense(
            self.num_classes,
            activation=self.class_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="class_head"
        )

        # Mask embedding MLP
        self.mask_mlp = self._build_mlp(
            self.hidden_dims,
            name_prefix="mask_mlp"
        ) if self.hidden_dims else None

        # Final mask projection
        self.mask_projection = layers.Dense(
            self.mask_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="mask_projection"
        )

        # Temperature parameter
        if self.learnable_temperature:
            self.temperature = self.add_weight(
                name="temperature",
                shape=(1,),
                initializer=initializers.Constant(self.mask_temperature),
                trainable=True,
                constraint=constraints.NonNeg() if self.mask_temperature > 0 else None
            )
        else:
            self.temperature = self.mask_temperature

    def _build_mlp(
            self,
            dims: List[int],
            name_prefix: str
    ) -> Optional[keras.Sequential]:
        """Build an MLP with specified dimensions.

        :param dims: List of hidden dimensions.
        :type dims: List[int]
        :param name_prefix: Name prefix for layers.
        :type name_prefix: str
        :return: Sequential MLP or None if dims is empty.
        :rtype: Optional[keras.Sequential]
        """
        if not dims:
            return None

        layers_list = []
        for i, dim in enumerate(dims):
            layers_list.append(
                layers.Dense(
                    dim,
                    activation=self.mlp_activation,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name=f"{name_prefix}_{i}"
                )
            )

            if self.mlp_dropout_rate > 0:
                layers_list.append(
                    layers.Dropout(
                        self.mlp_dropout_rate,
                        name=f"{name_prefix}_dropout_{i}"
                    )
                )

        return keras.Sequential(layers_list, name=name_prefix)

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """Build the layer and all its sub-layers.

        :param input_shape: Tuple of (query_shape, pixel_shape).
        :type input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        query_shape, pixel_shape = input_shape

        # Build normalization layers if present
        if self.class_norm is not None:
            self.class_norm.build(query_shape)
        if self.mask_norm is not None:
            self.mask_norm.build(query_shape)

        # Build class head path
        if self.class_mlp is not None:
            self.class_mlp.build(query_shape)
            # Get output shape of class MLP for class head
            class_mlp_output_shape = list(query_shape)
            class_mlp_output_shape[-1] = self.class_mlp_dims[-1]
            self.class_head.build(tuple(class_mlp_output_shape))
        else:
            self.class_head.build(query_shape)

        # Build mask head path
        if self.mask_mlp is not None:
            self.mask_mlp.build(query_shape)
            # Get output shape of mask MLP for projection
            mask_mlp_output_shape = list(query_shape)
            mask_mlp_output_shape[-1] = self.hidden_dims[-1]
            self.mask_projection.build(tuple(mask_mlp_output_shape))
        else:
            self.mask_projection.build(query_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass through the mask module.

        :param inputs: Either tuple of (query_tokens, pixel_features) or dict
            with those keys.
        :type inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], Dict[str, keras.KerasTensor]]
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Tuple of (class_predictions, mask_predictions).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Handle both tuple and dict inputs
        if isinstance(inputs, dict):
            query_tokens = inputs['query_tokens']
            pixel_features = inputs['pixel_features']
        else:
            query_tokens, pixel_features = inputs

        # Get shapes
        batch_size = ops.shape(query_tokens)[0]
        num_queries = ops.shape(query_tokens)[1]
        height = ops.shape(pixel_features)[1]
        width = ops.shape(pixel_features)[2]
        pixel_dim = ops.shape(pixel_features)[3]

        # CLASS PREDICTION PATH
        class_input = query_tokens
        if self.class_norm is not None:
            class_input = self.class_norm(class_input, training=training)

        if self.class_mlp is not None:
            class_input = self.class_mlp(class_input, training=training)

        class_predictions = self.class_head(class_input, training=training)

        # MASK PREDICTION PATH
        mask_input = query_tokens
        if self.mask_norm is not None:
            mask_input = self.mask_norm(mask_input, training=training)

        if self.mask_mlp is not None:
            mask_input = self.mask_mlp(mask_input, training=training)

        mask_embeddings = self.mask_projection(mask_input, training=training)

        # Compute mask logits via dot product
        # Reshape pixel features for efficient computation
        pixel_features_flat = ops.reshape(
            pixel_features, [batch_size, height * width, pixel_dim]
        )

        # Transpose for matrix multiplication
        pixel_features_transposed = ops.transpose(pixel_features_flat, [0, 2, 1])

        # Dot product: [batch, queries, mask_dim] @ [batch, mask_dim, H*W]
        mask_logits_flat = ops.matmul(mask_embeddings, pixel_features_transposed)

        # Apply temperature scaling
        if self.learnable_temperature or self.mask_temperature != 1.0:
            temperature = self.temperature if self.learnable_temperature else self.mask_temperature
            mask_logits_flat = mask_logits_flat / temperature

        # Reshape to spatial dimensions
        mask_predictions = ops.reshape(
            mask_logits_flat, [batch_size, num_queries, height, width]
        )

        # Apply mask activation if specified
        if self.mask_activation is not None:
            mask_predictions = self.mask_activation(mask_predictions)

        return class_predictions, mask_predictions

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]],
            Dict[str, Tuple[Optional[int], ...]]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes.

        :param input_shape: Input shapes as tuple or dict.
        :type input_shape: Union[Tuple, Dict]
        :return: Tuple of (class_shape, mask_shape).
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        if isinstance(input_shape, dict):
            query_shape = input_shape['query_tokens']
            pixel_shape = input_shape['pixel_features']
        else:
            query_shape, pixel_shape = input_shape

        batch_size = query_shape[0]
        num_queries = query_shape[1]
        height, width = pixel_shape[1], pixel_shape[2]

        class_shape = (batch_size, num_queries, self.num_classes)
        mask_shape = (batch_size, num_queries, height, width)

        return class_shape, mask_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'hidden_dims': self.hidden_dims,
            'mask_dim': self.mask_dim,
            'class_mlp_dims': self.class_mlp_dims,
            'use_class_norm': self.use_class_norm,
            'use_mask_norm': self.use_mask_norm,
            'normalization_type': self.normalization_type,
            'normalization_args': self.normalization_args,
            'mlp_activation': activations.serialize(self.mlp_activation),
            'mlp_dropout_rate': self.mlp_dropout_rate,
            'use_bias': self.use_bias,
            'mask_temperature': self.mask_temperature,
            'learnable_temperature': self.learnable_temperature,
            'class_activation': activations.serialize(self.class_activation) if self.class_activation else None,
            'mask_activation': activations.serialize(self.mask_activation) if self.mask_activation else None,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config
