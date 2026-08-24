"""
Contrastive language-image pretraining over two modernized transformer towers
(grouped-query attention, RMSNorm, SwiGLU, rotary position embeddings).

CLIP addresses a supervision problem rather than an architectural one. Labelled
image datasets are small and their label sets are closed, so a classifier
trained on one can only ever name the categories somebody enumerated in
advance. Image-caption pairs are abundant and their supervision is open-ended,
but a caption is not a label: predicting it token by token is expensive, and two
captions of the same picture rarely agree word for word. The contrastive
formulation extracts a usable signal anyway by asking a much weaker question --
which caption in this batch belongs to which image. Both modalities are mapped
into one embedding space, each feature is L2-normalized so an inner product is a
cosine, and a batch of `N` pairs produces an `N x N` similarity matrix whose
diagonal must dominate both its row and its column:

`S = tau * f_I(I) @ f_T(T)^T`

The training objective is the symmetric cross-entropy over `S`'s rows and its
columns. It scales because the batch supplies its own negatives -- the `N^2 - N`
mismatched pairs cost nothing to construct -- and it yields zero-shot
classification for free, since any set of class names can be encoded as text and
used directly as a classifier weight matrix. The loss itself is not implemented
here; this module produces the two logits matrices and the temperature, and the
loss lives in `dl_techniques.losses`.

The two towers never see each other. There is no cross-attention, no shared
trunk and no fusion layer: the only place the modalities meet is the final
matmul, and the only thing that forces them into agreement is the contrastive
gradient. Each tower therefore ends in its own bias-free `Dense(embed_dim)`
projection out of its native width (768 for vision, 512 for text at base scale)
into the shared space, followed by L2 normalization. Keeping the projection last
and the normalization after it is what makes the dot product a cosine and the
temperature the sole scale in the logits.

The vision tower is a strided convolution patch embedding, a learnable CLS token
prepended to the patch sequence, `vision_layers` transformer blocks, and a read
of position 0. The text tower is a token embedding, `text_layers` blocks, and a
read of the last non-padding position. Neither tower carries a learned
positional table: position enters only as RoPE inside the grouped-query
attention, rotating queries and keys by an angle proportional to index. For the
image side this means patches are positioned along the flattened raster order
with the CLS token at index 0, which is a real departure from CLIP's learned
positional embedding and worth knowing before comparing numbers.

Two pooling details are easy to get wrong. The CLS token is a single
`(1, 1, vision_width)` weight broadcast across the batch, so every image starts
from the same query vector and the attention blocks are what make its final
state image-specific; it is created in `build()` alongside the temperature rather
than in `__init__`. On the text side the pooled position is found by
`last_non_pad_token`, which counts non-pad tokens and reads index `count - 1`.
That assumes right padding and pad id `0` -- the id is hard-coded at this call
site. Left-padded batches, or a tokenizer whose pad id is not zero, will pool the
wrong position silently rather than raising. Where the tower still differs from
the reference implementation is *which* position it pools: OpenAI CLIP locates
the EOT token as the argmax of the token ids, this one counts non-pad tokens.
The masking now agrees -- `encode_text` builds a lower-triangular causal mask and
passes it to every text block, so the pooled last real token is the only
position that has read the whole sentence, which is what makes last-token
pooling meaningful. The mask is constructed in the masking factory's *block*
semantics and inverted once to the *attend* semantics the attention layers
expect, and it is broadcast to rank 3 deliberately: a rank-2 mask is read by
`GroupedQueryAttention` as a `(batch, seq)` padding mask, not as a
`(seq, seq)` score mask. The vision tower is bidirectional and stays so -- there
is no ordering over image patches to respect.

The temperature is stored as its logarithm. `logit_scale` is an unconstrained
scalar weight and `exp` is applied on every use, which keeps the multiplier
strictly positive under ordinary gradient descent without a constraint object.
Its default init of 2.6592 is `ln(1 / 0.07)`, the CLIP paper's starting
temperature of roughly 14.3. Unlike the MobileCLIP models in this repository,
this class applies no upper clamp to `exp(logit_scale)`: a diverging temperature
here produces `inf` logits and a `nan` loss with no other symptom, so a trainer
that expects OpenCLIP's clamp must supply it.

`call` is deliberately partial. Passing only `image` or only `text` returns just
that tower's features and omits the logits keys entirely, so encoding a caption
bank for retrieval does not require fabricating a dummy image batch. The dict
output shape is therefore input-dependent, which any consumer indexing the
result must account for.

References:
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision. (https://arxiv.org/abs/2103.00020)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
"""

import keras
from keras import layers, ops, initializers
from typing import Optional, Any, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.clip_utils import (
    compute_clip_logits,
    last_non_pad_token,
)
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.utils.masking import create_mask

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CLIP(keras.Model):
    """CLIP: two independent transformer towers meeting at one similarity matmul.

    A Vision Transformer encodes images and a text Transformer encodes
    captions; both project into a shared ``embed_dim`` space, L2-normalize, and
    produce an ``N x N`` cosine similarity matrix scaled by a learnable
    temperature. The towers share NO parameters and never attend to each other:
    the final matmul is the only point of contact, and the contrastive gradient
    is the only thing forcing them into agreement. Both towers are modernized
    relative to Radford et al. -- grouped-query attention, RMSNorm in the
    pre-norm position, SwiGLU feed-forward, and RoPE in place of a learned
    positional table. The contrastive loss itself is NOT here; this model emits
    the two logits matrices and the temperature.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────┐   ┌────────────────────────────┐
        │  Image [B, H, W, 3]        │   │  Text [B, L] token ids     │
        └─────────────┬──────────────┘   └─────────────┬──────────────┘
                      ▼                                ▼
        ┌────────────────────────────┐   ┌────────────────────────────┐
        │ Conv P×P /P  (patchify)    │   │ Embedding(vocab, W_t)      │
        │ → [B, N_patches, W_v]      │   │ → [B, L, W_t]              │
        │ NO positional table        │   │ NO positional table        │
        └─────────────┬──────────────┘   └─────────────┬──────────────┘
                      ▼                                │
        ┌────────────────────────────┐                 │
        │ prepend CLS token          │                 │
        │ ONE (1,1,W_v) weight,      │                 │
        │ broadcast over the batch   │                 │
        │ → [B, N+1, W_v]            │                 │
        └─────────────┬──────────────┘                 │
                      ▼                                ▼
        ┌────────────────────────────┐   ┌────────────────────────────┐
        │ TransformerLayer × L_v     │   │ TransformerLayer × L_t     │
        │  GQA + RoPE, RMSNorm(pre), │   │  GQA + RoPE, RMSNorm(pre), │
        │  SwiGLU FFN                │   │  SwiGLU FFN                │
        │  BIDIRECTIONAL             │   │  CAUSAL mask (rank 3)      │
        └─────────────┬──────────────┘   └─────────────┬──────────────┘
                      ▼                                ▼
        ┌────────────────────────────┐   ┌────────────────────────────┐
        │ pool position 0 (CLS)      │   │ pool last non-pad token    │
        │                            │   │ assumes RIGHT padding,     │
        │                            │   │ pad id 0 (hard-coded)      │
        └─────────────┬──────────────┘   └─────────────┬──────────────┘
                      ▼                                ▼
        ┌────────────────────────────┐   ┌────────────────────────────┐
        │ Dense(embed_dim), no bias  │   │ Dense(embed_dim), no bias  │
        │ → L2 normalize             │   │ → L2 normalize             │
        └─────────────┬──────────────┘   └─────────────┬──────────────┘
                      │                                │
                      └───────────────┬────────────────┘
                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │  S = exp(logit_scale) · f_I @ f_Tᵀ                       │
        │  logits_per_image [B, B]                                 │
        │  logits_per_text  [B, B]  (the transpose)                │
        │  the ONLY place the two towers meet                      │
        └──────────────────────────────────────────────────────────┘

    **Contrastive objective (loss lives in dl_techniques.losses):**

    .. code-block:: text

                     T₀   T₁   T₂   T₃         each row must pick its
                   ┌────┬────┬────┬────┐       own column, and each
              I₀   │ ✓  │    │    │    │       column its own row
                   ├────┼────┼────┼────┤
              I₁   │    │ ✓  │    │    │       diagonal  = N positives
                   ├────┼────┼────┼────┤       off-diag  = N² − N free
              I₂   │    │    │ ✓  │    │                   negatives
                   ├────┼────┼────┼────┤
              I₃   │    │    │    │ ✓  │       loss = ½(CE over rows
                   └────┴────┴────┴────┘              + CE over cols)

        temperature is stored as its LOG: exp() on every use keeps the
        multiplier positive with no constraint object.
        default 2.6592 = ln(1 / 0.07) ≈ 14.3
        NO upper clamp here (unlike MobileCLIP): a diverging temperature
        gives inf logits and a nan loss with no other symptom.

    **Text causal mask (semantics inversion and rank):**

    .. code-block:: text

        create_mask('causal')  BLOCK semantics    attention layers want
        True = mask OUT                            ATTEND semantics
              ┌───┬───┬───┬───┐                  ┌───┬───┬───┬───┐
              │ F │ T │ T │ T │                  │ T │ F │ F │ F │
              ├───┼───┼───┼───┤   logical_not    ├───┼───┼───┼───┤
              │ F │ F │ T │ T │  ──────────────► │ T │ T │ F │ F │
              ├───┼───┼───┼───┤                  ├───┼───┼───┼───┤
              │ F │ F │ F │ T │                  │ T │ T │ T │ F │
              ├───┼───┼───┼───┤                  ├───┼───┼───┼───┤
              │ F │ F │ F │ F │                  │ T │ T │ T │ T │
              └───┴───┴───┴───┘                  └───┴───┴───┴───┘

        then broadcast to rank 3 [B, L, L] ON PURPOSE: a rank-2 mask is
        read by GroupedQueryAttention as a (batch, seq) PADDING mask,
        not as a (seq, seq) score mask.

    **Variants:**

    .. code-block:: text

        variant     patch  L_v  W_v   H_v  KV_v  L_t  W_t   H_t  KV_t  embed
        ViT-B/32      32    12   768   12    4    12   512    8    8    512
        ViT-B/16      16    12   768   12    4    12   512    8    8    512
        ViT-L/14      14    24  1024   16    4    12   768   12   12    768
        ViT-H/14      14    32  1280   16    4    24  1024   16   16   1024

        L_v/W_v/H_v = vision layers / width / heads, KV = kv heads.
        H/14 is the one scale where the TEXT tower deepens (24, not 12);
        see the D-112 anchor in MODEL_VARIANTS.
        The kv-head columns are NOT from any released CLIP: no CLIP
        checkpoint uses grouped-query attention.

    :param image_size: Input image height and width. Must be positive and
        divisible by ``patch_size``. Defaults to 224.
    :type image_size: int
    :param patch_size: Edge length of a square image patch; also the stride of
        the patch convolution. Must divide ``image_size``. Defaults to 16.
    :type patch_size: int
    :param vision_layers: Number of transformer layers in the vision tower.
        Must be positive. Defaults to 12.
    :type vision_layers: int
    :param vision_width: Hidden dimension of the vision tower. Must be
        divisible by ``vision_heads``. Defaults to 768.
    :type vision_width: int
    :param vision_heads: Number of attention heads in the vision tower.
        Defaults to 12.
    :type vision_heads: int
    :param vision_kv_heads: Number of key-value heads for vision grouped-query
        attention. Must divide ``vision_heads``. Defaults to 4.
    :type vision_kv_heads: int
    :param vocab_size: Size of the text vocabulary. Must be positive. Defaults
        to 49408.
    :type vocab_size: int
    :param context_length: Maximum text sequence length. Must be positive.
        Defaults to 77.
    :type context_length: int
    :param text_layers: Number of transformer layers in the text tower. Must be
        positive. Defaults to 12.
    :type text_layers: int
    :param text_width: Hidden dimension of the text tower. Must be divisible by
        ``text_heads``. Defaults to 512.
    :type text_width: int
    :param text_heads: Number of attention heads in the text tower. Defaults
        to 8.
    :type text_heads: int
    :param text_kv_heads: Number of key-value heads for text grouped-query
        attention. Must divide ``text_heads``. Defaults to 8.
    :type text_kv_heads: int
    :param embed_dim: Dimension of the shared embedding space both towers
        project into. Must be positive. Defaults to 512.
    :type embed_dim: int
    :param ffn_expansion_factor: Expansion factor for both towers' SwiGLU
        feed-forward networks. Must be positive. Defaults to 4.
    :type ffn_expansion_factor: int
    :param ffn_multiple_of: Round the FFN hidden dimension up to a multiple of
        this value. Must be positive. Defaults to 256.
    :type ffn_multiple_of: int
    :param dropout_rate: General dropout probability, in ``[0, 1)``. Defaults
        to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout probability for attention weights,
        in ``[0, 1)``. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param logit_scale_init: Initial value of the LOG temperature. Defaults to
        2.6592, i.e. ``ln(1 / 0.07)``, so the initial multiplier is about 14.3.
        No upper clamp is applied on use.
    :type logit_scale_init: float
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``image_size`` is not divisible by ``patch_size``,
        if either width is not divisible by its head count, or if either head
        count is not divisible by its kv-head count.

    Input shape:
        A mapping with either or both of:

        - ``'image'``: 4D tensor ``(batch_size, height, width, channels)``
        - ``'text'``: 2D tensor ``(batch_size, sequence_length)``

        A tuple ``(images, texts)`` is accepted equivalently.

    Output shape:
        A mapping whose KEYS DEPEND ON THE INPUT:

        - ``'image_features'``: ``(batch_size, embed_dim)``, if an image was
          given.
        - ``'text_features'``: ``(batch_size, embed_dim)``, if text was given.
        - ``'logits_per_image'``, ``'logits_per_text'``:
          ``(batch_size, batch_size)``, and ``'logit_scale'``: scalar. Present
          ONLY when both modalities were given.

    :ivar num_patches: ``(image_size // patch_size) ** 2``.
    :vartype num_patches: int
    :ivar patch_conv: Strided convolution implementing patch embedding.
    :vartype patch_conv: keras.layers.Conv2D
    :ivar vision_transformer_layers: The vision tower's blocks.
    :vartype vision_transformer_layers: list[TransformerLayer]
    :ivar vision_projection: Bias-free projection into the shared space.
    :vartype vision_projection: keras.layers.Dense
    :ivar token_embedding: Text token embedding table.
    :vartype token_embedding: keras.layers.Embedding
    :ivar text_transformer_layers: The text tower's blocks.
    :vartype text_transformer_layers: list[TransformerLayer]
    :ivar text_projection: Bias-free projection into the shared space.
    :vartype text_projection: keras.layers.Dense
    :ivar logit_scale: Scalar LOG temperature weight, created in ``build``.
    :vartype logit_scale: keras.Variable
    :ivar class_token: Learnable ``(1, 1, vision_width)`` CLS token, created in
        ``build``.
    :vartype class_token: keras.Variable

    Example:
        .. code-block:: python

            # Create a model from a predefined variant
            model = CLIP.from_variant("ViT-B/16")

            # Build the model
            model.build({
                'image': (None, 224, 224, 3),
                'text': (None, 77)
            })

            images = keras.random.normal((32, 224, 224, 3))
            text_tokens = keras.random.uniform(
                (32, 77), 0, 49408, dtype='int32')

            # Full forward pass for training
            outputs = model({'image': images, 'text': text_tokens})

            # Single-tower inference
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            model.save('clip_model.keras')
            loaded_model = keras.models.load_model('clip_model.keras')

    Note:
        Text pooling assumes RIGHT padding with pad id 0. A left-padded batch,
        or a tokenizer whose pad id is not zero, pools the wrong position
        silently rather than raising.
    """

    MODEL_VARIANTS = {
        "ViT-B/32": {
            "patch_size": 32,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_heads": 12,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "text_kv_heads": 8,
            "embed_dim": 512,
        },
        "ViT-B/16": {
            "patch_size": 16,
            "vision_layers": 12,
            "vision_width": 768,
            "vision_heads": 12,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "text_kv_heads": 8,
            "embed_dim": 512,
        },
        "ViT-L/14": {
            "patch_size": 14,
            "vision_layers": 24,
            "vision_width": 1024,
            "vision_heads": 16,
            "vision_kv_heads": 4,
            "text_layers": 12,
            "text_width": 768,
            "text_heads": 12,
            "text_kv_heads": 12,
            "embed_dim": 768,
        },
        # DECISION plan-2026-08-22T035419-a11304c8/D-112
        # `text_layers` is 24, not 12. FETCHED 2026-08-22 from the only released
        # ViT-H/14 CLIP,
        # https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/raw/main/config.json
        # -- text_config: hidden_size 1024, num_hidden_layers 24,
        # num_attention_heads 16, projection_dim 1024; vision_config:
        # hidden_size 1280, num_hidden_layers 32, num_attention_heads 16,
        # patch_size 14. Every other field in this row already agreed. The 12
        # was the L/14 row's text depth carried down one line: B/32, B/16 and
        # L/14 all legitimately have 12 text layers (verified against
        # `openai/clip-vit-{base-patch32,base-patch16,large-patch14}`), H/14 is
        # the one scale where the text tower deepens, and copying the previous
        # row is exactly how that gets missed. Do NOT "restore consistency" by
        # putting 12 back.
        # `vision_kv_heads`/`text_kv_heads` are NOT from any released CLIP -- no
        # CLIP checkpoint uses grouped-query attention. They are this
        # implementation's declared modernization (GQA + RMSNorm + SwiGLU +
        # RoPE, module docstring line 3, Ainslie et al. 2023 cited), so they are
        # deliberately not traced to Radford et al. 2021.
        # See decisions.md D-112.
        "ViT-H/14": {
            "patch_size": 14,
            "vision_layers": 32,
            "vision_width": 1280,
            "vision_heads": 16,
            "vision_kv_heads": 4,
            "text_layers": 24,
            "text_width": 1024,
            "text_heads": 16,
            "text_kv_heads": 16,
            "embed_dim": 1024,
        },
    }

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_heads: int = 12,
        vision_kv_heads: int = 4,
        vocab_size: int = 49408,
        context_length: int = 77,
        text_layers: int = 12,
        text_width: int = 512,
        text_heads: int = 8,
        text_kv_heads: int = 8,
        embed_dim: int = 512,
        ffn_expansion_factor: int = 4,
        ffn_multiple_of: int = 256,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        logit_scale_init: float = 2.6592,
        **kwargs: Any
    ) -> None:
        """Initialize the CLIP model.

        All sub-layers are created here following the golden rule; the model's
        own weights (``logit_scale``, ``class_token``) are created in
        :meth:`build`.

        :param image_size: Input image height and width.
        :type image_size: int
        :param patch_size: Square patch edge length and convolution stride.
        :type patch_size: int
        :param vision_layers: Number of vision transformer layers.
        :type vision_layers: int
        :param vision_width: Vision tower hidden dimension.
        :type vision_width: int
        :param vision_heads: Vision attention heads.
        :type vision_heads: int
        :param vision_kv_heads: Vision key-value heads for GQA.
        :type vision_kv_heads: int
        :param vocab_size: Text vocabulary size.
        :type vocab_size: int
        :param context_length: Maximum text sequence length.
        :type context_length: int
        :param text_layers: Number of text transformer layers.
        :type text_layers: int
        :param text_width: Text tower hidden dimension.
        :type text_width: int
        :param text_heads: Text attention heads.
        :type text_heads: int
        :param text_kv_heads: Text key-value heads for GQA.
        :type text_kv_heads: int
        :param embed_dim: Shared embedding dimension.
        :type embed_dim: int
        :param ffn_expansion_factor: SwiGLU expansion factor.
        :type ffn_expansion_factor: int
        :param ffn_multiple_of: FFN hidden-dimension rounding multiple.
        :type ffn_multiple_of: int
        :param dropout_rate: General dropout probability.
        :type dropout_rate: float
        :param attention_dropout_rate: Attention-weight dropout probability.
        :type attention_dropout_rate: float
        :param logit_scale_init: Initial LOG temperature.
        :type logit_scale_init: float
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any divisibility constraint is violated.
        """
        super().__init__(**kwargs)

        # Store all configuration parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_heads = vision_heads
        self.vision_kv_heads = vision_kv_heads
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.text_layers = text_layers
        self.text_width = text_width
        self.text_heads = text_heads
        self.text_kv_heads = text_kv_heads
        self.embed_dim = embed_dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.logit_scale_init = logit_scale_init

        # Validate configuration
        self._validate_config()

        # Derived properties
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Create sub-layers in __init__ (golden rule)
        self._create_vision_encoder()
        self._create_text_encoder()

        # Weight attributes (created in build())
        self.logit_scale = None
        self.class_token = None

        logger.info(
            f"CLIPModel initialized with embed_dim={self.embed_dim}, "
            f"vision_layers={self.vision_layers}, "
            f"text_layers={self.text_layers}"
        )

    def _validate_config(self) -> None:
        """Validate the divisibility constraints between the stored parameters.

        :raises ValueError: If ``image_size`` is not divisible by
            ``patch_size``, if either tower's width is not divisible by its
            head count, or if either head count is not divisible by its
            kv-head count.
        """
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.vision_width % self.vision_heads != 0:
            raise ValueError(
                f"vision_width ({self.vision_width}) must be divisible by "
                f"vision_heads ({self.vision_heads})"
            )
        if self.vision_heads % self.vision_kv_heads != 0:
            raise ValueError(
                f"vision_heads ({self.vision_heads}) must be divisible by "
                f"vision_kv_heads ({self.vision_kv_heads})"
            )
        if self.text_width % self.text_heads != 0:
            raise ValueError(
                f"text_width ({self.text_width}) must be divisible by "
                f"text_heads ({self.text_heads})"
            )
        if self.text_heads % self.text_kv_heads != 0:
            raise ValueError(
                f"text_heads ({self.text_heads}) must be divisible by "
                f"text_kv_heads ({self.text_kv_heads})"
            )

    def _create_vision_encoder(self) -> None:
        """Create every vision-tower sub-layer.

        Follows the golden rule: layers are created in ``__init__``, not in
        ``build``. The CLS token is a WEIGHT and therefore belongs to
        :meth:`build` instead.
        """
        # Patch embedding layer
        self.patch_conv = layers.Conv2D(
            filters=self.vision_width,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='patch_conv'
        )

        # Vision transformer layers
        self.vision_transformer_layers = [
            TransformerLayer(
                hidden_size=self.vision_width,
                num_heads=self.vision_heads,
                intermediate_size=int(
                    self.vision_width * self.ffn_expansion_factor
                ),
                attention_type='group_query',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                n_kv_head=self.vision_kv_heads,
                ffn_args={
                    "ffn_expansion_factor": self.ffn_expansion_factor,
                    "ffn_multiple_of": self.ffn_multiple_of,
                },
                name=f'vision_transformer_{i}'
            )
            for i in range(self.vision_layers)
        ]

        # Vision projection layer
        self.vision_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='vision_projection'
        )

    def _create_text_encoder(self) -> None:
        """Create every text-tower sub-layer.

        Follows the golden rule: layers are created in ``__init__``, not in
        ``build``.
        """
        # Token embedding layer
        self.token_embedding = layers.Embedding(
            self.vocab_size,
            self.text_width,
            embeddings_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='token_embedding'
        )

        # Text transformer layers
        self.text_transformer_layers = [
            TransformerLayer(
                hidden_size=self.text_width,
                num_heads=self.text_heads,
                intermediate_size=int(
                    self.text_width * self.ffn_expansion_factor
                ),
                attention_type='group_query',
                normalization_type='rms_norm',
                normalization_position='pre',
                ffn_type='swiglu',
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                n_kv_head=self.text_kv_heads,
                ffn_args={
                    "ffn_expansion_factor": self.ffn_expansion_factor,
                    "ffn_multiple_of": self.ffn_multiple_of,
                },
                name=f'text_transformer_{i}'
            )
            for i in range(self.text_layers)
        ]

        # Text projection layer
        self.text_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            name='text_projection'
        )

    def build(
        self,
        input_shape: Union[Dict[str, Any], Tuple[Any, ...]]
    ) -> None:
        """Create the model's own weights and materialize every sub-layer.

        Following the golden rule, this creates the two weights that do not
        belong to any sub-layer (``logit_scale`` and ``class_token``) and then
        explicitly builds each tower's layers at their known shapes.

        :param input_shape: Either a mapping with ``'image'`` and ``'text'``
            shapes or a tuple of shapes. Only forwarded to
            ``super().build``; every sub-layer shape is derived from the
            constructor config.
        :type input_shape: Union[Dict[str, Any], Tuple[Any, ...]]
        """
        if self.built:
            return

        # Create learnable temperature parameter
        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=initializers.Constant(self.logit_scale_init),
            trainable=True
        )

        # Create learnable CLS token for vision encoder
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.vision_width),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Build vision encoder sub-layers
        # Patch conv expects (batch, height, width, channels)
        self.patch_conv.build((None, self.image_size, self.image_size, 3))

        # Vision transformers expect (batch, num_patches+1, vision_width)
        vision_seq_shape = (None, self.num_patches + 1, self.vision_width)
        for layer in self.vision_transformer_layers:
            layer.build(vision_seq_shape)

        # Vision projection expects (batch, vision_width)
        self.vision_projection.build((None, self.vision_width))

        # Build text encoder sub-layers
        # Token embedding expects (batch, seq_len)
        self.token_embedding.build((None, self.context_length))

        # Text transformers expect (batch, context_length, text_width)
        text_seq_shape = (None, self.context_length, self.text_width)
        for layer in self.text_transformer_layers:
            layer.build(text_seq_shape)

        # Text projection expects (batch, text_width)
        self.text_projection.build((None, self.text_width))

        # Mark as built
        super().build(input_shape)

    def _ensure_built(self) -> None:
        """Build the model's own weights if a public encoder is called first.

        ``class_token``, ``logit_scale`` and the positional embeddings are
        created in ``build``, which Keras runs on ``__call__``. ``encode_image``
        is a public entry point that does NOT go through ``__call__``, so on a
        fresh instance it reads ``self.class_token`` as None and fails inside
        ``ops.broadcast_to`` with

            ValueError: Attempt to convert a value (None) ... to a Tensor

        rather than saying the model is unbuilt. Shapes come from the
        constructor config, so no input is needed to resolve them.

        ``encode_text`` deliberately does NOT call this. It touches none of the
        weights ``build`` creates, and building there would LOCK the layer, so a
        caller that swaps or taps ``text_transformer_layers`` after a first
        ``encode_text`` -- which is exactly how the causality probe in
        ``tests/test_models/test_clip/test_model.py`` reads per-position hidden
        states -- would hit "You cannot add new elements of state to a layer
        that is already built".
        """
        if self.built:
            return
        self.build({
            'image': (None, self.image_size, self.image_size, 3),
            'text': (None, self.context_length),
        })

    def encode_image(
        self,
        images: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Encode images into the shared embedding space.

        :param images: Input images of shape
            ``(batch_size, height, width, channels)``.
        :type images: keras.KerasTensor
        :param training: Whether the model is in training mode. ``None`` uses
            the training mode from the call context.
        :type training: Optional[bool]
        :return: L2-normalized image features of shape
            ``(batch_size, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        self._ensure_built()
        batch_size = ops.shape(images)[0]

        # Convert to patches: (batch, num_patches, vision_width)
        patches = self.patch_conv(images, training=training)
        patches = ops.reshape(
            patches, (batch_size, self.num_patches, self.vision_width)
        )

        # Add class token: (batch, num_patches+1, vision_width)
        class_tokens = ops.broadcast_to(
            self.class_token, (batch_size, 1, self.vision_width)
        )
        x = ops.concatenate([class_tokens, patches], axis=1)

        # Apply vision transformer layers
        for transformer_layer in self.vision_transformer_layers:
            x = transformer_layer(x, training=training)

        # Extract class token representation
        class_token_features = x[:, 0]

        # Project to shared embedding space
        image_features = self.vision_projection(
            class_token_features, training=training
        )

        # L2 normalize features
        image_features = image_features / ops.norm(
            image_features, axis=-1, keepdims=True
        )

        return image_features

    def encode_text(
        self,
        text_ids: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Encode text into the shared embedding space.

        The tower is CAUSAL: a lower-triangular mask is built here and passed
        to every block, which is what makes the pooled last real token the only
        position that has read the whole sentence. Pooling assumes RIGHT
        padding with pad id 0.

        :param text_ids: Token IDs of shape ``(batch_size, sequence_length)``.
        :type text_ids: keras.KerasTensor
        :param training: Whether the model is in training mode. ``None`` uses
            the training mode from the call context.
        :type training: Optional[bool]
        :return: L2-normalized text features of shape
            ``(batch_size, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # Token embeddings: (batch, seq_len, text_width)
        x = self.token_embedding(text_ids, training=training)

        # Causal mask. `create_mask` returns BLOCK semantics (True = mask out);
        # the attention layers expect ATTEND semantics, hence the inversion.
        # Without this the tower is bidirectional and the pooled last token is
        # not the only position that has read the whole sentence -- which is
        # both a departure from OpenAI CLIP and the reason last-token pooling
        # is meaningful at all.
        # The mask is broadcast to rank 3 (batch, seq, seq) on purpose: the
        # attention layers read a rank-2 mask as a (batch, seq) PADDING mask.
        batch_size = ops.shape(text_ids)[0]
        seq_len = ops.shape(text_ids)[1]
        causal_block = create_mask('causal', seq_len=seq_len, dtype='bool')
        causal_block = ops.broadcast_to(
            ops.expand_dims(causal_block, axis=0),
            (batch_size, seq_len, seq_len),
        )
        attend_mask = ops.logical_not(causal_block)

        # Apply text transformer layers
        for transformer_layer in self.text_transformer_layers:
            x = transformer_layer(x, attention_mask=attend_mask, training=training)

        # Extract features from the last non-padding token
        # (assuming 0 is the padding id; right-padded sequences).
        text_features_raw = last_non_pad_token(x, text_ids, 0)

        # Project to shared embedding space
        text_features = self.text_projection(
            text_features_raw, training=training
        )

        # L2 normalize features
        text_features = text_features / ops.norm(
            text_features, axis=-1, keepdims=True
        )

        return text_features

    def call(
        self,
        inputs: Union[
            Dict[str, keras.KerasTensor],
            Tuple[keras.KerasTensor, ...]
        ],
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the CLIP model, partial by design.

        Handles single-modality and dual-modality inputs. Training needs both
        ``'image'`` and ``'text'``; inference may pass either, in which case
        only that tower's features are returned and the logits keys are omitted
        entirely, so encoding a caption bank does not require a dummy image
        batch. The output structure therefore DEPENDS ON THE INPUT.

        :param inputs: A mapping with keys ``'image'`` and/or ``'text'``, or a
            tuple ``(images, texts)``.
        :type inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Whether the model is in training mode. ``None`` uses
            the training mode from the call context.
        :type training: Optional[bool]
        :return: A dictionary with:

            - ``image_features``: image embeddings, if images were provided.
            - ``text_features``: text embeddings, if texts were provided.
            - ``logits_per_image``, ``logits_per_text``, ``logit_scale``: only
              when BOTH modalities were provided.

        :rtype: Dict[str, keras.KerasTensor]
        """
        # Parse inputs
        if isinstance(inputs, dict):
            images = inputs.get('image')
            texts = inputs.get('text')
        else:
            images = inputs[0] if len(inputs) > 0 else None
            texts = inputs[1] if len(inputs) > 1 else None

        # Encode modalities
        results = {}
        image_features, text_features = None, None

        if images is not None:
            image_features = self.encode_image(images, training=training)
            results['image_features'] = image_features

        if texts is not None:
            text_features = self.encode_text(texts, training=training)
            results['text_features'] = text_features

        # Compute similarity if both modalities are present
        if image_features is not None and text_features is not None:
            logit_scale = ops.exp(self.logit_scale)
            logits_per_image, logits_per_text = compute_clip_logits(
                image_features, text_features, logit_scale
            )

            results.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale
            })

        return results

    def compute_output_shape(
        self,
        input_shape: Union[Dict[str, Any], Tuple]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute the output shapes, mirroring ``call``'s partial contract.

        :param input_shape: A mapping with ``'image'`` and/or ``'text'``
            shapes, or a tuple of shapes.
        :type input_shape: Union[Dict[str, Any], Tuple]
        :return: Mapping from output key to shape, containing only the keys
            ``call`` would emit for this input.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        output_shapes = {}

        if isinstance(input_shape, dict):
            image_shape = input_shape.get('image')
            text_shape = input_shape.get('text')
        else:
            image_shape = input_shape[0] if len(input_shape) > 0 else None
            text_shape = input_shape[1] if len(input_shape) > 1 else None

        if image_shape is not None:
            batch_size = image_shape[0]
            output_shapes['image_features'] = (batch_size, self.embed_dim)

        if text_shape is not None:
            batch_size = text_shape[0]
            output_shapes['text_features'] = (batch_size, self.embed_dim)

        if image_shape is not None and text_shape is not None:
            batch_size = image_shape[0]
            output_shapes['logits_per_image'] = (batch_size, batch_size)
            output_shapes['logits_per_text'] = (batch_size, batch_size)
            output_shapes['logit_scale'] = ()

        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'vision_layers': self.vision_layers,
            'vision_width': self.vision_width,
            'vision_heads': self.vision_heads,
            'vision_kv_heads': self.vision_kv_heads,
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'text_layers': self.text_layers,
            'text_width': self.text_width,
            'text_heads': self.text_heads,
            'text_kv_heads': self.text_kv_heads,
            'embed_dim': self.embed_dim,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'ffn_multiple_of': self.ffn_multiple_of,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'logit_scale_init': self.logit_scale_init,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CLIP":
        """Create a model instance from its configuration.

        :param config: Dictionary containing constructor parameters.
        :type config: Dict[str, Any]
        :return: New CLIP instance.
        :rtype: CLIP
        """
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        **kwargs: Any
    ) -> "CLIP":
        """Create a CLIP model from a predefined variant.

        :param variant: One of ``"ViT-B/32"``, ``"ViT-B/16"``, ``"ViT-L/14"``,
            ``"ViT-H/14"``.
        :type variant: str
        :param kwargs: Additional arguments overriding the variant's defaults.
        :type kwargs: Any
        :return: A CLIP instance configured for the specified variant.
        :rtype: CLIP
        :raises ValueError: If the variant is not recognized.

        Example:
            .. code-block:: python

                # Create ViT-B/16 variant
                model = CLIP.from_variant("ViT-B/16")

                # Create with custom dropout
                model = CLIP.from_variant("ViT-B/16", dropout_rate=0.1)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)  # Allow overriding defaults

        logger.info(
            f"Creating CLIP {variant} variant with modern improvements"
        )
        return cls(**config)

# ---------------------------------------------------------------------

def create_clip_model(**kwargs: Any) -> CLIP:
    """Convenience function to create a CLIP model with a custom configuration.

    A thin wrapper around the :class:`CLIP` constructor.

    :param kwargs: Keyword arguments for the :class:`CLIP` constructor.
    :type kwargs: Any
    :return: A configured CLIP instance.
    :rtype: CLIP

    Example:
        .. code-block:: python

            model = create_clip_model(
                image_size=384,
                patch_size=16,
                vision_layers=24,
                embed_dim=768
            )
    """
    logger.info("Creating custom CLIP model")
    return CLIP(**kwargs)

# ---------------------------------------------------------------------

def create_clip_variant(variant: str, **kwargs: Any) -> CLIP:
    """Convenience function to create a CLIP model from a predefined variant.

    :param variant: The model variant string (e.g. ``"ViT-B/16"``).
    :type variant: str
    :param kwargs: Additional arguments overriding the variant's defaults.
    :type kwargs: Any
    :return: A configured CLIP instance.
    :rtype: CLIP
    :raises ValueError: If the variant is not recognized.

    Example:
        .. code-block:: python

            # Create standard ViT-B/16
            model = create_clip_variant("ViT-B/16")

            # Create with modifications
            model = create_clip_variant(
                "ViT-B/16",
                dropout_rate=0.1,
                attention_dropout_rate=0.1
            )
    """
    return CLIP.from_variant(variant, **kwargs)

# ---------------------------------------------------------------------