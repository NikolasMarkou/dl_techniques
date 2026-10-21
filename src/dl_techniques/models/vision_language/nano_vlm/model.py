"""
NanoVLM, built by the ``NanoVLM`` class, is a small vision-language model in
the nanoVLM design: image tokens are projected into the text embedding space,
put BEFORE the text token embeddings, and one causal decoder runs over the
joint sequence under a vocabulary head tied to the word embeddings.

An image is a grid of pixels with no order; a caption is a sequence of
vocabulary indices whose order is the point. This model makes the image look
like text first. A vision transformer (``output_mode='none'``) emits a
sequence of patch tokens; ``ModalityProjection`` folds each 2 x 2 block of
patch tokens into one wider token (pixel shuffle, lossless) and maps it to the
text width, so the image becomes a short prefix of "visual words". The text
decoder's word embeddings follow it, and the decoder's blocks run over
``[vision; text]`` with one causal mask, exactly as over plain text.

Every position attends only to itself and earlier positions, vision tokens
included (the reference nanoVLM's language model runs fully causal over the
concatenation; there is no bidirectional image prefix). So a text position
reads the whole image and the text before it and nothing after it, and
next-token training cannot see the token it predicts.

The output is unnormalized logits over the vocabulary at every position of
the joint sequence; no softmax is applied, so a loss takes
``from_logits=True``. The text logits are the trailing ``T`` positions.
Input/output embedding tying happens at call time via a matrix multiply
against the transposed embedding table, not by reassigning a layer's weight,
which Keras 3 forbids after build; a tied model builds no output ``Dense`` at
all. ``generate()`` is a plain sampling loop with no key-value cache, so cost
grows with the square of the number of generated tokens; it is meant for
evaluating a trained model, not for serving.

References:
    - Wiedmann et al., 2025. nanoVLM: the simplest repository to train your
      VLM in pure PyTorch. (https://github.com/huggingface/nanoVLM)
    - Liu et al., 2023. Visual Instruction Tuning. (https://arxiv.org/abs/2304.08485)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Shi et al., 2016. Real-Time Single Image and Video Super-Resolution
      Using an Efficient Sub-Pixel Convolutional Neural Network.
      (https://arxiv.org/abs/1609.05158)
    - Press and Wolf, 2016. Using the Output Embedding to Improve Language
      Models. (https://arxiv.org/abs/1608.05859)
"""

import copy
import keras

from typing import Dict, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.modality_projection import ModalityProjection
from dl_techniques.layers.transformers.text_decoder import TextDecoder
from dl_techniques.layers.transformers.vision_encoder import VisionEncoder, create_vision_encoder
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.utils.weight_transfer import refuse_pretrained

# ---------------------------------------------------------------------

#: ``ModalityProjection`` arguments NanoVLM sets itself from the two towers.
_DERIVED_PROJECTION_KEYS = ('input_dim', 'output_dim')

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.nano_vlm.model")
class NanoVLM(keras.Model):
    """
    Vision-language model: projected image tokens prefixed to the text, one
    causal decoder over both, one vocabulary head.

    Architecture:

    .. code-block:: text

        images [B,H,W,C]                    text_tokens [B,T]
              │                                    │
              ▼                                    ▼
        ┌────────────────┐               ┌───────────────────┐
        │ VisionEncoder  │               │ TextDecoder       │
        │ output_mode=   │               │ .embed_tokens     │
        │ 'none'         │               │ (word table only) │
        └───────┬────────┘               └─────────┬─────────┘
                │ [B,1+P,Dv]                       │
                ▼                                  │
        ┌────────────────────┐                     │
        │ ModalityProjection │ pixel shuffle s,    │
        │ to the text width  │ Dense to Dt         │
        └───────┬────────────┘                     │
                │ [B,Sv,Dt]                        │ [B,T,Dt]
                └──────────────┬───────────────────┘
                               ▼
                 concat on the sequence axis [B, Sv+T, Dt]
                               ▼
                ┌─────────────────────────────────────┐
                │ TextDecoder.decode_embeddings       │
                │ positions 0..Sv+T-1, ONE causal     │
                │ mask over vision and text, blocks,  │
                │ final norm                          │
                └──────────────┬──────────────────────┘
                               ▼
                dropout, then the vocabulary projection
                (tied: hidden @ word_embeddings.T)
                               ▼
                 logits [B, Sv+T, vocab_size]

    ``Sv`` is :attr:`num_vision_tokens`: ``(grid / s)^2 + 1`` for a
    ``grid x grid`` patch grid, pixel-shuffle factor ``s`` and the CLS token,
    so 50 at 224px with 16px patches and ``s = 2``.

    :param vision_config: Configuration for the vision encoder. Must include
        ``img_size``, ``patch_size``, ``embed_dim``, ``depth``, ``num_heads``.
        ``output_mode`` is forced to ``'none'``.
    :type vision_config: Dict[str, Any]
    :param text_config: Configuration for the :class:`TextDecoder`. Must
        include ``vocab_size``, ``embed_dim``, ``depth``, ``num_heads``.
        ``max_seq_len`` (default 512) is the TEXT length limit; the decoder is
        built with ``num_vision_tokens + max_seq_len`` positions, so the
        vision prefix never eats into it.
    :type text_config: Dict[str, Any]
    :param vocab_size: Vocabulary size for embeddings and the output
        projection. Must be positive. Defaults to ``32000``.
    :type vocab_size: int
    :param projection_config: Optional keyword arguments for
        :class:`ModalityProjection` (``scale_factor``, ``use_gelu``,
        ``use_layer_norm``, initializers, regularizers). ``input_dim`` and
        ``output_dim`` are the two towers' widths and may not be passed.
        ``None`` keeps the layer's defaults (``scale_factor=2``, GELU and
        layer norm on).
    :type projection_config: Optional[Dict[str, Any]]
    :param use_shared_embedding: Tie the output to the decoder's word
        embedding table: the logits are ``hidden @ embeddings.T`` and no
        output ``Dense`` is built. A decoder with
        ``embedding_type='factorized'`` has no full table to tie to and
        refuses ``True``. Defaults to ``True``.
    :type use_shared_embedding: bool
    :param output_dropout_rate: Dropout before the vocabulary projection, in
        ``[0, 1]``. Defaults to ``0.1``.
    :type output_dropout_rate: float
    :param initializer_range: Standard deviation of the truncated-normal
        initializer of the untied ``output_projection``; the towers and the
        projector use their own initializers. Must be positive. Defaults to
        ``0.02``.
    :type initializer_range: float
    :param kernel_regularizer: Optional regularizer for the untied
        ``output_projection`` kernel. A tied model has no such kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Forwarded to ``keras.Model``; an unknown keyword raises
        ``ValueError``.

    Input shape:
        Dict with:

        - ``'images'``: ``(batch_size, height, width, channels)``
        - ``'text_tokens'``: ``(batch_size, T)`` token IDs

        Optional: ``'attention_mask'`` ``(batch_size, T)``, 1 for a real
        text token and 0 for padding; the vision prefix is always attended.

    Output shape:
        ``(batch_size, num_vision_tokens + T, vocab_size)``; the text logits
        are the trailing ``T`` positions.

    :ivar vision_encoder: Vision encoder producing patch-token sequences.
    :ivar modality_projection: Pixel shuffle plus projection to the text width.
    :ivar text_decoder: The causal decoder; its word table embeds the text and
        is the tied output matrix.
    :ivar num_vision_tokens: Length of the vision prefix, ``Sv``.
    :ivar output_projection: Bias-free ``Dense(vocab_size)`` for an untied
        model, ``None`` for a tied one (the logits then come from the word
        embedding table, and no second vocabulary matrix is built).
    :ivar final_dropout: Dropout applied before the vocabulary projection.

    Example:

    .. code-block:: python

        model = NanoVLM(
            vision_config={
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'output_mode': 'none'
            },
            text_config={
                'vocab_size': 32000, 'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 512
            },
            vocab_size=32000
        )
        inputs = {
            'images': keras.random.normal((2, 224, 224, 3)),
            'text_tokens': keras.random.randint((2, 128), 0, 32000)
        }
        logits = model(inputs, training=True)   # (2, 50 + 128, 32000)

    :raises ValueError: If a configuration dict is missing a required key,
        if ``projection_config`` names a derived key, or if a numeric
        parameter is out of range.
    """

    #: Public-name registry of the four named NanoVLM configurations, read by
    #: :meth:`from_variant`. ``vocab_size`` is a caller argument, not a variant
    #: property, so it is absent here; ``from_variant`` writes it into a deep
    #: copy of the selected entry.
    #:
    #: ``'modern'`` is the former ``create_modern_nanovlm`` configuration:
    #: RMSNorm, SwiGLU, differential attention and pre-normalization in the
    #: vision tower; factorized word embeddings, RMSNorm, SwiGLU and
    #: pre-normalization in the decoder. A factorized table cannot be tied to
    #: the output (it is ``vocab x 128`` followed by a projection), so the row
    #: is untied and builds its own ``output_projection``.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        'mini': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6, 'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 384, 'depth': 6,
                'num_heads': 6, 'max_seq_len': 512
            },
        },
        'base': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 512
            },
        },
        'large': {
            'vision_config': {
                'img_size': 384, 'patch_size': 16, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16, 'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 1024, 'depth': 24,
                'num_heads': 16, 'max_seq_len': 1024
            },
        },
        'modern': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12,
                'attention_type': 'differential',
                'normalization_type': 'rms_norm',
                'normalization_position': 'pre',
                'ffn_type': 'swiglu',
                'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 1024,
                'embedding_type': 'factorized',
                'normalization_type': 'rms_norm',
                'normalization_position': 'pre',
                'ffn_type': 'swiglu'
            },
            'use_shared_embedding': False
        }
    }

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_config: Dict[str, Any],
            vocab_size: int = 32000,
            projection_config: Optional[Dict[str, Any]] = None,
            use_shared_embedding: bool = True,
            output_dropout_rate: float = 0.1,
            initializer_range: float = 0.02,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate basic parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not (0.0 <= output_dropout_rate <= 1.0):
            raise ValueError(f"output_dropout_rate must be between 0.0 and 1.0, got {output_dropout_rate}")
        if initializer_range <= 0.0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")

        # Store ALL configuration parameters for serialization (CRITICAL for Keras 3)
        self.vision_config = copy.deepcopy(vision_config)
        self.text_config = copy.deepcopy(text_config)
        self.projection_config = copy.deepcopy(projection_config or {})
        self.vocab_size = vocab_size
        self.use_shared_embedding = use_shared_embedding
        self.output_dropout_rate = output_dropout_rate
        self.initializer_range = initializer_range
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate and enhance configurations
        self._validate_and_prepare_configs()

        # CREATE all sub-layers in __init__ (Modern Keras 3 Pattern)
        self.vision_encoder = self._create_vision_encoder()
        self.modality_projection = ModalityProjection(
            input_dim=self.vision_config['embed_dim'],
            output_dim=self.text_config['embed_dim'],
            **self.projection_config,
            name='modality_projection'
        )
        self.num_vision_tokens = self.modality_projection.compute_output_shape(
            (None, self.vision_encoder.seq_len, self.vision_config['embed_dim'])
        )[1]

        # DECISION plan-2026-10-14T062616-81938e81/D-040: ONE causal decoder
        # over [projected vision; text embeddings], the reference nanoVLM.
        # Do not bring back a separate text tower plus a fusion layer that
        # lets vision tokens attend to the text: from its second block on
        # a later text token moved earlier text logits (8.4e-3, 2 blocks),
        # and the last block's vision side never trained. The decoder gets
        # num_vision_tokens extra positions so text_config['max_seq_len']
        # stays the TEXT limit. See decisions.md D-040.
        self.text_decoder = TextDecoder(
            **{**self.text_config,
               'max_seq_len': self.num_vision_tokens + self.text_config['max_seq_len']},
            name='text_decoder'
        )

        # The ONE place that decides where the logits come from; build(),
        # call() and generate() all read it.
        self.ties_output_to_embeddings = bool(use_shared_embedding)
        if self.ties_output_to_embeddings and not hasattr(self.text_decoder, 'word_embeddings'):
            raise ValueError(
                "use_shared_embedding=True ties the output to the decoder's "
                "word embedding table, but text_config embedding_type="
                f"{self.text_config.get('embedding_type')!r} builds no such "
                "table. Pass use_shared_embedding=False for a factorized "
                "embedding."
            )

        self.final_dropout = keras.layers.Dropout(
            rate=output_dropout_rate,
            name='final_dropout'
        ) if output_dropout_rate > 0.0 else None

        # DECISION plan-2026-10-14T062616-81938e81/D-037: a tied model builds NO
        # output Dense. Do not bring back a `shared_output_projection` "for
        # serialization" or "for the untied path": it was created, built, saved
        # and counted (24,576,000 params on 'base') and never called, the one
        # trainable variable with a None gradient. The untied path builds its
        # own `output_projection` below. See decisions.md D-037.
        self.output_projection = None if self.ties_output_to_embeddings else keras.layers.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=self.initializer_range),
            kernel_regularizer=self.kernel_regularizer,
            name='output_projection'
        )

    def _validate_and_prepare_configs(self) -> None:
        """Validate the configuration dicts and fill the derived values."""

        # Validate required keys
        required_vision_keys = ['img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads']
        for key in required_vision_keys:
            if key not in self.vision_config:
                raise ValueError(f"Missing required vision_config key: {key}")

        required_text_keys = ['vocab_size', 'embed_dim', 'depth', 'num_heads']
        for key in required_text_keys:
            if key not in self.text_config:
                raise ValueError(f"Missing required text_config key: {key}")

        derived = [key for key in _DERIVED_PROJECTION_KEYS if key in self.projection_config]
        if derived:
            raise ValueError(
                f"projection_config may not set {derived}: the projection maps "
                "vision_config['embed_dim'] to text_config['embed_dim']"
            )

        # The text length limit is stored, so a reloaded model has the same one.
        self.text_config.setdefault('max_seq_len', 512)

        # Ensure vocab_size consistency
        if self.text_config['vocab_size'] != self.vocab_size:
            logger.warning(
                f"Text config vocab_size ({self.text_config['vocab_size']}) differs from "
                f"model vocab_size ({self.vocab_size}). Using model vocab_size."
            )
            self.text_config['vocab_size'] = self.vocab_size

        # Enhance vision_heads config for sequence output
        if self.vision_config.get('output_mode') != 'none':
            logger.info("Setting vision_heads encoder output_mode to 'none' for sequence features")
            self.vision_config['output_mode'] = 'none'

    def _create_vision_encoder(self) -> VisionEncoder:
        """Create the vision tower, named so its weight paths are stable."""
        return create_vision_encoder(**self.vision_config, name='vision_encoder')

    def build(self, input_shape: Union[Dict[str, Tuple], Tuple]) -> None:
        """
        Build the NanoVLM and all its sub-layers.

        CRITICAL: Following modern Keras 3 patterns, explicitly build each sub-layer
        to ensure all weight variables exist before weight restoration during loading.

        :raises ValueError: If a static text length exceeds
            ``text_config['max_seq_len']``.
        """
        if self.built:
            return

        image_shape, text_shape = self._split_input_shape(input_shape)

        text_len = text_shape[1]
        max_text_len = self.text_config['max_seq_len']
        if text_len is not None and text_len > max_text_len:
            raise ValueError(
                f"text_tokens has {text_len} positions, more than "
                f"text_config['max_seq_len']={max_text_len}. Construct the "
                f"model with max_seq_len >= {text_len}, or truncate the text."
            )

        self.vision_encoder.build(image_shape)
        vision_shape = self.vision_encoder.compute_output_shape(image_shape)
        self.modality_projection.build(vision_shape)

        joint_len = None if text_len is None else self.num_vision_tokens + text_len
        self.text_decoder.build((text_shape[0], joint_len))

        hidden_shape = (text_shape[0], joint_len, self.text_config['embed_dim'])
        if self.final_dropout is not None:
            self.final_dropout.build(hidden_shape)
        if self.output_projection is not None:
            self.output_projection.build(hidden_shape)

        # Always call parent build at the end
        super().build(input_shape)

    @staticmethod
    def _split_input_shape(
            input_shape: Union[Dict[str, Tuple], Tuple]
    ) -> Tuple[Tuple, Tuple]:
        """Return ``(image_shape, text_shape)`` from a dict or a pair."""
        if isinstance(input_shape, dict):
            return input_shape['images'], input_shape['text_tokens']
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            return input_shape[0], input_shape[1]
        raise ValueError(
            "input_shape must be dict with 'images' and 'text_tokens' keys "
            "or tuple of (image_shape, text_shape)"
        )

    def call(
            self,
            inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None,
            **kwargs
    ) -> keras.KerasTensor:
        """
        Run the forward pass.

        :param inputs: Dict with ``'images'`` and ``'text_tokens'``, or a tuple
            ``(images, text_tokens)``. Optional key: ``'attention_mask'``
            ``(batch, T)`` over the text.
        :type inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Logits of shape ``(batch, num_vision_tokens + T, vocab_size)``.
        :rtype: keras.KerasTensor
        """
        if isinstance(inputs, dict):
            images = inputs['images']
            text_tokens = inputs['text_tokens']
            attention_mask = inputs.get('attention_mask')
        else:
            images, text_tokens = inputs
            attention_mask = None

        # DECISION plan-2026-08-19T163559-499b6f0e/D-084: no logging in call().
        # A `logger.debug` shape line under tf.function logs once at trace time only. See decisions.md.
        vision_tokens = self._vision_tokens(images, training=training)
        hidden = self._joint_hidden_states(
            vision_tokens, text_tokens, attention_mask, training=training
        )
        if self.final_dropout is not None:
            hidden = self.final_dropout(hidden, training=training)
        return self._vocab_logits(hidden)

    def _vision_tokens(
            self,
            images: keras.KerasTensor,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """Encode and project the images: ``(batch, num_vision_tokens, text_dim)``."""
        features = self.vision_encoder(images, training=training)
        return self.modality_projection(features, training=training)

    def _joint_hidden_states(
            self,
            vision_tokens: keras.KerasTensor,
            text_tokens: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor],
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """
        Run the causal decoder over ``[vision_tokens; embed(text_tokens)]``.

        Shared by :meth:`call` and :meth:`generate`, so the two cannot build
        the joint sequence differently.

        :param vision_tokens: ``(batch, Sv, text_dim)`` from :meth:`_vision_tokens`.
        :type vision_tokens: keras.KerasTensor
        :param text_tokens: ``(batch, T)`` token IDs.
        :type text_tokens: keras.KerasTensor
        :param attention_mask: Optional ``(batch, T)`` text padding mask; the
            vision prefix is prepended as all ones.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Hidden states ``(batch, Sv + T, text_dim)``.
        :rtype: keras.KerasTensor
        """
        text_embeddings = self.text_decoder.embed_tokens(text_tokens)
        joint = keras.ops.concatenate(
            [keras.ops.cast(vision_tokens, text_embeddings.dtype), text_embeddings],
            axis=1
        )
        if attention_mask is not None:
            attention_mask = keras.ops.convert_to_tensor(attention_mask)
            vision_mask = keras.ops.ones_like(vision_tokens[:, :, 0], dtype=attention_mask.dtype)
            attention_mask = keras.ops.concatenate([vision_mask, attention_mask], axis=1)
        return self.text_decoder.decode_embeddings(
            joint, attention_mask=attention_mask, training=training
        )

    def _vocab_logits(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """
        Project features to vocabulary logits, tied or untied.

        Shared by :meth:`call` and :meth:`generate`, so the two cannot disagree
        about where the logits come from.

        :param features: ``(batch, seq, embed_dim)``.
        :type features: keras.KerasTensor
        :return: Unnormalized logits ``(batch, seq, vocab_size)``; no softmax.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-06-15_2a23a001/D-001: tie embeddings at call time via
        # matmul against the transposed embedding table, never by reassigning a built layer's weight. See decisions.md.
        # DECISION plan-2026-09-12T123331-28fd855f/D-005: the matmul/transpose
        # EXPRESSION is consolidated into the shared `tied_embedding_logits`
        # helper (see decisions.md D-005 task 6); this does NOT contradict the
        # D-001 anchor above -- the tie is still resolved at call time via a
        # matmul against the transposed table, never by reassigning a built
        # layer's weight. See decisions.md D-005.
        if self.ties_output_to_embeddings:
            return tied_embedding_logits(
                features,
                self.text_decoder.word_embeddings.embeddings,
            )
        return self.output_projection(features)

    def generate(
            self,
            images: keras.KerasTensor,
            prompt_tokens: keras.KerasTensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            eos_token_id: int = 2,
            **kwargs
    ) -> keras.KerasTensor:
        """
        Generate text tokens autoregressively from images and a prompt.

        The images are encoded once; every step re-runs the decoder over
        ``[vision; text so far]`` (no key-value cache) and samples the next
        token from the logits at the last position. Each sequence finishes on
        its own ``eos_token_id``: a finished row keeps emitting
        ``eos_token_id``, and the loop ends when every row has finished or
        after ``max_length`` steps.

        :param images: Input images, shape ``(batch_size, height, width, channels)``.
        :type images: keras.KerasTensor
        :param prompt_tokens: Initial prompt tokens, shape ``(batch_size, prompt_length)``.
        :type prompt_tokens: keras.KerasTensor
        :param max_length: Maximum number of tokens to generate.
        :type max_length: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param top_k: Number of highest-probability tokens to sample from;
            ``0`` is greedy.
        :type top_k: int
        :param eos_token_id: Token ID that ends a sequence; also the fill of a
            finished row.
        :type eos_token_id: int
        :param kwargs: Unused, reserved for future generation parameters.
        :return: Generated token sequence, shape ``(batch_size, total_length)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``prompt_length + max_length`` exceeds
            ``text_config['max_seq_len']``: the decoder has no position for
            the excess tokens.
        """
        prompt_length = keras.ops.shape(prompt_tokens)[1]
        max_text_len = self.text_config['max_seq_len']
        if prompt_length + max_length > max_text_len:
            raise ValueError(
                f"prompt_length ({prompt_length}) + max_length ({max_length}) "
                f"exceeds text_config['max_seq_len']={max_text_len}"
            )
        # DECISION plan-2026-10-14T062616-81938e81/D-081: build() here is the
        # decoder's precondition, not a workaround. embed_tokens() and
        # decode_embeddings() require a BUILT TextDecoder (their contract);
        # called lazily they build its sub-layers but leave the decoder
        # unbuilt, and the model's next build() then re-builds them and Keras
        # refuses. Do not delete it (ModalityProjection.build is idempotent
        # now; the decoder is the reason this stays).
        if not self.built:
            self.build({'images': tuple(images.shape), 'text_tokens': (None, None)})

        # Process images once (cached for generation)
        vision_tokens = self._vision_tokens(images, training=False)

        current_tokens = prompt_tokens
        finished = keras.ops.zeros((keras.ops.shape(prompt_tokens)[0],), dtype="bool")
        for step in range(max_length):
            hidden = self._joint_hidden_states(
                vision_tokens, current_tokens, None, training=False
            )
            # The last position is the last TEXT position.
            next_token_logits = self._vocab_logits(hidden[:, -1:, :])[:, 0, :]

            # Sample next tokens for all sequences in batch
            next_tokens = self._sample_tokens_batch(
                next_token_logits, temperature, top_k
            )

            # A finished row keeps emitting EOS; a row finishes on its own EOS.
            next_tokens = keras.ops.where(
                finished, eos_token_id,
                keras.ops.cast(next_tokens, current_tokens.dtype))
            finished = keras.ops.logical_or(finished, next_tokens == eos_token_id)
            current_tokens = keras.ops.concatenate(
                [current_tokens, keras.ops.expand_dims(next_tokens, axis=1)], axis=1)

            if bool(keras.ops.all(finished)):
                break

        return current_tokens

    def _sample_tokens_batch(
            self,
            logits: keras.KerasTensor,
            temperature: float,
            top_k: int
    ) -> keras.KerasTensor:
        """Sample next tokens for a batch of sequences."""
        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            # Top-k sampling
            top_k_logits, top_k_indices = keras.ops.top_k(logits, k=top_k)
            # `categorical` takes LOGITS. Feeding it softmax probabilities
            # samples from softmax(softmax(z)), a flattened distribution.
            sampled_indices = keras.random.categorical(top_k_logits, num_samples=1)[:, 0]

            # Map back to original vocabulary
            next_tokens = keras.ops.take_along_axis(
                top_k_indices, keras.ops.expand_dims(sampled_indices, axis=1), axis=1
            )[:, 0]
        else:
            # Greedy sampling
            next_tokens = keras.ops.argmax(logits, axis=-1)

        return next_tokens

    def compute_output_shape(self, input_shape: Union[Dict, Tuple]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape for a given input shape.

        :param input_shape: Dict with ``'images'``/``'text_tokens'`` shapes, or
            a tuple of ``(image_shape, text_shape)``.
        :type input_shape: Union[Dict, Tuple]
        :return: Output shape ``(batch_size, num_vision_tokens + T, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        image_shape, text_shape = self._split_input_shape(input_shape)
        text_len = text_shape[1]
        joint_len = None if text_len is None else self.num_vision_tokens + text_len
        return (image_shape[0], joint_len, self.vocab_size)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            vocab_size: int = 32000,
            pretrained: Union[bool, str] = False,
            **kwargs: Any
    ) -> "NanoVLM":
        """
        Build a NanoVLM from a row of :attr:`MODEL_VARIANTS`.

        :param variant: One of ``'mini'``, ``'base'``, ``'large'``, ``'modern'``.
        :type variant: str
        :param vocab_size: Vocabulary size, written into the text config.
        :type vocab_size: int
        :param pretrained: Must be falsy. No NanoVLM weights are distributed,
            and the model has no backbone/head split to load partially: its
            output is the vocabulary logits of the whole model, tied to the
            word embedding table. ``True`` or a path raises
            ``NotImplementedError`` before anything is built; load a local
            archive with ``keras.models.load_model(path)``.
        :type pretrained: Union[bool, str]
        :param kwargs: Laid over the row. ``vision_config``, ``text_config``
            and ``projection_config`` are MERGED one level deep into the row's
            dicts (a key you pass replaces that key; the others stay); every
            other keyword replaces the row's value or goes to the constructor.
            An unknown keyword raises ``ValueError``.
        :type kwargs: Any
        :return: A configured, unbuilt model.
        :rtype: NanoVLM
        :raises ValueError: If ``variant`` is not a known row.
        :raises NotImplementedError: If ``pretrained`` is truthy.

        Example:

        .. code-block:: python

            model = NanoVLM.from_variant('mini', vocab_size=1000)
            small = NanoVLM.from_variant(
                'modern', vocab_size=64,
                vision_config={'embed_dim': 64, 'num_heads': 1},
                text_config={'embed_dim': 64, 'num_heads': 1},
            )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        refuse_pretrained(pretrained, "NanoVLM")

        # Deep-copied: the caller's values are written into the copy, and
        # writing them into the class table itself would make every later call
        # inherit this call's values.
        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        for key in ('vision_config', 'text_config', 'projection_config'):
            if key in kwargs:
                config[key] = {**config.get(key, {}), **copy.deepcopy(kwargs.pop(key))}
        config.update(kwargs)
        config['text_config']['vocab_size'] = vocab_size

        return cls(vocab_size=vocab_size, **config)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration for serialization.

        :return: Constructor arguments needed to reconstruct this model.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vision_config': self.vision_config,
            'text_config': self.text_config,
            'vocab_size': self.vocab_size,
            'projection_config': self.projection_config,
            'use_shared_embedding': self.use_shared_embedding,
            'output_dropout_rate': self.output_dropout_rate,
            'initializer_range': self.initializer_range,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def create_nanovlm(
        variant: str = "base",
        pretrained: Union[bool, str] = False,
        **kwargs: Any
) -> NanoVLM:
    """
    Build a NanoVLM from a named variant; delegates to :meth:`NanoVLM.from_variant`.

    :param variant: One of ``'mini'``, ``'base'``, ``'large'``, ``'modern'``.
    :type variant: str
    :param pretrained: Must be falsy; see :meth:`NanoVLM.from_variant`.
    :type pretrained: Union[bool, str]
    :param kwargs: Forwarded to :meth:`NanoVLM.from_variant`.
    :return: A configured model.
    :rtype: NanoVLM
    :raises ValueError: If ``variant`` is not a known preset.
    :raises NotImplementedError: If ``pretrained`` is truthy.

    Example:

    .. code-block:: python

        model = create_nanovlm('base', vocab_size=32000)
    """
    return NanoVLM.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------
