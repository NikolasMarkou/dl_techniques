"""
Score-based vision-language model: diffusion denoisers over frozen-width
encoder features, trained by denoising score matching and queried as a joint
score field.

An ordinary VLM is a conditional predictor — it maps an image to a caption, or a
caption to an image, one direction per trained head. This model is built on the
observation that both directions, and several things that are neither, are
readouts of one object: the score of the joint density, `grad_x log p(x)`.
Miyasawa's theorem (equivalently Tweedie's formula) is what makes that object
reachable without ever evaluating a density. If `x_t = x_0 + sigma * eps` and a
denoiser `D` is trained under plain MSE to recover `x_0`, then at the optimum

`grad_x log p(x_t) = (D(x_t) - x_t) / sigma^2`

so the residual of a denoiser *is* the score, up to a known scale. Training
reduces to regression against clean targets, and every generative behaviour
becomes a trajectory through the resulting vector field: run it in reverse from
noise to sample, query it at a point to ask which way probability increases, or
step along it while dragging one modality's coordinate to move the other.

The diffusion does not happen in pixel space or token space. Images pass through
a vision encoder and captions through a text encoder first, and the noise, the
denoisers and the whole reverse process operate on those `(B, seq, dim)` feature
sequences. That is the same trade latent diffusion makes — cost falls with the
dimensionality of the space being diffused, and the encoder already discards
detail the score field would otherwise have to model — but it has a consequence
that must not be glossed: **there is no pixel decoder in this package**.
``generate_from_text`` returns denoised *vision features*, not an image, and
turning them back into pixels requires a decoder that is not implemented here.
The image-to-text direction is complete by comparison, since a linear head over
the denoised text embeddings recovers token ids.

Both encoders are forced to sequence output (``output_mode='none'``) because the
conditional denoiser concatenates its condition along the sequence axis and
requires rank 3; the default CLS pooling would collapse that to rank 2.

The denoiser is not a U-Net. Noisy data and condition are each projected to a
common hidden width, a sinusoidal timestep embedding is added to the *data*
tokens only (broadcast across the sequence), the two are concatenated along the
sequence axis, residual MLP blocks with optional self-attention run over the
whole thing, and then only the data portion is sliced back out and projected
down. Conditioning therefore acts entirely through the concatenated prefix and
is discarded before the output. The final line is a residual:
``output = noisy_data + correction``, so the network learns a correction toward
the clean features rather than the features themselves — which is also the form
in which the Miyasawa residual is directly readable.

The scheduler follows the discrete DDPM convention. Timesteps are integer
indices in ``[0, num_timesteps)``, the forward process is
`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`, and the reverse
``step`` first recovers a predicted `x_0`, then takes the posterior mean
`q(x_{t-1} | x_t, x_0)`, then adds noise at every `t` except `t == 0`. Training
samples one timestep per batch element uniformly; inference walks
``np.linspace(T - 1, 0, num_inference_steps)`` on the host, so the loop is
Python-level and `t` reaches the scheduler as a Python scalar while reaching the
denoiser as a per-batch broadcast vector. Classifier-free guidance is done by
doubling the batch against a zeros condition and extrapolating between the two
predictions, rather than by a second forward pass.

One convention is load-bearing and is easier to see stated than derived. ``call``
supervises the denoisers against the *clean* features
(``target_vision``/``target_text``), so they are `x_0` predictors, and
``DiffusionScheduler`` therefore defaults to ``prediction_type='sample'`` rather
than to DDPM's usual ``'epsilon'`` — the shipped ``create_score_based_nanovlm``
presets pass no ``prediction_type`` at all, so that class default is what they
run. Setting ``'epsilon'`` in ``diffusion_config`` without also changing what
``call`` supervises makes ``step`` read an `x_0`-shaped output as noise, which
degrades samples silently instead of raising.

Weight materialization is explicit rather than lazy. The denoisers and the
decoder head would otherwise be built on first ``call``, which silently drops
roughly six hundred weights — the MultiHeadAttention and nested Sequential
blocks — on a ``.keras`` reload; ``build`` constructs them from the stored config
instead, and ``get_build_config``/``build_from_config`` carry the input shape so
the reload rebuilds the full sub-layer tree before weights are restored. The
scheduler is deliberately a plain Python class rather than a layer: it holds only
buffers derivable from ``diffusion_config``, so it is re-created from that dict
on load instead of being a serialized sub-object.

References:
    - Miyasawa, 1961. An empirical Bayes estimator of the mean of a normal
      population. Bulletin of the ISI 38.
    - Efron, 2011. Tweedie's Formula and Selection Bias. Journal of the American
      Statistical Association 106(496).
    - Vincent, 2011. A Connection Between Score Matching and Denoising
      Autoencoders. Neural Computation 23(7).
    - Ho et al., 2020. Denoising Diffusion Probabilistic Models.
      (https://arxiv.org/abs/2006.11239)
    - Song et al., 2020. Score-Based Generative Modeling through Stochastic
      Differential Equations. (https://arxiv.org/abs/2011.13456)
    - Nichol and Dhariwal, 2021. Improved Denoising Diffusion Probabilistic
      Models. (https://arxiv.org/abs/2102.09672)
    - Ho and Salimans, 2022. Classifier-Free Diffusion Guidance.
      (https://arxiv.org/abs/2207.12598)
    - Rombach et al., 2022. High-Resolution Image Synthesis with Latent
      Diffusion Models. (https://arxiv.org/abs/2112.10752)
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Dict, Optional, Tuple, Union, Any, Literal

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.layers.transformers.vision_encoder import create_vision_encoder

from .denoisers import VisionDenoiser, TextDenoiser, JointDenoiser
from .scheduler import DiffusionScheduler

@keras.saving.register_keras_serializable()
class ScoreBasedNanoVLM(keras.Model):
    """
    Score-Based nanoVLM: A Navigable Vision-Language World Model.

    Re-imagines VLMs through Miyasawa's theorem: instead of learning direct
    mappings (text→image or image→text), this model learns the score function
    ∇ log p(image, text) via Denoising Score Matching. All generative tasks
    become navigation of this learned semantic landscape via diffusion.

    **Theoretical Foundation:**
    By Miyasawa's theorem (Tweedie's formula), an optimal denoiser D(x_t, c, t)
    trained via MSE provides the score function:
        ∇_x log p(x_t | c) = (1/σ²) * (D(x_t, c, t) - x_t)

    This transforms VLM training from supervised prediction to learning a
    "physics of meaning" - a vector field defining how concepts attract and
    repel in semantic space.

    **Three Operational Modes:**
    1. **Text-to-Image**: Navigate p(image | text) via reverse diffusion
    2. **Image-to-Text**: Navigate p(text | image) in embedding space
    3. **Joint Reasoning**: Traverse the unified field ∇ log p(image, text)

    Args:
        vision_config: Configuration dict for vision processing.
        text_config: Configuration dict for text processing.
        diffusion_config: Configuration for diffusion scheduler.
        vocab_size: Vocabulary size for text. Defaults to 32000.
        generation_mode: Which generative mode to enable
            ('text_to_image', 'image_to_text', 'joint'). Defaults to 'joint'.
        use_classifier_free_guidance: Enable CFG for stronger conditioning.
            Defaults to True.
        **kwargs: Additional model arguments.

    Example:
        ```python
        # Create score-based VLM
        model = ScoreBasedNanoVLM(
            vision_config={'img_size': 224, 'embed_dim': 768, ...},
            text_config={'vocab_size': 32000, 'embed_dim': 768, ...},
            diffusion_config={'num_timesteps': 1000, 'beta_schedule': 'cosine'},
            generation_mode='joint'
        )

        # Training: Learn to denoise
        with tf.GradientTape() as tape:
            # Add noise to clean data
            noisy_vision, noise, timesteps = scheduler.add_noise(clean_vision, ...)

            # Denoise
            denoised = model.denoise_vision(noisy_vision, text_features, timesteps)

            # DSM loss: ||D(x_t, c) - x_0||²
            loss = mse(denoised, clean_vision)

        # Generation: Navigate score field
        generated_image = model.generate_from_text(text_prompt, num_steps=50)
        generated_text = model.generate_from_image(image, num_steps=50)
        ```
    """

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_config: Dict[str, Any],
            diffusion_config: Dict[str, Any],
            vocab_size: int = 32000,
            generation_mode: Literal['text_to_image', 'image_to_text', 'joint'] = 'joint',
            use_classifier_free_guidance: bool = True,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Store configurations
        self.vision_config = vision_config
        self.text_config = text_config
        self.diffusion_config = diffusion_config
        self.vocab_size = vocab_size
        self.generation_mode = generation_mode
        self.use_classifier_free_guidance = use_classifier_free_guidance

        # Validate dimensions
        vision_dim = vision_config.get('embed_dim', 768)
        text_dim = text_config.get('embed_dim', 768)

        logger.info(f"Initializing Score-Based nanoVLM in '{generation_mode}' mode")
        logger.info(f"  Vision dim: {vision_dim}, Text dim: {text_dim}")

        # === Core Encoding Components ===
        # Vision encoder (processes clean images to features)
        # Defensive: the denoisers' ConditionalDenoiser.call concatenates the
        # condition along axis=1 and requires a rank-3 [B, seq, D] sequence; the
        # default 'cls' pooling collapses to rank-2. Force sequence output.
        # Mirrors nano_vlm/model.py:334-336.
        if vision_config.get('output_mode', 'cls') != 'none':
            vision_config = dict(vision_config, output_mode='none')
        self.vision_config = vision_config
        self.vision_encoder = create_vision_encoder(**vision_config)

        # Text encoder (processes text to embeddings)
        self.text_encoder = TextEncoder(**text_config, name='text_encoder')

        # === Diffusion Scheduler ===
        # DiffusionScheduler is a plain Python class (not a Layer); it takes no `name`
        # kwarg. Re-created here from diffusion_config (inlined into get_config), so the
        # model round-trips without the scheduler being a serialized sub-layer. See
        # DECISION plan_2026-06-13_ae9ee2cd/D-004 in scheduler.py.
        self.scheduler = DiffusionScheduler(**diffusion_config)

        # === Denoiser Networks (The Core Innovation) ===


        if generation_mode in ['text_to_image', 'joint']:
            self.vision_denoiser = VisionDenoiser(
                vision_config=vision_config,
                text_dim=text_dim,
                num_layers=12,
                name='vision_denoiser'
            )

        if generation_mode in ['image_to_text', 'joint']:
            self.text_denoiser = TextDenoiser(
                text_dim=text_dim,
                vision_dim=vision_dim,
                num_layers=12,
                name='text_denoiser'
            )

        if generation_mode == 'joint':
            self.joint_denoiser = JointDenoiser(
                vision_dim=vision_dim,
                text_dim=text_dim,
                hidden_dim=max(vision_dim, text_dim),
                num_layers=16,
                name='joint_denoiser'
            )

        # === Output Heads ===
        # For image-to-text, we need to decode embeddings to tokens
        if generation_mode in ['image_to_text', 'joint']:
            self.text_decoder_head = layers.Dense(
                vocab_size,
                use_bias=False,
                kernel_initializer='glorot_uniform',
                name='text_decoder_head'
            )

        logger.info("Score-Based nanoVLM initialized successfully")

    def build(self, input_shape: Union[Dict, Tuple]) -> None:
        """Build all components, materializing every sub-layer's weights.

        The denoisers and the text-decoder head are otherwise built lazily on
        the first ``call``, which silently drops ~600 weights (MultiHeadAttention
        + nested Sequential blocks) on a ``.keras`` reload. Building them
        explicitly here pins variable creation to the build phase so the
        round-trip preserves every weight (M2). All denoiser weight shapes derive
        from the stored config, so the exact ``input_shape`` content is not
        required for them.

        Args:
            input_shape: Either a dict ``{'images': ..., 'text': ...}`` or a
                ``(vision_shape, text_shape)`` tuple.
        """
        if self.built:
            return

        # Remember the build shape so the .keras reload (build_from_config)
        # rebuilds the full sub-layer tree before weights are restored.
        self._build_input_shape = input_shape

        # Build encoders
        if isinstance(input_shape, dict):
            vision_shape = input_shape.get('images')
            text_shape = input_shape.get('text')
        else:
            vision_shape, text_shape = input_shape

        if vision_shape is not None:
            self.vision_encoder.build(vision_shape)

        if text_shape is not None:
            self.text_encoder.build({'input_ids': text_shape})

        # Build denoisers + output head explicitly (shapes from config).
        vision_dim = self.vision_config.get('embed_dim', 768)
        text_dim = self.text_config.get('embed_dim', 768)
        if self.generation_mode in ['text_to_image', 'joint']:
            self.vision_denoiser.build((None, None, vision_dim))
        if self.generation_mode in ['image_to_text', 'joint']:
            self.text_denoiser.build((None, None, text_dim))
        if self.generation_mode == 'joint':
            self.joint_denoiser.build((None, None, vision_dim))
        if self.generation_mode in ['image_to_text', 'joint']:
            self.text_decoder_head.build((None, None, text_dim))

        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        """Return the build config so reload rebuilds the full sub-layer tree."""
        return {'input_shape': getattr(self, '_build_input_shape', None)}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild from :meth:`get_build_config` before weight restore."""
        input_shape = config.get('input_shape')
        if input_shape is not None:
            self.build(input_shape)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass for training.

        During training, we perform Denoising Score Matching:
        1. Encode clean data to features
        2. Add noise according to diffusion schedule
        3. Denoise and compare to clean target

        Args:
            inputs: Dictionary containing:
                - 'images': Clean images [batch, H, W, C]
                - 'text': Text token IDs [batch, seq_len]
                - 'timesteps': Random timesteps for DSM [batch] (optional)
            training: Training mode flag

        Returns:
            Dictionary with denoised outputs for loss computation
        """
        images = inputs['images']
        text_tokens = inputs.get('text')
        timesteps = inputs.get('timesteps')

        # Encode clean data
        vision_features = self.vision_encoder(images, training=training)

        if text_tokens is not None:
            text_features = self.text_encoder(
                {'input_ids': text_tokens}, training=training
            )
        else:
            text_features = None

        # Sample random timesteps if not provided
        if timesteps is None:
            batch_size = ops.shape(images)[0]
            timesteps = keras.random.randint(
                shape=(batch_size,), minval=0,
                maxval=self.scheduler.num_timesteps
            )

        outputs = {}

        # Add noise and denoise based on mode
        if self.generation_mode in ['text_to_image', 'joint']:
            # Text-to-Image: Denoise vision conditioned on text
            noise_vision = keras.random.normal(ops.shape(vision_features))
            noisy_vision = self.scheduler.add_noise(vision_features, noise_vision, timesteps)

            denoised_vision = self.vision_denoiser(
                noisy_vision, text_features, timesteps, training=training
            )
            outputs['denoised_vision'] = denoised_vision
            # DECISION plan-2026-08-17T183311-79c63e38/D-029: the DSM regression
            # target is DETACHED. It is the vision encoder's own trainable output,
            # and ||D(x_t) - x_0||^2 with a trainable x_0 is globally minimised by a
            # CONSTANT encoder — which the zero-initialised output_proj already
            # predicts exactly. Do NOT remove stop_gradient to "let the encoder learn
            # from the diffusion loss": that is the representation-collapse setup, and
            # the encoder still receives gradient through the denoiser INPUT path
            # (add_noise is differentiable in vision_features). Precedent:
            # video_jepa/model.py:439. See decisions.md D-029.
            outputs['target_vision'] = ops.stop_gradient(vision_features)
            outputs['noise_vision'] = noise_vision

        if self.generation_mode in ['image_to_text', 'joint']:
            # Image-to-Text: Denoise text embeddings conditioned on vision
            noise_text = keras.random.normal(ops.shape(text_features))
            noisy_text = self.scheduler.add_noise(text_features, noise_text, timesteps)

            denoised_text = self.text_denoiser(
                noisy_text, vision_features, timesteps, training=training
            )
            outputs['denoised_text'] = denoised_text
            # Detached for the same reason as target_vision above (D-029).
            outputs['target_text'] = ops.stop_gradient(text_features)
            outputs['noise_text'] = noise_text

        if self.generation_mode == 'joint':
            # Joint: Denoise both simultaneously
            noise_v = keras.random.normal(ops.shape(vision_features))
            noise_t = keras.random.normal(ops.shape(text_features))
            noisy_v = self.scheduler.add_noise(vision_features, noise_v, timesteps)
            noisy_t = self.scheduler.add_noise(text_features, noise_t, timesteps)

            denoised_v, denoised_t = self.joint_denoiser(
                noisy_v, noisy_t, timesteps, training=training
            )
            outputs['joint_denoised_vision'] = denoised_v
            outputs['joint_denoised_text'] = denoised_t
            # Detached for the same reason as target_vision above (D-029).
            outputs['joint_target_vision'] = ops.stop_gradient(vision_features)
            outputs['joint_target_text'] = ops.stop_gradient(text_features)

        outputs['timesteps'] = timesteps
        return outputs

    def generate_from_text(
            self,
            text_features: keras.KerasTensor,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            generator: Optional[Any] = None
    ) -> keras.KerasTensor:
        """
        Generate images from text via reverse diffusion (Protocol 1).

        Implements the reverse-time SDE: starting from noise, iteratively
        denoise by following the score field ∇ log p(image | text).

        Args:
            text_features: Text conditioning [batch, seq_len, text_dim]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance strength (>1 for stronger)
            generator: Random generator for reproducibility

        Returns:
            Generated images [batch, H, W, C]
        """
        if not hasattr(self, 'vision_denoiser'):
            raise ValueError("Model not configured for text-to-image generation")

        batch_size = ops.shape(text_features)[0]

        # Get vision feature shape from encoder. The probe MUST follow the
        # configured img_size: the encoder's positional table is sized from it, so a
        # hardcoded 224 dies inside PositionalEmbedding at every other resolution.
        img_size = self.vision_config.get('img_size', 224)
        dummy_img = keras.random.normal((1, img_size, img_size, 3))
        vision_feat_shape = ops.shape(self.vision_encoder(dummy_img, training=False))
        seq_len, feat_dim = vision_feat_shape[1], vision_feat_shape[2]

        # Start from pure noise
        latent_shape = (batch_size, seq_len, feat_dim)
        latents = keras.random.normal(latent_shape)

        # Timestep schedule for inference
        timesteps = np.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=np.int32
        )

        # Reverse diffusion loop
        for i, t in enumerate(timesteps):
            t_tensor = ops.convert_to_tensor([t] * batch_size, dtype='int32')

            # Predict noise
            if self.use_classifier_free_guidance and guidance_scale != 1.0:
                # Classifier-Free Guidance: interpolate conditional and unconditional
                # Unconditional: use zero/null text features
                null_text = ops.zeros_like(text_features)

                # Concatenate for parallel processing
                latent_input = ops.concatenate([latents, latents], axis=0)
                text_input = ops.concatenate([text_features, null_text], axis=0)
                t_input = ops.concatenate([t_tensor, t_tensor], axis=0)

                noise_pred = self.vision_denoiser(
                    latent_input, text_input, t_input, training=False
                )

                # Split and apply guidance
                noise_cond, noise_uncond = ops.split(noise_pred, 2, axis=0)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.vision_denoiser(
                    latents, text_features, t_tensor, training=False
                )

            # DECISION plan-2026-08-14T183218-f4c612aa/D-016: hand the denoiser's raw
            # output straight to `step`. Do NOT reinstate a
            # `predict_noise_from_start` conversion here "for symmetry" with the
            # 'epsilon' convention: `step`'s 'sample' branch consumes x_0 DIRECTLY
            # (`pred_original_sample = model_output`), so converting first makes it
            # read the noise AS the clean sample and drives the whole reverse process
            # off the wrong quantity — measured max deviation 3.86 at t=50, silently,
            # on the path all three shipped presets take. `step` dispatches on
            # `prediction_type` itself; the caller must not pre-translate.
            latents, _ = self.scheduler.step(noise_pred, t, latents)

        # Decode latents to images (this would need a decoder)
        # For now, return the latent representation
        logger.info(f"Generated vision features via {num_inference_steps} diffusion steps")
        return latents

    def generate_from_image(
            self,
            vision_features: keras.KerasTensor,
            num_inference_steps: int = 50,
            max_length: int = 77,
            guidance_scale: float = 3.0
    ) -> keras.KerasTensor:
        """
        Generate text from images via latent diffusion (Protocol 2).

        Instead of autoregressive token-by-token generation, this performs
        holistic generation by denoising a text embedding, then decoding it.
        This avoids error propagation and enables semantic manipulation.

        Args:
            vision_features: Vision conditioning [batch, seq_len, vision_dim]
            num_inference_steps: Number of denoising steps
            max_length: Maximum text sequence length
            guidance_scale: Guidance strength

        Returns:
            Generated text embeddings [batch, max_length, text_dim]
        """
        if not hasattr(self, 'text_denoiser'):
            raise ValueError("Model not configured for image-to-text generation")

        batch_size = ops.shape(vision_features)[0]
        text_dim = self.text_config['embed_dim']

        # Start from noise in text embedding space
        latents = keras.random.normal((batch_size, max_length, text_dim))

        # Inference timestep schedule
        timesteps = np.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=np.int32
        )

        # Reverse diffusion in text space
        for i, t in enumerate(timesteps):
            t_tensor = ops.convert_to_tensor([t] * batch_size, dtype='int32')

            if self.use_classifier_free_guidance and guidance_scale != 1.0:
                null_vision = ops.zeros_like(vision_features)
                latent_input = ops.concatenate([latents, latents], axis=0)
                vision_input = ops.concatenate([vision_features, null_vision], axis=0)
                t_input = ops.concatenate([t_tensor, t_tensor], axis=0)

                text_pred = self.text_denoiser(
                    latent_input, vision_input, t_input, training=False
                )

                text_cond, text_uncond = ops.split(text_pred, 2, axis=0)
                text_pred = text_uncond + guidance_scale * (text_cond - text_uncond)
            else:
                text_pred = self.text_denoiser(
                    latents, vision_features, t_tensor, training=False
                )

            # Step
            latents, _ = self.scheduler.step(text_pred, t, latents)

        logger.info(f"Generated text embeddings via {num_inference_steps} diffusion steps")

        # Decode embeddings to tokens
        logits = self.text_decoder_head(latents)
        tokens = ops.argmax(logits, axis=-1)

        return tokens

    def compute_score_field(
            self,
            vision_features: keras.KerasTensor,
            text_features: keras.KerasTensor,
            timestep: int
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the joint score field ∇ log p(image, text) at a point.

        This is the core of Protocol 3: treating the VLM as a unified world
        model whose score field can be queried at any (image, text) coordinate.

        Args:
            vision_features: Vision point [batch, vision_seq, vision_dim]
            text_features: Text point [batch, text_seq, text_dim]
            timestep: Noise level to query at

        Returns:
            Tuple of (vision_score, text_score) representing ∇ log p
        """
        if not hasattr(self, 'joint_denoiser'):
            raise ValueError("Model not configured for joint score computation")

        batch_size = ops.shape(vision_features)[0]
        t = ops.convert_to_tensor([timestep] * batch_size, dtype='int32')

        # Denoise to get predicted clean samples
        denoised_v, denoised_t = self.joint_denoiser(
            vision_features, text_features, t, training=False
        )

        # DECISION plan-2026-08-17T183311-79c63e38/D-029: the denoisers are x_0
        # predictors, so their output must be converted to an epsilon estimate
        # BEFORE get_score_from_noise, which consumes epsilon and nothing else.
        # This used to pass `denoised - noisy` straight in, which is not an epsilon
        # by any parameterisation; the result was `-(D - x_t)/sqrt(1-a_t)`, exactly
        # OPPOSITE the true score at low noise (measured cosine -1.0 at t=2) and
        # wrong in scale everywhere, so navigate_semantic_space did gradient
        # DESCENT on log p. Do NOT "simplify" this back by negating in place: in
        # this VP schedule Tweedie is `(sqrt(a_t)*D - x_t)/(1 - a_t)`, which carries
        # a sqrt(a_t) on D and a (1-a_t) denominator; the familiar `(D - x)/sigma^2`
        # form is the variance-exploding special case a_t = 1 and is wrong for every
        # t > 0 here. predict_noise_from_start is that conversion and is reused
        # rather than re-derived. See decisions.md D-029.
        vision_score = self.scheduler.get_score_from_noise(
            self.scheduler.predict_noise_from_start(vision_features, t, denoised_v),
            t, vision_features
        )
        text_score = self.scheduler.get_score_from_noise(
            self.scheduler.predict_noise_from_start(text_features, t, denoised_t),
            t, text_features
        )

        return vision_score, text_score

    def navigate_semantic_space(
            self,
            start_vision: keras.KerasTensor,
            start_text: keras.KerasTensor,
            target_text: keras.KerasTensor,
            num_steps: int = 100,
            step_size: float = 0.01
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Navigate from one point to another in semantic space (Protocol 3).

        This implements "semantic calculus" - using the score field to
        traverse the manifold from one concept to another while staying in
        high-probability regions.

        Example: Given (image, "daytime"), navigate to (?, "nighttime")

        Args:
            start_vision: Starting vision features
            start_text: Starting text features
            target_text: Target text concept
            num_steps: Number of navigation steps
            step_size: Step size for gradient ascent

        Returns:
            Final (vision, text) coordinates after navigation
        """
        current_v = start_vision
        current_t = start_text

        # Compute direction in text space
        text_direction = target_text - start_text
        text_direction = text_direction / ops.norm(text_direction)

        # Navigate via gradient ascent on score field
        for step in range(num_steps):
            # Query score at current position
            timestep = self.scheduler.num_timesteps // 2  # Mid-noise level
            score_v, score_t = self.compute_score_field(current_v, current_t, timestep)

            # Move towards target while following score field
            # Text: move towards target
            current_t = current_t + step_size * text_direction
            # Vision: follow score (gradient ascent)
            current_v = current_v + step_size * score_v

            # Optional: project back onto data manifold periodically
            if step % 10 == 0:
                # Denoise slightly to stay on manifold
                t_denoise = ops.convert_to_tensor(
                    [timestep] * ops.shape(current_v)[0], dtype='int32'
                )
                current_v, current_t = self.joint_denoiser(
                    current_v, current_t, t_denoise, training=False
                )

        logger.info(f"Navigated semantic space in {num_steps} steps")
        return current_v, current_t

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'vision_config': self.vision_config,
            'text_config': self.text_config,
            'diffusion_config': self.diffusion_config,
            'vocab_size': self.vocab_size,
            'generation_mode': self.generation_mode,
            'use_classifier_free_guidance': self.use_classifier_free_guidance,
        })
        return config


# === Factory Functions ===

def create_score_based_nanovlm(
        variant: Literal['mini', 'base', 'large'] = 'base',
        mode: Literal['text_to_image', 'image_to_text', 'joint'] = 'joint',
        vocab_size: int = 32000,
        **kwargs
) -> ScoreBasedNanoVLM:
    """
    Create a score-based nanoVLM with predefined configurations.

    Args:
        variant: Model size ('mini', 'base', 'large')
        mode: Generation mode
        vocab_size: Vocabulary size
        **kwargs: Additional arguments

    Returns:
        Configured ScoreBasedNanoVLM
    """
    configs = {
        'mini': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6, 'output_mode': 'none'
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6, 'max_seq_len': 512
            },
            'diffusion_config': {
                'num_timesteps': 1000, 'beta_schedule': 'cosine'
            }
        },
        'base': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'output_mode': 'none'
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'max_seq_len': 512
            },
            'diffusion_config': {
                'num_timesteps': 1000, 'beta_schedule': 'cosine'
            }
        },
        'large': {
            'vision_config': {
                'img_size': 384, 'patch_size': 16, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16, 'output_mode': 'none'
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16, 'max_seq_len': 1024
            },
            'diffusion_config': {
                'num_timesteps': 1000, 'beta_schedule': 'cosine'
            }
        }
    }

    config = configs[variant]

    return ScoreBasedNanoVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        diffusion_config=config['diffusion_config'],
        vocab_size=vocab_size,
        generation_mode=mode,
        **kwargs
    )