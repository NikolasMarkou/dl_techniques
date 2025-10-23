"""
Free Transformer Layer with integrated Variational Autoencoder components.

This module implements "The Free Transformer" architecture proposed by François Fleuret
(arXiv:2510.17558v1). It extends the standard decoder Transformer by conditioning its
generative process on random latent variables learned without supervision through a
variational procedure.

The Free Transformer addresses limitations of purely auto-regressive models by allowing
the model to make explicit latent decisions about the structure of the sequence being
generated, rather than implicitly inferring these decisions from already-generated tokens.

Architecture Overview:
---------------------
The Free Transformer splits a standard decoder into two halves and injects a latent
variable Z at the middle layer:

1. **First Half**: Standard causal transformer blocks (layers 0 to L/2-1)

2. **Injection Layer** (layer L/2):
   - During training: An encoder infers Z from the sequence
   - During inference: Z is sampled uniformly from a categorical distribution
   - Z is projected and added to the key/value inputs of this layer's attention

3. **Second Half**: Standard causal transformer blocks (layers L/2+1 to L-1)

Encoder Architecture:
--------------------
The encoder (only active during training or KV cache pre-filling) consists of:
- A learned constant query vector ζ (zeta) replicated across the sequence
- One non-causal transformer block receiving ζ as queries and the first-half
  output as keys/values
- A linear readout producing H bit logits per token
- A Binary Mapper that samples Z as a one-hot vector of dimension 2^H

Mathematical Foundation:
-----------------------
The model learns to maximize P(S) = ∫ P(S|Z=z)P(Z=z)dz through a VAE objective:

    Loss = CrossEntropy(S) + β * max(0, KL(Q(Z|S) || P(Z)) - κ)

where:
- Q(Z|S) is the encoder's posterior distribution
- P(Z) is a uniform prior over 2^H categories
- κ is a "free bits" threshold to prevent KL collapse
- β is a weighting coefficient

References:
-----------
- Fleuret, F. (2025). The Free Transformer. arXiv:2510.17558v1.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
"""

import keras
import tensorflow as tf
from keras import layers
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .transformer import TransformerLayer
from ..ffn.factory import create_ffn_from_config, FFNType
from ..norms import create_normalization_layer, NormalizationType
from ..attention.factory import create_attention_layer, AttentionType

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BinaryMapper(keras.layers.Layer):
    """
    Samples one-hot vectors from bit logits with gradient pass-through.

    This layer implements the Binary Mapper described in Section 3.4 of "The Free
    Transformer" paper. It converts H independent bit logits into a one-hot vector
    of dimension 2^H through the following process:

    1. Interpret input logits as H independent Bernoulli distributions
    2. Sample H bits: B_h ~ Bernoulli(sigmoid(logit_h))
    3. Convert bit vector to integer: index = Σ(B_h * 2^h)
    4. Create one-hot vector: Z[index] = 1, Z[others] = 0
    5. Add gradient pass-through for training (Equation 8 in paper)

    **Architecture Flow**:
    ```
    Input: Bit Logits (B, T, H)
           ↓
    Sigmoid → Probabilities p_h = σ(logit_h)
           ↓
    Sample Binary Bits: B_h ~ Bernoulli(p_h)
           ↓
    Binary-to-Integer: index = Σ(B_h × 2^h)
           ↓
    One-Hot Encoding: Z[index] = 1
           ↓
    [Training Only] Gradient Pass-Through
           ↓
    Z ← Z + [G(Z) - stop_gradient(G(Z))]
    where G(Z) = exp(Σ log P(B_h))
           ↓
    Output: One-Hot Z (B, T, 2^H)
    ```

    The gradient pass-through mechanism enables backpropagation despite the discrete
    sampling operation. It works by computing the probability of the sampled vector
    under the current logits and using a straight-through estimator:

        output = one_hot(sampled_index) + [P(sampled) - stop_gradient(P(sampled))]

    This adds the gradient of P(sampled) without changing the forward pass value.

    Args:
        num_bits: Integer, number of latent bits H. The output dimension will be 2^H.
            Paper uses H=16, giving 65,536 latent categories.
        **kwargs: Additional layer arguments.

    Input Shape:
        (batch_size, sequence_length, num_bits): Bit logits for each token

    Output Shape:
        (batch_size, sequence_length, 2^num_bits): One-hot encoded latent variables

    Example:
        ```python
        # Create mapper for 4 bits (16 categories)
        mapper = BinaryMapper(num_bits=4)

        # Sample from logits
        bit_logits = keras.random.normal((2, 10, 4))  # (batch, seq, bits)
        z_one_hot = mapper(bit_logits, training=True)  # (2, 10, 16)

        # z_one_hot contains one-hot vectors with gradient flow
        ```

    Note:
        The monotonicity of the sigmoid function ensures stable gradients through
        the binary encoding, which is why binary encoding is preferred over directly
        outputting 2^H logits.
    """

    def __init__(
            self,
            num_bits: int,
            **kwargs: Any
    ):
        super().__init__(**kwargs)

        if not isinstance(num_bits, int) or num_bits <= 0:
            raise ValueError(
                f"num_bits must be a positive integer, got {num_bits}"
            )
        if num_bits > 20:
            raise ValueError(
                f"num_bits={num_bits} is too large (>20), would create 2^{num_bits} "
                f"= {2**num_bits} categories and consume excessive memory"
            )

        self.num_bits = num_bits
        self.num_categories = 2 ** num_bits

        # Powers of 2 for binary-to-integer conversion: [1, 2, 4, 8, ...]
        # Shape: (num_bits,)
        self.pow2 = tf.constant(
            [2 ** i for i in range(num_bits)],
            dtype=tf.int32,
            name='binary_powers'
        )

    def call(
            self,
            bit_logits: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: sample one-hot vectors from bit logits.

        Args:
            bit_logits: Tensor of shape (batch_size, sequence_length, num_bits)
                containing logits for each independent bit.
            training: Boolean flag for training mode. When True, applies gradient
                pass-through mechanism. When False, uses standard sampling.

        Returns:
            One-hot tensor of shape (batch_size, sequence_length, 2^num_bits).
            During training, includes gradient pass-through via straight-through
            estimator.
        """
        # Step 1: Compute bit probabilities
        # P(B_h = 1) = sigmoid(logit_h)
        probs = keras.ops.sigmoid(bit_logits)

        # Step 2: Sample bits using reparameterization trick
        # Sample uniform values and threshold by probability
        uniform_sample = keras.random.uniform(
            keras.ops.shape(probs),
            dtype=probs.dtype
        )
        sampled_bits = keras.ops.cast(
            uniform_sample < probs,
            dtype='int32'
        )

        # Step 3: Convert binary vector to integer index
        # index = Σ(B_h * 2^h) for h in [0, H-1]
        # Using einsum for efficient computation: (B, T, H) @ (H,) -> (B, T)
        indices = keras.ops.einsum('bth,h->bt', sampled_bits, self.pow2)

        # Step 4: Create one-hot encoding
        # Shape: (batch_size, sequence_length, 2^H)
        z_one_hot = keras.ops.one_hot(
            indices,
            num_classes=self.num_categories,
            dtype=self.compute_dtype
        )

        # Step 5: Gradient pass-through (Equation 8 in paper)
        if training:
            # Compute log probability of each bit choice
            # Using sigmoid_cross_entropy for numerical stability:
            # log P(B_h) = -sigmoid_cross_entropy(label=B_h, logit=logit_h)
            sampled_bits_float = keras.ops.cast(sampled_bits, self.compute_dtype)

            # Note: We need to use TensorFlow's sigmoid_cross_entropy for stability
            log_probs = -tf.nn.sigmoid_cross_entropy_with_logits(
                labels=sampled_bits_float,
                logits=bit_logits
            )

            # Sum log probabilities across bits to get log P(B_t = U(d-1))
            # where U(d-1) is the binary encoding of the sampled index
            # G_t,d = exp(Σ_h log P(B_t,h = U_h(d-1)))
            log_prob_sum = keras.ops.sum(log_probs, axis=-1, keepdims=True)
            g_td = keras.ops.exp(log_prob_sum)

            # Apply straight-through estimator: Y + [G - stop_gradient(G)]
            # This adds ∇G to the backward pass without changing the forward value
            z_one_hot = z_one_hot + g_td - keras.ops.stop_gradient(g_td)

        return z_one_hot

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape: replace last dimension with 2^num_bits.

        Args:
            input_shape: Input shape tuple (batch, sequence, num_bits)

        Returns:
            Output shape tuple (batch, sequence, 2^num_bits)
        """
        return input_shape[:-1] + (self.num_categories,)

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration."""
        config = super().get_config()
        config.update({
            'num_bits': self.num_bits
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FreeTransformerLayer(TransformerLayer):
    """
    A Transformer layer extended with the Free Transformer C-VAE architecture.

    This layer can function as either:
    1. A standard transformer block (when use_free_transformer=False)
    2. The "injection layer" of a Free Transformer (when use_free_transformer=True)

    When functioning as the injection layer, it:
    - Runs an encoder during training to infer latent variable Z from the sequence
    - Samples Z uniformly during inference
    - Modifies the attention mechanism to condition on Z by injecting it into keys/values

    **Full Model Architecture** (showing injection at layer L/2):
    ```
    ┌─────────────────────────────────────────────────┐
    │            Free Transformer Model               │
    ├─────────────────────────────────────────────────┤
    │ Input Sequence S                                │
    │       ↓                                         │
    │ Embedding Layer                                 │
    │       ↓                                         │
    │ ┌──────────────────────────────┐                │
    │ │  First Half (Layers 0..L/2-1)│                │
    │ │  - Standard Transformer Blocks│               │
    │ │  - Causal Attention           │               │
    │ │  - Feed-Forward Networks      │               │
    │ └──────────┬───────────────────┘                │
    │            ↓                                    │
    │ ┌──────────────────────────────────────────┐    │
    │ │    INJECTION LAYER (L/2)                 │    │
    │ │  ┌──────────────┐  ┌──────────────────┐  │    │
    │ │  │   Encoder    │  │    Decoder       │  │    │
    │ │  │  (Training)  │  │   (Always)       │  │    │
    │ │  │      ↓       │  │      ↓           │  │    │
    │ │  │  Infer Z     │  │  Attention       │  │    │
    │ │  │      ↓       │  │      +           │  │    │
    │ │  └──────┬───────┘  │  Inject Z        │  │    │
    │ │         ↓           │      ↓          │  │    │
    │ │    [Sample Z] ←─────┤  FFN            │  │    │
    │ │                     └────────┬─────────┘ │    │
    │ └─────────────────────────────┼────────────┘    │
    │                               ↓                 │
    │ ┌──────────────────────────────┐                │
    │ │ Second Half (Layers L/2+1..L)│                │
    │ │  - Standard Transformer Blocks│               │
    │ │  - Causal Attention           │               │
    │ │  - Feed-Forward Networks      │               │
    │ └──────────┬───────────────────┘                │
    │            ↓                                    │
    │ Output Logits                                   │
    └─────────────────────────────────────────────────┘
    ```

    **Injection Layer Architecture** (Pre-Normalization):
    ```
    ═══════════════════════════════════════════════════════════
                        INJECTION LAYER L/2
    ═══════════════════════════════════════════════════════════

    Input: X (B, T, D)
           ↓
    ┌──────────────────────────────────────────────────────┐
    │              ATTENTION SUB-LAYER                     │
    ├──────────────────────────────────────────────────────┤
    │  Residual ← X                                        │
    │      ↓                                               │
    │  Normalize(X)                                        │
    │      ↓                                               │
    │  Multi-Head Causal Attention                         │
    │      ↓                                               │
    │  Dropout                                             │
    │      ↓                                               │
    │  Add(Residual)                                       │
    │      ↓                                               │
    │  Attention Output → X_attn (B, T, D)                 │
    └──────────┬───────────────────────────────────────────┘
               ↓
    ┌──────────────────────────────────────────────────────┐
    │          ENCODER PATH (Training/Prefill Only)        │
    ├──────────────────────────────────────────────────────┤
    │  Learned Query ζ (1, 1, D)                           │
    │      ↓                                               │
    │  Tile to (B, T, D)                                   │
    │      ↓                                               │
    │  Normalize(ζ)                                        │
    │      ↓                                               │
    │  Non-Causal Cross-Attention:                         │
    │    Q = ζ (normalized)                                │
    │    K, V = X_attn (from above)                        │
    │      ↓                                               │
    │  Dropout + Add(ζ)                                    │
    │      ↓                                               │
    │  Normalize + FFN + Dropout + Residual                │
    │      ↓                                               │
    │  Linear Readout: D → H bits                          │
    │      ↓                                               │
    │  bit_logits (B, T, H)                                │
    │      ↓                                               │
    │  Binary Mapper: H bits → 2^H one-hot                 │
    │      ↓                                               │
    │  Z (B, T, 2^H) ───────────────────┐                  │
    └───────────────────────────────────┼──────────────────┘
                                        ↓
    ┌──────────────────────────────────────────────────────┐
    │          INFERENCE PATH (Inference Only)             │
    ├──────────────────────────────────────────────────────┤
    │  Sample indices ~ Uniform(0, 2^H - 1)                │
    │      ↓                                               │
    │  One-Hot Encode                                      │
    │      ↓                                               │
    │  Z (B, T, 2^H) ───────────────────┐                  │
    └───────────────────────────────────┼──────────────────┘
                                        ↓
    ┌──────────────────────────────────────────────────────┐
    │              CONDITIONING STEP                       │
    ├──────────────────────────────────────────────────────┤
    │  Z (B, T, 2^H)                                       │
    │      ↓                                               │
    │  Post-Sampler Linear: 2^H → D                        │
    │      ↓                                               │
    │  R (B, T, D)                                         │
    │      ↓                                               │
    │  Conditioned = X_attn + R                            │
    └──────────┬───────────────────────────────────────────┘
               ↓
    ┌──────────────────────────────────────────────────────┐
    │              FFN SUB-LAYER                           │
    ├──────────────────────────────────────────────────────┤
    │  Residual ← Conditioned                              │
    │      ↓                                               │
    │  Normalize(Conditioned)                              │
    │      ↓                                               │
    │  Feed-Forward Network                                │
    │      ↓                                               │
    │  Dropout                                             │
    │      ↓                                               │
    │  Add(Residual)                                       │
    │      ↓                                               │
    │  Output (B, T, D)                                    │
    └──────────────────────────────────────────────────────┘

    ═══════════════════════════════════════════════════════════
                    Return (Output, bit_logits) if training
                    Return Output only if inference
    ═══════════════════════════════════════════════════════════
    ```

    Architecture Details:
    --------------------
    For the injection layer (use_free_transformer=True), the forward pass is:

    1. **Pre-Injection**: Standard self-attention with causal masking
    2. **Encoder Path** (training/prefill only):
       - Learned query ζ → Non-causal encoder block ← First-half output (KV)
       - Encoder output → Linear readout → H bit logits
       - Bit logits → Binary Mapper → Z (one-hot, dim 2^H)
    3. **Conditioning**:
       - Project Z: R = Linear(Z), shape (T, D)
       - Modified attention: Attention(Q=X, K=X+R, V=X+R)
    4. **Post-Injection**: Standard FFN and residual connections

    Training Considerations:
    -----------------------
    When use_free_transformer=True and training=True, this layer returns a tuple:
        (output, bit_logits)

    The bit_logits must be used to compute the KL divergence term:
        KL_loss = compute_kl_divergence(bit_logits, threshold=kappa)

    The total loss should be:
        total_loss = cross_entropy_loss + beta * KL_loss

    Args:
        hidden_size: Integer, hidden size of the layer.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, size of the intermediate (feed-forward) layer.
        use_free_transformer: Boolean, whether to enable the VAE mechanism.
            When False, behaves as a standard TransformerLayer.
            When True, acts as the injection layer. Default: False.
        num_latent_bits: Integer, number of latent bits H. The latent space will
            have 2^H categories. Paper uses H=16 (65,536 categories). Default: 16.
        encoder_attention_type: AttentionType, type of attention for the encoder block.
            Default: 'multi_head'. The encoder uses non-causal attention.
        encoder_ffn_type: FFNType, type of FFN for the encoder block.
            Default: 'swiglu' (same as paper's baseline).
        encoder_attention_args: Optional dictionary of arguments for encoder attention.
            These override defaults (e.g., {'causal': False} is enforced).
        encoder_ffn_args: Optional dictionary of arguments for encoder FFN.
        encoder_normalization_type: NormalizationType, normalization for encoder.
            Default: 'rms_norm' (same as paper's baseline).
        **kwargs: All other arguments passed to base TransformerLayer.

    Example:
        ```python
        # Build a simple model with Free Transformer injection at middle layer
        inputs = keras.Input(shape=(sequence_length, hidden_size))

        # First half: standard layers
        x = TransformerLayer(hidden_size=768, num_heads=12,
                            intermediate_size=3072)(inputs)
        x = TransformerLayer(hidden_size=768, num_heads=12,
                            intermediate_size=3072)(x)

        # Injection layer
        x, bit_logits = FreeTransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            use_free_transformer=True,
            num_latent_bits=16
        )(x, training=True)

        # Second half: standard layers
        x = TransformerLayer(hidden_size=768, num_heads=12,
                            intermediate_size=3072)(x)

        # Output
        logits = keras.layers.Dense(vocab_size)(x)

        # Loss computation
        ce_loss = compute_cross_entropy(true_labels, logits)
        kl_loss = compute_kl_divergence_with_free_bits(bit_logits, kappa=0.5)
        total_loss = ce_loss + kl_loss
        ```

    Note:
        Only ONE layer in a model should have use_free_transformer=True,
        typically positioned at the middle of the model (layer L/2).
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            use_free_transformer: bool = False,
            num_latent_bits: int = 16,
            encoder_attention_type: AttentionType = 'multi_head',
            encoder_ffn_type: FFNType = 'swiglu',
            encoder_attention_args: Optional[Dict[str, Any]] = None,
            encoder_ffn_args: Optional[Dict[str, Any]] = None,
            encoder_normalization_type: NormalizationType = 'rms_norm',
            **kwargs: Any
    ):
        # Initialize base transformer layer
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            **kwargs
        )

        # Store Free Transformer configuration
        self.use_free_transformer = use_free_transformer
        self.num_latent_bits = num_latent_bits
        self.encoder_attention_type = encoder_attention_type
        self.encoder_ffn_type = encoder_ffn_type
        self.encoder_attention_args = encoder_attention_args or {}
        self.encoder_ffn_args = encoder_ffn_args or {}
        self.encoder_normalization_type = encoder_normalization_type

        # Only create encoder components if Free Transformer is enabled
        if not self.use_free_transformer:
            return

        # Validate configuration
        if num_latent_bits <= 0 or num_latent_bits > 20:
            raise ValueError(
                f"num_latent_bits must be between 1 and 20, got {num_latent_bits}"
            )

        self.num_latent_categories = 2 ** num_latent_bits

        # These will be built in build() method
        self.zeta = None  # Learned constant query for encoder
        self.encoder_attention = None  # Non-causal attention for encoder
        self.encoder_attention_norm = None
        self.encoder_ffn = None
        self.encoder_output_norm = None
        self.encoder_attention_dropout = None
        self.encoder_ffn_dropout = None
        self.encoder_readout = None  # Linear layer: D → H bits
        self.binary_mapper = None  # Samples one-hot Z from bit logits
        self.post_sampler_fc = None  # Linear layer: 2^H → D

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sub-layers including encoder components if enabled.

        Args:
            input_shape: Shape tuple (batch_size, sequence_length, hidden_size)
        """
        # Build base transformer components
        super().build(input_shape)

        if not self.use_free_transformer:
            return

        # Extract dimensions
        # input_shape is (batch, seq_len, hidden_size)

        # ---------------------------------------------------------------------
        # 1. Learned constant query vector ζ (zeta) for encoder
        # ---------------------------------------------------------------------
        # Shape: (1, 1, hidden_size) - will be tiled to (batch, seq_len, hidden_size)
        self.zeta = self.add_weight(
            name='zeta_query',
            shape=(1, 1, self.hidden_size),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype
        )

        # ---------------------------------------------------------------------
        # 2. Encoder attention block (non-causal)
        # ---------------------------------------------------------------------
        # Override attention args to ensure non-causal behavior
        encoder_attn_args = self.encoder_attention_args.copy()
        encoder_attn_args['causal'] = False  # Critical: encoder must be non-causal

        self.encoder_attention = create_attention_layer(
            attention_type=self.encoder_attention_type,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='encoder_attention',
            **encoder_attn_args
        )

        # Build attention layer
        self.encoder_attention.build(input_shape)

        # Attention normalization
        self.encoder_attention_norm = create_normalization_layer(
            normalization_type=self.encoder_normalization_type,
            name='encoder_attention_norm',
            **(self.attention_norm_args or {})
        )
        self.encoder_attention_norm.build(input_shape)

        # Attention dropout
        self.encoder_attention_dropout = layers.Dropout(
            rate=self.attention_dropout_rate,
            name='encoder_attention_dropout'
        )

        # ---------------------------------------------------------------------
        # 3. Encoder FFN block
        # ---------------------------------------------------------------------
        self.encoder_ffn = create_ffn_from_config(
            ffn_type=self.encoder_ffn_type,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='encoder_ffn',
            **self.encoder_ffn_args
        )
        self.encoder_ffn.build(input_shape)

        # FFN normalization
        self.encoder_output_norm = create_normalization_layer(
            normalization_type=self.encoder_normalization_type,
            name='encoder_output_norm',
            **(self.ffn_norm_args or {})
        )
        self.encoder_output_norm.build(input_shape)

        # FFN dropout
        self.encoder_ffn_dropout = layers.Dropout(
            rate=self.dropout_rate,
            name='encoder_ffn_dropout'
        )

        # ---------------------------------------------------------------------
        # 4. Encoder readout: hidden_size → num_latent_bits
        # ---------------------------------------------------------------------
        self.encoder_readout = layers.Dense(
            units=self.num_latent_bits,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='encoder_readout_fc',
            dtype=self.dtype
        )
        self.encoder_readout.build(input_shape)

        # ---------------------------------------------------------------------
        # 5. Binary Mapper: samples one-hot Z from bit logits
        # ---------------------------------------------------------------------
        self.binary_mapper = BinaryMapper(
            num_bits=self.num_latent_bits,
            name='binary_mapper',
            dtype=self.dtype
        )
        # Binary mapper input shape: (batch, seq, num_bits)
        self.binary_mapper.build(input_shape[:-1] + (self.num_latent_bits,))

        # ---------------------------------------------------------------------
        # 6. Post-sampler: projects Z back to hidden dimension
        # ---------------------------------------------------------------------
        # Input is one-hot Z of shape (batch, seq, 2^H)
        self.post_sampler_fc = layers.Dense(
            units=self.hidden_size,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='post_sampler_fc',
            dtype=self.dtype
        )
        self.post_sampler_fc.build(input_shape[:-1] + (self.num_latent_categories,))

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            layer_idx: int = 0,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass of the Free Transformer layer.

        Standard Mode (use_free_transformer=False):
            Returns: output tensor of shape (batch, seq, hidden_size)

        Free Mode (use_free_transformer=True):
            Training: Returns (output, bit_logits) tuple
            Inference: Returns output tensor only

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask: Optional attention mask. For the main (causal) attention,
                typical shapes are:
                - Padding mask: (batch_size, sequence_length)
                - Causal mask: (sequence_length, sequence_length)
                - Full mask: (batch_size, num_heads, sequence_length, sequence_length)
            layer_idx: Integer layer index for certain attention types.
            training: Boolean flag for training mode.

        Returns:
            If use_free_transformer=False or (use_free_transformer=True and not training):
                output: Tensor of shape (batch_size, sequence_length, hidden_size)

            If use_free_transformer=True and training=True:
                (output, bit_logits): Tuple where:
                    - output: Tensor (batch, seq, hidden_size)
                    - bit_logits: Tensor (batch, seq, num_latent_bits) for KL computation
        """
        # Standard transformer behavior when Free Transformer is disabled
        if not self.use_free_transformer:
            return super().call(
                inputs,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                training=training
            )

        # ---------------------------------------------------------------------
        # Free Transformer Forward Pass
        # ---------------------------------------------------------------------

        bit_logits = None  # Will be populated during training

        # Step 1: Standard self-attention (first sub-layer)
        residual = inputs

        if self.normalization_position == 'pre':
            # Pre-norm: Normalize → Attention → Dropout → Add
            x = self.attention_norm(inputs, training=training)

            if self.attention_type == 'differential':
                attention_output = self.attention(
                    x,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training
                )
            else:
                attention_output = self.attention(
                    x,
                    attention_mask=attention_mask,
                    training=training
                )

            attention_output = self.attention_dropout(attention_output, training=training)

            if self.attention_stochastic_depth is not None:
                attention_output = self.attention_stochastic_depth(
                    attention_output,
                    training=training
                )

            attention_output = attention_output + residual

        else:  # post-norm
            # Post-norm: Attention → Dropout → Add → Normalize
            if self.attention_type == 'differential':
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training
                )
            else:
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    training=training
                )

            x = self.attention_dropout(x, training=training)

            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)

            attention_output = self.attention_norm(x + residual, training=training)

        # ---------------------------------------------------------------------
        # Step 2: Encoder path (training/prefill) or uniform sampling (inference)
        # ---------------------------------------------------------------------

        if training:
            # === Encoder Path (Training) ===
            # Run the non-causal encoder block to infer Z from the sequence

            # Tile learned query zeta to match sequence length
            # Shape: (1, 1, D) → (batch, seq_len, D)
            batch_size = keras.ops.shape(inputs)[0]
            seq_len = keras.ops.shape(inputs)[1]
            zeta_queries = keras.ops.tile(
                self.zeta,
                [batch_size, seq_len, 1]
            )

            # Encoder block: Pre-norm architecture (same as base layer)
            # Query: learned zeta, Keys/Values: attention_output
            encoder_residual = zeta_queries

            # Normalize queries
            zeta_norm = self.encoder_attention_norm(zeta_queries, training=training)

            # Non-causal cross-attention: Q=zeta, KV=attention_output
            # Note: The encoder attention was built with causal=False
            encoder_attn_out = self.encoder_attention(
                zeta_norm,
                attention_mask=None,  # Non-causal, no mask needed
                training=training
            )
            encoder_attn_out = self.encoder_attention_dropout(
                encoder_attn_out,
                training=training
            )
            encoder_attn_out = encoder_attn_out + encoder_residual

            # Encoder FFN
            encoder_residual = encoder_attn_out
            encoder_x = self.encoder_output_norm(encoder_attn_out, training=training)
            encoder_x = self.encoder_ffn(encoder_x, training=training)
            encoder_x = self.encoder_ffn_dropout(encoder_x, training=training)
            encoder_output = encoder_x + encoder_residual

            # Readout: D → H bit logits
            bit_logits = self.encoder_readout(encoder_output, training=training)

            # Sample one-hot Z from bit logits with gradient pass-through
            z_one_hot = self.binary_mapper(bit_logits, training=training)

        else:
            # === Inference Path ===
            # Sample Z uniformly from categorical distribution
            batch_size = keras.ops.shape(inputs)[0]
            seq_len = keras.ops.shape(inputs)[1]

            # Sample random indices uniformly in [0, 2^H - 1]
            random_indices = keras.random.uniform(
                shape=(batch_size, seq_len),
                minval=0,
                maxval=self.num_latent_categories,
                dtype='int32'
            )

            # Convert to one-hot
            z_one_hot = keras.ops.one_hot(
                random_indices,
                num_classes=self.num_latent_categories,
                dtype=self.compute_dtype
            )

        # ---------------------------------------------------------------------
        # Step 3: Condition on Z by injecting into keys/values
        # ---------------------------------------------------------------------
        # Project Z: 2^H → D
        z_projected = self.post_sampler_fc(z_one_hot, training=training)

        # Add to attention output to create conditioned representation
        # This creates the "R" tensor from the paper: R = Linear(Z)
        # The next operations will use attention_output + R as keys/values
        conditioned_kv = attention_output + z_projected

        # ---------------------------------------------------------------------
        # Step 4: FFN sub-layer (second sub-layer)
        # ---------------------------------------------------------------------
        # Note: According to Algorithm 2, the injection happens BEFORE the FFN.
        # The paper states: "block B/2 + 1 gets in_q=x, in_kv=x+r"
        # But since we're implementing this as a single layer, we interpret this
        # as: the attention already happened, now we just need to condition
        # the representation before FFN.
        #
        # For a more literal implementation, you would need to modify the
        # attention mechanism itself to accept separate query and key/value inputs.
        # For simplicity, we condition the representation directly.

        residual = conditioned_kv

        if self.normalization_position == 'pre':
            # Pre-norm: Normalize → FFN → Dropout → Add
            x = self.output_norm(conditioned_kv, training=training)
            x = self.ffn_layer(x, training=training)
            x = self.dropout(x, training=training)

            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            layer_output = x + residual

        else:  # post-norm
            # Post-norm: FFN → Dropout → Add → Normalize
            x = self.ffn_layer(conditioned_kv, training=training)
            x = self.dropout(x, training=training)

            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            layer_output = self.output_norm(x + residual, training=training)

        # ---------------------------------------------------------------------
        # Return appropriate output based on mode
        # ---------------------------------------------------------------------

        if training:
            # During training, return both output and bit_logits for KL computation
            return layer_output, bit_logits
        else:
            # During inference, return only the output
            return layer_output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """
        Compute output shape(s) of the layer.

        Args:
            input_shape: Shape tuple (batch, sequence, hidden_size)

        Returns:
            If use_free_transformer=False:
                Single shape tuple (batch, sequence, hidden_size)

            If use_free_transformer=True:
                Tuple of two shapes:
                    - (batch, sequence, hidden_size)
                    - (batch, sequence, num_latent_bits)
        """
        if not self.use_free_transformer:
            return input_shape

        # Return shapes for both outputs
        output_shape = input_shape
        bit_logits_shape = input_shape[:-1] + (self.num_latent_bits,)

        return output_shape, bit_logits_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary with all parameters.
        """
        config = super().get_config()
        config.update({
            'use_free_transformer': self.use_free_transformer,
            'num_latent_bits': self.num_latent_bits,
            'encoder_attention_type': self.encoder_attention_type,
            'encoder_ffn_type': self.encoder_ffn_type,
            'encoder_attention_args': self.encoder_attention_args,
            'encoder_ffn_args': self.encoder_ffn_args,
            'encoder_normalization_type': self.encoder_normalization_type,
        })
        return config


# ---------------------------------------------------------------------
# Utility functions for computing VAE loss components
# ---------------------------------------------------------------------


def compute_kl_divergence_uniform_prior(
        bit_logits: keras.KerasTensor,
        num_bits: int,
        axis: int = -1
) -> keras.KerasTensor:
    """
    Compute KL divergence between encoder posterior Q(Z|S) and uniform prior P(Z).

    For the Free Transformer, the prior P(Z) is uniform over 2^H categories:
        P(Z_t = z) = 1 / 2^H  for all z ∈ {0, ..., 2^H - 1}

    The encoder outputs H independent bit logits, defining a factorized posterior:
        Q(Z_t = z | S) = ∏_{h=1}^H Q(B_h = U_h(z))
        where U(z) is the binary encoding of z.

    **Computation Flow**:
    ```
    Input: Bit Logits (B, T, H)
           ↓
    Sigmoid → p_h = σ(logit_h)
           ↓
    Clip probabilities: p_h ∈ [1e-7, 1-1e-7]
           ↓
    Binary Entropy: H(p_h) = -[p_h·log(p_h) + (1-p_h)·log(1-p_h)]
           ↓
    KL per bit: log(2) - H(p_h)
           ↓
    Sum over bits (axis=-1)
           ↓
    Output: KL Divergence (B, T)
    ```

    The KL divergence can be computed as (Equation 4 in paper):
        KL(Q(Z_t|S) || P(Z_t)) = H*log(2) + Σ_z Q(z|S) log Q(z|S)

    However, computing this sum over 2^H categories is expensive. Instead, we use
    the fact that for independent bits, the KL can be computed per-bit:
        KL = Σ_h KL(Q(B_h|S) || Uniform(B_h))

    For a Bernoulli with probability p:
        KL(Bernoulli(p) || Uniform) = H(1/2) - H(p)
                                     = log(2) - [p*log(p) + (1-p)*log(1-p)]

    Args:
        bit_logits: Tensor of shape (batch, sequence, num_bits) containing
            logits for each independent bit.
        num_bits: Integer H, number of bits.
        axis: Integer, axis along which to sum the KL (typically -1 for bits).
            Returns per-token KL of shape (batch, sequence).

    Returns:
        KL divergence tensor of shape (batch, sequence).
        To get per-token KL, use axis=-1.
        To get per-sequence KL, sum over sequence dimension after calling.

    Example:
        ```python
        # Compute per-token KL
        kl_per_token = compute_kl_divergence_uniform_prior(
            bit_logits, num_bits=16, axis=-1
        )
        # Shape: (batch, sequence)

        # Apply free bits threshold
        kappa = 0.5  # 0.5 bits per token
        kl_loss = keras.ops.mean(
            keras.ops.maximum(0.0, kl_per_token - kappa)
        )
        ```
    """
    # Compute bit probabilities: p_h = sigmoid(logit_h)
    probs = keras.ops.sigmoid(bit_logits)

    # Clip probabilities for numerical stability
    probs = keras.ops.clip(probs, 1e-7, 1.0 - 1e-7)

    # Compute binary entropy: H(p) = -[p*log(p) + (1-p)*log(1-p)]
    entropy = -(
            probs * keras.ops.log(probs) +
            (1.0 - probs) * keras.ops.log(1.0 - probs)
    )

    # KL per bit: log(2) - H(p)
    log_2 = keras.ops.cast(keras.ops.log(2.0), bit_logits.dtype)
    kl_per_bit = log_2 - entropy

    # Sum over bits to get KL per token
    kl_divergence = keras.ops.sum(kl_per_bit, axis=axis)

    return kl_divergence


def compute_free_bits_kl_loss(
        bit_logits: keras.KerasTensor,
        num_bits: int,
        kappa: float = 0.5,
        reduction: str = 'mean'
) -> keras.KerasTensor:
    """
    Compute KL divergence loss with free bits thresholding.

    The free bits method (Kingma et al., 2016) prevents KL collapse by only
    penalizing KL divergence above a threshold κ (kappa):

        KL_loss = (1/T) * Σ_t max(0, KL(Q(Z_t|S) || P(Z_t)) - κ)

    where T is the sequence length.

    **Free Bits Mechanism**:
    ```
    Input: Bit Logits (B, T, H)
           ↓
    Compute Per-Token KL → kl_t (B, T)
           ↓
    Apply Free Bits Threshold
           ↓
    ┌────────────────────────────┐
    │  For each token position t:│
    │                            │
    │  if kl_t < κ:              │
    │      kl_above_t = 0        │  ← Free bits (no penalty)
    │  else:                     │
    │      kl_above_t = kl_t - κ │  ← Penalize excess
    │                            │
    └────────┬───────────────────┘
             ↓
    Reduction (mean/sum/none)
             ↓
    Output: Scalar Loss

    Visual Example (κ = 0.5):
    ────────────────────────────────
    Token  │ KL  │ After Threshold
    ───────┼─────┼─────────────────
      1    │ 0.3 │     0.0    (below κ, free)
      2    │ 0.5 │     0.0    (at κ, free)
      3    │ 0.8 │     0.3    (penalized)
      4    │ 1.2 │     0.7    (penalized)
      5    │ 0.1 │     0.0    (below κ, free)
    ────────────────────────────────
    Mean Loss = (0.0 + 0.0 + 0.3 + 0.7 + 0.0) / 5 = 0.2
    ```

    Args:
        bit_logits: Tensor of shape (batch, sequence, num_bits).
        num_bits: Integer H, number of latent bits.
        kappa: Float, free bits threshold in bits per token.
            Paper experiments with κ ∈ {log(2)/4, log(2)/2, log(2), 2*log(2)}
            corresponding to {0.25, 0.5, 1.0, 2.0} bits.
            Default: 0.5 bits (best performing in paper).
        reduction: String, how to reduce the loss:
            - 'mean': Average over batch and sequence (default)
            - 'sum': Sum over batch and sequence
            - 'none': Return per-token loss (batch, sequence)

    Returns:
        Scalar loss tensor if reduction is 'mean' or 'sum'.
        Tensor of shape (batch, sequence) if reduction is 'none'.

    Example:
        ```python
        # During training
        output, bit_logits = free_layer(x, training=True)

        # Compute losses
        ce_loss = keras.losses.sparse_categorical_crossentropy(y_true, output)
        kl_loss = compute_free_bits_kl_loss(
            bit_logits,
            num_bits=16,
            kappa=0.5,
            reduction='mean'
        )

        # Total loss
        total_loss = ce_loss + kl_loss
        ```
    """
    # Compute per-token KL divergence
    kl_per_token = compute_kl_divergence_uniform_prior(
        bit_logits,
        num_bits=num_bits,
        axis=-1
    )

    # Apply free bits threshold
    # Only penalize KL above threshold
    kl_above_threshold = keras.ops.maximum(
        0.0,
        kl_per_token - kappa
    )

    # Apply reduction
    if reduction == 'mean':
        return keras.ops.mean(kl_above_threshold)
    elif reduction == 'sum':
        return keras.ops.sum(kl_above_threshold)
    elif reduction == 'none':
        return kl_above_threshold
    else:
        raise ValueError(
            f"Invalid reduction '{reduction}'. "
            f"Expected 'mean', 'sum', or 'none'."
        )


# ---------------------------------------------------------------------