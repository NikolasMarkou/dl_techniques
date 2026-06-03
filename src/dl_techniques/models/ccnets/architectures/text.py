"""Text/sentiment CCNet architecture: token-space orchestrators + Explainer/Reasoner/Producer(s) + factory.

This module extends the Causal Cooperative Networks paradigm
(``dl_techniques/models/ccnets``) from images to discrete token sequences (the
IMDB sentiment task), migrated verbatim from the former
``train/ccnets/text_sentiment.py`` training script:

    X = a movie review (token sequence)   -- the observation (effect)
    Y = sentiment (negative / positive)   -- the explicit cause (label)
    E = latent style / content            -- the latent cause

What differs from the image task (``architectures/mnist.py``):

* **The Producer ``P(X|Y,E)``.** Two variants, selected by
  ``ModelConfig.producer_type``:
    - ``'autoregressive'`` (default): a causal Transformer decoder
      (:class:`ARSentimentProducer`), teacher-forced on ``x_input``. Token ``i``
      is predicted from tokens ``< i`` plus a conditioning prefix built from
      ``(Y, E)``, with the sentiment ``Y`` re-injected into the residual stream
      at every layer. Word-dropout on the teacher-forced context stops the
      decoder from ignoring ``(Y, E)``. Used via :class:`ARTextCCNetOrchestrator`,
      which teacher-forces the Producer in ``forward_pass`` and decodes greedily
      for counterfactuals.
    - ``'nonautoregressive'``: :class:`SentimentProducer` emits the whole
      token-logit sequence in one shot from ``(Y, E)``. Simpler, but a small
      latent cannot carry a long review, so it collapses toward the unigram
      distribution (see PROTOTYPE SCOPE below).
  Both keep the differentiable label path (PRINCIPLES_CCNETS.md, P4 / CCNet
  Invariant 1) intact -- the label enters through a bias-free Dense projection,
  never an argmax index.
* **Token-space losses.** :class:`TextCCNetOrchestrator` overrides
  ``compute_losses``:
    - generation / reconstruction -> masked sparse categorical cross-entropy
      between Producer logits and the input tokens;
    - inference -> masked KL divergence between the two Producer output
      distributions.
  The pixel-norm losses (L1/L2/Huber) in ``losses.py`` do not apply to discrete X.

Everything else -- the variational Explainer, the cross-entropy-anchored Reasoner
error, the per-module gradient tapes, the live ``kl_weight`` -- is reused unchanged
from the base :class:`CCNetOrchestrator`. Only ``compute_losses`` (and, for the AR
variant, ``forward_pass``) is task-specific, which is Principle P11 (model-agnostic,
contract-based) in action.

PROTOTYPE SCOPE: this demonstrates the mechanism runs end-to-end on text --
cooperative gradient flow, token-space losses, sentiment counterfactuals. Note a deeper
caveat: a movie review is not *determined* by its sentiment, so the CCNet
necessity-&-sufficiency condition (PRINCIPLES_CCNETS.md, P1/P2) only partly holds for
this task. The autoregressive Producer makes generation fluent because each token is
predicted with its own context -- but that also means ``(Y, E)`` are modulators of a
conditional language model, not the sole cause of ``X``. Sentiment classification (the
Reasoner) is the strong, well-posed signal.

The CCNet framework symbols are imported absolutely from the framework submodules so
this module carries no dependency on ``train.*`` (data loading, decoding, plotting and
evaluation stay in the train script). If a downstream consumer needs a vocabulary size,
sequence length, etc., that is a constructor arg on :class:`ModelConfig`.
"""

import keras
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dl_techniques.models.ccnets.base import CCNetConfig, CCNetLosses
from dl_techniques.models.ccnets.orchestrators import CCNetOrchestrator
from dl_techniques.models.ccnets.utils import wrap_keras_model


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class ModelConfig:
    """Architecture parameters for the three text/sentiment CCNet modules."""
    vocab_size: int = 5000
    max_len: int = 80
    num_classes: int = 2          # negative / positive
    explanation_dim: int = 32

    embed_dim: int = 128
    encoder_hidden: int = 128     # GRU width for Explainer / Reasoner
    reasoner_dense_units: int = 64
    reasoner_dropout: float = 0.3

    producer_type: str = 'autoregressive'   # 'autoregressive' | 'nonautoregressive'
    producer_d_model: int = 128
    producer_layers: int = 2
    producer_heads: int = 4
    producer_ffn_dim: int = 256
    producer_word_dropout: float = 0.3       # AR only: teacher-forced-context dropout


# =====================================================================
# TOKEN-SPACE ORCHESTRATORS
# =====================================================================

class TextCCNetOrchestrator(CCNetOrchestrator):
    """CCNet orchestrator with token-space losses for discrete observations.

    Only ``compute_losses`` is overridden. ``forward_pass``, ``compute_model_errors``,
    ``train_step`` and the gradient-tape routing are inherited unchanged -- they
    operate on the three scalar losses regardless of how those scalars were formed.
    """

    def __init__(self, *args, pad_token: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token = pad_token

    def compute_losses(self, tensors: Dict[str, tf.Tensor]) -> CCNetLosses:
        x_input = tensors['x_input']               # [B, T] int token ids
        gen_logits = tensors['x_generated']        # [B, T, V]
        rec_logits = tensors['x_reconstructed']    # [B, T, V]

        # Mask out padding positions so they do not dilute the losses.
        mask = keras.ops.cast(
            keras.ops.not_equal(x_input, self.pad_token), dtype='float32'
        )                                          # [B, T]
        denom = keras.ops.sum(mask) + 1e-8

        # Generation / reconstruction: per-token cross-entropy in nats.
        ce_gen = keras.losses.sparse_categorical_crossentropy(
            x_input, gen_logits, from_logits=True
        )                                          # [B, T]
        ce_rec = keras.losses.sparse_categorical_crossentropy(
            x_input, rec_logits, from_logits=True
        )
        generation_loss = keras.ops.sum(ce_gen * mask) / denom
        reconstruction_loss = keras.ops.sum(ce_rec * mask) / denom

        # Inference loss: KL( P(x_generated) || P(x_reconstructed) ), per token.
        # Both come from the same Producer, so the gap reflects Y_truth vs Y_inferred.
        log_p = keras.ops.log_softmax(gen_logits, axis=-1)
        log_q = keras.ops.log_softmax(rec_logits, axis=-1)
        p = keras.ops.exp(log_p)
        kl = keras.ops.sum(p * (log_p - log_q), axis=-1)   # [B, T]
        inference_loss = keras.ops.sum(kl * mask) / denom

        return CCNetLosses(
            generation_loss=generation_loss,
            reconstruction_loss=reconstruction_loss,
            inference_loss=inference_loss,
        )


class ARTextCCNetOrchestrator(TextCCNetOrchestrator):
    """Token-space orchestrator for an autoregressive Producer.

    The autoregressive Producer needs the target sequence for teacher forcing, so
    ``forward_pass`` is overridden to pass ``x_input`` into the Producer. Token-space
    ``compute_losses`` is inherited from ``TextCCNetOrchestrator``.
    """

    def __init__(self, *args, max_len: int = 80, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_len = max_len

    def forward_pass(self, x_input, y_truth, training: bool = True):
        mu, log_var = self.explainer(x_input, training=training)
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        std = keras.ops.exp(0.5 * log_var)
        epsilon = keras.random.normal(shape=keras.ops.shape(mu))
        e_latent = mu + epsilon * std
        e_latent_no_grad = tf.stop_gradient(e_latent)

        y_inferred_probs = self.reasoner(x_input, e_latent_no_grad, training=training)

        # The autoregressive Producer is teacher-forced on x_input. The label still
        # enters differentiably (P4), so reconstruction gradient reaches the Reasoner.
        x_reconstructed = self.producer(
            y_inferred_probs, e_latent_no_grad, x_input, training=training)
        x_generated = self.producer(
            y_truth, e_latent, x_input, training=training)

        return {
            'x_input': x_input, 'y_truth': y_truth,
            'mu': mu, 'log_var': log_var, 'e_latent': e_latent,
            'y_inferred': y_inferred_probs,
            'x_reconstructed': x_reconstructed, 'x_generated': x_generated,
        }

    def counterfactual_generation(self, x_reference, y_target):
        """Greedy autoregressive decode conditioned on (y_target, style of x_reference)."""
        mu, _ = self.explainer(x_reference, training=False)
        batch = int(mu.shape[0])
        seq = np.zeros((batch, self.max_len), dtype="int32")
        for i in range(self.max_len):
            logits = self.producer(
                y_target, mu, tf.convert_to_tensor(seq), training=False)
            seq[:, i] = np.argmax(
                keras.ops.convert_to_numpy(logits[:, i]), axis=-1)
        return tf.convert_to_tensor(seq)


# =====================================================================
# MODULES — Explainer / Reasoner / Producer(s)
# =====================================================================

@keras.saving.register_keras_serializable(package="ccnets_text")
class SentimentExplainer(keras.Model):
    """Models P(E|X): review tokens -> (mu, log_var) of the latent cause."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding = keras.layers.Embedding(
            config.vocab_size, config.embed_dim, mask_zero=True, name="embedding"
        )
        self.encoder = keras.layers.Bidirectional(
            keras.layers.GRU(config.encoder_hidden), name="encoder"
        )
        self.fc_mu = keras.layers.Dense(config.explanation_dim, name="mu")
        self.fc_log_var = keras.layers.Dense(config.explanation_dim, name="log_var")

    def build(self, input_shape):
        c = self.config
        self.embedding.build((None, c.max_len))
        self.encoder.build((None, c.max_len, c.embed_dim))
        enc_dim = 2 * c.encoder_hidden
        self.fc_mu.build((None, enc_dim))
        self.fc_log_var.build((None, enc_dim))
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.encoder(self.embedding(x), training=training)
        return self.fc_mu(h), self.fc_log_var(h)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_text")
class SentimentReasoner(keras.Model):
    """Models P(Y|X,E): (review tokens, latent E) -> sentiment probabilities."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding = keras.layers.Embedding(
            config.vocab_size, config.embed_dim, mask_zero=True, name="embedding"
        )
        self.encoder = keras.layers.Bidirectional(
            keras.layers.GRU(config.encoder_hidden), name="encoder"
        )
        self.dense = keras.layers.Dense(
            config.reasoner_dense_units, activation="relu", name="dense"
        )
        self.dropout = keras.layers.Dropout(config.reasoner_dropout, name="dropout")
        self.classifier = keras.layers.Dense(
            config.num_classes, activation="softmax", name="classifier"
        )

    def build(self, input_shape):
        c = self.config
        self.embedding.build((None, c.max_len))
        self.encoder.build((None, c.max_len, c.embed_dim))
        combined = 2 * c.encoder_hidden + c.explanation_dim
        self.dense.build((None, combined))
        self.dropout.build((None, c.reasoner_dense_units))
        self.classifier.build((None, c.reasoner_dense_units))
        super().build(input_shape)

    def call(self, x, e, training=None):
        h = self.encoder(self.embedding(x), training=training)
        h = keras.ops.concatenate([h, e], axis=-1)
        h = self.dropout(self.dense(h), training=training)
        return self.classifier(h)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_text")
class SentimentProducer(keras.Model):
    """Models P(X|Y,E): (sentiment, latent E) -> token logits [B, T, vocab].

    Non-autoregressive: the conditioning vector is broadcast across all T
    positions, differentiated by positional embeddings, then refined by
    self-attention blocks. The label enters via a bias-free Dense projection
    (PRINCIPLES_CCNETS.md, P4 / CCNet Invariant 1) so gradient flows back to the
    Reasoner.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        d = config.producer_d_model

        # P4: differentiable label path -- Dense on the probability vector,
        # never argmax + Embedding.
        self.label_projection = keras.layers.Dense(d, use_bias=False, name="label_projection")
        self.style_projection = keras.layers.Dense(d, name="style_projection")
        self.merge = keras.layers.Dense(d, activation="gelu", name="merge")
        self.position_embedding = keras.layers.Embedding(
            config.max_len, d, name="position_embedding"
        )

        self.blocks: List[Dict[str, keras.layers.Layer]] = []
        for i in range(config.producer_layers):
            self.blocks.append({
                "attn": keras.layers.MultiHeadAttention(
                    num_heads=config.producer_heads, key_dim=d // config.producer_heads,
                    name=f"attn_{i}"),
                "norm1": keras.layers.LayerNormalization(name=f"norm1_{i}"),
                "ffn1": keras.layers.Dense(config.producer_ffn_dim, activation="gelu",
                                           name=f"ffn1_{i}"),
                "ffn2": keras.layers.Dense(d, name=f"ffn2_{i}"),
                "norm2": keras.layers.LayerNormalization(name=f"norm2_{i}"),
            })

        self.to_logits = keras.layers.Dense(config.vocab_size, name="to_logits")

    def build(self, input_shape):
        c = self.config
        d = c.producer_d_model
        self.label_projection.build((None, c.num_classes))
        self.style_projection.build((None, c.explanation_dim))
        self.merge.build((None, d))
        self.position_embedding.build((c.max_len,))
        seq = (None, c.max_len, d)
        for block in self.blocks:
            block["attn"].build(seq, seq)
            block["norm1"].build(seq)
            block["ffn1"].build(seq)
            block["ffn2"].build((None, c.max_len, c.producer_ffn_dim))
            block["norm2"].build(seq)
        self.to_logits.build(seq)
        super().build(input_shape)

    def call(self, y, e, training=None):
        content = self.label_projection(y)        # [B, d]
        style = self.style_projection(e)          # [B, d]
        seed = self.merge(content + style)        # [B, d]

        # Broadcast the conditioning vector to every sequence position.
        x = keras.ops.expand_dims(seed, axis=1)               # [B, 1, d]
        x = keras.ops.repeat(x, self.config.max_len, axis=1)  # [B, T, d]
        positions = keras.ops.arange(self.config.max_len)
        x = x + self.position_embedding(positions)            # [B, T, d]

        for block in self.blocks:
            attn = block["attn"](x, x, training=training)     # non-causal self-attention
            x = block["norm1"](x + attn)
            ffn = block["ffn2"](block["ffn1"](x))
            x = block["norm2"](x + ffn)

        return self.to_logits(x)                              # [B, T, vocab]

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_text")
class ARSentimentProducer(keras.Model):
    """Autoregressive P(X|Y,E): a causal Transformer decoder.

    Token ``i`` is predicted from tokens ``< i`` plus a conditioning prefix built
    from ``(Y, E)``. The sentiment ``Y`` is *additionally* injected into the
    residual stream at the input of every decoder layer, so the decoder cannot
    "forget" the conditioning across depth. Teacher-forced at training time;
    word-dropout on the teacher-forced context stops the decoder from ignoring
    ``(Y, E)`` and merely copying the surrounding text. Every label path is a
    bias-free / plain Dense projection of the probability vector
    (PRINCIPLES_CCNETS.md, P4 / CCNet Invariant 1), so reconstruction gradient
    still reaches the Reasoner.

    Call signature: ``producer(y, e, x_target, training=...)``.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        d = config.producer_d_model

        self.token_embedding = keras.layers.Embedding(
            config.vocab_size, d, name="token_embedding")
        self.position_embedding = keras.layers.Embedding(
            config.max_len, d, name="position_embedding")
        # P4: differentiable label path.
        self.label_projection = keras.layers.Dense(d, use_bias=False, name="label_projection")
        self.style_projection = keras.layers.Dense(d, name="style_projection")
        self.cond_merge = keras.layers.Dense(d, activation="gelu", name="cond_merge")

        self.blocks: List[Dict[str, keras.layers.Layer]] = []
        for i in range(config.producer_layers):
            self.blocks.append({
                # Per-layer sentiment injection (P4: differentiable label path).
                "label_inject": keras.layers.Dense(d, name=f"label_inject_{i}"),
                "attn": keras.layers.MultiHeadAttention(
                    num_heads=config.producer_heads, key_dim=d // config.producer_heads,
                    name=f"attn_{i}"),
                "norm1": keras.layers.LayerNormalization(name=f"norm1_{i}"),
                "ffn1": keras.layers.Dense(config.producer_ffn_dim, activation="gelu",
                                           name=f"ffn1_{i}"),
                "ffn2": keras.layers.Dense(d, name=f"ffn2_{i}"),
                "norm2": keras.layers.LayerNormalization(name=f"norm2_{i}"),
            })
        self.to_logits = keras.layers.Dense(config.vocab_size, name="to_logits")

    def _condition(self, y, e):
        return self.cond_merge(self.label_projection(y) + self.style_projection(e))

    def build(self, input_shape):
        c = self.config
        d = c.producer_d_model
        self.token_embedding.build((None, c.max_len - 1))
        self.position_embedding.build((c.max_len,))
        self.label_projection.build((None, c.num_classes))
        self.style_projection.build((None, c.explanation_dim))
        self.cond_merge.build((None, d))
        seq = (None, c.max_len, d)
        for block in self.blocks:
            block["label_inject"].build((None, c.num_classes))
            block["attn"].build(seq, seq)
            block["norm1"].build(seq)
            block["ffn1"].build(seq)
            block["ffn2"].build((None, c.max_len, c.producer_ffn_dim))
            block["norm2"].build(seq)
        self.to_logits.build(seq)
        super().build(input_shape)

    def call(self, y, e, x_target, training=None):
        cond = self._condition(y, e)               # [B, d]
        context_ids = x_target[:, :-1]             # [B, T-1] teacher-forcing context

        # Word dropout: replace some context tokens with <oov> (id 2) so the
        # decoder cannot ignore (Y, E) and simply copy the surrounding text.
        if training:
            drop = keras.random.uniform(keras.ops.shape(context_ids)) \
                < self.config.producer_word_dropout
            context_ids = keras.ops.where(
                drop, keras.ops.full_like(context_ids, 2), context_ids)

        context = self.token_embedding(context_ids)             # [B, T-1, d]
        # Conditioning prefix occupies position 0; the sequence stays length T.
        x = keras.ops.concatenate(
            [keras.ops.expand_dims(cond, axis=1), context], axis=1)   # [B, T, d]
        positions = keras.ops.arange(self.config.max_len)
        x = x + self.position_embedding(positions)

        for block in self.blocks:
            # Re-inject the sentiment Y at every layer's residual stream.
            x = x + keras.ops.expand_dims(block["label_inject"](y), axis=1)
            attn = block["attn"](x, x, use_causal_mask=True, training=training)
            x = block["norm1"](x + attn)
            ffn = block["ffn2"](block["ffn1"](x))
            x = block["norm2"](x + ffn)

        return self.to_logits(x)                                 # [B, T, vocab]

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


# =====================================================================
# CONSTRUCTION
# =====================================================================

# Defaults below mirror the former ``train/ccnets/text_sentiment.py``
# ``TrainingConfig`` so behavior is preserved byte-for-byte after the
# train->model decoupling (see decisions D-004/D-005/D-006).
def create_text_ccnet(
    model_config: ModelConfig,
    *,
    learning_rates: Optional[Dict[str, float]] = None,
    gradient_clip_norm: Optional[float] = 1.0,
    explainer_weights: Optional[Dict[str, float]] = None,
    reasoner_weights: Optional[Dict[str, float]] = None,
    producer_weights: Optional[Dict[str, float]] = None,
) -> TextCCNetOrchestrator:
    """Build the three modules, wrap them, and assemble the orchestrator.

    The Producer and orchestrator class are chosen by
    ``model_config.producer_type`` (``'autoregressive'`` -> :class:`ARSentimentProducer`
    + :class:`ARTextCCNetOrchestrator`; otherwise :class:`SentimentProducer` +
    :class:`TextCCNetOrchestrator`).

    Args:
        model_config: architecture-only configuration.
        learning_rates: per-module Adam learning rates. Defaults to
            ``{'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4}``.
        gradient_clip_norm: global grad-norm clip (default 1.0).
        explainer_weights: defaults to
            ``{'inference': 1.0, 'generation': 1.0, 'kl_divergence': 1e-3}``.
        reasoner_weights: defaults to ``{'inference': 1.0, 'reconstruction': 0.1}``.
        producer_weights: defaults to ``{'generation': 1.0, 'reconstruction': 1.0}``.

    Returns:
        A :class:`TextCCNetOrchestrator` (or :class:`ARTextCCNetOrchestrator` for
        the autoregressive Producer).
    """
    if learning_rates is None:
        learning_rates = {'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4}
    if explainer_weights is None:
        explainer_weights = {'inference': 1.0, 'generation': 1.0, 'kl_divergence': 1e-3}
    if reasoner_weights is None:
        reasoner_weights = {'inference': 1.0, 'reconstruction': 0.1}
    if producer_weights is None:
        producer_weights = {'generation': 1.0, 'reconstruction': 1.0}

    mc = model_config
    autoregressive = mc.producer_type == 'autoregressive'

    explainer = SentimentExplainer(mc)
    reasoner = SentimentReasoner(mc)
    producer = ARSentimentProducer(mc) if autoregressive else SentimentProducer(mc)

    # Build via dummy forward passes (the label dummy is a probability vector).
    dummy_x = keras.ops.zeros((1, mc.max_len), dtype="int32")
    dummy_y = keras.ops.zeros((1, mc.num_classes))
    dummy_e = keras.ops.zeros((1, mc.explanation_dim))
    explainer(dummy_x)
    reasoner(dummy_x, dummy_e)
    if autoregressive:
        producer(dummy_y, dummy_e, dummy_x)
    else:
        producer(dummy_y, dummy_e)

    ccnet_config = CCNetConfig(
        explanation_dim=mc.explanation_dim,
        loss_fn='l2',  # unused -- compute_losses is overridden for token space
        learning_rates=learning_rates,
        gradient_clip_norm=gradient_clip_norm,
        explainer_weights=explainer_weights,
        reasoner_weights=reasoner_weights,
        producer_weights=producer_weights,
    )
    modules = dict(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config,
        pad_token=0,
    )
    if autoregressive:
        return ARTextCCNetOrchestrator(max_len=mc.max_len, **modules)
    return TextCCNetOrchestrator(**modules)
