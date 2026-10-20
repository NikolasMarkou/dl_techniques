"""
Configuration classes for Mixture of Experts (MoE) models.

This module provides simplified configuration dataclasses for MoE components,
focused exclusively on FFN experts and leveraging the dl_techniques FFN factory.
"""

import keras
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Literal

# The count rule (bool/np.bool_ refused, numpy integers accepted and coerced,
# int32 ceiling) has one home, shared with the adapters and the heads (D-088).
from dl_techniques.utils.validation import positive_int

# ---------------------------------------------------------------------


@dataclass
class ExpertConfig:
    """
    Simplified configuration for FFN expert networks in MoE models.

    This dataclass defines FFN expert configuration by leveraging the existing
    dl_techniques FFN factory system, eliminating parameter duplication and
    ensuring consistency with the broader framework.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────┐
        │    ExpertConfig     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  ffn_config (dict)  │──▶ FFN Factory
        │  ├─ type            │    (create_ffn_from_config)
        │  ├─ output_dim      │
        │  └─ ...params       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Additional Layers  │
        │  (use_bias, init,   │
        │   regularizers)     │
        └─────────────────────┘

    :param ffn_config: Dictionary containing FFN configuration that will be passed
        directly to the FFN factory's create_ffn_from_config() function.
        This should include 'type' and any FFN-specific parameters.
    :type ffn_config: Dict[str, Any]
    :param use_bias: Whether to include bias terms in any additional linear layers
        (not part of the FFN itself).
    :type use_bias: bool
    :param kernel_initializer: Weight initialization strategy for any additional layers.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Bias initialization strategy for any additional layers.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularization applied to weights in additional layers.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularization applied to biases in additional layers.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]

    .. note::
        ``use_bias``, ``kernel_initializer``, ``bias_initializer``,
        ``kernel_regularizer`` and ``bias_regularizer`` apply only to *additional*
        linear layers wrapped around the FFN. The current ``FFNExpert`` delegates
        entirely to the FFN factory and builds no such extra layers, so these
        fields are serialized for forward-compatibility but are currently inert.
        Configure the FFN itself through ``ffn_config``.
    """
    ffn_config: Dict[str, Any] = field(default_factory=dict)

    # Additional layer parameters (not part of FFN itself)
    use_bias: bool = True
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform'
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros'
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None

    # Optional normalization for experts (instantiated via the norms factory).
    # When ``norm_type`` is set, ``pre_norm`` defaults to True and ``post_norm``
    # to False. Both can be toggled independently.
    norm_type: Optional[str] = None
    norm_config: Dict[str, Any] = field(default_factory=dict)
    pre_norm: bool = True
    post_norm: bool = False

    def __post_init__(self):
        """Validate FFN configuration after dataclass creation."""
        if not self.ffn_config:
            # Provide sensible default FFN configuration
            self.ffn_config = {
                "type": "mlp",
                "hidden_dim": 2048,
                "output_dim": 512
            }
        elif 'type' not in self.ffn_config:
            raise ValueError("ffn_config must contain 'type' field specifying FFN type")

# ---------------------------------------------------------------------

@dataclass
class GatingConfig:
    """
    Configuration for MoE gating networks (routers).

    This dataclass defines the routing mechanism for MoE models, supporting
    various gating strategies and load balancing techniques.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────┐
        │   Input Tokens      │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Gating Network    │
        │   (linear │ cosine  │
        │    │ softmoe)       │
        └──────────┬──────────┘
                   │
            ┌──────┴──────┐
            ▼             ▼
        ┌────────┐  ┌────────┐
        │ top-k  │  │ aux    │
        │ routes │  │ losses │
        └────────┘  └────────┘

    :param gating_type: Type of gating mechanism ('linear', 'cosine', 'softmoe').
    :type gating_type: Literal['linear', 'cosine', 'softmoe']
    :param top_k: Number of experts to select per token.
    :type top_k: int
    :param add_noise: Whether to add noise to gating logits for exploration.
    :type add_noise: bool
    :param noise_std: Standard deviation of gating noise.
    :type noise_std: float
    :param temperature: Temperature parameter for gating softmax.
    :type temperature: float
    :param use_bias: Whether to use bias in linear gating.
    :type use_bias: bool
    :param embedding_dim: Dimension of expert embeddings for cosine gating.
    :type embedding_dim: int
    :param learnable_temperature: Whether temperature is learnable in cosine gating.
    :type learnable_temperature: bool
    :param num_slots: Number of input slots per expert in SoftMoE.
    :type num_slots: int
    :param aux_loss_weight: Weight for the auxiliary load-balancing loss.

        **Its effective strength scales with** ``top_k``. The loss uses the
        Switch Transformer formula, which is calibrated for ``top_k = 1``, and
        it is deliberately not normalized (see the anchored decision in
        :func:`~dl_techniques.layers.moe.gating.compute_auxiliary_loss`).
        Perfectly balanced routing therefore does not reach zero but a floor of
        exactly ``aux_loss_weight * top_k``. MEASURED 2026-08-26 at
        ``aux_loss_weight=0.01``, exact uniform gate probabilities and
        round-robin balanced dispatch over 4096 tokens:

        .. code-block:: text

            num_experts   top_k=1    top_k=2    top_k=4
            -----------   --------   --------   --------
                      4   0.010000   0.020000   0.040000
                      8   0.010000   0.020000   0.040000
                     16   0.010000   0.020000   0.040000
                     64   0.010000   0.020000   0.040000

        The floor is independent of ``num_experts`` and of the token count. The
        worst case -- every token to the same ``k`` experts -- is
        ``aux_loss_weight * num_experts`` regardless of ``top_k``, so the
        regularizer's dynamic range is ``num_experts / top_k``. To hold the
        regularization strength fixed across ``top_k``, divide your intended
        weight by ``top_k`` yourself.
    :type aux_loss_weight: float
    :param z_loss_weight: Weight for router z-loss (entropy regularization).
    :type z_loss_weight: float
    """
    gating_type: Literal['linear', 'cosine', 'softmoe'] = 'linear'
    top_k: int = 1
    add_noise: bool = True
    noise_std: float = 1.0
    temperature: float = 1.0

    # Linear gating parameters
    use_bias: bool = False

    # Cosine gating parameters
    embedding_dim: int = 256
    learnable_temperature: bool = True

    # SoftMoE parameters
    num_slots: int = 4

    # Load balancing parameters
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 1e-3

    # Optional pre-gating normalization via the norms factory.
    norm_type: Optional[str] = None
    norm_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate gating configuration after dataclass creation.

        Mirrors ``ExpertConfig.__post_init__`` so both sub-configs fail loud at
        construction time rather than deep inside layer assembly.

        Integer fields (``top_k``, ``num_slots``, ``embedding_dim``) go through
        :func:`~dl_techniques.utils.validation.positive_int`, which rejects ``bool``/``np.bool_``
        before it applies the range test, accepts integral numpy scalars, and
        returns a Python ``int`` -- the return value is assigned BACK to the
        field, so a ``np.int64`` supplied by a caller never reaches
        ``get_config``/``json.dumps``.
        """
        valid_types = ('linear', 'cosine', 'softmoe')
        if self.gating_type not in valid_types:
            raise ValueError(
                f"gating_type must be one of {valid_types}, got '{self.gating_type}'"
            )
        self.top_k = positive_int(self.top_k, 'top_k', 'GatingConfig')
        self.num_slots = positive_int(self.num_slots, 'num_slots', 'GatingConfig')
        self.embedding_dim = positive_int(self.embedding_dim, 'embedding_dim', 'GatingConfig')
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")

# ---------------------------------------------------------------------

@dataclass
class MoEConfig:
    """
    Complete configuration for Mixture of Experts models focused on FFN experts.

    This dataclass combines expert and gating configurations with MoE-specific
    parameters to define complete MoE architectures using FFN experts exclusively.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │           MoEConfig              │
        │                                  │
        │  ┌────────────┐ ┌────────────┐   │
        │  │ExpertConfig│ │GatingConfig│   │
        │  └─────┬──────┘ └─────┬──────┘   │
        │        │              │          │
        │        ▼              ▼          │
        │  ┌───────────────────────────┐   │
        │  │    MoE Layer Assembly     │   │
        │  │  (N experts + router)     │   │
        │  └───────────────────────────┘   │
        └──────────────────────────────────┘

    :param num_experts: Total number of FFN expert networks.
    :type num_experts: int
    :param expert_config: Configuration for FFN expert networks.
    :type expert_config: ExpertConfig
    :param gating_config: Configuration for the gating network.
    :type gating_config: GatingConfig
    :param jitter_noise: Standard deviation for uniform noise added to the
        gating input during training. Note: ``LinearGating`` also injects
        learned-scale Gaussian noise to the gating *logits* when
        ``add_noise=True``; the two sources stack. Set ``jitter_noise=0`` to
        rely solely on the gating-level noise.
    :type jitter_noise: float
    :param drop_tokens: Diagnostic flag only. It is echoed by
        :meth:`MixtureOfExperts.get_expert_utilization` and gates **no**
        forward-path behaviour: neither the dense nor the sparse hard-routing
        kernel drops a token, so flipping it leaves the layer's output
        bit-identical (measured: ``max|delta| == 0.0``).
    :type drop_tokens: bool
    :param use_residual_connection: Diagnostic flag only, with the same status
        as ``drop_tokens`` — echoed by
        :meth:`MixtureOfExperts.get_expert_utilization`, read by no kernel.
        There are no dropped tokens for a residual to rescue.
    :type use_residual_connection: bool

    .. note::
        The capacity-based dispatch these two flags were once described as
        "reserved for" is **not** planned. ``capacity_factor`` (``GatingConfig``)
        and ``routing_dtype`` (``MoEConfig``) were removed for that reason, as
        were the older ``train_capacity_factor`` / ``eval_capacity_factor``.
        None is tolerated as a legacy key: a payload still naming one raises
        ``TypeError`` at construction and in :meth:`from_dict`.
    """
    num_experts: int = 8
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    gating_config: GatingConfig = field(default_factory=GatingConfig)

    # System-level parameters
    jitter_noise: float = 0.01
    drop_tokens: bool = True
    use_residual_connection: bool = True

    def __post_init__(self):
        """Validate the complete MoE configuration after dataclass creation.

        Mirrors ``ExpertConfig.__post_init__`` and ``GatingConfig.__post_init__`` so
        the top-level config fails loud at construction time rather than deep inside
        layer assembly. ``MoEConfig`` is the only place the cross-field invariant
        ``top_k <= num_experts`` can be checked at all: ``GatingConfig`` owns
        ``top_k`` but does not know ``num_experts``, which is a sibling field here.

        Validated:

        * ``num_experts`` integral, not a ``bool``/``np.bool_``, and within
          ``[1, 2**31 - 1]`` -- coerced to a Python ``int`` (see
          :func:`~dl_techniques.utils.validation.positive_int`).
        * ``top_k <= num_experts``, for ``gating_type`` in ``('linear', 'cosine')``
          only — see the note below.
        * ``jitter_noise >= 0`` (rejected, not silently disabled, matching
          ``GatingConfig``'s ``noise_std >= 0`` precedent).

        .. note::
            ``gating_type='softmoe'`` is **excluded** from the ``top_k`` cross-check
            on purpose. SoftMoE does not perform top-k routing: it dispatches every
            token to every expert through ``num_slots`` learned slots, and
            ``MixtureOfExperts.__init__`` forwards only ``num_slots`` to
            ``SoftMoEGating`` (``layer.py``), never ``top_k``. Requiring
            ``top_k <= num_experts`` there would reject configurations that are
            perfectly valid because the field is inert for that gating type.

        :raises ValueError: If any of the above invariants is violated.
        """
        self.num_experts = positive_int(self.num_experts, 'num_experts', 'MoEConfig')

        # DECISION plan-2026-08-26T100331-f3744602/D-012
        # The `top_k <= num_experts` cross-check is deliberately SKIPPED for
        # `gating_type='softmoe'`. Do NOT "fix the omission" by dropping the
        # gating_type test: SoftMoE ignores `top_k` entirely -- `layer.py`'s
        # gating_kwargs allow-list forwards only `num_slots` to SoftMoEGating --
        # so an unrelated `top_k` value is inert there, and validating it would
        # reject working configs (e.g. num_experts=4, top_k=999, softmoe) that
        # construct and run correctly today.
        if self.gating_config.gating_type in ('linear', 'cosine'):
            if self.gating_config.top_k > self.num_experts:
                raise ValueError(
                    f"top_k must be between 1 and num_experts ({self.num_experts}), "
                    f"got {self.gating_config.top_k} "
                    f"(gating_type='{self.gating_config.gating_type}')"
                )

        if self.jitter_noise < 0:
            raise ValueError(f"jitter_noise must be >= 0, got {self.jitter_noise}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        # Serialize ExpertConfig, handling Keras initializer/regularizer objects
        expert_dict = {}
        for k, v in self.expert_config.__dict__.items():
            if isinstance(v, keras.initializers.Initializer):
                expert_dict[k] = keras.initializers.serialize(v)
            elif isinstance(v, keras.regularizers.Regularizer):
                expert_dict[k] = keras.regularizers.serialize(v)
            else:
                expert_dict[k] = v

        return {
            'num_experts': self.num_experts,
            'expert_config': expert_dict,
            'gating_config': dict(self.gating_config.__dict__),
            'jitter_noise': self.jitter_noise,
            'drop_tokens': self.drop_tokens,
            'use_residual_connection': self.use_residual_connection,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEConfig':
        """Create configuration from dictionary (does not mutate input)."""
        config_dict = dict(config_dict)  # shallow copy to avoid mutating caller's dict

        # Deserialize ExpertConfig, handling serialized Keras objects
        expert_raw = config_dict.pop('expert_config', {})
        for k in ('kernel_initializer', 'bias_initializer'):
            if k in expert_raw and isinstance(expert_raw[k], dict):
                expert_raw[k] = keras.initializers.deserialize(expert_raw[k])
        for k in ('kernel_regularizer', 'bias_regularizer'):
            if k in expert_raw and isinstance(expert_raw[k], dict):
                expert_raw[k] = keras.regularizers.deserialize(expert_raw[k])
        expert_config = ExpertConfig(**expert_raw)

        gating_config = GatingConfig(**config_dict.pop('gating_config', {}))

        return cls(
            expert_config=expert_config,
            gating_config=gating_config,
            **config_dict
        )

# ---------------------------------------------------------------------
