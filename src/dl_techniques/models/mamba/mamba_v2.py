"""
Mamba-2 encoder: a stack of state-space-duality blocks with head-scalar state
decay, grouped B/C projections and an optional parallel gated-MLP path.

Mamba-1 pays for its selectivity with a shape problem. Because each channel of the
inner dimension carries its own `d_inner x d_state` transition, the recurrence is a
sequence of small element-wise products that no matrix-multiply unit can help with,
and the state must stay small — `d_state = 16` in v1 — or the parameter and memory
cost explodes. Mamba-2 removes the obstruction by restricting `A` to a *scalar per
head*: `A_log` has shape `(nheads,)`, so the transition is `a_t * I` rather than a
diagonal matrix, and the whole recurrence collapses to

`h_t = a_t * h_{t-1} + delta_t * (B_t x_t^T)`,  `y_t = C_t^T h_t`

This is the structured-state-space-duality view: unrolled, the map from inputs to
outputs is a lower-triangular matrix whose entries are `C_i^T (prod_{k} a_k) B_j`,
which is a *masked attention matrix* with a scalar decay mask in place of a
softmax. The restriction is what buys the freedom elsewhere — with the state
transition reduced to one number per head, `d_state` can grow to 128 (the default
here, eight times v1) at negligible cost, and the state is where a recurrent model's
capacity actually lives.

The layer's parameterization follows from that duality. `B` and `C` are shared
across `ngroups` head groups rather than being per-head, exactly as keys and values
are shared in grouped-query attention and for the same reason. `dt` is emitted
directly by the input projection as one scalar per head, with no low-rank `dt_rank`
bottleneck — v1 needed that bottleneck because it produced a `Δ` per inner channel.
Critically, `z`, `x`, `B`, `C` and `dt` all come out of a *single* `in_proj` at the
top of the block, before the convolution, whereas v1 computes `Δ, B, C` from the
post-convolution activations. That reordering is not cosmetic: it makes every
projection in the block a function of the block input alone, which is what allows
the block to be sharded across devices without a sequential dependency in the
middle.

`A` is used as `-exp(A_log)` with `A_log` initialized from `log U(1, 16)`, so decay
rates are negative by construction and the recurrence is unconditionally stable.
`dt_bias` is set to the inverse-softplus of a log-uniform draw in `[dt_min, dt_max]`
so heads begin with a spread of timescales. Setting `d_ssm < d_inner` splits the
inner width: the first `d_mlp` channels bypass the SSM entirely as a gated MLP
(`silu(z0) * x0`) and are concatenated back before the output projection, which lets
a block trade state-space capacity for cheap pointwise capacity.

**The scan here is a sequential `while_loop`, not the SSD chunked-matmul
algorithm.** The entire practical argument for Mamba-2 is that the scalar-`A`
structure permits a block-decomposed formulation that runs on matrix-multiply
hardware and is several times faster than v1's scan. This implementation reproduces
the architecture and the parameterization but evaluates the recurrence one timestep
at a time with a `scatter_update` per step, so none of that speedup is present. Use
it as a correctness reference; do not benchmark it against the paper.

Two behaviours worth knowing before use. The model returns only
`{'last_hidden_state'}` — there is no LM head and no tied output projection, so a
task head must be attached externally, and the final normalization is a plain
`LayerNormalization` regardless of the `rmsnorm` flag (that flag governs the
in-block SSM-output norm only). And the `'130m'` entry of `MODEL_VARIANTS` carries a
`'name'` key whose value `'base'` is forwarded verbatim into `keras.Model.__init__`,
so `from_variant('130m')` yields a model literally named `base`; the key was
evidently intended as an alias marker and is not one.

Residual handling matches v1: blocks return `(output, running_residual)` and the
final addition is deferred to the model tail, so a caller stacking blocks manually
must thread the residual through or end up with no skip connections at all.

References:
    - Dao and Gu, 2024. Transformers are SSMs: Generalized Models and Efficient
      Algorithms Through Structured State Space Duality.
      (https://arxiv.org/abs/2405.21060)
    - Gu and Dao, 2023. Mamba: Linear-Time Sequence Modeling with Selective State
      Spaces. (https://arxiv.org/abs/2312.00752)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
"""

import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .components_v2 import Mamba2ResidualBlock


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Mamba2(keras.Model):
    """
    Mamba v2 foundation model for efficient sequence modeling.

    This model stacks Mamba2ResidualBlocks to form a deep sequence model,
    implementing the architecture from "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces" with V2 block enhancements.

    :param vocab_size: Size of the vocabulary.
    :param d_model: Dimensionality of the model's hidden states.
    :param num_layers: Number of Mamba residual blocks.
    :param d_state: Dimensionality of SSM latent state.
    :param d_conv: Kernel size for causal convolutions.
    :param expand: Expansion factor for internal dimensions.
    :param headdim: Dimensionality of each SSM head.
    :param norm_epsilon: Epsilon for all normalization layers.
    :param pad_token_id: ID of the padding token.
    :param rmsnorm: If True, use RMSNorm instead of LayerNormalization.
    :param d_ssm: Dimensionality of the SSM. Defaults to `d_model * expand`.
    :param norm_before_gate: Forwarded to every
        :class:`~dl_techniques.models.mamba.components_v2.Mamba2Layer` in the
        stack; see that class for the semantics. Exposed here because its
        docstring names ``norm_before_gate=True`` as the remedy for a checkpoint
        trained under the pre-2026-08-15 default, and nothing between this model
        and that layer forwarded it.
    """

    MODEL_VARIANTS = {
        "2.8b": {"d_model": 2560, "num_layers": 64},
        "1.4b": {"d_model": 2048, "num_layers": 48},
        "780m": {"d_model": 1536, "num_layers": 36},
        "370m": {"d_model": 1024, "num_layers": 24},
        "130m": {"d_model": 768, "num_layers": 24, "name": "base"},
    }

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            d_state: int = 128,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            norm_epsilon: float = 1e-5,
            pad_token_id: int = 0,
            rmsnorm: bool = True,
            d_ssm: Optional[int] = None,
            norm_before_gate: bool = False,
            **kwargs: Any,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate

        # If d_ssm is not provided, it should default to d_inner.
        d_inner = d_model * expand
        if d_ssm is None:
            d_ssm = d_inner
        self.d_ssm = d_ssm

        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=d_model, name="embedding"
        )
        self.encoder_layers = []
        for i in range(num_layers):
            block = Mamba2ResidualBlock(
                d_model=d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                headdim=self.headdim,
                d_ssm=self.d_ssm,
                rmsnorm=self.rmsnorm,
                norm_epsilon=self.norm_epsilon,
                norm_before_gate=self.norm_before_gate,
                name=f"mamba2_block_{i}",
            )
            self.encoder_layers.append(block)

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=norm_epsilon, name="final_norm"
        )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(inputs, dict):
            if "input_ids" not in inputs:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        if input_ids is None:
            raise ValueError("Input 'input_ids' cannot be None.")

        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.encoder_layers:
            hidden_states, residual = layer(hidden_states, residual)

        final_residual = hidden_states + residual if residual is not None else hidden_states
        last_hidden_state = self.final_norm(final_residual)

        return {"last_hidden_state": last_hidden_state}

    @classmethod
    def from_variant(cls, variant: str, vocab_size: int, **kwargs: Any) -> "Mamba2":
        if variant == "base":
            variant = "130m"
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)
        config["vocab_size"] = vocab_size
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "num_layers": self.num_layers, "d_state": self.d_state,
            "d_conv": self.d_conv, "expand": self.expand,
            "headdim": self.headdim, "norm_epsilon": self.norm_epsilon,
            "pad_token_id": self.pad_token_id,
            "rmsnorm": self.rmsnorm,
            "d_ssm": self.d_ssm,
            "norm_before_gate": self.norm_before_gate,
        })
        return config

# ---------------------------------------------------------------------
