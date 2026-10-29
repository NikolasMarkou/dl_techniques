"""Probabilistic Tiny Recursive Model (PTRM): a Tiny Recursive Model (TRM) equipped
with a stochastic inference procedure for test-time compute scaling.

PTRM shares the exact same architecture as TRM: a single small network applied
recursively under Adaptive Computation Time (ACT), with a hierarchical latent
state (z_H, z_L) and a Q-head for learned halting. The innovation of PTRM is
purely at inference time: instead of a single deterministic rollout, PTRM runs
K parallel stochastic rollouts by injecting Gaussian noise into the reasoning
latent ``z_L`` at each supervision step, and selects the final answer by the
Q-head's halting logit ``q_halt`` (arXiv:2605.19943, Fig. 4 left).

This allows PTRM to escape bad basins in the latent space where deterministic
TRM gets stuck, without any retraining or task-specific augmentations.

References:
    - Sghaier et al., 2026. Probabilistic Tiny Recursive Model.
      (https://arxiv.org/abs/2605.19943)
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (https://arxiv.org/abs/2510.04871)
    - Graves, 2016. Adaptive Computation Time for Recurrent Neural Networks.
      (https://arxiv.org/abs/1603.08983)
    - Wang et al., 2025. Hierarchical Reasoning Model.
      (https://arxiv.org/abs/2506.21734)
"""

import keras
from typing import Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.language.trm.model import TRM
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.weight_transfer import (
    load_weights_or_raise,
    pretrained_path_or_raise,
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.ptrm.model")
class PTRM(TRM):
    """Probabilistic Tiny Recursive Model.

    Identical architecture to TRM. The PTRM innovation is in the inference
    procedure (see :class:`PTRMInference` in :mod:`dl_techniques.models.language.ptrm.inference`),
    not in the model architecture itself. This class exists to provide a
    distinct registration key and factory for PTRM checkpoints.

    Architecture:

    .. code-block:: text

        (carry, batch)
             │
             ▼
        ┌──────────────────────────┐
        │ TRMInner                 │  one z_L update, then one z_H
        │ (rope attention)         │  update; the new carry leaves
        └──────────────────────────┘  through stop_gradient
             │
             ▼  z_H
             │
      ┌──────┴──────┐
      ▼             ▼
  reasoning_output  q_head -> q_halt, q_continue
      │             │
      ▼             ▼
  carry frozen where halted (ops.where, not Python break)
             │
             ▼
      new_carry, outputs

    :param vocab_size: Size of the vocabulary for token embeddings.
    :param hidden_size: Dimensionality of hidden states. Must be divisible by `num_heads`.
    :param num_heads: Number of attention heads in transformer layers.
    :param expansion: FFN intermediate-size multiplier.
    :param seq_len: Length of the input sequence, excluding the puzzle embedding.
    :param puzzle_emb_len: Length of the puzzle embedding prefix. Default 16.
    :param h_layers: Number of layers in the H-level reasoning module. Default 2.
    :param l_layers: Number of layers in the L-level reasoning module. Default 2.
    :param halt_max_steps: Maximum number of ACT steps allowed. Must be >= 1.
    :param halt_exploration_prob: Probability of forcing extra exploration steps during training. Default 0.1.
    :param no_act_continue: If True, halt on `q_halt > 0`; if False, use Q-learning halting (`q_halt > q_continue`).
    :param rope_theta: RoPE base frequency. Default 10000.0.
    :param attention_type: Attention mechanism. Default 'group_query'.
    :param ffn_type: Feed-forward network type. Default 'swiglu'.
    :param normalization_type: Normalization layer type. Default 'rms_norm'.
    :param normalization_position: 'pre' or 'post'. Default 'post'.
    :param dropout_rate: Dropout rate for transformer layers. Default 0.0.
    :param attention_dropout_rate: Dropout rate for attention. Default 0.0.
    :param **kwargs: Forwarded to `keras.Model`.

    :raises ValueError: If `hidden_size` is not divisible by `num_heads`, if
        `halt_max_steps` is below 1, or if `halt_exploration_prob` is outside [0, 1].
    """

    # DECISION plan-2026-10-13T035756-5dd4d94d/D-019: WHAT NOT TO DO: do not
    # re-add "mlp_*" variants that build attention models. The paper's TRM-MLP
    # replaces attention with an MLP over the sequence axis (reference trm.py
    # `mlp_t`); HierarchicalReasoningModule always builds TransformerLayer blocks, so the
    # old mlp_* entries were attention models under an MLP name (mlp_base ==
    # att_base). See decisions.md D-019.
    #
    # Trainable weights, measured from built models at vocab_size=20,
    # seq_len=81, puzzle_emb_len=16 (Sudoku-Extreme's shape): att_small
    # 7,177,730; att_base 13,764,610; att_large 28,511,234. The paper's "7M"
    # TRM-Att reuses ONE 2-layer network for both updates (trm.py: H_layers
    # is ignored); this TRM has separate 2-layer H_level and L_level stacks
    # (6,821,888 weights each in att_base), so the same width builds about
    # twice the weights.
    MODEL_VARIANTS = {
        "att_small": {
            "hidden_size": 384,
            "num_heads": 6,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        "att_base": {
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        "att_large": {
            "hidden_size": 768,
            "num_heads": 12,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
    }

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        puzzle_emb_len: int = 16,
        h_layers: int = 2,
        l_layers: int = 2,
        halt_max_steps: int = 10,
        halt_exploration_prob: float = 0.1,
        no_act_continue: bool = True,
        rope_theta: float = 10000.0,
        attention_type: str = 'group_query',
        ffn_type: str = 'swiglu',
        normalization_type: str = 'rms_norm',
        normalization_position: str = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        # Delegate all initialization to TRM parent
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            h_layers=h_layers,
            l_layers=l_layers,
            halt_max_steps=halt_max_steps,
            halt_exploration_prob=halt_exploration_prob,
            no_act_continue=no_act_continue,
            rope_theta=rope_theta,
            attention_type=attention_type,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            **kwargs
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        seq_len: int,
        puzzle_emb_len: int = 16,
        pretrained: Union[bool, str] = False,
        **kwargs: Any
    ) -> "PTRM":
        """Create a built PTRM from a predefined variant.

        Built, so ``H_init`` / ``L_init`` exist before the first ``call`` and
        before :class:`PTRMInference` reads them.

        :param variant: String, one of "att_small", "att_base", "att_large".
        :param vocab_size: Integer, size of vocabulary.
        :param seq_len: Integer, maximum sequence length.
        :param puzzle_emb_len: Integer, puzzle embedding length. Default 16.
        :param pretrained: ``False`` (default) for random init. ``True`` raises
            :class:`NotImplementedError`: no PTRM weights are distributed. A
            string is the path of a local checkpoint of the SAME architecture
            (``.keras`` or ``.weights.h5``), restored whole and strictly; a load
            that changes no variable raises.
        :param **kwargs: Constructor arguments laid over the variant's entry.
        :return: A built PTRM instance.
        :raises ValueError: If variant is not recognized.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        :raises FileNotFoundError: If ``pretrained`` is a path that does not exist.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        # Refused before anything is built.
        weights_path = pretrained_path_or_raise(pretrained, "PTRM")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        model = cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            **config
        )
        model.build()
        if weights_path is not None:
            load_weights_or_raise(model, weights_path)
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        # Ensure the class name is PTRM for serialization
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PTRM":
        """Create model from configuration."""
        return cls(**config)


# ---------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------


def create_ptrm(
    variant: str,
    vocab_size: int,
    seq_len: int,
    puzzle_emb_len: int = 16,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> PTRM:
    """Create a built PTRM from a variant name.

    Delegates to :meth:`PTRM.from_variant` and holds no logic of its own, like
    ``create_resnet``. The paper's settings per benchmark (variant, ``K``,
    ``D``, ``sigma``) are the preset helpers in
    :mod:`dl_techniques.models.language.ptrm.factory`; a fully custom
    architecture is the :class:`PTRM` constructor (then call ``build()``).

    :param variant: One of "att_small", "att_base", "att_large".
    :param vocab_size: Size of the vocabulary for token embeddings.
    :param seq_len: Length of the input sequence (excluding puzzle embedding).
    :param puzzle_emb_len: Length of the puzzle embedding prefix. Default 16.
    :param pretrained: ``False`` for random init; ``True`` raises
        :class:`NotImplementedError`; a string is a local checkpoint path (see
        :meth:`PTRM.from_variant`).
    :param kwargs: Constructor arguments laid over the variant's entry
        (``hidden_size``, ``h_layers``, ``halt_max_steps``, ``name``, ...).

    :return: A built ``PTRM`` instance.
    :raises ValueError: For an unknown variant, and for the argument checks
        :class:`PTRM` makes.

    Example::

        config = get_ppbench_config()
        model = create_ptrm(
            config["variant"], vocab_size=config["vocab_size"],
            seq_len=config["seq_len"], puzzle_emb_len=config["puzzle_emb_len"],
        )
    """
    # DECISION plan-2026-10-13T035756-5dd4d94d/D-015: WHAT NOT TO DO: do not
    # put logic here (no hyperparameter defaults, no compile, no second variant
    # factory). Every language create_* delegates to from_variant, like
    # create_resnet; a fully custom architecture is the PTRM constructor. The
    # api-contract delegation sweep judges this body. See decisions.md D-015.
    return PTRM.from_variant(
        variant,
        vocab_size=vocab_size,
        seq_len=seq_len,
        puzzle_emb_len=puzzle_emb_len,
        pretrained=pretrained,
        **kwargs,
    )


# ---------------------------------------------------------------------