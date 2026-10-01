"""GPT-2 Pre-training with Harmonic Loss.

Replaces the inner-product LM head and softmax cross-entropy with the *harmonic
max* output unit and loss of Baek et al. (2025), "Harmonic Loss Trains
Interpretable AI Models" (arXiv:2502.01628). Token ``t`` is scored by the
Euclidean distance ``d_t = ||h - E_t||`` between the final hidden state ``h`` and
its (tied) embedding row, ``p_t = d_t^-n / sum_j d_j^-n``, and the model is
trained on ``-log p_target``. Embedding rows become the centres of the output
classes, which is what the paper credits for faster convergence, scale
invariance and more interpretable weights. The embedding table is shared with
the input, so the head adds no parameters.

``--harmonic-exponent`` is the paper's ``n``. The default (``None``) resolves to
``2 * embed_dim``, the value the reference GPT-2 script uses
(``dist_sq ** -n_embd``). The reference also trains harmonic GPT-2 with a ~10x
higher peak learning rate than its cross-entropy baseline and no weight decay;
those are this trainer's defaults (``--learning-rate``, ``--weight-decay``).

Everything else (data, tokenizer, schedule, checkpoints, generation probes,
resume) is :mod:`train.gpt2.pretrain`, reached through ``train_gpt2``'s
``model_factory`` hook.

Usage::

    # Harmonic GPT-2 small on Wikipedia, GPU 1
    python -m train.gpt2.pretrain_harmonic --gpu 1 --variant small --epochs 3

    # Smaller exponent (heavier-tailed distribution)
    python -m train.gpt2.pretrain_harmonic --harmonic-exponent 32

    # Resume from checkpoint
    python -m train.gpt2.pretrain_harmonic --resume path/to/step_0450000.keras

References:
    Baek, Liu, Tegmark et al. (2025). "Harmonic Loss Trains Interpretable AI
    Models." arXiv:2502.01628. Reference code: KindXiaoming/grow-crystals.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import keras

from train.common import setup_gpu
from train.gpt2.pretrain import (
    TrainingConfig,
    create_gpt2_model,
    train_gpt2,
    _build_parser,
    _config_from_args,
)
from dl_techniques.losses import HarmonicCausalLMLoss
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class HarmonicTrainingConfig(TrainingConfig):
    """Training config extended with the harmonic-head hyperparameters.

    :param harmonic_exponent: The paper's ``n`` in ``p ~ d^-n``. ``None``
        resolves to ``2 * embed_dim`` inside ``GPT2``.
    :param harmonic_eps: Floor added to the squared distance.
    """

    save_dir: str = "results/gpt2_harmonic_pretrain"
    learning_rate: float = 6e-3
    weight_decay: float = 0.0
    harmonic_exponent: Optional[float] = None
    harmonic_eps: float = 1e-6


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def create_harmonic_gpt2_model(config: HarmonicTrainingConfig) -> keras.Model:
    """Build GPT-2 with a harmonic head, wrapped for harmonic-loss training.

    :param config: Harmonic training configuration. ``tie_word_embeddings`` must
        be true: the embedding table supplies the class centres.
    :return: The built ``CausalLanguageModel`` wrapper.
    """
    loss_fn = HarmonicCausalLMLoss(label_smoothing=config.label_smoothing)
    logger.info(
        f"Loss: HarmonicCausalLMLoss(exponent={config.harmonic_exponent or '2*embed_dim'}, "
        f"eps={config.harmonic_eps})"
    )
    return create_gpt2_model(
        config,
        head_kwargs=dict(
            head_type="harmonic",
            harmonic_exponent=config.harmonic_exponent,
            harmonic_eps=config.harmonic_eps,
        ),
        loss_fn=loss_fn,
    )


def train_gpt2_harmonic(config: HarmonicTrainingConfig):
    """Run GPT-2 CLM pre-training with the harmonic head and loss."""
    return train_gpt2(
        config,
        model_factory=create_harmonic_gpt2_model,
        results_dir_prefix="gpt2_harmonic_pretrain",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


# Flags the shared parser declares that pick between ce and focal. The harmonic
# trainer always trains on harmonic loss, so these would be advertised in --help
# and silently ignored.
_INERT_FLAGS = ("--loss-type", "--focal-gamma")


def _drop_flags(parser: argparse.ArgumentParser, flags) -> None:
    """Remove already-registered options from ``parser`` and its groups.

    argparse has no public API for this. The namespace still needs the dests
    (``_config_from_args`` reads them), so callers restore them with
    ``set_defaults``.
    """
    for action in list(parser._actions):
        if any(opt in flags for opt in action.option_strings):
            parser._remove_action(action)
            # Groups keep their own list, used for --help rendering.
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
    for opt in flags:
        parser._option_string_actions.pop(opt, None)


def _build_harmonic_parser() -> argparse.ArgumentParser:
    """Extend the base parser with harmonic-loss args and reference defaults."""
    p = _build_parser()
    p.description = "GPT-2 Pre-training with Harmonic Loss (arXiv:2502.01628)"
    _drop_flags(p, _INERT_FLAGS)
    p.set_defaults(
        learning_rate=HarmonicTrainingConfig.learning_rate,
        save_dir=HarmonicTrainingConfig.save_dir,
        loss_type=HarmonicTrainingConfig.loss_type,
        focal_gamma=HarmonicTrainingConfig.focal_gamma,
    )

    group = p.add_argument_group("Harmonic loss")
    group.add_argument(
        "--harmonic-exponent", type=float, default=None,
        help="Harmonic exponent n of p ~ d^-n. Default 2*embed_dim "
             "(the reference script's setting).",
    )
    group.add_argument(
        "--harmonic-eps", type=float, default=1e-6,
        help="Floor added to the squared distance",
    )
    group.add_argument(
        "--weight-decay", type=float,
        default=HarmonicTrainingConfig.weight_decay,
        help="AdamW weight decay (reference harmonic run uses 0)",
    )
    return p


def _harmonic_config_from_args(args: argparse.Namespace) -> HarmonicTrainingConfig:
    """Map parsed CLI args to HarmonicTrainingConfig."""
    base = _config_from_args(args)
    fields = {**vars(base), "weight_decay": args.weight_decay}
    return HarmonicTrainingConfig(
        **fields,
        harmonic_exponent=args.harmonic_exponent,
        harmonic_eps=args.harmonic_eps,
    )


def main() -> None:
    """Main entry point for harmonic GPT-2 pre-training."""
    args = _build_harmonic_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _harmonic_config_from_args(args)
    logger.info(
        f"Config: variant={config.model_variant}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, wd={config.weight_decay}, "
        f"harmonic(n={config.harmonic_exponent or '2*embed_dim'}, "
        f"eps={config.harmonic_eps})"
    )

    train_gpt2_harmonic(config)


if __name__ == "__main__":
    main()
