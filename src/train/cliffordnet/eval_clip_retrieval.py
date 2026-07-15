"""COCO 2017 zero-shot image-text retrieval eval for CliffordCLIP.

Loads a saved ``CliffordCLIP`` ``.keras`` checkpoint, streams the COCO 2017
validation split through both encoders, and reports Recall@{1,5,10} in both
retrieval directions (image->text and text->image). All preprocessing matches
training exactly (``image_size``, ``context_length``, tiktoken ``gpt2``,
OpenAI CLIP mean/std, ``training=False``) via the shared image-text loaders, so
the reported numbers are meaningful against the training recipe.

The retrieval math itself is NOT reimplemented here: this script reuses
:func:`train.cliffordnet.train_clip._compute_retrieval_metrics` (the same
function the trainer uses for its in-run probes) so eval and probe numbers are
computed identically.

Scope note (eval semantics)
---------------------------
``load_coco2017_local_split("val", ...)`` emits exactly ONE caption per image,
giving the standard 5k-pair / one-correct-per-query retrieval setup. The full
academic COCO retrieval protocol uses all 5 captions per image (~25k captions
against 5k images, with 5 correct text targets per image). Supporting the
5-caption variant requires a loader change (multi-caption ground-truth sets in
``_compute_retrieval_metrics``) and is deliberately DEFERRED. Numbers from this
harness are directly comparable across runs but are NOT the 5-caption
literature protocol.

Usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python \\
        -m train.cliffordnet.eval_clip_retrieval \\
        --checkpoint results/.../checkpoints/final.keras \\
        --max-samples 1000 --gpu 1

A/B Launch Runbook (user-launched)
----------------------------------
The Clifford-head A/B is a multi-day GPU campaign and is USER-LAUNCHED (a
sub-agent's background job dies on exit; no parallel GPU jobs). The two arms
are ``head_kind=plain`` (control) vs ``head_kind=learned_query_residual``
(the Clifford geometric head). Keep seed / data / schedule IDENTICAL across
arms so the only difference is the head. Run on GPU0, batch 128, bf16.

Step A -- fixes already in place. This plan already shipped the code fixes
  (text_use_global_context kwarg, logit_scale float32 under bf16,
  --mixed-bfloat16, D-007 stagger + npz caption cache, this gamma probe).
  Nothing to launch.

Step B (optional) -- Wikipedia LM-pretrain of the text tower. Wired into
  train_clip via ``--pretrain-lm-steps N`` (default 50000; set 0 or pass
  ``--skip-pretrain`` to bypass). Point ``--pretrain-lm-hf-cache`` at a
  pre-downloaded HF Wikipedia cache. This runs as the first stage of the
  training invocation below, not as a separate command.

Step C -- the two A/B training runs (identical except --head-kind). Add
  ``text_use_global_context=True`` is NOT a CLI flag on its own; the G lever
  is passed through ``from_variant(**kwargs)`` only if a variant exposes it
  -- for the baseline A/B leave it default. GPU0::

    # Control arm (plain head)
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        -m train.cliffordnet.train_clip \\
        --head-kind plain \\
        --seed 42 --batch-size 128 --mixed-bfloat16 \\
        --gamma-probe-every-steps 500 --probe-every-steps 750

    # Treatment arm (Clifford learned-query-residual head)
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        -m train.cliffordnet.train_clip \\
        --head-kind learned_query_residual \\
        --seed 42 --batch-size 128 --mixed-bfloat16 \\
        --gamma-probe-every-steps 500 --probe-every-steps 750

  The treatment arm's training log will carry ``[gamma-probe]`` lines; gamma
  mean climbing above ~0.1 is the wedge-ignition signal (the geometric head
  is contributing). The plain arm logs no gamma-probe lines (no head_scale).

Step D -- evaluate each arm's checkpoint with THIS harness::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        -m train.cliffordnet.eval_clip_retrieval \\
        --checkpoint results/<run>/checkpoints/final.keras \\
        --coco-root /media/arxwn/data0_4tb/datasets/coco_2017

SIGNAL-FLOOR GATE
  Before attributing any A/B delta to the Clifford head, the PLAIN arm MUST
  clear COCO R@1 >= 5%. If it does not, the null result is a from-scratch
  backbone failure (constraint C20), NOT evidence against the Clifford head.
  Do NOT report a null A/B delta as "Clifford head adds nothing"; escalate to
  Path D (pretrained vision backbone) and re-run the A/B on top of it.

OUT-OF-SCOPE limitation (documented)
  Only the gamma probe is implemented. The query-pool / GAP cosine probe
  (the alternative wedge-ignition signal: cosine between the deterministic
  ``z_det`` and the geometric ``z_ctx`` branch) needs those locals from
  inside ``encode_image`` / ``encode_text`` and would require a forward hook
  into the encoders. It is NOT implemented here.
"""

import argparse
from typing import Dict

import keras
import tiktoken

from train.common import setup_gpu
from train.common.image_text import (
    load_coco2017_local_split,
    make_image_text_tf_dataset,
)
from train.cliffordnet.train_clip import (
    _compute_retrieval_metrics,
    ContrastiveCliffordCLIP,
)
from dl_techniques.models.clip.clifford_clip import CliffordCLIP
from dl_techniques.losses import CLIPContrastiveLoss
from dl_techniques.utils.logger import logger

# DECISION plan_2026-05-31_42743977/D-005
# ``StepCheckpointCallback`` saves the ``ContrastiveCliffordCLIP`` WRAPPER for
# ``checkpoints/step_*.keras`` and ``checkpoints/final.keras``; only the run-root
# ``cliffordclip_<variant>.keras`` is the bare ``CliffordCLIP`` (unwrapped at
# ``evaluate()`` via ``.clip_model``). The signal-floor gate MUST eval a
# mid-training ``step_*.keras`` (a wrapper), so both the wrapper and loss classes
# are registered here; do NOT assume a bare checkpoint. See decisions.md D-005.
_CUSTOM_OBJECTS = {
    "CliffordCLIP": CliffordCLIP,
    "ContrastiveCliffordCLIP": ContrastiveCliffordCLIP,
    "CLIPContrastiveLoss": CLIPContrastiveLoss,
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="COCO 2017 zero-shot retrieval eval for CliffordCLIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a saved CliffordCLIP .keras checkpoint",
    )
    p.add_argument(
        "--coco-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/coco_2017",
        help="COCO 2017 root (train2017/, val2017/, annotations/)",
    )
    p.add_argument(
        "--image-size", type=int, default=112,
        help="Image resolution (MUST match training)",
    )
    p.add_argument(
        "--context-length", type=int, default=64,
        help="Text sequence length (MUST match training)",
    )
    p.add_argument(
        "--max-samples", type=int, default=1000,
        help="Cap on the number of (image, caption) pairs evaluated",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Eval batch size",
    )
    p.add_argument(
        "--gpu", type=int, default=1, help="GPU device index",
    )
    return p


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    """Run COCO zero-shot retrieval and return the metrics dict.

    :param args: Parsed command-line arguments.
    :return: Metrics dict with ``i2t_r@{1,5,10}``, ``t2i_r@{1,5,10}`` and
        ``num_pairs`` keys (as returned by ``_compute_retrieval_metrics``).
    """
    setup_gpu(gpu_id=args.gpu)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = keras.models.load_model(
        args.checkpoint, compile=False, custom_objects=_CUSTOM_OBJECTS,
    )
    # Checkpoints from StepCheckpointCallback are the ContrastiveCliffordCLIP
    # WRAPPER (step_*.keras, final.keras); the run-root cliffordclip_<variant>.keras
    # is the bare CliffordCLIP. Unwrap to the inner CliffordCLIP either way so the
    # gate can eval mid-training step checkpoints.
    clip_model = getattr(model, "clip_model", model)
    logger.info(f"Loaded CliffordCLIP: {clip_model.count_params():,} params")

    encoder = tiktoken.get_encoding("gpt2")
    paths, token_ids = load_coco2017_local_split(
        split="val",
        coco_root=args.coco_root,
        max_samples=args.max_samples,
        encoder=encoder,
        context_length=args.context_length,
    )
    ds = make_image_text_tf_dataset(
        images=paths,
        token_ids=token_ids,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
    )

    logger.info(f"Computing retrieval metrics over {len(paths)} pairs...")
    metrics = _compute_retrieval_metrics(clip_model, ds)

    n = int(metrics.get("num_pairs", len(paths)))
    logger.info(f"COCO val zero-shot retrieval (N={n} pairs, 1 caption/image):")
    logger.info(
        "  image->text  R@1=%.2f%%  R@5=%.2f%%  R@10=%.2f%%"
        % (
            100.0 * metrics["i2t_r@1"],
            100.0 * metrics["i2t_r@5"],
            100.0 * metrics["i2t_r@10"],
        )
    )
    logger.info(
        "  text->image  R@1=%.2f%%  R@5=%.2f%%  R@10=%.2f%%"
        % (
            100.0 * metrics["t2i_r@1"],
            100.0 * metrics["t2i_r@5"],
            100.0 * metrics["t2i_r@10"],
        )
    )
    return metrics


def main() -> None:
    """Main entry point."""
    args = _build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
