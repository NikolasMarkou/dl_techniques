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
from train.cliffordnet.train_clip import _compute_retrieval_metrics
from dl_techniques.models.cliffordnet.clip import CliffordCLIP
from dl_techniques.utils.logger import logger

# Checkpoints saved by ``train_clip.py`` are the BARE ``CliffordCLIP``
# (``ContrastiveCliffordCLIP.clip_model`` is the saved object, see
# train_clip.py save path), so ``{"CliffordCLIP": CliffordCLIP}`` is sufficient.
# The Clifford head layers (``_LayerScale1D`` / ``_LearnedQueryPool1D``) and the
# Clifford blocks auto-register via ``@register_keras_serializable``, and a bare
# checkpoint carries no ``CLIPContrastiveLoss``. If a future wrapper checkpoint
# is passed, ``load_model`` will raise a clear "Unknown object" naming the class
# to add here.
_CUSTOM_OBJECTS = {"CliffordCLIP": CliffordCLIP}


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
    logger.info(f"Loaded CliffordCLIP: {model.count_params():,} params")

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
    metrics = _compute_retrieval_metrics(model, ds)

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
