r"""Sample bytes from a trained H-Net checkpoint.

Usage::

    # what the CLI offers -- allocates nothing, touches no GPU
    MPLBACKEND=Agg .venv/bin/python -m train.hnet.infer_hnet --help

    # continue a prompt from a checkpoint the trainer wrote
    MPLBACKEND=Agg .venv/bin/python -m train.hnet.infer_hnet \
        --checkpoint results/hnet_dev_.../best_model.keras \
        --prompt "The history of " --max-new-bytes 64 --gpu 0

This is the counterpart of :mod:`train.hnet.train_hnet`: the training entry
point owns ``fit()``, this one owns everything between a checkpoint and decoded
text. Both parse FIRST and touch a GPU only afterwards.

Why this file exists
--------------------
A byte-level language model with no way to sample from it is an incomplete
deliverable. The plan's SC-13 asked for "a real CLI invocation of the
generation/eval path"; at step 18 there was no such path -- generation had to be
driven from a scratch script -- and the criterion could only be graded PARTIAL.
This module closes that, on the same CLI contract as
:mod:`train.doc_res.infer_doc_res`: ``main(argv)``, parse first, ``--help``
prints a ``usage:`` line and allocates nothing.

Decoding is DEFENSIVE, and that is not decoration
-------------------------------------------------
The model emits BYTES, not characters. A sampled continuation is cut wherever
``--max-new-bytes`` says it is, which is routinely in the middle of a multi-byte
UTF-8 codepoint, and a naive ``bytes(ids).decode()`` raises ``UnicodeDecodeError``
there. The reference implementation handles the same hazard by decoding growing
buffers and printing only what decodes (``generate.py:191-198``).

This module reimplements no UTF-8 rules of its own.
:func:`split_at_codepoint_boundary` is ``codecs``' incremental UTF-8 decoder plus
one ``getstate()`` read: the decoder already buffers a valid multi-byte PREFIX
and replaces (U+FFFD) anything that can neither start nor continue a codepoint,
which is exactly the incomplete-versus-invalid line that has to be drawn here.
So the emitted text carries a replacement character only where the model really
did produce an invalid byte -- never merely because the sample was cut short --
and the held-back bytes are reported separately rather than dropped.
:func:`~dl_techniques.datasets.byte_lm.byte_ids_to_text` remains the right helper
for a COMPLETE buffer, and it is what the dataset pipeline uses; it replaces a
truncated tail rather than holding it, which is correct there and wrong here.

No KV cache, and that is deliberate
-----------------------------------
Each step re-runs the full prefix. H-Net's chunking layers have no incremental
state API in this port (the reference's ``inference_params`` path is not ported,
`README.md` § 5.2), so a cache here would be a second, unvalidated
implementation of the forward pass. Sampling a few hundred bytes at dev scale
costs seconds; correctness is worth more than that.

Sequence-length floor
---------------------
The model refuses an input shorter than its ``max_chunks[0]``... it did, until
plan step 6.1. Since that fix the layer pads symmetrically and any length works,
which is why a prompt of a dozen-odd bytes is a legitimate invocation here. See
`decisions.md` D-029.
"""

from __future__ import annotations

import argparse
import codecs
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.datasets.byte_lm import text_to_byte_ids
from dl_techniques.utils.logger import logger
from train.common import setup_gpu

__all__ = [
    "DEFAULT_MAX_NEW_BYTES",
    "DEFAULT_PROMPT",
    "DEFAULT_TEMPERATURE",
    "NON_CONFIG_DESTS",
    "PROGRAM_NAME",
    "build_parser",
    "generate_bytes",
    "load_checkpoint",
    "main",
    "parse_arguments",
    "sample_next_id",
    "split_at_codepoint_boundary",
]


PROGRAM_NAME: str = "infer_hnet.py"
"""``prog`` for the parser, so ``--help`` names the script rather than
``__main__`` when the module is run with ``python -m``."""

DEFAULT_PROMPT: str = "The history of "
"""A short ASCII prompt. Short ON PURPOSE: it exercises the ``L < max_chunks``
regime that DEFECT 1 (D-028/D-029) made unusable and step 6.1 repaired."""

DEFAULT_MAX_NEW_BYTES: int = 64
DEFAULT_TEMPERATURE: float = 1.0

NON_CONFIG_DESTS = frozenset({"gpu"})
"""``--gpu`` acts on the process and is consumed by :func:`setup_gpu`. This
module has no config dataclass; the name is kept for symmetry with
:mod:`train.hnet.train_hnet`, where the same flag is the one carve-out."""


# ---------------------------------------------------------------------
# Byte-boundary-safe decoding
# ---------------------------------------------------------------------


def split_at_codepoint_boundary(ids: Sequence[int]) -> Tuple[str, List[int]]:
    """Split a byte sequence into "decodes cleanly" and "incomplete tail".

    Interface contract: pure. Reads nothing, raises nothing, and NEVER raises
    ``UnicodeDecodeError`` -- that is the whole point. Callers get back the
    longest prefix that is valid UTF-8 by itself, plus the bytes that were held
    back because they are a truncated codepoint rather than a broken one.

    The distinction between INCOMPLETE and INVALID is the whole difficulty, and
    it is why this is not a "decode, and on failure chop a byte off the end"
    loop. Such a loop reports ``(A, [0xFF, 0x42])`` for ``[0x41, 0xFF, 0x42]``:
    it holds back a real ``B`` behind a byte that begins no codepoint at all,
    and on a model emitting garbage it holds back an unbounded tail forever.
    ``codecs``' incremental UTF-8 decoder already draws the line exactly --
    buffering a valid multi-byte PREFIX and replacing anything that cannot
    start or continue one -- and its ``getstate()`` hands back precisely the
    bytes it buffered. This function is that decoder plus the state read; it
    reimplements no UTF-8 rules of its own.

    :param ids: Byte ids in ``[0, 255]``.
    :type ids: Sequence[int]
    :return: ``(text, held_back)``. ``held_back`` is empty whenever the input
        ends on a codepoint boundary, INCLUDING when the input contains invalid
        bytes -- those are replaced (U+FFFD), not held.
    :rtype: Tuple[str, List[int]]
    """
    raw = bytes(bytearray(int(i) & 0xFF for i in ids))
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(raw, False)
    return text, list(decoder.getstate()[0])


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------


def sample_next_id(
    logits: np.ndarray,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Draw one byte id from a logit vector.

    Interface contract: pure apart from ``rng``. ``temperature == 0.0`` is
    greedy ``argmax`` and consumes no randomness at all, which is what makes a
    deterministic test possible; any positive temperature draws from the
    (optionally nucleus-truncated) softmax.

    :param logits: ``(vocab_size,)`` float array.
    :type logits: np.ndarray
    :param temperature: Softmax temperature. ``0.0`` means greedy.
    :type temperature: float
    :param top_p: Nucleus threshold in ``(0, 1]``. ``1.0`` disables truncation.
    :type top_p: float
    :param rng: Source of randomness. ``None`` uses a fresh default generator.
    :type rng: Optional[np.random.Generator]
    :return: The sampled byte id.
    :rtype: int
    :raises ValueError: if ``temperature`` is negative or ``top_p`` is outside
        ``(0, 1]``.
    """
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature == 0.0:
        return int(np.argmax(values))

    scaled = values / temperature
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs = probs / probs.sum()

    if top_p < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        # Always keep at least one id: `searchsorted` can return 0 when the top
        # probability already exceeds top_p.
        keep = int(np.searchsorted(cumulative, top_p) + 1)
        mask = np.zeros_like(probs)
        mask[order[:keep]] = probs[order[:keep]]
        probs = mask / mask.sum()

    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.choice(probs.shape[0], p=probs))


def generate_bytes(
    model,
    prompt_ids: Sequence[int],
    max_new_bytes: int = DEFAULT_MAX_NEW_BYTES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = 1.0,
    context_bytes: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Continue ``prompt_ids`` byte by byte.

    Interface contract: calls ``model`` ``max_new_bytes`` times with an
    ``(1, L)`` int32 array and ``training=False`` explicit, and returns ONLY the
    newly generated ids -- never the prompt. The prompt is the caller's, and
    returning it concatenated is how a caller ends up printing it twice.

    :param model: A built H-Net (or anything mapping ``(1, L)`` ids to
        ``(1, L, vocab)`` logits).
    :param prompt_ids: The conditioning bytes. May be empty.
    :type prompt_ids: Sequence[int]
    :param max_new_bytes: How many bytes to draw.
    :type max_new_bytes: int
    :param temperature: See :func:`sample_next_id`.
    :type temperature: float
    :param top_p: See :func:`sample_next_id`.
    :type top_p: float
    :param context_bytes: Trailing-context cap. ``None`` keeps everything.
    :type context_bytes: Optional[int]
    :param rng: Source of randomness.
    :type rng: Optional[np.random.Generator]
    :return: ``max_new_bytes`` freshly sampled ids.
    :rtype: List[int]
    :raises ValueError: if ``max_new_bytes`` is negative, or the prompt is empty
        (a model conditioned on nothing has no first input to be given).
    """
    if max_new_bytes < 0:
        raise ValueError(f"max_new_bytes must be >= 0, got {max_new_bytes}")
    context = [int(i) for i in prompt_ids]
    if not context:
        raise ValueError("prompt_ids is empty; there is nothing to condition on")

    produced: List[int] = []
    # DECISION plan-2026-09-09T042752-6d66ac56/D-032: the two index expressions
    # below are the autoregressive contract, and BOTH were silently unguarded
    # until review pass 2 measured it.
    #   * `context[-context_bytes:]` keeps the TAIL. Do NOT write
    #     `context[:context_bytes]`: it keeps the head, so the window FREEZES
    #     the moment the cap is reached and generation stops conditioning on
    #     the bytes it just produced. MEASURED: that spelling left 219 tests
    #     green (mutation S-8), because the only guard asserted the window's
    #     SHAPE, which head slicing reproduces exactly.
    #   * `[0, -1, :]` reads the LAST position's logits, which is the only row
    #     whose prediction is not already known -- `pack_byte_windows` trains
    #     position t to predict byte t+1 (`datasets/byte_lm.py`). Do NOT write
    #     `[0, 0, :]`. MEASURED: also 219 green (mutation S-6), and the two
    #     hand invocations D-031 offers as evidence CANNOT tell the two apart,
    #     because the only checkpoint that exists emits 0x20 from every
    #     position. The discriminating guards are
    #     `test_inference.py::TestTheSamplerReadsTheLastPosition` (a
    #     position-DEPENDENT stub) and `::TestContextBytesKeepsTheTail` (window
    #     CONTENT, not shape). Rationale: decisions.md D-032.
    for _ in range(max_new_bytes):
        window = context if context_bytes is None else context[-context_bytes:]
        batch = np.asarray([window], dtype="int32")
        logits = np.asarray(model(batch, training=False))[0, -1, :]
        next_id = sample_next_id(
            logits, temperature=temperature, top_p=top_p, rng=rng
        )
        produced.append(next_id)
        context.append(next_id)
    return produced


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


def load_checkpoint(path: str):
    """Load a ``.keras`` H-Net checkpoint.

    Interface contract: imports :mod:`dl_techniques.models.language.hnet` for
    its registration side effect and then calls stock ``load_model``. No
    ``custom_objects`` argument is needed or accepted -- the package registers
    ``HNet``, ``HNetStage``, ``HNetBlock``, ``HNetIsotropic`` and the three
    chunking layers on import, and passing ``custom_objects`` instead would hide
    a registration regression.

    ``keras`` is imported INSIDE the function, not at module scope, so
    ``--help`` neither initialises the eager context nor claims a device.

    :param path: Path to a ``.keras`` archive.
    :type path: str
    :return: The loaded model.
    """
    import keras

    import dl_techniques.models.language.hnet  # noqa: F401  (registration)

    return keras.models.load_model(path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the parser.

    Interface contract: pure. Constructs and returns a parser; parses nothing,
    reads no environment, touches no filesystem and allocates no device. It is
    the single parser both :func:`parse_arguments` and the CLI guard drive.

    :return: The parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "Sample bytes from a trained H-Net checkpoint. There is no "
            "tokenizer: the model reads and writes raw UTF-8 bytes, and the "
            "output is decoded defensively at codepoint boundaries."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .keras checkpoint written by train_hnet.py.",
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help=f"Conditioning text (default: {DEFAULT_PROMPT!r}).",
    )
    parser.add_argument(
        "--max-new-bytes", type=int, default=DEFAULT_MAX_NEW_BYTES,
        help=f"Bytes to generate (default: {DEFAULT_MAX_NEW_BYTES}).",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=(
            f"Softmax temperature (default: {DEFAULT_TEMPERATURE}). 0.0 is "
            "greedy argmax and is fully deterministic."
        ),
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Nucleus sampling threshold in (0, 1]. 1.0 disables truncation.",
    )
    parser.add_argument(
        "--context-bytes", type=int, default=None,
        help=(
            "Cap the trailing context fed back in each step. Omit to keep the "
            "whole sequence."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the sampler's RNG. Omit for a fresh one each run.",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help=(
            "GPU index to use (e.g. 1). Omit to let the process see whatever "
            "CUDA_VISIBLE_DEVICES exposes."
        ),
    )
    return parser


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse ``argv``.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: The namespace.
    :rtype: argparse.Namespace
    :raises SystemExit: As argparse does, on ``--help`` or a bad flag.
    """
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> str:
    """Parse the CLI, set the process up, load, sample, decode.

    The statement ORDER is the contract:

    1. parse -- so ``--help`` costs nothing;
    2. ``setup_gpu`` -- process-level, from the non-config ``--gpu`` dest;
    3. load the checkpoint, sample, decode.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: The decoded continuation (the generated bytes only, not the
        prompt). Returned as well as logged so a caller -- a test included --
        can assert on it without scraping stdout.
    :rtype: str
    """
    args = parse_arguments(argv)

    setup_gpu(gpu_id=args.gpu)

    model = load_checkpoint(args.checkpoint)
    prompt_ids = text_to_byte_ids(args.prompt)
    logger.info(
        "H-Net: sampling %d bytes from %s at temperature=%.3f, top_p=%.3f "
        "(prompt: %d bytes)",
        args.max_new_bytes, args.checkpoint, args.temperature, args.top_p,
        len(prompt_ids),
    )

    rng = np.random.default_rng(args.seed)
    produced = generate_bytes(
        model,
        prompt_ids,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_p=args.top_p,
        context_bytes=args.context_bytes,
        rng=rng,
    )

    text, held_back = split_at_codepoint_boundary(produced)
    logger.info("prompt: %s", args.prompt)
    logger.info("continuation: %s", text)
    if held_back:
        logger.info(
            "%d trailing byte(s) held back as an incomplete UTF-8 codepoint: %s",
            len(held_back), held_back,
        )
    return text


if __name__ == "__main__":
    main()
