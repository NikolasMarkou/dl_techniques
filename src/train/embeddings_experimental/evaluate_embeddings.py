"""Embedding-quality evaluation for a trained study cell.

Loads the encoder a cell wrote, embeds two locally-available corpora, and emits
a flat dictionary of metrics to ``eval.json``.

Two tasks, chosen because they are the only labelled text on this machine that
suits sentence embeddings, and no download is permitted:

**SQuAD v1.1 retrieval (primary).** Each question is a query and its gold
paragraph is the single relevant document; every other unique paragraph is a
distractor. This is the closest offline substitute for a real retrieval
benchmark.

**SST-2 linear probe (secondary).** A frozen encoder plus a logistic classifier,
which measures the space rather than a fine-tune.

STS is NOT evaluated, and that is a data fact rather than a choice: ``glue`` on
this machine is ``glue/sst2`` and nothing else -- no STS-B, no MRPC, no QQP, in
TFDS or in the raw download cache -- and there is no sentence-pair-plus-float
pipeline anywhere in the repository to feed one.

Four caveats that belong on every number this module produces
-------------------------------------------------------------
1. **Train/eval overlap.** SQuAD contexts ARE Wikipedia paragraphs and the MLM
   corpus is Wikipedia. All arms share the leak so a *relative* comparison
   survives it, but no absolute claim does.
2. **Prefix retrieval, not passage retrieval.** The position table is built at
   ``max_position_embeddings = max_seq_length``, so evaluation cannot exceed the
   training length. Measured on the local copy: contexts are mean 780, median
   705, p90 1166, max 4065 characters, against a window of a few hundred. The
   passage embedding sees a prefix.
3. **The padding confound is real and is measured, not assumed.** Stage 1 is
   packed and padding-free precisely because the Clifford arm's block has
   ``supports_masking = False``. Evaluation cannot pack -- a query is a query --
   so padding returns here. Length-sorted batching pads to the batch maximum
   rather than to ``max_length``, and every corpus reports its
   ``pad_fraction`` so the confound is visible rather than silent. SQuAD
   contexts are long and nearly fill the window; SST-2 sentences are short and
   do not, which is why SST-2 is secondary.
4. **The SimCSE projection head is not saved.** Stage 2 saves the encoder, not
   the wrapper, so evaluation uses ``pooled_output`` -- the encoder's own pooled
   representation -- and not the 256-d contrastive projection. The contrastive
   *loss* is measured in projection space and this evaluation in pooled space;
   they can move in opposite directions.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tokenizers.ascii_char import (
    PAD_ID,
    ASCIICharPreprocessor,
    encode_ascii,
)
from dl_techniques.metrics.embedding_quality import (
    alignment,
    anisotropy,
    effective_rank,
    embedding_norm_stats,
    l2_normalize,
    ranking_metrics,
    uniformity,
)
from dl_techniques.utils.logger import logger

# Registrar import: every model class is registered for serialization only when
# its module is imported, so `keras.saving.load_model` on an encoder.keras fails
# with a deserialization error unless the four arms have been imported first.
# `.config` imports all of them. Do NOT delete this as an unused import.
from . import config as _config_registrar  # noqa: F401
from .paths import encoder_path, eval_path

# ---------------------------------------------------------------------

__all__ = [
    "DEFAULT_TFDS_DATA_DIR",
    "EvalConfig",
    "embed_texts",
    "evaluate_run",
    "load_encoder",
    "main",
    "sst2_probe_eval",
    "squad_retrieval_eval",
]

DEFAULT_TFDS_DATA_DIR = "/media/arxwn/data0_4tb/datasets/tensorflow_datasets"


@dataclass
class EvalConfig:
    """Knobs for one evaluation pass.

    :param tfds_data_dir: Local TFDS cache. Nothing is ever downloaded.
    :param max_length: Character window; must not exceed the encoder's training
        length, since the position table is sized at build time.
    :param max_queries: Cap on SQuAD queries, sampled at random rather than
        taken as a prefix.
    :param probe_train_n: Cap on SST-2 probe training sentences.
    :param batch_size: Forward-pass batch size.
    :param seed: Controls the query subsample, the probe subsample and the
        uniformity subsample.
    :param run_squad: Whether to run the retrieval task.
    :param run_sst2: Whether to run the probe task.
    """

    tfds_data_dir: str = DEFAULT_TFDS_DATA_DIR
    max_length: int = 256
    max_queries: int = 2000
    probe_train_n: int = 8000
    batch_size: int = 64
    seed: int = 0
    run_squad: bool = True
    run_sst2: bool = True


def load_encoder(run_dir: str) -> keras.Model:
    """Load the encoder a training cell wrote.

    :param run_dir: The run directory.
    :type run_dir: str
    :return: The loaded encoder.
    :rtype: keras.Model
    :raises FileNotFoundError: If no encoder was written.
    """
    path = encoder_path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no encoder at {path}; the training cell did not finish stage 1"
        )
    return keras.saving.load_model(path, compile=False)


def _decode(value: Any) -> str:
    """Decode a TFDS bytes field the same way the training stream does."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def embed_texts(
    encoder: keras.Model,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int = 64,
    length_sorted: bool = True,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> Tuple[np.ndarray, float]:
    """Embed texts with the encoder's pooled output.

    Uses the SAME normalization the packed MLM stream used (lowercase, NFKD
    fold). A mismatch here would silently degrade every arm.

    When ``length_sorted`` is set, texts are grouped by encoded length and each
    batch is padded to ITS OWN maximum rather than to ``max_length``. That
    minimizes padding, which matters because one arm's block cannot honour a
    padding mask. The original input order is restored before returning.

    :param encoder: A loaded encoder.
    :param texts: Input strings.
    :param max_length: Character window, including the two framing tokens.
    :param batch_size: Forward-pass batch size.
    :param length_sorted: Sort by length to minimize padding.
    :param lowercase: Case-fold, matching training.
    :param normalize_unicode: NFKD-fold, matching training.
    :return: ``(embeddings (n, d) float32, mean_pad_fraction)``.
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), 0.0

    budget = max_length - 2
    content = [
        encode_ascii(
            t, lowercase=lowercase, normalize_unicode=normalize_unicode
        )[:budget]
        for t in texts
    ]
    lengths = np.array([len(c) + 2 for c in content], dtype=np.int64)

    order = np.argsort(lengths, kind="stable") if length_sorted else np.arange(n)

    preprocessor = ASCIICharPreprocessor(
        max_length=max_length,
        lowercase=lowercase,
        normalize_unicode=normalize_unicode,
    )
    cls_id, sep_id = preprocessor.cls_token_id, preprocessor.sep_token_id

    chunks: List[np.ndarray] = []
    padded_total = 0
    real_total = 0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        width = int(lengths[idx].max())
        ids = np.full((len(idx), width), PAD_ID, dtype=np.int32)
        for row, j in enumerate(idx):
            seq = [cls_id] + content[j] + [sep_id]
            ids[row, : len(seq)] = seq
        mask = (ids != PAD_ID).astype(np.int32)

        padded_total += ids.size
        real_total += int(mask.sum())

        out = encoder(
            {"input_ids": ids, "attention_mask": mask}, training=False
        )["pooled_output"]
        chunks.append(
            np.asarray(keras.ops.convert_to_numpy(out), dtype=np.float32)
        )

    stacked = np.concatenate(chunks, axis=0)
    restored = np.empty_like(stacked)
    restored[order] = stacked
    pad_fraction = 1.0 - (real_total / padded_total) if padded_total else 0.0
    return restored, float(pad_fraction)


def _load_tfds(name: str, split: str, data_dir: str):
    """Load a TFDS split, never downloading."""
    import tensorflow_datasets as tfds

    return tfds.load(
        name,
        split=split,
        data_dir=data_dir,
        download=False,
        shuffle_files=False,
    )


def squad_retrieval_eval(
    encoder: keras.Model,
    cfg: EvalConfig,
) -> Dict[str, float]:
    """Question-to-paragraph retrieval on SQuAD v1.1 validation.

    The pool is the set of UNIQUE contexts. Deduplication is mandatory, not
    tidy-up: SQuAD has roughly five questions per paragraph, so without it a
    retrieval of a byte-identical passage counts as an error and the metric is
    wrong by construction.

    Queries are sampled at random, never taken as a prefix -- SQuAD is ordered
    by article title, so the first N rows cover a handful of topics and turn
    retrieval into topic classification.

    :param encoder: A loaded encoder.
    :param cfg: Evaluation configuration.
    :return: Flat mapping of ``squad_*`` metrics.
    """
    rows = list(
        _load_tfds("squad/v1.1", "validation", cfg.tfds_data_dir)
        .as_numpy_iterator()
    )
    questions = [_decode(r["question"]) for r in rows]
    contexts = [_decode(r["context"]) for r in rows]

    pool: List[str] = []
    pool_index: Dict[str, int] = {}
    for ctx in contexts:
        if ctx not in pool_index:
            pool_index[ctx] = len(pool)
            pool.append(ctx)
    truth_all = np.array([pool_index[c] for c in contexts], dtype=np.int64)

    rng = np.random.default_rng(cfg.seed)
    if len(questions) > cfg.max_queries:
        picked = rng.permutation(len(questions))[: cfg.max_queries]
    else:
        picked = np.arange(len(questions))
    queries = [questions[i] for i in picked]
    truth = truth_all[picked]

    logger.info(
        f"SQuAD retrieval: {len(queries)} queries over {len(pool)} unique "
        f"contexts (from {len(rows)} rows)"
    )

    ctx_emb, ctx_pad = embed_texts(
        encoder, pool, max_length=cfg.max_length, batch_size=cfg.batch_size
    )
    q_emb, q_pad = embed_texts(
        encoder, queries, max_length=cfg.max_length, batch_size=cfg.batch_size
    )

    ctx_unit = l2_normalize(ctx_emb)
    q_unit = l2_normalize(q_emb)
    similarity = q_unit @ ctx_unit.T

    out: Dict[str, float] = {
        f"squad_{k}": v for k, v in ranking_metrics(similarity, truth).items()
    }
    out["squad_ctx_pad_fraction"] = ctx_pad
    out["squad_q_pad_fraction"] = q_pad
    out["squad_alignment"] = alignment(
        q_unit, ctx_unit[truth], already_normalized=True
    )
    out["squad_uniformity"] = uniformity(
        ctx_unit, already_normalized=True, rng=np.random.default_rng(cfg.seed)
    )
    out["squad_ctx_anisotropy"] = anisotropy(ctx_unit, already_normalized=True)
    out["squad_ctx_effective_rank"] = effective_rank(ctx_emb)
    for key, value in embedding_norm_stats(ctx_emb).items():
        out[f"squad_ctx_{key}"] = value
    return out


def sst2_probe_eval(
    encoder: keras.Model,
    cfg: EvalConfig,
) -> Dict[str, float]:
    """Linear probe on frozen SST-2 embeddings.

    The encoder is called once with ``training=False`` and its outputs converted
    to numpy; the probe is fitted by sklearn on that matrix. **There is no
    gradient path to the encoder** -- this measures the space, not a fine-tune.
    Do not "improve" it into one.

    ``C`` is swept by cross-validation on the probe-training subsample only. The
    arms are depth- and width-matched, not parameter-matched, so hidden widths
    differ at the same variant name; with an L2 penalty on unit-norm features a
    fixed ``C`` would favour whichever width happened to suit it, and the sweep
    removes that from the comparison.

    :param encoder: A loaded encoder.
    :param cfg: Evaluation configuration.
    :return: Flat mapping of ``sst2_*`` metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    train_rows = list(
        _load_tfds("glue/sst2", "train", cfg.tfds_data_dir).as_numpy_iterator()
    )
    # The GLUE `test` split is unlabelled (label = -1); validation is the test set.
    val_rows = list(
        _load_tfds("glue/sst2", "validation", cfg.tfds_data_dir)
        .as_numpy_iterator()
    )

    rng = np.random.default_rng(cfg.seed)
    if len(train_rows) > cfg.probe_train_n:
        picked = rng.permutation(len(train_rows))[: cfg.probe_train_n]
        train_rows = [train_rows[i] for i in picked]

    train_texts = [_decode(r["sentence"]) for r in train_rows]
    train_y = np.array([int(r["label"]) for r in train_rows])
    val_texts = [_decode(r["sentence"]) for r in val_rows]
    val_y = np.array([int(r["label"]) for r in val_rows])

    logger.info(
        f"SST-2 probe: {len(train_texts)} train / {len(val_texts)} validation"
    )

    train_x, train_pad = embed_texts(
        encoder, train_texts, max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )
    val_x, val_pad = embed_texts(
        encoder, val_texts, max_length=cfg.max_length, batch_size=cfg.batch_size
    )
    train_x = l2_normalize(train_x)
    val_x = l2_normalize(val_x)

    search = GridSearchCV(
        LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000, random_state=cfg.seed
        ),
        param_grid={"C": (0.01, 0.1, 1.0, 10.0, 100.0)},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed),
        scoring="accuracy",
        # Single-threaded on purpose: cells already run sequentially, and
        # sklearn thread pools inside a TF process are a known nondeterminism
        # source.
        n_jobs=1,
    )
    search.fit(train_x, train_y)

    counts = np.bincount(val_y)
    return {
        "sst2_probe_accuracy": float(search.score(val_x, val_y)),
        "sst2_probe_cv_accuracy": float(search.best_score_),
        "sst2_probe_best_c": float(search.best_params_["C"]),
        "sst2_probe_train_n": float(len(train_texts)),
        "sst2_val_n": float(len(val_texts)),
        "sst2_majority_baseline": float(counts.max() / counts.sum()),
        "sst2_train_pad_fraction": train_pad,
        "sst2_pad_fraction": val_pad,
    }


def evaluate_run(run_dir: str, cfg: Optional[EvalConfig] = None) -> Dict[str, float]:
    """Evaluate one finished training cell and write ``eval.json``.

    :param run_dir: The run directory.
    :param cfg: Evaluation configuration; defaults are used when omitted.
    :return: A flat mapping of metrics, also written to disk.
    """
    cfg = cfg or EvalConfig()
    encoder = load_encoder(run_dir)

    results: Dict[str, float] = {
        "eval_seed": float(cfg.seed),
        "eval_max_length": float(cfg.max_length),
    }
    if cfg.run_squad:
        results.update(squad_retrieval_eval(encoder, cfg))
    if cfg.run_sst2:
        results.update(sst2_probe_eval(encoder, cfg))
    results["eval_ok"] = True

    path = eval_path(run_dir)
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Wrote {path}")
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained embeddings_experimental encoder on SQuAD "
            "retrieval and an SST-2 linear probe."
        )
    )
    defaults = EvalConfig()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--tfds-data-dir", type=str, default=defaults.tfds_data_dir)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--max-queries", type=int, default=defaults.max_queries)
    parser.add_argument("--probe-train-n", type=int, default=defaults.probe_train_n)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--no-squad", dest="run_squad", action="store_false")
    parser.add_argument("--no-sst2", dest="run_sst2", action="store_false")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :return: Process exit code.
    """
    args = parse_args(argv)
    from train.common import setup_gpu

    setup_gpu(args.gpu)
    cfg = EvalConfig(
        tfds_data_dir=args.tfds_data_dir,
        max_length=args.max_length,
        max_queries=args.max_queries,
        probe_train_n=args.probe_train_n,
        batch_size=args.batch_size,
        seed=args.seed,
        run_squad=args.run_squad,
        run_sst2=args.run_sst2,
    )
    try:
        results = evaluate_run(args.run_dir, cfg)
    except Exception:
        logger.error("evaluation failed", exc_info=True)
        return 1
    for key in sorted(results):
        logger.info(f"  {key} = {results[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
