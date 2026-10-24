"""Controls for the topographic contrast: framing, length and label shuffle.

A contrast between two stimulus conditions is only worth quoting next to what the
same pipeline gives when the contrast is NOT the one under test. Three controls
answer the three ways the smoke-set contrast was found to mislead (D-024 to
D-027, D-029), and this module makes them reproducible from a finished run so the
numbers in the README do not rest on a script nobody can run:

* :func:`framing_table` -- the SAME contrast under the shipped context, other
  contexts, ``<eot>`` only and no prefix at all. The counts depend on the
  framing; Moran's I of the unthresholded map barely does.
* :func:`length_control` -- word lists of ``short`` against ``long`` words drawn
  from ONE vocabulary (3 against 7, and the calibration 5 against 5). Any
  contrast between them is length, not content.
* :func:`label_shuffle_null` -- every split of the stimuli into equal halves,
  scored with the same pipeline: how often a labelling at least as good as the
  real one arises from the sentences alone.

:func:`synthetic_illustration` draws what outlines look like when a unit
population IS selective, from made-up responses that go through the same chain.

Every contrast is scored by :func:`train.topolm.common.score_contrast`, the chain
:func:`~train.topolm.common.evaluate_topography` also runs (t-map, BH-FDR across
all taps, clusters, Moran's I), so a control scores exactly what the report
scores. The draws are seeded and their number is a parameter.

:func:`facts` writes those controls, per run, into a ``figure_facts.json``-shaped
file, and ``python -m train.topolm.controls --help`` lists the command line. The
module imports nothing from a plan directory.
"""

import argparse
import itertools
import json
import math
import os
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import keras
import numpy as np
from scipy.ndimage import gaussian_filter

from dl_techniques.layers.regularization.spatial_smoothness import SpatialLayout
from dl_techniques.metrics.spatial_autocorrelation import morans_i_grid
from dl_techniques.utils.logger import logger
from train.topolm import common as C

# ---------------------------------------------------------------------

#: Other contexts of different topic and length (31, 59 and 123 gpt2 tokens),
#: written after the shipped :data:`~train.topolm.common.STIMULUS_CONTEXT` and
#: reported whatever their numbers are. They show how much the counts depend on
#: the choice of context; none of them was chosen by its result.
ALTERNATIVE_CONTEXTS: Dict[str, str] = {
    "physics_31": (
        "In physics, a field assigns a number to every point in space and time, "
        "such as the temperature at each place on a weather map of the world."
    ),
    "sport_59": (
        "He played as a midfielder for several clubs in the national league "
        "before retiring in 2004. After his playing career he became a coach, "
        "and he led the youth team to two national titles. He was later "
        "appointed manager of the senior side and held the post for six years, "
        "until the club moved."
    ),
    "history_123": (
        "The monastery was founded in the twelfth century by a small group of "
        "monks who had travelled north from the abbey at Clairvaux. Over the "
        "following centuries it grew into a large and wealthy community, owning "
        "farms, mills and several churches across the region. In 1538 the king "
        "ordered the dissolution of all such houses, and the lands were sold to "
        "local families. Only the gatehouse and part of the cloister survive "
        "today, and they are open to the public on summer weekends. Visitors "
        "can also walk the old path that the monks used to reach their fields, "
        "which runs along the edge of a wood."
    ),
}

#: The word counts of the length control: the contrast (3 against 7) and its
#: calibration (5 against 5, which must give nothing).
DEFAULT_LENGTHS: Tuple[Tuple[int, int], ...] = ((3, 7), (5, 5))

#: Past this many splits the null must be subsampled (``max_splits``): C(20, 10)
#: is 184756 contrasts.
MAX_EXHAUSTIVE_SPLITS = 5000

#: The framing name of the bare sentence, with nothing before it.
FRAMING_NONE = "none"

# ---------------------------------------------------------------------


def condition_vocabulary(
    stimuli: Mapping[str, Sequence[str]], condition: str
) -> List[str]:
    """The sorted distinct words of one condition's sentences.

    Contract. ``stimuli`` is ``{condition: [sentence, ...]}``; the words are the
    whitespace-separated tokens. Raises ``ValueError`` naming the condition when
    it is absent.
    """
    if condition not in stimuli:
        raise ValueError(
            f"condition {condition!r} absent from the stimuli "
            f"(available: {sorted(stimuli)})"
        )
    return sorted({w for sentence in stimuli[condition] for w in sentence.split()})


def _finite_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _score(
    backbone: Any,
    values_a: Mapping[str, np.ndarray],
    values_b: Mapping[str, np.ndarray],
    condition_a: str,
    condition_b: str,
    min_cluster_size: int,
    fdr_alpha: float,
) -> Dict[str, Any]:
    """One contrast, summarised: the numbers every control reports.

    ``mean_morans_i`` is the mean over the taps of Moran's I of the unthresholded
    t-map (queen weights), over the taps where it is defined, exactly as
    ``evaluate_topography`` averages it.
    """
    # DECISION plan-2026-10-23T190557-972a9ac5/D-029: a control scores a contrast
    # with the SAME chain the report uses. Do NOT copy the t-map, BH-FDR and
    # cluster steps into this module (a second hand-kept copy drifts, and a
    # control that scores something else than the report is not a control): call
    # common.score_contrast. The controls are tracked code so the README numbers
    # can be regenerated from a run; do NOT read a script or file from a plan
    # directory here. See decisions.md D-029; guarded by TestTheSharedScorer,
    # TestTheFramingTable (each row equals evaluate_topography) and
    # TestTheModuleStandsOnTheRepoAlone.
    scored = C.score_contrast(
        backbone, values_a, values_b, condition_a, condition_b,
        min_cluster_size=min_cluster_size, fdr_alpha=fdr_alpha,
    )
    significant = [int(entry["sig_grid"].sum()) for entry in scored.values()]
    return {
        "significant_units_per_tap": significant,
        "mean_significant_units_per_tap": float(np.mean(significant)),
        "clusters_a": sum(len(e["clusters"][condition_a]) for e in scored.values()),
        "clusters_b": sum(len(e["clusters"][condition_b]) for e in scored.values()),
        "mean_morans_i": _finite_mean(
            [morans_i_grid(e["t_grid"], connectivity="queen") for e in scored.values()]
        ),
    }


def _extract(
    backbone: Any,
    preprocessor: Any,
    prompts: Sequence[str],
    context: Optional[str],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    return C.extract_tap_activations(
        backbone, list(prompts), preprocessor, batch_size=batch_size, context=context
    )


def _bare_activations(
    backbone: Any, preprocessor: Any, prompts: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Each sentence alone, with nothing before it, pooled over every position.

    One forward pass per sentence and no padding, so no framing, no pad and no
    batch can matter. Needs the preprocessor's raw encoder
    (``preprocessor.tokenizer.encode``). Same return shape as
    :func:`~train.topolm.common.extract_tap_activations`.
    """
    encoder = preprocessor.tokenizer
    if not backbone.built:
        backbone(np.asarray([encoder.encode(prompts[0])], dtype="int32"), training=False)
    rows: Dict[str, List[np.ndarray]] = {tap.path: [] for tap in backbone.tap_layers}
    for prompt in prompts:
        ids = np.asarray([encoder.encode(prompt)], dtype="int32")
        if ids.shape[1] == 0:
            raise ValueError(f"prompt {prompt!r} encodes to nothing")
        captures = [C.TapCapture() for _ in backbone.tap_layers]
        backbone(ids, training=False, taps=tuple(captures))
        for tap, capture in zip(backbone.tap_layers, captures):
            values = keras.ops.convert_to_numpy(capture.value).astype("float64")
            rows[tap.path].append(values[0].mean(axis=0))
    return {path: np.stack(values) for path, values in rows.items()}


# ---------------------------------------------------------------------
# the framing table
# ---------------------------------------------------------------------


def framing_table(
    backbone: Any,
    preprocessor: Any,
    stimuli: Optional[Mapping[str, Sequence[str]]] = None,
    contrast: Tuple[str, str] = ("a", "b"),
    extra_contexts: Optional[Mapping[str, str]] = None,
    min_cluster_size: int = 10,
    fdr_alpha: float = 0.05,
    batch_size: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """The same contrast under every framing: context, eot, no prefix.

    Contract. ``stimuli`` defaults to the smoke set. Returns an ordered
    ``{row: {...}}`` with the rows ``'context'`` (the shipped
    :data:`~train.topolm.common.STIMULUS_CONTEXT`), ``'eot'`` (``<eot>``
    sentence), ``'none'`` (the bare sentence) and ``'context:<name>'`` for each of
    ``extra_contexts``. (A ``'cls'`` row, ``[CLS] sentence [SEP]``, measured the
    call-only framing and went with it, D-031 f.) A row holds ``framing``
    (``'context+eot'``, ``'eot'`` or ``'none'``), ``context_tokens``, and
    ``significant_units_per_tap``, ``mean_significant_units_per_tap``,
    ``clusters_a``, ``clusters_b`` and ``mean_morans_i`` of the contrast. The
    preprocessor needs a raw encoder (``TypeError`` otherwise). Failure mode: the
    extraction's own errors (a window that does not fit).
    """
    stimuli = C.SMOKE_STIMULI if stimuli is None else stimuli
    condition_a, condition_b = contrast
    plan: List[Tuple[str, str]] = [
        ("context", C.STIMULUS_CONTEXT),
        ("eot", ""),
        ("none", FRAMING_NONE),
    ]
    plan += [(f"context:{name}", text) for name, text in (extra_contexts or {}).items()]
    max_positions = getattr(backbone, "max_seq_len", None)

    table: Dict[str, Dict[str, Any]] = {}
    for row, context in plan:
        if context == FRAMING_NONE:
            framing, context_tokens = FRAMING_NONE, 0
            acts_a = _bare_activations(backbone, preprocessor, stimuli[condition_a])
            acts_b = _bare_activations(backbone, preprocessor, stimuli[condition_b])
        else:
            block = C.stimulus_summary(
                preprocessor, {condition_a: stimuli[condition_a]}, max_positions,
                context=context,
            )
            framing, context_tokens = block["framing"], block["context_tokens"]
            acts_a = _extract(backbone, preprocessor, stimuli[condition_a], context, batch_size)
            acts_b = _extract(backbone, preprocessor, stimuli[condition_b], context, batch_size)
        table[row] = {
            "framing": framing,
            "context_tokens": context_tokens,
            **_score(backbone, acts_a, acts_b, condition_a, condition_b,
                     min_cluster_size, fdr_alpha),
        }
    return table


# ---------------------------------------------------------------------
# the length control
# ---------------------------------------------------------------------


def length_control(
    backbone: Any,
    preprocessor: Any,
    vocab_a: Sequence[str],
    vocab_b: Sequence[str],
    n_draws: int = 12,
    seed: int = 0,
    lengths: Sequence[Tuple[int, int]] = DEFAULT_LENGTHS,
    lists_per_side: int = 5,
    context: Optional[str] = C.STIMULUS_CONTEXT,
    min_cluster_size: int = 10,
    fdr_alpha: float = 0.05,
    batch_size: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Word lists of ``short`` against ``long`` words from one vocabulary.

    The reviewer's construction. For each vocabulary (``vocab_a`` is keyed ``'a'``,
    ``vocab_b`` ``'b'``), each ``(short, long)`` of ``lengths`` and each of
    ``n_draws`` draws: ``lists_per_side`` lists of ``short`` distinct words and
    as many of ``long``, every word drawn from that one vocabulary without
    replacement inside a list, contrasted with the same pipeline as the real
    contrast. Content cannot differ between the sides (same words), so a large
    count at 3 against 7 is length alone, and the 5 against 5 row is the
    calibration (it must be 0).

    Contract. Seeded: one ``numpy`` generator from ``seed``, drawn in the fixed
    order vocabulary, lengths, draw, short side then long side; the same arguments
    give the same lists and numbers. Returns ``{'<v>_vocab_<s>v<l>': {...}}`` with
    ``vocabulary``, ``short_words``, ``long_words``, ``draws``, ``values`` (mean
    significant units per tap, one per draw), ``median``, ``min``, ``max``,
    ``clusters_median`` and ``clusters_max`` (clusters a plus b per draw) and, for
    a preprocessor with a raw encoder, ``mean_tokens`` ``[short, long]``. Raises
    ``ValueError`` when a vocabulary has fewer distinct words than the longest
    list, or ``n_draws`` or ``lists_per_side`` is below 1.
    """
    if n_draws < 1 or lists_per_side < 1:
        raise ValueError(
            f"n_draws and lists_per_side must be >= 1, got {n_draws}, {lists_per_side}"
        )
    longest = max(max(pair) for pair in lengths)
    vocabularies = {"a": sorted(set(vocab_a)), "b": sorted(set(vocab_b))}
    for name, words in vocabularies.items():
        if len(words) < longest:
            raise ValueError(
                f"vocabulary {name!r} has {len(words)} distinct words, fewer than "
                f"the {longest} the longest list needs"
            )
    encoder = getattr(preprocessor, "tokenizer", None)
    rng = np.random.default_rng(seed)

    result: Dict[str, Dict[str, Any]] = {}
    for name, words in vocabularies.items():
        for short, long in lengths:
            values, clusters = [], []
            tokens: List[List[float]] = [[], []]
            for _ in range(n_draws):
                sides = []
                for count in (short, long):
                    lists = [
                        " ".join(words[i] for i in rng.choice(len(words), count, replace=False))
                        for _ in range(lists_per_side)
                    ]
                    sides.append(lists)
                    if encoder is not None:
                        tokens[len(sides) - 1] += [len(encoder.encode(t)) for t in lists]
                scored = _score(
                    backbone,
                    _extract(backbone, preprocessor, sides[0], context, batch_size),
                    _extract(backbone, preprocessor, sides[1], context, batch_size),
                    "short", "long", min_cluster_size, fdr_alpha,
                )
                values.append(scored["mean_significant_units_per_tap"])
                clusters.append(scored["clusters_a"] + scored["clusters_b"])
            entry: Dict[str, Any] = {
                "vocabulary": name,
                "short_words": short,
                "long_words": long,
                "draws": n_draws,
                "values": values,
                "median": float(np.median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "clusters_median": float(np.median(clusters)),
                "clusters_max": int(max(clusters)),
            }
            if encoder is not None:
                entry["mean_tokens"] = [float(np.mean(t)) for t in tokens]
            result[f"{name}_vocab_{short}v{long}"] = entry
    return result


# ---------------------------------------------------------------------
# the label-shuffle null
# ---------------------------------------------------------------------


def label_shuffle_null(
    backbone: Any,
    preprocessor: Any,
    stimuli: Optional[Mapping[str, Sequence[str]]] = None,
    contrast: Tuple[str, str] = ("a", "b"),
    context: Optional[str] = C.STIMULUS_CONTEXT,
    min_cluster_size: int = 10,
    fdr_alpha: float = 0.05,
    batch_size: int = 8,
    max_splits: Optional[int] = None,
    seed: int = 0,
    top: int = 6,
) -> Dict[str, Any]:
    """Every split of the sentences into two equal halves, scored like the real one.

    The responses to the ``2n`` sentences are extracted once; each split (``n``
    sentences called condition a, the rest b) is contrasted with the same
    pipeline. The real labelling is ``a = the first n sentences``. A split and its
    mirror (the complement) give the same counts, so the real labelling always
    has a tie, and ``p = n_at_or_above / n_splits`` cannot go below
    ``2 / n_splits``.

    Contract. The two conditions must have the same number of sentences, at least
    2. All ``C(2n, n)`` splits are scored (252 for n = 5); above
    :data:`MAX_EXHAUSTIVE_SPLITS` a ``max_splits`` is required, and then a seeded
    random subset of ``max_splits`` splits that always holds the real one is used.
    Returns ``n_splits``, ``real`` (``a_index`` and the contrast's numbers),
    ``per_split`` (the same for every split, lexicographic), ``n_at_or_above``,
    ``n_strictly_above``, ``p``, ``rank_from_top`` (``n_strictly_above + 1``),
    ``competitors`` (the splits strictly above the real one, best first), ``top``
    (the best ``top`` splits), ``sig`` (``median``, ``mean``, ``p95``, ``min``,
    ``max`` of the mean significant units per tap), ``n_splits_with_a_cluster``
    and ``morans_i`` (``min``, ``median``, ``max``, ``real`` and
    ``real_rank_from_top``: how many splits have a Moran's I at least the real
    one's). Raises ``ValueError`` for unequal conditions or an unaffordable
    split count.
    """
    stimuli = C.SMOKE_STIMULI if stimuli is None else stimuli
    condition_a, condition_b = contrast
    n = len(stimuli[condition_a])
    if len(stimuli[condition_b]) != n:
        raise ValueError(
            f"the label shuffle needs equal conditions, got {n} sentences for "
            f"{condition_a!r} and {len(stimuli[condition_b])} for {condition_b!r}"
        )
    if n < 2:
        raise ValueError(f"the label shuffle needs at least 2 sentences per condition, got {n}")
    real_index = tuple(range(n))
    splits = list(itertools.combinations(range(2 * n), n))
    if max_splits is not None and len(splits) > max_splits:
        others = [s for s in splits if s != real_index]
        chosen = np.random.default_rng(seed).choice(len(others), max_splits - 1, replace=False)
        splits = sorted([real_index] + [others[i] for i in chosen])
    elif max_splits is None and len(splits) > MAX_EXHAUSTIVE_SPLITS:
        raise ValueError(
            f"{len(splits)} splits for {n} + {n} sentences: pass max_splits to score "
            f"a seeded subset (the exhaustive limit is {MAX_EXHAUSTIVE_SPLITS})"
        )

    sentences = list(stimuli[condition_a]) + list(stimuli[condition_b])
    responses = _extract(backbone, preprocessor, sentences, context, batch_size)
    per_split: List[Dict[str, Any]] = []
    for index in splits:
        rest = [i for i in range(2 * n) if i not in index]
        scored = _score(
            backbone,
            {path: values[list(index)] for path, values in responses.items()},
            {path: values[rest] for path, values in responses.items()},
            "a", "b", min_cluster_size, fdr_alpha,
        )
        per_split.append({"a_index": list(index), **scored})

    real = next(entry for entry in per_split if tuple(entry["a_index"]) == real_index)
    sig = np.array([e["mean_significant_units_per_tap"] for e in per_split])
    morans = np.array([e["mean_morans_i"] for e in per_split])
    real_sig = real["mean_significant_units_per_tap"]
    above = sorted(
        (e for e in per_split if e["mean_significant_units_per_tap"] > real_sig),
        key=lambda e: (-e["mean_significant_units_per_tap"], e["a_index"]),
    )
    best_first = sorted(
        per_split, key=lambda e: (-e["mean_significant_units_per_tap"], e["a_index"])
    )
    return {
        "n_splits": len(per_split),
        "real": real,
        "per_split": per_split,
        "n_at_or_above": int((sig >= real_sig).sum()),
        "n_strictly_above": len(above),
        "p": float((sig >= real_sig).sum() / len(per_split)),
        "rank_from_top": len(above) + 1,
        "competitors": above,
        "top": best_first[:top],
        "sig": {
            "median": float(np.median(sig)),
            "mean": float(sig.mean()),
            "p95": float(np.percentile(sig, 95)),
            "min": float(sig.min()),
            "max": float(sig.max()),
        },
        "n_splits_with_a_cluster": sum(
            (e["clusters_a"] + e["clusters_b"]) > 0 for e in per_split
        ),
        "morans_i": {
            "min": float(np.nanmin(morans)),
            "median": float(np.nanmedian(morans)),
            "max": float(np.nanmax(morans)),
            "real": real["mean_morans_i"],
            "real_rank_from_top": int((morans >= real["mean_morans_i"]).sum()),
        },
    }


# ---------------------------------------------------------------------
# everything for one run, and the facts file
# ---------------------------------------------------------------------


def controls_for_run(
    backbone: Any,
    preprocessor: Any,
    stimuli: Optional[Mapping[str, Sequence[str]]] = None,
    contrast: Tuple[str, str] = ("a", "b"),
    n_draws: int = 12,
    seed: int = 0,
    lengths: Sequence[Tuple[int, int]] = DEFAULT_LENGTHS,
    extra_contexts: Optional[Mapping[str, str]] = None,
    min_cluster_size: int = 10,
    fdr_alpha: float = 0.05,
    batch_size: int = 8,
    max_splits: Optional[int] = None,
) -> Dict[str, Any]:
    """The three controls of one trained model, as one JSON-ready block.

    Contract. ``extra_contexts`` defaults to :data:`ALTERNATIVE_CONTEXTS`. Returns
    ``{observed_a_vs_b, framings, length_control, label_shuffle_null, n_draws,
    seed}``: ``observed_a_vs_b`` is the contrast under the shipped context,
    ``label_shuffle_null`` has its ``per_split`` list removed (252 entries per run
    would swamp the facts file). The block is strict JSON (no NaN). The length
    control draws its vocabularies from the two contrast conditions of ``stimuli``.
    """
    stimuli = C.SMOKE_STIMULI if stimuli is None else stimuli
    contexts = ALTERNATIVE_CONTEXTS if extra_contexts is None else extra_contexts
    common = dict(min_cluster_size=min_cluster_size, fdr_alpha=fdr_alpha, batch_size=batch_size)
    framings = framing_table(
        backbone, preprocessor, stimuli, contrast, extra_contexts=contexts, **common
    )
    null = label_shuffle_null(
        backbone, preprocessor, stimuli, contrast, max_splits=max_splits, seed=seed, **common
    )
    null.pop("per_split")
    observed = framings["context"]
    block = {
        "observed_a_vs_b": {
            key: observed[key]
            for key in ("mean_significant_units_per_tap", "clusters_a", "clusters_b",
                        "mean_morans_i")
        },
        "framings": framings,
        "length_control": length_control(
            backbone, preprocessor,
            condition_vocabulary(stimuli, contrast[0]),
            condition_vocabulary(stimuli, contrast[1]),
            n_draws=n_draws, seed=seed, lengths=lengths, **common,
        ),
        "label_shuffle_null": null,
        "n_draws": n_draws,
        "seed": seed,
    }
    return json.loads(json.dumps(block, allow_nan=False))


def load_run(run_dir: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """The backbone, the preprocessor and the config of a finished run.

    Read only: ``<run_dir>/checkpoints/final.keras`` and ``<run_dir>/config.json``
    are loaded, nothing is written. The preprocessor is built from the config
    exactly as the trainer builds it. Failure mode: a missing file raises
    ``FileNotFoundError``; a checkpoint without a TopoLM backbone raises
    ``ValueError``.
    """
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    model = keras.saving.load_model(os.path.join(run_dir, "checkpoints", "final.keras"))
    backbone = model if hasattr(model, "tap_layers") else next(
        (layer for layer in model.layers if hasattr(layer, "tap_layers")), None
    )
    if backbone is None:
        raise ValueError(f"{run_dir}/checkpoints/final.keras holds no backbone with taps")
    preprocessor = C.create_tokenizer(
        config["encoding_name"], config["max_seq_length"], config["cls_token_id"],
        config["sep_token_id"], config["pad_token_id"], config["mask_token_id"],
    )
    return backbone, preprocessor, config


def dump_facts(path: str, data: Mapping[str, Any]) -> None:
    """Write ``data`` the way ``figure_facts.json`` is stored: indent 1, no final newline.

    Loading the tracked file and dumping it back reproduces it byte for byte.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=1, allow_nan=False))


def facts(
    runs: Mapping[str, str],
    output_path: str,
    n_draws: int = 12,
    seed: int = 0,
    existing_path: Optional[str] = None,
    lengths: Sequence[Tuple[int, int]] = DEFAULT_LENGTHS,
    extra_contexts: Optional[Mapping[str, str]] = None,
    max_splits: Optional[int] = None,
) -> Dict[str, Any]:
    """Write the controls of each run into a ``figure_facts.json``-shaped file.

    Contract. ``runs`` maps a run name (``'2d'``, ``'3d'``) to its run directory.
    For each, the model is loaded read-only (:func:`load_run`), the controls of
    :func:`controls_for_run` are computed with the run's own ``min_cluster_size``,
    stimuli and contrast, and ``runs.<name>.controls`` is REPLACED and
    ``runs.<name>.stimuli`` set to the ``stimuli`` block of the run's
    ``topography_report.json`` (when it has one). Everything else in
    ``existing_path`` (the image facts, the timings, any key this function does not
    own) is carried over unchanged, and ``generated_from`` gains ``run_dirs``,
    ``n_draws``, ``controls_seed`` and ``controls_module``. The image facts and the
    GPU timings are measured outside this module and are never invented here.
    Returns the written dictionary. Failure mode: any exception from loading or
    scoring propagates before ``output_path`` is touched.
    """
    data: Dict[str, Any] = {}
    if existing_path is not None:
        with open(existing_path, encoding="utf-8") as handle:
            data = json.load(handle)
    data.setdefault("generated_from", {})
    data.setdefault("runs", {})
    for name, run_dir in runs.items():
        backbone, preprocessor, config = load_run(run_dir)
        logger.info(f"Controls for run {name!r}: {run_dir}")
        entry = data["runs"].setdefault(name, {})
        entry["controls"] = controls_for_run(
            backbone, preprocessor,
            stimuli=config.get("stimuli") or None,
            contrast=tuple(config.get("contrast_conditions") or ("a", "b")),
            n_draws=n_draws, seed=seed, lengths=lengths,
            extra_contexts=extra_contexts,
            min_cluster_size=int(config.get("min_cluster_size", 10)),
            max_splits=max_splits,
        )
        report_path = os.path.join(run_dir, "topography_report.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as handle:
                block = json.load(handle).get("stimuli")
            if block is not None:
                entry["stimuli"] = block
    data["generated_from"].update({
        "run_dirs": dict(runs),
        "n_draws": n_draws,
        "controls_seed": seed,
        "controls_module": "train.topolm.controls",
    })
    dump_facts(output_path, data)
    return data


# ---------------------------------------------------------------------
# the synthetic illustration
# ---------------------------------------------------------------------

#: The illustration's grid and its stimuli per condition (a t-test on 5 against 5
#: has 8 degrees of freedom, as the smoke contrast does).
ILLUSTRATION_GRID: Tuple[int, int] = (16, 16)
ILLUSTRATION_N_PER_CONDITION = 5

#: Planted selective blobs per panel, ``(row, column, sigma in cells, t at the
#: centre)``: two positive (more response to a) and one negative (more to b).
ILLUSTRATION_BLOBS: Dict[str, List[Tuple[float, float, float, float]]] = {
    "synthetic/tap_1": [(4.0, 4.5, 2.2, 7.0), (11.5, 11.0, 2.0, 6.0), (4.5, 12.0, 2.0, -6.0)],
    "synthetic/tap_2": [(3.5, 10.0, 2.4, 7.2), (12.0, 4.0, 2.1, 6.4), (10.5, 12.0, 2.2, -6.5)],
    "synthetic/tap_3": [(7.0, 3.5, 2.0, 6.5), (11.0, 10.5, 2.3, 7.0), (3.5, 11.5, 2.1, -5.8)],
    "synthetic/tap_4": [(5.0, 6.0, 2.5, 7.0), (12.5, 12.0, 1.9, 5.8), (11.5, 3.0, 2.0, -6.2)],
}


def synthetic_illustration(
    output_path: Optional[str] = None,
    seed: int = 100,
    fdr_alpha: float = 0.05,
    min_cluster_size: int = 10,
) -> Dict[str, Any]:
    """What outlines look like when a unit population IS selective: made-up responses.

    No model and no run. Each of the four panels is a made-up ``(10 stimuli, 256
    units)`` response matrix: spatially smooth unit-variance noise, plus a planted
    effect (two positive blobs and one negative one, :data:`ILLUSTRATION_BLOBS`)
    that is added to the five ``a`` stimuli and subtracted from the five ``b``
    ones. The matrices go through the real chain,
    :func:`~train.topolm.common.score_contrast` (Student's t-test, ONE BH-FDR
    correction across all four panels, queen cluster growth), and
    :func:`~train.topolm.plotting.plot_t_maps` draws the result, so the subtitle
    'after BH-FDR' is true: the mask is the BH-FDR output, not a threshold on t.
    The title and the note say SYNTHETIC.

    Contract. Seeded (panel ``k`` uses ``seed + k``); the same seed gives the same
    arrays and the same PNG. ``output_path`` is the PNG to write, or ``None`` to
    draw nothing. Returns ``{"facts": {...}, "arrays": {...}}``: ``facts`` is strict
    JSON (``kind: 'synthetic'``, the significance procedure, per-panel counts and
    cluster sizes, the planted blobs); ``arrays`` holds ``responses``, ``t_maps``,
    ``sig_grids``, ``label_grids`` (a) and ``label_grids_b`` per panel name.
    """
    from train.topolm import plotting  # matplotlib stays out of the module import

    height, width = ILLUSTRATION_GRID
    units = height * width
    n = ILLUSTRATION_N_PER_CONDITION
    rows, columns = np.mgrid[0:height, 0:width].astype("float64")

    taps, values_a, values_b = [], {}, {}
    responses: Dict[str, np.ndarray] = {}
    for panel, (name, blobs) in enumerate(ILLUSTRATION_BLOBS.items()):
        rng = np.random.default_rng(seed + panel)
        effect = np.zeros((height, width))
        for row, column, sigma, t_centre in blobs:
            # a mean difference of d between 5 and 5 unit-variance responses has
            # t = d / sqrt(2 / 5)
            effect += (t_centre * math.sqrt(2.0 / n)) * np.exp(
                -((rows - row) ** 2 + (columns - column) ** 2) / (2 * sigma**2)
            )
        noise = np.empty((2 * n, height, width))
        for stimulus in range(2 * n):
            smooth = gaussian_filter(rng.standard_normal((height, width)), 1.2)
            noise[stimulus] = smooth / smooth.std()
        matrix = noise.reshape(2 * n, units)
        matrix[:n] += effect.reshape(units) / 2.0
        matrix[n:] -= effect.reshape(units) / 2.0
        responses[name] = matrix
        values_a[name], values_b[name] = matrix[:n], matrix[n:]
        taps.append(types.SimpleNamespace(
            path=name,
            layout=SpatialLayout(units, grid_shape=ILLUSTRATION_GRID, permute=False),
        ))

    scored = C.score_contrast(
        types.SimpleNamespace(tap_layers=taps), values_a, values_b, "a", "b",
        min_cluster_size=min_cluster_size, fdr_alpha=fdr_alpha,
    )
    arrays = {
        "responses": responses,
        "t_maps": {k: v["t_grid"] for k, v in scored.items()},
        "sig_grids": {k: v["sig_grid"] for k, v in scored.items()},
        "label_grids": {k: v["label_grid"] for k, v in scored.items()},
        "label_grids_b": {k: v["label_grid_b"] for k, v in scored.items()},
    }
    if output_path is not None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plotting.plot_t_maps(
            arrays["t_maps"],
            sig_grids=arrays["sig_grids"],
            label_grids=arrays["label_grids"],
            label_grids_b=arrays["label_grids_b"],
            condition_a="a",
            condition_b="b",
            output_path=output_path,
            max_panels_per_row=4,
            suptitle=(
                "SYNTHETIC illustration (made-up responses, not a run): "
                "significance islands of a selective population"
            ),
            note="synthetic illustration, made-up responses, not a run",
        )
    facts = {
        "kind": "synthetic",
        "significance": (
            f"Student t-test, BH-FDR {fdr_alpha} jointly across the {len(scored)} panels, "
            f"clusters of at least {min_cluster_size} cells (queen connectivity)"
        ),
        "grid_shape": list(ILLUSTRATION_GRID),
        "panels": len(scored),
        "n_per_condition": n,
        "seed": seed,
        "fdr_alpha": fdr_alpha,
        "min_cluster_size": min_cluster_size,
        "blobs_per_panel": {k: [list(b) for b in v] for k, v in ILLUSTRATION_BLOBS.items()},
        "per_panel": {
            name: {
                "significant_units": int(entry["sig_grid"].sum()),
                "clusters_a": len(entry["clusters"]["a"]),
                "clusters_b": len(entry["clusters"]["b"]),
                "cluster_sizes_a": [len(c) for c in entry["clusters"]["a"]],
                "cluster_sizes_b": [len(c) for c in entry["clusters"]["b"]],
                "t_min": float(entry["t_grid"].min()),
                "t_max": float(entry["t_grid"].max()),
            }
            for name, entry in scored.items()
        },
        "written": output_path,
    }
    return {"facts": json.loads(json.dumps(facts, allow_nan=False)), "arrays": arrays}


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train.topolm.controls",
        description=(
            "Run the framing, length and label-shuffle controls of the "
            "topographic contrast on finished TopoLM runs and write them into a "
            "figure-facts JSON file. Read only on the runs. Set "
            "CUDA_VISIBLE_DEVICES= to keep it off the GPU."
        ),
    )
    parser.add_argument(
        "--run", action="append", default=[], metavar="NAME=DIR",
        help="a run to score, e.g. 2d=results/topolm_.../topolm_tiny_...; repeatable",
    )
    parser.add_argument(
        "--facts", default=None, metavar="PATH",
        help="the figure-facts JSON file to write (needed with --run)",
    )
    parser.add_argument(
        "--illustration", default=None, metavar="PNG",
        help="draw the SYNTHETIC illustration (made-up responses through the real "
             "BH-FDR chain) to this PNG; needs no run",
    )
    parser.add_argument(
        "--extend", default=None, metavar="PATH",
        help="an existing facts file whose other keys are carried over",
    )
    parser.add_argument("--n-draws", type=int, default=12,
                        help="random draws per length-control row (default 12)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the length-control draws and of a split subsample")
    parser.add_argument("--max-splits", type=int, default=None,
                        help="score a seeded subset of this many label splits")
    return parser


def _parse_runs(parser: argparse.ArgumentParser, specs: Sequence[str]) -> Dict[str, str]:
    runs: Dict[str, str] = {}
    for spec in specs:
        name, separator, directory = spec.partition("=")
        if not separator or not name or not directory:
            parser.error(f"--run expects NAME=DIR, got {spec!r}")
        runs[name] = directory
    return runs


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the arguments first, then draw the illustration and score the runs."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    runs = _parse_runs(parser, args.run)
    if not runs and args.illustration is None:
        parser.error("give --run NAME=DIR with --facts PATH, or --illustration PNG")
    if runs and args.facts is None:
        parser.error("--run needs --facts PATH, the file the controls are written to")
    if args.n_draws < 1:
        parser.error(f"--n-draws must be >= 1, got {args.n_draws}")
    if args.illustration is not None:
        synthetic_illustration(args.illustration)
        logger.info(f"Wrote {args.illustration}")
    if runs:
        facts(
            runs, args.facts, n_draws=args.n_draws, seed=args.seed,
            existing_path=args.extend, max_splits=args.max_splits,
        )
        logger.info(f"Wrote {args.facts}")


if __name__ == "__main__":
    main()
