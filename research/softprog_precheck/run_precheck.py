"""SoftProg premise precheck.

Question: do predicate-argument tuples paraphrase-align?
Concretely: on PAWS (paraphrases with high lexical overlap), does
PA-tuple match rate discriminate label=1 (paraphrase) from
label=0 (non-paraphrase, lexically-matched)?

If yes, structural retrieval has signal that dense embeddings would miss.
If no, the whole SoftProg thesis is dead.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import spacy
from datasets import load_dataset

OUT_DIR = Path(__file__).parent
RESULTS_FILE = OUT_DIR / "results.json"
SAMPLE_FILE = OUT_DIR / "samples.jsonl"

# Constraints from CLAUDE.md
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
random.seed(0)


def extract_tuple(doc) -> dict:
    """Return a structural skeleton: predicates with their canonical arguments.

    Captures every VERB with its (lemma, nsubj_lemma, dobj_lemma, iobj_lemma).
    Returns a frozenset-friendly dict for set comparison.
    """
    verbs = []
    for tok in doc:
        if tok.pos_ != "VERB":
            continue
        args = {"pred": tok.lemma_.lower()}
        for child in tok.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                args["subj"] = child.lemma_.lower()
            elif child.dep_ == "dobj":
                args["dobj"] = child.lemma_.lower()
            elif child.dep_ == "iobj":
                args["iobj"] = child.lemma_.lower()
            elif child.dep_ == "pobj":
                args.setdefault("pobjs", []).append(child.lemma_.lower())
        verbs.append(args)
    return {"verbs": verbs}


def tuple_signature(struct: dict, mode: str) -> frozenset:
    """Render extracted struct as a comparable set.

    mode="root":   {(pred,)} for the single ROOT verb (smallest signature).
    mode="pas":    {(pred, subj, dobj)} for every verb (medium).
    mode="full":   plus all pobjs and iobj (largest).
    """
    out = set()
    for v in struct["verbs"]:
        if mode == "root":
            out.add(("pred", v.get("pred", "")))
        elif mode == "pas":
            out.add((
                v.get("pred", ""),
                v.get("subj", ""),
                v.get("dobj", ""),
            ))
        elif mode == "full":
            base = (
                v.get("pred", ""),
                v.get("subj", ""),
                v.get("dobj", ""),
                v.get("iobj", ""),
            )
            pobjs = tuple(sorted(v.get("pobjs", [])))
            out.add(base + pobjs)
    return frozenset(out)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_lemmas(doc) -> frozenset:
    return frozenset(
        tok.lemma_.lower()
        for tok in doc
        if tok.pos_ in ("NOUN", "VERB", "PROPN", "ADJ") and not tok.is_stop
    )


def run() -> None:
    print("Loading spaCy en_core_web_sm…")
    nlp = spacy.load("en_core_web_sm")

    print("Loading PAWS (labeled_final, dev)…")
    ds = load_dataset("paws", "labeled_final", split="validation")
    print(f"PAWS dev size: {len(ds)}")

    # Sample 2000 pairs for speed; keep label balance natural.
    n = min(2000, len(ds))
    indices = random.sample(range(len(ds)), n)
    pairs = [ds[i] for i in indices]

    # Parse all unique sentences once.
    print("Parsing sentences…")
    sentence_to_struct = {}
    sentence_to_lemmas = {}
    all_sents = set()
    for p in pairs:
        all_sents.add(p["sentence1"])
        all_sents.add(p["sentence2"])
    all_sents = list(all_sents)
    print(f"Unique sentences: {len(all_sents)}")

    for doc, sent in zip(nlp.pipe(all_sents, batch_size=64), all_sents):
        sentence_to_struct[sent] = extract_tuple(doc)
        sentence_to_lemmas[sent] = content_lemmas(doc)

    # Per-pair metrics for each signature mode.
    results = {"modes": {}, "config": {"n_pairs": n, "seed": 0}}
    sample_lines = []

    for mode in ("root", "pas", "full"):
        pos_jacc = []
        neg_jacc = []
        pos_exact = 0
        neg_exact = 0
        pos_n = 0
        neg_n = 0
        for p in pairs:
            s1 = sentence_to_struct[p["sentence1"]]
            s2 = sentence_to_struct[p["sentence2"]]
            t1 = tuple_signature(s1, mode)
            t2 = tuple_signature(s2, mode)
            j = jaccard(t1, t2)
            exact = (t1 == t2 and len(t1) > 0)
            if p["label"] == 1:
                pos_jacc.append(j)
                pos_exact += int(exact)
                pos_n += 1
            else:
                neg_jacc.append(j)
                neg_exact += int(exact)
                neg_n += 1
        results["modes"][mode] = {
            "paraphrase_n": pos_n,
            "non_para_n": neg_n,
            "paraphrase_mean_jaccard": sum(pos_jacc) / max(pos_n, 1),
            "non_para_mean_jaccard": sum(neg_jacc) / max(neg_n, 1),
            "paraphrase_exact_match_rate": pos_exact / max(pos_n, 1),
            "non_para_exact_match_rate": neg_exact / max(neg_n, 1),
            "ratio_mean_jaccard": (
                (sum(pos_jacc) / max(pos_n, 1))
                / max(sum(neg_jacc) / max(neg_n, 1), 1e-9)
            ),
            "ratio_exact_match": (
                (pos_exact / max(pos_n, 1))
                / max(neg_exact / max(neg_n, 1), 1e-9)
            ),
        }

    # Random pair baseline: shuffle sentence2 among all pairs.
    print("Random pair baseline…")
    shuffled_s2 = [p["sentence2"] for p in pairs]
    random.shuffle(shuffled_s2)
    rand_jacc = {"root": [], "pas": [], "full": []}
    rand_lemma_jacc = []
    for p, s2_shuf in zip(pairs, shuffled_s2):
        s1_struct = sentence_to_struct[p["sentence1"]]
        s2_struct = sentence_to_struct[s2_shuf]
        for mode in ("root", "pas", "full"):
            t1 = tuple_signature(s1_struct, mode)
            t2 = tuple_signature(s2_struct, mode)
            rand_jacc[mode].append(jaccard(t1, t2))
        rand_lemma_jacc.append(
            jaccard(sentence_to_lemmas[p["sentence1"]], sentence_to_lemmas[s2_shuf])
        )
    for mode in ("root", "pas", "full"):
        results["modes"][mode]["random_pair_mean_jaccard"] = sum(rand_jacc[mode]) / len(rand_jacc[mode])
        results["modes"][mode]["ratio_para_vs_random"] = (
            results["modes"][mode]["paraphrase_mean_jaccard"]
            / max(results["modes"][mode]["random_pair_mean_jaccard"], 1e-9)
        )

    # Content-lemma baseline (the "dumb but strong" baseline structural retrieval must beat).
    print("Content-lemma baseline…")
    pos_lemma_j = []
    neg_lemma_j = []
    for p in pairs:
        j = jaccard(sentence_to_lemmas[p["sentence1"]], sentence_to_lemmas[p["sentence2"]])
        if p["label"] == 1:
            pos_lemma_j.append(j)
        else:
            neg_lemma_j.append(j)
    rand_lemma_mean = sum(rand_lemma_jacc) / max(len(rand_lemma_jacc), 1)
    results["content_lemma_baseline"] = {
        "paraphrase_mean_jaccard": sum(pos_lemma_j) / max(len(pos_lemma_j), 1),
        "non_para_mean_jaccard": sum(neg_lemma_j) / max(len(neg_lemma_j), 1),
        "random_pair_mean_jaccard": rand_lemma_mean,
        "ratio_para_vs_nonpara": (
            (sum(pos_lemma_j) / max(len(pos_lemma_j), 1))
            / max(sum(neg_lemma_j) / max(len(neg_lemma_j), 1), 1e-9)
        ),
        "ratio_para_vs_random": (
            (sum(pos_lemma_j) / max(len(pos_lemma_j), 1))
            / max(rand_lemma_mean, 1e-9)
        ),
    }

    # Coverage diagnostic: how many sentences had ANY PAS verb?
    pas_coverage = sum(
        1 for s in all_sents if len(tuple_signature(sentence_to_struct[s], "pas")) > 0
    )
    results["coverage"] = {
        "sentences_with_any_pas_tuple": pas_coverage,
        "total_unique_sentences": len(all_sents),
        "coverage_rate": pas_coverage / len(all_sents),
    }

    # Sample qualitative pairs.
    for p in pairs[:20]:
        sample_lines.append(json.dumps({
            "label": int(p["label"]),
            "s1": p["sentence1"],
            "s2": p["sentence2"],
            "s1_pas": list(tuple_signature(sentence_to_struct[p["sentence1"]], "pas")),
            "s2_pas": list(tuple_signature(sentence_to_struct[p["sentence2"]], "pas")),
        }))

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    SAMPLE_FILE.write_text("\n".join(sample_lines) + "\n")
    print(f"Results: {RESULTS_FILE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
