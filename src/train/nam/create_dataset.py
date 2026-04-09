"""
Create line-by-line arithmetic expression dataset files.

Generates text files where each line is a tab-separated record::

    expression<TAB>result<TAB>valid

Produces one file per curriculum phase plus a combined ``all.txt``.
Expressions span the full difficulty spectrum: single-op integer addition
up to deeply nested float expressions with division-by-zero cases.

Usage::

    python -m train.nam.create_dataset --output-dir data/nam --samples-per-phase 10000
    python -m train.nam.create_dataset --output-dir data/nam --samples-per-phase 50000 --seed 42

Output structure::

    data/nam/
    ├── phase_1.txt    # simple: 1-op, small ints, +/-
    ├── phase_2.txt    # medium: 1-2 ops, +/-/*
    ├── phase_3.txt    # hard: 2-4 ops, all operators
    ├── phase_4.txt    # parens: 2-4 ops, parentheses depth 1
    ├── phase_5.txt    # full: 2-8 ops, nested parens, floats
    ├── div_zero.txt   # curated division-by-zero cases
    ├── all.txt        # all phases combined + shuffled
    └── stats.json     # generation statistics
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from train.nam.data_generator import (
    ExpressionConfig,
    CURRICULUM,
    _generate_expr,
    _safe_eval,
)


# Extra configs not in the standard curriculum
EXTRA_CONFIGS: Dict[str, ExpressionConfig] = {
    # Targeted division-by-zero stress test
    "div_zero": ExpressionConfig(
        min_ops=1,
        max_ops=3,
        min_val=0,  # 0 as operand forces div-by-zero
        max_val=10,
        operators=["/", "+", "-"],
        allow_parentheses=True,
        max_paren_depth=1,
    ),
}


def _generate_samples(
    config: ExpressionConfig,
    n: int,
    max_retries: int = 3,
) -> List[Tuple[str, float, bool]]:
    """
    Generate n expression samples.

    Each sample is (expression_str, result, is_valid).
    Retries on malformed expressions (should be rare).
    """
    samples = []
    for _ in range(n):
        for _attempt in range(max_retries):
            expr = _generate_expr(config)
            result, valid = _safe_eval(expr)
            samples.append((expr, result, valid))
            break
    return samples


def _write_file(
    path: Path,
    samples: List[Tuple[str, float, bool]],
) -> Dict[str, int]:
    """
    Write samples to a tab-separated text file.

    Returns statistics dict.
    """
    valid_count = 0
    invalid_count = 0

    with open(path, "w", encoding="utf-8") as f:
        for expr, result, valid in samples:
            valid_int = 1 if valid else 0
            f.write(f"{expr}\t{result:.10g}\t{valid_int}\n")
            if valid:
                valid_count += 1
            else:
                invalid_count += 1

    return {
        "total": len(samples),
        "valid": valid_count,
        "invalid": invalid_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create NAM arithmetic expression dataset files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/nam",
        help="Directory to write dataset files",
    )
    parser.add_argument(
        "--samples-per-phase",
        type=int,
        default=10000,
        help="Number of samples per curriculum phase",
    )
    parser.add_argument(
        "--div-zero-samples",
        type=int,
        default=2000,
        help="Number of targeted division-by-zero samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    all_samples: List[Tuple[str, float, bool]] = []
    start_time = time.time()

    # Generate each curriculum phase
    for phase_name, config in CURRICULUM.items():
        print(f"Generating {phase_name}: {args.samples_per_phase} samples...", flush=True)
        samples = _generate_samples(config, args.samples_per_phase)
        path = output_dir / f"{phase_name}.txt"
        phase_stats = _write_file(path, samples)
        stats[phase_name] = phase_stats
        all_samples.extend(samples)
        print(
            f"  -> {path} "
            f"({phase_stats['valid']} valid, {phase_stats['invalid']} invalid)"
        )

    # Generate division-by-zero stress test
    print(f"Generating div_zero: {args.div_zero_samples} samples...", flush=True)
    div_samples = _generate_samples(
        EXTRA_CONFIGS["div_zero"], args.div_zero_samples
    )
    path = output_dir / "div_zero.txt"
    div_stats = _write_file(path, div_samples)
    stats["div_zero"] = div_stats
    all_samples.extend(div_samples)
    print(
        f"  -> {path} "
        f"({div_stats['valid']} valid, {div_stats['invalid']} invalid)"
    )

    # Write combined shuffled file
    print(f"Writing combined all.txt ({len(all_samples)} samples)...", flush=True)
    random.shuffle(all_samples)
    path = output_dir / "all.txt"
    all_stats = _write_file(path, all_samples)
    stats["all"] = all_stats
    print(f"  -> {path}")

    # Write stats
    elapsed = time.time() - start_time
    stats["_meta"] = {
        "samples_per_phase": args.samples_per_phase,
        "div_zero_samples": args.div_zero_samples,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 2),
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Stats: {stats_path}")
    print(f"Total samples: {all_stats['total']} "
          f"({all_stats['valid']} valid, {all_stats['invalid']} invalid)")


if __name__ == "__main__":
    main()
