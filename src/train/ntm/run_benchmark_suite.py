"""
Command-line entry point for the MANN benchmark suite.

Loads a saved ``.keras`` model and evaluates it with
:class:`~train.ntm.harness.BenchmarkHarness`, writing a JSON report. This is a
thin wrapper: every measurement lives in ``harness.py`` / ``metrics.py``, and
nothing here computes a metric of its own.

The suite contains benchmarks with different input signatures (copy task,
associative recall, SCAN, bAbI, ...), so a model built for one of them will
fail the others. ``BenchmarkHarness.run_full_suite`` contains each failure to
its own benchmark and keeps going; use ``--benchmarks`` to run only the ones a
given checkpoint is shaped for, and ``--quiet`` only when you do not need those
failures reported (it is the harness's ``verbose`` flag, which also gates the
per-benchmark error log).

Usage:
    python -m train.ntm.run_benchmark_suite --help
    python -m train.ntm.run_benchmark_suite --checkpoint results/ntm/final.keras
    python -m train.ntm.run_benchmark_suite --checkpoint m.keras \\
        --benchmarks copy_task length_generalization --gpu 1
"""

import argparse
import os
from datetime import datetime
from typing import List, Optional

import keras

from dl_techniques.utils.logger import logger
from train.common import setup_gpu

from .config import BenchmarkSuiteConfig
from .harness import BenchmarkHarness, SuiteReport

#: Reports are written under the repo-root ``results/`` tree, like every other
#: training artefact in this repo -- never under the source package.
DEFAULT_OUTPUT_DIR = "results/ntm_benchmarks"


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build and run the benchmark suite's command-line parser.

    :param argv: Argument list to parse. Uses ``sys.argv[1:]`` when None.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the MANN benchmark suite against a saved model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the saved .keras model to evaluate.')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory the JSON report is written into.')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=None,
                        choices=sorted(BenchmarkHarness.BENCHMARK_METHODS),
                        help='Subset of benchmarks to run. Runs all if omitted.')
    parser.add_argument('--model-name', type=str, default=None,
                        help="Name recorded in the report. Defaults to the "
                             "checkpoint's file name without its extension.")
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device index.')
    parser.add_argument('--quiet', action='store_true',
                        help='Silence the harness per-benchmark progress and '
                             'error logs (sets BenchmarkSuiteConfig.verbose).')
    return parser.parse_args(argv)


def run_suite(
    checkpoint: str,
    output_dir: str,
    benchmarks: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Load a checkpoint, run the benchmark suite and save the report.

    :param checkpoint: Path to a saved ``.keras`` model.
    :param output_dir: Directory the JSON report is written into. Created if
        it does not exist (by ``BenchmarkHarness.save_report``).
    :param benchmarks: Benchmark names to run, or None for all of them.
    :param model_name: Name recorded in the report. Derived from the
        checkpoint's file name when None.
    :param verbose: Passed to ``BenchmarkSuiteConfig.verbose``; gates the
        harness's per-benchmark progress and error logging.
    :return: Path of the written report.
    """
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(checkpoint))[0]

    logger.info(f"Loading model from {checkpoint}")
    model = keras.models.load_model(checkpoint)

    harness = BenchmarkHarness(BenchmarkSuiteConfig(verbose=verbose))
    report: SuiteReport = harness.run_full_suite(
        model, model_name=model_name, benchmarks=benchmarks
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        output_dir, f"benchmark_report_{model_name}_{timestamp}.json"
    )
    harness.save_report(report_path, report)

    # Logged unconditionally: the count is how the caller sees that a benchmark
    # died inside run_full_suite's own `except Exception` -- an empty or short
    # report is the observable symptom, and under --quiet it is the only one.
    logger.info(
        f"Recorded {len(report.runs)} benchmark run(s) in "
        f"{report.total_runtime:.2f}s -> {report_path}"
    )
    if not report.runs:
        logger.warning(
            "No benchmark produced a result. Re-run without --quiet to see "
            "why each one failed (input-signature mismatches are the usual "
            "cause: the suite's benchmarks do not share an input shape)."
        )
    return report_path


def main() -> None:
    """Parse arguments and run the suite."""
    args = parse_arguments()

    setup_gpu(args.gpu)

    run_suite(
        checkpoint=args.checkpoint,
        output_dir=args.output,
        benchmarks=args.benchmarks,
        model_name=args.model_name,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
