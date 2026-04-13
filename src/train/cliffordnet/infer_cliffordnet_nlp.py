"""CliffordNet NLP Inference with Power Sampling.

Standalone inference script for generating text from a trained CliffordNetLM
checkpoint using standard nucleus sampling, MCMC power sampling, or
deterministic max-swap.

Power sampling (arXiv:2510.14901, arXiv:2601.21590) samples from *p^alpha*
instead of *p*, producing globally more coherent trajectories without any
additional training.

Usage::

    # Standard nucleus sampling
    python -m train.cliffordnet.infer_cliffordnet_nlp \\
        --checkpoint results/.../checkpoints/final.keras \\
        --prompt "The capital of France is" \\
        --method standard

    # MCMC power sampling (improved reasoning)
    python -m train.cliffordnet.infer_cliffordnet_nlp \\
        --checkpoint results/.../checkpoints/final.keras \\
        --prompt "In mathematics, a prime number is" \\
        --method power --temperature 0.25 --mcmc-steps 10

    # Max-swap (deterministic trajectory optimization)
    python -m train.cliffordnet.infer_cliffordnet_nlp \\
        --checkpoint results/.../checkpoints/final.keras \\
        --prompt "Albert Einstein was born in" \\
        --method max_swap

    # Compare all methods side-by-side
    python -m train.cliffordnet.infer_cliffordnet_nlp \\
        --checkpoint results/.../checkpoints/final.keras \\
        --prompt "The theory of relativity states that" \\
        --compare

    # Batch prompts from file
    python -m train.cliffordnet.infer_cliffordnet_nlp \\
        --checkpoint results/.../checkpoints/final.keras \\
        --prompts-file prompts.txt --method power
"""

import os
import json
import time
import argparse
from typing import List, Optional, Tuple

import keras
import numpy as np
import tiktoken

from train.common import setup_gpu
from dl_techniques.models.cliffordnet.lm import CliffordNetLM
from dl_techniques.models.cliffordnet.power_sampling import (
    PowerSampler,
    PowerSamplingConfig,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
)
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(path: str) -> CliffordNetLM:
    """Load a CliffordNetLM from a ``.keras`` checkpoint."""
    logger.info(f"Loading checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "CliffordNetLM": CliffordNetLM,
            "CausalCliffordNetBlock": CausalCliffordNetBlock,
            "MaskedCausalLMLoss": MaskedCausalLMLoss,
            "FocalCausalLMLoss": FocalCausalLMLoss,
        },
    )
    logger.info(f"Loaded model: {model.count_params():,} params")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_single_prompt(
    sampler: PowerSampler,
    prompt: str,
    method: str,
    **kwargs,
) -> Tuple[str, dict]:
    """Generate text for a single prompt."""
    text, info = sampler.generate_text(prompt, method=method, **kwargs)
    return text, info


def run_comparison(
    sampler: PowerSampler,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.25,
    mcmc_steps: int = 10,
    block_num: int = 8,
) -> dict:
    """Run all three methods on the same prompt for comparison."""
    results = {"prompt": prompt, "methods": {}}

    # Standard
    logger.info("Running standard nucleus sampling...")
    text, info = sampler.generate_text(
        prompt, method="standard",
        temperature=0.85, max_tokens=max_tokens,
    )
    results["methods"]["standard"] = {"text": text, **info}

    # Power sampling
    logger.info("Running MCMC power sampling...")
    text, info = sampler.generate_text(
        prompt, method="power",
        temperature=temperature, mcmc_steps=mcmc_steps,
        max_tokens=max_tokens, block_num=block_num,
    )
    results["methods"]["power"] = {"text": text, **info}

    # Max-swap
    logger.info("Running max-swap power sampling...")
    text, info = sampler.generate_text(
        prompt, method="max_swap",
        temperature=temperature, mcmc_steps=mcmc_steps,
        max_tokens=max_tokens, block_num=block_num,
    )
    results["methods"]["max_swap"] = {"text": text, **info}

    return results


def print_results(results: dict) -> None:
    """Pretty-print comparison results."""
    print(f"\n{'=' * 70}")
    print(f"Prompt: \"{results['prompt']}\"")
    print(f"{'=' * 70}")

    for method, data in results["methods"].items():
        print(f"\n--- {method.upper()} ---")
        text = data.pop("text", "")
        print(f"Output: {text[:500]}")
        print(f"Info: {data}")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CliffordNet NLP Inference with Power Sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .keras checkpoint",
    )

    # Prompt source (mutually exclusive)
    prompt_group = p.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt", type=str, default=None,
        help="Single text prompt",
    )
    prompt_group.add_argument(
        "--prompts-file", type=str, default=None,
        help="File with one prompt per line",
    )

    # Method
    p.add_argument(
        "--method", type=str, default="power",
        choices=["standard", "power", "max_swap"],
        help="Generation method",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Run all methods side-by-side",
    )

    # Sampling parameters
    p.add_argument("--temperature", type=float, default=0.25,
                    help="Temperature (alpha=1/temp for power sampling)")
    p.add_argument("--mcmc-steps", type=int, default=10,
                    help="MCMC refinement steps per block")
    p.add_argument("--max-tokens", type=int, default=100,
                    help="Maximum tokens to generate")
    p.add_argument("--block-num", type=int, default=8,
                    help="Number of generation blocks")
    p.add_argument("--top-p", type=float, default=0.92,
                    help="Nucleus sampling threshold")
    p.add_argument("--repetition-penalty", type=float, default=1.3,
                    help="Repetition penalty factor")

    # Hardware
    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Output
    p.add_argument("--output-json", type=str, default=None,
                    help="Save results to JSON file")

    return p


def main() -> None:
    """Main entry point for CliffordNet NLP inference."""
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    # Load model
    model = load_model(args.checkpoint)

    # Build tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Create sampler
    config = PowerSamplingConfig(
        temperature=args.temperature,
        mcmc_steps=args.mcmc_steps,
        block_num=args.block_num,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    sampler = PowerSampler(model, enc, config=config)

    # Get prompts
    prompts: List[str] = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts
        prompts = [
            "The United States of America is a",
            "In mathematics, a prime number is",
            "Albert Einstein was born in",
        ]

    # Run inference
    all_results = []

    for prompt in prompts:
        if args.compare:
            results = run_comparison(
                sampler, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                mcmc_steps=args.mcmc_steps,
                block_num=args.block_num,
            )
            print_results(results)
            all_results.append(results)
        else:
            kwargs = {}
            if args.method == "standard":
                kwargs = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "repetition_penalty": args.repetition_penalty,
                }
            else:
                kwargs = {
                    "temperature": args.temperature,
                    "mcmc_steps": args.mcmc_steps,
                    "max_tokens": args.max_tokens,
                    "block_num": args.block_num,
                }

            text, info = run_single_prompt(
                sampler, prompt, args.method, **kwargs,
            )

            print(f"\nPrompt: \"{prompt}\"")
            print(f"Method: {args.method}")
            print(f"Output: {text[:500]}")
            print(f"Info: {info}\n")

            all_results.append({
                "prompt": prompt,
                "method": args.method,
                "text": text[:500],
                **info,
            })

    # Save results
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
