# Vision-Language Model Benchmarks

Reference targets for the dl_techniques vision-language experiments (instruction-tuned VLMs, video VLMs, CLIP-style dual encoders, agentic/GUI eval).
Numbers are from OpenCompass / HuggingFace Open VLM leaderboard snapshots, the MMMU and Video-MME official leaderboards, individual technical reports, and model release blogs; cite the date you pulled them. `VISION_BENCHMARKS.md` retains a small VLM summary table for cross-reference; this file is the canonical VLM reference.

> **Snapshot date**: 2026-05-13

## What VLM Benchmarks Measure

### General Multimodal Reasoning

- **MMMU** (Yue et al., 2023, arXiv:2311.16502). 11.5K college-level multimodal questions across 6 disciplines (Art, Business, Science, Health, Humanities, Tech). Accuracy on the val split is the headline. **MMMU-Pro** (2024) drops easy MC distractors and adds a "vision-only" track (question rendered into the image) to penalize text-shortcut models; scores drop 10-20 points vs vanilla MMMU.
- **MMStar** (Chen et al., 2024, arXiv:2403.20330). 1.5K curated vision-indispensable samples; removes the ~50% of MMMU/MMBench questions that text-only LLMs can answer without the image. Treat as the harder, contamination-resistant cousin of MMMU.
- **MMBench** (Liu et al., 2023, arXiv:2307.06281). 3K+ MC questions with CircularEval rotation across 20 ability dimensions. Reported as MMBench-EN-dev / MMBench-EN-test and MMBench-CN-dev/test.
- **MM-Vet** (Yu et al., 2023, arXiv:2308.02490). 200 open-ended questions graded by GPT-4 across 6 capabilities (recognition, OCR, knowledge, spatial, math, generation). Free-form, not MC.
- **MathVista** (Lu et al., 2023, arXiv:2310.02255). 6.1K visual math problems (testmini = 1K subset is the leaderboard track). **MathVision** (2024) is harder, 3K competition-style problems.
- **BLINK** (Fu et al., 2024, arXiv:2404.12390). 14 "core visual perception" tasks (depth, jigsaw, art style, IQ test) that humans solve in a blink but VLMs find very hard.
- **RealWorldQA** (xAI, 2024). 765 photos taken from cars with real-world spatial reasoning questions.
- **MMVP** (Tong et al., 2024). Adversarial pairs where two near-identical CLIP-style images have different ground truth; exposes the "CLIP-blind" failure mode of LLaVA-family encoders.
- **SEED-Bench / SEED-Bench-2** (Li et al., 2023). 19K MC questions across 12 (SEED) / 27 (SEED-2) dimensions including video and image generation.

### Document / Chart / OCR

- **DocVQA** (Mathew et al., 2021, arXiv:2007.00398). 50K QA on industry document images. Reported as ANLS on test.
- **ChartQA** (Masry et al., 2022, arXiv:2203.10244). 9.6K QA pairs over real-world charts. Accuracy with relaxed numeric match.
- **InfographicVQA** (Mathew et al., 2022). 30K QA on infographic-style images; harder layout than DocVQA.
- **OCRBench** (Liu et al., 2024, arXiv:2305.07895). Aggregate of 5 OCR tasks (text recognition, scene-text VQA, doc VQA, key-info extraction, handwritten math). Reports a score out of 1000. **OCRBench v2** (Liu et al., 2025, arXiv:2501.00321) replaces it with 10K samples across 31 sub-tasks including visual text localization.
- **TextVQA** (Singh et al., 2019). Scene-text VQA; accuracy.
- **AI2D** (Kembhavi et al., 2016). 5K science diagrams with MC.
- **ScienceQA** (Lu et al., 2022). 21K MC questions; subset has images. Image subset accuracy is the headline.

### Visual Grounding / Referring

- **RefCOCO / RefCOCO+ / RefCOCOg** (Yu et al., 2016). Phrase-to-box grounding; report precision@0.5 IoU on val / testA / testB.
- **Visual7W**, **Flickr30K Entities**. Older grounding benchmarks; mostly used by Qwen2-VL / InternVL grounding heads.

### Hallucination

- **POPE** (Li et al., 2023, arXiv:2305.10355). Yes/No object-existence probe; F1 on adversarial / popular / random splits.
- **HallusionBench** (Guan et al., 2023, arXiv:2310.14566). 1.1K image-question pairs targeting visual illusions, geographic / chart / OCR hallucination. Reports aAcc (all-accuracy) and qAcc (question-level).
- **MMHal-Bench** (Sun et al., 2023). 96 open-ended questions graded by GPT-4 for hallucination rate.
- **AMBER** (Wang et al., 2023). 1K image-text pairs with both generative and discriminative hallucination tracks.

### Video VLM

- **Video-MME** (Fu et al., 2024, arXiv:2405.21075). 900 manually annotated videos in 3 duration tiers (short <2min, medium 4-15min, long 30-60min). MC with / without subtitles; "overall" is the headline.
- **MVBench** (Li et al., 2023, arXiv:2311.17005). 20 temporal-reasoning tasks built from existing video datasets.
- **EgoSchema** (Mangalam et al., 2023, arXiv:2308.09126). 5K very-long-form (180s) egocentric MC; full set accuracy.
- **LongVideoBench** (Wu et al., 2024, arXiv:2407.15754). 3.7K MC over videos up to 1 hour with referred reasoning.
- **TempCompass**, **Perception Test** (Patraucean et al., 2023), **ActivityNet-QA**. Standard temporal-reasoning and action-QA benchmarks.

### Agentic / GUI / Embodied

- **ScreenSpot** (Cheng et al., 2024, arXiv:2401.10935). 1.3K GUI elements across web / mobile / desktop; reports click accuracy. **ScreenSpot-Pro** (2024) is 1.6K expert-annotated screenshots from 23 professional apps (MATLAB, Photoshop, VSCode); the harder, current discriminator at the frontier.
- **WebArena (multimodal)** and **VisualWebArena**. Realistic browser tasks; reports success rate.
- **AndroidWorld**, **OSWorld**. Live OS / mobile agent benchmarks; success rate on multi-step tasks. OSWorld human baseline is 72.4%; first model crossing it was GPT-5.4 in March 2026 at 75.0%.

### Zero-Shot Classification / Retrieval (CLIP-style)

- **ImageNet-1K zero-shot top-1**. The canonical retrieval-via-text-prompts metric for dual-encoder CLIP-style models.
- **COCO / Flickr30K retrieval R@1** (text-to-image, image-to-text). Standard cross-modal retrieval.

### Open-Ended / Arena

- **LMSYS Vision Arena ELO** (lmarena.ai). Head-to-head human preference; the only metric that captures "chat feel". Sample sizes <5K hits at frontier are noisy.
- **WildVision** (Lu et al., 2024). 8K real user vision queries from WildVision-Chat; GPT-4 graded.
- **MEGA-Bench** (Chen et al., 2024). 500 real-world multimodal tasks with task-specific metrics rather than aggregated accuracy.

## How to Read These Numbers

- **MMMU validation scores are reported with chain-of-thought on/off.** Many model cards quote CoT scores, OpenCompass quotes 0-shot direct answer. The 2-3 point gap between settings is roughly the same magnitude as the architectural gap between adjacent open models; always check the eval protocol.
- **VLMEvalKit vs lmms-eval divergence.** The two community harnesses produce 1-3 point differences on the same model for the same benchmark because of prompt template, image resize, and grading parser differences. OpenCompass and the HuggingFace Open VLM Leaderboard use VLMEvalKit; some model cards quote lmms-eval. Note which harness produced the number.
- **Contamination.** Heavy overlap exists between LLaVA-style instruction tuning data (LLaVA-Instruct-150K, ShareGPT4V, Sphinx mix, Cambrian) and "academic" benchmarks - especially ScienceQA, AI2D, and OCR-style tasks. Treat ranking-by-single-benchmark with skepticism; the 3-5 point gaps inside the open-weights top tier are almost certainly inside the contamination noise floor.
- **Image resolution matters more than architecture above 30B.** Qwen2-VL / Qwen2.5-VL and InternVL2.5 use dynamic resolution / AnyRes tiling that ingests 4M+ pixels for long documents; LLaVA-1.5 / Idefics2 use fixed 336 or 384. A "high-res" variant of the same model adds 5-10 points on DocVQA / ChartQA / OCRBench without any other change.
- **Pass@1 vs majority-vote.** Most VLM benchmarks are single-shot. Do not confuse with maj@16 / pass@k reported for reasoning-focused models (o1-style multimodal reasoning, Gemini 2.5 Pro thinking).
- **Arena ELO vs single-benchmark.** Vision Arena ELO is preference-based and correlates best with downstream user feel; single benchmarks like MMMU correlate with knowledge depth, not chat quality. A model 5 ELO ahead on Arena but 3 MMMU points behind another is the better default chat VLM.
- **"MoE total" vs "MoE active" parameter counts.** DeepSeek-VL2 quotes activated parameters (1.0B / 2.8B / 4.5B); total params are larger (3B / 16B / 27B). Compare on activated parameters for compute parity but on total parameters for memory footprint.

## Benchmark Tables

Scores below are MMMU val (0-shot, direct answer where reported by the leaderboard), MMStar overall, MMBench-EN-test, MathVista testmini, DocVQA test ANLS, ChartQA relaxed accuracy, OCRBench score / 1000, HallusionBench aAcc. "?" indicates no public number; do not fabricate.

### Tiny VLMs (<= 4B total params)

| Model | LLM backbone | Vision encoder | Total params | MMMU (val) | MMStar | MMBench-en (test) | MathVista | DocVQA (test) | ChartQA | OCRBench | HallusionBench | Year | License |
|-------|--------------|----------------|--------------|------------|--------|--------------------|-----------|---------------|---------|----------|-----------------|------|---------|
| Moondream2 | Phi-1.5-1.3B | SigLIP-SO | 1.9B | 32.4 | ? | 64.5 | ? | ? | ? | 612 | ? | 2024 | Apache-2.0 |
| MobileVLM V2-3B | MobileLLaMA-2.7B | CLIP-L | 3.0B | ? | ? | 63.2 | ? | ? | ? | ? | ? | 2024 | Apache-2.0 |
| SmolVLM-256M | SmolLM2-135M | SigLIP-B/16 | 0.26B | 33.7 | ? | ? | ? | 70.5 | ? | ? | ? | 2024 | Apache-2.0 |
| SmolVLM-500M | SmolLM2-360M | SigLIP-B/16 | 0.5B | 34.9 | ? | ? | ? | 77.4 | ? | ? | ? | 2024 | Apache-2.0 |
| SmolVLM-2.2B | SmolLM2-1.7B | SigLIP-SO | 2.2B | 38.8 | ? | ? | ? | 81.6 | ? | ? | ? | 2024 | Apache-2.0 |
| Idefics3-2B | SmolLM-1.7B | SigLIP-SO | 2.2B | 43.4 | ? | 76.4 | ? | 84.6 | 70.2 | 657 | ? | 2024 | Apache-2.0 |
| LLaVA-OV-0.5B | Qwen2-0.5B | SigLIP-SO | 1.0B | 31.4 | 37.5 | 52.1 | 34.8 | 73.7 | 61.4 | 565 | 27.9 | 2024 | Apache-2.0 |
| InternVL2-1B | Qwen2-0.5B | InternViT-300M | 0.94B | 36.7 | 45.7 | 65.4 | 37.7 | 81.7 | 72.9 | 754 | 34.0 | 2024 | MIT |
| InternVL2-2B | InternLM2-1.8B | InternViT-300M | 2.2B | 36.3 | 50.1 | 73.2 | 46.3 | 86.9 | 76.2 | 784 | 38.0 | 2024 | MIT |
| InternVL2.5-1B | Qwen2.5-0.5B | InternViT-300M | 0.94B | 40.9 | 50.1 | 70.7 | 43.2 | 84.8 | 75.9 | 785 | 39.0 | 2024 | MIT |
| InternVL2.5-2B | InternLM2.5-1.8B | InternViT-300M | 2.2B | 43.6 | 53.7 | 74.7 | 51.3 | 88.7 | 79.2 | 804 | 42.6 | 2024 | MIT |
| InternVL2.5-4B | Qwen2.5-3B | InternViT-300M | 3.7B | 52.3 | 58.3 | 81.1 | 60.5 | 91.6 | 84.0 | 828 | 46.3 | 2024 | MIT |
| Qwen2-VL-2B | Qwen2-1.5B | NaViT-675M | 2.2B | 41.1 | 48.0 | 74.9 | 43.0 | 90.1 | 73.5 | 809 | 41.7 | 2024 | Apache-2.0 |
| Qwen2.5-VL-3B | Qwen2.5-3B | NaViT-675M | 3.8B | 53.1 | 55.9 | 79.1 | 62.3 | 93.9 | 84.0 | 797 | 46.3 | 2025 | Qwen-RL |
| PaliGemma-3B-mix-448 | Gemma-2B | SigLIP-SO | 3.0B | 34.9 | ? | 65.6 | 28.7 | 82.0 | 33.7 | 614 | 32.2 | 2024 | Gemma |
| PaliGemma 2-3B-mix-448 | Gemma2-2B | SigLIP-SO | 3.0B | 41.1 | ? | 71.6 | 33.0 | 87.4 | 41.5 | 716 | 35.8 | 2024 | Gemma |
| Phi-3.5-vision | Phi-3.5-mini-3.8B | CLIP-L | 4.2B | 43.0 | 49.0 | 81.9 | 43.9 | 75.9 | 81.8 | 599 | 40.5 | 2024 | MIT |
| Phi-4-multimodal | Phi-4-mini-3.8B | SigLIP-SO | 5.6B | 55.1 | 54.8 | 86.7 | 62.4 | 93.2 | 81.4 | 813 | 49.0 | 2025 | MIT |
| DeepSeek-VL2-Tiny | DeepSeekMoE-3B (1.0B act) | SigLIP-SO | 3.4B total | 40.7 | ? | 74.6 | 53.6 | 88.9 | 81.0 | 809 | 39.6 | 2024 | DeepSeek |
| DeepSeek-VL2-Small | DeepSeekMoE-16B (2.8B act) | SigLIP-SO | 16.1B total | 51.1 | ? | 79.6 | 62.8 | 92.3 | 84.5 | 834 | 45.3 | 2024 | DeepSeek |
| MiniCPM-V 2.6 (8B, mobile-class) | Qwen2-7B | SigLIP-SO | 8.0B | 49.8 | 57.5 | 81.5 | 60.6 | 90.8 | ? | 852 | 48.1 | 2024 | Apache-2.0 |

### Small VLMs (4B - 10B)

| Model | LLM backbone | Vision encoder | Total params | MMMU (val) | MMStar | MMBench-en (test) | MathVista | DocVQA (test) | ChartQA | OCRBench | HallusionBench | Year | License |
|-------|--------------|----------------|--------------|------------|--------|--------------------|-----------|---------------|---------|----------|-----------------|------|---------|
| LLaVA-1.5-7B | Vicuna-7B | CLIP-L/14-336 | 7.2B | 35.7 | 33.1 | 64.3 | 25.6 | 21.5 | 17.8 | 318 | 27.6 | 2023 | LLaMA-2 |
| LLaVA-1.5-13B | Vicuna-13B | CLIP-L/14-336 | 13.4B | 36.4 | 34.3 | 67.7 | 27.6 | ? | 18.2 | 337 | 30.2 | 2023 | LLaMA-2 |
| LLaVA-1.6 (NeXT) 7B | Mistral-7B | CLIP-L/14-336 (AnyRes) | 7.6B | 35.8 | 38.7 | 67.4 | 37.6 | 74.4 | 54.8 | 532 | 27.6 | 2024 | Apache-2.0 |
| LLaVA-OneVision-7B | Qwen2-7B | SigLIP-SO | 8.0B | 48.8 | 61.7 | 80.8 | 63.2 | 87.5 | 80.0 | 622 | 47.5 | 2024 | Apache-2.0 |
| InternVL2-4B | Phi-3-3.8B | InternViT-300M | 4.2B | 47.9 | 53.9 | 78.6 | 58.6 | 89.2 | 81.5 | 788 | 41.4 | 2024 | MIT |
| InternVL2-8B | InternLM2.5-7B | InternViT-300M | 8.1B | 51.2 | 61.5 | 81.7 | 58.3 | 91.6 | 83.3 | 794 | 45.0 | 2024 | MIT |
| InternVL2.5-8B | InternLM2.5-7B | InternViT-300M | 8.1B | 56.0 | 62.8 | 84.6 | 64.4 | 93.0 | 84.8 | 822 | 49.0 | 2024 | MIT |
| Qwen2-VL-7B | Qwen2-7B | NaViT-675M | 8.3B | 54.1 | 60.7 | 83.0 | 58.2 | 94.5 | 83.0 | 866 | 50.6 | 2024 | Apache-2.0 |
| Qwen2.5-VL-7B | Qwen2.5-7B | NaViT-675M | 8.3B | 58.6 | 63.9 | 83.5 | 68.2 | 95.7 | 87.3 | 864 | 52.9 | 2025 | Apache-2.0 |
| MiniCPM-V 2.6 | Qwen2-7B | SigLIP-SO | 8.0B | 49.8 | 57.5 | 81.5 | 60.6 | 90.8 | ? | 852 | 48.1 | 2024 | Apache-2.0 |
| Molmo-7B-D | Qwen2-7B | OpenAI CLIP-L | 8.0B | 45.3 | 54.4 | 73.6 | 51.6 | 92.2 | 84.1 | 694 | 46.1 | 2024 | Apache-2.0 |
| Molmo-7B-O | OLMo-7B | OpenAI CLIP-L | 7.6B | 39.3 | 49.8 | 71.6 | 44.8 | 90.8 | 80.4 | 656 | 42.0 | 2024 | Apache-2.0 |
| Idefics3-8B | Llama-3-8B | SigLIP-SO | 8.5B | 46.6 | 55.9 | 76.4 | 58.4 | 87.7 | 74.8 | 643 | ? | 2024 | Apache-2.0 |
| DeepSeek-VL2 | DeepSeekMoE-27B (4.5B act) | SigLIP-SO | 27.5B total | 54.0 | 61.3 | 81.2 | 62.8 | 93.3 | 86.0 | 811 | 45.3 | 2024 | DeepSeek |
| Llama-3.2-11B-Vision-Instruct | Llama-3-8B + adapter | ViT-H/14 | 10.7B | 50.7 | 49.8 | 65.8 | 51.5 | 88.4 | 83.4 | 753 | 40.3 | 2024 | Llama-3 |

### Medium VLMs (10B - 40B)

| Model | LLM backbone | Vision encoder | Total params | MMMU (val) | MMStar | MMBench-en (test) | MathVista | DocVQA (test) | ChartQA | OCRBench | HallusionBench | Year | License |
|-------|--------------|----------------|--------------|------------|--------|--------------------|-----------|---------------|---------|----------|-----------------|------|---------|
| Pixtral-12B | Mistral-Nemo-12B | Pixtral-ViT-400M | 12.7B | 52.5 | 54.5 | 76.1 | 58.0 | 89.6 | 81.8 | 685 | 46.5 | 2024 | Apache-2.0 |
| LLaVA-NeXT-34B | Yi-34B | CLIP-L/14-336 | 34.7B | 51.1 | 50.0 | 79.3 | 46.5 | 84.0 | 68.7 | 574 | 34.8 | 2024 | Apache-2.0 |
| InternVL2-26B | InternLM2-20B | InternViT-6B | 25.5B | 51.2 | 61.2 | 83.4 | 59.4 | 92.9 | 84.9 | 825 | 50.7 | 2024 | MIT |
| InternVL2.5-26B | InternLM2.5-20B | InternViT-6B | 25.5B | 60.0 | 66.5 | 85.4 | 67.7 | 94.0 | 87.2 | 852 | 55.0 | 2024 | MIT |
| InternVL2-40B | Nous-Hermes-2-Yi-34B | InternViT-6B | 40.1B | 55.2 | 65.4 | 86.8 | 63.7 | 93.9 | 86.2 | 837 | 56.9 | 2024 | MIT |
| InternVL2.5-38B | Qwen2.5-32B | InternViT-6B | 38.4B | 63.9 | 67.9 | 86.5 | 71.9 | 95.3 | 88.2 | 842 | 56.8 | 2024 | MIT |
| Qwen2.5-VL-32B | Qwen2.5-32B | NaViT-675M | 33.5B | 70.0 | 69.5 | 87.4 | 74.7 | 94.8 | 87.3 | 856 | 56.4 | 2025 | Apache-2.0 |
| Aria (MoE) | Aria-25B (3.5B act) | SigLIP-SO | 25.3B total | 54.9 | 60.9 | 80.0 | 66.1 | 92.6 | 86.4 | 783 | 49.2 | 2024 | Apache-2.0 |

### Large / XL VLMs (>= 40B)

| Model | LLM backbone | Vision encoder | Total params | MMMU (val) | MMStar | MMBench-en (test) | MathVista | DocVQA (test) | ChartQA | OCRBench | HallusionBench | Year | License |
|-------|--------------|----------------|--------------|------------|--------|--------------------|-----------|---------------|---------|----------|-----------------|------|---------|
| LLaVA-OneVision-72B | Qwen2-72B | SigLIP-SO | 73.2B | 56.8 | 65.8 | 85.8 | 67.5 | 91.3 | 83.7 | 741 | 49.0 | 2024 | Apache-2.0 |
| InternVL2-Llama3-76B | Llama-3-70B | InternViT-6B | 76.3B | 58.2 | 67.4 | 86.5 | 65.5 | 94.1 | 88.4 | 839 | 55.2 | 2024 | MIT |
| InternVL2.5-78B | Qwen2.5-72B | InternViT-6B | 78.4B | 70.1 | 69.5 | 88.3 | 72.3 | 95.1 | 88.3 | 854 | 57.4 | 2024 | MIT |
| Qwen2-VL-72B | Qwen2-72B | NaViT-675M | 73.4B | 64.5 | 68.3 | 86.5 | 70.5 | 96.5 | 88.3 | 877 | 58.1 | 2024 | Qwen |
| Qwen2.5-VL-72B | Qwen2.5-72B | NaViT-675M | 73.4B | 70.2 | 70.8 | 88.6 | 74.8 | 96.4 | 89.5 | 885 | 55.2 | 2025 | Qwen |
| Llama-3.2-90B-Vision-Instruct | Llama-3-70B + adapter | ViT-H/14 | 88.6B | 60.3 | 56.0 | 73.3 | 57.3 | 90.1 | 85.5 | 783 | 44.1 | 2024 | Llama-3 |
| Molmo-72B | Qwen2-72B | OpenAI CLIP-L | 73.0B | 54.1 | 63.3 | 81.2 | 58.6 | 93.5 | 87.3 | 736 | 51.5 | 2024 | Apache-2.0 |
| NVLM-D-72B | Qwen2-72B | InternViT-6B | 79.4B | 58.7 | ? | 86.6 | 65.2 | 92.6 | 86.0 | 853 | ? | 2024 | CC-BY-NC-4.0 |
| NVLM-X-72B (MoE-style cross-attn) | Qwen2-72B | InternViT-6B | 79.4B | 60.2 | ? | 87.1 | 66.6 | ? | ? | ? | ? | 2024 | CC-BY-NC-4.0 |
| Pixtral-Large-124B | Mistral-Large-2-123B | Pixtral-ViT-1B | 124B | 64.0 | ? | ? | 69.4 | 93.3 | 88.1 | 741 | ? | 2024 | MRL |

### Proprietary / API

OpenCompass / MMMU-leaderboard snapshot 2026-05-13. Many proprietary 2026 model cards do not publish per-benchmark numbers for every dataset; "?" reflects what is not published.

| Model | MMMU (val) | MMStar | MMBench-en (test) | MathVista | DocVQA | ChartQA | OCRBench | HallusionBench | Year | License |
|-------|------------|--------|--------------------|-----------|--------|---------|----------|-----------------|------|---------|
| GPT-4o (2024-08) | 69.1 | 64.7 | 83.4 | 63.8 | 92.8 | 85.7 | 736 | 55.0 | 2024 | Proprietary |
| GPT-4o-mini | 60.0 | 54.8 | 76.0 | 56.7 | ? | ? | 785 | 46.1 | 2024 | Proprietary |
| GPT-4.1 | 74.8 | ? | ? | 72.2 | ? | ? | ? | ? | 2025 | Proprietary |
| GPT-5 (release) | 84.2 | ? | ? | ? | ? | ? | ? | ? | 2025 | Proprietary |
| o1 (vision) | 78.2 | ? | ? | 73.9 | ? | ? | ? | ? | 2024 | Proprietary |
| o3 (vision) | 82.9 | ? | ? | 86.8 | ? | ? | ? | ? | 2025 | Proprietary |
| o4-mini (vision) | 81.6 | ? | ? | 84.3 | ? | ? | ? | ? | 2025 | Proprietary |
| Claude 3.5 Sonnet (20241022) | 70.4 | 64.2 | 82.6 | 67.7 | 95.2 | 90.8 | 788 | 55.5 | 2024 | Proprietary |
| Claude 3.7 Sonnet | 75.0 | ? | ? | 73.1 | 96.0 | 90.8 | ? | ? | 2025 | Proprietary |
| Claude 4 Sonnet | 85.4 | ? | ? | ? | ? | ? | ? | ? | 2025 | Proprietary |
| Claude 4 Opus | 86.7 | ? | ? | ? | ? | ? | ? | ? | 2025 | Proprietary |
| Claude Sonnet 4.5 | 87.1 | ? | ? | ? | ? | ? | ? | ? | 2025 | Proprietary |
| Claude Sonnet 4.6 | 88.0 | ? | ? | ? | ? | ? | ? | ? | 2026 | Proprietary |
| Claude Opus 4.6 | 88.4 | ? | ? | ? | ? | ? | ? | ? | 2026 | Proprietary |
| Claude Opus 4.7 | 88.9 | ? | ? | ? | ? | ? | ? | ? | 2026 | Proprietary |
| Gemini 1.5 Pro | 62.2 | 59.1 | 73.9 | 63.9 | 93.1 | 87.2 | 754 | 45.6 | 2024 | Proprietary |
| Gemini 1.5 Flash | 56.1 | 51.5 | 68.7 | 58.4 | 89.9 | 78.3 | 727 | 41.9 | 2024 | Proprietary |
| Gemini 2.0 Pro | 72.7 | ? | ? | 71.3 | ? | ? | ? | ? | 2025 | Proprietary |
| Gemini 2.0 Flash | 70.4 | ? | ? | 70.4 | ? | ? | ? | ? | 2025 | Proprietary |
| Gemini 2.5 Pro | 84.0 | ? | ? | 80.9 | ? | ? | ? | ? | 2025 | Proprietary |
| Gemini 2.5 Flash | 76.7 | ? | ? | 74.8 | ? | ? | ? | ? | 2025 | Proprietary |
| Gemini 3.1 Pro | 86.1 | ? | ? | ? | ? | ? | ? | ? | 2026 | Proprietary |
| Grok-2-Vision | 66.1 | ? | ? | 69.0 | 93.6 | ? | ? | ? | 2024 | Proprietary |
| Grok-3-Vision | 78.0 | ? | ? | ? | ? | ? | ? | ? | 2025 | Proprietary |
| Grok-4-Vision | 83.4 | ? | ? | ? | ? | ? | ? | ? | 2025 | Proprietary |
| Reka Core | 56.3 | 58.4 | 80.6 | ? | ? | ? | 660 | ? | 2024 | Proprietary |
| Reka Flash | 53.3 | ? | ? | ? | ? | ? | ? | ? | 2024 | Proprietary |
| Yi-Vision (01.ai API) | 56.5 | ? | 82.2 | ? | ? | ? | ? | ? | 2024 | Proprietary |

### Video VLM Table

Video-MME "overall" is the unweighted mean across short/medium/long without subtitles (the public leaderboard headline). MVBench is 20-task average. EgoSchema is full-set accuracy. LongVideoBench val accuracy. Perception Test val accuracy.

| Model | Total params | Video-MME (overall) | MVBench | EgoSchema | LongVideoBench | Perception Test | Year | License |
|-------|--------------|---------------------|---------|-----------|----------------|-----------------|------|---------|
| LLaMA-VID-7B | 7B | ? | 41.4 | 38.5 | ? | 44.6 | 2023 | Apache-2.0 |
| VideoLLaMA 2-7B | 7B | 47.9 | 54.6 | 51.7 | ? | 51.4 | 2024 | Apache-2.0 |
| VideoLLaMA 3-7B | 7B | 66.2 | 69.7 | 63.3 | 59.8 | 71.8 | 2025 | Apache-2.0 |
| LLaVA-Video-7B | 7B | 63.3 | 58.6 | 57.3 | 58.2 | 67.9 | 2024 | Apache-2.0 |
| LLaVA-Video-72B | 73B | 70.6 | 64.1 | 65.6 | 61.9 | 74.3 | 2024 | Apache-2.0 |
| MiniCPM-V 2.6 | 8B | 60.9 | ? | ? | ? | ? | 2024 | Apache-2.0 |
| Apollo-7B | 7B | 61.3 | 62.7 | ? | 58.5 | 66.3 | 2024 | Apache-2.0 |
| LongVU-7B | 7B | 60.6 | 66.9 | 67.6 | ? | ? | 2024 | Apache-2.0 |
| NVILA-8B | 8B | 64.2 | 68.1 | 57.8 | 57.7 | 55.5 | 2024 | CC-BY-NC-4.0 |
| Aria (MoE, 3.5B act) | 25B total | 67.6 | ? | ? | 64.2 | ? | 2024 | Apache-2.0 |
| InternVL2-8B | 8B | 54.0 | 65.8 | ? | ? | ? | 2024 | MIT |
| InternVL2.5-8B | 8B | 64.2 | 72.0 | ? | 60.0 | ? | 2024 | MIT |
| InternVL2.5-78B | 78B | 72.1 | 76.4 | ? | 63.6 | ? | 2024 | MIT |
| Qwen2-VL-7B | 8B | 63.3 | 67.0 | 66.7 | 56.8 | 62.3 | 2024 | Apache-2.0 |
| Qwen2-VL-72B | 73B | 71.2 | 73.6 | 77.9 | ? | 68.0 | 2024 | Qwen |
| Qwen2.5-VL-7B | 8B | 65.1 | 69.6 | 65.0 | 56.0 | 70.5 | 2025 | Apache-2.0 |
| Qwen2.5-VL-72B | 73B | 73.3 | 70.4 | 76.2 | 60.7 | 73.2 | 2025 | Qwen |
| GPT-4o | n/a | 71.9 | 64.6 | 72.2 | 66.7 | ? | 2024 | Proprietary |
| Gemini 1.5 Pro | n/a | 75.0 | ? | 72.2 | 64.0 | ? | 2024 | Proprietary |
| Gemini 2.5 Pro | n/a | 84.8 | ? | ? | ? | ? | 2025 | Proprietary |

### CLIP-Style Dual Encoders

Zero-shot ImageNet-1K top-1 and COCO/Flickr R@1 retrieval. COCO R@1 (T->I) is text-to-image; Flickr30K R@1 is the higher direction (text-to-image where reported).

| Model | Params | Image size | IN-1K zero-shot | COCO R@1 (T->I) | Flickr30K R@1 | Year | License |
|-------|--------|------------|-----------------|------------------|----------------|------|---------|
| OpenAI CLIP-ViT-L/14 | 304M | 224 | 75.5 | 36.5 | 65.9 | 2021 | MIT |
| OpenAI CLIP-ViT-L/14-336 | 304M | 336 | 76.6 | 37.0 | 67.4 | 2022 | MIT |
| OpenCLIP-G/14 (LAION-2B) | 1.8B | 224 | 80.1 | 47.5 | 79.5 | 2022 | MIT |
| EVA-02-CLIP-L/14+ | 430M | 224 | 80.4 | 47.9 | 80.0 | 2023 | MIT |
| EVA-02-CLIP-E/14+ | 5.0B | 224 | 82.0 | 51.1 | 82.7 | 2023 | MIT |
| SigLIP-SO400M/14 | 400M | 384 | 83.2 | 47.4 | 79.5 | 2023 | Apache-2.0 |
| SigLIP 2 B/16 | 86M | 256 | 79.1 | 49.6 | 78.0 | 2025 | Apache-2.0 |
| SigLIP 2 L/16 | 304M | 256 | 82.5 | 52.2 | 81.4 | 2025 | Apache-2.0 |
| SigLIP 2 SO400M/14 | 400M | 384 | 84.1 | 53.2 | 83.4 | 2025 | Apache-2.0 |
| SigLIP 2 g-opt/16 | 1.1B | 256 | 85.6 | 54.6 | 84.6 | 2025 | Apache-2.0 |
| Meta CLIP 2 ViT-B/32 | 88M | 224 | 71.7 | ? | ? | 2025 | MIT |
| Meta CLIP 2 ViT-L/14 | 304M | 224 | 81.0 | ? | ? | 2025 | MIT |
| Meta CLIP 2 ViT-H/14 | 1.0B | 224 | 82.1 | 48.8 | 81.0 | 2025 | MIT |
| DFN5B-CLIP-ViT-H/14 | 1.0B | 378 | 84.4 | 50.1 | 84.7 | 2024 | Apple AML |
| InternVL-CLIP (InternViT-6B) | 5.9B | 224 | 83.2 | 49.0 | ? | 2023 | MIT |
| JinaCLIP v2 | 0.9B | 512 | 81.1 | 53.2 | 98.0 (I->T) | 2024 | CC-BY-NC-4.0 |

## 2026 SOTA Themes Captured in the Numbers

- **Native multimodal pretraining has overtaken adapter-style VLMs at every scale above 30B.** GPT-4o, Gemini 2.x / 2.5, Pixtral (Mistral), and Qwen2.5-VL all use native interleaved image-text pretraining rather than a frozen CLIP + projector + frozen LLM stack. LLaVA-family adapter recipes still produce competitive 7-13B open models (LLaVA-OneVision-7B at MMMU 48.8 is within 5 points of contemporaneous natively-pretrained Qwen2-VL-7B at 54.1) but lose decisively at the top of the table.
- **Dynamic resolution / AnyRes tiling is now standard.** Qwen2-VL / Qwen2.5-VL (NaViT-style), InternVL2.5 (AnyRes), Pixtral (variable patch grid), and DeepSeek-VL2 (dynamic tiling) all feed millions of pixels into long-document and chart tasks. The fixed 336-pixel LLaVA-1.5 / Idefics2 design hits a hard ceiling around DocVQA 75 / ChartQA 55, while AnyRes-style models clear 90 / 85.
- **Open weights are within 3 points of frontier proprietary on MMMU.** Snapshot 2026-05-13: InternVL2.5-78B 70.1, Qwen2.5-VL-72B 70.2, vs Claude 3.5 Sonnet 70.4, GPT-4o 69.1. The gap widens dramatically once Claude 4 / GPT-5 / Gemini 2.5 Pro enter (84-88 MMMU); however, those models are explicit reasoning models running with substantial test-time compute, and pass@1 single-shot numbers from open 70B-class models are no longer the bottleneck for most production use cases.
- **Video VLMs converging on Video-MME.** The benchmark has displaced K-400 / SSv2 as the canonical video benchmark for VLM eval. Open 70B-class models (Qwen2.5-VL-72B 73.3, InternVL2.5-78B 72.1) sit roughly at GPT-4o (71.9) and ~10 points below Gemini 2.5 Pro (84.8). Long-context video (>=1 hour) is still solidly proprietary territory because of native long-context handling.
- **Agentic / GUI eval is the discriminator at the frontier.** ScreenSpot-Pro and OSWorld separate Claude 4 / GPT-5 / o3 from everything else more cleanly than MMMU does. Anthropic's Claude Sonnet 4.6 at OSWorld 72.5% and Opus 4.6 at the top of the leaderboard sit 20+ points ahead of the best open VLM. GPT-5.4 crossed the OSWorld human baseline (72.4%) at 75.0% in March 2026.
- **CLIP efficiency has flipped.** SigLIP 2 SO400M (400M params, 84.1% IN-1K zero-shot) beats OpenAI CLIP-ViT-L/14 (304M, 75.5%) by 8.6 points and OpenCLIP-G/14 (1.8B, 80.1%) by 4 points at less than a quarter of the parameters. Meta CLIP 2 and DFN5B push the absolute SOTA for CLIP-style retrieval; SigLIP 2 SO400M is the new default backbone choice for downstream VLM construction (used by Phi-4-multimodal, SmolVLM, Aria).
- **Contamination is suspected for every open 70B+ VLM at MMMU >= 70.** Mixes used to fine-tune InternVL2.5 / Qwen2.5-VL touch ScienceQA, AI2D, and chart-style test distributions. 2-3 point gaps inside the open top tier should be treated as inside the noise.
- **MoE for VLMs is real.** DeepSeek-VL2-Small (2.8B activated / 16B total) reaches MMMU 51.1 / OCRBench 834, matching dense 7-8B models in throughput while dominating them on OCR. Aria (3.5B / 25B) similarly. Expect MoE VLMs to be the dominant open architecture by late 2026.

## Sources

- [OpenCompass VLM Leaderboard](https://rank.opencompass.org.cn/leaderboard-multimodal) - pulled 2026-05-13
- [HuggingFace Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) - pulled 2026-05-13
- [LMSYS Vision Arena (lmarena.ai)](https://lmarena.ai/) - pulled 2026-05-13
- [MMMU Benchmark Leaderboard](https://mmmu-benchmark.github.io/) - pulled 2026-05-13
- [MMBench OpenCompass Leaderboard](https://mmbench.opencompass.org.cn/leaderboard) - pulled 2026-05-13
- [Video-MME Leaderboard](https://video-mme.github.io/) - pulled 2026-05-13
- [OCRBench v2 Leaderboard (HuggingFace)](https://huggingface.co/spaces/ling99/OCRBench-v2-leaderboard) - pulled 2026-05-13
- [LMCouncil AI Model Benchmarks May 2026](https://lmcouncil.ai/benchmarks) - pulled 2026-05-13
- [Computer Use Leaderboard (Awesome Agents)](https://awesomeagents.ai/leaderboards/computer-use-leaderboard/) - pulled 2026-05-13
- Yue et al., *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*, arXiv:2311.16502 (2023)
- Liu et al., *MMBench: Is Your Multi-modal Model an All-around Player?*, arXiv:2307.06281 (2023)
- Lu et al., *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*, arXiv:2310.02255 (2023)
- Yu et al., *MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities*, arXiv:2308.02490 (2023)
- Mathew et al., *DocVQA: A Dataset for VQA on Document Images*, arXiv:2007.00398 (2020)
- Masry et al., *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning*, arXiv:2203.10244 (2022)
- Liu et al., *OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models*, arXiv:2305.07895 (2023)
- Liu et al., *OCRBench v2*, arXiv:2501.00321 (2025)
- Li et al., *Evaluating Object Hallucination in Large Vision-Language Models* (POPE), arXiv:2305.10355 (2023)
- Guan et al., *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion*, arXiv:2310.14566 (2023)
- Fu et al., *Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis*, arXiv:2405.21075 (2024)
- Li et al., *MVBench: A Comprehensive Multi-modal Video Understanding Benchmark*, arXiv:2311.17005 (2023)
- Mangalam et al., *EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding*, arXiv:2308.09126 (2023)
- Wu et al., *LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding*, arXiv:2407.15754 (2024)
- Cheng et al., *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents* (ScreenSpot), arXiv:2401.10935 (2024)
- Chen et al., *Are We on the Right Way for Evaluating Large Vision-Language Models?* (MMStar), arXiv:2403.20330 (2024)
- Fu et al., *BLINK: Multimodal Large Language Models Can See but Not Perceive*, arXiv:2404.12390 (2024)
- Wang et al., *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*, arXiv:2409.12191 (2024)
- Bai et al., *Qwen2.5-VL Technical Report*, arXiv:2502.13923 (2025)
- Chen et al., *Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling* (InternVL2.5), arXiv:2412.05271 (2024)
- Agrawal et al., *Pixtral 12B*, arXiv:2410.07073 (2024)
- Deitke et al., *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models*, arXiv:2409.17146 (2024)
- Li et al., *LLaVA-OneVision: Easy Visual Task Transfer*, arXiv:2408.03326 (2024)
- Yao et al., *MiniCPM-V: A GPT-4V Level MLLM on Your Phone*, arXiv:2408.01800 (2024)
- Wu et al., *DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding*, arXiv:2412.10302 (2024)
- Dai et al., *NVLM: Open Frontier-Class Multimodal LLMs*, arXiv:2409.11402 (2024)
- Tschannen et al., *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*, arXiv:2502.14786 (2025)
- Chuang et al., *Meta CLIP 2: A Worldwide Scaling Recipe*, arXiv:2507.22062 (2025)
- Fang et al., *Data Filtering Networks* (DFN), arXiv:2309.17425 (2023)
- [Jina CLIP v2 release blog](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/) - pulled 2026-05-13
- [Phi-4-multimodal release notes (Microsoft)](https://azure.microsoft.com/en-us/blog/empowering-innovation-the-next-generation-of-the-phi-family/) - pulled 2026-05-13
- [Anthropic Claude 4 / Sonnet 4.5 / Sonnet 4.6 / Opus 4.7 model cards](https://www.anthropic.com/) - accessed 2026-05-13
- [OpenAI GPT-4.1 / GPT-5 / o3 / o4-mini system cards](https://openai.com/) - accessed 2026-05-13
- [Google DeepMind Gemini 2.0 / 2.5 / 3.1 Pro release pages](https://deepmind.google/technologies/gemini/) - accessed 2026-05-13
- [xAI Grok-2/3/4 Vision blog posts](https://x.ai/) - accessed 2026-05-13
- Individual HuggingFace model cards for all listed open-weight VLMs, accessed 2026-05-13
