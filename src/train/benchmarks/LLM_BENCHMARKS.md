# LLM Benchmarks

Reference targets for the dl_techniques causal-LM training experiments.
Numbers are from public leaderboards, official model cards, and technical reports; cite the date you pulled them.

> **Snapshot date**: 2026-05-13

## What These Benchmarks Measure

- **MMLU** (Hendrycks et al., 2020, arXiv:2009.03300) - 57 subjects, 4-way multiple choice, undergraduate-level knowledge across STEM, humanities, social sciences. Saturated at the top (frontier models cluster 88-92).
- **MMLU-Pro** (Wang et al., 2024, arXiv:2406.01574) - reasoning-focused successor, 10-way MC, 12K questions, dedup'd noise from MMLU. Currently the standard "knowledge + reasoning" benchmark since MMLU saturated.
- **GPQA Diamond** (Rein et al., 2023, arXiv:2311.12022) - 198 hardest graduate-level physics/chem/bio questions, PhD experts 65%, Google-armed laypeople 34%. Frontier reasoning benchmark of 2025-2026.
- **HumanEval** (Chen et al., 2021, arXiv:2107.03374) - 164 Python function-completion problems, pass@1. Largely saturated (frontier > 90).
- **MBPP** - 974 Python problems, pass@1. Companion to HumanEval, similar saturation.
- **BBH** (Suzgun et al., 2022, arXiv:2210.09261) - BIG-Bench Hard, 23 challenging tasks (logical, temporal, arithmetic reasoning), 6,511 examples.
- **GSM8K** - 8.5K grade-school math word problems, exact-match. Saturated above 95.
- **MATH** (Hendrycks et al., 2021, arXiv:2103.03874) - 12.5K competition math problems with step-by-step solutions. MATH-500 is the standard eval subset.
- **HellaSwag / ARC / TruthfulQA** - older common-sense / science-MC / faithfulness benchmarks. Saturated, kept on Open LLM Leaderboard v2 for legacy comparison.
- **IFEval** (Zhou et al., 2023, arXiv:2311.07911) - 541 verifiable instruction-following prompts ("write exactly 3 paragraphs", "no commas"). Measures format compliance, not capability.
- **MT-Bench** (Zheng et al., 2023, arXiv:2306.05685) - 80 two-turn open-ended prompts judged by GPT-4. 0-10 scale. Largely deprecated in favor of Arena ELO.
- **Chatbot Arena (LMSYS / lmarena.ai)** - blind pairwise human preference, Bradley-Terry ELO. The only "wild" benchmark with no fixed answers; frontier models sit in 1450-1560.
- **AGIEval** - human-exam questions (SAT, LSAT, GMAT, Chinese gaokao). Mixed CN/EN.
- **LiveBench** (White et al., 2024) - monthly-refreshed contamination-resistant suite (math, coding, reasoning, language, IF, data analysis).
- **SWE-bench Verified / Lite / Pro** (Jimenez et al., 2023, arXiv:2310.06770) - real GitHub issue resolution with passing test patches. Verified is the 500-issue human-validated subset; Pro (Scale AI, 2026) is the contamination-hardened successor since OpenAI stopped reporting Verified.
- **AIME 2024 / 2025** - American Invitational Math Exam, 15 problems, integer answers 0-999. Now a primary reasoning-model benchmark; o3/GPT-5 hit 95+.
- **FrontierMath** (Epoch AI, 2024) - novel research-level math problems written by professional mathematicians, unpublished. Pre-o3 ceiling was ~2%; o3 broke 25%.
- **ZeroEval** - zero-shot reasoning aggregate (MMLU-Redux, ZebraLogic, GSM, MATH-L5, CRUX), no CoT prompting tricks.
- **TAU-bench / tau2-bench** - agentic tool-use in airline/retail/telecom customer-service scenarios.

## How to Read These Numbers

- **Contamination is the dominant artifact.** MMLU, HumanEval, MBPP, GSM8K, MATH are in essentially every modern training mix. Treat any 2025+ frontier score on these as a lower bound on capability and a useless signal for ranking. OpenAI stopped reporting SWE-bench Verified in early 2026 for this reason.
- **Reasoning models vs base models.** o1/o3/R1/Claude-thinking/Gemini-thinking spend test-time compute (CoT, search). Their scores are not comparable to single-pass base models. Always note whether a number is "with thinking" / "high reasoning effort" / "pass@1 single sample".
- **pass@1 vs maj@k vs cons@64.** Math benchmarks especially get inflated by self-consistency (maj@64) or best-of-n. Always read the fine print.
- **Self-reported model cards vs independent leaderboards.** Model cards game prompts and few-shot setups; LiveBench and Artificial Analysis re-run with frozen scaffolds. Where they disagree, prefer the leaderboard.
- **Arena ELO caveats.** Style and verbosity bias the crowd; Llama/GPT models with longer, more confidently-formatted outputs score above their capability. Use it for chat-quality, not reasoning.
- **MoE active-vs-total.** DeepSeek-V3 has 37B active / 671B total. Inference cost tracks active; capability tracks somewhere between.

## Benchmark Tables

Columns: Model | Params (active/total) | Ctx | MMLU | GPQA Diamond | HumanEval | MATH | MT-Bench / Arena ELO | Year | License. Use "?" for unreported.

### Tiny (<=2B)

| Model | Params | Ctx | MMLU | GPQA | HumanEval | MATH | Arena ELO | Year | License |
|-------|--------|-----|------|------|-----------|------|-----------|------|---------|
| Qwen2.5-0.5B-Instruct | 0.5B | 32K | 47.5 | ? | 30.5 | 27.9 | ? | 2024 | Apache-2.0 |
| Qwen2.5-1.5B-Instruct | 1.5B | 32K | 60.9 | ? | 61.6 | 55.2 | ? | 2024 | Apache-2.0 |
| SmolLM2-1.7B-Instruct | 1.7B | 8K | ? (19.3 MMLU-Pro) | ? | ? | ? (48.2 GSM8K) | ? | 2024 | Apache-2.0 |
| Llama-3.2-1B-Instruct | 1.2B | 128K | 49.3 | ? | ? | 16.8 | ? | 2024 | Llama-3.2 |
| Llama-3.2-3B-Instruct | 3.2B | 128K | 63.4 | ? | 71.3 | 19.0 | ? | 2024 | Llama-3.2 |
| Phi-3.5-mini-instruct | 3.8B | 128K | 69.0 | ? | 62.8 | 48.5 | ? | 2024 | MIT |
| Gemma-2-2B-it | 2.6B | 8K | 51.3 | ? | 17.7 | ? | ~1130 | 2024 | Gemma |

### Small (2B-9B)

| Model | Params | Ctx | MMLU | GPQA | HumanEval | MATH | Arena ELO | Year | License |
|-------|--------|-----|------|------|-----------|------|-----------|------|---------|
| Mistral-7B-Instruct-v0.3 | 7.2B | 32K | 62.5 | ? | 36.0 | 13.0 | ~1075 | 2024 | Apache-2.0 |
| Llama-3.1-8B-Instruct | 8B | 128K | 73.0 | 30.4 | 72.6 | 51.9 | ~1175 | 2024 | Llama-3.1 |
| Ministral-8B-Instruct | 8B | 128K | 65.0 | ? | 34.8 | 54.5 | ? | 2024 | Mistral-Research |
| Qwen2.5-7B-Instruct | 7.6B | 128K | 74.8 | 36.4 | 84.8 | 75.5 | ~1230 | 2024 | Apache-2.0 |
| Gemma-2-9B-it | 9.2B | 8K | 71.3 | ? | 40.2 | 36.6 | ~1190 | 2024 | Gemma |

### Medium (9B-30B)

| Model | Params | Ctx | MMLU | GPQA | HumanEval | MATH | Arena ELO | Year | License |
|-------|--------|-----|------|------|-----------|------|-----------|------|---------|
| Mistral-Nemo-12B-Instruct | 12.2B | 128K | 68.0 | ? | ? | ? | ~1145 | 2024 | Apache-2.0 |
| Qwen2.5-14B-Instruct | 14.7B | 128K | 79.7 | 45.5 | 83.5 | 80.0 | ? | 2024 | Apache-2.0 |
| Phi-4 | 14B | 16K | 84.8 | 56.1 | 82.6 | 80.4 | ? | 2024 | MIT |
| Gemma-2-27B-it | 27.2B | 8K | 75.2 | ? | 51.8 | ? | ~1218 | 2024 | Gemma |

### Large (30B-100B)

| Model | Params | Ctx | MMLU | GPQA | HumanEval | MATH | Arena ELO | Year | License |
|-------|--------|-----|------|------|-----------|------|-----------|------|---------|
| Command-R+ | 104B | 128K | 75.7 | ? | ? | ? | ~1190 | 2024 | CC-BY-NC |
| Llama-3.3-70B-Instruct | 70B | 128K | 86.0 (68.9 Pro) | 50.5 | 88.4 | 77.0 | ~1255 | 2024 | Llama-3.3 |
| Qwen2.5-72B-Instruct | 72.7B | 128K | 86.1 | 49.0 | 86.6 | 83.1 | ~1260 | 2024 | Qwen |
| Mistral-Large-2 (2407) | 123B | 128K | 84.0 | ? | 92.0 | 71.5 | ~1250 | 2024 | Mistral-Research |

### XL / MoE (>=100B total or frontier dense)

| Model | Active/Total | Ctx | MMLU | GPQA | HumanEval | MATH | Arena ELO | Year | License |
|-------|--------------|-----|------|------|-----------|------|-----------|------|---------|
| Mixtral-8x22B-Instruct | 39B/141B | 64K | 77.3 | ? | 75.0 | 41.8 | ~1155 | 2024 | Apache-2.0 |
| Llama-3.1-405B-Instruct | 405B | 128K | 87.3 | 50.7 | 89.0 | 73.8 | ~1265 | 2024 | Llama-3.1 |
| DeepSeek-V3 | 37B/671B | 128K | 88.5 (75.9 Pro) | 59.1 | 82.6 | 61.6 (MATH); 90.2 (M-500) | ~1310 | 2024 | DeepSeek |
| DeepSeek-R1 | 37B/671B | 128K | 90.8 (84.0 Pro) | 71.5 | ? (65.9 LCB) | 97.3 (M-500) | ~1360 | 2025 | MIT |
| Qwen2.5-Max | ?/? (MoE) | 32K | 87.9 (76.1 Pro) | 60.1 | 73.2 | 68.5 | ~1340 | 2025 | Proprietary |

### Proprietary / API (frontier, 2024-2026)

| Model | Ctx | MMLU(-Pro) | GPQA Diamond | HumanEval | MATH / AIME25 | Arena ELO | Year | License |
|-------|-----|------------|--------------|-----------|---------------|-----------|------|---------|
| GPT-4o (2024-08) | 128K | 88.7 / 74.7 | 53.6 | 90.2 | 76.6 / ? | ~1285 | 2024 | API |
| GPT-4.1 | 1M | 90.2 / 80.1 | 66.3 | ? | ? / ? | ~1370 | 2025 | API |
| o1 | 200K | 92.3 / ? | 78.0 | 89.0 | 94.8 / 83.3 | ~1355 | 2024 | API |
| o3 | 200K | ? / 85+ | 87.7 | ? | ? / 88.9 | ~1420 | 2025 | API |
| o4-mini | 200K | ? / 83 | 81.4 | ? | ? / 92.7 | ~1410 | 2025 | API |
| GPT-5 | 400K | ? / 87+ | 88.4 (pro) | ? | ? / 94.6 | ~1485 | 2025 | API |
| Claude 3.5 Sonnet (new) | 200K | 88.3 / 78.0 | 65.0 | 93.7 | 78.3 / ? | ~1290 | 2024 | API |
| Claude 3.7 Sonnet (thinking) | 200K | ? / 84.8 | 78.2 | ? | 82.2 / ? | ~1340 | 2025 | API |
| Claude Opus 4 | 200K | ? / 87.5 | 83.3 | ? | ? / 88.0 | ~1380 | 2025 | API |
| Claude Sonnet 4.5 | 200K | ? / ? | ~84 | ? | ? / ? | ~1430 | 2025 | API |
| Claude Sonnet 4.6 | 200K | ? / ? | 74.1 | ? | ? / ? | ~1455 | 2026 | API |
| Claude Opus 4.6 | 200K | ? / ? | 91.3 | ? | ? / ? | ~1500 | 2026 | API |
| Claude Opus 4.7 | 200K | ? / ? | 94.2 | ? | ? / ? | ~1510 | 2026 | API |
| Claude Haiku 4.5 | 200K | ? / ? | ? | ? | ? / ? | ~1380 | 2026 | API |
| Gemini 2.0 Flash | 1M | ? / 77.6 | 62.1 | ? | 89.7 / ? | ~1355 | 2025 | API |
| Gemini 2.5 Pro | 1M | ? / 86.0 | 84.0 | ? | ? / 86.7 | ~1450 | 2025 | API |
| Gemini 2.5 Flash | 1M | ? / 81 | 73 | ? | ? / 78 | ~1395 | 2025 | API |
| Grok-3 | 1M | ? / 79.9 | 84.6 | ? | ? / 93.3 | ~1400 | 2025 | API |
| Grok-4 | 256K | ? / 87.0 | 87.5 (88 Heavy) | ? | ? / 91.7 | ~1470 | 2025 | API |

### SWE-bench Verified (snapshot 2026-05; OpenAI now recommends SWE-bench Pro)

| Model | SWE-bench Verified | SWE-bench Pro | Source |
|-------|---------------------|---------------|--------|
| Claude Opus 4.7 (adaptive) | 87.6 | 64.3 | Anthropic |
| Claude Opus 4.6 | 80.8 | 53.4 | Anthropic |
| Claude Sonnet 4.6 | 79.6 | ? | Anthropic |
| Claude Haiku 4.5 | 73.3 | 39.5 | Anthropic |
| GPT-5.3 Codex | 85.0 | ? | OpenAI |
| Gemini 2.5 Pro | ~67 | ? | Google |
| DeepSeek-R1 | ~49 | ? | DeepSeek |
| Llama-3.1-405B | ~24 | ? | community |

## 2026 SOTA Themes Captured in the Numbers

- **Reasoning models dominate the frontier.** o3, GPT-5, Claude Opus 4.7-thinking, Gemini 2.5 Pro, Grok-4, and DeepSeek-R1 all spend test-time compute on extended chains-of-thought. They have pulled away from base models by 15-30 points on GPQA Diamond and AIME 2025 while remaining roughly tied on MMLU.
- **MoE wins at the top of open weights.** DeepSeek-V3 (37B active / 671B total) and DeepSeek-R1 reach frontier reasoning at inference cost comparable to a dense 40B. The dense-only era for >100B is largely over outside Meta (Llama-3.1-405B) and Mistral.
- **MMLU is saturated.** Frontier models cluster 88-92, within eval noise. Useful for ranking only sub-70B open models. Reporting has shifted to MMLU-Pro, GPQA Diamond, AIME, and FrontierMath.
- **Long context is now table stakes.** Gemini 2.5 Pro and GPT-4.1 ship 1M-token windows; Llama 3.1/3.2/3.3 ship 128K; Claude sits at 200K. Active research is now on context efficiency, not length.
- **Agentic / tool-use is the new frontier benchmark axis.** SWE-bench Verified/Pro, TAU-bench, and BrowseComp now matter more than HumanEval. Frontier models now resolve 60-90% of real GitHub issues end-to-end.
- **The open-closed gap has narrowed in capability but widened in product.** DeepSeek-R1 matches o1 on MATH-500 and approaches o3 on GPQA, but Anthropic/OpenAI/Google ship superior tool-use harnesses, browsing, memory, and code-execution agents.
- **Contamination has rendered HumanEval and SWE-bench Verified low-signal.** Frontier scores cluster within a few points; the real differentiation is on contamination-hardened benchmarks (LiveBench, SWE-bench Pro, FrontierMath, Humanity's Last Exam).

## Sources

- [LMSYS Chatbot Arena Leaderboard](https://lmarena.ai/leaderboard) - pulled 2026-05-13
- [Open LLM Leaderboard v2 (HuggingFace)](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [LiveBench](https://livebench.ai/)
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [Artificial Analysis - GPQA Diamond](https://artificialanalysis.ai/evaluations/gpqa-diamond)
- [Artificial Analysis - MMLU-Pro](https://artificialanalysis.ai/evaluations/mmlu-pro)
- [llm-stats.com - cross-benchmark aggregator](https://llm-stats.com/)
- [Vellum LLM Leaderboard 2026](https://www.vellum.ai/llm-leaderboard)
- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (MMLU), arXiv:2009.03300 (2020)
- Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Benchmark*, arXiv:2406.01574 (2024)
- Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, arXiv:2311.12022 (2023)
- Chen et al., *Evaluating Large Language Models Trained on Code* (HumanEval), arXiv:2107.03374 (2021)
- Suzgun et al., *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them* (BBH), arXiv:2210.09261 (2022)
- Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*, arXiv:2103.03874 (2021)
- Zhou et al., *Instruction-Following Evaluation for Large Language Models* (IFEval), arXiv:2311.07911 (2023)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685 (2023)
- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, arXiv:2310.06770 (2023)
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437 (2024)
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*, arXiv:2501.12948 (2025)
- Qwen Team, *Qwen2.5 Technical Report*, arXiv:2412.15115 (2024)
- Meta, *Llama 3.1 / 3.2 / 3.3* model cards on huggingface.co/meta-llama
- Microsoft, *Phi-4 Technical Report*, arXiv:2412.08905 (2024)
- Google DeepMind, *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv:2408.00118 (2024)
- [Anthropic - Claude 4.7 / 4.6 / 4.5 model cards](https://www.anthropic.com/news)
- [OpenAI - GPT-5, o3, o4-mini system cards](https://openai.com/index/)
- [Google - Gemini 2.5 Pro model card](https://modelcards.withgoogle.com/assets/documents/gemini-2.5-pro.pdf)
- [xAI - Grok 3 / Grok 4 announcement](https://x.ai/news/grok-4)
- [Mistral AI - Mistral Large 2 / NeMo / Ministral](https://mistral.ai/news/)
- Individual HuggingFace model cards for all listed open-weight models, accessed 2026-05-13
