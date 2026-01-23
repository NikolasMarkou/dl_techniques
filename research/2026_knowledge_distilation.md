# Knowledge Distillation for Small Specialized Language Models

A comprehensive guide to training task-specific small language models using knowledge distillation from large teacher models.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [The Distillation Pipeline](#the-distillation-pipeline)
4. [Data Preparation](#data-preparation)
5. [Teacher Model Selection and Validation](#teacher-model-selection-and-validation)
6. [Student Model Architecture](#student-model-architecture)
7. [Training Methodology](#training-methodology)
8. [Evaluation and Quality Assurance](#evaluation-and-quality-assurance)
9. [Deployment](#deployment)
10. [Case Study: Text2SQL](#case-study-text2sql)
11. [Common Pitfalls](#common-pitfalls)
12. [Tools and Resources](#tools-and-resources)

---

## Introduction

### The Problem

Large Language Models (LLMs) excel at general tasks but come with significant costs:
- High inference latency (100ms-10s per request)
- API costs at scale ($0.01-$0.10+ per request)
- Data privacy concerns (sensitive data leaves your infrastructure)
- Network dependency (offline use impossible)

Small Language Models (SLMs) in the 0.5B-3B parameter range solve these problems but suffer from poor performance on specialized tasks requiring exact outputs.

### The Solution

**Knowledge Distillation** compresses the capabilities of a large teacher model into a small student model for a specific task domain. The result: a tiny specialist that runs locally with near-teacher performance on the target task.

### When to Use This Approach

| Use Case | Suitability |
|----------|-------------|
| High-volume production inference | Excellent |
| Latency-critical applications | Excellent |
| Offline/edge deployment | Excellent |
| Data privacy requirements | Excellent |
| Rapidly changing task requirements | Poor |
| Tasks requiring broad knowledge | Poor |
| One-off or low-volume tasks | Poor |

---

## Core Concepts

### What is Knowledge Distillation?

Knowledge distillation transfers learned behavior from a large "teacher" model to a smaller "student" model. The student learns to imitate the teacher's outputs rather than learning from raw data directly.

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISTILLATION                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌───────────┐  │
│   │   Teacher   │      │  Synthetic  │      │  Student  │  │
│   │   Model     │─────▶│   Dataset   │─────▶│   Model   │  │
│   │  (7B-70B+)  │      │  (input,    │      │ (0.5B-3B) │  │
│   │             │      │   output)   │      │           │  │
│   └─────────────┘      └─────────────┘      └───────────┘  │
│                                                             │
│   Large, expensive,    High-quality         Small, fast,   │
│   slow, accurate       training pairs       specialized    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight

> Distillation amplifies **both** competence **and** incompetence.

If the teacher makes systematic errors, the student will learn those errors with high fidelity. This makes teacher validation critical.

### Distillation vs Fine-Tuning

| Aspect | Traditional Fine-Tuning | Knowledge Distillation |
|--------|------------------------|------------------------|
| Data Source | Human-labeled data | Teacher model outputs |
| Data Volume | Typically 1K-100K examples | Can generate unlimited |
| Data Quality | Limited by labeling quality | Limited by teacher quality |
| Iteration Speed | Slow (relabeling required) | Fast (regenerate from teacher) |
| Task Adaptation | Requires new labels | Requires new prompts |

---

## The Distillation Pipeline

### High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DISTILLATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SEED DATA          2. TEACHER EVAL       3. GENERATION         │
│  ┌───────────┐         ┌───────────┐         ┌───────────┐         │
│  │ 10-100    │────────▶│ Validate  │────────▶│ Generate  │         │
│  │ examples  │         │ teacher   │         │ synthetic │         │
│  │           │         │ accuracy  │         │ pairs     │         │
│  └───────────┘         └───────────┘         └───────────┘         │
│                              │                     │                │
│                              │ <80%? STOP          │                │
│                              ▼                     ▼                │
│                        [Fix teacher          4. TRAINING           │
│                         or task]             ┌───────────┐         │
│                                              │ Train     │         │
│                                              │ student   │         │
│                                              │ model     │         │
│                                              └───────────┘         │
│                                                    │                │
│                                                    ▼                │
│  6. DEPLOYMENT         5. EVALUATION         ┌───────────┐         │
│  ┌───────────┐         ┌───────────┐         │ Evaluate  │         │
│  │ Export    │◀────────│ Compare   │◀────────│ on held-  │         │
│  │ GGUF/HF   │         │ to        │         │ out set   │         │
│  │           │         │ baseline  │         │           │         │
│  └───────────┘         └───────────┘         └───────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Steps in Detail

#### Step 1: Seed Data Collection (10-100 examples)

Collect representative examples of your task. These serve two purposes:
- Validate teacher model performance
- Provide templates for synthetic generation

**Quality requirements:**
- Cover edge cases and variations
- Include both easy and hard examples
- Representative of production distribution

#### Step 2: Teacher Evaluation (Critical Gate)

Before any training, validate that the teacher model can actually perform the task.

**Minimum threshold:** 80% accuracy on seed data

If the teacher fails this gate:
- Try a different teacher model
- Improve task prompting
- Reconsider if the task is well-defined

#### Step 3: Synthetic Data Generation

Use the teacher to generate training pairs from:
- Seed examples (direct generation)
- Augmented examples (variations of seeds)
- Novel examples (teacher generates new inputs and outputs)

**Target volume:** 1,000-10,000 high-quality pairs

#### Step 4: Student Training

Train the small model to imitate teacher outputs using standard supervised learning.

#### Step 5: Evaluation

Compare student performance against:
- Base model (no training) - the improvement baseline
- Teacher model - the performance ceiling
- Production requirements - the deployment threshold

#### Step 6: Deployment

Export to efficient inference format (GGUF, HuggingFace, LoRA) for local deployment.

---

## Data Preparation

### Seed Data Format

Structure seed data as input-output pairs in JSONL format:

```jsonl
{"input": "Which artists have albums with over 1M sales?", "output": "SELECT artist_name FROM artists a JOIN albums al ON a.id = al.artist_id WHERE al.sales > 1000000 GROUP BY a.id", "metadata": {"difficulty": "medium", "tables": ["artists", "albums"]}}
{"input": "Count customers by country", "output": "SELECT country, COUNT(*) as customer_count FROM customers GROUP BY country ORDER BY customer_count DESC", "metadata": {"difficulty": "easy", "tables": ["customers"]}}
```

### Data Augmentation Strategies

#### 1. Input Paraphrasing

Generate variations of the same query:

```
Original: "Which artists have albums with over 1M sales?"
Variant 1: "List artists whose albums sold more than 1 million copies"
Variant 2: "Find all artists with album sales exceeding 1M"
Variant 3: "Show me artists that have sold over a million albums"
```

#### 2. Schema Substitution

Apply the same query patterns to different schemas:

```
Original (music DB): "Which artists have albums with over 1M sales?"
Substituted (retail DB): "Which suppliers have products with over 1M units sold?"
```

#### 3. Complexity Scaling

Generate simpler and more complex versions:

```
Simple: "Count all artists"
Medium: "Count artists by genre"
Complex: "Count artists by genre who have at least 3 albums released after 2010"
```

### Synthetic Data Generation Prompt Template

```
You are generating training data for a Text2SQL model.

SCHEMA:
{schema_definition}

TASK:
Generate {n} diverse SQL queries with their natural language descriptions.

REQUIREMENTS:
1. Cover a range of SQL operations: SELECT, JOIN, GROUP BY, HAVING, subqueries
2. Vary difficulty from simple single-table queries to complex multi-table operations
3. Include edge cases: NULL handling, date operations, string matching
4. Natural language should be conversational, not formulaic

OUTPUT FORMAT (JSONL, one per line):
{"input": "natural language question", "output": "SQL query"}

EXAMPLES:
{seed_examples}

Generate {n} new examples:
```

### Data Quality Validation

Before training, validate synthetic data:

```python
def validate_synthetic_data(data: list[dict], validator_fn) -> dict:
    """
    Validate synthetic training data quality.
    
    Args:
        data: List of {"input": str, "output": str} pairs
        validator_fn: Function that returns True if output is valid
    
    Returns:
        Validation statistics
    """
    results = {
        "total": len(data),
        "valid": 0,
        "invalid": 0,
        "invalid_examples": []
    }
    
    for item in data:
        if validator_fn(item["input"], item["output"]):
            results["valid"] += 1
        else:
            results["invalid"] += 1
            if len(results["invalid_examples"]) < 10:
                results["invalid_examples"].append(item)
    
    results["valid_ratio"] = results["valid"] / results["total"]
    return results
```

**Validation checks for Text2SQL:**
- SQL parses without syntax errors
- Referenced tables exist in schema
- Column names are valid
- Query executes successfully

---

## Teacher Model Selection and Validation

### Recommended Teacher Models

| Model | Parameters | Best For | Considerations |
|-------|------------|----------|----------------|
| Claude 3.5 Sonnet | ~70B (est.) | Complex reasoning, code | High quality, API-only |
| GPT-4o | ~200B (est.) | General tasks | High quality, expensive |
| DeepSeek-V3 | 671B MoE | Technical tasks, code | Open weights available |
| Llama 3.1 70B | 70B | Self-hosted needs | Run locally if resources permit |
| Qwen2.5-72B | 72B | Multilingual, code | Strong on structured outputs |

### Teacher Validation Protocol

```python
def validate_teacher(
    teacher_fn,
    seed_data: list[dict],
    evaluator_fn,
    threshold: float = 0.80
) -> dict:
    """
    Validate teacher model performance before distillation.
    
    Args:
        teacher_fn: Function that takes input and returns output
        seed_data: List of {"input": str, "expected": str} pairs
        evaluator_fn: Function(predicted, expected) -> bool
        threshold: Minimum accuracy required (default 80%)
    
    Returns:
        Validation results with pass/fail status
    """
    results = []
    
    for item in seed_data:
        predicted = teacher_fn(item["input"])
        is_correct = evaluator_fn(predicted, item["expected"])
        results.append({
            "input": item["input"],
            "expected": item["expected"],
            "predicted": predicted,
            "correct": is_correct
        })
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    
    return {
        "accuracy": accuracy,
        "passed": accuracy >= threshold,
        "threshold": threshold,
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "failures": [r for r in results if not r["correct"]]
    }
```

### LLM-as-Judge Evaluation

For tasks where exact matching is insufficient, use an LLM judge:

```python
JUDGE_PROMPT = """
You are evaluating a Text2SQL model output.

SCHEMA:
{schema}

QUESTION:
{question}

EXPECTED SQL:
{expected}

PREDICTED SQL:
{predicted}

Evaluate if the predicted SQL is semantically equivalent to the expected SQL.
Consider:
1. Do they return the same results for the given schema?
2. Are differences cosmetic (column order, alias names) or semantic?
3. Is the predicted SQL valid and executable?

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation.
"""
```

---

## Student Model Architecture

### Recommended Base Models

| Model | Parameters | Context | Use Case |
|-------|------------|---------|----------|
| Qwen2.5-0.5B | 0.5B | 32K | Ultra-lightweight, edge |
| Qwen2.5-1.5B | 1.5B | 32K | Balanced size/performance |
| Qwen2.5-3B | 3B | 32K | Higher quality needs |
| Phi-3.5-mini | 3.8B | 128K | Long context requirements |
| Llama-3.2-1B | 1B | 128K | Meta ecosystem |
| Llama-3.2-3B | 3B | 128K | Meta ecosystem, higher quality |

### Model Selection Criteria

1. **Deployment constraints:** Memory, latency, hardware
2. **Task complexity:** Simple classification vs. generation
3. **Context requirements:** How much input context needed
4. **Ecosystem:** Integration requirements (GGUF, HuggingFace, vLLM)

### Architecture Considerations

For structured output tasks (SQL, JSON, code), models with:
- Strong instruction following
- Good tokenizer coverage for target domain
- Pretrained on similar data distributions

---

## Training Methodology

### Training Objectives

#### 1. Standard Cross-Entropy (Supervised Fine-Tuning)

The simplest approach: train to predict teacher outputs.

```python
# Pseudocode
loss = cross_entropy(
    student_logits,      # Student model predictions
    teacher_output_ids   # Tokenized teacher outputs
)
```

#### 2. Sequence-Level Knowledge Distillation

Match teacher output sequences rather than token distributions.

```python
# Generate synthetic pairs
synthetic_data = []
for input in inputs:
    teacher_output = teacher.generate(input)
    synthetic_data.append((input, teacher_output))

# Train student on synthetic data
student.train(synthetic_data)
```

#### 3. Soft Label Distillation (Advanced)

Use teacher's token probability distributions as soft targets.

```python
# Requires access to teacher logits
loss = kl_divergence(
    student_logits / temperature,
    teacher_logits / temperature
) * temperature^2 + cross_entropy(student_logits, labels)
```

### Training Hyperparameters

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Learning Rate | 1e-5 to 5e-5 | Lower for smaller datasets |
| Batch Size | 4-32 | Larger batches more stable |
| Epochs | 3-10 | Monitor for overfitting |
| Warmup Ratio | 0.03-0.1 | Gradual learning rate increase |
| Weight Decay | 0.01-0.1 | Regularization |
| Max Sequence Length | Task-dependent | Pad/truncate consistently |

### Training Configuration Example

```python
training_config = {
    # Model
    "base_model": "Qwen/Qwen2.5-0.5B",
    "adapter": "lora",  # or "full" for full fine-tuning
    
    # LoRA config (if using)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # Training
    "learning_rate": 2e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,  # Effective batch = 32
    "num_epochs": 5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_length": 512,
    
    # Optimization
    "optimizer": "adamw",
    "lr_scheduler": "cosine",
    "bf16": True,  # Use bfloat16 if available
    "gradient_checkpointing": True,  # Save memory
    
    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "best",
    "metric_for_best": "eval_accuracy"
}
```

### LoRA vs Full Fine-Tuning

| Aspect | LoRA | Full Fine-Tuning |
|--------|------|------------------|
| Memory | 10-20% of full | 100% |
| Training Speed | Faster | Slower |
| Performance | ~95% of full | 100% |
| Export Size | Small adapter | Full model |
| Deployment | Base + adapter | Single model |
| Iteration Speed | Fast | Slow |

**Recommendation:** Start with LoRA for rapid iteration, use full fine-tuning for final production model if performance gap is significant.

---

## Evaluation and Quality Assurance

### Evaluation Metrics

#### Task-Specific Metrics

| Task | Primary Metric | Secondary Metrics |
|------|----------------|-------------------|
| Text2SQL | Execution Accuracy | Valid SQL %, Exact Match |
| Classification | F1 Score | Precision, Recall, Accuracy |
| Extraction | Exact Match | Partial Match, Character F1 |
| Generation | BLEU/ROUGE | Human Evaluation |

#### Model Comparison Framework

```python
def comprehensive_evaluation(
    models: dict[str, callable],
    test_data: list[dict],
    evaluators: dict[str, callable]
) -> dict:
    """
    Compare multiple models across multiple metrics.
    
    Args:
        models: {"name": model_fn} mapping
        test_data: List of test examples
        evaluators: {"metric_name": evaluator_fn} mapping
    
    Returns:
        Results matrix
    """
    results = {name: {} for name in models}
    
    for model_name, model_fn in models.items():
        predictions = [model_fn(item["input"]) for item in test_data]
        
        for metric_name, evaluator_fn in evaluators.items():
            scores = [
                evaluator_fn(pred, item["expected"]) 
                for pred, item in zip(predictions, test_data)
            ]
            results[model_name][metric_name] = sum(scores) / len(scores)
    
    return results
```

### Expected Results

A successful distillation should show:

| Model | Expected Performance |
|-------|---------------------|
| Base (untrained) | 20-40% |
| Distilled Student | 70-90% |
| Teacher | 80-95% |

**Red flags:**
- Student < 50% → Training issues or data quality problems
- Student > Teacher → Likely overfitting to test set
- Student = Base → Training didn't work

### Holdout Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SPLIT STRATEGY                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Seed Data (100 examples)                                   │
│  ├── Teacher Validation (20%)  → Gate: Is teacher good?    │
│  └── Test Set (80%)            → Final evaluation          │
│                                                             │
│  Synthetic Data (5000 examples)                             │
│  ├── Training (90%)            → Student training           │
│  └── Validation (10%)          → Hyperparameter tuning     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment

### Export Formats

#### GGUF (Recommended for Local Inference)

```bash
# Convert to GGUF using llama.cpp
python convert_hf_to_gguf.py \
    --model ./distilled-model \
    --outfile ./distilled-model.gguf \
    --outtype q4_k_m  # Quantization type
```

**Quantization options:**
| Type | Size Reduction | Quality Loss |
|------|----------------|--------------|
| f16 | 50% | None |
| q8_0 | 75% | Minimal |
| q4_k_m | 87% | Small |
| q4_0 | 87% | Moderate |

#### HuggingFace (For Python Integration)

```python
# Save in HuggingFace format
model.save_pretrained("./distilled-model")
tokenizer.save_pretrained("./distilled-model")

# Upload to Hub (optional)
model.push_to_hub("username/distilled-model")
```

#### LoRA Adapter (Minimal Storage)

```python
# Save only the adapter weights
model.save_pretrained("./distilled-adapter")
# Size: ~10-50MB vs 1-6GB for full model
```

### Inference Deployment Options

| Platform | Use Case | Setup Complexity |
|----------|----------|------------------|
| llama.cpp | Local CLI, C++ integration | Low |
| Ollama | Local with API | Very Low |
| vLLM | High-throughput server | Medium |
| TensorRT-LLM | NVIDIA optimized | High |
| ONNX Runtime | Cross-platform | Medium |

### Sample Deployment Script (Ollama)

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./distilled-model.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

SYSTEM "You are a SQL generation assistant. Convert natural language to SQL."
EOF

# Create Ollama model
ollama create sql-assistant -f Modelfile

# Test
ollama run sql-assistant "List all customers from Germany"
```

---

## Case Study: Text2SQL

### Overview

| Metric | Value |
|--------|-------|
| Task | Natural language → SQL |
| Base Model | Qwen2.5-0.5B |
| Teacher | DeepSeek-V3 |
| Seed Examples | ~100 |
| Synthetic Examples | ~5,000 |
| Training Time | ~2 hours (single GPU) |
| Final Model Size | 2.2GB (GGUF q4_k_m) |

### Results

| Model | Execution Accuracy |
|-------|-------------------|
| Base Qwen2.5-0.5B | 36% |
| Teacher (DeepSeek-V3) | 80% |
| **Distilled 0.5B** | **74%** |

### Before/After Examples

**Query:** "Which artists have albums with over 1M sales?"

| Model | Output | Valid? |
|-------|--------|--------|
| Base | `SELECT * FROM artists WHERE genre IS NULL` | No |
| Distilled | `SELECT DISTINCT a.name FROM artists a JOIN albums al ON a.id = al.artist_id WHERE al.sales > 1000000` | Yes |

**Query:** "Average order value by customer country"

| Model | Output | Valid? |
|-------|--------|--------|
| Base | `SELECT AVG(total) FROM orders` | Partial |
| Distilled | `SELECT c.country, AVG(o.total) as avg_order FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.country` | Yes |

---

## Common Pitfalls

### 1. Skipping Teacher Validation

**Problem:** Training on outputs from a teacher that can't do the task.

**Symptom:** Student performs poorly despite training.

**Solution:** Always validate teacher performance before generating synthetic data.

### 2. Insufficient Seed Diversity

**Problem:** Seed examples don't cover the task distribution.

**Symptom:** Student fails on common variations not in seed set.

**Solution:** Systematically sample seeds to cover:
- Difficulty levels (easy, medium, hard)
- Input variations (phrasing, terminology)
- Edge cases (nulls, empty results, errors)

### 3. Synthetic Data Contamination

**Problem:** Invalid or incorrect synthetic examples in training data.

**Symptom:** Student learns incorrect patterns.

**Solution:** Validate all synthetic data before training:
- Parse/compile outputs
- Execute and verify results
- Sample manual review

### 4. Overfitting to Synthetic Distribution

**Problem:** Student memorizes patterns rather than learning the task.

**Symptom:** High train accuracy, low test accuracy.

**Solution:**
- Use held-out test set from seed data (not synthetic)
- Increase synthetic data diversity
- Apply regularization (dropout, weight decay)

### 5. Wrong Base Model Selection

**Problem:** Base model lacks required capabilities.

**Symptom:** Training plateaus at low accuracy.

**Solution:** Verify base model has:
- Sufficient vocabulary coverage
- Basic instruction following
- Relevant pretraining

### 6. Inference Configuration Mismatch

**Problem:** Different tokenization or generation settings between training and inference.

**Symptom:** Good training metrics, poor production results.

**Solution:** Match exactly:
- Tokenizer settings
- System prompt (if any)
- Generation parameters (temperature, top_p)
- Stop tokens

---

## Tools and Resources

### Distillation Frameworks

| Tool | Description | Link |
|------|-------------|------|
| distil-cli | Claude Code skill for distillation | [GitHub](https://github.com/distil-labs/distil-cli-skill) |
| axolotl | General fine-tuning framework | [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl) |
| unsloth | Fast fine-tuning for Llama/Mistral | [GitHub](https://github.com/unslothai/unsloth) |
| LLaMA-Factory | Unified fine-tuning framework | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |

### Inference Runtimes

| Tool | Description | Link |
|------|-------------|------|
| llama.cpp | CPU/GPU inference, GGUF format | [GitHub](https://github.com/ggerganov/llama.cpp) |
| Ollama | Local LLM runner | [ollama.ai](https://ollama.ai) |
| vLLM | High-throughput inference | [GitHub](https://github.com/vllm-project/vllm) |
| TGI | HuggingFace inference server | [GitHub](https://github.com/huggingface/text-generation-inference) |

### Evaluation Tools

| Tool | Description | Link |
|------|-------------|------|
| lm-evaluation-harness | Standard LLM benchmarks | [GitHub](https://github.com/EleutherAI/lm-evaluation-harness) |
| sql-eval | Text2SQL evaluation | Various implementations |
| MTEB | Embedding benchmarks | [GitHub](https://github.com/embeddings-benchmark/mteb) |

### Example Projects

| Project | Description | Link |
|---------|-------------|------|
| Text2SQL with Claude | Complete distillation example | [GitHub](https://github.com/distil-labs/distil-example-text2sql-with-claude) |

---

## Appendix: Quick Reference

### Minimum Viable Distillation Checklist

- [ ] Collect 50-100 seed examples covering task distribution
- [ ] Validate teacher accuracy ≥80% on seed data
- [ ] Generate 1,000-10,000 synthetic training pairs
- [ ] Validate synthetic data quality (parse, execute, sample review)
- [ ] Split: 90% train, 10% validation (synthetic), 80% seed held for test
- [ ] Train with LoRA, lr=2e-5, epochs=3-5
- [ ] Evaluate on held-out seed test set
- [ ] Compare: base < student < teacher
- [ ] Export to GGUF/HuggingFace
- [ ] Test inference pipeline end-to-end

### Expected Timeline

| Phase | Duration |
|-------|----------|
| Seed data collection | 1-3 days |
| Teacher validation | 1-2 hours |
| Synthetic generation | 2-8 hours |
| Data validation | 2-4 hours |
| Training | 2-8 hours |
| Evaluation | 1-2 hours |
| Deployment setup | 1-4 hours |
| **Total** | **2-5 days** |

### Cost Estimation

| Component | Estimated Cost |
|-----------|---------------|
| Teacher API calls (5K generations) | $10-50 |
| GPU training (A100, 4 hours) | $10-20 |
| Iteration cycles (3-5x) | 3-5x above |
| **Total** | **$50-300** |

---

## References

- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) - Gou et al., 2020
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) - Hsieh et al., 2023
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) - Gunasekar et al., 2023
- [Orca: Progressive Learning from Complex Explanation Traces](https://arxiv.org/abs/2306.02707) - Mukherjee et al., 2023
- [TinyLlama](https://arxiv.org/abs/2401.02385) - Zhang et al., 2024

---

*Guide compiled from [@TheAhmadOsman](https://x.com/TheAhmadOsman) Twitter thread (Jan 2026) and expanded with general best practices.*