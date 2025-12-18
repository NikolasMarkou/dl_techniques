# Efficient Multi-Modal LLMs for Low-Resource Dialect Adaptation
## A Complete Technical Guide for English-Greek-Cypriot Speech & Text Systems (2024-2025)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [State-of-the-Art Models](#state-of-the-art-models)
4. [Speech-to-Text Systems](#speech-to-text-systems)
5. [Metrics & Evaluation](#metrics--evaluation)
6. [Loss Functions](#loss-functions)
7. [Finetuning Strategies](#finetuning-strategies)
8. [Data Resources](#data-resources)
9. [Hardware & Deployment](#hardware--deployment)
10. [Implementation Guide](#implementation-guide)
11. [Benchmarks & Performance](#benchmarks--performance)
12. [References & Resources](#references--resources)

---

## Executive Summary

This guide covers the complete pipeline for building an efficient multi-modal LLM handling text and audio for English, Greek, and Cypriot Greek dialects. Target: models <100B parameters suitable for local deployment.

**Key Recommendations:**
- **Primary ASR**: Whisper-large-v3 (1.55B) or NVIDIA Canary-1B-v2
- **Multi-modal**: Qwen2-Audio (8.2B) for audio understanding
- **Low-resource adaptation**: XLS-R-300M with LoRA finetuning
- **Deployment**: faster-whisper (GPU) or whisper.cpp (CPU/Edge)

**Current SOTA Performance (2024-2025):**
| Task | Model | Metric | Score |
|------|-------|--------|-------|
| English ASR | Whisper-large-v3 | WER | 1.8% |
| Multilingual ASR | SeamlessM4T v2 | Avg WER | 45% lower than Whisper |
| Greek ASR | Canary-1B-v2 | WER | ~8-12% |
| Low-resource transfer | XLS-R-2B | WER (10min data) | 15-25% |
| Audio Understanding | Qwen2-Audio | AudioBench | #1 overall |

---

## Architecture Overview

### Multi-Modal LLM Architecture Patterns

#### Pattern 1: Encoder-Decoder Fusion
```
┌───────────────────────────────────────────────────────────────────┐
│                    ENCODER-DECODER FUSION                         │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Audio Input ──► [Audio Encoder] ──┐                             │
│                   (Whisper/W2V2)    │                             │
│                                     ▼                             │
│                              [Projection Layer]                   │
│                                     │                             │
│   Text Input ───► [Tokenizer] ──────┼──► [LLM Decoder] ──► Output │
│                                     │    (Llama/Qwen)             │
│                              [Cross-Attention]                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Examples**: Qwen2-Audio, LLaMA-Omni, Ultravox

#### Pattern 2: Dual-Encoder Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    DUAL-ENCODER (SALMONN)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Audio ──► [Whisper Encoder] ──┐                           │
│             (Speech features)    │                          │
│                                  ▼                          │
│                           [Q-Former] ──► [LLM] ──► Output   │
│                                  ▲                          │
│   Audio ──► [BEATs Encoder] ────┘                           │
│             (Non-speech audio)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Examples**: SALMONN, Pengi

#### Pattern 3: Unified Multimodal (Speech + Text Translation)
```
┌─────────────────────────────────────────────────────────────┐
│                 UNIFIED MULTIMODAL (SeamlessM4T)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Speech ──► [Conformer Encoder] ──┐                        │
│                                     ▼                       │
│                              [Shared Encoder]               │
│                                     │                       │
│   Text ────► [Text Embeddings] ────┘                        │
│                                     │                       │
│                                     ▼                       │
│                              [Decoder(s)]                   │
│                              /    |    \                    │
│                           Text  Speech  Both                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Examples**: SeamlessM4T v2, MMS

### Component Specifications

| Component | Options | Parameters | Purpose |
|-----------|---------|------------|---------|
| Audio Encoder | Whisper, Wav2Vec2, Conformer | 300M-1.5B | Feature extraction |
| Projection | Linear, Q-Former, Adapter | 10M-100M | Modality alignment |
| LLM Backbone | Llama-3, Qwen-2, Mistral | 7B-70B | Language understanding |
| Output Head | LM Head, Vocoder, CTC | 10M-500M | Task-specific output |

---

## State-of-the-Art Models

### Tier 1: Multi-Modal Audio-Language Models (<100B params)

#### Qwen2-Audio (8.2B parameters)
**Architecture**: Whisper-large-v3 encoder → Qwen-7B decoder

**Capabilities**:
- Voice chat and audio analysis modes
- ASR, speech translation, emotion recognition
- Sound event detection, music understanding

**Performance**:
| Benchmark | Score | Rank |
|-----------|-------|------|
| LibriSpeech Clean WER | 1.83% | #1 |
| LibriSpeech Other WER | 3.47% | #1 |
| AudioBench Overall | Best | #1 |
| COVOST2 En-De BLEU | 29.4 | Top-3 |

**Greek Support**: Input via Whisper encoder (recognition), output limited to 8 languages

**License**: Apache 2.0

**Repository**: `github.com/QwenLM/Qwen2-Audio`

```python
# Inference example
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "audio.wav"},
        {"type": "text", "text": "Transcribe this audio in Greek."}
    ]}
]
```

---

#### SeamlessM4T v2 (~2.3B parameters)
**Architecture**: Conformer encoder + NLLB decoder + speech vocoder

**Capabilities**:
- Speech-to-text (101 languages)
- Text-to-speech (35 languages)
- Speech-to-speech translation
- Text-to-text translation (96 languages)

**Performance**:
| Metric | Value | Comparison |
|--------|-------|------------|
| FLEURS WER | 45% lower | vs Whisper-large-v2 |
| Noise Robustness | 42-66% better | vs Whisper |
| Speaker Variation | 62% more robust | vs baseline |

**Greek Support**: ✓ Text I/O, ✓ Speech input, ✗ Speech output

**License**: CC-BY-NC 4.0

**Repository**: `github.com/facebookresearch/seamless_communication`

```python
from seamless_communication.models.inference import Translator

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=torch.device("cuda")
)

# Speech-to-text (Greek to English)
transcribed_text, _, _ = translator.predict(
    input="greek_audio.wav",
    task_str="S2TT",  # Speech-to-Text Translation
    tgt_lang="eng",
    src_lang="ell"
)
```

---

#### SALMONN (7B/13B parameters)
**Architecture**: Whisper-large-v2 + BEATs → Window Q-Former → Vicuna

**Unique Features**:
- Dual audio encoder (speech + general audio)
- Window-level Q-Former for temporal alignment
- LoRA-based training (highly adaptable)

**Performance**:
| Task | Score |
|------|-------|
| LibriSpeech WER | 2.1% |
| AudioCaps CIDEr | 72.3 |
| Speech Translation BLEU | 25.8 |

**Greek Support**: Via Whisper backbone (input only)

**License**: Research use

**Repository**: `github.com/bytedance/SALMONN`

---

#### LLaMA-Omni (8B parameters)
**Architecture**: Whisper encoder → Speech Adapter → Llama-3.1-8B → Streaming Vocoder

**Key Innovation**: ~226ms response latency for real-time interaction

**Performance**:
| Metric | Value |
|--------|-------|
| Response Latency | 226ms |
| Speech Quality MOS | 3.8/5 |
| Instruction Following | Comparable to GPT-4o |

**License**: Apache 2.0 + Llama license

**Repository**: `github.com/ictnlp/LLaMA-Omni`

---

#### Ultravox (8B parameters)
**Architecture**: Whisper encoder → Llama-3.1-8B (merged audio-text embeddings)

**Features**:
- Real-time voice chat
- No separate ASR step (end-to-end)
- Supports interruption handling

**License**: MIT + Llama license

**Repository**: `github.com/fixie-ai/ultravox`

---

### Tier 2: Compact Models for Edge Deployment

| Model | Parameters | Specialty | License |
|-------|------------|-----------|---------|
| Mini-Omni | 500M | Edge deployment | Apache 2.0 |
| Whisper-tiny | 39M | Fastest ASR | MIT |
| Distil-Whisper | 756M | 6x faster, English | MIT |
| mHuBERT-147 | 95M | Few-shot champion | MIT |

---

## Speech-to-Text Systems

### Comprehensive ASR Model Comparison

#### OpenAI Whisper Family

| Variant | Parameters | VRAM | Speed | WER (en) | Greek Support |
|---------|------------|------|-------|----------|---------------|
| tiny | 39M | 1GB | 32x | 7.6% | ✓ (limited) |
| base | 74M | 1GB | 16x | 5.0% | ✓ |
| small | 244M | 2GB | 6x | 3.4% | ✓ |
| medium | 769M | 5GB | 2x | 2.5% | ✓ |
| large-v2 | 1.55B | 10GB | 1x | 2.1% | ✓ |
| large-v3 | 1.55B | 10GB | 1x | 1.8% | ✓ |
| large-v3-turbo | 809M | 6GB | 8x | 1.9% | ✓ |

**Key Finding**: Greek Podcast Corpus research (Interspeech 2024) showed finetuned Whisper-small/medium outperform Whisper-large-v2 on Greek domain-specific content.

```python
# Whisper with HuggingFace
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16
).to("cuda")

# Force Greek transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="greek", 
    task="transcribe"
)

input_features = processor(
    audio_array, 
    sampling_rate=16000, 
    return_tensors="pt"
).input_features.to("cuda", torch.float16)

predicted_ids = model.generate(
    input_features, 
    forced_decoder_ids=forced_decoder_ids
)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
```

---

#### NVIDIA Canary-1B-v2

**Architecture**: FastConformer encoder + Transformer decoder

**Specifications**:
- 1B parameters
- 25 European languages (including Greek)
- Up to 24 minutes audio with dynamic chunking
- Bidirectional English translation

**Performance**:
| Language | WER |
|----------|-----|
| English | 5.8% |
| German | 7.2% |
| Greek | ~8-12% (estimated) |

**License**: CC-BY-4.0 (commercial use allowed)

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")

# Transcribe Greek audio
transcription = model.transcribe(
    audio="greek_audio.wav",
    source_lang="el",
    target_lang="el"  # Same for transcription, different for translation
)
```

---

#### Meta MMS (Massively Multilingual Speech)

**Coverage**: 1,107 languages (including Greek)

**Architecture**: Wav2Vec2-based with CTC

**Performance**:
| Metric | MMS | Whisper |
|--------|-----|---------|
| Low-resource WER | 1x | 2x (higher) |
| Language coverage | 1,107 | 99 |
| Memory usage | ~20GB | ~10GB |

**Limitation**: Requires language specification (no auto-detection)

```python
from transformers import Wav2Vec2ForCTC, AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")

# Set target language to Greek
processor.tokenizer.set_target_lang("ell")
model.load_adapter("ell")

inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
```

---

#### XLS-R Models (Cross-lingual Speech Representations)

**Variants**:
| Model | Parameters | Pretraining Data |
|-------|------------|------------------|
| XLS-R-300M | 300M | 436K hours, 128 languages |
| XLS-R-1B | 1B | 436K hours, 128 languages |
| XLS-R-2B | 2B | 436K hours, 128 languages |

**Greek-Specific Results**:
- XLS-R-53 finetuned: **10.5% WER on Greek CommonVoice**
- Training time: 8 hours on single RTX 3080
- Works with as little as **10 minutes** of labeled data

**Why Choose XLS-R for Cypriot**:
1. Strong cross-lingual transfer from Greek
2. Few-shot learning capability
3. Adapter-based finetuning support
4. Open weights (Apache 2.0)

---

#### mHuBERT-147 (Multilingual HuBERT)

**Parameters**: 95M (10x smaller than competitors)

**Achievement**: #1 on ML-SUPERB 10-minute leaderboard

**Ideal Use Case**: Few-shot dialect adaptation with minimal data

---

### Optimized Inference Runtimes

#### faster-whisper (CTranslate2)
```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8"  # or float16, int8_float16
)

segments, info = model.transcribe(
    "audio.wav",
    language="el",
    beam_size=5,
    vad_filter=True
)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**Performance vs OpenAI Implementation**:
| Metric | OpenAI | faster-whisper |
|--------|--------|----------------|
| Speed | 1x | 4x |
| VRAM (large-v3) | 10GB | 4.5GB (int8) |
| Accuracy | Baseline | Identical |

---

#### whisper.cpp (CPU/Edge)
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make

# Download quantized model
./models/download-ggml-model.sh large-v3

# Transcribe
./main -m models/ggml-large-v3.bin -l el -f audio.wav
```

**Quantization Options**:
| Quantization | Model Size | Speed | Quality |
|--------------|------------|-------|---------|
| f16 | 3.09GB | 1x | 100% |
| q8_0 | 1.55GB | 1.5x | 99.5% |
| q5_1 | 1.08GB | 2x | 98% |
| q4_0 | 0.85GB | 2.5x | 96% |

---

## Metrics & Evaluation

### Primary ASR Metrics

#### Word Error Rate (WER)
The standard metric for ASR evaluation.

$$WER = \frac{S + D + I}{N} \times 100\%$$

Where:
- $S$ = Substitutions (wrong words)
- $D$ = Deletions (missing words)
- $I$ = Insertions (extra words)
- $N$ = Total words in reference

**Calculation Example**:
```
Reference: "Καλημέρα σας πώς είστε"
Hypothesis: "Καλημέρα σου πώς είσαι"

S=2 (σας→σου, είστε→είσαι), D=0, I=0, N=4
WER = (2+0+0)/4 = 50%
```

```python
import jiwer

reference = "Καλημέρα σας πώς είστε"
hypothesis = "Καλημέρα σου πώς είσαι"

wer = jiwer.wer(reference, hypothesis)
# Also available: mer, wil, wip
```

**Typical WER Ranges**:
| Quality | WER Range | Use Case |
|---------|-----------|----------|
| Excellent | <5% | Clean studio audio |
| Good | 5-10% | Broadcast quality |
| Acceptable | 10-20% | Conversational |
| Poor | >20% | Noisy/accented |

---

#### Character Error Rate (CER)
Better for morphologically rich languages like Greek.

$$CER = \frac{S_c + D_c + I_c}{N_c} \times 100\%$$

**When to use CER over WER**:
- Agglutinative/inflected languages (Greek)
- Languages without clear word boundaries
- Dialect evaluation with spelling variations

```python
cer = jiwer.cer(reference, hypothesis)
```

---

#### Additional Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MER** (Match Error Rate) | $(S+D+I)/(S+D+C)$ | Normalized by matches |
| **WIL** (Word Information Lost) | $1 - (C/N \times C/P)$ | Information theoretic |
| **WIP** (Word Information Preserved) | $C/N \times C/P$ | Complement of WIL |
| **SER** (Sentence Error Rate) | $E_{sent}/N_{sent}$ | Utterance-level |

---

### Translation Metrics

#### BLEU Score
For speech translation tasks (S2TT).

$$BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $BP$ = Brevity penalty
- $p_n$ = Modified n-gram precision
- $w_n$ = Weights (typically uniform 1/N)

```python
from sacrebleu import corpus_bleu

bleu = corpus_bleu(hypotheses, [references])
print(f"BLEU: {bleu.score:.1f}")
```

---

#### COMET Score
Neural metric correlating better with human judgment.

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [{"src": src, "mt": hyp, "ref": ref}]
scores = model.predict(data, batch_size=8)
```

---

### Audio Quality Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **MOS** (Mean Opinion Score) | 1-5 | Human quality rating |
| **PESQ** | -0.5 to 4.5 | Perceptual quality |
| **STOI** | 0-1 | Speech intelligibility |
| **SI-SDR** | dB | Source separation quality |

---

### Benchmark Datasets for Evaluation

| Dataset | Languages | Hours | Domain | Metrics |
|---------|-----------|-------|--------|---------|
| LibriSpeech | English | 960 | Audiobooks | WER |
| FLEURS | 102 | 12/lang | Read speech | WER, BLEU |
| CommonVoice | 100+ | Varies | Crowdsourced | WER |
| VoxPopuli | 16 EU | 1800 | Parliament | WER |
| CoVoST2 | 21→English | 2880 | Translation | BLEU |

**Greek-Specific Benchmarks**:
| Dataset | Size | WER Baseline |
|---------|------|--------------|
| CommonVoice Greek | 100+ hrs | 12-15% (Whisper-large) |
| Greek Podcast Corpus | 800 hrs | 8-12% (finetuned) |
| VoxPopuli Greek | 1.8K hrs | 10-14% |

---

## Loss Functions

### CTC Loss (Connectionist Temporal Classification)

Used by: Wav2Vec2, MMS, XLS-R, Conformer-CTC

**Mathematical Formulation**:

$$\mathcal{L}_{CTC} = -\log P(Y|X) = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi|X)$$

Where:
- $Y$ = Target label sequence
- $X$ = Input features
- $\pi$ = Alignment path
- $\mathcal{B}^{-1}$ = Inverse of collapsing function

**Key Properties**:
- No explicit alignment required
- Handles variable-length sequences
- Assumes conditional independence

```python
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# logits: (T, N, C) - time, batch, classes
# targets: (N, S) - batch, target length
# input_lengths: (N,) - actual input lengths
# target_lengths: (N,) - actual target lengths

loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

---

### Cross-Entropy Loss (Seq2Seq Models)

Used by: Whisper, SeamlessM4T decoder, LLM components

$$\mathcal{L}_{CE} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, X)$$

**With Label Smoothing**:
$$\mathcal{L}_{LS} = (1-\epsilon)\mathcal{L}_{CE} + \epsilon \cdot H(u)$$

Where $\epsilon$ is smoothing factor (typically 0.1) and $H(u)$ is uniform distribution entropy.

```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=0.1
)

# For Whisper-style models
logits = model(input_features, decoder_input_ids).logits
loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
```

---

### Contrastive Loss (Self-Supervised Pretraining)

Used by: Wav2Vec2, HuBERT, XLS-R pretraining

**InfoNCE Loss**:
$$\mathcal{L}_{NCE} = -\log \frac{\exp(sim(z_t, c_t)/\tau)}{\sum_{k} \exp(sim(z_t, c_k)/\tau)}$$

Where:
- $z_t$ = Contextualized representation
- $c_t$ = Quantized target
- $\tau$ = Temperature (typically 0.1)
- $k$ = Negative samples

---

### Multi-Task Learning Losses

For multi-modal models combining ASR, translation, and understanding:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{ASR} + \lambda_2 \mathcal{L}_{MT} + \lambda_3 \mathcal{L}_{LM}$$

**Typical Weights**:
| Task | Weight ($\lambda$) |
|------|-----|
| ASR (CTC/CE) | 1.0 |
| Translation | 0.5-1.0 |
| Language Modeling | 0.1-0.3 |

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, weights={'asr': 1.0, 'mt': 0.5, 'lm': 0.1}):
        super().__init__()
        self.weights = weights
        self.ctc = nn.CTCLoss(blank=0)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, outputs, targets):
        loss = 0
        if 'asr_logits' in outputs:
            loss += self.weights['asr'] * self.ctc(
                outputs['asr_logits'], 
                targets['asr_labels'],
                outputs['input_lengths'],
                targets['label_lengths']
            )
        if 'mt_logits' in outputs:
            loss += self.weights['mt'] * self.ce(
                outputs['mt_logits'].view(-1, outputs['mt_logits'].size(-1)),
                targets['mt_labels'].view(-1)
            )
        return loss
```

---

### Specialized Losses for Low-Resource Adaptation

#### Knowledge Distillation Loss
Transfer from large teacher to smaller student:

$$\mathcal{L}_{KD} = \alpha \cdot T^2 \cdot KL(p_s || p_t) + (1-\alpha) \cdot \mathcal{L}_{CE}$$

```python
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1)
    ) * (T * T)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

#### Focal Loss (Class Imbalance)
For rare phonemes/words in dialect data:

$$\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

---

## Finetuning Strategies

### Parameter-Efficient Finetuning (PEFT)

#### LoRA (Low-Rank Adaptation)

**Concept**: Decompose weight updates into low-rank matrices.

$$W' = W + \Delta W = W + BA$$

Where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$

**Recommended Configuration for ASR**:
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                          # Rank (8-64 typical)
    lora_alpha=32,                 # Scaling (2x rank recommended)
    target_modules=[
        "q_proj", "v_proj",        # Minimum
        "k_proj", "o_proj",        # Recommended
        "fc1", "fc2"               # For deeper adaptation
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# Output: trainable params: 6.5M || all params: 1.55B || trainable%: 0.42%
```

**LoRA Hyperparameter Guidelines**:
| Scenario | Rank (r) | Alpha | Target Modules |
|----------|----------|-------|----------------|
| Quick adaptation | 8 | 16 | q_proj, v_proj |
| Dialect finetuning | 16 | 32 | All attention |
| Domain shift | 32-64 | 64-128 | All + FFN |

---

#### QLoRA (Quantized LoRA)

Enables 7B model training on 24GB GPU.

```python
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    quantization_config=bnb_config,
    device_map="auto"
)

# Then apply LoRA on top
model = get_peft_model(model, lora_config)
```

---

#### Adapter Layers

Alternative to LoRA with similar efficiency.

```python
from transformers.adapters import AdapterConfig

adapter_config = AdapterConfig.load(
    "pfeiffer",
    reduction_factor=16
)

model.add_adapter("greek_dialect", config=adapter_config)
model.train_adapter("greek_dialect")
```

---

### Full Finetuning Strategy

For maximum adaptation with sufficient compute:

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-greek-cypriot",
    
    # Batch size
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 32
    
    # Learning rate
    learning_rate=1e-5,              # Lower for finetuning
    warmup_steps=500,
    
    # Schedule
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    
    # Optimization
    fp16=True,
    gradient_checkpointing=True,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    
    # Generation settings
    predict_with_generate=True,
    generation_max_length=225,
    
    # Logging
    logging_steps=50,
    report_to="wandb"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
```

---

### Transfer Learning Pipeline for Cypriot Greek

```
┌────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING PIPELINE                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Stage 1: Base Model                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Whisper-large-v3 (multilingual, 99 languages)           │   │
│  │ OR XLS-R-300M (128 languages, better few-shot)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  Stage 2: Greek Adaptation (Optional if data available)        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Continue pretraining on unlabeled Greek audio           │   │
│  │ Data: VoxPopuli Greek (17.7K hrs unlabeled)             │   │
│  │ Objective: Contrastive + MLM                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  Stage 3: Standard Greek Finetuning                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Supervised finetuning on labeled Greek                  │   │
│  │ Data: CommonVoice Greek + Greek Podcast Corpus          │   │
│  │ Method: Full finetuning or LoRA (r=16)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  Stage 4: Cypriot Dialect Adaptation                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Finetune on Cypriot Greek data                          │   │
│  │ Data: Voice of Cyprus + collected recordings            │   │
│  │ Method: LoRA (r=8-16) to preserve Greek knowledge       │   │
│  │ + LM fusion with Cypriot text                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### Data Augmentation for Low-Resource Scenarios

#### Audio Augmentation (SpecAugment)

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
])

# SpecAugment in training config
training_args = Seq2SeqTrainingArguments(
    # ... other args
    mask_time_prob=0.05,      # Time masking probability
    mask_time_length=10,       # Time mask length
    mask_feature_prob=0.0,     # Frequency masking (optional)
)
```

#### Synthetic Data Generation

Using TTS to expand Cypriot training data:

```python
# Generate Cypriot text with LLM
from transformers import pipeline

generator = pipeline("text-generation", model="meta-llama/Llama-3-8B")

prompt = """Generate 10 natural Cypriot Greek sentences about daily life.
Use Cypriot dialect features like:
- τζιαι instead of και
- έν instead of δεν
- που instead of ότι
Examples:"""

cypriot_sentences = generator(prompt, max_length=500)

# Synthesize with multilingual TTS
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
for sentence in cypriot_sentences:
    tts.tts_to_file(
        text=sentence,
        language="el",
        speaker_wav="cypriot_speaker.wav",  # Voice cloning
        file_path=f"synthetic_{i}.wav"
    )
```

**Reported Improvements**:
- LLM+TTS synthetic data: **14.3% absolute WER reduction** on extremely low-resource languages
- Voice cloning: **12.1% relative WER improvement** on in-domain data

---

## Data Resources

### Greek Speech Datasets

| Dataset | Hours | Type | Access | Link |
|---------|-------|------|--------|------|
| VoxPopuli Greek | 17.7K unlabeled / 1.8K labeled | Parliament | Free | github.com/facebookresearch/voxpopuli |
| CommonVoice Greek | 100+ | Crowdsourced | Free | commonvoice.mozilla.org |
| Greek Podcast Corpus | 800 | Podcasts (silver labels) | Research | Contact authors |
| CSS10 Greek | 24 | Single speaker TTS | Free | github.com/Kyubyong/css10 |

### Cypriot Greek Resources

| Resource | Size | Type | Status |
|----------|------|------|--------|
| Voice of Cyprus | ~300 hrs (growing) | Community collected | voiceofcyprus.org |
| Orientel Cypriot DB | 1000 speakers | Telephone (8kHz) | Academic request |
| YouTube Cypriot | Varies | Unlabeled | Requires scraping |

### Dataset Loading

```python
# VoxPopuli
from datasets import load_dataset

voxpopuli = load_dataset(
    "facebook/voxpopuli",
    "el",  # Greek
    split="train"
)

# CommonVoice
commonvoice = load_dataset(
    "mozilla-foundation/common_voice_16_0",
    "el",
    split="train",
    token=True  # Requires HF token
)

# Custom Cypriot dataset
from datasets import Dataset, Audio

def load_cypriot_data(audio_dir, transcript_file):
    # Load your collected data
    data = {"audio": [], "sentence": []}
    # ... populate data
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset
```

---

## Hardware & Deployment

### Hardware Requirements by Model Size

| Model Size | Min VRAM | Recommended | Quantized |
|------------|----------|-------------|-----------|
| <1B (Whisper) | 4GB | 8GB | 2GB (INT8) |
| 7-8B | 16GB | 24GB | 8GB (4-bit) |
| 13B | 32GB | 48GB | 16GB (4-bit) |
| 70B | 140GB | 160GB | 40GB (4-bit) |

### VRAM Estimation Formula

$$VRAM \approx P \times \frac{Q}{8} \times 1.2$$

Where:
- $P$ = Parameters in billions
- $Q$ = Precision bits (32, 16, 8, 4)
- 1.2 = Overhead factor

**Examples**:
- Whisper-large (1.55B) @ FP16: $1.55 \times 2 \times 1.2 = 3.7GB$
- Qwen2-Audio (8.2B) @ FP16: $8.2 \times 2 \times 1.2 = 19.7GB$
- Llama-3-70B @ INT4: $70 \times 0.5 \times 1.2 = 42GB$

### Apple Silicon Deployment

| Chip | Unified Memory | Max Model Size | Recommended Framework |
|------|----------------|----------------|----------------------|
| M2/M3 | 24GB | 13B (4-bit) | MLX |
| M2/M3 Pro | 36GB | 30B (4-bit) | MLX |
| M2/M3 Max | 128GB | 70B (4-bit) | MLX |
| M2/M3 Ultra | 192GB | 70B+ (FP16) | MLX |

```python
# MLX Whisper inference (Apple Silicon)
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    language="el"
)
```

### Quantization Comparison

| Method | Backend | Precision | Speed | Quality Loss |
|--------|---------|-----------|-------|--------------|
| BitsAndBytes | GPU | 4/8-bit | 1x | Minimal |
| GPTQ | GPU | 4-bit | 1.5x | <1% |
| AWQ | GPU | 4-bit | 2x | <0.5% |
| GGUF | CPU/Mac | 2-8 bit | Varies | Varies |
| CTranslate2 | GPU/CPU | INT8/FP16 | 4x | None |

### Deployment Frameworks

#### Ollama (Simplest)
```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Run model
ollama pull llama3.2
ollama run llama3.2
```

#### vLLM (Production)
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    quantization="awq"
)

outputs = llm.generate(prompts, SamplingParams(temperature=0.7))
```

#### llama.cpp (CPU/Edge)
```bash
# Build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Convert and quantize
python convert.py model/ --outtype f16
./quantize model/ggml-f16.gguf model/ggml-q4_k_m.gguf q4_k_m

# Run
./main -m model/ggml-q4_k_m.gguf -p "Translate to Greek:" -n 256
```

---

## Implementation Guide

### Complete Whisper Finetuning Pipeline

```python
"""
Complete pipeline for finetuning Whisper on Cypriot Greek
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
import evaluate

# 1. Load base model and processor
model_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# Set language and task
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="greek",
    task="transcribe"
)
model.config.suppress_tokens = []

# 2. Prepare dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    
    # Compute input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch

# Load and preprocess
dataset = load_dataset("your_cypriot_dataset")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

# 3. Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 4. Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 5. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-cypriot-greek",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=10,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=225,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# 6. Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# 7. Save model
trainer.save_model("./whisper-cypriot-greek-final")
processor.save_pretrained("./whisper-cypriot-greek-final")
```

### LoRA Finetuning Variant

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load quantized model
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Continue with training as above...
```

### Production Inference Pipeline

```python
"""
Production-ready inference with faster-whisper
"""

from faster_whisper import WhisperModel
import numpy as np
from typing import Generator, Tuple

class CypriotASR:
    def __init__(
        self,
        model_path: str = "whisper-cypriot-greek-final",
        device: str = "cuda",
        compute_type: str = "int8"
    ):
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "el",
        beam_size: int = 5,
        vad_filter: bool = True
    ) -> Generator[Tuple[float, float, str], None, None]:
        """
        Transcribe audio file with timestamps.
        
        Yields:
            (start_time, end_time, text) tuples
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            }
        )
        
        for segment in segments:
            yield (segment.start, segment.end, segment.text.strip())
    
    def transcribe_stream(
        self,
        audio_stream: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """Real-time transcription from audio buffer."""
        segments, _ = self.model.transcribe(
            audio_stream,
            language="el",
            beam_size=1,  # Faster for streaming
            without_timestamps=True
        )
        return " ".join(s.text for s in segments)

# Usage
asr = CypriotASR()

# File transcription
for start, end, text in asr.transcribe("cypriot_audio.wav"):
    print(f"[{start:.2f}s - {end:.2f}s]: {text}")

# Streaming (example with sounddevice)
import sounddevice as sd

def audio_callback(indata, frames, time, status):
    text = asr.transcribe_stream(indata.flatten())
    if text.strip():
        print(text, end=" ", flush=True)

with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
    print("Listening... Press Ctrl+C to stop")
    sd.sleep(60000)
```

---

## Benchmarks & Performance

### State-of-the-Art Performance Summary (December 2024)

#### ASR Benchmarks

| Model | LibriSpeech Clean | LibriSpeech Other | FLEURS (avg) | Greek WER |
|-------|-------------------|-------------------|--------------|-----------|
| Whisper-large-v3 | 1.8% | 3.5% | ~10% | 12-15% |
| Whisper-large-v3-turbo | 1.9% | 3.7% | ~10.5% | 12-16% |
| Qwen2-Audio | 1.83% | 3.47% | - | ~12% |
| NVIDIA Canary-1B-v2 | 5.8% | - | - | 8-12% |
| SeamlessM4T v2 | - | - | 45% lower | ~10% |
| MMS-1B | - | - | 50% lower | - |

#### Translation Benchmarks (BLEU)

| Model | En→De | En→Fr | En→El | El→En |
|-------|-------|-------|-------|-------|
| SeamlessM4T v2 | 31.2 | 38.5 | 28.1 | 32.4 |
| Whisper + NLLB | 28.7 | 35.2 | 25.3 | 29.8 |
| Qwen2-Audio | 29.4 | - | - | - |

#### Efficiency Benchmarks

| Model | RTF (GPU) | RTF (CPU) | Memory (FP16) | Memory (INT8) |
|-------|-----------|-----------|---------------|---------------|
| Whisper-large-v3 | 0.05x | 0.8x | 10GB | 5GB |
| faster-whisper large | 0.01x | 0.2x | 4.5GB | 2.9GB |
| whisper.cpp large | 0.03x | 0.3x | 3GB | 1.5GB |
| Canary-1B | 0.008x | 0.15x | 3GB | 1.5GB |

*RTF = Real-Time Factor (lower is faster)*

### Greek-Specific Results

From Greek Podcast Corpus (Interspeech 2024):

| Model | Domain-Specific WER | General WER |
|-------|---------------------|-------------|
| Whisper-large-v2 (base) | 15.2% | 12.8% |
| Whisper-medium (finetuned) | 9.4% | 10.1% |
| Whisper-small (finetuned) | 10.8% | 11.3% |

**Key Finding**: Finetuned smaller models outperform larger base models on Greek.

### Low-Resource Adaptation Results

| Starting Model | Target Language | Data Size | Final WER |
|----------------|-----------------|-----------|-----------|
| XLS-R-300M | Greek (CommonVoice) | 8 hrs | 10.5% |
| XLS-R-300M | Maltese | 7 hrs | 18.2% |
| MMS-1B | Various low-resource | 1 hr | 25-35% |
| mHuBERT-147 | ML-SUPERB langs | 10 min | 15-25% |

---

## References & Resources

### Key Papers (2024-2025)

1. **Qwen2-Audio Technical Report** (2024)
   - arxiv.org/abs/2407.10759

2. **SeamlessM4T: Massively Multilingual & Multimodal Translation** (2023-2024)
   - arxiv.org/abs/2308.11596
   - nature.com/articles/s41586-024-08359-z

3. **The Greek Podcast Corpus** (Interspeech 2024)
   - isca-archive.org/interspeech_2024/paraskevopoulos24_interspeech.pdf

4. **Scaling Speech Technology to 1000+ Languages (MMS)** (2023)
   - arxiv.org/abs/2305.13516

5. **Practical Tips for Finetuning LLMs Using LoRA** (2024)
   - Sebastian Raschka's Magazine

### Model Repositories

| Model | Repository |
|-------|------------|
| Whisper | github.com/openai/whisper |
| faster-whisper | github.com/SYSTRAN/faster-whisper |
| whisper.cpp | github.com/ggerganov/whisper.cpp |
| Qwen2-Audio | github.com/QwenLM/Qwen2-Audio |
| SeamlessM4T | github.com/facebookresearch/seamless_communication |
| SALMONN | github.com/bytedance/SALMONN |
| LLaMA-Omni | github.com/ictnlp/LLaMA-Omni |
| MMS | huggingface.co/facebook/mms-1b-all |
| XLS-R | huggingface.co/facebook/wav2vec2-xls-r-300m |

### Tools & Frameworks

| Tool | Purpose | Link |
|------|---------|------|
| HuggingFace Transformers | Model hub & training | huggingface.co |
| PEFT | Parameter-efficient finetuning | github.com/huggingface/peft |
| Ollama | Local LLM deployment | ollama.com |
| vLLM | Production inference | github.com/vllm-project/vllm |
| MLX | Apple Silicon inference | github.com/ml-explore/mlx |
| NeMo | NVIDIA ASR toolkit | github.com/NVIDIA/NeMo |

### Datasets

| Dataset | Link |
|---------|------|
| CommonVoice | commonvoice.mozilla.org |
| VoxPopuli | github.com/facebookresearch/voxpopuli |
| FLEURS | huggingface.co/datasets/google/fleurs |
| Voice of Cyprus | voiceofcyprus.org |

---

## Appendix: Quick Reference

### Model Selection Decision Tree

```
Need multi-modal (audio + text understanding)?
├── Yes → Qwen2-Audio (8.2B) or SALMONN (7B)
└── No → ASR only?
    ├── Yes → Greek native support needed?
    │   ├── Yes → NVIDIA Canary-1B-v2 (commercial OK)
    │   │         OR Whisper-large-v3 (MIT)
    │   └── No → Language coverage needed?
    │       ├── High (1000+) → MMS-1B
    │       └── Standard → Whisper-large-v3
    └── No → Translation needed?
        ├── Yes → SeamlessM4T v2
        └── No → Real-time interaction?
            ├── Yes → LLaMA-Omni or Ultravox
            └── No → Standard pipeline

Few-shot (<1hr data)?
├── Yes → XLS-R-300M or mHuBERT-147
└── No → Full Whisper finetuning
```

### Recommended Stack for Cypriot Greek

```
┌─────────────────────────────────────────────────┐
│              RECOMMENDED STACK                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ASR: Whisper-large-v3 + LoRA finetuning        │
│  Runtime: faster-whisper (GPU) / whisper.cpp    │
│  LLM: Qwen-2-7B or Llama-3-8B                   │
│  Framework: HuggingFace + PEFT                  │
│  Quantization: INT8 (inference) / QLoRA (train) │
│  Deployment: Ollama (simple) / vLLM (prod)      │
│                                                 │
│  Minimum Hardware:                              │
│  - Training: RTX 4090 (24GB) or cloud A100      │
│  - Inference: RTX 3060 (12GB) or M2 Mac         │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

*Last updated: December 2025*
*Guide version: 1.0*