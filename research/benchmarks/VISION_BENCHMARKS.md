# Vision Model Benchmarks

Reference targets for the dl_techniques vision experiments (classification, detection, segmentation, depth, multimodal, video).
Numbers are from public leaderboards (Papers With Code, MMBench OpenCompass, MMMU, OCRBench v2), original papers, and model release blogs; cite the date you pulled them.

> **Snapshot date**: 2026-05-13

## What These Benchmarks Measure

- **ImageNet-1K** (Russakovsky et al., 2015, arXiv:1409.0575). 1.28M training images / 50K val / 100K test, 1000 classes, 224x224 default. The headline metric is top-1 accuracy on the val split. ImageNet-21K (a.k.a. ImageNet-22K, ~14M images, ~21841 classes) is the standard large-scale pretraining set; numbers in the "IN-21K pretrain" column indicate fine-tuned accuracy after IN-21K pre-training.
- **COCO** (Lin et al., 2014, arXiv:1405.0312). 118K train / 5K val (val2017) / 20K test-dev, 80 thing classes. Detection reports box mAP@[0.50:0.05:0.95] ("mAP@50:95"); instance segmentation reports mask AP under the same protocol. Panoptic segmentation reports PQ.
- **ADE20K** (Zhou et al., 2017). 20K train / 2K val, 150 semantic classes. Reports mIoU.
- **Cityscapes** (Cordts et al., 2016). 2975 train / 500 val / 1525 test, 19 classes, urban scenes at 1024x2048. Reports mIoU.
- **NYUv2** (Silberman et al., 2012) and **KITTI** (Geiger et al., 2012). Indoor and driving depth benchmarks. AbsRel = mean(|d_pred - d_gt| / d_gt); delta_1 = % pixels with max(d_pred/d_gt, d_gt/d_pred) < 1.25.
- **MMMU** (Yue et al., 2023, arXiv:2311.16502). 11.5K college-level multimodal questions across 6 disciplines. Reports accuracy (val split, multiple choice + open-ended).
- **MMBench** (Liu et al., 2023, arXiv:2307.06281). 3K+ multiple-choice multimodal questions with CircularEval rotation. "MMBench-EN" dev set accuracy is the headline.
- **DocVQA** (Mathew et al., 2021, arXiv:2007.00398), **ChartQA** (Masry et al., 2022), **OCRBench / OCRBench v2** (Liu et al., arXiv:2305.07895 / arXiv:2501.00321) measure document, chart, and scene-text VQA accuracy.
- **Kinetics-400/600** (Kay et al., 2017) and **Something-Something v2** (Goyal et al., 2017) are the two pillars of video action recognition. K-400 = appearance-driven (400 action classes, 240K clips); SSv2 = motion/temporal-reasoning-driven (174 fine-grained interactions). Both report top-1 accuracy on val.

## How to Read These Numbers

- **Pretrain budget dwarfs architecture.** A ViT-L pretrained on JFT-3B (CoAtNet recipe) or on DINOv2-curated 142M images beats the same architecture trained from scratch on IN-1K by 4-8 points. When comparing classifiers, always check the pretrain column before drawing architectural conclusions.
- **Resolution matters at the top.** EVA-02-L hits 90.0 at 448x448 but ~88 at 224; ConvNeXt V2-H hits 88.9 at 512. A 1-2 point gap between models reported at different resolutions is mostly resolution, not architecture.
- **Self-supervised pretraining (MAE, FCMAE, DINOv2, EVA, BEiT) is now the default at scale.** Supervised IN-21K pretraining (the original ViT / Swin recipe) is still competitive at base scale but is consistently beaten at L/H scale by SSL recipes.
- **Linear probe vs fine-tune.** DINOv2 numbers are linear probe (features frozen); ConvNeXt V2 / EVA-02 are full fine-tune. Linear probe is the honest "how good are the features" signal; fine-tune is the honest "how good can this model get on this task" signal.
- **Contamination is a real problem for VLMs.** LLaVA-style instruction-tuning datasets (LLaVA-Instruct-150K, ShareGPT4V, the Sphinx mix, etc.) overlap non-trivially with MMMU, ChartQA, DocVQA, and OCRBench train/test splits. Numbers from models that train on these mixes are not directly comparable to numbers from frontier API models that explicitly held them out. Treat 5-point gaps with appropriate skepticism.
- **Self-reported vs leaderboard-evaluated.** OpenCompass MMBench, MMMU leaderboard, and Papers With Code re-run a fixed prompt; model cards often quote a tuned prompt or chain-of-thought variant that adds 2-5 points. Cite the source you used.
- **FPS / throughput numbers are hardware-dependent.** Most YOLO numbers are T4-TensorRT-FP16; ConvNeXt / ViT throughput is A100-PyTorch-FP16. Do not cross-compare without normalizing.

## Image Classification (ImageNet-1K top-1)

### Tiny (<=10M params)

| Model | Params | Resolution | IN-1K top-1 | IN-21K pretrain | Throughput hint | Year | License |
|-------|--------|------------|-------------|-----------------|-----------------|------|---------|
| MobileNetV3-Large | 5.4M | 224 | 75.2 | No | ~3.3ms iPhone | 2019 | Apache-2.0 |
| MobileNetV4-Conv-S | 3.8M | 224 | 73.8 | No | ~0.6ms Pixel 8 | 2024 | Apache-2.0 |
| MobileNetV4-Conv-M | 9.7M | 256 | 79.9 | No | ~1.6ms Pixel 8 | 2024 | Apache-2.0 |
| FastViT-T8 | 3.6M | 256 | 75.6 | No | 0.8ms iPhone 12 | 2023 | Apple AML |
| RepViT-M0.9 | 5.1M | 224 | 78.7 | No | 0.9ms iPhone 12 | 2023 | Apache-2.0 |
| ConvNeXt V2-Atto (FCMAE) | 3.7M | 224 | 76.7 | No (FCMAE) | ? | 2023 | CC-BY-NC-4.0 |
| ConvNeXt V2-Femto (FCMAE) | 5.2M | 224 | 78.5 | No (FCMAE) | ? | 2023 | CC-BY-NC-4.0 |
| ConvNeXt V2-Pico (FCMAE) | 9.1M | 224 | 80.3 | No (FCMAE) | ? | 2023 | CC-BY-NC-4.0 |
| EfficientNet-B0 | 5.3M | 224 | 77.1 | No | ~0.39 GFLOPs | 2019 | Apache-2.0 |

### Small (10-30M)

| Model | Params | Resolution | IN-1K top-1 | IN-21K pretrain | Throughput hint | Year | License |
|-------|--------|------------|-------------|-----------------|-----------------|------|---------|
| ResNet-50 | 25.6M | 224 | 76.1 (orig) / 80.4 (A1 recipe) | No | 4.1 GFLOPs | 2015 / 2021 | MIT |
| EfficientNet-B3 | 12M | 300 | 81.6 | No | 1.8 GFLOPs | 2019 | Apache-2.0 |
| DeiT III-S | 22M | 224 | 81.4 | No | 4.6 GFLOPs | 2022 | Apache-2.0 |
| Swin-T | 28M | 224 | 81.3 | No | 4.5 GFLOPs | 2021 | MIT |
| ConvNeXt-T | 29M | 224 | 82.1 | 82.9 | 4.5 GFLOPs | 2022 | MIT |
| ConvNeXt V2-Nano (FCMAE) | 15.6M | 224 | 81.9 | 82.1 | ? | 2023 | CC-BY-NC-4.0 |
| ConvNeXt V2-Tiny (FCMAE) | 28.6M | 224 | 83.0 | 83.9 | ? | 2023 | CC-BY-NC-4.0 |
| MaxViT-T | 31M | 224 | 83.6 | ? | 5.6 GFLOPs | 2022 | Apache-2.0 |
| RepViT-M2.3 | 23M | 224 | 83.7 | No | 2.3ms iPhone 12 | 2024 | Apache-2.0 |

### Base (30-100M)

| Model | Params | Resolution | IN-1K top-1 | IN-21K pretrain | Throughput hint | Year | License |
|-------|--------|------------|-------------|-----------------|-----------------|------|---------|
| ViT-B/16 (orig) | 86M | 224 | 77.9 | 84.0 | 17.6 GFLOPs | 2020 | Apache-2.0 |
| DeiT III-B | 86M | 224 | 83.8 | 86.7 | 17.6 GFLOPs | 2022 | Apache-2.0 |
| Swin-B | 88M | 224 | 83.5 | 85.2 | 15.4 GFLOPs | 2021 | MIT |
| Swin V2-B | 88M | 256 | 84.6 | 87.1 | ? | 2022 | MIT |
| ConvNeXt-B | 89M | 224 | 83.8 | 85.8 | 15.4 GFLOPs | 2022 | MIT |
| ConvNeXt V2-Base (FCMAE+IN-21K) | 89M | 384 | 87.7 | 87.7 | ? | 2023 | CC-BY-NC-4.0 |
| MAE ViT-B | 86M | 224 | 83.6 | No (MAE only) | 17.6 GFLOPs | 2022 | CC-BY-NC-4.0 |
| BEiT v2-B | 86M | 224 | 85.5 | No (MIM only) | 17.6 GFLOPs | 2022 | MIT |
| CoAtNet-1 | 42M | 224 | 83.3 | 85.1 | 8.4 GFLOPs | 2021 | Apache-2.0 |
| MaxViT-B | 120M | 224 | 84.9 | 88.4 | 23.4 GFLOPs | 2022 | Apache-2.0 |
| EfficientNet-B7 | 66M | 600 | 84.3 | No | 37 GFLOPs | 2019 | Apache-2.0 |
| DINOv2 ViT-B/14 (linear) | 86M | 224 | 84.5 | No (LVD-142M) | 17.6 GFLOPs | 2023 | Apache-2.0 |

### Large (100-350M)

| Model | Params | Resolution | IN-1K top-1 | IN-21K pretrain | Throughput hint | Year | License |
|-------|--------|------------|-------------|-----------------|-----------------|------|---------|
| ViT-L/16 | 304M | 224 | 76.5 | 85.2 | 61 GFLOPs | 2020 | Apache-2.0 |
| Swin-L | 197M | 384 | 87.3 | 87.3 | 104 GFLOPs | 2021 | MIT |
| Swin V2-L | 197M | 384 | 87.7 | 87.7 | ? | 2022 | MIT |
| ConvNeXt-L | 198M | 384 | 85.5 | 87.5 | 101 GFLOPs | 2022 | MIT |
| ConvNeXt V2-Large (FCMAE+IN-21K) | 198M | 384 | 88.2 | 88.2 | ? | 2023 | CC-BY-NC-4.0 |
| ConvNeXt V2-Large (FCMAE+IN-21K) | 198M | 512 | 88.6 | 88.6 | ? | 2023 | CC-BY-NC-4.0 |
| MAE ViT-L | 304M | 224 | 85.9 | No (MAE only) | 61 GFLOPs | 2022 | CC-BY-NC-4.0 |
| BEiT v2-L | 304M | 224 | 87.3 | No (MIM only) | 61 GFLOPs | 2022 | MIT |
| MaxViT-L | 212M | 512 | 86.7 | 88.7 | ? | 2022 | Apache-2.0 |
| CoAtNet-3 | 168M | 384 | 85.8 | 88.5 | 49 GFLOPs | 2021 | Apache-2.0 |
| EVA-02-L (IN-21K -> IN-1K) | 304M | 448 | 90.0 | Merged-38M + IN-21K | ? | 2023 | MIT |
| DINOv2 ViT-L/14 (linear) | 304M | 224 | 86.3 | No (LVD-142M) | 61 GFLOPs | 2023 | Apache-2.0 |
| DeiT III-L | 304M | 384 | 87.7 | 87.7 | ? | 2022 | Apache-2.0 |

### Huge / Giant (350M+)

| Model | Params | Resolution | IN-1K top-1 | IN-21K pretrain | Throughput hint | Year | License |
|-------|--------|------------|-------------|-----------------|-----------------|------|---------|
| ViT-H/14 | 632M | 224 | 77.9 | 85.1 | ? | 2020 | Apache-2.0 |
| ConvNeXt-XL | 350M | 384 | 85.5 | 87.8 | ? | 2022 | MIT |
| ConvNeXt V2-Huge (FCMAE+IN-21K) | 659M | 384 | 88.7 | 88.7 | ? | 2023 | CC-BY-NC-4.0 |
| ConvNeXt V2-Huge (FCMAE+IN-21K) | 659M | 512 | 88.9 | 88.9 | ? | 2023 | CC-BY-NC-4.0 |
| MAE ViT-H | 632M | 224 | 86.9 | No (MAE only) | ? | 2022 | CC-BY-NC-4.0 |
| BEiT v2-L (IN-21K) | 304M | 512 | 88.6 | 88.6 | ? | 2022 | MIT |
| Swin V2-G | 3.0B | 640 | 90.2 (test set) | IN-22K-ext-70M | ? | 2022 | MIT |
| CoAtNet-7 (JFT-3B) | 2.4B | 512 | 90.88 | JFT-3B | ? | 2021 | Apache-2.0 |
| DINOv2 ViT-g/14 (linear) | 1.1B | 224 | 86.5 | No (LVD-142M) | ? | 2023 | Apache-2.0 |
| EVA-02-CLIP-E/14+ (zero-shot) | 5.0B | 224 | 82.0 (0-shot) | LAION + Merged-2B | ? | 2023 | MIT |

## Object Detection (COCO val2017 box AP)

| Model | Params | Resolution | mAP@50:95 | FPS hint (T4) | Year | License |
|-------|--------|------------|-----------|---------------|------|---------|
| Mask R-CNN (R50-FPN) | 44M | 800 | 41.0 | 17 | 2017 | Apache-2.0 |
| EfficientDet-D7 | 52M | 1536 | 53.7 | 6 | 2020 | Apache-2.0 |
| DETR (R50) | 41M | 800 | 42.0 | 28 | 2020 | Apache-2.0 |
| DINO (Swin-L) | 218M | 1280 | 63.3 (test-dev) | ? | 2022 | Apache-2.0 |
| Co-DETR (Swin-L) | 218M | 1280 | 64.5 (test-dev) | ? | 2023 | MIT |
| Co-DETR (ViT-L, Objects365) | ~300M | 1280 | 66.0 (test-dev) | ? | 2023 | MIT |
| YOLOv8-n | 3.2M | 640 | 37.3 | 280 | 2023 | AGPL-3.0 |
| YOLOv8-x | 68.2M | 640 | 53.9 | 80 | 2023 | AGPL-3.0 |
| YOLOv9-c | 25.3M | 640 | 53.0 | 100 | 2024 | GPL-3.0 |
| YOLOv9-e | 57.3M | 640 | 55.6 | 60 | 2024 | GPL-3.0 |
| YOLOv10-n | 2.3M | 640 | 38.5 | 385 | 2024 | AGPL-3.0 |
| YOLOv10-s | 7.2M | 640 | 46.3 | 240 | 2024 | AGPL-3.0 |
| YOLOv10-x | 29.5M | 640 | 54.4 | 70 | 2024 | AGPL-3.0 |
| YOLOv11-n | 2.6M | 640 | 39.5 | 320 | 2024 | AGPL-3.0 |
| YOLOv11-x | 56.9M | 640 | 54.7 | 75 | 2024 | AGPL-3.0 |
| YOLOv12-n | 2.6M | 640 | 40.6 | 290 | 2025 | AGPL-3.0 |
| YOLOv12-x | 59.1M | 640 | 55.2 | ~80 | 2025 | AGPL-3.0 |
| RT-DETR-L | 32M | 640 | 53.0 | 114 | 2024 | Apache-2.0 |
| RT-DETR-X | 67M | 640 | 54.8 | 74 | 2024 | Apache-2.0 |
| RT-DETRv4-L | ~32M | 640 | 55.4 | 124 | 2025 | Apache-2.0 |
| RT-DETRv4-X | ~67M | 640 | 57.0 | ~70 | 2025 | Apache-2.0 |
| RF-DETR (2x-large) | ~130M | 640 | 60.5 | ~25 | 2026 | Apache-2.0 |
| Grounding DINO (Swin-L, zero-shot) | 218M | 1333 | 52.5 (0-shot) | ? | 2023 | Apache-2.0 |
| Grounding DINO (Swin-L, fine-tune) | 218M | 1333 | 63.0 | ? | 2023 | Apache-2.0 |

## Semantic Segmentation (ADE20K / Cityscapes mIoU)

| Model | Backbone | Params | ADE20K mIoU | Cityscapes mIoU | Year | License |
|-------|----------|--------|-------------|-----------------|------|---------|
| SegFormer-B0 | MiT-B0 | 3.8M | 37.4 | 76.2 | 2021 | Apache-2.0 |
| SegFormer-B2 | MiT-B2 | 27.5M | 46.5 | 81.0 | 2021 | Apache-2.0 |
| SegFormer-B4 | MiT-B4 | 64M | 50.3 | 83.8 | 2021 | Apache-2.0 |
| SegFormer-B5 | MiT-B5 | 84.7M | 51.8 | 84.0 | 2021 | Apache-2.0 |
| Mask2Former | Swin-B | 107M | 53.9 | 83.3 | 2022 | MIT |
| Mask2Former | Swin-L | 215M | 57.3 | 84.5 | 2022 | MIT |
| OneFormer | Swin-L | 219M | 57.7 | 84.6 | 2023 | MIT |
| OneFormer | DiNAT-L | 223M | 58.3 | 84.0 | 2023 | MIT |
| OneFormer | ConvNeXt-XL | 372M | 58.8 | 84.6 | 2023 | MIT |
| Mask DINO | Swin-L | 223M | 60.8 | ? | 2023 | Apache-2.0 |
| EVA-02 + Mask2Former | EVA-02-L | 304M | 62.0 | ? | 2023 | MIT |
| ONE-PEACE + Mask2Former | ONE-PEACE | 1.5B | 63.0 | ? | 2023 | Apache-2.0 |
| InternImage-H + Mask2Former | InternImage-H | 1.1B | 62.9 | 86.1 | 2023 | MIT |

## Instance Segmentation (COCO val2017 mask AP)

| Model | Backbone | Params | mask AP | box AP | Year | License |
|-------|----------|--------|---------|--------|------|---------|
| Mask R-CNN | R50-FPN | 44M | 37.5 | 41.0 | 2017 | Apache-2.0 |
| Mask2Former | R50 | 44M | 43.7 | ? | 2022 | MIT |
| Mask2Former | Swin-L | 216M | 50.1 | ? | 2022 | MIT |
| Mask DINO | Swin-L | 223M | 54.5 | 59.0 | 2023 | Apache-2.0 |
| OneFormer | Swin-L | 219M | 49.2 | ? | 2023 | MIT |
| SAM (ViT-H, 1-click) | ViT-H | 636M | 58.1 mIoU | n/a | 2023 | Apache-2.0 |
| SAM 2 (Hiera-L, 1-click) | Hiera-L | 224M | 58.9 mIoU | n/a | 2024 | Apache-2.0 |
| SAM 2 (Hiera-L, 5-click, 37 datasets) | Hiera-L | 224M | 81.7 mIoU | n/a | 2024 | Apache-2.0 |

## Depth Estimation (NYUv2 / KITTI)

| Model | Params | NYU AbsRel | NYU delta_1 | KITTI AbsRel | KITTI delta_1 | Year | License |
|-------|--------|------------|-------------|--------------|---------------|------|---------|
| DPT-Large (MiDaS) | 343M | 0.110 | 0.904 | ? | ? | 2021 | MIT |
| MiDaS v3.1 (ViT-L) | 345M | ? | ? | 0.127 | 0.850 | 2023 | MIT |
| ZoeDepth (NYU+KITTI) | 345M | 0.075 | 0.955 | 0.057 | 0.971 | 2023 | MIT |
| Marigold v1.1 | ~865M (SD2) | 0.055 | 0.964 | 0.099 | 0.916 | 2024 | Apache-2.0 |
| Depth Anything V1 (ViT-L) | 335M | 0.056 | 0.984 | 0.076 | 0.947 | 2024 | Apache-2.0 |
| Depth Anything V2 (ViT-S) | 25M | 0.053 | 0.971 | ? | ? | 2024 | Apache-2.0 |
| Depth Anything V2 (ViT-B) | 97M | 0.052 | 0.972 | ? | ? | 2024 | Apache-2.0 |
| Depth Anything V2 (ViT-L) | 335M | 0.045 | 0.979 | 0.074 | 0.946 | 2024 | Apache-2.0 |
| Metric3D v2 (ViT-L) | 335M | 0.058 | 0.974 | 0.058 | 0.974 | 2024 | BSD-2 |
| Depth Anything 3 | ? | ? | ? | ? | ? | 2025 | Apache-2.0 |

## Vision-Language / Multimodal

Scores are from the official MMMU leaderboard, OpenCompass MMBench leaderboard, and individual model technical reports. MMMU is the val split (5-shot or 0-shot as reported by the model); MMBench is dev-EN.

### Open-weights small (<5B total)

| Model | LLM | Vision params | MMMU | MMBench | DocVQA | ChartQA | Year | License |
|-------|-----|---------------|------|---------|--------|---------|------|---------|
| MiniCPM-V 2.6 | Qwen2-7B | 0.4B | 49.8 | 81.5 | 90.8 | ? | 2024 | Apache-2.0 |
| Phi-3.5-vision | Phi-3.5-mini-3.8B | 0.4B | 43.0 | 81.9 | 75.9 | 81.8 | 2024 | MIT |
| Phi-4-multimodal | Phi-4-mini-3.8B | 0.4B | 55.1 | 86.7 | 93.2 | ? | 2025 | MIT |
| Idefics3-8B | Llama-3-8B | 0.4B (SigLIP-SO) | 46.6 | 76.4 | 87.7 | 74.8 | 2024 | Apache-2.0 |
| Molmo-7B-D | Qwen2-7B | 0.3B (OpenAI CLIP) | 45.3 | 73.6 | 92.2 | 84.1 | 2024 | Apache-2.0 |
| LLaVA-OneVision-7B | Qwen2-7B | 0.4B (SigLIP) | 48.8 | 80.8 | 87.5 | 80.0 | 2024 | Apache-2.0 |
| InternVL 2.5-8B | InternLM2.5-7B | 0.3B (InternViT-300M) | 56.0 | 84.6 | 93.0 | 84.8 | 2024 | MIT |
| Qwen2-VL-7B | Qwen2-7B | 0.6B (NaViT-style) | 54.1 | 83.0 | 94.5 | 83.0 | 2024 | Apache-2.0 |
| Qwen2.5-VL-7B | Qwen2.5-7B | 0.6B | 58.6 | 83.5 | 95.7 | 87.3 | 2025 | Apache-2.0 |

### Open-weights large (>=10B total)

| Model | LLM | Vision params | MMMU | MMBench | DocVQA | ChartQA | Year | License |
|-------|-----|---------------|------|---------|--------|---------|------|---------|
| LLaVA-NeXT-34B | Yi-34B | 0.3B | 51.1 | 79.3 | 84.0 | 68.7 | 2024 | Apache-2.0 |
| InternVL 2-26B | InternLM2-20B | 6B (InternViT-6B) | 51.2 | 83.4 | 92.9 | 84.9 | 2024 | MIT |
| InternVL 2.5-26B | InternLM2.5-20B | 6B (InternViT-6B) | 60.0 | 85.4 | 94.0 | 87.2 | 2024 | MIT |
| InternVL 2.5-78B | Qwen2.5-72B | 6B | 70.1 | 88.3 | 95.1 | 88.3 | 2024 | MIT |
| Qwen2-VL-72B | Qwen2-72B | 0.6B | 64.5 | 86.5 | 96.5 | 88.3 | 2024 | Qwen license |
| Qwen2.5-VL-72B | Qwen2.5-72B | 0.6B | 70.2 | 88.6 | 96.4 | 89.5 | 2025 | Qwen license |
| Molmo-72B | Qwen2-72B | 0.3B | 54.1 | 81.2 | 93.5 | 87.3 | 2024 | Apache-2.0 |

### Proprietary frontier

| Model | LLM | Vision params | MMMU | MMBench | DocVQA | ChartQA | Year | License |
|-------|-----|---------------|------|---------|--------|---------|------|---------|
| GPT-4o (2024-08) | proprietary | n/a | 69.1 | 83.4 | 92.8 | 85.7 | 2024 | Proprietary |
| GPT-4.1 | proprietary | n/a | 74.8 | ? | ? | ? | 2025 | Proprietary |
| GPT-5 | proprietary | n/a | 84.2 | ? | ? | ? | 2025 | Proprietary |
| Claude 3.5 Sonnet | proprietary | n/a | 68.3 | 82.6 | 95.2 | 90.8 | 2024 | Proprietary |
| Claude 4 Sonnet | proprietary | n/a | 85.4 | ? | ? | ? | 2025 | Proprietary |
| Gemini 1.5 Pro | proprietary | n/a | 62.2 | 73.9 | 93.1 | 87.2 | 2024 | Proprietary |
| Gemini 2.0 Pro | proprietary | n/a | 72.7 | ? | ? | ? | 2025 | Proprietary |
| Gemini 2.5 Pro | proprietary | n/a | 84.0 | ? | ? | ? | 2025 | Proprietary |

### Zero-shot ImageNet (CLIP-family encoders, no fine-tune)

| Model | Image encoder | Params | IN-1K 0-shot | Pretrain | Year | License |
|-------|---------------|--------|--------------|----------|------|---------|
| OpenAI CLIP ViT-L/14 | ViT-L | 304M | 75.5 | WIT-400M | 2021 | MIT |
| OpenCLIP ViT-G/14 | ViT-G | 1.8B | 80.1 | LAION-2B | 2022 | MIT |
| SigLIP SO400M/14 | SoViT-400M | 400M | 83.2 | WebLI-10B | 2023 | Apache-2.0 |
| SigLIP 2 SO400M/14 | SoViT-400M | 400M | 84.1 | WebLI-multi | 2025 | Apache-2.0 |
| SigLIP 2 g/16 | ViT-g | 1.1B | 85.6 | WebLI-multi | 2025 | Apache-2.0 |
| EVA-02-CLIP-L/14+ | ViT-L | 430M | 80.4 | Merged-2B | 2023 | MIT |
| EVA-02-CLIP-E/14+ | ViT-E | 5.0B | 82.0 | LAION + Merged-2B | 2023 | MIT |
| Meta CLIP 2 ViT-H | ViT-H | 1.0B | 82.1 | MetaCLIP-Worldwide-29B | 2025 | MIT |
| DFN5B CLIP ViT-H | ViT-H | 1.0B | 84.4 | DFN-5B | 2024 | Apple AML |

## Video Understanding

| Model | Pretrain | Params | K-400 top-1 | SSv2 top-1 | Year | License |
|-------|----------|--------|-------------|------------|------|---------|
| VideoMAE-B | K-400 (SSL) | 87M | 81.5 | 70.8 | 2022 | CC-BY-NC-4.0 |
| VideoMAE-L | K-400 (SSL) | 305M | 85.2 | 74.3 | 2022 | CC-BY-NC-4.0 |
| VideoMAE-H | K-400 (SSL) | 633M | 87.4 | 75.4 | 2022 | CC-BY-NC-4.0 |
| VideoMAE V2-g | UnlabeledHybrid-1.35M | 1.0B | 90.0 | 77.0 | 2023 | CC-BY-NC-4.0 |
| InternVideo (1B) | mixed gen+disc | 1.0B | 91.1 | 77.2 | 2022 | Apache-2.0 |
| InternVideo2-6B | mixed | 6B | 92.1 | 77.5 | 2024 | Apache-2.0 |
| V-JEPA ViT-L | VideoMix-2M (SSL) | 304M | 81.9 | 72.2 | 2024 | CC-BY-NC-4.0 |
| V-JEPA ViT-H | VideoMix-2M (SSL) | 632M | 82.0 | 73.6 | 2024 | CC-BY-NC-4.0 |
| V-JEPA 2 ViT-L | VideoMix-22M (SSL) | 304M | 83.7 | 77.3 | 2025 | CC-BY-NC-4.0 |
| V-JEPA 2 ViT-g | VideoMix-22M (SSL) | 1.0B | 84.6 | 77.3 | 2025 | CC-BY-NC-4.0 |
| Hiera-H (MAE, K-400) | K-400 SSL | 673M | 87.3 | 75.1 | 2023 | Apache-2.0 |

## 2026 SOTA Themes Captured in the Numbers

- **VLM dominance and the death of pure-vision benchmarks for academia.** ImageNet top-1 has been pinned in the 88-91 range since 2022; the marginal gains from each new architecture (ConvNeXt V2, EVA-02, Swin V2-G) are now smaller than the resolution/pretrain noise floor. Frontier research has decisively shifted to MMMU, MMBench, OCRBench, and chart/document VQA, where the gap between the best open model (Qwen2.5-VL-72B, InternVL 2.5-78B at ~70 MMMU) and the best proprietary model (Claude 4 Sonnet at 85.4, Gemini 2.5 Pro at 84.0, GPT-5 at 84.2) is still 13-15 points and actively contested.
- **SAM and DINOv2 as universal backbones.** DINOv2 ViT-L features (linear-probe 86.3 IN-1K) are the default open feature extractor for downstream tasks in 2024-2026; SAM 2 with Hiera is the default mask producer. The "what backbone do I use for X" question has collapsed into "which of these two, and what head?" for the open-source stack.
- **The rise of Depth Anything.** Depth Anything V2 (0.045 AbsRel on NYU at 335M params) closed the gap with diffusion-based depth (Marigold) while being 10-100x cheaper at inference. Combined with Metric3D v2 for metric output, monocular depth is effectively a solved benchmark problem; the open research question is multi-view / 3D consistency (Depth Anything 3).
- **Detection has bifurcated into "YOLO-line" and "DETR-line".** YOLOv12 (55.2 mAP at 59M / ~80 FPS) and RT-DETRv4 (57.0 mAP at 67M / ~70 FPS) are now within 2 mAP of each other at comparable cost; the YOLO line wins on edge / pure throughput, the DETR line wins on tail classes and end-to-end NMS-free deployment. Co-DETR / RF-DETR push the absolute SOTA to ~60-66 mAP at the cost of 5-10x slower inference and heavier backbones.
- **Segmentation has unified.** OneFormer (one model for semantic + instance + panoptic) and Mask DINO (one head for detection + segmentation) replaced the specialist Mask2Former pipelines. EVA-02 / InternImage / ONE-PEACE backbones pushed ADE20K mIoU past 62 by 2024 and progress is now incremental.
- **Video-language unification via V-JEPA 2 / InternVideo2.** SSL video pretraining at 1B+ params (V-JEPA 2 ViT-g, InternVideo2-6B) closed the gap with supervised IN-21K-style recipes. K-400 top-1 is approaching the 92-93% saturation expected from label-noise floors; SSv2 sits stuck near 77-78 and remains the more diagnostic benchmark.
- **Contamination is suspected for every open VLM at the top.** Open-weight 70B-class VLMs reporting MMMU 70+ frequently train on instruction mixes that touch test-set distributions. Treat any cross-model gap under 3-4 points on MMMU / DocVQA / ChartQA as inside the noise.
- **Proprietary models are SOTA only on VLM benchmarks.** Unlike text embeddings (where open beat proprietary in 2024), in vision-language the gap is still real: Claude 4 Sonnet / GPT-5 / Gemini 2.5 Pro lead by 13-15 MMMU points. On pure-vision tasks (detection, segmentation, depth, classification) there is no proprietary entrant worth benchmarking.

## Sources

- [Papers With Code - ImageNet leaderboard](https://paperswithcode.com/sota/image-classification-on-imagenet) - pulled 2026-05-13
- [Papers With Code - COCO detection leaderboard](https://paperswithcode.com/sota/object-detection-on-coco) - pulled 2026-05-13
- [Papers With Code - ADE20K semantic segmentation](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k) - pulled 2026-05-13
- [MMMU benchmark leaderboard](https://mmmu-benchmark.github.io/) - pulled 2026-05-13
- [MMBench OpenCompass leaderboard](https://mmbench.opencompass.org.cn/leaderboard) - pulled 2026-05-13
- [OCRBench v2 leaderboard (HuggingFace)](https://huggingface.co/spaces/ling99/OCRBench-v2-leaderboard) - pulled 2026-05-13
- [Roboflow - Best object detection models 2026](https://blog.roboflow.com/best-object-detection-models/)
- [Roboflow - Best depth estimation models](https://blog.roboflow.com/depth-estimation-models/)
- [Meta AI - V-JEPA 2 release blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [Meta AI - DINOv2 release blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
- [HuggingFace - SigLIP 2 blog](https://huggingface.co/blog/siglip2)
- [open_clip pretrained model zoo](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md)
- Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge*, arXiv:1409.0575 (2015)
- Lin et al., *Microsoft COCO: Common Objects in Context*, arXiv:1405.0312 (2014)
- Zhou et al., *Scene Parsing through ADE20K Dataset*, CVPR 2017
- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), arXiv:2201.03545 (2022)
- Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, arXiv:2301.00808 (2023)
- Liu et al., *Swin Transformer V2: Scaling Up Capacity and Resolution*, arXiv:2111.09883 (2022)
- Touvron et al., *DeiT III: Revenge of the ViT*, arXiv:2204.07118 (2022)
- He et al., *Masked Autoencoders Are Scalable Vision Learners* (MAE), arXiv:2111.06377 (2021)
- Peng et al., *BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers*, arXiv:2208.06366 (2022)
- Fang et al., *EVA-02: A Visual Representation for Neon Genesis*, arXiv:2303.11331 (2023)
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193 (2023)
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP), arXiv:2303.15343 (2023)
- Tschannen et al., *SigLIP 2: Multilingual Vision-Language Encoders*, arXiv:2502.14786 (2025)
- Zhang et al., *DINO: DETR with Improved DeNoising Anchor Boxes*, arXiv:2203.03605 (2022)
- Zong et al., *DETRs with Collaborative Hybrid Assignments Training* (Co-DETR), arXiv:2211.12860 (2022)
- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR), arXiv:2304.08069 (2023)
- Tian et al., *YOLOv12: Attention-Centric Real-Time Object Detectors*, arXiv:2502.14740 (2025)
- Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, arXiv:2405.14458 (2024)
- Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, arXiv:2303.05499 (2023)
- Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation* (Mask2Former), arXiv:2112.01527 (2021)
- Jain et al., *OneFormer: One Transformer to Rule Universal Image Segmentation*, arXiv:2211.06220 (2022)
- Li et al., *Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation*, arXiv:2206.02777 (2022)
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, arXiv:2105.15203 (2021)
- Kirillov et al., *Segment Anything*, arXiv:2304.02643 (2023)
- Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714 (2024)
- Ranftl et al., *Vision Transformers for Dense Prediction* (DPT), arXiv:2103.13413 (2021)
- Bhat et al., *ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth*, arXiv:2302.12288 (2023)
- Ke et al., *Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation* (Marigold), arXiv:2312.02145 (2023)
- Yang et al., *Depth Anything V2*, arXiv:2406.09414 (2024)
- Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, arXiv:2203.12602 (2022)
- Wang et al., *VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking*, arXiv:2303.16727 (2023)
- Wang et al., *InternVideo: General Video Foundation Models*, arXiv:2212.03191 (2022)
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 (2025)
- Yue et al., *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*, arXiv:2311.16502 (2023)
- Liu et al., *MMBench: Is Your Multi-modal Model an All-around Player?*, arXiv:2307.06281 (2023)
- Mathew et al., *DocVQA: A Dataset for VQA on Document Images*, arXiv:2007.00398 (2020)
- Liu et al., *OCRBench v2*, arXiv:2501.00321 (2025)
- Li et al., *LLaVA-OneVision: Easy Visual Task Transfer*, arXiv:2408.03326 (2024)
- Bai et al., *Qwen2.5-VL Technical Report*, 2025
- Chen et al., *InternVL 2.5 Technical Report*, 2024
- Deitke et al., *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models*, 2024
- [LMCouncil AI Model Benchmarks May 2026](https://lmcouncil.ai/benchmarks) - pulled 2026-05-13
