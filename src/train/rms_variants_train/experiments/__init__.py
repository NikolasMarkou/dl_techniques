"""Experiment trainers for the RMSNorm-variants study.

Each experiment is a standalone trainer module exposing a ``main()``
entry point with argparse + ``--norm-type`` plumbed into the model factory.

Modules:
    e1_vit_cifar10           ViT-pico / CIFAR-10 (image classification)
    e2_resnet_cifar100       ResNet-18 / CIFAR-100 (deeper conv stack)
    e3_tinytransformer_imdb  4-layer transformer / IMDb (NLP)
    e4_deep_residual_reg     24-block deep residual / fp16 (norm stress test)
    e5_norm_layer_microbench Synthetic Gaussian regression (layer-level baseline)
"""
