"""Common argument parser utilities for training scripts."""

import argparse
from typing import Optional, List


# ---------------------------------------------------------------------

def create_base_argument_parser(
        description: str = "Train model",
        default_dataset: str = "cifar10",
        dataset_choices: Optional[List[str]] = None,
) -> argparse.ArgumentParser:
    """Create argument parser with standard training arguments.

    Scripts extend this parser with model-specific arguments:

        parser = create_base_argument_parser("Train MyModel")
        parser.add_argument('--variant', type=str, default='tiny')
        args = parser.parse_args()

    Args:
        description: Help text for the parser.
        default_dataset: Default dataset name.
        dataset_choices: Valid dataset names. Defaults to
            ['mnist', 'cifar10', 'cifar100', 'imagenet'].

    Returns:
        ArgumentParser with common training arguments.
    """
    if dataset_choices is None:
        dataset_choices = ['mnist', 'cifar10', 'cifar100', 'imagenet']

    parser = argparse.ArgumentParser(description=description)

    # Data
    parser.add_argument('--dataset', type=str, default=default_dataset,
                        choices=dataset_choices, help='Dataset to use')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (for ImageNet, default: 224)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')

    # Optimization
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'],
                        help='Learning rate schedule')

    # Early stopping
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # GPU
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device index to use (default: all GPUs)')

    # Output
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots interactively')

    return parser
