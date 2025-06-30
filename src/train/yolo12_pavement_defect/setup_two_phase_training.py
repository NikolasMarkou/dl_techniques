"""
Setup and Verification Script for Two-Phase YOLOv12 Training

This script helps set up the environment and verify that all dependencies
are correctly installed for the two-phase training workflow.

It performs the following checks and setups:
1. Verify existing dependencies (TensorFlow, Keras, etc.)
2. Install tensorflow-datasets for COCO loading
3. Check GPU availability and memory
4. Verify disk space for COCO dataset
5. Test COCO dataset loading
6. Create necessary directory structure
7. Provide configuration recommendations

Usage:
    python setup_two_phase_training.py [--install-deps] [--test-coco] [--check-space]

File: scripts/setup_two_phase_training.py
"""

import os
import sys
import subprocess
import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def print_status(message: str, status: str = "INFO") -> None:
    """Print a status message with emoji."""
    emoji_map = {
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "INFO": "ℹ️",
        "PROGRESS": "🔄"
    }
    emoji = emoji_map.get(status, "📝")
    print(f"{emoji} {message}")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_header("Python Version Check")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print_status("Python version is compatible", "SUCCESS")
        return True
    else:
        print_status(f"Python 3.8+ required, found {version.major}.{version.minor}", "ERROR")
        return False


def check_core_dependencies() -> Dict[str, bool]:
    """Check if core dependencies are installed."""
    print_header("Core Dependencies Check")

    dependencies = {
        'tensorflow': False,
        'keras': False,
        'numpy': False,
        'matplotlib': False,
        'pandas': False
    }

    for dep in dependencies:
        try:
            __import__(dep)
            print_status(f"{dep}: Installed", "SUCCESS")
            dependencies[dep] = True
        except ImportError:
            print_status(f"{dep}: Not found", "ERROR")

    return dependencies


def install_tensorflow_datasets() -> bool:
    """Install tensorflow-datasets if not present."""
    print_header("TensorFlow Datasets Installation")

    try:
        import tensorflow_datasets as tfds
        print_status("tensorflow-datasets already installed", "SUCCESS")
        print(f"Version: {tfds.__version__}")
        return True
    except ImportError:
        print_status("tensorflow-datasets not found, installing...", "PROGRESS")

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "tensorflow-datasets"
            ])

            # Verify installation
            import tensorflow_datasets as tfds
            print_status("tensorflow-datasets installed successfully", "SUCCESS")
            print(f"Version: {tfds.__version__}")
            return True

        except subprocess.CalledProcessError as e:
            print_status(f"Failed to install tensorflow-datasets: {e}", "ERROR")
            return False
        except ImportError:
            print_status("Installation completed but import still fails", "ERROR")
            return False


def check_gpu_availability() -> Dict[str, any]:
    """Check GPU availability and memory."""
    print_header("GPU Availability Check")

    gpu_info = {
        'available': False,
        'count': 0,
        'memory_info': []
    }

    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_info['available'] = len(gpus) > 0
        gpu_info['count'] = len(gpus)

        if gpu_info['available']:
            print_status(f"Found {len(gpus)} GPU(s)", "SUCCESS")

            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")

                # Try to get memory info
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print_status(f"Memory growth enabled for GPU {i}", "SUCCESS")
                except Exception as e:
                    print_status(f"Could not configure GPU {i}: {e}", "WARNING")
        else:
            print_status("No GPUs found - will use CPU", "WARNING")
            print_status("Training will be significantly slower on CPU", "WARNING")

    except Exception as e:
        print_status(f"Error checking GPU: {e}", "ERROR")

    return gpu_info


def check_disk_space(path: str = ".") -> Dict[str, float]:
    """Check available disk space."""
    print_header("Disk Space Check")

    try:
        statvfs = os.statvfs(path)
        # Convert to GB
        available_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024 ** 3)
        total_gb = (statvfs.f_blocks * statvfs.f_frsize) / (1024 ** 3)
        used_gb = total_gb - available_gb

        print(f"Path: {os.path.abspath(path)}")
        print(f"Total space: {total_gb:.1f} GB")
        print(f"Used space: {used_gb:.1f} GB")
        print(f"Available space: {available_gb:.1f} GB")

        # COCO dataset requires ~37GB + cache space
        required_space = 50  # GB

        if available_gb >= required_space:
            print_status(f"Sufficient disk space ({available_gb:.1f} GB available)", "SUCCESS")
        else:
            print_status(f"Insufficient disk space. Need {required_space} GB, have {available_gb:.1f} GB", "ERROR")

        return {
            'available_gb': available_gb,
            'total_gb': total_gb,
            'sufficient': available_gb >= required_space
        }

    except Exception as e:
        print_status(f"Could not check disk space: {e}", "ERROR")
        return {'available_gb': 0, 'total_gb': 0, 'sufficient': False}


def test_coco_loading() -> bool:
    """Test COCO dataset loading without downloading."""
    print_header("COCO Dataset Loading Test")

    try:
        import tensorflow_datasets as tfds

        print_status("Testing COCO dataset access...", "PROGRESS")

        # Try to get dataset info without downloading
        try:
            builder = tfds.builder('coco/2017')
            info = builder.info

            print_status("COCO dataset accessible", "SUCCESS")
            print(f"  Train examples: {info.splits['train'].num_examples:,}")
            print(f"  Validation examples: {info.splits['validation'].num_examples:,}")
            print(f"  Features: {list(info.features.keys())}")

            # Check if already downloaded
            if builder.is_downloaded:
                print_status("COCO dataset already downloaded", "SUCCESS")
            else:
                print_status("COCO dataset not downloaded (will download on first use)", "INFO")
                print_status("Download size: ~37 GB", "INFO")

            return True

        except Exception as e:
            print_status(f"COCO dataset access failed: {e}", "ERROR")
            return False

    except ImportError:
        print_status("tensorflow-datasets not available for testing", "ERROR")
        return False


def create_directory_structure(base_dir: str = ".") -> bool:
    """Create recommended directory structure."""
    print_header("Directory Structure Setup")

    directories = [
        "scripts",
        "coco_pretrain_results",
        "finetune_results",
        "data",
        "cache"
    ]

    try:
        base_path = Path(base_dir)

        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(exist_ok=True)
            print_status(f"Created/verified: {dir_path}", "SUCCESS")

        # Create a .gitignore for large files
        gitignore_content = """
# Large training files
coco_pretrain_results/
cache/
*.weights.h5
*.keras
*.h5

# Dataset caches
__pycache__/
.cache/
"""

        gitignore_path = base_path / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content.strip())
            print_status("Created .gitignore", "SUCCESS")

        return True

    except Exception as e:
        print_status(f"Failed to create directories: {e}", "ERROR")
        return False


def generate_config_recommendations(gpu_info: Dict[str, any], disk_info: Dict[str, float]) -> Dict[str, any]:
    """Generate configuration recommendations based on system specs."""
    print_header("Configuration Recommendations")

    recommendations = {
        'model_scale': 'n',
        'batch_size': 8,
        'img_size': 640,
        'cache_dir': './cache',
        'epochs': 40
    }

    # Adjust based on GPU memory (rough estimates)
    if gpu_info['available'] and gpu_info['count'] > 0:
        # These are rough estimates - actual memory usage varies
        print_status("GPU detected - optimizing for GPU training", "INFO")

        # Estimate GPU memory (we can't easily get this without actually allocating)
        # These recommendations are conservative
        recommendations.update({
            'model_scale': 's',
            'batch_size': 16,
            'epochs': 50
        })

        if gpu_info['count'] > 1:
            print_status("Multiple GPUs detected", "INFO")
            print_status("Note: Multi-GPU training requires additional configuration", "INFO")

    else:
        print_status("No GPU detected - CPU optimizations", "WARNING")
        recommendations.update({
            'model_scale': 'n',
            'batch_size': 4,
            'epochs': 20  # Shorter training on CPU
        })
        print_status("Training on CPU will be very slow (days instead of hours)", "WARNING")

    # Disk space optimizations
    if disk_info['available_gb'] < 100:
        print_status("Limited disk space - using minimal cache", "WARNING")
        recommendations['cache_dir'] = None
        recommendations['img_size'] = 512  # Smaller images

    print("Recommended configuration:")
    for key, value in recommendations.items():
        print(f"  --{key.replace('_', '-')}: {value}")

    return recommendations


def save_system_report(
        dependencies: Dict[str, bool],
        gpu_info: Dict[str, any],
        disk_info: Dict[str, float],
        recommendations: Dict[str, any]
) -> None:
    """Save a system compatibility report."""

    report = {
        'timestamp': str(subprocess.check_output(['date'], text=True).strip()),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'dependencies': dependencies,
        'gpu_info': gpu_info,
        'disk_info': disk_info,
        'recommendations': recommendations,
        'compatibility_score': calculate_compatibility_score(dependencies, gpu_info, disk_info)
    }

    try:
        with open('system_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print_status("System report saved to system_report.json", "SUCCESS")
    except Exception as e:
        print_status(f"Could not save system report: {e}", "WARNING")


def calculate_compatibility_score(
        dependencies: Dict[str, bool],
        gpu_info: Dict[str, any],
        disk_info: Dict[str, float]
) -> str:
    """Calculate overall compatibility score."""

    score = 0
    max_score = 10

    # Dependencies (5 points)
    dep_score = sum(dependencies.values()) / len(dependencies) * 5
    score += dep_score

    # GPU (3 points)
    if gpu_info['available']:
        score += 3

    # Disk space (2 points)
    if disk_info['sufficient']:
        score += 2

    percentage = (score / max_score) * 100

    if percentage >= 90:
        return f"EXCELLENT ({percentage:.0f}%)"
    elif percentage >= 70:
        return f"GOOD ({percentage:.0f}%)"
    elif percentage >= 50:
        return f"FAIR ({percentage:.0f}%)"
    else:
        return f"POOR ({percentage:.0f}%)"


def print_summary(
        dependencies: Dict[str, bool],
        gpu_info: Dict[str, any],
        disk_info: Dict[str, float],
        tfds_available: bool
) -> None:
    """Print setup summary."""
    print_header("Setup Summary")

    # Overall status
    all_deps = all(dependencies.values()) and tfds_available

    if all_deps and gpu_info['available'] and disk_info['sufficient']:
        print_status("System is ready for two-phase training!", "SUCCESS")
        print_status("You can proceed with COCO pre-training", "SUCCESS")
    elif all_deps:
        print_status("Dependencies installed, but check GPU/disk warnings above", "WARNING")
    else:
        print_status("Setup incomplete - please address the errors above", "ERROR")

    # Next steps
    print("\nNext steps:")
    if all_deps:
        print("  1. Run COCO pre-training:")
        print("     python scripts/coco_pretrain.py --scale s --epochs 50")
        print("  2. Then run fine-tuning:")
        print("     python scripts/finetune_pretrained.py --data-dir /path/to/sut")
    else:
        print("  1. Install missing dependencies")
        print("  2. Re-run this setup script")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description='Setup and verify environment for two-phase YOLOv12 training'
    )
    parser.add_argument('--install-deps', action='store_true',
                        help='Automatically install missing dependencies')
    parser.add_argument('--test-coco', action='store_true',
                        help='Test COCO dataset loading')
    parser.add_argument('--check-space', type=str, default='.',
                        help='Path to check disk space')

    args = parser.parse_args()

    print_header("YOLOv12 Two-Phase Training Setup")
    print("This script will verify your environment and install required dependencies.")

    # Run checks
    python_ok = check_python_version()
    if not python_ok:
        print_status("Please upgrade Python and re-run this script", "ERROR")
        return

    dependencies = check_core_dependencies()

    # Install tensorflow-datasets
    tfds_available = True
    if args.install_deps:
        tfds_available = install_tensorflow_datasets()
    else:
        try:
            import tensorflow_datasets
            print_status("tensorflow-datasets is available", "SUCCESS")
        except ImportError:
            print_status("tensorflow-datasets not found. Use --install-deps to install", "WARNING")
            tfds_available = False

    gpu_info = check_gpu_availability()
    disk_info = check_disk_space(args.check_space)

    # Test COCO if requested
    if args.test_coco and tfds_available:
        test_coco_loading()

    # Create directory structure
    create_directory_structure()

    # Generate recommendations
    recommendations = generate_config_recommendations(gpu_info, disk_info)

    # Save report
    save_system_report(dependencies, gpu_info, disk_info, recommendations)

    # Print summary
    print_summary(dependencies, gpu_info, disk_info, tfds_available)


if __name__ == '__main__':
    main()