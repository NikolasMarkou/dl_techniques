"""
Complete AB-UPT Pipeline Example

This script demonstrates the full pipeline from data generation to model training,
evaluation, and visualization. It serves as a comprehensive example of how to use
all the components together for CFD modeling with the AB-UPT architecture.

Components demonstrated:
1. Data generation and preprocessing
2. Model architecture setup
3. Training pipeline with validation
4. Model evaluation and metrics
5. Visualization and analysis
6. Model saving and loading

Run this script to see the complete AB-UPT pipeline in action.
"""

import os
import sys
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dl_techniques.utils.logger import logger
from dl_techniques.models.anchored_branched_upt import create_abupt_model, AnchoredBranchedUPT
from .train import (
    NormalizationStats,
    DataPreprocessor,
    CFDDataGenerator,
    CFDTrainer,
    create_synthetic_data
)
from .inference import (
    CFDInference,
    CFDEvaluator,
    CFDVisualizer,
    run_inference_pipeline
)


def setup_experiment_directory(base_dir: str = "experiments") -> str:
    """Create a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"abupt_experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ["checkpoints", "plots", "logs", "configs"]:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    logger.info(f"📁 Experiment directory created: {exp_dir}")
    return exp_dir


def save_experiment_config(exp_dir: str, config: Dict[str, Any]):
    """Save experiment configuration."""
    config_path = os.path.join(exp_dir, "configs", "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"💾 Configuration saved to {config_path}")


def create_data_splits(
        total_samples: int = 200,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
) -> tuple:
    """Create train/val/test data splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    logger.info(f"🎲 Generating {total_samples} synthetic CFD samples...")
    all_data = create_synthetic_data(num_samples=total_samples)

    # Split data
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    logger.info(f"📊 Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


def create_model_config() -> Dict[str, Any]:
    """Create model configuration."""
    return {
        "model_params": {
            "dim": 192,
            "num_heads": 3,
            "geometry_depth": 1,
            "blocks": "pscscs",
            "num_surface_blocks": 4,
            "num_volume_blocks": 4,
            "radius": 2.5,
            "dropout": 0.1
        },
        "data_params": {
            "num_geometry_points": 800,
            "num_surface_anchor_points": 300,
            "num_volume_anchor_points": 400,
            "num_geometry_supernodes": 150,
            "use_query_positions": True,
            "batch_size": 1
        },
        "training_params": {
            "learning_rate": 1e-4,
            "epochs": 20,
            "patience": 8,
            "loss_weights": {
                "surface_anchor_pressure": 1.0,
                "surface_anchor_wallshearstress": 0.5,
                "surface_query_pressure": 1.0,
                "surface_query_wallshearstress": 0.5,
                "volume_anchor_totalpcoeff": 1.0,
                "volume_anchor_velocity": 0.8,
                "volume_anchor_vorticity": 0.3,
                "volume_query_totalpcoeff": 1.0,
                "volume_query_velocity": 0.8,
                "volume_query_vorticity": 0.3,
            }
        }
    }


def run_complete_pipeline(
        exp_dir: str,
        config: Dict[str, Any],
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
):
    """Run the complete AB-UPT pipeline."""

    # Setup preprocessing
    logger.info("🔧 Setting up data preprocessing...")
    stats = NormalizationStats()
    preprocessor = DataPreprocessor(stats)

    # Create data generators
    logger.info("📦 Creating data generators...")
    data_params = config["data_params"]

    train_generator = CFDDataGenerator(
        train_data, preprocessor, **data_params, shuffle=True
    )
    val_generator = CFDDataGenerator(
        val_data, preprocessor, **data_params, shuffle=False
    )
    test_generator = CFDDataGenerator(
        test_data, preprocessor, **data_params, shuffle=False
    )

    # Create model
    logger.info("🏗️ Building AB-UPT model...")
    model = create_abupt_model(**config["model_params"])

    # Test model with single batch
    logger.info("🧪 Testing model with single batch...")
    try:
        inputs, targets = train_generator[0]
        outputs = model(inputs)
        logger.info("✅ Model forward pass successful!")

        # Log shapes
        logger.info("Input shapes:")
        for key, value in inputs.items():
            logger.info(f"  {key}: {value.shape}")

        logger.info("Output shapes:")
        for key, value in outputs.items():
            logger.info(f"  {key}: {value.shape}")

    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        raise

    # Setup trainer
    logger.info("🎯 Setting up trainer...")
    trainer = CFDTrainer(
        model=model,
        preprocessor=preprocessor,
        loss_weights=config["training_params"]["loss_weights"],
        learning_rate=config["training_params"]["learning_rate"]
    )

    # Train model
    logger.info("🚀 Starting training...")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    try:
        history = trainer.train(
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=config["training_params"]["epochs"],
            save_dir=checkpoint_dir,
            save_best_only=True,
            patience=config["training_params"]["patience"],
            verbose=1
        )

        # Save training history
        history_path = os.path.join(exp_dir, "logs", "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Plot training history
        plot_path = os.path.join(exp_dir, "plots", "training_history.png")
        trainer.plot_training_history(plot_path)

        logger.info("✅ Training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    # Evaluate model
    logger.info("📊 Evaluating model...")
    best_model_path = os.path.join(checkpoint_dir, "best_model.keras")

    if os.path.exists(best_model_path):
        try:
            # Run inference pipeline
            inference_dir = os.path.join(exp_dir, "inference")
            metrics = run_inference_pipeline(
                model_path=best_model_path,
                test_data=test_data,
                output_dir=inference_dir
            )

            logger.info("✅ Evaluation completed successfully!")
            return metrics

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    else:
        logger.warning("⚠️ Best model not found, skipping evaluation")
        return None


def create_summary_report(exp_dir: str, config: Dict[str, Any], metrics=None):
    """Create a summary report of the experiment."""
    report_path = os.path.join(exp_dir, "experiment_summary.md")

    with open(report_path, 'w') as f:
        f.write("# AB-UPT Experiment Summary\n\n")
        f.write(f"**Experiment Directory:** `{exp_dir}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(config["model_params"], indent=2))
        f.write("\n```\n\n")

        f.write("## Training Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(config["training_params"], indent=2))
        f.write("\n```\n\n")

        f.write("## Data Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(config["data_params"], indent=2))
        f.write("\n```\n\n")

        if metrics:
            f.write("## Evaluation Metrics\n\n")
            f.write(f"- **Overall MAE:** {metrics.overall_mae:.6f}\n")
            f.write(f"- **Overall RMSE:** {metrics.overall_rmse:.6f}\n\n")

            f.write("### Surface Quantities\n")
            f.write(f"- **Pressure MAE:** {metrics.surface_pressure_mae:.6f}\n")
            f.write(f"- **Pressure RMSE:** {metrics.surface_pressure_rmse:.6f}\n")
            f.write(f"- **Wall Shear MAE:** {metrics.surface_wallshear_mae:.6f}\n")
            f.write(f"- **Wall Shear RMSE:** {metrics.surface_wallshear_rmse:.6f}\n\n")

            f.write("### Volume Quantities\n")
            f.write(f"- **Pressure MAE:** {metrics.volume_pressure_mae:.6f}\n")
            f.write(f"- **Pressure RMSE:** {metrics.volume_pressure_rmse:.6f}\n")
            f.write(f"- **Velocity MAE:** {metrics.volume_velocity_mae:.6f}\n")
            f.write(f"- **Velocity RMSE:** {metrics.volume_velocity_rmse:.6f}\n")
            f.write(f"- **Vorticity MAE:** {metrics.volume_vorticity_mae:.6f}\n")
            f.write(f"- **Vorticity RMSE:** {metrics.volume_vorticity_rmse:.6f}\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `checkpoints/best_model.keras` - Best trained model\n")
        f.write("- `plots/training_history.png` - Training curves\n")
        f.write("- `logs/training_history.json` - Training metrics\n")
        f.write("- `inference/evaluation_metrics.json` - Evaluation results\n")
        f.write("- `inference/error_distributions.png` - Error analysis\n")
        f.write("- `inference/surface_pressure_comparison.png` - Surface predictions\n")
        f.write("- `inference/volume_velocity_slice.png` - Volume predictions\n\n")

        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("# Load the trained model\n")
        f.write("import keras\n")
        f.write(f"model = keras.models.load_model('{exp_dir}/checkpoints/best_model.keras')\n\n")
        f.write("# Use for inference\n")
        f.write("predictions = model(inputs)\n")
        f.write("```\n")

    logger.info(f"📋 Summary report saved to {report_path}")


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="AB-UPT Complete Pipeline")
    parser.add_argument("--samples", type=int, default=100, help="Total number of samples")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=192, help="Model dimension")
    parser.add_argument("--exp-dir", type=str, default="experiments", help="Base experiment directory")

    args = parser.parse_args()

    # Banner
    print("🌊" * 50)
    print("     AB-UPT: Anchored Branched Universal Physics Transformer")
    print("     Complete Pipeline for Computational Fluid Dynamics")
    print("🌊" * 50)
    print()

    # Setup experiment
    exp_dir = setup_experiment_directory(args.exp_dir)

    # Create configuration
    config = create_model_config()
    config["training_params"]["epochs"] = args.epochs
    config["model_params"]["dim"] = args.dim

    save_experiment_config(exp_dir, config)

    # Create data
    train_data, val_data, test_data = create_data_splits(total_samples=args.samples)

    # Run pipeline
    try:
        metrics = run_complete_pipeline(exp_dir, config, train_data, val_data, test_data)

        # Create summary
        create_summary_report(exp_dir, config, metrics)

        logger.info("🎉 Complete pipeline finished successfully!")
        logger.info(f"📁 Results saved to: {exp_dir}")

        # Final summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Results directory: {exp_dir}")
        print(f"🏆 Best model: {exp_dir}/checkpoints/best_model.keras")
        print(f"📊 Visualizations: {exp_dir}/plots/ and {exp_dir}/inference/")
        print(f"📋 Summary report: {exp_dir}/experiment_summary.md")

        if metrics:
            print(f"📈 Overall MAE: {metrics.overall_mae:.6f}")
            print(f"📈 Overall RMSE: {metrics.overall_rmse:.6f}")

        print("=" * 60)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()