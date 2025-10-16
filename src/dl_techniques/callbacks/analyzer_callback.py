"""
Keras Callback for In-Training Model Analysis
==============================================

This module provides a Keras callback that leverages the ModelAnalyzer toolkit
to perform deep analysis of a model's state at the end of specified epochs
during training.

This allows for a granular, time-series view of how model properties like
weight distributions and spectral characteristics evolve over the training
process.
"""

import os
import keras
from typing import Optional

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------


from dl_techniques.utils.logger import logger
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig

# ---------------------------------------------------------------------


class EpochAnalyzerCallback(keras.callbacks.Callback):
    """A Keras callback to run ModelAnalyzer at the end of specified epochs.

    This callback performs data-free analysis (weights and spectral) on the
    model at different stages of training, saving the results to structured
    subdirectories.

    :param output_dir: The root directory where analysis results will be saved.
        A subdirectory will be created for each analyzed epoch.
    :type output_dir: str
    :param analysis_config: An optional `AnalysisConfig` object to configure
        the analysis. If None, a default config for weight/spectral analysis
        is used.
    :type analysis_config: Optional[AnalysisConfig]
    :param start_epoch: The epoch number at which to start the analysis.
        Defaults to 1.
    :type start_epoch: int
    :param epoch_frequency: The frequency (in epochs) at which to run the
        analysis. For example, `epoch_frequency=5` runs analysis every 5th
        epoch. Defaults to 1.
    :type epoch_frequency: int
    :param model_name: A descriptive name for the model being trained, used
        for labeling in the analysis results. Defaults to "TrainingModel".
    :type model_name: str
    """

    def __init__(
        self,
        output_dir: str,
        analysis_config: Optional[AnalysisConfig] = None,
        start_epoch: int = 1,
        epoch_frequency: int = 1,
        model_name: str = "TrainingModel"
    ):
        super().__init__()
        self.output_dir = output_dir
        self.start_epoch = start_epoch
        self.epoch_frequency = epoch_frequency
        self.model_name = model_name

        # If no config is provided, create a default one focused on fast,
        # data-free weight and spectral analysis.
        if analysis_config is None:
            self.analysis_config = AnalysisConfig(
                analyze_weights=True,
                analyze_spectral=True,
                analyze_calibration=False,         # Skip data-dependent analyses
                analyze_information_flow=False,
                analyze_training_dynamics=False,
                verbose=False  # Keep logs clean during training
            )
        else:
            self.analysis_config = analysis_config

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"EpochAnalyzerCallback configured. Analysis will run every "
            f"{self.epoch_frequency} epochs, starting from epoch {self.start_epoch}."
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Runs the model analysis at the end of an epoch.

        :param epoch: The current epoch number (0-indexed).
        :type epoch: int
        :param logs: A dictionary of logs from the training process (e.g., loss, acc).
        :type logs: Optional[dict]
        """
        # Keras epochs are 0-indexed, but we often think of them as 1-indexed.
        current_epoch = epoch + 1

        # Check if this is an epoch we should analyze
        if (current_epoch >= self.start_epoch and
                (current_epoch - self.start_epoch) % self.epoch_frequency == 0):

            epoch_output_dir = os.path.join(self.output_dir, f"epoch_{current_epoch:03d}")
            logger.info(
                f"\n--- Running EpochAnalyzerCallback for epoch {current_epoch} ---"
            )
            logger.info(f"Saving results to: {epoch_output_dir}")

            try:
                # 1. Instantiate the ModelAnalyzer with the current model state
                analyzer = ModelAnalyzer(
                    models={self.model_name: self.model},
                    config=self.analysis_config,
                    output_dir=epoch_output_dir
                )

                # 2. Run the analysis (no data needed for this configuration)
                # The `analyze` method will save plots and JSON to the epoch_output_dir
                _ = analyzer.analyze()

                logger.info(
                    f"--- Epoch {current_epoch} analysis complete ---"
                )

            except Exception as e:
                logger.error(
                    f"EpochAnalyzerCallback failed at epoch {current_epoch}: {e}",
                    exc_info=True
                )