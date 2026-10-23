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
    :param analyze_on_train_begin: Also analyze the model once before training
        starts, into ``epoch_000``. Skipped with a warning when the model is
        not built yet (its weights do not exist). Defaults to True.
    :type analyze_on_train_begin: bool
    :param analyze_first_epoch: Also analyze after epoch 1 whatever
        ``start_epoch`` and ``epoch_frequency`` say. Defaults to True.
    :type analyze_first_epoch: bool

    The standard schedule is therefore ``epoch_000``, ``epoch_001``, then
    ``start_epoch + k * epoch_frequency``. Both flags False give the schedule
    this callback had before they existed.
    """

    def __init__(
        self,
        output_dir: str,
        analysis_config: Optional[AnalysisConfig] = None,
        start_epoch: int = 1,
        epoch_frequency: int = 1,
        model_name: str = "TrainingModel",
        analyze_on_train_begin: bool = True,
        analyze_first_epoch: bool = True,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.start_epoch = start_epoch
        self.epoch_frequency = epoch_frequency
        self.model_name = model_name
        self.analyze_on_train_begin = analyze_on_train_begin
        self.analyze_first_epoch = analyze_first_epoch

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

    def _analyze(self, epoch_number: int) -> None:
        """Run the analyzer into ``epoch_<NNN>`` for the current model state.

        :param epoch_number: 0 for the pre-training snapshot, else the
            1-indexed epoch that just finished.
        :type epoch_number: int
        """
        epoch_output_dir = os.path.join(self.output_dir, f"epoch_{epoch_number:03d}")
        logger.info(
            f"Running EpochAnalyzerCallback for epoch {epoch_number} ---"
        )
        logger.info(f"Saving results to: {epoch_output_dir}")

        # DECISION plan-2026-10-14T062616-81938e81/D-117
        # No try/except here: a failed analysis raises out of fit. Do NOT
        # log-and-continue; that hid the Step 13 DarkIR run's NaN weights
        # ("SVD did not converge" at every epoch) behind an exit code of 0.
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
            f" Epoch {epoch_number} analysis complete ---"
        )

    def on_train_begin(self, logs: Optional[dict] = None):
        """Analyze the untrained model into ``epoch_000``.

        :param logs: Unused.
        :type logs: Optional[dict]
        :raises Exception: Whatever the analysis raises (D-117).
        """
        if not self.analyze_on_train_begin:
            return
        # DECISION plan-2026-10-23T100412-e7eaaa02/D-003
        # An unbuilt model has no weights to analyze; skip it loudly instead of
        # letting a lazily-built model fail here when it would have trained fine.
        # This is a precondition, not a swallowed error: any failure of a built
        # model still raises (D-117).
        if not self.model.built:
            logger.warning(
                "EpochAnalyzerCallback: model is not built, skipping the "
                "pre-training snapshot (epoch_000)."
            )
            return
        self._analyze(0)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Runs the model analysis at the end of an epoch.

        :param epoch: The current epoch number (0-indexed).
        :type epoch: int
        :param logs: A dictionary of logs from the training process (e.g., loss, acc).
        :type logs: Optional[dict]
        :raises Exception: Whatever the analysis raises; it propagates out of
            ``fit``, so a run whose analysis failed does not exit 0.
        """
        # Keras epochs are 0-indexed, but we often think of them as 1-indexed.
        current_epoch = epoch + 1

        on_cadence = (
            current_epoch >= self.start_epoch
            and (current_epoch - self.start_epoch) % self.epoch_frequency == 0
        )
        if on_cadence or (self.analyze_first_epoch and current_epoch == 1):
            self._analyze(current_epoch)
