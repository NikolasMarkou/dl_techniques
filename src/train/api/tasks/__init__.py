"""The training tasks of ``python -m train``, one module each.

Interface contract
------------------
``register_builtin_tasks()``
    Register every task of this package into
    :data:`train.api.registry.TASKS`. Idempotent: a task already registered is
    left alone. Imports no task code: a ``TaskSpec`` holds ``"module:attr"``
    text (D-012), so ``python -m train help`` stays free of TensorFlow.

Importing this package registers nothing, as ``train.api.registry`` itself
registers nothing at import: ``test_registry`` swaps both dicts for empty ones.
"""

from .. import registry
from ..pretrain_args import BEIT, ET

#: One row per task module, in listing order.
BUILTIN_TASKS = (
    registry.TaskSpec(
        name="image_classification",
        config_class="train.api.tasks.image_classification:ImageClassificationConfig",
        function="train.api.tasks.image_classification:train_image_classification",
        summary="train an image classifier on a built-in dataset",
    ),
    registry.TaskSpec(
        name="image_segmentation",
        config_class="train.api.tasks.image_segmentation:ImageSegmentationConfig",
        function="train.api.tasks.image_segmentation:train_image_segmentation",
        summary="train a semantic segmenter on Oxford-IIIT Pet or on image and mask files",
    ),
    registry.TaskSpec(
        name="image_denoising",
        config_class="train.api.tasks.image_denoising:ImageDenoisingConfig",
        function="train.api.tasks.image_denoising:train_image_denoising",
        summary="train a blind image denoiser on folders of clean images with noise added on the fly",
    ),
    registry.TaskSpec(
        name="pretrain",
        config_class="train.api.tasks.pretrain:PretrainConfig",
        function="train.api.tasks.pretrain:train_pretrain",
        summary=f"pretrain a masked-image model ({ET} patch completion, {BEIT} against a frozen tokenizer)",
    ),
    registry.TaskSpec(
        name="time_series_forecasting",
        config_class="train.api.tasks.time_series_forecasting:TimeSeriesForecastingConfig",
        function="train.api.tasks.time_series_forecasting:train_time_series_forecasting",
        summary="train a forecaster on a synthetic series bank or a csv, scored against naive-last and seasonal-naive",
    ),
    registry.TaskSpec(
        name="depth_estimation",
        config_class="train.api.tasks.depth_estimation:DepthEstimationConfig",
        function="train.api.tasks.depth_estimation:train_depth_estimation",
        summary="train a relative-depth model on MegaDepth, scored against a constant and the average layout",
    ),
    registry.TaskSpec(
        name="language_model_pretraining",
        config_class="train.api.tasks.language_model_pretraining:LanguageModelPretrainingConfig",
        function="train.api.tasks.language_model_pretraining:train_language_model_pretraining",
        summary="train a GPT-2 as a causal language model on a text corpus, scored against the unigram of its selection targets",
    ),
    registry.TaskSpec(
        name="text_classification",
        config_class="train.api.tasks.text_classification:TextClassificationConfig",
        function="train.api.tasks.text_classification:train_text_classification",
        summary="train a BERT or GPT-2 text classifier on a labelled corpus, scored against the majority class of its selection split",
    ),
    registry.TaskSpec(
        name="beit_tokenizer",
        config_class="train.api.tasks.beit_tokenizer:BeitTokenizerConfig",
        function="train.api.tasks.beit_tokenizer:train_beit_tokenizer",
        summary="train the BEiT stage-0 tokenizer, a discrete VQ autoencoder, scored against the constant image and a codebook-usage floor",
    ),
    registry.TaskSpec(
        name="dino",
        config_class="train.api.tasks.dino:DinoConfig",
        function="train.api.tasks.dino:train_dino",
        summary="train a DINOv1 self-supervised ViT (a student, an EMA teacher, multi-crop views), scored by a k-NN probe against constant and untrained features",
    ),
)


def register_builtin_tasks() -> None:
    """Register the built-in tasks; see the module contract."""
    for spec in BUILTIN_TASKS:
        if spec.name not in registry.TASKS:
            registry.register_task(spec)
