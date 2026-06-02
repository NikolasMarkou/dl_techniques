"""Common utilities shared across training scripts."""

from train.common.gpu import setup_gpu
from train.common.args import create_base_argument_parser
from train.common.datasets import load_dataset, load_imagenet_dataset, get_class_names, CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD, make_imagenet_filesystem_dataset, collect_image_paths, DEFAULT_IMAGE_EXTENSIONS
from train.common.callbacks import create_callbacks, create_learning_rate_schedule, EpochMetricsPlotCallback
from train.common.evaluation import (
    validate_model_loading,
    convert_keras_history_to_training_history,
    create_classification_results,
    generate_training_curves,
    generate_comprehensive_visualizations,
    run_model_analysis,
)
from train.common.nlp import (
    create_tokenizer as create_nlp_tokenizer,
    decode_text,
    load_text_dataset,
    preprocess_mlm_dataset,
    preprocess_classification_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    estimate_clm_steps_per_epoch,
    GenerationProbeCallback,
)
from train.common.image_text import (
    IMAGE_MEAN,
    IMAGE_STD,
    read_decode_resize_uint8,
    augment_and_normalize,
    tokenize_captions,
    build_synthetic_image_text_dataset,
    load_coco2017_local_split,
    load_cc3m_local_split,
    make_image_text_tf_dataset,
)
from train.common.megadepth import (
    discover_megadepth_pairs,
    load_and_process_pair as load_megadepth_pair,
    MegaDepthDataset,
)
from train.common.seed import set_seeds
from train.common.config_io import save_config_json, json_numpy_default
from train.common.step_plots import plot_step_metrics, StepPlotCallback
from train.common.step_checkpoint import StepCheckpointCallback
from train.common.tfrecord import (
    SchemaSpec,
    IMAGE_TEXT_SCHEMA,
    IMAGE_LABEL_SCHEMA,
    bytes_feature,
    int64_feature,
    int64_list_feature,
    float_feature,
    float_list_feature,
    make_example,
    build_image_text_example,
    build_image_label_example,
    write_tfrecord_shards,
    list_tfrecord_shards,
    read_tfrecord_dataset,
    make_image_text_tf_dataset_from_tfrecord,
)
from train.common.token_superposition import (
    TSTConfig,
    TSTState,
    TSTEmbedding,
    TSTCausalLMLoss,
    TSTPhaseCallback,
    tst_phase1_transform,
    tst_phase2_transform,
    apply_tst,
)
