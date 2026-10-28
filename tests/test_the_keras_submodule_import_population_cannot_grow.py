"""Ratchet: the ``from keras import ...`` population under ``src/`` and ``tests/`` cannot grow.

The house rule is ``import keras`` and a qualified call site (``keras.ops.*``,
``keras.layers.*``). Two statement kinds break it, both found by an AST walk
(not a text grep, so docstrings and comments are never counted):

* ``from keras import X`` / ``from keras.sub import X`` (absolute only);
* ``import keras.sub`` (``import keras`` itself is the allowed form).

Measured at Step 12 of ``plan-2026-10-14T062616-81938e81``, after
``layers/pooling/pixel_shuffle.py``, ``layers/conv_blocks/universal_inverted_bottleneck.py``
and ``tests/test_train/test_cliffordnet/test_train_clip_textlm.py`` were
converted: **543 statements in 521 files**. The rest are not migrated in bulk:
the user stopped a mass migration of these files on 2026-08-25 ("just fix what
i broke, no need to change all those files"); a file is converted when it is
edited for another reason (decisions.md D-082).

So this is a ceiling, not a ban. Converting a file lowers the measured value;
lower the ceilings to match in the same commit. A new ``from keras import``
anywhere raises the count and fails here.

Re-measured at Step 2.6 of the same plan (D-162), after the user's
``timestral`` port (6d757c6a6, 10 statements in 8 files) was converted:
**538 statements in 516 files**. The old ceilings (543 / 521) were 5 above the
population, so the port's 10 only showed up as 548 > 543.

Step 7.1 of the same plan (D-216): **536 statements in 514 files**. One
statement left with a deleted test file (the shared PFT-SR variant block's
tests) and one with ``test_guide_conformance.py``, converted while its
deleted subject was dropped.

Step 12.7 of the same plan (D-252): **534 statements in 512 files**, after the
two SAM 2 files that step edits (``sam2/training_model.py`` and its test).

Step 15.2 of the same plan (D-254): every file this plan's commits edited is
converted (100 files, 103 statements; the user's rule converts a file when it is
edited, and these were edited without it), leaving **431 statements in 412
files**, none of them a file this plan touched; then **430 in 411** after
``tests/test_layers/test_the_kan_actually_splines.py``, edited for D-257, was
converted with it; then **429 in 410** after ``models/vision_language/sam/sam2/neck.py``,
edited for D-260 at step 12.8, was converted with it; then **428 in 409** after
``train/cliffordnet/train_cliffordnet_nlp.py``, edited for D-265 at step 12.10, was
converted with it; then **427 in 408** after ``models/vision/convunext/model.py``,
edited for D-268 at step 14, was converted with it. The rest stay as they are (the
2026-08-25 ruling). The ceilings became an exact census: :data:`KERAS_SUBMODULE_IMPORTS`
pins the statement count of every file, so a conversion, a new statement in a
new file, a second statement in a listed file, and a swap (one file fixed,
another broken) all fail until the census is edited in the same commit.
"""

import ast
import pathlib
import warnings
from typing import Dict

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("src", "tests")

#: Exact census, ``{path relative to the repo root: statements}``, re-derived at
#: Step 15.2 of plan-2026-10-14T062616-81938e81 (D-254): 431 statements, 412 files;
#: 430 / 411 after ``test_the_kan_actually_splines.py`` was edited for D-257;
#: 429 / 410 after ``sam2/neck.py`` was edited for D-260 (step 12.8);
#: 428 / 409 after ``train_cliffordnet_nlp.py`` was edited for D-265 (step 12.10);
#: 427 / 408 after ``convunext/model.py`` was edited for D-268 (step 14);
#: 426 / 407 after ``mambalct/uca_encoder.py`` moved to ``keras.ops`` (plan-2026-10-26T125959-7af851e5 step 16).
#: 425 / 406 after ``capsnet/model.py`` moved to ``keras.ops`` (plan-2026-10-26T125959-7af851e5 step 17).
#: 423 / 404 after ``deepsort/appearance.py`` and its test moved to ``keras.ops`` (the same plan, step 18).
#: 421 / 402 after ``siamfc/model.py`` and its test moved to ``keras.ops`` (the same plan, step 18).
#: 419 / 400 after ``dasiamrpn/model.py`` and its test moved to ``keras.ops`` (the same plan, step 18).
#: 418 / 399 after ``sam1/prompt_encoder.py`` moved to ``keras.ops`` and ``keras.layers`` (the same plan, step 19).
#: Edit it in the commit that converts a file (delete its line) or, with a
#: recorded reason, adds a statement.
KERAS_SUBMODULE_IMPORTS: Dict[str, int] = {
    "src/dl_techniques/callbacks/jepa_visualization.py": 1,
    "src/dl_techniques/callbacks/spatial_loss_logger.py": 1,
    "src/dl_techniques/constraints/value_range_constraint.py": 1,
    "src/dl_techniques/datasets/arc/arc_keras.py": 2,
    "src/dl_techniques/layers/adapters/gated_adapter.py": 1,
    "src/dl_techniques/layers/anchor_generator.py": 1,
    "src/dl_techniques/layers/attention/capsule_routing_attention.py": 1,
    "src/dl_techniques/layers/attention/energy_attention.py": 1,
    "src/dl_techniques/layers/attention/hopfield_attention.py": 1,
    "src/dl_techniques/layers/attention/linear_attention.py": 1,
    "src/dl_techniques/layers/blt/dynamic_patcher.py": 1,
    "src/dl_techniques/layers/blt/entropy_model.py": 1,
    "src/dl_techniques/layers/blt/local_decoder.py": 1,
    "src/dl_techniques/layers/blt/patch_pooling.py": 1,
    "src/dl_techniques/layers/conv_blocks/mobile_one_block.py": 1,
    "src/dl_techniques/layers/embedding/albert_factorized_embedding.py": 1,
    "src/dl_techniques/layers/embedding/axial_rope_2d.py": 1,
    "src/dl_techniques/layers/embedding/bert_embeddings.py": 1,
    "src/dl_techniques/layers/embedding/continuous_rope_embedding.py": 1,
    "src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py": 1,
    "src/dl_techniques/layers/embedding/hierarchical_codebook_embedding.py": 1,
    "src/dl_techniques/layers/embedding/modern_bert_embeddings.py": 1,
    "src/dl_techniques/layers/embedding/patch_embedding.py": 1,
    "src/dl_techniques/layers/embedding/positional_embedding.py": 1,
    "src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py": 1,
    "src/dl_techniques/layers/eomt_mask.py": 1,
    "src/dl_techniques/layers/fastvit/attention_block.py": 1,
    "src/dl_techniques/layers/fastvit/conv_mlp.py": 1,
    "src/dl_techniques/layers/fastvit/patch_embed.py": 1,
    "src/dl_techniques/layers/fastvit/rep_conditional_pos_enc.py": 1,
    "src/dl_techniques/layers/fastvit/rep_mixer.py": 1,
    "src/dl_techniques/layers/fastvit/reparam_large_kernel_conv.py": 1,
    "src/dl_techniques/layers/fastvit/stage.py": 1,
    "src/dl_techniques/layers/generative/restricted_boltzmann_machine.py": 3,
    "src/dl_techniques/layers/generative/sampling.py": 1,
    "src/dl_techniques/layers/generative/sparse_autoencoder.py": 1,
    "src/dl_techniques/layers/generative/topographic_product.py": 1,
    "src/dl_techniques/layers/geometric/clifford_rnn.py": 2,
    "src/dl_techniques/layers/graphs/entity_graph_refinement.py": 1,
    "src/dl_techniques/layers/io_preparation.py": 1,
    "src/dl_techniques/layers/norms/band_rms.py": 1,
    "src/dl_techniques/layers/norms/bias_free_batch_norm.py": 1,
    "src/dl_techniques/layers/norms/dynamic_tanh.py": 1,
    "src/dl_techniques/layers/norms/energy_layer_norm.py": 1,
    "src/dl_techniques/layers/norms/zero_centered_adaptive_band_rms_norm.py": 1,
    "src/dl_techniques/layers/norms/zero_centered_band_rms_norm.py": 1,
    "src/dl_techniques/layers/norms/zero_centered_rms_norm.py": 1,
    "src/dl_techniques/layers/one_hot_encoding.py": 1,
    "src/dl_techniques/layers/physics/approximate_lagrange_layer.py": 1,
    "src/dl_techniques/layers/physics/lagrange_layer.py": 1,
    "src/dl_techniques/layers/pooling/blur_pool.py": 1,
    "src/dl_techniques/layers/pooling/global_sum_pool.py": 1,
    "src/dl_techniques/layers/pooling/patch_merging.py": 1,
    "src/dl_techniques/layers/pooling/pixel_unshuffle.py": 1,
    "src/dl_techniques/layers/regularization/film.py": 1,
    "src/dl_techniques/layers/regularization/layer_scale.py": 1,
    "src/dl_techniques/layers/regularization/router.py": 1,
    "src/dl_techniques/layers/statistics/deep_kernel_pca.py": 1,
    "src/dl_techniques/layers/statistics/invertible_kernel_pca.py": 1,
    "src/dl_techniques/layers/statistics/local_support_gate.py": 1,
    "src/dl_techniques/layers/statistics/mdn_layer.py": 1,
    "src/dl_techniques/layers/statistics/moving_std.py": 1,
    "src/dl_techniques/layers/statistics/normalizing_flow.py": 1,
    "src/dl_techniques/layers/statistics/residual_acf.py": 1,
    "src/dl_techniques/layers/statistics/scaler.py": 1,
    "src/dl_techniques/layers/thera_heat_field.py": 1,
    "src/dl_techniques/layers/time_series/adaptive_lag_attention.py": 1,
    "src/dl_techniques/layers/time_series/deepar_blocks.py": 1,
    "src/dl_techniques/layers/time_series/ema_layer.py": 1,
    "src/dl_techniques/layers/time_series/forecasting_layers.py": 1,
    "src/dl_techniques/layers/time_series/nbeats_blocks.py": 1,
    "src/dl_techniques/layers/time_series/nbeatsx_blocks.py": 1,
    "src/dl_techniques/layers/time_series/prism_blocks.py": 1,
    "src/dl_techniques/layers/time_series/quantile_head_fixed_io.py": 1,
    "src/dl_techniques/layers/time_series/temporal_convolutional_network.py": 1,
    "src/dl_techniques/layers/time_series/temporal_fusion.py": 1,
    "src/dl_techniques/layers/time_series/xlstm_blocks.py": 1,
    "src/dl_techniques/layers/transformers/adaln_zero.py": 1,
    "src/dl_techniques/layers/transformers/energy_transformer.py": 1,
    "src/dl_techniques/layers/transformers/eomt_transformer.py": 1,
    "src/dl_techniques/layers/transformers/free_transformer.py": 1,
    "src/dl_techniques/layers/transformers/gated_linear_attention_block.py": 1,
    "src/dl_techniques/layers/transformers/swin_conv_block.py": 1,
    "src/dl_techniques/layers/transformers/swin_transformer_block.py": 1,
    "src/dl_techniques/layers/transformers/transformer_decoder.py": 1,
    "src/dl_techniques/layers/yolo12/yolo12_blocks.py": 1,
    "src/dl_techniques/losses/any_loss.py": 1,
    "src/dl_techniques/losses/capsule_margin_loss.py": 1,
    "src/dl_techniques/losses/clifford_detection_loss.py": 1,
    "src/dl_techniques/losses/clustering_loss.py": 1,
    "src/dl_techniques/losses/decoupled_information_loss.py": 1,
    "src/dl_techniques/losses/dino_loss.py": 1,
    "src/dl_techniques/losses/feature_alignment_loss.py": 1,
    "src/dl_techniques/losses/flow_matching_velocity_loss.py": 1,
    "src/dl_techniques/losses/focal_causal_lm_loss.py": 1,
    "src/dl_techniques/losses/focal_uncertainty_loss.py": 1,
    "src/dl_techniques/losses/huber_loss.py": 1,
    "src/dl_techniques/losses/lpips_loss.py": 1,
    "src/dl_techniques/losses/mase_loss.py": 1,
    "src/dl_techniques/losses/masked_causal_lm_loss.py": 1,
    "src/dl_techniques/losses/multi_labels_loss.py": 1,
    "src/dl_techniques/losses/multi_task_loss.py": 1,
    "src/dl_techniques/losses/sam3_detection_loss.py": 1,
    "src/dl_techniques/losses/sam_mask_loss.py": 1,
    "src/dl_techniques/losses/scaled_mse_loss.py": 1,
    "src/dl_techniques/losses/siglip_contrastive_loss.py": 1,
    "src/dl_techniques/losses/smape_loss.py": 1,
    "src/dl_techniques/losses/superpoint_loss.py": 1,
    "src/dl_techniques/losses/tabm_loss.py": 1,
    "src/dl_techniques/losses/topographic_vae_loss.py": 1,
    "src/dl_techniques/losses/utilization_loss.py": 1,
    "src/dl_techniques/losses/wasserstein_loss.py": 1,
    "src/dl_techniques/metrics/capsule_accuracy.py": 1,
    "src/dl_techniques/metrics/clip_accuracy.py": 1,
    "src/dl_techniques/metrics/primary_output_metrics.py": 1,
    "src/dl_techniques/metrics/probabilistic_forecast_metrics.py": 1,
    "src/dl_techniques/metrics/psnr_metric.py": 1,
    "src/dl_techniques/metrics/sequence_metrics.py": 1,
    "src/dl_techniques/metrics/ssim_metric.py": 1,
    "src/dl_techniques/metrics/time_series_metrics.py": 1,
    "src/dl_techniques/models/graph/graph_energy_transformer/model.py": 2,
    "src/dl_techniques/models/language/blt/model.py": 1,
    "src/dl_techniques/models/language/mini_vec2vec/example_alignment.py": 1,
    "src/dl_techniques/models/language/mini_vec2vec/model.py": 1,
    "src/dl_techniques/models/language/tree_transformer/components.py": 1,
    "src/dl_techniques/models/memory/som/model.py": 1,
    "src/dl_techniques/models/neural_computer/nam/cell.py": 1,
    "src/dl_techniques/models/point_cloud/latent_gmm_registration/model.py": 1,
    "src/dl_techniques/models/tabular/tabm/model.py": 1,
    "src/dl_techniques/models/time_series/adaptive_ema/model.py": 1,
    "src/dl_techniques/models/time_series/deepar/model.py": 1,
    "src/dl_techniques/models/time_series/mdn/model.py": 2,
    "src/dl_techniques/models/time_series/nbeats/nbeats.py": 1,
    "src/dl_techniques/models/time_series/nbeats/nbeatsx.py": 1,
    "src/dl_techniques/models/time_series/prism/model.py": 1,
    "src/dl_techniques/models/time_series/tirex/model.py": 1,
    "src/dl_techniques/models/time_series/tirex/model_extended.py": 1,
    "src/dl_techniques/models/time_series/xlstm/forecaster.py": 1,
    "src/dl_techniques/models/time_series/xlstm/model.py": 1,
    "src/dl_techniques/models/vision/detr/model.py": 1,
    "src/dl_techniques/models/vision/dino/training.py": 1,
    "src/dl_techniques/models/vision/fastvit/model.py": 1,
    "src/dl_techniques/models/vision/image_restoration/pw_fnet/model.py": 1,
    "src/dl_techniques/models/world_models/lewm/model.py": 1,
    "src/dl_techniques/models/world_models/lewm/predictor.py": 1,
    "src/dl_techniques/models/vision/mambalct/model.py": 1,
    "src/dl_techniques/models/vision/masked_autoencoder/patch_masking.py": 1,
    "src/dl_techniques/models/vision/thera/hypernetwork.py": 1,
    "src/dl_techniques/models/vision/thera/rdn_backbone.py": 1,
    "src/dl_techniques/models/vision/thera/tails.py": 1,
    "src/dl_techniques/models/world_models/video_jepa/encoder.py": 1,
    "src/dl_techniques/models/world_models/video_jepa/masking.py": 1,
    "src/dl_techniques/models/world_models/video_jepa/model.py": 1,
    "src/dl_techniques/models/world_models/video_jepa/predictor.py": 1,
    "src/dl_techniques/models/vision/vq_vae/model.py": 1,
    "src/dl_techniques/models/vision/yolo12/feature_extractor.py": 1,
    "src/dl_techniques/models/vision_language/fastvlm/components.py": 1,
    "src/dl_techniques/models/vision_language/mobile_clip/components.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam1/image_encoder.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam1/mask_decoder.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam1/preprocessing.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam1/training_model.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam1/transformer.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam2/hiera.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam2/memory_attention.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam2/memory_bank.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam2/memory_encoder.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/maskformer_segmentation.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/model_misc.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/necks.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/query_selection.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/training_model.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/vitdet.py": 1,
    "src/dl_techniques/optimization/gefen_optimizer.py": 1,
    "src/dl_techniques/optimization/muon_optimizer.py": 1,
    "src/dl_techniques/optimization/schedule.py": 1,
    "src/dl_techniques/optimization/sgld_optimizer.py": 1,
    "src/dl_techniques/optimization/sled_supervision.py": 1,
    "src/dl_techniques/optimization/vsgd_optimizer.py": 1,
    "src/dl_techniques/regularizers/entropy_regularizer.py": 1,
    "src/dl_techniques/regularizers/l2_custom.py": 1,
    "src/dl_techniques/regularizers/sigreg.py": 1,
    "src/dl_techniques/regularizers/srip.py": 1,
    "src/dl_techniques/utils/bounding_box.py": 1,
    "src/dl_techniques/utils/clip_utils.py": 1,
    "src/dl_techniques/utils/corruption.py": 1,
    "src/dl_techniques/utils/masking/factory.py": 1,
    "src/train/common/token_superposition.py": 1,
    "src/train/logic/clevr_hans_data.py": 1,
    "src/train/sam3/baselines.py": 1,
    "src/train/sam3/data.py": 1,
    "tests/test_callbacks/test_topographic_callbacks.py": 1,
    "tests/test_constraints/test_value_range_constraint.py": 1,
    "tests/test_initializers/test_clone_initializer.py": 1,
    "tests/test_initializers/test_he_orthonormal_initializer.py": 1,
    "tests/test_initializers/test_identity_plus_noise.py": 1,
    "tests/test_initializers/test_kan_initializer.py": 1,
    "tests/test_initializers/test_linear_up_initializer.py": 1,
    "tests/test_initializers/test_orthonormal_initializer.py": 1,
    "tests/test_layers/test_acc_unet/test_mlfc_layers.py": 1,
    "tests/test_layers/test_activations/test_gelu_tanh.py": 1,
    "tests/test_layers/test_activations/test_golu.py": 1,
    "tests/test_layers/test_activations/test_hard_activations.py": 1,
    "tests/test_layers/test_activations/test_probability_output.py": 1,
    "tests/test_layers/test_activations/test_relu_k.py": 1,
    "tests/test_layers/test_activations/test_sparsemax.py": 1,
    "tests/test_layers/test_activations/test_squash.py": 1,
    "tests/test_layers/test_attention/test_anchor_attention.py": 1,
    "tests/test_layers/test_attention/test_capsule_routing_attention.py": 1,
    "tests/test_layers/test_attention/test_channel_attention_layer.py": 1,
    "tests/test_layers/test_attention/test_common.py": 1,
    "tests/test_layers/test_attention/test_convolutional_block_attention.py": 1,
    "tests/test_layers/test_attention/test_gated_attention.py": 1,
    "tests/test_layers/test_attention/test_rpc_attention.py": 1,
    "tests/test_layers/test_attention/test_shared_weights_cross_attention.py": 1,
    "tests/test_layers/test_attention/test_window_attention_restructure_is_inert.py": 1,
    "tests/test_layers/test_capsule_roll.py": 1,
    "tests/test_layers/test_embedding/test_bert_embedding.py": 1,
    "tests/test_layers/test_embedding/test_modern_bert_embedding.py": 1,
    "tests/test_layers/test_embedding/test_patch_embed_3d.py": 1,
    "tests/test_layers/test_embedding/test_patch_embedding.py": 1,
    "tests/test_layers/test_embedding/test_video_rope.py": 1,
    "tests/test_layers/test_fastvit/test_attention_block.py": 1,
    "tests/test_layers/test_fastvit/test_conv_mlp.py": 1,
    "tests/test_layers/test_fastvit/test_patch_embed.py": 1,
    "tests/test_layers/test_fastvit/test_rep_conditional_pos_enc.py": 1,
    "tests/test_layers/test_fastvit/test_rep_mixer.py": 1,
    "tests/test_layers/test_fastvit/test_reparam_large_kernel_conv.py": 1,
    "tests/test_layers/test_fastvit/test_stage.py": 1,
    "tests/test_layers/test_ffn/test_gated_mlp.py": 1,
    "tests/test_layers/test_ffn/test_kan_linear.py": 1,
    "tests/test_layers/test_ffn/test_tversky_projection.py": 1,
    "tests/test_layers/test_generative/test_vector_quantizer.py": 1,
    "tests/test_layers/test_generative/test_vector_quantizer_rotation_trick.py": 1,
    "tests/test_layers/test_geometric/test_clifford_rnn.py": 2,
    "tests/test_layers/test_graphs/test_entity_graph_refinement.py": 1,
    "tests/test_layers/test_heads/test_vision.py": 1,
    "tests/test_layers/test_logic/test_arithmetic_operators.py": 2,
    "tests/test_layers/test_logic/test_factory.py": 1,
    "tests/test_layers/test_logic/test_logic_operators.py": 2,
    "tests/test_layers/test_logic/test_neural_circuit.py": 1,
    "tests/test_layers/test_mixtures/cluster_axis_oracle.py": 1,
    "tests/test_layers/test_mixtures/test_gmm.py": 1,
    "tests/test_layers/test_mixtures/test_kmeans.py": 1,
    "tests/test_layers/test_mixtures/test_radial_basis_function.py": 1,
    "tests/test_layers/test_norms/test_adaptive_band_rms.py": 1,
    "tests/test_layers/test_norms/test_band_logit_norm.py": 1,
    "tests/test_layers/test_norms/test_band_rms.py": 1,
    "tests/test_layers/test_norms/test_factory.py": 1,
    "tests/test_layers/test_norms/test_local_response_norm.py": 1,
    "tests/test_layers/test_norms/test_logit_norm.py": 1,
    "tests/test_layers/test_norms/test_max_logit_norm.py": 1,
    "tests/test_layers/test_norms/test_polar_weight_norm.py": 1,
    "tests/test_layers/test_norms/test_zero_centered_adaptive_band_rms_norm.py": 1,
    "tests/test_layers/test_power_mlp_branches_start_different.py": 1,
    "tests/test_layers/test_reasoning/test_hierarchical_reasoning_core.py": 1,
    "tests/test_layers/test_reasoning/test_hierarchical_reasoning_module.py": 1,
    "tests/test_layers/test_regularization/test_router.py": 1,
    "tests/test_layers/test_signal_processing/test_haar_wavelet_decomposition.py": 1,
    "tests/test_layers/test_signal_processing/test_shearlet_transform.py": 1,
    "tests/test_layers/test_spatial_attention_layer.py": 1,
    "tests/test_layers/test_statistics/test_deep_kernel_pca.py": 1,
    "tests/test_layers/test_statistics/test_invertible_kernel_pca.py": 1,
    "tests/test_layers/test_statistics/test_normalizing_flows.py": 1,
    "tests/test_layers/test_statistics/test_scaler.py": 1,
    "tests/test_layers/test_statistics/test_the_four_carried_defects_are_fixed.py": 1,
    "tests/test_layers/test_statistics/test_the_v2_oracle_can_fail.py": 1,
    "tests/test_layers/test_structured_linear/test_crystal_extreme_scales.py": 1,
    "tests/test_layers/test_structured_linear/test_crystal_training.py": 1,
    "tests/test_layers/test_structured_linear/test_kanvolution.py": 1,
    "tests/test_layers/test_structured_linear/test_mps_layer.py": 1,
    "tests/test_layers/test_structured_linear/test_orthogonal_butterfly.py": 1,
    "tests/test_layers/test_the_shared_initializer_symmetry_is_broken.py": 1,
    "tests/test_layers/test_thera_heat_field.py": 1,
    "tests/test_layers/test_time_series/test_xlstm_blocks.py": 1,
    "tests/test_layers/test_transformers/test_adaln_zero.py": 1,
    "tests/test_layers/test_transformers/test_perceiver_transformer.py": 1,
    "tests/test_layers/test_transformers/test_swin_conv_block.py": 1,
    "tests/test_layers/test_transformers/test_text_decoder.py": 1,
    "tests/test_layers/test_transformers/test_text_encoder.py": 1,
    "tests/test_layers/test_transformers/test_the_encoder_mask_is_the_explicit_one.py": 1,
    "tests/test_layers/test_transformers/test_transformer_beit_integration.py": 1,
    "tests/test_layers/test_transformers/test_transformer_decoder.py": 1,
    "tests/test_layers/test_transformers/test_transformer_new_ffn_types.py": 1,
    "tests/test_layers/test_transformers/test_vision_encoder.py": 1,
    "tests/test_losses/test_any_loss.py": 1,
    "tests/test_losses/test_capsule_margin_loss.py": 1,
    "tests/test_losses/test_clip_contrastive_loss.py": 1,
    "tests/test_losses/test_dino_loss.py": 1,
    "tests/test_losses/test_lpips_loss.py": 1,
    "tests/test_losses/test_multi_labels_loss.py": 1,
    "tests/test_losses/test_multi_task_loss.py": 1,
    "tests/test_losses/test_omnipoint_losses.py": 1,
    "tests/test_losses/test_sam_mask_loss.py": 1,
    "tests/test_losses/test_scaled_mse_loss.py": 1,
    "tests/test_losses/test_superpoint_loss.py": 1,
    "tests/test_losses/test_tabm_loss.py": 1,
    "tests/test_losses/test_topographic_vae_loss.py": 1,
    "tests/test_losses/test_utilization_loss.py": 1,
    "tests/test_losses/test_wasserstein.py": 1,
    "tests/test_metrics/test_capsule_accuracy.py": 1,
    "tests/test_metrics/test_clip_accuracy.py": 1,
    "tests/test_metrics/test_primary_output_metrics.py": 1,
    "tests/test_metrics/test_sequence_metrics.py": 1,
    "tests/test_metrics/test_ssim_metric.py": 1,
    "tests/test_metrics/test_topographic.py": 1,
    "tests/test_models/lazy_build_contract_oracle.py": 1,
    "tests/test_models/precision_arm_oracle.py": 1,
    "tests/test_models/smoke_contract_oracle.py": 1,
    "tests/test_models/test_accunet/test_the_xla_opt_out_holds.py": 1,
    "tests/test_models/test_adaptive_ema/test_model.py": 1,
    "tests/test_models/test_adaptive_ema/test_quantile_head_round_trip.py": 1,
    "tests/test_models/test_beit/test_architecture_invariants.py": 1,
    "tests/test_models/test_beit/test_masked_image_modeling_head.py": 1,
    "tests/test_models/test_beit/test_model.py": 1,
    "tests/test_models/test_beit/test_warm_start.py": 1,
    "tests/test_models/test_bert/test_precision_arm.py": 1,
    "tests/test_models/test_bert/test_predict_single_key_dict.py": 1,
    "tests/test_models/test_blt/test_custom_entropy_model_roundtrip.py": 1,
    "tests/test_models/test_blt/test_fit_runs_under_xla.py": 2,
    "tests/test_models/test_blt/test_max_patch_pooling_empty_patches.py": 1,
    "tests/test_models/test_deepar/test_sampling_and_likelihood.py": 1,
    "tests/test_models/test_dino/test_head_unit_norm_invariant.py": 1,
    "tests/test_models/test_distilbert/test_predict_single_key_dict.py": 1,
    "tests/test_models/test_fnet/test_predict_single_key_dict.py": 1,
    "tests/test_models/test_hrm/test_positional_signal.py": 1,
    "tests/test_models/test_lewm.py": 1,
    "tests/test_models/test_mamba/test_build_and_reload.py": 1,
    "tests/test_models/test_mamba/test_dt_proj_survives_stateless_build.py": 1,
    "tests/test_models/test_mamba/test_mamba2_group_routing.py": 1,
    "tests/test_models/test_masked_autoencoder/test_patch_masking.py": 1,
    "tests/test_models/test_lm_pretraining/test_clm.py": 1,
    "tests/test_models/test_mini_vec2vec/test_smoke.py": 1,
    "tests/test_models/test_mobile_clip/test_mobile_clip_v2.py": 1,
    "tests/test_models/test_modern_bert/test_positional_signal.py": 1,
    "tests/test_models/test_modern_bert/test_predict_single_key_dict.py": 1,
    "tests/test_models/test_nam/test_fp16_reduction_mask.py": 1,
    "tests/test_models/test_nbeats/test_model.py": 1,
    "tests/test_models/test_ntm/test_model.py": 1,
    "tests/test_models/test_prism/test_model.py": 2,
    "tests/test_models/test_pw_fnet/test_model.py": 1,
    "tests/test_models/test_pw_fnet/test_scale_contract.py": 1,
    "tests/test_models/test_qwen/test_components.py": 1,
    "tests/test_models/test_qwen/test_precision_arm.py": 1,
    "tests/test_models/test_roundtrip_instrument_oracle.py": 1,
    "tests/test_models/test_sam/dead_component_oracle.py": 1,
    "tests/test_models/test_sam/test_precision_arm.py": 1,
    "tests/test_models/test_sam/test_training_model.py": 1,
    "tests/test_models/test_sam2/test_hiera.py": 1,
    "tests/test_models/test_sam2/test_mask_decoder.py": 1,
    "tests/test_models/test_sam2/test_memory.py": 1,
    "tests/test_models/test_sam2/test_memory_bank.py": 1,
    "tests/test_models/test_sam2/test_neck.py": 1,
    "tests/test_models/test_sam3/test_decoder.py": 1,
    "tests/test_models/test_sam3/test_model_misc.py": 1,
    "tests/test_models/test_sam3/test_query_selection.py": 1,
    "tests/test_models/test_sam3/test_seg_head.py": 1,
    "tests/test_models/test_sam3/test_seg_head_mask_dim.py": 1,
    "tests/test_models/test_sam3/test_training_model.py": 1,
    "tests/test_models/test_sam3/test_vitdet.py": 1,
    "tests/test_models/test_sam3/test_vitdet_last_block_is_global.py": 1,
    "tests/test_models/test_sd3_mmdit/test_text_encoders.py": 1,
    "tests/test_models/test_shgcn/test_precision_arm.py": 1,
    "tests/test_models/test_shgcn/test_round_trip.py": 1,
    "tests/test_models/test_som/test_model.py": 1,
    "tests/test_models/test_som/test_second_train_and_batched_inference.py": 3,
    "tests/test_models/test_superpoint/test_precision_arm.py": 1,
    "tests/test_models/test_tabm/test_round_trip.py": 1,
    "tests/test_models/test_tirex/test_model.py": 1,
    "tests/test_models/test_tirex/test_model_extended.py": 1,
    "tests/test_models/test_trm/test_attention_type_contract.py": 1,
    "tests/test_models/test_trm/test_positional_signal.py": 1,
    "tests/test_models/test_trm/test_trm.py": 1,
    "tests/test_models/test_video_jepa/test_horizon_loss_uses_batch_frames.py": 1,
    "tests/test_models/test_video_jepa/test_stream_head_routing.py": 1,
    "tests/test_models/test_vq_vae/test_model.py": 1,
    "tests/test_models/test_vq_vae_rotation/test_vq_vae_rotation.py": 1,
    "tests/test_models/test_xlstm/test_model.py": 1,
    "tests/test_models/test_yolo12/test_round_trip.py": 1,
    "tests/test_optimization/test_gefen_optimizer.py": 1,
    "tests/test_optimization/test_muon_optimizer.py": 1,
    "tests/test_optimization/test_sgld_optimizer.py": 1,
    "tests/test_optimization/test_vsgd_optimizer.py": 1,
    "tests/test_regularizers/test_binary_preference.py": 2,
    "tests/test_regularizers/test_entropy_regularizer.py": 2,
    "tests/test_regularizers/test_l2_custom.py": 2,
    "tests/test_regularizers/test_sigreg.py": 1,
    "tests/test_regularizers/test_tri_state_preference.py": 2,
    "tests/test_serialization_registry.py": 1,
    "tests/test_the_ported_numeric_defaults_match_their_references.py": 1,
    "tests/test_train/test_logic/test_train_benchmark.py": 1,
    "tests/test_train/test_sam/test_data.py": 1,
    "tests/test_train/test_sam3/test_baselines.py": 3,
    "tests/test_train/test_topographic_vae/test_evaluate_model.py": 1,
    "tests/test_utils/test_bbox.py": 1,
    "tests/test_utils/test_corruption.py": 1,
    "tests/test_utils/test_dtype_policy.py": 1,
    "tests/test_utils/test_l2_normalize.py": 1,
    "tests/test_utils/test_windows.py": 1,
}


def keras_submodule_imports(tree: ast.AST) -> int:
    """Count the statements in ``tree`` that bind a keras submodule name.

    :param tree: A parsed module.
    :return: Number of ``from keras[.x] import ...`` (absolute) and
        ``import keras.x`` statements.
    """
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "keras" or module.startswith("keras.")):
                count += 1
        elif isinstance(node, ast.Import):
            if any(alias.name.startswith("keras.") for alias in node.names):
                count += 1
    return count


def _census():
    per_file = {}
    for root in ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            with warnings.catch_warnings():
                # Invalid escapes in other files' strings are not this census's business.
                warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
                tree = ast.parse(path.read_text(encoding="utf-8"))
            n = keras_submodule_imports(tree)
            if n:
                per_file[path.relative_to(REPO_ROOT).as_posix()] = n
    return per_file


@pytest.fixture(scope="module")
def census():
    return _census()


def test_the_census_is_exact(census):
    if census == KERAS_SUBMODULE_IMPORTS:
        return
    statements, files = sum(census.values()), len(census)
    added = sorted(set(census) - set(KERAS_SUBMODULE_IMPORTS))
    gone = sorted(set(KERAS_SUBMODULE_IMPORTS) - set(census))
    changed = sorted(
        f"{path}: {KERAS_SUBMODULE_IMPORTS[path]} -> {census[path]}"
        for path in set(census) & set(KERAS_SUBMODULE_IMPORTS)
        if census[path] != KERAS_SUBMODULE_IMPORTS[path])
    pytest.fail(
        f"{statements} keras-submodule import statements in {files} files; the "
        f"census pins {sum(KERAS_SUBMODULE_IMPORTS.values())} in "
        f"{len(KERAS_SUBMODULE_IMPORTS)}. New files: {[(p, census[p]) for p in added]}; "
        f"converted files: {gone}; changed: {changed}. A conversion: delete its "
        f"line from KERAS_SUBMODULE_IMPORTS, lower the ceiling to {statements}. A "
        "new statement: use `import keras` and qualify the call site "
        "(keras.ops.*, keras.layers.*)."
    )


def test_the_census_totals(census):
    assert (sum(KERAS_SUBMODULE_IMPORTS.values()), len(KERAS_SUBMODULE_IMPORTS)) == (418, 399), (
        "the census's own totals moved; re-derive the module docstring's numbers")
    for path in KERAS_SUBMODULE_IMPORTS:
        assert (REPO_ROOT / path).is_file(), f"census file moved or deleted: {path}"


@pytest.mark.parametrize("source, expected", [
    ("import keras\n", 0),
    ("from keras import ops\n", 1),
    ("from keras import ops, layers\n", 1),
    ("import keras.ops\n", 1),
    ("from keras.src import backend\n", 1),
    ("def f():\n    from keras import ops\n", 1),
    ('"""from keras import ops"""\n', 0),
    ("# from keras import ops\n", 0),
    ("from .keras import ops\n", 0),
])
def test_the_counter_on_planted_cases(source, expected):
    assert keras_submodule_imports(ast.parse(source)) == expected
