"""
Comprehensive test suite for the Mixture of Experts (MoE) module.

This module provides extensive testing for all MoE components to ensure
reliability, correctness, and compatibility with the dl_techniques framework.
"""

import os
import keras
import pytest
import tempfile
import numpy as np
from keras import ops

from dl_techniques.layers.moe import (
    MixtureOfExperts, MoEConfig, ExpertConfig, GatingConfig,
    create_ffn_moe, create_attention_moe, create_conv_moe,
    FFNExpert, AttentionExpert, Conv2DExpert,
    LinearGating, CosineGating, SoftMoEGating,
    get_preset_moe
)

