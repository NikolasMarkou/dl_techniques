"""General-purpose power sampling for any causal LLM/VLM + any tokenizer."""
from dl_techniques.models.power_sampling.config import PowerSamplingConfig
from dl_techniques.models.power_sampling.protocols import TokenizerProtocol, LogitsFn
from dl_techniques.models.power_sampling.ops import _log_softmax, _nucleus_sample
from dl_techniques.models.power_sampling.forward import (
    make_logits_fn,
    make_batch_logits_fn,
    VLMForwardAdapter,
)
from dl_techniques.models.power_sampling.sampler import PowerSampler

__all__ = [
    "PowerSampler",
    "PowerSamplingConfig",
    "TokenizerProtocol",
    "LogitsFn",
    "make_logits_fn",
    "make_batch_logits_fn",
    "VLMForwardAdapter",
    "_log_softmax",
    "_nucleus_sample",
]
