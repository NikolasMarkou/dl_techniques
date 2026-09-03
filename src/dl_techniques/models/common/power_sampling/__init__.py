"""General-purpose power sampling for any causal LLM/VLM, with any tokenizer.

This package holds an inference-time decoding algorithm, not a model
architecture: :class:`PowerSampler` and its :class:`PowerSamplingConfig`, the
model-call indirection that turns any causal LM/VLM into a ``LogitsFn``, and
two numeric helpers. Nothing here subclasses ``keras.Model``, so there is no
``create_*`` factory. Build the model with its own package's factory and pass
it to ``make_logits_fn``.

``_log_softmax`` and ``_nucleus_sample`` keep their leading underscore (they
are internal helpers) but are exported because the test suite and the
sampler's callers pin their exact numerics. ``_nucleus_sample`` returns
``(token_id, log_prob)``: the log probability of the drawn token under the
truncated, renormalized nucleus it was sampled from, which the caller cannot
recover from a full-vocabulary log-softmax.
"""
from dl_techniques.models.common.power_sampling.config import PowerSamplingConfig
from dl_techniques.models.common.power_sampling.protocols import TokenizerProtocol, LogitsFn
from dl_techniques.models.common.power_sampling.ops import _log_softmax, _nucleus_sample
from dl_techniques.models.common.power_sampling.forward import (
    make_logits_fn,
    make_batch_logits_fn,
    VLMForwardAdapter,
)
from dl_techniques.models.common.power_sampling.sampler import PowerSampler

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
