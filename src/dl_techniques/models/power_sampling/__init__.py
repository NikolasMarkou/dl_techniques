"""General-purpose power sampling for any causal LLM/VLM + any tokenizer.

**There is no model class here, and no ``create_*`` factory was added.** This
package is an inference-time *decoding* algorithm, not an architecture: it holds
a sampler (``PowerSampler``), its config, the model-call indirection that turns
any causal LM/VLM into a ``LogitsFn``, and two numeric helpers. Nothing here
subclasses ``keras.Model``, so there is nothing for a model factory to build --
inventing one would fabricate an architecture the package does not have. Build
your model with its own package's factory and pass it to ``make_logits_fn``.

``_log_softmax`` and ``_nucleus_sample`` keep their leading underscore (they are
internal by name) but are exported because the test suite and the sampler's
callers pin their exact numerics. ``_nucleus_sample`` returns
``(token_id, log_prob)`` -- the log probability is taken over the truncated and
renormalized nucleus, which is the density the token was drawn from and the one
the MH acceptance ratio needs; the caller cannot recover it from a
full-vocabulary log-softmax.
"""
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
