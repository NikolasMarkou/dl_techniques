"""Duck-typed interfaces that decouple the power sampler from any concrete model or tokenizer.

The power-sampling engine never imports a specific model class or a specific
tokenizer library. It depends only on the small, structural contracts
defined here:

- :class:`TokenizerProtocol` — anything exposing ``encode``/``decode`` can
  drive the sampler; those are the only two methods it calls.
  ``vocab_size`` and ``special_token_ids`` are optional on the tokenizer
  object, supplied instead through
  :class:`~dl_techniques.models.common.power_sampling.config.PowerSamplingConfig`.
- :data:`LogitsFn` — a callable mapping a token-id array to a single logit
  vector for the target position, letting any causal LM/VLM be injected as
  a closure.

These are ``typing.Protocol`` interfaces, so callers pass plain duck-typed
objects with no inheritance requirement, and the package stays free of
TensorFlow/tiktoken imports at module load.
"""

from typing import Callable, List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Structural interface for any tokenizer usable with the power sampler.

    Only :meth:`encode` and :meth:`decode` are required — they are the sole
    tokenizer methods the sampler calls. Token-vocabulary metadata such as
    ``vocab_size`` and ``special_token_ids`` is not part of this contract;
    it is provided via
    :class:`~dl_techniques.models.common.power_sampling.config.PowerSamplingConfig`
    so that tokenizers lacking those attributes still satisfy the protocol.
    """

    def encode(self, text: str) -> List[int]:
        """Encode ``text`` into a list of integer token IDs."""
        ...

    def decode(self, ids: List[int]) -> str:
        """Decode a list of integer token IDs back into text."""
        ...


# Maps an int32[1, T] (or int32[T]) token array to a float32[V] logit vector for
# the last/target position. Any causal LM/VLM is injected as such a closure.
LogitsFn = Callable[[np.ndarray], np.ndarray]


__all__ = ["TokenizerProtocol", "LogitsFn"]
