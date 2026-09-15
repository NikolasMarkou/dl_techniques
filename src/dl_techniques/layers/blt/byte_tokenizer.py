"""ByteTokenizer: convert text strings to and from byte token sequences, for
the Byte Latent Transformer (BLT).

Operates at the byte level, so there is no fixed subword vocabulary and no
out-of-vocabulary case: any UTF-8 text round-trips through ``text_to_bytes`` /
``tokens_to_text``. The layer has no ``call`` and no weights; both directions
are plain Python over lists of ints.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.blt.byte_tokenizer")
class ByteTokenizer(keras.layers.Layer):
    """Convert text strings to and from byte token sequences.

    Operates at the byte level, so there is no fixed subword vocabulary and no
    out-of-vocabulary case: any UTF-8 text round-trips through
    ``text_to_bytes`` / ``tokens_to_text``. The layer has no ``call`` and no
    weights; both directions are plain Python over lists of ints.

    Architecture:

    .. code-block:: text

        "Hello"
              │
              ▼
        ┌──────────────────────────────┐
        │ utf-8 encode                 │
        └──────────────────────────────┘
              │ 72, 101, 108, 108, 111
              ▼
        ┌──────────────────────────────┐
        │ add byte_offset              │
        └──────────────────────────────┘
              │ 76, 105, 112, 112, 115
              ▼
        ┌──────────────────────────────┐
        │ prepend bos, append eos      │
        └──────────────────────────────┘
              │
              ▼
        [1, 76, 105, 112, 112, 115, 2]

    Special ids are fixed at pad 0, bos 1, eos 2 and sep 3, so a
    ``byte_offset`` below 4 would collide with them. Nothing checks this, and
    nothing checks a token against ``vocab_size``, which is carried for the
    config only.

    :param vocab_size: Size of the vocabulary including special tokens. Stored
        for serialization; no method reads it.
    :type vocab_size: int
    :param byte_offset: Offset added to raw byte values, reserving IDs below
        it for special tokens (pad, BOS, EOS, sep).
    :type byte_offset: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            byte_offset: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset

        # A byte_offset at or below 3 would collide with these four ids.
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3

    def text_to_bytes(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a text string to a byte token sequence.

        Undecodable input is dropped rather than raising, since the encode uses
        ``errors='ignore'``.

        :param text: Input text string.
        :type text: str
        :param add_bos: Whether to prepend the begin-of-sequence token.
        :type add_bos: bool
        :param add_eos: Whether to append the end-of-sequence token.
        :type add_eos: bool
        :return: List of byte token IDs.
        :rtype: List[int]
        """
        byte_sequence = text.encode('utf-8', errors='ignore')

        tokens = [byte + self.byte_offset for byte in byte_sequence]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert a byte token sequence back to text.

        Tokens below ``byte_offset`` are dropped, which removes the special
        ids. A token that leaves a value above 255 after the offset is removed
        cannot form a byte, and the whole call then returns an empty string.

        :param tokens: List of byte token IDs.
        :type tokens: List[int]
        :return: Decoded text string, empty if the byte values are not a valid
            sequence.
        :rtype: str
        """
        byte_values = []
        for token in tokens:
            if token >= self.byte_offset:
                byte_values.append(token - self.byte_offset)

        try:
            text = bytes(byte_values).decode('utf-8', errors='ignore')
        except (ValueError, UnicodeDecodeError):
            text = ""

        return text

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        The sequence dimension is dynamic, since output length depends on
        text length.

        :param input_shape: Input shape tuple (ignored for this utility layer).
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, None)``.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
            return (input_shape[0], None)
        return (None, None)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'byte_offset': self.byte_offset
        })
        return config
