"""
Character-level ASCII tokenizer for English-only text.

This module provides a fixed, data-independent character vocabulary of 101
ids: six special tokens followed by the 95 printable ASCII characters. It
exists for models that consume text as characters rather than sub-word
pieces, where a learned sub-word vocabulary would contribute a large
embedding table that carries most of the parameter budget and receives the
sparsest gradients.

Two surfaces are provided, deliberately as two classes:

- :class:`ASCIICharTokenizer` -- a serializable ``keras.layers.Layer``,
  matching the house pattern of :class:`~dl_techniques.layers.tokenizers.bpe.BPETokenizer`.
  It is a configuration container and eager Python tokenizer; ``call()``
  raises, because string operations are not backend-agnostic.
- :class:`ASCIICharPreprocessor` -- a plain Python object matching the
  ``__call__(texts, return_tensors=...)`` contract of
  :class:`~dl_techniques.utils.tokenizer.TiktokenPreprocessor`, so it can be
  passed as the ``preprocessor`` argument of ``train.common.nlp``'s
  ``preprocess_mlm_dataset`` / ``preprocess_classification_dataset``.

The two are kept separate on purpose. Overriding ``__call__`` on a
``keras.layers.Layer`` would shadow Keras' own dispatch (build, dtype
casting, masking) rather than add to it, so the preprocessor contract is
served by an object that is not a Layer. Both delegate to the module-level
:func:`encode_ascii` / :func:`decode_ascii`, which are the single
implementation of the mapping.

Vocabulary layout (fixed; ``VOCAB_SIZE`` is 101)::

    0  [PAD]      padding sentinel; never emitted by encode_ascii
    1  [CLS]      sequence-start / pooling anchor
    2  [SEP]      sequence separator and terminator
    3  [MASK]     masked-language-model sentinel
    4  [UNK]      any character not representable after normalization
    5  [NL]       newline, so document structure survives packing
    6..100        printable ASCII 32..126, id = ord(c) - 32 + 6

Two properties of this layout are load-bearing and are pinned by tests:

1. ``encode_ascii`` can never emit a special id. Printable ids start at 6
   and unknown characters map to ``[UNK]`` (4), never to ``[PAD]`` (0). A
   consumer that derives a sequence length by counting non-pad positions --
   as masked pooling and last-token gathers do -- is therefore correct by
   construction rather than by convention.
2. Ids are independent of any corpus. Unlike a trained BPE vocabulary, the
   mapping is a pure function of the character, so a checkpoint's embedding
   table is meaningful without shipping a vocabulary file alongside it.

Non-ASCII input is NFKD-folded first, so ``e`` with an acute accent becomes
``e`` rather than ``[UNK]``. Whatever survives the fold becomes ``[UNK]``;
characters are never silently dropped, because dropping breaks the
positional alignment that makes a round-trip test meaningful.
"""

import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------

ENCODING_NAME: str = "ascii95"

PAD_ID: int = 0
CLS_ID: int = 1
SEP_ID: int = 2
MASK_ID: int = 3
UNK_ID: int = 4
NEWLINE_ID: int = 5

#: Number of reserved special ids preceding the printable range.
NUM_SPECIAL_TOKENS: int = 6

#: First and last printable ASCII code points covered by the vocabulary.
FIRST_PRINTABLE: int = 32
LAST_PRINTABLE: int = 126

#: Total vocabulary size: 6 specials + 95 printable ASCII characters.
VOCAB_SIZE: int = NUM_SPECIAL_TOKENS + (LAST_PRINTABLE - FIRST_PRINTABLE + 1)

SPECIAL_TOKEN_IDS: tuple = (PAD_ID, CLS_ID, SEP_ID, MASK_ID, UNK_ID, NEWLINE_ID)

_SPECIAL_TOKEN_NAMES: Dict[int, str] = {
    PAD_ID: "[PAD]",
    CLS_ID: "[CLS]",
    SEP_ID: "[SEP]",
    MASK_ID: "[MASK]",
    UNK_ID: "[UNK]",
    NEWLINE_ID: "\n",
}


def char_to_id(char: str) -> int:
    """Map a single character to its vocabulary id.

    Assumes ``char`` has already been case-folded and unicode-normalized by
    the caller; this function performs no normalization of its own so that
    it stays a pure lookup.

    :param char: A single character.
    :type char: str
    :return: The vocabulary id, in ``[0, VOCAB_SIZE)``. Never a padding id.
    :rtype: int
    """
    if char == "\n":
        return NEWLINE_ID
    if char in ("\t", "\r"):
        # Collapse the remaining common whitespace onto a literal space
        # rather than onto [UNK]: the distinction carries no signal at
        # character granularity and [UNK] would be a strictly worse anchor.
        return ord(" ") - FIRST_PRINTABLE + NUM_SPECIAL_TOKENS
    code_point = ord(char)
    if FIRST_PRINTABLE <= code_point <= LAST_PRINTABLE:
        return code_point - FIRST_PRINTABLE + NUM_SPECIAL_TOKENS
    return UNK_ID


def id_to_char(token_id: int) -> str:
    """Map a vocabulary id back to its character or special-token name.

    :param token_id: A vocabulary id.
    :type token_id: int
    :return: The decoded character; a bracketed name for a special token,
        except ``NEWLINE_ID`` which decodes to an actual newline.
    :rtype: str
    :raises ValueError: If ``token_id`` is outside ``[0, VOCAB_SIZE)``.
    """
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValueError(
            f"token_id must be in [0, {VOCAB_SIZE}), got {token_id}"
        )
    if token_id in _SPECIAL_TOKEN_NAMES:
        return _SPECIAL_TOKEN_NAMES[token_id]
    return chr(token_id - NUM_SPECIAL_TOKENS + FIRST_PRINTABLE)


def normalize_text(
    text: str,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """Apply the case and unicode normalization used before id lookup.

    :param text: Input text.
    :type text: str
    :param lowercase: Whether to case-fold. Defaults to ``True``.
    :type lowercase: bool
    :param normalize_unicode: Whether to NFKD-normalize and drop combining
        marks, folding accented Latin characters onto their ASCII base.
        Defaults to ``True``.
    :type normalize_unicode: bool
    :return: The normalized text.
    :rtype: str
    """
    if normalize_unicode:
        decomposed = unicodedata.normalize("NFKD", text)
        text = "".join(
            ch for ch in decomposed if not unicodedata.combining(ch)
        )
    if lowercase:
        text = text.lower()
    return text


def encode_ascii(
    text: str,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> List[int]:
    """Encode text to content ids, without any framing tokens.

    No ``[CLS]``/``[SEP]`` is added and no padding is applied; framing is
    the caller's responsibility. This mirrors ``tiktoken``'s ``encode``, so
    the two are interchangeable at call sites that do their own framing.

    :param text: Input text. ``bytes`` are decoded as utf-8, errors ignored.
    :type text: str
    :param lowercase: Whether to case-fold. Defaults to ``True``.
    :type lowercase: bool
    :param normalize_unicode: Whether to NFKD-fold accents. Defaults to
        ``True``.
    :type normalize_unicode: bool
    :return: Content ids. Never contains ``PAD_ID``, ``CLS_ID``, ``SEP_ID``
        or ``MASK_ID``.
    :rtype: list[int]
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    normalized = normalize_text(
        text, lowercase=lowercase, normalize_unicode=normalize_unicode
    )
    return [char_to_id(ch) for ch in normalized]


def decode_ascii(
    token_ids: Sequence[int],
    skip_special_tokens: bool = True,
) -> str:
    """Decode ids back to text.

    :param token_ids: Ids to decode.
    :type token_ids: Sequence[int]
    :param skip_special_tokens: Whether to omit ``[PAD]``/``[CLS]``/
        ``[SEP]``/``[MASK]``/``[UNK]`` from the output. ``NEWLINE_ID``
        always decodes to a newline, since it is a real character rather
        than a control token. Defaults to ``True``.
    :type skip_special_tokens: bool
    :return: The decoded string.
    :rtype: str
    """
    pieces: List[str] = []
    for raw_id in token_ids:
        token_id = int(raw_id)
        if token_id == NEWLINE_ID:
            pieces.append("\n")
            continue
        if skip_special_tokens and token_id in _SPECIAL_TOKEN_NAMES:
            continue
        pieces.append(id_to_char(token_id))
    return "".join(pieces)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.tokenizers.ascii_char")
class ASCIICharTokenizer(keras.layers.Layer):
    """
    Character-level ASCII tokenizer layer.

    A serializable configuration container and eager Python tokenizer over
    the fixed 101-id vocabulary documented in this module. It holds no
    weights and no learned vocabulary, so ``get_config`` carries only the
    three behavioural knobs.

    Like :class:`~dl_techniques.layers.tokenizers.bpe.BPETokenizer`, the
    ``call()`` method is not implemented: string operations are not
    backend-agnostic in Keras 3. Use :meth:`tokenize_texts` for batch
    tokenization, or :class:`ASCIICharPreprocessor` when a ``tf.data``
    pipeline needs the BERT-style three-key dictionary.

    **Framing.** :meth:`encode` returns content ids only, matching
    ``tiktoken``. :meth:`encode_with_framing` applies the BERT layout
    ``[CLS] + content + [SEP]`` padded to ``max_length``.

    :param max_length: Maximum sequence length produced by
        :meth:`encode_with_framing` and :meth:`tokenize_texts`. Defaults to
        512.
    :type max_length: int
    :param lowercase: Whether to case-fold input. Defaults to ``True``.
    :type lowercase: bool
    :param normalize_unicode: Whether to NFKD-fold accented characters onto
        their ASCII base before lookup. Defaults to ``True``.
    :type normalize_unicode: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :raises ValueError: If ``max_length`` is not positive, or is smaller
        than the two framing tokens require.
    """

    ENCODING_NAME: str = ENCODING_NAME
    VOCAB_SIZE: int = VOCAB_SIZE
    PAD_ID: int = PAD_ID
    CLS_ID: int = CLS_ID
    SEP_ID: int = SEP_ID
    MASK_ID: int = MASK_ID
    UNK_ID: int = UNK_ID
    NEWLINE_ID: int = NEWLINE_ID

    def __init__(
        self,
        max_length: int = 512,
        lowercase: bool = True,
        normalize_unicode: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise ValueError(
                f"max_length must be an int, got {max_length!r}"
            )
        if max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {max_length}"
            )
        if max_length < 3:
            raise ValueError(
                "max_length must leave room for [CLS], one content "
                f"character and [SEP]; got {max_length}"
            )

        self.max_length = max_length
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode

        logger.info(
            f"Created ASCIICharTokenizer (encoding={self.ENCODING_NAME}, "
            f"vocab_size={self.VOCAB_SIZE}, max_length={max_length}, "
            f"lowercase={lowercase})"
        )

    # -- tiktoken-compatible surface ----------------------------------

    @property
    def n_vocab(self) -> int:
        """Vocabulary size, under ``tiktoken``'s spelling.

        :return: ``VOCAB_SIZE``.
        :rtype: int
        """
        return self.VOCAB_SIZE

    @property
    def vocab_size(self) -> int:
        """Vocabulary size.

        :return: ``VOCAB_SIZE``.
        :rtype: int
        """
        return self.VOCAB_SIZE

    @property
    def encoding_name(self) -> str:
        """Stable encoding tag, safe to use in a cache key.

        A bare ``keras.layers.Layer.name`` is uniquified per process
        (``ascii_char_tokenizer_1`` and so on), so it must never be used to
        key a tokenization cache. This property is constant across
        instances.

        :return: ``"ascii95"``.
        :rtype: str
        """
        return self.ENCODING_NAME

    @property
    def pad_token_id(self) -> int:
        """Padding token id.

        :return: ``PAD_ID``.
        :rtype: int
        """
        return self.PAD_ID

    @property
    def cls_token_id(self) -> int:
        """Sequence-start token id.

        :return: ``CLS_ID``.
        :rtype: int
        """
        return self.CLS_ID

    @property
    def sep_token_id(self) -> int:
        """Separator token id.

        :return: ``SEP_ID``.
        :rtype: int
        """
        return self.SEP_ID

    @property
    def mask_token_id(self) -> int:
        """Masked-language-model sentinel id.

        :return: ``MASK_ID``.
        :rtype: int
        """
        return self.MASK_ID

    @property
    def unk_token_id(self) -> int:
        """Unknown-character token id.

        :return: ``UNK_ID``.
        :rtype: int
        """
        return self.UNK_ID

    # -- tokenization --------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode text to content ids, without framing or padding.

        :param text: Input text.
        :type text: str
        :return: Content ids.
        :rtype: list[int]
        """
        return encode_ascii(
            text,
            lowercase=self.lowercase,
            normalize_unicode=self.normalize_unicode,
        )

    def encode_with_framing(self, text: str) -> List[int]:
        """Encode text to a fixed-length framed, right-padded id sequence.

        Layout is ``[CLS] + content + [SEP]``, truncated so the ``[SEP]``
        is always present, then right-padded with ``[PAD]``.

        :param text: Input text.
        :type text: str
        :return: Exactly ``max_length`` ids.
        :rtype: list[int]
        """
        content = self.encode(text)[: self.max_length - 2]
        ids = [self.CLS_ID] + content + [self.SEP_ID]
        ids.extend([self.PAD_ID] * (self.max_length - len(ids)))
        return ids

    def tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of texts to framed, padded id sequences.

        :param texts: Input texts.
        :type texts: list[str]
        :return: One ``max_length``-long id list per input text.
        :rtype: list[list[int]]
        """
        return [self.encode_with_framing(text) for text in texts]

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode ids back to text.

        :param token_ids: Ids to decode.
        :type token_ids: Sequence[int]
        :param skip_special_tokens: Whether to omit special tokens.
            Defaults to ``True``.
        :type skip_special_tokens: bool
        :return: The decoded string.
        :rtype: str
        """
        return decode_ascii(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode ids back to text, under the house method name.

        :param token_ids: Ids to decode.
        :type token_ids: list[int]
        :return: The decoded string.
        :rtype: str
        """
        return self.decode(token_ids)

    # -- Keras surface -------------------------------------------------

    def call(
        self,
        inputs: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Not implemented; string operations are not backend-agnostic.

        :param inputs: Unused.
        :type inputs: Any
        :param training: Unused.
        :type training: bool | None
        :raises NotImplementedError: Always. Use :meth:`tokenize_texts`, or
            :class:`ASCIICharPreprocessor` inside a ``tf.data`` pipeline.
        """
        raise NotImplementedError(
            "ASCIICharTokenizer does not tokenize in-graph, because Keras 3 "
            "has no backend-agnostic string ops. Use tokenize_texts() for "
            "eager tokenization, or ASCIICharPreprocessor for a tf.data "
            "pipeline."
        )

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the shape produced by tokenizing ``input_shape`` texts.

        :param input_shape: Shape of the input text tensor.
        :type input_shape: Any
        :return: ``(batch, max_length)``.
        :rtype: Any
        """
        return (input_shape[0], self.max_length)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        The vocabulary is a module constant rather than instance state, so
        only the three behavioural knobs are serialized.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "max_length": self.max_length,
                "lowercase": self.lowercase,
                "normalize_unicode": self.normalize_unicode,
            }
        )
        return config


class ASCIICharPreprocessor:
    """
    BERT-style preprocessor over the ASCII character vocabulary.

    A plain Python object -- deliberately not a ``keras.layers.Layer`` --
    implementing the same ``__call__(texts, return_tensors=...)`` contract as
    :class:`~dl_techniques.utils.tokenizer.TiktokenPreprocessor`, so it can be
    passed straight to ``train.common.nlp``'s ``preprocess_mlm_dataset``,
    ``preprocess_classification_dataset`` and ``preprocess_clm_dataset``.

    Those pipelines index the result as ``encoded['input_ids'][0]`` and then
    apply ``tf.ensure_shape(ids, [seq_len])``, so every row this returns is
    **batched** and **exactly** ``max_length`` long. Padding is therefore not
    optional in the pipeline sense; ``padding='max_length'`` is the default
    and the only value that satisfies the contract.

    :param max_length: Fixed output sequence length. Defaults to 512.
    :type max_length: int
    :param lowercase: Whether to case-fold input. Defaults to ``True``.
    :type lowercase: bool
    :param normalize_unicode: Whether to NFKD-fold accented characters.
        Defaults to ``True``.
    :type normalize_unicode: bool
    :param truncation: Whether to truncate over-long text. When ``False``, an
        over-long input raises instead. Defaults to ``True``.
    :type truncation: bool
    :param padding: Padding strategy. Only ``'max_length'`` produces the
        fixed-length rows the ``tf.data`` pipelines require. Defaults to
        ``'max_length'``.
    :type padding: str
    :raises ValueError: If ``max_length`` is too small to hold both framing
        tokens, or ``padding`` is not a recognized strategy.
    """

    ENCODING_NAME: str = ENCODING_NAME
    VOCAB_SIZE: int = VOCAB_SIZE

    def __init__(
        self,
        max_length: int = 512,
        lowercase: bool = True,
        normalize_unicode: bool = True,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> None:
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise ValueError(f"max_length must be an int, got {max_length!r}")
        if max_length < 3:
            raise ValueError(
                "max_length must leave room for [CLS], one content character "
                f"and [SEP]; got {max_length}"
            )
        if padding not in ("max_length", "do_not_pad"):
            raise ValueError(
                "padding must be 'max_length' or 'do_not_pad', got "
                f"{padding!r}"
            )

        self.max_length = max_length
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.truncation = truncation
        self.padding = padding

        self.pad_token_id = PAD_ID
        self.cls_token_id = CLS_ID
        self.sep_token_id = SEP_ID
        self.mask_token_id = MASK_ID
        self.unk_token_id = UNK_ID

        logger.info(
            f"Created ASCIICharPreprocessor (encoding={self.ENCODING_NAME}, "
            f"vocab_size={self.VOCAB_SIZE}, max_length={max_length})"
        )

    @property
    def vocab_size(self) -> int:
        """Vocabulary size.

        :return: ``VOCAB_SIZE``.
        :rtype: int
        """
        return self.VOCAB_SIZE

    @property
    def encoding_name(self) -> str:
        """Stable encoding tag, safe to use in a cache key.

        :return: ``"ascii95"``.
        :rtype: str
        """
        return self.ENCODING_NAME

    def _preprocess_single(self, text: str) -> Dict[str, List[int]]:
        """Preprocess one string into the three BERT-style id lists.

        :param text: Input text.
        :type text: str
        :return: Dictionary with ``input_ids``, ``attention_mask`` and
            ``token_type_ids``, each a list of ints.
        :rtype: dict[str, list[int]]
        :raises ValueError: If the text is too long and ``truncation`` is
            ``False``.
        """
        content = encode_ascii(
            text,
            lowercase=self.lowercase,
            normalize_unicode=self.normalize_unicode,
        )

        # Reserve two slots so [SEP] is always present, matching
        # TiktokenPreprocessor's max_length - 2 budget.
        budget = self.max_length - 2
        if len(content) > budget:
            if not self.truncation:
                raise ValueError(
                    f"Sequence length {len(content)} exceeds max_length-2 "
                    f"({budget}) and truncation=False"
                )
            content = content[:budget]

        input_ids = [self.cls_token_id] + content + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)

        if self.padding == "max_length":
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)

        token_type_ids = [0] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Preprocess text(s) into batched model-ready arrays.

        A single string is treated as a batch of one, so ``input_ids`` is
        always rank 2 and the pipelines' ``[0]`` indexing is correct.

        :param texts: A string, or a list of strings.
        :type texts: str | list[str]
        :param return_tensors: ``'np'`` or ``None`` for NumPy arrays,
            ``'tf'`` for TensorFlow tensors. Defaults to ``None``.
        :type return_tensors: str | None
        :return: Dictionary with ``input_ids``, ``attention_mask`` and
            ``token_type_ids``, each of shape ``(batch, max_length)``.
        :rtype: dict[str, Any]
        :raises TypeError: If ``texts`` is not a string or a list of strings.
        :raises ValueError: If ``return_tensors`` is not ``'np'``, ``'tf'``
            or ``None``.
        """
        if return_tensors not in ("np", "tf", None):
            raise ValueError(
                "return_tensors must be 'np', 'tf' or None, got "
                f"{return_tensors!r}"
            )

        if isinstance(texts, (str, bytes)):
            batch = [texts]
        elif isinstance(texts, (list, tuple)):
            batch = list(texts)
        else:
            raise TypeError(
                "texts must be a string or a list of strings, got "
                f"{type(texts).__name__}"
            )

        encoded = [self._preprocess_single(text) for text in batch]
        result: Dict[str, Any] = {
            key: np.asarray([row[key] for row in encoded], dtype=np.int32)
            for key in ("input_ids", "attention_mask", "token_type_ids")
        }

        if return_tensors == "tf":
            # Imported lazily: this module is reachable from model code that
            # must not pull TensorFlow in just to define a vocabulary.
            import tensorflow as tf

            result = {
                key: tf.convert_to_tensor(value) for key, value in result.items()
            }

        return result

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Preprocess a single string.

        :param text: Input text.
        :type text: str
        :param return_tensors: See :meth:`__call__`.
        :type return_tensors: str | None
        :return: See :meth:`__call__`.
        :rtype: dict[str, Any]
        :raises TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, (str, bytes)):
            raise TypeError(
                f"encode expects a string, got {type(text).__name__}"
            )
        return self(text, return_tensors=return_tensors)

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """Decode ids back to text.

        :param token_ids: Ids to decode; a list, NumPy array or tensor.
        :type token_ids: Any
        :param skip_special_tokens: Whether to omit special tokens. Defaults
            to ``True``.
        :type skip_special_tokens: bool
        :param kwargs: Accepted and ignored, for API compatibility with other
            tokenizers.
        :return: The decoded string.
        :rtype: str
        """
        ids = np.asarray(token_ids).reshape(-1).tolist()
        return decode_ascii(ids, skip_special_tokens=skip_special_tokens)

    def __repr__(self) -> str:
        """Return a concise representation.

        :return: Representation string.
        :rtype: str
        """
        return (
            f"ASCIICharPreprocessor(encoding={self.ENCODING_NAME!r}, "
            f"vocab_size={self.VOCAB_SIZE}, max_length={self.max_length}, "
            f"lowercase={self.lowercase})"
        )
