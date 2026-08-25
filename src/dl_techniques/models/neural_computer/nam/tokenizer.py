"""
Character-level tokenizer for arithmetic expressions, over a fixed 21-token
vocabulary.

The vocabulary is enumerated, not learned. For arithmetic the symbol set is
closed and tiny -- ten digits, four operators, two parentheses, a decimal point,
a space, and three special tokens -- so there is nothing a subword algorithm
could discover that is not already known at authoring time, and fixing the table
buys two things a learned vocabulary cannot. Token ids become stable constants
that downstream code can compare against directly (``DIGIT_IDS``,
``OPERATOR_IDS``), and a digit is guaranteed to be exactly one token, so the
positional structure of a number survives tokenization instead of being carved
into whatever merges a corpus happened to favour.

Characters outside the vocabulary are skipped silently rather than mapped to an
unknown token. There is no ``<UNK>`` id, so an unrepresentable character has
nowhere to go; dropping it keeps the encoded sequence well-formed at the cost of
making ``decode(encode(s))`` lossy for such input.

Tokenizing a character is not the same as supporting it. ``DOT_ID`` round-trips
through ``encode``/``decode`` and ``get_number_mask`` includes it, but
``cell.py``'s number assembly has no fractional branch and ``models/nam`` is
integer-only, so ``"1.5 + 2"`` assembles the operand ``15``. Likewise ``(`` and
``)`` round-trip but do not group anything: they are lexed, never parsed. See the
``NAM`` module docstring.

Every sequence is padded to exactly ``max_len``, so a batch is a dense array
rather than a ragged one. Truncation preserves the terminator: an over-long
expression is cut to ``max_len - 1`` tokens and ``EOS`` is re-appended, so the
end-of-sequence marker is never the thing that gets dropped.

The two mask helpers exist because the downstream model routes on token class
rather than on token identity -- digits feed number assembly, operators feed a
4-way classifier -- and ``OPERATOR_TO_INDEX`` is the map from an operator's token
id to that classifier's output index.
"""

import numpy as np
from typing import List

# ---------------------------------------------------------------------

# Token vocabulary — fixed, not learned
VOCAB = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    " ": 3,
    "0": 4,
    "1": 5,
    "2": 6,
    "3": 7,
    "4": 8,
    "5": 9,
    "6": 10,
    "7": 11,
    "8": 12,
    "9": 13,
    "+": 14,
    "-": 15,
    "*": 16,
    "/": 17,
    "(": 18,
    ")": 19,
    ".": 20,
}

VOCAB_SIZE = len(VOCAB)
INV_VOCAB = {v: k for k, v in VOCAB.items()}

# Semantic groups
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SPACE_ID = 3
DIGIT_IDS = set(range(4, 14))
OPERATOR_IDS = {14, 15, 16, 17}  # +, -, *, /
PAREN_IDS = {18, 19}
DOT_ID = 20

# Operator index mapping (for the 4-way classifier)
OPERATOR_TO_INDEX = {14: 0, 15: 1, 16: 2, 17: 3}  # +, -, *, /

# ---------------------------------------------------------------------

class ArithmeticTokenizer:
    """
    Character-level tokenizer over the fixed 21-token arithmetic vocabulary.

    Converts expression strings to padded token-id sequences and back, and
    provides the token-class masks the downstream model routes on. One character
    is always one token, so a number's digit positions survive tokenization
    intact. Unknown characters are dropped silently -- there is no ``<UNK>`` id --
    which makes the round trip lossy for input outside the vocabulary.

    **Vocabulary:**

    .. code-block:: text

        id  token          group            constant
        ─────────────────────────────────────────────────────────
         0  <PAD>          padding          PAD_ID
         1  <BOS>          special          BOS_ID
         2  <EOS>          special          EOS_ID
         3  ' '            whitespace       SPACE_ID
         4  '0' … 13 '9'   digits           DIGIT_IDS
        14  '+'  15 '-'    operators        OPERATOR_IDS
        16  '*'  17 '/'                     (→ OPERATOR_TO_INDEX 0…3)
        18  '('  19 ')'    parentheses      PAREN_IDS  (lexed, never parsed)
        20  '.'            decimal point    DOT_ID     (lexed, never used)

    **Encoding pipeline:**

    .. code-block:: text

        "1 + 2"
           │
           ▼
        ┌──────────────────────────────────────┐
        │  prepend BOS                         │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  per character: VOCAB lookup         │
        │    hit  → append id                  │
        │    miss → SKIP silently (no <UNK>)   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  append EOS                          │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  fit to max_len                                              │
        │    too long  → truncate to max_len−1, re-append EOS          │
        │                (the terminator is never what gets dropped)   │
        │    too short → right-pad with PAD                            │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        [1, 5, 3, 14, 3, 6, 2, 0, 0, …]   length == max_len, always

    :param max_len: Maximum sequence length, INCLUDING ``BOS`` and ``EOS``. Every
        encoded sequence is exactly this long. Defaults to ``64``.
    :type max_len: int

    Output shape:
        - :meth:`encode`: a list of ``max_len`` ints.
        - :meth:`encode_batch`: an ``int32`` array of shape
          ``(batch_size, max_len)``.
        - :meth:`get_operator_mask` / :meth:`get_number_mask`: a boolean array of
          the same shape as their input.

    Example::

        tok = ArithmeticTokenizer(max_len=32)
        ids = tok.encode("1 + 2 * 3")
        # [1, 5, 3, 14, 3, 6, 3, 16, 3, 7, 2, 0, 0, ...]
        text = tok.decode(ids)
        # "1 + 2 * 3"

    Note:
        Round-tripping a character does not mean the model acts on it. ``.``,
        ``(`` and ``)`` all encode and decode, but number assembly has no
        fractional branch and nothing parses grouping, so ``"1.5 + 2"`` yields the
        operand ``15``.

    Attributes:
        vocab: The forward table, ``VOCAB``.
        inv_vocab: The reverse table, ``INV_VOCAB``.
        vocab_size: ``VOCAB_SIZE`` (21).
    """

    def __init__(self, max_len: int = 64) -> None:
        """Store the sequence length and bind the fixed vocabulary tables.

        Nothing is fitted: the tables are module-level constants, shared by every
        instance.

        :param max_len: Maximum sequence length, including ``BOS``/``EOS``.
        :type max_len: int
        """
        self.max_len = max_len
        self.vocab = VOCAB
        self.inv_vocab = INV_VOCAB
        self.vocab_size = VOCAB_SIZE

    def encode(self, expression: str) -> List[int]:
        """Encode an expression string to a padded token ID list.

        Characters outside the vocabulary are dropped rather than replaced, and
        truncation re-appends ``EOS`` so the terminator survives.

        :param expression: Arithmetic expression string.
        :type expression: str
        :return: List of token IDs, length = max_len.
        :rtype: List[int]
        """
        tokens = [BOS_ID]
        for ch in expression:
            if ch in self.vocab:
                tokens.append(self.vocab[ch])
            # skip unknown characters silently
        tokens.append(EOS_ID)

        # truncate or pad
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len - 1] + [EOS_ID]
        while len(tokens) < self.max_len:
            tokens.append(PAD_ID)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to an expression string.

        ``PAD``, ``BOS`` and ``EOS`` are dropped. An id outside the vocabulary
        decodes to ``"?"`` rather than raising.

        :param token_ids: List of token IDs.
        :type token_ids: List[int]
        :return: Decoded expression string (without BOS/EOS/PAD).
        :rtype: str
        """
        chars = []
        for tid in token_ids:
            if tid in (PAD_ID, BOS_ID, EOS_ID):
                continue
            chars.append(self.inv_vocab.get(tid, "?"))
        return "".join(chars)

    def encode_batch(self, expressions: List[str]) -> np.ndarray:
        """Encode a batch of expressions into one dense array.

        Every row is ``max_len`` long by construction, so the result is
        rectangular regardless of how the expressions differ in length.

        :param expressions: List of expression strings.
        :type expressions: List[str]
        :return: Array of shape (batch_size, max_len).
        :rtype: np.ndarray
        """
        return np.array(
            [self.encode(expr) for expr in expressions],
            dtype=np.int32,
        )

    def get_operator_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """Create a boolean mask indicating operator positions.

        Marks ``+``, ``-``, ``*`` and ``/`` only. Use ``OPERATOR_TO_INDEX`` to map
        a marked id to the 4-way classifier's output index.

        :param token_ids: Token IDs of shape (...).
        :type token_ids: np.ndarray
        :return: Boolean mask, True at operator positions.
        :rtype: np.ndarray
        """
        mask = np.zeros_like(token_ids, dtype=bool)
        for op_id in OPERATOR_IDS:
            mask |= token_ids == op_id
        return mask

    def get_number_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """Create a boolean mask indicating digit/dot positions.

        ``DOT_ID`` IS marked here even though downstream number assembly is
        integer-only; the mask describes the token class, not what the model does
        with it.

        :param token_ids: Token IDs of shape (...).
        :type token_ids: np.ndarray
        :return: Boolean mask, True at digit/dot positions.
        :rtype: np.ndarray
        """
        mask = np.zeros_like(token_ids, dtype=bool)
        for d_id in DIGIT_IDS:
            mask |= token_ids == d_id
        mask |= token_ids == DOT_ID
        return mask

# ---------------------------------------------------------------------
