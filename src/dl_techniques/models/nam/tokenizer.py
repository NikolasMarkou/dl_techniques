"""
Character-level tokenizer for arithmetic expressions.

Fixed vocabulary of 21 tokens covering digits, operators, parentheses,
decimal point, and special tokens.
"""

from typing import List, Optional

import numpy as np


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


class ArithmeticTokenizer:
    """
    Tokenizer for arithmetic expressions.

    Converts expression strings to token ID sequences and back.
    Uses a fixed 21-token vocabulary.

    :param max_len: Maximum sequence length (including BOS/EOS).
    :type max_len: int

    Example::

        tok = ArithmeticTokenizer(max_len=32)
        ids = tok.encode("1 + 2 * 3")
        # [1, 5, 3, 14, 3, 6, 3, 16, 3, 7, 2, 0, 0, ...]
        text = tok.decode(ids)
        # "1 + 2 * 3"
    """

    def __init__(self, max_len: int = 64) -> None:
        self.max_len = max_len
        self.vocab = VOCAB
        self.inv_vocab = INV_VOCAB
        self.vocab_size = VOCAB_SIZE

    def encode(self, expression: str) -> List[int]:
        """
        Encode an expression string to a padded token ID list.

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
        """
        Decode token IDs back to an expression string.

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
        """
        Encode a batch of expressions.

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
        """
        Create a boolean mask indicating operator positions.

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
        """
        Create a boolean mask indicating digit/dot positions.

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
