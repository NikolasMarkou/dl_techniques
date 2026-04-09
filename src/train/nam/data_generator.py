"""
Random arithmetic expression generator for NAM training.

Generates expressions of configurable complexity with ground-truth results
and validity flags. Supports smooth curriculum learning where difficulty
increases gradually and easier examples are always mixed in to prevent
catastrophic forgetting.

Difficulty dimensions (sampled per-example):
- Number of digits per operand (1 → 10)
- Operator set (+/- only → all four)
- Number of operators (1 → N)

The curriculum is controlled by a single ``progress`` parameter in [0, 1]
that shifts a probability distribution over difficulty levels.
"""

import ast
import math
import re
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Difficulty levels ──────────────────────────────────────────────────
# Each level defines a band of complexity. The curriculum samples from
# these with probabilities that shift toward harder levels over training.

@dataclass
class DifficultyLevel:
    """One band of expression complexity."""
    name: str
    num_digits_min: int      # min digits per operand
    num_digits_max: int      # max digits per operand
    operators: List[str]     # allowed operators
    num_ops: int = 1         # number of operators in expression

    def sample_operand(self) -> int:
        """Sample a random integer with the configured digit count."""
        n_digits = random.randint(self.num_digits_min, self.num_digits_max)
        if n_digits == 1:
            return random.randint(1, 9)
        lo = 10 ** (n_digits - 1)
        hi = 10 ** n_digits - 1
        return random.randint(lo, hi)


# 8 difficulty levels spanning 1-digit to 10-digit operands
DIFFICULTY_LEVELS: List[DifficultyLevel] = [
    # Level 0: 1-digit, add/sub only
    DifficultyLevel("1d_addsub", 1, 1, ["+", "-"]),
    # Level 1: 1-digit, all ops
    DifficultyLevel("1d_all", 1, 1, ["+", "-", "*", "/"]),
    # Level 2: 1-2 digits, all ops
    DifficultyLevel("1-2d", 1, 2, ["+", "-", "*", "/"]),
    # Level 3: 2-3 digits
    DifficultyLevel("2-3d", 2, 3, ["+", "-", "*", "/"]),
    # Level 4: 3-4 digits
    DifficultyLevel("3-4d", 3, 4, ["+", "-", "*", "/"]),
    # Level 5: 4-6 digits
    DifficultyLevel("4-6d", 4, 6, ["+", "-", "*", "/"]),
    # Level 6: 6-8 digits
    DifficultyLevel("6-8d", 6, 8, ["+", "-", "*", "/"]),
    # Level 7: 8-10 digits
    DifficultyLevel("8-10d", 8, 10, ["+", "-", "*", "/"]),
]

NUM_LEVELS = len(DIFFICULTY_LEVELS)


def _curriculum_probs(progress: float) -> np.ndarray:
    """
    Compute sampling probabilities over difficulty levels.

    Uses a shifted softmax: at progress=0, mass concentrates on easy levels.
    At progress=1, mass spreads across all levels but easy levels still
    retain ~15-20% probability to prevent forgetting.

    :param progress: Training progress in [0, 1].
    :return: Array of shape (NUM_LEVELS,) summing to 1.
    """
    # Center of the distribution shifts from level 0 to level NUM_LEVELS-1
    center = progress * (NUM_LEVELS - 1)
    # Temperature: starts tight (1.0), widens to 2.5
    temperature = 1.0 + 1.5 * progress

    indices = np.arange(NUM_LEVELS, dtype=np.float64)
    logits = -(indices - center) ** 2 / (2.0 * temperature ** 2)

    # Floor: every level always has at least 2% probability
    probs = np.exp(logits)
    probs = probs / probs.sum()
    floor = 0.02
    probs = probs * (1.0 - floor * NUM_LEVELS) + floor
    probs = probs / probs.sum()
    return probs


def _generate_single_op_expr(level: DifficultyLevel) -> Tuple[str, float, bool]:
    """
    Generate a single-operator expression at the given difficulty level.

    :return: (expression_string, result, is_valid)
    """
    left = level.sample_operand()
    right = level.sample_operand()
    op = random.choice(level.operators)

    # Avoid division by zero at generation time
    if op == "/" and right == 0:
        right = random.randint(1, 9)

    expr = f"{left} {op} {right}"
    result, valid = _safe_eval(expr)
    return expr, result, valid


# ── Legacy ExpressionConfig (kept for backward compatibility / tests) ──


@dataclass
class ExpressionConfig:
    """
    Configuration for expression generation.

    :param min_ops: Minimum number of operators.
    :param max_ops: Maximum number of operators.
    :param min_val: Minimum integer operand value.
    :param max_val: Maximum integer operand value.
    :param operators: List of operator strings to use.
    :param allow_parentheses: Whether to generate parenthesized sub-expressions.
    :param max_paren_depth: Maximum nesting depth of parentheses.
    :param allow_floats: Whether to use float operands.
    :param float_decimals: Number of decimal places for floats.
    """

    min_ops: int = 1
    max_ops: int = 4
    min_val: int = 0
    max_val: int = 100
    operators: List[str] = None
    allow_parentheses: bool = False
    max_paren_depth: int = 2
    allow_floats: bool = False
    float_decimals: int = 2

    def __post_init__(self):
        if self.operators is None:
            self.operators = ["+", "-", "*", "/"]


# Curriculum phases (legacy — kept for backward compat)
CURRICULUM = {
    "phase_1": ExpressionConfig(
        min_ops=1, max_ops=1,
        min_val=1, max_val=20,
        operators=["+", "-", "*", "/"],
        allow_parentheses=False,
    ),
    "phase_2": ExpressionConfig(
        min_ops=1, max_ops=2,
        min_val=1, max_val=50,
        operators=["+", "-", "*", "/"],
        allow_parentheses=False,
    ),
    "phase_3": ExpressionConfig(
        min_ops=2, max_ops=4,
        min_val=0, max_val=100,
        operators=["+", "-", "*", "/"],
        allow_parentheses=False,
    ),
    "phase_4": ExpressionConfig(
        min_ops=2, max_ops=4,
        min_val=0, max_val=100,
        operators=["+", "-", "*", "/"],
        allow_parentheses=True,
        max_paren_depth=1,
    ),
    "phase_5": ExpressionConfig(
        min_ops=2, max_ops=8,
        min_val=0, max_val=1000,
        operators=["+", "-", "*", "/"],
        allow_parentheses=True,
        max_paren_depth=2,
        allow_floats=True,
    ),
}


def _random_number(config: ExpressionConfig) -> str:
    """Generate a random number string."""
    if config.allow_floats and random.random() < 0.3:
        val = random.uniform(config.min_val, config.max_val)
        return f"{val:.{config.float_decimals}f}"
    return str(random.randint(config.min_val, config.max_val))


def _generate_expr(config: ExpressionConfig, depth: int = 0) -> str:
    """
    Recursively generate a random arithmetic expression.

    :param config: Expression configuration.
    :param depth: Current nesting depth.
    :return: Expression string.
    """
    num_ops = random.randint(config.min_ops, config.max_ops)

    parts = [_random_number(config)]
    for _ in range(num_ops):
        op = random.choice(config.operators)

        # Possibly generate a parenthesized sub-expression
        if (
            config.allow_parentheses
            and depth < config.max_paren_depth
            and random.random() < 0.3
        ):
            sub_config = ExpressionConfig(
                min_ops=1,
                max_ops=max(1, num_ops - 1),
                min_val=config.min_val,
                max_val=config.max_val,
                operators=config.operators,
                allow_parentheses=True,
                max_paren_depth=config.max_paren_depth,
                allow_floats=config.allow_floats,
                float_decimals=config.float_decimals,
            )
            operand = f"({_generate_expr(sub_config, depth + 1)})"
        else:
            operand = _random_number(config)

        parts.append(f" {op} {operand}")

    return "".join(parts)


def _safe_eval(expression: str) -> Tuple[float, bool]:
    """
    Safely evaluate an arithmetic expression.

    Returns (result, is_valid). Division by zero yields (0.0, False).

    :param expression: Arithmetic expression string.
    :return: Tuple of (result, validity).
    """
    try:
        # Use ast.literal_eval won't work for expressions with operators,
        # so we compile and eval with restricted builtins
        result = eval(  # noqa: S307 — safe: input is self-generated
            compile(expression, "<expr>", "eval"),
            {"__builtins__": {}},
        )
        if not isinstance(result, (int, float)):
            return 0.0, False
        if not np.isfinite(result):
            return 0.0, False
        return float(result), True
    except ZeroDivisionError:
        return 0.0, False
    except Exception:
        return 0.0, False


# Operator symbol → classifier index (matches tokenizer OPERATOR_TO_INDEX)
_OP_TO_INDEX = {"+": 0, "-": 1, "*": 2, "/": 3}


def _parse_single_op(
    expression: str,
    token_ids: List[int],
) -> Dict[str, Any]:
    """
    Parse a single-operator expression into structured labels.

    For expressions like ``"6 + 8"``, extracts the left operand value,
    right operand value, operator index, and the operator's token position.

    :param expression: Single-operator expression string (e.g., ``"6 + 8"``).
    :param token_ids: Token IDs from the tokenizer for this expression.
    :return: Dict with keys: left_operand, right_operand, operator_index,
        operator_position. Returns None values if parsing fails.
    """
    # Token IDs for operators: + = 14, - = 15, * = 16, / = 17
    operator_token_ids = {14, 15, 16, 17}

    # Find operator position in token_ids
    op_position = -1
    op_token_id = -1
    for i, tid in enumerate(token_ids):
        if tid in operator_token_ids:
            op_position = i
            op_token_id = tid
            break

    if op_position == -1:
        # No operator found — return null labels
        return {
            "left_operand": 0.0,
            "right_operand": 0.0,
            "operator_index": 0,
            "operator_position": 0,
        }

    # Map token ID to operator index: 14→0, 15→1, 16→2, 17→3
    operator_index = op_token_id - 14

    # Parse operand values from the expression string
    # Single-op expressions have the form: "left op right"
    # Split on the operator (handle negative numbers by splitting carefully)
    match = re.match(
        r"^\s*(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*$",
        expression,
    )
    if match:
        left_operand = float(match.group(1))
        right_operand = float(match.group(3))
    else:
        # Fallback: try ast-based parsing for edge cases
        try:
            tree = ast.parse(expression, mode="eval")
            if isinstance(tree.body, ast.BinOp):
                left_operand = float(ast.literal_eval(
                    ast.Expression(body=tree.body.left)
                ))
                right_operand = float(ast.literal_eval(
                    ast.Expression(body=tree.body.right)
                ))
            else:
                left_operand = 0.0
                right_operand = 0.0
        except Exception:
            left_operand = 0.0
            right_operand = 0.0

    return {
        "left_operand": left_operand,
        "right_operand": right_operand,
        "operator_index": operator_index,
        "operator_position": op_position,
    }


def generate_curriculum_batch(
    batch_size: int,
    progress: float,
    tokenizer: "ArithmeticTokenizer",  # noqa: F821
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Generate a batch using smooth curriculum sampling.

    Each example independently samples a difficulty level from a distribution
    that shifts from easy→hard as ``progress`` goes from 0→1. Earlier
    levels always retain probability mass to prevent forgetting.

    :param batch_size: Number of expressions.
    :param progress: Training progress in [0, 1].
    :param tokenizer: ArithmeticTokenizer instance.
    :return: Same 5-tuple as ``generate_batch``.
    """
    probs = _curriculum_probs(progress)
    levels = np.random.choice(NUM_LEVELS, size=batch_size, p=probs)

    expressions = []
    targets = []
    validity = []

    for lvl_idx in levels:
        level = DIFFICULTY_LEVELS[lvl_idx]
        expr, result, valid = _generate_single_op_expr(level)
        expressions.append(expr)
        targets.append(result)
        validity.append(float(valid))

    input_ids = tokenizer.encode_batch(expressions)
    targets_arr = np.array(targets, dtype=np.float32).reshape(-1, 1)
    validity_arr = np.array(validity, dtype=np.float32).reshape(-1, 1)

    # Parse structured labels
    left_operands = []
    right_operands = []
    operator_indices = []
    operator_positions = []

    for i, expr in enumerate(expressions):
        parsed = _parse_single_op(expr, input_ids[i].tolist())
        left_operands.append(parsed["left_operand"])
        right_operands.append(parsed["right_operand"])
        operator_indices.append(parsed["operator_index"])
        operator_positions.append(parsed["operator_position"])

    labels = {
        "left_operand": np.array(left_operands, dtype=np.float32).reshape(-1, 1),
        "right_operand": np.array(right_operands, dtype=np.float32).reshape(-1, 1),
        "operator_index": np.array(operator_indices, dtype=np.int32),
        "operator_position": np.array(operator_positions, dtype=np.int32),
    }

    return input_ids, targets_arr, validity_arr, expressions, labels


def generate_batch(
    batch_size: int,
    config: ExpressionConfig,
    tokenizer: "ArithmeticTokenizer",  # noqa: F821
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Generate a batch of arithmetic expressions with ground truth.

    :param batch_size: Number of expressions.
    :param config: Expression generation config.
    :param tokenizer: ArithmeticTokenizer instance.
    :return: Tuple of (input_ids, targets, validity, expressions, labels).
        - input_ids: (B, max_len) token IDs
        - targets: (B, 1) ground truth results
        - validity: (B, 1) whether expression evaluates successfully
        - expressions: list of expression strings
        - labels: dict with keys:
            - left_operand: (B, 1) left operand values
            - right_operand: (B, 1) right operand values
            - operator_index: (B,) operator class (0=+, 1=-, 2=*, 3=/)
            - operator_position: (B,) token position of the operator
    """
    expressions = []
    targets = []
    validity = []

    for _ in range(batch_size):
        # Keep trying until we get a valid expression
        for _attempt in range(10):
            expr = _generate_expr(config)
            result, valid = _safe_eval(expr)
            expressions.append(expr)
            targets.append(result)
            validity.append(float(valid))
            break

    input_ids = tokenizer.encode_batch(expressions)
    targets_arr = np.array(targets, dtype=np.float32).reshape(-1, 1)
    validity_arr = np.array(validity, dtype=np.float32).reshape(-1, 1)

    # Parse structured labels for each expression
    left_operands = []
    right_operands = []
    operator_indices = []
    operator_positions = []

    for i, expr in enumerate(expressions):
        parsed = _parse_single_op(expr, input_ids[i].tolist())
        left_operands.append(parsed["left_operand"])
        right_operands.append(parsed["right_operand"])
        operator_indices.append(parsed["operator_index"])
        operator_positions.append(parsed["operator_position"])

    labels = {
        "left_operand": np.array(left_operands, dtype=np.float32).reshape(-1, 1),
        "right_operand": np.array(right_operands, dtype=np.float32).reshape(-1, 1),
        "operator_index": np.array(operator_indices, dtype=np.int32),
        "operator_position": np.array(operator_positions, dtype=np.int32),
    }

    return input_ids, targets_arr, validity_arr, expressions, labels
