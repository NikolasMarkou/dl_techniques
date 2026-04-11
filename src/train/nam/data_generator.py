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
    # Level 8: multi-op, 1-2 digits, 2 operators
    DifficultyLevel("1-2d_2op", 1, 2, ["+", "-", "*", "/"], num_ops=2),
    # Level 9: multi-op, 1-3 digits, 2-3 operators
    DifficultyLevel("1-3d_3op", 1, 3, ["+", "-", "*", "/"], num_ops=3),
    # Level 10: multi-op, 1-4 digits, 2-4 operators
    DifficultyLevel("1-4d_4op", 1, 4, ["+", "-", "*", "/"], num_ops=4),
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


def _parse_multi_op(
    expression: str,
) -> List[Dict[str, Any]]:
    """
    Parse a multi-op expression into a sequence of reduction steps.

    Returns one dict per reduction step, in PEMDAS order:
    1. First all ``*`` and ``/`` (left to right)
    2. Then all ``+`` and ``-`` (left to right)

    Each step dict contains:
    - ``expression``: the expression string at this step (after prior reductions)
    - ``left_operand``: float value of the left operand
    - ``right_operand``: float value of the right operand
    - ``operator``: str, one of +, -, *, /
    - ``operator_index``: int, 0-3
    - ``result``: float result of this sub-expression
    - ``new_expression``: the expression after replacing the sub-expression with result

    For single-op expressions, returns a list with one step.
    For expressions with N operators, returns N steps.

    :param expression: Arithmetic expression string (e.g. "3 + 5 * 2 - 1").
    :return: List of per-step label dicts.
    """
    steps = []
    current_expr = expression.strip()

    # Simple tokenizer: split expression into number/operator tokens
    # "3 + 5 * 2 - 1" → ['3', '+', '5', '*', '2', '-', '1']
    def _tokenize_expr(expr_str):
        """Split expression into (value, operator, value, operator, ...) tokens."""
        tokens = []
        i = 0
        s = expr_str.strip()
        while i < len(s):
            if s[i] in ' ':
                i += 1
                continue
            if s[i] in '+-' and (not tokens or isinstance(tokens[-1], str) and tokens[-1] in '+-*/'):
                # Negative number sign
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                tokens.append(float(s[i:j]))
                i = j
            elif s[i].isdigit() or s[i] == '.':
                j = i
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                tokens.append(float(s[i:j]))
                i = j
            elif s[i] in '+-*/':
                tokens.append(s[i])
                i += 1
            else:
                i += 1  # skip unknown
        return tokens

    def _tokens_to_expr(tokens):
        """Convert token list back to expression string (integer-formatted where possible)."""
        parts = []
        for t in tokens:
            if isinstance(t, float):
                if t == int(t) and abs(t) < 1e15:
                    parts.append(str(int(t)))
                else:
                    parts.append(f"{t:.6g}")
            else:
                parts.append(str(t))
        return " ".join(parts)

    while True:
        tokens = _tokenize_expr(current_expr)
        if len(tokens) < 3:
            break  # fully reduced

        # Find the next operator to reduce (PEMDAS: * and / first, then + and -)
        reduce_idx = None

        # Pass 1: find leftmost * or /
        for i in range(1, len(tokens), 2):
            if isinstance(tokens[i], str) and tokens[i] in '*/':
                reduce_idx = i
                break

        # Pass 2: if no * or /, find leftmost + or -
        if reduce_idx is None:
            for i in range(1, len(tokens), 2):
                if isinstance(tokens[i], str) and tokens[i] in '+-':
                    reduce_idx = i
                    break

        if reduce_idx is None:
            break  # no operators left

        left_val = tokens[reduce_idx - 1]
        op_str = tokens[reduce_idx]
        right_val = tokens[reduce_idx + 1]

        # Compute result
        if op_str == '+':
            result = left_val + right_val
        elif op_str == '-':
            result = left_val - right_val
        elif op_str == '*':
            result = left_val * right_val
        elif op_str == '/':
            if abs(right_val) < 1e-10:
                result = 0.0
            else:
                result = left_val / right_val
        else:
            break

        # Record this step
        steps.append({
            "expression": current_expr,
            "left_operand": float(left_val),
            "right_operand": float(right_val),
            "operator": op_str,
            "operator_index": _OP_TO_INDEX[op_str],
            "result": float(result),
        })

        # Replace the sub-expression with the result
        new_tokens = tokens[:reduce_idx - 1] + [result] + tokens[reduce_idx + 2:]
        current_expr = _tokens_to_expr(new_tokens)

    return steps


def _generate_multi_op_expr(
    level: DifficultyLevel,
    num_ops: int,
) -> Tuple[str, float, bool]:
    """
    Generate a multi-op arithmetic expression.

    :param level: Difficulty level for operand sizes.
    :param num_ops: Number of operators (2-4).
    :return: Tuple of (expression_str, result, is_valid).
    """
    operands = [level.sample_operand() for _ in range(num_ops + 1)]
    operators = [random.choice(level.operators) for _ in range(num_ops)]

    # Build expression string
    parts = [str(operands[0])]
    for i, op in enumerate(operators):
        parts.append(f" {op} {operands[i + 1]}")
    expr = "".join(parts)

    result, valid = _safe_eval(expr)
    return expr, result, valid


def prepare_per_step_labels(
    expression: str,
    tokenizer: "ArithmeticTokenizer",
    max_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Prepare per-ACT-step labels for a single expression.

    For each reduction step, produces the updated token_ids (after prior
    reductions), the operator position in those tokens, and the operator type.

    For single-op expressions this produces 1 meaningful step. For multi-op
    with N operators, produces N steps. Remaining steps (up to max_steps) are
    padded with the final reduced expression and zeroed labels.

    :param expression: Expression string (e.g. "3 + 5 * 2").
    :param tokenizer: ArithmeticTokenizer instance.
    :param max_steps: Maximum ACT steps (padding target).
    :return: Dict with:
        - per_step_token_ids: (max_steps, L) int32
        - per_step_op_position: (max_steps,) int32
        - per_step_op_index: (max_steps,) int32
        - num_reduction_steps: int — how many real steps
    """
    steps = _parse_multi_op(expression)
    L = tokenizer.max_len

    step_token_ids = np.zeros((max_steps, L), dtype=np.int32)
    step_op_positions = np.zeros((max_steps,), dtype=np.int32)
    step_op_indices = np.zeros((max_steps,), dtype=np.int32)

    # For each reduction step, tokenize the current expression and find
    # the operator position in the token stream
    current_expr = expression.strip()
    op_tid_map = {"+": 14, "-": 15, "*": 16, "/": 17}

    for i, step in enumerate(steps):
        if i >= max_steps:
            break

        ids = tokenizer.encode(current_expr)
        step_token_ids[i] = ids

        # Find the operator position for THIS step's operator in the tokens
        target_op = step["operator"]
        target_left = step["left_operand"]
        target_op_tid = op_tid_map[target_op]

        # Scan for the correct operator (match by type and left-context value)
        found_pos = 0
        for pos, tid in enumerate(ids):
            if tid == target_op_tid:
                # Verify left operand by assembling digits to the left
                left_digits = []
                for j in range(pos - 1, -1, -1):
                    if 4 <= ids[j] <= 13:
                        left_digits.insert(0, ids[j] - 4)
                    elif ids[j] == 3:  # space
                        continue
                    else:
                        break
                if left_digits:
                    assembled = sum(
                        int(d) * 10 ** p for p, d in enumerate(reversed(left_digits))
                    )
                    if abs(assembled - abs(target_left)) < 0.5:
                        found_pos = pos
                        break

        step_op_positions[i] = found_pos
        step_op_indices[i] = _OP_TO_INDEX[target_op]

        # Simplify expression for the next step: replace the sub-expression
        # with its integer result (re-parse from the step's result)
        tokens = _parse_multi_op.__code__.co_consts  # just use the helper
        # Rebuild: we already have the new expression from _parse_multi_op
        # but need to reconstruct it. Simplest: re-run the tokenizer math.
        left = step["left_operand"]
        right = step["right_operand"]
        result = step["result"]

        # Build the new expression by replacing "left op right" with result
        # in the current expression string
        result_str = str(int(result)) if result == int(result) and abs(result) < 1e15 else f"{result:.6g}"
        left_str = str(int(left)) if left == int(left) and abs(left) < 1e15 else f"{left:.6g}"
        right_str = str(int(right)) if right == int(right) and abs(right) < 1e15 else f"{right:.6g}"
        sub_expr = f"{left_str} {target_op} {right_str}"

        # Replace first occurrence of the sub-expression
        new_expr = current_expr.replace(sub_expr, result_str, 1)
        if new_expr == current_expr:
            # Fallback: try with the original expression tokens
            new_expr = result_str
        current_expr = new_expr.strip()

    # Pad remaining steps with the final expression
    final_ids = tokenizer.encode(current_expr)
    for i in range(len(steps), max_steps):
        step_token_ids[i] = final_ids

    # Step validity mask: 1.0 for real reduction steps, 0.0 for padding
    step_valid_mask = np.zeros((max_steps,), dtype=np.float32)
    num_real_steps = min(len(steps), max_steps)
    step_valid_mask[:num_real_steps] = 1.0

    # For padding steps, copy the LAST valid step's targets so the model
    # gets consistent (not garbage) targets even on masked-out steps.
    if num_real_steps > 0 and num_real_steps < max_steps:
        for i in range(num_real_steps, max_steps):
            step_op_positions[i] = step_op_positions[num_real_steps - 1]
            step_op_indices[i] = step_op_indices[num_real_steps - 1]

    return {
        "per_step_token_ids": step_token_ids,
        "per_step_op_position": step_op_positions,
        "per_step_op_index": step_op_indices,
        "per_step_valid_mask": step_valid_mask,
        "num_reduction_steps": num_real_steps,
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
        if level.num_ops == 1:
            expr, result, valid = _generate_single_op_expr(level)
        else:
            expr, result, valid = _generate_multi_op_expr(level, level.num_ops)
        expressions.append(expr)
        targets.append(result)
        validity.append(float(valid))

    input_ids = tokenizer.encode_batch(expressions)
    targets_arr = np.array(targets, dtype=np.float32).reshape(-1, 1)
    validity_arr = np.array(validity, dtype=np.float32).reshape(-1, 1)

    # Parse structured labels — for multi-op, extract FIRST step's labels
    # (the reduction step with highest precedence). Per-step labels for
    # subsequent ACT steps are computed from the expression's _parse_multi_op
    # reduction sequence; the training loop can call _parse_multi_op directly
    # for multi-step supervision.
    left_operands = []
    right_operands = []
    operator_indices = []
    operator_positions = []
    all_reduction_steps = []  # per-sample list of multi-op step dicts

    for i, expr in enumerate(expressions):
        steps = _parse_multi_op(expr)
        all_reduction_steps.append(steps)

        if steps:
            first_step = steps[0]
            # Find operator position in the token_ids for the first step
            # Re-use _parse_single_op's token scan for finding position in tokens
            parsed_pos = _parse_single_op(expr, input_ids[i].tolist())

            # For multi-op, find the CORRECT operator position (the one
            # matching the first step's operator in PEMDAS order)
            if len(steps) > 1:
                target_op_str = first_step["operator"]
                target_left = first_step["left_operand"]
                # Scan token_ids for all operators of this type and find
                # the one whose left context matches
                op_tid = {"+": 14, "-": 15, "*": 16, "/": 17}[target_op_str]
                found_pos = -1
                for pos, tid in enumerate(input_ids[i].tolist()):
                    if tid == op_tid:
                        # Check if left operand matches by assembling tokens
                        # to the left of this position
                        left_digits = []
                        for j in range(pos - 1, -1, -1):
                            t = input_ids[i, j]
                            if 4 <= t <= 13:
                                left_digits.insert(0, t - 4)
                            elif t == 3:  # space
                                continue
                            else:
                                break
                        if left_digits:
                            assembled = sum(
                                int(d) * 10 ** p
                                for p, d in enumerate(reversed(left_digits))
                            )
                            if abs(assembled - target_left) < 0.5:
                                found_pos = pos
                                break
                if found_pos >= 0:
                    op_position = found_pos
                else:
                    op_position = parsed_pos["operator_position"]
            else:
                op_position = parsed_pos["operator_position"]

            left_operands.append(first_step["left_operand"])
            right_operands.append(first_step["right_operand"])
            operator_indices.append(first_step["operator_index"])
            operator_positions.append(op_position)
        else:
            # Fallback for unparseable expressions
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
        "reduction_steps": all_reduction_steps,  # per-sample list for multi-step training
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
