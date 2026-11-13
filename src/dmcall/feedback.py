"""Feedback representations: how the decision model's scores reach the generator (Φ in the paper).

Only the generator's copy of earlier feedback changes. Acceptance always uses the numeric
probabilities, and :attr:`dmcall.Result.trace` keeps them unchanged.

``"hybrid"`` (the default)
    Each probability keeps its value and threshold and gains a verbal label:
    Φ(p) = (p, ℓ(p)).
``"numeric"``
    Probabilities and thresholds only: Φ(p) = p.
``"text"``
    Labels only: Φ(p) = ℓ(p). Probabilities and thresholds are removed.

ℓ uses five bands of equal width, lower bound inclusive; the last band includes 1. A question
that was not asked (the argument question on a no call) keeps a null score and a null label.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from copy import deepcopy
from typing import Any

__all__ = ["CUT_POINTS", "FEEDBACK", "LABELS", "label", "represent", "represent_history", "single_check"]

FEEDBACK = ("hybrid", "numeric", "text")
CUT_POINTS = (0.2, 0.4, 0.6, 0.8)
LABELS = ("very likely incorrect", "likely incorrect", "uncertain", "likely correct", "very likely correct")

# Where a probability's label goes. Decision-model records use ``probability_yes``.
_LABEL_FIELDS = {
    "probability_yes": "feedback_label",
    "function_probability": "function_label",
    "arguments_probability": "arguments_label",
    "p_f": "function_label",
    "p_a": "arguments_label",
}
_THRESHOLDS = {"threshold", "function_threshold", "arguments_threshold"}
# Names of the two decision-model checks in a feedback record.
_CHECK_RECORDS = {"function": "function_selection", "arguments": "arguments"}


def label(probability: float | int | None) -> str | None:
    """ℓ(p): the verbal band of a probability; ``None`` for a question that was not asked."""
    if probability is None:
        return None
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("Feedback probabilities must be finite numbers between 0 and 1, or null.")
    return LABELS[bisect_right(CUT_POINTS, probability)]


def _check(representation: str) -> str:
    if representation not in FEEDBACK:
        raise ValueError(f"feedback must be one of {', '.join(map(repr, FEEDBACK))}.")
    return representation


def _hybrid(value: Any) -> Any:
    if isinstance(value, list):
        return [_hybrid(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        # Schema errors are diagnostics about the task input, not scores.
        result[key] = deepcopy(item) if key == "schema_errors" else _hybrid(item)
        if key in _LABEL_FIELDS:
            field = _LABEL_FIELDS[key]
            if field in value or field in result:
                raise ValueError("Repair feedback has conflicting added judgment fields.")
            result[field] = label(item)
    return result


def _text(value: Any) -> Any:
    if isinstance(value, list):
        return [_text(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in _THRESHOLDS:
            continue
        if key in _LABEL_FIELDS:
            field = _LABEL_FIELDS[key]
            if field in value or field in result:
                raise ValueError("Repair feedback has conflicting judgment fields.")
            result[field] = label(item)
        elif key == "instruction" and isinstance(item, str):
            result[key] = item.replace("The probabilities", "The feedback labels")
        elif key == "schema_errors":
            result[key] = deepcopy(item)
        else:
            result[key] = _text(item)
    return result


def represent(feedback: dict, representation: str = "hybrid") -> dict:
    """The generator's copy of one feedback record under a representation. The input is not changed."""
    _check(representation)
    if not isinstance(feedback, dict):
        raise ValueError("Repair feedback must be an object.")
    if representation == "numeric":
        return deepcopy(feedback)
    if representation == "text":
        return _text(feedback)
    return _hybrid(feedback)


def represent_history(history: list[dict], representation: str = "hybrid") -> list[dict]:
    """Every record of a repair history under a representation, in the same order."""
    _check(representation)
    result = deepcopy(history)
    for item in result:
        if not isinstance(item, dict) or not isinstance(item.get("feedback"), dict):
            raise ValueError("Each repair history record needs a feedback object.")
        item["feedback"] = represent(item["feedback"], representation)
    return result


def single_check(feedback: dict, keep: str) -> dict:
    """A feedback record without the other question, for DM-Call-Function and DM-Call-Argument.

    The source, schema errors, thresholds and instruction stay; only the hidden question's
    verdict and score are removed.
    """
    if keep not in _CHECK_RECORDS:
        raise ValueError("keep must be 'function' or 'arguments'.")
    hidden = "arguments" if keep == "function" else "function"
    result = deepcopy(feedback)
    if isinstance(result.get("checks"), dict):
        result["checks"] = {name: value for name, value in result["checks"].items()
                            if name != _CHECK_RECORDS[hidden]}
    if isinstance(result.get("failed_checks"), list):
        result["failed_checks"] = [name for name in result["failed_checks"] if name != _CHECK_RECORDS[hidden]]
    for key in [key for key in result if key.startswith(hidden + "_")]:
        del result[key]
    return result
