"""Opt-in probability-to-text repair feedback for separately recorded experiments.

The verifier and acceptance gate retain their original numerical probabilities.
Only the copy of earlier feedback sent to the generator changes. The default
``numeric`` scheme sends the same history and provider prompt as the baseline.
No labels imply benchmark correctness: they describe a verifier's judgment.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["FeedbackGenerator", "FeedbackScheme", "SCHEME_IDS"]

# Three bands follow the user's explicit decimal example rather than thirds.
_CUTS = {3: (0.33, 0.66), 4: (0.25, 0.5, 0.75), 5: (0.2, 0.4, 0.6, 0.8)}
_LABELS = {
    "solved": {
        3: ("unsolved", "partially solved", "solved"),
        4: ("unsolved", "weakly solved", "mostly solved", "solved"),
        5: ("unsolved", "slightly solved", "partially solved", "mostly solved", "solved"),
    },
    "correctness": {
        3: ("likely incorrect", "uncertain", "likely correct"),
        4: ("very likely incorrect", "likely incorrect", "likely correct", "very likely correct"),
        5: ("very likely incorrect", "likely incorrect", "uncertain", "likely correct", "very likely correct"),
    },
    "action": {
        3: ("replace", "reconsider", "keep"),
        4: ("replace", "revise", "recheck", "keep"),
        5: ("replace", "substantially revise", "revise", "recheck", "keep"),
    },
}
SCHEME_IDS = ("numeric",) + tuple(f"{family}_{bins}" for family in _LABELS for bins in _CUTS)
_PROBABILITY_LABELS = {
    "probability_yes": "feedback_label",
    "function_probability": "function_label",
    "arguments_probability": "arguments_label",
    "p_f": "function_label",
    "p_a": "arguments_label",
}
_THRESHOLDS = {"threshold", "function_threshold", "arguments_threshold"}


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeedbackScheme:
    """One immutable, prespecified feedback representation.

    ``FeedbackScheme()`` is the unchanged numerical baseline. Text variants use
    ``FeedbackScheme.from_id("solved_3")`` (or another member of ``SCHEME_IDS``).
    Labels/cut points are recorded by :meth:`config`, never added to the prompt.
    At a cut point the next band starts; the final band includes probability 1.
    """

    family: str = "numeric"
    bins: int | None = None

    def __post_init__(self) -> None:
        if self.family == "numeric":
            if self.bins is not None:
                raise ValueError("The numeric feedback scheme does not have bins.")
        elif (not isinstance(self.family, str) or self.family not in _LABELS
              or isinstance(self.bins, bool) or not isinstance(self.bins, int) or self.bins not in _CUTS):
            raise ValueError(f"Choose a feedback scheme from {', '.join(SCHEME_IDS)}.")

    @classmethod
    def from_id(cls, scheme_id: str) -> FeedbackScheme:
        if not isinstance(scheme_id, str) or scheme_id not in SCHEME_IDS:
            raise ValueError(f"Choose a feedback scheme from {', '.join(SCHEME_IDS)}.")
        if scheme_id == "numeric":
            return cls()
        family, bins = scheme_id.rsplit("_", 1)
        return cls(family, int(bins))

    @property
    def scheme_id(self) -> str:
        return "numeric" if self.family == "numeric" else f"{self.family}_{self.bins}"

    @property
    def labels(self) -> tuple[str, ...]:
        return () if self.bins is None else _LABELS[self.family][self.bins]

    @property
    def cut_points(self) -> tuple[float, ...]:
        return () if self.bins is None else _CUTS[self.bins]

    def config(self) -> dict:
        """Manifest-only definition of this arm, with no model or oracle fields."""
        return {
            "version": 1,
            "scheme_id": self.scheme_id,
            "family": self.family,
            "bins": self.bins,
            "cut_points": list(self.cut_points),
            "labels": list(self.labels),
            "interval_rule": "lower-inclusive, upper-exclusive; final interval includes 1",
            "null_probability_label": None,
            "generator_only": True,
            "gate_changed": False,
            "probability_field_replacement": {} if self.family == "numeric" else dict(_PROBABILITY_LABELS),
            "threshold_fields_removed": [] if self.family == "numeric" else sorted(_THRESHOLDS),
        }

    @property
    def config_sha256(self) -> str:
        return _sha256(self.config())

    def label(self, probability: float | int | None) -> str | float | int | None:
        """Map a finite probability to its band; preserve null for unrun checks."""
        if probability is None:
            return None
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not 0 <= probability <= 1 or not math.isfinite(probability)):
            raise ValueError("Feedback probabilities must be finite numbers between 0 and 1, or null.")
        if self.family == "numeric":
            return probability
        return self.labels[bisect_right(self.cut_points, probability)]

    def transform_feedback(self, feedback: dict) -> dict:
        """Copy feedback, replacing only judgment fields and their explanation.

        Schema errors and proposed arguments may contain task numbers and are
        preserved. Threshold/probability fields are removed recursively from
        verifier feedback, never from a proposed call or original conversation.
        A skipped argument check remains skipped with a null label.
        """
        if not isinstance(feedback, dict):
            raise ValueError("Repair feedback must be an object.")
        if self.family == "numeric":
            return deepcopy(feedback)

        def transform(value: Any) -> Any:
            if isinstance(value, list):
                return [transform(item) for item in value]
            if not isinstance(value, dict):
                return deepcopy(value)
            result = {}
            for key, item in value.items():
                if key in _THRESHOLDS:
                    continue
                if key in _PROBABILITY_LABELS:
                    replacement = _PROBABILITY_LABELS[key]
                    if replacement in value or replacement in result:
                        raise ValueError("Repair feedback has conflicting judgment fields.")
                    result[replacement] = self.label(item)
                elif key == "instruction" and isinstance(item, str):
                    # Keep the existing repair instruction, changing only the
                    # sentence that identifies the feedback representation.
                    result[key] = item.replace("The probabilities", "The feedback labels")
                elif key == "schema_errors":
                    result[key] = deepcopy(item)
                else:
                    result[key] = transform(item)
            return result

        return transform(feedback)

    def transform_history(self, history: list[dict] | None) -> list[dict] | None:
        """Preserve chronological rounds/calls and transform each feedback copy."""
        if history is None:
            return None
        if not isinstance(history, list):
            raise ValueError("Repair history must be a list or null.")
        if self.family == "numeric":
            return deepcopy(history)
        result = deepcopy(history)
        for item in result:
            if not isinstance(item, dict) or not isinstance(item.get("feedback"), dict):
                raise ValueError("Each repair history record needs a feedback object.")
            item["feedback"] = self.transform_feedback(item["feedback"])
        return result


class FeedbackGenerator:
    """Generator decorator that changes repair context without changing the gate.

    ``before_generate`` receives a detached request audit immediately before the
    underlying generator runs. ``last_request`` retains the same audit. Its
    ``payload`` is the existing provider's request body, when a payload builder
    is available; credentials, headers and oracle IDs are never collected.
    The stock frozen ``FunctionCallingLLM`` and canonical ``dmgfc`` providers
    build deterministic bodies, so this audit matches their actual request.
    Fake/decorated generators without a body builder record ``payload=None``.

    Generated proposals are returned unchanged. The state machine's trace thus
    retains original Jev probabilities, while the separate audit records the
    feedback the generator saw. The decorator does not own/close its generator.
    """

    def __init__(self, base_generator: Any, scheme: FeedbackScheme | str = "numeric", *,
                 before_generate: Callable[[dict], None] | None = None):
        if not callable(getattr(base_generator, "generate", None)):
            raise ValueError("The base generator must expose generate(example, history).")
        self.base_generator = base_generator
        self.scheme = FeedbackScheme.from_id(scheme) if isinstance(scheme, str) else scheme
        if not isinstance(self.scheme, FeedbackScheme):
            raise ValueError("scheme must be a FeedbackScheme or a registered scheme ID.")
        if before_generate is not None and not callable(before_generate):
            raise ValueError("before_generate must be callable or null.")
        self.before_generate = before_generate
        self.last_request: dict | None = None

    def generate(self, example: dict, history: list[dict] | None = None) -> dict:
        transformed = self.scheme.transform_history(history)
        public = {key: deepcopy(example[key]) for key in ("messages", "tools") if key in example}
        payload = None
        frozen_builder = getattr(self.base_generator, "_payload", None)
        provider_builder = getattr(self.base_generator, "payload", None)
        if callable(frozen_builder):
            payload = frozen_builder(public, transformed)
        elif callable(provider_builder):
            payload = provider_builder(public["messages"], public.get("tools") or [], transformed)
        audit = {
            "scheme_id": self.scheme.scheme_id,
            "scheme_sha256": self.scheme.config_sha256,
            "history": deepcopy(transformed),
            "payload": deepcopy(payload),
            "payload_sha256": _sha256(payload) if payload is not None else None,
        }
        self.last_request = deepcopy(audit)
        if self.before_generate is not None:
            self.before_generate(deepcopy(audit))
        return self.base_generator.generate(public, transformed)
