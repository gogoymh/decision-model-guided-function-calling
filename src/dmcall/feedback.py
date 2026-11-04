"""Hybrid repair feedback, the representation the paper uses for DM-Call.

The decision model's probabilities, verdicts and the acceptance gate are unchanged. Only the
copy of earlier feedback that the generator reads changes: each probability keeps its value
and gains a verbal label from five equal-width bands (very likely incorrect, likely
incorrect, uncertain, likely correct, very likely correct). ``HybridFeedbackScheme`` is the
same class as ``HybridFeedbackScheme`` in the experiment that produced the paper's tables
(``hybrid_correctness_5``), so the library sends the generator the same feedback.
"""

from __future__ import annotations

from copy import deepcopy

from dmgfc.feedback import FeedbackGenerator, FeedbackScheme

__all__ = ["HybridFeedbackScheme", "HybridFeedbackGenerator", "PROBABILITY_LABELS"]

PROBABILITY_LABELS = {
    "probability_yes": "feedback_label", "function_probability": "function_label",
    "arguments_probability": "arguments_label", "p_f": "function_label", "p_a": "arguments_label",
}


class HybridFeedbackScheme(FeedbackScheme):
    """Keep numeric feedback verbatim and append frozen verifier-band labels."""

    def __init__(self):
        super().__init__("correctness", 5)

    @property
    def scheme_id(self):
        return "hybrid_correctness_5"

    def config(self):
        config = super().config()
        config.update(version=1, scheme_id=self.scheme_id, family="hybrid_correctness",
            probability_field_replacement={}, probability_labels_added=dict(PROBABILITY_LABELS),
            threshold_fields_removed=[], original_instruction_changed=False,
            numeric_probability_values_preserved=True,
            source_text_scheme_sha256=FeedbackScheme.from_id("correctness_5").config_sha256)
        return config

    def transform_feedback(self, feedback):
        if not isinstance(feedback, dict):
            raise ValueError("Repair feedback must be an object.")

        def transform(value):
            if isinstance(value, list):
                return [transform(item) for item in value]
            if not isinstance(value, dict):
                return deepcopy(value)
            result = {}
            for key, item in value.items():
                # Schema errors are task input diagnostics, not verifier scores.
                result[key] = deepcopy(item) if key == "schema_errors" else transform(item)
                if key in PROBABILITY_LABELS:
                    label_field = PROBABILITY_LABELS[key]
                    if label_field in value or label_field in result:
                        raise ValueError("Repair feedback has conflicting added judgment fields.")
                    result[label_field] = self.label(item)
            return result

        return transform(feedback)


class HybridFeedbackGenerator(FeedbackGenerator):
    """A generator that reads Hybrid feedback; every other attribute is the wrapped generator's.

    Proposals come back unchanged, so traces keep the decision model's original probabilities.
    """

    def __init__(self, base_generator):
        super().__init__(base_generator, HybridFeedbackScheme())

    def __getattr__(self, name):
        base = self.__dict__.get("base_generator")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __repr__(self) -> str:
        return repr(self.base_generator)
