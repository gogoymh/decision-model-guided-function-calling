"""DM-Call names over the shared implementation in :mod:`dmgfc.core`.

``DMCall`` is the public class. Its method names (``dmcall``, ``dmcall-f``, ``dmcall-a``,
``dmcall-latest``) select the same repair loops as the earlier ``dmgfc`` names, and every
earlier name keeps working.
"""

from __future__ import annotations

from dmgfc import core as _shared
from dmgfc.core import DMFC, DMGFC, JBFC, Result, Verdict, verify  # noqa: F401

from .feedback import HybridFeedbackGenerator

__all__ = ["DMCall", "DMGFC", "DMFC", "JBFC", "METHODS", "Result", "Verdict", "verify", "public_method"]

# Public DM-Call method names and the earlier names they share a loop with.
_ALIASES = {
    "dmcall": "dmgfc",
    "dm-call": "dmgfc",
    "dmcall-latest": "dmgfc-latest",
    "dmcall-l": "dmgfc-l",
    "dmcall-f": "dmgfc-f",
    "dmcall-a": "dmgfc-a",
}
METHODS = {**{name: _shared.METHODS[target] for name, target in _ALIASES.items()}, **_shared.METHODS}
CANONICAL_METHODS = ("dmcall", "dmcall-latest", "dmcall-f", "dmcall-a", "schema", "self-refine", "direct")


def public_method(name) -> str:
    """Translate a DM-Call method name to the shared name; other names pass through."""
    key = str(name).strip().lower().replace("_", "-")
    return _ALIASES.get(key, name)


FEEDBACK = ("hybrid", "numeric")
# Methods whose repair feedback carries decision-model probabilities.
_DECISION_METHODS = {"jev_joint", "jev_joint_latest", "jev_function", "jev_arguments"}


class DMCall(DMGFC):
    """Decision-model-guided function calling (DM-Call).

    Same constructor as :class:`dmgfc.DMGFC`, with the paper's settings as defaults: the
    method is ``"dmcall"`` (both checks, cumulative feedback), a readable initial no-call is
    returned without a check (``preserve_initial_no_call=True``), and repair feedback is
    ``"hybrid"``: each decision-model probability keeps its value and gains a verbal label.
    ``feedback="numeric"`` sends the probabilities alone, as :class:`dmgfc.DMGFC` does.
    """

    def __init__(self, model, *, method: str = "dmcall", feedback: str = "hybrid", **options):
        if feedback not in FEEDBACK:
            raise ValueError(f"feedback must be one of {', '.join(FEEDBACK)}.")
        super().__init__(model, method=public_method(method), **options)
        self.feedback = feedback
        if feedback == "hybrid" and self.method in _DECISION_METHODS:
            self.model = HybridFeedbackGenerator(self.model)
