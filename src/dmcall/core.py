"""DM-Call names over the shared implementation in :mod:`dmgfc.core`.

``DMCall`` is the public class. Its method names (``dmcall``, ``dmcall-f``, ``dmcall-a``,
``dmcall-latest``) select the same repair loops as the earlier ``dmgfc`` names, and every
earlier name keeps working.
"""

from __future__ import annotations

from dmgfc import core as _shared
from dmgfc.core import DMFC, DMGFC, JBFC, Result, Verdict, verify  # noqa: F401

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


class DMCall(DMGFC):
    """Decision-model-guided function calling (DM-Call).

    Same constructor and behavior as :class:`dmgfc.DMGFC`; the default method is ``"dmcall"``
    (both checks, cumulative feedback).
    """

    def __init__(self, model, *, method: str = "dmcall", **options):
        super().__init__(model, method=public_method(method), **options)
