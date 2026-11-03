"""DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback.

A model proposes a function call; a decision model (Jev by default) checks its function
and arguments before execution, and the same model repairs rejected calls with the
accumulated feedback.

    import dmcall

    def get_weather(city: str, unit: str = "celsius") -> dict:
        \"\"\"Current weather for a city.\"\"\"
        ...

    agent = dmcall.DMCall("openai:gpt-5-nano", tools=[get_weather])  # OPENAI_API_KEY, TYPESAFE_API_KEY
    result = agent.run("Is it warm in Seoul right now?")
    result.call, result.accepted, result.output

``dmcall`` is the public name. It shares one implementation with the earlier ``dmgfc``,
``dmfc`` and ``jbfc`` packages, which stay importable as aliases so that recorded
experiments and their source fingerprints are unchanged.
"""

from dmgfc import *  # noqa: F401,F403
from dmgfc import __all__ as _shared_names
from dmgfc import __version__  # noqa: F401

from .core import DMCall, METHODS

__all__ = ["DMCall", *(name for name in _shared_names if name != "METHODS"), "METHODS"]
