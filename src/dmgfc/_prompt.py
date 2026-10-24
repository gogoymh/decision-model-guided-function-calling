"""The generator prompt of the paper, shared by every provider.

``SYSTEM_PROMPT`` and the revision-history message are exactly those used in the
experiments (``jev_fc``), so every provider sends the paper's prompt by default.
"""

from __future__ import annotations

import json
from copy import deepcopy

from jev_fc.llm import SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT", "functions_of", "history_message", "with_history"]

HISTORY_HEADER = "Accumulated revision history (oldest first; proposals NOT executed):\n"
HISTORY_FOOTER = ("\nUse these previous judgments as feedback. Reconsider both function choice and arguments "
                  "when answering the original request below.")


def history_message(history: list[dict]) -> dict:
    """One user message with every earlier proposal and its feedback, oldest first."""
    accumulated = [{"round": item.get("round", index + 1), "calls": deepcopy(item["calls"]),
                    "feedback": deepcopy(item["feedback"])} for index, item in enumerate(history)]
    return {"role": "user",
            "content": HISTORY_HEADER + json.dumps(accumulated, ensure_ascii=False) + HISTORY_FOOTER}


def with_history(messages: list[dict], history: list[dict] | None) -> list[dict]:
    """The conversation as sent: role and content only, preceded by the revision history if any."""
    result = [{"role": message["role"], "content": deepcopy(message["content"])} for message in messages]
    if history:
        result.insert(0, history_message(history))
    return result


def functions_of(tools: list[dict]) -> list[dict]:
    """Function definitions (name, description, parameters) from OpenAI-style or flat tool dicts."""
    functions = []
    for tool in tools:
        definition = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(definition, dict):
            raise ValueError("Every tool must be a function definition object.")
        functions.append({key: deepcopy(definition[key])
                          for key in ("name", "description", "parameters") if key in definition})
    return functions
