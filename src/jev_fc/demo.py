"""Synthetic example for testing orchestration; never part of empirical reports."""

from .pipeline import run_example


def demo():
    example = {"id": "synthetic_weather", "category": "synthetic", "messages": [
        {"role": "user", "content": "Get the weather in Seoul in Celsius."}], "tools": [{
        "type": "function", "function": {"name": "weather", "description": "Get weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city", "unit"]}}}]}

    class Generator:
        def generate(self, example, history):
            return {"calls": [{"name": "weather", "arguments": {
                "city": "Seoul", "unit": "celsius" if history else "fahrenheit"}}],
                "model": "offline-demo", "usage": {}, "latency_s": 0, "api_calls": 0}

    class Verifier:
        def verify(self, example, calls):
            correct = calls[0]["arguments"]["unit"] == "celsius"
            return {"function_pass": True, "arguments_pass": correct,
                    "function_probability": .99, "arguments_probability": .99 if correct else .05,
                    "accepted": correct, "feedback": {"function_pass": True, "arguments_pass": correct},
                    "model": "offline-demo", "usage": {}, "latency_s": 0, "api_calls": 0}

    executions = []
    record = run_example(example, Generator(), Verifier(), executor=lambda calls: executions.append(calls))
    record.update({"synthetic": True, "note": "Demonstration only. No LLM or Jev API was called.",
                   "execution_count": len(executions)})
    return record
