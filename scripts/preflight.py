"""Preflight for a live run: one generator call and one Jev call, no credentials printed.

Tries reasoning effort none, minimal, low, then the model default, and writes the
first working setting to the env file given as the first argument. An optional
second argument replaces that list (comma-separated; "default" is the model
default). Uses the generator and Jev settings from the environment and .env.
"""
import os
import shlex
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from jev_fc.benchmarks import load_examples  # noqa: E402
from jev_fc.jev import JevAPIError, JevVerifier  # noqa: E402
from jev_fc.llm import FunctionCallingLLM, LLMAPIError  # noqa: E402


def provider_message(response):
    try:
        error = response.json().get("error")
        message = error.get("message") if isinstance(error, dict) else error
    except ValueError:
        message = None
    return str(message or "")[:300]


def main(env_file, efforts=("none", "minimal", "low", "")):
    row = next(e for e in load_examples("data/bfcl/examples.jsonl") if e["category"] == "simple_python")
    example = {key: row[key] for key in ("id", "category", "messages", "tools")}
    chosen = None
    for effort in efforts:
        llm = FunctionCallingLLM(reasoning_effort=effort, max_retries=0)
        label = effort or "(model default)"
        try:
            started = time.perf_counter()
            out = llm.generate(example)
            print(f"generator OK: model={llm.model} returned={out['model']} effort={label} "
                  f"{time.perf_counter() - started:.1f}s tokens={out['usage']}")
            chosen = (effort, out)
        except LLMAPIError as exc:
            detail = ""
            if exc.status_code is not None:
                raw = httpx.post(llm.endpoint, headers=llm._headers, json=llm._payload(example, None), timeout=60)
                detail = provider_message(raw)
            print(f"generator failed: effort={label} HTTP {exc.status_code}: {detail or exc}")
            if exc.status_code in (401, 403, 404) or exc.status_code is None:
                break  # route, key or network problem: other efforts fail the same way
        finally:
            llm.close()
        if chosen:
            break
    if chosen is None:
        print("STOPPED: no working generator setting. Send these lines to Claude.")
        return 2
    effort, out = chosen
    verifier = JevVerifier(max_retries=0)
    try:
        started = time.perf_counter()
        check = verifier.verify(example, out["calls"])
        if check["api_calls"] == 0:  # unreadable proposal: ask the function question about no call
            check = verifier.verify(example, [])
        print(f"Jev OK: model={check['model']} {time.perf_counter() - started:.1f}s "
              f"p_function={check['function_probability']} p_arguments={check['arguments_probability']}")
    except JevAPIError as exc:
        print(f"Jev failed: HTTP {exc.status_code}: {exc}")
        print("STOPPED: Jev did not answer. Send these lines to Claude.")
        return 3
    finally:
        verifier.close()
    Path(env_file).write_text(f"export LLM_REASONING_EFFORT={shlex.quote(effort)}\n")
    print(f"preflight passed: LLM_REASONING_EFFORT={effort or '(model default)'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: preflight.py ENV_FILE [EFFORTS]  (EFFORTS e.g. none or low,default)")
    if len(sys.argv) == 3:
        chosen = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
        if not chosen or not all(item in {"none", "minimal", "low", "medium", "high", "default"} for item in chosen):
            sys.exit("EFFORTS must list none, minimal, low, medium, high or default")
        sys.exit(main(sys.argv[1], ["" if item == "default" else item for item in chosen]))
    sys.exit(main(sys.argv[1]))
