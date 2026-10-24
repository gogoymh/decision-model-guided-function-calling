"""DMGFC evaluation entry point backed by the unchanged research engine.

The public method names are translated at the command-line boundary. The engine
continues to save its original ``jev_*`` method IDs and source fingerprints.
The legacy ``dmfc-eval`` and ``jev-fc`` commands keep their original entry points.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from io import StringIO

from jev_fc.cli import main as _engine_main

from .core import _CANONICAL_METHODS, _method


def _option_end(arguments: Sequence[str]) -> int:
    """The standalone separator ends option parsing, including help handling."""
    return arguments.index("--") if "--" in arguments else len(arguments)


def _help_requested(arguments: Sequence[str]) -> bool:
    return any(argument in {"--help", "-h"} for argument in arguments[:_option_end(arguments)])


def _arguments(arguments: Sequence[str]) -> list[str]:
    """Translate actual run options while preserving literal paths and parser errors."""
    result = list(arguments)
    if not result or result[0] != "run" or _help_requested(result):
        return result
    end = _option_end(result)
    for index in range(1, end):
        argument = result[index]
        if argument == "--methods":
            # A missing value or a following option belongs to argparse, not semantic validation.
            if index + 1 < end and not result[index + 1].startswith("-"):
                result[index + 1] = ",".join(_method(name) for name in result[index + 1].split(","))
        elif argument.startswith("--methods="):
            methods = argument.partition("=")[2]
            result[index] = "--methods=" + ",".join(_method(name) for name in methods.split(","))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run evaluation commands; only ``run`` sends paid requests when configured."""
    try:
        arguments = _arguments(sys.argv[1:] if argv is None else argv)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    original = sys.argv
    try:
        # Keep the canonical command name for console scripts and ``python -m dmgfc``.
        sys.argv = ["dmgfc-eval", *arguments]
        if _help_requested(arguments):
            # Format help at the public boundary without patching the frozen parser.
            output = StringIO()
            try:
                with redirect_stdout(output):
                    return _engine_main()
            finally:
                help_text = output.getvalue().replace("JBFC", "DMGFC")
                print(help_text, end="")
                if help_text and arguments[0] == "run":
                    print("\nPublic --methods names: " + ", ".join(_CANONICAL_METHODS) + ".")
                    print("Legacy aliases and recorded jev_* IDs remain accepted; saved records retain those IDs.")
        return _engine_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    sys.exit(main())
