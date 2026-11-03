"""``dmcall-eval``: the evaluation command under the DM-Call names.

It translates DM-Call method names and delegates to the shared ``dmgfc-eval`` entry point,
which saves the original ``jev_*`` method IDs and source fingerprints.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from io import StringIO

from dmgfc import cli as _shared

from .core import CANONICAL_METHODS, public_method


def _arguments(arguments: Sequence[str]) -> list[str]:
    result = list(arguments)
    if not result or result[0] != "run":
        return result
    end = _shared._option_end(result)
    for index in range(1, end):
        argument = result[index]
        if argument == "--methods" and index + 1 < end and not result[index + 1].startswith("-"):
            result[index + 1] = ",".join(public_method(name) for name in result[index + 1].split(","))
        elif argument.startswith("--methods="):
            methods = argument.partition("=")[2]
            result[index] = "--methods=" + ",".join(public_method(name) for name in methods.split(","))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(sys.argv[1:] if argv is None else argv)
    if not _shared._help_requested(arguments):
        return _shared.main(arguments)
    output = StringIO()
    try:
        with redirect_stdout(output):
            status = _shared.main(arguments)
    finally:
        text = output.getvalue()
        text = text.replace("dmgfc-eval", "dmcall-eval").replace("DMGFC", "DM-Call")
        for old, new in (("dmgfc-latest", "dmcall-latest"), ("dmgfc-f", "dmcall-f"), ("dmgfc-a", "dmcall-a")):
            text = text.replace(old, new)
        text = text.replace("Public --methods names: dmgfc,", "Public --methods names: dmcall,")
        if arguments and arguments[0] == "run" and "Public --methods names" not in text:
            text += "\nPublic --methods names: " + ", ".join(CANONICAL_METHODS) + ".\n"
        print(text, end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
