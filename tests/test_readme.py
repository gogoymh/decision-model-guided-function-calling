"""The README's code matches the library: its examples parse and its options exist."""

import ast
import inspect
import re
from pathlib import Path

import dmcall

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, flags=re.S)


def test_python_examples_parse():
    assert BLOCKS
    for block in BLOCKS:
        ast.parse(block)


def test_documented_options_are_real():
    parameters = inspect.signature(dmcall.DMCall).parameters
    options = next(block for block in BLOCKS if "preserve_initial_no_call=True" in block)
    call = ast.parse(options).body[0].value
    documented = {keyword.arg for keyword in call.keywords}
    assert documented <= set(parameters)
    for keyword in call.keywords:  # the documented values are the defaults
        if keyword.arg in parameters and keyword.arg not in {"tools", "threshold"}:
            assert ast.literal_eval(keyword.value) == parameters[keyword.arg].default, keyword.arg


def test_documented_names_exist():
    # Code in other languages (```go, ```ts, ...) uses those packages' names; check Python and prose only.
    python_view = re.sub(r"```(?!python\n)\w+\n.*?```", "", README, flags=re.S)
    for name in set(re.findall(r"dmcall\.([A-Za-z_]+)", python_view)):
        assert hasattr(dmcall, name), name
