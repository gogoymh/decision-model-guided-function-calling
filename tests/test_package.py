"""The installed package: one public module, its exports and its metadata."""

import importlib
import tomllib
from pathlib import Path

import pytest

import dmcall

ROOT = Path(__file__).resolve().parents[1]


def test_public_names_resolve():
    for name in dmcall.__all__:
        assert getattr(dmcall, name) is not None, name


def test_version_matches_the_project_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert dmcall.__version__ == project["version"] and project["name"] == "dmcall"


@pytest.mark.parametrize("old", ["dmgfc", "dmfc", "jbfc", "jev_fc"])
def test_earlier_packages_are_gone(old):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)


def test_only_dmcall_is_packaged():
    wheel = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/dmcall"]
