import json
from types import SimpleNamespace

import httpx
import pytest

from jev_fc.cli import _prices, check_config
from jev_fc import cli
from jev_fc.pipeline import METHODS


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for key in ("API_GATEWAY", "APIM_SUBSCRIPTION_KEY", "LLM_API_KEY", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY", "LLM_MODEL", "LLM_API_STYLE",
                "LLM_REASONING_EFFORT", "LLM_TEMPERATURE", "LLM_MAX_COMPLETION_TOKENS",
                "LLM_INPUT_PRICE", "LLM_CACHED_INPUT_PRICE", "LLM_OUTPUT_PRICE",
                "LLM_CACHE_WRITE_5M_PRICE", "LLM_CACHE_WRITE_1H_PRICE", "JEV_INPUT_PRICE"):
        monkeypatch.delenv(key, raising=False)


def test_check_config_never_calls_network_or_prints_key(monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("Configuration check attempted a network request")
    monkeypatch.setattr(httpx.Client, "request", no_network)
    monkeypatch.setenv("API_GATEWAY", "apim")
    monkeypatch.setenv("APIM_BASE_URL", "https://gateway.invalid")
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "private-config-key")
    assert check_config() == 0
    output = capsys.readouterr().out
    assert "private-config-key" not in output
    info = json.loads(output)
    assert info["network_requests"] == 0
    assert info["llm_endpoint"] == "https://gateway.invalid/llm/openai/v1/responses"
    assert info["jev_endpoint"] == "https://gateway.invalid/llm/typesafe/v1/systemone"
    assert info["llm_key_configured"] and info["jev_key_configured"]


def test_check_config_reports_missing_shared_key_without_calling(monkeypatch, capsys):
    monkeypatch.setenv("API_GATEWAY", "apim")
    assert check_config() == 1
    info = json.loads(capsys.readouterr().out)
    assert not info["llm_key_configured"] and not info["jev_key_configured"]


def test_nondefault_models_do_not_inherit_openai_reference_prices():
    prices = _prices(SimpleNamespace(provider="anthropic", model="custom-model"))
    assert prices["llm_input"] is None and prices["llm_output"] is None
    assert prices["llm_cached_input"] is None


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "not-a-number"])
def test_invalid_price_fails_before_requests(monkeypatch, value):
    monkeypatch.setenv("LLM_INPUT_PRICE", value)
    with pytest.raises(ValueError, match="LLM_INPUT_PRICE"):
        _prices(SimpleNamespace(provider="openai", model="gpt-5.4-mini-2026-03-17"))


def test_cli_default_budget_is_explicitly_exploratory_and_selection_option_is_parsed(monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run", lambda args: observed.append(args))
    monkeypatch.setattr("sys.argv", ["jev-fc", "run"])
    assert cli.main() == 0
    assert observed[-1].max_repairs == 0
    assert observed[-1].repair_budget_selection is None
    assert observed[-1].output == "results/exploratory"
    assert observed[-1].methods.split(",") == list(METHODS)
    assert len(METHODS) == 7 and "jev_joint_latest" in METHODS
    monkeypatch.setattr("sys.argv", ["jev-fc", "run", "--max-repairs", "5",
                                    "--repair-budget-selection", "selection.json"])
    assert cli.main() == 0
    assert observed[-1].max_repairs == 5
    assert observed[-1].repair_budget_selection == "selection.json"
