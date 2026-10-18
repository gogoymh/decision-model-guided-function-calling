import pytest

from jev_fc.transport import gateway_mode, resolve_api_config


@pytest.mark.parametrize("provider,suffix", [
    ("openai", "/llm/openai/v1"), ("anthropic", "/llm/anthropic/v1"),
    ("typesafe", "/llm/typesafe"),
])
def test_apim_routes_share_subscription_auth_without_provider_secrets(monkeypatch, provider, suffix):
    monkeypatch.setenv("API_GATEWAY", "apim")
    monkeypatch.setenv("APIM_BASE_URL", "https://gateway.invalid/")
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "private-test-subscription")
    monkeypatch.setenv("LLM_API_KEY", "unused-provider-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "unused-typesafe-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://unused.invalid")
    config = resolve_api_config(provider)
    assert config.base_url == "https://gateway.invalid" + suffix
    assert config.headers == {"api-key": "private-test-subscription", "Content-Type": "application/json"}
    assert "private-test-subscription" not in repr(config)
    assert config.gateway == "apim"


def test_apim_does_not_fall_back_to_provider_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("APIM_SUBSCRIPTION_KEY", raising=False)
    with pytest.raises(ValueError, match="APIM_SUBSCRIPTION_KEY"):
        resolve_api_config("openai", gateway="apim")


@pytest.mark.parametrize("provider,header", [("openai", "Authorization"),
                                             ("typesafe", "Authorization"), ("anthropic", "x-api-key")])
def test_direct_provider_auth_remains_available(provider, header):
    config = resolve_api_config(provider, api_key="test-only", gateway="direct")
    assert header in config.headers and "api-key" not in config.headers


@pytest.mark.parametrize("url", ["https://secret@gateway.invalid", "https://gateway.invalid?key=secret",
                                  "https://gateway.invalid#secret", "file:///tmp/socket"])
def test_credentials_cannot_be_embedded_in_recorded_base_urls(url):
    with pytest.raises(ValueError) as error:
        resolve_api_config("openai", api_key="test-only", base_url=url, gateway="direct")
    assert "secret" not in str(error.value)


def test_invalid_gateway_fails_without_echoing_value():
    with pytest.raises(ValueError, match="direct or apim"):
        gateway_mode("not-a-valid-mode")


@pytest.mark.parametrize("azure_endpoint,explicit,expected", [
    (None, None, "https://gateway.invalid"),
    ("https://azure-gateway.invalid/", None, "https://azure-gateway.invalid"),
    ("https://unused.invalid", "https://explicit.invalid/", "https://explicit.invalid"),
])
def test_azure_apim_root_precedence_and_jev_route_unchanged(monkeypatch, azure_endpoint, explicit, expected):
    monkeypatch.setenv("APIM_BASE_URL", "https://gateway.invalid/")
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "subscription-only")
    monkeypatch.setenv("AZURE_API_KEY", "unused-azure-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-provider-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://unused-provider.invalid")
    if azure_endpoint is None:
        monkeypatch.delenv("AZURE_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("AZURE_ENDPOINT", azure_endpoint)
    config = resolve_api_config("openai", gateway="apim", base_url=explicit, azure_chat_completions=True)
    assert config.base_url == expected
    assert config.headers == {"api-key": "subscription-only", "Content-Type": "application/json"}
    assert "subscription-only" not in repr(config)
    assert resolve_api_config("typesafe", gateway="apim").base_url == "https://gateway.invalid/llm/typesafe"


def test_azure_apim_does_not_fall_back_to_azure_or_provider_credentials(monkeypatch):
    monkeypatch.delenv("APIM_SUBSCRIPTION_KEY", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "unused-azure-key")
    monkeypatch.setenv("LLM_API_KEY", "unused-provider-key")
    with pytest.raises(ValueError, match="APIM_SUBSCRIPTION_KEY"):
        resolve_api_config("openai", gateway="apim", azure_chat_completions=True)


@pytest.mark.parametrize("provider,gateway", [("typesafe", "apim"), ("openai", "direct")])
def test_azure_style_does_not_change_other_provider_or_direct_routing(provider, gateway):
    with pytest.raises(ValueError, match="requires the OpenAI provider and API_GATEWAY=apim"):
        resolve_api_config(provider, api_key="test-only", gateway=gateway, azure_chat_completions=True)


@pytest.mark.parametrize("url", ["https://SECRET@gateway.invalid", "https://gateway.invalid?key=SECRET",
                                  "https://gateway.invalid#SECRET"])
def test_azure_endpoint_rejects_embedded_credentials_without_echo(monkeypatch, url):
    monkeypatch.setenv("AZURE_ENDPOINT", url)
    with pytest.raises(ValueError) as caught:
        resolve_api_config("openai", api_key="test-only", gateway="apim", azure_chat_completions=True)
    assert "SECRET" not in str(caught.value)
