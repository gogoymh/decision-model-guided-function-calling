package dmcall

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

// Generators over plain HTTP: requests, decoding, unreadable outputs, keys and errors (ported from
// tests/test_providers.py).

var example = Example{Messages: exampleMessages, Tools: []Tool{weatherTool}}

var exampleHistory = []HistoryEntry{
	{Calls: busan(), Feedback: obj("failed_checks", []any{"arguments"}, "instruction", "Reconsider.")},
	{Round: 2, Calls: []Call{}, Feedback: obj("failed_checks", []any{"function_selection"})},
}

// chat is a Chat Completions response body; arguments may be JSON text or a value to encode.
func chat(finish string, content any, calls ...[2]any) *Object {
	toolCalls := []any{}
	for i, call := range calls {
		arguments, ok := call[1].(string)
		if !ok {
			arguments, _ = JSONText(call[1])
		}
		toolCalls = append(toolCalls, obj("id", "call_"+string(rune('0'+i)), "type", "function",
			"function", obj("name", call[0], "arguments", arguments)))
	}
	if finish == "" {
		finish = "stop"
		if len(toolCalls) > 0 {
			finish = "tool_calls"
		}
	}
	message := obj("role", "assistant", "content", content)
	if len(toolCalls) > 0 {
		message.Set("tool_calls", toolCalls)
	}
	return obj("model", "resolved-model", "choices", []any{obj("finish_reason", finish, "message", message)})
}

func serving(body any) func(sent) reply { return func(sent) reply { return reply{body: body} } }

func TestOpenAIRequestAndDecoding(t *testing.T) {
	isolate(t)
	body := chat("", nil, [2]any{"get_weather", obj("city", "Seoul")})
	body.Set("usage", obj("prompt_tokens", 120, "completion_tokens", 9, "prompt_tokens_details", obj("cached_tokens", 64)))
	client, seen := mock(serving(body))
	model, err := NewOpenAI("gpt-5-nano", WithAPIKey(secret), WithReasoningEffort("none"), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := model.Generate(ctx, example, exampleHistory)
	if err != nil {
		t.Fatal(err)
	}
	request := object((*seen)[0].body)
	messages := list(request.Value("messages"))
	if (*seen)[0].url != "https://api.openai.com/v1/chat/completions" ||
		(*seen)[0].header.Get("Authorization") != "Bearer "+secret {
		t.Errorf("request %s", (*seen)[0].url)
	}
	assertEqual(t, "system", messages[0], obj("role", "system", "content", SystemPrompt))
	if !strings.HasPrefix(object(messages[1]).Value("content").(string), "Accumulated revision history (oldest first; proposals NOT executed)") {
		t.Error("the history message must come first")
	}
	assertEqual(t, "original messages", messages[2:], MustParseJSON(`[{"role": "system", "content": "Be brief."},
		{"role": "user", "content": "Weather in Seoul?"}]`))
	assertEqual(t, "tools", request.Value("tools"), []any{weatherTool})
	if request.Value("tool_choice") != "auto" || request.Value("parallel_tool_calls") != false ||
		request.Value("max_completion_tokens") != int64(4096) || request.Value("reasoning_effort") != "none" {
		t.Errorf("settings %s", request)
	}
	if !Equal(proposal.Calls, seoul()) || proposal.Incomplete != "" || proposal.Model != "resolved-model" ||
		proposal.Provider != "openai" || proposal.APICalls != 1 {
		t.Errorf("proposal %+v", proposal)
	}
	assertEqual(t, "usage", proposal.Usage.jsonValue(), obj("input_tokens", 120, "cached_input_tokens", 64, "output_tokens", 9))
	if strings.Contains(model.String(), secret) {
		t.Error("String must not show the key")
	}
}

func TestHistoryListsRoundsOldestFirst(t *testing.T) {
	client, seen := mock(serving(chat("", nil)))
	model, _ := NewOpenAI("m", WithAPIKey("k"), WithHTTPClient(client))
	if _, err := model.Generate(ctx, example, exampleHistory); err != nil {
		t.Fatal(err)
	}
	content := object(list(object((*seen)[0].body).Value("messages"))[1]).Value("content").(string)
	inner := strings.SplitN(content, "\n", 2)[1]
	inner = inner[:strings.LastIndex(inner, "\nUse these previous judgments")]
	records := list(MustParseJSON(inner))
	if object(records[0]).Value("round") != int64(1) || object(records[1]).Value("round") != int64(2) ||
		!Equal(object(records[0]).Value("calls"), busan()) || !Equal(object(records[1]).Value("calls"), []any{}) {
		t.Errorf("records %s", inner)
	}
	if !strings.HasSuffix(content, "when answering the original request below.") {
		t.Error("the footer must end the message")
	}
}

func TestOpenAIOutcomes(t *testing.T) {
	refusal := chat("", nil)
	object(object(list(refusal.Value("choices"))[0]).Value("message")).Set("refusal", "I can't help with that.")
	for _, c := range []struct {
		name       string
		body       *Object
		calls      []Call
		incomplete string
	}{
		{"text", chat("", "No function applies."), []Call{}, ""},
		{"empty", chat("", nil), []Call{}, ""},
		{"length", chat("length", nil), nil, IncompleteLength},
		{"content filter", chat("content_filter", nil), nil, IncompleteContentFilter},
		{"refusal", refusal, nil, IncompleteRefusal},
		{"bad json", chat("", nil, [2]any{"get_weather", "{not json"}), []Call{{Name: "get_weather"}}, ""},
		{"nan arguments", chat("", nil, [2]any{"get_weather", `{"city": NaN}`}), []Call{{Name: "get_weather"}}, ""},
	} {
		client, _ := mock(serving(c.body))
		model, _ := NewOpenAI("m", WithAPIKey("k"), WithHTTPClient(client))
		proposal, err := model.Generate(ctx, example, nil)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !Equal(callsValue(proposal.Calls), callsValue(c.calls)) || proposal.Incomplete != c.incomplete {
			t.Errorf("%s: %+v", c.name, proposal)
		}
		if (c.incomplete != "" || (len(c.calls) > 0 && c.calls[0].Arguments == nil)) && len(proposal.ParseErrors) == 0 {
			t.Errorf("%s: parse errors expected", c.name)
		}
	}
}

func TestProtocolViolationsAreErrors(t *testing.T) {
	for _, body := range []any{
		obj("error", obj("message", "x")),
		obj("choices", []any{}),
		obj("choices", []any{obj("message", obj("role", "user"))}),
		obj("choices", []any{obj("message", obj("role", "assistant", "content", 3))}),
		obj("choices", []any{obj("message", obj("role", "assistant", "tool_calls", "x"))}),
		[]any{1},
		"not json",
	} {
		client, _ := mock(serving(body))
		model, _ := NewOpenAI("m", WithAPIKey("k"), WithHTTPClient(client))
		_, err := model.Generate(ctx, example, nil)
		var failure *ProviderError
		if !errors.As(err, &failure) || failure.StatusCode != 200 {
			t.Errorf("body %v: %v", body, err)
		}
	}
}

func TestHTTPErrorsAreSanitizedAndRetriesBounded(t *testing.T) {
	client, seen := mock(func(sent) reply {
		return reply{status: 401, body: obj("error", obj("code", "invalid_api_key", "message", secret)),
			header: map[string]string{"x-request-id": "req-1"}}
	})
	model, _ := NewOpenAI("m", WithAPIKey(secret), WithMaxRetries(2), WithHTTPClient(client), WithSleep(noSleep))
	_, err := model.Generate(ctx, example, nil)
	var failure *ProviderError
	if !errors.As(err, &failure) || len(*seen) != 1 || failure.StatusCode != 401 || failure.ProviderCode != "invalid_api_key" ||
		failure.RequestID != "req-1" || strings.Contains(failure.Error(), secret) {
		t.Errorf("error %#v", err)
	}
	replies := []reply{{status: 429, header: map[string]string{"retry-after": "1"}}, {status: 503},
		{body: chat("", nil, [2]any{"get_weather", obj("city", "Seoul")})}}
	var delays []float64
	client, _ = mock(func(sent) reply {
		next := replies[0]
		replies = replies[1:]
		return next
	})
	retried, _ := NewOpenAI("m", WithAPIKey("k"), WithMaxRetries(2), WithHTTPClient(client),
		WithSleep(func(_ context.Context, delay time.Duration) error {
			delays = append(delays, delay.Seconds())
			return nil
		}))
	proposal, err := retried.Generate(ctx, example, nil)
	if err != nil || proposal.APICalls != 3 || !Equal(proposal.Calls, seoul()) {
		t.Errorf("retried %+v, %v", proposal, err)
	}
	assertEqual(t, "delays", delays, []float64{1, 2})
	client, _ = mock(func(sent) reply { return reply{status: 500} })
	exhausted, _ := NewOpenAI("m", WithAPIKey("k"), WithMaxRetries(1), WithHTTPClient(client), WithSleep(noSleep))
	if _, err := exhausted.Generate(ctx, example, nil); err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Errorf("exhausted: %v", err)
	}
	broken, _ := NewOpenAI("m", WithAPIKey("k"), WithMaxRetries(2), WithHTTPClient(failingTransport(errors.New("refused"))),
		WithSleep(noSleep))
	if _, err := broken.Generate(ctx, example, nil); !errors.As(err, &failure) || failure.Attempts != 3 ||
		!strings.Contains(failure.Error(), "transport failed after 3 attempts") {
		t.Errorf("transport: %v", err)
	}
}

func TestKeysComeFromOptionsOrEnvironment(t *testing.T) {
	isolate(t)
	if _, err := NewOpenAI("m"); err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Errorf("missing key: %v", err)
	}
	t.Setenv("OPENAI_API_KEY", "from-env")
	t.Setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1/")
	client, seen := mock(serving(chat("", nil)))
	model, err := NewOpenAI("m", WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(ctx, example, nil); err != nil {
		t.Fatal(err)
	}
	if (*seen)[0].header.Get("Authorization") != "Bearer from-env" || (*seen)[0].url != "https://proxy.example.com/v1/chat/completions" {
		t.Errorf("request %s %v", (*seen)[0].url, (*seen)[0].header)
	}
	if _, err := NewOpenAI("m", WithBaseURL("https://user:pass@example.com/v1")); err == nil || !strings.Contains(err.Error(), "base_url") {
		t.Errorf("credentials in base URL: %v", err)
	}
	if _, err := NewOpenAI("m", WithAPIKey("a\nb")); err == nil || !strings.Contains(err.Error(), "single-line") {
		t.Errorf("multi-line key: %v", err)
	}
	for _, bad := range []ProviderOption{WithMaxRetries(-1), WithMaxOutputTokens(-1), WithTimeout(0), WithSystemPrompt(" "),
		WithThinkingBudget(0), WithEndpoint("https://x"), WithPath("chat")} {
		if _, err := NewOpenAI("m", bad); err == nil {
			t.Errorf("expected an error for %T", bad)
		}
	}
	if _, err := NewAnthropic("m", WithAPIKey("k"), WithReasoningEffort("none")); err == nil {
		t.Error("reasoning effort applies only to OpenAI-compatible providers")
	}
}

func TestOpenAICompatibleHosts(t *testing.T) {
	isolate(t)
	for _, c := range []struct {
		build      func(string, ...ProviderOption) (*Provider, error)
		env, url   string
		tokenField string
	}{{NewXAI, "XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "max_tokens"},
		{NewOpenRouter, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", "max_tokens"}} {
		t.Setenv(c.env, "k")
		client, seen := mock(serving(chat("", nil)))
		model, err := c.build("m", WithHTTPClient(client))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := model.Generate(ctx, example, nil); err != nil {
			t.Fatal(err)
		}
		if (*seen)[0].url != c.url || object((*seen)[0].body).Value(c.tokenField) != int64(4096) {
			t.Errorf("%s: %s %s", c.env, (*seen)[0].url, (*seen)[0].raw)
		}
	}
}

func TestMetaMuseRequest(t *testing.T) {
	isolate(t)
	t.Setenv("MODEL_API_KEY", secret)
	client, seen := mock(serving(chat("", nil, [2]any{"get_weather", obj("city", "Seoul")})))
	model, err := NewMeta("muse-spark-1.3", WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := model.Generate(ctx, example, exampleHistory)
	if err != nil {
		t.Fatal(err)
	}
	body := object((*seen)[0].body)
	if (*seen)[0].url != "https://api.meta.ai/v1/chat/completions" || (*seen)[0].header.Get("Authorization") != "Bearer "+secret {
		t.Errorf("request %s %v", (*seen)[0].url, (*seen)[0].header)
	}
	if body.Value("model") != "muse-spark-1.3" || body.Value("tool_choice") != "auto" ||
		body.Value("max_completion_tokens") != int64(4096) {
		t.Errorf("body %s", (*seen)[0].raw)
	}
	// Meta rejects reasoning_effort "none" for Muse models; neither option is sent unless set.
	if body.Has("parallel_tool_calls") || body.Has("reasoning_effort") {
		t.Errorf("body %s", (*seen)[0].raw)
	}
	if !Equal(proposal.Calls, seoul()) || proposal.Provider != "meta" {
		t.Errorf("proposal %+v", proposal)
	}
	t.Setenv("MODEL_API_KEY", "")
	if _, err := NewMeta("muse-spark-1.3"); err == nil || !strings.Contains(err.Error(), "MODEL_API_KEY") {
		t.Errorf("missing key: %v", err)
	}
	built, err := ProviderFromSpec("meta:muse-spark-1.3", WithAPIKey("k"), WithReasoningEffort("low"),
		WithParallelToolCalls(true))
	if err != nil || built.Name() != "meta" {
		t.Fatalf("spec: %v, %v", built, err)
	}
	payload, err := built.Payload(example, nil)
	if err != nil || payload.Value("reasoning_effort") != "low" || payload.Value("parallel_tool_calls") != true {
		t.Errorf("payload %s, %v", payload, err)
	}
}

func TestAnyOpenAICompatibleServerNeedsNoKey(t *testing.T) {
	isolate(t)
	client, seen := mock(serving(chat("", nil)))
	model, err := NewOpenAICompatible("gemma-4-31b-it", WithBaseURL("http://localhost:8000/v1"), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(ctx, Example{Messages: Prompt("hi")}, nil); err != nil {
		t.Fatal(err)
	}
	body := object((*seen)[0].body)
	if (*seen)[0].url != "http://localhost:8000/v1/chat/completions" || (*seen)[0].header.Get("Authorization") != "" ||
		body.Has("tools") || body.Has("tool_choice") || body.Has("parallel_tool_calls") {
		t.Errorf("request %s", (*seen)[0].raw)
	}
	if _, err := NewOpenAICompatible("m"); err == nil || !strings.Contains(err.Error(), "base_url") {
		t.Errorf("missing base URL: %v", err)
	}
	unset, seen := mock(serving(chat("", nil)))
	model, _ = NewOpenAICompatible("m", WithBaseURL("http://localhost:1/v1"), WithoutParallelToolCalls(),
		WithMaxOutputTokens(0), WithHTTPClient(unset))
	if _, err := model.Generate(ctx, example, nil); err != nil {
		t.Fatal(err)
	}
	body = object((*seen)[0].body)
	if body.Has("parallel_tool_calls") || body.Has("max_completion_tokens") || !body.Has("tools") {
		t.Errorf("request %s", (*seen)[0].raw)
	}
}

func TestAzureDeploymentURLAndHeader(t *testing.T) {
	isolate(t)
	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/")
	t.Setenv("AZURE_OPENAI_API_KEY", secret)
	client, seen := mock(serving(chat("", nil)))
	model, err := NewAzureOpenAI("my deployment", WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(ctx, example, nil); err != nil {
		t.Fatal(err)
	}
	if (*seen)[0].url != "https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions?api-version=2024-12-01-preview" ||
		(*seen)[0].header.Get("api-key") != secret || (*seen)[0].header.Get("Authorization") != "" {
		t.Errorf("request %s %v", (*seen)[0].url, (*seen)[0].header)
	}
	if _, err := NewAzureOpenAI("d", WithAPIVersion("latest")); err == nil || !strings.Contains(err.Error(), "API version") {
		t.Errorf("api version: %v", err)
	}
	t.Setenv("AZURE_OPENAI_ENDPOINT", "")
	if _, err := NewAzureOpenAI("d"); err == nil || !strings.Contains(err.Error(), "AZURE_OPENAI_ENDPOINT") {
		t.Errorf("missing endpoint: %v", err)
	}
}

func TestAnthropicRequestAndDecoding(t *testing.T) {
	isolate(t)
	body := MustParseJSON(`{"type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "tool_use",
		"content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Checking."},
		{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}],
		"usage": {"input_tokens": 50, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 20, "output_tokens": 7}}`).(*Object)
	client, seen := mock(serving(body))
	model, err := NewAnthropic("claude-x", WithAPIKey(secret), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := model.Generate(ctx, example, exampleHistory)
	if err != nil {
		t.Fatal(err)
	}
	request := object((*seen)[0].body)
	if (*seen)[0].url != "https://api.anthropic.com/v1/messages" || (*seen)[0].header.Get("x-api-key") != secret ||
		(*seen)[0].header.Get("anthropic-version") != "2023-06-01" || request.Value("system") != SystemPrompt+"\n\nBe brief." {
		t.Errorf("request %s", (*seen)[0].raw)
	}
	var roles []string
	for _, message := range list(request.Value("messages")) {
		roles = append(roles, object(message).Value("role").(string))
	}
	assertEqual(t, "roles", roles, []string{"user", "user"})
	assertEqual(t, "input schema", object(list(request.Value("tools"))[0]).Value("input_schema"), weatherTool.Parameters)
	assertEqual(t, "tool choice", request.Value("tool_choice"), obj("type", "auto", "disable_parallel_tool_use", true))
	if !Equal(proposal.Calls, seoul()) || proposal.Content != "Checking." {
		t.Errorf("proposal %+v", proposal)
	}
	assertEqual(t, "usage", proposal.Usage.jsonValue(), obj("input_tokens", 100, "cached_input_tokens", 30, "output_tokens", 7))
	for stop, reason := range map[string]string{"max_tokens": IncompleteLength, "refusal": IncompleteRefusal} {
		cut := body.Clone()
		cut.Set("stop_reason", stop)
		cut.Set("content", []any{})
		client, _ := mock(serving(cut))
		model, _ := NewAnthropic("c", WithAPIKey("k"), WithHTTPClient(client))
		if proposal, err := model.Generate(ctx, example, nil); err != nil || proposal.Incomplete != reason {
			t.Errorf("%s: %+v, %v", stop, proposal, err)
		}
	}
	if _, err := model.Generate(ctx, Example{Messages: []Message{{Role: "tool", Content: "x"}}}, nil); err == nil {
		t.Error("Anthropic accepts only user and assistant turns")
	}
}

func TestGeminiRequestAndDecoding(t *testing.T) {
	isolate(t)
	body := MustParseJSON(`{"modelVersion": "gemini-x-001", "candidates": [{"finishReason": "STOP", "content": {"role": "model",
		"parts": [{"text": "thinking", "thought": true}, {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]}}],
		"usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 6, "thoughtsTokenCount": 40}}`)
	client, seen := mock(serving(body))
	model, err := NewGemini("models/gemini-x", WithAPIKey(secret), WithTemperature(0), WithThinkingBudget(0),
		WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	messages := append(append([]Message{}, exampleMessages...), Message{Role: "assistant", Content: "Which unit?"},
		Message{Role: "user", Content: "Celsius."})
	proposal, err := model.Generate(ctx, Example{Messages: messages, Tools: []Tool{weatherTool}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	request := object((*seen)[0].body)
	if (*seen)[0].url != "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent" ||
		(*seen)[0].header.Get("x-goog-api-key") != secret {
		t.Errorf("request %s", (*seen)[0].url)
	}
	instruction := object(list(object(request.Value("systemInstruction")).Value("parts"))[0]).Value("text")
	if instruction != SystemPrompt+"\n\nBe brief." {
		t.Errorf("system instruction %v", instruction)
	}
	var roles []string
	for _, content := range list(request.Value("contents")) {
		roles = append(roles, object(content).Value("role").(string))
	}
	assertEqual(t, "roles", roles, []string{"user", "model", "user"})
	declaration := object(list(object(list(request.Value("tools"))[0]).Value("functionDeclarations"))[0])
	assertEqual(t, "declaration", declaration.Value("parametersJsonSchema"), weatherTool.Parameters)
	if declaration.Has("parameters") {
		t.Error("Gemini declarations use parametersJsonSchema")
	}
	assertEqual(t, "tool config", request.Value("toolConfig"), obj("functionCallingConfig", obj("mode", "AUTO")))
	assertEqual(t, "generation config", request.Value("generationConfig"),
		obj("maxOutputTokens", 4096, "temperature", 0, "thinkingConfig", obj("thinkingBudget", 0)))
	if !Equal(proposal.Calls, seoul()) || proposal.Content != "" || proposal.Model != "gemini-x-001" {
		t.Errorf("proposal %+v", proposal)
	}
	assertEqual(t, "usage", proposal.Usage.jsonValue(), obj("input_tokens", 80, "output_tokens", 46))
}

func TestProviderSpecs(t *testing.T) {
	isolate(t)
	for _, name := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY",
		"MODEL_API_KEY"} {
		t.Setenv(name, "k")
	}
	for spec, name := range map[string]string{"openai:gpt-5-nano": "openai", "anthropic:claude-x": "anthropic",
		"claude:c": "anthropic", "gemini:g": "gemini", "google:g": "gemini", "xai:grok-4": "xai", "grok:grok-4": "xai",
		"openrouter:meta/llama": "openrouter", "meta:muse-spark-1.3": "meta", "OpenAI:m": "openai"} {
		built, err := ProviderFromSpec(spec)
		if err != nil {
			t.Fatalf("%s: %v", spec, err)
		}
		if built.Name() != name || built.Model() != spec[strings.Index(spec, ":")+1:] {
			t.Errorf("%s: %s %s", spec, built.Name(), built.Model())
		}
	}
	built, err := ProviderFromSpec("azure:dep", WithEndpoint("https://r.openai.azure.com"), WithAPIKey("k"))
	if err != nil || built.Name() != "azure" {
		t.Errorf("azure: %v, %v", built, err)
	}
	for _, bad := range []string{"gpt-5-nano", "openai:", "mystery:model"} {
		if _, err := ProviderFromSpec(bad); err == nil {
			t.Errorf("%s must be refused", bad)
		}
	}
}

func failingTransport(err error) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, err })}
}
