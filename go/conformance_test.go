package dmcall

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"testing"
)

// The language-neutral conformance files in ../spec, replayed the way tests/test_spec.py and
// tests/test_paper_parity.py replay them for the Python reference.

func TestPromptsFileMatchesTheLibrary(t *testing.T) {
	data := spec(t, "prompts.json")
	checks := map[string]any{
		"system_prompt":      SystemPrompt,
		"history_header":     HistoryHeader,
		"history_footer":     HistoryFooter,
		"function_question":  FunctionQuestion.Object(),
		"argument_question":  ArgumentQuestion.Object(),
		"repair_instruction": RepairInstruction,
		"labels":             Labels[:],
		"cut_points":         CutPoints[:],
		"incomplete_reasons": IncompleteReasons,
	}
	for key, value := range checks {
		assertEqual(t, key, value, data.Value(key))
	}
	fields := map[string]any{}
	for key, value := range labelFields {
		fields[key] = value
	}
	assertEqual(t, "label_fields", fields, data.Value("label_fields"))
	var thresholds []string
	for key := range thresholdFields {
		thresholds = append(thresholds, key)
	}
	sort.Strings(thresholds)
	assertEqual(t, "threshold_fields", thresholds, data.Value("threshold_fields"))
	var statuses []int
	for status := range RetryStatuses {
		statuses = append(statuses, status)
	}
	sort.Ints(statuses)
	assertEqual(t, "retry_statuses", statuses, data.Value("retry_statuses"))
	assertEqual(t, "defaults", map[string]any{
		"threshold": DefaultThreshold, "max_repairs": DefaultMaxRepairs, "checks": ChecksBoth,
		"history": HistoryCumulative, "feedback": FeedbackHybrid, "on_exhaustion": ExhaustionFirst,
		"preserve_initial_no_call": true, "execute_unverified": false, "max_output_tokens": 4096,
	}, data.Value("defaults"))
	jev := object(data.Value("jev"))
	assertEqual(t, "jev", map[string]any{"model": JevDefaultModel, "base_url": JevDefaultBaseURL,
		"path": "/v1/systemone", "feedback_source": jevFeedbackSource, "key_env": "TYPESAFE_API_KEY",
		"model_env": "TYPESAFE_MODEL", "base_url_env": "TYPESAFE_BASE_URL"}, jev)
	pplx := object(data.Value("pplx_decider"))
	assertEqual(t, "pplx_decider", map[string]any{"model_id": PPLXModelID, "revision": PPLXModelRevision,
		"calibration_temperature": PPLXCalibrationTemperature, "max_input_tokens": PPLXMaxInputTokens,
		"server_model": PPLXServerModel, "wire_model": PPLXWireModel, "feedback_source": pplxFeedbackSource,
		"timeout_s": 180, "max_retries": 0}, pplx)

	// The defaults are the ones a client really uses.
	isolate(t)
	t.Setenv("TYPESAFE_API_KEY", "jev-env")
	client, err := New(&scriptedModel{})
	if err != nil {
		t.Fatal(err)
	}
	jevModel := client.DecisionModel().(*SystemOne)
	if client.MaxRepairs() != 3 || client.Checks() != "both" || client.History() != "cumulative" ||
		client.Feedback() != "hybrid" || client.OnExhaustion() != "first" || !client.PreserveInitialNoCall() ||
		client.ExecuteUnverified() || jevModel.Model() != "jev-1.13.0" || jevModel.FunctionThreshold() != 0.5 ||
		jevModel.ArgumentsThreshold() != 0.5 || jevModel.BaseURL() != "https://api.typesafe.ai" {
		t.Errorf("defaults differ from the paper's settings: %v", client)
	}
	decider, err := NewPPLXDecider("http://127.0.0.1:8000")
	if err != nil {
		t.Fatal(err)
	}
	if decider.timeout.Seconds() != 180 || decider.maxRetries != 0 {
		t.Errorf("PPLX Decider timeout %v, retries %d", decider.timeout, decider.maxRetries)
	}
	openai, err := NewOpenAI("m", WithAPIKey("k"))
	if err != nil {
		t.Fatal(err)
	}
	if openai.maxOutputTokens != 4096 {
		t.Errorf("max output tokens %d", openai.maxOutputTokens)
	}
}

func TestJSONText(t *testing.T) {
	for _, item := range list(spec(t, "conformance/json_text.json").Value("cases")) {
		c := object(item)
		input := c.Value("input").(string)
		value, err := ParseJSON([]byte(input))
		if err != nil {
			t.Errorf("%q: %v", input, err)
			continue
		}
		if got := text(t, value); got != c.Value("output") {
			t.Errorf("%q: got %q, want %q", input, got, c.Value("output"))
		}
	}
}

func TestSchemaCheck(t *testing.T) {
	data := spec(t, "conformance/schema_check.json")
	cases := list(data.Value("cases"))
	for _, item := range cases {
		c := object(item)
		t.Run(c.Value("name").(string), func(t *testing.T) {
			result, err := CheckSchema(mustTools(t, c.Value("tools")), c.Value("calls"))
			if err != nil {
				t.Fatal(err)
			}
			// Lists compare in order, so this checks error order and wording.
			assertEqual(t, "check_schema", result.Object(), c.Value("expected"))
		})
	}
	if len(cases) == 0 {
		t.Error("no schema cases found")
	}
	for _, item := range list(data.Value("configuration_errors")) {
		c := object(item)
		t.Run(c.Value("name").(string), func(t *testing.T) {
			if _, err := CheckSchema(mustTools(t, c.Value("tools")), []any{}); err == nil {
				t.Error("expected a configuration error")
			}
			// The client refuses the same tools.
			isolate(t)
			if _, err := New(&scriptedModel{}, WithTools(mustTools(t, c.Value("tools"))...),
				WithJevAPIKey("k")); err == nil {
				t.Error("expected New to refuse the tools")
			}
		})
	}
}

func TestFeedback(t *testing.T) {
	data := spec(t, "conformance/feedback.json")
	for _, item := range list(data.Value("labels")) {
		c := object(item)
		got, err := labelOf(c.Value("p"))
		if err != nil || got != c.Value("label") {
			t.Errorf("label(%v) = %v, %v; want %v", c.Value("p"), got, err, c.Value("label"))
		}
	}
	for _, bad := range list(data.Value("invalid_probabilities")) {
		if _, err := labelOf(bad); err == nil {
			t.Errorf("label(%v) should fail", bad)
		}
	}
	for i, item := range list(data.Value("represent")) {
		c := object(item)
		got, err := Represent(object(c.Value("feedback")), c.Value("representation").(string))
		if err != nil {
			t.Errorf("represent %d: %v", i, err)
			continue
		}
		assertEqual(t, fmt.Sprintf("represent %d (%s)", i, c.Value("representation")), got, c.Value("expected"))
		// Key order matters: the record is written into the history message.
		if text(t, got) != text(t, c.Value("expected")) {
			t.Errorf("represent %d: key order differs\n got: %s\nwant: %s", i, text(t, got), text(t, c.Value("expected")))
		}
	}
	for i, item := range list(data.Value("single_check")) {
		c := object(item)
		got, err := SingleCheck(object(c.Value("feedback")), c.Value("keep").(string))
		if err != nil {
			t.Fatal(err)
		}
		if text(t, got) != text(t, c.Value("expected")) {
			t.Errorf("single_check %d\n got: %s\nwant: %s", i, text(t, got), text(t, c.Value("expected")))
		}
	}
	conflicts := object(data.Value("conflicts"))
	for _, feedback := range list(conflicts.Value("feedback")) {
		for _, representation := range list(conflicts.Value("representations")) {
			if _, err := Represent(object(feedback), representation.(string)); err == nil {
				t.Errorf("%s of %s should fail", representation, text(t, feedback))
			}
		}
	}
}

func TestNormalizeCalls(t *testing.T) {
	for i, item := range list(spec(t, "conformance/normalize_calls.json").Value("cases")) {
		c := object(item)
		got, err := NormalizeCalls(Clone(c.Value("input")))
		if err != nil {
			t.Errorf("case %d: %v", i, err)
			continue
		}
		assertEqual(t, fmt.Sprintf("case %d", i), callsValue(got), c.Value("expected"))
	}
}

func TestProviderRequests(t *testing.T) {
	isolate(t)
	data := spec(t, "conformance/providers.json")
	example := object(data.Value("example"))
	recorded := history(t, data.Value("history"))
	full := Example{Messages: mustMessages(t, example.Value("messages")), Tools: mustTools(t, example.Value("tools"))}
	small := Example{Messages: Prompt("hi")}
	type build struct {
		provider func(...ProviderOption) (*Provider, error)
		example  Example
		history  []HistoryEntry
	}
	builds := map[string]build{
		"openai with history": {func(o ...ProviderOption) (*Provider, error) {
			return NewOpenAI("gpt-5-nano", append(o, WithAPIKey("KEY"), WithReasoningEffort("none"))...)
		}, full, recorded},
		"openai no tools": {func(o ...ProviderOption) (*Provider, error) {
			return NewOpenAI("gpt-5-nano", append(o, WithAPIKey("KEY"))...)
		}, small, nil},
		"openai-compatible local": {func(o ...ProviderOption) (*Provider, error) {
			return NewOpenAICompatible("gemma-4-31b-it", append(o, WithBaseURL("http://localhost:8000/v1"))...)
		}, full, nil},
		"xai": {func(o ...ProviderOption) (*Provider, error) {
			return NewXAI("grok-4", append(o, WithAPIKey("KEY"))...)
		}, full, nil},
		"openrouter": {func(o ...ProviderOption) (*Provider, error) {
			return NewOpenRouter("meta/llama", append(o, WithAPIKey("KEY"), WithTemperature(0.2))...)
		}, full, nil},
		"meta muse": {func(o ...ProviderOption) (*Provider, error) {
			return NewMeta("muse-spark-1.3", append(o, WithAPIKey("KEY"))...)
		}, full, recorded},
		"azure": {func(o ...ProviderOption) (*Provider, error) {
			return NewAzureOpenAI("my deployment", append(o, WithEndpoint("https://res.openai.azure.com/"),
				WithAPIKey("KEY"))...)
		}, full, recorded},
		"anthropic with history": {func(o ...ProviderOption) (*Provider, error) {
			return NewAnthropic("claude-sonnet-5-5", append(o, WithAPIKey("KEY"))...)
		}, full, recorded},
		"gemini": {func(o ...ProviderOption) (*Provider, error) {
			return NewGemini("models/gemini-2.5-flash", append(o, WithAPIKey("KEY"), WithTemperature(0),
				WithThinkingBudget(0))...)
		}, full, recorded},
	}
	requests := list(data.Value("requests"))
	if len(requests) != len(builds) {
		t.Fatalf("expected %d recorded requests, found %d", len(builds), len(requests))
	}
	for _, item := range requests {
		c := object(item)
		name := c.Value("name").(string)
		t.Run(name, func(t *testing.T) {
			b, ok := builds[name]
			if !ok {
				t.Fatalf("no construction for %q", name)
			}
			client, seen := mock(func(sent) reply { return reply{body: obj()} })
			provider, err := b.provider(WithHTTPClient(client))
			if err != nil {
				t.Fatal(err)
			}
			_, _ = provider.Generate(context.Background(), b.example, b.history) // {} is not a valid answer
			if len(*seen) != 1 {
				t.Fatalf("expected one request, saw %d", len(*seen))
			}
			request := (*seen)[0]
			if request.url != c.Value("url") {
				t.Errorf("url %s, want %s", request.url, c.Value("url"))
			}
			assertEqual(t, "headers", headersOf(request.header), c.Value("headers"))
			assertEqual(t, "body", request.body, c.Value("body"))
		})
	}
}

func TestProviderDecoding(t *testing.T) {
	isolate(t)
	data := spec(t, "conformance/providers.json")
	example := object(data.Value("example"))
	ex := Example{Messages: mustMessages(t, example.Value("messages")), Tools: mustTools(t, example.Value("tools"))}
	constructors := map[string]func(string, ...ProviderOption) (*Provider, error){
		"openai": NewOpenAI, "anthropic": NewAnthropic, "gemini": NewGemini}
	cases := list(data.Value("decode"))
	for _, item := range cases {
		c := object(item)
		t.Run(c.Value("name").(string), func(t *testing.T) {
			client, _ := mock(func(sent) reply { return reply{body: c.Value("response")} })
			provider, err := constructors[c.Value("provider").(string)]("m", WithAPIKey("KEY"), WithHTTPClient(client))
			if err != nil {
				t.Fatal(err)
			}
			proposal, err := provider.Generate(context.Background(), ex, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertEqual(t, "proposal", proposalObject(proposal), c.Value("expected"))
		})
	}
	if len(cases) == 0 {
		t.Error("no decode cases found")
	}
	filtered := object(data.Value("filtered_prompt"))
	client, _ := mock(func(sent) reply {
		return reply{status: int(filtered.Value("status").(int64)), body: filtered.Value("response")}
	})
	provider, err := NewOpenAI("m", WithAPIKey("KEY"), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := provider.Generate(context.Background(), ex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if proposal.Calls != nil || proposal.Incomplete != filtered.Value("expected_incomplete") || len(proposal.Usage) != 0 {
		t.Errorf("filtered prompt: %+v", proposal)
	}
}

func TestPaperParity(t *testing.T) {
	golden := spec(t, "conformance/paper_parity.json")
	scenarios := object(golden.Value("scenarios"))
	tools := mustTools(t, golden.Value("tools"))
	names := scenarios.Keys()
	if len(names) == 0 {
		t.Error("no scenarios found")
	}
	for _, name := range names {
		scenario := object(scenarios.Value(name))
		t.Run(name, func(t *testing.T) {
			isolate(t)
			generatorClient, generatorSent := queue(t, list(scenario.Value("generator_responses")))
			decisionClient, decisionSent := queue(t, list(scenario.Value("decision_responses")))
			jev, err := NewJev(WithAPIKey(golden.Value("jev_api_key").(string)), WithHTTPClient(decisionClient))
			if err != nil {
				t.Fatal(err)
			}
			key := golden.Value("generator_api_key").(string)
			var model *Provider
			if scenario.Value("provider") == "anthropic" {
				model, err = NewAnthropic("claude-sonnet-5-5", WithAPIKey(key), WithHTTPClient(generatorClient))
			} else {
				model, err = NewOpenAI("gpt-5-nano", WithAPIKey(key), WithReasoningEffort("none"),
					WithHTTPClient(generatorClient))
			}
			if err != nil {
				t.Fatal(err)
			}
			options := append([]Option{WithDecisionModel(jev), WithTools(tools...)},
				loopOptions(t, object(scenario.Value("options")))...)
			agent, err := New(model, options...)
			if err != nil {
				t.Fatal(err)
			}
			result, err := agent.Call(context.Background(), mustMessages(t, golden.Value("messages")))
			if err != nil {
				t.Fatal(err)
			}
			compareBodies(t, "generator", *generatorSent, list(scenario.Value("generator_requests")))
			compareBodies(t, "decision", *decisionSent, list(scenario.Value("decision_requests")))
			assertEqual(t, "result", summary(result, false), scenario.Value("result"))
		})
	}
}

// compareBodies checks request bodies as JSON values and, like the Python test, as the text
// json.dumps(body, ensure_ascii=False) writes: the history message is sent as text, so its exact
// characters matter, and key order and number kinds must match too.
func compareBodies(t *testing.T, what string, got []sent, want []any) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d requests, want %d", what, len(got), len(want))
	}
	for i := range got {
		assertEqual(t, fmt.Sprintf("%s request %d", what, i), got[i].body, want[i])
		if ours, theirs := text(t, got[i].body), text(t, want[i]); ours != theirs {
			t.Errorf("%s request %d text differs\n got: %s\nwant: %s", what, i, ours, theirs)
		}
	}
}

func TestVerifyParity(t *testing.T) {
	isolate(t)
	golden := spec(t, "conformance/paper_parity.json")
	recorded := object(golden.Value("verify"))
	client, seen := queue(t, list(recorded.Value("decision_responses")))
	jev, err := NewJev(WithAPIKey(golden.Value("jev_api_key").(string)), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	tools := mustTools(t, golden.Value("tools"))
	calls := list(recorded.Value("calls"))
	first, err := Verify(context.Background(), jev, mustMessages(t, recorded.Value("messages")), calls[0], tools)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Verify(context.Background(), jev, mustMessages(t, recorded.Value("second_messages")), calls[1], tools)
	if err != nil {
		t.Fatal(err)
	}
	compareBodies(t, "decision", *seen, list(recorded.Value("decision_requests")))
	var verdicts []any
	for _, v := range []*Verdict{first, second} {
		verdicts = append(verdicts, obj("accepted", v.Accepted, "calls", callsValue(v.Calls),
			"function_probability", optionalFloat(v.FunctionProbability),
			"arguments_probability", optionalFloat(v.ArgumentsProbability),
			"schema_errors", schemaErrorsValue(v.SchemaErrors), "feedback", v.Feedback))
	}
	assertEqual(t, "verdicts", verdicts, recorded.Value("verdicts"))
}

func TestLoopEdges(t *testing.T) {
	data := spec(t, "conformance/loop_edges.json")
	scenarios := object(data.Value("scenarios"))
	tools := mustTools(t, data.Value("tools"))
	names := scenarios.Keys()
	if len(names) == 0 {
		t.Error("no scenarios found")
	}
	for _, name := range names {
		scenario := object(scenarios.Value(name))
		t.Run(name, func(t *testing.T) {
			isolate(t)
			generatorClient, generatorSent := queue(t, list(scenario.Value("generator_responses")))
			decisionClient, decisionSent := queue(t, list(scenario.Value("decision_responses")))
			generator := buildGenerator(t, object(scenario.Value("generator")), generatorClient)
			decisionSpec := object(scenario.Value("decision_model"))
			var decision *SystemOne
			var err error
			if decisionSpec.Value("kind") == "pplx" {
				decision, err = NewPPLXDecider(decisionSpec.Value("base_url").(string), WithHTTPClient(decisionClient))
			} else {
				decision, err = NewJev(WithAPIKey("JEV-KEY"), WithHTTPClient(decisionClient))
			}
			if err != nil {
				t.Fatal(err)
			}
			options := append([]Option{WithDecisionModel(decision), WithTools(tools...)},
				loopOptions(t, object(scenario.Value("options")))...)
			agent, err := New(generator, options...)
			if err != nil {
				t.Fatal(err)
			}
			result, err := agent.Call(context.Background(), mustMessages(t, data.Value("messages")))
			if err != nil {
				t.Fatal(err)
			}
			compareRequests(t, "generator", *generatorSent, list(scenario.Value("generator_requests")))
			compareRequests(t, "decision", *decisionSent, list(scenario.Value("decision_requests")))
			assertEqual(t, "result", summary(result, true), scenario.Value("result"))
		})
	}
}

// compareRequests checks url, the listed headers and the body of each request; the history message
// inside a body is a string, so Equal compares it character for character.
func compareRequests(t *testing.T, what string, got []sent, want []any) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d requests, want %d", what, len(got), len(want))
	}
	for i := range got {
		expected := object(want[i])
		if got[i].url != expected.Value("url") {
			t.Errorf("%s request %d url %s, want %s", what, i, got[i].url, expected.Value("url"))
		}
		assertEqual(t, fmt.Sprintf("%s request %d headers", what, i), headersOf(got[i].header), expected.Value("headers"))
		assertEqual(t, fmt.Sprintf("%s request %d body", what, i), got[i].body, expected.Value("body"))
	}
}

func buildGenerator(t *testing.T, spec *Object, client *http.Client) *Provider {
	t.Helper()
	options := []ProviderOption{WithHTTPClient(client)}
	settings := object(spec.Value("options"))
	for _, key := range settings.Keys() {
		value := settings.Value(key)
		switch key {
		case "api_key":
			options = append(options, WithAPIKey(value.(string)))
		case "reasoning_effort":
			options = append(options, WithReasoningEffort(value.(string)))
		case "base_url":
			options = append(options, WithBaseURL(value.(string)))
		case "endpoint":
			options = append(options, WithEndpoint(value.(string)))
		case "temperature":
			number, _ := number(value)
			options = append(options, WithTemperature(number))
		default:
			t.Fatalf("unknown generator option %q", key)
		}
	}
	constructors := map[string]func(string, ...ProviderOption) (*Provider, error){
		"openai": NewOpenAI, "openai-compatible": NewOpenAICompatible, "azure": NewAzureOpenAI, "xai": NewXAI,
		"anthropic": NewAnthropic, "gemini": NewGemini}
	provider, err := constructors[spec.Value("kind").(string)](spec.Value("model").(string), options...)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}
