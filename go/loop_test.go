package dmcall

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
)

// The DM-Call loop: acceptance, cumulative repair, the paper's variants, fallback and execution
// (ported from tests/test_loop.py).

var weatherTool = Tool{Name: "get_weather", Description: "Current weather for a city.",
	Parameters: MustParseJSON(`{"type": "object", "properties": {"city": {"type": "string"},
		"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}`)}

func city(name string) []Call {
	return []Call{{Name: "get_weather", Arguments: obj("city", name)}}
}

func seoul() []Call { return city("Seoul") }
func busan() []Call { return city("Busan") }

// scriptedModel returns scripted candidates (nil: unreadable) and records the history it saw.
type scriptedModel struct {
	mu        sync.Mutex
	proposals [][]Call
	histories [][]any
}

func script(proposals ...[]Call) *scriptedModel { return &scriptedModel{proposals: proposals} }

func (m *scriptedModel) Generate(_ context.Context, _ Example, history []HistoryEntry) (*Proposal, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	seen := []any{}
	for _, item := range history {
		seen = append(seen, obj("round", int64(item.Round), "calls", callsValue(item.Calls), "feedback", item.Feedback.Clone()))
	}
	m.histories = append(m.histories, seen)
	if len(m.proposals) == 0 {
		return nil, errors.New("no scripted proposal left")
	}
	calls := m.proposals[0]
	m.proposals = m.proposals[1:]
	proposal := &Proposal{Calls: cloneCalls(calls), ParseErrors: []string{}, Usage: Usage{"input_tokens": 10, "output_tokens": 2}}
	if calls == nil {
		proposal.Incomplete, proposal.ParseErrors = IncompleteLength, []string{"cut off"}
	}
	if len(calls) == 0 {
		proposal.Content = "No tool fits."
	}
	return proposal, nil
}

func (m *scriptedModel) lengths() []int {
	out := make([]int, len(m.histories))
	for i, h := range m.histories {
		out[i] = len(h)
	}
	return out
}

// scriptedJev is Jev over a mocked transport: arguments are right only for goodCity; a no call scores noCall.
type scriptedJev struct {
	jev      *SystemOne
	requests *[]sent
}

func newScriptedJev(t *testing.T, goodCity string, function, noCall float64, opts ...DecisionOption) *scriptedJev {
	t.Helper()
	client, seen := mock(func(request sent) reply {
		calls := list(object(object(request.body).Value("state")).Value("proposed_calls"))
		if len(calls) == 0 {
			return reply{body: answerBody(noCall, nil)}
		}
		arguments := 0.05
		if object(object(calls[0]).Value("arguments")).Value("city") == goodCity {
			arguments = 0.95
		}
		return reply{body: answerBody(function, arguments)}
	})
	jev, err := NewJev(append([]DecisionOption{WithAPIKey("jev-key"), WithHTTPClient(client)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return &scriptedJev{jev: jev, requests: seen}
}

func defaultJev(t *testing.T) *scriptedJev { return newScriptedJev(t, "Seoul", 0.95, 0.05) }

// answerBody is a /v1/systemone response body; a nil arguments probability leaves that answer out.
func answerBody(function any, arguments any) *Object {
	answers := obj("function_correct", obj("type", "noul", "noul", function))
	if arguments != nil {
		answers.Set("arguments_correct", obj("type", "noul", "noul", arguments))
	}
	return obj("model", "jev-1.13.0", "answers", answers, "usage", obj("input_tokens", 300, "output_tokens", 0))
}

func agent(t *testing.T, decision *scriptedJev, proposals [][]Call, opts ...Option) (*Client, *scriptedModel) {
	t.Helper()
	model := script(proposals...)
	if decision == nil {
		decision = defaultJev(t)
	}
	client, err := New(model, append([]Option{WithDecisionModel(decision.jev), WithTools(weatherTool)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return client, model
}

var ctx = context.Background()

func TestRepairsWithCumulativeFeedbackThenRuns(t *testing.T) {
	var ran []string
	tool := weatherTool
	tool.Handler = TypedHandler(func(_ context.Context, in struct {
		City string `json:"city"`
	}) (map[string]any, error) {
		ran = append(ran, in.City)
		return map[string]any{"city": in.City, "temp": 21}, nil
	})
	model, decision := script(busan(), []Call{}, seoul()), defaultJev(t)
	client, err := New(model, WithDecisionModel(decision.jev), WithTools(tool))
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Run(ctx, Prompt("Weather in Seoul?"))
	if err != nil {
		t.Fatal(err)
	}
	if !Equal(result.Calls, seoul()) || !*result.Accepted || result.Repairs != 2 || result.Exhausted {
		t.Fatalf("result %v", result)
	}
	if len(ran) != 1 || ran[0] != "Seoul" || !result.Executed || !Equal(result.Output(), map[string]any{"city": "Seoul", "temp": 21}) {
		t.Errorf("execution: ran %v, %v", ran, result.Output())
	}
	assertEqual(t, "history lengths", model.lengths(), []int{0, 1, 2})
	last := model.histories[2]
	assertEqual(t, "first rejected calls", object(last[0]).Value("calls"), callsValue(busan()))
	assertEqual(t, "second rejected calls", object(last[1]).Value("calls"), []any{})
	assertEqual(t, "failed checks", object(object(last[0]).Value("feedback")).Value("failed_checks"), []any{"arguments"})
	// Hybrid feedback: the generator reads the score and its label; the trace keeps the score.
	shown := object(object(object(object(model.histories[1][0]).Value("feedback")).Value("checks")).Value("arguments"))
	if shown.Value("probability_yes") != 0.05 || shown.Value("feedback_label") != "very likely incorrect" {
		t.Errorf("shown feedback %s", shown)
	}
	if object(object(result.Trace[0].Feedback.Value("checks")).Value("arguments")).Has("feedback_label") {
		t.Error("the trace must keep the numeric feedback")
	}
	attempts := result.Attempts()
	if *attempts[0].Accepted || *attempts[1].Accepted || !*attempts[2].Accepted || attempts[1].ArgumentsProbability != nil {
		t.Errorf("attempts %+v", attempts)
	}
	usage := result.Usage()
	assertEqual(t, "usage", obj("generator", usage.Generator.jsonValue(), "decision_model", usage.DecisionModel.jsonValue()),
		MustParseJSON(`{"generator": {"input_tokens": 30, "output_tokens": 6},
			"decision_model": {"input_tokens": 900, "output_tokens": 0}}`))
	probabilities := result.Probabilities()
	if *probabilities.Function != 0.95 || *probabilities.Arguments != 0.95 {
		t.Errorf("probabilities %v %v", *probabilities.Function, *probabilities.Arguments)
	}
	if !strings.HasPrefix(result.String(), `Result(call=get_weather(city="Seoul"), accepted=True, repairs=2`) {
		t.Errorf("String %s", result)
	}
}

func TestRejectedCallsDoNotRunUnlessAsked(t *testing.T) {
	var ran []string
	tool := weatherTool
	tool.Handler = func(_ context.Context, args *Object) (any, error) {
		ran = append(ran, args.Value("city").(string))
		return nil, nil
	}
	model := script(busan(), busan(), busan(), busan())
	client, _ := New(model, WithDecisionModel(defaultJev(t).jev), WithTools(tool))
	result, err := client.Run(ctx, Prompt("Weather in Seoul?"))
	if err != nil {
		t.Fatal(err)
	}
	if *result.Accepted || !result.Exhausted || result.Repairs != 3 || result.Executed || len(ran) != 0 {
		t.Errorf("result %v, ran %v", result, ran)
	}
	again, _ := New(script(busan(), busan()), WithDecisionModel(defaultJev(t).jev), WithTools(tool), WithMaxRepairs(1))
	result, err = again.Run(ctx, Prompt("Weather in Seoul?"), WithExecuteUnverified(true))
	if err != nil {
		t.Fatal(err)
	}
	if result.Repairs != 1 || !result.Executed || len(ran) != 1 || ran[0] != "Busan" {
		t.Errorf("result %v, ran %v", result, ran)
	}
}

func TestWithoutAnAcceptedCandidateTheFirstOneIsFinal(t *testing.T) {
	var ran []string
	tool := weatherTool
	tool.Handler = func(_ context.Context, args *Object) (any, error) {
		ran = append(ran, args.Value("city").(string))
		return args.Value("city"), nil
	}
	first, _ := New(script([]Call{}, busan()), WithDecisionModel(defaultJev(t).jev), WithTools(tool),
		WithMaxRepairs(1), WithPreserveInitialNoCall(false))
	result, err := first.Run(ctx, Prompt("Weather in Seoul?"), WithExecuteUnverified(true))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Exhausted || *result.Accepted || result.Repairs != 1 || !result.NoCall() || result.FinalAttempt != 0 ||
		result.Executed || len(ran) != 0 || result.Text != "No tool fits." {
		t.Errorf("first: %v", result)
	}
	if p := result.Probabilities(); *p.Function != 0.05 || p.Arguments != nil {
		t.Errorf("probabilities %+v", p)
	}
	last, _ := New(script([]Call{}, busan()), WithDecisionModel(defaultJev(t).jev), WithTools(tool),
		WithMaxRepairs(1), WithOnExhaustion(ExhaustionLast), WithPreserveInitialNoCall(false))
	result, err = last.Run(ctx, Prompt("Weather in Seoul?"), WithExecuteUnverified(true))
	if err != nil {
		t.Fatal(err)
	}
	if !Equal(result.Calls, busan()) || result.FinalAttempt != 1 || !result.Executed || len(ran) != 1 {
		t.Errorf("last: %v, ran %v", result, ran)
	}
	perCall, _ := agent(t, nil, [][]Call{busan(), {}, busan(), seoul()}, WithMaxRepairs(1))
	result, err = perCall.Call(ctx, Prompt("Weather in Seoul?"), WithOnExhaustion(ExhaustionLast))
	if err != nil || !result.NoCall() {
		t.Errorf("per-call last: %v, %v", result, err)
	}
	passed, err := perCall.Call(ctx, Prompt("Weather in Seoul?"))
	if err != nil || !Equal(passed.Calls, seoul()) || !*passed.Accepted || passed.FinalAttempt != 1 {
		t.Errorf("passed: %v, %v", passed, err)
	}
	if _, err := perCall.Call(ctx, Prompt("x"), WithOnExhaustion("best")); err == nil ||
		!strings.Contains(err.Error(), "on_exhaustion") {
		t.Errorf("expected an on_exhaustion error, got %v", err)
	}
}

func TestUnreadableCandidateIsRepairedAndNeverRun(t *testing.T) {
	handlers := map[string]Handler{"get_weather": func(_ context.Context, args *Object) (any, error) {
		return args.Value("city"), nil
	}}
	decision := defaultJev(t)
	loop, _ := agent(t, decision, [][]Call{nil, seoul()}, WithHandlers(handlers))
	result, err := loop.Run(ctx, Prompt("Weather in Seoul?"))
	if err != nil {
		t.Fatal(err)
	}
	attempts := result.Attempts()
	if attempts[0].Calls != nil || len(attempts[0].SchemaErrors) == 0 || !*result.Accepted || result.Output() != "Seoul" ||
		len(*decision.requests) != 1 {
		t.Errorf("result %v", result)
	}
	stuck, _ := agent(t, nil, [][]Call{nil}, WithMaxRepairs(0), WithHandlers(handlers))
	result, err = stuck.Run(ctx, Prompt("Weather?"), WithExecuteUnverified(true))
	if err != nil {
		t.Fatal(err)
	}
	if result.Calls != nil || result.Executed || !strings.HasPrefix(result.String(), "Result(unreadable candidate") {
		t.Errorf("stuck %v", result)
	}
}

func TestInitialNoCallIsReturnedWithoutVerification(t *testing.T) {
	decision := newScriptedJev(t, "Seoul", 0.95, 0.97)
	loop, model := agent(t, decision, [][]Call{{}})
	result, err := loop.Run(ctx, Prompt("Tell me a joke."))
	if err != nil {
		t.Fatal(err)
	}
	if !result.NoCall() || result.Accepted != nil || !result.InitialNoCallPreserved || result.Executed ||
		result.Text != "No tool fits." || len(*decision.requests) != 0 || result.Attempts()[0].Accepted != nil ||
		len(model.histories) != 1 {
		t.Errorf("result %v", result)
	}
	decision = newScriptedJev(t, "Seoul", 0.95, 0.97)
	checked, _ := agent(t, decision, [][]Call{{}}, WithPreserveInitialNoCall(false))
	result, err = checked.Call(ctx, Prompt("Tell me a joke."))
	if err != nil {
		t.Fatal(err)
	}
	if !*result.Accepted || result.InitialNoCallPreserved || len(*decision.requests) != 1 {
		t.Errorf("checked %v", result)
	}
}

func TestLatestHistoryKeepsTheRealRound(t *testing.T) {
	loop, model := agent(t, nil, [][]Call{busan(), {}, seoul()}, WithHistory(HistoryLatest))
	if _, err := loop.Call(ctx, Prompt("Weather in Seoul?")); err != nil {
		t.Fatal(err)
	}
	assertEqual(t, "history lengths", model.lengths(), []int{0, 1, 1})
	latest := object(model.histories[2][0])
	if latest.Value("round") != int64(2) || !Equal(latest.Value("calls"), []any{}) {
		t.Errorf("latest %s", latest)
	}
}

func TestSingleQuestionVariants(t *testing.T) {
	functionOnly, _ := agent(t, nil, [][]Call{busan()}, WithChecks(ChecksFunction))
	result, err := functionOnly.Call(ctx, Prompt("Weather in Seoul?")) // the wrong city passes
	if err != nil || !*result.Accepted || result.Repairs != 0 {
		t.Fatalf("function only: %v, %v", result, err)
	}
	assertEqual(t, "checks", object(result.Trace[0].Feedback.Value("checks")).Keys(), []string{"function_selection"})
	argumentsOnly, _ := agent(t, nil, [][]Call{{}, seoul()}, WithChecks(ChecksArguments), WithPreserveInitialNoCall(false))
	result, err = argumentsOnly.Call(ctx, Prompt("Weather in Seoul?")) // a no call has no arguments to reject
	if err != nil || !*result.Accepted || !result.NoCall() || result.Repairs != 0 {
		t.Errorf("arguments only: %v, %v", result, err)
	}
}

func TestFeedbackRepresentationsReachOnlyTheGenerator(t *testing.T) {
	for _, c := range []struct{ representation, field string }{
		{FeedbackNumeric, "threshold"}, {FeedbackText, "feedback_label"}, {FeedbackHybrid, "feedback_label"}} {
		loop, model := agent(t, nil, [][]Call{busan(), seoul()}, WithFeedback(c.representation))
		result, err := loop.Call(ctx, Prompt("Weather in Seoul?"))
		if err != nil {
			t.Fatal(err)
		}
		shown := object(object(object(object(model.histories[1][0]).Value("feedback")).Value("checks")).Value("arguments"))
		traced := object(object(result.Trace[0].Feedback.Value("checks")).Value("arguments"))
		if !shown.Has(c.field) || traced.Value("probability_yes") != 0.05 ||
			shown.Has("probability_yes") != (c.representation != FeedbackText) {
			t.Errorf("%s: shown %s, traced %s", c.representation, shown, traced)
		}
	}
}

func TestOptionsAndDefaults(t *testing.T) {
	isolate(t)
	loop, _ := agent(t, nil, nil)
	if loop.MaxRepairs() != 3 || loop.Checks() != "both" || loop.History() != "cumulative" || loop.Feedback() != "hybrid" ||
		loop.OnExhaustion() != "first" || !loop.PreserveInitialNoCall() {
		t.Errorf("defaults %v", loop)
	}
	jev := defaultJev(t).jev
	for _, bad := range []Option{WithChecks("neither"), WithHistory("none"), WithFeedback("words"),
		WithOnExhaustion("best"), WithMaxRepairs(-1)} {
		if _, err := New(script(), WithDecisionModel(jev), bad); err == nil {
			t.Errorf("expected an error for %T", bad)
		}
	}
	if _, err := New(script(), WithDecisionModel(jev), WithThreshold(0.9)); err == nil ||
		!strings.Contains(err.Error(), "decision model itself") {
		t.Errorf("threshold with a decision model: %v", err)
	}
	if _, err := New(script(), WithDecisionModel(jev), WithJevAPIKey("k")); err == nil {
		t.Error("a Jev key with a decision model must be refused")
	}
	if _, err := New(nil); err == nil {
		t.Error("a nil generator must be refused")
	}
	if _, err := New(script()); err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
		t.Errorf("missing Jev key: %v", err)
	}
	t.Setenv("TYPESAFE_API_KEY", "jev-env")
	t.Setenv("OPENAI_API_KEY", "openai-env")
	generator, err := ProviderFromSpec("openai:gpt-5-nano", WithReasoningEffort("none"))
	if err != nil {
		t.Fatal(err)
	}
	built, err := New(generator, WithThreshold(0.9))
	if err != nil {
		t.Fatal(err)
	}
	decision := built.DecisionModel().(*SystemOne)
	if generator.reasoningEffort != "none" || decision.FunctionThreshold() != 0.9 || decision.ArgumentsThreshold() != 0.9 {
		t.Errorf("built %v", built)
	}
	if strings.Contains(built.String(), "jev-env") || strings.Contains(built.String(), "openai-env") {
		t.Error("String must not show keys")
	}
	paper, err := New(generator) // the paper's settings by default
	if err != nil {
		t.Fatal(err)
	}
	decision = paper.DecisionModel().(*SystemOne)
	if decision.Name() != "Jev" || decision.Model() != "jev-1.13.0" || decision.FunctionThreshold() != 0.5 ||
		decision.ArgumentsThreshold() != 0.5 {
		t.Errorf("paper defaults %v", decision)
	}
	if _, err := New(generator, WithThreshold(1.5)); err == nil {
		t.Error("an out-of-range threshold must be refused")
	}
}

func TestPerCallOverridesAndMissingHandlers(t *testing.T) {
	loop, _ := agent(t, nil, [][]Call{busan(), seoul()})
	result, err := loop.Call(ctx, Prompt("Weather in Seoul?"), WithMaxRepairs(0))
	if err != nil || result.Repairs != 0 || *result.Accepted || !result.Exhausted {
		t.Fatalf("max repairs 0: %v, %v", result, err)
	}
	if _, err := loop.Run(ctx, Prompt("Weather in Seoul?"), WithTools(weatherTool)); err == nil ||
		!strings.Contains(err.Error(), "no handler") {
		t.Errorf("missing handler: %v", err)
	}
	nop := func(context.Context, *Object) (any, error) { return nil, nil }
	if _, err := loop.Call(ctx, Prompt("x"), WithHandlers(map[string]Handler{"nope": nop})); err == nil ||
		!strings.Contains(err.Error(), "not one of the tools") {
		t.Errorf("unknown handler: %v", err)
	}
	if _, err := loop.Call(ctx, Prompt("x"), WithMaxRepairs(-1)); err == nil {
		t.Error("a negative per-call repair limit must be refused")
	}
}

func TestHandlerErrorsReturnTheResult(t *testing.T) {
	tool := weatherTool
	tool.Handler = func(context.Context, *Object) (any, error) { return nil, errors.New("boom") }
	client, _ := New(script(seoul()), WithDecisionModel(defaultJev(t).jev), WithTools(tool))
	result, err := client.Run(ctx, Prompt("Weather in Seoul?"))
	if err == nil || !strings.Contains(err.Error(), "boom") || result == nil || !*result.Accepted || result.Executed {
		t.Errorf("result %v, err %v", result, err)
	}
}

func TestMessagesAreValidated(t *testing.T) {
	loop, model := agent(t, nil, [][]Call{seoul(), seoul()})
	parts, err := ParseMessages(MustParseJSON(`[{"role": "user", "content": [{"type": "text", "text": "Weather"},
		{"type": "text", "text": "in Seoul?"}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	if parts[0].Content != "Weather\nin Seoul?" {
		t.Errorf("joined content %q", parts[0].Content)
	}
	if _, err := loop.Call(ctx, parts); err != nil {
		t.Fatal(err)
	}
	_ = model
	for _, bad := range [][]Message{nil, {{Role: "tool", Content: "{}"}}, {{Role: "robot", Content: "x"}}} {
		if _, err := loop.Call(ctx, bad); err == nil {
			t.Errorf("expected an error for %v", bad)
		}
	}
	if _, err := ParseMessages(MustParseJSON(`[{"role": "user", "content": {"x": 1}}]`)); err == nil {
		t.Error("object content must be refused")
	}
	if _, err := ParseMessages(MustParseJSON(`3`)); err == nil {
		t.Error("a number is not a conversation")
	}
}

func TestVerifyChecksCallsFromAnySDK(t *testing.T) {
	decision := defaultJev(t)
	tools := []Tool{weatherTool}
	shapes := []any{
		MustParseJSON(`{"name": "get_weather", "arguments": {"city": "Seoul"}}`),
		MustParseJSON(`{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}`),
		MustParseJSON(`{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Seoul"}}`),
		MustParseJSON(`{"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}`),
		[]ToolCall{{ID: "c1", Function: FunctionCall{Name: "get_weather", Arguments: `{"city": "Seoul"}`}}},
		seoul(),
	}
	for i, shape := range shapes {
		verdict, err := Verify(ctx, decision.jev, Prompt("Weather in Seoul?"), shape, tools)
		if err != nil {
			t.Fatal(err)
		}
		if !verdict.Accepted || !Equal(verdict.Calls, seoul()) || *verdict.ArgumentsProbability != 0.95 {
			t.Errorf("shape %d: %+v", i, verdict)
		}
	}
	if verdict, _ := Verify(ctx, decision.jev, Prompt("Weather in Seoul?"), busan(), tools); verdict.Accepted {
		t.Error("Busan must be rejected")
	}
	count := len(*decision.requests)
	malformed, err := Verify(ctx, decision.jev, Prompt("Weather?"),
		MustParseJSON(`{"name": "get_weather", "arguments": "{oops"}`), tools)
	if err != nil || malformed.Accepted || len(*decision.requests) != count {
		t.Errorf("malformed: %+v, %v", malformed, err) // rejected locally; nothing is sent
	}
	if verdict, _ := Verify(ctx, decision.jev, Prompt("Tell me a joke."), nil, tools); !Equal(verdict.Calls, []any{}) {
		t.Errorf("nil calls are no call: %+v", verdict)
	}
	if verdict, _ := Verify(ctx, newScriptedJev(t, "Seoul", 0.95, 0.9).jev, Prompt("Tell me a joke."), []any{}, tools); !verdict.Accepted {
		t.Error("a no call the decision model approves passes")
	}
	turns := []Message{{Role: "user", Content: "Weather in Busan, then Seoul?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c0", Type: "function",
			Function: FunctionCall{Name: "get_weather", Arguments: `{"city": "Busan"}`}}}},
		{Role: "tool", ToolCallID: "c0", Content: `{"temp": 18}`}}
	if verdict, err := Verify(ctx, decision.jev, turns, seoul(), tools); err != nil || !verdict.Accepted {
		t.Fatalf("conversation with tool turns: %+v, %v", verdict, err)
	}
	last := (*decision.requests)[len(*decision.requests)-1]
	sentMessages := list(object(object(last.body).Value("state")).Value("messages"))
	assertEqual(t, "sent messages", sentMessages, MustParseJSON(`[{"role": "user", "content": "Weather in Busan, then Seoul?"},
		{"role": "assistant", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_weather",
		"arguments": "{\"city\": \"Busan\"}"}}]}, {"role": "tool", "content": "{\"temp\": 18}", "tool_call_id": "c0"}]`))
	if _, err := Verify(ctx, decision.jev, []Message{{Role: "robot", Content: "x"}}, seoul(), tools); err == nil {
		t.Error("an unknown role must be refused")
	}
	loop, _ := agent(t, decision, nil)
	if verdict, err := loop.Verify(ctx, Prompt("Weather in Seoul?"), seoul()); err != nil || !verdict.Accepted {
		t.Errorf("client verify: %+v, %v", verdict, err)
	}
	if _, err := loop.Verify(ctx, Prompt("x"), seoul(), WithMaxRepairs(1)); err == nil {
		t.Error("Verify takes only WithTools")
	}
	isolate(t)
	if _, err := Verify(ctx, nil, Prompt("x"), seoul(), tools); err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
		t.Errorf("the default Jev needs a key: %v", err)
	}
}

func TestContextCancellationStopsTheLoop(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	client, _ := mock(func(sent) reply { return reply{body: obj()} })
	generator, err := NewOpenAI("m", WithAPIKey("k"), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	loop, err := New(generator, WithDecisionModel(defaultJev(t).jev), WithTools(weatherTool))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := loop.Call(cancelled, Prompt("Weather?")); !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}
