package dmcall

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// Jev and the PPLX Decider: the wire format, verdicts, keys, thresholds and failures (ported from
// tests/test_decision_models.py).

const secret = "sk-test-SECRET-123"

var exampleMessages = []Message{{Role: "system", Content: "Be brief."}, {Role: "user", Content: "Weather in Seoul?"}}

func jevWith(t *testing.T, handler func(sent) reply, opts ...DecisionOption) (*SystemOne, *[]sent) {
	t.Helper()
	client, seen := mock(handler)
	jev, err := NewJev(append([]DecisionOption{WithAPIKey("jev-key"), WithHTTPClient(client), WithSleep(noSleep)},
		opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return jev, seen
}

func answering(function, arguments any) func(sent) reply {
	return func(sent) reply { return reply{body: answerBody(function, arguments)} }
}

func TestWireFormatSendsOnlyObservableInputs(t *testing.T) {
	jev, seen := jevWith(t, answering(0.9, 0.8))
	tools := mustTools(t, MustParseJSON(`[{"type": "function", "function": {"name": "get_weather",
		"description": "Current weather for a city.", "parameters": {"type": "object", "properties": {
		"city": {"type": "string"}}, "required": ["city"]}}, "metadata": {"reference": "get_weather(city='Seoul')"}}]`))
	messages := mustMessages(t, MustParseJSON(`[{"role": "user", "content": "hi", "ground_truth": "nope"}]`))
	record, err := jev.Judge(ctx, messages, tools, seoul())
	if err != nil {
		t.Fatal(err)
	}
	body := object((*seen)[0].body)
	if (*seen)[0].url != "https://api.typesafe.ai/v1/systemone" || body.Value("model") != "jev-1.13.0" {
		t.Errorf("request %s %s", (*seen)[0].url, body)
	}
	state := object(body.Value("state"))
	assertEqual(t, "messages", state.Value("messages"), MustParseJSON(`[{"role": "user", "content": "hi"}]`))
	assertEqual(t, "tools", state.Value("tools"), MustParseJSON(`[{"name": "get_weather",
		"description": "Current weather for a city.", "parameters": {"type": "object", "properties": {
		"city": {"type": "string"}}, "required": ["city"]}}]`))
	assertEqual(t, "proposed calls", state.Value("proposed_calls"), callsValue(seoul()))
	assertEqual(t, "questions", body.Value("questions"),
		obj("function_correct", FunctionQuestion.Object(), "arguments_correct", ArgumentQuestion.Object()))
	if !record.Accepted || *record.FunctionProbability != 0.9 || *record.ArgumentsProbability != 0.8 ||
		len(list(record.Feedback.Value("failed_checks"))) != 0 || record.APICalls != 1 || record.Model != "jev-1.13.0" {
		t.Errorf("record %+v", record)
	}
	if (*seen)[0].header.Get("Authorization") != "Bearer jev-key" {
		t.Errorf("auth header %q", (*seen)[0].header.Get("Authorization"))
	}
}

func TestNoCallAsksOnlyTheFunctionQuestion(t *testing.T) {
	jev, seen := jevWith(t, answering(0.97, nil))
	record, err := jev.Judge(ctx, exampleMessages, []Tool{weatherTool}, []Call{})
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, "questions", object(object((*seen)[0].body).Value("questions")).Keys(), []string{"function_correct"})
	arguments := object(object(record.Feedback.Value("checks")).Value("arguments"))
	if !record.Accepted || !record.ArgumentsSkipped || record.ArgumentsProbability != nil || record.ArgumentsPass != nil ||
		arguments.Value("pass") != nil {
		t.Errorf("record %+v", record)
	}
}

func TestBothQuestionsMustPassAtTheThreshold(t *testing.T) {
	for _, c := range []struct {
		function, arguments float64
		failed              []any
	}{{0.4, 0.9, []any{"function_selection"}}, {0.9, 0.49, []any{"arguments"}}, {0.5, 0.5, []any{}}} {
		jev, _ := jevWith(t, answering(c.function, c.arguments))
		record, err := jev.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
		if err != nil {
			t.Fatal(err)
		}
		assertEqual(t, "failed checks", record.Feedback.Value("failed_checks"), c.failed)
		if record.Accepted != (len(c.failed) == 0) {
			t.Errorf("accepted %v for %v", record.Accepted, c)
		}
	}
}

func TestSchemaFailureIsJudgedButCannotBeOverruled(t *testing.T) {
	jev, seen := jevWith(t, answering(0.99, 0.99))
	record, err := jev.Judge(ctx, exampleMessages, []Tool{weatherTool}, []Call{{Name: "get_weather", Arguments: NewObject()}})
	if err != nil {
		t.Fatal(err)
	}
	if len(*seen) != 1 || *record.ArgumentsProbability != 0.99 || *record.ArgumentsPass || record.Accepted ||
		len(record.SchemaErrors) == 0 {
		t.Errorf("record %+v", record)
	}
}

func TestUnreadableCandidateIsNotSent(t *testing.T) {
	for _, calls := range [][]Call{nil, {{Name: "get_weather"}}} {
		jev, seen := jevWith(t, answering(0.9, 0.9))
		record, err := jev.Judge(ctx, exampleMessages, []Tool{weatherTool}, calls)
		if err != nil {
			t.Fatal(err)
		}
		if len(*seen) != 0 || record.APICalls != 0 || record.FunctionProbability != nil || record.Accepted {
			t.Errorf("calls %v: record %+v", calls, record)
		}
		assertEqual(t, "usage", record.Usage.jsonValue(), obj("input_tokens", 0, "output_tokens", 0))
	}
}

func TestThresholdsKeysAndModel(t *testing.T) {
	isolate(t)
	if _, err := NewJev(); err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
		t.Errorf("missing key: %v", err)
	}
	t.Setenv("TYPESAFE_API_KEY", secret)
	t.Setenv("TYPESAFE_MODEL", "jev-2")
	client, seen := mock(answering(0.75, 0.85))
	model, err := NewJev(WithThreshold(0.7), WithArgumentsThreshold(0.9), WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	record, err := model.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if err != nil {
		t.Fatal(err)
	}
	if (*seen)[0].header.Get("Authorization") != "Bearer "+secret || strings.Contains(model.String(), secret) {
		t.Errorf("auth %q, String %s", (*seen)[0].header.Get("Authorization"), model)
	}
	if object((*seen)[0].body).Value("model") != "jev-2" || !record.FunctionPass || *record.ArgumentsPass {
		t.Errorf("record %+v", record)
	}
	for _, bad := range []float64{1.5, -0.1} {
		if _, err := NewJev(WithAPIKey("k"), WithThreshold(bad)); err == nil || !strings.Contains(err.Error(), "thresholds") {
			t.Errorf("threshold %v: %v", bad, err)
		}
	}
	if _, err := NewJev(WithAPIKey("a\nb")); err == nil {
		t.Error("a multi-line key must be refused")
	}
	if _, err := NewJev(WithAPIKey("k"), WithBaseURL("https://user:pw@example.com")); err == nil {
		t.Error("credentials in the base URL must be refused")
	}
}

func TestRetriesAreBoundedAndErrorsSanitized(t *testing.T) {
	replies := []reply{{status: 429, header: map[string]string{"Retry-After": "0"}}, {body: answerBody(0.95, 0.95)}}
	jev, _ := jevWith(t, func(sent) reply {
		next := replies[0]
		replies = replies[1:]
		return next
	})
	record, err := jev.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if err != nil || record.APICalls != 2 {
		t.Fatalf("record %+v, %v", record, err)
	}
	failing, seen := jevWith(t, func(sent) reply {
		return reply{status: 401, body: secret, header: map[string]string{"x-request-id": "req-7"}}
	})
	_, err = failing.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	var failure *DecisionModelError
	if !errors.As(err, &failure) || len(*seen) != 1 || failure.StatusCode != 401 || failure.RequestID != "req-7" ||
		strings.Contains(failure.Error(), secret) || failure.Code != "http_error" {
		t.Errorf("error %#v", err)
	}
	overloaded, seen := jevWith(t, func(sent) reply { return reply{status: 503} }, WithMaxRetries(2))
	_, err = overloaded.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if !errors.As(err, &failure) || failure.Attempts != 3 || len(*seen) != 3 {
		t.Errorf("error %#v", err)
	}
	broken, _ := jevWith(t, nil, WithMaxRetries(1))
	transport := failingTransport(errors.New("connection refused"))
	broken.client = transport
	_, err = broken.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if !errors.As(err, &failure) || failure.Code != "connection_failed" || failure.Attempts != 2 {
		t.Errorf("error %#v", err)
	}
}

func TestInvalidResponsesAreErrorsNotRejections(t *testing.T) {
	invalid := []any{
		obj("answers", obj()),
		answerBody(1.5, 0.5),
		answerBody(true, 0.5),
		func() *Object { b := answerBody(0.5, 0.5); b.Set("usage", nil); return b }(),
		func() *Object { b := answerBody(0.5, 0.5); b.Set("model", ""); return b }(),
		func() *Object { b := answerBody(0.5, 0.5); b.Set("usage", obj("input_tokens", -1)); return b }(),
		[]any{"not", "an", "object"},
		"not json",
	}
	for i, body := range invalid {
		jev, _ := jevWith(t, func(sent) reply { return reply{body: body} })
		_, err := jev.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
		var failure *DecisionModelError
		if !errors.As(err, &failure) || failure.Code != "response_invalid" {
			t.Errorf("body %d: %v", i, err)
		}
	}
}

func TestCustomDecisionModel(t *testing.T) {
	var states []*Object
	asker := AskerFunc(func(_ context.Context, state, questions *Object) (*Answer, error) {
		states = append(states, state)
		probabilities := map[string]float64{"function_correct": 0.8}
		if questions.Has("arguments_correct") {
			probabilities["arguments_correct"] = 0.3
		}
		return &Answer{Probabilities: probabilities, Model: "mine", Usage: Usage{"input_tokens": 5}, APICalls: 1}, nil
	})
	decision, err := NewDecisionModel("Mine", "", asker, WithArgumentsThreshold(0.25))
	if err != nil {
		t.Fatal(err)
	}
	record, err := decision.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if err != nil {
		t.Fatal(err)
	}
	if !record.Accepted || record.Model != "mine" || record.Feedback.Value("source") != genericFeedbackSource ||
		len(states) != 1 || decision.FunctionThreshold() != 0.5 {
		t.Errorf("record %+v", record)
	}
	if _, err := NewDecisionModel("Mine", "", asker, WithAPIKey("k")); err == nil {
		t.Error("an Asker decision model takes no endpoint options")
	}
	loop, err := New(script(seoul()), WithDecisionModel(decision), WithTools(weatherTool))
	if err != nil {
		t.Fatal(err)
	}
	if result, err := loop.Call(ctx, Prompt("Weather in Seoul?")); err != nil || !*result.Accepted {
		t.Errorf("loop with a custom decision model: %v, %v", result, err)
	}
}

// PPLX Decider --------------------------------------------------------------------------------------

const revision = "14c9d90137962d300f3de95633f7b6535e80c5ce"

var provenance = obj("model_id", "perplexity-ai/pplx-decider-v1.1-27b", "revision", revision,
	"native_attention", "noncausal_full_attention", "max_input_tokens", 8192, "temperature", 1.0087417621345625)

func pplxAnswer(function, arguments any) *Object {
	body := answerBody(function, arguments)
	body.Set("model", "autojev-qwen3.8-27b")
	return body
}

func pplxWith(t *testing.T, handler func(sent) reply, opts ...DecisionOption) (*SystemOne, *[]sent) {
	t.Helper()
	client, seen := mock(handler)
	decider, err := NewPPLXDecider("http://127.0.0.1:8000", append([]DecisionOption{WithHTTPClient(client)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return decider, seen
}

func TestPPLXUsesTheSameQuestionsAndReportsTheCheckpoint(t *testing.T) {
	decider, seen := pplxWith(t, func(sent) reply { return reply{body: pplxAnswer(0.9, 0.2)} })
	record, err := decider.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if err != nil {
		t.Fatal(err)
	}
	body := object((*seen)[0].body)
	if (*seen)[0].url != "http://127.0.0.1:8000/v1/systemone" || body.Value("model") != "jev-latest" ||
		!Equal(object(body.Value("questions")).Value("function_correct"), FunctionQuestion.Object()) {
		t.Errorf("request %s", body)
	}
	if record.Model != PPLXModelID || record.Accepted || record.Feedback.Value("source") != pplxFeedbackSource ||
		(*seen)[0].header.Get("Authorization") != "" {
		t.Errorf("record %+v", record)
	}
}

func TestPPLXMissingUsageStaysUnknown(t *testing.T) {
	decider, _ := pplxWith(t, func(sent) reply {
		return reply{body: obj("model", "autojev-qwen3.8-27b", "answers", answerBody(0.95, 0.95).Value("answers"))}
	})
	record, err := decider.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	if err != nil || len(record.Usage) != 0 {
		t.Errorf("usage %v, %v", record, err)
	}
}

func TestPPLXRejectsAnotherModel(t *testing.T) {
	other := answerBody(0.95, 0.95)
	other.Set("model", "some-other-model")
	wrongRevision := answerBody(0.95, 0.95)
	wrongRevision.Set("model", PPLXModelID)
	wrongRevision.Set("revision", strings.Repeat("0", 40))
	for _, body := range []*Object{other, wrongRevision} {
		decider, _ := pplxWith(t, func(sent) reply { return reply{body: body} })
		_, err := decider.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
		var failure *DecisionModelError
		if !errors.As(err, &failure) || failure.Code != "model_mismatch" {
			t.Errorf("body %s: %v", body, err)
		}
	}
}

func TestPPLXProvenanceCheckIsOptIn(t *testing.T) {
	handler := func(request sent) reply {
		if strings.HasSuffix(request.url, "/provenance") {
			return reply{body: provenance}
		}
		return reply{body: pplxAnswer(0.95, 0.95)}
	}
	paths := func(seen []sent) []string {
		var out []string
		for _, request := range seen {
			out = append(out, request.url[strings.LastIndex(request.url, "/"):])
		}
		return out
	}
	decider, seen := pplxWith(t, handler, WithCheckProvenance(true))
	for i := 0; i < 2; i++ {
		if _, err := decider.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul()); err != nil {
			t.Fatal(err)
		}
	}
	assertEqual(t, "checked paths", paths(*seen), []string{"/provenance", "/systemone", "/systemone"})
	unchecked, seen := pplxWith(t, handler)
	if _, err := unchecked.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul()); err != nil {
		t.Fatal(err)
	}
	assertEqual(t, "unchecked paths", paths(*seen), []string{"/systemone"})
	wrong := provenance.Clone()
	wrong.Set("revision", "main")
	mismatched, _ := pplxWith(t, func(sent) reply { return reply{body: wrong} }, WithCheckProvenance(true))
	_, err := mismatched.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	var failure *DecisionModelError
	if !errors.As(err, &failure) || failure.Code != "provenance_mismatch" {
		t.Errorf("error %v", err)
	}
}

func TestPPLXContextLimitIsAnError(t *testing.T) {
	over, _ := pplxWith(t, func(sent) reply {
		return reply{status: 422, body: obj("detail", "Prepared input exceeds the 8192-token limit.")}
	})
	_, err := over.Judge(ctx, exampleMessages, []Tool{weatherTool}, seoul())
	var failure *DecisionModelError
	if !errors.As(err, &failure) || failure.Code != "context_limit_exceeded" {
		t.Errorf("error %v", err)
	}
}

func TestPPLXRejectsUnsafeURLs(t *testing.T) {
	for _, url := range []string{"http://10.0.0.4:8000", "https://user:pw@example.com", "http://localhost:8000/v1",
		"ftp://localhost"} {
		if _, err := NewPPLXDecider(url); err == nil {
			t.Errorf("%s must be refused", url)
		}
	}
	decider, err := NewPPLXDecider("https://decider.example.com")
	if err != nil || decider.BaseURL() != "https://decider.example.com" {
		t.Errorf("https origin: %v, %v", decider, err)
	}
	if _, err := NewPPLXDecider("http://[::1]:8000"); err != nil {
		t.Errorf("IPv6 loopback: %v", err)
	}
}
