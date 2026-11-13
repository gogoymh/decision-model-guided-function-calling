package dmcall

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// Every test is offline: HTTP goes through a RoundTripper that serves recorded or scripted bodies.

var keyNames = []string{"OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
	"GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY", "MODEL_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY",
	"AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT", "AZURE_OPENAI_API_VERSION", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL",
	"TYPESAFE_MODEL"}

// isolate clears every key and endpoint variable for the test.
func isolate(t *testing.T) {
	t.Helper()
	for _, name := range keyNames {
		t.Setenv(name, "")
	}
}

func noSleep(context.Context, time.Duration) error { return nil }

// spec reads a file of the language-neutral spec folder.
func spec(t *testing.T, name string) *Object {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "spec", name))
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	value, err := ParseJSON(data)
	if err != nil {
		t.Fatalf("parsing %s: %v", name, err)
	}
	return value.(*Object)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// sent is one request a test transport saw.
type sent struct {
	method, url string
	header      http.Header
	body        any
	raw         []byte
}

// reply is a scripted response.
type reply struct {
	status int
	header map[string]string
	body   any // a JSON value, []byte or string
}

// mock returns an HTTP client whose transport records each request and answers with handler.
func mock(handler func(sent) reply) (*http.Client, *[]sent) {
	var mu sync.Mutex
	seen := &[]sent{}
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var raw []byte
		if r.Body != nil {
			raw, _ = io.ReadAll(r.Body)
		}
		request := sent{method: r.Method, url: r.URL.String(), header: r.Header.Clone(), raw: raw}
		if len(raw) > 0 {
			request.body, _ = ParseJSON(raw)
		}
		mu.Lock()
		*seen = append(*seen, request)
		mu.Unlock()
		answer := handler(request)
		var data []byte
		switch body := answer.body.(type) {
		case nil:
		case []byte:
			data = body
		case string:
			data = []byte(body)
		default:
			text, err := encodeJSON(body, compactStyle)
			if err != nil {
				return nil, err
			}
			data = []byte(text)
		}
		header := http.Header{}
		for key, value := range answer.header {
			header.Set(key, value)
		}
		status := answer.status
		if status == 0 {
			status = 200
		}
		return &http.Response{StatusCode: status, Header: header, Body: io.NopCloser(bytes.NewReader(data)),
			Request: r}, nil
	})
	return &http.Client{Transport: transport}, seen
}

// queue answers with the given bodies in order, each with status 200.
func queue(t *testing.T, bodies []any) (*http.Client, *[]sent) {
	var mu sync.Mutex
	remaining := append([]any(nil), bodies...)
	return mock(func(sent) reply {
		mu.Lock()
		defer mu.Unlock()
		if len(remaining) == 0 {
			t.Errorf("more requests than recorded responses")
			return reply{status: 500}
		}
		body := remaining[0]
		remaining = remaining[1:]
		return reply{body: body}
	})
}

var listedHeaders = []string{"authorization", "x-api-key", "api-key", "x-goog-api-key", "anthropic-version", "content-type"}

// headersOf keeps the auth and content headers a conformance file records, with lower-case names.
func headersOf(header http.Header) *Object {
	out := NewObject()
	for _, name := range listedHeaders {
		if values := header.Values(name); len(values) > 0 {
			out.Set(name, strings.Join(values, ", "))
		}
	}
	return out
}

func list(v any) []any {
	items, _ := v.([]any)
	return items
}

func object(v any) *Object {
	o, _ := v.(*Object)
	return o
}

func text(t *testing.T, v any) string {
	t.Helper()
	out, err := JSONText(v)
	if err != nil {
		t.Fatalf("JSONText: %v", err)
	}
	return out
}

func mustTools(t *testing.T, v any) []Tool {
	t.Helper()
	var tools []Tool
	for _, item := range list(v) {
		tool, err := ParseTool(item)
		if err != nil {
			t.Fatalf("ParseTool: %v", err)
		}
		tools = append(tools, tool)
	}
	return tools
}

func mustMessages(t *testing.T, v any) []Message {
	t.Helper()
	messages, err := ParseMessages(v)
	if err != nil {
		t.Fatalf("ParseMessages: %v", err)
	}
	return messages
}

func mustCalls(t *testing.T, v any) []Call {
	t.Helper()
	calls, err := NormalizeCalls(v)
	if err != nil {
		t.Fatalf("NormalizeCalls: %v", err)
	}
	return calls
}

// history converts recorded history records ({"round"?, "calls", "feedback"}) for a generator.
func history(t *testing.T, v any) []HistoryEntry {
	t.Helper()
	var out []HistoryEntry
	for _, item := range list(v) {
		record := object(item)
		entry := HistoryEntry{Feedback: object(record.Value("feedback"))}
		if round, ok := record.Value("round").(int64); ok {
			entry.Round = int(round)
		}
		if record.Value("calls") != nil {
			entry.Calls = mustCalls(t, record.Value("calls"))
		}
		out = append(out, entry)
	}
	return out
}

func proposalObject(p *Proposal) *Object {
	errs := make([]any, len(p.ParseErrors))
	for i, e := range p.ParseErrors {
		errs[i] = e
	}
	orNil := func(s string) any {
		if s == "" {
			return nil
		}
		return s
	}
	return obj("calls", callsValue(p.Calls), "content", orNil(p.Content), "parse_errors", errs,
		"incomplete", orNil(p.Incomplete), "usage", p.Usage.jsonValue(), "model", orNil(p.Model))
}

func attemptsValue(attempts []Attempt) []any {
	out := make([]any, len(attempts))
	for i, a := range attempts {
		out[i] = obj("calls", callsValue(a.Calls), "accepted", optionalBool(a.Accepted),
			"function_probability", optionalFloat(a.FunctionProbability),
			"arguments_probability", optionalFloat(a.ArgumentsProbability),
			"schema_errors", schemaErrorsValue(a.SchemaErrors))
	}
	return out
}

// summary is the result summary the conformance files record; extended adds text, probabilities and usage.
func summary(r *Result, extended bool) *Object {
	out := obj("calls", callsValue(r.Calls), "accepted", optionalBool(r.Accepted), "exhausted", r.Exhausted,
		"repairs", int64(r.Repairs), "final_attempt", int64(r.FinalAttempt),
		"initial_no_call_preserved", r.InitialNoCallPreserved, "attempts", attemptsValue(r.Attempts()))
	if extended {
		var text any
		if r.Text != "" {
			text = r.Text
		}
		out.Set("text", text)
		probabilities := r.Probabilities()
		out.Set("probabilities", obj("function", optionalFloat(probabilities.Function),
			"arguments", optionalFloat(probabilities.Arguments)))
		usage := r.Usage()
		out.Set("usage", obj("generator", usage.Generator.jsonValue(), "decision_model", usage.DecisionModel.jsonValue()))
	}
	return out
}

// loopOptions maps a scenario's Python keyword options to client options.
func loopOptions(t *testing.T, options *Object) []Option {
	t.Helper()
	var out []Option
	for _, key := range options.Keys() {
		value := options.Value(key)
		switch key {
		case "feedback":
			out = append(out, WithFeedback(value.(string)))
		case "history":
			out = append(out, WithHistory(value.(string)))
		case "checks":
			out = append(out, WithChecks(value.(string)))
		case "on_exhaustion":
			out = append(out, WithOnExhaustion(value.(string)))
		case "max_repairs":
			out = append(out, WithMaxRepairs(int(value.(int64))))
		case "preserve_initial_no_call":
			out = append(out, WithPreserveInitialNoCall(value.(bool)))
		default:
			t.Fatalf("unknown loop option %q", key)
		}
	}
	return out
}

func assertEqual(t *testing.T, what string, got, want any) {
	t.Helper()
	if !Equal(got, want) {
		t.Errorf("%s differs\n got: %s\nwant: %s", what, text(t, got), text(t, want))
	}
}
