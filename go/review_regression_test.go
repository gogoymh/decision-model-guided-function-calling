package dmcall

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func TestDebugFormattingHidesKeys(t *testing.T) {
	provider, err := NewAnthropic("m", WithAPIKey(secret))
	if err != nil {
		t.Fatal(err)
	}
	decision, err := NewJev(WithAPIKey(secret))
	if err != nil {
		t.Fatal(err)
	}
	client, err := New(provider, WithDecisionModel(decision))
	if err != nil {
		t.Fatal(err)
	}
	values := []any{provider, decision, client, *provider, *client, reflect.ValueOf(decision).Elem().Interface(),
		[]any{provider, decision, client}, map[string]any{"provider": *provider, "client": *client}}
	for _, value := range values {
		// %s and %q on a struct value fall back to printing its fields unless the type formats itself.
		for _, format := range []string{"%v", "%+v", "%#v", "%s", "%q"} {
			if got := fmt.Sprintf(format, value); strings.Contains(got, secret) {
				t.Fatalf("%s formatting of %T disclosed a credential", format, value)
			}
		}
	}
}

func TestPPLXDeciderWorksWithWrappedDefaultTransport(t *testing.T) {
	previous := http.DefaultTransport
	defer func() { http.DefaultTransport = previous }()
	http.DefaultTransport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("constructor must not send a request")
		return nil, nil
	})
	decision, err := NewPPLXDecider("http://127.0.0.1:8000")
	if err != nil {
		t.Fatal(err)
	}
	transport, ok := decision.client.Transport.(*http.Transport)
	if !ok || transport.Proxy != nil || decision.client.CheckRedirect == nil {
		t.Fatal("native decision-model client must bypass proxies and refuse redirects")
	}
}

func TestRunRejectsMissingHandlersBeforeGeneration(t *testing.T) {
	decision := defaultJev(t)
	loop, model := agent(t, decision, [][]Call{seoul()})
	result, err := loop.Run(ctx, Prompt("Weather in Seoul?"))
	if err == nil || result != nil || len(model.histories) != 0 || len(*decision.requests) != 0 {
		t.Fatalf("missing handler was not rejected before model work: result %v, error %v", result, err)
	}
}

func TestPerCallToolsKeepRegisteredHandlersAndAllowReplacement(t *testing.T) {
	original := func(context.Context, *Object) (any, error) { return "original", nil }
	replacement := func(context.Context, *Object) (any, error) { return "replacement", nil }
	for _, embedded := range []bool{false, true} {
		for _, override := range []bool{false, true} {
			tool := weatherTool
			options := []Option{}
			if embedded {
				tool.Handler = original
			} else {
				options = append(options, WithHandlers(map[string]Handler{tool.Name: original}))
			}
			options = append(options, WithTools(tool))
			loop, err := New(script(seoul()), append(options, WithDecisionModel(defaultJev(t).jev))...)
			if err != nil {
				t.Fatal(err)
			}
			callOptions := []CallOption{WithTools(weatherTool)}
			expected := "original"
			if override {
				callOptions = append(callOptions, WithHandlers(map[string]Handler{tool.Name: replacement}))
				expected = "replacement"
			}
			result, err := loop.Run(ctx, Prompt("Weather in Seoul?"), callOptions...)
			if err != nil || result.Output() != expected || !result.Executed {
				t.Fatalf("embedded=%v override=%v: result %v, error %v", embedded, override, result, err)
			}
		}
	}
}

func TestToolDefinitionConformance(t *testing.T) {
	data := spec(t, "conformance/tools.json")
	for _, item := range list(data.Value("cases")) {
		c := object(item)
		t.Run(c.Value("name").(string), func(t *testing.T) {
			input := c.Value("input")
			before := text(t, input)
			tool, err := ParseTool(input)
			if err != nil {
				t.Fatal(err)
			}
			checked, err := checkedTool(tool)
			if err != nil {
				t.Fatal(err)
			}
			assertEqual(t, "definition", obj("name", checked.Name, "description", checked.Description, "parameters", checked.Parameters), c.Value("expected"))
			if text(t, input) != before {
				t.Fatal("normalization mutated the caller's definition")
			}
		})
	}
	for _, item := range list(data.Value("configuration_errors")) {
		c := object(item)
		t.Run(c.Value("name").(string), func(t *testing.T) {
			tool, err := ParseTool(c.Value("input"))
			if err == nil {
				_, err = New(script(), WithTools(tool), WithDecisionModel(defaultJev(t).jev))
			}
			if err == nil {
				t.Fatal("invalid definition was accepted")
			}
		})
	}
}

func TestCompiledSchemaCacheTracksSchemaChanges(t *testing.T) {
	parameters := obj("type", "object", "properties", obj("a", obj("type", "string")))
	first, err := compileSchema(parameters)
	if err != nil {
		t.Fatal(err)
	}
	again, err := compileSchema(parameters.Clone())
	if err != nil || first != again {
		t.Fatal("unchanged schemas must reuse an immutable validator")
	}
	object(object(parameters.Value("properties")).Value("a")).Set("type", "integer")
	changed, err := compileSchema(parameters)
	if err != nil || changed == first {
		t.Fatal("a changed schema must get a new validator")
	}
	if issues, err := first.validate(obj("a", "text")); err != nil || len(issues) != 0 {
		t.Fatal("caller mutation changed cached validation")
	}
	if issues, err := changed.validate(obj("a", "text")); err != nil || len(issues) != 1 {
		t.Fatal("changed schema was not used")
	}
}

func TestReferencesCannotPromoteDataOrSchemaMapsIntoSchemas(t *testing.T) {
	for _, parameters := range []string{
		`{"type":"object","default":{"type":"PREMIUM"},"properties":{"a":{"$ref":"#/default"}}}`,
		`{"type":"object","default":{"type":"integer","minimum":10},"properties":{"a":{"$ref":"#/default"}}}`,
		`{"type":"object","properties":{"type":{"type":"integer"},"a":{"$ref":"#/properties"}}}`,
	} {
		model := script(seoul())
		decision := defaultJev(t)
		tool := Tool{Name: "f", Parameters: MustParseJSON(parameters)}
		if _, err := New(model, WithTools(tool), WithDecisionModel(decision.jev)); err == nil {
			t.Fatal("a reference to data or a schema map must fail before generation")
		}
		if len(model.histories) != 0 || len(*decision.requests) != 0 {
			t.Fatal("invalid reference configuration sent model work")
		}
		if _, err := CheckSchema([]Tool{tool}, []Call{}); err == nil {
			t.Fatal("an empty candidate must not bypass reference configuration checks")
		}
	}
}

func TestTypeNormalizationPreservesAllNonSchemaData(t *testing.T) {
	input := MustParseJSON(`{"type":"OBJECT","dependentRequired":{"type":["PREMIUM"]},
		"x-extension":{"type":"CUSTOM"},"dependencies":{"type":["REQUIRED"],
		"details":{"type":"OBJECT","properties":{"type":{"type":"STRING"}}}},
		"contains":{"type":"INTEGER"},"if":{"type":"STRING"},"then":{"type":"STRING"},
		"not":{"type":"ARRAY","items":{"type":"STRING"}}}`).(*Object)
	actual := lowerTypes(input).(*Object)
	assertEqual(t, "dependency names", actual.Value("dependentRequired"), input.Value("dependentRequired"))
	assertEqual(t, "extension data", actual.Value("x-extension"), input.Value("x-extension"))
	assertEqual(t, "dependency array data", object(actual.Value("dependencies")).Value("type"), []any{"REQUIRED"})
	assertEqual(t, "dependency schema", object(actual.Value("dependencies")).Value("details"),
		obj("type", "object", "properties", obj("type", obj("type", "string"))))
	assertEqual(t, "contains schema", actual.Value("contains"), obj("type", "integer"))
	assertEqual(t, "conditional schema", actual.Value("if"), obj("type", "string"))
	assertEqual(t, "negative schema", actual.Value("not"), obj("type", "array", "items", obj("type", "string")))
	if input.Value("type") != "OBJECT" {
		t.Fatal("normalization changed the input")
	}
}

func TestMalformedGeminiFunctionCallPartsStayUnreadable(t *testing.T) {
	for _, value := range []any{nil, "not a call", []any{}} {
		candidate := obj("finishReason", "STOP", "content", obj("parts", []any{obj("functionCall", value)}))
		decoded, err := decodeGemini(obj("candidates", []any{candidate}))
		if err != nil || decoded.incomplete != IncompleteMalformedCall || decoded.calls != nil {
			t.Fatalf("malformed functionCall was not unreadable: %+v, %v", decoded, err)
		}
	}
}

func TestNilTypedHandlerFailsBeforeModelWork(t *testing.T) {
	handler := TypedHandler[struct{}, string](nil)
	if handler != nil {
		t.Fatal("a nil typed function must not be wrapped into a callable handler")
	}
	model := script(seoul())
	decision := defaultJev(t)
	if _, err := New(model, WithTools(weatherTool), WithHandlers(map[string]Handler{weatherTool.Name: handler}),
		WithDecisionModel(decision.jev)); err == nil {
		t.Fatal("a nil execution function must be refused during configuration")
	}
	if len(model.histories) != 0 || len(*decision.requests) != 0 {
		t.Fatal("a nil execution function sent model work")
	}
	tool := weatherTool
	tool.Handler = handler
	loop, err := New(model, WithTools(tool), WithDecisionModel(decision.jev))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := loop.Run(ctx, Prompt("Weather in Seoul?")); err == nil {
		t.Fatal("Run must reject a tool with a nil typed function")
	}
	if len(model.histories) != 0 || len(*decision.requests) != 0 {
		t.Fatal("Run sent model work for a tool with a nil typed function")
	}
}

func TestReferenceBooleanTargetsFollowTheDeclaredDraft(t *testing.T) {
	for _, item := range []struct {
		draft string
		valid bool
	}{
		{"http://json-schema.org/draft-04/schema#", false},
		{"http://json-schema.org/draft-07/schema#", true},
		{"https://json-schema.org/draft/2020-12/schema", true},
	} {
		parameters := obj("$schema", item.draft, "type", "object", "default", true,
			"properties", obj("a", obj("$ref", "#/default")))
		_, err := compileSchema(parameters)
		if (err == nil) != item.valid {
			t.Fatalf("boolean reference target for %s: %v", item.draft, err)
		}
	}
}
