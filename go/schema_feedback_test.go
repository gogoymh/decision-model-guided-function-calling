package dmcall

import (
	"math"
	"strings"
	"testing"
)

// The schema check S(c) and the feedback representations Φ (ported from
// tests/test_schema_and_feedback.py, plus cases beyond the conformance file).

func check(t *testing.T, tools []Tool, name string, arguments string) *SchemaCheck {
	t.Helper()
	result, err := CheckSchema(tools, []any{obj("name", name, "arguments", MustParseJSON(arguments))})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func toolWith(t *testing.T, name, parameters string) []Tool {
	t.Helper()
	return []Tool{{Name: name, Parameters: MustParseJSON(parameters)}}
}

func messages(result *SchemaCheck) []string {
	var out []string
	for _, e := range result.Errors {
		out = append(out, e.Path+": "+e.Message)
	}
	return out
}

func TestValidCallAndNoCallPass(t *testing.T) {
	if !check(t, []Tool{weatherTool}, "get_weather", `{"city": "Seoul"}`).Accepted {
		t.Error("a valid call passes")
	}
	empty, err := CheckSchema([]Tool{weatherTool}, []Call{})
	if err != nil || !empty.Accepted || !empty.WellFormed || len(empty.Errors) != 0 {
		t.Errorf("no call: %+v, %v", empty, err)
	}
	unreadable, err := CheckSchema([]Tool{weatherTool}, []Call(nil))
	if err != nil || unreadable.Accepted || unreadable.WellFormed {
		t.Errorf("unreadable: %+v, %v", unreadable, err)
	}
}

func TestUnreadableCandidatesAreNotWellFormed(t *testing.T) {
	for _, calls := range []any{nil, []any{"get_weather"}, MustParseJSON(`[{"name": "get_weather", "arguments": "{}"}]`),
		[]any{obj("name", "get_weather", "arguments", obj("city", math.NaN()))}, MustParseJSON(`[{"name": "", "arguments": {}}]`)} {
		result, err := CheckSchema([]Tool{weatherTool}, calls)
		if err != nil || result.Accepted || result.WellFormed {
			t.Errorf("%v: %+v, %v", calls, result, err)
		}
	}
}

func TestBeyondTheConformanceKeywords(t *testing.T) {
	cases := []struct {
		name, parameters, arguments string
		want                        []string
	}{
		{"pattern properties closed", `{"type": "object", "patternProperties": {"^x_": {"type": "integer"}},
			"additionalProperties": false}`, `{"x_a": "s", "b": 1, "a": 2}`,
			[]string{`calls[0].arguments["x_a"]: 's' is not of type 'integer'`,
				`calls[0].arguments: 'a', 'b' do not match any of the regexes: '^x_'`}},
		{"items false after prefix", `{"type": "object", "properties": {"p": {"type": "array",
			"prefixItems": [{"type": "string"}], "items": false}}}`, `{"p": ["a", 1, 2]}`,
			[]string{`calls[0].arguments["p"]: Expected at most 1 item but found 2 extra: [1, 2]`}},
		{"items false one extra", `{"type": "object", "properties": {"p": {"type": "array",
			"prefixItems": [{"type": "string"}, {"type": "string"}], "items": false}}}`, `{"p": ["a", "b", {"k": null}]}`,
			[]string{`calls[0].arguments["p"]: Expected at most 2 items but found 1 extra: {'k': None}`}},
		{"draft-07 ref siblings ignored", `{"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
			"definitions": {"s": {"type": "string"}}, "properties": {"a": {"$ref": "#/definitions/s", "type": "integer"},
			"t": {"type": "array", "items": [{"type": "integer"}]}}}`, `{"a": 3, "t": ["x", "y"]}`,
			[]string{`calls[0].arguments["a"]: 3 is not of type 'string'`,
				`calls[0].arguments["t"][0]: 'x' is not of type 'integer'`}},
		{"draft-04 integer excludes floats", `{"$schema": "http://json-schema.org/draft-04/schema#", "type": "object",
			"properties": {"n": {"type": "integer"}}}`, `{"n": 2.0}`,
			[]string{`calls[0].arguments["n"]: 2.0 is not of type 'integer'`}},
		{"quotes and unicode in keys and values", `{"type": "object", "properties": {"it's": {"type": "integer"},
			"\uc11c\uc6b8": {"type": "integer"}, "q\"": {"type": "integer"}}}`, `{"it's": "a'b", "\uc11c\uc6b8": "x\ny", "q\"": "\u00a0"}`,
			[]string{`calls[0].arguments["it's"]: "a'b" is not of type 'integer'`,
				`calls[0].arguments["q\""]: '\xa0' is not of type 'integer'`,
				"calls[0].arguments[\"\\uc11c\\uc6b8\"]: 'x\\ny' is not of type 'integer'"}},
		{"typed extras in order", `{"type": "object", "properties": {"q": {"type": "string"}},
			"additionalProperties": {"type": "integer"}}`, `{"z": "1", "a": "2", "q": 1}`,
			[]string{`calls[0].arguments["a"]: '2' is not of type 'integer'`,
				`calls[0].arguments["q"]: 1 is not of type 'string'`,
				`calls[0].arguments["z"]: '1' is not of type 'integer'`}},
		{"nested default inside a ref", `{"type": "object", "$defs": {"d": {"type": "string", "default": 0}},
			"properties": {"a": {"$ref": "#/$defs/d"}, "b": {"$ref": "#/$defs/d"}}}`, `{"a": 0, "b": 0.0}`,
			[]string{`calls[0].arguments["b"]: 0.0 is not of type 'string'`}},
		{"allOf and anyOf", `{"type": "object", "properties": {"a": {"allOf": [{"type": "object", "required": ["x"]},
			{"anyOf": [{"type": "object"}, {"type": "null"}]}]}}}`, `{"a": {}}`,
			[]string{`calls[0].arguments["a"]: 'x' is a required property`}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := check(t, toolWith(t, "f", c.parameters), "f", c.arguments)
			assertEqual(t, "errors", messages(result), c.want)
		})
	}
}

func TestInvalidToolSchemaIsAConfigurationError(t *testing.T) {
	for _, parameters := range []string{
		`{"type": "object", "properties": {"a": {"type": "nonsense"}}}`,
		`{"type": "object", "properties": {"a": {"type": []}}}`,
		`{"type": "object", "required": "a"}`,
		`{"type": "object", "properties": {"a": {"items": [{"type": "string"}]}}}`,
		`{"type": "object", "properties": {"a": {"pattern": "("}}}`,
		`{"type": "object", "properties": {"a": {"minLength": -1}}}`,
		`{"type": "object", "anyOf": []}`,
		`{"type": "object", "properties": {"a": 3}}`,
	} {
		if _, err := CheckSchema(toolWith(t, "f", parameters), []any{}); err == nil {
			t.Errorf("%s must be refused", parameters)
		}
	}
	if _, err := CheckSchema([]Tool{weatherTool, weatherTool}, []any{}); err == nil || !strings.Contains(err.Error(), "unique") {
		t.Errorf("duplicate names: %v", err)
	}
	if _, err := CheckSchema(toolWith(t, "f", `{"type": "object", "properties": {"a": {"$ref": "#/nowhere"}}}`),
		[]any{obj("name", "f", "arguments", obj("a", 1))}); err == nil {
		t.Error("an unresolvable reference is an error")
	}
	if _, err := CheckSchema(toolWith(t, "f", `{"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
		"properties": {"a": {"items": [{"type": "string"}]}}}`), []any{}); err != nil {
		t.Errorf("draft-07 allows items as a list: %v", err)
	}
}

func TestLabelBands(t *testing.T) {
	for p, want := range map[float64]string{0: "very likely incorrect", 0.19: "very likely incorrect", 0.2: "likely incorrect",
		0.48: "uncertain", 0.52: "uncertain", 0.6: "likely correct", 0.8: "very likely correct", 1: "very likely correct"} {
		if got, err := Label(p); err != nil || got != want {
			t.Errorf("Label(%v) = %q, %v", p, got, err)
		}
	}
	for _, bad := range []float64{-0.1, 1.1, math.NaN(), math.Inf(1)} {
		if _, err := Label(bad); err == nil {
			t.Errorf("Label(%v) must fail", bad)
		}
	}
}

var sampleFeedback = MustParseJSON(`{"source": "jev_noul_and_deterministic_schema_validation",
	"checks": {"function_selection": {"pass": true, "probability_yes": 0.95, "threshold": 0.5, "schema_pass": true},
		"arguments": {"skipped": false, "pass": false, "probability_yes": 0.12, "threshold": 0.5, "schema_pass": true}},
	"failed_checks": ["arguments"],
	"schema_errors": [{"check": "arguments", "path": "calls[0].arguments", "message": "'probability_yes' 0.3"}],
	"instruction": "Reconsider ... The probabilities are verifier judgments, not ground truth or explanations."}`).(*Object)

func TestHybridKeepsNumbersAndThresholdsAndAddsLabels(t *testing.T) {
	out, err := Represent(sampleFeedback, FeedbackHybrid)
	if err != nil {
		t.Fatal(err)
	}
	arguments := object(object(out.Value("checks")).Value("arguments"))
	assertEqual(t, "keys", arguments.Keys(), []string{"skipped", "pass", "probability_yes", "feedback_label", "threshold", "schema_pass"})
	if arguments.Value("feedback_label") != "very likely incorrect" ||
		object(object(out.Value("checks")).Value("function_selection")).Value("feedback_label") != "very likely correct" {
		t.Errorf("labels %s", out)
	}
	assertEqual(t, "schema errors", out.Value("schema_errors"), sampleFeedback.Value("schema_errors"))
	if object(object(sampleFeedback.Value("checks")).Value("arguments")).Has("feedback_label") {
		t.Error("the input must not change")
	}
}

func TestTextReplacesNumbersAndThresholdsWithLabels(t *testing.T) {
	out, err := Represent(sampleFeedback, FeedbackText)
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, "arguments", object(out.Value("checks")).Value("arguments"),
		obj("skipped", false, "pass", false, "feedback_label", "very likely incorrect", "schema_pass", true))
	if !strings.HasSuffix(out.Value("instruction").(string), "The feedback labels are verifier judgments, not ground truth or explanations.") {
		t.Errorf("instruction %v", out.Value("instruction"))
	}
	assertEqual(t, "schema errors", out.Value("schema_errors"), sampleFeedback.Value("schema_errors"))
}

func TestNumericIsUnchangedAndSkippedQuestionsStayNull(t *testing.T) {
	out, _ := Represent(sampleFeedback, FeedbackNumeric)
	if text(t, out) != text(t, sampleFeedback) {
		t.Error("numeric must equal the record")
	}
	skipped := MustParseJSON(`{"checks": {"arguments": {"skipped": true, "pass": null, "probability_yes": null, "threshold": 0.5}}}`).(*Object)
	hybrid, _ := Represent(skipped, FeedbackHybrid)
	if label, ok := object(object(hybrid.Value("checks")).Value("arguments")).Get("feedback_label"); !ok || label != nil {
		t.Error("a skipped question keeps a null label")
	}
	words, _ := Represent(skipped, FeedbackText)
	assertEqual(t, "text", object(words.Value("checks")).Value("arguments"), obj("skipped", true, "pass", nil, "feedback_label", nil))
	if _, err := Represent(sampleFeedback, "words"); err == nil || !strings.Contains(err.Error(), "feedback") {
		t.Errorf("unknown representation: %v", err)
	}
	represented, err := RepresentHistory([]HistoryEntry{{Calls: []Call{}, Feedback: sampleFeedback}}, FeedbackText)
	if err != nil || text(t, represented[0].Feedback) != text(t, words2(t)) {
		t.Errorf("history %v", err)
	}
	if _, err := RepresentHistory([]HistoryEntry{{Calls: []Call{}}}, FeedbackHybrid); err == nil {
		t.Error("a history record needs feedback")
	}
}

func words2(t *testing.T) *Object {
	out, err := Represent(sampleFeedback, FeedbackText)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestSingleCheckRemovesOnlyTheOtherQuestion(t *testing.T) {
	functionOnly, _ := SingleCheck(sampleFeedback, "function")
	assertEqual(t, "function checks", object(functionOnly.Value("checks")).Keys(), []string{"function_selection"})
	assertEqual(t, "function failed", functionOnly.Value("failed_checks"), []any{})
	argumentsOnly, _ := SingleCheck(sampleFeedback, "arguments")
	assertEqual(t, "argument checks", object(argumentsOnly.Value("checks")).Keys(), []string{"arguments"})
	assertEqual(t, "argument failed", argumentsOnly.Value("failed_checks"), []any{"arguments"})
	assertEqual(t, "schema errors", argumentsOnly.Value("schema_errors"), sampleFeedback.Value("schema_errors"))
	if _, err := SingleCheck(sampleFeedback, "both"); err == nil {
		t.Error("keep must be function or arguments")
	}
}
