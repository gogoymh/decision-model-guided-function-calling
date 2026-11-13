package dmcall

import (
	"context"
	"encoding/json"
	"math"
	"math/big"
	"strings"
	"testing"
)

// Tool definitions in every format, calls from any SDK (ported from tests/test_tools.py) and the
// ordered JSON layer.

var weatherDefinition = MustParseJSON(`{"type": "function", "function": {"name": "get_weather",
	"description": "Current weather for a city.", "parameters": {"type": "object", "properties": {
	"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}`)

func TestToolDefinitionFormats(t *testing.T) {
	function := object(weatherDefinition).Value("function").(*Object)
	parameters := function.Value("parameters")
	flatWithType := function.Clone()
	flatWithType.Set("type", "function")
	anthropic := obj("name", "get_weather", "description", "Current weather for a city.", "input_schema", parameters)
	gemini := MustParseJSON(`{"name": "get_weather", "description": "Current weather for a city.", "parameters": {
		"type": "OBJECT", "properties": {"city": {"type": "STRING"}, "unit": {"type": "STRING", "enum": ["celsius", "fahrenheit"]}},
		"required": ["city"]}}`)
	geminiJSON := obj("name", "get_weather", "description", "Current weather for a city.", "parametersJsonSchema", parameters)
	for i, definition := range []any{weatherDefinition, function, flatWithType, anthropic, gemini, geminiJSON,
		json.RawMessage(text(t, weatherDefinition)), []byte(text(t, weatherDefinition))} {
		tool, err := ParseTool(definition)
		if err != nil {
			t.Fatalf("definition %d: %v", i, err)
		}
		assertEqual(t, "definition", tool, weatherDefinition)
	}
	tools, err := ParseTools([]byte(`[` + text(t, weatherDefinition) + `, {"name": "ping"}]`))
	if err != nil || len(tools) != 2 || tools[1].Parameters != nil || tools[1].Description != "" {
		t.Errorf("ParseTools: %+v, %v", tools, err)
	}
	if _, err := ParseTool(MustParseJSON(`{"functionDeclarations": []}`)); err == nil {
		t.Error("a Gemini tool with several declarations must be split")
	}
	if _, err := ParseTool("get_weather"); err == nil {
		t.Error("a string is not a tool")
	}
}

func TestBadTools(t *testing.T) {
	nop := func(context.Context, *Object) (any, error) { return nil, nil }
	for _, tools := range [][]Tool{
		{{Name: " ", Parameters: MustParseJSON(`{"type": "object"}`)}},
		{{Name: "x", Parameters: MustParseJSON(`{"type": "array"}`)}},
		{{Name: "x", Parameters: MustParseJSON(`{"type": "object", "properties": {"a": {"type": "nonsense"}}}`)}},
		{{Name: "x", Parameters: []any{}}},
		{{Name: "x", Parameters: obj("type", "object", "minimum", math.NaN())}},
		{weatherTool, weatherTool},
	} {
		if _, _, err := toolset(tools, nil); err == nil {
			t.Errorf("%+v must be refused", tools)
		}
	}
	if _, _, err := toolset([]Tool{weatherTool}, map[string]Handler{"nope": nop}); err == nil ||
		!strings.Contains(err.Error(), "not one of the tools") {
		t.Errorf("unknown handler: %v", err)
	}
	schemas, registry, err := toolset([]Tool{{Name: "ping"}, {Name: "m", Parameters: map[string]any{"type": "object"}}},
		map[string]Handler{"ping": nop})
	if err != nil || registry["ping"] == nil {
		t.Fatalf("toolset: %v", err)
	}
	assertEqual(t, "default parameters", schemas[0].Parameters, emptyParameters())
	assertEqual(t, "map parameters", schemas[1].Parameters, obj("type", "object"))
}

func TestTypedHandler(t *testing.T) {
	type in struct {
		City string `json:"city"`
		Days int    `json:"days"`
	}
	handler := TypedHandler(func(_ context.Context, args in) (string, error) {
		return strings.Repeat(args.City, args.Days), nil
	})
	out, err := handler(context.Background(), obj("city", "Seoul", "days", 2))
	if err != nil || out != "SeoulSeoul" {
		t.Errorf("out %v, %v", out, err)
	}
	if _, err := handler(context.Background(), obj("days", "two")); err == nil {
		t.Error("arguments of the wrong type are an error")
	}
}

func TestCallsFromEverySDKShape(t *testing.T) {
	calls, err := NormalizeCalls(json.RawMessage(`[{"id": "c1", "type": "function",
		"function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]`))
	if err != nil || !Equal(calls, seoul()) {
		t.Errorf("raw message: %v, %v", calls, err)
	}
	calls, err = NormalizeCalls([]map[string]any{{"name": "get_weather", "args": map[string]any{"city": "Seoul"}}})
	if err != nil || !Equal(calls, seoul()) {
		t.Errorf("Go maps: %v, %v", calls, err)
	}
	if calls, _ := NormalizeCalls(MustParseJSON(`{"name": "f", "arguments": "{oops"}`)); calls[0].Arguments != nil {
		t.Error("invalid JSON arguments become nil")
	}
	if calls, _ := NormalizeCalls(MustParseJSON(`{"name": "f", "arguments": "[1]"}`)); calls[0].Arguments != nil {
		t.Error("non-object arguments become nil")
	}
	for _, bad := range []any{3, "get_weather", []any{"x"}} {
		if _, err := NormalizeCalls(bad); err == nil {
			t.Errorf("%v must be refused", bad)
		}
	}
	call := Call{Name: "f", Arguments: MustParseJSON(`{"city": "서울", "n": 2.0, "x": null}`).(*Object)}
	if call.String() != `f(city="서울", n=2.0, x=null)` || (Call{Name: "f"}).String() != "f(<malformed arguments>)" {
		t.Errorf("String %s", call)
	}
}

func TestParserFollowsPython(t *testing.T) {
	value := MustParseJSON(`{"a": 1, "b": 2, "a": 3}`).(*Object)
	if text(t, value) != `{"a": 3, "b": 2}` {
		t.Errorf("a repeated key keeps its first position and last value: %s", value)
	}
	for input, want := range map[string]string{
		`NaN`: `NaN`, `[Infinity, -Infinity]`: `[Infinity, -Infinity]`, `-0`: `0`, `1e400`: `Infinity`, `1E2`: `100.0`,
		`123456789012345678901234567890`: `123456789012345678901234567890`, `"\ud83c\udf24"`: `"🌤"`, `"\u00e9"`: `"é"`, `"\ud800x"`: `"�x"`,
		` [1 , 2] `: `[1, 2]`, `"a\/b"`: `"a/b"`, `2.5e-3`: `0.0025`,
	} {
		parsed, err := ParseJSON([]byte(input))
		if err != nil || text(t, parsed) != want {
			t.Errorf("%s -> %v, %v; want %s", input, parsed, err, want)
		}
	}
	for _, bad := range []string{``, `[1,]`, `{"a" 1}`, `01`, `1.`, `.5`, `"a` + "\n" + `"`, `tru`, `[1] x`, `{'a': 1}`, `"\x"`, `-`} {
		if _, err := ParseJSON([]byte(bad)); err == nil {
			t.Errorf("%q must be refused", bad)
		}
	}
	if _, ok := MustParseJSON(`99999999999999999999`).(*big.Int); !ok {
		t.Error("integers beyond int64 stay exact")
	}
}

func TestPythonText(t *testing.T) {
	floats := map[float64]string{1: "1.0", 0.1: "0.1", 1e16: "1e+16", 1e15: "1000000000000000.0", 1.5e-5: "1.5e-05",
		0.0001: "0.0001", 123.456: "123.456", -2.5e100: "-2.5e+100", 5e-324: "5e-324", math.MaxFloat64: "1.7976931348623157e+308"}
	for f, want := range floats {
		if got := pyFloat(f); got != want {
			t.Errorf("pyFloat(%v) = %s, want %s", f, got, want)
		}
	}
	for s, want := range map[string]string{"plain": `'plain'`, "it's": `"it's"`, `both ' "`: `'both \' "'`,
		"tab\t\x01\x7f": `'tab\t\x01\x7f'`, "é서울": `'é서울'`, "\u2028": `'\u2028'`, "\U000e0001": `'\U000e0001'`} {
		if got := pyStr(s); got != want {
			t.Errorf("pyStr(%q) = %s, want %s", s, got, want)
		}
	}
	ascii, _ := encodeJSON("서울 🌤 \x7f", asciiStyle)
	if ascii != `"\uc11c\uc6b8 \ud83c\udf24 \u007f"` {
		t.Errorf("ensure_ascii text %s", ascii)
	}
	if got := pyRepr(MustParseJSON(`{"a": [1, 2.0, null, true, "x"]}`)); got != `{'a': [1, 2.0, None, True, 'x']}` {
		t.Errorf("repr %s", got)
	}
}

func TestObjectAndEquality(t *testing.T) {
	o := obj("b", 1, "a", 2)
	o.Set("b", 3)
	o.Set("c", obj("x", []any{1}))
	assertEqual(t, "keys", o.Keys(), []string{"b", "a", "c"})
	copied := o.Clone()
	object(copied.Value("c")).Set("x", "changed")
	if !Equal(object(o.Value("c")).Value("x"), []any{1}) {
		t.Error("Clone is deep")
	}
	o.Delete("a")
	assertEqual(t, "after delete", o.Keys(), []string{"b", "c"})
	var decoded struct {
		B int `json:"b"`
		C struct {
			X []int `json:"x"`
		} `json:"c"`
	}
	if err := o.Decode(&decoded); err != nil || decoded.B != 3 || decoded.C.X[0] != 1 {
		t.Errorf("Decode %+v, %v", decoded, err)
	}
	data, err := json.Marshal(map[string]any{"wrapped": o})
	if err != nil || string(data) != `{"wrapped":{"b":3,"c":{"x":[1]}}}` {
		t.Errorf("MarshalJSON %s, %v", data, err)
	}
	var round Object
	if err := json.Unmarshal([]byte(`{"z": 1.0, "a": 2}`), &round); err != nil || text(t, &round) != `{"z": 1.0, "a": 2}` {
		t.Errorf("UnmarshalJSON %s, %v", &round, err)
	}
	if !Equal(1, 1.0) || Equal(true, 1) || !Equal(obj("a", 1, "b", 2), obj("b", 2, "a", 1)) || Equal([]any{1, 2}, []any{2, 1}) ||
		Equal(math.NaN(), math.NaN()) || Equal(MustParseJSON(`99999999999999999999`), 1e20) ||
		!Equal(MustParseJSON(`100000000000000000000`), 1e20) {
		t.Error("Equal follows Python's comparison of parsed JSON")
	}
	assertEqual(t, "Map", o.Map(), map[string]any{"b": 3, "c": map[string]any{"x": []any{1}}})
	var zero Object
	zero.Set("k", "v")
	if zero.Len() != 1 || (*Object)(nil).Len() != 0 || (*Object)(nil).Value("x") != nil {
		t.Error("the zero Object is usable and a nil Object reads as empty")
	}
}
