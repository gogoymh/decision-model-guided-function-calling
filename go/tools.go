package dmcall

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode"
)

// Handler runs a tool with the arguments of an accepted call. args is the call's JSON object;
// use args.Decode(&params) to fill a struct, or TypedHandler to do that for you.
type Handler func(ctx context.Context, args *Object) (any, error)

// TypedHandler adapts a function that takes its arguments as a struct (decoded with encoding/json).
func TypedHandler[In any, Out any](fn func(ctx context.Context, args In) (Out, error)) Handler {
	return func(ctx context.Context, args *Object) (any, error) {
		var in In
		if args != nil {
			if err := args.Decode(&in); err != nil {
				return nil, fmt.Errorf("dmcall: decoding tool arguments: %w", err)
			}
		}
		return fn(ctx, in)
	}
}

// Tool is a function the model may call: a name, a description and a JSON Schema for its arguments,
// optionally with the Handler that runs it.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object: an *Object (MustParseJSON keeps its key order), a
	// map[string]any, a json.RawMessage or any value encoding/json can marshal. Nil means no
	// parameters, {"type": "object", "properties": {}}.
	Parameters any
	Handler    Handler
}

func (t Tool) jsonValue() any {
	var parameters any = emptyParameters()
	if t.Parameters != nil {
		parameters = t.Parameters
	}
	return obj("type", "function", "function", obj("name", t.Name, "description", t.Description,
		"parameters", parameters))
}

func emptyParameters() *Object {
	return obj("type", "object", "properties", NewObject())
}

// ParseTool reads one tool definition in OpenAI (Chat Completions or Responses), Anthropic
// (input_schema) or Gemini (parameters or parametersJsonSchema, upper-case types allowed) format.
// def is a JSON value, a json.RawMessage, or JSON text as []byte.
func ParseTool(def any) (Tool, error) {
	if data, ok := def.([]byte); ok {
		def = json.RawMessage(data)
	}
	value, err := toValue(def)
	if err != nil {
		return Tool{}, err
	}
	definition, ok := value.(*Object)
	if !ok {
		return Tool{}, errors.New("dmcall: a tool definition must be a JSON object")
	}
	if inner, ok := definition.Value("function").(*Object); ok {
		definition = inner
	}
	if definition.Has("functionDeclarations") || definition.Has("function_declarations") {
		return Tool{}, errors.New("dmcall: pass each Gemini function declaration as its own tool")
	}
	tool := Tool{}
	if name, ok := definition.Value("name").(string); ok {
		tool.Name = name
	}
	if description, ok := definition.Value("description").(string); ok {
		tool.Description = description
	}
	for _, key := range []string{"parameters", "input_schema", "parametersJsonSchema", "parameters_json_schema"} {
		if parameters := definition.Value(key); parameters != nil {
			tool.Parameters = lowerTypes(Clone(parameters))
			break
		}
	}
	return tool, nil
}

// ParseTools reads a JSON array of tool definitions (or a single definition) in any format ParseTool accepts.
func ParseTools(data []byte) ([]Tool, error) {
	value, err := ParseJSON(data)
	if err != nil {
		return nil, err
	}
	items, ok := value.([]any)
	if !ok {
		items = []any{value}
	}
	tools := make([]Tool, 0, len(items))
	for _, item := range items {
		tool, err := ParseTool(item)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

// lowerTypes spells Gemini's OpenAPI-style types (OBJECT, STRING) the JSON Schema way.
func lowerTypes(v any) any {
	switch x := v.(type) {
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = lowerTypes(item)
		}
		return out
	case *Object:
		out := NewObject()
		for _, key := range x.keys {
			value := x.values[key]
			if key == "type" {
				switch t := value.(type) {
				case string:
					if isUpper(t) {
						value = strings.ToLower(t)
					}
				case []any:
					items := make([]any, len(t))
					for i, item := range t {
						if s, ok := item.(string); ok && isUpper(s) {
							item = strings.ToLower(s)
						}
						items[i] = item
					}
					value = items
				}
				out.Set(key, value)
				continue
			}
			out.Set(key, lowerTypes(value))
		}
		return out
	}
	return v
}

// isUpper is Python's str.isupper: at least one cased character and no lower-case one.
func isUpper(s string) bool {
	cased := false
	for _, r := range s {
		if unicode.IsLower(r) {
			return false
		}
		if unicode.IsUpper(r) || unicode.IsTitle(r) {
			cased = true
		}
	}
	return cased
}

// toolset checks the tools and returns normalized copies (parameters as JSON values, defaulted) and
// the handlers by name.
func toolset(tools []Tool, handlers map[string]Handler) ([]Tool, map[string]Handler, error) {
	out := make([]Tool, 0, len(tools))
	registry := map[string]Handler{}
	names := map[string]bool{}
	for _, tool := range tools {
		normalized, err := checkedTool(tool)
		if err != nil {
			return nil, nil, err
		}
		if names[normalized.Name] {
			return nil, nil, fmt.Errorf("dmcall: two tools are named %s", pyStr(normalized.Name))
		}
		names[normalized.Name] = true
		out = append(out, normalized)
		if tool.Handler != nil {
			registry[tool.Name] = tool.Handler
		}
	}
	for name, handler := range handlers {
		if !names[name] {
			return nil, nil, fmt.Errorf("dmcall: WithHandlers has %s, which is not one of the tools", pyStr(name))
		}
		if handler == nil {
			return nil, nil, fmt.Errorf("dmcall: the handler for %s is nil", pyStr(name))
		}
		registry[name] = handler
	}
	return out, registry, nil
}

func checkedTool(tool Tool) (Tool, error) {
	if strings.TrimSpace(tool.Name) == "" {
		return Tool{}, errors.New("dmcall: every tool needs a nonempty name")
	}
	parameters, err := toValue(tool.Parameters)
	if err != nil {
		return Tool{}, fmt.Errorf("dmcall: tool %s: %w", pyStr(tool.Name), err)
	}
	if parameters == nil {
		parameters = emptyParameters()
	}
	schema, ok := parameters.(*Object)
	if !ok || (schema.Has("type") && schema.Value("type") != "object") {
		return Tool{}, fmt.Errorf("dmcall: tool %s: parameters must be a JSON Schema object with type 'object'",
			pyStr(tool.Name))
	}
	if !finite(schema) {
		return Tool{}, fmt.Errorf("dmcall: tool %s: parameters are not a valid JSON Schema (ValueError)", pyStr(tool.Name))
	}
	if err := checkMetaSchema(schema); err != nil {
		return Tool{}, fmt.Errorf("dmcall: tool %s: parameters are not a valid JSON Schema (SchemaError)", pyStr(tool.Name))
	}
	return Tool{Name: tool.Name, Description: tool.Description, Parameters: schema, Handler: tool.Handler}, nil
}

// finite reports whether a JSON value holds no NaN or infinity (json.dumps(allow_nan=False) would pass).
func finite(v any) bool {
	switch x := v.(type) {
	case float64:
		return !math.IsNaN(x) && !math.IsInf(x, 0)
	case []any:
		for _, item := range x {
			if !finite(item) {
				return false
			}
		}
	case *Object:
		for _, key := range x.keys {
			if !finite(x.values[key]) {
				return false
			}
		}
	}
	return true
}

// Calls ------------------------------------------------------------------------------------------------

// Call is one proposed function call.
type Call struct {
	Name string
	// Arguments is the call's argument object; nil when the model's arguments were not a JSON object.
	Arguments *Object
}

// Object returns the call as {"name", "arguments"}.
func (c Call) Object() *Object {
	var arguments any
	if c.Arguments != nil {
		arguments = c.Arguments.Clone()
	}
	return obj("name", c.Name, "arguments", arguments)
}

func (c Call) jsonValue() any { return c.Object() }

// String formats the call like get_weather(city="Seoul").
func (c Call) String() string {
	if c.Arguments == nil {
		return c.Name + "(<malformed arguments>)"
	}
	parts := make([]string, 0, c.Arguments.Len())
	for _, key := range c.Arguments.keys {
		text, err := JSONText(c.Arguments.values[key])
		if err != nil {
			text = "<invalid>"
		}
		parts = append(parts, key+"="+text)
	}
	return c.Name + "(" + strings.Join(parts, ", ") + ")"
}

// callsValue is a candidate as a JSON value: null when unreadable (nil), a list otherwise.
func callsValue(calls []Call) any {
	if calls == nil {
		return nil
	}
	out := make([]any, len(calls))
	for i, call := range calls {
		out[i] = call.Object()
	}
	return out
}

func cloneCalls(calls []Call) []Call {
	if calls == nil {
		return nil
	}
	out := make([]Call, len(calls))
	for i, call := range calls {
		out[i] = Call{Name: call.Name, Arguments: call.Arguments.Clone()}
	}
	return out
}

// NormalizeCalls returns calls as []Call from any SDK shape: DM-Call's {"name", "arguments"},
// OpenAI Chat Completions ({"function": {"name", "arguments": "<json>"}}) and Responses
// (function_call items), Anthropic tool_use blocks ("input"), Gemini functionCall parts ("args"),
// a single call, a []Call or []ToolCall, or JSON text as json.RawMessage. nil means no call.
// Arguments that are not a JSON object (or not valid JSON) become nil, which the schema check rejects.
func NormalizeCalls(calls any) ([]Call, error) {
	switch x := calls.(type) {
	case nil:
		return []Call{}, nil
	case []Call:
		if x == nil {
			return []Call{}, nil
		}
		return cloneCalls(x), nil
	case Call:
		return cloneCalls([]Call{x}), nil
	case []ToolCall:
		out := make([]Call, 0, len(x))
		for _, call := range x {
			out = append(out, Call{Name: call.Function.Name, Arguments: decodeArguments(call.Function.Arguments)})
		}
		return out, nil
	case ToolCall:
		return NormalizeCalls([]ToolCall{x})
	case []byte:
		calls = json.RawMessage(x)
	case string:
		return nil, errors.New("dmcall: calls must be a call or a list of calls")
	}
	value, err := toValue(calls)
	if err != nil {
		return nil, err
	}
	var items []any
	switch x := value.(type) {
	case nil:
		return []Call{}, nil
	case *Object:
		items = []any{x}
	case []any:
		items = x
	default:
		return nil, errors.New("dmcall: calls must be a call or a list of calls")
	}
	out := make([]Call, 0, len(items))
	for _, item := range items {
		call, ok := item.(*Object)
		if !ok {
			return nil, errors.New("dmcall: each call must be an object or a tool-call value")
		}
		for _, key := range []string{"functionCall", "function_call", "function"} {
			if inner, ok := call.Value(key).(*Object); ok {
				call = inner
				break
			}
		}
		var arguments any = NewObject()
		for _, key := range []string{"arguments", "input", "args"} {
			if value, ok := call.Get(key); ok {
				arguments = value
				break
			}
		}
		normalized := Call{}
		normalized.Name, _ = call.Value("name").(string)
		switch a := arguments.(type) {
		case nil:
			normalized.Arguments = NewObject()
		case string:
			normalized.Arguments = decodeArguments(a)
		case *Object:
			normalized.Arguments = a.Clone()
		}
		out = append(out, normalized)
	}
	return out, nil
}

// decodeArguments reads encoded arguments: blank text is {}, invalid JSON or a non-object is nil.
func decodeArguments(text string) *Object {
	if strings.TrimSpace(text) == "" {
		return NewObject()
	}
	value, err := ParseJSON([]byte(text))
	if err != nil {
		return nil
	}
	arguments, _ := value.(*Object)
	return arguments
}

// Messages ------------------------------------------------------------------------------------------

// Message is one turn of the conversation. The generator receives role and content. Verify also
// passes earlier tool calls and tool results to the decision model.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is an earlier tool call in an assistant message, in OpenAI's shape.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall is the function part of a ToolCall; Arguments is JSON text.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (t ToolCall) jsonValue() any {
	kind := t.Type
	if kind == "" {
		kind = "function"
	}
	return obj("id", t.ID, "type", kind, "function", obj("name", t.Function.Name, "arguments", t.Function.Arguments))
}

// jsonValue is the message as the decision model sees it: role, content, name, tool_calls and
// tool_call_id. An assistant turn that only calls tools (empty content) has no content field.
func (m Message) jsonValue() any {
	out := obj("role", m.Role)
	if m.Content != "" || len(m.ToolCalls) == 0 {
		out.Set("content", m.Content)
	}
	if m.Name != "" {
		out.Set("name", m.Name)
	}
	if len(m.ToolCalls) > 0 {
		calls := make([]any, len(m.ToolCalls))
		for i, call := range m.ToolCalls {
			calls[i] = call.jsonValue()
		}
		out.Set("tool_calls", calls)
	}
	if m.ToolCallID != "" {
		out.Set("tool_call_id", m.ToolCallID)
	}
	return out
}

// Prompt is a conversation of one user message.
func Prompt(text string) []Message { return []Message{{Role: "user", Content: text}} }

var textParts = map[string]bool{"text": true, "input_text": true, "output_text": true}

// ParseMessages reads a conversation in OpenAI's chat format from a JSON value, a json.RawMessage
// or []byte: an array of messages, one message, or a string (one user message). Content may be
// text, null or a list of text parts (joined with newlines); tool_calls and tool_call_id are kept.
func ParseMessages(v any) ([]Message, error) {
	if data, ok := v.([]byte); ok {
		v = json.RawMessage(data)
	}
	value, err := toValue(v)
	if err != nil {
		return nil, err
	}
	var items []any
	switch x := value.(type) {
	case string:
		return Prompt(x), nil
	case *Object:
		items = []any{x}
	case []any:
		items = x
	default:
		return nil, errors.New("dmcall: messages must be a string, a message or a list of messages")
	}
	out := make([]Message, 0, len(items))
	for _, item := range items {
		message, ok := item.(*Object)
		if !ok {
			return nil, errors.New("dmcall: each message must be an object")
		}
		parsed := Message{}
		parsed.Role, _ = message.Value("role").(string)
		switch content := message.Value("content").(type) {
		case nil:
		case string:
			parsed.Content = content
		case []any:
			parts := make([]string, 0, len(content))
			for _, part := range content {
				p, ok := part.(*Object)
				kind, _ := p.Value("type").(string)
				text, isText := p.Value("text").(string)
				if !ok || !textParts[kind] || !isText {
					return nil, errors.New("dmcall: only text message content is supported")
				}
				parts = append(parts, text)
			}
			parsed.Content = strings.Join(parts, "\n")
		default:
			return nil, errors.New("dmcall: message content must be text")
		}
		parsed.Name, _ = message.Value("name").(string)
		parsed.ToolCallID, _ = message.Value("tool_call_id").(string)
		if calls, ok := message.Value("tool_calls").([]any); ok {
			for _, raw := range calls {
				call, ok := raw.(*Object)
				if !ok {
					return nil, errors.New("dmcall: each tool call must be an object")
				}
				parsedCall := ToolCall{}
				parsedCall.ID, _ = call.Value("id").(string)
				parsedCall.Type, _ = call.Value("type").(string)
				function, _ := call.Value("function").(*Object)
				parsedCall.Function.Name, _ = function.Value("name").(string)
				switch arguments := function.Value("arguments").(type) {
				case string:
					parsedCall.Function.Arguments = arguments
				case nil:
				default:
					text, err := JSONText(arguments)
					if err != nil {
						return nil, err
					}
					parsedCall.Function.Arguments = text
				}
				parsed.ToolCalls = append(parsed.ToolCalls, parsedCall)
			}
		}
		out = append(out, parsed)
	}
	return out, nil
}

var generatorRoles = map[string]bool{"system": true, "developer": true, "user": true, "assistant": true}
var conversationRoles = map[string]bool{"system": true, "developer": true, "user": true, "assistant": true,
	"tool": true, "function": true}

// generatorMessages checks a conversation for the loop: text turns with a generator role, as
// role and content only.
func generatorMessages(messages []Message) ([]Message, error) {
	if len(messages) == 0 {
		return nil, errors.New("dmcall: messages must be a nonempty list of messages")
	}
	out := make([]Message, len(messages))
	for i, message := range messages {
		if !generatorRoles[message.Role] {
			return nil, errors.New("dmcall: each message needs a role (system, developer, user or assistant) and " +
				"text content; give earlier tool results to the model as text")
		}
		out[i] = Message{Role: message.Role, Content: message.Content}
	}
	return out, nil
}

// conversation checks a conversation for Verify: text turns, earlier tool calls and tool results.
func conversation(messages []Message) ([]Message, error) {
	if len(messages) == 0 {
		return nil, errors.New("dmcall: messages must not be empty")
	}
	for _, message := range messages {
		if !conversationRoles[message.Role] {
			return nil, errors.New("dmcall: each message needs a role: system, developer, user, assistant or tool")
		}
	}
	out := make([]Message, len(messages))
	copy(out, messages)
	return out, nil
}
