package dmcall

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// The local schema check S(c) of the paper.
//
// S(c) is true when the candidate is readable, names an available function and supplies every
// required argument with its declared JSON type. Value constraints (enum, ranges, lengths,
// patterns, formats) are not checked here: whether a value fits the request and the parameter
// description is the decision model's question. A value equal to a parameter's declared default is
// admissible even when its JSON type differs (for example null), and, as in a Python call, keyword
// names outside a closed parameter list are rejected. An empty collection (no call) passes.
//
// The messages are those of Python's jsonschema package (Draft 2020-12 unless the schema's
// $schema names an earlier draft), and errors are ordered the way the Python reference orders them.

// SchemaError is one finding of the schema check.
type SchemaError struct {
	// Check is "function" or "arguments": which part of the candidate failed.
	Check   string `json:"check"`
	Path    string `json:"path"`
	Message string `json:"message"`
}

func (e SchemaError) jsonValue() any {
	return obj("check", e.Check, "path", e.Path, "message", e.Message)
}

// SchemaCheck is the result of S(c) for one candidate.
type SchemaCheck struct {
	FunctionPass  bool
	ArgumentsPass bool
	// Accepted is S(c): both parts pass.
	Accepted bool
	// WellFormed says the candidate is readable, so the decision model can judge it.
	WellFormed bool
	Errors     []SchemaError
}

func schemaErrorsValue(errs []SchemaError) []any {
	out := make([]any, len(errs))
	for i, e := range errs {
		out[i] = e.jsonValue()
	}
	return out
}

// Feedback returns the check's feedback record.
func (s *SchemaCheck) Feedback() *Object {
	return obj("source", "deterministic_schema_validation", "function_pass", s.FunctionPass,
		"arguments_pass", s.ArgumentsPass, "schema_errors", schemaErrorsValue(s.Errors))
}

// Object returns the result as the Python reference's dict.
func (s *SchemaCheck) Object() *Object {
	return obj("function_pass", s.FunctionPass, "arguments_pass", s.ArgumentsPass, "accepted", s.Accepted,
		"well_formed", s.WellFormed, "errors", schemaErrorsValue(s.Errors), "feedback", s.Feedback())
}

// CheckSchema runs S(c) on proposed calls against the available tools. calls is a JSON value
// (normally a list of {"name", "arguments"} objects), a []Call or JSON text as json.RawMessage; a
// nil value or a nil []Call is an unreadable candidate, an empty list a decision to make no call.
// An invalid tool definition is a configuration error and returns an error.
func CheckSchema(tools []Tool, calls any) (*SchemaCheck, error) {
	compiled, err := compileTools(tools)
	if err != nil {
		return nil, err
	}
	value, err := toValue(calls)
	if err != nil {
		return nil, err
	}
	return compiled.check(value)
}

type compiledTool struct {
	schema    *Object // as declared
	validator *schemaValidator
}

type compiledTools struct {
	byName map[string]*compiledTool
}

// functionDefinitions are the observable part of each tool: name, description and parameters.
func functionDefinitions(tools []Tool) ([]any, error) {
	out := make([]any, 0, len(tools))
	names := map[string]bool{}
	for _, tool := range tools {
		if tool.Name == "" || names[tool.Name] {
			return nil, errors.New("dmcall: available function names must be nonempty and unique")
		}
		names[tool.Name] = true
		parameters, err := toValue(tool.Parameters)
		if err != nil {
			return nil, err
		}
		if parameters == nil {
			parameters = emptyParameters()
		}
		out = append(out, obj("name", tool.Name, "description", tool.Description, "parameters", Clone(parameters)))
	}
	return out, nil
}

func compileTools(tools []Tool) (*compiledTools, error) {
	definitions, err := functionDefinitions(tools)
	if err != nil {
		return nil, err
	}
	compiled := &compiledTools{byName: map[string]*compiledTool{}}
	for _, item := range definitions {
		definition := item.(*Object)
		schema, ok := definition.Value("parameters").(*Object)
		if !ok {
			return nil, errors.New("dmcall: function parameters must be a JSON Schema object")
		}
		validator, err := compileSchema(schema)
		if err != nil {
			return nil, fmt.Errorf("dmcall: an available function has an invalid JSON Schema: %w", err)
		}
		compiled.byName[definition.Value("name").(string)] = &compiledTool{schema: schema, validator: validator}
	}
	return compiled, nil
}

// Cache immutable validators by schema text. Changed tools get a new validator;
// bounded retention prevents dynamic schemas from growing the cache indefinitely.
var compiledSchemas = struct {
	sync.Mutex
	entries map[string]*schemaValidator
	order   []string
}{entries: map[string]*schemaValidator{}}

func compileSchema(schema *Object) (*schemaValidator, error) {
	key, err := encodeJSON(schema, compactStyle)
	if err != nil {
		return nil, err
	}
	compiledSchemas.Lock()
	defer compiledSchemas.Unlock()
	if validator := compiledSchemas.entries[key]; validator != nil {
		return validator, nil
	}
	if err := checkMetaSchema(schema); err != nil {
		return nil, err
	}
	validator := &schemaValidator{root: typeSchema(schema), dialect: dialectOf(schema)}
	if err := validator.checkReferences(); err != nil {
		return nil, err
	}
	if len(compiledSchemas.order) >= 1024 {
		delete(compiledSchemas.entries, compiledSchemas.order[0])
		compiledSchemas.order = compiledSchemas.order[1:]
	}
	compiledSchemas.entries[key] = validator
	compiledSchemas.order = append(compiledSchemas.order, key)
	return validator, nil
}

var errUnresolvableReference = errors.New("dmcall: a JSON Schema $ref cannot be resolved inside the parameters")

// Reject missing references and cycles applied to the same value. Recursion
// through properties/items consumes part of the value and remains allowed.
func (v *schemaValidator) checkReferences() error {
	schemaNodes := map[*Object]bool{}
	state := map[*Object]int{}
	var follow func(*Object, int) error
	follow = func(schema *Object, depth int) error {
		if depth > maxSchemaDepth || state[schema] == 1 {
			return errUnresolvableReference
		}
		if state[schema] == 2 {
			return nil
		}
		state[schema] = 1
		if reference, ok := schema.Value("$ref").(string); ok {
			target, err := resolvePointer(v.root, reference)
			if err != nil {
				return errUnresolvableReference
			}
			switch target := target.(type) {
			case *Object:
				if !schemaNodes[target] {
					return errUnresolvableReference
				}
				if err := follow(target, depth+1); err != nil {
					return err
				}
			case bool:
				if err := metaCheck(target, v.dialect, 0); err != nil {
					return errUnresolvableReference
				}
			default:
				return errUnresolvableReference
			}
			if v.dialect <= draft7 {
				state[schema] = 2
				return nil
			}
		}
		for _, key := range []string{"allOf", "anyOf"} {
			branches, _ := schema.Value(key).([]any)
			for _, branch := range branches {
				if child, ok := branch.(*Object); ok {
					if err := follow(child, depth+1); err != nil {
						return err
					}
				}
			}
		}
		state[schema] = 2
		return nil
	}
	var walk func(any, int) error
	walk = func(value any, depth int) error {
		if depth > maxSchemaDepth {
			return errUnresolvableReference
		}
		switch node := value.(type) {
		case []any:
			for _, item := range node {
				if err := walk(item, depth+1); err != nil {
					return err
				}
			}
		case *Object:
			schemaNodes[node] = true
			for _, key := range node.keys {
				switch key {
				case "properties", "patternProperties", "$defs", "definitions":
					if members, ok := node.Value(key).(*Object); ok {
						for _, name := range members.keys {
							if err := walk(members.Value(name), depth+1); err != nil {
								return err
							}
						}
					}
				case "items", "prefixItems", "additionalProperties", "allOf", "anyOf":
					if err := walk(node.Value(key), depth+1); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}
	if err := walk(v.root, 0); err != nil {
		return err
	}
	for node := range schemaNodes {
		if err := follow(node, 0); err != nil {
			return err
		}
	}
	return nil
}

func (c *compiledTools) check(calls any) (*SchemaCheck, error) {
	var errs []SchemaError
	add := func(check, path, message string) {
		errs = append(errs, SchemaError{Check: check, Path: path, Message: message})
	}
	list, wellFormed := calls.([]any)
	if !wellFormed {
		add("function", "calls", "Proposed calls must be a list.")
		add("arguments", "calls", "No valid call collection to inspect.")
	}
	for index, item := range list {
		path := "calls[" + strconv.Itoa(index) + "]"
		call, ok := item.(*Object)
		if !ok {
			wellFormed = false
			add("function", path, "Each call must be an object.")
			add("arguments", path, "No argument object to inspect.")
			continue
		}
		name, isName := call.Value("name").(string)
		tool := c.byName[name]
		switch {
		case !isName || name == "":
			wellFormed = false
			add("function", path+".name", "Name must be a nonempty string.")
		case tool == nil:
			add("function", path+".name", "Selected function is not available.")
		}
		arguments, ok := call.Value("arguments").(*Object)
		if !ok {
			wellFormed = false
			add("arguments", path+".arguments", "Arguments must be a JSON object.")
			continue
		}
		if !finite(arguments) {
			wellFormed = false
			add("arguments", path+".arguments", "Arguments must be finite JSON values.")
			continue
		}
		if !isName || tool == nil {
			add("arguments", path+".arguments", "Cannot validate arguments for an unavailable function.")
			continue
		}
		issues, err := tool.validator.validate(arguments)
		if err != nil {
			return nil, err
		}
		for _, issue := range issues {
			if issue.declaredDefault() {
				continue
			}
			var suffix strings.Builder
			for _, component := range issue.path {
				text, _ := encodeJSON(component, asciiStyle)
				suffix.WriteString("[" + text + "]")
			}
			add("arguments", path+".arguments"+suffix.String(), issue.message)
		}
		// An ordinary function signature has closed top-level keyword names. Respect an explicit
		// additionalProperties or patternProperties rule.
		if properties, ok := tool.schema.Get("properties"); ok && !tool.schema.Has("additionalProperties") &&
			!tool.schema.Has("patternProperties") {
			declared, _ := properties.(*Object)
			var unknown []string
			for _, key := range arguments.keys {
				if !declared.Has(key) {
					unknown = append(unknown, key)
				}
			}
			sort.Strings(unknown)
			for _, key := range unknown {
				add("arguments", path+".arguments", "Unknown parameter: "+key+".")
			}
		}
	}
	functionPass, argumentsPass := true, true
	for _, e := range errs {
		if e.Check == "function" {
			functionPass = false
		} else {
			argumentsPass = false
		}
	}
	if errs == nil {
		errs = []SchemaError{}
	}
	return &SchemaCheck{FunctionPass: functionPass, ArgumentsPass: argumentsPass,
		Accepted: functionPass && argumentsPass, WellFormed: wellFormed, Errors: errs}, nil
}

// Reducing a schema to its structural typing -----------------------------------------------------

var typeKeywords = map[string]bool{
	"$schema": true, "$id": true, "$ref": true, "$defs": true, "definitions": true, "type": true,
	"properties": true, "required": true, "items": true, "prefixItems": true, "additionalProperties": true,
	"patternProperties": true, "anyOf": true, "allOf": true, "oneOf": true, "default": true,
}

// typeSchema keeps a schema's types, required names and nesting without value constraints. Without
// value constraints oneOf branches may overlap, so any branch suffices: anyOf and oneOf both become
// allOf: [{anyOf: branches}].
func typeSchema(schema any) any {
	switch s := schema.(type) {
	case []any:
		out := make([]any, len(s))
		for i, item := range s {
			out[i] = typeSchema(item)
		}
		return out
	case *Object:
		result := NewObject()
		for _, key := range s.keys {
			value := s.values[key]
			if !typeKeywords[key] {
				continue
			}
			members, isObject := value.(*Object)
			branches, isList := value.([]any)
			switch {
			case (key == "properties" || key == "patternProperties" || key == "$defs" || key == "definitions") && isObject:
				reduced := NewObject()
				for _, name := range members.keys {
					reduced.Set(name, typeSchema(members.values[name]))
				}
				result.Set(key, reduced)
			case key == "items" || key == "prefixItems" || key == "additionalProperties":
				result.Set(key, typeSchema(value))
			case (key == "anyOf" || key == "allOf" || key == "oneOf") && isList:
				reduced := typeSchema(branches).([]any)
				existing, _ := result.Value("allOf").([]any)
				if key == "allOf" {
					result.Set("allOf", append(existing, reduced...))
				} else {
					result.Set("allOf", append(existing, obj("anyOf", reduced)))
				}
			default:
				result.Set(key, Clone(value))
			}
		}
		return result
	}
	return Clone(schema)
}

// Draft detection ---------------------------------------------------------------------------------

const (
	draft4    = 4
	draft6    = 6
	draft7    = 7
	draft2019 = 2019
	draft2020 = 2020
)

var dialects = map[string]int{
	"https://json-schema.org/draft/2020-12/schema": draft2020,
	"https://json-schema.org/draft/2019-09/schema": draft2019,
	"http://json-schema.org/draft-07/schema":       draft7,
	"http://json-schema.org/draft-06/schema":       draft6,
	"http://json-schema.org/draft-04/schema":       draft4,
	"http://json-schema.org/draft-03/schema":       draft4, // approximated by draft-04
}

// dialectOf picks the draft from $schema the way jsonschema's validator_for does (an empty
// fragment is ignored); any other value means Draft 2020-12.
func dialectOf(schema *Object) int {
	uri, _ := schema.Value("$schema").(string)
	if d, ok := dialects[strings.TrimSuffix(uri, "#")]; ok {
		return d
	}
	return draft2020
}

// Validation --------------------------------------------------------------------------------------

type schemaValidator struct {
	root    any
	dialect int
}

type schemaIssue struct {
	path     []any // string keys and int indexes
	message  string
	keyword  string // the keyword that failed; "-" for a false schema
	schema   *Object
	instance any
}

// declaredDefault reports whether a type error is about a value equal to the parameter's declared default.
func (i schemaIssue) declaredDefault() bool {
	if i.keyword != "type" || i.schema == nil {
		return false
	}
	declared, ok := i.schema.Get("default")
	if !ok {
		return false
	}
	a, errA := encodeJSON(i.instance, canonicalStyle)
	b, errB := encodeJSON(declared, canonicalStyle)
	return errA == nil && errB == nil && a == b
}

// pathKey is Python's str(list(error.path)), the sort key of the reference.
func (i schemaIssue) pathKey() string {
	parts := make([]string, len(i.path))
	for n, component := range i.path {
		parts[n] = pyRepr(component)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

const maxSchemaDepth = 1000

var errSchemaDepth = errors.New("dmcall: schema recursion is too deep")

func (v *schemaValidator) validate(instance any) ([]schemaIssue, error) {
	issues, err := v.iter(instance, v.root, 0)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(issues, func(a, b int) bool { return issues[a].pathKey() < issues[b].pathKey() })
	return issues, nil
}

// descend validates instance against a subschema and puts component in front of each error's path.
// Like jsonschema's descend, a subschema that is literally false reports its error at the parent
// path: the error is returned before the path component is added.
func (v *schemaValidator) descend(instance, subschema, component any, depth int) ([]schemaIssue, error) {
	issues, err := v.iter(instance, subschema, depth+1)
	if err != nil {
		return nil, err
	}
	if allowed, isBool := subschema.(bool); isBool && !allowed {
		return issues, nil
	}
	for i := range issues {
		issues[i].path = append([]any{component}, issues[i].path...)
	}
	return issues, nil
}

func (v *schemaValidator) iter(instance, schema any, depth int) ([]schemaIssue, error) {
	if depth > maxSchemaDepth {
		return nil, errSchemaDepth
	}
	s, ok := schema.(*Object)
	if !ok {
		if allowed, isBool := schema.(bool); isBool && !allowed {
			return []schemaIssue{{message: "False schema does not allow " + pyRepr(instance), keyword: "-",
				instance: instance}}, nil
		}
		return nil, nil
	}
	keys := s.keys
	if v.dialect <= draft7 && s.Has("$ref") {
		keys = []string{"$ref"} // earlier drafts ignore the siblings of $ref
	}
	var out []schemaIssue
	for _, key := range keys {
		value := s.values[key]
		var issues []schemaIssue
		var err error
		switch key {
		case "type":
			issues = v.typeIssue(value, instance)
		case "properties":
			issues, err = v.properties(value, instance, depth)
		case "patternProperties":
			issues, err = v.patternProperties(value, instance, depth)
		case "additionalProperties":
			issues, err = v.additionalProperties(value, instance, s, depth)
		case "required":
			issues = required(value, instance)
		case "items":
			issues, err = v.items(value, instance, s, depth)
		case "prefixItems":
			if v.dialect >= draft2020 {
				issues, err = v.positional(value, instance, depth)
			}
		case "allOf":
			branches, _ := value.([]any)
			for _, branch := range branches {
				more, e := v.iter(instance, branch, depth+1)
				if e != nil {
					return nil, e
				}
				issues = append(issues, more...)
			}
		case "anyOf", "oneOf":
			issues, err = v.anyOf(value, instance, depth)
		case "$ref":
			issues, err = v.ref(value, instance, depth)
		default:
			continue
		}
		if err != nil {
			return nil, err
		}
		for i := range issues {
			if issues[i].keyword == "" {
				issues[i].keyword, issues[i].schema, issues[i].instance = key, s, instance
			}
		}
		out = append(out, issues...)
	}
	return out, nil
}

func (v *schemaValidator) isType(instance any, kind string) bool {
	switch kind {
	case "array":
		_, ok := instance.([]any)
		return ok
	case "boolean":
		_, ok := instance.(bool)
		return ok
	case "null":
		return instance == nil
	case "object":
		_, ok := instance.(*Object)
		return ok
	case "string":
		_, ok := instance.(string)
		return ok
	case "number":
		switch instance.(type) {
		case int64, *big.Int, float64:
			return true
		}
	case "integer":
		switch x := instance.(type) {
		case int64, *big.Int:
			return true
		case float64:
			// Draft 6 and later count an integer-valued float such as 2.0 as an integer.
			return v.dialect > draft4 && !math.IsInf(x, 0) && x == math.Trunc(x)
		}
	}
	return false
}

func (v *schemaValidator) typeIssue(value, instance any) []schemaIssue {
	var types []any
	switch t := value.(type) {
	case string:
		types = []any{t}
	case []any:
		types = t
	default:
		return nil
	}
	for _, kind := range types {
		if name, ok := kind.(string); ok && v.isType(instance, name) {
			return nil
		}
	}
	reprs := make([]string, len(types))
	for i, kind := range types {
		reprs[i] = pyRepr(kind)
	}
	return []schemaIssue{{message: pyRepr(instance) + " is not of type " + strings.Join(reprs, ", ")}}
}

func (v *schemaValidator) properties(value, instance any, depth int) ([]schemaIssue, error) {
	properties, ok := value.(*Object)
	target, isObject := instance.(*Object)
	if !ok || !isObject {
		return nil, nil
	}
	var out []schemaIssue
	for _, name := range properties.keys {
		item, present := target.Get(name)
		if !present {
			continue
		}
		issues, err := v.descend(item, properties.values[name], name, depth)
		if err != nil {
			return nil, err
		}
		out = append(out, issues...)
	}
	return out, nil
}

var (
	regexCache   = map[string]*regexp.Regexp{}
	regexCacheMu sync.Mutex
)

// compileRegex compiles a JSON Schema pattern. Go's RE2 syntax stands in for Python's re.
func compileRegex(pattern string) (*regexp.Regexp, error) {
	regexCacheMu.Lock()
	defer regexCacheMu.Unlock()
	if re, ok := regexCache[pattern]; ok {
		return re, nil
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	regexCache[pattern] = re
	return re, nil
}

func (v *schemaValidator) patternProperties(value, instance any, depth int) ([]schemaIssue, error) {
	patterns, ok := value.(*Object)
	target, isObject := instance.(*Object)
	if !ok || !isObject {
		return nil, nil
	}
	var out []schemaIssue
	for _, pattern := range patterns.keys {
		re, err := compileRegex(pattern)
		if err != nil {
			return nil, err
		}
		for _, key := range target.keys {
			if !re.MatchString(key) {
				continue
			}
			issues, err := v.descend(target.values[key], patterns.values[pattern], key, depth)
			if err != nil {
				return nil, err
			}
			out = append(out, issues...)
		}
	}
	return out, nil
}

func (v *schemaValidator) additionalProperties(value, instance any, schema *Object, depth int) ([]schemaIssue, error) {
	target, isObject := instance.(*Object)
	if !isObject {
		return nil, nil
	}
	declared, _ := schema.Value("properties").(*Object)
	patterns, _ := schema.Value("patternProperties").(*Object)
	var extras []string
	for _, key := range target.keys {
		if declared.Has(key) {
			continue
		}
		matched := false
		for _, pattern := range patterns.Keys() {
			re, err := compileRegex(pattern)
			if err != nil {
				return nil, err
			}
			if re.MatchString(key) {
				matched = true
				break
			}
		}
		if !matched {
			extras = append(extras, key)
		}
	}
	if _, isSchema := value.(*Object); isSchema {
		var out []schemaIssue
		for _, extra := range extras {
			issues, err := v.descend(target.values[extra], value, extra, depth)
			if err != nil {
				return nil, err
			}
			out = append(out, issues...)
		}
		return out, nil
	}
	if allowed, _ := value.(bool); allowed || len(extras) == 0 {
		return nil, nil
	}
	sort.Strings(extras)
	reprs := make([]string, len(extras))
	for i, extra := range extras {
		reprs[i] = pyStr(extra)
	}
	if schema.Has("patternProperties") {
		verb := "do"
		if len(extras) == 1 {
			verb = "does"
		}
		names := patterns.Keys()
		sort.Strings(names)
		for i, name := range names {
			names[i] = pyStr(name)
		}
		return []schemaIssue{{message: strings.Join(reprs, ", ") + " " + verb + " not match any of the regexes: " +
			strings.Join(names, ", ")}}, nil
	}
	verb := "were"
	if len(extras) == 1 {
		verb = "was"
	}
	return []schemaIssue{{message: "Additional properties are not allowed (" + strings.Join(reprs, ", ") + " " +
		verb + " unexpected)"}}, nil
}

func required(value, instance any) []schemaIssue {
	names, ok := value.([]any)
	target, isObject := instance.(*Object)
	if !ok || !isObject {
		return nil
	}
	var out []schemaIssue
	for _, name := range names {
		if key, ok := name.(string); ok && target.Has(key) {
			continue
		}
		out = append(out, schemaIssue{message: pyRepr(name) + " is a required property"})
	}
	return out
}

// positional validates array items against a list of schemas (prefixItems, or items as a list).
func (v *schemaValidator) positional(value, instance any, depth int) ([]schemaIssue, error) {
	schemas, ok := value.([]any)
	items, isArray := instance.([]any)
	if !ok || !isArray {
		return nil, nil
	}
	var out []schemaIssue
	for index := 0; index < len(items) && index < len(schemas); index++ {
		issues, err := v.descend(items[index], schemas[index], int64(index), depth)
		if err != nil {
			return nil, err
		}
		out = append(out, issues...)
	}
	return out, nil
}

func (v *schemaValidator) items(value, instance any, schema *Object, depth int) ([]schemaIssue, error) {
	items, isArray := instance.([]any)
	if !isArray {
		return nil, nil
	}
	if _, isList := value.([]any); isList {
		if v.dialect >= draft2020 {
			return nil, nil
		}
		return v.positional(value, instance, depth)
	}
	prefix := 0
	if v.dialect >= draft2020 {
		prefixItems, _ := schema.Value("prefixItems").([]any)
		prefix = len(prefixItems)
	}
	extra := len(items) - prefix
	if extra <= 0 {
		return nil, nil
	}
	if allowed, isBool := value.(bool); isBool && !allowed && v.dialect >= draft2020 {
		var rest any = items[prefix:]
		if extra == 1 {
			rest = items[prefix]
		}
		noun := "items"
		if prefix == 1 {
			noun = "item"
		}
		return []schemaIssue{{message: fmt.Sprintf("Expected at most %d %s but found %d extra: %s", prefix, noun,
			extra, pyRepr(rest))}}, nil
	}
	var out []schemaIssue
	for index := prefix; index < len(items); index++ {
		issues, err := v.descend(items[index], value, int64(index), depth)
		if err != nil {
			return nil, err
		}
		out = append(out, issues...)
	}
	return out, nil
}

func (v *schemaValidator) anyOf(value, instance any, depth int) ([]schemaIssue, error) {
	branches, _ := value.([]any)
	for _, branch := range branches {
		issues, err := v.iter(instance, branch, depth+1)
		if err != nil {
			return nil, err
		}
		if len(issues) == 0 {
			return nil, nil
		}
	}
	return []schemaIssue{{message: pyRepr(instance) + " is not valid under any of the given schemas"}}, nil
}

func (v *schemaValidator) ref(value, instance any, depth int) ([]schemaIssue, error) {
	reference, _ := value.(string)
	target, err := resolvePointer(v.root, reference)
	if err != nil {
		return nil, err
	}
	return v.iter(instance, target, depth+1)
}

// resolvePointer resolves a reference within the same document: "#" or "#/json/pointer".
func resolvePointer(root any, reference string) (any, error) {
	fragment, found := strings.CutPrefix(reference, "#")
	if !found {
		return nil, fmt.Errorf("dmcall: unresolvable JSON Schema reference %s (only references within the schema are supported)",
			pyStr(reference))
	}
	if unescaped, err := url.PathUnescape(fragment); err == nil {
		fragment = unescaped
	}
	if fragment == "" {
		return root, nil
	}
	if !strings.HasPrefix(fragment, "/") {
		return nil, fmt.Errorf("dmcall: unresolvable JSON Schema reference %s", pyStr(reference))
	}
	current := root
	for _, token := range strings.Split(fragment[1:], "/") {
		token = strings.ReplaceAll(strings.ReplaceAll(token, "~1", "/"), "~0", "~")
		switch node := current.(type) {
		case *Object:
			next, ok := node.Get(token)
			if !ok {
				return nil, fmt.Errorf("dmcall: unresolvable JSON Schema reference %s", pyStr(reference))
			}
			current = next
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || index < 0 || index >= len(node) {
				return nil, fmt.Errorf("dmcall: unresolvable JSON Schema reference %s", pyStr(reference))
			}
			current = node[index]
		default:
			return nil, fmt.Errorf("dmcall: unresolvable JSON Schema reference %s", pyStr(reference))
		}
	}
	return current, nil
}

// The metaschema check ------------------------------------------------------------------------------

var simpleTypes = map[string]bool{"array": true, "boolean": true, "integer": true, "null": true, "number": true,
	"object": true, "string": true}

var anchorPattern = regexp.MustCompile(`^[A-Za-z_][-A-Za-z0-9._]*$`)

// checkMetaSchema checks a parameter schema against its draft's metaschema, as jsonschema's
// check_schema does: keyword types, subschemas, type names, and regular expressions.
func checkMetaSchema(schema *Object) error {
	return metaCheck(schema, dialectOf(schema), 0)
}

func metaError(keyword, problem string) error {
	return fmt.Errorf("%s %s", pyStr(keyword), problem)
}

func metaCheck(schema any, d int, depth int) error {
	if depth > maxSchemaDepth {
		return errSchemaDepth
	}
	s, ok := schema.(*Object)
	if !ok {
		if _, isBool := schema.(bool); isBool && d > draft4 {
			return nil
		}
		return fmt.Errorf("%s is not a valid schema", pyRepr(schema))
	}
	sub := func(value any) error { return metaCheck(value, d, depth+1) }
	schemaMap := func(key string, value any) error {
		members, ok := value.(*Object)
		if !ok {
			return metaError(key, "must be an object")
		}
		for _, name := range members.keys {
			if err := sub(members.values[name]); err != nil {
				return err
			}
		}
		return nil
	}
	schemaArray := func(key string, value any) error {
		items, ok := value.([]any)
		if !ok || len(items) == 0 {
			return metaError(key, "must be a nonempty array of schemas")
		}
		for _, item := range items {
			if err := sub(item); err != nil {
				return err
			}
		}
		return nil
	}
	for _, key := range s.keys {
		value := s.values[key]
		var err error
		switch key {
		case "type":
			err = metaType(value)
		case "properties", "definitions":
			err = schemaMap(key, value)
		case "patternProperties":
			if err = schemaMap(key, value); err == nil {
				for _, pattern := range value.(*Object).keys {
					if _, e := compileRegex(pattern); e != nil {
						err = metaError(pattern, "is not a 'regex'")
						break
					}
				}
			}
		case "$defs", "dependentSchemas":
			if d >= draft2019 {
				err = schemaMap(key, value)
			}
		case "dependencies":
			members, ok := value.(*Object)
			if !ok {
				err = metaError(key, "must be an object")
				break
			}
			for _, name := range members.keys {
				if names, isList := members.values[name].([]any); isList {
					if err = metaStrings(key, names, d == draft4); err != nil {
						break
					}
				} else if err = sub(members.values[name]); err != nil {
					break
				}
			}
		case "dependentRequired":
			if d >= draft2019 {
				members, ok := value.(*Object)
				if !ok {
					err = metaError(key, "must be an object")
					break
				}
				for _, name := range members.keys {
					names, isList := members.values[name].([]any)
					if !isList {
						err = metaError(key, "must map names to arrays of strings")
						break
					}
					if err = metaStrings(key, names, false); err != nil {
						break
					}
				}
			}
		case "items":
			if _, isList := value.([]any); isList && d < draft2020 {
				err = schemaArray(key, value)
			} else {
				err = sub(value)
			}
		case "prefixItems":
			if d >= draft2020 {
				err = schemaArray(key, value)
			}
		case "additionalItems":
			if d < draft2020 {
				err = sub(value)
			}
		case "additionalProperties", "not":
			err = sub(value)
		case "propertyNames", "contains":
			if d >= draft6 {
				err = sub(value)
			}
		case "if", "then", "else":
			if d >= draft7 {
				err = sub(value)
			}
		case "unevaluatedItems", "unevaluatedProperties", "contentSchema":
			if d >= draft2019 {
				err = sub(value)
			}
		case "allOf", "anyOf", "oneOf":
			err = schemaArray(key, value)
		case "required":
			names, ok := value.([]any)
			if !ok {
				err = metaError(key, "must be an array of strings")
				break
			}
			err = metaStrings(key, names, d == draft4)
		case "enum":
			items, ok := value.([]any)
			switch {
			case !ok:
				err = metaError(key, "must be an array")
			case d == draft4 && (len(items) == 0 || !unique(items)):
				err = metaError(key, "must be a nonempty array of unique values")
			}
		case "multipleOf":
			if n, ok := number(value); !ok || n <= 0 {
				err = metaError(key, "must be a number greater than 0")
			}
		case "maximum", "minimum":
			if _, ok := number(value); !ok {
				err = metaError(key, "must be a number")
			}
		case "exclusiveMaximum", "exclusiveMinimum":
			if d == draft4 {
				if _, ok := value.(bool); !ok {
					err = metaError(key, "must be a boolean")
				}
			} else if _, ok := number(value); !ok {
				err = metaError(key, "must be a number")
			}
		case "maxLength", "minLength", "maxItems", "minItems", "maxProperties", "minProperties":
			err = metaCount(key, value, d)
		case "maxContains", "minContains":
			if d >= draft2019 {
				err = metaCount(key, value, d)
			}
		case "pattern":
			text, ok := value.(string)
			if !ok {
				err = metaError(key, "must be a string")
			} else if _, e := compileRegex(text); e != nil {
				err = metaError(text, "is not a 'regex'")
			}
		case "uniqueItems":
			if _, ok := value.(bool); !ok {
				err = metaError(key, "must be a boolean")
			}
		case "readOnly", "writeOnly":
			if _, ok := value.(bool); !ok && d >= draft7 {
				err = metaError(key, "must be a boolean")
			}
		case "deprecated", "$recursiveAnchor":
			if _, ok := value.(bool); !ok && d >= draft2019 {
				err = metaError(key, "must be a boolean")
			}
		case "format", "title", "description", "$ref", "$schema", "contentMediaType", "contentEncoding":
			if _, ok := value.(string); !ok {
				err = metaError(key, "must be a string")
			}
		case "$comment":
			if _, ok := value.(string); !ok && d >= draft7 {
				err = metaError(key, "must be a string")
			}
		case "$id":
			text, ok := value.(string)
			switch {
			case d < draft6:
			case !ok:
				err = metaError(key, "must be a string")
			case d >= draft2019 && strings.Contains(strings.TrimSuffix(text, "#"), "#"):
				err = metaError(key, "must not have a fragment")
			}
		case "id":
			if _, ok := value.(string); !ok && d == draft4 {
				err = metaError(key, "must be a string")
			}
		case "$anchor", "$dynamicAnchor":
			if text, ok := value.(string); d >= draft2019 && (!ok || !anchorPattern.MatchString(text)) {
				err = metaError(key, "must be an anchor name")
			}
		case "$dynamicRef", "$recursiveRef":
			if _, ok := value.(string); !ok && d >= draft2019 {
				err = metaError(key, "must be a string")
			}
		case "examples":
			if _, ok := value.([]any); !ok && d >= draft6 {
				err = metaError(key, "must be an array")
			}
		case "$vocabulary":
			if d >= draft2019 {
				members, ok := value.(*Object)
				if !ok {
					err = metaError(key, "must be an object")
					break
				}
				for _, name := range members.keys {
					if _, isBool := members.values[name].(bool); !isBool {
						err = metaError(key, "must map URIs to booleans")
						break
					}
				}
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func metaType(value any) error {
	switch t := value.(type) {
	case string:
		if simpleTypes[t] {
			return nil
		}
		return fmt.Errorf("%s is not valid under any of the given schemas", pyRepr(value))
	case []any:
		if len(t) == 0 || !unique(t) {
			return fmt.Errorf("%s is not valid under any of the given schemas", pyRepr(value))
		}
		for _, item := range t {
			if name, ok := item.(string); !ok || !simpleTypes[name] {
				return fmt.Errorf("%s is not valid under any of the given schemas", pyRepr(value))
			}
		}
		return nil
	}
	return fmt.Errorf("%s is not valid under any of the given schemas", pyRepr(value))
}

func metaStrings(key string, names []any, nonempty bool) error {
	if nonempty && len(names) == 0 {
		return metaError(key, "must not be empty")
	}
	for _, name := range names {
		if _, ok := name.(string); !ok {
			return metaError(key, "must hold strings")
		}
	}
	if !unique(names) {
		return metaError(key, "must hold unique strings")
	}
	return nil
}

func metaCount(key string, value any, d int) error {
	switch x := value.(type) {
	case int64:
		if x >= 0 {
			return nil
		}
	case *big.Int:
		if x.Sign() >= 0 {
			return nil
		}
	case float64:
		if d > draft4 && x >= 0 && !math.IsInf(x, 0) && x == math.Trunc(x) {
			return nil
		}
	}
	return metaError(key, "must be a nonnegative integer")
}

func number(value any) (float64, bool) {
	switch x := value.(type) {
	case int64:
		return float64(x), true
	case *big.Int:
		f, _ := new(big.Float).SetInt(x).Float64()
		return f, true
	case float64:
		return x, true
	}
	return 0, false
}

func unique(items []any) bool {
	for i := range items {
		for j := i + 1; j < len(items); j++ {
			if equalValues(items[i], items[j]) {
				return false
			}
		}
	}
	return true
}
