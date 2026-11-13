package dmcall

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

// JSON values in DM-Call keep two things that encoding/json drops: the order of object keys and
// whether a number was written as an integer or a float. Both reach the generator inside the
// history message, whose text must equal Python's json.dumps(value, ensure_ascii=False).
//
// A JSON value is one of: nil, bool, string, int64, *big.Int (an integer beyond int64), float64,
// []any or *Object. ParseJSON produces only these types. Functions that accept a value also take
// ordinary Go values (int, float32, map[string]any, slices, structs, json.RawMessage, Call, ...)
// and convert them; a Go map has no order, so its keys are sorted.

// Object is a JSON object that keeps its keys in insertion order, the way a Python dict does.
// The zero value is an empty object ready to use.
type Object struct {
	keys   []string
	values map[string]any
}

// NewObject returns an empty object.
func NewObject() *Object { return &Object{values: map[string]any{}} }

// obj builds an object from alternating keys and values.
func obj(pairs ...any) *Object {
	o := &Object{keys: make([]string, 0, len(pairs)/2), values: make(map[string]any, len(pairs)/2)}
	for i := 0; i+1 < len(pairs); i += 2 {
		o.Set(pairs[i].(string), pairs[i+1])
	}
	return o
}

// Len is the number of keys.
func (o *Object) Len() int {
	if o == nil {
		return 0
	}
	return len(o.keys)
}

// Keys returns the keys in order.
func (o *Object) Keys() []string {
	if o == nil {
		return nil
	}
	return append([]string(nil), o.keys...)
}

// Get returns the value of key and whether the key is present.
func (o *Object) Get(key string) (any, bool) {
	if o == nil || o.values == nil {
		return nil, false
	}
	value, ok := o.values[key]
	return value, ok
}

// Value returns the value of key, or nil when the key is absent.
func (o *Object) Value(key string) any {
	value, _ := o.Get(key)
	return value
}

// Has reports whether key is present.
func (o *Object) Has(key string) bool {
	_, ok := o.Get(key)
	return ok
}

// Set stores value under key. An existing key keeps its position, like assignment to a Python dict.
func (o *Object) Set(key string, value any) *Object {
	if o.values == nil {
		o.values = map[string]any{}
	}
	if _, ok := o.values[key]; !ok {
		o.keys = append(o.keys, key)
	}
	o.values[key] = value
	return o
}

// Delete removes key if present.
func (o *Object) Delete(key string) {
	if o == nil || o.values == nil {
		return
	}
	if _, ok := o.values[key]; !ok {
		return
	}
	delete(o.values, key)
	for i, k := range o.keys {
		if k == key {
			o.keys = append(o.keys[:i], o.keys[i+1:]...)
			break
		}
	}
}

// Clone returns a deep copy.
func (o *Object) Clone() *Object {
	if o == nil {
		return nil
	}
	c := &Object{keys: append([]string(nil), o.keys...), values: make(map[string]any, len(o.keys))}
	for _, k := range o.keys {
		c.values[k] = Clone(o.values[k])
	}
	return c
}

// Map converts the object to plain Go values: nested objects become map[string]any and arrays []any.
func (o *Object) Map() map[string]any {
	if o == nil {
		return nil
	}
	m := make(map[string]any, len(o.keys))
	for _, k := range o.keys {
		m[k] = plain(o.values[k])
	}
	return m
}

func plain(v any) any {
	switch x := v.(type) {
	case *Object:
		return x.Map()
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = plain(item)
		}
		return out
	}
	return v
}

// Decode unmarshals the object into target with encoding/json, for example into a struct.
func (o *Object) Decode(target any) error {
	data, err := o.MarshalJSON()
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

// MarshalJSON writes the object with its keys in order.
func (o *Object) MarshalJSON() ([]byte, error) {
	if o == nil {
		return []byte("null"), nil
	}
	text, err := encodeJSON(o, compactStyle)
	return []byte(text), err
}

// UnmarshalJSON reads a JSON object, keeping key order and number kinds.
func (o *Object) UnmarshalJSON(data []byte) error {
	value, err := ParseJSON(data)
	if err != nil {
		return err
	}
	parsed, ok := value.(*Object)
	if !ok {
		return errors.New("dmcall: JSON value is not an object")
	}
	*o = *parsed
	return nil
}

// String returns the object as Python's json.dumps writes it.
func (o *Object) String() string {
	text, err := JSONText(o)
	if err != nil {
		return "<invalid JSON object>"
	}
	return text
}

// Clone returns a deep copy of a JSON value. Objects and arrays are copied; other values are immutable.
func Clone(v any) any {
	switch x := v.(type) {
	case *Object:
		return x.Clone()
	case []any:
		if x == nil {
			return []any(nil)
		}
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = Clone(item)
		}
		return out
	case *big.Int:
		if x == nil {
			return nil
		}
		return new(big.Int).Set(x)
	}
	return v
}

// MustParseJSON parses JSON text and panics on an error. It is meant for literals such as tool schemas.
func MustParseJSON(text string) any {
	value, err := ParseJSON([]byte(text))
	if err != nil {
		panic(err)
	}
	return value
}

// ParseJSON parses JSON like Python's json.loads: object keys keep their order (a repeated key keeps
// its first position and its last value), integers stay int64 (or *big.Int) and other numbers are
// float64. NaN, Infinity and -Infinity are accepted, as Python accepts them.
func ParseJSON(data []byte) (any, error) {
	p := &jsonParser{data: data}
	p.skipSpace()
	value, err := p.value(0)
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if p.pos != len(p.data) {
		return nil, p.fail("extra data")
	}
	return value, nil
}

type jsonParser struct {
	data []byte
	pos  int
}

const maxJSONDepth = 10000

func (p *jsonParser) fail(message string) error {
	return fmt.Errorf("dmcall: invalid JSON at offset %d: %s", p.pos, message)
}

func (p *jsonParser) skipSpace() {
	for p.pos < len(p.data) {
		switch p.data[p.pos] {
		case ' ', '\t', '\n', '\r':
			p.pos++
		default:
			return
		}
	}
}

func (p *jsonParser) literal(word string, value any) (any, error) {
	if strings.HasPrefix(string(p.data[p.pos:min(len(p.data), p.pos+len(word))]), word) {
		p.pos += len(word)
		return value, nil
	}
	return nil, p.fail("expecting value")
}

func (p *jsonParser) value(depth int) (any, error) {
	if depth > maxJSONDepth {
		return nil, p.fail("nesting too deep")
	}
	if p.pos >= len(p.data) {
		return nil, p.fail("expecting value")
	}
	switch c := p.data[p.pos]; {
	case c == '{':
		return p.object(depth)
	case c == '[':
		return p.array(depth)
	case c == '"':
		return p.str()
	case c == 'n':
		return p.literal("null", nil)
	case c == 't':
		return p.literal("true", true)
	case c == 'f':
		return p.literal("false", false)
	case c == 'N':
		return p.literal("NaN", math.NaN())
	case c == 'I':
		return p.literal("Infinity", math.Inf(1))
	case c == '-' && p.pos+1 < len(p.data) && p.data[p.pos+1] == 'I':
		return p.literal("-Infinity", math.Inf(-1))
	case c == '-' || (c >= '0' && c <= '9'):
		return p.number()
	}
	return nil, p.fail("expecting value")
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func (p *jsonParser) number() (any, error) {
	start := p.pos
	if p.data[p.pos] == '-' {
		p.pos++
	}
	if p.pos >= len(p.data) || !isDigit(p.data[p.pos]) {
		p.pos = start
		return nil, p.fail("expecting value")
	}
	if p.data[p.pos] == '0' {
		p.pos++
	} else {
		for p.pos < len(p.data) && isDigit(p.data[p.pos]) {
			p.pos++
		}
	}
	isFloat := false
	if p.pos+1 < len(p.data) && p.data[p.pos] == '.' && isDigit(p.data[p.pos+1]) {
		isFloat = true
		p.pos++
		for p.pos < len(p.data) && isDigit(p.data[p.pos]) {
			p.pos++
		}
	}
	if p.pos < len(p.data) && (p.data[p.pos] == 'e' || p.data[p.pos] == 'E') {
		mark := p.pos
		p.pos++
		if p.pos < len(p.data) && (p.data[p.pos] == '+' || p.data[p.pos] == '-') {
			p.pos++
		}
		if p.pos < len(p.data) && isDigit(p.data[p.pos]) {
			isFloat = true
			for p.pos < len(p.data) && isDigit(p.data[p.pos]) {
				p.pos++
			}
		} else {
			p.pos = mark // not an exponent; the parser reports the extra data
		}
	}
	text := string(p.data[start:p.pos])
	if isFloat {
		f, err := strconv.ParseFloat(text, 64)
		if err != nil {
			var numErr *strconv.NumError
			if !errors.As(err, &numErr) || !errors.Is(numErr.Err, strconv.ErrRange) {
				return nil, p.fail("invalid number")
			}
		}
		return f, nil
	}
	if n, err := strconv.ParseInt(text, 10, 64); err == nil {
		return n, nil
	}
	n, ok := new(big.Int).SetString(text, 10)
	if !ok {
		return nil, p.fail("invalid number")
	}
	return n, nil
}

func hexValue(b []byte) (rune, bool) {
	var r rune
	for _, c := range b {
		r <<= 4
		switch {
		case c >= '0' && c <= '9':
			r |= rune(c - '0')
		case c >= 'a' && c <= 'f':
			r |= rune(c-'a') + 10
		case c >= 'A' && c <= 'F':
			r |= rune(c-'A') + 10
		default:
			return 0, false
		}
	}
	return r, true
}

func (p *jsonParser) str() (string, error) {
	p.pos++ // opening quote
	var b strings.Builder
	for {
		if p.pos >= len(p.data) {
			return "", p.fail("unterminated string")
		}
		c := p.data[p.pos]
		switch {
		case c == '"':
			p.pos++
			return b.String(), nil
		case c < 0x20:
			return "", p.fail("invalid control character in string")
		case c != '\\':
			b.WriteByte(c)
			p.pos++
			continue
		}
		p.pos++ // backslash
		if p.pos >= len(p.data) {
			return "", p.fail("unterminated string")
		}
		escape := p.data[p.pos]
		p.pos++
		switch escape {
		case '"', '\\', '/':
			b.WriteByte(escape)
		case 'b':
			b.WriteByte('\b')
		case 'f':
			b.WriteByte('\f')
		case 'n':
			b.WriteByte('\n')
		case 'r':
			b.WriteByte('\r')
		case 't':
			b.WriteByte('\t')
		case 'u':
			if p.pos+4 > len(p.data) {
				return "", p.fail("invalid \\uXXXX escape")
			}
			r, ok := hexValue(p.data[p.pos : p.pos+4])
			if !ok {
				return "", p.fail("invalid \\uXXXX escape")
			}
			p.pos += 4
			if utf16.IsSurrogate(r) && r < 0xdc00 && p.pos+6 <= len(p.data) &&
				p.data[p.pos] == '\\' && p.data[p.pos+1] == 'u' {
				if low, ok := hexValue(p.data[p.pos+2 : p.pos+6]); ok && low >= 0xdc00 && low <= 0xdfff {
					r = utf16.DecodeRune(r, low)
					p.pos += 6
				}
			}
			if utf16.IsSurrogate(r) {
				r = utf8.RuneError // a lone surrogate cannot be stored in a Go string
			}
			b.WriteRune(r)
		default:
			p.pos--
			return "", p.fail("invalid \\escape")
		}
	}
}

func (p *jsonParser) array(depth int) ([]any, error) {
	p.pos++
	out := []any{}
	p.skipSpace()
	if p.pos < len(p.data) && p.data[p.pos] == ']' {
		p.pos++
		return out, nil
	}
	for {
		p.skipSpace()
		value, err := p.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out = append(out, value)
		p.skipSpace()
		if p.pos >= len(p.data) {
			return nil, p.fail("unterminated array")
		}
		switch p.data[p.pos] {
		case ',':
			p.pos++
		case ']':
			p.pos++
			return out, nil
		default:
			return nil, p.fail("expecting ',' delimiter")
		}
	}
}

func (p *jsonParser) object(depth int) (*Object, error) {
	p.pos++
	out := NewObject()
	p.skipSpace()
	if p.pos < len(p.data) && p.data[p.pos] == '}' {
		p.pos++
		return out, nil
	}
	for {
		p.skipSpace()
		if p.pos >= len(p.data) || p.data[p.pos] != '"' {
			return nil, p.fail("expecting property name enclosed in double quotes")
		}
		key, err := p.str()
		if err != nil {
			return nil, err
		}
		p.skipSpace()
		if p.pos >= len(p.data) || p.data[p.pos] != ':' {
			return nil, p.fail("expecting ':' delimiter")
		}
		p.pos++
		p.skipSpace()
		value, err := p.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out.Set(key, value)
		p.skipSpace()
		if p.pos >= len(p.data) {
			return nil, p.fail("unterminated object")
		}
		switch p.data[p.pos] {
		case ',':
			p.pos++
		case '}':
			p.pos++
			return out, nil
		default:
			return nil, p.fail("expecting ',' delimiter")
		}
	}
}

// JSON text ---------------------------------------------------------------------------------------

type jsonStyle struct {
	itemSep, keySep string
	ascii           bool // escape every character outside printable ASCII (Python's ensure_ascii=True)
	sortKeys        bool
	allowNaN        bool
}

var (
	// json.dumps(value, ensure_ascii=False): the history message.
	pythonStyle = jsonStyle{itemSep: ", ", keySep: ": ", allowNaN: true}
	// json.dumps(value): schema error paths.
	asciiStyle = jsonStyle{itemSep: ", ", keySep: ": ", ascii: true, allowNaN: true}
	// json.dumps(value, sort_keys=True): comparing a value with a declared default.
	canonicalStyle = jsonStyle{itemSep: ", ", keySep: ": ", ascii: true, sortKeys: true, allowNaN: true}
	// Request bodies.
	compactStyle = jsonStyle{itemSep: ",", keySep: ":"}
)

// JSONText returns the text Python's json.dumps(value, ensure_ascii=False) writes for v: ", " and
// ": " separators, keys in order, floats in Python repr form (1.0, 1e+16, 1e-05), non-ASCII kept
// and control characters escaped.
func JSONText(v any) (string, error) { return encodeJSON(v, pythonStyle) }

func encodeJSON(v any, style jsonStyle) (string, error) {
	var b strings.Builder
	if err := writeJSON(&b, v, style, 0); err != nil {
		return "", err
	}
	return b.String(), nil
}

func writeJSON(b *strings.Builder, v any, style jsonStyle, depth int) error {
	if depth > maxJSONDepth {
		return errors.New("dmcall: JSON value nested too deeply")
	}
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case string:
		writeJSONString(b, x, style.ascii)
	case int64:
		b.WriteString(strconv.FormatInt(x, 10))
	case int:
		b.WriteString(strconv.Itoa(x))
	case *big.Int:
		if x == nil {
			b.WriteString("null")
		} else {
			b.WriteString(x.String())
		}
	case float64:
		switch {
		case math.IsNaN(x) || math.IsInf(x, 0):
			if !style.allowNaN {
				return errors.New("dmcall: out of range float values are not JSON compliant")
			}
			switch {
			case math.IsNaN(x):
				b.WriteString("NaN")
			case x > 0:
				b.WriteString("Infinity")
			default:
				b.WriteString("-Infinity")
			}
		default:
			b.WriteString(pyFloat(x))
		}
	case []any:
		if x == nil {
			b.WriteString("null")
			return nil
		}
		b.WriteByte('[')
		for i, item := range x {
			if i > 0 {
				b.WriteString(style.itemSep)
			}
			if err := writeJSON(b, item, style, depth+1); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case *Object:
		if x == nil {
			b.WriteString("null")
			return nil
		}
		keys := x.keys
		if style.sortKeys {
			keys = append([]string(nil), keys...)
			sort.Strings(keys)
		}
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(style.itemSep)
			}
			writeJSONString(b, k, style.ascii)
			b.WriteString(style.keySep)
			if err := writeJSON(b, x.values[k], style, depth+1); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		converted, err := toValue(v)
		if err != nil {
			return err
		}
		if reflect.TypeOf(converted) == reflect.TypeOf(v) {
			return fmt.Errorf("dmcall: cannot write %T as JSON", v)
		}
		return writeJSON(b, converted, style, depth+1)
	}
	return nil
}

const lowerHex = "0123456789abcdef"

func writeJSONString(b *strings.Builder, s string, ascii bool) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20 || (ascii && r >= 0x7f && r < 0x10000):
				writeUnicodeEscape(b, r)
			case ascii && r >= 0x10000:
				high, low := utf16.EncodeRune(r)
				writeUnicodeEscape(b, high)
				writeUnicodeEscape(b, low)
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

func writeUnicodeEscape(b *strings.Builder, r rune) {
	b.WriteString(`\u`)
	for shift := 12; shift >= 0; shift -= 4 {
		b.WriteByte(lowerHex[(r>>uint(shift))&0xf])
	}
}

// pyFloat formats a finite float like Python's repr: the shortest digits that round-trip, in
// positional form when the decimal exponent is in [-4, 16), otherwise as d.ddde+XX.
func pyFloat(f float64) string {
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	s := strconv.FormatFloat(f, 'e', -1, 64)
	negative := s[0] == '-'
	if negative {
		s = s[1:]
	}
	at := strings.IndexByte(s, 'e')
	digits := strings.Replace(s[:at], ".", "", 1)
	exponent, _ := strconv.Atoi(s[at+1:])
	var out string
	if exponent < -4 || exponent >= 16 {
		out = digits[:1]
		if len(digits) > 1 {
			out += "." + digits[1:]
		}
		sign := "+"
		if exponent < 0 {
			sign, exponent = "-", -exponent
		}
		e := strconv.Itoa(exponent)
		if len(e) < 2 {
			e = "0" + e
		}
		out += "e" + sign + e
	} else {
		point := exponent + 1
		switch {
		case point <= 0:
			out = "0." + strings.Repeat("0", -point) + digits
		case point >= len(digits):
			out = digits + strings.Repeat("0", point-len(digits)) + ".0"
		default:
			out = digits[:point] + "." + digits[point:]
		}
	}
	if negative {
		out = "-" + out
	}
	return out
}

// Conversion from Go values -------------------------------------------------------------------------

// jsonValuer is implemented by the package's own types that have a fixed JSON form.
type jsonValuer interface{ jsonValue() any }

// toValue converts a Go value to the JSON value model.
func toValue(v any) (any, error) {
	switch x := v.(type) {
	case nil:
		return nil, nil
	case bool, string, int64:
		return x, nil
	case float64:
		return x, nil
	case int:
		return int64(x), nil
	case int8:
		return int64(x), nil
	case int16:
		return int64(x), nil
	case int32:
		return int64(x), nil
	case uint8:
		return int64(x), nil
	case uint16:
		return int64(x), nil
	case uint32:
		return int64(x), nil
	case uint:
		return uintValue(uint64(x)), nil
	case uint64:
		return uintValue(x), nil
	case float32:
		return float64(x), nil
	case *big.Int:
		if x == nil {
			return nil, nil
		}
		if x.IsInt64() {
			return x.Int64(), nil
		}
		return x, nil
	case json.Number:
		return ParseJSON([]byte(x))
	case json.RawMessage:
		if x == nil {
			return nil, nil
		}
		return ParseJSON(x)
	case *Object:
		if x == nil {
			return nil, nil
		}
		out := &Object{keys: append([]string(nil), x.keys...), values: make(map[string]any, len(x.keys))}
		for _, k := range x.keys {
			value, err := toValue(x.values[k])
			if err != nil {
				return nil, err
			}
			out.values[k] = value
		}
		return out, nil
	case Object:
		return toValue(&x)
	case []any:
		if x == nil {
			return nil, nil
		}
		out := make([]any, len(x))
		for i, item := range x {
			value, err := toValue(item)
			if err != nil {
				return nil, err
			}
			out[i] = value
		}
		return out, nil
	case map[string]any:
		if x == nil {
			return nil, nil
		}
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := &Object{keys: keys, values: make(map[string]any, len(x))}
		for _, k := range keys {
			value, err := toValue(x[k])
			if err != nil {
				return nil, err
			}
			out.values[k] = value
		}
		return out, nil
	case jsonValuer:
		return toValue(x.jsonValue())
	}
	return reflectValue(reflect.ValueOf(v))
}

func uintValue(x uint64) any {
	if x <= math.MaxInt64 {
		return int64(x)
	}
	return new(big.Int).SetUint64(x)
}

func reflectValue(rv reflect.Value) (any, error) {
	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			return nil, nil
		}
		return toValue(rv.Elem().Interface())
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			break
		}
		if rv.IsNil() {
			return nil, nil
		}
		keys := make([]string, 0, rv.Len())
		for _, k := range rv.MapKeys() {
			keys = append(keys, k.String())
		}
		sort.Strings(keys)
		out := &Object{keys: keys, values: make(map[string]any, len(keys))}
		for _, k := range keys {
			value, err := toValue(rv.MapIndex(reflect.ValueOf(k).Convert(rv.Type().Key())).Interface())
			if err != nil {
				return nil, err
			}
			out.values[k] = value
		}
		return out, nil
	case reflect.Slice:
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			break // []byte is encoded the encoding/json way
		}
		if rv.IsNil() {
			return nil, nil
		}
		fallthrough
	case reflect.Array:
		out := make([]any, rv.Len())
		for i := range out {
			value, err := toValue(rv.Index(i).Interface())
			if err != nil {
				return nil, err
			}
			out[i] = value
		}
		return out, nil
	case reflect.String:
		return rv.String(), nil
	case reflect.Bool:
		return rv.Bool(), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return uintValue(rv.Uint()), nil
	case reflect.Float32, reflect.Float64:
		return rv.Float(), nil
	case reflect.Invalid:
		return nil, nil
	}
	data, err := json.Marshal(rv.Interface())
	if err != nil {
		return nil, fmt.Errorf("dmcall: cannot convert %s to JSON: %w", rv.Type(), err)
	}
	return ParseJSON(data)
}

// Comparison --------------------------------------------------------------------------------------

// Equal reports whether two JSON values are equal the way Python compares parsed JSON: object key
// order does not matter and an integer equals a float of the same value (1 == 1.0).
func Equal(a, b any) bool {
	x, err := toValue(a)
	if err != nil {
		return false
	}
	y, err := toValue(b)
	if err != nil {
		return false
	}
	return equalValues(x, y)
}

func equalValues(a, b any) bool {
	switch x := a.(type) {
	case nil:
		return b == nil
	case bool:
		y, ok := b.(bool)
		return ok && x == y
	case string:
		y, ok := b.(string)
		return ok && x == y
	case int64, *big.Int, float64:
		return numbersEqual(a, b)
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !equalValues(x[i], y[i]) {
				return false
			}
		}
		return true
	case *Object:
		y, ok := b.(*Object)
		if !ok || x.Len() != y.Len() {
			return false
		}
		for _, k := range x.keys {
			other, ok := y.Get(k)
			if !ok || !equalValues(x.values[k], other) {
				return false
			}
		}
		return true
	}
	return false
}

func bigOf(v any) (*big.Int, bool) {
	switch x := v.(type) {
	case int64:
		return big.NewInt(x), true
	case *big.Int:
		return x, true
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) || x != math.Trunc(x) {
			return nil, false
		}
		n, _ := new(big.Float).SetFloat64(x).Int(nil)
		return n, true
	}
	return nil, false
}

func numbersEqual(a, b any) bool {
	fa, aFloat := a.(float64)
	fb, bFloat := b.(float64)
	if aFloat && bFloat {
		return fa == fb
	}
	switch b.(type) {
	case int64, *big.Int, float64:
	default:
		return false
	}
	x, ok := bigOf(a)
	if !ok {
		return false
	}
	y, ok := bigOf(b)
	if !ok {
		return false
	}
	return x.Cmp(y) == 0
}

// Python repr ---------------------------------------------------------------------------------------

// pyRepr formats a JSON value like Python's repr: 'text', True, None, 1.5, ['a'], {'a': 1}.
func pyRepr(v any) string {
	var b strings.Builder
	writeRepr(&b, v)
	return b.String()
}

func writeRepr(b *strings.Builder, v any) {
	switch x := v.(type) {
	case nil:
		b.WriteString("None")
	case bool:
		if x {
			b.WriteString("True")
		} else {
			b.WriteString("False")
		}
	case string:
		b.WriteString(pyStr(x))
	case int64:
		b.WriteString(strconv.FormatInt(x, 10))
	case int:
		b.WriteString(strconv.Itoa(x))
	case *big.Int:
		b.WriteString(x.String())
	case float64:
		switch {
		case math.IsNaN(x):
			b.WriteString("nan")
		case math.IsInf(x, 1):
			b.WriteString("inf")
		case math.IsInf(x, -1):
			b.WriteString("-inf")
		default:
			b.WriteString(pyFloat(x))
		}
	case []any:
		b.WriteByte('[')
		for i, item := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			writeRepr(b, item)
		}
		b.WriteByte(']')
	case *Object:
		b.WriteByte('{')
		for i, k := range x.keys {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(pyStr(k))
			b.WriteString(": ")
			writeRepr(b, x.values[k])
		}
		b.WriteByte('}')
	default:
		if converted, err := toValue(v); err == nil && reflect.TypeOf(converted) != reflect.TypeOf(v) {
			writeRepr(b, converted)
		} else {
			fmt.Fprintf(b, "%v", v)
		}
	}
}

// pyStr is Python's repr of a str.
func pyStr(s string) string {
	quote := '\''
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.WriteRune(quote)
	for _, r := range s {
		switch {
		case r == quote || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x7f:
			b.WriteRune(r)
		case unicode.IsPrint(r):
			b.WriteRune(r)
		case r <= 0xff:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r <= 0xffff:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteRune(quote)
	return b.String()
}

// truthy is Python's truth value of a JSON value.
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case int64:
		return x != 0
	case *big.Int:
		return x.Sign() != 0
	case float64:
		return x != 0
	case []any:
		return len(x) > 0
	case *Object:
		return x.Len() > 0
	}
	return true
}
