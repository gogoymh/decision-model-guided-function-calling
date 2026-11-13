/**
 * JSON that behaves like Python's `json` module.
 *
 * DM-Call sends some JSON as text inside a message (the revision history), so its characters
 * must equal Python's `json.dumps(value, ensure_ascii=False)`. `JSON.parse` loses two things that
 * matter for that: whether a number was written as `4.0` or `4`, and the insertion order of
 * integer-like keys such as `"10"` and `"2"`. This module keeps both:
 *
 * - objects are `Map<string, Json>` (insertion order, any key),
 * - integers are `bigint` (Python `int`, any size),
 * - floats are {@link Float} instances (Python `float`, including NaN and infinities),
 * - values that are not JSON (functions, symbols) are {@link Unserializable}.
 *
 * Plain JavaScript values convert with {@link fromPlain} and back with {@link toPlain}.
 */

/** A Python `float`. Plain numbers never appear inside {@link Json}; floats are always boxed. */
export class Float {
  constructor(readonly value: number) {}
  valueOf(): number {
    return this.value;
  }
  toString(): string {
    return floatRepr(this.value);
  }
}

/** A value that has no JSON form (a function or a symbol). Encoding it fails, as in Python. */
export class Unserializable {
  constructor(readonly value: unknown) {}
}

export type JsonObject = Map<string, Json>;
export type Json = null | boolean | string | bigint | Float | Json[] | JsonObject | Unserializable;

export function isObject(value: unknown): value is JsonObject {
  return value instanceof Map;
}

export function isInt(value: unknown): value is bigint {
  return typeof value === "bigint";
}

export function isFloat(value: unknown): value is Float {
  return value instanceof Float;
}

/** A Python number (`int` or `float`, never `bool`). */
export function isNumber(value: unknown): value is bigint | Float {
  return typeof value === "bigint" || value instanceof Float;
}

export function numberValue(value: bigint | Float): number {
  return typeof value === "bigint" ? Number(value) : value.value;
}

// Parsing ------------------------------------------------------------------------------------------

/** Raised for text that Python's `json.loads` rejects. */
export class JsonParseError extends SyntaxError {
  constructor(message: string, readonly position: number) {
    super(`${message} (char ${position})`);
    this.name = "JsonParseError";
  }
}

const NUMBER = /(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?/y;
const HEX4 = /^[0-9a-fA-F]{4}$/;

class Parser {
  index = 0;

  constructor(private readonly text: string) {}

  fail(message: string, at = this.index): never {
    throw new JsonParseError(message, at);
  }

  whitespace(): void {
    const text = this.text;
    while (this.index < text.length) {
      const c = text.charCodeAt(this.index);
      if (c !== 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d) break;
      this.index++;
    }
  }

  literal(word: string, value: Json): Json {
    if (this.text.startsWith(word, this.index)) {
      this.index += word.length;
      return value;
    }
    return this.fail("Expecting value");
  }

  value(): Json {
    const text = this.text;
    if (this.index >= text.length) this.fail("Expecting value");
    const c = text[this.index];
    switch (c) {
      case '"':
        return this.string();
      case "{":
        return this.object();
      case "[":
        return this.array();
      case "n":
        return this.literal("null", null);
      case "t":
        return this.literal("true", true);
      case "f":
        return this.literal("false", false);
      case "N":
        return this.literal("NaN", new Float(NaN));
      case "I":
        return this.literal("Infinity", new Float(Infinity));
      case "-":
        if (text.startsWith("-Infinity", this.index)) {
          this.index += 9;
          return new Float(-Infinity);
        }
        return this.number();
      default:
        return this.number();
    }
  }

  number(): Json {
    NUMBER.lastIndex = this.index;
    const match = NUMBER.exec(this.text);
    if (!match) return this.fail("Expecting value");
    this.index = NUMBER.lastIndex;
    if (match[2] === undefined && match[3] === undefined) return BigInt(match[1]!);
    return new Float(Number(match[0]));
  }

  string(): string {
    const text = this.text;
    const start = this.index;
    this.index++; // opening quote
    let out = "";
    let chunk = this.index;
    for (;;) {
      if (this.index >= text.length) this.fail("Unterminated string starting at", start);
      const code = text.charCodeAt(this.index);
      if (code === 0x22) {
        out += text.slice(chunk, this.index);
        this.index++;
        return out;
      }
      if (code === 0x5c) {
        out += text.slice(chunk, this.index);
        const escape = text[this.index + 1];
        switch (escape) {
          case '"':
            out += '"';
            break;
          case "\\":
            out += "\\";
            break;
          case "/":
            out += "/";
            break;
          case "b":
            out += "\b";
            break;
          case "f":
            out += "\f";
            break;
          case "n":
            out += "\n";
            break;
          case "r":
            out += "\r";
            break;
          case "t":
            out += "\t";
            break;
          case "u": {
            const hex = text.slice(this.index + 2, this.index + 6);
            if (!HEX4.test(hex)) this.fail("Invalid \\uXXXX escape", this.index + 1);
            out += String.fromCharCode(parseInt(hex, 16));
            this.index += 4;
            break;
          }
          default:
            if (escape === undefined) this.fail("Unterminated string starting at", start);
            this.fail("Invalid \\escape", this.index);
        }
        this.index += 2;
        chunk = this.index;
        continue;
      }
      if (code < 0x20) this.fail("Invalid control character at");
      this.index++;
    }
  }

  object(): JsonObject {
    const result: JsonObject = new Map();
    this.index++; // {
    this.whitespace();
    if (this.text[this.index] === "}") {
      this.index++;
      return result;
    }
    for (;;) {
      if (this.text[this.index] !== '"') this.fail("Expecting property name enclosed in double quotes");
      const key = this.string();
      this.whitespace();
      if (this.text[this.index] !== ":") this.fail("Expecting ':' delimiter");
      this.index++;
      this.whitespace();
      result.set(key, this.value()); // a repeated key keeps its first position and its last value
      this.whitespace();
      const next = this.text[this.index];
      if (next === "}") {
        this.index++;
        return result;
      }
      if (next !== ",") this.fail("Expecting ',' delimiter");
      this.index++;
      this.whitespace();
      if (this.text[this.index] === "}") this.fail("Illegal trailing comma before end of object");
    }
  }

  array(): Json[] {
    const result: Json[] = [];
    this.index++; // [
    this.whitespace();
    if (this.text[this.index] === "]") {
      this.index++;
      return result;
    }
    for (;;) {
      result.push(this.value());
      this.whitespace();
      const next = this.text[this.index];
      if (next === "]") {
        this.index++;
        return result;
      }
      if (next !== ",") this.fail("Expecting ',' delimiter");
      this.index++;
      this.whitespace();
      if (this.text[this.index] === "]") this.fail("Illegal trailing comma before end of array");
    }
  }
}

/** Python's `json.loads(text)`: keeps number kinds and key order; accepts NaN and Infinity. */
export function parse(text: string): Json {
  if (text.startsWith("\ufeff")) throw new JsonParseError("Unexpected UTF-8 BOM", 0);
  const parser = new Parser(text);
  parser.whitespace();
  const value = parser.value();
  parser.whitespace();
  if (parser.index !== text.length) parser.fail("Extra data");
  return value;
}

// Encoding -----------------------------------------------------------------------------------------

/** Python's `repr(float)`: the shortest text that round-trips, `1.0`, `1e+16`, `1e-05`. */
export function floatRepr(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (x === Infinity) return "inf";
  if (x === -Infinity) return "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const sign = x < 0 ? "-" : "";
  // toExponential() without digits gives the shortest round-trip digits, as Python's repr does.
  const [mantissa, exponentText] = Math.abs(x).toExponential().split("e") as [string, string];
  const digits = mantissa.replace(".", "");
  const exponent = Number(exponentText);
  const point = exponent + 1; // digits are 0.d1d2... x 10^point
  if (point > 16 || point <= -4) {
    const head = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
    const magnitude = String(Math.abs(exponent)).padStart(2, "0");
    return `${sign}${head}e${exponent < 0 ? "-" : "+"}${magnitude}`;
  }
  if (point <= 0) return `${sign}0.${"0".repeat(-point)}${digits}`;
  if (point >= digits.length) return `${sign}${digits}${"0".repeat(point - digits.length)}.0`;
  return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
}

function hex(code: number, width: number): string {
  return code.toString(16).padStart(width, "0");
}

function encodeString(value: string, ensureAscii: boolean): string {
  let out = '"';
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (c === 0x22) out += '\\"';
    else if (c === 0x5c) out += "\\\\";
    else if (c === 0x0a) out += "\\n";
    else if (c === 0x0d) out += "\\r";
    else if (c === 0x09) out += "\\t";
    else if (c === 0x08) out += "\\b";
    else if (c === 0x0c) out += "\\f";
    else if (c < 0x20 || (ensureAscii && c > 0x7e)) out += `\\u${hex(c, 4)}`;
    else out += value[i];
  }
  return out + '"';
}

export interface StringifyOptions {
  /** Item and key separators. Python's default is `[", ", ": "]`. */
  separators?: readonly [string, string];
  /** Escape every non-ASCII character, as Python does by default. */
  ensureAscii?: boolean;
  /** Sort object keys by code point. */
  sortKeys?: boolean;
  /** Write NaN and infinities as `NaN` and `Infinity`; otherwise they throw. */
  allowNan?: boolean;
}

export function compareCodePoints(a: string, b: string): number {
  const left = Array.from(a);
  const right = Array.from(b);
  const length = Math.min(left.length, right.length);
  for (let i = 0; i < length; i++) {
    const difference = left[i]!.codePointAt(0)! - right[i]!.codePointAt(0)!;
    if (difference !== 0) return difference;
  }
  return left.length - right.length;
}

function encode(value: Json, options: Required<StringifyOptions>): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "string") return encodeString(value, options.ensureAscii);
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Float) {
    const x = value.value;
    if (Number.isFinite(x)) return floatRepr(x);
    if (!options.allowNan) throw new RangeError("Out of range float values are not JSON compliant");
    return Number.isNaN(x) ? "NaN" : x > 0 ? "Infinity" : "-Infinity";
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => encode(item, options)).join(options.separators[0])}]`;
  }
  if (value instanceof Map) {
    const entries = [...value];
    if (options.sortKeys) entries.sort(([a], [b]) => compareCodePoints(a, b));
    const parts = entries.map(([key, item]) => encodeString(key, options.ensureAscii) + options.separators[1]
      + encode(item, options));
    return `{${parts.join(options.separators[0])}}`;
  }
  throw new TypeError("Object is not JSON serializable");
}

/**
 * Python's `json.dumps`. Defaults match Python: `", "` and `": "` separators, ASCII escapes,
 * insertion order and NaN allowed. Plain JavaScript values are converted with {@link fromPlain}.
 */
export function stringify(value: unknown, options: StringifyOptions = {}): string {
  return encode(fromPlain(value), {
    separators: options.separators ?? [", ", ": "],
    ensureAscii: options.ensureAscii ?? true,
    sortKeys: options.sortKeys ?? false,
    allowNan: options.allowNan ?? true,
  });
}

/** `json.dumps(value, ensure_ascii=False)`: the text of DM-Call's history message. */
export function pythonText(value: Json): string {
  return encode(value, { separators: [", ", ": "], ensureAscii: false, sortKeys: false, allowNan: true });
}

/** Compact UTF-8 JSON for an HTTP body, as httpx 0.28 writes it. */
export function compactText(value: Json): string {
  return encode(value, { separators: [",", ":"], ensureAscii: false, sortKeys: false, allowNan: false });
}

/** Whether `json.dumps(value, allow_nan=False)` succeeds: only finite JSON values. */
export function isFiniteJson(value: Json): boolean {
  if (value instanceof Float) return Number.isFinite(value.value);
  if (value instanceof Unserializable) return false;
  if (Array.isArray(value)) return value.every(isFiniteJson);
  if (value instanceof Map) return [...value.values()].every(isFiniteJson);
  return true;
}

// Python repr --------------------------------------------------------------------------------------

const NOT_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

/** Python's `repr(str)`. */
export function stringRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of value) {
    const code = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += `\\${ch}`;
    else if (ch === "\t") out += "\\t";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (code < 0x20 || code === 0x7f) out += `\\x${hex(code, 2)}`;
    else if (code < 0x7f) out += ch;
    else if (!NOT_PRINTABLE.test(ch)) out += ch;
    else if (code <= 0xff) out += `\\x${hex(code, 2)}`;
    else if (code <= 0xffff) out += `\\u${hex(code, 4)}`;
    else out += `\\U${hex(code, 8)}`;
  }
  return out + quote;
}

/** Python's `repr()` of a JSON value: `'Seoul'`, `True`, `None`, `['a']`, `{'a': 1}`, `1.0`. */
export function pythonRepr(value: Json): string {
  if (value === null) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "string") return stringRepr(value);
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Float) return floatRepr(value.value);
  if (Array.isArray(value)) return `[${value.map(pythonRepr).join(", ")}]`;
  if (value instanceof Map) {
    return `{${[...value].map(([key, item]) => `${stringRepr(key)}: ${pythonRepr(item)}`).join(", ")}}`;
  }
  return "<object>";
}

/** Python's `str()`: strings as they are, everything else as `repr()`. */
export function pythonStr(value: Json): string {
  return typeof value === "string" ? value : pythonRepr(value);
}

// Python semantics ---------------------------------------------------------------------------------

/** Python truthiness. */
export function truthy(value: Json | undefined): boolean {
  if (value === null || value === undefined || value === false) return false;
  if (value === true) return true;
  if (typeof value === "string") return value.length > 0;
  if (typeof value === "bigint") return value !== 0n;
  if (value instanceof Float) return value.value !== 0;
  if (Array.isArray(value)) return value.length > 0;
  if (value instanceof Map) return value.size > 0;
  return true;
}

function numbersEqual(a: bigint | Float, b: bigint | Float): boolean {
  if (typeof a === "bigint" && typeof b === "bigint") return a === b;
  if (a instanceof Float && b instanceof Float) return a.value === b.value;
  const int = (typeof a === "bigint" ? a : b) as bigint;
  const float = (a instanceof Float ? a : b) as Float;
  return Number.isInteger(float.value) && BigInt(float.value) === int;
}

/**
 * Python's `==` for JSON values: `4 == 4.0`, objects compare without regard to key order.
 * With `ordered`, objects must also list their keys in the same order.
 */
export function equal(a: Json, b: Json, ordered = false): boolean {
  if (isNumber(a) || isNumber(b)) return isNumber(a) && isNumber(b) && numbersEqual(a, b);
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length
      && a.every((item, index) => equal(item, b[index]!, ordered));
  }
  if (a instanceof Map || b instanceof Map) {
    if (!(a instanceof Map && b instanceof Map) || a.size !== b.size) return false;
    if (ordered) {
      const left = [...a.keys()];
      const right = [...b.keys()];
      if (left.some((key, index) => key !== right[index])) return false;
    }
    for (const [key, item] of a) {
      if (!b.has(key) || !equal(item, b.get(key)!, ordered)) return false;
    }
    return true;
  }
  if (a instanceof Unserializable || b instanceof Unserializable) {
    return a instanceof Unserializable && b instanceof Unserializable && a.value === b.value;
  }
  return a === b;
}

/** A deep copy of a JSON value (or of plain data such as a feedback record). */
export function clone<T>(value: T): T {
  if (Array.isArray(value)) return value.map(clone) as T;
  if (value instanceof Map) return new Map([...value].map(([key, item]) => [key, clone(item)])) as T;
  if (value !== null && typeof value === "object" && !(value instanceof Float) && !(value instanceof Unserializable)) {
    const proto = Object.getPrototypeOf(value);
    if (proto === Object.prototype || proto === null) {
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(value)) out[key] = clone((value as Record<string, unknown>)[key]);
      return out as T;
    }
  }
  return value;
}

// Conversion ---------------------------------------------------------------------------------------

/**
 * Plain JavaScript data as {@link Json}. A whole number becomes an `int` (with `numbers: "float"`,
 * every number becomes a `float`); objects keep their key order as JavaScript reports it; `undefined`
 * properties are dropped; values with `toJSON()` use it. Values already in {@link Json} form pass through.
 */
export function fromPlain(value: unknown, numbers: "auto" | "float" = "auto"): Json {
  switch (typeof value) {
    case "undefined":
      return null;
    case "string":
    case "boolean":
    case "bigint":
      return value;
    case "number":
      if (numbers === "auto" && Number.isInteger(value) && !Object.is(value, -0)) return BigInt(value);
      return new Float(value);
    case "function":
    case "symbol":
      return new Unserializable(value);
  }
  if (value === null) return null;
  if (value instanceof Float || value instanceof Unserializable) return value;
  if (Array.isArray(value)) return value.map((item) => fromPlain(item, numbers));
  if (value instanceof Map) {
    return new Map([...value].map(([key, item]) => [String(key), fromPlain(item, numbers)]));
  }
  const record = value as Record<string, unknown>;
  if (typeof record.toJSON === "function") return fromPlain(record.toJSON(), numbers);
  const result: JsonObject = new Map();
  for (const key of Object.keys(record)) {
    const item = record[key];
    if (item !== undefined) result.set(key, fromPlain(item, numbers));
  }
  return result;
}

/** {@link Json} as plain JavaScript data: objects, arrays and numbers. */
export function toPlain(value: Json): unknown {
  if (typeof value === "bigint") return Number(value);
  if (value instanceof Float) return value.value;
  if (value instanceof Unserializable) return value.value;
  if (Array.isArray(value)) return value.map(toPlain);
  if (value instanceof Map) {
    const out: Record<string, unknown> = {};
    for (const [key, item] of value) {
      Object.defineProperty(out, key, { value: toPlain(item), enumerable: true, writable: true, configurable: true });
    }
    return out;
  }
  return value;
}

/** `value.get(key)` for a JSON object, `undefined` for anything else. */
export function field(value: Json | undefined, key: string): Json | undefined {
  return value instanceof Map ? value.get(key) : undefined;
}
