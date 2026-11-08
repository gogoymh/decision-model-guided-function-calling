/** The Python-compatible JSON layer: parsing, number kinds, key order, repr and conversion. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  Float, JsonParseError, Unserializable, equal, floatRepr, fromPlain, isFiniteJson, parse, pythonRepr, pythonText, stringRepr,
  stringify, toPlain,
} from "../src/json.js";

test("floats print as Python's repr", () => {
  // Expected strings from CPython 3.13 repr().
  const cases: [number, string][] = [
    [5e-324, "5e-324"], [2.2250738585072014e-308, "2.2250738585072014e-308"],
    [1.7976931348623157e308, "1.7976931348623157e+308"], [1e15, "1000000000000000.0"], [1e16, "1e+16"],
    [123456789012345678, "1.2345678901234568e+17"], [0.1 + 0.2, "0.30000000000000004"], [1 / 3, "0.3333333333333333"],
    [2 / 3, "0.6666666666666666"], [100, "100.0"], [1e-4, "0.0001"], [9.999999999999999e-5, "9.999999999999999e-05"],
    [1e-5, "1e-05"], [12345.678, "12345.678"], [-1.5e-7, "-1.5e-07"], [4.35, "4.35"], [0.000123, "0.000123"],
    [1e21, "1e+21"], [1e23, "1e+23"], [5e22, "5e+22"], [1234567890123456.8, "1234567890123456.8"], [-0, "-0.0"],
    [Infinity, "inf"], [-Infinity, "-inf"], [NaN, "nan"],
  ];
  for (const [value, expected] of cases) assert.equal(floatRepr(value), expected, String(value));
});

test("integers and floats keep their kind; integer-like keys keep their order", () => {
  const value = parse('{"10": 1, "2": 2.0, "a": [100000000000000000000, -0.0, 1e2]}');
  assert.equal(pythonText(value), '{"10": 1, "2": 2.0, "a": [100000000000000000000, -0.0, 100.0]}');
  assert.equal(stringify(value, { separators: [",", ":"] }), '{"10":1,"2":2.0,"a":[100000000000000000000,-0.0,100.0]}');
  assert.equal(stringify(value, { sortKeys: true }), '{"10": 1, "2": 2.0, "a": [100000000000000000000, -0.0, 100.0]}');
});

test("Python's json.loads accepts NaN and Infinity; json.dumps writes them unless allowNan is false", () => {
  const value = parse("[NaN, Infinity, -Infinity]");
  assert.equal(pythonText(value), "[NaN, Infinity, -Infinity]");
  assert.equal(isFiniteJson(value), false);
  assert.throws(() => stringify(value, { allowNan: false }), RangeError);
});

test("text that Python's json.loads rejects", () => {
  for (const bad of ["", "[1,]", '{"a": 1,}', "01", "1.", "1e", '"\u0001"', "'a'", '{"a" 1}', "[1 2]", "\ufeff{}", '"\\x"',
    '"\\u12"', "nul", "{1: 2}", "[1] x"]) {
    assert.throws(() => parse(bad), JsonParseError, JSON.stringify(bad));
  }
});

test("strings: escapes, surrogates and ASCII escaping", () => {
  assert.equal(parse('"\\ud83c\\udf24 \\u00e9"'), "🌤 é");
  assert.equal(parse('"\\ud800"'), "\ud800");
  assert.equal(stringify("서울 🌤"), '"\\uc11c\\uc6b8 \\ud83c\\udf24"');
  assert.equal(stringify("서울 🌤", { ensureAscii: false }), '"서울 🌤"');
  assert.equal(stringify("\u007f"), '"\\u007f"');
  assert.equal(stringify("\u007f", { ensureAscii: false }), '"\u007f"');
});

test("repr of strings and containers", () => {
  assert.equal(stringRepr("it's"), `"it's"`);
  assert.equal(stringRepr('say "hi"'), `'say "hi"'`);
  assert.equal(stringRepr(`both ' "`), `'both \\' "'`);
  assert.equal(stringRepr("\u0000\u007f\u00a0\u2028🌤\ud800"), "'\\x00\\x7f\\xa0\\u2028🌤\\ud800'");
  assert.equal(stringRepr("tab\there"), "'tab\\there'");
  assert.equal(pythonRepr(parse('{"a": [1, 2.0, null, true, false, "x"]}')), "{'a': [1, 2.0, None, True, False, 'x']}");
});

test("fromPlain and toPlain", () => {
  const plain = { a: 1, b: 1.5, c: [true, null, "x"], d: { e: undefined, f: 2n }, g: new Date(0) };
  const value = fromPlain(plain);
  assert.equal(pythonText(value), '{"a": 1, "b": 1.5, "c": [true, null, "x"], "d": {"f": 2}, "g": "1970-01-01T00:00:00.000Z"}');
  assert.deepEqual(toPlain(value), { a: 1, b: 1.5, c: [true, null, "x"], d: { f: 2 }, g: "1970-01-01T00:00:00.000Z" });
  assert.equal(pythonText(fromPlain({ p: 1, q: 0.5 }, "float")), '{"p": 1.0, "q": 0.5}');
  assert.ok(fromPlain({ f: () => 1 }) instanceof Map);
  assert.ok((fromPlain({ f: () => 1 }) as Map<string, unknown>).get("f") instanceof Unserializable);
  const json = parse('{"x": 4.0}');
  assert.equal(pythonText(fromPlain(json)), '{"x": 4.0}'); // JSON values pass through unchanged
  const proto = toPlain(parse('{"__proto__": 1}')) as Record<string, unknown>;
  assert.equal(Object.getPrototypeOf(proto), Object.prototype);
  assert.deepEqual(Object.keys(proto), ["__proto__"]);
});

test("Python equality: 4 == 4.0; key order only when asked", () => {
  assert.ok(equal(parse("4"), parse("4.0")));
  assert.ok(!equal(parse("4"), parse("4.5")));
  assert.ok(equal(parse('{"a": 1, "b": 2}'), parse('{"b": 2, "a": 1}')));
  assert.ok(!equal(parse('{"a": 1, "b": 2}'), parse('{"b": 2, "a": 1}'), true));
  assert.ok(!equal(parse("true"), parse("1")));
  assert.ok(new Float(0.5).valueOf() === 0.5 && String(new Float(2)) === "2.0");
});
