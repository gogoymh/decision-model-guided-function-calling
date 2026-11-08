/** Tool definitions in every format, tool() with handlers, and calls from any SDK (tests/test_tools.py, adapted). */

import assert from "node:assert/strict";
import { test } from "node:test";

import * as dmcall from "../src/index.js";
import { toolset } from "../src/tools.js";
import { WEATHER, isolateEnvironment } from "./helpers.js";

isolateEnvironment();

test("tool() pairs a definition with its handler", async () => {
  const getWeather = dmcall.tool({
    name: "weather",
    description: "Weather now.",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    execute: async ({ city }) => `sunny in ${city}`,
  });
  assert.ok(getWeather instanceof dmcall.Tool);
  assert.deepEqual(dmcall.schemaOf(getWeather), { type: "function", function: {
    name: "weather", description: "Weather now.",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } });
  assert.deepEqual(JSON.parse(JSON.stringify(getWeather)), dmcall.schemaOf(getWeather)); // the handler is not serialized
  assert.equal(await getWeather.execute!({ city: "Seoul" }), "sunny in Seoul");
  const bare = dmcall.tool({ name: "ping" });
  assert.deepEqual(dmcall.schemaOf(bare).function, { name: "ping", description: "", parameters: { type: "object", properties: {} } });
  assert.equal(bare.execute, undefined);
});

test("definitions in OpenAI, Anthropic and Gemini shape", () => {
  const parameters = WEATHER.function.parameters;
  for (const definition of [
    WEATHER, WEATHER.function, { type: "function", ...WEATHER.function },
    { name: "get_weather", description: "Current weather for a city.", input_schema: parameters },
    { name: "get_weather", description: "Current weather for a city.", parametersJsonSchema: parameters },
    { name: "get_weather", description: "Current weather for a city.", parameters: {
      type: "OBJECT", properties: { city: { type: "STRING" }, unit: { type: "STRING", enum: ["celsius", "fahrenheit"] } },
      required: ["city"] } },
  ]) {
    assert.deepEqual(dmcall.schemaOf(definition), WEATHER);
  }
  assert.throws(() => dmcall.schemaOf({ functionDeclarations: [WEATHER.function] }), /Gemini function declaration/);
});

test("handlers come from tool() or from functions", () => {
  const weather = dmcall.tool({ name: "weather", execute: () => "w" });
  const { schemas, registry } = toolset([weather, WEATHER], { get_weather: () => "g" });
  assert.deepEqual(schemas.map((schema) => (schema.get("function") as Map<string, unknown>).get("name")), ["weather", "get_weather"]);
  assert.deepEqual([...registry.keys()].sort(), ["get_weather", "weather"]);
});

for (const [name, bad, error] of [
  ["a bare function", (city: string) => city, TypeError],
  ["a string", "get_weather", TypeError],
  ["no name", { name: "", parameters: { type: "object" } }, Error],
  ["not an object schema", { name: "x", parameters: { type: "array" } }, Error],
  ["an invalid schema", { name: "x", parameters: { type: "object", properties: { a: { type: "nonsense" } } } }, Error],
  ["a description that is not text", { name: "x", description: 3, parameters: { type: "object" } }, Error],
] as const) {
  test(`a bad tool is rejected: ${name}`, () => {
    assert.throws(() => toolset([bad]), error);
  });
}

test("tool() checks its definition and handler", () => {
  assert.throws(() => dmcall.tool({ name: "" }), /nonempty name/);
  assert.throws(() => dmcall.tool({ name: "x", parameters: { type: "object", properties: { a: { type: 1 } } } }), /valid JSON Schema/);
  assert.throws(() => dmcall.tool({ name: "x", execute: "run" as never }), TypeError);
});

test("names must be unique and functions must name a tool", () => {
  assert.throws(() => toolset([WEATHER, WEATHER]), /Two tools/);
  assert.throws(() => toolset([WEATHER], { nope: () => null }), /not one of the tools/);
  assert.throws(() => toolset([WEATHER], { get_weather: "run" as never }), TypeError);
});

test("calls from every SDK shape", () => {
  const seoul = [{ name: "get_weather", arguments: { city: "Seoul" } }];
  const shapes = [
    { name: "get_weather", arguments: { city: "Seoul" } },
    { id: "c1", type: "function", function: { name: "get_weather", arguments: '{"city": "Seoul"}' } },
    { type: "function_call", call_id: "c1", name: "get_weather", arguments: '{"city": "Seoul"}' },
    { type: "tool_use", id: "t1", name: "get_weather", input: { city: "Seoul" } },
    { functionCall: { name: "get_weather", args: { city: "Seoul" } } },
  ];
  for (const shape of shapes) assert.deepEqual(dmcall.normalizeCalls([shape]), seoul);
  class SDKCall {
    toJSON() {
      return shapes[1];
    }
  }
  assert.deepEqual(dmcall.normalizeCalls([new SDKCall()]), seoul);
  assert.deepEqual(dmcall.normalizeCalls(null), []); // SDKs report no call as tool_calls: null
  assert.deepEqual(dmcall.normalizeCalls(undefined), []);
  assert.deepEqual(dmcall.normalizeCalls({ name: "f", arguments: "{oops" }), [{ name: "f", arguments: null }]);
  assert.throws(() => dmcall.normalizeCalls(3), TypeError);
  assert.throws(() => dmcall.normalizeCalls(["get_weather"]), TypeError);
});
