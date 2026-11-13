/** The schema check S(c) and the feedback representations Φ (tests/test_schema_and_feedback.py). */

import assert from "node:assert/strict";
import { test } from "node:test";

import * as dmcall from "../src/index.js";
import { represent, representHistory, singleCheck } from "../src/feedback.js";
import { WEATHER, isolateEnvironment } from "./helpers.js";

isolateEnvironment();

const TOOLS = [WEATHER];

function check(args: unknown, tools: unknown[] = TOOLS, name = "get_weather"): dmcall.SchemaCheck {
  return dmcall.checkSchema(tools, [{ name, arguments: args }]);
}

// S(c) ---------------------------------------------------------------------------------------------
test("a valid call and a no call pass", () => {
  assert.equal(check({ city: "Seoul" }).accepted, true);
  const empty = dmcall.checkSchema(TOOLS, []);
  assert.ok(empty.accepted && empty.wellFormed);
  assert.deepEqual(empty.errors, []);
});

for (const args of [{}, { city: "Seoul", invented: 1 }, { city: 3 }]) {
  test(`argument structure error: ${JSON.stringify(args)}`, () => {
    const result = check(args);
    assert.ok(result.functionPass && !result.argumentsPass && result.wellFormed);
  });
}

test("an unknown function is readable but not available", () => {
  const result = check({}, TOOLS, "invented");
  assert.ok(!result.functionPass && !result.argumentsPass && result.wellFormed);
});

for (const calls of [null, ["get_weather"], [{ name: "get_weather", arguments: "{}" }],
  [{ name: "get_weather", arguments: { city: Number.NaN } }], [{ name: "get_weather", arguments: { city: () => 1 } }],
  [{ name: "", arguments: {} }]]) {
  test(`an unreadable candidate is not well formed: ${String(JSON.stringify(calls))}`, () => {
    const result = dmcall.checkSchema(TOOLS, calls);
    assert.ok(!result.accepted && !result.wellFormed);
  });
}

test("types and required names only, not value constraints", () => {
  const tools = [{ type: "function", function: { name: "book", description: "", parameters: {
    type: "object",
    properties: {
      genre: { type: "string", enum: ["Horror", "Comedy"] },
      rooms: { type: "integer", minimum: 1 },
      metrics: { type: "array", enum: ["view", "trust"], items: { type: "string" } },
      start: { type: "string", default: null },
      end: { type: "string" },
      filters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
    required: ["genre"],
  } } }];
  // Values outside an enum or range are the decision model's question; a declared null default is admissible.
  const passed = check({ genre: "Any", rooms: 0, metrics: ["view"], start: null, filters: { city: "Seoul" } }, tools, "book");
  assert.ok(passed.accepted);
  assert.deepEqual(passed.errors, []);
  for (const args of [{ genre: true }, { genre: "Any", rooms: "2" }, { genre: "Any", end: null }, { genre: "Any", metrics: "view" },
    { rooms: 1 }, { genre: "Any", extra: 1 }, { genre: "Any", filters: {} }]) {
    const result = check(args, tools, "book");
    assert.ok(result.functionPass && !result.argumentsPass, JSON.stringify(args));
  }
});

test("alternatives and open keywords", () => {
  const parameters = { type: "object", additionalProperties: false, required: ["x"], properties: {
    x: { oneOf: [{ type: "string", enum: ["a"] }, { type: "string", maxLength: 1 }] } } };
  const tools = [{ type: "function", function: { name: "f", description: "", parameters } }];
  assert.equal(check({ x: "abc" }, tools, "f").accepted, true);
  for (const args of [{ x: 3 }, {}, { x: "a", y: 1 }]) assert.equal(check(args, tools, "f").accepted, false, JSON.stringify(args));
  const open = [{ type: "function", function: { ...WEATHER.function, parameters: { ...WEATHER.function.parameters,
    additionalProperties: true } } }];
  assert.equal(check({ city: "Seoul", extra: 1 }, open).accepted, true);
});

test("an invalid tool schema is a configuration error", () => {
  const bad = [{ type: "function", function: { name: "f", parameters: { type: "object", properties: { a: { type: "nonsense" } } } } }];
  assert.throws(() => dmcall.checkSchema(bad, []), /invalid JSON Schema/);
  assert.throws(() => dmcall.checkSchema([WEATHER, WEATHER], []), /unique/);
});

test("Python messages for the less common keywords", () => {
  const tool = (parameters: unknown) => [{ type: "function", function: { name: "f", description: "", parameters } }];
  const errors = (parameters: unknown, args: unknown) => check(args, tool(parameters), "f").errors.map((error) => [error.path,
    error.message]);
  assert.deepEqual(errors({ type: "object", patternProperties: { "^x": { type: "integer" } }, additionalProperties: false },
    { x1: "a", y: 1, z: 2 }), [
    ['calls[0].arguments["x1"]', "'a' is not of type 'integer'"],
    ["calls[0].arguments", "'y', 'z' do not match any of the regexes: '^x'"],
  ]);
  assert.deepEqual(errors({ type: "object", properties: { t: { type: "array", prefixItems: [{ type: "string" }], items: false } } },
    { t: ["a", 1, 2] }), [['calls[0].arguments["t"]', "Expected at most 1 item but found 2 extra: [1, 2]"]]);
  assert.deepEqual(errors({ type: "object", properties: { n: false } }, { n: { "it's": [1.0, null] } }),
    [["calls[0].arguments", `False schema does not allow {"it's": [1, None]}`]]); // as in jsonschema, without "n"
  assert.deepEqual(errors({ type: "object", properties: { "서울": { type: "string" } } }, { "서울": 1 }),
    [['calls[0].arguments["\\uc11c\\uc6b8"]', "1 is not of type 'string'"]]);
});

test("draft-07 schemas: $ref hides its siblings and items may be a tuple", () => {
  const parameters = {
    $schema: "http://json-schema.org/draft-07/schema#", type: "object",
    properties: { p: { $ref: "#/definitions/point", type: "string" }, pair: { type: "array", items: [{ type: "integer" }] } },
    definitions: { point: { type: "object", required: ["x"] } },
  };
  const tools = [{ type: "function", function: { name: "f", description: "", parameters } }];
  assert.deepEqual(check({ p: {}, pair: ["a", "b"] }, tools, "f").errors.map((error) => error.message),
    ["'x' is a required property", "'a' is not of type 'integer'"]);
});

// Φ -------------------------------------------------------------------------------------------------
for (const [p, expected] of [[0, "very likely incorrect"], [0.19, "very likely incorrect"], [0.2, "likely incorrect"],
  [0.48, "uncertain"], [0.52, "uncertain"], [0.6, "likely correct"], [0.8, "very likely correct"], [1, "very likely correct"],
  [null, null]] as const) {
  test(`label(${p})`, () => assert.equal(dmcall.label(p), expected));
}

for (const bad of [-0.1, 1.1, Number.NaN, true, "0.5"]) {
  test(`label rejects ${String(bad)}`, () => assert.throws(() => dmcall.label(bad as number), Error));
}

const FEEDBACK = {
  source: "jev_noul_and_deterministic_schema_validation",
  checks: {
    function_selection: { pass: true, probability_yes: 0.95, threshold: 0.5, schema_pass: true },
    arguments: { skipped: false, pass: false, probability_yes: 0.12, threshold: 0.5, schema_pass: true },
  },
  failed_checks: ["arguments"],
  schema_errors: [{ check: "arguments", path: "calls[0].arguments", message: "'probability_yes' 0.3" }],
  instruction: "Reconsider ... The probabilities are verifier judgments, not ground truth or explanations.",
};

test("hybrid keeps numbers and thresholds and adds labels", () => {
  const out = represent(FEEDBACK, "hybrid") as any;
  const args = out.checks.arguments;
  assert.deepEqual(Object.keys(args), ["skipped", "pass", "probability_yes", "feedback_label", "threshold", "schema_pass"]);
  assert.equal(args.probability_yes, 0.12);
  assert.equal(args.feedback_label, "very likely incorrect");
  assert.equal(out.checks.function_selection.feedback_label, "very likely correct");
  assert.deepEqual(out.schema_errors, FEEDBACK.schema_errors);
  assert.equal(out.instruction, FEEDBACK.instruction);
  assert.ok(!("feedback_label" in FEEDBACK.checks.arguments)); // the input is not changed
});

test("text replaces numbers and thresholds with labels", () => {
  const out = represent(FEEDBACK, "text") as any;
  assert.deepEqual(out.checks.arguments, { skipped: false, pass: false, feedback_label: "very likely incorrect", schema_pass: true });
  assert.ok(out.instruction.endsWith("The feedback labels are verifier judgments, not ground truth or explanations."));
  assert.deepEqual(out.schema_errors, FEEDBACK.schema_errors); // task diagnostics stay verbatim
});

test("numeric is unchanged and skipped questions stay null", () => {
  assert.deepEqual(represent(FEEDBACK, "numeric"), FEEDBACK);
  const skipped = { checks: { arguments: { skipped: true, pass: null, probability_yes: null, threshold: 0.5 } } };
  assert.equal((represent(skipped, "hybrid") as any).checks.arguments.feedback_label, null);
  assert.deepEqual((represent(skipped, "text") as any).checks.arguments, { skipped: true, pass: null, feedback_label: null });
  assert.throws(() => represent(FEEDBACK, "words" as never), /feedback/);
  const history = [{ calls: [], feedback: FEEDBACK }];
  assert.deepEqual(representHistory(history, "text")[0]!.feedback, represent(FEEDBACK, "text"));
  assert.throws(() => representHistory([{ calls: [] } as never], "hybrid"), Error);
});

test("singleCheck removes only the other question", () => {
  const functionOnly = singleCheck(FEEDBACK, "function");
  assert.deepEqual(Object.keys(functionOnly.checks), ["function_selection"]);
  assert.deepEqual(functionOnly.failed_checks, []);
  const argumentsOnly = singleCheck(FEEDBACK, "arguments");
  assert.deepEqual(Object.keys(argumentsOnly.checks), ["arguments"]);
  assert.deepEqual(argumentsOnly.failed_checks, ["arguments"]);
  assert.deepEqual(argumentsOnly.schema_errors, FEEDBACK.schema_errors);
  assert.throws(() => singleCheck(FEEDBACK, "both" as never), Error);
});
