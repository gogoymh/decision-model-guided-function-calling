/** The package: its public names, its metadata and its build settings (tests/test_package.py). */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import * as dmcall from "../src/index.js";
import { ROOT } from "./helpers.js";

const manifest = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));

test("public names resolve", () => {
  for (const name of [
    "DMCall", "Result", "Verdict", "verify", "CHECKS", "HISTORIES", "ON_EXHAUSTION", "DecisionModel", "DecisionModelError", "Jev",
    "PPLXDecider", "OpenAI", "OpenAICompatible", "XAI", "OpenRouter", "Meta", "AzureOpenAI", "Anthropic", "Gemini", "Provider",
    "ProviderError", "PROVIDERS", "provider", "checkSchema", "label", "normalizeCalls", "tool", "Tool", "schemaOf",
    "SYSTEM_PROMPT", "FUNCTION_QUESTION", "ARGUMENT_QUESTION", "FEEDBACK", "VERSION",
  ]) {
    assert.notEqual((dmcall as Record<string, unknown>)[name], undefined, name);
  }
});

test("the version and metadata match package.json", () => {
  assert.equal(dmcall.VERSION, manifest.version);
  assert.equal(manifest.name, "dmcall");
  assert.equal(manifest.version, "1.0.0");
  assert.equal(manifest.license, "MIT");
  assert.equal(manifest.type, "module");
  assert.equal(manifest.types, "./dist/index.d.ts");
  assert.deepEqual(Object.keys(manifest.exports), [".", "./json", "./package.json"]);
  assert.ok(manifest.files.includes("dist"));
  assert.deepEqual(manifest.dependencies ?? {}, {}); // zero runtime dependencies
  assert.ok(manifest.scripts.build && manifest.scripts.test);
});

test("the questions and the prompt cannot be changed by accident", () => {
  assert.ok(Object.isFrozen(dmcall.FUNCTION_QUESTION) && Object.isFrozen(dmcall.ARGUMENT_QUESTION.criteria));
});
