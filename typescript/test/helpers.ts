/** Shared test helpers. Every test is offline: HTTP goes through an injected fetch. */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { FetchLike, Result } from "../src/index.js";
import { type Json, compactText, equal, fromPlain, parse, pythonText, toPlain } from "../src/json.js";

export const KEY_NAMES = [
  "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY",
  "XAI_API_KEY", "OPENROUTER_API_KEY", "MODEL_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT",
  "AZURE_OPENAI_API_VERSION", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL",
];

/** Remove every key and endpoint variable, so tests never see the developer's environment. */
export function isolateEnvironment(): void {
  for (const name of KEY_NAMES) delete process.env[name];
}

/** Set environment variables for one test and restore them afterwards. */
export async function withEnv<T>(values: Record<string, string>, body: () => T | Promise<T>): Promise<T> {
  const saved = Object.fromEntries(Object.keys(values).map((name) => [name, process.env[name]]));
  Object.assign(process.env, values);
  try {
    return await body();
  } finally {
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
}

/** The package root (the folder with package.json), found from this file in src or in the build output. */
function packageRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "package.json");
    if (existsSync(candidate) && JSON.parse(readFileSync(candidate, "utf8")).name === "dmcall") return dir;
    const parent = dirname(dir);
    if (parent === dir) throw new Error("package.json of dmcall not found");
    dir = parent;
  }
}

export const ROOT = packageRoot();
export const SPEC = resolve(ROOT, "../spec");

/** A spec file parsed with number kinds and key order kept. */
export function loadSpec(name: string): Json {
  return parse(readFileSync(join(SPEC, name), "utf8"));
}

export function conformance(name: string): Json {
  return loadSpec(join("conformance", name));
}

/** Walk into a JSON value: `at(value, "scenarios", "x", 0)`. */
export function at(value: Json | undefined, ...path: (string | number)[]): Json {
  let node: Json | undefined = value;
  for (const key of path) {
    if (typeof key === "number") node = Array.isArray(node) ? node[key] : undefined;
    else node = node instanceof Map ? node.get(key) : undefined;
    if (node === undefined) throw new Error(`missing ${path.join(".")}`);
  }
  return node!;
}

export function list(value: Json): Json[] {
  if (!Array.isArray(value)) throw new TypeError("expected a JSON array");
  return value;
}

export function entries(value: Json): [string, Json][] {
  if (!(value instanceof Map)) throw new TypeError("expected a JSON object");
  return [...value];
}

/** Python's `==` between a plain or JSON value and an expected JSON value; `ordered` also checks key order. */
export function assertJson(actual: unknown, expected: Json, message?: string, ordered = false): void {
  const value = fromPlain(actual);
  if (!equal(value, expected, ordered)) {
    assert.fail(`${message ? `${message}: ` : ""}JSON differs${ordered ? " (key order checked)" : ""}\n`
      + `actual:   ${pythonText(value)}\nexpected: ${pythonText(expected)}`);
  }
}

export interface Seen {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string | undefined;
}

/** A fetch that records each request and answers with `handler`. */
export function mockFetch(handler: (request: Seen, index: number) => Response | Promise<Response>, seen: Seen[] = []): FetchLike {
  return async (url, init) => {
    const headers: Record<string, string> = {};
    for (const [name, value] of Object.entries(init.headers ?? {})) headers[name.toLowerCase()] = value;
    const request = { url, method: init.method, headers, body: init.body };
    seen.push(request);
    return handler(request, seen.length - 1);
  };
}

/** A JSON response; plain values are encoded with Python's number rules, JSON values as they are. */
export function respond(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(compactText(fromPlain(body)), { status, headers: { "content-type": "application/json", ...headers } });
}

/** A fetch that serves `bodies` in order and records the requests. */
export function serve(bodies: readonly Json[], seen: Seen[] = []): FetchLike {
  let next = 0;
  return mockFetch(() => {
    if (next >= bodies.length) throw new Error("no more scripted responses");
    return respond(bodies[next++]);
  }, seen);
}

export const noSleep = async (): Promise<void> => {};

/** camelCase usage keys back to the snake_case of the Python summary. */
export function snakeUsage(usage: Record<string, number | undefined>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(usage)) {
    if (value !== undefined) out[key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)] = value;
  }
  return out;
}

/** A result as the snake_case summary the conformance files record. */
export function summary(result: Result): Record<string, unknown> {
  return {
    calls: result.calls,
    accepted: result.accepted,
    exhausted: result.exhausted,
    repairs: result.repairs,
    final_attempt: result.finalAttempt,
    text: result.text,
    initial_no_call_preserved: result.initialNoCallPreserved,
    attempts: result.attempts.map((attempt) => ({
      calls: attempt.calls,
      accepted: attempt.accepted,
      function_probability: attempt.functionProbability,
      arguments_probability: attempt.argumentsProbability,
      schema_errors: attempt.schemaErrors,
    })),
    probabilities: result.probabilities,
    usage: { generator: snakeUsage(result.usage.generator), decision_model: snakeUsage(result.usage.decisionModel) },
  };
}

/** Only the keys a recorded summary has. */
export function pick(value: Record<string, unknown>, expected: Json): Record<string, unknown> {
  return Object.fromEntries(entries(expected).map(([key]) => [key, value[key]]));
}

/** snake_case option names of the conformance files as the camelCase options of the TypeScript API. */
export function camelOptions(options: Json): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of entries(options)) {
    const name = key.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
    out[name] = toPlain(value);
  }
  return out;
}

export const WEATHER = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Current weather for a city.",
    parameters: {
      type: "object",
      properties: { city: { type: "string" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } },
      required: ["city"],
    },
  },
};

export const EXAMPLE = {
  messages: [{ role: "system", content: "Be brief." }, { role: "user", content: "Weather in Seoul?" }],
  tools: [WEATHER],
};

export const HISTORY = [
  { calls: [{ name: "get_weather", arguments: { city: "Busan" } }],
    feedback: { failed_checks: ["arguments"], instruction: "Reconsider." } },
  { round: 2, calls: [], feedback: { failed_checks: ["function_selection"] } },
];

export const SECRET = "sk-test-SECRET-123";

/** A Chat Completions response body. */
export function chat(options: { calls?: [string, unknown][]; content?: string | null; finish?: string;
  usage?: Record<string, unknown>; message?: Record<string, unknown> } = {}): Record<string, unknown> {
  const toolCalls = (options.calls ?? []).map(([name, args], index) => ({
    id: `call_${index}`, type: "function",
    function: { name, arguments: typeof args === "string" ? args : JSON.stringify(args) },
  }));
  const message: Record<string, unknown> = { role: "assistant", content: options.content ?? null };
  if (toolCalls.length) message.tool_calls = toolCalls;
  Object.assign(message, options.message ?? {});
  const body: Record<string, unknown> = {
    model: "resolved-model",
    choices: [{ finish_reason: options.finish ?? (toolCalls.length ? "tool_calls" : "stop"), message }],
  };
  if (options.usage) body.usage = options.usage;
  return body;
}

/** A /v1/systemone response body. */
export function answer(fn: number = 0.95, args: number | null = 0.95, model = "jev-1.13.0"): Record<string, unknown> {
  const answers: Record<string, unknown> = { function_correct: { type: "noul", noul: fn } };
  if (args !== null) answers.arguments_correct = { type: "noul", noul: args };
  return { model, answers, usage: { input_tokens: 300, output_tokens: 0 } };
}

export const seoul = () => [{ name: "get_weather", arguments: { city: "Seoul" } }];
export const busan = () => [{ name: "get_weather", arguments: { city: "Busan" } }];
