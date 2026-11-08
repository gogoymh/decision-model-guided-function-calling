/** Jev and the PPLX Decider: the wire format, verdicts, keys, thresholds and failures (tests/test_decision_models.py). */

import assert from "node:assert/strict";
import { test } from "node:test";
import { inspect } from "node:util";

import * as dmcall from "../src/index.js";
import { EXAMPLE, SECRET, type Seen, WEATHER, answer, isolateEnvironment, mockFetch, noSleep, respond, seoul, withEnv } from "./helpers.js";

isolateEnvironment();

const MESSAGES = EXAMPLE.messages;
const TOOLS = EXAMPLE.tools;
const REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce";
const PROVENANCE = {
  model_id: "perplexity-ai/pplx-decider-v1.1-27b", revision: REVISION, native_attention: "noncausal_full_attention",
  max_input_tokens: 8192, temperature: 1.0087417621345625,
};

function jev(handler: (request: Seen) => Response, seen?: Seen[], options: dmcall.JevOptions = {}): dmcall.Jev {
  return new dmcall.Jev("jev-key", { fetch: mockFetch(handler, seen), sleep: noSleep, ...options });
}

const body = (seen: Seen) => JSON.parse(seen.body!);

test("the wire format sends only observable inputs", async () => {
  const seen: Seen[] = [];
  const tools = [{ ...WEATHER, metadata: { reference: "get_weather(city='Seoul')" } }];
  const messages = [...MESSAGES, { role: "user", content: "hi", ground_truth: "nope" }];
  const record = await jev(() => respond(answer(0.9, 0.8)), seen).judge(messages, tools, seoul());
  const request = body(seen[0]!);
  assert.equal(seen[0]!.url, "https://api.typesafe.ai/v1/systemone");
  assert.equal(request.model, "jev-1.13.0");
  assert.deepEqual(request.state.messages[request.state.messages.length - 1], { role: "user", content: "hi" });
  assert.deepEqual(request.state.tools, [WEATHER.function]);
  assert.deepEqual(request.state.proposed_calls, seoul());
  assert.deepEqual(request.questions, { function_correct: dmcall.FUNCTION_QUESTION, arguments_correct: dmcall.ARGUMENT_QUESTION });
  assert.ok(record.accepted);
  assert.equal(record.functionProbability, 0.9);
  assert.equal(record.argumentsProbability, 0.8);
  assert.deepEqual(record.feedback.failed_checks, []);
  assert.equal(record.apiCalls, 1);
});

test("a no call asks only the function question", async () => {
  const seen: Seen[] = [];
  const record = await jev(() => respond(answer(0.97, null)), seen).judge(MESSAGES, TOOLS, []);
  assert.deepEqual(Object.keys(body(seen[0]!).questions), ["function_correct"]);
  assert.ok(record.accepted && record.argumentsSkipped);
  assert.equal(record.argumentsProbability, null);
  assert.equal(record.feedback.checks.arguments!.pass, null);
});

for (const [fn, args, failed] of [[0.4, 0.9, ["function_selection"]], [0.9, 0.49, ["arguments"]], [0.5, 0.5, []]] as const) {
  test(`both questions must pass at the threshold (${fn}, ${args})`, async () => {
    const record = await jev(() => respond(answer(fn, args))).judge(MESSAGES, TOOLS, seoul());
    assert.deepEqual(record.feedback.failed_checks, failed);
    assert.equal(record.accepted, failed.length === 0);
  });
}

test("a schema failure is judged but cannot be overruled", async () => {
  const seen: Seen[] = [];
  const record = await jev(() => respond(answer(0.99, 0.99)), seen).judge(MESSAGES, TOOLS, [{ name: "get_weather", arguments: {} }]);
  assert.equal(seen.length, 1);
  assert.equal(record.argumentsProbability, 0.99); // a readable candidate is still judged
  assert.equal(record.argumentsPass, false);
  assert.equal(record.accepted, false);
  assert.ok(record.schemaErrors.length > 0);
});

for (const calls of [null, [{ name: "get_weather", arguments: null }]]) {
  test(`an unreadable candidate is not sent (${JSON.stringify(calls)})`, async () => {
    const record = await jev(() => assert.fail("An unreadable candidate must not reach the decision model")).judge(MESSAGES, TOOLS, calls);
    assert.equal(record.apiCalls, 0);
    assert.equal(record.functionProbability, null);
    assert.equal(record.accepted, false);
  });
}

test("thresholds, keys and model", async () => {
  assert.throws(() => new dmcall.Jev(), /TYPESAFE_API_KEY/);
  await withEnv({ TYPESAFE_API_KEY: SECRET, TYPESAFE_MODEL: "jev-2" }, async () => {
    const seen: Seen[] = [];
    const model = new dmcall.Jev(undefined, { threshold: 0.7, argumentsThreshold: 0.9,
      fetch: mockFetch(() => respond(answer(0.75, 0.85)), seen) });
    const record = await model.judge(MESSAGES, TOOLS, seoul());
    assert.equal(seen[0]!.headers.authorization, `Bearer ${SECRET}`);
    for (const shown of [String(model), JSON.stringify(model), inspect(model, { showHidden: true, depth: 5 })]) {
      assert.ok(!shown.includes(SECRET), shown);
    }
    assert.equal(body(seen[0]!).model, "jev-2");
    assert.equal(record.functionPass, true);
    assert.equal(record.argumentsPass, false);
  });
  for (const bad of [1.5, -0.1, true, Number.NaN]) {
    assert.throws(() => new dmcall.Jev("k", { threshold: bad as number }), /Thresholds/);
  }
});

test("retries are bounded and errors sanitized", async () => {
  const replies = [respond({}, 429, { "Retry-After": "0" }), respond(answer())];
  const record = await jev(() => replies.shift()!).judge(MESSAGES, TOOLS, seoul());
  assert.equal(record.apiCalls, 2);
  const seen: Seen[] = [];
  const failing = jev(() => new Response(SECRET, { status: 401, headers: { "x-request-id": "req-7" } }), seen);
  await assert.rejects(failing.judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
    assert.ok(error instanceof dmcall.DecisionModelError);
    assert.equal(seen.length, 1);
    assert.equal(error.statusCode, 401);
    assert.equal(error.requestId, "req-7");
    assert.ok(!error.message.includes(SECRET) && !inspect(error).includes(SECRET));
    return true;
  });
  const overloaded = jev(() => respond({}, 503), undefined, { maxRetries: 2 });
  await assert.rejects(overloaded.judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
    assert.ok(error instanceof dmcall.DecisionModelError);
    assert.equal(error.attempts, 3);
    return true;
  });
});

test("Retry-After as seconds or an HTTP date, otherwise exponential backoff", async () => {
  const waits: number[] = [];
  const later = new Date(Date.now() + 60_000).toUTCString();
  const replies = [respond({}, 503, { "Retry-After": "2.5" }), respond({}, 503, { "Retry-After": later }),
    respond({}, 503, { "Retry-After": "soon" }), respond(answer())];
  const model = new dmcall.Jev("k", { maxRetries: 3, fetch: mockFetch(() => replies.shift()!),
    sleep: async (ms) => { waits.push(ms); } });
  await model.judge(MESSAGES, TOOLS, seoul());
  assert.equal(waits[0], 2500);
  assert.equal(waits[1], 30_000); // capped at 30 seconds
  assert.equal(waits[2], 4000); // 2 ** attempt
});

test("a connection failure is an error, never a rejection", async () => {
  let calls = 0;
  const model = new dmcall.Jev("k", { maxRetries: 1, sleep: noSleep, fetch: async () => {
    calls++;
    throw new TypeError("fetch failed");
  } });
  await assert.rejects(model.judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
    assert.ok(error instanceof dmcall.DecisionModelError);
    assert.equal(error.code, "connection_failed");
    assert.equal(error.attempts, 2);
    return true;
  });
  assert.equal(calls, 2);
});

for (const [name, reply] of [
  ["no answers", { answers: {} }], ["probability above 1", answer(1.5)], ["usage null", { ...answer(), usage: null }],
  ["empty model", { ...answer(), model: "" }], ["not an object", ["not", "an", "object"]],
  ["float usage", { ...answer(), usage: { input_tokens: 1.5 } }], ["boolean probability", answer(true as unknown as number)],
] as const) {
  test(`an invalid response is an error, not a rejection: ${name}`, async () => {
    await assert.rejects(jev(() => respond(reply)).judge(MESSAGES, TOOLS, seoul()), dmcall.DecisionModelError);
  });
}

test("a body that is not JSON is an error", async () => {
  await assert.rejects(jev(() => new Response("<html>")).judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
    assert.ok(error instanceof dmcall.DecisionModelError);
    assert.equal(error.code, "response_invalid");
    return true;
  });
});

// PPLX Decider ------------------------------------------------------------------------------------
function pplx(handler: (request: Seen) => Response, seen?: Seen[], options: dmcall.PPLXDeciderOptions = {}): dmcall.PPLXDecider {
  return new dmcall.PPLXDecider("http://127.0.0.1:8000", { fetch: mockFetch(handler, seen), ...options });
}

test("PPLX uses the same questions and reports the checkpoint", async () => {
  const seen: Seen[] = [];
  const decider = pplx(() => respond(answer(0.9, 0.2, "autojev-qwen3.8-27b")), seen);
  const record = await decider.judge(MESSAGES, TOOLS, seoul());
  const request = body(seen[0]!);
  assert.equal(seen[0]!.url, "http://127.0.0.1:8000/v1/systemone");
  assert.equal(request.model, "jev-latest");
  assert.deepEqual(request.questions.function_correct, dmcall.FUNCTION_QUESTION);
  assert.equal(record.model, "perplexity-ai/pplx-decider-v1.1-27b");
  assert.equal(record.accepted, false);
  assert.equal(record.feedback.source, "perplexity_decider_noul_and_deterministic_schema_validation");
  assert.ok(!("authorization" in seen[0]!.headers));
});

test("PPLX missing usage stays unknown", async () => {
  const reply = { model: "autojev-qwen3.8-27b", answers: answer().answers };
  const record = await pplx(() => respond(reply)).judge(MESSAGES, TOOLS, seoul());
  assert.deepEqual(record.usage, { inputTokens: null, outputTokens: null });
});

for (const reply of [answer(0.95, 0.95, "some-other-model"),
  { ...answer(0.95, 0.95, "perplexity-ai/pplx-decider-v1.1-27b"), revision: "0".repeat(40) }]) {
  test(`PPLX rejects another model (${JSON.stringify(reply).slice(0, 40)})`, async () => {
    await assert.rejects(pplx(() => respond(reply)).judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
      assert.ok(error instanceof dmcall.DecisionModelError);
      assert.equal(error.code, "model_mismatch");
      return true;
    });
  });
}

test("the PPLX provenance check is opt-in", async () => {
  const seen: Seen[] = [];
  const handler = (request: Seen) => new URL(request.url).pathname === "/provenance" ? respond(PROVENANCE)
    : respond(answer(0.95, 0.95, "autojev-qwen3.8-27b"));
  const decider = pplx(handler, seen, { checkProvenance: true });
  await decider.judge(MESSAGES, TOOLS, seoul());
  await decider.judge(MESSAGES, TOOLS, seoul());
  assert.deepEqual(seen.map((request) => `${request.method} ${new URL(request.url).pathname}`),
    ["GET /provenance", "POST /v1/systemone", "POST /v1/systemone"]);
  const unchecked: Seen[] = [];
  await pplx(handler, unchecked).judge(MESSAGES, TOOLS, seoul());
  assert.deepEqual(unchecked.map((request) => new URL(request.url).pathname), ["/v1/systemone"]);
  const wrong = pplx(() => respond({ ...PROVENANCE, revision: "main" }), undefined, { checkProvenance: true });
  await assert.rejects(wrong.judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
    assert.ok(error instanceof dmcall.DecisionModelError);
    assert.equal(error.code, "provenance_mismatch");
    return true;
  });
  const float = pplx(() => respond({ ...PROVENANCE, max_input_tokens: 8192.0 }), undefined, { checkProvenance: true });
  assert.equal((await float.provenance()).max_input_tokens, 8192);
});

test("the PPLX context limit is a named error", async () => {
  const over = pplx(() => respond({ detail: "Prepared input exceeds the 8192-token limit." }, 422));
  await assert.rejects(over.judge(MESSAGES, TOOLS, seoul()), (error: unknown) => {
    assert.ok(error instanceof dmcall.DecisionModelError);
    assert.equal(error.code, "context_limit_exceeded");
    return true;
  });
});

for (const url of ["http://10.0.0.4:8000", "https://user:pw@example.com", "http://localhost:8000/v1", "ftp://localhost"]) {
  test(`PPLX rejects an unsafe URL: ${url}`, () => {
    assert.throws(() => new dmcall.PPLXDecider(url), Error);
  });
}

test("PPLX accepts HTTPS anywhere and HTTP on localhost", () => {
  assert.equal(new dmcall.PPLXDecider("https://decider.example.com").baseUrl, "https://decider.example.com");
  assert.equal(new dmcall.PPLXDecider("http://[::1]:8000/").baseUrl, "http://[::1]:8000");
  assert.equal(new dmcall.PPLXDecider("http://localhost:8000").maxRetries, 0);
});
