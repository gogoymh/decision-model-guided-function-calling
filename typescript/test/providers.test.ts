/** Generators over plain HTTP: requests, decoding, unreadable outputs, keys and errors (tests/test_providers.py). */

import assert from "node:assert/strict";
import { test } from "node:test";
import { inspect } from "node:util";

import * as dmcall from "../src/index.js";
import {
  EXAMPLE, HISTORY, SECRET, type Seen, WEATHER, chat, isolateEnvironment, mockFetch, noSleep, respond, seoul, withEnv,
} from "./helpers.js";

isolateEnvironment();

const body = (seen: Seen) => JSON.parse(seen.body!);

test("OpenAI request and decoding", async () => {
  const seen: Seen[] = [];
  const model = new dmcall.OpenAI("gpt-5-nano", {
    apiKey: SECRET, reasoningEffort: "none",
    fetch: mockFetch(() => respond(chat({ calls: [["get_weather", { city: "Seoul" }]],
      usage: { prompt_tokens: 120, completion_tokens: 9, prompt_tokens_details: { cached_tokens: 64 } } })), seen),
  });
  const proposal = await model.generate(EXAMPLE, HISTORY);
  const request = body(seen[0]!);
  assert.equal(seen[0]!.url, "https://api.openai.com/v1/chat/completions");
  assert.equal(seen[0]!.headers.authorization, `Bearer ${SECRET}`);
  assert.deepEqual(request.messages[0], { role: "system", content: dmcall.SYSTEM_PROMPT });
  assert.ok(request.messages[1].content.startsWith("Accumulated revision history (oldest first; proposals NOT executed)"));
  assert.deepEqual(request.messages.slice(2), EXAMPLE.messages); // the original request follows, unchanged
  assert.deepEqual(request.tools, [WEATHER]);
  assert.equal(request.tool_choice, "auto");
  assert.equal(request.parallel_tool_calls, false);
  assert.equal(request.max_completion_tokens, 4096);
  assert.equal(request.reasoning_effort, "none");
  assert.deepEqual(proposal.calls, seoul());
  assert.equal(proposal.incomplete, null);
  assert.deepEqual(proposal.usage, { inputTokens: 120, cachedInputTokens: 64, outputTokens: 9 });
  assert.equal(proposal.model, "resolved-model");
  assert.equal(proposal.provider, "openai");
  for (const shown of [String(model), JSON.stringify(model), inspect(model, { depth: 5, showHidden: true })]) {
    assert.ok(!shown.includes(SECRET), shown);
  }
});

test("the history lists rounds oldest first", async () => {
  const seen: Seen[] = [];
  const model = new dmcall.OpenAI("m", { apiKey: "k", fetch: mockFetch(() => respond(chat()), seen) });
  await model.generate(EXAMPLE, HISTORY);
  const text: string = body(seen[0]!).messages[1].content;
  const records = JSON.parse(text.split("\n").slice(1).join("\n").split("\nUse these previous judgments")[0]!);
  assert.deepEqual(records.map((record: { round: number }) => record.round), [1, 2]);
  assert.deepEqual(records[0].calls, HISTORY[0]!.calls);
  assert.deepEqual(records[1].calls, []);
  assert.ok(text.endsWith("when answering the original request below."));
});

const OUTCOMES: [string, Record<string, unknown>, unknown, string | null][] = [
  ["text", chat({ content: "No function applies." }), [], null],
  ["empty", chat(), [], null],
  ["length", chat({ finish: "length" }), null, "length"],
  ["content filter", chat({ finish: "content_filter" }), null, "content_filter"],
  ["refusal", chat({ message: { refusal: "I can't help with that." } }), null, "refusal"],
  ["bad json", chat({ calls: [["get_weather", "{not json"]] }), [{ name: "get_weather", arguments: null }], null],
];

for (const [name, reply, calls, incomplete] of OUTCOMES) {
  test(`OpenAI outcome: ${name}`, async () => {
    const model = new dmcall.OpenAI("m", { apiKey: "k", fetch: mockFetch(() => respond(reply)) });
    const proposal = await model.generate(EXAMPLE);
    assert.deepEqual(proposal.calls, calls);
    assert.equal(proposal.incomplete, incomplete);
    if (incomplete || (Array.isArray(calls) && calls.length && (calls[0] as { arguments: unknown }).arguments === null)) {
      assert.ok(proposal.parseErrors.length > 0);
    }
  });
}

test("a filtered prompt is an unreadable candidate, not an error", async () => {
  const model = new dmcall.OpenAI("m", { apiKey: "k",
    fetch: mockFetch(() => respond({ error: { code: "content_filter", message: "filtered" } }, 400)) });
  const proposal = await model.generate(EXAMPLE);
  assert.equal(proposal.calls, null);
  assert.equal(proposal.incomplete, "prompt_filter");
  assert.deepEqual(proposal.usage, {});
});

test("HTTP errors are sanitized and retries bounded", async () => {
  const seen: Seen[] = [];
  const model = new dmcall.OpenAI("m", { apiKey: SECRET, maxRetries: 2, sleep: noSleep, fetch: mockFetch(() =>
    respond({ error: { code: "invalid_api_key", message: SECRET } }, 401, { "x-request-id": "req-1" }), seen) });
  await assert.rejects(model.generate(EXAMPLE), (error: unknown) => {
    assert.ok(error instanceof dmcall.ProviderError);
    assert.equal(seen.length, 1);
    assert.equal(error.statusCode, 401);
    assert.equal(error.providerCode, "invalid_api_key");
    assert.equal(error.requestId, "req-1");
    assert.ok(!error.message.includes(SECRET) && !inspect(error).includes(SECRET));
    return true;
  });
  const replies = [respond({}, 429, { "retry-after": "1" }), respond({}, 503),
    respond(chat({ calls: [["get_weather", { city: "Seoul" }]] }))];
  const waits: number[] = [];
  const retried = new dmcall.OpenAI("m", { apiKey: "k", maxRetries: 2, fetch: mockFetch(() => replies.shift()!),
    sleep: async (ms) => { waits.push(ms); } });
  const proposal = await retried.generate(EXAMPLE);
  assert.equal(proposal.apiCalls, 3);
  assert.deepEqual(proposal.calls, seoul());
  assert.deepEqual(waits, [1000, 2000]); // Retry-After, then 2 ** attempt seconds
  const exhausted = new dmcall.OpenAI("m", { apiKey: "k", maxRetries: 1, sleep: noSleep, fetch: mockFetch(() => respond({}, 500)) });
  await assert.rejects(exhausted.generate(EXAMPLE), /HTTP 500/);
});

test("transport failures are retried, then reported without details", async () => {
  let calls = 0;
  const model = new dmcall.OpenAI("m", { apiKey: SECRET, maxRetries: 2, sleep: noSleep, fetch: async () => {
    calls++;
    throw new TypeError(`fetch failed for ${SECRET}`);
  } });
  await assert.rejects(model.generate(EXAMPLE), (error: unknown) => {
    assert.ok(error instanceof dmcall.ProviderError);
    assert.equal(error.message, "openai transport failed after 3 attempts (TypeError).");
    assert.equal(error.attempts, 3);
    return true;
  });
  assert.equal(calls, 3);
});

test("a response that does not match the protocol is a ProviderError", async () => {
  for (const reply of [new Response("not json"), respond([1, 2]), respond({ choices: [] }),
    respond({ choices: [{ finish_reason: ["x"], message: { role: "assistant" } }] }),
    respond({ choices: [{ message: { role: "assistant", content: 3 } }] }),
    respond(chat({ usage: { prompt_tokens: 1.5 } }))]) {
    const model = new dmcall.OpenAI("m", { apiKey: "k", fetch: async () => reply });
    await assert.rejects(model.generate(EXAMPLE), dmcall.ProviderError);
  }
});

test("keys come from options or the environment", async () => {
  assert.throws(() => new dmcall.OpenAI("m"), /OPENAI_API_KEY/);
  await withEnv({ OPENAI_API_KEY: "from-env", OPENAI_BASE_URL: "https://proxy.example.com/v1/" }, async () => {
    const seen: Seen[] = [];
    await new dmcall.OpenAI("m", { fetch: mockFetch(() => respond(chat()), seen) }).generate(EXAMPLE);
    assert.equal(seen[0]!.headers.authorization, "Bearer from-env");
    assert.equal(seen[0]!.url, "https://proxy.example.com/v1/chat/completions");
  });
  assert.throws(() => new dmcall.OpenAI("m", { apiKey: "k", baseUrl: "https://user:pass@example.com/v1" }), /baseUrl/);
  assert.throws(() => new dmcall.OpenAI("m", { apiKey: "a\nb" }), /single-line/);
  assert.throws(() => new dmcall.OpenAI("m", { apiKey: "k", maxOutputTokens: 0 }), /maxOutputTokens/);
  assert.throws(() => new dmcall.OpenAI("m", { apiKey: "k", reasoningEfort: "none" } as dmcall.OpenAICompatibleOptions),
    (error: Error) => error instanceof TypeError && /reasoningEfort/.test(error.message));
});

for (const [Cls, envName, url, field] of [
  [dmcall.XAI, "XAI_API_KEY", "https://api.x.ai/v1/chat/completions", "max_tokens"],
  [dmcall.OpenRouter, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", "max_tokens"],
] as const) {
  test(`OpenAI-compatible host: ${Cls.name}`, async () => {
    await withEnv({ [envName]: "k" }, async () => {
      const seen: Seen[] = [];
      await new Cls("m", { fetch: mockFetch(() => respond(chat()), seen) }).generate(EXAMPLE);
      assert.equal(seen[0]!.url, url);
      assert.equal(body(seen[0]!)[field], 4096);
    });
  });
}

test("Meta Muse request", async () => {
  await withEnv({ MODEL_API_KEY: SECRET }, async () => {
    const seen: Seen[] = [];
    const model = new dmcall.Meta("muse-spark-1.3", {
      fetch: mockFetch(() => respond(chat({ calls: [["get_weather", { city: "Seoul" }]] })), seen),
    });
    const proposal = await model.generate(EXAMPLE, HISTORY);
    const request = body(seen[0]!);
    assert.equal(seen[0]!.url, "https://api.meta.ai/v1/chat/completions");
    assert.equal(seen[0]!.headers.authorization, `Bearer ${SECRET}`);
    assert.equal(request.model, "muse-spark-1.3");
    assert.equal(request.tool_choice, "auto");
    assert.equal(request.max_completion_tokens, 4096);
    // Meta rejects reasoning_effort "none" for Muse models; neither option is sent unless set.
    assert.ok(!("parallel_tool_calls" in request) && !("reasoning_effort" in request));
    assert.deepEqual(proposal.calls, seoul());
    assert.equal(proposal.provider, "meta");
    assert.ok(!String(model).includes(SECRET) && !inspect(model, { showHidden: true, depth: 5 }).includes(SECRET));
  });
  assert.throws(() => new dmcall.Meta("muse-spark-1.3"), /MODEL_API_KEY/);
  const seen: Seen[] = [];
  const built = dmcall.provider("meta:muse-spark-1.3", {
    apiKey: "k", reasoningEffort: "low", parallelToolCalls: false, fetch: mockFetch(() => respond(chat()), seen),
  });
  assert.ok(built instanceof dmcall.Meta);
  await built.generate(EXAMPLE);
  assert.equal(body(seen[0]!).reasoning_effort, "low"); // sent when set
  assert.equal(body(seen[0]!).parallel_tool_calls, false);
});

test("any OpenAI-compatible server needs no key", async () => {
  const seen: Seen[] = [];
  const model = new dmcall.OpenAICompatible("gemma-4-31b-it", { baseUrl: "http://localhost:8000/v1",
    fetch: mockFetch(() => respond(chat()), seen) });
  await model.generate({ messages: [{ role: "user", content: "hi" }], tools: [] });
  const request = body(seen[0]!);
  assert.equal(seen[0]!.url, "http://localhost:8000/v1/chat/completions");
  assert.ok(!("authorization" in seen[0]!.headers));
  for (const key of ["tools", "tool_choice", "parallel_tool_calls"]) assert.ok(!(key in request)); // no functions, no tool settings
  assert.throws(() => new dmcall.OpenAICompatible("m"), /baseUrl/);
});

test("Azure deployment URL and header", async () => {
  await withEnv({ AZURE_OPENAI_ENDPOINT: "https://res.openai.azure.com/", AZURE_OPENAI_API_KEY: SECRET }, async () => {
    const seen: Seen[] = [];
    await new dmcall.AzureOpenAI("my deployment", { fetch: mockFetch(() => respond(chat()), seen) }).generate(EXAMPLE);
    assert.equal(seen[0]!.url,
      "https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions?api-version=2024-12-01-preview");
    assert.equal(seen[0]!.headers["api-key"], SECRET);
    assert.ok(!("authorization" in seen[0]!.headers));
    assert.throws(() => new dmcall.AzureOpenAI("d", { apiVersion: "latest" }), /apiVersion/);
  });
  await withEnv({ AZURE_OPENAI_API_KEY: "k" }, () => {
    assert.throws(() => new dmcall.AzureOpenAI("d"), /AZURE_OPENAI_ENDPOINT/);
  });
});

const ANTHROPIC_OK = {
  type: "message", role: "assistant", model: "claude-x", stop_reason: "tool_use",
  content: [{ type: "thinking", thinking: "..." }, { type: "text", text: "Checking." },
    { type: "tool_use", id: "t1", name: "get_weather", input: { city: "Seoul" } }],
  usage: { input_tokens: 50, cache_read_input_tokens: 30, cache_creation_input_tokens: 20, output_tokens: 7 },
};

test("Anthropic request and decoding", async () => {
  const seen: Seen[] = [];
  const model = new dmcall.Anthropic("claude-x", { apiKey: SECRET, fetch: mockFetch(() => respond(ANTHROPIC_OK), seen) });
  const proposal = await model.generate(EXAMPLE, HISTORY);
  const request = body(seen[0]!);
  assert.equal(seen[0]!.url, "https://api.anthropic.com/v1/messages");
  assert.equal(seen[0]!.headers["x-api-key"], SECRET);
  assert.equal(seen[0]!.headers["anthropic-version"], "2023-06-01");
  assert.equal(request.system, `${dmcall.SYSTEM_PROMPT}\n\nBe brief.`);
  assert.deepEqual(request.messages.map((message: { role: string }) => message.role), ["user", "user"]);
  assert.deepEqual(request.tools[0].input_schema, WEATHER.function.parameters);
  assert.deepEqual(request.tool_choice, { type: "auto", disable_parallel_tool_use: true });
  assert.deepEqual(proposal.calls, seoul());
  assert.equal(proposal.content, "Checking.");
  assert.deepEqual(proposal.usage, { inputTokens: 100, cachedInputTokens: 30, outputTokens: 7 });
  for (const [stop, incomplete] of [["max_tokens", "length"], ["refusal", "refusal"]]) {
    const cut = { ...ANTHROPIC_OK, stop_reason: stop, content: [] };
    const cutModel = new dmcall.Anthropic("c", { apiKey: "k", fetch: mockFetch(() => respond(cut)) });
    assert.equal((await cutModel.generate(EXAMPLE)).incomplete, incomplete);
  }
});

test("Gemini request and decoding", async () => {
  const seen: Seen[] = [];
  const reply = {
    modelVersion: "gemini-x-001",
    candidates: [{ finishReason: "STOP", content: { role: "model", parts: [{ text: "thinking", thought: true },
      { functionCall: { name: "get_weather", args: { city: "Seoul" } } }] } }],
    usageMetadata: { promptTokenCount: 80, candidatesTokenCount: 6, thoughtsTokenCount: 40 },
  };
  const model = new dmcall.Gemini("models/gemini-x", { apiKey: SECRET, temperature: 0, thinkingBudget: 0,
    fetch: mockFetch(() => respond(reply), seen) });
  const proposal = await model.generate({ ...EXAMPLE, messages: [...EXAMPLE.messages,
    { role: "assistant", content: "Which unit?" }, { role: "user", content: "Celsius." }] });
  const request = body(seen[0]!);
  assert.equal(seen[0]!.url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent");
  assert.equal(seen[0]!.headers["x-goog-api-key"], SECRET);
  assert.equal(request.systemInstruction.parts[0].text, `${dmcall.SYSTEM_PROMPT}\n\nBe brief.`);
  assert.deepEqual(request.contents.map((content: { role: string }) => content.role), ["user", "model", "user"]);
  const declaration = request.tools[0].functionDeclarations[0];
  assert.deepEqual(declaration.parametersJsonSchema, WEATHER.function.parameters);
  assert.ok(!("parameters" in declaration));
  assert.deepEqual(request.toolConfig, { functionCallingConfig: { mode: "AUTO" } });
  assert.deepEqual(request.generationConfig, { maxOutputTokens: 4096, temperature: 0, thinkingConfig: { thinkingBudget: 0 } });
  assert.deepEqual(proposal.calls, seoul());
  assert.equal(proposal.content, null);
  assert.equal(proposal.model, "gemini-x-001");
  assert.deepEqual(proposal.usage, { inputTokens: 80, outputTokens: 46 });
});

for (const [name, reply, incomplete] of [
  ["length", { candidates: [{ finishReason: "MAX_TOKENS", content: { parts: [] } }] }, "length"],
  ["safety", { candidates: [{ finishReason: "SAFETY" }] }, "content_filter"],
  ["malformed", { candidates: [{ finishReason: "MALFORMED_FUNCTION_CALL" }] }, "malformed_call"],
  ["blocked prompt", { promptFeedback: { blockReason: "SAFETY" } }, "prompt_filter"],
] as const) {
  test(`Gemini unreadable output: ${name}`, async () => {
    const model = new dmcall.Gemini("g", { apiKey: "k", fetch: mockFetch(() => respond(reply)) });
    const proposal = await model.generate(EXAMPLE);
    assert.equal(proposal.calls, null);
    assert.equal(proposal.incomplete, incomplete);
  });
}

test("provider specs", async () => {
  await withEnv({ OPENAI_API_KEY: "k", ANTHROPIC_API_KEY: "k", GEMINI_API_KEY: "k", XAI_API_KEY: "k",
    OPENROUTER_API_KEY: "k", MODEL_API_KEY: "k" }, () => {
    const expected: [string, abstract new (...args: never[]) => dmcall.Provider][] = [
      ["openai:gpt-5-nano", dmcall.OpenAI], ["anthropic:claude-x", dmcall.Anthropic], ["claude:c", dmcall.Anthropic],
      ["gemini:g", dmcall.Gemini], ["google:g", dmcall.Gemini], ["xai:grok-4", dmcall.XAI], ["grok:grok-4", dmcall.XAI],
      ["openrouter:meta/llama", dmcall.OpenRouter], ["meta:muse-spark-1.3", dmcall.Meta],
    ];
    for (const [spec, Cls] of expected) {
      const built = dmcall.provider(spec) as dmcall.Provider;
      assert.equal(built.constructor, Cls);
      assert.equal(built.model, spec.slice(spec.indexOf(":") + 1));
    }
  });
  assert.ok(dmcall.provider("azure:dep", { endpoint: "https://r.openai.azure.com", apiKey: "k" }) instanceof dmcall.AzureOpenAI);
  for (const bad of ["gpt-5-nano", "openai:", "mystery:model"]) assert.throws(() => dmcall.provider(bad), Error);
  assert.throws(() => dmcall.provider("muse-spark-1.3"), /"meta:muse-spark-1\.3"/);
  const standIn = { generate: () => ({ calls: [], content: null, parseErrors: [], incomplete: null }) };
  assert.equal(dmcall.provider(standIn), standIn);
  assert.throws(() => dmcall.provider(standIn, { apiKey: "k" }), Error);
  assert.throws(() => dmcall.provider(42 as never), TypeError);
});
