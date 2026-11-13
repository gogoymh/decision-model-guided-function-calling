/** Regression coverage for issues found in the public-package review. All HTTP is local or mocked. */
import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";

import * as dmcall from "../src/index.js";
import { compiledSchema } from "../src/jsonschema.js";
import { fromPlain } from "../src/json.js";
import { WEATHER, answer, isolateEnvironment, mockFetch, respond, seoul } from "./helpers.js";

isolateEnvironment();

async function listen(server: Server): Promise<string> {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

async function close(server: Server): Promise<void> {
  const closed = new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  server.closeAllConnections();
  await closed;
}

test("native fetch never follows provider redirects with credential headers", async () => {
  let targetRequests = 0;
  const target = createServer((_request, reply) => { targetRequests++; reply.end("{}"); });
  const targetUrl = await listen(target);
  const source = createServer((_request, reply) => {
    reply.writeHead(307, { location: `${targetUrl}/capture` });
    reply.end();
  });
  const sourceUrl = await listen(source);
  try {
    const models = [
      new dmcall.Anthropic("m", { apiKey: "TEST-SECRET", baseUrl: sourceUrl, maxRetries: 0 }),
      new dmcall.Gemini("m", { apiKey: "TEST-SECRET", baseUrl: sourceUrl, maxRetries: 0 }),
      new dmcall.AzureOpenAI("m", { apiKey: "TEST-SECRET", endpoint: sourceUrl, maxRetries: 0 }),
    ];
    for (const model of models) {
      await assert.rejects(model.generate({ messages: [{ role: "user", content: "hi" }] }),
        (error: unknown) => error instanceof dmcall.ProviderError && error.statusCode === 307);
    }
    assert.equal(targetRequests, 0);
  } finally {
    await close(source);
    await close(target);
  }
});

test("run refuses missing handlers before generating, even if the model would make no call", async () => {
  let requested = 0;
  const model = { generate: () => { requested++; return { calls: [], content: null, parseErrors: [], incomplete: null }; } };
  const agent = new dmcall.DMCall(model, { tools: [WEATHER], decisionModel: new dmcall.Jev("KEY") });
  await assert.rejects(agent.run("hi"), /No function/);
  assert.equal(requested, 0);
  assert.equal((await agent.call("hi")).initialNoCallPreserved, true);
  assert.equal(requested, 1);
});

test("replacement definitions retain registered handlers and explicit replacements take precedence", async () => {
  const model = { generate: () => ({ calls: seoul(), content: null, parseErrors: [], incomplete: null }) };
  const decisionModel = new dmcall.Jev("KEY", { fetch: mockFetch(() => respond(answer(0.99, 0.99))) });
  const tool = dmcall.tool({ ...WEATHER.function, execute: async ({ city }) => `old:${city}` });
  const agent = new dmcall.DMCall(model, { tools: [tool], decisionModel });
  assert.equal((await agent.run("Weather?", { tools: [WEATHER] })).output, "old:Seoul");
  assert.equal((await agent.run("Weather?", { tools: [WEATHER],
    functions: { get_weather: ({ city }) => `new:${city}` } })).output, "new:Seoul");
  const own = dmcall.tool({ ...WEATHER.function, execute: ({ city }) => `tool:${city}` });
  assert.equal((await agent.run("Weather?", { tools: [own] })).output, "tool:Seoul");
  await assert.rejects(agent.run("Weather?", { tools: [{ name: "other" }] }), /No function/);
});

test("per-request tools null uses the client's tools as in Python", async () => {
  const model = { generate: (example: dmcall.Example) => {
    assert.deepEqual(example.tools, [WEATHER]);
    return { calls: seoul(), content: null, parseErrors: [], incomplete: null };
  } };
  const decisionModel = new dmcall.Jev("KEY", { fetch: mockFetch(() => respond(answer(0.99, 0.99))) });
  const agent = new dmcall.DMCall(model, { tools: [WEATHER], functions: { get_weather: ({ city }) => city }, decisionModel });
  assert.equal((await agent.call("Weather?", { tools: null })).accepted, true);
  assert.equal((await agent.run("Weather?", { tools: null })).output, "Seoul");
  assert.equal((await agent.verify("Weather?", seoul(), null)).accepted, true);
});

test("provider generate overrides are honored by DMCall", async () => {
  class Customized extends dmcall.OpenAI {
    requests = 0;
    override async generate(): Promise<dmcall.Proposal> {
      this.requests++;
      return { calls: [], content: "custom", parseErrors: [], incomplete: null };
    }
  }
  const model = new Customized("m", { apiKey: "KEY", fetch: mockFetch(() => assert.fail("override handles generation")) });
  const result = await new dmcall.DMCall(model, { decisionModel: new dmcall.Jev("KEY") }).call("hi");
  assert.equal(model.requests, 1);
  assert.equal(result.text, "custom");
});

test("ask overrides can redact state before calling the standard transport", async () => {
  let body: any;
  class Redacted extends dmcall.Jev {
    override async ask(state: dmcall.DecisionState, questions: dmcall.Questions): Promise<dmcall.Answer> {
      state.messages.splice(0, state.messages.length, { role: "user", content: "redacted" });
      return super.ask(state, questions);
    }
  }
  const judge = new Redacted("KEY", { fetch: mockFetch((request) => {
    body = JSON.parse(request.body!);
    return respond(answer(0.99, 0.99));
  }) });
  await judge.judge([{ role: "user", content: "private message" }], [WEATHER], seoul());
  assert.deepEqual(body.state.messages, [{ role: "user", content: "redacted" }]);
  assert.ok(!JSON.stringify(body).includes("private message"));
});

test("schema preparation does not cache a stale schema after caller edits", () => {
  const tool = { name: "f", parameters: { type: "object", properties: { x: { type: "string" } } } };
  const calls = [{ name: "f", arguments: { x: "one" } }];
  assert.equal(dmcall.checkSchema([tool], calls).accepted, true);
  tool.parameters.properties.x.type = "integer";
  assert.equal(dmcall.checkSchema([tool], calls).accepted, false);
  assert.equal(dmcall.checkSchema([tool], [{ name: "f", arguments: { x: 1 } }]).accepted, true);
  assert.throws(() => compiledSchema(fromPlain({ type: "object", $ref: "#" })));
});

test("references to data or schema containers fail before any model request", async () => {
  let requests = 0;
  const model = { generate: () => { requests++; return { calls: [], content: null, parseErrors: [], incomplete: null }; } };
  const decisionModel = new dmcall.Jev("KEY", { fetch: mockFetch(() => { requests++; assert.fail("invalid configuration"); }) });
  const agent = new dmcall.DMCall(model, { decisionModel });
  const schemas = [
    { $schema: "http://json-schema.org/draft-04/schema#", type: "object", properties: { x: { $ref: "#/default" } }, default: true },
    { type: "object", properties: { x: { $ref: "#/default" } }, default: { type: "PREMIUM" } },
    { type: "object", properties: { x: { $ref: "#/default" } }, default: { type: "string" } },
    { type: "object", properties: { x: { $ref: "#/properties" }, type: { type: "integer" } } },
    { type: "object", properties: { x: { $ref: "#/default" } },
      default: { type: "object", properties: { child: { $ref: "#/missing" } } } },
  ];
  for (const parameters of schemas) {
    const definition = { name: "f", parameters };
    assert.throws(() => new dmcall.DMCall(model, { decisionModel, tools: [definition] }), /\$ref/);
    await assert.rejects(agent.call("x", { tools: [definition] }), /\$ref/);
  }
  assert.equal(requests, 0);
});

test("Gemini malformed native calls never become preserved no-calls", async () => {
  for (const functionCall of [null, "not-an-object", [], true]) {
    const model = new dmcall.Gemini("m", { apiKey: "KEY", fetch: mockFetch(() => respond({ candidates: [
      { finishReason: "STOP", content: { parts: [{ functionCall }] } },
    ] })) });
    const result = await new dmcall.DMCall(model, { decisionModel: new dmcall.Jev("KEY"), maxRepairs: 0 }).call("hi");
    assert.equal(result.initialNoCallPreserved, false);
    assert.equal(result.calls, null);
    const proposal = await model.generate({ messages: [{ role: "user", content: "hi" }] });
    assert.equal(proposal.incomplete, "malformed_call");
  }
});

test("schema type normalization preserves dependency names and extension data", () => {
  const definition = { name: "f", parameters: {
    type: "OBJECT", properties: { type: { type: "STRING" } },
    dependentRequired: { type: ["PREMIUM"] }, customExtension: { type: "PREMIUM" },
    dependencies: { type: ["PREMIUM"], legacy: { type: "OBJECT", properties: { x: { type: "INTEGER" } } } },
  } };
  assert.deepEqual(dmcall.schemaOf(definition).function.parameters, {
    type: "object", properties: { type: { type: "string" } },
    dependentRequired: { type: ["PREMIUM"] }, customExtension: { type: "PREMIUM" },
    dependencies: { type: ["PREMIUM"], legacy: { type: "object", properties: { x: { type: "integer" } } } },
  });
  assert.equal(definition.parameters.type, "OBJECT");
});
