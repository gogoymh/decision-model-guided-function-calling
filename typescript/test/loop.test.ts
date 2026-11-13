/** The DM-Call loop: acceptance, cumulative repair, the paper's variants, fallback and execution (tests/test_loop.py). */

import assert from "node:assert/strict";
import { test } from "node:test";

import * as dmcall from "../src/index.js";
import type { Call, Generator, HistoryRecord, Proposal } from "../src/index.js";
import {
  WEATHER, answer, busan, isolateEnvironment, mockFetch, noSleep, respond, seoul, withEnv,
} from "./helpers.js";

isolateEnvironment();

type Shown = { round: number; calls: Call[]; feedback: Record<string, any> };

/** A provider stand-in that returns scripted candidates and records the history it saw. */
class ScriptedModel implements Generator {
  readonly proposals: (Call[] | null)[];
  readonly histories: Shown[][] = [];

  constructor(...proposals: (Call[] | null)[]) {
    this.proposals = proposals;
  }

  generate(_example: unknown, history?: readonly HistoryRecord[] | null): Proposal {
    this.histories.push(JSON.parse(JSON.stringify(history ?? [])));
    const calls = this.proposals.shift()!;
    return {
      calls, content: calls && calls.length ? null : "No tool fits.", incomplete: calls === null ? "length" : null,
      parseErrors: calls === null ? ["cut off"] : [], usage: { inputTokens: 10, outputTokens: 2 }, latencyMs: 10,
    };
  }
}

/** Jev over mocked HTTP: arguments are right only for `goodCity`; a no call scores `noCall`. */
class ScriptedJev {
  readonly requests: any[] = [];
  readonly jev: dmcall.Jev;

  constructor(readonly goodCity = "Seoul", readonly fn = 0.95, readonly noCall = 0.05, options: dmcall.JevOptions = {}) {
    this.jev = new dmcall.Jev("jev-key", { fetch: mockFetch((request) => this.reply(request.body!)), sleep: noSleep, ...options });
  }

  reply(text: string): Response {
    const body = JSON.parse(text);
    this.requests.push(body);
    const calls = body.state.proposed_calls;
    if (!calls.length) return respond(answer(this.noCall, null));
    return respond(answer(this.fn, calls[0].arguments.city === this.goodCity ? 0.95 : 0.05));
  }
}

function agent(proposals: (Call[] | null)[], options: dmcall.DMCallOptions & { decision?: ScriptedJev } = {}) {
  const model = new ScriptedModel(...proposals);
  const { decision = new ScriptedJev(), ...rest } = options;
  return { loop: new dmcall.DMCall(model, { decisionModel: decision.jev, tools: [WEATHER], ...rest }), model, decision };
}

function weatherTool(execute: dmcall.ToolHandler): dmcall.Tool {
  return dmcall.tool({
    name: "get_weather",
    description: "Current weather for a city.",
    parameters: { type: "object", properties: { city: { type: "string" }, unit: { type: "string", default: "celsius" } },
      required: ["city"] },
    execute,
  });
}

test("repairs with cumulative feedback, then runs", async () => {
  const ran: string[] = [];
  const getWeather = weatherTool(({ city }) => {
    ran.push(city as string);
    return { city, temp: 21 };
  });
  const model = new ScriptedModel(busan(), [], seoul());
  const decision = new ScriptedJev();
  const result = await new dmcall.DMCall(model, { decisionModel: decision.jev, tools: [getWeather] }).run("Weather in Seoul?");
  assert.deepEqual(result.calls, seoul());
  assert.equal(result.accepted, true);
  assert.equal(result.repairs, 2);
  assert.equal(result.exhausted, false);
  assert.deepEqual(ran, ["Seoul"]);
  assert.equal(result.executed, true);
  assert.deepEqual(result.output, { city: "Seoul", temp: 21 });
  assert.deepEqual(model.histories.map((history) => history.length), [0, 1, 2]);
  assert.deepEqual(model.histories[2]![0]!.calls, busan());
  assert.deepEqual(model.histories[2]![1]!.calls, []);
  assert.deepEqual(model.histories[2]![0]!.feedback.failed_checks, ["arguments"]);
  // Hybrid feedback: the generator reads the score and its label; the trace keeps the score.
  const shown = model.histories[1]![0]!.feedback.checks.arguments;
  assert.equal(shown.probability_yes, 0.05);
  assert.equal(shown.feedback_label, "very likely incorrect");
  assert.ok(!("feedback_label" in (result.trace[0]!.feedback as dmcall.FeedbackRecord).checks.arguments!));
  assert.deepEqual(result.attempts.map((step) => step.accepted), [false, false, true]);
  assert.equal(result.attempts[1]!.argumentsProbability, null); // a no call skips the argument question
  assert.deepEqual(result.usage, {
    generator: { inputTokens: 30, outputTokens: 6 }, decisionModel: { inputTokens: 900, outputTokens: 0 },
  });
  assert.deepEqual(result.probabilities, { function: 0.95, arguments: 0.95 });
  assert.ok(String(result).startsWith('Result(call=get_weather(city="Seoul"), accepted=true, repairs=2'));
});

test("rejected calls do not run unless asked", async () => {
  const ran: string[] = [];
  const getWeather = weatherTool(({ city }) => {
    ran.push(city as string);
  });
  const model = new ScriptedModel(busan(), busan(), busan(), busan());
  const result = await new dmcall.DMCall(model, { decisionModel: new ScriptedJev().jev, tools: [getWeather] })
    .run("Weather in Seoul?");
  assert.equal(result.accepted, false);
  assert.equal(result.exhausted, true);
  assert.equal(result.repairs, 3);
  assert.equal(result.executed, false);
  assert.deepEqual(ran, []);
  const again = new dmcall.DMCall(new ScriptedModel(busan(), busan()), {
    decisionModel: new ScriptedJev().jev, tools: [getWeather], maxRepairs: 1,
  });
  const forced = await again.run("Weather in Seoul?", { executeUnverified: true });
  assert.equal(forced.repairs, 1);
  assert.equal(forced.executed, true);
  assert.deepEqual(ran, ["Busan"]);
});

test("without an accepted candidate the first one is final", async () => {
  const ran: string[] = [];
  const getWeather = weatherTool(({ city }) => {
    ran.push(city as string);
    return city;
  });
  const first = new dmcall.DMCall(new ScriptedModel([], busan()), {
    decisionModel: new ScriptedJev().jev, tools: [getWeather], maxRepairs: 1, preserveInitialNoCall: false,
  });
  let result = await first.run("Weather in Seoul?", { executeUnverified: true });
  assert.ok(result.exhausted && result.accepted === false && result.repairs === 1);
  assert.ok(result.noCall && result.finalAttempt === 0 && !result.executed);
  assert.deepEqual(ran, []);
  assert.equal(result.text, "No tool fits.");
  assert.deepEqual(result.probabilities, { function: 0.05, arguments: null });
  const last = new dmcall.DMCall(new ScriptedModel([], busan()), {
    decisionModel: new ScriptedJev().jev, tools: [getWeather], maxRepairs: 1, onExhaustion: "last",
    preserveInitialNoCall: false,
  });
  result = await last.run("Weather in Seoul?", { executeUnverified: true });
  assert.deepEqual(result.calls, busan());
  assert.equal(result.finalAttempt, 1);
  assert.equal(result.executed, true);
  assert.deepEqual(ran, ["Busan"]);
  const { loop: perCall } = agent([busan(), [], busan(), seoul()], { maxRepairs: 1 });
  assert.deepEqual((await perCall.call("Weather in Seoul?", { onExhaustion: "last" })).calls, []);
  const passed = await perCall.call("Weather in Seoul?");
  assert.deepEqual(passed.calls, seoul());
  assert.ok(passed.accepted && passed.finalAttempt === 1);
  await assert.rejects(perCall.call("x", { onExhaustion: "best" as never }), /onExhaustion/);
});

test("an unreadable candidate is repaired and never run", async () => {
  const { loop, decision } = agent([null, seoul()], { functions: { get_weather: ({ city }) => city } });
  let result = await loop.run("Weather in Seoul?");
  assert.equal(result.attempts[0]!.calls, null);
  assert.ok(result.attempts[0]!.schemaErrors.length > 0);
  assert.equal(result.accepted, true);
  assert.equal(result.output, "Seoul");
  assert.equal(decision.requests.length, 1);
  const { loop: stuck } = agent([null], { maxRepairs: 0, functions: { get_weather: ({ city }) => city } });
  result = await stuck.run("Weather?", { executeUnverified: true });
  assert.equal(result.calls, null);
  assert.equal(result.executed, false);
  assert.ok(String(result).startsWith("Result(unreadable candidate"));
});

test("an initial no call is returned without verification", async () => {
  const preserved = agent([[]], { decision: new ScriptedJev("Seoul", 0.95, 0.97),
    functions: { get_weather: () => assert.fail("a no-call never executes") } });
  let result = await preserved.loop.run("Tell me a joke.");
  assert.ok(result.noCall && result.accepted === null && result.initialNoCallPreserved);
  assert.equal(result.executed, false);
  assert.equal(result.text, "No tool fits.");
  assert.deepEqual(preserved.decision.requests, []);
  assert.equal(result.attempts[0]!.accepted, null);
  assert.equal(preserved.model.histories.length, 1);
  const checked = agent([[]], { decision: new ScriptedJev("Seoul", 0.95, 0.97), preserveInitialNoCall: false });
  result = await checked.loop.call("Tell me a joke.");
  assert.equal(result.accepted, true);
  assert.equal(result.initialNoCallPreserved, false);
  assert.equal(checked.decision.requests.length, 1);
});

test("latest history keeps the real round", async () => {
  const { loop, model } = agent([busan(), [], seoul()], { history: "latest" });
  await loop.call("Weather in Seoul?");
  assert.deepEqual(model.histories.map((history) => history.length), [0, 1, 1]);
  assert.equal(model.histories[2]![0]!.round, 2);
  assert.deepEqual(model.histories[2]![0]!.calls, []);
});

test("single-question variants", async () => {
  const functionOnly = agent([busan()], { checks: "function" });
  let result = await functionOnly.loop.call("Weather in Seoul?"); // the wrong city passes when only the function is checked
  assert.ok(result.accepted && result.repairs === 0);
  assert.deepEqual(Object.keys((result.trace[0]!.feedback as dmcall.FeedbackRecord).checks), ["function_selection"]);
  const argumentsOnly = agent([[], seoul()], { checks: "arguments", preserveInitialNoCall: false });
  result = await argumentsOnly.loop.call("Weather in Seoul?"); // a no call has no arguments to reject
  assert.ok(result.accepted && result.noCall && result.repairs === 0);
});

test("feedback representations reach only the generator", async () => {
  for (const [representation, field] of [["numeric", "threshold"], ["text", "feedback_label"], ["hybrid", "feedback_label"]] as const) {
    const { loop, model } = agent([busan(), seoul()], { feedback: representation });
    const result = await loop.call("Weather in Seoul?");
    const shown = model.histories[1]![0]!.feedback.checks.arguments;
    assert.ok(field in shown);
    assert.equal((result.trace[0]!.feedback as dmcall.FeedbackRecord).checks.arguments!.probability_yes, 0.05);
    assert.equal("probability_yes" in shown, representation !== "text");
  }
});

test("options and defaults", async () => {
  const { loop } = agent([]);
  assert.deepEqual([loop.maxRepairs, loop.checks, loop.history, loop.feedback, loop.onExhaustion, loop.preserveInitialNoCall],
    [3, "both", "cumulative", "hybrid", "first", true]);
  for (const bad of [{ checks: "neither" }, { history: "none" }, { feedback: "words" }, { onExhaustion: "best" },
    { maxRepairs: -1 }, { maxRepairs: 1.5 }, { preserveInitialNoCall: "yes" }]) {
    assert.throws(() => agent([], bad as dmcall.DMCallOptions), Error);
  }
  assert.throws(() => new dmcall.DMCall(new ScriptedModel(), { decisionModel: new ScriptedJev().jev, threshold: 0.9 }),
    /decision model itself/);
  assert.throws(() => new dmcall.DMCall(new ScriptedModel(), { decisionModel: {} as dmcall.Judge }),
    (error: Error) => error instanceof TypeError && /judge/.test(error.message));
  assert.throws(() => new dmcall.DMCall(new ScriptedModel()), /TYPESAFE_API_KEY/);
  await withEnv({ TYPESAFE_API_KEY: "jev-env", OPENAI_API_KEY: "openai-env" }, () => {
    const built = new dmcall.DMCall("openai:gpt-5-nano", { threshold: 0.9, reasoningEffort: "none" });
    assert.ok(built.model instanceof dmcall.OpenAI);
    assert.equal((built.model as dmcall.OpenAI).reasoningEffort, "none");
    const jev = built.decisionModel as dmcall.Jev;
    assert.equal(jev.functionThreshold, 0.9);
    assert.equal(jev.argumentsThreshold, 0.9);
    const defaults = new dmcall.DMCall("openai:gpt-5-nano"); // the paper's settings by default
    assert.ok(defaults.decisionModel instanceof dmcall.Jev);
    assert.equal((defaults.decisionModel as dmcall.Jev).model, "jev-1.13.0");
    assert.equal((defaults.decisionModel as dmcall.Jev).functionThreshold, 0.5);
    assert.equal((defaults.decisionModel as dmcall.Jev).argumentsThreshold, 0.5);
    assert.throws(() => new dmcall.DMCall(new ScriptedModel(), { apiKey: "k" }), /provider itself/);
  });
});

test("per-call overrides and missing functions", async () => {
  const { loop } = agent([busan(), seoul()]);
  const result = await loop.call("Weather in Seoul?", { maxRepairs: 0 });
  assert.ok(result.repairs === 0 && result.accepted === false && result.exhausted);
  await assert.rejects(loop.run("Weather in Seoul?", { tools: [WEATHER] }), /No function for tool/);
  await assert.rejects(loop.call("x", { functions: { nope: () => null } }), /not one of the tools/);
});

test("call, run and verify are asynchronous; async handlers are awaited", async () => {
  const { loop } = agent([seoul(), seoul()], {
    functions: { get_weather: async ({ city }) => (city as string).toUpperCase() },
  });
  const called = await loop.call("Weather in Seoul?");
  const ran = await loop.run("Weather in Seoul?");
  const verdict = await loop.verify("Weather in Seoul?", seoul());
  assert.ok(called.accepted && !called.executed);
  assert.equal(ran.output, "SEOUL");
  assert.equal(verdict.accepted, true);
});

test("messages are validated", async () => {
  const { loop } = agent([seoul(), seoul()]);
  await loop.call([{ role: "user", content: [{ type: "text", text: "Weather" }, { type: "text", text: "in Seoul?" }] }]);
  for (const bad of [[], [{ role: "tool", content: "{}" }], [{ role: "user", content: { x: 1 } }], 3]) {
    await assert.rejects(loop.call(bad as dmcall.Messages), Error);
  }
});

test("verify checks calls from any SDK", async () => {
  const decision = new ScriptedJev();
  const shapes = [
    { name: "get_weather", arguments: { city: "Seoul" } },
    { id: "c1", type: "function", function: { name: "get_weather", arguments: '{"city": "Seoul"}' } },
    { type: "tool_use", id: "t1", name: "get_weather", input: { city: "Seoul" } },
    { functionCall: { name: "get_weather", args: { city: "Seoul" } } },
  ];
  for (const shape of shapes) {
    const verdict = await dmcall.verify("Weather in Seoul?", [shape], [WEATHER], { decisionModel: decision.jev });
    assert.equal(verdict.accepted, true);
    assert.deepEqual(verdict.calls, seoul());
    assert.equal(verdict.argumentsProbability, 0.95);
  }
  assert.equal((await dmcall.verify("Weather in Seoul?", busan(), [WEATHER], { decisionModel: decision.jev })).accepted, false);
  const count = decision.requests.length;
  const malformed = await dmcall.verify("Weather?", { name: "get_weather", arguments: "{oops" }, [WEATHER],
    { decisionModel: decision.jev });
  assert.equal(malformed.accepted, false);
  assert.equal(decision.requests.length, count); // rejected locally; nothing is sent
  assert.deepEqual((await dmcall.verify("Tell me a joke.", null, [WEATHER], { decisionModel: decision.jev })).calls, []);
  assert.ok((await dmcall.verify("Tell me a joke.", [], [WEATHER],
    { decisionModel: new ScriptedJev("Seoul", 0.95, 0.9).jev })).accepted);

  class SDKMessage { // what an SDK may return for an assistant turn with a tool call
    toJSON() {
      return { role: "assistant", content: null, refusal: null, tool_calls: [
        { id: "c0", type: "function", function: { name: "get_weather", arguments: '{"city": "Busan"}' } }] };
    }
  }
  const history = [{ role: "user", content: "Weather in Busan, then Seoul?" }, new SDKMessage(),
    { role: "tool", tool_call_id: "c0", content: '{"temp": 18}' }];
  assert.ok((await dmcall.verify(history, seoul(), [WEATHER], { decisionModel: decision.jev })).accepted);
  const sent = decision.requests[decision.requests.length - 1].state.messages;
  assert.deepEqual(sent.map((message: { role: string }) => message.role), ["user", "assistant", "tool"]);
  assert.ok(!("refusal" in sent[1]));
  await assert.rejects(dmcall.verify([{ role: "robot", content: "x" }], seoul(), [WEATHER], { decisionModel: decision.jev }),
    Error);
  await assert.rejects(dmcall.verify("x", seoul(), [WEATHER], { decisionModel: decision.jev, threshold: 0.9 }),
    /decision model itself/);
  const { loop } = agent([], { decision });
  assert.equal((await loop.verify("Weather in Seoul?", seoul())).accepted, true);
});

test("a custom decision model: subclass DecisionModel and override ask()", async () => {
  class Rules extends dmcall.DecisionModel {
    readonly seen: dmcall.DecisionState[] = [];
    constructor() {
      super("https://rules.invalid", { model: "rules-1" });
    }
    override async ask(state: dmcall.DecisionState, questions: dmcall.Questions): Promise<dmcall.Answer> {
      this.seen.push(state);
      const call = state.proposed_calls[0];
      const probabilities: Record<string, number> = { function_correct: call?.name === "get_weather" ? 0.9 : 0.1 };
      if ("arguments_correct" in questions) probabilities.arguments_correct = call?.arguments.city === "Seoul" ? 0.9 : 0.1;
      return { probabilities, model: "rules-1", usage: { inputTokens: null, outputTokens: null }, apiCalls: 0, retryWaitMs: 0 };
    }
  }
  const rules = new Rules();
  const result = await new dmcall.DMCall(new ScriptedModel(busan(), seoul()), { decisionModel: rules, tools: [WEATHER] })
    .call("Weather in Seoul?");
  assert.ok(result.accepted && result.repairs === 1);
  assert.deepEqual(rules.seen[0]!.proposed_calls, busan());
  assert.equal(rules.seen[0]!.tools[0]!.name, "get_weather");
  assert.deepEqual(result.usage.decisionModel, {});
});

test("any object with judge() can decide; it receives plain objects", async () => {
  const seen: unknown[] = [];
  const judge: dmcall.Judge = {
    async judge(messages, tools, calls) {
      seen.push({ messages, tools, calls });
      return new ScriptedJev().jev.judge(messages, tools, calls);
    },
  };
  const result = await new dmcall.DMCall(new ScriptedModel(busan(), seoul()), { decisionModel: judge, tools: [WEATHER] })
    .call("Weather in Seoul?");
  assert.ok(result.accepted);
  const first = seen[0] as { messages: unknown; tools: { function: { name: string } }[]; calls: unknown };
  assert.deepEqual(first.calls, busan());
  assert.equal(first.tools[0]!.function.name, "get_weather");
  assert.deepEqual(first.messages, [{ role: "user", content: "Weather in Seoul?" }]);
});
