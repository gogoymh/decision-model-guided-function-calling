/** The language-neutral conformance files in ../spec, replayed the way tests/test_spec.py and tests/test_paper_parity.py do. */

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import * as dmcall from "../src/index.js";
import { represent, singleCheck } from "../src/feedback.js";
import { HISTORY_FOOTER, HISTORY_HEADER, REPAIR_INSTRUCTION } from "../src/prompts.js";
import { type Json, compactText, fromPlain, parse, pythonText, stringify, toPlain } from "../src/json.js";
import {
  type Seen, assertJson, at, camelOptions, conformance, entries, isolateEnvironment, list, loadSpec, mockFetch, noSleep,
  pick, respond, serve, summary,
} from "./helpers.js";

isolateEnvironment();

const text = (value: Json) => value as string;

describe("prompts.json", () => {
  test("constants equal the recorded strings and values", () => {
    const data = loadSpec("prompts.json");
    assert.equal(dmcall.SYSTEM_PROMPT, text(at(data, "system_prompt")));
    assert.equal(HISTORY_HEADER, text(at(data, "history_header")));
    assert.equal(HISTORY_FOOTER, text(at(data, "history_footer")));
    assert.equal(REPAIR_INSTRUCTION, text(at(data, "repair_instruction")));
    assertJson(dmcall.FUNCTION_QUESTION, at(data, "function_question"), "function question", true);
    assertJson(dmcall.ARGUMENT_QUESTION, at(data, "argument_question"), "argument question", true);
    assertJson(dmcall.LABELS, at(data, "labels"));
    assertJson(dmcall.CUT_POINTS, at(data, "cut_points"));
    assertJson(dmcall.LABEL_FIELDS, at(data, "label_fields"));
    assertJson([...dmcall.RETRY_STATUSES].sort((a, b) => a - b), at(data, "retry_statuses"));
    assertJson(dmcall.INCOMPLETE, at(data, "incomplete_reasons"));
    const defaults = at(data, "defaults");
    assert.equal(dmcall.DEFAULT_THRESHOLD, toPlain(at(defaults, "threshold")));
    assert.equal(dmcall.DEFAULT_MAX_REPAIRS, toPlain(at(defaults, "max_repairs")));
    const jev = at(data, "jev");
    assert.equal(dmcall.Jev.DEFAULT_MODEL, text(at(jev, "model")));
    assert.equal(dmcall.Jev.DEFAULT_BASE_URL, text(at(jev, "base_url")));
    assert.equal(dmcall.Jev.feedbackSource, text(at(jev, "feedback_source")));
    const pplx = at(data, "pplx_decider");
    assert.equal(dmcall.PPLXDecider.MODEL_ID, text(at(pplx, "model_id")));
    assert.equal(dmcall.PPLXDecider.MODEL_REVISION, text(at(pplx, "revision")));
    assert.equal(dmcall.PPLXDecider.CALIBRATION_TEMPERATURE, toPlain(at(pplx, "calibration_temperature")));
    assert.equal(dmcall.PPLXDecider.MAX_INPUT_TOKENS, toPlain(at(pplx, "max_input_tokens")));
    assert.equal(dmcall.PPLXDecider.SERVER_MODEL, text(at(pplx, "server_model")));
    assert.equal(dmcall.PPLXDecider.WIRE_MODEL, text(at(pplx, "wire_model")));
    assert.equal(dmcall.PPLXDecider.feedbackSource, text(at(pplx, "feedback_source")));
  });

  test("defaults of DMCall and the decision models", () => {
    const defaults = at(loadSpec("prompts.json"), "defaults");
    const generator = { generate: () => ({ calls: [], content: null, parseErrors: [], incomplete: null }) };
    const agent = new dmcall.DMCall(generator, { decisionModel: new dmcall.Jev("k") });
    assert.equal(agent.maxRepairs, toPlain(at(defaults, "max_repairs")));
    assert.equal(agent.checks, text(at(defaults, "checks")));
    assert.equal(agent.history, text(at(defaults, "history")));
    assert.equal(agent.feedback, text(at(defaults, "feedback")));
    assert.equal(agent.onExhaustion, text(at(defaults, "on_exhaustion")));
    assert.equal(agent.preserveInitialNoCall, at(defaults, "preserve_initial_no_call"));
    assert.equal(agent.executeUnverified, at(defaults, "execute_unverified"));
    assert.equal(new dmcall.OpenAI("m", { apiKey: "k" }).maxOutputTokens, toPlain(at(defaults, "max_output_tokens")));
    const pplx = new dmcall.PPLXDecider("http://localhost:8000");
    const recorded = at(loadSpec("prompts.json"), "pplx_decider");
    assert.equal(pplx.timeoutMs, Number(toPlain(at(recorded, "timeout_s"))) * 1000);
    assert.equal(pplx.maxRetries, toPlain(at(recorded, "max_retries")));
  });
});

describe("json_text.json", () => {
  for (const testCase of list(at(conformance("json_text.json"), "cases"))) {
    const input = text(at(testCase, "input"));
    test(input.slice(0, 40), () => {
      assert.equal(pythonText(parse(input)), text(at(testCase, "output")));
      assert.equal(stringify(parse(input), { ensureAscii: false }), text(at(testCase, "output")));
    });
  }
});

describe("schema_check.json", () => {
  const data = conformance("schema_check.json");
  for (const testCase of list(at(data, "cases"))) {
    test(text(at(testCase, "name")), () => {
      const result = dmcall.checkSchema(at(testCase, "tools"), at(testCase, "calls"));
      const snake = {
        function_pass: result.functionPass, arguments_pass: result.argumentsPass, accepted: result.accepted,
        well_formed: result.wellFormed, errors: result.errors, feedback: result.feedback,
      };
      assertJson(snake, at(testCase, "expected"), undefined, true);
    });
  }
  for (const testCase of list(at(data, "configuration_errors"))) {
    test(`configuration error: ${text(at(testCase, "name"))}`, () => {
      assert.throws(() => dmcall.checkSchema(at(testCase, "tools"), []), Error);
    });
  }
});

describe("feedback.json", () => {
  const data = conformance("feedback.json");
  test("labels", () => {
    for (const testCase of list(at(data, "labels"))) {
      assert.equal(dmcall.label(toPlain(at(testCase, "p")) as number | null), toPlain(at(testCase, "label")));
    }
    for (const bad of list(at(data, "invalid_probabilities"))) {
      assert.throws(() => dmcall.label(toPlain(bad) as number), Error);
    }
  });
  test("represent", () => {
    for (const testCase of list(at(data, "represent"))) {
      const representation = text(at(testCase, "representation")) as dmcall.FeedbackRepresentation;
      const out = represent(toPlain(at(testCase, "feedback")) as object, representation);
      assertJson(out, at(testCase, "expected"), representation, true);
    }
  });
  test("single_check", () => {
    for (const testCase of list(at(data, "single_check"))) {
      const keep = text(at(testCase, "keep")) as "function" | "arguments";
      assertJson(singleCheck(toPlain(at(testCase, "feedback")) as object, keep), at(testCase, "expected"), keep, true);
    }
  });
  test("conflicts", () => {
    const conflicts = at(data, "conflicts");
    for (const feedback of list(at(conflicts, "feedback"))) {
      for (const representation of list(at(conflicts, "representations"))) {
        assert.throws(() => represent(toPlain(feedback) as object, text(representation) as dmcall.FeedbackRepresentation),
          Error);
      }
    }
  });
});

describe("normalize_calls.json", () => {
  test("every SDK shape", () => {
    for (const testCase of list(at(conformance("normalize_calls.json"), "cases"))) {
      assertJson(dmcall.normalizeCalls(toPlain(at(testCase, "input"))), at(testCase, "expected"));
      assertJson(dmcall.normalizeCalls(at(testCase, "input")), at(testCase, "expected"));
    }
  });
});

describe("tools.json", () => {
  const data = conformance("tools.json");
  for (const testCase of list(at(data, "cases"))) {
    test(text(at(testCase, "name")), () => {
      const input = at(testCase, "input") as unknown as dmcall.ToolInput;
      assertJson(dmcall.schemaOf(input).function, at(testCase, "expected"), undefined, true);
    });
  }
  for (const testCase of list(at(data, "configuration_errors"))) {
    test(`configuration error: ${text(at(testCase, "name"))}`, () => {
      const input = at(testCase, "input") as unknown as dmcall.ToolInput;
      assert.throws(() => dmcall.schemaOf(input));
      assert.throws(() => new dmcall.DMCall({ generate: () => {
        assert.fail("invalid tools must fail before a request");
      } }, { tools: [input], decisionModel: new dmcall.Jev("KEY") }));
    });
  }
});

const HEADERS = new Set(["authorization", "x-api-key", "api-key", "x-goog-api-key", "anthropic-version", "content-type"]);

function listedHeaders(seen: Seen): Record<string, string> {
  return Object.fromEntries(Object.entries(seen.headers).filter(([name]) => HEADERS.has(name)));
}

describe("providers.json", () => {
  const data = conformance("providers.json");
  const example = toPlain(at(data, "example")) as dmcall.Example;
  const history = toPlain(at(data, "history")) as dmcall.HistoryRecord[];
  const builders: Record<string, [(fetch: dmcall.FetchLike) => dmcall.Provider, unknown, dmcall.Example]> = {
    "openai with history": [(fetch) => new dmcall.OpenAI("gpt-5-nano", { apiKey: "KEY", reasoningEffort: "none", fetch }),
      history, example],
    "openai no tools": [(fetch) => new dmcall.OpenAI("gpt-5-nano", { apiKey: "KEY", fetch }), null,
      { messages: [{ role: "user", content: "hi" }], tools: [] }],
    "openai-compatible local": [(fetch) => new dmcall.OpenAICompatible("gemma-4-31b-it",
      { baseUrl: "http://localhost:8000/v1", fetch }), null, example],
    xai: [(fetch) => new dmcall.XAI("grok-4", { apiKey: "KEY", fetch }), null, example],
    openrouter: [(fetch) => new dmcall.OpenRouter("meta/llama", { apiKey: "KEY", temperature: 0.2, fetch }), null, example],
    "meta muse": [(fetch) => new dmcall.Meta("muse-spark-1.3", { apiKey: "KEY", fetch }), history, example],
    azure: [(fetch) => new dmcall.AzureOpenAI("my deployment", { endpoint: "https://res.openai.azure.com/", apiKey: "KEY",
      fetch }), history, example],
    "anthropic with history": [(fetch) => new dmcall.Anthropic("claude-sonnet-5-5", { apiKey: "KEY", fetch }), history,
      example],
    gemini: [(fetch) => new dmcall.Gemini("models/gemini-2.5-flash", { apiKey: "KEY", temperature: 0, thinkingBudget: 0,
      fetch }), history, example],
  };
  for (const recorded of list(at(data, "requests"))) {
    const name = text(at(recorded, "name"));
    test(`request: ${name}`, async () => {
      const builder = builders[name];
      assert.ok(builder, `no builder for ${name}`);
      const seen: Seen[] = [];
      const model = builder[0](mockFetch(() => respond({}), seen));
      await assert.rejects(model.generate(builder[2], builder[1] as dmcall.HistoryRecord[] | null), dmcall.ProviderError);
      assert.equal(seen.length, 1);
      assert.equal(seen[0]!.url, text(at(recorded, "url")));
      assertJson(listedHeaders(seen[0]!), at(recorded, "headers"), "headers");
      assertJson(parse(seen[0]!.body!), at(recorded, "body"), "body", true);
      // httpx 0.28 sends json.dumps(body, ensure_ascii=False, separators=(",", ":")): the same bytes.
      assert.equal(seen[0]!.body, compactText(at(recorded, "body")));
    });
  }

  const classes: Record<string, typeof dmcall.OpenAI | typeof dmcall.Anthropic | typeof dmcall.Gemini> = {
    openai: dmcall.OpenAI, anthropic: dmcall.Anthropic, gemini: dmcall.Gemini,
  };
  for (const testCase of list(at(data, "decode"))) {
    test(`decode: ${text(at(testCase, "name"))}`, async () => {
      const Cls = classes[text(at(testCase, "provider"))]!;
      const model = new Cls("m", { apiKey: "KEY", fetch: serve([at(testCase, "response")]) });
      const proposal = await model.generate(example);
      const snake: Record<string, unknown> = {
        calls: proposal.calls, content: proposal.content, parse_errors: proposal.parseErrors,
        incomplete: proposal.incomplete, usage: dmcallUsage(proposal.usage), model: proposal.model,
      };
      assertJson(pick(snake, at(testCase, "expected")), at(testCase, "expected"));
    });
  }

  test("a filtered prompt is an unreadable candidate", async () => {
    const filtered = at(data, "filtered_prompt");
    const model = new dmcall.OpenAI("m", {
      apiKey: "KEY", fetch: mockFetch(() => respond(at(filtered, "response"), Number(toPlain(at(filtered, "status"))))),
    });
    const proposal = await model.generate(example);
    assert.equal(proposal.calls, null);
    assert.equal(proposal.incomplete, text(at(filtered, "expected_incomplete")));
    assert.deepEqual(proposal.usage, {});
  });
});

function dmcallUsage(usage: dmcall.Usage | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(usage ?? {})) {
    if (value !== undefined) out[key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)] = value;
  }
  return out;
}

describe("paper_parity.json", () => {
  const golden = conformance("paper_parity.json");
  const tools = at(golden, "tools") as unknown as dmcall.ToolInput[];
  const messages = toPlain(at(golden, "messages")) as dmcall.Messages;
  const generatorKey = text(at(golden, "generator_api_key"));
  const jevKey = text(at(golden, "jev_api_key"));

  for (const [name, scenario] of entries(at(golden, "scenarios"))) {
    test(name, async () => {
      const sentGenerator: Seen[] = [];
      const sentDecision: Seen[] = [];
      const generatorFetch = serve(list(at(scenario, "generator_responses")), sentGenerator);
      const jev = new dmcall.Jev(jevKey, { fetch: serve(list(at(scenario, "decision_responses")), sentDecision) });
      const model = text(at(scenario, "provider")) === "anthropic"
        ? new dmcall.Anthropic("claude-sonnet-5-5", { apiKey: generatorKey, fetch: generatorFetch })
        : new dmcall.OpenAI("gpt-5-nano", { apiKey: generatorKey, reasoningEffort: "none", fetch: generatorFetch });
      const agent = new dmcall.DMCall(model, { decisionModel: jev, tools, ...camelOptions(at(scenario, "options")) });
      const result = await agent.call(messages);

      const generatorRequests = list(at(scenario, "generator_requests"));
      assert.equal(sentGenerator.length, generatorRequests.length, "generator requests");
      sentGenerator.forEach((request, index) => {
        assertJson(parse(request.body!), generatorRequests[index]!, `generator request ${index}`, true);
        // The history message is sent as text, so its exact characters matter, not only its JSON.
        assert.equal(pythonText(parse(request.body!)), pythonText(generatorRequests[index]!));
        assert.equal(request.body, compactText(generatorRequests[index]!), `generator request ${index} bytes`);
      });
      const decisionRequests = list(at(scenario, "decision_requests"));
      assert.equal(sentDecision.length, decisionRequests.length, "decision requests");
      sentDecision.forEach((request, index) => {
        assertJson(parse(request.body!), decisionRequests[index]!, `decision request ${index}`, true);
        assert.equal(request.body, compactText(decisionRequests[index]!), `decision request ${index} bytes`);
      });
      const expected = at(scenario, "result");
      assertJson(pick(summary(result), expected), expected, "result");
    });
  }

  test("verify", async () => {
    const recorded = at(golden, "verify");
    const sent: Seen[] = [];
    const jev = new dmcall.Jev(jevKey, { fetch: serve(list(at(recorded, "decision_responses")), sent) });
    const calls = list(at(recorded, "calls"));
    const verdicts = [
      await dmcall.verify(toPlain(at(recorded, "messages")), toPlain(calls[0]!), tools, { decisionModel: jev }),
      await dmcall.verify(text(at(recorded, "second_messages")), toPlain(calls[1]!), tools, { decisionModel: jev }),
    ];
    const requests = list(at(recorded, "decision_requests"));
    assert.equal(sent.length, requests.length);
    sent.forEach((request, index) => {
      assertJson(parse(request.body!), requests[index]!, `request ${index}`, true);
      assert.equal(request.body, compactText(requests[index]!), `request ${index} bytes`);
    });
    assertJson(verdicts.map((verdict) => ({
      accepted: verdict.accepted, calls: verdict.calls, function_probability: verdict.functionProbability,
      arguments_probability: verdict.argumentsProbability, schema_errors: verdict.schemaErrors, feedback: verdict.feedback,
    })), at(recorded, "verdicts"), "verdicts");
  });
});

describe("loop_edges.json", () => {
  const data = conformance("loop_edges.json");
  const tools = at(data, "tools") as unknown as dmcall.ToolInput[];
  const messages = toPlain(at(data, "messages")) as dmcall.Messages;

  function generator(spec: Json, fetch: dmcall.FetchLike): dmcall.Provider {
    const kind = text(at(spec, "kind"));
    const model = text(at(spec, "model"));
    const options = { ...camelOptions(spec instanceof Map && spec.has("options") ? at(spec, "options") : new Map()), fetch };
    switch (kind) {
      case "openai": return new dmcall.OpenAI(model, options);
      case "openai-compatible": return new dmcall.OpenAICompatible(model, options);
      case "azure": return new dmcall.AzureOpenAI(model, options);
      case "xai": return new dmcall.XAI(model, options);
      case "anthropic": return new dmcall.Anthropic(model, options);
      case "gemini": return new dmcall.Gemini(model, options);
      default: throw new Error(`unknown generator kind ${kind}`);
    }
  }

  function decision(spec: Json, fetch: dmcall.FetchLike): dmcall.DecisionModel {
    if (text(at(spec, "kind")) === "pplx") return new dmcall.PPLXDecider(text(at(spec, "base_url")), { fetch });
    return new dmcall.Jev("JEV-KEY", { fetch, sleep: noSleep });
  }

  for (const [name, scenario] of entries(at(data, "scenarios"))) {
    test(name, async () => {
      const sentGenerator: Seen[] = [];
      const sentDecision: Seen[] = [];
      const model = generator(at(scenario, "generator"), serve(list(at(scenario, "generator_responses")), sentGenerator));
      const judge = decision(at(scenario, "decision_model"), serve(list(at(scenario, "decision_responses")), sentDecision));
      const agent = new dmcall.DMCall(model, { decisionModel: judge, tools, ...camelOptions(at(scenario, "options")) });
      const result = await agent.call(messages);
      for (const [sent, key] of [[sentGenerator, "generator_requests"], [sentDecision, "decision_requests"]] as const) {
        const recorded = list(at(scenario, key));
        assert.equal(sent.length, recorded.length, key);
        sent.forEach((request, index) => {
          const expected = recorded[index]!;
          assert.equal(request.url, text(at(expected, "url")), `${key}[${index}] url`);
          assertJson(listedHeaders(request), at(expected, "headers"), `${key}[${index}] headers`);
          assertJson(parse(request.body!), at(expected, "body"), `${key}[${index}] body`, true);
          assert.equal(request.body, compactText(at(expected, "body")), `${key}[${index}] bytes`);
        });
      }
      assertJson(summary(result), at(scenario, "result"), "result");
    });
  }
});

test("fromPlain keeps conformance values intact", () => {
  const value = conformance("json_text.json");
  assert.equal(pythonText(fromPlain(value)), pythonText(value));
});
