/**
 * Decision models: D in the paper.
 *
 * A decision model answers two yes/no questions about a candidate call and returns the probability
 * of yes for each: the function question D_f (is this the right function, or is making no call
 * right?) and the argument question D_a (are the arguments present, supported by the request and in
 * the right parameters?). The argument question is not asked for a no call, and neither question is
 * asked for an unreadable candidate; such a score is `null`.
 *
 * Both decision models of the paper speak the same HTTP protocol (`POST /v1/systemone`):
 * {@link Jev}, TypeSafe's hosted decision model (`jev-1.13.0`), and {@link PPLXDecider}, the
 * open-weight Perplexity Decider v1.1 27B served on your own machine.
 *
 * Only the request, the function schemas and the candidate are sent; never a reference answer. To
 * plug in another decision model, subclass {@link DecisionModel} and override `ask()`.
 */

import { type Json, type JsonObject, Float, clone, compactText, fromPlain, parse, pythonStr, stringRepr, toPlain } from "./json.js";
import {
  type FetchLike, type HttpResponse, type Sleep, checkBaseUrl, defaultSleep, env, now, pythonFloat, requestId,
  resolveFetch, send, urlsplit, wait,
} from "./http.js";
import { ARGUMENT_QUESTION, FUNCTION_QUESTION, type Question, REPAIR_INSTRUCTION } from "./prompts.js";
import { type SchemaIssue, checkSchemaJson, functionDefinitions } from "./schema.js";

export const RETRY_STATUSES: ReadonlySet<number> = new Set([408, 429, 500, 502, 503, 504, 529]);
const OBSERVABLE_KEYS = ["role", "content", "name", "tool_calls", "tool_call_id"];

/** A transport or protocol failure. It is never a rejection, and it never holds keys or bodies. */
export class DecisionModelError extends Error {
  code: string;
  statusCode: number | null;
  requestId: string | null;
  attempts: number | null;

  constructor(message: string, details: { code?: string; statusCode?: number | null; requestId?: string | null;
    attempts?: number | null } = {}) {
    super(message);
    this.name = "DecisionModelError";
    this.code = details.code ?? "decision_model_error";
    this.statusCode = details.statusCode ?? null;
    this.requestId = details.requestId ?? null;
    this.attempts = details.attempts ?? null;
  }
}

/** What the decision model sees: the conversation, the function definitions and the candidate. */
export interface DecisionState {
  messages: Record<string, unknown>[];
  tools: { name: string; description: unknown; parameters: unknown }[];
  proposed_calls: { name: string; arguments: Record<string, unknown> }[];
}

/** The questions asked, by answer key: `function_correct` and, unless there is no call, `arguments_correct`. */
export type Questions = Record<string, Question>;

/** What `ask()` returns. */
export interface Answer {
  /** The probability of yes for each question key. */
  probabilities: Record<string, number>;
  /** The model the server reports. */
  model: string;
  usage: { inputTokens: number | null; outputTokens: number | null };
  apiCalls: number;
  retryWaitMs: number;
}

/** One check in a feedback record. */
export interface FeedbackCheck {
  skipped?: boolean;
  pass: boolean | null;
  probability_yes: number | null;
  threshold: number;
  schema_pass: boolean;
}

/** The feedback record d_t the generator reads about a rejected candidate. Field names are part of the protocol. */
export interface FeedbackRecord {
  source: string;
  checks: { function_selection?: FeedbackCheck; arguments?: FeedbackCheck };
  failed_checks: string[];
  schema_errors: SchemaIssue[];
  instruction: string;
}

/** A decision model's judgment of one candidate. */
export interface Judgment {
  functionPass: boolean;
  /** `null` when the argument question was not asked (no call). */
  argumentsPass: boolean | null;
  argumentsSkipped: boolean;
  functionProbability: number | null;
  argumentsProbability: number | null;
  accepted: boolean;
  feedback: FeedbackRecord;
  schemaErrors: SchemaIssue[];
  usage: { inputTokens: number | null; outputTokens: number | null };
  model: string | null;
  requestedModel: string;
  latencyMs: number;
  retryWaitMs: number;
  apiCalls: number;
}

/** Anything that can judge a candidate, like {@link Jev}. */
export interface Judge {
  judge(messages: unknown, tools: unknown, calls: unknown): Promise<Judgment>;
}

/** The {@link Json} form of each state handed to `ask()`, so the default request keeps number kinds. */
const STATES = new WeakMap<object, JsonObject>();

function threshold(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("Thresholds must be numbers between 0 and 1.");
  }
  return value;
}

function retryDelaySeconds(response: HttpResponse | null, attempt: number): number {
  if (response !== null) {
    const value = response.headers.get("Retry-After");
    if (value !== null) {
      let delay = pythonFloat(value);
      if (delay === undefined) {
        const stamp = Date.parse(value);
        delay = Number.isNaN(stamp) ? -1 : (stamp - Date.now()) / 1000;
      }
      if (Number.isFinite(delay) && delay >= 0) return Math.min(delay, 30);
    }
  }
  return Math.min(2 ** attempt, 30);
}

export interface DecisionModelOptions {
  /** The model name sent with each request. */
  model: string;
  apiKey?: string;
  /** Pass mark of both questions (0.5 in the paper). */
  threshold?: number;
  functionThreshold?: number;
  argumentsThreshold?: number;
  timeoutMs?: number;
  /** HTTP retries after the first attempt (not DM-Call's repair limit). */
  maxRetries?: number;
  fetch?: FetchLike;
  sleep?: Sleep;
}

/**
 * A decision model reached over the `/v1/systemone` protocol.
 *
 * `threshold` is the pass mark of both questions (0.5 in the paper); `functionThreshold` and
 * `argumentsThreshold` override one of them. `maxRetries` counts HTTP retries after the first
 * attempt and has nothing to do with DM-Call's repair limit.
 */
export class DecisionModel implements Judge {
  /** Name used in error messages. */
  static displayName = "Decision model";
  /** The `source` field of the feedback records the generator reads. */
  static feedbackSource = "decision_model_noul_and_deterministic_schema_validation";

  readonly functionThreshold: number;
  readonly argumentsThreshold: number;
  readonly baseUrl: string;
  readonly model: string;
  readonly timeoutMs: number;
  readonly maxRetries: number;
  readonly #headers: Record<string, string>;
  readonly #apiKey: string | undefined;
  readonly #fetch: FetchLike;
  readonly #sleep: Sleep;

  constructor(baseUrl: string, options: DecisionModelOptions) {
    const cls = new.target as typeof DecisionModel;
    const { threshold: shared = 0.5, functionThreshold, argumentsThreshold } = options;
    this.functionThreshold = threshold(functionThreshold ?? shared);
    this.argumentsThreshold = threshold(argumentsThreshold ?? shared);
    const maxRetries = options.maxRetries ?? 3;
    if (!Number.isInteger(maxRetries) || maxRetries < 0) throw new Error("maxRetries must be a nonnegative integer.");
    const timeoutMs = options.timeoutMs ?? 60_000;
    if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new Error("timeoutMs must be a positive number of milliseconds.");
    }
    const apiKey = options.apiKey;
    if (apiKey !== undefined && (typeof apiKey !== "string" || !apiKey.trim() || apiKey.includes("\n")
      || apiKey.includes("\r"))) {
      throw new Error("API keys must be nonempty single-line strings.");
    }
    if (typeof options.model !== "string" || !options.model.trim()) throw new Error(`${cls.displayName} needs a model name.`);
    const url = checkBaseUrl(baseUrl);
    if (!url) throw new Error(`${cls.name} needs an http(s) baseUrl without credentials, query or fragment.`);
    this.baseUrl = url;
    this.model = options.model;
    this.timeoutMs = timeoutMs;
    this.maxRetries = maxRetries;
    this.#apiKey = apiKey;
    this.#headers = apiKey !== undefined ? { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" }
      : { "Content-Type": "application/json" };
    this.#fetch = resolveFetch(options.fetch);
    this.#sleep = options.sleep ?? defaultSleep;
  }

  protected get displayName(): string {
    return (this.constructor as typeof DecisionModel).displayName;
  }

  toString(): string {
    return `${this.constructor.name}(model=${JSON.stringify(this.model)}, baseUrl=${JSON.stringify(this.baseUrl)}, `
      + `functionThreshold=${this.functionThreshold}, argumentsThreshold=${this.argumentsThreshold})`;
  }

  toJSON(): Record<string, unknown> {
    return { model: this.model, baseUrl: this.baseUrl, functionThreshold: this.functionThreshold,
      argumentsThreshold: this.argumentsThreshold };
  }

  // HTTP ------------------------------------------------------------------------------------------
  protected requestIdOf(response: HttpResponse): string | null {
    return requestId(response.headers, ["x-request-id", "request-id", "apim-request-id"], [this.#apiKey]);
  }

  protected errorFor(response: HttpResponse, attempts: number): DecisionModelError {
    return new DecisionModelError(`${this.displayName} HTTP ${response.status}; no judgment recorded.`, {
      code: "http_error", statusCode: response.status, requestId: this.requestIdOf(response), attempts,
    });
  }

  /** A request to this model's server. */
  protected request(path: string, method: string, body?: string): Promise<HttpResponse> {
    return send(this.#fetch, this.baseUrl + path, method, this.#headers, body, this.timeoutMs);
  }

  /** POST with bounded retries. Returns the body, the attempts and the time spent waiting. */
  protected async post(payload: JsonObject): Promise<{ body: JsonObject; apiCalls: number; retryWaitMs: number }> {
    const text = compactText(payload);
    let waited = 0;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      let response: HttpResponse;
      try {
        response = await this.request("/v1/systemone", "POST", text);
      } catch {
        if (attempt < this.maxRetries) {
          waited += await wait(this.#sleep, retryDelaySeconds(null, attempt) * 1000);
          continue;
        }
        throw new DecisionModelError(`${this.displayName} connection failed after ${attempt + 1} attempts.`,
          { code: "connection_failed", attempts: attempt + 1 });
      }
      if (RETRY_STATUSES.has(response.status) && attempt < this.maxRetries) {
        waited += await wait(this.#sleep, retryDelaySeconds(response, attempt) * 1000);
        continue;
      }
      if (response.status < 200 || response.status >= 300) throw this.errorFor(response, attempt + 1);
      let body: Json;
      try {
        body = parse(response.text);
      } catch {
        throw new DecisionModelError(`${this.displayName} response was not valid JSON.`, { code: "response_invalid" });
      }
      if (!(body instanceof Map)) {
        throw new DecisionModelError(`${this.displayName} response must be a JSON object.`, { code: "response_invalid" });
      }
      return { body: this.checked(body), apiCalls: attempt + 1, retryWaitMs: waited };
    }
    throw new Error("unreachable");
  }

  /** Hook for a subclass to check the response's model identity. */
  protected checked(body: JsonObject): JsonObject {
    return body;
  }

  // The protocol ----------------------------------------------------------------------------------
  /**
   * Ask `questions` about `state`; return the probabilities of yes, model, usage and timing.
   *
   * `state` holds `messages`, `tools` and `proposed_calls`. Override this method to connect a
   * decision model that speaks another protocol.
   */
  async ask(state: DecisionState, questions: Questions): Promise<Answer> {
    const original = STATES.get(state);
    const current = fromPlain(state);
    // Keep recorded number kinds only while the state has not been changed by an ask() override.
    const stateJson = original && compactText(fromPlain(toPlain(original))) === compactText(current) ? original : current;
    const payload: JsonObject = new Map<string, Json>([
      ["model", this.model], ["state", stateJson], ["questions", fromPlain(questions)],
    ]);
    const { body, apiCalls, retryWaitMs } = await this.post(payload);
    const answers = body.get("answers");
    const probabilities: Record<string, number> = {};
    for (const key of Object.keys(questions)) {
      const answer = answers instanceof Map ? answers.get(key) : undefined;
      if (!(answer instanceof Map) || answer.get("type") !== "noul") {
        throw new DecisionModelError(`${this.displayName} response is missing the answer ${stringRepr(key)}.`,
          { code: "response_invalid" });
      }
      const value = answer.get("noul");
      const number = typeof value === "bigint" ? Number(value) : value instanceof Float ? value.value : undefined;
      if (number === undefined || !Number.isFinite(number) || number < 0 || number > 1) {
        throw new DecisionModelError(`${this.displayName} returned an invalid probability for ${stringRepr(key)}.`,
          { code: "response_invalid" });
      }
      probabilities[key] = number;
    }
    const model = body.get("model");
    if (typeof model !== "string" || !model) {
      throw new DecisionModelError(`${this.displayName} response is missing the resolved model ID.`, { code: "response_invalid" });
    }
    const usage = body.get("usage");
    if (!(usage instanceof Map)) {
      throw new DecisionModelError(`${this.displayName} response is missing token usage.`, { code: "response_invalid" });
    }
    const counts: { inputTokens: number | null; outputTokens: number | null } = { inputTokens: null, outputTokens: null };
    for (const [kind, target] of [["input_tokens", "inputTokens"], ["output_tokens", "outputTokens"]] as const) {
      const count = usage.get(kind);
      if (count !== undefined && count !== null && (typeof count !== "bigint" || count < 0n)) {
        throw new DecisionModelError(`${this.displayName} response contains invalid token usage.`, { code: "response_invalid" });
      }
      counts[target] = count === undefined || count === null ? null : Number(count);
    }
    return { probabilities, model, usage: counts, apiCalls, retryWaitMs };
  }

  /**
   * Judge one candidate: the schema check S, both questions, verdicts V and the feedback record.
   *
   * `messages` is the conversation (objects with `role` and `content`, plus `tool_calls` or
   * `tool_call_id` for earlier tool turns), `tools` the function definitions and `calls` the
   * candidate as `[{name, arguments}]` (`[]` for no call).
   */
  async judge(messages: unknown, tools: unknown, calls: unknown): Promise<Judgment> {
    const started = now();
    const toolsJson = fromPlain(tools);
    const callsJson = fromPlain(calls);
    const local = checkSchemaJson(toolsJson, callsJson);
    const argumentsSkipped = Array.isArray(callsJson) && callsJson.length === 0;
    let functionProbability: number | null = null;
    let argumentsProbability: number | null = null;
    let usage: Judgment["usage"] = { inputTokens: 0, outputTokens: 0 };
    let resolvedModel: string | null = null;
    let apiCalls = 0;
    let waited = 0;
    if (local.wellFormed) { // an unreadable candidate is not sent to the decision model
      const conversation = fromPlain(messages);
      if (!Array.isArray(conversation)) throw new Error("messages must be a list");
      const observable = conversation.map((message) => {
        if (!(message instanceof Map)) throw new Error("Each message must be an object");
        const kept: JsonObject = new Map();
        for (const key of OBSERVABLE_KEYS) if (message.has(key)) kept.set(key, clone(message.get(key)!));
        return kept;
      });
      const proposed = (callsJson as Json[]).map((call) => {
        const entry = call as JsonObject;
        return new Map<string, Json>([["name", entry.get("name")!], ["arguments", clone(entry.get("arguments")!)]]);
      });
      const stateJson: JsonObject = new Map<string, Json>([
        ["messages", observable], ["tools", functionDefinitions(toolsJson)], ["proposed_calls", proposed],
      ]);
      const questions: Questions = { function_correct: clone(FUNCTION_QUESTION) };
      if (!argumentsSkipped) questions.arguments_correct = clone(ARGUMENT_QUESTION);
      const state = toPlain(stateJson) as DecisionState;
      STATES.set(state, stateJson);
      const answer = await this.ask(state, questions);
      functionProbability = answer.probabilities.function_correct ?? null;
      if (!argumentsSkipped) argumentsProbability = answer.probabilities.arguments_correct ?? null;
      resolvedModel = answer.model;
      usage = answer.usage;
      apiCalls = answer.apiCalls;
      waited = answer.retryWaitMs;
    }

    // V(p) is true iff p is not null and p >= threshold; S must pass too.
    const functionPass = local.functionPass && functionProbability !== null && functionProbability >= this.functionThreshold;
    const argumentsPass = argumentsSkipped ? null
      : local.argumentsPass && argumentsProbability !== null && argumentsProbability >= this.argumentsThreshold;
    const checks = {
      function_selection: { pass: functionPass, probability_yes: functionProbability, threshold: this.functionThreshold,
        schema_pass: local.functionPass },
      arguments: { skipped: argumentsSkipped, pass: argumentsPass, probability_yes: argumentsProbability,
        threshold: this.argumentsThreshold, schema_pass: local.argumentsPass },
    };
    const failed = Object.entries(checks)
      .filter(([, check]) => !(check as { skipped?: boolean }).skipped && !check.pass).map(([name]) => name);
    const feedback: FeedbackRecord = {
      source: (this.constructor as typeof DecisionModel).feedbackSource,
      checks,
      failed_checks: failed,
      schema_errors: local.errors,
      instruction: REPAIR_INSTRUCTION,
    };
    return {
      functionPass,
      argumentsPass,
      argumentsSkipped,
      functionProbability,
      argumentsProbability,
      accepted: functionPass && (argumentsSkipped || argumentsPass === true),
      feedback,
      schemaErrors: local.errors,
      usage,
      model: resolvedModel,
      requestedModel: this.model,
      latencyMs: Math.max(0, now() - started - waited),
      retryWaitMs: waited,
      apiCalls,
    };
  }
}

export interface JevOptions extends Omit<DecisionModelOptions, "model"> {
  /** Defaults to `TYPESAFE_MODEL`, then `jev-1.13.0` (the paper's). */
  model?: string;
  /** Defaults to `TYPESAFE_BASE_URL`, then `https://api.typesafe.ai`. */
  baseUrl?: string;
}

/**
 * TypeSafe's hosted decision model. Key: `apiKey` or `TYPESAFE_API_KEY`.
 *
 * The default model is `jev-1.13.0`, the one used in the paper (`TYPESAFE_MODEL` or `model`
 * changes it), and the default endpoint is `https://api.typesafe.ai` (`TYPESAFE_BASE_URL` or
 * `baseUrl`).
 */
export class Jev extends DecisionModel {
  static override displayName = "Jev";
  static override feedbackSource = "jev_noul_and_deterministic_schema_validation";
  static readonly DEFAULT_MODEL = "jev-1.13.0";
  static readonly DEFAULT_BASE_URL = "https://api.typesafe.ai";

  constructor(apiKey?: string | null, options: JevOptions = {}) {
    const key = apiKey || env("TYPESAFE_API_KEY")?.trim() || undefined;
    if (!key) throw new Error("Jev needs an API key: pass apiKey or set TYPESAFE_API_KEY.");
    const { baseUrl, model, ...rest } = options;
    super(baseUrl || env("TYPESAFE_BASE_URL") || Jev.DEFAULT_BASE_URL, {
      ...rest, apiKey: key, model: model || env("TYPESAFE_MODEL") || Jev.DEFAULT_MODEL,
    });
  }
}

export interface PPLXDeciderOptions extends Omit<DecisionModelOptions, "model"> {
  /** First read `GET /provenance` and stop unless the server reports the pinned checkpoint. */
  checkProvenance?: boolean;
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

/** What `GET /provenance` must report. */
export interface Provenance {
  model_id: string;
  revision: string;
  native_attention: string;
  max_input_tokens: number;
  temperature: number;
}

/**
 * Perplexity Decider v1.1 27B, an open-weight decision model, served on your own machine.
 *
 * `baseUrl` is the origin of a server running the checkpoint (`perplexity-ai/pplx-decider-v1.1-27b`
 * at revision `14c9d901...`) with its own native inference code, which serves `/v1/systemone`. A
 * generic causal chat server is not this model. Plain HTTP is accepted only for `localhost`; use an
 * SSH tunnel or HTTPS otherwise.
 *
 * With `checkProvenance: true` the client first reads `GET /provenance` and stops unless the server
 * reports the pinned checkpoint, revision, attention mode and calibration temperature. The paper's
 * runs used that check through a small deployment wrapper; the checkpoint's own server does not
 * provide the route, so it is off by default.
 */
export class PPLXDecider extends DecisionModel {
  static override displayName = "PPLX Decider";
  static override feedbackSource = "perplexity_decider_noul_and_deterministic_schema_validation";
  static readonly MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b";
  static readonly MODEL_REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce";
  static readonly CALIBRATION_TEMPERATURE = 1.0087417621345625;
  static readonly MAX_INPUT_TOKENS = 8192;
  /** The checkpoint's server reports this inherited name and accepts `jev-latest` as the request model. */
  static readonly SERVER_MODEL = "autojev-qwen3.8-27b";
  static readonly WIRE_MODEL = "jev-latest";

  readonly checkProvenance: boolean;
  #provenance: Provenance | null = null;

  constructor(baseUrl: string, options: PPLXDeciderOptions = {}) {
    const parts = urlsplit(typeof baseUrl === "string" ? baseUrl : "");
    if (parts.path !== "" && parts.path !== "/") throw new Error("PPLXDecider baseUrl must be an origin without a path.");
    if (parts.scheme === "http" && !(parts.hostname !== null && LOCAL_HOSTS.has(parts.hostname))) {
      throw new Error("Use HTTPS, or an SSH tunnel to localhost, for a remote PPLX Decider.");
    }
    const { checkProvenance = false, ...rest } = options;
    if (typeof checkProvenance !== "boolean") throw new Error("checkProvenance must be true or false.");
    super(baseUrl, {
      ...rest, timeoutMs: rest.timeoutMs ?? 180_000, maxRetries: rest.maxRetries ?? 0, model: PPLXDecider.WIRE_MODEL,
    });
    this.checkProvenance = checkProvenance;
  }

  /** The server's reported checkpoint, verified against the pinned one. */
  async provenance(): Promise<Provenance> {
    let response: HttpResponse;
    try {
      response = await this.request("/provenance", "GET");
    } catch {
      throw new DecisionModelError("PPLX Decider provenance could not be reached.", { code: "connection_failed" });
    }
    if (response.status !== 200) {
      throw new DecisionModelError("PPLX Decider provenance is unavailable.",
        { code: "provenance_unavailable", statusCode: response.status });
    }
    let body: Json;
    try {
      body = parse(response.text);
    } catch {
      throw new DecisionModelError("PPLX Decider provenance was not valid JSON.", { code: "provenance_invalid" });
    }
    if (!(body instanceof Map)) {
      throw new DecisionModelError("PPLX Decider provenance must be an object.", { code: "provenance_invalid" });
    }
    const expected: [string, Json][] = [
      ["model_id", PPLXDecider.MODEL_ID], ["revision", PPLXDecider.MODEL_REVISION],
      ["native_attention", "noncausal_full_attention"], ["max_input_tokens", BigInt(PPLXDecider.MAX_INPUT_TOKENS)],
    ];
    const temperature = body.get("temperature");
    const value = typeof temperature === "bigint" ? Number(temperature) : temperature instanceof Float ? temperature.value
      : undefined;
    const mismatch = expected.some(([key, wanted]) => {
      const got = body.get(key);
      if (typeof wanted === "bigint") {
        const number = typeof got === "bigint" ? Number(got) : got instanceof Float ? got.value : undefined;
        return number !== Number(wanted);
      }
      return got !== wanted;
    });
    if (mismatch || value === undefined || !(Math.abs(value - PPLXDecider.CALIBRATION_TEMPERATURE) <= 1e-12)) {
      throw new DecisionModelError("The server does not run the pinned PPLX Decider checkpoint.", { code: "provenance_mismatch" });
    }
    return {
      model_id: PPLXDecider.MODEL_ID, revision: PPLXDecider.MODEL_REVISION, native_attention: "noncausal_full_attention",
      max_input_tokens: PPLXDecider.MAX_INPUT_TOKENS, temperature: value,
    };
  }

  override async ask(state: DecisionState, questions: Questions): Promise<Answer> {
    if (this.checkProvenance && this.#provenance === null) this.#provenance = await this.provenance();
    return super.ask(state, questions);
  }

  protected override errorFor(response: HttpResponse, attempts: number): DecisionModelError {
    const error = super.errorFor(response, attempts);
    if (response.status === 422) {
      try {
        const body = parse(response.text);
        const detail = body instanceof Map && body.has("detail") ? body.get("detail")! : "";
        if (pythonStr(detail).includes("8192-token limit")) error.code = "context_limit_exceeded";
      } catch {
        // not JSON: keep the generic HTTP error
      }
    }
    return error;
  }

  protected override checked(body: JsonObject): JsonObject {
    const model = body.get("model");
    if (model !== PPLXDecider.MODEL_ID && model !== PPLXDecider.SERVER_MODEL) {
      throw new DecisionModelError("PPLX Decider response has an unexpected model ID.", { code: "model_mismatch" });
    }
    if (body.has("revision") && body.get("revision") !== PPLXDecider.MODEL_REVISION) {
      throw new DecisionModelError("PPLX Decider response has an unexpected model revision.", { code: "model_mismatch" });
    }
    // Missing usage is unknown, never zero.
    const result: JsonObject = new Map(body);
    result.set("model", PPLXDecider.MODEL_ID);
    if (!body.has("usage")) result.set("usage", new Map<string, Json>([["input_tokens", null], ["output_tokens", null]]));
    return result;
  }

  override toString(): string {
    return `PPLXDecider(model=${JSON.stringify(PPLXDecider.MODEL_ID)}, revision=${JSON.stringify(PPLXDecider.MODEL_REVISION.slice(0, 8))}, `
      + `baseUrl=${JSON.stringify(this.baseUrl)}, functionThreshold=${this.functionThreshold}, `
      + `argumentsThreshold=${this.argumentsThreshold})`;
  }
}
