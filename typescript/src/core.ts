/**
 * DM-Call: propose a function call, verify it with a decision model, repair it with cumulative feedback.
 *
 * At step t the generator proposes a candidate c_t (one call, or no call). The candidate is accepted
 * when it passes the schema check S and every decision-model question that applies to it (Eq. 2 of
 * the paper). A rejected candidate and its feedback d_t join the history H, which the generator
 * reads, oldest first and marked as not executed, before the unchanged request. The loop returns
 * the first accepted candidate; if none of c_0..c_r is accepted it returns c_0. Nothing is executed
 * while the loop runs.
 */

import { type Json, type JsonObject, clone, fromPlain, pythonText, stringRepr, toPlain } from "./json.js";
import { DecisionModel, type FeedbackRecord, type Judge, type Judgment, Jev } from "./decision.js";
import { type FeedbackRepresentation, FEEDBACK, representHistory, singleCheck } from "./feedback.js";
import type { FetchLike, Sleep } from "./http.js";
import { now } from "./http.js";
import {
  type AnyProviderOptions, type Generator, type HistoryRecord, type JsonProposal, type Proposal, type Usage,
  GENERATE_JSON, Provider, provider as makeProvider,
} from "./providers.js";
import { type SchemaCheck, type SchemaIssue, checkSchemaJson } from "./schema.js";
import { type Call, type ToolHandler, type ToolInput, normalizeCallsJson, toolset } from "./tools.js";
import type { TextMessage } from "./prompts.js";

/** Pass mark of both questions in the paper. */
export const DEFAULT_THRESHOLD = 0.5;
/** r* in the paper. */
export const DEFAULT_MAX_REPAIRS = 3;

export type Checks = "both" | "function" | "arguments";
export type History = "cumulative" | "latest";
export type OnExhaustion = "first" | "last";

/** DM-Call, DM-Call-Function, DM-Call-Argument. */
export const CHECKS: readonly Checks[] = Object.freeze(["both", "function", "arguments"]);
/** DM-Call, DM-Call-Latest. */
export const HISTORIES: readonly History[] = Object.freeze(["cumulative", "latest"]);
export const ON_EXHAUSTION: readonly OnExhaustion[] = Object.freeze(["first", "last"]);

const ROLES = new Set(["system", "developer", "user", "assistant"]);
const HISTORY_ROLES = new Set([...ROLES, "tool", "function"]);
const HISTORY_KEYS = ["role", "content", "name", "tool_calls", "tool_call_id"];
const TEXT_PARTS = new Set(["text", "input_text", "output_text"]);

/** A conversation: a prompt string, one message, or a list of messages. */
export type Messages = string | Record<string, unknown> | readonly Record<string, unknown>[];

function choice<T extends string>(name: string, value: unknown, choices: readonly T[]): T {
  if (!choices.includes(value as T)) throw new Error(`${name} must be one of ${choices.map((c) => `'${c}'`).join(", ")}.`);
  return value as T;
}

function repairLimit(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new Error("maxRepairs must be a nonnegative integer.");
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** A prompt string or chat messages as `[{role, content}]` with text content. */
function messagesOf(messages: unknown): TextMessage[] {
  let list: unknown = messages;
  if (typeof list === "string") list = [{ role: "user", content: list }];
  else if (isRecord(list)) list = [list];
  if (!Array.isArray(list) || list.length === 0) {
    throw new Error("messages must be a string or a nonempty list of {role, content} messages.");
  }
  return list.map((message) => {
    if (!isRecord(message) || typeof message.role !== "string" || !ROLES.has(message.role)) {
      throw new Error("Each message needs a role (system, developer, user or assistant) and text content; "
        + "give earlier tool results to the model as text.");
    }
    let content = message.content;
    if (Array.isArray(content)) {
      if (!content.every((part) => isRecord(part) && typeof part.type === "string" && TEXT_PARTS.has(part.type)
        && typeof part.text === "string")) {
        throw new Error("Only text message content is supported.");
      }
      content = content.map((part) => (part as { text: string }).text).join("\n");
    }
    if (typeof content !== "string") throw new Error("Message content must be text.");
    return { role: message.role, content };
  });
}

/**
 * A conversation as the decision model sees it: text turns, earlier tool calls and tool results.
 * Messages may be plain objects or SDK message objects; only role, content, name, tool_calls and
 * tool_call_id are kept.
 */
function conversationOf(messages: unknown): JsonObject[] {
  if (typeof messages === "string") return [new Map<string, Json>([["role", "user"], ["content", messages]])];
  const list = Array.isArray(messages) ? messages : [messages];
  const result = list.map((message) => {
    const value = fromPlain(message);
    const role = value instanceof Map ? value.get("role") : undefined;
    if (!(value instanceof Map) || typeof role !== "string" || !HISTORY_ROLES.has(role)) {
      throw new Error("Each message needs a role: system, developer, user, assistant or tool.");
    }
    const kept: JsonObject = new Map();
    for (const key of HISTORY_KEYS) {
      const item = value.get(key);
      if (item !== undefined && item !== null) kept.set(key, clone(item));
    }
    return kept;
  });
  if (!result.length) throw new Error("messages must not be empty.");
  return result;
}

function formatCall(call: Json): string {
  if (!(call instanceof Map)) return "<malformed call>";
  const name = call.get("name");
  const label = typeof name === "string" ? name : name === undefined || name === null ? "None" : pythonText(name);
  const args = call.get("arguments");
  if (!(args instanceof Map)) return `${label}(<malformed arguments>)`;
  try {
    return `${label}(${[...args].map(([key, value]) => `${key}=${pythonText(value)}`).join(", ")})`;
  } catch {
    return `${label}(<malformed arguments>)`;
  }
}

function addCounts(total: Record<string, number>, counts: object | undefined | null): void {
  for (const [key, value] of Object.entries(counts ?? {})) {
    if (typeof value === "number" && Number.isInteger(value)) total[key] = (total[key] ?? 0) + value;
  }
}

/** A completely parsed decision to make no call. Unreadable output (`calls: null`) is not one. */
function readableNoCall(proposal: JsonProposal): boolean {
  return Array.isArray(proposal.calls) && proposal.calls.length === 0 && !proposal.incomplete
    && !(proposal.parseErrors && proposal.parseErrors.length);
}

/** The verification record of an initial no call returned without asking the decision model. */
export interface PreservedNoCall {
  kind: "initial_no_call_preserved";
  accepted: null;
  functionProbability: null;
  argumentsProbability: null;
  functionPass: null;
  argumentsPass: null;
  argumentsSkipped: true;
  apiCalls: 0;
  usage: Record<string, never>;
  latencyMs: 0;
  feedback: Record<string, never>;
}

/** One candidate in {@link Result.trace}: the proposal, its judgment and the feedback the generator was given. */
export interface TraceStep {
  attempt: number;
  proposal: Proposal;
  verification: Judgment | PreservedNoCall;
  schema: SchemaCheck;
  /** The feedback record as judged (numeric); the generator's copy may use another representation. */
  feedback: FeedbackRecord | Record<string, never>;
  /** Whether the candidate was accepted (`null` for a preserved initial no call). */
  gatePass: boolean | null;
}

/** Each candidate with its verdict, scores and schema errors. */
export interface Attempt {
  calls: Call[] | null;
  accepted: boolean | null;
  functionProbability: number | null;
  argumentsProbability: number | null;
  schemaErrors: SchemaIssue[];
}

interface InternalStep {
  attempt: number;
  proposal: JsonProposal;
  verification: Judgment | PreservedNoCall;
  schema: SchemaCheck;
  feedback: FeedbackRecord | Record<string, never>;
  gatePass: boolean | null;
}

interface ResultInit {
  calls: Json | null;
  accepted: boolean | null;
  exhausted: boolean;
  repairs: number;
  text: string | null;
  executed?: boolean;
  outputs?: unknown[];
  latencyMs: number;
  trace: InternalStep[];
  finalAttempt: number | null;
  initialNoCallPreserved?: boolean;
}

/**
 * The outcome of one DM-Call request.
 *
 * `calls` is the final candidate as `[{name, arguments}]`: `[]` is a decision to make no call and
 * `null` an unreadable candidate (cut off, filtered or refused). `accepted` says whether it passed
 * every check; it is `null` for an initial no call returned without verification
 * (`initialNoCallPreserved`). When no candidate passed within the repair limit (`exhausted`), the
 * final candidate is the first one (`onExhaustion: "first"`, the paper's rule) or the last one;
 * `finalAttempt` is its index in `trace`, which keeps every candidate, judgment and feedback record
 * in order.
 */
export class Result {
  readonly calls: Call[] | null;
  readonly accepted: boolean | null;
  readonly exhausted: boolean;
  readonly repairs: number;
  readonly text: string | null;
  readonly executed: boolean;
  readonly outputs: unknown[];
  readonly latencyMs: number;
  readonly trace: TraceStep[];
  readonly finalAttempt: number | null;
  readonly initialNoCallPreserved: boolean;
  /** The first call of the final candidate, or `null` for no call. */
  readonly call: Call | null;
  /** The final candidate is a decision to make no call. */
  readonly noCall: boolean;
  /** What the executed function returned (an array if several calls ran). */
  readonly output: unknown;
  /** Each candidate with its verdict, scores and schema errors, oldest first. */
  readonly attempts: Attempt[];
  /** The decision model's probabilities of yes for the final candidate (`null` if not asked). */
  readonly probabilities: { function: number | null; arguments: number | null };
  /** Token counts summed over every generator request and every decision-model request. */
  readonly usage: { generator: Usage; decisionModel: Record<string, number> };

  constructor(init: ResultInit) {
    this.calls = init.calls === null ? null : (toPlain(init.calls) as Call[]);
    this.accepted = init.accepted;
    this.exhausted = init.exhausted;
    this.repairs = init.repairs;
    this.text = init.text;
    this.executed = init.executed ?? false;
    this.outputs = init.outputs ?? [];
    this.latencyMs = init.latencyMs;
    this.finalAttempt = init.finalAttempt;
    this.initialNoCallPreserved = init.initialNoCallPreserved ?? false;
    this.trace = init.trace.map((step) => ({
      attempt: step.attempt,
      proposal: { ...step.proposal, calls: step.proposal.calls === null ? null : (toPlain(step.proposal.calls) as Call[]) },
      verification: step.verification,
      schema: step.schema,
      feedback: step.feedback,
      gatePass: step.gatePass,
    }));
    this.call = Array.isArray(this.calls) && this.calls.length ? this.calls[0]! : null;
    this.noCall = Array.isArray(this.calls) && this.calls.length === 0;
    this.output = this.outputs.length === 0 ? null : this.outputs.length === 1 ? this.outputs[0] : [...this.outputs];
    this.attempts = init.trace.map((step) => ({
      calls: step.proposal.calls === null ? null : (toPlain(clone(step.proposal.calls)) as Call[]),
      accepted: step.gatePass,
      functionProbability: step.verification.functionProbability ?? null,
      argumentsProbability: step.verification.argumentsProbability ?? null,
      schemaErrors: clone(step.schema.errors ?? []),
    }));
    const final = this.finalAttempt === null ? init.trace.length - 1 : this.finalAttempt;
    const check = init.trace[final]?.verification;
    this.probabilities = { function: check?.functionProbability ?? null, arguments: check?.argumentsProbability ?? null };
    const generator: Record<string, number> = {};
    const decisionModel: Record<string, number> = {};
    for (const step of init.trace) {
      addCounts(generator, step.proposal.usage);
      addCounts(decisionModel, step.verification.usage);
    }
    this.usage = { generator, decisionModel };
    this.#display = init.calls === null ? "unreadable candidate"
      : Array.isArray(init.calls) && init.calls.length === 0 ? "no call"
        : `call=${(Array.isArray(init.calls) ? init.calls : [init.calls]).map(formatCall).join("; ")}`;
  }

  readonly #display: string;

  toString(): string {
    const parts = [this.#display, `accepted=${this.accepted}`, `repairs=${this.repairs}`];
    if (this.executed) {
      let shown: string;
      try {
        shown = JSON.stringify(this.output) ?? String(this.output);
      } catch {
        shown = String(this.output);
      }
      parts.push(`output=${shown}`);
    }
    return `Result(${parts.join(", ")})`;
  }
}

/** The schema check's and the decision model's judgment of calls from any source. */
export class Verdict {
  constructor(
    /** Passed the schema check and every question that applies. */
    readonly accepted: boolean,
    readonly calls: Call[],
    readonly functionProbability: number | null,
    readonly argumentsProbability: number | null,
    readonly schemaErrors: SchemaIssue[],
    /** The record DM-Call would send back to the model for a repair. */
    readonly feedback: FeedbackRecord,
    /** The full judgment. */
    readonly raw: Judgment,
  ) {}

  toString(): string {
    return `Verdict(accepted=${this.accepted}, functionProbability=${this.functionProbability}, `
      + `argumentsProbability=${this.argumentsProbability})`;
  }
}

function hasJudge(value: unknown): value is Judge {
  return value !== null && typeof value === "object" && typeof (value as Judge).judge === "function";
}

/** Ask a decision model; a custom one (anything with `judge`) receives plain objects. */
async function judgeWith(decisionModel: Judge, messages: unknown, tools: Json[], calls: Json | null): Promise<Judgment> {
  if (decisionModel instanceof DecisionModel) return decisionModel.judge(messages, tools, calls);
  const plainMessages = Array.isArray(messages) ? messages.map((message) => toPlain(message as Json)) : messages;
  return decisionModel.judge(plainMessages, toPlain(tools), calls === null ? null : toPlain(calls));
}

async function verdictOf(decisionModel: Judge, messages: unknown, calls: unknown, tools: unknown): Promise<Verdict> {
  const { schemas } = toolset(tools);
  const conversation = conversationOf(messages);
  const proposed = normalizeCallsJson(calls);
  const local = checkSchemaJson(schemas, proposed);
  const check = await judgeWith(decisionModel, conversation, schemas, proposed);
  const accepted = local.accepted && check.functionPass && (check.argumentsSkipped || check.argumentsPass === true);
  return new Verdict(accepted, toPlain(proposed) as Call[], check.functionProbability, check.argumentsProbability,
    local.errors, check.feedback, check);
}

export interface VerifyOptions {
  /** Defaults to {@link Jev} with `threshold`. */
  decisionModel?: Judge;
  /** Pass mark of both questions for the default Jev (0.5). */
  threshold?: number;
  /** Key for the default Jev; defaults to `TYPESAFE_API_KEY`. */
  jevApiKey?: string;
  fetch?: FetchLike;
  sleep?: Sleep;
}

/**
 * Check calls from any source before running them; nothing is generated or executed.
 *
 * `messages` is the conversation so far, including earlier tool calls and tool results. `calls` may
 * be plain objects or the tool-call objects of the OpenAI, Anthropic or Google SDKs. The calls pass
 * if they pass the schema check and the decision model's function question and, unless there is no
 * call, its argument question (0.5 by default). `decisionModel` defaults to {@link Jev}.
 */
export async function verify(messages: unknown, calls: unknown, tools: unknown, options: VerifyOptions = {}): Promise<Verdict> {
  const { decisionModel, threshold, jevApiKey, fetch, sleep } = options;
  if (decisionModel !== undefined) {
    if (jevApiKey !== undefined || threshold !== undefined) {
      throw new Error("Configure the decision model itself (apiKey, threshold).");
    }
    return verdictOf(decisionModel, messages, calls, tools);
  }
  const jev = new Jev(jevApiKey, { threshold: threshold ?? DEFAULT_THRESHOLD, fetch, sleep });
  return verdictOf(jev, messages, calls, tools);
}

/** Options of {@link DMCall}. Options not listed here (`apiKey`, `baseUrl`, `reasoningEffort`, ...) go to the provider. */
export interface DMCallOptions extends AnyProviderOptions {
  /** Tool definitions, optionally made with `tool({ ..., execute })`. */
  tools?: ToolInput | readonly ToolInput[] | null;
  /** Handlers by tool name, for definitions without `execute`. */
  functions?: Record<string, ToolHandler>;
  /** Defaults to {@link Jev} with `threshold`. */
  decisionModel?: Judge;
  /** Pass mark of both questions for the default Jev (0.5). */
  threshold?: number;
  /** r: repairs after the first candidate (3). */
  maxRepairs?: number;
  /** `"function"` or `"arguments"`: use only that question for acceptance and repair feedback; both are sent. */
  checks?: Checks;
  /** `"latest"` (DM-Call-Latest): show only the latest rejected candidate. */
  history?: History;
  /** How scores reach the generator: `"hybrid"`, `"numeric"` or `"text"`. */
  feedback?: FeedbackRepresentation;
  /** If nothing is accepted: return c_0 (`"first"`) or the last repair (`"last"`). */
  onExhaustion?: OnExhaustion;
  /** Return a readable first no call without verification (true). */
  preserveInitialNoCall?: boolean;
  /** `run()` also executes the final candidate when none was accepted (false). */
  executeUnverified?: boolean;
  /** Key for the default Jev; defaults to `TYPESAFE_API_KEY`. */
  jevApiKey?: string;
}

/** Per-request options of {@link DMCall.call}. */
export interface CallOptions {
  tools?: ToolInput | readonly ToolInput[] | null;
  functions?: Record<string, ToolHandler>;
  maxRepairs?: number;
  onExhaustion?: OnExhaustion;
}

/** Per-request options of {@link DMCall.run}. */
export interface RunOptions extends CallOptions {
  executeUnverified?: boolean;
}

/**
 * Decision-model-guided function calling.
 *
 * ```ts
 * const agent = new DMCall("openai:gpt-5-nano", { tools: [getWeather] }); // OPENAI_API_KEY, TYPESAFE_API_KEY
 * const result = await agent.run("Is it warm in Seoul right now?");
 * ```
 *
 * `model` is `"provider:model"` (openai, anthropic, gemini, xai, azure, openrouter) or a provider
 * object; options DM-Call does not use (`apiKey`, `baseUrl`, `reasoningEffort`, ...) go to the
 * provider. `decisionModel` defaults to {@link Jev} with `threshold` (0.5) for both questions; pass a
 * configured {@link Jev} or {@link PPLXDecider} instead to change it.
 *
 * The defaults are the paper's settings: at most `maxRepairs: 3` repairs, both questions
 * (`checks: "both"`), the whole history (`history: "cumulative"`), Hybrid feedback
 * (`feedback: "hybrid"`), the first candidate when none passes (`onExhaustion: "first"`) and a
 * readable initial no call returned without verification (`preserveInitialNoCall: true`).
 */
export class DMCall {
  readonly model: Generator;
  readonly decisionModel: Judge;
  readonly maxRepairs: number;
  readonly checks: Checks;
  readonly history: History;
  readonly feedback: FeedbackRepresentation;
  readonly onExhaustion: OnExhaustion;
  readonly preserveInitialNoCall: boolean;
  readonly executeUnverified: boolean;
  readonly tools: ToolInput[];
  readonly functions: Record<string, ToolHandler>;

  constructor(model: string | Generator, options: DMCallOptions = {}) {
    const {
      tools, functions, decisionModel, threshold, maxRepairs = DEFAULT_MAX_REPAIRS, checks = "both", history = "cumulative",
      feedback = "hybrid", onExhaustion = "first", preserveInitialNoCall = true, executeUnverified = false, jevApiKey,
      fetch, sleep, ...providerOptions
    } = options;
    this.maxRepairs = repairLimit(maxRepairs);
    this.checks = choice("checks", checks, CHECKS);
    this.history = choice("history", history, HISTORIES);
    this.feedback = choice("feedback", feedback, FEEDBACK);
    this.onExhaustion = choice("onExhaustion", onExhaustion, ON_EXHAUSTION);
    if (typeof preserveInitialNoCall !== "boolean") throw new Error("preserveInitialNoCall must be true or false.");
    this.preserveInitialNoCall = preserveInitialNoCall;
    this.executeUnverified = Boolean(executeUnverified);
    this.tools = Array.isArray(tools) ? [...tools] : tools === undefined || tools === null ? [] : [tools as ToolInput];
    this.functions = { ...(functions ?? {}) };
    toolset(this.tools, this.functions); // fail fast on a bad tool
    if (decisionModel !== undefined) {
      if (jevApiKey !== undefined || threshold !== undefined) {
        throw new Error("Configure the decision model itself (apiKey, threshold), not DMCall.");
      }
      if (!hasJudge(decisionModel)) {
        throw new TypeError("decisionModel must have judge(messages, tools, calls), like dmcall.Jev.");
      }
    }
    this.model = typeof model === "string" ? makeProvider(model, { ...providerOptions, fetch, sleep })
      : makeProvider(model, providerOptions);
    this.decisionModel = decisionModel ?? new Jev(jevApiKey, { threshold: threshold ?? DEFAULT_THRESHOLD, fetch, sleep });
  }

  /** Propose, verify and repair a call. Nothing is executed. */
  call(messages: Messages, options: CallOptions = {}): Promise<Result> {
    return this.solve(messages, options, false, false);
  }

  /**
   * {@link call}, then execute the final call with its handler if it was accepted.
   *
   * With `executeUnverified: true` the final candidate also runs when none was accepted: the first
   * one, or the last with `onExhaustion: "last"`. Every tool must have a handler before any request is sent.
   * Replacement definitions keep the client's handler of the same name unless another is supplied.
   */
  run(messages: Messages, options: RunOptions = {}): Promise<Result> {
    const unverified = options.executeUnverified === undefined ? this.executeUnverified : Boolean(options.executeUnverified);
    return this.solve(messages, options, true, unverified);
  }

  /** Check a proposed call (plain objects or SDK objects) with the schema check and both questions. */
  verify(messages: unknown, calls: unknown, tools?: unknown): Promise<Verdict> {
    return verdictOf(this.decisionModel, messages, calls, tools === undefined || tools === null ? this.tools : tools);
  }

  toString(): string {
    return `DMCall(model=${String(this.model)}, decisionModel=${String(this.decisionModel)}, checks='${this.checks}', `
      + `history='${this.history}', feedback='${this.feedback}', maxRepairs=${this.maxRepairs}, `
      + `onExhaustion='${this.onExhaustion}', preserveInitialNoCall=${this.preserveInitialNoCall})`;
  }

  // The loop ----------------------------------------------------------------------------------------
  /** A_t (Eq. 2) for the configured questions, and the feedback record the generator gets. */
  private accept(local: SchemaCheck, check: Judgment): [boolean, FeedbackRecord] {
    if (this.checks === "function") return [local.accepted && check.functionPass, singleCheck(check.feedback, "function")];
    const args = check.argumentsSkipped || check.argumentsPass === true;
    if (this.checks === "arguments") return [local.accepted && args, singleCheck(check.feedback, "arguments")];
    return [local.accepted && check.functionPass && args, check.feedback];
  }

  private async propose(messages: TextMessage[], schemas: JsonObject[], history: HistoryRecord[]): Promise<JsonProposal> {
    if (this.model instanceof Provider && this.model.generate === Provider.prototype.generate) {
      return this.model[GENERATE_JSON]({ messages, tools: schemas }, history);
    }
    const plainHistory = history.map((record) => ({ ...record, calls: toPlain(record.calls as Json) }));
    const proposal = await this.model.generate({ messages, tools: toPlain(schemas) as unknown[] }, plainHistory);
    if (!isRecord(proposal)) throw new TypeError("generate() must return a proposal object with calls.");
    return {
      ...proposal,
      calls: proposal.calls === null || proposal.calls === undefined ? null : fromPlain(proposal.calls),
      content: proposal.content ?? null,
      parseErrors: proposal.parseErrors ?? [],
      incomplete: proposal.incomplete ?? null,
    };
  }

  private async solve(messages: Messages, options: RunOptions, execute: boolean, unverified: boolean): Promise<Result> {
    const limit = options.maxRepairs === undefined ? this.maxRepairs : repairLimit(options.maxRepairs);
    const rule = options.onExhaustion === undefined ? this.onExhaustion
      : choice("onExhaustion", options.onExhaustion, ON_EXHAUSTION);
    const { schemas, registry } = options.tools === undefined || options.tools === null
      ? toolset(this.tools, { ...this.functions, ...(options.functions ?? {}) })
      : toolset(options.tools, options.functions);
    if (options.tools !== undefined && options.tools !== null) {
      const known = toolset(this.tools, this.functions).registry;
      for (const schema of schemas) {
        const name = (schema.get("function") as JsonObject).get("name") as string;
        if (!registry.has(name) && known.has(name)) registry.set(name, known.get(name)!);
      }
    }
    if (execute) {
      for (const schema of schemas) {
        const name = (schema.get("function") as JsonObject).get("name") as string;
        if (!registry.has(name)) {
          throw new Error(`No function for tool ${stringRepr(name)}; pass it with tool({ execute }) or in functions.`);
        }
      }
    }
    const conversation = messagesOf(messages);
    const started = now();
    const history: HistoryRecord[] = [];
    const trace: InternalStep[] = [];
    let accepted = false;
    for (let attempt = 0; attempt <= limit; attempt++) {
      let shown = history;
      if (this.history === "latest" && history.length) {
        // Only the latest rejection, with its real round number.
        shown = [{ ...clone(history[history.length - 1]!), round: attempt }];
      }
      const proposal = await this.propose(conversation, schemas, representHistory(shown, this.feedback));
      const calls = proposal.calls;
      const local = checkSchemaJson(schemas, calls);
      if (attempt === 0 && this.preserveInitialNoCall && readableNoCall(proposal) && local.accepted) {
        // A readable initial no call is returned without verification or repair.
        const check: PreservedNoCall = {
          kind: "initial_no_call_preserved", accepted: null, functionProbability: null, argumentsProbability: null,
          functionPass: null, argumentsPass: null, argumentsSkipped: true, apiCalls: 0, usage: {}, latencyMs: 0,
          feedback: {},
        };
        trace.push({ attempt: 0, proposal, verification: check, schema: local, feedback: {}, gatePass: null });
        return new Result({
          calls: [], accepted: null, exhausted: false, repairs: 0, text: proposal.content ?? null,
          latencyMs: now() - started, trace, finalAttempt: 0, initialNoCallPreserved: true,
        });
      }
      const check = await judgeWith(this.decisionModel, conversation, schemas, calls);
      const [passed, feedback] = this.accept(local, check);
      accepted = passed;
      trace.push({ attempt, proposal, verification: check, schema: local, feedback: clone(feedback), gatePass: passed });
      if (passed) break;
      history.push({ calls: clone(calls), feedback: clone(feedback) as unknown as Record<string, unknown> });
    }

    const exhausted = !accepted;
    const final = exhausted && rule === "first" ? 0 : trace.length - 1;
    const calls = clone(trace[final]!.proposal.calls);
    const outputs: unknown[] = [];
    let executed = false;
    if (execute && (accepted || unverified)) executed = await DMCall.execute(calls, registry, outputs);
    return new Result({
      calls, accepted, exhausted, repairs: trace.length - 1, text: trace[final]!.proposal.content ?? null, executed,
      outputs, latencyMs: now() - started, trace, finalAttempt: final,
    });
  }

  /** Run the final calls with their handlers. An unreadable candidate never runs. */
  private static async execute(calls: Json | null, registry: Map<string, ToolHandler>, outputs: unknown[]): Promise<boolean> {
    if (!Array.isArray(calls) || !calls.every((call) => call instanceof Map && typeof call.get("name") === "string"
      && call.get("arguments") instanceof Map)) return false;
    const entries = calls as JsonObject[];
    const missing = entries.map((call) => call.get("name") as string).filter((name) => !registry.has(name));
    if (missing.length) {
      throw new Error(`No function for tool ${stringRepr(missing[0]!)}; pass it with tool({ execute }) or in functions.`);
    }
    for (const call of entries) {
      const handler = registry.get(call.get("name") as string)!;
      outputs.push(await handler(toPlain(clone(call.get("arguments")!)) as Record<string, unknown>));
    }
    return entries.length > 0;
  }
}
