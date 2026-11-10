/**
 * Function-calling models over plain HTTP (`fetch` only, no vendor SDKs).
 *
 * Each provider turns (messages, tools, repair history) into one request and the response into a
 * proposal: a list of calls `[{name, arguments}]` (`[]` for no call), or `null` when the output holds
 * no complete proposal (cut off at the token limit, filtered or refused). Keys come from options or
 * environment variables and never appear in errors, string forms or results.
 */

import { type Json, type JsonObject, compactText, fromPlain, isFiniteJson, parse, toPlain, truthy } from "./json.js";
import {
  type FetchLike, type HttpResponse, type Sleep, checkBaseUrl, defaultSleep, errorName, firstEnv, now, pythonFloat,
  quote, requestId, resolveFetch, send, wait,
} from "./http.js";
import { SYSTEM_PROMPT, type TextMessage, functionsOf, withHistory } from "./prompts.js";
import type { Call } from "./tools.js";

/** Why an output holds no complete proposal. Such a proposal is rejected and repaired. */
export const INCOMPLETE: Readonly<Record<string, string>> = Object.freeze({
  length: "The response reached the output token limit before a complete proposal.",
  content_filter: "The provider's content filter stopped the response before a complete proposal.",
  refusal: "The response was a refusal and contains no proposal.",
  prompt_filter: "The provider's content filter rejected the request; no response was generated.",
  malformed_call: "The model produced a malformed function call.",
  abnormal_stop: "The response stopped abnormally before a complete proposal.",
});

/** A sanitized provider failure. It never contains keys or response bodies. */
export class ProviderError extends Error {
  statusCode: number | null;
  requestId: string | null;
  attempts: number | null;
  providerCode: string | null;

  constructor(message: string, details: { statusCode?: number | null; requestId?: string | null; attempts?: number | null;
    providerCode?: string | null } = {}) {
    super(message);
    this.name = "ProviderError";
    this.statusCode = details.statusCode ?? null;
    this.requestId = details.requestId ?? null;
    this.attempts = details.attempts ?? null;
    this.providerCode = details.providerCode ?? null;
  }
}

/** Token counts of one request. */
export interface Usage {
  inputTokens?: number;
  cachedInputTokens?: number;
  outputTokens?: number;
  [kind: string]: number | undefined;
}

/** What a generator returns for one request. */
export interface Proposal {
  /** The calls (`[]`: no call), or `null` when the output holds no complete proposal. */
  calls: Call[] | null;
  content: string | null;
  parseErrors: string[];
  /** Why the output is unreadable (`length`, `content_filter`, `refusal`, `prompt_filter`, `malformed_call`, `abnormal_stop`). */
  incomplete: string | null;
  provider?: string;
  requestedModel?: string;
  /** The model the server reports. */
  model?: string | null;
  usage?: Usage;
  latencyMs?: number;
  retryWaitMs?: number;
  apiCalls?: number;
}

/** The request a generator answers: the conversation and the tool definitions. */
export interface Example {
  messages: TextMessage[];
  tools?: unknown[];
}

/** One earlier rejected candidate and the feedback the generator reads about it. */
export interface HistoryRecord {
  round?: number;
  calls: unknown;
  feedback: Record<string, unknown>;
}

/**
 * What DM-Call needs from a model: `generate(example, history)`. Any object with this method can
 * stand in for a provider.
 */
export interface Generator {
  generate(example: Example, history?: readonly HistoryRecord[] | null): Promise<Proposal> | Proposal;
}

/** A proposal with calls in {@link Json} form (number kinds and key order kept). */
export interface JsonProposal extends Omit<Proposal, "calls"> {
  calls: Json | null;
}

/** Internal: generate with calls in {@link Json} form, so the history keeps `4.0` and key order. */
export const GENERATE_JSON = Symbol("dmcall.generateJson");

interface Decoded {
  calls: Json[] | null;
  parseErrors: string[];
  content: string | null;
  incomplete: string | null;
}

class ProtocolError extends Error {}

function countOf(value: Json | undefined): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "bigint" || value < 0n) throw new ProviderError("The response contained invalid token usage.");
  return Number(value);
}

/** Malformed arguments stay a rejected, repairable call, never a no-call. */
function decodeArguments(value: Json | undefined, encoded: boolean): [Json, string | null] {
  let args: Json | undefined = value;
  if (encoded && typeof args === "string") {
    try {
      args = /^[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/.test(args)
        ? new Map() : parse(args);
    } catch {
      return [null, "Function arguments are not valid JSON."];
    }
  }
  if (args === undefined || args === null) args = new Map();
  if (!(args instanceof Map)) return [null, "Function arguments must be a JSON object."];
  if (!isFiniteJson(args)) return [null, "Function arguments must contain finite JSON values."];
  return [args, null];
}

function appendCall(calls: Json[], errors: string[], name: Json | undefined, args: Json | undefined, encoded: boolean): void {
  if (typeof name !== "string" || !name) throw new ProviderError("The response contained an invalid function name.");
  const [value, error] = decodeArguments(args, encoded);
  calls.push(new Map<string, Json>([["name", name], ["arguments", value]]));
  if (error) errors.push(error);
}

function incomplete(reason: string): Decoded {
  return { calls: null, parseErrors: [INCOMPLETE[reason]!], content: null, incomplete: reason };
}

/** A complete proposal. A response with neither a call nor text is a decision to make no call. */
function complete(calls: Json[], errors: string[], text: string[]): Decoded {
  return { calls, parseErrors: errors, content: text.length ? text.join("\n") : null, incomplete: null };
}

/** `value in {...}` for a Python set: an unhashable value (a list or an object) raises. */
function among(value: Json | undefined, choices: readonly string[]): boolean {
  if (Array.isArray(value) || value instanceof Map) throw new ProtocolError();
  return typeof value === "string" && choices.includes(value);
}

/** Python iteration over a JSON value: list items, object keys or string characters. */
function iterate(value: Json): Json[] {
  if (Array.isArray(value)) return value;
  if (value instanceof Map) return [...value.keys()];
  if (typeof value === "string") return [...value];
  throw new ProtocolError();
}

function retryDelaySeconds(response: HttpResponse, attempt: number): number {
  const header = response.headers.get("retry-after");
  const parsed = header === null ? 2 ** attempt : pythonFloat(header);
  const delay = parsed === undefined ? 2 ** attempt : parsed;
  return Number.isFinite(delay) ? Math.max(0, Math.min(delay, 30)) : Math.min(2 ** attempt, 30);
}

function errorCode(text: string): string | null {
  let body: Json;
  try {
    body = parse(text);
  } catch {
    return null;
  }
  if (!(body instanceof Map)) return null;
  const error = body.get("error");
  let code: Json | undefined = null;
  if (error instanceof Map) {
    const value = error.get("code");
    if (typeof value === "string") code = value;
    else {
      const type = error.get("type");
      code = truthy(type) ? type : error.get("status");
    }
  }
  return typeof code === "string" && /^[A-Za-z0-9_.-]{1,64}$/.test(code) ? code : null;
}

const ID_HEADERS = ["x-request-id", "request-id", "apim-request-id", "x-goog-request-id"];

/** Options every provider takes. */
export interface ProviderOptions {
  /** API key; defaults to the provider's environment variable. */
  apiKey?: string;
  /** API base URL; defaults to the provider's public endpoint. */
  baseUrl?: string;
  /** Output token limit (4096 by default); `null` sends none where the API allows it. */
  maxOutputTokens?: number | null;
  temperature?: number | null;
  /** The generator system prompt; {@link SYSTEM_PROMPT} (the paper's) by default. */
  systemPrompt?: string;
  timeoutMs?: number;
  /** HTTP retries after the first attempt (on 408, 429 and 5xx). */
  maxRetries?: number;
  /** Extra request headers. */
  headers?: Record<string, string>;
  fetch?: FetchLike;
  sleep?: Sleep;
}

const PROVIDER_OPTIONS = ["apiKey", "baseUrl", "maxOutputTokens", "temperature", "systemPrompt", "timeoutMs", "maxRetries",
  "headers", "fetch", "sleep"];

function checkOptions(owner: string, options: object, allowed: readonly string[]): void {
  for (const key of Object.keys(options)) {
    if (!allowed.includes(key) && (options as Record<string, unknown>)[key] !== undefined) {
      throw new TypeError(`${owner} got an unknown option '${key}'.`);
    }
  }
}

/**
 * Base class: build a request, send it with bounded retries, decode a proposal.
 *
 * Subclasses define `url`, `payload()`, `decode()` and usually `authHeaders()` and `usage()`.
 */
export abstract class Provider implements Generator {
  /** Short provider name used in proposals and errors. */
  static providerName = "provider";
  static defaultBaseUrl: string | null = null;
  static baseUrlEnv: readonly string[] = [];
  static keyEnv: readonly string[] = [];
  static needsKey = true;
  protected static allowedOptions: readonly string[] = PROVIDER_OPTIONS;

  readonly name: string;
  readonly model: string;
  readonly baseUrl: string;
  readonly maxOutputTokens: number | null;
  readonly temperature: number | null;
  readonly systemPrompt: string;
  readonly timeoutMs: number;
  readonly maxRetries: number;
  readonly extraHeaders: Record<string, string>;
  readonly #apiKey: string | undefined;
  readonly #fetch: FetchLike;
  readonly #sleep: Sleep;

  constructor(model: string, options: ProviderOptions = {}) {
    const cls = new.target as unknown as typeof Provider;
    const className = cls.name;
    checkOptions(className, options, cls.allowedOptions);
    if (typeof model !== "string" || !model.trim()) throw new Error(`${className} needs a model name.`);
    const maxRetries = options.maxRetries ?? 3;
    if (!Number.isInteger(maxRetries) || maxRetries < 0) throw new Error("maxRetries must be a nonnegative integer.");
    const maxOutputTokens = options.maxOutputTokens === undefined ? 4096 : options.maxOutputTokens;
    if (maxOutputTokens !== null && (!Number.isInteger(maxOutputTokens) || maxOutputTokens < 1)) {
      throw new Error("maxOutputTokens must be a positive integer or null.");
    }
    const timeoutMs = options.timeoutMs ?? 120_000;
    if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new Error("timeoutMs must be a positive number of milliseconds.");
    }
    const temperature = options.temperature ?? null;
    if (temperature !== null && (typeof temperature !== "number" || !Number.isFinite(temperature))) {
      throw new Error("temperature must be a finite number or null.");
    }
    const systemPrompt = options.systemPrompt ?? SYSTEM_PROMPT;
    if (typeof systemPrompt !== "string" || !systemPrompt.trim()) throw new Error("systemPrompt must be nonempty text.");
    this.name = cls.providerName;
    this.model = model;
    const apiKey = options.apiKey || firstEnv(cls.keyEnv);
    if (cls.needsKey && !apiKey) {
      const names = cls.keyEnv.length ? cls.keyEnv.join(" or ") : "apiKey";
      throw new Error(`${className} needs an API key: pass apiKey or set ${names}.`);
    }
    if (apiKey !== undefined && (typeof apiKey !== "string" || apiKey.includes("\n") || apiKey.includes("\r"))) {
      throw new Error("API keys must be single-line strings.");
    }
    this.#apiKey = apiKey;
    const baseUrl = checkBaseUrl(options.baseUrl || firstEnv(cls.baseUrlEnv) || cls.defaultBaseUrl);
    if (!baseUrl) throw new Error(`${className} needs an http(s) baseUrl without credentials, query or fragment.`);
    this.baseUrl = baseUrl;
    this.maxOutputTokens = maxOutputTokens;
    this.temperature = temperature;
    this.systemPrompt = systemPrompt;
    this.timeoutMs = timeoutMs;
    this.maxRetries = maxRetries;
    this.extraHeaders = { ...(options.headers ?? {}) };
    this.#fetch = resolveFetch(options.fetch);
    this.#sleep = options.sleep ?? defaultSleep;
  }

  /** The API key, for subclasses that place it in a header. */
  protected get apiKey(): string | undefined {
    return this.#apiKey;
  }

  abstract get url(): string;

  protected authHeaders(): Record<string, string> {
    return this.#apiKey ? { Authorization: `Bearer ${this.#apiKey}` } : {};
  }

  protected abstract payload(messages: readonly TextMessage[], tools: readonly Json[], history: readonly unknown[] | null): JsonObject;

  protected abstract decode(data: JsonObject): Decoded;

  protected usage(_data: JsonObject): Usage {
    return {};
  }

  protected filteredRequest(_status: number, _code: string | null): boolean {
    return false;
  }

  get headers(): Record<string, string> {
    return { ...this.extraHeaders, ...this.authHeaders(), "Content-Type": "application/json" };
  }

  toString(): string {
    return `${this.constructor.name}(model=${JSON.stringify(this.model)}, baseUrl=${JSON.stringify(this.baseUrl)})`;
  }

  toJSON(): Record<string, unknown> {
    return { provider: this.name, model: this.model, baseUrl: this.baseUrl };
  }

  /** One proposal for `example` (`{messages, tools}`) after the given repair history. */
  async generate(example: Example, history: readonly HistoryRecord[] | null = null): Promise<Proposal> {
    const proposal = await this[GENERATE_JSON](example, history);
    return { ...proposal, calls: proposal.calls === null ? null : (toPlain(proposal.calls) as Call[]) };
  }

  /** {@link generate} with calls in {@link Json} form. */
  async [GENERATE_JSON](example: { messages: readonly TextMessage[]; tools?: readonly unknown[] },
    history: readonly unknown[] | null = null): Promise<JsonProposal> {
    const tools = (example.tools ?? []).map((tool) => fromPlain(tool));
    const body = compactText(this.payload(example.messages, tools, history));
    const started = now();
    let waited = 0;
    let attempts = 0;
    let response: HttpResponse | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      attempts++;
      try {
        response = await send(this.#fetch, this.url, "POST", this.headers, body, this.timeoutMs);
      } catch (error) {
        if (attempt === this.maxRetries) {
          throw new ProviderError(`${this.name} transport failed after ${attempts} attempts (${errorName(error)}).`,
            { attempts });
        }
        waited += await wait(this.#sleep, Math.min(2 ** attempt, 8) * 1000);
        continue;
      }
      if ((response.status === 408 || response.status === 429 || response.status >= 500) && attempt < this.maxRetries) {
        waited += await wait(this.#sleep, retryDelaySeconds(response, attempt) * 1000);
        continue;
      }
      break;
    }
    const latencyMs = Math.max(0, now() - started - waited);
    const reply = response!;
    const id = () => requestId(reply.headers, ID_HEADERS, [this.#apiKey]);
    let data: JsonObject = new Map();
    let usage: Usage = {};
    let decoded: Decoded;
    if (reply.status < 200 || reply.status >= 300) {
      const code = errorCode(reply.text);
      if (!this.filteredRequest(reply.status, code)) {
        throw new ProviderError(`${this.name} HTTP ${reply.status}${code ? ` (${code})` : ""}; check the model, key `
          + "and endpoint.", { statusCode: reply.status, requestId: id(), attempts, providerCode: code });
      }
      decoded = incomplete("prompt_filter");
    } else {
      try {
        let parsed: Json;
        try {
          parsed = parse(reply.text);
        } catch {
          throw new ProtocolError();
        }
        if (!(parsed instanceof Map)) throw new ProviderError("The response was not a JSON object.");
        data = parsed;
        decoded = this.decode(data);
        usage = this.usage(data);
      } catch (error) {
        if (error instanceof ProviderError) {
          error.statusCode = reply.status;
          error.requestId = id();
          error.attempts = attempts;
          throw error;
        }
        throw new ProviderError(`The ${this.name} response did not match its protocol.`,
          { statusCode: reply.status, requestId: id(), attempts });
      }
    }
    const reported = truthy(data.get("model")) ? data.get("model") : data.get("modelVersion");
    return {
      calls: decoded.calls,
      content: decoded.content,
      parseErrors: decoded.parseErrors,
      incomplete: decoded.incomplete,
      provider: this.name,
      requestedModel: this.model,
      model: typeof reported === "string" ? reported : null,
      usage,
      latencyMs,
      retryWaitMs: waited,
      apiCalls: attempts,
    };
  }
}

/** Options of OpenAI-style providers. */
export interface OpenAICompatibleOptions extends ProviderOptions {
  /** `reasoning_effort` (for example `"none"` for gpt-5 models). */
  reasoningEffort?: string | null;
  /** `parallel_tool_calls`; `false` by default, `null` to omit it. */
  parallelToolCalls?: boolean | null;
  /** The token-limit field: `max_completion_tokens` by default, `max_tokens` for xAI and OpenRouter. */
  maxTokensField?: string | null;
  /** Path appended to the base URL. */
  path?: string;
}

const OPENAI_OPTIONS = [...PROVIDER_OPTIONS, "reasoningEffort", "parallelToolCalls", "maxTokensField", "path"];

/**
 * Any Chat Completions endpoint with function tools: vLLM, Ollama, LM Studio, Together, Groq, ...
 *
 * `new OpenAICompatible("llama3.1", { baseUrl: "http://localhost:11434/v1" })` needs no key.
 */
export class OpenAICompatible extends Provider {
  static override providerName = "openai-compatible";
  static override needsKey = false;
  static maxTokensField = "max_completion_tokens";
  protected static override allowedOptions: readonly string[] = OPENAI_OPTIONS;

  readonly reasoningEffort: string | null;
  readonly parallelToolCalls: boolean | null;
  readonly maxTokensField: string;
  readonly path: string;

  constructor(model: string, options: OpenAICompatibleOptions = {}) {
    super(model, options);
    const path = options.path ?? "/chat/completions";
    if (typeof path !== "string" || !path.startsWith("/")) throw new Error("path must start with '/'.");
    this.reasoningEffort = options.reasoningEffort ?? null;
    this.parallelToolCalls = options.parallelToolCalls === undefined ? false : options.parallelToolCalls;
    this.maxTokensField = options.maxTokensField || (this.constructor as typeof OpenAICompatible).maxTokensField;
    this.path = path;
  }

  get url(): string {
    return this.baseUrl + this.path;
  }

  protected payload(messages: readonly TextMessage[], tools: readonly Json[], history: readonly unknown[] | null): JsonObject {
    const payload: JsonObject = new Map();
    payload.set("model", this.model);
    payload.set("messages", fromPlain([{ role: "system", content: this.systemPrompt }, ...withHistory(messages, history)]));
    const functions = functionsOf(tools);
    if (functions.length) { // a request without functions carries no tool settings
      payload.set("tools", functions.map((definition) => new Map<string, Json>([["type", "function"], ["function", definition]])));
      payload.set("tool_choice", "auto");
      if (this.parallelToolCalls !== null) payload.set("parallel_tool_calls", this.parallelToolCalls);
    }
    if (this.maxOutputTokens) payload.set(this.maxTokensField, BigInt(this.maxOutputTokens));
    if (this.reasoningEffort) payload.set("reasoning_effort", this.reasoningEffort);
    if (this.temperature !== null) payload.set("temperature", fromPlain(this.temperature));
    return payload;
  }

  protected decode(data: JsonObject): Decoded {
    const error = data.get("error");
    if (error !== undefined && error !== null) throw new ProviderError("The response contained a provider error.");
    const choices = data.get("choices");
    if (!Array.isArray(choices) || !choices.length || !(choices[0] instanceof Map)) {
      throw new ProviderError("The response did not contain a choices list.");
    }
    const choice = choices[0];
    const finish = choice.get("finish_reason");
    if (among(finish, ["length", "content_filter"])) return incomplete(finish as string);
    const message = choice.get("message");
    if (!(message instanceof Map) || (message.has("role") ? message.get("role") : "assistant") !== "assistant") {
      throw new ProviderError("The response did not contain an assistant message.");
    }
    if (truthy(message.get("refusal"))) return incomplete("refusal");
    const calls: Json[] = [];
    const errors: string[] = [];
    const text: string[] = [];
    const nativeValue = message.get("tool_calls");
    const native = truthy(nativeValue) ? nativeValue! : [];
    if (!Array.isArray(native)) throw new ProviderError("The response tool_calls must be a list.");
    for (const item of native) {
      const fn = item instanceof Map ? item.get("function") : undefined;
      if (!(fn instanceof Map)) throw new ProviderError("The response contained an invalid function-call object.");
      appendCall(calls, errors, fn.get("name"), fn.has("arguments") ? fn.get("arguments") : "{}", true);
    }
    const legacy = message.get("function_call");
    if (native.length === 0 && legacy !== undefined && legacy !== null) {
      if (!(legacy instanceof Map)) throw new ProviderError("The response contained an invalid function-call object.");
      appendCall(calls, errors, legacy.get("name"), legacy.has("arguments") ? legacy.get("arguments") : "{}", true);
    }
    const content = message.get("content");
    if (typeof content === "string" && content) text.push(content);
    else if (content !== undefined && content !== null && typeof content !== "string") {
      throw new ProviderError("The message content was not text.");
    }
    return complete(calls, errors, text);
  }

  protected override usage(data: JsonObject): Usage {
    const raw = data.get("usage");
    if (!(raw instanceof Map)) return {};
    const details = raw.get("prompt_tokens_details");
    const result: Usage = {};
    const input = countOf(raw.get("prompt_tokens"));
    const cached = details instanceof Map ? countOf(details.get("cached_tokens")) : undefined;
    const output = countOf(raw.get("completion_tokens"));
    if (input !== undefined) result.inputTokens = input;
    if (cached !== undefined) result.cachedInputTokens = cached;
    if (output !== undefined) result.outputTokens = output;
    return result;
  }

  protected override filteredRequest(status: number, code: string | null): boolean {
    return status === 400 && (code === "content_filter" || code === "invalid_prompt");
  }
}

/** OpenAI Chat Completions. Key: `OPENAI_API_KEY` (base URL: `OPENAI_BASE_URL`, optional). */
export class OpenAI extends OpenAICompatible {
  static override providerName = "openai";
  static override defaultBaseUrl = "https://api.openai.com/v1";
  static override baseUrlEnv = ["OPENAI_BASE_URL"];
  static override keyEnv = ["OPENAI_API_KEY"];
  static override needsKey = true;
}

/** xAI Grok, OpenAI-compatible. Key: `XAI_API_KEY`. */
export class XAI extends OpenAICompatible {
  static override providerName = "xai";
  static override defaultBaseUrl = "https://api.x.ai/v1";
  static override keyEnv = ["XAI_API_KEY"];
  static override needsKey = true;
  static override maxTokensField = "max_tokens";
}

/**
 * Meta Model API (Muse models such as `muse-spark-1.3`), OpenAI-compatible. Key: `MODEL_API_KEY`.
 *
 * Muse models are reasoning models: Meta rejects `reasoning_effort: "none"` for them, and
 * `tool_choice` other than `"auto"`. This provider therefore sends neither `reasoning_effort` nor
 * `parallel_tool_calls` unless you set them.
 */
export class Meta extends OpenAICompatible {
  static override providerName = "meta";
  static override defaultBaseUrl = "https://api.meta.ai/v1";
  static override keyEnv = ["MODEL_API_KEY"];
  static override needsKey = true;

  constructor(model: string, options: OpenAICompatibleOptions = {}) {
    super(model, { ...options, parallelToolCalls: options.parallelToolCalls === undefined ? null : options.parallelToolCalls });
  }
}

/** OpenRouter, one OpenAI-compatible endpoint for many models. Key: `OPENROUTER_API_KEY`. */
export class OpenRouter extends OpenAICompatible {
  static override providerName = "openrouter";
  static override defaultBaseUrl = "https://openrouter.ai/api/v1";
  static override keyEnv = ["OPENROUTER_API_KEY"];
  static override needsKey = true;
  static override maxTokensField = "max_tokens";
}

export interface AzureOpenAIOptions extends Omit<OpenAICompatibleOptions, "baseUrl"> {
  /** The resource endpoint; defaults to `AZURE_OPENAI_ENDPOINT`. */
  endpoint?: string;
  /** Defaults to `AZURE_OPENAI_API_VERSION`, then `2024-12-01-preview`. */
  apiVersion?: string;
}

/**
 * An Azure OpenAI deployment, directly or behind an API Management gateway.
 * Key: `AZURE_OPENAI_API_KEY`, endpoint: `AZURE_OPENAI_ENDPOINT`, version: `AZURE_OPENAI_API_VERSION`.
 */
export class AzureOpenAI extends OpenAICompatible {
  static override providerName = "azure";
  static override keyEnv = ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"];
  static override needsKey = true;
  protected static override allowedOptions: readonly string[] = [...OPENAI_OPTIONS, "endpoint", "apiVersion"];

  readonly apiVersion: string;

  constructor(deployment: string, options: AzureOpenAIOptions = {}) {
    const endpoint = options.endpoint || firstEnv(["AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"]);
    if (!endpoint) throw new Error("AzureOpenAI needs endpoint or AZURE_OPENAI_ENDPOINT.");
    const apiVersion = options.apiVersion || firstEnv(["AZURE_OPENAI_API_VERSION"]) || "2024-12-01-preview";
    if (!/^\d{4}-\d{2}-\d{2}(?:-preview)?$/.test(apiVersion)) throw new Error("apiVersion must look like 2024-12-01-preview.");
    if (typeof deployment !== "string" || deployment === "." || deployment === ".." || /[\x00-\x1f]/.test(deployment)) {
      throw new Error("The deployment name contains invalid characters.");
    }
    const { endpoint: _endpoint, apiVersion: _version, ...rest } = options;
    super(deployment, { ...rest, baseUrl: endpoint });
    this.apiVersion = apiVersion;
  }

  override get url(): string {
    return `${this.baseUrl}/openai/deployments/${quote(this.model)}/chat/completions?api-version=${this.apiVersion}`;
  }

  protected override authHeaders(): Record<string, string> {
    return { "api-key": this.apiKey! };
  }
}

/** Anthropic Messages API. Key: `ANTHROPIC_API_KEY` (base URL: `ANTHROPIC_BASE_URL`, optional). */
export class Anthropic extends Provider {
  static override providerName = "anthropic";
  static override defaultBaseUrl = "https://api.anthropic.com/v1";
  static override baseUrlEnv = ["ANTHROPIC_BASE_URL"];
  static override keyEnv = ["ANTHROPIC_API_KEY"];
  static readonly version = "2023-06-01";

  get url(): string {
    return `${this.baseUrl}/messages`;
  }

  protected override authHeaders(): Record<string, string> {
    return { "x-api-key": this.apiKey!, "anthropic-version": Anthropic.version };
  }

  protected payload(messages: readonly TextMessage[], tools: readonly Json[], history: readonly unknown[] | null): JsonObject {
    const system = [this.systemPrompt];
    const conversation: TextMessage[] = [];
    for (const message of withHistory(messages, history)) {
      if (message.role === "system" || message.role === "developer") {
        if (typeof message.content !== "string") throw new Error("Anthropic needs text system messages.");
        system.push(message.content);
      } else if (message.role === "user" || message.role === "assistant") {
        conversation.push(message);
      } else {
        throw new Error("Anthropic accepts only user and assistant conversation messages.");
      }
    }
    const payload: JsonObject = new Map<string, Json>([
      ["model", this.model],
      ["system", system.join("\n\n")],
      ["messages", fromPlain(conversation)],
      ["max_tokens", BigInt(this.maxOutputTokens || 4096)],
    ]);
    const functions = functionsOf(tools);
    if (functions.length) {
      payload.set("tools", functions.map((definition) => new Map<string, Json>([
        ["name", definition.get("name")!],
        ["description", definition.has("description") ? definition.get("description")! : ""],
        ["input_schema", definition.has("parameters") ? definition.get("parameters")!
          : new Map<string, Json>([["type", "object"], ["properties", new Map()]])],
      ])));
      payload.set("tool_choice", new Map<string, Json>([["type", "auto"], ["disable_parallel_tool_use", true]]));
    }
    if (this.temperature !== null) payload.set("temperature", fromPlain(this.temperature));
    return payload;
  }

  protected decode(data: JsonObject): Decoded {
    const error = data.get("error");
    if (data.get("type") === "error" || (error !== undefined && error !== null)) {
      throw new ProviderError("The response contained a provider error.");
    }
    const content = data.get("content");
    if (data.get("role") !== "assistant" || !Array.isArray(content)) {
      throw new ProviderError("The Anthropic response was not an assistant message.");
    }
    const details = data.get("stop_details");
    const reason = data.get("stop_reason");
    if (among(reason, ["max_tokens", "model_context_window_exceeded"])) return incomplete("length");
    if (reason === "refusal" || (details instanceof Map && details.get("type") === "refusal")) return incomplete("refusal");
    const calls: Json[] = [];
    const errors: string[] = [];
    const text: string[] = [];
    for (const block of content) {
      if (!(block instanceof Map)) throw new ProviderError("An Anthropic content block was malformed.");
      if (block.get("type") === "tool_use") appendCall(calls, errors, block.get("name"), block.get("input"), false);
      else if (block.get("type") === "text") {
        const value = block.get("text");
        if (typeof value === "string" && value) text.push(value);
      }
    }
    return complete(calls, errors, text);
  }

  protected override usage(data: JsonObject): Usage {
    const raw = data.get("usage");
    if (!(raw instanceof Map)) return {};
    const read = countOf(raw.get("cache_read_input_tokens")) || 0;
    const written = countOf(raw.get("cache_creation_input_tokens")) || 0;
    const base = countOf(raw.get("input_tokens"));
    const output = countOf(raw.get("output_tokens"));
    const result: Usage = { cachedInputTokens: read };
    if (output !== undefined) result.outputTokens = output;
    if (base !== undefined) result.inputTokens = base + read + written;
    return result;
  }
}

export interface GeminiOptions extends ProviderOptions {
  /** `thinkingConfig.thinkingBudget`. */
  thinkingBudget?: number | null;
}

const FILTERED = ["SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "LANGUAGE"];

/** Google Gemini API (`generateContent` with function declarations). Key: `GEMINI_API_KEY` or `GOOGLE_API_KEY`. */
export class Gemini extends Provider {
  static override providerName = "gemini";
  static override defaultBaseUrl = "https://generativelanguage.googleapis.com/v1beta";
  static override keyEnv = ["GEMINI_API_KEY", "GOOGLE_API_KEY"];
  protected static override allowedOptions: readonly string[] = [...PROVIDER_OPTIONS, "thinkingBudget"];

  readonly thinkingBudget: number | null;

  constructor(model: string, options: GeminiOptions = {}) {
    const { thinkingBudget, ...rest } = options;
    super(typeof model === "string" && model.startsWith("models/") ? model.slice(7) : model, rest);
    if (thinkingBudget !== undefined && thinkingBudget !== null && !Number.isInteger(thinkingBudget)) {
      throw new Error("thinkingBudget must be an integer or null.");
    }
    this.thinkingBudget = thinkingBudget ?? null;
  }

  get url(): string {
    return `${this.baseUrl}/models/${quote(this.model)}:generateContent`;
  }

  protected override authHeaders(): Record<string, string> {
    return { "x-goog-api-key": this.apiKey! };
  }

  private static text(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) { // OpenAI-style text parts
      return content.filter((part) => part !== null && typeof part === "object" && !Array.isArray(part))
        .map((part) => {
          const value = (part as Record<string, unknown>).text;
          if (value === undefined) return "";
          if (typeof value !== "string") throw new TypeError("Gemini message parts need text.");
          return value;
        }).join("\n");
    }
    throw new Error("Gemini messages need text content.");
  }

  protected payload(messages: readonly TextMessage[], tools: readonly Json[], history: readonly unknown[] | null): JsonObject {
    const system = [this.systemPrompt];
    const contents: Json[] = [];
    for (const message of withHistory(messages, history)) {
      if (message.role === "system" || message.role === "developer") system.push(Gemini.text(message.content));
      else if (message.role === "user" || message.role === "assistant") {
        contents.push(new Map<string, Json>([
          ["role", message.role === "assistant" ? "model" : "user"],
          ["parts", [new Map<string, Json>([["text", Gemini.text(message.content)]])]],
        ]));
      } else {
        throw new Error("Gemini accepts only user and assistant conversation messages.");
      }
    }
    const payload: JsonObject = new Map<string, Json>([
      ["systemInstruction", new Map<string, Json>([["parts", [new Map<string, Json>([["text", system.join("\n\n")]])]]])],
      ["contents", contents],
    ]);
    const functions = functionsOf(tools);
    if (functions.length) {
      payload.set("tools", [new Map<string, Json>([["functionDeclarations", functions.map((definition) => new Map<string, Json>([
        ["name", definition.get("name")!],
        ["description", definition.has("description") ? definition.get("description")! : ""],
        ["parametersJsonSchema", definition.has("parameters") ? definition.get("parameters")!
          : new Map<string, Json>([["type", "object"], ["properties", new Map()]])],
      ]))]])]);
      payload.set("toolConfig", new Map<string, Json>([
        ["functionCallingConfig", new Map<string, Json>([["mode", "AUTO"]])],
      ]));
    }
    const config: JsonObject = new Map();
    if (this.maxOutputTokens) config.set("maxOutputTokens", BigInt(this.maxOutputTokens));
    if (this.temperature !== null) config.set("temperature", fromPlain(this.temperature));
    if (this.thinkingBudget !== null) {
      config.set("thinkingConfig", new Map<string, Json>([["thinkingBudget", fromPlain(this.thinkingBudget)]]));
    }
    if (config.size) payload.set("generationConfig", config);
    return payload;
  }

  protected decode(data: JsonObject): Decoded {
    const error = data.get("error");
    if (error !== undefined && error !== null) throw new ProviderError("The response contained a provider error.");
    const candidates = data.get("candidates");
    if (!truthy(candidates)) {
      const feedback = data.get("promptFeedback");
      if (feedback instanceof Map && truthy(feedback.get("blockReason"))) return incomplete("prompt_filter");
      throw new ProviderError("The Gemini response had no candidates.");
    }
    let candidate: Json | undefined;
    if (Array.isArray(candidates)) candidate = candidates[0];
    else if (typeof candidates === "string") candidate = candidates[0];
    else throw new ProtocolError();
    if (!(candidate instanceof Map)) throw new ProtocolError();
    const reason = candidate.get("finishReason");
    if (reason === "MAX_TOKENS") return incomplete("length");
    if (among(reason, FILTERED)) return incomplete("content_filter");
    if (among(reason, ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])) return incomplete("malformed_call");
    const contentValue = candidate.get("content");
    if (reason !== undefined && reason !== null && !among(reason, ["STOP", "FINISH_REASON_UNSPECIFIED"])) {
      return incomplete("abnormal_stop");
    }
    if (reason !== "STOP" && !(contentValue instanceof Map)) return incomplete("abnormal_stop");
    const content: Json = truthy(contentValue) ? contentValue! : new Map();
    if (!(content instanceof Map)) throw new ProtocolError();
    const partsValue = content.get("parts");
    const parts = truthy(partsValue) ? iterate(partsValue!) : [];
    const calls: Json[] = [];
    const errors: string[] = [];
    const text: string[] = [];
    for (const part of parts) {
      if (!(part instanceof Map)) throw new ProviderError("A Gemini content part was malformed.");
      const call = part.get("functionCall");
      const value = part.get("text");
      if (part.has("functionCall") && !(call instanceof Map)) return incomplete("malformed_call");
      if (call instanceof Map) appendCall(calls, errors, call.get("name"), call.get("args"), false);
      else if (typeof value === "string" && value && !truthy(part.get("thought"))) text.push(value);
    }
    return complete(calls, errors, text);
  }

  protected override usage(data: JsonObject): Usage {
    const raw = data.get("usageMetadata");
    if (!(raw instanceof Map)) return {};
    const output = countOf(raw.get("candidatesTokenCount"));
    const thoughts = countOf(raw.get("thoughtsTokenCount")) || 0;
    const input = countOf(raw.get("promptTokenCount"));
    const cached = countOf(raw.get("cachedContentTokenCount"));
    const result: Usage = {};
    if (input !== undefined) result.inputTokens = input;
    if (cached !== undefined) result.cachedInputTokens = cached;
    if (!(output === undefined && !thoughts)) result.outputTokens = (output || 0) + thoughts;
    return result;
  }
}

/** A provider class: `new Cls(model, options)`. */
export type ProviderClass = new (model: string, options?: AnyProviderOptions) => Provider;

/** Provider names a `"provider:model"` spec can use. */
export const PROVIDERS: Readonly<Record<string, ProviderClass>> = Object.freeze({
  openai: OpenAI,
  azure: AzureOpenAI,
  anthropic: Anthropic,
  claude: Anthropic,
  gemini: Gemini,
  google: Gemini,
  xai: XAI,
  grok: XAI,
  openrouter: OpenRouter,
  meta: Meta,
} as unknown as Record<string, ProviderClass>);

/** Options for {@link provider}: whatever the chosen provider class takes. */
export type AnyProviderOptions = OpenAICompatibleOptions & AzureOpenAIOptions & GeminiOptions;

function hasGenerate(value: unknown): value is Generator {
  return value !== null && typeof value === "object" && typeof (value as Generator).generate === "function";
}

/**
 * A provider from `"provider:model"`, for example `"openai:gpt-5-nano"` or `"azure:<deployment>"`.
 *
 * Options go to the provider (`apiKey`, `baseUrl`, `endpoint`, ...). A provider instance, or any
 * object with `generate(example, history)`, passes through unchanged.
 */
export function provider(spec: string | Generator, options: AnyProviderOptions = {}): Generator {
  if (typeof spec !== "string") {
    if (hasGenerate(spec)) {
      if (Object.values(options).some((value) => value !== undefined)) {
        throw new Error("Pass options to the provider itself, not with an instance.");
      }
      return spec;
    }
    throw new TypeError('A model is a "provider:model" string or a provider object.');
  }
  const separator = spec.indexOf(":");
  const name = separator === -1 ? spec : spec.slice(0, separator);
  const model = separator === -1 ? "" : spec.slice(separator + 1);
  if (separator === -1 || !model) {
    throw new Error('Use "provider:model", e.g. "openai:gpt-5-nano", "anthropic:claude-sonnet-4-5", '
      + '"gemini:gemini-2.5-flash", "xai:grok-4", "meta:muse-spark-1.3" or "azure:<deployment>".');
  }
  const cls = Object.prototype.hasOwnProperty.call(PROVIDERS, name.toLowerCase()) ? PROVIDERS[name.toLowerCase()] : undefined;
  if (!cls) {
    throw new Error(`Unknown provider '${name}'; choose from ${Object.keys(PROVIDERS).sort().join(", ")}, or pass `
      + "new OpenAICompatible(model, { baseUrl }) for any other endpoint.");
  }
  return new cls(model, options);
}
