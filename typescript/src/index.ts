/**
 * DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback.
 *
 * An LLM proposes a function call; a decision model checks whether the function and the arguments
 * are right before anything runs; a rejected call goes back to the same LLM with every earlier
 * candidate and its feedback, until a candidate passes or the repair limit is reached.
 *
 * ```ts
 * import { DMCall, tool } from "dmcall";
 *
 * const getWeather = tool({
 *   name: "get_weather",
 *   description: "Current weather for a city.",
 *   parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
 *   execute: async ({ city }) => ({ city, temp: 12 }),
 * });
 * const agent = new DMCall("openai:gpt-5-nano", { tools: [getWeather] }); // OPENAI_API_KEY, TYPESAFE_API_KEY
 * const result = await agent.run("Is it warm in Seoul right now?");
 * result.call; result.accepted; result.output;
 * ```
 */

export {
  CHECKS, DEFAULT_MAX_REPAIRS, DEFAULT_THRESHOLD, DMCall, HISTORIES, ON_EXHAUSTION, Result, Verdict, verify,
} from "./core.js";
export type {
  Attempt, CallOptions, Checks, DMCallOptions, History, Messages, OnExhaustion, PreservedNoCall, RunOptions, TraceStep,
  VerifyOptions,
} from "./core.js";
export { DecisionModel, DecisionModelError, Jev, PPLXDecider, RETRY_STATUSES } from "./decision.js";
export type {
  Answer, DecisionModelOptions, DecisionState, FeedbackCheck, FeedbackRecord, Judge, Judgment, JevOptions,
  PPLXDeciderOptions, Provenance, Questions,
} from "./decision.js";
export { CUT_POINTS, FEEDBACK, LABEL_FIELDS, LABELS, label, represent, representHistory, singleCheck } from "./feedback.js";
export type { FeedbackRepresentation } from "./feedback.js";
export type { FetchLike, RequestInitLike, ResponseLike, Sleep } from "./http.js";
export {
  ARGUMENT_QUESTION, FUNCTION_QUESTION, HISTORY_FOOTER, HISTORY_HEADER, REPAIR_INSTRUCTION, SYSTEM_PROMPT, historyMessage,
} from "./prompts.js";
export type { Question, TextMessage } from "./prompts.js";
export {
  Anthropic, AzureOpenAI, Gemini, INCOMPLETE, Meta, OpenAI, OpenAICompatible, OpenRouter, PROVIDERS, Provider, ProviderError,
  XAI, provider,
} from "./providers.js";
export type {
  AnyProviderOptions, AzureOpenAIOptions, Example, GeminiOptions, Generator, HistoryRecord, OpenAICompatibleOptions,
  Proposal, ProviderClass, ProviderOptions, Usage,
} from "./providers.js";
export { checkSchema } from "./schema.js";
export type { SchemaCheck, SchemaFeedback, SchemaIssue } from "./schema.js";
export { Tool, normalizeCalls, schemaOf, tool } from "./tools.js";
export type { Call, FunctionTool, JsonSchema, ToolHandler, ToolInput, ToolSpec } from "./tools.js";

export const VERSION = "1.0.0";
