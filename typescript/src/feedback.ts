/**
 * Feedback representations: how the decision model's scores reach the generator (Φ in the paper).
 *
 * Only the generator's copy of earlier feedback changes. Acceptance always uses the numeric
 * probabilities, and `Result.trace` keeps them unchanged.
 *
 * - `"hybrid"` (the default): each probability keeps its value and threshold and gains a verbal
 *   label, Φ(p) = (p, ℓ(p)).
 * - `"numeric"`: probabilities and thresholds only, Φ(p) = p.
 * - `"text"`: labels only, Φ(p) = ℓ(p). Probabilities and thresholds are removed.
 *
 * ℓ uses five bands of equal width, lower bound inclusive; the last band includes 1. A question
 * that was not asked (the argument question on a no call) keeps a null score and a null label.
 */

import { clone, Float } from "./json.js";

export type FeedbackRepresentation = "hybrid" | "numeric" | "text";
export const FEEDBACK: readonly FeedbackRepresentation[] = Object.freeze(["hybrid", "numeric", "text"]);
export const CUT_POINTS: readonly number[] = Object.freeze([0.2, 0.4, 0.6, 0.8]);
export const LABELS: readonly string[] = Object.freeze([
  "very likely incorrect", "likely incorrect", "uncertain", "likely correct", "very likely correct",
]);

/** Where a probability's label goes. Decision-model records use `probability_yes`. */
export const LABEL_FIELDS: Readonly<Record<string, string>> = Object.freeze({
  probability_yes: "feedback_label",
  function_probability: "function_label",
  arguments_probability: "arguments_label",
  p_f: "function_label",
  p_a: "arguments_label",
});
const THRESHOLDS = new Set(["threshold", "function_threshold", "arguments_threshold"]);
/** Names of the two decision-model checks in a feedback record. */
const CHECK_RECORDS: Readonly<Record<string, string>> = { function: "function_selection", arguments: "arguments" };

type Plain = Record<string, unknown>;

function isRecord(value: unknown): value is Plain {
  return value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof Map)
    && !(value instanceof Float);
}

function has(record: Plain, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

/** ℓ(p): the verbal band of a probability; `null` for a question that was not asked. */
export function label(probability: number | bigint | Float | null | undefined): string | null {
  if (probability === null || probability === undefined) return null;
  const value = typeof probability === "bigint" ? Number(probability)
    : probability instanceof Float ? probability.value : probability;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("Feedback probabilities must be finite numbers between 0 and 1, or null.");
  }
  return LABELS[CUT_POINTS.filter((cut) => cut <= value).length]!;
}

function checkRepresentation(representation: unknown): FeedbackRepresentation {
  if (!FEEDBACK.includes(representation as FeedbackRepresentation)) {
    throw new Error(`feedback must be one of ${FEEDBACK.map((name) => `'${name}'`).join(", ")}.`);
  }
  return representation as FeedbackRepresentation;
}

function hybrid(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(hybrid);
  if (!isRecord(value)) return clone(value);
  const result: Plain = {};
  for (const key of Object.keys(value)) {
    const item = value[key];
    // Schema errors are diagnostics about the task input, not scores.
    result[key] = key === "schema_errors" ? clone(item) : hybrid(item);
    const target = LABEL_FIELDS[key];
    if (target !== undefined && has(LABEL_FIELDS, key)) {
      if (has(value, target) || has(result, target)) {
        throw new Error("Repair feedback has conflicting added judgment fields.");
      }
      result[target] = label(item as number | null);
    }
  }
  return result;
}

function text(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(text);
  if (!isRecord(value)) return clone(value);
  const result: Plain = {};
  for (const key of Object.keys(value)) {
    const item = value[key];
    if (THRESHOLDS.has(key)) continue;
    if (has(LABEL_FIELDS, key)) {
      const target = LABEL_FIELDS[key]!;
      if (has(value, target) || has(result, target)) throw new Error("Repair feedback has conflicting judgment fields.");
      result[target] = label(item as number | null);
    } else if (key === "instruction" && typeof item === "string") {
      result[key] = item.replaceAll("The probabilities", "The feedback labels");
    } else if (key === "schema_errors") {
      result[key] = clone(item);
    } else {
      result[key] = text(item);
    }
  }
  return result;
}

/** The generator's copy of one feedback record under a representation. The input is not changed. */
export function represent<T extends object>(feedback: T, representation: FeedbackRepresentation = "hybrid"): T {
  checkRepresentation(representation);
  if (!isRecord(feedback)) throw new Error("Repair feedback must be an object.");
  if (representation === "numeric") return clone(feedback);
  return (representation === "text" ? text(feedback) : hybrid(feedback)) as T;
}

/** Every record of a repair history under a representation, in the same order. */
export function representHistory<T extends { feedback: object }>(
  history: readonly T[], representation: FeedbackRepresentation = "hybrid",
): T[] {
  checkRepresentation(representation);
  return history.map((record) => {
    if (!isRecord(record) || !isRecord((record as Plain).feedback)) {
      throw new Error("Each repair history record needs a feedback object.");
    }
    const copy = clone(record) as Plain;
    copy.feedback = represent(record.feedback, representation);
    return copy as T;
  });
}

/**
 * A feedback record without the other question, for DM-Call-Function and DM-Call-Argument.
 *
 * The source, schema errors, thresholds and instruction stay; only the hidden question's verdict
 * and score are removed.
 */
export function singleCheck<T extends object>(feedback: T, keep: "function" | "arguments"): T {
  if (keep !== "function" && keep !== "arguments") throw new Error("keep must be 'function' or 'arguments'.");
  const hidden = keep === "function" ? "arguments" : "function";
  const hiddenRecord = CHECK_RECORDS[hidden]!;
  const result = clone(feedback) as Plain;
  if (isRecord(result.checks)) {
    const checks: Plain = {};
    for (const name of Object.keys(result.checks)) {
      if (name !== hiddenRecord) checks[name] = result.checks[name];
    }
    result.checks = checks;
  }
  if (Array.isArray(result.failed_checks)) {
    result.failed_checks = result.failed_checks.filter((name) => name !== hiddenRecord);
  }
  for (const key of Object.keys(result)) {
    if (key.startsWith(`${hidden}_`)) delete result[key];
  }
  return result as T;
}
