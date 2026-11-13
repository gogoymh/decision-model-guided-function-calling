/**
 * The local schema check S(c) of the paper.
 *
 * S(c) is true when the candidate is readable, names an available function and supplies every
 * required argument with its declared JSON type. Value constraints (enum, ranges, lengths,
 * patterns, formats) are not checked here: whether a value fits the request and the parameter
 * description is the decision model's question. A value equal to a parameter's declared default is
 * admissible even when its JSON type differs (for example `null`), and, as in a Python call, keyword
 * names outside a closed parameter list are rejected. An empty collection (no call) passes.
 */

import { type Json, type JsonObject, clone, compareCodePoints, fromPlain, isFiniteJson, stringify, toPlain } from "./json.js";
import {
  type Dialect, type ValidationIssue, ReferenceError, SchemaError, compiledSchema, pathKey, validationErrors,
} from "./jsonschema.js";

/** One schema check error: which question it bears on, where, and Python jsonschema's message. */
export interface SchemaIssue {
  check: "function" | "arguments";
  path: string;
  message: string;
}

/** The schema check's feedback record, as the generator would read it. */
export interface SchemaFeedback {
  source: "deterministic_schema_validation";
  function_pass: boolean;
  arguments_pass: boolean;
  schema_errors: SchemaIssue[];
}

/** The result of the schema check S(c). */
export interface SchemaCheck {
  /** No error concerns the function choice. */
  functionPass: boolean;
  /** No error concerns the arguments. */
  argumentsPass: boolean;
  /** S(c): both pass. */
  accepted: boolean;
  /** The candidate is readable, so the decision model can judge it. */
  wellFormed: boolean;
  errors: SchemaIssue[];
  feedback: SchemaFeedback;
}

const DEFAULT_PARAMETERS = (): JsonObject => new Map<string, Json>([["type", "object"], ["properties", new Map()]]);

/**
 * The observable part of each tool: name, description and parameters. Any other field of a tool
 * definition stays out of what the decision model sees.
 */
export function functionDefinitions(tools: Json): JsonObject[] {
  if (!Array.isArray(tools)) throw new Error("tools must be a list of function definitions");
  const names = new Set<string>();
  return tools.map((tool) => {
    if (!(tool instanceof Map)) throw new Error("Every tool must be a function definition object");
    const definition = tool.has("function") ? tool.get("function") : tool;
    if (!(definition instanceof Map)) throw new Error("Every tool.function must be an object");
    const name = definition.get("name");
    if (typeof name !== "string" || !name || names.has(name)) {
      throw new Error("Available function names must be nonempty and unique");
    }
    names.add(name);
    return new Map<string, Json>([
      ["name", name],
      ["description", definition.has("description") ? clone(definition.get("description")!) : ""],
      ["parameters", definition.has("parameters") ? clone(definition.get("parameters")!) : DEFAULT_PARAMETERS()],
    ]);
  });
}

function issue(check: "function" | "arguments", path: string, message: string): SchemaIssue {
  return { check, path, message };
}

function canonical(value: Json): string | null {
  try {
    return stringify(value, { sortKeys: true });
  } catch {
    return null;
  }
}

/** A value equal to the parameter's own declared default is admissible. */
function isDeclaredDefault(error: ValidationIssue): boolean {
  if (error.validator !== "type" || !(error.schema instanceof Map) || !error.schema.has("default")) return false;
  const instance = canonical(error.instance);
  return instance !== null && instance === canonical(error.schema.get("default")!);
}

/** S(c) on calls in {@link Json} form. Invalid tool definitions throw: they are configuration errors. */
export function checkSchemaJson(tools: Json, calls: Json): SchemaCheck {
  const definitions = functionDefinitions(tools);
  const validators = new Map<string, { schema: Json; dialect: Dialect }>();
  const schemas = new Map<string, JsonObject>();
  for (const definition of definitions) {
    const name = definition.get("name") as string;
    const schema = definition.get("parameters")!;
    if (!(schema instanceof Map)) throw new Error("Function parameters must be a JSON Schema object");
    let compiled: { schema: Json; dialect: Dialect };
    try {
      compiled = compiledSchema(schema);
    } catch (error) {
      if (error instanceof ReferenceError) throw new Error("An available function's JSON Schema has a $ref that cannot be resolved");
      if (error instanceof SchemaError) throw new Error("An available function has an invalid JSON Schema");
      throw error;
    }
    validators.set(name, compiled);
    schemas.set(name, schema);
  }

  const errors: SchemaIssue[] = [];
  let wellFormed = Array.isArray(calls);
  if (!Array.isArray(calls)) {
    errors.push(issue("function", "calls", "Proposed calls must be a list."),
      issue("arguments", "calls", "No valid call collection to inspect."));
  } else {
    calls.forEach((call, index) => {
      const path = `calls[${index}]`;
      if (!(call instanceof Map)) {
        wellFormed = false;
        errors.push(issue("function", path, "Each call must be an object."),
          issue("arguments", path, "No argument object to inspect."));
        return;
      }
      const name = call.get("name");
      if (typeof name !== "string" || !name) {
        wellFormed = false;
        errors.push(issue("function", `${path}.name`, "Name must be a nonempty string."));
      } else if (!validators.has(name)) {
        errors.push(issue("function", `${path}.name`, "Selected function is not available."));
      }
      const args = call.get("arguments");
      if (!(args instanceof Map)) {
        wellFormed = false;
        errors.push(issue("arguments", `${path}.arguments`, "Arguments must be a JSON object."));
        return;
      }
      if (!isFiniteJson(args)) {
        wellFormed = false;
        errors.push(issue("arguments", `${path}.arguments`, "Arguments must be finite JSON values."));
        return;
      }
      if (typeof name !== "string" || !validators.has(name)) {
        errors.push(issue("arguments", `${path}.arguments`, "Cannot validate arguments for an unavailable function."));
        return;
      }
      const { schema, dialect } = validators.get(name)!;
      const found = validationErrors(schema, args, dialect)
        .map((error, order) => ({ error, order, key: pathKey(error.path) }))
        .sort((a, b) => compareCodePoints(a.key, b.key) || a.order - b.order);
      for (const { error } of found) {
        if (isDeclaredDefault(error)) continue;
        const suffix = error.path.map((part) => `[${stringify(part)}]`).join("");
        errors.push(issue("arguments", `${path}.arguments${suffix}`, error.message));
      }
      const original = schemas.get(name)!;
      // An ordinary function signature has closed top-level keyword names.
      // Respect an explicit additionalProperties or patternProperties rule.
      const properties = original.get("properties");
      if (original.has("properties") && !original.has("additionalProperties") && !original.has("patternProperties")) {
        const declared = properties instanceof Map ? properties : new Map();
        const unexpected = [...args.keys()].filter((key) => !declared.has(key)).sort(compareCodePoints);
        for (const key of unexpected) errors.push(issue("arguments", `${path}.arguments`, `Unknown parameter: ${key}.`));
      }
    });
  }

  const functionPass = !errors.some((item) => item.check === "function");
  const argumentsPass = !errors.some((item) => item.check === "arguments");
  return {
    functionPass,
    argumentsPass,
    accepted: functionPass && argumentsPass,
    wellFormed,
    errors,
    feedback: {
      source: "deterministic_schema_validation",
      function_pass: functionPass,
      arguments_pass: argumentsPass,
      schema_errors: errors.map((item) => ({ ...item })),
    },
  };
}

/**
 * Run S(c) on proposed `calls` (`[{name, arguments}]`) against the available `tools`.
 *
 * Returns `accepted` (S(c)), `functionPass` and `argumentsPass` (which part failed), `wellFormed`
 * (the candidate is readable, so the decision model can judge it), `errors` and the `feedback`
 * record of the check. Invalid tool definitions throw: they are configuration errors, not a
 * property of the candidate.
 */
export function checkSchema(tools: unknown, calls: unknown): SchemaCheck {
  return checkSchemaJson(fromPlain(tools), fromPlain(calls));
}

/** Plain copies of schema errors for public results. */
export function plainIssues(issues: readonly SchemaIssue[]): SchemaIssue[] {
  return toPlain(fromPlain(issues)) as SchemaIssue[];
}
