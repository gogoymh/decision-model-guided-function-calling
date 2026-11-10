/**
 * Tools, and calls from any SDK.
 *
 * A tool is a function the model may call: a name, a description and a JSON Schema for its
 * arguments. `tool({ name, description, parameters, execute })` pairs a definition with the code
 * that runs it. Plain definitions in OpenAI (Chat Completions or Responses), Anthropic
 * (`input_schema`) or Gemini (`parameters` with upper-case types, or `parametersJsonSchema`) shape
 * work too; give their handlers with the `functions` option.
 */

import { type Json, type JsonObject, clone, fromPlain, isFiniteJson, parse, stringRepr, toPlain, truthy } from "./json.js";
import { SchemaError, ReferenceError, compiledSchema } from "./jsonschema.js";

/** The code behind a tool. It receives the call's arguments as one object and may be async. */
export type ToolHandler = (args: Record<string, unknown>) => unknown;

/** A JSON Schema object (plain data). */
export type JsonSchema = Record<string, unknown>;

/** A function call: `{ name, arguments }`. `arguments` is `null` when the model's arguments were unreadable. */
export interface Call {
  name: string;
  arguments: Record<string, unknown> | null;
}

/** The function definition DM-Call sends to models, in OpenAI Chat Completions shape. */
export interface FunctionTool {
  type: "function";
  function: { name: string; description: string; parameters: JsonSchema };
}

export interface ToolSpec {
  name: string;
  description?: string;
  /** JSON Schema of the arguments. Defaults to an object with no parameters. */
  parameters?: JsonSchema;
  /** Runs an accepted call. Without it, give the handler with the `functions` option. */
  execute?: ToolHandler;
}

/** A tool definition paired with its handler; made by {@link tool}. */
export class Tool implements FunctionTool {
  readonly type = "function" as const;
  readonly function: { name: string; description: string; parameters: JsonSchema };
  readonly execute: ToolHandler | undefined;

  constructor(definition: { name: string; description: string; parameters: JsonSchema }, execute?: ToolHandler) {
    this.function = definition;
    this.execute = execute;
    Object.freeze(this);
  }

  toJSON(): FunctionTool {
    return { type: "function", function: this.function };
  }
}

/** Anything accepted where a tool is expected. */
export type ToolInput = Tool | FunctionTool | Record<string, unknown>;

function checked(definition: JsonObject): JsonObject {
  const name = definition.get("name");
  if (typeof name !== "string" || !name.trim()) throw new Error("Every tool needs a nonempty name.");
  if (typeof definition.get("description") !== "string") {
    throw new Error(`Tool ${stringRepr(name)}: the description must be text.`);
  }
  const parameters = definition.get("parameters");
  const type = parameters instanceof Map ? (parameters.has("type") ? parameters.get("type") : "object") : undefined;
  if (!(parameters instanceof Map) || type !== "object") {
    throw new Error(`Tool ${stringRepr(name)}: parameters must be a JSON Schema object with type 'object'.`);
  }
  if (!isFiniteJson(parameters)) {
    throw new Error(`Tool ${stringRepr(name)}: parameters are not a valid JSON Schema (ValueError).`);
  }
  try {
    compiledSchema(parameters);
  } catch (error) {
    if (error instanceof ReferenceError) {
      throw new Error(`Tool ${stringRepr(name)}: a $ref in its parameters cannot be resolved inside them.`);
    }
    if (error instanceof SchemaError) {
      throw new Error(`Tool ${stringRepr(name)}: parameters are not a valid JSON Schema (SchemaError).`);
    }
    throw error;
  }
  return definition;
}

/** Gemini's OpenAPI-style schemas spell types in capitals (OBJECT, STRING); JSON Schema does not. */
function isUpper(text: string): boolean {
  return text.toUpperCase() === text && text.toLowerCase() !== text;
}

const NAMED_SCHEMAS = new Set(["properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"]);
const CHILD_SCHEMAS = new Set(["items", "prefixItems", "additionalItems", "additionalProperties", "unevaluatedItems",
  "unevaluatedProperties", "contains", "propertyNames", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "contentSchema"]);

function lowerTypes(schema: Json): Json {
  if (Array.isArray(schema)) return schema.map(lowerTypes);
  if (!(schema instanceof Map)) return schema;
  const result: JsonObject = new Map();
  for (const [key, value] of schema) {
    if (key === "type" && typeof value === "string" && isUpper(value)) result.set(key, value.toLowerCase());
    else if (key === "type" && Array.isArray(value)) {
      result.set(key, value.map((item) => (typeof item === "string" && isUpper(item) ? item.toLowerCase() : item)));
    } else if (NAMED_SCHEMAS.has(key) && value instanceof Map) {
      result.set(key, new Map([...value].map(([name, item]) => [name, lowerTypes(item)])));
    } else if (CHILD_SCHEMAS.has(key)) result.set(key, lowerTypes(value));
    else result.set(key, value);
  }
  return result;
}

/**
 * Pair a JSON Schema definition with the code that runs it.
 *
 * ```ts
 * const getWeather = tool({
 *   name: "get_weather",
 *   description: "Current weather for a city.",
 *   parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
 *   execute: async ({ city }) => ({ city, temp: 21 }),
 * });
 * ```
 */
export function tool(spec: ToolSpec): Tool {
  if (spec === null || typeof spec !== "object") throw new TypeError("tool() takes { name, description, parameters, execute }.");
  if (spec.execute !== undefined && typeof spec.execute !== "function") {
    throw new TypeError(`Tool ${stringRepr(String(spec.name))}: execute must be a function.`);
  }
  const definition = checked(new Map<string, Json>([
    ["name", fromPlain(spec.name)],
    ["description", spec.description === undefined ? "" : fromPlain(spec.description)],
    ["parameters", spec.parameters === undefined ? new Map<string, Json>([["type", "object"], ["properties", new Map()]])
      : fromPlain(spec.parameters)],
  ]));
  return new Tool(toPlain(definition) as Tool["function"], spec.execute);
}

/** A tool definition as `{type: "function", function: {...}}` in {@link Json} form. */
export function schemaOfJson(input: unknown): JsonObject {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    throw new TypeError("A tool is a definition object or dmcall.tool({ name, description, parameters, execute }).");
  }
  const value = fromPlain(input instanceof Tool ? input.toJSON() : input);
  if (!(value instanceof Map)) throw new TypeError("A tool is a definition object.");
  const definition = value.get("function") instanceof Map ? (value.get("function") as JsonObject) : value;
  if (definition.has("functionDeclarations") || definition.has("function_declarations")) {
    throw new Error("Pass each Gemini function declaration as its own tool.");
  }
  let parameters: Json = new Map<string, Json>([["type", "object"], ["properties", new Map()]]);
  for (const key of ["parameters", "input_schema", "parametersJsonSchema", "parameters_json_schema"]) {
    const candidate = definition.get(key);
    if (candidate !== undefined && candidate !== null) {
      parameters = candidate;
      break;
    }
  }
  const description = definition.get("description");
  const result = new Map<string, Json>([
    ["name", definition.has("name") ? definition.get("name")! : null],
    ["description", truthy(description) ? description! : ""],
    ["parameters", lowerTypes(clone(parameters))],
  ]);
  return new Map<string, Json>([["type", "function"], ["function", checked(result)]]);
}

/**
 * The definition DM-Call sends for a tool, `{type: "function", function: {name, description, parameters}}`,
 * from an OpenAI, Anthropic or Gemini definition or a {@link tool}.
 */
export function schemaOf(input: ToolInput): FunctionTool {
  return toPlain(schemaOfJson(input)) as FunctionTool;
}

/** Tool definitions for the model, and the handlers that implement them by name. */
export function toolset(tools: unknown, functions?: Record<string, ToolHandler> | null): {
  schemas: JsonObject[];
  registry: Map<string, ToolHandler>;
} {
  let list: unknown[];
  if (tools === undefined || tools === null) list = [];
  else if (Array.isArray(tools)) list = tools;
  else if (typeof tools === "object") list = [tools];
  else throw new TypeError("tools must be a list of tool definitions or dmcall.tool() results.");
  const schemas: JsonObject[] = [];
  const registry = new Map<string, ToolHandler>();
  const names = new Set<string>();
  for (const item of list) {
    const schema = schemaOfJson(item);
    const name = (schema.get("function") as JsonObject).get("name") as string;
    if (names.has(name)) throw new Error(`Two tools are named ${stringRepr(name)}.`);
    names.add(name);
    schemas.push(schema);
    const execute = (item as { execute?: unknown }).execute;
    if (typeof execute === "function") registry.set(name, execute as ToolHandler);
  }
  for (const [name, handler] of Object.entries(functions ?? {})) {
    if (!names.has(name)) throw new Error(`functions has ${stringRepr(name)}, which is not one of the tools.`);
    if (typeof handler !== "function") throw new TypeError(`functions[${stringRepr(name)}] is not a function.`);
    registry.set(name, handler);
  }
  return { schemas, registry };
}

function trimmedEmpty(text: string): boolean {
  // Python's str.strip() whitespace.
  return /^[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/.test(text);
}

/** Calls in {@link Json} form, from any SDK shape. Arguments that are not valid JSON become `null`. */
export function normalizeCallsJson(calls: unknown): Json[] {
  if (calls === null || calls === undefined) return [];
  let value = fromPlain(calls);
  if (value instanceof Map) value = [value];
  if (!Array.isArray(value)) throw new TypeError("calls must be a call or a list of calls.");
  return value.map((entry) => {
    let item = entry;
    if (!(item instanceof Map)) throw new TypeError("Each call must be an object or an SDK tool-call object.");
    for (const key of ["functionCall", "function_call", "function"]) {
      const inner = item.get(key);
      if (inner instanceof Map) {
        item = inner;
        break;
      }
    }
    let args: Json = new Map();
    for (const key of ["arguments", "input", "args"]) {
      if (item.has(key)) {
        args = item.get(key)!;
        break;
      }
    }
    if (args === null) args = new Map();
    else if (typeof args === "string") {
      try {
        args = trimmedEmpty(args) ? new Map() : parse(args);
      } catch {
        args = null;
      }
    }
    return new Map<string, Json>([["name", item.has("name") ? item.get("name")! : null], ["arguments", clone(args)]]);
  });
}

/**
 * Calls as `[{name, arguments}]` from DM-Call, OpenAI, Anthropic or Gemini shapes.
 *
 * `null` or `undefined` means no call (OpenAI SDKs report `tool_calls: null`). Arguments that are
 * not valid JSON become `null`, which the schema check rejects.
 */
export function normalizeCalls(calls: unknown): Call[] {
  return toPlain(normalizeCallsJson(calls)) as Call[];
}
