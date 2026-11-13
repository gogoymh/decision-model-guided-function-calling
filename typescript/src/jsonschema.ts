/**
 * The part of Python's `jsonschema` that the schema check S(c) uses: a structural check of a
 * schema against its dialect's meta-schema, and type-only validation with the same error messages
 * and order. Schemas and instances are {@link Json} values, so `1.0` and `1` stay distinct.
 */

import {
  type Json, type JsonObject, Float, compareCodePoints, isNumber, pythonRepr, stringRepr, stringify, truthy,
} from "./json.js";

/** How a JSON Schema dialect differs from draft 2020-12 in the keywords used here. */
export interface Dialect {
  readonly name: string;
  /** `items` may be an array (tuple validation) and `prefixItems` is not a keyword. */
  readonly legacyItems: boolean;
  /** `$ref` hides its sibling keywords (drafts 4 to 7). */
  readonly refOverrides: boolean;
  /** `2.0` is not an integer, schemas cannot be booleans and `exclusiveMaximum` is a flag (draft 4). */
  readonly draft4: boolean;
  /** The keywords its meta-schema constrains; others are free annotations. */
  readonly keywords: ReadonlySet<string>;
  /** `enum` must be a nonempty array of unique values (draft 4). */
  readonly strictEnum: boolean;
  /** `$id` may not carry a fragment (2019-09 and 2020-12). */
  readonly plainId: boolean;
}

const DRAFT4_KEYWORDS = ["type", "properties", "patternProperties", "required", "items", "additionalProperties",
  "additionalItems", "anyOf", "allOf", "oneOf", "not", "enum", "multipleOf", "maximum", "minimum", "exclusiveMaximum",
  "exclusiveMinimum", "maxLength", "minLength", "maxItems", "minItems", "maxProperties", "minProperties", "pattern",
  "format", "title", "description", "uniqueItems", "definitions", "dependencies", "$schema"];
const DRAFT6_KEYWORDS = [...DRAFT4_KEYWORDS, "$id", "$ref", "const", "contains", "propertyNames", "examples"];
const DRAFT7_KEYWORDS = [...DRAFT6_KEYWORDS, "if", "then", "else", "$comment", "readOnly", "writeOnly",
  "contentMediaType", "contentEncoding"];
const DRAFT2019_KEYWORDS = [...DRAFT7_KEYWORDS, "$defs", "$anchor", "$vocabulary", "dependentRequired",
  "dependentSchemas", "unevaluatedItems", "unevaluatedProperties", "maxContains", "minContains", "contentSchema",
  "deprecated"];
const DRAFT2020_KEYWORDS = [...DRAFT2019_KEYWORDS.filter((keyword) => keyword !== "additionalItems"), "prefixItems",
  "$dynamicRef", "$dynamicAnchor"];

const DRAFT2020: Dialect = { name: "2020-12", legacyItems: false, refOverrides: false, draft4: false,
  keywords: new Set(DRAFT2020_KEYWORDS), strictEnum: false, plainId: true };
const DRAFT2019: Dialect = { name: "2019-09", legacyItems: true, refOverrides: false, draft4: false,
  keywords: new Set(DRAFT2019_KEYWORDS), strictEnum: false, plainId: true };
const DRAFT7: Dialect = { name: "draft-07", legacyItems: true, refOverrides: true, draft4: false,
  keywords: new Set(DRAFT7_KEYWORDS), strictEnum: false, plainId: false };
const DRAFT6: Dialect = { name: "draft-06", legacyItems: true, refOverrides: true, draft4: false,
  keywords: new Set(DRAFT6_KEYWORDS), strictEnum: false, plainId: false };
const DRAFT4: Dialect = { name: "draft-04", legacyItems: true, refOverrides: true, draft4: true,
  keywords: new Set(DRAFT4_KEYWORDS), strictEnum: true, plainId: false };
const DIALECTS: Readonly<Record<string, Dialect>> = {
  "http://json-schema.org/draft-03/schema": DRAFT4,
  "http://json-schema.org/draft-04/schema": DRAFT4,
  "http://json-schema.org/draft-06/schema": DRAFT6,
  "http://json-schema.org/draft-07/schema": DRAFT7,
  "https://json-schema.org/draft/2019-09/schema": DRAFT2019,
  "https://json-schema.org/draft/2020-12/schema": DRAFT2020,
};

/** `validator_for(schema, default=Draft202012Validator)`: the dialect named by `$schema`, else 2020-12. */
export function dialectOf(schema: Json): Dialect {
  if (!(schema instanceof Map)) return DRAFT2020;
  const uri = schema.get("$schema");
  if (typeof uri !== "string") return DRAFT2020;
  const scheme = /^[A-Za-z][A-Za-z0-9+.-]*:/.exec(uri);
  let normalized = scheme ? scheme[0].toLowerCase() + uri.slice(scheme[0].length) : uri;
  if (normalized.endsWith("#")) normalized = normalized.slice(0, -1);
  return DIALECTS[normalized] ?? DRAFT2020;
}

// Meta-schema check ------------------------------------------------------------------------------

export class SchemaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SchemaError";
  }
}

const SIMPLE_TYPES = new Set(["array", "boolean", "integer", "null", "number", "object", "string"]);
const SCHEMA_KEYWORDS = new Set([
  "additionalProperties", "additionalItems", "not", "if", "then", "else", "contains", "propertyNames",
  "unevaluatedItems", "unevaluatedProperties", "contentSchema",
]);
const SCHEMA_MAPS = new Set(["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]);
const SCHEMA_LISTS = new Set(["anyOf", "allOf", "oneOf"]);
const COUNTS = new Set([
  "maxLength", "minLength", "maxItems", "minItems", "maxProperties", "minProperties", "maxContains", "minContains",
]);
const TEXT = new Set([
  "pattern", "format", "title", "description", "$comment", "$anchor", "$dynamicAnchor", "$dynamicRef",
  "contentEncoding", "contentMediaType", "$ref", "$id", "$schema",
]);
const FLAGS = new Set(["uniqueItems", "deprecated", "readOnly", "writeOnly"]);

function isInteger(value: Json, dialect: Dialect): boolean {
  if (typeof value === "bigint") return true;
  return !dialect.draft4 && value instanceof Float && Number.isInteger(value.value);
}

function unique(values: Json[]): boolean {
  const seen = new Set<string>();
  for (const value of values) {
    const key = typeof value === "string" ? `s${value}` : `j${stringify(value, { sortKeys: true })}`;
    if (seen.has(key)) return false;
    seen.add(key);
  }
  return true;
}

function invalid(keyword: string): never {
  throw new SchemaError(`The schema keyword ${stringRepr(keyword)} has an invalid value.`);
}

function checkNode(schema: Json, dialect: Dialect): void {
  if (typeof schema === "boolean") {
    if (dialect.draft4) throw new SchemaError("Draft 4 schemas must be objects.");
    return;
  }
  if (!(schema instanceof Map)) throw new SchemaError("A schema must be an object or a boolean.");
  for (const [keyword, value] of schema) {
    if (!dialect.keywords.has(keyword)) continue;
    if (keyword === "type") {
      if (typeof value === "string") {
        if (!SIMPLE_TYPES.has(value)) invalid(keyword);
      } else if (Array.isArray(value)) {
        if (value.length === 0 || !unique(value)
          || !value.every((item) => typeof item === "string" && SIMPLE_TYPES.has(item))) invalid(keyword);
      } else invalid(keyword);
    } else if (SCHEMA_MAPS.has(keyword)) {
      if (!(value instanceof Map)) invalid(keyword);
      if (keyword === "patternProperties" && !dialect.draft4 && [...value.keys()].some((key) => compile(key) === null)) {
        invalid(keyword);
      }
      for (const item of value.values()) checkNode(item, dialect);
    } else if (keyword === "items") {
      if (Array.isArray(value)) {
        if (!dialect.legacyItems || value.length === 0) invalid(keyword);
        value.forEach((item) => checkNode(item, dialect));
      } else checkNode(value, dialect);
    } else if (keyword === "prefixItems") {
      if (dialect.legacyItems) continue; // not a keyword before 2020-12
      if (!Array.isArray(value) || value.length === 0) invalid(keyword);
      value.forEach((item) => checkNode(item, dialect));
    } else if (SCHEMA_KEYWORDS.has(keyword)) {
      if (dialect.draft4 && typeof value === "boolean" && (keyword === "additionalProperties"
        || keyword === "additionalItems")) continue;
      checkNode(value, dialect);
    } else if (SCHEMA_LISTS.has(keyword)) {
      if (!Array.isArray(value) || value.length === 0) invalid(keyword);
      value.forEach((item) => checkNode(item, dialect));
    } else if (keyword === "required") {
      if (!Array.isArray(value) || !value.every((item) => typeof item === "string") || !unique(value)
        || (dialect.draft4 && value.length === 0)) invalid(keyword);
    } else if (keyword === "enum") {
      if (!Array.isArray(value) || (dialect.strictEnum && (value.length === 0 || !unique(value)))) invalid(keyword);
    } else if (keyword === "multipleOf") {
      if (!isNumber(value) || !(Number(value instanceof Float ? value.value : value) > 0)) invalid(keyword);
    } else if (keyword === "maximum" || keyword === "minimum") {
      if (!isNumber(value)) invalid(keyword);
    } else if (keyword === "exclusiveMaximum" || keyword === "exclusiveMinimum") {
      if (dialect.draft4 ? typeof value !== "boolean" || !schema.has(keyword === "exclusiveMaximum" ? "maximum" : "minimum")
        : !isNumber(value)) invalid(keyword);
    } else if (COUNTS.has(keyword)) {
      if (!isInteger(value, dialect) || Number(value instanceof Float ? value.value : value) < 0) invalid(keyword);
    } else if (TEXT.has(keyword)) {
      if (typeof value !== "string") invalid(keyword);
      if (keyword === "pattern" && compile(value) === null) invalid(keyword);
      if (keyword === "$id" && dialect.plainId && !/^[^#]*#?$/.test(value)) invalid(keyword);
      if ((keyword === "$anchor" || keyword === "$dynamicAnchor") && !/^[A-Za-z_][-A-Za-z0-9._]*$/.test(value)) {
        invalid(keyword);
      }
    } else if (FLAGS.has(keyword)) {
      if (typeof value !== "boolean") invalid(keyword);
    } else if (keyword === "examples") {
      if (!Array.isArray(value)) invalid(keyword);
    } else if (keyword === "dependentRequired") {
      if (!(value instanceof Map)) invalid(keyword);
      for (const item of value.values()) {
        if (!Array.isArray(item) || !item.every((name) => typeof name === "string") || !unique(item)) invalid(keyword);
      }
    } else if (keyword === "dependencies") {
      if (!(value instanceof Map)) invalid(keyword);
      for (const item of value.values()) {
        if (Array.isArray(item)) {
          if (!item.every((name) => typeof name === "string")) invalid(keyword);
        } else checkNode(item, dialect);
      }
    } else if (keyword === "$vocabulary") {
      if (!(value instanceof Map) || ![...value.values()].every((item) => typeof item === "boolean")) invalid(keyword);
    }
  }
}

/** `Validator.check_schema(schema)`: raises {@link SchemaError} for a schema its meta-schema rejects. */
export function checkMetaSchema(schema: Json, dialect: Dialect = dialectOf(schema)): void {
  checkNode(schema, dialect);
}

// Type-only schema -------------------------------------------------------------------------------

/** Only the structural typing part of a schema is checked. */
const TYPE_KEYWORDS = new Set([
  "$schema", "$id", "$ref", "$defs", "definitions", "type", "properties", "required",
  "items", "prefixItems", "additionalProperties", "patternProperties", "anyOf", "allOf",
  "oneOf", "default",
]);
const NESTED_MAPS = new Set(["properties", "patternProperties", "$defs", "definitions"]);

function copy(value: Json): Json {
  if (Array.isArray(value)) return value.map(copy);
  if (value instanceof Map) return new Map([...value].map(([key, item]) => [key, copy(item)]));
  return value;
}

/** A schema's types, required names and nesting without value constraints. */
export function typeSchema(schema: Json): Json {
  if (Array.isArray(schema)) return schema.map(typeSchema);
  if (!(schema instanceof Map)) return copy(schema);
  const result: JsonObject = new Map();
  for (const [key, value] of schema) {
    if (!TYPE_KEYWORDS.has(key)) continue;
    if (NESTED_MAPS.has(key) && value instanceof Map) {
      result.set(key, new Map([...value].map(([name, item]) => [name, typeSchema(item)])));
    } else if (key === "items" || key === "prefixItems" || key === "additionalProperties") {
      result.set(key, typeSchema(value));
    } else if ((key === "anyOf" || key === "allOf" || key === "oneOf") && Array.isArray(value)) {
      const branches = value.map(typeSchema);
      if (!result.has("allOf")) result.set("allOf", []);
      const allOf = result.get("allOf") as Json[];
      if (key === "allOf") allOf.push(...branches);
      // Without value constraints, oneOf branches may overlap; any branch suffices.
      else allOf.push(new Map([["anyOf", branches]]));
    } else {
      result.set(key, copy(value));
    }
  }
  return result;
}

// Validation -------------------------------------------------------------------------------------

/** One validation error, as Python's `jsonschema.ValidationError` reports it. */
export interface ValidationIssue {
  /** Path from the instance root: property names and array indices. */
  readonly path: readonly (string | number)[];
  readonly message: string;
  /** The keyword that failed (`null` for a `false` schema). */
  readonly validator: string | null;
  /** The (sub)schema holding that keyword. */
  readonly schema: Json;
  readonly instance: Json;
}

function isType(instance: Json, type: string, dialect: Dialect): boolean {
  switch (type) {
    case "array":
      return Array.isArray(instance);
    case "boolean":
      return typeof instance === "boolean";
    case "integer":
      return typeof instance === "bigint"
        || (!dialect.draft4 && instance instanceof Float && Number.isInteger(instance.value));
    case "null":
      return instance === null;
    case "number":
      return typeof instance === "bigint" || instance instanceof Float;
    case "object":
      return instance instanceof Map;
    case "string":
      return typeof instance === "string";
    default:
      throw new SchemaError(`Unknown type ${stringRepr(type)}.`);
  }
}

const REGEXES = new Map<string, RegExp | null>();

/**
 * A Python `re` pattern as a JavaScript RegExp (`null` if it does not compile). Python's named groups
 * `(?P<name>...)` and `(?P=name)` and leading `i`, `m` and `s` flags such as `(?i)` are translated;
 * other syntax is read as JavaScript's.
 */
function compile(pattern: string): RegExp | null {
  if (REGEXES.has(pattern)) return REGEXES.get(pattern)!;
  let source = pattern.replace(/\(\?P<([A-Za-z_][A-Za-z0-9_]*)>/g, "(?<$1>").replace(/\(\?P=([A-Za-z_][A-Za-z0-9_]*)\)/g, "\\k<$1>");
  let inline = "";
  const leading = /^\(\?([ims]+)\)/.exec(source);
  if (leading) {
    inline = [...new Set(leading[1]!)].join("");
    source = source.slice(leading[0].length);
  }
  let compiled: RegExp | null = null;
  for (const flags of ["u", ""]) {
    try {
      compiled = new RegExp(source, flags + inline);
      break;
    } catch {
      // try the next mode
    }
  }
  REGEXES.set(pattern, compiled);
  return compiled;
}

function search(pattern: string, text: string): boolean {
  return compile(pattern)?.test(text) ?? false;
}

function resolvePointer(root: Json, reference: string): Json {
  if (!reference.startsWith("#")) {
    throw new SchemaError(`Unresolvable JSON Schema reference ${stringRepr(reference)}: only local references are supported.`);
  }
  let pointer: string;
  try {
    pointer = decodeURIComponent(reference.slice(1));
  } catch {
    throw new SchemaError(`Unresolvable JSON Schema reference ${stringRepr(reference)}.`);
  }
  if (pointer === "") return root;
  if (!pointer.startsWith("/")) {
    throw new SchemaError(`Unresolvable JSON Schema reference ${stringRepr(reference)}: anchors are not supported.`);
  }
  let node: Json | undefined = root;
  for (const raw of pointer.slice(1).split("/")) {
    const token = raw.replace(/~1/g, "/").replace(/~0/g, "~");
    if (node instanceof Map) node = node.get(token);
    else if (Array.isArray(node) && /^(0|[1-9][0-9]*)$/.test(token)) node = node[Number(token)];
    else node = undefined;
    if (node === undefined) throw new SchemaError(`Unresolvable JSON Schema reference ${stringRepr(reference)}.`);
  }
  return node;
}

/** A reference that cannot be followed inside the tool parameters, or recurses without consuming input. */
export class ReferenceError extends SchemaError {}

function checkReferences(root: Json, dialect: Dialect): void {
  const state = new Map<JsonObject, boolean>();
  const schemaNodes = new Set<JsonObject>();
  const fail = (): never => { throw new ReferenceError("An available function's JSON Schema has a $ref that cannot be resolved"); };
  const follow = (node: Json): void => {
    if (!(node instanceof Map)) return;
    if (state.get(node) === false) return;
    if (state.get(node)) fail();
    state.set(node, true);
    const reference = node.get("$ref");
    if (typeof reference === "string") {
      const target = (() => {
        try { return resolvePointer(root, reference); } catch { return fail(); }
      })();
      if (typeof target !== "boolean" && (!(target instanceof Map) || !schemaNodes.has(target))) fail();
      if (typeof target === "boolean") {
        try { checkMetaSchema(target, dialect); } catch { fail(); }
      }
      follow(target);
      if (dialect.refOverrides) { state.set(node, false); return; }
    }
    for (const key of ["allOf", "anyOf"]) {
      const branches = node.get(key);
      if (Array.isArray(branches)) branches.forEach(follow);
    }
    state.set(node, false);
  };
  const visit = (node: Json): void => {
    if (Array.isArray(node)) { node.forEach(visit); return; }
    if (!(node instanceof Map)) return;
    schemaNodes.add(node);
    for (const [key, value] of node) {
      if (NESTED_MAPS.has(key) && value instanceof Map) [...value.values()].forEach(visit);
      else if (["items", "prefixItems", "additionalProperties", "allOf", "anyOf"].includes(key)) visit(value);
    }
  };
  try {
    visit(root);
    for (const node of schemaNodes) follow(node);
  } catch (error) {
    if (error instanceof RangeError) fail();
    throw error;
  }
}

const COMPILED = new Map<string, { schema: Json; dialect: Dialect }>();

/** Validate and prepare a type-only schema once per schema text, with a bounded cache. */
export function compiledSchema(input: Json): { schema: Json; dialect: Dialect } {
  const key = stringify(input);
  const cached = COMPILED.get(key);
  if (cached) { COMPILED.delete(key); COMPILED.set(key, cached); return cached; }
  const dialect = dialectOf(input);
  checkMetaSchema(input, dialect);
  const schema = typeSchema(input);
  checkReferences(schema, dialect);
  const result = { schema, dialect };
  if (COMPILED.size >= 1024) COMPILED.delete(COMPILED.keys().next().value!);
  COMPILED.set(key, result);
  return result;
}

type Segment = string | number;

/**
 * Validation as jsonschema does it: `descend(instance, schema, segment)` prefixes the errors of the
 * schema's keywords with `segment`. A `false` subschema yields its error before that prefix is added,
 * so, as in Python, its own segment is not part of the path.
 */
class Validator {
  constructor(private readonly root: Json, private readonly dialect: Dialect) {}

  errors(instance: Json): ValidationIssue[] {
    return this.descend(instance, this.root);
  }

  private issue(message: string, validator: string | null, schema: Json, instance: Json): ValidationIssue {
    return { path: [], message, validator, schema, instance };
  }

  descend(instance: Json, schema: Json, segment?: Segment): ValidationIssue[] {
    if (schema === true) return [];
    if (schema === false) return [this.issue(`False schema does not allow ${pythonRepr(instance)}`, null, schema, instance)];
    if (!(schema instanceof Map)) return [];
    let keywords: [string, Json][] = [...schema];
    if (this.dialect.refOverrides && schema.has("$ref") && schema.get("$ref") !== null) {
      keywords = [["$ref", schema.get("$ref")!]];
    }
    const out: ValidationIssue[] = [];
    for (const [keyword, value] of keywords) {
      for (const found of this.keyword(keyword, value, instance, schema)) {
        out.push(segment === undefined ? found : { ...found, path: [segment, ...found.path] });
      }
    }
    return out;
  }

  private keyword(keyword: string, value: Json, instance: Json, schema: JsonObject): ValidationIssue[] {
    const dialect = this.dialect;
    switch (keyword) {
      case "$ref":
        if (typeof value !== "string") return [];
        return this.descend(instance, resolvePointer(this.root, value));
      case "type": {
        const types = typeof value === "string" ? [value] : Array.isArray(value) ? value : [];
        if (types.some((type) => typeof type === "string" && isType(instance, type, dialect))) return [];
        const names = types.map((type) => pythonRepr(type)).join(", ");
        return [this.issue(`${pythonRepr(instance)} is not of type ${names}`, keyword, schema, instance)];
      }
      case "properties": {
        if (!(instance instanceof Map) || !(value instanceof Map)) return [];
        const out: ValidationIssue[] = [];
        for (const [name, subschema] of value) {
          if (instance.has(name)) out.push(...this.descend(instance.get(name)!, subschema, name));
        }
        return out;
      }
      case "patternProperties": {
        if (!(instance instanceof Map) || !(value instanceof Map)) return [];
        const out: ValidationIssue[] = [];
        for (const [pattern, subschema] of value) {
          for (const [name, item] of instance) {
            if (search(pattern, name)) out.push(...this.descend(item, subschema, name));
          }
        }
        return out;
      }
      case "additionalProperties": {
        if (!(instance instanceof Map)) return [];
        const properties = schema.get("properties");
        const patterns = schema.get("patternProperties");
        const joined = patterns instanceof Map ? [...patterns.keys()].join("|") : "";
        const extras = [...instance.keys()].filter((name) => !(properties instanceof Map && properties.has(name))
          && !(joined && search(joined, name)));
        if (value instanceof Map) {
          const out: ValidationIssue[] = [];
          for (const extra of extras) out.push(...this.descend(instance.get(extra)!, value, extra));
          return out;
        }
        if (truthy(value) || extras.length === 0) return [];
        const sorted = [...extras].sort(compareCodePoints);
        if (schema.has("patternProperties")) {
          const verb = extras.length === 1 ? "does" : "do";
          const names = sorted.map(stringRepr).join(", ");
          const regexes = (patterns instanceof Map ? [...patterns.keys()] : []).sort(compareCodePoints)
            .map(stringRepr).join(", ");
          return [this.issue(`${names} ${verb} not match any of the regexes: ${regexes}`, keyword, schema, instance)];
        }
        const verb = extras.length === 1 ? "was" : "were";
        return [this.issue(`Additional properties are not allowed (${sorted.map(stringRepr).join(", ")} ${verb} `
          + "unexpected)", keyword, schema, instance)];
      }
      case "required": {
        if (!(instance instanceof Map) || !Array.isArray(value)) return [];
        return value.filter((name) => !(typeof name === "string" && instance.has(name)))
          .map((name) => this.issue(`${pythonRepr(name)} is a required property`, keyword, schema, instance));
      }
      case "items":
        return dialect.legacyItems ? this.legacyItems(value, instance) : this.items(value, instance, schema);
      case "prefixItems": {
        if (dialect.legacyItems || !Array.isArray(instance) || !Array.isArray(value)) return [];
        const out: ValidationIssue[] = [];
        const count = Math.min(instance.length, value.length);
        for (let index = 0; index < count; index++) out.push(...this.descend(instance[index]!, value[index]!, index));
        return out;
      }
      case "allOf": {
        if (!Array.isArray(value)) return [];
        return value.flatMap((subschema) => this.descend(instance, subschema));
      }
      case "anyOf": {
        if (!Array.isArray(value)) return [];
        for (const subschema of value) {
          if (this.descend(instance, subschema).length === 0) return [];
        }
        return [this.issue(`${pythonRepr(instance)} is not valid under any of the given schemas`, keyword, schema, instance)];
      }
      case "oneOf": {
        // typeSchema() turns oneOf into anyOf; this branch only runs on schemas used directly.
        if (!Array.isArray(value)) return [];
        const passing = value.filter((subschema) => this.descend(instance, subschema).length === 0).length;
        if (passing === 1) return [];
        const message = passing === 0 ? `${pythonRepr(instance)} is not valid under any of the given schemas`
          : `${pythonRepr(instance)} is valid under each of ${value.map(pythonRepr).join(", ")}`;
        return [this.issue(message, keyword, schema, instance)];
      }
      default:
        return [];
    }
  }

  private items(value: Json, instance: Json, schema: JsonObject): ValidationIssue[] {
    if (!Array.isArray(instance)) return [];
    const prefixItems = schema.get("prefixItems");
    const prefix = Array.isArray(prefixItems) ? prefixItems.length : 0;
    const extra = instance.length - prefix;
    if (extra <= 0) return [];
    if (value === false) {
      const rest: Json = extra !== 1 ? instance.slice(prefix) : instance[prefix]!;
      const noun = prefix !== 1 ? "items" : "item";
      return [this.issue(`Expected at most ${prefix} ${noun} but found ${extra} extra: ${pythonRepr(rest)}`, "items",
        schema, instance)];
    }
    const out: ValidationIssue[] = [];
    for (let index = prefix; index < instance.length; index++) out.push(...this.descend(instance[index]!, value, index));
    return out;
  }

  private legacyItems(value: Json, instance: Json): ValidationIssue[] {
    if (!Array.isArray(instance)) return [];
    const out: ValidationIssue[] = [];
    if (Array.isArray(value)) {
      const count = Math.min(instance.length, value.length);
      for (let index = 0; index < count; index++) out.push(...this.descend(instance[index]!, value[index]!, index));
    } else {
      instance.forEach((item, index) => out.push(...this.descend(item, value, index)));
    }
    return out;
  }
}

/** `str(list(error.path))`: the key Python's schema check sorts errors by. */
export function pathKey(path: readonly (string | number)[]): string {
  return `[${path.map((part) => (typeof part === "number" ? String(part) : stringRepr(part))).join(", ")}]`;
}

/** Every validation error of `instance` under `schema`, in the order Python's `iter_errors` yields them. */
export function validationErrors(schema: Json, instance: Json, dialect: Dialect = dialectOf(schema)): ValidationIssue[] {
  return new Validator(schema, dialect).errors(instance);
}
