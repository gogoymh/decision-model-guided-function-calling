package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The local schema check S(c) of the paper.
 *
 * <p>S(c) is true when the candidate is readable, names an available function and supplies every
 * required argument with its declared JSON type. Value constraints (enum, ranges, lengths,
 * patterns, formats) are not checked here: whether a value fits the request and the parameter
 * description is the decision model's question. A value equal to a parameter's declared default is
 * admissible even when its JSON type differs (for example {@code null}), and, as in a function
 * call, keyword names outside a closed parameter list are rejected. An empty collection (no call)
 * passes. Error messages follow Python jsonschema's wording.
 */
public final class SchemaCheck {
    // Schema text keys also detect mutations by callers. Validators own a copy of their schema.
    private static final Map<String, TypeSchemaValidator> VALIDATORS = new LinkedHashMap<>(16, 0.75f, true);

    static synchronized TypeSchemaValidator validator(JsonNode schema) {
        String key = Json.compact(schema);
        TypeSchemaValidator cached = VALIDATORS.get(key);
        if (cached != null) {
            return cached;
        }
        MetaSchema.check(schema);
        TypeSchemaValidator compiled = new TypeSchemaValidator(TypeSchemaValidator.typeSchema(schema));
        VALIDATORS.put(key, compiled);
        if (VALIDATORS.size() > 1024) {
            VALIDATORS.remove(VALIDATORS.keySet().iterator().next());
        }
        return compiled;
    }

    private SchemaCheck() {
    }

    /**
     * Runs S(c) on proposed {@code calls} against the available {@code tools}.
     *
     * @param tools tool definitions: {@link Tool}s, maps or JSON (OpenAI style or flat)
     * @param calls a list of {@link Call}s or JSON calls {@code [{"name", "arguments"}]}; {@code null}
     *              is an unreadable candidate
     * @throws IllegalArgumentException for invalid tool definitions, a configuration error
     */
    public static SchemaReport check(Object tools, Object calls) {
        return checkJson(Json.node(tools), Json.node(calls));
    }

    /** The observable part of each tool: name, description and parameters. */
    public static ArrayNode functionDefinitions(Object tools) {
        return functionDefinitionsJson(Json.node(tools));
    }

    static ArrayNode functionDefinitionsJson(JsonNode tools) {
        if (tools == null || !tools.isArray()) {
            throw new IllegalArgumentException("tools must be a list of function definitions");
        }
        ArrayNode definitions = Json.array();
        Set<String> names = new HashSet<>();
        for (JsonNode tool : tools) {
            if (!tool.isObject()) {
                throw new IllegalArgumentException("Every tool must be a function definition object");
            }
            JsonNode definition = tool.has("function") ? tool.get("function") : tool;
            if (!definition.isObject()) {
                throw new IllegalArgumentException("Every tool.function must be an object");
            }
            String name = Py.text(definition.get("name"));
            if (name == null || name.isEmpty() || !names.add(name)) {
                throw new IllegalArgumentException("Available function names must be nonempty and unique");
            }
            ObjectNode item = Json.object();
            item.put("name", name);
            if (definition.has("description")) {
                item.set("description", definition.get("description").deepCopy());
            } else {
                item.put("description", "");
            }
            if (definition.has("parameters")) {
                item.set("parameters", definition.get("parameters").deepCopy());
            } else {
                ObjectNode parameters = item.putObject("parameters");
                parameters.put("type", "object");
                parameters.putObject("properties");
            }
            definitions.add(item);
        }
        return definitions;
    }

    static SchemaReport checkJson(JsonNode tools, JsonNode calls) {
        ArrayNode definitions = functionDefinitionsJson(tools);
        Map<String, TypeSchemaValidator> validators = new LinkedHashMap<>();
        Map<String, JsonNode> schemas = new LinkedHashMap<>();
        for (JsonNode definition : definitions) {
            JsonNode schema = definition.get("parameters");
            if (!schema.isObject()) {
                throw new IllegalArgumentException("Function parameters must be a JSON Schema object");
            }
            try {
                validators.put(definition.get("name").textValue(), validator(schema));
            } catch (MetaSchema.InvalidSchema e) {
                throw new IllegalArgumentException("An available function has an invalid JSON Schema", e);
            }
            String name = definition.get("name").textValue();
            schemas.put(name, schema);
        }

        List<SchemaError> errors = new ArrayList<>();
        boolean wellFormed = calls != null && calls.isArray();
        if (!wellFormed) {
            errors.add(new SchemaError("function", "calls", "Proposed calls must be a list."));
            errors.add(new SchemaError("arguments", "calls", "No valid call collection to inspect."));
        } else {
            for (int index = 0; index < calls.size(); index++) {
                JsonNode call = calls.get(index);
                String path = "calls[" + index + "]";
                if (!call.isObject()) {
                    wellFormed = false;
                    errors.add(new SchemaError("function", path, "Each call must be an object."));
                    errors.add(new SchemaError("arguments", path, "No argument object to inspect."));
                    continue;
                }
                String name = Py.text(call.get("name"));
                if (name == null || name.isEmpty()) {
                    wellFormed = false;
                    errors.add(new SchemaError("function", path + ".name", "Name must be a nonempty string."));
                } else if (!validators.containsKey(name)) {
                    errors.add(new SchemaError("function", path + ".name", "Selected function is not available."));
                }
                JsonNode arguments = call.get("arguments");
                if (arguments == null || !arguments.isObject()) {
                    wellFormed = false;
                    errors.add(new SchemaError("arguments", path + ".arguments", "Arguments must be a JSON object."));
                    continue;
                }
                if (!Json.isFinite(arguments)) {
                    wellFormed = false;
                    errors.add(new SchemaError("arguments", path + ".arguments",
                            "Arguments must be finite JSON values."));
                    continue;
                }
                if (name == null || !validators.containsKey(name)) {
                    errors.add(new SchemaError("arguments", path + ".arguments",
                            "Cannot validate arguments for an unavailable function."));
                    continue;
                }
                List<TypeSchemaValidator.Issue> issues = new ArrayList<>(validators.get(name).errors(arguments));
                // Python sorts by str(list(error.path)); the sort is stable.
                issues.sort(Comparator.comparing((TypeSchemaValidator.Issue issue) -> Py.reprPath(issue.path()),
                        Py::compare));
                for (TypeSchemaValidator.Issue issue : issues) {
                    if (isDeclaredDefault(issue)) {
                        continue;
                    }
                    StringBuilder suffix = new StringBuilder();
                    for (Object component : issue.path()) {
                        suffix.append('[').append(Json.dumps(TypeSchemaValidator.component(component), true, false))
                                .append(']');
                    }
                    errors.add(new SchemaError("arguments", path + ".arguments" + suffix, issue.message()));
                }
                JsonNode schema = schemas.get(name);
                // An ordinary function signature has closed top-level keyword names.
                // Respect an explicit additionalProperties or patternProperties rule.
                if (schema.has("properties") && !schema.has("additionalProperties")
                        && !schema.has("patternProperties")) {
                    JsonNode properties = schema.get("properties");
                    List<String> unexpected = new ArrayList<>();
                    for (Iterator<String> it = arguments.fieldNames(); it.hasNext(); ) {
                        String key = it.next();
                        if (!properties.has(key)) {
                            unexpected.add(key);
                        }
                    }
                    unexpected.sort(Py::compare);
                    for (String key : unexpected) {
                        errors.add(new SchemaError("arguments", path + ".arguments", "Unknown parameter: " + key + "."));
                    }
                }
            }
        }
        boolean functionPass = errors.stream().noneMatch(error -> error.check().equals("function"));
        boolean argumentsPass = errors.stream().noneMatch(error -> error.check().equals("arguments"));
        return new SchemaReport(functionPass, argumentsPass, functionPass && argumentsPass, wellFormed, errors);
    }

    /** A value equal to the parameter's own declared default is admissible. */
    private static boolean isDeclaredDefault(TypeSchemaValidator.Issue issue) {
        JsonNode schema = issue.schema();
        if (!"type".equals(issue.keyword()) || schema == null || !schema.isObject() || !schema.has("default")) {
            return false;
        }
        return Json.dumps(issue.instance(), true, true).equals(Json.dumps(schema.get("default"), true, true));
    }
}
