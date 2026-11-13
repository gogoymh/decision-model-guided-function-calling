package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Tool definitions in every format, and the handlers that implement them by name. */
final class Tools {
    private Tools() {
    }

    /** The tool definitions for the model, and the handlers by name. */
    record Toolset(ArrayNode schemas, Map<String, ToolHandler> registry) {
    }

    /** An OpenAI-style tool definition {@code {"type": "function", "function": {...}}} for a definition. */
    static ObjectNode schemaOf(JsonNode tool) {
        if (tool == null || !tool.isObject()) {
            throw new IllegalArgumentException("A tool is a function definition object.");
        }
        JsonNode function = tool.get("function");
        JsonNode definition = function != null && function.isObject() ? function : tool;
        if (definition.has("functionDeclarations") || definition.has("function_declarations")) {
            throw new IllegalArgumentException("Pass each Gemini function declaration as its own tool.");
        }
        JsonNode parameters = null;
        for (String key : new String[]{"parameters", "input_schema", "parametersJsonSchema", "parameters_json_schema"}) {
            JsonNode value = definition.get(key);
            if (!Py.isNone(value)) {
                parameters = value;
                break;
            }
        }
        if (parameters == null) {
            ObjectNode empty = Json.object();
            empty.put("type", "object");
            empty.putObject("properties");
            parameters = empty;
        }
        ObjectNode result = Json.object();
        result.set("name", Py.isNone(definition.get("name")) ? NullNode.getInstance()
                : definition.get("name").deepCopy());
        JsonNode description = definition.get("description");
        result.set("description", Py.truthy(description) ? description.deepCopy() : TextNode.valueOf(""));
        result.set("parameters", lowerTypes(parameters));
        ObjectNode schema = Json.object();
        schema.put("type", "function");
        schema.set("function", checked(result));
        return schema;
    }

    private static ObjectNode checked(ObjectNode definition) {
        String name = Py.text(definition.get("name"));
        if (name == null || Py.strip(name).isEmpty()) {
            throw new IllegalArgumentException("Every tool needs a nonempty name.");
        }
        if (!definition.get("description").isTextual()) {
            throw new IllegalArgumentException("Tool " + Py.strRepr(name) + ": the description must be text.");
        }
        JsonNode parameters = definition.get("parameters");
        JsonNode type = parameters.isObject() ? parameters.get("type") : null;
        if (!parameters.isObject() || (type != null && !"object".equals(Py.text(type)))) {
            throw new IllegalArgumentException("Tool " + Py.strRepr(name)
                    + ": parameters must be a JSON Schema object with type 'object'.");
        }
        String problem = null;
        if (!Json.isFinite(parameters)) {
            problem = "ValueError";
        } else {
            try {
                SchemaCheck.validator(parameters);
            } catch (MetaSchema.InvalidSchema e) {
                problem = "SchemaError";
            }
        }
        if (problem != null) {
            throw new IllegalArgumentException("Tool " + Py.strRepr(name) + ": parameters are not a valid JSON Schema ("
                    + problem + ").");
        }
        return definition;
    }

    private static final Set<String> NAMED_SCHEMAS = Set.of("properties", "patternProperties", "$defs",
            "definitions", "dependentSchemas");
    private static final Set<String> SCHEMA_CHILDREN = Set.of("items", "prefixItems", "additionalProperties",
            "additionalItems", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "contains",
            "propertyNames", "unevaluatedProperties", "unevaluatedItems", "contentSchema");

    /** Normalize schema types while preserving data values and names of properties and definitions. */
    static JsonNode lowerTypes(JsonNode schema) {
        if (schema.isArray()) {
            ArrayNode result = Json.array();
            for (JsonNode item : schema) {
                result.add(lowerTypes(item));
            }
            return result;
        }
        if (!schema.isObject()) {
            return schema.deepCopy();
        }
        ObjectNode result = Json.object();
        for (Iterator<Map.Entry<String, JsonNode>> it = schema.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            if (key.equals("type") && value.isTextual() && Py.isUpper(value.textValue())) {
                result.put(key, value.textValue().toLowerCase(Locale.ROOT));
            } else if (key.equals("type") && value.isArray()) {
                ArrayNode types = Json.array();
                for (JsonNode item : value) {
                    types.add(item.isTextual() && Py.isUpper(item.textValue())
                            ? TextNode.valueOf(item.textValue().toLowerCase(Locale.ROOT)) : item.deepCopy());
                }
                result.set(key, types);
            } else if (NAMED_SCHEMAS.contains(key) && value.isObject()) {
                ObjectNode mapped = Json.object();
                value.fields().forEachRemaining(item -> mapped.set(item.getKey(), lowerTypes(item.getValue())));
                result.set(key, mapped);
            } else if (key.equals("dependencies") && value.isObject()) {
                ObjectNode mapped = Json.object();
                value.fields().forEachRemaining(item -> mapped.set(item.getKey(), item.getValue().isArray()
                        ? item.getValue().deepCopy() : lowerTypes(item.getValue())));
                result.set(key, mapped);
            } else if (SCHEMA_CHILDREN.contains(key)) {
                result.set(key, lowerTypes(value));
            } else {
                result.set(key, value.deepCopy());
            }
        }
        return result;
    }

    /** Items of a tools argument: null, one tool or definition, or a collection, array or JSON array. */
    static List<Object> items(Object tools) {
        List<Object> items = new ArrayList<>();
        if (tools == null) {
            return items;
        }
        if (tools instanceof Tool || tools instanceof Map || (tools instanceof JsonNode && ((JsonNode) tools).isObject())) {
            items.add(tools);
        } else if (tools instanceof JsonNode && ((JsonNode) tools).isArray()) {
            ((JsonNode) tools).forEach(items::add);
        } else if (tools instanceof Collection) {
            items.addAll((Collection<?>) tools);
        } else if (tools instanceof Object[]) {
            items.addAll(List.of((Object[]) tools));
        } else {
            throw new IllegalArgumentException("tools must be a list of tools or function definitions.");
        }
        return items;
    }

    static List<Tool> asTools(Object tools) {
        List<Tool> result = new ArrayList<>();
        for (Object item : items(tools)) {
            result.add(item instanceof Tool ? (Tool) item : Tool.of(item));
        }
        return result;
    }

    /** Tool definitions for the model and their handlers; {@code functions} adds or replaces handlers. */
    static Toolset toolset(Object tools, Map<String, ToolHandler> functions) {
        ArrayNode schemas = Json.array();
        Map<String, ToolHandler> registry = new LinkedHashMap<>();
        Set<String> names = new HashSet<>();
        for (Tool tool : asTools(tools)) {
            String name = tool.name();
            if (!names.add(name)) {
                throw new IllegalArgumentException("Two tools are named " + Py.strRepr(name) + ".");
            }
            schemas.add(tool.definition());
            if (tool.handler() != null) {
                registry.put(name, tool.handler());
            }
        }
        if (functions != null) {
            for (Map.Entry<String, ToolHandler> entry : functions.entrySet()) {
                if (!names.contains(entry.getKey())) {
                    throw new IllegalArgumentException("functions has " + Py.strRepr(String.valueOf(entry.getKey()))
                            + ", which is not one of the tools.");
                }
                if (entry.getValue() == null) {
                    throw new IllegalArgumentException("functions[" + Py.strRepr(entry.getKey()) + "] is null.");
                }
                registry.put(entry.getKey(), entry.getValue());
            }
        }
        return new Toolset(schemas, registry);
    }
}
