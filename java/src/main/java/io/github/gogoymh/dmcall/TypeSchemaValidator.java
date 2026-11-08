package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.IntNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * The structural part of a JSON Schema, validated the way Python jsonschema's Draft 2020-12
 * validator does: the same keyword order, error paths and messages (which use Python {@code repr}).
 */
final class TypeSchemaValidator {
    /** Only the structural typing part of a schema is checked. */
    static final Set<String> TYPE_KEYWORDS = Set.of("$schema", "$id", "$ref", "$defs", "definitions", "type",
            "properties", "required", "items", "prefixItems", "additionalProperties", "patternProperties", "anyOf",
            "allOf", "oneOf", "default");

    /** One validation error: where (path within the arguments), what, and which schema produced it. */
    record Issue(List<Object> path, String message, String keyword, JsonNode schema, JsonNode instance) {
    }

    private final JsonNode root;

    TypeSchemaValidator(JsonNode typeSchema) {
        this.root = typeSchema;
    }

    /** A schema's types, required names and nesting without value constraints. */
    static JsonNode typeSchema(JsonNode schema) {
        if (schema.isArray()) {
            ArrayNode result = Json.array();
            for (JsonNode item : schema) {
                result.add(typeSchema(item));
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
            if (!TYPE_KEYWORDS.contains(key)) {
                continue;
            }
            boolean mapping = key.equals("properties") || key.equals("patternProperties") || key.equals("$defs")
                    || key.equals("definitions");
            if (mapping && value.isObject()) {
                ObjectNode mapped = Json.object();
                for (Iterator<Map.Entry<String, JsonNode>> m = value.fields(); m.hasNext(); ) {
                    Map.Entry<String, JsonNode> item = m.next();
                    mapped.set(item.getKey(), typeSchema(item.getValue()));
                }
                result.set(key, mapped);
            } else if (key.equals("items") || key.equals("prefixItems") || key.equals("additionalProperties")) {
                result.set(key, typeSchema(value));
            } else if ((key.equals("anyOf") || key.equals("allOf") || key.equals("oneOf")) && value.isArray()) {
                ArrayNode branches = (ArrayNode) typeSchema(value);
                JsonNode existing = result.get("allOf");
                ArrayNode allOf = existing instanceof ArrayNode ? (ArrayNode) existing : result.putArray("allOf");
                if (key.equals("allOf")) {
                    allOf.addAll(branches);
                } else {
                    // Without value constraints, oneOf branches may overlap; any branch suffices.
                    ObjectNode anyOf = Json.object();
                    anyOf.set("anyOf", branches);
                    allOf.add(anyOf);
                }
            } else {
                result.set(key, value.deepCopy());
            }
        }
        return result;
    }

    /** Python jsonschema's Draft 2020-12 type checker. */
    static boolean isType(JsonNode value, String type) {
        switch (type) {
            case "array":
                return value.isArray();
            case "boolean":
                return value.isBoolean();
            case "integer":
                if (value.isIntegralNumber()) {
                    return true;
                }
                if (value.isFloatingPointNumber()) {
                    double d = value.doubleValue();
                    return !Double.isInfinite(d) && !Double.isNaN(d) && d == Math.rint(d);
                }
                return false;
            case "null":
                return value.isNull();
            case "number":
                return value.isNumber();
            case "object":
                return value.isObject();
            case "string":
                return value.isTextual();
            default:
                throw new IllegalArgumentException("Unknown JSON Schema type " + Py.strRepr(type) + ".");
        }
    }

    /** Every error for {@code instance}, in the order Python jsonschema yields them. */
    List<Issue> errors(JsonNode instance) {
        List<Issue> out = new ArrayList<>();
        descend(instance, root, new ArrayList<>(), null, out, 0);
        return out;
    }

    /**
     * Python jsonschema's {@code descend}: validates {@code instance} (at {@code parent} plus
     * {@code component}, if any) against {@code schema}. A {@code false} subschema reports its error
     * before the component is added, so the error sits at the parent's path, as in jsonschema.
     */
    private void descend(JsonNode instance, JsonNode schema, List<Object> parent, Object component, List<Issue> out,
                         int depth) {
        if (depth > 256) {
            throw new IllegalArgumentException("A JSON Schema reference does not terminate.");
        }
        if (schema.isBoolean()) {
            if (!schema.booleanValue()) {
                out.add(new Issue(List.copyOf(parent), "False schema does not allow " + Py.repr(instance), null, schema,
                        instance));
            }
            return;
        }
        if (!schema.isObject()) {
            return;
        }
        List<Object> path = component == null ? parent : with(parent, component);
        for (Iterator<Map.Entry<String, JsonNode>> it = schema.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            switch (key) {
                case "$ref":
                    descend(instance, resolve(value.asText()), path, null, out, depth + 1);
                    break;
                case "type":
                    typeKeyword(instance, value, schema, path, out);
                    break;
                case "properties":
                    if (instance.isObject()) {
                        for (Iterator<Map.Entry<String, JsonNode>> p = value.fields(); p.hasNext(); ) {
                            Map.Entry<String, JsonNode> property = p.next();
                            if (instance.has(property.getKey())) {
                                descend(instance.get(property.getKey()), property.getValue(), path,
                                        property.getKey(), out, depth + 1);
                            }
                        }
                    }
                    break;
                case "patternProperties":
                    if (instance.isObject()) {
                        for (Iterator<Map.Entry<String, JsonNode>> p = value.fields(); p.hasNext(); ) {
                            Map.Entry<String, JsonNode> pattern = p.next();
                            Pattern regex = Py.regex(pattern.getKey());
                            for (Iterator<Map.Entry<String, JsonNode>> f = instance.fields(); f.hasNext(); ) {
                                Map.Entry<String, JsonNode> field = f.next();
                                if (regex.matcher(field.getKey()).find()) {
                                    descend(field.getValue(), pattern.getValue(), path, field.getKey(), out,
                                            depth + 1);
                                }
                            }
                        }
                    }
                    break;
                case "additionalProperties":
                    additionalProperties(instance, value, schema, path, out, depth);
                    break;
                case "required":
                    if (instance.isObject() && value.isArray()) {
                        for (JsonNode property : value) {
                            if (!property.isTextual() || !instance.has(property.textValue())) {
                                out.add(new Issue(List.copyOf(path), Py.repr(property) + " is a required property",
                                        key, schema, instance));
                            }
                        }
                    }
                    break;
                case "items":
                    items(instance, value, schema, path, out, depth);
                    break;
                case "prefixItems":
                    if (instance.isArray() && value.isArray()) {
                        int n = Math.min(instance.size(), value.size());
                        for (int i = 0; i < n; i++) {
                            descend(instance.get(i), value.get(i), path, i, out, depth + 1);
                        }
                    }
                    break;
                case "allOf":
                    for (JsonNode branch : value) {
                        descend(instance, branch, path, null, out, depth + 1);
                    }
                    break;
                case "anyOf":
                    boolean valid = false;
                    for (JsonNode branch : value) {
                        List<Issue> branchErrors = new ArrayList<>();
                        descend(instance, branch, path, null, branchErrors, depth + 1);
                        if (branchErrors.isEmpty()) {
                            valid = true;
                            break;
                        }
                    }
                    if (!valid) {
                        out.add(new Issue(List.copyOf(path),
                                Py.repr(instance) + " is not valid under any of the given schemas", key, schema,
                                instance));
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private static void typeKeyword(JsonNode instance, JsonNode types, JsonNode schema, List<Object> path,
                                    List<Issue> out) {
        List<String> names = new ArrayList<>();
        if (types.isTextual()) {
            names.add(types.textValue());
        } else {
            for (JsonNode type : types) {
                names.add(type.asText());
            }
        }
        for (String type : names) {
            if (isType(instance, type)) {
                return;
            }
        }
        StringBuilder reprs = new StringBuilder();
        for (String type : names) {
            if (reprs.length() > 0) {
                reprs.append(", ");
            }
            reprs.append(Py.strRepr(type));
        }
        out.add(new Issue(List.copyOf(path), Py.repr(instance) + " is not of type " + reprs, "type", schema,
                instance));
    }

    private void additionalProperties(JsonNode instance, JsonNode rule, JsonNode schema, List<Object> path,
                                      List<Issue> out, int depth) {
        if (!instance.isObject()) {
            return;
        }
        JsonNode properties = schema.get("properties");
        JsonNode patterns = schema.get("patternProperties");
        Pattern joined = null;
        if (patterns != null && patterns.isObject() && patterns.size() > 0) {
            List<String> names = new ArrayList<>();
            patterns.fieldNames().forEachRemaining(names::add);
            joined = Py.regex(String.join("|", names));
        }
        List<String> extras = new ArrayList<>();
        for (Iterator<String> it = instance.fieldNames(); it.hasNext(); ) {
            String property = it.next();
            if (properties != null && properties.isObject() && properties.has(property)) {
                continue;
            }
            if (joined != null && joined.matcher(property).find()) {
                continue;
            }
            extras.add(property);
        }
        if (rule.isObject()) {
            for (String extra : extras) {
                descend(instance.get(extra), rule, path, extra, out, depth + 1);
            }
        } else if (rule.isBoolean() && !rule.booleanValue() && !extras.isEmpty()) {
            extras.sort(Py::compare);
            String message;
            if (schema.has("patternProperties")) {
                List<String> names = new ArrayList<>();
                schema.get("patternProperties").fieldNames().forEachRemaining(names::add);
                names.sort(Py::compare);
                message = joinRepr(extras) + (extras.size() == 1 ? " does" : " do") + " not match any of the regexes: "
                        + joinRepr(names);
            } else {
                message = "Additional properties are not allowed (" + joinRepr(extras)
                        + (extras.size() == 1 ? " was" : " were") + " unexpected)";
            }
            out.add(new Issue(List.copyOf(path), message, "additionalProperties", schema, instance));
        }
    }

    private void items(JsonNode instance, JsonNode rule, JsonNode schema, List<Object> path, List<Issue> out,
                       int depth) {
        if (!instance.isArray()) {
            return;
        }
        JsonNode prefixItems = schema.get("prefixItems");
        int prefix = prefixItems != null && prefixItems.isArray() ? prefixItems.size() : 0;
        int total = instance.size();
        int extra = total - prefix;
        if (extra <= 0) {
            return;
        }
        if (rule.isBoolean() && !rule.booleanValue()) {
            JsonNode rest;
            if (extra != 1) {
                ArrayNode tail = Json.array();
                for (int i = prefix; i < total; i++) {
                    tail.add(instance.get(i));
                }
                rest = tail;
            } else {
                rest = instance.get(prefix);
            }
            String item = prefix != 1 ? "items" : "item";
            out.add(new Issue(List.copyOf(path), "Expected at most " + prefix + " " + item + " but found " + extra
                    + " extra: " + Py.repr(rest), "items", schema, instance));
            return;
        }
        for (int index = prefix; index < total; index++) {
            descend(instance.get(index), rule, path, index, out, depth + 1);
        }
    }

    private JsonNode resolve(String reference) {
        if (reference.equals("#")) {
            return root;
        }
        if (!reference.startsWith("#/")) {
            throw new IllegalArgumentException("Unresolvable JSON Schema reference " + Py.strRepr(reference) + ".");
        }
        JsonNode node = root;
        for (String raw : reference.substring(2).split("/", -1)) {
            String token = URLDecoder.decode(raw.replace("+", "%2B"), StandardCharsets.UTF_8)
                    .replace("~1", "/").replace("~0", "~");
            if (node != null && node.isObject()) {
                node = node.get(token);
            } else if (node != null && node.isArray() && token.matches("0|[1-9][0-9]*") && token.length() < 10) {
                node = node.get(Integer.parseInt(token));
            } else {
                node = null;
            }
            if (node == null) {
                throw new IllegalArgumentException("Unresolvable JSON Schema reference " + Py.strRepr(reference) + ".");
            }
        }
        return node;
    }

    private static List<Object> with(List<Object> path, Object component) {
        List<Object> next = new ArrayList<>(path.size() + 1);
        next.addAll(path);
        next.add(component);
        return next;
    }

    private static String joinRepr(List<String> values) {
        StringBuilder sb = new StringBuilder();
        for (String value : values) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(Py.strRepr(value));
        }
        return sb.toString();
    }

    /** A JSON path component as a JSON value, for error paths. */
    static JsonNode component(Object component) {
        return component instanceof Integer ? IntNode.valueOf((Integer) component)
                : TextNode.valueOf(String.valueOf(component));
    }
}
