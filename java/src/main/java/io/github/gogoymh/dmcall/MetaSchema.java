package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * A check of a parameter schema against the JSON Schema Draft 2020-12 meta-schema, covering the
 * keywords of the core, applicator, validation, meta-data, format and content vocabularies (what
 * Python jsonschema's {@code check_schema} rejects). Formats other than {@code regex} are not checked,
 * as in Python jsonschema without optional format packages.
 */
final class MetaSchema {
    static final Set<String> SIMPLE_TYPES = Set.of("array", "boolean", "integer", "null", "number", "object",
            "string");
    private static final Pattern ANCHOR = Pattern.compile("[A-Za-z_][-A-Za-z0-9._]*");
    private static final Pattern ID = Pattern.compile("[^#]*#?");

    /** An invalid schema; Python jsonschema's {@code SchemaError}. */
    static final class InvalidSchema extends RuntimeException {
        InvalidSchema(String message) {
            super(message);
        }
    }

    private MetaSchema() {
    }

    static void check(JsonNode schema) {
        check(schema, 0);
    }

    private static void check(JsonNode schema, int depth) {
        if (depth > 512) {
            throw new InvalidSchema("The schema is nested too deeply.");
        }
        if (schema.isBoolean()) {
            return;
        }
        if (!schema.isObject()) {
            throw new InvalidSchema("A schema must be an object or a boolean.");
        }
        for (Iterator<Map.Entry<String, JsonNode>> it = schema.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            switch (key) {
                case "type":
                    if (value.isTextual()) {
                        require(SIMPLE_TYPES.contains(value.textValue()), key);
                    } else {
                        require(value.isArray() && value.size() > 0 && unique(value), key);
                        for (JsonNode item : value) {
                            require(item.isTextual() && SIMPLE_TYPES.contains(item.textValue()), key);
                        }
                    }
                    break;
                case "properties":
                case "$defs":
                case "definitions":
                case "dependentSchemas":
                    require(value.isObject(), key);
                    for (JsonNode item : value) {
                        check(item, depth + 1);
                    }
                    break;
                case "patternProperties":
                    require(value.isObject(), key);
                    for (Iterator<Map.Entry<String, JsonNode>> p = value.fields(); p.hasNext(); ) {
                        Map.Entry<String, JsonNode> pattern = p.next();
                        requireRegex(pattern.getKey(), key);
                        check(pattern.getValue(), depth + 1);
                    }
                    break;
                case "dependencies":
                    require(value.isObject(), key);
                    for (JsonNode item : value) {
                        if (item.isArray()) {
                            requireStringArray(item, key);
                        } else {
                            check(item, depth + 1);
                        }
                    }
                    break;
                case "additionalProperties":
                case "items":
                case "not":
                case "if":
                case "then":
                case "else":
                case "contains":
                case "propertyNames":
                case "unevaluatedItems":
                case "unevaluatedProperties":
                case "contentSchema":
                    check(value, depth + 1);
                    break;
                case "prefixItems":
                case "allOf":
                case "anyOf":
                case "oneOf":
                    require(value.isArray() && value.size() > 0, key);
                    for (JsonNode item : value) {
                        check(item, depth + 1);
                    }
                    break;
                case "required":
                    requireStringArray(value, key);
                    break;
                case "enum":
                case "examples":
                    require(value.isArray(), key);
                    break;
                case "multipleOf":
                    require(value.isNumber() && value.doubleValue() > 0, key);
                    break;
                case "maximum":
                case "exclusiveMaximum":
                case "minimum":
                case "exclusiveMinimum":
                    require(value.isNumber(), key);
                    break;
                case "maxLength":
                case "minLength":
                case "maxItems":
                case "minItems":
                case "maxProperties":
                case "minProperties":
                case "maxContains":
                case "minContains":
                    require(TypeSchemaValidator.isType(value, "integer") && value.doubleValue() >= 0, key);
                    break;
                case "pattern":
                    require(value.isTextual(), key);
                    requireRegex(value.textValue(), key);
                    break;
                case "uniqueItems":
                case "deprecated":
                case "readOnly":
                case "writeOnly":
                case "$recursiveAnchor":
                    require(value.isBoolean(), key);
                    break;
                case "dependentRequired":
                    require(value.isObject(), key);
                    for (JsonNode item : value) {
                        requireStringArray(item, key);
                    }
                    break;
                case "format":
                case "title":
                case "description":
                case "$comment":
                case "contentEncoding":
                case "contentMediaType":
                case "$ref":
                case "$schema":
                case "$dynamicRef":
                case "$recursiveRef":
                    require(value.isTextual(), key);
                    break;
                case "$id":
                    require(value.isTextual() && ID.matcher(value.textValue()).matches(), key);
                    break;
                case "$anchor":
                case "$dynamicAnchor":
                    require(value.isTextual() && ANCHOR.matcher(value.textValue()).matches(), key);
                    break;
                case "$vocabulary":
                    require(value.isObject(), key);
                    for (JsonNode item : value) {
                        require(item.isBoolean(), key);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private static void require(boolean condition, String keyword) {
        if (!condition) {
            throw new InvalidSchema("Invalid value for the JSON Schema keyword " + keyword + ".");
        }
    }

    private static void requireStringArray(JsonNode value, String keyword) {
        require(value.isArray() && unique(value), keyword);
        for (JsonNode item : value) {
            require(item.isTextual(), keyword);
        }
    }

    private static void requireRegex(String pattern, String keyword) {
        try {
            Py.regex(pattern);
        } catch (IllegalArgumentException e) {
            throw new InvalidSchema("Invalid regular expression for the JSON Schema keyword " + keyword + ".");
        }
    }

    private static boolean unique(JsonNode array) {
        List<String> seen = new ArrayList<>();
        for (JsonNode item : array) {
            String canonical = Json.dumps(item, true, true);
            if (seen.contains(canonical)) {
                return false;
            }
            seen.add(canonical);
        }
        return true;
    }
}
