package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * JSON values as DM-Call holds them: Jackson trees that keep key order and whether a number was
 * written as an integer or a float, and text written exactly like Python's {@code json.dumps}.
 *
 * <p>The repair history reaches the generator as JSON text inside a message, so its characters
 * must match what the paper's experiments sent: {@code ", "} and {@code ": "} separators, floats in
 * Python {@code repr} form ({@code 1.0}, {@code 1e+16}, {@code 1e-05}) and non-ASCII text kept.
 */
public final class Json {
    private static final ObjectMapper MAPPER = JsonMapper.builder()
            .enable(JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .build();
    private static final JsonNodeFactory NODES = JsonNodeFactory.instance;

    private Json() {
    }

    /** The configured mapper: floats stay doubles, integers stay integers, key order is kept. */
    public static ObjectMapper mapper() {
        return MAPPER;
    }

    public static ObjectNode object() {
        return NODES.objectNode();
    }

    public static ArrayNode array() {
        return NODES.arrayNode();
    }

    /** Parses JSON text. Invalid text raises {@link IllegalArgumentException} without echoing it. */
    public static JsonNode parse(String text) {
        try {
            JsonNode node = MAPPER.readTree(text);
            if (node == null || node.isMissingNode()) {
                throw new IllegalArgumentException("The text is not a JSON value.");
            }
            return node;
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("The text is not valid JSON.");
        }
    }

    /** Parses a UTF-8, UTF-16 or UTF-32 JSON document. */
    public static JsonNode parse(byte[] bytes) {
        try {
            JsonNode node = MAPPER.readTree(bytes == null ? new byte[0] : bytes);
            if (node == null || node.isMissingNode()) {
                throw new IllegalArgumentException("The body is not a JSON value.");
            }
            return node;
        } catch (IOException e) {
            throw new IllegalArgumentException("The body is not valid JSON.");
        }
    }

    /**
     * A JSON tree for a Java value: {@link JsonNode} (copied), {@link JsonConvertible}, String,
     * Boolean, numbers (integers stay integers, {@code Double} and {@code Float} become floats),
     * {@code Map}, {@code Iterable} and arrays. Anything else goes through Jackson.
     */
    public static JsonNode node(Object value) {
        if (value == null) {
            return NullNode.getInstance();
        }
        if (value instanceof JsonNode) {
            return ((JsonNode) value).deepCopy();
        }
        if (value instanceof JsonConvertible) {
            return ((JsonConvertible) value).toJson();
        }
        if (value instanceof CharSequence) {
            return NODES.textNode(value.toString());
        }
        if (value instanceof Boolean) {
            return NODES.booleanNode((Boolean) value);
        }
        if (value instanceof Integer || value instanceof Short || value instanceof Byte) {
            return NODES.numberNode(((Number) value).intValue());
        }
        if (value instanceof Long) {
            return NODES.numberNode((Long) value);
        }
        if (value instanceof BigInteger) {
            return NODES.numberNode((BigInteger) value);
        }
        if (value instanceof Float) {
            return NODES.numberNode(Double.parseDouble(value.toString()));
        }
        if (value instanceof BigDecimal || value instanceof Number) {
            return NODES.numberNode(((Number) value).doubleValue());
        }
        if (value instanceof Map) {
            ObjectNode result = object();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                result.set(String.valueOf(entry.getKey()), node(entry.getValue()));
            }
            return result;
        }
        if (value instanceof Iterable) {
            ArrayNode result = array();
            for (Object item : (Iterable<?>) value) {
                result.add(node(item));
            }
            return result;
        }
        if (value instanceof Object[]) {
            ArrayNode result = array();
            for (Object item : (Object[]) value) {
                result.add(node(item));
            }
            return result;
        }
        return MAPPER.valueToTree(value);
    }

    /** {@code json.dumps(value, ensure_ascii=False)} in Python. */
    public static String dumps(JsonNode value) {
        return dumps(value, false, false);
    }

    /** {@code json.dumps(value, ensure_ascii=ensureAscii, sort_keys=sortKeys)} in Python. */
    public static String dumps(JsonNode value, boolean ensureAscii, boolean sortKeys) {
        StringBuilder sb = new StringBuilder();
        write(sb, value, ensureAscii, sortKeys, ", ", ": ");
        return sb.toString();
    }

    /** Compact JSON for HTTP bodies: separators {@code ","} and {@code ":"}, non-ASCII kept. */
    static String compact(JsonNode value) {
        StringBuilder sb = new StringBuilder();
        write(sb, value, false, false, ",", ":");
        return sb.toString();
    }

    /** Whether every number in the value is finite ({@code json.dumps(..., allow_nan=False)} succeeds). */
    public static boolean isFinite(JsonNode value) {
        if (value == null) {
            return true;
        }
        if (value.isFloatingPointNumber()) {
            double d = value.doubleValue();
            return !Double.isNaN(d) && !Double.isInfinite(d);
        }
        if (value.isContainerNode()) {
            for (JsonNode item : value) {
                if (!isFinite(item)) {
                    return false;
                }
            }
        }
        return true;
    }

    private static void write(StringBuilder sb, JsonNode value, boolean ascii, boolean sort, String itemSep,
                              String keySep) {
        if (value == null || value.isNull() || value.isMissingNode()) {
            sb.append("null");
        } else if (value.isBoolean()) {
            sb.append(value.booleanValue() ? "true" : "false");
        } else if (value.isIntegralNumber()) {
            sb.append(value.bigIntegerValue());
        } else if (value.isNumber()) {
            sb.append(Py.floatJson(value.doubleValue()));
        } else if (value.isTextual()) {
            writeString(sb, value.textValue(), ascii);
        } else if (value.isArray()) {
            sb.append('[');
            for (int i = 0; i < value.size(); i++) {
                if (i > 0) {
                    sb.append(itemSep);
                }
                write(sb, value.get(i), ascii, sort, itemSep, keySep);
            }
            sb.append(']');
        } else if (value.isObject()) {
            List<Map.Entry<String, JsonNode>> entries = new ArrayList<>();
            for (Iterator<Map.Entry<String, JsonNode>> it = value.fields(); it.hasNext(); ) {
                entries.add(it.next());
            }
            if (sort) {
                entries.sort((a, b) -> Py.compare(a.getKey(), b.getKey()));
            }
            sb.append('{');
            for (int i = 0; i < entries.size(); i++) {
                if (i > 0) {
                    sb.append(itemSep);
                }
                writeString(sb, entries.get(i).getKey(), ascii);
                sb.append(keySep);
                write(sb, entries.get(i).getValue(), ascii, sort, itemSep, keySep);
            }
            sb.append('}');
        } else {
            writeString(sb, value.asText(), ascii);
        }
    }

    private static void writeString(StringBuilder sb, String s, boolean ascii) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                case '\b':
                    sb.append("\\b");
                    break;
                case '\f':
                    sb.append("\\f");
                    break;
                default:
                    if (c < 0x20 || (ascii && c > 0x7e)) {
                        sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
    }
}
