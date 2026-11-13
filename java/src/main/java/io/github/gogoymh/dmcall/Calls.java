package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Calls from any SDK shape, and conversions between {@link Call} lists and JSON. */
public final class Calls {
    private Calls() {
    }

    /**
     * Calls as {@code [{"name", "arguments"}]} from DM-Call, OpenAI (Chat Completions or
     * Responses), Anthropic or Gemini shapes.
     *
     * <p>{@code null} means no call (OpenAI SDKs report {@code tool_calls = null}). A single call
     * object is a list of one. Arguments that are not valid JSON become {@code null}, which the
     * schema check rejects. A name that is not a string becomes {@code null}.
     *
     * @throws IllegalArgumentException if {@code calls} is not a call or a list of calls
     */
    public static List<Call> normalize(Object calls) {
        return fromJson(normalizeJson(Json.node(calls)));
    }

    static ArrayNode normalizeJson(JsonNode calls) {
        ArrayNode result = Json.array();
        if (Py.isNone(calls)) {
            return result;
        }
        if (calls.isObject()) {
            ArrayNode single = Json.array();
            single.add(calls);
            calls = single;
        }
        if (!calls.isArray()) {
            throw new IllegalArgumentException("calls must be a call or a list of calls.");
        }
        for (JsonNode item : calls) {
            if (!item.isObject()) {
                throw new IllegalArgumentException("Each call must be an object or an SDK tool-call object.");
            }
            for (String key : new String[]{"functionCall", "function_call", "function"}) {
                JsonNode inner = item.get(key);
                if (inner != null && inner.isObject()) {
                    item = inner;
                    break;
                }
            }
            JsonNode arguments = Json.object();
            for (String key : new String[]{"arguments", "input", "args"}) {
                if (item.has(key)) {
                    arguments = item.get(key);
                    break;
                }
            }
            if (arguments.isNull()) {
                arguments = Json.object();
            } else if (arguments.isTextual()) {
                String text = arguments.textValue();
                if (Py.strip(text).isEmpty()) {
                    arguments = Json.object();
                } else {
                    try {
                        arguments = Json.parse(text);
                    } catch (IllegalArgumentException e) {
                        arguments = NullNode.getInstance();
                    }
                }
            }
            ObjectNode call = Json.object();
            call.set("name", item.has("name") ? item.get("name").deepCopy() : NullNode.getInstance());
            call.set("arguments", arguments.deepCopy());
            result.add(call);
        }
        return result;
    }

    /** Calls from JSON {@code [{"name", "arguments"}]}; JSON {@code null} gives {@code null}. */
    public static List<Call> fromJson(JsonNode calls) {
        if (Py.isNone(calls)) {
            return null;
        }
        if (!calls.isArray()) {
            throw new IllegalArgumentException("calls must be a list of calls.");
        }
        List<Call> result = new ArrayList<>();
        for (JsonNode item : calls) {
            if (!item.isObject()) {
                throw new IllegalArgumentException("Each call must be an object.");
            }
            result.add(new Call(Py.text(item.get("name")), item.get("arguments")));
        }
        return Collections.unmodifiableList(result);
    }

    /** JSON for a call list; {@code null} (an unreadable candidate) gives JSON {@code null}. */
    public static JsonNode toJson(List<Call> calls) {
        if (calls == null) {
            return NullNode.getInstance();
        }
        ArrayNode array = Json.array();
        for (Call call : calls) {
            array.add(call.toJson());
        }
        return array;
    }

    static List<Call> copy(List<Call> calls) {
        return calls == null ? null : Collections.unmodifiableList(new ArrayList<>(calls));
    }
}
