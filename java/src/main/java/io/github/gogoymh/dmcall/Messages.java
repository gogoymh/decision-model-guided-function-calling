package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.util.Set;

/** Conversations as the generator and the decision model see them. */
final class Messages {
    private static final Set<String> ROLES = Set.of("system", "developer", "user", "assistant");
    private static final Set<String> HISTORY_ROLES = Set.of("system", "developer", "user", "assistant", "tool",
            "function");
    private static final String[] HISTORY_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"};
    private static final Set<String> TEXT_PARTS = Set.of("text", "input_text", "output_text");

    private Messages() {
    }

    private static JsonNode asNode(Object messages) {
        if (messages instanceof String) {
            ArrayNode single = Json.array();
            ObjectNode message = single.addObject();
            message.put("role", "user");
            message.put("content", (String) messages);
            return single;
        }
        JsonNode node = Json.node(messages);
        if (node.isTextual()) {
            return asNode(node.textValue());
        }
        return node;
    }

    /** A prompt string or chat messages as {@code [{"role", "content"}]} with text content. */
    static ArrayNode forGenerator(Object messages) {
        JsonNode node = asNode(messages);
        if (node.isObject()) {
            ArrayNode single = Json.array();
            single.add(node);
            node = single;
        }
        if (!node.isArray() || node.size() == 0) {
            throw new IllegalArgumentException(
                    "messages must be a string or a nonempty list of {'role', 'content'} messages.");
        }
        ArrayNode result = Json.array();
        for (JsonNode message : node) {
            String role = message.isObject() ? Py.text(message.get("role")) : null;
            if (role == null || !ROLES.contains(role)) {
                throw new IllegalArgumentException("Each message needs a role (system, developer, user or assistant)"
                        + " and text content; give earlier tool results to the model as text.");
            }
            JsonNode content = message.get("content");
            if (content != null && content.isArray()) {
                StringBuilder text = new StringBuilder();
                boolean first = true;
                for (JsonNode part : content) {
                    if (!part.isObject() || !TEXT_PARTS.contains(Py.text(part.get("type")))
                            || part.get("text") == null || !part.get("text").isTextual()) {
                        throw new IllegalArgumentException("Only text message content is supported.");
                    }
                    if (!first) {
                        text.append('\n');
                    }
                    first = false;
                    text.append(part.get("text").textValue());
                }
                content = TextNode.valueOf(text.toString());
            }
            if (content == null || !content.isTextual()) {
                throw new IllegalArgumentException("Message content must be text.");
            }
            ObjectNode copy = result.addObject();
            copy.put("role", role);
            copy.put("content", content.textValue());
        }
        return result;
    }

    /**
     * A conversation as the decision model sees it: text turns, earlier tool calls and tool results.
     * Only role, content, name, tool_calls and tool_call_id are kept.
     */
    static ArrayNode conversation(Object messages) {
        JsonNode node = asNode(messages);
        if (!node.isArray()) {
            ArrayNode single = Json.array();
            single.add(node);
            node = single;
        }
        ArrayNode result = Json.array();
        for (JsonNode message : node) {
            String role = message.isObject() ? Py.text(message.get("role")) : null;
            if (role == null || !HISTORY_ROLES.contains(role)) {
                throw new IllegalArgumentException(
                        "Each message needs a role: system, developer, user, assistant or tool.");
            }
            ObjectNode copy = result.addObject();
            for (String key : HISTORY_KEYS) {
                JsonNode value = message.get(key);
                if (!Py.isNone(value)) {
                    copy.set(key, value.deepCopy());
                }
            }
        }
        if (result.isEmpty()) {
            throw new IllegalArgumentException("messages must not be empty.");
        }
        return result;
    }
}
