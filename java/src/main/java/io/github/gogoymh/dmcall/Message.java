package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * A text chat message. For {@link DMCall#verify} conversations with earlier tool calls and tool
 * results, pass maps or JSON with {@code tool_calls} and {@code tool_call_id} instead.
 *
 * @param role    system, developer, user or assistant
 * @param content the text
 */
public record Message(String role, String content) implements JsonConvertible {
    public static Message system(String content) {
        return new Message("system", content);
    }

    public static Message developer(String content) {
        return new Message("developer", content);
    }

    public static Message user(String content) {
        return new Message("user", content);
    }

    public static Message assistant(String content) {
        return new Message("assistant", content);
    }

    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.put("role", role);
        node.put("content", content);
        return node;
    }
}
