package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * One finding of the schema check: which check failed ({@code function} or {@code arguments}),
 * where ({@code calls[0].arguments["city"]}) and the message.
 */
public record SchemaError(String check, String path, String message) implements JsonConvertible {
    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.put("check", check);
        node.put("path", path);
        node.put("message", message);
        return node;
    }
}
