package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * The decision model's probabilities of yes for the final candidate; {@code null} where the
 * question was not asked.
 *
 * @param function  the function question
 * @param arguments the argument question
 */
public record Probabilities(Double function, Double arguments) implements JsonConvertible {
    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.put("function", function);
        node.put("arguments", arguments);
        return node;
    }
}
