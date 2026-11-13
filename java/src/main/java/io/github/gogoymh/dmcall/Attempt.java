package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * One candidate with its verdict, scores and schema errors.
 *
 * @param calls                the candidate ({@code []} no call, {@code null} unreadable)
 * @param accepted             whether it passed; {@code null} for an unverified initial no call
 * @param functionProbability  the function question's probability of yes, or {@code null}
 * @param argumentsProbability the argument question's probability of yes, or {@code null}
 * @param schemaErrors         the schema check's findings
 */
public record Attempt(List<Call> calls, Boolean accepted, Double functionProbability, Double argumentsProbability,
                      List<SchemaError> schemaErrors) implements JsonConvertible {
    public Attempt {
        calls = Calls.copy(calls);
        schemaErrors = List.copyOf(schemaErrors);
    }

    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.set("calls", Calls.toJson(calls));
        node.put("accepted", accepted);
        node.put("function_probability", functionProbability);
        node.put("arguments_probability", argumentsProbability);
        ArrayNode errors = node.putArray("schema_errors");
        schemaErrors.forEach(error -> errors.add(error.toJson()));
        return node;
    }
}
