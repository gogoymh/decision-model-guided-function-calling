package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * The schema check's and the decision model's judgment of one candidate from any source.
 *
 * @param accepted             the calls passed the schema check and every applicable question
 * @param calls                the calls as normalized
 * @param functionProbability  the function question's probability of yes, or {@code null}
 * @param argumentsProbability the argument question's probability of yes, or {@code null}
 * @param schemaErrors         the schema check's findings
 * @param feedback             the feedback record
 * @param raw                  the full judgment
 */
public record Verdict(boolean accepted, List<Call> calls, Double functionProbability, Double argumentsProbability,
                      List<SchemaError> schemaErrors, ObjectNode feedback, Judgment raw) implements JsonConvertible {
    public Verdict {
        calls = Calls.copy(calls);
        schemaErrors = List.copyOf(schemaErrors);
        feedback = feedback.deepCopy();
    }

    /** A copy of the feedback record. */
    @Override
    public ObjectNode feedback() {
        return feedback.deepCopy();
    }

    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.put("accepted", accepted);
        node.set("calls", Calls.toJson(calls));
        node.put("function_probability", functionProbability);
        node.put("arguments_probability", argumentsProbability);
        ArrayNode errors = node.putArray("schema_errors");
        schemaErrors.forEach(error -> errors.add(error.toJson()));
        node.set("feedback", feedback.deepCopy());
        return node;
    }

    @Override
    public String toString() {
        return "Verdict(accepted=" + accepted + ", calls=" + calls + ", function=" + functionProbability
                + ", arguments=" + argumentsProbability + ")";
    }
}
