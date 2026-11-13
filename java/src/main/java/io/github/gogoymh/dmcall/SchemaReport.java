package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * The result of the schema check S(c).
 *
 * @param functionPass  no function finding (the function exists and is named correctly)
 * @param argumentsPass no argument finding (required arguments present with their declared types)
 * @param accepted      S(c): both pass
 * @param wellFormed    the candidate is readable, so the decision model can judge it
 * @param errors        the findings in order
 */
public record SchemaReport(boolean functionPass, boolean argumentsPass, boolean accepted, boolean wellFormed,
                           List<SchemaError> errors) implements JsonConvertible {
    public SchemaReport {
        errors = List.copyOf(errors);
    }

    /** The errors as JSON, as they appear in feedback records. */
    public ArrayNode errorsJson() {
        ArrayNode array = Json.array();
        for (SchemaError error : errors) {
            array.add(error.toJson());
        }
        return array;
    }

    /** The check's own feedback record. */
    public ObjectNode feedback() {
        ObjectNode feedback = Json.object();
        feedback.put("source", "deterministic_schema_validation");
        feedback.put("function_pass", functionPass);
        feedback.put("arguments_pass", argumentsPass);
        feedback.set("schema_errors", errorsJson());
        return feedback;
    }

    /** The same fields as the Python reference's {@code check_schema} result. */
    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.put("function_pass", functionPass);
        node.put("arguments_pass", argumentsPass);
        node.put("accepted", accepted);
        node.put("well_formed", wellFormed);
        node.set("errors", errorsJson());
        node.set("feedback", feedback());
        return node;
    }
}
