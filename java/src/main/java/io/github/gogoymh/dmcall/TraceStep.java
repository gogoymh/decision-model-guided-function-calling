package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * One round of the loop: the candidate, its judgment, the schema check, the feedback record (as
 * the decision model produced it, before any representation) and whether it was accepted.
 *
 * @param attempt      the round, 0 for the first proposal
 * @param proposal     the candidate
 * @param verification the decision model's judgment
 * @param schema       the schema check
 * @param feedback     the feedback record for this round's question set
 * @param gatePass     whether the candidate was accepted; {@code null} for an unverified initial no call
 */
public record TraceStep(int attempt, Proposal proposal, Judgment verification, SchemaReport schema,
                        ObjectNode feedback, Boolean gatePass) implements JsonConvertible {
    public TraceStep {
        feedback = feedback == null ? Json.object() : feedback.deepCopy();
    }

    /** A copy of the feedback record. */
    @Override
    public ObjectNode feedback() {
        return feedback.deepCopy();
    }

    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.put("attempt", attempt);
        node.set("proposal", proposal.toJson());
        node.set("verification", verification.toJson());
        node.set("schema", schema.toJson());
        node.set("feedback", feedback.deepCopy());
        node.put("gate_pass", gatePass);
        return node;
    }
}
