package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.node.ArrayNode;

import java.util.List;

/**
 * A decision model (D in the paper). It answers two yes/no questions about a candidate call and
 * returns the probability of yes for each: the function question (is this the right function, or
 * is making no call right?) and the argument question (are the arguments present, supported by the
 * request and in the right parameters?). The argument question is not asked for a no call, and
 * neither question is asked for an unreadable candidate.
 *
 * <p>{@link SystemOneDecisionModel} implements the HTTP protocol both of the paper's decision
 * models speak; {@link Jev} and {@link PPLXDecider} extend it. Only the request, the function
 * schemas and the candidate are sent, never a reference answer.
 */
public interface DecisionModel extends AutoCloseable {
    /**
     * Judges one candidate: the schema check, both questions, the verdicts and the feedback record.
     *
     * @param messages the conversation ({@code role} and {@code content}, plus {@code tool_calls} or
     *                 {@code tool_call_id} for earlier tool turns)
     * @param tools    the tool definitions
     * @param calls    the candidate ({@code []} for no call, {@code null} if unreadable)
     */
    Judgment judge(ArrayNode messages, ArrayNode tools, List<Call> calls);

    /** Releases resources this decision model created. */
    @Override
    default void close() {
    }
}
