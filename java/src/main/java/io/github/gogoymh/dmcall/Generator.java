package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.node.ArrayNode;

/**
 * The model that proposes calls (the generator G of the paper). The built-in providers implement
 * it over HTTP; any other implementation can stand in.
 */
public interface Generator extends AutoCloseable {
    /**
     * One proposal after the given repair history.
     *
     * @param messages the request as {@code [{"role", "content"}]} text messages
     * @param tools    tool definitions {@code [{"type": "function", "function": {...}}]}
     * @param history  earlier rejected candidates and their feedback, oldest first, each
     *                 {@code {"calls", "feedback"}} with an optional {@code "round"}; empty on the first
     *                 proposal
     */
    Proposal generate(ArrayNode messages, ArrayNode tools, ArrayNode history);

    /** Releases resources this generator created. */
    @Override
    default void close() {
    }
}
