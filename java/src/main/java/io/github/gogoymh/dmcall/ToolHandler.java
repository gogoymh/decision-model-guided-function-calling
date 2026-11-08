package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.node.ObjectNode;

/** The code behind a tool. {@link DMCall#run} calls it with an accepted call's arguments. */
@FunctionalInterface
public interface ToolHandler {
    /**
     * Executes the tool.
     *
     * @param arguments the accepted call's arguments (a copy)
     * @return the tool's output, returned as {@link Result#output()}
     * @throws Exception any failure; checked exceptions reach the caller as {@link ToolExecutionException}
     */
    Object call(ObjectNode arguments) throws Exception;
}
