package io.github.gogoymh.dmcall;

import java.util.LinkedHashMap;
import java.util.Map;

/** Per-request overrides for {@link DMCall#call} and {@link DMCall#run}. */
public final class CallOptions {
    Object tools;
    Map<String, ToolHandler> functions;
    Integer maxRepairs;
    OnExhaustion onExhaustion;
    Boolean executeUnverified;

    private CallOptions() {
    }

    public static CallOptions create() {
        return new CallOptions();
    }

    /** Tools for this request instead of the client's (and their handlers). */
    public CallOptions tools(Object tools) {
        this.tools = tools;
        return this;
    }

    /** A handler for a tool, for this request. */
    public CallOptions function(String name, ToolHandler handler) {
        if (functions == null) {
            functions = new LinkedHashMap<>();
        }
        functions.put(name, handler);
        return this;
    }

    public CallOptions maxRepairs(int maxRepairs) {
        this.maxRepairs = maxRepairs;
        return this;
    }

    public CallOptions onExhaustion(OnExhaustion onExhaustion) {
        this.onExhaustion = onExhaustion;
        return this;
    }

    public CallOptions onExhaustion(String onExhaustion) {
        this.onExhaustion = OnExhaustion.fromValue(onExhaustion);
        return this;
    }

    /** For {@link DMCall#run}: also execute the final candidate when none was accepted. */
    public CallOptions executeUnverified(boolean executeUnverified) {
        this.executeUnverified = executeUnverified;
        return this;
    }
}
