package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;

/**
 * Decision-model-guided function calling with cumulative verification feedback.
 *
 * <p>At step t the generator proposes a candidate (one call, or no call). The candidate is accepted
 * when it passes the schema check and every decision-model question that applies to it. A rejected
 * candidate and its feedback join the history, which the generator reads, oldest first and marked
 * as not executed, before the unchanged request. The loop returns the first accepted candidate; if
 * none of the r + 1 candidates is accepted it returns the first. Nothing is executed while the loop
 * runs.
 *
 * <pre>{@code
 * DMCall agent = DMCall.builder(OpenAI.builder("gpt-5-nano").reasoningEffort("none").build())
 *         .tools(weather)                      // a Tool with a handler
 *         .build();                            // Jev from TYPESAFE_API_KEY, the paper's settings
 * Result result = agent.run("Is it warm in Seoul right now?");
 * }</pre>
 *
 * <p>The defaults are the paper's settings: at most 3 repairs, both questions, the whole history,
 * hybrid feedback, the first candidate when none passes, and a readable initial no call returned
 * without verification. The paper's ablations are {@link Checks#FUNCTION} (DM-Call-Function),
 * {@link Checks#ARGUMENTS} (DM-Call-Argument) and {@link History#LATEST} (DM-Call-Latest).
 */
public final class DMCall implements AutoCloseable {
    /** The pass mark of both questions in the paper. */
    public static final double DEFAULT_THRESHOLD = 0.5;
    /** The repair limit r* in the paper. */
    public static final int DEFAULT_MAX_REPAIRS = 3;

    private final Generator generator;
    private final DecisionModel decisionModel;
    private final List<Tool> tools;
    private final Map<String, ToolHandler> functions;
    private final int maxRepairs;
    private final Checks checks;
    private final History history;
    private final FeedbackMode feedback;
    private final OnExhaustion onExhaustion;
    private final boolean preserveInitialNoCall;
    private final boolean executeUnverified;
    private final List<AutoCloseable> owned;

    private DMCall(Builder b, Generator generator, DecisionModel decisionModel, List<AutoCloseable> owned) {
        this.generator = generator;
        this.decisionModel = decisionModel;
        this.tools = b.tools;
        this.functions = b.functions;
        this.maxRepairs = b.maxRepairs;
        this.checks = b.checks;
        this.history = b.history;
        this.feedback = b.feedback;
        this.onExhaustion = b.onExhaustion;
        this.preserveInitialNoCall = b.preserveInitialNoCall;
        this.executeUnverified = b.executeUnverified;
        this.owned = owned;
    }

    /** A client around a generator, which stays the caller's to close. */
    public static Builder builder(Generator generator) {
        if (generator == null) {
            throw new IllegalArgumentException("A generator is required.");
        }
        return new Builder(generator, null);
    }

    /**
     * A client around a provider from a {@code "provider:model"} spec such as
     * {@code "openai:gpt-5-nano"}, configured from environment variables. The client owns and
     * closes it.
     */
    public static Builder builder(String spec) {
        if (spec == null) {
            throw new IllegalArgumentException("A model is a \"provider:model\" string or a generator.");
        }
        return new Builder(null, spec);
    }

    // Public API ---------------------------------------------------------------------------
    /**
     * Proposes, verifies and repairs a call. Nothing is executed.
     *
     * @param messages a prompt string, a {@link Message}, or a list of messages ({@link Message}s, maps
     *                 or JSON objects with {@code role} and text {@code content})
     */
    public Result call(Object messages) {
        return call(messages, CallOptions.create());
    }

    /** {@link #call(Object)} with per-request overrides. */
    public Result call(Object messages, CallOptions options) {
        return solve(messages, options == null ? CallOptions.create() : options, false, false);
    }

    /**
     * {@link #call(Object)}, then executes the final call with its tool's handler if it was accepted.
     * With {@code executeUnverified} the final candidate also runs when none was accepted: the first
     * one, or the last with {@link OnExhaustion#LAST}.
     *
     * Every available tool must have a handler before any generator or decision-model request.
     * @throws NoSuchElementException   if an available tool has no handler
     * @throws ToolExecutionException   if a handler throws a checked exception
     */
    public Result run(Object messages) {
        return run(messages, CallOptions.create());
    }

    /** {@link #run(Object)} with per-request overrides. */
    public Result run(Object messages, CallOptions options) {
        CallOptions o = options == null ? CallOptions.create() : options;
        boolean unverified = o.executeUnverified != null ? o.executeUnverified : executeUnverified;
        return solve(messages, o, true, unverified);
    }

    /**
     * Checks a proposed call from any source with the schema check and both questions; nothing is
     * generated or executed.
     *
     * @param messages the conversation so far, including earlier tool calls and tool results
     * @param calls    {@link Call}s, or calls in any SDK shape (see {@link Calls#normalize})
     */
    public Verdict verify(Object messages, Object calls) {
        return verify(messages, calls, null);
    }

    /** {@link #verify(Object, Object)} against other tools. */
    public Verdict verify(Object messages, Object calls, Object tools) {
        return Verifier.verdict(decisionModel, messages, calls, tools == null ? this.tools : tools);
    }

    /** {@link #call(Object)} on the common pool. */
    public CompletableFuture<Result> callAsync(Object messages) {
        return callAsync(messages, CallOptions.create(), ForkJoinPool.commonPool());
    }

    public CompletableFuture<Result> callAsync(Object messages, CallOptions options, Executor executor) {
        return CompletableFuture.supplyAsync(() -> call(messages, options), executor);
    }

    /** {@link #run(Object)} on the common pool; the handlers run there too. */
    public CompletableFuture<Result> runAsync(Object messages) {
        return runAsync(messages, CallOptions.create(), ForkJoinPool.commonPool());
    }

    public CompletableFuture<Result> runAsync(Object messages, CallOptions options, Executor executor) {
        return CompletableFuture.supplyAsync(() -> run(messages, options), executor);
    }

    /** {@link #verify(Object, Object)} on the common pool. */
    public CompletableFuture<Verdict> verifyAsync(Object messages, Object calls) {
        return CompletableFuture.supplyAsync(() -> verify(messages, calls), ForkJoinPool.commonPool());
    }

    public Generator generator() {
        return generator;
    }

    public DecisionModel decisionModel() {
        return decisionModel;
    }

    public List<Tool> tools() {
        return tools;
    }

    public int maxRepairs() {
        return maxRepairs;
    }

    public Checks checks() {
        return checks;
    }

    public History history() {
        return history;
    }

    public FeedbackMode feedback() {
        return feedback;
    }

    public OnExhaustion onExhaustion() {
        return onExhaustion;
    }

    public boolean preserveInitialNoCall() {
        return preserveInitialNoCall;
    }

    public boolean executeUnverified() {
        return executeUnverified;
    }

    /** Closes the generator and decision model this client created, not ones passed in. */
    @Override
    public void close() {
        for (AutoCloseable resource : owned) {
            try {
                resource.close();
            } catch (Exception ignored) {
                // Closing is best effort.
            }
        }
    }

    @Override
    public String toString() {
        return "DMCall(model=" + generator + ", decisionModel=" + decisionModel + ", checks=" + checks
                + ", history=" + history + ", feedback=" + feedback + ", maxRepairs=" + maxRepairs
                + ", onExhaustion=" + onExhaustion + ", preserveInitialNoCall=" + preserveInitialNoCall + ")";
    }

    // The loop -----------------------------------------------------------------------------
    private record Acceptance(boolean accepted, ObjectNode feedback) {
    }

    /** A_t for the configured questions, and the feedback record the generator gets. */
    private Acceptance accept(SchemaReport local, Judgment check) {
        boolean functionPass = Boolean.TRUE.equals(check.functionPass());
        boolean arguments = check.argumentsSkipped() || Boolean.TRUE.equals(check.argumentsPass());
        switch (checks) {
            case FUNCTION:
                return new Acceptance(local.accepted() && functionPass,
                        Feedback.singleCheck(check.feedback(), "function"));
            case ARGUMENTS:
                return new Acceptance(local.accepted() && arguments,
                        Feedback.singleCheck(check.feedback(), "arguments"));
            default:
                return new Acceptance(local.accepted() && functionPass && arguments, check.feedback());
        }
    }

    private Result solve(Object messages, CallOptions options, boolean execute, boolean unverified) {
        int limit = options.maxRepairs == null ? maxRepairs : repairs(options.maxRepairs);
        OnExhaustion rule = options.onExhaustion == null ? onExhaustion : options.onExhaustion;
        Tools.Toolset toolset;
        if (options.tools == null) {
            Map<String, ToolHandler> merged = new LinkedHashMap<>(functions);
            if (options.functions != null) {
                merged.putAll(options.functions);
            }
            toolset = Tools.toolset(tools, merged);
        } else {
            toolset = Tools.toolset(options.tools, options.functions);
            Map<String, ToolHandler> known = Tools.toolset(tools, functions).registry();
            for (var schema : toolset.schemas()) {
                String name = schema.get("function").get("name").textValue();
                if (!toolset.registry().containsKey(name) && known.containsKey(name)) {
                    toolset.registry().put(name, known.get(name));
                }
            }
        }
        ArrayNode schemas = toolset.schemas();
        if (execute) {
            for (var schema : schemas) {
                String name = schema.get("function").get("name").textValue();
                if (!toolset.registry().containsKey(name)) {
                    throw new NoSuchElementException("No handler for tool " + Py.strRepr(name) + ".");
                }
            }
        }
        ArrayNode conversation = Messages.forGenerator(messages);
        long started = System.nanoTime();
        List<ObjectNode> rejected = new ArrayList<>();
        List<TraceStep> trace = new ArrayList<>();
        boolean accepted = false;
        for (int attempt = 0; attempt <= limit; attempt++) {
            ArrayNode shown = Json.array();
            if (history == History.LATEST && !rejected.isEmpty()) {
                // Only the latest rejection, with its real round number.
                ObjectNode latest = rejected.get(rejected.size() - 1).deepCopy();
                latest.put("round", attempt);
                shown.add(latest);
            } else {
                rejected.forEach(item -> shown.add(item.deepCopy()));
            }
            Proposal proposal = generator.generate(conversation.deepCopy(), schemas.deepCopy(),
                    Feedback.representHistory(shown, feedback));
            List<Call> calls = proposal.calls();
            SchemaReport local = SchemaCheck.checkJson(schemas, Calls.toJson(calls));
            if (attempt == 0 && preserveInitialNoCall && proposal.isReadableNoCall() && local.accepted()) {
                // A readable initial no call is returned without verification or repair.
                Judgment check = Judgment.builder()
                        .kind("initial_no_call_preserved")
                        .argumentsSkipped(true)
                        .apiCalls(0)
                        .feedback(Json.object())
                        .build();
                trace.add(new TraceStep(0, proposal, check, local, Json.object(), null));
                return new Result(List.of(), null, false, 0, proposal.content(), false, List.of(),
                        (System.nanoTime() - started) / 1e9, trace, 0, true);
            }
            Judgment check = decisionModel.judge(conversation.deepCopy(), schemas.deepCopy(), calls);
            Acceptance acceptance = accept(local, check);
            accepted = acceptance.accepted();
            trace.add(new TraceStep(attempt, proposal, check, local, acceptance.feedback(), accepted));
            if (accepted) {
                break;
            }
            ObjectNode record = Json.object();
            record.set("calls", Calls.toJson(calls));
            record.set("feedback", acceptance.feedback().deepCopy());
            rejected.add(record);
        }

        boolean exhausted = !accepted;
        int finalAttempt = exhausted && rule == OnExhaustion.FIRST ? 0 : trace.size() - 1;
        List<Call> calls = trace.get(finalAttempt).proposal().calls();
        List<Object> outputs = new ArrayList<>();
        boolean executed = false;
        if (execute && (accepted || unverified)) {
            executed = execute(calls, toolset.registry(), outputs);
        }
        return new Result(calls, accepted, exhausted, trace.size() - 1, trace.get(finalAttempt).proposal().content(),
                executed, outputs, (System.nanoTime() - started) / 1e9, trace, finalAttempt, false);
    }

    /** Runs the final calls with their handlers. An unreadable candidate never runs. */
    private static boolean execute(List<Call> calls, Map<String, ToolHandler> registry, List<Object> outputs) {
        if (calls == null) {
            return false;
        }
        for (Call call : calls) {
            if (call.name() == null || call.arguments() == null || !call.arguments().isObject()) {
                return false;
            }
        }
        for (Call call : calls) {
            if (!registry.containsKey(call.name())) {
                throw new NoSuchElementException("No handler for tool " + Py.strRepr(call.name())
                        + "; give the tool a handler or pass one with function(name, handler).");
            }
        }
        for (Call call : calls) {
            ToolHandler handler = registry.get(call.name());
            try {
                outputs.add(handler.call(call.argumentsObject()));
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new ToolExecutionException(call.name(), e);
            }
        }
        return !calls.isEmpty();
    }

    static int repairs(int value) {
        if (value < 0) {
            throw new IllegalArgumentException("maxRepairs must be a nonnegative integer.");
        }
        return value;
    }

    /** Builds a {@link DMCall}. Every option defaults to the paper's setting. */
    public static final class Builder {
        private final Generator generatorInstance;
        private final String spec;
        private final List<Object> toolItems = new ArrayList<>();
        private final Map<String, ToolHandler> functionMap = new LinkedHashMap<>();
        private DecisionModel decisionModel;
        private Double threshold;
        private String jevApiKey;
        private int maxRepairs = DEFAULT_MAX_REPAIRS;
        private Checks checks = Checks.BOTH;
        private History history = History.CUMULATIVE;
        private FeedbackMode feedback = FeedbackMode.HYBRID;
        private OnExhaustion onExhaustion = OnExhaustion.FIRST;
        private boolean preserveInitialNoCall = true;
        private boolean executeUnverified = false;

        private List<Tool> tools;
        private Map<String, ToolHandler> functions;

        private Builder(Generator generator, String spec) {
            this.generatorInstance = generator;
            this.spec = spec;
        }

        /** Adds tools: {@link Tool}s, or definitions as maps or JSON. */
        public Builder tools(Object... tools) {
            for (Object tool : tools) {
                toolItems.addAll(Tools.items(tool));
            }
            return this;
        }

        /** Adds tools from a list. */
        public Builder tools(List<?> tools) {
            toolItems.addAll(Tools.items(tools));
            return this;
        }

        /** Adds one tool. */
        public Builder tool(Object tool) {
            toolItems.addAll(Tools.items(tool));
            return this;
        }

        /** A handler for the tool named {@code name}. */
        public Builder function(String name, ToolHandler handler) {
            functionMap.put(name, handler);
            return this;
        }

        /** Handlers by tool name. */
        public Builder functions(Map<String, ToolHandler> handlers) {
            functionMap.putAll(handlers);
            return this;
        }

        /** The decision model ({@link Jev} by default). Configure its key and thresholds on it. */
        public Builder decisionModel(DecisionModel decisionModel) {
            this.decisionModel = decisionModel;
            return this;
        }

        /** The default Jev's pass mark for both questions (0.5). */
        public Builder threshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        /** The default Jev's key ({@code TYPESAFE_API_KEY} otherwise). */
        public Builder jevApiKey(String jevApiKey) {
            this.jevApiKey = jevApiKey;
            return this;
        }

        /** The repair limit r (3). Up to r + 1 candidates are proposed. */
        public Builder maxRepairs(int maxRepairs) {
            this.maxRepairs = maxRepairs;
            return this;
        }

        public Builder checks(Checks checks) {
            this.checks = checks;
            return this;
        }

        public Builder checks(String checks) {
            return checks(Checks.fromValue(checks));
        }

        public Builder history(History history) {
            this.history = history;
            return this;
        }

        public Builder history(String history) {
            return history(History.fromValue(history));
        }

        public Builder feedback(FeedbackMode feedback) {
            this.feedback = feedback;
            return this;
        }

        public Builder feedback(String feedback) {
            return feedback(FeedbackMode.fromValue(feedback));
        }

        public Builder onExhaustion(OnExhaustion onExhaustion) {
            this.onExhaustion = onExhaustion;
            return this;
        }

        public Builder onExhaustion(String onExhaustion) {
            return onExhaustion(OnExhaustion.fromValue(onExhaustion));
        }

        /** Return a readable initial no call without verification (true, the paper's policy). */
        public Builder preserveInitialNoCall(boolean preserveInitialNoCall) {
            this.preserveInitialNoCall = preserveInitialNoCall;
            return this;
        }

        /** Let {@link #run} execute the final candidate even when none was accepted (false). */
        public Builder executeUnverified(boolean executeUnverified) {
            this.executeUnverified = executeUnverified;
            return this;
        }

        public DMCall build() {
            repairs(maxRepairs);
            if (checks == null || history == null || feedback == null || onExhaustion == null) {
                throw new IllegalArgumentException("checks, history, feedback and onExhaustion must not be null.");
            }
            tools = List.copyOf(Tools.asTools(toolItems));
            functions = new LinkedHashMap<>(functionMap);
            Tools.toolset(tools, functions);  // fail fast on a bad tool
            if (decisionModel != null && (jevApiKey != null || threshold != null)) {
                throw new IllegalArgumentException(
                        "Configure the decision model itself (api_key, threshold), not DMCall.");
            }
            List<AutoCloseable> owned = new ArrayList<>();
            Generator generator = generatorInstance;
            if (generator == null) {
                Provider provider = Providers.fromSpec(spec);
                owned.add(provider);
                generator = provider;
            }
            DecisionModel model = decisionModel;
            if (model == null) {
                try {
                    model = Jev.builder().apiKey(jevApiKey)
                            .threshold(threshold == null ? DEFAULT_THRESHOLD : threshold).build();
                } catch (RuntimeException e) {
                    for (AutoCloseable resource : owned) {
                        try {
                            resource.close();
                        } catch (Exception ignored) {
                            // Closing is best effort.
                        }
                    }
                    throw e;
                }
                owned.add(model);
            }
            return new DMCall(this, generator, model, owned);
        }
    }
}
